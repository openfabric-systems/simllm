"""M1 sanity studies: probes, scatter/gather validation, PP-decode signature.

Runs the workloads pre-registered in expectations.md against the wired
htsim_rnic profiles and writes one CSV row per (run, metric) to
<out>/summary.csv. Raw GOALs and completion CSVs land under <out> as well.

Usage: python run_m1.py [--out runs/m1] [--sections probes,a,b]
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

from simllm.backends import (
    HtsimRnicConfig,
    RnicRunResult,
    normalized_fct,
    run_htsim_rnic,
)
from simllm.compute import GpuSpec, KernelSpec, RooflineProvider
from simllm.goal import GoalTrace, to_binary
from simllm.traffic import gather, scatter

HERE = Path(__file__).resolve().parent
CLOS_64 = HERE / "topologies" / "clos_64_400g.topo"

G = 1_000_000_000
MIB = 1 << 20

# Declared B100 envelope for the reference config (envelope, not vendor truth)
B100 = GpuSpec(name="b100-bf16", peak_flops=1.75e15, mem_bandwidth=8.0e12)
ROOFLINE = RooflineProvider(efficiency=0.7)

# 70B-class model on PP=8 x TP=8, hidden 8192, 1024 prompt tokens
PARAMS = 70e9
HIDDEN, TOKENS, PP, TP = 8192, 1024, 8, 8
S_PRE = TOKENS * HIDDEN * 2
S_DEC = HIDDEN * 2
S_TOK = 64
DECODE_STEPS = 8


def idle_fill(trace: GoalTrace, used: set[int]) -> None:
    """Give unused ranks one zero-cost op so every rank block is populated."""
    for r in range(trace.num_ranks):
        if r not in used:
            trace.rank(r).calc(0)


def echo_goal(calc_cost: int, size: int = MIB) -> GoalTrace:
    trace = GoalTrace(2)
    tx = trace.rank(0).send(size, to=1, tag=1)
    rx = trace.rank(1).recv(size, source=0, tag=1)
    c = trace.rank(1).calc(calc_cost)
    trace.rank(1).requires(c, rx)
    back = trace.rank(1).send(size, to=0, tag=2)
    trace.rank(1).requires(back, c)
    trace.rank(0).recv(size, source=1, tag=2)
    _ = tx
    return trace


def single_flow_goal(size: int) -> GoalTrace:
    trace = GoalTrace(2)
    trace.rank(0).send(size, to=1, tag=1)
    trace.rank(1).recv(size, source=0, tag=1)
    return trace


def scatter_gather_goal(num_ranks: int, workers: int, size: int, calc_cost: int) -> GoalTrace:
    trace = GoalTrace(num_ranks)
    ws = list(range(1, workers + 1))
    sdone = scatter(trace, root=0, workers=ws, size_bytes=size, tag=1)
    cdone = {}
    for w in ws:
        c = trace.rank(w).calc(calc_cost)
        trace.rank(w).requires(c, sdone[w])
        cdone[w] = c
    gather(trace, root=0, workers=ws, size_bytes=size, tag=2, after=cdone)
    idle_fill(trace, {0, *ws})
    return trace


def pp_decode_goal(calc_pre: int, calc_dec: int) -> GoalTrace:
    """PP=8 chain on NIC rank 8*i of each node; tags encode phase and step."""
    trace = GoalTrace(64)
    stages = [8 * i for i in range(PP)]
    prev_label = {}
    # prefill: activation chain with per-stage compute
    for i, r in enumerate(stages):
        deps = []
        if i > 0:
            rx = trace.rank(r).recv(S_PRE, source=stages[i - 1], tag=100 + i)
            deps.append(rx)
        c = trace.rank(r).calc(calc_pre)
        for d in deps:
            trace.rank(r).requires(c, d)
        prev_label[r] = c
        if i < PP - 1:
            tx = trace.rank(r).send(S_PRE, to=stages[i + 1], tag=100 + i + 1)
            trace.rank(r).requires(tx, c)
    # decode: token d loops last stage -> first stage, then activations forward
    last = stages[-1]
    for d in range(DECODE_STEPS):
        tok = trace.rank(last).send(S_TOK, to=stages[0], tag=300 + d)
        trace.rank(last).requires(tok, prev_label[last])
        rx0 = trace.rank(stages[0]).recv(S_TOK, source=last, tag=300 + d)
        carry = rx0
        for i, r in enumerate(stages):
            c = trace.rank(r).calc(calc_dec)
            trace.rank(r).requires(c, carry if r == stages[0] else prev_label[r])
            prev_label[r] = c
            if i < PP - 1:
                tag = 1000 + 16 * d + i
                tx = trace.rank(r).send(S_DEC, to=stages[i + 1], tag=tag)
                trace.rank(r).requires(tx, c)
                nxt = trace.rank(stages[i + 1]).recv(S_DEC, source=r, tag=tag)
                prev_label[stages[i + 1]] = nxt  # compute waits on activation
    idle_fill(trace, set(stages))
    return trace


def run(goal: GoalTrace, name: str, profile: str, linkspeed_bps: int, out: Path,
        topo: Path | None = None, extra: dict[str, str] | None = None) -> RnicRunResult:
    goal_bin = to_binary(goal.write(out / f"{name}.goal"))
    return run_htsim_rnic(HtsimRnicConfig(
        goal_bin=goal_bin, profile=profile, linkspeed_bps=linkspeed_bps,
        completion_csv=out / f"{name}.{profile}.csv", topology=topo,
        extra_flags=extra or {},
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/m1")
    parser.add_argument("--sections", default="probes,a,b")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sections = set(args.sections.split(","))
    rows: list[dict] = []

    def emit(**kw) -> None:
        rows.append(kw)
        print(" ".join(f"{k}={v}" for k, v in kw.items()))

    if "probes" in sections:
        for profile in ("rnic-nn-fluid", "rnic-nn"):
            for cost in (0, 1_000_000):
                r = run(echo_goal(cost), f"p1-c{cost}", profile, 400 * G, out)
                emit(section="P1", profile=profile, calc=cost,
                     jct_ps=r.job_completion_time_ps())
            for size in (MIB, 4 * MIB):
                r = run(single_flow_goal(size), f"p2-s{size}", profile, 400 * G, out)
                emit(section="P2", profile=profile, size=size, fct_ps=r.flows[0].fct_ps)

    if "a" in sections:
        for profile in ("rnic-nn-fluid", "rnic-nn"):
            for bw in (100, 200, 400):
                r = run(scatter_gather_goal(32, 4, MIB, 0), f"a-b{bw}-w4", profile,
                        bw * G, out)
                emit(section="A-B", profile=profile, bw_g=bw, workers=4,
                     jct_ps=r.job_completion_time_ps())
            for w in (2, 4, 8):
                r = run(scatter_gather_goal(32, w, MIB, 0), f"a-w{w}", profile,
                        400 * G, out)
                emit(section="A-W", profile=profile, bw_g=400, workers=w,
                     jct_ps=r.job_completion_time_ps())
        # rnic-cn on the generated 32-node ns-tm3 Clos, normalized against nn
        for w in (2, 4, 8):
            cn = run(scatter_gather_goal(32, w, MIB, 0), f"a-w{w}", "rnic-cn",
                     400 * G, out)
            nn = run(scatter_gather_goal(32, w, MIB, 0), f"a-w{w}", "rnic-nn",
                     400 * G, out)
            emit(section="A-CN", profile="rnic-cn", bw_g=400, workers=w,
                 jct_ps=cn.job_completion_time_ps())
            for n in normalized_fct(cn.flows, nn.flows):
                emit(section="A-CN-NFCT", workers=w, tag=n.tag, source=n.source,
                     destination=n.destination, slowdown=round(n.slowdown, 6))

    if "b" in sections:
        pre = ROOFLINE.estimate(KernelSpec(
            "pp_stage_prefill", flops=2 * PARAMS / PP * TOKENS / TP,
            bytes_moved=PARAMS * 2 / (PP * TP)), B100)
        dec = ROOFLINE.estimate(KernelSpec(
            "pp_stage_decode", flops=2 * PARAMS / PP / TP,
            bytes_moved=PARAMS * 2 / (PP * TP)), B100)
        # calc unit resolved by P1; workload B uses ns per P1 (see RESULTS.md)
        calc_pre, calc_dec = pre.duration_ps // 1000, dec.duration_ps // 1000
        emit(section="B-calc", calc_pre_ps=pre.duration_ps, calc_dec_ps=dec.duration_ps,
             prefill_bound=pre.bound, decode_bound=dec.bound)
        results_400 = {}
        for profile, topo in (("rnic-nn-fluid", None), ("rnic-nn", None),
                              ("rnic-cn", CLOS_64)):
            for bw in (400, 800):
                if topo is not None and bw != 400:
                    continue  # the topo file fixes 400G links
                r = run(pp_decode_goal(calc_pre, calc_dec), f"b-{bw}", profile,
                        bw * G, out, topo=topo)
                if bw == 400:
                    results_400[profile] = r
                tokens = sorted(
                    (f.tag - 300, f.completion_time_ps)
                    for f in r.flows if 300 <= f.tag < 300 + DECODE_STEPS
                )
                ttft = tokens[0][1]
                deltas = [b - a for (_, a), (_, b) in itertools.pairwise(tokens)]
                emit(section="B", profile=profile, bw_g=bw, ttft_ps=ttft,
                     tpot_ps=round(sum(deltas) / len(deltas)),
                     jct_ps=r.job_completion_time_ps())
        # chained flows are aligned-start, so the per-flow form is valid here
        norm = normalized_fct(results_400["rnic-cn"].flows,
                              results_400["rnic-nn"].flows)
        for cls, flows in (
            ("prefill", [n for n in norm if 100 <= n.tag < 200]),
            ("token", [n for n in norm if 300 <= n.tag < 400]),
            ("decode_act", [n for n in norm if n.tag >= 1000]),
        ):
            slow = sorted(n.slowdown for n in flows)
            emit(section="B-CN-NFCT", flow_class=cls, count=len(flows),
                 min_slowdown=round(slow[0], 4), max_slowdown=round(slow[-1], 4))

    with open(out / "summary.csv", "w", newline="") as handle:
        fields: list[str] = []
        for row in rows:
            fields += [k for k in row if k not in fields]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} rows -> {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
