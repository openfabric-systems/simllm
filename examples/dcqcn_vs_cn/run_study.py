"""DCQCN vs rnic-cn study: run the pre-registered checks of expectations.md.

Four scenarios on the 64-node 2-tier Clos: incast (S-INC), cross-leaf
ECMP-collision permutation (S-ECMP), mixed lognormal all-to-all (S-A2A)
and the contention-free TP request (S-TP). rnic-nn-fluid gives the ideal
per-flow denominator; rnic-cn and fluid are deterministic; DCQCN runs
seeded ensembles at 1 MiB buffers with per-run mechanism counters (drops,
silent RTOs, ECN marks, PFC pauses) so every claim is checked against its
mechanism signature, not just the FCT ordering.

Usage:
    SIMLLM_HTSIM_RNIC=... SIMLLM_HTSIM_DCQCN=... SIMLLM_TXT2BIN=... \\
    python examples/dcqcn_vs_cn/run_study.py \\
        --out /data3/yifeng/simllm-dev/dcqcn-vs-cn-runs
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from simllm.backends import (
    HtsimDcqcnConfig,
    HtsimRnicConfig,
    RnicRunResult,
    normalized_fct,
    run_htsim_dcqcn,
    run_htsim_rnic,
)
from simllm.compute import (
    GPU_ENVELOPES,
    HostInitiationModel,
    ModelDims,
    RooflineProvider,
    estimate_step_latency_ps,
)
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.goal import GoalTrace, to_binary
from simllm.traffic import render_step_goal
from simllm.workload import LogNormalLengths

HERE = Path(__file__).resolve().parent
TOPO = HERE.parent / "m1" / "topologies" / "clos_64_400g.topo"
G400 = 400_000_000_000
SMALL_BUFFERS = {"-shared_buffer_bytes": "1048576", "-egress_buffer_bytes": "1048576"}
COUNTERS = ["ns_tm3_dropped_packets", "silent_rtos", "ecn_marked_packets",
            "loss_rate_cuts", "dcqcn_pfc_pause_frames", "dcqcn_pfc_max_cascade_depth"]
# Real manifest keys, read off a live rnic-cn manifest (an earlier draft
# used phantom names that matched nothing, caught in review; the merged
# cn_ladder study hit the same bug class).
CN_RECOVERY = ["rnic_cn_gap_nacks_dispatched", "rnic_cn_late_data_packets",
               "rnic_cn_deterministic_retransmissions",
               "rnic_cn_maximum_retry_attempt_observed"]
LAYERS = 32


def incast_goal(fan_in: int, size: int) -> GoalTrace:
    trace = GoalTrace(64)
    for s in range(1, fan_in + 1):
        trace.rank(s).send(size, to=0, tag=1)
        trace.rank(0).recv(size, source=s, tag=1)
    for r in set(range(64)) - set(range(fan_in + 1)):
        trace.rank(r).calc(0)
    return trace


def ecmp_perm_goal(size: int = 8 << 20) -> GoalTrace:
    """16 cross-leaf flows, two up and two down per leaf (see expectations)."""
    trace = GoalTrace(64)
    used = set()
    for i in range(8):
        j = (i + 1) % 8
        for k in range(2):
            src, dst = 8 * i + k, 8 * j + 2 + k
            trace.rank(src).send(size, to=dst, tag=10 + 2 * i + k)
            trace.rank(dst).recv(size, source=src, tag=10 + 2 * i + k)
            used |= {src, dst}
    for r in set(range(64)) - used:
        trace.rank(r).calc(0)
    return trace


def mixed_all_to_all_goal(seed: int = 7) -> GoalTrace:
    """The cn_ladder mixed a2a: 16 ranks, lognormal sizes, mean 256 KiB."""
    sizes = LogNormalLengths(mean=256 << 10, sigma=1.5, seed=seed, minimum=4096)
    trace = GoalTrace(64)
    ranks = list(range(16))
    draw = iter(sizes.sample(len(ranks) * (len(ranks) - 1)))
    for s in ranks:
        for d in ranks:
            if s == d:
                continue
            size = int(next(draw))
            trace.rank(s).send(size, to=d, tag=100 + s)
            trace.rank(d).recv(size, source=s, tag=100 + s)
    for r in range(16, 64):
        trace.rank(r).calc(0)
    return trace


def tp_step_goals() -> list[GoalTrace]:
    """The breakdown request at TP=8 spread one rank per leaf."""
    dims = ModelDims(num_layers=LAYERS, hidden_size=4096, intermediate_size=14336 // 8,
                     num_heads=4, num_kv_heads=1, head_size=128, vocab_size=128256,
                     dtype_bytes=2)
    tp_ranks = [8 * i + 7 for i in range(8)]
    provider, gpu, host = RooflineProvider(efficiency=0.7), GPU_ENVELOPES["b100"], (
        HostInitiationModel())
    records = [StepRecord(0, 0, [ScheduledRequest(
        "r", RequestPhase.PREFILL, 2048, context_length=2048)])]
    records += [StepRecord(1 + i, 0, [ScheduledRequest(
        "r", RequestPhase.DECODE, 1, context_length=2049 + i)]) for i in range(7)]
    traces = []
    for record in records:
        estimate = estimate_step_latency_ps(dims, record, 1, provider, gpu, host)
        traces.append(render_step_goal(
            record, dims, tp_ranks, estimate // (LAYERS * 1000), num_goal_ranks=64))
    return traces


def manifest_counters(result: RnicRunResult, keys: list[str]) -> dict[str, int]:
    out = dict.fromkeys(keys, 0)
    for line in result.manifest:
        for token in line.split():
            for key in keys:
                if token.startswith(key + "="):
                    try:
                        out[key] = int(token.split("=", 1)[1])
                    except ValueError:
                        pass
    return out


#: every run's integer manifest counters, written to counters.csv
COUNTER_ROWS: list[dict] = []


def _log_counters(run_name: str, result: RnicRunResult) -> None:
    for line in result.manifest:
        for token in line.split():
            if "=" in token:
                key, _, value = token.partition("=")
                try:
                    COUNTER_ROWS.append(
                        {"run": run_name, "counter": key, "value": int(value)})
                except ValueError:
                    pass


def run_profile(goal_bin: Path, profile: str, csv_path: Path) -> RnicRunResult:
    result = run_htsim_rnic(HtsimRnicConfig(
        goal_bin=goal_bin, profile=profile, linkspeed_bps=G400,
        completion_csv=csv_path,
        topology=TOPO if profile == "rnic-cn" else None))
    _log_counters(csv_path.name, result)
    return result


def run_dcqcn(goal_bin: Path, mode: str, seed: int, csv_path: Path) -> RnicRunResult:
    flags = dict(SMALL_BUFFERS)
    flags["-pfc"] = "on" if mode == "ecn-pfc" else "off"
    flags["-seed"] = str(seed)
    flags["-ecn_seed"] = str(seed)
    result = run_htsim_dcqcn(HtsimDcqcnConfig(
        goal_bin=goal_bin, topology=TOPO, link_bps=G400,
        completion_csv=csv_path, extra_flags=flags))
    _log_counters(csv_path.name, result)
    return result


def nfct_stats(result: RnicRunResult, ideal: RnicRunResult) -> dict[str, float]:
    values = sorted(n.slowdown for n in normalized_fct(result.flows, ideal.flows))
    return {
        "median": statistics.median(values),
        "p99": values[min(len(values) - 1, int(0.99 * (len(values) - 1)))],
        "max": values[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/dcqcn_vs_cn")
    parser.add_argument("--a2a-seeds", type=int, default=8)
    parser.add_argument("--inc-seeds", type=int, default=5)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []
    dist_rows: list[dict] = []

    def emit(**kw) -> None:
        checks.append(kw)
        print(" ".join(f"{k}={v}" for k, v in kw.items()))

    def record_dist(scenario: str, run: str, result: RnicRunResult,
                    ideal: RnicRunResult) -> None:
        for n in normalized_fct(result.flows, ideal.flows):
            dist_rows.append({"scenario": scenario, "run": run,
                              "ratio": round(n.slowdown, 6)})

    # S-INC ---------------------------------------------------------------
    for fan_in in (8, 32):
        for size in (64 << 10, 2 << 20):
            name = f"inc{fan_in}-s{size}"
            goal_bin = to_binary(incast_goal(fan_in, size).write(out / f"{name}.goal"))
            ideal = run_profile(goal_bin, "rnic-nn-fluid", out / f"{name}.fluid.csv")
            cn = run_profile(goal_bin, "rnic-cn", out / f"{name}.cn.csv")
            cn_stats = nfct_stats(cn, ideal)
            cn_recovery = sum(manifest_counters(cn, CN_RECOVERY).values())
            record_dist(name, "rnic-cn", cn, ideal)
            if size == 2 << 20:
                nn = run_profile(goal_bin, "rnic-nn", out / f"{name}.nn.csv")
                nn_stats = nfct_stats(nn, ideal)
                record_dist(name, "rnic-nn", nn, ideal)
                emit(check="R5b", cell=name,
                     cn_p99=round(cn_stats["p99"], 3),
                     nn_p99=round(nn_stats["p99"], 3),
                     ratio=round(cn_stats["p99"] / nn_stats["p99"], 3),
                     ok=cn_stats["p99"] <= 1.25 * nn_stats["p99"])
            for mode in ("ecn-only", "ecn-pfc"):
                per_seed = []
                counters_all = []
                for seed in range(1, args.inc_seeds + 1):
                    r = run_dcqcn(goal_bin, mode, seed,
                                  out / f"{name}.dcqcn-{mode}-s{seed}.csv")
                    per_seed.append(nfct_stats(r, ideal))
                    counters_all.append(manifest_counters(r, COUNTERS))
                    record_dist(name, f"dcqcn-{mode}-s{seed}", r, ideal)
                dq_p99 = max(s["p99"] for s in per_seed)
                dq_max = max(s["max"] for s in per_seed)
                emit(check="R1a", cell=name, mode=mode,
                     cn_p99=round(cn_stats["p99"], 3), dq_p99=round(dq_p99, 3),
                     cn_max=round(cn_stats["max"], 3), dq_max=round(dq_max, 3),
                     dq_median=round(statistics.median(
                         s["median"] for s in per_seed), 3),
                     cn_median=round(cn_stats["median"], 3),
                     ok=(dq_p99 > cn_stats["p99"] and dq_max > cn_stats["max"]))
                if fan_in == 32 and size == 2 << 20:
                    if mode == "ecn-only":
                        emit(check="R1b", cell=name,
                             drops_per_seed=[c["ns_tm3_dropped_packets"]
                                             for c in counters_all],
                             cn_recovery_events=cn_recovery,
                             ok=(all(c["ns_tm3_dropped_packets"] > 0
                                     for c in counters_all)
                                 and cn_recovery == 0))
                    else:
                        emit(check="R1c", cell=name,
                             pauses_per_seed=[c["dcqcn_pfc_pause_frames"]
                                              for c in counters_all],
                             ok=all(c["dcqcn_pfc_pause_frames"] > 0
                                    for c in counters_all))

    # S-ECMP --------------------------------------------------------------
    goal_bin = to_binary(ecmp_perm_goal().write(out / "ecmp16.goal"))
    ideal = run_profile(goal_bin, "rnic-nn-fluid", out / "ecmp16.fluid.csv")
    cn = run_profile(goal_bin, "rnic-cn", out / "ecmp16.cn.csv")
    cn2 = run_profile(goal_bin, "rnic-cn", out / "ecmp16.cn2.csv")
    cn_stats = nfct_stats(cn, ideal)
    record_dist("ecmp16", "rnic-cn", cn, ideal)
    nn = run_profile(goal_bin, "rnic-nn", out / "ecmp16.nn.csv")
    record_dist("ecmp16", "rnic-nn", nn, ideal)
    emit(check="R5a", cn_jct_ps=cn.job_completion_time_ps(),
         nn_jct_ps=nn.job_completion_time_ps(),
         ratio=round(cn.job_completion_time_ps() / nn.job_completion_time_ps(), 3),
         ok=cn.job_completion_time_ps() <= 1.25 * nn.job_completion_time_ps())
    emit(check="R2a", cn_max=round(cn_stats["max"], 3),
         deterministic=[(f.fct_ps) for f in cn.flows] == [(f.fct_ps) for f in cn2.flows],
         ok=(cn_stats["max"] < 1.5
             and [f.fct_ps for f in cn.flows] == [f.fct_ps for f in cn2.flows]))
    seed_maxes = []
    for seed in range(1, 9):
        r = run_dcqcn(goal_bin, "ecn-pfc", seed, out / f"ecmp16.dcqcn-s{seed}.csv")
        seed_maxes.append(nfct_stats(r, ideal)["max"])
        record_dist("ecmp16", f"dcqcn-s{seed}", r, ideal)
    collisions = sum(1 for m in seed_maxes if m >= 1.7)
    emit(check="R2b", seed_maxes=[round(m, 3) for m in seed_maxes],
         colliding_seeds=collisions,
         ok=(collisions >= 5 and max(seed_maxes) > cn_stats["max"]))

    # S-A2A ---------------------------------------------------------------
    goal_bin = to_binary(mixed_all_to_all_goal().write(out / "a2a16.goal"))
    ideal = run_profile(goal_bin, "rnic-nn-fluid", out / "a2a16.fluid.csv")
    cn = run_profile(goal_bin, "rnic-cn", out / "a2a16.cn.csv")
    cn_stats = nfct_stats(cn, ideal)
    cn_recovery = sum(manifest_counters(cn, CN_RECOVERY).values())
    record_dist("a2a16", "rnic-cn", cn, ideal)
    nn = run_profile(goal_bin, "rnic-nn", out / "a2a16.nn.csv")
    nn_stats = nfct_stats(nn, ideal)
    record_dist("a2a16", "rnic-nn", nn, ideal)
    emit(check="R5c", cn_p99=round(cn_stats["p99"], 3),
         nn_p99=round(nn_stats["p99"], 3),
         ratio=round(cn_stats["p99"] / nn_stats["p99"], 3))
    for mode in ("ecn-only", "ecn-pfc"):
        per_seed = []
        counters_all = []
        for seed in range(1, args.a2a_seeds + 1):
            r = run_dcqcn(goal_bin, mode, seed, out / f"a2a16.dcqcn-{mode}-s{seed}.csv")
            per_seed.append(nfct_stats(r, ideal))
            counters_all.append(manifest_counters(r, COUNTERS))
            record_dist("a2a16", f"dcqcn-{mode}-s{seed}", r, ideal)
        dq_p99 = max(s["p99"] for s in per_seed)
        dq_max = max(s["max"] for s in per_seed)
        emit(check="R3a", mode=mode, cn_p99=round(cn_stats["p99"], 2),
             dq_p99=round(dq_p99, 2), cn_max=round(cn_stats["max"], 2),
             dq_max=round(dq_max, 2),
             ok=(dq_p99 > 10 * cn_stats["p99"] and dq_max > 20 * cn_stats["max"]))
        if mode == "ecn-only":
            emit(check="R3b",
                 recovery_per_seed=[c["ns_tm3_dropped_packets"] + c["silent_rtos"]
                                    for c in counters_all],
                 cn_recovery_events=cn_recovery,
                 ok=(all(c["ns_tm3_dropped_packets"] + c["silent_rtos"] > 0
                         for c in counters_all) and cn_recovery == 0))
        emit(check="R3c", mode=mode,
             cn_median=round(cn_stats["median"], 3),
             dq_median=round(statistics.median(s["median"] for s in per_seed), 3))

    # S-TP ----------------------------------------------------------------
    totals = {"fluid": 0, "cn": 0, "cn2": 0}
    dq_totals = {seed: 0 for seed in range(1, 4)}
    for index, trace in enumerate(tp_step_goals()):
        goal_bin = to_binary(trace.write(out / f"tp-step{index}.goal"))
        totals["fluid"] += run_profile(
            goal_bin, "rnic-nn-fluid",
            out / f"tp-step{index}.fluid.csv").job_completion_time_ps()
        totals["cn"] += run_profile(
            goal_bin, "rnic-cn", out / f"tp-step{index}.cn.csv").job_completion_time_ps()
        totals["cn2"] += run_profile(
            goal_bin, "rnic-cn",
            out / f"tp-step{index}.cn2.csv").job_completion_time_ps()
        for seed in dq_totals:
            dq_totals[seed] += run_dcqcn(
                goal_bin, "ecn-pfc", seed,
                out / f"tp-step{index}.dcqcn-s{seed}.csv").job_completion_time_ps()
    dq_values = list(dq_totals.values())
    spread = (max(dq_values) - min(dq_values)) / statistics.median(dq_values)
    emit(check="R4a", fluid_ps=totals["fluid"], cn_ps=totals["cn"],
         dcqcn_ps=dq_values, dcqcn_spread=round(spread, 5),
         ok=(max(dq_values) < totals["cn"] and spread < 0.01
             and min(dq_values) >= totals["fluid"] and totals["cn"] >= totals["fluid"]))
    emit(check="R4b", ok=totals["cn"] == totals["cn2"])

    with open(out / "summary.csv", "w", newline="") as handle:
        fields: list[str] = []
        for row in checks:
            fields += [k for k in row if k not in fields]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(checks)
    with open(out / "distributions.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario", "run", "ratio"])
        writer.writeheader()
        writer.writerows(dist_rows)
    with open(out / "tp_totals.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "total_ps"])
        writer.writerow(["fluid", totals["fluid"]])
        writer.writerow(["rnic-cn", totals["cn"]])
        for seed, value in dq_totals.items():
            writer.writerow([f"dcqcn-s{seed}", value])
    with open(out / "counters.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "counter", "value"])
        writer.writeheader()
        writer.writerows(COUNTER_ROWS)
    print(f"\n{len(checks)} check rows -> {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
