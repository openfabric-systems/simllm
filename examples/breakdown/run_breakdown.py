"""Request-time breakdown study: run the pre-registered checks.

One canonical request (2048-token prefill, then 7 decode steps) through
`HtsimStepSink` per configuration, decomposing its total latency into
compute-bound kernel, memory-bound kernel and network components,
compared against the frozen tables of expectations.md. Emits
`<out>/summary.csv` (check rows) and `<out>/components.csv` (per-config
component sums feeding the plots).

Usage:
    SIMLLM_HTSIM_RNIC=... SIMLLM_TXT2BIN=... \\
    python examples/breakdown/run_breakdown.py \\
        --out /data3/yifeng/simllm-dev/breakdown-runs
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
from simllm.compute import (
    GPU_ENVELOPES,
    ModelDims,
    RooflineProvider,
    step_kernel,
)
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.placement import declared_manifest

G = 1_000_000_000
LAYERS, HIDDEN, DTYPE = 32, 4096, 2
CLOS_TOPO = Path(__file__).resolve().parents[1] / "m1" / "topologies" / "clos_64_400g.topo"

# Frozen check-F tables (expectations.md): (tp, link) -> components in ps.
FROZEN_F: dict[tuple[int, str], tuple[int, int, int, int]] = {
    (2, "400G"): (11_781_312_000, 10_205_472_000, 23_596_236_800, 45_583_020_800),
    (2, "100G"): (11_781_312_000, 10_205_472_000, 88_240_947_200, 110_227_731_200),
    (4, "400G"): (5_891_072_000, 5_759_360_000, 38_466_355_200, 50_116_787_200),
    (4, "100G"): (5_891_072_000, 5_759_360_000, 135_433_420_800, 147_083_852_800),
    (8, "400G"): (2_945_952_000, 3_536_288_000, 52_045_414_400, 58_527_654_400),
    (8, "100G"): (2_945_952_000, 3_536_288_000, 165_173_657_600, 171_655_897_600),
}

# Frozen check-N network point form and band (expectations.md): tp -> ps.
FROZEN_N: dict[int, tuple[int, int]] = {
    2: (24_018_124_800, 85_196_800),
    4: (39_228_702_720, 255_590_400),
    8: (53_237_022_720, 596_377_600),
}

LINK_BPS = {"400G": 400 * G, "100G": 100 * G}


def dims_tp(tp: int) -> ModelDims:
    return ModelDims(num_layers=LAYERS, hidden_size=HIDDEN,
                     intermediate_size=14336 // tp, num_heads=32 // tp,
                     num_kv_heads=max(8 // tp, 1), head_size=128,
                     vocab_size=128256, dtype_bytes=DTYPE)


def request_steps() -> list[StepRecord]:
    steps = [StepRecord(0, 0, [ScheduledRequest(
        "r", RequestPhase.PREFILL, 2048, context_length=2048)])]
    steps += [StepRecord(1 + i, 0, [ScheduledRequest(
        "r", RequestPhase.DECODE, 1, context_length=2049 + i)]) for i in range(7)]
    return steps


def run_config(profile: str, tp: int, link: str, workdir: Path,
               topology: Path | None = None) -> list[dict]:
    """Run the 8-step request; return per-step component rows."""
    if topology is None:
        tp_ranks = declared_manifest(tp=tp, pp=1, dp=1).group_ranks(0, "tp")
    else:
        # rnic-cn enforces that the resolved GOAL layout matches the
        # topology's node count (64 here). render_step_goal sizes the GOAL
        # as max(rank) + 1, so placing the TP group on the last node's
        # GPUs pads the GOAL to 64 ranks without changing the traffic
        # (the group still shares one first-tier switch, as on node 0).
        tp_ranks = list(range(64 - tp, 64))
    sink = HtsimStepSink(HtsimStepSinkConfig(
        profile=profile, tp_ranks=tp_ranks,
        dims=dims_tp(tp), workdir=workdir,
        linkspeed_bps=LINK_BPS[link], topology=topology,
    ))
    provider = RooflineProvider(efficiency=0.7)
    gpu = GPU_ENVELOPES["b100"]
    rows = []
    for record in request_steps():
        result = sink(record)
        assert result is not None
        outcome = sink.outcomes[-1]
        kernel_ps = LAYERS * max(outcome.per_layer_calc_ns, 1) * 1000
        bound = provider.estimate(step_kernel(sink.config.dims, record, 1), gpu).bound
        rows.append({
            "profile": profile, "tp": tp, "link": link,
            "step": record.step_index,
            "phase": record.scheduled[0].phase.value, "bound": bound,
            "kernel_ps": kernel_ps,
            "network_ps": outcome.makespan_ps - kernel_ps,
            "makespan_ps": outcome.makespan_ps,
        })
    return rows


def components(rows: list[dict]) -> tuple[int, int, int, int]:
    comp = sum(r["kernel_ps"] for r in rows if r["bound"] == "compute")
    mem = sum(r["kernel_ps"] for r in rows if r["bound"] == "memory")
    net = sum(r["network_ps"] for r in rows)
    return comp, mem, net, comp + mem + net


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/breakdown")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []
    all_steps: list[dict] = []
    per_config: dict[tuple[str, str, int], list[dict]] = {}

    def emit(**kw) -> None:
        checks.append(kw)
        print(" ".join(f"{k}={v}" for k, v in kw.items()))

    configs = [("rnic-nn-fluid", link, tp, None)
               for link in ("400G", "100G") for tp in (2, 4, 8)]
    configs += [("rnic-nn", "400G", tp, None) for tp in (2, 4, 8)]
    configs += [("rnic-cn", "400G", tp, CLOS_TOPO) for tp in (2, 4, 8)]

    for profile, link, tp, topo in configs:
        workdir = out / f"{profile}-{link}-tp{tp}"
        rows = run_config(profile, tp, link, workdir, topo)
        all_steps.extend(rows)
        per_config[(profile, link, tp)] = rows

    # Q1: bound structure (prefill compute-bound, decodes memory-bound).
    for tp in (2, 4, 8):
        rows = per_config[("rnic-nn-fluid", "400G", tp)]
        emit(check="Q1", tp=tp,
             ok=(rows[0]["bound"] == "compute"
                 and all(r["bound"] == "memory" for r in rows[1:])))

    # F: fluid components exact against the frozen table.
    for (tp, link), frozen in FROZEN_F.items():
        comp, mem, net, total = components(per_config[("rnic-nn-fluid", link, tp)])
        emit(check="F", tp=tp, link=link, compute_ps=comp, memory_ps=mem,
             network_ps=net, total_ps=total,
             residual_ps=(comp - frozen[0], mem - frozen[1],
                          net - frozen[2], total - frozen[3]),
             exact=(comp, mem, net, total) == frozen)

    # Q2: network share strictly increasing in TP per profile column.
    for profile, link in (("rnic-nn-fluid", "400G"), ("rnic-nn-fluid", "100G"),
                          ("rnic-nn", "400G"), ("rnic-cn", "400G")):
        shares = []
        for tp in (2, 4, 8):
            comp, mem, net, total = components(per_config[(profile, link, tp)])
            shares.append(net / total)
        emit(check="Q2", profile=profile, link=link,
             shares=tuple(round(s, 4) for s in shares),
             ok=shares[0] < shares[1] < shares[2])

    # Q3: link speed touches only the wire term.
    for tp in (2, 4, 8):
        fast = per_config[("rnic-nn-fluid", "400G", tp)]
        slow = per_config[("rnic-nn-fluid", "100G", tp)]
        kernels_equal = all(a["kernel_ps"] == b["kernel_ps"]
                            for a, b in zip(fast, slow))
        prefill_ratio = slow[0]["network_ps"] / fast[0]["network_ps"]
        decode_ratios = [b["network_ps"] / a["network_ps"]
                         for a, b in zip(fast[1:], slow[1:])]
        emit(check="Q3", tp=tp, kernels_equal=kernels_equal,
             prefill_ratio=round(prefill_ratio, 3),
             max_decode_ratio=round(max(decode_ratios), 3),
             ok=(kernels_equal and prefill_ratio > 2.0
                 and max(decode_ratios) < 1.1))

    # N: rnic-nn network inside the registered band, never below fluid.
    for tp, (point, band) in FROZEN_N.items():
        _, _, net_nn, _ = components(per_config[("rnic-nn", "400G", tp)])
        _, _, net_fluid, _ = components(per_config[("rnic-nn-fluid", "400G", tp)])
        emit(check="N", tp=tp, network_ps=net_nn, point_ps=point,
             deviation_ps=net_nn - point, band_ps=band,
             ok=(abs(net_nn - point) <= band and net_nn >= net_fluid))

    # C: cn on the Clos, directional.
    for tp in (2, 4, 8):
        cn = per_config[("rnic-cn", "400G", tp)]
        fluid = per_config[("rnic-nn-fluid", "400G", tp)]
        nn = per_config[("rnic-nn", "400G", tp)]
        cn_total = sum(r["makespan_ps"] for r in cn)
        fluid_total = sum(r["makespan_ps"] for r in fluid)
        nn_total = sum(r["makespan_ps"] for r in nn)
        prefill_infl = cn[0]["network_ps"] / fluid[0]["network_ps"]
        decode_infl = (sum(r["network_ps"] for r in cn[1:])
                       / sum(r["network_ps"] for r in fluid[1:]))
        flows = 8 * 2 * LAYERS * 2 * (tp - 1) * tp
        overhead_per_flow_ps = (cn_total - fluid_total) / flows
        emit(check="C", tp=tp, cn_total_ps=cn_total,
             ge_fluid=cn_total >= fluid_total,
             prefill_inflation=round(prefill_infl, 3),
             decode_inflation=round(decode_infl, 3),
             decode_gt_prefill=decode_infl > prefill_infl,
             cn_vs_nn_ratio=round(cn_total / nn_total, 3),
             overhead_per_flow_us=round(overhead_per_flow_ps / 1e6, 2))

    with open(out / "steps.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_steps[0]))
        writer.writeheader()
        writer.writerows(all_steps)
    with open(out / "components.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["profile", "link", "tp", "compute_ps", "memory_ps",
                         "network_ps", "total_ps"])
        for (profile, link, tp), rows in per_config.items():
            writer.writerow([profile, link, tp, *components(rows)])
    with open(out / "summary.csv", "w", newline="") as handle:
        fields: list[str] = []
        for row in checks:
            fields += [k for k in row if k not in fields]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(checks)
    print(f"\n{len(checks)} check rows -> {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
