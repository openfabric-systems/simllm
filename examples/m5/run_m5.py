"""M5 MoE slice: run the pre-registered checks of expectations.md.

Sections mirror the frozen registration: (a) standalone symmetric pairwise
all-to-allvs on the fluid profile, (b) the MoE step makespan grid on fluid
through HtsimStepSink with an EP group and TP world of 1, and (c) the
registered qualitative directions (a2av growth with EP width, fluid vs nn
ordering on the check-A grid, decode/prefill makespan monotonicity). One
CSV row per (run, metric) goes to <out>/summary.csv; raw GOALs and
completion CSVs land under <out> as well (keep <out> on a data volume,
configured by SIMLLM_DATA_ROOT).

Every expected value is a frozen constant from expectations.md, never a
runtime-derived number (the M4 audit rule); the runtime roofline estimate
and calc split are additionally checked against the frozen inputs per
cell.

Usage:
    SIMLLM_HTSIM_RNIC=... SIMLLM_TXT2BIN=... SIMLLM_DATA_ROOT=... \\
    python examples/m5/run_m5.py
"""

from __future__ import annotations

import argparse
import csv
from itertools import pairwise
from pathlib import Path

from simllm._local_config import path_from_env
from simllm.backends import (
    HtsimRnicConfig,
    HtsimStepSink,
    HtsimStepSinkConfig,
    run_htsim_rnic,
)
from simllm.compute import ModelDims
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.goal import GoalTrace, to_binary
from simllm.traffic import pairwise_all_to_allv

G = 1_000_000_000

# Declared granite-3.0-1b-a400m-like MoE geometry (frozen in expectations.md)
LAYERS = 24
HIDDEN = 1024
DTYPE_BYTES = 2
NUM_EXPERTS = 32
TOP_K = 8
MOE_INTERMEDIATE = 512

# Frozen check-A predictions (expectations.md A1-A6): standalone symmetric
# pairwise all-to-allv JCT on rnic-nn-fluid, keyed (per-pair bytes, W).
FROZEN_A_JCT_PS: dict[tuple[int, int], int] = {
    (65_536, 2): 3_310_720,
    (65_536, 4): 5_932_161,
    (65_536, 8): 11_175_041,
    (16_777_216, 2): 337_544_320,
    (16_777_216, 4): 1_008_632_961,
    (16_777_216, 8): 2_350_810_241,
}

# Frozen check-B roofline inputs (expectations.md B1-B6): whole-step
# estimate E in ps and per-layer calc c in ns, keyed (shape, W).
FROZEN_B_INPUTS: dict[tuple[str, int], tuple[int, int]] = {
    ("decode8x2048", 2): (404_450_742, 16_852),
    ("decode8x2048", 4): (296_597_211, 12_358),
    ("decode8x2048", 8): (242_670_445, 10_111),
    ("prefill2048", 2): (1_390_831_206, 57_951),
    ("prefill2048", 4): (1_390_831_206, 57_951),
    ("prefill2048", 8): (1_390_831_206, 57_951),
}

# Frozen check-B makespans (expectations.md B1-B6), keyed (shape, W).
FROZEN_B_MAKESPAN_PS: dict[tuple[str, int], int] = {
    ("decode8x2048", 2): 563_362_560,
    ("decode8x2048", 4): 486_963_888,
    ("decode8x2048", 8): 448_764_528,
    ("prefill2048", 2): 17_592_951_360,
    ("prefill2048", 4): 25_646_015_088,
    ("prefill2048", 8): 29_672_546_928,
}


def moe_dims(world: int) -> ModelDims:
    return ModelDims(
        num_layers=LAYERS,
        hidden_size=HIDDEN,
        intermediate_size=MOE_INTERMEDIATE,
        num_heads=16,
        num_kv_heads=8,
        head_size=64,
        vocab_size=49152,
        dtype_bytes=DTYPE_BYTES,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        moe_intermediate_size=MOE_INTERMEDIATE,
        local_num_experts=NUM_EXPERTS // world,
    )


def decode_record() -> StepRecord:
    """decode8x2048: 8 requests, 1 new token each, context 2048."""
    return StepRecord(step_index=0, virtual_time_ps=0, scheduled=[
        ScheduledRequest(f"d{i}", RequestPhase.DECODE, num_new_tokens=1,
                         context_length=2048)
        for i in range(8)
    ])


def prefill_record() -> StepRecord:
    """prefill2048: one 2048-token prefill chunk at context 2048."""
    return StepRecord(step_index=0, virtual_time_ps=0, scheduled=[
        ScheduledRequest("p0", RequestPhase.PREFILL, num_new_tokens=2048,
                         context_length=2048)
    ])


STEP_SHAPES = {"decode8x2048": decode_record, "prefill2048": prefill_record}


def run_a2av(out: Path, profile: str, per_pair: int, world: int) -> int:
    """One standalone symmetric pairwise all-to-allv; returns the JCT."""
    trace = GoalTrace(world)
    ranks = list(range(world))
    send_bytes = {(s, d): per_pair for s in ranks for d in ranks if s != d}
    pairwise_all_to_allv(trace, ranks=ranks, send_bytes=send_bytes, tag=1)
    name = f"a2av-s{per_pair}-w{world}-{profile}"
    goal_bin = to_binary(trace.write(out / f"{name}.goal"))
    result = run_htsim_rnic(HtsimRnicConfig(
        goal_bin=goal_bin, profile=profile, linkspeed_bps=400 * G,
        completion_csv=out / f"{name}.csv",
    ))
    assert len(result.flows) == world * (world - 1)
    return result.job_completion_time_ps()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sections", default="a,b,c")
    args = parser.parse_args()
    if args.out is None:
        data_root = path_from_env("SIMLLM_DATA_ROOT")
        if data_root is None:
            parser.error("--out is required when SIMLLM_DATA_ROOT is not set")
        args.out = data_root / "m5"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    sections = set(args.sections.split(","))
    rows: list[dict] = []

    def emit(**kw) -> None:
        rows.append(kw)
        print(" ".join(f"{k}={v}" for k, v in kw.items()))

    fluid_a: dict[tuple[int, int], int] = {}
    if "a" in sections:
        for per_pair in (65_536, 16_777_216):
            for world in (2, 4, 8):
                measured = run_a2av(out, "rnic-nn-fluid", per_pair, world)
                fluid_a[(per_pair, world)] = measured
                expected = FROZEN_A_JCT_PS[(per_pair, world)]
                emit(section="A", per_pair=per_pair, world=world,
                     jct_ps=measured, expected_ps=expected,
                     residual_ps=measured - expected,
                     match=(measured == expected))

    b_makespans: dict[tuple[str, int], int] = {}
    if "b" in sections:
        for shape, build in STEP_SHAPES.items():
            for world in (2, 4, 8):
                workdir = out / f"moe-{shape}-ep{world}-rnic-nn-fluid"
                sink = HtsimStepSink(HtsimStepSinkConfig(
                    profile="rnic-nn-fluid",
                    tp_ranks=(0,),
                    dims=moe_dims(world),
                    workdir=workdir,
                    ep_ranks=tuple(range(world)),
                    linkspeed_bps=400 * G,
                ))
                result = sink(build())
                assert result is not None
                outcome = sink.outcomes[0]
                measured = result.step_latency_ps
                b_makespans[(shape, world)] = measured
                frozen_e, frozen_c = FROZEN_B_INPUTS[(shape, world)]
                expected = FROZEN_B_MAKESPAN_PS[(shape, world)]
                emit(section="B", shape=shape, ep=world,
                     estimate_ps=outcome.compute_estimate_ps,
                     estimate_matches_frozen=(
                         outcome.compute_estimate_ps == frozen_e),
                     calc_ns=outcome.per_layer_calc_ns,
                     calc_matches_frozen=(
                         outcome.per_layer_calc_ns == frozen_c),
                     makespan_ps=measured, expected_ps=expected,
                     residual_ps=measured - expected,
                     match=(measured == expected))

    if "c" in sections:
        # C-q1: network component of the check-B makespan grows with W
        # (network = measured makespan minus the FROZEN calc total)
        for shape in STEP_SHAPES:
            network = {
                world: b_makespans[(shape, world)]
                - LAYERS * FROZEN_B_INPUTS[(shape, world)][1] * 1000
                for world in (2, 4, 8)
                if (shape, world) in b_makespans
            }
            values = [network[w] for w in sorted(network)]
            emit(section="C-q1", shape=shape,
                 network_ps_by_ep=";".join(str(v) for v in values),
                 strictly_increasing=(values == sorted(values)
                                      and len(set(values)) == len(values)))
        # C-q2: the check-A grid on rnic-nn must dominate fluid per cell
        for per_pair in (65_536, 16_777_216):
            for world in (2, 4, 8):
                measured_nn = run_a2av(out, "rnic-nn", per_pair, world)
                fluid = fluid_a.get((per_pair, world))
                emit(section="C-q2", per_pair=per_pair, world=world,
                     jct_nn_ps=measured_nn, jct_fluid_ps=fluid,
                     nn_ge_fluid=(fluid is None or measured_nn >= fluid))
        # C-q3: decode makespan decreases with W, prefill increases
        for shape, direction in (("decode8x2048", "decreasing"),
                                 ("prefill2048", "increasing")):
            values = [b_makespans[(shape, w)] for w in (2, 4, 8)
                      if (shape, w) in b_makespans]
            monotone = all(
                (a > b) if direction == "decreasing" else (a < b)
                for a, b in pairwise(values)
            )
            emit(section="C-q3", shape=shape, direction=direction,
                 makespans_ps=";".join(str(v) for v in values),
                 holds=monotone)

    with open(out / "summary.csv", "w", newline="") as handle:
        fields: list[str] = []
        for row in rows:
            fields += [k for k in row if k not in fields]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} rows -> {out / 'summary.csv'}")
    # Operational FAIL bar (audit fix): a mismatch exits nonzero, and a
    # partial --sections run must not report vacuously-true directions
    # computed over empty inputs.
    verdict_keys = ("match", "strictly_increasing", "nn_ge_fluid", "holds")
    failed = [row for row in rows
              if any(row.get(key) is False for key in verdict_keys)]
    if sections != {"a", "b", "c"} :
        print("partial --sections run: direction checks may lack inputs, "
              "no verdict is claimed")
    if failed:
        for row in failed:
            print("FAILED:", row)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
