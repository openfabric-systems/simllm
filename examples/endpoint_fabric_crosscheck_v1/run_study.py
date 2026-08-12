"""Cross-check the analytic endpoint charge against the fluid fabric (CORE-43).

This module currently carries the frozen registry and the check-only dry run.
The observation, comparison and reporting code lands in a later commit, after
this freeze.
"""

from __future__ import annotations

import argparse
from pathlib import Path

#: recorded scheduler steps replayed from the capture tree
REPLAY_STEPS = 32
#: MoE layers, and one dispatch plus one combine phase per layer
MOE_LAYERS = 24
PHASES_PER_STEP = 2 * MOE_LAYERS
#: graph artifacts per step: one compute artifact per layer plus the phases
ARTIFACTS_PER_STEP = MOE_LAYERS + PHASES_PER_STEP
EP_WIDTH = 8
#: largest directed segment count any phase of this capture carries
MAX_SEGMENTS_PER_PHASE = EP_WIDTH - 1
#: fixed propagation the fluid manifold adds once after the last serviced bit
PROPAGATION_PS = 2_000_000
#: whole-nanosecond GOAL calc quantum of the analytic charge
ANALYTIC_QUANTUM_PS = 1_000
#: declared NVLink rate, used only for the deployment TTFT and TPOT arm
NVLINK_BYTES_PER_SECOND = 450_000_000_000

#: matched rate pairs: fluid bits/s, analytic bytes/s, picoseconds per byte
MATCHED_RATES = {
    400_000_000_000: (50_000_000_000, 20),
    200_000_000_000: (25_000_000_000, 40),
}

#: pre-freeze input characterization; see expectations.md
CAPTURE_LEDGER = {
    "prefill_step": 0,
    "prefill_peak_endpoint_bytes": 25_563_136,
    "prefill_peak_egress_bytes": 15_249_408,
    "prefill_segments": 336,
    "prefill_dispatch_bytes": 12_781_568,
    "prefill_combine_peak_egress_bytes": 2_467_840,
    "total_peak_endpoint_bytes": 54_218_752,
    "total_peak_egress_bytes": 32_567_296,
    "total_segments": 9_108,
}

#: frozen serialization and analytic-charge literals, picoseconds
FROZEN_SERIALIZATION_PS = {
    ("prefill", 400_000_000_000): 511_262_720,
    ("prefill", 200_000_000_000): 1_022_525_440,
    ("total", 400_000_000_000): 1_084_375_040,
    ("total", 200_000_000_000): 2_168_750_080,
}
FROZEN_ANALYTIC_PS = {
    ("prefill", 400_000_000_000): 511_290_000,
    ("prefill", 200_000_000_000): 1_022_550_000,
    ("total", 400_000_000_000): 1_084_962_000,
    ("total", 200_000_000_000): 2_169_586_000,
}

#: scored behavioral registry
SCORED_FAMILIES = {
    "CORE-F1": PHASES_PER_STEP * REPLAY_STEPS * len(MATCHED_RATES),
    "CORE-F2": REPLAY_STEPS * 2,
    "CORE-F3": REPLAY_STEPS * len(MATCHED_RATES),
}

#: recorded source artifacts under the capture tree
SOURCE_ARTIFACTS = {
    "steps": "replay-400g/steps.jsonl",
    "routing": "replay-400g/routed-experts.json",
}


def _analytic_charge_ps(peak_endpoint_bytes: int, ps_per_byte: int) -> int:
    """Whole-nanosecond analytic phase service for one peak endpoint load."""

    ideal_ps = ps_per_byte * peak_endpoint_bytes
    quanta = -(-ideal_ps // ANALYTIC_QUANTUM_PS)
    return quanta * ANALYTIC_QUANTUM_PS


def _check_frozen_registry() -> None:
    if PHASES_PER_STEP != 48 or ARTIFACTS_PER_STEP != 72:
        raise AssertionError("phase and artifact inventory drifted")
    if set(MATCHED_RATES) != {400_000_000_000, 200_000_000_000}:
        raise AssertionError("matched rate registry drifted")
    for linkspeed_bps, (analytic_bps, ps_per_byte) in MATCHED_RATES.items():
        if analytic_bps * 8 != linkspeed_bps:
            raise AssertionError("analytic bytes/s does not match the fluid bits/s")
        if ps_per_byte * linkspeed_bps != 8 * 1_000_000_000_000:
            raise AssertionError("picoseconds per byte is not the matched rate")
    if MATCHED_RATES[200_000_000_000][1] != 2 * MATCHED_RATES[400_000_000_000][1]:
        raise AssertionError("halving the rate must double picoseconds per byte")

    ledger = CAPTURE_LEDGER
    if ledger["prefill_peak_endpoint_bytes"] <= ledger["prefill_peak_egress_bytes"]:
        raise AssertionError("the capture-scale correction lost its direction")
    expected_ratio = 1.676336
    observed_ratio = (
        ledger["prefill_peak_endpoint_bytes"] / ledger["prefill_peak_egress_bytes"]
    )
    if abs(observed_ratio - expected_ratio) > 1e-6:
        raise AssertionError("the capture-scale undercharge literal drifted")
    dispatch = ledger["prefill_dispatch_bytes"]
    if ledger["prefill_peak_endpoint_bytes"] != 2 * dispatch:
        raise AssertionError("prefill dispatch and combine loads are not equal")
    if (
        ledger["prefill_peak_egress_bytes"]
        != dispatch + ledger["prefill_combine_peak_egress_bytes"]
    ):
        raise AssertionError("prefill egress-only aggregate does not decompose")
    if ledger["prefill_segments"] > MAX_SEGMENTS_PER_PHASE * PHASES_PER_STEP:
        raise AssertionError("prefill segment count exceeds the star bound")
    if ledger["total_segments"] > MAX_SEGMENTS_PER_PHASE * PHASES_PER_STEP * REPLAY_STEPS:
        raise AssertionError("total segment count exceeds the star bound")
    if ledger["total_peak_endpoint_bytes"] <= ledger["prefill_peak_endpoint_bytes"]:
        raise AssertionError("the prefill step cannot dominate every step")

    for cell, bytes_key in (
        ("prefill", "prefill_peak_endpoint_bytes"),
        ("total", "total_peak_endpoint_bytes"),
    ):
        for linkspeed_bps, (_, ps_per_byte) in MATCHED_RATES.items():
            ideal_ps = ps_per_byte * ledger[bytes_key]
            if FROZEN_SERIALIZATION_PS[(cell, linkspeed_bps)] != ideal_ps:
                raise AssertionError("frozen serialization literal drifted")
            analytic_ps = FROZEN_ANALYTIC_PS[(cell, linkspeed_bps)]
            if analytic_ps < ideal_ps:
                raise AssertionError("analytic charge fell below its own floor")
            phases = PHASES_PER_STEP if cell == "prefill" else PHASES_PER_STEP * REPLAY_STEPS
            if analytic_ps - ideal_ps >= phases * ANALYTIC_QUANTUM_PS:
                raise AssertionError("analytic quantization exceeds one quantum per phase")
            if analytic_ps % ANALYTIC_QUANTUM_PS:
                raise AssertionError("analytic charge is not whole-nanosecond")
        doubled = FROZEN_SERIALIZATION_PS[(cell, 400_000_000_000)] * 2
        if FROZEN_SERIALIZATION_PS[(cell, 200_000_000_000)] != doubled:
            raise AssertionError("serialization does not scale as one over bandwidth")

    if _analytic_charge_ps(1, 20) != ANALYTIC_QUANTUM_PS:
        raise AssertionError("analytic quantization identity drifted")
    if _analytic_charge_ps(50, 20) != ANALYTIC_QUANTUM_PS:
        raise AssertionError("a whole-nanosecond load must not be rounded up")
    if _analytic_charge_ps(51, 20) != 2 * ANALYTIC_QUANTUM_PS:
        raise AssertionError("a load past the quantum must take the next nanosecond")

    if SCORED_FAMILIES != {"CORE-F1": 3_072, "CORE-F2": 64, "CORE-F3": 64}:
        raise AssertionError("scored evidence registry drifted")
    step_propagation_ps = PHASES_PER_STEP * PROPAGATION_PS
    if step_propagation_ps != 96_000_000:
        raise AssertionError("per-step propagation literal drifted")
    lower = step_propagation_ps - PHASES_PER_STEP * (ANALYTIC_QUANTUM_PS - 1)
    if lower != 95_952_048:
        raise AssertionError("CORE-F3 lower bound drifted")
    if set(SOURCE_ARTIFACTS) != {"steps", "routing"}:
        raise AssertionError("source artifact registry drifted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--htsim-rnic", required=True, type=Path)
    parser.add_argument("--txt2bin", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def check_only(args: argparse.Namespace) -> None:
    _check_frozen_registry()
    if any(
        not str(path)
        for path in (args.out, args.source_root, args.htsim_rnic, args.txt2bin)
    ):
        raise AssertionError("registered path argument is empty")
    print(
        "check-only validated the matched rate pairs, the capture ledger, the "
        "frozen serialization and analytic literals and three scored families; "
        "no artifacts produced"
    )


def main() -> int:
    args = parse_args()
    if args.check_only:
        check_only(args)
        return 0
    raise SystemExit(
        "the observation and comparison path lands after this expectations-only "
        "commit; rerun with --check-only"
    )


if __name__ == "__main__":
    raise SystemExit(main())
