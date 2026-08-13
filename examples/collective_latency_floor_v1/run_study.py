"""Run the frozen collective-latency-floor study.

Only the check-only expectation validator exists in the freeze commit. The
result-producing implementation lands later.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROFILE = "b200-nccl-2.27-local-v1"
EFFECTIVE_BANDWIDTH_BYTES_PER_SECOND = 70_027_079_100
PARTICIPANT_LATENCY_PS = {
    2: 10_722_112,
    4: 15_745_167,
    8: 30_128_029,
}
PROPAGATION_REFERENCE_PS = 2_000_000
COLLECTIVES_PER_STEP = 48
EXPECTED_STEP_FLOOR_PS = 1_446_145_392
HELD_OUT_ROWS = (
    (2, 4_096, 4_096, 10_520_000, 10_780_604, 1_052_000),
    (4, 4_096, 6_144, 16_180_000, 15_832_905, 1_618_000),
    (8, 4_096, 7_168, 30_310_000, 30_230_390, 3_031_000),
)
SENSITIVITY_ENDPOINT_BYTES = 200_704
SENSITIVITY_TRANSPORT_PS = {
    400_000_000_000: 6_014_080,
    200_000_000_000: 10_028_160,
}


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _prediction_ps(width: int, endpoint_bytes: int) -> int:
    return PARTICIPANT_LATENCY_PS[width] + _ceil_div(
        endpoint_bytes * 1_000_000_000_000,
        EFFECTIVE_BANDWIDTH_BYTES_PER_SECOND,
    )


def check_only(args: argparse.Namespace) -> None:
    """Validate frozen inputs and arithmetic without producing artifacts."""

    if not args.htsim_rnic.is_file() or not os.access(args.htsim_rnic, os.X_OK):
        raise SystemExit(f"htsim_rnic is missing or not executable: {args.htsim_rnic}")
    txt2bin = os.environ.get("SIMLLM_TXT2BIN")
    if txt2bin is not None:
        converter = Path(txt2bin)
        if not converter.is_file() or not os.access(converter, os.X_OK):
            raise SystemExit(f"SIMLLM_TXT2BIN is not executable: {converter}")
    if args.run_dir.exists():
        raise SystemExit(f"check-only run directory must not exist: {args.run_dir}")
    if tuple(PARTICIPANT_LATENCY_PS) != (2, 4, 8):
        raise AssertionError("participant calibration widths changed")
    if EFFECTIVE_BANDWIDTH_BYTES_PER_SECOND >= 900_000_000_000:
        raise AssertionError("effective bandwidth must stay below the physical bound")
    for width, payload_bytes, endpoint_bytes, observed, expected, allowed in HELD_OUT_ROWS:
        if endpoint_bytes != 2 * (width - 1) * payload_bytes // width:
            raise AssertionError("held-out endpoint-byte relation changed")
        predicted = _prediction_ps(width, endpoint_bytes)
        if predicted != expected:
            raise AssertionError("held-out integer prediction changed")
        if allowed != max(1_000_000, observed // 10):
            raise AssertionError("held-out acceptance tolerance changed")
        if abs(predicted - observed) > allowed:
            raise AssertionError("a frozen held-out prediction misses its bar")
        physical_floor_ps = _ceil_div(endpoint_bytes * 1_000_000_000_000, 900_000_000_000)
        if not physical_floor_ps < predicted < 50_000_000:
            raise AssertionError("held-out physical sanity envelope changed")
    if COLLECTIVES_PER_STEP * PARTICIPANT_LATENCY_PS[8] != EXPECTED_STEP_FLOOR_PS:
        raise AssertionError("flagship collective-count arithmetic changed")
    fast = SENSITIVITY_TRANSPORT_PS[400_000_000_000]
    slow = SENSITIVITY_TRANSPORT_PS[200_000_000_000]
    if 2 * fast - slow != PROPAGATION_REFERENCE_PS:
        raise AssertionError("frozen propagation cancellation changed")
    if slow - fast != SENSITIVITY_ENDPOINT_BYTES * 20:
        raise AssertionError("frozen bandwidth sensitivity changed")
    enabled_ratio = (slow + PARTICIPANT_LATENCY_PS[8]) / (
        fast + PARTICIPANT_LATENCY_PS[8]
    )
    if not 1.10 <= enabled_ratio <= 1.12:
        raise AssertionError("enabled sensitivity ratio left its frozen band")
    print(
        f"check-only profile={PROFILE}; run-dir={args.run_dir}; validated "
        "frozen calibration and produced no artifacts"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--htsim-rnic", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    check_only(args)
    if not args.check_only:
        raise SystemExit("result-producing study implementation has not landed")


if __name__ == "__main__":
    main()
