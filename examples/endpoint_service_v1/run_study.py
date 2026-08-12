"""Run the frozen CORE-41 endpoint-service study after implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

EP_WIDTHS = (2, 4, 8)
PAYLOAD_BYTES = (1_024, 2_048)
SHAPES = ("symmetric", "dispatch-star", "combine-star")
NVLINK_BYTES_PER_SECOND = 450_000_000_000
SCORED_FAMILIES = 4
SCORED_INSTANCES = 20

EXPECTED_ENDPOINT_SERVICE_PS = {
    (1_024, 2): 3_000,
    (1_024, 4): 7_000,
    (1_024, 8): 16_000,
    (2_048, 2): 5_000,
    (2_048, 4): 14_000,
    (2_048, 8): 32_000,
}
EXPECTED_COMBINE_CHANGE_PS = {
    (1_024, 2): 0,
    (1_024, 4): 4_000,
    (1_024, 8): 13_000,
    (2_048, 2): 0,
    (2_048, 4): 9_000,
    (2_048, 8): 27_000,
}
DEPENDENCY_AUTHORITY_REFREEZE = {
    1_024: {
        "old_service_ps": 4_538_000,
        "new_service_ps": 6_652_000,
        "old_jct_ps": 4_562_000,
        "new_jct_ps": 6_676_000,
    },
    2_048: {
        "old_service_ps": 9_047_000,
        "new_service_ps": 13_286_000,
        "old_jct_ps": 9_071_000,
        "new_jct_ps": 13_310_000,
    },
}
SEQGEN_ENDPOINT_BYTES = 16_384
SEQGEN_FLOORS_PS = {
    200_000_000_000: 655_360,
    400_000_000_000: 327_680,
}


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _service_ps(payload_bytes: int) -> int:
    return (
        _ceil_div(
            payload_bytes * 1_000_000_000,
            NVLINK_BYTES_PER_SECOND,
        )
        * 1_000
    )


def _check_frozen_registry() -> None:
    keys = {
        (payload_bytes, width)
        for payload_bytes in PAYLOAD_BYTES
        for width in EP_WIDTHS
    }
    if set(EXPECTED_ENDPOINT_SERVICE_PS) != keys:
        raise AssertionError("endpoint-service grid is incomplete")
    if set(EXPECTED_COMBINE_CHANGE_PS) != keys:
        raise AssertionError("combine-change grid is incomplete")
    if SHAPES != ("symmetric", "dispatch-star", "combine-star"):
        raise AssertionError("fixture-shape registry drifted")

    for payload_bytes, width in sorted(keys):
        endpoint_bytes = (width - 1) * payload_bytes
        expected_service = _service_ps(endpoint_bytes)
        if EXPECTED_ENDPOINT_SERVICE_PS[(payload_bytes, width)] != expected_service:
            raise AssertionError("endpoint-service arithmetic drifted")
        old_combine_service = _service_ps(payload_bytes)
        expected_change = expected_service - old_combine_service
        if EXPECTED_COMBINE_CHANGE_PS[(payload_bytes, width)] != expected_change:
            raise AssertionError("combine-change arithmetic drifted")

        floor_numerator = endpoint_bytes * 1_000_000_000_000
        service_numerator = expected_service * NVLINK_BYTES_PER_SECOND
        if not (
            floor_numerator <= service_numerator
            < floor_numerator + 1_000 * NVLINK_BYTES_PER_SECOND
        ):
            raise AssertionError("service lies outside its physical bounds")
        if expected_service % 1_000:
            raise AssertionError("analytic service is not whole-nanosecond")

    if any(
        EXPECTED_COMBINE_CHANGE_PS[(payload_bytes, 2)] != 0
        for payload_bytes in PAYLOAD_BYTES
    ):
        raise AssertionError("width-two compatibility drifted")
    if any(
        EXPECTED_COMBINE_CHANGE_PS[(payload_bytes, width)] <= 0
        for payload_bytes in PAYLOAD_BYTES
        for width in (4, 8)
    ):
        raise AssertionError("combine response lost its positive direction")
    for width in EP_WIDTHS:
        if (
            EXPECTED_ENDPOINT_SERVICE_PS[(2_048, width)]
            != _service_ps(2 * (width - 1) * 1_024)
        ):
            raise AssertionError("payload scaling arithmetic drifted")

    for row in DEPENDENCY_AUTHORITY_REFREEZE.values():
        service_change = row["new_service_ps"] - row["old_service_ps"]
        jct_change = row["new_jct_ps"] - row["old_jct_ps"]
        if service_change <= 0 or jct_change != service_change:
            raise AssertionError("dependency-authority refreeze arithmetic drifted")
        if row["old_jct_ps"] - row["old_service_ps"] != 24_000:
            raise AssertionError("old dependency compute term drifted")
        if row["new_jct_ps"] - row["new_service_ps"] != 24_000:
            raise AssertionError("new dependency compute term drifted")

    for rate_bps, expected_floor_ps in SEQGEN_FLOORS_PS.items():
        observed_floor_ps = SEQGEN_ENDPOINT_BYTES * 8 * 1_000_000_000_000 // rate_bps
        if observed_floor_ps != expected_floor_ps:
            raise AssertionError("seqgen endpoint floor arithmetic drifted")
    if (SCORED_FAMILIES, SCORED_INSTANCES) != (4, 20):
        raise AssertionError("behavioral evidence registry drifted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "corrected"), required=True)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--htsim-rnic", type=Path, required=True)
    parser.add_argument("--txt2bin", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def check_only(args: argparse.Namespace) -> None:
    _check_frozen_registry()
    if args.mode == "baseline" and args.baseline_summary is not None:
        raise ValueError("baseline mode does not accept --baseline-summary")
    if args.mode == "corrected" and args.baseline_summary is None:
        raise ValueError("corrected mode requires --baseline-summary")
    if any(not str(path) for path in (args.out, args.htsim_rnic, args.txt2bin)):
        raise AssertionError("registered path argument is empty")
    print(
        f"check-only mode={args.mode} out={args.out}; "
        "validated frozen literals and produced no artifacts"
    )


def main() -> None:
    args = parse_args()
    if args.check_only:
        check_only(args)
        return
    raise RuntimeError("result-producing endpoint-service runner is not implemented")


if __name__ == "__main__":
    main()
