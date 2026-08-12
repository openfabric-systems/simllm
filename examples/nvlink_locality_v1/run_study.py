"""Run the frozen TRAF-10 NVLink locality study after implementation lands."""

from __future__ import annotations

import argparse
from pathlib import Path

TRACE_SHA256 = "36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341"
NVLINK_BYTES_PER_SECOND = 450_000_000_000
VECTOR_BYTES = (1_024, 2_048)
PLACEMENTS = {
    "AAAA": ("a", "a", "a", "a"),
    "AABB": ("a", "a", "b", "b"),
    "ABCD": ("a", "b", "c", "d"),
}
EXPECTED_CELLS = {
    (1_024, "AAAA"): (11_870_208, 0, 11_870_208, 7_097_000),
    (1_024, "AABB"): (11_870_208, 7_913_472, 3_956_736, 2_442_000),
    (1_024, "ABCD"): (11_870_208, 11_870_208, 0, 0),
    (2_048, "AAAA"): (23_740_416, 0, 23_740_416, 14_156_000),
    (2_048, "AABB"): (23_740_416, 15_826_944, 7_913_472, 4_838_000),
    (2_048, "ABCD"): (23_740_416, 23_740_416, 0, 0),
}
LEGACY_GOAL_ORACLES = {
    1_024: (
        72_819,
        "0417832c8788a0477d48b414cf2d8456b87215abd1d0193ba46fb8db46185d8a",
    ),
    2_048: (
        72_819,
        "bcd72e63546d03efaddd48c16e160457d1e28f19795036d1f871788d78cf5a02",
    ),
}
EXPECTED_JCT_BANDS = {
    (1_024, "AAAA"): (7_121_000, 7_121_000),
    (1_024, "AABB"): (139_195_840, 139_195_840),
    (1_024, "ABCD"): (160_781_760, 160_781_808),
    (2_048, "AAAA"): (14_180_000, 14_180_000),
    (2_048, "AABB"): (182_367_680, 182_367_680),
    (2_048, "ABCD"): (225_539_520, 225_539_568),
}
EXPECTED_PHASES = 48
EXPECTED_POSITIVE_PAIRS = 576


def _check_frozen_registry() -> None:
    if len(TRACE_SHA256) != 64:
        raise AssertionError("trace SHA-256 must contain 64 hexadecimal digits")
    if NVLINK_BYTES_PER_SECOND <= 0:
        raise AssertionError("NVLink rate must be positive")
    expected_keys = {
        (vector_bytes, placement)
        for vector_bytes in VECTOR_BYTES
        for placement in PLACEMENTS
    }
    if set(EXPECTED_CELLS) != expected_keys:
        raise AssertionError("expected-cell grid is incomplete")
    if set(EXPECTED_JCT_BANDS) != expected_keys:
        raise AssertionError("JCT-band grid is incomplete")
    for key, (total, fabric, local, service_ps) in EXPECTED_CELLS.items():
        if min(total, fabric, local, service_ps) < 0:
            raise AssertionError(f"cell {key} contains a negative literal")
        if fabric + local != total:
            raise AssertionError(f"cell {key} does not conserve directed bytes")
    for vector_bytes in VECTOR_BYTES:
        one = EXPECTED_CELLS[(vector_bytes, "AAAA")]
        two = EXPECTED_CELLS[(vector_bytes, "AABB")]
        remote = EXPECTED_CELLS[(vector_bytes, "ABCD")]
        if not (one[1] < two[1] < remote[1]):
            raise AssertionError("fabric-byte direction is not strictly increasing")
        if not (one[2] > two[2] > remote[2]):
            raise AssertionError("NVLink-byte direction is not strictly decreasing")
        low, high = EXPECTED_JCT_BANDS[(vector_bytes, "ABCD")]
        if high - low != EXPECTED_PHASES:
            raise AssertionError("all-remote quantization band drifted")
        ordered = [EXPECTED_JCT_BANDS[(vector_bytes, name)][0] for name in PLACEMENTS]
        if not ordered[0] < ordered[1] < ordered[2]:
            raise AssertionError("JCT direction is not strictly increasing")
    if EXPECTED_CELLS[(2_048, "ABCD")][0] != 2 * EXPECTED_CELLS[(1_024, "ABCD")][0]:
        raise AssertionError("payload sweep no longer doubles directed bytes")
    if any(size <= 0 or len(digest) != 64 for size, digest in LEGACY_GOAL_ORACLES.values()):
        raise AssertionError("legacy GOAL oracle is malformed")
    if EXPECTED_PHASES <= 0 or EXPECTED_POSITIVE_PAIRS <= 0:
        raise AssertionError("phase and pair counts must be positive")


def check_only(args: argparse.Namespace) -> None:
    _check_frozen_registry()
    print(
        f"check-only out={args.out}; validated the frozen TRAF-10 registry "
        "and produced no artifacts"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        check_only(args)
        return
    parser.error("result-producing mode lands with the TRAF-10 implementation")


if __name__ == "__main__":
    main()
