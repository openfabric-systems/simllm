"""Run the frozen TRAF-12 dependency-authority study after implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

EVIDENCE_AUTHORED_AGAINST = "dcbef8682b1d74fb059a95d5b8b6f0c4ae07c9eb"
VECTOR_BYTES = (1_024, 2_048)
PLACEMENTS = ("AAAA", "AABB", "ABCD")
PHASE_COUNT = 48
ADJACENT_TRANSITIONS = 47
LEGACY_JCT_PS = {
    1_024: 156_569_755,
    2_048: 217_222_486,
}
EXPECTED_JCT_BANDS = {
    (1_024, "AAAA"): (7_121_000, 7_121_000),
    (1_024, "AABB"): (139_195_840, 139_195_840),
    (1_024, "ABCD"): (160_781_760, 160_781_808),
    (2_048, "AAAA"): (14_180_000, 14_180_000),
    (2_048, "AABB"): (182_367_680, 182_367_680),
    (2_048, "ABCD"): (225_539_520, 225_539_568),
}
EXPECTED_SIGNED_CHANGE_BANDS = {
    1_024: (4_212_005, 4_212_053),
    2_048: (8_317_034, 8_317_082),
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
FROZEN_GRAPH_CENSUS = {
    "operations": 144,
    "explicit_participant_local_references": 212,
    "implicit_fifo_edges": 139,
    "distributed_fifo_edges": 47,
}


def _check_frozen_registry() -> None:
    expected_keys = {
        (vector_bytes, placement)
        for vector_bytes in VECTOR_BYTES
        for placement in PLACEMENTS
    }
    if set(EXPECTED_JCT_BANDS) != expected_keys:
        raise AssertionError("JCT grid is incomplete")
    if set(EXPECTED_SIGNED_CHANGE_BANDS) != set(VECTOR_BYTES):
        raise AssertionError("signed-change grid is incomplete")
    if set(LEGACY_GOAL_ORACLES) != set(VECTOR_BYTES):
        raise AssertionError("legacy GOAL grid is incomplete")
    if PHASE_COUNT - 1 != ADJACENT_TRANSITIONS:
        raise AssertionError("adjacent-transition count drifted")
    if FROZEN_GRAPH_CENSUS["distributed_fifo_edges"] != ADJACENT_TRANSITIONS:
        raise AssertionError("distributed FIFO census drifted")
    for vector_bytes in VECTOR_BYTES:
        low, high = EXPECTED_JCT_BANDS[(vector_bytes, "ABCD")]
        delta_low, delta_high = EXPECTED_SIGNED_CHANGE_BANDS[vector_bytes]
        if (low - LEGACY_JCT_PS[vector_bytes], high - LEGACY_JCT_PS[vector_bytes]) != (
            delta_low,
            delta_high,
        ):
            raise AssertionError("signed-change arithmetic drifted")
        if delta_low <= 0 or delta_high < delta_low:
            raise AssertionError("signed-change direction or band is invalid")
        ordered = [
            EXPECTED_JCT_BANDS[(vector_bytes, placement)][0]
            for placement in PLACEMENTS
        ]
        if not ordered[0] < ordered[1] < ordered[2]:
            raise AssertionError("node-span JCT direction is not increasing")
        size, digest = LEGACY_GOAL_ORACLES[vector_bytes]
        if size <= 0 or len(digest) != 64:
            raise AssertionError("legacy GOAL oracle is malformed")


def check_only(args: argparse.Namespace) -> None:
    _check_frozen_registry()
    print(
        f"check-only out={args.out}; validated frozen literals and produced no artifacts"
    )


def run_study(out: Path) -> None:
    raise RuntimeError(
        "dependency-authority implementation has not landed; use --check-only"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        check_only(args)
        return
    run_study(args.out)


if __name__ == "__main__":
    main()
