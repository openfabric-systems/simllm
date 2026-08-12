"""Run the frozen TRAF-14 immutable collective-plan qualification."""

from __future__ import annotations

import argparse
from pathlib import Path

EVIDENCE_AUTHORED_AGAINST = "76223875557a552deb5aa2c2c529a07f000135ba"
BASE_TAG = 1_000
RATES_BPS = (200_000_000_000, 400_000_000_000)
RANKS_BY_WORLD = {
    2: (0, 8),
    4: (0, 8, 16, 24),
}
PAYLOAD_BYTES = (3, 4, 4_096)
SPARSE_CASES = {
    "dispatch": ((0, 8, 3), (0, 16, 5)),
    "combine": ((8, 0, 3), (16, 0, 5)),
    "all-local": (),
}
EXPECTED_RING_ROWS = {
    (2, 3): (2, 1, 4, 4, 4),
    (2, 4): (2, 2, 4, 8, 4),
    (2, 4_096): (2, 2_048, 4, 8_192, 4),
    (4, 3): (6, 1, 24, 24, 40),
    (4, 4): (6, 1, 24, 24, 40),
    (4, 4_096): (6, 1_024, 24, 24_576, 40),
}
EXPECTED_SENTINEL_METRICS_PS = {
    200_000_000_000: 240,
    400_000_000_000: 120,
}
EXPECTED_LARGE_RING_BOUNDS_PS = {
    (2, 200_000_000_000): (163_840, 327_680),
    (2, 400_000_000_000): (81_920, 163_840),
    (4, 200_000_000_000): (245_760, 983_040),
    (4, 400_000_000_000): (122_880, 491_520),
}
LEGACY_WIRE_BYTES = 559
LEGACY_WIRE_SHA256 = (
    "f4a5a70f5bd4a0c2fed874baa88f3035266a54f386a59927e115872c2bcff0a3"
)
EXPECTED_BEHAVIORAL_FAMILIES = 2
EXPECTED_BEHAVIORAL_INSTANCES = 6


def _serialization_ps(byte_count: int, rate_bps: int) -> int:
    numerator = byte_count * 8 * 1_000_000_000_000
    return (numerator + rate_bps - 1) // rate_bps


def _check_frozen_registry() -> None:
    expected_keys = {
        (world, payload)
        for world in RANKS_BY_WORLD
        for payload in PAYLOAD_BYTES
    }
    if set(EXPECTED_RING_ROWS) != expected_keys:
        raise AssertionError("ring sweep is incomplete")
    if set(EXPECTED_SENTINEL_METRICS_PS) != set(RATES_BPS):
        raise AssertionError("sentinel rate sweep is incomplete")
    if set(SPARSE_CASES) != {"dispatch", "combine", "all-local"}:
        raise AssertionError("sparse case registry is incomplete")
    if sum(size for _, _, size in SPARSE_CASES["dispatch"]) != 8:
        raise AssertionError("dispatch byte arithmetic drifted")
    if sum(size for _, _, size in SPARSE_CASES["combine"]) != 8:
        raise AssertionError("combine byte arithmetic drifted")
    if SPARSE_CASES["all-local"]:
        raise AssertionError("all-local semantic collective gained traffic")

    for (world, payload), row in EXPECTED_RING_ROWS.items():
        rounds, chunk, messages, directed_bytes, dependencies = row
        if rounds != 2 * (world - 1):
            raise AssertionError("ring round arithmetic drifted")
        if chunk != max(1, payload // world):
            raise AssertionError("ring chunk arithmetic drifted")
        if messages != world * rounds:
            raise AssertionError("ring message arithmetic drifted")
        if directed_bytes != messages * chunk:
            raise AssertionError("ring byte arithmetic drifted")
        if dependencies != 2 * world * (rounds - 1):
            raise AssertionError("ring dependency arithmetic drifted")

    for rate_bps, predicted_ps in EXPECTED_SENTINEL_METRICS_PS.items():
        floor_ps = 6 * _serialization_ps(1, rate_bps)
        ceiling_ps = 24 * _serialization_ps(1, rate_bps)
        if predicted_ps != floor_ps or not floor_ps <= predicted_ps <= ceiling_ps:
            raise AssertionError("sentinel physical bounds drifted")

    for (world, rate_bps), (floor_ps, ceiling_ps) in (
        EXPECTED_LARGE_RING_BOUNDS_PS.items()
    ):
        rounds, chunk, messages, _, _ = EXPECTED_RING_ROWS[(world, 4_096)]
        if floor_ps != rounds * _serialization_ps(chunk, rate_bps):
            raise AssertionError("large-ring floor drifted")
        if ceiling_ps != messages * _serialization_ps(chunk, rate_bps):
            raise AssertionError("large-ring ceiling drifted")
        if floor_ps > ceiling_ps:
            raise AssertionError("large-ring physical interval is empty")

    if EXPECTED_SENTINEL_METRICS_PS[200_000_000_000] != (
        2 * EXPECTED_SENTINEL_METRICS_PS[400_000_000_000]
    ):
        raise AssertionError("sentinel inverse-rate relation drifted")
    if len(LEGACY_WIRE_SHA256) != 64 or LEGACY_WIRE_BYTES <= 0:
        raise AssertionError("legacy wire oracle is malformed")
    if EXPECTED_BEHAVIORAL_FAMILIES != 2:
        raise AssertionError("behavioral family count drifted")
    if EXPECTED_BEHAVIORAL_INSTANCES != 6:
        raise AssertionError("behavioral instance count drifted")


def check_only(args: argparse.Namespace) -> None:
    _check_frozen_registry()
    print(
        f"check-only out={args.out}; validated frozen literals and produced no artifacts"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_only:
        check_only(args)
        return
    raise RuntimeError("study implementation lands after the expectations freeze")


if __name__ == "__main__":
    main()
