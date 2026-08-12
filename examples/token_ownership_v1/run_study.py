"""Run the frozen TRAF-25 token-ownership study after implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

EP_WORLDS = (2, 4, 8)
BANDWIDTHS_BPS = (200_000_000_000, 400_000_000_000)
PROFILES = ("rnic-nn-fluid", "rnic-nn")
VECTOR_BYTES = 2_048
TOKEN_COUNT = 54
TOP_K = 8
LAYER_COUNT = 24
PHASE_COUNT = 48
PROPAGATION_PS = 2_000_000
COMPUTE_PS = 99_360_000
PUBLISHED_FLUID_JCT_PS = 974_838_253

BYTE_CELLS = {
    2: {
        "legacy_total": 10_612_736,
        "corrected_total": 5_304_320,
        "legacy_peak": 5_306_368,
        "corrected_peak": 2_652_160,
    },
    4: {
        "legacy_total": 58_773_504,
        "corrected_total": 14_594_048,
        "legacy_peak": 14_792_704,
        "corrected_peak": 7_297_024,
    },
    8: {
        "legacy_total": 207_499_264,
        "corrected_total": 25_563_136,
        "legacy_peak": 27_060_224,
        "corrected_peak": 12_781_568,
    },
}

CORRECTED_FLUID_JCT_PS = {
    (2, 200_000_000_000): 407_532_800,
    (2, 400_000_000_000): 301_446_400,
    (4, 200_000_000_000): 779_121_920,
    (4, 400_000_000_000): 487_240_960,
    (8, 200_000_000_000): 1_217_885_440,
    (8, 400_000_000_000): 706_622_720,
}

REQUEST_BYTES_W8 = {
    "r0": 10_403_840,
    "r1": 5_701_632,
    "r2": 9_457_664,
}


def _check_frozen_registry() -> None:
    if tuple(BYTE_CELLS) != EP_WORLDS:
        raise AssertionError("EP-width byte registry is incomplete")
    if set(CORRECTED_FLUID_JCT_PS) != {
        (world, bandwidth)
        for world in EP_WORLDS
        for bandwidth in BANDWIDTHS_BPS
    }:
        raise AssertionError("fluid JCT registry is incomplete")
    if sum(REQUEST_BYTES_W8.values()) != BYTE_CELLS[8]["corrected_total"]:
        raise AssertionError("request bytes do not conserve corrected total")
    hop_bound = TOKEN_COUNT * TOP_K * LAYER_COUNT * 2
    corrected_hops = BYTE_CELLS[8]["corrected_total"] // VECTOR_BYTES
    legacy_hops = BYTE_CELLS[8]["legacy_total"] // VECTOR_BYTES
    if not corrected_hops <= hop_bound < legacy_hops:
        raise AssertionError("hop bound does not discriminate the defect")
    total_ratios = []
    peak_ratios = []
    for world in EP_WORLDS:
        cell = BYTE_CELLS[world]
        total_ratios.append(cell["legacy_total"] / cell["corrected_total"])
        peak_ratios.append(cell["legacy_peak"] / cell["corrected_peak"])
    if not total_ratios[0] < total_ratios[1] < total_ratios[2]:
        raise AssertionError("total-byte ratios must increase with EP width")
    if not all(1.9 < ratio < 2.3 for ratio in peak_ratios):
        raise AssertionError("peak-rank ratios lost the frozen factor-two shape")
    if not peak_ratios[-1] < total_ratios[-1] / 3:
        raise AssertionError("critical-rank and population responses are conflated")
    for world in EP_WORLDS:
        corrected_bytes = BYTE_CELLS[world]["corrected_total"]
        for bandwidth in BANDWIDTHS_BPS:
            expected = (
                COMPUTE_PS
                + PHASE_COUNT * PROPAGATION_PS
                + corrected_bytes * 8 * 10**12 // bandwidth
            )
            if CORRECTED_FLUID_JCT_PS[(world, bandwidth)] != expected:
                raise AssertionError("fluid JCT arithmetic drifted")
    floor_ps = BYTE_CELLS[8]["corrected_peak"] * 8 * 10**12 // BANDWIDTHS_BPS[1]
    if floor_ps != 255_631_360:
        raise AssertionError("serialization floor drifted")
    if CORRECTED_FLUID_JCT_PS[(8, BANDWIDTHS_BPS[1])] <= floor_ps:
        raise AssertionError("fluid JCT must remain above serialization floor")
    if PUBLISHED_FLUID_JCT_PS <= CORRECTED_FLUID_JCT_PS[(8, BANDWIDTHS_BPS[1])]:
        raise AssertionError("published-to-corrected direction drifted")


def check_only(args: argparse.Namespace) -> None:
    _check_frozen_registry()
    print(
        f"check-only source_root={args.source_root} out={args.out}; "
        "validated frozen literals and produced no artifacts"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_only:
        check_only(args)
        return
    raise SystemExit("token-ownership implementation has not landed")


if __name__ == "__main__":
    main()
