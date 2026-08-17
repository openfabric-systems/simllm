"""Validate the regime-aware collective serializer against measured sweeps.

Builds one :class:`CollectiveBandwidthCurve` per machine and width from the
frozen anchor rule in expectations.md, then scores the interpolated model
against the held-out payloads of the A100 and GH200 hardware envelopes.

The measurements predate this model, so this is a post-specified regression
check, not a pre-registered prediction. What was frozen ahead of the evidence
is the 15 percent bar, registered in TRAF-43, and the anchor rule and held-out
split, frozen in expectations.md before any error here was computed.

Usage::

    python run_validation.py --out validation.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

from simllm.traffic.collective_latency import (
    PICOSECONDS_PER_SECOND,
    CollectiveBandwidthCurve,
)

#: collective payloads the freeze pins the curve at
ANCHOR_PAYLOADS = (8 << 10, 256 << 10, 4 << 20, 64 << 20, 1 << 30)

#: the held-out window; every measured payload inside it that is not an anchor
HELD_OUT_MIN = 1 << 10
HELD_OUT_MAX = 512 << 20

#: the bar TRAF-43 registered before any regime-aware form existed
ACCEPTANCE_WORST_ERROR = 0.15

#: E-5 requires the curve to beat the single slope by at least this factor
MIN_IMPROVEMENT_FACTOR = 3.0

MEASUREMENTS = {
    "a100": Path(__file__).resolve().parents[1]
    / "a100_hardware_envelope_v1"
    / "measurements"
    / "lane_b_result.json",
    "gh200": Path(__file__).resolve().parents[1]
    / "gh200_hardware_envelope_v1"
    / "measurements"
    / "lane_b_result.json",
}


@dataclass(frozen=True)
class Sample:
    """One measured all-reduce point."""

    payload_bytes: int
    endpoint_bytes: int
    total_us: float


def endpoint_bytes(payload_bytes: int, width: int) -> int:
    """Return the nccl-tests bus-convention endpoint load of an all-reduce."""

    return 2 * (width - 1) * payload_bytes // width


def load_samples(path: Path, width: int) -> list[Sample]:
    rows = [
        row
        for row in json.loads(path.read_text())["collectives"]
        if row.get("op") == "allreduce"
        and row.get("width") == width
        and row.get("status") == "measured"
    ]
    rows.sort(key=lambda row: row["bytes"])
    return [
        Sample(row["bytes"], endpoint_bytes(row["bytes"], width), row["time_us"])
        for row in rows
    ]


def build_curve(samples: list[Sample], width: int, floor_us: float, label: str):
    """Return the curve pinned at the frozen anchors of one measured sweep."""

    by_payload = {sample.payload_bytes: sample for sample in samples}
    points: list[tuple[int, int]] = []
    for payload in ANCHOR_PAYLOADS:
        sample = by_payload[payload]
        serialization_us = sample.total_us - floor_us
        if serialization_us <= 0:
            raise ValueError(
                f"{label}: anchor {payload} B sits at or below the latency floor"
            )
        rate = round(sample.endpoint_bytes / (serialization_us * 1e-6))
        points.append((sample.endpoint_bytes, rate))
    return CollectiveBandwidthCurve(curve_id=label, points=tuple(points))


def modeled_us(curve: CollectiveBandwidthCurve, sample: Sample, floor_us: float) -> float:
    rate: Fraction = curve.bandwidth_bytes_per_second(sample.endpoint_bytes)
    serialization_ps = (
        sample.endpoint_bytes * PICOSECONDS_PER_SECOND * rate.denominator
    ) / rate.numerator
    return floor_us + serialization_ps / 1e6


def single_slope_us(sample: Sample, floor_us: float, asymptotic_rate: float) -> float:
    """Return the incumbent two-anchor model's prediction."""

    return floor_us + sample.endpoint_bytes / asymptotic_rate * 1e6


def evaluate(machine: str, width: int) -> dict[str, Any]:
    samples = load_samples(MEASUREMENTS[machine], width)
    floor_us = samples[0].total_us  # the 8 byte point is the latency floor
    label = f"{machine}-nccl-2.31-allreduce-w{width}"
    curve = build_curve(samples, width, floor_us, label)

    largest = samples[-1]
    asymptotic_rate = largest.endpoint_bytes / ((largest.total_us - floor_us) * 1e-6)

    held_out = [
        sample
        for sample in samples
        if HELD_OUT_MIN <= sample.payload_bytes <= HELD_OUT_MAX
        and sample.payload_bytes not in ANCHOR_PAYLOADS
    ]

    rows = []
    for sample in held_out:
        curve_us = modeled_us(curve, sample, floor_us)
        slope_us = single_slope_us(sample, floor_us, asymptotic_rate)
        rows.append(
            {
                "payload_bytes": sample.payload_bytes,
                "measured_us": sample.total_us,
                "curve_us": curve_us,
                "curve_error": (curve_us - sample.total_us) / sample.total_us,
                "single_slope_us": slope_us,
                "single_slope_error": (slope_us - sample.total_us) / sample.total_us,
            }
        )

    anchor_rows = []
    for payload in ANCHOR_PAYLOADS:
        sample = next(s for s in samples if s.payload_bytes == payload)
        curve_us = modeled_us(curve, sample, floor_us)
        anchor_rows.append(
            {
                "payload_bytes": payload,
                "measured_us": sample.total_us,
                "curve_us": curve_us,
                "curve_error": (curve_us - sample.total_us) / sample.total_us,
            }
        )

    worst_curve = max(rows, key=lambda row: abs(row["curve_error"]))
    worst_slope = max(rows, key=lambda row: abs(row["single_slope_error"]))
    monotone = all(
        modeled_us(curve, a, floor_us) < modeled_us(curve, b, floor_us)
        for a, b in pairwise(samples)
    )
    signs = {row["curve_error"] > 0 for row in rows}

    return {
        "machine": machine,
        "width": width,
        "curve_id": label,
        "floor_us": floor_us,
        "anchors": list(curve.points),
        "asymptotic_rate_bytes_per_second": asymptotic_rate,
        "held_out_count": len(rows),
        "held_out": rows,
        "anchor_checks": anchor_rows,
        "worst_curve_error": worst_curve["curve_error"],
        "worst_curve_payload": worst_curve["payload_bytes"],
        "worst_single_slope_error": worst_slope["single_slope_error"],
        "worst_single_slope_payload": worst_slope["payload_bytes"],
        "improvement_factor": abs(worst_slope["single_slope_error"])
        / abs(worst_curve["curve_error"]),
        "monotone": monotone,
        "has_both_signs": signs == {True, False},
        "worst_anchor_error": max(abs(row["curve_error"]) for row in anchor_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results = [
        evaluate(machine, width)
        for machine in ("a100", "gh200")
        for width in (2, 4)
    ]

    checks = []
    for result in results:
        tag = f"{result['machine']} width {result['width']}"
        checks.append(
            (
                "E-1",
                tag,
                abs(result["worst_curve_error"]) <= ACCEPTANCE_WORST_ERROR,
                (
                    f"worst held-out error "
                    f"{result['worst_curve_error'] * 100:+.2f} percent at "
                    f"{result['worst_curve_payload']} B, bar 15"
                ),
            )
        )
        checks.append(
            (
                "E-2",
                tag,
                result["worst_anchor_error"] <= 1e-9,
                f"worst anchor error {result['worst_anchor_error'] * 100:.2e} percent",
            )
        )
        checks.append(
            (
                "E-3",
                tag,
                result["has_both_signs"],
                (
                    "held-out errors carry both signs"
                    if result["has_both_signs"]
                    else "held-out errors are all one sign"
                ),
            )
        )
        checks.append(
            ("E-4", tag, result["monotone"], "modeled time is strictly increasing")
        )
        checks.append(
            (
                "E-5",
                tag,
                result["improvement_factor"] >= MIN_IMPROVEMENT_FACTOR,
                (
                    f"beats the single slope by {result['improvement_factor']:.2f}x "
                    f"({result['worst_single_slope_error'] * 100:+.1f} percent worst), "
                    f"floor 3"
                ),
            )
        )

    passed = sum(1 for _, _, ok, _ in checks if ok)
    print(f"Scored {passed} of {len(checks)} across four curves\n")
    for ident, tag, ok, detail in checks:
        print(f"  {'pass' if ok else 'FAIL'} {ident} {tag}: {detail}")

    args.out.write_text(
        json.dumps(
            {
                "study": "collective_regime_curve_v1",
                "chronology": "post-specified regression check; measurements predate the model",
                "scored_total": len(checks),
                "scored_passed": passed,
                "curves": results,
                "checks": [
                    {"id": i, "curve": t, "passed": ok, "detail": d}
                    for i, t, ok, d in checks
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"\nwrote {args.out}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
