"""Tests for the regime-aware collective serializer of TRAF-43."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from simllm.traffic.collective_latency import (
    B200_NCCL_2_27_LOCAL_PROFILE,
    COLLECTIVE_FIXED_COST_FLOOR_PROFILE,
    INTRA_NODE_COLLECTIVE_FIXED_COST_ENVELOPE,
    PICOSECONDS_PER_SECOND,
    CollectiveBandwidthCurve,
    CollectiveFixedCostEnvelope,
    CollectiveLatencyProfile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATION = REPO_ROOT / "examples" / "collective_regime_curve_v1" / "run_validation.py"


def _curve() -> CollectiveBandwidthCurve:
    return CollectiveBandwidthCurve(
        curve_id="test-curve",
        points=((1_000, 10_000_000_000), (1_000_000, 100_000_000_000)),
    )


def test_curve_refuses_a_single_anchor() -> None:
    with pytest.raises(ValueError, match="at least two anchors"):
        CollectiveBandwidthCurve(curve_id="x", points=((1_000, 1),))


def test_curve_refuses_non_increasing_endpoint_bytes() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        CollectiveBandwidthCurve(
            curve_id="x", points=((1_000, 1), (1_000, 2))
        )


def test_curve_refuses_a_blank_identifier() -> None:
    with pytest.raises(ValueError, match="nonblank"):
        CollectiveBandwidthCurve(curve_id="  ", points=((1, 1), (2, 2)))


def test_curve_refuses_a_nonpositive_rate() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        CollectiveBandwidthCurve(curve_id="x", points=((1, 0), (2, 2)))


def test_curve_reproduces_its_anchors_exactly() -> None:
    curve = _curve()
    assert curve.bandwidth_bytes_per_second(1_000) == Fraction(10_000_000_000)
    assert curve.bandwidth_bytes_per_second(1_000_000) == Fraction(100_000_000_000)


def test_curve_clamps_outside_its_anchors() -> None:
    curve = _curve()
    assert curve.bandwidth_bytes_per_second(1) == Fraction(10_000_000_000)
    assert curve.bandwidth_bytes_per_second(0) == Fraction(10_000_000_000)
    assert curve.bandwidth_bytes_per_second(10**12) == Fraction(100_000_000_000)


def test_curve_interpolates_geometrically() -> None:
    # Anchors three decades apart in bytes and one in rate, so the geometric
    # midpoint in log bytes sits at 10**0.5 times the lower rate.
    curve = _curve()
    midpoint = curve.bandwidth_bytes_per_second(31_623)  # sqrt(1e3 * 1e6)
    expected = 10_000_000_000 * (10 ** 0.5)
    assert abs(float(midpoint) - expected) / expected < 1e-3


def test_curve_interpolation_is_monotone_in_bytes() -> None:
    curve = _curve()
    rates = [
        float(curve.bandwidth_bytes_per_second(size))
        for size in (1_000, 3_000, 10_000, 100_000, 1_000_000)
    ]
    assert rates == sorted(rates)


def test_profile_without_a_curve_keeps_the_flat_slope() -> None:
    """Bypass guard B-1: an uncurved profile is bit-for-bit what it was."""

    profile = B200_NCCL_2_27_LOCAL_PROFILE
    assert profile.bandwidth_curves == ()
    for width in profile.supported_participant_counts:
        low, high = profile.endpoint_byte_bounds(width)
        for endpoint_bytes in (low, (low + high) // 2, high):
            expected = -(
                -(endpoint_bytes * PICOSECONDS_PER_SECOND)
                // profile.bandwidth_bytes_per_second
            )
            assert profile.endpoint_serialization_ps(width, endpoint_bytes) == expected
            assert profile.bandwidth_curve(width) is None


def test_shipped_profiles_carry_no_curve() -> None:
    """Bypass guard B-2: no accepted artifact moves in this change."""

    for profile in (
        B200_NCCL_2_27_LOCAL_PROFILE,
        COLLECTIVE_FIXED_COST_FLOOR_PROFILE,
    ):
        assert profile.bandwidth_curves == ()


def _curved_profile(curve: CollectiveBandwidthCurve) -> CollectiveLatencyProfile:
    return CollectiveLatencyProfile(
        profile_id="curved-test",
        bandwidth_bytes_per_second=50_000_000_000,
        participant_latency_ps=((2, 1_000_000), (4, 2_000_000)),
        source_payload_bytes_min=8,
        source_payload_bytes_max=1 << 30,
        propagation_reference_ps=0,
        bandwidth_curves=((2, curve),),
    )


def test_a_curved_width_charges_the_curve_and_others_keep_the_slope() -> None:
    curve = _curve()
    profile = _curved_profile(curve)
    endpoint_bytes = 1_000_000

    curved = profile.endpoint_serialization_ps(2, endpoint_bytes)
    expected_curved = -(
        -(endpoint_bytes * PICOSECONDS_PER_SECOND) // 100_000_000_000
    )
    assert curved == expected_curved

    uncurved = profile.endpoint_serialization_ps(4, endpoint_bytes)
    expected_uncurved = -(
        -(endpoint_bytes * PICOSECONDS_PER_SECOND) // 50_000_000_000
    )
    assert uncurved == expected_uncurved
    assert profile.bandwidth_curve(4) is None


def test_curved_service_time_is_strictly_increasing_in_bytes() -> None:
    profile = _curved_profile(_curve())
    times = [
        profile.endpoint_serialization_ps(2, size)
        for size in (2_000, 20_000, 200_000, 900_000)
    ]
    assert times == sorted(times)
    assert len(set(times)) == len(times)


def test_a_curve_for_an_unsupported_width_is_refused() -> None:
    with pytest.raises(ValueError, match="latency table does not support"):
        CollectiveLatencyProfile(
            profile_id="bad-width",
            bandwidth_bytes_per_second=1,
            participant_latency_ps=((2, 0),),
            source_payload_bytes_min=8,
            source_payload_bytes_max=16,
            propagation_reference_ps=0,
            bandwidth_curves=((8, _curve()),),
        )


def test_curve_widths_must_be_unique_and_increasing() -> None:
    with pytest.raises(ValueError, match="unique and increasing"):
        CollectiveLatencyProfile(
            profile_id="dup-width",
            bandwidth_bytes_per_second=1,
            participant_latency_ps=((2, 0), (4, 0)),
            source_payload_bytes_min=8,
            source_payload_bytes_max=16,
            propagation_reference_ps=0,
            bandwidth_curves=((4, _curve()), (2, _curve())),
        )


def test_envelope_arms_must_agree_on_their_curves() -> None:
    """Bypass guard B-3: a curve on one arm only is refused."""

    shipped = INTRA_NODE_COLLECTIVE_FIXED_COST_ENVELOPE
    assert shipped.lower_profile.bandwidth_curves == ()
    assert shipped.upper_profile.bandwidth_curves == ()

    curve = CollectiveBandwidthCurve(
        curve_id="one-armed",
        points=((8, 1_000_000_000), (1 << 30, 70_027_079_100)),
    )
    upper = replace(
        shipped.upper_profile,
        profile_id="curved-upper",
        bandwidth_curves=((2, curve),),
    )
    with pytest.raises(ValueError, match="share their bandwidth curves"):
        CollectiveFixedCostEnvelope(
            envelope_id="disagreeing",
            claim=shipped.claim,
            lower_profile=shipped.lower_profile,
            upper_profile=upper,
        )


def test_held_out_validation_reproduces_the_recorded_outcome(tmp_path: Path) -> None:
    """Lock the recorded 16 of 20, including which curves fail and why.

    The frozen five-anchor rule passes at width 4 on both machines and fails at
    width 2 on both, because measured serialization bandwidth dips by 22 to 26
    percent around 1 MiB and no interpolation between the 256 KiB and 4 MiB
    anchors can represent a dip between them. A change that improves this must
    move these numbers deliberately.
    """

    out = tmp_path / "validation.json"
    result = subprocess.run(
        [sys.executable, str(VALIDATION), "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert out.exists(), result.stderr
    report = json.loads(out.read_text())

    assert report["scored_total"] == 20
    assert report["scored_passed"] == 16

    by_curve = {
        (entry["machine"], entry["width"]): entry for entry in report["curves"]
    }
    for key in (("a100", 2), ("a100", 4), ("gh200", 2), ("gh200", 4)):
        entry = by_curve[key]
        assert entry["monotone"] is True
        assert entry["worst_anchor_error"] < 1e-9

    # Width 4 clears the bar on both machines.
    assert abs(by_curve[("a100", 4)]["worst_curve_error"]) < 0.15
    assert abs(by_curve[("gh200", 4)]["worst_curve_error"]) < 0.15
    # Width 2 does not, on either, and both miss at the same payload.
    assert abs(by_curve[("a100", 2)]["worst_curve_error"]) > 0.15
    assert abs(by_curve[("gh200", 2)]["worst_curve_error"]) > 0.15
    assert by_curve[("a100", 2)]["worst_curve_payload"] == 1 << 20
    assert by_curve[("gh200", 2)]["worst_curve_payload"] == 1 << 20
