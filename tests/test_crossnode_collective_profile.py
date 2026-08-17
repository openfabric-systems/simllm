"""The measured cross-node collective profile, and the lock on what it must not move.

The profile is the model action of
examples/crossnode_collective_envelope_v1. These tests cover the plumbing that
the study's freeze promised: the profile is selectable by id, it fails closed on
every width the study did not measure, its curve is actually in the pricing path
rather than merely attached, and adding it leaves every shipped profile and
envelope bit-for-bit unchanged.
"""

from __future__ import annotations

import itertools

import pytest

from simllm.traffic.collective_latency import (
    A100_NCCL_2_31_CROSS_NODE_SOCKET_PROFILE as CROSS_NODE_SOCKET,
)
from simllm.traffic.collective_latency import (
    B200_NCCL_2_27_CROSS_NODE_PROVISIONAL_PROFILE,
    B200_NCCL_2_27_LOCAL_PROFILE,
    COLLECTIVE_FIXED_COST_FLOOR_PROFILE,
    CROSS_NODE_COLLECTIVE_FIXED_COST_ENVELOPE,
    INTRA_NODE_COLLECTIVE_FIXED_COST_ENVELOPE,
    PICOSECONDS_PER_SECOND,
    _ceil_div,
    resolve_collective_latency_profile,
)

MEASURED_WIDTHS = (2,)

#: the measured LL to SIMPLE boundary, where completion time genuinely falls
PROTOCOL_STEP_LOW_BYTES = 786_432
PROTOCOL_STEP_HIGH_BYTES = 1_048_576


def test_profile_resolves_by_id_and_by_object() -> None:
    assert (
        resolve_collective_latency_profile("a100-nccl-2.31-cross-node-socket-v1")
        is CROSS_NODE_SOCKET
    )
    assert resolve_collective_latency_profile(CROSS_NODE_SOCKET) is CROSS_NODE_SOCKET


def test_only_the_measured_widths_are_supported() -> None:
    assert CROSS_NODE_SOCKET.supported_participant_counts == MEASURED_WIDTHS


@pytest.mark.parametrize("width", [3, 4, 5, 6, 7, 8, 9, 16])
def test_an_unmeasured_width_fails_closed(width: int) -> None:
    """The freeze's fail-closed clause, with the raise branch driven.

    Widths 4 and 8 are the interesting cases. The study could not place one rank
    per node at width 4 because two of the five cluster nodes were reserved, and
    its width-8 allocation never scheduled, while the shipped B200 profiles
    support 2, 4 and 8. A consumer that silently fell back to a neighbouring
    width would be inventing a measurement.
    """

    with pytest.raises(ValueError) as excinfo:
        CROSS_NODE_SOCKET.base_latency_ps(width)
    assert "does not support participant count" in str(excinfo.value)
    assert "supported counts are 2" in str(excinfo.value)


def test_provenance_is_calibrated_and_complete() -> None:
    provenance = CROSS_NODE_SOCKET.require_provenance()
    assert provenance.evidence_class == "calibrated"
    assert provenance.banded_participant_counts == MEASURED_WIDTHS
    for field in (provenance.source, provenance.locator, provenance.transfer):
        assert field.strip()
    # The transport is the load-bearing qualifier on this profile, so it has to
    # be legible from the provenance record itself and not only from the study.
    assert "socket" in provenance.source.lower()
    assert "gpudirect rdma disabled" in provenance.source.lower()


@pytest.mark.parametrize("width", MEASURED_WIDTHS)
def test_the_band_brackets_the_charged_value(width: int) -> None:
    low, high = CROSS_NODE_SOCKET.base_latency_band_ps(width)
    charged = CROSS_NODE_SOCKET.base_latency_ps(width)
    assert low <= charged <= high
    # The band is the back-to-back to isolated bracket declared in the freeze,
    # so the charged value sits on the lower edge by construction: the
    # back-to-back method amortizes the launch and synchronize roundtrip that
    # the isolated method pays in full.
    assert charged == low
    assert high > low


@pytest.mark.parametrize("width", MEASURED_WIDTHS)
def test_every_measured_width_carries_a_curve(width: int) -> None:
    curve = CROSS_NODE_SOCKET.bandwidth_curve(width)
    assert curve is not None
    assert len(curve.points) >= 2


@pytest.mark.parametrize("width", MEASURED_WIDTHS)
def test_the_curve_is_in_the_pricing_path_not_merely_attached(width: int) -> None:
    """The bug this catches is a silent fall back to the flat slope.

    ``bandwidth_curves`` is optional, and a width missing from it charges the
    profile's single ``bandwidth_bytes_per_second`` at every payload, which is
    exactly the defect TRAF-43 exists to remove. Asserting only that a curve
    object exists would pass even then, so this compares the priced
    serialization against what the flat slope would have returned and requires
    them to differ.
    """

    low, high = CROSS_NODE_SOCKET.endpoint_byte_bounds(width)
    flat = CROSS_NODE_SOCKET.bandwidth_bytes_per_second
    differed = 0
    for load in (low * 8, low * 64, 262_144, 1_048_576, 4_194_304):
        if not low <= load <= high:
            continue
        priced = CROSS_NODE_SOCKET.endpoint_serialization_ps(width, load)
        flat_priced = _ceil_div(load * PICOSECONDS_PER_SECOND, flat)
        if priced != flat_priced:
            differed += 1
    assert differed > 0, (
        f"width {width} priced every sampled load exactly as the flat slope "
        f"would have, so the curve is not in the path"
    )


@pytest.mark.parametrize("width", MEASURED_WIDTHS)
def test_each_curve_reproduces_its_own_anchors(width: int) -> None:
    curve = CROSS_NODE_SOCKET.bandwidth_curve(width)
    assert curve is not None
    for anchor_bytes, anchor_rate in curve.points:
        assert curve.bandwidth_bytes_per_second(anchor_bytes) == anchor_rate


@pytest.mark.parametrize("width", MEASURED_WIDTHS)
def test_service_increases_except_across_the_protocol_step(width: int) -> None:
    """Service rises everywhere except the one span the hardware itself falls in.

    NCCL 2.31.2 switched this collective from the LL protocol to SIMPLE between
    786,432 and 1,048,576 bytes, which the debug log names per call. LL carries
    a 4-byte flag for every 4 bytes of payload, so it puts twice the bytes on
    the wire; that is nearly free on NVLink and dominant on a bandwidth-starved
    socket transport. The measured completion time therefore *falls* by a third
    across that step, and a profile that smoothed it would be optimistic below
    the step and pessimistic above it.
    """

    low, high = CROSS_NODE_SOCKET.endpoint_byte_bounds(width)
    loads = [low]
    load = max(low * 2, low + 1)
    while load < high:
        loads.append(load)
        load *= 4
    loads.append(high)
    services = [CROSS_NODE_SOCKET.total_service_ps(width, load) for load in loads]
    for (earlier_load, earlier), (later_load, later) in itertools.pairwise(
        list(zip(loads, services, strict=True))
    ):
        spans_step = (
            earlier_load < PROTOCOL_STEP_HIGH_BYTES <= later_load
            or earlier_load <= PROTOCOL_STEP_LOW_BYTES < later_load
        )
        if spans_step:
            continue
        assert later > earlier, (
            f"width {width} service did not increase from {earlier_load} to "
            f"{later_load} bytes, and that span does not cross the protocol step"
        )


@pytest.mark.parametrize("width", MEASURED_WIDTHS)
def test_the_protocol_step_is_carried_rather_than_smoothed(width: int) -> None:
    """The model must reproduce the measured fall, not average it away."""

    before = CROSS_NODE_SOCKET.total_service_ps(width, PROTOCOL_STEP_LOW_BYTES)
    after = CROSS_NODE_SOCKET.total_service_ps(width, PROTOCOL_STEP_HIGH_BYTES)
    assert after < before
    # The measured pair is 1456.486 us and 822.835 us, a fall to 0.565 of the
    # smaller payload's time. Pin the direction and the rough size, so a later
    # refit that quietly smooths the step fails here.
    assert 0.4 < after / before < 0.75


@pytest.mark.parametrize("width", MEASURED_WIDTHS)
def test_loads_outside_the_measured_envelope_are_refused(width: int) -> None:
    low, high = CROSS_NODE_SOCKET.endpoint_byte_bounds(width)
    with pytest.raises(ValueError, match="outside profile"):
        CROSS_NODE_SOCKET.validate_endpoint_bytes(width, low - 1)
    with pytest.raises(ValueError, match="outside profile"):
        CROSS_NODE_SOCKET.validate_endpoint_bytes(width, high + 1)


def test_the_new_profile_joins_no_envelope() -> None:
    """Nothing selects the new profile, so no reported metric can move.

    A measured-width-only profile cannot be an envelope arm at all:
    ``CollectiveFixedCostEnvelope`` requires both arms to support identical
    participant counts, and the shipped B200 arms support 2, 4 and 8. That
    limitation is registered rather than worked around.
    """

    for envelope in (
        INTRA_NODE_COLLECTIVE_FIXED_COST_ENVELOPE,
        CROSS_NODE_COLLECTIVE_FIXED_COST_ENVELOPE,
    ):
        assert envelope.lower_profile is not CROSS_NODE_SOCKET
        assert envelope.upper_profile is not CROSS_NODE_SOCKET


def test_shipped_profiles_are_unchanged() -> None:
    """The regression lock behind the freeze's no-artifact-moves promise."""

    assert B200_NCCL_2_27_LOCAL_PROFILE.participant_latency_ps == (
        (2, 10_722_112),
        (4, 15_745_167),
        (8, 30_128_029),
    )
    assert B200_NCCL_2_27_LOCAL_PROFILE.bandwidth_bytes_per_second == 70_027_079_100
    assert B200_NCCL_2_27_LOCAL_PROFILE.bandwidth_curves == ()

    assert COLLECTIVE_FIXED_COST_FLOOR_PROFILE.participant_latency_ps == (
        (2, 0),
        (4, 0),
        (8, 0),
    )

    assert B200_NCCL_2_27_CROSS_NODE_PROVISIONAL_PROFILE.participant_latency_ps == (
        (2, 13_487_792),
        (4, 24_042_207),
        (8, 49_487_789),
    )
    provisional = B200_NCCL_2_27_CROSS_NODE_PROVISIONAL_PROFILE.require_provenance()
    assert provisional.participant_latency_band_ps == (
        (2, 11_487_792, 17_487_792),
        (4, 18_042_207, 36_042_207),
        (8, 35_487_789, 77_487_789),
    )


def test_shipped_envelopes_are_unchanged() -> None:
    intra = INTRA_NODE_COLLECTIVE_FIXED_COST_ENVELOPE
    cross = CROSS_NODE_COLLECTIVE_FIXED_COST_ENVELOPE
    assert intra.supported_participant_counts == (2, 4, 8)
    assert cross.supported_participant_counts == (2, 4, 8)
    assert intra.bracket_ps(8) == (0, 30_128_029)
    assert cross.bracket_ps(8) == (30_128_029, 49_487_789)
    assert intra.realized_bracket_ps(8) == (2_000_000, 32_128_029)
    assert cross.realized_bracket_ps(8) == (32_128_029, 51_487_789)
    assert intra.arm_evidence_class("upper") == "transferred-at-use"
    assert cross.arm_evidence_class("lower") == "transferred-at-use"
    assert cross.arm_evidence_class("upper") == "provisional-transferred"
