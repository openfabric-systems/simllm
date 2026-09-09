"""One common consumer for native wire and peer packet observations."""

from dataclasses import replace

import pytest

from simllm.core.packet_port import (
    PACKET_EVENT_FIELDS,
    PacketAttemptEvent,
    PacketExtentAdmission,
    PacketExtentTerminal,
    PacketPortContext,
    PacketPortError,
    PacketPortLedger,
)


def event(kind="packet_tx_started", at=10, *, token=2, attempt=0, **changes):
    values = {
        "attempt_token": token,
        "extent_token": 1,
        "wqe_id": 17,
        "event_kind": kind,
        "event_time_ps": at,
        "extent_index": 0,
        "packet_index": 0,
        "transmission_attempt": attempt,
        "payload_offset_bytes": 0,
        "payload_bytes": 256,
        "wire_bytes": 272,
        "packet_kind": "data" if attempt == 0 else "retransmission",
    }
    values.update(changes)
    return PacketAttemptEvent(**values)


def ledger(*, boundary="producer_issue"):
    result = PacketPortLedger(PacketPortContext("session", "port", tx_start_boundary=boundary))
    result.admit(PacketExtentAdmission(1, 17, 256, 10))
    return result


def delivered(result, *, token=2, attempt=0, start=10, **changes):
    for kind, at in (
        ("packet_tx_started", start),
        ("packet_tx_finished", start + 10),
        ("packet_rx_arrived", start + 12),
        ("delivered", start + 40),
    ):
        result.observe(event(kind, at, token=token, attempt=attempt, **changes))


@pytest.mark.parametrize("boundary", ("producer_issue", "link_grant"))
def test_one_consumer_accepts_both_declared_boundaries(boundary):
    result = ledger(boundary=boundary)
    delivered(result)
    result.retire(PacketExtentTerminal(1, 17, "delivered", 50))
    snapshot = result.snapshot(final=True)
    assert snapshot.delivered_payload_bytes == 256
    assert snapshot.live_attempt_tokens == snapshot.live_extent_tokens == ()
    assert snapshot.started_attempt_tokens == snapshot.retired_attempt_tokens == (2,)
    rows = [row.to_row() for row in result.events]
    assert all(tuple(row) == PACKET_EVENT_FIELDS for row in rows)
    assert tuple(PacketAttemptEvent.from_row(row) for row in rows) == result.events


def test_intermediate_snapshot_retains_acknowledgement_tail_and_session_tokens():
    result = ledger(boundary="link_grant")
    for kind, at in (
        ("packet_tx_started", 10),
        ("packet_tx_finished", 20),
        ("packet_rx_arrived", 22),
    ):
        result.observe(event(kind, at))
    live = result.snapshot()
    assert live.live_extent_tokens == (1,)
    assert live.live_attempt_tokens == (2,)
    with pytest.raises(PacketPortError, match="missing transport terminals"):
        result.snapshot(final=True)
    assert result.snapshot() == live
    # A later logical phase may be admitted before the first transport retires.
    result.admit(PacketExtentAdmission(3, 18, 256, 25))
    for kind, at in (
        ("packet_tx_started", 25),
        ("packet_tx_finished", 35),
        ("packet_rx_arrived", 37),
    ):
        result.observe(event(kind, at, token=4, extent_token=3, wqe_id=18))
    result.observe(event("delivered", 50))
    result.retire(PacketExtentTerminal(1, 17, "delivered", 50))
    result.observe(event("delivered", 65, token=4, extent_token=3, wqe_id=18))
    result.retire(PacketExtentTerminal(3, 18, "delivered", 65))
    assert result.snapshot(final=True).delivered_payload_bytes == 512
    for token in (1, 2, 3, 4):
        before = result.snapshot()
        with pytest.raises(PacketPortError, match="already used"):
            result.admit(PacketExtentAdmission(token, 19, 1, 100))
        assert result.snapshot() == before


def test_early_drop_and_retry_do_not_double_credit_payload():
    result = ledger()
    result.observe(event())
    result.observe(event("dropped", 10))
    delivered(result, token=3, attempt=1, start=30)
    result.retire(PacketExtentTerminal(1, 17, "delivered", 70))
    snapshot = result.snapshot(final=True)
    assert snapshot.delivered_payload_bytes == 256
    assert snapshot.retired_attempt_tokens == (2, 3)


def test_identical_successful_retry_range_is_counted_once():
    result = ledger()
    delivered(result)
    delivered(result, token=3, attempt=1, start=60)
    result.retire(PacketExtentTerminal(1, 17, "delivered", 100))
    assert result.snapshot(final=True).delivered_payload_bytes == 256


@pytest.mark.parametrize(
    "mutation",
    (
        {"wire_bytes": 288},
        {"payload_bytes": 128},
        {"wqe_id": 18},
        {"extent_index": 1},
        {"packet_index": 1},
        {"transmission_attempt": 1},
        {"event_kind": "packet_rx_arrived"},
        {"event_time_ps": 9},
    ),
)
def test_invalid_next_observation_is_atomic(mutation):
    result = ledger()
    result.observe(event())
    before, rows = result.snapshot(), result.events
    bad = replace(event("packet_tx_finished", 20), **mutation)
    with pytest.raises(PacketPortError):
        result.observe(bad)
    assert result.snapshot() == before and result.events == rows


@pytest.mark.parametrize("ranges", (((0, 128),), ((0, 128), (64, 128)), ((0, 64), (128, 128))))
def test_missing_or_overlapping_logical_coverage_rejects_retirement(ranges):
    result = ledger()
    for index, (offset, size) in enumerate(ranges):
        delivered(
            result,
            token=2 + index,
            start=10 + 50 * index,
            packet_index=index,
            payload_offset_bytes=offset,
            payload_bytes=size,
        )
    before = result.snapshot()
    with pytest.raises(PacketPortError, match="coverage|gap|overlap"):
        result.retire(PacketExtentTerminal(1, 17, "delivered", 200))
    assert result.snapshot() == before


def test_parent_cannot_retire_before_child_or_accept_a_late_child():
    result = ledger()
    result.observe(event())
    with pytest.raises(PacketPortError, match="outstanding child"):
        result.retire(PacketExtentTerminal(1, 17, "dropped", 11))
    result.observe(event("dropped", 12))
    result.retire(PacketExtentTerminal(1, 17, "dropped", 12))
    before = result.snapshot()
    with pytest.raises(PacketPortError, match="follows parent"):
        result.observe(event("delivered", 13))
    with pytest.raises(PacketPortError, match="duplicate extent"):
        result.retire(PacketExtentTerminal(1, 17, "dropped", 12))
    assert result.snapshot() == before


@pytest.mark.parametrize(
    "capability",
    (
        "ecn_marking",
        "priority_flow_control",
        "congestion_notification",
        "unknown",
    ),
)
def test_unadvertised_capability_rejects_before_construction(capability):
    context = PacketPortContext("session", "gpu-peer")
    with pytest.raises(PacketPortError, match="does not support"):
        context.require(capability)
    assert context == PacketPortContext("session", "gpu-peer")


def test_disabled_and_version_one_packet_emission_are_explicit_rejections():
    with pytest.raises(PacketPortError, match="version 2"):
        PacketPortLedger(PacketPortContext("session", "port", abi_version=1))
    with pytest.raises(PacketPortError, match="does not support"):
        PacketPortLedger(PacketPortContext("session", "port", packet_attempt_events=False))


@pytest.mark.parametrize("kind", ("cnp", "pfc"))
def test_unadvertised_control_event_is_rejected_atomically(kind):
    result = ledger()
    before = result.snapshot()
    with pytest.raises(PacketPortError, match="does not advertise"):
        result.observe(event(packet_kind=kind, payload_bytes=0))
    assert result.snapshot() == before and result.events == ()


def test_explicit_control_decoder_capability_agrees_with_permitted_events():
    context = PacketPortContext("session", "decoder", packet_kinds=("data", "pfc"))
    context.require("priority_flow_control")
    result = PacketPortLedger(context)
    result.admit(PacketExtentAdmission(1, 17, 256, 10))
    result.observe(event(packet_kind="pfc", payload_bytes=0))
    assert result.snapshot().live_attempt_tokens == (2,)


def test_separate_arrays_preserve_observable_packet_callback_order():
    result = PacketPortLedger(PacketPortContext("session", "port", event_order_evidence="separate_arrays"))
    result.admit(PacketExtentAdmission(1, 17, 256, 0))
    result.admit(PacketExtentAdmission(3, 18, 256, 0))
    delivered(result, start=100)
    before = result.snapshot()
    with pytest.raises(PacketPortError, match="packet event array time moved backwards"):
        result.observe(event(at=50, token=4, extent_token=3, wqe_id=18))
    assert result.snapshot() == before


@pytest.mark.parametrize(
    "changes",
    (
        {"event_kind": "ecn_marked"},
        {"attempt_token": True},
        {"attempt_token": 2**64},
        {"transmission_attempt": 2**32},
        {"extent_index": 2**32},
        {"wire_bytes": 1},
    ),
)
def test_unsupported_grammar_and_native_integer_overflow_reject(changes):
    with pytest.raises(PacketPortError):
        event(**changes)
