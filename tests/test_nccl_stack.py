from __future__ import annotations

from dataclasses import replace

import pytest

import simllm.compute.nccl_stack as nccl_stack_module
from simllm import compute
from simllm.compute import (
    NCCL_STACK_EVENT_SCHEMA,
    NcclRoute,
    NcclStack,
    NcclStackConfig,
    NcclStackEventKind,
    nccl_stack_event_from_json,
    nccl_stack_events_from_json,
    nccl_stack_events_to_json,
    ncclAllReduce,
    ncclCommInitRank,
    validate_nccl_stack_events,
)
from simllm.core import VirtualClock

REFERENCE_CONFIG = NcclStackConfig(
    channel_count=1,
    chunk_bytes=1_024,
    fifo_slots_per_channel=2,
)


def _run_reference(rank: int, route: NcclRoute, *, start_ps: int = 0):
    clock = VirtualClock(start_ps=start_ps)
    stack = NcclStack(clock=clock, config=REFERENCE_CONFIG)
    communicator = ncclCommInitRank(
        stack,
        nranks=4,
        communicator_id="ring-4",
        rank=rank,
    )
    result = ncclAllReduce(
        communicator,
        payload_bytes=4_096,
        operation_id=f"allreduce-r{rank}",
        route=route,
    )
    return clock, stack, communicator, result


def _projection(stack: NcclStack):
    return [
        (event.function, event.kind.value, event.lane.value, event.subject)
        for event in stack.events
    ]


def _expected_inter_projection():
    expected = [
        ("ncclCommInitRank", "call", "cpu", None),
        ("ncclBuildRings", "call", "cpu", None),
        ("initChannel", "call", "cpu", "logical_channel"),
        ("ncclAllReduce", "call", "cpu", None),
        ("ncclEnqueueCheck", "call", "cpu", None),
        ("scheduleCollTasksToPlan", "call", "cpu", None),
        ("calcCollChunking", "call", "cpu", None),
        ("ncclProxySaveOp", "call", "cpu", "proxy_operation"),
        ("ncclLaunchKernel", "call", "cpu", None),
        ("ncclKernelMain", "call", "gpu", None),
        ("runRing", "call", "gpu", None),
    ]
    for batch in range(3):
        for _ in range(2):
            if batch:
                expected.append(("waitPeer", "poll_observes", "gpu", "head_counter"))
            expected.extend(
                (
                    ("waitPeer", "signal_store", "gpu", "ready_flag"),
                    ("genericOp", "call", "gpu", "data_fifo_slot"),
                    ("postPeer", "signal_store", "gpu", "tail_counter"),
                )
            )
        expected.extend(
            (
                ("ncclProxyProgress", "call", "cpu", None),
                ("sendProxyProgress", "call", "cpu", None),
            )
        )
        for _ in range(2):
            expected.extend(
                (
                    ("sendProxyProgress", "poll_observes", "cpu", "tail_counter"),
                    ("sendProxyProgress", "poll_observes", "cpu", "ready_flag"),
                    ("ncclNet.isend", "call", "cpu", None),
                    ("wrap_ibv_post_send", "call", "cpu", None),
                    ("simllmRnicRingDoorbell", "signal_store", "rnic", "doorbell"),
                )
            )
        expected.extend(
            (
                (
                    "simllmNetworkComplete",
                    "signal_store",
                    "rnic",
                    "completion_queue_entry",
                ),
                (
                    "simllmNetworkComplete",
                    "signal_store",
                    "rnic",
                    "completion_queue_entry",
                ),
                ("ncclProxyProgress", "call", "cpu", None),
                ("sendProxyProgress", "call", "cpu", None),
            )
        )
        for _ in range(2):
            expected.extend(
                (
                    ("ncclNet.test", "call", "cpu", None),
                    (
                        "wrap_ibv_poll_cq",
                        "poll_observes",
                        "cpu",
                        "completion_queue_entry",
                    ),
                    ("sendProxyProgress", "signal_store", "cpu", "ready_flag"),
                    ("sendProxyProgress", "signal_store", "cpu", "head_counter"),
                )
            )
    expected.append(
        ("simllmKernelComplete", "signal_store", "gpu", "kernel_completion")
    )
    return expected


def _expected_intra_projection():
    return [
        ("ncclCommInitRank", "call", "cpu", None),
        ("ncclBuildRings", "call", "cpu", None),
        ("initChannel", "call", "cpu", "logical_channel"),
        ("ncclAllReduce", "call", "cpu", None),
        ("ncclEnqueueCheck", "call", "cpu", None),
        ("scheduleCollTasksToPlan", "call", "cpu", None),
        ("calcCollChunking", "call", "cpu", None),
        ("ncclLaunchKernel", "call", "cpu", None),
        ("ncclKernelMain", "call", "gpu", None),
        ("runRing", "call", "gpu", None),
        *[("genericOp", "call", "gpu", "nvlink")] * 6,
        ("simllmKernelComplete", "signal_store", "gpu", "kernel_completion"),
    ]


@pytest.mark.parametrize(
    ("rank", "send_peer", "receive_peer"),
    [(0, 1, 3), (1, 2, 0), (2, 3, 1), (3, 0, 2)],
)
def test_inter_node_reference_sequence_per_rank(rank, send_peer, receive_peer):
    clock, stack, communicator, result = _run_reference(rank, NcclRoute.INTER_NODE)

    assert _projection(stack) == _expected_inter_projection()
    assert [event.sequence for event in stack.events] == list(range(106))
    assert all(event.timestamp_ps == 0 for event in stack.events)
    assert clock.now_ps == 0
    assert communicator.logical_channels[0].send_peer == send_peer
    assert communicator.logical_channels[0].receive_peer == receive_peer
    channel_event = stack.events[2]
    assert channel_event.send_peer_rank == send_peer
    assert channel_event.receive_peer_rank == receive_peer
    assert channel_event.value is None
    assert all(
        event.receive_peer_rank is None
        for event in stack.events
        if event.sequence != 2
    )
    assert all(
        event.send_peer_rank == send_peer
        for event in stack.events
        if event.send_peer_rank is not None
    )

    observed = {
        19: 13,
        20: 11,
        24: 16,
        25: 14,
        34: 29,
        38: 30,
        41: 36,
        45: 40,
        51: 44,
        52: 42,
        56: 48,
        57: 46,
        66: 61,
        70: 62,
        73: 68,
        77: 72,
        83: 76,
        84: 74,
        88: 80,
        89: 78,
        98: 93,
        102: 94,
    }
    assert {
        event.sequence: event.observed_signal_sequence
        for event in stack.events
        if event.kind is NcclStackEventKind.POLL_OBSERVES
    } == observed
    assert result.plan.wire_bytes == 6_144
    assert result.plan.step_count == 6
    assert result.plan.chunks_per_step() == (1, 1, 1, 1, 1, 1)
    assert [chunk.ring_step for chunk in result.plan.chunks] == list(range(6))
    assert [chunk.phase for chunk in result.plan.chunks] == [
        "reduce_scatter",
        "reduce_scatter",
        "reduce_scatter",
        "all_gather",
        "all_gather",
        "all_gather",
    ]
    assert all(chunk.byte_count == 1_024 for chunk in result.plan.chunks)
    assert result.events == stack.events[3:]
    assert result.completion_event == stack.events[105]
    snapshot = result.channel_snapshots[0]
    assert (snapshot.head, snapshot.tail, snapshot.high_watermark) == (6, 6, 2)
    assert snapshot.ready_flags == (False, False)
    assert snapshot.slot_chunk_ids == (None, None)
    assert snapshot.slot_byte_counts == (0, 0)
    validate_nccl_stack_events(stack.events)


@pytest.mark.parametrize("rank", range(4))
def test_intra_node_reference_sequence_has_no_proxy_or_net(rank):
    clock, stack, _, result = _run_reference(rank, NcclRoute.INTRA_NODE)

    assert _projection(stack) == _expected_intra_projection()
    assert len(stack.events) == 17
    assert clock.now_ps == 0
    forbidden = {
        "ncclProxySaveOp",
        "ncclProxyProgress",
        "sendProxyProgress",
        "ncclNet.isend",
        "ncclNet.test",
        "wrap_ibv_post_send",
        "wrap_ibv_poll_cq",
        "simllmRnicRingDoorbell",
        "simllmNetworkComplete",
    }
    assert not forbidden.intersection(event.function for event in stack.events)
    snapshot = result.channel_snapshots[0]
    assert (snapshot.head, snapshot.tail, snapshot.high_watermark) == (0, 0, 0)
    assert snapshot.ready_flags == (False, False)
    assert snapshot.slot_chunk_ids == (None, None)
    assert snapshot.slot_byte_counts == (0, 0)


@pytest.mark.parametrize(
    ("payload_bytes", "channel_count", "wire_bytes", "chunk_count", "per_channel"),
    [
        (4_096, 1, 6_144, 24, (24,)),
        (4_096, 2, 6_144, 24, (12, 12)),
        (4_096, 4, 6_144, 24, (6, 6, 6, 6)),
        (8_192, 1, 12_288, 48, (48,)),
        (8_192, 2, 12_288, 48, (24, 24)),
        (8_192, 4, 12_288, 48, (12, 12, 12, 12)),
        (16_384, 1, 24_576, 96, (96,)),
        (16_384, 2, 24_576, 96, (48, 48)),
        (16_384, 4, 24_576, 96, (24, 24, 24, 24)),
    ],
)
def test_planner_matches_explicit_ring_step_structure(
    payload_bytes,
    channel_count,
    wire_bytes,
    chunk_count,
    per_channel,
):
    stack = NcclStack(
        clock=VirtualClock(),
        config=NcclStackConfig(
            channel_count=channel_count,
            chunk_bytes=256,
            fifo_slots_per_channel=2,
        ),
    )
    communicator = ncclCommInitRank(
        stack,
        nranks=4,
        communicator_id="planner-ring",
        rank=0,
    )
    result = ncclAllReduce(
        communicator,
        payload_bytes=payload_bytes,
        operation_id="planner-sweep",
        route=NcclRoute.INTRA_NODE,
    )
    plan = result.plan

    assert plan.wire_bytes == wire_bytes
    assert len(plan.chunks) == chunk_count
    assert len(communicator.logical_channels) == channel_count
    assert plan.channel_count == channel_count
    assert plan.step_count == 6
    assert plan.chunks_per_channel() == per_channel
    assert plan.chunks_per_step() == (chunk_count // 6,) * 6
    assert [chunk.chunk_id for chunk in plan.chunks] == list(range(chunk_count))
    assert [chunk.ring_step for chunk in plan.chunks] == [
        step for step in range(6) for _ in range(chunk_count // 6)
    ]
    assert sum(chunk.byte_count for chunk in plan.chunks) == wire_bytes
    assert all(chunk.byte_count == 256 for chunk in plan.chunks)


def test_proxy_batch_reaches_fifo_depth_two_before_first_head_advance():
    _, stack, _, result = _run_reference(0, NcclRoute.INTER_NODE)

    assert [stack.events[index].value for index in (13, 16)] == [1, 2]
    first_head = next(
        event
        for event in stack.events
        if event.kind is NcclStackEventKind.SIGNAL_STORE
        and event.subject == "head_counter"
    )
    assert first_head.sequence == 36
    assert result.channel_snapshots[0].high_watermark == 2


def test_posts_precede_doorbells_and_external_completions():
    _, stack, _, _ = _run_reference(0, NcclRoute.INTER_NODE)

    for post, doorbell, completion in zip(
        (22, 27, 54, 59, 86, 91),
        (23, 28, 55, 60, 87, 92),
        (29, 30, 61, 62, 93, 94),
        strict=True,
    ):
        assert stack.events[post].function == "wrap_ibv_post_send"
        assert stack.events[doorbell].function == "simllmRnicRingDoorbell"
        assert stack.events[completion].function == "simllmNetworkComplete"
        assert post < doorbell < completion


def test_event_json_is_strict_versioned_and_round_trips():
    _, stack, _, _ = _run_reference(0, NcclRoute.INTER_NODE)
    payload = nccl_stack_events_to_json(stack.events)

    assert all(item["schema"] == NCCL_STACK_EVENT_SCHEMA for item in payload)
    assert all("peer_rank" not in item for item in payload)
    assert nccl_stack_events_from_json(payload) == stack.events
    assert nccl_stack_event_from_json(payload[0]) == stack.events[0]

    with_unknown = dict(payload[0], unknown=True)
    with pytest.raises(ValueError, match="unknown fields"):
        nccl_stack_event_from_json(with_unknown)
    missing = dict(payload[0])
    del missing["lane"]
    with pytest.raises(ValueError, match="missing fields"):
        nccl_stack_event_from_json(missing)
    wrong_schema = dict(payload[0], schema="simllm-completion-event-v1")
    with pytest.raises(ValueError, match="unsupported schema"):
        nccl_stack_event_from_json(wrong_schema)


def test_stream_validation_rejects_wrong_poll_producer_and_sequence():
    _, stack, _, _ = _run_reference(0, NcclRoute.INTER_NODE)
    events = list(stack.events)
    events[19] = replace(events[19], observed_signal_sequence=11)
    with pytest.raises(ValueError, match="disagree"):
        validate_nccl_stack_events(events)

    events = list(stack.events)
    events[5] = replace(events[5], sequence=4)
    with pytest.raises(ValueError, match="expected 5"):
        validate_nccl_stack_events(events)


def test_observer_rejects_a_foreign_stack_producer_before_mutation():
    _, first, _, _ = _run_reference(0, NcclRoute.INTER_NODE)
    _, second, _, _ = _run_reference(0, NcclRoute.INTER_NODE)
    foreign_producer = first.events[11]
    before = second.events

    with pytest.raises(ValueError, match="belong to this stack observer"):
        second._observer.poll(
            "sendProxyProgress",
            nccl_stack_module.NcclStackLane.CPU,
            foreign_producer,
        )
    assert second.events == before


def test_all_reduce_uses_incremental_observer_validation(monkeypatch):
    stack = NcclStack(clock=VirtualClock(), config=REFERENCE_CONFIG)
    communicator = ncclCommInitRank(
        stack,
        nranks=4,
        communicator_id="incremental-ring",
        rank=0,
    )

    def reject_full_stream_validation(_events):
        raise AssertionError("ncclAllReduce revalidated its cumulative stream")

    monkeypatch.setattr(
        nccl_stack_module,
        "validate_nccl_stack_events",
        reject_full_stream_validation,
    )
    result = ncclAllReduce(
        communicator,
        payload_bytes=4_096,
        operation_id="incremental-allreduce",
        route=NcclRoute.INTER_NODE,
    )
    assert result.completion_event.function == "simllmKernelComplete"


def test_all_boundaries_read_the_caller_supplied_clock_without_advancing_it():
    clock = VirtualClock(start_ps=123)
    stack = NcclStack(clock=clock, config=REFERENCE_CONFIG)
    communicator = ncclCommInitRank(
        stack,
        nranks=4,
        communicator_id="clock-ring",
        rank=0,
    )
    assert [event.timestamp_ps for event in stack.events] == [123, 123, 123]

    clock.advance_to(456)
    ncclAllReduce(
        communicator,
        payload_bytes=4_096,
        operation_id="clock-allreduce",
        route=NcclRoute.INTER_NODE,
    )
    assert [event.timestamp_ps for event in stack.events[:3]] == [123, 123, 123]
    assert all(event.timestamp_ps == 456 for event in stack.events[3:])
    assert clock.now_ps == 456


def test_receive_leg_and_mutable_internals_are_not_package_exports():
    forbidden_exports = {
        "Ibverbs",
        "IbverbsRequest",
        "NcclChannelSnapshot",
        "NcclChunk",
        "NcclCollectivePlan",
        "NcclCommunicator",
        "NcclDataFifoSlot",
        "NcclGpuChannel",
        "NcclLogicalChannel",
        "NcclNetPlugin",
        "NcclNetRequest",
        "NcclProxyProgressEngine",
        "NcclTrafficPlanner",
    }
    assert forbidden_exports.isdisjoint(compute.__all__)
    assert all(not hasattr(compute, name) for name in forbidden_exports)
    stack = NcclStack(clock=VirtualClock(), config=REFERENCE_CONFIG)
    assert not hasattr(stack, "nccl_net")
    assert not hasattr(stack._nccl_net, "irecv")
    assert not hasattr(stack._ibverbs, "post_receive")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"channel_count": 0},
        {"chunk_bytes": 0},
        {"fifo_slots_per_channel": 0},
        {"warps_per_channel": 0},
        {"channel_count": True},
    ],
)
def test_stack_config_rejects_nonpositive_structural_values(kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        NcclStackConfig(**kwargs)


def test_stack_requires_the_core_virtual_clock():
    with pytest.raises(TypeError, match="VirtualClock"):
        NcclStack(clock=object())


@pytest.mark.parametrize(
    ("payload_bytes", "message"),
    [
        (4_097, "divide evenly"),
        (2_048, "per-warp step bytes"),
    ],
)
def test_all_reduce_rejects_invalid_ring_layout_before_emitting(payload_bytes, message):
    stack = NcclStack(clock=VirtualClock(), config=REFERENCE_CONFIG)
    communicator = ncclCommInitRank(
        stack,
        nranks=4,
        communicator_id="bad-plan",
        rank=0,
    )
    before = stack.events

    with pytest.raises(ValueError, match=message):
        ncclAllReduce(
            communicator,
            payload_bytes=payload_bytes,
            operation_id="bad-allreduce",
            route=NcclRoute.INTER_NODE,
        )
    assert stack.events == before
