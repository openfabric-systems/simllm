from __future__ import annotations

from dataclasses import replace

import pytest

from simllm.compute import (
    NCCL_STACK_EVENT_SCHEMA,
    NcclChunk,
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
    chunk_bytes=6_144,
    fifo_slots_per_channel=1,
)

INTER_NODE_PROJECTION = [
    ("ncclCommInitRank", "call", "cpu", None),
    ("ncclBuildLogicalChannels", "call", "cpu", None),
    ("ncclConstructLogicalChannel", "call", "cpu", "logical_channel"),
    ("ncclAllReduce", "call", "cpu", None),
    ("ncclPlanAllReduce", "call", "cpu", None),
    ("ncclChunkPayload", "call", "cpu", None),
    ("ncclAssignChunksToChannels", "call", "cpu", None),
    ("ncclLaunchCollectiveKernel", "call", "cpu", None),
    ("ncclCollectiveKernel", "call", "gpu", None),
    ("ncclCopyChunkToFifo", "call", "gpu", "data_fifo_slot"),
    ("ncclStoreReadyFlag", "signal_store", "gpu", "ready_flag"),
    ("ncclStoreHead", "signal_store", "gpu", "head_counter"),
    ("ncclProxyProgress", "call", "cpu", None),
    ("ncclProxyPollHead", "poll_observes", "cpu", "head_counter"),
    ("ncclProxyPollReady", "poll_observes", "cpu", "ready_flag"),
    ("ncclNet.isend", "call", "cpu", None),
    ("ibverbs.post_send", "call", "cpu", None),
    ("ibverbs.write_cqe", "signal_store", "rnic", "completion_queue_entry"),
    ("ncclNet.test", "call", "cpu", None),
    ("ibverbs.poll_cq", "poll_observes", "cpu", "completion_queue_entry"),
    ("ncclProxyStoreTail", "signal_store", "cpu", "tail_counter"),
    ("ncclKernelPollTail", "poll_observes", "gpu", "tail_counter"),
    ("ncclReleaseFifoSlot", "signal_store", "gpu", "ready_flag"),
    ("ncclKernelComplete", "signal_store", "gpu", "kernel_completion"),
]

INTRA_NODE_PROJECTION = [
    ("ncclCommInitRank", "call", "cpu", None),
    ("ncclBuildLogicalChannels", "call", "cpu", None),
    ("ncclConstructLogicalChannel", "call", "cpu", "logical_channel"),
    ("ncclAllReduce", "call", "cpu", None),
    ("ncclPlanAllReduce", "call", "cpu", None),
    ("ncclChunkPayload", "call", "cpu", None),
    ("ncclAssignChunksToChannels", "call", "cpu", None),
    ("ncclLaunchCollectiveKernel", "call", "cpu", None),
    ("ncclCollectiveKernel", "call", "gpu", None),
    ("ncclNvlinkCollective", "call", "gpu", "nvlink"),
    ("ncclKernelComplete", "signal_store", "gpu", "kernel_completion"),
]


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


@pytest.mark.parametrize(
    ("rank", "send_peer", "receive_peer"),
    [(0, 1, 3), (1, 2, 0), (2, 3, 1), (3, 0, 2)],
)
def test_inter_node_reference_sequence_per_rank(rank, send_peer, receive_peer):
    clock, stack, communicator, result = _run_reference(rank, NcclRoute.INTER_NODE)

    assert _projection(stack) == INTER_NODE_PROJECTION
    assert [event.sequence for event in stack.events] == list(range(24))
    assert all(event.timestamp_ps == 0 for event in stack.events)
    assert clock.now_ps == 0
    assert communicator.logical_channels[0].send_peer == send_peer
    assert communicator.logical_channels[0].receive_peer == receive_peer
    assert stack.events[2].peer_rank == send_peer
    assert stack.events[2].value == receive_peer
    assert [stack.events[index].observed_signal_sequence for index in (13, 14, 19, 21)] == [
        11,
        10,
        17,
        20,
    ]
    assert all(event.peer_rank == send_peer for event in stack.events[9:23])
    assert result.plan.wire_bytes == 6_144
    assert result.plan.chunks == (NcclChunk(0, 0, 6_144, 0),)
    assert result.events == stack.events[3:]
    assert result.completion_event == stack.events[23]
    assert result.channel_snapshots[0].head == 1
    assert result.channel_snapshots[0].tail == 1
    assert result.channel_snapshots[0].ready_flags == (False,)
    assert result.channel_snapshots[0].slot_chunk_ids == (None,)
    assert result.channel_snapshots[0].slot_byte_counts == (0,)
    validate_nccl_stack_events(stack.events)


@pytest.mark.parametrize("rank", range(4))
def test_intra_node_reference_sequence_has_no_proxy_or_net(rank):
    clock, stack, _, result = _run_reference(rank, NcclRoute.INTRA_NODE)

    assert _projection(stack) == INTRA_NODE_PROJECTION
    assert clock.now_ps == 0
    assert not any(
        event.function.startswith(("ncclProxy", "ncclNet.", "ibverbs."))
        for event in stack.events
    )
    snapshot = result.channel_snapshots[0]
    assert (snapshot.head, snapshot.tail) == (0, 0)
    assert snapshot.ready_flags == (False,)
    assert snapshot.slot_chunk_ids == (None,)
    assert snapshot.slot_byte_counts == (0,)


@pytest.mark.parametrize(
    ("payload_bytes", "channel_count", "wire_bytes", "chunk_count", "per_channel"),
    [
        (4_096, 1, 6_144, 6, (6,)),
        (4_096, 2, 6_144, 6, (3, 3)),
        (4_096, 4, 6_144, 6, (2, 2, 1, 1)),
        (5_120, 1, 7_680, 8, (8,)),
        (5_120, 2, 7_680, 8, (4, 4)),
        (5_120, 4, 7_680, 8, (2, 2, 2, 2)),
        (8_192, 1, 12_288, 12, (12,)),
        (8_192, 2, 12_288, 12, (6, 6)),
        (8_192, 4, 12_288, 12, (3, 3, 3, 3)),
    ],
)
def test_planner_sweep_exact_counts(
    payload_bytes,
    channel_count,
    wire_bytes,
    chunk_count,
    per_channel,
):
    stack = NcclStack(
        clock=VirtualClock(),
        config=NcclStackConfig(channel_count=channel_count, chunk_bytes=1_024),
    )
    communicator = ncclCommInitRank(
        stack,
        nranks=4,
        communicator_id="planner-ring",
        rank=0,
    )
    plan = stack.planner.plan_all_reduce(
        communicator,
        payload_bytes=payload_bytes,
        operation_id="planner-sweep",
    )

    assert plan.wire_bytes == wire_bytes
    assert len(plan.chunks) == chunk_count
    assert plan.channel_count == channel_count
    assert plan.chunks_per_channel() == per_channel
    assert [chunk.chunk_id for chunk in plan.chunks] == list(range(chunk_count))
    assert [chunk.channel_id for chunk in plan.chunks] == [
        chunk_id % channel_count for chunk_id in range(chunk_count)
    ]
    assert sum(chunk.byte_count for chunk in plan.chunks) == wire_bytes
    assert all(chunk.byte_count == 1_024 for chunk in plan.chunks[:-1])
    expected_last = 512 if payload_bytes == 5_120 else 1_024
    assert plan.chunks[-1].byte_count == expected_last


def test_inter_node_reuses_fifo_slots_and_quiesces_every_channel():
    stack = NcclStack(
        clock=VirtualClock(),
        config=NcclStackConfig(
            channel_count=4,
            chunk_bytes=1_024,
            fifo_slots_per_channel=1,
        ),
    )
    communicator = ncclCommInitRank(
        stack,
        nranks=4,
        communicator_id="multi-chunk",
        rank=0,
    )
    result = ncclAllReduce(
        communicator,
        payload_bytes=4_096,
        operation_id="multi-chunk-allreduce",
        route=NcclRoute.INTER_NODE,
    )

    assert result.plan.chunks_per_channel() == (2, 2, 1, 1)
    assert [(item.head, item.tail) for item in result.channel_snapshots] == [
        (2, 2),
        (2, 2),
        (1, 1),
        (1, 1),
    ]
    assert all(item.ready_flags == (False,) for item in result.channel_snapshots)
    assert all(item.slot_chunk_ids == (None,) for item in result.channel_snapshots)
    assert all(item.slot_byte_counts == (0,) for item in result.channel_snapshots)


def test_event_json_is_strict_versioned_and_round_trips():
    _, stack, _, _ = _run_reference(0, NcclRoute.INTER_NODE)
    payload = nccl_stack_events_to_json(stack.events)

    assert all(item["schema"] == NCCL_STACK_EVENT_SCHEMA for item in payload)
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
    events[13] = replace(events[13], observed_signal_sequence=10)
    with pytest.raises(ValueError, match="disagree"):
        validate_nccl_stack_events(events)

    events = list(stack.events)
    events[5] = replace(events[5], sequence=4)
    with pytest.raises(ValueError, match="expected 5"):
        validate_nccl_stack_events(events)


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


def test_nccl_net_irecv_and_test_are_name_mirrored_observable_seams():
    stack = NcclStack(clock=VirtualClock(), config=REFERENCE_CONFIG)
    communicator = ncclCommInitRank(
        stack,
        nranks=4,
        communicator_id="receive-ring",
        rank=0,
    )
    chunk = NcclChunk(chunk_id=7, offset_bytes=0, byte_count=128, channel_id=0)
    request = stack.nccl_net.irecv(
        communicator,
        operation_id="receive-probe",
        chunk=chunk,
        peer_rank=3,
    )

    assert stack.nccl_net.test(request)
    assert [event.function for event in stack.events[-5:]] == [
        "ncclNet.irecv",
        "ibverbs.post_recv",
        "ibverbs.write_cqe",
        "ncclNet.test",
        "ibverbs.poll_cq",
    ]
    assert stack.events[-1].kind is NcclStackEventKind.POLL_OBSERVES
    assert stack.events[-1].observed_signal_sequence == stack.events[-3].sequence
    validate_nccl_stack_events(stack.events)


@pytest.mark.parametrize(
    "config",
    [
        NcclStackConfig,
    ],
)
def test_stack_requires_the_core_virtual_clock(config):
    with pytest.raises(TypeError, match="VirtualClock"):
        NcclStack(clock=object(), config=config())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"channel_count": 0},
        {"chunk_bytes": 0},
        {"fifo_slots_per_channel": 0},
        {"channel_count": True},
    ],
)
def test_stack_config_rejects_nonpositive_structural_values(kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        NcclStackConfig(**kwargs)


def test_all_reduce_rejects_nonintegral_ring_share():
    stack = NcclStack(clock=VirtualClock(), config=REFERENCE_CONFIG)
    communicator = ncclCommInitRank(
        stack,
        nranks=4,
        communicator_id="bad-plan",
        rank=0,
    )
    before = stack.events

    with pytest.raises(ValueError, match="divide evenly"):
        ncclAllReduce(
            communicator,
            payload_bytes=4_097,
            operation_id="bad-allreduce",
            route=NcclRoute.INTER_NODE,
        )
    assert stack.events == before
