from __future__ import annotations

import importlib.util
import inspect

import pytest

from simllm.adapters.vllm import (
    FLOAT32,
    INT32,
    GroupCoordinatorObserver,
    ShapeTensor,
    SimGroupCoordinator,
)
from simllm.compute import NcclStackConfig
from simllm.core import CollectiveWork, VirtualClock

TORCH_INSTALLED = importlib.util.find_spec("torch") is not None


def make_group(
    size: int,
    *,
    group_name: str = "tp",
    clock: VirtualClock | None = None,
    observer: GroupCoordinatorObserver | None = None,
    chunk_bytes: int = 4,
) -> SimGroupCoordinator:
    if clock is None:
        clock = VirtualClock(start_ps=123_000)
    return SimGroupCoordinator(
        group_name=group_name,
        ranks=tuple(range(size)),
        rank=0,
        local_rank=0,
        clock=clock,
        observer=observer,
        stack_config=NcclStackConfig(
            channel_count=1,
            chunk_bytes=chunk_bytes,
            fifo_slots_per_channel=2,
        ),
    )


@pytest.mark.parametrize(
    ("name", "parameters"),
    [
        ("all_reduce", ("self", "input_")),
        ("all_gather", ("self", "input_", "dim")),
        ("broadcast", ("self", "input_", "src")),
        ("send", ("self", "tensor", "dst")),
        ("recv", ("self", "size", "dtype", "src")),
    ],
)
def test_mirrored_operation_parameter_names(name, parameters):
    signature = inspect.signature(getattr(SimGroupCoordinator, name))
    assert tuple(signature.parameters) == parameters


def test_mirrored_operation_defaults():
    assert inspect.signature(SimGroupCoordinator.all_gather).parameters["dim"].default == -1
    assert inspect.signature(SimGroupCoordinator.broadcast).parameters["src"].default == 0
    assert inspect.signature(SimGroupCoordinator.send).parameters["dst"].default is None
    assert inspect.signature(SimGroupCoordinator.recv).parameters["src"].default is None


def test_rank_and_membership_surface_matches_group_formulas():
    group = SimGroupCoordinator(
        group_name="tp",
        ranks=(4, 7, 9, 12),
        rank=9,
        local_rank=2,
        clock=VirtualClock(),
        stack_config=NcclStackConfig(chunk_bytes=4),
    )

    assert group.rank == 9
    assert group.ranks == [4, 7, 9, 12]
    assert group.group_ranks == [[4, 7, 9, 12]]
    assert group.world_size == 4
    assert group.local_rank == 2
    assert group.rank_in_group == 2
    assert group.first_rank == 4
    assert group.last_rank == 12
    assert group.is_first_rank is False
    assert group.is_last_rank is False
    assert group.next_rank == 12
    assert group.prev_rank == 7
    assert group.cpu_group is group
    assert group.device_group is group


@pytest.mark.parametrize("group_size", [2, 4])
@pytest.mark.parametrize("extent", [8, 16])
def test_shape_and_payload_sweep(group_size, extent):
    group = make_group(group_size)
    input_ = ShapeTensor(
        (4, extent),
        dtype=FLOAT32,
        element_size_bytes=FLOAT32.itemsize,
    )

    reduced = group.all_reduce(input_)
    gathered = group.all_gather(input_, dim=1)
    broadcast = group.broadcast(input_, src=0)
    sent = group.send(input_, dst=1)
    received = group.recv((4, extent), FLOAT32, src=1)

    assert reduced.shape == (4, extent)
    assert reduced.dtype is FLOAT32
    assert gathered.shape == (4, group_size * extent)
    assert gathered.dtype is FLOAT32
    assert broadcast is input_
    assert sent is None
    assert received.shape == (4, extent)
    assert received.dtype is FLOAT32

    assert tuple(event.operation for event in group.events) == (
        "all_reduce",
        "all_gather",
        "broadcast",
        "send",
        "recv",
    )
    assert {event.payload_bytes for event in group.events} == {16 * extent}
    assert tuple(event.sequence for event in group.events) == tuple(range(5))
    assert {event.schema for event in group.events} == {
        "simllm-vllm-group-coordinator-event-v1"
    }
    assert all(event.timestamp_ps == 123_000 for event in group.events)
    assert all(event.stack_events for event in group.events)
    assert {event.stack_disposition for event in group.events} == {"entered"}
    assert tuple(event.work.collective for event in group.events) == (
        "all-reduce",
        "all-gather",
        "broadcast",
        "send",
        "recv",
    )


def test_all_reduce_lowers_to_collective_work_and_frozen_stack_sequence():
    group = make_group(4, chunk_bytes=1_024)
    input_ = ShapeTensor(
        (16, 64),
        dtype=FLOAT32,
        element_size_bytes=FLOAT32.itemsize,
    )

    output = group.all_reduce(input_)
    event = group.events[0]
    expected_stack = (
        "ncclAllReduce",
        "ncclEnqueueCheck",
        "scheduleCollTasksToPlan",
        "calcCollChunking",
        "ncclLaunchKernel",
        "ncclKernelMain",
        "runRing",
        "genericOp",
        "genericOp",
        "genericOp",
        "genericOp",
        "genericOp",
        "genericOp",
        "simllmKernelComplete",
    )

    assert output.shape == input_.shape
    assert event.work == CollectiveWork("all-reduce", (0, 1, 2, 3), 4_096, "ring")
    assert tuple(stack_event.function for stack_event in event.stack_events) == expected_stack
    assert tuple(stack_event.function for stack_event in group.stack_events) == (
        "ncclCommInitRank",
        "ncclBuildRings",
        "initChannel",
        *expected_stack,
    )


def test_singleton_is_the_identity_stack_bypass():
    clock = VirtualClock(start_ps=123_000)
    group = make_group(1, clock=clock)
    input_ = ShapeTensor((4, 8), dtype=FLOAT32, element_size_bytes=4)

    assert group.all_reduce(input_) is input_
    assert group.all_gather(input_, dim=1) is input_
    assert group.broadcast(input_) is input_
    assert group.stack_events == ()
    assert all(event.stack_events == () for event in group.events)
    assert {event.stack_disposition for event in group.events} == {
        "singleton_bypass"
    }
    assert clock.now_ps == 123_000


def test_zero_payload_emits_an_explicit_stack_bypass_event():
    group = make_group(4, chunk_bytes=1)
    output = group.all_reduce(ShapeTensor((0,), element_size_bytes=4))
    event = group.events[0]

    assert output.shape == (0,)
    assert event.sequence == 0
    assert event.operation_id == "tp:all_reduce:0"
    assert event.payload_bytes == 0
    assert event.stack_disposition == "zero_payload_bypass"
    assert event.stack_events == ()
    assert event.work == CollectiveWork("all-reduce", (0, 1, 2, 3), 0, "ring")


def test_unservable_payload_does_not_consume_an_operation_id():
    group = make_group(4, chunk_bytes=1)
    invalid = ShapeTensor((5,), element_size_bytes=2)

    with pytest.raises(ValueError, match="payload_bytes must divide evenly"):
        group.all_reduce(invalid)

    assert group.events == ()
    group.all_reduce(ShapeTensor((1_024,), element_size_bytes=4))
    event = group.events[0]
    assert event.sequence == 0
    assert event.operation_id == "tp:all_reduce:0"
    assert event.stack_disposition == "entered"


@pytest.mark.skipif(not TORCH_INSTALLED, reason="torch is not installed")
def test_recv_accepts_a_real_torch_dtype_when_available():
    import torch

    group = make_group(1)
    received = group.recv(torch.Size((4, 8)), torch.float32)

    assert isinstance(received, torch.Tensor)
    assert received.shape == (4, 8)
    assert received.dtype is torch.float32
    assert group.events[0].payload_bytes == 128


def test_shared_observer_preserves_cross_group_call_order():
    clock = VirtualClock(start_ps=123_000)
    observer = GroupCoordinatorObserver(clock)
    dp_group = make_group(4, group_name="dp", clock=clock, observer=observer)
    tp_group = make_group(
        4,
        group_name="tp",
        clock=clock,
        observer=observer,
        chunk_bytes=1_024,
    )

    dp_group.all_reduce(ShapeTensor((4, 4), dtype=INT32, element_size_bytes=4))
    tp_group.all_reduce(ShapeTensor((16, 64), dtype=FLOAT32, element_size_bytes=4))

    assert tuple((event.operation, event.group, event.payload_bytes) for event in observer.events) == (
        ("all_reduce", "dp", 64),
        ("all_reduce", "tp", 4_096),
    )
    assert tuple(len(event.stack_events) for event in observer.events) == (32, 14)


def test_invalid_groups_and_local_peers_fail_before_observation():
    with pytest.raises(ValueError, match="unique"):
        SimGroupCoordinator(
            group_name="tp",
            ranks=(0, 0),
            rank=0,
            local_rank=0,
            clock=VirtualClock(),
        )
    group = make_group(2)
    tensor = ShapeTensor((4, 8))
    with pytest.raises(ValueError, match="src"):
        group.broadcast(tensor, src=2)
    with pytest.raises(ValueError, match="dst"):
        group.send(tensor, dst=2)
    assert group.events == ()
