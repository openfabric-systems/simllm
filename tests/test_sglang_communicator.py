from __future__ import annotations

import inspect
import json

import pytest

from simllm.adapters.sglang import (
    FLOAT32,
    GroupCoordinatorEventStream,
    ShapeTensor,
    SimGroupCoordinator,
)
from simllm.adapters.sglang.worker import SimModelRunnerStub
from simllm.adapters.vllm import SimGroupCoordinator as VllmSimGroupCoordinator
from simllm.compute import NcclStackConfig
from simllm.core import CollectiveWork, VirtualClock


def make_group(
    size: int,
    *,
    clock: VirtualClock | None = None,
    chunk_bytes: int = 4,
) -> SimGroupCoordinator:
    return SimGroupCoordinator(
        group_name="tp",
        ranks=tuple(range(size)),
        rank=0,
        local_rank=0,
        clock=clock or VirtualClock(start_ps=123_000),
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
        (
            "all_gather",
            ("self", "input_", "dim", "output_tensor_list"),
        ),
        ("broadcast", ("self", "input_", "src")),
        ("send", ("self", "tensor", "dst")),
        ("recv", ("self", "size", "dtype", "src")),
    ],
)
def test_sglang_mirrored_operation_parameter_names(name, parameters):
    signature = inspect.signature(getattr(SimGroupCoordinator, name))
    assert tuple(signature.parameters) == parameters


def test_sglang_mirrored_operation_annotations():
    signatures = {
        name: inspect.signature(getattr(SimGroupCoordinator, name))
        for name in ("all_reduce", "all_gather", "broadcast", "send", "recv")
    }

    assert str(signatures["all_reduce"]) == (
        "(self, input_: 'torch.Tensor') -> 'torch.Tensor'"
    )
    assert str(signatures["all_gather"]) == (
        "(self, input_: 'torch.Tensor', dim: 'int' = -1, "
        "output_tensor_list: 'Optional[List[torch.Tensor]]' = None) -> 'torch.Tensor'"
    )
    assert str(signatures["broadcast"]) == (
        "(self, input_: 'torch.Tensor', src: 'int' = 0)"
    )
    assert str(signatures["send"]) == (
        "(self, tensor: 'torch.Tensor', dst: 'Optional[int]' = None) -> 'None'"
    )
    assert str(signatures["recv"]) == (
        "(self, size: 'torch.Size', dtype: 'torch.dtype', "
        "src: 'Optional[int]' = None) -> 'torch.Tensor'"
    )


def test_sglang_mirrored_operation_defaults():
    gather = inspect.signature(SimGroupCoordinator.all_gather).parameters
    assert gather["dim"].default == -1
    assert gather["output_tensor_list"].default is None
    assert inspect.signature(SimGroupCoordinator.broadcast).parameters["src"].default == 0
    assert inspect.signature(SimGroupCoordinator.send).parameters["dst"].default is None
    assert inspect.signature(SimGroupCoordinator.recv).parameters["src"].default is None


@pytest.mark.parametrize("group_size", [2, 4])
@pytest.mark.parametrize("extent", [8, 16])
def test_sglang_shape_and_payload_sweep(group_size, extent):
    group = make_group(group_size)
    input_ = ShapeTensor((4, extent), dtype=FLOAT32, element_size_bytes=4)
    parts = [input_.new_empty(input_.shape) for _ in range(group_size)]

    reduced = group.all_reduce(input_)
    gathered = group.all_gather(input_, dim=1)
    gathered_into = group.all_gather(
        input_, dim=1, output_tensor_list=parts
    )
    broadcast = group.broadcast(input_, src=0)
    sent = group.send(input_, dst=1)
    received = group.recv((4, extent), FLOAT32, src=1)

    assert reduced.shape == input_.shape
    assert reduced.dtype is FLOAT32
    assert gathered.shape == (4, group_size * extent)
    assert gathered.dtype is FLOAT32
    assert gathered_into is None
    assert len(parts) == group_size
    assert {part.shape for part in parts} == {input_.shape}
    assert broadcast is input_
    assert sent is None
    assert received.shape == input_.shape
    assert received.dtype is FLOAT32
    assert tuple(event.operation for event in group.events) == (
        "all_reduce",
        "all_gather",
        "all_gather",
        "broadcast",
        "send",
        "recv",
    )
    assert {event.payload_bytes for event in group.events} == {16 * extent}
    assert {event.schema for event in group.events} == {
        "simllm-vllm-group-coordinator-event-v1"
    }
    assert group.clock.now_ps == 123_000


def test_sglang_output_list_is_validated_before_observation():
    group = make_group(2)
    input_ = ShapeTensor((4, 8))

    with pytest.raises(ValueError, match="length must equal world size"):
        group.all_gather(input_, output_tensor_list=[input_])
    with pytest.raises(ValueError, match="does not match input shape"):
        group.all_gather(
            input_, output_tensor_list=[input_, ShapeTensor((2, 16))]
        )

    assert group.events == ()


def test_sglang_reference_call_uses_shared_collective_stack():
    group = make_group(4, chunk_bytes=1_024)
    input_ = ShapeTensor((1, 1_024), dtype=FLOAT32, element_size_bytes=4)
    group.all_reduce(input_)
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

    assert event.work == CollectiveWork("all-reduce", (0, 1, 2, 3), 4_096, "ring")
    assert tuple(item.function for item in event.stack_events) == expected_stack
    assert tuple(item.function for item in group.stack_events) == (
        "ncclCommInitRank",
        "ncclBuildRings",
        "initChannel",
        *expected_stack,
    )


def test_matching_vllm_and_sglang_calls_have_identical_shared_events():
    kwargs = {
        "group_name": "tp",
        "ranks": (0, 1, 2, 3),
        "rank": 0,
        "local_rank": 0,
        "stack_config": NcclStackConfig(
            channel_count=1,
            chunk_bytes=1_024,
            fifo_slots_per_channel=2,
        ),
    }
    vllm = VllmSimGroupCoordinator(clock=VirtualClock(start_ps=123_000), **kwargs)
    sglang = SimGroupCoordinator(clock=VirtualClock(start_ps=123_000), **kwargs)
    input_ = ShapeTensor((1, 1_024), dtype=FLOAT32, element_size_bytes=4)

    vllm_output = vllm.all_reduce(input_)
    sglang_output = sglang.all_reduce(input_)

    assert vllm_output == sglang_output
    assert vllm.events == sglang.events
    assert vllm.stack_events == sglang.stack_events


def test_stub_runner_emits_and_streams_the_bound_tp_event(tmp_path):
    clock = VirtualClock(start_ps=123_000)
    group = make_group(4, clock=clock, chunk_bytes=1_024)
    stream = GroupCoordinatorEventStream(tmp_path / "events.jsonl")
    runner = object.__new__(SimModelRunnerStub)
    runner.bind_simulated_tp_group(group, event_stream=stream)

    event = runner.observe_tp_step()

    assert event is group.events[0]
    assert runner.coordinator_events == (event,)
    assert event.payload_bytes == 4_096
    assert event.timestamp_ps == 123_000
    assert clock.now_ps == 123_000
    rows = [json.loads(line) for line in stream.path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["schema"] == "simllm-vllm-group-coordinator-event-v1"
    assert rows[0]["operation_id"] == "tp:all_reduce:0"
    assert rows[0]["work"] == {
        "algorithm_hint": "ring",
        "channel_hint": None,
        "collective": "all-reduce",
        "payload_bytes": 4_096,
        "ranks": [0, 1, 2, 3],
    }
    assert len(rows[0]["stack_events"]) == 14


def test_stub_runner_without_a_bound_group_is_the_identity_bypass():
    runner = object.__new__(SimModelRunnerStub)

    assert runner.observe_tp_step() is None
    assert runner.coordinator_events == ()
