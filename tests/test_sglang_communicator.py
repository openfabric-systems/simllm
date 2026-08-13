from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from simllm.adapters.sglang import (
    FLOAT32,
    GroupCoordinatorEventStream,
    ShapeTensor,
    SimGroupCoordinator,
)
from simllm.adapters.sglang.worker import (
    BatchRow,
    SglStepTranslator,
    SimModelRunnerStub,
    SimWorkerConfig,
)
from simllm.adapters.vllm import SimGroupCoordinator as VllmSimGroupCoordinator
from simllm.compute import NcclStackConfig
from simllm.core import CollectiveWork, StepRecordStream, VirtualClock

FROZEN_FLAG_IDENTITY_FIXTURE = (
    Path(__file__).parent / "fixtures" / "sglang" / "communicator_flag_identity.jsonl"
)
FROZEN_SAMPLED_IDENTITY_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "sglang"
    / "communicator_flag_identity_sampled.jsonl"
)


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
        clock=clock if clock is not None else VirtualClock(start_ps=123_000),
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
    event_path = tmp_path / "events.jsonl"
    event_path.write_text("stale event that first append must remove\n")
    stream = GroupCoordinatorEventStream(event_path)
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

    first_stream_bytes = stream.path.read_bytes()
    second_stream = GroupCoordinatorEventStream(stream.path)
    with pytest.raises(RuntimeError, match="already opened"):
        second_stream.append(event)
    assert stream.path.read_bytes() == first_stream_bytes


def test_stub_runner_without_a_bound_group_is_the_identity_bypass():
    runner = object.__new__(SimModelRunnerStub)

    assert runner.observe_tp_step() is None
    assert runner.coordinator_events == ()


def _stream_frozen_steps(path: Path, *, env: dict[str, str]):
    clock = VirtualClock(start_ps=123_000)
    runner = object.__new__(SimModelRunnerStub)
    config = SimWorkerConfig.from_env(env)
    if config.communicator_tp_size is not None:
        runner.bind_simulated_tp_group(
            make_group(
                config.communicator_tp_size,
                clock=clock,
                chunk_bytes=1_024,
            )
        )
    translator = SglStepTranslator(sample_identity=config.sample_identity)
    stream = StepRecordStream(path)
    rows_by_step = (
        [
            BatchRow(
                rid="sgl11-byte",
                is_decode=False,
                num_new_tokens=4,
                context_length=6,
                cached_tokens=2,
            )
        ],
        [
            BatchRow(
                rid="sgl11-byte",
                is_decode=True,
                num_new_tokens=1,
                context_length=7,
            )
        ],
    )
    for step_index, rows in enumerate(rows_by_step):
        record = translator.translate(
            step_index=step_index,
            virtual_time_ps=clock.now_ps,
            rows=rows,
        )
        runner.observe_tp_step()
        stream.append(record)
        clock.advance_to(clock.now_ps + 1_000)
    return stream.path.read_bytes(), runner.coordinator_events


def test_flag_states_preserve_frozen_step_record_bytes(tmp_path):
    """The communicator flag never changes the step-record bytes.

    The property is checked in both sampled-identity states. SGL-12 added the
    exact sampled count and identity to every record, so the accepted
    pre-SGL-12 fixture is now the compatibility baseline
    (``SIMLLM_SGLANG_SAMPLE_IDENTITY=0``) and the current default has its own
    frozen fixture. Neither fixture may move when the communicator is bound.
    """

    compatibility = {"SIMLLM_SGLANG_SAMPLE_IDENTITY": "0"}
    expected = FROZEN_FLAG_IDENTITY_FIXTURE.read_bytes()
    expected_sampled = FROZEN_SAMPLED_IDENTITY_FIXTURE.read_bytes()

    baseline_bytes, baseline_events = _stream_frozen_steps(
        tmp_path / "flag_off.jsonl",
        env=compatibility,
    )
    enabled_bytes, enabled_events = _stream_frozen_steps(
        tmp_path / "flag_on.jsonl",
        env={**compatibility, "SIMLLM_SGLANG_COMMUNICATOR_TP_SIZE": "4"},
    )
    sampled_baseline_bytes, _ = _stream_frozen_steps(
        tmp_path / "sampled_flag_off.jsonl",
        env={},
    )
    sampled_enabled_bytes, sampled_events = _stream_frozen_steps(
        tmp_path / "sampled_flag_on.jsonl",
        env={"SIMLLM_SGLANG_COMMUNICATOR_TP_SIZE": "4"},
    )

    assert b"\r\n" not in expected
    assert b"\r\n" not in expected_sampled
    assert baseline_bytes == expected
    assert enabled_bytes == expected
    assert baseline_bytes == enabled_bytes
    assert sampled_baseline_bytes == expected_sampled
    assert sampled_enabled_bytes == expected_sampled
    assert sampled_baseline_bytes != baseline_bytes
    assert baseline_events == ()
    assert [event.timestamp_ps for event in enabled_events] == [123_000, 124_000]
    assert [event.timestamp_ps for event in sampled_events] == [123_000, 124_000]
