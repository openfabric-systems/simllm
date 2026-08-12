from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest

from simllm.backends import (
    DeviceRuntimeStepSink,
    SerialStepLowerer,
    SerialStepLowererConfig,
)
from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
from simllm.core import (
    ExecutionObservations,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    VirtualClock,
    execution_graph_to_json,
)


class FixedProvider(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=2_000_000, bound="compute")


DIMS = ModelDims(
    num_layers=2,
    hidden_size=64,
    intermediate_size=128,
    num_heads=4,
    num_kv_heads=4,
    head_size=16,
    vocab_size=256,
    dtype_bytes=2,
)
CONFIG = SerialStepLowererConfig(DIMS, (0,), provider=FixedProvider())


def _record(step_index: int, released_at_ps: int, phase: RequestPhase) -> StepRecord:
    return StepRecord(
        step_index,
        released_at_ps,
        [ScheduledRequest("r0", phase, 1, context_length=8 + step_index)],
        num_sampled=1,
        sampled_request_ids=["r0"],
    )


def _serial_observations(record: StepRecord) -> ExecutionObservations:
    graph = SerialStepLowerer(CONFIG).lower(record)
    return ExecutionObservations(
        operations=graph.operations,
        completion_operation_ids=graph.completion_operation_ids,
    )


def test_device_step_sink_requires_and_keeps_one_adapter_clock():
    sink = DeviceRuntimeStepSink(CONFIG)
    record = _record(0, 123_000, RequestPhase.PREFILL)

    with pytest.raises(RuntimeError, match=r"bind_clock\(\)"):
        sink(record, None)

    clock = VirtualClock(start_ps=123_000)
    sink.bind_clock(clock)
    sink.bind_clock(clock)
    with pytest.raises(RuntimeError, match="another clock"):
        sink.bind_clock(VirtualClock(start_ps=123_000))


def test_device_step_sink_none_is_the_exact_serial_lowering():
    clock = VirtualClock(start_ps=123_000)
    sink = DeviceRuntimeStepSink(CONFIG)
    sink.bind_clock(clock)
    record = _record(0, clock.now_ps, RequestPhase.PREFILL)
    expected = SerialStepLowerer(CONFIG).lower(record)

    result = sink(record, None)

    assert execution_graph_to_json(sink.outcomes[0].graph) == execution_graph_to_json(
        expected
    )
    assert sink.outcomes[0].observations is None
    assert result.step_latency_ps == 2_000_000
    assert result.completed_at_ps == 2_123_000
    assert result.request_metrics[0].request_id == "r0"
    assert result.request_metrics[0].ttft_ps == 2_000_000
    assert clock.now_ps == result.completed_at_ps


def test_device_step_sink_routes_observations_and_preserves_request_metrics():
    clock = VirtualClock(start_ps=0)
    sink = DeviceRuntimeStepSink(CONFIG)
    sink.bind_clock(clock)

    prefill = _record(0, clock.now_ps, RequestPhase.PREFILL)
    prefill_observations = _serial_observations(prefill)
    first = sink(prefill, prefill_observations)

    decode = _record(1, clock.now_ps, RequestPhase.DECODE)
    decode_observations = _serial_observations(decode)
    second = sink(decode, decode_observations)

    assert len(sink.outcomes) == 2
    assert sink.outcomes[0].observations is prefill_observations
    assert sink.outcomes[1].observations is decode_observations
    assert first.request_metrics[0].ttft_ps == 2_000_000
    assert first.request_metrics[0].tpot_ps is None
    assert second.request_metrics[0].request_id == "r0"
    assert second.request_metrics[0].tpot_ps == Fraction(2_000_000, 1)
    assert clock.now_ps == second.completed_at_ps == 4_000_000


def test_device_step_sink_rejects_a_clock_record_disagreement():
    clock = VirtualClock(start_ps=100)
    sink = DeviceRuntimeStepSink(CONFIG)
    sink.bind_clock(clock)

    with pytest.raises(ValueError, match="does not equal"):
        sink(_record(0, 99, RequestPhase.PREFILL), None)


def test_device_step_outcomes_are_immutable_snapshots():
    clock = VirtualClock()
    sink = DeviceRuntimeStepSink(CONFIG)
    sink.bind_clock(clock)
    record = _record(0, 0, RequestPhase.PREFILL)
    sink(record, _serial_observations(record))

    with pytest.raises(FrozenInstanceError):
        sink.outcomes[0].record = "not-a-record"
