import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from types import SimpleNamespace

import pytest

from simllm.backends import (
    BypassArtifacts,
    DeviceRuntimeStepSink,
    ObservedStepLowerer,
    SerialStepLowerer,
    SerialStepLowererConfig,
    assert_bypass_artifact_identity,
    canonical_bypass_parameters,
)
from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
from simllm.core import (
    CoarseDeviceRuntime,
    CompletionReducer,
    ExecutionObservations,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    VirtualClock,
    execution_graph_to_json,
    execution_result_to_json,
    step_result_to_json,
)
from simllm.traffic import project_execution_graph_goal, render_step_goal


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


def _canonical_bytes(value) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _serial_artifact_bundle(config, outcomes) -> BypassArtifacts:
    lowerer = SerialStepLowerer(config)
    diagnostic_goal = bytearray()
    projected_goal = bytearray()
    graph_rows = []
    execution_rows = []
    completion_rows = []
    step_rows = []
    request_rows = []
    for outcome in outcomes:
        graph_rows.append(execution_graph_to_json(outcome.graph))
        execution = execution_result_to_json(outcome.execution_result)
        execution_rows.append(execution)
        completion_rows.append(
            [
                (
                    event["operation_id"],
                    event["phase"],
                    event["timestamp_ps"],
                    event.get("subject_object_id"),
                )
                for event in execution["events"]
            ]
        )
        step = step_result_to_json(outcome.step_result)
        step_rows.append(step)
        request_rows.append(
            [
                (
                    metric["request_id"],
                    metric["completed_at_ps"],
                    metric["ttft_ps"],
                    metric["tpot_ps"],
                )
                for metric in step["request_metrics"]
            ]
        )
        diagnostic_goal.extend(
            render_step_goal(
                outcome.record,
                config.dims,
                config.tp_ranks,
                per_layer_calc_ns=lowerer.timing(outcome.record).layer_calc_ns,
            )
            .render()
            .encode()
        )
        projection = project_execution_graph_goal(outcome.graph)
        projected_goal.extend(
            _canonical_bytes(
                {
                    "artifacts": [
                        artifact.trace.render() for artifact in projection.artifacts
                    ],
                    "boundaries": len(projection.boundaries),
                    "serialized_edges": len(projection.serialized_edges),
                }
            )
        )
    return BypassArtifacts(
        goal_text=bytes(diagnostic_goal),
        # This analytic path has no GOAL compiler. The binary slot therefore
        # locks the graph-derived GOAL artifact bytes, following the existing
        # comparator convention for an unavailable compiler.
        goal_binary=bytes(projected_goal),
        topology=_canonical_bytes(graph_rows),
        profile="coarse-device-default",
        seed=0,
        baseline_parameters=canonical_bypass_parameters(
            {
                "num_layers": config.dims.num_layers,
                "tp_ranks": ",".join(str(rank) for rank in config.tp_ranks),
            }
        ),
        completion_csv=_canonical_bytes(execution_rows),
        canonical_completion=_canonical_bytes(completion_rows),
        step_results=_canonical_bytes(step_rows),
        replay_summary=_canonical_bytes(request_rows),
    )


def _direct_serial_outcomes(config):
    clock = VirtualClock(start_ps=123_000)
    lowerer = SerialStepLowerer(config)
    runtime = CoarseDeviceRuntime()
    reducer = CompletionReducer(clock)
    outcomes = []
    for step_index, phase in enumerate((RequestPhase.PREFILL, RequestPhase.DECODE)):
        record = _record(step_index, clock.now_ps, phase)
        graph = lowerer.lower(record)
        execution = runtime.execute(graph)
        report = runtime.last_report
        assert report is not None
        result = reducer.reduce(record, graph, execution, report)
        outcomes.append(
            SimpleNamespace(
                record=record,
                graph=graph,
                execution_result=execution,
                step_result=result,
            )
        )
    return outcomes


def test_producer_disabled_path_preserves_complete_serial_artifact_bytes():
    config = SerialStepLowererConfig(DIMS, (0, 1), provider=FixedProvider())
    reference = _serial_artifact_bundle(config, _direct_serial_outcomes(config))

    clock = VirtualClock(start_ps=123_000)
    sink = DeviceRuntimeStepSink(config)
    sink.bind_clock(clock)
    for step_index, phase in enumerate((RequestPhase.PREFILL, RequestPhase.DECODE)):
        sink(_record(step_index, clock.now_ps, phase), None)
    candidate = _serial_artifact_bundle(config, sink.outcomes)

    comparison = assert_bypass_artifact_identity(reference, candidate)
    assert comparison.equivalent
    request_rows = json.loads(candidate.replay_summary)
    assert request_rows[0][0][0] == "r0"
    assert request_rows[0][0][2] == 2_004_552
    assert request_rows[0][0][3] is None
    assert request_rows[1][0][0] == "r0"
    assert request_rows[1][0][3] == {"denominator": 1, "numerator": 2_004_552}

    for field in (
        "goal_text",
        "goal_binary",
        "topology",
        "completion_csv",
        "canonical_completion",
        "step_results",
        "replay_summary",
    ):
        changed = replace(candidate, **{field: getattr(candidate, field) + b"!"})
        with pytest.raises(ValueError, match=field):
            assert_bypass_artifact_identity(reference, changed)


def test_producer_disabled_fixed_serial_fixture_is_byte_locked():
    class FlopProvider(ComputeProvider):
        def estimate(self, kernel, gpu):
            return DurationEstimate(duration_ps=int(kernel.flops), bound="compute")

    dims = ModelDims(2, 64, 128, 4, 4, 16, 256, 2)
    record = StepRecord(
        0,
        0,
        [
            ScheduledRequest(
                "p",
                RequestPhase.PREFILL,
                4,
                num_cached_tokens=4,
                context_length=8,
            ),
            ScheduledRequest("d", RequestPhase.DECODE, 1, context_length=32),
        ],
    )
    config = SerialStepLowererConfig(
        dims,
        (0, 1),
        provider=FlopProvider(),
        attach_collective_plan=False,
    )
    graph = ObservedStepLowerer(config).lower(record, None)
    wire = _canonical_bytes(execution_graph_to_json(graph))
    goal = render_step_goal(
        record,
        dims,
        config.tp_ranks,
        per_layer_calc_ns=SerialStepLowerer(config).timing(record).layer_calc_ns,
    ).render().encode()

    assert len(wire) == 4_127
    assert hashlib.sha256(wire).hexdigest() == (
        "aa3c836fe559973a7bf0940384c2e8a84e6af84e0fbd2c02d3b89774ee0c8e2d"
    )
    assert len(goal) == 1_880
    assert hashlib.sha256(goal).hexdigest() == (
        "7087db6780f7e34f5a559a6505eeccc15d984c7b478cd8f0bc5838053825d4b6"
    )
