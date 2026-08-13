"""CORE-46: the scalar report fields are a checked projection of the segments.

The fixture finishes its collective ranks out of rank order: the
highest-numbered participant completes first, so one successor is admitted from
a rank-local frontier while the others are admitted from the whole-operation
completion. All three name the same causal predecessor, and only the boundary
comparison separates an additive predecessor from a participant-local one.
"""

from dataclasses import replace

import pytest

from simllm.core.clock import VirtualClock
from simllm.core.completion import CompletionReducer
from simllm.core.execution import (
    CollectiveWork,
    ComputeWork,
    ExecutionGraph,
    ExecutionOperation,
    OperationCorrelation,
)
from simllm.core.runtime import CoarseDeviceProfile, CoarseDeviceRuntime
from simllm.core.step import RequestPhase, ScheduledRequest, StepRecord

CORRELATION = OperationCorrelation(request_ids=("request",))


def _graph():
    return ExecutionGraph(
        execution_id="scalar-projection-out-of-order",
        step_index=0,
        released_at_ps=0,
        operations=(
            ExecutionOperation(
                "collective",
                0,
                "collective",
                CollectiveWork(
                    "all-to-allv",
                    (0, 8, 16),
                    0,
                    "pairwise",
                    pair_payload_bytes=((8, 0, 1_000_000), (16, 8, 1)),
                ),
                correlation=CORRELATION,
            ),
            ExecutionOperation(
                "early",
                16,
                "compute",
                ComputeWork("early", nominal_duration_ps=10),
                participant_local_depends_on=("collective",),
                correlation=CORRELATION,
            ),
            ExecutionOperation(
                "late",
                0,
                "compute",
                ComputeWork("late", nominal_duration_ps=5),
                participant_local_depends_on=("collective",),
                correlation=CORRELATION,
            ),
            ExecutionOperation(
                "barrier",
                8,
                "compute",
                ComputeWork("barrier", nominal_duration_ps=1),
                depends_on=("collective",),
                correlation=CORRELATION,
            ),
        ),
        completion_operation_ids=("early", "late", "barrier"),
    )


def _record():
    return StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[ScheduledRequest("request", RequestPhase.PREFILL, 1)],
        num_sampled=1,
        sampled_request_ids=["request"],
    )


def _execute():
    graph = _graph()
    runtime = CoarseDeviceRuntime(CoarseDeviceProfile())
    execution = runtime.execute(graph)
    assert runtime.last_report is not None
    return graph, execution, runtime.last_report


def _scalar(operation_id, **fields):
    def transform(record):
        if record.operation_id != operation_id:
            return record
        return replace(record, **fields)

    return transform


def _shorten_early(record):
    if record.operation_id != "early":
        return record
    breakdown = record.breakdown
    attribution = record.attribution
    return replace(
        record,
        breakdown=replace(
            breakdown,
            external_dependency_ps=breakdown.external_dependency_ps - 5,
            operation_latency_ps=breakdown.operation_latency_ps - 5,
        ),
        attribution=replace(attribution, queue_ps=attribution.queue_ps - 5),
    )


def test_out_of_order_collective_projects_each_scalar_from_its_segments():
    graph, execution, report = _execute()
    by_id = {record.operation_id: record for record in report.operations}

    collective = by_id["collective"]
    participants = dict(collective.participant_completed_at_ps)
    assert participants[16] == 20
    assert participants[0] == participants[8] == 20_000_000
    assert collective.completed_at_ps == 20_000_000
    assert collective.physical_completed_at_ps == 20_000_000

    early = by_id["early"]
    assert early.causal_predecessor_id == "collective"
    assert early.causal_predecessor_completed_at_ps == 20
    assert early.critical_predecessor_id is None
    assert early.completed_at_ps == 30
    assert early.breakdown.operation_latency_ps == 30
    assert early.critical_segments[0].breakdown.operation_latency_ps == 10

    for name, expected_completion in (("late", 20_000_005), ("barrier", 20_000_001)):
        record = by_id[name]
        assert record.causal_predecessor_id == "collective"
        assert record.causal_predecessor_completed_at_ps == 20_000_000
        assert record.critical_predecessor_id == "collective"
        assert record.completed_at_ps == expected_completion

    assert report.realized_critical_path_operation_ids == ("collective", "late")
    assert report.realized_critical_path_segments == (("collective", 0), ("late", 0))

    clock = VirtualClock(0)
    step_result = CompletionReducer(clock).reduce(
        _record(), graph, execution, report
    )
    assert step_result.completed_at_ps == 20_000_005
    assert execution.completed_at_ps == 20_000_005


@pytest.mark.parametrize(
    ("name", "transform"),
    [
        ("additive-claimed-on-local-boundary", _scalar("early", critical_predecessor_id="collective")),
        ("additive-dropped-on-whole-boundary", _scalar("late", critical_predecessor_id=None)),
        ("invented-boundary", _scalar("early", causal_predecessor_completed_at_ps=17)),
        ("renamed-causal-predecessor", _scalar("late", causal_predecessor_id="barrier")),
        ("invented-physical-completion", _scalar("collective", physical_completed_at_ps=20_000_007)),
        ("shortened-scalar-breakdown", _shorten_early),
    ],
)
def test_scalar_contradiction_is_rejected_without_touching_state(name, transform):
    graph, execution, report = _execute()
    mutated = replace(
        report,
        operations=tuple(transform(record) for record in report.operations),
    )
    clock = VirtualClock(0)
    reducer = CompletionReducer(clock)

    with pytest.raises(ValueError):
        reducer.reduce(_record(), graph, execution, mutated)

    assert clock.now_ps == 0
    assert reducer.latest_request_metrics == ()

    accepted = reducer.reduce(_record(), graph, execution, report)
    assert accepted.completed_at_ps == 20_000_005


def test_asynchronous_scheduler_boundary_may_precede_the_segment_maximum():
    """An early framework release is legal; an invented timestamp is not."""

    from simllm.core.execution import ControlMode, ControlWork

    graph = ExecutionGraph(
        execution_id="async-control",
        step_index=0,
        released_at_ps=0,
        operations=(
            ExecutionOperation(
                "background",
                0,
                "control",
                ControlWork(
                    "background",
                    (8,),
                    1_048_576,
                    ControlMode.ASYNCHRONOUS,
                ),
                correlation=CORRELATION,
            ),
        ),
        completion_operation_ids=("background",),
    )
    runtime = CoarseDeviceRuntime(CoarseDeviceProfile())
    execution = runtime.execute(graph)
    report = runtime.last_report
    assert report is not None
    record = report.operations[0]
    completions = [
        segment.completed_at_ps for segment in record.critical_segments
    ]
    assert record.completed_at_ps in completions
    assert record.completed_at_ps < max(completions)
    assert record.physical_completed_at_ps == max(completions)

    step_record = StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[ScheduledRequest("request", RequestPhase.PREFILL, 1)],
        num_sampled=1,
        sampled_request_ids=["request"],
    )
    accepted = CompletionReducer(VirtualClock(0)).reduce(
        step_record, graph, execution, report
    )
    assert accepted.completed_at_ps == record.completed_at_ps

    invented = replace(
        report,
        operations=(replace(record, completed_at_ps=record.completed_at_ps + 1),),
    )
    with pytest.raises(ValueError):
        CompletionReducer(VirtualClock(0)).reduce(
            step_record, graph, execution, invented
        )
