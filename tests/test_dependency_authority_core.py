import pytest

from simllm.core.clock import VirtualClock
from simllm.core.completion import CompletionReducer
from simllm.core.execution import (
    CollectiveWork,
    ComputeWork,
    ControlMode,
    ControlWork,
    DependencyOrigin,
    DependencyScope,
    EffectiveDependencyEdge,
    ExecutionGraph,
    ExecutionOperation,
    OperationCorrelation,
)
from simllm.core.execution_io import effective_dependency_edges, validate_execution_graph
from simllm.core.runtime import CoarseDeviceProfile, CoarseDeviceRuntime
from simllm.core.step import RequestPhase, ScheduledRequest, StepRecord


def test_effective_dependency_edges_expand_every_authoritative_scope():
    graph = ExecutionGraph(
        execution_id="edge-census",
        step_index=0,
        released_at_ps=0,
        operations=(
            ExecutionOperation(
                "whole",
                2,
                "whole-queue",
                ComputeWork("whole", nominal_duration_ps=1),
            ),
            ExecutionOperation(
                "local",
                0,
                "shared-queue",
                CollectiveWork("all-reduce", (0, 1), 2, "ring"),
            ),
            ExecutionOperation(
                "target",
                0,
                "shared-queue",
                CollectiveWork("all-reduce", (0, 1), 2, "ring"),
                depends_on=("whole",),
                participant_local_depends_on=("local",),
            ),
        ),
    )
    assert effective_dependency_edges(graph) == (
        EffectiveDependencyEdge(
            "whole",
            "target",
            DependencyScope.WHOLE_OPERATION,
            DependencyOrigin.EXPLICIT,
        ),
        EffectiveDependencyEdge(
            "local",
            "target",
            DependencyScope.PARTICIPANT_LOCAL,
            DependencyOrigin.EXPLICIT,
            0,
        ),
        EffectiveDependencyEdge(
            "local",
            "target",
            DependencyScope.PARTICIPANT_LOCAL,
            DependencyOrigin.EXPLICIT,
            1,
        ),
        EffectiveDependencyEdge(
            "local",
            "target",
            DependencyScope.WHOLE_OPERATION,
            DependencyOrigin.LOGICAL_QUEUE_FIFO,
        ),
    )


def test_collective_anchor_must_be_a_canonical_participant():
    graph = ExecutionGraph(
        "bad-anchor",
        0,
        0,
        (
            ExecutionOperation(
                "collective",
                7,
                "collective",
                CollectiveWork("all-reduce", (0, 1), 2, "ring"),
            ),
        ),
    )

    with pytest.raises(ValueError, match="anchor rank"):
        validate_execution_graph(graph)


def test_participant_local_report_keeps_exact_asymmetric_causal_boundary():
    correlation = OperationCorrelation(request_ids=("request",))
    graph = ExecutionGraph(
        execution_id="asymmetric-local-boundary",
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
                    pair_payload_bytes=(
                        (0, 8, 1),
                        (8, 16, 1_000_000),
                    ),
                ),
                correlation=correlation,
            ),
            ExecutionOperation(
                "target",
                0,
                "compute",
                ComputeWork("target", nominal_duration_ps=10),
                participant_local_depends_on=("collective",),
                correlation=correlation,
            ),
        ),
        completion_operation_ids=("target",),
    )
    runtime = CoarseDeviceRuntime(CoarseDeviceProfile())

    result = runtime.execute(graph)

    assert runtime.last_report is not None
    report = runtime.last_report
    by_id = {operation.operation_id: operation for operation in report.operations}
    collective = by_id["collective"]
    target = by_id["target"]
    assert dict(collective.participant_completed_at_ps)[0] == 20
    assert collective.completed_at_ps == 20_000_000
    assert target.causal_predecessor_id == "collective"
    assert target.causal_predecessor_completed_at_ps == 20
    assert target.critical_predecessor_id is None
    assert target.eligible_at_ps == 20
    assert target.completed_at_ps == 30
    assert target.breakdown.operation_latency_ps == 30
    assert report.realized_critical_path_operation_ids == ("target",)
    assert result.completed_at_ps == 30
    assert result.quiesced_at_ps == 20_000_000

    record = StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[ScheduledRequest("request", RequestPhase.PREFILL, 1)],
        num_sampled=1,
        sampled_request_ids=["request"],
    )
    clock = VirtualClock(0)
    reduced = CompletionReducer(clock).reduce(record, graph, result, report)
    assert reduced.step_latency_ps == 30
    assert reduced.completed_at_ps == 30
    assert reduced.request_metrics[0].completed_at_ps == 30
    assert reduced.request_metrics[0].latency_ps == 30
    assert clock.now_ps == 30


def test_sparse_collective_uses_completion_path_causal_witness():
    graph = ExecutionGraph(
        execution_id="sparse-collective-causal-witness",
        step_index=0,
        released_at_ps=0,
        operations=(
            ExecutionOperation(
                "uncovered-rank-readiness",
                0,
                "compute",
                ComputeWork("uncovered-rank-readiness", nominal_duration_ps=1_000),
            ),
            ExecutionOperation(
                "collective",
                0,
                "collective",
                CollectiveWork(
                    "all-to-allv",
                    (0, 1, 2),
                    0,
                    "pairwise",
                    pair_payload_bytes=((1, 2, 1_000),),
                ),
                participant_local_depends_on=("uncovered-rank-readiness",),
            ),
        ),
        completion_operation_ids=("collective",),
    )
    runtime = CoarseDeviceRuntime(CoarseDeviceProfile())

    result = runtime.execute(graph)

    assert runtime.last_report is not None
    report = runtime.last_report
    collective = next(
        operation
        for operation in report.operations
        if operation.operation_id == "collective"
    )
    assert dict(collective.participant_completed_at_ps) == {
        0: 1_000,
        1: 8_889,
        2: 8_889,
    }
    assert collective.completed_at_ps == 8_889
    assert collective.causal_predecessor_id is None
    assert collective.causal_predecessor_completed_at_ps is None
    assert collective.critical_predecessor_id is None
    assert collective.breakdown.operation_latency_ps == 8_889
    assert collective.attribution.collective_ps == 8_889
    assert report.realized_critical_path_operation_ids == ("collective",)
    assert result.completed_at_ps == 8_889


def test_tied_collective_paths_keep_witness_and_attribution_together():
    graph = ExecutionGraph(
        execution_id="tied-collective-paths",
        step_index=0,
        released_at_ps=0,
        operations=(
            ExecutionOperation(
                "local-predecessor",
                0,
                "compute",
                ComputeWork("local-predecessor", nominal_duration_ps=1_111),
            ),
            ExecutionOperation(
                "collective",
                0,
                "collective",
                CollectiveWork(
                    "all-to-allv",
                    (0, 1, 2, 8),
                    0,
                    "pairwise",
                    pair_payload_bytes=(
                        (0, 1, 1_000),
                        (2, 8, 500),
                    ),
                ),
                participant_local_depends_on=("local-predecessor",),
            ),
        ),
        completion_operation_ids=("collective",),
    )
    runtime = CoarseDeviceRuntime(CoarseDeviceProfile())

    result = runtime.execute(graph)

    assert runtime.last_report is not None
    report = runtime.last_report
    collective = next(
        operation
        for operation in report.operations
        if operation.operation_id == "collective"
    )
    assert collective.completed_at_ps == 10_000
    assert collective.causal_predecessor_id is None
    assert collective.causal_predecessor_completed_at_ps is None
    assert collective.critical_predecessor_id is None
    assert collective.attribution.collective_ps == 0
    assert collective.attribution.nic_ps == 10_000
    assert report.realized_critical_path_operation_ids == ("collective",)
    assert result.completed_at_ps == 10_000


def test_runtime_uses_canonical_control_participants_for_local_readiness():
    graph = ExecutionGraph(
        execution_id="control-participant-readiness",
        step_index=0,
        released_at_ps=0,
        operations=(
            ExecutionOperation(
                "rank-1-compute",
                1,
                "compute",
                ComputeWork("rank-1-compute", nominal_duration_ps=5_000),
            ),
            ExecutionOperation(
                "control",
                0,
                "control",
                ControlWork("notify", (1,), 1, ControlMode.SYNCHRONOUS),
                participant_local_depends_on=("rank-1-compute",),
            ),
        ),
        completion_operation_ids=("control",),
    )
    runtime = CoarseDeviceRuntime(CoarseDeviceProfile())

    result = runtime.execute(graph)

    assert runtime.last_report is not None
    control = next(
        operation
        for operation in runtime.last_report.operations
        if operation.operation_id == "control"
    )
    assert control.causal_predecessor_id == "rank-1-compute"
    assert control.causal_predecessor_completed_at_ps == 5_000
    assert control.eligible_at_ps == 0
    assert result.completed_at_ps >= 5_000


def test_collective_transfer_waits_for_destination_local_predecessor():
    graph = ExecutionGraph(
        execution_id="collective-destination-readiness",
        step_index=0,
        released_at_ps=0,
        operations=(
            ExecutionOperation(
                "rank-1-compute",
                1,
                "compute",
                ComputeWork("rank-1-compute", nominal_duration_ps=1_000_000),
            ),
            ExecutionOperation(
                "collective",
                0,
                "collective",
                CollectiveWork(
                    "all-to-allv",
                    (0, 1),
                    0,
                    "pairwise",
                    pair_payload_bytes=((0, 1, 1),),
                ),
                participant_local_depends_on=("rank-1-compute",),
            ),
        ),
        completion_operation_ids=("collective",),
    )
    runtime = CoarseDeviceRuntime(CoarseDeviceProfile())

    runtime.execute(graph)

    assert runtime.last_report is not None
    transfers = tuple(
        visit
        for visit in runtime.last_report.visits
        if visit.operation_id == "collective" and visit.service_bytes == 1
    )
    assert len(transfers) == 1
    assert transfers[0].eligible_at_ps >= 1_000_000
    assert transfers[0].started_at_ps >= 1_000_000


def test_runtime_rejects_async_control_destination_local_dependency():
    graph = ExecutionGraph(
        "async-control-local",
        0,
        0,
        (
            ExecutionOperation(
                "rank-1-compute",
                1,
                "compute",
                ComputeWork("rank-1-compute", nominal_duration_ps=1_000),
            ),
            ExecutionOperation(
                "control",
                0,
                "control",
                ControlWork("notify", (1,), 1, ControlMode.ASYNCHRONOUS),
                participant_local_depends_on=("rank-1-compute",),
            ),
        ),
        ("control",),
    )

    with pytest.raises(ValueError, match="asynchronous control destination"):
        CoarseDeviceRuntime(CoarseDeviceProfile()).execute(graph)
