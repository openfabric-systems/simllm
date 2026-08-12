"""ExecutionGraph authority for placement-backed communication phases."""

from dataclasses import replace

import pytest

from simllm.core import (
    CollectiveWork,
    DmaWork,
    ExecutionGraph,
    ExecutionOperation,
    OperationCorrelation,
)
from simllm.placement import PlacementManifest, RankMapper, RankPlacement
from simllm.traffic.step_comm import (
    plan_execution_graph_locality,
    render_fabric_phase_goal,
    validate_execution_graph_locality_projection,
)


def _mapper() -> RankMapper:
    return RankMapper(
        PlacementManifest(
            ranks=[
                RankPlacement(0, "node-a", 0),
                RankPlacement(1, "node-a", 1),
                RankPlacement(2, "node-b", 0),
                RankPlacement(3, "node-b", 1),
            ]
        )
    )


def _collective_graph() -> ExecutionGraph:
    ranks = (0, 1, 2, 3)
    ring = ExecutionOperation(
        operation_id="ring",
        rank=0,
        logical_queue="cuda:0:nccl",
        work=CollectiveWork(
            "all-reduce",
            ranks,
            24,
            algorithm_hint="ring",
            channel_hint="attention",
        ),
        correlation=OperationCorrelation(layer=0),
    )
    pairwise = ExecutionOperation(
        operation_id="pairwise",
        rank=0,
        logical_queue="cuda:0:nccl",
        work=CollectiveWork(
            "all-to-allv",
            ranks,
            3,
            algorithm_hint="pairwise",
            channel_hint="dispatch",
        ),
        participant_local_depends_on=("ring",),
        correlation=OperationCorrelation(layer=0),
    )
    return ExecutionGraph(
        execution_id="projection",
        step_index=0,
        released_at_ps=0,
        operations=(ring, pairwise),
        completion_operation_ids=("pairwise",),
    )


def _request_partition_graph(*, swap_requests: bool = False) -> ExecutionGraph:
    if swap_requests:
        request_pairs = (
            ("request-a", 0, 2, 6),
            ("request-b", 0, 1, 4),
        )
    else:
        request_pairs = (
            ("request-a", 0, 1, 4),
            ("request-b", 0, 2, 6),
        )
    pairwise = ExecutionOperation(
        operation_id="attributed-pairwise",
        rank=0,
        logical_queue="cuda:0:nccl",
        work=CollectiveWork(
            "all-to-allv",
            (0, 1, 2),
            0,
            algorithm_hint="pairwise",
            channel_hint="dispatch",
            pair_payload_bytes=((0, 1, 4), (0, 2, 6)),
            request_pair_payload_bytes=request_pairs,
        ),
        correlation=OperationCorrelation(
            request_ids=("request-a", "request-b"),
            layer=0,
        ),
    )
    return ExecutionGraph(
        execution_id="request-partition",
        step_index=0,
        released_at_ps=0,
        operations=(pairwise,),
        completion_operation_ids=(pairwise.operation_id,),
    )


def test_graph_locality_preserves_operations_tags_edges_and_bytes():
    graph = _collective_graph()

    plan = plan_execution_graph_locality(
        graph,
        rank_mapper=_mapper(),
        base_tag=40,
    )

    assert plan.graph_execution_id == graph.execution_id
    assert [phase.phase.operation_id for phase in plan.phases] == [
        "ring",
        "ring",
        "ring",
        "ring",
        "ring",
        "ring",
        "pairwise",
    ]
    assert [
        tuple(segment.tag for segment in phase.phase.segments)
        for phase in plan.phases
    ] == [(40,) * 4, (41,) * 4, (42,) * 4, (43,) * 4, (44,) * 4, (45,) * 4, (46,) * 12]
    assert [
        (
            boundary.predecessor_id,
            boundary.operation_id,
            boundary.scope.value,
            boundary.origin.value,
            boundary.participant_rank,
        )
        for boundary in plan.dependency_edges
    ] == [
        ("ring", "pairwise", "participant-local", "explicit", 0),
        ("ring", "pairwise", "participant-local", "explicit", 1),
        ("ring", "pairwise", "participant-local", "explicit", 2),
        ("ring", "pairwise", "participant-local", "explicit", 3),
        ("ring", "pairwise", "whole-operation", "logical-queue-fifo", None),
    ]
    assert plan.fabric_bytes == 96
    assert plan.nvlink_bytes == 84
    assert plan.fabric_bytes + plan.nvlink_bytes == plan.total_directed_bytes == 180
    validate_execution_graph_locality_projection(
        graph,
        plan,
        rank_mapper=_mapper(),
        base_tag=40,
    )


def test_graph_locality_preserves_request_partitions_through_fabric_goal():
    graph = _request_partition_graph()
    plan = plan_execution_graph_locality(graph, rank_mapper=_mapper(), base_tag=70)
    phase = plan.phases[0]

    assert [
        (segment.source_rank, segment.destination_rank, segment.request_payload_bytes)
        for segment in phase.nvlink_segments
    ] == [(0, 1, (("request-a", 4),))]
    assert [
        (segment.source_rank, segment.destination_rank, segment.request_payload_bytes)
        for segment in phase.fabric_segments
    ] == [(0, 2, (("request-b", 6),))]

    trace = render_fabric_phase_goal(phase, rank_mapper=_mapper())
    assert [
        (
            message.operation_id,
            message.source_rank,
            message.destination_rank,
            message.payload_bytes,
            message.tag,
            message.request_payload_bytes,
        )
        for message in trace.messages
    ] == [
        (
            "attributed-pairwise",
            0,
            2,
            6,
            70,
            (("request-b", 6),),
        )
    ]


def test_projection_rejects_request_partition_only_mutation():
    graph = _request_partition_graph()
    plan = plan_execution_graph_locality(graph, rank_mapper=_mapper())

    with pytest.raises(ValueError, match="does not exactly project"):
        validate_execution_graph_locality_projection(
            _request_partition_graph(swap_requests=True),
            plan,
            rank_mapper=_mapper(),
        )


def test_projection_negative_control_rejects_one_removed_graph_edge():
    graph = _collective_graph()
    plan = plan_execution_graph_locality(graph, rank_mapper=None)
    pairwise = replace(graph.operations[1], participant_local_depends_on=())
    perturbed = replace(graph, operations=(graph.operations[0], pairwise))

    with pytest.raises(ValueError, match="does not exactly project"):
        validate_execution_graph_locality_projection(
            perturbed,
            plan,
            rank_mapper=None,
        )


def test_projection_rejects_mutated_operation_identity():
    graph = _collective_graph()
    plan = plan_execution_graph_locality(graph, rank_mapper=None)
    first = replace(
        plan.phases[0],
        phase=replace(plan.phases[0].phase, operation_id="wrong-operation"),
    )
    perturbed = replace(plan, phases=(first, *plan.phases[1:]))

    with pytest.raises(ValueError, match="does not exactly project"):
        validate_execution_graph_locality_projection(
            graph,
            perturbed,
            rank_mapper=None,
        )


def test_projection_rejects_unsupported_work_before_locality_classification():
    graph = ExecutionGraph(
        execution_id="unsupported",
        step_index=0,
        released_at_ps=0,
        operations=(
            ExecutionOperation(
                operation_id="dma",
                rank=0,
                logical_queue="cuda:0:copy",
                work=DmaWork("copy", "host", "cuda:0", 16),
            ),
        ),
    )

    with pytest.raises(TypeError, match="DmaWork.*locality projection"):
        plan_execution_graph_locality(graph, rank_mapper=_mapper())


def test_phase_renderer_uses_identity_mapping_when_placement_is_absent():
    plan = plan_execution_graph_locality(_collective_graph(), rank_mapper=None)

    rendered = render_fabric_phase_goal(
        plan.phases[-1],
        rank_mapper=None,
    ).render()

    assert rendered.startswith("num_ranks 4\n")
    assert rendered.count(": send 3b") == 12
    assert rendered.count(": recv 3b") == 12
