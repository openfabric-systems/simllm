from dataclasses import replace

import pytest

from simllm.core import CollectiveWork, ComputeWork, ExecutionGraph, ExecutionOperation
from simllm.goal import GoalDependencyKind
from simllm.traffic import (
    ExecutionGoalArtifact,
    project_execution_graph_goal,
    render_serial_execution_graph_goal,
    verify_execution_goal_projection,
)


def _ring(operation_id: str, logical_queue: str = "nccl") -> ExecutionOperation:
    return ExecutionOperation(
        operation_id,
        0,
        logical_queue,
        CollectiveWork("all-reduce", (0, 1), 8, "ring"),
    )


def test_rank_local_direct_render_preserves_goal_bytes_and_projection_serializes():
    first = ExecutionOperation(
        "first",
        0,
        "compute",
        ComputeWork("first", nominal_duration_ps=1_000),
    )
    second = ExecutionOperation(
        "second",
        0,
        "other-compute",
        ComputeWork("second", nominal_duration_ps=2_000),
        depends_on=("first",),
    )
    graph = ExecutionGraph("local", 0, 0, (first, second), ("second",))

    direct = render_serial_execution_graph_goal(graph)
    projection = project_execution_graph_goal(graph)

    assert [artifact.operation_ids for artifact in projection.artifacts] == [
        ("first",),
        ("second",),
    ]
    assert not projection.boundaries
    assert len(projection.serialized_edges) == 1
    assert projection.serialized_edges[0].scope == "whole-operation"
    assert not any(
        dependency.provenance is not None
        and dependency.provenance.kind is GoalDependencyKind.EXECUTION_GRAPH
        for artifact in projection.artifacts
        for dependency in artifact.trace.dependencies
    )
    assert direct.render() == (
        "num_ranks 1\n"
        "rank 0 {\n"
        "r0op0: calc 1\n"
        "r0op1: calc 2\n"
        "r0op1 requires r0op0\n"
        "}\n"
    )
    assert len(direct.dependencies) == 1
    provenance = direct.dependencies[0].provenance
    assert provenance is not None
    assert provenance.kind is GoalDependencyKind.EXECUTION_GRAPH
    assert {edge.predecessor_id for edge in provenance.graph_edges} == {"first"}


def test_distributed_fifo_becomes_checked_ordered_artifact_boundary():
    graph = ExecutionGraph(
        "distributed",
        0,
        0,
        (_ring("first"), _ring("second")),
        ("second",),
    )

    projection = project_execution_graph_goal(graph)
    assert [artifact.operation_ids for artifact in projection.artifacts] == [
        ("first",),
        ("second",),
    ]
    assert len(projection.boundaries) == 1
    edge = projection.boundaries[0].edge
    assert (edge.predecessor_id, edge.operation_id) == ("first", "second")
    assert edge.scope == "whole-operation"
    assert edge.origin == "logical-queue-fifo"
    assert all(
        dependency.provenance is not None
        for artifact in projection.artifacts
        for dependency in artifact.trace.dependencies
    )


def test_direct_renderer_rejects_implicit_distributed_fifo():
    graph = ExecutionGraph(
        "distributed-direct",
        0,
        0,
        (_ring("first"), _ring("second")),
        ("second",),
    )

    with pytest.raises(ValueError, match="distributed whole-operation edge"):
        render_serial_execution_graph_goal(graph)


def test_participant_local_edge_retains_scope_across_causal_level_artifacts():
    ring = _ring("ring")
    compute = ExecutionOperation(
        "compute",
        0,
        "compute",
        ComputeWork("compute", nominal_duration_ps=1_000),
        participant_local_depends_on=("ring",),
    )
    graph = ExecutionGraph("participant-local", 0, 0, (ring, compute))

    projection = project_execution_graph_goal(graph)

    assert [artifact.operation_ids for artifact in projection.artifacts] == [
        ("ring",),
        ("compute",),
    ]
    assert not projection.boundaries
    assert len(projection.serialized_edges) == 1
    assert projection.serialized_edges[0].scope == "participant-local"
    assert projection.serialized_edges[0].participant_rank == 0
    graph_dependencies = [
        dependency
        for dependency in projection.artifacts[0].trace.dependencies
        if dependency.provenance is not None
        and dependency.provenance.kind is GoalDependencyKind.EXECUTION_GRAPH
    ]
    assert not graph_dependencies


def test_only_distributed_whole_edge_forms_boundary_in_causal_stage():
    first = _ring("first")
    compute0 = ExecutionOperation(
        "compute-0",
        0,
        "compute-0",
        ComputeWork("compute-0", nominal_duration_ps=1_000),
        participant_local_depends_on=("first",),
    )
    compute1 = ExecutionOperation(
        "compute-1",
        1,
        "compute-1",
        ComputeWork("compute-1", nominal_duration_ps=1_000),
        participant_local_depends_on=("first",),
    )
    second = replace(
        _ring("second"),
        participant_local_depends_on=("compute-0", "compute-1"),
    )
    graph = ExecutionGraph(
        "causal-stages",
        0,
        0,
        (first, compute0, compute1, second),
        ("second",),
    )

    projection = project_execution_graph_goal(graph)

    assert [artifact.operation_ids for artifact in projection.artifacts] == [
        ("first",),
        ("compute-0", "compute-1"),
        ("second",),
    ]
    assert len(projection.boundaries) == 1
    assert projection.boundaries[0].edge.scope == "whole-operation"
    assert projection.boundaries[0].edge.origin == "logical-queue-fifo"
    assert projection.serialized_edges
    assert all(
        edge.scope != "participant-local"
        for boundary in projection.boundaries
        for edge in (boundary.edge,)
    )
    assert {edge.scope for edge in projection.serialized_edges} == {
        "participant-local"
    }


def test_projection_checker_rejects_one_perturbed_edge():
    graph = ExecutionGraph(
        "negative-control",
        0,
        0,
        (_ring("first"), _ring("second")),
        ("second",),
    )
    projection = project_execution_graph_goal(graph)
    boundary = projection.boundaries[0]
    perturbed = replace(
        projection,
        boundaries=(
            replace(
                boundary,
                edge=replace(boundary.edge, predecessor_id="not-in-the-graph"),
            ),
        ),
    )

    with pytest.raises(ValueError, match="edge absent from the graph"):
        verify_execution_goal_projection(graph, perturbed)


def test_projection_rejects_completion_subset_before_rendering():
    graph = ExecutionGraph(
        "early-completion",
        0,
        0,
        (
            ExecutionOperation(
                "done",
                0,
                "done",
                ComputeWork("done", nominal_duration_ps=1_000),
            ),
            ExecutionOperation(
                "background",
                1,
                "background",
                ComputeWork("background", nominal_duration_ps=10_000),
            ),
        ),
        ("done",),
    )

    with pytest.raises(ValueError, match="early completion subset"):
        project_execution_graph_goal(graph)


def test_projection_rejects_partial_participant_completion_boundary():
    collective = _ring("collective")
    target = ExecutionOperation(
        "target",
        0,
        "compute",
        ComputeWork("target", nominal_duration_ps=1_000),
        participant_local_depends_on=("collective",),
    )
    graph = ExecutionGraph(
        "partial-completion",
        0,
        0,
        (collective, target),
        ("target",),
    )

    with pytest.raises(ValueError, match="early completion subset"):
        project_execution_graph_goal(graph)


def test_projection_checker_rejects_duplicate_boundary():
    graph = ExecutionGraph(
        "duplicate-boundary",
        0,
        0,
        (_ring("first"), _ring("second")),
        ("second",),
    )
    projection = project_execution_graph_goal(graph)
    perturbed = replace(
        projection,
        boundaries=projection.boundaries + (projection.boundaries[0],),
    )

    with pytest.raises(ValueError, match="edge mismatch"):
        verify_execution_goal_projection(graph, perturbed)


def test_projection_checker_rejects_invented_independent_serialization():
    first = ExecutionOperation(
        "first",
        0,
        "first",
        ComputeWork("first", nominal_duration_ps=1_000),
    )
    second = ExecutionOperation(
        "second",
        1,
        "second",
        ComputeWork("second", nominal_duration_ps=10_000),
    )
    graph = ExecutionGraph("independent", 0, 0, (first, second))
    projection = project_execution_graph_goal(graph)
    perturbed = replace(
        projection,
        artifacts=(
            ExecutionGoalArtifact(
                ("first",),
                render_serial_execution_graph_goal(
                    ExecutionGraph("independent", 0, 0, (first,)),
                    num_goal_ranks=2,
                ),
            ),
            ExecutionGoalArtifact(
                ("second",),
                render_serial_execution_graph_goal(
                    ExecutionGraph("independent", 0, 0, (second,)),
                    num_goal_ranks=2,
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="does not depend on"):
        verify_execution_goal_projection(graph, perturbed)


def test_projection_uses_global_tags_when_pairwise_precedes_ring():
    graph = ExecutionGraph(
        "tag-order",
        0,
        0,
        (
            ExecutionOperation(
                "pairwise",
                0,
                "pairwise",
                CollectiveWork("all-to-allv", (0, 1), 8, "pairwise"),
            ),
            _ring("ring", "ring"),
        ),
    )

    projection = project_execution_graph_goal(graph)
    messages = projection.artifacts[0].trace.messages

    assert {message.tag for message in messages if message.operation_id == "pairwise"} == {
        1002
    }
    assert {message.tag for message in messages if message.operation_id == "ring"} == {
        1000,
        1001,
    }


def test_projection_checker_rejects_message_payload_mutation():
    graph = ExecutionGraph(
        "message-mutation",
        0,
        0,
        (_ring("ring"),),
        ("ring",),
    )
    projection = project_execution_graph_goal(graph)
    trace = projection.artifacts[0].trace
    trace._messages[0] = replace(
        trace._messages[0],
        payload_bytes=trace._messages[0].payload_bytes + 1,
    )

    with pytest.raises(ValueError, match="canonical graph projection"):
        verify_execution_goal_projection(graph, projection)


def test_projection_checker_rejects_noncanonical_in_goal_graph_edge():
    first = ExecutionOperation(
        "first",
        0,
        "first",
        ComputeWork("first", nominal_duration_ps=1_000),
    )
    second = ExecutionOperation(
        "second",
        0,
        "second",
        ComputeWork("second", nominal_duration_ps=1_000),
        depends_on=("first",),
    )
    graph = ExecutionGraph("provenance-mutation", 0, 0, (first, second), ("second",))
    projection = project_execution_graph_goal(graph)
    direct = render_serial_execution_graph_goal(graph)
    perturbed = replace(
        projection,
        artifacts=(ExecutionGoalArtifact(("first", "second"), direct),),
        serialized_edges=(),
    )

    with pytest.raises(ValueError, match="canonical causal-level partition"):
        verify_execution_goal_projection(graph, perturbed)


def test_projection_checker_counts_duplicate_graph_edge_occurrences():
    first = ExecutionOperation(
        "first",
        0,
        "first",
        ComputeWork("first", nominal_duration_ps=1_000),
    )
    second = ExecutionOperation(
        "second",
        0,
        "second",
        ComputeWork("second", nominal_duration_ps=1_000),
        depends_on=("first",),
    )
    graph = ExecutionGraph("duplicate-occurrence", 0, 0, (first, second), ("second",))
    projection = project_execution_graph_goal(graph)
    perturbed = replace(
        projection,
        serialized_edges=projection.serialized_edges
        + (projection.serialized_edges[0],),
    )

    with pytest.raises(ValueError, match="edge mismatch.*extra="):
        verify_execution_goal_projection(graph, perturbed)
