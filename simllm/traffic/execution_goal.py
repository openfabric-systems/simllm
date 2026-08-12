"""Render the serial execution-graph subset as a GOAL program."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import pairwise

from simllm.core.execution import (
    CollectiveWork,
    ComputeWork,
    DependencyScope,
    EffectiveDependencyEdge,
    ExecutionGraph,
    ExecutionOperation,
)
from simllm.core.execution_io import (
    effective_dependency_edges,
    operation_participant_ranks,
    validate_execution_graph,
)
from simllm.core.runtime import collective_goal_tags
from simllm.goal import (
    GoalDependencyKind,
    GoalDependencyProvenance,
    GoalGraphEdge,
    GoalTrace,
)
from simllm.traffic.collective_plan import (
    collective_plan_by_operation,
    render_collective_plan,
)
from simllm.traffic.patterns import pairwise_all_to_allv, ring_allreduce
from simllm.traffic.request_fidelity import compare_goal_request_attribution


def _is_ring(work: CollectiveWork) -> bool:
    return work.collective == "all-reduce" and work.algorithm_hint == "ring"


def _is_pairwise(work: CollectiveWork) -> bool:
    return work.collective == "all-to-allv" and work.algorithm_hint == "pairwise"


def _request_partitions(
    work: CollectiveWork,
) -> dict[tuple[int, int], tuple[tuple[str, int], ...]]:
    by_pair: dict[tuple[int, int], list[tuple[str, int]]] = {}
    for request_id, source, destination, size in work.request_pair_payload_bytes:
        by_pair.setdefault((source, destination), []).append((request_id, size))
    return {pair: tuple(sorted(entries)) for pair, entries in by_pair.items()}


def _operation_ranks(operation: ExecutionOperation) -> set[int]:
    return set(operation_participant_ranks(operation))


def _goal_edge(edge: EffectiveDependencyEdge) -> GoalGraphEdge:
    return GoalGraphEdge(
        predecessor_id=edge.predecessor_id,
        operation_id=edge.operation_id,
        scope=edge.scope.value,
        origin=edge.origin.value,
        participant_rank=edge.participant_rank,
    )


def _requires_artifact_boundary(
    edge: EffectiveDependencyEdge,
    operation_by_id: Mapping[str, ExecutionOperation],
) -> bool:
    predecessor = operation_by_id[edge.predecessor_id]
    operation = operation_by_id[edge.operation_id]
    predecessor_ranks = _operation_ranks(predecessor)
    operation_ranks = _operation_ranks(operation)
    return edge.scope is DependencyScope.WHOLE_OPERATION and (
        len(operation_ranks) != 1 or predecessor_ranks != operation_ranks
    )


def _entry_dependencies_by_rank(
    operation: ExecutionOperation,
    frontiers: dict[str, dict[int, str]],
    edges: tuple[EffectiveDependencyEdge, ...],
) -> dict[int, tuple[tuple[str, GoalDependencyProvenance], ...]]:
    """Return each exact graph edge at its target-rank entry point."""

    target_ranks = _operation_ranks(operation)
    result: dict[int, list[tuple[str, GoalDependencyProvenance]]] = {}
    for edge in edges:
        if edge.operation_id != operation.operation_id:
            continue
        if edge.scope is DependencyScope.WHOLE_OPERATION:
            if len(target_ranks) != 1:
                raise ValueError(
                    f"operation {operation.operation_id!r} has a distributed "
                    "whole-operation edge inside one GOAL artifact"
                )
            rank = next(iter(target_ranks))
        else:
            rank = edge.participant_rank
            if rank is None:
                raise AssertionError("participant-local edge has no participant rank")
        try:
            predecessor_label = frontiers[edge.predecessor_id][rank]
        except KeyError as exc:
            raise ValueError(
                f"operation {operation.operation_id!r} has no rendered rank-{rank} "
                f"frontier for edge from {edge.predecessor_id!r}"
            ) from exc
        result.setdefault(rank, []).append(
            (
                predecessor_label,
                GoalDependencyProvenance(
                    GoalDependencyKind.EXECUTION_GRAPH,
                    operation.operation_id,
                    (_goal_edge(edge),),
                ),
            )
        )
    return {rank: tuple(entries) for rank, entries in result.items()}


def _collective_entry_gates(
    trace: GoalTrace,
    operation: ExecutionOperation,
    dependencies_by_rank: dict[
        int,
        tuple[tuple[str, GoalDependencyProvenance], ...],
    ],
) -> tuple[
    dict[int, str],
    dict[int, GoalDependencyProvenance],
]:
    """Render one rank-local gate before a collective's physical fanout."""

    after: dict[int, str] = {}
    internal_by_rank: dict[int, GoalDependencyProvenance] = {}
    for rank, dependencies in dependencies_by_rank.items():
        gate = trace.rank(rank).calc(0, operation_id=operation.operation_id)
        for predecessor_label, provenance in dependencies:
            trace.rank(rank).requires(
                gate,
                predecessor_label,
                provenance=provenance,
            )
        after[rank] = gate
        internal_by_rank[rank] = GoalDependencyProvenance(
            GoalDependencyKind.COLLECTIVE_INTERNAL,
            operation.operation_id,
        )
    return after, internal_by_rank


def render_serial_execution_graph_goal(
    graph: ExecutionGraph,
    num_goal_ranks: int | None = None,
    base_tag: int = 1000,
    *,
    collective_tags: Mapping[str, tuple[int, ...]] | None = None,
) -> GoalTrace:
    """Render the validated serial compatibility subset of an execution graph.

    Supported work is per-rank ``ComputeWork``, ring all-reduce and pairwise
    all-to-allv. Participant-local edges preserve independently arriving
    collective ranks. Operation-scoped edges are accepted only when both
    operations have one identical rank; cross-rank completion barriers require
    the stateful resource runtime. Logical KV work, DMA, control work, timing
    gates and other collective algorithms are rejected instead of being
    silently dropped. Sparse pair tables, including an empty semantic
    all-to-allv, retain exact zero-work rank frontiers when a later dependency
    needs them.

    Ring tag blocks are reserved for every layer before pairwise tags are
    assigned.  This matches ``render_step_goal`` even though graph submission
    order interleaves TP and EP operations layer by layer.
    """

    validate_execution_graph(graph)
    if base_tag < 0:
        raise ValueError("base_tag must be non-negative")
    graph_edges = effective_dependency_edges(graph)
    operation_by_id = {
        operation.operation_id: operation for operation in graph.operations
    }
    unsupported_edges = tuple(
        edge
        for edge in graph_edges
        if _requires_artifact_boundary(edge, operation_by_id)
    )
    if unsupported_edges:
        first = unsupported_edges[0]
        raise ValueError(
            "the serial GOAL renderer cannot encode distributed whole-operation "
            f"edge {first.predecessor_id!r} to {first.operation_id!r}; use the "
            "ordered execution-graph projection"
        )
    needed_frontiers = {edge.predecessor_id for edge in graph_edges}
    plan_by_operation = collective_plan_by_operation(graph)

    used_ranks: set[int] = set()
    expected_tags = collective_goal_tags(graph, base_tag=base_tag)
    if collective_tags is None:
        selected_tags = expected_tags
    else:
        selected_tags = dict(collective_tags)
        if set(selected_tags) != set(expected_tags):
            raise ValueError("collective tag map does not cover the graph exactly")
        for operation_id, tags in selected_tags.items():
            expected_count = len(expected_tags[operation_id])
            if not isinstance(tags, tuple) or len(tags) != expected_count:
                raise ValueError(
                    f"collective tag map for {operation_id!r} has the wrong shape"
                )
            if any(type(tag) is not int or tag < 0 for tag in tags):
                raise ValueError("collective tags must be nonnegative integers")
        flattened = tuple(tag for tags in selected_tags.values() for tag in tags)
        if len(flattened) != len(set(flattened)):
            raise ValueError("collective tag map contains duplicate tags")

    for operation in graph.operations:
        if operation.not_before_ps != 0:
            raise ValueError(f"operation {operation.operation_id!r} has a nonzero timing gate")
        if operation.priority != 0:
            raise ValueError(f"operation {operation.operation_id!r} has a nonzero priority")
        work = operation.work
        if isinstance(work, ComputeWork):
            used_ranks.add(operation.rank)
            continue
        if not isinstance(work, CollectiveWork):
            raise TypeError(
                f"operation {operation.operation_id!r} carries unsupported "
                f"{type(work).__name__} in the serial GOAL renderer"
            )
        used_ranks.update(work.ranks)
        if _is_ring(work):
            pass
        elif _is_pairwise(work):
            if len(work.ranks) < 2:
                raise ValueError(
                    f"operation {operation.operation_id!r} is a single-rank "
                    "pairwise all-to-allv; the serial GOAL renderer rejects it "
                    "instead of silently dropping the collective"
                )
        else:
            raise ValueError(
                f"operation {operation.operation_id!r} uses unsupported collective "
                f"{work.collective!r} with algorithm {work.algorithm_hint!r}"
            )

    minimum_ranks = max(used_ranks, default=0) + 1
    if num_goal_ranks is None:
        num_goal_ranks = minimum_ranks
    if num_goal_ranks < minimum_ranks:
        raise ValueError(f"num_goal_ranks={num_goal_ranks} cannot contain rank {minimum_ranks - 1}")
    trace = GoalTrace(num_goal_ranks)

    frontiers: dict[str, dict[int, str]] = {}
    for operation in graph.operations:
        work = operation.work
        entry_dependencies = _entry_dependencies_by_rank(
            operation,
            frontiers,
            graph_edges,
        )
        if isinstance(work, ComputeWork):
            if work.nominal_duration_ps is None:
                raise ValueError(
                    f"compute operation {operation.operation_id!r} has no nominal duration"
                )
            if work.nominal_duration_ps % 1000:
                raise ValueError(
                    f"compute operation {operation.operation_id!r} duration is not "
                    "representable in whole GOAL nanoseconds"
                )
            calc = trace.rank(operation.rank).calc(
                work.nominal_duration_ps // 1000,
                operation_id=operation.operation_id,
            )
            for predecessor_label, provenance in entry_dependencies.get(
                operation.rank,
                (),
            ):
                trace.rank(operation.rank).requires(
                    calc,
                    predecessor_label,
                    provenance=provenance,
                )
            frontier = {operation.rank: calc}
        elif isinstance(work, CollectiveWork) and plan_by_operation:
            after, after_provenance = _collective_entry_gates(
                trace,
                operation,
                entry_dependencies,
            )
            plan = plan_by_operation[operation.operation_id]
            plan_tags = tuple(round_.tag for round_ in plan.rounds)
            if selected_tags[operation.operation_id] != plan_tags:
                raise ValueError(
                    f"collective tags for {operation.operation_id!r} disagree "
                    "with its immutable plan"
                )
            frontier = render_collective_plan(
                trace,
                after=after,
                plan=plan,
                after_provenance=after_provenance,
                exact_frontier=operation.operation_id in needed_frontiers,
            )
        elif isinstance(work, CollectiveWork) and _is_ring(work):
            after, after_provenance = _collective_entry_gates(
                trace,
                operation,
                entry_dependencies,
            )
            frontier = ring_allreduce(
                trace,
                ranks=list(work.ranks),
                size_bytes=work.payload_bytes,
                base_tag=selected_tags[operation.operation_id][0],
                after=after,
                operation_id=operation.operation_id,
                after_provenance=after_provenance,
                exact_frontier=operation.operation_id in needed_frontiers,
            )
        elif isinstance(work, CollectiveWork) and _is_pairwise(work):
            after, after_provenance = _collective_entry_gates(
                trace,
                operation,
                entry_dependencies,
            )
            if work.pair_payload_bytes:
                send_bytes = {
                    (source, destination): payload_bytes
                    for source, destination, payload_bytes in work.pair_payload_bytes
                }
            else:
                send_bytes = {
                    (source, destination): work.payload_bytes
                    for source in work.ranks
                    for destination in work.ranks
                    if source != destination
                }
            frontier = pairwise_all_to_allv(
                trace,
                ranks=list(work.ranks),
                send_bytes=send_bytes,
                tag=selected_tags[operation.operation_id][0],
                after=after,
                operation_id=operation.operation_id,
                after_provenance=after_provenance,
                exact_frontier=operation.operation_id in needed_frontiers,
                request_send_bytes=(
                    _request_partitions(work) if work.request_pair_payload_bytes else None
                ),
            )
        else:
            raise AssertionError("unsupported work passed the renderer preflight")

        frontiers[operation.operation_id] = frontier

    for rank in range(num_goal_ranks):
        if rank not in used_ranks:
            trace.rank(rank).calc(0)
    attributed_operations = [
        operation
        for operation in graph.operations
        if isinstance(operation.work, CollectiveWork) and operation.work.request_pair_payload_bytes
    ]
    if attributed_operations:
        expected_request_rows = tuple(
            (
                operation.operation_id,
                request_id,
                source,
                destination,
                size,
            )
            for operation in attributed_operations
            for request_id, source, destination, size in (operation.work.request_pair_payload_bytes)
        )
        expected_aggregate_rows = tuple(
            (operation.operation_id, source, destination, size)
            for operation in attributed_operations
            for source, destination, size in operation.work.pair_payload_bytes
        )
        compare_goal_request_attribution(
            expected_request_rows,
            expected_aggregate_rows,
            trace.messages,
        ).require_match()
    return trace


@dataclass(frozen=True)
class ExecutionGoalArtifact:
    """One GOAL artifact covering a contiguous graph-operation interval."""

    operation_ids: tuple[str, ...]
    trace: GoalTrace


@dataclass(frozen=True)
class ExecutionGoalArtifactBoundary:
    """One graph edge enforced by ordered execution of two GOAL artifacts."""

    predecessor_artifact_index: int
    operation_artifact_index: int
    edge: GoalGraphEdge


@dataclass(frozen=True)
class ExecutionGoalProjection:
    """A loss-checked ordered-artifact projection of one execution graph.

    ``serialized_edges`` inventories graph edges whose endpoints land in
    different causal-level artifacts. Their original scope remains explicit,
    but the process boundary can strengthen participant-local timing;
    TRAF-16 owns that precision gap.
    """

    execution_id: str
    num_goal_ranks: int
    base_tag: int
    artifacts: tuple[ExecutionGoalArtifact, ...]
    boundaries: tuple[ExecutionGoalArtifactBoundary, ...]
    serialized_edges: tuple[GoalGraphEdge, ...] = ()


def _artifact_subgraph(
    graph: ExecutionGraph,
    operations: tuple[ExecutionOperation, ...],
) -> ExecutionGraph:
    operation_ids = {operation.operation_id for operation in operations}
    return replace(
        graph,
        operations=tuple(
            replace(
                operation,
                depends_on=tuple(
                    dependency
                    for dependency in operation.depends_on
                    if dependency in operation_ids
                ),
                participant_local_depends_on=tuple(
                    dependency
                    for dependency in operation.participant_local_depends_on
                    if dependency in operation_ids
                ),
            )
            for operation in operations
        ),
        completion_operation_ids=tuple(
            operation_id
            for operation_id in graph.completion_operation_ids
            if operation_id in operation_ids
        ),
        collective_plans=tuple(
            plan
            for plan in graph.collective_plans
            if plan.operation_id in operation_ids
        ),
    )


def _validate_completion_boundary(
    graph: ExecutionGraph,
    edges: tuple[EffectiveDependencyEdge, ...],
) -> None:
    """Reject logical completion boundaries that GOAL quiescence would widen."""

    if not graph.completion_operation_ids:
        return
    operation_by_id = {
        operation.operation_id: operation for operation in graph.operations
    }
    incoming: dict[str, list[EffectiveDependencyEdge]] = {
        operation.operation_id: [] for operation in graph.operations
    }
    for edge in edges:
        incoming[edge.operation_id].append(edge)

    Fragment = tuple[str, int]
    ancestors: dict[Fragment, set[Fragment]] = {}
    for operation in graph.operations:
        operation_id = operation.operation_id
        target_ranks = operation_participant_ranks(operation)
        for rank in target_ranks:
            fragment = (operation_id, rank)
            fragment_ancestors: set[Fragment] = set()
            for edge in incoming[operation_id]:
                if (
                    edge.scope is DependencyScope.PARTICIPANT_LOCAL
                    and edge.participant_rank != rank
                ):
                    continue
                predecessor_ranks = operation_participant_ranks(
                    operation_by_id[edge.predecessor_id]
                )
                if edge.scope is DependencyScope.PARTICIPANT_LOCAL:
                    predecessor_ranks = (rank,)
                for predecessor_rank in predecessor_ranks:
                    predecessor = (edge.predecessor_id, predecessor_rank)
                    if predecessor not in ancestors:
                        raise ValueError(
                            "graph cannot be represented by ordered GOAL artifacts: "
                            "forward or non-monotone dependencies are unsupported"
                        )
                    fragment_ancestors.add(predecessor)
                    fragment_ancestors.update(ancestors[predecessor])
            ancestors[fragment] = fragment_ancestors

    completed_fragments = {
        (operation_id, rank)
        for operation_id in graph.completion_operation_ids
        for rank in operation_participant_ranks(operation_by_id[operation_id])
    }
    covered = set(completed_fragments)
    for fragment in completed_fragments:
        covered.update(ancestors[fragment])
    if covered != set(ancestors):
        raise ValueError(
            "graph completion boundary is not full terminal quiescence; "
            "the GOAL projection cannot represent an early completion subset"
        )


def _logical_completion_ancestors(
    graph: ExecutionGraph,
    edges: tuple[EffectiveDependencyEdge, ...],
) -> dict[str, set[str]]:
    """Return predecessors whose whole logical completion is dominated."""

    operation_by_id = {
        operation.operation_id: operation for operation in graph.operations
    }
    direct_predecessors: dict[str, set[str]] = {
        operation.operation_id: set() for operation in graph.operations
    }
    local_ranks: dict[tuple[str, str], set[int]] = {}
    for edge in edges:
        if edge.scope is DependencyScope.WHOLE_OPERATION:
            direct_predecessors[edge.operation_id].add(edge.predecessor_id)
        elif edge.participant_rank is not None:
            local_ranks.setdefault(
                (edge.predecessor_id, edge.operation_id),
                set(),
            ).add(edge.participant_rank)
    for (predecessor_id, operation_id), ranks in local_ranks.items():
        predecessor_ranks = set(
            operation_participant_ranks(operation_by_id[predecessor_id])
        )
        if predecessor_ranks.issubset(ranks):
            direct_predecessors[operation_id].add(predecessor_id)

    ancestors: dict[str, set[str]] = {}
    for operation in graph.operations:
        operation_ancestors: set[str] = set()
        for predecessor_id in direct_predecessors[operation.operation_id]:
            if predecessor_id not in ancestors:
                raise ValueError(
                    "graph cannot be represented by ordered GOAL artifacts: "
                    "forward or non-monotone dependencies are unsupported"
                )
            operation_ancestors.add(predecessor_id)
            operation_ancestors.update(ancestors[predecessor_id])
        ancestors[operation.operation_id] = operation_ancestors
    return ancestors


def _validate_artifact_ordering(
    graph: ExecutionGraph,
    operation_groups: tuple[tuple[ExecutionOperation, ...], ...],
    edges: tuple[EffectiveDependencyEdge, ...],
) -> None:
    """Require every ordered stage transition to carry a graph barrier."""

    expected_ids = tuple(operation.operation_id for operation in graph.operations)
    grouped_ids = tuple(
        operation.operation_id
        for group in operation_groups
        for operation in group
    )
    if grouped_ids != expected_ids:
        raise ValueError("projection artifacts do not partition graph operations in order")
    if not operation_groups:
        return

    operation_index = {
        operation.operation_id: index for index, operation in enumerate(graph.operations)
    }
    for edge in edges:
        if operation_index[edge.predecessor_id] >= operation_index[edge.operation_id]:
            raise ValueError(
                "graph cannot be represented by ordered GOAL artifacts: "
                "forward or non-monotone dependencies are unsupported"
            )
    direct_predecessors: dict[str, set[str]] = {
        operation.operation_id: set() for operation in graph.operations
    }
    for edge in edges:
        direct_predecessors[edge.operation_id].add(edge.predecessor_id)
    ancestors: dict[str, set[str]] = {}
    for operation in graph.operations:
        operation_ancestors: set[str] = set()
        for predecessor_id in direct_predecessors[operation.operation_id]:
            operation_ancestors.add(predecessor_id)
            operation_ancestors.update(ancestors[predecessor_id])
        ancestors[operation.operation_id] = operation_ancestors

    for predecessor_group, operation_group in pairwise(operation_groups):
        for predecessor in predecessor_group:
            for operation in operation_group:
                if predecessor.operation_id not in ancestors[operation.operation_id]:
                    raise ValueError(
                        "graph cannot be represented by ordered GOAL artifacts: "
                        f"{operation.operation_id!r} does not depend on "
                        f"{predecessor.operation_id!r}"
                    )


def _causal_level_operation_groups(
    graph: ExecutionGraph,
    edges: tuple[EffectiveDependencyEdge, ...],
) -> tuple[tuple[ExecutionOperation, ...], ...]:
    """Partition contiguous operations at every effective dependency level."""

    incoming_edges: dict[str, list[EffectiveDependencyEdge]] = {}
    for edge in edges:
        incoming_edges.setdefault(edge.operation_id, []).append(edge)

    stage_by_operation: dict[str, int] = {}
    previous_stage = 0
    for operation in graph.operations:
        stage = max(
            (
                stage_by_operation[edge.predecessor_id] + 1
                for edge in incoming_edges.get(operation.operation_id, ())
            ),
            default=previous_stage,
        )
        if stage < previous_stage:
            raise ValueError(
                "graph dependencies do not form monotone causal-level artifacts"
            )
        stage_by_operation[operation.operation_id] = stage
        previous_stage = stage

    groups: list[list[ExecutionOperation]] = []
    group_stages: list[int] = []
    for operation in graph.operations:
        stage = stage_by_operation[operation.operation_id]
        if not group_stages or group_stages[-1] != stage:
            groups.append([])
            group_stages.append(stage)
        groups[-1].append(operation)

    operation_groups = tuple(tuple(group) for group in groups)
    _validate_artifact_ordering(graph, operation_groups, edges)
    return operation_groups


def _trace_inventory(trace: GoalTrace) -> tuple[object, ...]:
    return (
        trace.num_ranks,
        trace.render(),
        trace.operations,
        trace.messages,
        trace.dependencies,
    )


def project_execution_graph_goal(
    graph: ExecutionGraph,
    num_goal_ranks: int | None = None,
    base_tag: int = 1000,
) -> ExecutionGoalProjection:
    """Project a graph into quiescent-backend causal-level artifacts.

    Each effective graph edge advances its target to a later artifact. This
    keeps inter-operation ordering out of GOAL, whose zero-duration gates are
    not timing neutral in the supported compiler. Truly unrepresentable
    distributed whole-operation edges are registered as required boundaries;
    all other cross-artifact edges retain their graph scope in the serialized
    inventory. The ancestry check rejects a partition that would serialize
    unrelated operations.
    """

    validate_execution_graph(graph)
    edges = effective_dependency_edges(graph)
    _validate_completion_boundary(graph, edges)
    operation_by_id = {
        operation.operation_id: operation for operation in graph.operations
    }
    operation_groups = _causal_level_operation_groups(graph, edges)

    global_collective_tags = collective_goal_tags(graph, base_tag=base_tag)
    minimum_goal_ranks = max(
        (
            rank
            for operation in graph.operations
            for rank in operation_participant_ranks(operation)
        ),
        default=0,
    ) + 1
    resolved_num_goal_ranks = (
        minimum_goal_ranks if num_goal_ranks is None else num_goal_ranks
    )
    if resolved_num_goal_ranks < minimum_goal_ranks:
        raise ValueError(
            f"num_goal_ranks={resolved_num_goal_ranks} cannot contain rank "
            f"{minimum_goal_ranks - 1}"
        )
    artifacts = tuple(
        ExecutionGoalArtifact(
            tuple(operation.operation_id for operation in operations),
            render_serial_execution_graph_goal(
                _artifact_subgraph(graph, operations),
                num_goal_ranks=resolved_num_goal_ranks,
                base_tag=base_tag,
                collective_tags={
                    operation.operation_id: global_collective_tags[
                        operation.operation_id
                    ]
                    for operation in operations
                    if isinstance(operation.work, CollectiveWork)
                },
            ),
        )
        for operations in operation_groups
    )
    artifact_by_operation = {
        operation_id: artifact_index
        for artifact_index, artifact in enumerate(artifacts)
        for operation_id in artifact.operation_ids
    }
    boundaries = tuple(
        ExecutionGoalArtifactBoundary(
            artifact_by_operation[edge.predecessor_id],
            artifact_by_operation[edge.operation_id],
            _goal_edge(edge),
        )
        for edge in edges
        if _requires_artifact_boundary(edge, operation_by_id)
    )
    serialized_edges = tuple(
        _goal_edge(edge)
        for edge in edges
        if artifact_by_operation[edge.predecessor_id]
        != artifact_by_operation[edge.operation_id]
        and not _requires_artifact_boundary(edge, operation_by_id)
    )
    projection = ExecutionGoalProjection(
        graph.execution_id,
        resolved_num_goal_ranks,
        base_tag,
        artifacts,
        boundaries,
        serialized_edges,
    )
    verify_execution_goal_projection(graph, projection)
    return projection


def _edge_tuple(edge: GoalGraphEdge) -> tuple[str, str, str, str, int | None]:
    return (
        edge.predecessor_id,
        edge.operation_id,
        edge.scope,
        edge.origin,
        edge.participant_rank,
    )


def verify_execution_goal_projection(
    graph: ExecutionGraph,
    projection: ExecutionGoalProjection,
) -> None:
    """Reject any lost, invented or unattributed GOAL ordering relation."""

    validate_execution_graph(graph)
    if not isinstance(projection, ExecutionGoalProjection):
        raise TypeError("projection: expected ExecutionGoalProjection")
    if projection.execution_id != graph.execution_id:
        raise ValueError("projection execution identity does not match the graph")
    if type(projection.num_goal_ranks) is not int or projection.num_goal_ranks < 1:
        raise ValueError("projection num_goal_ranks must be a positive integer")
    if type(projection.base_tag) is not int or projection.base_tag < 0:
        raise ValueError("projection base_tag must be a nonnegative integer")

    edges = effective_dependency_edges(graph)
    _validate_completion_boundary(graph, edges)
    operation_by_id = {
        operation.operation_id: operation for operation in graph.operations
    }
    operation_groups = []
    for artifact in projection.artifacts:
        if not isinstance(artifact, ExecutionGoalArtifact):
            raise TypeError("projection artifact has the wrong type")
        try:
            operation_groups.append(
                tuple(operation_by_id[operation_id] for operation_id in artifact.operation_ids)
            )
        except KeyError as exc:
            raise ValueError("projection artifact cites an unknown graph operation") from exc
    groups = tuple(operation_groups)
    _validate_artifact_ordering(graph, groups, edges)
    expected_groups = _causal_level_operation_groups(graph, edges)
    if groups != expected_groups:
        raise ValueError(
            "projection artifacts are not the canonical causal-level partition"
        )

    expected_edges = Counter(_edge_tuple(_goal_edge(edge)) for edge in edges)
    emitted_edges: Counter[tuple[str, str, str, str, int | None]] = Counter()
    boundary_edges: Counter[tuple[str, str, str, str, int | None]] = Counter()
    serialized_edges: Counter[tuple[str, str, str, str, int | None]] = Counter()
    artifact_by_operation = {
        operation_id: artifact_index
        for artifact_index, artifact in enumerate(projection.artifacts)
        for operation_id in artifact.operation_ids
    }
    global_collective_tags = collective_goal_tags(
        graph,
        base_tag=projection.base_tag,
    )
    noncanonical_artifacts = []

    for artifact_index, artifact in enumerate(projection.artifacts):
        operation_by_label = {
            operation.label: operation for operation in artifact.trace.operations
        }
        allowed_operation_ids = set(artifact.operation_ids)
        for operation in artifact.trace.operations:
            if operation.operation_id is None:
                if operation.text != "calc 0":
                    raise ValueError("GOAL operation has no execution-graph owner")
            elif operation.operation_id not in allowed_operation_ids:
                raise ValueError("GOAL operation owner is outside its artifact")
        for message in artifact.trace.messages:
            if message.operation_id not in allowed_operation_ids:
                raise ValueError("GOAL message owner is outside its artifact")
        for dependency in artifact.trace.dependencies:
            provenance = dependency.provenance
            if provenance is None:
                raise ValueError("GOAL dependency has no semantic provenance")
            operation = operation_by_label[dependency.operation_label]
            predecessor = operation_by_label[dependency.predecessor_label]
            if operation.operation_id != provenance.operation_id:
                raise ValueError("GOAL dependency target does not match its semantic owner")
            if provenance.kind is GoalDependencyKind.COLLECTIVE_INTERNAL:
                if predecessor.operation_id != provenance.operation_id:
                    raise ValueError("collective-internal dependency crosses semantic operations")
                continue
            if dependency.relation != "requires":
                raise ValueError("execution-graph ordering must use completion dependency")
            if len(provenance.graph_edges) != 1:
                raise ValueError(
                    "inter-operation GOAL dependency must cite exactly one graph edge"
                )
            edge = provenance.graph_edges[0]
            key = _edge_tuple(edge)
            if key not in expected_edges:
                raise ValueError("GOAL dependency cites an edge absent from the graph")
            if edge.operation_id != operation.operation_id:
                raise ValueError("GOAL dependency cites the wrong target operation")
            if predecessor.operation_id != edge.predecessor_id:
                raise ValueError("GOAL dependency predecessor does not match its graph edge")
            if (
                edge.participant_rank is not None
                and edge.participant_rank != dependency.rank
            ):
                raise ValueError("participant-local edge is rendered on the wrong rank")
            if artifact_by_operation[edge.predecessor_id] != artifact_index:
                raise ValueError("cross-artifact graph edge was emitted inside GOAL")
            emitted_edges[key] += 1

        canonical = render_serial_execution_graph_goal(
            _artifact_subgraph(graph, groups[artifact_index]),
            num_goal_ranks=projection.num_goal_ranks,
            base_tag=projection.base_tag,
            collective_tags={
                operation.operation_id: global_collective_tags[operation.operation_id]
                for operation in groups[artifact_index]
                if isinstance(operation.work, CollectiveWork)
            },
        )
        if _trace_inventory(artifact.trace) != _trace_inventory(canonical):
            noncanonical_artifacts.append(artifact_index)

    for boundary in projection.boundaries:
        if not isinstance(boundary, ExecutionGoalArtifactBoundary):
            raise TypeError("projection boundary has the wrong type")
        key = _edge_tuple(boundary.edge)
        if key not in expected_edges:
            raise ValueError("artifact boundary cites an edge absent from the graph")
        expected_predecessor_index = artifact_by_operation[boundary.edge.predecessor_id]
        expected_operation_index = artifact_by_operation[boundary.edge.operation_id]
        if (
            boundary.predecessor_artifact_index != expected_predecessor_index
            or boundary.operation_artifact_index != expected_operation_index
        ):
            raise ValueError("artifact boundary indexes do not match graph operations")
        if expected_predecessor_index >= expected_operation_index:
            raise ValueError("artifact boundary does not advance ordered execution")
        boundary_edges[key] += 1

    for edge in projection.serialized_edges:
        if not isinstance(edge, GoalGraphEdge):
            raise TypeError("serialized projection edge has the wrong type")
        key = _edge_tuple(edge)
        if key not in expected_edges:
            raise ValueError("serialized projection cites an edge absent from the graph")
        if artifact_by_operation[edge.predecessor_id] >= artifact_by_operation[
            edge.operation_id
        ]:
            raise ValueError("serialized projection edge does not advance artifact order")
        effective = next(
            candidate
            for candidate in edges
            if _edge_tuple(_goal_edge(candidate)) == key
        )
        if _requires_artifact_boundary(effective, operation_by_id):
            raise ValueError(
                "distributed whole-operation edge must form an artifact boundary"
            )
        serialized_edges[key] += 1

    overlap = (
        emitted_edges.keys() & boundary_edges.keys()
        or emitted_edges.keys() & serialized_edges.keys()
        or boundary_edges.keys() & serialized_edges.keys()
    )
    if overlap:
        raise ValueError("graph edge is enforced by multiple projection mechanisms")
    observed_edges = emitted_edges + boundary_edges + serialized_edges
    if observed_edges != expected_edges:
        missing = sorted((expected_edges - observed_edges).elements())
        extra = sorted((observed_edges - expected_edges).elements())
        raise ValueError(f"GOAL projection edge mismatch: missing={missing}, extra={extra}")
    if noncanonical_artifacts:
        raise ValueError(
            "GOAL artifacts differ from their canonical graph projection: "
            f"indexes={noncanonical_artifacts}"
        )

    expected_boundaries = tuple(
        ExecutionGoalArtifactBoundary(
            artifact_by_operation[edge.predecessor_id],
            artifact_by_operation[edge.operation_id],
            _goal_edge(edge),
        )
        for edge in edges
        if _requires_artifact_boundary(edge, operation_by_id)
    )
    if projection.boundaries != expected_boundaries:
        raise ValueError("artifact boundaries are not in canonical graph-edge order")
    expected_serialized_edges = tuple(
        _goal_edge(edge)
        for edge in edges
        if artifact_by_operation[edge.predecessor_id]
        != artifact_by_operation[edge.operation_id]
        and not _requires_artifact_boundary(edge, operation_by_id)
    )
    if projection.serialized_edges != expected_serialized_edges:
        raise ValueError("serialized edges are not in canonical graph-edge order")
