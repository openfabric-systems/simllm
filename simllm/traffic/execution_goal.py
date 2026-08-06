"""Render the serial execution-graph subset as a GOAL program."""

from __future__ import annotations

from collections.abc import Iterable

from simllm.core.execution import (
    CollectiveWork,
    ComputeWork,
    ExecutionGraph,
    ExecutionOperation,
)
from simllm.core.execution_io import validate_execution_graph
from simllm.goal import GoalTrace
from simllm.traffic.patterns import pairwise_all_to_allv, ring_allreduce


def _is_ring(work: CollectiveWork) -> bool:
    return work.collective == "all-reduce" and work.algorithm_hint == "ring"


def _is_pairwise(work: CollectiveWork) -> bool:
    return work.collective == "all-to-allv" and work.algorithm_hint == "pairwise"


def _participant_local_dependency_ids(
    operation: ExecutionOperation,
    queue_tail: dict[tuple[int, str], str],
    frontiers: dict[str, dict[int, str]],
    local_ancestors: dict[str, dict[int, set[str]]],
) -> tuple[str, ...]:
    dependencies = list(operation.participant_local_depends_on)
    predecessor = queue_tail.get((operation.rank, operation.logical_queue))
    if (
        predecessor is not None
        and predecessor not in operation.depends_on
        and predecessor not in dependencies
    ):
        dependencies.append(predecessor)
    target_ranks = (
        set(operation.work.ranks)
        if isinstance(operation.work, CollectiveWork)
        else {operation.rank}
    )
    reduced: list[str] = []
    for dependency in dependencies:
        relevant_ranks = target_ranks.intersection(frontiers.get(dependency, {}))
        redundant = bool(relevant_ranks) and all(
            any(
                other != dependency
                and dependency in local_ancestors.get(other, {}).get(rank, set())
                for other in dependencies
            )
            for rank in relevant_ranks
        )
        if not redundant:
            reduced.append(dependency)
    return tuple(reduced)


def _after_by_rank(
    operation: ExecutionOperation,
    ranks: Iterable[int],
    operation_dependency_ids: tuple[str, ...],
    participant_local_dependency_ids: tuple[str, ...],
    frontiers: dict[str, dict[int, str]],
) -> dict[int, str]:
    target_ranks = tuple(ranks)
    labels: dict[int, str] = {}
    for dependency_id in operation_dependency_ids:
        try:
            dependency = frontiers[dependency_id]
        except KeyError as exc:
            raise ValueError(
                f"operation {operation.operation_id!r} depends on {dependency_id!r}, "
                "which has not been rendered"
            ) from exc
        if len(target_ranks) != 1 or set(dependency) != set(target_ranks):
            raise ValueError(
                f"operation {operation.operation_id!r} has an operation-scoped "
                f"dependency on {dependency_id!r}; the serial GOAL renderer "
                "cannot encode its cross-rank completion barrier"
            )
        rank = target_ranks[0]
        if rank in labels and labels[rank] != dependency[rank]:
            raise ValueError(
                f"operation {operation.operation_id!r} has multiple GOAL "
                f"predecessors on rank {rank}; the serial renderer supports one"
            )
        labels[rank] = dependency[rank]

    for dependency_id in participant_local_dependency_ids:
        try:
            dependency = frontiers[dependency_id]
        except KeyError as exc:
            raise ValueError(
                f"operation {operation.operation_id!r} depends on {dependency_id!r}, "
                "which has not been rendered"
            ) from exc

        matched = False
        for rank in target_ranks:
            label = dependency.get(rank)
            if label is None:
                continue
            matched = True
            previous = labels.get(rank)
            if previous is not None and previous != label:
                raise ValueError(
                    f"operation {operation.operation_id!r} has multiple GOAL "
                    f"predecessors on rank {rank}; the serial renderer supports one"
                )
            labels[rank] = label
        if not matched:
            raise ValueError(
                f"dependency {dependency_id!r} of {operation.operation_id!r} "
                "has no frontier on a participating rank"
            )
    return labels


def render_serial_execution_graph_goal(
    graph: ExecutionGraph,
    num_goal_ranks: int | None = None,
    base_tag: int = 1000,
) -> GoalTrace:
    """Render the validated serial compatibility subset of an execution graph.

    Supported work is per-rank ``ComputeWork``, ring all-reduce and pairwise
    all-to-allv. Participant-local edges preserve independently arriving
    collective ranks. Operation-scoped edges are accepted only when both
    operations have one identical rank; cross-rank completion barriers require
    the stateful resource runtime. Logical KV work, DMA, control work, timing
    gates and other collective algorithms are rejected instead of being
    silently dropped.

    Ring tag blocks are reserved for every layer before pairwise tags are
    assigned.  This matches ``render_step_goal`` even though graph submission
    order interleaves TP and EP operations layer by layer.
    """

    validate_execution_graph(graph)
    if base_tag < 0:
        raise ValueError("base_tag must be non-negative")

    used_ranks: set[int] = set()
    ring_tags: dict[str, int] = {}
    pairwise_operations: list[ExecutionOperation] = []
    next_ring_tag = base_tag

    for operation in graph.operations:
        if operation.not_before_ps != 0:
            raise ValueError(
                f"operation {operation.operation_id!r} has a nonzero timing gate"
            )
        if operation.priority != 0:
            raise ValueError(
                f"operation {operation.operation_id!r} has a nonzero priority"
            )
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
            ring_tags[operation.operation_id] = next_ring_tag
            next_ring_tag += 2 * (len(work.ranks) - 1)
        elif _is_pairwise(work):
            if work.payload_bytes == 0:
                raise ValueError(
                    f"operation {operation.operation_id!r} is a zero-payload "
                    "pairwise all-to-allv; the serial GOAL renderer rejects it "
                    "instead of silently dropping the collective"
                )
            if len(work.ranks) < 2:
                raise ValueError(
                    f"operation {operation.operation_id!r} is a single-rank "
                    "pairwise all-to-allv; the serial GOAL renderer rejects it "
                    "instead of silently dropping the collective"
                )
            pairwise_operations.append(operation)
        else:
            raise ValueError(
                f"operation {operation.operation_id!r} uses unsupported collective "
                f"{work.collective!r} with algorithm {work.algorithm_hint!r}"
            )

    pairwise_tags = {
        operation.operation_id: next_ring_tag + index
        for index, operation in enumerate(pairwise_operations)
    }

    minimum_ranks = max(used_ranks, default=0) + 1
    if num_goal_ranks is None:
        num_goal_ranks = minimum_ranks
    if num_goal_ranks < minimum_ranks:
        raise ValueError(
            f"num_goal_ranks={num_goal_ranks} cannot contain rank {minimum_ranks - 1}"
        )
    trace = GoalTrace(num_goal_ranks)

    frontiers: dict[str, dict[int, str]] = {}
    queue_tail: dict[tuple[int, str], str] = {}
    local_ancestors: dict[str, dict[int, set[str]]] = {}
    for operation in graph.operations:
        work = operation.work
        local_dependencies = _participant_local_dependency_ids(
            operation,
            queue_tail,
            frontiers,
            local_ancestors,
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
            calc = trace.rank(operation.rank).calc(work.nominal_duration_ps // 1000)
            after = _after_by_rank(
                operation,
                (operation.rank,),
                operation.depends_on,
                local_dependencies,
                frontiers,
            )
            if operation.rank in after:
                trace.rank(operation.rank).requires(calc, after[operation.rank])
            frontier = {operation.rank: calc}
        elif isinstance(work, CollectiveWork) and _is_ring(work):
            after = _after_by_rank(
                operation,
                work.ranks,
                operation.depends_on,
                local_dependencies,
                frontiers,
            )
            frontier = ring_allreduce(
                trace,
                ranks=list(work.ranks),
                size_bytes=work.payload_bytes,
                base_tag=ring_tags[operation.operation_id],
                after=after,
            )
        elif isinstance(work, CollectiveWork) and _is_pairwise(work):
            after = _after_by_rank(
                operation,
                work.ranks,
                operation.depends_on,
                local_dependencies,
                frontiers,
            )
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
                tag=pairwise_tags[operation.operation_id],
                after=after,
            )
        else:
            raise AssertionError("unsupported work passed the renderer preflight")

        frontiers[operation.operation_id] = frontier
        local_ancestors[operation.operation_id] = {}
        for rank in frontier:
            rank_ancestors: set[str] = set()
            for dependency in (*operation.depends_on, *local_dependencies):
                if rank not in frontiers[dependency]:
                    continue
                rank_ancestors.add(dependency)
                rank_ancestors.update(
                    local_ancestors.get(dependency, {}).get(rank, set())
                )
            local_ancestors[operation.operation_id][rank] = rank_ancestors
        queue_tail[(operation.rank, operation.logical_queue)] = operation.operation_id

    for rank in range(num_goal_ranks):
        if rank not in used_ranks:
            trace.rank(rank).calc(0)
    return trace
