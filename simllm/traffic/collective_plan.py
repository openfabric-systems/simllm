"""Immutable traffic-owned expansion of execution-graph collectives."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace

from simllm.core.execution import (
    CollectivePlan,
    CollectivePlanAction,
    CollectivePlanActionKind,
    CollectivePlanExtent,
    CollectivePlanRound,
    CollectiveWork,
    ExecutionGraph,
)
from simllm.core.execution_io import (
    collective_plan_integrity_sha256,
    validate_execution_graph,
)
from simllm.core.runtime import DEFAULT_GOAL_BASE_TAG, collective_goal_tags
from simllm.goal import (
    GoalDependencyKind,
    GoalDependencyProvenance,
    GoalMessage,
    GoalTrace,
)

DEFAULT_COLLECTIVE_PLAN_BASE_TAG = DEFAULT_GOAL_BASE_TAG


def _action_id(operation_id: str, round_index: int, rank: int, role: str) -> str:
    return f"{operation_id}:plan-round-{round_index}:rank-{rank}:{role}"


def _extent_id(operation_id: str, round_index: int, extent_index: int) -> str:
    return f"{operation_id}:plan-round-{round_index}:extent-{extent_index}"


def _rank_action_roots(
    rank_order: tuple[int, ...],
    actions: tuple[CollectivePlanAction, ...],
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    return tuple(
        (
            rank,
            tuple(
                action.action_id
                for action in actions
                if action.rank == rank and not action.depends_on
            ),
        )
        for rank in rank_order
    )


def _rank_action_terminals(
    rank_order: tuple[int, ...],
    actions: tuple[CollectivePlanAction, ...],
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    predecessor_ids = {
        dependency for action in actions for dependency in action.depends_on
    }
    return tuple(
        (
            rank,
            tuple(
                action.action_id
                for action in actions
                if action.rank == rank and action.action_id not in predecessor_ids
            ),
        )
        for rank in rank_order
    )


def _finish_plan(
    *,
    operation_id: str,
    work: CollectiveWork,
    rounds: tuple[CollectivePlanRound, ...],
    actions: tuple[CollectivePlanAction, ...],
    extents: tuple[CollectivePlanExtent, ...],
) -> CollectivePlan:
    if work.algorithm_hint is None:
        raise ValueError(f"collective {operation_id!r} has no algorithm_hint")
    plan = CollectivePlan(
        operation_id=operation_id,
        collective=work.collective,
        algorithm=work.algorithm_hint,
        channel_id=work.channel_hint or "default",
        rank_order=work.ranks,
        payload_bytes=work.payload_bytes,
        pair_payload_bytes=work.pair_payload_bytes,
        request_pair_payload_bytes=work.request_pair_payload_bytes,
        rounds=rounds,
        actions=actions,
        extents=extents,
        entry_action_ids=_rank_action_roots(work.ranks, actions),
        terminal_action_ids=_rank_action_terminals(work.ranks, actions),
        integrity_sha256="0" * 64,
    )
    return replace(plan, integrity_sha256=collective_plan_integrity_sha256(plan))


def _ring_plan(
    operation_id: str,
    work: CollectiveWork,
    tags: tuple[int, ...],
) -> CollectivePlan:
    world = len(work.ranks)
    if world < 2:
        raise ValueError("ring all-reduce needs at least two ranks")
    expected_rounds = 2 * (world - 1)
    if len(tags) != expected_rounds:
        raise ValueError("ring tag count disagrees with its round count")
    channel = work.channel_hint or "default"
    chunk_bytes = max(1, work.payload_bytes // world)
    rounds = tuple(
        CollectivePlanRound(
            round_index=round_index,
            tag=tag,
            channel_id=f"{channel}:round-{round_index}",
        )
        for round_index, tag in enumerate(tags)
    )

    actions = []
    extents = []
    for round_index in range(expected_rounds):
        for source_index, source_rank in enumerate(work.ranks):
            destination_rank = work.ranks[(source_index + 1) % world]
            extent_id = _extent_id(operation_id, round_index, source_index)
            extents.append(
                CollectivePlanExtent(
                    extent_id=extent_id,
                    round_index=round_index,
                    source_rank=source_rank,
                    destination_rank=destination_rank,
                    payload_bytes=chunk_bytes,
                    send_action_id=_action_id(
                        operation_id,
                        round_index,
                        source_rank,
                        "send",
                    ),
                    receive_action_id=_action_id(
                        operation_id,
                        round_index,
                        destination_rank,
                        "receive",
                    ),
                )
            )
        for rank_index, rank in enumerate(work.ranks):
            successor_extent = extents[-world + rank_index]
            predecessor_extent = extents[-world + ((rank_index - 1) % world)]
            dependencies = (
                ()
                if round_index == 0
                else (
                    _action_id(
                        operation_id,
                        round_index - 1,
                        rank,
                        "receive",
                    ),
                )
            )
            actions.extend(
                (
                    CollectivePlanAction(
                        action_id=successor_extent.send_action_id,
                        rank=rank,
                        kind=CollectivePlanActionKind.SEND,
                        extent_id=successor_extent.extent_id,
                        depends_on=dependencies,
                    ),
                    CollectivePlanAction(
                        action_id=predecessor_extent.receive_action_id,
                        rank=rank,
                        kind=CollectivePlanActionKind.RECEIVE,
                        extent_id=predecessor_extent.extent_id,
                        depends_on=dependencies,
                    ),
                )
            )
    return _finish_plan(
        operation_id=operation_id,
        work=work,
        rounds=rounds,
        actions=tuple(actions),
        extents=tuple(extents),
    )


def _request_partitions(
    work: CollectiveWork,
) -> dict[tuple[int, int], tuple[tuple[str, int], ...]]:
    by_pair: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)
    for request_id, source, destination, payload_bytes in (
        work.request_pair_payload_bytes
    ):
        by_pair[(source, destination)].append((request_id, payload_bytes))
    return {pair: tuple(entries) for pair, entries in by_pair.items()}


def _pairwise_plan(
    operation_id: str,
    work: CollectiveWork,
    tags: tuple[int, ...],
) -> CollectivePlan:
    if len(work.ranks) < 2:
        raise ValueError("pairwise all-to-allv needs at least two ranks")
    if len(tags) != 1:
        raise ValueError("pairwise all-to-allv needs exactly one tag")
    channel = work.channel_hint or "default"
    rounds = (CollectivePlanRound(0, tags[0], channel),)
    if work.pair_payload_bytes:
        pair_payloads = work.pair_payload_bytes
    else:
        pair_payloads = tuple(
            (source, destination, work.payload_bytes)
            for source in work.ranks
            for destination in work.ranks
            if source != destination and work.payload_bytes > 0
        )
    request_partitions = _request_partitions(work)
    actions = []
    extents = []
    for extent_index, (source, destination, payload_bytes) in enumerate(
        pair_payloads
    ):
        extent_id = _extent_id(operation_id, 0, extent_index)
        send_action_id = f"{extent_id}:send"
        receive_action_id = f"{extent_id}:receive"
        extents.append(
            CollectivePlanExtent(
                extent_id=extent_id,
                round_index=0,
                source_rank=source,
                destination_rank=destination,
                payload_bytes=payload_bytes,
                send_action_id=send_action_id,
                receive_action_id=receive_action_id,
                request_payload_bytes=request_partitions.get(
                    (source, destination),
                    (),
                ),
            )
        )
        actions.extend(
            (
                CollectivePlanAction(
                    action_id=send_action_id,
                    rank=source,
                    kind=CollectivePlanActionKind.SEND,
                    extent_id=extent_id,
                ),
                CollectivePlanAction(
                    action_id=receive_action_id,
                    rank=destination,
                    kind=CollectivePlanActionKind.RECEIVE,
                    extent_id=extent_id,
                ),
            )
        )
    return _finish_plan(
        operation_id=operation_id,
        work=work,
        rounds=rounds,
        actions=tuple(actions),
        extents=tuple(extents),
    )


def plan_execution_graph_collectives(
    graph: ExecutionGraph,
    *,
    base_tag: int = DEFAULT_COLLECTIVE_PLAN_BASE_TAG,
) -> ExecutionGraph:
    """Attach one complete traffic-owned physical plan to ``graph``.

    An already planned graph is returned unchanged after validation. A graph
    with no collectives retains an empty plan tuple. Mixed planned and
    unplanned collective authority is not representable.

    Tags come from the accepted allocator in :mod:`simllm.core`, so this
    function never becomes a second tag authority. Once the plan is attached
    the same allocator reads the tags back out of it.
    """

    validate_execution_graph(graph)
    if graph.collective_plans:
        return graph
    tags = collective_goal_tags(graph, base_tag=base_tag)
    plans = []
    for operation in graph.operations:
        work = operation.work
        if not isinstance(work, CollectiveWork):
            continue
        operation_tags = tags[operation.operation_id]
        if work.collective == "all-reduce" and work.algorithm_hint == "ring":
            plan = _ring_plan(operation.operation_id, work, operation_tags)
        elif work.collective == "all-to-allv" and work.algorithm_hint == "pairwise":
            plan = _pairwise_plan(operation.operation_id, work, operation_tags)
        else:
            raise AssertionError("collective tag allocation accepted unsupported work")
        plans.append(plan)
    planned = replace(graph, collective_plans=tuple(plans))
    validate_execution_graph(planned)
    return planned


def collective_plan_by_operation(graph: ExecutionGraph) -> dict[str, CollectivePlan]:
    """Return the explicit plan inventory after validating the graph."""

    validate_execution_graph(graph)
    return {plan.operation_id: plan for plan in graph.collective_plans}


def render_collective_plan(
    trace: GoalTrace,
    plan: CollectivePlan,
    *,
    after: Mapping[int, str] | None = None,
    after_provenance: Mapping[int, GoalDependencyProvenance] | None = None,
    exact_frontier: bool = False,
) -> dict[int, str]:
    """Render declared endpoint actions without reconstructing an algorithm."""

    extent_by_id = {extent.extent_id: extent for extent in plan.extents}
    round_by_index = {round_.round_index: round_ for round_ in plan.rounds}
    label_by_action = {}
    for action in plan.actions:
        extent = extent_by_id[action.extent_id]
        tag = round_by_index[extent.round_index].tag
        if action.kind is CollectivePlanActionKind.SEND:
            label = trace.rank(action.rank).send(
                extent.payload_bytes,
                to=extent.destination_rank,
                tag=tag,
                operation_id=plan.operation_id,
            )
        else:
            label = trace.rank(action.rank).recv(
                extent.payload_bytes,
                source=extent.source_rank,
                tag=tag,
                operation_id=plan.operation_id,
            )
        label_by_action[action.action_id] = label

    internal = GoalDependencyProvenance(
        GoalDependencyKind.COLLECTIVE_INTERNAL,
        plan.operation_id,
    )
    for action in plan.actions:
        label = label_by_action[action.action_id]
        for dependency in action.depends_on:
            trace.rank(action.rank).requires(
                label,
                label_by_action[dependency],
                provenance=internal,
            )
    if after:
        for rank, action_ids in plan.entry_action_ids:
            predecessor = after.get(rank)
            if predecessor is None:
                continue
            provenance = (
                None
                if after_provenance is None
                else after_provenance.get(rank)
            )
            for action_id in action_ids:
                trace.rank(rank).requires(
                    label_by_action[action_id],
                    predecessor,
                    provenance=provenance,
                )

    for extent in plan.extents:
        round_ = round_by_index[extent.round_index]
        trace.record_message(
            GoalMessage(
                operation_id=plan.operation_id,
                source_rank=extent.source_rank,
                destination_rank=extent.destination_rank,
                payload_bytes=extent.payload_bytes,
                tag=round_.tag,
                send_label=label_by_action[extent.send_action_id],
                receive_label=label_by_action[extent.receive_action_id],
                request_payload_bytes=extent.request_payload_bytes,
            )
        )

    frontiers = {}
    for rank, action_ids in plan.terminal_action_ids:
        labels = tuple(label_by_action[action_id] for action_id in action_ids)
        if exact_frontier:
            if not labels:
                if after and rank in after:
                    frontiers[rank] = after[rank]
                else:
                    frontiers[rank] = trace.rank(rank).calc(
                        0,
                        operation_id=plan.operation_id,
                    )
                continue
            join = trace.rank(rank).calc(0, operation_id=plan.operation_id)
            for label in labels:
                trace.rank(rank).requires(join, label, provenance=internal)
            frontiers[rank] = join
        elif labels:
            frontiers[rank] = labels[-1]
    return frontiers


__all__ = [
    "DEFAULT_COLLECTIVE_PLAN_BASE_TAG",
    "collective_plan_by_operation",
    "plan_execution_graph_collectives",
    "render_collective_plan",
]
