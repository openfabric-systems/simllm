"""Compare graph-authoritative ordering with an independent GOAL schedule.

The comparison is diagnostic.  A disagreement is returned as evidence and
never selects a different completion time for the authoritative run.
Malformed or incomplete comparison inputs remain fatal because they cannot
support a meaningful cross-check.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from simllm.core.execution import ExecutionGraph
from simllm.core.execution_io import operation_participant_ranks
from simllm.goal import GoalGraphEdge, GoalMessage, GoalTrace


@dataclass(frozen=True)
class DependencyOrderingComparison:
    """Structural coverage of one graph ordering boundary in direct GOAL."""

    predecessor_id: str
    operation_id: str
    scope: str
    origin: str
    participant_rank: int | None
    predecessor_ranks: tuple[int, ...]
    target_ranks: tuple[int, ...]
    predecessor_terminal_labels: tuple[tuple[int, tuple[str, ...]], ...]
    target_entry_labels: tuple[tuple[int, tuple[str, ...]], ...]
    missing_predecessor_ranks_by_target: tuple[tuple[int, tuple[int, ...]], ...]
    missing_terminal_count: int
    disagreement: bool


@dataclass(frozen=True)
class DependencyPhaseFrontierComparison:
    """Observed timing at one graph ordering boundary."""

    predecessor_id: str
    operation_id: str
    predecessor_tags: tuple[int, ...]
    target_tags: tuple[int, ...]
    authority_predecessor_completion_ps: int | None
    authority_target_start_ps: int | None
    authority_gap_ps: int | None
    cross_check_predecessor_completion_ps: int | None
    cross_check_target_start_ps: int | None
    cross_check_gap_ps: int | None
    signed_gap_difference_ps: int | None
    evaluated: bool
    disagreement: bool


@dataclass(frozen=True)
class DependencyCrossCheckPlan:
    """Validated structural comparison retained until both runs complete."""

    execution_id: str
    step_index: int
    operation_ids: tuple[str, ...]
    expected_message_count: int
    ordering_comparisons: tuple[DependencyOrderingComparison, ...]
    boundary_tags: tuple[
        tuple[str, str, tuple[int, ...], tuple[int, ...]], ...
    ]


@dataclass(frozen=True)
class DependencyCrossCheckReport:
    """Diagnostic comparison of one authoritative and one cross-check run."""

    authority_mechanism: str
    cross_check_mechanism: str
    execution_id: str
    step_index: int
    ordering_comparisons: tuple[DependencyOrderingComparison, ...]
    phase_frontier_comparisons: tuple[DependencyPhaseFrontierComparison, ...]
    ordering_disagreement_count: int
    phase_frontier_disagreement_count: int
    authority_completion_ps: int
    cross_check_completion_ps: int
    signed_completion_difference_ps: int
    completion_tolerance_ps: int
    completion_disagreement: bool
    authority_artifact_names: tuple[str, ...]
    authority_artifact_sha256: tuple[str, ...]
    authority_artifact_bytes: tuple[int, ...]
    cross_check_artifact_name: str
    cross_check_artifact_sha256: str
    cross_check_artifact_bytes: int
    authority_quiescent: bool
    cross_check_quiescent: bool
    authority_flow_count: int
    cross_check_flow_count: int
    has_disagreement: bool


def _message_identity(message: GoalMessage) -> tuple[object, ...]:
    """Return physical identity without artifact-local GOAL labels."""

    return (
        message.operation_id,
        message.source_rank,
        message.destination_rank,
        message.payload_bytes,
        message.tag,
        message.request_payload_bytes,
    )


def _requires_reachability(trace: GoalTrace) -> dict[str, frozenset[str]]:
    labels = {operation.label for operation in trace.operations}
    successors: dict[str, set[str]] = {label: set() for label in labels}
    for dependency in trace.dependencies:
        if dependency.relation == "requires":
            successors[dependency.predecessor_label].add(
                dependency.operation_label
            )

    reachable: dict[str, frozenset[str]] = {}

    def visit(label: str, active: set[str]) -> frozenset[str]:
        if label in reachable:
            return reachable[label]
        if label in active:
            raise ValueError("direct GOAL requires dependencies contain a cycle")
        next_active = active | {label}
        result: set[str] = set()
        for successor in successors[label]:
            result.add(successor)
            result.update(visit(successor, next_active))
        reachable[label] = frozenset(result)
        return reachable[label]

    for label in labels:
        visit(label, set())
    return reachable


def _owned_labels(
    trace: GoalTrace,
) -> dict[str, dict[int, tuple[str, ...]]]:
    labels: dict[str, dict[int, list[str]]] = {}
    for operation in trace.operations:
        if operation.operation_id is None:
            continue
        labels.setdefault(operation.operation_id, {}).setdefault(
            operation.rank, []
        ).append(operation.label)
    return {
        operation_id: {
            rank: tuple(rank_labels)
            for rank, rank_labels in sorted(by_rank.items())
        }
        for operation_id, by_rank in labels.items()
    }


def _entry_and_terminal_labels(
    operation_labels: tuple[str, ...],
    reachable: dict[str, frozenset[str]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    owned = set(operation_labels)
    entries = tuple(
        label
        for label in operation_labels
        if not any(label in reachable[other] for other in owned if other != label)
    )
    terminals = tuple(
        label
        for label in operation_labels
        if not (reachable[label] & (owned - {label}))
    )
    if not entries or not terminals:
        raise ValueError("direct GOAL operation has no requires entry or terminal")
    return entries, terminals


def _boundary_ordering_comparison(
    edge: GoalGraphEdge,
    *,
    graph_ranks: dict[str, tuple[int, ...]],
    labels: dict[str, dict[int, tuple[str, ...]]],
    reachable: dict[str, frozenset[str]],
) -> DependencyOrderingComparison:
    predecessor_ranks = graph_ranks[edge.predecessor_id]
    target_ranks = graph_ranks[edge.operation_id]

    predecessor_terminals: list[tuple[int, tuple[str, ...]]] = []
    for rank in predecessor_ranks:
        try:
            rank_labels = labels[edge.predecessor_id][rank]
        except KeyError as exc:
            raise ValueError(
                "direct GOAL lacks a predecessor participant for boundary "
                f"{edge.predecessor_id!r} to {edge.operation_id!r} on rank {rank}"
            ) from exc
        _, terminals = _entry_and_terminal_labels(rank_labels, reachable)
        predecessor_terminals.append((rank, terminals))

    target_entries: list[tuple[int, tuple[str, ...]]] = []
    for rank in target_ranks:
        try:
            rank_labels = labels[edge.operation_id][rank]
        except KeyError as exc:
            raise ValueError(
                "direct GOAL lacks a target participant for boundary "
                f"{edge.predecessor_id!r} to {edge.operation_id!r} on rank {rank}"
            ) from exc
        entries, _ = _entry_and_terminal_labels(rank_labels, reachable)
        target_entries.append((rank, entries))

    required_predecessor_ranks = predecessor_ranks
    checked_target_entries = target_entries
    if edge.scope == "participant-local":
        if edge.participant_rank is None:
            raise ValueError("participant-local boundary lacks its participant rank")
        required_predecessor_ranks = (edge.participant_rank,)
        checked_target_entries = [
            entry for entry in target_entries if entry[0] == edge.participant_rank
        ]
        if not checked_target_entries:
            raise ValueError("participant-local boundary rank is absent from target")

    terminals_by_rank = dict(predecessor_terminals)
    missing_by_target: list[tuple[int, tuple[int, ...]]] = []
    missing_terminal_count = 0
    for target_rank, entries in checked_target_entries:
        missing_ranks: list[int] = []
        for predecessor_rank in required_predecessor_ranks:
            terminals = terminals_by_rank[predecessor_rank]
            missing_terminals = tuple(
                terminal
                for terminal in terminals
                if any(entry not in reachable[terminal] for entry in entries)
            )
            if missing_terminals:
                missing_ranks.append(predecessor_rank)
                missing_terminal_count += len(missing_terminals)
        missing_by_target.append((target_rank, tuple(missing_ranks)))

    return DependencyOrderingComparison(
        predecessor_id=edge.predecessor_id,
        operation_id=edge.operation_id,
        scope=edge.scope,
        origin=edge.origin,
        participant_rank=edge.participant_rank,
        predecessor_ranks=predecessor_ranks,
        target_ranks=target_ranks,
        predecessor_terminal_labels=tuple(predecessor_terminals),
        target_entry_labels=tuple(target_entries),
        missing_predecessor_ranks_by_target=tuple(missing_by_target),
        missing_terminal_count=missing_terminal_count,
        disagreement=missing_terminal_count != 0,
    )


def plan_dependency_cross_check(
    graph: ExecutionGraph,
    direct_trace: GoalTrace,
    boundary_edges: tuple[GoalGraphEdge, ...],
    expected_messages: tuple[GoalMessage, ...],
) -> DependencyCrossCheckPlan:
    """Validate inputs and plan an independent dependency comparison."""

    graph_operation_ids = tuple(
        operation.operation_id for operation in graph.operations
    )
    if len(graph_operation_ids) != len(set(graph_operation_ids)):
        raise ValueError("graph contains duplicate operation identities")
    labels = _owned_labels(direct_trace)
    if set(labels) != set(graph_operation_ids):
        missing = sorted(set(graph_operation_ids) - set(labels))
        unexpected = sorted(set(labels) - set(graph_operation_ids))
        raise ValueError(
            "direct GOAL semantic operation inventory differs from graph: "
            f"missing={missing}, unexpected={unexpected}"
        )

    expected_inventory = Counter(_message_identity(row) for row in expected_messages)
    direct_inventory = Counter(_message_identity(row) for row in direct_trace.messages)
    if direct_inventory != expected_inventory:
        missing = sum((expected_inventory - direct_inventory).values())
        unexpected = sum((direct_inventory - expected_inventory).values())
        raise ValueError(
            "direct GOAL physical message inventory differs from graph projection: "
            f"missing={missing}, unexpected={unexpected}"
        )

    graph_ranks = {
        operation.operation_id: operation_participant_ranks(operation)
        for operation in graph.operations
    }
    reachable = _requires_reachability(direct_trace)
    ordering = tuple(
        _boundary_ordering_comparison(
            edge,
            graph_ranks=graph_ranks,
            labels=labels,
            reachable=reachable,
        )
        for edge in boundary_edges
    )
    tags_by_operation: dict[str, set[int]] = {
        operation_id: set() for operation_id in graph_operation_ids
    }
    for message in direct_trace.messages:
        if message.operation_id is not None:
            tags_by_operation[message.operation_id].add(message.tag)
    boundary_tags = tuple(
        (
            edge.predecessor_id,
            edge.operation_id,
            tuple(sorted(tags_by_operation[edge.predecessor_id])),
            tuple(sorted(tags_by_operation[edge.operation_id])),
        )
        for edge in boundary_edges
    )
    return DependencyCrossCheckPlan(
        execution_id=graph.execution_id,
        step_index=graph.step_index,
        operation_ids=graph_operation_ids,
        expected_message_count=len(expected_messages),
        ordering_comparisons=ordering,
        boundary_tags=boundary_tags,
    )


def _validate_rows(
    rows: Iterable[tuple[int, int, int]],
    name: str,
) -> tuple[tuple[int, int, int], ...]:
    result = tuple(rows)
    for index, row in enumerate(result):
        if not isinstance(row, tuple) or len(row) != 3:
            raise TypeError(f"{name}[{index}]: expected a three-item tuple")
        tag, start_ps, completion_ps = row
        if any(type(value) is not int for value in row):
            raise TypeError(f"{name}[{index}]: values must be integers")
        if tag < 0 or start_ps < 0 or completion_ps < start_ps:
            raise ValueError(f"{name}[{index}]: invalid completion interval")
    return result


def _frontier_gap(
    rows: tuple[tuple[int, int, int], ...],
    predecessor_tags: tuple[int, ...],
    target_tags: tuple[int, ...],
    name: str,
) -> tuple[int, int, int] | None:
    if not predecessor_tags or not target_tags:
        return None
    observed_tags = {tag for tag, _, _ in rows}
    expected_tags = set(predecessor_tags) | set(target_tags)
    missing = sorted(expected_tags - observed_tags)
    if missing:
        raise ValueError(f"{name} rows lack boundary tags {missing}")
    predecessor_completion = max(
        completion_ps
        for tag, _, completion_ps in rows
        if tag in predecessor_tags
    )
    target_start = min(
        start_ps for tag, start_ps, _ in rows if tag in target_tags
    )
    return predecessor_completion, target_start, target_start - predecessor_completion


def complete_dependency_cross_check(
    plan: DependencyCrossCheckPlan,
    *,
    authority_rows: Iterable[tuple[int, int, int]],
    cross_check_rows: Iterable[tuple[int, int, int]],
    authority_completion_ps: int,
    cross_check_completion_ps: int,
    tolerance_ps: int,
    authority_artifact_names: tuple[str, ...],
    authority_artifact_sha256: tuple[str, ...],
    authority_artifact_bytes: tuple[int, ...],
    cross_check_artifact_name: str,
    cross_check_artifact_sha256: str,
    cross_check_artifact_bytes: int,
    authority_quiescent: bool,
    cross_check_quiescent: bool,
    authority_flow_count: int,
    cross_check_flow_count: int,
) -> DependencyCrossCheckReport:
    """Complete a diagnostic comparison without rejecting disagreements."""

    if not isinstance(plan, DependencyCrossCheckPlan):
        raise TypeError("plan: expected DependencyCrossCheckPlan")
    authority = _validate_rows(authority_rows, "authority_rows")
    cross_check = _validate_rows(cross_check_rows, "cross_check_rows")
    for name, value in (
        ("authority_completion_ps", authority_completion_ps),
        ("cross_check_completion_ps", cross_check_completion_ps),
        ("tolerance_ps", tolerance_ps),
        ("cross_check_artifact_bytes", cross_check_artifact_bytes),
        ("authority_flow_count", authority_flow_count),
        ("cross_check_flow_count", cross_check_flow_count),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not (
        len(authority_artifact_names)
        == len(authority_artifact_sha256)
        == len(authority_artifact_bytes)
    ):
        raise ValueError("authority artifact evidence has mismatched lengths")
    if not authority_artifact_names:
        raise ValueError("authority artifact evidence cannot be empty")
    if any(type(size) is not int or size < 0 for size in authority_artifact_bytes):
        raise ValueError("authority artifact byte counts must be nonnegative integers")
    if type(authority_quiescent) is not bool or type(cross_check_quiescent) is not bool:
        raise TypeError("quiescent fields must be booleans")
    if not authority_quiescent or not cross_check_quiescent:
        raise ValueError("both dependency comparison runs must reach quiescence")
    if authority_flow_count != len(authority):
        raise ValueError("authority flow count does not match timing row inventory")
    if cross_check_flow_count != len(cross_check):
        raise ValueError("cross-check flow count does not match timing row inventory")
    if authority_flow_count != plan.expected_message_count:
        raise ValueError("authority flow inventory is incomplete")
    if cross_check_flow_count != plan.expected_message_count:
        raise ValueError("cross-check flow inventory is incomplete")

    frontier_comparisons: list[DependencyPhaseFrontierComparison] = []
    for predecessor_id, operation_id, predecessor_tags, target_tags in plan.boundary_tags:
        authority_frontier = _frontier_gap(
            authority,
            predecessor_tags,
            target_tags,
            "authority",
        )
        cross_check_frontier = _frontier_gap(
            cross_check,
            predecessor_tags,
            target_tags,
            "cross_check",
        )
        evaluated = authority_frontier is not None and cross_check_frontier is not None
        authority_predecessor_completion = (
            authority_frontier[0] if authority_frontier is not None else None
        )
        authority_target_start = (
            authority_frontier[1] if authority_frontier is not None else None
        )
        authority_gap = (
            authority_frontier[2] if authority_frontier is not None else None
        )
        cross_check_predecessor_completion = (
            cross_check_frontier[0] if cross_check_frontier is not None else None
        )
        cross_check_target_start = (
            cross_check_frontier[1] if cross_check_frontier is not None else None
        )
        cross_check_gap = (
            cross_check_frontier[2] if cross_check_frontier is not None else None
        )
        signed_difference = (
            cross_check_gap - authority_gap if evaluated else None
        )
        disagreement = bool(
            evaluated and cross_check_gap < 0 <= authority_gap
        )
        frontier_comparisons.append(
            DependencyPhaseFrontierComparison(
                predecessor_id=predecessor_id,
                operation_id=operation_id,
                predecessor_tags=predecessor_tags,
                target_tags=target_tags,
                authority_predecessor_completion_ps=(
                    authority_predecessor_completion
                ),
                authority_target_start_ps=authority_target_start,
                authority_gap_ps=authority_gap,
                cross_check_predecessor_completion_ps=(
                    cross_check_predecessor_completion
                ),
                cross_check_target_start_ps=cross_check_target_start,
                cross_check_gap_ps=cross_check_gap,
                signed_gap_difference_ps=signed_difference,
                evaluated=evaluated,
                disagreement=disagreement,
            )
        )

    completion_difference = cross_check_completion_ps - authority_completion_ps
    completion_disagreement = abs(completion_difference) > tolerance_ps
    ordering_count = sum(
        comparison.disagreement for comparison in plan.ordering_comparisons
    )
    frontier_count = sum(
        comparison.disagreement for comparison in frontier_comparisons
    )
    return DependencyCrossCheckReport(
        authority_mechanism="execution-graph-projection",
        cross_check_mechanism="atlahs-independent-goal",
        execution_id=plan.execution_id,
        step_index=plan.step_index,
        ordering_comparisons=plan.ordering_comparisons,
        phase_frontier_comparisons=tuple(frontier_comparisons),
        ordering_disagreement_count=ordering_count,
        phase_frontier_disagreement_count=frontier_count,
        authority_completion_ps=authority_completion_ps,
        cross_check_completion_ps=cross_check_completion_ps,
        signed_completion_difference_ps=completion_difference,
        completion_tolerance_ps=tolerance_ps,
        completion_disagreement=completion_disagreement,
        authority_artifact_names=authority_artifact_names,
        authority_artifact_sha256=authority_artifact_sha256,
        authority_artifact_bytes=authority_artifact_bytes,
        cross_check_artifact_name=cross_check_artifact_name,
        cross_check_artifact_sha256=cross_check_artifact_sha256,
        cross_check_artifact_bytes=cross_check_artifact_bytes,
        authority_quiescent=authority_quiescent,
        cross_check_quiescent=cross_check_quiescent,
        authority_flow_count=authority_flow_count,
        cross_check_flow_count=cross_check_flow_count,
        has_disagreement=(
            ordering_count != 0
            or frontier_count != 0
            or completion_disagreement
        ),
    )
