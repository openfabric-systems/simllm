"""Compare graph-authoritative ordering with an independent GOAL schedule.

The comparison is diagnostic.  A disagreement is returned as evidence and
never selects a different completion time for the authoritative run.
Malformed or incomplete comparison inputs remain fatal because they cannot
support a meaningful cross-check.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass

from simllm.core.execution import ExecutionGraph
from simllm.core.execution_io import (
    effective_dependency_edges,
    operation_participant_ranks,
)
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
    frontier_boundary_keys: tuple[
        tuple[str, str, str, str, int | None], ...
    ]
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
    ordering_edge_count: int
    ordering_disagreement_count: int
    ordering_disagreement_classes: tuple[tuple[str, str, int], ...]
    frontier_boundary_count: int
    boundary_ordering_disagreement_count: int
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


def _edge_identity(
    edge: GoalGraphEdge | DependencyOrderingComparison,
) -> tuple[str, str, str, str, int | None]:
    return (
        edge.predecessor_id,
        edge.operation_id,
        edge.scope,
        edge.origin,
        edge.participant_rank,
    )


@dataclass(frozen=True)
class _RequiresGraph:
    """Iterative, query-directed reachability for one direct GOAL program.

    Materializing the full transitive closure is quadratic for serial schedules
    and exhausts memory at realistic rank and layer counts.  Cross-checking
    asks only whether selected boundary labels are reachable, so retain the
    linear direct graph and search one queried source at a time.
    """

    successors: dict[str, tuple[str, ...]]
    predecessors: dict[str, tuple[str, ...]]
    ranks: dict[str, int]
    owner_keys: dict[str, tuple[str, int]]
    _cache: dict[tuple[str, str], bool]
    _frontier_cache: dict[
        tuple[str, ...], tuple[tuple[str, ...], tuple[str, ...]]
    ]

    def reaches(self, predecessor_label: str, operation_label: str) -> bool:
        if predecessor_label not in self.successors:
            raise ValueError(
                f"direct GOAL requires predecessor {predecessor_label!r} is unknown"
            )
        if operation_label not in self.successors:
            raise ValueError(
                f"direct GOAL requires target {operation_label!r} is unknown"
            )
        if self.ranks[predecessor_label] != self.ranks[operation_label]:
            return False
        key = (predecessor_label, operation_label)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        pending = list(self.successors[predecessor_label])
        visited: set[str] = set()
        while pending:
            label = pending.pop()
            if label == operation_label:
                self._cache[key] = True
                return True
            if label in visited:
                continue
            visited.add(label)
            pending.extend(self.successors[label])
        self._cache[key] = False
        return False


def _require_acyclic(
    successors: dict[object, set[object]],
    *,
    description: str,
) -> None:
    indegree = {label: 0 for label in successors}
    for next_labels in successors.values():
        for next_label in next_labels:
            indegree[next_label] += 1
    ready = deque(label for label, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        label = ready.popleft()
        visited += 1
        for next_label in successors[label]:
            indegree[next_label] -= 1
            if indegree[next_label] == 0:
                ready.append(next_label)
    if visited != len(successors):
        raise ValueError(f"direct GOAL {description} contains a cycle")


def _requires_graph(trace: GoalTrace) -> _RequiresGraph:
    operations = trace.operations
    labels = {operation.label for operation in operations}
    successors: dict[str, list[str]] = {label: [] for label in labels}
    predecessors: dict[str, list[str]] = {label: [] for label in labels}
    for dependency in trace.dependencies:
        if dependency.relation == "requires":
            try:
                successors[dependency.predecessor_label].append(
                    dependency.operation_label
                )
                predecessors[dependency.operation_label].append(
                    dependency.predecessor_label
                )
            except KeyError as exc:
                raise ValueError(
                    "direct GOAL requires dependency cites an unknown label"
                ) from exc
    _require_acyclic(
        {label: set(next_labels) for label, next_labels in successors.items()},
        description="requires dependencies",
    )

    ranks = {operation.label: operation.rank for operation in operations}
    owner_keys = {
        operation.label: (
            operation.operation_id
            if operation.operation_id is not None
            else f"unowned:{operation.label}",
            operation.rank,
        )
        for operation in operations
    }
    owner_successors: dict[tuple[str, int], set[tuple[str, int]]] = {
        owner: set() for owner in owner_keys.values()
    }
    for predecessor_label, next_labels in successors.items():
        predecessor_owner = owner_keys[predecessor_label]
        for operation_label in next_labels:
            operation_owner = owner_keys[operation_label]
            if predecessor_owner != operation_owner:
                owner_successors[predecessor_owner].add(operation_owner)
    _require_acyclic(
        owner_successors,
        description="semantic-owner dependency projection",
    )
    return _RequiresGraph(
        successors={
            label: tuple(next_labels) for label, next_labels in successors.items()
        },
        predecessors={
            label: tuple(previous_labels)
            for label, previous_labels in predecessors.items()
        },
        ranks=ranks,
        owner_keys=owner_keys,
        _cache={},
        _frontier_cache={},
    )


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
    requires: _RequiresGraph,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    cached = requires._frontier_cache.get(operation_labels)
    if cached is not None:
        return cached
    owned = set(operation_labels)
    owner_keys = {requires.owner_keys[label] for label in owned}
    if len(owner_keys) != 1:
        raise ValueError("direct GOAL operation labels do not share one semantic owner")
    entries = tuple(
        label
        for label in operation_labels
        if not any(previous in owned for previous in requires.predecessors[label])
    )
    terminals = tuple(
        label
        for label in operation_labels
        if not any(next_label in owned for next_label in requires.successors[label])
    )
    if not entries or not terminals:
        raise ValueError("direct GOAL operation has no requires entry or terminal")
    result = entries, terminals
    requires._frontier_cache[operation_labels] = result
    return result


def _boundary_ordering_comparison(
    edge: GoalGraphEdge,
    *,
    graph_ranks: dict[str, tuple[int, ...]],
    labels: dict[str, dict[int, tuple[str, ...]]],
    requires: _RequiresGraph,
) -> DependencyOrderingComparison:
    predecessor_ranks = graph_ranks[edge.predecessor_id]
    target_ranks = graph_ranks[edge.operation_id]

    checked_predecessor_ranks = predecessor_ranks
    checked_target_ranks = target_ranks
    if edge.scope == "participant-local":
        if edge.participant_rank is None:
            raise ValueError("participant-local boundary lacks its participant rank")
        checked_predecessor_ranks = (edge.participant_rank,)
        checked_target_ranks = (edge.participant_rank,)

    predecessor_terminals: list[tuple[int, tuple[str, ...]]] = []
    for rank in checked_predecessor_ranks:
        try:
            rank_labels = labels[edge.predecessor_id][rank]
        except KeyError as exc:
            raise ValueError(
                "direct GOAL lacks a predecessor participant for boundary "
                f"{edge.predecessor_id!r} to {edge.operation_id!r} on rank {rank}"
            ) from exc
        _, terminals = _entry_and_terminal_labels(rank_labels, requires)
        predecessor_terminals.append((rank, terminals))

    target_entries: list[tuple[int, tuple[str, ...]]] = []
    for rank in checked_target_ranks:
        try:
            rank_labels = labels[edge.operation_id][rank]
        except KeyError as exc:
            raise ValueError(
                "direct GOAL lacks a target participant for boundary "
                f"{edge.predecessor_id!r} to {edge.operation_id!r} on rank {rank}"
            ) from exc
        entries, _ = _entry_and_terminal_labels(rank_labels, requires)
        target_entries.append((rank, entries))

    required_predecessor_ranks = checked_predecessor_ranks
    checked_target_entries = target_entries
    if edge.scope == "participant-local":
        required_predecessor_ranks = (edge.participant_rank,)
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
                if any(not requires.reaches(terminal, entry) for entry in entries)
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
    requires = _requires_graph(direct_trace)
    ordering_edges = tuple(
        GoalGraphEdge(
            edge.predecessor_id,
            edge.operation_id,
            edge.scope.value,
            edge.origin.value,
            edge.participant_rank,
        )
        for edge in effective_dependency_edges(graph)
    )
    ordering_edge_keys = tuple(_edge_identity(edge) for edge in ordering_edges)
    ordering_edge_key_set = set(ordering_edge_keys)
    boundary_keys = tuple(_edge_identity(edge) for edge in boundary_edges)
    if len(boundary_keys) != len(set(boundary_keys)):
        raise ValueError("cross-check frontier boundary inventory contains duplicates")
    unexpected_boundaries = tuple(
        key for key in boundary_keys if key not in ordering_edge_key_set
    )
    if unexpected_boundaries:
        raise ValueError(
            "cross-check frontier boundary is absent from the execution graph: "
            f"{unexpected_boundaries[0]!r}"
        )
    ordering = tuple(
        _boundary_ordering_comparison(
            edge,
            graph_ranks=graph_ranks,
            labels=labels,
            requires=requires,
        )
        for edge in ordering_edges
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
        frontier_boundary_keys=boundary_keys,
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
        disagreement = bool(evaluated and cross_check_gap != authority_gap)
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
    ordering_classes = Counter(
        (comparison.scope, comparison.origin)
        for comparison in plan.ordering_comparisons
        if comparison.disagreement
    )
    boundary_keys = set(plan.frontier_boundary_keys)
    boundary_ordering_count = sum(
        comparison.disagreement
        for comparison in plan.ordering_comparisons
        if _edge_identity(comparison) in boundary_keys
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
        ordering_edge_count=len(plan.ordering_comparisons),
        ordering_disagreement_count=ordering_count,
        ordering_disagreement_classes=tuple(
            (scope, origin, count)
            for (scope, origin), count in sorted(ordering_classes.items())
        ),
        frontier_boundary_count=len(plan.frontier_boundary_keys),
        boundary_ordering_disagreement_count=boundary_ordering_count,
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
