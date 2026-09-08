"""Checked projections of one native session and its declared GOAL actions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from simllm.core.authority import work_completed_bytes
from simllm.core.execution import (
    CompletionEvent,
    DependencyScope,
    EventPhase,
    ExecutionGraph,
    ExecutionResult,
    ResourceKind,
    ResourceRef,
)
from simllm.core.execution_io import (
    effective_dependency_edges,
    execution_result_to_json,
    operation_participant_ranks,
)
from simllm.core.step import StepResult

if TYPE_CHECKING:
    from .flow_session import FlowSession, FlowSessionDrain
    from .goal_session import GoalArtifactResult, GoalTraceSnapshot


@dataclass(frozen=True)
class SessionStepEvidence:
    """Original graph evidence, with native drain time retained separately."""

    graph: ExecutionGraph
    execution_result: ExecutionResult
    artifacts: tuple[GoalArtifactResult, ...]
    native_drain: FlowSessionDrain
    native_events: tuple
    transcript: tuple[tuple[str, bytes], ...]

    def validate_result(self, result: StepResult) -> None:
        if (result.step_index != self.graph.step_index
                or result.completed_at_ps != self.execution_result.completed_at_ps
                or result.step_latency_ps != result.completed_at_ps - self.graph.released_at_ps):
            raise ValueError("native session evidence disagrees with the returned StepResult")


def build_session_evidence(
    graph: ExecutionGraph,
    snapshots: tuple[GoalTraceSnapshot, ...],
    artifacts: tuple[GoalArtifactResult, ...],
    session: FlowSession,
    drain: FlowSessionDrain,
) -> SessionStepEvidence:
    """Join immutable native rows and action frontiers to the checked graph."""

    if len(snapshots) != len(artifacts) or not artifacts:
        raise ValueError("session lost its planned artifact inventory")
    if len({artifact.artifact_id for artifact in artifacts}) != len(artifacts):
        raise ValueError("session duplicated a planned artifact")
    operations = {operation.operation_id: operation for operation in graph.operations}
    owned = {name: [] for name in operations}
    native_rows = []
    sequences = []
    for snapshot, artifact in zip(snapshots, artifacts, strict=True):
        expected = {(op.rank, op.label): op for op in snapshot.operations}
        actions = {(action.rank, action.label): action for action in artifact.actions}
        if len(actions) != len(artifact.actions) or actions.keys() != expected.keys():
            raise ValueError("session action inventory is missing or duplicated")
        if artifact.completed_at_ps != max(action.completed_at_ps for action in artifact.actions):
            raise ValueError("artifact boundary disagrees with its action completions")
        for key, action in actions.items():
            operation = expected[key]
            if action.operation_id != operation.operation_id:
                raise ValueError("action semantic owner changed after planning")
            if not (artifact.released_at_ps <= action.eligible_at_ps
                    <= action.started_at_ps <= action.completed_at_ps):
                raise ValueError("action timestamps violate their causal order")
            if action.operation_id is not None:
                if action.operation_id not in operations:
                    raise ValueError("action names an operation outside the original graph")
                owned[action.operation_id].append(action)
        wanted = Counter((m.operation_id, m.source_rank, m.destination_rank,
                          m.payload_bytes, m.tag) for m in snapshot.messages)
        observed = Counter((r["operation_id"], r["source"], r["destination"],
                            r["payload_bytes"], r["tag"]) for r in artifact.completion_rows)
        if wanted != observed:
            raise ValueError("native completions do not conserve the planned message inventory")
        if tuple(r["sequence"] for r in artifact.completion_rows) != artifact.accepted_sequences:
            raise ValueError("artifact completion rows disagree with their accepted sequences")
        native_rows.extend(artifact.completion_rows)
        sequences.extend(artifact.accepted_sequences)
    if sequences != list(range(1, len(sequences) + 1)):
        raise ValueError("artifacts lost or duplicated native sequence ownership")
    if tuple(native_rows) != session.completion_rows or tuple(native_rows) != drain.completion_rows:
        raise ValueError("artifact projections disagree with the retained native completion authority")
    if drain.quiesced_at_ps < max((row["completion_time_ps"] for row in native_rows), default=0):
        raise ValueError("native quiescence precedes an authoritative physical completion")
    if any(row["execution_id"] != graph.execution_id for row in native_rows):
        raise ValueError("native row belongs to another checked graph")

    starts, finishes, participant_starts, participant_finishes = {}, {}, {}, {}
    for name, operation in operations.items():
        actions = owned[name]
        ranks = operation_participant_ranks(operation)
        if not actions or {action.rank for action in actions} != set(ranks):
            raise ValueError("original graph operation lost a participant action frontier")
        starts[name] = min(action.eligible_at_ps for action in actions)
        finishes[name] = max(action.completed_at_ps for action in actions)
        if starts[name] < max(graph.released_at_ps, operation.not_before_ps):
            raise ValueError("operation starts before its declared graph release")
        for rank in ranks:
            selected = [action for action in actions if action.rank == rank]
            participant_starts[name, rank] = min(action.eligible_at_ps for action in selected)
            participant_finishes[name, rank] = max(action.completed_at_ps for action in selected)
    for edge in effective_dependency_edges(graph):
        if edge.scope is DependencyScope.WHOLE_OPERATION:
            before, after = finishes[edge.predecessor_id], starts[edge.operation_id]
        else:
            before = participant_finishes[edge.predecessor_id, edge.participant_rank]
            after = participant_starts[edge.operation_id, edge.participant_rank]
        if after < before:
            raise ValueError("session action frontier violates an effective graph dependency")
    boundary_ids = graph.completion_operation_ids or tuple(operations)
    completed_at = max(finishes[name] for name in boundary_ids)
    if completed_at != artifacts[-1].completed_at_ps:
        raise ValueError("serial session boundary is not the original graph completion boundary")

    phase_map = {"accepted": EventPhase.SUBMITTED, "queued": EventPhase.QUEUED,
                 "started": EventPhase.STARTED, "completed": EventPhase.COMPLETED}
    events = []
    for row in session.events:
        if row["execution_id"] != graph.execution_id or row["operation_id"] not in operations:
            raise ValueError("native lifecycle event is outside the original graph")
        events.append(CompletionEvent(
            execution_id=graph.execution_id, operation_id=row["operation_id"],
            phase=phase_map[row["kind"]], timestamp_ps=row["timestamp_ps"],
            resource=ResourceRef(ResourceKind.NIC_SEND_QUEUE,
                                 f"{session.session_id}:rank-{row['source']}:sq-{row['sq_id']}"),
            completed_bytes=row["payload_bytes"] if row["kind"] == "completed" else None,
            subject_object_id=row["flow_id"],
        ))
    for name, operation in operations.items():
        for phase, at in ((EventPhase.SUBMITTED, graph.released_at_ps),
                          (EventPhase.QUEUED, starts[name]),
                          (EventPhase.STARTED, min(a.started_at_ps for a in owned[name])),
                          (EventPhase.COMPLETED, finishes[name])):
            events.append(CompletionEvent(
                graph.execution_id, name, phase, at,
                completed_bytes=work_completed_bytes(operation) if phase is EventPhase.COMPLETED else None,
            ))
    # Local actions are graph work too. Preserve native Q; the graph drains
    # when both its last local action and the native authority have finished.
    quiesced_at = max(drain.quiesced_at_ps, max(a.completed_at_ps for a in artifacts))
    result = ExecutionResult(graph.execution_id, completed_at,
                             tuple(sorted(events, key=lambda event: event.timestamp_ps)), quiesced_at)
    execution_result_to_json(result)
    return SessionStepEvidence(graph, result, artifacts, drain, session.events, session.transcript)
