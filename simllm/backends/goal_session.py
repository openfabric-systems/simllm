from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from simllm.goal.emitter import (
    GoalDependency,
    GoalDependencyKind,
    GoalMessage,
    GoalOperation,
    GoalTrace,
)

if TYPE_CHECKING:
    from .flow_session import FlowSession

_U64 = (1 << 64) - 1
_U32 = (1 << 32) - 1
_LABEL = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_CALC = re.compile(r"calc ([0-9]+)(?: cpu (0))?")
_MESSAGE = re.compile(
    r"(send|recv) ([0-9]+)b (to|from) ([0-9]+) tag ([0-9]+)"
    r"(?: cpu (0))?(?: nic (0))?"
)
ActionKey = tuple[int, str]


def _uint(value: object, name: str, maximum: int = _U64, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be nonblank text")
    value.encode("utf-8")
    return value


@dataclass(frozen=True)
class _Action:
    operation: GoalOperation
    kind: str
    duration_ps: int = 0
    peer: int | None = None
    payload_bytes: int = 0
    tag: int = 0

    @property
    def key(self) -> ActionKey:
        return self.operation.rank, self.operation.label


def _parse(operation: GoalOperation, num_ranks: int) -> _Action:
    if not isinstance(operation, GoalOperation):
        raise TypeError("operations must contain GoalOperation records")
    _uint(operation.rank, "operation.rank", num_ranks - 1)
    if type(operation.label) is not str or _LABEL.fullmatch(operation.label) is None:
        raise ValueError("operation label is outside the supported GOAL grammar")
    if operation.operation_id is not None:
        _text(operation.operation_id, "operation_id")
    if type(operation.text) is not str:
        raise ValueError("operation text must be a string")
    match = _CALC.fullmatch(operation.text)
    if match:
        cost = _uint(int(match[1]), "calc cost", _U64 // 1000)
        if operation.operation_id is None and cost != 0:
            raise ValueError("only calc 0 padding may lack a semantic owner")
        return _Action(operation, "calc", max(cost, 1) * 1000)
    match = _MESSAGE.fullmatch(operation.text)
    if match is None or (match[1], match[3]) not in {("send", "to"), ("recv", "from")}:
        raise ValueError("unsupported GOAL grammar or nonzero CPU/NIC selector")
    if operation.operation_id is None:
        raise ValueError("physical actions require a semantic owner")
    peer = _uint(int(match[4]), "message peer", num_ranks - 1)
    if peer == operation.rank:
        raise ValueError("self/local messages are unsupported")
    return _Action(operation, match[1], peer=peer,
                   payload_bytes=_uint(int(match[2]), "payload", minimum=1),
                   tag=_uint(int(match[5]), "tag", _U32))


def _render(num_ranks: int, operations: tuple[GoalOperation, ...],
            dependencies: tuple[GoalDependency, ...]) -> bytes:
    lines = [f"num_ranks {num_ranks}"]
    for rank in range(num_ranks):
        lines.append(f"rank {rank} {{")
        lines.extend(f"{op.label}: {op.text}" for op in operations if op.rank == rank)
        lines.extend(dep.render() for dep in dependencies if dep.rank == rank)
        lines.append("}")
    return ("\n".join(lines) + "\n").encode("utf-8")


@dataclass(frozen=True)
class GoalTraceSnapshot:
    num_ranks: int
    operations: tuple[GoalOperation, ...]
    messages: tuple[GoalMessage, ...]
    dependencies: tuple[GoalDependency, ...]
    rendered_bytes: bytes
    sha256: str

    @classmethod
    def from_trace(cls, trace: GoalTrace) -> GoalTraceSnapshot:
        if not isinstance(trace, GoalTrace):
            raise TypeError("snapshot requires a structured GoalTrace")
        rendered = trace.render().encode("utf-8")
        return cls(trace.num_ranks, trace.operations, trace.messages, trace.dependencies,
                   rendered, hashlib.sha256(rendered).hexdigest())

    def __post_init__(self) -> None:
        _uint(self.num_ranks, "num_ranks", (1 << 31) - 1, 1)
        for name in ("operations", "messages", "dependencies"):
            if type(getattr(self, name)) is not tuple:
                raise TypeError(f"snapshot {name} must be a tuple")
        parsed = tuple(_parse(operation, self.num_ranks) for operation in self.operations)
        actions = {action.key: action for action in parsed}
        if len(actions) != len(parsed):
            raise ValueError("duplicate rank/label action identity")
        if not actions:
            raise ValueError("GOAL snapshot has no actions")
        if tuple(op.rank for op in self.operations) != tuple(
                sorted(op.rank for op in self.operations)):
            raise ValueError("operations must preserve rank-major rendered order")
        padding = {key for key, action in actions.items()
                   if action.operation.operation_id is None}
        for rank, _ in padding:
            if sum(operation.rank == rank for operation in self.operations) != 1:
                raise ValueError("unowned padding must be the only action on an inactive rank")
        matched: set[ActionKey] = set()
        envelopes: set[tuple[int, int, int]] = set()
        for message in self.messages:
            if not isinstance(message, GoalMessage):
                raise TypeError("messages must contain GoalMessage records")
            # Revalidate nested immutable metadata as well as the rendered join.
            message.__post_init__()
            envelope = (message.source_rank, message.destination_rank, message.tag)
            if envelope in envelopes:
                raise ValueError("ambiguous repeated source/destination/tag message")
            envelopes.add(envelope)
            for rank, label, kind, peer in (
                (message.source_rank, message.send_label, "send", message.destination_rank),
                (message.destination_rank, message.receive_label, "recv", message.source_rank),
            ):
                key = rank, label
                action = actions.get(key)
                if key in matched or action is None:
                    raise ValueError("duplicate or missing message action identity")
                if (action.kind, action.peer, action.payload_bytes, action.tag,
                        action.operation.operation_id) != (
                        kind, peer, message.payload_bytes, message.tag, message.operation_id):
                    raise ValueError("message metadata disagrees with rendered action or owner")
                matched.add(key)
        if matched != {key for key, action in actions.items() if action.kind != "calc"}:
            raise ValueError("message inventory does not cover every send and receive")
        incoming: dict[ActionKey, set[ActionKey]] = {key: set() for key in actions}
        for dependency in self.dependencies:
            if not isinstance(dependency, GoalDependency):
                raise TypeError("dependencies must contain GoalDependency records")
            dependency.__post_init__()
            if dependency.relation != "requires":
                raise ValueError("irequires needs explicit start evidence and is unsupported")
            target = dependency.rank, dependency.operation_label
            predecessor = dependency.rank, dependency.predecessor_label
            if target not in actions or predecessor not in actions:
                raise ValueError("dependency references an unknown rank/label")
            if target in padding or predecessor in padding:
                raise ValueError("unowned padding cannot carry semantic dependencies")
            if predecessor in incoming[target]:
                raise ValueError("duplicate dependency identity")
            incoming[target].add(predecessor)
            provenance = dependency.provenance
            if provenance is not None:
                provenance.__post_init__()
                owner = actions[target].operation.operation_id
                prior_owner = actions[predecessor].operation.operation_id
                if owner != provenance.operation_id:
                    raise ValueError("dependency semantic target owner disagrees")
                if provenance.kind is GoalDependencyKind.COLLECTIVE_INTERNAL:
                    if prior_owner != owner:
                        raise ValueError("internal dependency crosses semantic owners")
                else:
                    for edge in provenance.graph_edges:
                        edge.__post_init__()
                        if edge.predecessor_id != prior_owner or (
                                edge.participant_rank is not None and
                                edge.participant_rank != dependency.rank):
                            raise ValueError("dependency graph-edge identity disagrees")
        remaining = {key: set(predecessors) for key, predecessors in incoming.items()}
        while remaining:
            ready = {key for key, predecessors in remaining.items() if not predecessors}
            if not ready:
                raise ValueError("requires dependency graph is cyclic")
            remaining = {key: predecessors - ready for key, predecessors in remaining.items()
                         if key not in ready}
        if type(self.rendered_bytes) is not bytes or self.rendered_bytes != _render(
                self.num_ranks, self.operations, self.dependencies):
            raise ValueError("rendered GOAL bytes disagree with the structured snapshot")
        if self.sha256 != hashlib.sha256(self.rendered_bytes).hexdigest():
            raise ValueError("rendered GOAL SHA-256 disagrees")


@dataclass(frozen=True)
class GoalActionResult:
    rank: int
    label: str
    operation_id: str | None
    kind: str
    eligible_at_ps: int
    started_at_ps: int
    completed_at_ps: int
    causal_sequences: tuple[int, ...]
    sequence: int | None = None


@dataclass(frozen=True)
class GoalArtifactResult:
    execution_id: str
    artifact_id: str
    snapshot: GoalTraceSnapshot
    released_at_ps: int
    completed_at_ps: int
    actions: tuple[GoalActionResult, ...]
    flow_sequences: Mapping[ActionKey, int]
    completion_rows: tuple[Mapping[str, Any], ...]
    causal_sequences: tuple[int, ...]

    @property
    def accepted_sequences(self) -> tuple[int, ...]:
        return tuple(sorted(self.flow_sequences.values()))

    @property
    def action_by_key(self) -> Mapping[ActionKey, GoalActionResult]:
        return MappingProxyType({(action.rank, action.label): action for action in self.actions})

    @property
    def operation_frontiers(self) -> Mapping[str, Mapping[int, int]]:
        frontiers: dict[str, dict[int, int]] = {}
        for action in self.actions:
            if action.operation_id is not None:
                ranks = frontiers.setdefault(action.operation_id, {})
                ranks[action.rank] = max(ranks.get(action.rank, 0), action.completed_at_ps)
        return MappingProxyType({owner: MappingProxyType(ranks)
                                 for owner, ranks in frontiers.items()})


class GoalSessionError(RuntimeError):
    pass


@dataclass
class _State:
    action: _Action
    dependencies: tuple[ActionKey, ...]
    eligible_at_ps: int | None = None
    started_at_ps: int | None = None
    completed_at_ps: int | None = None
    causal_sequences: set[int] = field(default_factory=set)
    sequence: int | None = None


class GoalSessionExecutor:
    def __init__(self, session: FlowSession, execution_id: str) -> None:
        self.session = session
        self.execution_id = _text(execution_id, "execution_id")
        self._artifacts: set[str] = set()
        self._completed_at_ps = 0
        self._failed = False

    def run(self, snapshot: GoalTraceSnapshot, artifact_id: str, released_at_ps: int,
            predecessor_sequences: Sequence[int] = ()) -> GoalArtifactResult:
        if self._failed:
            raise GoalSessionError("GOAL executor is terminal after an earlier failure")
        try:
            if not isinstance(snapshot, GoalTraceSnapshot):
                raise TypeError("run requires a validated GoalTraceSnapshot")
            _text(artifact_id, "artifact_id")
            _uint(released_at_ps, "released_at_ps")
            if artifact_id in self._artifacts or released_at_ps < self._completed_at_ps:
                raise GoalSessionError("duplicate artifact or backward artifact release")
            prior = tuple(_uint(value, "predecessor sequence", _U32, 1)
                          for value in predecessor_sequences)
            if tuple(sorted(set(prior))) != prior or not set(prior) <= set(
                    self.session.completed_sequences):
                raise GoalSessionError("artifact predecessors must be sorted completed sequences")
            self._artifacts.add(artifact_id)
            result = self._execute(snapshot, artifact_id, released_at_ps, prior)
            self._completed_at_ps = result.completed_at_ps
            return result
        except BaseException:
            self._failed = True
            self.session.abort()
            raise

    def _execute(self, snapshot: GoalTraceSnapshot, artifact_id: str, released_at_ps: int,
                 prior: tuple[int, ...]) -> GoalArtifactResult:
        states = {}
        for operation in snapshot.operations:
            action = _parse(operation, snapshot.num_ranks)
            dependencies = tuple((dep.rank, dep.predecessor_label) for dep in snapshot.dependencies
                                 if (dep.rank, dep.operation_label) == action.key)
            states[action.key] = _State(action, dependencies)
        pairs = {(msg.source_rank, msg.send_label): (msg.destination_rank, msg.receive_label)
                 for msg in snapshot.messages}
        cpu_until = dict.fromkeys(range(snapshot.num_ranks), released_at_ps)
        cpu_ancestry: dict[int, set[int]] = {rank: set() for rank in cpu_until}
        flow_sequences: dict[ActionKey, int] = {}
        physical: dict[int, Mapping[str, Any]] = {}
        now = released_at_ps
        while any(state.completed_at_ps is None for state in states.values()):
            changed = True
            while changed:
                changed = False
                for state in states.values():
                    action = state.action
                    if state.completed_at_ps is not None:
                        continue
                    if (action.kind == "calc" and state.started_at_ps is not None and
                            state.started_at_ps + action.duration_ps <= now):
                        state.completed_at_ps = state.started_at_ps + action.duration_ps
                        changed = True
                    if state.started_at_ps is not None:
                        continue
                    dependencies = [states[key] for key in state.dependencies]
                    if any(dep.completed_at_ps is None for dep in dependencies):
                        continue
                    state.eligible_at_ps = max(
                        [released_at_ps] + [dep.completed_at_ps for dep in dependencies])
                    if cpu_until[action.operation.rank] > now:
                        continue
                    state.started_at_ps = now
                    state.causal_sequences = set(prior) | cpu_ancestry[action.operation.rank]
                    for dependency in dependencies:
                        state.causal_sequences.update(dependency.causal_sequences)
                    if action.kind == "calc":
                        finish = _uint(now + action.duration_ps, "calc completion")
                        cpu_until[action.operation.rank] = finish
                        cpu_ancestry[action.operation.rank] = set(state.causal_sequences)
                    elif action.kind == "send":
                        if now == self.session.boundary_time_ps and not state.causal_sequences:
                            raise GoalSessionError(
                                "exact-boundary injection has no causal completed predecessor")
                        sequence = self.session.inject(
                            execution_id=self.execution_id,
                            operation_id=action.operation.operation_id,
                            flow_id=f"{quote(artifact_id, safe='')}/{action.operation.rank}/"
                                    f"{action.operation.label}",
                            source=action.operation.rank, destination=action.peer,
                            tag=action.tag, payload_bytes=action.payload_bytes,
                            eligible_at_ps=now,
                            predecessor_sequences=tuple(sorted(state.causal_sequences)),
                        )
                        _uint(sequence, "accepted sequence", _U32, 1)
                        if sequence in flow_sequences.values():
                            raise GoalSessionError("native session reused an accepted sequence")
                        state.sequence = sequence
                        flow_sequences[action.key] = sequence
                    changed = True
                for send_key, receive_key in pairs.items():
                    send, receive = states[send_key], states[receive_key]
                    if (send.completed_at_ps is not None or receive.started_at_ps is None or
                            send.sequence not in physical):
                        continue
                    row = physical[send.sequence]
                    completed = max(row["completion_time_ps"], receive.started_at_ps)
                    if completed > now:
                        raise GoalSessionError("physical completion escaped the current frontier")
                    ancestry = send.causal_sequences | receive.causal_sequences | {send.sequence}
                    send.completed_at_ps = receive.completed_at_ps = completed
                    send.causal_sequences = set(ancestry)
                    receive.causal_sequences = set(ancestry)
                    receive.sequence = send.sequence
                    changed = True
            if all(state.completed_at_ps is not None for state in states.values()):
                break
            local_times = [state.started_at_ps + state.action.duration_ps
                           for state in states.values() if state.action.kind == "calc" and
                           state.started_at_ps is not None and state.completed_at_ps is None]
            local = min(local_times) if local_times else None
            pending = self.session.pending_sequences
            if pending:
                # Leave the local timestamp open for independent work at that time.
                update = self.session.await_completion(pending,
                                                       through_ps=None if local is None else local - 1)
                if update.reason == "horizon":
                    if local is None:
                        raise GoalSessionError("native horizon yield has no pending local action")
                    now = local
                elif update.reason == "completion":
                    if not update.completion_rows or update.event_time_ps < now:
                        raise GoalSessionError("native completion made no forward progress")
                    now = update.event_time_ps
                else:
                    raise GoalSessionError("unexpected native completion response reason")
                for row in update.completion_rows:
                    sequence = row["sequence"]
                    keys = [key for key, value in flow_sequences.items() if value == sequence]
                    if not keys:
                        continue
                    if sequence in physical:
                        raise GoalSessionError("native session repeated a completion row")
                    self._check_row(row, states[keys[0]], artifact_id)
                    physical[sequence] = MappingProxyType(dict(row))
            elif local is not None:
                now = local
            else:
                raise GoalSessionError("GOAL action graph stalled before logical completion")
        actions = tuple(GoalActionResult(
            state.action.operation.rank, state.action.operation.label,
            state.action.operation.operation_id, state.action.kind, state.eligible_at_ps,
            state.started_at_ps, state.completed_at_ps, tuple(sorted(state.causal_sequences)),
            state.sequence,
        ) for state in states.values())
        if set(physical) != set(flow_sequences.values()):
            raise GoalSessionError("artifact native completion inventory is incomplete")
        return GoalArtifactResult(
            self.execution_id, artifact_id, snapshot, released_at_ps,
            max(action.completed_at_ps for action in actions), actions,
            MappingProxyType(dict(flow_sequences)), tuple(physical[seq] for seq in sorted(physical)),
            tuple(sorted(set(prior).union(*(state.causal_sequences for state in states.values())))),
        )

    def _check_row(self, row: Mapping[str, Any], state: _State, artifact_id: str) -> None:
        action = state.action
        expected = {
            "execution_id": self.execution_id, "operation_id": action.operation.operation_id,
            "flow_id": f"{quote(artifact_id, safe='')}/{action.operation.rank}/{action.operation.label}",
            "source": action.operation.rank, "destination": action.peer, "tag": action.tag,
            "payload_bytes": action.payload_bytes, "start_time_ps": state.started_at_ps,
            "sequence": state.sequence, "completion_status": "success",
        }
        if any(type(row.get(name)) is not type(value) or row.get(name) != value
               for name, value in expected.items()):
            raise GoalSessionError("native completion row disagrees with its GOAL action")
        finish = _uint(row.get("completion_time_ps"), "native completion_time_ps")
        fct = _uint(row.get("fct_ps"), "native fct_ps")
        if finish < state.started_at_ps or fct != finish - state.started_at_ps:
            raise GoalSessionError("native completion timing does not conserve")
