"""Minimal GOAL trace writer.

Emits the text grammar consumed by ATLAHS/LogGOPSim ``txt2bin``::

    num_ranks <N>
    rank <r> {
      l0: calc <cost>
      l1: send <size>b to <peer> tag <t>
      l2: recv <size>b from <peer> tag <t>
      l1 requires l0     // l1 starts after l0 finishes
      l2 irequires l1    // l2 starts after l1 starts
    }

Optional ``cpu <c>`` / ``nic <n>`` clauses pin operations to resources and
default to 0 when omitted. Convert to the binary format with
``txt2bin -i trace.goal -o trace.bin`` before handing it to a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


@dataclass(frozen=True)
class GoalOperation:
    """One rendered GOAL operation and its optional semantic owner."""

    rank: int
    label: str
    text: str
    operation_id: str | None = None


class GoalDependencyKind(str, Enum):
    """Authority behind one rendered GOAL dependency."""

    EXECUTION_GRAPH = "execution-graph"
    COLLECTIVE_INTERNAL = "collective-internal"


@dataclass(frozen=True)
class GoalGraphEdge:
    """Stable identity of an execution-graph edge projected into GOAL."""

    predecessor_id: str
    operation_id: str
    scope: str
    origin: str
    participant_rank: int | None = None

    def __post_init__(self) -> None:
        _text(self.predecessor_id, "edge.predecessor_id")
        _text(self.operation_id, "edge.operation_id")
        if self.scope not in {"whole-operation", "participant-local"}:
            raise ValueError("edge.scope: expected whole-operation or participant-local")
        if self.origin not in {"explicit", "logical-queue-fifo"}:
            raise ValueError("edge.origin: expected explicit or logical-queue-fifo")
        if self.scope == "participant-local":
            if self.participant_rank is None:
                raise ValueError("edge.participant_rank: required for participant-local edge")
            _integer(self.participant_rank, "edge.participant_rank")
        elif self.participant_rank is not None:
            raise ValueError("edge.participant_rank: forbidden for whole-operation edge")


@dataclass(frozen=True)
class GoalDependencyProvenance:
    """Semantic ownership of a rendered GOAL dependency."""

    kind: GoalDependencyKind
    operation_id: str
    graph_edges: tuple[GoalGraphEdge, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, GoalDependencyKind):
            raise TypeError("dependency.kind: expected GoalDependencyKind")
        _text(self.operation_id, "dependency.operation_id")
        if not isinstance(self.graph_edges, tuple):
            raise TypeError("dependency.graph_edges: in-memory contract requires a tuple")
        for index, edge in enumerate(self.graph_edges):
            if not isinstance(edge, GoalGraphEdge):
                raise TypeError(f"dependency.graph_edges[{index}]: expected GoalGraphEdge")
            if edge.operation_id != self.operation_id:
                raise ValueError(
                    "dependency.graph_edges: target operation does not match provenance owner"
                )
        if self.kind is GoalDependencyKind.EXECUTION_GRAPH and not self.graph_edges:
            raise ValueError("dependency.graph_edges: graph provenance requires at least one edge")
        if self.kind is GoalDependencyKind.COLLECTIVE_INTERNAL and self.graph_edges:
            raise ValueError("dependency.graph_edges: collective-internal provenance forbids edges")


@dataclass(frozen=True)
class GoalDependency:
    """One rendered dependency plus optional semantic provenance."""

    rank: int
    operation_label: str
    predecessor_label: str
    relation: str
    provenance: GoalDependencyProvenance | None = None

    def __post_init__(self) -> None:
        _integer(self.rank, "dependency.rank")
        _text(self.operation_label, "dependency.operation_label")
        _text(self.predecessor_label, "dependency.predecessor_label")
        if self.relation not in {"requires", "irequires"}:
            raise ValueError("dependency.relation: expected requires or irequires")
        if self.provenance is not None and not isinstance(
            self.provenance, GoalDependencyProvenance
        ):
            raise TypeError("dependency.provenance: expected GoalDependencyProvenance")

    def render(self) -> str:
        return f"{self.operation_label} {self.relation} {self.predecessor_label}"


def _integer(value: object, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path}: expected an integer")
    if value < minimum:
        raise ValueError(f"{path}: expected a value of at least {minimum}")
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: expected a nonblank string")
    return value


@dataclass(frozen=True)
class GoalMessage:
    """One rendered physical message plus its optional request partition.

    ``payload_bytes`` remains the physical GOAL service demand. The optional
    ``request_payload_bytes`` tuple is a read-only partition of that demand;
    it is not emitted into GOAL text and therefore cannot change packetization
    or timing. Each entry is ``(request_id, bytes)``.
    """

    operation_id: str | None
    source_rank: int
    destination_rank: int
    payload_bytes: int
    tag: int
    send_label: str
    receive_label: str
    request_payload_bytes: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.operation_id is not None:
            _text(self.operation_id, "message.operation_id")
        _integer(self.source_rank, "message.source_rank")
        _integer(self.destination_rank, "message.destination_rank")
        if self.source_rank == self.destination_rank:
            raise ValueError("message ranks must differ")
        _integer(self.payload_bytes, "message.payload_bytes", minimum=1)
        _integer(self.tag, "message.tag")
        _text(self.send_label, "message.send_label")
        _text(self.receive_label, "message.receive_label")
        if not isinstance(self.request_payload_bytes, tuple):
            raise TypeError("message.request_payload_bytes: in-memory contract requires a tuple")
        request_ids = []
        for index, entry in enumerate(self.request_payload_bytes):
            path = f"message.request_payload_bytes[{index}]"
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError(f"{path}: expected a two-item tuple")
            request_ids.append(_text(entry[0], f"{path}[0]"))
            _integer(entry[1], f"{path}[1]", minimum=1)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("message.request_payload_bytes: duplicate request identity")
        if request_ids != sorted(request_ids):
            raise ValueError("message.request_payload_bytes: entries must be request-major")
        if self.request_payload_bytes:
            if self.operation_id is None:
                raise ValueError("message.operation_id: required for request-attributed traffic")
            attributed = sum(size for _, size in self.request_payload_bytes)
            if attributed != self.payload_bytes:
                raise ValueError(
                    "message.request_payload_bytes: partition sums to "
                    f"{attributed}, expected physical payload {self.payload_bytes}"
                )


@dataclass
class RankProgram:
    """Operations and dependencies of a single rank."""

    rank: int
    _ops: list[GoalOperation] = field(default_factory=list)
    _deps: list[GoalDependency] = field(default_factory=list)

    def _add(self, text: str, operation_id: str | None = None) -> str:
        if operation_id is not None:
            _text(operation_id, "operation_id")
        label = f"r{self.rank}op{len(self._ops)}"
        self._ops.append(GoalOperation(self.rank, label, text, operation_id))
        return label

    def calc(
        self,
        cost: int,
        cpu: int | None = None,
        *,
        operation_id: str | None = None,
    ) -> str:
        suffix = f" cpu {cpu}" if cpu is not None else ""
        return self._add(f"calc {cost}{suffix}", operation_id)

    def send(
        self,
        size_bytes: int,
        to: int,
        tag: int = 0,
        nic: int | None = None,
        *,
        operation_id: str | None = None,
    ) -> str:
        suffix = f" nic {nic}" if nic is not None else ""
        return self._add(f"send {size_bytes}b to {to} tag {tag}{suffix}", operation_id)

    def recv(
        self,
        size_bytes: int,
        source: int,
        tag: int = 0,
        nic: int | None = None,
        *,
        operation_id: str | None = None,
    ) -> str:
        suffix = f" nic {nic}" if nic is not None else ""
        return self._add(f"recv {size_bytes}b from {source} tag {tag}{suffix}", operation_id)

    def requires(
        self,
        op: str,
        after: str,
        *,
        provenance: GoalDependencyProvenance | None = None,
    ) -> None:
        """``op`` starts only after ``after`` has finished."""
        self._record_dependency(op, after, "requires", provenance)

    def irequires(
        self,
        op: str,
        after: str,
        *,
        provenance: GoalDependencyProvenance | None = None,
    ) -> None:
        """``op`` starts only after ``after`` has started."""
        self._record_dependency(op, after, "irequires", provenance)

    def _record_dependency(
        self,
        op: str,
        after: str,
        relation: str,
        provenance: GoalDependencyProvenance | None,
    ) -> None:
        labels = {operation.label for operation in self._ops}
        if op not in labels:
            raise ValueError(f"dependency operation label {op!r} is not present on rank {self.rank}")
        if after not in labels:
            raise ValueError(
                f"dependency predecessor label {after!r} is not present on rank {self.rank}"
            )
        self._deps.append(GoalDependency(self.rank, op, after, relation, provenance))

    def render(self) -> str:
        lines = [f"rank {self.rank} {{"]
        lines += [f"{op.label}: {op.text}" for op in self._ops]
        lines += [dependency.render() for dependency in self._deps]
        lines.append("}")
        return "\n".join(lines)


class GoalTrace:
    """A GOAL schedule across ``num_ranks`` ranks."""

    def __init__(self, num_ranks: int):
        if num_ranks < 1:
            raise ValueError("num_ranks must be >= 1")
        self.num_ranks = num_ranks
        self._ranks = [RankProgram(r) for r in range(num_ranks)]
        self._messages: list[GoalMessage] = []

    def rank(self, r: int) -> RankProgram:
        return self._ranks[r]

    @property
    def messages(self) -> tuple[GoalMessage, ...]:
        """Return the immutable structured projection of rendered messages."""

        return tuple(self._messages)

    @property
    def dependencies(self) -> tuple[GoalDependency, ...]:
        """Return rendered dependencies with their semantic provenance."""

        return tuple(dependency for rank in self._ranks for dependency in rank._deps)

    @property
    def operations(self) -> tuple[GoalOperation, ...]:
        """Return rendered operations with their semantic owners."""

        return tuple(operation for rank in self._ranks for operation in rank._ops)

    def record_message(self, message: GoalMessage) -> None:
        """Attach one validated physical-message projection to this trace."""

        if not isinstance(message, GoalMessage):
            raise TypeError("message: expected GoalMessage")
        for field_name, rank in (
            ("source_rank", message.source_rank),
            ("destination_rank", message.destination_rank),
        ):
            if rank >= self.num_ranks:
                raise ValueError(
                    f"message.{field_name}: rank {rank} is outside num_ranks={self.num_ranks}"
                )
        source_labels = {operation.label for operation in self._ranks[message.source_rank]._ops}
        destination_labels = {
            operation.label for operation in self._ranks[message.destination_rank]._ops
        }
        if message.send_label not in source_labels:
            raise ValueError("message.send_label: not present on the source rank")
        if message.receive_label not in destination_labels:
            raise ValueError("message.receive_label: not present on the destination rank")
        self._messages.append(message)

    def render(self) -> str:
        parts = [f"num_ranks {self.num_ranks}"]
        parts += [rp.render() for rp in self._ranks]
        return "\n".join(parts) + "\n"

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(self.render(), newline="\n")
        return path
