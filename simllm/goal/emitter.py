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
from pathlib import Path


@dataclass
class _Op:
    label: str
    text: str


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
    _ops: list[_Op] = field(default_factory=list)
    _deps: list[str] = field(default_factory=list)

    def _add(self, text: str) -> str:
        label = f"r{self.rank}op{len(self._ops)}"
        self._ops.append(_Op(label, text))
        return label

    def calc(self, cost: int, cpu: int | None = None) -> str:
        suffix = f" cpu {cpu}" if cpu is not None else ""
        return self._add(f"calc {cost}{suffix}")

    def send(self, size_bytes: int, to: int, tag: int = 0, nic: int | None = None) -> str:
        suffix = f" nic {nic}" if nic is not None else ""
        return self._add(f"send {size_bytes}b to {to} tag {tag}{suffix}")

    def recv(self, size_bytes: int, source: int, tag: int = 0, nic: int | None = None) -> str:
        suffix = f" nic {nic}" if nic is not None else ""
        return self._add(f"recv {size_bytes}b from {source} tag {tag}{suffix}")

    def requires(self, op: str, after: str) -> None:
        """``op`` starts only after ``after`` has finished."""
        self._deps.append(f"{op} requires {after}")

    def irequires(self, op: str, after: str) -> None:
        """``op`` starts only after ``after`` has started."""
        self._deps.append(f"{op} irequires {after}")

    def render(self) -> str:
        lines = [f"rank {self.rank} {{"]
        lines += [f"{op.label}: {op.text}" for op in self._ops]
        lines += self._deps
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
