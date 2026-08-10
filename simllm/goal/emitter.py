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

    def rank(self, r: int) -> RankProgram:
        return self._ranks[r]

    def render(self) -> str:
        parts = [f"num_ranks {self.num_ranks}"]
        parts += [rp.render() for rp in self._ranks]
        return "\n".join(parts) + "\n"

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(self.render(), newline="\n")
        return path
