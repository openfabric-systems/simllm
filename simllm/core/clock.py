"""Virtual clock: deterministic event ordering on a picosecond timeline.

The clock owns simulated time. Producers schedule payloads at absolute
picosecond timestamps; the owner drains events in time order and advances
`now_ps` monotonically. Ties are broken by insertion order so identical
timestamps stay deterministic across runs.
"""

from __future__ import annotations

import heapq
import itertools
from typing import Any


class VirtualClock:
    def __init__(self, start_ps: int = 0):
        self._now_ps = start_ps
        self._heap: list[tuple[int, int, Any]] = []
        self._tie = itertools.count()

    @property
    def now_ps(self) -> int:
        return self._now_ps

    def schedule(self, at_ps: int, payload: Any) -> None:
        if at_ps < self._now_ps:
            raise ValueError(f"cannot schedule at {at_ps} ps, now is {self._now_ps} ps")
        heapq.heappush(self._heap, (at_ps, next(self._tie), payload))

    def peek_next_time(self) -> int | None:
        return self._heap[0][0] if self._heap else None

    def pop_next(self) -> tuple[int, Any]:
        """Advance to and return the next event as (time_ps, payload)."""
        if not self._heap:
            raise IndexError("virtual clock has no pending events")
        at_ps, _, payload = heapq.heappop(self._heap)
        self._now_ps = at_ps
        return at_ps, payload

    def advance_to(self, at_ps: int) -> None:
        """Move time forward with no event (e.g. absorbing a sim result)."""
        if at_ps < self._now_ps:
            raise ValueError(f"cannot rewind to {at_ps} ps from {self._now_ps} ps")
        self._now_ps = at_ps

    def __len__(self) -> int:
        return len(self._heap)
