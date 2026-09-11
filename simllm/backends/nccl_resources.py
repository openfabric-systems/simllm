"""Finite GPU residency and service on the retained physical event calendar."""

from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class NcclGpuProfile:
    """Explicit instruction-cycle and shared-service inputs, without hidden fits."""

    available_sms: int
    clock_hz: int
    memory_bytes_per_second: int
    block_setup_cycles: int
    warp_issue_cycles: int
    barrier_cycles_per_warp: int
    poll_issue_cycles: int
    poll_interval_cycles: int
    publication_cycles: int
    kernel_entry_cycles: int
    registers_per_thread: int
    shared_bytes_per_block: int
    blocks_per_sm: int
    warps_per_sm: int
    registers_per_sm: int
    shared_bytes_per_sm: int

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{item.name} must be a nonnegative integer")
        if min(self.available_sms, self.clock_hz, self.memory_bytes_per_second,
               self.blocks_per_sm, self.warps_per_sm, self.registers_per_sm,
               self.shared_bytes_per_sm, self.poll_interval_cycles) <= 0:
            raise ValueError("GPU capacities and polling cadence must be positive")

    def cycles_ps(self, cycles: int) -> int:
        return (cycles * 10**12 + self.clock_hz - 1) // self.clock_hz

    def residency(self, warps: int) -> int:
        limits = [self.blocks_per_sm, self.warps_per_sm // warps]
        if self.registers_per_thread:
            limits.append(self.registers_per_sm // (32 * warps * self.registers_per_thread))
        if self.shared_bytes_per_block:
            limits.append(self.shared_bytes_per_sm // self.shared_bytes_per_block)
        return min(limits)


@dataclass(frozen=True)
class NcclResourceVisit:
    block_id: str
    rank: int
    resource: str
    kind: str
    submitted_at_ps: int
    eligible_at_ps: int
    started_at_ps: int
    finished_at_ps: int
    completed_at_ps: int
    units: int

    @property
    def wait_ps(self) -> int:
        return self.started_at_ps - self.eligible_at_ps


class NcclGpuResources:
    """Blocks retain residency while peers stall; ready work shares service."""

    def __init__(self, calendar, profile: NcclGpuProfile) -> None:
        self.calendar = calendar
        self.profile = profile
        self._resident: dict[str, tuple[int, int, int, int]] = {}
        self._pending: dict[int, deque] = {}
        self._cursors: dict[tuple, int] = {}
        self._visits: list[NcclResourceVisit] = []
        self._residency_events: list[dict] = []
        self._identities: set[str] = set()

    @property
    def visits(self) -> tuple[NcclResourceVisit, ...]:
        return tuple(self._visits)

    @property
    def residency_events(self) -> tuple[dict, ...]:
        return tuple(dict(row) for row in self._residency_events)

    def admit(self, block_id: str, rank: int, warps: int, callback: Callable[[], None]) -> None:
        if block_id in self._identities or self.profile.residency(warps) < 1:
            raise ValueError("block identity repeats or its resource footprint cannot reside")
        self._identities.add(block_id)
        self._pending.setdefault(rank, deque()).append((block_id, warps, self.calendar.now_ps, callback))
        self._grant(rank)

    def _grant(self, rank: int) -> None:
        queue = self._pending[rank]
        while queue:
            block, warps, submitted, callback = queue[0]
            selected = None
            for sm in range(self.profile.available_sms):
                occupants = [value for value in self._resident.values() if value[:2] == (rank, sm)]
                used_warps = sum(value[2] for value in occupants)
                if (len(occupants) < self.profile.blocks_per_sm
                        and used_warps + warps <= self.profile.warps_per_sm
                        and (used_warps + warps) * 32 * self.profile.registers_per_thread <= self.profile.registers_per_sm
                        and (len(occupants) + 1) * self.profile.shared_bytes_per_block <= self.profile.shared_bytes_per_sm):
                    selected = sm
                    break
            if selected is None:
                return
            queue.popleft()
            self._resident[block] = rank, selected, warps, self.calendar.now_ps
            self._residency_events.append({"block_id": block, "rank": rank, "sm": selected,
                                              "warps": warps, "at_ps": self.calendar.now_ps,
                                              "kind": "admit", "submitted_at_ps": submitted})
            self.calendar.schedule_callback(self.calendar.now_ps, callback)

    def release(self, block_id: str) -> None:
        rank, sm, warps, started = self._resident.pop(block_id)
        self._residency_events.append({"block_id": block_id, "rank": rank, "sm": sm, "warps": warps,
                                          "at_ps": self.calendar.now_ps, "kind": "release",
                                          "submitted_at_ps": started})
        self._grant(rank)

    def service(self, block_id: str, kind: str, units: int, callback: Callable[[], None],
                *, memory: bool = False) -> None:
        """Issue ready instruction cycles or local bytes, never a whole-CTA fit."""
        if type(units) is not int or units < 0:
            raise ValueError("resource work units must be nonnegative integers")
        rank, sm, _, _ = self._resident[block_id]
        key = (rank, "memory") if memory else (rank, "sm", sm)
        eligible = self.calendar.now_ps
        start = max(eligible, self._cursors.get(key, 0))
        duration = ((units * 10**12 + self.profile.memory_bytes_per_second - 1)
                    // self.profile.memory_bytes_per_second if memory else self.profile.cycles_ps(units))
        finish = start + duration
        self._cursors[key] = finish
        visit = NcclResourceVisit(block_id, rank, ":".join(map(str, key)), kind,
                                  eligible, eligible, start, finish, finish, units)

        def complete():
            self._visits.append(visit)
            callback()

        self.calendar.schedule_callback(finish, complete)

    def evidence(self) -> dict:
        return {"profile": asdict(self.profile), "visits": [asdict(v) for v in self.visits],
                "residency_events": self.residency_events,
                "wait_reduction": "sum of resource visits is work, not wall latency"}
