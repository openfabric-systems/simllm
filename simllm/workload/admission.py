"""Virtual-time request admission without framework batching policy."""

from __future__ import annotations

import enum
from collections.abc import Callable

from simllm.core import (
    BookkeepingScope,
    FrameworkRequestArrival,
    OperationCorrelation,
    ProcessingStage,
    RequestBookkeeper,
    StagePhase,
    StageRecord,
    VirtualClock,
    framework_request_arrivals,
)


class AdmissionMode(str, enum.Enum):
    """Whether request creation time gates framework entry."""

    ALL_AT_ONCE = "all-at-once"
    ARRIVAL_GATED = "arrival-gated"


class RequestAdmissionGate:
    """Release bookkeeping-backed requests to a caller-owned engine loop.

    The gate owns no batching decision. ``admit_ready`` calls the supplied
    framework-entry callback in deterministic arrival order. The caller then
    lets its framework scheduler choose the next batch. In gated mode, a
    successful callback is projected as one scheduler-entry bookkeeping fact.

    ``ALL_AT_ONCE`` is the identity off mode. It preserves ledger order,
    releases every request on the first call, and mutates neither the virtual
    clock nor the bookkeeping ledger.
    """

    def __init__(
        self,
        clock: VirtualClock,
        bookkeeper: RequestBookkeeper,
        *,
        mode: AdmissionMode = AdmissionMode.ALL_AT_ONCE,
    ) -> None:
        if not isinstance(clock, VirtualClock):
            raise TypeError("clock must be a VirtualClock")
        if not isinstance(bookkeeper, RequestBookkeeper):
            raise TypeError("bookkeeper must be a RequestBookkeeper")
        if not isinstance(mode, AdmissionMode):
            try:
                mode = AdmissionMode(mode)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"unsupported admission mode {mode!r}") from exc
        arrivals = framework_request_arrivals(bookkeeper.snapshot())
        if mode is AdmissionMode.ARRIVAL_GATED:
            arrivals = tuple(
                sorted(
                    arrivals,
                    key=lambda arrival: (arrival.arrived_at_ps, arrival.sequence),
                )
            )
        self.clock = clock
        self.bookkeeper = bookkeeper
        self.mode = mode
        self._pending = list(arrivals)
        self._admitted: list[FrameworkRequestArrival] = []

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    @property
    def pending_request_ids(self) -> tuple[str, ...]:
        return tuple(arrival.request_id for arrival in self._pending)

    @property
    def admitted_request_ids(self) -> tuple[str, ...]:
        return tuple(arrival.request_id for arrival in self._admitted)

    @property
    def next_arrival_ps(self) -> int | None:
        if not self._pending:
            return None
        if self.mode is AdmissionMode.ALL_AT_ONCE:
            return self.clock.now_ps
        return self._pending[0].arrived_at_ps

    def ready(self) -> tuple[FrameworkRequestArrival, ...]:
        """Return the next releasable prefix without changing gate state."""

        if self.mode is AdmissionMode.ALL_AT_ONCE:
            return tuple(self._pending)
        count = 0
        for arrival in self._pending:
            if arrival.arrived_at_ps > self.clock.now_ps:
                break
            count += 1
        return tuple(self._pending[:count])

    def advance_to_next_arrival(self) -> int:
        """Advance an idle gated loop to its next external arrival."""

        if self.mode is AdmissionMode.ALL_AT_ONCE:
            raise RuntimeError("all-at-once admission has no future arrival gate")
        if not self._pending:
            raise IndexError("admission gate has no pending requests")
        if self.ready():
            raise RuntimeError("admit ready requests before advancing the clock")
        self.clock.advance_to(self._pending[0].arrived_at_ps)
        return self.clock.now_ps

    def admit_ready(
        self,
        add_request: Callable[[FrameworkRequestArrival], object],
    ) -> tuple[FrameworkRequestArrival, ...]:
        """Call ``add_request`` for the ready prefix and commit each handoff."""

        if not callable(add_request):
            raise TypeError("add_request must be callable")
        ready = self.ready()
        admitted: list[FrameworkRequestArrival] = []
        for arrival in ready:
            add_request(arrival)
            if self.mode is AdmissionMode.ARRIVAL_GATED:
                self.bookkeeper.append(
                    StageRecord(
                        stage=ProcessingStage.SCHEDULER,
                        phase=StagePhase.ENTERED,
                        timestamp_ps=self.clock.now_ps,
                        scope=BookkeepingScope(
                            correlation=self._request_correlation(arrival)
                        ),
                        object_refs=(arrival.request_ref,),
                    )
                )
            popped = self._pending.pop(0)
            if popped is not arrival:
                raise RuntimeError("admission pending order changed unexpectedly")
            self._admitted.append(arrival)
            admitted.append(arrival)
        return tuple(admitted)

    @staticmethod
    def _request_correlation(
        arrival: FrameworkRequestArrival,
    ) -> OperationCorrelation:
        return OperationCorrelation(request_ids=(arrival.request_id,))


__all__ = ["AdmissionMode", "RequestAdmissionGate"]
