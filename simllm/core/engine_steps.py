"""One completion authority for independent whole-engine service visits.

The native frontend may prepare one step per engine. This authority advances
the shared clock and permits publication only at that step's due event.
These coarse visits represent declared engine service, not kernel occupancy.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field

from simllm.core.clock import VirtualClock
from simllm.core.execution import CompletionEvent, EventPhase, ResourceKind, ResourceRef
from simllm.core.runtime import QueueVisit
from simllm.core.step import StepRecord, StepResult
from simllm.core.value_snapshot import value_snapshot

INDEPENDENT_ENGINE_AUTHORITY = "simllm-independent-engine-completion-v1"
SERIALIZED_ENGINE_AUTHORITY = "simllm-serialized-engine-service-v1"


@dataclass(frozen=True)
class EngineStepReceipt:
    """Read-only identity and timing, with no independently mutable lifecycle."""

    engine_id: str
    step_index: int
    sequence: int
    submitted_at_ps: int
    completed_at_ps: int


@dataclass
class _Publication:
    publisher: object
    payload: object
    read_values: Callable[[object], object]
    state: tuple


@dataclass(frozen=True)
class PublicationBinding:
    """Name a prepared projection whose actual values the authority freezes."""

    publisher: object
    payload: object
    read_values: Callable[[object], object]


@dataclass
class _PendingStep:
    receipt: EngineStepReceipt
    receipt_state: tuple
    record: StepRecord
    result: StepResult
    record_state: tuple
    result_state: tuple
    guard: Callable[[], None]
    publish: Callable[[StepResult], None]
    token: object
    publications: tuple[_Publication, ...]
    publishers: list[object] = field(default_factory=list)


class EngineStepRuntime:
    """Own pending engine slots, deterministic due order and completed evidence."""

    authority = INDEPENDENT_ENGINE_AUTHORITY

    def __init__(self, clock: VirtualClock):
        if not isinstance(clock, VirtualClock):
            raise TypeError("independent engine completion requires a VirtualClock")
        owner = getattr(clock, "_engine_step_owner", None)
        if owner is not None and owner() is not None:
            raise ValueError("the clock already has an engine completion authority")
        if len(clock):
            raise ValueError("independent engine completion requires an empty event clock")
        self.clock = clock
        clock._engine_step_owner = weakref.ref(self)
        self._now_ps = clock.now_ps
        self._pending: dict[str, _PendingStep] = {}
        self._used: set[tuple[str, int]] = set()
        self._sequence = 0
        self._events: list[CompletionEvent] = []
        self._visits: list[QueueVisit] = []
        self._results: list[tuple[EngineStepReceipt, StepResult]] = []
        self._retiring: _PendingStep | None = None
        self._failure: str | None = None
        self._closed = False

    @property
    def events(self) -> tuple[CompletionEvent, ...]:
        return tuple(self._events)

    @property
    def visits(self) -> tuple[QueueVisit, ...]:
        return tuple(self._visits)

    @property
    def results(self) -> tuple[tuple[EngineStepReceipt, StepResult], ...]:
        return tuple(deepcopy(self._results))

    @property
    def failure(self) -> str | None:
        return self._failure

    def pending_for(self, engine_id: str) -> EngineStepReceipt | None:
        pending = self._pending.get(engine_id)
        return None if pending is None else pending.receipt

    @property
    def next_completion_ps(self) -> int | None:
        self._check_live()
        return self.clock.peek_next_time()

    def invalidate(self, reason: str) -> None:
        """Poison the run after an error; pending native work cannot be retried."""
        if self._failure is None:
            self._failure = reason

    def _reject(self, message: str) -> None:
        self.invalidate(message)
        raise RuntimeError(message)

    def _check_live(self) -> None:
        if self._failure is not None:
            raise RuntimeError("independent engine runtime is poisoned: " + self._failure)
        if self._closed:
            raise RuntimeError("independent engine runtime is closed")
        owner = getattr(self.clock, "_engine_step_owner", None)
        if not callable(owner) or owner() is not self:
            self._reject("engine clock authority was replaced")
        if self.clock.now_ps != self._now_ps:
            self._reject("engine clock advanced outside its completion authority")
        active = [item for item in self._pending.values() if item is not self._retiring]
        for item in self._pending.values():
            try:
                if value_snapshot(item.receipt) != item.receipt_state:
                    self._reject("pending engine receipt values changed")
            except Exception as error:
                self.invalidate(str(error))
                raise
        earliest = min((item.receipt.completed_at_ps for item in active), default=None)
        if len(self.clock) != len(active) or self.clock.peek_next_time() != earliest:
            self._reject("engine event queue changed outside its completion authority")

    def _check_action(self) -> None:
        self._check_live()
        if self._retiring is not None:
            self._reject("engine submission or clock movement during retirement is forbidden")

    def submit(
        self,
        engine_id: str,
        record: StepRecord,
        result: StepResult,
        *,
        guard: Callable[[], None],
        publish: Callable[[StepResult], None],
        publications: tuple[PublicationBinding, ...] = (),
    ) -> EngineStepReceipt:
        """Reserve one free engine slot without publishing its future result."""
        self._check_action()
        try:
            if self.clock.peek_next_time() == self._now_ps:
                raise ValueError("existing due completions must retire before submission")
            if not isinstance(engine_id, str) or not engine_id.strip():
                raise ValueError("engine_id must be a nonblank string")
            if not isinstance(record, StepRecord) or not isinstance(result, StepResult):
                raise TypeError("engine service requires StepRecord and StepResult")
            if engine_id in self._pending:
                raise ValueError("engine already has a pending step")
            key = (engine_id, record.step_index)
            if key in self._used:
                raise ValueError("engine step identity was already submitted")
            if (type(record.step_index) is not int or record.step_index < 0
                    or type(record.virtual_time_ps) is not int or record.virtual_time_ps != self._now_ps):
                raise ValueError("step must be released at the current engine clock")
            result.__post_init__()
            if (result.step_index != record.step_index
                    or result.completed_at_ps != self._now_ps + result.step_latency_ps):
                raise ValueError("engine price disagrees with its input or release")
            if record.scheduled:
                if record.total_new_tokens <= 0 or result.step_latency_ps <= 0:
                    raise ValueError("scheduled engine work must have positive service")
            elif (result.step_latency_ps != 0
                  or not (record.finished_request_ids or record.preempted_request_ids)):
                raise ValueError("zero-service drain requires native completion identities")
            if not callable(guard) or not callable(publish):
                raise TypeError("engine receipt requires validation and publication callbacks")
            record_state, result_state = value_snapshot(record), value_snapshot(result)
            owned_publications: list[_Publication] = []
            for binding in publications:
                if not isinstance(binding, PublicationBinding) or not callable(binding.read_values):
                    raise TypeError("engine publication requires an explicit value reader")
                if any(item.publisher is binding.publisher for item in owned_publications):
                    raise ValueError("engine publisher was bound twice")
                owned_publications.append(_Publication(
                    binding.publisher, binding.payload, binding.read_values,
                    value_snapshot(binding.read_values(binding.payload))))
            guard()
            if value_snapshot(record) != record_state or value_snapshot(result) != result_state:
                raise ValueError("engine guard changed its input or price")
            receipt = EngineStepReceipt(engine_id, record.step_index, self._sequence,
                                        self._now_ps, result.completed_at_ps)
            pending = _PendingStep(receipt, value_snapshot(receipt), record, result, record_state,
                                   result_state, guard, publish, object(), tuple(owned_publications))
            self._check_values(pending)
            self._check_action()
            self._pending[engine_id] = pending
            self._used.add(key)
            self._sequence += 1
            self.clock.schedule(result.completed_at_ps, pending.token)
            for phase in (EventPhase.SUBMITTED, EventPhase.QUEUED, EventPhase.STARTED):
                self._event(receipt, phase, self._now_ps)
            return receipt
        except Exception as error:
            self.invalidate(str(error))
            raise

    def _event(self, receipt: EngineStepReceipt, phase: EventPhase, at_ps: int) -> None:
        self._events.append(CompletionEvent(
            execution_id="engine-service:" + receipt.engine_id,
            operation_id="step-" + str(receipt.step_index), phase=phase, timestamp_ps=at_ps,
            resource=ResourceRef(ResourceKind.GPU_WORK_QUEUE, receipt.engine_id),
        ))

    def advance_to(self, at_ps: int) -> None:
        """Advance to an arrival or deadline, never beyond unfinished work."""
        self._check_action()
        due = self.clock.peek_next_time()
        if type(at_ps) is not int or at_ps < self._now_ps or (due is not None and at_ps > due):
            self._reject("engine clock cannot rewind or skip a completion")
        self.clock.advance_to(at_ps)
        self._now_ps = at_ps

    def claim_publication(
        self, receipt: EngineStepReceipt, *, publisher: object,
        payload: object, record: StepRecord, result: StepResult,
    ) -> None:
        """Authorize one owner's projection inside the sole retirement callback."""
        self._check_live()
        pending = self._retiring
        if (pending is None or pending.receipt is not receipt
                or pending.record is not record or pending.result is not result):
            self._reject("publication is outside its owned engine retirement")
        if any(owner is publisher for owner in pending.publishers):
            self._reject("engine owner already published this receipt")
        if not any(item.publisher is publisher and item.payload is payload for item in pending.publications):
            self._reject("engine publisher or prepared payload is not bound to this receipt")
        try:
            self._check_values(pending)
        except Exception as error:
            self.invalidate(str(error))
            raise
        pending.publishers.append(publisher)

    def complete(self, receipt: EngineStepReceipt) -> None:
        """Consume exactly the next due receipt and permit native publication."""
        self._check_action()
        pending = self._pending.get(receipt.engine_id) if isinstance(receipt, EngineStepReceipt) else None
        if pending is None or pending.receipt is not receipt:
            self._reject("foreign or already consumed engine receipt")
        next_pending = min(self._pending.values(), key=lambda item: (
            item.receipt.completed_at_ps, item.receipt.sequence))
        if pending is not next_pending or receipt.completed_at_ps != self._now_ps:
            self._reject("engine retirement is early or out of due order")
        try:
            self._check_values(pending)
            pending.guard()
            self._check_values(pending)
            self._check_action()
            at_ps, token = self.clock.pop_next()
            if token is not pending.token or at_ps != self._now_ps:
                raise ValueError("foreign engine completion event")
            self._retiring = pending
            pending.publish(pending.result)
            self._check_live()
            self._check_values(pending)
            if len(pending.publishers) != len(pending.publications):
                raise ValueError("engine retirement omitted a bound publication")
            resource = ResourceRef(ResourceKind.GPU_WORK_QUEUE, receipt.engine_id)
            self._visits.append(QueueVisit(
                "engine-service:" + receipt.engine_id, "step-" + str(receipt.step_index), resource,
                receipt.submitted_at_ps, receipt.submitted_at_ps, receipt.submitted_at_ps,
                at_ps, at_ps, stage="declared-whole-engine-service"))
            self._event(receipt, EventPhase.PROGRESS, at_ps)
            self._event(receipt, EventPhase.COMPLETED, at_ps)
            self._results.append((receipt, deepcopy(pending.result)))
            del self._pending[receipt.engine_id]
            self._retiring = None
        except Exception as error:
            self.invalidate(str(error))
            raise

    @staticmethod
    def _check_values(pending: _PendingStep) -> None:
        if (value_snapshot(pending.receipt) != pending.receipt_state
                or value_snapshot(pending.record) != pending.record_state
                or value_snapshot(pending.result) != pending.result_state):
            raise ValueError("pending engine receipt, input or price changed")
        for item in pending.publications:
            if (not any(owner is item.publisher for owner in pending.publishers)
                    and value_snapshot(item.read_values(item.payload)) != item.state):
                raise ValueError("pending engine prepared publication values changed")

    def complete_due(self) -> tuple[EngineStepReceipt, ...]:
        """Retire all existing same-time completions before new admissions."""
        self._check_action()
        due = sorted((item.receipt for item in self._pending.values()
                      if item.receipt.completed_at_ps == self._now_ps), key=lambda item: item.sequence)
        for receipt in due:
            self.complete(receipt)
        return tuple(due)

    def close(self) -> None:
        if self._failure is not None and not self._pending and not len(self.clock):
            # A failed installation with no reservations can release its own
            # clock claim. The failure remains recorded and cannot be retried.
            owner = getattr(self.clock, "_engine_step_owner", None)
            if callable(owner) and owner() is self:
                self._closed = True
                del self.clock._engine_step_owner
                return
        self._check_action()
        if self._pending:
            self._reject("pending engine work cannot be cancelled or reset (CORE-69)")
        self._closed = True
        del self.clock._engine_step_owner
