"""Per-request routing ownership and end-to-end close-out.

The records in this module are framework neutral.  They retain only opaque
join provenance and an arena view descriptor; the core never imports the
pre-play representation that produced either value.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from simllm.core.execution import (
    CollectiveWork,
    EventPhase,
    ExecutionGraph,
    ExecutionResult,
)
from simllm.core.step import StepRecord

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PHASES = ("dispatch", "combine")


class RequestLifecycleError(RuntimeError):
    """A request routing view violated its fail-closed lifetime contract."""


def _nonblank(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True, kw_only=True)
class JoinProvenance:
    """Opaque identities of the joined run and its source trace."""

    run_schema: str
    trace_schema: str
    trace_sha256: str

    def __post_init__(self) -> None:
        _nonblank(self.run_schema, "run_schema")
        _nonblank(self.trace_schema, "trace_schema")
        if not isinstance(self.trace_sha256, str) or _SHA256.fullmatch(
            self.trace_sha256
        ) is None:
            raise ValueError(
                "trace_sha256 must contain 64 lowercase hexadecimal digits"
            )


@dataclass(frozen=True, kw_only=True)
class RoutingViewDescriptor:
    """One zero-copy request slice in an externally owned routing arena."""

    arena_id: str
    token_offset: int
    token_count: int
    prompt_token_count: int
    release_callback: Callable[[], None] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _nonblank(self.arena_id, "arena_id")
        _nonnegative_integer(self.token_offset, "token_offset")
        count = _nonnegative_integer(self.token_count, "token_count")
        prompt = _nonnegative_integer(
            self.prompt_token_count,
            "prompt_token_count",
        )
        if count == 0:
            raise ValueError("token_count must be positive")
        if prompt == 0 or prompt > count:
            raise ValueError("prompt_token_count must be in [1, token_count]")
        if not callable(self.release_callback):
            raise TypeError("release_callback must be callable")


class RequestLifecycleState(enum.Enum):
    """The only legal lifecycle sequence for a joined request."""

    JOINED = "joined"
    ADMITTED = "admitted"
    EXECUTING = "executing"
    FINISH_FLAGGED = "finish-flagged"
    DRAINED = "drained"
    CLOSED = "closed"


@dataclass
class RequestRoutingLifetime:
    """The single mutable identity and routing lifetime of one request."""

    request_id: str
    join_provenance: JoinProvenance
    arrived_at_ps: int
    routing_view: RoutingViewDescriptor
    moe_layer_indices: tuple[int, ...]
    state: RequestLifecycleState = RequestLifecycleState.JOINED
    consumption_cursor: int = 0
    scheduler_finished: bool = False
    dispatch_end_mask: int = 0
    combine_end_mask: int = 0
    view_released: bool = False

    def __post_init__(self) -> None:
        _nonblank(self.request_id, "request_id")
        if not isinstance(self.join_provenance, JoinProvenance):
            raise TypeError("join_provenance must be JoinProvenance")
        _nonnegative_integer(self.arrived_at_ps, "arrived_at_ps")
        if not isinstance(self.routing_view, RoutingViewDescriptor):
            raise TypeError("routing_view must be RoutingViewDescriptor")
        if not isinstance(self.moe_layer_indices, tuple):
            raise TypeError("moe_layer_indices must be a tuple")
        if not self.moe_layer_indices or len(self.moe_layer_indices) > 64:
            raise ValueError("moe_layer_indices must contain between 1 and 64 layers")
        for index, layer in enumerate(self.moe_layer_indices):
            _nonnegative_integer(layer, f"moe_layer_indices[{index}]")
        if tuple(sorted(set(self.moe_layer_indices))) != self.moe_layer_indices:
            raise ValueError("moe_layer_indices must be unique and increasing")
        if not isinstance(self.state, RequestLifecycleState):
            raise TypeError("state must be RequestLifecycleState")
        cursor = _nonnegative_integer(
            self.consumption_cursor,
            "consumption_cursor",
        )
        if cursor > self.token_count:
            raise ValueError("consumption_cursor exceeds token_count")

    @property
    def arena_id(self) -> str:
        return self.routing_view.arena_id

    @property
    def provenance(self) -> JoinProvenance:
        return self.join_provenance

    @property
    def view(self) -> RoutingViewDescriptor:
        return self.routing_view

    @property
    def token_offset(self) -> int:
        return self.routing_view.token_offset

    @property
    def token_count(self) -> int:
        return self.routing_view.token_count

    @property
    def prompt_token_count(self) -> int:
        return self.routing_view.prompt_token_count

    @property
    def full_end_mask(self) -> int:
        return (1 << len(self.moe_layer_indices)) - 1

    @property
    def dispatch_complete(self) -> bool:
        return self.dispatch_end_mask == self.full_end_mask

    @property
    def combine_complete(self) -> bool:
        return self.combine_end_mask == self.full_end_mask

    @property
    def routing_consumed(self) -> bool:
        return self.consumption_cursor == self.token_count

    @property
    def drained(self) -> bool:
        return (
            self.routing_consumed
            and self.dispatch_complete
            and self.combine_complete
        )

    @property
    def execution_drained(self) -> bool:
        return self.drained

    def missing_layers(self, phase: str) -> tuple[int, ...]:
        """Return model layers whose final-token end flag is absent."""

        if phase == "dispatch":
            mask = self.dispatch_end_mask
        elif phase == "combine":
            mask = self.combine_end_mask
        else:
            raise ValueError("phase must be 'dispatch' or 'combine'")
        return tuple(
            layer
            for bit, layer in enumerate(self.moe_layer_indices)
            if not mask & (1 << bit)
        )

    def release_view(self) -> None:
        """Release the arena slice once, and never before CLOSED."""

        if self.state is not RequestLifecycleState.CLOSED:
            raise RequestLifecycleError(
                f"request {self.request_id!r}: routing view release before CLOSED "
                f"from state {self.state.value}"
            )
        if self.view_released:
            return
        self.routing_view.release_callback()
        self.view_released = True


class RequestLifetimeRegistry:
    """Own and atomically advance all joined request lifetime records."""

    def __init__(self, moe_layer_indices: tuple[int, ...]) -> None:
        if not isinstance(moe_layer_indices, tuple):
            raise TypeError("moe_layer_indices must be a tuple")
        if not moe_layer_indices or len(moe_layer_indices) > 64:
            raise ValueError("moe_layer_indices must contain between 1 and 64 layers")
        for index, layer in enumerate(moe_layer_indices):
            _nonnegative_integer(layer, f"moe_layer_indices[{index}]")
        if tuple(sorted(set(moe_layer_indices))) != moe_layer_indices:
            raise ValueError("moe_layer_indices must be unique and increasing")
        self.moe_layer_indices = moe_layer_indices
        self._records: dict[str, RequestRoutingLifetime] = {}
        self._consumed_execution_ids: set[str] = set()

    def register(
        self,
        request_id: str,
        provenance: JoinProvenance,
        arrived_at_ps: int,
        view: RoutingViewDescriptor,
    ) -> RequestRoutingLifetime:
        """Register one joined view without admitting it to execution."""

        _nonblank(request_id, "request_id")
        if request_id in self._records:
            raise ValueError(f"duplicate request lifetime {request_id!r}")
        lifetime = RequestRoutingLifetime(
            request_id=request_id,
            join_provenance=provenance,
            arrived_at_ps=arrived_at_ps,
            routing_view=view,
            moe_layer_indices=self.moe_layer_indices,
        )
        self._records[request_id] = lifetime
        return lifetime

    def by_request_id(self, request_id: str) -> RequestRoutingLifetime:
        """Return the one mutable lifetime with this joined identity."""

        try:
            return self._records[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown request lifetime {request_id!r}") from exc

    @property
    def requests(self) -> tuple[RequestRoutingLifetime, ...]:
        """Return registered lifetimes in stable join order."""

        return tuple(self._records.values())

    @property
    def closed_request_count(self) -> int:
        return sum(
            record.state is RequestLifecycleState.CLOSED
            for record in self._records.values()
        )

    @property
    def live_request_count(self) -> int:
        return len(self._records) - self.closed_request_count

    @property
    def live_view_count(self) -> int:
        return sum(not record.view_released for record in self._records.values())

    @staticmethod
    def _validate_step_contract(
        record: StepRecord,
        graph: ExecutionGraph,
        result: ExecutionResult,
    ) -> dict[str, object]:
        if not isinstance(record, StepRecord):
            raise TypeError("record must be a StepRecord")
        if not isinstance(graph, ExecutionGraph):
            raise TypeError("graph must be an ExecutionGraph")
        if not isinstance(result, ExecutionResult):
            raise TypeError("result must be an ExecutionResult")
        if graph.step_index != record.step_index:
            raise ValueError("graph step index disagrees with StepRecord")
        if graph.released_at_ps != record.virtual_time_ps:
            raise ValueError("graph release disagrees with StepRecord virtual time")
        if result.execution_id != graph.execution_id:
            raise ValueError("ExecutionResult execution ID disagrees with graph")
        operations = {operation.operation_id: operation for operation in graph.operations}
        if len(operations) != len(graph.operations):
            raise ValueError("ExecutionGraph contains duplicate operation IDs")
        for event in result.events:
            if event.execution_id != graph.execution_id:
                raise ValueError("completion event execution ID disagrees with graph")
            if event.operation_id not in operations:
                raise ValueError("completion event names an unknown operation")
        return operations

    @staticmethod
    def _cover_scheduled_tokens(
        lifetime: RequestRoutingLifetime,
        scheduled: object,
    ) -> bool:
        num_new_tokens = _nonnegative_integer(
            getattr(scheduled, "num_new_tokens", None),
            "scheduled.num_new_tokens",
        )
        context_length = _nonnegative_integer(
            getattr(scheduled, "context_length", None),
            "scheduled.context_length",
        )
        cached = _nonnegative_integer(
            getattr(scheduled, "num_cached_tokens", None),
            "scheduled.num_cached_tokens",
        )
        if num_new_tokens == 0:
            return False
        start = context_length - num_new_tokens
        if start < 0:
            raise ValueError(
                f"request {lifetime.request_id!r} schedules more tokens than its context"
            )
        if context_length > lifetime.token_count:
            raise RequestLifecycleError(
                f"request {lifetime.request_id!r} cursor overflow: "
                f"{context_length}/{lifetime.token_count}"
            )
        if start > lifetime.consumption_cursor and not (
            lifetime.consumption_cursor == 0 and cached == start
        ):
            raise RequestLifecycleError(
                f"request {lifetime.request_id!r} has uncovered token gap "
                f"[{lifetime.consumption_cursor}, {start})"
            )
        previous = lifetime.consumption_cursor
        lifetime.consumption_cursor = max(previous, context_length)
        return previous < lifetime.token_count == lifetime.consumption_cursor

    @staticmethod
    def _enter_execution(lifetime: RequestRoutingLifetime) -> None:
        if lifetime.state is RequestLifecycleState.CLOSED:
            raise RequestLifecycleError(
                f"closed request {lifetime.request_id!r} was scheduled again"
            )
        if lifetime.scheduler_finished:
            raise RequestLifecycleError(
                f"finished request {lifetime.request_id!r} was scheduled again"
            )
        if lifetime.state is RequestLifecycleState.JOINED:
            lifetime.state = RequestLifecycleState.ADMITTED
        if lifetime.state is RequestLifecycleState.ADMITTED:
            lifetime.state = RequestLifecycleState.EXECUTING

    def _mark_end_flag(
        self,
        record: RequestRoutingLifetime,
        phase: str,
        layer: int,
    ) -> None:
        """Set one end bit; subclasses may suppress a bit for fault injection."""

        try:
            bit = self.moe_layer_indices.index(layer)
        except ValueError as exc:
            raise ValueError(f"MoE completion names unknown model layer {layer}") from exc
        if phase == "dispatch":
            record.dispatch_end_mask |= 1 << bit
        elif phase == "combine":
            record.combine_end_mask |= 1 << bit
        else:
            raise ValueError(f"unknown MoE phase {phase!r}")

    @staticmethod
    def _reconcile(record: RequestRoutingLifetime) -> None:
        if (
            record.scheduler_finished
            and record.state is RequestLifecycleState.EXECUTING
        ):
            record.state = RequestLifecycleState.FINISH_FLAGGED
        if (
            record.state is RequestLifecycleState.FINISH_FLAGGED
            and record.drained
        ):
            record.state = RequestLifecycleState.DRAINED
        if record.state is RequestLifecycleState.DRAINED:
            record.state = RequestLifecycleState.CLOSED

    def consume_step(
        self,
        record: StepRecord,
        graph: ExecutionGraph,
        result: ExecutionResult,
    ) -> None:
        """Atomically consume scheduler coverage and logical MoE completions."""

        if (
            isinstance(graph, ExecutionGraph)
            and graph.execution_id in self._consumed_execution_ids
        ):
            raise ValueError(f"execution ID already consumed: {graph.execution_id!r}")
        operations = self._validate_step_contract(record, graph, result)
        scheduled_ids = [request.request_id for request in record.scheduled]
        if len(scheduled_ids) != len(set(scheduled_ids)):
            raise ValueError("StepRecord.scheduled contains duplicate request IDs")
        if len(record.finished_request_ids) != len(set(record.finished_request_ids)):
            raise ValueError("finished_request_ids contains duplicate request IDs")
        mentioned = (
            set(scheduled_ids)
            | set(record.finished_request_ids)
            | set(record.preempted_request_ids)
        )
        unknown = sorted(mentioned - self._records.keys())
        if unknown:
            raise RequestLifecycleError(
                f"step names requests absent from lifetime registry: {unknown}"
            )

        staged = {
            request_id: replace(lifetime)
            for request_id, lifetime in self._records.items()
        }
        final_reachers: set[str] = set()
        for scheduled in record.scheduled:
            lifetime = staged[scheduled.request_id]
            self._enter_execution(lifetime)
            if self._cover_scheduled_tokens(lifetime, scheduled):
                final_reachers.add(scheduled.request_id)

        for request_id in record.finished_request_ids:
            lifetime = staged[request_id]
            if lifetime.state is RequestLifecycleState.JOINED:
                raise RequestLifecycleError(
                    f"request {request_id!r} finished before admission"
                )
            if lifetime.scheduler_finished:
                raise RequestLifecycleError(
                    f"request {request_id!r} has a duplicate scheduler finish"
                )
            lifetime.scheduler_finished = True

        for event in result.events:
            if event.phase is not EventPhase.COMPLETED or event.subject_object_id is not None:
                continue
            operation = operations[event.operation_id]
            work = operation.work
            if not isinstance(work, CollectiveWork):
                continue
            phase = work.channel_hint
            if (
                work.collective != "all-to-allv"
                or work.algorithm_hint != "pairwise"
                or phase not in _PHASES
            ):
                continue
            layer = operation.correlation.layer
            if layer is None:
                raise ValueError(
                    f"MoE operation {operation.operation_id!r} has no correlation layer"
                )
            if layer not in self.moe_layer_indices:
                raise ValueError(
                    f"MoE operation {operation.operation_id!r} names unknown layer {layer}"
                )
            for request_id in operation.correlation.request_ids:
                if request_id in final_reachers:
                    self._mark_end_flag(staged[request_id], phase, layer)

        for lifetime in staged.values():
            self._reconcile(lifetime)

        mutable_fields = (
            "state",
            "consumption_cursor",
            "scheduler_finished",
            "dispatch_end_mask",
            "combine_end_mask",
        )
        for request_id, staged_record in staged.items():
            target = self._records[request_id]
            for name in mutable_fields:
                setattr(target, name, getattr(staged_record, name))
        self._consumed_execution_ids.add(graph.execution_id)
        for lifetime in self._records.values():
            if lifetime.state is RequestLifecycleState.CLOSED:
                lifetime.release_view()

    def audit_closed(self) -> None:
        """Fail closed at end of run with request-specific missing evidence."""

        failures = []
        for record in self._records.values():
            if record.state is RequestLifecycleState.CLOSED and record.view_released:
                continue
            details = [f"state={record.state.value}"]
            if not record.scheduler_finished:
                details.append("scheduler finish flag missing")
            dispatch = record.missing_layers("dispatch")
            if dispatch:
                details.append(f"dispatch missing layers {list(dispatch)}")
            combine = record.missing_layers("combine")
            if combine:
                details.append(f"combine missing layers {list(combine)}")
            if not record.routing_consumed:
                details.append(
                    f"cursor {record.consumption_cursor}/{record.token_count}"
                )
            if not record.view_released:
                details.append("routing view live")
            failures.append(f"request {record.request_id!r}: " + "; ".join(details))
        if failures or self.live_view_count:
            summary = "; ".join(failures) if failures else "closed view leak"
            raise RequestLifecycleError(
                f"request lifetime audit failed ({summary}); "
                f"live view count={self.live_view_count}"
            )


__all__ = [
    "JoinProvenance",
    "RequestLifecycleError",
    "RequestLifecycleState",
    "RequestLifetimeRegistry",
    "RequestRoutingLifetime",
    "RoutingViewDescriptor",
]
