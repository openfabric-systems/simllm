"""Central append-only request and created-object bookkeeping.

The execution graph is immutable plan truth. Framework adapters, lowerers and
device runtimes append frozen facts here as a request moves through the plan
and concrete objects are created. Native handles remain opaque owner data.
Network bookkeeping deliberately stops at WQE completion; packet identities
and packet lifecycle stay private to the selected backend.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TypeAlias

from simllm.core.execution import (
    CompletionEvent,
    ConfigScalar,
    EventPhase,
    ExecutionGraph,
    OperationCorrelation,
    ResourceKind,
    ResourceRef,
)

BOOKKEEPING_SCHEMA = "simllm-request-bookkeeping-v1"


class ProcessingStage(enum.Enum):
    """Stable request-processing stages shared by all producers."""

    REQUEST = "request"
    SCHEDULER = "scheduler"
    FRAMEWORK_MEMORY = "framework-memory"
    EXECUTION = "execution"
    COLLECTIVE = "collective"
    NETWORK = "network"
    COMPLETION = "completion"


class StagePhase(enum.Enum):
    """Whether a request entered or completed a semantic stage."""

    ENTERED = "entered"
    COMPLETED = "completed"


class CreatedObjectKind(enum.Enum):
    """Portable identities for objects created while processing requests."""

    FRAMEWORK_REQUEST = "framework-request"
    VRAM_ALLOCATION = "vram-allocation"
    EXECUTION_OPERATION = "execution-operation"
    NCCL_COMMAND = "nccl-command"
    SEND_QUEUE = "send-queue"
    RECEIVE_QUEUE = "receive-queue"
    COMPLETION_QUEUE = "completion-queue"
    NETWORK_WQE = "network-wqe"
    DCQCN_QP = "dcqcn-qp"
    RNIC_CN_LINK_PAIR = "rnic-cn-link-pair"


class ObjectOwner(enum.Enum):
    """Layer that owns the lifetime and meaning of a created object."""

    FRAMEWORK = "framework"
    CORE = "core"
    NCCL = "nccl"
    DEVICE_RUNTIME = "device-runtime"
    NETWORK_BACKEND = "network-backend"


_REUSABLE_OBJECT_KINDS = frozenset(
    {
        CreatedObjectKind.VRAM_ALLOCATION,
        CreatedObjectKind.SEND_QUEUE,
        CreatedObjectKind.RECEIVE_QUEUE,
        CreatedObjectKind.COMPLETION_QUEUE,
        CreatedObjectKind.DCQCN_QP,
        CreatedObjectKind.RNIC_CN_LINK_PAIR,
    }
)


ObjectMetadata: TypeAlias = tuple[tuple[str, ConfigScalar], ...]


@dataclass(frozen=True)
class CreatedObjectRef:
    """Typed portable identity for one owner-managed object."""

    kind: CreatedObjectKind
    object_id: str


@dataclass(frozen=True)
class BookkeepingScope:
    """Request, step, execution and operation correlation for one fact."""

    correlation: OperationCorrelation = field(default_factory=OperationCorrelation)
    step_index: int | None = None
    execution_id: str | None = None
    operation_id: str | None = None


@dataclass(frozen=True)
class CreatedObjectRecord:
    """Creation and lineage record for one opaque framework/runtime object."""

    ref: CreatedObjectRef
    owner: ObjectOwner
    created_at_ps: int
    scope: BookkeepingScope = field(default_factory=BookkeepingScope)
    native_id: str | None = None
    parent_refs: tuple[CreatedObjectRef, ...] = ()
    metadata: ObjectMetadata = ()


@dataclass(frozen=True)
class StageRecord:
    """One request-processing stage transition."""

    stage: ProcessingStage
    phase: StagePhase
    timestamp_ps: int
    scope: BookkeepingScope
    object_refs: tuple[CreatedObjectRef, ...] = ()


BookkeepingFact: TypeAlias = CreatedObjectRecord | StageRecord | CompletionEvent


@dataclass(frozen=True)
class BookkeepingEntry:
    """One ledger-assigned sequence number and its immutable fact."""

    sequence: int
    fact: BookkeepingFact


@dataclass(frozen=True)
class BookkeepingLedger:
    """Immutable snapshot of all facts known to one bookkeeper."""

    entries: tuple[BookkeepingEntry, ...] = field(default_factory=tuple)


def _require_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: must be a nonblank string")
    return value


def _require_integer(value: object, path: str, *, nonnegative: bool = False) -> int:
    if type(value) is not int:
        raise ValueError(f"{path}: must be an integer")
    if nonnegative and value < 0:
        raise ValueError(f"{path}: must be nonnegative")
    return value


def _validate_correlation(value: OperationCorrelation, path: str) -> None:
    if not isinstance(value, OperationCorrelation):
        raise TypeError(f"{path}: expected OperationCorrelation")
    if not isinstance(value.request_ids, tuple):
        raise TypeError(f"{path}.request_ids: in-memory contract requires a tuple")
    for index, request_id in enumerate(value.request_ids):
        _require_text(request_id, f"{path}.request_ids[{index}]")
    if len(value.request_ids) != len(set(value.request_ids)):
        raise ValueError(f"{path}.request_ids: contains duplicate values")
    if value.batch_id is not None:
        _require_text(value.batch_id, f"{path}.batch_id")
    for field_name in ("layer", "microbatch", "iteration"):
        field_value = getattr(value, field_name)
        if field_value is not None:
            _require_integer(field_value, f"{path}.{field_name}", nonnegative=True)


def _validate_scope(value: BookkeepingScope, path: str) -> None:
    if not isinstance(value, BookkeepingScope):
        raise TypeError(f"{path}: expected BookkeepingScope")
    _validate_correlation(value.correlation, f"{path}.correlation")
    if value.step_index is not None:
        _require_integer(value.step_index, f"{path}.step_index", nonnegative=True)
    if value.execution_id is not None:
        _require_text(value.execution_id, f"{path}.execution_id")
    if value.operation_id is not None:
        _require_text(value.operation_id, f"{path}.operation_id")
        if value.execution_id is None:
            raise ValueError(f"{path}.operation_id: requires execution_id")


def _validate_ref(value: CreatedObjectRef, path: str) -> None:
    if not isinstance(value, CreatedObjectRef):
        raise TypeError(f"{path}: expected CreatedObjectRef")
    if not isinstance(value.kind, CreatedObjectKind):
        raise TypeError(f"{path}.kind: expected CreatedObjectKind")
    _require_text(value.object_id, f"{path}.object_id")


def _validate_ref_tuple(values: tuple[CreatedObjectRef, ...], path: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{path}: in-memory contract requires a tuple")
    for index, value in enumerate(values):
        _validate_ref(value, f"{path}[{index}]")
    if len(values) != len(set(values)):
        raise ValueError(f"{path}: contains duplicate values")


def _validate_metadata(values: ObjectMetadata, path: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{path}: in-memory contract requires a tuple")
    names: list[str] = []
    for index, entry in enumerate(values):
        entry_path = f"{path}[{index}]"
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ValueError(f"{entry_path}: expected a two-item tuple")
        name = _require_text(entry[0], f"{entry_path}[0]")
        metadata_value = entry[1]
        if (
            isinstance(metadata_value, bool)
            or type(metadata_value) is int
            or isinstance(metadata_value, str)
            or (type(metadata_value) is float and math.isfinite(metadata_value))
        ):
            pass
        else:
            raise ValueError(f"{entry_path}[1]: expected a finite JSON scalar")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError(f"{path}: metadata keys must be unique")


def _metadata_dict(values: ObjectMetadata) -> dict[str, ConfigScalar]:
    return dict(values)


def _validate_scope_lineage(
    parents: tuple[CreatedObjectRecord, ...],
    child: BookkeepingScope,
    path: str,
) -> None:
    """Validate causal scope without binding reusable resources to one use."""

    causal_parents = tuple(
        parent for parent in parents if parent.ref.kind not in _REUSABLE_OBJECT_KINDS
    )
    if not causal_parents:
        return

    parent_requests = {
        request_id
        for parent in causal_parents
        for request_id in parent.scope.correlation.request_ids
    }
    child_requests = set(child.correlation.request_ids)
    introduced_requests = sorted(child_requests - parent_requests)
    if introduced_requests:
        raise ValueError(
            f"{path}: child scope introduces request IDs absent from causal parents: "
            f"{introduced_requests}"
        )

    for field_name in ("step_index", "execution_id", "operation_id"):
        parent_values = {
            getattr(parent.scope, field_name)
            for parent in causal_parents
            if getattr(parent.scope, field_name) is not None
        }
        if len(parent_values) > 1:
            raise ValueError(f"{path}: causal parents disagree on {field_name}")
        if parent_values and getattr(child, field_name) != next(iter(parent_values)):
            raise ValueError(f"{path}: child scope does not inherit parent {field_name}")

    for field_name in ("batch_id", "layer", "microbatch", "iteration"):
        parent_values = {
            getattr(parent.scope.correlation, field_name)
            for parent in causal_parents
            if getattr(parent.scope.correlation, field_name) is not None
        }
        if len(parent_values) > 1:
            raise ValueError(f"{path}: causal parents disagree on {field_name}")
        if parent_values and getattr(child.correlation, field_name) != next(iter(parent_values)):
            raise ValueError(f"{path}: child scope does not inherit parent {field_name}")


def _fact_timestamp(fact: BookkeepingFact) -> int:
    if isinstance(fact, CreatedObjectRecord):
        return fact.created_at_ps
    if isinstance(fact, StageRecord):
        return fact.timestamp_ps
    if isinstance(fact, CompletionEvent):
        return fact.timestamp_ps
    raise TypeError(f"unsupported bookkeeping fact {type(fact).__name__}")


def _fact_scope(fact: BookkeepingFact) -> BookkeepingScope:
    if isinstance(fact, CreatedObjectRecord):
        return fact.scope
    if isinstance(fact, StageRecord):
        return fact.scope
    if isinstance(fact, CompletionEvent):
        return BookkeepingScope(
            execution_id=fact.execution_id,
            operation_id=fact.operation_id,
        )
    raise TypeError(f"unsupported bookkeeping fact {type(fact).__name__}")


def _validate_completion_event(fact: CompletionEvent, path: str) -> None:
    _require_text(fact.execution_id, f"{path}.execution_id")
    _require_text(fact.operation_id, f"{path}.operation_id")
    if not isinstance(fact.phase, EventPhase):
        raise TypeError(f"{path}.phase: expected EventPhase")
    _require_integer(fact.timestamp_ps, f"{path}.timestamp_ps", nonnegative=True)
    if fact.completed_bytes is not None:
        _require_integer(fact.completed_bytes, f"{path}.completed_bytes", nonnegative=True)
    if fact.resource is not None:
        if not isinstance(fact.resource, ResourceRef):
            raise ValueError(f"{path}.resource: expected ResourceRef")
        if not isinstance(fact.resource.kind, ResourceKind):
            raise ValueError(f"{path}.resource.kind: expected ResourceKind")
        _require_text(fact.resource.resource_id, f"{path}.resource.resource_id")
    if fact.subject_object_id is not None:
        _require_text(fact.subject_object_id, f"{path}.subject_object_id")


def validate_bookkeeping_ledger(ledger: BookkeepingLedger) -> None:
    """Validate sequence, lineage, WQE context and completion consistency."""

    if not isinstance(ledger, BookkeepingLedger):
        raise TypeError("ledger: expected BookkeepingLedger")
    if not isinstance(ledger.entries, tuple):
        raise TypeError("ledger.entries: in-memory contract requires a tuple")

    objects: dict[str, CreatedObjectRecord] = {}
    subject_timestamp: dict[str, int] = {}
    completed_wqes: set[str] = set()
    for index, entry in enumerate(ledger.entries):
        path = f"ledger.entries[{index}]"
        if not isinstance(entry, BookkeepingEntry):
            raise TypeError(f"{path}: expected BookkeepingEntry")
        if entry.sequence != index:
            raise ValueError(f"{path}.sequence: expected contiguous value {index}")
        fact = entry.fact
        timestamp = _fact_timestamp(fact)
        _require_integer(timestamp, f"{path}.timestamp_ps", nonnegative=True)

        if isinstance(fact, CreatedObjectRecord):
            _validate_ref(fact.ref, f"{path}.fact.ref")
            if fact.ref.object_id in objects:
                raise ValueError(f"{path}.fact.ref.object_id: duplicate object ID")
            if not isinstance(fact.owner, ObjectOwner):
                raise TypeError(f"{path}.fact.owner: expected ObjectOwner")
            _validate_scope(fact.scope, f"{path}.fact.scope")
            if fact.native_id is not None:
                _require_text(fact.native_id, f"{path}.fact.native_id")
            _validate_ref_tuple(fact.parent_refs, f"{path}.fact.parent_refs")
            if fact.ref in fact.parent_refs:
                raise ValueError(f"{path}.fact.parent_refs: object cannot parent itself")
            parent_records: list[CreatedObjectRecord] = []
            for parent_index, parent in enumerate(fact.parent_refs):
                prior = objects.get(parent.object_id)
                if prior is None or prior.ref != parent:
                    raise ValueError(
                        f"{path}.fact.parent_refs[{parent_index}]: unknown object reference"
                    )
                if fact.created_at_ps < prior.created_at_ps:
                    raise ValueError(
                        f"{path}.fact.parent_refs[{parent_index}]: child predates parent"
                    )
                parent_records.append(prior)
            _validate_scope_lineage(
                tuple(parent_records),
                fact.scope,
                f"{path}.fact.parent_refs",
            )
            _validate_metadata(fact.metadata, f"{path}.fact.metadata")
            if fact.ref.kind is CreatedObjectKind.NETWORK_WQE:
                parent_kinds = [parent.kind for parent in fact.parent_refs]
                for queue_kind, queue_name in (
                    (CreatedObjectKind.SEND_QUEUE, "send queue"),
                    (CreatedObjectKind.RECEIVE_QUEUE, "receive queue"),
                    (CreatedObjectKind.COMPLETION_QUEUE, "completion queue"),
                ):
                    if parent_kinds.count(queue_kind) != 1:
                        raise ValueError(
                            f"{path}.fact.parent_refs: WQE requires exactly one {queue_name}"
                        )
                transports = parent_kinds.count(
                    CreatedObjectKind.DCQCN_QP
                ) + parent_kinds.count(CreatedObjectKind.RNIC_CN_LINK_PAIR)
                if transports > 1:
                    raise ValueError(
                        f"{path}.fact.parent_refs: WQE permits at most one QP or link pair"
                    )
                transport_metadata = _metadata_dict(fact.metadata).get("transport_kind")
                if transports == 0 and transport_metadata != "none":
                    raise ValueError(
                        f"{path}.fact.metadata: transport-free WQE requires transport_kind=none"
                    )
                if transports == 1 and transport_metadata == "none":
                    raise ValueError(
                        f"{path}.fact.metadata: physical WQE cannot use transport_kind=none"
                    )
            objects[fact.ref.object_id] = fact
            continue

        if isinstance(fact, StageRecord):
            if not isinstance(fact.stage, ProcessingStage):
                raise TypeError(f"{path}.fact.stage: expected ProcessingStage")
            if not isinstance(fact.phase, StagePhase):
                raise TypeError(f"{path}.fact.phase: expected StagePhase")
            _validate_scope(fact.scope, f"{path}.fact.scope")
            _validate_ref_tuple(fact.object_refs, f"{path}.fact.object_refs")
            referenced_records: list[CreatedObjectRecord] = []
            for ref_index, ref in enumerate(fact.object_refs):
                prior = objects.get(ref.object_id)
                if prior is None or prior.ref != ref:
                    raise ValueError(
                        f"{path}.fact.object_refs[{ref_index}]: unknown object reference"
                    )
                if fact.timestamp_ps < prior.created_at_ps:
                    raise ValueError(
                        f"{path}.fact.object_refs[{ref_index}]: stage predates object"
                    )
                referenced_records.append(prior)
            _validate_scope_lineage(
                tuple(referenced_records),
                fact.scope,
                f"{path}.fact.object_refs",
            )
            continue

        if not isinstance(fact, CompletionEvent):
            raise TypeError(f"{path}.fact: unsupported fact {type(fact).__name__}")
        _validate_completion_event(fact, f"{path}.fact")
        subject_id = fact.subject_object_id
        if subject_id is None:
            continue
        subject = objects.get(subject_id)
        if subject is None:
            raise ValueError(f"{path}.fact.subject_object_id: unknown object ID")
        scope = subject.scope
        if scope.execution_id is not None and scope.execution_id != fact.execution_id:
            raise ValueError(f"{path}.fact.execution_id: does not match subject scope")
        if scope.operation_id is not None and scope.operation_id != fact.operation_id:
            raise ValueError(f"{path}.fact.operation_id: does not match subject scope")
        if fact.timestamp_ps < subject.created_at_ps:
            raise ValueError(f"{path}.fact.timestamp_ps: completion predates subject")
        prior_timestamp = subject_timestamp.get(subject_id, -1)
        if fact.timestamp_ps < prior_timestamp:
            raise ValueError(f"{path}.fact.timestamp_ps: subject timestamps must be nondecreasing")
        subject_timestamp[subject_id] = fact.timestamp_ps
        if fact.phase is EventPhase.COMPLETED and subject.ref.kind is CreatedObjectKind.NETWORK_WQE:
            if subject_id in completed_wqes:
                raise ValueError(f"{path}.fact: WQE has more than one completion")
            completion_queues = tuple(
                parent
                for parent in subject.parent_refs
                if parent.kind is CreatedObjectKind.COMPLETION_QUEUE
            )
            expected_resource = ResourceRef(
                ResourceKind.COMPLETION_QUEUE,
                completion_queues[0].object_id,
            )
            if fact.resource != expected_resource:
                raise ValueError(
                    f"{path}.fact.resource: WQE completion must name its completion queue"
                )
            completed_wqes.add(subject_id)


class RequestBookkeeper:
    """Mutable append seam with immutable records and atomic validation."""

    def __init__(self, ledger: BookkeepingLedger | None = None) -> None:
        initial = ledger or BookkeepingLedger()
        validate_bookkeeping_ledger(initial)
        self._entries = list(initial.entries)

    def append(self, fact: BookkeepingFact) -> BookkeepingEntry:
        """Append one fact and return its ledger-assigned entry."""

        return self.extend((fact,))[0]

    def extend(self, facts: Iterable[BookkeepingFact]) -> tuple[BookkeepingEntry, ...]:
        """Atomically append facts after validating the full candidate ledger."""

        start = len(self._entries)
        additions = tuple(
            BookkeepingEntry(start + offset, fact) for offset, fact in enumerate(facts)
        )
        candidate = BookkeepingLedger((*self._entries, *additions))
        validate_bookkeeping_ledger(candidate)
        self._entries.extend(additions)
        return additions

    def register_graph(self, graph: ExecutionGraph) -> tuple[BookkeepingEntry, ...]:
        """Append graph and operation facts without mutating the graph."""

        from simllm.core.execution_io import validate_execution_graph

        validate_execution_graph(graph)

        correlations = tuple(
            dict.fromkeys(
                request_id
                for operation in graph.operations
                for request_id in operation.correlation.request_ids
            )
        )
        facts: list[BookkeepingFact] = [
            StageRecord(
                ProcessingStage.EXECUTION,
                StagePhase.ENTERED,
                graph.released_at_ps,
                BookkeepingScope(
                    correlation=OperationCorrelation(request_ids=correlations),
                    step_index=graph.step_index,
                    execution_id=graph.execution_id,
                ),
            )
        ]
        facts.extend(
            CreatedObjectRecord(
                ref=CreatedObjectRef(
                    CreatedObjectKind.EXECUTION_OPERATION,
                    f"execution-operation:{graph.execution_id}:{operation.operation_id}",
                ),
                owner=ObjectOwner.CORE,
                created_at_ps=graph.released_at_ps,
                scope=BookkeepingScope(
                    correlation=operation.correlation,
                    step_index=graph.step_index,
                    execution_id=graph.execution_id,
                    operation_id=operation.operation_id,
                ),
                native_id=operation.operation_id,
            )
            for operation in graph.operations
        )
        return self.extend(facts)

    def snapshot(self) -> BookkeepingLedger:
        """Return an immutable ledger snapshot."""

        return BookkeepingLedger(tuple(self._entries))

    def for_request(self, request_id: str) -> tuple[BookkeepingEntry, ...]:
        """Return facts correlated with one request in append order."""

        _require_text(request_id, "request_id")
        objects = {
            entry.fact.ref.object_id: entry.fact
            for entry in self._entries
            if isinstance(entry.fact, CreatedObjectRecord)
        }
        operation_scopes = {
            (entry.fact.scope.execution_id, entry.fact.scope.operation_id): entry.fact.scope
            for entry in self._entries
            if isinstance(entry.fact, CreatedObjectRecord)
            and entry.fact.ref.kind is CreatedObjectKind.EXECUTION_OPERATION
        }
        selected: list[BookkeepingEntry] = []
        for entry in self._entries:
            scope = _fact_scope(entry.fact)
            if isinstance(entry.fact, CompletionEvent):
                if entry.fact.subject_object_id is not None:
                    subject = objects.get(entry.fact.subject_object_id)
                    if subject is not None:
                        scope = subject.scope
                else:
                    scope = operation_scopes.get(
                        (entry.fact.execution_id, entry.fact.operation_id),
                        scope,
                    )
            if request_id in scope.correlation.request_ids:
                selected.append(entry)
        return tuple(selected)

    def for_execution(self, execution_id: str) -> tuple[BookkeepingEntry, ...]:
        """Return facts correlated with one execution in append order."""

        _require_text(execution_id, "execution_id")
        return tuple(
            entry
            for entry in self._entries
            if _fact_scope(entry.fact).execution_id == execution_id
        )

    def for_object(
        self,
        ref: CreatedObjectRef,
        *,
        recursive: bool = False,
    ) -> tuple[BookkeepingEntry, ...]:
        """Return direct facts, and optionally all descendant-object facts."""

        _validate_ref(ref, "ref")
        created_refs = {
            entry.fact.ref
            for entry in self._entries
            if isinstance(entry.fact, CreatedObjectRecord)
        }
        if ref not in created_refs:
            return ()
        selected = {ref}
        if recursive:
            changed = True
            while changed:
                changed = False
                for entry in self._entries:
                    fact = entry.fact
                    if not isinstance(fact, CreatedObjectRecord):
                        continue
                    if any(parent in selected for parent in fact.parent_refs) and fact.ref not in selected:
                        selected.add(fact.ref)
                        changed = True

        result: list[BookkeepingEntry] = []
        selected_ids = {item.object_id for item in selected}
        operation_keys = {
            (entry.fact.scope.execution_id, entry.fact.scope.operation_id)
            for entry in self._entries
            if isinstance(entry.fact, CreatedObjectRecord)
            and entry.fact.ref in selected
            and entry.fact.ref.kind is CreatedObjectKind.EXECUTION_OPERATION
        }
        for entry in self._entries:
            fact = entry.fact
            if isinstance(fact, CreatedObjectRecord):
                if fact.ref in selected:
                    result.append(entry)
            elif isinstance(fact, StageRecord):
                if any(item in selected for item in fact.object_refs):
                    result.append(entry)
            elif (
                fact.subject_object_id in selected_ids
                or (fact.execution_id, fact.operation_id) in operation_keys
            ):
                result.append(entry)
        return tuple(result)
