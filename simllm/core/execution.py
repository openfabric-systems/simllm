"""Versioned execution and completion contracts for device-level simulation.

``StepRecord`` remains the framework scheduler's semantic decision.  An
``ExecutionLowerer`` turns that decision, plus framework observations such as
KV-cache events and stream ordering, into an ``ExecutionGraph``.  A
``DeviceRuntime`` executes the graph against progressively more detailed
resource models and reports an ``ExecutionResult`` made of
``CompletionEvent`` records.

This module intentionally implements no lowering or scheduling policy.  It
defines the boundary those implementations plug into.  In particular, the
framework declares legal overlap by queue placement and dependency edges; the
runtime decides whether that legal overlap is physically achieved after GPU,
HBM, copy-engine, DMA, NCCL and NIC contention.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

from simllm.core.step import StepRecord

if TYPE_CHECKING:
    from simllm.core.bookkeeping import RequestBookkeeper

#: Versioned wire-format names; execution_io implements the JSON readers and writers.
EXECUTION_GRAPH_SCHEMA = "simllm-execution-graph-v1"
COMPLETION_EVENT_SCHEMA = "simllm-completion-event-v1"
EXECUTION_RESULT_SCHEMA = "simllm-execution-result-v1"

ConfigScalar: TypeAlias = str | int | float | bool


class KvCacheAction(enum.Enum):
    """Logical KV-cache lifecycle events observed from the real framework."""

    RESERVE = "reserve"
    ALLOCATE = "allocate"
    BIND_PREFIX = "bind-prefix"
    TOUCH = "touch"
    READ = "read"
    WRITE = "write"
    RETAIN = "retain"
    RELEASE = "release"
    EVICT = "evict"
    FREE = "free"
    SWAP = "swap"
    TRANSFER = "transfer"
    RECOMPUTE = "recompute"


class ControlMode(enum.Enum):
    """Whether graph progress waits for delivery of a control message."""

    ASYNCHRONOUS = "asynchronous"
    SYNCHRONOUS = "synchronous"


class ResourceKind(enum.Enum):
    """Observable runtime resources, not resource assignments in the graph.

    The device runtime owns mapping and arbitration.  These names let its
    completion stream attribute queueing and service time without leaking a
    backend-specific class into the core contract.
    """

    HOST_LAUNCH_QUEUE = "host-launch-queue"
    CUDA_STREAM = "cuda-stream"
    GPU_WORK_QUEUE = "gpu-work-queue"
    GPU_SCHEDULER = "gpu-scheduler"
    HBM_QUEUE = "hbm-queue"
    COPY_ENGINE = "copy-engine"
    NVLINK = "nvlink"
    NCCL_CHANNEL = "nccl-channel"
    NIC_SEND_QUEUE = "nic-send-queue"
    NIC_RECEIVE_QUEUE = "nic-receive-queue"
    NIC = "nic"
    COMPLETION_QUEUE = "completion-queue"
    CONTROL_QUEUE = "control-queue"


class EventPhase(enum.Enum):
    """One observable point in an operation's resource lifecycle."""

    SUBMITTED = "submitted"
    QUEUED = "queued"
    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"


class DependencyScope(str, enum.Enum):
    """How one effective graph edge gates its target operation."""

    WHOLE_OPERATION = "whole-operation"
    PARTICIPANT_LOCAL = "participant-local"


class DependencyOrigin(str, enum.Enum):
    """Which authoritative graph rule contributes an effective edge."""

    EXPLICIT = "explicit"
    LOGICAL_QUEUE_FIFO = "logical-queue-fifo"


@dataclass(frozen=True)
class ComputeWork:
    """One kernel or fused kernel region submitted for GPU execution.

    ``nominal_duration_ps`` is supplied by the selected compute provider.
    Shape and demand stay on the record so a higher-fidelity runtime may
    replace or adjust that estimate without changing the graph structure.
    """

    kernel: str
    config: tuple[tuple[str, ConfigScalar], ...] = ()
    flops: int = 0
    hbm_bytes: int = 0
    nominal_duration_ps: int | None = None
    uncertainty_fraction: float | None = None


@dataclass(frozen=True)
class KvCacheWork:
    """One explicit logical KV-cache lifecycle operation.

    Block identifiers are strings because vLLM and SGLang use different
    native identifiers.  Adapters normalize them while preserving the
    framework's allocation, prefix binding, eviction and ownership choices.
    Physical reads, writes, swaps and transfers are separate ``DmaWork`` or
    compute operations connected by graph dependencies.
    """

    action: KvCacheAction
    pool_id: str
    request_id: str | None = None
    block_ids: tuple[str, ...] = ()
    token_start: int | None = None
    token_end: int | None = None
    layer: int | None = None
    dtype: str | None = None
    byte_count: int = 0
    placement_epoch: int = 0
    reference_count: int | None = None
    cause: str | None = None
    correlation_id: str | None = None


@dataclass(frozen=True)
class DmaWork:
    """One data-mover descriptor; the runtime selects a physical engine."""

    descriptor_id: str
    source: str
    destination: str
    byte_count: int


@dataclass(frozen=True)
class CollectiveWork:
    """One semantic collective before algorithm and channel expansion.

    ``payload_bytes`` is algorithm-relative, and consumers must branch on
    ``(collective, algorithm_hint)``: a ring all-reduce carries the full
    reduced payload, while a pairwise all-to-allv carries the bytes each
    rank sends to each other rank (one uniform ordered-pair share).

    ``pair_payload_bytes`` is the sparse variable-size form for pairwise
    all-to-allv. Each entry is ``(source_rank, destination_rank, bytes)``;
    omitted pairs carry zero bytes. A nonempty table is authoritative and
    requires ``payload_bytes == 0``. An empty table retains the uniform scalar
    interpretation and its original v1 wire encoding.

    ``request_pair_payload_bytes`` is an optional loss-checked partition of
    that sparse table. Each entry is
    ``(request_id, source_rank, destination_rank, bytes)``. It never replaces
    or changes the aggregate physical service demand; its sums must reproduce
    ``pair_payload_bytes`` exactly.
    """

    collective: str
    ranks: tuple[int, ...]
    payload_bytes: int
    algorithm_hint: str | None = None
    channel_hint: str | None = None
    pair_payload_bytes: tuple[tuple[int, int, int], ...] = ()
    request_pair_payload_bytes: tuple[tuple[str, int, int, int], ...] = ()


@dataclass(frozen=True)
class ControlWork:
    """One local or in-band control message."""

    message: str
    destination_ranks: tuple[int, ...] = ()
    payload_bytes: int = 0
    mode: ControlMode = ControlMode.ASYNCHRONOUS


WorkPayload: TypeAlias = ComputeWork | KvCacheWork | DmaWork | CollectiveWork | ControlWork


@dataclass(frozen=True)
class OperationCorrelation:
    """Stable semantic identities shared by plans, events and metrics.

    One operation may serve several requests in a scheduler batch.  Layer,
    microbatch and iteration are optional because the same graph contract is
    used by serving and future training-plan producers.
    """

    request_ids: tuple[str, ...] = ()
    batch_id: str | None = None
    layer: int | None = None
    microbatch: int | None = None
    iteration: int | None = None


@dataclass(frozen=True)
class ExecutionOperation:
    """One node in an execution graph.

    Operations are submitted in graph order.  A ``logical_queue`` is FIFO;
    operations on different queues may overlap unless a dependency forbids
    it.  The queue name is a framework-level lane such as ``cuda:0:compute``
    or ``cuda:0:nccl``.  It is not a physical resource assignment.
    """

    operation_id: str
    rank: int
    logical_queue: str
    work: WorkPayload
    depends_on: tuple[str, ...] = ()
    not_before_ps: int = 0
    priority: int = 0
    correlation: OperationCorrelation = field(default_factory=OperationCorrelation)
    placement_epoch: int = 0
    participant_local_depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionObservations:
    """Framework-neutral device work observed for one scheduler step.

    Adapters normalize native cache, stream, event, kernel, collective, DMA
    and control observations into these typed operations.  The core does not
    reconstruct framework policy from token counts.  Tuple order preserves
    observed submission order; dependencies and logical queues preserve the
    framework's concurrency contract.
    """

    operations: tuple[ExecutionOperation, ...] = ()
    completion_operation_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionGraph:
    """Versioned device-level DAG lowered from one scheduler step.

    This passive record has no execution policy.  Its tuple order defines
    FIFO submission order within each logical queue. ``depends_on`` supplies
    operation-completion edges, while ``participant_local_depends_on`` lets
    each rank of a distributed operation wait for the predecessor frontier on
    that same rank. Keeping the two edge kinds separate permits a target to
    carry both semantics without weakening either one.
    """

    execution_id: str
    step_index: int
    released_at_ps: int
    operations: tuple[ExecutionOperation, ...] = ()
    #: operations whose logical completion releases the framework; empty means all
    completion_operation_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EffectiveDependencyEdge:
    """One fully expanded ordering edge carried by an execution graph.

    Whole-operation edges gate every target participant on the predecessor's
    logical completion. Participant-local edges are expanded to one record per
    shared rank and gate only that rank on its predecessor frontier.
    """

    predecessor_id: str
    operation_id: str
    scope: DependencyScope
    origin: DependencyOrigin
    participant_rank: int | None = None


@dataclass(frozen=True)
class ResourceRef:
    """One concrete queue or resource selected by a device runtime."""

    kind: ResourceKind
    resource_id: str


@dataclass(frozen=True)
class CompletionEvent:
    """Timestamped evidence of an operation or created object making progress.

    ``subject_object_id`` identifies the finer-grained object when one graph
    operation expands into several runtime objects, e.g. one WQE per rank and
    peer for a collective. Packet identifiers never cross this boundary.
    """

    execution_id: str
    operation_id: str
    phase: EventPhase
    timestamp_ps: int
    resource: ResourceRef | None = None
    completed_bytes: int | None = None
    subject_object_id: str | None = None


@dataclass(frozen=True)
class ExecutionResult:
    """Reached graph boundary plus events observed up to that boundary.

    ``completed_at_ps`` is the framework-visible completion boundary.
    ``quiesced_at_ps`` is present only when every background physical action
    from this graph has also drained.  A stateful runtime may therefore return
    an asynchronous graph with ``quiesced_at_ps=None`` and report later
    resource completions through its event stream.
    """

    execution_id: str
    completed_at_ps: int
    events: tuple[CompletionEvent, ...] = field(default_factory=tuple)
    quiesced_at_ps: int | None = None


CompletionHandler: TypeAlias = Callable[[CompletionEvent], None]


@runtime_checkable
class ExecutionLowerer(Protocol):
    """Translate a framework step and observations into the shared graph."""

    def lower(
        self,
        record: StepRecord,
        observations: ExecutionObservations | None = None,
    ) -> ExecutionGraph:
        """Return the device-level graph for ``record``."""
        ...


@runtime_checkable
class DeviceRuntime(Protocol):
    """Advance a graph to its boundary while streaming resource events."""

    def execute(
        self,
        graph: ExecutionGraph,
        *,
        on_event: CompletionHandler | None = None,
        bookkeeping: RequestBookkeeper | None = None,
    ) -> ExecutionResult:
        """Advance ``graph`` and append runtime facts when requested."""
        ...
