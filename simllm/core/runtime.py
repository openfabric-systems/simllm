"""Coarse resource runtime for validated :mod:`simllm.core` execution graphs.

The runtime owns inter-operation orchestration, not GPU microarchitecture or
RNIC internals. It delegates traced concurrent kernels and isolated copy
descriptors to :mod:`simllm.compute`. Cross-node semantic sends use exactly one
WQE authority: :class:`AtlahsWqeLedger` in explicit bypass mode, or a supplied
native session in structural mode.
"""

from __future__ import annotations

import enum
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from simllm.compute import (
        CopyDirection,
        CopyEngineServiceModel,
        KernelLaunch,
        SmSchedulerModel,
    )
from simllm.core.bookkeeping import (
    BookkeepingScope,
    CreatedObjectKind,
    CreatedObjectRecord,
    CreatedObjectRef,
    ObjectOwner,
    ProcessingStage,
    RequestBookkeeper,
    StagePhase,
    StageRecord,
)
from simllm.core.execution import (
    CollectiveWork,
    CompletionEvent,
    CompletionHandler,
    ComputeWork,
    ControlMode,
    ControlWork,
    DmaWork,
    EventPhase,
    ExecutionGraph,
    ExecutionOperation,
    ExecutionResult,
    KvCacheAction,
    KvCacheWork,
    ResourceKind,
    ResourceRef,
)
from simllm.core.execution_io import execution_result_to_json, validate_execution_graph
from simllm.core.step import LatencyAttribution

PS_PER_SECOND = 1_000_000_000_000
DEFAULT_GPUS_PER_NODE = 8
DEFAULT_RNICS_PER_NODE = 8
DEFAULT_RNIC_RATE_BPS = 400_000_000_000
DEFAULT_NVLINK_RATE_BPS = 900_000_000_000
DEFAULT_GOAL_BASE_TAG = 1000
CONTROL_GOAL_TAG_OFFSET = 1_000_000
CONTROL_GOAL_TAG_STRIDE = 1024


def _require_int(name: str, value: object, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _serialization_ps(byte_count: int, rate_bps: int) -> int:
    _require_int("byte_count", byte_count)
    _require_int("rate_bps", rate_bps, positive=True)
    numerator = byte_count * 8 * PS_PER_SECOND
    return (numerator + rate_bps - 1) // rate_bps


def _cycles_to_ps(cycles: int, clock_hz: int) -> int:
    numerator = cycles * PS_PER_SECOND
    return (numerator + clock_hz - 1) // clock_hz


def _escape_id(value: str) -> str:
    return value.replace("%", "%25").replace(":", "%3A")


def _operation_object_ref(execution_id: str, operation_id: str) -> CreatedObjectRef:
    return CreatedObjectRef(
        CreatedObjectKind.EXECUTION_OPERATION,
        "execution-operation:"
        f"{_escape_id(execution_id)}:{_escape_id(operation_id)}",
    )


class RnicAuthorityMode(str, enum.Enum):
    """Which sole mutable authority owns SQ, RQ, CQ, and WQE state."""

    BYPASS = "bypass"
    STRUCTURAL = "structural"


@dataclass(frozen=True)
class CoarseDeviceProfile:
    """Fixed first device profile plus ideal study-adjustable rates."""

    gpus_per_node: int = DEFAULT_GPUS_PER_NODE
    rnics_per_node: int = DEFAULT_RNICS_PER_NODE
    rnic_rate_bps: int = DEFAULT_RNIC_RATE_BPS
    nvlink_rate_bps: int = DEFAULT_NVLINK_RATE_BPS
    launch_service_ps: int = 0
    control_service_ps: int = 0
    nccl_channel_service_ps: int = 0
    completion_delivery_ps: int = 0
    goal_base_tag: int = DEFAULT_GOAL_BASE_TAG
    copy_engines: tuple[CopyEngineServiceModel, ...] = ()

    def __post_init__(self) -> None:
        from simllm.compute import CopyEngineServiceModel

        if self.gpus_per_node != DEFAULT_GPUS_PER_NODE:
            raise ValueError("the first coarse profile requires exactly eight GPUs per node")
        if self.rnics_per_node != DEFAULT_RNICS_PER_NODE:
            raise ValueError("the first coarse profile requires exactly eight RNICs per node")
        for name in ("rnic_rate_bps", "nvlink_rate_bps"):
            _require_int(name, getattr(self, name), positive=True)
        for name in (
            "launch_service_ps",
            "control_service_ps",
            "nccl_channel_service_ps",
            "completion_delivery_ps",
            "goal_base_tag",
        ):
            _require_int(name, getattr(self, name))
        engines = tuple(self.copy_engines)
        object.__setattr__(self, "copy_engines", engines)
        if any(not isinstance(engine, CopyEngineServiceModel) for engine in engines):
            raise TypeError("copy_engines must contain CopyEngineServiceModel instances")
        engine_ids = [engine.engine_id for engine in engines]
        if len(engine_ids) != len(set(engine_ids)):
            raise ValueError("copy engine IDs must be unique")

    def node_gpu(self, rank: int) -> tuple[int, int]:
        """Return the fixed node and local-GPU affinity for a global rank."""

        _require_int("rank", rank)
        return divmod(rank, self.gpus_per_node)

    def rnic_id(self, rank: int) -> str:
        node, gpu = self.node_gpu(rank)
        return f"node-{node}:rnic-{gpu}"


@dataclass(frozen=True)
class ArbitrationCandidate:
    """One legal ready candidate at a replaceable policy seam."""

    operation_id: str
    baseline_sequence: int
    eligible_at_ps: int
    class_label: int


@runtime_checkable
class ArbitrationPolicy(Protocol):
    """Select one candidate after mandatory legality and ordering filters."""

    def select(self, candidates: tuple[ArbitrationCandidate, ...]) -> ArbitrationCandidate:
        """Return one member of ``candidates``."""
        ...


@dataclass(frozen=True)
class IdentityArbitrationPolicy:
    """Feature-off policy that ignores class labels and preserves baseline order."""

    def select(self, candidates: tuple[ArbitrationCandidate, ...]) -> ArbitrationCandidate:
        if not candidates:
            raise ValueError("identity arbitration requires at least one candidate")
        return min(candidates, key=lambda item: item.baseline_sequence)


@dataclass(frozen=True)
class QueueVisit:
    """One visit using the repository-wide queue timestamp contract."""

    execution_id: str
    operation_id: str
    resource: ResourceRef
    submitted_at_ps: int
    eligible_at_ps: int
    started_at_ps: int
    finished_at_ps: int
    completed_at_ps: int
    service_bytes: int = 0
    subject_object_id: str | None = None
    stage: str | None = None

    def __post_init__(self) -> None:
        _require_text("execution_id", self.execution_id)
        _require_text("operation_id", self.operation_id)
        if not isinstance(self.resource, ResourceRef):
            raise TypeError("resource must be a ResourceRef")
        for name in (
            "submitted_at_ps",
            "eligible_at_ps",
            "started_at_ps",
            "finished_at_ps",
            "completed_at_ps",
            "service_bytes",
        ):
            _require_int(name, getattr(self, name))
        if self.subject_object_id is not None:
            _require_text("subject_object_id", self.subject_object_id)
        if self.stage is not None:
            _require_text("stage", self.stage)
        if self.eligible_at_ps < self.submitted_at_ps:
            raise ValueError("queue visit becomes eligible before submission")
        if self.started_at_ps < self.eligible_at_ps:
            raise ValueError("queue visit starts before eligibility")
        if self.finished_at_ps < self.started_at_ps:
            raise ValueError("queue visit finishes before service starts")
        if self.completed_at_ps < self.finished_at_ps:
            raise ValueError("queue visit completes before resource release")

    @property
    def queue_wait_ps(self) -> int:
        return self.started_at_ps - self.eligible_at_ps

    @property
    def service_ps(self) -> int:
        return self.finished_at_ps - self.started_at_ps

    @property
    def visibility_ps(self) -> int:
        return self.completed_at_ps - self.finished_at_ps


@dataclass(frozen=True)
class CriticalPathBreakdown:
    """Selected-path accounting, separate from additive visit work."""

    launch_queue_ps: int
    device_queue_ps: int
    service_ps: int
    completion_delivery_ps: int
    external_dependency_ps: int
    operation_latency_ps: int
    critical_path_queue_ps: int

    def __post_init__(self) -> None:
        for name in (
            "launch_queue_ps",
            "device_queue_ps",
            "service_ps",
            "completion_delivery_ps",
            "external_dependency_ps",
            "operation_latency_ps",
            "critical_path_queue_ps",
        ):
            _require_int(name, getattr(self, name))
        total = (
            self.launch_queue_ps
            + self.device_queue_ps
            + self.service_ps
            + self.completion_delivery_ps
            + self.external_dependency_ps
        )
        if total != self.operation_latency_ps:
            raise ValueError("critical-path breakdown does not conserve operation latency")


@dataclass(frozen=True)
class RuntimeOperationRecord:
    """Coarse realized timing for one immutable graph operation."""

    operation_id: str
    class_label: int
    submitted_at_ps: int
    eligible_at_ps: int
    completed_at_ps: int
    physical_completed_at_ps: int
    participant_completed_at_ps: tuple[tuple[int, int], ...]
    breakdown: CriticalPathBreakdown
    attribution: LatencyAttribution
    critical_predecessor_id: str | None
    sum_visit_wait_ps: int


@dataclass(frozen=True)
class SemanticWqeSubmission:
    """Framework-neutral send handed to the selected WQE authority."""

    execution_id: str
    operation_id: str
    source_rank: int
    destination_rank: int
    payload_bytes: int
    goal_tag: int
    extent_index: int
    channel_id: str
    submitted_at_ps: int
    eligible_at_ps: int
    class_label: int = 0
    nccl_command_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("execution_id", "operation_id", "channel_id"):
            _require_text(name, getattr(self, name))
        for name in (
            "source_rank",
            "destination_rank",
            "payload_bytes",
            "goal_tag",
            "extent_index",
            "submitted_at_ps",
            "eligible_at_ps",
        ):
            _require_int(name, getattr(self, name))
        if self.source_rank == self.destination_rank:
            raise ValueError("semantic WQE source and destination must differ")
        if self.eligible_at_ps < self.submitted_at_ps:
            raise ValueError("semantic WQE becomes eligible before submission")
        if self.nccl_command_id is not None:
            _require_text("nccl_command_id", self.nccl_command_id)


@dataclass(frozen=True)
class WqeLifecycleProjection:
    """Loss-checked immutable projection returned by the WQE authority."""

    authority: str
    execution_id: str
    operation_id: str
    wqe_id: str
    native_wqe_id: str
    sq_id: str
    rq_id: str
    cq_id: str
    qp_id: str
    rnic_id: str
    source_rank: int
    destination_rank: int
    payload_bytes: int
    goal_tag: int
    extent_index: int
    sq_post_sequence: int
    cq_post_sequence: int
    submitted_at_ps: int
    eligible_at_ps: int
    started_at_ps: int
    finished_at_ps: int
    completed_at_ps: int
    channel_id: str
    nccl_command_id: str | None = None
    doorbell_started_at_ps: int | None = None
    doorbell_completed_at_ps: int | None = None
    network_eligible_at_ps: int | None = None
    network_started_at_ps: int | None = None
    network_finished_at_ps: int | None = None
    network_accepted_at_ps: int | None = None
    first_packet_at_ps: int | None = None
    last_packet_at_ps: int | None = None
    packet_tx_started_at_ps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "authority",
            "execution_id",
            "operation_id",
            "wqe_id",
            "native_wqe_id",
            "sq_id",
            "rq_id",
            "cq_id",
            "qp_id",
            "rnic_id",
            "channel_id",
        ):
            _require_text(name, getattr(self, name))
        for name in (
            "source_rank",
            "destination_rank",
            "payload_bytes",
            "goal_tag",
            "extent_index",
            "sq_post_sequence",
            "cq_post_sequence",
            "submitted_at_ps",
            "eligible_at_ps",
            "started_at_ps",
            "finished_at_ps",
            "completed_at_ps",
        ):
            _require_int(name, getattr(self, name))
        if self.sq_post_sequence <= 0 or self.cq_post_sequence <= 0:
            raise ValueError("WQE queue sequences must be positive")
        if self.eligible_at_ps < self.submitted_at_ps:
            raise ValueError("WQE becomes eligible before submission")
        if self.started_at_ps < self.eligible_at_ps:
            raise ValueError("WQE starts before eligibility")
        if self.finished_at_ps < self.started_at_ps:
            raise ValueError("WQE finishes before start")
        if self.completed_at_ps < self.finished_at_ps:
            raise ValueError("WQE completion precedes resource release")
        if self.nccl_command_id is not None:
            _require_text("nccl_command_id", self.nccl_command_id)
        native_stages = (
            self.doorbell_started_at_ps,
            self.doorbell_completed_at_ps,
            self.network_eligible_at_ps,
            self.network_started_at_ps,
            self.network_finished_at_ps,
        )
        if any(value is not None for value in native_stages):
            if any(value is None for value in native_stages):
                raise ValueError("native WQE stage timestamps must be all present")
            for name, value in zip(
                (
                    "doorbell_started_at_ps",
                    "doorbell_completed_at_ps",
                    "network_eligible_at_ps",
                    "network_started_at_ps",
                    "network_finished_at_ps",
                ),
                native_stages,
                strict=True,
            ):
                _require_int(name, value)
            assert self.doorbell_started_at_ps is not None
            assert self.doorbell_completed_at_ps is not None
            assert self.network_eligible_at_ps is not None
            assert self.network_started_at_ps is not None
            assert self.network_finished_at_ps is not None
            if not (
                self.submitted_at_ps
                <= self.doorbell_started_at_ps
                <= self.doorbell_completed_at_ps
                <= self.network_eligible_at_ps
                <= self.network_started_at_ps
                <= self.network_finished_at_ps
                <= self.completed_at_ps
            ):
                raise ValueError("native WQE stage timestamps are not monotonic")
            if (
                self.started_at_ps != self.network_started_at_ps
                or self.finished_at_ps != self.network_finished_at_ps
            ):
                raise ValueError(
                    "WQE start and finish must project the native network stage"
                )
        packet_fields = (
            self.network_accepted_at_ps,
            self.first_packet_at_ps,
            self.last_packet_at_ps,
        )
        if any(value is not None for value in packet_fields) or bool(
            self.packet_tx_started_at_ps
        ):
            if not all(value is not None for value in packet_fields) or not isinstance(
                self.packet_tx_started_at_ps, tuple
            ) or not self.packet_tx_started_at_ps:
                raise ValueError("WQE packet timeline must be all present")
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in self.packet_tx_started_at_ps
            ):
                raise TypeError("WQE packet TX starts must be integers")
            accepted = self.network_accepted_at_ps
            first = self.first_packet_at_ps
            last = self.last_packet_at_ps
            assert accepted is not None
            assert first is not None
            assert last is not None
            if tuple(sorted(self.packet_tx_started_at_ps)) != self.packet_tx_started_at_ps:
                raise ValueError("WQE packet TX starts must be monotonic")
            if (
                first != min(self.packet_tx_started_at_ps)
                or last != max(self.packet_tx_started_at_ps)
            ):
                raise ValueError("WQE packet timeline must derive from TX starts")
            if (
                self.network_eligible_at_ps is None
                or self.network_started_at_ps is None
                or self.network_finished_at_ps is None
                or not (
                    self.network_eligible_at_ps
                    <= accepted
                    <= first
                    <= last
                    <= self.network_finished_at_ps
                )
                or self.network_started_at_ps != first
            ):
                raise ValueError("WQE packet timeline is not monotonic")


@runtime_checkable
class NativeRnicTransaction(Protocol):
    """Isolated native submission plan with an atomic prepare/commit boundary."""

    @property
    def authority_name(self) -> str:
        """Return the sole session authority represented by this transaction."""
        ...

    @property
    def random_draw_count(self) -> int:
        """Return random draws consumed by this staged transaction."""
        ...

    def submit(self, submission: SemanticWqeSubmission) -> WqeLifecycleProjection:
        """Stage and project one semantic WQE without mutating the session."""
        ...

    def prepare(self) -> None:
        """Validate that the subsequent atomic commit cannot fail."""
        ...

    def commit(self) -> None:
        """Atomically install every staged WQE in the sole native session."""
        ...

    def abort(self) -> None:
        """Discard staged state without changing the sole native session."""
        ...


@runtime_checkable
class NativeRnicSession(Protocol):
    """Structural session seam used by the HTSIM-9 composed native RNIC."""

    @property
    def authority_name(self) -> str:
        """Return the native authority identifier recorded in run evidence."""
        ...

    def begin_transaction(self) -> NativeRnicTransaction:
        """Return isolated state whose prepared commit is atomic and infallible."""
        ...


class AtlahsWqeLedger:
    """Timing-neutral bypass authority for SQ, RQ, CQ, QP, and WQE state."""

    authority_name = "AtlahsWqeLedger"

    def __init__(
        self,
        profile: CoarseDeviceProfile,
        *,
        _available_at_ps: Mapping[str, int] | None = None,
        _sq_sequences: Mapping[str, int] | None = None,
        _cq_sequences: Mapping[str, int] | None = None,
        _records: Sequence[WqeLifecycleProjection] = (),
    ) -> None:
        if not isinstance(profile, CoarseDeviceProfile):
            raise TypeError("profile must be a CoarseDeviceProfile")
        self.profile = profile
        self._available_at_ps = dict(_available_at_ps or {})
        self._sq_sequences = dict(_sq_sequences or {})
        self._cq_sequences = dict(_cq_sequences or {})
        self._records = list(_records)

    def clone(self) -> AtlahsWqeLedger:
        """Return an isolated authority copy for transactional execution."""

        return AtlahsWqeLedger(
            self.profile,
            _available_at_ps=self._available_at_ps,
            _sq_sequences=self._sq_sequences,
            _cq_sequences=self._cq_sequences,
            _records=self._records,
        )

    @property
    def records(self) -> tuple[WqeLifecycleProjection, ...]:
        return tuple(self._records)

    @property
    def random_draw_count(self) -> int:
        return 0

    def submit(self, submission: SemanticWqeSubmission) -> WqeLifecycleProjection:
        if not isinstance(submission, SemanticWqeSubmission):
            raise TypeError("submission must be a SemanticWqeSubmission")
        source_node, source_gpu = self.profile.node_gpu(submission.source_rank)
        destination_node, destination_gpu = self.profile.node_gpu(
            submission.destination_rank
        )
        rnic_id = self.profile.rnic_id(submission.source_rank)
        sq_id = f"atlahs:node-{source_node}:gpu-{source_gpu}:sq"
        rq_id = f"atlahs:node-{destination_node}:gpu-{destination_gpu}:rq"
        cq_id = f"atlahs:node-{source_node}:gpu-{source_gpu}:cq"
        qp_id = f"atlahs:node-{source_node}:gpu-{source_gpu}:qp"
        sq_sequence = self._sq_sequences.get(sq_id, 0) + 1
        cq_sequence = self._cq_sequences.get(cq_id, 0) + 1
        started_at_ps = max(
            submission.eligible_at_ps,
            self._available_at_ps.get(rnic_id, 0),
        )
        finished_at_ps = started_at_ps + _serialization_ps(
            submission.payload_bytes,
            self.profile.rnic_rate_bps,
        )
        completed_at_ps = finished_at_ps + self.profile.completion_delivery_ps
        wqe_id = (
            "atlahs-wqe:"
            f"{_escape_id(submission.execution_id)}:"
            f"{_escape_id(submission.operation_id)}:"
            f"{submission.extent_index}"
        )
        record = WqeLifecycleProjection(
            authority=self.authority_name,
            execution_id=submission.execution_id,
            operation_id=submission.operation_id,
            wqe_id=wqe_id,
            native_wqe_id=f"{qp_id}:post-{sq_sequence}",
            sq_id=sq_id,
            rq_id=rq_id,
            cq_id=cq_id,
            qp_id=qp_id,
            rnic_id=rnic_id,
            source_rank=submission.source_rank,
            destination_rank=submission.destination_rank,
            payload_bytes=submission.payload_bytes,
            goal_tag=submission.goal_tag,
            extent_index=submission.extent_index,
            sq_post_sequence=sq_sequence,
            cq_post_sequence=cq_sequence,
            submitted_at_ps=submission.submitted_at_ps,
            eligible_at_ps=submission.eligible_at_ps,
            started_at_ps=started_at_ps,
            finished_at_ps=finished_at_ps,
            completed_at_ps=completed_at_ps,
            channel_id=submission.channel_id,
            nccl_command_id=submission.nccl_command_id,
        )
        self._available_at_ps[rnic_id] = finished_at_ps
        self._sq_sequences[sq_id] = sq_sequence
        self._cq_sequences[cq_id] = cq_sequence
        self._records.append(record)
        return record


@dataclass(frozen=True)
class RuntimeReport:
    """Diagnostics from the last execution, with reductions kept distinct."""

    execution_id: str
    authority: str
    operations: tuple[RuntimeOperationRecord, ...]
    visits: tuple[QueueVisit, ...]
    wqes: tuple[WqeLifecycleProjection, ...]
    sum_visit_wait_ps: int
    critical_path_queue_ps: int
    realized_critical_path_operation_ids: tuple[str, ...]
    class_service_bytes: tuple[tuple[int, int], ...]
    random_draw_count: int


@dataclass
class _RuntimeState:
    host_available: dict[int, int] = field(default_factory=dict)
    gpu_available: dict[int, int] = field(default_factory=dict)
    hbm_available: dict[int, int] = field(default_factory=dict)
    copy_available: dict[tuple[int, str], int] = field(default_factory=dict)
    control_available: dict[int, int] = field(default_factory=dict)
    nccl_available: dict[tuple[int, str], int] = field(default_factory=dict)
    nvlink_available: dict[int, int] = field(default_factory=dict)
    execution_ids: set[str] = field(default_factory=set)

    def clone(self) -> _RuntimeState:
        return _RuntimeState(
            host_available=dict(self.host_available),
            gpu_available=dict(self.gpu_available),
            hbm_available=dict(self.hbm_available),
            copy_available=dict(self.copy_available),
            control_available=dict(self.control_available),
            nccl_available=dict(self.nccl_available),
            nvlink_available=dict(self.nvlink_available),
            execution_ids=set(self.execution_ids),
        )


@dataclass
class _ScheduledOperation:
    operation: ExecutionOperation
    logical_completed_at_ps: int
    physical_completed_at_ps: int
    participant_completed_at_ps: dict[int, int]
    visits: list[QueueVisit]
    logical_paths: list[list[QueueVisit]]
    eligible_at_ps: int
    critical_predecessor_id: str | None
    nccl_command_id: str | None = None
    wqes: list[WqeLifecycleProjection] = field(default_factory=list)


@dataclass(frozen=True)
class _SemanticSendSchedule:
    visits: tuple[QueueVisit, ...]
    projection: WqeLifecycleProjection | None

    @property
    def completed_at_ps(self) -> int:
        if not self.visits:
            raise RuntimeError("semantic send produced no resource visit")
        return self.visits[-1].completed_at_ps


_DEVICE_ENDPOINT_RE = re.compile(r"^(?:gpu|cuda|rank):(\d+)(?::|$)", re.IGNORECASE)


def _dma_endpoint(value: str, local_rank: int) -> tuple[str, int | None]:
    lowered = value.strip().lower()
    if lowered == "host" or lowered.startswith("host:"):
        return "host", None
    match = _DEVICE_ENDPOINT_RE.match(lowered)
    if match is not None:
        return "device", int(match.group(1))
    if lowered in {"device", "hbm", "local", "local:hbm"}:
        return "device", local_rank
    raise ValueError(
        f"DMA endpoint {value!r} must identify host or a gpu/cuda/rank endpoint"
    )


def _dma_endpoints(
    work: DmaWork,
    local_rank: int,
) -> tuple[tuple[str, int | None], tuple[str, int | None]]:
    return (
        _dma_endpoint(work.source, local_rank),
        _dma_endpoint(work.destination, local_rank),
    )


def _dma_direction(work: DmaWork, local_rank: int) -> CopyDirection:
    from simllm.compute import CopyDirection

    (source_kind, source_rank), (destination_kind, destination_rank) = _dma_endpoints(
        work,
        local_rank,
    )
    if source_kind == "host" and destination_kind == "device":
        return CopyDirection.HOST_TO_DEVICE
    if source_kind == "device" and destination_kind == "host":
        return CopyDirection.DEVICE_TO_HOST
    if source_kind == destination_kind == "device":
        if source_rank == destination_rank:
            return CopyDirection.DEVICE_TO_DEVICE
        return CopyDirection.PEER_TO_PEER
    raise ValueError("DMA host-to-host descriptors are outside the device runtime")


def collective_goal_tags(
    graph: ExecutionGraph,
    *,
    base_tag: int = DEFAULT_GOAL_BASE_TAG,
) -> dict[str, tuple[int, ...]]:
    """Return collective tags matching the frozen serial GOAL allocation."""

    validate_execution_graph(graph)
    _require_int("base_tag", base_tag)
    next_ring_tag = base_tag
    result: dict[str, tuple[int, ...]] = {}
    pairwise: list[ExecutionOperation] = []
    for operation in graph.operations:
        work = operation.work
        if not isinstance(work, CollectiveWork):
            continue
        if work.collective == "all-reduce" and work.algorithm_hint == "ring":
            rounds = 2 * (len(work.ranks) - 1)
            if rounds <= 0:
                raise ValueError("ring all-reduce requires at least two ranks")
            result[operation.operation_id] = tuple(
                range(next_ring_tag, next_ring_tag + rounds)
            )
            next_ring_tag += rounds
        elif work.collective == "all-to-allv" and work.algorithm_hint == "pairwise":
            pairwise.append(operation)
        else:
            raise ValueError(
                f"unsupported collective {work.collective!r} with algorithm "
                f"{work.algorithm_hint!r}"
            )
    for offset, operation in enumerate(pairwise):
        result[operation.operation_id] = (next_ring_tag + offset,)
    collective_tag_limit = base_tag + CONTROL_GOAL_TAG_OFFSET
    allocated_tags = tuple(tag for tags in result.values() for tag in tags)
    if allocated_tags and max(allocated_tags) >= collective_tag_limit:
        raise ValueError(
            "collective GOAL tags overlap the control-tag range reserved at "
            f"base + {CONTROL_GOAL_TAG_OFFSET}"
        )
    return result


class CoarseDeviceRuntime:
    """Execute graphs on the fixed eight-GPU coarse resource profile."""

    def __init__(
        self,
        profile: CoarseDeviceProfile | None = None,
        *,
        authority_mode: RnicAuthorityMode = RnicAuthorityMode.BYPASS,
        native_session: NativeRnicSession | None = None,
        arbitration_policy: ArbitrationPolicy | None = None,
        kernel_services: Mapping[int, SmSchedulerModel] | None = None,
        kernel_launches: Mapping[str, KernelLaunch] | None = None,
    ) -> None:
        from simllm.compute import KernelLaunch, SmSchedulerModel

        self.profile = profile or CoarseDeviceProfile()
        if not isinstance(self.profile, CoarseDeviceProfile):
            raise TypeError("profile must be a CoarseDeviceProfile")
        if not isinstance(authority_mode, RnicAuthorityMode):
            raise TypeError("authority_mode must be a RnicAuthorityMode")
        if arbitration_policy is not None and not isinstance(
            arbitration_policy,
            ArbitrationPolicy,
        ):
            raise TypeError("arbitration_policy must implement ArbitrationPolicy")
        self.authority_mode = authority_mode
        self.arbitration_policy = arbitration_policy
        self.kernel_services = dict(kernel_services or {})
        self.kernel_launches = dict(kernel_launches or {})
        for rank, service in self.kernel_services.items():
            _require_int("kernel service rank", rank)
            if not isinstance(service, SmSchedulerModel):
                raise TypeError("kernel_services values must be SmSchedulerModel instances")
        for key, launch in self.kernel_launches.items():
            _require_text("kernel launch key", key)
            if not isinstance(launch, KernelLaunch):
                raise TypeError("kernel_launches values must be KernelLaunch records")

        if authority_mode is RnicAuthorityMode.BYPASS:
            if native_session is not None:
                raise ValueError("bypass mode cannot also supply a native RNIC session")
            self._bypass_ledger: AtlahsWqeLedger | None = AtlahsWqeLedger(self.profile)
            self._native_session = None
        else:
            if native_session is None:
                raise ValueError("structural mode requires a native RNIC session")
            if not isinstance(native_session, NativeRnicSession):
                raise TypeError("native_session does not implement NativeRnicSession")
            self._bypass_ledger = None
            self._native_session = native_session
        self._state = _RuntimeState()
        self.last_report: RuntimeReport | None = None

    @property
    def bypass_ledger(self) -> AtlahsWqeLedger | None:
        """Expose the selected bypass authority for audit, never as a second seam."""

        return self._bypass_ledger

    @property
    def authority_name(self) -> str:
        if self._bypass_ledger is not None:
            return self._bypass_ledger.authority_name
        assert self._native_session is not None
        return self._native_session.authority_name

    def execute(
        self,
        graph: ExecutionGraph,
        *,
        on_event: CompletionHandler | None = None,
        bookkeeping: RequestBookkeeper | None = None,
    ) -> ExecutionResult:
        """Advance ``graph`` to its logical boundary and physical quiescence."""

        validate_execution_graph(graph)
        self._preflight(graph)
        if graph.execution_id in self._state.execution_ids:
            raise ValueError(f"execution ID {graph.execution_id!r} was already executed")
        if on_event is not None and not callable(on_event):
            raise TypeError("on_event must be callable")
        if bookkeeping is not None and not isinstance(bookkeeping, RequestBookkeeper):
            raise TypeError("bookkeeping must be a RequestBookkeeper")

        state = self._state.clone()
        bypass = self._bypass_ledger.clone() if self._bypass_ledger is not None else None
        native_transaction = (
            self._native_session.begin_transaction()
            if self._native_session is not None
            else None
        )
        if native_transaction is not None:
            if not isinstance(native_transaction, NativeRnicTransaction):
                raise TypeError(
                    "native session begin_transaction() must return "
                    "NativeRnicTransaction"
                )
            if native_transaction.authority_name != self.authority_name:
                raise ValueError("native transaction authority does not match its session")
        wqe_authority: AtlahsWqeLedger | NativeRnicTransaction | None = (
            bypass if bypass is not None else native_transaction
        )
        assert wqe_authority is not None
        native_committed = False
        try:
            tags = collective_goal_tags(graph, base_tag=self.profile.goal_base_tag)
            operation_by_id = {
                operation.operation_id: operation for operation in graph.operations
            }
            operation_index = {
                operation.operation_id: index
                for index, operation in enumerate(graph.operations)
            }
            queue_predecessor = self._queue_predecessors(graph)
            launch_visits = self._schedule_launches(graph, state)
            scheduled: dict[str, _ScheduledOperation] = {}
            unscheduled = {operation.operation_id for operation in graph.operations}
            all_wqes: list[WqeLifecycleProjection] = []

            while unscheduled:
                ready_data: dict[
                    str, tuple[dict[int, int], int, str | None]
                ] = {}
                for operation in graph.operations:
                    operation_id = operation.operation_id
                    if operation_id not in unscheduled:
                        continue
                    predecessors = set(operation.depends_on)
                    predecessors.update(operation.participant_local_depends_on)
                    previous = queue_predecessor.get(operation_id)
                    if previous is not None:
                        predecessors.add(previous)
                    if not predecessors.issubset(scheduled):
                        continue
                    by_rank, eligible_at_ps, critical_predecessor = (
                        self._operation_readiness(
                            graph,
                            operation,
                            launch_visits[operation_id],
                            scheduled,
                            previous,
                        )
                    )
                    ready_data[operation_id] = (
                        by_rank,
                        eligible_at_ps,
                        critical_predecessor,
                    )
                if not ready_data:
                    raise RuntimeError("validated graph made no scheduling progress")

                selected_id = self._select_ready_operation(
                    ready_data,
                    operation_by_id,
                    operation_index,
                )
                selected = operation_by_id[selected_id]
                if isinstance(selected.work, ComputeWork):
                    group_ids = self._compute_group(
                        selected,
                        ready_data,
                        operation_by_id,
                        operation_index,
                        state,
                    )
                    group = self._schedule_compute_group(
                        graph,
                        tuple(
                            operation_by_id[operation_id]
                            for operation_id in group_ids
                        ),
                        ready_data,
                        launch_visits,
                        state,
                    )
                    for operation_id, outcome in group.items():
                        scheduled[operation_id] = outcome
                        unscheduled.remove(operation_id)
                    continue

                by_rank, eligible_at_ps, critical_predecessor = ready_data[selected_id]
                outcome = self._schedule_noncompute(
                    graph,
                    selected,
                    by_rank,
                    eligible_at_ps,
                    critical_predecessor,
                    launch_visits[selected_id],
                    tags,
                    state,
                    wqe_authority,
                    operation_index[selected_id],
                )
                scheduled[selected_id] = outcome
                unscheduled.remove(selected_id)
                all_wqes.extend(outcome.wqes)

            required_ids = graph.completion_operation_ids or tuple(operation_by_id)
            completed_at_ps = max(
                (
                    scheduled[operation_id].logical_completed_at_ps
                    for operation_id in required_ids
                ),
                default=graph.released_at_ps,
            )
            quiesced_at_ps = max(
                (outcome.physical_completed_at_ps for outcome in scheduled.values()),
                default=graph.released_at_ps,
            )
            events = self._completion_events(graph, scheduled, all_wqes)
            result = ExecutionResult(
                execution_id=graph.execution_id,
                completed_at_ps=completed_at_ps,
                events=events,
                quiesced_at_ps=quiesced_at_ps,
            )
            execution_result_to_json(result)
            report = self._runtime_report(
                graph,
                scheduled,
                all_wqes,
                required_ids,
                wqe_authority,
            )

            if bookkeeping is not None:
                self._validate_bookkeeping_append(
                    bookkeeping,
                    graph,
                    scheduled,
                    all_wqes,
                    events,
                )
            if native_transaction is not None:
                native_transaction.prepare()
            if bookkeeping is not None:
                self._append_bookkeeping(
                    bookkeeping,
                    graph,
                    scheduled,
                    all_wqes,
                    events,
                )
            if native_transaction is not None:
                native_transaction.commit()
                native_committed = True

            state.execution_ids.add(graph.execution_id)
            self._state = state
            if bypass is not None:
                self._bypass_ledger = bypass
            self.last_report = report
        except BaseException:
            if native_transaction is not None and not native_committed:
                native_transaction.abort()
            raise
        if on_event is not None:
            for event in events:
                on_event(event)
        return result

    def _preflight(self, graph: ExecutionGraph) -> None:
        collective_goal_tags(graph, base_tag=self.profile.goal_base_tag)
        for operation in graph.operations:
            work = operation.work
            self.profile.node_gpu(operation.rank)
            if isinstance(work, ComputeWork):
                if operation.rank in self.kernel_services:
                    self._kernel_launch(operation)
                elif work.nominal_duration_ps is None:
                    raise ValueError(
                        f"compute operation {operation.operation_id!r} needs a nominal "
                        "duration or a mapped concurrent kernel service"
                    )
            elif isinstance(work, DmaWork):
                direction = _dma_direction(work, operation.rank)
                for endpoint_kind, endpoint_rank in _dma_endpoints(
                    work,
                    operation.rank,
                ):
                    if endpoint_kind == "device":
                        assert endpoint_rank is not None
                        self.profile.node_gpu(endpoint_rank)
                candidates = self._copy_candidates(direction)
                if not candidates:
                    raise ValueError(
                        f"DMA operation {operation.operation_id!r} has no copy engine "
                        f"supporting {direction.value}"
                    )
            elif isinstance(work, CollectiveWork):
                for rank in work.ranks:
                    self.profile.node_gpu(rank)
                if work.collective == "all-reduce" and work.algorithm_hint == "ring":
                    rank_count = len(work.ranks)
                    if work.payload_bytes < rank_count:
                        raise ValueError(
                            "ring all-reduce payload must provide at least one byte "
                            "per rank; CORE-16 owns remainder chunking"
                        )
                    if work.payload_bytes % rank_count:
                        raise ValueError(
                            "ring all-reduce payload must divide evenly among ranks; "
                            "CORE-16 owns remainder chunking"
                        )
                if (
                    work.collective == "all-to-allv"
                    and work.payload_bytes == 0
                    and not work.pair_payload_bytes
                ):
                    raise ValueError("pairwise all-to-allv requires a nonzero payload")
            elif isinstance(work, ControlWork):
                if len(work.destination_ranks) > CONTROL_GOAL_TAG_STRIDE:
                    raise ValueError(
                        "control work supports at most 1024 destinations per "
                        "reserved GOAL-tag block; CORE-16 owns wider allocation"
                    )
                for rank in work.destination_ranks:
                    self.profile.node_gpu(rank)
            elif isinstance(work, KvCacheWork):
                if (
                    work.action in {KvCacheAction.READ, KvCacheAction.WRITE}
                    and work.byte_count > 0
                ):
                    raise ValueError(
                        "byte-carrying KV READ/WRITE requires CORE-3 HBM lowering"
                    )
            else:
                raise TypeError(f"unsupported work payload {type(work).__name__}")

    @staticmethod
    def _queue_predecessors(graph: ExecutionGraph) -> dict[str, str]:
        tails: dict[tuple[int, str], str] = {}
        result: dict[str, str] = {}
        for operation in graph.operations:
            key = (operation.rank, operation.logical_queue)
            previous = tails.get(key)
            if previous is not None:
                result[operation.operation_id] = previous
            tails[key] = operation.operation_id
        return result

    def _schedule_launches(
        self,
        graph: ExecutionGraph,
        state: _RuntimeState,
    ) -> dict[str, QueueVisit]:
        result: dict[str, QueueVisit] = {}
        for operation in graph.operations:
            node, _ = self.profile.node_gpu(operation.rank)
            eligible_at_ps = max(graph.released_at_ps, operation.not_before_ps)
            started_at_ps = max(eligible_at_ps, state.host_available.get(node, 0))
            finished_at_ps = started_at_ps + self.profile.launch_service_ps
            visit = QueueVisit(
                execution_id=graph.execution_id,
                operation_id=operation.operation_id,
                resource=ResourceRef(
                    ResourceKind.HOST_LAUNCH_QUEUE,
                    f"node-{node}:framework-launch",
                ),
                submitted_at_ps=graph.released_at_ps,
                eligible_at_ps=eligible_at_ps,
                started_at_ps=started_at_ps,
                finished_at_ps=finished_at_ps,
                completed_at_ps=finished_at_ps,
            )
            result[operation.operation_id] = visit
            state.host_available[node] = finished_at_ps
        return result

    @staticmethod
    def _operation_ranks(operation: ExecutionOperation) -> tuple[int, ...]:
        if isinstance(operation.work, CollectiveWork):
            return operation.work.ranks
        return (operation.rank,)

    def _operation_readiness(
        self,
        graph: ExecutionGraph,
        operation: ExecutionOperation,
        launch: QueueVisit,
        scheduled: Mapping[str, _ScheduledOperation],
        queue_predecessor: str | None,
    ) -> tuple[dict[int, int], int, str | None]:
        ranks = self._operation_ranks(operation)
        readiness = {rank: launch.completed_at_ps for rank in ranks}
        predecessor_times: list[tuple[int, str]] = []
        for predecessor_id in operation.depends_on:
            completed = scheduled[predecessor_id].logical_completed_at_ps
            predecessor_times.append((completed, predecessor_id))
            for rank in ranks:
                readiness[rank] = max(readiness[rank], completed)
        if queue_predecessor is not None:
            completed = scheduled[queue_predecessor].logical_completed_at_ps
            predecessor_times.append((completed, queue_predecessor))
            for rank in ranks:
                readiness[rank] = max(readiness[rank], completed)
        for predecessor_id in operation.participant_local_depends_on:
            predecessor = scheduled[predecessor_id]
            for rank in ranks:
                completed = predecessor.participant_completed_at_ps.get(rank)
                if completed is None:
                    continue
                readiness[rank] = max(readiness[rank], completed)
                predecessor_times.append((completed, predecessor_id))
        critical = (
            max(predecessor_times, key=lambda item: (item[0], item[1]))[1]
            if predecessor_times
            else None
        )
        return readiness, min(readiness.values()), critical

    def _select_ready_operation(
        self,
        ready_data: Mapping[str, tuple[dict[int, int], int, str | None]],
        operation_by_id: Mapping[str, ExecutionOperation],
        operation_index: Mapping[str, int],
    ) -> str:
        earliest = min(value[1] for value in ready_data.values())
        candidates = tuple(
            ArbitrationCandidate(
                operation_id=operation_id,
                baseline_sequence=operation_index[operation_id],
                eligible_at_ps=eligible_at_ps,
                class_label=operation_by_id[operation_id].priority,
            )
            for operation_id, (_, eligible_at_ps, _) in ready_data.items()
            if eligible_at_ps == earliest
        )
        policy = self.arbitration_policy or IdentityArbitrationPolicy()
        selected = policy.select(candidates)
        if selected not in candidates:
            raise ValueError("arbitration policy selected a candidate outside the legal set")
        return selected.operation_id

    def _kernel_launch(self, operation: ExecutionOperation) -> KernelLaunch:
        assert isinstance(operation.work, ComputeWork)
        launch = self.kernel_launches.get(operation.operation_id)
        if launch is None:
            launch = self.kernel_launches.get(operation.work.kernel)
        if launch is None:
            raise KeyError(
                f"no KernelLaunch mapping for operation {operation.operation_id!r} "
                f"or kernel {operation.work.kernel!r}"
            )
        return launch

    def _compute_group(
        self,
        selected: ExecutionOperation,
        ready_data: Mapping[str, tuple[dict[int, int], int, str | None]],
        operation_by_id: Mapping[str, ExecutionOperation],
        operation_index: Mapping[str, int],
        state: _RuntimeState,
    ) -> tuple[str, ...]:
        assert isinstance(selected.work, ComputeWork)
        rank = selected.rank
        selected_eligible = ready_data[selected.operation_id][0][rank]
        start = max(selected_eligible, state.gpu_available.get(rank, 0))
        if selected.work.hbm_bytes:
            start = max(start, state.hbm_available.get(rank, 0))
        candidates: list[str] = []
        for operation_id, (by_rank, _, _) in ready_data.items():
            operation = operation_by_id[operation_id]
            if not isinstance(operation.work, ComputeWork) or operation.rank != rank:
                continue
            if by_rank[rank] > start:
                continue
            if operation.work.hbm_bytes and state.hbm_available.get(rank, 0) > start:
                continue
            candidates.append(operation_id)
        return tuple(sorted(candidates, key=operation_index.__getitem__))

    def _schedule_compute_group(
        self,
        graph: ExecutionGraph,
        operations: tuple[ExecutionOperation, ...],
        ready_data: Mapping[str, tuple[dict[int, int], int, str | None]],
        launches: Mapping[str, QueueVisit],
        state: _RuntimeState,
    ) -> dict[str, _ScheduledOperation]:
        from simllm.compute import GpuTask, GpuTaskKind

        rank = operations[0].rank
        if any(operation.rank != rank for operation in operations):
            raise AssertionError("concurrent compute group crossed GPUs")
        eligible = {
            operation.operation_id: ready_data[operation.operation_id][0][rank]
            for operation in operations
        }
        started_at_ps = max(
            max(eligible.values()),
            state.gpu_available.get(rank, 0),
        )
        uses_hbm = any(
            isinstance(operation.work, ComputeWork) and operation.work.hbm_bytes
            for operation in operations
        )
        if uses_hbm:
            started_at_ps = max(started_at_ps, state.hbm_available.get(rank, 0))

        completions: dict[str, int] = {}
        service = self.kernel_services.get(rank)
        if service is None:
            for operation in operations:
                assert isinstance(operation.work, ComputeWork)
                assert operation.work.nominal_duration_ps is not None
                completions[operation.operation_id] = (
                    started_at_ps + operation.work.nominal_duration_ps
                )
        else:
            tasks = tuple(
                GpuTask(
                    task_id=operation.operation_id,
                    kind=(
                        GpuTaskKind.MEMORY
                        if isinstance(operation.work, ComputeWork)
                        and operation.work.flops == 0
                        and operation.work.hbm_bytes > 0
                        else GpuTaskKind.COMPUTE
                    ),
                    launch=self._kernel_launch(operation),
                )
                for operation in operations
            )
            estimate = service.estimate_concurrent(tasks)
            task_estimates = {task.task_id: task for task in estimate.tasks}
            for operation in operations:
                task = task_estimates[operation.operation_id]
                completions[operation.operation_id] = started_at_ps + _cycles_to_ps(
                    task.completion_cycle,
                    service.architecture.clock_hz,
                )

        batch_finished = max(completions.values())
        state.gpu_available[rank] = batch_finished
        if uses_hbm:
            state.hbm_available[rank] = batch_finished
        node, gpu = self.profile.node_gpu(rank)
        outcomes: dict[str, _ScheduledOperation] = {}
        for operation in operations:
            assert isinstance(operation.work, ComputeWork)
            operation_id = operation.operation_id
            completed = completions[operation_id]
            gpu_visit = QueueVisit(
                execution_id=graph.execution_id,
                operation_id=operation_id,
                resource=ResourceRef(
                    ResourceKind.GPU_WORK_QUEUE,
                    f"node-{node}:gpu-{gpu}:work",
                ),
                submitted_at_ps=launches[operation_id].completed_at_ps,
                eligible_at_ps=eligible[operation_id],
                started_at_ps=started_at_ps,
                finished_at_ps=completed,
                completed_at_ps=completed + self.profile.completion_delivery_ps,
                service_bytes=operation.work.hbm_bytes,
            )
            visits = [launches[operation_id], gpu_visit]
            if operation.work.hbm_bytes:
                visits.append(
                    QueueVisit(
                        execution_id=graph.execution_id,
                        operation_id=operation_id,
                        resource=ResourceRef(
                            ResourceKind.HBM_QUEUE,
                            f"node-{node}:gpu-{gpu}:hbm",
                        ),
                        submitted_at_ps=launches[operation_id].completed_at_ps,
                        eligible_at_ps=eligible[operation_id],
                        started_at_ps=started_at_ps,
                        finished_at_ps=completed,
                        completed_at_ps=completed,
                        service_bytes=operation.work.hbm_bytes,
                    )
                )
            logical_completed = gpu_visit.completed_at_ps
            outcomes[operation_id] = _ScheduledOperation(
                operation=operation,
                logical_completed_at_ps=logical_completed,
                physical_completed_at_ps=logical_completed,
                participant_completed_at_ps={rank: logical_completed},
                visits=visits,
                logical_paths=[[launches[operation_id], gpu_visit]],
                eligible_at_ps=eligible[operation_id],
                critical_predecessor_id=ready_data[operation_id][2],
            )
        return outcomes

    def _copy_candidates(
        self,
        direction: CopyDirection,
    ) -> tuple[CopyEngineServiceModel, ...]:
        return tuple(
            engine
            for engine in self.profile.copy_engines
            if direction in engine.engine.directions
        )

    def _schedule_noncompute(
        self,
        graph: ExecutionGraph,
        operation: ExecutionOperation,
        by_rank: dict[int, int],
        eligible_at_ps: int,
        critical_predecessor: str | None,
        launch: QueueVisit,
        tags: Mapping[str, tuple[int, ...]],
        state: _RuntimeState,
        wqe_authority: AtlahsWqeLedger | NativeRnicTransaction,
        operation_index: int,
    ) -> _ScheduledOperation:
        if isinstance(operation.work, DmaWork):
            return self._schedule_dma(
                graph,
                operation,
                by_rank[operation.rank],
                critical_predecessor,
                launch,
                state,
            )
        if isinstance(operation.work, KvCacheWork):
            completed = max(launch.completed_at_ps, by_rank[operation.rank])
            return _ScheduledOperation(
                operation=operation,
                logical_completed_at_ps=completed,
                physical_completed_at_ps=completed,
                participant_completed_at_ps={operation.rank: completed},
                visits=[launch],
                logical_paths=[[launch]],
                eligible_at_ps=eligible_at_ps,
                critical_predecessor_id=critical_predecessor,
            )
        if isinstance(operation.work, ControlWork):
            return self._schedule_control(
                graph,
                operation,
                by_rank[operation.rank],
                critical_predecessor,
                launch,
                state,
                wqe_authority,
                operation_index,
            )
        if isinstance(operation.work, CollectiveWork):
            return self._schedule_collective(
                graph,
                operation,
                by_rank,
                critical_predecessor,
                launch,
                tags[operation.operation_id],
                state,
                wqe_authority,
            )
        raise AssertionError("preflight accepted unsupported work")

    def _schedule_dma(
        self,
        graph: ExecutionGraph,
        operation: ExecutionOperation,
        eligible_at_ps: int,
        critical_predecessor: str | None,
        launch: QueueVisit,
        state: _RuntimeState,
    ) -> _ScheduledOperation:
        from simllm.compute import CopyTransfer

        work = operation.work
        assert isinstance(work, DmaWork)
        direction = _dma_direction(work, operation.rank)
        candidates = self._copy_candidates(direction)
        transfer = None
        if work.byte_count > 0:
            transfer = CopyTransfer(
                transfer_id=work.descriptor_id,
                direction=direction,
                bytes=work.byte_count,
                source=work.source,
                destination=work.destination,
            )
        choices: list[tuple[int, int, CopyEngineServiceModel, int]] = []
        for index, engine in enumerate(candidates):
            started = max(
                eligible_at_ps,
                state.copy_available.get((operation.rank, engine.engine_id), 0),
                state.hbm_available.get(operation.rank, 0),
            )
            duration = 0 if transfer is None else engine.estimate(transfer).duration_ps
            choices.append((started, index, engine, duration))
        started_at_ps, _, engine, duration_ps = min(choices, key=lambda item: item[:2])
        finished_at_ps = started_at_ps + duration_ps
        completed_at_ps = finished_at_ps + self.profile.completion_delivery_ps
        state.copy_available[(operation.rank, engine.engine_id)] = finished_at_ps
        state.hbm_available[operation.rank] = finished_at_ps
        node, gpu = self.profile.node_gpu(operation.rank)
        copy_visit = QueueVisit(
            execution_id=graph.execution_id,
            operation_id=operation.operation_id,
            resource=ResourceRef(
                ResourceKind.COPY_ENGINE,
                f"node-{node}:gpu-{gpu}:copy:{engine.engine_id}",
            ),
            submitted_at_ps=launch.completed_at_ps,
            eligible_at_ps=eligible_at_ps,
            started_at_ps=started_at_ps,
            finished_at_ps=finished_at_ps,
            completed_at_ps=completed_at_ps,
            service_bytes=work.byte_count,
        )
        hbm_visit = QueueVisit(
            execution_id=graph.execution_id,
            operation_id=operation.operation_id,
            resource=ResourceRef(
                ResourceKind.HBM_QUEUE,
                f"node-{node}:gpu-{gpu}:hbm",
            ),
            submitted_at_ps=launch.completed_at_ps,
            eligible_at_ps=eligible_at_ps,
            started_at_ps=started_at_ps,
            finished_at_ps=finished_at_ps,
            completed_at_ps=finished_at_ps,
            service_bytes=work.byte_count,
        )
        participant_ranks = {operation.rank}
        for endpoint_kind, endpoint_rank in _dma_endpoints(work, operation.rank):
            if endpoint_kind == "device":
                assert endpoint_rank is not None
                participant_ranks.add(endpoint_rank)
        return _ScheduledOperation(
            operation=operation,
            logical_completed_at_ps=completed_at_ps,
            physical_completed_at_ps=completed_at_ps,
            participant_completed_at_ps={
                rank: completed_at_ps for rank in participant_ranks
            },
            visits=[launch, copy_visit, hbm_visit],
            logical_paths=[[launch, copy_visit]],
            eligible_at_ps=eligible_at_ps,
            critical_predecessor_id=critical_predecessor,
        )

    def _schedule_control(
        self,
        graph: ExecutionGraph,
        operation: ExecutionOperation,
        eligible_at_ps: int,
        critical_predecessor: str | None,
        launch: QueueVisit,
        state: _RuntimeState,
        wqe_authority: AtlahsWqeLedger | NativeRnicTransaction,
        operation_index: int,
    ) -> _ScheduledOperation:
        work = operation.work
        assert isinstance(work, ControlWork)
        rank = operation.rank
        node, gpu = self.profile.node_gpu(rank)
        started_at_ps = max(eligible_at_ps, state.control_available.get(rank, 0))
        finished_at_ps = started_at_ps + self.profile.control_service_ps
        control_visit = QueueVisit(
            execution_id=graph.execution_id,
            operation_id=operation.operation_id,
            resource=ResourceRef(
                ResourceKind.CONTROL_QUEUE,
                f"node-{node}:gpu-{gpu}:control",
            ),
            submitted_at_ps=launch.completed_at_ps,
            eligible_at_ps=eligible_at_ps,
            started_at_ps=started_at_ps,
            finished_at_ps=finished_at_ps,
            completed_at_ps=finished_at_ps,
            service_bytes=work.payload_bytes,
        )
        state.control_available[rank] = finished_at_ps
        visits = [launch, control_visit]
        physical_paths: list[list[QueueVisit]] = [[launch, control_visit]]
        wqes: list[WqeLifecycleProjection] = []
        physical_completed = control_visit.completed_at_ps
        participant_completed = {rank: control_visit.completed_at_ps}
        for extent_index, destination in enumerate(work.destination_ranks):
            if destination == rank:
                continue
            tag = (
                self.profile.goal_base_tag
                + CONTROL_GOAL_TAG_OFFSET
                + operation_index * CONTROL_GOAL_TAG_STRIDE
                + extent_index
            )
            transfer = self._schedule_semantic_send(
                graph=graph,
                operation=operation,
                source_rank=rank,
                destination_rank=destination,
                payload_bytes=work.payload_bytes,
                goal_tag=tag,
                extent_index=extent_index,
                channel_id=f"control:{work.message}",
                submitted_at_ps=control_visit.completed_at_ps,
                eligible_at_ps=control_visit.completed_at_ps,
                nccl_command_id=None,
                state=state,
                wqe_authority=wqe_authority,
            )
            visits.extend(transfer.visits)
            physical_paths.append([launch, control_visit, *transfer.visits])
            physical_completed = max(physical_completed, transfer.completed_at_ps)
            participant_completed[destination] = max(
                participant_completed.get(destination, 0),
                transfer.completed_at_ps,
            )
            if transfer.projection is not None:
                wqes.append(transfer.projection)
        if work.mode is ControlMode.SYNCHRONOUS:
            logical_completed = physical_completed
            logical_paths = physical_paths
        else:
            logical_completed = control_visit.completed_at_ps
            logical_paths = [[launch, control_visit]]
        participant_completed[rank] = logical_completed
        return _ScheduledOperation(
            operation=operation,
            logical_completed_at_ps=logical_completed,
            physical_completed_at_ps=physical_completed,
            participant_completed_at_ps=participant_completed,
            visits=visits,
            logical_paths=logical_paths,
            eligible_at_ps=eligible_at_ps,
            critical_predecessor_id=critical_predecessor,
            wqes=wqes,
        )

    def _schedule_collective(
        self,
        graph: ExecutionGraph,
        operation: ExecutionOperation,
        readiness: dict[int, int],
        critical_predecessor: str | None,
        launch: QueueVisit,
        goal_tags: tuple[int, ...],
        state: _RuntimeState,
        wqe_authority: AtlahsWqeLedger | NativeRnicTransaction,
    ) -> _ScheduledOperation:
        work = operation.work
        assert isinstance(work, CollectiveWork)
        command_id = (
            "nccl-command:"
            f"{_escape_id(graph.execution_id)}:{_escape_id(operation.operation_id)}"
        )
        channel = work.channel_hint or "default"
        visits: list[QueueVisit] = [launch]
        paths: list[list[QueueVisit]] = []
        wqes: list[WqeLifecycleProjection] = []
        participant_completed = dict(readiness)
        extent_index = 0

        if work.collective == "all-reduce" and work.algorithm_hint == "ring":
            chunk_bytes = work.payload_bytes // len(work.ranks)
            frontier = dict(readiness)
            frontier_paths = {rank: [launch] for rank in work.ranks}
            for round_index, goal_tag in enumerate(goal_tags):
                round_records: list[
                    tuple[
                        int,
                        int,
                        _SemanticSendSchedule,
                        list[QueueVisit],
                    ]
                ] = []
                for index, source_rank in enumerate(work.ranks):
                    destination_rank = work.ranks[(index + 1) % len(work.ranks)]
                    channel_visit = self._schedule_nccl_channel(
                        graph,
                        operation,
                        source_rank,
                        channel,
                        launch.completed_at_ps,
                        frontier[source_rank],
                        state,
                    )
                    transfer = self._schedule_semantic_send(
                        graph=graph,
                        operation=operation,
                        source_rank=source_rank,
                        destination_rank=destination_rank,
                        payload_bytes=chunk_bytes,
                        goal_tag=goal_tag,
                        extent_index=extent_index,
                        channel_id=f"{channel}:round-{round_index}",
                        submitted_at_ps=channel_visit.completed_at_ps,
                        eligible_at_ps=channel_visit.completed_at_ps,
                        nccl_command_id=command_id,
                        state=state,
                        wqe_authority=wqe_authority,
                    )
                    extent_index += 1
                    visits.append(channel_visit)
                    visits.extend(transfer.visits)
                    source_path = [
                        *frontier_paths[source_rank],
                        channel_visit,
                        *transfer.visits,
                    ]
                    if transfer.projection is not None:
                        wqes.append(transfer.projection)
                    round_records.append(
                        (
                            source_rank,
                            destination_rank,
                            transfer,
                            source_path,
                        )
                    )
                next_frontier = dict(frontier)
                next_frontier_paths = dict(frontier_paths)
                for (
                    source_rank,
                    destination_rank,
                    transfer,
                    source_path,
                ) in round_records:
                    for rank in (source_rank, destination_rank):
                        if transfer.completed_at_ps >= next_frontier[rank]:
                            next_frontier[rank] = transfer.completed_at_ps
                            next_frontier_paths[rank] = source_path
                frontier = next_frontier
                frontier_paths = next_frontier_paths
            participant_completed = frontier
            paths = list(frontier_paths.values())
        elif work.collective == "all-to-allv" and work.algorithm_hint == "pairwise":
            goal_tag = goal_tags[0]
            if work.pair_payload_bytes:
                pair_payloads = work.pair_payload_bytes
            else:
                pair_payloads = tuple(
                    (source_rank, destination_rank, work.payload_bytes)
                    for source_rank in work.ranks
                    for destination_rank in work.ranks
                    if source_rank != destination_rank
                )
            for source_rank, destination_rank, payload_bytes in pair_payloads:
                channel_visit = self._schedule_nccl_channel(
                    graph,
                    operation,
                    source_rank,
                    channel,
                    launch.completed_at_ps,
                    readiness[source_rank],
                    state,
                )
                transfer = self._schedule_semantic_send(
                    graph=graph,
                    operation=operation,
                    source_rank=source_rank,
                    destination_rank=destination_rank,
                    payload_bytes=payload_bytes,
                    goal_tag=goal_tag,
                    extent_index=extent_index,
                    channel_id=channel,
                    submitted_at_ps=channel_visit.completed_at_ps,
                    eligible_at_ps=channel_visit.completed_at_ps,
                    nccl_command_id=command_id,
                    state=state,
                    wqe_authority=wqe_authority,
                )
                extent_index += 1
                visits.append(channel_visit)
                visits.extend(transfer.visits)
                paths.append([launch, channel_visit, *transfer.visits])
                if transfer.projection is not None:
                    wqes.append(transfer.projection)
                participant_completed[source_rank] = max(
                    participant_completed[source_rank], transfer.completed_at_ps
                )
                participant_completed[destination_rank] = max(
                    participant_completed[destination_rank], transfer.completed_at_ps
                )
        else:
            raise AssertionError("collective tag preflight accepted an unsupported algorithm")

        logical_completed = max(participant_completed.values())
        if not paths:
            paths = [[launch]]
        return _ScheduledOperation(
            operation=operation,
            logical_completed_at_ps=logical_completed,
            physical_completed_at_ps=logical_completed,
            participant_completed_at_ps=participant_completed,
            visits=visits,
            logical_paths=paths,
            eligible_at_ps=min(readiness.values()),
            critical_predecessor_id=critical_predecessor,
            nccl_command_id=command_id,
            wqes=wqes,
        )

    def _schedule_nccl_channel(
        self,
        graph: ExecutionGraph,
        operation: ExecutionOperation,
        rank: int,
        channel: str,
        submitted_at_ps: int,
        eligible_at_ps: int,
        state: _RuntimeState,
    ) -> QueueVisit:
        key = (rank, channel)
        started_at_ps = max(eligible_at_ps, state.nccl_available.get(key, 0))
        finished_at_ps = started_at_ps + self.profile.nccl_channel_service_ps
        state.nccl_available[key] = finished_at_ps
        node, gpu = self.profile.node_gpu(rank)
        return QueueVisit(
            execution_id=graph.execution_id,
            operation_id=operation.operation_id,
            resource=ResourceRef(
                ResourceKind.NCCL_CHANNEL,
                f"node-{node}:gpu-{gpu}:nccl:{channel}",
            ),
            submitted_at_ps=submitted_at_ps,
            eligible_at_ps=eligible_at_ps,
            started_at_ps=started_at_ps,
            finished_at_ps=finished_at_ps,
            completed_at_ps=finished_at_ps,
        )

    def _schedule_semantic_send(
        self,
        *,
        graph: ExecutionGraph,
        operation: ExecutionOperation,
        source_rank: int,
        destination_rank: int,
        payload_bytes: int,
        goal_tag: int,
        extent_index: int,
        channel_id: str,
        submitted_at_ps: int,
        eligible_at_ps: int,
        nccl_command_id: str | None,
        state: _RuntimeState,
        wqe_authority: AtlahsWqeLedger | NativeRnicTransaction,
    ) -> _SemanticSendSchedule:
        source_node, source_gpu = self.profile.node_gpu(source_rank)
        destination_node, _ = self.profile.node_gpu(destination_rank)
        if source_node == destination_node:
            started_at_ps = max(
                eligible_at_ps,
                state.nvlink_available.get(source_rank, 0),
            )
            finished_at_ps = started_at_ps + _serialization_ps(
                payload_bytes,
                self.profile.nvlink_rate_bps,
            )
            completed_at_ps = finished_at_ps + self.profile.completion_delivery_ps
            state.nvlink_available[source_rank] = finished_at_ps
            return _SemanticSendSchedule(
                visits=(
                    QueueVisit(
                        execution_id=graph.execution_id,
                        operation_id=operation.operation_id,
                        resource=ResourceRef(
                            ResourceKind.NVLINK,
                            f"node-{source_node}:gpu-{source_gpu}:nvlink",
                        ),
                        submitted_at_ps=submitted_at_ps,
                        eligible_at_ps=eligible_at_ps,
                        started_at_ps=started_at_ps,
                        finished_at_ps=finished_at_ps,
                        completed_at_ps=completed_at_ps,
                        service_bytes=payload_bytes,
                    ),
                ),
                projection=None,
            )

        submission = SemanticWqeSubmission(
            execution_id=graph.execution_id,
            operation_id=operation.operation_id,
            source_rank=source_rank,
            destination_rank=destination_rank,
            payload_bytes=payload_bytes,
            goal_tag=goal_tag,
            extent_index=extent_index,
            channel_id=channel_id,
            submitted_at_ps=submitted_at_ps,
            eligible_at_ps=eligible_at_ps,
            class_label=operation.priority,
            nccl_command_id=nccl_command_id,
        )
        projection = wqe_authority.submit(submission)
        self._validate_wqe_projection(submission, projection)
        return _SemanticSendSchedule(
            visits=self._native_wqe_visits(projection),
            projection=projection,
        )

    @staticmethod
    def _native_wqe_visits(
        projection: WqeLifecycleProjection,
    ) -> tuple[QueueVisit, ...]:
        if projection.doorbell_started_at_ps is None:
            return (
                QueueVisit(
                    execution_id=projection.execution_id,
                    operation_id=projection.operation_id,
                    resource=ResourceRef(ResourceKind.NIC, projection.rnic_id),
                    submitted_at_ps=projection.submitted_at_ps,
                    eligible_at_ps=projection.eligible_at_ps,
                    started_at_ps=projection.started_at_ps,
                    finished_at_ps=projection.finished_at_ps,
                    completed_at_ps=projection.completed_at_ps,
                    service_bytes=projection.payload_bytes,
                    subject_object_id=projection.wqe_id,
                ),
            )

        assert projection.doorbell_completed_at_ps is not None
        assert projection.network_eligible_at_ps is not None
        assert projection.network_started_at_ps is not None
        assert projection.network_finished_at_ps is not None
        doorbell = QueueVisit(
            execution_id=projection.execution_id,
            operation_id=projection.operation_id,
            resource=ResourceRef(ResourceKind.NIC, projection.rnic_id),
            submitted_at_ps=projection.submitted_at_ps,
            eligible_at_ps=projection.doorbell_started_at_ps,
            started_at_ps=projection.doorbell_started_at_ps,
            finished_at_ps=projection.doorbell_completed_at_ps,
            completed_at_ps=projection.doorbell_completed_at_ps,
            subject_object_id=projection.wqe_id,
            stage="native_doorbell",
        )
        network = QueueVisit(
            execution_id=projection.execution_id,
            operation_id=projection.operation_id,
            resource=ResourceRef(ResourceKind.NIC, projection.rnic_id),
            submitted_at_ps=projection.submitted_at_ps,
            eligible_at_ps=projection.network_eligible_at_ps,
            started_at_ps=projection.network_started_at_ps,
            finished_at_ps=projection.network_finished_at_ps,
            completed_at_ps=projection.completed_at_ps,
            service_bytes=projection.payload_bytes,
            subject_object_id=projection.wqe_id,
            stage="native_network",
        )
        return (doorbell, network)

    def _validate_wqe_projection(
        self,
        submission: SemanticWqeSubmission,
        projection: WqeLifecycleProjection,
    ) -> None:
        if not isinstance(projection, WqeLifecycleProjection):
            raise TypeError("RNIC session must return a WqeLifecycleProjection")
        expected = {
            "execution_id": submission.execution_id,
            "operation_id": submission.operation_id,
            "source_rank": submission.source_rank,
            "destination_rank": submission.destination_rank,
            "payload_bytes": submission.payload_bytes,
            "goal_tag": submission.goal_tag,
            "extent_index": submission.extent_index,
            "channel_id": submission.channel_id,
            "nccl_command_id": submission.nccl_command_id,
        }
        for name, value in expected.items():
            if getattr(projection, name) != value:
                raise ValueError(f"WQE projection {name} disagrees with semantic submission")
        if projection.authority != self.authority_name:
            raise ValueError("WQE projection authority does not match the selected session")
        if projection.submitted_at_ps != submission.submitted_at_ps:
            raise ValueError("WQE projection changed logical submission time")
        if projection.eligible_at_ps != submission.eligible_at_ps:
            raise ValueError("WQE projection changed external eligibility time")

    @staticmethod
    def _work_completed_bytes(operation: ExecutionOperation) -> int | None:
        work = operation.work
        if isinstance(work, ComputeWork):
            return work.hbm_bytes
        if isinstance(work, DmaWork):
            return work.byte_count
        if isinstance(work, KvCacheWork):
            return work.byte_count
        if isinstance(work, CollectiveWork):
            if work.pair_payload_bytes:
                return sum(entry[2] for entry in work.pair_payload_bytes)
            return work.payload_bytes
        if isinstance(work, ControlWork):
            return work.payload_bytes
        return None

    def _completion_events(
        self,
        graph: ExecutionGraph,
        scheduled: Mapping[str, _ScheduledOperation],
        wqes: Sequence[WqeLifecycleProjection],
    ) -> tuple[CompletionEvent, ...]:
        sequenced: list[tuple[int, int, CompletionEvent]] = []
        sequence = 0

        def add(event: CompletionEvent) -> None:
            nonlocal sequence
            sequenced.append((event.timestamp_ps, sequence, event))
            sequence += 1

        for operation in graph.operations:
            outcome = scheduled[operation.operation_id]
            for visit in outcome.visits:
                if visit.subject_object_id is not None:
                    continue
                for phase, timestamp, completed_bytes in (
                    (EventPhase.SUBMITTED, visit.submitted_at_ps, None),
                    (EventPhase.QUEUED, visit.eligible_at_ps, None),
                    (EventPhase.STARTED, visit.started_at_ps, None),
                    (EventPhase.PROGRESS, visit.finished_at_ps, visit.service_bytes),
                ):
                    add(
                        CompletionEvent(
                            execution_id=graph.execution_id,
                            operation_id=operation.operation_id,
                            phase=phase,
                            timestamp_ps=timestamp,
                            resource=visit.resource,
                            completed_bytes=completed_bytes,
                        )
                    )

        for wqe in wqes:
            queued_at_ps = (
                wqe.network_eligible_at_ps
                if wqe.network_eligible_at_ps is not None
                else wqe.eligible_at_ps
            )
            started_at_ps = (
                wqe.network_started_at_ps
                if wqe.network_started_at_ps is not None
                else wqe.started_at_ps
            )
            finished_at_ps = (
                wqe.network_finished_at_ps
                if wqe.network_finished_at_ps is not None
                else wqe.finished_at_ps
            )
            for phase, timestamp, resource, completed_bytes in (
                (
                    EventPhase.SUBMITTED,
                    wqe.submitted_at_ps,
                    ResourceRef(ResourceKind.NIC_SEND_QUEUE, wqe.sq_id),
                    None,
                ),
                (
                    EventPhase.QUEUED,
                    queued_at_ps,
                    ResourceRef(ResourceKind.NIC, wqe.rnic_id),
                    None,
                ),
                (
                    EventPhase.STARTED,
                    started_at_ps,
                    ResourceRef(ResourceKind.NIC, wqe.rnic_id),
                    None,
                ),
                (
                    EventPhase.PROGRESS,
                    finished_at_ps,
                    ResourceRef(ResourceKind.NIC, wqe.rnic_id),
                    wqe.payload_bytes,
                ),
                (
                    EventPhase.COMPLETED,
                    wqe.completed_at_ps,
                    ResourceRef(ResourceKind.COMPLETION_QUEUE, wqe.cq_id),
                    wqe.payload_bytes,
                ),
            ):
                add(
                    CompletionEvent(
                        execution_id=graph.execution_id,
                        operation_id=wqe.operation_id,
                        phase=phase,
                        timestamp_ps=timestamp,
                        resource=resource,
                        completed_bytes=completed_bytes,
                        subject_object_id=wqe.wqe_id,
                    )
                )

        for operation in graph.operations:
            outcome = scheduled[operation.operation_id]
            final_visit = max(
                (path[-1] for path in outcome.logical_paths),
                key=lambda visit: (visit.completed_at_ps, visit.resource.resource_id),
            )
            add(
                CompletionEvent(
                    execution_id=graph.execution_id,
                    operation_id=operation.operation_id,
                    phase=EventPhase.COMPLETED,
                    timestamp_ps=outcome.logical_completed_at_ps,
                    resource=final_visit.resource,
                    completed_bytes=self._work_completed_bytes(operation),
                )
            )
        sequenced.sort(key=lambda item: (item[0], item[1]))
        return tuple(event for _, _, event in sequenced)

    def _runtime_report(
        self,
        graph: ExecutionGraph,
        scheduled: Mapping[str, _ScheduledOperation],
        wqes: Sequence[WqeLifecycleProjection],
        required_ids: tuple[str, ...],
        wqe_authority: AtlahsWqeLedger | NativeRnicTransaction,
    ) -> RuntimeReport:
        operation_records: list[RuntimeOperationRecord] = []
        all_visits: list[QueueVisit] = []
        by_operation_record: dict[str, RuntimeOperationRecord] = {}
        for operation in graph.operations:
            outcome = scheduled[operation.operation_id]
            all_visits.extend(outcome.visits)
            path = max(
                outcome.logical_paths,
                key=lambda candidate: (
                    candidate[-1].completed_at_ps,
                    candidate[-1].resource.resource_id,
                ),
            )
            launch = path[0]
            predecessor_id = outcome.critical_predecessor_id
            segment_start_ps = (
                scheduled[predecessor_id].logical_completed_at_ps
                if predecessor_id is not None
                else graph.released_at_ps
            )
            cursor_ps = segment_start_ps

            launch_wait_ps = max(
                0,
                launch.started_at_ps - max(launch.eligible_at_ps, cursor_ps),
            )
            launch_service_ps = max(
                0,
                launch.finished_at_ps - max(launch.started_at_ps, cursor_ps),
            )
            launch_visibility_ps = max(
                0,
                launch.completed_at_ps - max(launch.finished_at_ps, cursor_ps),
            )
            launch_queue_ps = (
                launch_wait_ps + launch_service_ps + launch_visibility_ps
            )
            cursor_ps = max(cursor_ps, launch.completed_at_ps)

            device_queue_ps = 0
            service_ps = 0
            completion_delivery_ps = 0
            for visit in path[1:]:
                device_queue_ps += max(
                    0,
                    visit.started_at_ps - max(visit.eligible_at_ps, cursor_ps),
                )
                service_ps += max(
                    0,
                    visit.finished_at_ps - max(visit.started_at_ps, cursor_ps),
                )
                completion_delivery_ps += max(
                    0,
                    visit.completed_at_ps - max(visit.finished_at_ps, cursor_ps),
                )
                cursor_ps = max(cursor_ps, visit.completed_at_ps)

            operation_latency_ps = outcome.logical_completed_at_ps - segment_start_ps
            covered = (
                launch_queue_ps
                + device_queue_ps
                + service_ps
                + completion_delivery_ps
            )
            external_dependency_ps = operation_latency_ps - covered
            if external_dependency_ps < 0:
                raise ValueError(
                    f"operation {operation.operation_id!r} has overlapping visits on its "
                    "selected additive critical path"
                )
            breakdown = CriticalPathBreakdown(
                launch_queue_ps=launch_queue_ps,
                device_queue_ps=device_queue_ps,
                service_ps=service_ps,
                completion_delivery_ps=completion_delivery_ps,
                external_dependency_ps=external_dependency_ps,
                operation_latency_ps=operation_latency_ps,
                critical_path_queue_ps=launch_wait_ps + device_queue_ps,
            )
            attribution = self._critical_path_attribution(
                operation,
                path,
                segment_start_ps,
                outcome.logical_completed_at_ps,
            )
            if attribution.total_ps != operation_latency_ps:
                raise ValueError(
                    f"operation {operation.operation_id!r} attribution does not "
                    "conserve its critical-path segment"
                )
            record = RuntimeOperationRecord(
                operation_id=operation.operation_id,
                class_label=operation.priority,
                submitted_at_ps=graph.released_at_ps,
                eligible_at_ps=outcome.eligible_at_ps,
                completed_at_ps=outcome.logical_completed_at_ps,
                physical_completed_at_ps=outcome.physical_completed_at_ps,
                participant_completed_at_ps=tuple(
                    sorted(outcome.participant_completed_at_ps.items())
                ),
                breakdown=breakdown,
                attribution=attribution,
                critical_predecessor_id=predecessor_id,
                sum_visit_wait_ps=sum(visit.queue_wait_ps for visit in outcome.visits),
            )
            operation_records.append(record)
            by_operation_record[operation.operation_id] = record

        operation_index = {
            operation.operation_id: index for index, operation in enumerate(graph.operations)
        }
        if required_ids:
            endpoint = max(
                required_ids,
                key=lambda operation_id: (
                    scheduled[operation_id].logical_completed_at_ps,
                    operation_index[operation_id],
                ),
            )
            reverse_chain: list[str] = []
            seen: set[str] = set()
            current: str | None = endpoint
            while current is not None:
                if current in seen:
                    raise RuntimeError("critical predecessor chain contains a cycle")
                seen.add(current)
                reverse_chain.append(current)
                current = scheduled[current].critical_predecessor_id
            critical_chain = tuple(reversed(reverse_chain))
        else:
            critical_chain = ()
        critical_path_queue_ps = sum(
            by_operation_record[operation_id].breakdown.critical_path_queue_ps
            for operation_id in critical_chain
        )
        if critical_chain:
            chain_latency_ps = sum(
                by_operation_record[operation_id].breakdown.operation_latency_ps
                for operation_id in critical_chain
            )
            endpoint_latency_ps = (
                scheduled[critical_chain[-1]].logical_completed_at_ps
                - graph.released_at_ps
            )
            if chain_latency_ps != endpoint_latency_ps:
                raise ValueError(
                    "realized critical-path segments do not conserve graph JCT"
                )
        class_bytes: dict[int, int] = defaultdict(int)
        for operation in graph.operations:
            if isinstance(operation.work, ControlWork):
                fanout = max(1, len(operation.work.destination_ranks))
                class_bytes[operation.priority] += operation.work.payload_bytes * fanout
        return RuntimeReport(
            execution_id=graph.execution_id,
            authority=wqe_authority.authority_name,
            operations=tuple(operation_records),
            visits=tuple(all_visits),
            wqes=tuple(wqes),
            sum_visit_wait_ps=sum(visit.queue_wait_ps for visit in all_visits),
            critical_path_queue_ps=critical_path_queue_ps,
            realized_critical_path_operation_ids=critical_chain,
            class_service_bytes=tuple(sorted(class_bytes.items())),
            random_draw_count=wqe_authority.random_draw_count,
        )

    @staticmethod
    def _attribution_field(
        operation: ExecutionOperation,
        resource_kind: ResourceKind,
    ) -> str:
        if resource_kind in {ResourceKind.HOST_LAUNCH_QUEUE, ResourceKind.CONTROL_QUEUE}:
            return "control_ps"
        if resource_kind is ResourceKind.CUDA_STREAM:
            return "control_ps"
        if resource_kind in {ResourceKind.GPU_WORK_QUEUE, ResourceKind.GPU_SCHEDULER}:
            return "kernel_ps"
        if resource_kind is ResourceKind.HBM_QUEUE:
            if isinstance(operation.work, KvCacheWork):
                return "kv_ps"
            if isinstance(operation.work, DmaWork):
                return "dma_ps"
            if isinstance(operation.work, ComputeWork):
                return "kernel_ps"
            raise ValueError("HBM critical-path visit has no supported semantic owner")
        if resource_kind is ResourceKind.COPY_ENGINE:
            return "dma_ps"
        if resource_kind in {ResourceKind.NCCL_CHANNEL, ResourceKind.NVLINK}:
            return "collective_ps"
        if resource_kind in {
            ResourceKind.NIC_SEND_QUEUE,
            ResourceKind.NIC_RECEIVE_QUEUE,
            ResourceKind.NIC,
            ResourceKind.COMPLETION_QUEUE,
        }:
            return "nic_ps"
        raise ValueError(
            f"resource kind {resource_kind.value!r} has no attribution owner"
        )

    def _critical_path_attribution(
        self,
        operation: ExecutionOperation,
        path: Sequence[QueueVisit],
        segment_start_ps: int,
        operation_completed_at_ps: int,
    ) -> LatencyAttribution:
        values = {
            "queue_ps": 0,
            "kv_ps": 0,
            "kernel_ps": 0,
            "dma_ps": 0,
            "collective_ps": 0,
            "nic_ps": 0,
            "control_ps": 0,
        }
        cursor_ps = segment_start_ps
        for visit in path:
            values["queue_ps"] += max(0, visit.eligible_at_ps - cursor_ps)
            values["queue_ps"] += max(
                0,
                visit.started_at_ps - max(visit.eligible_at_ps, cursor_ps),
            )
            field = self._attribution_field(operation, visit.resource.kind)
            values[field] += max(
                0,
                visit.finished_at_ps - max(visit.started_at_ps, cursor_ps),
            )
            values[field] += max(
                0,
                visit.completed_at_ps - max(visit.finished_at_ps, cursor_ps),
            )
            cursor_ps = max(cursor_ps, visit.completed_at_ps)
        values["queue_ps"] += max(0, operation_completed_at_ps - cursor_ps)
        return LatencyAttribution(**values)

    def _validate_bookkeeping_append(
        self,
        bookkeeper: RequestBookkeeper,
        graph: ExecutionGraph,
        scheduled: Mapping[str, _ScheduledOperation],
        wqes: Sequence[WqeLifecycleProjection],
        events: tuple[CompletionEvent, ...],
    ) -> None:
        staged = RequestBookkeeper(bookkeeper.snapshot())
        self._append_bookkeeping(staged, graph, scheduled, wqes, events)

    def _append_bookkeeping(
        self,
        bookkeeper: RequestBookkeeper,
        graph: ExecutionGraph,
        scheduled: Mapping[str, _ScheduledOperation],
        wqes: Sequence[WqeLifecycleProjection],
        events: tuple[CompletionEvent, ...],
    ) -> None:
        bookkeeper.register_graph(graph)
        object_facts = self._bookkeeping_objects(bookkeeper, graph, scheduled, wqes)
        if object_facts:
            bookkeeper.extend(object_facts)
        if events:
            bookkeeper.extend(events)
        required_ids = graph.completion_operation_ids or tuple(scheduled)
        completed_at_ps = max(
            (scheduled[operation_id].logical_completed_at_ps for operation_id in required_ids),
            default=graph.released_at_ps,
        )
        bookkeeper.append(
            StageRecord(
                stage=ProcessingStage.COMPLETION,
                phase=StagePhase.COMPLETED,
                timestamp_ps=completed_at_ps,
                scope=BookkeepingScope(
                    step_index=graph.step_index,
                    execution_id=graph.execution_id,
                ),
            )
        )

    def _bookkeeping_objects(
        self,
        bookkeeper: RequestBookkeeper,
        graph: ExecutionGraph,
        scheduled: Mapping[str, _ScheduledOperation],
        wqes: Sequence[WqeLifecycleProjection],
    ) -> tuple[CreatedObjectRecord, ...]:
        existing = {
            entry.fact.ref.object_id: entry.fact
            for entry in bookkeeper.snapshot().entries
            if isinstance(entry.fact, CreatedObjectRecord)
        }
        additions: list[CreatedObjectRecord] = []

        def add(record: CreatedObjectRecord) -> None:
            prior = existing.get(record.ref.object_id)
            if prior is not None:
                if prior.ref != record.ref or prior.native_id != record.native_id:
                    raise ValueError(
                        f"runtime object {record.ref.object_id!r} conflicts with bookkeeping"
                    )
                return
            existing[record.ref.object_id] = record
            additions.append(record)

        operation_by_id = {operation.operation_id: operation for operation in graph.operations}
        for operation in graph.operations:
            outcome = scheduled[operation.operation_id]
            if outcome.nccl_command_id is None:
                continue
            scope = BookkeepingScope(
                correlation=operation.correlation,
                step_index=graph.step_index,
                execution_id=graph.execution_id,
                operation_id=operation.operation_id,
            )
            add(
                CreatedObjectRecord(
                    ref=CreatedObjectRef(
                        CreatedObjectKind.NCCL_COMMAND,
                        outcome.nccl_command_id,
                    ),
                    owner=ObjectOwner.NCCL,
                    created_at_ps=graph.released_at_ps,
                    scope=scope,
                    native_id=operation.operation_id,
                    parent_refs=(
                        _operation_object_ref(graph.execution_id, operation.operation_id),
                    ),
                    metadata=(
                        ("channel", operation.work.channel_hint or "default"),
                        ("collective", operation.work.collective),
                        ("algorithm", operation.work.algorithm_hint or "unspecified"),
                    ),
                )
            )

        for wqe in wqes:
            operation = operation_by_id[wqe.operation_id]
            scope = BookkeepingScope(
                correlation=operation.correlation,
                step_index=graph.step_index,
                execution_id=graph.execution_id,
                operation_id=operation.operation_id,
            )
            sq_ref = CreatedObjectRef(CreatedObjectKind.SEND_QUEUE, wqe.sq_id)
            rq_ref = CreatedObjectRef(CreatedObjectKind.RECEIVE_QUEUE, wqe.rq_id)
            cq_ref = CreatedObjectRef(CreatedObjectKind.COMPLETION_QUEUE, wqe.cq_id)
            qp_ref = CreatedObjectRef(CreatedObjectKind.DCQCN_QP, wqe.qp_id)
            for ref, native_id, queue_kind in (
                (sq_ref, wqe.sq_id, "sq"),
                (rq_ref, wqe.rq_id, "rq"),
                (cq_ref, wqe.cq_id, "cq"),
            ):
                add(
                    CreatedObjectRecord(
                        ref=ref,
                        owner=ObjectOwner.NETWORK_BACKEND,
                        created_at_ps=graph.released_at_ps,
                        native_id=native_id,
                        metadata=(
                            ("authority", wqe.authority),
                            ("queue_kind", queue_kind),
                        ),
                    )
                )
            add(
                CreatedObjectRecord(
                    ref=qp_ref,
                    owner=ObjectOwner.NETWORK_BACKEND,
                    created_at_ps=graph.released_at_ps,
                    native_id=wqe.qp_id,
                    parent_refs=(sq_ref, rq_ref, cq_ref),
                    metadata=(
                        ("authority", wqe.authority),
                        ("compatibility_qp", True),
                    ),
                )
            )
            parents: list[CreatedObjectRef] = []
            if wqe.nccl_command_id is not None:
                parents.append(
                    CreatedObjectRef(CreatedObjectKind.NCCL_COMMAND, wqe.nccl_command_id)
                )
            parents.extend((sq_ref, rq_ref, cq_ref, qp_ref))
            add(
                CreatedObjectRecord(
                    ref=CreatedObjectRef(CreatedObjectKind.NETWORK_WQE, wqe.wqe_id),
                    owner=ObjectOwner.NETWORK_BACKEND,
                    created_at_ps=wqe.submitted_at_ps,
                    scope=scope,
                    native_id=wqe.native_wqe_id,
                    parent_refs=tuple(parents),
                    metadata=(
                        ("authority", wqe.authority),
                        ("bytes", wqe.payload_bytes),
                        ("channel", wqe.channel_id),
                        ("cq_post_sequence", wqe.cq_post_sequence),
                        ("destination_rank", wqe.destination_rank),
                        ("extent_index", wqe.extent_index),
                        ("goal_tag", wqe.goal_tag),
                        ("graph_operation_id", wqe.operation_id),
                        ("source_rank", wqe.source_rank),
                        ("sq_post_sequence", wqe.sq_post_sequence),
                        ("transport_kind", "compatibility-qp"),
                    ),
                )
            )
        return tuple(additions)


__all__ = [
    "DEFAULT_GOAL_BASE_TAG",
    "DEFAULT_GPUS_PER_NODE",
    "DEFAULT_NVLINK_RATE_BPS",
    "DEFAULT_RNICS_PER_NODE",
    "DEFAULT_RNIC_RATE_BPS",
    "ArbitrationCandidate",
    "ArbitrationPolicy",
    "AtlahsWqeLedger",
    "CoarseDeviceProfile",
    "CoarseDeviceRuntime",
    "CriticalPathBreakdown",
    "IdentityArbitrationPolicy",
    "NativeRnicSession",
    "QueueVisit",
    "RnicAuthorityMode",
    "RuntimeOperationRecord",
    "RuntimeReport",
    "SemanticWqeSubmission",
    "WqeLifecycleProjection",
    "collective_goal_tags",
]
