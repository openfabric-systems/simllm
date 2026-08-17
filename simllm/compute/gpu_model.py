"""Trace-calibrated, isolated GPU kernel and copy-engine service models."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum

from simllm.compute.provider import (
    PS_PER_SECOND,
    ComputeProvider,
    DurationEstimate,
    GpuSpec,
    KernelSpec,
    ProfileKey,
)

GPU_MODEL_IMPLEMENTATION = "simllm-gpu-service-v2"


class PipelineKind(str, Enum):
    ALU = "alu"
    INT = "int"
    FP64 = "fp64"
    TENSOR = "tensor"
    LOAD_STORE = "load_store"
    SPECIAL_FUNCTION = "special_function"
    CONTROL = "control"


class MemorySpace(str, Enum):
    HBM = "hbm"
    L2 = "l2"
    L1 = "l1"
    SHARED = "shared"
    NVLINK = "nvlink"


_NVLINK_EGRESS_OPCODES = frozenset({"ST", "STG"})


class CopyDirection(str, Enum):
    HOST_TO_DEVICE = "host_to_device"
    DEVICE_TO_HOST = "device_to_host"
    DEVICE_TO_DEVICE = "device_to_device"
    PEER_TO_PEER = "peer_to_peer"


class WarpSchedulerPolicy(str, Enum):
    LOOSE_ROUND_ROBIN = "loose_round_robin"
    GREEDY_THEN_OLDEST = "greedy_then_oldest"


@dataclass(frozen=True, kw_only=True)
class PipelineProfile:
    kind: PipelineKind
    opcodes: tuple[str, ...]
    latency_cycles: int
    issue_width_per_sm: int
    initiation_interval_cycles: int = 1
    opcode_latencies: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "opcodes", tuple(self.opcodes))
        object.__setattr__(self, "opcode_latencies", tuple(self.opcode_latencies))
        _require_enum("kind", self.kind, PipelineKind)
        _require_nonempty("opcodes", self.opcodes)
        for opcode in self.opcodes:
            _require_text("pipeline opcode", opcode)
        if len(set(self.opcodes)) != len(self.opcodes):
            raise ValueError(f"pipeline {self.kind.value} contains duplicate opcodes")
        latency_opcodes = [opcode for opcode, _ in self.opcode_latencies]
        if len(set(latency_opcodes)) != len(latency_opcodes):
            raise ValueError(f"pipeline {self.kind.value} contains duplicate opcode latencies")
        if any(opcode not in self.opcodes for opcode in latency_opcodes):
            raise ValueError("opcode latency overrides must name an opcode in the pipeline")
        for opcode, latency in self.opcode_latencies:
            _require_text("opcode latency key", opcode)
            _require_positive_int("opcode latency", latency)
        _require_positive_int("latency_cycles", self.latency_cycles)
        _require_positive_int("issue_width_per_sm", self.issue_width_per_sm)
        _require_positive_int("initiation_interval_cycles", self.initiation_interval_cycles)

    def latency_for(self, opcode: str) -> int:
        """Return an exact opcode override or the pipeline default."""

        for candidate, latency in self.opcode_latencies:
            if candidate == opcode:
                return latency
        return self.latency_cycles


@dataclass(frozen=True, kw_only=True)
class MemoryHierarchyProfile:
    hbm_latency_cycles: int
    hbm_bandwidth_bytes_per_cycle: float
    l2_latency_cycles: int = 0
    l1_latency_cycles: int = 0
    shared_latency_cycles: int = 0

    def __post_init__(self) -> None:
        _require_nonnegative_int("hbm_latency_cycles", self.hbm_latency_cycles)
        _require_positive_number(
            "hbm_bandwidth_bytes_per_cycle", self.hbm_bandwidth_bytes_per_cycle
        )
        for name in ("l2_latency_cycles", "l1_latency_cycles", "shared_latency_cycles"):
            _require_nonnegative_int(name, getattr(self, name))


@dataclass(frozen=True, kw_only=True)
class NvlinkProfile:
    """Flat same-generation NVLink egress service seen by one GPU.

    The first cut is a single per-GPU egress serializer: every NVLINK store
    from any SM shares one bandwidth cursor, mirroring the HBM cursor. The
    intra-node path stays inside this model rather than reaching the fabric
    backend, which is the split TRAF-10 owns. Peer topology, per-link
    routing and ingress service are deferred under COMP-31, which retains
    those clauses from the closed COMP-11.
    """

    latency_cycles: int
    bandwidth_bytes_per_cycle: float

    def __post_init__(self) -> None:
        _require_nonnegative_int("latency_cycles", self.latency_cycles)
        _require_positive_number(
            "bandwidth_bytes_per_cycle", self.bandwidth_bytes_per_cycle
        )


@dataclass(frozen=True, kw_only=True)
class CopyDirectionProfile:
    direction: CopyDirection
    setup_cycles: int
    bandwidth_bytes_per_cycle: float

    def __post_init__(self) -> None:
        _require_enum("direction", self.direction, CopyDirection)
        _require_nonnegative_int("setup_cycles", self.setup_cycles)
        _require_positive_number("bandwidth_bytes_per_cycle", self.bandwidth_bytes_per_cycle)


@dataclass(frozen=True, kw_only=True)
class CopyEngineProfile:
    engine_id: str
    clock_hz: int
    direction_profiles: tuple[CopyDirectionProfile, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction_profiles", tuple(self.direction_profiles))
        _require_text("engine_id", self.engine_id)
        _require_positive_int("clock_hz", self.clock_hz)
        _require_nonempty("direction_profiles", self.direction_profiles)
        if any(
            not isinstance(profile, CopyDirectionProfile) for profile in self.direction_profiles
        ):
            raise TypeError("direction_profiles must contain CopyDirectionProfile records")
        _require_unique(
            "copy directions", tuple(profile.direction for profile in self.direction_profiles)
        )

    @property
    def directions(self) -> tuple[CopyDirection, ...]:
        return tuple(profile.direction for profile in self.direction_profiles)

    def service(self, direction: CopyDirection) -> CopyDirectionProfile:
        for profile in self.direction_profiles:
            if profile.direction is direction:
                return profile
        raise KeyError(
            f"copy engine {self.engine_id!r} does not support direction {direction.value!r}"
        )


@dataclass(frozen=True, kw_only=True)
class GpuModelProvenance:
    source: str
    version: str
    gpu: str
    created: str
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "references", tuple(self.references))
        for name in ("source", "version", "gpu", "created"):
            _require_text(name, getattr(self, name))
        for reference in self.references:
            _require_text("provenance reference", reference)
        _require_unique("provenance references", self.references)


@dataclass(frozen=True, kw_only=True)
class GpuCalibrationProfile:
    calibration_id: str
    target_architecture_profile_id: str
    provenance: GpuModelProvenance
    core_clock_hz: int
    target_memory_clock_hz: int | None
    pipelines: tuple[PipelineProfile, ...]
    memory: MemoryHierarchyProfile
    copy_engines: tuple[CopyEngineProfile, ...]
    warp_scheduler_policy: WarpSchedulerPolicy
    relative_uncertainty: float
    nvlink: NvlinkProfile | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pipelines", tuple(self.pipelines))
        object.__setattr__(self, "copy_engines", tuple(self.copy_engines))
        _require_text("calibration_id", self.calibration_id)
        _require_text("target_architecture_profile_id", self.target_architecture_profile_id)
        _require_positive_int("core_clock_hz", self.core_clock_hz)
        if self.target_memory_clock_hz is not None:
            _require_positive_int("target_memory_clock_hz", self.target_memory_clock_hz)
        if not isinstance(self.provenance, GpuModelProvenance):
            raise TypeError("provenance must be a GpuModelProvenance")
        _require_nonempty("pipelines", self.pipelines)
        if any(not isinstance(item, PipelineProfile) for item in self.pipelines):
            raise TypeError("pipelines must contain PipelineProfile records")
        if not isinstance(self.memory, MemoryHierarchyProfile):
            raise TypeError("memory must be a MemoryHierarchyProfile")
        if self.nvlink is not None and not isinstance(self.nvlink, NvlinkProfile):
            raise TypeError("nvlink must be an NvlinkProfile or None")
        if any(not isinstance(item, CopyEngineProfile) for item in self.copy_engines):
            raise TypeError("copy_engines must contain CopyEngineProfile records")
        _require_enum("warp_scheduler_policy", self.warp_scheduler_policy, WarpSchedulerPolicy)
        kinds = [pipeline.kind for pipeline in self.pipelines]
        if len(set(kinds)) != len(kinds):
            raise ValueError("calibration contains duplicate pipeline kinds")
        opcodes = [opcode for pipeline in self.pipelines for opcode in pipeline.opcodes]
        if len(set(opcodes)) != len(opcodes):
            raise ValueError("each normalized opcode must belong to exactly one pipeline")
        engine_ids = [engine.engine_id for engine in self.copy_engines]
        if len(set(engine_ids)) != len(engine_ids):
            raise ValueError("calibration contains duplicate copy engine IDs")
        if (
            isinstance(self.relative_uncertainty, bool)
            or not isinstance(self.relative_uncertainty, int | float)
            or not math.isfinite(self.relative_uncertainty)
            or self.relative_uncertainty < 0.0
        ):
            raise ValueError("relative_uncertainty must be finite and non-negative")

    def pipeline(self, kind: PipelineKind) -> PipelineProfile:
        for profile in self.pipelines:
            if profile.kind is kind:
                return profile
        raise KeyError(f"calibration {self.calibration_id!r} has no {kind.value!r} pipeline")

    def copy_engine(self, engine_id: str) -> CopyEngineProfile:
        for profile in self.copy_engines:
            if profile.engine_id == engine_id:
                return profile
        raise KeyError(f"calibration {self.calibration_id!r} has no copy engine {engine_id!r}")


@dataclass(frozen=True, kw_only=True)
class GpuArchitectureProfile:
    profile_id: str
    gpu_name: str
    sm_count: int
    warp_size: int
    scheduler_count_per_sm: int
    max_blocks_per_sm: int
    max_warps_per_sm: int
    max_threads_per_sm: int
    max_threads_per_block: int
    registers_per_sm: int
    max_registers_per_thread: int
    register_allocation_granularity_per_warp: int
    shared_memory_per_sm: int
    max_static_shared_memory_per_block: int
    max_shared_memory_per_block: int
    shared_memory_allocation_granularity: int
    calibration: GpuCalibrationProfile
    dispatch_width_per_scheduler: int = 1
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", tuple(self.aliases))
        _require_text("profile_id", self.profile_id)
        _require_text("gpu_name", self.gpu_name)
        for name in (
            "sm_count",
            "warp_size",
            "scheduler_count_per_sm",
            "dispatch_width_per_scheduler",
            "max_blocks_per_sm",
            "max_warps_per_sm",
            "max_threads_per_sm",
            "max_threads_per_block",
            "registers_per_sm",
            "max_registers_per_thread",
            "register_allocation_granularity_per_warp",
            "shared_memory_per_sm",
            "max_static_shared_memory_per_block",
            "max_shared_memory_per_block",
            "shared_memory_allocation_granularity",
        ):
            _require_positive_int(name, getattr(self, name))
        if self.max_shared_memory_per_block > self.shared_memory_per_sm:
            raise ValueError("max shared memory per block cannot exceed shared memory per SM")
        if self.max_static_shared_memory_per_block > self.max_shared_memory_per_block:
            raise ValueError(
                "max static shared memory per block cannot exceed total shared-memory limit"
            )
        if self.max_threads_per_block > self.max_threads_per_sm:
            raise ValueError("max threads per block cannot exceed max threads per SM")
        if not isinstance(self.calibration, GpuCalibrationProfile):
            raise TypeError("calibration must be a GpuCalibrationProfile")
        if self.calibration.target_architecture_profile_id != self.profile_id:
            raise ValueError(
                f"calibration target architecture profile does not match {self.profile_id!r}"
            )
        names = (self.profile_id, self.gpu_name, *self.aliases)
        for name in names:
            _require_text("profile ID, GPU name, or alias", name)
        _require_unique("profile ID, GPU name, and aliases", names)

    @property
    def all_names(self) -> tuple[str, ...]:
        return (self.profile_id, self.gpu_name, *self.aliases)

    def pipeline(self, kind: PipelineKind) -> PipelineProfile:
        return self.calibration.pipeline(kind)

    def copy_engine(self, engine_id: str) -> CopyEngineProfile:
        return self.calibration.copy_engine(engine_id)

    @property
    def pipelines(self) -> tuple[PipelineProfile, ...]:
        return self.calibration.pipelines

    @property
    def clock_hz(self) -> int:
        return self.calibration.core_clock_hz

    @property
    def memory(self) -> MemoryHierarchyProfile:
        return self.calibration.memory

    @property
    def copy_engines(self) -> tuple[CopyEngineProfile, ...]:
        return self.calibration.copy_engines


@dataclass(frozen=True, kw_only=True)
class SassInstruction:
    opcode: str
    pipeline: PipelineKind
    repeat: int = 1
    dependent: bool = False
    dependency_indices: tuple[int, ...] = ()
    source_registers: tuple[str, ...] = ()
    destination_registers: tuple[str, ...] = ()
    memory_space: MemorySpace | None = None
    requested_bytes: int = 0
    transacted_bytes: int = 0
    barrier: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "dependency_indices", tuple(self.dependency_indices))
        object.__setattr__(self, "source_registers", tuple(self.source_registers))
        object.__setattr__(self, "destination_registers", tuple(self.destination_registers))
        _require_text("opcode", self.opcode)
        _require_enum("pipeline", self.pipeline, PipelineKind)
        _require_positive_int("repeat", self.repeat)
        _require_nonnegative_int("requested_bytes", self.requested_bytes)
        _require_nonnegative_int("transacted_bytes", self.transacted_bytes)
        _require_bool("dependent", self.dependent)
        _require_bool("barrier", self.barrier)
        for index in self.dependency_indices:
            _require_nonnegative_int("dependency index", index)
        _require_unique("dependency indices", self.dependency_indices)
        _require_unique("source registers", self.source_registers)
        _require_unique("destination registers", self.destination_registers)
        for register in (*self.source_registers, *self.destination_registers):
            _require_text("register", register)
        if self.memory_space is None and (self.requested_bytes or self.transacted_bytes):
            raise ValueError("instruction byte counts require a memory_space")
        if self.memory_space is not None:
            _require_enum("memory_space", self.memory_space, MemorySpace)
            if self.requested_bytes == 0 or self.transacted_bytes == 0:
                raise ValueError("a memory instruction must request and transact bytes")
            if self.pipeline is not PipelineKind.LOAD_STORE:
                raise ValueError("memory instructions must use the load_store pipeline")
        elif self.pipeline is PipelineKind.LOAD_STORE:
            raise ValueError("load_store instructions require memory_space and bytes")


@dataclass(frozen=True, kw_only=True)
class SassWarpTrace:
    warp_id: int
    instructions: tuple[SassInstruction, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "instructions", tuple(self.instructions))
        _require_nonnegative_int("warp_id", self.warp_id)
        _require_nonempty("instructions", self.instructions)
        if any(not isinstance(instruction, SassInstruction) for instruction in self.instructions):
            raise TypeError("warp trace instructions must be SassInstruction records")

    def expanded_instructions(self) -> tuple[SassInstruction, ...]:
        return tuple(
            instruction for instruction in self.instructions for _ in range(instruction.repeat)
        )


@dataclass(frozen=True, kw_only=True)
class CtaTrace:
    """One reusable CTA trace class bound to explicit linear block IDs."""

    trace_class_id: str
    block_ids: tuple[int, ...]
    warp_traces: tuple[SassWarpTrace, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_ids", tuple(self.block_ids))
        object.__setattr__(self, "warp_traces", tuple(self.warp_traces))
        _require_text("trace_class_id", self.trace_class_id)
        _require_nonempty("block_ids", self.block_ids)
        for block_id in self.block_ids:
            _require_nonnegative_int("block ID", block_id)
        _require_unique("block IDs", self.block_ids)
        _require_nonempty("warp_traces", self.warp_traces)
        if any(not isinstance(trace, SassWarpTrace) for trace in self.warp_traces):
            raise TypeError("warp_traces must contain SassWarpTrace records")
        _require_unique("warp IDs", tuple(trace.warp_id for trace in self.warp_traces))


@dataclass(frozen=True, kw_only=True)
class KernelLaunch:
    implementation_id: str
    trace_id: str
    grid_blocks: int
    threads_per_block: int
    registers_per_thread: int
    static_shared_memory_bytes: int
    dynamic_shared_memory_bytes: int
    cta_traces: tuple[CtaTrace, ...]
    cooperative: bool = False
    cluster_blocks: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "cta_traces", tuple(self.cta_traces))
        _require_text("implementation_id", self.implementation_id)
        _require_text("trace_id", self.trace_id)
        for name in ("grid_blocks", "threads_per_block", "cluster_blocks"):
            _require_positive_int(name, getattr(self, name))
        for name in (
            "registers_per_thread",
            "static_shared_memory_bytes",
            "dynamic_shared_memory_bytes",
        ):
            _require_nonnegative_int(name, getattr(self, name))
        _require_nonempty("cta_traces", self.cta_traces)
        if any(not isinstance(trace, CtaTrace) for trace in self.cta_traces):
            raise TypeError("cta_traces must contain CtaTrace records")
        _require_bool("cooperative", self.cooperative)
        _require_unique(
            "CTA trace class IDs", tuple(trace.trace_class_id for trace in self.cta_traces)
        )
        block_ids = tuple(block_id for trace in self.cta_traces for block_id in trace.block_ids)
        _require_unique("CTA block IDs", block_ids)
        if set(block_ids) != set(range(self.grid_blocks)):
            raise ValueError("CTA trace classes must cover every linear block ID exactly once")


@dataclass(frozen=True, kw_only=True)
class GpuKernelEstimate:
    model_implementation: str
    architecture_profile_id: str
    calibration_id: str
    implementation_id: str
    trace_id: str
    duration_cycles: int
    duration_ps: int
    resident_blocks_per_sm: int
    cta_waves: int
    issued_instructions: int
    scheduler_stall_cycles: int
    dependency_stall_cycles: int
    pipeline_stall_cycles: int
    completion_drain_cycles: int
    pipeline_issue_counts: tuple[tuple[PipelineKind, int], ...]
    hbm_requested_bytes: int
    hbm_transacted_bytes: int
    hbm_serviced_bytes: int
    hbm_request_instructions: int
    completed_blocks: int
    #: per-SM final instruction-completion cycle (end of the busy span,
    #: including interior idle gaps), not an activity count. The historical
    #: field name remains the stored constructor field for Python API
    #: compatibility; use ``sm_last_completion_cycles`` in new code.
    sm_active_cycles: tuple[int, ...]
    sm_scheduler_pressure_cycles: tuple[int, ...]
    sm_dependency_idle_cycles: tuple[int, ...]
    sm_pipeline_idle_cycles: tuple[int, ...]
    sm_completion_drain_cycles: tuple[int, ...]
    relative_uncertainty: float
    nvlink_requested_bytes: int = 0
    nvlink_transacted_bytes: int = 0
    nvlink_request_instructions: int = 0

    @property
    def sm_last_completion_cycles(self) -> tuple[int, ...]:
        """Return the accurately named view of the legacy SM counter."""

        return self.sm_active_cycles


@dataclass(frozen=True, kw_only=True)
class CopyTransfer:
    transfer_id: str
    direction: CopyDirection
    bytes: int
    source: str
    destination: str

    def __post_init__(self) -> None:
        _require_text("transfer_id", self.transfer_id)
        _require_enum("direction", self.direction, CopyDirection)
        _require_positive_int("bytes", self.bytes)
        _require_text("source", self.source)
        _require_text("destination", self.destination)


@dataclass(frozen=True, kw_only=True)
class CopyServiceEstimate:
    transfer_id: str
    engine_id: str
    direction: CopyDirection
    source: str
    destination: str
    duration_cycles: int
    duration_ps: int
    setup_cycles: int
    transfer_cycles: int
    bytes_transferred: int
    effective_bandwidth_bytes_per_cycle: float
    relative_uncertainty: float


class GpuTaskKind(str, Enum):
    """Coarse workload class of one concurrently scheduled kernel task."""

    COMPUTE = "compute"
    MEMORY = "memory"
    NETWORK = "network"


@dataclass(frozen=True, kw_only=True)
class GpuTask:
    """One kernel launch submitted to the concurrent GPU scheduler.

    ``kind`` is an attribution label for reporting; the replay prices every
    task by its instructions and resources, never by its label.
    """

    task_id: str
    kind: GpuTaskKind
    launch: KernelLaunch
    submitted_cycle: int = 0
    eligible_cycle: int = 0

    def __post_init__(self) -> None:
        _require_text("task_id", self.task_id)
        _require_enum("kind", self.kind, GpuTaskKind)
        if not isinstance(self.launch, KernelLaunch):
            raise TypeError("launch must be a KernelLaunch")
        _require_nonnegative_int("submitted_cycle", self.submitted_cycle)
        _require_nonnegative_int("eligible_cycle", self.eligible_cycle)
        if self.eligible_cycle < self.submitted_cycle:
            raise ValueError("eligible_cycle must not precede submitted_cycle")


@dataclass(frozen=True, kw_only=True)
class GpuTaskEstimate:
    """Per-task attribution inside one concurrent replay."""

    task_id: str
    kind: GpuTaskKind
    implementation_id: str
    trace_id: str
    submitted_cycle: int
    eligible_cycle: int
    #: residency this launch would receive alone; a concurrent replay shares
    #: the SM's currencies, so admitted_cycle is what shows real admission
    isolated_resident_blocks_per_sm: int
    admitted_cycle: int
    completion_cycle: int
    issued_instructions: int
    hbm_requested_bytes: int
    hbm_transacted_bytes: int
    hbm_request_instructions: int
    nvlink_requested_bytes: int
    nvlink_transacted_bytes: int
    nvlink_request_instructions: int

    @property
    def hbm_serviced_bytes(self) -> int:
        """Return serviced HBM bytes, equal to transacted bytes in this model."""

        return self.hbm_transacted_bytes


@dataclass(frozen=True, kw_only=True)
class GpuConcurrentEstimate:
    """Makespan and shared-resource attribution of one concurrent replay."""

    model_implementation: str
    architecture_profile_id: str
    calibration_id: str
    duration_cycles: int
    duration_ps: int
    issued_instructions: int
    scheduler_stall_cycles: int
    dependency_stall_cycles: int
    pipeline_stall_cycles: int
    completion_drain_cycles: int
    hbm_requested_bytes: int
    hbm_transacted_bytes: int
    hbm_request_instructions: int
    nvlink_requested_bytes: int
    nvlink_transacted_bytes: int
    nvlink_request_instructions: int
    tasks: tuple[GpuTaskEstimate, ...]
    relative_uncertainty: float

    @property
    def hbm_serviced_bytes(self) -> int:
        """Return serviced HBM bytes, equal to transacted bytes in this model."""

        return self.hbm_transacted_bytes


class MixedMakespanRegime(str, Enum):
    """Which measured mechanism set a concurrent makespan.

    ``ISSUE_ORDER`` is the case in which every task admitted its first CTAs
    at its own eligibility cycle, so the tasks overlapped and the makespan
    exceeded the longest isolated control only by the submission-order issue
    delay. ``RESIDENCY_SERIALIZED`` is the case in which at least one task
    could not admit at eligibility because the SM's residency currencies were
    already claimed, so the makespan carries that task's whole isolated
    duration after its admission.
    """

    ISSUE_ORDER = "issue-order"
    RESIDENCY_SERIALIZED = "residency-serialized"


@dataclass(frozen=True, kw_only=True)
class MixedMakespanForm:
    """Measured decomposition of one concurrent replay against its controls.

    This record is a read-only projection of a replay that already happened.
    It is not a second estimator: every field is arithmetic over the supplied
    :class:`GpuConcurrentEstimate` and the isolated single-task controls
    measured on the same architecture. ``simllm.compute`` keeps one timing
    authority, :class:`SmSchedulerModel`, and this record only names the terms
    of its result so a study or regression can compare them.

    The two registered forms come from findings G1 and G2 of the task-mix
    study. They are the measured behavior of the exact frozen fixtures there,
    not a general scheduling law for other launch shapes or architectures.
    """

    regime: MixedMakespanRegime
    #: task identifiers in the order they were submitted to the replay
    task_ids: tuple[str, ...]
    #: isolated control duration of each task, same order as ``task_ids``
    isolated_cycles: tuple[int, ...]
    #: eligibility, admission and completion cycles, same order
    eligible_cycles: tuple[int, ...]
    admitted_cycles: tuple[int, ...]
    completion_cycles: tuple[int, ...]
    #: measured concurrent makespan
    mixed_cycles: int
    #: identifier of the task that could not admit at its own eligibility
    residency_gated_task_id: str | None

    @property
    def concurrent_floor_cycles(self) -> int:
        """Longest isolated control: no concurrent replay can finish sooner."""

        return max(self.isolated_cycles)

    @property
    def serialized_ceiling_cycles(self) -> int:
        """Fully serialized controls: the conservative upper bound."""

        return sum(self.isolated_cycles)

    @property
    def issue_delay_cycles(self) -> int:
        """G1 term: makespan above the longest isolated control.

        This subtraction is a definition, not a prediction. What the task-mix
        study measured, and what a replication has to reproduce, is that the
        term follows the actual submitted task order and disappears only when
        both the per-SM scheduler budget and the load/store issue width are
        widened together.
        """

        return self.mixed_cycles - self.concurrent_floor_cycles

    @property
    def residency_delay_cycles(self) -> tuple[int, ...]:
        """Cycles each task waited for SM residency after becoming eligible."""

        return tuple(
            admitted - eligible
            for admitted, eligible in zip(
                self.admitted_cycles,
                self.eligible_cycles,
                strict=True,
            )
        )

    @property
    def residency_serialized_cycles(self) -> int | None:
        """G2 form: gated admission plus that task's whole isolated control.

        ``None`` in the issue-order regime, where no task was gated. Unlike
        :attr:`issue_delay_cycles` this is a real predicate: it asserts that a
        residency-gated task pays its full isolated duration starting at its
        admission cycle, which a makespan value alone would not identify.
        """

        if self.residency_gated_task_id is None:
            return None
        index = self.task_ids.index(self.residency_gated_task_id)
        return self.admitted_cycles[index] + self.isolated_cycles[index]

    @property
    def within_physical_interval(self) -> bool:
        """Whether the makespan sits inside its floor and serialized ceiling."""

        return (
            self.concurrent_floor_cycles
            <= self.mixed_cycles
            <= self.serialized_ceiling_cycles
        )


def decompose_mixed_makespan(
    estimate: GpuConcurrentEstimate,
    isolated_cycles: Mapping[str, int],
) -> MixedMakespanForm:
    """Name the measured terms of one concurrent replay.

    ``isolated_cycles`` maps each task identifier to the duration the same
    launch measured alone on the same architecture, i.e. the single-task
    control. Every task in ``estimate`` must have one, because a decomposition
    against a missing control would silently invent a bound.
    """

    if not isinstance(estimate, GpuConcurrentEstimate):
        raise TypeError("estimate must be a GpuConcurrentEstimate")
    if not isinstance(isolated_cycles, Mapping):
        raise TypeError("isolated_cycles must be a mapping")
    if not estimate.tasks:
        raise ValueError("a mixed makespan needs at least one task estimate")
    missing = tuple(
        task.task_id for task in estimate.tasks if task.task_id not in isolated_cycles
    )
    if missing:
        raise KeyError(f"no isolated control for tasks {missing}")
    for task in estimate.tasks:
        _require_positive_int(
            f"isolated control for {task.task_id!r}", isolated_cycles[task.task_id]
        )

    gated = tuple(
        task for task in estimate.tasks if task.admitted_cycle > task.eligible_cycle
    )
    if len(gated) > 1:
        raise ValueError(
            "more than one task waited for residency; the registered G2 form "
            "covers a single gated task, so this replay needs its own "
            "registration before it is decomposed"
        )
    return MixedMakespanForm(
        regime=(
            MixedMakespanRegime.RESIDENCY_SERIALIZED
            if gated
            else MixedMakespanRegime.ISSUE_ORDER
        ),
        task_ids=tuple(task.task_id for task in estimate.tasks),
        isolated_cycles=tuple(isolated_cycles[task.task_id] for task in estimate.tasks),
        eligible_cycles=tuple(task.eligible_cycle for task in estimate.tasks),
        admitted_cycles=tuple(task.admitted_cycle for task in estimate.tasks),
        completion_cycles=tuple(task.completion_cycle for task in estimate.tasks),
        mixed_cycles=estimate.duration_cycles,
        residency_gated_task_id=gated[0].task_id if gated else None,
    )


# Mutable replay state is deliberately private. Public workload and profile
# records above remain immutable and portable.
@dataclass
class _WarpState:
    warp_id: int
    instructions: tuple[SassInstruction, ...]
    pc: int = 0
    next_issue_cycle: int = 0
    previous_completion: int = 0
    last_completion: int = 0
    register_ready: dict[str, int] = field(default_factory=dict)
    instruction_completions: list[int] = field(default_factory=list)


@dataclass
class _BlockState:
    task_index: int
    block_id: int
    warps: list[_WarpState]


@dataclass
class _SmState:
    sm_id: int
    blocks: list[_BlockState] = field(default_factory=list)
    pipeline_available: dict[PipelineKind, list[int]] = field(default_factory=dict)
    last_issued_warp: tuple[int, int, int] | None = None
    used_warps: int = 0
    used_threads: int = 0
    used_registers: int = 0
    used_shared: int = 0


@dataclass
class _TaskRun:
    launch: KernelLaunch
    expanded_by_block: dict[int, tuple[tuple[int, tuple[SassInstruction, ...]], ...]]
    pending_block_ids: list[int]
    next_pending: int = 0
    warps_per_block: int = 0
    threads_per_block: int = 0
    registers_per_block: int = 0
    shared_per_block: int = 0
    submitted_cycle: int = 0
    eligible_cycle: int = 0
    admitted_cycle: int | None = None
    completion_cycle: int = 0
    issued_instructions: int = 0
    hbm_requested_bytes: int = 0
    hbm_transacted_bytes: int = 0
    hbm_request_instructions: int = 0
    nvlink_requested_bytes: int = 0
    nvlink_transacted_bytes: int = 0
    nvlink_request_instructions: int = 0


@dataclass
class _ReplayOutcome:
    duration_cycles: int
    issued_instructions: int
    scheduler_stall_cycles: int
    dependency_stall_cycles: int
    pipeline_stall_cycles: int
    completion_drain_cycles: int
    pipeline_issue_counts: dict[PipelineKind, int]
    hbm_requested_bytes: int
    hbm_transacted_bytes: int
    hbm_request_instructions: int
    nvlink_requested_bytes: int
    nvlink_transacted_bytes: int
    nvlink_request_instructions: int
    completed_blocks: int
    sm_last_completion: list[int]
    sm_scheduler_pressure: list[int]
    sm_dependency_idle: list[int]
    sm_pipeline_idle: list[int]
    sm_completion_drain: list[int]
    runs: list[_TaskRun]


class SmSchedulerModel:
    """Deterministic event replay of one isolated kernel launch."""

    def __init__(self, architecture: GpuArchitectureProfile):
        if not isinstance(architecture, GpuArchitectureProfile):
            raise TypeError("architecture must be a GpuArchitectureProfile")
        self.architecture = architecture

    def resident_blocks_per_sm(self, launch: KernelLaunch) -> int:
        """Return CTA residency after every allocation granularity is applied."""

        self._validate_launch(launch)
        arch = self.architecture
        warps = _ceil_div(launch.threads_per_block, arch.warp_size)
        registers_per_warp = _round_up(
            launch.registers_per_thread * arch.warp_size,
            arch.register_allocation_granularity_per_warp,
        )
        registers = registers_per_warp * warps
        shared = _round_up(
            launch.static_shared_memory_bytes + launch.dynamic_shared_memory_bytes,
            arch.shared_memory_allocation_granularity,
        )
        limits = [
            arch.max_blocks_per_sm,
            arch.max_warps_per_sm // warps,
            arch.max_threads_per_sm // launch.threads_per_block,
        ]
        if registers:
            limits.append(arch.registers_per_sm // registers)
        if shared:
            limits.append(arch.shared_memory_per_sm // shared)
        resident = min(limits)
        if resident < 1:
            raise ValueError(
                f"launch {launch.implementation_id!r} cannot admit one CTA on "
                f"architecture {arch.profile_id!r}"
            )
        return resident

    def estimate(self, launch: KernelLaunch) -> GpuKernelEstimate:
        """Replay a representative CTA trace across an isolated GPU."""

        resident = self.resident_blocks_per_sm(launch)
        arch = self.architecture
        outcome = self._replay((launch,))
        return GpuKernelEstimate(
            model_implementation=GPU_MODEL_IMPLEMENTATION,
            architecture_profile_id=arch.profile_id,
            calibration_id=arch.calibration.calibration_id,
            implementation_id=launch.implementation_id,
            trace_id=launch.trace_id,
            duration_cycles=outcome.duration_cycles,
            duration_ps=_cycles_to_ps(outcome.duration_cycles, arch.clock_hz),
            resident_blocks_per_sm=resident,
            cta_waves=_ceil_div(launch.grid_blocks, arch.sm_count * resident),
            issued_instructions=outcome.issued_instructions,
            scheduler_stall_cycles=outcome.scheduler_stall_cycles,
            dependency_stall_cycles=outcome.dependency_stall_cycles,
            pipeline_stall_cycles=outcome.pipeline_stall_cycles,
            completion_drain_cycles=outcome.completion_drain_cycles,
            pipeline_issue_counts=tuple(
                (kind, outcome.pipeline_issue_counts[kind])
                for kind in PipelineKind
                if kind in outcome.pipeline_issue_counts
            ),
            hbm_requested_bytes=outcome.hbm_requested_bytes,
            hbm_transacted_bytes=outcome.hbm_transacted_bytes,
            hbm_serviced_bytes=outcome.hbm_transacted_bytes,
            hbm_request_instructions=outcome.hbm_request_instructions,
            nvlink_requested_bytes=outcome.nvlink_requested_bytes,
            nvlink_transacted_bytes=outcome.nvlink_transacted_bytes,
            nvlink_request_instructions=outcome.nvlink_request_instructions,
            completed_blocks=outcome.completed_blocks,
            sm_active_cycles=tuple(outcome.sm_last_completion),
            sm_scheduler_pressure_cycles=tuple(outcome.sm_scheduler_pressure),
            sm_dependency_idle_cycles=tuple(outcome.sm_dependency_idle),
            sm_pipeline_idle_cycles=tuple(outcome.sm_pipeline_idle),
            sm_completion_drain_cycles=tuple(outcome.sm_completion_drain),
            relative_uncertainty=arch.calibration.relative_uncertainty,
        )

    def estimate_concurrent(self, tasks: tuple[GpuTask, ...]) -> GpuConcurrentEstimate:
        """Replay several kernel tasks contending for one GPU.

        Tasks model concurrent stream launches: every task's blocks admit in
        their own linear order, but a later task may backfill SM capacity a
        stalled earlier task cannot use. SM residency limits, scheduler issue
        budgets, pipelines, the HBM cursor and the NVLink cursor are shared.
        """

        tasks = tuple(tasks)
        _require_nonempty("tasks", tasks)
        for task in tasks:
            if not isinstance(task, GpuTask):
                raise TypeError("tasks must contain GpuTask records")
        _require_unique("task IDs", tuple(task.task_id for task in tasks))
        residents = [self.resident_blocks_per_sm(task.launch) for task in tasks]
        arch = self.architecture
        outcome = self._replay(
            tuple(task.launch for task in tasks),
            tuple((task.submitted_cycle, task.eligible_cycle) for task in tasks),
        )
        task_estimates = tuple(
            GpuTaskEstimate(
                task_id=task.task_id,
                kind=task.kind,
                implementation_id=task.launch.implementation_id,
                trace_id=task.launch.trace_id,
                submitted_cycle=run.submitted_cycle,
                eligible_cycle=run.eligible_cycle,
                isolated_resident_blocks_per_sm=residents[index],
                admitted_cycle=run.admitted_cycle if run.admitted_cycle is not None else 0,
                completion_cycle=run.completion_cycle,
                issued_instructions=run.issued_instructions,
                hbm_requested_bytes=run.hbm_requested_bytes,
                hbm_transacted_bytes=run.hbm_transacted_bytes,
                hbm_request_instructions=run.hbm_request_instructions,
                nvlink_requested_bytes=run.nvlink_requested_bytes,
                nvlink_transacted_bytes=run.nvlink_transacted_bytes,
                nvlink_request_instructions=run.nvlink_request_instructions,
            )
            for index, (task, run) in enumerate(zip(tasks, outcome.runs, strict=True))
        )
        return GpuConcurrentEstimate(
            model_implementation=GPU_MODEL_IMPLEMENTATION,
            architecture_profile_id=arch.profile_id,
            calibration_id=arch.calibration.calibration_id,
            duration_cycles=outcome.duration_cycles,
            duration_ps=_cycles_to_ps(outcome.duration_cycles, arch.clock_hz),
            issued_instructions=outcome.issued_instructions,
            scheduler_stall_cycles=outcome.scheduler_stall_cycles,
            dependency_stall_cycles=outcome.dependency_stall_cycles,
            pipeline_stall_cycles=outcome.pipeline_stall_cycles,
            completion_drain_cycles=outcome.completion_drain_cycles,
            hbm_requested_bytes=outcome.hbm_requested_bytes,
            hbm_transacted_bytes=outcome.hbm_transacted_bytes,
            hbm_request_instructions=outcome.hbm_request_instructions,
            nvlink_requested_bytes=outcome.nvlink_requested_bytes,
            nvlink_transacted_bytes=outcome.nvlink_transacted_bytes,
            nvlink_request_instructions=outcome.nvlink_request_instructions,
            tasks=task_estimates,
            relative_uncertainty=arch.calibration.relative_uncertainty,
        )

    def _replay(
        self,
        launches: tuple[KernelLaunch, ...],
        task_times: tuple[tuple[int, int], ...] | None = None,
    ) -> _ReplayOutcome:
        arch = self.architecture
        if task_times is None:
            task_times = tuple((0, 0) for _ in launches)
        if len(task_times) != len(launches):
            raise AssertionError("GPU replay task timing cardinality mismatch")
        runs: list[_TaskRun] = []
        for launch, (submitted_cycle, eligible_cycle) in zip(
            launches, task_times, strict=True
        ):
            self._validate_launch(launch)
            expanded_by_block: dict[
                int, tuple[tuple[int, tuple[SassInstruction, ...]], ...]
            ] = {}
            for cta_trace in launch.cta_traces:
                expanded = tuple(
                    (trace.warp_id, trace.expanded_instructions())
                    for trace in cta_trace.warp_traces
                )
                for block_id in cta_trace.block_ids:
                    expanded_by_block[block_id] = expanded
            warps = _ceil_div(launch.threads_per_block, arch.warp_size)
            registers_per_warp = _round_up(
                launch.registers_per_thread * arch.warp_size,
                arch.register_allocation_granularity_per_warp,
            )
            shared = _round_up(
                launch.static_shared_memory_bytes + launch.dynamic_shared_memory_bytes,
                arch.shared_memory_allocation_granularity,
            )
            runs.append(
                _TaskRun(
                    launch=launch,
                    expanded_by_block=expanded_by_block,
                    pending_block_ids=list(range(launch.grid_blocks)),
                    warps_per_block=warps,
                    threads_per_block=launch.threads_per_block,
                    registers_per_block=registers_per_warp * warps,
                    shared_per_block=shared,
                    submitted_cycle=submitted_cycle,
                    eligible_cycle=eligible_cycle,
                )
            )
        total_blocks = sum(run.launch.grid_blocks for run in runs)
        profiles = {profile.kind: profile for profile in arch.pipelines}
        sms = [
            _SmState(
                sm_id=sm_id,
                pipeline_available={
                    kind: [0] * profile.issue_width_per_sm for kind, profile in profiles.items()
                },
            )
            for sm_id in range(arch.sm_count)
        ]
        completed_blocks = 0
        current = 0
        hbm_available = current
        nvlink_available = current
        issued = 0
        scheduler_stalls = 0
        dependency_stalls = 0
        pipeline_stalls = 0
        completion_drain = 0
        hbm_requested_bytes = 0
        hbm_transacted_bytes = 0
        hbm_request_instructions = 0
        nvlink_requested_bytes = 0
        nvlink_transacted_bytes = 0
        nvlink_request_instructions = 0
        pipeline_issues = {kind: 0 for kind in profiles}
        sm_last_completion = [0] * arch.sm_count
        sm_scheduler_pressure = [0] * arch.sm_count
        sm_dependency_idle = [0] * arch.sm_count
        sm_pipeline_idle = [0] * arch.sm_count
        sm_completion_drain = [0] * arch.sm_count

        def fits(sm: _SmState, run: _TaskRun) -> bool:
            return (
                len(sm.blocks) < arch.max_blocks_per_sm
                and sm.used_warps + run.warps_per_block <= arch.max_warps_per_sm
                and sm.used_threads + run.threads_per_block <= arch.max_threads_per_sm
                and sm.used_registers + run.registers_per_block <= arch.registers_per_sm
                and sm.used_shared + run.shared_per_block <= arch.shared_memory_per_sm
            )

        def admit() -> None:
            made_progress = True
            while made_progress:
                made_progress = False
                for sm in sms:
                    for task_index, run in enumerate(runs):
                        if run.next_pending >= len(run.pending_block_ids):
                            continue
                        if run.eligible_cycle > current:
                            continue
                        if not fits(sm, run):
                            continue
                        block_id = run.pending_block_ids[run.next_pending]
                        warps = [
                            _WarpState(warp_id=warp_id, instructions=instructions)
                            for warp_id, instructions in run.expanded_by_block[block_id]
                        ]
                        sm.blocks.append(
                            _BlockState(task_index=task_index, block_id=block_id, warps=warps)
                        )
                        sm.used_warps += run.warps_per_block
                        sm.used_threads += run.threads_per_block
                        sm.used_registers += run.registers_per_block
                        sm.used_shared += run.shared_per_block
                        run.next_pending += 1
                        if run.admitted_cycle is None:
                            run.admitted_cycle = current
                        made_progress = True
                        break

        admit()
        while completed_blocks < total_blocks:
            for sm in sms:
                retained: list[_BlockState] = []
                for block in sm.blocks:
                    if self._block_finished(block, current):
                        completed_blocks += 1
                        run = runs[block.task_index]
                        run.completion_cycle = max(
                            run.completion_cycle,
                            max(warp.last_completion for warp in block.warps),
                        )
                        sm.used_warps -= run.warps_per_block
                        sm.used_threads -= run.threads_per_block
                        sm.used_registers -= run.registers_per_block
                        sm.used_shared -= run.shared_per_block
                    else:
                        retained.append(block)
                sm.blocks = retained
            admit()
            if completed_blocks == total_blocks:
                break

            issued_this_cycle = 0
            ready_without_pipeline = False
            has_unissued_work = False
            scheduler_limited = False
            sm_idle_reasons: list[str | None] = [None] * arch.sm_count
            for sm in sms:
                budget = arch.scheduler_count_per_sm * arch.dispatch_width_per_scheduler
                candidates = sorted(
                    (
                        (block.task_index, block.block_id, warp.warp_id, warp)
                        for block in sm.blocks
                        for warp in block.warps
                        if warp.pc < len(warp.instructions)
                        and self._warp_ready_cycle(warp) <= current
                    ),
                    key=lambda item: (item[0], item[1], item[2]),
                )
                candidates = self._order_candidates(sm, candidates)
                ready_without_pipeline |= bool(candidates)
                sm_has_unissued_work = any(
                    warp.pc < len(warp.instructions) for block in sm.blocks for warp in block.warps
                )
                has_unissued_work |= sm_has_unissued_work
                sm_issued = 0
                sm_scheduler_limited = False
                for task_index, block_id, warp_id, warp in candidates:
                    if sm_issued >= budget:
                        sm_scheduler_limited = True
                        scheduler_limited = True
                        break
                    instruction = warp.instructions[warp.pc]
                    profile = profiles[instruction.pipeline]
                    slots = sm.pipeline_available[instruction.pipeline]
                    lane = min(range(len(slots)), key=lambda index: (slots[index], index))
                    if slots[lane] > current:
                        continue
                    completion = current + profile.latency_for(instruction.opcode)
                    if instruction.memory_space is not None:
                        memory_completion, hbm_available, nvlink_available = (
                            self._memory_completion(
                                instruction, current, hbm_available, nvlink_available
                            )
                        )
                        completion = max(completion, memory_completion)
                        if instruction.memory_space is MemorySpace.HBM:
                            hbm_requested_bytes += instruction.requested_bytes
                            hbm_transacted_bytes += instruction.transacted_bytes
                            hbm_request_instructions += 1
                            runs[task_index].hbm_requested_bytes += instruction.requested_bytes
                            runs[task_index].hbm_transacted_bytes += instruction.transacted_bytes
                            runs[task_index].hbm_request_instructions += 1
                        elif instruction.memory_space is MemorySpace.NVLINK:
                            nvlink_requested_bytes += instruction.requested_bytes
                            nvlink_transacted_bytes += instruction.transacted_bytes
                            nvlink_request_instructions += 1
                            runs[task_index].nvlink_requested_bytes += instruction.requested_bytes
                            runs[task_index].nvlink_transacted_bytes += (
                                instruction.transacted_bytes
                            )
                            runs[task_index].nvlink_request_instructions += 1
                    slots[lane] = current + profile.initiation_interval_cycles
                    warp.pc += 1
                    warp.next_issue_cycle = current + 1
                    warp.previous_completion = completion
                    warp.last_completion = max(warp.last_completion, completion)
                    warp.instruction_completions.append(completion)
                    for register in instruction.destination_registers:
                        warp.register_ready[register] = completion
                    issued += 1
                    pipeline_issues[instruction.pipeline] += 1
                    runs[task_index].issued_instructions += 1
                    sm_issued += 1
                    issued_this_cycle += 1
                    sm.last_issued_warp = (task_index, block_id, warp_id)
                    sm_last_completion[sm.sm_id] = max(sm_last_completion[sm.sm_id], completion)
                if sm_scheduler_limited:
                    sm_scheduler_pressure[sm.sm_id] += 1
                if sm_issued == 0 and sm.blocks:
                    if candidates:
                        sm_idle_reasons[sm.sm_id] = "pipeline"
                    elif sm_has_unissued_work:
                        sm_idle_reasons[sm.sm_id] = "dependency"
                    else:
                        sm_idle_reasons[sm.sm_id] = "completion"

            if scheduler_limited:
                scheduler_stalls += 1
            if issued_this_cycle:
                for sm_id, reason in enumerate(sm_idle_reasons):
                    if reason == "pipeline":
                        sm_pipeline_idle[sm_id] += 1
                    elif reason == "dependency":
                        sm_dependency_idle[sm_id] += 1
                    elif reason == "completion":
                        sm_completion_drain[sm_id] += 1
                current += 1
                continue

            if not any(sm.blocks for sm in sms):
                future_eligibility = [
                    run.eligible_cycle
                    for run in runs
                    if run.next_pending < len(run.pending_block_ids)
                    and run.eligible_cycle > current
                ]
                if not future_eligibility:
                    raise RuntimeError(
                        "GPU replay has pending CTAs without a future eligibility"
                    )
                current = min(future_eligibility)
                admit()
                continue

            next_cycle = self._next_event_cycle(sms, runs, current)
            if next_cycle is None:
                raise RuntimeError("GPU replay reached a dead state with unfinished CTAs")
            delta = next_cycle - current
            if delta <= 0:
                raise RuntimeError("GPU replay failed to advance virtual time")
            if ready_without_pipeline:
                pipeline_stalls += delta
            elif has_unissued_work:
                dependency_stalls += delta
            else:
                completion_drain += delta
            for sm_id, reason in enumerate(sm_idle_reasons):
                if reason == "pipeline":
                    sm_pipeline_idle[sm_id] += delta
                elif reason == "dependency":
                    sm_dependency_idle[sm_id] += delta
                elif reason == "completion":
                    sm_completion_drain[sm_id] += delta
            current = next_cycle

        return _ReplayOutcome(
            duration_cycles=current,
            issued_instructions=issued,
            scheduler_stall_cycles=scheduler_stalls,
            dependency_stall_cycles=dependency_stalls,
            pipeline_stall_cycles=pipeline_stalls,
            completion_drain_cycles=completion_drain,
            pipeline_issue_counts=pipeline_issues,
            hbm_requested_bytes=hbm_requested_bytes,
            hbm_transacted_bytes=hbm_transacted_bytes,
            hbm_request_instructions=hbm_request_instructions,
            nvlink_requested_bytes=nvlink_requested_bytes,
            nvlink_transacted_bytes=nvlink_transacted_bytes,
            nvlink_request_instructions=nvlink_request_instructions,
            completed_blocks=completed_blocks,
            sm_last_completion=sm_last_completion,
            sm_scheduler_pressure=sm_scheduler_pressure,
            sm_dependency_idle=sm_dependency_idle,
            sm_pipeline_idle=sm_pipeline_idle,
            sm_completion_drain=sm_completion_drain,
            runs=runs,
        )

    def _validate_launch(self, launch: KernelLaunch) -> None:
        if not isinstance(launch, KernelLaunch):
            raise TypeError("launch must be a KernelLaunch")
        arch = self.architecture
        if launch.cooperative:
            raise ValueError("cooperative launches are not supported by isolated replay")
        if launch.cluster_blocks != 1:
            raise ValueError("thread-block clusters are not supported by isolated replay")
        if launch.threads_per_block > arch.max_threads_per_block:
            raise ValueError(
                f"launch requests {launch.threads_per_block} threads per CTA, above "
                f"the profile limit {arch.max_threads_per_block}"
            )
        if launch.registers_per_thread > arch.max_registers_per_thread:
            raise ValueError(
                f"launch requests {launch.registers_per_thread} registers per thread, above "
                f"the profile limit {arch.max_registers_per_thread}"
            )
        shared_bytes = launch.static_shared_memory_bytes + launch.dynamic_shared_memory_bytes
        if launch.static_shared_memory_bytes > arch.max_static_shared_memory_per_block:
            raise ValueError(
                f"launch requests {launch.static_shared_memory_bytes} static shared-memory "
                f"bytes per CTA, above the profile limit "
                f"{arch.max_static_shared_memory_per_block}"
            )
        if shared_bytes > arch.max_shared_memory_per_block:
            raise ValueError(
                f"launch requests {shared_bytes} shared-memory bytes per CTA, above "
                f"the profile limit {arch.max_shared_memory_per_block}"
            )
        expected_warps = _ceil_div(launch.threads_per_block, arch.warp_size)
        opcode_owner = {
            opcode: profile.kind for profile in arch.pipelines for opcode in profile.opcodes
        }
        for cta_trace in launch.cta_traces:
            observed_warps = {trace.warp_id for trace in cta_trace.warp_traces}
            if observed_warps != set(range(expected_warps)):
                raise ValueError(
                    f"CTA trace class {cta_trace.trace_class_id!r} must contain one trace "
                    f"for each warp ID in range(0, {expected_warps})"
                )
            for trace in cta_trace.warp_traces:
                flattened_index = 0
                for instruction in trace.instructions:
                    if instruction.barrier:
                        raise ValueError(
                            "barrier instructions are not supported by isolated replay"
                        )
                    if (
                        instruction.memory_space is MemorySpace.NVLINK
                        and arch.calibration.nvlink is None
                    ):
                        raise ValueError(
                            "NVLINK instructions require an nvlink profile in the "
                            "calibration; this calibration has none"
                        )
                    if (
                        instruction.memory_space is MemorySpace.NVLINK
                        and instruction.opcode not in _NVLINK_EGRESS_OPCODES
                    ):
                        raise ValueError(
                            "NVLINK memory space only supports normalized egress "
                            "store opcodes ST and STG"
                        )
                    owner = opcode_owner.get(instruction.opcode)
                    if owner is None:
                        raise ValueError(f"unknown normalized opcode {instruction.opcode!r}")
                    if owner is not instruction.pipeline:
                        raise ValueError(
                            f"opcode {instruction.opcode!r} belongs to {owner.value}, not "
                            f"{instruction.pipeline.value}"
                        )
                    for _ in range(instruction.repeat):
                        if any(
                            index >= flattened_index for index in instruction.dependency_indices
                        ):
                            raise ValueError(
                                "dependency indices must name earlier dynamic instructions"
                            )
                        flattened_index += 1

    @staticmethod
    def _warp_ready_cycle(warp: _WarpState) -> int:
        instruction = warp.instructions[warp.pc]
        ready = warp.next_issue_cycle
        if instruction.dependent:
            ready = max(ready, warp.previous_completion)
        for register in (*instruction.source_registers, *instruction.destination_registers):
            ready = max(ready, warp.register_ready.get(register, 0))
        for index in instruction.dependency_indices:
            ready = max(ready, warp.instruction_completions[index])
        return ready

    @staticmethod
    def _block_finished(block: _BlockState, cycle: int) -> bool:
        return all(
            warp.pc == len(warp.instructions) and warp.last_completion <= cycle
            for warp in block.warps
        )

    def _order_candidates(
        self,
        sm: _SmState,
        candidates: list[tuple[int, int, int, _WarpState]],
    ) -> list[tuple[int, int, int, _WarpState]]:
        if not candidates or sm.last_issued_warp is None:
            return candidates
        policy = self.architecture.calibration.warp_scheduler_policy
        if policy is WarpSchedulerPolicy.GREEDY_THEN_OLDEST:
            return sorted(
                candidates,
                key=lambda item: (
                    (item[0], item[1], item[2]) != sm.last_issued_warp,
                    item[0],
                    item[1],
                    item[2],
                ),
            )
        for index, (task_index, block_id, warp_id, _) in enumerate(candidates):
            if (task_index, block_id, warp_id) > sm.last_issued_warp:
                return candidates[index:] + candidates[:index]
        return candidates

    def _memory_completion(
        self,
        instruction: SassInstruction,
        issue_cycle: int,
        hbm_available: int,
        nvlink_available: int,
    ) -> tuple[int, int, int]:
        memory = self.architecture.memory
        if instruction.memory_space is MemorySpace.HBM:
            start = max(issue_cycle, hbm_available)
            service = _checked_service_cycles(
                transacted_bytes=instruction.transacted_bytes,
                bandwidth_bytes_per_cycle=memory.hbm_bandwidth_bytes_per_cycle,
                resource="HBM",
            )
            service_end = start + service
            completion = service_end + memory.hbm_latency_cycles
            return completion, service_end, nvlink_available
        if instruction.memory_space is MemorySpace.NVLINK:
            nvlink = self.architecture.calibration.nvlink
            start = max(issue_cycle, nvlink_available)
            service = _checked_service_cycles(
                transacted_bytes=instruction.transacted_bytes,
                bandwidth_bytes_per_cycle=nvlink.bandwidth_bytes_per_cycle,
                resource="NVLink",
            )
            service_end = start + service
            completion = service_end + nvlink.latency_cycles
            return completion, hbm_available, service_end
        latency = {
            MemorySpace.L2: memory.l2_latency_cycles,
            MemorySpace.L1: memory.l1_latency_cycles,
            MemorySpace.SHARED: memory.shared_latency_cycles,
        }[instruction.memory_space]
        return issue_cycle + latency, hbm_available, nvlink_available

    def _next_event_cycle(
        self,
        sms: list[_SmState],
        runs: list[_TaskRun],
        current: int,
    ) -> int | None:
        events = [
            run.eligible_cycle
            for run in runs
            if run.next_pending < len(run.pending_block_ids)
            and run.eligible_cycle > current
        ]
        for sm in sms:
            for block in sm.blocks:
                if all(warp.pc == len(warp.instructions) for warp in block.warps):
                    events.extend(
                        warp.last_completion
                        for warp in block.warps
                        if warp.last_completion > current
                    )
                for warp in block.warps:
                    if warp.pc >= len(warp.instructions):
                        continue
                    instruction = warp.instructions[warp.pc]
                    ready = self._warp_ready_cycle(warp)
                    slots = sm.pipeline_available[instruction.pipeline]
                    events.append(max(ready, min(slots)))
        future = [event for event in events if event > current]
        return min(future) if future else None


class CopyEngineServiceModel:
    """Pure service estimate for one explicitly selected copy engine."""

    def __init__(self, architecture: GpuArchitectureProfile, engine_id: str):
        if not isinstance(architecture, GpuArchitectureProfile):
            raise TypeError("architecture must be a GpuArchitectureProfile")
        self.architecture = architecture
        self.engine_id = engine_id
        self.engine = architecture.copy_engine(engine_id)

    def estimate(self, transfer: CopyTransfer) -> CopyServiceEstimate:
        if not isinstance(transfer, CopyTransfer):
            raise TypeError("transfer must be a CopyTransfer")
        if transfer.direction not in self.engine.directions:
            raise ValueError(
                f"copy engine {self.engine_id!r} does not support {transfer.direction.value!r}"
            )
        service = self.engine.service(transfer.direction)
        transfer_cycles = math.ceil(transfer.bytes / service.bandwidth_bytes_per_cycle)
        duration_cycles = service.setup_cycles + transfer_cycles
        return CopyServiceEstimate(
            transfer_id=transfer.transfer_id,
            engine_id=self.engine_id,
            direction=transfer.direction,
            source=transfer.source,
            destination=transfer.destination,
            duration_cycles=duration_cycles,
            duration_ps=_cycles_to_ps(duration_cycles, self.engine.clock_hz),
            setup_cycles=service.setup_cycles,
            transfer_cycles=transfer_cycles,
            bytes_transferred=transfer.bytes,
            effective_bandwidth_bytes_per_cycle=transfer.bytes / duration_cycles,
            relative_uncertainty=self.architecture.calibration.relative_uncertainty,
        )


class TraceCalibratedGpuProvider(ComputeProvider):
    """O(1) ``ComputeProvider`` adapter over estimates replayed at construction."""

    precision_compute_level = "profile-table"

    def __init__(
        self,
        architectures: Iterable[GpuArchitectureProfile],
        catalog: Mapping[ProfileKey, KernelLaunch],
    ):
        self.architectures = tuple(architectures)
        _require_nonempty("architectures", self.architectures)
        self._architectures_by_name: dict[str, GpuArchitectureProfile] = {}
        for architecture in self.architectures:
            if not isinstance(architecture, GpuArchitectureProfile):
                raise TypeError("architectures must contain GpuArchitectureProfile records")
            for name in architecture.all_names:
                if name in self._architectures_by_name:
                    raise ValueError(f"duplicate GPU profile name or alias {name!r}")
                self._architectures_by_name[name] = architecture
        self.catalog: dict[ProfileKey, KernelLaunch] = {}
        for (kernel_name, config, gpu_name), launch in catalog.items():
            _require_text("kernel name", kernel_name)
            architecture = self._architectures_by_name.get(gpu_name)
            if architecture is None:
                raise ValueError(f"catalog references unknown GPU profile or alias {gpu_name!r}")
            normalized = (kernel_name, tuple(config), architecture.profile_id)
            if normalized in self.catalog:
                raise ValueError(f"duplicate normalized trace catalog key {normalized!r}")
            if not isinstance(launch, KernelLaunch):
                raise TypeError("trace catalog values must be KernelLaunch records")
            self.catalog[normalized] = launch
        self._replays = {
            key: SmSchedulerModel(self._architectures_by_name[key[2]]).estimate(launch)
            for key, launch in self.catalog.items()
        }

    def replay(self, kernel: KernelSpec, gpu: GpuSpec) -> GpuKernelEstimate:
        architecture = self._architectures_by_name.get(gpu.name)
        if architecture is None:
            raise KeyError(f"no trace-calibrated architecture for GPU {gpu.name!r}")
        key = (kernel.name, kernel.config, architecture.profile_id)
        try:
            replay = self._replays[key]
        except KeyError:
            raise KeyError(f"no exact trace launch for {key!r}") from None
        return replay

    def estimate(self, kernel: KernelSpec, gpu: GpuSpec) -> DurationEstimate:
        architecture = self._architectures_by_name.get(gpu.name)
        if architecture is None:
            raise KeyError(f"no trace-calibrated architecture for GPU {gpu.name!r}")
        replay = self.replay(kernel, gpu)
        return DurationEstimate(
            duration_ps=replay.duration_ps,
            bound="sass-replay",
            uncertainty=architecture.calibration.relative_uncertainty,
        )


def a100_sxm_80gb_seed_profile() -> GpuArchitectureProfile:
    """Return the high-uncertainty public-data A100 seed profile."""

    return _seed_profile("a100")


def h100_sxm_80gb_seed_profile() -> GpuArchitectureProfile:
    """Return the high-uncertainty public-data H100 seed profile."""

    return _seed_profile("h100")


def _seed_profile(gpu: str) -> GpuArchitectureProfile:
    if gpu == "a100":
        gpu_name = "NVIDIA A100-SXM4-80GB"
        profile_id = "a100-sxm-80gb-public-seed-v2"
        aliases = ("a100", "A100", "A100-SXM-80GB")
        sm_count = 108
        clock_hz = 1_410_000_000
        shared_memory = 164 * 1024
        max_shared_memory_per_block = 163 * 1024
        hbm_bytes_per_second = 2_039_000_000_000
        provenance_source = (
            "NVIDIA A100 SXM public structural specifications plus non-SKU-matched "
            "A100 microbenchmark timing priors transferred at high uncertainty"
        )
        provenance_version = "a100-public-structure-transferred-timing-v2"
        references = (
            (
                "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/"
                "nvidia-ampere-architecture-whitepaper.pdf"
            ),
            (
                "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/"
                "a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf"
            ),
            "https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html",
            "https://arxiv.org/abs/2208.11174",
            "https://arxiv.org/abs/2501.12084",
        )
    elif gpu == "h100":
        gpu_name = "NVIDIA H100 SXM 80GB HBM3"
        profile_id = "h100-sxm-80gb-public-seed-v2"
        aliases = ("h100", "H100", "H100-SXM-80GB")
        sm_count = 132
        clock_hz = 1_980_000_000
        shared_memory = 228 * 1024
        max_shared_memory_per_block = 227 * 1024
        hbm_bytes_per_second = 3_350_000_000_000
        provenance_source = (
            "NVIDIA H100 SXM public structural specifications plus H800 PCIe "
            "microbenchmark timing priors transferred at high uncertainty"
        )
        provenance_version = "h100-public-structure-h800-timing-transfer-v2"
        references = (
            "https://resources.nvidia.com/en-us-tensor-core/nvidia-h100-datasheet",
            "https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html",
            "https://arxiv.org/abs/2402.13499",
            "https://arxiv.org/abs/2501.12084",
        )
    else:
        raise ValueError(f"unknown built-in seed profile {gpu!r}")

    is_hopper = gpu == "h100"
    # Initiation intervals are sustained-throughput priors derived from the
    # whitepaper per-SM unit counts by one rule: units per SM divided by the
    # 32 threads of a warp gives sustained warp-instructions per cycle, and
    # issue_width (4, the four subcore pipes) divided by the initiation
    # interval must equal that rate. A100 64 FP32 cores/SM sustain 2 (4/2),
    # H100 128 sustain 4 (4/1); INT32 64/SM sustains 2 on both; FP64 32/SM
    # sustains 1 on A100 and 64/SM sustains 2 on H100; 32 LD/ST units/SM
    # sustain 1 (4/4) and 16 SFU units/SM sustain 0.5 (4/8). Tensor
    # initiation is a high-uncertainty throughput prior pending real
    # captures, like every other seed number here.
    pipelines = (
        PipelineProfile(
            kind=PipelineKind.ALU,
            opcodes=("ALU", "FP32", "LOGIC"),
            latency_cycles=4,
            issue_width_per_sm=4,
            initiation_interval_cycles=1 if is_hopper else 2,
        ),
        PipelineProfile(
            kind=PipelineKind.INT,
            opcodes=("INT",),
            latency_cycles=4,
            issue_width_per_sm=4,
            initiation_interval_cycles=2,
        ),
        PipelineProfile(
            kind=PipelineKind.FP64,
            opcodes=("FP64",),
            latency_cycles=4,
            issue_width_per_sm=4,
            initiation_interval_cycles=2 if is_hopper else 4,
        ),
        PipelineProfile(
            kind=PipelineKind.TENSOR,
            opcodes=("TENSOR", "HMMA"),
            latency_cycles=24 if is_hopper else 25,
            issue_width_per_sm=4,
            initiation_interval_cycles=4,
        ),
        PipelineProfile(
            kind=PipelineKind.LOAD_STORE,
            opcodes=("MEMORY", "LD", "ST", "LDG", "STG"),
            latency_cycles=1,
            issue_width_per_sm=4,
            initiation_interval_cycles=4,
        ),
        PipelineProfile(
            kind=PipelineKind.SPECIAL_FUNCTION,
            opcodes=("SFU", "MUFU"),
            latency_cycles=8,
            issue_width_per_sm=4,
            initiation_interval_cycles=8,
        ),
        PipelineProfile(
            kind=PipelineKind.CONTROL,
            opcodes=("CONTROL", "BRANCH", "NOP"),
            latency_cycles=1,
            issue_width_per_sm=4,
        ),
    )
    provenance = GpuModelProvenance(
        source=provenance_source,
        version=provenance_version,
        gpu=gpu_name,
        created="2026-08-06",
        references=references,
    )
    return GpuArchitectureProfile(
        profile_id=profile_id,
        gpu_name=gpu_name,
        aliases=aliases,
        sm_count=sm_count,
        warp_size=32,
        scheduler_count_per_sm=4,
        dispatch_width_per_scheduler=1,
        max_blocks_per_sm=32,
        max_warps_per_sm=64,
        max_threads_per_sm=2048,
        max_threads_per_block=1024,
        registers_per_sm=65_536,
        max_registers_per_thread=255,
        register_allocation_granularity_per_warp=256,
        shared_memory_per_sm=shared_memory,
        max_static_shared_memory_per_block=48 * 1024,
        max_shared_memory_per_block=max_shared_memory_per_block,
        shared_memory_allocation_granularity=256,
        calibration=GpuCalibrationProfile(
            calibration_id=f"{profile_id}-calibration",
            target_architecture_profile_id=profile_id,
            provenance=provenance,
            core_clock_hz=clock_hz,
            target_memory_clock_hz=None,
            pipelines=pipelines,
            memory=MemoryHierarchyProfile(
                hbm_latency_cycles=656 if is_hopper else 566,
                hbm_bandwidth_bytes_per_cycle=hbm_bytes_per_second / clock_hz,
                l2_latency_cycles=265 if is_hopper else 203,
                l1_latency_cycles=32 if is_hopper else 33,
                shared_latency_cycles=29,
            ),
            copy_engines=(),
            warp_scheduler_policy=WarpSchedulerPolicy.LOOSE_ROUND_ROBIN,
            relative_uncertainty=0.50,
        ),
    )


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_enum(name: str, value: object, enum_type: type[Enum]) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{name} must be a {enum_type.__name__}")


def _require_nonempty(name: str, values: object) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")


def _require_unique(name: str, values: tuple[object, ...]) -> None:
    try:
        unique_count = len(set(values))
    except TypeError as exc:
        raise ValueError(f"{name} must contain hashable values") from exc
    if unique_count != len(values):
        raise ValueError(f"{name} must not contain duplicates")


def _require_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_bool(name: str, value: object) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")


def _require_positive_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _round_up(value: int, granularity: int) -> int:
    if value == 0:
        return 0
    return _ceil_div(value, granularity) * granularity


def _checked_service_cycles(
    *,
    transacted_bytes: int,
    bandwidth_bytes_per_cycle: float,
    resource: str,
) -> int:
    """Return the existing float-based service price with explicit overflow."""

    try:
        return math.ceil(transacted_bytes / bandwidth_bytes_per_cycle)
    except OverflowError as exc:
        raise ValueError(
            f"{resource} service cycle count exceeds the supported numeric range"
        ) from exc


def _cycles_to_ps(cycles: int, clock_hz: int) -> int:
    return _ceil_div(cycles * PS_PER_SECOND, clock_hz)


__all__ = [
    "GPU_MODEL_IMPLEMENTATION",
    "CopyDirection",
    "CopyDirectionProfile",
    "CopyEngineProfile",
    "CopyEngineServiceModel",
    "CopyServiceEstimate",
    "CopyTransfer",
    "CtaTrace",
    "GpuArchitectureProfile",
    "GpuCalibrationProfile",
    "GpuConcurrentEstimate",
    "GpuKernelEstimate",
    "GpuModelProvenance",
    "GpuTask",
    "GpuTaskEstimate",
    "GpuTaskKind",
    "KernelLaunch",
    "MemoryHierarchyProfile",
    "MemorySpace",
    "NvlinkProfile",
    "PipelineKind",
    "PipelineProfile",
    "SassInstruction",
    "SassWarpTrace",
    "SmSchedulerModel",
    "TraceCalibratedGpuProvider",
    "WarpSchedulerPolicy",
    "a100_sxm_80gb_seed_profile",
    "h100_sxm_80gb_seed_profile",
]
