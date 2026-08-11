"""GPU task construction and scheduling for RNIC submission producers.

The compute scheduler owns producer-task timing. Native RNIC records receive
only the immutable link returned here and remain projections of their existing
WQE authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from simllm.compute.gpu_model import (
    CtaTrace,
    GpuConcurrentEstimate,
    GpuTask,
    GpuTaskKind,
    KernelLaunch,
    MemorySpace,
    PipelineKind,
    SassInstruction,
    SassWarpTrace,
    SmSchedulerModel,
)

RNIC_PRODUCER_IMPLEMENTATION = "simllm-rnic-producer-v1"


class RnicProducerShape(str, Enum):
    """Submission ownership shape shared with the native RNIC boundary."""

    HOST_CPU_DRIVER = "host_cpu_driver"
    CPU_PROXY = "cpu_proxy"
    GPU_INITIATED = "gpu_initiated"


@dataclass(frozen=True, kw_only=True)
class RnicProducerRequest:
    """One caller-timed RNIC submission awaiting optional GPU production."""

    task_id: str
    producer_shape: RnicProducerShape
    wqe_count: int
    baseline_submission_cycle: int
    submitted_cycle: int = 0
    eligible_cycle: int = 0
    wqe_bytes: int = 64
    descriptor_bytes: int = 64
    doorbell_record_bytes: int = 4

    def __post_init__(self) -> None:
        _require_text("task_id", self.task_id)
        if not isinstance(self.producer_shape, RnicProducerShape):
            raise TypeError("producer_shape must be a RnicProducerShape")
        for name in (
            "wqe_count",
            "wqe_bytes",
            "descriptor_bytes",
            "doorbell_record_bytes",
        ):
            _require_positive(name, getattr(self, name))
        for name in (
            "baseline_submission_cycle",
            "submitted_cycle",
            "eligible_cycle",
        ):
            _require_nonnegative(name, getattr(self, name))
        if self.eligible_cycle < self.submitted_cycle:
            raise ValueError("eligible_cycle must not precede submitted_cycle")


@dataclass(frozen=True, kw_only=True)
class RnicProducerTaskLink:
    """Immutable queue-visit projection returned by the compute authority."""

    task_id: str
    producer_shape: RnicProducerShape
    submitted_cycle: int
    eligible_cycle: int
    started_cycle: int
    finished_cycle: int
    completed_cycle: int

    def __post_init__(self) -> None:
        _require_text("task_id", self.task_id)
        if self.producer_shape is RnicProducerShape.HOST_CPU_DRIVER:
            raise ValueError("host CPU submission cannot carry a GPU task link")
        if not isinstance(self.producer_shape, RnicProducerShape):
            raise TypeError("producer_shape must be a RnicProducerShape")
        for name in (
            "submitted_cycle",
            "eligible_cycle",
            "started_cycle",
            "finished_cycle",
            "completed_cycle",
        ):
            _require_nonnegative(name, getattr(self, name))
        if not (
            self.submitted_cycle
            <= self.eligible_cycle
            <= self.started_cycle
            <= self.finished_cycle
            <= self.completed_cycle
        ):
            raise ValueError("producer task-link cycles must be monotonic")


@dataclass(frozen=True, kw_only=True)
class RnicSubmissionScheduleEntry:
    """One caller deadline resolved against optional producer completion."""

    task_id: str
    producer_shape: RnicProducerShape
    baseline_submission_cycle: int
    effective_submission_cycle: int
    producer_task: RnicProducerTaskLink | None


@dataclass(frozen=True, kw_only=True)
class RnicProducerSchedule:
    """Ordered submission projections and their optional concurrent replay."""

    entries: tuple[RnicSubmissionScheduleEntry, ...]
    estimate: GpuConcurrentEstimate | None


class RnicProducerCoupling:
    """Resolve RNIC submission deadlines through the concurrent GPU service.

    The disabled path and the host-CPU shape do not invoke the scheduler. A
    non-host task may finish inside caller slack, in which case its accepted
    RNIC submission timestamp remains byte-identical to the baseline.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        scheduler: SmSchedulerModel | None = None,
    ) -> None:
        if type(enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        if scheduler is not None and not isinstance(scheduler, SmSchedulerModel):
            raise TypeError("scheduler must be a SmSchedulerModel or None")
        self.enabled = enabled
        self.scheduler = scheduler

    def schedule(
        self,
        requests: tuple[RnicProducerRequest, ...],
        *,
        concurrent_tasks: tuple[GpuTask, ...] = (),
    ) -> RnicProducerSchedule:
        """Return caller-visible submission times in request order."""

        requests = tuple(requests)
        concurrent_tasks = tuple(concurrent_tasks)
        if not requests:
            raise ValueError("requests must not be empty")
        if any(not isinstance(request, RnicProducerRequest) for request in requests):
            raise TypeError("requests must contain RnicProducerRequest records")
        if any(not isinstance(task, GpuTask) for task in concurrent_tasks):
            raise TypeError("concurrent_tasks must contain GpuTask records")
        task_ids = tuple(request.task_id for request in requests)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("producer request task IDs must be unique")

        if not self.enabled:
            return RnicProducerSchedule(
                entries=tuple(_bypass_entry(request) for request in requests),
                estimate=None,
            )

        producer_tasks = tuple(
            task
            for request in requests
            if (task := rnic_submission_producer_task(request)) is not None
        )
        if not producer_tasks:
            return RnicProducerSchedule(
                entries=tuple(_bypass_entry(request) for request in requests),
                estimate=None,
            )
        if self.scheduler is None:
            raise ValueError("enabled non-host producer coupling requires a scheduler")

        estimate = self.scheduler.estimate_concurrent(
            (*concurrent_tasks, *producer_tasks)
        )
        estimates_by_id = {task.task_id: task for task in estimate.tasks}
        entries: list[RnicSubmissionScheduleEntry] = []
        for request in requests:
            if request.producer_shape is RnicProducerShape.HOST_CPU_DRIVER:
                entries.append(_bypass_entry(request))
                continue
            task = estimates_by_id[request.task_id]
            link = RnicProducerTaskLink(
                task_id=request.task_id,
                producer_shape=request.producer_shape,
                submitted_cycle=task.submitted_cycle,
                eligible_cycle=task.eligible_cycle,
                started_cycle=task.admitted_cycle,
                finished_cycle=task.completion_cycle,
                completed_cycle=task.completion_cycle,
            )
            entries.append(
                RnicSubmissionScheduleEntry(
                    task_id=request.task_id,
                    producer_shape=request.producer_shape,
                    baseline_submission_cycle=request.baseline_submission_cycle,
                    effective_submission_cycle=max(
                        request.baseline_submission_cycle,
                        link.completed_cycle,
                    ),
                    producer_task=link,
                )
            )
        return RnicProducerSchedule(entries=tuple(entries), estimate=estimate)


def rnic_submission_producer_task(
    request: RnicProducerRequest,
) -> GpuTask | None:
    """Build the GPU-side descriptor or WQE publication task.

    The trace is a normalized GPU instruction-demand shape. Host-visible
    memory and UAR service remain under the native RNIC and PCIe authorities;
    this task accounts only for the producing GPU thread block.
    """

    if not isinstance(request, RnicProducerRequest):
        raise TypeError("request must be a RnicProducerRequest")
    if request.producer_shape is RnicProducerShape.HOST_CPU_DRIVER:
        return None

    if request.producer_shape is RnicProducerShape.CPU_PROXY:
        stores = tuple(
            _ordered_hbm_store(request.descriptor_bytes)
            for _ in range(request.wqe_count)
        )
    else:
        stores = (
            *(
                _ordered_hbm_store(request.wqe_bytes)
                for _ in range(request.wqe_count)
            ),
            _ordered_hbm_store(request.doorbell_record_bytes),
        )
    instructions = (
        *stores,
        SassInstruction(
            opcode="CONTROL",
            pipeline=PipelineKind.CONTROL,
            dependent=True,
        ),
    )
    name = (
        f"{RNIC_PRODUCER_IMPLEMENTATION}-{request.producer_shape.value}-"
        f"wqes{request.wqe_count}"
    )
    launch = KernelLaunch(
        implementation_id=name,
        trace_id=f"{name}-trace",
        grid_blocks=1,
        threads_per_block=32,
        registers_per_thread=8,
        static_shared_memory_bytes=1,
        dynamic_shared_memory_bytes=0,
        cta_traces=(
            CtaTrace(
                trace_class_id=f"{name}-producer-cta",
                block_ids=(0,),
                warp_traces=(
                    SassWarpTrace(warp_id=0, instructions=instructions),
                ),
            ),
        ),
    )
    return GpuTask(
        task_id=request.task_id,
        kind=GpuTaskKind.NETWORK,
        launch=launch,
        submitted_cycle=request.submitted_cycle,
        eligible_cycle=request.eligible_cycle,
    )


def _ordered_hbm_store(byte_count: int) -> SassInstruction:
    return SassInstruction(
        opcode="STG",
        pipeline=PipelineKind.LOAD_STORE,
        dependent=True,
        memory_space=MemorySpace.HBM,
        requested_bytes=byte_count,
        transacted_bytes=byte_count,
    )


def _bypass_entry(request: RnicProducerRequest) -> RnicSubmissionScheduleEntry:
    return RnicSubmissionScheduleEntry(
        task_id=request.task_id,
        producer_shape=request.producer_shape,
        baseline_submission_cycle=request.baseline_submission_cycle,
        effective_submission_cycle=request.baseline_submission_cycle,
        producer_task=None,
    )


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_positive(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


__all__ = [
    "RNIC_PRODUCER_IMPLEMENTATION",
    "RnicProducerCoupling",
    "RnicProducerRequest",
    "RnicProducerSchedule",
    "RnicProducerShape",
    "RnicProducerTaskLink",
    "RnicSubmissionScheduleEntry",
    "rnic_submission_producer_task",
]
