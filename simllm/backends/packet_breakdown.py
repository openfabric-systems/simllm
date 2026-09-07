"""Read-only runtime projections of ordered packet-step artifact visits.

The five timing components and the semantic/media partition are different
projections of the same intervals. Neither reduction adds masked medium work.
Flow start is the backend's published WQE start, not an invented switch probe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from itertools import groupby
from typing import TYPE_CHECKING

from simllm.core import (
    CoarseDeviceRuntime,
    CompletionEvent,
    CriticalPathBreakdown,
    EventPhase,
    ExecutionResult,
    LatencyAttribution,
    QueueVisit,
    ResourceKind,
    ResourceRef,
    RuntimeCriticalSegment,
    RuntimeOperationRecord,
    RuntimeReport,
    StepResult,
    step_result_from_json,
    step_result_to_json,
)
from simllm.core._wire import (
    _array,
    _enum_value,
    _fields,
    _integer,
    _object,
    _optional_string,
    _string,
)

if TYPE_CHECKING:
    from simllm.backends.step_sink import _PlannedStep

SCHEMA = "simllm-packet-step-breakdown-v1"
BREAKDOWN_FIELDS = tuple(f.name for f in fields(CriticalPathBreakdown))


def sum_breakdowns(*values: CriticalPathBreakdown) -> CriticalPathBreakdown:
    """Add disjoint intervals, never concurrent queue visits."""
    return CriticalPathBreakdown(**{
        name: sum(getattr(value, name) for value in values) for name in BREAKDOWN_FIELDS
    })


def dependency_breakdown(duration_ps: int) -> CriticalPathBreakdown:
    return CriticalPathBreakdown(0, 0, 0, 0, duration_ps, duration_ps, 0)


def _media(visits):
    from simllm.backends.step_attribution import MediumAttribution

    values = {f.name: 0 for f in fields(MediumAttribution)}
    for visit in visits:
        if visit.stage not in values:
            raise ValueError("packet visit has an unknown semantic owner")
        owners = {
            "kernel_ps": (ResourceKind.HOST_LAUNCH_QUEUE, ResourceKind.GPU_WORK_QUEUE),
            "collective_registration_ps": (ResourceKind.HOST_LAUNCH_QUEUE,),
            "collective_base_ps": (ResourceKind.CONTROL_QUEUE,),
            "collective_floor_ps": (ResourceKind.CONTROL_QUEUE,),
            "fabric_ps": (ResourceKind.NIC,), "nvlink_ps": (ResourceKind.NVLINK,),
            "co_critical_ps": (ResourceKind.NCCL_CHANNEL,),
        }
        if visit.resource.kind not in owners.get(visit.stage, ()):
            raise ValueError("packet visit resource disagrees with its semantic owner")
        values[visit.stage] += visit.completed_at_ps - visit.eligible_at_ps
    return MediumAttribution(**values)


def _attribution(media):
    return LatencyAttribution(
        queue_ps=media.queue_ps, kernel_ps=media.kernel_ps,
        collective_ps=media.collective_ps, control_ps=media.control_ps,
    )


@dataclass(frozen=True)
class PacketStepBreakdown:
    """Executed visits are the authority; reports and events are projections.

    Each operation ID names an executed artifact, not a packet or a fabricated
    per-rank frontier. Stage retains the existing MediumAttribution field name.
    """

    step_index: int
    visits: tuple[QueueVisit, ...]
    masked_nvlink_ps: int = 0
    masked_fabric_ps: int = 0

    def __post_init__(self):
        for name in ("step_index", "masked_nvlink_ps", "masked_fabric_ps"):
            _integer(getattr(self, name), name, nonnegative=True)
        if not isinstance(self.visits, tuple) or not self.visits:
            raise ValueError("packet breakdown requires a nonempty visit tuple")
        if any(not isinstance(v, QueueVisit) for v in self.visits):
            raise TypeError("packet visits must be QueueVisit records")
        execution_ids = {v.execution_id for v in self.visits}
        if len(execution_ids) != 1:
            raise ValueError("packet visits must belong to one execution")
        cursor = self.visits[0].eligible_at_ps
        submission = self.visits[0].submitted_at_ps
        if submission != cursor:
            raise ValueError("packet release differs from initial eligibility")
        seen = set()
        subjects = set()
        for operation_id, group in groupby(self.visits, key=lambda v: v.operation_id):
            group = tuple(group)
            if operation_id in seen or group[0].resource.kind != ResourceKind.HOST_LAUNCH_QUEUE:
                raise ValueError("artifact must begin with one launch visit")
            primary = {ResourceKind.GPU_WORK_QUEUE, ResourceKind.NIC,
                       ResourceKind.NVLINK, ResourceKind.NCCL_CHANNEL}
            if (group[-1].resource.kind not in primary
                    or sum(v.resource.kind in primary for v in group) != 1):
                raise ValueError("artifact requires exactly one terminal service visit")
            seen.add(operation_id)
            for index, visit in enumerate(group):
                if not isinstance(visit, QueueVisit):
                    raise TypeError("packet visits must be QueueVisit records")
                visit.__post_init__()
                if index and visit.resource.kind == ResourceKind.HOST_LAUNCH_QUEUE:
                    raise ValueError("artifact has multiple launch visits")
                if visit.submitted_at_ps != submission or visit.eligible_at_ps != cursor:
                    raise ValueError("packet visits do not form ordered artifact intervals")
                if visit.subject_object_id is None or visit.subject_object_id in subjects:
                    raise ValueError("packet visit identity is missing or duplicated")
                subjects.add(visit.subject_object_id)
                cursor = visit.completed_at_ps
        _media(self.visits)

    @property
    def segments(self) -> tuple[RuntimeCriticalSegment, ...]:
        result = []
        predecessor = None
        for operation_id, group in groupby(self.visits, key=lambda v: v.operation_id):
            visits = tuple(group)
            start, end = visits[0].eligible_at_ps, visits[-1].completed_at_ps
            # GPU service gates the communication path as external dependency.
            path = tuple(v for v in visits if v.resource.kind != ResourceKind.GPU_WORK_QUEUE)
            result.append(RuntimeCriticalSegment(
                operation_id, 0, start, end, predecessor,
                None if predecessor is None else 0,
                CoarseDeviceRuntime._critical_path_breakdown(operation_id, path, start, end),
                _attribution(_media(visits)),
            ))
            predecessor = operation_id
        return tuple(result)

    @property
    def breakdown(self) -> CriticalPathBreakdown:
        return sum_breakdowns(*(s.breakdown for s in self.segments))

    @property
    def detail(self):
        from simllm.backends.step_attribution import MaskedMediumService, StepAttribution

        media = _media(self.visits)
        return StepAttribution(_attribution(media), media,
                               MaskedMediumService(self.masked_nvlink_ps, self.masked_fabric_ps))

    @property
    def execution_result(self) -> ExecutionResult:
        events = []
        for visit in self.visits:
            for phase, timestamp in (
                (EventPhase.SUBMITTED, visit.submitted_at_ps),
                (EventPhase.QUEUED, visit.eligible_at_ps),
                (EventPhase.STARTED, visit.started_at_ps),
                (EventPhase.PROGRESS, visit.finished_at_ps),
                (EventPhase.COMPLETED, visit.completed_at_ps),
            ):
                events.append(CompletionEvent(
                    visit.execution_id, visit.operation_id, phase, timestamp,
                    visit.resource, subject_object_id=visit.subject_object_id,
                ))
        return ExecutionResult(
            self.visits[0].execution_id, self.visits[-1].completed_at_ps,
            tuple(sorted(events, key=lambda e: e.timestamp_ps)),
            self.visits[-1].completed_at_ps,
        )

    @property
    def runtime_report(self) -> RuntimeReport:
        operations = []
        for segment in self.segments:
            visits = tuple(v for v in self.visits if v.operation_id == segment.operation_id)
            operations.append(RuntimeOperationRecord(
                segment.operation_id, 0, visits[0].submitted_at_ps,
                segment.started_at_ps, segment.completed_at_ps, segment.completed_at_ps,
                ((0, segment.completed_at_ps),), (segment,), segment.breakdown,
                segment.attribution, segment.predecessor_operation_id,
                None if segment.predecessor_operation_id is None else segment.started_at_ps,
                segment.predecessor_operation_id, sum(v.queue_wait_ps for v in visits),
            ))
        return RuntimeReport(
            self.visits[0].execution_id, "ordered-packet-artifacts", tuple(operations),
            self.visits, (), sum(v.queue_wait_ps for v in self.visits),
            self.breakdown.critical_path_queue_ps,
            tuple(s.operation_id for s in self.segments),
            tuple((s.operation_id, 0) for s in self.segments), (), 0,
        )

    def validate_result(self, result: StepResult) -> None:
        if (self.step_index != result.step_index
                or self.visits[-1].completed_at_ps != result.completed_at_ps
                or self.breakdown.operation_latency_ps != result.step_latency_ps):
            raise ValueError("packet breakdown does not match StepResult")


def build_packet_breakdown(plan: _PlannedStep, fabric_services, starts, finishes):
    """Project retained backend rows and analytic intervals without rescheduling."""
    visits = []
    cursor = plan.virtual_time_ps
    host_remaining = plan.exposed_host_ps
    masked_local = masked_fabric = 0
    for artifact, fabric, first, last in zip(
        plan.artifacts, fabric_services, starts, finishes, strict=True
    ):
        operation_id = artifact.artifact_id
        compute = artifact.collective_operation_id is None
        local = artifact.local_service_ps
        host = min(host_remaining, local) if compute else 0
        host_remaining -= host

        def append(kind, stage, duration, *, start=0, finish=None, operation_id=operation_id):
            nonlocal cursor
            end = cursor + duration
            visits.append(QueueVisit(
                plan.locality.graph_execution_id, operation_id,
                ResourceRef(kind, f"packet:{kind.value}"), plan.virtual_time_ps,
                cursor, cursor + start, cursor + (duration if finish is None else finish),
                end, subject_object_id=f"{operation_id}:visit-{len(visits)}", stage=stage,
            ))
            cursor = end

        append(ResourceKind.HOST_LAUNCH_QUEUE,
               "kernel_ps" if compute else "collective_registration_ps",
               host if compute else artifact.registration_cost_ps)
        for stage, duration in (
            ("collective_base_ps", artifact.collective_base_latency_ps),
            ("collective_floor_ps", artifact.aggregate_collective_floor_ps),
        ):
            if duration:
                append(ResourceKind.CONTROL_QUEUE, stage, duration, finish=0)
        if compute:
            append(ResourceKind.GPU_WORK_QUEUE, "kernel_ps", local - host)
        elif fabric > local:
            append(ResourceKind.NIC, "fabric_ps", fabric, start=first, finish=last)
            masked_local += local
        elif local > fabric:
            append(ResourceKind.NVLINK, "nvlink_ps", local)
            masked_fabric += fabric
        else:
            append(ResourceKind.NCCL_CHANNEL, "co_critical_ps", local)
    if host_remaining:
        raise ValueError("represented host floor exceeds charged compute artifacts")
    return PacketStepBreakdown(plan.step_index, tuple(visits), masked_local, masked_fabric)


def packet_step_to_json(result: StepResult, breakdown: PacketStepBreakdown | None = None):
    """Extend the canonical result only when the projection was selected."""
    payload = step_result_to_json(result)
    if breakdown is not None:
        breakdown.__post_init__()
        breakdown.validate_result(result)
        visits = []
        for visit in breakdown.visits:
            row = asdict(visit)
            row["resource"]["kind"] = visit.resource.kind.value
            visits.append(row)
        payload["critical_path_breakdown"] = {
            "schema": SCHEMA, "visits": visits,
            "breakdown": asdict(breakdown.breakdown),
            "masked_nvlink_ps": breakdown.masked_nvlink_ps,
            "masked_fabric_ps": breakdown.masked_fabric_ps,
        }
    return payload


def packet_step_from_json(value):
    """Read old result bytes or strictly validate their optional projection."""
    payload = dict(_object(value, "packet_step"))
    if "critical_path_breakdown" not in payload:
        return step_result_from_json(payload), None
    extension = payload.pop("critical_path_breakdown")
    result = step_result_from_json(payload)
    path = "packet_step.critical_path_breakdown"
    extension = _object(extension, path)
    _fields(extension, path, required={
        "schema", "visits", "breakdown", "masked_nvlink_ps", "masked_fabric_ps",
    })
    if _string(extension["schema"], path + ".schema") != SCHEMA:
        raise ValueError("unsupported packet breakdown schema")
    visits = []
    for index, entry in enumerate(_array(extension["visits"], path + ".visits")):
        p = f"{path}.visits[{index}]"
        row = dict(_object(entry, p))
        _fields(row, p, required={f.name for f in fields(QueueVisit)})
        resource = _object(row.pop("resource"), p + ".resource")
        _fields(resource, p + ".resource", required={"kind", "resource_id"})
        for name, item in row.items():
            if name in ("execution_id", "operation_id"):
                row[name] = _string(item, p + "." + name)
            elif name in ("subject_object_id", "stage"):
                row[name] = _optional_string(item, p + "." + name)
            else:
                row[name] = _integer(item, p + "." + name, nonnegative=True)
        visits.append(QueueVisit(**row, resource=ResourceRef(
            _enum_value(ResourceKind, resource["kind"], p + ".resource.kind"),
            _string(resource["resource_id"], p + ".resource.resource_id"),
        )))
    projection = PacketStepBreakdown(result.step_index, tuple(visits), **{
        name: _integer(extension[name], path + "." + name, nonnegative=True)
        for name in ("masked_nvlink_ps", "masked_fabric_ps")
    })
    raw = _object(extension["breakdown"], path + ".breakdown")
    _fields(raw, path + ".breakdown", required=set(BREAKDOWN_FIELDS))
    declared = CriticalPathBreakdown(**{
        name: _integer(raw[name], path + ".breakdown." + name, nonnegative=True)
        for name in BREAKDOWN_FIELDS
    })
    if declared != projection.breakdown:
        raise ValueError("declared breakdown disagrees with packet visits")
    projection.validate_result(result)
    return result, projection
