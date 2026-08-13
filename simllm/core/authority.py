"""One cross-layer projection authority for runtime evidence.

Every layer above the device runtime carries a copy of facts the runtime owns.
The completion stream repeats each queue visit as four phase events and each
WQE lifecycle as five, the request bookkeeper repeats each WQE as a created
object with byte, tag, rank and sequence metadata, and both repeat the
operation completions the runtime report already carries.

`docs/modules/core.md` has always declared those copies to be read-only
projections, including that `CompletionEvent.QUEUED` projects eligibility and
`CompletionEvent.STARTED` projects the resource grant. This module is what
enforces the declaration. The `RuntimeReport` is the authority: its queue
visits own visit timing, its WQE projections own WQE lifecycle timing, its
operation records own operation completion, and the `ExecutionGraph` owns the
semantic work bytes an operation declares. Each check below derives the
projection the authority implies, joins it to the projection a producer
actually emitted by stable identity, and rejects loss, duplication and
timestamp disagreement.

The checks are read-only. They create no event, visit, WQE or ledger fact and
change no timestamp, digest, completion identity, request metric or random
draw. They exist so that two layers cannot quietly disagree about a quantity
one of them owns.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, TypeAlias

from simllm.core.bookkeeping import (
    BookkeepingLedger,
    CreatedObjectKind,
    CreatedObjectRecord,
    ProcessingStage,
    StagePhase,
    StageRecord,
)
from simllm.core.execution import (
    CollectiveWork,
    CompletionEvent,
    ComputeWork,
    ControlWork,
    DmaWork,
    EventPhase,
    ExecutionGraph,
    ExecutionOperation,
    ExecutionResult,
    KvCacheWork,
    ResourceKind,
    ResourceRef,
)

if TYPE_CHECKING:  # pragma: no cover - import graph only
    from simllm.core.runtime import RuntimeReport

#: Identity an event projection is joined on, plus the values it must carry.
_EventKey: TypeAlias = tuple[
    str,
    str | None,
    EventPhase,
    ResourceRef | None,
    int,
    int | None,
]

#: The stable join key alone, without the projected values.
_JoinKey: TypeAlias = tuple[str, str | None, EventPhase, ResourceRef | None]


def work_completed_bytes(operation: ExecutionOperation) -> int | None:
    """Return the semantic byte count one operation's completion declares.

    This is the graph's declared payload, not the bytes any resource served.
    A ring all-reduce declares its reduced payload here while its physical
    expansion moves several times that; the two are different quantities with
    different owners and are never summed. Both the producer of the logical
    completion event and every consumer that checks it use this one rule.
    """

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


def class_service_bytes(graph: ExecutionGraph) -> tuple[tuple[int, int], ...]:
    """Return the per-class control-work byte reduction the graph implies.

    Only control work carries a class-labelled byte total in the coarse
    report; a destination fan-out sends the payload once per destination.
    """

    totals: dict[int, int] = {}
    for operation in graph.operations:
        work = operation.work
        if not isinstance(work, ControlWork):
            continue
        fanout = max(1, len(work.destination_ranks))
        totals[operation.priority] = (
            totals.get(operation.priority, 0) + work.payload_bytes * fanout
        )
    return tuple(sorted(totals.items()))


def _event_key(event: CompletionEvent) -> _EventKey:
    return (
        event.operation_id,
        event.subject_object_id,
        event.phase,
        event.resource,
        event.timestamp_ps,
        event.completed_bytes,
    )


def _join_key(key: _EventKey) -> _JoinKey:
    return key[0], key[1], key[2], key[3]


def _describe(key: _EventKey) -> str:
    operation_id, subject_object_id, phase, resource, timestamp_ps, byte_count = key
    subject = "" if subject_object_id is None else f" object {subject_object_id!r}"
    place = "" if resource is None else f" on {resource.kind.value} {resource.resource_id!r}"
    payload = "" if byte_count is None else f" carrying {byte_count} bytes"
    return (
        f"{phase.value} event of operation {operation_id!r}{subject}{place} "
        f"at {timestamp_ps} ps{payload}"
    )


def _visit_event_keys(report: RuntimeReport) -> Counter[_EventKey]:
    """Return the phase events every subjectless queue visit projects."""

    expected: Counter[_EventKey] = Counter()
    for visit in report.visits:
        if visit.subject_object_id is not None:
            continue
        for phase, timestamp_ps, byte_count in (
            (EventPhase.SUBMITTED, visit.submitted_at_ps, None),
            (EventPhase.QUEUED, visit.eligible_at_ps, None),
            (EventPhase.STARTED, visit.started_at_ps, None),
            (EventPhase.PROGRESS, visit.finished_at_ps, visit.service_bytes),
        ):
            expected[
                (visit.operation_id, None, phase, visit.resource, timestamp_ps, byte_count)
            ] += 1
    return expected


def _wqe_event_keys(report: RuntimeReport) -> Counter[_EventKey]:
    """Return the lifecycle events every WQE projection projects."""

    expected: Counter[_EventKey] = Counter()
    for wqe in report.wqes:
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
        send_queue = ResourceRef(ResourceKind.NIC_SEND_QUEUE, wqe.sq_id)
        nic = ResourceRef(ResourceKind.NIC, wqe.rnic_id)
        completion_queue = ResourceRef(ResourceKind.COMPLETION_QUEUE, wqe.cq_id)
        for phase, timestamp_ps, resource, byte_count in (
            (EventPhase.SUBMITTED, wqe.submitted_at_ps, send_queue, None),
            (EventPhase.QUEUED, queued_at_ps, nic, None),
            (EventPhase.STARTED, started_at_ps, nic, None),
            (EventPhase.PROGRESS, finished_at_ps, nic, wqe.payload_bytes),
            (
                EventPhase.COMPLETED,
                wqe.completed_at_ps,
                completion_queue,
                wqe.payload_bytes,
            ),
        ):
            expected[
                (wqe.operation_id, wqe.wqe_id, phase, resource, timestamp_ps, byte_count)
            ] += 1
    return expected


def _logical_completion_keys(
    graph: ExecutionGraph,
    result: ExecutionResult,
    report: RuntimeReport,
) -> Counter[_EventKey]:
    """Return the one logical completion each operation projects.

    The completion timestamp and byte count come from the authority. Only the
    resource is read back from the emitted event, because the report does not
    say which of an operation's visits carried its logical path; the event's
    choice is then required to be a visit of that operation which completes
    exactly at the authoritative completion.
    """

    by_id = {record.operation_id: record for record in report.operations}
    resources_at: dict[tuple[str, int], set[ResourceRef]] = {}
    for visit in report.visits:
        resources_at.setdefault(
            (visit.operation_id, visit.completed_at_ps), set()
        ).add(visit.resource)

    emitted: dict[str, list[CompletionEvent]] = {}
    for event in result.events:
        if event.subject_object_id is None and event.phase is EventPhase.COMPLETED:
            emitted.setdefault(event.operation_id, []).append(event)

    expected: Counter[_EventKey] = Counter()
    for operation in graph.operations:
        name = operation.operation_id
        record = by_id.get(name)
        if record is None:
            raise ValueError(
                f"operation {name!r} has no runtime record to project a completion from"
            )
        events = emitted.get(name, [])
        if len(events) != 1:
            raise ValueError(
                f"operation {name!r} projects exactly one logical completion event, "
                f"the stream carries {len(events)}"
            )
        event = events[0]
        if event.timestamp_ps != record.completed_at_ps:
            raise ValueError(
                f"operation {name!r} logical completion event at "
                f"{event.timestamp_ps} ps disagrees with the runtime record's "
                f"{record.completed_at_ps} ps"
            )
        declared_bytes = work_completed_bytes(operation)
        if event.completed_bytes != declared_bytes:
            raise ValueError(
                f"operation {name!r} logical completion event carries "
                f"{event.completed_bytes} bytes, the graph declares {declared_bytes}"
            )
        candidates = resources_at.get((name, record.completed_at_ps), set())
        if event.resource not in candidates:
            raise ValueError(
                f"operation {name!r} logical completion event names "
                f"{event.resource}, which is not a visit of that operation "
                f"completing at {record.completed_at_ps} ps"
            )
        expected[_event_key(event)] += 1
    return expected


def _reject_difference(
    expected: Counter[_EventKey],
    actual: Counter[_EventKey],
) -> None:
    missing = expected - actual
    extra = actual - expected
    if not missing and not extra:
        return

    extra_by_join: dict[_JoinKey, list[_EventKey]] = {}
    for key in extra:
        extra_by_join.setdefault(_join_key(key), []).append(key)

    for key in sorted(missing, key=_describe):
        rivals = extra_by_join.get(_join_key(key), [])
        for rival in rivals:
            if rival[4] != key[4]:
                raise ValueError(
                    "completion stream disagrees with the runtime authority: "
                    f"expected {_describe(key)}, the stream reports {rival[4]} ps"
                )
            raise ValueError(
                "completion stream disagrees with the runtime authority: "
                f"expected {_describe(key)}, the stream reports "
                f"{rival[5]} bytes"
            )
        raise ValueError(
            f"completion stream lost a projection: no {_describe(key)}"
        )

    for key in sorted(extra, key=_describe):
        raise ValueError(
            "completion stream carries a projection the runtime authority does "
            f"not: {_describe(key)}"
        )


def check_completion_event_projection(
    graph: ExecutionGraph,
    result: ExecutionResult,
    report: RuntimeReport,
) -> None:
    """Reject any completion stream that is not the report's exact projection.

    The stream must equal, as a multiset joined on
    ``(operation_id, subject_object_id, phase, resource)``, the four phase
    events of every subjectless queue visit, the five lifecycle events of
    every WQE projection, and one logical completion per graph operation.
    """

    for wqe in report.wqes:
        if wqe.execution_id != graph.execution_id:
            raise ValueError(
                f"WQE {wqe.wqe_id!r} names execution {wqe.execution_id!r}, "
                f"not {graph.execution_id!r}"
            )
    subject_ids = {
        visit.subject_object_id
        for visit in report.visits
        if visit.subject_object_id is not None
    }
    wqe_ids = {wqe.wqe_id for wqe in report.wqes}
    if subject_ids != wqe_ids:
        unowned = sorted(subject_ids - wqe_ids)
        unvisited = sorted(wqe_ids - subject_ids)
        raise ValueError(
            "runtime visit ledger and WQE authority disagree on created objects: "
            f"visits without a WQE {unowned}, WQEs without a visit {unvisited}"
        )
    declared = class_service_bytes(graph)
    if tuple(report.class_service_bytes) != declared:
        raise ValueError(
            "report class service bytes "
            f"{tuple(report.class_service_bytes)} disagree with the graph's "
            f"control-work reduction {declared}"
        )

    expected = _visit_event_keys(report)
    expected += _wqe_event_keys(report)
    expected += _logical_completion_keys(graph, result, report)
    actual: Counter[_EventKey] = Counter(_event_key(event) for event in result.events)
    _reject_difference(expected, actual)


def check_bookkeeping_projection(
    ledger: BookkeepingLedger,
    graph: ExecutionGraph,
    result: ExecutionResult,
    report: RuntimeReport,
) -> None:
    """Reject any ledger that is not the runtime authority's exact projection.

    The ledger is joined to the report by WQE identity and to the result by
    completion-event identity. It must carry one network-WQE object per
    reported WQE with that WQE's creation time, bytes, tag, ranks, sequences
    and channel, exactly the result's completion events for this execution,
    and exactly one completion stage record at the result boundary.
    """

    if not isinstance(ledger, BookkeepingLedger):
        raise TypeError("ledger must be a BookkeepingLedger")

    wqe_objects: dict[str, CreatedObjectRecord] = {}
    for entry in ledger.entries:
        fact = entry.fact
        if not isinstance(fact, CreatedObjectRecord):
            continue
        if fact.ref.kind is not CreatedObjectKind.NETWORK_WQE:
            continue
        if fact.scope.execution_id != graph.execution_id:
            continue
        if fact.ref.object_id in wqe_objects:
            raise ValueError(
                f"bookkeeping ledger carries more than one object for WQE "
                f"{fact.ref.object_id!r}"
            )
        wqe_objects[fact.ref.object_id] = fact

    for wqe in report.wqes:
        record = wqe_objects.pop(wqe.wqe_id, None)
        if record is None:
            raise ValueError(
                f"bookkeeping ledger lost WQE {wqe.wqe_id!r}, which the runtime "
                "authority reports"
            )
        if record.created_at_ps != wqe.submitted_at_ps:
            raise ValueError(
                f"bookkeeping ledger creates WQE {wqe.wqe_id!r} at "
                f"{record.created_at_ps} ps, the WQE authority submits it at "
                f"{wqe.submitted_at_ps} ps"
            )
        if record.native_id != wqe.native_wqe_id:
            raise ValueError(
                f"bookkeeping ledger names WQE {wqe.wqe_id!r} natively "
                f"{record.native_id!r}, the WQE authority names it "
                f"{wqe.native_wqe_id!r}"
            )
        metadata = dict(record.metadata)
        for name, owned in (
            ("bytes", wqe.payload_bytes),
            ("goal_tag", wqe.goal_tag),
            ("source_rank", wqe.source_rank),
            ("destination_rank", wqe.destination_rank),
            ("extent_index", wqe.extent_index),
            ("sq_post_sequence", wqe.sq_post_sequence),
            ("cq_post_sequence", wqe.cq_post_sequence),
            ("channel", wqe.channel_id),
            ("graph_operation_id", wqe.operation_id),
            ("authority", wqe.authority),
        ):
            if metadata.get(name) != owned:
                raise ValueError(
                    f"bookkeeping ledger records {name}={metadata.get(name)!r} for "
                    f"WQE {wqe.wqe_id!r}, the WQE authority owns {owned!r}"
                )
    if wqe_objects:
        raise ValueError(
            "bookkeeping ledger invents WQE objects the runtime authority does "
            f"not report: {sorted(wqe_objects)}"
        )

    ledger_events: Counter[_EventKey] = Counter()
    for entry in ledger.entries:
        fact = entry.fact
        if isinstance(fact, CompletionEvent) and fact.execution_id == graph.execution_id:
            ledger_events[_event_key(fact)] += 1
    result_events: Counter[_EventKey] = Counter(
        _event_key(event) for event in result.events
    )
    if ledger_events != result_events:
        missing = result_events - ledger_events
        extra = ledger_events - result_events
        if missing:
            key = min(missing, key=_describe)
            raise ValueError(
                f"bookkeeping ledger lost a completion event: no {_describe(key)}"
            )
        key = min(extra, key=_describe)
        raise ValueError(
            "bookkeeping ledger carries a completion event the execution result "
            f"does not: {_describe(key)}"
        )

    stages = [
        entry.fact
        for entry in ledger.entries
        if isinstance(entry.fact, StageRecord)
        and entry.fact.stage is ProcessingStage.COMPLETION
        and entry.fact.phase is StagePhase.COMPLETED
        and entry.fact.scope.execution_id == graph.execution_id
    ]
    if len(stages) != 1:
        raise ValueError(
            "bookkeeping ledger projects exactly one completion stage for "
            f"execution {graph.execution_id!r}, it carries {len(stages)}"
        )
    if stages[0].timestamp_ps != result.completed_at_ps:
        raise ValueError(
            f"bookkeeping ledger completes execution {graph.execution_id!r} at "
            f"{stages[0].timestamp_ps} ps, the execution result completes at "
            f"{result.completed_at_ps} ps"
        )


__all__ = [
    "check_bookkeeping_projection",
    "check_completion_event_projection",
    "class_service_bytes",
    "work_completed_bytes",
]
