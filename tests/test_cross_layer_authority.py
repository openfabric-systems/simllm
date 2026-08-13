"""CORE-8: the completion stream and the ledger project the runtime authority.

Every check here submits a hand-built contradiction that the consumer accepted
before this module existed. The fixture is built so eligibility, the resource
grant and logical submission are three different numbers on at least one visit;
under a zero-wait visit all three coincide and a renderer that confuses them
agrees with the authority by accident.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from simllm.core.authority import (
    check_bookkeeping_projection,
    check_completion_event_projection,
    class_service_bytes,
    work_completed_bytes,
)
from simllm.core.bookkeeping import (
    BookkeepingEntry,
    BookkeepingLedger,
    CreatedObjectKind,
    CreatedObjectRecord,
    ProcessingStage,
    RequestBookkeeper,
    StagePhase,
    StageRecord,
    validate_bookkeeping_ledger,
)
from simllm.core.clock import VirtualClock
from simllm.core.completion import CompletionReducer
from simllm.core.execution import (
    CollectiveWork,
    CompletionEvent,
    ComputeWork,
    ControlWork,
    EventPhase,
    ExecutionGraph,
    ExecutionOperation,
    OperationCorrelation,
)
from simllm.core.runtime import CoarseDeviceProfile, CoarseDeviceRuntime
from simllm.core.step import RequestPhase, ScheduledRequest, StepRecord

CORRELATION = OperationCorrelation(request_ids=("request",))


def _transfer(name: str, queue: str, payload_bytes: int) -> ExecutionOperation:
    return ExecutionOperation(
        name,
        0,
        queue,
        CollectiveWork(
            "all-to-allv",
            (0, 8),
            0,
            "pairwise",
            pair_payload_bytes=((0, 8, payload_bytes),),
        ),
        correlation=CORRELATION,
    )


def _graph(execution_id: str = "cross-layer-authority") -> ExecutionGraph:
    return ExecutionGraph(
        execution_id=execution_id,
        step_index=0,
        released_at_ps=0,
        operations=(
            ExecutionOperation(
                "compute-a",
                0,
                "cuda:0:compute",
                ComputeWork("a", nominal_duration_ps=100_000, hbm_bytes=1024),
                correlation=CORRELATION,
            ),
            _transfer("xfer", "cuda:0:nccl", 4096),
            _transfer("xfer-fifo", "cuda:0:nccl", 2048),
            _transfer("xfer-rival", "cuda:0:nccl-b", 8192),
            ExecutionOperation(
                "ctrl",
                0,
                "cuda:0:ctrl",
                ControlWork("sync", (8,), 128),
                correlation=CORRELATION,
            ),
        ),
    )


def _profile() -> CoarseDeviceProfile:
    return CoarseDeviceProfile(
        launch_service_ps=1000,
        nccl_channel_service_ps=5000,
        control_service_ps=2000,
        completion_delivery_ps=700,
    )


def _record() -> StepRecord:
    return StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[ScheduledRequest("request", RequestPhase.PREFILL, 1)],
        num_sampled=1,
        sampled_request_ids=["request"],
    )


def _evidence(execution_id: str = "cross-layer-authority"):
    graph = _graph(execution_id)
    bookkeeper = RequestBookkeeper()
    runtime = CoarseDeviceRuntime(_profile())
    result = runtime.execute(graph, bookkeeping=bookkeeper)
    report = runtime.last_report
    assert report is not None
    return graph, result, report, bookkeeper.snapshot()


def _reduce(graph, result, report, events=None):
    if events is not None:
        result = replace(
            result,
            events=tuple(sorted(events, key=lambda event: event.timestamp_ps)),
        )
    clock = VirtualClock(0)
    reducer = CompletionReducer(clock)
    reducer.reduce(_record(), graph, result, report)
    return clock


def _relabelled(facts) -> BookkeepingLedger:
    return BookkeepingLedger(
        tuple(BookkeepingEntry(index, fact) for index, fact in enumerate(facts))
    )


def _gated_visit(report):
    return next(
        visit
        for visit in report.visits
        if visit.subject_object_id is None
        and visit.eligible_at_ps > visit.submitted_at_ps
    )


def _contended_visit(report):
    return next(
        visit
        for visit in report.visits
        if visit.subject_object_id is None and visit.queue_wait_ps > 0
    )


def test_fixture_separates_submission_eligibility_and_grant():
    _, _, report, _ = _evidence()

    gated = _gated_visit(report)
    contended = _contended_visit(report)

    assert gated.submitted_at_ps < gated.eligible_at_ps
    assert contended.eligible_at_ps < contended.started_at_ps
    assert len(report.wqes) == 4


def test_unmutated_evidence_is_accepted_by_both_projections():
    graph, result, report, ledger = _evidence()

    check_completion_event_projection(graph, result, report)
    check_bookkeeping_projection(ledger, graph, result, report)
    clock = _reduce(graph, result, report)

    assert clock.now_ps == result.completed_at_ps


def test_queued_event_must_project_eligibility_not_submission():
    graph, result, report, _ = _evidence()
    gated = _gated_visit(report)
    events = [
        replace(event, timestamp_ps=gated.submitted_at_ps)
        if (
            event.phase is EventPhase.QUEUED
            and event.subject_object_id is None
            and event.operation_id == gated.operation_id
            and event.resource == gated.resource
            and event.timestamp_ps == gated.eligible_at_ps
        )
        else event
        for event in result.events
    ]

    with pytest.raises(ValueError, match="disagrees with the runtime authority"):
        _reduce(graph, result, report, events)


def test_started_event_must_project_the_resource_grant():
    graph, result, report, _ = _evidence()
    contended = _contended_visit(report)
    events = []
    for event in result.events:
        if (
            event.subject_object_id is None
            and event.operation_id == contended.operation_id
            and event.resource == contended.resource
        ):
            if (
                event.phase is EventPhase.QUEUED
                and event.timestamp_ps == contended.eligible_at_ps
            ):
                event = replace(event, timestamp_ps=contended.started_at_ps)
            elif (
                event.phase is EventPhase.STARTED
                and event.timestamp_ps == contended.started_at_ps
            ):
                event = replace(event, timestamp_ps=contended.eligible_at_ps)
        events.append(event)

    with pytest.raises(ValueError, match="disagrees with the runtime authority"):
        _reduce(graph, result, report, events)


def test_lost_phase_event_is_rejected():
    graph, result, report, _ = _evidence()
    dropped = next(
        event
        for event in result.events
        if event.phase is EventPhase.SUBMITTED and event.subject_object_id is None
    )
    events = [event for event in result.events if event is not dropped]

    with pytest.raises(ValueError, match="lost a projection"):
        _reduce(graph, result, report, events)


def test_duplicated_phase_event_is_rejected():
    graph, result, report, _ = _evidence()
    duplicated = next(
        event
        for event in result.events
        if event.phase is EventPhase.STARTED and event.subject_object_id is None
    )

    with pytest.raises(ValueError, match="the runtime authority does not"):
        _reduce(graph, result, report, [*result.events, duplicated])


def test_progress_byte_count_must_project_the_visit():
    graph, result, report, _ = _evidence()
    events = []
    changed = False
    for event in result.events:
        if (
            not changed
            and event.phase is EventPhase.PROGRESS
            and event.subject_object_id is None
            and event.completed_bytes
        ):
            event = replace(event, completed_bytes=event.completed_bytes + 1)
            changed = True
        events.append(event)
    assert changed

    with pytest.raises(ValueError, match="the stream reports"):
        _reduce(graph, result, report, events)


def test_wqe_completion_timestamp_must_project_the_wqe_authority():
    graph, result, report, _ = _evidence()
    wqe = report.wqes[0]
    events = [
        replace(event, timestamp_ps=event.timestamp_ps - 1)
        if (
            event.subject_object_id == wqe.wqe_id
            and event.phase is EventPhase.COMPLETED
        )
        else event
        for event in result.events
    ]

    with pytest.raises(ValueError, match="disagrees with the runtime authority"):
        _reduce(graph, result, report, events)


def test_event_for_an_uncreated_object_is_rejected():
    graph, result, report, _ = _evidence()
    wqe = report.wqes[0]
    phantom = replace(
        next(
            event
            for event in result.events
            if event.subject_object_id == wqe.wqe_id
        ),
        subject_object_id="wqe:phantom",
    )

    with pytest.raises(ValueError, match="the runtime authority does not"):
        _reduce(graph, result, report, [*result.events, phantom])


def test_logical_completion_must_carry_the_declared_semantic_bytes():
    graph, result, report, _ = _evidence()
    events = [
        replace(event, completed_bytes=(event.completed_bytes or 0) + 1)
        if (
            event.subject_object_id is None
            and event.phase is EventPhase.COMPLETED
            and event.operation_id == "compute-a"
        )
        else event
        for event in result.events
    ]

    with pytest.raises(ValueError, match="the graph declares"):
        _reduce(graph, result, report, events)


def test_class_service_bytes_must_project_the_graph():
    graph, result, report, _ = _evidence()
    mutated = replace(report, class_service_bytes=((0, 4096),))

    with pytest.raises(ValueError, match="class service bytes"):
        _reduce(graph, result, mutated)


def test_refused_stream_leaves_the_reducer_untouched():
    graph, result, report, _ = _evidence()
    events = [
        event
        for event in result.events
        if not (event.phase is EventPhase.SUBMITTED and event.subject_object_id is None)
    ]
    clock = VirtualClock(0)
    reducer = CompletionReducer(clock)
    refused = replace(result, events=tuple(events))

    with pytest.raises(ValueError):
        reducer.reduce(_record(), graph, refused, report)

    assert clock.now_ps == 0
    assert reducer.latest_request_metrics == ()
    assert reducer.reduce(_record(), graph, result, report).completed_at_ps == (
        result.completed_at_ps
    )


def _wqe_object_index(facts) -> int:
    return next(
        position
        for position, fact in enumerate(facts)
        if isinstance(fact, CreatedObjectRecord)
        and fact.ref.kind is CreatedObjectKind.NETWORK_WQE
    )


def test_ledger_wqe_creation_must_project_its_submission():
    graph, result, report, ledger = _evidence()
    facts = [entry.fact for entry in ledger.entries]
    index = _wqe_object_index(facts)
    facts[index] = replace(facts[index], created_at_ps=facts[index].created_at_ps - 1)
    mutated = _relabelled(facts)

    validate_bookkeeping_ledger(mutated)
    with pytest.raises(ValueError, match="the WQE authority submits it at"):
        check_bookkeeping_projection(mutated, graph, result, report)


def test_ledger_wqe_metadata_must_project_the_wqe_authority():
    graph, result, report, ledger = _evidence()
    facts = [entry.fact for entry in ledger.entries]
    index = _wqe_object_index(facts)
    facts[index] = replace(
        facts[index],
        metadata=tuple(
            (name, value + 1 if name == "bytes" else value)
            for name, value in facts[index].metadata
        ),
    )
    mutated = _relabelled(facts)

    validate_bookkeeping_ledger(mutated)
    with pytest.raises(ValueError, match="the WQE authority owns"):
        check_bookkeeping_projection(mutated, graph, result, report)


def test_ledger_must_not_lose_a_reported_wqe():
    graph, result, report, ledger = _evidence()
    facts = [entry.fact for entry in ledger.entries]
    dropped = facts[_wqe_object_index(facts)].ref.object_id
    remaining = [
        fact
        for fact in facts
        if not (isinstance(fact, CreatedObjectRecord) and fact.ref.object_id == dropped)
        and not (
            isinstance(fact, CompletionEvent) and fact.subject_object_id == dropped
        )
    ]
    mutated = _relabelled(remaining)

    validate_bookkeeping_ledger(mutated)
    with pytest.raises(ValueError, match="bookkeeping ledger lost WQE"):
        check_bookkeeping_projection(mutated, graph, result, report)


def test_ledger_must_not_carry_an_event_the_result_does_not():
    graph, result, report, ledger = _evidence()
    facts = [entry.fact for entry in ledger.entries]
    extra = next(
        fact
        for fact in facts
        if isinstance(fact, CompletionEvent)
        and fact.subject_object_id is None
        and fact.phase is EventPhase.PROGRESS
    )
    mutated = _relabelled([*facts, replace(extra, timestamp_ps=extra.timestamp_ps + 5)])

    validate_bookkeeping_ledger(mutated)
    with pytest.raises(ValueError, match="the execution result does not"):
        check_bookkeeping_projection(mutated, graph, result, report)


def test_ledger_completion_stage_must_project_the_result_boundary():
    graph, result, report, ledger = _evidence()
    facts = [entry.fact for entry in ledger.entries]
    for position, fact in enumerate(facts):
        if (
            isinstance(fact, StageRecord)
            and fact.stage is ProcessingStage.COMPLETION
            and fact.phase is StagePhase.COMPLETED
        ):
            facts[position] = replace(fact, timestamp_ps=fact.timestamp_ps + 1)
            break
    mutated = _relabelled(facts)

    validate_bookkeeping_ledger(mutated)
    with pytest.raises(ValueError, match="the execution result completes at"):
        check_bookkeeping_projection(mutated, graph, result, report)


class _DriftingRuntime(CoarseDeviceRuntime):
    """A runtime whose ledger claims one byte more than its WQE authority."""

    def _bookkeeping_objects(self, bookkeeper, graph, scheduled, wqes):
        records = super()._bookkeeping_objects(bookkeeper, graph, scheduled, wqes)
        drifted = []
        bumped = False
        for record in records:
            if not bumped and record.ref.kind is CreatedObjectKind.NETWORK_WQE:
                record = replace(
                    record,
                    metadata=tuple(
                        (name, value + 1 if name == "bytes" else value)
                        for name, value in record.metadata
                    ),
                )
                bumped = True
            drifted.append(record)
        assert bumped
        return tuple(drifted)


def test_drifting_ledger_is_refused_before_the_bookkeeper_is_mutated():
    graph = _graph("drifting-ledger")
    bookkeeper = RequestBookkeeper()
    runtime = _DriftingRuntime(_profile())

    with pytest.raises(ValueError, match="the WQE authority owns"):
        runtime.execute(graph, bookkeeping=bookkeeper)

    assert bookkeeper.snapshot().entries == ()
    assert runtime.last_report is None


def test_work_completed_bytes_declares_semantic_payload_not_expansion():
    ring = ExecutionOperation(
        "ring",
        0,
        "cuda:0:nccl",
        CollectiveWork("all-reduce", (0, 8, 16, 24), 4096, "ring"),
    )
    sparse = ExecutionOperation(
        "pairwise",
        0,
        "cuda:0:nccl",
        CollectiveWork(
            "all-to-allv",
            (0, 8),
            0,
            "pairwise",
            pair_payload_bytes=((0, 8, 32), (8, 0, 64)),
        ),
    )
    control = ExecutionOperation("ctrl", 0, "cuda:0:ctrl", ControlWork("sync", (8, 16), 10))

    assert work_completed_bytes(ring) == 4096
    assert work_completed_bytes(sparse) == 96
    assert work_completed_bytes(control) == 10
    assert class_service_bytes(ExecutionGraph("g", 0, 0, (control,))) == ((0, 20),)
