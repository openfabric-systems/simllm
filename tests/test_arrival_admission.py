from __future__ import annotations

import pytest

from simllm.core import (
    BookkeepingScope,
    CoarseDeviceProfile,
    CoarseDeviceRuntime,
    CompletionReducer,
    ComputeWork,
    CreatedObjectKind,
    CreatedObjectRecord,
    CreatedObjectRef,
    ExecutionGraph,
    ExecutionOperation,
    LatencyAttribution,
    ObjectOwner,
    OperationCorrelation,
    ProcessingStage,
    RequestBookkeeper,
    RequestPhase,
    ScheduledRequest,
    StagePhase,
    StageRecord,
    StepRecord,
    VirtualClock,
    framework_request_arrivals,
)
from simllm.workload import AdmissionMode, RequestAdmissionGate


def _request(
    request_id: str,
    arrived_at_ps: int,
    *,
    object_id: str | None = None,
    owner: ObjectOwner = ObjectOwner.FRAMEWORK,
) -> CreatedObjectRecord:
    return CreatedObjectRecord(
        ref=CreatedObjectRef(
            CreatedObjectKind.FRAMEWORK_REQUEST,
            object_id or f"request:{request_id}",
        ),
        owner=owner,
        created_at_ps=arrived_at_ps,
        scope=BookkeepingScope(
            correlation=OperationCorrelation(request_ids=(request_id,))
        ),
        native_id=request_id,
        metadata=(("preplay_arrived_at_ps", arrived_at_ps + 999),),
    )


def _bookkeeper(*records: CreatedObjectRecord) -> RequestBookkeeper:
    bookkeeper = RequestBookkeeper()
    bookkeeper.extend(records)
    return bookkeeper


def _step_inputs(
    request_id: str,
    *,
    release_ps: int,
    service_ps: int,
) -> tuple[StepRecord, ExecutionGraph]:
    correlation = OperationCorrelation(request_ids=(request_id,))
    operation = ExecutionOperation(
        operation_id="compute",
        rank=0,
        logical_queue="compute",
        work=ComputeWork("first-token", nominal_duration_ps=service_ps),
        correlation=correlation,
    )
    return (
        StepRecord(
            step_index=0,
            virtual_time_ps=release_ps,
            scheduled=[
                ScheduledRequest(request_id, RequestPhase.PREFILL, 1)
            ],
            num_sampled=1,
            sampled_request_ids=[request_id],
        ),
        ExecutionGraph(
            execution_id="arrival-step",
            step_index=0,
            released_at_ps=release_ps,
            operations=(operation,),
            completion_operation_ids=(operation.operation_id,),
        ),
    )


def _reduce_first_token(
    reducer: CompletionReducer,
    record: StepRecord,
    graph: ExecutionGraph,
):
    runtime = CoarseDeviceRuntime(CoarseDeviceProfile())
    execution = runtime.execute(graph)
    assert runtime.last_report is not None
    return reducer.reduce(record, graph, execution, runtime.last_report)


def test_framework_arrival_projection_uses_creation_time_not_metadata():
    bookkeeper = _bookkeeper(_request("r0", 25))

    arrivals = framework_request_arrivals(bookkeeper.snapshot())

    assert len(arrivals) == 1
    assert arrivals[0].request_id == "r0"
    assert arrivals[0].arrived_at_ps == 25
    assert arrivals[0].sequence == 0
    assert arrivals[0].request_ref.object_id == "request:r0"


def test_framework_arrival_projection_rejects_wrong_owner():
    bookkeeper = _bookkeeper(
        _request("r0", 0, owner=ObjectOwner.CORE)
    )

    with pytest.raises(ValueError, match="must be owned by the framework"):
        framework_request_arrivals(bookkeeper.snapshot())


def test_gated_admission_orders_arrivals_and_records_successful_handoffs():
    bookkeeper = _bookkeeper(
        _request("late-first", 10),
        _request("early", 0),
        _request("late-second", 10),
    )
    clock = VirtualClock()
    gate = RequestAdmissionGate(
        clock,
        bookkeeper,
        mode=AdmissionMode.ARRIVAL_GATED,
    )
    submitted: list[str] = []

    assert gate.pending_request_ids == ("early", "late-first", "late-second")
    assert tuple(row.request_id for row in gate.ready()) == ("early",)
    gate.admit_ready(lambda arrival: submitted.append(arrival.request_id))
    assert submitted == ["early"]
    assert gate.admitted_request_ids == ("early",)
    assert gate.next_arrival_ps == 10

    assert gate.advance_to_next_arrival() == 10
    assert tuple(row.request_id for row in gate.ready()) == (
        "late-first",
        "late-second",
    )
    gate.admit_ready(lambda arrival: submitted.append(arrival.request_id))

    assert submitted == ["early", "late-first", "late-second"]
    assert not gate.has_pending
    assert gate.next_arrival_ps is None
    stage_facts = [
        entry.fact
        for entry in bookkeeper.snapshot().entries
        if isinstance(entry.fact, StageRecord)
    ]
    assert [fact.timestamp_ps for fact in stage_facts] == [0, 10, 10]
    assert all(fact.stage is ProcessingStage.SCHEDULER for fact in stage_facts)
    assert all(fact.phase is StagePhase.ENTERED for fact in stage_facts)
    assert [fact.scope.correlation.request_ids for fact in stage_facts] == [
        ("early",),
        ("late-first",),
        ("late-second",),
    ]


def test_failed_add_request_leaves_gate_and_bookkeeping_unchanged():
    bookkeeper = _bookkeeper(_request("r0", 0))
    before = bookkeeper.snapshot()
    gate = RequestAdmissionGate(
        VirtualClock(),
        bookkeeper,
        mode=AdmissionMode.ARRIVAL_GATED,
    )

    def fail(_arrival):
        raise RuntimeError("framework rejected request")

    with pytest.raises(RuntimeError, match="framework rejected request"):
        gate.admit_ready(fail)

    assert gate.pending_request_ids == ("r0",)
    assert gate.admitted_request_ids == ()
    assert bookkeeper.snapshot() == before


def test_all_at_once_default_preserves_ledger_order_clock_and_bookkeeping():
    bookkeeper = _bookkeeper(
        _request("future", 100),
        _request("earlier", 50),
    )
    before = bookkeeper.snapshot()
    clock = VirtualClock(7)
    gate = RequestAdmissionGate(clock, bookkeeper)
    submitted: list[str] = []

    admitted = gate.admit_ready(
        lambda arrival: submitted.append(arrival.request_id)
    )

    assert gate.mode is AdmissionMode.ALL_AT_ONCE
    assert tuple(row.request_id for row in admitted) == ("future", "earlier")
    assert submitted == ["future", "earlier"]
    assert clock.now_ps == 7
    assert bookkeeper.snapshot() == before
    with pytest.raises(RuntimeError, match="no future arrival gate"):
        gate.advance_to_next_arrival()


def test_completion_reducer_seeds_ttft_and_queue_from_bookkeeping_arrival():
    bookkeeper = _bookkeeper(_request("r0", 25))
    record, graph = _step_inputs("r0", release_ps=100, service_ps=50)
    reducer = CompletionReducer(
        VirtualClock(100),
        bookkeeping=bookkeeper.snapshot(),
    )

    step = _reduce_first_token(reducer, record, graph)

    metric = step.request_metrics[0]
    assert step.step_latency_ps == 50
    assert metric.completed_at_ps == 150
    assert metric.latency_ps == metric.ttft_ps == 125
    assert metric.attribution == LatencyAttribution(queue_ps=75, kernel_ps=50)


def test_completion_reducer_without_bookkeeping_keeps_legacy_metric():
    record, graph = _step_inputs("r0", release_ps=100, service_ps=50)

    metric = _reduce_first_token(
        CompletionReducer(VirtualClock(100)),
        record,
        graph,
    ).request_metrics[0]

    assert metric.latency_ps == metric.ttft_ps == 50
    assert metric.attribution == LatencyAttribution(kernel_ps=50)


def test_completion_reducer_rejects_service_before_arrival_atomically():
    bookkeeper = _bookkeeper(_request("r0", 125))
    record, graph = _step_inputs("r0", release_ps=100, service_ps=50)
    clock = VirtualClock(100)
    reducer = CompletionReducer(clock, bookkeeping=bookkeeper.snapshot())

    with pytest.raises(ValueError, match="predates its bookkeeping arrival"):
        _reduce_first_token(reducer, record, graph)

    assert clock.now_ps == 100
    assert reducer.latest_request_metrics == ()


def test_completion_reducer_rejects_missing_bookkeeping_origin_atomically():
    bookkeeper = _bookkeeper(_request("other", 0))
    record, graph = _step_inputs("r0", release_ps=100, service_ps=50)
    clock = VirtualClock(100)
    reducer = CompletionReducer(clock, bookkeeping=bookkeeper.snapshot())

    with pytest.raises(ValueError, match="has no framework-request bookkeeping"):
        _reduce_first_token(reducer, record, graph)

    assert clock.now_ps == 100
    assert reducer.latest_request_metrics == ()
