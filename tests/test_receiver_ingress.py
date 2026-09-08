from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict, fields, replace
from types import SimpleNamespace

import pytest

from simllm.core import (
    AtlahsWqeLedger,
    CoarseDeviceProfile,
    CoarseDeviceRuntime,
    CollectiveWork,
    CompletionReducer,
    EventPhase,
    ExecutionGraph,
    ExecutionOperation,
    IdentityArbitrationPolicy,
    OperationCorrelation,
    RequestBookkeeper,
    RequestPhase,
    ResourceKind,
    RnicAuthorityMode,
    ScheduledRequest,
    SemanticWqeSubmission,
    StepRecord,
    VirtualClock,
    WqeLifecycleProjection,
    check_bookkeeping_projection,
    check_completion_event_projection,
)
from simllm.traffic import plan_execution_graph_collectives


def _graph(pairs, *, execution_id="ingress", step=0, released=0):
    ranks = tuple(sorted({rank for pair in pairs for rank in pair[:2]})) or (0, 8)
    return ExecutionGraph(
        execution_id,
        step,
        released,
        (
            ExecutionOperation(
                "transfer",
                ranks[0],
                "cuda:nccl",
                CollectiveWork(
                    "all-to-allv", ranks, 0, "pairwise", pair_payload_bytes=tuple(pairs)
                ),
                correlation=OperationCorrelation(request_ids=("request",)),
            ),
        ),
        ("transfer",),
    )


def _run(pairs, *, enabled=True, profile=None, planned=False):
    graph = _graph(pairs)
    if planned:
        graph = plan_execution_graph_collectives(graph)
    runtime = CoarseDeviceRuntime(profile, receiver_ingress=enabled)
    bookkeeper = RequestBookkeeper()
    result = runtime.execute(graph, bookkeeping=bookkeeper)
    report = runtime.last_report
    assert report is not None
    check_bookkeeping_projection(bookkeeper.snapshot(), graph, result, report)
    return graph, runtime, result, report


def _submission(source, destination, size, index=0, eligible=0):
    return SemanticWqeSubmission(
        "ledger",
        "transfer",
        source,
        destination,
        size,
        1000,
        index,
        "channel",
        0,
        eligible,
    )


def _physical(rows):
    return tuple(
        (
            row.wqe_id,
            row.source_rank,
            row.destination_rank,
            row.payload_bytes,
            row.started_at_ps,
            row.finished_at_ps,
            row.completed_at_ps,
        )
        for row in rows
    )


@pytest.mark.parametrize("selection", [None, 0, 1, "true", [], {}])
def test_selection_requires_an_actual_bool(selection):
    with pytest.raises(TypeError, match="receiver_ingress must be a bool"):
        AtlahsWqeLedger(CoarseDeviceProfile(), receiver_ingress=selection)
    with pytest.raises(TypeError, match="receiver_ingress must be a bool"):
        CoarseDeviceRuntime(receiver_ingress=selection)


def test_structural_selection_rejects_double_ownership_before_a_transaction():
    calls = []
    session = SimpleNamespace(
        authority_name="native",
        begin_transaction=lambda: calls.append("began"),
    )
    with pytest.raises(ValueError, match="coarse bypass authority"):
        CoarseDeviceRuntime(
            authority_mode=RnicAuthorityMode.STRUCTURAL,
            native_session=session,
            receiver_ingress=True,
        )
    for selection in ({}, {"receiver_ingress": False}):
        runtime = CoarseDeviceRuntime(
            authority_mode=RnicAuthorityMode.STRUCTURAL,
            native_session=session,
            **selection,
        )
        assert runtime.bypass_ledger is None
    assert calls == []


def test_disabled_records_keep_the_exact_legacy_serialized_field_set():
    graph = _graph(((8, 0, 3), (16, 0, 5)))
    snapshots = []
    for selection in ({}, {"receiver_ingress": False}):
        runtime = CoarseDeviceRuntime(**selection)
        bookkeeper = RequestBookkeeper()
        result = runtime.execute(graph, bookkeeping=bookkeeper)
        report = runtime.last_report
        assert report is not None
        assert all(type(row) is WqeLifecycleProjection for row in report.wqes)
        snapshots.append(
            json.dumps(
                [asdict(result), asdict(report), asdict(bookkeeper.snapshot())],
                sort_keys=True,
                default=str,
            )
        )
    assert snapshots[0] == snapshots[1]
    assert tuple(field.name for field in fields(WqeLifecycleProjection)) == (
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
        "channel_id",
        "nccl_command_id",
        "doorbell_started_at_ps",
        "doorbell_completed_at_ps",
        "network_eligible_at_ps",
        "network_started_at_ps",
        "network_finished_at_ps",
        "network_accepted_at_ps",
        "first_packet_at_ps",
        "last_packet_at_ps",
        "packet_tx_started_at_ps",
    )
    assert not any(visit.stage for visit in report.visits if visit.subject_object_id)


@pytest.mark.parametrize("rate,ps_per_byte", [(200_000_000_000, 40), (400_000_000_000, 20)])
@pytest.mark.parametrize("fan_in", [1, 2, 4, 8])
@pytest.mark.parametrize("size", [3, 4096])
def test_incast_charges_receiver_wait_and_one_wire_service(rate, ps_per_byte, fan_in, size):
    d = size * ps_per_byte
    pairs = tuple((8 * (index + 1), 0, size) for index in range(fan_in))
    graph, runtime, result, report = _run(
        pairs,
        profile=CoarseDeviceProfile(rnic_rate_bps=rate),
    )
    receivers = [visit for visit in report.visits if visit.stage == "coarse_receiver_service"]
    sources = [visit for visit in report.visits if visit.stage == "coarse_source_admission"]
    assert result.completed_at_ps == fan_in * d
    assert [visit.queue_wait_ps for visit in receivers] == [i * d for i in range(fan_in)]
    assert all(visit.service_ps == d for visit in receivers)
    assert all(
        visit.queue_wait_ps == visit.service_ps == visit.service_bytes == 0 for visit in sources
    )
    assert report.sum_visit_wait_ps == fan_in * (fan_in - 1) * d // 2
    assert report.critical_path_queue_ps == (fan_in - 1) * d
    operation = report.operations[0]
    assert operation.breakdown.service_ps == d
    assert operation.attribution.nic_ps == d
    assert operation.attribution.total_ps == fan_in * d
    assert sum(visit.service_bytes for visit in sources + receivers) == fan_in * size
    ledger = runtime.bypass_ledger
    assert ledger is not None
    assert sum(size for _, size in ledger.source_byte_ledger) == fan_in * size
    assert ledger.receiver_byte_ledger == (("node-0:rnic-0", fan_in * size),)
    assert len({row.wqe_id for row in ledger.receiver_port_reservations}) == fan_in
    assert [
        (row.started_at_ps, row.finished_at_ps) for row in ledger.receiver_port_reservations
    ] == [(i * d, (i + 1) * d) for i in range(fan_in)]
    check_completion_event_projection(graph, result, report)


@pytest.mark.parametrize(
    "pairs",
    [
        ((0, 8, 3), (0, 16, 5)),
        ((0, 8, 4096), (8, 0, 4096), (16, 24, 4096), (24, 16, 4096)),
        ((16, 0, 4096), (24, 1, 4096)),
    ],
)
@pytest.mark.parametrize("rate", [200_000_000_000, 400_000_000_000])
def test_dispatch_duplex_and_distinct_receiver_rails_keep_physical_times(pairs, rate):
    profile = CoarseDeviceProfile(rnic_rate_bps=rate)
    _, _, before, baseline = _run(pairs, enabled=False, profile=profile)
    _, runtime, after, report = _run(pairs, profile=profile)
    assert before.completed_at_ps == after.completed_at_ps
    assert _physical(baseline.wqes) == _physical(report.wqes)
    assert all(
        visit.queue_wait_ps == 0
        for visit in report.visits
        if visit.stage == "coarse_receiver_service"
    )
    ledger = runtime.bypass_ledger
    assert ledger is not None
    for reservations in (ledger.source_port_reservations, ledger.receiver_port_reservations):
        by_port = {}
        for row in reservations:
            assert row.started_at_ps >= by_port.get(row.rnic_id, 0)
            by_port[row.rnic_id] = row.finished_at_ps
    assert sum(value for _, value in ledger.source_byte_ledger) == sum(p[2] for p in pairs)
    assert sum(value for _, value in ledger.receiver_byte_ledger) == sum(p[2] for p in pairs)


def test_joint_reservation_declares_source_head_of_line_blocking():
    ledger = AtlahsWqeLedger(CoarseDeviceProfile(), receiver_ingress=True)
    ledger.submit(_submission(8, 0, 10, 0))
    blocked = ledger.submit(_submission(16, 0, 1, 1))
    free_receiver = ledger.submit(_submission(16, 24, 1, 2))
    assert (blocked.source_ready_at_ps, blocked.started_at_ps, blocked.finished_at_ps) == (
        0,
        200,
        220,
    )
    assert free_receiver.source_ready_at_ps == free_receiver.started_at_ps == 220


def test_future_reservation_order_is_preserved_without_calendar_insertion():
    ledger = AtlahsWqeLedger(CoarseDeviceProfile(), receiver_ingress=True)
    future = ledger.submit(_submission(8, 0, 1, 0, eligible=1000))
    earlier = ledger.submit(_submission(16, 0, 1, 1))
    assert (future.started_at_ps, future.finished_at_ps) == (1000, 1020)
    assert earlier.source_ready_at_ps == 0
    assert (earlier.started_at_ps, earlier.finished_at_ps) == (1020, 1040)


def test_clone_isolates_receiver_source_sequences_and_read_only_projections():
    ledger = AtlahsWqeLedger(CoarseDeviceProfile(), receiver_ingress=True)
    ledger.submit(_submission(8, 0, 3))
    before = ledger.records, ledger.source_byte_ledger, ledger.receiver_byte_ledger
    clone = ledger.clone()
    second = clone.submit(_submission(16, 0, 5, 1))
    assert second.started_at_ps == 60
    assert (ledger.records, ledger.source_byte_ledger, ledger.receiver_byte_ledger) == before
    assert ledger.submit(_submission(16, 0, 5, 1)) == second
    with pytest.raises(FrozenInstanceError):
        clone.receiver_port_reservations[0].service_bytes = 0
    with pytest.raises(FrozenInstanceError):
        second.source_ready_at_ps = 1


def test_zero_byte_extents_keep_one_identity_and_no_physical_service():
    ledger = AtlahsWqeLedger(CoarseDeviceProfile(completion_delivery_ps=7), receiver_ingress=True)
    ledger.submit(_submission(8, 0, 3))
    empty = ledger.submit(_submission(16, 0, 0, 1))
    assert (empty.started_at_ps, empty.finished_at_ps, empty.completed_at_ps) == (60, 60, 67)
    assert len(ledger.records) == 2
    assert ledger.receiver_byte_ledger == (("node-0:rnic-0", 3),)


@pytest.mark.parametrize("pairs", [(), ((0, 1, 3),)])
def test_empty_and_intranode_plans_keep_their_existing_authority(pairs):
    _, _, before, baseline = _run(pairs, enabled=False, planned=True)
    _, runtime, after, report = _run(pairs, planned=True)
    assert after == before
    assert report == baseline
    assert report.wqes == ()
    assert runtime.bypass_ledger.receiver_port_reservations == ()


@pytest.mark.parametrize("planned", [False, True])
def test_receiver_completion_reaches_ttft_tpot_and_causal_next_step(planned):
    clock = VirtualClock(0)
    reducer = CompletionReducer(clock)
    runtime = CoarseDeviceRuntime(
        CoarseDeviceProfile(completion_delivery_ps=7),
        receiver_ingress=True,
    )
    for index in range(3):
        record = StepRecord(
            index,
            clock.now_ps,
            [
                ScheduledRequest(
                    "request",
                    RequestPhase.PREFILL if index == 0 else RequestPhase.DECODE,
                    1,
                    context_length=index + 1,
                )
            ],
            num_sampled=1,
            sampled_request_ids=["request"],
        )
        graph = _graph(
            ((8, 0, 3), (16, 0, 5)), execution_id=f"step-{index}", step=index, released=clock.now_ps
        )
        if planned:
            graph = plan_execution_graph_collectives(graph)
        streamed = []
        result = runtime.execute(graph, on_event=streamed.append, bookkeeping=RequestBookkeeper())
        assert tuple(streamed) == result.events
        report = runtime.last_report
        step = reducer.reduce(record, graph, result, report)
        assert step.step_latency_ps == 167
        metric = step.request_metrics[0]
        if index == 0:
            assert metric.ttft_ps == 167
        else:
            assert metric.tpot_ps == 167
        assert report.operations[0].breakdown.service_ps == 100
        assert report.operations[0].breakdown.completion_delivery_ps == 7
        assert report.critical_path_queue_ps == 60
    assert clock.now_ps == 501


@pytest.mark.parametrize("phase", list(EventPhase))
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "mistimed"])
def test_receiver_and_completion_event_projection_is_exact(phase, mutation):
    graph, _, result, report = _run(((8, 0, 3), (16, 0, 5)))
    target = next(
        event
        for event in result.events
        if event.subject_object_id == report.wqes[-1].wqe_id
        and event.phase is phase
        and event.resource.kind
        is (
            ResourceKind.COMPLETION_QUEUE
            if phase is EventPhase.COMPLETED
            else ResourceKind.NIC_RECEIVE_QUEUE
        )
    )
    events = list(result.events)
    if mutation == "missing":
        events.remove(target)
    elif mutation == "duplicate":
        events.append(target)
    else:
        events[events.index(target)] = replace(target, timestamp_ps=target.timestamp_ps + 1)
    with pytest.raises(ValueError):
        check_completion_event_projection(graph, replace(result, events=tuple(events)), report)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "mistimed", "bytes", "stage"])
def test_receiver_visits_are_checked_independently_of_event_production(mutation):
    graph, _, result, report = _run(((8, 0, 3), (16, 0, 5)))
    target = next(
        visit
        for visit in report.visits
        if visit.stage == "coarse_receiver_service" and visit.queue_wait_ps > 0
    )
    visits = list(report.visits)
    if mutation == "missing":
        visits.remove(target)
    elif mutation == "duplicate":
        visits.append(target)
    else:
        changes = {
            "mistimed": {"eligible_at_ps": 1},
            "bytes": {"service_bytes": 4},
            "stage": {"stage": "unknown"},
        }[mutation]
        visits[visits.index(target)] = replace(target, **changes)
    with pytest.raises(ValueError, match="receiver ingress visits"):
        check_completion_event_projection(graph, result, replace(report, visits=tuple(visits)))


def test_late_projection_failure_rolls_back_every_runtime_and_bookkeeping_state(monkeypatch):
    runtime = CoarseDeviceRuntime(receiver_ingress=True)
    bookkeeper = RequestBookkeeper()
    clock = VirtualClock(0)
    seed = _graph(((8, 0, 3),), execution_id="seed")
    runtime.execute(seed, bookkeeping=bookkeeper)
    before = (
        runtime.bypass_ledger.records,
        runtime.bypass_ledger.source_byte_ledger,
        runtime.bypass_ledger.receiver_byte_ledger,
        runtime.last_report,
        runtime.selected_critical_visits,
        bookkeeper.snapshot(),
        clock.now_ps,
    )
    original = runtime._completion_events

    def invalid_events(*args):
        events = original(*args)
        return tuple(
            event for event in events if event.resource.kind is not ResourceKind.NIC_RECEIVE_QUEUE
        )

    graph = _graph(((8, 0, 3), (16, 0, 5)), execution_id="retry")
    with monkeypatch.context() as patch:
        patch.setattr(runtime, "_completion_events", invalid_events)
        with pytest.raises(ValueError):
            runtime.execute(graph, bookkeeping=bookkeeper)
    after = (
        runtime.bypass_ledger.records,
        runtime.bypass_ledger.source_byte_ledger,
        runtime.bypass_ledger.receiver_byte_ledger,
        runtime.last_report,
        runtime.selected_critical_visits,
        bookkeeper.snapshot(),
        clock.now_ps,
    )
    assert after == before
    actual = runtime.execute(graph, bookkeeping=bookkeeper)
    fresh = CoarseDeviceRuntime(receiver_ingress=True)
    fresh_bookkeeper = RequestBookkeeper()
    fresh.execute(seed, bookkeeping=fresh_bookkeeper)
    expected = fresh.execute(graph, bookkeeping=fresh_bookkeeper)
    assert actual == expected
    assert runtime.last_report == fresh.last_report
    assert bookkeeper.snapshot() == fresh_bookkeeper.snapshot()
    assert runtime.bypass_ledger.records == fresh.bypass_ledger.records


def test_identity_arbitration_ignores_class_labels_with_receiver_contention():
    graph = _graph(((8, 0, 3),))
    first = graph.operations[0]
    second = replace(
        first,
        operation_id="second",
        logical_queue="other",
        work=replace(first.work, pair_payload_bytes=((16, 0, 5),), ranks=(0, 16)),
    )
    snapshots = []
    for labels in ((0, 9), (9, 0)):
        labeled = replace(
            graph,
            operations=tuple(
                replace(operation, priority=label)
                for operation, label in zip((first, second), labels, strict=True)
            ),
            completion_operation_ids=("transfer", "second"),
        )
        runtime = CoarseDeviceRuntime(
            receiver_ingress=True, arbitration_policy=IdentityArbitrationPolicy()
        )
        result = runtime.execute(labeled)
        report = runtime.last_report
        snapshots.append(
            (result.events, _physical(report.wqes), report.visits, report.random_draw_count)
        )
    assert snapshots[0] == snapshots[1]


@pytest.mark.parametrize("width,size", [(2, 4), (2, 4096), (4, 4), (4, 4096), (4, 3)])
@pytest.mark.parametrize("rate", [200_000_000_000, 400_000_000_000])
@pytest.mark.parametrize("channel,delivery", [(0, 0), (7000, 7)])
def test_accepted_rings_keep_physical_times_and_one_service_per_round(
    width, size, rate, channel, delivery
):
    ranks = tuple(8 * i for i in range(width))
    graph = plan_execution_graph_collectives(
        ExecutionGraph(
            "ring",
            0,
            0,
            (
                ExecutionOperation(
                    "ring", 0, "nccl", CollectiveWork("all-reduce", ranks, size, "ring")
                ),
            ),
            ("ring",),
        )
    )
    profile = CoarseDeviceProfile(
        rnic_rate_bps=rate, nccl_channel_service_ps=channel, completion_delivery_ps=delivery
    )
    snapshots = []
    for enabled in (False, True):
        runtime = CoarseDeviceRuntime(profile, receiver_ingress=enabled)
        result = runtime.execute(graph, bookkeeping=RequestBookkeeper())
        chunk_ps = max(1, size // width) * 8_000_000_000_000 // rate
        assert result.completed_at_ps == 2 * (width - 1) * (channel + chunk_ps + delivery)
        snapshots.append((result.completed_at_ps, _physical(runtime.last_report.wqes)))
    assert snapshots[0] == snapshots[1]


@pytest.mark.parametrize("width,multiplier", [(2, 1), (4, 5)])
def test_complete_remote_all_to_all_keeps_declared_source_major_order(width, multiplier):
    ranks = tuple(8 * i for i in range(width))
    pairs = tuple(
        (source, destination, 3)
        for source in ranks
        for destination in ranks
        if source != destination
    )
    _, _, result, report = _run(pairs, planned=True)
    assert result.completed_at_ps == multiplier * 60
    assert (
        tuple((row.source_rank, row.destination_rank, row.payload_bytes) for row in report.wqes)
        == pairs
    )
    _, _, baseline, _ = _run(pairs, enabled=False, planned=True)
    assert baseline.completed_at_ps == (width - 1) * 60


@pytest.mark.parametrize(
    "change",
    [
        {"source_ready_at_ps": -1},
        {"source_ready_at_ps": 1},
        {"source_ready_at_ps": True},
        {"destination_rnic_id": ""},
        {
            "doorbell_started_at_ps": 0,
            "doorbell_completed_at_ps": 0,
            "network_eligible_at_ps": 0,
            "network_started_at_ps": 0,
            "network_finished_at_ps": 60,
        },
    ],
)
def test_enabled_metadata_rejects_invalid_boundaries_and_native_stages(change):
    ledger = AtlahsWqeLedger(CoarseDeviceProfile(), receiver_ingress=True)
    row = ledger.submit(_submission(8, 0, 3))
    with pytest.raises((TypeError, ValueError)):
        replace(row, **change)


def test_projection_checker_rejects_duplicate_wqe_identity_and_stale_stage():
    graph, _, result, report = _run(((8, 0, 3), (16, 0, 5)))
    with pytest.raises(ValueError, match="duplicate receiver ingress WQE identity"):
        check_completion_event_projection(
            graph,
            result,
            replace(report, wqes=(*report.wqes, report.wqes[0])),
        )
    changed = replace(report.wqes[-1], source_ready_at_ps=1)
    with pytest.raises(ValueError, match="receiver ingress visits"):
        check_completion_event_projection(
            graph,
            result,
            replace(report, wqes=(*report.wqes[:-1], changed)),
        )
