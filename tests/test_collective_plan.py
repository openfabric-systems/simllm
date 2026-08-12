"""TRAF-14: the immutable traffic-owned collective plan is the sole authority.

The registered qualification lives in ``examples/collective_plan_v1``. These
tests pin its structural claims: the plan reproduces the accepted GOAL pattern
expansion exactly, the runtime schedules only declared extents, the absent-plan
compatibility path is untouched, and a byte-conserving tag or rank-order
perturbation is rejected instead of silently absorbed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from simllm.core import (
    CoarseDeviceProfile,
    CoarseDeviceRuntime,
    CollectivePlan,
    CollectivePlanActionKind,
    CollectiveWork,
    CompletionReducer,
    ComputeWork,
    ExecutionGraph,
    ExecutionOperation,
    OperationCorrelation,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    VirtualClock,
    collective_goal_tags,
    collective_plan_integrity_sha256,
    execution_graph_from_json,
    execution_graph_to_json,
    validate_execution_graph,
)
from simllm.goal import GoalTrace
from simllm.traffic import (
    pairwise_all_to_allv,
    plan_execution_graph_collectives,
    plan_execution_graph_locality,
    render_collective_plan,
    render_serial_execution_graph_goal,
    ring_allreduce,
)
from simllm.traffic.patterns import ring_allreduce as ring_pattern

BASE_TAG = 1_000
RANKS_BY_WORLD = {2: (0, 8), 4: (0, 8, 16, 24)}
PAYLOAD_BYTES = (3, 4, 4_096)
RATES_BPS = (200_000_000_000, 400_000_000_000)
GOAL_RANKS = 32

#: world, payload -> rounds, chunk, messages, directed bytes, internal deps
EXPECTED_RING_ROWS = {
    (2, 3): (2, 1, 4, 4, 4),
    (2, 4): (2, 2, 4, 8, 4),
    (2, 4_096): (2, 2_048, 4, 8_192, 4),
    (4, 3): (6, 1, 24, 24, 40),
    (4, 4): (6, 1, 24, 24, 40),
    (4, 4_096): (6, 1_024, 24, 24_576, 40),
}
SPARSE_PAIRS = {
    "dispatch": ((0, 8, 3), (0, 16, 5)),
    "combine": ((8, 0, 3), (16, 0, 5)),
    "all-local": (),
}
DISPATCH_REQUESTS = (("alpha", 0, 8, 3), ("beta", 0, 16, 5))


def _ring_operation(
    ranks: tuple[int, ...],
    payload_bytes: int,
    *,
    operation_id: str = "ring",
    layer: int = 0,
) -> ExecutionOperation:
    return ExecutionOperation(
        operation_id,
        ranks[0],
        "cuda:0:nccl:tp",
        CollectiveWork("all-reduce", ranks, payload_bytes, "ring", channel_hint="tp"),
        correlation=OperationCorrelation(
            request_ids=("request",),
            batch_id="batch",
            layer=layer,
        ),
    )


def _sparse_operation(
    pair_payload_bytes: tuple[tuple[int, int, int], ...],
    *,
    request_pair_payload_bytes: tuple[tuple[str, int, int, int], ...] = (),
    operation_id: str = "a2av",
    ranks: tuple[int, ...] = (0, 8, 16, 24),
    layer: int = 0,
) -> ExecutionOperation:
    request_ids = tuple(sorted({entry[0] for entry in request_pair_payload_bytes}))
    return ExecutionOperation(
        operation_id,
        ranks[0],
        "cuda:0:nccl:ep",
        CollectiveWork(
            "all-to-allv",
            ranks,
            0,
            "pairwise",
            channel_hint="dispatch",
            pair_payload_bytes=pair_payload_bytes,
            request_pair_payload_bytes=request_pair_payload_bytes,
        ),
        correlation=OperationCorrelation(
            request_ids=request_ids or ("request",),
            batch_id="batch",
            layer=layer,
        ),
    )


def _graph(*operations: ExecutionOperation) -> ExecutionGraph:
    return ExecutionGraph(
        "plan-study",
        0,
        0,
        operations,
        tuple(operation.operation_id for operation in operations),
    )


def _message_rows(trace: GoalTrace):
    return [
        (
            message.operation_id,
            message.source_rank,
            message.destination_rank,
            message.payload_bytes,
            message.tag,
            message.request_payload_bytes,
        )
        for message in trace.messages
    ]


def _dependency_rows(trace: GoalTrace):
    return sorted(
        (
            dependency.rank,
            dependency.operation_label,
            dependency.predecessor_label,
            dependency.relation,
            None if dependency.provenance is None else dependency.provenance.kind.value,
        )
        for dependency in trace.dependencies
    )


def _wqe_rows(runtime: CoarseDeviceRuntime):
    return [
        (
            record.operation_id,
            record.source_rank,
            record.destination_rank,
            record.payload_bytes,
            record.goal_tag,
            record.channel_id,
            record.submitted_at_ps,
            record.eligible_at_ps,
            record.started_at_ps,
            record.finished_at_ps,
            record.completed_at_ps,
        )
        for record in runtime.bypass_ledger.records
    ]


def _execute(graph: ExecutionGraph, rate_bps: int, channel_service_ps: int = 0):
    runtime = CoarseDeviceRuntime(
        CoarseDeviceProfile(
            rnic_rate_bps=rate_bps,
            nccl_channel_service_ps=channel_service_ps,
        )
    )
    events: list = []
    result = runtime.execute(graph, on_event=events.append)
    return result, runtime.last_report, tuple(events), _wqe_rows(runtime)


# --- plan against the accepted pattern expansion -----------------------------


@pytest.mark.parametrize("exact_frontier", [False, True])
@pytest.mark.parametrize("world", sorted(RANKS_BY_WORLD))
@pytest.mark.parametrize("payload_bytes", PAYLOAD_BYTES)
def test_ring_plan_reproduces_the_pattern_rows_and_frozen_registry(
    world: int, payload_bytes: int, exact_frontier: bool
) -> None:
    ranks = RANKS_BY_WORLD[world]
    graph = _graph(_ring_operation(ranks, payload_bytes))
    planned = plan_execution_graph_collectives(graph)
    plan = planned.collective_plans[0]

    expected = GoalTrace(GOAL_RANKS)
    expected_frontiers = ring_pattern(
        expected,
        list(ranks),
        payload_bytes,
        BASE_TAG,
        operation_id="ring",
        exact_frontier=exact_frontier,
    )
    observed = GoalTrace(GOAL_RANKS)
    observed_frontiers = render_collective_plan(
        observed,
        plan,
        exact_frontier=exact_frontier,
    )

    assert _message_rows(observed) == _message_rows(expected)
    assert _dependency_rows(observed) == _dependency_rows(expected)
    assert observed_frontiers == expected_frontiers
    assert observed.render() == expected.render()

    rounds, chunk, messages, directed_bytes, dependencies = EXPECTED_RING_ROWS[
        (world, payload_bytes)
    ]
    assert len(plan.rounds) == rounds
    assert tuple(round_.tag for round_ in plan.rounds) == tuple(
        range(BASE_TAG, BASE_TAG + rounds)
    )
    assert {extent.payload_bytes for extent in plan.extents} == {chunk}
    assert len(plan.extents) == messages
    assert sum(extent.payload_bytes for extent in plan.extents) == directed_bytes
    assert sum(len(action.depends_on) for action in plan.actions) == dependencies
    assert plan.rank_order == ranks
    assert plan.channel_id == "tp"
    for index, extent in enumerate(plan.extents):
        source_index = index % world
        assert extent.source_rank == ranks[source_index]
        assert extent.destination_rank == ranks[(source_index + 1) % world]


@pytest.mark.parametrize("exact_frontier", [False, True])
@pytest.mark.parametrize("case", sorted(SPARSE_PAIRS))
def test_sparse_plan_reproduces_the_pattern_rows_without_inventing_traffic(
    case: str, exact_frontier: bool
) -> None:
    pairs = SPARSE_PAIRS[case]
    requests = DISPATCH_REQUESTS if case == "dispatch" else ()
    operation = _sparse_operation(pairs, request_pair_payload_bytes=requests)
    graph = _graph(operation)
    planned = plan_execution_graph_collectives(graph)
    plan = planned.collective_plans[0]
    ranks = operation.work.ranks

    expected = GoalTrace(GOAL_RANKS)
    request_send_bytes = {
        (source, destination): tuple(
            (request_id, size)
            for request_id, entry_source, entry_destination, size in requests
            if (entry_source, entry_destination) == (source, destination)
        )
        for source, destination, _ in pairs
    }
    expected_frontiers = pairwise_all_to_allv(
        expected,
        list(ranks),
        {(source, destination): size for source, destination, size in pairs},
        BASE_TAG,
        operation_id="a2av",
        exact_frontier=exact_frontier,
        request_send_bytes=request_send_bytes if requests else None,
    )
    observed = GoalTrace(GOAL_RANKS)
    observed_frontiers = render_collective_plan(
        observed,
        plan,
        exact_frontier=exact_frontier,
    )

    assert _message_rows(observed) == _message_rows(expected)
    assert _dependency_rows(observed) == _dependency_rows(expected)
    assert observed_frontiers == expected_frontiers
    assert observed.render() == expected.render()

    assert len(plan.rounds) == 1
    assert plan.rounds[0].tag == BASE_TAG
    assert len(plan.extents) == len(pairs)
    assert sum(extent.payload_bytes for extent in plan.extents) == sum(
        size for _, _, size in pairs
    )
    assert tuple(rank for rank, _ in plan.entry_action_ids) == ranks
    assert tuple(rank for rank, _ in plan.terminal_action_ids) == ranks

    sources = {extent.source_rank for extent in plan.extents}
    destinations = {extent.destination_rank for extent in plan.extents}
    if case == "dispatch":
        assert sources == {0}
        assert destinations == {8, 16}
    elif case == "combine":
        assert sources == {8, 16}
        assert destinations == {0}
    else:
        assert not plan.extents
        assert not plan.actions
        assert not observed.messages
        if exact_frontier:
            assert set(observed_frontiers) == set(ranks)


def test_ring_and_sparse_tags_follow_the_accepted_block_order() -> None:
    graph = _graph(
        _ring_operation(RANKS_BY_WORLD[4], 4_096),
        _sparse_operation(SPARSE_PAIRS["dispatch"], operation_id="dispatch"),
        _sparse_operation(SPARSE_PAIRS["combine"], operation_id="combine"),
        _sparse_operation(SPARSE_PAIRS["all-local"], operation_id="all-local"),
    )
    planned = plan_execution_graph_collectives(graph)

    assert collective_goal_tags(graph) == collective_goal_tags(planned)
    assert {
        plan.operation_id: tuple(round_.tag for round_ in plan.rounds)
        for plan in planned.collective_plans
    } == {
        "ring": tuple(range(1_000, 1_006)),
        "dispatch": (1_006,),
        "combine": (1_007,),
        "all-local": (1_008,),
    }


# --- runtime consumes the plan instead of reconstructing it ------------------


@pytest.mark.parametrize("rate_bps", RATES_BPS)
@pytest.mark.parametrize(
    "operation",
    [
        _ring_operation(RANKS_BY_WORLD[2], 4),
        _ring_operation(RANKS_BY_WORLD[2], 4_096),
        _ring_operation(RANKS_BY_WORLD[4], 4),
        _ring_operation(RANKS_BY_WORLD[4], 4_096),
        _sparse_operation(SPARSE_PAIRS["dispatch"], request_pair_payload_bytes=DISPATCH_REQUESTS),
        _sparse_operation(SPARSE_PAIRS["combine"]),
    ],
    ids=["ring-2-4", "ring-2-4096", "ring-4-4", "ring-4-4096", "dispatch", "combine"],
)
def test_explicit_plan_and_absent_plan_arms_are_timing_identical(
    operation: ExecutionOperation, rate_bps: int
) -> None:
    graph = _graph(operation)
    planned = plan_execution_graph_collectives(graph)

    # A nonzero channel service would expose a per-round channel resource split.
    for channel_service_ps in (0, 7_000):
        absent = _execute(graph, rate_bps, channel_service_ps)
        explicit = _execute(planned, rate_bps, channel_service_ps)
        assert explicit == absent


@pytest.mark.parametrize("case", sorted(SPARSE_PAIRS))
@pytest.mark.parametrize("world", sorted(RANKS_BY_WORLD))
def test_runtime_work_requests_equal_the_declared_extents(
    case: str, world: int
) -> None:
    del world
    pairs = SPARSE_PAIRS[case]
    requests = DISPATCH_REQUESTS if case == "dispatch" else ()
    graph = _graph(_sparse_operation(pairs, request_pair_payload_bytes=requests))
    planned = plan_execution_graph_collectives(graph)
    plan = planned.collective_plans[0]
    tag_by_round = {round_.round_index: round_.tag for round_ in plan.rounds}

    _, _, _, wqes = _execute(planned, 400_000_000_000)

    assert [row[:5] for row in wqes] == [
        (
            plan.operation_id,
            extent.source_rank,
            extent.destination_rank,
            extent.payload_bytes,
            tag_by_round[extent.round_index],
        )
        for extent in plan.extents
    ]


def test_empty_semantic_collective_keeps_every_frontier_and_creates_no_traffic() -> None:
    graph = _graph(_sparse_operation(SPARSE_PAIRS["all-local"]))
    planned = plan_execution_graph_collectives(graph)
    plan = planned.collective_plans[0]

    with pytest.raises(ValueError, match="requires a nonzero payload"):
        _execute(graph, 400_000_000_000)

    result, _, _, wqes = _execute(planned, 400_000_000_000)

    assert not wqes
    assert result.completed_at_ps == 0
    assert len(plan.rounds) == 1
    assert tuple(rank for rank, action_ids in plan.terminal_action_ids) == (0, 8, 16, 24)
    assert all(not action_ids for _, action_ids in plan.terminal_action_ids)

    trace = GoalTrace(GOAL_RANKS)
    frontiers = render_collective_plan(trace, plan, exact_frontier=True)
    assert not trace.messages
    assert set(frontiers) == {0, 8, 16, 24}
    assert {operation.rank for operation in trace.operations} == {0, 8, 16, 24}


def test_explicit_plan_carries_a_sub_chunk_ring_the_surrogate_rejects() -> None:
    ranks = RANKS_BY_WORLD[4]
    graph = _graph(_ring_operation(ranks, 3))
    planned = plan_execution_graph_collectives(graph)

    for rate_bps, expected_ps in ((400_000_000_000, 120), (200_000_000_000, 240)):
        with pytest.raises(ValueError, match="at least one byte per rank"):
            _execute(graph, rate_bps)
        result, _, _, wqes = _execute(planned, rate_bps)
        assert result.completed_at_ps == expected_ps
        assert len(wqes) == 24
        assert {row[3] for row in wqes} == {1}


@pytest.mark.parametrize("rate_bps", RATES_BPS)
def test_explicit_plan_reaches_ttft_and_tpot_through_the_reported_chain(
    rate_bps: int,
) -> None:
    expected_ps = {400_000_000_000: 120, 200_000_000_000: 240}[rate_bps]
    clock = VirtualClock(0)
    reducer = CompletionReducer(clock)
    runtime = CoarseDeviceRuntime(CoarseDeviceProfile(rnic_rate_bps=rate_bps))
    ttft_ps = None
    tpot_ps = None
    for step_index in range(3):
        record = StepRecord(
            step_index,
            clock.now_ps,
            [
                ScheduledRequest(
                    "request",
                    RequestPhase.PREFILL if step_index == 0 else RequestPhase.DECODE,
                    1,
                    context_length=step_index + 1,
                )
            ],
            num_sampled=1,
            sampled_request_ids=["request"],
        )
        graph = plan_execution_graph_collectives(
            ExecutionGraph(
                f"step-{step_index}",
                step_index,
                record.virtual_time_ps,
                (_ring_operation(RANKS_BY_WORLD[4], 3, operation_id="ring"),),
                ("ring",),
            )
        )
        streamed: list = []
        result = runtime.execute(graph, on_event=streamed.append)
        report = runtime.last_report
        assert report is not None
        assert tuple(streamed) == result.events
        step = reducer.reduce(record, graph, result, report)
        metric = step.request_metrics[0]
        assert step.step_latency_ps == expected_ps
        if metric.ttft_ps is not None:
            ttft_ps = metric.ttft_ps
        if metric.tpot_ps is not None:
            tpot_ps = metric.tpot_ps

    assert ttft_ps == expected_ps
    assert tpot_ps == expected_ps


# --- the perturbations a runtime-side reconstruction would absorb ------------


def _perturbed_tag_graph() -> ExecutionGraph:
    planned = plan_execution_graph_collectives(_graph(_ring_operation(RANKS_BY_WORLD[4], 4)))
    plan = planned.collective_plans[0]
    rounds = list(plan.rounds)
    rounds[0] = replace(rounds[0], tag=rounds[0].tag + 500)
    return replace(planned, collective_plans=(replace(plan, rounds=tuple(rounds)),))


def _perturbed_rank_order_graph() -> ExecutionGraph:
    planned = plan_execution_graph_collectives(_graph(_ring_operation(RANKS_BY_WORLD[4], 4)))
    operation = planned.operations[0]
    reordered = replace(operation, work=replace(operation.work, ranks=(0, 16, 8, 24)))
    return replace(planned, operations=(reordered,))


@pytest.mark.parametrize(
    "build, message",
    [
        (_perturbed_tag_graph, "collective plan integrity mismatch"),
        (_perturbed_rank_order_graph, "rank order disagrees with semantic work"),
    ],
    ids=["tag", "rank-order"],
)
def test_byte_conserving_perturbations_are_rejected_before_any_work_request(
    build, message: str
) -> None:
    graph = build()

    with pytest.raises(ValueError, match=message):
        validate_execution_graph(graph)

    runtime = CoarseDeviceRuntime(CoarseDeviceProfile())
    with pytest.raises(ValueError, match=message):
        runtime.execute(graph)
    assert not runtime.bypass_ledger.records


def test_the_surrogate_absorbs_the_rank_order_perturbation_it_should_reject() -> None:
    """The negative control: without a plan the runtime cannot see the change."""

    graph = _graph(_ring_operation(RANKS_BY_WORLD[4], 4))
    operation = graph.operations[0]
    reordered = replace(
        graph,
        operations=(replace(operation, work=replace(operation.work, ranks=(0, 16, 8, 24))),),
    )

    baseline = _execute(graph, 400_000_000_000)
    absorbed = _execute(reordered, 400_000_000_000)

    assert baseline[0].completed_at_ps == absorbed[0].completed_at_ps
    assert sum(row[3] for row in baseline[3]) == sum(row[3] for row in absorbed[3])


def test_a_recomputed_identity_still_cannot_hide_a_semantic_disagreement() -> None:
    planned = plan_execution_graph_collectives(_graph(_ring_operation(RANKS_BY_WORLD[4], 4)))
    plan = planned.collective_plans[0]
    reordered = replace(plan, rank_order=(0, 16, 8, 24))
    resealed = replace(
        reordered,
        integrity_sha256=collective_plan_integrity_sha256(
            replace(reordered, integrity_sha256="0" * 64)
        ),
    )

    with pytest.raises(ValueError, match="rank order disagrees with semantic work"):
        validate_execution_graph(replace(planned, collective_plans=(resealed,)))


def test_partial_plan_authority_is_not_representable() -> None:
    graph = _graph(
        _ring_operation(RANKS_BY_WORLD[2], 4),
        _sparse_operation(SPARSE_PAIRS["dispatch"], operation_id="dispatch"),
    )
    planned = plan_execution_graph_collectives(graph)
    assert len(planned.collective_plans) == 2

    half = replace(planned, collective_plans=planned.collective_plans[:1])
    with pytest.raises(ValueError, match="must cover every collective operation"):
        validate_execution_graph(half)


def test_extent_and_action_pairing_is_enforced() -> None:
    planned = plan_execution_graph_collectives(_graph(_ring_operation(RANKS_BY_WORLD[2], 4)))
    plan = planned.collective_plans[0]
    extents = list(plan.extents)
    extents[0] = replace(extents[0], payload_bytes=extents[0].payload_bytes + 1)
    with pytest.raises(ValueError, match="collective plan integrity mismatch"):
        validate_execution_graph(
            replace(planned, collective_plans=(replace(plan, extents=tuple(extents)),))
        )

    send_actions = [
        action
        for action in plan.actions
        if action.kind is CollectivePlanActionKind.SEND
    ]
    assert send_actions
    dropped = replace(
        plan,
        actions=tuple(action for action in plan.actions if action is not send_actions[0]),
    )
    with pytest.raises(ValueError, match="unknown action ID"):
        validate_execution_graph(replace(planned, collective_plans=(dropped,)))


@pytest.mark.parametrize(
    "collective",
    [
        _ring_operation(RANKS_BY_WORLD[4], 4_096),
        _sparse_operation(SPARSE_PAIRS["dispatch"], operation_id="a2av"),
        _sparse_operation(SPARSE_PAIRS["combine"], operation_id="a2av"),
        _sparse_operation(SPARSE_PAIRS["all-local"], operation_id="a2av"),
    ],
    ids=["ring", "dispatch", "combine", "all-local"],
)
def test_a_consumed_frontier_renders_the_same_goal_with_and_without_the_plan(
    collective: ExecutionOperation,
) -> None:
    """A successor forces the exact frontier, which is where the two expansions
    are easiest to drift apart."""

    successors = tuple(
        ExecutionOperation(
            f"post-{rank}",
            rank,
            f"cuda:{rank}:compute",
            ComputeWork("post", nominal_duration_ps=1_000),
            correlation=collective.correlation,
            participant_local_depends_on=(collective.operation_id,),
        )
        for rank in collective.work.ranks
    )
    graph = ExecutionGraph(
        "chained",
        0,
        0,
        (collective, *successors),
        tuple(operation.operation_id for operation in successors),
    )
    planned = plan_execution_graph_collectives(graph)

    absent_trace = render_serial_execution_graph_goal(graph, num_goal_ranks=GOAL_RANKS)
    plan_trace = render_serial_execution_graph_goal(planned, num_goal_ranks=GOAL_RANKS)

    assert plan_trace.render() == absent_trace.render()
    assert _message_rows(plan_trace) == _message_rows(absent_trace)
    assert _dependency_rows(plan_trace) == _dependency_rows(absent_trace)


# --- the absent-plan compatibility level ------------------------------------


def test_absent_plan_graph_keeps_the_frozen_v1_wire_bytes() -> None:
    graph = _graph(
        ExecutionOperation(
            "a2av",
            0,
            "cuda:0:nccl:ep",
            CollectiveWork("all-to-allv", (0, 1), 2048, "pairwise"),
        )
    )
    graph = replace(graph, execution_id="exec", step_index=3, released_at_ps=17)
    payload = execution_graph_to_json(graph)

    assert "collective_plans" not in payload
    wire = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    assert execution_graph_from_json(json.loads(wire)) == graph
    assert hashlib.sha256(wire).hexdigest() == hashlib.sha256(wire).hexdigest()


@pytest.mark.parametrize("case", sorted(SPARSE_PAIRS))
def test_planned_graphs_round_trip_exactly(case: str) -> None:
    requests = DISPATCH_REQUESTS if case == "dispatch" else ()
    planned = plan_execution_graph_collectives(
        _graph(
            _ring_operation(RANKS_BY_WORLD[4], 4_096),
            _sparse_operation(
                SPARSE_PAIRS[case],
                request_pair_payload_bytes=requests,
                operation_id="a2av",
            ),
        )
    )
    payload = execution_graph_to_json(planned)

    assert "collective_plans" in payload
    assert execution_graph_from_json(json.loads(json.dumps(payload))) == planned
    for plan in planned.collective_plans:
        assert isinstance(plan, CollectivePlan)
        assert collective_plan_integrity_sha256(plan) == plan.integrity_sha256


def test_planning_is_idempotent_and_leaves_plan_free_graphs_alone() -> None:
    graph = _graph(_ring_operation(RANKS_BY_WORLD[2], 4))
    once = plan_execution_graph_collectives(graph)
    assert plan_execution_graph_collectives(once) == once

    compute_only = ExecutionGraph("empty", 0, 0, (), ())
    assert plan_execution_graph_collectives(compute_only).collective_plans == ()


@pytest.mark.parametrize("case", sorted(SPARSE_PAIRS))
def test_locality_projection_is_unchanged_by_the_plan(case: str) -> None:
    for graph in (
        _graph(_ring_operation(RANKS_BY_WORLD[4], 4_096)),
        _graph(_sparse_operation(SPARSE_PAIRS[case])),
    ):
        assert plan_execution_graph_locality(
            plan_execution_graph_collectives(graph)
        ) == plan_execution_graph_locality(graph)


def test_ring_pattern_export_is_the_compared_authority() -> None:
    """The comparison above uses the shipped pattern, not a private copy."""

    assert ring_allreduce is ring_pattern
