"""Run the frozen TRAF-14 immutable collective-plan qualification."""

from __future__ import annotations

import argparse
from pathlib import Path

EVIDENCE_AUTHORED_AGAINST = "76223875557a552deb5aa2c2c529a07f000135ba"
BASE_TAG = 1_000
RATES_BPS = (200_000_000_000, 400_000_000_000)
RANKS_BY_WORLD = {
    2: (0, 8),
    4: (0, 8, 16, 24),
}
PAYLOAD_BYTES = (3, 4, 4_096)
SPARSE_CASES = {
    "dispatch": ((0, 8, 3), (0, 16, 5)),
    "combine": ((8, 0, 3), (16, 0, 5)),
    "all-local": (),
}
EXPECTED_RING_ROWS = {
    (2, 3): (2, 1, 4, 4, 4),
    (2, 4): (2, 2, 4, 8, 4),
    (2, 4_096): (2, 2_048, 4, 8_192, 4),
    (4, 3): (6, 1, 24, 24, 40),
    (4, 4): (6, 1, 24, 24, 40),
    (4, 4_096): (6, 1_024, 24, 24_576, 40),
}
EXPECTED_SENTINEL_METRICS_PS = {
    200_000_000_000: 240,
    400_000_000_000: 120,
}
EXPECTED_LARGE_RING_BOUNDS_PS = {
    (2, 200_000_000_000): (163_840, 327_680),
    (2, 400_000_000_000): (81_920, 163_840),
    (4, 200_000_000_000): (245_760, 983_040),
    (4, 400_000_000_000): (122_880, 491_520),
}
LEGACY_WIRE_BYTES = 559
LEGACY_WIRE_SHA256 = (
    "f4a5a70f5bd4a0c2fed874baa88f3035266a54f386a59927e115872c2bcff0a3"
)
EXPECTED_BEHAVIORAL_FAMILIES = 2
EXPECTED_BEHAVIORAL_INSTANCES = 6


def _serialization_ps(byte_count: int, rate_bps: int) -> int:
    numerator = byte_count * 8 * 1_000_000_000_000
    return (numerator + rate_bps - 1) // rate_bps


def _check_frozen_registry() -> None:
    expected_keys = {
        (world, payload)
        for world in RANKS_BY_WORLD
        for payload in PAYLOAD_BYTES
    }
    if set(EXPECTED_RING_ROWS) != expected_keys:
        raise AssertionError("ring sweep is incomplete")
    if set(EXPECTED_SENTINEL_METRICS_PS) != set(RATES_BPS):
        raise AssertionError("sentinel rate sweep is incomplete")
    if set(SPARSE_CASES) != {"dispatch", "combine", "all-local"}:
        raise AssertionError("sparse case registry is incomplete")
    if sum(size for _, _, size in SPARSE_CASES["dispatch"]) != 8:
        raise AssertionError("dispatch byte arithmetic drifted")
    if sum(size for _, _, size in SPARSE_CASES["combine"]) != 8:
        raise AssertionError("combine byte arithmetic drifted")
    if SPARSE_CASES["all-local"]:
        raise AssertionError("all-local semantic collective gained traffic")

    for (world, payload), row in EXPECTED_RING_ROWS.items():
        rounds, chunk, messages, directed_bytes, dependencies = row
        if rounds != 2 * (world - 1):
            raise AssertionError("ring round arithmetic drifted")
        if chunk != max(1, payload // world):
            raise AssertionError("ring chunk arithmetic drifted")
        if messages != world * rounds:
            raise AssertionError("ring message arithmetic drifted")
        if directed_bytes != messages * chunk:
            raise AssertionError("ring byte arithmetic drifted")
        if dependencies != 2 * world * (rounds - 1):
            raise AssertionError("ring dependency arithmetic drifted")

    for rate_bps, predicted_ps in EXPECTED_SENTINEL_METRICS_PS.items():
        floor_ps = 6 * _serialization_ps(1, rate_bps)
        ceiling_ps = 24 * _serialization_ps(1, rate_bps)
        if predicted_ps != floor_ps or not floor_ps <= predicted_ps <= ceiling_ps:
            raise AssertionError("sentinel physical bounds drifted")

    for (world, rate_bps), (floor_ps, ceiling_ps) in (
        EXPECTED_LARGE_RING_BOUNDS_PS.items()
    ):
        rounds, chunk, messages, _, _ = EXPECTED_RING_ROWS[(world, 4_096)]
        if floor_ps != rounds * _serialization_ps(chunk, rate_bps):
            raise AssertionError("large-ring floor drifted")
        if ceiling_ps != messages * _serialization_ps(chunk, rate_bps):
            raise AssertionError("large-ring ceiling drifted")
        if floor_ps > ceiling_ps:
            raise AssertionError("large-ring physical interval is empty")

    if EXPECTED_SENTINEL_METRICS_PS[200_000_000_000] != (
        2 * EXPECTED_SENTINEL_METRICS_PS[400_000_000_000]
    ):
        raise AssertionError("sentinel inverse-rate relation drifted")
    if len(LEGACY_WIRE_SHA256) != 64 or LEGACY_WIRE_BYTES <= 0:
        raise AssertionError("legacy wire oracle is malformed")
    if EXPECTED_BEHAVIORAL_FAMILIES != 2:
        raise AssertionError("behavioral family count drifted")
    if EXPECTED_BEHAVIORAL_INSTANCES != 6:
        raise AssertionError("behavioral instance count drifted")


def check_only(args: argparse.Namespace) -> None:
    _check_frozen_registry()
    print(
        f"check-only out={args.out}; validated frozen literals and produced no artifacts"
    )


# --- production run ---------------------------------------------------------
#
# Everything above this line is the frozen expectation. Everything below reads
# the implementation and reports what it observed. The evaluation order is the
# registered one: the two genuine-risk families are scored from raw validation
# and runtime observations first, and only then are the exact plan, GOAL, wire
# and timing oracles checked as fatal-unscored guards.


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _observed_provenance() -> dict:
    import subprocess

    def _git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=_repository_root(),
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip() or None

    return {
        "evidence_authored_against": EVIDENCE_AUTHORED_AGAINST,
        "observed_revision": _git("rev-parse", "HEAD"),
        "observed_htsim_gitlink": _git("rev-parse", "HEAD:third_party/htsim"),
        "note": (
            "the authored and observed revisions are recorded separately; no "
            "equality between them or a live submodule pin is assumed"
        ),
    }


def _ring_operation(ranks, payload_bytes, operation_id="ring"):
    from simllm.core import CollectiveWork, ExecutionOperation, OperationCorrelation

    return ExecutionOperation(
        operation_id,
        ranks[0],
        "cuda:0:nccl:tp",
        CollectiveWork("all-reduce", ranks, payload_bytes, "ring", channel_hint="tp"),
        correlation=OperationCorrelation(
            request_ids=("request",),
            batch_id="batch",
            layer=0,
        ),
    )


def _sparse_operation(pairs, operation_id="a2av", requests=()):
    from simllm.core import CollectiveWork, ExecutionOperation, OperationCorrelation

    ranks = RANKS_BY_WORLD[4]
    request_ids = tuple(sorted({entry[0] for entry in requests}))
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
            pair_payload_bytes=tuple(pairs),
            request_pair_payload_bytes=tuple(requests),
        ),
        correlation=OperationCorrelation(
            request_ids=request_ids or ("request",),
            batch_id="batch",
            layer=0,
        ),
    )


def _graph(*operations):
    from simllm.core import ExecutionGraph

    return ExecutionGraph(
        "collective-plan-v1",
        0,
        0,
        tuple(operations),
        tuple(operation.operation_id for operation in operations),
    )


def _execute(graph, rate_bps, channel_service_ps=0):
    from simllm.core import CoarseDeviceProfile, CoarseDeviceRuntime

    runtime = CoarseDeviceRuntime(
        CoarseDeviceProfile(
            rnic_rate_bps=rate_bps,
            nccl_channel_service_ps=channel_service_ps,
        )
    )
    events: list = []
    result = runtime.execute(graph, on_event=events.append)
    wqes = [
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
    return result, runtime.last_report, tuple(events), wqes


def _live_metrics(rate_bps, *, planned):
    """Drive one prefill and two decode steps to TTFT and TPOT."""

    from simllm.core import (
        CoarseDeviceProfile,
        CoarseDeviceRuntime,
        CompletionReducer,
        ExecutionGraph,
        RequestPhase,
        ScheduledRequest,
        StepRecord,
        VirtualClock,
    )
    from simllm.traffic import plan_execution_graph_collectives

    clock = VirtualClock(0)
    reducer = CompletionReducer(clock)
    runtime = CoarseDeviceRuntime(CoarseDeviceProfile(rnic_rate_bps=rate_bps))
    ranks = RANKS_BY_WORLD[4]
    ttft_ps = None
    tpot_ps = None
    step_latencies = []
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
        graph = ExecutionGraph(
            f"step-{step_index}",
            step_index,
            record.virtual_time_ps,
            (_ring_operation(ranks, 3),),
            ("ring",),
        )
        if planned:
            graph = plan_execution_graph_collectives(graph)
        streamed: list = []
        result = runtime.execute(graph, on_event=streamed.append)
        report = runtime.last_report
        if tuple(streamed) != result.events:
            raise AssertionError("streamed events disagree with the result record")
        step = reducer.reduce(record, graph, result, report)
        metric = step.request_metrics[0]
        step_latencies.append(step.step_latency_ps)
        if metric.ttft_ps is not None:
            ttft_ps = metric.ttft_ps
        if metric.tpot_ps is not None:
            tpot_ps = int(metric.tpot_ps)
    return ttft_ps, tpot_ps, tuple(step_latencies)


def _observe_perturbations() -> list[dict]:
    """Family one: a byte-conserving change the surrogate cannot see."""

    from dataclasses import replace

    from simllm.core import (
        CoarseDeviceProfile,
        CoarseDeviceRuntime,
        validate_execution_graph,
    )
    from simllm.traffic import plan_execution_graph_collectives

    ranks = RANKS_BY_WORLD[4]
    base = _graph(_ring_operation(ranks, 4))
    planned = plan_execution_graph_collectives(base)
    plan = planned.collective_plans[0]

    rounds = list(plan.rounds)
    rounds[0] = replace(rounds[0], tag=rounds[0].tag + 500)
    changed_tag = replace(
        planned,
        collective_plans=(replace(plan, rounds=tuple(rounds)),),
    )

    operation = planned.operations[0]
    changed_order = replace(
        planned,
        operations=(
            replace(operation, work=replace(operation.work, ranks=(0, 16, 8, 24))),
        ),
    )

    instances = []
    for name, graph, conserved_bytes in (
        ("plan-tag", changed_tag, sum(e.payload_bytes for e in plan.extents)),
        ("rank-order", changed_order, sum(e.payload_bytes for e in plan.extents)),
    ):
        validation_error = None
        try:
            validate_execution_graph(graph)
        except ValueError as exc:
            validation_error = str(exc)
        runtime = CoarseDeviceRuntime(CoarseDeviceProfile())
        runtime_error = None
        try:
            runtime.execute(graph)
        except ValueError as exc:
            runtime_error = str(exc)
        instances.append(
            {
                "instance": name,
                "total_bytes_conserved": conserved_bytes,
                "validation_error": validation_error,
                "runtime_error": runtime_error,
                "work_requests_submitted": len(runtime.bypass_ledger.records),
                "passed": bool(validation_error)
                and bool(runtime_error)
                and not runtime.bypass_ledger.records,
            }
        )

    # Negative control: the surrogate absorbs the same rank-order change.
    surrogate_operation = base.operations[0]
    surrogate = replace(
        base,
        operations=(
            replace(
                surrogate_operation,
                work=replace(surrogate_operation.work, ranks=(0, 16, 8, 24)),
            ),
        ),
    )
    baseline_result, _, _, baseline_wqes = _execute(base, 400_000_000_000)
    absorbed_result, _, _, absorbed_wqes = _execute(surrogate, 400_000_000_000)
    control = {
        "instance": "surrogate-control",
        "scored": False,
        "baseline_completed_at_ps": baseline_result.completed_at_ps,
        "reordered_completed_at_ps": absorbed_result.completed_at_ps,
        "baseline_bytes": sum(row[3] for row in baseline_wqes),
        "reordered_bytes": sum(row[3] for row in absorbed_wqes),
        "absorbed_silently": (
            baseline_result.completed_at_ps == absorbed_result.completed_at_ps
            and sum(row[3] for row in baseline_wqes)
            == sum(row[3] for row in absorbed_wqes)
        ),
    }
    return instances, control


def _observe_live_metrics() -> list[dict]:
    """Family two: the plan reaches TTFT and TPOT, the surrogate cannot."""

    instances = []
    for rate_bps in sorted(RATES_BPS):
        expected_ps = EXPECTED_SENTINEL_METRICS_PS[rate_bps]
        ttft_ps, tpot_ps, step_latencies = _live_metrics(rate_bps, planned=True)
        legacy_error = None
        try:
            _live_metrics(rate_bps, planned=False)
        except ValueError as exc:
            legacy_error = str(exc)
        instances.append(
            {
                "instance": f"ttft@{rate_bps}",
                "expected_ps": expected_ps,
                "observed_ps": ttft_ps,
                "passed": ttft_ps == expected_ps,
            }
        )
        instances.append(
            {
                "instance": f"tpot@{rate_bps}",
                "expected_ps": expected_ps,
                "observed_ps": tpot_ps,
                "passed": tpot_ps == expected_ps,
                "step_latencies_ps": list(step_latencies),
                "legacy_absent_plan_error": legacy_error,
            }
        )
    return instances


def _plan_versus_pattern_rows() -> list[dict]:
    """Fatal-unscored: every plan row equals the accepted pattern expansion."""

    from simllm.goal import GoalTrace
    from simllm.traffic import (
        pairwise_all_to_allv,
        plan_execution_graph_collectives,
        render_collective_plan,
        ring_allreduce,
    )

    rows = []
    for world, ranks in sorted(RANKS_BY_WORLD.items()):
        for payload_bytes in PAYLOAD_BYTES:
            planned = plan_execution_graph_collectives(
                _graph(_ring_operation(ranks, payload_bytes))
            )
            plan = planned.collective_plans[0]
            for exact_frontier in (False, True):
                expected = GoalTrace(32)
                expected_frontiers = ring_allreduce(
                    expected,
                    list(ranks),
                    payload_bytes,
                    BASE_TAG,
                    operation_id="ring",
                    exact_frontier=exact_frontier,
                )
                observed = GoalTrace(32)
                observed_frontiers = render_collective_plan(
                    observed,
                    plan,
                    exact_frontier=exact_frontier,
                )
                rounds, chunk, messages, directed, dependencies = EXPECTED_RING_ROWS[
                    (world, payload_bytes)
                ]
                rows.append(
                    {
                        "case": f"ring:W={world}:payload={payload_bytes}",
                        "exact_frontier": exact_frontier,
                        "goal_identical": observed.render() == expected.render(),
                        "frontiers_identical": (
                            observed_frontiers == expected_frontiers
                        ),
                        "rounds": len(plan.rounds),
                        "expected_rounds": rounds,
                        "chunk_bytes": sorted(
                            {extent.payload_bytes for extent in plan.extents}
                        ),
                        "expected_chunk_bytes": chunk,
                        "messages": len(plan.extents),
                        "expected_messages": messages,
                        "directed_bytes": sum(
                            extent.payload_bytes for extent in plan.extents
                        ),
                        "expected_directed_bytes": directed,
                        "internal_dependencies": sum(
                            len(action.depends_on) for action in plan.actions
                        ),
                        "expected_internal_dependencies": dependencies,
                        "tags": [round_.tag for round_ in plan.rounds],
                    }
                )

    for case, pairs in sorted(SPARSE_CASES.items()):
        requests = (
            (("alpha", 0, 8, 3), ("beta", 0, 16, 5)) if case == "dispatch" else ()
        )
        planned = plan_execution_graph_collectives(
            _graph(_sparse_operation(pairs, requests=requests))
        )
        plan = planned.collective_plans[0]
        ranks = RANKS_BY_WORLD[4]
        request_send_bytes = {
            (source, destination): tuple(
                (request_id, size)
                for request_id, entry_source, entry_destination, size in requests
                if (entry_source, entry_destination) == (source, destination)
            )
            for source, destination, _ in pairs
        }
        for exact_frontier in (False, True):
            expected = GoalTrace(32)
            expected_frontiers = pairwise_all_to_allv(
                expected,
                list(ranks),
                {(source, destination): size for source, destination, size in pairs},
                BASE_TAG,
                operation_id="a2av",
                exact_frontier=exact_frontier,
                request_send_bytes=request_send_bytes if requests else None,
            )
            observed = GoalTrace(32)
            observed_frontiers = render_collective_plan(
                observed,
                plan,
                exact_frontier=exact_frontier,
            )
            rows.append(
                {
                    "case": f"sparse:{case}",
                    "exact_frontier": exact_frontier,
                    "goal_identical": observed.render() == expected.render(),
                    "frontiers_identical": observed_frontiers == expected_frontiers,
                    "rounds": len(plan.rounds),
                    "expected_rounds": 1,
                    "messages": len(plan.extents),
                    "expected_messages": len(pairs),
                    "directed_bytes": sum(
                        extent.payload_bytes for extent in plan.extents
                    ),
                    "expected_directed_bytes": sum(size for _, _, size in pairs),
                    "sources": sorted(
                        {extent.source_rank for extent in plan.extents}
                    ),
                    "destinations": sorted(
                        {extent.destination_rank for extent in plan.extents}
                    ),
                    "frontier_ranks": sorted(observed_frontiers)
                    if exact_frontier
                    else None,
                    "tags": [round_.tag for round_ in plan.rounds],
                }
            )
    return rows


def _compatibility_rows() -> list[dict]:
    """Fatal-unscored: the absent-plan arm keeps its accepted behavior."""

    import hashlib
    import json

    from simllm.core import (
        CollectiveWork,
        ExecutionGraph,
        ExecutionOperation,
        execution_graph_from_json,
        execution_graph_to_json,
    )
    from simllm.traffic import plan_execution_graph_collectives

    rows = []
    for world, ranks in sorted(RANKS_BY_WORLD.items()):
        for payload_bytes in (4, 4_096):
            base = _graph(_ring_operation(ranks, payload_bytes))
            planned = plan_execution_graph_collectives(base)
            for rate_bps in sorted(RATES_BPS):
                # A nonzero channel service exposes a per-round resource split.
                for channel_service_ps in (0, 7_000):
                    absent = _execute(base, rate_bps, channel_service_ps)
                    explicit = _execute(planned, rate_bps, channel_service_ps)
                    rows.append(
                        {
                            "case": f"ring:W={world}:payload={payload_bytes}",
                            "rate_bps": rate_bps,
                            "channel_service_ps": channel_service_ps,
                            "identical": explicit == absent,
                            "completed_at_ps": absent[0].completed_at_ps,
                            "work_requests": len(absent[3]),
                        }
                    )

    # This is the exact graph the frozen v1 wire oracle names, reproduced from
    # tests/test_execution_io.py. Its identity fields are part of the oracle.
    wire_graph = ExecutionGraph(
        "core6-uniform",
        7,
        11,
        (
            ExecutionOperation(
                "a2av",
                0,
                "cuda:0:nccl:ep",
                CollectiveWork("all-to-allv", (0, 1), 2048, "pairwise"),
            ),
        ),
        ("a2av",),
    )
    payload = execution_graph_to_json(wire_graph)
    wire = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    rows.append(
        {
            "case": "absent-plan-wire",
            "collective_plans_field_present": "collective_plans" in payload,
            "wire_bytes": len(wire),
            "expected_wire_bytes": LEGACY_WIRE_BYTES,
            "wire_sha256": hashlib.sha256(wire).hexdigest(),
            "expected_wire_sha256": LEGACY_WIRE_SHA256,
            "round_trips": execution_graph_from_json(json.loads(wire)) == wire_graph,
        }
    )
    return rows


def _integrity_rows() -> list[dict]:
    """Fatal-unscored: plan identity, coverage and lossless round trips."""

    import json
    from dataclasses import replace

    from simllm.core import (
        collective_plan_integrity_sha256,
        execution_graph_from_json,
        execution_graph_to_json,
        validate_execution_graph,
    )
    from simllm.traffic import plan_execution_graph_collectives

    combined = _graph(
        _ring_operation(RANKS_BY_WORLD[4], 4_096),
        _sparse_operation(SPARSE_CASES["dispatch"], operation_id="dispatch"),
        _sparse_operation(SPARSE_CASES["combine"], operation_id="combine"),
        _sparse_operation(SPARSE_CASES["all-local"], operation_id="all-local"),
    )
    planned = plan_execution_graph_collectives(combined)
    payload = execution_graph_to_json(planned)
    round_tripped = execution_graph_from_json(json.loads(json.dumps(payload)))

    coverage_error = None
    try:
        validate_execution_graph(
            replace(planned, collective_plans=planned.collective_plans[:1])
        )
    except ValueError as exc:
        coverage_error = str(exc)

    return [
        {
            "row": "integrity-identity",
            "sealed": all(
                collective_plan_integrity_sha256(plan) == plan.integrity_sha256
                for plan in planned.collective_plans
            ),
        },
        {
            "row": "plan-coverage",
            "plans": len(planned.collective_plans),
            "collectives": len(combined.operations),
            "partial_authority_rejected": coverage_error is not None,
            "partial_authority_error": coverage_error,
        },
        {
            "row": "round-trip",
            "lossless": round_tripped == planned,
            "field_emitted": "collective_plans" in payload,
        },
        {
            "row": "tag-block-order",
            "tags": {
                plan.operation_id: [round_.tag for round_ in plan.rounds]
                for plan in planned.collective_plans
            },
            "expected": {
                "ring": list(range(1_000, 1_006)),
                "dispatch": [1_006],
                "combine": [1_007],
                "all-local": [1_008],
            },
        },
        {
            "row": "idempotent",
            "stable": plan_execution_graph_collectives(planned) == planned,
        },
    ]


def _physical_rows() -> list[dict]:
    """Fatal-unscored: every measured value sits inside its own bounds."""

    from simllm.traffic import plan_execution_graph_collectives

    rows = []
    for rate_bps in sorted(RATES_BPS):
        floor_ps = 6 * _serialization_ps(1, rate_bps)
        ceiling_ps = 24 * _serialization_ps(1, rate_bps)
        planned = plan_execution_graph_collectives(
            _graph(_ring_operation(RANKS_BY_WORLD[4], 3))
        )
        result, _, _, wqes = _execute(planned, rate_bps)
        rows.append(
            {
                "case": f"sentinel-ring@{rate_bps}",
                "floor_ps": floor_ps,
                "observed_ps": result.completed_at_ps,
                "ceiling_ps": ceiling_ps,
                "inside": floor_ps <= result.completed_at_ps <= ceiling_ps,
                "work_requests": len(wqes),
            }
        )
        for world, ranks in sorted(RANKS_BY_WORLD.items()):
            floor_ps, ceiling_ps = EXPECTED_LARGE_RING_BOUNDS_PS[(world, rate_bps)]
            planned = plan_execution_graph_collectives(
                _graph(_ring_operation(ranks, 4_096))
            )
            result, _, _, _ = _execute(planned, rate_bps)
            rows.append(
                {
                    "case": f"large-ring:W={world}@{rate_bps}",
                    "floor_ps": floor_ps,
                    "observed_ps": result.completed_at_ps,
                    "ceiling_ps": ceiling_ps,
                    "inside": floor_ps <= result.completed_at_ps <= ceiling_ps,
                }
            )
        planned = plan_execution_graph_collectives(
            _graph(_sparse_operation(SPARSE_CASES["dispatch"]))
        )
        result, _, _, _ = _execute(planned, rate_bps)
        egress_floor_ps = sum(
            _serialization_ps(size, rate_bps) for _, _, size in SPARSE_CASES["dispatch"]
        )
        rows.append(
            {
                "case": f"dispatch@{rate_bps}",
                "floor_ps": egress_floor_ps,
                "observed_ps": result.completed_at_ps,
                "ceiling_ps": egress_floor_ps,
                "inside": result.completed_at_ps == egress_floor_ps,
            }
        )
        planned = plan_execution_graph_collectives(
            _graph(_sparse_operation(SPARSE_CASES["combine"]))
        )
        result, _, _, _ = _execute(planned, rate_bps)
        rows.append(
            {
                "case": f"combine@{rate_bps}",
                "structural_only": True,
                "note": (
                    "the coarse model has no destination-ingress serializer, so "
                    "a many-source combine time is not a physical oracle"
                ),
                "observed_ps": result.completed_at_ps,
                "inside": True,
            }
        )
    return rows


def _zero_work_rows() -> list[dict]:
    """Fatal-unscored: an idle rank stays idle and gains no invented traffic."""

    from simllm.goal import GoalTrace
    from simllm.traffic import plan_execution_graph_collectives, render_collective_plan

    planned = plan_execution_graph_collectives(
        _graph(_sparse_operation(SPARSE_CASES["all-local"]))
    )
    plan = planned.collective_plans[0]
    trace = GoalTrace(32)
    frontiers = render_collective_plan(trace, plan, exact_frontier=True)
    result, _, _, wqes = _execute(planned, 400_000_000_000)

    legacy_error = None
    try:
        _execute(_graph(_sparse_operation(SPARSE_CASES["all-local"])), 400_000_000_000)
    except ValueError as exc:
        legacy_error = str(exc)

    return [
        {
            "row": "empty-semantic-collective",
            "semantic_rounds": len(plan.rounds),
            "extents": len(plan.extents),
            "work_requests": len(wqes),
            "goal_messages": len(trace.messages),
            "frontier_ranks": sorted(frontiers),
            "completed_at_ps": result.completed_at_ps,
            "legacy_absent_plan_error": legacy_error,
            "passed": (
                len(plan.rounds) == 1
                and not plan.extents
                and not wqes
                and not trace.messages
                and sorted(frontiers) == list(RANKS_BY_WORLD[4])
                and result.completed_at_ps == 0
            ),
        }
    ]


def _fatal_failures(summary: dict) -> list[str]:
    failures = []
    for row in summary["plan_versus_pattern"]:
        if not row["goal_identical"] or not row["frontiers_identical"]:
            failures.append(f"plan-versus-pattern:{row['case']}")
        if row["rounds"] != row["expected_rounds"]:
            failures.append(f"rounds:{row['case']}")
        if row["messages"] != row["expected_messages"]:
            failures.append(f"messages:{row['case']}")
        if row["directed_bytes"] != row["expected_directed_bytes"]:
            failures.append(f"bytes:{row['case']}")
        if row.get("expected_internal_dependencies") is not None and (
            row["internal_dependencies"] != row["expected_internal_dependencies"]
        ):
            failures.append(f"dependencies:{row['case']}")
    for row in summary["compatibility"]:
        if row["case"] == "absent-plan-wire":
            if row["collective_plans_field_present"]:
                failures.append("absent-plan-wire:field-emitted")
            if row["wire_bytes"] != row["expected_wire_bytes"]:
                failures.append("absent-plan-wire:bytes")
            if row["wire_sha256"] != row["expected_wire_sha256"]:
                failures.append("absent-plan-wire:sha256")
            if not row["round_trips"]:
                failures.append("absent-plan-wire:round-trip")
        elif not row["identical"]:
            failures.append(
                f"compatibility:{row['case']}@{row['rate_bps']}"
                f":service={row['channel_service_ps']}"
            )
    for row in summary["integrity"]:
        if row["row"] == "integrity-identity" and not row["sealed"]:
            failures.append("integrity-identity")
        if row["row"] == "plan-coverage" and not row["partial_authority_rejected"]:
            failures.append("plan-coverage")
        if row["row"] == "round-trip" and not (
            row["lossless"] and row["field_emitted"]
        ):
            failures.append("round-trip")
        if row["row"] == "tag-block-order" and row["tags"] != row["expected"]:
            failures.append("tag-block-order")
        if row["row"] == "idempotent" and not row["stable"]:
            failures.append("idempotent")
    for row in summary["physical"]:
        if not row["inside"]:
            failures.append(f"physical:{row['case']}")
    for row in summary["zero_work"]:
        if not row["passed"]:
            failures.append(f"zero-work:{row['row']}")
    control = summary["perturbation_control"]
    if not control["absorbed_silently"]:
        failures.append("perturbation-control:surrogate-did-not-absorb")
    scaling = summary["metric_scaling"]
    if not scaling["holds"]:
        failures.append("metric-scaling")
    return failures


def run(out: Path) -> int:
    import json

    _check_frozen_registry()

    # Family one and family two are read from raw observations first.
    perturbations, control = _observe_perturbations()
    live_metrics = _observe_live_metrics()

    observed_by_rate = {
        rate_bps: next(
            row["observed_ps"]
            for row in live_metrics
            if row["instance"] == f"ttft@{rate_bps}"
        )
        for rate_bps in RATES_BPS
    }
    metric_scaling = {
        "relation": "the 200 Gbit/s metric is exactly twice the 400 Gbit/s metric",
        "observed": observed_by_rate,
        "holds": observed_by_rate[200_000_000_000]
        == 2 * observed_by_rate[400_000_000_000],
        "scored": False,
        "note": "entailed by the four exact metric instances above, so unscored",
    }

    behavioral = [*perturbations, *live_metrics]
    summary = {
        "study": "collective_plan_v1",
        "provenance": _observed_provenance(),
        "evaluation_order": [
            "perturbation family from raw validation and runtime observations",
            "live metric family from raw runtime and reducer observations",
            "exact plan versus pattern rows",
            "compatibility, integrity, physical and zero-work guards",
        ],
        "behavioral": behavioral,
        "perturbation_control": control,
        "metric_scaling": metric_scaling,
        "plan_versus_pattern": _plan_versus_pattern_rows(),
        "compatibility": _compatibility_rows(),
        "integrity": _integrity_rows(),
        "physical": _physical_rows(),
        "zero_work": _zero_work_rows(),
    }
    failures = _fatal_failures(summary)
    passed = sum(1 for row in behavioral if row["passed"])
    summary["fatal_failures"] = failures
    summary["behavioral_families"] = EXPECTED_BEHAVIORAL_FAMILIES
    summary["behavioral_instances"] = len(behavioral)
    summary["behavioral_passed"] = None if failures else passed
    summary["void"] = bool(failures)

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    if len(behavioral) != EXPECTED_BEHAVIORAL_INSTANCES:
        raise AssertionError("behavioral instance count disagrees with the freeze")
    if failures:
        print(f"VOID: {len(failures)} fatal guards violated")
        for failure in failures:
            print(f"  fatal: {failure}")
        print("no behavioral fraction is reported for a void run")
        return 1
    print(
        f"behavioral: {passed} of {len(behavioral)} instances in "
        f"{EXPECTED_BEHAVIORAL_FAMILIES} families"
    )
    for row in behavioral:
        print(f"  {row['instance']}: {'pass' if row['passed'] else 'FAIL'}")
    print(f"summary written to {out / 'summary.json'}")
    return 0 if passed == len(behavioral) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_only:
        check_only(args)
        return
    raise SystemExit(run(args.out))


if __name__ == "__main__":
    main()
