"""TRAF-28 qualification: the traffic-owned collective plan as the lowering default.

The study exercises both shipped lowerers in their default and bypass modes,
replays the real Granite step for the live TTFT and TPOT arm, and demonstrates
that the coarse runtime's absent-plan reconstruction is unreachable once the
plan is the default.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

RATES_BPS = (200_000_000_000, 400_000_000_000)
TP_RANKS_BY_WIDTH = {2: (0, 8), 4: (0, 8, 16, 24)}
TP_WIDTHS = (2, 4)
LOWERING_PATHS = ("serial", "observed")
SENTINEL_PAYLOAD_BYTES = 4_096

LEGACY_WIRE_BYTES = 559
LEGACY_WIRE_SHA256 = (
    "f4a5a70f5bd4a0c2fed874baa88f3035266a54f386a59927e115872c2bcff0a3"
)
PERTURBED_RANK_ORDER = (0, 16, 8, 24)

FROZEN_RATIO_BAND = (1.95, 2.05)
FROZEN_SCORED_FAMILIES = 4
FROZEN_SCORED_INSTANCES = 20

EXPECTATIONS = "examples/collective_plan_default_v1/expectations.md"
REPO_ROOT = Path(__file__).resolve().parents[2]

GRANITE_LAYERS = 24
GRANITE_EP_WIDTH = 8
GRANITE_STEP_COUNT = 3


def serialization_ps(byte_count: int, rate_bps: int) -> int:
    return -(-byte_count * 8 * 1_000_000_000_000 // rate_bps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--granite-root", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def check_only(args: argparse.Namespace) -> None:
    """Validate the frozen registries and their arithmetic, and nothing else."""

    if RATES_BPS != (200_000_000_000, 400_000_000_000):
        raise AssertionError("rate registry drifted")
    if TP_WIDTHS != (2, 4):
        raise AssertionError("tensor-parallel width registry drifted")
    if TP_RANKS_BY_WIDTH != {2: (0, 8), 4: (0, 8, 16, 24)}:
        raise AssertionError("tensor-parallel rank registry drifted")
    if any(len(ranks) != width for width, ranks in TP_RANKS_BY_WIDTH.items()):
        raise AssertionError("tensor-parallel widths disagree with their rank tuples")
    if LOWERING_PATHS != ("serial", "observed"):
        raise AssertionError("lowering path registry drifted")
    if (LEGACY_WIRE_BYTES, len(LEGACY_WIRE_SHA256)) != (559, 64):
        raise AssertionError("legacy wire anchor drifted")
    if sorted(PERTURBED_RANK_ORDER) != sorted(TP_RANKS_BY_WIDTH[4]):
        raise AssertionError("the rank-order perturbation changes the participant set")
    if PERTURBED_RANK_ORDER == TP_RANKS_BY_WIDTH[4]:
        raise AssertionError("the rank-order perturbation changes nothing")
    if FROZEN_RATIO_BAND != (1.95, 2.05):
        raise AssertionError("inverse-rate band drifted")

    identity_instances = len(RATES_BPS) * len(TP_WIDTHS) * len(LOWERING_PATHS)
    perturbation_instances = 3 * len(LOWERING_PATHS)
    unreachability_instances = 2 * len(LOWERING_PATHS)
    live_instances = 2
    registered = (
        identity_instances
        + perturbation_instances
        + unreachability_instances
        + live_instances
    )
    if (FROZEN_SCORED_FAMILIES, registered) != (4, FROZEN_SCORED_INSTANCES):
        raise AssertionError("evidence accounting drifted")

    for rate in RATES_BPS:
        if serialization_ps(SENTINEL_PAYLOAD_BYTES, rate) <= 0:
            raise AssertionError("serialization arithmetic is degenerate")
    if serialization_ps(SENTINEL_PAYLOAD_BYTES, RATES_BPS[0]) != 2 * serialization_ps(
        SENTINEL_PAYLOAD_BYTES,
        RATES_BPS[1],
    ):
        raise AssertionError("serialization is not inverse in the rate")

    if not (REPO_ROOT / EXPECTATIONS).is_file():
        raise AssertionError("the frozen expectations record is missing")
    if any(not str(path) for path in (args.out, args.granite_root)):
        raise AssertionError("registered path argument is empty")
    print(
        "check-only validated the frozen collective-plan-default registries, "
        "perturbation family and evidence accounting; produced no artifacts"
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _validate_result_inputs(args: argparse.Namespace) -> dict[str, Any]:
    configured_root = os.environ.get("SIMLLM_WAVE10_RUN_ROOT")
    if not configured_root:
        raise RuntimeError(
            "SIMLLM_WAVE10_RUN_ROOT must name the external wave-10 run root"
        )
    run_root = Path(configured_root).resolve()
    try:
        args.out.resolve().relative_to(run_root)
    except ValueError as exc:
        raise ValueError(
            "study output must remain under SIMLLM_WAVE10_RUN_ROOT"
        ) from exc
    if args.out.exists():
        raise FileExistsError(f"study output already exists: {args.out}")
    steps = args.granite_root / "replay-400g" / "steps.jsonl"
    if not steps.is_file():
        raise FileNotFoundError(f"missing Granite steps: {steps}")
    import hashlib
    import subprocess

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "observed_simllm_commit": revision,
        "granite_steps_sha256": hashlib.sha256(steps.read_bytes()).hexdigest(),
        "granite_steps_bytes": steps.stat().st_size,
    }


# --- lowering ----------------------------------------------------------------


def _dims(num_layers: int, *, moe: bool):
    from simllm.compute import ModelDims

    if moe:
        return ModelDims(
            num_layers=num_layers,
            hidden_size=1_024,
            intermediate_size=512,
            num_heads=16,
            num_kv_heads=8,
            head_size=64,
            vocab_size=49_152,
            dtype_bytes=2,
            num_experts=32,
            top_k=8,
            moe_intermediate_size=512,
            local_num_experts=4,
        )
    return ModelDims(
        num_layers=num_layers,
        hidden_size=1_024,
        intermediate_size=2_048,
        num_heads=16,
        num_kv_heads=8,
        head_size=64,
        vocab_size=1_024,
        dtype_bytes=2,
    )


def _step_record(step_index: int, virtual_time_ps: int, tokens: int, *, prefill: bool):
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord

    return StepRecord(
        step_index=step_index,
        virtual_time_ps=virtual_time_ps,
        scheduled=[
            ScheduledRequest(
                "request",
                RequestPhase.PREFILL if prefill else RequestPhase.DECODE,
                tokens,
                context_length=tokens,
            )
        ],
        num_sampled=1,
        sampled_request_ids=["request"],
    )


def _lower(record, dims, tp_ranks, *, path: str, attach: bool, ep_ranks=None):
    """Lower one record through the requested shipped lowerer."""

    from simllm.backends import (
        ObservedStepLowerer,
        SerialStepLowerer,
        SerialStepLowererConfig,
    )

    config = SerialStepLowererConfig(
        dims=dims,
        tp_ranks=tp_ranks,
        ep_ranks=ep_ranks,
        attach_collective_plan=attach,
    )
    if path == "serial":
        return SerialStepLowerer(config).lower(record)
    lowerer = ObservedStepLowerer(config)
    baseline = SerialStepLowerer(
        SerialStepLowererConfig(
            dims=dims,
            tp_ranks=tp_ranks,
            ep_ranks=ep_ranks,
            attach_collective_plan=False,
        )
    ).lower(record)
    observations = _observations_from_graph(baseline)
    return lowerer.lower(record, observations)


def _observations_from_graph(graph):
    """Replay a serial graph as framework-neutral observations."""

    from simllm.core import ExecutionObservations

    return ExecutionObservations(
        operations=graph.operations,
        completion_operation_ids=graph.completion_operation_ids,
    )


def _execute(graph, rate_bps: int):
    from simllm.core import CoarseDeviceProfile, CoarseDeviceRuntime

    runtime = CoarseDeviceRuntime(CoarseDeviceProfile(rnic_rate_bps=rate_bps))
    events: list = []
    result = runtime.execute(graph, on_event=events.append)
    wqes = tuple(
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
    )
    return result, tuple(events), wqes


def _graph_bytes_and_load(graph) -> tuple[int, int, int]:
    """Return total directed bytes, peak full-duplex endpoint load and messages."""

    from simllm.core import CollectiveWork
    from simllm.traffic import collective_plan_by_operation

    plans = collective_plan_by_operation(graph)
    directed: list[tuple[int, int, int]] = []
    for operation in graph.operations:
        work = operation.work
        if not isinstance(work, CollectiveWork):
            continue
        plan = plans.get(operation.operation_id)
        if plan is None:
            continue
        for extent in plan.extents:
            directed.append(
                (extent.source_rank, extent.destination_rank, extent.payload_bytes)
            )
    loads: dict[int, list[int]] = {}
    for source, destination, payload in directed:
        loads.setdefault(source, [0, 0])[0] += payload
        loads.setdefault(destination, [0, 0])[1] += payload
    peak = max((max(pair) for pair in loads.values()), default=0)
    return sum(row[2] for row in directed), peak, len(directed)


def _compute_ps(graph) -> int:
    """Represented compute on the busiest rank of a lowered graph."""

    from simllm.core import ComputeWork

    per_rank: dict[int, int] = {}
    for operation in graph.operations:
        work = operation.work
        if isinstance(work, ComputeWork):
            per_rank[operation.rank] = (
                per_rank.get(operation.rank, 0) + work.nominal_duration_ps
            )
    return max(per_rank.values(), default=0)


# --- families ----------------------------------------------------------------


def _family_identity(output_dir: Path) -> list[dict[str, Any]]:
    """Family A: default and bypass runtime records are identical."""

    rows = []
    for width in TP_WIDTHS:
        tp_ranks = TP_RANKS_BY_WIDTH[width]
        dims = _dims(2, moe=False)
        record = _step_record(0, 0, 8, prefill=True)
        for path in LOWERING_PATHS:
            default_graph = _lower(record, dims, tp_ranks, path=path, attach=True)
            bypass_graph = _lower(record, dims, tp_ranks, path=path, attach=False)
            for rate in RATES_BPS:
                default_run = _execute(default_graph, rate)
                bypass_run = _execute(bypass_graph, rate)
                rows.append(
                    {
                        "tp_width": width,
                        "lowering_path": path,
                        "rate_bps": rate,
                        "completed_at_ps": default_run[0].completed_at_ps,
                        "bypass_completed_at_ps": bypass_run[0].completed_at_ps,
                        "quiesced_at_ps": default_run[0].quiesced_at_ps,
                        "events_identical": default_run[1] == bypass_run[1],
                        "wqes_identical": default_run[2] == bypass_run[2],
                        "passed": (
                            default_run[0].completed_at_ps
                            == bypass_run[0].completed_at_ps
                            and default_run[0].quiesced_at_ps
                            == bypass_run[0].quiesced_at_ps
                            and default_run[1] == bypass_run[1]
                            and default_run[2] == bypass_run[2]
                        ),
                    }
                )
    _write_json(output_dir / "family-identity.json", rows)
    return rows


def _family_perturbation(output_dir: Path) -> list[dict[str, Any]]:
    """Family B: byte-conserving perturbations the surrogate cannot see."""

    from dataclasses import replace

    from simllm.core import CollectiveWork, validate_execution_graph

    rows = []
    tp_ranks = TP_RANKS_BY_WIDTH[4]
    dims = _dims(1, moe=False)
    record = _step_record(0, 0, 8, prefill=True)
    for path in LOWERING_PATHS:
        planned = _lower(record, dims, tp_ranks, path=path, attach=True)
        bypass = _lower(record, dims, tp_ranks, path=path, attach=False)

        plan = planned.collective_plans[0]
        perturbed_round = replace(plan.rounds[0], tag=plan.rounds[0].tag + 500)
        perturbed_plan = replace(
            plan,
            rounds=(perturbed_round, *plan.rounds[1:]),
        )
        tag_graph = replace(
            planned,
            collective_plans=(perturbed_plan, *planned.collective_plans[1:]),
        )
        tag_error = None
        submitted = None
        try:
            validate_execution_graph(tag_graph)
        except ValueError as exc:
            tag_error = str(exc)
        if tag_error is None:
            try:
                _, _, wqes = _execute(tag_graph, RATES_BPS[1])
                submitted = len(wqes)
            except (ValueError, AssertionError) as exc:
                tag_error = str(exc)
                submitted = 0
        rows.append(
            {
                "family": "plan-tag",
                "lowering_path": path,
                "rejected": tag_error is not None,
                "work_requests_submitted": submitted,
                "error": tag_error,
                "passed": tag_error is not None and (submitted in (None, 0)),
            }
        )

        operations = []
        for operation in planned.operations:
            work = operation.work
            if isinstance(work, CollectiveWork) and work.ranks == tp_ranks:
                operations.append(
                    replace(operation, work=replace(work, ranks=PERTURBED_RANK_ORDER))
                )
            else:
                operations.append(operation)
        order_graph = replace(planned, operations=tuple(operations))
        order_error = None
        order_submitted = None
        try:
            validate_execution_graph(order_graph)
        except ValueError as exc:
            order_error = str(exc)
        if order_error is None:
            try:
                _, _, wqes = _execute(order_graph, RATES_BPS[1])
                order_submitted = len(wqes)
            except (ValueError, AssertionError) as exc:
                order_error = str(exc)
                order_submitted = 0
        rows.append(
            {
                "family": "plan-rank-order",
                "lowering_path": path,
                "rejected": order_error is not None,
                "work_requests_submitted": order_submitted,
                "error": order_error,
                "passed": order_error is not None and (order_submitted in (None, 0)),
            }
        )

        absorbed_operations = []
        for operation in bypass.operations:
            work = operation.work
            if isinstance(work, CollectiveWork) and work.ranks == tp_ranks:
                absorbed_operations.append(
                    replace(operation, work=replace(work, ranks=PERTURBED_RANK_ORDER))
                )
            else:
                absorbed_operations.append(operation)
        absorbed = replace(bypass, operations=tuple(absorbed_operations))
        baseline_run = _execute(bypass, RATES_BPS[1])
        absorbed_run = _execute(absorbed, RATES_BPS[1])
        baseline_bytes = sum(row[3] for row in baseline_run[2])
        absorbed_bytes = sum(row[3] for row in absorbed_run[2])
        rows.append(
            {
                "family": "surrogate-absorbs-rank-order",
                "lowering_path": path,
                "baseline_completed_at_ps": baseline_run[0].completed_at_ps,
                "absorbed_completed_at_ps": absorbed_run[0].completed_at_ps,
                "baseline_bytes": baseline_bytes,
                "absorbed_bytes": absorbed_bytes,
                "passed": (
                    baseline_bytes == absorbed_bytes
                    and baseline_run[0].completed_at_ps
                    == absorbed_run[0].completed_at_ps
                ),
            }
        )
    _write_json(output_dir / "family-perturbation.json", rows)
    return rows


def _family_unreachability(output_dir: Path) -> list[dict[str, Any]]:
    """Family C: the absent-plan reconstruction never runs on the default path."""

    from simllm.core import runtime as runtime_module

    original = runtime_module.CoarseDeviceRuntime._schedule_collective

    def sentinel(self, graph, operation, *args, **kwargs):
        plan = kwargs.get("plan")
        if plan is None:
            positional = list(args)
            plan = positional[4] if len(positional) > 4 else None
        if plan is None:
            raise RuntimeError("absent-plan collective reconstruction was reached")
        return original(self, graph, operation, *args, **kwargs)

    rows = []
    tp_ranks = TP_RANKS_BY_WIDTH[4]
    dims = _dims(2, moe=False)
    record = _step_record(0, 0, 8, prefill=True)
    runtime_module.CoarseDeviceRuntime._schedule_collective = sentinel
    try:
        for path in LOWERING_PATHS:
            default_graph = _lower(record, dims, tp_ranks, path=path, attach=True)
            bypass_graph = _lower(record, dims, tp_ranks, path=path, attach=False)
            default_error = None
            try:
                _execute(default_graph, RATES_BPS[1])
            except RuntimeError as exc:
                default_error = str(exc)
            bypass_error = None
            try:
                _execute(bypass_graph, RATES_BPS[1])
            except RuntimeError as exc:
                bypass_error = str(exc)
            rows.append(
                {
                    "family": "default-avoids-surrogate",
                    "lowering_path": path,
                    "error": default_error,
                    "passed": default_error is None,
                }
            )
            rows.append(
                {
                    "family": "bypass-reaches-surrogate",
                    "lowering_path": path,
                    "error": bypass_error,
                    "passed": bypass_error is not None,
                }
            )
    finally:
        runtime_module.CoarseDeviceRuntime._schedule_collective = original
    _write_json(output_dir / "family-unreachability.json", rows)
    return rows


def _live_arm(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    """The real replayed Granite step, driven to TTFT and TPOT."""

    from simllm.core import (
        CoarseDeviceProfile,
        CoarseDeviceRuntime,
        CompletionReducer,
        VirtualClock,
        step_records_from_jsonl,
    )

    records = step_records_from_jsonl(
        args.granite_root / "replay-400g" / "steps.jsonl"
    )
    replayed = [record for record in records if record.total_new_tokens > 0]
    replayed = replayed[:GRANITE_STEP_COUNT]
    dims = _dims(GRANITE_LAYERS, moe=True)
    tp_ranks = tuple(range(GRANITE_EP_WIDTH))

    cells: dict[str, Any] = {}
    for rate in RATES_BPS:
        for attach in (True, False):
            mode = "default" if attach else "bypass"
            clock = VirtualClock(0)
            reducer = CompletionReducer(clock)
            runtime = CoarseDeviceRuntime(CoarseDeviceProfile(rnic_rate_bps=rate))
            ttft_ps = None
            tpot_ps = None
            step_rows = []
            for index, source in enumerate(replayed):
                record = _step_record(
                    index,
                    clock.now_ps,
                    source.total_new_tokens,
                    prefill=index == 0,
                )
                graph = _lower(
                    record,
                    dims,
                    tp_ranks,
                    path="serial",
                    attach=attach,
                    ep_ranks=tp_ranks,
                )
                streamed: list = []
                result = runtime.execute(graph, on_event=streamed.append)
                if tuple(streamed) != result.events:
                    raise AssertionError("streamed events disagree with the record")
                step = reducer.reduce(record, graph, result, runtime.last_report)
                metric = step.request_metrics[0]
                if metric.ttft_ps is not None:
                    ttft_ps = int(metric.ttft_ps)
                if metric.tpot_ps is not None:
                    tpot_ps = int(metric.tpot_ps)
                directed_bytes, peak_load, message_count = _graph_bytes_and_load(
                    _lower(
                        record,
                        dims,
                        tp_ranks,
                        path="serial",
                        attach=True,
                        ep_ranks=tp_ranks,
                    )
                )
                compute_ps = _compute_ps(graph)
                step_rows.append(
                    {
                        "step_index": index,
                        "tokens": source.total_new_tokens,
                        "step_latency_ps": step.step_latency_ps,
                        "compute_ps": compute_ps,
                        "directed_bytes": directed_bytes,
                        "peak_endpoint_load_bytes": peak_load,
                        "message_count": message_count,
                        "floor_ps": compute_ps + serialization_ps(peak_load, rate),
                        "ceiling_ps": (
                            compute_ps
                            + serialization_ps(directed_bytes, rate)
                            + 1_000 * message_count
                        ),
                    }
                )
            cells[f"{mode}.{rate}"] = {
                "mode": mode,
                "rate_bps": rate,
                "ttft_ps": ttft_ps,
                "tpot_ps": tpot_ps,
                "steps": step_rows,
            }
    _write_json(output_dir / "live-arm.json", cells)
    return cells


def _family_live(cells: dict[str, Any]) -> list[dict[str, Any]]:
    """Family D: the network term doubles when the rate halves."""

    low, high = FROZEN_RATIO_BAND
    rows = []
    slow = cells[f"default.{RATES_BPS[0]}"]["steps"]
    fast = cells[f"default.{RATES_BPS[1]}"]["steps"]
    for label, index in (("prefill", 0), ("decode", 1)):
        if index >= len(slow) or index >= len(fast):
            rows.append({"step": label, "ratio": None, "passed": False})
            continue
        slow_network = slow[index]["step_latency_ps"] - slow[index]["compute_ps"]
        fast_network = fast[index]["step_latency_ps"] - fast[index]["compute_ps"]
        ratio = slow_network / fast_network if fast_network > 0 else None
        rows.append(
            {
                "step": label,
                "network_term_200g_ps": slow_network,
                "network_term_400g_ps": fast_network,
                "ratio": ratio,
                "expected_band": [low, high],
                "passed": ratio is not None and low <= ratio <= high,
            }
        )
    return rows


def _exact_oracles(output_dir: Path) -> dict[str, Any]:
    """Fatal-unscored: coverage, emptiness, difference, equivalence, integrity."""

    import hashlib
    from dataclasses import replace

    from simllm.core import (
        CollectiveWork,
        ExecutionGraph,
        ExecutionOperation,
        collective_plan_integrity_sha256,
        execution_graph_from_json,
        execution_graph_to_json,
        validate_execution_graph,
    )
    from simllm.traffic import plan_execution_graph_collectives

    checks: dict[str, Any] = {}
    coverage = []
    for width in TP_WIDTHS:
        tp_ranks = TP_RANKS_BY_WIDTH[width]
        dims = _dims(2, moe=False)
        record = _step_record(0, 0, 8, prefill=True)
        for path in LOWERING_PATHS:
            default_graph = _lower(record, dims, tp_ranks, path=path, attach=True)
            bypass_graph = _lower(record, dims, tp_ranks, path=path, attach=False)
            collectives = tuple(
                operation
                for operation in default_graph.operations
                if isinstance(operation.work, CollectiveWork)
            )
            validate_execution_graph(default_graph)
            payload = execution_graph_to_json(bypass_graph)
            coverage.append(
                {
                    "tp_width": width,
                    "lowering_path": path,
                    "collectives": len(collectives),
                    "plans": len(default_graph.collective_plans),
                    "coverage": len(default_graph.collective_plans) == len(collectives)
                    and len(collectives) > 0,
                    "bypass_empty": bypass_graph.collective_plans == (),
                    "bypass_field_absent": "collective_plans" not in payload,
                    "plan_is_only_difference": (
                        replace(default_graph, collective_plans=()) == bypass_graph
                    ),
                    "equivalent": (
                        plan_execution_graph_collectives(bypass_graph) == default_graph
                    ),
                    "idempotent": (
                        plan_execution_graph_collectives(default_graph) == default_graph
                    ),
                    "integrity": all(
                        collective_plan_integrity_sha256(plan) == plan.integrity_sha256
                        for plan in default_graph.collective_plans
                    ),
                }
            )
    checks["coverage"] = all(row["coverage"] for row in coverage)
    checks["bypass_empty"] = all(row["bypass_empty"] for row in coverage)
    checks["bypass_field_absent"] = all(row["bypass_field_absent"] for row in coverage)
    checks["plan_is_only_difference"] = all(
        row["plan_is_only_difference"] for row in coverage
    )
    checks["equivalent"] = all(row["equivalent"] for row in coverage)
    checks["idempotent"] = all(row["idempotent"] for row in coverage)
    checks["integrity"] = all(row["integrity"] for row in coverage)

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
    checks["legacy_wire_bytes"] = len(wire) == LEGACY_WIRE_BYTES
    checks["legacy_wire_sha256"] = (
        hashlib.sha256(wire).hexdigest() == LEGACY_WIRE_SHA256
    )
    checks["legacy_wire_field_absent"] = "collective_plans" not in payload
    checks["legacy_wire_round_trip"] = (
        execution_graph_from_json(json.loads(wire)) == wire_graph
    )
    _write_json(output_dir / "exact-oracles.json", {"coverage": coverage, "checks": checks})
    return {"coverage": coverage, "checks": checks, "all_passed": all(checks.values())}


def _bounds_rows(cells: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, cell in sorted(cells.items()):
        for step in cell["steps"]:
            rows.append(
                {
                    "cell": f"{key}.step-{step['step_index']}",
                    "step_latency_ps": step["step_latency_ps"],
                    "floor_ps": step["floor_ps"],
                    "ceiling_ps": step["ceiling_ps"],
                    "passed": step["floor_ps"]
                    <= step["step_latency_ps"]
                    <= step["ceiling_ps"],
                }
            )
    return rows


def _live_identity_rows(cells: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for rate in RATES_BPS:
        default = cells[f"default.{rate}"]
        bypass = cells[f"bypass.{rate}"]
        rows.append(
            {
                "rate_bps": rate,
                "ttft_ps": default["ttft_ps"],
                "bypass_ttft_ps": bypass["ttft_ps"],
                "tpot_ps": default["tpot_ps"],
                "bypass_tpot_ps": bypass["tpot_ps"],
                "step_latencies_identical": [
                    step["step_latency_ps"] for step in default["steps"]
                ]
                == [step["step_latency_ps"] for step in bypass["steps"]],
                "passed": (
                    default["ttft_ps"] == bypass["ttft_ps"]
                    and default["tpot_ps"] == bypass["tpot_ps"]
                    and [step["step_latency_ps"] for step in default["steps"]]
                    == [step["step_latency_ps"] for step in bypass["steps"]]
                ),
            }
        )
    return rows


def run_study(args: argparse.Namespace) -> None:
    inputs = _validate_result_inputs(args)
    output_dir = args.out
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "inputs.json", inputs)

    identity = _family_identity(output_dir)
    perturbation = _family_perturbation(output_dir)
    unreachability = _family_unreachability(output_dir)
    live_cells = _live_arm(args, output_dir)
    live = _family_live(live_cells)

    families = {
        "default_bypass_identity": identity,
        "perturbation_rejection": perturbation,
        "surrogate_unreachability": unreachability,
        "live_inverse_rate": live,
    }
    instances = [row for rows in families.values() for row in rows]
    family_results = {
        name: all(bool(row["passed"]) for row in rows) for name, rows in families.items()
    }

    exact = _exact_oracles(output_dir)
    bounds = _bounds_rows(live_cells)
    live_identity = _live_identity_rows(live_cells)
    fatal = {
        "exact_oracles": exact["all_passed"],
        "physical_bounds": all(row["passed"] for row in bounds),
        "live_metric_identity": all(row["passed"] for row in live_identity),
    }
    void = not all(fatal.values())
    summary = {
        "study": "collective_plan_default_v1",
        "expectations": EXPECTATIONS,
        "inputs": inputs,
        "families": families,
        "registered_family_classes": FROZEN_SCORED_FAMILIES,
        "registered_instances": FROZEN_SCORED_INSTANCES,
        "observed_instances": len(instances),
        "passed_family_classes": sum(family_results.values()),
        "passed_instances": sum(bool(row["passed"]) for row in instances),
        "family_results": family_results,
        "exact_oracles": exact,
        "physical_bounds_rows": bounds,
        "live_metric_identity_rows": live_identity,
        "live_cells": live_cells,
        "fatal_guards": fatal,
        "void": void,
        "outcome": (
            "void"
            if void
            else (
                "passed"
                if all(bool(row["passed"]) for row in instances)
                else "failed"
            )
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "outcome": summary["outcome"],
                "void": void,
                "fatal_guards": fatal,
                "passed_instances": summary["passed_instances"],
                "registered_instances": FROZEN_SCORED_INSTANCES,
                "family_results": family_results,
            },
            indent=2,
        )
    )
    if void:
        raise SystemExit(2)
    if summary["passed_instances"] != len(instances):
        raise SystemExit(3)


def main() -> None:
    args = parse_args()
    if args.check_only:
        check_only(args)
        return
    run_study(args)


if __name__ == "__main__":
    main()
