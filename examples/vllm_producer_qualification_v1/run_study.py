"""Run the frozen VLLM-22 producer qualification."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path


def _load_overlap_study():
    path = (
        Path(__file__).resolve().parents[1]
        / "vllm_observed_overlap_v1"
        / "run_study.py"
    )
    spec = importlib.util.spec_from_file_location("vllm_observed_overlap_study", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load landed overlap study from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


overlap = _load_overlap_study()

SERIAL_REDUCTION_BANDS = {
    "single-node": (Fraction(1, 80), Fraction(33, 2_000)),
    "cross-node": (Fraction(21, 200), Fraction(1, 8)),
}
REDUCTION_RATIO_BAND = (Fraction(15, 2), Fraction(21, 2))
STRUCTURE_FRACTION_MAX = Fraction(1, 10_000)
EXPECTED_GENUINE_RISK_INSTANCES = 3
EXPECTATIONS_COMMIT = "6459c3c469bb4bbddb47fbfdd1f6ed5ba8b86949"
ORIGINAL_REQUEST_IDS = ("r0", "r1", "r2")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--replay-run", type=Path, required=True)
    parser.add_argument("--routed-experts", type=Path, required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--vllm-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def check_expectation_registry(args: argparse.Namespace) -> None:
    """Validate frozen inputs and literals without importing target code."""

    overlap.check_expectation_registry(args)
    expected = {
        "single-node": (Fraction(1, 80), Fraction(33, 2_000)),
        "cross-node": (Fraction(21, 200), Fraction(1, 8)),
    }
    if SERIAL_REDUCTION_BANDS != expected:
        raise SystemExit("serial-reduction bands drifted from the freeze")
    if not all(0 < low < high < 1 for low, high in expected.values()):
        raise SystemExit("serial-reduction bands must be positive proper fractions")
    if not REDUCTION_RATIO_BAND[0] < 9 < REDUCTION_RATIO_BAND[1]:
        raise SystemExit("rate-ratio band must contain the ninefold expectation")
    if STRUCTURE_FRACTION_MAX != Fraction(1, 10_000):
        raise SystemExit("structure bound drifted from 0.01 percent")
    if EXPECTED_GENUINE_RISK_INSTANCES != 3:
        raise SystemExit("genuine-risk denominator drifted")
    if overlap.COMPUTE_FLOOR_PS != 69_206_016:
        raise SystemExit("decode compute floor drifted")
    if overlap.DECODE_TPOT_CEILING_PS != 150_000_000:
        raise SystemExit("decode TPOT ceiling drifted")
    if overlap.PREFILL_TTFT_CEILING_PS != 500_000_000:
        raise SystemExit("prefill TTFT ceiling drifted")
    if overlap.COMM_CEILING_PS != {
        "single-node": 1_456_158,
        "cross-node": 13_105_420,
    }:
        raise SystemExit("communication ceilings drifted")
    mechanism_sources = {
        "v1/worker/gpu_ubatch_wrapper.py",
        "v1/worker/ubatching.py",
        "model_executor/layers/fused_moe/modular_kernel.py",
        "model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py",
    }
    if not mechanism_sources <= overlap.SOURCE_HASHES.keys():
        raise SystemExit("named concurrency source inventory is incomplete")


def _canonical_identity(value: object) -> dict[str, object]:
    wire = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return {
        "bytes": len(wire),
        "sha256": hashlib.sha256(wire).hexdigest(),
    }


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args_with_output(args: argparse.Namespace, output: Path) -> argparse.Namespace:
    copied = copy.copy(args)
    copied.output_dir = output
    return copied


def _run_disabled_driver(args: argparse.Namespace, output: Path) -> Path:
    repo = Path(__file__).resolve().parents[2]
    driver = (
        Path(__file__).resolve().parents[1]
        / "vllm_observed_schedule_v1"
        / "live_driver.py"
    )
    log_path = output / "live_driver.log"
    environment = dict(os.environ)
    previous_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(repo)
        if not previous_pythonpath
        else str(repo) + os.pathsep + previous_pythonpath
    )
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    command = [
        str(args.vllm_python),
        str(driver),
        "--capture",
        str(args.capture),
        "--replay-run",
        str(args.replay_run),
        "--routed-experts",
        str(args.routed_experts),
        "--output-dir",
        str(output),
        "--producer-mode",
        "disabled",
    ]
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        completed = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            "producer-disabled live vLLM driver failed with "
            f"{completed.returncode}; see {log_path}"
        )
    live_path = output / "live_observations.jsonl"
    if not live_path.is_file():
        raise RuntimeError("producer-disabled live driver produced no step stream")
    return live_path


def _fraction(value: dict[str, object]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def _evaluate_behavioral(
    modes: dict[str, dict[str, object]],
) -> dict[str, object]:
    rows = []
    passed = 0
    absolute_reductions: dict[str, Fraction] = {}
    for placement, band in SERIAL_REDUCTION_BANDS.items():
        serial = _fraction(modes[f"{placement}:serial"]["r0_tpot_ps"])
        observed = _fraction(modes[f"{placement}:observed"]["r0_tpot_ps"])
        absolute = serial - observed
        relative = absolute / serial
        absolute_reductions[placement] = absolute
        in_band = absolute > 0 and band[0] <= relative <= band[1]
        passed += int(in_band)
        rows.append(
            {
                "absolute_reduction_ps": overlap._fraction_to_json(absolute),
                "b1_enabled_reduction_passed": in_band,
                "observed_tpot_ps": overlap._fraction_to_json(observed),
                "placement": placement,
                "relative_reduction": float(relative),
                "relative_reduction_band": [float(bound) for bound in band],
                "serial_tpot_ps": overlap._fraction_to_json(serial),
            }
        )
    ratio = (
        absolute_reductions["cross-node"]
        / absolute_reductions["single-node"]
        if absolute_reductions["single-node"] > 0
        else None
    )
    ratio_passed = ratio is not None and (
        REDUCTION_RATIO_BAND[0] <= ratio <= REDUCTION_RATIO_BAND[1]
    )
    passed += int(ratio_passed)
    return {
        "b2_absolute_reduction_ratio": None if ratio is None else float(ratio),
        "b2_absolute_reduction_ratio_band": [
            float(bound) for bound in REDUCTION_RATIO_BAND
        ],
        "b2_absolute_reduction_ratio_passed": ratio_passed,
        "evaluation_order": "raw request metrics before every fatal guard",
        "genuine_risk_executed": EXPECTED_GENUINE_RISK_INSTANCES,
        "genuine_risk_passed": passed,
        "rows": rows,
    }


def _expected_operation_ids(step_index: int) -> tuple[str, ...]:
    microbatches = (
        (0, 1) if step_index in overlap.R0_DECODE_STEP_INDICES else (None,)
    )
    result = []
    for layer in range(overlap.NUM_LAYERS):
        for microbatch in microbatches:
            index = 0 if microbatch is None else microbatch
            result.extend(
                f"step-{step_index}:ubatch-{index}:layer-{layer}:rank-{rank}"
                ":pre-dispatch"
                for rank in range(overlap.DP_SIZE)
            )
        for microbatch in microbatches:
            index = 0 if microbatch is None else microbatch
            result.append(
                f"step-{step_index}:ubatch-{index}:layer-{layer}:ep-dispatch"
            )
            result.extend(
                f"step-{step_index}:ubatch-{index}:layer-{layer}:rank-{rank}:experts"
                for rank in range(overlap.DP_SIZE)
            )
        for microbatch in microbatches:
            index = 0 if microbatch is None else microbatch
            result.append(
                f"step-{step_index}:ubatch-{index}:layer-{layer}:ep-combine"
            )
    result.extend(
        f"step-{step_index}:rank-{rank}:logits"
        for rank in range(overlap.DP_SIZE)
    )
    result.extend(
        f"step-{step_index}:ubatch-{0 if microbatch is None else microbatch}"
        ":requests-visible"
        for microbatch in microbatches
    )
    return tuple(result)


def _source_inventory(live_path: Path) -> dict[str, object]:
    from simllm.core import (
        CollectiveWork,
        OperationCorrelation,
        execution_graph_from_json,
        step_record_from_json,
    )

    rows = [
        json.loads(line)
        for line in live_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    order_checks = []
    correlation_checks = []
    dependency_checks = []
    logical_stream_checks = []
    frontier_checks = []
    original_identity_checks = []
    for row in rows:
        record = step_record_from_json(row["record"])
        graph = execution_graph_from_json(row["source_graph"])
        operations = graph.operations
        order_checks.append(
            tuple(operation.operation_id for operation in operations)
            == _expected_operation_ids(record.step_index)
        )
        queues_valid = True
        for operation in operations:
            operation_id = operation.operation_id
            if isinstance(operation.work, CollectiveWork):
                expected_queue = "cuda:0:comm:ep"
            elif operation_id.endswith(":requests-visible"):
                expected_queue = (
                    "cuda:0:visibility:1"
                    if ":ubatch-1:" in operation_id
                    else "cuda:0:visibility:0"
                )
            else:
                expected_queue = f"cuda:{operation.rank}:compute"
            queues_valid = queues_valid and operation.logical_queue == expected_queue
        logical_stream_checks.append(queues_valid)

        operation_by_id = {
            operation.operation_id: operation for operation in operations
        }
        scheduled_ids = tuple(request.request_id for request in record.scheduled)
        if record.step_index in overlap.R0_DECODE_STEP_INDICES:
            split = len(scheduled_ids) // 2
            source_slices = (
                (0, scheduled_ids[:split]),
                (1, scheduled_ids[split:]),
            )
        else:
            source_slices = ((None, scheduled_ids),)
        expected_dependencies = {}
        expected_correlations = {}
        previous_combines: dict[int, str] = {}
        for layer in range(overlap.NUM_LAYERS):
            for slice_index, (microbatch, request_ids) in enumerate(source_slices):
                ubatch = 0 if microbatch is None else microbatch
                prefix = f"step-{record.step_index}:ubatch-{ubatch}:layer-{layer}"
                correlation = OperationCorrelation(
                    request_ids=request_ids,
                    batch_id=f"step-{record.step_index}",
                    layer=layer,
                    microbatch=microbatch,
                )
                previous = (
                    (previous_combines[slice_index],)
                    if slice_index in previous_combines
                    else ()
                )
                pre_ids = tuple(
                    f"{prefix}:rank-{rank}:pre-dispatch"
                    for rank in range(overlap.DP_SIZE)
                )
                for operation_id in pre_ids:
                    expected_dependencies[operation_id] = ((), previous)
                    expected_correlations[operation_id] = correlation
                dispatch_id = f"{prefix}:ep-dispatch"
                expected_dependencies[dispatch_id] = ((), pre_ids)
                expected_correlations[dispatch_id] = correlation
                expert_ids = tuple(
                    f"{prefix}:rank-{rank}:experts"
                    for rank in range(overlap.DP_SIZE)
                )
                for operation_id in expert_ids:
                    expected_dependencies[operation_id] = ((), (dispatch_id,))
                    expected_correlations[operation_id] = correlation
                combine_id = f"{prefix}:ep-combine"
                expected_dependencies[combine_id] = ((), expert_ids)
                expected_correlations[combine_id] = correlation
                previous_combines[slice_index] = combine_id

        logits = tuple(
            f"step-{record.step_index}:rank-{rank}:logits"
            for rank in range(overlap.DP_SIZE)
        )
        final_combines = tuple(
            previous_combines[index] for index in range(len(source_slices))
        )
        logits_correlation = OperationCorrelation(
            request_ids=scheduled_ids,
            batch_id=f"step-{record.step_index}",
        )
        for logits_id in logits:
            expected_dependencies[logits_id] = ((), final_combines)
            expected_correlations[logits_id] = logits_correlation
        for slice_index, (microbatch, request_ids) in enumerate(source_slices):
            ubatch = 0 if microbatch is None else microbatch
            completion_id = (
                f"step-{record.step_index}:ubatch-{ubatch}:requests-visible"
            )
            expected_dependencies[completion_id] = (logits, ())
            expected_correlations[completion_id] = OperationCorrelation(
                request_ids=request_ids,
                batch_id=f"step-{record.step_index}",
                microbatch=microbatch,
            )
        dependency_checks.append(
            set(expected_dependencies) == set(operation_by_id)
            and all(
                (
                    operation_by_id[operation_id].depends_on,
                    operation_by_id[operation_id].participant_local_depends_on,
                )
                == dependencies
                for operation_id, dependencies in expected_dependencies.items()
            )
        )
        correlation_checks.append(
            set(expected_correlations) == set(operation_by_id)
            and all(
                operation_by_id[operation_id].correlation == correlation
                for operation_id, correlation in expected_correlations.items()
            )
        )
        frontier_valid = True
        frontier_requests = []
        for logits_id in logits:
            frontier_valid = (
                frontier_valid
                and operation_by_id[logits_id].participant_local_depends_on
                == final_combines
            )
        for completion_id in graph.completion_operation_ids:
            operation = operation_by_id[completion_id]
            frontier_valid = frontier_valid and operation.depends_on == logits
            frontier_requests.extend(operation.correlation.request_ids)
        frontier_checks.append(
            frontier_valid
            and tuple(frontier_requests) == scheduled_ids
            and len(frontier_requests) == len(set(frontier_requests))
        )
        metric_ids = tuple(
            metric["request_id"] for metric in row["step_result"]["request_metrics"]
        )
        original_identity_checks.append(metric_ids == scheduled_ids)

    schedule_path = (
        Path(__file__).resolve().parents[2]
        / "simllm"
        / "adapters"
        / "vllm"
        / "schedule.py"
    )
    tree = ast.parse(schedule_path.read_text(encoding="utf-8"), schedule_path)
    imported_modules = set()
    referenced_names = set()
    referenced_attributes = set()
    argument_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module or "")
        elif isinstance(node, ast.Name):
            referenced_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced_attributes.add(node.attr)
        elif isinstance(node, ast.arg):
            argument_names.add(node.arg)
    forbidden_names = {
        "ExecutionGraph",
        "ObservedStepLowerer",
        "SerialStepLowerer",
        "execution_graph_from_observations",
    }
    forbidden_fragments = (
        "compatibility",
        "discount",
        "overlap",
        "percent",
        "random",
    )
    identifiers = referenced_names | referenced_attributes | argument_names
    synthetic_concurrency_absent = (
        not any(module == "random" for module in imported_modules)
        and not any(module.startswith("simllm.backends") for module in imported_modules)
        and not forbidden_names & referenced_names
        and not any(
            fragment in identifier.lower()
            for identifier in identifiers
            for fragment in forbidden_fragments
        )
    )
    return {
        "correlation_grammar_checks": correlation_checks,
        "dependency_grammar_checks": dependency_checks,
        "exact_submission_order_checks": order_checks,
        "frontier_after_logits_checks": frontier_checks,
        "logical_stream_checks": logical_stream_checks,
        "nonempty_steps": len(rows),
        "original_identity_checks": original_identity_checks,
        "synthetic_concurrency_absent": synthetic_concurrency_absent,
    }


def _single_batch_identity(
    modes: dict[str, dict[str, object]],
    live_path: Path,
    routed_experts: Path,
) -> dict[str, object]:
    """Compare control and observed from the same single-batch input time."""

    from simllm.backends import DeviceRuntimeStepSink, SerialStepLowererConfig
    from simllm.compute import RooflineProvider
    from simllm.core import (
        CoarseDeviceRuntime,
        VirtualClock,
        execution_graph_to_json,
        execution_result_to_json,
        step_record_from_json,
        step_result_to_json,
    )

    rows = [
        json.loads(line)
        for line in live_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    config = SerialStepLowererConfig(
        overlap._model_dims(),
        (0,),
        ep_ranks=tuple(range(overlap.DP_SIZE)),
        provider=RooflineProvider(efficiency=0.7),
        routed_moe_supply=overlap._routed_supply(routed_experts),
    )
    checks = []
    details = []
    cumulative_checks = []
    cumulative_details = []
    for placement in overlap.COMM_CEILING_PS:
        for row in rows:
            record = step_record_from_json(row["record"])
            if record.step_index not in overlap.SINGLE_BATCH_STEP_INDICES:
                continue
            observed = overlap._observations_from_source(row["source_graph"])
            control, added_edges = overlap._control_observations(
                observed,
                record.step_index,
            )
            artifacts = []
            for observations in (control, observed):
                clock = VirtualClock(start_ps=record.virtual_time_ps)
                sink = DeviceRuntimeStepSink(
                    config,
                    runtime=CoarseDeviceRuntime(overlap._profile(placement)),
                )
                sink.bind_clock(clock)
                step_result = sink(record, observations)
                outcome = sink.outcomes[-1]
                artifacts.append(
                    {
                        "execution": execution_result_to_json(
                            outcome.execution_result
                        ),
                        "graph": execution_graph_to_json(outcome.graph),
                        "step_result": step_result_to_json(step_result),
                    }
                )
            identical = added_edges == 0 and artifacts[0] == artifacts[1]
            checks.append(identical)
            details.append(
                {
                    "artifact_identity": _canonical_identity(artifacts[0]),
                    "control_edges": added_edges,
                    "identical": identical,
                    "placement": placement,
                    "step_index": record.step_index,
                }
            )
        control_rows = overlap._rows_by_step(modes[f"{placement}:control"])
        observed_rows = overlap._rows_by_step(modes[f"{placement}:observed"])
        for step_index in overlap.SINGLE_BATCH_STEP_INDICES:
            control_row = control_rows[step_index]
            observed_row = observed_rows[step_index]

            def relative_step_result(row: dict[str, object]) -> dict[str, object]:
                result = row["step_result"]
                return {
                    "additive_visit_totals": result["additive_visit_totals"],
                    "request_metrics": [
                        {
                            key: metric[key]
                            for key in (
                                "additive_visit_totals",
                                "attribution",
                                "latency_ps",
                                "phase",
                                "request_id",
                                "token_index",
                            )
                        }
                        for metric in result["request_metrics"]
                    ],
                    "schema": result["schema"],
                    "step_index": result["step_index"],
                    "step_latency_ps": result["step_latency_ps"],
                }

            relative_equal = all(
                control_row[field] == observed_row[field]
                for field in (
                    "latency_ps",
                    "moe_phase_ps",
                    "request_pair_identity",
                    "terminal_ps",
                )
            ) and relative_step_result(control_row) == relative_step_result(
                observed_row
            )
            cumulative_checks.append(relative_equal)
            cumulative_details.append(
                {
                    "control_release_offset_ps": (
                        control_row["released_at_ps"]
                        - observed_row["released_at_ps"]
                    ),
                    "placement": placement,
                    "relative_artifacts_identical": relative_equal,
                    "step_index": step_index,
                }
            )
    return {
        "all_checks": (
            all(checks)
            and len(checks) == 18
            and all(cumulative_checks)
            and len(cumulative_checks) == 18
        ),
        "checks": checks,
        "cumulative_relative_checks": cumulative_checks,
        "cumulative_relative_details": cumulative_details,
        "details": details,
    }


def _structure_is_bounded(modes: dict[str, dict[str, object]]) -> bool:
    for placement in overlap.COMM_CEILING_PS:
        serial = _fraction(modes[f"{placement}:serial"]["r0_tpot_ps"])
        control = _fraction(modes[f"{placement}:control"]["r0_tpot_ps"])
        if abs(serial - control) > STRUCTURE_FRACTION_MAX * serial:
            return False
    return True


def _original_request_identities(
    modes: dict[str, dict[str, object]], live_path: Path, disabled_path: Path
) -> bool:
    enabled_rows = [
        json.loads(line)
        for line in live_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    disabled_rows = [
        json.loads(line)
        for line in disabled_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = [
        [scheduled["request_id"] for scheduled in row["record"]["scheduled"]]
        for row in enabled_rows
    ]
    if len(expected) != overlap.NONEMPTY_STEPS or len(disabled_rows) != len(expected):
        return False
    for result in modes.values():
        if result["request_metric_ids"] != expected:
            return False
    disabled_ids = [
        [metric["request_id"] for metric in row["step_result"]["request_metrics"]]
        for row in disabled_rows
    ]
    return (
        disabled_ids == expected
        and tuple(dict.fromkeys(request for ids in expected for request in ids))
        == ORIGINAL_REQUEST_IDS
    )


def _projected_goal_json(graph: object, step_index: int) -> dict[str, object]:
    from simllm.traffic import project_execution_graph_goal

    projection = project_execution_graph_goal(graph)
    return {
        "artifacts": [artifact.trace.render() for artifact in projection.artifacts],
        "boundaries": len(projection.boundaries),
        "serialized_edges": len(projection.serialized_edges),
        "step_index": step_index,
    }


def _serial_reference_rows(live_path: Path, routed_experts: Path) -> list[dict[str, object]]:
    from simllm.backends import SerialStepLowerer, SerialStepLowererConfig
    from simllm.compute import RooflineProvider
    from simllm.core import (
        CoarseDeviceRuntime,
        CompletionReducer,
        VirtualClock,
        execution_graph_to_json,
        execution_result_to_json,
        step_record_from_json,
        step_record_to_json,
        step_result_to_json,
    )
    from simllm.traffic import render_step_goal

    config = SerialStepLowererConfig(
        overlap._model_dims(),
        (0,),
        ep_ranks=tuple(range(overlap.DP_SIZE)),
        provider=RooflineProvider(efficiency=0.7),
        routed_moe_supply=overlap._routed_supply(routed_experts),
    )
    lowerer = SerialStepLowerer(config)
    runtime = CoarseDeviceRuntime(overlap._profile("cross-node"))
    clock = VirtualClock()
    reducer = CompletionReducer(clock)
    enabled_rows = [
        json.loads(line)
        for line in live_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    for index, enabled in enumerate(enabled_rows):
        source_record = step_record_from_json(enabled["record"])
        record = replace(source_record, virtual_time_ps=clock.now_ps)
        graph = lowerer.lower(record)
        execution = runtime.execute(graph)
        report = runtime.last_report
        if report is None:
            raise RuntimeError("serial reference runtime produced no report")
        result = reducer.reduce(record, graph, execution, report)
        goal = render_step_goal(
            record,
            config.dims,
            config.tp_ranks,
            per_layer_calc_ns=lowerer.timing(record).layer_calc_ns,
            ep_ranks=config.ep_ranks,
            routed_supply=config.routed_moe_supply,
        ).render()
        rows.append(
            {
                "execution_result": execution_result_to_json(execution),
                "legacy_call_arity": 1,
                "legacy_call_index": index,
                "legacy_goal": goal,
                "lowered_graph": execution_graph_to_json(graph),
                "observations_absent": True,
                "projected_goal": _projected_goal_json(graph, record.step_index),
                "record": step_record_to_json(record),
                "source_graph": None,
                "step_result": step_result_to_json(result),
            }
        )
    return rows


def _bypass_artifacts(rows: list[dict[str, object]]):
    from simllm.backends import BypassArtifacts, canonical_bypass_parameters

    # The analytic path has no compiled GOAL or completion CSV. Preserve the
    # standard comparator classes by placing graph-derived GOAL bytes in the
    # binary slot, canonical graph JSON in the CSV slot and canonical runtime
    # completion JSON in the completion slot. RESULTS records this mapping.
    replay_rows = [
        {
            "completion_operation_ids": row["lowered_graph"][
                "completion_operation_ids"
            ],
            "events": [
                (
                    event["operation_id"],
                    event["phase"],
                    event["timestamp_ps"],
                    event.get("subject_object_id"),
                )
                for event in row["execution_result"]["events"]
            ],
            "request_metrics": row["step_result"]["request_metrics"],
            "step_index": row["record"]["step_index"],
        }
        for row in rows
    ]
    return BypassArtifacts(
        goal_text="".join(str(row["legacy_goal"]) for row in rows).encode(),
        goal_binary=_canonical_bytes([row["projected_goal"] for row in rows]),
        topology=_canonical_bytes([row["record"] for row in rows]),
        profile="coarse-device-cross-node-400g-one-rank-per-node",
        seed=0,
        baseline_parameters=canonical_bypass_parameters(
            {
                "dp_size": overlap.DP_SIZE,
                "gpu": "b100",
                "num_layers": overlap.NUM_LAYERS,
                "roofline_efficiency": "0.7",
            }
        ),
        completion_csv=_canonical_bytes([row["lowered_graph"] for row in rows]),
        canonical_completion=_canonical_bytes(
            [row["execution_result"] for row in rows]
        ),
        step_results=_canonical_bytes([row["step_result"] for row in rows]),
        replay_summary=_canonical_bytes(replay_rows),
    )


def _disabled_identity(
    live_path: Path,
    disabled_path: Path,
    routed_experts: Path,
) -> dict[str, object]:
    from simllm.backends import (
        assert_bypass_artifact_identity,
        compare_bypass_artifacts,
    )

    reference_rows = _serial_reference_rows(live_path, routed_experts)
    disabled_rows = [
        json.loads(line)
        for line in disabled_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(reference_rows) != len(disabled_rows):
        return {
            "all_checks": False,
            "artifact_comparison": None,
            "checks": [],
            "compared_steps": 0,
            "details": [],
        }
    reference = _bypass_artifacts(reference_rows)
    candidate = _bypass_artifacts(disabled_rows)
    comparison = compare_bypass_artifacts(reference, candidate)
    assertion_error = None
    try:
        assert_bypass_artifact_identity(reference, candidate)
    except ValueError as exc:
        assertion_error = str(exc)

    checks = []
    details = []
    fields = (
        "execution_result",
        "legacy_goal",
        "lowered_graph",
        "projected_goal",
        "record",
        "step_result",
    )
    for index, (expected, disabled) in enumerate(
        zip(reference_rows, disabled_rows, strict=True)
    ):
        step_checks = {field: disabled.get(field) == expected[field] for field in fields}
        step_checks["legacy_call"] = (
            disabled.get("legacy_call_arity") == 1
            and disabled.get("legacy_call_index") == index
            and disabled.get("observations_absent") is True
            and disabled.get("source_graph") is None
        )
        checks.append(all(step_checks.values()))
        details.append(
            {
                "checks": step_checks,
                "step_index": expected["record"]["step_index"],
            }
        )
    return {
        "all_checks": (
            comparison.equivalent
            and assertion_error is None
            and all(checks)
            and len(checks) == overlap.NONEMPTY_STEPS
        ),
        "artifact_comparison": {
            "assertion_error": assertion_error,
            "behavioral_matches": list(comparison.behavioral_matches),
            "changed_artifacts": list(comparison.changed_artifacts),
            "changed_inputs": list(comparison.changed_inputs),
            "equivalent": comparison.equivalent,
            "input_matches": list(comparison.input_matches),
        },
        "checks": checks,
        "compared_steps": len(checks),
        "details": details,
    }


def _rank_files_passed(directory: Path, mode: str) -> bool:
    rows = [
        json.loads((directory / f"live_rank_{rank}.json").read_text(encoding="utf-8"))
        for rank in range(overlap.DP_SIZE)
    ]
    return all(
        row.get("status") == "PASS" and row.get("producer_mode") == mode
        for row in rows
    )


def _pytest_disabled_byte_lock() -> dict[str, object]:
    repo = Path(__file__).resolve().parents[2]
    node_id = (
        "tests/test_device_step_sink.py::"
        "test_live_legacy_wrapper_preserves_complete_serial_artifact_bytes"
    )
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", node_id],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": ["python", "-m", "pytest", "-q", node_id],
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "stderr": completed.stderr,
        "stdout": completed.stdout,
    }


def _git_head() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _run_study(args: argparse.Namespace) -> dict[str, object]:
    enabled_dir = args.output_dir / "enabled-live"
    modes_dir = args.output_dir / "enabled-modes"
    disabled_dir = args.output_dir / "disabled-live"
    for directory in (enabled_dir, modes_dir, disabled_dir):
        directory.mkdir()

    enabled_args = _args_with_output(args, enabled_dir)
    live_path = overlap._run_live_driver(enabled_args)
    modes = overlap._run_modes(_args_with_output(args, modes_dir), live_path)
    disabled_path = _run_disabled_driver(args, disabled_dir)

    # The headline reads raw request metrics before any exact or fatal check.
    behavioral = _evaluate_behavioral(modes)
    decomposition = overlap._decomposition(modes)
    supporting_scored = overlap._evaluate_scored(modes, decomposition)
    base_inventory = overlap._producer_inventory(live_path)
    source_inventory = _source_inventory(live_path)
    fixture = overlap._serial_fixture()
    fatal = overlap._fatal_guards(
        modes,
        supporting_scored,
        base_inventory,
        fixture,
    )
    single_batch = _single_batch_identity(modes, live_path, args.routed_experts)
    disabled = _disabled_identity(
        live_path,
        disabled_path,
        args.routed_experts,
    )
    pytest_byte_lock = _pytest_disabled_byte_lock()
    disabled_step_records = (
        disabled["compared_steps"] == overlap.NONEMPTY_STEPS
        and all(
            row["checks"]["record"] and row["checks"]["legacy_call"]
            for row in disabled["details"]
        )
    )
    fatal.update(
        {
            "control_observed_single_batch_identity": single_batch["all_checks"],
            "disabled_live_artifacts": disabled["all_checks"],
            "disabled_live_ranks": _rank_files_passed(disabled_dir, "disabled"),
            "disabled_live_step_records": disabled_step_records,
            "enabled_live_ranks": _rank_files_passed(enabled_dir, "enabled"),
            "original_request_identities": _original_request_identities(
                modes, live_path, disabled_path
            ),
            "producer_completion_frontiers": all(
                source_inventory["frontier_after_logits_checks"]
            ),
            "producer_exact_submission_order": all(
                source_inventory["exact_submission_order_checks"]
            ),
            "producer_correlation_grammar": all(
                source_inventory["correlation_grammar_checks"]
            ),
            "producer_dependency_grammar": all(
                source_inventory["dependency_grammar_checks"]
            ),
            "producer_logical_streams": all(
                source_inventory["logical_stream_checks"]
            ),
            "producer_no_synthetic_concurrency": source_inventory[
                "synthetic_concurrency_absent"
            ],
            "pytest_disabled_byte_lock": pytest_byte_lock["passed"],
            "structure_is_bounded": _structure_is_bounded(modes),
        }
    )
    fatal_failed = sorted(name for name, value in fatal.items() if not value)
    summary = {}
    for key, result in modes.items():
        public_result = {
            field: value for field, value in result.items() if field != "raw_rows"
        }
        public_result["raw_rows"] = [
            row
            for row in result["raw_rows"]
            if row["step_index"] in overlap.SINGLE_BATCH_STEP_INDICES
        ]
        summary[key] = public_result
    behavioral_passed = (
        behavioral["genuine_risk_passed"]
        == behavioral["genuine_risk_executed"]
    )
    return {
        "behavioral_evidence": behavioral,
        "capture": {
            "rows": overlap._line_count(args.capture),
            "sha256": _sha256(args.capture),
        },
        "closure_candidate": not fatal_failed and behavioral_passed,
        "decomposition": decomposition,
        "disabled_live_identity": disabled,
        "fatal_unscored_guards": fatal,
        "mode_results": summary,
        "producer_inventory": {
            "landed_overlap_inventory": base_inventory,
            "qualification_inventory": source_inventory,
        },
        "pytest_disabled_byte_lock": pytest_byte_lock,
        "provenance": {
            "expectations_commit": EXPECTATIONS_COMMIT,
            "repository_commit_observed": _git_head(),
            "vllm_authored_against_commit": overlap.VLLM_AUTHORED_AGAINST_COMMIT,
            "vllm_observed_file_sha256": overlap.SOURCE_HASHES,
            "vllm_observed_version": overlap.VLLM_VERSION,
        },
        "run_status": "VOID" if fatal_failed else "VALID",
        "schema": "simllm-vllm-producer-qualification-v1",
        "serial_fixture": fixture,
        "single_batch_identity": single_batch,
    }


def main() -> None:
    args = _parse_args()
    check_expectation_registry(args)
    if args.check_only:
        print("check-only: frozen expectation registry passed; no artifacts written")
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = _run_study(args)
    result_path = args.output_dir / "results.json"
    result_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failed = sorted(
        name for name, value in report["fatal_unscored_guards"].items() if not value
    )
    if failed:
        print("run status: VOID")
        print("fatal guards violated: " + ", ".join(failed))
    else:
        behavior = report["behavioral_evidence"]
        print("run status: VALID")
        print(
            "genuine-risk instances: "
            f"{behavior['genuine_risk_passed']}/{behavior['genuine_risk_executed']}"
        )
        print("fatal guards violated: none")
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
