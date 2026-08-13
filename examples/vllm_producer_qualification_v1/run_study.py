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
        execution_graph_from_json,
        step_record_from_json,
    )

    rows = [
        json.loads(line)
        for line in live_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    order_checks = []
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
        logits = tuple(
            f"step-{record.step_index}:rank-{rank}:logits"
            for rank in range(overlap.DP_SIZE)
        )
        microbatches = (
            (0, 1)
            if record.step_index in overlap.R0_DECODE_STEP_INDICES
            else (0,)
        )
        final_combines = tuple(
            f"step-{record.step_index}:ubatch-{microbatch}:layer-23:ep-combine"
            for microbatch in microbatches
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
        scheduled_ids = tuple(request.request_id for request in record.scheduled)
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
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module or "")
        elif isinstance(node, ast.Name):
            referenced_names.add(node.id)
    forbidden_names = {
        "ExecutionGraph",
        "ObservedStepLowerer",
        "SerialStepLowerer",
        "execution_graph_from_observations",
    }
    synthetic_concurrency_absent = (
        not any(module == "random" for module in imported_modules)
        and not any(module.startswith("simllm.backends") for module in imported_modules)
        and not forbidden_names & referenced_names
    )
    return {
        "exact_submission_order_checks": order_checks,
        "frontier_after_logits_checks": frontier_checks,
        "logical_stream_checks": logical_stream_checks,
        "nonempty_steps": len(rows),
        "original_identity_checks": original_identity_checks,
        "synthetic_concurrency_absent": synthetic_concurrency_absent,
    }


def _single_batch_identity(modes: dict[str, dict[str, object]]) -> bool:
    for placement in overlap.COMM_CEILING_PS:
        control = overlap._rows_by_step(modes[f"{placement}:control"])
        observed = overlap._rows_by_step(modes[f"{placement}:observed"])
        for step_index in overlap.SINGLE_BATCH_STEP_INDICES:
            left = control[step_index]
            right = observed[step_index]
            for field in (
                "edge_free_graph_identity",
                "execution_result_identity",
                "graph_identity",
                "latency_ps",
                "moe_phase_ps",
                "request_pair_identity",
                "step_result",
                "terminal_ps",
            ):
                if left[field] != right[field]:
                    return False
    return True


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


def _disabled_identity(
    modes: dict[str, dict[str, object]], live_path: Path, disabled_path: Path
) -> dict[str, object]:
    from simllm.core import step_record_from_json, step_record_to_json

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
    serial_result = modes["cross-node:serial"]
    serial_rows = overlap._rows_by_step(serial_result)
    goals = {
        int(row["step_index"]): row for row in serial_result["serial_goal_identities"]
    }
    projections = {
        int(row["step_index"]): row
        for row in serial_result["serial_projection_identities"]
    }
    checks = []
    details = []
    if len(enabled_rows) != len(disabled_rows):
        return {
            "all_checks": False,
            "checks": [],
            "compared_steps": 0,
            "details": [],
        }
    for index, (enabled, disabled) in enumerate(zip(enabled_rows, disabled_rows)):
        source_record = step_record_from_json(enabled["record"])
        serial_row = serial_rows[source_record.step_index]
        expected_record = replace(
            source_record,
            virtual_time_ps=serial_row["released_at_ps"],
        )
        expected_record_json = step_record_to_json(expected_record)
        step_checks = {
            "execution_result": (
                disabled["execution_result_identity"]
                == serial_row["execution_result_identity"]
            ),
            "graph": disabled["lowered_graph_identity"] == serial_row["graph_identity"],
            "legacy_call": (
                disabled["legacy_call_arity"] == 1
                and disabled["legacy_call_index"] == index
                and disabled["observations_absent"] is True
            ),
            "legacy_goal": (
                disabled["legacy_goal_identity"] == goals[source_record.step_index]
            ),
            "projected_goal": (
                disabled["projected_goal_identity"]
                == projections[source_record.step_index]
            ),
            "step_record": (
                disabled["record"] == expected_record_json
                and disabled["record_identity"]
                == _canonical_identity(expected_record_json)
            ),
            "step_result": disabled["step_result"] == serial_row["step_result"],
        }
        checks.append(all(step_checks.values()))
        details.append(
            {
                "checks": step_checks,
                "step_index": source_record.step_index,
            }
        )
    return {
        "all_checks": all(checks) and len(checks) == overlap.NONEMPTY_STEPS,
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
    disabled = _disabled_identity(modes, live_path, disabled_path)
    fatal.update(
        {
            "control_observed_single_batch_identity": _single_batch_identity(modes),
            "disabled_live_artifacts": disabled["all_checks"],
            "disabled_live_ranks": _rank_files_passed(disabled_dir, "disabled"),
            "disabled_live_step_records": (
                disabled["all_checks"]
                and disabled["compared_steps"] == overlap.NONEMPTY_STEPS
            ),
            "enabled_live_ranks": _rank_files_passed(enabled_dir, "enabled"),
            "original_request_identities": _original_request_identities(
                modes, live_path, disabled_path
            ),
            "producer_completion_frontiers_after_logits": all(
                source_inventory["frontier_after_logits_checks"]
            ),
            "producer_exact_submission_order": all(
                source_inventory["exact_submission_order_checks"]
            ),
            "producer_logical_streams": all(
                source_inventory["logical_stream_checks"]
            ),
            "producer_no_synthetic_concurrency": source_inventory[
                "synthetic_concurrency_absent"
            ],
            "structure_is_bounded": _structure_is_bounded(modes),
        }
    )
    fatal_failed = sorted(name for name, value in fatal.items() if not value)
    summary = {
        key: {field: value for field, value in result.items() if field != "raw_rows"}
        for key, result in modes.items()
    }
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
