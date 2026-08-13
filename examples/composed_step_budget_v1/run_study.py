"""Measure the composed step of the merged host-cost and collective-floor code.

Wave 12 landed a fixed per-step host cost and a calibrated collective latency
floor on main. The two branches published arithmetic projections of the same
merged code that disagree by a factor of 1.75, and neither ran the chain. This
study runs `examples/end_to_end_replay_v1`, the repository's mission chain,
with both features enabled and with both disabled, and reports which
composition the code actually computes.

Every guard, relation, interval and decision rule this script evaluates is
frozen in `expectations.md` and `expectations.json`, committed before this file
existed and before any run.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

MISSION_STUDY_PATH = REPOSITORY_ROOT / "examples/end_to_end_replay_v1/run_study.py"

PS_PER_SECOND = 1_000_000_000_000
#: request rows the disabled cell must reproduce, in declaration order
DISABLED_REQUEST_IDS = ("r00", "r01", "r02")


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mission_module() -> Any:
    """Return the accepted mission study as an importable module."""

    return _load_module(MISSION_STUDY_PATH, "end_to_end_replay_v1")


def check_module() -> Any:
    """Return this study's freeze validator."""

    return _load_module(STUDY_DIR / "check_only.py", "composed_step_budget_check")


# ------------------------------------------------------------------ driver ---


def _stage_commands(args: argparse.Namespace, frozen: dict[str, Any]) -> list[list[str]]:
    mission = mission_module()
    commands = []
    for cell in frozen["cells"]:
        commands.append(
            mission.child_command(
                args,
                f"cell:{cell['source_cell']}",
                cell_label=cell["label"],
                host_profile=cell["host_profile"],
                host_launch_count=cell["host_launch_count"],
                collective_latency_profile=cell["collective_latency_profile"],
            )
        )
    return commands


def _run(command: list[str]) -> int:
    completed = subprocess.run(command, check=False, env=os.environ.copy())
    return completed.returncode


def run_cells(args: argparse.Namespace, frozen: dict[str, Any]) -> None:
    """Run the capture once, then every frozen cell as an isolated child."""

    mission = mission_module()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    os.environ["SIMLLM_HTSIM_RNIC"] = str(args.htsim_rnic)
    code = _run(mission.child_command(args, "capture"))
    if code != 0:
        raise SystemExit(f"capture stage failed with code {code}")
    (args.run_dir / "cells").mkdir(parents=True, exist_ok=True)
    commands = _stage_commands(args, frozen)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        codes = list(pool.map(_run, commands))
    failed = [
        frozen["cells"][index]["label"]
        for index, code in enumerate(codes)
        if code != 0
    ]
    if failed:
        raise SystemExit(f"cell stages failed: {', '.join(failed)}")


# ---------------------------------------------------------------- analysis ---


def _read_cell(run_dir: Path, label: str) -> dict[str, Any]:
    return json.loads(
        (run_dir / "cells" / label / "cell.json").read_text(encoding="utf-8")
    )


def _read_composition(run_dir: Path, label: str) -> dict[str, Any] | None:
    path = run_dir / "cells" / label / "composition.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decode_steps(cell: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        step
        for step in cell["steps"]
        if step.get("simulated")
        and step["scheduled"]
        and all(scheduled["phase"] == "decode" for scheduled in step["scheduled"])
    ]


def _composition_key(step: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        (
            scheduled["request_id"],
            scheduled["phase"],
            scheduled["num_new_tokens"],
            scheduled["context_length"],
        )
        for scheduled in step["scheduled"]
    )


def _decode_by_composition(cell: dict[str, Any]) -> dict[Any, dict[str, Any]]:
    rows: dict[Any, dict[str, Any]] = {}
    for step in _decode_steps(cell):
        key = _composition_key(step)
        if key in rows:
            raise AssertionError("two decode steps share a scheduling composition")
        rows[key] = step
    return rows


def _request_row(cell: dict[str, Any], request_id: str) -> dict[str, Any]:
    for row in cell["requests"]:
        if row["request_id"] == request_id:
            return row
    raise AssertionError(f"cell has no request {request_id}")


def _disabled_identity(
    run_dir: Path,
    cell: dict[str, Any],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    """Score G1 and G2, the two off paths reproducing their baseline together."""

    literals = frozen["disabled_path_literals"]
    findings = []

    def compare(name: str, expected: Any, actual: Any) -> None:
        if expected != actual:
            findings.append({"field": name, "expected": expected, "actual": actual})

    steps_path = run_dir / "cells" / "off-400g" / "steps.jsonl"
    compare("steps_jsonl_sha256", literals["steps_jsonl_sha256"], _sha256(steps_path))
    compare("scheduler_steps", literals["scheduler_steps"], cell["scheduler_steps"])
    compare("htsim_invocations", literals["htsim_invocations"], cell["htsim_invocations"])
    compare(
        "total_routed_bytes",
        literals["total_routed_bytes"],
        cell["total_routed_bytes"],
    )
    compare(
        "peak_rank_egress_bytes",
        literals["peak_rank_egress_bytes"],
        max(int(value) for value in cell["routing"]["rank_egress_bytes"].values()),
    )
    for request_id, expected_bytes in literals["request_routed_bytes"].items():
        compare(
            f"request_routed_bytes.{request_id}",
            expected_bytes,
            cell["routing"]["request_bytes"][request_id],
        )
    by_index = {step["step_index"]: step for step in cell["steps"]}
    for index, expected_ps in literals["step_latency_ps"].items():
        compare(
            f"step_latency_ps.{index}",
            expected_ps,
            by_index[int(index)]["step_latency_ps"],
        )
    for request_id, expected in literals["requests"].items():
        row = _request_row(cell, request_id)
        compare(f"{request_id}.ttft_ps", expected["ttft_ps"], row["ttft_ps"])
        attribution = row["ttft_attribution"]
        for component in ("queue_ps", "kernel_ps", "collective_ps"):
            compare(
                f"{request_id}.{component}",
                expected[component],
                attribution[component],
            )
    capture_path = run_dir / "capture" / "greedy.jsonl"
    compare("capture_sha256", literals["capture_sha256"], _sha256(capture_path))
    return {"held": not findings, "findings": findings}


def _conservation_guard(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    findings = []
    for label, cell in cells.items():
        conservation = cell["conservation"]
        for name in (
            "makespan_conservation_failures",
            "interval_conservation_failures",
            "artifact_partition_failures",
            "kernel_disagrees_with_compute_service",
        ):
            if conservation[name]:
                findings.append({"cell": label, "field": name, "value": conservation[name]})
        for step in cell["steps"]:
            if step["completed_at_ps"] != step["virtual_time_ps"] + step["step_latency_ps"]:
                findings.append(
                    {"cell": label, "field": "completion", "step": step["step_index"]}
                )
    return {
        "held": not findings,
        "findings": findings,
        "intervals": sum(cell["conservation"]["intervals"] for cell in cells.values()),
    }


def _inactive_component_guard(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    violations = sum(
        cell["conservation"]["inactive_component_violations"] for cell in cells.values()
    )
    return {"held": violations == 0, "violations": violations}


def _replay_guard(cells: dict[str, dict[str, Any]], mission: Any) -> dict[str, Any]:
    findings = []
    for label, cell in cells.items():
        for row in cell["requests"]:
            if not row["served_matches_oracle"]:
                findings.append({"cell": label, "request_id": row["request_id"], "field": "tokens"})
            if not mission._stop_reason_agrees(row):
                findings.append({"cell": label, "request_id": row["request_id"], "field": "stop"})
            if row["ttft_ps"] is None or row["ttft_ps"] <= 0:
                findings.append({"cell": label, "request_id": row["request_id"], "field": "ttft"})
    return {"held": not findings, "findings": findings}


def _backend_health_guard(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    findings = []
    simulated = 0
    for label, cell in cells.items():
        for step in cell["steps"]:
            if not step.get("simulated"):
                continue
            simulated += 1
            if (
                step["routing_mode"] != "captured"
                or step["placement_epoch"] != 0
                or step["quiescent"] is not True
            ):
                findings.append({"cell": label, "step": step["step_index"]})
    return {"held": not findings, "findings": findings, "simulated_steps": simulated}


def _floor_reach_guard(
    compositions: dict[str, dict[str, Any]],
    disabled: dict[str, Any] | None,
    frozen: dict[str, Any],
) -> dict[str, Any]:
    constants = frozen["constants_ps"]
    findings = []
    charged_steps = 0
    for label, composition in compositions.items():
        for step in composition["steps"]:
            charged_steps += 1
            if step["collective_base_charges"] != frozen["collectives_per_step"]:
                findings.append(
                    {
                        "cell": label,
                        "step": step["step_index"],
                        "field": "charges",
                        "value": step["collective_base_charges"],
                    }
                )
            if step["collective_base_sum_ps"] != constants["collective_floor_total"]:
                findings.append(
                    {
                        "cell": label,
                        "step": step["step_index"],
                        "field": "sum",
                        "value": step["collective_base_sum_ps"],
                    }
                )
            if step["distinct_base_charge_ps"] not in (
                [],
                [constants["collective_floor_width8"]],
            ):
                findings.append(
                    {
                        "cell": label,
                        "step": step["step_index"],
                        "field": "distinct",
                        "value": step["distinct_base_charge_ps"],
                    }
                )
    if disabled is not None:
        findings.append({"cell": "off-400g", "field": "unexpected_composition_record"})
    return {"held": not findings, "findings": findings, "charged_steps": charged_steps}


def _envelope_guard(
    compositions: dict[str, dict[str, Any]],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    low, high = frozen["endpoint_envelope_bytes"]
    findings = []
    observed_min: int | None = None
    observed_max: int | None = None
    for label, composition in compositions.items():
        for step in composition["steps"]:
            for value in step["critical_endpoint_bytes"]:
                if value == 0:
                    continue
                observed_min = value if observed_min is None else min(observed_min, value)
                observed_max = value if observed_max is None else max(observed_max, value)
                if not low <= value <= high:
                    findings.append(
                        {"cell": label, "step": step["step_index"], "bytes": value}
                    )
    return {
        "held": not findings,
        "findings": findings,
        "observed_endpoint_bytes": [observed_min, observed_max],
        "envelope_bytes": [low, high],
    }


def _device_guard(
    compositions: dict[str, dict[str, Any]],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    findings = []
    by_label = {cell["label"]: cell for cell in frozen["cells"]}
    for label, composition in compositions.items():
        if composition["gpu_key"] != "gtx1660-ti-sm75":
            findings.append({"cell": label, "field": "gpu_key", "value": composition["gpu_key"]})
        if composition["provider_envelope"] != "b100":
            findings.append({"cell": label, "field": "provider_envelope"})
        if composition["host_profile"] != by_label[label]["host_profile"]:
            findings.append({"cell": label, "field": "host_profile"})
        if composition["host_launch_count"] != by_label[label]["host_launch_count"]:
            findings.append({"cell": label, "field": "host_launch_count"})
        if composition["collective_latency_profile"] != by_label[label][
            "collective_latency_profile"
        ]:
            findings.append({"cell": label, "field": "collective_latency_profile"})
    return {"held": not findings, "findings": findings}


def _provider_input_guard(
    compositions: dict[str, dict[str, Any]],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    """G10: the pinned provider kept the accepted compute input."""

    expected = frozen["constants_ps"]["accepted_decode_compute"]
    findings = []
    observed = set()
    for label, composition in compositions.items():
        for step in composition["steps"]:
            observed.add(step["provider_compute_ps"])
            if step["provider_compute_ps"] > 2 * expected:
                findings.append(
                    {
                        "cell": label,
                        "step": step["step_index"],
                        "provider_compute_ps": step["provider_compute_ps"],
                    }
                )
    return {
        "held": not findings and expected in observed,
        "findings": findings,
        "accepted_decode_compute_ps": expected,
        "observed_provider_compute_ps": sorted(observed),
    }


def _relation_f1(
    cells: dict[str, dict[str, Any]],
    compositions: dict[str, dict[str, Any]],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    intervals = frozen["intervals_ps"]
    quantized = {
        "on-graph440-400g": frozen["constants_ps"]["quantized_graph440"],
        "on-graph440-200g": frozen["constants_ps"]["quantized_graph440"],
        "on-eager567-400g": frozen["constants_ps"]["quantized_eager567"],
    }
    rows = []
    failures = []
    for label in sorted(compositions):
        additive = intervals["additive"][label]
        overlapped = intervals["overlapped"][label]
        decode = _decode_steps(cells[label])
        latencies = [step["step_latency_ps"] for step in decode]
        inside_additive = [
            value for value in latencies if additive[0] <= value <= additive[1]
        ]
        inside_overlapped = [
            value for value in latencies if overlapped[0] <= value <= overlapped[1]
        ]
        services = sorted(
            {step["compute_service_ps"] for step in compositions[label]["steps"]}
        )
        row = {
            "cell": label,
            "decode_steps": len(latencies),
            "decode_latency_ps": [min(latencies), max(latencies)] if latencies else [],
            "additive_interval_ps": additive,
            "overlapped_interval_ps": overlapped,
            "inside_additive": len(inside_additive),
            "inside_overlapped": len(inside_overlapped),
            "compute_service_ps": services,
            "expected_compute_service_ps": quantized[label],
        }
        rows.append(row)
        if not latencies:
            failures.append({"cell": label, "reason": "no decode step"})
        if len(inside_additive) != len(latencies) or inside_overlapped:
            failures.append({"cell": label, "reason": "interval"})
        if services != [quantized[label]]:
            failures.append({"cell": label, "reason": "compute service"})
    return {
        "class": "scored_behavioral",
        "evaluated": True,
        "passed": not failures,
        "failures": failures,
        "rows": rows,
    }


def _relation_f2(
    cells: dict[str, dict[str, Any]],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    expected = frozen["relations"]["F2"]["exact_difference_ps"]
    minimum = frozen["relations"]["F2"]["minimum_matched_compositions"]
    graph = _decode_by_composition(cells["on-graph440-400g"])
    eager = _decode_by_composition(cells["on-eager567-400g"])
    shared = sorted(set(graph) & set(eager), key=repr)
    differences = [
        {
            "composition": [list(entry) for entry in key],
            "graph_ps": graph[key]["step_latency_ps"],
            "eager_ps": eager[key]["step_latency_ps"],
            "difference_ps": eager[key]["step_latency_ps"] - graph[key]["step_latency_ps"],
        }
        for key in shared
    ]
    if len(shared) < minimum:
        return {
            "class": "scored_behavioral",
            "evaluated": False,
            "reason": "fewer matched decode compositions than the frozen minimum",
            "matched": len(shared),
            "minimum": minimum,
            "differences": differences,
        }
    bad = [row for row in differences if row["difference_ps"] != expected]
    return {
        "class": "scored_behavioral",
        "evaluated": True,
        "passed": not bad,
        "matched": len(shared),
        "expected_difference_ps": expected,
        "violations": bad,
        "differences": differences,
    }


def _relation_f3(
    cells: dict[str, dict[str, Any]],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    band = frozen["relations"]["F3"]["ratio_band"]
    minimum = frozen["relations"]["F3"]["minimum_matched_compositions"]
    fast = _decode_by_composition(cells["on-graph440-400g"])
    slow = _decode_by_composition(cells["on-graph440-200g"])
    shared = sorted(set(fast) & set(slow), key=repr)
    ratios = [
        {
            "composition": [list(entry) for entry in key],
            "fast_ps": fast[key]["step_latency_ps"],
            "slow_ps": slow[key]["step_latency_ps"],
            "ratio": slow[key]["step_latency_ps"] / fast[key]["step_latency_ps"],
        }
        for key in shared
    ]
    if len(shared) < minimum:
        return {
            "class": "scored_behavioral",
            "evaluated": False,
            "reason": "fewer matched decode compositions than the frozen minimum",
            "matched": len(shared),
            "minimum": minimum,
            "ratios": ratios,
        }
    bad = [row for row in ratios if not band[0] < row["ratio"] <= band[1]]
    return {
        "class": "scored_behavioral",
        "evaluated": True,
        "passed": not bad,
        "matched": len(shared),
        "ratio_band": band,
        "observed_ratio_range": [
            min(row["ratio"] for row in ratios),
            max(row["ratio"] for row in ratios),
        ],
        "published_disabled_path_ratio_range": frozen["relations"]["F3"][
            "published_disabled_path_ratio_range"
        ],
        "violations": bad,
    }


def _exact_unscored(
    cells: dict[str, dict[str, Any]],
    compositions: dict[str, dict[str, Any]],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    e1 = []
    for label, composition in compositions.items():
        for step in composition["steps"]:
            expected = (
                step["compute_service_ps"]
                + step["collective_base_sum_ps"]
                + step["fabric_service_sum_ps"]
            )
            if expected != step["step_latency_ps"]:
                e1.append({"cell": label, "step": step["step_index"]})
    off = cells["off-400g"]
    ideal_low, ideal_high = frozen["intervals_ps"]["ideal_compute"]
    e2 = [
        step["step_index"]
        for step in off["steps"]
        if step.get("simulated")
        and not ideal_low <= step["compute_service_ps"] <= ideal_high
    ]
    e3 = []
    for label, cell in cells.items():
        for row in cell["requests"]:
            attribution = row["ttft_attribution"]
            if row["ttft_ps"] != attribution["total_ps"]:
                e3.append({"cell": label, "request_id": row["request_id"], "field": "ttft"})
            if row["tpot_numerator"] is None or row["token_count"] < 2:
                continue
            tpot = Fraction(row["tpot_numerator"], row["tpot_denominator"])
            decode = row["decode_attribution"]
            if tpot * (row["token_count"] - 1) != decode["total_ps"]:
                e3.append({"cell": label, "request_id": row["request_id"], "field": "tpot"})
    return {
        "E1": {"class": "exact_unscored", "passed": not e1, "failures": e1},
        "E2": {"class": "exact_unscored", "passed": not e2, "failures": e2},
        "E3": {"class": "exact_unscored", "passed": not e3, "failures": e3},
    }


def _diagnostics(
    cells: dict[str, dict[str, Any]],
    compositions: dict[str, dict[str, Any]],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    """Reported, never scored: budget, coverage, decode rate, counterfactual."""

    coverage = frozen["traffic_coverage"]
    added = coverage["added_floor_ps"]
    hidden = coverage["overlap_counterfactual_hidden_ps"]
    rows = []
    for label in sorted(compositions):
        decode = _decode_steps(cells[label])
        if not decode:
            continue
        latencies = sorted(step["step_latency_ps"] for step in decode)
        median = latencies[len(latencies) // 2]
        composition = compositions[label]
        by_index = {step["step_index"]: step for step in composition["steps"]}
        sample = by_index[decode[len(decode) // 2]["step_index"]]
        rows.append(
            {
                "cell": label,
                "median_decode_step_ps": median,
                "decode_step_range_ps": [latencies[0], latencies[-1]],
                "compute_share": sample["compute_service_ps"] / median,
                "collective_floor_share": sample["collective_base_sum_ps"] / median,
                "fabric_share": sample["fabric_service_sum_ps"] / median,
                "tensor_parallel_addition_ps": added,
                "tensor_parallel_share": added / median,
                "half_overlap_saving_ps": hidden,
                "half_overlap_share": hidden / median,
                "implied_tokens_per_second": PS_PER_SECOND / median,
            }
        )
    tpot_rows = []
    for label, cell in cells.items():
        for row in cell["requests"]:
            if row["tpot_numerator"] is None:
                continue
            tpot = row["tpot_numerator"] / row["tpot_denominator"]
            tpot_rows.append(
                {
                    "cell": label,
                    "request_id": row["request_id"],
                    "tpot_ps": tpot,
                    "tokens_per_second": PS_PER_SECOND / tpot,
                    "ttft_ps": row["ttft_ps"],
                }
            )
    return {"composed_steps": rows, "per_request": tpot_rows}


def analyze(args: argparse.Namespace, frozen: dict[str, Any]) -> dict[str, Any]:
    """Evaluate every frozen guard and relation over the produced cells."""

    mission = mission_module()
    labels = [cell["label"] for cell in frozen["cells"]]
    cells = {label: _read_cell(args.run_dir, label) for label in labels}
    compositions = {}
    disabled_composition = None
    for cell in frozen["cells"]:
        record = _read_composition(args.run_dir, cell["label"])
        if cell["enabled"]:
            if record is None:
                raise SystemExit(f"enabled cell {cell['label']} wrote no composition record")
            compositions[cell["label"]] = record
        elif record is not None:
            disabled_composition = record

    guards = {
        "G1_disabled_path_identity": _disabled_identity(
            args.run_dir, cells["off-400g"], frozen
        ),
        "G2_oracle_identity": {
            "held": _sha256(args.run_dir / "capture" / "greedy.jsonl")
            == frozen["disabled_path_literals"]["capture_sha256"]
        },
        "G3_conservation": _conservation_guard(cells),
        "G4_inactive_components": _inactive_component_guard(cells),
        "G5_replay_identity": _replay_guard(cells, mission),
        "G6_backend_health": _backend_health_guard(cells),
        "G7_floor_reach": _floor_reach_guard(compositions, disabled_composition, frozen),
        "G8_endpoint_envelope": _envelope_guard(compositions, frozen),
        "G9_device_disclosure": _device_guard(compositions, frozen),
        "G10_provider_input": _provider_input_guard(compositions, frozen),
    }
    violated = sorted(name for name, guard in guards.items() if not guard["held"])

    relations = {
        "F1": _relation_f1(cells, compositions, frozen),
        "F2": _relation_f2(cells, frozen),
        "F3": _relation_f3(cells, frozen),
    }
    evaluated = [name for name, row in relations.items() if row["evaluated"]]
    passed = [name for name in evaluated if relations[name]["passed"]]

    summary = {
        "schema": "simllm-composed-step-budget-summary-v1",
        "study": "composed_step_budget_v1",
        "void": bool(violated),
        "violated_fatal_guards": violated,
        "fatal_guards": guards,
        "scored_behavioral": relations,
        "behavioral_evaluated": sorted(evaluated),
        "behavioral_passed": sorted(passed),
        "exact_unscored": _exact_unscored(cells, compositions, frozen),
        "diagnostics": _diagnostics(cells, compositions, frozen),
        "cells": {
            label: {
                "scheduler_steps": cell["scheduler_steps"],
                "simulated_steps": cell["simulated_steps"],
                "htsim_invocations": cell["htsim_invocations"],
                "total_routed_bytes": cell["total_routed_bytes"],
                "wall_seconds": cell["wall_seconds"],
            }
            for label, cell in cells.items()
        },
    }
    return summary


# --------------------------------------------------------------- entrypoint ---


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--htsim-rnic", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--internal", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    checker = check_module()
    frozen = checker.load_expectations()
    checker.check_arithmetic(frozen)
    checker.check_inputs(args)
    if args.check_only:
        print(
            f"check-only run-dir={args.run_dir}; validated the frozen composition "
            "contract and produced no artifacts"
        )
        return
    if args.internal != "analyze":
        run_cells(args, frozen)
    summary = analyze(args, frozen)
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "void": summary["void"],
                "violated_fatal_guards": summary["violated_fatal_guards"],
                "behavioral_evaluated": summary["behavioral_evaluated"],
                "behavioral_passed": summary["behavioral_passed"],
                "composed_steps": summary["diagnostics"]["composed_steps"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
