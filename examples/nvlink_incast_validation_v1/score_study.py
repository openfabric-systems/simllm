#!/usr/bin/env python3
"""Score TRAF-73 hardware rows against the frozen simulation predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAF70_ROOT = HERE.parent / "a100_nvlink_packet_v2"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_campaign = _load_module("_traf73_run_campaign", HERE / "run_campaign.py")
_previous_run_study = sys.modules.get("run_study")
sys.modules["run_study"] = run_campaign.traf70_run
try:
    traf70_score = _load_module(
        "_traf73_traf70_score_hardware", TRAF70_ROOT / "score_hardware.py"
    )
finally:
    if _previous_run_study is None:
        del sys.modules["run_study"]
    else:
        sys.modules["run_study"] = _previous_run_study

SCORE_SCHEMA = "simllm-nvlink-incast-validation-score-v1"
COMPARISON_FIELDS = (
    "degree",
    "size_bytes",
    "hardware_completion_us_by_source",
    "simulation_completion_us_by_source",
    "completion_signed_relative_error_by_source",
    "hardware_aggregate_gbps",
    "simulation_aggregate_gbps",
    "aggregate_signed_relative_error",
    "physical_floor_us",
    "physical_ceiling_us",
    "physical_sanity",
    "verdict",
    "responsible_parameter",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--scheduler-job", required=True)
    parser.add_argument("--residual-task", default="")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    score = audit_hardware(
        args.bulk_root,
        expected_head=args.expected_head,
        scheduler_job=args.scheduler_job,
        residual_task=args.residual_task,
    )
    if args.json_out is not None:
        write_json(args.json_out, score)
    if args.csv_out is not None:
        write_comparison_csv(args.csv_out, score["comparisons"])
    if args.markdown_out is not None:
        write_text(args.markdown_out, render_markdown(score))
    if args.json_out is None and args.csv_out is None and args.markdown_out is None:
        print(json.dumps(score, indent=2, sort_keys=True))
    return 0


def audit_hardware(
    bulk_root: Path,
    *,
    expected_head: str,
    scheduler_job: str,
    residual_task: str = "",
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", expected_head) is None:
        raise ValueError("expected head must be a full lowercase Git SHA")
    if not scheduler_job:
        raise ValueError("scheduler job identity is required")
    if residual_task and re.fullmatch(r"TRAF-[0-9]+", residual_task) is None:
        raise ValueError("residual task must be an exact TRAF ID")
    frozen = run_campaign.load_expectations()
    run_campaign.verify_preservation(frozen)
    attempt = load_attempt(bulk_root, expected_head=expected_head)
    rows = read_rows(attempt / "results.jsonl")
    guard_result = score_fatal_guards(attempt, rows, frozen)
    measurement_valid = guard_result["verdict"] == "PASS"
    sample_rows = summarize_samples(rows, frozen)
    comparisons = compare_cells(sample_rows, frozen, measurement_valid=measurement_valid)
    miss_count = sum(row["verdict"] == "MISS" for row in comparisons)
    pass_count = sum(row["verdict"] == "PASS" for row in comparisons)
    if not measurement_valid:
        status = "VOID_FATAL_GUARD"
        task_status = "OPEN"
        registry_effect = "TRAF-73 stays open because the run is void"
    elif miss_count:
        if not residual_task:
            raise RuntimeError("a valid miss requires --residual-task before publication")
        status = f"VALID_MIXED_{pass_count}_PASS_{miss_count}_MISS"
        task_status = "CLOSED_WITH_FINDING"
        registry_effect = (
            f"TRAF-73 closes as a completed validation; {residual_task} owns the "
            "identified model precision residual"
        )
    else:
        status = "VALID_PASS_6_OF_6"
        task_status = "CLOSED_VALIDATED"
        registry_effect = "TRAF-73 closes with no residual model task"
    return {
        "schema": SCORE_SCHEMA,
        "task_id": "TRAF-73",
        "status": status,
        "task_status": task_status,
        "registry_effect": registry_effect,
        "residual_task": residual_task or None,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "scheduler_job": scheduler_job,
        "execution_head": expected_head,
        "expectations_commit": run_campaign.EXPECTATIONS_COMMIT,
        "expectations_sha256": run_campaign.EXPECTATIONS_SHA256,
        "attempt_manifest_sha256": run_campaign.sha256(attempt / "manifest.json"),
        "producer_binary_sha256": load_json(attempt / "plan.json")[
            "producer_binary_sha256"
        ],
        "measurement_validity": (
            "VALID_FOR_FROZEN_COMPARISON" if measurement_valid else "VOID_FATAL_GUARD"
        ),
        "coverage": {
            "expected_rows": 42,
            "observed_rows": len(rows),
            "hardware_cells": 6,
            "repetitions_per_cell": 7,
            "sample_rows": len(sample_rows),
        },
        "fatal_guards": guard_result,
        "hardware_samples": sample_rows,
        "comparisons": comparisons,
        "summary": {
            "pass_cells": pass_count,
            "miss_cells": miss_count,
            "void_cells": sum(row["verdict"] == "VOID" for row in comparisons),
            "worst_absolute_signed_relative_error": max(
                (
                    max(
                        abs(float(row["aggregate_signed_relative_error"])),
                        *(abs(float(value)) for value in row["completion_signed_relative_error_by_source"]),
                    )
                    for row in comparisons
                    if row["aggregate_signed_relative_error"] is not None
                ),
                default=None,
            ),
        },
        "scope_limits": frozen["scope_limits"],
        "preservation": {
            "verdict": "PASS",
            "artifact_count": frozen["preservation_lock"]["artifact_count"],
            "artifacts_sha256": frozen["preservation_lock"]["artifacts_sha256"],
        },
    }


def load_attempt(bulk_root: Path, *, expected_head: str) -> Path:
    root = run_campaign.cell_root(bulk_root.resolve())
    matches = []
    for attempt in sorted(root.glob("attempt-*")):
        if not run_campaign.verify_attempt(attempt):
            continue
        plan = load_json(attempt / "plan.json")
        environment = load_json(attempt / "environment.json")
        summary = load_json(attempt / "summary.json")
        if plan.get("mode") != "hardware":
            continue
        if plan.get("expected_head") != expected_head:
            continue
        if plan.get("implementation_sha256") != run_campaign.implementation_sha256():
            continue
        if environment.get("source_head") != expected_head:
            continue
        if environment.get("slurm_partition") != "a100-hourly":
            continue
        if summary.get("status") != "hardware_unscored":
            continue
        matches.append(attempt)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one digest-complete matching hardware attempt, found {len(matches)}"
        )
    return matches[0]


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid result JSON at line {line_number}") from error
            if row.get("schema") != run_campaign.OBSERVATION_SCHEMA:
                raise RuntimeError("unexpected TRAF-73 observation schema")
            if row.get("mode") != "hardware" or row.get("measurement_claim") != "unscored":
                raise RuntimeError("TRAF-73 row is not an unscored hardware observation")
            rows.append(row)
    return rows


def score_fatal_guards(
    attempt: Path,
    rows: list[dict[str, Any]],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    aggregate: dict[str, dict[str, object]] = {
        f"FG{index:02d}": {"decidable": True, "pass": True, "findings": []}
        for index in range(1, 14)
    }
    for row in rows:
        record = traf70_score.score_row_guards(row)
        for guard_id, value in record["guards"].items():
            target = aggregate[guard_id]
            target["decidable"] = bool(target["decidable"] and value["decidable"])
            target["pass"] = bool(target["pass"] and value["pass"])
            if not value["pass"]:
                target["findings"].append(
                    {"point_id": row.get("point_id"), "reason": value["reason"]}
                )
    before = (attempt / "guards_before.txt").read_text(encoding="utf-8")
    after = (attempt / "guards_after.txt").read_text(encoding="utf-8")
    topology_ok = all(
        traf70_score._gpu_list_count(text) == 4
        and traf70_score._nv4_row_count(text) == 4
        for text in (before, after)
    )
    processes_clear = all(
        not traf70_score._process_section(text).strip() for text in (before, after)
    )
    environment = load_json(attempt / "environment.json")
    allocation_ok = (
        environment.get("slurm_partition") == "a100-hourly"
        and bool(environment.get("slurm_job_id"))
    )
    aggregate["FG02"]["pass"] = bool(aggregate["FG02"]["pass"] and topology_ok)
    aggregate["FG07"]["pass"] = bool(
        aggregate["FG07"]["pass"] and processes_clear and allocation_ok
    )
    sample_rows = summarize_samples(rows, frozen, require_complete=False)
    launch_rows = []
    for sample in sample_rows:
        completion = sample["completion_us_by_source"]
        minimum_ps = min(completion) * 1_000_000
        skew_ps = (sample["degree"] - 1) * frozen["hardware_arm"]["launch_skew"][
            "per_additional_sender_budget_ps"
        ]
        fraction = skew_ps / minimum_ps
        launch_rows.append(
            {
                "degree": sample["degree"],
                "size_bytes": sample["size_bytes"],
                "launch_skew_fraction": fraction,
            }
        )
        if fraction > frozen["hardware_arm"]["launch_skew"]["negligible_fraction_high"]:
            aggregate["FG11"]["pass"] = False
            aggregate["FG11"]["findings"].append(launch_rows[-1])
    expected_keys = {
        (degree, size, repetition)
        for degree in frozen["hardware_arm"]["degrees"]
        for size in frozen["hardware_arm"]["flow_sizes_bytes"]
        for repetition in range(frozen["hardware_arm"]["repetitions_per_cell"])
    }
    observed_keys = []
    for row in rows:
        key = observation_key(row)
        observed_keys.append(key)
        if len(flow_ledgers(row)) != key[0]:
            aggregate["FG12"]["pass"] = False
            aggregate["FG12"]["findings"].append(
                {"point_id": row.get("point_id"), "reason": "source ledger count"}
            )
    matrix_ok = set(observed_keys) == expected_keys and len(observed_keys) == len(expected_keys)
    aggregate["FG12"]["pass"] = bool(aggregate["FG12"]["pass"] and matrix_ok)
    if not matrix_ok:
        aggregate["FG12"]["findings"].append({"reason": "repetition matrix mismatch"})
    try:
        run_campaign.verify_preservation(frozen)
    except RuntimeError as error:
        aggregate["FG13"]["pass"] = False
        aggregate["FG13"]["findings"].append({"reason": str(error)})
    guards = []
    for guard_id, value in aggregate.items():
        if not value["decidable"]:
            status = "UNDECIDABLE"
        elif value["pass"]:
            status = "PASS"
        else:
            status = "FAIL"
        guards.append({"id": guard_id, "status": status, **value})
    verdict = "PASS" if all(guard["status"] == "PASS" for guard in guards) else "VOID"
    return {"verdict": verdict, "guards": guards, "launch_skew_rows": launch_rows}


def observation_key(row: dict[str, Any]) -> tuple[int, int, int]:
    controls = row.get("applied_controls")
    if not isinstance(controls, dict):
        raise TypeError("TRAF-73 row has no applied controls")
    sources = [value for value in str(controls["sources"]).split(",") if value]
    size_bytes = int(row["payload_bytes"]) * int(row["message_count"])
    match = re.search(r":repeat=(\d+)$", str(row.get("point_id", "")))
    if match is None:
        raise RuntimeError("TRAF-73 point has no repetition identity")
    return len(sources), size_bytes, int(match.group(1))


def flow_ledgers(row: dict[str, Any]) -> list[dict[str, Any]]:
    values = [*row.get("latency_flow_ledger", []), *row.get("bulk_flow_ledger", [])]
    if any(not isinstance(value, dict) for value in values):
        raise RuntimeError("TRAF-73 flow ledger is malformed")
    return sorted(values, key=lambda value: int(value["source"]))


def summarize_samples(
    rows: list[dict[str, Any]],
    frozen: dict[str, Any],
    *,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    samples = []
    seen = set()
    for row in rows:
        degree, size_bytes, repetition = observation_key(row)
        key = (degree, size_bytes, repetition)
        if key in seen:
            raise RuntimeError(f"duplicate TRAF-73 observation key {key}")
        seen.add(key)
        ledgers = flow_ledgers(row)
        completions = [float(value["completion_us"]) for value in ledgers]
        if len(completions) != degree or any(value <= 0 for value in completions):
            raise RuntimeError("TRAF-73 per-flow completion ledger is incomplete")
        per_flow_goodput = [size_bytes / (value * 1000) for value in completions]
        aggregate = degree * size_bytes / (max(completions) * 1000)
        samples.append(
            {
                "degree": degree,
                "size_bytes": size_bytes,
                "repetition": repetition,
                "completion_us_by_source": completions,
                "goodput_gbps_by_source": per_flow_goodput,
                "aggregate_receiver_goodput_gbps": aggregate,
                "row_sha256": hashlib.sha256(
                    (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(
                        "utf-8"
                    )
                ).hexdigest(),
            }
        )
    samples.sort(key=lambda value: (value["size_bytes"], value["degree"], value["repetition"]))
    if require_complete:
        expected = (
            len(frozen["hardware_arm"]["degrees"])
            * len(frozen["hardware_arm"]["flow_sizes_bytes"])
            * frozen["hardware_arm"]["repetitions_per_cell"]
        )
        if len(samples) != expected:
            raise RuntimeError(f"TRAF-73 has {len(samples)} samples, expected {expected}")
    return samples


def compare_cells(
    samples: list[dict[str, Any]],
    frozen: dict[str, Any],
    *,
    measurement_valid: bool,
) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[(sample["degree"], sample["size_bytes"])].append(sample)
    prediction_by_key = {
        (row["degree"], row["size_bytes"]): row
        for row in frozen["simulation_arm"]["predictions"]
    }
    preliminary = []
    for key in sorted(grouped, key=lambda value: (value[1], value[0])):
        rows = grouped[key]
        prediction = prediction_by_key[key]
        degree, size_bytes = key
        completion_us = [
            statistics.median(row["completion_us_by_source"][source] for row in rows)
            for source in range(degree)
        ]
        goodput = [size_bytes / (value * 1000) for value in completion_us]
        aggregate = statistics.median(
            row["aggregate_receiver_goodput_gbps"] for row in rows
        )
        simulation_completion_us = [
            value / 1_000_000 for value in prediction["completion_ps_by_source"]
        ]
        completion_error = [
            (simulated - hardware) / hardware
            for simulated, hardware in zip(
                simulation_completion_us, completion_us, strict=True
            )
        ]
        simulation_aggregate = float(prediction["aggregate_payload_gbps"])
        aggregate_error = (simulation_aggregate - aggregate) / aggregate
        within = all(
            frozen["comparison"]["acceptance_low"]
            <= value
            <= frozen["comparison"]["acceptance_high"]
            for value in [aggregate_error, *completion_error]
        )
        physical_sanity = (
            "PASS"
            if all(
                prediction["physical_floor_ps"] / 1_000_000 <= value <= 1000.0
                for value in completion_us
            )
            else "FAIL"
        )
        preliminary.append(
            {
                "degree": degree,
                "size_bytes": size_bytes,
                "hardware_completion_us_by_source": completion_us,
                "hardware_goodput_gbps_by_source": goodput,
                "simulation_completion_us_by_source": simulation_completion_us,
                "completion_signed_relative_error_by_source": completion_error,
                "hardware_aggregate_gbps": aggregate,
                "simulation_aggregate_gbps": simulation_aggregate,
                "aggregate_signed_relative_error": aggregate_error,
                "physical_floor_us": prediction["physical_floor_ps"] / 1_000_000,
                "physical_ceiling_us": 1000.0,
                "physical_sanity": physical_sanity,
                "within_frozen_band": within,
                "verdict": "VOID" if not measurement_valid else "PASS" if within else "MISS",
                "responsible_parameter": "none" if within else "pending_attribution",
            }
        )
    attribution = attribute_misses(preliminary)
    for row in preliminary:
        if row["verdict"] == "MISS":
            row["responsible_parameter"] = attribution[row["degree"]]
        elif row["verdict"] == "VOID":
            row["responsible_parameter"] = "undecidable_under_void_run"
    return preliminary


def attribute_misses(rows: list[dict[str, Any]]) -> dict[int, str]:
    by_degree = {degree: sorted(
        (row for row in rows if row["degree"] == degree),
        key=lambda value: value["size_bytes"],
    ) for degree in (1, 2, 3)}
    result = {}
    for degree, cells in by_degree.items():
        if len(cells) != 2:
            raise RuntimeError(f"degree {degree} does not have both frozen sizes")
        small, large = cells
        small_error = abs(float(small["aggregate_signed_relative_error"]))
        large_error = abs(float(large["aggregate_signed_relative_error"]))
        if small_error - large_error > 0.05:
            result[degree] = "packetization"
            continue
        small_residual = statistics.median(small["hardware_completion_us_by_source"]) - statistics.median(
            small["simulation_completion_us_by_source"]
        )
        large_residual = statistics.median(large["hardware_completion_us_by_source"]) - statistics.median(
            large["simulation_completion_us_by_source"]
        )
        if abs(small_residual - large_residual) <= 1.0:
            result[degree] = "credit_round"
        else:
            result[degree] = "rx_ingress_plateau" if degree == 3 else "tx_egress_plateau"
    return result


def write_comparison_csv(path: Path, comparisons: list[dict[str, Any]]) -> None:
    rows = []
    for comparison in comparisons:
        row = {}
        for field in COMPARISON_FIELDS:
            value = comparison[field]
            if isinstance(value, list):
                row[field] = ";".join(f"{float(item):.9f}" for item in value)
            else:
                row[field] = value
        rows.append(row)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(score: dict[str, Any]) -> str:
    comparisons = score["comparisons"]
    lines = [
        "# TRAF-73 NV4 long-flow incast validation result",
        "",
        "## Hardware against simulation",
        "",
        "| Degree | Flow | Hardware aggregate GB/s | Simulation aggregate GB/s | Signed error | Hardware completion us by source | Simulation completion us by source | Verdict | Responsible parameter |",
        "|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in comparisons:
        hardware_completion = ", ".join(
            f"{value:.6f}" for value in row["hardware_completion_us_by_source"]
        )
        simulation_completion = ", ".join(
            f"{value:.6f}" for value in row["simulation_completion_us_by_source"]
        )
        lines.append(
            f"| {row['degree']} | {row['size_bytes'] // 1024} KiB | "
            f"{row['hardware_aggregate_gbps']:.6f} | "
            f"{row['simulation_aggregate_gbps']:.6f} | "
            f"{100 * row['aggregate_signed_relative_error']:+.3f}% | "
            f"{hardware_completion} | {simulation_completion} | {row['verdict']} | "
            f"`{row['responsible_parameter']}` |"
        )
    worst = score["summary"]["worst_absolute_signed_relative_error"]
    lines += [
        "",
        "Signed relative error is `(simulation - hardware) / hardware`; the frozen",
        "acceptance band is [-15%, +15%]. Per-flow goodput and all seven repetitions",
        "remain in the compact JSON record, while the table leads with the receiver",
        "aggregate and per-source completion values that decide each cell.",
        "",
        "## What ran",
        "",
        "One short exclusive `a100-hourly` cell ran the unchanged corrected TRAF-70",
        "persistent peer-write producer on one qualified four-A100 `NV4` node. It",
        "covered 256 KiB and 512 KiB flows at incast degrees 1, 2 and 3 with seven",
        "repetitions per cell. The comparison uses the six predictions frozen at",
        f"commit `{score['expectations_commit'][:7]}` before the hardware run.",
        "",
        "## What came out",
        "",
        f"The run status is **{score['status']}**. The deciding worst absolute signed",
        f"relative error is {100 * worst:.3f} percent.",
        f"{score['summary']['pass_cells']} of 6 cells pass.",
        f"{score['summary']['miss_cells']} miss; fatal guards are reported separately.",
        "and are never added to that behavioral count.",
        "",
        "## What it changes for the project",
        "",
        f"{score['registry_effect']}. The result directly tests the scored NVLink",
        "domain at the only incast degrees this node can realize.",
        "",
        "## What it does not change",
        "",
        "Degrees 4, 8 and 16 remain DECLARED SIMULATION with no hardware counterpart",
        "on an NV4 node. This result covers long flows only. Agreement at degrees 1",
        "to 3 supports but does not prove the higher-degree extrapolation, and no",
        "small-flow hardware validity claim follows from it.",
        "",
        "## Fatal guards and preservation",
        "",
        f"Fatal-guard verdict: **{score['fatal_guards']['verdict']}**.",
        f"All {score['preservation']['artifact_count']} merged study and scored source",
        "artifacts remain byte-identical. The raw capture stays outside Git; this",
        "study publishes its own compact score, comparison table and figure.",
    ]
    return "\n".join(lines) + "\n"


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8", newline="") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path.name}")
    return value


def write_json(path: Path, payload: object) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


if __name__ == "__main__":
    raise SystemExit(main())
