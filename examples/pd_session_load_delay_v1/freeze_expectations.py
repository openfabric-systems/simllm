"""Build the VLLM-39 surface projection and expectations-only freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from queue_model import (
    CALIBRATION_CONFIGURATIONS,
    HANDOFF_PS,
    HELD_OUT_CONFIGURATIONS,
    MAX_BATCH_SIZE,
    OFFERED_LOADS,
    OUTPUT_TOKENS,
    POOL_RATIOS,
    PREFILL_SERVICE_PS,
    PROMPT_LENGTHS,
    REQUESTS_PER_CELL,
    decode_capacity_requests_per_second,
    fraction_json,
    predict_point,
    predicted_segments,
)

from simllm.calibration.batch_service_surface import BatchServicePoint
from simllm.calibration.canonical import canonical_sha256

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
RECORD_SHA256 = "ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52"
ACCESS_SCHEMA = "simllm-pd-session-load-delay-access-v1"
EXPECTED_READER_COMMITS = (
    "bc1cc69",
    "bd72cc7",
    "b1aab70",
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracked_paths(prefix: str) -> tuple[Path, ...]:
    completed = subprocess.run(
        ["git", "ls-files", prefix],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(
        REPOSITORY_ROOT / line
        for line in completed.stdout.splitlines()
        if line
    )


def _lock(paths: tuple[Path, ...]) -> dict[str, Any]:
    rows = [
        {
            "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(paths)
    ]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "artifact_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _preservation_locks() -> dict[str, Any]:
    core51 = _tracked_paths("examples/pd_session_v1")
    comparator = _tracked_paths("examples/pd_session_concurrent_v1")
    deployment = _tracked_paths("examples/deployment_curve_v1")
    flagship = tuple(
        path
        for path in deployment
        if path.name.startswith("flagship")
        or path.name.startswith("RESULTS")
        or "figures" in path.parts
    )
    return {
        "core51_one_request_control": _lock(core51),
        "deterministic_concurrent_comparator": _lock(comparator),
        "scored_flagship_artifacts": _lock(flagship),
        "flagship_selection_rule": "tracked deployment_curve_v1 basenames beginning flagship or RESULTS, plus every tracked figures member",
    }


def _surface(projection: dict[str, Any], consumed: int) -> dict[str, Any]:
    entries = projection["entries"]
    points = []
    for entry in entries:
        key = entry["key"]
        points.append(
            {
                "batch_size": key["shape"]["batch_size"],
                "measured_service_ps": entry["measured_service_ps"],
                "trimmed_coefficient_of_variation_ppm": entry["distribution"][
                    "trimmed_coefficient_of_variation_ppm"
                ],
                "replay_count": entry["distribution"]["replay_count"],
                "implementation_id": entry["implementation_id"],
                "entry_key": key,
                "entry_key_sha256": canonical_sha256(key),
                "evidence_class": entry["evidence"]["service_class"],
                "split": entry["evidence"]["split"],
            }
        )
    return {
        "schema": "simllm-pd-session-load-delay-surface-v1",
        "record_sha256": RECORD_SHA256,
        "acceptance_status": projection["acceptance_status"],
        "campaign_id": projection["campaign_id"],
        "coverage": projection["coverage"],
        "record_device_kind_id": projection["device_kind_id"],
        "calibration_claim": False,
        "interpolation": "power-law-linear-in-log-batch-and-log-total-service",
        "points": sorted(points, key=lambda point: point["batch_size"]),
        "record_bytes_consumed": consumed,
        "record_total_bytes": 57_417,
        "whole_record_loaded": False,
        "held_out_batch_32_decoded_or_captured": False,
        "deepseek_row_decoded_or_captured_by_successful_reader": False,
    }


def _points(surface: dict[str, Any]) -> tuple[BatchServicePoint, ...]:
    return tuple(
        BatchServicePoint(
            batch_size=row["batch_size"],
            duration_ps=row["measured_service_ps"],
            uncertainty_fraction=(
                row["trimmed_coefficient_of_variation_ppm"] / 1_000_000
            ),
            entry_key_sha256=row["entry_key_sha256"],
            evidence_class=row["evidence_class"],
            split=row["split"],
        )
        for row in surface["points"]
    )


def build_freeze(
    projection: dict[str, Any],
    access_rows: list[dict[str, Any]],
    *,
    surface_sha256: str,
    access_ledger_sha256: str,
) -> dict[str, Any]:
    """Build the complete pre-run freeze from permitted surface fields."""

    successful = [row for row in access_rows if row["status"] == "PASS"]
    consumed_values = {row["bytes_consumed_total"] for row in successful}
    if len(successful) != 5 or consumed_values != {45_043}:
        raise ValueError("successful field-access ledger disagrees")
    if sum(row["status"] == "REJECTED" for row in access_rows) != 2:
        raise ValueError("rejected field-access attempts disagree")
    if access_rows[0]["status"] != "CONTAMINATED":
        raise ValueError("pre-protocol incident is missing")
    surface = _surface(projection, consumed_values.pop())
    points = _points(surface)
    configurations = tuple(
        (prefill, decode, prompt)
        for prefill, decode in POOL_RATIOS
        for prompt in PROMPT_LENGTHS
    )
    return {
        "schema": "simllm-pd-session-load-delay-expectations-v1",
        "status": "EXPECTATIONS_ONLY",
        "task": "VLLM-39",
        "date": "2026-08-27",
        "authored_against": "b1aab70",
        "chronology": {
            "reader_commits_before_successful_access": list(
                EXPECTED_READER_COMMITS
            ),
            "successful_surface_access_existed_before_freeze": True,
            "concurrent_surface_priced_run_existed_before_freeze": False,
            "curve_values_accessed_before_freeze": False,
        },
        "exposure": {
            "status": "CONTAMINATED",
            "clean_close_permitted": False,
            "access_ledger_sha256": access_ledger_sha256,
            "access_event_count": len(access_rows),
            "successful_field_count": len(successful),
            "rejected_attempt_count": 2,
            "record_bytes_consumed": 45_043,
            "record_total_bytes": 57_417,
            "whole_record_loaded_by_successful_reader": False,
            "source_held_out_batch_32_decoded_or_captured": False,
            "successful_reader_deepseek_rows_decoded_or_captured": False,
            "pre_protocol_scan_usable_for_freeze_or_score": False,
        },
        "surface": {
            "path": "examples/pd_session_load_delay_v1/surface.json",
            "sha256": surface_sha256,
            "record_sha256": RECORD_SHA256,
            "acceptance_status": surface["acceptance_status"],
            "calibration_claim": False,
            "selected_keys": [
                {
                    "batch_size": row["batch_size"],
                    "entry_key_sha256": row["entry_key_sha256"],
                    "entry_key": row["entry_key"],
                    "evidence_class": row["evidence_class"],
                    "split": row["split"],
                    "measured_service_ps": row["measured_service_ps"],
                }
                for row in surface["points"]
            ],
        },
        "sweep": {
            "offered_load_requests_per_second": list(OFFERED_LOADS),
            "interarrival_ps": [1_000_000_000_000 // load for load in OFFERED_LOADS],
            "prompt_tokens": list(PROMPT_LENGTHS),
            "pool_ratios": [list(ratio) for ratio in POOL_RATIOS],
            "requests_per_cell": REQUESTS_PER_CELL,
            "decode_output_tokens_per_request": OUTPUT_TOKENS,
            "maximum_scheduler_batch_size": MAX_BATCH_SIZE,
            "handoff_ps": HANDOFF_PS,
            "curve_count": len(configurations),
            "point_count": len(configurations) * len(OFFERED_LOADS),
        },
        "split": {
            "calibration_configurations": [
                list(configuration) for configuration in CALIBRATION_CONFIGURATIONS
            ],
            "held_out_configurations": [
                list(configuration) for configuration in HELD_OUT_CONFIGURATIONS
            ],
            "held_out_prompt_length": 16,
            "held_out_pool_ratio": [2, 1],
        },
        "decomposition": {
            "batching_gain": "four decode steps times imported total batch service S(b), divided by b; S(b)/b must fall from batch 1 through 8",
            "scheduler_queue_wait": "arrival-to-admission wait is prefill_queue_ps plus decode_admission_wait_ps from each exact request timeline; it is reported separately from provider service",
            "queue_model": "finite D/D/c overload wait across 64 requests, max batch 8 and four decode steps; negative excess service interval is clamped to zero",
            "prefill_comparator_ps": {str(key): value for key, value in PREFILL_SERVICE_PS.items()},
            "prediction_band": "central surface-plus-queue prediction plus or minus 15 percent of decode service, one max-batch per-request service share, 25 percent of modeled queue wait and 10 percent of comparator prefill service, divided across four output tokens",
        },
        "predicted_knees_requests_per_second": {
            str(decode_engines): fraction_json(
                decode_capacity_requests_per_second(points, decode_engines)
            )
            for decode_engines in (1, 2)
        },
        "expected_segments": [
            row
            for configuration in configurations
            for row in predicted_segments(points, configuration)
        ],
        "held_out_prediction_bands": [
            predict_point(
                points,
                prefill_engines=prefill,
                decode_engines=decode,
                prompt_tokens=prompt,
                offered_load=load,
            )
            for prefill, decode, prompt in HELD_OUT_CONFIGURATIONS
            for load in OFFERED_LOADS
        ],
        "preservation_locks": _preservation_locks(),
        "decision_rule": {
            "direction": "compare every observed adjacent delay point with the pre-run expected sign; no zero tolerance or post-run sign amendment",
            "held_out_band": "PASS only when every held-out observed per-token request delay lies within its inclusive frozen band",
            "monotonic_claim": "validate only if all 30 observed segments increase; otherwise withdraw it and name batching gain below the knee plus scheduler wait above the knee",
            "task_state": "VLLM-39 remains open because exposure is contaminated; allocate VLLM-40 to a clean repetition and VLLM-41 only to any distinct model residual exposed by the run",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-projection", required=True, type=Path)
    parser.add_argument("--access-ledger", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    projection = json.loads(args.surface_projection.read_text(encoding="utf-8"))
    access_rows = [
        json.loads(line)
        for line in args.access_ledger.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if any(row.get("schema") != ACCESS_SCHEMA for row in access_rows):
        raise ValueError("access ledger schema disagrees")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    published_access = args.output_dir / "access_ledger.jsonl"
    with published_access.open("w", encoding="utf-8", newline="\n") as stream:
        for row in access_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    consumed = max(
        row["bytes_consumed_total"]
        for row in access_rows
        if row["status"] == "PASS"
    )
    surface = _surface(projection, consumed)
    surface_path = args.output_dir / "surface.json"
    _write_json(surface_path, surface)
    freeze = build_freeze(
        projection,
        access_rows,
        surface_sha256=_sha256(surface_path),
        access_ledger_sha256=_sha256(published_access),
    )
    _write_json(args.output_dir / "expectations.json", freeze)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
