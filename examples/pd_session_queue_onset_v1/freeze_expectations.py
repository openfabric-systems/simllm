"""Build the expectations-only VLLM-41 lower-load freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from queue_model import (
    HELD_OUT_LOADS,
    HELD_OUT_POOL_RATIOS,
    MAX_BATCH_SIZE,
    OFFERED_LOADS,
    OUTPUT_TOKENS,
    POOL_RATIOS,
    PROMPT_LENGTHS,
    PS_PER_SECOND,
    REQUESTS_PER_CELL,
    SURFACE_SCENARIOS,
    THREE_SIGMA_MULTIPLIER,
    first_queue_dominated_segment,
    held_out_points,
    onset_rate_band,
    predicted_onset_segments,
    predicted_segments,
    surface_cv_envelope_ppm,
)

from simllm.calibration.batch_service_surface import BatchServicePoint
from simllm.calibration.canonical import canonical_sha256

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
REFERENCE_STUDY_DIR = REPOSITORY_ROOT / "examples" / "pd_session_load_delay_v1"
REFERENCE_SURFACE_PATH = REFERENCE_STUDY_DIR / "surface.json"
FIELD_READER_PATH = REFERENCE_STUDY_DIR / "field_reader.py"
RECORD_SHA256 = "ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52"
REFERENCE_SURFACE_SHA256 = (
    "26fc547d8b47ccec7108872e05fbedfe71ebb6229b88799ca254089d3f2b6e9d"
)
ACCESS_SCHEMA = "simllm-pd-session-load-delay-access-v1"
AUTHORED_AGAINST = "b602d0d19ac809ddb3eed19b509ada56318ee7e2"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob_sha1(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def _tracked_paths(prefix: str) -> tuple[Path, ...]:
    completed = subprocess.run(
        ["git", "ls-files", prefix],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(
        REPOSITORY_ROOT / line for line in completed.stdout.splitlines() if line
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
        "selection": [row["path"] for row in rows],
    }


def preservation_locks() -> dict[str, Any]:
    """Lock prior load-delay records and the three inherited guard classes."""

    deployment = _tracked_paths("examples/deployment_curve_v1")
    flagship = tuple(
        path
        for path in deployment
        if path.name.startswith("flagship")
        or path.name.startswith("RESULTS")
        or "figures" in path.parts
    )
    return {
        "prior_load_delay_lineage": _lock(
            _tracked_paths("examples/pd_session_load_delay_v1")
        ),
        "core51_one_request_control": _lock(
            _tracked_paths("examples/pd_session_v1")
        ),
        "deterministic_concurrent_comparator": _lock(
            _tracked_paths("examples/pd_session_concurrent_v1")
        ),
        "scored_flagship_artifacts": _lock(flagship),
        "flagship_selection_rule": "tracked deployment_curve_v1 basenames beginning flagship or RESULTS, plus every tracked figures member",
    }


def _surface_projection(projection: dict[str, Any], consumed: int) -> dict[str, Any]:
    points = []
    for entry in projection["entries"]:
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


def _validate_access(
    projection: dict[str, Any], access_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], int]:
    if len(access_rows) != 5 or any(
        row.get("schema") != ACCESS_SCHEMA for row in access_rows
    ):
        raise ValueError("fresh field-access ledger must contain five schema rows")
    if any(row.get("status") != "PASS" for row in access_rows):
        raise ValueError("fresh field-access ledger contains a non-pass row")
    if any(row.get("whole_record_loaded") is not False for row in access_rows):
        raise ValueError("fresh field-access ledger loaded the whole record")
    consumed_values = {row["bytes_consumed_total"] for row in access_rows}
    if consumed_values != {45_043}:
        raise ValueError("fresh field-access byte boundary disagrees")
    surface = _surface_projection(projection, consumed_values.pop())
    payload = (json.dumps(surface, indent=2, sort_keys=True) + "\n").encode()
    if hashlib.sha256(payload).hexdigest() != REFERENCE_SURFACE_SHA256:
        raise ValueError("fresh field projection changed the imported surface")
    if payload != REFERENCE_SURFACE_PATH.read_bytes():
        raise ValueError("fresh field projection is not byte-identical to the import")
    if surface["acceptance_status"] != "candidate":
        raise ValueError("candidate status changed")
    if surface["calibration_claim"] is not False:
        raise ValueError("candidate import cannot make a calibration claim")
    return surface, 45_043


def build_freeze(
    projection: dict[str, Any],
    access_rows: list[dict[str, Any]],
    *,
    access_ledger_sha256: str,
    projection_sha256: str,
) -> dict[str, Any]:
    """Build the pre-run ladder, model bands, splits, and decision rules."""

    surface, consumed = _validate_access(projection, access_rows)
    points = _points(surface)
    configurations = tuple(
        (prefill, decode, prompt)
        for prefill, decode in POOL_RATIOS
        for prompt in PROMPT_LENGTHS
    )
    scenario_onsets = [
        {
            "configuration": list(configuration),
            "segments": {
                scenario: (
                    None
                    if (
                        segment := first_queue_dominated_segment(
                            points, configuration, scenario=scenario
                        )
                    )
                    is None
                    else list(segment)
                )
                for scenario in SURFACE_SCENARIOS
            },
        }
        for configuration in configurations
    ]
    return {
        "schema": "simllm-pd-session-queue-onset-expectations-v1",
        "status": "EXPECTATIONS_ONLY",
        "task": "VLLM-41",
        "date": "2026-08-27",
        "authored_against": AUTHORED_AGAINST,
        "chronology": {
            "field_reader_committed_before_access": True,
            "fresh_surface_access_existed_before_freeze": True,
            "lower_ladder_run_existed_before_freeze": False,
            "vllm41_curve_values_accessed_before_freeze": False,
            "observed_curve_fit_permitted": False,
        },
        "source_access": {
            "status": "CLEAN",
            "field_reader_path": FIELD_READER_PATH.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "field_reader_sha256": _sha256(FIELD_READER_PATH),
            "field_reader_git_blob_sha1": _git_blob_sha1(FIELD_READER_PATH),
            "access_ledger_path": "examples/pd_session_queue_onset_v1/access_ledger.jsonl",
            "access_ledger_sha256": access_ledger_sha256,
            "projection_sha256": projection_sha256,
            "successful_field_count": len(access_rows),
            "record_bytes_consumed": consumed,
            "record_total_bytes": surface["record_total_bytes"],
            "whole_record_loaded": False,
            "held_out_batch_32_decoded_or_captured": False,
            "deepseek_row_decoded_or_captured": False,
        },
        "surface": {
            "path": REFERENCE_SURFACE_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": REFERENCE_SURFACE_SHA256,
            "record_sha256": RECORD_SHA256,
            "acceptance_status": surface["acceptance_status"],
            "calibration_claim": False,
            "interpolation": surface["interpolation"],
            "selected_keys": [
                {
                    "batch_size": row["batch_size"],
                    "entry_key_sha256": row["entry_key_sha256"],
                    "evidence_class": row["evidence_class"],
                    "split": row["split"],
                    "measured_service_ps": row["measured_service_ps"],
                    "trimmed_coefficient_of_variation_ppm": row[
                        "trimmed_coefficient_of_variation_ppm"
                    ],
                    "replay_count": row["replay_count"],
                }
                for row in surface["points"]
            ],
        },
        "sweep": {
            "offered_load_requests_per_second": list(OFFERED_LOADS),
            "interarrival_ps": [PS_PER_SECOND // load for load in OFFERED_LOADS],
            "prompt_tokens": list(PROMPT_LENGTHS),
            "pool_ratios": [list(ratio) for ratio in POOL_RATIOS],
            "requests_per_cell": REQUESTS_PER_CELL,
            "decode_output_tokens_per_request": OUTPUT_TOKENS,
            "maximum_scheduler_batch_size": MAX_BATCH_SIZE,
            "curve_count": len(configurations),
            "point_count": len(configurations) * len(OFFERED_LOADS),
            "minimum_load_is_below_vllm39": min(OFFERED_LOADS) < 250,
            "maximum_load_repeats_vllm39_boundary_only": max(OFFERED_LOADS) == 250,
        },
        "held_out": {
            "loads": list(HELD_OUT_LOADS),
            "pool_ratios": [list(ratio) for ratio in HELD_OUT_POOL_RATIOS],
            "union_rule": "score a point as held out when its load or pool ratio is held out",
            "point_count": len(held_out_points(points)),
            "prediction_bands": held_out_points(points),
        },
        "queue_model": {
            "name": "deterministic-shared-clock-round-robin-bulk-service-v1",
            "numerical_inputs": [
                "logged imported batch-1 and batch-8 measured service and CV fields",
                "frozen deterministic interarrival process",
            ],
            "observed_curve_inputs": [],
            "fit_parameters": [],
            "driver_structure": "one shared virtual clock chooses one nonempty prefill or decode engine in work-conserving round-robin order; decode requests retain four visits and each engine batches at most eight",
            "boundary_abstraction": "prefill and handoff service are zero-cost phase boundaries in the numerical model; no comparator or observed delay enters the prediction",
            "surface_uncertainty": {
                "rule": "scale every interpolated batch service by plus or minus three times the maximum imported trimmed CV",
                "multiplier": THREE_SIGMA_MULTIPLIER,
                "envelope_ppm": surface_cv_envelope_ppm(points),
            },
            "wait_band": "take the min and max modeled wait across the surface scenarios, then add at most one upper batch-8 service residual at each of the two admission boundaries",
            "queue_dominated_segment": "mean scheduler-wait increase divided by four exceeds any batching-service-per-token reduction, so their separated component sum increases",
            "isolated_onset_rate_band_requests_per_second": onset_rate_band(points),
            "first_queue_dominated_segment_prediction": predicted_onset_segments(
                points
            ),
            "scenario_onsets": scenario_onsets,
        },
        "decomposition": {
            "batching_service_per_token": "sum imported S(batch) over every stock-scheduler decode batch, divided by the number of scheduled request-token visits",
            "arrival_to_prefill_wait": "mean of prefill_eligible_at_ps minus admitted_at_ps over the 64 exact request timelines",
            "handoff_to_decode_admission_wait": "mean of decode_eligible_at_ps minus handoff.completed_at_ps over the same timelines",
            "scheduler_queue_wait": "arrival-to-prefill wait plus handoff-to-decode admission wait; never add provider service to this field",
            "segment_comparison": "divide the scheduler-wait delta by four output tokens, retain batching service per token separately, and mark queue-dominated only when the wait delta is positive and the separated component sum rises",
        },
        "expected_segments": [
            row
            for configuration in configurations
            for row in predicted_segments(points, configuration)
        ],
        "preservation_locks": preservation_locks(),
        "decision_rule": {
            "no_monotonic_rescore": "do not compute or score total per-token delay direction; VLLM-40 already validated 250 to 8,000 requests/s",
            "held_out_band": "compare observed queue wait and batching service separately with each inclusive frozen component band; publish every miss without widening",
            "onset": "for each of six configurations, report the first queue-dominated adjacent segment exactly as observed",
            "vllm41_close": "close only if all six configurations have a first queue-dominated segment strictly below 250 and at least one preceding non-queue-dominated segment",
            "vllm42_residual": "register any held-out quantitative component-band miss on VLLM-42",
            "vllm43_residual": "register differing first-onset segments across pool ratios or prompt lengths, or any configuration without a resolved onset, on VLLM-43",
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    published_access = args.output_dir / "access_ledger.jsonl"
    with published_access.open("w", encoding="utf-8", newline="\n") as stream:
        for row in access_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    freeze = build_freeze(
        projection,
        access_rows,
        access_ledger_sha256=_sha256(published_access),
        projection_sha256=_sha256(args.surface_projection),
    )
    _write_json(args.output_dir / "expectations.json", freeze)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
