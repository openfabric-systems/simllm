"""Publish the compact VLLM-39 result from the immutable external raw run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
SURFACE_PATH = STUDY_DIR / "surface.json"
DEFAULT_OUTPUT = STUDY_DIR / "results.json"
RAW_RESULT_SHA256 = (
    "1521181817ac942318a6fda589b980ee8a5bf523853f19e17a2cf345652dc583"
)
RAW_RESULT_SCHEMA = "simllm-pd-session-load-delay-result-v1"
COMPACT_RESULT_SCHEMA = "simllm-pd-session-load-delay-compact-result-v1"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    return json.loads(payload), payload


def _selected_surface(surface: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_sha256": surface["record_sha256"],
        "record_device_kind_id": surface["record_device_kind_id"],
        "acceptance_status": surface["acceptance_status"],
        "calibration_claim": surface["calibration_claim"],
        "coverage": surface["coverage"],
        "interpolation": surface["interpolation"],
        "whole_record_loaded": surface["whole_record_loaded"],
        "held_out_batch_32_decoded_or_captured": surface[
            "held_out_batch_32_decoded_or_captured"
        ],
        "deepseek_row_decoded_or_captured_by_successful_reader": surface[
            "deepseek_row_decoded_or_captured_by_successful_reader"
        ],
        "points": [
            {
                "batch_size": point["batch_size"],
                "entry_key_sha256": point["entry_key_sha256"],
                "implementation_id": point["implementation_id"],
                "evidence_class": point["evidence_class"],
                "split": point["split"],
                "measured_service_ps": point["measured_service_ps"],
                "replay_count": point["replay_count"],
                "trimmed_coefficient_of_variation_ppm": point[
                    "trimmed_coefficient_of_variation_ppm"
                ],
            }
            for point in surface["points"]
        ],
    }


def compact_result(raw: dict[str, Any], raw_payload: bytes) -> dict[str, Any]:
    """Reduce request-heavy raw evidence to the complete scored public record."""

    if raw["schema"] != RAW_RESULT_SCHEMA:
        raise ValueError("raw result schema disagrees")
    raw_sha256 = _sha256_bytes(raw_payload)
    if raw_sha256 != RAW_RESULT_SHA256:
        raise ValueError(f"raw result hash disagrees: {raw_sha256}")
    provenance = raw["provenance"]
    analysis = raw["analysis"]
    observation = raw["observation"]
    surface, surface_payload = _load_json(SURFACE_PATH)
    if _sha256_bytes(surface_payload) != provenance["surface_sha256"]:
        raise ValueError("surface projection hash disagrees")
    if len(observation["curves"]) != 6 or len(observation["cells"]) != 36:
        raise ValueError("raw curve or cell registry is incomplete")
    if analysis["direction_summary"] != {
        "evaluated": 30,
        "matched": 16,
        "observed_decreases": 0,
        "observed_flats": 0,
        "observed_increases": 30,
    }:
        raise ValueError("direction verdict disagrees")
    if analysis["held_out_band_summary"] != {"evaluated": 24, "held": 1}:
        raise ValueError("held-out band verdict disagrees")
    return {
        "schema": COMPACT_RESULT_SCHEMA,
        "status": analysis["status"],
        "evidence_classification": provenance["comparator_registry"][
            "evidence_classification"
        ],
        "raw_run": {
            "sha256": raw_sha256,
            "bytes": len(raw_payload),
            "run_head": provenance["run_head"],
        },
        "freeze": {
            "commit": provenance["freeze_commit"],
            "expectations_sha256": provenance["expectations_sha256"],
        },
        "exposure": {
            "status": provenance["exposure_status"],
            "access_ledger_sha256": provenance["access_ledger_sha256"],
            **analysis["exposure_ruling"],
        },
        "surface": _selected_surface(surface),
        "fatal_guards": analysis["fatal_guards"],
        "preservation_locks": provenance["preservation_locks"],
        "core51_control": analysis["core51_control"],
        "conservation": analysis["conservation"],
        "decomposition_rows": analysis["exact_conservation_rows"],
        "curve_records": observation["curves"],
        "direction": {
            "summary": analysis["direction_summary"],
            "segment_verdicts": analysis["segment_verdicts"],
        },
        "held_out_bands": {
            "summary": analysis["held_out_band_summary"],
            "verdicts": analysis["held_out_band_verdicts"],
        },
        "knees": analysis["knees"],
        "monotonic_delay_claim": analysis["monotonic_delay_claim"],
        "measured_mechanism": analysis["measured_mechanism"],
        "task_effect": {
            "vllm_35": "open pending clean VLLM-40 qualification",
            "vllm_39": "open because the scored attempt is exposure-contaminated",
            "vllm_40": "clean field-addressed repetition",
            "vllm_41": "identify the sub-sweep scheduler queue-wait onset",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-result", required=True, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw, payload = _load_json(args.raw_result)
    compact = compact_result(raw, payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(compact, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(args.output.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
