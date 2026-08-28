#!/usr/bin/env python3
"""Run and publish the frozen clean CORE-63 residency repetition once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core63_clean_field_reader import read_clean_inputs
from core63_clean_residency import (
    build_clean_result,
    render_markdown,
    write_new_json,
    write_new_text,
)

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
EXPECTATIONS_PATH = STUDY_DIR / "core63_clean_expectations.json"
ACCESS_LEDGER_PATH = STUDY_DIR / "core63_clean_access_ledger.jsonl"
FORBIDDEN_LEDGER_PATH = STUDY_DIR / "core63_clean_forbidden_access_ledger.json"
RESULT_JSON_PATH = STUDY_DIR / "core63_clean_calibration_result.json"
RESULT_MARKDOWN_PATH = STUDY_DIR / "core63_clean_calibration_result.md"
CLASSIFICATION_PATH = STUDY_DIR / "core63_clean_component_classification.json"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path.name} contains a non-object row")
            values.append(value)
    return values


def _require_under(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("run directory must remain under the declared bulk root") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--bulk-root", required=True, type=Path)
    parser.add_argument("--expectations-commit", required=True)
    parser.add_argument("--kernelprobe-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--runner-commit", required=True)
    args = parser.parse_args()

    if not args.bulk_root.is_dir():
        raise ValueError("bulk root must exist")
    if not args.run_dir.is_dir() or any(args.run_dir.iterdir()):
        raise ValueError("run directory must exist and be empty")
    _require_under(args.run_dir, args.bulk_root)
    for output in (
        ACCESS_LEDGER_PATH,
        FORBIDDEN_LEDGER_PATH,
        RESULT_JSON_PATH,
        RESULT_MARKDOWN_PATH,
        CLASSIFICATION_PATH,
    ):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")

    clean_expectations = _load_json(EXPECTATIONS_PATH)
    write_new_json(FORBIDDEN_LEDGER_PATH, [])
    inputs = read_clean_inputs(
        kernelprobe_root=args.kernelprobe_root,
        access_ledger=ACCESS_LEDGER_PATH,
    )
    access_events = _load_jsonl(ACCESS_LEDGER_PATH)
    result = build_clean_result(
        clean_expectations,
        inputs,
        access_events,
        repository_root=REPOSITORY_ROOT,
        expectations_commit=args.expectations_commit,
        runner_commit=args.runner_commit,
        base_commit=args.base_commit,
    )
    classification = {
        "component_classification_ledger": result["independent_recomputation"][
            "component_classification_ledger"
        ],
        "rule": clean_expectations["component_rule"],
        "schema": "simllm-deployment-curve-core63-clean-classification-v1",
        "task": "CORE-63",
    }
    rendered_json = json.dumps(result, indent=2, sort_keys=True) + "\n"
    rendered_markdown = render_markdown(result)
    rendered_classification = json.dumps(classification, indent=2, sort_keys=True) + "\n"
    write_new_text(RESULT_JSON_PATH, rendered_json)
    write_new_text(RESULT_MARKDOWN_PATH, rendered_markdown)
    write_new_text(CLASSIFICATION_PATH, rendered_classification)

    write_new_text(args.run_dir / "access-ledger.jsonl", ACCESS_LEDGER_PATH.read_text())
    write_new_text(args.run_dir / "calibration-result.json", rendered_json)
    write_new_text(args.run_dir / "calibration-result.md", rendered_markdown)
    write_new_text(args.run_dir / "component-classification.json", rendered_classification)
    write_new_json(args.run_dir / "forbidden-access-ledger.json", [])

    calibration = result["calibration_only"]
    corrected = calibration["residency_corrected"]
    movement = calibration["movement"]
    step = result["residency_derivation"]["step"]["residency_corrected_ps"]
    print(
        json.dumps(
            {
                "classification": corrected["classification"],
                "core63_entry": inputs["core63_entry"],
                "core64_entry": inputs["core64_entry"],
                "corrected_prediction_tokens_per_second_per_node": corrected[
                    "prediction_tokens_per_second_per_node"
                ],
                "corrected_signed_residual_percent": corrected[
                    "signed_residual_percent"
                ],
                "corrected_step_ps_round_half_up": step[
                    "published_ps_round_half_up"
                ],
                "forbidden_access_ledger": [],
                "prediction_movement_tokens_per_second_per_node": movement[
                    "prediction_tokens_per_second_per_node"
                ],
                "preservation_lock_count": result["preservation_lock"][
                    "checked_count"
                ],
                "signed_residual_movement_percentage_points": movement[
                    "signed_residual_percentage_points"
                ],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
