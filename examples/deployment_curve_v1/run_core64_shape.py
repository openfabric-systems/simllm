#!/usr/bin/env python3
"""Run and publish the frozen CORE-64 calibration-only shape study once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core64_field_reader import read_core64_inputs
from core64_shape import (
    build_result,
    render_markdown,
    verify_preservation,
    write_new_json,
    write_new_text,
)

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
EXPECTATIONS_PATH = STUDY_DIR / "core64_expectations.json"
ACCESS_LEDGER_PATH = STUDY_DIR / "core64_access_ledger.jsonl"
FORBIDDEN_LEDGER_PATH = STUDY_DIR / "core64_forbidden_access_ledger.json"
RESULT_JSON_PATH = STUDY_DIR / "core64_shape_result.json"
RESULT_MARKDOWN_PATH = STUDY_DIR / "core64_shape_result.md"
CLASSIFICATION_PATH = STUDY_DIR / "core64_component_classification.json"


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
        RESULT_JSON_PATH,
        RESULT_MARKDOWN_PATH,
        CLASSIFICATION_PATH,
    ):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")

    expectations = _load_json(EXPECTATIONS_PATH)
    if json.loads(FORBIDDEN_LEDGER_PATH.read_text(encoding="utf-8")) != []:
        raise ValueError("forbidden-access ledger is not empty")
    inputs = read_core64_inputs(ACCESS_LEDGER_PATH)
    events = _load_jsonl(ACCESS_LEDGER_PATH)
    preservation = verify_preservation(expectations, REPOSITORY_ROOT)
    result = build_result(
        expectations,
        inputs,
        events,
        preservation,
        base_commit=args.base_commit,
        expectations_commit=args.expectations_commit,
        runner_commit=args.runner_commit,
    )
    classification = {
        "component_classification": result["component_classification"],
        "schema": "simllm-deployment-curve-core64-component-classification-v1",
        "task": "CORE-64",
    }
    rendered_json = json.dumps(result, indent=2, sort_keys=True) + "\n"
    rendered_markdown = render_markdown(result)
    rendered_classification = json.dumps(classification, indent=2, sort_keys=True) + "\n"
    write_new_text(RESULT_JSON_PATH, rendered_json)
    write_new_text(RESULT_MARKDOWN_PATH, rendered_markdown)
    write_new_text(CLASSIFICATION_PATH, rendered_classification)

    write_new_text(
        args.run_dir / "access-ledger.jsonl",
        ACCESS_LEDGER_PATH.read_text(encoding="utf-8"),
    )
    write_new_text(args.run_dir / "shape-result.json", rendered_json)
    write_new_text(args.run_dir / "shape-result.md", rendered_markdown)
    write_new_text(
        args.run_dir / "component-classification.json",
        rendered_classification,
    )
    write_new_json(args.run_dir / "forbidden-access-ledger.json", [])

    calibration = result["calibration_only"]
    print(
        json.dumps(
            {
                "final_prediction_tokens_per_second_per_node": calibration[
                    "final_prediction_tokens_per_second_per_node"
                ],
                "final_signed_residual_percent": calibration[
                    "final_signed_residual_percent"
                ],
                "forbidden_access_ledger": [],
                "prediction_movement_tokens_per_second_per_node": calibration[
                    "prediction_movement_tokens_per_second_per_node"
                ],
                "preservation_lock_count": result["preservation_lock"][
                    "checked_count"
                ],
                "shape_mismatch_count": result["per_rank_shape"][
                    "shape_mismatch_count"
                ],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
