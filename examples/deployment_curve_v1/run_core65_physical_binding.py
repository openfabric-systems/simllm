#!/usr/bin/env python3
"""Publish the CORE-65 total inventory, null result, and CORE-66 remainder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core65_physical_binding import (
    build_result,
    render_hardware_remainder,
    render_markdown,
    verify_preservation,
    write_new_json,
    write_new_text,
)

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
EXPECTATIONS_PATH = STUDY_DIR / "core65_expectations.json"
FORBIDDEN_PATH = STUDY_DIR / "core65_forbidden_access_ledger.json"
ACCESS_PATH = STUDY_DIR / "core65_access_ledger.jsonl"
CAPTURE_ACCESS_PATH = STUDY_DIR / "core65_capture_profile_access_ledger.jsonl"
INVENTORY_PATH = STUDY_DIR / "core65_kernel_inventory.json"
RESULT_JSON_PATH = STUDY_DIR / "core65_physical_binding_result.json"
RESULT_MARKDOWN_PATH = STUDY_DIR / "core65_physical_binding_result.md"
REMAINDER_JSON_PATH = STUDY_DIR / "core66_hardware_remainder.json"
REMAINDER_MARKDOWN_PATH = STUDY_DIR / "core66_hardware_remainder.md"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    except ValueError as error:
        raise ValueError("path must remain under the declared bulk root") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk-root", required=True, type=Path)
    parser.add_argument("--expectations-commit", required=True)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--runner-commit", required=True)
    parser.add_argument("--core66-free-on-base-main", action="store_true")
    args = parser.parse_args()

    for path in (args.input_dir, args.run_dir):
        _require_under(path, args.bulk_root)
    if not args.input_dir.is_dir():
        raise ValueError("input directory must exist")
    if not args.run_dir.is_dir() or any(args.run_dir.iterdir()):
        raise ValueError("run directory must exist and be empty")
    outputs = (
        ACCESS_PATH,
        CAPTURE_ACCESS_PATH,
        INVENTORY_PATH,
        RESULT_JSON_PATH,
        RESULT_MARKDOWN_PATH,
        REMAINDER_JSON_PATH,
        REMAINDER_MARKDOWN_PATH,
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite a CORE-65 publication")

    expectations = _load_json(EXPECTATIONS_PATH)
    selected = _load_json(args.input_dir / "selected-inputs.json")
    access_events = _load_jsonl(args.input_dir / "access-ledger.jsonl")
    capture_events = _load_jsonl(args.input_dir / "capture-profile-access-ledger.jsonl")
    forbidden = _load_json(FORBIDDEN_PATH)
    preservation = verify_preservation(expectations, REPOSITORY_ROOT)
    inventory, result, remainder = build_result(
        expectations=expectations,
        selected=selected,
        access_events=access_events,
        capture_events=capture_events,
        forbidden_access_ledger=forbidden,
        preservation=preservation,
        expectations_commit=args.expectations_commit,
        runner_commit=args.runner_commit,
        core66_free_on_base_main=args.core66_free_on_base_main,
    )
    result_markdown = render_markdown(result, inventory)
    remainder_markdown = render_hardware_remainder(remainder)

    write_new_text(
        ACCESS_PATH,
        (args.input_dir / "access-ledger.jsonl").read_text(encoding="utf-8"),
    )
    write_new_text(
        CAPTURE_ACCESS_PATH,
        (args.input_dir / "capture-profile-access-ledger.jsonl").read_text(encoding="utf-8"),
    )
    write_new_json(INVENTORY_PATH, inventory)
    write_new_json(RESULT_JSON_PATH, result)
    write_new_text(RESULT_MARKDOWN_PATH, result_markdown)
    write_new_json(REMAINDER_JSON_PATH, remainder)
    write_new_text(REMAINDER_MARKDOWN_PATH, remainder_markdown)

    write_new_json(args.run_dir / "kernel-inventory.json", inventory)
    write_new_json(args.run_dir / "physical-binding-result.json", result)
    write_new_text(args.run_dir / "physical-binding-result.md", result_markdown)
    write_new_json(args.run_dir / "core66-hardware-remainder.json", remainder)
    write_new_text(args.run_dir / "core66-hardware-remainder.md", remainder_markdown)
    write_new_json(args.run_dir / "forbidden-access-ledger.json", forbidden)
    write_new_text(
        args.run_dir / "access-ledger.jsonl",
        (args.input_dir / "access-ledger.jsonl").read_text(encoding="utf-8"),
    )
    write_new_text(
        args.run_dir / "capture-profile-access-ledger.jsonl",
        (args.input_dir / "capture-profile-access-ledger.jsonl").read_text(encoding="utf-8"),
    )

    calibration = result["calibration_only"]
    print(
        json.dumps(
            {
                "inventory_coverage": result["physical_binding"]["coverage"],
                "prediction_movement_tokens_per_second_per_node": calibration[
                    "prediction_movement_tokens_per_second_per_node"
                ],
                "preservation_lock_count": preservation["checked_count"],
                "registered_remainder": "CORE-66",
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
