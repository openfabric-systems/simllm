#!/usr/bin/env python3
"""Publish TRAF-67 from one previously extracted visible-row projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from traf67_clean_boundary import build_result

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
EXPECTATIONS_PATH = STUDY_DIR / "traf67_expectations.json"
TRAF66_EXPECTATIONS_PATH = STUDY_DIR / "traf66_expectations.json"
COMP75_EXPECTATIONS_PATH = STUDY_DIR / "comp75_expectations.json"
COMP75_RECORD_PATH = STUDY_DIR / "comp75_calibration_result.json"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _load_access_log(path: Path) -> list[dict[str, Any]]:
    entries = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("TRAF-67 access entry must be a JSON object")
            entries.append(value)
    return entries


def _write_new_json(path: Path, value: object) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing output: {path.name}")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--visible-row", required=True, type=Path)
    parser.add_argument("--expectations-commit", required=True)
    parser.add_argument("--publish-output", type=Path)
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        raise SystemExit("--run-dir must be the existing clean exposure directory")
    access_log = args.run_dir / "access-ledger.jsonl"
    result, event_ledger = build_result(
        _load_json(EXPECTATIONS_PATH),
        _load_json(TRAF66_EXPECTATIONS_PATH),
        _load_json(COMP75_EXPECTATIONS_PATH),
        _load_json(args.visible_row),
        _load_access_log(access_log),
        repository_root=REPOSITORY_ROOT,
        study_dir=STUDY_DIR,
        expectations_commit=args.expectations_commit,
        record_size_bytes=COMP75_RECORD_PATH.stat().st_size,
    )
    _write_new_json(args.run_dir / "event-ledger.json", event_ledger)
    _write_new_json(args.run_dir / "calibration-result.json", result)
    if args.publish_output is not None:
        _write_new_json(args.publish_output, result)
    print(
        json.dumps(
            {
                "held_out_access_count": len(
                    result["access"]["held_out_access_ledger"]
                ),
                "preservation_lock_count": result["preservation_lock"][
                    "checked_count"
                ],
                "status": result["status"],
                "visible_access_count": result["access"]["visible_access_count"],
                "visible_percentages": result["visible_percentages"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
