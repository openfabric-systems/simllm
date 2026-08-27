"""Audit all local mock cells and render one deterministic TRAF-65 summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import run_study

FREEZE_COMMIT = "d74b123"
LOCAL_RESULT_SCHEMA = "simllm-a100-nvlink-packet-local-validation-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=run_study.LOCAL_BULK_ROOT,
        help="TRAF-65 bulk root",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    freeze = run_study._load_freeze(run_study.FREEZE_SHA256)
    cells = run_study._cells(freeze)
    root = args.root / run_study.FREEZE_SHA256 / "cells"
    cell_records = []
    total_rows = 0
    producer_counts: Counter[str] = Counter()
    aggregate = hashlib.sha256()

    for cell in cells:
        matches = []
        for attempt in sorted((root / cell.cell_id).glob("attempt-*")):
            if not run_study._verify_attempt(attempt):
                continue
            plan = _read_json(attempt / "plan.json")
            summary = _read_json(attempt / "summary.json")
            if plan.get("mode") != "mock":
                continue
            if plan.get("implementation_sha256") != run_study._implementation_digest():
                continue
            if plan.get("candidate_profile_sha256") not in (
                run_study._admissible_candidate_plan_digests()
            ):
                continue
            if summary.get("status") != "mock_complete":
                continue
            matches.append((attempt, plan, summary))
        if not matches:
            raise RuntimeError(f"no current digest-complete mock attempt for {cell.cell_id}")
        attempt, plan, summary = matches[-1]
        manifest_digest = run_study._sha256(attempt / "manifest.json")
        aggregate.update(cell.cell_id.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(manifest_digest.encode("ascii"))
        aggregate.update(b"\n")
        row_count = int(summary["row_count"])
        if row_count != int(plan["point_count"]):
            raise RuntimeError(f"row count differs from plan for {cell.cell_id}")
        total_rows += row_count
        producer_counts.update(
            {str(name): int(count) for name, count in summary["producer_counts"].items()}
        )
        cell_records.append(
            {
                "cell_id": cell.cell_id,
                "frame": cell.frame,
                "attempt": attempt.name,
                "point_count": row_count,
                "manifest_sha256": manifest_digest,
                "producer_binary_sha256": plan["producer_binary_sha256"],
            }
        )

    result = {
        "schema": LOCAL_RESULT_SCHEMA,
        "status": "VALID_LOCAL_MOCK_86_OF_86",
        "task_status": "OPEN",
        "freeze_commit": FREEZE_COMMIT,
        "freeze_sha256": run_study.FREEZE_SHA256,
        "candidate_profile_sha256": run_study._sha256(run_study.CANDIDATE_PROFILE_PATH),
        "implementation_sha256": run_study._implementation_digest(),
        "cell_count": len(cell_records),
        "isolated_cell_count": sum(cell.frame == "isolated" for cell in cells),
        "corner_frame_count": sum(cell.frame == "corner_frame" for cell in cells),
        "all_corners_frame_count": sum(cell.frame == "all_corners_frame" for cell in cells),
        "result_row_count": total_rows,
        "producer_counts": dict(sorted(producer_counts.items())),
        "aggregate_manifest_sha256": aggregate.hexdigest(),
        "measurement_claim": False,
        "hardware_remainder": {
            "status": "not_run",
            "reason": "Merlin maintenance reservation SD26082026",
            "available_after": "2026-08-28T06:30",
        },
        "cells": cell_records,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
    return 0


def _read_json(path: Path) -> dict[str, object]:
    with open(path, encoding="utf-8", newline="") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
