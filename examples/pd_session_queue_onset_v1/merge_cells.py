"""Merge complete frozen VLLM-41 cell runs and apply the frozen analysis."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

MERGED_RESULT_SCHEMA = "simllm-pd-session-queue-onset-result-v1"
CELL_RESULT_SCHEMA = "simllm-pd-session-queue-onset-cell-result-v1"
STUDY_DIR = Path(__file__).resolve().parent


def _local_module(filename: str, name: str):
    path = STUDY_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _local_module("run_study.py", "vllm41_merge_study")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_cell_documents(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate and order one complete set of independently run cells."""

    expected_keys = {
        (prefill, decode, prompt, load)
        for prefill, decode in study.POOL_RATIOS
        for prompt in study.PROMPT_LENGTHS
        for load in study.OFFERED_LOADS
    }
    if len(documents) != len(expected_keys):
        raise ValueError("cell result count disagrees with the frozen ladder")
    if any(row.get("schema") != CELL_RESULT_SCHEMA for row in documents):
        raise ValueError("cell result schema disagrees")
    keys = [study._cell_key(row["cell"]) for row in documents]
    if set(keys) != expected_keys or len(keys) != len(set(keys)):
        raise ValueError("cell result registry is incomplete or duplicated")
    provenances = [row["provenance"] for row in documents]
    if any(row != provenances[0] for row in provenances[1:]):
        raise ValueError("cell result provenance differs across shards")
    if any(row["total_delay_direction_scored"] is not False for row in documents):
        raise ValueError("a cell result rescored total-delay direction")
    runtimes = [row["runtime"] for row in documents]
    if any(row != runtimes[0] for row in runtimes[1:]):
        raise ValueError("cell runtime identity differs across shards")
    return {
        "provenance": provenances[0],
        "runtime": runtimes[0],
        "cells": [
            row["cell"]
            for row in sorted(documents, key=lambda row: study._cell_key(row["cell"]))
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"selected output directory already exists: {args.output_dir}")
    study._require_clean_worktree()
    study._validate_run_dir(args.output_dir)
    paths = sorted(args.cells_root.glob("*/cell-result.json"))
    documents = [study._load_json(path) for path in paths]
    merged = merge_cell_documents(documents)
    freeze = study._load_json(study.EXPECTATIONS_PATH)
    study._validate_freeze(freeze)
    analysis = study.analyze_observation(
        {
            "runtime": merged["runtime"],
            "cells": merged["cells"],
            "total_delay_curves": None,
            "total_delay_direction_scored": False,
        },
        freeze,
    )
    cell_manifest = [
        {
            "path": path.relative_to(args.cells_root).as_posix(),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    ]
    result = {
        "schema": MERGED_RESULT_SCHEMA,
        "provenance": merged["provenance"],
        "cell_run_manifest": cell_manifest,
        "observation": {
            "runtime": merged["runtime"],
            "cells": merged["cells"],
            "total_delay_curves": None,
            "total_delay_direction_scored": False,
        },
        "analysis": analysis,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    study._write_json(args.output_dir / "result.json", result)
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "conservation": analysis["conservation"],
                "onset_summary": analysis["onset_summary"],
                "held_out_band_summary": analysis["held_out_band_summary"],
                "closure": analysis["closure"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if analysis["status"] == "VOID":
        raise SystemExit("merged VLLM-41 cells violate a fatal guard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
