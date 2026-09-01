"""Merge one complete frozen VLLM-42 cell split and apply its bands."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

CELL_RESULT_SCHEMA = "simllm-pd-session-batching-service-cell-result-v1"
STUDY_DIR = Path(__file__).resolve().parent


def _local_module(filename: str, name: str):
    path = STUDY_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _local_module("run_study.py", "vllm42_merge_study")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_cell_documents(
    split: str,
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate and order one complete independently run disclosure split."""

    expected_keys = {
        (prefill, decode, prompt, load)
        for prefill, decode in study.POOL_RATIOS
        for prompt in study.PROMPT_LENGTHS
        for load in study.OFFERED_LOADS
        if study._cell_selected(split, prefill, decode, load)
    }
    if len(documents) != len(expected_keys):
        raise ValueError("cell result count disagrees with the frozen split")
    if any(row.get("schema") != CELL_RESULT_SCHEMA for row in documents):
        raise ValueError("cell result schema disagrees")
    if any(row.get("split") != split for row in documents):
        raise ValueError("cell disclosure split disagrees")
    keys = [study._cell_key(row["cell"]) for row in documents]
    if set(keys) != expected_keys or len(keys) != len(set(keys)):
        raise ValueError("cell result registry is incomplete or duplicated")
    provenances = [row["provenance"] for row in documents]
    if any(row != provenances[0] for row in provenances[1:]):
        raise ValueError("cell result provenance differs across shards")
    if any(row["onset_scored"] is not False for row in documents):
        raise ValueError("a cell result rescored onset")
    if any(row["monotonic_direction_scored"] is not False for row in documents):
        raise ValueError("a cell result rescored high-load direction")
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
    parser.add_argument("--split", choices=study.SPLITS, required=True)
    parser.add_argument("--cells-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"selected output directory already exists: {args.output_dir}")
    study._require_clean_worktree()
    study._validate_run_dir(args.output_dir)
    paths = sorted(args.cells_root.glob("*/cell-result.json"))
    documents = [study._load_json(path) for path in paths]
    merged = merge_cell_documents(args.split, documents)
    freeze = study._load_json(study.EXPECTATIONS_PATH)
    study._validate_freeze(freeze)
    analysis = study.analyze_observation(
        {"runtime": merged["runtime"], "split": args.split, "cells": merged["cells"]},
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
        "schema": study.RESULT_SCHEMA,
        "provenance": merged["provenance"],
        "cell_run_manifest": cell_manifest,
        "observation": {
            "runtime": merged["runtime"],
            "split": args.split,
            "cells": merged["cells"],
            "onset_scored": False,
            "monotonic_direction_scored": False,
        },
        "analysis": analysis,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    study._write_json(args.output_dir / "result.json", result)
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "split": args.split,
                "conservation": analysis["conservation"],
                "service_band_summary": analysis["service_band_summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if analysis["status"] == "VOID":
        raise SystemExit("merged VLLM-42 cells violate a fatal guard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
