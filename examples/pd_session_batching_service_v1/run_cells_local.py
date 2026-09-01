"""Launch one frozen VLLM-42 disclosure split across local processes."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parent
CELL_RUNNER_PATH = STUDY_DIR / "run_cell.py"


def _local_module(filename: str, name: str):
    path = STUDY_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _local_module("run_study.py", "vllm42_local_launcher_study")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=study.SPLITS, required=True)
    parser.add_argument("--cells-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, required=True)
    return parser.parse_args()


def _cell_keys(split: str) -> list[tuple[int, int, int, int]]:
    return [
        (prefill, decode, prompt, load)
        for prefill, decode in study.POOL_RATIOS
        for prompt in study.PROMPT_LENGTHS
        for load in study.OFFERED_LOADS
        if study._cell_selected(split, prefill, decode, load)
    ]


def _cell_label(key: tuple[int, int, int, int]) -> str:
    prefill, decode, prompt, load = key
    return f"p{prefill}-d{decode}-prompt{prompt}-load{load}"


def _run_cell(
    split: str,
    cells_root: Path,
    key: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], int]:
    label = _cell_label(key)
    cell_dir = cells_root / label
    log_path = cells_root / "logs" / f"{label}.log"
    command = [
        sys.executable,
        str(CELL_RUNNER_PATH),
        "--split",
        split,
        "--run-dir",
        str(cell_dir),
        "--prefill-engines",
        str(key[0]),
        "--decode-engines",
        str(key[1]),
        "--prompt-tokens",
        str(key[2]),
        "--offered-load",
        str(key[3]),
    ]
    with log_path.open("w", encoding="utf-8", newline="\n") as stream:
        completed = subprocess.run(
            command,
            cwd=study.REPOSITORY_ROOT,
            env=os.environ.copy(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return key, completed.returncode


def main() -> int:
    args = parse_args()
    if args.jobs < 1:
        raise SystemExit("jobs must be positive")
    if args.cells_root.exists():
        raise SystemExit(f"selected cells root already exists: {args.cells_root}")
    study.check_registry(args.split, args.cells_root)
    study._require_clean_worktree()
    study._validate_run_dir(args.cells_root)
    keys = _cell_keys(args.split)
    args.cells_root.mkdir(parents=True, exist_ok=False)
    (args.cells_root / "logs").mkdir()
    with ThreadPoolExecutor(max_workers=min(args.jobs, len(keys))) as pool:
        outcomes = list(
            pool.map(
                lambda key: _run_cell(args.split, args.cells_root, key),
                keys,
            )
        )
    failures = [key for key, returncode in outcomes if returncode != 0]
    if failures:
        raise SystemExit(f"local cell failures: {failures}")
    print(f"completed {len(outcomes)} {args.split} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
