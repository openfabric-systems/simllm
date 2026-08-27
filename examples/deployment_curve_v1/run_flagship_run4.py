#!/usr/bin/env python3
"""Run the frozen fourth CORE-54 scored MTP anchor exactly once."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path, PurePath
from typing import Any

from flagship_run4_field_reader import (
    ANCHOR_FREEZE,
    RUN3_PUBLICATION,
    read_mtp_anchor,
    read_run3_publication,
)
from flagship_run4_tools import (
    build_result,
    build_shape_observation,
    score_mtp_anchor,
    sha256,
    validate_config,
    validate_expectations,
    verify_preservation_lock,
    write_json,
)

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
EXPECTATIONS_PATH = STUDY_DIR / "scored_run4_expectations.json"
CONFIG_PATH = STUDY_DIR / "flagship_run4_config.json"
EXPECTATIONS_COMMIT = "80088f4b6ca31239c2a6c55d966643dcfaf408cc"
RUN_ROOT_ENV = "SIMLLM_CORE54RUN4_RUN_ROOT"


def render_cli_path(path: PurePath) -> str:
    """Render paths with POSIX separators on every platform."""

    return path.as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _require_clean_tracked_worktree() -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip():
        raise SystemExit("the fourth scored flagship requires a clean tracked worktree")


def _require_expectations_ancestor() -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTATIONS_COMMIT, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if completed.returncode:
        raise SystemExit("the fourth-run expectations commit is not an ancestor")


def _load_access(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("run-4 access row must be an object")
            rows.append(value)
    return rows


def check_registry() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    """Validate the freeze, configuration and preservation class."""

    frozen = _load_json(EXPECTATIONS_PATH)
    config = _load_json(CONFIG_PATH)
    validate_expectations(frozen)
    validate_config(config, frozen)
    _require_expectations_ancestor()
    preservation = verify_preservation_lock(REPOSITORY_ROOT, frozen)
    return frozen, config, preservation


def run_once(
    run_dir: Path,
    access_log: Path,
    frozen: dict[str, Any],
    config: dict[str, Any],
    preservation: list[dict[str, str]],
) -> dict[str, Any]:
    """Serialize the prediction, realize the shape, read once and score once."""

    run_dir.mkdir(parents=True, exist_ok=False)
    prediction_path = run_dir / "frozen-prediction.json"
    write_json(
        prediction_path,
        {
            "schema": "simllm-deployment-curve-run4-addressed-prediction-v1",
            "fit": frozen["fit_rule"],
            "prediction": frozen["pre_fit_prediction_layers"][0],
        },
    )
    prediction_sha256 = sha256(prediction_path)
    write_json(run_dir / "inherited-fit.json", frozen["fit_rule"])

    run3 = read_run3_publication(RUN3_PUBLICATION, access_log)
    write_json(run_dir / "run3-carry-forward.json", run3)
    shape = build_shape_observation(config, frozen)
    write_json(run_dir / "mtp-shape-observation.json", shape)

    anchor = read_mtp_anchor(ANCHOR_FREEZE, access_log)
    score = score_mtp_anchor(anchor, frozen, prediction_sha256=prediction_sha256)
    write_json(run_dir / "held-out-score.json", score)
    access = _load_access(access_log)
    result = build_result(frozen, config, run3, shape, score, preservation, access)
    result["provenance"] = {
        "expectations_commit": EXPECTATIONS_COMMIT,
        "expectations_sha256": sha256(EXPECTATIONS_PATH),
        "config_sha256": sha256(CONFIG_PATH),
        "prediction_sha256": prediction_sha256,
        "run3_publication_sha256": sha256(RUN3_PUBLICATION),
        "anchor_record_sha256": sha256(ANCHOR_FREEZE),
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "model_weights_loaded": False,
        "web_pages_fetched": False,
    }
    write_json(run_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frozen, config, preservation = check_registry()
    if args.check_only:
        print(
            "check-only: fourth freeze, exact MTP shape, inherited fit and 57 "
            "preservation locks passed; no run-3 result or anchor was read"
        )
        return 0
    if args.run_dir is None:
        raise SystemExit("--run-dir is required for the scored attempt")
    configured = os.environ.get(RUN_ROOT_ENV)
    if not configured:
        raise SystemExit(f"{RUN_ROOT_ENV} must name the external run root")
    root = Path(configured).resolve()
    if root.parts[-2:] != ("wave-runs", "core54run4"):
        raise SystemExit(f"{RUN_ROOT_ENV} must name the requested external run root")
    try:
        args.run_dir.resolve().relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"run directory must remain under {RUN_ROOT_ENV}") from exc
    if args.run_dir.exists():
        raise SystemExit(f"selected run directory already exists: {args.run_dir}")
    if sys.version_info[:3] != (3, 10, 18):
        raise SystemExit("the fourth flagship run requires Python 3.10.18")
    _require_clean_tracked_worktree()
    access_log = root / "access.jsonl"
    result = run_once(args.run_dir, access_log, frozen, config, preservation)
    error = result["mtp_score"]["layers"][
        "physics_plus_boundary_plus_attenuation"
    ]["absolute_relative_error"]
    print(
        f"{result['verdict']}: mtp_absolute_error="
        f"{error['numerator']}/{error['denominator']}; wrote "
        f"{render_cli_path(args.run_dir / 'result.json')}"
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
