"""Run one frozen VLLM-41 cell for parallel external orchestration."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

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


study = _local_module("run_study.py", "vllm41_cell_study")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--prefill-engines", required=True, type=int)
    parser.add_argument("--decode-engines", required=True, type=int)
    parser.add_argument("--prompt-tokens", required=True, type=int)
    parser.add_argument("--offered-load", required=True, type=int)
    return parser.parse_args()


def _validate_cell(args: argparse.Namespace) -> tuple[int, int, int, int]:
    key = (
        args.prefill_engines,
        args.decode_engines,
        args.prompt_tokens,
        args.offered_load,
    )
    if key[:2] not in study.POOL_RATIOS:
        raise SystemExit("cell pool ratio is outside the frozen ladder")
    if key[2] not in study.PROMPT_LENGTHS:
        raise SystemExit("cell prompt length is outside the frozen ladder")
    if key[3] not in study.OFFERED_LOADS:
        raise SystemExit("cell offered load is outside the frozen ladder")
    return key


def main() -> int:
    args = parse_args()
    key = _validate_cell(args)
    provenance = study.check_registry(args)
    study._require_clean_worktree()
    study._validate_run_dir(args.run_dir)
    vllm_version = study._validate_runtime()
    args.run_dir.mkdir(parents=True, exist_ok=False)

    from simllm.adapters.vllm.pd_session import VllmDisaggregatedSession

    base = study._base_runner_module()
    prompt = base._prompt_tokens()
    surface = study._load_json(study.SURFACE_PATH)
    points = study._surface_points(surface)
    provider = base._surface_provider(surface)
    with VllmDisaggregatedSession(
        base._session_config(
            args.run_dir,
            prefill_engines=key[0],
            decode_engines=key[1],
            decode_provider=provider,
        )
    ) as session:
        cell = study._cell_observation(
            session,
            prompt=prompt,
            prompt_tokens=key[2],
            offered_load=key[3],
            prefill_engines=key[0],
            decode_engines=key[1],
            points=points,
        )
    result = {
        "schema": CELL_RESULT_SCHEMA,
        "provenance": provenance,
        "runtime": {
            "python": study.sys.version.split()[0],
            "vllm": vllm_version,
            "offline": True,
        },
        "cell": cell,
        "total_delay_curve": None,
        "total_delay_direction_scored": False,
    }
    study._write_json(args.run_dir / "cell-result.json", result)
    print(json.dumps({"cell": list(key), "status": "COMPLETE"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
