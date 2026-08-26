#!/usr/bin/env python3
"""Reproduce TRAF-66's event ledger and visible-only boundary comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from traf66_independent_sign import sign_visible_movement
from traf66_overlap_boundary import (
    calibration_comparison,
    compare_component_inputs,
    validate_expectations,
    verify_preservation_locks,
)

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
EXPECTATIONS_PATH = STUDY_DIR / "traf66_expectations.json"
COMP75_EXPECTATIONS_PATH = STUDY_DIR / "comp75_expectations.json"
COMP75_RESULT_PATH = STUDY_DIR / "comp75_calibration_result.json"
RUN_ROOT_ENV = "SIMLLM_TRAF66_RUN_ROOT"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _external_run_dir(name: str) -> Path:
    configured = os.environ.get(RUN_ROOT_ENV)
    if not configured:
        raise SystemExit(f"{RUN_ROOT_ENV} must name the external TRAF-66 run root")
    root = Path(configured).resolve()
    run_dir = root / name
    if run_dir.exists():
        raise SystemExit(f"selected TRAF-66 run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="visible-boundary-672afcf")
    args = parser.parse_args()

    expectations = _load_json(EXPECTATIONS_PATH)
    comp75_expectations = _load_json(COMP75_EXPECTATIONS_PATH)
    comp75_result = _load_json(COMP75_RESULT_PATH)
    validate_expectations(expectations)
    components = compare_component_inputs(expectations, comp75_expectations)
    if not all(components.values()):
        raise SystemExit("TRAF-66 component reuse disagrees")
    preservation = verify_preservation_locks(expectations, REPOSITORY_ROOT)
    row = calibration_comparison(expectations, comp75_result)
    composition = expectations["composition"]
    independent = sign_visible_movement(
        per_node_tokens=row["per_node_tokens"],
        published_numerator=row["published"]["numerator"],
        published_denominator=row["published"]["denominator"],
        compute_service_ps=composition["candidate_compute_service_ps"],
        packet_service_ps=composition["packet_service_ps"]["selected"],
        children=expectations["event_conservation"]["counts"]["children"],
    )
    if independent["movement"]["direction"] != "decrease":
        raise SystemExit("TRAF-66 independent movement sign disagrees")

    run_dir = _external_run_dir(args.run_name)
    _write_json(
        run_dir / "event-ledger.json",
        {
            "component_reuse": components,
            "event_conservation": expectations["event_conservation"],
            "preservation_lock": preservation,
            "source_contracts": expectations["source_contracts"],
        },
    )
    _write_json(
        run_dir / "calibration-result.json",
        {
            "calibration_rows": [row],
            "held_out_access_ledger": expectations["calibration_split"][
                "held_out_access_ledger"
            ],
            "independent_signoff": independent,
            "scored_flagship_rerun_performed": False,
            "status": "PROTOCOL_VOID_HELD_OUT_COMPONENT_ACCESS",
        },
    )
    print(run_dir.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
