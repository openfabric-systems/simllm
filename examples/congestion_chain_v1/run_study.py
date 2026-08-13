"""Run the frozen BACK-38 and BRIDGE-2 congestion-chain study.

The expectations-only version of this file implements only the artifact-free
``--check-only`` gate. The result-producing harness lands after the freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

SOURCE_FILES = {
    "cells/a-ep8-100g/steps.jsonl": (
        "893fd939460556a6e0639572ef41db58b478c72850f3a43b62de626bbade5706"
    ),
    "cells/a-ep8-100g/replay-run.json": (
        "61109794360613d04ffbc4ed4a2e4eee6fb06adbd005db68d50b84961bea8ef3"
    ),
    "cells/a-ep8-100g/routed-experts.json": (
        "2c3e5c961ed432a6eb9633983d4ccedc2cd2c26ccb24415f9f4092fac7b58dba"
    ),
    "capture/greedy.jsonl": ("ef570a67fd8bbbb6a8d73b8ad9f73171d3eaaf9a51efc04f676bd9f25c8988fe"),
}
SOURCE_CELL = "cells/a-ep8-100g/cell.json"
SOURCE_GOAL_ROOT = "cells/a-ep8-100g/htsim"
SOURCE_STEP_COUNT = 36
SOURCE_GOAL_COUNT = 1728
SOURCE_ROUTED_BYTES = 50_802_688
SOURCE_PROCESS_COUNT = 1728
SOURCE_WALL_SECONDS = 600.23
SOURCE_REQUEST_IDS = ("r00", "r01", "r02")
SOURCE_TTFT_PS = (903_795_888, 936_896_110, 1_161_066_114)
SOURCE_TPOT_PS = (
    Fraction(7_397_037_050, 23),
    Fraction(612_085_002, 1),
    Fraction(7_934_142_022, 31),
)

PROFILE = "rnic-cn"
CELLS = (
    ("cn-100g-a", 100_000_000_000),
    ("cn-100g-b", 100_000_000_000),
    ("cn-200g", 200_000_000_000),
)
PROMPT_TOKENS = (19, 15, 18)
NUM_LAYERS = 24
PHASES_PER_LAYER = 2
VECTOR_BYTES = 2048
MIN_REMOTE_DESTINATIONS = 1
MAX_REMOTE_DESTINATIONS = 7
MAX_FORWARDED_TOKENS = 108
MIN_COMPUTE_PS = 60_000_000
MAX_COMPUTE_PS = 200_000_000
ARRIVAL_SPAN_PS = 2_000_000_000
TTFT_FLOOR_PS = 177_964_800
END_TO_END_CEILING_PS = 250_000_000_000
EXPECTED_BEHAVIORAL_INSTANCES = 7


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(value: Any, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise SystemExit(f"{name} must be a JSON object")
    return value


def _nearest_rank(values: tuple[int, ...], numerator: int, denominator: int) -> int:
    if not values:
        raise ValueError("nearest-rank input must not be empty")
    ordered = sorted(values)
    one_based = (numerator * len(ordered) + denominator - 1) // denominator
    return ordered[max(1, one_based) - 1]


def _validate_source(args: argparse.Namespace) -> None:
    for relative, expected in SOURCE_FILES.items():
        path = args.mission_run / relative
        if not path.is_file():
            raise SystemExit(f"missing frozen source artifact: {path}")
        observed = _sha256(path)
        if observed != expected:
            raise SystemExit(f"frozen source artifact digest changed for {relative}: {observed}")

    step_path = args.mission_run / "cells/a-ep8-100g/steps.jsonl"
    step_count = sum(1 for line in step_path.read_text(encoding="utf-8").splitlines() if line)
    if step_count != SOURCE_STEP_COUNT:
        raise SystemExit(
            f"frozen step count changed: expected {SOURCE_STEP_COUNT}, got {step_count}"
        )

    goal_root = args.mission_run / SOURCE_GOAL_ROOT
    goal_count = sum(1 for path in goal_root.rglob("*.goal") if path.is_file())
    if goal_count != SOURCE_GOAL_COUNT:
        raise SystemExit(
            f"frozen GOAL count changed: expected {SOURCE_GOAL_COUNT}, got {goal_count}"
        )

    cell_path = args.mission_run / SOURCE_CELL
    if not cell_path.is_file():
        raise SystemExit(f"missing frozen source cell: {cell_path}")
    cell = _object(json.loads(cell_path.read_text(encoding="utf-8")), SOURCE_CELL)
    required = {
        "bandwidth_bps": 100_000_000_000,
        "ep_world": 8,
        "scheduler_steps": SOURCE_STEP_COUNT,
        "simulated_steps": SOURCE_STEP_COUNT,
        "htsim_invocations": SOURCE_PROCESS_COUNT,
        "total_routed_bytes": SOURCE_ROUTED_BYTES,
        "wall_seconds": SOURCE_WALL_SECONDS,
    }
    for name, expected in required.items():
        if cell.get(name) != expected:
            raise SystemExit(
                f"frozen source cell {name} changed: expected {expected!r}, got {cell.get(name)!r}"
            )

    requests = cell.get("requests")
    if type(requests) is not list or len(requests) != len(SOURCE_REQUEST_IDS):
        raise SystemExit("frozen source request rows changed cardinality")
    observed_ids = tuple(row.get("request_id") for row in requests)
    observed_ttft = tuple(row.get("ttft_ps") for row in requests)
    observed_tpot = tuple(
        Fraction(row["tpot_numerator"], row["tpot_denominator"]) for row in requests
    )
    if observed_ids != SOURCE_REQUEST_IDS:
        raise SystemExit(f"frozen request order changed: {observed_ids}")
    if observed_ttft != SOURCE_TTFT_PS:
        raise SystemExit(f"frozen TTFT rows changed: {observed_ttft}")
    if observed_tpot != SOURCE_TPOT_PS:
        raise SystemExit(f"frozen TPOT rows changed: {observed_tpot}")


def check_only(args: argparse.Namespace) -> None:
    """Validate frozen inputs and arithmetic without producing artifacts."""

    _validate_source(args)
    if not args.htsim_rnic.is_file() or not os.access(args.htsim_rnic, os.X_OK):
        raise SystemExit(f"htsim_rnic is missing or not executable: {args.htsim_rnic}")
    if args.run_dir.exists():
        raise SystemExit(f"run directory already exists: {args.run_dir}")

    if len({name for name, _ in CELLS}) != len(CELLS):
        raise AssertionError("cell names must be unique")
    if tuple(rate for _, rate in CELLS) != (
        100_000_000_000,
        100_000_000_000,
        200_000_000_000,
    ):
        raise AssertionError("frozen link-rate cells changed")
    if _nearest_rank(SOURCE_TTFT_PS, 50, 100) != 936_896_110:
        raise AssertionError("frozen fluid TTFT p50 changed")
    if _nearest_rank(SOURCE_TTFT_PS, 99, 100) != 1_161_066_114:
        raise AssertionError("frozen fluid TTFT p99 changed")

    byte_time_ps = 8_000_000_000_000 // CELLS[0][1]
    if byte_time_ps != 80:
        raise AssertionError("100 Gbit/s byte time changed")
    smallest_prompt_floor = (
        min(PROMPT_TOKENS)
        * NUM_LAYERS
        * PHASES_PER_LAYER
        * MIN_REMOTE_DESTINATIONS
        * VECTOR_BYTES
        * byte_time_ps
    )
    if smallest_prompt_floor + MIN_COMPUTE_PS != TTFT_FLOOR_PS:
        raise AssertionError("TTFT physical floor changed")
    all_bytes_serial_ps = (
        MAX_FORWARDED_TOKENS
        * NUM_LAYERS
        * PHASES_PER_LAYER
        * MAX_REMOTE_DESTINATIONS
        * VECTOR_BYTES
        * byte_time_ps
    )
    conservative_bound_ps = (
        all_bytes_serial_ps
        + SOURCE_STEP_COUNT * MAX_COMPUTE_PS
        + ARRIVAL_SPAN_PS
        + SOURCE_GOAL_COUNT * 10_000_000
    )
    if conservative_bound_ps >= END_TO_END_CEILING_PS:
        raise AssertionError("frozen end-to-end ceiling no longer covers the bound")
    if EXPECTED_BEHAVIORAL_INSTANCES != 1 + 1 + 4 + 1:
        raise AssertionError("behavioral instance denominator changed")

    print(
        f"check-only mission-run={args.mission_run}; run-dir={args.run_dir}; "
        "validated frozen congestion-chain inputs and produced no artifacts"
    )


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    """Run the frozen treatment cells after the expectations commit."""

    raise NotImplementedError(
        "result-producing congestion-chain harness lands after the expectations freeze"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission-run", type=Path, required=True)
    parser.add_argument("--htsim-rnic", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_only(args)
    if args.check_only:
        return
    result = run_study(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
