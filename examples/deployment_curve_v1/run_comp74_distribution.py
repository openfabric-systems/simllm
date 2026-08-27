#!/usr/bin/env python3
"""Run the local COMP-74 repeat-derived distribution propagation study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePath
from typing import Any

from comp74_distribution import (
    build_result,
    file_sha256,
    verify_preservation_lock,
    verify_source_digests,
    write_band_table,
    write_json,
    write_key_table,
)
from comp74_field_reader import (
    CURVE_CONFIG,
    RUN4_PUBLICATION,
    SUCCESSOR_RESULT,
    read_curve_config,
    read_run4_publication,
    read_successor_repeats,
)

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
EXPECTATIONS_PATH = STUDY_DIR / "comp74_expectations.json"


def render_cli_path(path: PurePath) -> str:
    """Render command paths with POSIX separators on every host."""

    return path.as_posix()


def _load_expectations() -> dict[str, Any]:
    value = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("COMP-74 expectations must contain an object")
    return value


def _load_access(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("COMP-74 access rows must be objects")
            rows.append(value)
    return rows


def _validate_run_dir(path: Path) -> None:
    resolved = path.resolve()
    if resolved.parent.parts[-2:] != ("wave-runs", "comp74"):
        raise SystemExit("COMP-74 run directory must stay under wave-runs/comp74")
    if not resolved.name.startswith("attempt-"):
        raise SystemExit("COMP-74 run directory must use an attempt-N name")
    if resolved.exists():
        raise SystemExit(f"COMP-74 refuses to overwrite {render_cli_path(resolved)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--digest-output", required=True, type=Path)
    parser.add_argument("--key-table", required=True, type=Path)
    parser.add_argument("--band-table", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_run_dir(args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    access_path = args.run_dir / "access.jsonl"
    frozen = _load_expectations()
    source_checks = verify_source_digests(REPOSITORY_ROOT, frozen)
    preservation = verify_preservation_lock(REPOSITORY_ROOT, frozen)
    successor = read_successor_repeats(SUCCESSOR_RESULT, access_path)
    run4 = read_run4_publication(RUN4_PUBLICATION, access_path)
    curve_config = read_curve_config(CURVE_CONFIG, access_path)
    access = _load_access(access_path)
    result = build_result(
        frozen,
        successor,
        run4,
        curve_config,
        source_checks,
        preservation,
        access,
    )

    external_result = args.run_dir / "result.json"
    external_key_table = args.run_dir / "per-key-intervals.csv"
    external_band_table = args.run_dir / "band-table.csv"
    write_json(external_result, result)
    write_key_table(external_key_table, result)
    write_band_table(external_band_table, result)
    write_json(args.output, result)
    write_key_table(args.key_table, result)
    write_band_table(args.band_table, result)
    digest = file_sha256(args.output)
    args.digest_output.write_text(
        f"{digest}  {args.output.name}\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"wrote {render_cli_path(args.output)} at sha256:{digest}; "
        f"bulk retained under {render_cli_path(args.run_dir)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
