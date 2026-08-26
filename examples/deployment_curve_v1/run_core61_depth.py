"""Run the single permitted retained-evidence access for CORE-61."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core61_depth_extrapolation import derive_result
from core61_depth_field_reader import ALLOWED_RECORD, read_retained_depth_basis

STUDY_DIR = Path(__file__).resolve().parent


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expectations-commit", required=True)
    args = parser.parse_args()

    access_log = args.run_dir / "access-ledger.jsonl"
    output = args.run_dir / "core61-depth-result.json"
    if access_log.exists() or output.exists():
        raise FileExistsError("CORE-61 run paths must be fresh; retained access is single-use")

    expectations = _read_json(STUDY_DIR / "core61_depth_expectations.json")
    basis = read_retained_depth_basis(ALLOWED_RECORD, access_log)
    access_lines = access_log.read_text(encoding="utf-8").splitlines()
    if len(access_lines) != 1:
        raise ValueError("CORE-61 access ledger must contain exactly one entry")
    access_entry = json.loads(access_lines[0])
    result = derive_result(
        expectations,
        basis,
        access_entry,
        expectations_commit=args.expectations_commit,
    )
    _write_json(output, result)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
