#!/usr/bin/env python3
"""Create or verify the deterministic TRAF-77 submitted-source manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

SUBMITTED_FILES = (
    "analyze_capture.py",
    "capture_rank_identity.sh",
    "capture_w2_four_port.sbatch",
    "capture_w2_one_port.sbatch",
    "capture_w8_four_port.sbatch",
    "capture_w8_one_port.sbatch",
    "hash_manifest.py",
    "merlin_collective_lane.cu",
    "run_capture.sh",
    "snapshot_counters.py",
    "study_config.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_text(root: Path) -> str:
    missing = [name for name in SUBMITTED_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"submitted-source files are missing: {', '.join(missing)}")
    return "".join(f"{sha256_file(root / name)}  {name}\n" for name in SUBMITTED_FILES)


def check_manifest(root: Path, manifest: Path) -> list[str]:
    expected = manifest.read_text(encoding="utf-8")
    actual = manifest_text(root)
    if actual == expected:
        return []
    expected_rows = {line[66:]: line[:64] for line in expected.splitlines() if len(line) >= 66}
    actual_rows = {line[66:]: line[:64] for line in actual.splitlines() if len(line) >= 66}
    names = sorted(set(expected_rows) | set(actual_rows))
    return [name for name in names if expected_rows.get(name) != actual_rows.get(name)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--check", type=Path)
    args = parser.parse_args(argv)

    if args.output is not None:
        args.output.write_text(manifest_text(args.root), encoding="utf-8")
        return 0

    mismatches = check_manifest(args.root, args.check)
    if mismatches:
        print("submitted-source manifest mismatch: " + ", ".join(mismatches))
        return 1
    print("submitted-source manifest verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
