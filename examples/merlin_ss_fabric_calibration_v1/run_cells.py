#!/usr/bin/env python3
"""Run the frozen TRAF-51 calibration cells against htsim_ss_dragonfly.

Drives the pinned backend binary directly (standing drive-simulators-directly
rule), executes every frozen cell twice for the repeat-determinism guard,
re-runs the four backend sanity arms as the build-level determinism
precondition, and writes a run manifest with SHA-256 identities for the
binary, the topology files and every produced artifact.

The frozen run matrix lives in expectations.md; this runner adds nothing to
it. Outputs go to --out-root (bulk storage, never the repository).

    python run_cells.py --binary PATH --sanity-topo PATH --out-root DIR
        [--topo-dir DIR] [--submodule-dir DIR] [--sanity-reference-dir DIR]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

EXPECTED_BINARY_SHA256 = "5075021a6af762e914d782a1c69c1633d9b084f767ac82d9b6931bd33f69f787"
EXPECTED_PIN = "89b7a5a8caa16185c257899a4200935de0cc69c4"

MERLIN_TOPO = "merlin_a100_singleswitch_v1.topo"
MERLIN_TOPO_100G = "merlin_a100_singleswitch_v1_100g.topo"

# The frozen cell matrix (expectations.md, "Cells and the run matrix").
CELLS: tuple[dict[str, object], ...] = (
    {
        "name": "cal-solo-a",
        "topo": MERLIN_TOPO,
        "args": [
            "-pattern", "incast", "-receiver", "5", "-degree", "1",
            "-duration_ps", "200000000000", "-bin_ps", "1000000",
            "-routing", "adaptive", "-wire_bytes", "9038", "-header_bytes", "90",
        ],
    },
    {
        "name": "cal-solo-b",
        "topo": MERLIN_TOPO,
        "args": [
            "-pattern", "incast", "-receiver", "17", "-degree", "1",
            "-duration_ps", "200000000000", "-bin_ps", "1000000",
            "-routing", "adaptive", "-wire_bytes", "9038", "-header_bytes", "90",
        ],
    },
    {
        "name": "cal-solo-4160",
        "topo": MERLIN_TOPO,
        "args": [
            "-pattern", "incast", "-receiver", "5", "-degree", "1",
            "-duration_ps", "200000000000", "-bin_ps", "1000000",
            "-routing", "adaptive", "-wire_bytes", "4160", "-header_bytes", "64",
        ],
    },
    {
        "name": "cal-solo-100g",
        "topo": MERLIN_TOPO_100G,
        "args": [
            "-pattern", "incast", "-receiver", "5", "-degree", "1",
            "-duration_ps", "200000000000", "-bin_ps", "1000000",
            "-routing", "adaptive", "-wire_bytes", "9038", "-header_bytes", "90",
        ],
    },
    {
        "name": "cal-echo-2",
        "topo": MERLIN_TOPO,
        "args": [
            "-pattern", "incast", "-receiver", "5", "-degree", "2",
            "-duration_ps", "100000000000", "-bin_ps", "100000000",
            "-routing", "adaptive", "-wire_bytes", "9038", "-header_bytes", "90",
        ],
    },
    {
        "name": "cal-echo-4",
        "topo": MERLIN_TOPO,
        "args": [
            "-pattern", "incast", "-receiver", "5", "-degree", "4",
            "-duration_ps", "100000000000", "-bin_ps", "100000000",
            "-routing", "adaptive", "-wire_bytes", "9038", "-header_bytes", "90",
        ],
    },
)

# The four backend sanity arms rerun as the determinism precondition (FG-2).
SANITY_ARMS: tuple[dict[str, object], ...] = (
    {
        "name": "incast_1_minimal",
        "args": [
            "-pattern", "incast", "-receiver", "0", "-degree", "1",
            "-duration_ps", "2000000000", "-bin_ps", "100000000",
            "-routing", "minimal",
        ],
    },
    {
        "name": "incast_2_adaptive",
        "args": [
            "-pattern", "incast", "-receiver", "0", "-degree", "2",
            "-duration_ps", "2000000000", "-bin_ps", "100000000",
            "-routing", "adaptive",
        ],
    },
    {
        "name": "incast_8_adaptive",
        "args": [
            "-pattern", "incast", "-receiver", "0", "-degree", "8",
            "-duration_ps", "2000000000", "-bin_ps", "100000000",
            "-routing", "adaptive",
        ],
    },
    {
        "name": "join_adaptive",
        "args": [
            "-pattern", "join", "-receiver", "0", "-degree", "4",
            "-join_interval_ps", "250000000",
            "-duration_ps", "1500000000", "-bin_ps", "100000000",
            "-routing", "adaptive",
        ],
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_once(binary: Path, topo: Path, args: list[str], csv_path: Path,
             log_path: Path) -> int:
    command = [str(binary), "-topo", str(topo), *args, "-out", str(csv_path)]
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=False)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        default=os.environ.get("SIMLLM_HTSIM_SS_DRAGONFLY", ""),
        help="htsim_ss_dragonfly binary (or SIMLLM_HTSIM_SS_DRAGONFLY)")
    parser.add_argument(
        "--out-root", required=True,
        help="bulk output directory (outside the repository)")
    parser.add_argument(
        "--topo-dir", default=str(Path(__file__).resolve().parent),
        help="directory holding the frozen topology files")
    parser.add_argument(
        "--sanity-topo", required=True,
        help="path to the backend p2a2h1g3_200g.topo for the FG-2 arms")
    parser.add_argument(
        "--submodule-dir", default="",
        help="optional third_party/htsim checkout to verify the pin")
    parser.add_argument(
        "--sanity-reference-dir", default="",
        help="optional archived sanity CSV directory for byte comparison")
    options = parser.parse_args()

    if not options.binary:
        parser.error("--binary or SIMLLM_HTSIM_SS_DRAGONFLY is required")
    binary = Path(options.binary).resolve()
    out_root = Path(options.out_root).resolve()
    topo_dir = Path(options.topo_dir).resolve()
    sanity_topo = Path(options.sanity_topo).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "study": "merlin_ss_fabric_calibration_v1",
        "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "binary_sha256": sha256_file(binary),
        "binary_sha256_expected": EXPECTED_BINARY_SHA256,
        "topologies": {},
        "cells": {},
        "sanity_arms": {},
    }
    guard_failures: list[str] = []

    if manifest["binary_sha256"] != EXPECTED_BINARY_SHA256:
        guard_failures.append("FG-1: binary sha256 mismatch")

    if options.submodule_dir:
        head = subprocess.run(
            ["git", "-C", options.submodule_dir, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False)
        pin = head.stdout.strip()
        manifest["submodule_head"] = pin
        manifest["submodule_head_expected"] = EXPECTED_PIN
        if head.returncode != 0 or pin != EXPECTED_PIN:
            guard_failures.append("FG-1: submodule pin mismatch")

    topo_hashes: dict[str, str] = {}
    for topo_name in (MERLIN_TOPO, MERLIN_TOPO_100G):
        topo_hashes[topo_name] = sha256_file(topo_dir / topo_name)
    topo_hashes[sanity_topo.name] = sha256_file(sanity_topo)
    manifest["topologies"] = topo_hashes

    for arm in SANITY_ARMS:
        name = str(arm["name"])
        record: dict[str, object] = {"repeat_identical": False}
        csvs: list[Path] = []
        for rep in ("a", "b"):
            csv_path = out_root / f"sanity_{name}.{rep}.csv"
            log_path = out_root / f"sanity_{name}.{rep}.log"
            code = run_once(binary, sanity_topo, list(arm["args"]), csv_path,
                            log_path)
            record[f"exit_{rep}"] = code
            if code != 0:
                guard_failures.append(f"FG-5: sanity {name}.{rep} exit {code}")
            csvs.append(csv_path)
        if all(p.exists() for p in csvs):
            record["repeat_identical"] = csvs[0].read_bytes() == csvs[1].read_bytes()
            record["csv_sha256"] = sha256_file(csvs[0])
        if not record["repeat_identical"]:
            guard_failures.append(f"FG-2: sanity {name} repeats differ")
        if options.sanity_reference_dir:
            reference = Path(options.sanity_reference_dir) / f"{name}.csv"
            if reference.exists():
                record["matches_reference"] = (
                    csvs[0].read_bytes() == reference.read_bytes())
        manifest["sanity_arms"][name] = record

    for cell in CELLS:
        name = str(cell["name"])
        topo_path = topo_dir / str(cell["topo"])
        record = {
            "topo": str(cell["topo"]),
            "topo_sha256": topo_hashes[str(cell["topo"])],
            "args": list(cell["args"]),
            "repeat_identical": False,
        }
        csvs = []
        for rep in ("a", "b"):
            csv_path = out_root / f"{name}.{rep}.csv"
            log_path = out_root / f"{name}.{rep}.log"
            code = run_once(binary, topo_path, list(cell["args"]), csv_path,
                            log_path)
            record[f"exit_{rep}"] = code
            if code != 0:
                guard_failures.append(f"FG-5: cell {name}.{rep} exit {code}")
            csvs.append(csv_path)
        if all(p.exists() for p in csvs):
            record["repeat_identical"] = csvs[0].read_bytes() == csvs[1].read_bytes()
            record["csv_sha256"] = sha256_file(csvs[0])
        if not record["repeat_identical"]:
            guard_failures.append(f"FG-3: cell {name} repeats differ")
        manifest["cells"][name] = record

    manifest["finished_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    manifest["guard_failures"] = guard_failures
    manifest_path = out_root / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    if guard_failures:
        for failure in guard_failures:
            print(f"FATAL {failure}", file=sys.stderr)
        return 1
    print("all runs complete, no runner-level guard failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
