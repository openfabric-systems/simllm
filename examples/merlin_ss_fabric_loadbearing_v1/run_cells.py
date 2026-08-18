#!/usr/bin/env python3
"""Run the frozen TRAF-51 load-bearing recalibration cells.

Drives the pinned htsim_ss_dragonfly binary directly (standing
drive-simulators-directly rule), executes every frozen cell twice for the
repeat-determinism guard, strictly sequentially, and writes a run manifest
with SHA-256 identities for the binary, the topology instances and every
produced artifact. The frozen run matrix lives in expectations.md; this
runner adds nothing to it. The x4r-v1 cell's registered EXPECTED outcome is
the closed-loop drop fault (scored row BE-3), so its nonzero exit is
recorded, never treated as a runner-level failure.

    python run_cells.py --binary PATH --out-root DIR
        [--only CELL[,CELL...]] [--late DESCRIPTOR.json]
        [--submodule-dir DIR]
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

STUDY_DIR = Path(__file__).resolve().parent
V1_TOPO = (
    STUDY_DIR.parent / "merlin_ss_fabric_calibration_v1" / "merlin_a100_singleswitch_v1.topo"
)
TOPOLOGIES: dict[str, Path] = {
    "v1": V1_TOPO,
    "x4buf32": STUDY_DIR / "merlin_a100_singleswitch_v1_x4buf32.topo",
    "x4buf64": STUDY_DIR / "merlin_a100_singleswitch_v1_x4buf64.topo",
}

EXPECTED_BINARY_SHA256 = "662416918457542c4fcab18aebd99f43f00cd80b5518ed78947a65e65620ce92"
EXPECTED_PIN = "1dcbfec36a33753bf978cf6323bade1a6645fe4f"

COMMON = ["-pattern", "explicit", "-routing", "adaptive",
          "-wire_bytes", "9038", "-header_bytes", "90", "-bin_ps", "100000000"]

CTRL_FLOWS = ["-flow", "src=8,dst=5,think_ps=1858227000",
              "-flow", "src=16,dst=6,think_ps=1256438000"]

X4_RATE_FLOWS = [
    "-flow", "src=4,dst=19,think_ps=4642143756",
    "-flow", "src=5,dst=19,think_ps=2255636718,start_ps=400000000",
    "-flow", "src=6,dst=19,think_ps=2263071395,start_ps=800000000",
    "-flow", "src=7,dst=19,think_ps=2495255483,start_ps=1200000000",
]
X4_P50_FLOWS = [
    "-flow", "src=4,dst=19,think_ps=4798387720",
    "-flow", "src=5,dst=19,think_ps=1783214720,start_ps=400000000",
    "-flow", "src=6,dst=19,think_ps=1835331720,start_ps=800000000",
    "-flow", "src=7,dst=19,think_ps=1794560720,start_ps=1200000000",
]
X4_PACED_FLOWS = [
    "-flow", "src=4,dst=19,offered_bps=13468749005",
    "-flow", "src=5,dst=19,offered_bps=25850334413",
    "-flow", "src=6,dst=19,offered_bps=25776514662",
    "-flow", "src=7,dst=19,offered_bps=23665940890",
]
SAT_FLOWS = ["-flow", "src=8,dst=5", "-flow", "src=16,dst=5,start_ps=278092"]

# The frozen cell matrix (expectations.md, "Cells and the run matrix").
CELLS: tuple[dict[str, object], ...] = (
    {
        "name": "ctrl-v1", "topo": "v1", "chunks": True, "expected_exit": 0,
        "args": [*COMMON, "-duration_ps", "200000000000", "-source", "closed",
                 "-chunk_payload_bytes", "8388608", *CTRL_FLOWS],
    },
    {
        "name": "ctrl-b32", "topo": "x4buf32", "chunks": True, "expected_exit": 0,
        "args": [*COMMON, "-duration_ps", "200000000000", "-source", "closed",
                 "-chunk_payload_bytes", "8388608", *CTRL_FLOWS],
    },
    {
        "name": "ctrl-b64", "topo": "x4buf64", "chunks": True, "expected_exit": 0,
        "args": [*COMMON, "-duration_ps", "200000000000", "-source", "closed",
                 "-chunk_payload_bytes", "8388608", *CTRL_FLOWS],
    },
    {
        "name": "mj2x-v1", "topo": "v1", "chunks": True, "expected_exit": 0,
        "args": [*COMMON, "-duration_ps", "300000000000", "-source", "closed",
                 "-chunk_payload_bytes", "8388608",
                 "-flow", "src=8,dst=5,think_ps=1858227000",
                 "-flow", "src=16,dst=6,think_ps=1256438000,start_ps=100000000000"],
    },
    {
        "name": "x4p-v1", "topo": "v1", "chunks": False, "expected_exit": 0,
        "args": [*COMMON, "-duration_ps", "1000000000000", "-source", "paced",
                 *X4_PACED_FLOWS],
    },
    {
        "name": "x4r-b32", "topo": "x4buf32", "chunks": True, "expected_exit": 0,
        "args": [*COMMON, "-duration_ps", "6000000000000", "-source", "closed",
                 "-chunk_payload_bytes", "8388608", *X4_RATE_FLOWS],
    },
    {
        "name": "x4r-b64", "topo": "x4buf64", "chunks": True, "expected_exit": 0,
        "args": [*COMMON, "-duration_ps", "6000000000000", "-source", "closed",
                 "-chunk_payload_bytes", "8388608", *X4_RATE_FLOWS],
    },
    {
        "name": "x4r-v1", "topo": "v1", "chunks": True, "expected_exit": 2,
        "args": [*COMMON, "-duration_ps", "6000000000000", "-source", "closed",
                 "-chunk_payload_bytes", "8388608", *X4_RATE_FLOWS],
    },
    {
        "name": "x4p50-b32", "topo": "x4buf32", "chunks": True, "expected_exit": 0,
        "args": [*COMMON, "-duration_ps", "6000000000000", "-source", "closed",
                 "-chunk_payload_bytes", "8388608", *X4_P50_FLOWS],
    },
    {
        "name": "sat-v1", "topo": "v1", "chunks": False, "expected_exit": 0,
        "args": [*COMMON, "-duration_ps", "5000000000", "-source", "paced",
                 "-offered_bps", "130000000000", *SAT_FLOWS],
    },
    {
        "name": "sat-b32", "topo": "x4buf32", "chunks": False, "expected_exit": 0,
        "args": [*COMMON, "-duration_ps", "5000000000", "-source", "paced",
                 "-offered_bps", "130000000000", *SAT_FLOWS],
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def late_cell_to_cell(descriptor: dict[str, object]) -> dict[str, object]:
    """Turn a --derive-late-cell descriptor into a runnable cell entry."""
    flows: list[str] = []
    for flow in descriptor["flows"]:  # type: ignore[index]
        spec = f"src={flow['src']},dst={flow['dst']},think_ps={flow['think_ps']}"
        if flow.get("start_ps"):
            spec += f",start_ps={flow['start_ps']}"
        flows.extend(["-flow", spec])
    return {
        "name": str(descriptor["cell"]),
        "topo": str(descriptor["instance"]),
        "chunks": True,
        "expected_exit": 0,
        "args": [*COMMON, "-duration_ps", str(descriptor["duration_ps"]),
                 "-source", "closed",
                 "-chunk_payload_bytes", str(descriptor["chunk_payload_bytes"]),
                 *flows],
    }


def run_cell(binary: Path, cell: dict[str, object], out_root: Path,
             manifest: dict[str, object]) -> None:
    name = str(cell["name"])
    topo_path = TOPOLOGIES[str(cell["topo"])]
    record: dict[str, object] = {
        "topo": str(cell["topo"]),
        "topo_sha256": sha256_file(topo_path),
        "expected_exit": cell["expected_exit"],
        "args": list(cell["args"]),  # type: ignore[arg-type]
    }
    for rep in ("r1", "r2"):
        csv_path = out_root / f"{name}_{rep}.csv"
        args = [str(binary), "-topo", str(topo_path), *[str(a) for a in cell["args"]]]
        if cell["chunks"]:
            args += ["-chunk_out", str(out_root / f"{name}_{rep}.chunks.csv")]
        args += ["-out", str(csv_path)]
        (out_root / f"{name}_{rep}.cmd").write_text(" ".join(args) + "\n", encoding="utf-8")
        started = _dt.datetime.now(_dt.timezone.utc).isoformat()
        completed = subprocess.run(args, capture_output=True, text=True, check=False)
        finished = _dt.datetime.now(_dt.timezone.utc).isoformat()
        (out_root / f"{name}_{rep}.stdout").write_text(completed.stdout, encoding="utf-8")
        (out_root / f"{name}_{rep}.stderr").write_text(completed.stderr, encoding="utf-8")
        record[f"exit_{rep}"] = completed.returncode
        record[f"started_{rep}"] = started
        record[f"finished_{rep}"] = finished
    manifest["cells"][name] = record  # type: ignore[index]
    print(f"cell {name}: exits {record['exit_r1']}/{record['exit_r2']} "
          f"(expected {cell['expected_exit']})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", default=os.environ.get("SIMLLM_HTSIM_SS_DRAGONFLY", ""),
        help="htsim_ss_dragonfly binary (or SIMLLM_HTSIM_SS_DRAGONFLY)")
    parser.add_argument("--out-root", required=True,
                        help="bulk output directory (outside the repository)")
    parser.add_argument("--only", default="",
                        help="comma-separated subset of frozen cell names")
    parser.add_argument("--late", default="",
                        help="late-cell descriptor JSON emitted by the analyzer")
    parser.add_argument("--submodule-dir", default="",
                        help="optional third_party/htsim checkout to verify the pin")
    options = parser.parse_args()

    if not options.binary:
        parser.error("--binary or SIMLLM_HTSIM_SS_DRAGONFLY is required")
    binary = Path(options.binary).resolve()
    out_root = Path(options.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "study": "merlin_ss_fabric_loadbearing_v1",
        "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "binary_sha256": sha256_file(binary),
        "binary_sha256_expected": EXPECTED_BINARY_SHA256,
        "topologies": {key: sha256_file(path) for key, path in TOPOLOGIES.items()},
        "cells": {},
    }
    failures: list[str] = []
    if manifest["binary_sha256"] != EXPECTED_BINARY_SHA256:
        failures.append("FG-1: binary sha256 mismatch")
    if options.submodule_dir:
        head = subprocess.run(["git", "-C", options.submodule_dir, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=False)
        pin = head.stdout.strip()
        manifest["submodule_head"] = pin
        manifest["submodule_head_expected"] = EXPECTED_PIN
        if head.returncode != 0 or pin != EXPECTED_PIN:
            failures.append("FG-1: submodule pin mismatch")

    if options.late:
        descriptor = json.loads(Path(options.late).read_text(encoding="utf-8"))
        cells: tuple[dict[str, object], ...] = (late_cell_to_cell(descriptor),)
    else:
        cells = CELLS
        if options.only:
            wanted = set(options.only.split(","))
            cells = tuple(c for c in CELLS if c["name"] in wanted)

    for cell in cells:
        run_cell(binary, cell, out_root, manifest)

    manifest["finished_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    manifest["runner_failures"] = failures
    manifest_path = out_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(f"wrote {manifest_path}")
    if failures:
        for failure in failures:
            print(f"FATAL {failure}", file=sys.stderr)
        return 1
    print("all runs complete, no runner-level failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
