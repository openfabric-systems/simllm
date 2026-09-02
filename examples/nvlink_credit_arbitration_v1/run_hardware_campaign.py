#!/usr/bin/env python3
"""Run or resume one frozen TRAF-73 NV4 hardware hypothesis family."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import re
import signal
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAF70_ROOT = HERE.parent / "a100_nvlink_packet_v2"
PRODUCER_EXTENSION_PATH = HERE / "producer_traf73.patch"
EXPECTATIONS_PATH = HERE / "aligned_expectations.json"
# Re-pin amendment, 2026-09-02. The default branch was rewritten and force
# pushed on 2026-09-01 to purge co-author trailers, which gave every commit
# from the rewrite point onward a new identifier. The freeze itself did not
# move. The original pin f3f2624e7a96efe3ad67eac5940fee8746e40b98 and the pin
# below carry the same tree 54ef266d5f81437c583c2522701c0bd744dd7b44, the same
# subject "Freeze aligned NVLink identification" and the same author date
# 2026-09-01T15:53:41+02:00. The frozen digest below is unchanged, and the
# published TRAF-73 record keeps the pre-rewrite identifier it ran under.
EXPECTATIONS_COMMIT = "7cab9cd84dfcfce50c7b45553bcd4a54ec4a8ea0"
EXPECTATIONS_SHA256 = "a17b9e298d11a4a6ba92b382121c15a5a48f8100b6f343893260419b1d3382f6"
DERIVED_PRODUCER_SHA256 = "3e4b24382314f5f0dd84f4b54d126c5777e6c92a71c3644f8835b2f5cd3a4694"
CELL_SCHEMA = "simllm-nvlink-credit-identification-cell-v1"
MANIFEST_SCHEMA = "simllm-nvlink-credit-identification-manifest-v1"
OBSERVATION_SCHEMA = "simllm-a100-nvlink-packet-observation-v2"
BULK_ROOT_ENV = "SIMLLM_TRAF73_BULK_ROOT"
FAMILY_TIMEOUT_SECONDS = {"h1": 25 * 60, "h2": 25 * 60, "h3": 12 * 60}

CUSTOM_FIELDS = (
    "traf73_warmup_repetitions",
    "traf73_timed_repetitions",
    "traf73_excluded_repetitions_each_edge",
    "traf73_flow_offered_rate_percents",
    "traf73_window_warmup_ms",
    "traf73_window_measurement_ms",
    "traf73_window_drain_ms",
    "traf73_ring_bytes",
)

IMPLEMENTATION_PATHS = (
    Path(__file__),
    HERE / "run_merlin_hypothesis.sbatch",
    PRODUCER_EXTENSION_PATH,
    EXPECTATIONS_PATH,
    TRAF70_ROOT / "case_matrix.py",
    TRAF70_ROOT / "nvlink_packet_lane.cu",
    TRAF70_ROOT / "run_study.py",
    TRAF70_ROOT / "sha256.h",
)

_ACTIVE_CHILD: subprocess.Popen[str] | None = None
_STOP_SIGNAL: int | None = None


class StudyStopped(RuntimeError):
    """Raised after a scheduler or operator stop signal."""


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


traf70_cases = _load_module("_traf73_case_matrix", TRAF70_ROOT / "case_matrix.py")
_previous_case_matrix = sys.modules.get("case_matrix")
sys.modules["case_matrix"] = traf70_cases
try:
    traf70_run = _load_module("_traf73_corrected_producer_runner", TRAF70_ROOT / "run_study.py")
finally:
    if _previous_case_matrix is None:
        del sys.modules["case_matrix"]
    else:
        sys.modules["case_matrix"] = _previous_case_matrix

TSV_FIELDS = (*traf70_cases.TSV_FIELDS, *CUSTOM_FIELDS)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("h1", "h2", "h3"), required=True)
    parser.add_argument("--mode", choices=("mock", "hardware"), default="mock")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--expected-head", default="")
    parser.add_argument("--freeze-sha256", default=EXPECTATIONS_SHA256)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--pending", action="store_true")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_expectations(
    expected_digest: str = EXPECTATIONS_SHA256,
) -> dict[str, Any]:
    if expected_digest != EXPECTATIONS_SHA256:
        raise ValueError("the requested TRAF-73 freeze digest is not committed")
    if sha256(EXPECTATIONS_PATH) != EXPECTATIONS_SHA256:
        raise RuntimeError("the aligned TRAF-73 expectations digest changed")
    if (ROOT / ".git").exists():
        completed = subprocess.run(
            ("git", "merge-base", "--is-ancestor", EXPECTATIONS_COMMIT, "HEAD"),
            cwd=ROOT,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError("the aligned expectations commit is not an ancestor")
    with open(EXPECTATIONS_PATH, encoding="utf-8", newline="") as handle:
        frozen = json.load(handle)
    if frozen["study"]["status"] != "EXPECTATIONS_ONLY":
        raise RuntimeError("the aligned TRAF-73 authority is not expectations-only")
    return frozen


def verify_frozen_authority(frozen: dict[str, Any]) -> None:
    authority = frozen["aligned_authority"]
    module_path = ROOT / authority["module_path"]
    profile = authority["candidate_profile"]
    if sha256(module_path) != authority["module_base_blob_sha256"]:
        raise RuntimeError("the aligned NVLink module moved after the freeze")
    if sha256(ROOT / profile["path"]) != profile["sha256"]:
        raise RuntimeError("the candidate profile moved after the freeze")
    producer = frozen["producer_contract"]["source"]
    if sha256(ROOT / producer["path"]) != producer["sha256"]:
        raise RuntimeError("the corrected TRAF-70 producer lineage moved after the freeze")
    if frozen["promotion_rule"]["task_id"] != "TRAF-85":
        raise RuntimeError("the assigned promotion residual changed")


def _base_point(**overrides: object) -> dict[str, object]:
    point: dict[str, object] = {
        "case_name": "TRAF73_UNASSIGNED",
        "point_id": "TRAF73_UNASSIGNED",
        "producer": "persistent_sm_peer_write",
        "payload_bytes": 256,
        "message_count": 1,
        "source": 0,
        "destination": 1,
        "sources": "0",
        "destinations": "1",
        "source_alignment": 0,
        "destination_alignment": 0,
        "access_width": 16,
        "active_lanes": 32,
        "lane_mask": "contiguous",
        "stride": 1,
        "stream_count": 1,
        "outstanding": 256,
        "burst_messages": 256,
        "gap_ns": 0,
        "offered_rate_percent": 100,
        "pattern": "unidirectional",
        "traf73_warmup_repetitions": 0,
        "traf73_timed_repetitions": 0,
        "traf73_excluded_repetitions_each_edge": 0,
        "traf73_flow_offered_rate_percents": "",
        "traf73_window_warmup_ms": 0,
        "traf73_window_measurement_ms": 0,
        "traf73_window_drain_ms": 0,
        "traf73_ring_bytes": 0,
    }
    point.update(overrides)
    if set(point) != set(TSV_FIELDS):
        raise RuntimeError("TRAF-73 point fields diverged from the producer plan")
    return point


def _seeded_sizes(values: Sequence[int]) -> list[int]:
    ordered = list(values)
    random.Random(7301).shuffle(ordered)
    return ordered


def h1_points(frozen: dict[str, Any]) -> tuple[dict[str, object], ...]:
    h1 = frozen["h1_credit_window_and_return"]
    rows = []
    for source, destination in h1["directed_pairs"]:
        for size_bytes in _seeded_sizes(h1["payload_sizes_bytes"]):
            rows.append(
                _base_point(
                    case_name="TRAF73_H1_CREDIT_WINDOW_RETURN",
                    point_id=f"H1:{source}->{destination}:bytes={size_bytes}",
                    payload_bytes=size_bytes,
                    message_count=h1["timed_repetitions_per_pair_and_size"],
                    source=source,
                    destination=destination,
                    sources=str(source),
                    destinations=str(destination),
                    pattern="traf73_latency_batch",
                    traf73_warmup_repetitions=h1["warmups_per_pair_and_size"],
                    traf73_timed_repetitions=h1[
                        "timed_repetitions_per_pair_and_size"
                    ],
                )
            )
    if len(rows) != h1["configuration_count"]:
        raise RuntimeError("H1 point expansion differs from the freeze")
    return tuple(rows)


def h2_points(frozen: dict[str, Any]) -> tuple[dict[str, object], ...]:
    h2 = frozen["h2_pool_scope"]
    rows = []
    for sources in h2["source_sets"]:
        for size_bytes in _seeded_sizes(h2["payload_sizes_bytes"]):
            rows.append(
                _base_point(
                    case_name="TRAF73_H2_POOL_SCOPE",
                    point_id=(
                        f"H2:sources={','.join(str(value) for value in sources)}:"
                        f"receiver={h2['receiver']}:bytes={size_bytes}"
                    ),
                    payload_bytes=size_bytes,
                    message_count=h2[
                        "timed_repetitions_per_source_count_and_size"
                    ],
                    source=sources[0],
                    destination=h2["receiver"],
                    sources=",".join(str(value) for value in sources),
                    destinations=str(h2["receiver"]),
                    pattern="traf73_latency_batch",
                    traf73_warmup_repetitions=h2[
                        "warmups_per_source_count_and_size"
                    ],
                    traf73_timed_repetitions=h2[
                        "timed_repetitions_per_source_count_and_size"
                    ],
                    traf73_excluded_repetitions_each_edge=20,
                )
            )
    if len(rows) != h2["configuration_count"]:
        raise RuntimeError("H2 point expansion differs from the freeze")
    return tuple(rows)


def h3_points(frozen: dict[str, Any]) -> tuple[dict[str, object], ...]:
    h3 = frozen["h3_arbitration"]
    rows = []
    all_sources = {0, 1, 2}
    for greedy in h3["greedy_role_sources"]:
        sources = [greedy, *sorted(all_sources - {greedy})]
        rows.append(
            _base_point(
                case_name="TRAF73_H3_ARBITRATION",
                point_id=f"H3:greedy={greedy}:receiver={h3['receiver']}",
                payload_bytes=h3["chunk_bytes"],
                message_count=1,
                source=greedy,
                destination=h3["receiver"],
                sources=",".join(str(value) for value in sources),
                destinations=str(h3["receiver"]),
                pattern="traf73_steady_arbitration",
                traf73_flow_offered_rate_percents="100,60,60",
                traf73_window_warmup_ms=h3["warmup_ms"],
                traf73_window_measurement_ms=h3["measurement_ms"],
                traf73_window_drain_ms=h3["drain_ms"],
                traf73_ring_bytes=h3["chunk_bytes"],
            )
        )
    if len(rows) != 3:
        raise RuntimeError("H3 point expansion differs from the freeze")
    return tuple(rows)


def campaign_points(
    frozen: dict[str, Any], family: str
) -> tuple[dict[str, object], ...]:
    return {"h1": h1_points, "h2": h2_points, "h3": h3_points}[family](frozen)


def resolve_output_root(value: Path | None) -> Path:
    selected = value
    if selected is None:
        configured = os.environ.get(BULK_ROOT_ENV)
        if not configured:
            raise ValueError(f"set {BULK_ROOT_ENV} or pass --output-root")
        selected = Path(configured)
    return selected.resolve()


def check_arguments(args: argparse.Namespace) -> None:
    if args.expected_head and re.fullmatch(r"[0-9a-f]{40}", args.expected_head) is None:
        raise ValueError("expected head must be a full lowercase Git SHA")
    if args.mode == "hardware" and not args.expected_head:
        raise ValueError("hardware mode requires --expected-head")
    if not args.dry_run and not args.check_only and not args.pending:
        if args.binary is None:
            raise ValueError("a producer binary is required")
        if not os.access(args.binary, os.X_OK):
            raise FileNotFoundError("the producer binary is not executable")


def cell_root(output_root: Path, family: str) -> Path:
    return output_root / EXPECTATIONS_SHA256 / "cells" / family


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in IMPLEMENTATION_PATHS:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_attempt(attempt: Path) -> bool:
    manifest_path = attempt / "manifest.json"
    complete_path = attempt / "COMPLETE.json"
    if not manifest_path.is_file() or not complete_path.is_file():
        return False
    try:
        with open(manifest_path, encoding="utf-8", newline="") as handle:
            manifest = json.load(handle)
        with open(complete_path, encoding="utf-8", newline="") as handle:
            complete = json.load(handle)
        if manifest["schema"] != MANIFEST_SCHEMA or complete["schema"] != MANIFEST_SCHEMA:
            return False
        if complete["manifest_sha256"] != sha256(manifest_path):
            return False
        names = set()
        for payload in manifest["payloads"]:
            name = payload["path"]
            if Path(name).name != name or name in names:
                return False
            names.add(name)
            path = attempt / name
            if (
                not path.is_file()
                or path.stat().st_size != payload["bytes"]
                or sha256(path) != payload["sha256"]
            ):
                return False
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def attempt_matches(attempt: Path, args: argparse.Namespace) -> bool:
    try:
        with open(attempt / "plan.json", encoding="utf-8", newline="") as handle:
            plan = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if plan.get("mode") != args.mode or plan.get("family") != args.family:
        return False
    if plan.get("implementation_sha256") != implementation_sha256():
        return False
    if args.binary is None or plan.get("producer_binary_sha256") != sha256(args.binary):
        return False
    return not args.expected_head or plan.get("expected_head") == args.expected_head


def complete_attempt(
    output_root: Path, args: argparse.Namespace
) -> Path | None:
    root = cell_root(output_root, args.family)
    for attempt in sorted(root.glob("attempt-*")):
        if verify_attempt(attempt) and attempt_matches(attempt, args):
            return attempt
    return None


def new_attempt_path(output_root: Path, family: str) -> Path:
    root = cell_root(output_root, family)
    root.mkdir(parents=True, exist_ok=True)
    attempts = sorted(root.glob("attempt-*"))
    path = root / f"attempt-{len(attempts) + 1:04d}"
    path.mkdir()
    return path


def write_points(path: Path, points: tuple[dict[str, object], ...]) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=TSV_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(points)


def run_child(
    command: Sequence[str], *, timeout: int
) -> subprocess.CompletedProcess[str]:
    global _ACTIVE_CHILD
    _ACTIVE_CHILD = subprocess.Popen(
        list(command),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = _ACTIVE_CHILD.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        _ACTIVE_CHILD.terminate()
        try:
            stdout, stderr = _ACTIVE_CHILD.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            _ACTIVE_CHILD.kill()
            stdout, stderr = _ACTIVE_CHILD.communicate()
        raise RuntimeError(f"producer timed out after {timeout} seconds") from error
    finally:
        child = _ACTIVE_CHILD
        _ACTIVE_CHILD = None
    if _STOP_SIGNAL is not None:
        raise StudyStopped(f"received signal {_STOP_SIGNAL}")
    return subprocess.CompletedProcess(command, child.returncode, stdout, stderr)


def summarize_results(
    path: Path,
    *,
    expected_rows: int,
    family: str,
    mode: str,
) -> dict[str, object]:
    rows = []
    with open(path, encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid result JSON at line {line_number}") from error
    expected_mode = "steady_arbitration" if family == "h3" else "latency_batch"
    if len(rows) != expected_rows:
        raise RuntimeError(f"producer returned {len(rows)} rows, expected {expected_rows}")
    for row in rows:
        if row.get("schema") != OBSERVATION_SCHEMA or row.get("mode") != mode:
            raise RuntimeError("producer returned the wrong observation authority")
        if row.get("traf73", {}).get("mode") != expected_mode:
            raise RuntimeError("producer omitted the frozen TRAF-73 observation mode")
        if row.get("measurement_claim") not in (False, "unscored"):
            raise RuntimeError("producer made a scored measurement claim")
    return {
        "schema": CELL_SCHEMA,
        "status": "mock_complete" if mode == "mock" else "hardware_unscored",
        "family": family,
        "row_count": len(rows),
        "checksum_failure_count": sum(row.get("checksum_ok") is not True for row in rows),
        "measurement_claim": False if mode == "mock" else "pending_scoring",
    }


def write_manifest(attempt: Path, family: str) -> Path:
    payloads = []
    for path in sorted(attempt.iterdir()):
        if not path.is_file() or path.name in {"manifest.json", "COMPLETE.json"}:
            continue
        payloads.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    path = attempt / "manifest.json"
    write_json_exclusive(
        path,
        {
            "schema": MANIFEST_SCHEMA,
            "family": family,
            "freeze_sha256": EXPECTATIONS_SHA256,
            "payloads": payloads,
        },
    )
    return path


def produce_attempt(
    attempt: Path,
    points: tuple[dict[str, object], ...],
    args: argparse.Namespace,
) -> str:
    if args.binary is None:
        raise ValueError("producer binary is required")
    write_json_exclusive(
        attempt / "plan.json",
        {
            "schema": CELL_SCHEMA,
            "status": "planned",
            "family": args.family,
            "mode": args.mode,
            "freeze_sha256": args.freeze_sha256,
            "freeze_commit": EXPECTATIONS_COMMIT,
            "implementation_sha256": implementation_sha256(),
            "producer_source_sha256": sha256(
                TRAF70_ROOT / "nvlink_packet_lane.cu"
            ),
            "producer_extension_sha256": sha256(PRODUCER_EXTENSION_PATH),
            "producer_derived_source_sha256": DERIVED_PRODUCER_SHA256,
            "producer_binary_sha256": sha256(args.binary),
            "expected_head": args.expected_head or None,
            "point_count": len(points),
        },
    )
    write_points(attempt / "points.tsv", points)
    write_json_exclusive(
        attempt / "environment.json",
        {
            "schema": CELL_SCHEMA,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "family": args.family,
            "mode": args.mode,
            "hostname": os.environ.get("HOSTNAME", "unknown"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_job_node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "source_head": args.expected_head or local_head(),
        },
    )
    if args.mode == "hardware":
        traf70_run._collect_hardware_guard(attempt, "before")
    else:
        write_text_exclusive(
            attempt / "guards_before.txt",
            "mode=mock\nhardware_guards=not_applicable\n",
        )
    result_path = attempt / "results.jsonl"
    completed = run_child(
        (
            str(args.binary),
            "--points",
            str(attempt / "points.tsv"),
            "--output",
            str(result_path),
            "--mode",
            args.mode,
        ),
        timeout=FAMILY_TIMEOUT_SECONDS[args.family],
    )
    write_text_exclusive(attempt / "stdout.txt", completed.stdout)
    write_text_exclusive(attempt / "stderr.txt", completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"producer exited {completed.returncode}")
    summary = summarize_results(
        result_path,
        expected_rows=len(points),
        family=args.family,
        mode=args.mode,
    )
    write_json_exclusive(attempt / "summary.json", summary)
    if args.mode == "hardware":
        traf70_run._collect_hardware_guard(attempt, "after")
    else:
        write_text_exclusive(
            attempt / "guards_after.txt",
            "mode=mock\nhardware_guards=not_applicable\n",
        )
    manifest = write_manifest(attempt, args.family)
    write_json_exclusive(
        attempt / "COMPLETE.json",
        {
            "schema": MANIFEST_SCHEMA,
            "status": "complete",
            "family": args.family,
            "manifest_sha256": sha256(manifest),
        },
    )
    if not verify_attempt(attempt):
        raise RuntimeError("the completed TRAF-73 attempt failed its digest audit")
    return f"complete ({len(points)} points)"


def install_signal_handlers() -> None:
    for name in ("SIGHUP", "SIGINT", "SIGTERM"):
        value = getattr(signal, name, None)
        if value is not None:
            signal.signal(value, handle_stop)


def handle_stop(signal_number: int, _frame: object) -> None:
    global _STOP_SIGNAL
    _STOP_SIGNAL = signal_number
    if _ACTIVE_CHILD is not None:
        _ACTIVE_CHILD.terminate()


def local_head() -> str | None:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def write_json_exclusive(path: Path, payload: object) -> None:
    write_text_exclusive(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_text_exclusive(path: Path, value: str) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    frozen = load_expectations(args.freeze_sha256)
    verify_frozen_authority(frozen)
    points = campaign_points(frozen, args.family)
    output_root = resolve_output_root(args.output_root)
    check_arguments(args)
    if args.check_only:
        print(
            f"TRAF-73 {args.family} check passed: {len(points)} rows, "
            f"freeze {EXPECTATIONS_SHA256}"
        )
        return 0
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema": CELL_SCHEMA,
                    "family": args.family,
                    "mode": args.mode,
                    "freeze_sha256": args.freeze_sha256,
                    "expected_head": args.expected_head or None,
                    "output_root": output_root.as_posix(),
                    "point_count": len(points),
                    "implementation_sha256": implementation_sha256(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.pending:
        print("" if complete_attempt(output_root, args) is not None else args.family)
        return 0
    if complete_attempt(output_root, args) is not None:
        print(f"TRAF-73 {args.family}: already complete and digest verified")
        return 0
    install_signal_handlers()
    attempt = new_attempt_path(output_root, args.family)
    try:
        result = produce_attempt(attempt, points, args)
    except BaseException as error:
        write_json_exclusive(
            attempt / "STOPPED.json",
            {
                "schema": CELL_SCHEMA,
                "status": "stopped",
                "family": args.family,
                "signal": _STOP_SIGNAL,
                "error_type": type(error).__name__,
                "error": str(error),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
    print(f"TRAF-73 {args.family}: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
