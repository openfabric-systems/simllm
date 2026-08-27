"""Run or resume digest-pinned TRAF-70 cells in mock or hardware mode."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from case_matrix import (
    TSV_FIELDS,
    expand_producers,
    point_to_tsv_row,
    points_for_case,
    protocol_validation_points,
)

STUDY_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_ROOT.parents[1]
EXPECTATIONS_PATH = STUDY_ROOT / "expectations.json"
PROTECTED_CANDIDATE_PROFILE_PATH = STUDY_ROOT.parent / "a100_nvlink_packet_v1" / (
    "candidate-profile.json"
)
PROTECTED_CANDIDATE_SHA256 = (
    "899712c4734f7a6b410d80231291663a404511528d46aab7497b73831e0e354f"
)
IMPLEMENTATION_PATHS = (
    STUDY_ROOT / "case_matrix.py",
    STUDY_ROOT / "nvlink_packet_lane.cu",
    STUDY_ROOT / "run_study.py",
    STUDY_ROOT / "sha256.h",
)
FREEZE_SHA256 = "f0ab026e054873a56614af63ab3a7ae3219dc0b045423808cb41522910fa6da6"
LOCAL_BULK_ROOT = REPOSITORY_ROOT.parents[1] / "wave-runs" / "traf70"
CELL_SCHEMA = "simllm-a100-nvlink-packet-cell-v2"
MANIFEST_SCHEMA = "simllm-a100-nvlink-packet-attempt-manifest-v2"
CELL_TIMEOUT_SECONDS = 10 * 60

_ACTIVE_CHILD: subprocess.Popen[str] | None = None
_STOP_SIGNAL: int | None = None


@dataclass(frozen=True, kw_only=True)
class Cell:
    index: int
    cell_id: str
    frame: str
    case_names: tuple[str, ...]


class StudyStopped(RuntimeError):
    """Raised after a scheduler or operator stop signal."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--array-index", type=int)
    selection.add_argument("--cell-id")
    selection.add_argument("--all-cells", action="store_true")
    parser.add_argument("--mode", choices=("mock", "hardware"), default="mock")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output-root", type=Path, default=LOCAL_BULK_ROOT)
    parser.add_argument("--expected-head", default="")
    parser.add_argument("--freeze-sha256", default=FREEZE_SHA256)
    parser.add_argument("--pace-seconds", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--list-cells", action="store_true")
    parser.add_argument("--pending-indices", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    freeze = _load_freeze(args.freeze_sha256)
    _verify_protected_candidate()
    cells = _cells(freeze)
    if args.list_cells:
        print(json.dumps([asdict(cell) for cell in cells], indent=2, sort_keys=True))
        return 0
    if args.pending_indices:
        print(",".join(str(index) for index in _pending_indices(cells, args)))
        return 0
    if args.check_only:
        _check_cli(args, cells)
        print(
            f"TRAF-70 local check passed: {len(freeze['catalog'])} cases, "
            f"{len(cells)} resumable cells, freeze {FREEZE_SHA256}"
        )
        return 0
    selected = _select_cells(args, cells)
    _check_cli(args, cells)
    if args.dry_run:
        payload = {
            "schema": CELL_SCHEMA,
            "mode": args.mode,
            "freeze_sha256": args.freeze_sha256,
            "binary": None if args.binary is None else str(args.binary),
            "output_root": str(args.output_root),
            "cells": [
                {
                    **asdict(cell),
                    "point_count": len(_expanded_points(cell, freeze)),
                }
                for cell in selected
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    _install_signal_handlers()
    failures = 0
    for offset, cell in enumerate(selected):
        if offset and args.pace_seconds:
            time.sleep(args.pace_seconds)
        try:
            status = _run_cell(cell, freeze, args)
            print(f"{cell.cell_id}: {status}")
        except (RuntimeError, OSError, ValueError) as error:
            failures += 1
            print(f"{cell.cell_id}: FAILED: {error}", file=sys.stderr)
            if args.mode == "hardware":
                break
    return 1 if failures else 0


def _load_freeze(expected_digest: str) -> dict[str, Any]:
    if expected_digest != FREEZE_SHA256:
        raise ValueError(
            f"requested freeze {expected_digest} does not equal committed freeze {FREEZE_SHA256}"
        )
    actual = _sha256(EXPECTATIONS_PATH)
    if actual != expected_digest:
        raise RuntimeError(f"TRAF-70 expectations digest is {actual}, expected {expected_digest}")
    with open(EXPECTATIONS_PATH, encoding="utf-8", newline="") as handle:
        freeze = json.load(handle)
    catalog = freeze.get("catalog")
    if not isinstance(catalog, list) or len(catalog) != 80:
        raise RuntimeError("TRAF-70 freeze must contain exactly 80 cases")
    ordinals = [case.get("ordinal") for case in catalog]
    if ordinals != list(range(1, 81)):
        raise RuntimeError("TRAF-70 case ordinals are not the frozen 1 through 80")
    names = [case.get("stable_name") for case in catalog]
    if len(set(names)) != 80:
        raise RuntimeError("TRAF-70 stable case names are not unique")
    return freeze


def _verify_protected_candidate() -> None:
    if _sha256(PROTECTED_CANDIDATE_PROFILE_PATH) != PROTECTED_CANDIDATE_SHA256:
        raise RuntimeError("protected A100 candidate changed before TRAF-70 scoring")
    with open(PROTECTED_CANDIDATE_PROFILE_PATH, encoding="utf-8", newline="") as handle:
        profile = json.load(handle)
    if profile.get("status") != "candidate":
        raise RuntimeError("TRAF-70 profile must remain candidate before hardware scoring")
    if profile.get("evidence_class") != "declared_candidate_not_hardware_measurement":
        raise RuntimeError("TRAF-70 candidate profile makes an invalid evidence claim")


def _cells(freeze: dict[str, Any]) -> tuple[Cell, ...]:
    catalog = freeze["catalog"]
    cells = [
        Cell(
            index=index,
            cell_id=f"isolated-{index + 1:03d}",
            frame="isolated",
            case_names=(str(case["stable_name"]),),
        )
        for index, case in enumerate(catalog)
    ]
    corners = []
    for case in catalog:
        corner = str(case["corner"])
        if corner not in corners:
            corners.append(corner)
    for corner in corners:
        cells.append(
            Cell(
                index=len(cells),
                cell_id=f"corner-frame-{corner.replace('_', '-')}",
                frame="corner_frame",
                case_names=tuple(
                    str(case["stable_name"]) for case in catalog if case["corner"] == corner
                ),
            )
        )
    cells.append(
        Cell(
            index=len(cells),
            cell_id="all-corners-frame",
            frame="all_corners_frame",
            case_names=tuple(str(case["stable_name"]) for case in catalog),
        )
    )
    if len(cells) != 86:
        raise RuntimeError(f"TRAF-70 cell construction produced {len(cells)}, expected 86")
    return tuple(cells)


def _select_cells(args: argparse.Namespace, cells: tuple[Cell, ...]) -> tuple[Cell, ...]:
    if args.all_cells:
        return cells
    if args.cell_id is not None:
        selected = tuple(cell for cell in cells if cell.cell_id == args.cell_id)
        if not selected:
            raise ValueError(f"unknown TRAF-70 cell {args.cell_id!r}")
        return selected
    index = 0 if args.array_index is None else args.array_index
    if index < 0 or index >= len(cells):
        raise ValueError(f"array index must be in [0, {len(cells) - 1}]")
    return (cells[index],)


def _check_cli(args: argparse.Namespace, cells: tuple[Cell, ...]) -> None:
    del cells
    if args.pace_seconds < 0 or args.pace_seconds > 60:
        raise ValueError("pace seconds must be in [0, 60]")
    if args.expected_head and re.fullmatch(r"[0-9a-f]{40}", args.expected_head) is None:
        raise ValueError("expected head must be a full lowercase Git SHA")
    if not args.dry_run and not args.check_only and args.binary is None:
        raise ValueError("--binary is required for a result-producing run")
    if args.binary is not None and not args.dry_run and not os.access(args.binary, os.X_OK):
        raise FileNotFoundError(f"TRAF-70 producer binary is not executable: {args.binary}")
    if args.mode == "hardware" and not args.expected_head:
        raise ValueError("hardware mode requires --expected-head")


def _expanded_points(cell: Cell, freeze: dict[str, Any]) -> tuple[object, ...]:
    by_name = {str(case["stable_name"]): case for case in freeze["catalog"]}
    points = []
    for case_name in cell.case_names:
        case = by_name[case_name]
        points.extend(expand_producers(case, points_for_case(case)))
    if cell.frame == "all_corners_frame":
        points.extend(protocol_validation_points(cell.cell_id))
    return tuple(points)


def _run_cell(cell: Cell, freeze: dict[str, Any], args: argparse.Namespace) -> str:
    cell_root = args.output_root / args.freeze_sha256 / "cells" / cell.cell_id
    cell_root.mkdir(parents=True, exist_ok=True)
    if any(
        _verify_attempt(path) and _attempt_matches(path, args)
        for path in sorted(cell_root.glob("attempt-*"))
    ):
        return "already complete and digest verified"
    attempts = sorted(cell_root.glob("attempt-*"))
    attempt = cell_root / f"attempt-{len(attempts) + 1:04d}"
    attempt.mkdir()
    try:
        return _produce_attempt(attempt, cell, freeze, args)
    except BaseException as error:
        stopped = {
            "schema": CELL_SCHEMA,
            "status": "stopped",
            "cell_id": cell.cell_id,
            "signal": _STOP_SIGNAL,
            "error_type": type(error).__name__,
            "error": str(error),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_exclusive(attempt / "STOPPED.json", stopped)
        raise


def _produce_attempt(
    attempt: Path,
    cell: Cell,
    freeze: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    points = _expanded_points(cell, freeze)
    plan = {
        "schema": CELL_SCHEMA,
        "status": "planned",
        "cell": asdict(cell),
        "mode": args.mode,
        "freeze_sha256": args.freeze_sha256,
        "protected_candidate_profile_sha256": _sha256(
            PROTECTED_CANDIDATE_PROFILE_PATH
        ),
        "implementation_sha256": _implementation_digest(),
        "producer_binary_sha256": _sha256(args.binary),
        "expected_head": args.expected_head or None,
        "point_count": len(points),
        "nccl_scope": "protocol validation only; never packet-format authority",
    }
    _write_json_exclusive(attempt / "plan.json", plan)
    _write_points(attempt / "points.tsv", points)
    environment = {
        "schema": CELL_SCHEMA,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "hostname": os.environ.get("HOSTNAME", "unknown"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_job_node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "source_head": args.expected_head or _local_head(),
    }
    _write_json_exclusive(attempt / "environment.json", environment)
    if args.mode == "hardware":
        _collect_hardware_guard(attempt, "before")
    else:
        _write_text_exclusive(
            attempt / "guards_before.txt",
            "mode=mock\nhardware_guards=not_applicable\n",
        )
    result_path = attempt / "results.jsonl"
    command = [
        str(args.binary),
        "--points",
        str(attempt / "points.tsv"),
        "--output",
        str(result_path),
        "--mode",
        args.mode,
    ]
    completed = _run_child(command, timeout=CELL_TIMEOUT_SECONDS)
    _write_text_exclusive(attempt / "stdout.txt", completed.stdout)
    _write_text_exclusive(attempt / "stderr.txt", completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"producer exited {completed.returncode}")
    if not result_path.is_file():
        raise RuntimeError("producer did not create results.jsonl")
    summary = _summarize_results(result_path, len(points), args.mode)
    _write_json_exclusive(attempt / "summary.json", summary)
    if args.mode == "hardware":
        _collect_hardware_guard(attempt, "after")
    else:
        _write_text_exclusive(
            attempt / "guards_after.txt",
            "mode=mock\nhardware_guards=not_applicable\n",
        )
    manifest = _write_manifest(attempt, cell, args.freeze_sha256)
    _write_json_exclusive(
        attempt / "COMPLETE.json",
        {
            "schema": MANIFEST_SCHEMA,
            "status": "complete",
            "cell_id": cell.cell_id,
            "manifest_sha256": _sha256(manifest),
        },
    )
    if not _verify_attempt(attempt):
        raise RuntimeError("newly completed attempt failed its digest audit")
    return f"complete ({len(points)} points)"


def _write_points(path: Path, points: tuple[object, ...]) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(TSV_FIELDS)
        for point in points:
            writer.writerow(point_to_tsv_row(point))


def _run_child(command: Sequence[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    global _ACTIVE_CHILD
    _ACTIVE_CHILD = subprocess.Popen(
        list(command),
        cwd=REPOSITORY_ROOT,
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


def _collect_hardware_guard(attempt: Path, when: str) -> None:
    commands = (
        ("gpu_list", ("nvidia-smi", "-L")),
        ("topology", ("nvidia-smi", "topo", "-m")),
        (
            "clocks",
            (
                "nvidia-smi",
                "--query-gpu=index,name,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu",
                "--format=csv",
            ),
        ),
        (
            "processes",
            (
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ),
        ),
        ("nvlink_data", ("nvidia-smi", "nvlink", "-gt", "d")),
        ("nvlink_raw", ("nvidia-smi", "nvlink", "-gt", "r")),
        ("nvlink_errors", ("nvidia-smi", "nvlink", "-e")),
        ("nvlink_crc", ("nvidia-smi", "nvlink", "-ec")),
    )
    outputs = []
    records: dict[str, subprocess.CompletedProcess[str]] = {}
    for name, command in commands:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        records[name] = completed
        outputs.append(
            f"=== {name} returncode={completed.returncode} ===\n"
            f"{completed.stdout}{completed.stderr}"
        )
    _write_text_exclusive(attempt / f"guards_{when}.txt", "\n".join(outputs))
    if any(record.returncode != 0 for record in records.values()):
        raise RuntimeError(f"hardware guard command failed {when}")
    gpu_lines = [line for line in records["gpu_list"].stdout.splitlines() if line.strip()]
    if len(gpu_lines) != 4 or any("A100-SXM4-80GB" not in line for line in gpu_lines):
        raise RuntimeError("hardware cell requires exactly four A100-SXM4-80GB GPUs")
    topology = records["topology"].stdout
    gpu_rows = [line.split() for line in topology.splitlines() if re.match(r"^GPU[0-3]\s", line)]
    if len(gpu_rows) != 4 or any(row.count("NV4") != 3 for row in gpu_rows):
        raise RuntimeError("hardware cell is not a four-GPU NV4 direct mesh")
    if records["processes"].stdout.strip():
        raise RuntimeError("hardware cell has a competing compute process")


def _summarize_results(path: Path, expected_rows: int, mode: str) -> dict[str, Any]:
    rows = []
    with open(path, encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid result JSON at line {line_number}") from error
            rows.append(row)
    if len(rows) != expected_rows:
        raise RuntimeError(f"producer returned {len(rows)} rows, expected {expected_rows}")
    if any(row.get("mode") != mode for row in rows):
        raise RuntimeError("producer result mode does not match the requested mode")
    forbidden = {"candidate_packet_count", "candidate_raw_bytes", "predicted_raw_bytes"}
    for row in rows:
        if forbidden.intersection(row):
            raise RuntimeError("producer mixed candidate-derived fields into observations")
        if row.get("schema") != "simllm-a100-nvlink-packet-observation-v2":
            raise RuntimeError("producer returned an unexpected observation schema")
        required = {
            "observed_data_bytes",
            "observed_raw_bytes",
            "observed_counter_deltas",
            "destination_checksum",
            "ordering_ledger",
            "applied_controls",
            "applied_control_sha256",
            "throttle_verdict",
            "copy_engine_host_enqueue_count",
            "replay_recovery_crc_ecc_deltas",
            "latency_flow_ledger",
            "bulk_flow_ledger",
            "drain_time_us",
            "candidate_blind_fit_membership",
        }
        missing = sorted(required.difference(row))
        if missing:
            raise RuntimeError(f"producer row omitted required fields: {missing}")
        if row.get("producer") == "copy_engine_reference":
            enqueue_count = row.get("copy_engine_host_enqueue_count")
            message_count = row.get("message_count")
            if not isinstance(enqueue_count, int) or not isinstance(message_count, int):
                raise RuntimeError("copy-engine batch ledger is not integral")
            if message_count > 1 and enqueue_count >= message_count:
                raise RuntimeError("copy engine enqueued once per logical message")
    return {
        "schema": CELL_SCHEMA,
        "status": "mock_complete" if mode == "mock" else "hardware_unscored",
        "row_count": len(rows),
        "producer_counts": {
            producer: sum(row.get("producer") == producer for row in rows)
            for producer in sorted({str(row.get("producer")) for row in rows})
        },
        "checksum_failure_count": sum(row.get("checksum_ok") is not True for row in rows),
        "measurement_claim": False if mode == "mock" else "pending_scoring",
    }


def _write_manifest(attempt: Path, cell: Cell, freeze_digest: str) -> Path:
    payloads = []
    for path in sorted(attempt.iterdir()):
        if not path.is_file() or path.name in {"manifest.json", "COMPLETE.json"}:
            continue
        payloads.append({"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    manifest_path = attempt / "manifest.json"
    _write_json_exclusive(
        manifest_path,
        {
            "schema": MANIFEST_SCHEMA,
            "cell_id": cell.cell_id,
            "freeze_sha256": freeze_digest,
            "payloads": payloads,
        },
    )
    return manifest_path


def _verify_attempt(attempt: Path) -> bool:
    manifest_path = attempt / "manifest.json"
    complete_path = attempt / "COMPLETE.json"
    if not manifest_path.is_file() or not complete_path.is_file():
        return False
    try:
        with open(manifest_path, encoding="utf-8", newline="") as handle:
            manifest = json.load(handle)
        with open(complete_path, encoding="utf-8", newline="") as handle:
            complete = json.load(handle)
        if complete.get("manifest_sha256") != _sha256(manifest_path):
            return False
        for payload in manifest.get("payloads", []):
            path = attempt / str(payload["path"])
            if (
                not path.is_file()
                or path.stat().st_size != payload["bytes"]
                or _sha256(path) != payload["sha256"]
            ):
                return False
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _attempt_matches(attempt: Path, args: argparse.Namespace) -> bool:
    try:
        with open(attempt / "plan.json", encoding="utf-8", newline="") as handle:
            plan = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if plan.get("implementation_sha256") != _implementation_digest():
        return False
    if plan.get("protected_candidate_profile_sha256") != _sha256(
        PROTECTED_CANDIDATE_PROFILE_PATH
    ):
        return False
    if args.binary is None or plan.get("producer_binary_sha256") != _sha256(args.binary):
        return False
    return not args.expected_head or plan.get("expected_head") == args.expected_head


def _pending_indices(cells: tuple[Cell, ...], args: argparse.Namespace) -> tuple[int, ...]:
    if args.expected_head and re.fullmatch(r"[0-9a-f]{40}", args.expected_head) is None:
        raise ValueError("expected head must be a full lowercase Git SHA")
    root = args.output_root / args.freeze_sha256 / "cells"
    pending = []
    for cell in cells:
        complete = False
        for attempt in sorted((root / cell.cell_id).glob("attempt-*")):
            if not _verify_attempt(attempt):
                continue
            try:
                with open(attempt / "plan.json", encoding="utf-8", newline="") as handle:
                    plan = json.load(handle)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if plan.get("mode") != "hardware":
                continue
            if plan.get("implementation_sha256") != _implementation_digest():
                continue
            if plan.get("protected_candidate_profile_sha256") != _sha256(
                PROTECTED_CANDIDATE_PROFILE_PATH
            ):
                continue
            if args.expected_head and plan.get("expected_head") != args.expected_head:
                continue
            complete = True
            break
        if not complete:
            pending.append(cell.index)
    return tuple(pending)


def _install_signal_handlers() -> None:
    for name in ("SIGHUP", "SIGINT", "SIGTERM"):
        signal_number = getattr(signal, name, None)
        if signal_number is not None:
            signal.signal(signal_number, _handle_stop)


def _handle_stop(signal_number: int, _frame: object) -> None:
    global _STOP_SIGNAL
    _STOP_SIGNAL = signal_number
    if _ACTIVE_CHILD is not None:
        _ACTIVE_CHILD.terminate()


def _local_head() -> str | None:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _write_json_exclusive(path: Path, payload: object) -> None:
    _write_text_exclusive(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_text_exclusive(path: Path, text: str) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in IMPLEMENTATION_PATHS:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
