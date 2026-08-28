#!/usr/bin/env python3
"""Run or resume the single frozen TRAF-73 hardware-validation cell."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
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


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


traf70_cases = _load_module(
    "_traf73_traf70_case_matrix", TRAF70_ROOT / "case_matrix.py"
)
_previous_case_matrix = sys.modules.get("case_matrix")
sys.modules["case_matrix"] = traf70_cases
try:
    traf70_run = _load_module(
        "_traf73_traf70_run_study", TRAF70_ROOT / "run_study.py"
    )
finally:
    if _previous_case_matrix is None:
        del sys.modules["case_matrix"]
    else:
        sys.modules["case_matrix"] = _previous_case_matrix

EXPECTATIONS_PATH = HERE / "expectations.json"
EXPECTATIONS_COMMIT = "092080e682acaee9d68779c6ebb2195e97d0d6fb"
EXPECTATIONS_SHA256 = "9f50aadba0085a54e78c156d61837e4c7db19a498d8fef9c1aba7b32e0a163b4"
CELL_ID = "nv4-long-flow-incast"
CELL_SCHEMA = "simllm-nvlink-incast-validation-cell-v1"
MANIFEST_SCHEMA = "simllm-nvlink-incast-validation-attempt-manifest-v1"
OBSERVATION_SCHEMA = "simllm-a100-nvlink-packet-observation-v2"
CELL_TIMEOUT_SECONDS = 10 * 60
BULK_ROOT_ENV = "SIMLLM_NVINC_BULK_ROOT"

IMPLEMENTATION_PATHS = (
    Path(__file__),
    HERE / "run_merlin_cell.sbatch",
    TRAF70_ROOT / "case_matrix.py",
    TRAF70_ROOT / "nvlink_packet_lane.cu",
    TRAF70_ROOT / "run_study.py",
    TRAF70_ROOT / "sha256.h",
)

_ACTIVE_CHILD: subprocess.Popen[str] | None = None
_STOP_SIGNAL: int | None = None


class StudyStopped(RuntimeError):
    """Raised after a scheduler or operator stop signal."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mock", "hardware"), default="mock")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--expected-head", default="")
    parser.add_argument("--freeze-sha256", default=EXPECTATIONS_SHA256)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--pending", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    frozen = load_expectations(args.freeze_sha256)
    verify_preservation(frozen)
    points = campaign_points(frozen)
    output_root = resolve_output_root(args.output_root)
    check_arguments(args, output_root)
    if args.check_only:
        print(
            f"TRAF-73 check passed: {len(points)} rows, freeze {EXPECTATIONS_SHA256}"
        )
        return 0
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema": CELL_SCHEMA,
                    "cell_id": CELL_ID,
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
        print("" if complete_attempt(output_root, args) is not None else "0")
        return 0
    if complete_attempt(output_root, args) is not None:
        print(f"{CELL_ID}: already complete and digest verified")
        return 0
    install_signal_handlers()
    attempt = new_attempt_path(output_root)
    try:
        result = produce_attempt(attempt, points, frozen, args)
    except BaseException as error:
        write_json_exclusive(
            attempt / "STOPPED.json",
            {
                "schema": CELL_SCHEMA,
                "status": "stopped",
                "cell_id": CELL_ID,
                "signal": _STOP_SIGNAL,
                "error_type": type(error).__name__,
                "error": str(error),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
    print(f"{CELL_ID}: {result}")
    return 0


def load_expectations(expected_digest: str = EXPECTATIONS_SHA256) -> dict[str, Any]:
    if expected_digest != EXPECTATIONS_SHA256:
        raise ValueError("requested TRAF-73 freeze does not equal the committed freeze")
    if sha256(EXPECTATIONS_PATH) != EXPECTATIONS_SHA256:
        raise RuntimeError("TRAF-73 expectations digest changed")
    repository = subprocess.run(
        ("git", "rev-parse", "--is-inside-work-tree"),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if repository.returncode == 0:
        completed = subprocess.run(
            ("git", "merge-base", "--is-ancestor", EXPECTATIONS_COMMIT, "HEAD"),
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError("the TRAF-73 expectations commit is not an ancestor")
    with open(EXPECTATIONS_PATH, encoding="utf-8", newline="") as handle:
        frozen = json.load(handle)
    if frozen.get("study", {}).get("status") != "expectations_only":
        raise RuntimeError("TRAF-73 expectations status changed")
    return frozen


def verify_preservation(frozen: dict[str, Any]) -> None:
    lock = frozen["preservation_lock"]
    artifacts = lock["artifacts"]
    if len(artifacts) != lock["artifact_count"]:
        raise RuntimeError("TRAF-73 preservation inventory count changed")
    for artifact in artifacts:
        path = ROOT / str(artifact["path"])
        if not path.is_file():
            raise RuntimeError(f"preserved artifact is missing: {artifact['path']}")
        if path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"preservation lock mismatch: {artifact['path']}")


def campaign_points(frozen: dict[str, Any]) -> tuple[traf70_cases.SweepPoint, ...]:
    arm = frozen["hardware_arm"]
    rows = []
    for size_bytes in arm["flow_sizes_bytes"]:
        if size_bytes % arm["producer_payload_bytes"]:
            raise RuntimeError("flow size is not divisible by the producer payload")
        message_count = size_bytes // arm["producer_payload_bytes"]
        for degree in arm["degrees"]:
            sources = ",".join(str(source) for source in arm["senders_by_degree"][str(degree)])
            for repetition in range(arm["repetitions_per_cell"]):
                rows.append(
                    traf70_cases.SweepPoint(
                        case_name=f"TRAF73_NVINC_LONG_D{degree}",
                        point_id=(
                            f"TRAF73_NVINC_LONG_D{degree}:size={size_bytes}:"
                            f"repeat={repetition:02d}"
                        ),
                        producer=arm["producer"],
                        payload_bytes=arm["producer_payload_bytes"],
                        message_count=message_count,
                        source=1,
                        destination=arm["receiver"],
                        sources=sources,
                        destinations=str(arm["receiver"]),
                        pattern=(
                            "one_source"
                            if degree == 1
                            else "two_source_simultaneous"
                            if degree == 2
                            else "three_source_simultaneous"
                        ),
                    )
                )
    expected = (
        len(arm["flow_sizes_bytes"])
        * len(arm["degrees"])
        * arm["repetitions_per_cell"]
    )
    if len(rows) != expected:
        raise RuntimeError("TRAF-73 point expansion changed")
    return tuple(rows)


def resolve_output_root(cli_value: Path | None) -> Path:
    value = cli_value
    if value is None:
        configured = os.environ.get(BULK_ROOT_ENV)
        if not configured:
            raise ValueError(f"set {BULK_ROOT_ENV} or pass --output-root")
        value = Path(configured)
    return value.resolve()


def check_arguments(args: argparse.Namespace, output_root: Path) -> None:
    del output_root
    if args.expected_head and re.fullmatch(r"[0-9a-f]{40}", args.expected_head) is None:
        raise ValueError("expected head must be a full lowercase Git SHA")
    if args.mode == "hardware" and not args.expected_head:
        raise ValueError("hardware mode requires --expected-head")
    if not args.dry_run and not args.check_only and args.binary is None:
        raise ValueError("--binary is required for a result-producing run")
    if (
        args.binary is not None
        and not args.dry_run
        and not args.check_only
        and not os.access(args.binary, os.X_OK)
    ):
        raise FileNotFoundError(
            f"TRAF-73 producer binary is not executable: {args.binary}"
        )


def cell_root(output_root: Path) -> Path:
    return output_root / EXPECTATIONS_SHA256 / "cells" / CELL_ID


def complete_attempt(
    output_root: Path, args: argparse.Namespace
) -> Path | None:
    root = cell_root(output_root)
    for attempt in sorted(root.glob("attempt-*")):
        if verify_attempt(attempt) and attempt_matches(attempt, args):
            return attempt
    return None


def new_attempt_path(output_root: Path) -> Path:
    root = cell_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    attempts = sorted(root.glob("attempt-*"))
    path = root / f"attempt-{len(attempts) + 1:04d}"
    path.mkdir()
    return path


def produce_attempt(
    attempt: Path,
    points: tuple[traf70_cases.SweepPoint, ...],
    frozen: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    del frozen
    if args.binary is None:
        raise ValueError("producer binary is required")
    plan = {
        "schema": CELL_SCHEMA,
        "status": "planned",
        "cell_id": CELL_ID,
        "mode": args.mode,
        "freeze_sha256": args.freeze_sha256,
        "freeze_commit": EXPECTATIONS_COMMIT,
        "implementation_sha256": implementation_sha256(),
        "producer_source_sha256": sha256(TRAF70_ROOT / "nvlink_packet_lane.cu"),
        "producer_binary_sha256": sha256(args.binary),
        "expected_head": args.expected_head or None,
        "point_count": len(points),
    }
    write_json_exclusive(attempt / "plan.json", plan)
    write_points(attempt / "points.tsv", points)
    environment = {
        "schema": CELL_SCHEMA,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "hostname": os.environ.get("HOSTNAME", "unknown"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_job_node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "source_head": args.expected_head or local_head(),
    }
    write_json_exclusive(attempt / "environment.json", environment)
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
        timeout=CELL_TIMEOUT_SECONDS,
    )
    write_text_exclusive(attempt / "stdout.txt", completed.stdout)
    write_text_exclusive(attempt / "stderr.txt", completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"producer exited {completed.returncode}")
    summary = traf70_run._summarize_results(result_path, len(points), args.mode)
    write_json_exclusive(attempt / "summary.json", summary)
    if args.mode == "hardware":
        traf70_run._collect_hardware_guard(attempt, "after")
    else:
        write_text_exclusive(
            attempt / "guards_after.txt",
            "mode=mock\nhardware_guards=not_applicable\n",
        )
    manifest = write_manifest(attempt)
    write_json_exclusive(
        attempt / "COMPLETE.json",
        {
            "schema": MANIFEST_SCHEMA,
            "status": "complete",
            "cell_id": CELL_ID,
            "manifest_sha256": sha256(manifest),
        },
    )
    if not verify_attempt(attempt):
        raise RuntimeError("newly completed TRAF-73 attempt failed its digest audit")
    return f"complete ({len(points)} points)"


def write_points(
    path: Path, points: tuple[traf70_cases.SweepPoint, ...]
) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(traf70_cases.TSV_FIELDS)
        for point in points:
            writer.writerow(traf70_cases.point_to_tsv_row(point))


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


def write_manifest(attempt: Path) -> Path:
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
            "cell_id": CELL_ID,
            "freeze_sha256": EXPECTATIONS_SHA256,
            "payloads": payloads,
        },
    )
    return path


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
        if manifest.get("schema") != MANIFEST_SCHEMA:
            return False
        if complete.get("schema") != MANIFEST_SCHEMA:
            return False
        if complete.get("manifest_sha256") != sha256(manifest_path):
            return False
        names = set()
        for payload in manifest.get("payloads", []):
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
    if plan.get("mode") != args.mode:
        return False
    if plan.get("implementation_sha256") != implementation_sha256():
        return False
    if args.binary is None or plan.get("producer_binary_sha256") != sha256(args.binary):
        return False
    return not args.expected_head or plan.get("expected_head") == args.expected_head


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in IMPLEMENTATION_PATHS:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
