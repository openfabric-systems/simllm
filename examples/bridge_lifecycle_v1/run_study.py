"""Run the frozen BRIDGE-3 child-lifetime study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SIMLLM_BASE_COMMIT = "90ada43070adb3b1e624b6819aff34d8620e8571"
AUTHORING_HTSIM_COMMIT = "4885c647eecdfdf81479d1df052223c016ad086b"
EXPECTATIONS_COMMIT = "78ef408c4d14f599359c673e146ffa9ecc012d2e"
CELLS = (
    {"cell": "D1", "mode": "diagnostic", "max_workers": None, "targets": 1},
    {"cell": "P2", "mode": "prepared", "max_workers": 2, "targets": 2},
    {"cell": "P4", "mode": "prepared", "max_workers": 4, "targets": 4},
)
READY_TIMEOUT_S = 10.0
OWNER_EXIT_TIMEOUT_S = 10.0
POLL_TIMEOUT_S = 5.0
POLL_INTERVAL_S = 0.02
STANDIN_SLEEP_S = 30.0
VLLM_FIXTURE = REPO_ROOT / "examples" / "m4" / "fixtures" / "vllm-m2-steps.jsonl"
STANDIN = HERE / "standin_htsim.py"


def _configured_root() -> Path:
    raw = os.environ.get("SIMLLM_WAVE5_RUN_ROOT")
    if not raw:
        raise ValueError("SIMLLM_WAVE5_RUN_ROOT must name the external branch run root")
    return Path(raw).resolve()


def _validate_output(out: Path) -> None:
    root = _configured_root()
    try:
        out.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError("study output must remain under SIMLLM_WAVE5_RUN_ROOT") from error


def _configured_executable(path: Path, option: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"{option} must name an executable file: {resolved}")
    return resolved


def _observed_htsim_gitlink() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD:third_party/htsim"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_registry(arguments: argparse.Namespace) -> dict[str, object]:
    _validate_output(arguments.out)
    real_htsim = _configured_executable(arguments.real_htsim, "--real-htsim")
    real_txt2bin = _configured_executable(arguments.real_txt2bin, "--real-txt2bin")
    if tuple(row["targets"] for row in CELLS) != (1, 2, 4):
        raise AssertionError("frozen target counts drifted")
    if tuple(row["max_workers"] for row in CELLS) != (None, 2, 4):
        raise AssertionError("frozen worker matrix drifted")
    if len({row["cell"] for row in CELLS}) != len(CELLS):
        raise AssertionError("frozen cells must be unique")
    if (READY_TIMEOUT_S, OWNER_EXIT_TIMEOUT_S, POLL_TIMEOUT_S, POLL_INTERVAL_S) != (
        10.0,
        10.0,
        5.0,
        0.02,
    ):
        raise AssertionError("frozen lifecycle time bounds drifted")
    if STANDIN_SLEEP_S <= READY_TIMEOUT_S + POLL_TIMEOUT_S:
        raise AssertionError("stand-in must outlive readiness and post-kill polling")
    observed_gitlink = _observed_htsim_gitlink()
    if re.fullmatch(r"[0-9a-f]{40}", observed_gitlink) is None:
        raise AssertionError(
            f"observed HTSIM gitlink is not a full commit hash: {observed_gitlink}"
        )
    required = (
        REPO_ROOT / "examples" / "bridge_persistent_v1" / "run_study.py",
        VLLM_FIXTURE,
        STANDIN,
        REPO_ROOT / "simllm" / "backends" / "htsim_rnic.py",
        REPO_ROOT / "simllm" / "backends" / "step_sink.py",
    )
    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"frozen study inputs are missing: {missing}")
    return {
        "artifacts_created": False,
        "authoring_htsim_commit": AUTHORING_HTSIM_COMMIT,
        "cells": CELLS,
        "expectations_commit": EXPECTATIONS_COMMIT,
        "observed_htsim_gitlink": observed_gitlink,
        "real_htsim": str(real_htsim),
        "real_txt2bin": str(real_txt2bin),
        "standin_sleep_s": STANDIN_SLEEP_S,
        "time_bounds_s": {
            "owner_exit": OWNER_EXIT_TIMEOUT_S,
            "poll": POLL_TIMEOUT_S,
            "poll_interval": POLL_INTERVAL_S,
            "ready": READY_TIMEOUT_S,
        },
    }


def _dims_tp8() -> Any:
    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=32,
        hidden_size=4096,
        intermediate_size=14336 // 8,
        num_heads=32 // 8,
        num_kv_heads=1,
        head_size=128,
        vocab_size=128256,
        dtype_bytes=2,
    )


def _standin_to_binary(
    goal_path: str | Path,
    bin_path: str | Path | None = None,
    tool: str | Path | None = None,
) -> Path:
    del tool
    source = Path(goal_path)
    destination = Path(bin_path) if bin_path is not None else source.with_suffix(".bin")
    destination.write_bytes(b"simllm-lifecycle-standin-v1\n")
    return destination


def _worker(arguments: argparse.Namespace) -> None:
    import simllm.backends.step_sink as step_sink_module
    from simllm.backends import (
        HtsimPersistentStepSink,
        HtsimStepSink,
        HtsimStepSinkConfig,
    )
    from simllm.core import step_records_from_jsonl
    from simllm.placement import declared_manifest

    records = tuple(step_records_from_jsonl(VLLM_FIXTURE))[: arguments.targets]
    if len(records) != arguments.targets:
        raise ValueError("worker did not load the requested frozen record count")
    environment_binary = Path(arguments.binary).resolve()
    os.environ["SIMLLM_HTSIM_RNIC"] = str(environment_binary)
    if arguments.binary_kind == "standin":
        step_sink_module.to_binary = _standin_to_binary
    else:
        os.environ["SIMLLM_TXT2BIN"] = str(Path(arguments.txt2bin).resolve())
    manifest = declared_manifest(tp=8, pp=1, dp=1)
    config = HtsimStepSinkConfig(
        profile="rnic-nn-fluid",
        tp_ranks=manifest.group_ranks(0, "tp"),
        dims=_dims_tp8(),
        workdir=arguments.workdir,
        linkspeed_bps=400_000_000_000,
        unsafe_disable_child_lifetime_binding=arguments.unsafe,
    )
    if arguments.mode == "diagnostic":
        HtsimStepSink(config)(records[0])
    else:
        with HtsimPersistentStepSink(
            config,
            max_workers=arguments.max_workers,
        ) as persistent:
            persistent.prepare(records)
    raise RuntimeError("lifecycle worker returned before its owner was killed")


def _parse_proc_stat(pid: int) -> dict[str, object] | None:
    path = Path("/proc") / str(pid) / "stat"
    try:
        stat = path.read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    closing = stat.rfind(")")
    if closing < 0:
        raise ValueError(f"cannot parse procfs identity for PID {pid}")
    fields = stat[closing + 2 :].split()
    if len(fields) <= 19:
        raise ValueError(f"incomplete procfs identity for PID {pid}")
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "start_time_token": fields[19],
        "state": fields[0],
    }


def _snapshot(marker: dict[str, object]) -> dict[str, object]:
    pid = int(marker["child_pid"])
    state = _parse_proc_stat(pid)
    if state is None:
        return {
            "exists": False,
            "pid": pid,
            "ppid": None,
            "start_time_token": None,
            "state": None,
        }
    expected_start = str(marker["start_time_token"])
    if state["start_time_token"] != expected_start:
        raise RuntimeError(
            f"PID reuse for target {pid}: expected start {expected_start}, "
            f"observed {state['start_time_token']}"
        )
    return {"exists": True, **state}


def _read_markers(
    directory: Path,
    *,
    nonce: str,
    owner_pid: int,
    targets: int,
) -> list[dict[str, object]]:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        paths = sorted(directory.glob("*.json"))
        if len(paths) > targets:
            raise RuntimeError(f"marker count exceeded target count {targets}")
        if len(paths) == targets:
            markers = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
            break
        time.sleep(POLL_INTERVAL_S)
    else:
        raise TimeoutError(f"only {len(list(directory.glob('*.json')))} of {targets} markers appeared")
    child_pids = set()
    for marker in markers:
        if marker.get("schema") != "simllm-child-lifetime-marker-v1":
            raise ValueError(f"unexpected marker schema: {marker}")
        if marker.get("run_nonce") != nonce or int(marker["owner_pid"]) != owner_pid:
            raise ValueError(f"marker ownership disagreement: {marker}")
        child_pid = int(marker["child_pid"])
        if child_pid in child_pids:
            raise ValueError(f"duplicate child PID marker: {child_pid}")
        child_pids.add(child_pid)
        if len(str(marker.get("command_sha256", ""))) != 64:
            raise ValueError(f"invalid marker command digest: {marker}")
        ready_state = _snapshot(marker)
        if not ready_state["exists"]:
            raise RuntimeError(f"target {child_pid} exited during marker validation")
    return markers


def _wait_for_exec(
    markers: list[dict[str, object]],
    *,
    expected_binary: Path,
    binary_kind: str,
) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        ready = True
        for marker in markers:
            state = _snapshot(marker)
            if not state["exists"]:
                raise RuntimeError(f"target {state['pid']} exited before kill readiness")
            pid = int(state["pid"])
            if binary_kind == "real":
                try:
                    executable = (Path("/proc") / str(pid) / "exe").resolve(strict=True)
                except FileNotFoundError:
                    ready = False
                    break
                if executable != expected_binary.resolve():
                    ready = False
                    break
            else:
                try:
                    command = (Path("/proc") / str(pid) / "cmdline").read_bytes()
                except FileNotFoundError:
                    ready = False
                    break
                if os.fsencode(str(expected_binary)) not in command.split(b"\0"):
                    ready = False
                    break
        if ready:
            return
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"{binary_kind} targets did not reach executable readiness")


def _poll_remaining(markers: list[dict[str, object]]) -> tuple[list[dict[str, object]], int]:
    started_ns = time.perf_counter_ns()
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while True:
        snapshots = [_snapshot(marker) for marker in markers]
        if not any(snapshot["exists"] for snapshot in snapshots):
            return snapshots, time.perf_counter_ns() - started_ns
        if time.monotonic() >= deadline:
            return snapshots, time.perf_counter_ns() - started_ns
        time.sleep(POLL_INTERVAL_S)


def _cleanup_negative(markers: list[dict[str, object]]) -> None:
    for marker in markers:
        snapshot = _snapshot(marker)
        if snapshot["exists"]:
            os.kill(int(snapshot["pid"]), signal.SIGKILL)
    snapshots, _ = _poll_remaining(markers)
    survivors = [snapshot for snapshot in snapshots if snapshot["exists"]]
    if survivors:
        raise RuntimeError(f"negative-control cleanup left targets: {survivors}")


def _owner_command(
    *,
    cell: dict[str, object],
    workdir: Path,
    binary: Path,
    binary_kind: str,
    txt2bin: Path,
    unsafe: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--mode",
        str(cell["mode"]),
        "--max-workers",
        str(cell["max_workers"] or 1),
        "--targets",
        str(cell["targets"]),
        "--workdir",
        str(workdir),
        "--binary",
        str(binary),
        "--binary-kind",
        binary_kind,
        "--txt2bin",
        str(txt2bin),
    ]
    if unsafe:
        command.append("--unsafe")
    return command


def _run_kill_cell(
    *,
    out: Path,
    cell: dict[str, object],
    binary: Path,
    binary_kind: str,
    txt2bin: Path,
    unsafe: bool,
) -> dict[str, object]:
    binding = "unsafe-unmanaged" if unsafe else "managed"
    run_dir = out / str(cell["cell"]) / f"{binary_kind}-{binding}"
    marker_dir = run_dir / "markers"
    workdir = run_dir / "work"
    marker_dir.mkdir(parents=True)
    workdir.mkdir()
    nonce = f"{cell['cell']}-{binary_kind}-{binding}-{uuid.uuid4().hex}"
    environment = os.environ.copy()
    environment["SIMLLM_CHILD_LIFETIME_MARKER_DIR"] = str(marker_dir)
    environment["SIMLLM_CHILD_LIFETIME_RUN_NONCE"] = nonce
    owner = subprocess.Popen(
        _owner_command(
            cell=cell,
            workdir=workdir,
            binary=binary,
            binary_kind=binary_kind,
            txt2bin=txt2bin,
            unsafe=unsafe,
        ),
        env=environment,
        start_new_session=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        markers = _read_markers(
            marker_dir,
            nonce=nonce,
            owner_pid=owner.pid,
            targets=int(cell["targets"]),
        )
        _wait_for_exec(markers, expected_binary=binary, binary_kind=binary_kind)
        if owner.poll() is not None:
            raise RuntimeError(f"owner {owner.pid} exited before registered SIGTERM")
        os.kill(owner.pid, signal.SIGTERM)
        owner.wait(timeout=OWNER_EXIT_TIMEOUT_S)
        stdout, stderr = owner.communicate(timeout=1.0)
        snapshots, poll_elapsed_ns = _poll_remaining(markers)
        remaining = sum(bool(snapshot["exists"]) for snapshot in snapshots)
        row = {
            "binary_kind": binary_kind,
            "binding": binding,
            "cell": cell["cell"],
            "invocation": cell["mode"],
            "markers": markers,
            "max_workers": cell["max_workers"],
            "owner_pid": owner.pid,
            "owner_returncode": owner.returncode,
            "owner_stderr": stderr,
            "owner_stdout": stdout,
            "poll_elapsed_ns": poll_elapsed_ns,
            "post_kill_targets": snapshots,
            "remaining_count": remaining,
            "targeted_count": cell["targets"],
        }
        if unsafe:
            _cleanup_negative(markers)
            row["negative_cleanup_verified"] = True
        print(
            f"cell={cell['cell']} binary={binary_kind} binding={binding} "
            f"remaining={remaining}/{cell['targets']}"
        )
        return row
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=OWNER_EXIT_TIMEOUT_S)


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_study(arguments: argparse.Namespace, plan: dict[str, object]) -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("the registered SIGTERM and procfs study requires Linux")
    if arguments.out.exists():
        raise FileExistsError("--out must not exist; choose a fresh external directory")
    standin = _configured_executable(STANDIN, "stand-in")
    arguments.out.mkdir(parents=True)
    rows = []
    for cell in CELLS:
        rows.append(
            _run_kill_cell(
                out=arguments.out,
                cell=cell,
                binary=standin,
                binary_kind="standin",
                txt2bin=arguments.real_txt2bin,
                unsafe=True,
            )
        )
        rows.append(
            _run_kill_cell(
                out=arguments.out,
                cell=cell,
                binary=standin,
                binary_kind="standin",
                txt2bin=arguments.real_txt2bin,
                unsafe=False,
            )
        )

    scored = []
    for cell in CELLS:
        unsafe_row = next(
            row
            for row in rows
            if row["cell"] == cell["cell"] and row["binding"] == "unsafe-unmanaged"
        )
        managed_row = next(
            row
            for row in rows
            if row["cell"] == cell["cell"] and row["binding"] == "managed"
        )
        targeted = int(cell["targets"])
        unsafe_count = int(unsafe_row["remaining_count"])
        managed_count = int(managed_row["remaining_count"])
        checks = {
            "managed_exact_zero": managed_count == 0,
            "signed_difference_exact": managed_count - unsafe_count == -targeted,
            "unsafe_exact_targeted": unsafe_count == targeted,
        }
        scored.append(
            {
                "cell": cell["cell"],
                "checks": checks,
                "genuine_risk": True,
                "managed_minus_unsafe": managed_count - unsafe_count,
                "managed_remaining": managed_count,
                "passed": all(checks.values()),
                "targeted": targeted,
                "unsafe_remaining": unsafe_count,
            }
        )

    real_cell = CELLS[0]
    real_row = _run_kill_cell(
        out=arguments.out,
        cell=real_cell,
        binary=arguments.real_htsim,
        binary_kind="real",
        txt2bin=arguments.real_txt2bin,
        unsafe=False,
    )
    real_passed = real_row["remaining_count"] == 0
    failed_scored = [row for row in scored if not row["passed"]]
    summary = {
        "schema": "simllm-bridge-lifecycle-study-v1",
        "chronology": {
            "expectations_commit": EXPECTATIONS_COMMIT,
            "implementation_commit": _git_revision(),
            "simllm_base_commit": SIMLLM_BASE_COMMIT,
        },
        "binary_sha256": {
            "real_htsim": hashlib.sha256(arguments.real_htsim.read_bytes()).hexdigest(),
            "real_txt2bin": hashlib.sha256(arguments.real_txt2bin.read_bytes()).hexdigest(),
            "standin": hashlib.sha256(standin.read_bytes()).hexdigest(),
        },
        "entailment_analysis": {
            "conclusion": "not entailed",
            "detail": (
                "raw owner exits and post-kill target states are captured before "
                "the exact U/M relation is evaluated; readiness and cleanup guards "
                "do not constrain either post-kill count"
            ),
        },
        "fatal_unscored": {
            "real_binary_managed_zero": real_passed,
            "targeting_cleanup_and_readiness": True,
        },
        "host": {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "system": platform.system(),
        },
        "htsim_provenance": {
            "authoring_source_audit_commit": plan["authoring_htsim_commit"],
            "observed_superproject_gitlink": plan["observed_htsim_gitlink"],
        },
        "raw_observations": rows,
        "real_binary_corroboration": real_row,
        "scored_relation_family": {
            "genuine_risk_passed": sum(
                bool(row["passed"] and row["genuine_risk"]) for row in scored
            ),
            "genuine_risk_total": sum(bool(row["genuine_risk"]) for row in scored),
            "instances": scored,
            "name": "BRIDGE-3 live killed-owner descendant count",
            "passed": sum(bool(row["passed"]) for row in scored),
            "total": len(scored),
        },
    }
    summary_path = arguments.out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"summary={summary_path}")
    print(
        f"scored={summary['scored_relation_family']['passed']}/{len(scored)} "
        f"real_binary={'PASS' if real_passed else 'FAIL'}"
    )
    if failed_scored:
        raise AssertionError(f"scored lifecycle failures: {failed_scored}")
    if not real_passed:
        raise AssertionError(f"real-binary corroboration failed: {real_row}")


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--mode", choices=("diagnostic", "prepared"), required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument("--targets", type=int, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-kind", choices=("standin", "real"), required=True)
    parser.add_argument("--txt2bin", type=Path, required=True)
    parser.add_argument("--unsafe", action="store_true")
    return parser


def main() -> None:
    if "--worker" in sys.argv[1:]:
        _worker(_worker_parser().parse_args())
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--real-htsim", required=True, type=Path)
    parser.add_argument("--real-txt2bin", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    plan = _validate_registry(arguments)
    if arguments.check_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    _run_study(arguments, plan)


if __name__ == "__main__":
    main()
