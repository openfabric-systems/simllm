#!/usr/bin/env python3
"""Execute the frozen LogGOPSim packet-reference acceptance study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, TypeVar

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from simllm.backends import (
    HtsimRnicConfig,
    LogGopsimConfig,
    LogGopsimFanInError,
    inspect_loggopsim_fan_in,
    run_htsim_rnic,
    run_loggopsim,
)
from simllm.goal import GoalTrace, to_binary
from simllm.traffic import ordered_pairwise_messages

STUDY_DIR = Path(__file__).resolve().parent
EXPECTATIONS_PATH = STUDY_DIR / "expectations.md"
PINNED_RECORD_PATH = REPOSITORY_ROOT / "examples" / "deployment_frontier_v1" / "result.json"

RESULT_SCHEMA = "simllm-loggopsim-acceptance-result-v1"
ATTEMPT_SCHEMA = "simllm-loggopsim-acceptance-attempt-v1"
EXPECTATIONS_COMMIT = "30a9af9dd6e424b1458eff8a0f97598efe5ebd03"
EXPECTATIONS_SHA256 = "ae189d42cd5889152a101d63feb86cb44004d67700c102e9a72a0deacbecd832"
PINNED_RECORD_SHA256 = "f2f216068bf5ba914853c62a2ee965ede0ebfc0a6f29e3d11cfa5f45eac359ad"
PINNED_LOGGOPSIM_SHA256 = "7e0f13ee3c87a20e9d2e94dbbd74c46075fd03df2f1b04d1ed9739c43ee0a2bf"
EXACT_G_STRING = "0.02"
LINKSPEED_BPS = 400_000_000_000
LATENCY_NS = 2_000
WALL_SAMPLES = 7
BATCHES = (1, 2, 4, 8, 16, 32)
CONFIGURATIONS = (
    "h100-two-node-serialized",
    "h100-nine-node-incast",
)
PACKET_PROFILE = "rnic-nn"

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class AcceptanceCell:
    """One flow set and its pinned packet-reference observation."""

    configuration_id: str
    batch_per_gpu: int
    payloads: tuple[int, ...]
    reference_kind: str
    reference_completion_ps: int

    @property
    def cell_id(self) -> str:
        family = "serialized" if self.serialized else "incast"
        return f"{family}-b{self.batch_per_gpu}"

    @property
    def serialized(self) -> bool:
        return self.configuration_id == CONFIGURATIONS[0]

    @property
    def acknowledge_fan_in(self) -> bool:
        return not self.serialized

    @property
    def rank_count(self) -> int:
        return len(self.payloads) + 1


@dataclass(frozen=True, slots=True)
class PreparedCell:
    """One rendered GOAL shared by both measured arms."""

    cell: AcceptanceCell
    goal_path: Path
    binary_goal_path: Path
    goal_text_sha256: str
    goal_binary_sha256: str


@dataclass(frozen=True, slots=True)
class NativeSample:
    """One retained native execution."""

    arm: str
    elapsed_seconds: float
    completion_ps: int
    argv: tuple[str, ...]
    evidence_path: str
    stdout_sha256: str
    stderr_sha256: str
    quiescent: bool
    source: str = "native-execution"

    def to_json(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "argv": list(self.argv),
            "completion_ps": self.completion_ps,
            "elapsed_seconds": self.elapsed_seconds,
            "evidence_path": self.evidence_path,
            "quiescent": self.quiescent,
            "source": self.source,
            "stderr_sha256": self.stderr_sha256,
            "stdout_sha256": self.stdout_sha256,
        }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _git_hash(revision: str) -> str:
    completed = _git("rev-parse", revision)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or f"cannot resolve {revision}")
    return completed.stdout.strip()


def _is_ancestor(older: str, newer: str) -> bool:
    return _git("merge-base", "--is-ancestor", older, newer).returncode == 0


def _outside_repository(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return True
    return False


def _begin_attempt(run_root: Path, args: argparse.Namespace) -> Path:
    root = run_root.resolve()
    if not _outside_repository(root):
        raise SystemExit("--run-dir must be outside the repository")
    root.mkdir(parents=True, exist_ok=True)
    attempts: list[tuple[int, Path]] = []
    for path in root.glob("attempt-*"):
        match = re.fullmatch(r"attempt-(\d+)", path.name)
        if match is not None and path.is_dir():
            attempts.append((int(match.group(1)), path))
    incomplete = [path for _, path in attempts if not (path / "verdict.json").is_file()]
    if incomplete:
        names = ", ".join(path.name for path in sorted(incomplete))
        raise SystemExit(f"cannot start a later attempt while verdict records are missing: {names}")
    number = max((number for number, _ in attempts), default=0) + 1
    attempt_dir = root / f"attempt-{number}"
    attempt_dir.mkdir(parents=False, exist_ok=False)
    _write_json(
        attempt_dir / "attempt.json",
        {
            "htsim_rnic": str(args.htsim_rnic.resolve()),
            "loggopsim": str(args.loggopsim.resolve()),
            "run_root": str(root),
            "started_unix_time_ns": time.time_ns(),
            "txt2bin": str(args.txt2bin.resolve()),
        },
    )
    return attempt_dir


def _portable_argv(argv: list[str] | tuple[str, ...], attempt_dir: Path) -> list[str]:
    portable = list(argv)
    executable = Path(portable[0]).name
    portable[0] = "LogGOPSim" if executable == "LogGOPSim" else "htsim_rnic"
    for flag in ("-f", "-goal", "-completion_csv"):
        if flag not in portable:
            continue
        index = portable.index(flag) + 1
        target = Path(portable[index]).resolve()
        try:
            portable[index] = target.relative_to(attempt_dir.resolve()).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"native path escapes the attempt directory: {target}") from exc
    absolute = [item for item in portable[1:] if Path(item).is_absolute()]
    if absolute:
        raise RuntimeError(f"portable argv retains absolute paths: {absolute}")
    return portable


def _bytes(value: str | bytes | None) -> bytes:
    if value is None:
        return b""
    return value if isinstance(value, bytes) else value.encode("utf-8")


class NativeEvidenceRecorder:
    """Retain stdout, stderr, argv, and side artifacts for every execution."""

    def __init__(self, attempt_dir: Path) -> None:
        self.attempt_dir = attempt_dir
        self._counts: dict[tuple[str, str], int] = defaultdict(int)

    def capture(
        self,
        *,
        arm: str,
        label: str,
        argv: list[str] | tuple[str, ...],
        completed: subprocess.CompletedProcess[Any],
        elapsed_seconds: float,
        completion_ps: int,
        quiescent: bool,
        completion_csv: Path | None = None,
    ) -> NativeSample:
        key = (arm, label)
        self._counts[key] += 1
        directory = self.attempt_dir / "native" / arm / label
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"execution-{self._counts[key]:04d}"
        stdout = _bytes(completed.stdout)
        stderr = _bytes(completed.stderr)
        stdout_path = directory / f"{stem}.stdout"
        stderr_path = directory / f"{stem}.stderr"
        manifest_path = directory / f"{stem}.json"
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
        artifacts: dict[str, object] = {}
        if completion_csv is not None:
            if not completion_csv.is_file():
                raise RuntimeError("packet runner did not produce its completion CSV")
            artifacts["completion_csv"] = {
                "path": completion_csv.relative_to(self.attempt_dir).as_posix(),
                "sha256": _sha256_path(completion_csv),
                "bytes": completion_csv.stat().st_size,
            }
        recorded_argv = _portable_argv(argv, self.attempt_dir)
        _write_json(
            manifest_path,
            {
                "arm": arm,
                "argv": recorded_argv,
                "artifacts": artifacts,
                "completion_ps": completion_ps,
                "elapsed_seconds": elapsed_seconds,
                "quiescent": quiescent,
                "returncode": completed.returncode,
                "source": "native-execution",
                "stderr_file": stderr_path.name,
                "stderr_sha256": _sha256_bytes(stderr),
                "stdout_file": stdout_path.name,
                "stdout_sha256": _sha256_bytes(stdout),
            },
        )
        return NativeSample(
            arm=arm,
            elapsed_seconds=elapsed_seconds,
            completion_ps=completion_ps,
            argv=tuple(recorded_argv),
            evidence_path=manifest_path.relative_to(self.attempt_dir).as_posix(),
            stdout_sha256=_sha256_bytes(stdout),
            stderr_sha256=_sha256_bytes(stderr),
            quiescent=quiescent,
        )


def _load_cells(record: dict[str, Any] | None = None) -> tuple[AcceptanceCell, ...]:
    if record is None:
        record = json.loads(PINNED_RECORD_PATH.read_text(encoding="utf-8"))
    selected = [row for row in record["points"] if row["configuration_id"] in CONFIGURATIONS]
    cells = []
    for row in selected:
        serialized = row["configuration_id"] == CONFIGURATIONS[0]
        observation = row["fabric_observation"]
        cells.append(
            AcceptanceCell(
                configuration_id=row["configuration_id"],
                batch_per_gpu=row["batch_per_gpu"],
                payloads=tuple(row["byte_partition"]["remote_flow_payload_bytes"]),
                reference_kind=("isolated_service_ps" if serialized else "concurrent_service_ps"),
                reference_completion_ps=observation[
                    "isolated_service_ps" if serialized else "concurrent_service_ps"
                ],
            )
        )
    cells.sort(key=lambda cell: (CONFIGURATIONS.index(cell.configuration_id), cell.batch_per_gpu))
    if len(cells) != 12:
        raise RuntimeError(f"pinned record produced {len(cells)} acceptance cells, not 12")
    for configuration in CONFIGURATIONS:
        batches = tuple(
            cell.batch_per_gpu for cell in cells if cell.configuration_id == configuration
        )
        if batches != BATCHES:
            raise RuntimeError(f"pinned batch sweep changed for {configuration}: {batches}")
    if any(len(cell.payloads) != (1 if cell.serialized else 8) for cell in cells):
        raise RuntimeError("pinned flow counts changed")
    return tuple(cells)


def _render_goal_text(cell: AcceptanceCell) -> str:
    trace = GoalTrace(cell.rank_count)
    messages = [
        (f"flow-{index}", index + 1, 0, payload) for index, payload in enumerate(cell.payloads)
    ]
    ordered_pairwise_messages(
        trace,
        ranks=list(range(cell.rank_count)),
        messages=messages,
        tag=20_020,
        operation_id=f"loggopsim-acceptance:{cell.cell_id}",
    )
    if tuple(message.payload_bytes for message in trace.messages) != cell.payloads:
        raise AssertionError("rendered GOAL changed the pinned byte partition")
    return trace.render()


def _prepare_goals(
    cells: tuple[AcceptanceCell, ...], attempt_dir: Path, txt2bin: Path
) -> dict[str, PreparedCell]:
    goal_dir = attempt_dir / "goals"
    goal_dir.mkdir(parents=True, exist_ok=True)
    prepared = {}
    for cell in cells:
        goal_path = goal_dir / f"{cell.cell_id}.goal"
        goal_path.write_text(_render_goal_text(cell), encoding="utf-8", newline="\n")
        binary_path = to_binary(
            goal_path,
            goal_dir / f"{cell.cell_id}.bin",
            tool=txt2bin,
        )
        prepared[cell.cell_id] = PreparedCell(
            cell=cell,
            goal_path=goal_path,
            binary_goal_path=binary_path,
            goal_text_sha256=_sha256_path(goal_path),
            goal_binary_sha256=_sha256_path(binary_path),
        )
    return prepared


def _ideal_config(prepared: PreparedCell) -> LogGopsimConfig:
    return LogGopsimConfig(
        goal_bin=prepared.binary_goal_path,
        latency_ns=LATENCY_NS,
        overhead_ns=0,
        message_gap_ns=0,
        byte_gap_ns=float(EXACT_G_STRING),
        byte_gap_ns_string=EXACT_G_STRING,
        byte_overhead_ns=0,
        rendezvous_threshold_bytes=max(prepared.cell.payloads) + 1,
        network_type="LogGP",
    )


class _SubprocessProxy:
    def __init__(self, run: Callable[..., subprocess.CompletedProcess[Any]]) -> None:
        self.run = run


def _run_ideal_sample(
    prepared: PreparedCell,
    *,
    binary: Path,
    recorder: NativeEvidenceRecorder,
    label: str,
    timeout_s: int,
) -> NativeSample:
    module = importlib.import_module("simllm.backends.loggopsim")
    original_subprocess = module.subprocess
    captured: list[tuple[list[str], subprocess.CompletedProcess[Any]]] = []

    def capture(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        check = kwargs.pop("check", False)
        completed = subprocess.run(argv, check=check, **kwargs)
        captured.append((list(argv), completed))
        return completed

    module.subprocess = _SubprocessProxy(capture)
    error: BaseException | None = None
    result = None
    started = time.perf_counter_ns()
    try:
        result = run_loggopsim(_ideal_config(prepared), binary=binary, timeout_s=timeout_s)
    except Exception as caught:  # noqa: BLE001, retain failed-child evidence
        error = caught
    finally:
        elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
        module.subprocess = original_subprocess
    if len(captured) != 1:
        raise RuntimeError(f"ideal runner launched {len(captured)} child processes")
    argv, completed = captured[0]
    completion_ps = 0 if result is None else result.job_completion_time_ps()
    sample = recorder.capture(
        arm="ideal",
        label=label,
        argv=argv,
        completed=completed,
        elapsed_seconds=elapsed,
        completion_ps=completion_ps,
        quiescent=False if result is None else result.quiescent,
    )
    if error is not None:
        raise error
    assert result is not None
    return sample


def _run_packet_sample(
    prepared: PreparedCell,
    *,
    binary: Path,
    recorder: NativeEvidenceRecorder,
    label: str,
    sample_number: int,
    timeout_s: int,
) -> NativeSample:
    module = importlib.import_module("simllm.backends.htsim_rnic")
    original_runner = module.run_owned_process
    captured: list[tuple[list[str], subprocess.CompletedProcess[Any]]] = []

    def capture(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        completed = original_runner(argv, **kwargs)
        captured.append((list(argv), completed))
        return completed

    completion_csv = (
        recorder.attempt_dir
        / "native"
        / "packet"
        / label
        / f"execution-{sample_number:04d}.completion.csv"
    )
    completion_csv.parent.mkdir(parents=True, exist_ok=True)
    config = HtsimRnicConfig(
        goal_bin=prepared.binary_goal_path,
        profile=PACKET_PROFILE,
        linkspeed_bps=LINKSPEED_BPS,
        completion_csv=completion_csv,
    )
    module.run_owned_process = capture
    error: BaseException | None = None
    result = None
    started = time.perf_counter_ns()
    try:
        result = run_htsim_rnic(config, binary=binary, timeout_s=timeout_s)
    except Exception as caught:  # noqa: BLE001, retain failed-child evidence
        error = caught
    finally:
        elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
        module.run_owned_process = original_runner
    if len(captured) != 1:
        raise RuntimeError(f"packet runner launched {len(captured)} child processes")
    argv, completed = captured[0]
    completion_ps = 0 if result is None else result.job_completion_time_ps()
    sample = recorder.capture(
        arm="packet",
        label=label,
        argv=argv,
        completed=completed,
        elapsed_seconds=elapsed,
        completion_ps=completion_ps,
        quiescent=False if result is None else result.quiescent,
        completion_csv=completion_csv,
    )
    if error is not None:
        raise error
    assert result is not None
    observed_payloads = sorted(flow.payload_bytes for flow in result.flows)
    if observed_payloads != sorted(prepared.cell.payloads):
        raise RuntimeError("packet completion payloads differ from the shared GOAL")
    return sample


def _gate_and_execute(
    goal_text: str,
    *,
    acknowledge_fan_in: bool,
    execute: Callable[[], _T],
) -> tuple[_T, dict[str, object]]:
    stamp = inspect_loggopsim_fan_in(
        goal_text,
        acknowledge_fan_in=acknowledge_fan_in,
    )
    return execute(), stamp.to_json()


def _run_refusal_cell(goal_text: str) -> dict[str, object]:
    execution_calls = 0

    def armed_executor() -> None:
        nonlocal execution_calls
        execution_calls += 1
        raise AssertionError("unacknowledged fan-in reached native execution")

    diagnostic = ""
    refused = False
    try:
        _gate_and_execute(
            goal_text,
            acknowledge_fan_in=False,
            execute=armed_executor,
        )
    except LogGopsimFanInError as error:
        refused = True
        diagnostic = str(error)
    required = (
        "receiver per-byte gap is unmodeled",
        "examples/frontier_ladder_v1/RESULTS.md",
    )
    acknowledged_execution_calls = 0

    def acknowledged_executor() -> None:
        nonlocal acknowledged_execution_calls
        acknowledged_execution_calls += 1

    _, acknowledged_stamp = _gate_and_execute(
        goal_text,
        acknowledge_fan_in=True,
        execute=acknowledged_executor,
    )
    passed = (
        refused
        and execution_calls == 0
        and all(text in diagnostic for text in required)
        and acknowledged_execution_calls == 1
        and acknowledged_stamp["fan_in_detected"] is True
        and acknowledged_stamp["acknowledged"] is True
    )
    return {
        "id": "C-1",
        "expected": "default refusal before execution with mechanism and ladder diagnostic",
        "observed": {
            "diagnostic": diagnostic,
            "execution_calls": execution_calls,
            "refused": refused,
        },
        "mutation_control": {
            "kind": "end-to-end acknowledged-option mutation",
            "mutant_acknowledge_fan_in": True,
            "native_boundary_reached": acknowledged_execution_calls == 1,
            "rejected_by_frozen_refusal_predicate": acknowledged_execution_calls == 1,
        },
        "passed": passed,
    }


def _canonical_execution_record(
    *,
    prepared: PreparedCell,
    sample: NativeSample,
    stamp: dict[str, object],
    acknowledge_fan_in_option: bool,
) -> dict[str, object]:
    return {
        "acknowledge_fan_in_option": acknowledge_fan_in_option,
        "arm": "ideal",
        "argv": list(sample.argv),
        "completion_ps": sample.completion_ps,
        "exact_g_string": EXACT_G_STRING,
        "fan_in": stamp,
        "goal_binary_sha256": prepared.goal_binary_sha256,
        "goal_text_sha256": prepared.goal_text_sha256,
        "quiescent": sample.quiescent,
        "source": sample.source,
    }


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _run_family_c(
    prepared: dict[str, PreparedCell],
    *,
    binary: Path,
    recorder: NativeEvidenceRecorder,
    timeout_s: int,
) -> dict[str, object]:
    serialized = prepared["serialized-b1"]
    incast = prepared["incast-b1"]
    incast_text = incast.goal_path.read_text(encoding="utf-8")
    serialized_text = serialized.goal_path.read_text(encoding="utf-8")
    c1 = _run_refusal_cell(incast_text)

    c2_sample, c2_stamp = _gate_and_execute(
        incast_text,
        acknowledge_fan_in=True,
        execute=lambda: _run_ideal_sample(
            incast,
            binary=binary,
            recorder=recorder,
            label="C-2-acknowledged-incast",
            timeout_s=timeout_s,
        ),
    )
    c2_record = _canonical_execution_record(
        prepared=incast,
        sample=c2_sample,
        stamp=c2_stamp,
        acknowledge_fan_in_option=True,
    )
    c2_passed = (
        c2_sample.source == "native-execution"
        and c2_sample.quiescent
        and c2_stamp["fan_in_detected"] is True
        and c2_stamp["acknowledged"] is True
    )
    c2 = {
        "id": "C-2",
        "expected": "acknowledged incast executes and carries both fan-in stamps",
        "observed": c2_record,
        "native_evidence": c2_sample.evidence_path,
        "passed": c2_passed,
    }

    without_sample, without_stamp = _gate_and_execute(
        serialized_text,
        acknowledge_fan_in=False,
        execute=lambda: _run_ideal_sample(
            serialized,
            binary=binary,
            recorder=recorder,
            label="C-3-clean-without-option",
            timeout_s=timeout_s,
        ),
    )
    with_sample, with_stamp = _gate_and_execute(
        serialized_text,
        acknowledge_fan_in=True,
        execute=lambda: _run_ideal_sample(
            serialized,
            binary=binary,
            recorder=recorder,
            label="C-3-clean-with-option",
            timeout_s=timeout_s,
        ),
    )
    without_record = _canonical_execution_record(
        prepared=serialized,
        sample=without_sample,
        stamp=without_stamp,
        acknowledge_fan_in_option=False,
    )
    with_record = _canonical_execution_record(
        prepared=serialized,
        sample=with_sample,
        stamp=with_stamp,
        acknowledge_fan_in_option=True,
    )
    normalized_without = dict(without_record)
    normalized_with = dict(with_record)
    del normalized_without["acknowledge_fan_in_option"]
    del normalized_with["acknowledge_fan_in_option"]
    differing_fields = sorted(
        key for key in without_record if without_record[key] != with_record[key]
    )
    c3_passed = (
        _json_bytes(normalized_without) == _json_bytes(normalized_with)
        and differing_fields == ["acknowledge_fan_in_option"]
        and without_sample.source == with_sample.source == "native-execution"
    )
    c3 = {
        "id": "C-3",
        "expected": "clean records are byte-identical apart from the recorded option",
        "observed": {
            "differing_fields": differing_fields,
            "normalized_record_sha256": _sha256_bytes(_json_bytes(normalized_without)),
            "with_option_record": with_record,
            "without_option_record": without_record,
        },
        "native_evidence": [without_sample.evidence_path, with_sample.evidence_path],
        "passed": c3_passed,
    }
    rows = [c1, c2, c3]
    return {
        "evidence_class": "enforced-envelope",
        "denominator": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "rows": rows,
    }


def _measurement_row(
    prepared: PreparedCell,
    arm: str,
    samples: list[NativeSample],
    fan_in: dict[str, object] | None,
) -> dict[str, object]:
    elapsed = [sample.elapsed_seconds for sample in samples]
    completions = [sample.completion_ps for sample in samples]
    return {
        "arm": arm,
        "argv": [list(sample.argv) for sample in samples],
        "batch_per_gpu": prepared.cell.batch_per_gpu,
        "cell_id": prepared.cell.cell_id,
        "configuration_id": prepared.cell.configuration_id,
        "exact_g_string": EXACT_G_STRING if arm == "ideal" else None,
        "executions": len(samples),
        "fan_in": fan_in,
        "goal_binary_sha256": prepared.goal_binary_sha256,
        "goal_text_sha256": prepared.goal_text_sha256,
        "median_completion_ps": int(statistics.median(completions)),
        "median_seconds": statistics.median(elapsed),
        "payload_bytes": list(prepared.cell.payloads),
        "samples": [sample.to_json() for sample in samples],
        "source": "native-execution",
    }


def _fraction_record(numerator: float, denominator: float) -> dict[str, object]:
    if isinstance(numerator, int) and isinstance(denominator, int):
        value = Fraction(numerator, denominator)
        return {
            "decimal": float(value),
            "denominator": denominator,
            "numerator": numerator,
            "reduced_denominator": value.denominator,
            "reduced_numerator": value.numerator,
        }
    value = float(numerator) / float(denominator)
    return {"decimal": value, "denominator": denominator, "numerator": numerator}


def _score_family_a(measurements: list[dict[str, object]]) -> dict[str, object]:
    ideal_rows = [row for row in measurements if row["arm"] == "ideal"]
    packet_rows = [row for row in measurements if row["arm"] == "packet"]
    ideal_total = sum(float(row["median_seconds"]) for row in ideal_rows)
    packet_total = sum(float(row["median_seconds"]) for row in packet_rows)
    ratio = packet_total / ideal_total
    score_rows = [
        {
            "id": "A-1",
            "expected": "packet_total / ideal_total >= 50",
            "observed": ratio,
            "passed": ratio >= 50.0,
        },
        {
            "id": "A-2",
            "expected": "ideal_total <= 1 second",
            "observed": ideal_total,
            "passed": ideal_total <= 1.0,
        },
    ]
    by_key = {(row["cell_id"], row["arm"]): row for row in measurements}
    gains = []
    for cell_id in [cell.cell_id for cell in _load_cells()]:
        ideal = by_key[(cell_id, "ideal")]
        packet = by_key[(cell_id, "packet")]
        gains.append(
            {
                "id": f"A-3-{cell_id}",
                "cell_id": cell_id,
                "ideal_median_seconds": ideal["median_seconds"],
                "packet_median_seconds": packet["median_seconds"],
                "packet_over_ideal": float(packet["median_seconds"])
                / float(ideal["median_seconds"]),
                "scored": False,
            }
        )
    return {
        "evidence_class": "wall-time",
        "denominator": len(score_rows),
        "passed": sum(row["passed"] for row in score_rows),
        "rows": score_rows,
        "gain_ratio": ratio,
        "ideal_total_seconds": ideal_total,
        "packet_total_seconds": packet_total,
        "per_shape_gain": gains,
    }


def _score_family_b(measurements: list[dict[str, object]]) -> dict[str, object]:
    packet = {row["cell_id"]: row for row in measurements if row["arm"] == "packet"}
    rows = []
    for cell in _load_cells():
        measurement = packet[cell.cell_id]
        observed = int(measurement["median_completion_ps"])
        quotient = Fraction(observed, cell.reference_completion_ps)
        sample_quotients = [
            _fraction_record(int(sample["completion_ps"]), cell.reference_completion_ps)
            for sample in measurement["samples"]
        ]
        passed = Fraction(98, 100) <= quotient <= Fraction(102, 100)
        rows.append(
            {
                "id": f"B-{cell.cell_id}",
                "cell_id": cell.cell_id,
                "observed_packet_completion_ps": observed,
                "pinned_reference_completion_ps": cell.reference_completion_ps,
                "pinned_reference_kind": cell.reference_kind,
                "quotient": _fraction_record(observed, cell.reference_completion_ps),
                "sample_quotients": sample_quotients,
                "expected_band": [0.98, 1.02],
                "observed_source": "native-execution",
                "oracle_source": "digest-pinned deployment frontier record",
                "passed": passed,
            }
        )
    return {
        "evidence_class": "packet-reference-anchoring",
        "denominator": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "rows": rows,
    }


def _input_rows(*, loggopsim: Path, htsim_rnic: Path, txt2bin: Path) -> list[dict[str, object]]:
    definitions = (
        ("frozen expectations", EXPECTATIONS_PATH, EXPECTATIONS_SHA256, True),
        ("pinned frontier record", PINNED_RECORD_PATH, PINNED_RECORD_SHA256, True),
        ("LogGOPSim", loggopsim, PINNED_LOGGOPSIM_SHA256, True),
        ("htsim_rnic", htsim_rnic, None, False),
        ("txt2bin", txt2bin, None, False),
    )
    rows = []
    for name, path, expected, required_match in definitions:
        observed = _sha256_path(path) if path.is_file() else None
        rows.append(
            {
                "filename": path.name,
                "name": name,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "required_match": required_match,
                "matched": observed == expected if expected is not None else observed is not None,
            }
        )
    return rows


def _flow_set_rows(prepared: dict[str, PreparedCell]) -> list[dict[str, object]]:
    return [
        {
            "cell_id": item.cell.cell_id,
            "goal_binary_sha256_by_arm": {
                "ideal": item.goal_binary_sha256,
                "packet": item.goal_binary_sha256,
            },
            "goal_text_sha256_by_arm": {
                "ideal": item.goal_text_sha256,
                "packet": item.goal_text_sha256,
            },
            "payload_bytes": list(item.cell.payloads),
        }
        for item in prepared.values()
    ]


def _fg1_holds(record: dict[str, object]) -> bool:
    inputs = record["input_hashes"]
    flow_sets = record["flow_sets"]
    return (
        all((not row["required_match"]) or row["matched"] for row in inputs)
        and len(flow_sets) == 12
        and all(
            len(set(row["goal_text_sha256_by_arm"].values())) == 1
            and len(set(row["goal_binary_sha256_by_arm"].values())) == 1
            for row in flow_sets
        )
    )


def _argv_is_portable(argv: list[str]) -> bool:
    return argv[0] in {"LogGOPSim", "htsim_rnic"} and not any(
        Path(item).is_absolute() for item in argv[1:]
    )


def _fg2_holds(record: dict[str, object]) -> bool:
    measurements = record["measurements"]
    if len(measurements) != 24:
        return False
    for row in measurements:
        samples = row.get("samples", [])
        argv = row.get("argv", [])
        elapsed = [sample.get("elapsed_seconds") for sample in samples]
        if (
            row.get("arm") not in {"ideal", "packet"}
            or len(samples) != WALL_SAMPLES
            or len(argv) != WALL_SAMPLES
            or not all(_argv_is_portable(item) for item in argv)
            or not all(sample.get("evidence_path") for sample in samples)
            or statistics.median(elapsed) != row.get("median_seconds")
        ):
            return False
        if row["arm"] == "ideal" and row.get("exact_g_string") != EXACT_G_STRING:
            return False
        if row["arm"] == "packet" and row.get("exact_g_string") is not None:
            return False
    return True


def _fg3_holds(record: dict[str, object]) -> bool:
    measurements = record["measurements"]
    arms = {row["arm"] for row in measurements}
    return arms == {"ideal", "packet"} and all(
        row.get("source") == "native-execution"
        and row.get("executions") == WALL_SAMPLES
        and all(sample.get("source") == "native-execution" for sample in row["samples"])
        for row in measurements
    )


def _fg4_holds(record: dict[str, object]) -> bool:
    chronology = record["chronology"]
    expected = chronology.get("expectations_commit")
    implementation = chronology.get("implementation_commit")
    return (
        expected == EXPECTATIONS_COMMIT
        and isinstance(implementation, str)
        and implementation != expected
        and _is_ancestor(expected, implementation)
    )


def _fatal_guards(base_record: dict[str, object]) -> list[dict[str, object]]:
    definitions: tuple[
        tuple[str, str, Callable[[dict[str, object]], bool], Callable[[dict[str, object]], None]],
        ...,
    ] = (
        (
            "FG-1",
            "pinned inputs and byte-identical per-arm GOALs",
            _fg1_holds,
            lambda mutant: mutant["flow_sets"][0]["goal_text_sha256_by_arm"].__setitem__(
                "ideal", "0" * 64
            ),
        ),
        (
            "FG-2",
            "portable median-of-seven provenance on every measurement row",
            _fg2_holds,
            lambda mutant: mutant["measurements"][0]["samples"].pop(),
        ),
        (
            "FG-3",
            "both scored arms are native executions rather than closed forms",
            _fg3_holds,
            lambda mutant: mutant["measurements"][0].__setitem__("source", "closed-form"),
        ),
        (
            "FG-4",
            "expectations-only commit precedes the implementation and run",
            _fg4_holds,
            lambda mutant: mutant["chronology"].__setitem__(
                "implementation_commit", EXPECTATIONS_COMMIT
            ),
        ),
    )
    rows = []
    for identifier, predicate, check, mutate in definitions:
        held = check(base_record)
        mutant = deepcopy(base_record)
        mutate(mutant)
        rejected = not check(mutant)
        rows.append(
            {
                "id": identifier,
                "predicate": predicate,
                "held": held,
                "mutation_control": {
                    "kind": "end-to-end predicate mutation",
                    "same_predicate_exercised": True,
                    "rejected": rejected,
                },
            }
        )
    return rows


def _machine_disclosure() -> dict[str, object]:
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    memory_bytes = None
    try:
        memory_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        pass
    return {
        "architecture": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "memory_bytes": memory_bytes,
        "operating_system": platform.system(),
        "os_release": platform.release(),
        "python": platform.python_version(),
    }


def _physical_sanity(
    measurements: list[dict[str, object]], family_b: dict[str, object]
) -> dict[str, object]:
    packet = {row["cell_id"]: row for row in measurements if row["arm"] == "packet"}
    ideal = {row["cell_id"]: row for row in measurements if row["arm"] == "ideal"}
    rows = []
    for cell in _load_cells():
        payload_floor_ps = sum(cell.payloads) * 20
        wire_bytes = sum(math.ceil(payload / 4096) * 4160 for payload in cell.payloads)
        packet_ceiling_ps = wire_bytes * 20 + 2_000_000 + len(cell.payloads) * 83_200 + 1_000
        packet_observed = int(packet[cell.cell_id]["median_completion_ps"])
        ideal_exact_ps = max(cell.payloads) * 20 + 2_000_000
        ideal_observed = int(ideal[cell.cell_id]["median_completion_ps"])
        rows.append(
            {
                "cell_id": cell.cell_id,
                "packet_floor_ps": payload_floor_ps,
                "packet_ceiling_ps": packet_ceiling_ps,
                "packet_observed_ps": packet_observed,
                "packet_inside_bounds": payload_floor_ps <= packet_observed <= packet_ceiling_ps,
                "ideal_quantized_lower_ps": ideal_exact_ps - 999,
                "ideal_quantized_upper_ps": ideal_exact_ps,
                "ideal_observed_ps": ideal_observed,
                "ideal_inside_quantization_bounds": ideal_exact_ps - 999
                <= ideal_observed
                <= ideal_exact_ps,
            }
        )
    b_rows = {row["cell_id"]: row for row in family_b["rows"]}
    packet_incast_scaling = _fraction_record(
        b_rows["incast-b32"]["observed_packet_completion_ps"],
        b_rows["incast-b16"]["observed_packet_completion_ps"],
    )
    ideal_scaling = _fraction_record(
        int(ideal["incast-b32"]["median_completion_ps"]),
        int(ideal["incast-b16"]["median_completion_ps"]),
    )
    serialized_ratio = Fraction(
        int(packet["serialized-b32"]["median_completion_ps"]),
        int(ideal["serialized-b32"]["median_completion_ps"]),
    )
    incast_ratio = Fraction(
        int(packet["incast-b32"]["median_completion_ps"]),
        int(ideal["incast-b32"]["median_completion_ps"]),
    )
    return {
        "bounds_declared_before_measurement": (
            "implementation commit fixes bytes/rate floors, 4160/4096 packet-wire "
            "ceilings, 2 us propagation, one 83.2 ns slot per flow, and 1 ns "
            "reported-time quantization"
        ),
        "serialization_physics": {
            "all_inside_bounds": all(row["packet_inside_bounds"] for row in rows),
            "rows": rows,
        },
        "batch_scaling": {
            "expected": "batch 32 over batch 16 remains within [1.9, 2.1] on both arms",
            "ideal_incast_b32_over_b16": ideal_scaling,
            "packet_incast_b32_over_b16": packet_incast_scaling,
            "passed": 1.9 <= ideal_scaling["decimal"] <= 2.1
            and 1.9 <= packet_incast_scaling["decimal"] <= 2.1,
        },
        "mechanism_separation": {
            "expected": "serialized packet/ideal is near one while shared-ingress incast is near eight",
            "serialized_packet_over_ideal": _fraction_record(
                serialized_ratio.numerator, serialized_ratio.denominator
            ),
            "incast_packet_over_ideal": _fraction_record(
                incast_ratio.numerator, incast_ratio.denominator
            ),
            "passed": Fraction(1, 1) <= serialized_ratio <= Fraction(11, 10)
            and Fraction(15, 2) <= incast_ratio <= Fraction(17, 2),
        },
    }


def _csv_rows(result: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for row in result["families"]["A"]["rows"]:
        rows.append(
            {
                "evidence_class": "wall-time",
                "family": "A",
                "id": row["id"],
                "expected": row["expected"],
                "observed": row["observed"],
                "units": "ratio" if row["id"] == "A-1" else "seconds",
                "passed": str(row["passed"]).lower(),
                "scored": "true",
            }
        )
    for row in result["families"]["A"]["per_shape_gain"]:
        rows.append(
            {
                "evidence_class": "wall-time",
                "family": "A",
                "id": row["id"],
                "expected": "reported",
                "observed": row["packet_over_ideal"],
                "units": "ratio",
                "passed": "",
                "scored": "false",
            }
        )
    for row in result["families"]["B"]["rows"]:
        rows.append(
            {
                "evidence_class": "packet-reference-anchoring",
                "family": "B",
                "id": row["id"],
                "expected": "[0.98, 1.02]",
                "observed": row["quotient"]["decimal"],
                "units": "quotient",
                "passed": str(row["passed"]).lower(),
                "scored": "true",
            }
        )
    for row in result["families"]["C"]["rows"]:
        rows.append(
            {
                "evidence_class": "enforced-envelope",
                "family": "C",
                "id": row["id"],
                "expected": row["expected"],
                "observed": "predicate held" if row["passed"] else "predicate missed",
                "units": "predicate",
                "passed": str(row["passed"]).lower(),
                "scored": "true",
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = (
        "evidence_class",
        "family",
        "id",
        "expected",
        "observed",
        "units",
        "passed",
        "scored",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_study(args: argparse.Namespace) -> dict[str, object]:
    """Execute both arms seven times per shape and score the frozen surface."""

    implementation_commit = _git_hash("HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status.returncode or status.stdout.strip():
        raise SystemExit("the frozen acceptance run requires a clean tracked worktree")
    for name in ("loggopsim", "htsim_rnic", "txt2bin"):
        path = Path(getattr(args, name)).resolve()
        if not path.is_file():
            raise SystemExit(f"{name} is not a file: {path}")

    input_hashes = _input_rows(
        loggopsim=args.loggopsim.resolve(),
        htsim_rnic=args.htsim_rnic.resolve(),
        txt2bin=args.txt2bin.resolve(),
    )
    cells = _load_cells()
    prepared = _prepare_goals(cells, args.run_dir.resolve(), args.txt2bin.resolve())
    recorder = NativeEvidenceRecorder(args.run_dir.resolve())
    ideal_samples: dict[str, list[NativeSample]] = defaultdict(list)
    packet_samples: dict[str, list[NativeSample]] = defaultdict(list)
    ideal_stamps: dict[str, dict[str, object]] = {}

    for cell_index, cell in enumerate(cells, start=1):
        item = prepared[cell.cell_id]
        goal_text = item.goal_path.read_text(encoding="utf-8")
        for sample_number in range(1, WALL_SAMPLES + 1):
            sample, stamp = _gate_and_execute(
                goal_text,
                acknowledge_fan_in=cell.acknowledge_fan_in,
                execute=lambda item=item: _run_ideal_sample(
                    item,
                    binary=args.loggopsim.resolve(),
                    recorder=recorder,
                    label=item.cell.cell_id,
                    timeout_s=args.timeout_s,
                ),
            )
            ideal_samples[cell.cell_id].append(sample)
            ideal_stamps[cell.cell_id] = stamp
        for sample_number in range(1, WALL_SAMPLES + 1):
            packet_samples[cell.cell_id].append(
                _run_packet_sample(
                    item,
                    binary=args.htsim_rnic.resolve(),
                    recorder=recorder,
                    label=item.cell.cell_id,
                    sample_number=sample_number,
                    timeout_s=args.timeout_s,
                )
            )
        print(
            f"[{cell_index}/12] {cell.cell_id}: "
            f"ideal={statistics.median(s.elapsed_seconds for s in ideal_samples[cell.cell_id]):.6f}s "
            f"packet={statistics.median(s.elapsed_seconds for s in packet_samples[cell.cell_id]):.6f}s",
            flush=True,
        )

    measurements = []
    for cell in cells:
        item = prepared[cell.cell_id]
        measurements.append(
            _measurement_row(
                item,
                "ideal",
                ideal_samples[cell.cell_id],
                ideal_stamps[cell.cell_id],
            )
        )
        measurements.append(_measurement_row(item, "packet", packet_samples[cell.cell_id], None))

    family_a = _score_family_a(measurements)
    family_b = _score_family_b(measurements)
    family_c = _run_family_c(
        prepared,
        binary=args.loggopsim.resolve(),
        recorder=recorder,
        timeout_s=args.timeout_s,
    )
    base_record: dict[str, object] = {
        "chronology": {
            "expectations_commit": EXPECTATIONS_COMMIT,
            "expectations_sha256": EXPECTATIONS_SHA256,
            "implementation_commit": implementation_commit,
        },
        "flow_sets": _flow_set_rows(prepared),
        "input_hashes": input_hashes,
        "measurements": measurements,
    }
    fatal_guards = _fatal_guards(base_record)
    guard_void = not all(
        guard["held"] and guard["mutation_control"]["rejected"] for guard in fatal_guards
    )
    score_pass = all(
        family["passed"] == family["denominator"] for family in (family_a, family_b, family_c)
    )
    verdict = "VOID" if guard_void else ("PASS" if score_pass else "REFUTED")
    result: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "verdict": verdict,
        "findings": [guard["id"] for guard in fatal_guards if not guard["held"]]
        + [
            family_id
            for family_id, family in (("A", family_a), ("B", family_b), ("C", family_c))
            if family["passed"] != family["denominator"]
        ],
        "scope": (
            "Executed LogGOPSim ideal and rnic-nn packet pricing of the twelve "
            "digest-pinned frontier flow sets. This qualifies speed and the "
            "declared fan-in envelope, not packet or silicon fidelity beyond "
            "the pinned frontier record."
        ),
        **base_record,
        "native_tools": {
            "htsim_rnic": {
                "filename": args.htsim_rnic.name,
                "sha256": _sha256_path(args.htsim_rnic),
            },
            "loggopsim": {
                "filename": args.loggopsim.name,
                "sha256": _sha256_path(args.loggopsim),
            },
            "txt2bin": {
                "filename": args.txt2bin.name,
                "sha256": _sha256_path(args.txt2bin),
            },
        },
        "declared_parameters": {
            "ideal": {
                "G_ns_per_byte": EXACT_G_STRING,
                "L_ns": LATENCY_NS,
                "O_ns_per_byte": 0,
                "g_ns": 0,
                "o_ns": 0,
                "protocol": "eager for every cell",
            },
            "packet": {
                "linkspeed_bps": LINKSPEED_BPS,
                "profile": PACKET_PROFILE,
            },
        },
        "families": {"A": family_a, "B": family_b, "C": family_c},
        "fatal_guards": fatal_guards,
        "physical_sanity": _physical_sanity(measurements, family_b),
        "machine": _machine_disclosure(),
        "attempt_evidence": {
            "bulk_attempt_id": args.run_dir.name,
            "policy": (
                "each run uses a fresh attempt-N directory and refuses a later "
                "attempt until every earlier attempt has a verdict"
            ),
            "ideal_native_executions": 12 * WALL_SAMPLES + 3,
            "packet_native_executions": 12 * WALL_SAMPLES,
            "stdout_bytes": "native/*/*/execution-*.stdout",
            "stderr_bytes": "native/*/*/execution-*.stderr",
            "portable_argv": "native/*/*/execution-*.json",
            "verdict": "verdict.json",
        },
        "evidence_separation": (
            "A wall time, B packet anchoring, and C enforcement keep separate "
            "denominators; fatal guards are unscored and void the full run"
        ),
    }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--loggopsim",
        type=Path,
        default=os.environ.get("SIMLLM_LOGGOPSIM"),
        required="SIMLLM_LOGGOPSIM" not in os.environ,
    )
    parser.add_argument(
        "--txt2bin",
        type=Path,
        default=os.environ.get("SIMLLM_TXT2BIN"),
        required="SIMLLM_TXT2BIN" not in os.environ,
    )
    parser.add_argument(
        "--htsim-rnic",
        type=Path,
        default=os.environ.get("SIMLLM_HTSIM_RNIC"),
        required="SIMLLM_HTSIM_RNIC" not in os.environ,
    )
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--results-json", type=Path)
    parser.add_argument("--results-csv", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    attempt_dir = _begin_attempt(args.run_dir, args)
    attempt_args = argparse.Namespace(**vars(args))
    attempt_args.run_dir = attempt_dir
    verdict_path = attempt_dir / "verdict.json"
    try:
        result = run_study(attempt_args)
    except BaseException as error:
        _write_json(
            verdict_path,
            {
                "schema": ATTEMPT_SCHEMA,
                "verdict": "ERROR",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    _write_json(verdict_path, result)
    if args.results_json is not None:
        _write_json(args.results_json, result)
    csv_rows = _csv_rows(result)
    _write_csv(attempt_dir / "results.csv", csv_rows)
    if args.results_csv is not None:
        _write_csv(args.results_csv, csv_rows)
    print(f"verdict={result['verdict']} results={verdict_path}")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
