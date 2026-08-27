#!/usr/bin/env python3
"""Execute the frozen frontier ladder study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.frontier_ladder_v1.plot_study import prepare_plot_data
from simllm.backends import (
    LogGopsimConfig,
    build_loggopsim_command,
    parse_loggopsim_stdout,
)
from simllm.deploy import (
    ExternalAnchor,
    FrontierLadderPoint,
    FrontierLadderRecord,
    FrontierRung,
    FrontierRungPoint,
    PointClass,
    RungAuthorityClass,
    RungProvenance,
    frontier_ladder_record_from_json,
    frontier_ladder_record_to_json,
    ladder_pareto_front,
)
from simllm.goal import GoalTrace, to_binary
from simllm.traffic import ordered_pairwise_messages

STUDY_DIR = Path(__file__).resolve().parent
EXPECTATIONS_PATH = STUDY_DIR / "expectations.md"
PINNED_RECORD_PATH = (
    REPOSITORY_ROOT / "examples" / "deployment_frontier_v1" / "result.json"
)
LOGGOPSIM_EXPECTATIONS_PATH = (
    REPOSITORY_ROOT / "examples" / "loggopsim_ideal_v1" / "expectations.md"
)

RESULT_SCHEMA = "simllm-frontier-ladder-study-v1"
EXPECTATIONS_COMMIT = "228f3c77b98af1f0f60985405a8db67ebb67c0a6"
EXPECTATIONS_SHA256 = "e3e83264df6e72e83736a06dddcba11a501c75a25c8c1fb0a9c7b1e9c0caeea3"
PINNED_RECORD_SHA256 = "f2f216068bf5ba914853c62a2ee965ede0ebfc0a6f29e3d11cfa5f45eac359ad"
LOGGOPSIM_EXPECTATIONS_SHA256 = (
    "934ee355e4d5a376d1eecdb1d0e62f6e4f7acfd9ada93def5ba1bcf8fa8508ff"
)
PINNED_LOGGOPSIM_SHA256 = (
    "7e0f13ee3c87a20e9d2e94dbbd74c46075fd03df2f1b04d1ed9739c43ee0a2bf"
)
PICOSECONDS_PER_SECOND = 1_000_000_000_000
BATCHES = (1, 2, 4, 8, 16, 32)
EXACT_G_STRING = "0.02"
WALL_SAMPLES = 7
WALL_CEILING_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class FabricCell:
    """One frozen native fabric-leg shape and literal oracle."""

    family: str
    configuration_id: str
    batch_per_gpu: int
    rank_count: int
    flow_count: int
    payload_bytes: int
    expected_ns: int

    @property
    def cell_id(self) -> str:
        return f"{self.family}-b{self.batch_per_gpu}"

    @property
    def stem(self) -> str:
        return f"{self.configuration_id}-b{self.batch_per_gpu}"


@dataclass(frozen=True, slots=True)
class NativeExecution:
    """One captured native invocation."""

    elapsed_seconds: float
    max_finish_ps: int
    stdout_sha256: str
    stderr_sha256: str
    evidence_path: str


def _cells() -> tuple[FabricCell, ...]:
    serialized = (
        (1, 6_651_904, 135_038),
        (2, 13_303_808, 268_076),
        (4, 26_607_616, 534_152),
        (8, 53_215_232, 1_066_304),
        (16, 106_430_464, 2_130_609),
        (32, 212_860_928, 4_259_218),
    )
    incast = (
        (1, 1_478_201, 31_564),
        (2, 2_956_402, 61_128),
        (4, 5_912_804, 120_256),
        (8, 11_825_608, 238_512),
        (16, 23_651_215, 475_024),
        (32, 47_302_429, 948_048),
    )
    return tuple(
        [
            FabricCell(
                "L-A",
                "h100-two-node-serialized",
                batch,
                2,
                1,
                payload,
                expected,
            )
            for batch, payload, expected in serialized
        ]
        + [
            FabricCell(
                "L-B",
                "h100-nine-node-incast",
                batch,
                9,
                8,
                payload,
                expected,
            )
            for batch, payload, expected in incast
        ]
    )


class NativeEvidenceRecorder:
    """Retain stdout and stderr bytes for every append-only attempt."""

    def __init__(self, attempt_dir: Path) -> None:
        self._attempt_dir = attempt_dir
        self._root = attempt_dir / "native"
        self._counts: dict[str, int] = defaultdict(int)

    def capture(
        self,
        label: str,
        argv: list[str],
        completed: subprocess.CompletedProcess[bytes],
        elapsed_seconds: float,
        max_finish_ps: int | None,
    ) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
        self._counts[safe] += 1
        directory = self._root / safe
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"execution-{self._counts[safe]:04d}"
        stdout_path = directory / f"{stem}.stdout"
        stderr_path = directory / f"{stem}.stderr"
        manifest_path = directory / f"{stem}.json"
        stdout_path.write_bytes(completed.stdout)
        stderr_path.write_bytes(completed.stderr)
        _write_json(
            manifest_path,
            {
                "argv": _portable_argv(argv, self._attempt_dir),
                "elapsed_seconds": elapsed_seconds,
                "max_finish_ps": max_finish_ps,
                "returncode": completed.returncode,
                "stderr_file": stderr_path.name,
                "stderr_sha256": _sha256_bytes(completed.stderr),
                "stdout_file": stdout_path.name,
                "stdout_sha256": _sha256_bytes(completed.stdout),
            },
        )
        return manifest_path.relative_to(self._attempt_dir).as_posix()


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


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "evidence_class",
        "family",
        "id",
        "expected",
        "observed",
        "units",
        "passed",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
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
        raise SystemExit(
            f"cannot start a later attempt while verdict records are missing: {names}"
        )
    number = max((number for number, _ in attempts), default=0) + 1
    attempt_dir = root / f"attempt-{number}"
    attempt_dir.mkdir(parents=False, exist_ok=False)
    _write_json(
        attempt_dir / "attempt.json",
        {
            "binary": str(args.binary.resolve()),
            "run_root": str(root),
            "started_unix_time_ns": time.time_ns(),
            "txt2bin": str(args.txt2bin.resolve()),
        },
    )
    return attempt_dir


def _portable_argv(argv: list[str], attempt_dir: Path) -> list[str]:
    portable = list(argv)
    portable[0] = "LogGOPSim"
    goal_index = portable.index("-f") + 1
    goal_path = Path(portable[goal_index]).resolve()
    try:
        portable[goal_index] = goal_path.relative_to(attempt_dir.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError("native GOAL path escapes the attempt directory") from exc
    return portable


def _render_goal(cell: FabricCell, goal_dir: Path, txt2bin: Path) -> tuple[Path, Path]:
    trace = GoalTrace(cell.rank_count)
    messages = tuple(
        (f"flow-{index}", index + 1, 0, cell.payload_bytes)
        for index in range(cell.flow_count)
    )
    ordered_pairwise_messages(
        trace,
        list(range(cell.rank_count)),
        messages,
        tag=20_068,
        operation_id=f"frontier-ladder:{cell.stem}",
    )
    if len(trace.messages) != cell.flow_count:
        raise AssertionError("rendered GOAL changed the frozen flow count")
    if any(message.payload_bytes != cell.payload_bytes for message in trace.messages):
        raise AssertionError("rendered GOAL changed the frozen payload")
    text_path = trace.write(goal_dir / f"{cell.stem}.goal")
    binary_path = to_binary(
        text_path,
        goal_dir / f"{cell.stem}.bin",
        tool=txt2bin,
    )
    return text_path, binary_path


def _config(cell: FabricCell, binary_goal: Path) -> LogGopsimConfig:
    return LogGopsimConfig(
        goal_bin=binary_goal,
        latency_ns=2_000,
        overhead_ns=0,
        message_gap_ns=0,
        byte_gap_ns=0.02,
        byte_gap_ns_string=EXACT_G_STRING,
        byte_overhead_ns=0,
        rendezvous_threshold_bytes=cell.payload_bytes + 1,
        network_type="LogGP",
    )


def _execute(
    argv: list[str],
    recorder: NativeEvidenceRecorder,
    label: str,
) -> NativeExecution:
    started = time.perf_counter_ns()
    completed = subprocess.run(argv, capture_output=True, check=False)
    elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
    parsed = None
    if completed.returncode == 0:
        parsed = parse_loggopsim_stdout(completed.stdout.decode("utf-8"))
    evidence_path = recorder.capture(
        label,
        argv,
        completed,
        elapsed,
        None if parsed is None else parsed.max_finish_ps,
    )
    if completed.returncode:
        raise RuntimeError(
            f"LogGOPSim exited {completed.returncode}: "
            f"{completed.stderr.decode(errors='replace').strip()}"
        )
    assert parsed is not None
    if not parsed.quiescent:
        raise RuntimeError("LogGOPSim reported unmatched messages")
    return NativeExecution(
        elapsed_seconds=elapsed,
        max_finish_ps=parsed.max_finish_ps,
        stdout_sha256=_sha256_bytes(completed.stdout),
        stderr_sha256=_sha256_bytes(completed.stderr),
        evidence_path=evidence_path,
    )


def _input_rows(binary: Path) -> list[dict[str, object]]:
    inputs = (
        ("frontier expectations", EXPECTATIONS_PATH, EXPECTATIONS_SHA256),
        ("pinned frontier record", PINNED_RECORD_PATH, PINNED_RECORD_SHA256),
        (
            "LogGOPSim ideal expectations",
            LOGGOPSIM_EXPECTATIONS_PATH,
            LOGGOPSIM_EXPECTATIONS_SHA256,
        ),
        ("LogGOPSim binary", binary, PINNED_LOGGOPSIM_SHA256),
    )
    rows = []
    for name, path, expected in inputs:
        observed = _sha256_path(path) if path.is_file() else None
        rows.append(
            {
                "name": name,
                "path": (
                    path.relative_to(REPOSITORY_ROOT).as_posix()
                    if path.is_relative_to(REPOSITORY_ROOT)
                    else path.name
                ),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "matched": observed == expected,
            }
        )
    return rows


def _anchors(legacy: dict[str, Any]) -> tuple[ExternalAnchor, ...]:
    paired = legacy["published_context"]["paired"][0]
    y_only = legacy["published_context"]["y_only"][0]
    return (
        ExternalAnchor(
            anchor_id=paired["id"],
            label=paired["label"],
            x_tokens_per_second_per_request=Fraction(
                paired["tokens_per_second_per_node"],
                paired["batch_per_node"],
            ),
            y_tokens_per_second_per_gpu=Fraction(
                paired["tokens_per_second_per_node"],
                paired["gpus_per_node"],
            ),
        ),
        ExternalAnchor(
            anchor_id=y_only["id"],
            label=y_only["label"],
            y_tokens_per_second_per_gpu=Fraction(
                y_only["tokens_per_second_per_node"],
                y_only["gpus_per_node"],
            ),
        ),
    )


def _point(
    *,
    row: dict[str, Any],
    ideal: dict[str, Any] | None,
    binary_sha256: str,
) -> FrontierLadderPoint:
    batch = row["batch_per_gpu"]
    analytical_step = row["accounting"]["analytical_step_ps"]
    packet_step = row["accounting"]["simulated_step_ps"]
    estimate_fabric = row["ideal_network"]["ideal_fabric_wire_ps"]
    packet_fabric = row["fabric_observation"]["concurrent_service_ps"]
    if ideal is None:
        if packet_fabric or estimate_fabric:
            raise AssertionError("a nonzero fabric point is missing native ideal evidence")
        ideal_fabric = 0
        ideal_step = analytical_step
        ideal_source_path = PINNED_RECORD_PATH.relative_to(REPOSITORY_ROOT).as_posix()
        ideal_source_sha256 = PINNED_RECORD_SHA256
        ideal_goal_sha256 = None
        ideal_argv: tuple[str, ...] = ()
    else:
        ideal_fabric = ideal["observed_ns"] * 1_000
        ideal_step = max(
            row["kernel"]["kernel_floor_ps"],
            ideal_fabric,
            row["ideal_network"]["ideal_intra_node_wire_ps"],
        )
        ideal_source_path = ideal["goal_path"]
        ideal_source_sha256 = ideal["goal_text_sha256"]
        ideal_goal_sha256 = ideal["goal_binary_sha256"]
        ideal_argv = tuple(ideal["argv"])

    def rung(
        identity: FrontierRung,
        point_class: PointClass,
        step_ps: int,
        fabric_leg_ps: int,
        provenance: RungProvenance,
    ) -> FrontierRungPoint:
        x_value = Fraction(PICOSECONDS_PER_SECOND, step_ps)
        return FrontierRungPoint(
            rung=identity,
            point_class=point_class,
            step_ps=step_ps,
            fabric_leg_ps=fabric_leg_ps,
            x_tokens_per_second_per_request=x_value,
            y_tokens_per_second_per_gpu=batch * x_value,
            provenance=provenance,
        )

    pinned_path = PINNED_RECORD_PATH.relative_to(REPOSITORY_ROOT).as_posix()
    return FrontierLadderPoint(
        configuration_id=row["configuration_id"],
        configuration_label=row["configuration_label"],
        batch_per_gpu=batch,
        rungs=(
            rung(
                FrontierRung.ESTIMATE,
                PointClass.ESTIMATE,
                analytical_step,
                estimate_fabric,
                RungProvenance(
                    authority_class=RungAuthorityClass.ESTIMATOR,
                    authority="closed-form",
                    source_path=pinned_path,
                    source_sha256=PINNED_RECORD_SHA256,
                ),
            ),
            rung(
                FrontierRung.LOGGOPSIM_IDEAL,
                PointClass.SIMULATED,
                ideal_step,
                ideal_fabric,
                RungProvenance(
                    authority_class=RungAuthorityClass.LEVEL,
                    authority="loggopsim-ideal",
                    source_path=ideal_source_path,
                    source_sha256=ideal_source_sha256,
                    binary_sha256=binary_sha256,
                    goal_sha256=ideal_goal_sha256,
                    argv=ideal_argv,
                ),
            ),
            rung(
                FrontierRung.PACKET,
                PointClass.SIMULATED,
                packet_step,
                packet_fabric,
                RungProvenance(
                    authority_class=RungAuthorityClass.LEVEL,
                    authority="rnic-nn",
                    source_path=pinned_path,
                    source_sha256=PINNED_RECORD_SHA256,
                    binary_sha256=row.get("_packet_binary_sha256"),
                ),
            ),
        ),
    )


def _build_record(
    legacy: dict[str, Any],
    ideal_rows: dict[tuple[str, int], dict[str, Any]],
    binary_sha256: str,
) -> FrontierLadderRecord:
    packet_binary_sha256 = legacy["provenance"]["htsim_rnic"]["sha256"]
    points = []
    for raw in legacy["points"]:
        row = dict(raw)
        row["_packet_binary_sha256"] = packet_binary_sha256
        key = (row["configuration_id"], row["batch_per_gpu"])
        points.append(
            _point(
                row=row,
                ideal=ideal_rows.get(key),
                binary_sha256=binary_sha256,
            )
        )
    return FrontierLadderRecord(points=tuple(points), anchors=_anchors(legacy))


def _fraction_record(numerator: int, denominator: int) -> dict[str, object]:
    reduced = Fraction(numerator, denominator)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "reduced_numerator": reduced.numerator,
        "reduced_denominator": reduced.denominator,
        "decimal": float(reduced),
    }


def _in_band(numerator: int, denominator: int, low: Fraction, high: Fraction) -> bool:
    value = Fraction(numerator, denominator)
    return low <= value <= high


def _mechanism_rows(
    legacy_rows: dict[tuple[str, int], dict[str, Any]],
    ideal_rows: dict[tuple[str, int], dict[str, Any]],
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    definitions = (
        (
            "M-1",
            "h100-two-node-serialized",
            "concurrent_service_ps",
            Fraction(1),
            Fraction(51, 50),
            (4_325_821_000, 4_259_218_000),
        ),
        (
            "M-2",
            "h100-nine-node-incast",
            "concurrent_service_ps",
            Fraction(15, 2),
            Fraction(17, 2),
            (7_689_053_000, 948_048_000),
        ),
        (
            "M-3",
            "h100-nine-node-incast",
            "isolated_service_ps",
            Fraction(1),
            Fraction(207, 200),
            (962_915_000, 948_048_000),
        ),
    )
    families: dict[str, dict[str, object]] = {}
    envelope = []
    for family, configuration_id, packet_field, low, high, batch32_literal in definitions:
        rows = []
        for batch in BATCHES:
            key = (configuration_id, batch)
            packet_ps = legacy_rows[key]["fabric_observation"][packet_field]
            ideal_ps = ideal_rows[key]["observed_ns"] * 1_000
            literal_pass = batch != 32 or (packet_ps, ideal_ps) == batch32_literal
            passed = _in_band(packet_ps, ideal_ps, low, high) and literal_pass
            row = {
                "id": f"{family}-b{batch}",
                "batch_per_gpu": batch,
                "configuration_id": configuration_id,
                "packet_quantity": packet_field,
                "packet_ps": packet_ps,
                "ideal_ps": ideal_ps,
                "quotient": _fraction_record(packet_ps, ideal_ps),
                "expected_band": {
                    "lower": _fraction_record(low.numerator, low.denominator),
                    "upper": _fraction_record(high.numerator, high.denominator),
                },
                "batch32_literal": (
                    None
                    if batch != 32
                    else {
                        "numerator": batch32_literal[0],
                        "denominator": batch32_literal[1],
                    }
                ),
                "passed": passed,
            }
            rows.append(row)
            envelope.append({"family": family, **row})
        families[family] = {
            "evidence_class": "behavioral-relation",
            "denominator": len(rows),
            "passed": sum(bool(row["passed"]) for row in rows),
            "rows": rows,
        }
    return families, envelope


def _step_family(record: FrontierLadderRecord) -> dict[str, object]:
    rows = []
    exceptional = ("b100-one-node-intra", 32)
    for point in record.points:
        values = {rung.rung.value: rung.step_ps for rung in point.rungs}
        key = (point.configuration_id, point.batch_per_gpu)
        if key == exceptional:
            expected = {
                FrontierRung.ESTIMATE.value: 4_257_218_560,
                FrontierRung.LOGGOPSIM_IDEAL.value: 4_257_218_560,
                FrontierRung.PACKET.value: 4_523_298_348,
            }
            passed = values == expected
            relation = "packet intra-node excess is visible; estimate and ideal agree"
        else:
            expected = "all three step times equal"
            passed = len(set(values.values())) == 1
            relation = expected
        rows.append(
            {
                "id": f"S-{point.configuration_id}-b{point.batch_per_gpu}",
                "configuration_id": point.configuration_id,
                "batch_per_gpu": point.batch_per_gpu,
                "expected": expected,
                "observed_step_ps": values,
                "relation": relation,
                "passed": passed,
            }
        )
    return {
        "evidence_class": "behavioral-relation",
        "denominator": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "rows": rows,
    }


def _plot_family(record: FrontierLadderRecord, envelope: list[dict[str, object]]) -> dict[str, object]:
    rendered = frontier_ladder_record_to_json(record)
    parsed = frontier_ladder_record_from_json(rendered)
    strict_round_trip = frontier_ladder_record_to_json(parsed) == rendered
    class_rows = [
        (rung.rung.value, rung.point_class.value, rung.provenance.authority)
        for point in record.points
        for rung in point.rungs
    ]
    expected_classes = {
        FrontierRung.ESTIMATE.value: (PointClass.ESTIMATE.value, "closed-form"),
        FrontierRung.LOGGOPSIM_IDEAL.value: (
            PointClass.SIMULATED.value,
            "loggopsim-ideal",
        ),
        FrontierRung.PACKET.value: (PointClass.SIMULATED.value, "rnic-nn"),
    }
    classes_pass = all(
        (point_class, authority) == expected_classes[rung]
        for rung, point_class, authority in class_rows
    )
    packet_front = [
        (point.configuration_id, point.batch_per_gpu)
        for point in ladder_pareto_front(record, FrontierRung.PACKET)
    ]
    expected_front = [("b100-one-node-intra", batch) for batch in BATCHES]
    plot = prepare_plot_data(
        {
            "schema": RESULT_SCHEMA,
            "ladder_record": rendered,
            "fabric_leg_envelope": envelope,
        }
    )
    projection_round_trip = prepare_plot_data(
        {
            "schema": RESULT_SCHEMA,
            "ladder_record": frontier_ladder_record_to_json(parsed),
            "fabric_leg_envelope": json.loads(json.dumps(envelope)),
        }
    ) == plot
    rows = [
        {
            "id": "P-record-round-trip",
            "predicate": "strict ladder record round trip",
            "passed": strict_round_trip,
        },
        {
            "id": "P-rung-classes",
            "predicate": "per-rung point classes and authorities remain distinct",
            "passed": classes_pass,
        },
        {
            "id": "P-packet-pareto",
            "predicate": "packet Pareto front is exactly the six B100 points",
            "expected": expected_front,
            "observed": packet_front,
            "passed": packet_front == expected_front,
        },
        {
            "id": "P-plot-projection",
            "predicate": "plot projection round trips through the strict record",
            "passed": projection_round_trip,
        },
    ]
    return {
        "evidence_class": "plot-contract",
        "denominator": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "rows": rows,
        "plot_data": plot,
    }


def _machine_disclosure() -> dict[str, object]:
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
        "cpu_model": cpu_model,
    }


def _mutation_control(mutant: str, rejected: bool) -> dict[str, object]:
    return {
        "kind": "predicate-exercised",
        "mutant": mutant,
        "rejected": rejected,
    }


def _fg1_holds(input_rows: list[dict[str, object]]) -> bool:
    return all(bool(row["matched"]) for row in input_rows)


def _record_holds(value: object) -> bool:
    try:
        record = frontier_ladder_record_from_json(value)
    except (TypeError, ValueError):
        return False
    return len(record.points) == 18 and all(len(point.rungs) == 3 for point in record.points)


def _execution_holds(value: object) -> bool:
    try:
        record = frontier_ladder_record_from_json(value)
    except (TypeError, ValueError):
        return False
    executed = []
    for point in record.points:
        ideal = point.rung(FrontierRung.LOGGOPSIM_IDEAL)
        if ideal.fabric_leg_ps == 0:
            continue
        argv = list(ideal.provenance.argv)
        required = ("-f", "-L", "-o", "-g", "-G", "-O", "-S", "-n")
        if len(argv) != 1 + 2 * len(required):
            return False
        if any(argv.count(flag) != 1 for flag in required):
            return False
        if argv[0] != "LogGOPSim" or argv[argv.index("-G") + 1] != EXACT_G_STRING:
            return False
        goal_path = Path(argv[argv.index("-f") + 1])
        if goal_path.is_absolute() or ".." in goal_path.parts:
            return False
        if ideal.provenance.binary_sha256 != PINNED_LOGGOPSIM_SHA256:
            return False
        executed.append((point.configuration_id, point.batch_per_gpu))
    return len(executed) == 12 and len(executed) == len(set(executed))


def _fatal_guards(
    *,
    input_rows: list[dict[str, object]],
    record_json: dict[str, object],
    implementation_commit: str,
) -> list[dict[str, object]]:
    record_valid = _record_holds(record_json)
    execution_valid = _execution_holds(record_json)
    chronology = _is_ancestor(EXPECTATIONS_COMMIT, implementation_commit)

    mutant_inputs = deepcopy(input_rows)
    mutant_inputs[1]["observed_sha256"] = "0" * 64
    mutant_inputs[1]["matched"] = False
    class_mutant = deepcopy(record_json)
    class_mutant["points"][0]["rungs"][2]["point_class"] = PointClass.ESTIMATE.value  # type: ignore[index]
    execution_mutant = deepcopy(record_json)
    ideal_rung = next(
        rung
        for point in execution_mutant["points"]  # type: ignore[index]
        for rung in point["rungs"]
        if rung["rung"] == FrontierRung.LOGGOPSIM_IDEAL.value
        and rung["fabric_leg_ps"] > 0
    )
    argv = ideal_rung["provenance"]["argv"]
    gap_index = argv.index("-G")
    del argv[gap_index : gap_index + 2]
    return [
        {
            "id": "FG-1",
            "held": _fg1_holds(input_rows),
            "enforcement": "runtime",
            "evaluated": "pinned record, contract and LogGOPSim binary hashes match before execution",
            "mutation_control": _mutation_control(
                "pinned record digest replaced with 64 zeroes",
                not _fg1_holds(mutant_inputs),
            ),
        },
        {
            "id": "FG-2",
            "held": record_valid,
            "enforcement": "runtime",
            "evaluated": "all 18 ladder points carry three strict, correctly classified and distinct rung authorities",
            "mutation_control": _mutation_control(
                "one packet rung relabeled ESTIMATE on the wire",
                not _record_holds(class_mutant),
            ),
        },
        {
            "id": "FG-3",
            "held": execution_valid,
            "enforcement": "runtime",
            "evaluated": "all 12 nonzero ideal fabric legs carry executed native provenance, portable argv and exact G",
            "mutation_control": _mutation_control(
                "one real ideal-rung argv passed to the predicate without -G",
                not _execution_holds(execution_mutant),
            ),
        },
        {
            "id": "FG-4",
            "held": chronology,
            "enforcement": "runtime",
            "evaluated": "the frozen expectations commit is an ancestor of the implementation run commit",
            "mutation_control": _mutation_control(
                "implementation and expectations revisions reversed",
                not _is_ancestor(implementation_commit, EXPECTATIONS_COMMIT),
            ),
        },
    ]


def _preflight_void(
    input_rows: list[dict[str, object]],
    implementation_commit: str,
    finding: str,
) -> dict[str, object]:
    mutant_inputs = deepcopy(input_rows)
    if mutant_inputs:
        mutant_inputs[0]["matched"] = False
    return {
        "schema": RESULT_SCHEMA,
        "verdict": "VOID",
        "findings": [finding],
        "input_hashes": input_rows,
        "fatal_guards": [
            {
                "id": "FG-1" if not _fg1_holds(input_rows) else "FG-4",
                "held": False,
                "enforcement": "runtime",
                "evaluated": finding,
                "mutation_control": _mutation_control(
                    "one preflight predicate input changed",
                    not _fg1_holds(mutant_inputs),
                ),
            }
        ],
        "chronology": {
            "expectations_commit": EXPECTATIONS_COMMIT,
            "implementation_commit": implementation_commit,
        },
        "score_classes": {},
    }


def _physical_sanity(
    legacy_rows: dict[tuple[str, int], dict[str, Any]],
    ideal_rows: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, object]:
    serialized = legacy_rows[("h100-two-node-serialized", 32)]
    incast = legacy_rows[("h100-nine-node-incast", 32)]
    serialized_payload = max(serialized["byte_partition"]["remote_flow_payload_bytes"])
    serialized_floor_ps = serialized_payload * 20
    serialized_ceiling_ps = serialized_floor_ps + 2_000_000
    serialized_observed_ps = (
        ideal_rows[("h100-two-node-serialized", 32)]["observed_ns"] * 1_000
    )
    remote_bytes = incast["byte_partition"]["remote_logical_bytes"]
    incast_floor_ps = remote_bytes * 20
    incast_ceiling_ps = 8 * incast["fabric_observation"]["isolated_service_ps"]
    incast_packet_ps = incast["fabric_observation"]["concurrent_service_ps"]
    ideal_b16 = ideal_rows[("h100-nine-node-incast", 16)]["observed_ns"]
    ideal_b32 = ideal_rows[("h100-nine-node-incast", 32)]["observed_ns"]
    packet_b16 = legacy_rows[("h100-nine-node-incast", 16)]["fabric_observation"][
        "concurrent_service_ps"
    ]
    packet_b32 = incast_packet_ps
    return {
        "serialization_physics": {
            "floor_ps": serialized_floor_ps,
            "ceiling_ps": serialized_ceiling_ps,
            "observed_ps": serialized_observed_ps,
            "inside_bounds": serialized_floor_ps
            <= serialized_observed_ps
            <= serialized_ceiling_ps,
            "meaning": "payload serialization is the floor and one declared propagation delay is the ceiling increment",
        },
        "shared_ingress_physics": {
            "floor_ps": incast_floor_ps,
            "ceiling_ps": incast_ceiling_ps,
            "observed_ps": incast_packet_ps,
            "inside_bounds": incast_floor_ps <= incast_packet_ps <= incast_ceiling_ps,
            "meaning": "eight payloads must cross one ingress and cannot exceed eight isolated completions in series",
        },
        "batch_scaling": {
            "ideal_b32_over_b16": _fraction_record(ideal_b32, ideal_b16),
            "packet_b32_over_b16": _fraction_record(packet_b32, packet_b16),
            "expected_shape": "both serialization-dominated legs approximately double",
        },
        "step_plausibility": {
            "h100_batch32_kernel_floor_ps": incast["kernel"]["kernel_floor_ps"],
            "h100_batch32_packet_fabric_ps": incast_packet_ps,
            "kernel_masks_fabric": incast["kernel"]["kernel_floor_ps"] > incast_packet_ps,
            "meaning": "the 9.536 ms H100 kernel is slower than the 7.689 ms packet fabric leg, so the step does not move",
        },
    }


def _csv_rows(result: dict[str, Any]) -> list[dict[str, object]]:
    rows = []
    for family in ("L-A", "L-B"):
        for row in result["score_classes"]["exact_oracles"]["families"][family]["rows"]:
            rows.append(
                {
                    "evidence_class": "exact-oracle",
                    "family": family,
                    "id": row["id"],
                    "expected": row["expected_ns"],
                    "observed": row["observed_ns"],
                    "units": "ns",
                    "passed": str(row["passed"]).lower(),
                }
            )
    for family in ("M-1", "M-2", "M-3"):
        for row in result["score_classes"]["behavioral_relations"]["families"][family]["rows"]:
            rows.append(
                {
                    "evidence_class": "behavioral-relation",
                    "family": family,
                    "id": row["id"],
                    "expected": json.dumps(row["expected_band"], sort_keys=True),
                    "observed": json.dumps(row["quotient"], sort_keys=True),
                    "units": "quotient",
                    "passed": str(row["passed"]).lower(),
                }
            )
    for row in result["score_classes"]["behavioral_relations"]["families"]["S"]["rows"]:
        rows.append(
            {
                "evidence_class": "behavioral-relation",
                "family": "S",
                "id": row["id"],
                "expected": json.dumps(row["expected"], sort_keys=True),
                "observed": json.dumps(row["observed_step_ps"], sort_keys=True),
                "units": "ps",
                "passed": str(row["passed"]).lower(),
            }
        )
    for row in result["score_classes"]["plot_contract"]["rows"]:
        rows.append(
            {
                "evidence_class": "plot-contract",
                "family": "P",
                "id": row["id"],
                "expected": row["predicate"],
                "observed": "predicate held" if row["passed"] else "predicate missed",
                "units": "predicate",
                "passed": str(row["passed"]).lower(),
            }
        )
    wall = result["score_classes"]["wall_time"]
    rows.append(
        {
            "evidence_class": "wall-time",
            "family": "W",
            "id": "W-total",
            "expected": wall["ceiling_seconds"],
            "observed": wall["median_seconds"],
            "units": "s",
            "passed": str(wall["passed"] == wall["denominator"]).lower(),
        }
    )
    return rows


def run_study(args: argparse.Namespace) -> dict[str, object]:
    """Execute seven native batches and score every frozen family."""

    attempt_dir = args.run_dir.resolve()
    binary = args.binary.resolve()
    txt2bin = args.txt2bin.resolve()
    implementation_commit = _git_hash("HEAD")
    input_rows = _input_rows(binary)
    if not _fg1_holds(input_rows):
        return _preflight_void(input_rows, implementation_commit, "FG-1 input hash mismatch")
    if not _is_ancestor(EXPECTATIONS_COMMIT, implementation_commit):
        return _preflight_void(
            input_rows,
            implementation_commit,
            "FG-4 expectations commit is not an ancestor of the implementation commit",
        )
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status.returncode or status.stdout.strip():
        raise SystemExit("the frozen ladder run requires a clean tracked worktree")
    if not txt2bin.is_file():
        raise SystemExit(f"txt2bin is not a file: {txt2bin}")

    legacy = json.loads(PINNED_RECORD_PATH.read_text(encoding="utf-8"))
    legacy_rows = {
        (row["configuration_id"], row["batch_per_gpu"]): row
        for row in legacy["points"]
    }
    goal_dir = attempt_dir / "goals"
    goal_dir.mkdir(parents=True, exist_ok=True)
    prepared: dict[str, dict[str, Any]] = {}
    for cell in _cells():
        source, binary_goal = _render_goal(cell, goal_dir, txt2bin)
        prepared[cell.cell_id] = {
            "cell": cell,
            "goal_path": source,
            "goal_binary_path": binary_goal,
            "goal_text_sha256": _sha256_path(source),
            "goal_binary_sha256": _sha256_path(binary_goal),
            "argv": build_loggopsim_command(binary, _config(cell, binary_goal)),
        }

    recorder = NativeEvidenceRecorder(attempt_dir)
    executions: dict[str, list[NativeExecution]] = defaultdict(list)
    wall_samples = []
    for sample in range(1, WALL_SAMPLES + 1):
        started = time.perf_counter_ns()
        for cell in _cells():
            execution = _execute(
                prepared[cell.cell_id]["argv"],
                recorder,
                f"{cell.cell_id}-wall-sample-{sample}",
            )
            executions[cell.cell_id].append(execution)
        wall_samples.append((time.perf_counter_ns() - started) / 1_000_000_000)

    exact_families: dict[str, dict[str, object]] = {}
    ideal_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for family in ("L-A", "L-B"):
        rows = []
        for cell in (item for item in _cells() if item.family == family):
            cell_executions = executions[cell.cell_id]
            observed_ns = cell_executions[0].max_finish_ps // 1_000
            all_observed_ns = [item.max_finish_ps // 1_000 for item in cell_executions]
            passed = all(value == cell.expected_ns for value in all_observed_ns)
            source = prepared[cell.cell_id]["goal_path"]
            binary_goal = prepared[cell.cell_id]["goal_binary_path"]
            row = {
                "id": cell.cell_id,
                "family": family,
                "configuration_id": cell.configuration_id,
                "batch_per_gpu": cell.batch_per_gpu,
                "rank_count": cell.rank_count,
                "flow_count": cell.flow_count,
                "payload_bytes": cell.payload_bytes,
                "expected_ns": cell.expected_ns,
                "observed_ns": observed_ns,
                "all_observed_ns": all_observed_ns,
                "parameters": {
                    "L_ns": 2_000,
                    "o_ns": 0,
                    "g_ns": 0,
                    "G_ns_per_byte": EXACT_G_STRING,
                    "O_ns_per_byte": 0,
                    "S_bytes": cell.payload_bytes + 1,
                },
                "argv": _portable_argv(prepared[cell.cell_id]["argv"], attempt_dir),
                "goal_path": source.relative_to(attempt_dir).as_posix(),
                "goal_text_sha256": prepared[cell.cell_id]["goal_text_sha256"],
                "goal_binary_path": binary_goal.relative_to(attempt_dir).as_posix(),
                "goal_binary_sha256": prepared[cell.cell_id]["goal_binary_sha256"],
                "stdout_sha256": [item.stdout_sha256 for item in cell_executions],
                "stderr_sha256": [item.stderr_sha256 for item in cell_executions],
                "native_evidence": [item.evidence_path for item in cell_executions],
                "executions": len(cell_executions),
                "oracle_source": "frozen literal in expectations.md",
                "passed": passed,
            }
            rows.append(row)
            ideal_rows[(cell.configuration_id, cell.batch_per_gpu)] = row
        exact_families[family] = {
            "evidence_class": "exact-oracle",
            "denominator": len(rows),
            "passed": sum(row["passed"] for row in rows),
            "rows": rows,
        }

    record = _build_record(legacy, ideal_rows, PINNED_LOGGOPSIM_SHA256)
    record_json = frontier_ladder_record_to_json(record)
    mechanism_families, envelope = _mechanism_rows(legacy_rows, ideal_rows)
    step_family = _step_family(record)
    plot_family = _plot_family(record, envelope)
    wall_median = statistics.median(wall_samples)
    wall_pass = wall_median <= WALL_CEILING_SECONDS
    wall_family = {
        "evidence_class": "wall-time",
        "denominator": 1,
        "passed": int(wall_pass),
        "samples_seconds": wall_samples,
        "median_seconds": wall_median,
        "ceiling_seconds": WALL_CEILING_SECONDS,
        "native_legs_per_sample": 12,
        "native_executions": 12 * WALL_SAMPLES,
        "packet_executions": 0,
        "machine": _machine_disclosure(),
    }
    guards = _fatal_guards(
        input_rows=input_rows,
        record_json=record_json,
        implementation_commit=implementation_commit,
    )
    guard_void = not all(bool(guard["held"]) for guard in guards)
    score_pass = (
        all(
            family["passed"] == family["denominator"]
            for family in exact_families.values()
        )
        and all(
            family["passed"] == family["denominator"]
            for family in mechanism_families.values()
        )
        and step_family["passed"] == step_family["denominator"]
        and plot_family["passed"] == plot_family["denominator"]
        and wall_family["passed"] == wall_family["denominator"]
    )
    verdict = "VOID" if guard_void else ("PASS" if score_pass else "REFUTED")
    result: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "verdict": verdict,
        "findings": [
            guard["id"] for guard in guards if not guard["held"]
        ],
        "scope": (
            "Three-rung deployment frontier over the pinned CORE-62 grid. "
            "Only the twelve LogGOPSim ideal fabric legs execute here; packet values "
            "are read from the pinned record and no silicon accuracy is claimed."
        ),
        "chronology": {
            "expectations_commit": EXPECTATIONS_COMMIT,
            "expectations_sha256": EXPECTATIONS_SHA256,
            "implementation_commit": implementation_commit,
        },
        "input_hashes": input_rows,
        "native_tools": {
            "loggopsim": {
                "filename": binary.name,
                "sha256": _sha256_path(binary),
            },
            "txt2bin": {
                "filename": txt2bin.name,
                "sha256": _sha256_path(txt2bin),
            },
        },
        "attempt_evidence": {
            "policy": (
                "each run uses a fresh attempt-N directory and refuses a later "
                "attempt until every earlier attempt has a verdict"
            ),
            "stdout_bytes": "native/*/execution-*.stdout",
            "stderr_bytes": "native/*/execution-*.stderr",
            "portable_argv": "native/*/execution-*.json",
            "verdict": "verdict.json",
        },
        "ladder_record": record_json,
        "fabric_leg_envelope": envelope,
        "score_classes": {
            "exact_oracles": {
                "evidence_class": "exact-oracle",
                "families": exact_families,
            },
            "behavioral_relations": {
                "evidence_class": "behavioral-relation",
                "families": {**mechanism_families, "S": step_family},
            },
            "plot_contract": plot_family,
            "wall_time": wall_family,
        },
        "fatal_guards": guards,
        "physical_sanity": _physical_sanity(legacy_rows, ideal_rows),
        "evidence_separation": (
            "exact-oracle, behavioral-relation, plot-contract and wall-time "
            "denominators remain separate; fatal guards are unscored and void the run"
        ),
    }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--binary",
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
                "schema": "simllm-frontier-ladder-attempt-v1",
                "verdict": "ERROR",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    _write_json(verdict_path, result)
    if args.results_json is not None:
        _write_json(args.results_json, result)
    csv_path = args.results_csv or attempt_dir / "results.csv"
    if result.get("score_classes"):
        _write_csv(csv_path, _csv_rows(result))
    sys.stdout.write(f"verdict={result['verdict']} results={verdict_path}\n")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
