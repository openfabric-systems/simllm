#!/usr/bin/env python3
"""Fit the frozen TRAF-81 donor and score untouched wide-rank measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import statistics
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simllm.compute import nearest_rank
from simllm.traffic import (
    CollectiveFloorCalibration,
    CollectiveFloorCell,
    CollectiveFloorCurveBoundaries,
    CollectiveFloorRegime,
    CollectiveFloorSourceIdentity,
    choose_collective_floor_boundaries,
    fit_collective_floor_calibration,
)

STUDY_ROOT = Path(__file__).resolve().parent
FREEZE_PATH = STUDY_ROOT / "expectations.json"
OPERATIONS = ("all_gather", "reduce_scatter")
TRAINING_RANKS = (2, 4)
SCORED_RANKS = (8, 16)
FREEZE_COMMIT = "3f0aa24ea16573e3fc2ca030d541009cf308d12f"


class StudyError(RuntimeError):
    """Raised when an input violates a frozen structural requirement."""


@dataclass(frozen=True)
class MeasurementRow:
    operation: str
    ranks: int
    message_bytes: int
    latency_ps: int
    samples_us: tuple[float, ...]

    @property
    def curve_key(self) -> tuple[str, int]:
        return self.operation, self.ranks


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_freeze(path: Path = FREEZE_PATH) -> dict[str, Any]:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("schema") != "simllm-collective-floor-extrapolation-freeze-v1":
        raise StudyError("unsupported TRAF-81 freeze schema")
    if freeze.get("task_id") != "TRAF-81":
        raise StudyError("freeze task identity is not TRAF-81")
    if freeze["fit"]["training_ranks"] != list(TRAINING_RANKS):
        raise StudyError("freeze training ranks changed")
    if freeze["fit"]["donor_rank"] != 4:
        raise StudyError("freeze donor rank changed")
    return freeze


def load_measurement(path: Path, freeze: dict[str, Any]) -> tuple[dict[str, Any], list[MeasurementRow]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "simllm-collective-floor-measurement-v1":
        raise StudyError(f"{path}: unsupported measurement schema")
    ranks = document.get("world")
    rank_config = next(
        (row for row in freeze["rank_cells"] if row["ranks"] == ranks), None
    )
    if rank_config is None:
        raise StudyError(f"{path}: rank {ranks!r} is outside the frozen grid")
    if document.get("warmup_iterations") != freeze["harness"]["warmup_iterations"]:
        raise StudyError(f"{path}: warmup count changed")
    repetitions = freeze["harness"]["timed_repetitions"]
    if document.get("timed_repetitions") != repetitions:
        raise StudyError(f"{path}: timed repetition count changed")
    if document.get("sample_reduction") != "maximum-over-ranks":
        raise StudyError(f"{path}: sample reduction changed")
    if document.get("aggregation") != "observed-median":
        raise StudyError(f"{path}: aggregation changed")
    if document.get("nccl_version") != freeze["environment"]["expected_nccl_version"]:
        raise StudyError(f"{path}: NCCL version differs from the freeze")
    if "A100-SXM4-80GB" not in document.get("rank0_device_name", ""):
        raise StudyError(f"{path}: rank-zero GPU identity is not A100 SXM4 80GB")

    expected_keys = {
        (operation, message_bytes)
        for operation in OPERATIONS
        for message_bytes in freeze["byte_grid"]
    }
    observed_keys: set[tuple[str, int]] = set()
    rows: list[MeasurementRow] = []
    q = Fraction(ranks - 1, ranks)
    ceiling = rank_config["ceiling_bytes_per_second"]
    for raw in document.get("measurements", []):
        operation = raw.get("operation")
        message_bytes = raw.get("operation_buffer_bytes")
        key = operation, message_bytes
        if key in observed_keys:
            raise StudyError(f"{path}: duplicate measurement {key!r}")
        observed_keys.add(key)
        if key not in expected_keys:
            raise StudyError(f"{path}: unexpected measurement {key!r}")
        samples = tuple(float(value) for value in raw.get("samples_us", []))
        if len(samples) != repetitions:
            raise StudyError(f"{path}: {key!r} does not carry {repetitions} samples")
        if any(not math.isfinite(value) or value <= 0.0 for value in samples):
            raise StudyError(f"{path}: {key!r} contains a nonpositive sample")
        observed_median = statistics.median(samples)
        recorded_median = float(raw.get("median_us"))
        if not math.isclose(observed_median, recorded_median, rel_tol=1e-10, abs_tol=1e-9):
            raise StudyError(f"{path}: {key!r} median does not match its samples")
        if raw.get("max_rank_mismatches") != 0:
            raise StudyError(f"{path}: {key!r} violates value conservation")
        latency_ps = round(recorded_median * 1_000_000)
        bandwidth = float(q * message_bytes * 1_000_000_000_000 / latency_ps)
        if bandwidth > ceiling * (1.0 + 1e-9):
            raise StudyError(
                f"{path}: {key!r} reports {bandwidth} B/s above ceiling {ceiling}"
            )
        rows.append(
            MeasurementRow(
                operation=operation,
                ranks=ranks,
                message_bytes=message_bytes,
                latency_ps=latency_ps,
                samples_us=samples,
            )
        )
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        raise StudyError(f"{path}: frozen rows are missing: {missing!r}")
    rows.sort(key=lambda row: (row.operation, row.message_bytes))
    return document, rows


def _key_values(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def _hash_manifest(path: Path) -> dict[str, str]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise StudyError(f"{path}: malformed SHA-256 manifest line {line!r}")
        records[Path(name).name] = digest
    return records


def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *arguments),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _section(text: str, start: str, end: str) -> str:
    try:
        return text.split(start, 1)[1].split(end, 1)[0]
    except IndexError as error:
        raise StudyError(f"guard output lacks section {start!r} to {end!r}") from error


def _assigned_gpu(text: str) -> tuple[str, str, str]:
    matches = re.findall(r"^assigned_gpu=([^,]+), ([^,]+), (.+)$", text, re.MULTILINE)
    if len(matches) != 1:
        raise StudyError("guard output does not identify exactly one assigned GPU")
    return matches[0]


def _nv4_rows(text: str, expected_gpus: int) -> int:
    count = 0
    for line in text.splitlines():
        fields = line.split()
        if fields and re.fullmatch(r"GPU[0-9]+", fields[0]):
            count += fields[1:].count("NV4") == expected_gpus - 1
    return count


def _audit_evidence_cell(
    ranks: int,
    root: Path,
    measurement_path: Path,
    freeze: dict[str, Any],
) -> dict[str, Any]:
    rank_config = _rank_config(freeze, ranks)
    required = (
        "cell_state.txt",
        "inputs.sha256",
        "job_context.txt",
        "measurement.json",
        "nvcc_version.txt",
        "outputs.sha256",
        "run.exit_status",
        "run.txt",
        "transport_summary.txt",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise StudyError(f"rank {ranks} evidence lacks files: {missing}")
    if (root / "cell_state.txt").read_text(encoding="utf-8").strip() != "MEASURED":
        raise StudyError(f"rank {ranks} evidence is not in MEASURED state")
    if (root / "run.exit_status").read_text(encoding="utf-8").strip() != "0":
        raise StudyError(f"rank {ranks} target lane did not exit successfully")
    if "release 12.2" not in (root / "nvcc_version.txt").read_text(encoding="utf-8"):
        raise StudyError(f"rank {ranks} did not use the frozen CUDA compiler")

    context = _key_values(root / "job_context.txt")
    expected_cell = f"rank-{ranks}"
    expected_context = {
        "cell": expected_cell,
        "freeze_commit": FREEZE_COMMIT,
        "nodes": str(rank_config["nodes"]),
        "tasks": str(ranks),
    }
    for key, expected in expected_context.items():
        if context.get(key) != expected:
            raise StudyError(
                f"rank {ranks} job context {key} is {context.get(key)!r}, "
                f"expected {expected!r}"
            )
    expected_output = (
        f"{context['data_root'].rstrip('/')}/{expected_cell}/{context['job_id']}"
    )
    if context.get("output_root") != expected_output:
        raise StudyError(f"rank {ranks} result path escaped the configured run root")

    harness_commit = context.get("harness_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", harness_commit):
        raise StudyError(f"rank {ranks} lacks a full harness commit identity")
    ancestor = _git("merge-base", "--is-ancestor", FREEZE_COMMIT, harness_commit)
    if ancestor.returncode != 0:
        raise StudyError(f"rank {ranks} harness commit does not follow the freeze")
    freeze_time = _git("show", "-s", "--format=%cI", FREEZE_COMMIT)
    harness_time = _git("show", "-s", "--format=%cI", harness_commit)
    if freeze_time.returncode or harness_time.returncode:
        raise StudyError("cannot read freeze or harness commit chronology")
    job_time = datetime.fromisoformat(context["date"])
    latest_commit_time = max(
        datetime.fromisoformat(freeze_time.stdout.strip()),
        datetime.fromisoformat(harness_time.stdout.strip()),
    )
    if job_time <= latest_commit_time:
        raise StudyError(f"rank {ranks} job did not start after frozen commits")

    inputs = _hash_manifest(root / "inputs.sha256")
    committed_names = (
        "collective_lane.cu",
        "collect_guard.sh",
        "run_merlin_cell.sh",
        f"measure_r{ranks}.sbatch",
    )
    for name in committed_names:
        committed = _git(
            "show",
            f"{harness_commit}:examples/collective_floor_extrapolation_v1/{name}",
        )
        if committed.returncode != 0:
            raise StudyError(f"rank {ranks} harness commit does not contain {name}")
        digest = _sha256_bytes(committed.stdout.encode("utf-8"))
        if inputs.get(name) != digest:
            raise StudyError(f"rank {ranks} staged {name} differs from its commit")
    for name in ("collective_lane", "libnccl.so.2", "nccl.h"):
        if name not in inputs:
            raise StudyError(f"rank {ranks} input identity lacks {name}")

    outputs = _hash_manifest(root / "outputs.sha256")
    for name in ("measurement.json", "job_context.txt", "transport_summary.txt"):
        if outputs.get(name) != _sha256_path(root / name):
            raise StudyError(f"rank {ranks} output digest disagrees for {name}")
    if _sha256_path(measurement_path) != outputs["measurement.json"]:
        raise StudyError(f"rank {ranks} scored measurement is not the retained output")

    before_paths = sorted(root.glob("rank_*/guards_before.txt"))
    after_paths = sorted(root.glob("rank_*/guards_after.txt"))
    if len(before_paths) != ranks or len(after_paths) != ranks:
        raise StudyError(f"rank {ranks} does not carry one before/after guard per rank")
    after_by_parent = {path.parent.name: path for path in after_paths}
    assigned = []
    host_topologies: dict[str, int] = {}
    cassini_counts: dict[str, int] = {}
    for before in before_paths:
        after = after_by_parent.get(before.parent.name)
        if after is None:
            raise StudyError(f"rank {ranks} guard pairing is incomplete")
        before_text = before.read_text(encoding="utf-8")
        after_text = after.read_text(encoding="utf-8")
        before_gpu = _assigned_gpu(before_text)
        after_gpu = _assigned_gpu(after_text)
        if before_gpu != after_gpu:
            raise StudyError(f"rank {ranks} assigned GPU changed during measurement")
        if before_gpu[1] != freeze["environment"]["gpu"]:
            raise StudyError(f"rank {ranks} was not placed on the frozen A100 target")
        assigned.append(before_gpu[0])
        for text in (before_text, after_text):
            processes = _section(
                text, "=== compute processes ===", "=== high-speed ports ==="
            )
            if re.search(r"^GPU-[^,]+, [0-9]+,", processes, re.MULTILINE):
                raise StudyError(f"rank {ranks} assigned GPU had a foreign process")
        host_match = re.search(r"^host=(.+)$", before_text, re.MULTILINE)
        cassini_match = re.search(
            r"^cassini_port_count=(\d+)$", before_text, re.MULTILINE
        )
        if host_match is None or cassini_match is None:
            raise StudyError(f"rank {ranks} guard lacks host or port identity")
        host = host_match.group(1)
        host_topologies[host] = _nv4_rows(
            before_text, rank_config["gpus_per_node"]
        )
        cassini_counts[host] = int(cassini_match.group(1))
    if len(set(assigned)) != ranks:
        raise StudyError(f"rank {ranks} does not map ranks to distinct GPUs")
    if len(host_topologies) != rank_config["nodes"]:
        raise StudyError(f"rank {ranks} ran on the wrong node count")
    if any(
        value != rank_config["gpus_per_node"]
        for value in host_topologies.values()
    ):
        raise StudyError(f"rank {ranks} GPUs are not a direct NV4 submesh")
    if ranks in SCORED_RANKS and any(value != 4 for value in cassini_counts.values()):
        raise StudyError(f"rank {ranks} node lacks four Cassini ports")

    transport_text = (root / "transport_summary.txt").read_text(encoding="utf-8")
    network_classes = sorted(
        set(re.findall(r"Using network ([A-Za-z0-9_-]+)", transport_text))
    )
    gdr_states = sorted(set(re.findall(r"\bGDR ([01])\b", transport_text)))
    if ranks in SCORED_RANKS:
        if not network_classes or not gdr_states or "via NET/" not in transport_text:
            raise StudyError(f"rank {ranks} cross-node transport identity is incomplete")
    elif "via P2P/" not in transport_text:
        raise StudyError(f"rank {ranks} intra-node transport lacks a P2P path")

    return {
        "ranks": ranks,
        "job_id": context["job_id"],
        "nodelist": context["nodelist"],
        "freeze_commit": context["freeze_commit"],
        "harness_commit": harness_commit,
        "input_sha256": inputs,
        "output_sha256": outputs,
        "assigned_gpu_uuids": sorted(assigned),
        "hosts": sorted(host_topologies),
        "nv4_rows_per_host": host_topologies,
        "cassini_ports_per_host": cassini_counts,
        "network_classes": network_classes,
        "gdr_states": gdr_states,
    }


def audit_evidence(
    evidence_paths: dict[int, Path],
    measurement_paths: dict[int, Path],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    if set(evidence_paths) != set(measurement_paths):
        raise StudyError("every measured rank needs exactly one evidence directory")
    cells = [
        _audit_evidence_cell(
            ranks,
            evidence_paths[ranks],
            measurement_paths[ranks],
            freeze,
        )
        for ranks in sorted(evidence_paths)
    ]
    wide_transports = {
        (
            tuple(cell["network_classes"]),
            tuple(cell["gdr_states"]),
        )
        for cell in cells
        if cell["ranks"] in SCORED_RANKS
    }
    if len(wide_transports) > 1:
        raise StudyError("rank 8 and rank 16 used different network transport classes")
    harness_commits = {cell["harness_commit"] for cell in cells}
    if len(harness_commits) != 1:
        raise StudyError("rank cells used different harness commits")
    return {
        "status": "HELD",
        "cells": cells,
        "fatal_guards": [
            {"id": f"FG-{index}", "held": True}
            for index in range(1, 8)
        ],
    }


def _source(measurement_hash: str) -> CollectiveFloorSourceIdentity:
    return CollectiveFloorSourceIdentity(
        artifact_sha256=measurement_hash,
        tool="TRAF-81 minimal CUDA NCCL lane",
        aiconfigurator_version="not-used",
        aiconfigurator_core_version="not-used",
        system="Merlin A100-SXM4-80GB",
        backend="NCCL 2.31.2",
        database_version="TRAF-81-v1",
        row_version="measured-median-v1",
        duplicate_resolution="duplicates forbidden",
    )


def _floor_cell(row: MeasurementRow) -> CollectiveFloorCell:
    return CollectiveFloorCell(
        cell_id=f"{row.operation}-r{row.ranks}-b{row.message_bytes}",
        dtype="half",
        operation=row.operation,
        ranks=row.ranks,
        source_elements=row.message_bytes // 2,
        message_bytes=row.message_bytes,
        latency_ps=row.latency_ps,
    )


def _fit_training(
    rows: Sequence[MeasurementRow], freeze: dict[str, Any], measurement_hash: str
) -> tuple[CollectiveFloorCalibration, dict[tuple[str, int], tuple[int, ...]]]:
    training = [row for row in rows if row.ranks in TRAINING_RANKS]
    expected_curve_keys = {
        (operation, ranks) for operation in OPERATIONS for ranks in TRAINING_RANKS
    }
    if {row.curve_key for row in training} != expected_curve_keys:
        raise StudyError("both training ranks and operations must be measured")

    boundaries: dict[tuple[str, int], tuple[int, ...]] = {}
    boundary_records = []
    for operation, ranks in sorted(expected_curve_keys):
        cells = [
            _floor_cell(row)
            for row in training
            if row.operation == operation and row.ranks == ranks
        ]
        selected = choose_collective_floor_boundaries(
            cells,
            maximum_regimes=freeze["fit"]["maximum_regimes"],
            minimum_cells_per_regime=freeze["fit"]["minimum_cells_per_regime"],
        )
        boundaries[operation, ranks] = selected
        boundary_records.append(
            CollectiveFloorCurveBoundaries(
                dtype="half",
                operation=operation,
                ranks=ranks,
                lower_bounds_of_following_regimes=selected,
            )
        )

    calibration = fit_collective_floor_calibration(
        calibration_id="traf81-a100-r2-r4-training-v1",
        source=_source(measurement_hash),
        cells=[_floor_cell(row) for row in training],
        boundaries=boundary_records,
        fitted_byte_range=(min(freeze["byte_grid"]), max(freeze["byte_grid"])),
    )
    return calibration, boundaries


def _fit_wide_curve(
    rows: Sequence[MeasurementRow],
    operation: str,
    ranks: int,
    boundaries: tuple[int, ...],
    freeze: dict[str, Any],
    measurement_hash: str,
) -> tuple[CollectiveFloorRegime, ...]:
    curve_rows = [
        row for row in rows if row.operation == operation and row.ranks == ranks
    ]
    calibration = fit_collective_floor_calibration(
        calibration_id=f"traf81-a100-r{ranks}-{operation}-descriptive-v1",
        source=_source(measurement_hash),
        cells=[_floor_cell(row) for row in curve_rows],
        boundaries=[
            CollectiveFloorCurveBoundaries(
                dtype="half",
                operation=operation,
                ranks=ranks,
                lower_bounds_of_following_regimes=boundaries,
            )
        ],
        fitted_byte_range=(min(freeze["byte_grid"]), max(freeze["byte_grid"])),
    )
    return calibration.regimes


def _regime(
    regimes: Iterable[CollectiveFloorRegime], message_bytes: int
) -> CollectiveFloorRegime:
    try:
        return next(
            regime
            for regime in regimes
            if regime.lower_bytes <= message_bytes <= regime.upper_bytes
        )
    except StopIteration as error:
        raise StudyError(f"no fitted regime contains {message_bytes} bytes") from error


def _rank_config(freeze: dict[str, Any], ranks: int) -> dict[str, Any]:
    return next(row for row in freeze["rank_cells"] if row["ranks"] == ranks)


def _scale(freeze: dict[str, Any], ranks: int) -> Fraction:
    donor_rank = freeze["fit"]["donor_rank"]
    donor_ceiling = _rank_config(freeze, donor_rank)["ceiling_bytes_per_second"]
    ceiling = _rank_config(freeze, ranks)["ceiling_bytes_per_second"]
    q_donor = Fraction(donor_rank - 1, donor_rank)
    q_target = Fraction(ranks - 1, ranks)
    return q_target / q_donor * Fraction(donor_ceiling, ceiling)


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise StudyError("cannot summarize an empty sequence")
    return {
        "median": float(statistics.median(values)),
        "p95_nearest_rank": float(nearest_rank(values, 0.95)),
        "maximum": float(max(values)),
    }


def _score_dips(
    rows: Sequence[MeasurementRow], freeze: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {"curves": []}
    minimum_location = freeze["scored_families"]["S4"]["minimum_location_bytes"]
    maximum_location = freeze["scored_families"]["S4"]["maximum_location_bytes"]
    minimum_depth = freeze["scored_families"]["S4"]["minimum_depth"]
    rank_two_passes = []
    for ranks in sorted({row.ranks for row in rows}):
        ceiling = _rank_config(freeze, ranks)["ceiling_bytes_per_second"]
        q = Fraction(ranks - 1, ranks)
        for operation in OPERATIONS:
            curve = sorted(
                (
                    row
                    for row in rows
                    if row.ranks == ranks and row.operation == operation
                ),
                key=lambda row: row.message_bytes,
            )
            if not curve:
                continue
            small_floor = round(statistics.median(row.latency_ps for row in curve[:3]))
            efficiencies: list[tuple[int, float]] = []
            speedup_steps = []
            for previous, row in itertools.pairwise(curve):
                if row.latency_ps < previous.latency_ps:
                    speedup_steps.append(
                        {
                            "from_bytes": previous.message_bytes,
                            "to_bytes": row.message_bytes,
                            "latency_ratio": row.latency_ps / previous.latency_ps,
                        }
                    )
            for row in curve[3:]:
                residual = row.latency_ps - small_floor
                if residual <= 0:
                    continue
                efficiency = float(
                    q
                    * row.message_bytes
                    * 1_000_000_000_000
                    / residual
                    / ceiling
                )
                efficiencies.append((row.message_bytes, efficiency))
            dips = []
            for before, point, after in zip(
                efficiencies, efficiencies[1:], efficiencies[2:]
            ):
                if point[1] < before[1] and point[1] < after[1]:
                    depth = 1.0 - point[1] / min(before[1], after[1])
                    dips.append({"bytes": point[0], "depth": depth})
            strongest = max(dips, key=lambda item: item["depth"], default=None)
            curve_pass = any(
                minimum_location <= item["bytes"] <= maximum_location
                and item["depth"] >= minimum_depth
                for item in dips
            )
            if ranks == 2:
                rank_two_passes.append(curve_pass)
            result["curves"].append(
                {
                    "operation": operation,
                    "ranks": ranks,
                    "small_message_floor_ps": small_floor,
                    "strongest_dip": strongest,
                    "all_dips": dips,
                    "speedup_steps": speedup_steps,
                    "frozen_rank_two_predicate": curve_pass if ranks == 2 else None,
                }
            )
    result["held"] = any(rank_two_passes)
    return result


def score_measurements(
    documents: dict[int, dict[str, Any]],
    rows: Sequence[MeasurementRow],
    freeze: dict[str, Any],
    *,
    blocked: dict[int, str] | None = None,
    guard_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blocked = {} if blocked is None else dict(blocked)
    measurement_hash = _sha256_bytes(
        _json_bytes({str(rank): documents[rank] for rank in sorted(documents)})
    )
    calibration, boundaries = _fit_training(rows, freeze, measurement_hash)
    donor_regimes = {
        operation: tuple(
            regime
            for regime in calibration.regimes
            if regime.operation == operation and regime.ranks == 4
        )
        for operation in OPERATIONS
    }

    extrapolation_rows = []
    wide_fits: dict[tuple[str, int], tuple[CollectiveFloorRegime, ...] | None] = {}
    wide_fit_errors: dict[tuple[str, int], str] = {}
    for ranks in SCORED_RANKS:
        if ranks not in documents:
            continue
        for operation in OPERATIONS:
            try:
                wide_fits[operation, ranks] = _fit_wide_curve(
                    rows,
                    operation,
                    ranks,
                    boundaries[operation, 4],
                    freeze,
                    measurement_hash,
                )
            except (ValueError, StudyError) as error:
                wide_fits[operation, ranks] = None
                wide_fit_errors[operation, ranks] = str(error)

        scale = _scale(freeze, ranks)
        ceiling = _rank_config(freeze, ranks)["ceiling_bytes_per_second"]
        q = Fraction(ranks - 1, ranks)
        for row in (candidate for candidate in rows if candidate.ranks == ranks):
            donor = _regime(donor_regimes[row.operation], row.message_bytes)
            predicted = (donor.floor_ps + donor.slope_ps_per_byte * row.message_bytes) * scale
            predicted_ps = float(predicted)
            signed_error = (predicted_ps - row.latency_ps) / row.latency_ps
            measured_efficiency = float(
                q
                * row.message_bytes
                * 1_000_000_000_000
                / row.latency_ps
                / ceiling
            )
            predicted_efficiency = float(
                q
                * row.message_bytes
                * 1_000_000_000_000
                / predicted
                / ceiling
            )
            floor_fraction_difference = None
            wide = wide_fits[row.operation, ranks]
            if wide is not None:
                observed = _regime(wide, row.message_bytes)
                donor_floor_fraction = float(
                    donor.floor_ps
                    / (donor.floor_ps + donor.slope_ps_per_byte * row.message_bytes)
                )
                observed_floor_fraction = float(
                    observed.floor_ps
                    / (
                        observed.floor_ps
                        + observed.slope_ps_per_byte * row.message_bytes
                    )
                )
                floor_fraction_difference = abs(
                    donor_floor_fraction - observed_floor_fraction
                )
            extrapolation_rows.append(
                {
                    "operation": row.operation,
                    "ranks": ranks,
                    "operation_buffer_bytes": row.message_bytes,
                    "measured_ps": row.latency_ps,
                    "predicted_ps": predicted_ps,
                    "signed_relative_error": signed_error,
                    "absolute_relative_error": abs(signed_error),
                    "measured_normalized_efficiency": measured_efficiency,
                    "predicted_normalized_efficiency": predicted_efficiency,
                    "floor_fraction_absolute_difference": floor_fraction_difference,
                    "donor_regime_index": donor.regime_index,
                }
            )

    s1_rows = []
    s2_rows = []
    for ranks in SCORED_RANKS:
        if ranks not in documents:
            continue
        for operation in OPERATIONS:
            curve = [
                row
                for row in extrapolation_rows
                if row["ranks"] == ranks and row["operation"] == operation
            ]
            absolute_errors = [row["absolute_relative_error"] for row in curve]
            error_summary = _summary(absolute_errors)
            s1_held = (
                error_summary["median"]
                <= freeze["scored_families"]["S1"][
                    "per_curve_median_absolute_error_max"
                ]
                and error_summary["p95_nearest_rank"]
                <= freeze["scored_families"]["S1"][
                    "per_curve_p95_absolute_error_max"
                ]
            )
            s1_rows.append(
                {
                    "operation": operation,
                    "ranks": ranks,
                    **error_summary,
                    "held": s1_held,
                }
            )

            differences = [
                row["floor_fraction_absolute_difference"]
                for row in curve
                if row["floor_fraction_absolute_difference"] is not None
            ]
            if len(differences) != len(curve):
                s2_rows.append(
                    {
                        "operation": operation,
                        "ranks": ranks,
                        "held": False,
                        "fit_error": wide_fit_errors.get((operation, ranks)),
                    }
                )
            else:
                difference_summary = _summary(differences)
                s2_rows.append(
                    {
                        "operation": operation,
                        "ranks": ranks,
                        **difference_summary,
                        "held": (
                            difference_summary["median"]
                            <= freeze["scored_families"]["S2"][
                                "per_curve_median_max"
                            ]
                            and difference_summary["p95_nearest_rank"]
                            <= freeze["scored_families"]["S2"][
                                "per_curve_p95_max"
                            ]
                        ),
                    }
                )

    s3_rows = []
    s3_evaluated = all(rank in documents for rank in SCORED_RANKS)
    if s3_evaluated:
        neutral_low, neutral_high = freeze["scored_families"]["S3"][
            "neutral_signed_median_interval"
        ]
        for operation in OPERATIONS:
            by_rank = {}
            for ranks in SCORED_RANKS:
                curve = [
                    row
                    for row in extrapolation_rows
                    if row["ranks"] == ranks and row["operation"] == operation
                ]
                by_rank[ranks] = {
                    "signed_median": statistics.median(
                        row["signed_relative_error"] for row in curve
                    ),
                    "absolute_p95": nearest_rank(
                        [row["absolute_relative_error"] for row in curve], 0.95
                    ),
                }
            rank8 = by_rank[8]
            rank16 = by_rank[16]
            neutral = any(
                neutral_low <= by_rank[ranks]["signed_median"] <= neutral_high
                for ranks in SCORED_RANKS
            )
            sign_consistent = (
                neutral
                or math.copysign(1.0, rank8["signed_median"])
                == math.copysign(1.0, rank16["signed_median"])
            )
            growth = rank16["absolute_p95"] - rank8["absolute_p95"]
            s3_rows.append(
                {
                    "operation": operation,
                    "rank8": rank8,
                    "rank16": rank16,
                    "p95_growth_absolute_points": growth,
                    "sign_consistent_or_neutral": sign_consistent,
                    "held": (
                        sign_consistent
                        and growth
                        <= freeze["scored_families"]["S3"][
                            "p95_growth_max_absolute_points"
                        ]
                    ),
                }
            )

    s1_held = bool(s1_rows) and all(row["held"] for row in s1_rows)
    s2_held = bool(s2_rows) and all(row["held"] for row in s2_rows)
    s3_held = s3_evaluated and all(row["held"] for row in s3_rows)
    if blocked or not all(rank in documents for rank in SCORED_RANKS):
        verdict = "BLOCKED"
    elif s1_held and s2_held and s3_held:
        verdict = "FIT-SMALL-EXTRAPOLATE-WIDE HOLDS"
    else:
        verdict = "FIT-SMALL-EXTRAPOLATE-WIDE REFUTED"

    fit_rows = [
        {
            **regime.as_dict(),
            "fit_role": "training" if regime.ranks in TRAINING_RANKS else "descriptive",
        }
        for regime in calibration.regimes
    ]
    for (operation, ranks), regimes in sorted(wide_fits.items()):
        if regimes is None:
            continue
        fit_rows.extend(
            {
                **regime.as_dict(),
                "fit_role": "descriptive",
            }
            for regime in regimes
        )

    return {
        "schema": "simllm-collective-floor-extrapolation-result-v1",
        "task_id": "TRAF-81",
        "verdict": verdict,
        "freeze_sha256": _sha256_path(FREEZE_PATH),
        "measurement_set_sha256": measurement_hash,
        "measured_ranks": sorted(documents),
        "blocked_cells": [
            {"ranks": ranks, "reason": blocked[ranks]} for ranks in sorted(blocked)
        ],
        "guard_audit": guard_audit,
        "fit_boundaries": [
            {
                "operation": operation,
                "ranks": ranks,
                "lower_bounds_of_following_regimes": list(values),
            }
            for (operation, ranks), values in sorted(boundaries.items())
        ],
        "fits": fit_rows,
        "wide_fit_errors": [
            {"operation": operation, "ranks": ranks, "error": error}
            for (operation, ranks), error in sorted(wide_fit_errors.items())
        ],
        "extrapolation_rows": extrapolation_rows,
        "scores": {
            "S1": {"held": s1_held, "curves": s1_rows},
            "S2": {"held": s2_held, "curves": s2_rows},
            "S3": {
                "evaluated": s3_evaluated,
                "held": s3_held,
                "operations": s3_rows,
            },
            "S4": _score_dips(rows, freeze),
        },
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "record.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_csv(
        output_dir / "extrapolation.csv",
        result["extrapolation_rows"],
        (
            "operation",
            "ranks",
            "operation_buffer_bytes",
            "measured_ps",
            "predicted_ps",
            "signed_relative_error",
            "absolute_relative_error",
            "measured_normalized_efficiency",
            "predicted_normalized_efficiency",
            "floor_fraction_absolute_difference",
            "donor_regime_index",
        ),
    )
    _write_csv(
        output_dir / "fits.csv",
        result["fits"],
        (
            "operation",
            "ranks",
            "regime_index",
            "lower_bytes",
            "upper_bytes",
            "floor_ps",
            "slope_ps_per_byte",
            "effective_bandwidth_bytes_per_second",
            "fit_role",
        ),
    )


def _parse_blocked(values: Sequence[str]) -> dict[int, str]:
    blocked = {}
    for value in values:
        rank_text, separator, reason = value.partition("=")
        if not separator or not reason:
            raise StudyError("--blocked must be RANK=REASON")
        ranks = int(rank_text)
        if ranks not in SCORED_RANKS:
            raise StudyError("only scored ranks 8 and 16 may be blocked")
        blocked[ranks] = reason
    return blocked


def _parse_evidence(values: Sequence[str]) -> dict[int, Path]:
    evidence = {}
    for value in values:
        rank_text, separator, path_text = value.partition("=")
        if not separator or not path_text:
            raise StudyError("--evidence must be RANK=PATH")
        ranks = int(rank_text)
        if ranks not in (*TRAINING_RANKS, *SCORED_RANKS):
            raise StudyError(f"evidence rank {ranks} is outside the frozen cells")
        if ranks in evidence:
            raise StudyError(f"duplicate evidence directory for rank {ranks}")
        evidence[ranks] = Path(path_text)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", action="append", type=Path, default=[])
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--blocked", action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    freeze = load_freeze()
    documents: dict[int, dict[str, Any]] = {}
    measurement_paths: dict[int, Path] = {}
    rows: list[MeasurementRow] = []
    for path in args.measurement:
        document, loaded_rows = load_measurement(path, freeze)
        ranks = document["world"]
        if ranks in documents:
            raise StudyError(f"duplicate measurement document for rank {ranks}")
        documents[ranks] = document
        measurement_paths[ranks] = path
        rows.extend(loaded_rows)
    evidence_paths = _parse_evidence(args.evidence)
    blocked = _parse_blocked(args.blocked)
    overlap = sorted(set(documents) & set(blocked))
    if overlap:
        raise StudyError(f"rank cells cannot be measured and blocked: {overlap}")
    guard_audit = audit_evidence(evidence_paths, measurement_paths, freeze)
    result = score_measurements(
        documents,
        rows,
        freeze,
        blocked=blocked,
        guard_audit=guard_audit,
    )
    if args.check:
        expected = json.loads((STUDY_ROOT / "record.json").read_text(encoding="utf-8"))
        if result != expected:
            raise StudyError("tracked TRAF-81 record differs from fresh scoring")
        print("TRAF-81 tracked record matches fresh scoring")
        return 0
    if args.output_dir is None:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        write_outputs(result, args.output_dir)
        print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
