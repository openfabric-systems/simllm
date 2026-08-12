"""Measured CUDA-kernel capture and profile-table calibration records.

The production parser consumes the public ``cuda_gpu_trace`` CSV report from
Nsight Systems. Raw profiler databases remain external bulk evidence. The
tracked calibration artifact retains the immutable launch plan, all measured
durations, hashes and the train or held-out split needed to reproduce the
compact :class:`~simllm.compute.provider.ProfileTableProvider` table.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from simllm.compute.provider import (
    PS_PER_SECOND,
    ComputeProvider,
    GpuSpec,
    KernelSpec,
    ProfileTableProvenance,
    ProfileTableProvider,
)

COMPUTE_CALIBRATION_SCHEMA = "simllm-compute-calibration-v1"
TRAIN_SPLIT = "train"
HELD_OUT_SPLIT = "held-out"
_SPLITS = frozenset((TRAIN_SPLIT, HELD_OUT_SPLIT))
_SHA256 = re.compile(r"[0-9a-f]{64}")
_KERNEL_IDENTITY = re.compile(r"simllm_(?P<family>[a-z0-9_]+)_kernel<(?P<dtype>float|double)>")


def _require_nonblank(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def nearest_rank(values: Sequence[float | int], percentile: float) -> float:
    """Return the deterministic nearest-rank percentile of nonempty values."""

    if not values:
        raise ValueError("percentile values must not be empty")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def absolute_percentage_error(predicted: int, measured: int) -> float:
    """Return absolute percentage error, rejecting a zero observation."""

    if measured <= 0:
        raise ValueError("measured duration must be positive")
    return abs(predicted - measured) * 100.0 / measured


@dataclass(frozen=True)
class KernelLaunchMetadata:
    """Launch fields exported by the Nsight Systems CUDA GPU trace report."""

    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    registers_per_thread: int
    static_shared_memory_bytes: int
    dynamic_shared_memory_bytes: int
    device: str
    context_id: int
    stream_id: int

    def __post_init__(self) -> None:
        if len(self.grid) != 3 or any(value <= 0 for value in self.grid):
            raise ValueError("launch grid must contain three positive integers")
        if len(self.block) != 3 or any(value <= 0 for value in self.block):
            raise ValueError("launch block must contain three positive integers")
        for name in (
            "registers_per_thread",
            "static_shared_memory_bytes",
            "dynamic_shared_memory_bytes",
            "context_id",
            "stream_id",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        _require_nonblank("launch device", self.device)

    def to_json(self) -> dict[str, Any]:
        return {
            "grid": list(self.grid),
            "block": list(self.block),
            "registers_per_thread": self.registers_per_thread,
            "static_shared_memory_bytes": self.static_shared_memory_bytes,
            "dynamic_shared_memory_bytes": self.dynamic_shared_memory_bytes,
            "device": self.device,
            "context_id": self.context_id,
            "stream_id": self.stream_id,
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> KernelLaunchMetadata:
        return cls(
            grid=tuple(int(item) for item in value["grid"]),
            block=tuple(int(item) for item in value["block"]),
            registers_per_thread=int(value["registers_per_thread"]),
            static_shared_memory_bytes=int(value["static_shared_memory_bytes"]),
            dynamic_shared_memory_bytes=int(value["dynamic_shared_memory_bytes"]),
            device=str(value["device"]),
            context_id=int(value["context_id"]),
            stream_id=int(value["stream_id"]),
        )


@dataclass(frozen=True)
class CapturePlanCell:
    """One expected contiguous group of target launches in a profiler trace."""

    family: str
    dtype: str
    gpu_profile: str
    config: tuple[tuple[str, int], ...]
    split: str
    sample_count: int
    work_items: int
    source_flops: int
    compulsory_input_bytes: int
    total_bytes: int
    expected_grid_x: int
    expected_block_x: int

    def __post_init__(self) -> None:
        _require_nonblank("family", self.family)
        if self.dtype not in ("fp32", "fp64"):
            raise ValueError("dtype must be fp32 or fp64")
        _require_nonblank("gpu_profile", self.gpu_profile)
        if self.split not in _SPLITS:
            raise ValueError(f"split must be one of {sorted(_SPLITS)}")
        for name in (
            "sample_count",
            "work_items",
            "compulsory_input_bytes",
            "total_bytes",
            "expected_grid_x",
            "expected_block_x",
        ):
            _require_positive_int(name, getattr(self, name))
        if isinstance(self.source_flops, bool) or self.source_flops < 0:
            raise ValueError("source_flops must be a nonnegative integer")
        if self.total_bytes < self.compulsory_input_bytes:
            raise ValueError("total_bytes cannot be below compulsory input bytes")
        if not self.config:
            raise ValueError("config must contain at least one semantic axis")
        if any(
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for name, value in self.config
        ):
            raise ValueError("config axes must have names and positive integer values")


@dataclass(frozen=True)
class KernelCaptureCell:
    """Raw duration samples and immutable identity for one calibration cell."""

    family: str
    dtype: str
    gpu_profile: str
    config: tuple[tuple[str, int], ...]
    split: str
    kernel_name: str
    work_items: int
    source_flops: int
    compulsory_input_bytes: int
    total_bytes: int
    launch: KernelLaunchMetadata
    durations_ps: tuple[int, ...]

    def __post_init__(self) -> None:
        plan = CapturePlanCell(
            family=self.family,
            dtype=self.dtype,
            gpu_profile=self.gpu_profile,
            config=self.config,
            split=self.split,
            sample_count=len(self.durations_ps),
            work_items=self.work_items,
            source_flops=self.source_flops,
            compulsory_input_bytes=self.compulsory_input_bytes,
            total_bytes=self.total_bytes,
            expected_grid_x=self.launch.grid[0],
            expected_block_x=self.launch.block[0],
        )
        del plan
        _require_nonblank("kernel_name", self.kernel_name)
        if not self.durations_ps or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.durations_ps
        ):
            raise ValueError("durations_ps must contain positive integers")

    @property
    def median_duration_ps(self) -> int:
        value = statistics.median(self.durations_ps)
        if isinstance(value, float):
            return round(value)
        return int(value)

    @property
    def p05_duration_ps(self) -> int:
        return round(nearest_rank(self.durations_ps, 0.05))

    @property
    def p95_duration_ps(self) -> int:
        return round(nearest_rank(self.durations_ps, 0.95))

    @property
    def coefficient_of_variation(self) -> float:
        mean = statistics.fmean(self.durations_ps)
        return statistics.pstdev(self.durations_ps, mu=mean) / mean

    @property
    def p95_relative_deviation(self) -> float:
        median = self.median_duration_ps
        deviations = [abs(value - median) / median for value in self.durations_ps]
        return nearest_rank(deviations, 0.95)

    @property
    def profile_key(self) -> tuple[str, tuple[tuple[str, int], ...], str]:
        return self.family, self.config, self.gpu_profile

    def summary_json(self) -> dict[str, Any]:
        return {
            "count": len(self.durations_ps),
            "minimum_duration_ps": min(self.durations_ps),
            "p05_duration_ps": self.p05_duration_ps,
            "median_duration_ps": self.median_duration_ps,
            "p95_duration_ps": self.p95_duration_ps,
            "maximum_duration_ps": max(self.durations_ps),
            "coefficient_of_variation": self.coefficient_of_variation,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "dtype": self.dtype,
            "gpu_profile": self.gpu_profile,
            "config": [[name, value] for name, value in self.config],
            "split": self.split,
            "kernel_name": self.kernel_name,
            "work_items": self.work_items,
            "source_flops": self.source_flops,
            "compulsory_input_bytes": self.compulsory_input_bytes,
            "total_bytes": self.total_bytes,
            "launch": self.launch.to_json(),
            "durations_ps": list(self.durations_ps),
            "summary": self.summary_json(),
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> KernelCaptureCell:
        cell = cls(
            family=str(value["family"]),
            dtype=str(value["dtype"]),
            gpu_profile=str(value["gpu_profile"]),
            config=tuple((str(name), int(axis_value)) for name, axis_value in value["config"]),
            split=str(value["split"]),
            kernel_name=str(value["kernel_name"]),
            work_items=int(value["work_items"]),
            source_flops=int(value["source_flops"]),
            compulsory_input_bytes=int(value["compulsory_input_bytes"]),
            total_bytes=int(value["total_bytes"]),
            launch=KernelLaunchMetadata.from_json(value["launch"]),
            durations_ps=tuple(int(item) for item in value["durations_ps"]),
        )
        if value.get("summary") != cell.summary_json():
            raise ValueError(f"capture summary does not match raw samples for {cell.profile_key}")
        return cell


@dataclass(frozen=True)
class ComputeCalibrationProvenance:
    """Tool, device, clock, identity and policy provenance for one capture."""

    gpu_model: str
    gpu_uuid: str
    compute_capability: str
    driver_version: str
    cuda_version: str
    nsight_systems_version: str
    nsight_compute_version: str
    source_sha256: str
    binary_sha256: str
    static_sass_sha256: str
    capture_sha256: str
    creation_date: str
    warmup_policy: str
    cache_policy: str
    clock_policy: str
    core_clock_before_mhz: int
    core_clock_after_mhz: int
    memory_clock_before_mhz: int
    memory_clock_after_mhz: int
    performance_counter_status: str
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "gpu_model",
            "gpu_uuid",
            "compute_capability",
            "driver_version",
            "cuda_version",
            "nsight_systems_version",
            "nsight_compute_version",
            "creation_date",
            "warmup_policy",
            "cache_policy",
            "clock_policy",
            "performance_counter_status",
        ):
            _require_nonblank(name, getattr(self, name))
        for name in (
            "source_sha256",
            "binary_sha256",
            "static_sass_sha256",
            "capture_sha256",
        ):
            value = getattr(self, name)
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        for name in (
            "core_clock_before_mhz",
            "core_clock_after_mhz",
            "memory_clock_before_mhz",
            "memory_clock_after_mhz",
        ):
            _require_positive_int(name, getattr(self, name))
        if len(set(self.references)) != len(self.references):
            raise ValueError("calibration references must not contain duplicates")
        for reference in self.references:
            _require_nonblank("calibration reference", reference)

    def to_json(self) -> dict[str, Any]:
        return {
            name: list(value) if name == "references" else value
            for name, value in self.__dict__.items()
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> ComputeCalibrationProvenance:
        fields = dict(value)
        fields["references"] = tuple(fields.get("references", ()))
        return cls(**fields)


@dataclass(frozen=True)
class ComputeCalibrationArtifact:
    """Strict, JSON-serializable capture artifact with immutable split."""

    provenance: ComputeCalibrationProvenance
    cells: tuple[KernelCaptureCell, ...]

    def __post_init__(self) -> None:
        if not self.cells:
            raise ValueError("calibration artifact must contain capture cells")
        keys = [cell.profile_key for cell in self.cells]
        if len(keys) != len(set(keys)):
            raise ValueError("calibration artifact contains duplicate profile keys")
        if {cell.split for cell in self.cells} != _SPLITS:
            raise ValueError("calibration artifact must contain train and held-out cells")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": COMPUTE_CALIBRATION_SCHEMA,
            "provenance": self.provenance.to_json(),
            "cells": [cell.to_json() for cell in self.cells],
        }

    def to_json_bytes(self) -> bytes:
        return (json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n").encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json_bytes()).hexdigest()

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.write_bytes(self.to_json_bytes())
        return output

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> ComputeCalibrationArtifact:
        if value.get("schema") != COMPUTE_CALIBRATION_SCHEMA:
            raise ValueError(
                f"expected schema {COMPUTE_CALIBRATION_SCHEMA!r}, got {value.get('schema')!r}"
            )
        cells = tuple(KernelCaptureCell.from_json(item) for item in value["cells"])
        return cls(
            provenance=ComputeCalibrationProvenance.from_json(value["provenance"]),
            cells=cells,
        )

    @classmethod
    def load(cls, path: str | Path) -> ComputeCalibrationArtifact:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("calibration artifact root must be an object")
        return cls.from_json(value)


@dataclass(frozen=True)
class _TraceRow:
    family: str
    dtype: str
    kernel_name: str
    duration_ps: int
    launch: KernelLaunchMetadata


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _find_column(fieldnames: Sequence[str], *candidates: str) -> str:
    normalized = {_normalize_header(name): name for name in fieldnames}
    for candidate in candidates:
        key = _normalize_header(candidate)
        if key in normalized:
            return normalized[key]
        prefixed = [
            original
            for normalized_name, original in normalized.items()
            if normalized_name.startswith(key)
        ]
        if len(prefixed) == 1:
            return prefixed[0]
    raise ValueError(f"Nsight CSV is missing one of columns {candidates}; found {list(fieldnames)}")


def _parse_int(row: dict[str, str], column: str) -> int:
    value = row[column].strip()
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"Nsight CSV column {column!r} is not an integer: {value!r}") from error
    integral = parsed.to_integral_value()
    if parsed != integral:
        raise ValueError(f"Nsight CSV column {column!r} is not integral: {value!r}")
    return int(integral)


def _memory_bytes(row: dict[str, str], column: str) -> int:
    value = row[column].strip()
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(
            f"Nsight CSV memory column {column!r} is not numeric: {value!r}"
        ) from error
    lowered = column.lower()
    if "(gb)" in lowered:
        parsed *= 1_000_000_000
    elif "(mb)" in lowered:
        parsed *= 1_000_000
    elif "(kb)" in lowered:
        parsed *= 1_000
    elif "(byte)" not in lowered and "(b)" not in lowered and parsed != 0:
        raise ValueError(f"Nsight CSV memory column has unknown unit: {column!r}")
    integral = parsed.to_integral_value(rounding=ROUND_HALF_UP)
    if parsed != integral:
        raise ValueError(f"Nsight CSV memory value loses bytes after unit conversion: {value!r}")
    return int(integral)


def _duration_ps(row: dict[str, str], column: str) -> int:
    try:
        duration_ns = Decimal(row[column].strip())
    except InvalidOperation as error:
        raise ValueError(f"Nsight CSV duration is not numeric: {row[column]!r}") from error
    duration_ps = int((duration_ns * 1000).to_integral_value(rounding=ROUND_HALF_UP))
    if duration_ps <= 0:
        raise ValueError("Nsight CSV target duration must be positive")
    return duration_ps


def parse_nsight_cuda_gpu_trace_csv(
    path: str | Path,
    plan: Sequence[CapturePlanCell],
) -> tuple[KernelCaptureCell, ...]:
    """Parse ordered target launches from an Nsight ``cuda_gpu_trace`` CSV.

    Non-target rows such as cache flushes and CUDA memory operations are
    ignored. Target rows must match the immutable plan in order and count.
    """

    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("Nsight CSV has no header")
        name_col = _find_column(reader.fieldnames, "Name")
        duration_col = _find_column(reader.fieldnames, "Duration (ns)", "Duration")
        grid_cols = tuple(
            _find_column(reader.fieldnames, name) for name in ("GrdX", "GrdY", "GrdZ")
        )
        block_cols = tuple(
            _find_column(reader.fieldnames, name) for name in ("BlkX", "BlkY", "BlkZ")
        )
        registers_col = _find_column(reader.fieldnames, "Reg/Trd")
        static_col = _find_column(reader.fieldnames, "StcSMem (byte)", "StcSMem")
        dynamic_col = _find_column(reader.fieldnames, "DymSMem (byte)", "DymSMem")
        device_col = _find_column(reader.fieldnames, "Device")
        context_col = _find_column(reader.fieldnames, "Ctx")
        stream_col = _find_column(reader.fieldnames, "Strm")

        rows = []
        for row in reader:
            kernel_name = row[name_col]
            match = _KERNEL_IDENTITY.search(kernel_name)
            if match is None:
                continue
            dtype = "fp32" if match.group("dtype") == "float" else "fp64"
            rows.append(
                _TraceRow(
                    family=match.group("family"),
                    dtype=dtype,
                    kernel_name=kernel_name,
                    duration_ps=_duration_ps(row, duration_col),
                    launch=KernelLaunchMetadata(
                        grid=tuple(_parse_int(row, column) for column in grid_cols),
                        block=tuple(_parse_int(row, column) for column in block_cols),
                        registers_per_thread=_parse_int(row, registers_col),
                        static_shared_memory_bytes=_memory_bytes(row, static_col),
                        dynamic_shared_memory_bytes=_memory_bytes(row, dynamic_col),
                        device=row[device_col],
                        context_id=_parse_int(row, context_col),
                        stream_id=_parse_int(row, stream_col),
                    ),
                )
            )

    expected_count = sum(cell.sample_count for cell in plan)
    if len(rows) != expected_count:
        raise ValueError(f"Nsight trace has {len(rows)} target rows; expected {expected_count}")
    result = []
    offset = 0
    for expected in plan:
        group = rows[offset : offset + expected.sample_count]
        offset += expected.sample_count
        if any(row.family != expected.family for row in group):
            raise ValueError(f"target launch order drifted at {expected.family} {expected.config}")
        if any(row.dtype != expected.dtype for row in group):
            raise ValueError(f"target dtype order drifted at {expected.family} {expected.config}")
        launch = group[0].launch
        if any(row.launch != launch for row in group):
            raise ValueError(f"launch metadata changed within {expected.family} {expected.config}")
        if launch.grid != (expected.expected_grid_x, 1, 1):
            raise ValueError(
                f"grid mismatch for {expected.family} {expected.config}: {launch.grid}"
            )
        if launch.block != (expected.expected_block_x, 1, 1):
            raise ValueError(
                f"block mismatch for {expected.family} {expected.config}: {launch.block}"
            )
        kernel_names = {row.kernel_name for row in group}
        if len(kernel_names) != 1:
            raise ValueError(f"kernel identity changed within {expected.family} {expected.config}")
        result.append(
            KernelCaptureCell(
                family=expected.family,
                dtype=expected.dtype,
                gpu_profile=expected.gpu_profile,
                config=expected.config,
                split=expected.split,
                kernel_name=group[0].kernel_name,
                work_items=expected.work_items,
                source_flops=expected.source_flops,
                compulsory_input_bytes=expected.compulsory_input_bytes,
                total_bytes=expected.total_bytes,
                launch=launch,
                durations_ps=tuple(row.duration_ps for row in group),
            )
        )
    return tuple(result)


def _family_holdout_uncertainty(
    artifact: ComputeCalibrationArtifact,
    provisional: ProfileTableProvider,
) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for cell in artifact.cells:
        if cell.split != HELD_OUT_SPLIT:
            continue
        prediction = provisional.estimate(
            KernelSpec(cell.family, cell.source_flops, cell.total_bytes, cell.config),
            GpuSpec(cell.gpu_profile, 1.0, 1.0),
        )
        error = (
            absolute_percentage_error(
                prediction.duration_ps,
                cell.median_duration_ps,
            )
            / 100.0
        )
        grouped.setdefault((cell.family, cell.dtype), []).append(error)
    return {key: nearest_rank(values, 0.95) for key, values in grouped.items()}


def calibration_artifact_to_profile_table(
    artifact: ComputeCalibrationArtifact,
    *,
    enable_family_sum: bool = False,
) -> ProfileTableProvider:
    """Compile train medians and held-out uncertainty into the compact table."""

    train_cells = [cell for cell in artifact.cells if cell.split == TRAIN_SPLIT]
    provisional = ProfileTableProvider(
        {cell.profile_key: cell.median_duration_ps for cell in train_cells}
    )
    fit_uncertainty = _family_holdout_uncertainty(artifact, provisional)
    table = {
        cell.profile_key: (
            cell.median_duration_ps,
            max(
                ProfileTableProvider.EXACT_UNCERTAINTY,
                cell.p95_relative_deviation,
                fit_uncertainty[(cell.family, cell.dtype)],
            ),
        )
        for cell in train_cells
    }
    provenance = ProfileTableProvenance(
        source="capture",
        version=artifact.provenance.nsight_systems_version,
        gpu=artifact.provenance.gpu_model,
        created=artifact.provenance.creation_date,
        references=(
            f"{COMPUTE_CALIBRATION_SCHEMA}:sha256:{artifact.sha256}",
            *artifact.provenance.references,
        ),
    )
    return ProfileTableProvider(
        table,
        provenance=provenance,
        enable_family_sum=enable_family_sum,
    )


def physical_duration_bounds_ps(
    cell: KernelCaptureCell,
    *,
    dtype_peak_flops_per_second: int,
    memory_bandwidth_bytes_per_second: int,
    serial_operations_per_second: int,
    serial_memory_bytes_per_second: int,
) -> tuple[int, int]:
    """Return the frozen compulsory-traffic floor and serial-service ceiling."""

    for name, value in (
        ("dtype_peak_flops_per_second", dtype_peak_flops_per_second),
        ("memory_bandwidth_bytes_per_second", memory_bandwidth_bytes_per_second),
        ("serial_operations_per_second", serial_operations_per_second),
        ("serial_memory_bytes_per_second", serial_memory_bytes_per_second),
    ):
        _require_positive_int(name, value)
    compute_floor = cell.source_flops / dtype_peak_flops_per_second
    memory_floor = cell.compulsory_input_bytes / memory_bandwidth_bytes_per_second
    floor_ps = math.ceil(max(compute_floor, memory_floor) * PS_PER_SECOND)
    ceiling_seconds = (
        cell.source_flops / serial_operations_per_second
        + cell.total_bytes / serial_memory_bytes_per_second
    )
    ceiling_ps = math.ceil(ceiling_seconds * PS_PER_SECOND)
    if ceiling_ps < floor_ps:
        raise ValueError("physical ceiling is below the physical floor")
    return floor_ps, ceiling_ps


def held_out_errors(
    artifact: ComputeCalibrationArtifact,
    calibrated: ProfileTableProvider,
    rooflines: dict[str, ComputeProvider],
    gpus: dict[str, GpuSpec],
) -> list[dict[str, Any]]:
    """Return raw per-cell calibrated and comparator errors.

    ``rooflines`` is keyed by exact GPU profile and values need only implement
    the repository ``ComputeProvider`` estimate method. The broad annotation
    avoids a second provider protocol beside ``ComputeProvider``.
    """

    rows = []
    for cell in artifact.cells:
        if cell.split != HELD_OUT_SPLIT:
            continue
        kernel = KernelSpec(
            cell.family,
            float(cell.source_flops),
            float(cell.total_bytes),
            cell.config,
        )
        gpu = gpus[cell.gpu_profile]
        calibrated_ps = calibrated.estimate(kernel, gpu).duration_ps
        roofline_ps = rooflines[cell.gpu_profile].estimate(kernel, gpu).duration_ps
        measured_ps = cell.median_duration_ps
        rows.append(
            {
                "family": cell.family,
                "dtype": cell.dtype,
                "config": [[name, value] for name, value in cell.config],
                "measured_median_ps": measured_ps,
                "calibrated_prediction_ps": calibrated_ps,
                "roofline_prediction_ps": roofline_ps,
                "calibrated_absolute_percentage_error": absolute_percentage_error(
                    calibrated_ps, measured_ps
                ),
                "roofline_absolute_percentage_error": absolute_percentage_error(
                    roofline_ps, measured_ps
                ),
            }
        )
    return rows


def sha256_file(path: str | Path) -> str:
    """Return a lowercase SHA-256 digest without loading a bulk file at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
