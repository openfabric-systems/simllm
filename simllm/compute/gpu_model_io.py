"""Strict JSON artifact boundary for trace-calibrated GPU service replay.

``simllm-gpu-model-artifact-v2`` keeps the evidence needed to audit one
service-model replay.  It is intentionally richer than the compact profile
table consumed by an online step loop: architecture and calibration identity,
kernel and trace hashes, the semantic catalog key, replay counters, immutable
calibration split, uncertainty, and optional silicon samples all survive a
round trip.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from simllm.compute.gpu_model import (
    GPU_MODEL_IMPLEMENTATION,
    CopyDirection,
    CopyDirectionProfile,
    CopyEngineProfile,
    CopyEngineServiceModel,
    CopyServiceEstimate,
    CopyTransfer,
    CtaTrace,
    GpuArchitectureProfile,
    GpuCalibrationProfile,
    GpuKernelEstimate,
    GpuModelProvenance,
    KernelLaunch,
    MemoryHierarchyProfile,
    MemorySpace,
    NvlinkProfile,
    PipelineKind,
    PipelineProfile,
    SassInstruction,
    SassWarpTrace,
    SmSchedulerModel,
    WarpSchedulerPolicy,
)
from simllm.compute.provider import ProfileTableProvenance, ProfileTableProvider
from simllm.core._wire import (
    _boolean,
    _enum_value,
    _fields,
    _integer,
    _number,
    _optional_string,
    _string,
)

GPU_MODEL_ARTIFACT_SCHEMA = "simllm-gpu-model-artifact-v2"
_LEGACY_GPU_MODEL_ARTIFACT_SCHEMA = "simllm-gpu-model-artifact-v1"
_LEGACY_GPU_MODEL_IMPLEMENTATION = "simllm-isolated-gpu-service-v1"

_SHA256_RE = re.compile(r"(?:sha256:)?[0-9a-f]{64}\Z")


def _object(value: object, path: str) -> Mapping[str, Any]:
    """Accept the artifact API's historical read-only mapping views."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path}: object keys must be strings")
    return value


def _array(value: object, path: str) -> Sequence[Any]:
    """Accept tuple views without widening the strict core wire codecs."""

    if not isinstance(value, list | tuple):
        raise TypeError(f"{path}: expected an array")
    return value


def _freeze_pairs(values: Sequence[Sequence[Any]]) -> tuple[tuple[Any, Any], ...]:
    frozen = tuple((pair[0], pair[1]) for pair in values)
    return tuple(sorted(frozen, key=lambda pair: str(pair[0])))


class CalibrationSplit(str, Enum):
    """Immutable role of one replay or measurement in a calibration."""

    TRAIN = "train"
    HELD_OUT = "held-out"
    UNASSIGNED = "unassigned"


@dataclass(frozen=True, kw_only=True)
class GpuCaptureEnvironment:
    """Pinned software, hardware, and measurement envelope for one capture."""

    capture_id: str
    model_id: str
    gpu_name: str
    gpu_uuid: str
    framework: str
    framework_version: str
    driver_version: str
    cuda_version: str
    execution_mode: str
    clock_policy: str
    observed_core_clock_hz: int
    observed_memory_clock_hz: int
    warmup_iterations: int
    measured_iterations: int
    tools: tuple[tuple[str, str], ...]
    libraries: tuple[tuple[str, str], ...]
    created: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools", _freeze_pairs(self.tools))
        object.__setattr__(self, "libraries", _freeze_pairs(self.libraries))


@dataclass(frozen=True, kw_only=True)
class GpuKernelCatalogRecord:
    """One semantic lookup key bound to exact binary, function, and trace."""

    kernel: str
    config: tuple[tuple[str, int], ...]
    gpu: str
    binary_sha256: str
    function_sha256: str
    trace_sha256: str
    launch: KernelLaunch
    capture_id: str | None = None
    launch_order: int | None = None
    stream_id: str | None = None
    attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", _freeze_pairs(self.config))
        object.__setattr__(self, "attributes", _freeze_pairs(self.attributes))

    @property
    def key(self) -> tuple[str, tuple[tuple[str, int], ...], str]:
        return self.kernel, self.config, self.gpu


@dataclass(frozen=True, kw_only=True)
class GpuReplayRecord:
    """Deterministic result of replaying one catalog entry on one profile."""

    replay_id: str
    implementation_id: str
    architecture_profile_id: str
    calibration_id: str
    calibration_split: CalibrationSplit
    relative_uncertainty: float
    estimate: GpuKernelEstimate

    def __post_init__(self) -> None:
        if not isinstance(self.calibration_split, CalibrationSplit):
            raise TypeError("calibration_split must be a CalibrationSplit")


@dataclass(frozen=True, kw_only=True)
class GpuCopyReplayRecord:
    """Deterministic service result for one explicitly selected copy engine."""

    replay_id: str
    architecture_profile_id: str
    calibration_id: str
    calibration_split: CalibrationSplit
    relative_uncertainty: float
    transfer: CopyTransfer
    estimate: CopyServiceEstimate
    capture_id: str | None = None
    launch_order: int | None = None
    stream_id: str | None = None
    attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", _freeze_pairs(self.attributes))
        if not isinstance(self.calibration_split, CalibrationSplit):
            raise TypeError("calibration_split must be a CalibrationSplit")


@dataclass(frozen=True, kw_only=True)
class GpuMeasurementRecord:
    """Optional silicon samples correlated to exactly one replay."""

    measurement_id: str
    replay_id: str
    capture_id: str
    capture_sha256: str
    duration_samples_ps: tuple[int, ...]
    mean_duration_ps: float
    p50_duration_ps: float
    coefficient_of_variation: float
    calibration_split: CalibrationSplit
    residual_fraction: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration_samples_ps", tuple(self.duration_samples_ps))
        if not isinstance(self.calibration_split, CalibrationSplit):
            raise TypeError("calibration_split must be a CalibrationSplit")


@dataclass(frozen=True, kw_only=True)
class GpuModelArtifact:
    """Auditable architecture, catalog, replay, and optional measurement set."""

    artifact_id: str
    architecture: GpuArchitectureProfile
    captures: tuple[GpuCaptureEnvironment, ...]
    catalog: tuple[GpuKernelCatalogRecord, ...]
    replays: tuple[GpuReplayRecord, ...]
    copy_replays: tuple[GpuCopyReplayRecord, ...]
    measurements: tuple[GpuMeasurementRecord, ...] = ()

    def __post_init__(self) -> None:
        for name in ("captures", "catalog", "replays", "copy_replays", "measurements"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def to_json(self) -> dict[str, Any]:
        return gpu_model_artifact_to_json(self)

    def save(self, path: str | Path) -> Path:
        return save_gpu_model_artifact(self, path)

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> GpuModelArtifact:
        return gpu_model_artifact_from_json(payload)

    @classmethod
    def load(cls, path: str | Path) -> GpuModelArtifact:
        return load_gpu_model_artifact(path)


def _sha256(value: object, path: str) -> str:
    text = _string(value, path)
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{path}: expected 64 lowercase hex digits, optionally prefixed sha256:")
    return text


def _sha256_digest(value: str) -> str | None:
    match = _SHA256_RE.fullmatch(value)
    if match is None:
        return None
    return value.removeprefix("sha256:")


def _unique(values: Sequence[Any], path: str) -> None:
    try:
        unique_count = len(set(values))
    except TypeError as exc:
        raise ValueError(f"{path}: values must be hashable") from exc
    if unique_count != len(values):
        raise ValueError(f"{path}: duplicate values are not allowed")


def _text_tuple(value: object, path: str, *, nonempty: bool = False) -> tuple[str, ...]:
    values = tuple(
        _string(item, f"{path}[{index}]") for index, item in enumerate(_array(value, path))
    )
    if nonempty and not values:
        raise ValueError(f"{path}: must not be empty")
    _unique(values, path)
    return values


def _text_pairs(value: object, path: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for index, raw_pair in enumerate(_array(value, path)):
        pair_path = f"{path}[{index}]"
        pair = _array(raw_pair, pair_path)
        if len(pair) != 2:
            raise ValueError(f"{pair_path}: expected [name, value]")
        pairs.append((_string(pair[0], f"{pair_path}[0]"), _string(pair[1], f"{pair_path}[1]")))
    _unique(tuple(name for name, _ in pairs), f"{path} names")
    return tuple(pairs)


def _optional_integer(value: object, path: str, *, minimum: int = 0) -> int | None:
    """Nonnegative-by-default optional integer, the artifact's convention."""

    return None if value is None else _integer(value, path, minimum=minimum)


def _provenance_to_json(value: GpuModelProvenance) -> dict[str, Any]:
    return {
        "source": value.source,
        "version": value.version,
        "gpu": value.gpu,
        "created": value.created,
        "references": list(value.references),
    }


def _provenance_from_json(value: object, path: str) -> GpuModelProvenance:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={"source", "version", "gpu", "created", "references"},
    )
    return GpuModelProvenance(
        source=_string(payload["source"], f"{path}.source"),
        version=_string(payload["version"], f"{path}.version"),
        gpu=_string(payload["gpu"], f"{path}.gpu"),
        created=_string(payload["created"], f"{path}.created"),
        references=_text_tuple(payload["references"], f"{path}.references"),
    )


def _capture_to_json(value: GpuCaptureEnvironment) -> dict[str, Any]:
    return {
        "capture_id": value.capture_id,
        "model_id": value.model_id,
        "gpu_name": value.gpu_name,
        "gpu_uuid": value.gpu_uuid,
        "framework": value.framework,
        "framework_version": value.framework_version,
        "driver_version": value.driver_version,
        "cuda_version": value.cuda_version,
        "execution_mode": value.execution_mode,
        "clock_policy": value.clock_policy,
        "observed_core_clock_hz": value.observed_core_clock_hz,
        "observed_memory_clock_hz": value.observed_memory_clock_hz,
        "warmup_iterations": value.warmup_iterations,
        "measured_iterations": value.measured_iterations,
        "tools": [list(item) for item in value.tools],
        "libraries": [list(item) for item in value.libraries],
        "created": value.created,
    }


def _capture_from_json(value: object, path: str) -> GpuCaptureEnvironment:
    payload = _object(value, path)
    text_fields = {
        "capture_id",
        "model_id",
        "gpu_name",
        "gpu_uuid",
        "framework",
        "framework_version",
        "driver_version",
        "cuda_version",
        "execution_mode",
        "clock_policy",
        "created",
    }
    _fields(
        payload,
        path,
        required=text_fields
        | {
            "observed_core_clock_hz",
            "observed_memory_clock_hz",
            "warmup_iterations",
            "measured_iterations",
            "tools",
            "libraries",
        },
    )
    texts = {name: _string(payload[name], f"{path}.{name}") for name in text_fields}
    return GpuCaptureEnvironment(
        **texts,
        observed_core_clock_hz=_integer(
            payload["observed_core_clock_hz"],
            f"{path}.observed_core_clock_hz",
            minimum=1,
        ),
        observed_memory_clock_hz=_integer(
            payload["observed_memory_clock_hz"],
            f"{path}.observed_memory_clock_hz",
            minimum=1,
        ),
        warmup_iterations=_integer(
            payload["warmup_iterations"], f"{path}.warmup_iterations", minimum=0
        ),
        measured_iterations=_integer(
            payload["measured_iterations"], f"{path}.measured_iterations", minimum=1
        ),
        tools=_text_pairs(payload["tools"], f"{path}.tools"),
        libraries=_text_pairs(payload["libraries"], f"{path}.libraries"),
    )


def _calibration_to_json(value: GpuCalibrationProfile) -> dict[str, Any]:
    return {
        "calibration_id": value.calibration_id,
        "target_architecture_profile_id": value.target_architecture_profile_id,
        "provenance": _provenance_to_json(value.provenance),
        "core_clock_hz": value.core_clock_hz,
        "target_memory_clock_hz": value.target_memory_clock_hz,
        "pipelines": [_pipeline_to_json(item) for item in value.pipelines],
        "memory": _memory_to_json(value.memory),
        "nvlink": _nvlink_to_json(value.nvlink),
        "copy_engines": [_copy_engine_to_json(item) for item in value.copy_engines],
        "warp_scheduler_policy": value.warp_scheduler_policy.value,
        "relative_uncertainty": float(value.relative_uncertainty),
    }


def _nvlink_to_json(value: NvlinkProfile | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "latency_cycles": value.latency_cycles,
        "bandwidth_bytes_per_cycle": float(value.bandwidth_bytes_per_cycle),
    }


def _nvlink_from_json(value: object, path: str) -> NvlinkProfile | None:
    if value is None:
        return None
    payload = _object(value, path)
    _fields(payload, path, required={"latency_cycles", "bandwidth_bytes_per_cycle"})
    return NvlinkProfile(
        latency_cycles=_integer(
            payload["latency_cycles"], f"{path}.latency_cycles", minimum=0
        ),
        bandwidth_bytes_per_cycle=_number(
            payload["bandwidth_bytes_per_cycle"],
            f"{path}.bandwidth_bytes_per_cycle",
            minimum=0.0,
        ),
    )


def _calibration_from_json(value: object, path: str) -> GpuCalibrationProfile:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={
            "calibration_id",
            "target_architecture_profile_id",
            "provenance",
            "core_clock_hz",
            "target_memory_clock_hz",
            "pipelines",
            "memory",
            "nvlink",
            "copy_engines",
            "warp_scheduler_policy",
            "relative_uncertainty",
        },
    )
    pipelines = tuple(
        _pipeline_from_json(item, f"{path}.pipelines[{index}]")
        for index, item in enumerate(_array(payload["pipelines"], f"{path}.pipelines"))
    )
    copy_engines = tuple(
        _copy_engine_from_json(item, f"{path}.copy_engines[{index}]")
        for index, item in enumerate(_array(payload["copy_engines"], f"{path}.copy_engines"))
    )
    return GpuCalibrationProfile(
        calibration_id=_string(payload["calibration_id"], f"{path}.calibration_id"),
        target_architecture_profile_id=_string(
            payload["target_architecture_profile_id"],
            f"{path}.target_architecture_profile_id",
        ),
        provenance=_provenance_from_json(payload["provenance"], f"{path}.provenance"),
        core_clock_hz=_integer(payload["core_clock_hz"], f"{path}.core_clock_hz", minimum=1),
        target_memory_clock_hz=_optional_integer(
            payload["target_memory_clock_hz"],
            f"{path}.target_memory_clock_hz",
            minimum=1,
        ),
        pipelines=pipelines,
        memory=_memory_from_json(payload["memory"], f"{path}.memory"),
        nvlink=_nvlink_from_json(payload["nvlink"], f"{path}.nvlink"),
        copy_engines=copy_engines,
        warp_scheduler_policy=_enum_value(
            WarpSchedulerPolicy,
            payload["warp_scheduler_policy"],
            f"{path}.warp_scheduler_policy",
        ),
        relative_uncertainty=_number(
            payload["relative_uncertainty"],
            f"{path}.relative_uncertainty",
            minimum=0.0,
        ),
    )


def _pipeline_to_json(value: PipelineProfile) -> dict[str, Any]:
    return {
        "kind": value.kind.value,
        "opcodes": list(value.opcodes),
        "latency_cycles": value.latency_cycles,
        "issue_width_per_sm": value.issue_width_per_sm,
        "initiation_interval_cycles": value.initiation_interval_cycles,
        "opcode_latencies": [[opcode, latency] for opcode, latency in value.opcode_latencies],
    }


def _pipeline_from_json(value: object, path: str) -> PipelineProfile:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={
            "kind",
            "opcodes",
            "latency_cycles",
            "issue_width_per_sm",
            "initiation_interval_cycles",
            "opcode_latencies",
        },
    )
    opcode_latencies: list[tuple[str, int]] = []
    for index, raw_entry in enumerate(
        _array(payload["opcode_latencies"], f"{path}.opcode_latencies")
    ):
        entry_path = f"{path}.opcode_latencies[{index}]"
        entry = _array(raw_entry, entry_path)
        if len(entry) != 2:
            raise ValueError(f"{entry_path}: expected [opcode, latency_cycles]")
        opcode_latencies.append(
            (
                _string(entry[0], f"{entry_path}[0]"),
                _integer(entry[1], f"{entry_path}[1]", minimum=1),
            )
        )
    _unique(tuple(opcode for opcode, _ in opcode_latencies), f"{path}.opcode_latencies")
    return PipelineProfile(
        kind=_enum_value(PipelineKind, payload["kind"], f"{path}.kind"),
        opcodes=_text_tuple(payload["opcodes"], f"{path}.opcodes", nonempty=True),
        latency_cycles=_integer(payload["latency_cycles"], f"{path}.latency_cycles", minimum=1),
        issue_width_per_sm=_integer(
            payload["issue_width_per_sm"], f"{path}.issue_width_per_sm", minimum=1
        ),
        initiation_interval_cycles=_integer(
            payload["initiation_interval_cycles"],
            f"{path}.initiation_interval_cycles",
            minimum=1,
        ),
        opcode_latencies=tuple(opcode_latencies),
    )


def _memory_to_json(value: MemoryHierarchyProfile) -> dict[str, Any]:
    return {
        "hbm_latency_cycles": value.hbm_latency_cycles,
        "hbm_bandwidth_bytes_per_cycle": float(value.hbm_bandwidth_bytes_per_cycle),
        "l2_latency_cycles": value.l2_latency_cycles,
        "l1_latency_cycles": value.l1_latency_cycles,
        "shared_latency_cycles": value.shared_latency_cycles,
    }


def _memory_from_json(value: object, path: str) -> MemoryHierarchyProfile:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={
            "hbm_latency_cycles",
            "hbm_bandwidth_bytes_per_cycle",
            "l2_latency_cycles",
            "l1_latency_cycles",
            "shared_latency_cycles",
        },
    )
    return MemoryHierarchyProfile(
        hbm_latency_cycles=_integer(
            payload["hbm_latency_cycles"], f"{path}.hbm_latency_cycles", minimum=0
        ),
        hbm_bandwidth_bytes_per_cycle=_number(
            payload["hbm_bandwidth_bytes_per_cycle"],
            f"{path}.hbm_bandwidth_bytes_per_cycle",
            minimum=0.0,
        ),
        l2_latency_cycles=_integer(
            payload["l2_latency_cycles"], f"{path}.l2_latency_cycles", minimum=0
        ),
        l1_latency_cycles=_integer(
            payload["l1_latency_cycles"], f"{path}.l1_latency_cycles", minimum=0
        ),
        shared_latency_cycles=_integer(
            payload["shared_latency_cycles"], f"{path}.shared_latency_cycles", minimum=0
        ),
    )


def _copy_direction_profile_to_json(value: CopyDirectionProfile) -> dict[str, Any]:
    return {
        "direction": value.direction.value,
        "setup_cycles": value.setup_cycles,
        "bandwidth_bytes_per_cycle": float(value.bandwidth_bytes_per_cycle),
    }


def _copy_direction_profile_from_json(value: object, path: str) -> CopyDirectionProfile:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={"direction", "setup_cycles", "bandwidth_bytes_per_cycle"},
    )
    bandwidth = _number(
        payload["bandwidth_bytes_per_cycle"],
        f"{path}.bandwidth_bytes_per_cycle",
        minimum=0.0,
    )
    if bandwidth == 0.0:
        raise ValueError(f"{path}.bandwidth_bytes_per_cycle: must be positive")
    return CopyDirectionProfile(
        direction=_enum_value(CopyDirection, payload["direction"], f"{path}.direction"),
        setup_cycles=_integer(payload["setup_cycles"], f"{path}.setup_cycles", minimum=0),
        bandwidth_bytes_per_cycle=bandwidth,
    )


def _copy_engine_to_json(value: CopyEngineProfile) -> dict[str, Any]:
    return {
        "engine_id": value.engine_id,
        "clock_hz": value.clock_hz,
        "direction_profiles": [
            _copy_direction_profile_to_json(profile) for profile in value.direction_profiles
        ],
    }


def _copy_engine_from_json(value: object, path: str) -> CopyEngineProfile:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={
            "engine_id",
            "clock_hz",
            "direction_profiles",
        },
    )
    direction_profiles = tuple(
        _copy_direction_profile_from_json(item, f"{path}.direction_profiles[{index}]")
        for index, item in enumerate(
            _array(payload["direction_profiles"], f"{path}.direction_profiles")
        )
    )
    return CopyEngineProfile(
        engine_id=_string(payload["engine_id"], f"{path}.engine_id"),
        clock_hz=_integer(payload["clock_hz"], f"{path}.clock_hz", minimum=1),
        direction_profiles=direction_profiles,
    )


def _architecture_to_json(value: GpuArchitectureProfile) -> dict[str, Any]:
    return {
        "profile_id": value.profile_id,
        "gpu_name": value.gpu_name,
        "sm_count": value.sm_count,
        "warp_size": value.warp_size,
        "scheduler_count_per_sm": value.scheduler_count_per_sm,
        "dispatch_width_per_scheduler": value.dispatch_width_per_scheduler,
        "max_blocks_per_sm": value.max_blocks_per_sm,
        "max_warps_per_sm": value.max_warps_per_sm,
        "max_threads_per_sm": value.max_threads_per_sm,
        "max_threads_per_block": value.max_threads_per_block,
        "registers_per_sm": value.registers_per_sm,
        "max_registers_per_thread": value.max_registers_per_thread,
        "register_allocation_granularity_per_warp": (
            value.register_allocation_granularity_per_warp
        ),
        "shared_memory_per_sm": value.shared_memory_per_sm,
        "max_static_shared_memory_per_block": value.max_static_shared_memory_per_block,
        "max_shared_memory_per_block": value.max_shared_memory_per_block,
        "shared_memory_allocation_granularity": value.shared_memory_allocation_granularity,
        "calibration": _calibration_to_json(value.calibration),
        "aliases": list(value.aliases),
    }


def _architecture_from_json(value: object, path: str) -> GpuArchitectureProfile:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={
            "profile_id",
            "gpu_name",
            "sm_count",
            "warp_size",
            "scheduler_count_per_sm",
            "dispatch_width_per_scheduler",
            "max_blocks_per_sm",
            "max_warps_per_sm",
            "max_threads_per_sm",
            "max_threads_per_block",
            "registers_per_sm",
            "max_registers_per_thread",
            "register_allocation_granularity_per_warp",
            "shared_memory_per_sm",
            "max_static_shared_memory_per_block",
            "max_shared_memory_per_block",
            "shared_memory_allocation_granularity",
            "calibration",
            "aliases",
        },
    )
    return GpuArchitectureProfile(
        profile_id=_string(payload["profile_id"], f"{path}.profile_id"),
        gpu_name=_string(payload["gpu_name"], f"{path}.gpu_name"),
        sm_count=_integer(payload["sm_count"], f"{path}.sm_count", minimum=1),
        warp_size=_integer(payload["warp_size"], f"{path}.warp_size", minimum=1),
        scheduler_count_per_sm=_integer(
            payload["scheduler_count_per_sm"],
            f"{path}.scheduler_count_per_sm",
            minimum=1,
        ),
        dispatch_width_per_scheduler=_integer(
            payload["dispatch_width_per_scheduler"],
            f"{path}.dispatch_width_per_scheduler",
            minimum=1,
        ),
        max_blocks_per_sm=_integer(
            payload["max_blocks_per_sm"], f"{path}.max_blocks_per_sm", minimum=1
        ),
        max_warps_per_sm=_integer(
            payload["max_warps_per_sm"], f"{path}.max_warps_per_sm", minimum=1
        ),
        max_threads_per_sm=_integer(
            payload["max_threads_per_sm"], f"{path}.max_threads_per_sm", minimum=1
        ),
        max_threads_per_block=_integer(
            payload["max_threads_per_block"],
            f"{path}.max_threads_per_block",
            minimum=1,
        ),
        registers_per_sm=_integer(
            payload["registers_per_sm"], f"{path}.registers_per_sm", minimum=1
        ),
        max_registers_per_thread=_integer(
            payload["max_registers_per_thread"],
            f"{path}.max_registers_per_thread",
            minimum=1,
        ),
        register_allocation_granularity_per_warp=_integer(
            payload["register_allocation_granularity_per_warp"],
            f"{path}.register_allocation_granularity_per_warp",
            minimum=1,
        ),
        shared_memory_per_sm=_integer(
            payload["shared_memory_per_sm"], f"{path}.shared_memory_per_sm", minimum=1
        ),
        max_static_shared_memory_per_block=_integer(
            payload["max_static_shared_memory_per_block"],
            f"{path}.max_static_shared_memory_per_block",
            minimum=1,
        ),
        max_shared_memory_per_block=_integer(
            payload["max_shared_memory_per_block"],
            f"{path}.max_shared_memory_per_block",
            minimum=1,
        ),
        shared_memory_allocation_granularity=_integer(
            payload["shared_memory_allocation_granularity"],
            f"{path}.shared_memory_allocation_granularity",
            minimum=1,
        ),
        calibration=_calibration_from_json(payload["calibration"], f"{path}.calibration"),
        aliases=_text_tuple(payload["aliases"], f"{path}.aliases"),
    )


def _instruction_to_json(value: SassInstruction) -> dict[str, Any]:
    return {
        "opcode": value.opcode,
        "pipeline": value.pipeline.value,
        "repeat": value.repeat,
        "dependent": value.dependent,
        "dependency_indices": list(value.dependency_indices),
        "source_registers": list(value.source_registers),
        "destination_registers": list(value.destination_registers),
        "memory_space": None if value.memory_space is None else value.memory_space.value,
        "requested_bytes": value.requested_bytes,
        "transacted_bytes": value.transacted_bytes,
        "barrier": value.barrier,
    }


def _instruction_from_json(
    value: object,
    path: str,
    *,
    dynamic_instruction_offset: int,
) -> SassInstruction:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={
            "opcode",
            "pipeline",
            "repeat",
            "dependent",
            "dependency_indices",
            "source_registers",
            "destination_registers",
            "memory_space",
            "requested_bytes",
            "transacted_bytes",
            "barrier",
        },
    )
    dependencies = tuple(
        _integer(item, f"{path}.dependency_indices[{index}]", minimum=0)
        for index, item in enumerate(
            _array(payload["dependency_indices"], f"{path}.dependency_indices")
        )
    )
    _unique(dependencies, f"{path}.dependency_indices")
    if any(index >= dynamic_instruction_offset for index in dependencies):
        raise ValueError(f"{path}.dependency_indices: dependencies must precede instruction")
    memory_space = payload["memory_space"]
    parsed_memory_space = (
        None if memory_space is None else _enum_value(MemorySpace, memory_space, f"{path}.memory_space")
    )
    requested_bytes = _integer(payload["requested_bytes"], f"{path}.requested_bytes", minimum=0)
    transacted_bytes = _integer(payload["transacted_bytes"], f"{path}.transacted_bytes", minimum=0)
    if parsed_memory_space is None and (requested_bytes != 0 or transacted_bytes != 0):
        raise ValueError(f"{path}: non-memory instruction must carry zero byte counts")
    return SassInstruction(
        opcode=_string(payload["opcode"], f"{path}.opcode"),
        pipeline=_enum_value(PipelineKind, payload["pipeline"], f"{path}.pipeline"),
        repeat=_integer(payload["repeat"], f"{path}.repeat", minimum=1),
        dependent=_boolean(payload["dependent"], f"{path}.dependent"),
        dependency_indices=dependencies,
        source_registers=_text_tuple(payload["source_registers"], f"{path}.source_registers"),
        destination_registers=_text_tuple(
            payload["destination_registers"], f"{path}.destination_registers"
        ),
        memory_space=parsed_memory_space,
        requested_bytes=requested_bytes,
        transacted_bytes=transacted_bytes,
        barrier=_boolean(payload["barrier"], f"{path}.barrier"),
    )


def _warp_to_json(value: SassWarpTrace) -> dict[str, Any]:
    return {
        "warp_id": value.warp_id,
        "instructions": [_instruction_to_json(item) for item in value.instructions],
    }


def _warp_from_json(value: object, path: str) -> SassWarpTrace:
    payload = _object(value, path)
    _fields(payload, path, required={"warp_id", "instructions"})
    instructions: list[SassInstruction] = []
    dynamic_offset = 0
    for index, item in enumerate(_array(payload["instructions"], f"{path}.instructions")):
        instruction = _instruction_from_json(
            item,
            f"{path}.instructions[{index}]",
            dynamic_instruction_offset=dynamic_offset,
        )
        instructions.append(instruction)
        dynamic_offset += instruction.repeat
    if not instructions:
        raise ValueError(f"{path}.instructions: must not be empty")
    return SassWarpTrace(
        warp_id=_integer(payload["warp_id"], f"{path}.warp_id", minimum=0),
        instructions=tuple(instructions),
    )


def _cta_trace_to_json(value: CtaTrace) -> dict[str, Any]:
    return {
        "trace_class_id": value.trace_class_id,
        "block_ids": list(value.block_ids),
        "warp_traces": [_warp_to_json(item) for item in value.warp_traces],
    }


def _cta_trace_from_json(value: object, path: str) -> CtaTrace:
    payload = _object(value, path)
    _fields(payload, path, required={"trace_class_id", "block_ids", "warp_traces"})
    block_ids = tuple(
        _integer(item, f"{path}.block_ids[{index}]", minimum=0)
        for index, item in enumerate(_array(payload["block_ids"], f"{path}.block_ids"))
    )
    warps = tuple(
        _warp_from_json(item, f"{path}.warp_traces[{index}]")
        for index, item in enumerate(_array(payload["warp_traces"], f"{path}.warp_traces"))
    )
    return CtaTrace(
        trace_class_id=_string(payload["trace_class_id"], f"{path}.trace_class_id"),
        block_ids=block_ids,
        warp_traces=warps,
    )


def _launch_to_json(value: KernelLaunch) -> dict[str, Any]:
    return {
        "implementation_id": value.implementation_id,
        "trace_id": value.trace_id,
        "grid_blocks": value.grid_blocks,
        "threads_per_block": value.threads_per_block,
        "registers_per_thread": value.registers_per_thread,
        "static_shared_memory_bytes": value.static_shared_memory_bytes,
        "dynamic_shared_memory_bytes": value.dynamic_shared_memory_bytes,
        "cta_traces": [_cta_trace_to_json(item) for item in value.cta_traces],
        "cooperative": value.cooperative,
        "cluster_blocks": value.cluster_blocks,
    }


def _launch_from_json(value: object, path: str) -> KernelLaunch:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={
            "implementation_id",
            "trace_id",
            "grid_blocks",
            "threads_per_block",
            "registers_per_thread",
            "static_shared_memory_bytes",
            "dynamic_shared_memory_bytes",
            "cta_traces",
            "cooperative",
            "cluster_blocks",
        },
    )
    cta_traces = tuple(
        _cta_trace_from_json(item, f"{path}.cta_traces[{index}]")
        for index, item in enumerate(_array(payload["cta_traces"], f"{path}.cta_traces"))
    )
    if not cta_traces:
        raise ValueError(f"{path}.cta_traces: must not be empty")
    return KernelLaunch(
        implementation_id=_string(payload["implementation_id"], f"{path}.implementation_id"),
        trace_id=_string(payload["trace_id"], f"{path}.trace_id"),
        grid_blocks=_integer(payload["grid_blocks"], f"{path}.grid_blocks", minimum=1),
        threads_per_block=_integer(
            payload["threads_per_block"], f"{path}.threads_per_block", minimum=1
        ),
        registers_per_thread=_integer(
            payload["registers_per_thread"], f"{path}.registers_per_thread", minimum=0
        ),
        static_shared_memory_bytes=_integer(
            payload["static_shared_memory_bytes"],
            f"{path}.static_shared_memory_bytes",
            minimum=0,
        ),
        dynamic_shared_memory_bytes=_integer(
            payload["dynamic_shared_memory_bytes"],
            f"{path}.dynamic_shared_memory_bytes",
            minimum=0,
        ),
        cta_traces=cta_traces,
        cooperative=_boolean(payload["cooperative"], f"{path}.cooperative"),
        cluster_blocks=_integer(payload["cluster_blocks"], f"{path}.cluster_blocks", minimum=1),
    )


def _catalog_to_json(value: GpuKernelCatalogRecord) -> dict[str, Any]:
    return {
        "kernel": value.kernel,
        "config": [[name, item] for name, item in value.config],
        "gpu": value.gpu,
        "binary_sha256": value.binary_sha256,
        "function_sha256": value.function_sha256,
        "trace_sha256": value.trace_sha256,
        "launch": _launch_to_json(value.launch),
        "capture_id": value.capture_id,
        "launch_order": value.launch_order,
        "stream_id": value.stream_id,
        "attributes": [list(item) for item in value.attributes],
    }


def _catalog_from_json(value: object, path: str) -> GpuKernelCatalogRecord:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={
            "kernel",
            "config",
            "gpu",
            "binary_sha256",
            "function_sha256",
            "trace_sha256",
            "launch",
            "capture_id",
            "launch_order",
            "stream_id",
            "attributes",
        },
    )
    config: list[tuple[str, int]] = []
    for index, raw_entry in enumerate(_array(payload["config"], f"{path}.config")):
        entry_path = f"{path}.config[{index}]"
        entry = _array(raw_entry, entry_path)
        if len(entry) != 2:
            raise ValueError(f"{entry_path}: expected [name, integer]")
        config.append((_string(entry[0], f"{entry_path}[0]"), _integer(entry[1], f"{entry_path}[1]")))
    _unique(tuple(name for name, _ in config), f"{path}.config names")
    return GpuKernelCatalogRecord(
        kernel=_string(payload["kernel"], f"{path}.kernel"),
        config=tuple(config),
        gpu=_string(payload["gpu"], f"{path}.gpu"),
        binary_sha256=_sha256(payload["binary_sha256"], f"{path}.binary_sha256"),
        function_sha256=_sha256(payload["function_sha256"], f"{path}.function_sha256"),
        trace_sha256=_sha256(payload["trace_sha256"], f"{path}.trace_sha256"),
        launch=_launch_from_json(payload["launch"], f"{path}.launch"),
        capture_id=_optional_string(payload["capture_id"], f"{path}.capture_id"),
        launch_order=_optional_integer(payload["launch_order"], f"{path}.launch_order", minimum=0),
        stream_id=_optional_string(payload["stream_id"], f"{path}.stream_id"),
        attributes=_text_pairs(payload["attributes"], f"{path}.attributes"),
    )


_ESTIMATE_INTEGER_FIELDS = {
    "duration_cycles",
    "duration_ps",
    "resident_blocks_per_sm",
    "cta_waves",
    "issued_instructions",
    "scheduler_stall_cycles",
    "dependency_stall_cycles",
    "pipeline_stall_cycles",
    "completion_drain_cycles",
    "hbm_requested_bytes",
    "hbm_transacted_bytes",
    "hbm_serviced_bytes",
    "hbm_request_instructions",
    "nvlink_requested_bytes",
    "nvlink_transacted_bytes",
    "nvlink_request_instructions",
    "completed_blocks",
}


def _estimate_to_json(value: GpuKernelEstimate) -> dict[str, Any]:
    payload: dict[str, Any] = {
        name: getattr(value, name) for name in sorted(_ESTIMATE_INTEGER_FIELDS)
    }
    payload.update(
        {
            "model_implementation": value.model_implementation,
            "architecture_profile_id": value.architecture_profile_id,
            "calibration_id": value.calibration_id,
            "implementation_id": value.implementation_id,
            "trace_id": value.trace_id,
            "pipeline_issue_counts": [
                [kind.value, count] for kind, count in value.pipeline_issue_counts
            ],
            "sm_last_completion_cycles": list(value.sm_last_completion_cycles),
            "sm_scheduler_pressure_cycles": list(value.sm_scheduler_pressure_cycles),
            "sm_dependency_idle_cycles": list(value.sm_dependency_idle_cycles),
            "sm_pipeline_idle_cycles": list(value.sm_pipeline_idle_cycles),
            "sm_completion_drain_cycles": list(value.sm_completion_drain_cycles),
            "relative_uncertainty": float(value.relative_uncertainty),
        }
    )
    return payload


def _estimate_from_json(value: object, path: str) -> GpuKernelEstimate:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required=_ESTIMATE_INTEGER_FIELDS
        | {
            "architecture_profile_id",
            "model_implementation",
            "calibration_id",
            "implementation_id",
            "trace_id",
            "pipeline_issue_counts",
            "sm_last_completion_cycles",
            "sm_scheduler_pressure_cycles",
            "sm_dependency_idle_cycles",
            "sm_pipeline_idle_cycles",
            "sm_completion_drain_cycles",
            "relative_uncertainty",
        },
    )
    values = {
        name: _integer(payload[name], f"{path}.{name}", minimum=0)
        for name in _ESTIMATE_INTEGER_FIELDS
    }
    pipeline_counts: list[tuple[PipelineKind, int]] = []
    for index, raw_entry in enumerate(
        _array(payload["pipeline_issue_counts"], f"{path}.pipeline_issue_counts")
    ):
        entry_path = f"{path}.pipeline_issue_counts[{index}]"
        entry = _array(raw_entry, entry_path)
        if len(entry) != 2:
            raise ValueError(f"{entry_path}: expected [pipeline, issue_count]")
        pipeline_counts.append(
            (
                _enum_value(PipelineKind, entry[0], f"{entry_path}[0]"),
                _integer(entry[1], f"{entry_path}[1]", minimum=0),
            )
        )
    _unique(tuple(kind for kind, _ in pipeline_counts), f"{path}.pipeline_issue_counts")
    sm_last_completion_cycles = tuple(
        _integer(item, f"{path}.sm_last_completion_cycles[{index}]", minimum=0)
        for index, item in enumerate(
            _array(payload["sm_last_completion_cycles"], f"{path}.sm_last_completion_cycles")
        )
    )
    per_sm_counters: dict[str, tuple[int, ...]] = {}
    for name in (
        "sm_scheduler_pressure_cycles",
        "sm_dependency_idle_cycles",
        "sm_pipeline_idle_cycles",
        "sm_completion_drain_cycles",
    ):
        per_sm_counters[name] = tuple(
            _integer(item, f"{path}.{name}[{index}]", minimum=0)
            for index, item in enumerate(_array(payload[name], f"{path}.{name}"))
        )
    return GpuKernelEstimate(
        model_implementation=_string(payload["model_implementation"], f"{path}.model_implementation"),
        architecture_profile_id=_string(
            payload["architecture_profile_id"], f"{path}.architecture_profile_id"
        ),
        calibration_id=_string(payload["calibration_id"], f"{path}.calibration_id"),
        implementation_id=_string(payload["implementation_id"], f"{path}.implementation_id"),
        trace_id=_string(payload["trace_id"], f"{path}.trace_id"),
        pipeline_issue_counts=tuple(pipeline_counts),
        sm_active_cycles=sm_last_completion_cycles,
        **per_sm_counters,
        relative_uncertainty=_number(
            payload["relative_uncertainty"], f"{path}.relative_uncertainty", minimum=0.0
        ),
        **values,
    )


def _replay_to_json(value: GpuReplayRecord) -> dict[str, Any]:
    return {
        "replay_id": value.replay_id,
        "implementation_id": value.implementation_id,
        "architecture_profile_id": value.architecture_profile_id,
        "calibration_id": value.calibration_id,
        "calibration_split": value.calibration_split.value,
        "relative_uncertainty": float(value.relative_uncertainty),
        "estimate": _estimate_to_json(value.estimate),
    }


def _replay_from_json(value: object, path: str) -> GpuReplayRecord:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={
            "replay_id",
            "implementation_id",
            "architecture_profile_id",
            "calibration_id",
            "calibration_split",
            "relative_uncertainty",
            "estimate",
        },
    )
    return GpuReplayRecord(
        replay_id=_string(payload["replay_id"], f"{path}.replay_id"),
        implementation_id=_string(payload["implementation_id"], f"{path}.implementation_id"),
        architecture_profile_id=_string(
            payload["architecture_profile_id"], f"{path}.architecture_profile_id"
        ),
        calibration_id=_string(payload["calibration_id"], f"{path}.calibration_id"),
        calibration_split=_enum_value(
            CalibrationSplit, payload["calibration_split"], f"{path}.calibration_split"
        ),
        relative_uncertainty=_number(
            payload["relative_uncertainty"], f"{path}.relative_uncertainty", minimum=0.0
        ),
        estimate=_estimate_from_json(payload["estimate"], f"{path}.estimate"),
    )


def _copy_transfer_to_json(value: CopyTransfer) -> dict[str, Any]:
    return {
        "transfer_id": value.transfer_id,
        "direction": value.direction.value,
        "bytes": value.bytes,
        "source": value.source,
        "destination": value.destination,
    }


def _copy_transfer_from_json(value: object, path: str) -> CopyTransfer:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={"transfer_id", "direction", "bytes", "source", "destination"},
    )
    return CopyTransfer(
        transfer_id=_string(payload["transfer_id"], f"{path}.transfer_id"),
        direction=_enum_value(CopyDirection, payload["direction"], f"{path}.direction"),
        bytes=_integer(payload["bytes"], f"{path}.bytes", minimum=1),
        source=_string(payload["source"], f"{path}.source"),
        destination=_string(payload["destination"], f"{path}.destination"),
    )


def _copy_estimate_to_json(value: CopyServiceEstimate) -> dict[str, Any]:
    return {
        "transfer_id": value.transfer_id,
        "engine_id": value.engine_id,
        "direction": value.direction.value,
        "source": value.source,
        "destination": value.destination,
        "duration_cycles": value.duration_cycles,
        "duration_ps": value.duration_ps,
        "setup_cycles": value.setup_cycles,
        "transfer_cycles": value.transfer_cycles,
        "bytes_transferred": value.bytes_transferred,
        "effective_bandwidth_bytes_per_cycle": float(
            value.effective_bandwidth_bytes_per_cycle
        ),
        "relative_uncertainty": float(value.relative_uncertainty),
    }


def _copy_estimate_from_json(value: object, path: str) -> CopyServiceEstimate:
    payload = _object(value, path)
    integer_fields = {
        "duration_cycles",
        "duration_ps",
        "setup_cycles",
        "transfer_cycles",
        "bytes_transferred",
    }
    _fields(
        payload,
        path,
        required=integer_fields
        | {
            "transfer_id",
            "engine_id",
            "direction",
            "source",
            "destination",
            "effective_bandwidth_bytes_per_cycle",
            "relative_uncertainty",
        },
    )
    integers = {
        name: _integer(payload[name], f"{path}.{name}", minimum=0) for name in integer_fields
    }
    return CopyServiceEstimate(
        transfer_id=_string(payload["transfer_id"], f"{path}.transfer_id"),
        engine_id=_string(payload["engine_id"], f"{path}.engine_id"),
        direction=_enum_value(CopyDirection, payload["direction"], f"{path}.direction"),
        source=_string(payload["source"], f"{path}.source"),
        destination=_string(payload["destination"], f"{path}.destination"),
        effective_bandwidth_bytes_per_cycle=_number(
            payload["effective_bandwidth_bytes_per_cycle"],
            f"{path}.effective_bandwidth_bytes_per_cycle",
            minimum=0.0,
        ),
        relative_uncertainty=_number(
            payload["relative_uncertainty"],
            f"{path}.relative_uncertainty",
            minimum=0.0,
        ),
        **integers,
    )


def _copy_replay_to_json(value: GpuCopyReplayRecord) -> dict[str, Any]:
    return {
        "replay_id": value.replay_id,
        "architecture_profile_id": value.architecture_profile_id,
        "calibration_id": value.calibration_id,
        "calibration_split": value.calibration_split.value,
        "relative_uncertainty": float(value.relative_uncertainty),
        "transfer": _copy_transfer_to_json(value.transfer),
        "estimate": _copy_estimate_to_json(value.estimate),
        "capture_id": value.capture_id,
        "launch_order": value.launch_order,
        "stream_id": value.stream_id,
        "attributes": [list(item) for item in value.attributes],
    }


def _copy_replay_from_json(value: object, path: str) -> GpuCopyReplayRecord:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={
            "replay_id",
            "architecture_profile_id",
            "calibration_id",
            "calibration_split",
            "relative_uncertainty",
            "transfer",
            "estimate",
            "capture_id",
            "launch_order",
            "stream_id",
            "attributes",
        },
    )
    return GpuCopyReplayRecord(
        replay_id=_string(payload["replay_id"], f"{path}.replay_id"),
        architecture_profile_id=_string(
            payload["architecture_profile_id"], f"{path}.architecture_profile_id"
        ),
        calibration_id=_string(payload["calibration_id"], f"{path}.calibration_id"),
        calibration_split=_enum_value(
            CalibrationSplit, payload["calibration_split"], f"{path}.calibration_split"
        ),
        relative_uncertainty=_number(
            payload["relative_uncertainty"], f"{path}.relative_uncertainty", minimum=0.0
        ),
        transfer=_copy_transfer_from_json(payload["transfer"], f"{path}.transfer"),
        estimate=_copy_estimate_from_json(payload["estimate"], f"{path}.estimate"),
        capture_id=_optional_string(payload["capture_id"], f"{path}.capture_id"),
        launch_order=_optional_integer(payload["launch_order"], f"{path}.launch_order", minimum=0),
        stream_id=_optional_string(payload["stream_id"], f"{path}.stream_id"),
        attributes=_text_pairs(payload["attributes"], f"{path}.attributes"),
    )


def _measurement_to_json(value: GpuMeasurementRecord) -> dict[str, Any]:
    return {
        "measurement_id": value.measurement_id,
        "replay_id": value.replay_id,
        "capture_id": value.capture_id,
        "capture_sha256": value.capture_sha256,
        "duration_samples_ps": list(value.duration_samples_ps),
        "sample_count": len(value.duration_samples_ps),
        "mean_duration_ps": float(value.mean_duration_ps),
        "p50_duration_ps": float(value.p50_duration_ps),
        "coefficient_of_variation": float(value.coefficient_of_variation),
        "calibration_split": value.calibration_split.value,
        "residual_fraction": (
            None if value.residual_fraction is None else float(value.residual_fraction)
        ),
    }


def _measurement_from_json(value: object, path: str) -> GpuMeasurementRecord:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={
            "measurement_id",
            "replay_id",
            "capture_id",
            "capture_sha256",
            "duration_samples_ps",
            "sample_count",
            "mean_duration_ps",
            "p50_duration_ps",
            "coefficient_of_variation",
            "calibration_split",
            "residual_fraction",
        },
    )
    samples = tuple(
        _integer(item, f"{path}.duration_samples_ps[{index}]", minimum=1)
        for index, item in enumerate(
            _array(payload["duration_samples_ps"], f"{path}.duration_samples_ps")
        )
    )
    if not samples:
        raise ValueError(f"{path}.duration_samples_ps: must not be empty")
    sample_count = _integer(payload["sample_count"], f"{path}.sample_count", minimum=1)
    if sample_count != len(samples):
        raise ValueError(
            f"{path}.sample_count: {sample_count} does not match {len(samples)} samples"
        )
    residual = payload["residual_fraction"]
    mean = _number(payload["mean_duration_ps"], f"{path}.mean_duration_ps", minimum=0.0)
    p50 = _number(payload["p50_duration_ps"], f"{path}.p50_duration_ps", minimum=0.0)
    coefficient_of_variation = _number(
        payload["coefficient_of_variation"],
        f"{path}.coefficient_of_variation",
        minimum=0.0,
    )
    expected_mean = statistics.fmean(samples)
    expected_p50 = float(statistics.median(samples))
    expected_cv = statistics.pstdev(samples) / expected_mean
    for name, observed, expected in (
        ("mean_duration_ps", mean, expected_mean),
        ("p50_duration_ps", p50, expected_p50),
        ("coefficient_of_variation", coefficient_of_variation, expected_cv),
    ):
        if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{path}.{name}: does not match duration samples")
    return GpuMeasurementRecord(
        measurement_id=_string(payload["measurement_id"], f"{path}.measurement_id"),
        replay_id=_string(payload["replay_id"], f"{path}.replay_id"),
        capture_id=_string(payload["capture_id"], f"{path}.capture_id"),
        capture_sha256=_sha256(payload["capture_sha256"], f"{path}.capture_sha256"),
        duration_samples_ps=samples,
        mean_duration_ps=mean,
        p50_duration_ps=p50,
        coefficient_of_variation=coefficient_of_variation,
        calibration_split=_enum_value(
            CalibrationSplit, payload["calibration_split"], f"{path}.calibration_split"
        ),
        residual_fraction=(
            None if residual is None else _number(residual, f"{path}.residual_fraction")
        ),
    )


def _artifact_to_payload(value: GpuModelArtifact) -> dict[str, Any]:
    if not isinstance(value, GpuModelArtifact):
        raise TypeError(f"expected GpuModelArtifact, got {type(value).__name__}")
    return {
        "schema": GPU_MODEL_ARTIFACT_SCHEMA,
        "artifact_id": value.artifact_id,
        "architecture": _architecture_to_json(value.architecture),
        "captures": [_capture_to_json(item) for item in value.captures],
        "catalog": [_catalog_to_json(item) for item in value.catalog],
        "replays": [_replay_to_json(item) for item in value.replays],
        "copy_replays": [_copy_replay_to_json(item) for item in value.copy_replays],
        "measurements": [_measurement_to_json(item) for item in value.measurements],
    }


def _mutable_json_copy(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _mutable_json_copy(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_mutable_json_copy(item) for item in value]
    return value


def _upgrade_v1_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade the losslessly representable v1 fields to the v2 wire shape."""

    upgraded = _mutable_json_copy(value)
    if not isinstance(upgraded, dict):
        raise TypeError("artifact: expected an object")
    upgraded["schema"] = GPU_MODEL_ARTIFACT_SCHEMA

    architecture = upgraded.get("architecture")
    if isinstance(architecture, dict):
        calibration = architecture.get("calibration")
        if isinstance(calibration, dict):
            if "nvlink" in calibration:
                raise ValueError(
                    "artifact.architecture.calibration.nvlink: field is not valid in v1"
                )
            calibration["nvlink"] = None

    replays = upgraded.get("replays")
    if isinstance(replays, list):
        for index, replay in enumerate(replays):
            if not isinstance(replay, dict):
                continue
            estimate = replay.get("estimate")
            if not isinstance(estimate, dict):
                continue
            path = f"artifact.replays[{index}].estimate"
            if "sm_last_completion_cycles" in estimate:
                raise ValueError(f"{path}.sm_last_completion_cycles: field is not valid in v1")
            if "sm_active_cycles" in estimate:
                estimate["sm_last_completion_cycles"] = estimate.pop("sm_active_cycles")
            model_implementation = estimate.get("model_implementation")
            if model_implementation == _LEGACY_GPU_MODEL_IMPLEMENTATION:
                estimate["model_implementation"] = GPU_MODEL_IMPLEMENTATION
            elif model_implementation is not None:
                raise ValueError(
                    f"{path}.model_implementation: unsupported v1 implementation "
                    f"{model_implementation!r}"
                )
            for field in (
                "nvlink_requested_bytes",
                "nvlink_transacted_bytes",
                "nvlink_request_instructions",
            ):
                if field in estimate:
                    raise ValueError(f"{path}.{field}: field is not valid in v1")
                estimate[field] = 0
    return upgraded


def _artifact_from_payload(value: object) -> GpuModelArtifact:
    payload = _object(value, "artifact")
    # Version discrimination precedes structural validation so a payload from
    # another schema version reports the version, not a confusing field shape.
    if "schema" not in payload:
        raise ValueError("artifact: missing fields ['schema']")
    if payload["schema"] == _LEGACY_GPU_MODEL_ARTIFACT_SCHEMA:
        payload = _upgrade_v1_payload(payload)
    elif payload["schema"] != GPU_MODEL_ARTIFACT_SCHEMA:
        raise ValueError(
            f"artifact.schema: unsupported {payload['schema']!r}; "
            f"expected {GPU_MODEL_ARTIFACT_SCHEMA!r}"
        )
    _fields(
        payload,
        "artifact",
        required={
            "schema",
            "artifact_id",
            "architecture",
            "captures",
            "catalog",
            "replays",
            "copy_replays",
            "measurements",
        },
    )
    architecture = _architecture_from_json(payload["architecture"], "artifact.architecture")
    captures = tuple(
        _capture_from_json(item, f"artifact.captures[{index}]")
        for index, item in enumerate(_array(payload["captures"], "artifact.captures"))
    )
    catalog = tuple(
        _catalog_from_json(item, f"artifact.catalog[{index}]")
        for index, item in enumerate(_array(payload["catalog"], "artifact.catalog"))
    )
    replays = tuple(
        _replay_from_json(item, f"artifact.replays[{index}]")
        for index, item in enumerate(_array(payload["replays"], "artifact.replays"))
    )
    copy_replays = tuple(
        _copy_replay_from_json(item, f"artifact.copy_replays[{index}]")
        for index, item in enumerate(_array(payload["copy_replays"], "artifact.copy_replays"))
    )
    measurements = tuple(
        _measurement_from_json(item, f"artifact.measurements[{index}]")
        for index, item in enumerate(_array(payload["measurements"], "artifact.measurements"))
    )

    _unique(
        tuple((item.kernel, item.config, architecture.profile_id) for item in catalog),
        "artifact.catalog keys",
    )
    _unique(
        tuple(_sha256_digest(item.launch.trace_id) for item in catalog),
        "artifact.catalog trace IDs",
    )
    _unique(tuple(item.capture_id for item in captures), "artifact capture IDs")
    _unique(
        tuple(item.replay_id for item in (*replays, *copy_replays)),
        "artifact replay IDs",
    )
    _unique(
        tuple(
            (
                _sha256_digest(item.implementation_id),
                _sha256_digest(item.estimate.trace_id),
            )
            for item in replays
        ),
        "artifact kernel replay identities",
    )
    _unique(
        tuple(item.transfer.transfer_id for item in copy_replays),
        "artifact copy transfer IDs",
    )
    _unique(
        tuple(
            (
                item.estimate.engine_id,
                item.transfer.direction,
                item.transfer.bytes,
                item.transfer.source,
                item.transfer.destination,
                item.attributes,
            )
            for item in copy_replays
        ),
        "artifact copy service identities",
    )
    _unique(tuple(item.measurement_id for item in measurements), "artifact.measurement IDs")

    capture_by_id = {item.capture_id: item for item in captures}
    for index, capture in enumerate(captures):
        if capture.gpu_name != architecture.gpu_name:
            raise ValueError(
                f"artifact.captures[{index}].gpu_name: does not match architecture GPU"
            )
        if capture.observed_core_clock_hz != architecture.calibration.core_clock_hz:
            raise ValueError(
                f"artifact.captures[{index}].observed_core_clock_hz: "
                "does not match calibration core clock"
            )
        target_memory_clock_hz = architecture.calibration.target_memory_clock_hz
        if target_memory_clock_hz is None:
            raise ValueError(
                "artifact.architecture.calibration.target_memory_clock_hz: "
                "captured artifacts require a numeric target memory clock"
            )
        if capture.observed_memory_clock_hz != target_memory_clock_hz:
            raise ValueError(
                f"artifact.captures[{index}].observed_memory_clock_hz: "
                "does not match calibration target memory clock"
            )
    catalog_by_replay_identity = {
        (
            _sha256_digest(item.launch.implementation_id),
            _sha256_digest(item.launch.trace_id),
        ): item
        for item in catalog
    }
    replay_by_id: dict[str, GpuReplayRecord | GpuCopyReplayRecord] = {
        item.replay_id: item for item in (*replays, *copy_replays)
    }
    replay_capture_ids: dict[str, str | None] = {}
    for index, entry in enumerate(catalog):
        path = f"artifact.catalog[{index}]"
        if entry.gpu not in architecture.all_names:
            raise ValueError(f"{path}.gpu: does not match architecture GPU")
        if _sha256_digest(entry.launch.implementation_id) != _sha256_digest(entry.function_sha256):
            raise ValueError(f"{path}.launch.implementation_id: function hash mismatch")
        if _sha256_digest(entry.launch.trace_id) != _sha256_digest(entry.trace_sha256):
            raise ValueError(f"{path}.launch.trace_id: trace hash mismatch")
        metadata = (entry.capture_id, entry.stream_id, entry.launch_order)
        if any(item is not None for item in metadata) and not all(
            item is not None for item in metadata
        ):
            raise ValueError(
                f"{path}: capture_id, stream_id, and launch_order must be all set or all absent"
            )
        if entry.capture_id is not None:
            if entry.capture_id not in capture_by_id:
                raise ValueError(f"{path}.capture_id: unknown capture")
            if entry.launch_order is None or entry.stream_id is None:
                raise ValueError(f"{path}: captured kernels require launch_order and stream_id")
        SmSchedulerModel(architecture).resident_blocks_per_sm(entry.launch)
    for index, replay in enumerate(replays):
        path = f"artifact.replays[{index}]"
        entry = catalog_by_replay_identity.get(
            (
                _sha256_digest(replay.implementation_id),
                _sha256_digest(replay.estimate.trace_id),
            )
        )
        if entry is None:
            raise ValueError(f"{path}: unknown catalog implementation and trace identity")
        if replay.architecture_profile_id != architecture.profile_id:
            raise ValueError(f"{path}.architecture_profile_id: does not match architecture")
        if replay.calibration_id != architecture.calibration.calibration_id:
            raise ValueError(f"{path}.calibration_id: does not match architecture calibration")
        if replay.estimate.architecture_profile_id != replay.architecture_profile_id:
            raise ValueError(f"{path}.estimate.architecture_profile_id: identity mismatch")
        if replay.estimate.model_implementation != GPU_MODEL_IMPLEMENTATION:
            raise ValueError(f"{path}.estimate.model_implementation: unsupported model")
        if replay.estimate.calibration_id != replay.calibration_id:
            raise ValueError(f"{path}.estimate.calibration_id: identity mismatch")
        if _sha256_digest(replay.estimate.implementation_id) != _sha256_digest(
            replay.implementation_id
        ):
            raise ValueError(f"{path}.estimate.implementation_id: identity mismatch")
        if _sha256_digest(replay.estimate.trace_id) != _sha256_digest(entry.launch.trace_id):
            raise ValueError(f"{path}.estimate.trace_id: does not match catalog launch")
        if replay.estimate.relative_uncertainty != replay.relative_uncertainty:
            raise ValueError(f"{path}.estimate.relative_uncertainty: replay mismatch")
        if replay.relative_uncertainty != architecture.calibration.relative_uncertainty:
            raise ValueError(f"{path}.relative_uncertainty: calibration mismatch")
        if replay.estimate.completed_blocks != entry.launch.grid_blocks:
            raise ValueError(f"{path}.estimate.completed_blocks: does not match launch grid")
        if len(replay.estimate.sm_last_completion_cycles) != architecture.sm_count:
            raise ValueError(f"{path}.estimate.sm_last_completion_cycles: one counter per SM is required")
        for name in (
            "sm_scheduler_pressure_cycles",
            "sm_dependency_idle_cycles",
            "sm_pipeline_idle_cycles",
            "sm_completion_drain_cycles",
        ):
            if len(getattr(replay.estimate, name)) != architecture.sm_count:
                raise ValueError(f"{path}.estimate.{name}: one counter per SM is required")
        architecture_kinds = {profile.kind for profile in architecture.pipelines}
        replay_kinds = {kind for kind, _ in replay.estimate.pipeline_issue_counts}
        if replay_kinds != architecture_kinds:
            raise ValueError(f"{path}.estimate.pipeline_issue_counts: pipeline set mismatch")
        if sum(count for _, count in replay.estimate.pipeline_issue_counts) != (
            replay.estimate.issued_instructions
        ):
            raise ValueError(f"{path}.estimate.pipeline_issue_counts: issue total mismatch")
        if replay.estimate.hbm_serviced_bytes != replay.estimate.hbm_transacted_bytes:
            raise ValueError(
                f"{path}.estimate: isolated replay must service every transacted HBM byte"
            )
        expected_duration_ps = (
            replay.estimate.duration_cycles * 1_000_000_000_000 + architecture.clock_hz - 1
        ) // architecture.clock_hz
        if replay.estimate.duration_ps != expected_duration_ps:
            raise ValueError(f"{path}.estimate.duration_ps: inconsistent with cycles and clock")
        expected_estimate = SmSchedulerModel(architecture).estimate(entry.launch)
        if replay.estimate != expected_estimate:
            raise ValueError(f"{path}.estimate: deterministic replay mismatch")
        replay_capture_ids[replay.replay_id] = entry.capture_id

    for index, replay in enumerate(copy_replays):
        path = f"artifact.copy_replays[{index}]"
        if replay.architecture_profile_id != architecture.profile_id:
            raise ValueError(f"{path}.architecture_profile_id: does not match architecture")
        if replay.calibration_id != architecture.calibration.calibration_id:
            raise ValueError(f"{path}.calibration_id: does not match architecture calibration")
        if replay.relative_uncertainty != architecture.calibration.relative_uncertainty:
            raise ValueError(f"{path}.relative_uncertainty: calibration mismatch")
        metadata = (replay.capture_id, replay.stream_id, replay.launch_order)
        if any(item is not None for item in metadata) and not all(
            item is not None for item in metadata
        ):
            raise ValueError(
                f"{path}: capture_id, stream_id, and launch_order must be all set or all absent"
            )
        if replay.capture_id is not None:
            if replay.capture_id not in capture_by_id:
                raise ValueError(f"{path}.capture_id: unknown capture")
            if replay.launch_order is None or replay.stream_id is None:
                raise ValueError(f"{path}: captured copies require launch_order and stream_id")
        expected_copy = CopyEngineServiceModel(architecture, replay.estimate.engine_id).estimate(
            replay.transfer
        )
        if replay.estimate != expected_copy:
            raise ValueError(f"{path}.estimate: deterministic copy replay mismatch")
        replay_capture_ids[replay.replay_id] = replay.capture_id

    operation_orders = tuple(
        (item.capture_id, item.stream_id, item.launch_order)
        for item in catalog
        if item.capture_id is not None
    ) + tuple(
        (item.capture_id, item.stream_id, item.launch_order)
        for item in copy_replays
        if item.capture_id is not None
    )
    _unique(operation_orders, "artifact captured stream launch orders")

    capture_splits: dict[str, CalibrationSplit] = {}
    for index, measurement in enumerate(measurements):
        path = f"artifact.measurements[{index}]"
        replay = replay_by_id.get(measurement.replay_id)
        if replay is None:
            raise ValueError(f"{path}.replay_id: unknown replay")
        if measurement.calibration_split is not replay.calibration_split:
            raise ValueError(f"{path}.calibration_split: does not match replay split")
        if measurement.capture_id not in capture_by_id:
            raise ValueError(f"{path}.capture_id: unknown capture")
        if replay_capture_ids[measurement.replay_id] != measurement.capture_id:
            raise ValueError(f"{path}.capture_id: does not match replay capture")
        previous_split = capture_splits.setdefault(
            _sha256_digest(measurement.capture_sha256), measurement.calibration_split
        )
        if previous_split is not measurement.calibration_split:
            raise ValueError(f"{path}.capture_sha256: capture crosses calibration splits")
        if measurement.residual_fraction is not None:
            expected_residual = (
                replay.estimate.duration_ps - measurement.mean_duration_ps
            ) / measurement.mean_duration_ps
            if not math.isclose(
                measurement.residual_fraction,
                expected_residual,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{path}.residual_fraction: expected "
                    "(model duration - measured mean) / measured mean"
                )

    return GpuModelArtifact(
        artifact_id=_string(payload["artifact_id"], "artifact.artifact_id"),
        architecture=architecture,
        captures=captures,
        catalog=catalog,
        replays=replays,
        copy_replays=copy_replays,
        measurements=measurements,
    )


def gpu_model_artifact_to_json(value: GpuModelArtifact) -> dict[str, Any]:
    """Return a canonical JSON-compatible mapping after full validation."""

    payload = _artifact_to_payload(value)
    _artifact_from_payload(payload)
    return payload


def gpu_model_artifact_from_json(value: Mapping[str, Any]) -> GpuModelArtifact:
    """Parse and validate one already-decoded artifact mapping."""

    return _artifact_from_payload(value)


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def save_gpu_model_artifact(value: GpuModelArtifact, path: str | Path) -> Path:
    """Validate and write one deterministic UTF-8 JSON artifact."""

    output = Path(path)
    payload = gpu_model_artifact_to_json(value)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def load_gpu_model_artifact(path: str | Path) -> GpuModelArtifact:
    """Load JSON while rejecting duplicate keys and non-finite numbers."""

    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_object_keys,
        parse_constant=_reject_nonfinite_constant,
    )
    return _artifact_from_payload(payload)


def gpu_model_artifact_to_profile_table(
    artifact: GpuModelArtifact,
) -> ProfileTableProvider:
    """Compile validated kernel replays into the O(1) online lookup surface."""

    validated = _artifact_from_payload(_artifact_to_payload(artifact))
    catalog_by_identity = {
        (
            _sha256_digest(item.launch.implementation_id),
            _sha256_digest(item.launch.trace_id),
        ): item
        for item in validated.catalog
    }
    table = {}
    for replay in validated.replays:
        entry = catalog_by_identity[
            (
                _sha256_digest(replay.implementation_id),
                _sha256_digest(replay.estimate.trace_id),
            )
        ]
        key = (entry.kernel, entry.config, validated.architecture.gpu_name)
        if key in table:
            raise ValueError(f"duplicate online profile key {key!r}")
        table[key] = (replay.estimate.duration_ps, replay.relative_uncertainty)
    return ProfileTableProvider(
        table,
        provenance=ProfileTableProvenance(
            source=f"gpu-model-artifact:{validated.artifact_id}",
            version=GPU_MODEL_IMPLEMENTATION,
            gpu=validated.architecture.gpu_name,
            created=validated.architecture.calibration.provenance.created,
            references=validated.architecture.calibration.provenance.references,
        ),
    )


__all__ = [
    "GPU_MODEL_ARTIFACT_SCHEMA",
    "CalibrationSplit",
    "GpuCaptureEnvironment",
    "GpuCopyReplayRecord",
    "GpuKernelCatalogRecord",
    "GpuMeasurementRecord",
    "GpuModelArtifact",
    "GpuReplayRecord",
    "gpu_model_artifact_from_json",
    "gpu_model_artifact_to_json",
    "gpu_model_artifact_to_profile_table",
    "load_gpu_model_artifact",
    "save_gpu_model_artifact",
]
