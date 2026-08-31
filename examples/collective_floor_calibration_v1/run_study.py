"""Run the corrected true-byte aggregate collective-floor study.

The three expectation documents and ``study_config.json`` freeze every scored
relation. A normal invocation creates two fresh-process evaluations below one
new append-only attempt directory, compares their records after removing only
fields named ``wall_time_seconds``, and writes the tracked record and CSV.

    python examples/collective_floor_calibration_v1/run_study.py \
        --workdir <new attempt directory>
    python examples/collective_floor_calibration_v1/run_study.py --check
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from enum import Enum
from fractions import Fraction
from io import StringIO
from pathlib import Path
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from examples.collective_floor_calibration_v1.bypass_fixture import (
    PRE_WAVE_COMMIT,
    produce_bypass_record,
)
from simllm.backends import (
    CollectiveFloorTransferError,
    HtsimRequestMetricReducer,
    HtsimRnicConfig,
    HtsimStepSink,
    HtsimStepSinkConfig,
    attribute_step_detail,
    run_htsim_rnic,
)
from simllm.calibration.external_nccl import ExternalNcclDatabase
from simllm.compute import (
    GPU_ENVELOPES,
    HostInitiationModel,
    ModelDims,
    RooflineProvider,
)
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.goal import to_binary
from simllm.placement import PlacementManifest, RankMapper, RankPlacement
from simllm.traffic import (
    B200_NCCL_2_27_LOCAL_PROFILE,
    COLLECTIVE_FLOOR_CALIBRATED,
    COLLECTIVE_FLOOR_TRANSFERRED,
    CollectiveCommunicationPhase,
    CollectiveFloorCalibration,
    CollectiveFloorCell,
    CollectiveFloorCurveBoundaries,
    CollectiveFloorSourceIdentity,
    DirectedCollectiveSegment,
    classify_step_locality,
    fit_collective_floor_calibration,
    render_fabric_phase_goal,
    source_elements_for_bytes,
)

CONFIG_PATH = STUDY_DIR / "study_config.json"
TRACKED_RECORD = STUDY_DIR / "record.json"
TRACKED_CSV = STUDY_DIR / "results.csv"
PRE_WAVE_GOLDEN = STUDY_DIR / "pre_wave_bypass_golden.json"
SCHEMA = "simllm-collective-floor-calibration-record-v1"
CALIBRATION_ID = "h200-nccl-2.26.2-aggregate-floor-v1"
CONFIG_COMMIT = "fdffaec"
IMPLEMENTATION_COMMIT = "a983c8c"
COORDINATE_FREEZE_COMMIT = "6df368885a715b12e2c2fdfa1bd7ccb2223236a7"
HTSIM_ENV = "SIMLLM_HTSIM_RNIC"
TXT2BIN_ENV = "SIMLLM_TXT2BIN"
EXTERNAL_VENV_ENV = "SIMLLM_EXTERNAL_AIC_VENV"
LOCAL_BANDWIDTH_BYTES_PER_SECOND = 450_000_000_000
WALL_BOUND_SECONDS = 600.0
REPRESENTED_LAYERS = 65
MINIMAX_RECORD = REPOSITORY_ROOT / "examples/minimax_ep_scaling_v1/record.json"
MINIMAX_CONFIG = REPOSITORY_ROOT / "examples/minimax_ep_scaling_v1/study_config.json"


class StudyUnavailable(RuntimeError):
    """A named environmental dependency is unavailable."""


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, Path):
        return value.name
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    return value


def _without_wall_time(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_wall_time(item)
            for key, item in value.items()
            if key != "wall_time_seconds"
        }
    if isinstance(value, list):
        return [_without_wall_time(item) for item in value]
    return value


def _load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("schema") != "simllm-collective-floor-calibration-study-v1":
        raise RuntimeError("study_config.json has an unsupported schema")
    if config["axis"]["source_coordinate"] != "ELEMENTS":
        raise RuntimeError("the corrected source axis is not pinned to ELEMENTS")
    if config["axis"]["physical_fit_axis"] != "BYTES":
        raise RuntimeError("the physical fit axis is not pinned to BYTES")
    if len(config["membership"]["training_cells"]) != 63:
        raise RuntimeError("the training membership does not contain 63 cells")
    if len(config["membership"]["holdout_cells"]) != 63:
        raise RuntimeError("the holdout membership does not contain 63 cells")
    return config


def _source(config: dict[str, Any]) -> CollectiveFloorSourceIdentity:
    return CollectiveFloorSourceIdentity(**config["source"])


def _observed_cell(
    database: ExternalNcclDatabase,
    member: dict[str, Any],
) -> CollectiveFloorCell:
    observed = database.query(
        dtype=member["dtype"],
        operation=member["operation"],
        ranks=member["ranks"],
        message_size=member["source_elements"],
    )
    return CollectiveFloorCell(
        cell_id=member["cell_id"],
        dtype=member["dtype"],
        operation=member["operation"],
        ranks=member["ranks"],
        source_elements=member["source_elements"],
        message_bytes=member["true_bytes"],
        latency_ps=round(observed.latency_ms * 1_000_000_000),
    )


def _fit(
    config: dict[str, Any],
    database: ExternalNcclDatabase,
) -> tuple[CollectiveFloorCalibration, tuple[CollectiveFloorCell, ...]]:
    training = tuple(
        _observed_cell(database, member)
        for member in config["membership"]["training_cells"]
    )
    boundaries = tuple(
        CollectiveFloorCurveBoundaries(
            dtype=row["dtype"],
            operation=row["operation"],
            ranks=row["ranks"],
            lower_bounds_of_following_regimes=tuple(
                row["lower_bounds_of_following_regimes"]
            ),
        )
        for row in config["fit"]["regime_boundaries_true_bytes"]
    )
    byte_range = config["fit"]["true_byte_range"]
    calibration = fit_collective_floor_calibration(
        calibration_id=CALIBRATION_ID,
        source=_source(config),
        cells=training,
        boundaries=boundaries,
        fitted_byte_range=(byte_range["minimum"], byte_range["maximum"]),
    )
    return calibration, training


def _axis_check(
    config: dict[str, Any],
    database: ExternalNcclDatabase,
) -> dict[str, Any]:
    guard = config["axis"]["equal_byte_guard"]
    observed: dict[str, dict[str, Any]] = {}
    for dtype in ("half", "int8"):
        frozen = guard[f"dtype_{dtype}"]
        elements = source_elements_for_bytes(dtype, guard["true_bytes"])
        latency = database.query(
            dtype=dtype,
            operation=frozen["operation"],
            ranks=frozen["ranks"],
            message_size=elements,
        ).latency_ms
        observed[dtype] = {
            "true_bytes": guard["true_bytes"],
            "source_elements": elements,
            "latency_ms": format(latency, ".5f"),
            "matches_frozen_cell": (
                elements == frozen["source_elements"]
                and format(latency, ".5f") == frozen["measured_latency_ms"]
            ),
        }
    distinct = observed["half"]["latency_ms"] != observed["int8"]["latency_ms"]
    return {
        "source_coordinate": "ELEMENTS",
        "physical_axis": "BYTES",
        "conversion": config["axis"]["conversion"],
        "observed": observed,
        "equal_byte_cells_are_distinct": distinct,
        "passed": distinct
        and all(row["matches_frozen_cell"] for row in observed.values()),
    }


def _ceil_serialization_ps(message_bytes: int) -> int:
    numerator = message_bytes * 1_000_000_000_000
    return (numerator + LOCAL_BANDWIDTH_BYTES_PER_SECOND - 1) // (
        LOCAL_BANDWIDTH_BYTES_PER_SECOND
    )


def _current_ring_half_ps(message_bytes: int, ranks: int) -> int:
    """Price one semantic half exactly as the current ring path does."""

    chunk_bytes = max(1, message_bytes // ranks)
    phase_ns = math.ceil(
        chunk_bytes * 1_000_000_000 / LOCAL_BANDWIDTH_BYTES_PER_SECOND
    )
    return phase_ns * 1_000 * (ranks - 1)


def _ring_physical_floor_ps(message_bytes: int, ranks: int) -> int:
    chunk_bytes = max(1, message_bytes // ranks)
    transmitted_bytes = chunk_bytes * (ranks - 1)
    numerator = transmitted_bytes * 1_000_000_000_000
    return math.ceil(numerator / LOCAL_BANDWIDTH_BYTES_PER_SECOND)


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _family_h(
    config: dict[str, Any],
    database: ExternalNcclDatabase,
    calibration: CollectiveFloorCalibration,
) -> dict[str, Any]:
    rows = []
    before_errors = []
    after_errors = []
    by_curve: dict[str, list[bool]] = {}
    for member in config["membership"]["holdout_cells"]:
        cell = _observed_cell(database, member)
        estimate = calibration.estimate(
            dtype=cell.dtype,
            operation=cell.operation,
            ranks=cell.ranks,
            message_bytes=cell.message_bytes,
        )
        before_ps = _current_ring_half_ps(cell.message_bytes, cell.ranks)
        physical_floor_ps = _ring_physical_floor_ps(
            cell.message_bytes,
            cell.ranks,
        )
        before_error = abs(before_ps - cell.latency_ps) / cell.latency_ps
        after_error = abs(estimate.completion_ps - cell.latency_ps) / cell.latency_ps
        tolerance = max(
            0.10,
            config["clock"]["two_gpu_cycles_ps_ceiling"] / cell.latency_ps,
        )
        passed = after_error <= tolerance
        curve = f"{cell.operation}/r{cell.ranks}"
        by_curve.setdefault(curve, []).append(passed)
        before_errors.append(before_error)
        after_errors.append(after_error)
        rows.append(
            {
                "cell_id": cell.cell_id,
                "operation": cell.operation,
                "ranks": cell.ranks,
                "source_elements": cell.source_elements,
                "true_bytes": cell.message_bytes,
                "measured_ps": cell.latency_ps,
                "current_ring_ps": before_ps,
                "physical_ring_floor_ps": physical_floor_ps,
                "calibrated_ps": estimate.completion_ps,
                "before_relative_error": before_error,
                "after_relative_error": after_error,
                "tolerance": tolerance,
                "regime_index": estimate.regime.regime_index,
                "passed": passed,
            }
        )
    passed_count = sum(row["passed"] for row in rows)
    return {
        "id": "H",
        "status": "PASS" if passed_count == 63 else "REFUTED",
        "passed": passed_count,
        "denominator": 63,
        "band": "relative error <= max(10 percent, two GPU cycles)",
        "two_gpu_cycles_ps_ceiling": config["clock"][
            "two_gpu_cycles_ps_ceiling"
        ],
        "summary": {
            "before_median_relative_error": _median(before_errors),
            "before_p95_relative_error_nearest_rank": _nearest_rank_percentile(
                before_errors, 0.95
            ),
            "after_median_relative_error": _median(after_errors),
            "after_p95_relative_error_nearest_rank": _nearest_rank_percentile(
                after_errors, 0.95
            ),
            "median_improvement_factor": (
                _median(before_errors) / _median(after_errors)
            ),
        },
        "curve_tallies": {
            curve: {"passed": sum(values), "denominator": len(values)}
            for curve, values in sorted(by_curve.items())
        },
        "physical_sanity": {
            "floor": "(world - 1) payload/world bytes divided by 450 GB/s",
            "ceiling": "unbounded because the source exposes no algorithm progress bound",
            "all_measurements_above_ring_floor": all(
                row["measured_ps"] >= row["physical_ring_floor_ps"] for row in rows
            ),
            "before_column_is_current_ring_path": True,
        },
        "rows": rows,
    }


def _manifest(hosts: tuple[str, ...]) -> PlacementManifest:
    counts: dict[str, int] = {}
    ranks = []
    for global_rank, hostname in enumerate(hosts):
        local_rank = counts.get(hostname, 0)
        counts[hostname] = local_rank + 1
        ranks.append(
            RankPlacement(
                global_rank=global_rank,
                hostname=hostname,
                local_rank=local_rank,
            )
        )
    return PlacementManifest(ranks=ranks)


def _bypass_dims() -> ModelDims:
    return ModelDims(
        num_layers=1,
        hidden_size=1_024,
        intermediate_size=2_048,
        num_heads=8,
        num_kv_heads=8,
        head_size=128,
        vocab_size=256,
        dtype_bytes=2,
    )


def _records(*, prompt_tokens: int, steps: int) -> list[StepRecord]:
    records = [
        StepRecord(
            step_index=0,
            virtual_time_ps=0,
            scheduled=[
                ScheduledRequest(
                    "r0",
                    RequestPhase.PREFILL,
                    prompt_tokens,
                    context_length=prompt_tokens,
                )
            ],
            num_sampled=1,
        )
    ]
    for step_index in range(1, steps):
        records.append(
            StepRecord(
                step_index=step_index,
                virtual_time_ps=0,
                scheduled=[
                    ScheduledRequest(
                        "r0",
                        RequestPhase.DECODE,
                        1,
                        context_length=prompt_tokens + step_index,
                    )
                ],
                num_sampled=1,
            )
        )
    return records


def _artifact_manifest(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(paths, key=lambda candidate: candidate.name):
        raw = path.read_bytes()
        if path.suffix == ".goal":
            sends = [int(value) for value in re.findall(rb"send ([0-9]+)b", raw)]
            rows.append(
                {
                    "kind": "goal",
                    "name": path.name,
                    "bytes": len(raw),
                    "sha256": _sha256_bytes(raw),
                    "application_send_bytes": sum(sends),
                }
            )
        elif path.suffix == ".csv":
            rows.append(
                {
                    "kind": "completion",
                    "name": path.name,
                    "bytes": len(raw),
                    "sha256": _sha256_bytes(raw),
                }
            )
        else:
            raise ValueError(f"unsupported study artifact {path.name!r}")
    return rows


def _require_executable(name: str) -> Path:
    raw = os.environ.get(name)
    if not raw:
        raise StudyUnavailable(f"{name} is not configured")
    path = Path(raw)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise StudyUnavailable(f"{name} does not name an executable file")
    return path


def _binary_identity(name: str) -> dict[str, Any]:
    try:
        path = _require_executable(name)
    except StudyUnavailable as error:
        return {"available": False, "skip_reason": str(error)}
    return {
        "available": True,
        "filename": path.name,
        "sha256": _sha256_file(path),
        "skip_reason": None,
    }


def _first_divergent_field(expected: Any, observed: Any, path: str = "$") -> str | None:
    if type(expected) is not type(observed):
        return path
    if isinstance(expected, dict):
        if expected.keys() != observed.keys():
            return path
        for key in expected:
            divergence = _first_divergent_field(
                expected[key],
                observed[key],
                f"{path}.{key}",
            )
            if divergence is not None:
                return divergence
        return None
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return path
        for index, (expected_item, observed_item) in enumerate(
            zip(expected, observed, strict=True)
        ):
            divergence = _first_divergent_field(
                expected_item,
                observed_item,
                f"{path}[{index}]",
            )
            if divergence is not None:
                return divergence
        return None
    return None if expected == observed else path


def _family_b(workdir: Path) -> dict[str, Any]:
    try:
        _require_executable(HTSIM_ENV)
        _require_executable(TXT2BIN_ENV)
    except StudyUnavailable as error:
        return {
            "id": "B",
            "status": "SKIPPED",
            "passed": None,
            "denominator": 1,
            "skip_reason": str(error),
        }
    golden = json.loads(PRE_WAVE_GOLDEN.read_text(encoding="utf-8"))
    if golden.get("generating_commit") != PRE_WAVE_COMMIT:
        raise RuntimeError("the pre-wave golden names the wrong generating commit")
    expected = golden["record"]
    observed = produce_bypass_record(workdir / "post-wave-default-off")
    expected_bytes = _json_bytes(expected)
    observed_bytes = _json_bytes(observed)
    passed = expected_bytes == observed_bytes
    return {
        "id": "B",
        "status": "PASS" if passed else "FAIL",
        "passed": int(passed),
        "denominator": 1,
        "skip_reason": None,
        "golden_path": "examples/collective_floor_calibration_v1/pre_wave_bypass_golden.json",
        "golden_generating_commit": PRE_WAVE_COMMIT,
        "golden_file_sha256": _sha256_file(PRE_WAVE_GOLDEN),
        "golden_record_sha256": _sha256_bytes(expected_bytes),
        "post_wave_default_off_sha256": _sha256_bytes(observed_bytes),
        "first_divergent_field": _first_divergent_field(expected, observed),
        "checked_fields": [
            "phase and step timestamps",
            "local and fabric segment tuples",
            "application and wire byte counts",
            "completion order",
            "backend invocation order",
            "random-generator state",
        ],
        "observed": observed,
    }


def _write_width_clos(path: Path, *, width: int) -> Path:
    gpus_per_node = 8
    leaf_count = width // gpus_per_node
    text = f"""Nodes {width}
Tiers 2
Podsize {width}

Tier 0
Downlink_speed_Gbps 400
Radix_Down {gpus_per_node}
Radix_Up {gpus_per_node}
Downlink_Latency_ns 1000
Switch_Latency_ns 0

Tier 1
Downlink_speed_Gbps 400
Radix_Down {leaf_count}
Downlink_Latency_ns 1000
Switch_Latency_ns 0
"""
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _dense_phases(width: int) -> tuple[Any, ...]:
    chunk_bytes = 24_576
    ranks = tuple(range(width))
    raw = tuple(
        CollectiveCommunicationPhase(
            phase_id=f"family-d8:{operation}",
            layer=0,
            participants=ranks,
            segments=tuple(
                DirectedCollectiveSegment(
                    source_rank=source,
                    destination_rank=destination,
                    payload_bytes=chunk_bytes,
                    tag=2_000 + phase_index,
                )
                for source in ranks
                for destination in ranks
                if source != destination
            ),
            operation_id=f"family-d8:{operation}",
        )
        for phase_index, operation in enumerate(
            ("all_gather", "reduce_scatter")
        )
    )
    placement = _manifest(
        tuple(f"node-{rank // 8}" for rank in range(width))
    )
    return classify_step_locality(
        raw,
        rank_mapper=RankMapper(placement),
    ).phases


def _run_dense_width(
    width: int,
    workdir: Path,
    calibration: CollectiveFloorCalibration,
    *,
    operation_buffer_bytes: int,
    htsim: Path | None,
    txt2bin: Path | None,
) -> dict[str, Any]:
    workdir.mkdir(parents=True, exist_ok=False)
    topology = None
    if width > 8:
        if htsim is None or txt2bin is None:
            raise StudyUnavailable("mixed-width diagnostics require htsim and txt2bin")
        topology = _write_width_clos(workdir / f"clos-{width}-400g.topo", width=width)
    def price_phase(phase: Any) -> dict[str, Any]:
        operation = phase.phase.operation_id.rsplit(":", 1)[-1]
        fabric_ps = 0
        manifest = []
        if phase.fabric_segments:
            trace = render_fabric_phase_goal(
                phase,
                rank_mapper=RankMapper(
                    _manifest(tuple(f"node-{rank // 8}" for rank in range(width)))
                ),
            )
            stem = phase.phase.phase_id.replace(":", "-")
            goal_path = trace.write(workdir / f"{stem}.goal")
            goal_bin = workdir / f"{stem}.bin"
            completion_csv = workdir / f"{stem}.completion.csv"
            to_binary(goal_path, goal_bin, tool=txt2bin)
            run = run_htsim_rnic(
                HtsimRnicConfig(
                    goal_bin=goal_bin,
                    profile="rnic-cn",
                    linkspeed_bps=400_000_000_000,
                    completion_csv=completion_csv,
                    topology=topology,
                    extra_flags={
                        "-rnic_cn_ring_capacity_bytes": "2097152",
                        "-rnic_cn_ns_tm3_buffer_bytes": "33554432",
                    },
                ),
                binary=htsim,
                timeout_s=900,
            )
            fabric_ps = run.job_completion_time_ps()
            manifest = _artifact_manifest((goal_path, completion_csv))
        estimate = calibration.estimate(
            dtype="half",
            operation=operation,
            ranks=width,
            message_bytes=operation_buffer_bytes,
            donor=("half", operation, 8) if width != 8 else None,
        )
        physical_endpoint_estimate = calibration.estimate(
            dtype="half",
            operation=operation,
            ranks=width,
            message_bytes=phase.nvlink_peak_endpoint_bytes,
            donor=("half", operation, 8) if width != 8 else None,
        )
        current_phase_ps = max(phase.nvlink_service_ps, fabric_ps)
        calibrated_phase_ps = estimate.floor_charge_ps + max(
            estimate.serialization_ps,
            fabric_ps,
        )
        physical_endpoint_phase_ps = physical_endpoint_estimate.floor_charge_ps + max(
            physical_endpoint_estimate.serialization_ps,
            fabric_ps,
        )
        return {
            "operation": operation,
            "width": width,
            "local_endpoint_bytes": phase.nvlink_peak_endpoint_bytes,
            "operation_buffer_bytes": operation_buffer_bytes,
            "current_local_serialization_ps": phase.nvlink_service_ps,
            "calibrated_local_serialization_ps": estimate.serialization_ps,
            "aggregate_floor_ps": estimate.floor_charge_ps,
            "fabric_service_ps_unchanged": fabric_ps,
            "current_composed_phase_ps": current_phase_ps,
            "calibrated_composed_phase_ps": calibrated_phase_ps,
            "evidence_class": estimate.evidence_class,
            "transfer_reason": estimate.transfer_reason,
            "regime": estimate.regime.as_dict(),
            "physical_endpoint_reading": {
                "query_bytes": phase.nvlink_peak_endpoint_bytes,
                "calibrated_local_serialization_ps": (
                    physical_endpoint_estimate.serialization_ps
                ),
                "aggregate_floor_ps": physical_endpoint_estimate.floor_charge_ps,
                "composed_phase_ps": physical_endpoint_phase_ps,
                "evidence_class": physical_endpoint_estimate.evidence_class,
                "transfer_reason": physical_endpoint_estimate.transfer_reason,
                "regime": physical_endpoint_estimate.regime.as_dict(),
            },
            "artifacts": manifest,
        }

    phases = _dense_phases(width)
    if any(phase.fabric_segments for phase in phases):
        with ThreadPoolExecutor(max_workers=len(phases)) as executor:
            rows = list(executor.map(price_phase, phases))
        execution_mode = "parallel-independent-semantic-halves"
    else:
        rows = [price_phase(phase) for phase in phases]
        execution_mode = "analytic-local-only"
    return {
        "expert_parallel": width,
        "execution_mode": execution_mode,
        "current_dispatch_combine_ms": (
            sum(row["current_composed_phase_ps"] for row in rows)
            * REPRESENTED_LAYERS
            / 1_000_000_000
        ),
        "calibrated_dispatch_combine_ms": (
            sum(row["calibrated_composed_phase_ps"] for row in rows)
            * REPRESENTED_LAYERS
            / 1_000_000_000
        ),
        "physical_endpoint_dispatch_combine_ms": (
            sum(row["physical_endpoint_reading"]["composed_phase_ps"] for row in rows)
            * REPRESENTED_LAYERS
            / 1_000_000_000
        ),
        "phases": rows,
    }


def _family_d8(
    workdir: Path,
    calibration: CollectiveFloorCalibration,
) -> dict[str, Any]:
    minimax = json.loads(MINIMAX_RECORD.read_text(encoding="utf-8"))
    minimax_config = json.loads(MINIMAX_CONFIG.read_text(encoding="utf-8"))
    anchors = {
        row["expert_parallel"]: row
        for row in minimax["rows"]
        if row["expert_parallel"] in (8, 32, 128)
    }
    htsim = None
    txt2bin = None
    mixed_skip_reason = None
    try:
        htsim = _require_executable(HTSIM_ENV)
        txt2bin = _require_executable(TXT2BIN_ENV)
    except StudyUnavailable as error:
        mixed_skip_reason = str(error)
    tokens_per_rank = minimax_config["operating_point"][
        "local_batch_per_attention_dp_rank"
    ] * (minimax_config["model"]["nextn"] + 1)

    def operation_buffer_bytes(width: int) -> int:
        elements = tokens_per_rank * minimax_config["model"]["hidden_size"] * width
        return elements * 2

    widths = [
        _run_dense_width(
            8,
            workdir / "ep-8",
            calibration,
            operation_buffer_bytes=operation_buffer_bytes(8),
            htsim=None,
            txt2bin=None,
        )
    ]
    if mixed_skip_reason is None:
        widths.extend(
            _run_dense_width(
                width,
                workdir / f"ep-{width}",
                calibration,
                operation_buffer_bytes=operation_buffer_bytes(width),
                htsim=htsim,
                txt2bin=txt2bin,
            )
            for width in (32, 128)
        )
    for row in widths:
        frozen = anchors[row["expert_parallel"]]["family_d_packet_ms"]
        row["legacy_packet_anchor_ms"] = frozen
        row["legacy_anchor_reproduced"] = math.isclose(
            row["current_dispatch_combine_ms"], frozen, rel_tol=0, abs_tol=1e-12
        )
        row["scored"] = row["expert_parallel"] == 8
    ep8 = widths[0]
    external_ms = anchors[8]["family_d_external_ms"]
    quotient = ep8["calibrated_dispatch_combine_ms"] / external_ms
    physical_endpoint_quotient = (
        ep8["physical_endpoint_dispatch_combine_ms"] / external_ms
    )
    passed = 0.90 <= quotient <= 1.10 and ep8["legacy_anchor_reproduced"]
    return {
        "id": "D8",
        "status": "PASS" if passed else "REFUTED",
        "passed": int(passed),
        "denominator": 1,
        "band": [0.90, 1.10],
        "external_arm_ms": external_ms,
        "before_packet_ms": ep8["current_dispatch_combine_ms"],
        "before_quotient": ep8["current_dispatch_combine_ms"] / external_ms,
        "attempt_0002_wrong_query_bytes": 344_064,
        "physical_endpoint_query_bytes": ep8["phases"][0]["local_endpoint_bytes"],
        "physical_endpoint_packet_ms": ep8[
            "physical_endpoint_dispatch_combine_ms"
        ],
        "physical_endpoint_quotient": physical_endpoint_quotient,
        "matched_operation_buffer_elements": operation_buffer_bytes(8) // 2,
        "matched_operation_buffer_bytes": operation_buffer_bytes(8),
        "calibrated_packet_ms": ep8["calibrated_dispatch_combine_ms"],
        "calibrated_quotient": quotient,
        "mixed_width_skip_reason": mixed_skip_reason,
        "widths": widths,
        "physical_sanity": {
            "one_phase_serialization_floor_ps": _ceil_serialization_ps(172_032),
            "current_one_phase_ps": 383_000,
            "ceiling": "unbounded because the table exposes no algorithm progress bound",
            "calibrated_value_above_bare_floor": (
                ep8["calibrated_dispatch_combine_ms"]
                > ep8["current_dispatch_combine_ms"]
            ),
        },
    }


def _metric_dims() -> ModelDims:
    return ModelDims(
        num_layers=32,
        hidden_size=4_096,
        intermediate_size=11_008,
        num_heads=32,
        num_kv_heads=32,
        head_size=128,
        vocab_size=32_000,
        dtype_bytes=2,
    )


def _metric_arm(
    workdir: Path,
    calibration: CollectiveFloorCalibration,
    arm: str,
) -> dict[str, Any]:
    selection: dict[str, Any] = {}
    if arm == "off":
        selection = {
            "collective_floor_calibration": None,
            "collective_floor_dtype": None,
        }
    elif arm == "on":
        selection = {
            "collective_floor_calibration": calibration,
            "collective_floor_dtype": "half",
        }
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=_metric_dims(),
            workdir=workdir,
            placement_manifest=_manifest(("node", "node")),
            provider=RooflineProvider(efficiency=0.7),
            gpu=GPU_ENVELOPES["b100"],
            **selection,
        )
    )
    reducer = HtsimRequestMetricReducer({"r0": 0})
    virtual_time_ps = 0
    steps = []
    for record in _records(prompt_tokens=512, steps=3):
        record.virtual_time_ps = virtual_time_ps
        result = sink(record)
        if result is None:
            raise RuntimeError("the metric-chain fixture produced no StepResult")
        locality = sink.locality_outcomes[-1]
        attribution = attribute_step_detail(result, locality)
        reducer.consume(record, result, locality)
        steps.append(
            {
                "step_index": record.step_index,
                "step_latency_ps": result.step_latency_ps,
                "completed_at_ps": result.completed_at_ps,
                "media": _normalize(attribution.media),
                "locality": _normalize(locality),
            }
        )
        virtual_time_ps = result.completed_at_ps
    (totals,) = reducer.totals()
    metrics = _normalize(totals)
    if totals.tpot_ps is not None:
        metrics["tpot_ps"] = (
            totals.tpot_ps.numerator
            if totals.tpot_ps.denominator == 1
            else totals.tpot_ps.numerator / totals.tpot_ps.denominator
        )
        metrics["tpot_ps_exact"] = (
            f"{totals.tpot_ps.numerator}/{totals.tpot_ps.denominator}"
        )
    return {
        "steps": steps,
        "metrics": metrics,
        "floor_timing": _normalize(sink.collective_floor_timing_outcomes),
        "host_launch_floor_ps": [outcome.host_launch_floor_ps for outcome in sink.outcomes],
        "semantic_profile_selected": sink.config.resolved_collective_latency_profile
        is not None,
        "registration_enabled": sink.collective_registration_ledger.enabled,
    }


def _family_m(
    workdir: Path,
    calibration: CollectiveFloorCalibration,
) -> dict[str, Any]:
    arms = {
        "feature_absent": _metric_arm(workdir / "feature-absent", calibration, "absent"),
        "off": _metric_arm(workdir / "off", calibration, "off"),
        "on": _metric_arm(workdir / "on", calibration, "on"),
    }
    off_exact = _json_bytes(arms["feature_absent"]) == _json_bytes(arms["off"])
    off_metrics = arms["off"]["metrics"]
    on_metrics = arms["on"]["metrics"]
    ttft_delta = on_metrics["ttft_ps"] - off_metrics["ttft_ps"]
    tpot_delta = on_metrics["tpot_ps"] - off_metrics["tpot_ps"]
    step_deltas = []
    arithmetic_holds = True
    for off_step, on_step in zip(
        arms["off"]["steps"], arms["on"]["steps"], strict=True
    ):
        observed = on_step["step_latency_ps"] - off_step["step_latency_ps"]
        expected = (
            on_step["media"]["collective_floor_ps"]
            + on_step["media"]["nvlink_ps"]
            - off_step["media"]["nvlink_ps"]
        )
        arithmetic_holds = arithmetic_holds and observed == expected
        step_deltas.append(
            {
                "step_index": off_step["step_index"],
                "observed_delta_ps": observed,
                "floor_plus_serialization_replacement_ps": expected,
                "equation_holds": observed == expected,
            }
        )
    expected_tpot_delta = sum(row["observed_delta_ps"] for row in step_deltas[1:]) // 2
    passed = (
        off_exact
        and ttft_delta > 0
        and tpot_delta > 0
        and arithmetic_holds
        and ttft_delta == step_deltas[0]["observed_delta_ps"]
        and tpot_delta == expected_tpot_delta
    )
    return {
        "id": "M",
        "status": "PASS" if passed else "REFUTED",
        "passed": int(passed),
        "denominator": 1,
        "off_reproduces_feature_absent": off_exact,
        "off_ttft_ps": off_metrics["ttft_ps"],
        "on_ttft_ps": on_metrics["ttft_ps"],
        "ttft_delta_ps": ttft_delta,
        "off_tpot_ps": off_metrics["tpot_ps"],
        "on_tpot_ps": on_metrics["tpot_ps"],
        "tpot_delta_ps": tpot_delta,
        "expected_tpot_delta_ps": expected_tpot_delta,
        "step_arithmetic": step_deltas,
        "arms": arms,
        "physical_sanity": {
            "floor": "every collective remains above its true-byte serialization floor",
            "ceiling": "unbounded because the source exposes no algorithm progress bound",
            "both_signature_metrics_move_up": ttft_delta > 0 and tpot_delta > 0,
        },
    }


def _double_count_guard(
    workdir: Path,
    calibration: CollectiveFloorCalibration,
    family_m: dict[str, Any],
) -> dict[str, Any]:
    rejections = []
    selections = (
        (
            {"collective_latency_profile": B200_NCCL_2_27_LOCAL_PROFILE},
            "semantic collective base surcharge",
        ),
        (
            {"collective_registration": "nccl-channel-registration-v1"},
            "registration charge",
        ),
        (
            {"host_model": HostInitiationModel(initiation_delay_ps=800)},
            "host launch model",
        ),
        (
            {"nvlink_bandwidth_bytes_per_second": 900_000_000_000},
            "second NVLink rate",
        ),
    )
    for kwargs, label in selections:
        try:
            HtsimStepSinkConfig(
                profile="rnic-nn-fluid",
                tp_ranks=(0, 1),
                dims=_bypass_dims(),
                workdir=workdir / label.replace(" ", "-"),
                placement_manifest=_manifest(("node", "node")),
                collective_floor_calibration=calibration,
                collective_floor_dtype="half",
                **kwargs,
            )
        except ValueError as error:
            rejections.append({"selection": label, "rejected": True, "reason": str(error)})
        else:
            rejections.append({"selection": label, "rejected": False, "reason": None})
    on = family_m["arms"]["on"]
    active_projection_clean = (
        not on["semantic_profile_selected"]
        and not on["registration_enabled"]
        and all(value == 0 for value in on["host_launch_floor_ps"])
        and all(
            not any(step["locality"]["base_phase_latency_ps"])
            and step["locality"]["registration_phase_cost_ps"] == []
            for step in on["steps"]
        )
    )
    return {
        "rejections": rejections,
        "active_projection_has_no_duplicate_charge": active_projection_clean,
        "held": all(row["rejected"] for row in rejections)
        and active_projection_clean,
    }


def _consumer_transfer_guard(
    workdir: Path,
    calibration: CollectiveFloorCalibration,
) -> dict[str, Any]:
    record = StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                "r0",
                RequestPhase.PREFILL,
                262_145,
                context_length=262_145,
            )
        ],
        num_sampled=1,
    )
    common = {
        "profile": "rnic-nn-fluid",
        "tp_ranks": tuple(range(8)),
        "dims": _bypass_dims(),
        "placement_manifest": _manifest(("node",) * 8),
        "collective_floor_calibration": calibration,
        "collective_floor_dtype": "half",
    }
    refused = HtsimStepSink(
        HtsimStepSinkConfig(workdir=workdir / "refused", **common)
    )
    refusal = None
    try:
        refused(record)
    except CollectiveFloorTransferError as error:
        refusal = {
            "error_type": type(error).__name__,
            "message": str(error),
        }
    acknowledged = HtsimStepSink(
        HtsimStepSinkConfig(
            workdir=workdir / "acknowledged",
            acknowledge_collective_floor_transfer=True,
            **common,
        )
    )
    result = acknowledged(record)
    timing = (
        acknowledged.collective_floor_timing_outcomes[0]
        if acknowledged.collective_floor_timing_outcomes
        else None
    )
    transferred = (
        []
        if timing is None
        else [
            artifact.estimate.evidence_class
            for artifact in timing.artifacts
        ]
    )
    held = (
        refusal is not None
        and "acknowledge_collective_floor_transfer=True" in refusal["message"]
        and refused.outcomes == []
        and refused.collective_floor_timing_outcomes == []
        and result is not None
        and timing is not None
        and timing.transferred_at_use_acknowledged
        and transferred
        and all(value == COLLECTIVE_FLOOR_TRANSFERRED for value in transferred)
    )
    return {
        "held": bool(held),
        "default_path": {
            "refused": refusal is not None,
            "error": refusal,
            "published_outcomes": len(refused.outcomes),
        },
        "acknowledged_path": {
            "completed_at_ps": None if result is None else result.completed_at_ps,
            "outcome_stamped": (
                False if timing is None else timing.transferred_at_use_acknowledged
            ),
            "artifact_evidence_classes": transferred,
        },
    }


def _evidence_guard(
    workdir: Path,
    calibration: CollectiveFloorCalibration,
) -> dict[str, Any]:
    exact = calibration.estimate(
        dtype="half", operation="all_gather", ranks=8, message_bytes=196_608
    )
    transfers = (
        calibration.estimate(
            dtype="half",
            operation="all_reduce",
            ranks=8,
            message_bytes=196_608,
            donor=("half", "all_gather", 8),
        ),
        calibration.estimate(
            dtype="int8",
            operation="all_gather",
            ranks=8,
            message_bytes=196_608,
            donor=("half", "all_gather", 8),
        ),
        calibration.estimate(
            dtype="half",
            operation="all_gather",
            ranks=32,
            message_bytes=196_608,
            donor=("half", "all_gather", 8),
        ),
        calibration.estimate(
            dtype="half",
            operation="all_gather",
            ranks=8,
            message_bytes=calibration.fitted_byte_range[1] + 1,
        ),
    )
    consumer = _consumer_transfer_guard(workdir / "consumer", calibration)
    held = exact.evidence_class == COLLECTIVE_FLOOR_CALIBRATED and all(
        estimate.evidence_class == COLLECTIVE_FLOOR_TRANSFERRED
        and estimate.transfer_reason
        for estimate in transfers
    ) and consumer["held"]
    return {
        "held": held,
        "exact": exact.as_dict(),
        "transfers": [estimate.as_dict() for estimate in transfers],
        "consumer_fence": consumer,
    }


def _chronology_guard() -> dict[str, Any]:
    def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=REPOSITORY_ROOT,
            check=check,
            capture_output=True,
            text=True,
        )

    config_before_implementation = (
        git("merge-base", "--is-ancestor", CONFIG_COMMIT, IMPLEMENTATION_COMMIT, check=False).returncode
        == 0
    )
    config_payload = git(
        "show",
        f"{CONFIG_COMMIT}:examples/collective_floor_calibration_v1/study_config.json",
    ).stdout.encode()
    fit_absent_before_config = (
        git(
            "cat-file",
            "-e",
            f"{CONFIG_COMMIT}^:simllm/traffic/collective_floor.py",
            check=False,
        ).returncode
        != 0
    )
    implementation_present = (
        git(
            "cat-file",
            "-e",
            f"{IMPLEMENTATION_COMMIT}:simllm/traffic/collective_floor.py",
            check=False,
        ).returncode
        == 0
    )
    repair_implementation_commit = git("rev-parse", "HEAD").stdout.strip()
    coordinate_freeze_payload = git(
        "show",
        f"{COORDINATE_FREEZE_COMMIT}:examples/collective_floor_calibration_v1/expectations_v3.md",
    ).stdout.encode()
    coordinate_freeze_before_repair = (
        git(
            "merge-base",
            "--is-ancestor",
            COORDINATE_FREEZE_COMMIT,
            repair_implementation_commit,
            check=False,
        ).returncode
        == 0
    )
    return {
        "configuration_commit": CONFIG_COMMIT,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "config_before_implementation": config_before_implementation,
        "config_sha256_at_commit": _sha256_bytes(config_payload),
        "config_sha256_current": _sha256_file(CONFIG_PATH),
        "fit_absent_before_configuration": fit_absent_before_config,
        "implementation_present_at_implementation_commit": implementation_present,
        "coordinate_mapping_freeze_commit": COORDINATE_FREEZE_COMMIT,
        "coordinate_freeze_sha256_at_commit": _sha256_bytes(
            coordinate_freeze_payload
        ),
        "coordinate_freeze_sha256_current": _sha256_file(
            STUDY_DIR / "expectations_v3.md"
        ),
        "coordinate_freeze_before_repair": coordinate_freeze_before_repair,
        "repair_implementation_commit": repair_implementation_commit,
        "held": config_before_implementation
        and _sha256_bytes(config_payload) == _sha256_file(CONFIG_PATH)
        and fit_absent_before_config
        and implementation_present
        and coordinate_freeze_before_repair
        and _sha256_bytes(coordinate_freeze_payload)
        == _sha256_file(STUDY_DIR / "expectations_v3.md"),
    }


def _evaluate(output_dir: Path) -> dict[str, Any]:
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=False)
    config = _load_config()
    database = ExternalNcclDatabase.load()
    calibration, training = _fit(config, database)
    axis = _axis_check(config, database)
    family_h = _family_h(config, database, calibration)
    family_b = _family_b(output_dir / "family-b")
    family_d8 = _family_d8(output_dir / "family-d8", calibration)
    family_m = _family_m(output_dir / "family-m", calibration)
    wall_time_seconds = time.perf_counter() - start
    family_w = {
        "id": "W",
        "status": "PASS" if wall_time_seconds <= WALL_BOUND_SECONDS else "FAIL",
        "passed": int(wall_time_seconds <= WALL_BOUND_SECONDS),
        "denominator": 1,
        "bound_seconds": WALL_BOUND_SECONDS,
        "wall_time_seconds": wall_time_seconds,
    }

    exact_terms = all(
        set(regime.as_dict())
        == {
            "dtype",
            "operation",
            "ranks",
            "regime_index",
            "lower_bytes",
            "upper_bytes",
            "floor_ps",
            "slope_ps_per_byte",
            "effective_bandwidth_bytes_per_second",
            "training_cell_ids",
        }
        for regime in calibration.regimes
    )
    double_count = _double_count_guard(
        output_dir / "double-count", calibration, family_m
    )
    evidence = _evidence_guard(output_dir / "evidence", calibration)
    a100_inputs = {
        "packet_geometry",
        "credits",
        "link_count",
        "link_rate",
        "switch_buffer",
        "arbitration",
        "a100_candidate_profile",
    }
    a100_fence = a100_inputs.isdisjoint(calibration.input_surface)
    chronology = _chronology_guard()
    record_chronology = json.loads(json.dumps(config["chronology"]))
    record_chronology["attempt_0001"]["artifact_status"] = (
        "No attempt-0001 directory exists by construction because the worker "
        "stopped at the axis check before creating any artifact; the worker "
        "report is the only evidence."
    )
    record_chronology["coordinate_mapping_freeze_commit"] = (
        COORDINATE_FREEZE_COMMIT
    )
    record_chronology["attempt_0002"] = {
        "status": "SUPERSEDED",
        "findings": [
            "D8 doubled already physical endpoint bytes and queried 344064 bytes instead of the matched 196608-byte operation buffer.",
            "Family B compared two post-wave all-remote runs instead of the pre-wave mixed-locality path.",
            "Family H labeled a bare physical floor as the current ring implementation.",
            "The production consumer allowed transferred-at-use timing into signature metrics without explicit acknowledgement.",
        ],
    }
    record_chronology["attempt_0003"] = {
        "status": "VOID",
        "finding": (
            "The second fresh evaluation took 657.1472301706672 seconds, "
            "above the unchanged 600-second Family W ceiling; the first took "
            "564.1278964616358 seconds, so W also made FG-6 differ."
        ),
        "interpretation": (
            "The corrected H, B, D8, and M findings are retained but cannot "
            "support publication from this void attempt."
        ),
    }
    bypass_held = family_b["status"] == "PASS"
    guards = [
        {
            "id": "FG-1",
            "claim": "no invented terms",
            "held": exact_terms and axis["passed"],
            "evaluated": "generated regime records contain only source identity, boundaries, fitted floors, fitted byte slopes and derived bandwidth",
        },
        {
            "id": "FG-2",
            "claim": "no double counting",
            "held": double_count["held"],
            "evaluated": double_count,
        },
        {
            "id": "FG-3",
            "claim": "evidence classes",
            "held": evidence["held"],
            "evaluated": evidence,
        },
        {
            "id": "FG-4",
            "claim": "exact bypass",
            "held": bypass_held,
            "evaluated": {
                "family": "B",
                "status": family_b["status"],
                "skip_reason": family_b.get("skip_reason"),
            },
        },
        {
            "id": "FG-5",
            "claim": "A100 fence",
            "held": a100_fence,
            "evaluated": {
                "calibration_input_surface": list(calibration.input_surface),
                "prohibited_inputs": sorted(a100_inputs),
            },
        },
        {
            "id": "FG-7",
            "claim": "chronology",
            "held": chronology["held"],
            "evaluated": chronology,
        },
    ]
    return {
        "schema": SCHEMA,
        "study": config["study"],
        "chronology": record_chronology,
        "axis": axis,
        "source": config["source"],
        "clock": config["clock"],
        "fit": {
            "calibration_id": calibration.calibration_id,
            "fitted_byte_range": list(calibration.fitted_byte_range),
            "input_surface": list(calibration.input_surface),
            "training_cells": len(training),
            "holdout_cells": len(config["membership"]["holdout_cells"]),
            "regimes": [regime.as_dict() for regime in calibration.regimes],
        },
        "fatal_guards_without_determinism": guards,
        "families": {
            "H": family_h,
            "B": family_b,
            "D8": family_d8,
            "M": family_m,
            "W": family_w,
        },
        "environment": {
            "machine": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "htsim": _binary_identity(HTSIM_ENV),
            "txt2bin": _binary_identity(TXT2BIN_ENV),
            "external_aiconfigurator_environment_configured": bool(
                os.environ.get(EXTERNAL_VENV_ENV)
            ),
        },
    }


def _csv_bytes(record: dict[str, Any]) -> bytes:
    columns = (
        "family",
        "instance",
        "status",
        "measured_ps",
        "before_ps",
        "calibrated_ps",
        "before_relative_error",
        "after_relative_error",
        "observed",
        "lower_bound",
        "upper_bound",
        "passed",
    )
    rows = []
    for row in record["families"]["H"]["rows"]:
        rows.append(
            {
                "family": "H",
                "instance": row["cell_id"],
                "status": "PASS" if row["passed"] else "FAIL",
                "measured_ps": row["measured_ps"],
                "before_ps": (
                    row["current_ring_ps"]
                    if "current_ring_ps" in row
                    else row["bare_serialization_ps"]
                ),
                "calibrated_ps": row["calibrated_ps"],
                "before_relative_error": row["before_relative_error"],
                "after_relative_error": row["after_relative_error"],
                "observed": row["after_relative_error"],
                "lower_bound": 0,
                "upper_bound": row["tolerance"],
                "passed": row["passed"],
            }
        )
    for family, instance, observed, lower, upper in (
        ("B", "byte-exact-bypass", record["families"]["B"]["passed"], 1, 1),
        (
            "D8",
            "ep-8-quotient",
            record["families"]["D8"]["calibrated_quotient"],
            0.90,
            1.10,
        ),
        ("M", "supported-metric-chain", record["families"]["M"]["passed"], 1, 1),
        (
            "W",
            "wall-time-seconds",
            record["families"]["W"]["wall_time_seconds"],
            0,
            WALL_BOUND_SECONDS,
        ),
    ):
        status = record["families"][family]["status"]
        rows.append(
            {
                "family": family,
                "instance": instance,
                "status": status,
                "observed": observed,
                "lower_bound": lower,
                "upper_bound": upper,
                "passed": record["families"][family]["passed"],
            }
        )
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _coordinator(workdir: Path, *, write_tracked: bool) -> dict[str, Any]:
    if workdir.exists():
        raise SystemExit(f"refusing to overwrite append-only attempt {workdir}")
    workdir.mkdir(parents=True)
    evaluations = []
    for label in ("evaluation-1", "evaluation-2"):
        output = workdir / f"{label}.json"
        completed = subprocess.run(
            [
                sys.executable,
                os.fspath(Path(__file__).resolve()),
                "--internal-output",
                os.fspath(output),
                "--internal-workdir",
                os.fspath(workdir / label),
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=1_800,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"{label} failed: {detail}")
        evaluations.append(json.loads(output.read_text(encoding="utf-8")))
    deterministic_a = _json_bytes(_without_wall_time(evaluations[0]))
    deterministic_b = _json_bytes(_without_wall_time(evaluations[1]))
    deterministic = deterministic_a == deterministic_b
    record = evaluations[0]
    wall = max(
        evaluation["families"]["W"]["wall_time_seconds"]
        for evaluation in evaluations
    )
    record["families"]["W"]["wall_time_seconds"] = wall
    record["families"]["W"]["passed"] = int(wall <= WALL_BOUND_SECONDS)
    record["families"]["W"]["status"] = (
        "PASS" if wall <= WALL_BOUND_SECONDS else "FAIL"
    )
    fg6 = {
        "id": "FG-6",
        "claim": "determinism",
        "held": deterministic,
        "evaluated": {
            "fresh_processes": 2,
            "excluded_field_name": "wall_time_seconds",
            "evaluation_1_sha256": _sha256_bytes(deterministic_a),
            "evaluation_2_sha256": _sha256_bytes(deterministic_b),
        },
    }
    guards = record.pop("fatal_guards_without_determinism")
    guards.insert(5, fg6)
    record["fatal_guards"] = guards
    record["verdict"] = (
        "interpretable" if all(guard["held"] for guard in guards) else "VOID"
    )
    attempt_key = workdir.name.replace("-", "_")
    record["chronology"][attempt_key] = {
        "status": record["verdict"].upper(),
        "fresh_process_evaluations": 2,
        "coordinate_mapping_freeze_commit": COORDINATE_FREEZE_COMMIT,
        "repair_implementation_commit": guards[-1]["evaluated"][
            "repair_implementation_commit"
        ],
    }
    record["family_tallies"] = {
        family: {
            "status": payload["status"],
            "passed": payload["passed"],
            "denominator": payload["denominator"],
        }
        for family, payload in record["families"].items()
    }
    record_bytes = _json_bytes(record)
    csv_bytes = _csv_bytes(record)
    (workdir / "record.json").write_bytes(record_bytes)
    (workdir / "results.csv").write_bytes(csv_bytes)
    if b"\r" in record_bytes or b"\r" in csv_bytes:
        raise RuntimeError("generated artifacts do not use LF line endings")
    if write_tracked:
        TRACKED_RECORD.write_bytes(record_bytes)
        TRACKED_CSV.write_bytes(csv_bytes)
    return record


def _check() -> None:
    if not TRACKED_RECORD.is_file() or not TRACKED_CSV.is_file():
        raise SystemExit("tracked record.json and results.csv are required")
    record_bytes = TRACKED_RECORD.read_bytes()
    csv_bytes = TRACKED_CSV.read_bytes()
    if b"\r" in record_bytes or b"\r" in csv_bytes:
        raise SystemExit("tracked study artifacts must use LF line endings")
    record = json.loads(record_bytes)
    if record.get("schema") != SCHEMA:
        raise SystemExit("record.json has an unsupported schema")
    if _csv_bytes(record) != csv_bytes:
        raise SystemExit("results.csv has drifted from record.json")
    if record["verdict"] != "interpretable":
        raise SystemExit("the corrected study record is void")
    print(
        "record and CSV are current; "
        f"record_sha256={_sha256_bytes(record_bytes)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-write-tracked", action="store_true")
    parser.add_argument("--internal-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--internal-workdir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.internal_output is not None:
        if args.internal_workdir is None:
            parser.error("--internal-output requires --internal-workdir")
        evaluation = _evaluate(args.internal_workdir)
        args.internal_output.write_bytes(_json_bytes(evaluation))
        return 0
    if args.check:
        _check()
        return 0
    if args.workdir is None:
        parser.error("--workdir is required unless --check is selected")
    record = _coordinator(
        args.workdir,
        write_tracked=not args.no_write_tracked,
    )
    print(
        f"verdict={record['verdict']} "
        + " ".join(
            f"{family}={payload['status']}"
            for family, payload in record["family_tallies"].items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
