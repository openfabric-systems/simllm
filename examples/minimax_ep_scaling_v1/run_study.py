# SPDX-FileCopyrightText: Copyright (c) 2026 Yifeng Wang
# SPDX-License-Identifier: Apache-2.0
"""Run the frozen MiniMax-M2.5 expert-parallel scaling study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

CONFIG_PATH = STUDY / "study_config.json"
EXPECTATIONS_PATH = STUDY / "expectations.md"
CORRECTED_EXPECTATIONS_PATH = STUDY / "expectations_v2.md"
TRACKED_RECORD = STUDY / "record.json"
TRACKED_CSV = STUDY / "results.csv"
TRACKED_FIGURES = STUDY / "figures"

BULK_ROOT_ENV = "SIMLLM_MINIMAX_FIX_BULK_ROOT"
EXTERNAL_VENV_ENV = "SIMLLM_EXTERNAL_AIC_VENV"
HTSIM_ENV = "SIMLLM_HTSIM_RNIC"
TXT2BIN_ENV = "SIMLLM_TXT2BIN"
SCHEMA = "simllm-minimax-ep-scaling-record-v3"
LEGACY_SCHEMA = "simllm-minimax-ep-scaling-record-v1"
SUPPORTED_RECORD_SCHEMAS = {
    "simllm-minimax-ep-scaling-record-v2",
    SCHEMA,
}
WALL_BOUND_SECONDS = 3600.0
EXPECTED_FREEZE_COMMITS = ("61b66c4", "5a29bb0", "4d1e41c")
PDF_TEXT_TOOL = "pdftotext"


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


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _is_ancestor(older: str, newer: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", older, newer],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )


def _configured_path(name: str) -> Path:
    raw = os.environ.get(name)
    if not raw:
        raise SystemExit(f"set {name} to the required executable or environment")
    path = Path(raw)
    if name == EXTERNAL_VENV_ENV:
        candidates = (path / "bin/python", path / "Scripts/python.exe")
        if not any(candidate.is_file() for candidate in candidates):
            raise SystemExit(f"{name} does not name a usable Python environment")
        return path
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"{name} does not name an executable file")
    return path


def _venv_python(venv_root: Path) -> Path:
    for candidate in (venv_root / "bin/python", venv_root / "Scripts/python.exe"):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"{EXTERNAL_VENV_ENV} has no usable Python interpreter")


def _new_attempt(root: Path) -> tuple[Path, int]:
    root.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in root.iterdir():
        match = re.fullmatch(r"attempt-(\d{4})", path.name)
        if path.is_dir() and match:
            numbers.append(int(match.group(1)))
    number = max(numbers, default=0) + 1
    attempt = root / f"attempt-{number:04d}"
    attempt.mkdir()
    return attempt, number


def _load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("schema") != "simllm-minimax-ep-scaling-study-config-v2":
        raise SystemExit("study_config.json has an unsupported schema")
    if [row["expert_parallel"] for row in config["widths"]] != [8, 32, 128, 256]:
        raise SystemExit("study_config.json does not carry the frozen width sweep")
    if config["model"].get("nextn") != 3:
        raise SystemExit("the faithful study requires explicit nextn=3")
    sampling = config["packet_sampling"]
    if sampling.get("dense_widest_anchor_width") != 128:
        raise SystemExit("the corrected study requires the frozen EP 128 dense anchor")
    if "(256 - 8) / (128 - 8) = 31 / 15" not in sampling.get(
        "dense_widest_extrapolation_rule", ""
    ):
        raise SystemExit("the post-specified dense diagnostic rule is missing")
    if "multiply only the EP 128 fabric service" not in sampling.get(
        "dense_widest_corrected_extrapolation_rule", ""
    ):
        raise SystemExit("the corrected component-wise diagnostic rule is missing")
    chronology = config["chronology"]
    if chronology.get("diagnostic_extrapolation_rule_commit") != "a6ba97f":
        raise SystemExit("the diagnostic extrapolation chronology is missing")
    if chronology.get("diagnostic_extrapolation_rule_frozen") is not False:
        raise SystemExit("the diagnostic extrapolation must remain explicitly unfrozen")
    if sampling.get("dense_widest_extrapolation_scored") is not False:
        raise SystemExit("the post-specified EP 256 extrapolation cannot be scored")
    correction = config["collective_floor_correction"]
    for field in ("source_config", "source_record"):
        path = ROOT / correction[field]
        if not path.is_file():
            raise SystemExit(f"collective-floor {field} does not exist")
        expected = correction[f"{field}_sha256"]
        if _sha256_file(path) != expected:
            raise SystemExit(f"collective-floor {field} digest differs from the pin")
    if correction.get("acknowledge_transferred_at_use") is not True:
        raise SystemExit("the transferred collective-floor correction must be explicit")
    return config


def _load_collective_floor_calibration(
    config: dict[str, Any],
    database: Any,
) -> Any:
    """Recreate the landed fitted authority from its immutable training inputs."""

    from simllm.traffic import (
        CollectiveFloorCell,
        CollectiveFloorCurveBoundaries,
        CollectiveFloorSourceIdentity,
        fit_collective_floor_calibration,
    )

    correction = config["collective_floor_correction"]
    calibration_config = json.loads(
        (ROOT / correction["source_config"]).read_text(encoding="utf-8")
    )
    training = []
    for member in calibration_config["membership"]["training_cells"]:
        observed = database.query(
            dtype=member["dtype"],
            operation=member["operation"],
            ranks=member["ranks"],
            message_size=member["source_elements"],
        )
        training.append(
            CollectiveFloorCell(
                cell_id=member["cell_id"],
                dtype=member["dtype"],
                operation=member["operation"],
                ranks=member["ranks"],
                source_elements=member["source_elements"],
                message_bytes=member["true_bytes"],
                latency_ps=round(observed.latency_ms * 1_000_000_000),
            )
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
        for row in calibration_config["fit"]["regime_boundaries_true_bytes"]
    )
    byte_range = calibration_config["fit"]["true_byte_range"]
    return fit_collective_floor_calibration(
        calibration_id=correction["calibration_id"],
        source=CollectiveFloorSourceIdentity(**calibration_config["source"]),
        cells=tuple(training),
        boundaries=boundaries,
        fitted_byte_range=(byte_range["minimum"], byte_range["maximum"]),
    )


def _collective_floor_estimate(
    calibration: Any,
    *,
    dtype: str,
    operation: str,
    ranks: int,
    message_bytes: int,
    donor: tuple[str, str, int] | None,
    acknowledge_transfer: bool,
) -> Any:
    """Serve one estimate and apply the production transfer refusal contract."""

    from simllm.backends import CollectiveFloorTransferError
    from simllm.traffic import COLLECTIVE_FLOOR_TRANSFERRED

    estimate = calibration.estimate(
        dtype=dtype,
        operation=operation,
        ranks=ranks,
        message_bytes=message_bytes,
        donor=donor,
    )
    if (
        estimate.evidence_class == COLLECTIVE_FLOOR_TRANSFERRED
        and not acknowledge_transfer
    ):
        raise CollectiveFloorTransferError(
            "MiniMax packet pricing refuses transferred-at-use collective-floor "
            f"timing for {operation}, ranks={ranks}, message_bytes={message_bytes}: "
            f"{estimate.transfer_reason}. Set acknowledge_transferred_at_use=true "
            "only for a deliberate transferred study."
        )
    return estimate


def _run_live_sdk(config: dict[str, Any]) -> dict[str, Any]:
    import aiconfigurator_core
    from aiconfigurator_core.sdk import common
    from aiconfigurator_core.sdk.backends.factory import get_backend
    from aiconfigurator_core.sdk.config import ModelConfig, RuntimeConfig
    from aiconfigurator_core.sdk.models import get_model
    from aiconfigurator_core.sdk.perf_database import get_database_view

    del aiconfigurator_core
    source = config["source"]
    operating = config["operating_point"]
    database = get_database_view(
        system=source["system"],
        backend=source["backend"],
        version=source["database_version"],
        database_mode=common.DatabaseMode.SILICON,
        shared_layer=False,
        strict_provenance=False,
    )
    backend = get_backend(source["backend"])
    runtime = RuntimeConfig(
        batch_size=operating["local_batch_per_attention_dp_rank"],
        beam_width=1,
        isl=operating["input_length"],
        osl=operating["output_length"],
        prefix=0,
        seq_imbalance_correction_scale=1.0,
        gen_seq_imbalance_correction_scale=1.0,
    )
    widths = []
    for frozen in config["widths"]:
        width = int(frozen["expert_parallel"])
        model_config = ModelConfig(
            tp_size=1,
            pp_size=1,
            moe_tp_size=1,
            moe_ep_size=width,
            attention_dp_size=width,
            workload_distribution=operating["workload_distribution"],
            nextn=3,
        )
        model = get_model(config["model"]["model_id"], model_config, source["backend"])
        _, _, generation, _, _, generation_sources = backend._run_static_breakdown(
            model,
            database,
            runtime,
            "static_gen",
            operating["generation_stride"],
            1.0,
            include_energy=False,
        )
        pre = generation["generation_moe_pre_dispatch"]
        post = generation["generation_moe_post_dispatch"]
        total = sum(generation.values())
        widths.append(
            {
                "expert_parallel": width,
                "decode_step_ms": total,
                "decode_step_hex": total.hex(),
                "pre_dispatch_ms": pre,
                "pre_dispatch_hex": pre.hex(),
                "post_dispatch_ms": post,
                "post_dispatch_hex": post.hex(),
                "dispatch_ms": pre + post,
                "dispatch_hex": (pre + post).hex(),
                "operation_sources": dict(sorted(generation_sources.items())),
            }
        )
    return {
        "package_versions": {
            "aiconfigurator": importlib.metadata.version("aiconfigurator"),
            "aiconfigurator-core": importlib.metadata.version(
                "aiconfigurator-core"
            ),
        },
        "widths": widths,
    }


def _live_sdk_subprocess(config: dict[str, Any]) -> dict[str, Any]:
    venv_root = _configured_path(EXTERNAL_VENV_ENV)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [os.fspath(_venv_python(venv_root)), os.fspath(Path(__file__)), "--live-sdk-worker"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "live SDK worker failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("live SDK worker returned no JSON")
    result = json.loads(lines[-1])
    frozen_by_width = {
        int(row["expert_parallel"]): row for row in config["widths"]
    }
    for row in result["widths"]:
        frozen = frozen_by_width[int(row["expert_parallel"])]
        if row["decode_step_hex"] != frozen["live_decode_step_hex"]:
            raise RuntimeError("live SDK decode value differs from the frozen oracle")
        if row["dispatch_hex"] != frozen["live_dispatch_hex"]:
            raise RuntimeError("live SDK dispatch value differs from the frozen oracle")
    return result


def _max_receiver_fanin(flows: list[Any]) -> dict[str, int]:
    by_destination: dict[int, list[Any]] = defaultdict(list)
    for flow in flows:
        by_destination[flow.destination].append(flow)
    best = {
        "receiver": -1,
        "ingress_occupancy_ps": 0,
        "maximum_simultaneous_senders": 0,
        "sender_count": 0,
    }
    for destination, inbound in sorted(by_destination.items()):
        intervals = sorted(
            (flow.start_time_ps, flow.completion_time_ps, flow.source)
            for flow in inbound
        )
        merged: list[list[int]] = []
        for start, finish, _ in intervals:
            if not merged or start > merged[-1][1]:
                merged.append([start, finish])
            else:
                merged[-1][1] = max(merged[-1][1], finish)
        occupancy = sum(finish - start for start, finish in merged)
        events = []
        for start, finish, source in intervals:
            events.append((start, 1, source))
            events.append((finish, -1, source))
        active: dict[int, int] = defaultdict(int)
        maximum = 0
        for _, kind, source in sorted(events, key=lambda item: (item[0], item[1])):
            if kind < 0:
                active[source] -= 1
                if active[source] == 0:
                    del active[source]
            else:
                active[source] += 1
                maximum = max(maximum, len(active))
        candidate = {
            "receiver": destination,
            "ingress_occupancy_ps": occupancy,
            "maximum_simultaneous_senders": maximum,
            "sender_count": len({flow.source for flow in inbound}),
        }
        if (
            candidate["maximum_simultaneous_senders"],
            candidate["ingress_occupancy_ps"],
            -candidate["receiver"],
        ) > (
            best["maximum_simultaneous_senders"],
            best["ingress_occupancy_ps"],
            -best["receiver"],
        ):
            best = candidate
    return best


def _completion_geometry(
    flows: list[Any],
    *,
    local_segments: tuple[Any, ...],
    width: int,
) -> dict[str, Any]:
    """Reconstruct completed routing geometry from backend and local completions."""

    destinations_by_source: dict[int, set[int]] = defaultdict(set)
    fabric_senders_by_receiver: dict[int, set[int]] = defaultdict(set)
    for flow in flows:
        destinations_by_source[flow.source].add(flow.destination)
        fabric_senders_by_receiver[flow.destination].add(flow.source)
    for segment in local_segments:
        destinations_by_source[segment.source_rank].add(segment.destination_rank)
    return {
        "completed_distinct_destinations_per_source": sum(
            len(destinations_by_source[source]) for source in range(width)
        )
        / width,
        "completed_cross_node_senders_per_receiver": sum(
            len(fabric_senders_by_receiver[destination])
            for destination in range(width)
        )
        / width,
        "maximum_completed_cross_node_senders_per_receiver": max(
            map(len, fabric_senders_by_receiver.values()), default=0
        ),
        "source": (
            "simulator completion rows for cross-node flows plus analytically "
            "completed same-node segments"
        ),
    }


def _write_width_clos(path: Path, *, width: int, gpus_per_node: int) -> Path:
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


def _dense_floor_requests(
    config: dict[str, Any],
    *,
    width: int,
) -> tuple[dict[str, Any], ...]:
    correction = config["collective_floor_correction"]
    tokens_per_rank = config["operating_point"][
        "local_batch_per_attention_dp_rank"
    ] * (config["model"]["nextn"] + 1)
    operation_buffer_bytes = (
        tokens_per_rank * config["model"]["hidden_size"] * width * 2
    )
    return tuple(
        {
            "dtype": "half",
            "operation": operation,
            "message_bytes": operation_buffer_bytes,
            "donor": (
                None
                if width == correction["donor_ranks"]
                else ("half", operation, correction["donor_ranks"])
            ),
        }
        for operation in ("all_gather", "reduce_scatter")
    )


def _sparse_floor_requests(
    config: dict[str, Any],
    *,
    width: int,
) -> tuple[dict[str, Any], ...]:
    correction = config["collective_floor_correction"]
    model = config["model"]
    tokens_per_rank = config["operating_point"][
        "local_batch_per_attention_dp_rank"
    ] * (model["nextn"] + 1)
    elements = tokens_per_rank * model["num_experts_per_tok"] * model["hidden_size"]
    requests = []
    for name, byte_width in (
        ("sparse_dispatch_transfer", 1),
        ("sparse_combine_transfer", 2),
    ):
        mapping = correction[name]
        requests.append(
            {
                "dtype": mapping["requested_dtype"],
                "operation": mapping["requested_operation"],
                "message_bytes": elements * byte_width,
                "donor": (
                    mapping["donor_dtype"],
                    mapping["donor_operation"],
                    correction["donor_ranks"],
                ),
            }
        )
    return tuple(requests)


def _simulate_packet_phases(
    config: dict[str, Any],
    *,
    width: int,
    output_dir: Path,
    htsim: Path,
    txt2bin: Path,
    phases: tuple[Any, ...],
    floor_requests: tuple[dict[str, Any], ...],
    calibration: Any,
) -> dict[str, Any]:
    from simllm.backends import HtsimRnicConfig, run_htsim_rnic
    from simllm.goal import to_binary
    from simllm.placement import RankMapper, declared_manifest
    from simllm.traffic import render_fabric_phase_goal

    operating = config["operating_point"]
    output_dir.mkdir(parents=True)
    placement = declared_manifest(
        tp=1,
        pp=1,
        dp=width,
        nodes=width // operating["gpus_per_node"],
        gpus_per_node=operating["gpus_per_node"],
    )
    mapper = RankMapper(placement)
    topology = _write_width_clos(
        output_dir / f"clos-{width}-400g.topo",
        width=width,
        gpus_per_node=operating["gpus_per_node"],
    )
    results = []
    if len(floor_requests) != len(phases):
        raise ValueError("collective-floor requests must match packet phases")
    acknowledge_transfer = bool(
        config["collective_floor_correction"]["acknowledge_transferred_at_use"]
    )
    for classified, floor_request in zip(phases, floor_requests, strict=True):
        phase_name = classified.phase.phase_id.replace(":", "-")
        flow_rows = []
        fabric_jct_ps = 0
        goal_sha256 = None
        goal_binary_sha256 = None
        completion_sha256 = None
        if classified.fabric_segments:
            trace = render_fabric_phase_goal(classified, rank_mapper=mapper)
            goal_path = trace.write(output_dir / f"{phase_name}.goal")
            goal_bin = output_dir / f"{phase_name}.bin"
            completion_csv = output_dir / f"{phase_name}.completion.csv"
            to_binary(goal_path, goal_bin, tool=txt2bin)
            run = run_htsim_rnic(
                HtsimRnicConfig(
                    goal_bin=goal_bin,
                    profile=operating["packet_profile"],
                    linkspeed_bps=operating["link_rate_bits_per_second"],
                    completion_csv=completion_csv,
                    topology=topology,
                    extra_flags={
                        "-rnic_cn_ring_capacity_bytes": str(
                            operating["ring_cam_wire_capacity_bytes"]
                        ),
                        "-rnic_cn_ns_tm3_buffer_bytes": str(
                            operating["switch_shared_buffer_bytes"]
                        ),
                    },
                ),
                binary=htsim,
                timeout_s=900,
            )
            flow_rows = run.flows
            fabric_jct_ps = run.job_completion_time_ps()
            goal_sha256 = _sha256_file(goal_path)
            goal_binary_sha256 = _sha256_file(goal_bin)
            completion_sha256 = _sha256_file(completion_csv)
        estimate = _collective_floor_estimate(
            calibration,
            dtype=floor_request["dtype"],
            operation=floor_request["operation"],
            ranks=width,
            message_bytes=floor_request["message_bytes"],
            donor=floor_request["donor"],
            acknowledge_transfer=acknowledge_transfer,
        )
        uncalibrated_phase_duration_ps = max(
            fabric_jct_ps,
            classified.nvlink_service_ps,
        )
        calibrated_phase_duration_ps = estimate.floor_charge_ps + max(
            fabric_jct_ps,
            estimate.serialization_ps,
        )
        results.append(
            {
                "phase": phase_name,
                "fabric_segments": len(classified.fabric_segments),
                "fabric_bytes": classified.fabric_bytes,
                "nvlink_segments": len(classified.nvlink_segments),
                "nvlink_bytes": classified.nvlink_bytes,
                "nvlink_service_ps": classified.nvlink_service_ps,
                "fabric_jct_ps": fabric_jct_ps,
                "uncalibrated_phase_duration_ps": uncalibrated_phase_duration_ps,
                "phase_duration_ps": calibrated_phase_duration_ps,
                "collective_floor_estimate": estimate.as_dict(),
                "transferred_at_use_acknowledged": (
                    acknowledge_transfer
                    and estimate.evidence_class == "transferred-at-use"
                ),
                "flow_count": len(flow_rows),
                "flow_payload_bytes": sum(flow.payload_bytes for flow in flow_rows),
                "fanin": _max_receiver_fanin(flow_rows),
                "completion_geometry": _completion_geometry(
                    flow_rows,
                    local_segments=classified.nvlink_segments,
                    width=width,
                ),
                "goal_sha256": goal_sha256,
                "goal_binary_sha256": goal_binary_sha256,
                "completion_csv_sha256": completion_sha256,
            }
        )
    layer_packet_ps = sum(phase["phase_duration_ps"] for phase in results)
    layer_uncalibrated_packet_ps = sum(
        phase["uncalibrated_phase_duration_ps"] for phase in results
    )
    return {
        "topology_sha256": _sha256_file(topology),
        "layer_packet_ps": layer_packet_ps,
        "layer_packet_ms": layer_packet_ps / 1_000_000_000,
        "layer_uncalibrated_packet_ps": layer_uncalibrated_packet_ps,
        "layer_uncalibrated_packet_ms": (
            layer_uncalibrated_packet_ps / 1_000_000_000
        ),
        "simulated_messages_per_layer": sum(
            phase["fabric_segments"] + phase["nvlink_segments"]
            for phase in results
        ),
        "simulated_bytes_per_layer": sum(
            phase["fabric_bytes"] + phase["nvlink_bytes"] for phase in results
        ),
        "phases": results,
    }


def _sparse_packet_width(
    config: dict[str, Any],
    *,
    width: int,
    output_dir: Path,
    htsim: Path,
    txt2bin: Path,
    calibration: Any,
) -> dict[str, Any]:
    from simllm.compute import ModelDims
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord
    from simllm.placement import RankMapper, declared_manifest
    from simllm.traffic import MoeActivationPrecision, plan_step_locality

    model = config["model"]
    operating = config["operating_point"]
    strategy = config["strategies"]["sparse_routed_payload"]
    sampling = config["packet_sampling"]
    tokens_per_rank = operating["local_batch_per_attention_dp_rank"] * (
        model["nextn"] + 1
    )
    precision = MoeActivationPrecision(
        dispatch_bytes_per_element=int(strategy["dispatch_bytes_per_element"]),
        combine_bytes_per_element=int(strategy["combine_bytes_per_element"]),
    )
    dims = ModelDims(
        num_layers=1,
        hidden_size=model["hidden_size"],
        intermediate_size=model["intermediate_size"],
        num_heads=model["num_attention_heads"],
        num_kv_heads=model["num_key_value_heads"],
        head_size=model["head_dim"],
        vocab_size=model["vocab_size"],
        dtype_bytes=precision.dispatch_bytes_per_element,
        num_experts=model["num_local_experts"],
        top_k=model["num_experts_per_tok"],
        moe_intermediate_size=model["intermediate_size"],
        local_num_experts=model["num_local_experts"] // width,
    )
    record = StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                "minimax-decode",
                RequestPhase.DECODE,
                num_new_tokens=tokens_per_rank,
                context_length=operating["input_length"] + 1,
            )
        ],
    )
    placement = declared_manifest(
        tp=1,
        pp=1,
        dp=width,
        nodes=width // operating["gpus_per_node"],
        gpus_per_node=operating["gpus_per_node"],
    )
    mapper = RankMapper(placement)
    plan = plan_step_locality(
        record,
        dims,
        (0,),
        ep_ranks=tuple(range(width)),
        uniform_tokens_per_rank=tokens_per_rank,
        activation_precision=precision,
        rank_mapper=mapper,
    )
    simulated = _simulate_packet_phases(
        config,
        width=width,
        output_dir=output_dir,
        htsim=htsim,
        txt2bin=txt2bin,
        phases=plan.phases,
        floor_requests=_sparse_floor_requests(config, width=width),
        calibration=calibration,
    )
    dispatch = plan.phases[0]
    combine = plan.phases[1]
    expected_destination_count = (width - 1) * (
        1 - ((width - model["num_experts_per_tok"]) / width) ** tokens_per_rank
    )
    cross_node_candidates = width - operating["gpus_per_node"]
    expected_cross_node_senders = cross_node_candidates * (
        1 - ((width - model["num_experts_per_tok"]) / width) ** tokens_per_rank
    )
    completed_geometry = simulated["phases"][0]["completion_geometry"]
    represented = int(sampling["represented_layer_executions"])
    dispatch_bytes = dispatch.fabric_bytes + dispatch.nvlink_bytes
    combine_bytes = combine.fabric_bytes + combine.nvlink_bytes
    return {
        **simulated,
        "expert_parallel": width,
        "strategy": strategy["name"],
        "traffic_definition": strategy["logical_traffic_definition"],
        "precision_justification": strategy["precision_justification"],
        "dispatch_bytes_per_element": precision.dispatch_bytes_per_element,
        "combine_bytes_per_element": precision.combine_bytes_per_element,
        "sampled": True,
        "sample_label": sampling["label_full"],
        "sampling_rule": sampling["rule"],
        "population_status": "full rank and realized-message population",
        "population_scored": False,
        "dispatch_bytes_per_layer": dispatch_bytes,
        "combine_bytes_per_layer": combine_bytes,
        "dispatch_bytes_per_rank": dispatch_bytes / width,
        "combine_bytes_per_rank": combine_bytes / width,
        "dispatch_plus_combine_bytes_per_rank": (
            dispatch_bytes + combine_bytes
        )
        / width,
        "dispatch_plus_combine_fabric_bytes_per_rank": (
            dispatch.fabric_bytes + combine.fabric_bytes
        )
        / width,
        "represented_layer_executions": represented,
        "represented_messages": simulated["simulated_messages_per_layer"]
        * represented,
        "represented_bytes": simulated["simulated_bytes_per_layer"] * represented,
        "superseded_packet_dispatch_combine_ms": (
            simulated["layer_uncalibrated_packet_ms"] * represented
        ),
        "packet_dispatch_combine_ms": simulated["layer_packet_ms"] * represented,
        "routing_geometry": {
            "expected_distinct_destinations_per_source": expected_destination_count,
            "realized_distinct_destinations_per_source": completed_geometry[
                "completed_distinct_destinations_per_source"
            ],
            "expected_cross_node_senders_per_receiver": expected_cross_node_senders,
            "realized_cross_node_senders_per_receiver": completed_geometry[
                "completed_cross_node_senders_per_receiver"
            ],
            "maximum_cross_node_senders_per_receiver": completed_geometry[
                "maximum_completed_cross_node_senders_per_receiver"
            ],
            "realized_geometry_source": completed_geometry["source"],
        },
    }


def _dense_packet_phases(config: dict[str, Any], *, width: int) -> tuple[Any, ...]:
    from simllm.placement import RankMapper, declared_manifest
    from simllm.traffic import (
        CollectiveCommunicationPhase,
        DirectedCollectiveSegment,
        classify_step_locality,
    )

    operating = config["operating_point"]
    tokens_per_rank = operating["local_batch_per_attention_dp_rank"] * (
        config["model"]["nextn"] + 1
    )
    chunk_bytes = tokens_per_rank * config["model"]["hidden_size"] * 2
    ranks = tuple(range(width))
    phases = tuple(
        CollectiveCommunicationPhase(
            phase_id=f"family-d:{phase}",
            layer=0,
            participants=ranks,
            segments=tuple(
                DirectedCollectiveSegment(
                    source_rank=source,
                    destination_rank=destination,
                    payload_bytes=chunk_bytes,
                    tag=2000 + phase_index,
                )
                for source in ranks
                for destination in ranks
                if source != destination
            ),
            operation_id=f"family-d:{phase}",
        )
        for phase_index, phase in enumerate(("dense-all-gather", "dense-reduce-scatter"))
    )
    placement = declared_manifest(
        tp=1,
        pp=1,
        dp=width,
        nodes=width // operating["gpus_per_node"],
        gpus_per_node=operating["gpus_per_node"],
    )
    return classify_step_locality(
        phases,
        rank_mapper=RankMapper(placement),
    ).phases


def _dense_packet_width(
    config: dict[str, Any],
    *,
    width: int,
    output_dir: Path,
    htsim: Path,
    txt2bin: Path,
    calibration: Any,
) -> dict[str, Any]:
    strategy = config["strategies"]["dense_sm90_general_fallback"]
    sampling = config["packet_sampling"]
    simulated = _simulate_packet_phases(
        config,
        width=width,
        output_dir=output_dir,
        htsim=htsim,
        txt2bin=txt2bin,
        phases=_dense_packet_phases(config, width=width),
        floor_requests=_dense_floor_requests(config, width=width),
        calibration=calibration,
    )
    represented = int(sampling["represented_layer_executions"])
    return {
        **simulated,
        "expert_parallel": width,
        "strategy": strategy["name"],
        "traffic_definition": strategy["logical_traffic_definition"],
        "packet_pricing_definition": strategy["packet_pricing_definition"],
        "sampled": True,
        "sample_label": sampling["label_full"],
        "sampling_rule": sampling["rule"],
        "population_status": "measured full rank and message population",
        "population_scored": True,
        "represented_layer_executions": represented,
        "represented_messages": simulated["simulated_messages_per_layer"]
        * represented,
        "represented_bytes": simulated["simulated_bytes_per_layer"] * represented,
        "superseded_packet_dispatch_combine_ms": (
            simulated["layer_uncalibrated_packet_ms"] * represented
        ),
        "packet_dispatch_combine_ms": simulated["layer_packet_ms"] * represented,
        "extrapolation": None,
    }


def _extrapolate_dense_packet_width(
    config: dict[str, Any],
    *,
    width: int,
    anchor: dict[str, Any],
    calibration: Any,
) -> dict[str, Any]:
    strategy = config["strategies"]["dense_sm90_general_fallback"]
    sampling = config["packet_sampling"]
    operating = config["operating_point"]
    anchor_width = int(anchor["expert_parallel"])
    local_width = int(operating["gpus_per_node"])
    factor = Fraction(width - local_width, anchor_width - local_width)
    tokens_per_rank = operating["local_batch_per_attention_dp_rank"] * (
        config["model"]["nextn"] + 1
    )
    chunk_bytes = tokens_per_rank * config["model"]["hidden_size"] * 2
    messages_per_layer = 2 * width * (width - 1)
    bytes_per_layer = messages_per_layer * chunk_bytes
    represented = int(sampling["represented_layer_executions"])
    requests = _dense_floor_requests(config, width=width)
    acknowledge_transfer = bool(
        config["collective_floor_correction"]["acknowledge_transferred_at_use"]
    )
    projected_phases = []
    for anchor_phase, request in zip(anchor["phases"], requests, strict=True):
        estimate = _collective_floor_estimate(
            calibration,
            dtype=request["dtype"],
            operation=request["operation"],
            ranks=width,
            message_bytes=request["message_bytes"],
            donor=request["donor"],
            acknowledge_transfer=acknowledge_transfer,
        )
        extrapolated_fabric_ps = float(
            Fraction(anchor_phase["fabric_jct_ps"]) * factor
        )
        extrapolated_uncalibrated_ps = float(
            Fraction(anchor_phase["uncalibrated_phase_duration_ps"]) * factor
        )
        corrected_phase_ps = estimate.floor_charge_ps + max(
            estimate.serialization_ps,
            extrapolated_fabric_ps,
        )
        projected_phases.append(
            {
                "phase": anchor_phase["phase"].replace(
                    f"family-d-{anchor_width}",
                    f"family-d-{width}",
                ),
                "anchor_expert_parallel": anchor_width,
                "anchor_fabric_service_ps": anchor_phase["fabric_jct_ps"],
                "fabric_jct_ps": extrapolated_fabric_ps,
                "uncalibrated_phase_duration_ps": extrapolated_uncalibrated_ps,
                "phase_duration_ps": corrected_phase_ps,
                "collective_floor_estimate": estimate.as_dict(),
                "transferred_at_use_acknowledged": True,
                "projection": "EP 128 fabric service scaled by 31 / 15",
            }
        )
    layer_uncalibrated_packet_ps = sum(
        phase["uncalibrated_phase_duration_ps"] for phase in projected_phases
    )
    layer_packet_ps = sum(phase["phase_duration_ps"] for phase in projected_phases)
    return {
        "expert_parallel": width,
        "strategy": strategy["name"],
        "traffic_definition": strategy["logical_traffic_definition"],
        "packet_pricing_definition": strategy["packet_pricing_definition"],
        "sampled": True,
        "sample_label": sampling["label_full"],
        "sampling_rule": sampling["rule"],
        "population_status": (
            "unscored post-specified diagnostic extrapolation from measured full "
            "EP 128 population"
        ),
        "population_scored": False,
        "simulated_messages_per_layer": 0,
        "simulated_bytes_per_layer": 0,
        "represented_layer_executions": represented,
        "represented_messages": messages_per_layer * represented,
        "represented_bytes": bytes_per_layer * represented,
        "layer_uncalibrated_packet_ps": layer_uncalibrated_packet_ps,
        "layer_uncalibrated_packet_ms": layer_uncalibrated_packet_ps / 1_000_000_000,
        "layer_packet_ps": layer_packet_ps,
        "layer_packet_ms": layer_packet_ps / 1_000_000_000,
        "superseded_packet_dispatch_combine_ms": (
            layer_uncalibrated_packet_ps * represented / 1_000_000_000
        ),
        "packet_dispatch_combine_ms": (
            layer_packet_ps * represented / 1_000_000_000
        ),
        "topology_sha256": None,
        "phases": projected_phases,
        "extrapolation": {
            "anchor_expert_parallel": anchor_width,
            "anchor_population_status": anchor["population_status"],
            "cross_node_bytes_per_rank_factor": float(factor),
            "superseded_rule": sampling["dense_widest_extrapolation_rule"],
            "superseded_rule_status": sampling[
                "dense_widest_extrapolation_rule_status"
            ],
            "rule": sampling["dense_widest_corrected_extrapolation_rule"],
            "linearity_break": (
                "the fixed aggregate floor is additive and must not be multiplied "
                "by the cross-node byte factor"
            ),
            "rule_commit": config["chronology"][
                "diagnostic_extrapolation_rule_commit"
            ],
            "frozen_before_implementation": config["chronology"][
                "diagnostic_extrapolation_rule_frozen"
            ],
            "scored": sampling["dense_widest_extrapolation_scored"],
        },
    }


def _run_evaluation(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    from simllm.calibration.external_db import ExternalOperationDatabase
    from simllm.calibration.external_nccl import (
        NCCL_INTRA_NODE_BANDWIDTH_BYTES_PER_SECOND,
        ExternalNcclDatabase,
    )
    from simllm.calibration.external_pass import ExternalModelConfig, ExternalPassModel

    htsim = _configured_path(HTSIM_ENV)
    txt2bin = _configured_path(TXT2BIN_ENV)
    live = _live_sdk_subprocess(config)
    operation_database = ExternalOperationDatabase.load()
    nccl_database = ExternalNcclDatabase.load()
    collective_floor = _load_collective_floor_calibration(config, nccl_database)
    widths = []
    dense_anchor: dict[str, Any] | None = None
    for frozen in config["widths"]:
        width = int(frozen["expert_parallel"])
        model = ExternalModelConfig.from_mapping(
            config["model"],
            architecture="moe",
            tensor_parallel=1,
            pipeline_parallel=1,
            expert_parallel=width,
            workload_distribution=config["operating_point"][
                "workload_distribution"
            ],
            gemm_quant_mode=config["model"]["gemm_quant_mode"],
            attention_quant_mode=config["model"]["kv_cache_quant_mode"],
        )
        composed = ExternalPassModel(
            operation_database,
            model,
            nccl_database=nccl_database,
        ).run_generation(
            batch_size=config["operating_point"][
                "local_batch_per_attention_dp_rank"
            ],
            isl=config["operating_point"]["input_length"],
            osl=config["operating_point"]["output_length"],
            stride=config["operating_point"]["generation_stride"],
        )
        operations = composed.operation_latencies()
        dispatch = (
            operations["generation_moe_pre_dispatch"]
            + operations["generation_moe_post_dispatch"]
        )
        sparse_packet = _sparse_packet_width(
            config,
            width=width,
            output_dir=output_dir / f"ep-{width}" / "family-s",
            htsim=htsim,
            txt2bin=txt2bin,
            calibration=collective_floor,
        )
        if width <= int(
            config["packet_sampling"]["dense_direct_full_population_max_width"]
        ):
            dense_packet = _dense_packet_width(
                config,
                width=width,
                output_dir=output_dir / f"ep-{width}" / "family-d",
                htsim=htsim,
                txt2bin=txt2bin,
                calibration=collective_floor,
            )
            if width == int(
                config["packet_sampling"]["dense_widest_anchor_width"]
            ):
                dense_anchor = dense_packet
        else:
            if dense_anchor is None:
                raise RuntimeError("dense extrapolation reached before its full anchor")
            dense_packet = _extrapolate_dense_packet_width(
                config,
                width=width,
                anchor=dense_anchor,
                calibration=collective_floor,
            )
        sparse_step = (
            composed.total.latency_ms
            - dispatch
            + sparse_packet["packet_dispatch_combine_ms"]
        )
        superseded_sparse_step = (
            composed.total.latency_ms
            - dispatch
            + sparse_packet["superseded_packet_dispatch_combine_ms"]
        )
        fixed_overhead_analysis = None
        if width == 128:
            tokens_per_rank = config["operating_point"][
                "local_batch_per_attention_dp_rank"
            ] * (config["model"]["nextn"] + 1)
            message_elements = tokens_per_rank * config["model"]["hidden_size"] * width
            donor_latency_ms = sum(
                nccl_database.query(
                    dtype="half",
                    operation=operation,
                    ranks=8,
                    message_size=message_elements,
                ).latency_ms
                for operation in ("all_gather", "reduce_scatter")
            )
            ideal_serialization_ms = (
                2
                * (7 / 8)
                * (message_elements * 2)
                / NCCL_INTRA_NODE_BANDWIDTH_BYTES_PER_SECOND
                * 1_000
            )
            residual_ms = donor_latency_ms - ideal_serialization_ms
            rank_factor = ((width - 1) / width) * (8 / 7) * (450 / 50)
            inflated_residual_ms = (
                residual_ms
                * rank_factor
                * int(config["packet_sampling"]["represented_layer_executions"])
            )
            observed_gap_ms = dispatch - dense_packet[
                "superseded_packet_dispatch_combine_ms"
            ]
            fixed_overhead_analysis = {
                "status": "superseded by the landed collective-floor binding",
                "expert_parallel": width,
                "message_axis_name": "message_bytes",
                "caller_argument_semantics": "half-precision element count",
                "dtype_identifier": "half",
                "dtype_interpretation": (
                    "generic half precision; the source table does not identify BF16"
                ),
                "donor_latency_microseconds_per_layer": donor_latency_ms * 1_000,
                "ideal_ring_serialization_microseconds_per_layer": (
                    ideal_serialization_ms * 1_000
                ),
                "fixed_and_algorithmic_residual_microseconds_per_layer": (
                    residual_ms * 1_000
                ),
                "rank_factor": rank_factor,
                "inflated_residual_ms_over_65_layers": inflated_residual_ms,
                "external_minus_packet_gap_ms": observed_gap_ms,
                "inflated_residual_exceeds_gap": inflated_residual_ms
                > observed_gap_ms,
            }
        widths.append(
            {
                "expert_parallel": width,
                "composer_decode_step_ms": composed.total.latency_ms,
                "composer_decode_step_hex": composed.total.hex,
                "composer_dispatch_ms": dispatch,
                "composer_dispatch_hex": dispatch.hex(),
                "non_dispatch_timing_base_ms": composed.total.latency_ms - dispatch,
                "family_d_packet_dispatch_combine_ms": dense_packet[
                    "packet_dispatch_combine_ms"
                ],
                "family_d_packet_to_external_ratio": (
                    dense_packet["packet_dispatch_combine_ms"] / dispatch
                ),
                "family_d_superseded_packet_dispatch_combine_ms": dense_packet[
                    "superseded_packet_dispatch_combine_ms"
                ],
                "family_d_superseded_packet_to_external_ratio": (
                    dense_packet["superseded_packet_dispatch_combine_ms"] / dispatch
                ),
                "family_d_fixed_overhead_analysis": fixed_overhead_analysis,
                "family_s_packet_priced_step_ms": sparse_step,
                "family_s_packet_to_external_step_ratio": (
                    sparse_step / float(frozen["live_decode_step_ms"])
                ),
                "family_s_superseded_packet_priced_step_ms": superseded_sparse_step,
                "family_s_superseded_packet_to_external_step_ratio": (
                    superseded_sparse_step / float(frozen["live_decode_step_ms"])
                ),
                "operation_evidence_classes": {
                    entry.operation: entry.evidence_class for entry in composed.operations
                },
                "packet_evidence_class": "SIM-DERIVED",
                "dense_packet": dense_packet,
                "sparse_packet": sparse_packet,
            }
        )
    return {
        "live_sdk": live,
        "widths": widths,
        "operation_database_identity": operation_database.source.as_dict(),
        "operation_database_payload_sha256": operation_database.payload_sha256,
        "nccl_database_identity": nccl_database.source.as_dict(),
        "nccl_database_payload_sha256": nccl_database.payload_sha256,
        "collective_floor_calibration": {
            "calibration_id": collective_floor.calibration_id,
            "source": collective_floor.source.as_dict(),
            "input_surface": list(collective_floor.input_surface),
            "fitted_byte_range": list(collective_floor.fitted_byte_range),
            "curve_keys": [list(key) for key in collective_floor.curve_keys],
        },
    }


def _run_fresh_worker(attempt: Path, index: int) -> dict[str, Any]:
    worker_dir = attempt / f"evaluation-{index}"
    worker_dir.mkdir()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            os.fspath(Path(__file__)),
            "--evaluation-worker",
            "--worker-output",
            os.fspath(worker_dir),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    (worker_dir / "worker.stdout.txt").write_text(
        completed.stdout,
        encoding="utf-8",
        newline="\n",
    )
    (worker_dir / "worker.stderr.txt").write_text(
        completed.stderr,
        encoding="utf-8",
        newline="\n",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"evaluation worker {index} failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"evaluation worker {index} returned no JSON")
    return json.loads(lines[-1])


def _deterministic_payload(evaluation: dict[str, Any]) -> bytes:
    return json.dumps(
        evaluation,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _family_d_assessment(
    *,
    width: int,
    gpus_per_node: int,
    ratio: float,
    population_scored: bool,
) -> dict[str, Any]:
    scored = bool(population_scored)
    passed = ratio >= 1.0 if scored else None
    return {
        "contention_comparison": False,
        "cross_node_contention_present": width > gpus_per_node,
        "score_status": (
            "scored measured cell" if scored else "unscored post-specified diagnostic"
        ),
        "outcome": (
            "PASS" if passed else "REFUTED"
        )
        if scored
        else "UNSCORED DIAGNOSTIC",
        "passed": passed,
        "scored": scored,
    }


def _collective_floor_phase_summaries(packet: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for phase in packet["phases"]:
        estimate = phase["collective_floor_estimate"]
        summaries.append(
            {
                "phase": phase["phase"],
                "requested_dtype": estimate["requested_dtype"],
                "requested_operation": estimate["requested_operation"],
                "requested_ranks": estimate["requested_ranks"],
                "message_bytes": estimate["message_bytes"],
                "donor_dtype": estimate["regime"]["dtype"],
                "donor_operation": estimate["regime"]["operation"],
                "donor_ranks": estimate["regime"]["ranks"],
                "aggregate_floor_ps": estimate["floor_charge_ps"],
                "calibrated_serialization_ps": estimate["serialization_ps"],
                "fabric_service_ps": phase["fabric_jct_ps"],
                "composed_phase_ps": phase["phase_duration_ps"],
                "evidence_class": estimate["evidence_class"],
                "transfer_reason": estimate["transfer_reason"],
                "transferred_at_use_acknowledged": phase[
                    "transferred_at_use_acknowledged"
                ],
            }
        )
    return summaries


def _score(
    config: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    elapsed_seconds: float,
    deterministic: bool,
    chronology: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    live_by_width = {
        row["expert_parallel"]: row for row in evaluation["live_sdk"]["widths"]
    }
    composed_by_width = {
        row["expert_parallel"]: row for row in evaluation["widths"]
    }
    dense = config["strategies"]["dense_sm90_general_fallback"]
    sparse = config["strategies"]["sparse_routed_payload"]
    void_by_width = {
        int(row["expert_parallel"]): row for row in config["void_first_run"]["rows"]
    }
    rows = []
    e_cells = []
    c_cells = []
    d_cells = []
    for frozen in config["widths"]:
        width = int(frozen["expert_parallel"])
        live = live_by_width[width]
        composed = composed_by_width[width]
        dispatch_passed = composed["composer_dispatch_hex"] == frozen[
            "live_dispatch_hex"
        ]
        quotient = composed["composer_decode_step_ms"] / live["decode_step_ms"]
        composition_passed = 0.98 <= quotient <= 1.02
        d_ratio = composed["family_d_packet_to_external_ratio"]
        d_superseded_ratio = composed[
            "family_d_superseded_packet_to_external_ratio"
        ]
        d_assessment = _family_d_assessment(
            width=width,
            gpus_per_node=int(config["operating_point"]["gpus_per_node"]),
            ratio=d_ratio,
            population_scored=bool(composed["dense_packet"]["population_scored"]),
        )
        e_cells.append(
            {
                "expert_parallel": width,
                "expected_hex": frozen["live_dispatch_hex"],
                "actual_hex": composed["composer_dispatch_hex"],
                "passed": dispatch_passed,
            }
        )
        c_cells.append(
            {
                "expert_parallel": width,
                "quotient": quotient,
                "lower": 0.98,
                "upper": 1.02,
                "passed": composition_passed,
            }
        )
        d_cells.append(
            {
                "expert_parallel": width,
                "ratio": d_ratio,
                "lower": 1.0,
                "passed": d_assessment["passed"],
                "scored": d_assessment["scored"],
                "outcome": d_assessment["outcome"],
                "population_status": composed["dense_packet"][
                    "population_status"
                ],
            }
        )
        void_row = void_by_width[width]
        rows.append(
            {
                "expert_parallel": width,
                "sampled": True,
                "sample_label": composed["sparse_packet"]["sample_label"],
                "aiconfigurator_step_ms": live["decode_step_ms"],
                "aiconfigurator_dispatch_ms": live["dispatch_ms"],
                "dispatch_share": live["dispatch_ms"] / live["decode_step_ms"],
                "composer_step_ms": composed["composer_decode_step_ms"],
                "composer_quotient": quotient,
                "family_d_external_arm": "D-external",
                "family_d_external_strategy": dense["name"],
                "family_d_external_traffic_definition": dense[
                    "logical_traffic_definition"
                ],
                "family_d_external_pricing_definition": dense[
                    "external_pricing_definition"
                ],
                "family_d_external_dtype_identifier": dense[
                    "external_dtype_identifier"
                ],
                "family_d_external_dtype_interpretation": dense[
                    "external_dtype_interpretation"
                ],
                "family_d_external_message_axis_name": dense[
                    "external_message_axis_name"
                ],
                "family_d_external_message_argument_interpretation": dense[
                    "external_message_argument_interpretation"
                ],
                "family_d_packet_arm": "D-packet",
                "family_d_packet_strategy": dense["name"],
                "family_d_packet_traffic_definition": dense[
                    "logical_traffic_definition"
                ],
                "family_d_packet_pricing_definition": dense[
                    "packet_pricing_definition"
                ],
                "family_d_superseded_packet_pricing_definition": dense[
                    "superseded_packet_pricing_definition"
                ],
                "family_d_external_ms": live["dispatch_ms"],
                "family_d_packet_ms": composed["dense_packet"][
                    "packet_dispatch_combine_ms"
                ],
                "family_d_ratio": d_ratio,
                "family_d_superseded_packet_ms": composed[
                    "family_d_superseded_packet_dispatch_combine_ms"
                ],
                "family_d_superseded_ratio": d_superseded_ratio,
                "family_d_collective_floor_phases": (
                    _collective_floor_phase_summaries(composed["dense_packet"])
                ),
                "family_d_same_logical_element_count": True,
                "family_d_contention_comparison": d_assessment[
                    "contention_comparison"
                ],
                "family_d_cross_node_contention_present": d_assessment[
                    "cross_node_contention_present"
                ],
                "family_d_comparison_interpretation": (
                    "ratio of an opaque external NCCL-table cost model to a "
                    "direct all-pairs packet cost model; not contention isolation"
                ),
                "family_d_score_status": d_assessment["score_status"],
                "family_d_outcome": d_assessment["outcome"],
                "family_d_population_status": composed["dense_packet"][
                    "population_status"
                ],
                "family_d_simulated_messages_per_layer": composed["dense_packet"][
                    "simulated_messages_per_layer"
                ],
                "family_d_represented_messages": composed["dense_packet"][
                    "represented_messages"
                ],
                "family_d_extrapolation": composed["dense_packet"]["extrapolation"],
                "family_s_dense_arm": "S-dense-external",
                "family_s_dense_strategy": dense["name"],
                "family_s_dense_traffic_definition": dense[
                    "logical_traffic_definition"
                ],
                "family_s_sparse_arm": "S-sparse-packet",
                "family_s_sparse_strategy": sparse["name"],
                "family_s_sparse_traffic_definition": sparse[
                    "logical_traffic_definition"
                ],
                "family_s_packet_pricing_definition": sparse[
                    "packet_pricing_definition"
                ],
                "family_s_superseded_packet_pricing_definition": sparse[
                    "superseded_packet_pricing_definition"
                ],
                "family_s_sparse_dispatch_bytes_per_element": sparse[
                    "dispatch_bytes_per_element"
                ],
                "family_s_sparse_combine_bytes_per_element": sparse[
                    "combine_bytes_per_element"
                ],
                "family_s_sparse_precision_justification": sparse[
                    "precision_justification"
                ],
                "family_s_packet_dispatch_combine_ms": composed["sparse_packet"][
                    "packet_dispatch_combine_ms"
                ],
                "family_s_packet_step_ms": composed[
                    "family_s_packet_priced_step_ms"
                ],
                "family_s_sparse_to_dense_step_ratio": composed[
                    "family_s_packet_to_external_step_ratio"
                ],
                "family_s_superseded_packet_dispatch_combine_ms": composed[
                    "sparse_packet"
                ]["superseded_packet_dispatch_combine_ms"],
                "family_s_superseded_packet_step_ms": composed[
                    "family_s_superseded_packet_priced_step_ms"
                ],
                "family_s_superseded_sparse_to_dense_step_ratio": composed[
                    "family_s_superseded_packet_to_external_step_ratio"
                ],
                "family_s_collective_floor_phases": (
                    _collective_floor_phase_summaries(composed["sparse_packet"])
                ),
                "family_s_population_status": composed["sparse_packet"][
                    "population_status"
                ],
                "family_s_simulated_messages_per_layer": composed[
                    "sparse_packet"
                ]["simulated_messages_per_layer"],
                "family_s_represented_messages": composed["sparse_packet"][
                    "represented_messages"
                ],
                "family_s_dispatch_bytes_per_rank": composed["sparse_packet"][
                    "dispatch_bytes_per_rank"
                ],
                "family_s_combine_bytes_per_rank": composed["sparse_packet"][
                    "combine_bytes_per_rank"
                ],
                "family_s_payload_bytes_per_rank": composed["sparse_packet"][
                    "dispatch_plus_combine_bytes_per_rank"
                ],
                "family_s_routing_geometry": composed["sparse_packet"][
                    "routing_geometry"
                ],
                "void_first_run_status": config["void_first_run"]["status"],
                "void_first_run_dense_strategy": dense["name"],
                "void_first_run_dense_traffic_definition": dense[
                    "logical_traffic_definition"
                ],
                "void_first_run_sparse_strategy": (
                    "sparse all-pairs fluidized FP8 payload"
                ),
                "void_first_run_sparse_traffic_definition": (
                    "fractional FP8 bytes over every directed rank pair in both "
                    "dispatch and combine"
                ),
                "void_first_run_packet_dispatch_combine_ms": void_row[
                    "packet_dispatch_combine_ms"
                ],
                "void_first_run_packet_step_ms": void_row["packet_priced_step_ms"],
                "void_first_run_ratio": void_row[
                    "packet_to_external_step_ratio"
                ],
                "void_first_run_population_status": void_row["population_status"],
                "e_passed": dispatch_passed,
                "c_passed": composition_passed,
            }
        )

    families = {
        "E": {
            "passed": sum(cell["passed"] for cell in e_cells),
            "denominator": 4,
            "cells": e_cells,
        },
        "C": {
            "passed": sum(cell["passed"] for cell in c_cells),
            "denominator": 4,
            "cells": c_cells,
            "interpretation": "end-to-end parity reusing the dispatch code validated by E",
        },
        "D": {
            "passed": sum(cell["passed"] is True for cell in d_cells),
            "denominator": sum(cell["scored"] for cell in d_cells),
            "cells": d_cells,
            "rule": "D-packet divided by D-external is at least 1.0",
            "interpretation": (
                "comparison of two cost models on the same requested logical "
                "element count; not evidence that contention is the only difference"
            ),
        },
        "S": {
            "scored": False,
            "cells": [
                {
                    "expert_parallel": row["expert_parallel"],
                    "sparse_to_dense_step_ratio": row[
                        "family_s_sparse_to_dense_step_ratio"
                    ],
                    "superseded_sparse_to_dense_step_ratio": row[
                        "family_s_superseded_sparse_to_dense_step_ratio"
                    ],
                    "population_status": row["family_s_population_status"],
                }
                for row in rows
            ],
        },
        "W": {
            "passed": int(elapsed_seconds <= WALL_BOUND_SECONDS),
            "denominator": 1,
            "elapsed_seconds": elapsed_seconds,
            "upper_seconds": WALL_BOUND_SECONDS,
        },
    }
    declared_adjustments = config["declared_adjustments"]
    adjustments_are_sourced = bool(declared_adjustments) and all(
        adjustment.get("id")
        and adjustment.get("behavior")
        and adjustment.get("sources")
        and all(source.get("file") and source.get("line") for source in adjustment["sources"])
        for adjustment in declared_adjustments
    )
    routing_guards = []
    for composed in evaluation["widths"]:
        if composed["expert_parallel"] != 256:
            continue
        geometry = composed["sparse_packet"]["routing_geometry"]
        expected = geometry["expected_cross_node_senders_per_receiver"]
        maximum = geometry["maximum_cross_node_senders_per_receiver"]
        routing_guards.append(
            maximum == 0 if expected == 0 else maximum <= 1.2 * expected
        )
    precision_guard = all(
        row["sparse_packet"]["dispatch_bytes_per_element"] == 1
        and row["sparse_packet"]["combine_bytes_per_element"] == 2
        and "BF16" in row["sparse_packet"]["precision_justification"]
        for row in evaluation["widths"]
    )
    population_guard = all(
        not row["dense_packet"]["population_scored"]
        or "measured full" in row["dense_packet"]["population_status"]
        for row in evaluation["widths"]
    )
    collective_floor_guard = all(
        phase["collective_floor_estimate"]["evidence_class"]
        in {"calibrated", "transferred-at-use"}
        and (
            phase["collective_floor_estimate"]["evidence_class"] != "transferred-at-use"
            or phase["transferred_at_use_acknowledged"]
        )
        for row in evaluation["widths"]
        for packet in (row["dense_packet"], row["sparse_packet"])
        for phase in packet["phases"]
    )
    fatal_guards = {
        "FG-1": adjustments_are_sourced
        and all(
            set(row["operation_evidence_classes"].values()) == {"MEASURED-EXTERNAL"}
            for row in evaluation["widths"]
        ),
        "FG-2": all(
            set(row["operation_evidence_classes"].values()) == {"MEASURED-EXTERNAL"}
            and row["packet_evidence_class"] == "SIM-DERIVED"
            for row in evaluation["widths"]
        )
        and collective_floor_guard,
        "FG-3": config["model"]["nextn"] == 3,
        "FG-4": False,
        "FG-5": all(
            row["sampled"] and "sampled layer" in row["sample_label"]
            for row in rows
        ),
        "FG-6": deterministic,
        "FG-7": chronology,
        "FG-8": all(routing_guards),
        "FG-9": precision_guard,
        "FG-10": population_guard,
    }
    return {
        "families": families,
        "fatal_guards": fatal_guards,
        "run_state": "nonvoid" if all(fatal_guards.values()) else "void",
    }, rows


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("results CSV needs at least one row")
    columns = tuple(rows[0])
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {
            key: (
                json.dumps(value, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (dict, list))
                else value
            )
            for key, value in row.items()
        }
        for row in rows
    )
    return output.getvalue().encode("utf-8")


def _render_figures(
    record_path: Path,
    output_dir: Path,
    *,
    python_executable: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            os.fspath(python_executable),
            os.fspath(STUDY / "plot_results.py"),
            "--record",
            os.fspath(record_path),
            "--output-dir",
            os.fspath(output_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "figure rendering failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    return {
        "png": "figures/minimax_ep_scaling.png",
        "pdf": "figures/minimax_ep_scaling.pdf",
        "metadata": "figures/minimax_ep_scaling.metadata.json",
    }


def _has_dense_definition(value: str) -> bool:
    lowered = value.lower()
    return all(
        term in lowered
        for term in ("dense", "half-precision", "all-gather", "reduce-scatter")
    )


def _has_sparse_definition(value: str) -> bool:
    lowered = value.lower()
    return all(term in lowered for term in ("sparse", "routed", "fp8", "bf16"))


def _markdown_table_rows(report_text: str, *, heading: str) -> list[str]:
    lines = report_text.splitlines()
    try:
        start = lines.index(heading)
    except ValueError as error:
        raise RuntimeError(f"FG-4 RESULTS.md omits section {heading!r}") from error
    heading_level = len(heading) - len(heading.lstrip("#"))
    rows = []
    for line in lines[start + 1 :]:
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            if level <= heading_level:
                break
        stripped = line.strip()
        if re.match(r"^\|\s*(8|32|128|256)\s*\|", stripped):
            rows.append(stripped)
    return rows


def _inspect_results_table_disclosures(results_path: Path) -> int:
    report_text = results_path.read_text(encoding="utf-8")
    sections = {
        "Family D": (
            "### Family D cost-model comparison",
            {
                2: (
                    "external nccl-table cost model",
                    "dense sm90 fallback",
                    "half-precision all-gather",
                    "reduce-scatter",
                ),
                3: (
                    "packet clos cost model",
                    "dense sm90 fallback",
                    "half-precision all-gather",
                    "reduce-scatter",
                ),
            },
        ),
        "first-run void": (
            "## First-run void evidence",
            {
                1: (
                    "dense sm90 fallback",
                    "half-precision all-gather",
                    "reduce-scatter",
                ),
                2: ("sparse all-pairs fluidized fp8",),
            },
        ),
        "Family S": (
            "## Family S: published strategy comparison, unscored",
            {
                1: (
                    "dense sm90 fallback",
                    "half-precision all-gather",
                    "reduce-scatter",
                ),
                2: (
                    "sparse realized top-k routing",
                    "fp8 dispatch",
                    "bf16 combine",
                ),
            },
        ),
    }
    inspected = 0
    for section_name, (heading, cell_requirements) in sections.items():
        rows = _markdown_table_rows(report_text, heading=heading)
        if len(rows) != 4:
            raise RuntimeError(
                f"FG-4 RESULTS.md {section_name} table requires four EP rows"
            )
        for index, row in enumerate(rows):
            cells = [cell.strip().lower() for cell in row.strip("|").split("|")]
            for cell_index, required_terms in cell_requirements.items():
                missing = [
                    term for term in required_terms if term not in cells[cell_index]
                ]
                if missing:
                    raise RuntimeError(
                        f"FG-4 RESULTS.md {section_name} row {index} omits "
                        + ", ".join(repr(term) for term in missing)
                    )
        inspected += len(rows)
    family_d_rows = _markdown_table_rows(
        report_text,
        heading="### Family D cost-model comparison",
    )
    if "not a contention cell" not in family_d_rows[0].lower():
        raise RuntimeError("FG-4 RESULTS.md EP 8 row hides its interpretation limit")
    if "unscored diagnostic" not in family_d_rows[-1].lower():
        raise RuntimeError("FG-4 RESULTS.md EP 256 row hides its unscored status")
    return inspected


def _inspect_artifact_disclosures(
    *,
    record_path: Path,
    csv_path: Path,
    figures_dir: Path,
    results_path: Path,
) -> dict[str, Any]:
    """Inspect emitted records, CSV rows and figure text for FG-4 disclosure."""

    record = json.loads(record_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(record.get("rows", ())) != 4 or len(csv_rows) != 4:
        raise RuntimeError("FG-4 artifact inspection requires four record and CSV rows")
    disclosure_fields = (
        "family_d_external_strategy",
        "family_d_external_traffic_definition",
        "family_d_packet_strategy",
        "family_d_packet_traffic_definition",
        "family_s_dense_strategy",
        "family_s_dense_traffic_definition",
        "family_s_sparse_strategy",
        "family_s_sparse_traffic_definition",
        "void_first_run_dense_strategy",
        "void_first_run_dense_traffic_definition",
        "void_first_run_sparse_strategy",
        "void_first_run_sparse_traffic_definition",
    )
    for artifact_name, rows in (
        ("record", record["rows"]),
        ("CSV", csv_rows),
    ):
        for index, row in enumerate(rows):
            if any(not str(row.get(field, "")).strip() for field in disclosure_fields):
                raise RuntimeError(
                    f"FG-4 {artifact_name} row {index} omits a strategy disclosure"
                )
            if row["family_d_external_traffic_definition"] != row[
                "family_d_packet_traffic_definition"
            ]:
                raise RuntimeError(
                    f"FG-4 {artifact_name} row {index} does not carry identical D traffic"
                )
            if not _has_dense_definition(
                "dense " + row["family_d_external_traffic_definition"]
            ):
                raise RuntimeError(
                    f"FG-4 {artifact_name} row {index} does not define dense D traffic"
                )
            if not _has_sparse_definition(
                "sparse " + row["family_s_sparse_traffic_definition"]
            ):
                raise RuntimeError(
                    f"FG-4 {artifact_name} row {index} does not define sparse S traffic"
                )
            if not str(row.get("void_first_run_status", "")).startswith("VOID"):
                raise RuntimeError(
                    f"FG-4 {artifact_name} row {index} hides the first-run void"
                )

    metadata_path = figures_dir / "minimax_ep_scaling.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    series = metadata.get("series")
    expected_series_count = 8 if record.get("schema") == SCHEMA else 5
    if not isinstance(series, list) or len(series) != expected_series_count:
        raise RuntimeError("FG-4 figure metadata has an incomplete series inventory")
    for index, item in enumerate(series):
        if any(
            not isinstance(item.get(field), str) or not item[field].strip()
            for field in ("label", "strategy", "traffic_definition")
        ):
            raise RuntimeError(f"FG-4 figure series {index} omits strategy or traffic")
    caption = str(metadata.get("caption", ""))
    if not _has_dense_definition("dense " + caption) or not _has_sparse_definition(
        "sparse routed FP8 BF16 " + caption
    ):
        raise RuntimeError("FG-4 figure caption omits dense or sparse traffic")
    if record.get("schema") == SCHEMA and (
        "corrected ratios" not in caption or "superseded" not in caption
    ):
        raise RuntimeError("FG-4 figure caption hides the correction history")

    results_rows_inspected = _inspect_results_table_disclosures(results_path)
    pdf_path = figures_dir / "minimax_ep_scaling.pdf"
    if not pdf_path.is_file():
        raise RuntimeError("FG-4 generated PDF is missing")
    pdf_text_tool = shutil.which(PDF_TEXT_TOOL)
    pdf_text_inspection = {
        "performed": False,
        "skip_reason": f"{PDF_TEXT_TOOL} is not available on PATH",
        "tool": PDF_TEXT_TOOL,
    }
    if pdf_text_tool is not None:
        completed = subprocess.run(
            [pdf_text_tool, os.fspath(pdf_path), "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode != 0:
            raise RuntimeError("FG-4 could not inspect generated PDF text")
        pdf_text = completed.stdout.lower()
        for term in (
            "dense fallback",
            "half all-gather",
            "reduce-scatter",
            "sparse routed",
            "fp8 dispatch",
            "bf16 combine",
        ):
            if term not in pdf_text:
                raise RuntimeError(f"FG-4 generated PDF omits {term!r}")
        pdf_text_inspection = {
            "performed": True,
            "skip_reason": None,
            "tool": PDF_TEXT_TOOL,
        }
    return {
        "record_rows_inspected": len(record["rows"]),
        "csv_rows_inspected": len(csv_rows),
        "figure_series_inspected": len(series),
        "figure_caption_inspected": True,
        "pdf_text_inspection": pdf_text_inspection,
        "results_table_rows_inspected": results_rows_inspected,
    }


def _format_pdf_text_inspection(inspection: dict[str, Any]) -> str:
    pdf_text = inspection["pdf_text_inspection"]
    if pdf_text["performed"]:
        return f"fg4_pdf_text=passed tool={pdf_text['tool']}"
    return f"fg4_pdf_text=skipped missing_tool={pdf_text['tool']}"


def _coordinator(bulk_root: Path, *, write_tracked: bool) -> dict[str, Any]:
    started = time.monotonic()
    config = _load_config()
    dense = config["strategies"]["dense_sm90_general_fallback"]
    htsim = _configured_path(HTSIM_ENV)
    txt2bin = _configured_path(TXT2BIN_ENV)
    venv = _configured_path(EXTERNAL_VENV_ENV)
    attempt, attempt_number = _new_attempt(bulk_root)
    first = _run_fresh_worker(attempt, 1)
    second = _run_fresh_worker(attempt, 2)
    first_payload = _deterministic_payload(first)
    second_payload = _deterministic_payload(second)
    deterministic = first_payload == second_payload
    run_commit = _git("rev-parse", "HEAD")
    chronology = all(_is_ancestor(commit, run_commit) for commit in EXPECTED_FREEZE_COMMITS)
    elapsed_before_figure = time.monotonic() - started
    scored, rows = _score(
        config,
        first,
        elapsed_seconds=elapsed_before_figure,
        deterministic=deterministic,
        chronology=chronology,
    )
    widest_width = int(config["widths"][-1]["expert_parallel"])
    tokens_per_rank = config["operating_point"][
        "local_batch_per_attention_dp_rank"
    ] * (config["model"]["nextn"] + 1)
    link_bytes_per_second = (
        config["operating_point"]["link_rate_bits_per_second"] // 8
    )
    widest_sparse = next(
        row["sparse_packet"]
        for row in first["widths"]
        if row["expert_parallel"] == widest_width
    )
    fixed_overhead_analysis = next(
        row["family_d_fixed_overhead_analysis"]
        for row in first["widths"]
        if row["family_d_fixed_overhead_analysis"] is not None
    )
    dense_chunk_bytes = tokens_per_rank * config["model"]["hidden_size"] * 2
    dense_bytes_per_rank = 2 * (widest_width - 1) * dense_chunk_bytes
    dense_fabric_bytes_per_rank = (
        2
        * (widest_width - config["operating_point"]["gpus_per_node"])
        * dense_chunk_bytes
    )
    record = {
        "schema": SCHEMA,
        "study": "MiniMax-M2.5 expert-parallel scaling",
        "run_commit": run_commit,
        "freeze_commits": list(EXPECTED_FREEZE_COMMITS),
        "attempt": f"attempt-{attempt_number:04d}",
        "bulk_evidence": f"${{{BULK_ROOT_ENV}}}/attempt-{attempt_number:04d}",
        "configuration_sha256": _sha256_file(CONFIG_PATH),
        "expectations_sha256": _sha256_file(EXPECTATIONS_PATH),
        "corrected_expectations_sha256": _sha256_file(CORRECTED_EXPECTATIONS_PATH),
        "operating_point": config["operating_point"],
        "sampling": config["packet_sampling"],
        "strategies": config["strategies"],
        "collective_floor_binding": {
            **first["collective_floor_calibration"],
            "configuration": config["collective_floor_correction"],
            "transferred_at_use_acknowledged": True,
            "composition": config["collective_floor_correction"]["composition"],
        },
        "void_first_run": config["void_first_run"],
        "physical_sanity": {
            "widest_expert_parallel": widest_width,
            "link_bytes_per_second": link_bytes_per_second,
            "sparse_dispatch_bytes_per_rank": widest_sparse[
                "dispatch_bytes_per_rank"
            ],
            "sparse_combine_bytes_per_rank": widest_sparse[
                "combine_bytes_per_rank"
            ],
            "sparse_dispatch_plus_combine_bytes_per_rank": widest_sparse[
                "dispatch_plus_combine_bytes_per_rank"
            ],
            "sparse_dispatch_plus_combine_fabric_bytes_per_rank": widest_sparse[
                "dispatch_plus_combine_fabric_bytes_per_rank"
            ],
            "sparse_serialization_floor_microseconds_per_layer": (
                widest_sparse["dispatch_plus_combine_fabric_bytes_per_rank"]
                / link_bytes_per_second
                * 1_000_000
            ),
            "dense_half_buffer_bytes_per_rank": (
                tokens_per_rank
                * config["model"]["hidden_size"]
                * widest_width
                * 2
            ),
            "dense_dispatch_plus_combine_wire_bytes_per_rank": dense_bytes_per_rank,
            "dense_dispatch_plus_combine_fabric_bytes_per_rank": (
                dense_fabric_bytes_per_rank
            ),
            "dense_serialization_floor_microseconds_per_layer": (
                dense_fabric_bytes_per_rank
                / link_bytes_per_second
                * 1_000_000
            ),
        },
        "traffic_model_disclosure": {
            "family_d": (
                "same requested dense logical element count in both arms; the "
                "packet arm composes acknowledged aggregate collective floors "
                "and byte slopes with direct fabric service, while the external "
                "arm remains an opaque NCCL-table cost model; the ratio is not "
                "contention isolation"
            ),
            "family_s": (
                "dense SM90 general fallback versus sparse routed FP8 dispatch "
                "and BF16 combine with transferred aggregate collective timing; "
                "unscored strategy comparison"
            ),
            "deployment_strategy_selection": "unknown to this study",
        },
        "external_table_identification": {
            "dtype_identifier": dense["external_dtype_identifier"],
            "dtype_interpretation": dense["external_dtype_interpretation"],
            "message_axis_name": dense["external_message_axis_name"],
            "caller_argument_interpretation": dense[
                "external_message_argument_interpretation"
            ],
        },
        "family_d_fixed_overhead_analysis": fixed_overhead_analysis,
        "evidence_classes": {
            "compute_and_external_dispatch": "MEASURED-EXTERNAL",
            "packet_fabric_dispatch_and_combine": "SIM-DERIVED",
            "collective_floor_exact_domain": "calibrated",
            "collective_floor_rank_or_semantic_transfer": "transferred-at-use",
        },
        "source_artifacts": {
            "operation_database": {
                "identity": first["operation_database_identity"],
                "payload_sha256": first["operation_database_payload_sha256"],
            },
            "nccl_database": {
                "identity": first["nccl_database_identity"],
                "payload_sha256": first["nccl_database_payload_sha256"],
                "collection_version": "2.26.2",
                "row_versions": ["2.29.2"],
            },
        },
        "binary_identities": {
            "htsim_rnic": {"name": htsim.name, "sha256": _sha256_file(htsim)},
            "txt2bin": {"name": txt2bin.name, "sha256": _sha256_file(txt2bin)},
            "external_python": {
                "name": _venv_python(venv).name,
                "environment": EXTERNAL_VENV_ENV,
            },
        },
        "fresh_evaluations": {
            "count": 2,
            "first_sha256": _sha256_bytes(first_payload),
            "second_sha256": _sha256_bytes(second_payload),
            "bit_equal": deterministic,
        },
        "machine": {
            "architecture": platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "system": platform.system(),
        },
        "rows": rows,
        "family_tallies": scored["families"],
        "fatal_guards": scored["fatal_guards"],
        "run_state": scored["run_state"],
        "figures": {
            "png": "figures/minimax_ep_scaling.png",
            "pdf": "figures/minimax_ep_scaling.pdf",
            "metadata": "figures/minimax_ep_scaling.metadata.json",
        },
    }
    attempt_record = attempt / "record.json"
    attempt_csv = attempt / "results.csv"
    attempt_record.write_bytes(_json_bytes(record))
    attempt_csv.write_bytes(_csv_bytes(rows))
    _render_figures(
        attempt_record,
        attempt / "figures",
        python_executable=_venv_python(venv),
    )
    disclosure_inspection = _inspect_artifact_disclosures(
        record_path=attempt_record,
        csv_path=attempt_csv,
        figures_dir=attempt / "figures",
        results_path=STUDY / "RESULTS.md",
    )
    record["artifact_disclosure_inspection"] = disclosure_inspection
    record["fatal_guards"]["FG-4"] = True
    elapsed = time.monotonic() - started
    record["family_tallies"]["W"]["elapsed_seconds"] = elapsed
    record["family_tallies"]["W"]["passed"] = int(elapsed <= WALL_BOUND_SECONDS)
    record["run_state"] = (
        "nonvoid" if all(record["fatal_guards"].values()) else "void"
    )
    attempt_record.write_bytes(_json_bytes(record))
    _inspect_artifact_disclosures(
        record_path=attempt_record,
        csv_path=attempt_csv,
        figures_dir=attempt / "figures",
        results_path=STUDY / "RESULTS.md",
    )
    if write_tracked:
        TRACKED_RECORD.write_bytes(_json_bytes(record))
        TRACKED_CSV.write_bytes(_csv_bytes(rows))
        TRACKED_FIGURES.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "pdf", "metadata.json"):
            source = attempt / "figures" / f"minimax_ep_scaling.{suffix}"
            destination = TRACKED_FIGURES / source.name
            destination.write_bytes(source.read_bytes())
    return record


def _validate_record(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") == LEGACY_SCHEMA:
        return record, None
    if record.get("schema") not in SUPPORTED_RECORD_SCHEMAS:
        raise SystemExit("record has an unsupported schema")
    if len(record.get("rows", ())) != 4:
        raise SystemExit("record must contain four expert-parallel rows")
    if set(record.get("family_tallies", ())) != {"E", "C", "D", "S", "W"}:
        raise SystemExit("record family inventory is incomplete")
    if set(record.get("fatal_guards", ())) != {
        *(f"FG-{index}" for index in range(1, 11)),
    }:
        raise SystemExit("record fatal-guard inventory is incomplete")
    if not all(record["fatal_guards"].values()):
        raise SystemExit("tracked corrected run has a violated fatal guard")
    disclosure_inspection = None
    if path.resolve() == TRACKED_RECORD.resolve():
        disclosure_inspection = _inspect_artifact_disclosures(
            record_path=path,
            csv_path=TRACKED_CSV,
            figures_dir=TRACKED_FIGURES,
            results_path=STUDY / "RESULTS.md",
        )
        results_text = (STUDY / "RESULTS.md").read_text(encoding="utf-8")
        opening = " ".join(results_text[:8000].split())
        required_opening = [
            "VOID against FG-4",
            "0.2742607736975033",
            "strategy comparison",
            "does not know which strategy",
            "two cost models",
            "NOT evidence",
            "UNSCORED DIAGNOSTIC",
        ]
        if record.get("schema") == SCHEMA:
            required_opening.extend(
                (
                    "1 of 3",
                    "42.816396866840726",
                    "component-wise",
                    "superseded",
                )
            )
        else:
            required_opening.append("0 of 3")
        for required in required_opening:
            if required not in opening:
                raise SystemExit(f"RESULTS.md opening omits {required!r}")
    return record, disclosure_inspection


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk-root", type=Path)
    parser.add_argument("--write-tracked", action="store_true")
    parser.add_argument("--validate-tracked", action="store_true")
    parser.add_argument("--live-sdk-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--evaluation-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _load_config()
    if args.live_sdk_worker:
        print(json.dumps(_run_live_sdk(config), sort_keys=True, separators=(",", ":")))
        return 0
    if args.evaluation_worker:
        if args.worker_output is None:
            raise SystemExit("--evaluation-worker requires --worker-output")
        print(
            json.dumps(
                _run_evaluation(config, args.worker_output),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    if args.validate_tracked:
        record, disclosure_inspection = _validate_record(TRACKED_RECORD)
        print(f"run_state={record['run_state']}")
        if disclosure_inspection is not None:
            print(_format_pdf_text_inspection(disclosure_inspection))
        return 0
    raw_bulk = args.bulk_root or os.environ.get(BULK_ROOT_ENV)
    if raw_bulk is None:
        raise SystemExit(f"pass --bulk-root or set {BULK_ROOT_ENV}")
    record = _coordinator(Path(raw_bulk), write_tracked=args.write_tracked)
    print(
        f"run_state={record['run_state']} "
        f"elapsed_seconds={record['family_tallies']['W']['elapsed_seconds']:.6f}"
    )
    for family in ("E", "C", "D", "W"):
        tally = record["family_tallies"][family]
        print(f"{family}={tally['passed']}/{tally['denominator']}")
    print("S=unscored")
    print(f"fatal_guards={record['fatal_guards']}")
    print(_format_pdf_text_inspection(record["artifact_disclosure_inspection"]))
    return 0 if record["run_state"] == "nonvoid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
