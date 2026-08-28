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
import subprocess
import sys
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

CONFIG_PATH = STUDY / "study_config.json"
EXPECTATIONS_PATH = STUDY / "expectations.md"
TRACKED_RECORD = STUDY / "record.json"
TRACKED_CSV = STUDY / "results.csv"
TRACKED_FIGURES = STUDY / "figures"

BULK_ROOT_ENV = "SIMLLM_MINIMAX_T1_BULK_ROOT"
EXTERNAL_VENV_ENV = "SIMLLM_EXTERNAL_AIC_VENV"
HTSIM_ENV = "SIMLLM_HTSIM_RNIC"
TXT2BIN_ENV = "SIMLLM_TXT2BIN"
SCHEMA = "simllm-minimax-ep-scaling-record-v1"
QWEN_REFERENCE_RATIO = 1.042715399805
WALL_BOUND_SECONDS = 3600.0
EXPECTED_FREEZE_COMMITS = ("61b66c4", "5a29bb0")
AI_CONFIGURATOR_TRAFFIC_MODEL = (
    "half-precision NCCL all-gather plus reduce-scatter over tokens times hidden "
    "times expert width elements"
)
SIMLLM_TRAFFIC_MODEL = (
    "uniform routed FP8 expert payload over every directed rank pair, with "
    "placement, routing, queues and receiver fan-in"
)
TRAFFIC_COMPARISON_RULE = (
    "replace the AIConfigurator dispatch surrogate with the SimLLM packet "
    "duration on the same non-dispatch timing base; the traffic abstractions "
    "are not equivalent"
)


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
    if config.get("schema") != "simllm-minimax-ep-scaling-study-config-v1":
        raise SystemExit("study_config.json has an unsupported schema")
    if [row["expert_parallel"] for row in config["widths"]] != [8, 32, 128, 256]:
        raise SystemExit("study_config.json does not carry the frozen width sweep")
    if config["model"].get("nextn") != 3:
        raise SystemExit("the faithful study requires explicit nextn=3")
    return config


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


def _run_packet_width(
    config: dict[str, Any],
    *,
    width: int,
    output_dir: Path,
    htsim: Path,
    txt2bin: Path,
) -> dict[str, Any]:
    from simllm.backends import HtsimRnicConfig, run_htsim_rnic
    from simllm.compute import ModelDims
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord
    from simllm.goal import to_binary
    from simllm.placement import RankMapper, declared_manifest
    from simllm.traffic import plan_step_locality, render_fabric_phase_goal
    from simllm.traffic.locality import (
        CollectiveCommunicationPhase,
        classify_step_locality,
    )

    model = config["model"]
    operating = config["operating_point"]
    sampling = config["packet_sampling"]
    output_dir.mkdir(parents=True)
    tokens_per_rank = operating["local_batch_per_attention_dp_rank"] * (
        model["nextn"] + 1
    )
    dims = ModelDims(
        num_layers=1,
        hidden_size=model["hidden_size"],
        intermediate_size=model["intermediate_size"],
        num_heads=model["num_attention_heads"],
        num_kv_heads=model["num_key_value_heads"],
        head_size=model["head_dim"],
        vocab_size=model["vocab_size"],
        dtype_bytes=1,
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
        rank_mapper=mapper,
    )
    topology = _write_width_clos(
        output_dir / f"clos-{width}-400g.topo",
        width=width,
        gpus_per_node=operating["gpus_per_node"],
    )
    expected_messages = 2 * width * (width - 1)
    per_pair_bytes = (
        tokens_per_rank
        * model["num_experts_per_tok"]
        * model["hidden_size"]
        // width
    )
    expected_bytes = expected_messages * per_pair_bytes
    if plan.total_directed_bytes != expected_bytes:
        raise RuntimeError("packet projection failed directed-byte conservation")
    if plan.fabric_segments + plan.nvlink_segments != expected_messages:
        raise RuntimeError("packet projection failed message conservation")

    peer_subset = width > int(sampling["full_population_max_width"])
    simulated_phases = plan.phases
    if peer_subset:
        receiver_stride = int(sampling["receiver_stride_at_widest"])
        selected = tuple(range(0, width, receiver_stride))
        selected_phases = tuple(
            CollectiveCommunicationPhase(
                phase_id=classified.phase.phase_id,
                layer=classified.phase.layer,
                participants=classified.phase.participants,
                segments=tuple(
                    segment
                    for segment in classified.phase.segments
                    if segment.destination_rank in selected
                ),
                operation_id=classified.phase.operation_id,
            )
            for classified in plan.phases
        )
        simulated_phases = classify_step_locality(
            selected_phases,
            rank_mapper=mapper,
        ).phases

    phases = []
    for classified in simulated_phases:
        phase_name = classified.phase.phase_id.rsplit("-", 1)[-1]
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
        phases.append(
            {
                "phase": phase_name,
                "fabric_segments": len(classified.fabric_segments),
                "fabric_bytes": classified.fabric_bytes,
                "nvlink_segments": len(classified.nvlink_segments),
                "nvlink_bytes": classified.nvlink_bytes,
                "nvlink_service_ps": classified.nvlink_service_ps,
                "fabric_jct_ps": fabric_jct_ps,
                "phase_duration_ps": max(
                    fabric_jct_ps,
                    classified.nvlink_service_ps,
                ),
                "flow_count": len(flow_rows),
                "flow_payload_bytes": sum(flow.payload_bytes for flow in flow_rows),
                "fanin": _max_receiver_fanin(flow_rows),
                "goal_sha256": goal_sha256,
                "goal_binary_sha256": goal_binary_sha256,
                "completion_csv_sha256": completion_sha256,
            }
        )
    layer_packet_ps = sum(phase["phase_duration_ps"] for phase in phases)
    represented = sampling["represented_layer_executions"]
    simulated_messages = sum(
        len(phase.fabric_segments) + len(phase.nvlink_segments)
        for phase in simulated_phases
    )
    simulated_bytes = sum(
        phase.fabric_bytes + phase.nvlink_bytes for phase in simulated_phases
    )
    return {
        "expert_parallel": width,
        "sampled": True,
        "sample_label": (
            sampling["label_receiver_subset"]
            if peer_subset
            else sampling["label_full"]
        ),
        "sampling_rule": sampling["rule"],
        "peer_population": (
            "one receiver per node with every source retained"
            if peer_subset
            else "full"
        ),
        "peer_subset": peer_subset,
        "topology_sha256": _sha256_file(topology),
        "messages_per_sampled_layer": expected_messages,
        "per_pair_bytes": per_pair_bytes,
        "bytes_per_sampled_layer": expected_bytes,
        "simulated_messages_per_sampled_layer": simulated_messages,
        "simulated_bytes_per_sampled_layer": simulated_bytes,
        "simulated_message_fraction": simulated_messages / expected_messages,
        "represented_layer_executions": represented,
        "represented_messages": expected_messages * represented,
        "represented_bytes": expected_bytes * represented,
        "packet_dispatch_combine_ps": layer_packet_ps * represented,
        "packet_dispatch_combine_ms": layer_packet_ps * represented / 1_000_000_000,
        "phases": phases,
    }


def _run_evaluation(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    from simllm.calibration.external_db import ExternalOperationDatabase
    from simllm.calibration.external_nccl import ExternalNcclDatabase
    from simllm.calibration.external_pass import ExternalModelConfig, ExternalPassModel

    htsim = _configured_path(HTSIM_ENV)
    txt2bin = _configured_path(TXT2BIN_ENV)
    live = _live_sdk_subprocess(config)
    operation_database = ExternalOperationDatabase.load()
    nccl_database = ExternalNcclDatabase.load()
    widths = []
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
        packet = _run_packet_width(
            config,
            width=width,
            output_dir=output_dir / f"ep-{width}",
            htsim=htsim,
            txt2bin=txt2bin,
        )
        packet_step = composed.total.latency_ms - dispatch + packet[
            "packet_dispatch_combine_ms"
        ]
        widths.append(
            {
                "expert_parallel": width,
                "composer_decode_step_ms": composed.total.latency_ms,
                "composer_decode_step_hex": composed.total.hex,
                "composer_dispatch_ms": dispatch,
                "composer_dispatch_hex": dispatch.hex(),
                "non_dispatch_timing_base_ms": composed.total.latency_ms - dispatch,
                "packet_priced_step_ms": packet_step,
                "packet_to_aiconfigurator_ratio": (
                    packet_step / float(frozen["live_decode_step_ms"])
                ),
                "operation_evidence_classes": {
                    entry.operation: entry.evidence_class for entry in composed.operations
                },
                "packet_evidence_class": "SIM-DERIVED",
                "packet": packet,
            }
        )
    return {
        "live_sdk": live,
        "widths": widths,
        "operation_database_identity": operation_database.source.as_dict(),
        "operation_database_payload_sha256": operation_database.payload_sha256,
        "nccl_database_identity": nccl_database.source.as_dict(),
        "nccl_database_payload_sha256": nccl_database.payload_sha256,
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
    rows = []
    e_cells = []
    c_cells = []
    for frozen in config["widths"]:
        width = int(frozen["expert_parallel"])
        live = live_by_width[width]
        composed = composed_by_width[width]
        dispatch_passed = composed["composer_dispatch_hex"] == frozen[
            "live_dispatch_hex"
        ]
        quotient = composed["composer_decode_step_ms"] / live["decode_step_ms"]
        composition_passed = 0.98 <= quotient <= 1.02
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
        rows.append(
            {
                "expert_parallel": width,
                "sampled": composed["packet"]["sampled"],
                "sample_label": composed["packet"]["sample_label"],
                "peer_subset": composed["packet"]["peer_subset"],
                "aiconfigurator_step_ms": live["decode_step_ms"],
                "aiconfigurator_dispatch_ms": live["dispatch_ms"],
                "dispatch_share": live["dispatch_ms"] / live["decode_step_ms"],
                "composer_step_ms": composed["composer_decode_step_ms"],
                "composer_quotient": quotient,
                "packet_dispatch_combine_ms": composed["packet"][
                    "packet_dispatch_combine_ms"
                ],
                "simllm_step_ms": composed["packet_priced_step_ms"],
                "ratio": composed["packet_to_aiconfigurator_ratio"],
                "represented_messages": composed["packet"]["represented_messages"],
                "represented_bytes": composed["packet"]["represented_bytes"],
                "simulated_messages_per_sampled_layer": composed["packet"][
                    "simulated_messages_per_sampled_layer"
                ],
                "simulated_message_fraction": composed["packet"][
                    "simulated_message_fraction"
                ],
                "fabric_messages_per_sampled_layer": sum(
                    phase["fabric_segments"] for phase in composed["packet"]["phases"]
                ),
                "nvlink_messages_per_sampled_layer": sum(
                    phase["nvlink_segments"] for phase in composed["packet"]["phases"]
                ),
                "e_passed": dispatch_passed,
                "c_passed": composition_passed,
            }
        )

    ratios = [row["ratio"] for row in rows]
    n1 = all(right >= left for left, right in pairwise(ratios))
    n2 = ratios[-1] >= 1.25
    widest = composed_by_width[256]["packet"]
    fanin_phases = [phase["fanin"] for phase in widest["phases"]]
    n3 = {
        "expert_parallel": 256,
        "phases": [
            {"phase": phase["phase"], **phase["fanin"]}
            for phase in widest["phases"]
        ],
        "maximum_receiver_ingress_occupancy_ps": max(
            phase["ingress_occupancy_ps"] for phase in fanin_phases
        ),
        "maximum_simultaneous_senders_per_receiver": max(
            phase["maximum_simultaneous_senders"] for phase in fanin_phases
        ),
        "passed": all(
            phase["ingress_occupancy_ps"] > 0
            and phase["maximum_simultaneous_senders"] > 0
            for phase in fanin_phases
        ),
    }
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
        },
        "N": {
            "passed": int(n1) + int(n2) + int(n3["passed"]),
            "denominator": 3,
            "bands": {
                "N1": {
                    "passed": n1,
                    "rule": "ratios are non-decreasing with expert-parallel width",
                    "ratios": ratios,
                },
                "N2": {
                    "passed": n2,
                    "rule": "widest ratio is at least 1.25",
                    "actual": ratios[-1],
                    "lower": 1.25,
                },
                "N3": n3,
            },
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
        ),
        "FG-3": config["model"]["nextn"] == 3,
        "FG-4": AI_CONFIGURATOR_TRAFFIC_MODEL != SIMLLM_TRAFFIC_MODEL
        and "not equivalent" in TRAFFIC_COMPARISON_RULE,
        "FG-5": all(
            row["sampled"]
            and row["sample_label"]
            and (
                not row["peer_subset"]
                or "one receiver per node" in row["sample_label"]
            )
            for row in rows
        ),
        "FG-6": deterministic,
        "FG-7": chronology,
    }
    return {
        "families": families,
        "fatal_guards": fatal_guards,
        "run_state": "nonvoid" if all(fatal_guards.values()) else "void",
        "n3": n3,
    }, rows


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    columns = (
        "expert_parallel",
        "sampled",
        "sample_label",
        "peer_subset",
        "aiconfigurator_step_ms",
        "aiconfigurator_dispatch_ms",
        "dispatch_share",
        "composer_step_ms",
        "composer_quotient",
        "packet_dispatch_combine_ms",
        "simllm_step_ms",
        "ratio",
        "represented_messages",
        "represented_bytes",
        "simulated_messages_per_sampled_layer",
        "simulated_message_fraction",
        "fabric_messages_per_sampled_layer",
        "nvlink_messages_per_sampled_layer",
        "e_passed",
        "c_passed",
    )
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _render_figures(record_path: Path, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
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
    }


def _coordinator(bulk_root: Path, *, write_tracked: bool) -> dict[str, Any]:
    started = time.monotonic()
    config = _load_config()
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
    routed_pair_bytes = (
        tokens_per_rank
        * config["model"]["num_experts_per_tok"]
        * config["model"]["hidden_size"]
        // widest_width
    )
    per_rank_full_width_bytes = 2 * (widest_width - 1) * routed_pair_bytes
    link_bytes_per_second = (
        config["operating_point"]["link_rate_bits_per_second"] // 8
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
        "operating_point": config["operating_point"],
        "sampling": config["packet_sampling"],
        "physical_sanity": {
            "widest_expert_parallel": widest_width,
            "routed_fp8_payload_per_rank_pair_bytes": routed_pair_bytes,
            "dispatch_plus_combine_bytes_per_rank": per_rank_full_width_bytes,
            "link_bytes_per_second": link_bytes_per_second,
            "full_rank_serialization_floor_microseconds": (
                per_rank_full_width_bytes / link_bytes_per_second * 1_000_000
            ),
        },
        "traffic_model_disclosure": {
            "AIConfigurator": AI_CONFIGURATOR_TRAFFIC_MODEL,
            "SimLLM": SIMLLM_TRAFFIC_MODEL,
            "comparison_rule": TRAFFIC_COMPARISON_RULE,
        },
        "evidence_classes": {
            "compute_and_external_dispatch": "MEASURED-EXTERNAL",
            "packet_dispatch_and_combine": "SIM-DERIVED",
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
        "n3": scored["n3"],
        "family_tallies": scored["families"],
        "fatal_guards": scored["fatal_guards"],
        "run_state": scored["run_state"],
        "qwen_reference_ratio": QWEN_REFERENCE_RATIO,
        "figures": {
            "png": "figures/minimax_ep_scaling.png",
            "pdf": "figures/minimax_ep_scaling.pdf",
        },
    }
    attempt_record = attempt / "record.json"
    attempt_record.write_bytes(_json_bytes(record))
    (attempt / "results.csv").write_bytes(_csv_bytes(rows))
    _render_figures(attempt_record, attempt / "figures")
    elapsed = time.monotonic() - started
    record["family_tallies"]["W"]["elapsed_seconds"] = elapsed
    record["family_tallies"]["W"]["passed"] = int(elapsed <= WALL_BOUND_SECONDS)
    attempt_record.write_bytes(_json_bytes(record))
    if write_tracked:
        TRACKED_RECORD.write_bytes(_json_bytes(record))
        TRACKED_CSV.write_bytes(_csv_bytes(rows))
        TRACKED_FIGURES.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "pdf"):
            source = attempt / "figures" / f"minimax_ep_scaling.{suffix}"
            destination = TRACKED_FIGURES / source.name
            destination.write_bytes(source.read_bytes())
    return record


def _validate_record(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != SCHEMA:
        raise SystemExit("record has an unsupported schema")
    if len(record.get("rows", ())) != 4:
        raise SystemExit("record must contain four expert-parallel rows")
    if set(record.get("family_tallies", ())) != {"E", "C", "N", "W"}:
        raise SystemExit("record family inventory is incomplete")
    if set(record.get("fatal_guards", ())) != {
        "FG-1",
        "FG-2",
        "FG-3",
        "FG-4",
        "FG-5",
        "FG-6",
        "FG-7",
    }:
        raise SystemExit("record fatal-guard inventory is incomplete")
    return record


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
        record = _validate_record(TRACKED_RECORD)
        print(f"run_state={record['run_state']}")
        return 0
    raw_bulk = args.bulk_root or os.environ.get(BULK_ROOT_ENV)
    if raw_bulk is None:
        raise SystemExit(f"pass --bulk-root or set {BULK_ROOT_ENV}")
    record = _coordinator(Path(raw_bulk), write_tracked=args.write_tracked)
    print(
        f"run_state={record['run_state']} "
        f"elapsed_seconds={record['family_tallies']['W']['elapsed_seconds']:.6f}"
    )
    for family in ("E", "C", "N", "W"):
        tally = record["family_tallies"][family]
        print(f"{family}={tally['passed']}/{tally['denominator']}")
    print(f"fatal_guards={record['fatal_guards']}")
    return 0 if record["run_state"] == "nonvoid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
