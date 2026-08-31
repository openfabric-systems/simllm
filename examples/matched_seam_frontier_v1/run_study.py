#!/usr/bin/env python3
"""Run the frozen matched-seam frontier study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))
CONFIG_PATH = STUDY_DIR / "study_config.json"
EXPECTATIONS_PATH = STUDY_DIR / "expectations.md"
EXPECTATIONS_V2_PATH = STUDY_DIR / "expectations_v2.md"
ADJUSTMENTS_PATH = STUDY_DIR / "external_adjustments.json"
AGG_PATH = REPOSITORY_ROOT / "examples/frontier_comparison_v1/external/agg_pareto.csv"
DISAGG_PATH = REPOSITORY_ROOT / "examples/frontier_comparison_v1/external/disagg_pareto.csv"
ARTIFACT_PATH = (
    REPOSITORY_ROOT
    / "offline/calibration/external-databases"
    / "85e72f990f00ea457de522d0b773e678f5e067740689912df5646f6296273284"
)
PARITY_DIR = REPOSITORY_ROOT / "examples/external_db_parity_v1"
RESULT_PATH = STUDY_DIR / "record.json"
CSV_PATH = STUDY_DIR / "results.csv"
PDF_PATH = STUDY_DIR / "figures/matched-seam-frontier.pdf"
PNG_PATH = STUDY_DIR / "figures/matched-seam-frontier.png"

SCHEMA = "simllm-matched-seam-frontier-record-v2"
EXPECTATIONS_COMMIT = "4c7ec887ed86abb09d3b15f93e9b04f521252819"
CORRECTED_EXPECTATIONS_COMMIT = "4ed8d1aae540d4cf548eace3c5b4008ba3500d9b"
SERVICE_FREEZE_COMMIT = "5760301efb430aee99573ac4f89f1d572c040614"
BINDING_COMMIT = "9e6782d04a99fd773d08dbf422df6d8ce9c81dbe"
EXTERNAL_VENV_ENV = "SIMLLM_EXTERNAL_AIC_VENV"
BULK_ROOT_ENV = "SIMLLM_P3T_T1_BULK_ROOT"
HTSIM_ENV = "SIMLLM_HTSIM_RNIC"
TXT2BIN_ENV = "SIMLLM_TXT2BIN"

PICOSECONDS_PER_SECOND = 1_000_000_000_000
LINKSPEED_BPS = 400_000_000_000
OUTPUT_TOKENS = 500
KV_BYTES = 458_752_000
PREFILL_TP = 4
PACKET_RANKS = 16
PACKET_TAG = 62_089
PCIE_SUBMISSION_PS = 20_000_000
WALL_CEILING_SECONDS = 600.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name}: expected a JSON object")
    return value


def _adjustment_table() -> dict[str, Any]:
    table = _load_json(ADJUSTMENTS_PATH)
    if table.get("schema") != "simllm-matched-seam-external-adjustments-v1":
        raise ValueError("external adjustment table has an unexpected schema")
    adjustments = table.get("adjustments")
    if not isinstance(adjustments, list) or not adjustments:
        raise ValueError("external adjustment table must contain adjustment rows")
    identifiers = [str(row["id"]) for row in adjustments]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("external adjustment table contains duplicate IDs")
    return table


def _adjustments_by_id(table: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in table["adjustments"]}


def _adjustment_float(table: dict[str, Any], adjustment_id: str) -> float:
    return float(_adjustments_by_id(table)[adjustment_id]["value"])


def _validate_adjustment_config(config: dict[str, Any], table: dict[str, Any]) -> None:
    mapping = {
        "prefill_latency_correction_hex": "prefill_latency_correction",
        "decode_latency_correction_hex": "decode_latency_correction",
        "prefill_rate_matching_degradation_hex": "prefill_rate_matching_degradation",
        "decode_rate_matching_degradation_hex": "decode_rate_matching_degradation",
        "autoscale_ttft_correction_hex": "autoscale_ttft_correction",
    }
    for config_key, adjustment_id in mapping.items():
        configured = float.fromhex(config["composition"][config_key])
        declared = _adjustment_float(table, adjustment_id)
        if configured.hex() != declared.hex():
            raise ValueError(
                f"{config_key} differs from external adjustment {adjustment_id}"
            )
    frozen = config["frozen_inputs"]["external_adjustments"]
    if frozen["path"] != ADJUSTMENTS_PATH.relative_to(REPOSITORY_ROOT).as_posix():
        raise ValueError("external adjustment table path differs from the frozen input")
    if frozen["sha256"] != _sha256(ADJUSTMENTS_PATH):
        raise ValueError("external adjustment table hash differs from the frozen input")


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _git_output(*args: str) -> str:
    completed = _git(*args)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def _is_ancestor(older: str, newer: str = "HEAD") -> bool:
    return _git("merge-base", "--is-ancestor", older, newer).returncode == 0


def _fraction_json(value: Fraction) -> dict[str, int | float]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def _fraction(value: dict[str, Any]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def _decimal_fraction(value: str) -> Fraction:
    return Fraction(Decimal(value))


def _float_hex_fraction(value: str) -> Fraction:
    return Fraction.from_float(float.fromhex(value))


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _new_attempt(bulk_root: Path) -> tuple[Path, int]:
    bulk_root.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in bulk_root.glob("attempt-*"):
        try:
            numbers.append(int(path.name.rsplit("-", 1)[1]))
        except ValueError:
            continue
    number = max(numbers, default=0) + 1
    attempt = bulk_root / f"attempt-{number:04d}"
    attempt.mkdir(exist_ok=False)
    return attempt, number


def _portable_artifact(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _tracked_immutability_paths() -> tuple[Path, ...]:
    external = tuple(
        sorted(
            path
            for path in (REPOSITORY_ROOT / "examples/frontier_comparison_v1/external").rglob("*")
            if path.is_file()
        )
    )
    parity = tuple(
        PARITY_DIR / name
        for name in (
            "RESULTS.md",
            "expectations.md",
            "query_points.json",
            "query_points_supplement.json",
            "record.json",
            "results.csv",
        )
    )
    return (*external, *parity)


def _tracked_hashes() -> dict[str, str]:
    return {
        path.relative_to(REPOSITORY_ROOT).as_posix(): _sha256(path)
        for path in _tracked_immutability_paths()
    }


def _external_curve(path: Path, *, disaggregated: bool) -> list[dict[str, Any]]:
    rows = []
    for index, raw in enumerate(_csv_rows(path), start=1):
        row = {
            "row": index,
            "evidence_class": "MEASURED-EXTERNAL",
            "x_tokens_per_second_per_user": raw["tokens/s/user"],
            "y_tokens_per_second_per_gpu": raw["tokens/s/gpu"],
            "num_total_gpus": int(raw["num_total_gpus"]),
            "ttft_ms": raw["ttft"],
            "tpot_ms": raw["tpot"],
        }
        if disaggregated:
            row["configuration"] = {
                "prefill_tp": int(raw["(p)tp"]),
                "prefill_workers": int(raw["(p)workers"]),
                "decode_tp": int(raw["(d)tp"]),
                "decode_workers": int(raw["(d)workers"]),
                "decode_batch": int(raw["(d)bs"]),
            }
        else:
            row["configuration"] = {
                "tensor_parallel": int(raw["tp"]),
                "workers": int(raw["num_total_gpus"]) // int(raw["tp"]),
                "batch": int(raw["bs"]),
            }
        rows.append(row)
    return rows


def _family_r_quotients(
    services: dict[str, float],
) -> list[tuple[int, Fraction]]:
    quotients = []
    for external in _external_curve(DISAGG_PATH, disaggregated=True):
        configuration = external["configuration"]
        service_id = (
            f"decode-tp{configuration['decode_tp']}-b{configuration['decode_batch']}"
        )
        quotient = Fraction.from_float(services[service_id]) / _decimal_fraction(
            external["tpot_ms"]
        )
        quotients.append((int(external["row"]), quotient))
    return quotients


def _family_r_sensitivity(
    *,
    config: dict[str, Any],
    adjustment_table: dict[str, Any],
    baseline_services: dict[str, float],
) -> list[dict[str, Any]]:
    from simllm.calibration.external_db import ExternalOperationDatabase
    from simllm.deploy import ExternalQwen32BDeploymentBinding

    baseline = _family_r_quotients(baseline_services)

    def evaluate_removed(adjustment_id: str) -> list[tuple[int, Fraction]]:
        database = ExternalOperationDatabase.load(ARTIFACT_PATH)
        decode_scale = _adjustment_float(
            adjustment_table, "decode_latency_correction"
        )
        if adjustment_id == "decode_latency_correction":
            decode_scale = float(
                _adjustments_by_id(adjustment_table)[adjustment_id]["removal_value"]
            )
        elif adjustment_id == "memory_bandwidth_empirical_scale":
            database.system_spec["gpu"]["mem_bw_empirical_scaling_factor"] = float(
                _adjustments_by_id(adjustment_table)[adjustment_id]["removal_value"]
            )
        elif adjustment_id == "memory_empirical_constant_latency":
            database.system_spec["gpu"]["mem_empirical_constant_latency"] = float(
                _adjustments_by_id(adjustment_table)[adjustment_id]["removal_value"]
            )
        else:
            raise ValueError(
                f"Family R sensitivity lacks a remove-one evaluator for {adjustment_id}"
            )
        binding = ExternalQwen32BDeploymentBinding(database)
        services = {}
        for oracle in config["oracles"]["decode"]:
            value = binding.decode_service(
                tensor_parallel=int(oracle["tensor_parallel"]),
                batch_size=int(oracle["batch_size"]),
                isl=int(oracle["isl"]),
                osl=int(oracle["osl"]),
                prefix=int(oracle["prefix"]),
                stride=int(oracle["stride"]),
                latency_correction_scale=decode_scale,
            )
            services[value.configuration_id] = value.service_ms
        return _family_r_quotients(services)

    rows = []
    for adjustment in adjustment_table["adjustments"]:
        adjustment_id = str(adjustment["id"])
        if adjustment["family_r_reachable"]:
            removed = evaluate_removed(adjustment_id)
            method = "full decode-composition reevaluation with only this factor removed"
        else:
            removed = baseline
            method = (
                "complete Family R dependency trace proves this factor unreachable; "
                "the baseline decode reevaluation is reused exactly"
            )
        quotients = [quotient for _, quotient in removed]
        rows.append(
            {
                "adjustment_id": adjustment_id,
                "baseline_reachable": bool(adjustment["family_r_reachable"]),
                "removed_value": adjustment["removal_value"],
                "minimum_quotient": _fraction_json(min(quotients)),
                "maximum_quotient": _fraction_json(max(quotients)),
                "method": method,
                "rows": [
                    {"row": row_number, "quotient": _fraction_json(quotient)}
                    for row_number, quotient in removed
                ],
            }
        )
    return rows


def _candidate(
    raw: dict[str, str],
    *,
    row_number: int,
    disaggregated: bool,
    inventory_sha256: str,
) -> Any:
    from simllm.deploy import (
        BudgetSpec,
        DeploymentCandidate,
        FabricSpec,
        ModelRef,
        PoolSpec,
        SlaSpec,
        WorkloadPoint,
    )

    total_gpus = int(raw["num_total_gpus"])
    if disaggregated:
        pools = (
            PoolSpec(
                role="prefill",
                engines=int(raw["(p)workers"]),
                gpus_per_engine=int(raw["(p)tp"]),
                tensor_parallel=int(raw["(p)tp"]),
                pipeline_parallel=1,
                expert_parallel=1,
                data_parallel=1,
                device="h200",
            ),
            PoolSpec(
                role="decode",
                engines=int(raw["(d)workers"]),
                gpus_per_engine=int(raw["(d)tp"]),
                tensor_parallel=int(raw["(d)tp"]),
                pipeline_parallel=1,
                expert_parallel=1,
                data_parallel=1,
                device="h200",
            ),
        )
        candidate_id = f"external-disagg-row-{row_number:02d}"
    else:
        tensor_parallel = int(raw["tp"])
        pools = (
            PoolSpec(
                role="combined",
                engines=total_gpus // tensor_parallel,
                gpus_per_engine=tensor_parallel,
                tensor_parallel=tensor_parallel,
                pipeline_parallel=1,
                expert_parallel=1,
                data_parallel=1,
                device="h200",
            ),
        )
        candidate_id = f"external-agg-row-{row_number:02d}"
    return DeploymentCandidate(
        candidate_id=candidate_id,
        model=ModelRef(
            framework="external-trtllm",
            model_id="Qwen/Qwen3-32B-FP8",
            inventory_sha256=inventory_sha256,
        ),
        pools=pools,
        fabric=FabricSpec(
            inter_node_bits_per_second=LINKSPEED_BPS,
            intra_node_bytes_per_second=900_000_000_000,
        ),
        workload=WorkloadPoint(
            arrival_rate_rps=None,
            prompt_tokens=int(raw["isl"]),
            output_tokens=int(raw["osl"]),
            kv_context_tokens=int(raw["isl"]) + int(raw["osl"]) // 2,
        ),
        sla=SlaSpec(tpot_target_ps=None, ttft_target_ps=None),
        budget=BudgetSpec(max_gpus=total_gpus, max_nodes=None),
    )


def _model_work(inventory_sha256: str) -> Any:
    from simllm.deploy import ModelWork

    return ModelWork(
        kernel_name="external-qwen3-32b-pass",
        flops_per_batch_item=1,
        static_logical_hbm_bytes=0,
        dynamic_hbm_bytes_per_batch_item=0,
        logical_collective_bytes_per_gpu_per_batch_item=0,
        inventory_sha256=inventory_sha256,
        source="identity only; MEASURED-EXTERNAL surface owns positive service",
    )


def _envelope() -> Any:
    from simllm.deploy import EnvelopeSpec

    return EnvelopeSpec(
        device="h200",
        peak_flops_per_second=1_979_000_000_000_000,
        hbm_bytes_per_second=4_800_000_000_000,
        efficiency=1.0,
        source="unused because MEASURED-EXTERNAL owns the scored service",
    )


def _packet_cell(
    *,
    decode_tp: int,
    packet_root: Path,
    txt2bin: Path,
    htsim_rnic: Path,
) -> dict[str, Any]:
    from simllm.backends import HtsimRnicConfig, run_htsim_rnic
    from simllm.goal import GoalTrace, to_binary
    from simllm.traffic.patterns import pairwise_all_to_allv

    cell_root = packet_root / f"tp4-to-tp{decode_tp}"
    cell_root.mkdir(parents=True, exist_ok=False)
    prefill_ranks = tuple(range(PREFILL_TP))
    decode_ranks = tuple(range(8, 8 + decode_tp))
    pairs = tuple((source, destination) for source in prefill_ranks for destination in decode_ranks)
    quotient, remainder = divmod(KV_BYTES, len(pairs))
    send_bytes = {
        pair: quotient + (1 if index < remainder else 0) for index, pair in enumerate(pairs)
    }
    if sum(send_bytes.values()) != KV_BYTES:
        raise AssertionError("packet KV split did not conserve bytes")
    trace = GoalTrace(PACKET_RANKS)
    after = {
        rank: trace.rank(rank).calc(
            PCIE_SUBMISSION_PS // 1_000,
            operation_id=f"matched-seam:tp4-to-tp{decode_tp}",
        )
        for rank in prefill_ranks
    }
    pairwise_all_to_allv(
        trace,
        list(range(PACKET_RANKS)),
        send_bytes,
        PACKET_TAG + decode_tp,
        after,
        operation_id=f"matched-seam:tp4-to-tp{decode_tp}",
    )
    goal_path = trace.write(cell_root / "kv-redistribution.goal")
    binary_path = to_binary(
        goal_path,
        cell_root / "kv-redistribution.bin",
        tool=txt2bin,
    )
    completion_path = cell_root / "flow-completions.csv"
    run = run_htsim_rnic(
        HtsimRnicConfig(
            goal_bin=binary_path,
            profile="rnic-nn",
            linkspeed_bps=LINKSPEED_BPS,
            completion_csv=completion_path,
        ),
        binary=htsim_rnic,
    )
    if len(run.flows) != len(pairs):
        raise AssertionError("packet completion count differs from the GOAL ledger")
    observed = sorted((flow.source, flow.destination, flow.payload_bytes) for flow in run.flows)
    expected = sorted(
        (source, destination, payload) for (source, destination), payload in send_bytes.items()
    )
    if observed != expected:
        raise AssertionError("packet completion rows do not conserve endpoints and bytes")
    first_start_ps = min(flow.start_time_ps for flow in run.flows)
    last_completion_ps = max(flow.completion_time_ps for flow in run.flows)
    packet_service_ps = last_completion_ps - first_start_ps
    sender_floor_ps = KV_BYTES * 8 * PICOSECONDS_PER_SECOND // (PREFILL_TP * LINKSPEED_BPS)
    receiver_floor_ps = KV_BYTES * 8 * PICOSECONDS_PER_SECOND // (decode_tp * LINKSPEED_BPS)
    serialization_floor_ps = max(sender_floor_ps, receiver_floor_ps)
    if packet_service_ps < serialization_floor_ps:
        raise AssertionError("packet service beat its endpoint serialization floor")
    manifest = {
        "schema": "simllm-matched-seam-packet-cell-v1",
        "configuration_id": f"tp4-to-tp{decode_tp}",
        "profile": "rnic-nn",
        "evidence_class": "SIM-DERIVED",
        "aggregate_kv_bytes": KV_BYTES,
        "flow_count": len(run.flows),
        "linkspeed_bps": LINKSPEED_BPS,
        "first_start_ps": first_start_ps,
        "last_completion_ps": last_completion_ps,
        "packet_service_ps": packet_service_ps,
        "serialization_floor_ps": serialization_floor_ps,
        "quiescent": run.quiescent,
    }
    manifest_path = cell_root / "packet-cell.json"
    _write_json(manifest_path, manifest)
    return {
        **manifest,
        "artifacts": {
            "goal": _portable_artifact(goal_path, packet_root),
            "goal_binary": _portable_artifact(binary_path, packet_root),
            "completion_csv": _portable_artifact(completion_path, packet_root),
            "manifest": _portable_artifact(manifest_path, packet_root),
        },
    }


def _point_xy(point: dict[str, Any]) -> tuple[Fraction, Fraction]:
    return (
        _fraction(point["x_tokens_per_second_per_user"]),
        _fraction(point["y_tokens_per_second_per_gpu"]),
    )


def _local_worker(
    *,
    packet_root: Path,
    txt2bin: Path,
    htsim_rnic: Path,
) -> dict[str, Any]:
    import simllm.calibration.external_db as external_db_module
    import simllm.deploy.estimator as estimator_module
    from simllm.calibration.external_db import ExternalOperationDatabase
    from simllm.deploy import (
        EstimatorInputs,
        EvidenceClass,
        ExternalQwen32BDeploymentBinding,
        candidate_key,
        candidate_to_json,
        estimate_decode_step,
        estimate_prefill_request,
        estimate_stamp_to_json,
        validate_external_scored_stamp,
        weak_dominance_pareto,
    )

    config = _load_json(CONFIG_PATH)
    adjustment_table = _adjustment_table()
    _validate_adjustment_config(config, adjustment_table)
    applied_adjustments: set[str] = set()

    def use_adjustment(adjustment_id: str) -> float:
        applied_adjustments.add(adjustment_id)
        return _adjustment_float(adjustment_table, adjustment_id)

    database = ExternalOperationDatabase.load(ARTIFACT_PATH)
    memory_bandwidth_scale = use_adjustment("memory_bandwidth_empirical_scale")
    memory_constant = use_adjustment("memory_empirical_constant_latency")
    if float(database.system_spec["gpu"]["mem_bw_empirical_scaling_factor"]).hex() != (
        memory_bandwidth_scale.hex()
    ):
        raise ValueError("imported memory-bandwidth adjustment differs from the table")
    if float(database.system_spec["gpu"]["mem_empirical_constant_latency"]).hex() != (
        memory_constant.hex()
    ):
        raise ValueError("imported memory-latency adjustment differs from the table")

    context_calls = 0
    original_context_attention = external_db_module.ExternalQwen32BPassModel._context_attention

    def traced_context_attention(model: Any, **kwargs: Any) -> Any:
        nonlocal context_calls
        context_calls += 1
        applied_adjustments.add("context_attention_extra_latency_correction")
        return original_context_attention(model, **kwargs)

    roofline_calls = 0
    original_roofline_provider = estimator_module.RooflineProvider

    class ForbiddenScoredRooflineProvider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            nonlocal roofline_calls
            roofline_calls += 1
            raise AssertionError(
                "SimLLM RooflineProvider reached the corrected matched-seam scored path"
            )

    external_db_module.ExternalQwen32BPassModel._context_attention = traced_context_attention
    estimator_module.RooflineProvider = ForbiddenScoredRooflineProvider
    binding = ExternalQwen32BDeploymentBinding(database)
    decode_scale = use_adjustment("decode_latency_correction")
    prefill_scale = use_adjustment("prefill_latency_correction")
    decode_services = {}
    surfaces: dict[int, list[Any]] = {}
    service_rows = []
    for oracle in config["oracles"]["decode"]:
        value = binding.decode_service(
            tensor_parallel=oracle["tensor_parallel"],
            batch_size=oracle["batch_size"],
            isl=oracle["isl"],
            osl=oracle["osl"],
            prefix=oracle["prefix"],
            stride=oracle["stride"],
            latency_correction_scale=decode_scale,
        )
        decode_services[value.configuration_id] = value
        surfaces.setdefault(value.tensor_parallel, []).append(value.as_batch_service_point())
        service_rows.append(
            {
                "id": value.configuration_id,
                "phase": value.phase,
                "tensor_parallel": value.tensor_parallel,
                "batch_size": value.batch_size,
                "service_ms": value.service_ms,
                "service_ms_hex": value.service_ms_hex,
                "total_ms_hex": value.total_ms_hex,
                "service_ps": value.service_ps,
                "source": value.source,
                "entry_key_sha256": value.entry_key_sha256,
                "evidence_class": value.evidence_class,
            }
        )
    prefill_services = {}
    for oracle in config["oracles"]["prefill"]:
        value = binding.prefill_service(
            tensor_parallel=oracle["tensor_parallel"],
            batch_size=oracle["batch_size"],
            isl=oracle["isl"],
            prefix=oracle["prefix"],
            latency_correction_scale=prefill_scale,
        )
        prefill_services[value.configuration_id] = value
        service_rows.append(
            {
                "id": value.configuration_id,
                "phase": value.phase,
                "tensor_parallel": value.tensor_parallel,
                "batch_size": value.batch_size,
                "service_ms": value.service_ms,
                "service_ms_hex": value.service_ms_hex,
                "total_ms_hex": value.total_ms_hex,
                "service_ps": value.service_ps,
                "source": value.source,
                "entry_key_sha256": value.entry_key_sha256,
                "evidence_class": value.evidence_class,
            }
        )
    raw_prefill = binding.prefill_service(
        tensor_parallel=4,
        batch_size=1,
        isl=4000,
        prefix=500,
        latency_correction_scale=1.0,
    )

    packet_cells = {
        decode_tp: _packet_cell(
            decode_tp=decode_tp,
            packet_root=packet_root,
            txt2bin=txt2bin,
            htsim_rnic=htsim_rnic,
        )
        for decode_tp in (2, 4, 8)
    }

    inventory_sha256 = str(database.manifest["source"]["model_config_sha256"])
    model_work = _model_work(inventory_sha256)
    envelope = _envelope()
    disagg_raw = _csv_rows(DISAGG_PATH)
    agg_raw = _csv_rows(AGG_PATH)
    disagg_candidates = [
        _candidate(
            row,
            row_number=index,
            disaggregated=True,
            inventory_sha256=inventory_sha256,
        )
        for index, row in enumerate(disagg_raw, start=1)
    ]
    agg_candidates = [
        _candidate(
            row,
            row_number=index,
            disaggregated=False,
            inventory_sha256=inventory_sha256,
        )
        for index, row in enumerate(agg_raw, start=1)
    ]
    prefill_factor = Fraction.from_float(
        use_adjustment("prefill_rate_matching_degradation")
    )
    decode_factor = Fraction.from_float(
        use_adjustment("decode_rate_matching_degradation")
    )
    prefill_service = prefill_services["prefill-tp4-b1"]
    ideal_points = []
    packet_points = []
    for row_number, (raw, candidate) in enumerate(
        zip(disagg_raw, disagg_candidates, strict=True), start=1
    ):
        decode_tp = int(raw["(d)tp"])
        decode_batch = int(raw["(d)bs"])
        decode = estimate_decode_step(
            candidate,
            decode_batch,
            EstimatorInputs(
                model_work=model_work,
                envelopes={"h200": envelope},
                surfaces=tuple(sorted(surfaces[decode_tp], key=lambda point: point.batch_size)),
                surface_evidence=EvidenceClass.MEASURED_EXTERNAL,
                surface_source=f"external decode surface tp{decode_tp}",
            ),
        )
        prefill = estimate_prefill_request(
            candidate,
            EstimatorInputs(
                model_work=model_work,
                envelopes={"h200": envelope},
                prefill_service=prefill_service.as_term(),
                handoff_ps=0,
                handoff_source="identity network seam has zero added handoff service",
            ),
        )
        validate_external_scored_stamp(decode.stamp)
        validate_external_scored_stamp(prefill.stamp)
        prefill_workers = int(raw["(p)workers"])
        decode_workers = int(raw["(d)workers"])
        used_gpus = int(raw["num_total_gpus"])
        prefill_capacity = (
            Fraction(prefill_workers * PICOSECONDS_PER_SECOND, prefill.request_ps) * prefill_factor
        )
        decode_capacity = (
            Fraction(
                decode_workers * decode_batch * PICOSECONDS_PER_SECOND,
                OUTPUT_TOKENS * decode.step_ps,
            )
            * decode_factor
        )
        request_capacity = min(prefill_capacity, decode_capacity)
        x_value = Fraction(PICOSECONDS_PER_SECOND, decode.step_ps)
        y_value = request_capacity * OUTPUT_TOKENS / used_gpus
        point = {
            "row": row_number,
            "candidate_id": candidate.candidate_id,
            "candidate_key": candidate_key(candidate),
            "configuration": {
                "prefill_tp": int(raw["(p)tp"]),
                "prefill_workers": prefill_workers,
                "decode_tp": decode_tp,
                "decode_workers": decode_workers,
                "decode_batch": decode_batch,
                "used_gpus": used_gpus,
            },
            "evidence_class": "MEASURED-EXTERNAL",
            "decode_step_ps": decode.step_ps,
            "prefill_request_ps": prefill.request_ps,
            "prefill_capacity_requests_per_second": _fraction_json(prefill_capacity),
            "decode_capacity_requests_per_second": _fraction_json(decode_capacity),
            "request_capacity_per_second": _fraction_json(request_capacity),
            "capacity_limiter": "prefill" if prefill_capacity <= decode_capacity else "decode",
            "x_tokens_per_second_per_user": _fraction_json(x_value),
            "y_tokens_per_second_per_gpu": _fraction_json(y_value),
            "decode_stamp": estimate_stamp_to_json(decode.stamp),
            "prefill_stamp": estimate_stamp_to_json(prefill.stamp),
        }
        ideal_points.append(point)

        packet_cell = packet_cells[decode_tp]
        packet_prefill_ps = prefill.request_ps + int(packet_cell["packet_service_ps"])
        packet_prefill_capacity = (
            Fraction(prefill_workers * PICOSECONDS_PER_SECOND, packet_prefill_ps) * prefill_factor
        )
        packet_request_capacity = min(packet_prefill_capacity, decode_capacity)
        packet_y = packet_request_capacity * OUTPUT_TOKENS / used_gpus
        packet_points.append(
            {
                **{key: value for key, value in point.items() if key != "evidence_class"},
                "evidence_class": "MEASURED-EXTERNAL + SIM-DERIVED",
                "prefill_request_ps": packet_prefill_ps,
                "prefill_capacity_requests_per_second": _fraction_json(packet_prefill_capacity),
                "request_capacity_per_second": _fraction_json(packet_request_capacity),
                "capacity_limiter": (
                    "prefill" if packet_prefill_capacity <= decode_capacity else "decode"
                ),
                "y_tokens_per_second_per_gpu": _fraction_json(packet_y),
                "packet_term": {
                    "duration_ps": packet_cell["packet_service_ps"],
                    "evidence_class": "SIM-DERIVED",
                    "source": f"packet-cell:tp4-to-tp{decode_tp}",
                },
                "packet_to_ideal_capacity_step_quotient": _fraction_json(y_value / packet_y),
            }
        )

    ideal_frontier = weak_dominance_pareto(
        ideal_points,
        coordinate=_point_xy,
        identity=lambda point: str(point["candidate_key"]),
    )
    packet_frontier = weak_dominance_pareto(
        packet_points,
        coordinate=_point_xy,
        identity=lambda point: str(point["candidate_key"]),
    )
    sensitivity = _family_r_sensitivity(
        config=config,
        adjustment_table=adjustment_table,
        baseline_services={
            row["id"]: float(row["service_ms"])
            for row in service_rows
            if row["phase"] == "decode"
        },
    )
    estimator_module.RooflineProvider = original_roofline_provider
    external_db_module.ExternalQwen32BPassModel._context_attention = (
        original_context_attention
    )
    return {
        "worker": "local",
        "source": database.source.as_dict(),
        "services": service_rows,
        "raw_prefill_tp4_b1": {
            "service_ms": raw_prefill.service_ms,
            "service_ms_hex": raw_prefill.service_ms_hex,
            "service_ps": raw_prefill.service_ps,
            "evidence_class": raw_prefill.evidence_class,
            "source": raw_prefill.source,
        },
        "packet_cells": [packet_cells[width] for width in (2, 4, 8)],
        "candidate_grid": {
            "agg": [candidate_to_json(candidate) for candidate in agg_candidates],
            "disagg": [candidate_to_json(candidate) for candidate in disagg_candidates],
        },
        "ideal_points": ideal_points,
        "packet_points": packet_points,
        "ideal_frontier": list(ideal_frontier),
        "packet_frontier": list(packet_frontier),
        "external_adjustments": {
            "applied_ids": sorted(applied_adjustments),
            "context_attention_calls": context_calls,
            "declared_ids": sorted(
                str(row["id"]) for row in adjustment_table["adjustments"]
            ),
        },
        "family_r_sensitivity": sensitivity,
        "simllm_roofline_provider_calls": roofline_calls,
    }


def _sdk_database_model(tensor_parallel: int) -> tuple[Any, Any, Any]:
    from aiconfigurator_core.sdk import common
    from aiconfigurator_core.sdk.backends.factory import get_backend
    from aiconfigurator_core.sdk.config import ModelConfig
    from aiconfigurator_core.sdk.models import get_model
    from aiconfigurator_core.sdk.perf_database import get_database_view

    database = get_database_view(
        system="h200_sxm",
        backend="trtllm",
        version="1.3.0rc10",
        database_mode=common.DatabaseMode.SILICON,
        shared_layer=False,
        strict_provenance=False,
    )
    model_config = ModelConfig(
        tp_size=tensor_parallel,
        pp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.fp8_block,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
        comm_quant_mode=common.CommQuantMode.half,
        moe_tp_size=1,
        moe_ep_size=1,
        attention_dp_size=1,
        enable_encoder_dp=True,
        cp_size=1,
        cp_style="none",
        workload_distribution="power_law",
        nextn=0,
        overwrite_num_layers=0,
        sms=20,
        moe_backend=None,
        attention_backend="flashinfer",
        enable_wideep=False,
        enable_eplb=False,
        wideep_num_slots=None,
    )
    model = get_model("Qwen/Qwen3-32B-FP8", model_config, "trtllm")
    return database, model, get_backend("trtllm")


def _live_sdk_worker() -> dict[str, Any]:
    import importlib.metadata
    import inspect

    import aiconfigurator
    from aiconfigurator_core.sdk.config import RuntimeConfig

    config = _load_json(CONFIG_PATH)
    results = []
    cached = {}
    for phase in ("decode", "prefill"):
        for oracle in config["oracles"][phase]:
            tensor_parallel = int(oracle["tensor_parallel"])
            if tensor_parallel not in cached:
                cached[tensor_parallel] = _sdk_database_model(tensor_parallel)
            database, model, backend = cached[tensor_parallel]
            runtime = RuntimeConfig(
                batch_size=int(oracle["batch_size"]),
                beam_width=1,
                isl=int(oracle["isl"]),
                osl=int(oracle["osl"]),
                prefix=int(oracle["prefix"]),
                seq_imbalance_correction_scale=1.0,
                gen_seq_imbalance_correction_scale=1.0,
            )
            correction_key = (
                "decode_latency_correction_hex"
                if phase == "decode"
                else "prefill_latency_correction_hex"
            )
            total = float(
                backend.run_static_latency_only(
                    model,
                    database,
                    runtime,
                    oracle["mode"],
                    stride=int(oracle["stride"]),
                    latency_correction_scale=float.fromhex(config["composition"][correction_key]),
                )
            )
            service = total / (int(oracle["osl"]) - 1) if phase == "decode" else total
            results.append(
                {
                    "id": oracle["id"],
                    "phase": phase,
                    "service_ms_hex": service.hex(),
                    "total_ms_hex": total.hex(),
                }
            )
    site_packages = Path(inspect.getfile(aiconfigurator)).resolve().parent.parent
    adjustment_table = _adjustment_table()
    source_verification = []
    for adjustment in adjustment_table["adjustments"]:
        locations = {}
        for kind in ("source", "documentation"):
            declared = adjustment[kind]
            path = site_packages / declared["path"]
            lines = path.read_text(encoding="utf-8").splitlines()
            start_line = int(declared["start_line"])
            end_line = int(declared["end_line"])
            locations[kind] = {
                "path": declared["path"],
                "sha256": _sha256(path),
                "sha256_matches": _sha256(path) == declared["sha256"],
                "line_range_exists": 1 <= start_line <= end_line <= len(lines),
                "start_line": start_line,
                "end_line": end_line,
            }
        source_verification.append(
            {"adjustment_id": adjustment["id"], "locations": locations}
        )
    return {
        "worker": "live-sdk",
        "versions": {
            "aiconfigurator": importlib.metadata.version("aiconfigurator"),
            "aiconfigurator_core": importlib.metadata.version("aiconfigurator-core"),
        },
        "services": results,
        "external_adjustment_sources": source_verification,
    }


def _live_sdk_for_evaluation(
    *, external_python: Path, evaluation_root: Path
) -> dict[str, Any]:
    command = _worker_command(external_python, "live-sdk")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    _write_new(evaluation_root / "live-sdk.stdout.json", completed.stdout.encode())
    _write_new(evaluation_root / "live-sdk.stderr.txt", completed.stderr.encode())
    if completed.returncode:
        raise RuntimeError(
            "live SDK evaluation failed with status "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise TypeError("live SDK evaluation did not return a JSON object")
    return value


def _full_evaluation_worker(
    *,
    packet_root: Path,
    txt2bin: Path,
    htsim_rnic: Path,
    external_python: Path,
) -> dict[str, Any]:
    evaluation_root = packet_root.parent
    hashes_before = _tracked_hashes()
    local = _local_worker(
        packet_root=packet_root,
        txt2bin=txt2bin,
        htsim_rnic=htsim_rnic,
    )
    live_sdk = _live_sdk_for_evaluation(
        external_python=external_python,
        evaluation_root=evaluation_root,
    )
    hashes_after = _tracked_hashes()
    rows, families = _evaluate(
        local=local,
        live_sdk=live_sdk,
        hashes_before=hashes_before,
        hashes_after=hashes_after,
    )
    return {
        "schema": "simllm-matched-seam-scored-evaluation-v1",
        "worker": "evaluation",
        "source": local["source"],
        "services": local["services"],
        "live_sdk": {
            "versions": live_sdk["versions"],
            "services": live_sdk["services"],
        },
        "families": families,
        "rows": rows,
        "fatal_guards_without_fg6": {
            row["id"]: row["passed"] for row in rows if row["kind"] == "fatal"
        },
        "family_tallies_without_wall_time": _family_tallies(rows),
    }


def _worker_command(
    python: Path,
    worker: str,
    *,
    packet_root: Path | None = None,
    txt2bin: Path | None = None,
    htsim_rnic: Path | None = None,
    external_python: Path | None = None,
) -> list[str]:
    command = [os.fspath(python), os.fspath(Path(__file__).resolve()), "--worker", worker]
    if worker in {"local", "evaluation"}:
        assert packet_root is not None and txt2bin is not None and htsim_rnic is not None
        command.extend(
            (
                "--packet-root",
                os.fspath(packet_root),
                "--txt2bin",
                os.fspath(txt2bin),
                "--htsim-rnic",
                os.fspath(htsim_rnic),
            )
        )
    if worker == "evaluation":
        assert external_python is not None
        command.extend(("--external-python", os.fspath(external_python)))
    return command


def _run_worker(
    *,
    python: Path,
    worker: str,
    attempt: Path,
    repetition: int,
    txt2bin: Path | None = None,
    htsim_rnic: Path | None = None,
    external_python: Path | None = None,
) -> tuple[dict[str, Any], bytes]:
    packet_root = None
    if worker in {"local", "evaluation"}:
        packet_root = attempt / f"{worker}-run-{repetition}" / "packet"
        packet_root.parent.mkdir(parents=True, exist_ok=False)
    command = _worker_command(
        python,
        worker,
        packet_root=packet_root,
        txt2bin=txt2bin,
        htsim_rnic=htsim_rnic,
        external_python=external_python,
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    stem = f"{worker}-run-{repetition}"
    _write_new(attempt / f"{stem}.stdout.json", completed.stdout.encode())
    _write_new(attempt / f"{stem}.stderr.txt", completed.stderr.encode())
    if completed.returncode:
        raise RuntimeError(
            f"{worker} worker {repetition} failed with status {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise TypeError(f"{worker} worker did not return a JSON object")
    return value, completed.stdout.encode()


def _stamp_positive_evidence(stamp: dict[str, Any]) -> list[str]:
    return [
        str(term["evidence"])
        for term in stamp["terms"]
        if int(term["duration_ps"]) > 0
    ]


def _frontier_answer(
    frontier: list[dict[str, Any]], external_x: Fraction
) -> tuple[Fraction, dict[str, Any] | None]:
    eligible = [point for point in frontier if _point_xy(point)[0] >= external_x]
    if not eligible:
        return Fraction(), None
    point = max(eligible, key=lambda value: (_point_xy(value)[1], value["candidate_key"]))
    return _point_xy(point)[1], point


def _scored_row(
    family: str,
    row_id: str,
    passed: bool,
    *,
    expected: str,
    observed: str,
    units: str = "",
    evidence_class: str = "MEASURED-EXTERNAL",
    detail: str = "",
) -> dict[str, Any]:
    return {
        "family": family,
        "id": row_id,
        "kind": "scored",
        "passed": passed,
        "expected": expected,
        "observed": observed,
        "units": units,
        "evidence_class": evidence_class,
        "detail": detail,
    }


def _fatal_row(guard_id: str, passed: bool, detail: str) -> dict[str, Any]:
    return {
        "family": "FG",
        "id": guard_id,
        "kind": "fatal",
        "passed": passed,
        "expected": "true",
        "observed": str(passed).lower(),
        "units": "",
        "evidence_class": "GUARD",
        "detail": detail,
    }


def _unscored_row(
    row_id: str,
    observed: str,
    units: str,
    detail: str,
    *,
    family: str = "D",
    evidence_class: str = "MEASURED-EXTERNAL",
) -> dict[str, Any]:
    return {
        "family": family,
        "id": row_id,
        "kind": "unscored",
        "passed": None,
        "expected": "",
        "observed": observed,
        "units": units,
        "evidence_class": evidence_class,
        "detail": detail,
    }


def _family_tallies(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result = {}
    for family in ("S", "R", "F", "M", "W"):
        selected = [row for row in rows if row["kind"] == "scored" and row["family"] == family]
        result[family] = {
            "passed": sum(bool(row["passed"]) for row in selected),
            "denominator": len(selected),
        }
    return result


def _validate_scored_value_trace(trace: dict[str, Any]) -> list[str]:
    nodes = {str(node["id"]): node for node in trace["nodes"]}
    if len(nodes) != len(trace["nodes"]):
        return ["duplicate value-trace node ID"]
    failures = []
    reachable: set[str] = set()
    pending = list(trace["scored_roots"])
    while pending:
        node_id = str(pending.pop())
        if node_id in reachable:
            continue
        node = nodes.get(node_id)
        if node is None:
            failures.append(f"missing value-trace node {node_id}")
            continue
        reachable.add(node_id)
        pending.extend(str(dependency) for dependency in node["dependencies"])
    forbidden_kinds = {
        "roofline",
        "declared_efficiency",
        "fitted_constant",
        "fitted_curve",
    }
    for node_id in sorted(reachable):
        node = nodes[node_id]
        if str(node["origin"]).startswith("simllm") and node["kind"] in forbidden_kinds:
            failures.append(
                f"forbidden SimLLM-authored {node['kind']} reaches scored root through {node_id}"
            )
    return failures


def _scored_value_trace(
    local: dict[str, Any], adjustment_table: dict[str, Any]
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    roots: list[str] = []

    def add(
        node_id: str,
        *,
        kind: str,
        origin: str,
        value: object,
        dependencies: tuple[str, ...] = (),
    ) -> str:
        nodes.append(
            {
                "id": node_id,
                "kind": kind,
                "origin": origin,
                "value": value,
                "dependencies": list(dependencies),
            }
        )
        return node_id

    for adjustment in adjustment_table["adjustments"]:
        add(
            f"adjustment:{adjustment['id']}",
            kind="external_adjustment",
            origin="external-aiconfigurator",
            value=adjustment["value"],
        )

    service_nodes = {}
    for service in local["services"]:
        phase_dependencies = (
            (
                "adjustment:decode_latency_correction",
                "adjustment:memory_bandwidth_empirical_scale",
                "adjustment:memory_empirical_constant_latency",
            )
            if service["phase"] == "decode"
            else (
                "adjustment:prefill_latency_correction",
                "adjustment:memory_bandwidth_empirical_scale",
                "adjustment:memory_empirical_constant_latency",
                "adjustment:context_attention_extra_latency_correction",
            )
        )
        node_id = add(
            f"external-service:{service['id']}",
            kind="external_composed_service",
            origin="external-aiconfigurator",
            value=int(service["service_ps"]),
            dependencies=phase_dependencies,
        )
        service_nodes[str(service["id"])] = node_id
        roots.append(node_id)

    service_by_id = {str(row["id"]): row for row in local["services"]}
    prefill_service = service_by_id["prefill-tp4-b1"]
    prefill_factor = Fraction.from_float(
        _adjustment_float(adjustment_table, "prefill_rate_matching_degradation")
    )
    decode_factor = Fraction.from_float(
        _adjustment_float(adjustment_table, "decode_rate_matching_degradation")
    )
    packet_by_row = {int(point["row"]): point for point in local["packet_points"]}
    for point in local["ideal_points"]:
        row = int(point["row"])
        decode_id = (
            f"decode-tp{point['configuration']['decode_tp']}"
            f"-b{point['configuration']['decode_batch']}"
        )
        decode_service = service_by_id[decode_id]
        if int(point["decode_step_ps"]) != int(decode_service["service_ps"]):
            raise AssertionError("deployment decode step differs from traced external service")
        if int(point["prefill_request_ps"]) != int(prefill_service["service_ps"]):
            raise AssertionError("unpriced-network prefill differs from traced external service")
        decode_node = add(
            f"projection:decode-step:{row}",
            kind="deterministic_projection",
            origin="simllm-projection",
            value=int(point["decode_step_ps"]),
            dependencies=(service_nodes[decode_id],),
        )
        zero_network = add(
            f"projection:unpriced-network:{row}",
            kind="identity_zero_network_service",
            origin="simllm-projection",
            value=0,
        )
        prefill_node = add(
            f"projection:prefill-request:{row}",
            kind="deterministic_projection",
            origin="simllm-projection",
            value=int(point["prefill_request_ps"]),
            dependencies=(service_nodes["prefill-tp4-b1"], zero_network),
        )
        x_value = Fraction(PICOSECONDS_PER_SECOND, int(point["decode_step_ps"]))
        prefill_capacity = (
            Fraction(
                int(point["configuration"]["prefill_workers"])
                * PICOSECONDS_PER_SECOND,
                int(point["prefill_request_ps"]),
            )
            * prefill_factor
        )
        decode_capacity = (
            Fraction(
                int(point["configuration"]["decode_workers"])
                * int(point["configuration"]["decode_batch"])
                * PICOSECONDS_PER_SECOND,
                OUTPUT_TOKENS * int(point["decode_step_ps"]),
            )
            * decode_factor
        )
        request_capacity = min(prefill_capacity, decode_capacity)
        y_value = (
            request_capacity
            * OUTPUT_TOKENS
            / int(point["configuration"]["used_gpus"])
        )
        if _fraction(point["x_tokens_per_second_per_user"]) != x_value:
            raise AssertionError("traced x coordinate differs from the scored value")
        if _fraction(point["y_tokens_per_second_per_gpu"]) != y_value:
            raise AssertionError("traced y coordinate differs from the scored value")
        x_node = add(
            f"scored:ideal-x:{row}",
            kind="scored_coordinate",
            origin="simllm-projection",
            value=_fraction_json(x_value),
            dependencies=(decode_node,),
        )
        y_node = add(
            f"scored:ideal-y:{row}",
            kind="scored_coordinate",
            origin="simllm-projection",
            value=_fraction_json(y_value),
            dependencies=(
                decode_node,
                prefill_node,
                "adjustment:prefill_rate_matching_degradation",
                "adjustment:decode_rate_matching_degradation",
            ),
        )
        packet = packet_by_row[row]
        packet_term = add(
            f"simulation:packet-network:{row}",
            kind="packet_simulation",
            origin="simllm-simulation",
            value=int(packet["packet_term"]["duration_ps"]),
        )
        packet_y = _fraction(packet["y_tokens_per_second_per_gpu"])
        packet_node = add(
            f"scored:packet-y:{row}",
            kind="scored_coordinate",
            origin="simllm-projection",
            value=_fraction_json(packet_y),
            dependencies=(
                decode_node,
                prefill_node,
                packet_term,
                "adjustment:prefill_rate_matching_degradation",
                "adjustment:decode_rate_matching_degradation",
            ),
        )
        roots.extend((x_node, y_node, packet_node))
    trace = {
        "schema": "simllm-matched-seam-scored-value-trace-v1",
        "nodes": nodes,
        "scored_roots": sorted(set(roots)),
    }
    trace["validation_failures"] = _validate_scored_value_trace(trace)
    return trace


def _evaluate(
    *,
    local: dict[str, Any],
    live_sdk: dict[str, Any],
    hashes_before: dict[str, str],
    hashes_after: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = _load_json(CONFIG_PATH)
    adjustment_table = _adjustment_table()
    _validate_adjustment_config(config, adjustment_table)
    external_disagg = _external_curve(DISAGG_PATH, disaggregated=True)
    external_agg = _external_curve(AGG_PATH, disaggregated=False)
    services = {row["id"]: row for row in local["services"]}
    live_services = {row["id"]: row for row in live_sdk["services"]}
    rows: list[dict[str, Any]] = []

    all_stamps = [
        point[stamp]
        for point in local["ideal_points"]
        for stamp in ("decode_stamp", "prefill_stamp")
    ]
    positive_evidence = [
        evidence for stamp in all_stamps for evidence in _stamp_positive_evidence(stamp)
    ]
    value_trace = _scored_value_trace(local, adjustment_table)
    fg1a = (
        int(local["simllm_roofline_provider_calls"]) == 0
        and not value_trace["validation_failures"]
    )
    rows.append(
        _fatal_row(
            "FG-1a",
            fg1a,
            "the deploy RooflineProvider interception fired zero times and the complete scored-value dependency graph contains no reachable SimLLM roofline, efficiency, fitted constant or fitted curve",
        )
    )
    declared_adjustments = {
        str(adjustment["id"]) for adjustment in adjustment_table["adjustments"]
    }
    applied_adjustments = set(local["external_adjustments"]["applied_ids"])
    applied_adjustments.add("autoscale_ttft_correction")
    source_locations = [
        location
        for adjustment in live_sdk["external_adjustment_sources"]
        for location in adjustment["locations"].values()
    ]
    fg1b = (
        applied_adjustments == declared_adjustments
        and all(location["sha256_matches"] for location in source_locations)
        and all(location["line_range_exists"] for location in source_locations)
        and local["external_adjustments"]["context_attention_calls"] > 0
    )
    rows.append(
        _fatal_row(
            "FG-1b",
            fg1b,
            "applied external adjustments equal the tracked table exactly and every pinned source and documentation line verifies",
        )
    )
    sensitivity_ids = {
        str(row["adjustment_id"]) for row in local["family_r_sensitivity"]
    }
    fg1c = sensitivity_ids == declared_adjustments and all(
        len(row["rows"]) == len(external_disagg)
        for row in local["family_r_sensitivity"]
    )
    rows.append(
        _fatal_row(
            "FG-1c",
            fg1c,
            "every declared applied adjustment has a ten-row remove-one Family R sensitivity disclosure",
        )
    )
    source_slice = config["external_identity"]["slice_sha256"]
    fg2 = all(
        service["evidence_class"] == "MEASURED-EXTERNAL"
        and f"slice-sha256:{source_slice}" in service["source"]
        for service in local["services"]
    ) and all(evidence == "MEASURED-EXTERNAL" for evidence in positive_evidence)
    rows.append(
        _fatal_row(
            "FG-2",
            fg2,
            "service values and every positive ideal stamp retain MEASURED-EXTERNAL plus the frozen slice identity",
        )
    )
    rows.append(
        _fatal_row(
            "FG-3",
            hashes_before == hashes_after,
            "external frontier tables and parity publication artifacts are byte-identical before and after",
        )
    )
    chronology = {
        "expectations_before_service_freeze": _is_ancestor(
            EXPECTATIONS_COMMIT, SERVICE_FREEZE_COMMIT
        ),
        "service_freeze_before_binding": _is_ancestor(SERVICE_FREEZE_COMMIT, BINDING_COMMIT),
        "binding_before_run": _is_ancestor(BINDING_COMMIT),
        "corrected_expectations_before_run": _is_ancestor(
            CORRECTED_EXPECTATIONS_COMMIT
        ),
    }
    rows.append(
        _fatal_row("FG-4", all(chronology.values()), json.dumps(chronology, sort_keys=True))
    )
    corrected_prefill = services["prefill-tp4-b1"]["service_ms"]
    published_ttft = float(external_disagg[0]["ttft_ms"])
    rows.append(
        _fatal_row(
            "FG-5",
            corrected_prefill != published_ttft,
            "published disagg TTFT appears only in Family D and is not equated with isolated prefill service",
        )
    )
    s_rows = []
    for phase in ("decode", "prefill"):
        expected_field = "expected_step_ms_hex" if phase == "decode" else "expected_service_ms_hex"
        for oracle in config["oracles"][phase]:
            service = services[oracle["id"]]
            live = live_services[oracle["id"]]
            expected = oracle[expected_field]
            passed = service["service_ms_hex"] == live["service_ms_hex"] == expected
            row = _scored_row(
                "S",
                oracle["id"],
                passed,
                expected=expected,
                observed=service["service_ms_hex"],
                units="binary64 ms",
                detail=f"live-sdk={live['service_ms_hex']}",
            )
            rows.append(row)
            s_rows.append(row)

    r_rows = []
    for external in external_disagg:
        configuration = external["configuration"]
        service = services[
            f"decode-tp{configuration['decode_tp']}-b{configuration['decode_batch']}"
        ]
        quotient = Fraction.from_float(float(service["service_ms"])) / _decimal_fraction(
            external["tpot_ms"]
        )
        passed = Fraction(98, 100) <= quotient <= Fraction(102, 100)
        row = _scored_row(
            "R",
            f"R-{external['row']:02d}",
            passed,
            expected="[0.98, 1.02]",
            observed=f"{float(quotient):.12f}",
            units="quotient",
            detail=(
                f"tp{configuration['decode_tp']} batch {configuration['decode_batch']}: "
                f"ours {service['service_ms']:.15f} ms / published {external['tpot_ms']} ms"
            ),
        )
        rows.append(row)
        r_rows.append({**row, "row": external["row"], "quotient": _fraction_json(quotient)})

    for sensitivity in local["family_r_sensitivity"]:
        minimum = _fraction(sensitivity["minimum_quotient"])
        maximum = _fraction(sensitivity["maximum_quotient"])
        rows.append(
            _unscored_row(
                f"R-SENS-{sensitivity['adjustment_id']}",
                f"[{float(minimum):.12f}, {float(maximum):.12f}]",
                "Family R quotient range",
                (
                    f"remove {sensitivity['adjustment_id']} by replacing it with "
                    f"{sensitivity['removed_value']}; {sensitivity['method']}"
                ),
                family="R",
                evidence_class="DISCLOSURE",
            )
        )

    ideal_frontier = local["ideal_frontier"]
    packet_frontier = local["packet_frontier"]
    worked = ideal_frontier[3]
    worked_x = Fraction(PICOSECONDS_PER_SECOND, int(worked["decode_step_ps"]))
    worked_y = (
        _fraction(worked["request_capacity_per_second"])
        * OUTPUT_TOKENS
        / int(worked["configuration"]["used_gpus"])
    )
    f1 = _point_xy(worked) == (worked_x, worked_y)
    rows.append(
        _scored_row(
            "F",
            "F-1",
            f1,
            expected="x=1e12/decode_step_ps; y=request_capacity*500/used_gpus",
            observed=(
                f"x={float(worked_x):.9f}, y={float(worked_y):.9f}, "
                f"candidate={worked['candidate_id']}"
            ),
            units="tokens/s",
        )
    )
    f2_rows = []
    boundary_rows = []
    answer_counts: Counter[str] = Counter()
    for external in external_disagg:
        external_x = _decimal_fraction(external["x_tokens_per_second_per_user"])
        external_y = _decimal_fraction(external["y_tokens_per_second_per_gpu"])
        answer_y, answer = _frontier_answer(ideal_frontier, external_x)
        quotient = answer_y / external_y if answer_y else Fraction()
        passed = Fraction(75, 100) <= quotient <= Fraction(135, 100)
        answer_id = "none" if answer is None else str(answer["candidate_id"])
        answer_counts[answer_id] += 1
        row = _scored_row(
            "F",
            f"F-2-{external['row']:02d}",
            passed,
            expected="[0.75, 1.35]",
            observed=f"{float(quotient):.12f}",
            units="frontier/external quotient",
            detail=f"frontier answer {answer_id}",
        )
        rows.append(row)
        f2_rows.append(
            {
                **row,
                "row": external["row"],
                "quotient": _fraction_json(quotient),
                "frontier_answer": answer_id,
            }
        )
        local_point = next(
            point for point in local["ideal_points"] if point["row"] == external["row"]
        )
        exact_local_x = _fraction(local_point["x_tokens_per_second_per_user"])
        signed_difference = exact_local_x - external_x
        if -Fraction(1, 1000) <= signed_difference < 0:
            boundary_rows.append(
                {
                    "row": external["row"],
                    "exact_local_x": _fraction_json(exact_local_x),
                    "published_x": external["x_tokens_per_second_per_user"],
                    "published_rounding_unit": "0.001",
                    "signed_difference": _fraction_json(signed_difference),
                    "selected_frontier_answer": answer_id,
                }
            )
    for boundary in boundary_rows:
        difference = _fraction(boundary["signed_difference"])
        local_x = _fraction(boundary["exact_local_x"])
        rows.append(
            _unscored_row(
                f"F-BOUNDARY-{int(boundary['row']):02d}",
                f"{float(difference):+.15f}",
                "tokens/s/user",
                (
                    f"exact local x={float(local_x):.15f}; "
                    f"published x={boundary['published_x']}; "
                    f"selected={boundary['selected_frontier_answer']}"
                ),
                family="F",
                evidence_class="DISCLOSURE",
            )
        )
    coordinates = [_point_xy(point) for point in ideal_frontier]
    monotone = all(
        left_x < right_x and left_y > right_y
        for (left_x, left_y), (right_x, right_y) in pairwise(coordinates)
    )
    rows.append(
        _scored_row(
            "F",
            "F-3",
            monotone and len(ideal_frontier) >= 8,
            expected="monotone with at least 8 points",
            observed=f"monotone={monotone}, points={len(ideal_frontier)}",
        )
    )
    max_answers = max(answer_counts.values(), default=0)
    rows.append(
        _scored_row(
            "F",
            "F-4",
            max_answers <= 3,
            expected="maximum endpoint use <= 3",
            observed=str(max_answers),
            units="external rows per frontier point",
            detail=json.dumps(dict(sorted(answer_counts.items())), sort_keys=True),
        )
    )

    m_rows = []
    for ideal, packet in zip(local["ideal_points"], local["packet_points"], strict=True):
        quotient = _fraction(packet["packet_to_ideal_capacity_step_quotient"])
        m_rows.append(
            {
                "row": ideal["row"],
                "candidate_id": ideal["candidate_id"],
                "decode_tp": ideal["configuration"]["decode_tp"],
                "ideal_y": ideal["y_tokens_per_second_per_gpu"],
                "packet_y": packet["y_tokens_per_second_per_gpu"],
                "quotient": _fraction_json(quotient),
            }
        )
    minimum_m = min((_fraction(row["quotient"]) for row in m_rows), default=Fraction())
    maximum_m = max((_fraction(row["quotient"]) for row in m_rows), default=Fraction())
    rows.append(
        _scored_row(
            "M",
            "M-1",
            minimum_m >= 1,
            expected=">= 1.000000 for every candidate",
            observed=f"minimum={float(minimum_m):.12f}",
            units="packet-priced/unpriced-network capacity-step quotient",
            evidence_class="MEASURED-EXTERNAL + SIM-DERIVED",
            detail="the unpriced-network arm charges exactly zero network service",
        )
    )
    rows.append(
        _scored_row(
            "M",
            "M-2",
            maximum_m >= Fraction(102, 100),
            expected=">= 1.02 for at least one candidate",
            observed=f"maximum={float(maximum_m):.12f}",
            units="packet-priced/unpriced-network capacity-step quotient",
            evidence_class="MEASURED-EXTERNAL + SIM-DERIVED",
            detail="the unpriced-network arm charges exactly zero network service",
        )
    )

    raw_prefill = float(local["raw_prefill_tp4_b1"]["service_ms"])
    corrected = float(corrected_prefill)
    autoscale = _adjustment_float(adjustment_table, "autoscale_ttft_correction")
    autoscaled = corrected * autoscale
    residual_from_pass = published_ttft - raw_prefill
    service_correction = corrected - raw_prefill
    autoscale_correction = autoscaled - corrected
    table_residual = published_ttft - autoscaled
    d_rows = [
        _unscored_row("D-1", f"{raw_prefill:.15f}", "ms", "imported TP4 batch-1 prefill pass"),
        _unscored_row(
            "D-2", f"{residual_from_pass:.15f}", "ms", "published TTFT minus raw prefill pass"
        ),
        _unscored_row(
            "D-3", f"{service_correction:.15f}", "ms", "1.1 prefill service correction contribution"
        ),
        _unscored_row(
            "D-4", f"{autoscale_correction:.15f}", "ms", "1.8 autoscale correction contribution"
        ),
        _unscored_row(
            "D-5",
            f"{table_residual:.15f}",
            "ms",
            "three-decimal publication reconciliation residual",
        ),
    ]
    rows.extend(d_rows)
    return rows, {
        "S": {"rows": s_rows},
        "R": {
            "rows": r_rows,
            "remove_one_sensitivity": local["family_r_sensitivity"],
        },
        "F": {
            "worked_axis_cell": worked,
            "bracket_rows": f2_rows,
            "answer_counts": dict(sorted(answer_counts.items())),
            "ideal_points": local["ideal_points"],
            "packet_points": local["packet_points"],
            "ideal_frontier": ideal_frontier,
            "packet_frontier": packet_frontier,
            "boundary_proximity_rows": boundary_rows,
        },
        "M": {
            "rows": m_rows,
            "minimum_quotient": _fraction_json(minimum_m),
            "maximum_packet_priced_to_unpriced_network_quotient": _fraction_json(
                maximum_m
            ),
            "packet_cells": local["packet_cells"],
            "unpriced_network_service_ps": 0,
            "third_loggopsim_priced_arm": {
                "ran": False,
                "scope": "not run; no isolated receiver-side serialization claim is made",
            },
        },
        "D": {
            "published_ttft_ms": published_ttft,
            "raw_prefill_pass_ms": raw_prefill,
            "corrected_prefill_service_ms": corrected,
            "autoscaled_prefill_ms": autoscaled,
            "residual_from_raw_pass_ms": residual_from_pass,
            "attribution_ms": {
                "prefill_service_correction": service_correction,
                "autoscale_ttft_correction": autoscale_correction,
                "publication_reconciliation": table_residual,
            },
            "status": "fully attributed within the tracked table's three-decimal precision",
            "rows": d_rows,
        },
        "external_curves": {"agg": external_agg, "disagg": external_disagg},
        "candidate_grid": local["candidate_grid"],
        "chronology": chronology,
        "immutability": {"before": hashes_before, "after": hashes_after},
        "external_adjustments": {
            "table": adjustment_table,
            "applied_ids": sorted(applied_adjustments),
            "source_verification": live_sdk["external_adjustment_sources"],
        },
        "scored_value_trace": value_trace,
    }


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    import io

    output = io.StringIO(newline="")
    fieldnames = (
        "evidence_class",
        "family",
        "id",
        "kind",
        "passed",
        "expected",
        "observed",
        "units",
        "detail",
    )
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _coordinator(
    *,
    bulk_root: Path,
    external_venv: Path,
    txt2bin: Path,
    htsim_rnic: Path,
    write_tracked: bool,
) -> dict[str, Any]:
    from examples.matched_seam_frontier_v1.plot_study import render

    for name, path in (
        (EXTERNAL_VENV_ENV, external_venv),
        (TXT2BIN_ENV, txt2bin),
        (HTSIM_ENV, htsim_rnic),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    external_python = next(
        (
            path
            for path in (
                external_venv / "bin/python",
                external_venv / "Scripts/python.exe",
            )
            if path.is_file()
        ),
        None,
    )
    if external_python is None:
        raise FileNotFoundError(f"{EXTERNAL_VENV_ENV} has no Python interpreter")
    attempt, attempt_number = _new_attempt(bulk_root)
    started = time.monotonic()
    evaluation_runs_with_bytes = [
        _run_worker(
            python=Path(sys.executable),
            worker="evaluation",
            attempt=attempt,
            repetition=repetition,
            txt2bin=txt2bin,
            htsim_rnic=htsim_rnic,
            external_python=external_python,
        )
        for repetition in (1, 2)
    ]
    evaluations = [value for value, _ in evaluation_runs_with_bytes]
    evaluation_bytes = [payload for _, payload in evaluation_runs_with_bytes]
    rows = list(evaluations[0]["rows"])
    families = evaluations[0]["families"]
    deterministic = evaluation_bytes[0] == evaluation_bytes[1]
    evaluation_hashes = [hashlib.sha256(payload).hexdigest() for payload in evaluation_bytes]
    rows.append(
        _fatal_row(
            "FG-6",
            deterministic,
            (
                "complete scored evaluation JSON is byte-identical across two fresh "
                "processes; elapsed_seconds and W-1 are excluded by name"
            ),
        )
    )
    failed_guards = [row["id"] for row in rows if row["kind"] == "fatal" and not row["passed"]]
    record = {
        "schema": SCHEMA,
        "study": "matched_seam_frontier_v1",
        "run_state": "void" if failed_guards else "nonvoid",
        "voiding_guards": failed_guards,
        "attempt": f"attempt-{attempt_number:04d}",
        "bulk_evidence": f"${{{BULK_ROOT_ENV}}}/attempt-{attempt_number:04d}",
        "run_commit": _git_output("rev-parse", "HEAD"),
        "freeze_commits": {
            "first_expectations": EXPECTATIONS_COMMIT,
            "corrected_expectations": CORRECTED_EXPECTATIONS_COMMIT,
            "service_oracles": SERVICE_FREEZE_COMMIT,
            "binding": BINDING_COMMIT,
        },
        "config_sha256": _sha256(CONFIG_PATH),
        "expectations_sha256": {
            "first_void_freeze": _sha256(EXPECTATIONS_PATH),
            "corrected_freeze": _sha256(EXPECTATIONS_V2_PATH),
        },
        "external_adjustments_sha256": _sha256(ADJUSTMENTS_PATH),
        "source": evaluations[0]["source"],
        "native_tools": {
            "htsim_rnic": {"filename": htsim_rnic.name, "sha256": _sha256(htsim_rnic)},
            "txt2bin": {"filename": txt2bin.name, "sha256": _sha256(txt2bin)},
        },
        "machine": {
            "architecture": platform.machine(),
            "cpu": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
            "system": platform.system(),
        },
        "fatal_guards": {row["id"]: row["passed"] for row in rows if row["kind"] == "fatal"},
        "family_tallies": _family_tallies(rows),
        "families": families,
        "rows": rows,
        "determinism": {
            "comparison": "byte-for-byte complete scored evaluation JSON",
            "fresh_processes": 2,
            "evaluation_sha256": evaluation_hashes,
            "excluded_by_name": ["elapsed_seconds", "W-1"],
            "equal": deterministic,
        },
        "first_published_run": {
            "state": "void",
            "voiding_guard": "FG-1",
            "attempt": "attempt-0002",
            "run_commit": "3e752d58c9e874f234110af69851384ea02873cd",
            "reason": "the first freeze forbade roofline and fitted terms throughout a composition whose external resolver and empirical adjustments require them",
        },
        "reporting_rule": "fatal guards, S, R, F, M, D and W are separate evidence classes and are never summed",
        "figure": {
            "pdf": PDF_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "png": PNG_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "caption": (
                "The external aggregate and disaggregated curves and the SimLLM "
                "unpriced-network curve share MEASURED-EXTERNAL operation timing. "
                "The packet-priced curve adds SIM-DERIVED network service. Its gap "
                "prices the complete network against zero network service and does not "
                "isolate receiver-side serialization."
            ),
        },
    }
    render(
        record,
        pdf_path=attempt / "matched-seam-frontier.pdf",
        png_path=attempt / "matched-seam-frontier.png",
    )
    elapsed_seconds = time.monotonic() - started
    wall_row = _scored_row(
        "W",
        "W-1",
        elapsed_seconds <= WALL_CEILING_SECONDS,
        expected=f"<= {WALL_CEILING_SECONDS:.0f}",
        observed=f"{elapsed_seconds:.6f}",
        units="seconds",
        evidence_class="WALL",
        detail="two complete fresh-process scored evaluations and the figure",
    )
    rows.append(wall_row)
    record["elapsed_seconds"] = elapsed_seconds
    record["family_tallies"] = _family_tallies(rows)
    record["rows"] = rows
    csv_payload = _csv_bytes(rows)
    _write_new(
        attempt / "record.json", (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    )
    _write_new(attempt / "results.csv", csv_payload)
    if write_tracked:
        _write_json(RESULT_PATH, record)
        CSV_PATH.write_bytes(csv_payload)
        PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
        PDF_PATH.write_bytes((attempt / "matched-seam-frontier.pdf").read_bytes())
        PNG_PATH.write_bytes((attempt / "matched-seam-frontier.png").read_bytes())
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("local", "live-sdk", "evaluation"))
    parser.add_argument("--packet-root", type=Path)
    parser.add_argument("--txt2bin", type=Path)
    parser.add_argument("--htsim-rnic", type=Path)
    parser.add_argument("--external-python", type=Path)
    parser.add_argument("--bulk-root", type=Path)
    parser.add_argument("--external-venv", type=Path)
    parser.add_argument("--write-tracked", action="store_true")
    args = parser.parse_args()
    if args.worker == "local":
        if args.packet_root is None or args.txt2bin is None or args.htsim_rnic is None:
            parser.error("local worker requires --packet-root, --txt2bin and --htsim-rnic")
        result = _local_worker(
            packet_root=args.packet_root,
            txt2bin=args.txt2bin,
            htsim_rnic=args.htsim_rnic,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.worker == "live-sdk":
        print(json.dumps(_live_sdk_worker(), sort_keys=True))
        return
    if args.worker == "evaluation":
        if (
            args.packet_root is None
            or args.txt2bin is None
            or args.htsim_rnic is None
            or args.external_python is None
        ):
            parser.error(
                "evaluation worker requires --packet-root, --txt2bin, "
                "--htsim-rnic and --external-python"
            )
        result = _full_evaluation_worker(
            packet_root=args.packet_root,
            txt2bin=args.txt2bin,
            htsim_rnic=args.htsim_rnic,
            external_python=args.external_python,
        )
        print(json.dumps(result, sort_keys=True))
        return
    bulk_root = args.bulk_root or (
        Path(os.environ[BULK_ROOT_ENV]) if BULK_ROOT_ENV in os.environ else None
    )
    external_venv = args.external_venv or (
        Path(os.environ[EXTERNAL_VENV_ENV]) if EXTERNAL_VENV_ENV in os.environ else None
    )
    txt2bin = args.txt2bin or (Path(os.environ[TXT2BIN_ENV]) if TXT2BIN_ENV in os.environ else None)
    htsim_rnic = args.htsim_rnic or (
        Path(os.environ[HTSIM_ENV]) if HTSIM_ENV in os.environ else None
    )
    missing = [
        name
        for name, value in (
            ("--bulk-root or " + BULK_ROOT_ENV, bulk_root),
            ("--external-venv or " + EXTERNAL_VENV_ENV, external_venv),
            ("--txt2bin or " + TXT2BIN_ENV, txt2bin),
            ("--htsim-rnic or " + HTSIM_ENV, htsim_rnic),
        )
        if value is None
    ]
    if missing:
        parser.error("missing " + ", ".join(missing))
    assert bulk_root is not None
    assert external_venv is not None
    assert txt2bin is not None
    assert htsim_rnic is not None
    result = _coordinator(
        bulk_root=bulk_root,
        external_venv=external_venv,
        txt2bin=txt2bin,
        htsim_rnic=htsim_rnic,
        write_tracked=args.write_tracked,
    )
    print(json.dumps(result["family_tallies"], sort_keys=True))


if __name__ == "__main__":
    main()
