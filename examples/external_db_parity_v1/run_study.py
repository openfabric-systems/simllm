"""Run the frozen external operation-database parity study.

The coordinator imports the pinned source into a new append-only attempt,
executes the local resolver and live external SDK twice in fresh processes,
and evaluates fatal guards plus the I1, I2, P1 and W families. Bulk evidence
goes to a caller-supplied root. ``--write-tracked`` additionally writes the
portable compact record and CSV in this study directory and refuses to
overwrite either file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
QUERY_CONFIG = STUDY / "query_points.json"
EXPECTATIONS = STUDY / "expectations.md"
TRACKED_ARTIFACT = (
    ROOT
    / "offline/calibration/external-databases"
    / "85e72f990f00ea457de522d0b773e678f5e067740689912df5646f6296273284"
)
EXTERNAL_VENV_ENV = "SIMLLM_EXTERNAL_AIC_VENV"
BULK_ROOT_ENV = "SIMLLM_P3X_T1_BULK_ROOT"
SCHEMA = "simllm-external-db-parity-record-v1"

EXPECTED_IDENTITY = {
    "aiconfigurator": "0.11.0",
    "aiconfigurator_core": "0.11.0",
    "system": "h200_sxm",
    "backend": "trtllm",
    "database_version": "1.3.0rc10",
    "database_mode": "SILICON",
    "shared_layer": False,
    "surface": "python",
    "slice_sha256": "85e72f990f00ea457de522d0b773e678f5e067740689912df5646f6296273284",
    "closure_sha256": "d559d6694f30ad269ecbf697e0193c7d95e4aba1cfb929836d381a46b675876f",
    "system_sha256": "142584d6bddd98207fd04e844029b0ba5d6fcd4c6f41016c5e77f0cbe4053614",
    "model_sha256": "e546dacd2c772660270233f5579e9ab923cc2a7ec5ed3c58c27c2bc62cbf5169",
}
FREEZE_COMMITS = {
    "expectations": "44959ef",
    "resolver_queries": "afe7ee6",
    "mutation_guard": "f7ec05a",
}
P1_ORACLES = (
    {
        "id": "P1-01",
        "mode": "static_ctx",
        "batch_size": 1,
        "isl": 3500,
        "osl": 1,
        "stride": 32,
        "expected_hex": "0x1.8d2164d537eb3p+6",
    },
    {
        "id": "P1-02",
        "mode": "static_gen",
        "batch_size": 64,
        "isl": 4000,
        "osl": 2,
        "stride": 32,
        "expected_hex": "0x1.6344a3614677ep+3",
    },
    {
        "id": "P1-03",
        "mode": "static_ctx",
        "batch_size": 2,
        "isl": 1750,
        "osl": 1,
        "stride": 32,
        "expected_hex": "0x1.7b53c1bc0e6d2p+6",
    },
    {
        "id": "P1-04",
        "mode": "static_gen",
        "batch_size": 32,
        "isl": 8000,
        "osl": 2,
        "stride": 32,
        "expected_hex": "0x1.66a63c02685c1p+3",
    },
)
I1_ROWS = (
    (
        "I1-01",
        "gemm",
        ("bfloat16", 4353, 65536, 51200),
        "0x1.4d4fa15555555p+5",
    ),
    ("I1-02", "gemm", ("fp8", 1024, 32, 32), "0x1.9da0d77777778p-7"),
    (
        "I1-03",
        "generation_attention",
        ("fp8", 8, 64, 0, 96, 64, 2),
        "0x1.03b2840000000p-7",
    ),
    (
        "I1-04",
        "moe",
        ("bfloat16", "balanced", 8, 128, 4096, 1536, 4, 4, 256),
        "0x1.b6d43d5555555p-4",
    ),
    (
        "I1-05",
        "custom_allreduce",
        ("half", 8, "AUTO", 536870912),
        "0x1.007923d70a3d7p+2",
    ),
    (
        "I1-06",
        "gdn",
        (
            "fused_sigmoid_gating_delta_rule_update",
            "generation",
            2048,
            16,
            128,
            16,
            128,
            4,
            512,
        ),
        "0x1.5b7571999999ap-2",
    ),
    (
        "I1-07",
        "compute_scale",
        ("fp8", 32768, 51200),
        "0x1.356a966666667p+1",
    ),
)
I1_COUNTS = {
    "compute_scale": 1628,
    "context_attention": 50574,
    "context_mla": 1760,
    "context_mla_module": 3873,
    "custom_allreduce": 69,
    "encoder_attention": 6314,
    "gdn": 1862,
    "gemm": 101010,
    "generation_attention": 24438,
    "generation_dsa_module": 2944,
    "generation_mla": 2896,
    "generation_mla_module": 5888,
    "mamba2": 469,
    "mla_bmm": 848,
    "moe": 74358,
    "scale_matrix": 1628,
    "wideep_moe": 4158,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode()


def _write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _query_config() -> dict[str, Any]:
    return json.loads(QUERY_CONFIG.read_text(encoding="utf-8"))


def _local_worker(artifact: Path) -> dict[str, Any]:
    sys.path.insert(0, os.fspath(ROOT))
    from simllm.calibration.external_db import (
        EXTERNAL_EVIDENCE_CLASS,
        ExternalCompositeError,
        ExternalCompositionLedger,
        ExternalDatabaseGapError,
        ExternalOperationDatabase,
        ExternalQwen32BPassModel,
    )

    database = ExternalOperationDatabase.load(artifact)
    config = _query_config()
    queries: dict[str, dict[str, Any]] = {}
    served = []
    for query in config["queries"]:
        value = database.query_operation(query["operation"], query["args"])
        served.append(value)
        queries[query["id"]] = {
            "hex": value.hex,
            "operation": value.operation,
            "rule": value.rule,
        }

    passes: dict[str, dict[str, Any]] = {}
    pass_model = ExternalQwen32BPassModel(database)
    for oracle in P1_ORACLES:
        if oracle["mode"] == "static_ctx":
            result = pass_model.run_context(
                batch_size=oracle["batch_size"],
                isl=oracle["isl"],
            )
        else:
            result = pass_model.run_generation(
                batch_size=oracle["batch_size"],
                isl=oracle["isl"],
                osl=oracle["osl"],
                stride=oracle["stride"],
            )
        served.extend(result.operations)
        served.append(result.total)
        passes[oracle["id"]] = {
            "hex": result.total.hex,
            "terms": {entry.operation: entry.hex for entry in result.operations},
        }

    i1 = {
        row_id: database.raw_latency(table, key).hex()
        for row_id, table, key, _expected in I1_ROWS
    }
    mutation = config["load_time_mutation_guard"]
    mutation_cells = [
        cell
        for cell in database.load_time_mutations("gemm")
        if tuple(cell["key"])
        == (
            mutation["args"]["quant_mode"],
            mutation["args"]["m"],
            mutation["args"]["n"],
            mutation["args"]["k"],
        )
    ]
    mutation_served = database.query_operation(mutation["operation"], mutation["args"])
    served.append(mutation_served)

    overlap_rejected = False
    ledger = ExternalCompositionLedger(database)
    ledger.claim("attn_score+kv_read", composite=True)
    try:
        ledger.claim("attn_score")
    except ExternalCompositeError:
        overlap_rejected = True
    gap_rejected = False
    try:
        database.require_mapping("gdn_gated_norm")
    except ExternalDatabaseGapError as error:
        gap_rejected = "gdn_gated_norm" in str(error)
    undeclared_rejected = False
    try:
        database.require_mapping("not_in_the_mapping")
    except ExternalDatabaseGapError as error:
        undeclared_rejected = "absent" in str(error)

    evidence_ok = all(
        value.evidence_class == EXTERNAL_EVIDENCE_CLASS
        and value.source == database.source
        for value in served
    )
    return {
        "worker": "local",
        "source": database.source.as_dict(),
        "payload_sha256": database.payload_sha256,
        "versions": sorted(database.versions()),
        "row_count": database.row_count,
        "row_counts": database.row_counts,
        "i1": i1,
        "i2": queries,
        "p1": passes,
        "mutation": {
            "cells": mutation_cells,
            "query_hex": mutation_served.hex,
            "gemm_mutation_count": len(database.load_time_mutations("gemm")),
            "generation_attention_mutation_count": len(
                database.load_time_mutations("generation_attention")
            ),
        },
        "evidence": {
            "class": EXTERNAL_EVIDENCE_CLASS,
            "served_values": len(served),
            "all_match": evidence_ok,
        },
        "mapping": {
            "sha256": _sha256(artifact / "family-mapping.json"),
            "statuses": {
                status: sum(
                    rule.get("status") == status
                    for rule in database.family_mapping["rules"]
                )
                for status in ("exact", "composite", "gap")
            },
            "overlap_rejected": overlap_rejected,
            "gap_rejected": gap_rejected,
            "undeclared_rejected": undeclared_rejected,
        },
    }


def _external_package_root() -> Path:
    spec = importlib.util.find_spec("aiconfigurator_core")
    if spec is None or spec.origin is None:
        raise RuntimeError("aiconfigurator_core is unavailable in the external worker")
    return Path(spec.origin).resolve().parent


def _hash_manifest(root: Path, relative_paths: list[str]) -> bytes:
    return "".join(
        f"{_sha256(root / relative)}  {relative}\n"
        for relative in sorted(relative_paths)
    ).encode()


def _external_identity(package_root: Path) -> dict[str, Any]:
    data_prefix = Path("systems") / "data" / "h200_sxm"
    data_root = package_root / data_prefix
    suffix = "/trtllm/1.3.0rc10/"
    slice_paths = sorted(
        path.relative_to(data_root).as_posix()
        for path in data_root.rglob("*")
        if path.is_file() and suffix in f"/{path.relative_to(data_root).as_posix()}"
    )
    slice_hash = hashlib.sha256(_hash_manifest(data_root, slice_paths)).hexdigest()
    closure_paths = [(data_prefix / path).as_posix() for path in slice_paths]
    closure_paths.append("systems/h200_sxm.yaml")
    return {
        "aiconfigurator": importlib.metadata.version("aiconfigurator"),
        "aiconfigurator_core": importlib.metadata.version("aiconfigurator-core"),
        "slice_file_count": len(slice_paths),
        "slice_sha256": slice_hash,
        "closure_sha256": hashlib.sha256(
            _hash_manifest(package_root, closure_paths)
        ).hexdigest(),
        "system_sha256": _sha256(package_root / "systems/h200_sxm.yaml"),
        "model_sha256": _sha256(
            package_root / "model_configs/Qwen--Qwen3-32B-FP8_config.json"
        ),
    }


def _external_database() -> tuple[Any, Any, Any]:
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
        tp_size=4,
        pp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
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


def _external_query(database: Any, operation: str, arguments: dict[str, Any]) -> float:
    from aiconfigurator_core.sdk import common

    args = dict(arguments)
    if operation == "gemm":
        args["quant_mode"] = common.GEMMQuantMode[args["quant_mode"]]
        return float(database.query_gemm(**args))
    if operation == "compute_scale":
        args["quant_mode"] = common.GEMMQuantMode[args["quant_mode"]]
        return float(database.query_compute_scale(**args))
    if operation == "scale_matrix":
        args["quant_mode"] = common.GEMMQuantMode[args["quant_mode"]]
        return float(database.query_scale_matrix(**args))
    if operation == "context_attention":
        args["kvcache_quant_mode"] = common.KVCacheQuantMode[
            args.pop("kv_quant_mode")
        ]
        args["fmha_quant_mode"] = common.FMHAQuantMode[args["fmha_quant_mode"]]
        return float(database.query_context_attention(**args))
    if operation == "generation_attention":
        args["kvcache_quant_mode"] = common.KVCacheQuantMode[
            args.pop("kv_quant_mode")
        ]
        return float(database.query_generation_attention(**args))
    if operation == "moe":
        args["quant_mode"] = common.MoEQuantMode[args["quant_mode"]]
        return float(database.query_moe(**args))
    if operation == "custom_allreduce":
        args["quant_mode"] = common.CommQuantMode[args["quant_mode"]]
        return float(database.query_custom_allreduce(**args))
    if operation == "gdn":
        return float(database.query_gdn(**args))
    raise ValueError(f"unsupported external query operation {operation!r}")


def _external_worker() -> dict[str, Any]:
    from aiconfigurator_core.sdk.config import RuntimeConfig

    database, model, backend = _external_database()
    config = _query_config()
    queries = {
        query["id"]: {
            "hex": _external_query(database, query["operation"], query["args"]).hex()
        }
        for query in config["queries"]
    }
    passes: dict[str, dict[str, Any]] = {}
    for oracle in P1_ORACLES:
        runtime = RuntimeConfig(
            batch_size=oracle["batch_size"],
            beam_width=1,
            isl=oracle["isl"],
            osl=oracle["osl"],
            prefix=0,
            seq_imbalance_correction_scale=1.0,
            gen_seq_imbalance_correction_scale=1.0,
        )
        context, _, generation, _, _, _ = backend._run_static_breakdown(
            model,
            database,
            runtime,
            oracle["mode"],
            oracle["stride"],
            1.0,
            include_energy=False,
        )
        total = backend.run_static_latency_only(
            model,
            database,
            runtime,
            oracle["mode"],
            stride=oracle["stride"],
            latency_correction_scale=1.0,
        )
        passes[oracle["id"]] = {
            "hex": float(total).hex(),
            "terms": {
                name: float(value).hex()
                for name, value in [*context.items(), *generation.items()]
            },
        }
    mutation = config["load_time_mutation_guard"]
    return {
        "worker": "external",
        "identity": _external_identity(_external_package_root()),
        "i2": queries,
        "p1": passes,
        "mutation_hex": _external_query(
            database,
            mutation["operation"],
            mutation["args"],
        ).hex(),
    }


def _worker_command(python: Path, worker: str, artifact: Path) -> list[str]:
    command = [os.fspath(python), os.fspath(Path(__file__).resolve()), "--worker", worker]
    if worker == "local":
        command.extend(("--artifact", os.fspath(artifact)))
    return command


def _run_worker(
    *,
    python: Path,
    worker: str,
    artifact: Path,
    attempt: Path,
    repetition: int,
) -> dict[str, Any]:
    completed = subprocess.run(
        _worker_command(python, worker, artifact),
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": os.fspath(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    stem = f"{worker}-run-{repetition}"
    _write_new(attempt / f"{stem}.stderr.txt", completed.stderr.encode())
    _write_new(attempt / f"{stem}.stdout.json", completed.stdout.encode())
    if completed.returncode != 0:
        raise RuntimeError(
            f"{worker} worker repetition {repetition} failed with status "
            f"{completed.returncode}; see {stem}.stderr.txt"
        )
    return json.loads(completed.stdout)


def _new_attempt(root: Path) -> tuple[Path, int]:
    root.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in root.iterdir():
        match = re.fullmatch(r"attempt-(\d{4})", path.name)
        if match:
            existing.append(int(match.group(1)))
    number = max(existing, default=0) + 1
    attempt = root / f"attempt-{number:04d}"
    attempt.mkdir()
    return attempt, number


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _is_ancestor(commit: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=False,
    )
    return completed.returncode == 0


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _ulp_distance(left_hex: str, right_hex: str) -> int:
    left = struct.unpack(">Q", struct.pack(">d", float.fromhex(left_hex)))[0]
    right = struct.unpack(">Q", struct.pack(">d", float.fromhex(right_hex)))[0]
    return abs(left - right)


def _scored_row(
    family: str,
    row_id: str,
    passed: bool,
    *,
    expected_hex: str = "",
    local_hex: str = "",
    external_hex: str = "",
    detail: str = "",
) -> dict[str, Any]:
    ulps = ""
    if local_hex and external_hex:
        ulps = _ulp_distance(local_hex, external_hex)
    return {
        "family": family,
        "id": row_id,
        "kind": "scored",
        "passed": passed,
        "expected_hex": expected_hex,
        "local_hex": local_hex,
        "external_hex": external_hex,
        "ulp_distance": ulps,
        "detail": detail,
    }


def _fatal_row(guard_id: str, passed: bool, detail: str) -> dict[str, Any]:
    return {
        "family": "FG",
        "id": guard_id,
        "kind": "fatal",
        "passed": passed,
        "expected_hex": "",
        "local_hex": "",
        "external_hex": "",
        "ulp_distance": "",
        "detail": detail,
    }


def _evaluate(
    *,
    local: dict[str, Any],
    external: dict[str, Any],
    local_deterministic: bool,
    external_deterministic: bool,
    imported_artifact: Path,
    attempt_number: int,
    elapsed_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = _query_config()
    rows: list[dict[str, Any]] = []

    for row_id, _table, _key, expected_hex in I1_ROWS:
        actual = local["i1"][row_id]
        rows.append(
            _scored_row(
                "I1",
                row_id,
                actual == expected_hex,
                expected_hex=expected_hex,
                local_hex=actual,
                detail="representative raw binary64 row",
            )
        )
    for table, expected_count in I1_COUNTS.items():
        actual_count = local["row_counts"].get(table)
        rows.append(
            _scored_row(
                "I1",
                f"I1-count-{table}",
                actual_count == expected_count,
                detail=f"expected {expected_count} rows, found {actual_count}",
            )
        )
    rows.append(
        _scored_row(
            "I1",
            "I1-count-total",
            local["row_count"] == 284717,
            detail=f"expected 284717 rows, found {local['row_count']}",
        )
    )

    for query in config["queries"]:
        row_id = query["id"]
        local_hex = local["i2"][row_id]["hex"]
        external_hex = external["i2"][row_id]["hex"]
        rows.append(
            _scored_row(
                "I2",
                row_id,
                local_hex == external_hex,
                local_hex=local_hex,
                external_hex=external_hex,
                detail=query["rule"],
            )
        )

    ulp_findings = []
    for oracle in P1_ORACLES:
        row_id = oracle["id"]
        local_value = local["p1"][row_id]
        external_value = external["p1"][row_id]
        expected_hex = oracle["expected_hex"]
        passed = local_value["hex"] == external_value["hex"] == expected_hex
        diverging_terms = []
        for name in sorted(set(local_value["terms"]) | set(external_value["terms"])):
            local_hex = local_value["terms"].get(name, "")
            external_hex = external_value["terms"].get(name, "")
            if local_hex != external_hex:
                diverging_terms.append(
                    {
                        "term": name,
                        "local_hex": local_hex,
                        "external_hex": external_hex,
                        "ulp_distance": (
                            _ulp_distance(local_hex, external_hex)
                            if local_hex and external_hex
                            else None
                        ),
                    }
                )
        if not passed:
            ulp_findings.append(
                {
                    "oracle": row_id,
                    "total_ulp_distance": _ulp_distance(
                        local_value["hex"], external_value["hex"]
                    ),
                    "diverging_terms": diverging_terms,
                }
            )
        rows.append(
            _scored_row(
                "P1",
                row_id,
                passed,
                expected_hex=expected_hex,
                local_hex=local_value["hex"],
                external_hex=external_value["hex"],
                detail=(
                    "bit-equal total and per-term composition"
                    if not diverging_terms
                    else json.dumps(diverging_terms, sort_keys=True)
                ),
            )
        )

    rows.append(
        _scored_row(
            "W",
            "W-01",
            elapsed_seconds <= 120.0,
            detail=f"conversion plus evaluation took {elapsed_seconds:.6f} seconds",
        )
    )

    identity = external["identity"]
    imported_manifest = json.loads(
        (imported_artifact / "manifest.json").read_text(encoding="utf-8")
    )
    source = imported_manifest["source"]
    fg1 = (
        identity["slice_file_count"] == 27
        and identity["slice_sha256"] == EXPECTED_IDENTITY["slice_sha256"]
        and identity["closure_sha256"] == EXPECTED_IDENTITY["closure_sha256"]
        and identity["system_sha256"] == EXPECTED_IDENTITY["system_sha256"]
        and identity["model_sha256"] == EXPECTED_IDENTITY["model_sha256"]
        and source["data_slice_sha256"] == EXPECTED_IDENTITY["slice_sha256"]
        and _tree_hashes(imported_artifact) == _tree_hashes(TRACKED_ARTIFACT)
    )
    rows.append(_fatal_row("FG-1", fg1, "installed hashes and tracked conversion identity"))

    license_text = (TRACKED_ARTIFACT / "LICENSE").read_text(encoding="utf-8")
    third_party = (TRACKED_ARTIFACT / "THIRD_PARTY_NOTICE").read_text(encoding="utf-8")
    modified = (TRACKED_ARTIFACT / "MODIFIED").read_text(encoding="utf-8")
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    fg2 = (
        "Apache License" in license_text
        and "Version 2.0" in license_text
        and "SPDX-License-Identifier: Apache-2.0" in third_party
        and "NVIDIA CORPORATION & AFFILIATES" in third_party
        and "converted" in modified
        and "otherwise altered" in modified
        and "offline/calibration/external-databases" in notice
    )
    rows.append(_fatal_row("FG-2", fg2, "license, notices and modified-file statement"))

    fg3 = local["versions"] == ["1.3.0rc10"] and source["shared_layer"] is False
    rows.append(_fatal_row("FG-3", fg3, "all converted rows come from the primary version"))

    local_source = local["source"]
    fg4 = (
        local["evidence"]["all_match"]
        and local["evidence"]["class"] == "MEASURED-EXTERNAL"
        and local_source["aiconfigurator_version"] == EXPECTED_IDENTITY["aiconfigurator"]
        and local_source["aiconfigurator_core_version"]
        == EXPECTED_IDENTITY["aiconfigurator_core"]
        and local_source["data_slice_sha256"] == EXPECTED_IDENTITY["slice_sha256"]
    )
    rows.append(_fatal_row("FG-4", fg4, "every served value carries frozen external identity"))

    mutation = config["load_time_mutation_guard"]
    mutation_cells = local["mutation"]["cells"]
    fg5 = (
        len(mutation_cells) == 1
        and mutation_cells[0]["raw_hex"] == mutation["raw_hex"]
        and mutation_cells[0]["served_hex"] == mutation["served_hex"]
        and local["mutation"]["query_hex"] == mutation["served_hex"]
        and external["mutation_hex"] == mutation["served_hex"]
        and local["mutation"]["gemm_mutation_count"] == 3
        and local["mutation"]["generation_attention_mutation_count"] == 367
    )
    rows.append(_fatal_row("FG-5", fg5, "frozen below-SOL cell and mutation counts"))

    fg6 = local_deterministic and external_deterministic
    rows.append(_fatal_row("FG-6", fg6, "both sides repeated bit-equal in fresh processes"))

    mapping = local["mapping"]
    fg7 = (
        mapping["overlap_rejected"]
        and mapping["gap_rejected"]
        and mapping["undeclared_rejected"]
        and mapping["sha256"] == _sha256(TRACKED_ARTIFACT / "family-mapping.json")
        and sum(mapping["statuses"].values()) > 0
    )
    rows.append(_fatal_row("FG-7", fg7, "declared mappings reject gaps and overlap"))

    ancestry = {name: _is_ancestor(commit) for name, commit in FREEZE_COMMITS.items()}
    fg8 = all(ancestry.values()) and attempt_number >= 1
    rows.append(
        _fatal_row(
            "FG-8",
            fg8,
            f"freeze commits precede attempt {attempt_number:04d}: {ancestry}",
        )
    )

    return rows, {"ulp_findings": ulp_findings, "freeze_ancestry": ancestry}


def _family_tallies(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    tallies = {}
    for family in ("I1", "I2", "P1", "W"):
        family_rows = [row for row in rows if row["kind"] == "scored" and row["family"] == family]
        tallies[family] = {
            "passed": sum(row["passed"] for row in family_rows),
            "denominator": len(family_rows),
        }
    return tallies


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    import io

    output = io.StringIO(newline="")
    fieldnames = [
        "family",
        "id",
        "kind",
        "passed",
        "expected_hex",
        "local_hex",
        "external_hex",
        "ulp_distance",
        "detail",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def _coordinator(bulk_root: Path, *, write_tracked: bool) -> dict[str, Any]:
    from simllm.calibration.external_db import import_external_database

    raw_venv = os.environ.get(EXTERNAL_VENV_ENV)
    if raw_venv is None:
        raise RuntimeError(f"set {EXTERNAL_VENV_ENV} to the pinned external environment")
    venv = Path(raw_venv)
    external_python = next(
        (path for path in (venv / "bin/python", venv / "Scripts/python.exe") if path.is_file()),
        None,
    )
    if external_python is None:
        raise RuntimeError(f"{EXTERNAL_VENV_ENV} has no usable Python interpreter")

    attempt, attempt_number = _new_attempt(bulk_root)
    start = time.monotonic()
    imported_artifact = import_external_database(
        venv_root=venv,
        output_root=attempt / "imported",
    )
    local_runs = [
        _run_worker(
            python=Path(sys.executable),
            worker="local",
            artifact=imported_artifact,
            attempt=attempt,
            repetition=repetition,
        )
        for repetition in (1, 2)
    ]
    external_runs = [
        _run_worker(
            python=external_python,
            worker="external",
            artifact=imported_artifact,
            attempt=attempt,
            repetition=repetition,
        )
        for repetition in (1, 2)
    ]
    elapsed = time.monotonic() - start
    rows, findings = _evaluate(
        local=local_runs[0],
        external=external_runs[0],
        local_deterministic=local_runs[0] == local_runs[1],
        external_deterministic=external_runs[0] == external_runs[1],
        imported_artifact=imported_artifact,
        attempt_number=attempt_number,
        elapsed_seconds=elapsed,
    )
    failed_guards = [row["id"] for row in rows if row["kind"] == "fatal" and not row["passed"]]
    run_state = "void" if failed_guards else "nonvoid"
    record = {
        "schema": SCHEMA,
        "study": "external_db_parity_v1",
        "run_state": run_state,
        "voiding_guards": failed_guards,
        "attempt": f"attempt-{attempt_number:04d}",
        "bulk_evidence": f"${{{BULK_ROOT_ENV}}}/attempt-{attempt_number:04d}",
        "expectations_sha256": _sha256(EXPECTATIONS),
        "query_config_sha256": _sha256(QUERY_CONFIG),
        "freeze_commits": FREEZE_COMMITS,
        "run_commit": _git("rev-parse", "HEAD"),
        "artifact": {
            "directory_identity": EXPECTED_IDENTITY["slice_sha256"],
            "payload_sha256": local_runs[0]["payload_sha256"],
            "files": _tree_hashes(TRACKED_ARTIFACT),
        },
        "licensing": {
            name: _sha256(TRACKED_ARTIFACT / name)
            for name in ("LICENSE", "THIRD_PARTY_NOTICE", "MODIFIED")
        },
        "machine": {
            "cpu": _cpu_model(),
            "logical_cpus": os.cpu_count(),
            "architecture": platform.machine(),
            "system": platform.system(),
            "python": platform.python_version(),
        },
        "elapsed_seconds": elapsed,
        "family_tallies": _family_tallies(rows),
        "fatal_guards": {
            row["id"]: row["passed"] for row in rows if row["kind"] == "fatal"
        },
        "rows": rows,
        **findings,
    }
    record_bytes = _json_bytes(record)
    csv_bytes = _csv_bytes(rows)
    _write_new(attempt / "record.json", record_bytes)
    _write_new(attempt / "results.csv", csv_bytes)
    if write_tracked:
        _write_new(STUDY / "record.json", record_bytes)
        _write_new(STUDY / "results.csv", csv_bytes)
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk-root", type=Path)
    parser.add_argument("--write-tracked", action="store_true")
    parser.add_argument("--worker", choices=("local", "external"), help=argparse.SUPPRESS)
    parser.add_argument("--artifact", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker == "local":
        if args.artifact is None:
            raise SystemExit("local worker requires --artifact")
        print(json.dumps(_local_worker(args.artifact), sort_keys=True, separators=(",", ":")))
        return 0
    if args.worker == "external":
        print(json.dumps(_external_worker(), sort_keys=True, separators=(",", ":")))
        return 0
    raw_bulk = args.bulk_root or os.environ.get(BULK_ROOT_ENV)
    if raw_bulk is None:
        raise SystemExit(f"pass --bulk-root or set {BULK_ROOT_ENV}")
    record = _coordinator(Path(raw_bulk), write_tracked=args.write_tracked)
    tallies = record["family_tallies"]
    print(f"run_state={record['run_state']} elapsed_seconds={record['elapsed_seconds']:.6f}")
    for family in ("I1", "I2", "P1", "W"):
        print(f"{family}={tallies[family]['passed']}/{tallies[family]['denominator']}")
    print(f"fatal_guards={record['fatal_guards']}")
    print(f"ulp_findings={len(record['ulp_findings'])}")
    return 0 if record["run_state"] == "nonvoid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
