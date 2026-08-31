# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Yifeng Wang
# SPDX-License-Identifier: Apache-2.0
"""Offline import and exact resolution of a pinned external operation database.

The converted artifact keeps one record for every physical Parquet row. Float
values use Python's hexadecimal binary64 spelling, so loading the artifact does
not pass through a decimal parser. The runtime depends only on the standard
library. PyArrow and PyYAML are needed only by the isolated import worker that
runs inside the pinned external installation.

Parts of the interpolation engine are adapted from
``aiconfigurator_core.sdk.perf_interp`` version 0.11.0. The installed source is
Apache 2.0 and carries the NVIDIA copyright notice preserved above and in the
tracked artifact directory.
"""

from __future__ import annotations

import argparse
import bisect
import ctypes
import ctypes.util
import hashlib
import importlib.metadata
import importlib.util
import json
import lzma
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

EXTERNAL_DATABASE_SCHEMA = "simllm-external-operation-database-v1"
EXTERNAL_DATABASE_ROW_SCHEMA = "simllm-external-operation-row-v1"
EXTERNAL_DATABASE_CONVERTER_SCHEMA = "simllm-external-database-converter-v1"
EXTERNAL_FAMILY_MAPPING_SCHEMA = "simllm-external-family-mapping-v1"
EXTERNAL_EVIDENCE_CLASS = "MEASURED-EXTERNAL"
EXTERNAL_VENV_ENV = "SIMLLM_EXTERNAL_AIC_VENV"

EXPECTED_AICONFIGURATOR_VERSION = "0.11.0"
EXPECTED_CORE_VERSION = "0.11.0"
EXPECTED_SYSTEM = "h200_sxm"
EXPECTED_BACKEND = "trtllm"
EXPECTED_DATABASE_VERSION = "1.3.0rc10"
EXPECTED_SLICE_HASH = "85e72f990f00ea457de522d0b773e678f5e067740689912df5646f6296273284"
EXPECTED_CLOSURE_HASH = "d559d6694f30ad269ecbf697e0193c7d95e4aba1cfb929836d381a46b675876f"
EXPECTED_SYSTEM_HASH = "142584d6bddd98207fd04e844029b0ba5d6fcd4c6f41016c5e77f0cbe4053614"
EXPECTED_MODEL_HASH = "e546dacd2c772660270233f5579e9ab923cc2a7ec5ed3c58c27c2bc62cbf5169"
EXPECTED_APACHE_LICENSE_HASH = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
EXPECTED_ROW_COUNT = 284_717

SYSTEM_CONVERSION_NOTICE = (
    "This file was converted by SimLLM from aiconfigurator-core 0.11.0 "
    "systems/h200_sxm.yaml; see MODIFIED."
)
MODEL_CONVERSION_NOTICE = (
    "This file was converted by SimLLM from aiconfigurator-core 0.11.0 "
    "model_configs/Qwen--Qwen3-32B-FP8_config.json; see MODIFIED."
)
NVIDIA_SYSTEM_COPYRIGHT = (
    "SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & "
    "AFFILIATES. All rights reserved."
)
NVIDIA_COLLECTION_COPYRIGHT = (
    "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & "
    "AFFILIATES. All rights reserved."
)

ARTIFACT_DIRECTORY_NAME = EXPECTED_SLICE_HASH
ARTIFACT_RELATIVE_ROOT = Path("offline/calibration/external-databases")

_TABLE_INVENTORY: tuple[tuple[str, str, int], ...] = (
    ("context_attention", "attention/context_attention_perf.parquet", 50_574),
    ("generation_attention", "attention/generation_attention_perf.parquet", 24_438),
    ("custom_allreduce", "comm/custom_allreduce_perf.parquet", 69),
    ("encoder_attention", "encoder_attention/encoder_attention_perf.parquet", 6_314),
    ("gemm", "gemm/gemm_perf.parquet", 101_010),
    ("gdn", "linear_attention/gdn_perf.parquet", 1_862),
    ("mamba2", "linear_attention/mamba2_perf.parquet", 469),
    ("context_mla", "mla/context_mla_perf.parquet", 1_760),
    ("generation_mla", "mla/generation_mla_perf.parquet", 2_896),
    ("context_mla_module", "mla/mla_context_module_perf.parquet", 3_873),
    ("generation_mla_module", "mla/mla_generation_module_perf.parquet", 5_888),
    ("mla_bmm", "mla_bmm/mla_bmm_perf.parquet", 848),
    ("moe", "moe/moe_perf.parquet", 74_358),
    ("wideep_moe", "moe/wideep_moe_perf.parquet", 4_158),
    ("compute_scale", "quantize/computescale_perf.parquet", 1_628),
    ("scale_matrix", "quantize/scale_matrix_perf.parquet", 1_628),
    ("generation_dsa_module", "sparse_attention/dsa_generation_module_perf.parquet", 2_944),
)

_SLICE_METADATA_PATHS = (
    "attention/collection_meta.yaml",
    "comm/collection_meta.yaml",
    "encoder_attention/collection_meta.yaml",
    "gemm/collection_meta.yaml",
    "linear_attention/collection_meta.yaml",
    "mla/collection_meta.yaml",
    "mla_bmm/collection_meta.yaml",
    "moe/collection_meta.yaml",
    "quantize/collection_meta.yaml",
    "sparse_attention/collection_meta.yaml",
)

_QUANT_MEMORY_BYTES = {
    "bfloat16": 2.0,
    "half": 2.0,
    "int8": 1.0,
    "fp8": 1.0,
    "fp8_block": 1.0,
    "fp8_static": 1.0,
    "w4a16_mxfp4": 0.5,
}
_QUANT_COMPUTE_FACTOR = {
    "bfloat16": 1.0,
    "fp8": 2.0,
    "fp8_block": 2.0,
    "fp8_static": 2.0,
    "w4a16_mxfp4": 1.0,
}

_GDN_DECODE_REKEY = {
    "fused_recurrent_gated_delta_rule": "fused_sigmoid_gating_delta_rule_update",
    "fused_recurrent_gated_delta_rule_packed_decode": "fused_sigmoid_gating_delta_rule_update",
}


class ExternalDatabaseError(ValueError):
    """Base error for a malformed, unsupported, or incomplete external database."""


class ExternalDatabaseIdentityError(ExternalDatabaseError):
    """The installed source or converted artifact does not match its frozen identity."""


class ExternalDatabaseGapError(ExternalDatabaseError):
    """No declared measured correspondence can price the requested family."""


class ExternalCompositeError(ExternalDatabaseError):
    """A composite pricing request is missing or would double count a fused value."""


class InterpolationDataNotAvailableError(ExternalDatabaseGapError):
    """A measured table has no legal anchor for one interpolation query."""


@dataclass(frozen=True)
class ExternalSourceIdentity:
    """Frozen identity carried by every value served from the artifact."""

    tool: str
    aiconfigurator_version: str
    core_version: str
    system: str
    backend: str
    database_version: str
    slice_hash: str
    database_mode: str = "SILICON"
    shared_layer: bool = False
    estimator_surface: str = "python"

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> ExternalSourceIdentity:
        source = _require_mapping(manifest.get("source"), "manifest.source")
        return cls(
            tool=_require_string(source.get("tool"), "manifest.source.tool"),
            aiconfigurator_version=_require_string(
                source.get("aiconfigurator_version"),
                "manifest.source.aiconfigurator_version",
            ),
            core_version=_require_string(
                source.get("aiconfigurator_core_version"),
                "manifest.source.aiconfigurator_core_version",
            ),
            system=_require_string(source.get("system"), "manifest.source.system"),
            backend=_require_string(source.get("backend"), "manifest.source.backend"),
            database_version=_require_string(
                source.get("database_version"),
                "manifest.source.database_version",
            ),
            slice_hash=_require_string(source.get("data_slice_sha256"), "manifest.source.data_slice_sha256"),
            database_mode=_require_string(source.get("database_mode"), "manifest.source.database_mode"),
            shared_layer=_require_bool(source.get("shared_layer"), "manifest.source.shared_layer"),
            estimator_surface=_require_string(
                source.get("estimator_surface"),
                "manifest.source.estimator_surface",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe source record."""

        return {
            "tool": self.tool,
            "aiconfigurator_version": self.aiconfigurator_version,
            "aiconfigurator_core_version": self.core_version,
            "system": self.system,
            "backend": self.backend,
            "database_version": self.database_version,
            "data_slice_sha256": self.slice_hash,
            "database_mode": self.database_mode,
            "shared_layer": self.shared_layer,
            "estimator_surface": self.estimator_surface,
        }


@dataclass(frozen=True)
class ExternalLatency:
    """One latency value with non-substitutable external measured evidence."""

    latency_ms: float
    source: ExternalSourceIdentity
    operation: str
    rule: str
    evidence_class: str

    def __post_init__(self) -> None:
        if self.evidence_class != EXTERNAL_EVIDENCE_CLASS:
            raise ValueError(
                f"external latency evidence must be {EXTERNAL_EVIDENCE_CLASS!r}, "
                f"got {self.evidence_class!r}"
            )
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0.0:
            raise ValueError("external latency must be finite and non-negative")

    @property
    def hex(self) -> str:
        """Return the exact IEEE-754 binary64 spelling."""

        return self.latency_ms.hex()


@dataclass(frozen=True)
class _Row:
    table: str
    version: str
    key: tuple[Any, ...]
    raw_latency: float
    served_latency: float


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExternalDatabaseIdentityError(f"{path} must be an object")
    return value


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExternalDatabaseIdentityError(f"{path} must be a non-empty string")
    return value


def _require_bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise ExternalDatabaseIdentityError(f"{path} must be a boolean")
    return value


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _liblzma_version() -> str:
    library_name = ctypes.util.find_library("lzma")
    if library_name is None:
        raise ExternalDatabaseIdentityError("the linked liblzma version is unavailable")
    library = ctypes.CDLL(library_name)
    version_string = library.lzma_version_string
    version_string.restype = ctypes.c_char_p
    raw = version_string()
    if raw is None:
        raise ExternalDatabaseIdentityError("the linked liblzma version is unavailable")
    return raw.decode("ascii")


def _conversion_recipe(*, pyyaml_version: str, liblzma_version: str) -> dict[str, Any]:
    return {
        "converter_schema": EXTERNAL_DATABASE_CONVERTER_SCHEMA,
        "pyyaml": pyyaml_version,
        "liblzma": liblzma_version,
        "row_ordering": (
            "tables follow the declared manifest inventory order; rows within each "
            "table retain PyArrow to_pylist source order; no row sort is applied"
        ),
        "float_encoding": (
            "Python float.hex() encodes raw and served IEEE-754 binary64 values"
        ),
        "json_lines": {
            "record": "one six-element JSON array per source row",
            "separators": [",", ":"],
            "ensure_ascii": True,
            "encoding": "ASCII",
            "line_termination": "LF (0x0a) after every record",
        },
        "xz": {
            "format": "FORMAT_XZ",
            "check": "CHECK_CRC64",
            "preset": 9,
            "extreme": True,
            "preset_expression": "9 | PRESET_EXTREME",
            "stream_layout": (
                "one XZ stream from one lzma.compress call over the complete JSON Lines payload"
            ),
        },
    }


def _validate_conversion_recipe(conversion: Mapping[str, Any]) -> None:
    recipe = _require_mapping(conversion.get("recipe"), "manifest.conversion.recipe")
    if recipe.get("converter_schema") != EXTERNAL_DATABASE_CONVERTER_SCHEMA:
        raise ExternalDatabaseIdentityError("external converter schema identity mismatch")
    for name in ("pyyaml", "liblzma"):
        _require_string(recipe.get(name), f"manifest.conversion.recipe.{name}")
    expected = _conversion_recipe(
        pyyaml_version=str(recipe["pyyaml"]),
        liblzma_version=str(recipe["liblzma"]),
    )
    if dict(recipe) != expected:
        raise ExternalDatabaseIdentityError("external conversion recipe is incomplete or changed")


def _sorted_sha256_manifest(root: Path, relative_paths: Iterable[str]) -> bytes:
    lines = []
    for relative in sorted(relative_paths):
        path = root / relative
        if not path.is_file():
            raise ExternalDatabaseIdentityError(f"missing frozen source file {relative}")
        lines.append(f"{_sha256_file(path)}  {relative}\n")
    return "".join(lines).encode("utf-8")


def _slice_relative_paths(data_root: Path) -> list[str]:
    suffix = f"/{EXPECTED_BACKEND}/{EXPECTED_DATABASE_VERSION}/"
    return sorted(
        path.relative_to(data_root).as_posix()
        for path in data_root.rglob("*")
        if path.is_file() and suffix in f"/{path.relative_to(data_root).as_posix()}"
    )


def _installed_python(venv_root: Path) -> Path:
    candidates = (venv_root / "bin/python", venv_root / "Scripts/python.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ExternalDatabaseIdentityError(
        f"{EXTERNAL_VENV_ENV} does not name a Python virtual environment with a usable interpreter"
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_artifact_dir() -> Path:
    """Return the tracked artifact directory in a source checkout."""

    return _repo_root() / ARTIFACT_RELATIVE_ROOT / ARTIFACT_DIRECTORY_NAME


def import_external_database(
    *,
    venv_root: str | os.PathLike[str] | None = None,
    output_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Run the converter in the pinned environment and return its artifact path."""

    raw_venv = venv_root if venv_root is not None else os.environ.get(EXTERNAL_VENV_ENV)
    if raw_venv is None:
        raise ExternalDatabaseIdentityError(
            f"set {EXTERNAL_VENV_ENV} to the pinned aiconfigurator virtual environment"
        )
    venv = Path(raw_venv).expanduser()
    python = _installed_python(venv)
    destination_root = Path(output_root) if output_root is not None else _repo_root() / ARTIFACT_RELATIVE_ROOT
    command = [
        os.fspath(python),
        os.fspath(Path(__file__).resolve()),
        "--worker-import",
        "--output-root",
        os.fspath(destination_root),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "converter failed without diagnostics"
        raise ExternalDatabaseError(f"external database import failed: {detail}")
    artifact = destination_root / ARTIFACT_DIRECTORY_NAME
    if not artifact.is_dir():
        raise ExternalDatabaseError("converter reported success without creating the expected artifact")
    return artifact


def _package_root(package: str) -> Path:
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise ExternalDatabaseIdentityError(f"installed package {package!r} is unavailable")
    return Path(spec.origin).resolve().parent


def _verify_installed_identity() -> tuple[Path, dict[str, str]]:
    versions = {
        "aiconfigurator": importlib.metadata.version("aiconfigurator"),
        "aiconfigurator-core": importlib.metadata.version("aiconfigurator-core"),
        "numpy": importlib.metadata.version("numpy"),
        "pyarrow": importlib.metadata.version("pyarrow"),
    }
    expected = {
        "aiconfigurator": EXPECTED_AICONFIGURATOR_VERSION,
        "aiconfigurator-core": EXPECTED_CORE_VERSION,
    }
    mismatches = [f"{name}={versions[name]} (expected {version})" for name, version in expected.items() if versions[name] != version]
    if mismatches:
        raise ExternalDatabaseIdentityError("installed package identity mismatch: " + ", ".join(mismatches))
    return _package_root("aiconfigurator_core"), versions


def _gemm_sol_ms(system_spec: Mapping[str, Any], quant: str, m: int, n: int, k: int) -> float:
    gpu = _require_mapping(system_spec.get("gpu"), "system.gpu")
    compute = _QUANT_COMPUTE_FACTOR[quant]
    if compute == 1.0:
        flops = float(gpu["bfloat16_tc_flops"])
    elif compute == 2.0:
        flops = float(gpu.get("fp8_tc_flops", float(gpu["bfloat16_tc_flops"]) * compute))
    else:
        flops = float(gpu["bfloat16_tc_flops"]) * compute
    sol_math = 2.0 * m * n * k / flops * 1000.0
    memory = _QUANT_MEMORY_BYTES[quant]
    sol_mem = memory * (m * n + m * k + n * k) / float(gpu["mem_bw"]) * 1000.0
    return max(sol_math, sol_mem)


def _generation_attention_sol_ms(
    system_spec: Mapping[str, Any],
    *,
    batch: int,
    sequence: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    window_size: int,
    kv_quant: str,
) -> float:
    gpu = _require_mapping(system_spec.get("gpu"), "system.gpu")
    kv_len = min(sequence - 1, window_size) if window_size > 0 else sequence - 1
    kv_len = max(0, kv_len)
    ops = 2 * batch * num_heads * head_size * 2 * kv_len
    mem_bytes = batch * (
        num_heads * head_size * 2
        + 2 * num_kv_heads * kv_len * head_size * _QUANT_MEMORY_BYTES[kv_quant]
        + num_heads * head_size * 2
    )
    compute = 2.0 if kv_quant == "fp8" else 1.0
    sol_math = ops / float(gpu["bfloat16_tc_flops"]) * 1000.0 / compute
    sol_mem = mem_bytes / float(gpu["mem_bw"]) * 1000.0
    return max(sol_math, sol_mem)


def _normalized_key(table: str, row: Mapping[str, Any]) -> tuple[Any, ...]:
    integer = lambda name: int(row[name])
    if table == "gemm":
        return (row["gemm_dtype"], integer("m"), integer("n"), integer("k"))
    if table in {"compute_scale", "scale_matrix"}:
        return (row["quant_dtype"], integer("m"), integer("k"))
    if table == "context_attention":
        n = integer("num_heads")
        n_kv = integer("num_key_value_heads")
        return (
            row["attn_dtype"],
            row["kv_cache_dtype"],
            0 if n == n_kv else n_kv,
            integer("head_dim"),
            integer("window_size"),
            n,
            integer("isl"),
            integer("batch_size"),
        )
    if table == "generation_attention":
        n = integer("num_heads")
        n_kv = integer("num_key_value_heads")
        return (
            row["kv_cache_dtype"],
            0 if n == n_kv else n_kv,
            integer("head_dim"),
            integer("window_size"),
            n,
            integer("batch_size"),
            integer("isl") + integer("step"),
        )
    if table == "encoder_attention":
        return (
            row["attn_dtype"],
            integer("head_dim"),
            integer("num_heads"),
            integer("isl"),
            integer("batch_size"),
        )
    if table == "custom_allreduce":
        return ("half", integer("num_gpus"), "AUTO", integer("message_size"))
    if table == "moe":
        return (
            row["moe_dtype"],
            row["distribution"],
            integer("topk"),
            integer("num_experts"),
            integer("hidden_size"),
            integer("inter_size"),
            integer("moe_tp_size"),
            integer("moe_ep_size"),
            integer("num_tokens"),
        )
    if table == "wideep_moe":
        return (
            row["kernel_source"],
            row["moe_dtype"],
            row["distribution"],
            integer("topk"),
            integer("num_experts"),
            integer("hidden_size"),
            integer("inter_size"),
            integer("num_slots"),
            integer("moe_tp_size"),
            integer("moe_ep_size"),
            integer("num_tokens"),
        )
    if table == "gdn":
        source = _GDN_DECODE_REKEY.get(str(row["kernel_source"]), str(row["kernel_source"]))
        key = (
            source,
            row["phase"],
            integer("d_model"),
            integer("num_k_heads"),
            integer("head_k_dim"),
            integer("num_v_heads"),
            integer("head_v_dim"),
            integer("d_conv"),
            integer("batch_size"),
        )
        return key + ((integer("seq_len"),) if row["phase"] == "context" else ())
    if table == "mamba2":
        key = (
            row["kernel_source"],
            row["phase"],
            integer("d_model"),
            integer("d_state"),
            integer("d_conv"),
            integer("nheads"),
            integer("head_dim"),
            integer("n_groups"),
            integer("chunk_size"),
            integer("batch_size"),
        )
        return key + ((integer("seq_len"),) if row["phase"] == "context" else ())
    if table == "context_mla":
        return (
            row["mla_dtype"],
            row["kv_cache_dtype"],
            integer("num_heads"),
            integer("isl"),
            integer("batch_size"),
        )
    if table == "generation_mla":
        return (
            row["kv_cache_dtype"],
            integer("num_heads"),
            integer("batch_size"),
            integer("isl") + integer("step"),
        )
    if table == "context_mla_module":
        return (
            row["mla_dtype"],
            row["kv_cache_dtype"],
            row["gemm_type"],
            integer("num_heads"),
            integer("isl"),
            integer("batch_size"),
        )
    if table == "generation_mla_module":
        return (
            row["kv_cache_dtype"],
            row["gemm_type"],
            integer("num_heads"),
            integer("batch_size"),
            integer("isl") + integer("step"),
        )
    if table == "mla_bmm":
        return (
            row["bmm_dtype"],
            row["op_name"],
            integer("num_heads"),
            integer("num_tokens"),
        )
    if table == "generation_dsa_module":
        return (
            row["kv_cache_dtype"],
            row["gemm_type"],
            row["architecture"],
            ("trtllm", "flashmla_kv") if row["kv_cache_dtype"] == "bfloat16" else ("trtllm",),
            integer("num_heads"),
            integer("batch_size"),
            integer("isl") + integer("step"),
        )
    raise ExternalDatabaseError(f"no key normalizer declared for table {table!r}")


def _served_latency(
    table: str,
    row: Mapping[str, Any],
    key: tuple[Any, ...],
    system_spec: Mapping[str, Any],
) -> float:
    raw = float(row["latency"])
    if table == "gemm":
        quant, m, n, k = key
        return max(raw, _gemm_sol_ms(system_spec, str(quant), int(m), int(n), int(k)))
    if table == "generation_attention":
        kv_quant, n_kv_key, head_size, window_size, n, batch, sequence = key
        n_kv = int(n) if int(n_kv_key) == 0 else int(n_kv_key)
        sol = _generation_attention_sol_ms(
            system_spec,
            batch=int(batch),
            sequence=int(sequence),
            num_heads=int(n),
            num_kv_heads=n_kv,
            head_size=int(head_size),
            window_size=int(window_size),
            kv_quant=str(kv_quant),
        )
        return max(raw, sol)
    return raw


def _artifact_notices(payload_name: str) -> tuple[str, str]:
    notice = f"""Third-party converted operation database

{NVIDIA_SYSTEM_COPYRIGHT}
{NVIDIA_COLLECTION_COPYRIGHT}
SPDX-License-Identifier: Apache-2.0

Source packages: aiconfigurator 0.11.0 and aiconfigurator-core 0.11.0.
The 2025-2026 line comes from systems/h200_sxm.yaml. The 2026 line comes
from each source collection_meta.yaml in the converted database slice.
"""
    modified = f"""Modified-file statement

SimLLM converted the source Parquet and YAML slice into deterministic compressed
JSON Lines plus portable JSON metadata. The conversion normalizes audited lookup
keys and records both the source latency and the latency after the source tool's
GEMM and generation-attention speed-of-light clamps. No measured latency was
otherwise altered. The converted files are not original NVIDIA distribution files.

Artifact derivations:

- LICENSE: byte-identical Apache License 2.0 text from the aiconfigurator 0.11.0
  installed package license.
- MODIFIED: SimLLM's file-by-file conversion and modification statement.
- THIRD_PARTY_NOTICE: the SPDX and copyright lines retained from
  systems/h200_sxm.yaml and the ten collection_meta.yaml source files.
- family-mapping.json: SimLLM's audited exact, composite and gap projection of
  the imported external operation families.
- manifest.json: SimLLM's generated source identity, conversion recipe, table
  inventory, hashes, mutations and resolver provenance record.
- model-config.json: JSON conversion of
  model_configs/Qwen--Qwen3-32B-FP8_config.json with a file-local notice.
- {payload_name}: row-preserving conversion of the 17 source Parquet tables
  to ordered JSON Lines with raw and served float.hex values, then one XZ stream.
- source-files.sha256: sorted SHA-256 manifest of the 27 source slice files.
- system.json: JSON conversion of systems/h200_sxm.yaml with a file-local notice.
"""
    return notice, modified


def external_artifact_licensing_findings(
    artifact_dir: str | os.PathLike[str],
    repository_notice: str | os.PathLike[str],
) -> tuple[str, ...]:
    """Return exact licensing-guard findings for one converted artifact."""

    artifact = Path(artifact_dir)
    findings = []
    license_path = artifact / "LICENSE"
    if not license_path.is_file() or _sha256_file(license_path) != EXPECTED_APACHE_LICENSE_HASH:
        findings.append("LICENSE is not byte-identical to the frozen upstream Apache 2.0 text")

    try:
        third_party = (artifact / "THIRD_PARTY_NOTICE").read_text(encoding="utf-8")
    except FileNotFoundError:
        third_party = ""
    for line in (
        NVIDIA_SYSTEM_COPYRIGHT,
        NVIDIA_COLLECTION_COPYRIGHT,
        "SPDX-License-Identifier: Apache-2.0",
    ):
        if line not in third_party.splitlines():
            findings.append(f"THIRD_PARTY_NOTICE is missing exact line {line!r}")

    expected_notices = {
        "system.json": SYSTEM_CONVERSION_NOTICE,
        "model-config.json": MODEL_CONVERSION_NOTICE,
    }
    for filename, expected_notice in expected_notices.items():
        try:
            document = json.loads((artifact / filename).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            findings.append(f"{filename} is missing or invalid")
            continue
        if document.get("notice") != expected_notice:
            findings.append(f"{filename} has no exact file-local conversion notice")

    try:
        modified = (artifact / "MODIFIED").read_text(encoding="utf-8")
    except FileNotFoundError:
        modified = ""
    manifest = artifact / "manifest.json"
    payload_names = []
    if manifest.is_file():
        try:
            payload_names.append(
                str(json.loads(manifest.read_text(encoding="utf-8"))["conversion"]["payload"])
            )
        except (KeyError, TypeError, json.JSONDecodeError):
            pass
    for filename in (
        "LICENSE",
        "MODIFIED",
        "THIRD_PARTY_NOTICE",
        "family-mapping.json",
        "manifest.json",
        "model-config.json",
        *payload_names,
        "source-files.sha256",
        "system.json",
    ):
        if f"- {filename}:" not in modified:
            findings.append(f"MODIFIED does not enumerate {filename}")

    try:
        repository_lines = Path(repository_notice).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        repository_lines = []
    repository_required = (
        "offline/calibration/external-databases is derived from aiconfigurator 0.11.0",
        "and aiconfigurator-core 0.11.0. Copyright (c) 2025-2026 NVIDIA CORPORATION &",
        "AFFILIATES. All rights reserved. Licensed under the Apache License, Version",
    )
    for line in repository_required:
        if line not in repository_lines:
            findings.append(f"repository NOTICE is missing exact line {line!r}")
    return tuple(findings)


def _source_provenance() -> list[dict[str, Any]]:
    return [
        {"semantic": "generic resolver", "path": "aiconfigurator_core/sdk/perf_interp/engine.py", "line": 105},
        {"semantic": "resolver configuration", "path": "aiconfigurator_core/sdk/perf_interp/config.py", "line": 111},
        {"semantic": "GEMM loader and keys", "path": "aiconfigurator_core/sdk/operations/gemm.py", "line": 835},
        {"semantic": "attention loaders and keys", "path": "aiconfigurator_core/sdk/operations/attention.py", "line": 1127},
        {"semantic": "custom all-reduce rekey", "path": "aiconfigurator_core/sdk/operations/communication.py", "line": 613},
        {"semantic": "MLA loaders and keys", "path": "aiconfigurator_core/sdk/operations/mla.py", "line": 1538},
        {"semantic": "MoE loaders and keys", "path": "aiconfigurator_core/sdk/operations/moe.py", "line": 2391},
        {"semantic": "phase composition", "path": "aiconfigurator_core/sdk/backends/base_backend.py", "line": 313},
    ]


def _write_worker_artifact(output_root: Path) -> Path:
    try:
        import pyarrow.parquet as pq
        import yaml
    except ImportError as error:
        raise ExternalDatabaseIdentityError(
            "the import worker must run inside the pinned external environment with PyArrow and PyYAML"
        ) from error

    package_root, versions = _verify_installed_identity()
    systems_root = package_root / "systems"
    data_root = systems_root / "data" / EXPECTED_SYSTEM
    slice_paths = _slice_relative_paths(data_root)
    if len(slice_paths) != 27:
        raise ExternalDatabaseIdentityError(
            f"frozen slice must contain 27 files, found {len(slice_paths)}"
        )
    slice_manifest = _sorted_sha256_manifest(data_root, slice_paths)
    slice_hash = _sha256_bytes(slice_manifest)
    if slice_hash != EXPECTED_SLICE_HASH:
        raise ExternalDatabaseIdentityError(
            f"data-slice hash mismatch: expected {EXPECTED_SLICE_HASH}, found {slice_hash}"
        )

    data_prefix = Path("systems") / "data" / EXPECTED_SYSTEM
    closure_paths = [(data_prefix / relative).as_posix() for relative in slice_paths]
    closure_paths.append(f"systems/{EXPECTED_SYSTEM}.yaml")
    closure_manifest = _sorted_sha256_manifest(package_root, closure_paths)
    closure_hash = _sha256_bytes(closure_manifest)
    if closure_hash != EXPECTED_CLOSURE_HASH:
        raise ExternalDatabaseIdentityError(
            f"pricing-closure hash mismatch: expected {EXPECTED_CLOSURE_HASH}, found {closure_hash}"
        )

    system_yaml_path = systems_root / f"{EXPECTED_SYSTEM}.yaml"
    model_path = package_root / "model_configs" / "Qwen--Qwen3-32B-FP8_config.json"
    if _sha256_file(system_yaml_path) != EXPECTED_SYSTEM_HASH:
        raise ExternalDatabaseIdentityError("H200 system specification hash mismatch")
    if _sha256_file(model_path) != EXPECTED_MODEL_HASH:
        raise ExternalDatabaseIdentityError("Qwen3-32B-FP8 model configuration hash mismatch")
    system_spec = yaml.safe_load(system_yaml_path.read_text(encoding="utf-8"))
    model_config = json.loads(model_path.read_text(encoding="utf-8"))
    system_spec["notice"] = SYSTEM_CONVERSION_NOTICE
    model_config["notice"] = MODEL_CONVERSION_NOTICE

    table_records: list[bytes] = []
    table_manifest: list[dict[str, Any]] = []
    total_rows = 0
    version_values: set[str] = set()
    for table_name, relative, expected_rows in _TABLE_INVENTORY:
        source_path = data_root / Path(relative).parent / EXPECTED_BACKEND / EXPECTED_DATABASE_VERSION / Path(relative).name
        arrow = pq.read_table(source_path)
        if arrow.num_rows != expected_rows:
            raise ExternalDatabaseIdentityError(
                f"{table_name} row count mismatch: expected {expected_rows}, found {arrow.num_rows}"
            )
        rows = arrow.to_pylist()
        table_manifest.append(
            {
                "table": table_name,
                "source_path": source_path.relative_to(data_root).as_posix(),
                "source_sha256": _sha256_file(source_path),
                "rows": len(rows),
            }
        )
        total_rows += len(rows)
        for row in rows:
            version = str(row["version"])
            version_values.add(version)
            key = _normalized_key(table_name, row)
            raw = float(row["latency"])
            served = _served_latency(table_name, row, key, system_spec)
            record = [
                EXTERNAL_DATABASE_ROW_SCHEMA,
                table_name,
                version,
                list(key),
                raw.hex(),
                served.hex(),
            ]
            table_records.append(json.dumps(record, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n")

    if total_rows != EXPECTED_ROW_COUNT:
        raise ExternalDatabaseIdentityError(
            f"converted row total mismatch: expected {EXPECTED_ROW_COUNT}, found {total_rows}"
        )
    if version_values != {EXPECTED_DATABASE_VERSION}:
        raise ExternalDatabaseIdentityError(
            f"converted rows contain donor versions: {sorted(version_values)!r}"
        )

    payload_raw = b"".join(table_records)
    payload = lzma.compress(
        payload_raw,
        format=lzma.FORMAT_XZ,
        check=lzma.CHECK_CRC64,
        preset=9 | lzma.PRESET_EXTREME,
    )
    payload_hash = _sha256_bytes(payload)
    payload_name = f"rows-{payload_hash}.jsonl.xz"
    family_mapping_bytes = (_repo_root() / "simllm/calibration/external_family_mapping.json").read_bytes()
    system_spec_bytes = _json_bytes(system_spec)
    model_config_bytes = _json_bytes(model_config)
    converted_files_sha256 = {
        "family-mapping.json": _sha256_bytes(family_mapping_bytes),
        "model-config.json": _sha256_bytes(model_config_bytes),
        "system.json": _sha256_bytes(system_spec_bytes),
    }
    manifest = {
        "schema": EXTERNAL_DATABASE_SCHEMA,
        "source": {
            "tool": "NVIDIA AIConfigurator",
            "aiconfigurator_version": versions["aiconfigurator"],
            "aiconfigurator_core_version": versions["aiconfigurator-core"],
            "system": EXPECTED_SYSTEM,
            "backend": EXPECTED_BACKEND,
            "database_version": EXPECTED_DATABASE_VERSION,
            "database_mode": "SILICON",
            "shared_layer": False,
            "strict_provenance": False,
            "manifest_provenance": "legacy",
            "estimator_surface": "python",
            "data_slice_sha256": slice_hash,
            "pricing_closure_sha256": closure_hash,
            "system_spec_sha256": EXPECTED_SYSTEM_HASH,
            "model_config_sha256": EXPECTED_MODEL_HASH,
        },
        "conversion": {
            "format": "XZ-compressed JSON Lines with binary64 hex values",
            "modified": True,
            "row_schema": EXTERNAL_DATABASE_ROW_SCHEMA,
            "rows": total_rows,
            "payload": payload_name,
            "payload_sha256": payload_hash,
            "payload_bytes": len(payload),
            "uncompressed_bytes": len(payload_raw),
            "python": platform_python_version(),
            "numpy": versions["numpy"],
            "pyarrow": versions["pyarrow"],
            "recipe": _conversion_recipe(
                pyyaml_version=str(yaml.__version__),
                liblzma_version=_liblzma_version(),
            ),
        },
        "tables": table_manifest,
        "source_hash_manifest": "source-files.sha256",
        "converted_files_sha256": converted_files_sha256,
        "family_mapping_sha256": converted_files_sha256["family-mapping.json"],
        "resolver_sources": _source_provenance(),
        "ignored_dimensions": {
            "generation_attention": ["attn_dtype"],
            "generation_mla_module": ["mla_dtype", "tp_size"],
            "mla": ["tp_size"],
            "gdn_mamba": ["model_name", "num_tokens"],
            "standard_moe": ["kernel_source except low-latency table selection"],
            "wideep_moe": ["dp_num_tokens", "rank0_num_tokens", "moe_kernel", "simulation_mode"],
        },
        "rewritten_dimensions": {
            "custom_allreduce": {"source": "bfloat16", "lookup": "half"},
            "gdn_generation_recurrence": {
                "source": "fused_recurrent_gated_delta_rule",
                "lookup": "fused_sigmoid_gating_delta_rule_update",
            },
        },
        "load_time_mutations": [
            "GEMM latency is raised to its analytical speed-of-light floor",
            "generation-attention latency is raised to its analytical speed-of-light floor",
        ],
        "power_fields": "absent in every source table; imported power and energy are zero",
    }

    output_root.mkdir(parents=True, exist_ok=True)
    artifact = output_root / ARTIFACT_DIRECTORY_NAME
    if artifact.exists():
        raise ExternalDatabaseError(f"refusing to overwrite existing artifact {artifact}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{ARTIFACT_DIRECTORY_NAME}.", dir=output_root))
    try:
        (temporary / payload_name).write_bytes(payload)
        (temporary / "manifest.json").write_bytes(_json_bytes(manifest))
        (temporary / "system.json").write_bytes(system_spec_bytes)
        (temporary / "model-config.json").write_bytes(model_config_bytes)
        (temporary / "source-files.sha256").write_bytes(slice_manifest)
        notice, modified = _artifact_notices(payload_name)
        (temporary / "THIRD_PARTY_NOTICE").write_text(notice, encoding="utf-8", newline="\n")
        (temporary / "MODIFIED").write_text(modified, encoding="utf-8", newline="\n")
        license_candidates = sorted(package_root.parent.glob("aiconfigurator_core-*.dist-info/licenses/LICENSE"))
        if not license_candidates:
            license_candidates = sorted(package_root.parent.glob("aiconfigurator-*.dist-info/licenses/LICENSE"))
        if not license_candidates:
            raise ExternalDatabaseIdentityError("installed Apache 2.0 license text is unavailable")
        if _sha256_file(license_candidates[0]) != EXPECTED_APACHE_LICENSE_HASH:
            raise ExternalDatabaseIdentityError(
                "installed Apache 2.0 license text differs from the frozen upstream bytes"
            )
        shutil.copyfile(license_candidates[0], temporary / "LICENSE")
        (temporary / "family-mapping.json").write_bytes(family_mapping_bytes)
        os.replace(temporary, artifact)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return artifact


def platform_python_version() -> str:
    """Return the interpreter version without machine-specific details."""

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


class _ValueTransform(str, Enum):
    RAW = "raw"
    SQRT = "sqrt"


@dataclass(frozen=True)
class _ScatteredSites:
    site_axes: tuple[str, ...]
    curve_axis: str
    nn_sites: int = 4
    max_site_distance: float | None = None
    require_curve_coverage: bool = True
    k_tail: int = 3


@dataclass(frozen=True)
class _Grid:
    k_tail: int = 1


@dataclass(frozen=True)
class _InterpConfig:
    axes: tuple[str, ...]
    resolver: _ScatteredSites | _Grid
    sol_fn: Callable[..., float]
    value_transform: _ValueTransform = _ValueTransform.RAW
    transform_axis: str | None = None


class _OutOfRangeError(Exception):
    pass


_MISSING = object()


def _miss(config: _InterpConfig, coordinates: Sequence[Any], reason: str) -> InterpolationDataNotAvailableError:
    named = dict(zip(config.axes, coordinates, strict=True))
    return InterpolationDataNotAvailableError(f"external resolver has no data for {named} ({reason})")


def _to_space(transform: _ValueTransform, latency: float) -> float:
    if transform is _ValueTransform.SQRT:
        return math.sqrt(latency) if latency > 0.0 else 0.0
    return latency


def _from_space(transform: _ValueTransform, value: float) -> float:
    return value * value if transform is _ValueTransform.SQRT else value


def _walk_leaves(
    node: Mapping[Any, Any],
    depth: int,
    axes: int,
    prefix: list[Any],
    output: list[tuple[tuple[Any, ...], float]],
) -> None:
    if depth == axes:
        output.append((tuple(prefix), float(node)))
        return
    for key, child in node.items():
        prefix.append(key)
        _walk_leaves(child, depth + 1, axes, prefix, output)
        prefix.pop()


def _full_coordinates(
    axes: int,
    curve_position: int,
    site_positions: tuple[int, ...],
    curve_value: Any,
    site_values: tuple[Any, ...],
) -> tuple[Any, ...]:
    coordinates: list[Any] = [None] * axes
    coordinates[curve_position] = curve_value
    for position, value in zip(site_positions, site_values, strict=True):
        coordinates[position] = value
    return tuple(coordinates)


def _hold_curve_utilization(
    config: _InterpConfig,
    tail: Sequence[tuple[Any, float]],
    query: Any,
    axes: int,
    curve_position: int,
    site_positions: tuple[int, ...],
    site_values: tuple[Any, ...],
    coordinates: Sequence[Any],
) -> float:
    utilizations = []
    for curve_value, latency in tail:
        sol = config.sol_fn(
            *_full_coordinates(axes, curve_position, site_positions, curve_value, site_values)
        )
        if latency > 0.0 and sol > 0.0:
            utilizations.append(sol / latency)
    if not utilizations:
        raise _miss(config, coordinates, "no positive-utilization boundary anchor")
    sol_query = config.sol_fn(
        *_full_coordinates(axes, curve_position, site_positions, query, site_values)
    )
    if sol_query <= 0.0:
        raise _miss(config, coordinates, "non-positive speed-of-light value at query")
    return sol_query / statistics.median(utilizations)


def _evaluate_curve(
    config: _InterpConfig,
    curve: Sequence[tuple[Any, float]],
    query: Any,
    axes: int,
    curve_position: int,
    site_positions: tuple[int, ...],
    site_values: tuple[Any, ...],
    coordinates: Sequence[Any],
) -> float:
    curve_coordinates = [coordinate for coordinate, _ in curve]
    index = bisect.bisect_left(curve_coordinates, query)
    if index < len(curve_coordinates) and curve_coordinates[index] == query:
        return curve[index][1]
    resolver = config.resolver
    if not isinstance(resolver, _ScatteredSites):
        raise TypeError("curve evaluation requires scattered-site configuration")
    if query < curve_coordinates[0] or query > curve_coordinates[-1] or len(curve) < 2:
        tail = curve[: resolver.k_tail] if query < curve_coordinates[0] else curve[-resolver.k_tail :]
        return _hold_curve_utilization(
            config,
            tail,
            query,
            axes,
            curve_position,
            site_positions,
            site_values,
            coordinates,
        )
    (low_coordinate, low_latency), (high_coordinate, high_latency) = curve[index - 1], curve[index]
    weight = (query - low_coordinate) / (high_coordinate - low_coordinate)
    return _from_space(
        config.value_transform,
        _to_space(config.value_transform, low_latency)
        + (_to_space(config.value_transform, high_latency) - _to_space(config.value_transform, low_latency))
        * weight,
    )


def _resolve_scattered(config: _InterpConfig, data: Mapping[Any, Any], coordinates: Sequence[Any]) -> float:
    resolver = config.resolver
    if not isinstance(resolver, _ScatteredSites):
        raise TypeError("scattered resolver requires a scattered-site configuration")
    axes = len(config.axes)
    curve_position = config.axes.index(resolver.curve_axis)
    site_positions = tuple(config.axes.index(axis) for axis in resolver.site_axes)
    leaves: list[tuple[tuple[Any, ...], float]] = []
    _walk_leaves(data, 0, axes, [], leaves)
    sites: dict[tuple[Any, ...], list[tuple[Any, float]]] = {}
    for leaf_coordinates, latency in leaves:
        site = tuple(leaf_coordinates[position] for position in site_positions)
        sites.setdefault(site, []).append((leaf_coordinates[curve_position], latency))
    for curve in sites.values():
        curve.sort(key=lambda item: item[0])
    site_keys = list(sites)
    site_logs = [tuple(math.log2(max(float(value), 1e-12)) for value in site) for site in site_keys]
    query_site = tuple(coordinates[position] for position in site_positions)
    curve_query = coordinates[curve_position]
    if query_site in sites:
        return _evaluate_curve(
            config,
            sites[query_site],
            curve_query,
            axes,
            curve_position,
            site_positions,
            query_site,
            coordinates,
        )
    if not site_keys:
        raise _miss(config, coordinates, "no sites collected")
    query_log = tuple(math.log2(max(float(value), 1e-12)) for value in query_site)

    def distance(index: int) -> float:
        return math.sqrt(
            sum(
                (source - target) ** 2
                for source, target in zip(site_logs[index], query_log, strict=True)
            )
        )

    candidates = list(range(len(site_keys)))
    if resolver.require_curve_coverage:
        covering = [
            index
            for index in candidates
            if sites[site_keys[index]][0][0] <= curve_query <= sites[site_keys[index]][-1][0]
        ]
        if covering:
            candidates = covering
    ranked = sorted(candidates, key=distance)
    if resolver.max_site_distance is not None:
        ranked = [index for index in ranked if distance(index) <= resolver.max_site_distance]
        if not ranked:
            raise _miss(config, coordinates, "no site within max_site_distance")

    weight_sum = 0.0
    utilization_sum = 0.0
    for index in ranked[: resolver.nn_sites]:
        neighbor = site_keys[index]
        try:
            latency = _evaluate_curve(
                config,
                sites[neighbor],
                curve_query,
                axes,
                curve_position,
                site_positions,
                neighbor,
                coordinates,
            )
        except InterpolationDataNotAvailableError:
            continue
        sol_neighbor = config.sol_fn(
            *_full_coordinates(
                axes,
                curve_position,
                site_positions,
                curve_query,
                neighbor,
            )
        )
        if not (
            math.isfinite(latency)
            and latency > 0.0
            and math.isfinite(sol_neighbor)
            and sol_neighbor > 0.0
        ):
            continue
        weight = 1.0 / (distance(index) ** 2 + 1e-12)
        utilization_sum += weight * (sol_neighbor / latency)
        weight_sum += weight
    if weight_sum <= 0.0:
        raise _miss(config, coordinates, "no usable neighbor site")
    sol_query = config.sol_fn(*coordinates)
    if sol_query <= 0.0:
        raise _miss(config, coordinates, "non-positive speed-of-light value at query")
    return sol_query / (utilization_sum / weight_sum)


def _grid_interior(
    config: _InterpConfig,
    node: Mapping[Any, Any] | float,
    coordinates: Sequence[Any],
    depth: int,
) -> float:
    if depth == len(config.axes):
        return float(node)
    if not isinstance(node, Mapping) or not node:
        raise _miss(config, coordinates, f"empty branch at axis {config.axes[depth]!r}")
    coordinate = coordinates[depth]
    if coordinate in node:
        return _grid_interior(config, node[coordinate], coordinates, depth + 1)
    keys = sorted(node)
    if coordinate < keys[0] or coordinate > keys[-1]:
        raise _OutOfRangeError
    index = bisect.bisect_left(keys, coordinate)
    low_key, high_key = keys[index - 1], keys[index]
    results: list[tuple[Any, float]] = []
    errors: list[Exception] = []
    for key in (low_key, high_key):
        try:
            results.append((key, _grid_interior(config, node[key], coordinates, depth + 1)))
        except (_OutOfRangeError, InterpolationDataNotAvailableError) as error:
            errors.append(error)
    if not results:
        if any(isinstance(error, _OutOfRangeError) for error in errors):
            raise _OutOfRangeError
        raise _miss(config, coordinates, f"no usable branch at axis {config.axes[depth]!r}")
    if len(results) == 1:
        surviving_key, latency = results[0]
        snapped = tuple(
            surviving_key if position == depth else coordinates[position]
            for position in range(len(coordinates))
        )
        sol_query = config.sol_fn(*coordinates)
        sol_surviving = config.sol_fn(*snapped)
        if (
            math.isfinite(sol_query)
            and math.isfinite(sol_surviving)
            and sol_query > 0.0
            and sol_surviving > 0.0
        ):
            return latency * (sol_query / sol_surviving)
        return latency
    low_latency = results[0][1]
    high_latency = results[1][1]
    weight = (coordinate - low_key) / (high_key - low_key)
    transform = config.value_transform
    if config.transform_axis is not None and config.axes[depth] != config.transform_axis:
        transform = _ValueTransform.RAW
    return _from_space(
        transform,
        _to_space(transform, low_latency)
        + (_to_space(transform, high_latency) - _to_space(transform, low_latency)) * weight,
    )


def _grid_hold(config: _InterpConfig, data: Mapping[Any, Any], coordinates: Sequence[Any]) -> float:
    node: Mapping[Any, Any] = data
    snapped: list[Any] = []
    for depth in range(len(config.axes) - 1):
        if not node:
            raise _miss(config, coordinates, f"empty branch at axis {config.axes[depth]!r}")
        coordinate = coordinates[depth]
        key = coordinate if coordinate in node else min(node, key=lambda candidate: abs(candidate - coordinate))
        snapped.append(key)
        child = node[key]
        if not isinstance(child, Mapping):
            raise _miss(config, coordinates, f"malformed branch at axis {config.axes[depth]!r}")
        node = child
    if not node:
        raise _miss(config, coordinates, f"empty branch at axis {config.axes[-1]!r}")
    keys = sorted(node)
    coordinate = coordinates[-1]
    resolver = config.resolver
    if not isinstance(resolver, _Grid):
        raise TypeError("grid hold requires grid configuration")
    if coordinate > keys[-1]:
        tail = keys[-resolver.k_tail :]
    elif coordinate < keys[0]:
        tail = keys[: resolver.k_tail]
    else:
        tail = [min(keys, key=lambda candidate: abs(candidate - coordinate))]
    utilizations = []
    for key in tail:
        latency = float(node[key])
        sol = config.sol_fn(*snapped, key)
        if latency > 0.0 and sol > 0.0:
            utilizations.append(sol / latency)
    if not utilizations:
        raise _miss(config, coordinates, "no positive-utilization boundary anchor")
    sol_query = config.sol_fn(*coordinates)
    if sol_query <= 0.0:
        raise _miss(config, coordinates, "non-positive speed-of-light value at query")
    return sol_query / statistics.median(utilizations)


def _interpolate(config: _InterpConfig, data: Mapping[Any, Any], *coordinates: Any) -> float:
    if len(coordinates) != len(config.axes):
        raise ValueError(
            f"query has {len(coordinates)} coordinates but table axes are {config.axes}"
        )
    if not data:
        raise _miss(config, coordinates, "empty table")
    node: Any = data
    for coordinate in coordinates:
        if not isinstance(node, Mapping) or coordinate not in node:
            node = _MISSING
            break
        node = node[coordinate]
    if node is not _MISSING:
        return float(node)
    if isinstance(config.resolver, _ScatteredSites):
        return _resolve_scattered(config, data, coordinates)
    try:
        return _grid_interior(config, data, coordinates, 0)
    except _OutOfRangeError:
        return _grid_hold(config, data, coordinates)


def _nested_insert(root: dict[Any, Any], key: Sequence[Any], value: float, *, replace: bool) -> None:
    node = root
    for component in key[:-1]:
        node = node.setdefault(component, {})
    if replace or key[-1] not in node:
        node[key[-1]] = value


def _freeze_key(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze_key(item) for item in value)
    return value


class ExternalOperationDatabase:
    """Read-only measured database with exact external resolver semantics."""

    def __init__(
        self,
        *,
        artifact_dir: Path,
        manifest: Mapping[str, Any],
        system_spec: Mapping[str, Any],
        model_config: Mapping[str, Any],
        family_mapping: Mapping[str, Any],
        rows: Sequence[_Row],
    ) -> None:
        self.artifact_dir = artifact_dir
        self.manifest = dict(manifest)
        self.system_spec = dict(system_spec)
        self.model_config = dict(model_config)
        self.family_mapping = dict(family_mapping)
        self.source = ExternalSourceIdentity.from_manifest(manifest)
        self._rows = tuple(rows)
        self._tables: dict[str, dict[Any, Any]] = defaultdict(dict)
        self._raw_index: dict[tuple[str, tuple[Any, ...]], float] = {}
        for row in self._rows:
            key = tuple(_freeze_key(component) for component in row.key)
            self._raw_index.setdefault((row.table, key), row.raw_latency)
            if row.table == "generation_dsa_module":
                backends = key[3]
                for backend in backends:
                    expanded = key[:3] + (backend,) + key[4:]
                    _nested_insert(
                        self._tables[row.table],
                        expanded,
                        row.served_latency,
                        replace=True,
                    )
            else:
                _nested_insert(
                    self._tables[row.table],
                    key,
                    row.served_latency,
                    replace=False,
                )

    @classmethod
    def load(cls, artifact_dir: str | os.PathLike[str] | None = None) -> ExternalOperationDatabase:
        artifact = Path(artifact_dir) if artifact_dir is not None else default_artifact_dir()
        if not artifact.is_dir():
            raise ExternalDatabaseIdentityError(f"external database artifact is missing: {artifact}")
        manifest_path = artifact / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise ExternalDatabaseIdentityError("external database manifest is missing or invalid") from error
        if manifest.get("schema") != EXTERNAL_DATABASE_SCHEMA:
            raise ExternalDatabaseIdentityError(
                f"unsupported external database schema {manifest.get('schema')!r}"
            )
        source = ExternalSourceIdentity.from_manifest(manifest)
        expected_source = ExternalSourceIdentity(
            tool="NVIDIA AIConfigurator",
            aiconfigurator_version=EXPECTED_AICONFIGURATOR_VERSION,
            core_version=EXPECTED_CORE_VERSION,
            system=EXPECTED_SYSTEM,
            backend=EXPECTED_BACKEND,
            database_version=EXPECTED_DATABASE_VERSION,
            slice_hash=EXPECTED_SLICE_HASH,
        )
        if source != expected_source:
            raise ExternalDatabaseIdentityError(
                f"artifact source identity differs from the frozen source: {source.as_dict()}"
            )
        conversion = _require_mapping(manifest.get("conversion"), "manifest.conversion")
        _validate_conversion_recipe(conversion)
        payload_name = _require_string(conversion.get("payload"), "manifest.conversion.payload")
        payload_hash = _require_string(
            conversion.get("payload_sha256"),
            "manifest.conversion.payload_sha256",
        )
        if payload_name != f"rows-{payload_hash}.jsonl.xz":
            raise ExternalDatabaseIdentityError("payload filename is not content-addressed by its declared hash")
        payload_path = artifact / payload_name
        if _sha256_file(payload_path) != payload_hash:
            raise ExternalDatabaseIdentityError("external database payload hash mismatch")
        rows = []
        with lzma.open(payload_path, "rt", encoding="ascii", newline="\n") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                    schema, table, version, key, raw_hex, served_hex = record
                except (ValueError, TypeError) as error:
                    raise ExternalDatabaseIdentityError(
                        f"invalid converted row at payload line {line_number}"
                    ) from error
                if schema != EXTERNAL_DATABASE_ROW_SCHEMA:
                    raise ExternalDatabaseIdentityError(
                        f"invalid row schema at payload line {line_number}: {schema!r}"
                    )
                if version != EXPECTED_DATABASE_VERSION:
                    raise ExternalDatabaseIdentityError(
                        f"donor database version at payload line {line_number}: {version!r}"
                    )
                try:
                    raw_latency = float.fromhex(raw_hex)
                    served_latency = float.fromhex(served_hex)
                except (TypeError, ValueError) as error:
                    raise ExternalDatabaseIdentityError(
                        f"invalid binary64 latency at payload line {line_number}"
                    ) from error
                rows.append(
                    _Row(
                        table=str(table),
                        version=str(version),
                        key=tuple(key),
                        raw_latency=raw_latency,
                        served_latency=served_latency,
                    )
                )
        expected_rows = conversion.get("rows")
        if expected_rows != EXPECTED_ROW_COUNT or len(rows) != EXPECTED_ROW_COUNT:
            raise ExternalDatabaseIdentityError(
                f"artifact row count mismatch: manifest={expected_rows}, payload={len(rows)}, "
                f"expected={EXPECTED_ROW_COUNT}"
            )
        source_manifest = (artifact / "source-files.sha256").read_bytes()
        if _sha256_bytes(source_manifest) != EXPECTED_SLICE_HASH:
            raise ExternalDatabaseIdentityError("stored source hash recipe does not reproduce the frozen slice hash")
        converted_hashes = _require_mapping(
            manifest.get("converted_files_sha256"),
            "manifest.converted_files_sha256",
        )
        converted_names = {
            "system.json",
            "model-config.json",
            "family-mapping.json",
        }
        if set(converted_hashes) != converted_names:
            raise ExternalDatabaseIdentityError(
                "converted file hash bindings must cover system, model and family mapping exactly"
            )
        for filename in sorted(converted_names):
            expected_hash = _require_string(
                converted_hashes.get(filename),
                f"manifest.converted_files_sha256.{filename}",
            )
            if _sha256_file(artifact / filename) != expected_hash:
                raise ExternalDatabaseIdentityError(
                    f"converted artifact hash mismatch for {filename}"
                )
        system_spec = json.loads((artifact / "system.json").read_text(encoding="utf-8"))
        model_config = json.loads((artifact / "model-config.json").read_text(encoding="utf-8"))
        family_mapping = json.loads((artifact / "family-mapping.json").read_text(encoding="utf-8"))
        if converted_hashes["family-mapping.json"] != manifest.get("family_mapping_sha256"):
            raise ExternalDatabaseIdentityError("external family mapping hash mismatch")
        if system_spec.get("notice") != SYSTEM_CONVERSION_NOTICE:
            raise ExternalDatabaseIdentityError("system.json conversion notice mismatch")
        if model_config.get("notice") != MODEL_CONVERSION_NOTICE:
            raise ExternalDatabaseIdentityError("model-config.json conversion notice mismatch")
        if family_mapping.get("schema") != EXTERNAL_FAMILY_MAPPING_SCHEMA:
            raise ExternalDatabaseIdentityError("external family mapping schema mismatch")
        if family_mapping.get("source_slice_sha256") != EXPECTED_SLICE_HASH:
            raise ExternalDatabaseIdentityError("external family mapping is not bound to the frozen slice")
        return cls(
            artifact_dir=artifact,
            manifest=manifest,
            system_spec=system_spec,
            model_config=model_config,
            family_mapping=family_mapping,
            rows=rows,
        )

    @property
    def row_count(self) -> int:
        return len(self._rows)

    @property
    def row_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for row in self._rows:
            counts[row.table] += 1
        return dict(counts)

    @property
    def payload_sha256(self) -> str:
        return str(self.manifest["conversion"]["payload_sha256"])

    def versions(self) -> frozenset[str]:
        return frozenset(row.version for row in self._rows)

    def load_time_mutations(self, table: str | None = None) -> tuple[Mapping[str, Any], ...]:
        """Return imported cells whose served latency was raised at load time."""

        return tuple(
            {
                "table": row.table,
                "key": row.key,
                "raw_hex": row.raw_latency.hex(),
                "served_hex": row.served_latency.hex(),
            }
            for row in self._rows
            if row.raw_latency != row.served_latency and (table is None or row.table == table)
        )

    def raw_latency(self, table: str, key: Sequence[Any]) -> float:
        frozen = tuple(_freeze_key(component) for component in key)
        try:
            return self._raw_index[(table, frozen)]
        except KeyError as error:
            raise ExternalDatabaseGapError(
                f"raw external row is unavailable for table={table!r}, key={frozen!r}"
            ) from error

    def served_latency(self, table: str, key: Sequence[Any]) -> float:
        """Return one exact table cell after the source loader's mutations."""

        frozen = tuple(_freeze_key(component) for component in key)
        try:
            node: Any = self._tables[table]
            for component in frozen:
                node = node[component]
        except KeyError as error:
            raise ExternalDatabaseGapError(
                f"served external row is unavailable for table={table!r}, key={frozen!r}"
            ) from error
        if isinstance(node, Mapping):
            raise ExternalDatabaseGapError(
                f"served external row key is incomplete for table={table!r}, key={frozen!r}"
            )
        return float(node)

    def _slice(self, table: str, *categorical: Any) -> Mapping[Any, Any]:
        try:
            node: Any = self._tables[table]
            for value in categorical:
                node = node[value]
        except KeyError as error:
            raise ExternalDatabaseGapError(
                f"external table gap: table={table!r}, categorical_key={categorical!r}"
            ) from error
        if not isinstance(node, Mapping) or not node:
            raise ExternalDatabaseGapError(
                f"external table has no measured values: table={table!r}, categorical_key={categorical!r}"
            )
        return node

    def _result(self, latency: float, operation: str, rule: str) -> ExternalLatency:
        return ExternalLatency(
            latency_ms=float(latency),
            source=self.source,
            operation=operation,
            rule=rule,
            evidence_class=EXTERNAL_EVIDENCE_CLASS,
        )

    def _query_gemm_with_cap(
        self,
        *,
        m: int,
        n: int,
        k: int,
        quant_mode: str,
        max_site_distance: float | None,
    ) -> ExternalLatency:
        data = self._slice("gemm", quant_mode)
        config = _InterpConfig(
            axes=("m", "n", "k"),
            resolver=_ScatteredSites(
                site_axes=("n", "k"),
                curve_axis="m",
                max_site_distance=max_site_distance,
            ),
            sol_fn=lambda m_value, n_value, k_value: _gemm_sol_ms(
                self.system_spec,
                quant_mode,
                int(m_value),
                int(n_value),
                int(k_value),
            ),
        )
        return self._result(_interpolate(config, data, m, n, k), "gemm", "scattered-site")

    def query_gemm(self, *, m: int, n: int, k: int, quant_mode: str) -> ExternalLatency:
        return self._query_gemm_with_cap(
            m=m,
            n=n,
            k=k,
            quant_mode=quant_mode,
            max_site_distance=2.0,
        )

    def query_gemm_cap_off_diagnostic(
        self,
        *,
        m: int,
        n: int,
        k: int,
        quant_mode: str,
    ) -> ExternalLatency:
        """Run the study-only GEMM resolver with its distance cap disabled."""

        return self._query_gemm_with_cap(
            m=m,
            n=n,
            k=k,
            quant_mode=quant_mode,
            max_site_distance=None,
        )

    def _quantize_query(
        self,
        table: str,
        *,
        m: int,
        k: int,
        quant_mode: str,
    ) -> ExternalLatency:
        data = self._slice(table, quant_mode)
        m_keys = sorted(data)
        m_clamped = max(m_keys[0], min(int(m), m_keys[-1]))
        k_min = min(min(row) for row in data.values() if row)
        k_max = max(max(row) for row in data.values() if row)
        k_clamped = max(k_min, min(int(k), k_max))
        multiplier = 2 if table == "compute_scale" else 3
        memory_bandwidth = float(self.system_spec["gpu"]["mem_bw"])
        sol = lambda m_value, k_value: multiplier * m_value * k_value / memory_bandwidth * 1000.0
        config = _InterpConfig(
            axes=("m", "k"),
            resolver=_Grid(),
            sol_fn=sol,
        )
        latency = _interpolate(config, data, m_clamped, k_clamped)
        if table == "scale_matrix" and (m_clamped != int(m) or k_clamped != int(k)):
            boundary_sol = sol(m_clamped, k_clamped)
            query_sol = sol(int(m), int(k))
            if boundary_sol <= 0.0:
                raise ExternalDatabaseGapError("scale-matrix clamped boundary has non-positive SOL")
            latency *= query_sol / boundary_sol
        return self._result(latency, table, "clamped-grid")

    def query_compute_scale(self, *, m: int, k: int, quant_mode: str) -> ExternalLatency:
        return self._quantize_query("compute_scale", m=m, k=k, quant_mode=quant_mode)

    def query_scale_matrix(self, *, m: int, k: int, quant_mode: str) -> ExternalLatency:
        return self._quantize_query("scale_matrix", m=m, k=k, quant_mode=quant_mode)

    def _context_attention_sol(
        self,
        *,
        b: int,
        s: int,
        prefix: int,
        n: int,
        n_kv: int,
        head_size: int,
        window_size: int,
        kv_quant_mode: str,
        fmha_quant_mode: str,
    ) -> float:
        full_sequence = s + prefix
        if window_size > 0 and full_sequence > window_size:
            operations = 2 * b * (full_sequence - prefix) * window_size * n * head_size * 2
        else:
            operations = 2 * b * (full_sequence * full_sequence - prefix * prefix) * n * head_size * 2 / 2
        mem_bytes = 2 * b * (
            n * (full_sequence - prefix) * head_size
            + n * (full_sequence - prefix) * head_size
        ) + _QUANT_MEMORY_BYTES[kv_quant_mode] * b * (2 * n_kv * full_sequence * head_size)
        gpu = self.system_spec["gpu"]
        sol_math = (
            operations
            / float(gpu["bfloat16_tc_flops"])
            * 1000
            / _QUANT_COMPUTE_FACTOR[fmha_quant_mode]
        )
        sol_mem = mem_bytes / float(gpu["mem_bw"]) * 1000
        return max(sol_math, sol_mem)

    def query_context_attention(
        self,
        *,
        b: int,
        s: int,
        prefix: int,
        n: int,
        n_kv: int,
        kv_quant_mode: str,
        fmha_quant_mode: str,
        window_size: int = 0,
        head_size: int = 128,
    ) -> ExternalLatency:
        if n_kv > n:
            raise ValueError("n_kv must be no greater than n")
        full_sequence = s + prefix
        if full_sequence <= 0:
            raise ValueError("context full sequence must be positive")
        n_kv_key = 0 if n == n_kv else n_kv
        data = self._slice(
            "context_attention",
            fmha_quant_mode,
            kv_quant_mode,
            n_kv_key,
            head_size,
            window_size,
        )
        config = _InterpConfig(
            axes=("num_heads", "seq_len", "batch"),
            resolver=_Grid(),
            sol_fn=lambda n_value, s_value, b_value: self._context_attention_sol(
                b=int(b_value),
                s=int(s_value),
                prefix=0,
                n=int(n_value),
                n_kv=int(n_value) if n_kv_key == 0 else n_kv_key,
                head_size=head_size,
                window_size=window_size,
                kv_quant_mode=kv_quant_mode,
                fmha_quant_mode=fmha_quant_mode,
            ),
            value_transform=_ValueTransform.SQRT,
            transform_axis="seq_len",
        )
        latency = _interpolate(config, data, n, full_sequence, b)
        prefix_fraction = (full_sequence * full_sequence - prefix * prefix) / (
            full_sequence * full_sequence
        )
        return self._result(
            latency * prefix_fraction,
            "context_attention",
            "sqrt-sequence-grid-with-prefix",
        )

    def query_generation_attention(
        self,
        *,
        b: int,
        s: int,
        n: int,
        n_kv: int,
        kv_quant_mode: str,
        window_size: int = 0,
        head_size: int = 128,
        attn_dtype: str | None = None,
    ) -> ExternalLatency:
        del attn_dtype
        if n_kv > n:
            raise ValueError("n_kv must be no greater than n")
        n_kv_key = 0 if n == n_kv else n_kv
        data = self._slice(
            "generation_attention",
            kv_quant_mode,
            n_kv_key,
            head_size,
            window_size,
        )
        config = _InterpConfig(
            axes=("num_heads", "batch", "seq_len"),
            resolver=_Grid(),
            sol_fn=lambda n_value, b_value, s_value: _generation_attention_sol_ms(
                self.system_spec,
                batch=int(b_value),
                sequence=int(s_value),
                num_heads=int(n_value),
                num_kv_heads=int(n_value) if n_kv_key == 0 else n_kv_key,
                head_size=head_size,
                window_size=window_size,
                kv_quant=kv_quant_mode,
            ),
        )
        sequence_min = max(1, int(s * 0.9))
        sequence_max = max(sequence_min, int(s * 1.1))
        samples = [
            sequence_min + (sequence_max - sequence_min) * index // 4
            for index in range(5)
        ]
        latency_sum = 0.0
        for sample in samples:
            latency_sum += _interpolate(config, data, n, b, sample)
        return self._result(
            latency_sum / 5,
            "generation_attention",
            "five-point-smoothed-grid",
        )

    def _moe_sol(
        self,
        *,
        num_tokens: int,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        quant_mode: str,
        is_gated: bool,
    ) -> float:
        num_gemms = 3 if is_gated else 2
        total_tokens = num_tokens * topk
        operations = (
            total_tokens
            * hidden_size
            * inter_size
            * num_gemms
            * 2
            // moe_ep_size
            // moe_tp_size
        )
        memory = _QUANT_MEMORY_BYTES[quant_mode]
        mem_bytes = memory * (
            total_tokens // moe_ep_size * hidden_size * 2
            + total_tokens // moe_ep_size * inter_size * num_gemms // moe_tp_size
            + hidden_size
            * inter_size
            * num_gemms
            // moe_tp_size
            * min(num_experts // moe_ep_size, total_tokens // moe_ep_size)
        )
        gpu = self.system_spec["gpu"]
        sol_math = (
            operations
            / (float(gpu["bfloat16_tc_flops"]) * _QUANT_COMPUTE_FACTOR[quant_mode])
            * 1000
        )
        sol_mem = mem_bytes / float(gpu["mem_bw"]) * 1000
        return max(sol_math, sol_mem)

    def query_moe(
        self,
        *,
        num_tokens: int,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        quant_mode: str,
        workload_distribution: str,
        is_gated: bool = True,
        kernel_source: str | None = None,
    ) -> ExternalLatency:
        if kernel_source == "moe_torch_flow_min_latency":
            raise ExternalDatabaseGapError(
                "the low-latency standard MoE table has no dispatched resolver"
            )
        by_distribution = self._slice("moe", quant_mode)
        distribution = (
            workload_distribution
            if workload_distribution in by_distribution
            else "uniform"
        )
        try:
            data = self._slice(
                "moe",
                quant_mode,
                distribution,
                topk,
                num_experts,
                hidden_size,
                inter_size,
                moe_tp_size,
                moe_ep_size,
            )
        except ExternalDatabaseGapError as error:
            raise ExternalDatabaseGapError(
                "external MoE mapping has no declared measured curve for "
                f"quant={quant_mode}, distribution={distribution}, topk={topk}, "
                f"experts={num_experts}, hidden={hidden_size}, intermediate={inter_size}, "
                f"tp={moe_tp_size}, ep={moe_ep_size}"
            ) from error
        config = _InterpConfig(
            axes=("num_tokens",),
            resolver=_Grid(),
            sol_fn=lambda tokens: self._moe_sol(
                num_tokens=int(tokens),
                hidden_size=hidden_size,
                inter_size=inter_size,
                topk=topk,
                num_experts=num_experts,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
                quant_mode=quant_mode,
                is_gated=is_gated,
            ),
        )
        return self._result(
            _interpolate(config, data, num_tokens),
            "moe",
            "token-curve",
        )

    def _p2p_bandwidth(self, num_gpus: int) -> float:
        node = self.system_spec["node"]
        if num_gpus <= int(node["num_gpus_per_node"]):
            return float(node["intra_node_bw"])
        if num_gpus <= float(node.get("num_gpus_per_rack", float("inf"))):
            return float(node["inter_node_bw"])
        return float(node.get("inter_rack_bw", node["inter_node_bw"]))

    def _allreduce_sol(self, quant_mode: str, tp_size: int, size: int) -> float:
        del quant_mode
        if tp_size == 1:
            return 0.0
        bandwidth = self._p2p_bandwidth(tp_size)
        sol_time = 2 * size * 2 / tp_size * (tp_size - 1) / bandwidth
        return sol_time * 1000

    def query_custom_allreduce(
        self,
        *,
        quant_mode: str,
        tp_size: int,
        size: int,
    ) -> ExternalLatency:
        if quant_mode != "half":
            raise ExternalDatabaseGapError(
                "custom all-reduce imported rows are reachable only through the audited half rekey"
            )
        if tp_size <= 1:
            raise ExternalDatabaseGapError(
                "tp_size=1 is an analytical no-op, not a measured external database value"
            )
        per_node = int(self.system_spec["node"]["num_gpus_per_node"])
        effective_tp = min(tp_size, per_node)
        data = self._slice("custom_allreduce", quant_mode, effective_tp, "AUTO")
        config = _InterpConfig(
            axes=("message_bytes",),
            resolver=_Grid(),
            sol_fn=lambda message_bytes: self._allreduce_sol(
                quant_mode,
                effective_tp,
                int(message_bytes),
            ),
        )
        latency = _interpolate(config, data, size)
        if tp_size > per_node:
            base_bandwidth = self._p2p_bandwidth(per_node)
            target_bandwidth = self._p2p_bandwidth(tp_size)
            scale_factor = (
                (tp_size - 1)
                / tp_size
                * per_node
                / (per_node - 1)
                * base_bandwidth
                / target_bandwidth
            )
            latency *= scale_factor
        return self._result(latency, "custom_allreduce", "message-byte-curve")

    def _gdn_sol(
        self,
        *,
        phase: str,
        kernel_source: str,
        batch_size: int,
        seq_len: int | None,
        d_model: int,
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
        d_conv: int,
    ) -> float:
        tokens = batch_size * seq_len if phase == "context" and seq_len else batch_size
        if kernel_source in ("causal_conv1d_fn", "causal_conv1d_update"):
            channels = num_k_heads * head_k_dim + num_v_heads * head_v_dim
            read_bytes = tokens * channels * (d_conv + 1) * 2
            write_bytes = tokens * channels * 2
        elif kernel_source == "chunk_gated_delta_rule":
            state_size = num_v_heads * head_k_dim * head_v_dim
            chunks = seq_len // 64 if seq_len else 0
            chunk_bytes = chunks * state_size * 2 * batch_size
            read_bytes = (
                tokens * (num_k_heads * head_k_dim + num_v_heads * head_v_dim) * 2
                + state_size * 2 * batch_size
                + chunk_bytes
            )
            write_bytes = (
                tokens * num_v_heads * head_v_dim * 2
                + state_size * 2 * batch_size
                + chunk_bytes
            )
        elif kernel_source == "fused_sigmoid_gating_delta_rule_update":
            state_size = num_v_heads * head_k_dim * head_v_dim
            read_bytes = (
                tokens * (num_k_heads * head_k_dim + num_v_heads * head_v_dim) * 2
                + state_size * 2 * batch_size
            )
            write_bytes = tokens * num_v_heads * head_v_dim * 2 + state_size * 2 * batch_size
        else:
            read_bytes = tokens * d_model * 2
            write_bytes = tokens * d_model * 2
        return (read_bytes + write_bytes) / float(self.system_spec["gpu"]["mem_bw"]) * 1000

    def query_gdn(
        self,
        *,
        phase: str,
        kernel_source: str,
        batch_size: int,
        seq_len: int | None,
        d_model: int,
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
        d_conv: int,
        model_name: str | None = None,
        num_tokens: int | None = None,
    ) -> ExternalLatency:
        del model_name, num_tokens
        categorical = (
            kernel_source,
            phase,
            d_model,
            num_k_heads,
            head_k_dim,
            num_v_heads,
            head_v_dim,
            d_conv,
        )
        data = self._slice("gdn", *categorical)
        if phase == "context":
            if seq_len is None or seq_len <= 0:
                raise ValueError("context GDN requires a positive sequence length")
            config = _InterpConfig(
                axes=("batch", "seq_len"),
                resolver=_Grid(),
                sol_fn=lambda batch, sequence: self._gdn_sol(
                    phase=phase,
                    kernel_source=kernel_source,
                    batch_size=int(batch),
                    seq_len=int(sequence),
                    d_model=d_model,
                    num_k_heads=num_k_heads,
                    head_k_dim=head_k_dim,
                    num_v_heads=num_v_heads,
                    head_v_dim=head_v_dim,
                    d_conv=d_conv,
                ),
            )
            latency = _interpolate(config, data, batch_size, seq_len)
        else:
            config = _InterpConfig(
                axes=("batch",),
                resolver=_Grid(),
                sol_fn=lambda batch: self._gdn_sol(
                    phase=phase,
                    kernel_source=kernel_source,
                    batch_size=int(batch),
                    seq_len=seq_len,
                    d_model=d_model,
                    num_k_heads=num_k_heads,
                    head_k_dim=head_k_dim,
                    num_v_heads=num_v_heads,
                    head_v_dim=head_v_dim,
                    d_conv=d_conv,
                ),
            )
            latency = _interpolate(config, data, batch_size)
        return self._result(latency, "gdn", f"{phase}-grid")

    def query_memory_operation(self, mem_bytes: int) -> float:
        """Return the audited H200 analytical memory-operation latency in ms."""

        gpu = self.system_spec["gpu"]
        return (
            mem_bytes
            / (float(gpu["mem_bw"]) * float(gpu["mem_bw_empirical_scaling_factor"]))
            + float(gpu["mem_empirical_constant_latency"])
        ) * 1000

    def query_operation(self, operation: str, arguments: Mapping[str, Any]) -> ExternalLatency:
        dispatch = {
            "gemm": self.query_gemm,
            "compute_scale": self.query_compute_scale,
            "scale_matrix": self.query_scale_matrix,
            "context_attention": self.query_context_attention,
            "generation_attention": self.query_generation_attention,
            "moe": self.query_moe,
            "custom_allreduce": self.query_custom_allreduce,
            "gdn": self.query_gdn,
        }
        try:
            query = dispatch[operation]
        except KeyError as error:
            raise ExternalDatabaseGapError(
                f"operation {operation!r} has no declared external resolver"
            ) from error
        return query(**dict(arguments))

    def mapping_rule(self, family: str) -> Mapping[str, Any]:
        for rule in self.family_mapping["rules"]:
            if rule.get("family") == family:
                return rule
        raise ExternalDatabaseGapError(f"family {family!r} is absent from the declared mapping table")

    def require_mapping(self, family: str, *, composite: bool = False) -> Mapping[str, Any]:
        rule = self.mapping_rule(family)
        status = rule.get("status")
        if status == "gap":
            raise ExternalDatabaseGapError(
                f"family mapping {family!r} is a declared gap: {rule.get('declaration')}"
            )
        if status == "composite" and not composite:
            raise ExternalCompositeError(
                f"family mapping {family!r} requires an explicit composite declaration: "
                f"{rule.get('declaration')}"
            )
        if status == "exact" and composite:
            raise ExternalCompositeError(f"family mapping {family!r} is exact, not composite")
        return rule


@dataclass(frozen=True)
class ExternalPassResult:
    """One external context or generation pass with its ordered breakdown."""

    mode: str
    total: ExternalLatency
    operations: tuple[ExternalLatency, ...]

    def operation_latencies(self) -> dict[str, float]:
        """Return the ordered operation breakdown as ordinary millisecond values."""

        return {entry.operation: entry.latency_ms for entry in self.operations}


class ExternalQwen32BPassModel:
    """Audited TensorRT-LLM Python composition for Qwen3-32B-FP8."""

    _HIDDEN_SIZE = 5120
    _INTERMEDIATE_SIZE = 25600
    _VOCAB_SIZE = 151936
    _NUM_LAYERS = 64
    _NUM_HEADS = 64
    _NUM_KV_HEADS = 8
    _HEAD_SIZE = 128
    def __init__(
        self,
        database: ExternalOperationDatabase,
        *,
        tensor_parallel: int = 4,
        kv_cache_quant_mode: str = "bfloat16",
        fmha_quant_mode: str = "bfloat16",
        communication_quant_mode: str = "half",
    ) -> None:
        self.database = database
        if tensor_parallel not in {2, 4, 8}:
            raise ValueError("tensor_parallel must be one of 2, 4 or 8")
        if kv_cache_quant_mode not in {"bfloat16", "fp8"}:
            raise ValueError("kv_cache_quant_mode must be bfloat16 or fp8")
        if fmha_quant_mode not in {"bfloat16", "fp8"}:
            raise ValueError("fmha_quant_mode must be bfloat16 or fp8")
        if communication_quant_mode != "half":
            raise ValueError("communication_quant_mode must be half")
        self.tensor_parallel = tensor_parallel
        self.kv_cache_quant_mode = kv_cache_quant_mode
        self.fmha_quant_mode = fmha_quant_mode
        self.communication_quant_mode = communication_quant_mode
        expected_model = {
            "hidden_size": self._HIDDEN_SIZE,
            "intermediate_size": self._INTERMEDIATE_SIZE,
            "vocab_size": self._VOCAB_SIZE,
            "num_hidden_layers": self._NUM_LAYERS,
            "num_attention_heads": self._NUM_HEADS,
            "num_key_value_heads": self._NUM_KV_HEADS,
            "head_dim": self._HEAD_SIZE,
        }
        mismatches = {
            key: (database.model_config.get(key), value)
            for key, value in expected_model.items()
            if database.model_config.get(key) != value
        }
        if mismatches:
            raise ExternalDatabaseIdentityError(
                f"Qwen3-32B-FP8 model configuration differs from the frozen composition: {mismatches!r}"
            )

    def _result(self, latency: float, operation: str, rule: str) -> ExternalLatency:
        return self.database._result(latency, operation, rule)

    def _memory_operation(
        self,
        *,
        operation: str,
        mem_bytes: int,
        scale_factor: float,
    ) -> ExternalLatency:
        latency = self.database.query_memory_operation(mem_bytes) * scale_factor
        return self._result(latency, operation, "analytical-h200-empirical-memory")

    def _gemm(
        self,
        *,
        operation: str,
        m: int,
        n: int,
        k: int,
        quant_mode: str,
        scale_factor: float,
    ) -> ExternalLatency:
        base = self.database.query_gemm(m=m, n=n, k=k, quant_mode=quant_mode)
        return self._result(
            base.latency_ms * scale_factor,
            operation,
            "external-gemm-times-repeat-count",
        )

    def _allreduce(
        self,
        *,
        operation: str,
        tokens: int,
        scale_factor: float,
    ) -> ExternalLatency:
        base = self.database.query_custom_allreduce(
            quant_mode=self.communication_quant_mode,
            tp_size=self.tensor_parallel,
            size=tokens * self._HIDDEN_SIZE,
        )
        return self._result(
            base.latency_ms * scale_factor,
            operation,
            "external-half-allreduce-times-repeat-count",
        )

    def _context_attention(
        self,
        *,
        batch_size: int,
        sequence: int,
        prefix: int,
    ) -> ExternalLatency:
        num_heads = self._NUM_HEADS // self.tensor_parallel
        num_kv_heads = self._NUM_KV_HEADS // self.tensor_parallel
        result = self.database.query_context_attention(
            b=batch_size,
            s=sequence,
            prefix=prefix,
            n=num_heads,
            n_kv=num_kv_heads,
            kv_quant_mode=self.kv_cache_quant_mode,
            fmha_quant_mode=self.fmha_quant_mode,
            window_size=0,
            head_size=self._HEAD_SIZE,
        ).latency_ms

        query_elements = num_heads * self._HEAD_SIZE
        key_elements = num_kv_heads * self._HEAD_SIZE
        value_elements = num_kv_heads * self._HEAD_SIZE
        qk_norm_latency = (
            2 * self.database.query_memory_operation(query_elements * 2)
            + 2 * self.database.query_memory_operation(key_elements * 2)
        )
        extra_latency = qk_norm_latency * 2
        apply_rope_latency = 2 * self.database.query_memory_operation(
            query_elements * 2 + key_elements * 2
        )
        kv_element_bytes = 1 if self.fmha_quant_mode == "fp8" else 2
        kv_write_latency = self.database.query_memory_operation(
            key_elements * kv_element_bytes
        ) + self.database.query_memory_operation(value_elements * kv_element_bytes)
        extra_latency += apply_rope_latency + kv_write_latency
        result += extra_latency * 1.1
        return self._result(
            result * self._NUM_LAYERS,
            "context_attention",
            "external-context-attention-plus-qk-norm-rope-kv-write",
        )

    def _generation_attention(self, *, batch_size: int, sequence: int) -> ExternalLatency:
        base = self.database.query_generation_attention(
            b=batch_size,
            s=sequence,
            n=self._NUM_HEADS // self.tensor_parallel,
            n_kv=self._NUM_KV_HEADS // self.tensor_parallel,
            kv_quant_mode=self.kv_cache_quant_mode,
            window_size=0,
            head_size=self._HEAD_SIZE,
        )
        return self._result(
            base.latency_ms * self._NUM_LAYERS,
            "generation_attention",
            "external-generation-attention-times-repeat-count",
        )

    def _context_operations(
        self,
        *,
        batch_size: int,
        effective_isl: int,
        prefix: int,
    ) -> tuple[ExternalLatency, ...]:
        tokens = batch_size * effective_isl
        vocab_per_rank = self._VOCAB_SIZE // self.tensor_parallel
        intermediate_per_rank = self._INTERMEDIATE_SIZE // self.tensor_parallel
        qkv_width = (
            self._NUM_HEADS * self._HEAD_SIZE // self.tensor_parallel
            + self._HEAD_SIZE * (self._NUM_KV_HEADS // self.tensor_parallel) * 2
        )
        return (
            self._memory_operation(
                operation="context_embedding",
                mem_bytes=tokens * self._HIDDEN_SIZE * 2,
                scale_factor=1,
            ),
            self._memory_operation(
                operation="context_add_norm_1",
                mem_bytes=tokens * (2 * self._HIDDEN_SIZE + 2 * self._HIDDEN_SIZE) * 2,
                scale_factor=self._NUM_LAYERS,
            ),
            self._gemm(
                operation="context_qkv_gemm",
                m=tokens,
                n=qkv_width,
                k=self._HIDDEN_SIZE,
                quant_mode="fp8_block",
                scale_factor=self._NUM_LAYERS,
            ),
            self._context_attention(
                batch_size=batch_size,
                sequence=effective_isl,
                prefix=prefix,
            ),
            self._gemm(
                operation="context_proj_gemm",
                m=tokens,
                n=self._HIDDEN_SIZE,
                k=self._NUM_HEADS * self._HEAD_SIZE // self.tensor_parallel,
                quant_mode="fp8_block",
                scale_factor=self._NUM_LAYERS,
            ),
            self._memory_operation(
                operation="context_add_norm_2",
                mem_bytes=tokens * (2 * self._HIDDEN_SIZE + 2 * self._HIDDEN_SIZE) * 2,
                scale_factor=self._NUM_LAYERS,
            ),
            self._gemm(
                operation="context_gate_ffn1_gemm",
                m=tokens,
                n=2 * intermediate_per_rank,
                k=self._HIDDEN_SIZE,
                quant_mode="fp8_block",
                scale_factor=self._NUM_LAYERS,
            ),
            self._memory_operation(
                operation="context_act_gate",
                mem_bytes=tokens * (2 * intermediate_per_rank + intermediate_per_rank) * 2,
                scale_factor=self._NUM_LAYERS,
            ),
            self._gemm(
                operation="context_ffn2_gemm",
                m=tokens,
                n=self._HIDDEN_SIZE,
                k=intermediate_per_rank,
                quant_mode="fp8_block",
                scale_factor=self._NUM_LAYERS,
            ),
            self._gemm(
                operation="context_logits_gemm",
                m=batch_size,
                n=vocab_per_rank,
                k=self._HIDDEN_SIZE,
                quant_mode="bfloat16",
                scale_factor=1,
            ),
            self._allreduce(operation="context_embedding_ar", tokens=tokens, scale_factor=1),
            self._allreduce(
                operation="context_ar_1",
                tokens=tokens,
                scale_factor=self._NUM_LAYERS,
            ),
            self._allreduce(
                operation="context_ar_2",
                tokens=tokens,
                scale_factor=self._NUM_LAYERS,
            ),
            self._result(0.0, "context_p2p", "pipeline-width-one-no-op"),
        )

    def _generation_operations(
        self,
        *,
        batch_size: int,
        sequence: int,
    ) -> tuple[ExternalLatency, ...]:
        tokens = batch_size
        vocab_per_rank = self._VOCAB_SIZE // self.tensor_parallel
        intermediate_per_rank = self._INTERMEDIATE_SIZE // self.tensor_parallel
        qkv_width = (
            self._NUM_HEADS * self._HEAD_SIZE // self.tensor_parallel
            + self._HEAD_SIZE * (self._NUM_KV_HEADS // self.tensor_parallel) * 2
        )
        return (
            self._memory_operation(
                operation="generation_embedding",
                mem_bytes=tokens * self._HIDDEN_SIZE * 2,
                scale_factor=1.0,
            ),
            self._memory_operation(
                operation="generation_add_norm_1",
                mem_bytes=tokens * (2 * self._HIDDEN_SIZE + 2 * self._HIDDEN_SIZE) * 2,
                scale_factor=float(self._NUM_LAYERS),
            ),
            self._gemm(
                operation="generation_qkv_gemm",
                m=tokens,
                n=qkv_width,
                k=self._HIDDEN_SIZE,
                quant_mode="fp8_block",
                scale_factor=float(self._NUM_LAYERS),
            ),
            self._generation_attention(batch_size=batch_size, sequence=sequence),
            self._gemm(
                operation="generation_proj_gemm",
                m=tokens,
                n=self._HIDDEN_SIZE,
                k=self._NUM_HEADS * self._HEAD_SIZE // self.tensor_parallel,
                quant_mode="fp8_block",
                scale_factor=float(self._NUM_LAYERS),
            ),
            self._memory_operation(
                operation="generation_add_norm_2",
                mem_bytes=tokens * (2 * self._HIDDEN_SIZE + 2 * self._HIDDEN_SIZE) * 2,
                scale_factor=float(self._NUM_LAYERS),
            ),
            self._gemm(
                operation="generation_gate_ffn1_gemm",
                m=tokens,
                n=2 * intermediate_per_rank,
                k=self._HIDDEN_SIZE,
                quant_mode="fp8_block",
                scale_factor=float(self._NUM_LAYERS),
            ),
            self._memory_operation(
                operation="generation_act_gate",
                mem_bytes=tokens * (2 * intermediate_per_rank + intermediate_per_rank) * 2,
                scale_factor=float(self._NUM_LAYERS),
            ),
            self._gemm(
                operation="generation_ffn2_gemm",
                m=tokens,
                n=self._HIDDEN_SIZE,
                k=intermediate_per_rank,
                quant_mode="fp8_block",
                scale_factor=float(self._NUM_LAYERS),
            ),
            self._gemm(
                operation="generation_logits_gemm",
                m=tokens,
                n=vocab_per_rank,
                k=self._HIDDEN_SIZE,
                quant_mode="bfloat16",
                scale_factor=1.0,
            ),
            self._allreduce(operation="generation_embedding_ar", tokens=tokens, scale_factor=1.0),
            self._allreduce(
                operation="generation_ar_1",
                tokens=tokens,
                scale_factor=float(self._NUM_LAYERS),
            ),
            self._allreduce(
                operation="generation_ar_2",
                tokens=tokens,
                scale_factor=float(self._NUM_LAYERS),
            ),
            self._result(0.0, "generation_p2p", "pipeline-width-one-no-op"),
        )

    def run_context(
        self,
        *,
        batch_size: int,
        isl: int,
        prefix: int = 0,
        latency_correction_scale: float = 1.0,
    ) -> ExternalPassResult:
        """Evaluate one frozen static-context pass."""

        effective_isl = isl - prefix
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if effective_isl <= 0:
            raise ValueError("isl must remain positive after removing prefix")
        if not math.isfinite(latency_correction_scale) or latency_correction_scale <= 0:
            raise ValueError("latency_correction_scale must be finite and positive")
        raw_operations = self._context_operations(
            batch_size=batch_size,
            effective_isl=effective_isl,
            prefix=prefix,
        )
        operations = (
            raw_operations
            if latency_correction_scale == 1.0
            else tuple(
                self._result(
                    entry.latency_ms * latency_correction_scale,
                    entry.operation,
                    f"{entry.rule};latency-correction",
                )
                for entry in raw_operations
            )
        )
        total = math.fsum(entry.latency_ms for entry in operations)
        return ExternalPassResult(
            mode="static_ctx",
            total=self._result(total, "qwen3_32b_fp8_context_pass", "ordered-python-phase-sum"),
            operations=operations,
        )

    def run_generation(
        self,
        *,
        batch_size: int,
        isl: int,
        osl: int,
        stride: int = 32,
        beam_width: int = 1,
        latency_correction_scale: float = 1.0,
    ) -> ExternalPassResult:
        """Evaluate the frozen sampled static-generation pass."""

        if batch_size <= 0 or isl <= 0:
            raise ValueError("batch_size and isl must be positive")
        if osl <= 1:
            raise ValueError("generation requires osl greater than one")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if beam_width != 1:
            raise ValueError("the frozen generation composition supports beam_width=1 only")
        if not math.isfinite(latency_correction_scale) or latency_correction_scale <= 0:
            raise ValueError("latency_correction_scale must be finite and positive")

        names: list[str] = []
        totals: dict[str, float] = {}
        rules: dict[str, str] = {}
        for index in range(0, osl - 1, stride):
            sampled = self._generation_operations(
                batch_size=batch_size,
                sequence=isl + index + 1,
            )
            repeat_count = min(stride, osl - 1 - index)
            for entry in sampled:
                if entry.operation not in totals:
                    names.append(entry.operation)
                    totals[entry.operation] = 0.0
                    rules[entry.operation] = entry.rule
                totals[entry.operation] += entry.latency_ms * repeat_count
        correction_suffix = ";latency-correction" if latency_correction_scale != 1.0 else ""
        operations = tuple(
            self._result(
                totals[name] * latency_correction_scale,
                name,
                f"{rules[name]};stride-repeat{correction_suffix}",
            )
            for name in names
        )
        total = math.fsum(entry.latency_ms for entry in operations)
        return ExternalPassResult(
            mode="static_gen",
            total=self._result(total, "qwen3_32b_fp8_generation_pass", "ordered-python-phase-sum"),
            operations=operations,
        )


class ExternalCompositionLedger:
    """Reject overlap between a fused composite and its constituent claims."""

    def __init__(self, database: ExternalOperationDatabase) -> None:
        self._database = database
        self._claimed: set[str] = set()

    def claim(self, family_or_constituent: str, *, composite: bool = False) -> None:
        try:
            rule = self._database.mapping_rule(family_or_constituent)
        except ExternalDatabaseGapError:
            rule = None
        if rule is not None:
            self._database.require_mapping(family_or_constituent, composite=composite)
            constituents = set(rule.get("constituents", ()))
            conflicts = constituents & self._claimed
            if conflicts:
                raise ExternalCompositeError(
                    f"composite {family_or_constituent!r} overlaps already claimed constituents "
                    f"{sorted(conflicts)!r}"
                )
            if family_or_constituent in self._claimed:
                raise ExternalCompositeError(f"family {family_or_constituent!r} was already claimed")
            self._claimed.add(family_or_constituent)
            self._claimed.update(constituents)
            return
        owners = []
        for candidate in self._database.family_mapping["rules"]:
            if family_or_constituent in candidate.get("constituents", ()):
                owners.append(str(candidate["family"]))
        if not owners:
            raise ExternalDatabaseGapError(
                f"undeclared family or composite constituent {family_or_constituent!r}"
            )
        conflicts = set(owners) & self._claimed
        if conflicts or family_or_constituent in self._claimed:
            raise ExternalCompositeError(
                f"constituent {family_or_constituent!r} overlaps claimed composite {sorted(conflicts)!r}"
            )
        self._claimed.add(family_or_constituent)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-external", action="store_true")
    parser.add_argument("--worker-import", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    if args.worker_import:
        if args.output_root is None:
            raise SystemExit("--worker-import requires --output-root")
        artifact = _write_worker_artifact(args.output_root)
        print(artifact)
        return 0
    if args.import_external:
        artifact = import_external_database(venv_root=args.venv, output_root=args.output_root)
        print(artifact)
        return 0
    raise SystemExit("select --import-external")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_RELATIVE_ROOT",
    "EXPECTED_SLICE_HASH",
    "EXTERNAL_DATABASE_SCHEMA",
    "EXTERNAL_EVIDENCE_CLASS",
    "ExternalCompositeError",
    "ExternalCompositionLedger",
    "ExternalDatabaseError",
    "ExternalDatabaseGapError",
    "ExternalDatabaseIdentityError",
    "ExternalLatency",
    "ExternalOperationDatabase",
    "ExternalPassResult",
    "ExternalQwen32BPassModel",
    "ExternalSourceIdentity",
    "default_artifact_dir",
    "external_artifact_licensing_findings",
    "import_external_database",
]
