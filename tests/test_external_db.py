# SPDX-FileCopyrightText: Copyright (c) 2026 Yifeng Wang
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import lzma
import shutil
from pathlib import Path

import pytest

from simllm.calibration.external_db import (
    EXPECTED_APACHE_LICENSE_HASH,
    EXPECTED_SLICE_HASH,
    EXTERNAL_EVIDENCE_CLASS,
    MODEL_CONVERSION_NOTICE,
    SYSTEM_CONVERSION_NOTICE,
    ExternalCompositeError,
    ExternalCompositionLedger,
    ExternalDatabaseGapError,
    ExternalDatabaseIdentityError,
    ExternalLatency,
    ExternalOperationDatabase,
    ExternalQwen32BPassModel,
    default_artifact_dir,
    external_artifact_licensing_findings,
    import_external_database,
)

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_COUNTS = {
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


@pytest.fixture(scope="module")
def database() -> ExternalOperationDatabase:
    return ExternalOperationDatabase.load()


def test_artifact_identity_and_inventory(database: ExternalOperationDatabase) -> None:
    assert database.artifact_dir == default_artifact_dir()
    assert database.source.slice_hash == EXPECTED_SLICE_HASH
    assert database.source.database_mode == "SILICON"
    assert database.source.shared_layer is False
    assert database.source.estimator_surface == "python"
    assert database.row_count == 284717
    assert database.row_counts == EXPECTED_COUNTS
    assert database.versions() == frozenset({"1.3.0rc10"})
    assert database.payload_sha256 == "0f606718c5e413e898d9ad33a3d7e803e532d5d5bd8a1c29929e7f8b2458e8ef"
    manifest = database.manifest
    assert manifest["conversion"]["recipe"] == {
        "converter_schema": "simllm-external-database-converter-v1",
        "float_encoding": "Python float.hex() encodes raw and served IEEE-754 binary64 values",
        "json_lines": {
            "encoding": "ASCII",
            "ensure_ascii": True,
            "line_termination": "LF (0x0a) after every record",
            "record": "one six-element JSON array per source row",
            "separators": [",", ":"],
        },
        "liblzma": "5.2.3",
        "pyyaml": "6.0.3",
        "row_ordering": (
            "tables follow the declared manifest inventory order; rows within each "
            "table retain PyArrow to_pylist source order; no row sort is applied"
        ),
        "xz": {
            "check": "CHECK_CRC64",
            "extreme": True,
            "format": "FORMAT_XZ",
            "preset": 9,
            "preset_expression": "9 | PRESET_EXTREME",
            "stream_layout": (
                "one XZ stream from one lzma.compress call over the complete JSON Lines payload"
            ),
        },
    }
    assert set(manifest["converted_files_sha256"]) == {
        "system.json",
        "model-config.json",
        "family-mapping.json",
    }
    assert database.system_spec["notice"] == SYSTEM_CONVERSION_NOTICE
    assert database.model_config["notice"] == MODEL_CONVERSION_NOTICE


def test_artifact_preserves_license_notice_and_modified_statement() -> None:
    artifact = default_artifact_dir()
    third_party = (artifact / "THIRD_PARTY_NOTICE").read_text(encoding="utf-8")
    modified = (artifact / "MODIFIED").read_text(encoding="utf-8")
    repository_notice = (ROOT / "NOTICE").read_text(encoding="utf-8")

    assert hashlib.sha256((artifact / "LICENSE").read_bytes()).hexdigest() == (
        EXPECTED_APACHE_LICENSE_HASH
    )
    assert (
        "SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & "
        "AFFILIATES. All rights reserved."
    ) in third_party.splitlines()
    assert (
        "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & "
        "AFFILIATES. All rights reserved."
    ) in third_party.splitlines()
    assert "- system.json:" in modified
    assert "- model-config.json:" in modified
    assert "Copyright (c) 2025-2026 NVIDIA CORPORATION &" in repository_notice
    assert external_artifact_licensing_findings(artifact, ROOT / "NOTICE") == ()


def test_licensing_guard_rejects_truncated_license_and_wrong_year_notice(
    tmp_path: Path,
) -> None:
    artifact = default_artifact_dir()
    truncated = tmp_path / "truncated-license"
    shutil.copytree(artifact, truncated)
    (truncated / "LICENSE").write_bytes((artifact / "LICENSE").read_bytes()[:64])
    assert any(
        "LICENSE" in finding
        for finding in external_artifact_licensing_findings(truncated, ROOT / "NOTICE")
    )

    wrong_year = tmp_path / "wrong-year-notice"
    shutil.copytree(artifact, wrong_year)
    notice_path = wrong_year / "THIRD_PARTY_NOTICE"
    notice_path.write_text(
        notice_path.read_text(encoding="utf-8").replace("2025-2026", "2026"),
        encoding="utf-8",
    )
    assert any(
        "2025-2026" in finding
        for finding in external_artifact_licensing_findings(wrong_year, ROOT / "NOTICE")
    )


@pytest.mark.parametrize(
    ("table", "key", "expected_hex"),
    [
        ("gemm", ("bfloat16", 4353, 65536, 51200), "0x1.4d4fa15555555p+5"),
        ("gemm", ("fp8", 1024, 32, 32), "0x1.9da0d77777778p-7"),
        (
            "generation_attention",
            ("fp8", 8, 64, 0, 96, 64, 2),
            "0x1.03b2840000000p-7",
        ),
        (
            "moe",
            ("bfloat16", "balanced", 8, 128, 4096, 1536, 4, 4, 256),
            "0x1.b6d43d5555555p-4",
        ),
        ("custom_allreduce", ("half", 8, "AUTO", 536870912), "0x1.007923d70a3d7p+2"),
        (
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
        ("compute_scale", ("fp8", 32768, 51200), "0x1.356a966666667p+1"),
    ],
)
def test_representative_raw_rows_round_trip_exactly(
    database: ExternalOperationDatabase,
    table: str,
    key: tuple[object, ...],
    expected_hex: str,
) -> None:
    assert database.raw_latency(table, key).hex() == expected_hex


def test_frozen_load_time_mutation_matches_served_query(database: ExternalOperationDatabase) -> None:
    mutations = database.load_time_mutations()
    assert len(mutations) == 370
    assert len(database.load_time_mutations("gemm")) == 3
    assert len(database.load_time_mutations("generation_attention")) == 367

    frozen = next(
        cell
        for cell in mutations
        if cell["table"] == "gemm" and cell["key"] == ("bfloat16", 32768, 64, 512)
    )
    assert frozen["raw_hex"] == "0x1.eb4af55555555p-8"
    assert frozen["served_hex"] == "0x1.02253ae9a795bp-7"
    served = database.query_gemm(m=32768, n=64, k=512, quant_mode="bfloat16")
    assert served.hex == frozen["served_hex"]

    generation = next(
        cell
        for cell in mutations
        if cell["table"] == "generation_attention"
        and cell["key"] == ("bfloat16", 8, 128, 0, 128, 128, 1024)
    )
    assert generation["raw_hex"] == "0x1.b78732aaaaaabp-4"
    assert generation["served_hex"] == "0x1.d0d73a2abadb5p-4"
    assert database.served_latency("generation_attention", generation["key"]).hex() == (
        generation["served_hex"]
    )


def test_gemm_distance_cap_has_discriminating_local_control(
    database: ExternalOperationDatabase,
) -> None:
    arguments = {
        "m": 257,
        "n": 131072,
        "k": 131072,
        "quant_mode": "bfloat16",
    }
    capped = database.query_gemm(**arguments)
    cap_off = database.query_gemm_cap_off_diagnostic(**arguments)
    assert capped.hex == "0x1.d16e300d9dc77p+3"
    assert cap_off.hex == "0x1.cc9259aaacb10p+3"
    assert capped.hex != cap_off.hex


def test_dispatched_ignored_dimensions_are_invariant(
    database: ExternalOperationDatabase,
) -> None:
    generation = {
        "b": 64,
        "s": 4000,
        "n": 96,
        "n_kv": 8,
        "kv_quant_mode": "fp8",
        "window_size": 0,
        "head_size": 64,
    }
    assert database.query_generation_attention(
        **generation, attn_dtype="bfloat16"
    ).hex == database.query_generation_attention(**generation, attn_dtype="fp8").hex

    moe = {
        "num_tokens": 256,
        "hidden_size": 4096,
        "inter_size": 1536,
        "topk": 8,
        "num_experts": 128,
        "moe_tp_size": 4,
        "moe_ep_size": 4,
        "quant_mode": "bfloat16",
        "workload_distribution": "balanced",
    }
    assert database.query_moe(
        **moe, kernel_source="moe_torch_flow"
    ).hex == database.query_moe(**moe, kernel_source="moe_torch_flow_cutlass").hex

    gdn = {
        "phase": "generation",
        "kernel_source": "fused_sigmoid_gating_delta_rule_update",
        "batch_size": 512,
        "seq_len": None,
        "d_model": 2048,
        "num_k_heads": 16,
        "head_k_dim": 128,
        "num_v_heads": 16,
        "head_v_dim": 128,
        "d_conv": 4,
    }
    assert database.query_gdn(
        **gdn, model_name="Qwen/Qwen3.5-0.8B"
    ).hex == database.query_gdn(**gdn, model_name="Qwen/Qwen3.5-397B-A17B").hex
    assert database.query_gdn(**gdn, num_tokens=512).hex == database.query_gdn(
        **gdn, num_tokens=1024
    ).hex


def test_external_latency_requires_explicit_external_evidence_class(
    database: ExternalOperationDatabase,
) -> None:
    with pytest.raises(TypeError, match="evidence_class"):
        ExternalLatency(  # type: ignore[call-arg]
            latency_ms=1.0,
            source=database.source,
            operation="gemm",
            rule="test",
        )
    with pytest.raises(ValueError, match="MEASURED-EXTERNAL"):
        ExternalLatency(
            latency_ms=1.0,
            source=database.source,
            operation="gemm",
            rule="test",
            evidence_class="MEASURED",
        )


def test_every_served_value_carries_external_identity(database: ExternalOperationDatabase) -> None:
    values = [
        database.query_gemm(m=5000, n=65536, k=51200, quant_mode="bfloat16"),
        database.query_context_attention(
            b=8,
            s=14336,
            prefix=0,
            n=96,
            n_kv=1,
            kv_quant_mode="bfloat16",
            fmha_quant_mode="bfloat16",
            window_size=128,
            head_size=64,
        ),
        database.query_generation_attention(
            b=64,
            s=4000,
            n=96,
            n_kv=8,
            kv_quant_mode="fp8",
            head_size=64,
        ),
    ]
    for value in values:
        assert value.evidence_class == EXTERNAL_EVIDENCE_CLASS
        assert value.source == database.source
        assert value.source.aiconfigurator_version == "0.11.0"
        assert value.source.core_version == "0.11.0"


def test_mapping_table_fails_closed_and_rejects_composite_overlap(
    database: ExternalOperationDatabase,
) -> None:
    assert database.require_mapping("lm_head")["status"] == "exact"
    with pytest.raises(ExternalCompositeError, match="explicit composite"):
        database.require_mapping("attn_score+kv_read")
    with pytest.raises(ExternalDatabaseGapError, match="gdn_gated_norm"):
        database.require_mapping("gdn_gated_norm")
    with pytest.raises(ExternalDatabaseGapError, match="absent"):
        database.require_mapping("undeclared_family")

    ledger = ExternalCompositionLedger(database)
    ledger.claim("attn_score+kv_read", composite=True)
    with pytest.raises(ExternalCompositeError, match="overlaps claimed composite"):
        ledger.claim("attn_score")


@pytest.mark.parametrize(
    ("phase", "batch_size", "isl", "osl", "expected_hex"),
    [
        ("context", 1, 3500, 1, "0x1.8d2164d537eb3p+6"),
        ("generation", 64, 4000, 2, "0x1.6344a3614677ep+3"),
        ("context", 2, 1750, 1, "0x1.7b53c1bc0e6d2p+6"),
        ("generation", 32, 8000, 2, "0x1.66a63c02685c1p+3"),
    ],
)
def test_qwen_pass_composition_matches_frozen_oracles(
    database: ExternalOperationDatabase,
    phase: str,
    batch_size: int,
    isl: int,
    osl: int,
    expected_hex: str,
) -> None:
    model = ExternalQwen32BPassModel(database)
    if phase == "context":
        result = model.run_context(batch_size=batch_size, isl=isl)
    else:
        result = model.run_generation(batch_size=batch_size, isl=isl, osl=osl, stride=32)
    assert result.total.hex == expected_hex
    assert len(result.operations) == 14
    assert all(value.evidence_class == EXTERNAL_EVIDENCE_CLASS for value in result.operations)


def test_import_requires_explicit_pinned_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIMLLM_EXTERNAL_AIC_VENV", raising=False)
    with pytest.raises(ExternalDatabaseIdentityError, match="SIMLLM_EXTERNAL_AIC_VENV"):
        import_external_database()


def test_payload_filename_is_checked_before_decompression(tmp_path: Path) -> None:
    artifact = default_artifact_dir()
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    manifest["conversion"]["payload_sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ExternalDatabaseIdentityError, match="content-addressed"):
        ExternalOperationDatabase.load(tmp_path)


@pytest.mark.parametrize(
    "filename",
    ["system.json", "model-config.json", "family-mapping.json"],
)
def test_loader_verifies_every_converted_json_hash(
    tmp_path: Path,
    filename: str,
) -> None:
    tampered = tmp_path / filename.replace(".", "-")
    shutil.copytree(default_artifact_dir(), tampered)
    path = tampered / filename
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ExternalDatabaseIdentityError, match=filename):
        ExternalOperationDatabase.load(tampered)


@pytest.mark.parametrize(
    ("filename", "expected_message"),
    [
        ("system.json", "system.json conversion notice mismatch"),
        ("model-config.json", "model-config.json conversion notice mismatch"),
    ],
)
def test_loader_verifies_file_local_conversion_notices(
    tmp_path: Path,
    filename: str,
    expected_message: str,
) -> None:
    tampered = tmp_path / filename.replace(".", "-")
    shutil.copytree(default_artifact_dir(), tampered)
    path = tampered / filename
    document = json.loads(path.read_text(encoding="utf-8"))
    document["notice"] = "wrong conversion notice"
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest_path = tampered / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["converted_files_sha256"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ExternalDatabaseIdentityError, match=expected_message):
        ExternalOperationDatabase.load(tampered)


def test_loader_version_scan_catches_rehashed_donor_row(tmp_path: Path) -> None:
    tampered = tmp_path / "donor-row"
    shutil.copytree(default_artifact_dir(), tampered)
    manifest_path = tampered / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_payload = tampered / manifest["conversion"]["payload"]
    raw = lzma.decompress(original_payload.read_bytes(), format=lzma.FORMAT_XZ)
    lines = raw.splitlines(keepends=True)
    first = json.loads(lines[0])
    first[2] = "1.2.0rc5"
    lines[0] = (
        json.dumps(first, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        + b"\n"
    )
    tampered_raw = b"".join(lines)
    tampered_payload = lzma.compress(
        tampered_raw,
        format=lzma.FORMAT_XZ,
        check=lzma.CHECK_CRC64,
        preset=9 | lzma.PRESET_EXTREME,
    )
    payload_hash = hashlib.sha256(tampered_payload).hexdigest()
    payload_name = f"rows-{payload_hash}.jsonl.xz"
    (tampered / payload_name).write_bytes(tampered_payload)
    manifest["conversion"].update(
        {
            "payload": payload_name,
            "payload_sha256": payload_hash,
            "payload_bytes": len(tampered_payload),
            "uncompressed_bytes": len(tampered_raw),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ExternalDatabaseIdentityError, match="donor database version"):
        ExternalOperationDatabase.load(tampered)
