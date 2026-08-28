# SPDX-FileCopyrightText: Copyright (c) 2026 Yifeng Wang
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simllm.calibration.external_db import (
    EXPECTED_SLICE_HASH,
    EXTERNAL_EVIDENCE_CLASS,
    ExternalCompositeError,
    ExternalCompositionLedger,
    ExternalDatabaseGapError,
    ExternalDatabaseIdentityError,
    ExternalOperationDatabase,
    ExternalQwen32BPassModel,
    default_artifact_dir,
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


def test_artifact_preserves_license_notice_and_modified_statement() -> None:
    artifact = default_artifact_dir()
    license_text = (artifact / "LICENSE").read_text(encoding="utf-8")
    third_party = (artifact / "THIRD_PARTY_NOTICE").read_text(encoding="utf-8")
    modified = (artifact / "MODIFIED").read_text(encoding="utf-8")
    repository_notice = (ROOT / "NOTICE").read_text(encoding="utf-8")

    assert "Apache License" in license_text
    assert "Version 2.0" in license_text
    assert "SPDX-License-Identifier: Apache-2.0" in third_party
    assert "NVIDIA CORPORATION & AFFILIATES" in third_party
    assert "converted" in modified
    assert "otherwise altered" in modified
    assert "offline/calibration/external-databases" in repository_notice
    assert "NVIDIA AIConfigurator" in repository_notice


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
