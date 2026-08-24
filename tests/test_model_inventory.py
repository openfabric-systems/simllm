from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import simllm.calibration.extraction as extraction_module
from simllm.calibration.extraction import (
    ModelExtractionError,
    extract_model_inventory,
)
from simllm.calibration.model_inventory import (
    ABSENT_BY_DESIGN,
    MODEL_KERNEL_INVENTORY_SCHEMA,
    FrameworkIdentity,
    ModelKernelInventory,
)
from simllm.calibration.validation import validate_path, validate_typed_record
from simllm.compute import ModelDims

REVISION = "a" * 40
FRAMEWORK_COMMIT = "b" * 40


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _suite(config: bytes, weights: bytes) -> dict[str, object]:
    return {
        "schema": "simllm-transformer-dag-suite-v1",
        "suite": "unit-transformer-dag-v1",
        "state": "authored-inputs-only",
        "reference_model": {
            "name": "unit/granite",
            "revision": REVISION,
            "config_sha256": _sha256(config),
            "weight_sha256": _sha256(weights),
            "weight_bytes": len(weights),
            "dtype": "bfloat16",
            "quantization": "none",
            "geometry": {
                "layers": 24,
                "hidden_size": 1024,
                "intermediate_size": 512,
                "num_heads": 16,
                "num_kv_heads": 8,
                "head_size": 64,
                "num_experts": 32,
                "top_k": 8,
                "vocab_size": 49155,
            },
        },
        "frameworks": [
            {
                "id": "vllm",
                "version": "0.26.0",
                "source_commit": FRAMEWORK_COMMIT,
            }
        ],
        "graph_cells": [
            {
                "id": "prefill",
                "family": "compute-prefill",
                "phase": "prefill",
                "split": "train",
                "requests": 2,
                "prompt_tokens_per_request": 8,
                "total_prompt_tokens": 16,
            },
            {
                "id": "decode",
                "family": "memory-decode",
                "phase": "decode",
                "split": "validation",
                "batch": 2,
                "context_tokens": 64,
                "new_tokens_per_request": 1,
            },
            {
                "id": "moe",
                "family": "moe-communication-decode",
                "phase": "decode",
                "split": "test",
                "batch": 4,
                "context_tokens": 128,
                "new_tokens_per_request": 1,
                "expert_participants": 4,
                "parallelism_override": {"expert": 4},
            },
        ],
    }


def _dims(*, defaulted_fields: tuple[str, ...] = ()) -> ModelDims:
    return ModelDims(
        num_layers=24,
        hidden_size=1024,
        intermediate_size=512,
        num_heads=16,
        num_kv_heads=8,
        head_size=64,
        vocab_size=49155,
        dtype_bytes=2,
        weight_dtype_bytes=2,
        kv_dtype_bytes=2,
        defaulted_fields=defaulted_fields,
        num_experts=32,
        top_k=8,
        moe_intermediate_size=512,
        local_num_experts=32,
    )


@pytest.fixture
def extracted(tmp_path: Path) -> ModelKernelInventory:
    config = b'{"model_type":"granitemoe"}'
    weights = b"small deterministic weights"
    checkpoint = tmp_path / REVISION
    checkpoint.mkdir()
    (checkpoint / "config.json").write_bytes(config)
    (checkpoint / "model.safetensors").write_bytes(weights)
    return extract_model_inventory(
        suite_raw=json.dumps(_suite(config, weights)).encode(),
        framework=FrameworkIdentity(
            framework_id="vllm",
            version="0.26.0",
            source_commit=FRAMEWORK_COMMIT,
            source_tree=None,
            entry_seam="flagged-skeleton-step-record-v1",
        ),
        checkpoint_root=checkpoint,
        framework_dims=_dims(),
        step_records_path=tmp_path / "steps.jsonl",
    )


def test_inventory_is_canonical_strict_and_typed(
    extracted: ModelKernelInventory,
    tmp_path: Path,
) -> None:
    record = extracted.record
    assert record.schema == MODEL_KERNEL_INVENTORY_SCHEMA
    assert ModelKernelInventory.from_obj(extracted.to_obj()) == extracted
    assert validate_typed_record(extracted.to_obj()) == extracted
    path = tmp_path / f"{record.record_id}.json"
    path.write_bytes(record.canonical)
    result = validate_path(path)
    assert result.record_schema == MODEL_KERNEL_INVENTORY_SCHEMA
    assert result.record_sha256 == record.record_id


def test_inventory_has_total_ordered_cases_and_frozen_launch_counts(
    extracted: ModelKernelInventory,
) -> None:
    assert [case.case_id for case in extracted.cases] == [
        "prefill",
        "decode",
        "moe",
    ]
    assert [family.family_id for family in extracted.kernel_families] == [
        "attn_gemm",
        "attn_score",
        "mlp_gemm",
        "lm_head",
        "kv_read",
    ]
    assert [
        sum(item.logical_launch_count for item in case.kernel_projections)
        for case in extracted.cases
    ] == [97, 97, 97]
    assert len({case.template_graph_sha256 for case in extracted.cases}) == 2
    for case in extracted.cases:
        assert sum(item.aggregate_flops for item in case.kernel_projections) > 0
        assert sum(item.aggregate_hbm_bytes for item in case.kernel_projections) > 0


def test_physical_identity_is_explicitly_absent_by_design(
    extracted: ModelKernelInventory,
) -> None:
    envelope = extracted.implementation_identity.to_obj()
    assert envelope == {
        "code_object_hashes": {"state": ABSENT_BY_DESIGN, "value": None},
        "observed_launches": {"state": ABSENT_BY_DESIGN, "value": None},
        "join_tasks": ["COMP-6", "VLLM-12"],
    }


def test_inventory_reader_rejects_unknown_partial_and_reordered_content(
    extracted: ModelKernelInventory,
) -> None:
    extra = extracted.to_obj()
    extra["unknown"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        ModelKernelInventory.from_obj(extra)

    partial = extracted.to_obj()
    partial["cases"] = partial["cases"][:-1]
    with pytest.raises(ValueError, match="cases, expected"):
        ModelKernelInventory.from_obj(partial)

    reordered = extracted.to_obj()
    reordered["cases"][0]["kernel_projections"].reverse()
    with pytest.raises(ValueError, match="projection order"):
        ModelKernelInventory.from_obj(reordered)


def test_inventory_reader_rejects_a_fabricated_physical_identity(
    extracted: ModelKernelInventory,
) -> None:
    fabricated = extracted.to_obj()
    fabricated["implementation_identity"]["observed_launches"] = {
        "state": "known",
        "value": [],
    }
    with pytest.raises(ValueError, match="absent-by-design"):
        ModelKernelInventory.from_obj(fabricated)


def test_extraction_is_byte_deterministic(
    extracted: ModelKernelInventory,
    tmp_path: Path,
) -> None:
    config = (tmp_path / REVISION / "config.json").read_bytes()
    weights = (tmp_path / REVISION / "model.safetensors").read_bytes()
    second = extract_model_inventory(
        suite_raw=json.dumps(_suite(config, weights)).encode(),
        framework=extracted.framework,
        checkpoint_root=tmp_path / REVISION,
        framework_dims=_dims(),
        step_records_path=tmp_path / "second.jsonl",
    )
    assert second.record.canonical == extracted.record.canonical


def test_extraction_rejects_unknown_family_without_an_inventory(tmp_path: Path) -> None:
    config = b"{}"
    weights = b"weights"
    checkpoint = tmp_path / REVISION
    checkpoint.mkdir()
    (checkpoint / "config.json").write_bytes(config)
    (checkpoint / "model.safetensors").write_bytes(weights)
    suite = _suite(config, weights)
    suite["graph_cells"][0]["family"] = "unknown"
    with pytest.raises(ModelExtractionError, match="unknown family"):
        extract_model_inventory(
            suite_raw=json.dumps(suite).encode(),
            framework=FrameworkIdentity(
                "vllm",
                "0.26.0",
                FRAMEWORK_COMMIT,
                None,
                "flagged-skeleton-step-record-v1",
            ),
            checkpoint_root=checkpoint,
            framework_dims=_dims(),
            step_records_path=tmp_path / "rejected.jsonl",
        )
    assert not (tmp_path / "rejected.jsonl").exists()


def test_extraction_rejects_malformed_framework_rows_without_an_inventory(
    tmp_path: Path,
) -> None:
    config = b"{}"
    weights = b"weights"
    checkpoint = tmp_path / REVISION
    checkpoint.mkdir()
    (checkpoint / "config.json").write_bytes(config)
    (checkpoint / "model.safetensors").write_bytes(weights)
    suite = _suite(config, weights)
    suite["frameworks"] = ["vllm"]

    with pytest.raises(ModelExtractionError, match="entries must be objects"):
        extract_model_inventory(
            suite_raw=json.dumps(suite).encode(),
            framework=FrameworkIdentity(
                "vllm",
                "0.26.0",
                FRAMEWORK_COMMIT,
                None,
                "flagged-skeleton-step-record-v1",
            ),
            checkpoint_root=checkpoint,
            framework_dims=_dims(),
            step_records_path=tmp_path / "rejected.jsonl",
        )
    assert not (tmp_path / "rejected.jsonl").exists()


def test_extraction_rejects_a_partial_step_record_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = b"{}"
    weights = b"weights"
    checkpoint = tmp_path / REVISION
    checkpoint.mkdir()
    (checkpoint / "config.json").write_bytes(config)
    (checkpoint / "model.safetensors").write_bytes(weights)
    real_reader = extraction_module.step_records_from_jsonl

    def partial_reader(path: Path):
        return tuple(real_reader(path))[:-1]

    monkeypatch.setattr(
        extraction_module,
        "step_records_from_jsonl",
        partial_reader,
    )
    with pytest.raises(ModelExtractionError, match="complete case set"):
        extract_model_inventory(
            suite_raw=json.dumps(_suite(config, weights)).encode(),
            framework=FrameworkIdentity(
                "vllm",
                "0.26.0",
                FRAMEWORK_COMMIT,
                None,
                "flagged-skeleton-step-record-v1",
            ),
            checkpoint_root=checkpoint,
            framework_dims=_dims(),
            step_records_path=tmp_path / "partial.jsonl",
        )


def test_extraction_rejects_checkpoint_and_framework_geometry_mismatch(
    tmp_path: Path,
) -> None:
    config = b"{}"
    weights = b"weights"
    checkpoint = tmp_path / REVISION
    checkpoint.mkdir()
    (checkpoint / "config.json").write_bytes(config)
    (checkpoint / "model.safetensors").write_bytes(weights + b"changed")
    suite = _suite(config, weights)
    arguments = {
        "suite_raw": json.dumps(suite).encode(),
        "framework": FrameworkIdentity(
            "vllm",
            "0.26.0",
            FRAMEWORK_COMMIT,
            None,
            "flagged-skeleton-step-record-v1",
        ),
        "checkpoint_root": checkpoint,
        "framework_dims": _dims(),
        "step_records_path": tmp_path / "steps.jsonl",
    }
    with pytest.raises(ModelExtractionError, match="weight byte count"):
        extract_model_inventory(**arguments)

    (checkpoint / "model.safetensors").write_bytes(weights)
    arguments["framework_dims"] = _dims(defaulted_fields=("hidden_size",))
    with pytest.raises(ModelExtractionError, match="used defaults"):
        extract_model_inventory(**arguments)


def test_record_constructor_rejects_missing_case_even_without_wire_parse(
    extracted: ModelKernelInventory,
) -> None:
    with pytest.raises(ValueError, match="cases, expected"):
        replace(extracted, cases=extracted.cases[:-1])


def test_inventory_value_copy_does_not_mutate_typed_record(
    extracted: ModelKernelInventory,
) -> None:
    value = extracted.to_obj()
    copied = copy.deepcopy(value)
    copied["cases"][0]["case_id"] = "changed"
    assert extracted.cases[0].case_id == "prefill"
