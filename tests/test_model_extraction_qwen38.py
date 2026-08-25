from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from simllm.calibration.extraction import (
    FrameworkConfigurationProjection,
    FrameworkTextStack,
    ModelExtractionError,
    case_records_from_suite,
    extract_model_inventory,
    load_extraction_suite,
)
from simllm.calibration.model_inventory import FrameworkIdentity, ModelGeometry
from simllm.compute import ModelDims
from simllm.core import RequestPhase

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "offline/calibration/suites/qwen3.8-27b-text-v1/suite.json"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
FRAMEWORK = FrameworkIdentity(
    framework_id="vllm",
    version="0.26.0",
    source_commit="568afb3a13806beb53bb2e6bd518269357b237c0",
    source_tree=None,
    entry_seam="flagged-skeleton-step-record-v1",
)


def _dims() -> ModelDims:
    return ModelDims(
        num_layers=64,
        hidden_size=5120,
        intermediate_size=17408,
        num_heads=24,
        num_kv_heads=4,
        head_size=256,
        vocab_size=248320,
        dtype_bytes=2,
        weight_dtype_bytes=2,
        kv_dtype_bytes=2,
    )


def _suite(config: bytes) -> dict[str, object]:
    value = json.loads(SUITE.read_bytes())
    value["reference_model"]["config_sha256"] = hashlib.sha256(config).hexdigest()
    return value


def _projection(suite: dict[str, object]) -> FrameworkConfigurationProjection:
    model = suite["reference_model"]
    text = model["text_stack"]
    schedule = tuple(text["layer_pattern"]) * text["pattern_repetitions"]
    return FrameworkConfigurationProjection(
        framework=FRAMEWORK,
        configuration_seam="ModelConfig-with-skip-tokenizer-init",
        architecture_binding="model_executor/models/registry.py:573",
        text_implementation="QwenGatedDeltaNetAttention",
        text_stack=FrameworkTextStack(
            architecture=model["architecture"],
            wrapper_model_type=model["model_type"],
            text_model_type=text["model_type"],
            scope=text["scope"],
            geometry=ModelGeometry.from_obj(model["geometry"]),
            layer_types=schedule,
            linear_attention_mechanism=text["linear_attention_mechanism"],
            linear_conv_kernel_dim=text["linear_conv_kernel_dim"],
            linear_key_head_dim=text["linear_key_head_dim"],
            linear_value_head_dim=text["linear_value_head_dim"],
            linear_num_key_heads=text["linear_num_key_heads"],
            linear_num_value_heads=text["linear_num_value_heads"],
            attn_output_gate=text["attn_output_gate"],
            output_gate_type=text["output_gate_type"],
            state_dtype=text["state_dtype"],
            excluded_components=tuple(text["excluded_components"]),
        ),
    )


def _extract(tmp_path: Path, suite: dict[str, object], config: bytes) -> None:
    checkpoint = tmp_path / REVISION
    checkpoint.mkdir()
    (checkpoint / "config.json").write_bytes(config)
    extract_model_inventory(
        suite_raw=json.dumps(suite).encode(),
        framework=FRAMEWORK,
        checkpoint_root=checkpoint,
        framework_dims=_dims(),
        step_records_path=tmp_path / "steps.jsonl",
        framework_projection=_projection(suite),
    )


def test_qwen38_metadata_only_extraction_rejects_before_writing(tmp_path: Path) -> None:
    config = b'{"model_type":"qwen3_5"}'
    suite = _suite(config)

    with pytest.raises(ModelExtractionError, match=r"COMP-62.*Gated DeltaNet"):
        _extract(tmp_path, suite, config)

    assert not (tmp_path / "steps.jsonl").exists()


def test_qwen38_metadata_manifest_mutation_rejects(tmp_path: Path) -> None:
    config = b'{"model_type":"qwen3_5"}'
    suite = _suite(config)
    suite["reference_model"]["weight_shards"][0]["bytes"] += 1

    with pytest.raises(ModelExtractionError, match="byte total"):
        _extract(tmp_path, suite, config)

    assert not (tmp_path / "steps.jsonl").exists()


def test_qwen38_local_weight_presence_rejects(tmp_path: Path) -> None:
    config = b'{"model_type":"qwen3_5"}'
    suite = _suite(config)
    checkpoint = tmp_path / REVISION
    checkpoint.mkdir()
    (checkpoint / "config.json").write_bytes(config)
    (checkpoint / "model-00001-of-00018.safetensors").write_bytes(b"forbidden")

    with pytest.raises(ModelExtractionError, match="weight-free local substrate"):
        extract_model_inventory(
            suite_raw=json.dumps(suite).encode(),
            framework=FRAMEWORK,
            checkpoint_root=checkpoint,
            framework_dims=_dims(),
            step_records_path=tmp_path / "steps.jsonl",
            framework_projection=_projection(suite),
        )

    assert not (tmp_path / "steps.jsonl").exists()


def test_qwen38_dense_batch_cells_preserve_all_shape_axes() -> None:
    suite, _ = load_extraction_suite(SUITE.read_bytes())
    records = case_records_from_suite(suite)

    assert len(records) == 15
    assert [record.num_tokens_after_padding for record in records[:5]] == [
        128,
        768,
        2048,
        512,
        1024,
    ]
    assert [record.scheduled[0].context_length for record in records[5:10]] == [
        128,
        1024,
        8192,
        512,
        2048,
    ]
    assert [len(record.scheduled) for record in records[10:]] == [1, 16, 64, 4, 8]
    assert all(
        request.phase == RequestPhase.DECODE and request.num_new_tokens == 1
        for record in records[5:]
        for request in record.scheduled
    )


def test_qwen38_framework_projection_mismatch_rejects(tmp_path: Path) -> None:
    config = b'{"model_type":"qwen3_5"}'
    suite = _suite(config)
    projection = _projection(suite)
    wrong_stack = replace(
        projection.text_stack,
        linear_conv_kernel_dim=projection.text_stack.linear_conv_kernel_dim + 1,
    )
    checkpoint = tmp_path / REVISION
    checkpoint.mkdir()
    (checkpoint / "config.json").write_bytes(config)

    with pytest.raises(ModelExtractionError, match="does not match the suite"):
        extract_model_inventory(
            suite_raw=json.dumps(suite).encode(),
            framework=FRAMEWORK,
            checkpoint_root=checkpoint,
            framework_dims=_dims(),
            step_records_path=tmp_path / "steps.jsonl",
            framework_projection=replace(projection, text_stack=wrong_stack),
        )

    assert not (tmp_path / "steps.jsonl").exists()
