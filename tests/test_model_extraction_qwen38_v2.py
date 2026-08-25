from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import simllm.calibration.extraction as extraction_module
from simllm.calibration.extraction import (
    QWEN_GATED_DELTA_NET_FAMILIES,
    FrameworkConfigurationProjection,
    FrameworkTextStack,
    ModelExtractionError,
    extract_model_inventory,
)
from simllm.calibration.model_inventory import (
    FrameworkIdentity,
    ModelGeometry,
    ModelKernelInventory,
)
from simllm.compute import ModelDims

ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = (
    ROOT
    / "offline/calibration/suites"
    / "qwen3.8-27b-text-v1-frameworks-2026-08-25/suite.json"
)
EXPECTATIONS_PATH = ROOT / "examples/model_extraction_qwen38_v2/expectations.json"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _framework(framework_id: str) -> FrameworkIdentity:
    rows = {
        row["id"]: row for row in _load(SUITE_PATH)["frameworks"]
    }
    row = rows[framework_id]
    return FrameworkIdentity(
        framework_id=framework_id,
        version=row["version"],
        source_commit=row["source_commit"],
        source_tree=row.get("source_tree"),
        entry_seam=(
            "flagged-skeleton-step-record-v1"
            if framework_id == "vllm"
            else "cpu-engine-step-record-v1"
        ),
    )


def _projection(
    suite: dict[str, object],
    framework: FrameworkIdentity,
) -> FrameworkConfigurationProjection:
    model = suite["reference_model"]
    text = model["text_stack"]
    framework_row = next(
        row for row in suite["frameworks"] if row["id"] == framework.framework_id
    )
    return FrameworkConfigurationProjection(
        framework=framework,
        configuration_seam=(
            "ModelConfig-with-skip-tokenizer-init"
            if framework.framework_id == "vllm"
            else "DeviceConfig-cpu-plus-ModelConfig-with-multimodal-disabled"
        ),
        architecture_binding=framework_row["architecture_binding"],
        text_implementation=framework_row["text_implementation"],
        text_stack=FrameworkTextStack(
            architecture=model["architecture"],
            wrapper_model_type=model["model_type"],
            text_model_type=text["model_type"],
            scope=text["scope"],
            geometry=ModelGeometry.from_obj(model["geometry"]),
            layer_types=tuple(text["layer_pattern"]) * text["pattern_repetitions"],
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


def _authored_suite(config: bytes) -> dict[str, object]:
    suite = _load(SUITE_PATH)
    suite["reference_model"]["config_sha256"] = hashlib.sha256(config).hexdigest()
    return suite


def _extract(
    tmp_path: Path,
    framework_id: str,
    *,
    projection_mutation: str | None = None,
) -> ModelKernelInventory:
    config = b'{"model_type":"qwen3_5"}'
    suite = _authored_suite(config)
    framework = _framework(framework_id)
    projection = _projection(suite, framework)
    if projection_mutation == "state-dtype":
        projection = replace(
            projection,
            text_stack=replace(projection.text_stack, state_dtype="bfloat16"),
        )
    elif projection_mutation == "conv-width":
        projection = replace(
            projection,
            text_stack=replace(projection.text_stack, linear_conv_kernel_dim=3),
        )
    elif projection_mutation == "schedule-order":
        schedule = list(projection.text_stack.layer_types)
        schedule[2], schedule[3] = schedule[3], schedule[2]
        projection = replace(
            projection,
            text_stack=replace(projection.text_stack, layer_types=tuple(schedule)),
        )
    elif projection_mutation == "scope-exclusions":
        projection = replace(
            projection,
            text_stack=replace(
                projection.text_stack,
                excluded_components=("multimodal-vision-encoder",),
            ),
        )
    elif projection_mutation == "architecture":
        projection = replace(
            projection,
            text_stack=replace(
                projection.text_stack,
                architecture="UnsupportedForConditionalGeneration",
            ),
        )
    checkpoint = tmp_path / framework_id / REVISION
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_bytes(config)
    return extract_model_inventory(
        suite_raw=json.dumps(suite).encode(),
        framework=framework,
        checkpoint_root=checkpoint,
        framework_dims=_dims(),
        step_records_path=tmp_path / framework_id / "steps.jsonl",
        framework_projection=projection,
    )


def _neutral_inventory(inventory: ModelKernelInventory) -> dict[str, object]:
    value = inventory.to_obj()
    value.pop("framework")
    value["implementation_identity"]["join_tasks"] = ["COMP-6"]
    return value


def _expected_family_work(
    family: dict[str, object],
    oracle: dict[str, object],
) -> tuple[int, int, tuple[int, ...]]:
    layers = family["layers"]
    tokens = oracle["new_tokens"]
    sequences = oracle["sequences"]
    kv_tokens = oracle["kv_tokens"]
    pairs = oracle["attention_pairs"]
    sampled = oracle["sampled"]
    shapes = {
        "new_tokens": tokens,
        "sequences": sequences,
        "kv_tokens": kv_tokens,
        "sampled": sampled,
    }
    flops = 0
    hbm_bytes = 0
    if "flops_per_new_token_per_layer" in family:
        flops += family["flops_per_new_token_per_layer"] * tokens * layers
    if "flops_per_attention_pair_per_layer" in family:
        flops += family["flops_per_attention_pair_per_layer"] * pairs * layers
    if "flops_per_sampled_token" in family:
        flops += family["flops_per_sampled_token"] * sampled
    if "static_hbm_bytes_per_layer" in family:
        hbm_bytes += family["static_hbm_bytes_per_layer"] * layers
    if "static_hbm_bytes" in family:
        hbm_bytes += family["static_hbm_bytes"]
    if "hbm_bytes_per_kv_token_per_layer" in family:
        hbm_bytes += family["hbm_bytes_per_kv_token_per_layer"] * kv_tokens * layers
    if "hbm_bytes_per_sequence_per_layer" in family:
        hbm_bytes += family["hbm_bytes_per_sequence_per_layer"] * sequences * layers
    if "state_read_write_hbm_bytes_per_sequence_per_layer" in family:
        hbm_bytes += (
            family["state_read_write_hbm_bytes_per_sequence_per_layer"]
            * sequences
            * layers
        )
    return (
        flops,
        hbm_bytes,
        tuple(shapes[axis] for axis in family["shape_axes"]),
    )


@pytest.mark.parametrize("framework_id", ["vllm", "sglang"])
def test_qwen38_v2_all_families_conserve_every_frozen_case(
    tmp_path: Path,
    framework_id: str,
) -> None:
    inventory = _extract(tmp_path, framework_id)
    freeze = _load(EXPECTATIONS_PATH)
    frozen_families = {
        family["id"]: family for family in freeze["inventory_contract"]["families"]
    }

    assert [family.family_id for family in inventory.kernel_families] == list(
        QWEN_GATED_DELTA_NET_FAMILIES
    )
    assert len(inventory.cases) == 15
    for case, oracle in zip(inventory.cases, freeze["exact_case_oracles"], strict=True):
        assert case.case_id == oracle["case_id"]
        assert [item.family_id for item in case.kernel_projections] == list(
            QWEN_GATED_DELTA_NET_FAMILIES
        )
        assert sum(
            item.logical_launch_count for item in case.kernel_projections
        ) == 449
        assert sum(item.aggregate_flops for item in case.kernel_projections) == (
            oracle["aggregate_flops"]
        )
        assert sum(
            item.aggregate_hbm_bytes for item in case.kernel_projections
        ) == oracle["aggregate_hbm_bytes"]
        assert all(
            type(value) is int
            for item in case.kernel_projections
            for value in (
                item.aggregate_flops,
                item.aggregate_hbm_bytes,
                *item.shape_vector.values,
            )
        )
        for projection in case.kernel_projections:
            expected_flops, expected_bytes, expected_shape = _expected_family_work(
                frozen_families[projection.family_id],
                oracle,
            )
            assert projection.aggregate_flops == expected_flops
            assert projection.aggregate_hbm_bytes == expected_bytes
            assert projection.shape_vector.values == expected_shape


def test_qwen38_v2_frameworks_are_structurally_identical(tmp_path: Path) -> None:
    vllm = _extract(tmp_path, "vllm")
    sglang = _extract(tmp_path, "sglang")

    assert _neutral_inventory(vllm) == _neutral_inventory(sglang)
    assert vllm.record.record_id != sglang.record.record_id
    assert vllm.cases == sglang.cases


@pytest.mark.parametrize("framework_id", ["vllm", "sglang"])
def test_qwen38_v2_extraction_is_byte_deterministic(
    tmp_path: Path,
    framework_id: str,
) -> None:
    first = _extract(tmp_path / "first", framework_id)
    second = _extract(tmp_path / "second", framework_id)

    assert first.record.canonical == second.record.canonical
    assert (tmp_path / "first" / framework_id / "steps.jsonl").read_bytes() == (
        tmp_path / "second" / framework_id / "steps.jsonl"
    ).read_bytes()


@pytest.mark.parametrize(
    "mutation",
    [
        "state-dtype",
        "conv-width",
        "schedule-order",
        "scope-exclusions",
        "architecture",
    ],
)
def test_qwen38_v2_text_stack_mutations_reject_before_writing(
    tmp_path: Path,
    mutation: str,
) -> None:
    with pytest.raises(ModelExtractionError, match="does not match the suite"):
        _extract(tmp_path, "vllm", projection_mutation=mutation)

    assert not (tmp_path / "vllm" / "steps.jsonl").exists()


def test_qwen38_v2_missing_family_rejects_total_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_builder = extraction_module._qwen_gdn_family_work

    def missing_family(*args: object, **kwargs: object):
        return real_builder(*args, **kwargs)[:-1]

    monkeypatch.setattr(
        extraction_module,
        "_qwen_gdn_family_work",
        missing_family,
    )
    with pytest.raises(ModelExtractionError, match="frozen family order"):
        _extract(tmp_path, "vllm")


def test_qwen38_v2_multimodal_family_injection_rejects_total_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_builder = extraction_module._qwen_gdn_family_work

    def extra_family(*args: object, **kwargs: object):
        families = list(real_builder(*args, **kwargs))
        families.append(replace(families[-1], name="multimodal_vision_encoder"))
        return tuple(families)

    monkeypatch.setattr(
        extraction_module,
        "_qwen_gdn_family_work",
        extra_family,
    )
    with pytest.raises(ModelExtractionError, match="frozen family order"):
        _extract(tmp_path, "vllm")


def test_qwen38_v2_noninteger_family_work_rejects_total_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_builder = extraction_module._qwen_gdn_family_work

    def fractional_family(*args: object, **kwargs: object):
        families = list(real_builder(*args, **kwargs))
        family = families[0]
        object.__setattr__(family, "flops", 0.5)
        return tuple(families)

    monkeypatch.setattr(
        extraction_module,
        "_qwen_gdn_family_work",
        fractional_family,
    )
    with pytest.raises(ModelExtractionError, match="exact nonnegative integer work"):
        _extract(tmp_path, "vllm")
