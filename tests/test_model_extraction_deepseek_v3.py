from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import simllm.calibration.extraction as extraction_module
from simllm.calibration.canonical import canonical_bytes
from simllm.calibration.deepseek_deployment import (
    DEEPSEEK_DEPLOYMENT_PROJECTION_SCHEMA,
    build_deepseek_deployment_projection,
)
from simllm.calibration.extraction import (
    DEEPSEEK_V3_FAMILIES,
    FrameworkConfigurationProjection,
    FrameworkDeepseekStack,
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
    / "deepseek-v3-text-v1-frameworks-2026-08-25/suite.json"
)
EXPECTATIONS_PATH = ROOT / "examples/model_extraction_deepseek_v3_v1/expectations.json"
REVISION = "e815299b0bcbac849fa540c768ef21845365c9eb"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dims() -> ModelDims:
    return ModelDims(
        num_layers=61,
        hidden_size=7168,
        intermediate_size=18432,
        num_heads=128,
        num_kv_heads=128,
        head_size=192,
        vocab_size=129280,
        dtype_bytes=2,
        weight_dtype_bytes=1,
        kv_dtype_bytes=2,
        num_experts=256,
        top_k=8,
        moe_intermediate_size=2048,
        local_num_experts=256,
    )


def _framework(framework_id: str) -> FrameworkIdentity:
    rows = {row["id"]: row for row in _load(SUITE_PATH)["frameworks"]}
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
    stack = model["deepseek_stack"]
    framework_row = next(
        row for row in suite["frameworks"] if row["id"] == framework.framework_id
    )
    geometry = ModelGeometry.from_obj(model["geometry"])
    return FrameworkConfigurationProjection(
        framework=framework,
        configuration_seam=(
            "ModelConfig-with-skip-tokenizer-init"
            if framework.framework_id == "vllm"
            else "DeviceConfig-cpu-plus-ModelConfig-with-multimodal-disabled"
        ),
        architecture_binding=framework_row["architecture_binding"],
        text_implementation=framework_row["text_implementation"],
        deepseek_stack=FrameworkDeepseekStack(
            architecture=model["architecture"],
            wrapper_model_type=model["model_type"],
            scope=stack["scope"],
            geometry=geometry,
            layer_types=("dense",) * stack["first_k_dense_replace"]
            + ("moe",)
            * (geometry.layers - stack["first_k_dense_replace"]),
            q_lora_rank=stack["q_lora_rank"],
            kv_lora_rank=stack["kv_lora_rank"],
            qk_nope_head_dim=stack["qk_nope_head_dim"],
            qk_rope_head_dim=stack["qk_rope_head_dim"],
            v_head_dim=stack["v_head_dim"],
            first_k_dense_replace=stack["first_k_dense_replace"],
            moe_intermediate_size=stack["moe_intermediate_size"],
            moe_layer_freq=stack["moe_layer_freq"],
            n_shared_experts=stack["n_shared_experts"],
            n_group=stack["n_group"],
            topk_group=stack["topk_group"],
            scoring_func=stack["scoring_func"],
            topk_method=stack["topk_method"],
            norm_topk_prob=stack["norm_topk_prob"],
            routed_scaling_factor=stack["routed_scaling_factor"],
            num_nextn_predict_layers=stack["num_nextn_predict_layers"],
            weight_block_size=tuple(stack["weight_block_size"]),
            excluded_components=tuple(stack["excluded_components"]),
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
    config = b'{"model_type":"deepseek_v3"}'
    suite = _authored_suite(config)
    framework = _framework(framework_id)
    projection = _projection(suite, framework)
    assert projection.deepseek_stack is not None
    stack = projection.deepseek_stack
    if projection_mutation == "q-rank":
        stack = replace(stack, q_lora_rank=1024)
    elif projection_mutation == "schedule-order":
        schedule = list(stack.layer_types)
        schedule[2], schedule[3] = schedule[3], schedule[2]
        stack = replace(stack, layer_types=tuple(schedule))
    elif projection_mutation == "weight-block":
        stack = replace(stack, weight_block_size=(64, 128))
    elif projection_mutation == "scope-exclusions":
        stack = replace(
            stack,
            excluded_components=(
                "input-embedding-family",
                "normalization-family",
            ),
        )
    elif projection_mutation == "mtp-layers":
        stack = replace(stack, num_nextn_predict_layers=2)
    projection = replace(projection, deepseek_stack=stack)
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
    kv_tokens = oracle["kv_tokens"]
    pairs = oracle["attention_pairs"]
    sampled = oracle["sampled"]
    mtp = oracle["mtp_enabled"]
    shapes = {
        "new_tokens": tokens,
        "kv_tokens": kv_tokens,
        "sampled": sampled,
        "mtp_enabled": mtp,
    }
    if family["id"] == "multi_token_prediction_head":
        flops = mtp * (
            family["fixed_flops_per_new_token"] * tokens
            + family[f"{oracle['phase']}_flops_per_attention_pair"] * pairs
            + family["flops_per_sampled_token"] * sampled
        )
        hbm_bytes = mtp * (
            family["static_hbm_bytes_when_enabled"]
            + family["hbm_bytes_per_kv_token_when_enabled"] * kv_tokens
        )
    else:
        flops = family.get("flops_per_new_token_per_layer", 0) * tokens * layers
        flops += (
            family.get(
                f"{oracle['phase']}_flops_per_attention_pair_per_layer",
                0,
            )
            * pairs
            * layers
        )
        flops += family.get("flops_per_sampled_token", 0) * sampled
        hbm_bytes = family.get("static_hbm_bytes_per_layer", 0) * layers
        hbm_bytes += family.get("static_hbm_bytes", 0)
        hbm_bytes += (
            family.get("hbm_bytes_per_kv_token_per_layer", 0)
            * kv_tokens
            * layers
        )
    return flops, hbm_bytes, tuple(shapes[axis] for axis in family["shape_axes"])


@pytest.mark.parametrize("framework_id", ["vllm", "sglang"])
def test_deepseek_v3_all_families_conserve_every_frozen_case(
    tmp_path: Path,
    framework_id: str,
) -> None:
    inventory = _extract(tmp_path, framework_id)
    freeze = _load(EXPECTATIONS_PATH)
    frozen_families = {
        family["id"]: family for family in freeze["inventory_contract"]["families"]
    }

    assert [family.family_id for family in inventory.kernel_families] == list(
        DEEPSEEK_V3_FAMILIES
    )
    assert len(inventory.cases) == 20
    for case, oracle in zip(inventory.cases, freeze["exact_case_oracles"], strict=True):
        assert case.case_id == oracle["case_id"]
        assert [item.family_id for item in case.kernel_projections] == list(
            DEEPSEEK_V3_FAMILIES
        )
        assert sum(
            item.logical_launch_count for item in case.kernel_projections
        ) == oracle["logical_visit_count"]
        assert sum(item.aggregate_flops for item in case.kernel_projections) == (
            oracle["aggregate_flops"]
        )
        assert sum(
            item.aggregate_hbm_bytes for item in case.kernel_projections
        ) == oracle["aggregate_hbm_bytes"]
        for projection in case.kernel_projections:
            expected_flops, expected_bytes, expected_shape = _expected_family_work(
                frozen_families[projection.family_id], oracle
            )
            assert projection.aggregate_flops == expected_flops
            assert projection.aggregate_hbm_bytes == expected_bytes
            assert projection.shape_vector.values == expected_shape


def test_deepseek_v3_frameworks_are_structurally_identical(tmp_path: Path) -> None:
    vllm = _extract(tmp_path, "vllm")
    sglang = _extract(tmp_path, "sglang")

    assert _neutral_inventory(vllm) == _neutral_inventory(sglang)
    assert vllm.record.record_id != sglang.record.record_id
    assert vllm.cases == sglang.cases


def test_deepseek_v3_sharded_projection_conserves_rank_classes(
    tmp_path: Path,
) -> None:
    suite = _authored_suite(b'{"model_type":"deepseek_v3"}')
    vllm = build_deepseek_deployment_projection(
        suite, _extract(tmp_path / "vllm", "vllm")
    )
    sglang = build_deepseek_deployment_projection(
        suite, _extract(tmp_path / "sglang", "sglang")
    )
    freeze = _load(EXPECTATIONS_PATH)["deployment_projection_contract"]

    assert vllm == sglang
    assert vllm["schema"] == DEEPSEEK_DEPLOYMENT_PROJECTION_SCHEMA
    assert canonical_bytes(vllm).endswith(b"}")
    assert vllm["expert_contract"]["per_expert_base_static_hbm_bytes"] == (
        freeze["per_expert_static_hbm_bytes_all_base_moe_layers"]
    )
    assert vllm["expert_contract"]["per_expert_layer_flops"] * 58 == (
        freeze["per_expert_flops_per_dispatch_all_base_moe_layers"]
    )
    for unit in vllm["units"]:
        classes = unit["static_rank_classes"]
        assert sum(
            row["rank_count"] * row["logical_experts_per_rank"]
            for row in classes
        ) == 256
        assert sum(
            row["rank_count"] * row["physical_slots_per_rank"]
            for row in classes
        ) == 288
        if unit["id"].startswith("sglang-"):
            assert unit["dynamic_projection_state"] == "exact-disclosed-workload"
            assert unit["case_projections"]
            for case in unit["case_projections"]:
                assert case["conservation"]["reference_base_flops"] == (
                    case["conservation"]["rank_class_base_flops"]
                )
                assert case["conservation"]["reference_mtp_flops"] == (
                    case["conservation"]["rank_class_mtp_flops"]
                )
        else:
            assert unit["dynamic_projection_state"] == (
                "not-claimed-no-disclosed-workload-shape"
            )
            assert unit["case_projections"] == []


@pytest.mark.parametrize("framework_id", ["vllm", "sglang"])
def test_deepseek_v3_extraction_is_byte_deterministic(
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
    ["q-rank", "schedule-order", "weight-block", "scope-exclusions", "mtp-layers"],
)
def test_deepseek_v3_stack_mutations_reject_before_writing(
    tmp_path: Path,
    mutation: str,
) -> None:
    with pytest.raises(ModelExtractionError, match="does not match the suite"):
        _extract(tmp_path, "vllm", projection_mutation=mutation)

    assert not (tmp_path / "vllm" / "steps.jsonl").exists()


def test_deepseek_v3_disabled_mtp_has_zero_work_bytes_and_visits(
    tmp_path: Path,
) -> None:
    inventory = _extract(tmp_path, "vllm")
    disabled = inventory.cases[0].kernel_projections[-1]
    enabled = inventory.cases[-1].kernel_projections[-1]

    assert (
        disabled.logical_launch_count,
        disabled.aggregate_flops,
        disabled.aggregate_hbm_bytes,
    ) == (0, 0, 0)
    assert enabled.logical_launch_count == 1
    assert enabled.aggregate_flops > 0
    assert enabled.aggregate_hbm_bytes > 0
    assert ModelKernelInventory.from_obj(inventory.to_obj()) == inventory


def test_deepseek_v3_missing_family_rejects_total_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_builder = extraction_module._deepseek_family_work

    def missing_family(*args: object, **kwargs: object):
        return real_builder(*args, **kwargs)[:-1]

    monkeypatch.setattr(extraction_module, "_deepseek_family_work", missing_family)
    with pytest.raises(ModelExtractionError, match="frozen family order"):
        _extract(tmp_path, "vllm")


def test_deepseek_v3_noninteger_family_work_rejects_total_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_builder = extraction_module._deepseek_family_work

    def fractional_family(*args: object, **kwargs: object):
        families = list(real_builder(*args, **kwargs))
        object.__setattr__(families[0], "flops", 0.5)
        return tuple(families)

    monkeypatch.setattr(extraction_module, "_deepseek_family_work", fractional_family)
    with pytest.raises(ModelExtractionError, match="exact nonnegative integer work"):
        _extract(tmp_path, "vllm")
