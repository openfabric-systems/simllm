"""K3 extraction joins and rejection boundaries without framework imports."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from simllm.calibration.extraction import (
    FrameworkConfigurationProjection,
    extract_model_inventory,
    load_extraction_suite,
)
from simllm.calibration.kimi_k3 import validate_kimi_k3_native_contract
from simllm.calibration.model_inventory import FrameworkIdentity, ModelKernelInventory
from simllm.compute.kimi_k3 import KimiK3Spec

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "offline/calibration/suites/kimi-k3-text-v1-frameworks-2026-09-09/suite.json"
CONFIG = ROOT / "tests/fixtures/kimi_k3_config.json"


def arguments(tmp_path, framework="vllm"):
    raw = SUITE.read_bytes()
    suite, model = load_extraction_suite(raw)
    declared = next(item for item in suite["frameworks"] if item["id"] == framework)
    identity = FrameworkIdentity(framework, declared["version"], declared["source_commit"],
                                 declared.get("source_tree"), "unit-config-seam")
    projection = FrameworkConfigurationProjection(
        identity, "unit-native-projection", declared["architecture_binding"],
        declared["text_implementation"], kimi_k3_stack=model.geometry,
    )
    checkpoint = tmp_path / model.revision
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "config.json").write_bytes(CONFIG.read_bytes())
    return {"suite_raw": raw, "framework": identity, "checkpoint_root": checkpoint,
                "framework_dims": None, "framework_projection": projection,
                "step_records_path": tmp_path / "steps.jsonl"}


@pytest.fixture(scope="module")
def inventory(tmp_path_factory):
    return extract_model_inventory(**arguments(tmp_path_factory.mktemp("k3-extraction")))


def test_inventory_preserves_unknown_demands_and_total_case_visits(inventory):
    assert ModelKernelInventory.from_obj(inventory.to_obj()) == inventory
    assert isinstance(inventory.model.geometry, KimiK3Spec)
    assert len(inventory.cases) == 12
    for case in inventory.cases:
        assert sum(p.logical_launch_count for p in case.kernel_projections) == (3244 if case.phase == "prefill" else 3268)
        for projection in case.kernel_projections:
            assert projection.scope == "logical-operator"
            if projection.logical_launch_count:
                assert projection.aggregate_hbm_bytes is None
            else:
                assert projection.aggregate_flops == projection.aggregate_hbm_bytes == 0
    assert any(p.aggregate_flops is None for p in inventory.cases[0].kernel_projections)


def test_inventory_cannot_turn_logical_regions_into_physical_kernels(inventory):
    obj = inventory.to_obj()
    for case in obj["cases"]:
        for projection in case["kernel_projections"]:
            projection.update(scope="kernel-region", aggregate_flops=0, aggregate_hbm_bytes=0)
    with pytest.raises(ValueError, match="scope disagrees"):
        ModelKernelInventory.from_obj(obj)


def test_old_uniform_inventory_cannot_claim_unknown_logical_demand():
    for path in sorted((ROOT / "offline/calibration/model-inventories").glob("*.json")):
        obj = json.loads(path.read_bytes())
        if "cases" in obj and "layer_types" not in obj["model"]["geometry"]:
            obj["cases"][0]["kernel_projections"][0].update(scope="logical-operator", aggregate_flops=None, aggregate_hbm_bytes=None)
            with pytest.raises(ValueError, match="scope disagrees"):
                ModelKernelInventory.from_obj(obj)
            break
    else:
        pytest.fail("missing legacy compatibility inventory")


@pytest.mark.parametrize("mutation", [
    lambda s: s["base_parallelism"].update(expert=4),
    lambda s: s["base_parallelism"].update(tensor=True),
    lambda s: s["phase_scope"].update(mode="multimodal"),
    lambda s: s["phase_scope"]["optional_components"].append("vision"),
    lambda s: s["graph_cells"][0].update(mtp_enabled=True),
    lambda s: s["graph_cells"][3].update(family="moe-communication-decode", expert_participants=4, parallelism_override={"expert": 4}),
    lambda s: s["reference_model"].update(architecture="KimiK2ForCausalLM"),
])
def test_undeclared_suite_modes_fail_before_record_publication(tmp_path, mutation):
    kwargs = arguments(tmp_path)
    suite = json.loads(kwargs["suite_raw"])
    mutation(suite)
    kwargs["suite_raw"] = json.dumps(suite).encode()
    with pytest.raises((TypeError, ValueError)):
        extract_model_inventory(**kwargs)
    assert not kwargs["step_records_path"].exists()


def test_native_geometry_mismatch_and_generic_dimensions_are_rejected(tmp_path):
    kwargs = arguments(tmp_path)
    projection = kwargs["framework_projection"]
    changed = replace(projection.kimi_k3_stack, hidden_size=8192)
    kwargs["framework_projection"] = replace(projection, kimi_k3_stack=changed)
    with pytest.raises(ValueError, match="native K3 geometry"):
        extract_model_inventory(**kwargs)
    kwargs["framework_projection"] = projection
    kwargs["framework_dims"] = object()
    with pytest.raises(ValueError, match="uniform ModelDims"):
        extract_model_inventory(**kwargs)
    assert not kwargs["step_records_path"].exists()


@pytest.mark.parametrize("mutation", [
    lambda t: t["linear_attn_config"]["kda_layers"].append(93),
    lambda t: t["linear_attn_config"]["full_attn_layers"].pop(),
    lambda t: t.update(num_nextn_predict_layers=1),
    lambda t: t["quantization_config"]["config_groups"]["group_0"]["weights"].update(group_size=64),
    lambda t: t["quantization_config"]["config_groups"]["group_0"]["weights"].update(dynamic=True),
    lambda t: t["quantization_config"]["ignore"].pop(),
])
def test_native_policy_changes_are_not_silently_normalized(mutation):
    text = json.loads(CONFIG.read_bytes())["text_config"]
    layers = tuple("mla" if i + 1 in text["linear_attn_config"]["full_attn_layers"] else "kda" for i in range(93))
    assert validate_kimi_k3_native_contract(SimpleNamespace(**text), layers, quant_method="compressed-tensors") == "mxfp4-e2m1-group32-e8m0"
    modified = copy.deepcopy(text)
    mutation(modified)
    with pytest.raises((TypeError, ValueError)):
        validate_kimi_k3_native_contract(SimpleNamespace(**modified), layers, quant_method="compressed-tensors")


def test_framework_specific_native_quantization_label_is_exact():
    text = json.loads(CONFIG.read_bytes())["text_config"]
    layers = tuple("mla" if i + 1 in text["linear_attn_config"]["full_attn_layers"] else "kda" for i in range(93))
    with pytest.raises(ValueError, match="encoding"):
        validate_kimi_k3_native_contract(SimpleNamespace(**text), layers, quant_method="mxfp4")
    text["quantization_config"]["quant_method"] = "mxfp4"
    assert validate_kimi_k3_native_contract(SimpleNamespace(**text), layers, quant_method="mxfp4") == "mxfp4-e2m1-group32-e8m0"
