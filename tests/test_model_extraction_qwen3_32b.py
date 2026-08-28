from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simllm.calibration.extraction import ORDERED_FAMILIES, load_extraction_suite
from simllm.calibration.model_inventory import ModelKernelInventory

REPOSITORY = Path(__file__).resolve().parents[1]
SUITE = (
    REPOSITORY
    / "offline/calibration/suites"
    / "qwen3-32b-fp8-text-v1-frameworks-2026-08-28/suite.json"
)
INVENTORIES = {
    "vllm": REPOSITORY
    / "offline/calibration/model-inventories"
    / "c8832ba8ba21e49517b6b74e89554c2abdb0d9e76530f647a7849f3f8448ec56.json",
    "sglang": REPOSITORY
    / "offline/calibration/model-inventories"
    / "51740b52625002a964e75fddb679e9f8394a08a7d7c62556d2535c3bc60515e3.json",
}


def _load_inventory(path: Path) -> ModelKernelInventory:
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == path.stem
    inventory = ModelKernelInventory.from_obj(json.loads(raw))
    assert inventory.record.canonical == raw
    return inventory


def _neutral(inventory: ModelKernelInventory) -> dict[str, object]:
    value = inventory.to_obj()
    value.pop("framework")
    value["implementation_identity"].pop("join_tasks")
    return value


def test_qwen3_32b_suite_locks_checkpoint_identity_and_fg2_geometry() -> None:
    raw = SUITE.read_bytes()
    suite, identity = load_extraction_suite(raw)
    assert hashlib.sha256(raw).hexdigest() == (
        "f0830d3692029dca5464af6932f273d7147258d72edaf5986aada37a0ba25435"
    )
    assert identity.name == "Qwen/Qwen3-32B-FP8"
    assert identity.revision == "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
    assert identity.config_sha256 == (
        "e546dacd2c772660270233f5579e9ab923cc2a7ec5ed3c58c27c2bc62cbf5169"
    )
    assert identity.weight_sha256 == (
        "80ce47beb772bd057b497ba29aecc6202f80fea8d87e183ba86dff17bc034ef3"
    )
    assert identity.weight_bytes == 34_322_567_640
    assert identity.geometry.to_obj() == {
        "layers": 64,
        "hidden_size": 5120,
        "intermediate_size": 25600,
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_size": 128,
        "num_experts": 0,
        "top_k": 0,
        "vocab_size": 151936,
    }
    assert suite["reference_model"]["parameter_count"] == 32_762_123_264
    assert len(suite["reference_model"]["weight_shards"]) == 7


def test_qwen3_32b_inventories_are_complete_and_framework_neutral() -> None:
    inventories = {
        framework: _load_inventory(path) for framework, path in INVENTORIES.items()
    }
    assert {item.framework.framework_id for item in inventories.values()} == {
        "vllm",
        "sglang",
    }
    for inventory in inventories.values():
        assert inventory.suite.case_count == 15
        assert tuple(item.family_id for item in inventory.kernel_families) == (
            ORDERED_FAMILIES
        )
        assert all(
            sum(item.logical_launch_count for item in case.kernel_projections) == 257
            for case in inventory.cases
        )
    assert _neutral(inventories["vllm"]) == _neutral(inventories["sglang"])


def test_qwen3_32b_inventory_prices_frozen_fp8_work() -> None:
    inventory = _load_inventory(INVENTORIES["vllm"])
    case = next(item for item in inventory.cases if item.case_id == "db-train-b64-c2048")
    work = {item.family_id: item for item in case.kernel_projections}
    assert work["attn_gemm"].aggregate_flops == 773_094_113_280
    assert work["attn_gemm"].aggregate_hbm_bytes == 6_039_797_760
    assert work["attn_score"].aggregate_flops == 274_743_689_216
    assert work["mlp_gemm"].aggregate_flops == 3_221_225_472_000
    assert work["mlp_gemm"].aggregate_hbm_bytes == 25_165_824_000
    assert work["lm_head"].aggregate_flops == 99_572_776_960
    assert work["lm_head"].aggregate_hbm_bytes == 777_912_320
    assert work["kv_read"].aggregate_hbm_bytes == 34_359_738_368
    assert sum(item.aggregate_hbm_bytes for item in case.kernel_projections) == (
        66_343_272_448
    )
