from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simllm.calibration.canonical import canonical_bytes, strict_json_loads
from simllm.calibration.deepseek_deployment import (
    DEEPSEEK_DEPLOYMENT_PROJECTION_SCHEMA,
)
from simllm.calibration.extraction import DEEPSEEK_V3_FAMILIES
from simllm.calibration.model_inventory import ModelKernelInventory

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/model_extraction_deepseek_v3_v1/RESULTS.md"
COVERAGE = ROOT / "docs/design/calibration-coverage.md"
COMPUTE = ROOT / "docs/modules/compute.md"
VLLM = ROOT / "docs/modules/adapters-vllm.md"
SGLANG = ROOT / "docs/modules/adapters-sglang.md"
LEDGER = ROOT / "docs/task-ledger.json"
INVENTORIES = {
    "vllm": "2209f1bdb2055007d935d5e64e79e9cc89d36585415eb220dc90be9f333f53ff",
    "sglang": "5f3f92884fd028532aef0eaa884218a865060780dac92dc02e910beb260967a3",
}
PROJECTION_ID = "ee154ed5f07c104269df9cf60d8730b8c6dced0ccf619fb7ff7146ec2ddfd5a2"


def _load_inventory(record_id: str) -> ModelKernelInventory:
    path = ROOT / "offline/calibration/model-inventories" / f"{record_id}.json"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record_id
    inventory = ModelKernelInventory.from_obj(strict_json_loads(raw))
    assert inventory.record.canonical == raw
    assert inventory.record.record_id == record_id
    return inventory


def _neutral(inventory: ModelKernelInventory) -> dict[str, object]:
    value = inventory.to_obj()
    value.pop("framework")
    value["implementation_identity"].pop("join_tasks")
    return value


def _task_lines(path: Path, identifier: str) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith(f"- {identifier} ")
    ]


def test_deepseek_published_inventories_are_complete_and_content_addressed() -> None:
    loaded = {
        framework: _load_inventory(record_id)
        for framework, record_id in INVENTORIES.items()
    }

    for framework, inventory in loaded.items():
        assert inventory.framework.framework_id == framework
        assert len(inventory.cases) == 20
        assert [family.family_id for family in inventory.kernel_families] == list(
            DEEPSEEK_V3_FAMILIES
        )
        visits = [
            sum(item.logical_launch_count for item in case.kernel_projections)
            for case in inventory.cases
        ]
        assert visits[:-1] == [666] * 19
        assert visits[-1] == 667
    assert _neutral(loaded["vllm"]) == _neutral(loaded["sglang"])


def test_deepseek_published_projection_conserves_every_rank_class() -> None:
    path = (
        ROOT
        / "offline/calibration/deployment-projections"
        / f"{PROJECTION_ID}.json"
    )
    raw = path.read_bytes()
    projection = strict_json_loads(raw)

    assert hashlib.sha256(raw).hexdigest() == PROJECTION_ID
    assert canonical_bytes(projection) == raw
    assert projection["schema"] == DEEPSEEK_DEPLOYMENT_PROJECTION_SCHEMA
    assert projection["expert_contract"] == {
        "base_moe_layers": 58,
        "logical_experts": 256,
        "per_expert_base_static_hbm_bytes": 2_554_954_752,
        "per_expert_layer_flops": 88_080_384,
        "per_expert_mtp_static_hbm_bytes": 44_050_944,
        "physical_slots": 288,
        "redundancy_rule": (
            "physical residency only; duplicate slots create no logical work"
        ),
        "redundant_physical_slots": 32,
        "top_k": 8,
    }
    expected_physical_bytes = {
        "sglang-prefill-ep32-dp-attention": 40_221_416_800,
        "sglang-decode-ep72-dp-attention": 27_446_643_040,
        "deepseek-production-prefill-ep32": 40_221_416_800,
        "deepseek-production-decode-ep144": 22_336_733_536,
    }
    assert len(projection["units"]) == 4
    for unit in projection["units"]:
        classes = unit["static_rank_classes"]
        assert sum(
            row["rank_count"] * row["logical_experts_per_rank"]
            for row in classes
        ) == 256
        assert sum(
            row["rank_count"] * row["physical_slots_per_rank"]
            for row in classes
        ) == 288
        assert {
            row["base"]["physical_total_hbm_bytes_per_rank"] for row in classes
        } == {expected_physical_bytes[unit["id"]]}
        for case in unit["case_projections"]:
            assert case["conservation"]["reference_base_flops"] == (
                case["conservation"]["rank_class_base_flops"]
            )
            assert case["conservation"]["reference_mtp_flops"] == (
                case["conservation"]["rank_class_mtp_flops"]
            )


def test_deepseek_results_report_the_nonvoid_deciding_numbers() -> None:
    text = RESULTS.read_text(encoding="utf-8")

    assert "No fatal guard was violated" in text
    assert "COMP-67 closes" in text
    assert "666 logical family visits" in text
    assert "667" in text
    assert "40 * 288 + 32 * 216 = 18,432" in text
    assert "40 * 144 + 32 * 108 = 9,216" in text
    assert all(record_id in text for record_id in INVENTORIES.values())
    assert PROJECTION_ID in text
    assert "daef040982bf277c89b4dda7fc093358cb3cdff947e70c58c6238a9b9b4f87dd" in text


def test_deepseek_is_the_only_published_state_in_its_coverage_cell() -> None:
    rows = [
        line
        for line in COVERAGE.read_text(encoding="utf-8").splitlines()
        if line.startswith("| DeepSeek-V3 |")
    ]

    assert len(rows) == 1
    assert "model_extraction_deepseek_v3_v1" in rows[0]
    assert "Planned" not in rows[0]
    assert all(record_id in rows[0] for record_id in INVENTORIES.values())
    assert PROJECTION_ID in rows[0]


def test_comp67_closure_and_reserved_residuals_are_recorded_once() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))

    assert _task_lines(COMPUTE, "COMP-67") == []
    assert ledger["closed"].count("COMP-67") == 1
    assert len(_task_lines(COMPUTE, "COMP-69")) == 1
    assert len(_task_lines(COMPUTE, "COMP-70")) == 1
    assert len(_task_lines(VLLM, "VLLM-38")) == 1
    assert len(_task_lines(SGLANG, "SGL-34")) == 1
