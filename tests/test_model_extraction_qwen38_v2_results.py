from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simllm.calibration.canonical import strict_json_loads
from simllm.calibration.extraction import QWEN_GATED_DELTA_NET_FAMILIES
from simllm.calibration.model_inventory import ModelKernelInventory

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/model_extraction_qwen38_v2/RESULTS.md"
COVERAGE = ROOT / "docs/design/calibration-coverage.md"
COMPUTE = ROOT / "docs/modules/compute.md"
LEDGER = ROOT / "docs/task-ledger.json"
INVENTORIES = {
    "vllm": "77ea0abb4803d2cab5689f6893563e9a973f0e29160ec1975f1c76b3046e30d1",
    "sglang": "9d1c6164d149b98a4d019bee31d8d3a7ce3ee38cb96b7e0bab03393fde3d4747",
}


def _load_inventory(record_id: str) -> tuple[ModelKernelInventory, bytes]:
    path = ROOT / "offline/calibration/model-inventories" / f"{record_id}.json"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record_id
    inventory = ModelKernelInventory.from_obj(strict_json_loads(raw))
    assert inventory.record.canonical == raw
    assert inventory.record.record_id == record_id
    return inventory, raw


def _neutral(inventory: ModelKernelInventory) -> dict[str, object]:
    value = json.loads(json.dumps(inventory.to_obj()))
    value.pop("framework")
    value["implementation_identity"].pop("join_tasks")
    return value


def test_qwen38_v2_published_inventories_are_complete_and_content_addressed() -> None:
    loaded = {
        framework: _load_inventory(record_id)[0]
        for framework, record_id in INVENTORIES.items()
    }

    for framework, inventory in loaded.items():
        assert inventory.framework.framework_id == framework
        assert len(inventory.cases) == 15
        assert [family.family_id for family in inventory.kernel_families] == list(
            QWEN_GATED_DELTA_NET_FAMILIES
        )
        assert all(
            sum(
                projection.logical_launch_count
                for projection in case.kernel_projections
            )
            == 449
            for case in inventory.cases
        )
    assert _neutral(loaded["vllm"]) == _neutral(loaded["sglang"])


def test_qwen38_v2_results_report_the_frozen_nonvoid_outcome() -> None:
    text = RESULTS.read_text(encoding="utf-8")

    assert "bf14d7563bdb52a0c8052309f477a022f1951cc4" in text
    assert "a0345119455e65ba9d59e26d6b78f1e5d43f25f919a7438c24a114b6b193bf02" in text
    assert all(record_id in text for record_id in INVENTORIES.values())
    assert "No fatal guard was violated" in text
    assert "COMP-62 closes" in text
    assert "COMP-54 and COMP-59 stay open" in text
    assert "449 logical family visits" in text


def test_qwen38_v2_is_the_only_published_state_in_the_qwen_coverage_cell() -> None:
    rows = [
        line
        for line in COVERAGE.read_text(encoding="utf-8").splitlines()
        if line.startswith("| Qwen3.8-27B |")
    ]

    assert len(rows) == 1
    assert "model_extraction_qwen38_v2" in rows[0]
    assert "model_extraction_qwen38_v1" not in rows[0]
    assert all(record_id in rows[0] for record_id in INVENTORIES.values())
    assert "COMP-54 stays open for the Kimi K3 structure half" in rows[0]


def test_comp62_closure_is_recorded_once() -> None:
    open_lines = [
        line
        for line in COMPUTE.read_text(encoding="utf-8").splitlines()
        if line.startswith("- COMP-62 ")
    ]
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))

    assert open_lines == []
    assert ledger["closed"].count("COMP-62") == 1
