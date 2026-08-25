from __future__ import annotations

import hashlib
from pathlib import Path

from simllm.calibration.canonical import strict_json_loads
from simllm.calibration.model_inventory import ModelKernelInventory

REPOSITORY = Path(__file__).resolve().parents[1]
REGISTRY = REPOSITORY / "offline" / "calibration" / "model-inventories"
EXPECTED = {
    "vllm": "e74e995a89588a304aa852593d3505cfab9a94d2c068c82dbe9c776da7119af9",
    "sglang": "147fe4398d5615afe7954c9199134de37f706da2cecda8fc37d6514ad936c54c",
}


def _load(record_id: str) -> ModelKernelInventory:
    raw = (REGISTRY / f"{record_id}.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record_id
    inventory = ModelKernelInventory.from_obj(strict_json_loads(raw))
    assert inventory.record.canonical == raw
    return inventory


def test_granite_inventory_artifacts_are_canonical_and_content_addressed() -> None:
    inventories = {
        framework: _load(record_id) for framework, record_id in EXPECTED.items()
    }
    for framework, inventory in inventories.items():
        assert inventory.framework.framework_id == framework
        assert inventory.suite.suite_id == "transformer-dag-v1"
        assert inventory.suite.case_count == 15
        assert len(inventory.cases) == 15
        assert {
            sum(item.logical_launch_count for item in case.kernel_projections)
            for case in inventory.cases
        } == {97}


def test_framework_artifacts_have_identical_structure_denominators() -> None:
    values = []
    for record_id in EXPECTED.values():
        value = _load(record_id).to_obj()
        value.pop("framework")
        value["implementation_identity"].pop("join_tasks")
        values.append(value)
    assert values[0] == values[1]
