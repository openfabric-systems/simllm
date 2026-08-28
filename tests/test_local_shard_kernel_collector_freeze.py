"""Lock local-shard collector expectations before implementation or execution."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "local_shard_kernel_collector_v1"


def _expectations() -> dict[str, object]:
    return json.loads((STUDY / "expectations.json").read_text(encoding="utf-8"))


def test_freeze_names_existing_owner_and_preimplementation_chronology() -> None:
    frozen = _expectations()

    assert frozen["schema"] == "simllm-local-shard-kernel-collector-expectations-v1"
    assert frozen["owner_task"] == "COMP-50"
    assert frozen["chronology"] == "expectations-only-before-implementation-and-run"


def test_freeze_separates_logical_parallelism_physical_shard_and_architecture() -> None:
    frozen = _expectations()
    identity = frozen["identity_contract"]

    assert identity["logical_parallelism_is_separate_from_physical_shard"] is True
    assert identity["framework_model_revision_device_isa_and_numeric_format_are_required"] is True
    assert "target-device-isa-differs-from-request" in frozen["fatal_guards"]
    assert "distributed-collective-timing-from-an-isolated-shard" in frozen["scope"][
        "excluded"
    ]


def test_freeze_declares_two_parameter_grid_and_non_effect() -> None:
    frozen = _expectations()
    grid = frozen["conformance_grid"]

    assert grid["tensor_parallel"] == [1, 4]
    assert grid["batch_size"] == [1, 8]
    assert len(grid["expected_relations"]) == 5
    assert "No GPU constant" in frozen["project_non_effect"]
