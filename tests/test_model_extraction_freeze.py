"""Lock the expectations-only COMP-54 model extraction study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
STUDY = REPOSITORY / "examples" / "model_extraction_v1"
EXPECTATIONS = STUDY / "expectations.json"
SUITE = (
    REPOSITORY
    / "offline"
    / "calibration"
    / "suites"
    / "transformer-dag-v1"
    / "suite.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_model_extraction_freeze_retains_only_authored_values() -> None:
    assert EXPECTATIONS.is_file()
    assert (STUDY / "expectations.md").is_file()
    freeze = _load(EXPECTATIONS)
    assert freeze["schema"] == "simllm-model-extraction-expectations-v1"
    assert freeze["study"] == "model-extraction-v1"
    assert freeze["working_tree_before_freeze"] == "clean"
    assert "results" not in freeze
    assert "observed_inventories" not in freeze
    assert freeze["closure"] == {
        "closes": [],
        "keeps_open": ["COMP-54"],
        "physical_followups": ["COMP-6", "VLLM-12", "SGL-10"],
    }


def test_frozen_suite_bytes_and_model_identity_are_exact() -> None:
    freeze = _load(EXPECTATIONS)
    suite = freeze["suite"]
    assert isinstance(suite, dict)
    assert hashlib.sha256(SUITE.read_bytes()).hexdigest() == suite["sha256"]
    assert suite["case_count"] == 15
    assert freeze["model"] == {
        "name": "ibm-granite/granite-3.0-1b-a400m-instruct",
        "revision": "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445",
        "config_sha256": (
            "ca4bb3a5c1bdef988ab413e0d731640446da65316e4ed16de3666cd96ecc3a0b"
        ),
        "weight_sha256": (
            "f7ae1cee56a9ea6c5360437b1c0407f8d84816b2cc75470f4e7e5236fa2a07dc"
        ),
        "weight_bytes": 2_669_283_096,
        "layers": 24,
    }


def test_freeze_varies_three_parameters_and_separates_evidence_classes() -> None:
    freeze = _load(EXPECTATIONS)
    sweep = freeze["sweep"]
    assert isinstance(sweep, dict)
    parameters = sweep["parameters"]
    assert isinstance(parameters, list)
    assert len(parameters) == 3
    assert all(len(parameter["values"]) == 5 for parameter in parameters)
    assert freeze["evidence_classes"] == [
        "run-configuration",
        "exact-oracle",
        "behavioral-relation",
        "structural-invariant",
        "native-test-executable",
    ]
    relations = freeze["relations"]
    assert isinstance(relations, list)
    assert [relation["id"] for relation in relations] == [
        "R1-exact-family-projection",
        "R2-layer-launch-scaling",
        "R3-shape-axis-sensitivity",
        "R4-template-equivalence-classes",
        "R5-byte-determinism",
    ]


def test_inventory_contract_freezes_order_counts_and_unknown_markers() -> None:
    freeze = _load(EXPECTATIONS)
    contract = freeze["inventory_contract"]
    assert isinstance(contract, dict)
    assert contract["ordered_families"] == [
        "attn_gemm",
        "attn_score",
        "mlp_gemm",
        "lm_head",
        "kv_read",
    ]
    assert contract["logical_launch_count_formula"] == "4 * layers + 1"
    assert contract["logical_launch_count_at_frozen_layers"] == 97
    assert contract["physical_fields"] == [
        "code_object_hashes",
        "observed_launches",
    ]
    assert contract["physical_field_state"] == "absent-by-design"
    assert contract["physical_field_value"] is None


def test_negative_controls_and_fatal_guards_are_not_scored_relations() -> None:
    freeze = _load(EXPECTATIONS)
    relations = freeze["relations"]
    guards = freeze["fatal_guards"]
    controls = freeze["negative_controls"]
    assert isinstance(relations, list)
    assert isinstance(guards, list)
    assert isinstance(controls, list)
    relation_ids = {relation["id"] for relation in relations}
    assert relation_ids.isdisjoint(guards)
    assert relation_ids.isdisjoint(controls)
    assert len(guards) == 8
    assert len(controls) == 4
