"""Lock the kernel_cycle_lut_v1 expectations before implementation or scoring."""

from __future__ import annotations

import json
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1] / "examples" / "kernel_cycle_lut_v1"
EXPECTATIONS_JSON = STUDY / "expectations.json"
EXPECTATIONS_MD = STUDY / "expectations.md"


def _freeze() -> dict:
    return json.loads(EXPECTATIONS_JSON.read_text(encoding="utf-8"))


def test_freeze_precedes_implementation_and_records_clean_tree() -> None:
    freeze = _freeze()

    assert freeze["schema"] == "simllm-study-freeze-v1"
    assert freeze["study"] == "kernel_cycle_lut_v1"
    assert freeze["authored_against"] == "7e5b1ce26104d5252f94ee94350947d874f844c9"
    assert freeze["working_tree_before_freeze"] == "clean"
    assert freeze["execution_scope"] == "retained-fixtures-only-no-gpu"


def test_lookup_key_and_routing_contract_are_exact() -> None:
    freeze = _freeze()

    assert freeze["lookup_schema"] == "simllm-kernel-cycle-lut-v1"
    assert freeze["lookup_key_order"] == [
        "framework_identity",
        "model_identity",
        "pool",
        "launch_mode",
        "parallelism",
        "shape",
        "routing_for_routed_families_only",
    ]
    contract = freeze["key_contract"]
    assert contract["decode_shape_fields"] == [
        "batch_size",
        "per_request_kv_lengths",
    ]
    assert contract["prefill_shape_fields"] == [
        "computed_new_tokens",
        "existing_context_tokens",
    ]
    assert contract["dense_routing_rule"] == "routing-field-is-forbidden"
    assert contract["routed_routing_rule"].startswith("routing-field-is-required")


def test_component_and_distribution_contracts_are_exact() -> None:
    freeze = _freeze()

    components = freeze["component_contract"]
    assert components["composition"] == "sum-kernels-of-max-compute-memory-plus-fixed"
    assert components["reconstruction_error_ps_max"] == 1
    assert components["unknown_memory_rule"].startswith("all-byte-and-bandwidth-fields-null")

    distribution = freeze["distribution_contract"]
    assert distribution["graph_minimum_replays"] == 256
    assert distribution["eager_minimum_replays"] == 64
    assert distribution["retained_fixture_expected_verdict"] == "insufficient-replays"
    assert distribution["device_model_consumes_observed_spread"] is True


def test_campaign_matrix_freezes_sixteen_kv_points_and_code_harvest() -> None:
    protocol = _freeze()["campaign_protocol"]

    assert protocol["pools"] == ["decode", "prefill"]
    assert protocol["launch_modes"] == ["cuda-graph", "eager"]
    assert len(protocol["kv_grid_basis_points"]) == 16
    assert protocol["kv_grid_basis_points"][0] == 100
    assert protocol["kv_grid_basis_points"][-1] == 10000
    assert protocol["kv_placement_modes"] == [
        "fresh-contiguous",
        "deliberately-fragmented",
    ]
    assert protocol["ncu_clock_control"] == "none"
    assert protocol["clean_harvest_runs"] == 2
    assert protocol["harvest_acceptance"] == "byte-identical-canonical-manifests"


def test_evidence_classes_do_not_mix_structural_guards_into_score() -> None:
    freeze = _freeze()
    evidence = freeze["evidence_classes"]
    relations = freeze["scored_relations"]
    guards = freeze["fatal_guards"]

    assert evidence == {
        "behavioral_relation_families": 1,
        "behavioral_parameterized_instances": 5,
        "fatal_guards": 7,
        "exact_oracle_rows": 0,
        "native_executables": 0,
    }
    assert [relation["id"] for relation in relations] == [
        "R1-cross-instrument-elapsed-agreement"
    ]
    assert relations[0]["instances"] == 5
    assert relations[0]["ratio_lower_ppm"] == 500000
    assert relations[0]["ratio_upper_ppm"] == 2000000
    assert [guard["id"] for guard in guards] == [f"G{index}" for index in range(1, 8)]


def test_freeze_closes_nothing_and_names_expected_residuals() -> None:
    closure = _freeze()["closure"]

    assert closure["closes"] == []
    assert closure["keeps_open"] == ["COMP-64"]
    assert closure["expected_residual_ids"] == ["COMP-65", "COMP-66"]
    assert set(closure["does_not_claim"]) == {
        "gpu-campaign-execution",
        "numerical-calibration",
        "compile-time-graph-inference",
        "program-counter-attribution",
    }


def test_freeze_documents_use_no_em_dash() -> None:
    assert "—" not in EXPECTATIONS_MD.read_text(encoding="utf-8")
    assert "—" not in EXPECTATIONS_JSON.read_text(encoding="utf-8")
