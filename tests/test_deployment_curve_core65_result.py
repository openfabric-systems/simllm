from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/deployment_curve_v1"


def _load(name: str):
    return json.loads((STUDY / name).read_text(encoding="utf-8"))


def test_inventory_is_total_and_reconstructs_the_retained_step() -> None:
    inventory = _load("core65_kernel_inventory.json")

    assert inventory["coverage"] == "TOTAL_46_OF_46_ROWS_NO_UNMAPPED_KERNEL"
    assert len(inventory["rows"]) == inventory["kernel_row_count"] == 46
    assert len({row["first_launch_order"] for row in inventory["rows"]}) == 46
    assert all(row["name"] and row["family"] for row in inventory["rows"])
    assert sum(row["captured_service_ps"] for row in inventory["rows"]) == (
        inventory["measured_service_ps"]
    ) == 1_875_680_000
    assert inventory["bucket_row_counts"] == {
        "all_layer_common": 9,
        "dense_only": 5,
        "first_layer_only": 3,
        "later_layer_attention": 3,
        "moe_only": 14,
        "step_or_output_once": 12,
    }
    assert inventory["bucket_service_ps"] == {
        "all_layer_common": 684_256_000,
        "dense_only": 397_696_000,
        "first_layer_only": 8_544_000,
        "later_layer_attention": 25_248_000,
        "moe_only": 203_616_000,
        "step_or_output_once": 556_320_000,
    }


def test_candidate_verdicts_separate_compute_population_and_memory() -> None:
    result = _load("core65_physical_binding_result.json")
    verdicts = result["candidate_verdicts"]

    assert verdicts["layer_type_composition"]["finding"] == {
        "dense_layers": 3,
        "moe_layers": 1,
        "proof": (
            "effective num_hidden_layers=4 with first_k_dense_replace=3 "
            "and moe_layer_freq=1"
        ),
        "state": "DECIDED_THREE_DENSE_THEN_ONE_MOE",
    }
    expert = verdicts["expert_population"]["finding"]
    assert expert["assignment_tracked_compute_scale"] == "1/9_inherited_not_refit"
    assert expert["expert_count_or_per_layer_resident_weight_scale"] == "1/64"
    assert expert["full_model_layer_adjusted_resident_routed_weight_scale"] == (
        "29/32"
    )
    weight = verdicts["weight_read_volume"]["finding"]
    assert weight["captured_reduced4_tp1_resident"] == 15_116_101_504
    assert weight["naive_captured_resident_scaled_by_61_over_4"] == 230_520_547_936
    assert weight["ep72_declared_physical_per_rank"] == 27_446_643_040
    assert weight["actual_capture_per_step_read_bytes"] == "UNAVAILABLE"
    assert weight["actual_ep72_per_step_read_bytes"] == "UNAVAILABLE"
    assert verdicts["other_kernel_counterparts"]["finding"] == {
        "capture_only_vllm_routed_rows": [96, 98, 100, 101, 104],
        "ep72_required_absent_families": [
            "deepep_dispatch_a",
            "deepep_dispatch_b",
            "deepep_combine_a",
            "deepep_combine_b",
        ],
    }
    assert all(not row["numeric_calibration_movement_admissible"] for row in verdicts.values())


def test_publication_is_null_and_conditional_diagnostic_is_rejected() -> None:
    result = _load("core65_physical_binding_result.json")

    assert result["status"] == "PROTOCOL_VOID_NULL_MOVEMENT_EXACT_HARDWARE_REMAINDER"
    assert result["calibration_only"] == {
        "anchor_tokens_per_second_per_node": "22282",
        "classification": "UNDERCORRECTION",
        "corrected_step_ps_round_half_up": 26_821_286_365,
        "final_prediction_tokens_per_second_per_node": "9544.657796",
        "final_signed_residual_percent": "-57.164268",
        "prediction_movement_tokens_per_second_per_node": "0.000000",
        "signed_difference_from_anchor_tokens_per_second_per_node": (
            "-12737.342204"
        ),
        "signed_residual_movement_percentage_points": "0.000000",
    }
    diagnostic = result["conditional_frequency_only_diagnostic"]
    assert diagnostic["admissible_for_calibration"] is False
    assert diagnostic["step_ps"] == {
        "denominator": 3,
        "numerator": 50_794_696_000,
        "round_half_up": 16_931_565_333,
    }
    assert diagnostic["prediction_movement_tokens_per_second_per_node"] == (
        "5575.031079"
    )


def test_protocol_preservation_registry_and_hardware_remainder_are_literal() -> None:
    result = _load("core65_physical_binding_result.json")
    remainder = _load("core66_hardware_remainder.json")
    registry = (ROOT / "docs/modules/core.md").read_text(encoding="utf-8")
    report = (STUDY / "core65_physical_binding_result.md").read_text(
        encoding="utf-8"
    )

    assert result["access"]["forbidden_access_incident_count"] == 2
    assert result["access"]["forbidden_access_ledger_empty"] is False
    assert result["access"]["literal_protocol_satisfied"] is False
    assert result["preservation_lock"]["checked_count"] == 154
    assert result["registry_disposition"] == {
        "core65": "REMAINS_OPEN_PROTOCOL_VOID_AND_PHYSICAL_SERVICE_UNRESOLVED",
        "exact_remainder_id": "CORE-66",
        "reserved_id_free_on_base_main": True,
    }
    assert all(value is False for value in result["scope"].values())
    assert remainder["task"] == "CORE-66"
    assert "--tp-size 72 --dp-size 72 --ep-size 72" in remainder["command"]
    assert report.index("## Total retained kernel inventory") < report.index(
        "## Candidate verdicts"
    ) < report.index("## Published signed movement")
    assert "CORE-65 remains open" in registry
    assert "- CORE-66 (Precision; P0; L):" in registry
