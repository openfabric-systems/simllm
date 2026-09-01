from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/deployment_curve_v1"


def _result() -> dict:
    return json.loads(
        (STUDY / "core66_decode_kernel_ladder_result.json").read_text(
            encoding="utf-8"
        )
    )


def test_published_confirmation_follows_freeze_and_passes_residual() -> None:
    result = _result()
    confirmation = result["published_confirmation"]

    assert result["status"] == "PASS"
    assert result["chronology"]["expectations_commit"] == "7919f7b"
    assert result["chronology"]["confirmatory_commit"].startswith("ae158a2")
    assert result["chronology"]["expectations_preceded_confirmation"] is True
    assert confirmation["cold_native_service_ps"] == 1_290_176_000
    assert confirmation["predicted_graph_service_ps"] == 1_343_872_000
    assert confirmation["composition_residual_ps"] == -53_696_000
    assert abs(confirmation["composition_residual_ps"]) <= confirmation[
        "tolerance_ps"
    ]


def test_evidence_classes_remain_separate_and_complete() -> None:
    evidence = _result()["evidence_classes"]

    assert len(evidence["behavioral_relations"]) == 3
    assert all(evidence["behavioral_relations"].values())
    assert len(evidence["fatal_structural_guards"]) == 8
    assert all(evidence["fatal_structural_guards"].values())
    assert evidence["preservation_locks"] == {"passed": 7, "required": 7}
    assert evidence["run_configuration_is_unscored"] is True


def test_rung_zero_has_twelve_roofline_bound_kernel_families() -> None:
    rung = _result()["rungs"]["rung_0_individual_kernels"]
    families = rung["families"]

    assert len(families) == 12
    assert rung["roofline"] == {
        "bf16_peak_flops_per_second": 494_000_000_000_000,
        "fp8_peak_flops_per_second": 989_000_000_000_000,
        "hbm_bytes_per_second": 4_000_000_000_000,
    }
    assert all(row["cold_service_ps"] >= row["roofline_floor_ps"] for row in families)
    assert all(row["raw_service_ps"] > row["subtracted_service_ps"] for row in families)
    assert {row["family"] for row in families} >= {
        "MLA attention core and KV read",
        "routed expert, one assignment",
        "LM head",
    }


def test_fused_layers_and_contention_are_not_mixed() -> None:
    rungs = _result()["rungs"]
    fused = rungs["rung_1_fused_layers"]
    contention = rungs["rung_2_contention"]

    assert fused["dense"]["cold_native_service_ps"] == 308_256_000
    assert fused["dense"]["native_fusion_delta_ps"] == -8_064_000
    assert fused["moe"]["cold_native_service_ps"] == 415_648_000
    assert fused["moe"]["native_fusion_delta_ps"] == 163_712_000
    assert fused["stream_count"] == 1
    assert contention["counter_status"] == "DENIED_ERR_NVGPUCTRPERM"
    assert "absolute MLA service was not repeat-independent" in contention[
        "refutation"
    ]
    lm_head = next(
        row for row in contention["factors"] if row["family"] == "LM head"
    )
    assert lm_head["width_2"] == "2.004"
    assert lm_head["width_4"] == "2.033"


def test_both_mega_kernel_shapes_fit_without_a_fitted_correction() -> None:
    rung = _result()["rungs"]["rung_3_mega_kernel"]
    published = rung["published_four_layer_confirmation"]
    deeper = rung["deeper_all_moe_diagnostic"]

    assert published["status"] == "PASS"
    assert published["node_count"] == 100
    assert abs(published["composition_residual_ps"]) <= 67_193_600
    assert deeper["node_count"] == 173
    assert abs(deeper["composition_residual_ps"]) <= round(
        0.05 * deeper["predicted_service_ps"]
    )


def test_layer_model_reconstructs_services_and_calibration_endpoints() -> None:
    result = _result()
    model = result["layer_model"]
    symbols = model["symbols_ps"]

    assert symbols["C_common"] + symbols["D_dense_specific"] == model[
        "dense_layer_service_ps"
    ]
    assert (
        symbols["C_common"]
        + symbols["N_moe_nonrouted_specific"]
        + symbols["G_routed_group_at_R256_A256"]
        == model["moe_layer_service_ps"]
    )

    c = Decimal(symbols["C_common"])
    d = Decimal(symbols["D_dense_specific"])
    n = Decimal(symbols["N_moe_nonrouted_specific"])
    g = Decimal(symbols["G_routed_group_at_R256_A256"])
    fixed = Decimal(symbols["F_per_step_fixed"])
    assignment_g = g * (Decimal(256) / Decimal(9)) / Decimal(256)
    resident_g = g * Decimal(4) / Decimal(256)
    assignment_step = Decimal(61) * c + Decimal(3) * d + Decimal(58) * (
        n + assignment_g
    ) + fixed
    resident_step = Decimal(61) * c + Decimal(3) * d + Decimal(58) * (
        n + resident_g
    ) + fixed
    calibration = result["calibration_only"]

    assert assignment_step.quantize(Decimal("0.000001")) == Decimal(
        calibration["assignment_endpoint"]["compute_step_ps"]
    )
    assert resident_step == Decimal(
        calibration["resident_endpoint"]["compute_step_ps"]
    )


def test_signed_movement_keeps_communication_direction_unpriced() -> None:
    result = _result()
    calibration = result["calibration_only"]
    communication = calibration["expert_parallel_communication"]

    assert Decimal(
        calibration["assignment_endpoint"]["movement_from_inherited_prediction"]
    ) > 0
    assert Decimal(
        calibration["resident_endpoint"]["movement_from_inherited_prediction"]
    ) > 0
    assert communication["measured_in_this_study"] is False
    assert communication["publication_disposition"] == "ABSENT_AND_UNPRICED"
    assert "downward" in communication["signed_effect"]
    assert calibration["single_deployed_expert_parallel_prediction_published"] is False
    assert result["runtime_substitution"]["sglang_physical_binding"] == (
        "DECLARED_LIMITATION_NOT_A_GOAL"
    )


def test_access_disclosure_and_core_registry_limit_are_explicit() -> None:
    result = _result()
    registry = (ROOT / "docs/modules/core.md").read_text(encoding="utf-8")

    assert result["access"] == {
        "fatal_held_out_use_occurred": False,
        "incidental_exposure_count": 2,
        "incidental_exposure_disposition": "SURVIVABLE_AND_DISCLOSED",
        "numeric_values_entering_arithmetic_comparison_fitting_or_reproduction": 0,
        "standard_decode_anchor_is_calibration": True,
    }
    assert "core66_decode_kernel_ladder_result.md" in registry
    assert "The ladder is complete, while CORE-66 remains open" in registry
    assert "- CORE-66 (Precision; P0; L):" in registry
