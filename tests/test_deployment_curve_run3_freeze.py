from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from math import comb
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples" / "deployment_curve_v1"
FREEZE_PATH = STUDY_DIR / "scored_run3_expectations.json"
PS_PER_SECOND = 1_000_000_000_000


def _freeze() -> dict[str, object]:
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


def _fraction(value: dict[str, int]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def test_run3_freeze_precedes_every_third_scored_output():
    frozen = _freeze()

    assert frozen["schema"] == "simllm-deployment-curve-scored-run3-expectations-v1"
    assert frozen["status"] == "EXPECTATIONS_ONLY"
    chronology = frozen["chronology"]
    assert chronology["second_scored_run_existed_before_this_freeze"] is True
    assert chronology["traf66_boundary_study_existed_before_this_freeze"] is True
    assert chronology["traf67_clean_repetition_existed_before_this_freeze"] is True
    assert chronology["third_scored_runner_existed_before_this_freeze"] is False
    assert chronology["third_fitted_constants_existed_before_this_freeze"] is False
    assert chronology["third_held_out_score_existed_before_this_freeze"] is False
    assert chronology["third_flagship_figure_existed_before_this_freeze"] is False
    assert chronology["held_out_anchor_numeric_values_accessed"] is False


def test_run3_freeze_contains_no_held_out_anchor_numeric_values():
    raw = FREEZE_PATH.read_text(encoding="utf-8")

    assert "54543" not in raw
    assert "50302" not in raw


def test_run3_overlap_fraction_is_exactly_the_clean_derived_bracket():
    frozen = _freeze()
    prefill = frozen["pricing_configuration"]["prefill"]
    constants = {row["id"]: row for row in frozen["constants"]["tunable"]}
    exposed = constants["overlap_exposed_fraction"]

    assert prefill["physics_only_operator"] == "max(C, P)"
    assert prefill["boundary_operator"] == "max(C, P) + f * min(C, P)"
    assert _fraction(exposed["envelope"]["lower"]) == 0
    assert _fraction(exposed["envelope"]["upper"]) == Fraction(1, 2)
    assert exposed["floor_authority"] == "COMP-75 clean perfect-overlap form"
    assert exposed["ceiling_authority"] == "TRAF-67 clean two-child boundary form"
    assert frozen["constants"]["new_unbounded_or_free"] == []


def test_run3_locality_refinement_is_derived_and_cross_architecture_candidate_rejected():
    refinements = {
        row["id"]: row for row in _freeze()["pricing_configuration"]["derived_refinements"]
    }
    locality = refinements["ep32_rank_layout_locality_split"]
    rejected = refinements["a100_three_module_packet_candidate_substitution"]

    assert locality["status"] == "FROZEN_DERIVED"
    assert locality["anchor_numeric_input_count"] == 0
    assert locality["fit_allowed"] is False
    assert locality["topology"]["same_node_destination_peers"] == 7
    assert locality["topology"]["fabric_destination_peers"] == 24
    assert 7 + 24 == 32 - 1
    assert rejected["status"] == "REJECTED"
    assert "A100" in rejected["reason"] and "H100" in rejected["reason"]


def test_run3_admitted_attenuation_factor_is_independent_hypergeometric_arithmetic():
    layer = _freeze()["attenuation_layer"]
    assert layer["admitted_factor_count"] == 1
    factor = layer["factors"][0]
    magnitude = factor["magnitude"]

    experts = magnitude["logical_experts"]
    experts_per_rank = magnitude["experts_per_rank"]
    ranks = magnitude["expert_parallel_ranks"]
    top_k = magnitude["top_k"]
    p = Fraction(
        comb(experts, top_k) - comb(experts - experts_per_rank, top_k),
        comb(experts, top_k),
    )
    expected_unique = ranks * p
    expected_factor = expected_unique / top_k

    assert _fraction(magnitude["p_one_rank_hit"]) == p
    assert _fraction(magnitude["expected_unique_destination_ranks"]) == expected_unique
    assert _fraction(magnitude["factor"]) == expected_factor
    assert factor["anchor_numeric_input_count"] == 0
    assert len(factor["applies_to_anchor_ids"]) == 3
    assert layer["admitted_factor_count"] < len(factor["applies_to_anchor_ids"])


def test_run3_attenuation_uncertainty_is_exact_indicator_covariance():
    factor = _freeze()["attenuation_layer"]["factors"][0]
    magnitude = factor["magnitude"]
    uncertainty = factor["uncertainty"]
    experts = magnitude["logical_experts"]
    experts_per_rank = magnitude["experts_per_rank"]
    ranks = magnitude["expert_parallel_ranks"]
    top_k = magnitude["top_k"]
    p = _fraction(magnitude["p_one_rank_hit"])
    p_two = Fraction(
        comb(experts, top_k)
        - 2 * comb(experts - experts_per_rank, top_k)
        + comb(experts - 2 * experts_per_rank, top_k),
        comb(experts, top_k),
    )
    variance = ranks * p * (1 - p) + ranks * (ranks - 1) * (p_two - p * p)
    standard_error = math.sqrt(float(variance) / uncertainty["tokens"]) / top_k
    interval = uncertainty["interval"]
    point = _fraction(interval["point"])

    assert _fraction(uncertainty["p_two_named_ranks_hit"]) == p_two
    assert _fraction(uncertainty["unique_rank_count_variance"]) == variance
    assert math.isclose(
        float(_fraction(uncertainty["standard_error"])),
        standard_error,
        rel_tol=0,
        abs_tol=5e-16,
    )
    assert _fraction(interval["lower"]) <= point - 2 * standard_error
    assert _fraction(interval["upper"]) >= point + 2 * standard_error


def test_run3_packing_factor_is_not_admitted_and_decode_is_not_attenuated():
    layer = _freeze()["attenuation_layer"]
    rejected = {row["id"]: row for row in layer["rejected_candidates"]}

    assert rejected["exact_length_packing_vs_per_request_overhead"]["factor"] is None
    assert rejected["exact_length_packing_vs_per_request_overhead"]["status"] == (
        "NOT_ADMITTED"
    )
    assert rejected["decode_depth_attenuation"]["status"] == (
        "FORBIDDEN_BY_POLICY_RULE_FIVE"
    )
    assert _freeze()["pricing_configuration"]["decode"]["attenuation_allowed"] is False


def test_run3_prefill_layers_reproduce_physics_boundary_and_attenuation_arithmetic():
    frozen = _freeze()
    prefill = frozen["pricing_configuration"]["prefill"]
    packet = prefill["point_arm"]["communication_service_ps"]
    factor_interval = frozen["attenuation_layer"]["factors"][0]["uncertainty"][
        "interval"
    ]
    rows = {
        row["anchor_id"]: row
        for row in frozen["pre_fit_prediction_layers"]
        if row["anchor_id"].startswith("sglang_prefill")
    }

    assert set(rows) == {
        "sglang_prefill_1k",
        "sglang_prefill_2k",
        "sglang_prefill_4k",
    }
    numerator = 131_072 * PS_PER_SECOND
    for row in rows.values():
        compute = row["candidate_compute_service_ps"]
        physics = row["physics_only"]
        boundary = row["physics_plus_boundary"]
        attenuated = row["physics_plus_boundary_plus_attenuation"]
        physics_lower = Fraction(numerator, max(compute, packet["upper"]))
        physics_point = Fraction(numerator, max(compute, packet["selected"]))
        boundary_lower_service = max(compute, packet["upper"]) + Fraction(
            min(compute, packet["upper"]), 2
        )

        assert _fraction(physics["lower"]) == physics_lower
        assert _fraction(physics["point"]) == physics_point
        assert _fraction(physics["upper"]) == physics_point
        assert _fraction(boundary["lower"]) == Fraction(
            numerator, boundary_lower_service
        )
        assert _fraction(boundary["point"]) == physics_point
        assert _fraction(boundary["upper"]) == physics_point
        assert _fraction(attenuated["lower"]) == (
            _fraction(boundary["lower"]) * _fraction(factor_interval["lower"])
        )
        assert _fraction(attenuated["point"]) == (
            _fraction(boundary["point"]) * _fraction(factor_interval["point"])
        )
        assert _fraction(attenuated["upper"]) == (
            _fraction(boundary["upper"]) * _fraction(factor_interval["upper"])
        )


def test_run3_decode_layers_are_identical_and_mtp_is_blocked():
    rows = {row["anchor_id"]: row for row in _freeze()["pre_fit_prediction_layers"]}
    decode = rows["sglang_decode_standard"]
    expected = Fraction(256 * PS_PER_SECOND, 28_604_120_000)

    for layer in (
        "physics_only",
        "physics_plus_boundary",
        "physics_plus_boundary_plus_attenuation",
    ):
        assert _fraction(decode[layer]["lower"]) == expected
        assert _fraction(decode[layer]["point"]) == expected
        assert _fraction(decode[layer]["upper"]) == expected
    assert decode["attenuation_applied"] is False
    assert rows["sglang_decode_simulated_mtp"]["status"] == "BLOCKED"
    assert rows["sglang_decode_simulated_mtp"]["prediction"] is None


def test_run3_preservation_lock_matches_every_named_prior_artifact():
    frozen = _freeze()
    artifacts = frozen["preservation_lock"]["artifacts"]

    assert len(artifacts) >= 30
    for artifact in artifacts:
        path = REPOSITORY_ROOT / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
