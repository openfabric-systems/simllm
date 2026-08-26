from __future__ import annotations

import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPO_ROOT / "examples" / "deployment_curve_v1"
sys.path.insert(0, str(STUDY_DIR))


def _module(name: str):
    path = STUDY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def boundary():
    return _module("traf66_overlap_boundary")


@pytest.fixture(scope="module")
def signer():
    return _module("traf66_independent_sign")


def _json(name: str):
    return json.loads((STUDY_DIR / name).read_text(encoding="utf-8"))


def test_expectations_freeze_is_precomparison_and_records_protocol_void(boundary):
    expectations = _json("traf66_expectations.json")
    boundary.validate_expectations(expectations)
    chronology = expectations["chronology"]
    assert chronology["allowlist_extension_preceded_source_inspection"] is True
    assert chronology["boundary_form_derived_before_visible_comparison"] is True
    assert chronology["visible_calibration_comparison_performed"] is False
    assert chronology["scored_comparison_performed"] is False
    assert chronology["scored_flagship_rerun_performed"] is False
    assert chronology["web_pages_fetched"] is False
    assert chronology["held_out_component_record_accessed"] is True
    assert len(expectations["calibration_split"]["held_out_access_ledger"]) == 1
    assert expectations["parameters"]["free"] == []
    assert expectations["parameters"]["fitted"] == []


def test_source_ranges_and_twenty_four_plus_prior_records_are_locked(boundary):
    expectations = _json("traf66_expectations.json")
    lock = boundary.verify_preservation_locks(expectations, REPO_ROOT)
    assert lock["status"] == "PASS"
    assert lock["prior_records_mutated"] is False
    assert lock["checked_count"] == 27
    assert len(expectations["source_contracts"]) == 4
    assert {row["range_sha256"] for row in expectations["source_contracts"]} == {
        "39c572ed113a16b8fe697e891b378a97f993408294f856a04dbbf54b3289aa06",
        "70c89306abc488c4a9b583c088dc39b65c86ab806ac24f4e70c9845fe526480f",
        "4599687f7fad87739bdc5aba91bba758c8fd1f6313ee04a97ddf7159818041f8",
        "235220c3b206047ff02b108c7a580a79290f382cad234afb12098cccbc8229f9",
    }


def test_child_stage_and_async_event_conservation_is_exact(boundary):
    counts = boundary.conserved_event_counts(58)
    assert counts == {
        "children": 2,
        "moe_layers": 58,
        "dispatch_launches_per_child": 58,
        "dispatch_completions_per_child": 58,
        "combine_launches_per_child": 58,
        "combine_completions_per_child": 58,
        "yield_boundaries_per_child": 116,
        "stages_per_child": 117,
        "dispatch_launches_total": 116,
        "dispatch_completions_total": 116,
        "combine_launches_total": 116,
        "combine_completions_total": 116,
        "yield_boundaries_total": 232,
        "stage_advances_total": 234,
    }
    assert counts["dispatch_launches_total"] == counts["dispatch_completions_total"]
    assert counts["combine_launches_total"] == counts["combine_completions_total"]
    assert counts["stage_advances_total"] == 2 * counts["stages_per_child"]


@pytest.mark.parametrize(
    ("compute", "packet"),
    [
        (12, 20),
        (20, 12),
        (12, 12),
        (13, 20),
    ],
)
def test_finite_form_conserves_two_child_services(boundary, compute, packet):
    result = boundary.finite_two_batch_service(compute, packet)
    assert 2 * result["child_compute_service_ps"] == compute
    assert 2 * result["child_packet_service_ps"] == packet
    assert result["total_service_ps"] == max(compute, packet) + Fraction(
        min(compute, packet), 2
    )
    assert result["boundary_service_ps"] == Fraction(min(compute, packet), 2)


def test_frozen_envelope_reuses_components_and_adds_compute_half(boundary):
    expectations = _json("traf66_expectations.json")
    previous = _json("comp75_expectations.json")
    contracts = boundary.compare_component_inputs(expectations, previous)
    assert contracts == {
        "candidate_compute_service_reused": True,
        "packet_service_envelope_reused": True,
        "moe_layer_count_reused": True,
    }
    composition = expectations["composition"]
    compute = composition["candidate_compute_service_ps"]
    for edge in ("lower", "selected", "upper"):
        packet = composition["packet_service_ps"][edge]
        result = boundary.finite_two_batch_service(compute, packet)
        assert result["boundary_service_ps"] == Fraction(compute, 2)
        assert result["total_service_ps"] == packet + Fraction(compute, 2)
        assert result["total_service_ps"] == composition["total_service_ps"][edge]


def test_signed_movement_is_derived_without_target_value(boundary):
    expectations = _json("traf66_expectations.json")
    composition = expectations["composition"]
    compute = composition["candidate_compute_service_ps"]
    previous = composition["packet_service_ps"]["selected"]
    updated = boundary.finite_two_batch_service(compute, previous)["total_service_ps"]
    assert updated > previous
    arbitrary_positive_target = Fraction(57_000)
    old_prediction = Fraction(131_072 * boundary.PS_PER_SECOND, previous)
    new_prediction = Fraction(131_072 * boundary.PS_PER_SECOND, updated)
    old_residual = old_prediction / arbitrary_positive_target - 1
    new_residual = new_prediction / arbitrary_positive_target - 1
    assert new_prediction < old_prediction
    assert new_residual < old_residual


def test_held_out_statement_is_structural_only(boundary):
    expectations = _json("traf66_expectations.json")
    structural = expectations["held_out_structural_prediction"]
    assert structural == {
        "compute_dependence_reenters": True,
        "numeric_values_accessed_for_prediction": False,
        "rows": ["sglang_prefill_2k", "sglang_prefill_4k"],
        "service_term": "packet_service_ps + compute_service_ps / 2",
    }


def test_visible_result_matches_frozen_form(boundary):
    expectations = _json("traf66_expectations.json")
    previous = _json("comp75_calibration_result.json")
    result = _json("traf66_calibration_result.json")
    boundary.validate_result(result, expectations, previous)
    row = result["calibration_rows"][0]
    assert row["boundary_service_ps"] == 681_624_980_000
    assert row["total_service_ps"] == 2_967_804_740_360
    assert row["signed_movement_from_comp75"]["direction"] == "decrease"
    assert Fraction(
        row["signed_relative_error_after"]["numerator"],
        row["signed_relative_error_after"]["denominator"],
    ) < Fraction(
        row["signed_relative_error_before"]["numerator"],
        row["signed_relative_error_before"]["denominator"],
    )


def test_independent_signer_reconstructs_visible_movement(boundary, signer):
    expectations = _json("traf66_expectations.json")
    result = _json("traf66_calibration_result.json")
    row = result["calibration_rows"][0]
    composition = expectations["composition"]
    signed = signer.sign_visible_movement(
        per_node_tokens=row["per_node_tokens"],
        published_numerator=row["published"]["numerator"],
        published_denominator=row["published"]["denominator"],
        compute_service_ps=composition["candidate_compute_service_ps"],
        packet_service_ps=composition["packet_service_ps"]["selected"],
        children=expectations["event_conservation"]["counts"]["children"],
    )
    assert signed == result["independent_signoff"]
    assert signed["movement"]["direction"] == "decrease"
    assert signed["signed_residual_movement"]["direction"] == "more_negative"


def test_result_retains_preservation_and_scope_fences(boundary):
    expectations = _json("traf66_expectations.json")
    result = _json("traf66_calibration_result.json")
    lock = boundary.verify_preservation_locks(expectations, REPO_ROOT)
    assert result["preservation_lock"] == {
        "checked_count": lock["checked_count"],
        "prior_records_mutated": lock["prior_records_mutated"],
        "status": lock["status"],
    }
    assert result["scored_flagship_rerun_performed"] is False
    assert result["decode_pricing_changed"] is False
    assert result["nvlink_scope_touched"] is False
    assert result["remainder"]["id"] == "TRAF-67"
    assert result["held_out_numeric_values_used_or_compared"] is False


@pytest.mark.parametrize("bad", [True, 0, -1, 1.5, "2"])
def test_boundary_rejects_invalid_services(boundary, bad):
    with pytest.raises((TypeError, ValueError)):
        boundary.finite_two_batch_service(bad, 2)
