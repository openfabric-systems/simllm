from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from fractions import Fraction
from math import comb
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPO_ROOT / "examples" / "deployment_curve_v1"
SCRIPT = STUDY_DIR / "run_comp75_repetition.py"
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
def repetition():
    return _module("comp75_repetition")


@pytest.fixture(scope="module")
def signer():
    return _module("comp75_independent_sign")


def _json(name: str):
    return json.loads((STUDY_DIR / name).read_text(encoding="utf-8"))


def _fraction(value):
    return Fraction(value["numerator"], value["denominator"])


def test_expectations_freeze_precedes_visible_comparison_and_has_no_fit(repetition):
    expectations = _json("comp75_expectations.json")
    repetition.validate_expectations(expectations)
    chronology = expectations["chronology"]
    assert chronology["allowlist_committed_before_source_inspection"] is True
    assert chronology["mechanism_service_derived_before_comparison"] is True
    assert chronology["visible_calibration_comparison_performed"] is False
    assert chronology["visible_calibration_numeric_values_accessed"] is False
    assert chronology["void_core60_external_source_values_accessed"] is False
    assert chronology["web_pages_fetched"] is False
    assert expectations["calibration_split"]["held_out_access_ledger"] == []
    assert expectations["parameters"]["free"] == []
    assert expectations["parameters"]["fitted"] == []


def test_source_allowlist_component_and_void_locks_hold(repetition):
    expectations = _json("comp75_expectations.json")
    lock = repetition.verify_evidence_locks(expectations, REPO_ROOT)
    assert lock["status"] == "PASS"
    assert lock["void_core60_mutated"] is False
    assert len(lock["void_core60_artifacts"]) == 8
    assert expectations["source_allowlist"]["preregistered_sha256"] == (
        "e5bc633175a1636615c9867a2caa10e591cace4ab40bcef13c18108d77c4190b"
    )


def test_destination_arithmetic_and_packet_widths_are_exact(repetition):
    expectations = _json("comp75_expectations.json")
    repetition.validate_expectations(expectations)
    traffic = expectations["traffic"]
    probability = Fraction(comb(256, 8) - comb(248, 8), comb(256, 8))
    row = traffic["dedup_probability"]
    assert probability == Fraction(row["p_rank_numerator"], row["p_rank_denominator"])
    assert 32 * probability == Fraction(
        row["expected_unique_destinations_numerator"],
        row["expected_unique_destinations_denominator"],
    )
    assert 31 * probability == Fraction(
        row["expected_remote_destinations_numerator"],
        row["expected_remote_destinations_denominator"],
    )
    assert traffic["dispatch"]["vector_bytes"] == 7_168 + (7_168 // 128) * 4
    assert traffic["combine"]["vector_bytes"] == 7_168 * 2
    assert traffic["dispatch"]["per_pair_bytes"]["selected"] == 27_502_686
    assert traffic["combine"]["per_pair_bytes"]["selected"] == 53_338_543


def test_packet_services_and_max_composition_use_no_overlap_parameter(repetition):
    expectations = _json("comp75_expectations.json")
    repetition.validate_expectations(expectations)
    composition = expectations["composition"]
    assert composition["operator"] == "max"
    assert expectations["parameters"]["free"] == []
    assert expectations["parameters"]["fitted"] == []
    arms = {row["id"]: row for row in composition["service_arms"]}
    assert arms["point"]["dispatch_phase_service_ps"] == {
        "lower": 13_410_556_120,
        "selected": 13_410_556_120,
        "upper": 13_410_556_140,
    }
    assert arms["point"]["combine_phase_service_ps"] == {
        "lower": 26_006_336_300,
        "selected": 26_006_336_300,
        "upper": 26_006_336_320,
    }
    compute = composition["candidate_compute_service_ps"]
    for arm in arms.values():
        for edge in ("lower", "selected", "upper"):
            communication = 58 * (
                arm["dispatch_phase_service_ps"][edge]
                + arm["combine_phase_service_ps"][edge]
            )
            assert arm["communication_service_ps"][edge] == communication
            assert arm["total_step_service_ps"][edge] == max(compute, communication)


def test_clean_contracts_reproduce_void_record_without_promoting_it(repetition):
    comparison = repetition.compare_core60_contracts(
        _json("comp75_expectations.json"), _json("core60_expectations.json")
    )
    assert comparison["verdict"] == "REPRODUCED"
    assert set(comparison["contracts"].values()) == {"REPRODUCED"}
    assert comparison["core60_external_source_values_accessed"] is False
    assert comparison["core60_record_status_unchanged"] == "VOID"
    assert comparison["core60_promoted"] is False


def test_visible_comparison_selects_only_1k_and_moves_as_preregistered(repetition):
    result = repetition.calibration_comparison(
        _json("expectations.json"),
        _json("scored_expectations.json"),
        _json("core59_expectations.json"),
        _json("comp75_expectations.json"),
    )
    assert result["status"] == "PASS"
    assert result["accessed_visible_anchor_ids"] == ["sglang_prefill_1k"]
    assert result["held_out_access_ledger"] == []
    assert result["held_out_numeric_values_accessed"] is False
    assert result["held_out_score_performed"] is False
    assert result["scored_flagship_rerun_performed"] is False
    assert result["decode_pricing_changed"] is False
    row = result["calibration_rows"][0]
    assert row["signed_movement_from_candidate_only"]["direction"] == "decrease"
    assert row["signed_movement_from_core59"]["direction"] == "increase"
    assert _fraction(row["prediction"]["point"]) == Fraction(
        3_276_800_000_000_000, 57_154_494_009
    )
    assert _fraction(row["absolute_remaining_error_tokens_per_second_per_node"]) == Fraction(
        -19_528_287_475_066, 57_154_494_009
    )
    assert _fraction(row["signed_relative_error_after"]) == Fraction(
        -9_764_143_737_533, 1_648_164_143_737_533
    )


def test_independent_signer_matches_projection_without_importing_composition(signer):
    signoff = signer.sign_visible_movement(
        per_node_tokens=131_072,
        candidate_service_ps=1_363_249_960_000,
        core59_total_service_ps=4_684_122_088_000,
        comp75_total_service_ps=2_286_179_760_360,
        published_tokens_per_second_per_node=57_674,
    )
    assert signoff["movement_from_candidate_only"]["direction"] == "decrease"
    assert signoff["movement_from_core59"]["direction"] == "increase"
    assert _fraction(signoff["absolute_remaining_error_tokens_per_second_per_node"]) == Fraction(
        -19_528_287_475_066, 57_154_494_009
    )
    assert _fraction(signoff["signed_relative_remaining_error"]) == Fraction(
        -9_764_143_737_533, 1_648_164_143_737_533
    )


def test_published_clean_result_matches_recomputed_projection(repetition, signer):
    published = _json("comp75_calibration_result.json")
    recomputed = repetition.calibration_comparison(
        _json("expectations.json"),
        _json("scored_expectations.json"),
        _json("core59_expectations.json"),
        _json("comp75_expectations.json"),
    )
    assert published["calibration_rows"] == recomputed["calibration_rows"]
    assert published["record_comparison"]["verdict"] == "REPRODUCED"
    assert published["held_out_access_ledger"] == []
    assert published["fitted_parameters"] == []
    assert published["free_parameters"] == []
    row = published["calibration_rows"][0]
    expected_signoff = signer.sign_visible_movement(
        per_node_tokens=row["per_node_tokens"],
        candidate_service_ps=row["candidate_service_ps"],
        core59_total_service_ps=row["core59_total_service_ps"],
        comp75_total_service_ps=row["total_service_ps"],
        published_tokens_per_second_per_node=row["published"]["numerator"],
    )
    assert published["independent_signoff"] == expected_signoff
    assert (STUDY_DIR / "comp75_calibration_result.json").read_bytes().endswith(b"\n")


def test_check_only_cli_reports_empty_ledger_and_preservation_pass():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--check-only"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "PASS"
    assert result["held_out_access_ledger"] == []
    assert result["held_out_numeric_values_accessed"] is False
    assert result["scored_flagship_rerun_performed"] is False
    assert result["preservation_lock"]["void_core60_mutated"] is False
