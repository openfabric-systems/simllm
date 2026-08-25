from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "kernel_cycle_lut_v1"
FIXTURE = ROOT / "tests" / "fixtures" / "kernel_cycle_lut_v1"
RESULTS = STUDY / "results.json"
REPORT = STUDY / "RESULTS.md"
RUNNER = STUDY / "run_study.py"

SPEC = importlib.util.spec_from_file_location("kernel_cycle_lut_run_study", RUNNER)
assert SPEC is not None and SPEC.loader is not None
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def _results() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def test_committed_result_reproduces_from_retained_fixture(tmp_path: Path) -> None:
    assert study.evaluate(FIXTURE, tmp_path) == _results()


def test_run_is_nonvoid_and_guards_stay_out_of_score() -> None:
    results = _results()

    assert results["run_state"] == "nonvoid"
    assert results["voiding_guards"] == []
    assert all(guard["passed"] for guard in results["fatal_guards"].values())
    assert results["evidence_classes"] == {
        "behavioral_parameterized_instances": 5,
        "behavioral_relation_families": 1,
        "exact_oracle_rows": 0,
        "fatal_guards": 7,
        "native_executables": 0,
    }
    assert results["scored_denominator_relation_families"] == 1
    assert results["scored_passed_relation_families"] == 1


def test_deciding_ratio_and_conservation_result_are_exact() -> None:
    results = _results()
    relation = results["relations"]["R1-cross-instrument-elapsed-agreement"]

    assert relation["maximum_ratio_ppm"] == 1_739_130
    assert relation["passed"] is True
    assert len(relation["instances"]) == results["scored_parameterized_instances"] == 5
    assert results["fatal_guards"]["G3"] == {
        "max_abs_reconstruction_error_ps": 0,
        "measured_service_ps": 2_047_488_000,
        "passed": True,
        "reconstructed_service_ps": 2_047_488_000,
    }


def test_candidate_compilation_changes_neither_service_duration() -> None:
    compilation = _results()["fatal_guards"]["G6"]

    assert compilation["acceptance_status"] == "candidate"
    assert compilation["profile_table_schema"] == "simllm-profile-table-v1"
    assert compilation["profile_table_duration_ps"] == 2_047_488_000
    assert compilation["device_service_duration_ps"] == 2_047_488_000


def test_result_keeps_comp64_open_and_names_only_reserved_residuals() -> None:
    closure = _results()["closure"]

    assert closure["closes"] == []
    assert closure["keeps_open"] == ["COMP-64"]
    assert closure["expected_residual_ids"] == ["COMP-65", "COMP-66"]
    assert set(closure["does_not_claim"]) == {
        "gpu-campaign-execution",
        "numerical-calibration",
        "compile-time-graph-inference",
        "program-counter-attribution",
    }


def test_report_uses_plain_result_shape_and_no_em_dash() -> None:
    text = REPORT.read_text(encoding="utf-8")

    assert "What ran:" in text
    assert "What came out:" in text
    assert "What it changes:" in text
    assert "What it does not change:" in text
    assert "1.739130" in text
    assert "COMP-64 stays open" in text
    assert chr(0x2014) not in text
