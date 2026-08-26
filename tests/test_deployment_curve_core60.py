from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from fractions import Fraction
from math import comb
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples/deployment_curve_v1"


def _module():
    name = "deployment_curve_core60_composition"
    sys.path.insert(0, str(STUDY_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            name,
            STUDY_DIR / "core60_composition.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(STUDY_DIR))


def _json(name: str):
    return json.loads((STUDY_DIR / name).read_text(encoding="utf-8"))


def _fraction(value):
    return Fraction(value["numerator"], value["denominator"])


def test_expectations_freeze_has_four_source_backed_contracts_and_no_fit():
    composition = _module()
    expectations = _json("core60_expectations.json")

    composition.validate_expectations(expectations)

    contracts = {row["contract"]: row for row in expectations["contracts"]}
    assert contracts["per-rank token ownership"]["status"] == "verified-existing"
    assert contracts["routed wire precision"]["status"] == "selected"
    assert contracts["same-destination expert deduplication"]["status"] == "selected"
    assert contracts[
        "framework-supported compute and communication overlap"
    ]["status"] == "selected"
    assert expectations["parameters"]["fitted"] == []
    assert expectations["parameters"]["free"] == []
    assert "overlap_fraction" in expectations["parameters"]["rejected"]
    assert expectations["chronology"]["mechanism_service_derived_before_comparison"]
    assert not expectations["chronology"]["core60_calibration_comparison_performed"]
    assert expectations["chronology"]["held_out_numeric_values_accessed"]
    assert expectations["invalidation"]["replacement_owner"] == "COMP-75"


def test_uniform_destination_arithmetic_and_wire_bytes_are_exact():
    expectations = _json("core60_expectations.json")
    traffic = expectations["traffic"]
    probability = Fraction(comb(256, 8) - comb(248, 8), comb(256, 8))

    assert probability == Fraction(939_691_952_959, 4_138_017_124_000)
    assert 32 * probability == Fraction(939_691_952_959, 129_313_035_125)
    assert traffic["dispatch"]["vector_bytes"] == 7_168 + 56 * 4 == 7_392
    assert traffic["combine"]["vector_bytes"] == 7_168 * 2 == 14_336

    for phase in ("dispatch", "combine"):
        row = traffic[phase]
        exact = Fraction(16_384 * row["vector_bytes"]) * probability
        pair = row["per_pair_bytes"]
        assert exact == Fraction(pair["exact_numerator"], pair["exact_denominator"])
        assert pair["selected"] == exact.numerator // exact.denominator
        assert pair["upper"] == pair["selected"] + 1
        assert row["local_bytes_per_phase"]["selected"] == 7 * pair["selected"]
        assert row["fabric_bytes_per_phase"]["selected"] == 24 * pair["selected"]


def test_max_composition_uses_only_component_compute_as_hiding_budget():
    composition = _module()
    expectations = _json("core60_expectations.json")
    core59 = _json("core59_expectations.json")
    shape = composition.calibration_shapes(expectations, core59)["sglang_prefill_1k"]
    arms = {row["id"]: row for row in expectations["composition"]["service_arms"]}

    assert composition.total_service_ps(expectations, shape) == 2_286_179_760_360
    assert (
        composition.total_service_ps(expectations, shape, arm_id="sensitivity")
        == 4_572_127_520_720
    )
    compute = expectations["composition"]["candidate_compute_service_ps"]
    for arm in arms.values():
        for edge in ("lower", "selected", "upper"):
            communication = arm["communication_service_ps"][edge]
            assert arm["total_step_service_ps"][edge] == max(compute, communication)
            assert arm["exposed_incremental_service_ps"][edge] == max(
                communication - compute, 0
            )


def test_shape_gate_rejects_interpolation_and_decode_pricing_is_unchanged():
    composition = _module()
    expectations = _json("core60_expectations.json")
    core59 = _json("core59_expectations.json")
    shapes = composition.calibration_shapes(expectations, core59)

    with pytest.raises(ValueError, match="outside the frozen CORE-60 gate"):
        composition.total_service_ps(
            expectations,
            {**shapes["sglang_prefill_1k"], "new_tokens_per_rank": 16_383},
        )

    decode = {**shapes["sglang_decode_standard"], "candidate_service_ps": 28_604_120_000}
    assert composition.total_service_ps(expectations, decode) == 28_604_120_000


def test_preservation_lock_covers_core59_and_the_first_scored_run():
    composition = _module()
    expectations = _json("core60_expectations.json")

    result = composition.verify_preservation_locks(expectations, REPOSITORY_ROOT)

    assert result["status"] == "PASS"
    assert not result["core59_mutated"]
    assert not result["first_scored_run_mutated"]
    assert len(result["checked_artifacts"]) == 13
    assert sum(
        row["path"].startswith("examples/deployment_curve_v1/core59_")
        for row in result["checked_artifacts"]
    ) == 4


def test_calibration_comparison_records_source_exposure_and_moves_as_frozen():
    composition = _module()
    expectations = _json("core60_expectations.json")
    result = composition.fit_calibration_only(
        _json("expectations.json"),
        _json("scored_expectations.json"),
        _json("core59_expectations.json"),
        expectations,
    )

    assert result["classification"] == "CALIBRATION_ONLY_NOT_SCORED_PROTOCOL_VOID"
    assert result["accessed_anchor_ids"] == [
        "sglang_prefill_1k",
        "sglang_decode_standard",
    ]
    assert result["forbidden_anchor_ids_accessed"] == []
    assert result["held_out_numeric_values_accessed"]
    assert result["externally_exposed_held_out_anchor_ids"] == [
        "sglang_prefill_2k",
        "sglang_prefill_4k",
    ]
    assert not result["held_out_score_performed"]
    assert not result["scored_flagship_rerun_performed"]
    assert not result["decode_pricing_changed"]
    assert result["fitted_parameters"] == []

    rows = {row["anchor_id"]: row for row in result["calibration_rows"]}
    prefill = rows["sglang_prefill_1k"]
    assert prefill["signed_movement_from_candidate_only"]["direction"] == "decrease"
    assert prefill["signed_movement_from_core59"]["direction"] == "increase"
    assert _fraction(prefill["prediction"]["point"]) > _fraction(prefill["core59_point"])
    assert _fraction(prefill["signed_relative_error_after"]) == Fraction(
        -9_764_143_737_533,
        1_648_164_143_737_533,
    )
    decode = rows["sglang_decode_standard"]
    assert decode["signed_movement_from_candidate_only"]["direction"] == "unchanged"
    assert decode["signed_movement_from_core59"]["direction"] == "unchanged"


def test_check_only_cli_verifies_the_freeze_and_reports_protocol_void():
    completed = subprocess.run(
        [
            sys.executable,
            str(STUDY_DIR / "run_core60_composition.py"),
            "--check-only",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["status"] == "PASS"
    assert result["expectations_sha256"] == hashlib.sha256(
        (STUDY_DIR / "core60_expectations.json").read_bytes()
    ).hexdigest()
    assert len(result["preservation_lock"]["checked_artifacts"]) == 13
    assert result["held_out_numeric_values_accessed"]
    assert result["externally_exposed_held_out_anchor_ids"] == [
        "sglang_prefill_2k",
        "sglang_prefill_4k",
    ]
    assert not result["held_out_score_performed"]
    assert not result["scored_flagship_rerun_performed"]


def test_published_result_matches_the_frozen_calibration_projection():
    result_path = STUDY_DIR / "core60_calibration_result.json"
    if not result_path.exists():
        pytest.skip("published result is added after the expectations-only checkpoint")
    composition = _module()
    expectations = _json("core60_expectations.json")
    computed = composition.fit_calibration_only(
        _json("expectations.json"),
        _json("scored_expectations.json"),
        _json("core59_expectations.json"),
        expectations,
    )
    published = _json("core60_calibration_result.json")
    published_rows = []
    for row in published["calibration_rows"]:
        selected = dict(row)
        selected.pop("decimal_summary")
        published_rows.append(selected)

    assert published_rows == computed["calibration_rows"]
    for name in (
        "schema",
        "status",
        "classification",
        "accessed_anchor_ids",
        "forbidden_anchor_ids_accessed",
        "held_out_numeric_values_accessed",
        "externally_exposed_held_out_anchor_ids",
        "held_out_score_performed",
        "scored_flagship_rerun_performed",
        "fitted_parameters",
        "decode_pricing_changed",
    ):
        assert published[name] == computed[name]
    assert published["expectations_sha256"] == hashlib.sha256(
        (STUDY_DIR / "core60_expectations.json").read_bytes()
    ).hexdigest()
    assert published["preservation_lock"] == {
        "checked_artifact_count": 13,
        "core59_mutated": False,
        "first_scored_run_mutated": False,
        "status": "PASS",
    }
    assert published["remainder"]["id"] == "TRAF-66"
    assert not published["remainder"]["applied_to_projection"]
