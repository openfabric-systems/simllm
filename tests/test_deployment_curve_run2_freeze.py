from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples/deployment_curve_v1"
FREEZE_PATH = STUDY_DIR / "scored_run2_expectations.json"


def _freeze() -> dict[str, object]:
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


def test_run2_freeze_precedes_every_second_scored_output():
    frozen = _freeze()

    assert frozen["schema"] == "simllm-deployment-curve-scored-run2-expectations-v1"
    assert frozen["status"] == "EXPECTATIONS_ONLY"
    chronology = frozen["chronology"]
    assert chronology["first_scored_run_existed_before_this_freeze"] is True
    assert chronology["comp75_clean_repetition_existed_before_this_freeze"] is True
    assert chronology["sgl38_remote_kv_projection_existed_before_this_freeze"] is True
    assert chronology["second_scored_runner_existed_before_this_freeze"] is False
    assert chronology["second_fitted_constants_existed_before_this_freeze"] is False
    assert chronology["second_held_out_score_existed_before_this_freeze"] is False
    assert chronology["second_flagship_figure_existed_before_this_freeze"] is False


def test_run2_prefill_bands_follow_clean_max_composition():
    frozen = _freeze()
    prefill = frozen["pricing_configuration"]["prefill"]
    point_arm = prefill["point_arm"]["communication_service_ps"]
    rows = {
        row["anchor_id"]: row
        for row in frozen["pre_fit_predicted_bands"]
        if row["anchor_id"].startswith("sglang_prefill")
    }

    assert prefill["new_free_or_fitted_tunables"] == 0
    assert prefill["per_rank_new_tokens"] == 16_384
    assert set(rows) == {
        "sglang_prefill_1k",
        "sglang_prefill_2k",
        "sglang_prefill_4k",
    }
    numerator = 131_072 * 1_000_000_000_000
    for row in rows.values():
        compute = row["candidate_compute_service_ps"]
        expected_point = Fraction(numerator, max(compute, point_arm["selected"]))
        expected_lower = Fraction(numerator, max(compute, point_arm["upper"]))
        assert Fraction(**row["point"]) == expected_point
        assert Fraction(**row["upper"]) == expected_point
        assert Fraction(**row["lower"]) == expected_lower


def test_run2_decode_price_and_known_direction_are_frozen():
    frozen = _freeze()
    decode = frozen["pricing_configuration"]["decode"]
    row = next(
        value
        for value in frozen["pre_fit_predicted_bands"]
        if value["anchor_id"] == "sglang_decode_standard"
    )

    assert decode["mechanism_count"] == 0
    assert decode["remote_kv_projection"]["enabled_for_this_run"] is True
    assert decode["remote_kv_projection"]["exact_candidate_key_sha256"] == (
        "05d1c33cdef9c12e25eb9159adc9dc80f1cd57b6333778f9efb5fb24cd6a74aa"
    )
    assert decode["declared_full_depth_service_ps"] == (
        decode["measured_four_layer_basis_ps"] * 61 // 4
    )
    predicted = Fraction(**row["point"])
    published = Fraction(decode["visible_calibration_tokens_per_second_per_node"])
    assert predicted < published
    assert Fraction(**row["lower"]) == predicted == Fraction(**row["upper"])


def test_run2_inherits_constant_envelope_with_zero_mechanism_applications():
    frozen = _freeze()
    constants = frozen["constants"]
    inherited = constants["tunable"]

    assert constants["composition_new_tunables"] == []
    assert len(inherited) == 1
    assert inherited[0]["id"] == "intra_node_collective_surcharge_ps"
    assert inherited[0]["initial"] == 15_064_014
    assert inherited[0]["envelope"] == {"lower": 0, "upper": 30_128_029}
    assert inherited[0]["second_run_mechanism_path_application_count_per_step"] == 0
    assert frozen["fit_rule"]["expected_flat_fit_value_ps"] == 0


def test_run2_freeze_contains_no_held_out_numeric_values():
    raw = FREEZE_PATH.read_text(encoding="utf-8")

    assert "54543" not in raw
    assert "50302" not in raw


def test_run2_preservation_lock_matches_every_prior_record():
    frozen = _freeze()

    for artifact in frozen["preservation_lock"]["artifacts"]:
        path = REPOSITORY_ROOT / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
