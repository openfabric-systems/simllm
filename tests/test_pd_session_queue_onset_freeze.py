from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

from simllm.calibration.batch_service_surface import BatchServicePoint

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples" / "pd_session_queue_onset_v1"
REFERENCE_SURFACE = (
    REPOSITORY_ROOT / "examples" / "pd_session_load_delay_v1" / "surface.json"
)


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, STUDY_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(STUDY_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(STUDY_DIR))
    return module


def _freeze() -> dict:
    return json.loads((STUDY_DIR / "expectations.json").read_text(encoding="utf-8"))


def _fraction(value: dict[str, int]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def _points() -> tuple[BatchServicePoint, ...]:
    return tuple(
        BatchServicePoint(
            row["batch_size"],
            row["measured_service_ps"],
            row["trimmed_coefficient_of_variation_ppm"] / 1_000_000,
            row["entry_key_sha256"],
            row["evidence_class"],
            row["split"],
        )
        for row in _freeze()["surface"]["selected_keys"]
    )


def test_freeze_precedes_every_vllm41_observation_and_refuses_curve_fit() -> None:
    freeze = _freeze()

    assert freeze["status"] == "EXPECTATIONS_ONLY"
    assert freeze["task"] == "VLLM-41"
    assert freeze["chronology"] == {
        "field_reader_committed_before_access": True,
        "fresh_surface_access_existed_before_freeze": True,
        "lower_ladder_run_existed_before_freeze": False,
        "observed_curve_fit_permitted": False,
        "vllm41_curve_values_accessed_before_freeze": False,
    }
    assert freeze["queue_model"]["observed_curve_inputs"] == []
    assert freeze["queue_model"]["fit_parameters"] == []
    assert freeze["decision_rule"]["no_monotonic_rescore"].startswith(
        "do not compute or score"
    )


def test_lower_ladder_and_held_out_union_are_literal() -> None:
    freeze = _freeze()
    sweep = freeze["sweep"]

    assert sweep["offered_load_requests_per_second"] == [
        50,
        100,
        150,
        175,
        200,
        210,
        220,
        225,
        230,
        235,
        240,
        245,
        250,
    ]
    assert sweep["minimum_load_is_below_vllm39"] is True
    assert sweep["maximum_load_repeats_vllm39_boundary_only"] is True
    assert sweep["point_count"] == 78
    assert freeze["held_out"]["loads"] == [240]
    assert freeze["held_out"]["pool_ratios"] == [[2, 1]]
    assert freeze["held_out"]["point_count"] == 30


def test_logged_projection_keeps_candidate_surface_unchanged() -> None:
    freeze = _freeze()
    access = freeze["source_access"]
    ledger_path = STUDY_DIR / "access_ledger.jsonl"
    rows = [
        json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]

    assert access["status"] == "CLEAN"
    assert access["successful_field_count"] == 5
    assert access["whole_record_loaded"] is False
    assert access["held_out_batch_32_decoded_or_captured"] is False
    assert access["deepseek_row_decoded_or_captured"] is False
    assert access["access_ledger_sha256"] == hashlib.sha256(
        ledger_path.read_bytes()
    ).hexdigest()
    assert len(rows) == 5
    assert all(row["status"] == "PASS" for row in rows)
    assert all(row["whole_record_loaded"] is False for row in rows)
    assert [row["bytes_consumed_total"] for row in rows] == [45_043] * 5
    assert freeze["surface"]["sha256"] == hashlib.sha256(
        REFERENCE_SURFACE.read_bytes()
    ).hexdigest()
    assert freeze["surface"]["acceptance_status"] == "candidate"
    assert freeze["surface"]["calibration_claim"] is False


def test_queue_onset_derives_from_surface_and_arrivals_only() -> None:
    model = _module("queue_model")
    points = _points()
    freeze = _freeze()
    onset = freeze["queue_model"]["isolated_onset_rate_band_requests_per_second"]

    assert Fraction(222) < _fraction(onset["lower"]) < Fraction(223)
    assert Fraction(225) < _fraction(onset["central"]) < Fraction(226)
    assert Fraction(228) < _fraction(onset["upper"]) < Fraction(229)
    assert freeze["queue_model"]["surface_uncertainty"]["envelope_ppm"] == 12_696
    assert model.predicted_onset_segments(points) == freeze["queue_model"][
        "first_queue_dominated_segment_prediction"
    ]
    assert freeze["queue_model"]["first_queue_dominated_segment_prediction"] == {
        "by_surface_scenario": {
            "central": [225, 230],
            "lower": [225, 230],
            "upper": [220, 225],
        },
        "central": [225, 230],
        "inclusive_admitted_segments": [[220, 225], [225, 230]],
        "reference_configuration": [1, 1, 8],
    }


def test_every_segment_and_held_out_band_rederives_pre_run() -> None:
    model = _module("queue_model")
    points = _points()
    freeze = _freeze()
    configurations = tuple(
        (prefill, decode, prompt)
        for prefill, decode in model.POOL_RATIOS
        for prompt in model.PROMPT_LENGTHS
    )
    segments = [
        row
        for configuration in configurations
        for row in model.predicted_segments(points, configuration)
    ]

    assert segments == freeze["expected_segments"]
    assert len(segments) == 72
    assert model.held_out_points(points) == freeze["held_out"]["prediction_bands"]
    assert len(model.held_out_points(points)) == 30


def test_decomposition_and_preservation_locks_are_frozen() -> None:
    freeze = _freeze()
    decomposition = freeze["decomposition"]
    locks = freeze["preservation_locks"]

    assert "never add provider service" in decomposition["scheduler_queue_wait"]
    assert "sum imported S(batch)" in decomposition["batching_service_per_token"]
    assert locks["prior_load_delay_lineage"] == {
        "artifact_count": 17,
        "manifest_sha256": (
            "ae964f9ccecc2554764f9ef69300ca06a84c4a8609682c678063f73c0d41538d"
        ),
        "selection": locks["prior_load_delay_lineage"]["selection"],
        "total_bytes": 279_928,
    }
    assert locks["core51_one_request_control"]["manifest_sha256"] == (
        "092d79c35c7632e87427804cda11bc6fe0890d2c98ceec23ff30af2e1143ad4d"
    )
    assert locks["deterministic_concurrent_comparator"]["manifest_sha256"] == (
        "d09202846afeba9efc019d4f44f881fde6639457c0398f1ed4ae8da1e0c804c3"
    )
    assert locks["scored_flagship_artifacts"] == {
        "artifact_count": 26,
        "manifest_sha256": (
            "375d2359e0c9dff9cae98c576eaf8a9e24b0c7621b0af0dcfde187662c57955b"
        ),
        "selection": locks["scored_flagship_artifacts"]["selection"],
        "total_bytes": 1_715_149,
    }
