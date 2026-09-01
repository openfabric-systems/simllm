from __future__ import annotations

import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples" / "pd_session_batching_service_v1"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fraction(value: dict[str, int]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def test_freeze_has_complete_split_and_no_observed_curve_input() -> None:
    freeze = json.loads(
        (STUDY_DIR / "expectations.json").read_text(encoding="utf-8")
    )

    assert freeze["status"] == "EXPECTATIONS_ONLY"
    assert freeze["task"] == "VLLM-42"
    assert freeze["chronology"]["successor_run_existed_before_freeze"] is False
    assert freeze["predictor"]["observed_curve_inputs"] == []
    assert freeze["predictor"]["fit_parameters"] == []
    assert len(freeze["prediction_bands"]) == 78
    assert freeze["holdout"]["non_held_out_cell_count"] == 48
    assert freeze["holdout"]["held_out_cell_count"] == 30


def test_holdout_is_the_frozen_union_of_load_and_ratio() -> None:
    freeze = json.loads(
        (STUDY_DIR / "expectations.json").read_text(encoding="utf-8")
    )

    held_out = [
        row for row in freeze["prediction_bands"] if row["split"] == "held-out"
    ]
    assert freeze["holdout"]["loads"] == [240]
    assert freeze["holdout"]["pool_ratios"] == [[2, 1]]
    assert all(
        row["offered_load_requests_per_second"] == 240
        or row["configuration"][:2] == [2, 1]
        for row in held_out
    )


def test_predictions_stay_inside_service_physics() -> None:
    freeze = json.loads(
        (STUDY_DIR / "expectations.json").read_text(encoding="utf-8")
    )
    floor = _fraction(freeze["physical_bounds"]["floor_service_per_token_ps"])
    ceiling = _fraction(
        freeze["physical_bounds"]["ceiling_service_per_token_ps"]
    )

    assert freeze["physical_bounds"]["all_prediction_bands_inside_bounds"] is True
    assert all(
        floor
        <= _fraction(row["batch_service_per_token_band_ps"]["lower"])
        <= _fraction(row["batch_service_per_token_band_ps"]["upper"])
        <= ceiling
        for row in freeze["prediction_bands"]
    )


def test_phase_complete_model_has_the_frozen_signed_mechanism() -> None:
    freeze = json.loads(
        (STUDY_DIR / "expectations.json").read_text(encoding="utf-8")
    )
    changed = freeze["predictor"]["phase_changed_cells"]

    assert changed
    assert all(_fraction(row["signed_delta_ps"]) <= 0 for row in changed)
    assert any(
        row["offered_load_requests_per_second"] in {225, 230, 235, 240, 245, 250}
        for row in changed
    )


def test_preservation_manifest_covers_queue_onset_and_earlier_sessions() -> None:
    freeze = json.loads(
        (STUDY_DIR / "expectations.json").read_text(encoding="utf-8")
    )
    preservation = freeze["preservation"]

    assert preservation["queue_onset_artifact_count"] > 0
    assert preservation["artifact_count"] > preservation["queue_onset_artifact_count"]
    assert all(
        not row["path"].startswith("examples/pd_session_batching_service_v1/")
        for row in preservation["rows"]
    )


def test_builder_reproduces_committed_freeze() -> None:
    previous = sys.modules.get("service_model")
    sys.modules["service_model"] = _module(
        STUDY_DIR / "service_model.py",
        "vllm42_service_model",
    )
    try:
        builder = _module(
            STUDY_DIR / "freeze_expectations.py",
            "vllm42_freeze_builder",
        )
    finally:
        if previous is None:
            del sys.modules["service_model"]
        else:
            sys.modules["service_model"] = previous
    committed = json.loads(
        (STUDY_DIR / "expectations.json").read_text(encoding="utf-8")
    )
    markdown = (STUDY_DIR / "EXPECTATIONS.md").read_text(encoding="utf-8")

    assert builder.build_freeze() == committed
    assert builder.render_markdown(committed) == markdown
