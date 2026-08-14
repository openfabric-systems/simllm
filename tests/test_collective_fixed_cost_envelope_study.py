"""Regression checks for the frozen fixed-cost envelope contract."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
STUDY_PATH = REPOSITORY / "examples/collective_fixed_cost_envelope_v1/run_study.py"


def _study_module():
    spec = importlib.util.spec_from_file_location(
        "collective_fixed_cost_envelope_v1",
        STUDY_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_predictions_are_derived_from_the_installed_calibrations():
    study = _study_module()
    frozen = study.load_expectations()

    derived = study.check_arithmetic(frozen)

    assert derived["surcharge_ps"]["off"] == {4: 0, 8: 0}
    assert derived["surcharge_ps"]["floor"] == {4: 0, 8: 0}
    assert derived["surcharge_ps"]["local"] == {4: 15_745_167, 8: 30_128_029}
    assert derived["surcharge_ps"]["cross"] == {4: 24_042_207, 8: 49_487_789}
    assert derived["predicted_step_latency_ps"]["ep8-decode-400g-off"] == 208_707_291
    assert derived["predicted_step_latency_ps"]["ep8-decode-400g-cross"] == (
        2_584_121_163
    )


def test_a_moved_calibration_breaks_the_frozen_predictions():
    study = _study_module()
    frozen = copy.deepcopy(study.load_expectations())
    frozen["predicted_step_latency_ps"]["ep4-decode-400g-local"] += 1

    with pytest.raises(AssertionError, match="not reproducible"):
        study.check_arithmetic(frozen)


def test_the_frozen_sweep_shape_is_the_one_the_runner_executes():
    study = _study_module()
    frozen = study.load_expectations()
    cells = len(study.STUDY_ARMS) * len(study.LINK_RATES_BPS) * len(
        study.EXPERT_PARALLEL_WIDTHS
    )

    assert cells == frozen["expected_cells"]
    assert cells * len(study.PHASES) == frozen["expected_simulated_steps"]
    assert study.EXPECTED_SCORED_FAMILIES == frozen["expected_scored_families"]
    assert set(study.SCORED_RELATION_NAMES) == set(frozen["scored"])
    assert set(study.FATAL_GUARD_NAMES) == set(frozen["fatal_guards"])
    assert set(study.EXACT_UNSCORED_ROW_NAMES) == set(frozen["exact_unscored_rows"])


def test_guards_and_scored_families_never_share_a_relation():
    study = _study_module()
    registered = {
        name: {"passed": True, "held": True}
        for name in study.SCORED_RELATION_NAMES + study.FATAL_GUARD_NAMES
    }

    scored, guards = study._partition_registered_relations(registered)

    assert set(scored) == set(study.SCORED_RELATION_NAMES)
    assert set(guards) == set(study.FATAL_GUARD_NAMES)
    assert set(scored).isdisjoint(guards)
    assert len(scored) == study.EXPECTED_SCORED_FAMILIES


def test_the_ratio_bands_are_tighter_than_the_absolute_band_allows():
    study = _study_module()
    scored = study.load_expectations()["scored"]

    assert scored["S2"]["relative_tolerance"] < scored["S1"]["relative_tolerance"] / 2
    assert scored["S3"]["relative_tolerance"] < scored["S1"]["relative_tolerance"] / 2


def test_the_frozen_width_ratios_bracket_unity_across_the_arms():
    study = _study_module()
    values = study.load_expectations()["scored"]["S2"]["values"]

    for link, arms in values.items():
        ratios = list(arms.values())
        assert min(ratios) < 1.0 < max(ratios), link
        assert arms["off"] == arms["floor"]
        assert arms["cross"] < arms["local"] < arms["off"]


def test_the_frozen_bandwidth_ratios_compress_toward_unity():
    study = _study_module()
    values = study.load_expectations()["scored"]["S3"]["values"]

    for cell, arms in values.items():
        assert arms["off"] == arms["floor"], cell
        assert arms["off"] > arms["local"] > arms["cross"] > 1.0, cell
    assert values["ep8-prefill"]["off"] > values["ep8-decode"]["off"]
    assert values["ep4-prefill"]["off"] > values["ep4-decode"]["off"]


def _recorded_summary():
    import json

    path = REPOSITORY / "examples/collective_fixed_cost_envelope_v1/results-summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_recorded_run_is_not_void_and_every_fatal_guard_held():
    summary = _recorded_summary()

    assert summary["verdict"] == {
        "void": False,
        "scored_families": 3,
        "scored_families_passed": 3,
    }
    assert all(guard["held"] for guard in summary["fatal_guards"].values())
    assert all(row["held"] for row in summary["exact_unscored_rows"].values())
    assert summary["run_configurations"] == {
        "frozen_cells": 16,
        "frozen_simulated_steps": 32,
        "guard_cells": 4,
        "guard_simulated_steps": 8,
    }


def test_the_recorded_latencies_still_satisfy_the_frozen_tolerances():
    study = _study_module()
    summary = _recorded_summary()
    predicted = study.load_expectations()["predicted_step_latency_ps"]
    tolerance = study.load_expectations()["scored"]["S1"]["relative_tolerance"]

    measured = summary["measured_step_latency_ps"]

    assert set(measured) == set(predicted)
    for key, value in measured.items():
        assert abs(value - predicted[key]) / predicted[key] <= tolerance, key


def test_the_recorded_run_reproduces_the_closed_form_to_the_picosecond():
    study = _study_module()
    summary = _recorded_summary()
    surcharge = summary["derived"]["surcharge_ps"]
    collectives = study.load_expectations()["constants_ps"]["collectives_per_step"]

    for key, measured in summary["measured_step_latency_ps"].items():
        width, phase, link, arm = key.split("-")
        cell = f"{width}-{link}-{arm}-{phase}"
        fabric = summary["measured_fabric_service_ps"][cell]
        compute = summary["measured_compute_service_ps"][cell]
        assert len(fabric) == 1, key
        assert measured == compute + collectives * (
            fabric[0] + surcharge[arm][width.removeprefix("ep")]
        ), key


def test_the_tracked_summary_carries_no_machine_specific_path():
    import json

    summary = _recorded_summary()
    text = json.dumps(summary)

    assert summary["environment"]["htsim_rnic_configured"] is True
    assert "htsim_rnic" not in summary["environment"]
    separator = "/"
    for root in ("home", "data", "Users", "scratch", "mnt", "~"):
        segment = root + separator if root == "~" else separator + root
        assert segment not in text, segment


def test_the_run_directory_has_no_personal_default():
    study = _study_module()

    with pytest.raises(SystemExit, match=study.RUN_ROOT_ENV):
        study._resolve_run_dir(None)
