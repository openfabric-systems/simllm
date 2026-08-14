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


def test_the_run_directory_has_no_personal_default():
    study = _study_module()

    with pytest.raises(SystemExit, match=study.RUN_ROOT_ENV):
        study._resolve_run_dir(None)
