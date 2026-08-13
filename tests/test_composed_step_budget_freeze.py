"""Regression checks for the frozen composed-step budget contract."""

from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
CHECK_PATH = REPOSITORY / "examples/composed_step_budget_v1/check_only.py"


def _check_module():
    spec = importlib.util.spec_from_file_location("composed_step_budget_check", CHECK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_arithmetic_is_derived_from_the_installed_calibrations():
    module = _check_module()
    frozen = module.load_expectations()

    derived = module.check_arithmetic(frozen)

    assert derived["collective_floor_total_ps"] == 1_446_145_392
    assert derived["quantized_service_ps"]["on-graph440-400g"] == 356_095_000
    assert derived["quantized_service_ps"]["on-eager567-400g"] == 1_340_533_000
    assert derived["overlapped_point_ps"] == 1_650_672_126
    assert derived["additive_points_ps"]["additive_graph440"] == 1_907_743_126
    assert derived["additive_points_ps"]["additive_eager567"] == 2_892_181_126
    assert derived["traffic_coverage_added_ps"] == 1_446_145_392
    assert derived["endpoint_envelope_bytes"] == [14, 458_752]


def test_frozen_intervals_discriminate_the_two_compositions():
    module = _check_module()
    intervals = module.load_expectations()["intervals_ps"]

    for label, additive in intervals["additive"].items():
        overlapped = intervals["overlapped"][label]
        assert additive[0] > overlapped[1], label

    assert (
        intervals["additive"]["on-eager567-400g"][0]
        > intervals["additive"]["on-graph440-400g"][1]
    )


def test_check_refuses_an_interval_pair_that_cannot_discriminate():
    module = _check_module()
    frozen = copy.deepcopy(module.load_expectations())
    frozen["intervals_ps"]["ideal_compute"] = [95_000_000, 400_000_000]
    frozen["intervals_ps"]["overlapped"]["on-graph440-400g"] = [
        1_637_145_392,
        1_986_145_392,
    ]

    with pytest.raises(SystemExit, match="discriminate"):
        module.check_arithmetic(frozen)


def test_check_refuses_a_drifted_frozen_constant():
    module = _check_module()
    frozen = copy.deepcopy(module.load_expectations())
    frozen["constants_ps"]["quantized_graph440"] = 356_094_640

    with pytest.raises(SystemExit, match="graph quantized service"):
        module.check_arithmetic(frozen)


def test_check_only_refuses_missing_inputs_and_writes_nothing(tmp_path):
    missing = tmp_path / "absent"

    completed = subprocess.run(
        [
            sys.executable,
            str(CHECK_PATH),
            "--cache-dir",
            str(missing),
            "--htsim-rnic",
            str(missing),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "pinned model snapshot is missing" in completed.stderr
    assert not missing.exists()
    assert list(tmp_path.iterdir()) == []
