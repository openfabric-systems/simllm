"""Regression checks for the corrected host-step calibration harness."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
STUDY_PATH = REPOSITORY / "examples/host_step_cost_v1/run_calibration.py"
EXPECTATIONS_PATH = REPOSITORY / "examples/host_step_cost_v1/expectations.json"
CALIBRATION_PATH = REPOSITORY / "examples/host_step_cost_v1/calibration.json"


def _study_module():
    spec = importlib.util.spec_from_file_location("host_step_calibration", STUDY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_scalars_preserve_exact_decimal_picoseconds():
    study = _study_module()
    parsed = study._parse_scalars(
        "device_name,NVIDIA GeForce GTX 1660 Ti\n"
        "mode,launch\n"
        "empty_graph_ns,630.124\n"
    )

    assert parsed["device_name"] == "NVIDIA GeForce GTX 1660 Ti"
    assert parsed["mode"] == "launch"
    assert study._picoseconds(parsed["empty_graph_ns"]) == 630_124


@pytest.mark.parametrize(
    "text",
    [
        "empty_graph_ns,630.124,extra\n",
        "empty_graph_ns,630.124\nempty_graph_ns,631.000\n",
    ],
)
def test_probe_scalars_refuse_malformed_or_duplicate_rows(text):
    study = _study_module()

    with pytest.raises(ValueError, match="malformed or duplicate"):
        study._parse_scalars(text)


def test_picosecond_parser_refuses_subpicosecond_values():
    study = _study_module()

    with pytest.raises(ValueError, match="sub-picosecond"):
        study._picoseconds("0.0001")


@pytest.mark.parametrize(
    ("guards", "scored", "expected"),
    [
        ([{"passed": False}], [{"passed": True}], "void"),
        ([{"passed": True}], [{"passed": False}], "not_accepted"),
        ([{"passed": True}], [{"passed": True}], "accepted"),
    ],
)
def test_disposition_keeps_fatal_guards_out_of_the_score(guards, scored, expected):
    study = _study_module()

    assert study._disposition(guards, scored) == expected


def test_clean_head_refuses_untracked_state(monkeypatch):
    study = _study_module()

    def fake_run(command, **_kwargs):
        normalized = tuple(command)
        return subprocess.CompletedProcess(
            normalized,
            0,
            stdout="?? untracked.txt\n",
            stderr="",
        )

    monkeypatch.setattr(study, "_run", fake_run)

    with pytest.raises(RuntimeError, match="clean tracked and untracked"):
        study._clean_head()


def test_frozen_calibration_inventory_and_corrected_oracle():
    study = _study_module()
    expectations = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))

    assert sum(expectations["scored_calibration_relations"].values()) == 4
    assert "scored_live_relations" not in expectations
    assert sum(
        expectations["attempt_two_relations_originally_scored"].values()
    ) == 12
    assert expectations["live_attempt_two"]["genuine_risk_instances"] == 0
    holdout = expectations["live_attempt_three"]
    assert holdout["scored_relations"] == {}
    assert holdout["genuine_risk_instances"] == 0
    assert holdout["entailed_findings"] == 12
    assert sum(holdout["entailed_relations"].values()) == 12
    assert expectations["prior_attempt"]["integer_scaling_residual_ps"] == 1
    assert study._zero_work_oracle() == {
        "zero_work_ps": 0,
        "single_ps": 793_650_793,
        "doubled_ps": 1_587_301_587,
        "positive_scaling_residual_ps": 1,
    }


def test_tracked_calibration_is_accepted_when_present():
    if not CALIBRATION_PATH.exists():
        pytest.skip("calibration capture has not run yet")
    artifact = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))

    assert artifact["run_status"] == "accepted"
    assert artifact["behavioral_score_interpretable"] is True
    assert artifact["fatal_guard_failures"] == []
    assert all(row["passed"] for row in artifact["fatal_guards"])
    assert all(row["passed"] for row in artifact["scored_relations"])
    assert set(artifact["profiles"]) == {
        "turing-cuda-graph",
        "turing-eager-host",
    }
    assert artifact["attempt"] == 3
