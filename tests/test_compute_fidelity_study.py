"""Regression checks for the compute-fidelity study record."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
STUDY_PATH = REPOSITORY / "examples/compute_fidelity_v1/run_study.py"
RESULTS_PATH = REPOSITORY / "examples/compute_fidelity_v1/results.json"
EXPECTATIONS_PATH = REPOSITORY / "examples/compute_fidelity_v1/expectations.json"


def _study_module():
    spec = importlib.util.spec_from_file_location("compute_fidelity_v1", STUDY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("status", [" M tracked.py\n", "?? untracked.txt\n"])
def test_git_revision_refuses_tracked_and_untracked_changes(monkeypatch, status):
    study = _study_module()
    calls: list[tuple[str, ...]] = []

    def fake_run(command, **_kwargs):
        normalized = tuple(command)
        calls.append(normalized)
        return subprocess.CompletedProcess(normalized, 0, stdout=status, stderr="")

    monkeypatch.setattr(study, "_run", fake_run)

    with pytest.raises(RuntimeError, match="production evidence requires a clean worktree"):
        study._git_revision()

    assert calls == [("git", "status", "--porcelain=v1", "--untracked-files=all")]


def test_git_revision_returns_head_only_after_clean_status(monkeypatch):
    study = _study_module()
    revision = "a" * 40
    responses = iter(("", f"{revision}\n"))

    def fake_run(command, **_kwargs):
        normalized = tuple(command)
        return subprocess.CompletedProcess(
            normalized,
            0,
            stdout=next(responses),
            stderr="",
        )

    monkeypatch.setattr(study, "_run", fake_run)

    assert study._git_revision() == revision


def test_production_refuses_dirty_state_before_creating_output(monkeypatch, tmp_path):
    study = _study_module()
    output = tmp_path / "must-not-exist"

    def refuse_dirty():
        raise RuntimeError("production evidence requires a clean worktree")

    monkeypatch.setattr(study, "_git_revision", refuse_dirty)
    args = argparse.Namespace(out=output)

    with pytest.raises(RuntimeError, match="production evidence requires a clean worktree"):
        study._production(args, {})

    assert not output.exists()


def test_tracked_result_is_void_and_records_truthful_capture_provenance():
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    guard = next(
        item
        for item in results["fatal_guards"]
        if item["id"] == "XFER-G4_roofline_has_no_additive_term"
    )

    assert results["run_status"] == "void"
    assert results["behavioral_score_interpretable"] is False
    assert results["fatal_guard_failures"] == [guard["id"]]
    assert results["observed_commit"] is None
    assert results["capture_provenance"] == {
        "capture_harness_content_commit": "96632de6ea1eb37285c271a86848ee739a7d2332",
        "capture_harness_content_commit_timing": "committed_after_capture",
        "capture_harness_sha256": (
            "7b24342d4f3923f12aeb78525506bfe8f3691be0c6dd94bab5e07c7528c352a8"
        ),
        "recorded_head_commit": "7de897701e93d16873bf9350139fa9f71d870703",
        "repository_clean": False,
    }
    assert guard["passed"] is False
    assert guard["detail"]["proportionality_residual_ps"] == 1
    assert guard["detail"]["post_specified_zero_work_check_passed"] is True
    assert guard["detail"]["zero_work_ps"] == 0


def test_frozen_xfer_guard_fails_while_zero_work_check_holds():
    study = _study_module()
    expectations = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    guard = next(
        item
        for item in study._xfer_guards(expectations)
        if item["id"] == "XFER-G4_roofline_has_no_additive_term"
    )

    assert guard["passed"] is False
    assert guard["detail"]["proportionality_residual_ps"] == 1
    assert guard["detail"]["post_specified_zero_work_check_passed"] is True


def test_projection_rows_use_corrected_anchors_and_omitted_excess():
    expectations = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    fixed = expectations["fixed_cost"]
    modeled_compute = fixed["modeled_decode_compute_us"]
    prefill = fixed["published_prefill_step0_makespan_ps"] / 1.0e6
    decode_low, decode_high = (
        value / 1.0e6 for value in fixed["published_decode_step_makespan_band_ps"]
    )

    assert prefill == 706.622768
    assert (decode_low, decode_high) == (204.0, 215.0)
    for name, bounds in results["bounds"].items():
        assert bounds["omitted_multiple_low"] == pytest.approx(
            bounds["omitted_low_us"] / modeled_compute
        )
        assert bounds["omitted_multiple_high"] == pytest.approx(
            bounds["omitted_high_us"] / modeled_compute
        )
        ttft = results["ttft_projection"][name]
        assert ttft["modeled_ttft_us"] == prefill
        assert ttft["ttft_with_fixed_low_us"] == pytest.approx(
            prefill + bounds["omitted_low_us"]
        )
        assert ttft["ttft_with_fixed_high_us"] == pytest.approx(
            prefill + bounds["omitted_high_us"]
        )
        decode = results["decode_step_projection"][name]
        assert decode["step_with_fixed_low_us"] == pytest.approx(
            decode_low + bounds["omitted_low_us"]
        )
        assert decode["step_with_fixed_high_us"] == pytest.approx(
            decode_high + bounds["omitted_high_us"]
        )


def test_reports_use_omitted_excess_headline_and_void_disposition():
    study_report = (RESULTS_PATH.parent / "RESULTS.md").read_text(encoding="utf-8")
    module_report = (REPOSITORY / "docs/modules/compute.md").read_text(encoding="utf-8")
    normalized_module_report = " ".join(module_report.split())

    for report in (study_report, module_report):
        assert "1.79 to 12.31 times" in report
        assert "2.8 to 13.3 times" not in report
    assert "void with findings" in study_report
    assert "void with findings" in normalized_module_report


def test_published_artifact_hashes_match_lf_normalized_bytes():
    study_report = (RESULTS_PATH.parent / "RESULTS.md").read_text(encoding="utf-8")

    for path in (RESULTS_PATH, EXPECTATIONS_PATH):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest in study_report
