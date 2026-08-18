"""Byte lock for the kernel_determinism_v1 artifact.

The committed `results.json` must regenerate byte for byte, its summary must
say the run was nonvoid, and the two mutation controls below must break it.
Without those controls a lock that only checked equality would keep passing if
the harness stopped measuring anything.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = ROOT / "examples" / "kernel_determinism_v1"
STUDY_PATH = STUDY_DIR / "run_study.py"
RESULTS = STUDY_DIR / "results.json"
EXPECTATIONS = STUDY_DIR / "expectations.md"

SPEC = importlib.util.spec_from_file_location(
    "kernel_determinism_v1_study_lock", STUDY_PATH
)
assert SPEC is not None
assert SPEC.loader is not None
study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


def test_the_committed_artifact_regenerates_byte_for_byte():
    rendered = study.render(study.build_payload())

    assert rendered.encode("utf-8") == RESULTS.read_bytes()


def test_the_check_mode_agrees_with_the_committed_bytes(capsys):
    exit_code = study.main(["--check"])
    capsys.readouterr()

    assert exit_code == 0


def test_the_run_was_nonvoid_and_scored_every_frozen_instance():
    summary = json.loads(RESULTS.read_text(encoding="utf-8"))["summary"]

    assert summary["void"] is False
    assert summary["fatal_guards_violated"] == []
    assert summary["fatal_guards"] == 23
    assert summary["controls"] == 3
    assert summary["controls_failed"] == []
    assert summary["scored_total"] == 8
    assert summary["scored_passed"] == 8
    assert summary["derived_failed"] == []


def test_the_artifact_declares_its_schema_and_its_freeze():
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))

    assert payload["schema"] == "simllm-kernel-determinism-v1"
    assert payload["freeze"] == "examples/kernel_determinism_v1/expectations.md"
    assert EXPECTATIONS.exists()


def test_every_evidence_class_is_reported_separately():
    """Counts from different evidence classes are never added into one total."""

    rows = json.loads(RESULTS.read_text(encoding="utf-8"))["rows"]
    classes: dict[str, int] = {}
    for row in rows:
        classes[row["evidence_class"]] = classes.get(row["evidence_class"], 0) + 1

    assert classes == {
        "fatal_guard": 23,
        "control": 3,
        "scored": 8,
        "derived": 5,
        "observation": 8,
    }


def test_the_byte_lock_sees_a_changed_prediction(monkeypatch, tmp_path):
    """First mutation control: a wrong frozen constant must break the artifact."""

    mutated = dict(study.ROOFLINE_PREDICTIONS)
    mutated["A1"] = (mutated["A1"][0] + 1, mutated["A1"][1])
    monkeypatch.setattr(study, "ROOFLINE_PREDICTIONS", mutated)

    payload = study.build_payload()

    assert study.render(payload).encode("utf-8") != RESULTS.read_bytes()
    assert payload["summary"]["scored_failed"] == ["A1"]
    assert study.main(["--check", "--out", str(tmp_path / "results.json")]) == 1


def test_the_byte_lock_sees_a_changed_fixture(monkeypatch):
    """Second mutation control: a changed fixture must break the artifact."""

    monkeypatch.setattr(study, "MECHANISTIC_TRANSACTION_BYTES", 256)

    payload = study.build_payload()

    assert study.render(payload).encode("utf-8") != RESULTS.read_bytes()
    assert payload["summary"]["scored_failed"] == ["B1", "B2", "C1"]


@pytest.mark.parametrize("row_id", ["A1", "A2", "A3", "A4", "B1", "B2", "C1", "D1"])
def test_every_frozen_scored_instance_is_present_and_passing(row_id):
    rows = {
        row["id"]: row
        for row in json.loads(RESULTS.read_text(encoding="utf-8"))["rows"]
    }

    assert rows[row_id]["evidence_class"] == "scored"
    assert rows[row_id]["status"] == "PASS"
