from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/external_db_parity_v1"
ARTIFACT = (
    ROOT
    / "offline/calibration/external-databases"
    / "85e72f990f00ea457de522d0b773e678f5e067740689912df5646f6296273284"
)
RECORD = STUDY / "record.json"
RESULTS_CSV = STUDY / "results.csv"
RECORD_SHA256 = "cc8ef61ee6615b070a0d5bb12998334dc60ca7b913c468e0c75ddea3e5541e97"
RESULTS_CSV_SHA256 = "6d9ccae2438b478fa26312a2388a0aaf4ceb954bfa6c69c29b24409ab15a1bdd"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tracked_record_is_locked_and_nonvoid() -> None:
    assert _sha256(RECORD) == RECORD_SHA256
    assert _sha256(RESULTS_CSV) == RESULTS_CSV_SHA256
    assert b"\r" not in RESULTS_CSV.read_bytes()
    assert len(RESULTS_CSV.read_text(encoding="utf-8").splitlines()) == 65

    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["schema"] == "simllm-external-db-parity-record-v1"
    assert record["run_state"] == "nonvoid"
    assert record["voiding_guards"] == []
    assert record["family_tallies"] == {
        "I1": {"denominator": 25, "passed": 25},
        "I2": {"denominator": 26, "passed": 26},
        "P1": {"denominator": 4, "passed": 4},
        "W": {"denominator": 1, "passed": 1},
    }
    assert all(record["fatal_guards"].values())
    assert record["ulp_findings"] == []


def test_local_worker_runs_directly_without_pythonpath(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            os.fspath(STUDY / "run_study.py"),
            "--worker",
            "local",
            "--artifact",
            os.fspath(ARTIFACT),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert result["row_count"] == 284717
    assert len(result["i2"]) == 26
    assert len(result["p1"]) == 4
    assert result["evidence"]["all_match"] is True

    record = json.loads(RECORD.read_text(encoding="utf-8"))
    expected_i1 = {
        row["id"]: row["local_hex"]
        for row in record["rows"]
        if row["family"] == "I1" and row["local_hex"]
    }
    assert result["i1"] == expected_i1
    assert result["payload_sha256"] == record["artifact"]["payload_sha256"]


def test_live_sdk_families_match_the_tracked_record_when_available() -> None:
    raw_venv = os.environ.get("SIMLLM_EXTERNAL_AIC_VENV")
    if raw_venv is None:
        pytest.skip(
            "SIMLLM_EXTERNAL_AIC_VENV is absent; live I2 and P1 parity require the pinned SDK"
        )
    venv = Path(raw_venv)
    python = next(
        (path for path in (venv / "bin/python", venv / "Scripts/python.exe") if path.is_file()),
        None,
    )
    if python is None:
        pytest.skip(
            "SIMLLM_EXTERNAL_AIC_VENV has no interpreter; live I2 and P1 parity is unavailable"
        )
    completed = subprocess.run(
        [os.fspath(python), os.fspath(STUDY / "run_study.py"), "--worker", "external"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    expected_i2 = {
        row["id"]: row["external_hex"]
        for row in record["rows"]
        if row["family"] == "I2"
    }
    expected_p1 = {
        row["id"]: row["external_hex"]
        for row in record["rows"]
        if row["family"] == "P1"
    }
    assert {row_id: value["hex"] for row_id, value in result["i2"].items()} == expected_i2
    assert {row_id: value["hex"] for row_id, value in result["p1"].items()} == expected_p1
    assert result["mutation_hex"] == "0x1.02253ae9a795bp-7"
