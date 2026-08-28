from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import math
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
RECORD_SHA256 = "38d616b3245a8a42bd06ee9f79d3397d16476cd4acd46dfecd7bd503a55a0e96"
RESULTS_CSV_SHA256 = "273ad38c8ce309fdc6612c1d620f33b35be85beefb7c754c4585845621dd5687"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "external_db_parity_run_study",
        STUDY / "run_study.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tracked_record_is_locked_and_nonvoid() -> None:
    assert _sha256(RECORD) == RECORD_SHA256
    assert _sha256(RESULTS_CSV) == RESULTS_CSV_SHA256
    assert b"\r" not in RESULTS_CSV.read_bytes()
    assert len(RESULTS_CSV.read_text(encoding="utf-8").splitlines()) == 78

    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["schema"] == "simllm-external-db-parity-record-v1"
    assert record["run_state"] == "nonvoid"
    assert record["voiding_guards"] == []
    assert record["family_tallies"] == {
        "I1": {"denominator": 25, "passed": 25},
        "I2": {"denominator": 26, "passed": 26},
        "I2S": {"denominator": 13, "passed": 13},
        "P1": {"denominator": 4, "passed": 4},
        "W": {"denominator": 1, "passed": 1},
    }
    assert all(record["fatal_guards"].values())
    assert record["ulp_findings"] == []
    assert record["attempt"] == "attempt-0003"
    assert record["freeze_commits"]["review_addendum"].startswith("25dc6b5")
    assert record["freeze_commits"]["cap_supplement"].startswith("a679b0e")
    assert record["scoring_registers"]["I2"] == {
        "denominator": 26,
        "freeze_commit": "afe7ee6e2947616c3b64e6e7c2dbc1fcf3553ef1",
        "specification_status": "pre-specified",
    }
    assert record["scoring_registers"]["I2S"]["denominator"] == 13
    assert record["scoring_registers"]["I2S"]["reporting_rule"] == (
        "report separately from I2 and never sum the two registers"
    )
    assert record["supplement_integration"] == {
        "freeze_commit": "a679b0e85733219f877f520421bf8b45221febaa",
        "freeze_origin": "concurrent session",
        "work_class": "integration of frozen rows, not new specification",
    }
    assert len(record["rows"]) == 77
    assert all(row["freeze_commit"] for row in record["rows"])
    assert all(
        row["specification_status"] in {"pre-specified", "post-specified"}
        for row in record["rows"]
    )
    miss = next(row for row in record["rows"] if row["id"] == "I2S-02")
    assert miss["passed"] is True
    assert miss["local_refusal_kind"] == "InterpolationDataNotAvailableError"
    assert miss["external_refusal_kind"] == "PerfDataNotAvailableError"
    fg5 = next(row for row in record["rows"] if row["id"] == "FG-5")
    assert fg5["specification_status"] == "post-specified"
    assert "25dc6b5" in fg5["provenance_note"]

    with RESULTS_CSV.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(record["rows"])
    assert all(row["freeze_commit"] for row in csv_rows)
    assert all(
        row["specification_status"] in {"pre-specified", "post-specified"}
        for row in csv_rows
    )


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
    assert len(result["i2s"]) == 13
    assert len(result["p1"]) == 4
    assert result["evidence"]["all_match"] is True
    assert result["row_counts"] == result["manifest_row_counts"]
    assert result["row_count"] == result["manifest_row_count"]
    assert result["diagnostics"]["I2-27-cap-off"] == {
        "differs_from_capped": True,
        "frozen_hex": "0x1.cc9259aaacb10p+3",
        "local_hex": "0x1.cc9259aaacb10p+3",
        "ulp_distance_from_capped": 85475858518375,
    }
    assert result["diagnostics"]["I2S-01-cap-off"] == {
        "cap_on_hex": "0x1.649d515151514p-11",
        "cap_on_minus_cap_off_ms": 0.0002849416034136591,
        "differs_from_capped": True,
        "frozen_hex": "0x1.9e7234bf96873p-12",
        "local_hex": "0x1.9e7234bf96873p-12",
        "matches_frozen": True,
        "ulp_distance_from_capped": 3486215443295393,
    }
    assert result["i2s"]["I2S-01"]["hex"] == "0x1.649d515151514p-11"
    assert result["i2s"]["I2S-02"] == {
        "kind": "refusal",
        "refusal_kind": "InterpolationDataNotAvailableError",
    }

    record = json.loads(RECORD.read_text(encoding="utf-8"))
    expected_i1 = {
        row["id"]: row["local_hex"]
        for row in record["rows"]
        if row["family"] == "I1" and row["local_hex"]
    }
    assert result["i1"] == expected_i1
    expected_i2 = {
        row["id"]: row["local_hex"]
        for row in record["rows"]
        if row["family"] == "I2" and int(row["id"].rsplit("-", 1)[1]) <= 26
    }
    regenerated_i2 = {row_id: result["i2"][row_id]["hex"] for row_id in expected_i2}
    assert regenerated_i2 == expected_i2
    expected_i2s = {
        row["id"]: row["local_hex"]
        for row in record["rows"]
        if row["local_hex"]
        and (
            row["family"] == "I2S"
            or (
                row["family"] == "I2"
                and int(row["id"].rsplit("-", 1)[1]) > 26
            )
        )
    }
    regenerated_i2s = {
        row_id: result["i2s"][row_id]["hex"] for row_id in expected_i2s
    }
    assert regenerated_i2s == expected_i2s
    expected_p1 = {
        row["id"]: row["local_hex"]
        for row in record["rows"]
        if row["family"] == "P1"
    }
    assert {row_id: result["p1"][row_id]["hex"] for row_id in expected_p1} == (
        expected_p1
    )
    assert result["payload_sha256"] == record["artifact"]["payload_sha256"]


def test_freeze_working_bytes_match_recorded_git_blobs() -> None:
    runner = _load_runner()
    status = runner._freeze_blob_status()
    assert set(status) == {
        "expectations",
        "query_points",
        "query_points_supplement",
        "review_addendum",
    }
    assert all(entry["git_show_succeeded"] for entry in status.values())
    assert all(entry["matches"] for entry in status.values())


def test_live_worker_failure_writes_compact_void_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    attempt = tmp_path / "attempt-0007"
    attempt.mkdir()

    def fail_worker(**kwargs):
        stderr = kwargs["attempt"] / "external-run-1.stderr.txt"
        stderr.write_text("retained live worker stderr\n", encoding="utf-8")
        raise RuntimeError("external worker repetition 1 failed with status 9")

    monkeypatch.setattr(runner, "_run_worker", fail_worker)
    runs, record = runner._live_worker_runs_or_void(
        python=Path(sys.executable),
        artifact=ARTIFACT,
        attempt=attempt,
        attempt_number=7,
    )
    assert runs is None
    assert record is not None
    assert record["run_state"] == "void"
    assert record["voiding_guards"] == ["LIVE-WORKER"]
    assert record["failure"]["stderr_file"] == "external-run-1.stderr.txt"
    assert record["failure"]["stderr"] == "retained live worker stderr\n"
    assert json.loads((attempt / "record.json").read_text(encoding="utf-8")) == record


def test_i2_and_p1_term_misses_both_publish_ulp_findings() -> None:
    runner = _load_runner()
    local = runner._local_worker(ARTIFACT)
    external = {
        "identity": {
            "slice_file_count": 27,
            "slice_sha256": runner.EXPECTED_IDENTITY["slice_sha256"],
            "closure_sha256": runner.EXPECTED_IDENTITY["closure_sha256"],
            "system_sha256": runner.EXPECTED_IDENTITY["system_sha256"],
            "model_sha256": runner.EXPECTED_IDENTITY["model_sha256"],
        },
        "i2": copy.deepcopy(local["i2"]),
        "i2s": copy.deepcopy(local["i2s"]),
        "p1": copy.deepcopy(local["p1"]),
        "mutation_hex": {
            guard_id: value["query_hex"]
            for guard_id, value in local["mutation"]["guards"].items()
        },
    }
    external["i2s"]["I2S-02"]["refusal_kind"] = "PerfDataNotAvailableError"
    i2_local = float.fromhex(local["i2"]["I2-01"]["hex"])
    external["i2"]["I2-01"]["hex"] = math.nextafter(i2_local, math.inf).hex()
    term = next(iter(external["p1"]["P1-01"]["terms"]))
    term_local = float.fromhex(local["p1"]["P1-01"]["terms"][term])
    external["p1"]["P1-01"]["terms"][term] = math.nextafter(
        term_local,
        math.inf,
    ).hex()

    rows, findings = runner._evaluate(
        local=local,
        external=external,
        local_deterministic=True,
        external_deterministic=True,
        imported_artifact=ARTIFACT,
        attempt_number=2,
        elapsed_seconds=1.0,
    )
    assert any(
        finding.get("family") == "I2" and finding.get("row") == "I2-01"
        for finding in findings["ulp_findings"]
    )
    assert any(
        finding.get("family") == "P1"
        and finding.get("oracle") == "P1-01"
        and finding["diverging_terms"]
        for finding in findings["ulp_findings"]
    )
    p1_row = next(row for row in rows if row["id"] == "P1-01")
    assert p1_row["passed"] is True
    assert runner._family_tallies(rows)["I2"] == {
        "denominator": 26,
        "passed": 25,
    }
    assert runner._family_tallies(rows)["I2S"] == {
        "denominator": 13,
        "passed": 13,
    }
    assert all(row["freeze_commit"] for row in rows)
    assert all(
        row["specification_status"] in {"pre-specified", "post-specified"}
        for row in rows
    )


def test_i2s_structured_miss_scores_refusal_kinds_not_messages() -> None:
    runner = _load_runner()
    local = runner._local_worker(ARTIFACT)
    external = {
        "identity": {
            "slice_file_count": 27,
            "slice_sha256": runner.EXPECTED_IDENTITY["slice_sha256"],
            "closure_sha256": runner.EXPECTED_IDENTITY["closure_sha256"],
            "system_sha256": runner.EXPECTED_IDENTITY["system_sha256"],
            "model_sha256": runner.EXPECTED_IDENTITY["model_sha256"],
        },
        "i2": copy.deepcopy(local["i2"]),
        "i2s": copy.deepcopy(local["i2s"]),
        "p1": copy.deepcopy(local["p1"]),
        "mutation_hex": {
            guard_id: value["query_hex"]
            for guard_id, value in local["mutation"]["guards"].items()
        },
    }
    external["i2s"]["I2S-02"] = {
        "kind": "refusal",
        "refusal_kind": "PerfDataNotAvailableError",
    }
    rows, _ = runner._evaluate(
        local=local,
        external=external,
        local_deterministic=True,
        external_deterministic=True,
        imported_artifact=ARTIFACT,
        attempt_number=2,
        elapsed_seconds=1.0,
    )
    miss = next(row for row in rows if row["id"] == "I2S-02")
    assert miss["passed"] is True
    assert miss["local_refusal_kind"] == "InterpolationDataNotAvailableError"
    assert miss["external_refusal_kind"] == "PerfDataNotAvailableError"

    external["i2s"]["I2S-02"] = {
        "kind": "value",
        "hex": "0x0.0p+0",
    }
    failed_rows, _ = runner._evaluate(
        local=local,
        external=external,
        local_deterministic=True,
        external_deterministic=True,
        imported_artifact=ARTIFACT,
        attempt_number=2,
        elapsed_seconds=1.0,
    )
    failed_miss = next(row for row in failed_rows if row["id"] == "I2S-02")
    assert failed_miss["passed"] is False


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
        if row["family"] == "I2" and int(row["id"].rsplit("-", 1)[1]) <= 26
    }
    expected_i2s = {
        row["id"]: row["external_hex"]
        for row in record["rows"]
        if row["external_hex"]
        and (
            row["family"] == "I2S"
            or (
                row["family"] == "I2"
                and int(row["id"].rsplit("-", 1)[1]) > 26
            )
        )
    }
    expected_p1 = {
        row["id"]: row["external_hex"]
        for row in record["rows"]
        if row["family"] == "P1"
    }
    assert {row_id: result["i2"][row_id]["hex"] for row_id in expected_i2} == expected_i2
    assert {row_id: result["i2s"][row_id]["hex"] for row_id in expected_i2s} == (
        expected_i2s
    )
    assert result["i2s"]["I2S-01"] == {
        "hex": "0x1.649d515151514p-11",
        "kind": "value",
    }
    assert result["i2s"]["I2S-02"] == {
        "kind": "refusal",
        "refusal_kind": "PerfDataNotAvailableError",
    }
    assert {row_id: result["p1"][row_id]["hex"] for row_id in expected_p1} == expected_p1
    assert result["mutation_hex"] == {
        "FG-5-GEMM": "0x1.02253ae9a795bp-7",
        "FG-5-GENERATION-ATTENTION": "0x1.d0d73a2abadb5p-4",
    }
