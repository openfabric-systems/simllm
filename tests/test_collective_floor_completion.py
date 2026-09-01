"""Lock the TRAF-76 aggregate-completion publication."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/collective_floor_calibration_v1"
RECORD_PATH = STUDY / "completion_record.json"
CSV_PATH = STUDY / "completion_results.csv"
RUNNER = STUDY / "run_completion.py"
RECORD_SHA256 = "f784eeb48fdf06cf7e7f7814d2d3554b9394307a0e0311a0c913522f2315f3d5"
CSV_SHA256 = "2b224e12da10558d1e128560aa4eeae008fb019dcd4302c3a32396401d0e4e08"


def _record() -> dict:
    return json.loads(RECORD_PATH.read_text(encoding="utf-8"))


def test_family_h_completion_is_locked() -> None:
    family = _record()["families"]["H"]
    assert family["status"] == "PASS"
    assert family["passed"] == family["denominator"] == 63
    assert family["median_relative_error"] == 0.02500973884657236
    assert family["p95_relative_error_nearest_rank"] == 0.08695796246648793
    assert family["maximum_relative_error"] == 0.0992616899097621
    assert all(row["passed"] for row in family["rows"])
    assert all(row["above_physical_floor"] for row in family["rows"])
    worst = max(family["rows"], key=lambda row: row["final_relative_error"])
    assert worst["cell_id"] == "half/reduce_scatter/r8/i10"
    assert family["authority_sha256_before_holdout_load"] == (
        "04b694b18c9ccd02963617d71bbdddbed4b0873eab5cc64354126ea3cdb85f09"
    )


def test_d8_model_form_bias_is_resolved_without_moving_the_band() -> None:
    family = _record()["families"]["D8"]
    assert family["status"] == "PASS"
    assert family["band"] == [0.9, 1.1]
    assert family["operation_buffer_bytes"] == 196_608
    assert family["modeled_ms"] == 1.819675065
    assert family["quotient"] == 0.9467365911396686
    assert family["contributions"]["reduce_scatter"]["completion_ps"] == 13_186_667
    assert family["contributions"]["all_gather"]["completion_ps"] == 14_808_334
    assert {
        contribution["rule"]
        for contribution in family["contributions"].values()
    } == {"same-operation-affine"}


def test_packet_family_stops_before_an_unidentified_run() -> None:
    family = _record()["families"]["C"]
    assert family["status"] == "UNDECIDABLE"
    assert family["denominator"] is None
    assert family["before_after_phase_completion_errors"] is None
    assert len(family["cells"]) == 8
    assert {row["receiver_fan_in"] for row in family["cells"]} == {0, 3, 7}
    assert [guard["id"] for guard in family["guards"]] == ["PC-FG-1", "PC-FG-2"]
    assert all(not guard["decidable"] for guard in family["guards"])


def test_all_shared_guards_hold_and_legacy_surfaces_are_exact() -> None:
    record = _record()
    assert record["verdict"] == "INTERPRETABLE"
    assert all(guard["held"] for guard in record["fatal_guards"])
    determinism = next(
        guard for guard in record["fatal_guards"] if guard["id"] == "A-H-FG-6"
    )["finding"]
    assert determinism["fresh_processes"] == 2
    assert determinism["evaluation_1_sha256"] == determinism["evaluation_2_sha256"]
    assert record["bypass"]["held"]
    assert record["minimax_legacy_queries"] == {
        "checked_queries": 16,
        "held": True,
        "mismatches": [],
        "record_sha256": "7f8a3a07867faf18a4f7f307889a9f90e6780eb7c06591a05fad6163ca381f02",
    }


def test_completion_artifacts_and_standalone_check_are_current() -> None:
    record_bytes = RECORD_PATH.read_bytes()
    csv_bytes = CSV_PATH.read_bytes()
    assert b"\r" not in record_bytes
    assert b"\r" not in csv_bytes
    assert hashlib.sha256(record_bytes).hexdigest() == RECORD_SHA256
    assert hashlib.sha256(csv_bytes).hexdigest() == CSV_SHA256
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 63

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, os.fspath(RUNNER), "--check"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert RECORD_SHA256 in completed.stdout
