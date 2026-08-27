from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPO_ROOT / "examples" / "pd_session_load_delay_v1"
RESULT_PATH = STUDY_DIR / "vllm40_results.json"
REPORT_PATH = STUDY_DIR / "VLLM40_RESULTS.md"
LEDGER_PATH = STUDY_DIR / "vllm40_access_ledger.jsonl"
MODULE_PATH = REPO_ROOT / "docs" / "modules" / "adapters-vllm.md"
TASK_LEDGER_PATH = REPO_ROOT / "docs" / "task-ledger.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob_sha1(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def _result() -> dict[str, object]:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_vllm40_access_ledger_is_clean_and_exactly_allowlisted():
    result = _result()
    access = result["access"]
    ledger = [
        json.loads(line)
        for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines()
    ]

    assert access["status"] == "CLEAN"
    assert access["field_access_count"] == 5
    assert access["successful_accesses"] == 5
    assert access["whole_record_loaded"] is False
    assert access["deepseek_decoded_or_captured"] is False
    assert access["held_out_batch_32_decoded_or_captured"] is False
    assert access["ledger_sha256"] == _sha256(LEDGER_PATH)
    assert access["returned_selectors"] == [row["selector"] for row in ledger]
    assert all(row["status"] == "PASS" for row in ledger)
    assert all(row["whole_record_loaded"] is False for row in ledger)
    assert [row["bytes_consumed_total"] for row in ledger] == [45043] * 5
    assert "batch_size=1" in ledger[3]["selector"]
    assert "batch_size=8" in ledger[4]["selector"]

    reader = STUDY_DIR / "field_reader.py"
    assert access["field_reader_sha256"] == _sha256(reader)
    assert access["field_reader_git_blob_sha1"] == _git_blob_sha1(reader)


def test_vllm40_surface_and_frozen_inputs_are_identical():
    result = _result()
    surface = result["surface_repetition"]
    freeze = result["freeze"]

    assert surface["status"] == "IDENTICAL"
    assert surface["provenance_identical"] is True
    assert surface["calibration_claim"] is False
    assert [row["batch_size"] for row in surface["points"]] == [1, 8]
    assert all(row["frozen_surface_identical"] for row in surface["points"])
    assert [row["evidence_class"] for row in surface["points"]] == [
        "MEASURED",
        "MEASURED",
    ]
    assert freeze["identities_unchanged"] is True
    for relative, expected in freeze["frozen_input_sha256"].items():
        assert _sha256(REPO_ROOT / relative) == expected

    locks = result["preservation_locks"]
    assert locks["core51_one_request_control"] == {
        "artifact_count": 6,
        "manifest_sha256": (
            "092d79c35c7632e87427804cda11bc6fe0890d2c98ceec23ff30af2e1143ad4d"
        ),
        "total_bytes": 61248,
    }
    assert locks["deterministic_concurrent_comparator"] == {
        "artifact_count": 9,
        "manifest_sha256": (
            "d09202846afeba9efc019d4f44f881fde6639457c0398f1ed4ae8da1e0c804c3"
        ),
        "total_bytes": 56495,
    }
    assert locks["scored_flagship_artifacts"] == {
        "artifact_count": 17,
        "manifest_sha256": (
            "7630ebdaf91a722ff5004184a03a38fac98bbf11f2adbbfd5e8e32838ff130d5"
        ),
        "total_bytes": 1198680,
    }


def test_vllm40_publishes_the_complete_refutation():
    result = _result()
    claim = result["claim_verdict"]
    conservation = result["conservation"]
    segments = result["segment_verdicts"]
    bands = result["held_out_band_verdicts"]

    assert result["status"] == "REFUTED"
    assert result["clean_repetition_status"] == "PASS"
    assert claim["runner_analysis_status"] == "REFUTED"
    assert claim["monotonic_delay_claim"] == "VALIDATED"
    assert claim["direction_summary"] == {
        "evaluated": 30,
        "matched": 16,
        "observed_decreases": 0,
        "observed_flats": 0,
        "observed_increases": 30,
    }
    assert claim["held_out_band_summary"] == {"evaluated": 24, "held": 1}
    assert len(segments) == 30
    assert all(row["observed_direction"] == "increase" for row in segments)
    assert sum(row["held"] for row in segments) == 16
    assert len(bands) == 24
    assert sum(row["held"] for row in bands) == 1

    summary = conservation["summary"]
    assert conservation["all_exact_rows_held"] is True
    assert conservation["fatal_guards"] == {"findings": [], "status": "HELD"}
    assert len(conservation["rows"]) == 36
    assert all(row["held"] for row in conservation["rows"])
    assert summary == {
        "admissions": 2304,
        "cells": 36,
        "handoffs": 2304,
        "maximum_ttft_residual_ps": 0,
        "terminal_decode_tokens": 9216,
        "terminals": 2304,
    }
    assert result["batching_evidence"] == {
        "cells_with_decode_batch_above_one": 36,
        "cells_with_prefill_batch_above_one": 26,
        "maximum_decode_batch_size": 8,
        "maximum_prefill_batch_size": 8,
    }


def test_vllm40_registry_closure_and_report_are_literal():
    result = _result()
    task_ledger = json.loads(TASK_LEDGER_PATH.read_text(encoding="utf-8"))
    module = MODULE_PATH.read_text(encoding="utf-8")
    report = REPORT_PATH.read_text(encoding="utf-8")

    assert result["closure"] == {
        "VLLM-35": "CLOSED",
        "VLLM-39": "CLOSED",
        "VLLM-40": "CLOSED",
        "VLLM-41": "UNCHANGED_OPEN",
        "VLLM-42": "RESERVED_UNTOUCHED",
    }
    assert {"VLLM-35", "VLLM-39", "VLLM-40"} <= set(task_ledger["closed"])
    assert "VLLM-41" not in task_ledger["closed"]
    assert "- VLLM-35 " not in module
    assert "- VLLM-39 " not in module
    assert "- VLLM-40 " not in module
    assert "- VLLM-41 " in module
    assert report.startswith("# VLLM-40 clean load-delay qualification\n")
    assert "Frozen qualification status: **REFUTED**" in report
    assert "Monotonic-delay claim: **VALIDATED**" in report
    assert "VLLM-41 unchanged and open" in report
