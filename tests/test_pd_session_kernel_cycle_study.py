from __future__ import annotations

import importlib.util
import json
from pathlib import Path, PureWindowsPath

STUDY = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "pd_session_kernel_cycle_v1"
)


def _runner():
    spec = importlib.util.spec_from_file_location(
        "pd_session_kernel_cycle_study",
        STUDY / "run_study.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_study_pins_the_expectation_and_implementation_commits() -> None:
    runner = _runner()

    assert runner.EXPECTATION_COMMIT == (
        "fda6eed557aef037bf1794da1c1d8556a10a1ee0"
    )
    assert runner.IMPLEMENTATION_COMMIT == (
        "6817019376d153be2a4b6cdd972bbec36dfa23e6"
    )


def test_frozen_grid_has_the_exact_signed_movement_oracle() -> None:
    freeze = json.loads((STUDY / "expectations.json").read_text(encoding="utf-8"))
    rows = freeze["movement_oracle"]["cells"]

    assert len(rows) == 4
    assert [row["signed_ttft_delta_ps"] for row in rows] == [
        0,
        0,
        1_972_200_000,
        1_972_200_000,
    ]
    assert [row["signed_tpot_delta_ps"] for row in rows] == [0, 0, 0, 0]


def test_study_renders_command_paths_with_posix_separators() -> None:
    runner = _runner()

    assert runner.render_cli_path(PureWindowsPath("C:/study/run")) == "C:/study/run"


def test_void_result_retains_exact_movement_without_scoring_it() -> None:
    result = json.loads((STUDY / "results.json").read_text(encoding="utf-8"))

    assert result["status"] == "VOID"
    assert result["voiding_guard"] == "record-absent-cell-or-request-byte-drift"
    assert [row["signed_ttft_delta_ps"] for row in result["cells"]] == [
        0,
        0,
        1_972_200_000,
        1_972_200_000,
    ]
    assert [row["signed_tpot_delta_ps"] for row in result["cells"]] == [
        0,
        0,
        0,
        0,
    ]


def test_record_absent_result_identifies_the_exact_byte_difference() -> None:
    result = json.loads((STUDY / "results.json").read_text(encoding="utf-8"))
    identity = result["record_absent_identity"]

    assert identity["accepted_compact_cells_identical"] is True
    assert identity["kv_bytes_and_timestamps_identical"] is True
    assert identity["complete_request_result_bytes_identical"] is False
    assert identity["differing_fields"] == [
        "decode_internal_request_id",
        "prefill_internal_request_id",
    ]
    assert identity["diagnostic_chronology"] == "post-specified-after-the-void-run"


def test_candidate_provenance_never_claims_calibration_or_total_coverage() -> None:
    result = json.loads((STUDY / "results.json").read_text(encoding="utf-8"))
    candidate = result["candidate_record"]

    assert candidate["acceptance_status"] == "candidate"
    assert candidate["coverage"] == "partial-kernel-subset"
    assert candidate["calibration_claim"] is False
    assert candidate["prefill_lookup_hits"] == 0
    assert candidate["decode_lookup_hits"] == 2
    assert result["task_effect"] == {
        "core_53": "open",
        "core_58": "registered",
        "comp_73": "registered",
        "comp_64": "unchanged-open",
        "milestone": "unchanged",
    }
