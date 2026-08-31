from __future__ import annotations

import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples" / "pd_session_queue_onset_v1"
RESULTS_PATH = STUDY_DIR / "results.json"
REPORT_PATH = STUDY_DIR / "RESULTS.md"


def _result() -> dict:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def _publisher():
    path = STUDY_DIR / "publish_results.py"
    spec = importlib.util.spec_from_file_location("pd_queue_onset_publisher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fraction(value: dict[str, int]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def test_compact_result_is_canonical_and_pins_the_raw_run() -> None:
    payload = RESULTS_PATH.read_text(encoding="utf-8")
    result = json.loads(payload)

    assert payload == json.dumps(result, indent=2, sort_keys=True) + "\n"
    assert result["schema"] == "simllm-pd-session-queue-onset-compact-result-v1"
    assert result["status"] == "IDENTIFIED"
    assert result["raw_run"]["path"] == ("$SIMLLM_VLLM41_RUN_ROOT/qualified-sharded-v1/result.json")
    assert result["raw_run"]["sha256"] == (
        "0cdd0f2bf6244d7c3daf75cfbaee5e56fa3fcc95bfb4718bbb11f7e5beca0248"
    )
    assert result["raw_run"]["bytes"] == 18_833_582
    assert result["raw_run"]["run_head"] == ("9d1ad344d9c21fc46c1bfb1c379e692ac231e49f")
    assert result["raw_run"]["runner_exit_status"] == 0
    assert len(result["raw_run"]["cell_run_manifest"]) == 78
    assert hashlib.sha256(RESULTS_PATH.read_bytes()).hexdigest() == (
        "27ec9540979302625a85d2f1f1866e885bb980df69947dc50ee39e52dad26488"
    )


def test_report_is_generated_from_compact_result_and_is_path_portable() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")

    assert report == _publisher().render_report(_result())
    assert report.startswith("# VLLM-41 scheduler queue-wait onset\n")
    assert "/data3/" not in report
    assert "/home/" not in report
    assert "## Predicted versus observed onset" in report
    assert "## Held-out band verdicts" in report
    assert "## Per-cell decomposition" in report
    assert "## Per-segment decomposition" in report


def test_observed_onset_closes_vllm41_without_using_vllm43() -> None:
    result = _result()
    onset = result["onset"]

    assert onset["summary"] == {
        "configurations_inside_prediction_band": 0,
        "configurations_resolved": 6,
        "distinct_observed_segments": [[210, 220]],
        "predicted_central_segment": [225, 230],
        "predicted_inclusive_segments": [[220, 225], [225, 230]],
    }
    assert len(onset["configurations"]) == 6
    assert {
        tuple(row["observed_first_queue_dominated_segment"]) for row in onset["configurations"]
    } == {(210, 220)}
    assert {row["preceding_non_queue_dominated_segments"] for row in onset["configurations"]} == {5}
    assert result["closure"] == {
        "VLLM-41": "CLOSED",
        "VLLM-42": "REGISTER_RESIDUAL",
        "VLLM-43": "UNUSED_RESERVED",
    }


def test_held_out_component_bands_register_vllm42_without_widening() -> None:
    bands = _result()["held_out_bands"]

    assert bands["summary"] == {
        "batch_service_held": 14,
        "evaluated": 30,
        "joint_held": 14,
        "queue_wait_held": 30,
    }
    assert len(bands["verdicts"]) == 30
    assert all(row["queue_wait_held"] for row in bands["verdicts"])
    misses = [row for row in bands["verdicts"] if not row["batch_service_held"]]
    assert len(misses) == 16
    observed_misses = {
        (tuple(row["configuration"]), row["offered_load_requests_per_second"]) for row in misses
    }
    expected_misses = {
        ((1, 1, 8), 240),
        ((1, 1, 16), 240),
        ((1, 2, 8), 240),
        ((1, 2, 16), 240),
    }
    expected_misses.update(
        ((2, 1, prompt), load) for prompt in (8, 16) for load in (225, 230, 235, 240, 245, 250)
    )
    assert observed_misses == expected_misses


def test_every_cell_and_segment_keeps_wait_and_service_separate() -> None:
    result = _result()
    rows = result["decomposition_rows"]
    segments = result["segment_decompositions"]

    assert len(rows) == 78
    assert len(segments) == 72
    assert all(row["held"] for row in rows)
    assert all(row["maximum_ttft_residual_ps"] == 0 for row in rows)
    for row in rows:
        assert _fraction(row["mean_scheduler_queue_wait_ps"]) == (
            _fraction(row["mean_prefill_queue_ps"])
            + _fraction(row["mean_decode_admission_wait_ps"])
        )
        assert "amortized_decode_batch_service_per_token_ps" in row
    observed_first = [row for row in segments if row["from_load"] == 210 and row["to_load"] == 220]
    assert len(observed_first) == 6
    assert all(row["queue_dominated"] for row in observed_first)
    assert all(not row["predicted_queue_dominated"] for row in observed_first)
    assert all(_fraction(row["observed_scheduler_wait_delta_ps"]) > 0 for row in observed_first)


def test_conservation_surface_refusal_and_preservation_are_literal() -> None:
    result = _result()
    freeze = result["freeze"]

    assert result["fatal_guards"] == {"findings": [], "status": "HELD"}
    assert result["conservation"] == {
        "admissions": 4_992,
        "cells": 78,
        "handoffs": 4_992,
        "maximum_ttft_residual_ps": 0,
        "terminal_decode_tokens": 19_968,
        "terminals": 4_992,
    }
    assert freeze["surface"]["acceptance_status"] == "candidate"
    assert freeze["surface"]["calibration_claim"] is False
    assert freeze["queue_model"]["observed_curve_inputs"] == []
    assert freeze["queue_model"]["fit_parameters"] == []
    assert result["total_delay_direction_scored"] is False
    assert result["prior_250_to_8000_monotonic_direction"] == ("PRESERVED_NOT_REOPENED")
    locks = result["preservation_locks"]
    assert locks["prior_load_delay_lineage"]["manifest_sha256"] == (
        "ae964f9ccecc2554764f9ef69300ca06a84c4a8609682c678063f73c0d41538d"
    )
    assert locks["core51_one_request_control"]["manifest_sha256"] == (
        "092d79c35c7632e87427804cda11bc6fe0890d2c98ceec23ff30af2e1143ad4d"
    )
    assert locks["deterministic_concurrent_comparator"]["manifest_sha256"] == (
        "d09202846afeba9efc019d4f44f881fde6639457c0398f1ed4ae8da1e0c804c3"
    )
    assert locks["scored_flagship_artifacts"]["manifest_sha256"] == (
        "375d2359e0c9dff9cae98c576eaf8a9e24b0c7621b0af0dcfde187662c57955b"
    )


def test_registry_closes_vllm41_and_registers_only_vllm42() -> None:
    module = (REPOSITORY_ROOT / "docs" / "modules" / "adapters-vllm.md").read_text(encoding="utf-8")
    ledger = json.loads((REPOSITORY_ROOT / "docs" / "task-ledger.json").read_text(encoding="utf-8"))

    assert "- VLLM-41 (Precision; P1; M):" not in module
    assert "- VLLM-42 (Precision; P1; M):" in module
    assert "- VLLM-43 (Precision; P1; M):" not in module
    assert "VLLM-41" in ledger["closed"]
