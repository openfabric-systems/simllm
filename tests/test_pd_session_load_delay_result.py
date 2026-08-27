from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples" / "pd_session_load_delay_v1"
RESULTS_PATH = STUDY_DIR / "results.json"


def _result() -> dict:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def _fraction(value: dict[str, int]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def test_compact_result_is_canonical_and_pins_the_scored_run() -> None:
    payload = RESULTS_PATH.read_text(encoding="utf-8")
    result = json.loads(payload)

    assert payload == json.dumps(result, indent=2, sort_keys=True) + "\n"
    assert result["schema"] == "simllm-pd-session-load-delay-compact-result-v1"
    assert result["status"] == "REFUTED"
    assert result["raw_run"] == {
        "bytes": 7_169_930,
        "run_head": "98fb8cf3652826a72ed24d15fec52533e82dc361",
        "sha256": (
            "1521181817ac942318a6fda589b980ee8a5bf523853f19e17a2cf345652dc583"
        ),
    }
    assert result["freeze"] == {
        "commit": "121345e950b12a36018404084c7dcf9bd507f962",
        "expectations_sha256": (
            "28cee81deffe771836b5c38d7fe605185f4dc31a953087c80288ceb7a3a84e22"
        ),
    }


def test_surface_and_exposure_provenance_stay_literal() -> None:
    result = _result()
    surface = result["surface"]

    assert result["exposure"] == {
        "access_ledger_sha256": (
            "0394d2789a11e8dc68c6d3a18c563d19f493d1d27c21d53b3ea74f37b3d14fec"
        ),
        "clean_repetition_residual": "VLLM-40",
        "status": "CONTAMINATED",
        "vllm39_clean_close_permitted": False,
    }
    assert surface["record_sha256"] == (
        "ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52"
    )
    assert surface["acceptance_status"] == "candidate"
    assert surface["calibration_claim"] is False
    assert surface["whole_record_loaded"] is False
    assert surface["held_out_batch_32_decoded_or_captured"] is False
    assert surface["deepseek_row_decoded_or_captured_by_successful_reader"] is False
    assert [point["batch_size"] for point in surface["points"]] == [1, 8]
    assert [point["evidence_class"] for point in surface["points"]] == [
        "MEASURED",
        "MEASURED",
    ]


def test_scored_result_conserves_and_publishes_every_verdict() -> None:
    result = _result()

    assert result["fatal_guards"] == {"findings": [], "status": "HELD"}
    assert result["conservation"] == {
        "admissions": 2_304,
        "cells": 36,
        "handoffs": 2_304,
        "maximum_ttft_residual_ps": 0,
        "terminal_decode_tokens": 9_216,
        "terminals": 2_304,
    }
    assert result["core51_control"]["held"] is True
    assert len(result["curve_records"]) == 6
    assert all(len(curve["points"]) == 6 for curve in result["curve_records"])
    assert len(result["decomposition_rows"]) == 36
    assert result["direction"]["summary"] == {
        "evaluated": 30,
        "matched": 16,
        "observed_decreases": 0,
        "observed_flats": 0,
        "observed_increases": 30,
    }
    assert len(result["direction"]["segment_verdicts"]) == 30
    assert {
        row["observed_direction"]
        for row in result["direction"]["segment_verdicts"]
    } == {"increase"}
    assert result["monotonic_delay_claim"] == "VALIDATED"
    assert result["measured_mechanism"] == (
        "scheduler queue wait dominates every observed segment"
    )


def test_held_out_bands_refute_honestly_and_knees_move_earlier() -> None:
    result = _result()
    bands = result["held_out_bands"]

    assert bands["summary"] == {"evaluated": 24, "held": 1}
    held = [row for row in bands["verdicts"] if row["held"]]
    assert [(row["configuration"], row["offered_load_requests_per_second"]) for row in held] == [
        ([2, 1, 8], 8_000)
    ]
    assert len(result["knees"]) == 6
    assert {tuple(row["observed_bracket"]) for row in result["knees"]} == {
        (250, 500)
    }
    predicted = {
        row["configuration"][1]: _fraction(row["predicted_requests_per_second"])
        for row in result["knees"]
    }
    assert predicted == {
        1: Fraction(4_000_000_000, 3_785_663),
        2: Fraction(8_000_000_000, 3_785_663),
    }


def test_batching_gain_and_queue_wait_are_separate_and_preserved() -> None:
    result = _result()
    by_configuration: dict[tuple[int, int, int], list[dict]] = {}
    for row in result["decomposition_rows"]:
        by_configuration.setdefault(tuple(row["cell"][:3]), []).append(row)

    assert len(by_configuration) == 6
    for rows in by_configuration.values():
        rows.sort(key=lambda row: row["cell"][3])
        assert _fraction(rows[-1]["amortized_decode_batch_service_per_token_ps"]) < (
            _fraction(rows[0]["amortized_decode_batch_service_per_token_ps"])
        )
        assert _fraction(rows[-1]["mean_scheduler_queue_wait_ps"]) > _fraction(
            rows[0]["mean_scheduler_queue_wait_ps"]
        )

    locks = result["preservation_locks"]
    assert locks["core51_one_request_control"]["manifest_sha256"] == (
        "092d79c35c7632e87427804cda11bc6fe0890d2c98ceec23ff30af2e1143ad4d"
    )
    assert locks["deterministic_concurrent_comparator"]["manifest_sha256"] == (
        "d09202846afeba9efc019d4f44f881fde6639457c0398f1ed4ae8da1e0c804c3"
    )
    assert locks["scored_flagship_artifacts"]["manifest_sha256"] == (
        "7630ebdaf91a722ff5004184a03a38fac98bbf11f2adbbfd5e8e32838ff130d5"
    )


def test_registry_keeps_vllm39_open_and_registers_reserved_residuals() -> None:
    module = (REPOSITORY_ROOT / "docs" / "modules" / "adapters-vllm.md").read_text(
        encoding="utf-8"
    )

    assert "- VLLM-39 (Precision; P1; M):" in module
    assert "- VLLM-40 (Precision; P1; M):" in module
    assert "- VLLM-41 (Precision; P1; M):" in module
    assert "it stays open until VLLM-40 permits the literal" in module
