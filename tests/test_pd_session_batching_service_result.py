from __future__ import annotations

import importlib.util
import json
from fractions import Fraction
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples" / "pd_session_batching_service_v1"
RESULT_PATH = STUDY_DIR / "non_held_out_results.json"
COMBINED_RESULT_PATH = STUDY_DIR / "results.json"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fraction(value: dict[str, int]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def test_non_held_out_result_passes_every_frozen_band() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["status"] == "PASS"
    assert result["split"] == "non-held-out"
    assert result["fatal_guards"] == {"status": "HELD", "findings": []}
    assert result["service_band_summary"] == {
        "held": 48,
        "missed": 0,
        "evaluated": 48,
    }
    assert len(result["service_band_verdicts"]) == 48
    assert all(row["held"] for row in result["service_band_verdicts"])


def test_non_held_out_result_contains_no_held_out_cell() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert all(
        row["cell"][3] != 240 and row["cell"][:2] != [2, 1]
        for row in result["service_band_verdicts"]
    )


def test_non_held_out_conservation_and_fields_remain_separate() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["conservation"] == {
        "cells": 48,
        "admissions": 3072,
        "handoffs": 3072,
        "terminals": 3072,
        "terminal_decode_tokens": 12288,
        "maximum_ttft_residual_ps": 0,
    }
    assert result["separate_fields"] == {
        "arrival_to_prefill_published": True,
        "handoff_to_decode_published": True,
        "batching_service_published": True,
    }
    assert all(
        "arrival_to_prefill_wait_ps" in row
        and "handoff_to_decode_admission_wait_ps" in row
        and "observed_batching_service_per_token_ps" in row
        for row in result["service_band_verdicts"]
    )


def test_non_held_out_values_satisfy_physics_and_signed_relations() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    sanity = result["physical_sanity"]
    rows = {tuple(row["cell"]): row for row in result["service_band_verdicts"]}

    assert sanity["all_observations_inside_physical_bounds"] is True
    assert sanity["scored"] is False
    ceiling = _fraction(sanity["ceiling_service_per_token_ps"])
    assert all(
        _fraction(rows[(prefill, decode, prompt, 50)][
            "observed_batching_service_per_token_ps"
        ])
        == ceiling
        for prefill, decode in ((1, 1), (1, 2))
        for prompt in (8, 16)
    )
    for prompt in (8, 16):
        one_decode = _fraction(
            rows[(1, 1, prompt, 250)][
                "observed_batching_service_per_token_ps"
            ]
        )
        two_decode = _fraction(
            rows[(1, 2, prompt, 250)][
                "observed_batching_service_per_token_ps"
            ]
        )
        assert one_decode < two_decode < ceiling


def test_non_held_out_markdown_is_a_projection_of_json() -> None:
    publisher = _module(
        STUDY_DIR / "publish_results.py",
        "vllm42_non_held_out_result_publisher",
    )
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    markdown = (STUDY_DIR / "NON_HELD_OUT_RESULTS.md").read_text(
        encoding="utf-8"
    )

    assert publisher.render_split_markdown(result) == markdown


def test_combined_result_qualifies_vllm42_without_a_residual() -> None:
    result = json.loads(COMBINED_RESULT_PATH.read_text(encoding="utf-8"))

    assert result["status"] == "PASS"
    assert result["closure"] == {
        "VLLM-42": "QUALIFIED_PENDING_INTEGRATOR_REGISTRY_COMMIT",
        "VLLM-50": "UNUSED",
    }
    assert result["service_band_summary"] == {
        "held": 78,
        "missed": 0,
        "evaluated": 78,
        "non_held_out_held": 48,
        "held_out_held": 30,
    }
    assert len(result["service_band_verdicts"]) == 78
    assert len({tuple(row["cell"]) for row in result["service_band_verdicts"]}) == 78


def test_combined_result_preserves_chronology_access_and_prior_claims() -> None:
    result = json.loads(COMBINED_RESULT_PATH.read_text(encoding="utf-8"))

    assert result["holdout"]["disclosure_order_held"] is True
    assert result["holdout"]["non_held_out_publication_commit"]
    assert result["guarded_record_access"] == {
        "successful_field_accesses": 0,
        "whole_record_loaded": False,
        "forbidden_access_ledger": [],
    }
    assert result["preservation"]["worktree_bytes_held"] is True
    assert result["settled_claims"] == {
        "onset": "PRESERVED_NOT_RESCORED",
        "monotonic_250_to_8000": "PRESERVED_NOT_RESCORED",
    }
    assert result["predictor"]["observed_curve_inputs"] == []
    assert result["predictor"]["fit_parameters"] == []


def test_combined_conservation_and_physical_sanity_hold() -> None:
    result = json.loads(COMBINED_RESULT_PATH.read_text(encoding="utf-8"))

    assert result["conservation"] == {
        "cells": 78,
        "admissions": 4992,
        "handoffs": 4992,
        "terminals": 4992,
        "terminal_decode_tokens": 19968,
        "maximum_ttft_residual_ps": 0,
    }
    sanity = result["physical_sanity"]
    assert sanity["all_observations_inside_physical_bounds"] is True
    assert sanity["scored"] is False
    assert _fraction(sanity["floor_service_per_token_ps"]) <= _fraction(
        sanity["observed_minimum_service_per_token_ps"]
    )
    assert _fraction(sanity["observed_maximum_service_per_token_ps"]) <= _fraction(
        sanity["ceiling_service_per_token_ps"]
    )


def test_combined_markdown_is_a_projection_of_json() -> None:
    publisher = _module(
        STUDY_DIR / "publish_results.py",
        "vllm42_combined_result_publisher",
    )
    result = json.loads(COMBINED_RESULT_PATH.read_text(encoding="utf-8"))
    markdown = (STUDY_DIR / "RESULTS.md").read_text(encoding="utf-8")

    assert publisher.render_combined_markdown(result) == markdown
