from __future__ import annotations

import importlib.util
import json
from fractions import Fraction
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples" / "pd_session_batching_service_v1"
RESULT_PATH = STUDY_DIR / "non_held_out_results.json"


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
