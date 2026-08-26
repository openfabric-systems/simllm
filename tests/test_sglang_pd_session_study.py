"""Framework-free checks for the frozen SGL-33 study entry point."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/sglang_pd_session_v1/run_study.py"
EXPECTATIONS = ROOT / "examples/sglang_pd_session_v1/expectations.json"
RESULTS = ROOT / "examples/sglang_pd_session_v1/results.json"


def _namespace():
    return runpy.run_path(STUDY)


def test_frozen_registry_and_harness_literals_agree():
    namespace = _namespace()
    frozen = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))

    namespace["_validate_frozen_registry"](frozen)
    assert namespace["POOL_RATIOS"] == ((1, 1), (1, 2), (2, 1))
    assert namespace["OFFERED_LOADS"] == (8_000, 16_000, 32_000)
    assert namespace["INTERARRIVAL_PS"] == (
        125_000_000,
        62_500_000,
        31_250_000,
    )


def test_prompt_cells_change_prefix_without_changing_shape():
    distinct_prompt = _namespace()["_distinct_prompt"]
    base = tuple(range(22))

    first = distinct_prompt(base, 8, 1)
    second = distinct_prompt(base, 8, 2)

    assert len(first) == len(second) == 8
    assert first[0] != second[0]
    assert first[1:] == second[1:] == base[1:8]


def test_flagship_render_is_literal_104_rank_structure():
    flagship = _namespace()["_flagship_render"]()

    assert flagship["nodes"] == 13
    assert flagship["ranks"] == flagship["gpus"] == flagship["nics"] == 104
    assert len(flagship["prefill_ep"]) == 32
    assert len(flagship["decode_ep"]) == 72
    assert flagship["prefill_tp"] == [0]
    assert flagship["decode_tp"] == [32]
    assert flagship["decode_dense_dp"] == [32]
    assert flagship["core54_claimed_ranks"] == 96


def test_identity_conservation_does_not_require_handoff_completion_order():
    identity_ledgers_hold = _namespace()["_identity_ledgers_hold"]
    admitted = [f"request-{index}" for index in range(8)]
    handed_off = [admitted[index] for index in (0, 1, 3, 2, 5, 4, 7, 6)]

    assert identity_ledgers_hold(admitted, handed_off, list(admitted))
    assert not identity_ledgers_hold(admitted, handed_off[:-1], list(admitted))
    assert not identity_ledgers_hold(admitted, handed_off[:-1] + [admitted[0]], list(admitted))


def test_compact_result_preserves_the_deciding_evidence():
    result = json.loads(RESULTS.read_text(encoding="utf-8"))

    assert result["status"] == "REFUTED"
    assert result["fatal_guards"] == {"status": "HELD", "findings": []}
    assert result["conservation"] == {
        "cells": 18,
        "admissions": 144,
        "handoffs": 144,
        "terminals": 144,
        "decode_tokens": 576,
        "maximum_ttft_residual_ps": 0,
    }
    assert result["packet_exact"]["signed_ttft_delta_ps"] == -76_918_400
    assert result["packet_exact"]["metric_residual_ps"] == 0
    assert result["behavioral_families"][
        "throughput_nondecreasing_with_load"
    ] == {
        "held": 4,
        "evaluated": 6,
        "refuted_configuration_ids": [
            "sglang-p1-d2-prompt8",
            "sglang-p2-d1-prompt16",
        ],
    }
    assert result["example_curve"]["schema"] == "simllm-deployment-curve-v1"
    assert len(result["example_curve"]["points"]) == 3
    assert result["task_effect"] == {
        "SGL-33": "OPEN",
        "SGL-35": "OPEN",
        "SGL-36": "OPEN",
        "CORE-57": "OPEN",
    }
