from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = REPOSITORY_ROOT / "examples/pd_session_v1/run_study.py"


def _study():
    spec = importlib.util.spec_from_file_location("pd_session_v1_study", STUDY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cell(prompt_tokens, handoff_ps, ttft_ps, prefill_ps, decode_ps, tpot_ps):
    return {
        "label": f"prompt-{prompt_tokens}-handoff-{handoff_ps}",
        "prompt_tokens": prompt_tokens,
        "handoff_ps": handoff_ps,
        "kv_bytes": prompt_tokens * 49_152,
        "ttft_ps": ttft_ps,
        "tpot_ps": {"numerator": tpot_ps, "denominator": 1},
        "prefill_service_ps": prefill_ps,
        "decode_admission_wait_ps": 0,
        "decode_first_token_service_ps": decode_ps,
        "decomposition_total_ps": ttft_ps,
        "decomposition_residual_ps": 0,
        "decode_token_ids": [512] * 4,
    }


def test_frozen_registry_arithmetic_is_self_consistent():
    study = _study()
    frozen = json.loads(
        (REPOSITORY_ROOT / "examples/pd_session_v1/expectations.json").read_text()
    )

    study._validate_frozen_arithmetic(frozen)


def test_behavior_analysis_keeps_exact_and_scored_classes_separate():
    study = _study()
    behavior = {
        "cells": [
            _cell(8, 100_000_000, 270_000_000, 90_000_000, 80_000_000, 78_000_000),
            _cell(8, 200_000_000, 370_000_000, 90_000_000, 80_000_000, 78_000_000),
            _cell(16, 100_000_000, 290_000_000, 105_000_000, 85_000_000, 79_000_000),
            _cell(16, 200_000_000, 390_000_000, 105_000_000, 85_000_000, 79_000_000),
        ]
    }

    result = study.analyze_behavior(behavior)

    assert len(result["exact_oracle_rows"]) == 4
    assert all(row["held"] for row in result["exact_oracle_rows"])
    assert result["behavioral_instances"] == 6
    assert result["behavioral_held"] == 6


def test_behavior_analysis_catches_handoff_leaking_into_tpot():
    study = _study()
    behavior = {
        "cells": [
            _cell(8, 100_000_000, 270, 90, 80, 78),
            _cell(8, 200_000_000, 100_000_270, 90, 80, 79),
            _cell(16, 100_000_000, 290, 105, 85, 79),
            _cell(16, 200_000_000, 100_000_290, 105, 85, 79),
        ]
    }

    result = study.analyze_behavior(behavior)

    handoff_rows = [
        row for row in result["behavioral_relations"] if row["family"] == "handoff-movement"
    ]
    assert [row["held"] for row in handoff_rows] == [False, True]


def test_scale_child_command_uses_posix_rendering_for_windows_paths():
    study = _study()
    args = SimpleNamespace(
        vllm_python=PureWindowsPath("C:/env/python.exe"),
        vllm_source=PureWindowsPath("C:/env/site-packages/vllm"),
        model_config=PureWindowsPath("D:/cache/model/config.json"),
    )

    command = study.scale_child_command(
        args,
        prefill_engines=2,
        decode_engines=2,
        run_dir=PureWindowsPath("D:/runs/core51/2p-2d"),
        output=PureWindowsPath("D:/runs/core51/scale.json"),
    )

    assert all("\\" not in value for value in command)
    assert "C:/env/python.exe" in command
    assert "D:/runs/core51/2p-2d" in command


def test_scale_summary_is_descriptive_and_never_claims_fit():
    study = _study()
    cells = [
        {
            "prefill_engines": 1,
            "decode_engines": 1,
            "retained_engines": 2,
            "baseline_peak_rss_kib": 100,
            "final_peak_rss_kib": 300,
            "baseline_current_rss_kib": 90,
            "final_current_rss_kib": 270,
            "total_construction_seconds": 2.0,
        },
        {
            "prefill_engines": 2,
            "decode_engines": 2,
            "retained_engines": 4,
            "baseline_peak_rss_kib": 100,
            "final_peak_rss_kib": 540,
            "baseline_current_rss_kib": 90,
            "final_current_rss_kib": 510,
            "total_construction_seconds": 3.6,
        },
    ]

    result = study.summarize_scale(cells)

    assert result["target_engine_count"] == 56
    assert result["target_incremental_peak_rss_kib_range"] == [5600.0, 6160.0]
    assert result["target_sequential_construction_seconds_range"] == [50.4, 56.0]
    assert result["fit_claim"] is False
