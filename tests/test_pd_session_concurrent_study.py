from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = REPOSITORY_ROOT / "examples/pd_session_concurrent_v1/run_study.py"


def _study():
    spec = importlib.util.spec_from_file_location("pd_session_concurrent_study", STUDY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fraction(value):
    return {"numerator": value, "denominator": 1}


def _request(request_id, admitted, prompt_tokens=8):
    first = admitted + 100
    return {
        "request_id": request_id,
        "expected_admitted_at_ps": admitted,
        "timeline": {
            "request_id": request_id,
            "admitted_at_ps": admitted,
            "ttft_ps": 100,
            "handoff": {"kv_bytes": prompt_tokens * 49_152},
            "decomposition": {
                "total_ps": 100,
                "prefill_service_ps": 40_000_000,
                "decode_first_token_service_ps": 30_000_000,
            },
            "decode_token_completed_at_ps": [first, first + 30, first + 60, first + 90],
        },
        "prefill_engine_id": "prefill-0",
        "decode_engine_id": "decode-0",
        "prefill_internal_request_id": f"prefill-{request_id}",
        "decode_internal_request_id": f"decode-{request_id}",
        "decode_token_ids": [1, 2, 3, 4],
        "kv_transfer_params": {},
        "prefill_step_count": 1,
        "decode_step_count": 4,
        "tpot_ps": _fraction(30_000_000),
    }


def _cell(prefill, decode, prompt, load, throughput, delay):
    requests = [_request(f"{prefill}-{decode}-{prompt}-{load}-{index}", index, prompt) for index in range(8)]
    return {
        "prefill_engines": prefill,
        "decode_engines": decode,
        "prompt_tokens": prompt,
        "offered_load_requests_per_second": load,
        "interarrival_ps": 1,
        "requests": requests,
        "prefill_batches": [[row["request_id"] for row in requests]],
        "decode_batches": [[row["request_id"] for row in requests]],
        "maximum_prefill_batch_size": 8,
        "maximum_decode_batch_size": 8,
        "curve_point": {
            "request_count": 8,
            "output_token_count": 32,
            "aggregated_output_throughput_tokens_per_second": _fraction(throughput),
            "per_token_request_delay_ps": _fraction(delay),
        },
    }


def _observation():
    cells = []
    for prefill, decode in ((1, 1), (1, 2), (2, 1)):
        for prompt in (8, 16):
            for load, throughput, delay in (
                (8_000, 10, 10),
                (16_000, 20, 20),
                (32_000, 30, 30),
            ):
                cells.append(_cell(prefill, decode, prompt, load, throughput, delay))
    return {
        "baseline": {
            "ttft_ps": 273_376_000,
            "tpot_ps": _fraction(77_952_000),
            "decomposition": {
                "prefill_service_ps": 95_424_000,
                "handoff_ps": 100_000_000,
                "decode_first_token_service_ps": 77_952_000,
            },
        },
        "cells": cells,
        "curves": [],
    }


def test_frozen_registry_arithmetic_is_self_consistent():
    study = _study()
    frozen = json.loads(
        (REPOSITORY_ROOT / "examples/pd_session_concurrent_v1/expectations.json").read_text()
    )

    study._validate_frozen_arithmetic(frozen)


def test_load_amendment_is_self_consistent():
    study = _study()
    amendment = json.loads(
        (
            REPOSITORY_ROOT
            / "examples/pd_session_concurrent_v1/expectations-load-amendment.json"
        ).read_text()
    )

    study._validate_load_amendment(amendment)


def test_analysis_keeps_fatal_and_behavioral_evidence_separate():
    study = _study()

    analysis = study.analyze_observation(_observation())

    assert analysis["status"] == "PASS"
    assert analysis["fatal_guards"] == {"status": "HELD", "findings": []}
    assert len(analysis["exact_oracle_rows"]) == 18
    assert analysis["behavioral_total"] == 24
    assert analysis["behavioral_held"] == 24


def test_analysis_reports_a_curve_direction_refutation_without_voiding():
    study = _study()
    observation = _observation()
    observation["cells"][2]["curve_point"][
        "aggregated_output_throughput_tokens_per_second"
    ] = _fraction(1)

    analysis = study.analyze_observation(observation)

    assert analysis["status"] == "REFUTED"
    assert analysis["fatal_guards"]["status"] == "HELD"
    assert analysis["behavioral_held"] == 23


def test_analysis_voids_on_terminal_token_loss():
    study = _study()
    observation = _observation()
    observation["cells"][0]["requests"][0]["decode_token_ids"].pop()

    analysis = study.analyze_observation(observation)

    assert analysis["status"] == "VOID"
    assert analysis["fatal_guards"]["status"] == "VIOLATED"


def test_vllm_version_command_uses_posix_rendering_for_windows_paths(monkeypatch):
    study = _study()
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(stdout="0.27.1\n")

    monkeypatch.setattr(study.subprocess, "run", fake_run)

    assert study._vllm_version(PureWindowsPath("C:/env/python.exe")) == "0.27.1"
    assert captured["command"][0] == "C:/env/python.exe"
    assert "\\" not in captured["command"][0]


def test_study_imports_without_vllm_or_unix_only_resource(monkeypatch):
    original_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "resource" or name == "vllm" or name.startswith("vllm."):
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    study = _study()
    assert study.RESULT_SCHEMA == "simllm-pd-session-concurrent-study-result-v1"
