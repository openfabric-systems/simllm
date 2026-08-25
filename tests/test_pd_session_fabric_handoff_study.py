from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = (
    REPOSITORY_ROOT / "examples/pd_session_fabric_handoff_v1/run_study.py"
)
EXPECTATIONS_PATH = (
    REPOSITORY_ROOT / "examples/pd_session_fabric_handoff_v1/expectations.json"
)


def _study():
    spec = importlib.util.spec_from_file_location("pd_fabric_handoff_study", STUDY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fraction(value):
    return {"numerator": value, "denominator": 1}


def _control(prompt_tokens):
    constant_ttft = 273_376_000 if prompt_tokens == 8 else 292_912_000
    tpot = 77_952_000 if prompt_tokens == 8 else 77_976_000
    tokens = [512, 512, 512, 512]
    return {
        "off": {
            "timeline": {
                "ttft_ps": constant_ttft - 100_000_000,
                "decomposition": {"handoff_ps": 0},
            },
            "decode_token_ids": tokens,
            "kv_transfer_params": {},
            "tpot_ps": _fraction(tpot),
        },
        "constant": {
            "timeline": {
                "ttft_ps": constant_ttft,
                "decomposition": {"handoff_ps": 100_000_000},
            },
            "decode_token_ids": tokens,
            "kv_transfer_params": {},
            "tpot_ps": _fraction(tpot),
        },
        "packet_artifact_count_before_controls": 0,
        "packet_artifact_count_after_controls": 0,
    }


def _cell(prompt_tokens, bandwidth, service_ps):
    aggregate = prompt_tokens * 49_152
    chunk = aggregate // 8
    duration = 20_000_000 + service_ps
    submitted = 1_000
    constant = _control(prompt_tokens)["constant"]
    messages = [
        {
            "source_rank": index,
            "destination_rank": index + 8,
            "tag": 6_200,
            "payload_bytes": chunk,
        }
        for index in range(8)
    ]
    flows = [
        {
            "source": index,
            "destination": index + 8,
            "tag": 6_200,
            "payload_bytes": chunk,
            "start_time_ps": 20_000_000,
            "completion_time_ps": duration,
        }
        for index in range(8)
    ]
    return {
        "prompt_tokens": prompt_tokens,
        "link_bandwidth_bps": bandwidth,
        "packet": {
            "timeline": {
                "ttft_ps": constant["timeline"]["ttft_ps"]
                + duration
                - 100_000_000,
                "handoff": {
                    "kv_bytes": aggregate,
                    "submitted_at_ps": submitted,
                    "eligible_at_ps": submitted + 20_000_000,
                    "started_at_ps": submitted + 20_000_000,
                    "finished_at_ps": submitted + duration,
                    "completed_at_ps": submitted + duration,
                },
            },
            "decode_token_ids": [512, 512, 512, 512],
            "kv_transfer_params": {},
            "tpot_ps": constant["tpot_ps"],
        },
        "artifact": {
            "aggregate_kv_bytes": aggregate,
            "chunk_bytes": [chunk] * 8,
            "messages": messages,
            "flows": flows,
            "last_required_arrival_ps": duration,
            "quiescent": True,
        },
        "artifact_sha256": {},
    }


def _observation():
    return {
        "controls": {"8": _control(8), "16": _control(16)},
        "cells": [
            _cell(8, 200_000_000_000, 4_000_000),
            _cell(8, 400_000_000_000, 3_000_000),
            _cell(16, 200_000_000_000, 8_000_000),
            _cell(16, 400_000_000_000, 6_000_000),
        ],
        "packet_artifact_request_directories": 4,
    }


def test_frozen_packet_registry_arithmetic_is_self_consistent():
    study = _study()
    frozen = json.loads(EXPECTATIONS_PATH.read_text())

    study._validate_frozen_arithmetic(frozen)


def test_analysis_passes_exact_conservation_and_behavioral_relations():
    study = _study()
    frozen = json.loads(EXPECTATIONS_PATH.read_text())

    analysis = study.analyze_observation(_observation(), frozen)

    assert analysis["status"] == "PASS"
    assert analysis["fatal_guards"] == {"status": "HELD", "findings": []}
    assert len(analysis["exact_oracle_rows"]) == 4
    assert analysis["behavioral_held"] == analysis["behavioral_total"] == 8
    assert all(row["metric_residual_ps"] == 0 for row in analysis["exact_oracle_rows"])


def test_analysis_voids_on_endpoint_or_byte_loss():
    study = _study()
    frozen = json.loads(EXPECTATIONS_PATH.read_text())
    observation = _observation()
    observation["cells"][0]["artifact"]["flows"][0]["payload_bytes"] -= 1

    analysis = study.analyze_observation(observation, frozen)

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
    assert study.RESULT_SCHEMA == "simllm-pd-session-fabric-handoff-study-result-v1"
