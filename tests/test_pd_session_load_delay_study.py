from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from pathlib import Path, PureWindowsPath

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples" / "pd_session_load_delay_v1"
sys.path.insert(0, str(STUDY_DIR))


def _study():
    spec = importlib.util.spec_from_file_location(
        "pd_session_load_delay_study",
        STUDY_DIR / "run_study.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fraction(value: int) -> dict[str, int]:
    return {"numerator": value, "denominator": 1}


def _cell(prefill: int, decode: int, prompt: int, load: int, delay: int) -> dict:
    requests = []
    for index in range(64):
        request_id = f"p{prefill}-d{decode}-prompt{prompt}-load{load}-r{index}"
        admitted = index * (1_000_000_000_000 // load)
        requests.append(
            {
                "request_id": request_id,
                "expected_admitted_at_ps": admitted,
                "prefill_internal_request_id": f"prefill-{request_id}",
                "decode_internal_request_id": f"decode-{request_id}",
                "decode_token_ids": [1, 2, 3, 4],
                "timeline": {
                    "admitted_at_ps": admitted,
                    "ttft_ps": 10,
                    "decomposition": {
                        "prefill_queue_ps": delay,
                        "prefill_service_ps": 1,
                        "handoff_ps": 1,
                        "decode_admission_wait_ps": 0,
                        "decode_first_token_service_ps": 8,
                        "total_ps": 10,
                    },
                },
            }
        )
    return {
        "prefill_engines": prefill,
        "decode_engines": decode,
        "prompt_tokens": prompt,
        "offered_load_requests_per_second": load,
        "requests": requests,
        "compute_pricing": {
            "prefill": None,
            "decode": {
                "record_sha256": "ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52",
                "acceptance_status": "candidate",
                "calibration_claim": False,
            },
        },
        "amortized_decode_batch_service_per_token_ps": _fraction(1),
        "curve_point": {
            "request_count": 64,
            "output_token_count": 256,
            "per_token_request_delay_ps": _fraction(delay),
        },
    }


def _observation(study, freeze: dict) -> dict:
    expected = {
        (
            *row["configuration"],
            row["offered_load_requests_per_second"],
        ): row["predicted_per_token_request_delay_ps"]["numerator"]
        // row["predicted_per_token_request_delay_ps"]["denominator"]
        for row in freeze["held_out_prediction_bands"]
    }
    cells = []
    for prefill, decode in study.POOL_RATIOS:
        for prompt in study.PROMPT_LENGTHS:
            values = [300, 200, 100, 400, 500, 600]
            for load, default in zip(study.OFFERED_LOADS, values, strict=True):
                delay = expected.get((prefill, decode, prompt, load), default)
                cells.append(_cell(prefill, decode, prompt, load, delay))
    return {
        "core51_control": {
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


def test_study_pins_the_expectations_only_commit_and_bytes() -> None:
    study = _study()

    assert study.FREEZE_COMMIT == "121345e950b12a36018404084c7dcf9bd507f962"
    assert study.EXPECTATIONS_SHA256 == (
        "28cee81deffe771836b5c38d7fe605185f4dc31a953087c80288ceb7a3a84e22"
    )


def test_analysis_conserves_requests_and_scores_all_frozen_rows() -> None:
    study = _study()
    freeze = json.loads((STUDY_DIR / "expectations.json").read_text())

    analysis = study.analyze_observation(_observation(study, freeze), freeze)

    assert analysis["fatal_guards"] == {"status": "HELD", "findings": []}
    assert analysis["conservation"] == {
        "cells": 36,
        "admissions": 2_304,
        "handoffs": 2_304,
        "terminals": 2_304,
        "terminal_decode_tokens": 9_216,
        "maximum_ttft_residual_ps": 0,
    }
    assert len(analysis["segment_verdicts"]) == 30
    assert len(analysis["held_out_band_verdicts"]) == 24
    assert analysis["held_out_band_summary"] == {"held": 24, "evaluated": 24}
    assert analysis["monotonic_delay_claim"] == "WITHDRAWN"


def test_analysis_publishes_an_honest_held_out_refutation() -> None:
    study = _study()
    freeze = json.loads((STUDY_DIR / "expectations.json").read_text())
    observation = _observation(study, freeze)
    target = next(
        cell
        for cell in observation["cells"]
        if (cell["prefill_engines"], cell["decode_engines"], cell["prompt_tokens"])
        == (2, 1, 16)
        and cell["offered_load_requests_per_second"] == 8_000
    )
    target["curve_point"]["per_token_request_delay_ps"] = _fraction(10**15)

    analysis = study.analyze_observation(observation, freeze)

    assert analysis["status"] == "REFUTED"
    assert analysis["held_out_band_summary"] == {"held": 23, "evaluated": 24}


def test_posix_rendering_and_module_scope_are_portable(monkeypatch) -> None:
    study = _study()
    assert study.render_cli_path(PureWindowsPath("C:/run/result")) == "C:/run/result"

    original_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "resource" or name == "vllm" or name.startswith("vllm."):
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    assert _study().RESULT_SCHEMA == "simllm-pd-session-load-delay-result-v1"
