from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from pathlib import Path, PureWindowsPath

from simllm.calibration.batch_service_surface import BatchServicePoint

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples" / "pd_session_queue_onset_v1"


def _module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, STUDY_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(STUDY_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(STUDY_DIR))
    return module


def _study():
    return _module("run_study.py", "pd_session_queue_onset_study")


def _model():
    return _module("queue_model.py", "pd_session_queue_onset_model")


def _freeze() -> dict:
    return json.loads((STUDY_DIR / "expectations.json").read_text(encoding="utf-8"))


def _points() -> tuple[BatchServicePoint, ...]:
    return tuple(
        BatchServicePoint(
            row["batch_size"],
            row["measured_service_ps"],
            row["trimmed_coefficient_of_variation_ppm"] / 1_000_000,
            row["entry_key_sha256"],
            row["evidence_class"],
            row["split"],
        )
        for row in _freeze()["surface"]["selected_keys"]
    )


def _cell(study, model, prefill: int, decode: int, prompt: int, load: int) -> dict:
    point = model.predict_point(
        _points(),
        prefill_engines=prefill,
        decode_engines=decode,
        prompt_tokens=prompt,
        offered_load=load,
    )
    wait = model.fraction_from_json(point["predicted_mean_scheduler_queue_wait_ps"])
    total_wait = wait * study.REQUESTS_PER_CELL
    assert total_wait.denominator == 1
    base_wait, remainder = divmod(total_wait.numerator, study.REQUESTS_PER_CELL)
    requests = []
    for index in range(study.REQUESTS_PER_CELL):
        request_id = f"p{prefill}-d{decode}-prompt{prompt}-load{load}-r{index}"
        admitted = index * (study.PS_PER_SECOND // load)
        prefill_wait = base_wait + (index < remainder)
        requests.append(
            {
                "request_id": request_id,
                "expected_admitted_at_ps": admitted,
                "prefill_internal_request_id": f"prefill-{request_id}",
                "decode_internal_request_id": f"decode-{request_id}",
                "decode_token_ids": [1, 2, 3, 4],
                "compute_pricing": {
                    "prefill": None,
                    "decode": {
                        "record_sha256": (
                            "ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52"
                        ),
                        "acceptance_status": "candidate",
                        "calibration_claim": False,
                    },
                },
                "timeline": {
                    "admitted_at_ps": admitted,
                    "ttft_ps": prefill_wait,
                    "decomposition": {
                        "prefill_queue_ps": prefill_wait,
                        "decode_admission_wait_ps": 0,
                        "total_ps": prefill_wait,
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
        "amortized_decode_batch_service_per_token_ps": point[
            "predicted_batch_service_per_token_ps"
        ],
        "maximum_prefill_batch_size": 1,
        "maximum_decode_batch_size": point["predicted_maximum_decode_batch_size"],
    }


def _observation(study, model) -> dict:
    return {
        "cells": [
            _cell(study, model, prefill, decode, prompt, load)
            for prefill, decode in study.POOL_RATIOS
            for prompt in study.PROMPT_LENGTHS
            for load in study.OFFERED_LOADS
        ],
        "total_delay_curves": None,
        "total_delay_direction_scored": False,
    }


def test_runner_pins_the_committed_expectations_only_freeze() -> None:
    study = _study()

    assert study.FREEZE_COMMIT == "b3e225e6a4b97280c86536bef136e9945cc239fb"
    assert study.EXPECTATIONS_SHA256 == (
        "859efc475534bd461761a0e34a039594bd52877520a232a91f2f9c4309c73308"
    )
    assert study.QUEUE_MODEL_SHA256 == (
        "d3b63d1a50e3615c7c65d0396a6dc038bbdcab569d43c5e8620babb9fbbce1e3"
    )


def test_exact_model_observation_identifies_onset_without_delay_rescore() -> None:
    study = _study()
    model = _model()

    analysis = study.analyze_observation(
        _observation(study, model),
        _freeze(),
    )

    assert analysis["status"] == "IDENTIFIED"
    assert analysis["fatal_guards"] == {"status": "HELD", "findings": []}
    assert analysis["conservation"] == {
        "admissions": 4_992,
        "cells": 78,
        "handoffs": 4_992,
        "maximum_ttft_residual_ps": 0,
        "terminal_decode_tokens": 19_968,
        "terminals": 4_992,
    }
    assert len(analysis["decomposition_rows"]) == 78
    assert len(analysis["segment_decompositions"]) == 72
    assert all(row["prediction_held"] for row in analysis["segment_decompositions"])
    assert analysis["onset_summary"] == {
        "configurations_inside_prediction_band": 6,
        "configurations_resolved": 6,
        "distinct_observed_segments": [[225, 230]],
        "predicted_central_segment": [225, 230],
        "predicted_inclusive_segments": [[220, 225], [225, 230]],
    }
    assert analysis["held_out_band_summary"] == {
        "batch_service_held": 30,
        "evaluated": 30,
        "joint_held": 30,
        "queue_wait_held": 30,
    }
    assert analysis["total_delay_direction_scored"] is False
    assert analysis["prior_250_to_8000_monotonic_direction"] == (
        "PRESERVED_NOT_REOPENED"
    )
    assert analysis["closure"] == {
        "VLLM-41": "CLOSED",
        "VLLM-42": "UNUSED_RESERVED",
        "VLLM-43": "UNUSED_RESERVED",
    }


def test_held_out_component_miss_registers_vllm42_without_widening() -> None:
    study = _study()
    model = _model()
    observation = _observation(study, model)
    target = next(
        cell
        for cell in observation["cells"]
        if (
            cell["prefill_engines"],
            cell["decode_engines"],
            cell["prompt_tokens"],
            cell["offered_load_requests_per_second"],
        )
        == (2, 1, 16, 240)
    )
    target["amortized_decode_batch_service_per_token_ps"] = {
        "numerator": 10**15,
        "denominator": 1,
    }

    analysis = study.analyze_observation(observation, _freeze())

    assert analysis["held_out_band_summary"]["joint_held"] == 29
    assert analysis["closure"]["VLLM-42"] == "REGISTER_RESIDUAL"


def test_runner_is_portable_and_has_no_module_scope_vllm_import(monkeypatch) -> None:
    study = _study()
    assert study.render_cli_path(PureWindowsPath("C:/run/result")) == "C:/run/result"

    original_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "resource" or name == "vllm" or name.startswith("vllm."):
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    reloaded = _study()
    assert reloaded.RESULT_SCHEMA == "simllm-pd-session-queue-onset-result-v1"
