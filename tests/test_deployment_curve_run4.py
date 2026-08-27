from __future__ import annotations

import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path, PureWindowsPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/deployment_curve_v1"


def _module(filename: str, name: str):
    sys.path.insert(0, str(STUDY))
    try:
        spec = importlib.util.spec_from_file_location(name, STUDY / filename)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _json(filename: str) -> dict[str, object]:
    return json.loads((STUDY / filename).read_text(encoding="utf-8"))


def _run3_fixture() -> dict[str, object]:
    layers = {
        "physics_plus_boundary_plus_attenuation": {
            "point_passes_5_percent": True
        }
    }
    return {
        "schema": "simllm-deployment-curve-flagship-run3-publication-v1",
        "status": "PASS",
        "verdict": "SCORABLE_HELD_OUT_PASS_MTP_BLOCKED",
        "held_out_score": {
            "rows": [
                {"anchor_id": "sglang_prefill_2k", "layers": layers},
                {"anchor_id": "sglang_prefill_4k", "layers": layers},
            ]
        },
        "anchor_predictions": [],
        "curves": [],
        "second_legend": {},
        "decode_calibration_miss": {},
    }


def test_run4_config_realizes_exact_per_gpu_shape_and_emission() -> None:
    tools = _module("flagship_run4_tools.py", "flagship_run4_tools_test")
    frozen = _json("scored_run4_expectations.json")
    config = _json("flagship_run4_config.json")

    observation = tools.build_shape_observation(config, frozen)

    assert observation["status"] == "PASS"
    assert observation["request_count"] == 128
    assert observation["requests_per_gpu"] == [16] * 8
    assert observation["total_emitted_tokens"] == 256
    assert {row["kv_tokens"] for row in observation["requests"]} == {4000}
    assert {row["emitted_tokens"] for row in observation["requests"]} == {2}
    assert observation["weights_loaded"] is False


def test_run4_scores_all_three_layers_once_without_attenuation() -> None:
    tools = _module("flagship_run4_tools.py", "flagship_run4_tools_score_test")
    frozen = _json("scored_run4_expectations.json")
    anchor = {
        "id": "sglang_decode_simulated_mtp",
        "role": "held-out",
        "value": 10_000,
    }

    score = tools.score_mtp_anchor(anchor, frozen, prediction_sha256="1" * 64)

    assert score["score_attempt_count"] == 1
    assert score["attenuation_applied"] is False
    assert len(score["layers"]) == 3
    comparisons = list(score["layers"].values())
    assert comparisons[0] == comparisons[1] == comparisons[2]
    expected = Fraction(1_024_000_000_000, 124_071_011) / 10_000 - 1
    assert Fraction(**comparisons[0]["signed_relative_error"]) == expected


def test_run4_result_carries_run3_rows_without_rescoring(monkeypatch) -> None:
    tools = _module("flagship_run4_tools.py", "flagship_run4_tools_result_test")
    frozen = _json("scored_run4_expectations.json")
    config = _json("flagship_run4_config.json")
    run3 = _run3_fixture()
    shape = tools.build_shape_observation(config, frozen)
    score = tools.score_mtp_anchor(
        {"id": "sglang_decode_simulated_mtp", "role": "held-out", "value": 10_000},
        frozen,
        prediction_sha256="1" * 64,
    )
    monkeypatch.setattr(tools, "sha256", lambda path: frozen["inherited_run3_rows"]["authority_sha256"])

    result = tools.build_result(frozen, config, run3, shape, score, [], [])

    assert result["status"] == "REFUTED"
    assert result["combined_held_out_rows"][:2] == run3["held_out_score"]["rows"]
    assert result["combined_held_out_rows"][2] is score
    assert result["run3_carry_forward"]["status"] == "BYTE_IDENTICAL_NOT_RESCORED"
    assert result["deployment_frontier"]["status"] == "UNCHANGED_FROZEN_CONTRACT"


def test_run4_check_only_needs_no_anchor_or_unix_only_import() -> None:
    runner = _module("run_flagship_run4.py", "run_flagship_run4_test")

    frozen, config, preservation = runner.check_registry()

    assert frozen["status"] == "EXPECTATIONS_ONLY"
    assert config["model"]["weights_required"] is False
    assert len(preservation) == 57
    assert runner.render_cli_path(PureWindowsPath("C:/runs/result.json")) == (
        "C:/runs/result.json"
    )


def test_run4_score_refuses_unaddressed_prediction() -> None:
    tools = _module("flagship_run4_tools.py", "flagship_run4_tools_refusal_test")
    frozen = _json("scored_run4_expectations.json")

    with pytest.raises(ValueError, match="addressed prediction"):
        tools.score_mtp_anchor(
            {"id": "sglang_decode_simulated_mtp", "role": "held-out", "value": 1},
            frozen,
            prediction_sha256="short",
        )
