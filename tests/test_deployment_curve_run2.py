from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path, PureWindowsPath

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples/deployment_curve_v1"


def _module(name: str, filename: str):
    sys.path.insert(0, str(STUDY_DIR))
    try:
        spec = importlib.util.spec_from_file_location(name, STUDY_DIR / filename)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(STUDY_DIR))


def _tools():
    return _module("deployment_curve_flagship_run2_tools", "flagship_run2_tools.py")


def _runner():
    return _module("deployment_curve_flagship_run2_runner", "run_flagship_run2.py")


def _plot():
    return _module("deployment_curve_flagship_run2_plot", "plot_flagship_run2.py")


def _publisher():
    return _module(
        "deployment_curve_flagship_run2_publisher",
        "publish_flagship_run2.py",
    )


def _json(name: str):
    return json.loads((STUDY_DIR / name).read_text(encoding="utf-8"))


def _synthetic_anchors() -> dict[str, object]:
    return {
        "schema": "synthetic-anchor-freeze",
        "anchors": [
            {"id": "sglang_prefill_1k", "value": 57_000},
            {"id": "sglang_decode_standard", "value": 9_000},
            {"id": "sglang_prefill_2k", "value": 56_000},
            {"id": "sglang_prefill_4k", "value": 50_000},
            {"id": "sglang_decode_simulated_mtp", "value": 1},
        ],
    }


def _synthetic_result() -> tuple[dict[str, object], dict[str, object]]:
    tools = _tools()
    frozen = _json("scored_run2_expectations.json")
    config = _json("flagship_run2_config.json")
    anchors = _synthetic_anchors()
    fit = tools.fit_inherited_constant(anchors, frozen)
    score = tools.score_frozen_fit(anchors, frozen, fit, fit_sha256="1" * 64)
    predictions = []
    for row in frozen["pre_fit_predicted_bands"]:
        if row.get("status") == "BLOCKED":
            predictions.append(
                {
                    "anchor_id": row["anchor_id"],
                    "status": "BLOCKED",
                    "prediction": None,
                    "dependency": row["dependency"],
                }
            )
        else:
            predictions.append(
                {
                    "anchor_id": row["anchor_id"],
                    "status": "PREDICTED",
                    "prediction": tools.prediction_interval(row, frozen, 0),
                }
            )
    decode_capacity = next(
        row["prediction"]
        for row in predictions
        if row["anchor_id"] == "sglang_decode_standard"
    )
    decode_fit = next(
        row
        for row in fit["calibration_rows"]
        if row["anchor_id"] == "sglang_decode_standard"
    )
    result = {
        "schema": "simllm-deployment-curve-flagship-run2-result-v1",
        "status": score["status"],
        "verdict": f"SCORABLE_HELD_OUT_{score['status']}_MTP_BLOCKED",
        "scope": score["scope"],
        "core54_closure": False,
        "closure_reason": "synthetic test",
        "provenance": {},
        "allocation": {},
        "scale_mapping": {},
        "topology": {},
        "pricing_configuration": frozen["pricing_configuration"],
        "constant_fit": fit,
        "constant_fit_sha256": "1" * 64,
        "held_out_score": score,
        "held_out_score_sha256": "2" * 64,
        "anchor_predictions": predictions,
        "curves": [
            tools.build_publication_curve(curve, decode_capacity)
            for curve in config["publication_curves"]
        ],
        "offered_load_sweep_requests_per_second": frozen[
            "offered_load_sweep_requests_per_second"
        ],
        "stable_identity_guard": {"equal": True},
        "packet_observation": {},
        "session_observations": [
            {
                "anchor_id": "sglang_decode_standard",
                "pool": "decode",
                "candidate_entry_index": 3,
                "admissions": 32,
                "terminals": 32,
                "prompt_tokens_per_request": 2000,
                "total_prompt_tokens": 64_000,
                "remote_kv_projection_enabled": True,
                "stable_projection_sha256": "3" * 64,
                "stable_requests": [{"bulk": True}],
                "batches": {"prefill": [["request"]], "decode": [["request"]]},
                "prefill_ranks": list(range(8)),
                "decode_ranks": list(range(32, 40)),
            }
        ],
        "candidate_selections": [],
        "decode_calibration_miss": {
            "declared_step_ps": 28_604_120_000,
            "published_throughput_implied_step_ps": {
                "numerator": 128_000_000_000_000,
                "denominator": 11_141,
            },
            "absolute_relative_error": decode_fit["absolute_relative_error"],
        },
        "dominant_held_out_contributor": {"anchor_id": "sglang_prefill_4k"},
        "preservation_lock": {"status": "PASS"},
        "residuals_required": [],
    }
    plot_anchors = {
        "anchors": [
            {
                "id": "sglang_prefill_1k",
                "value": 57_000,
                "role": "calibration",
            },
            {
                "id": "sglang_prefill_2k",
                "value": 56_000,
                "role": "held-out",
            },
            {
                "id": "sglang_prefill_4k",
                "value": 50_000,
                "role": "held-out",
            },
            {
                "id": "sglang_decode_standard",
                "aggregated_value": 200_538,
            },
            {
                "id": "sglang_decode_simulated_mtp",
                "aggregated_value": 288_240,
            },
            {
                "id": "deepseek_production_decode",
                "aggregated_value": 266_400,
            },
        ]
    }
    return result, plot_anchors


def test_run2_config_enables_exact_remote_kv_binding_and_keeps_separate_roles():
    tools = _tools()
    frozen = _json("scored_run2_expectations.json")
    config = _json("flagship_run2_config.json")

    tools.validate_config(config, frozen)

    assert config["live_session"]["project_remote_kv_length"] is True
    assert config["model"]["weights_required"] is False
    assert [
        row["prompt_tokens"] * row["requests"]
        for row in config["exact_shape_observations"][:3]
    ] == [16_384, 16_384, 16_384]
    allocation = frozen["inherited_rulings"]["allocation"]
    assert allocation["prefill_experiment"]["ranks"] == 32
    assert allocation["decode_experiment"]["ranks"] == 72
    assert allocation["structural_comparator_only"]["ranks"] == 104
    assert allocation["structural_comparator_only"]["may_be_called_96_gpu_system"] is False


def test_run2_prediction_propagates_record_constant_and_distribution_entries():
    tools = _tools()
    frozen = _json("scored_run2_expectations.json")
    rows = {row["anchor_id"]: row for row in frozen["pre_fit_predicted_bands"]}

    interval = tools.prediction_interval(rows["sglang_prefill_4k"], frozen, 0)

    assert interval["lower"] == rows["sglang_prefill_4k"]["lower"]
    assert interval["point"] == rows["sglang_prefill_4k"]["point"]
    assert interval["upper"] == rows["sglang_prefill_4k"]["upper"]
    assert [row["source_kind"] for row in interval["contributions"]] == [
        "comp75-clean-composition-record",
        "constant-envelope",
        "distribution",
    ]
    assert interval["contributions"][1]["application_count"] == 0
    assert interval["contributions"][2] == {
        "source_kind": "distribution",
        "source_id": "comp74-zero-width-insufficient-replays",
        "relative_half_width": 0,
        "stability_claim": False,
    }


def test_run2_fit_reads_only_synthetic_calibration_values_and_selects_tie_floor():
    tools = _tools()
    frozen = _json("scored_run2_expectations.json")

    fit = tools.fit_inherited_constant(_synthetic_anchors(), frozen)

    assert fit["status"] == "FROZEN"
    assert fit["fitted_ps"] == 0
    assert fit["application_count_per_step"] == 0
    assert fit["tie_break_applied"] is True
    assert fit["accessed_anchor_ids"] == [
        "sglang_decode_standard",
        "sglang_prefill_1k",
    ]
    assert fit["forbidden_anchor_ids_accessed"] == []


def test_run2_score_requires_serialized_fit_address_before_synthetic_held_out_read():
    tools = _tools()
    frozen = _json("scored_run2_expectations.json")
    anchors = _synthetic_anchors()
    fit = tools.fit_inherited_constant(anchors, frozen)

    with pytest.raises(ValueError, match="serialized fit SHA-256"):
        tools.score_frozen_fit(anchors, frozen, fit, fit_sha256="not-addressed")

    score = tools.score_frozen_fit(anchors, frozen, fit, fit_sha256="1" * 64)

    assert score["accessed_anchor_ids"] == [
        "sglang_prefill_2k",
        "sglang_prefill_4k",
    ]
    assert score["forbidden_anchor_ids_accessed"] == []
    assert score["fit_sha256"] == "1" * 64
    assert score["blocked_rows"] == [
        {
            "anchor_id": "sglang_decode_simulated_mtp",
            "status": "BLOCKED",
            "published": None,
            "prediction": None,
            "reason": "candidate record has no EP72 MTP batch-16 KV-4000 cell",
            "dependency": "COMP-72 resumable Merlin execution",
        }
    ]


def test_run2_pre_score_guard_requires_the_exact_decode_key():
    runner = _runner()
    frozen = _json("scored_run2_expectations.json")
    expected_key = frozen["pricing_configuration"]["decode"][
        "remote_kv_projection"
    ]["exact_candidate_key_sha256"]

    def observation(anchor_id: str, pool: str, key: str) -> dict[str, object]:
        return {
            "anchor_id": anchor_id,
            "pool": pool,
            "pricing_provenance": {
                pool: {
                    "record_sha256": "2" * 64,
                    "acceptance_status": "candidate",
                    "lookup_hits": 1,
                    "lookup_misses": 0,
                    "selected_entry_key_sha256s": [key],
                }
            },
            "stable_requests": [{"request_id": anchor_id}],
        }

    observations = [
        observation("sglang_prefill_1k", "prefill", "3" * 64),
        observation("sglang_prefill_2k", "prefill", "4" * 64),
        observation("sglang_prefill_4k", "prefill", "5" * 64),
        observation("sglang_decode_standard", "decode", expected_key),
    ]
    packet = {
        "byte_conserved": True,
        "endpoint_conserved": True,
        "rows": [{"quiescent": True}, {"quiescent": True}],
    }

    _, violations = runner._pre_score_guards(
        frozen,
        observations,
        {"stable_requests": observations[-1]["stable_requests"]},
        packet,
    )
    assert violations == []

    observations[-1]["pricing_provenance"]["decode"][
        "selected_entry_key_sha256s"
    ] = ["6" * 64]
    _, violations = runner._pre_score_guards(
        frozen,
        observations,
        {"stable_requests": observations[-1]["stable_requests"]},
        packet,
    )
    assert violations == ["the exact SGL-38 decode key did not bind"]


def test_run2_publication_curve_keeps_ordered_axes_and_exact_load_grid():
    tools = _tools()
    frozen = _json("scored_run2_expectations.json")
    config = _json("flagship_run2_config.json")
    decode = next(
        row
        for row in frozen["pre_fit_predicted_bands"]
        if row["anchor_id"] == "sglang_decode_standard"
    )
    interval = tools.prediction_interval(decode, frozen, 0)

    curve = tools.build_publication_curve(config["publication_curves"][0], interval)

    assert curve["orientation"] == {
        "x": "aggregated-output-throughput-rightward",
        "y": "inverse-per-token-request-delay-upward",
    }
    assert len(curve["points"]) == 5
    assert [
        Fraction(**point["offered_load_requests_per_second"])
        for point in curve["points"]
    ] == [Fraction(value) for value in (2_000, 4_000, 8_000, 16_000, 32_000)]


def test_run2_plot_preparation_carries_scope_mtp_and_decode_miss():
    plot = _plot()
    result, anchors = _synthetic_result()

    prepared = plot.prepare_flagship_plot(result, anchors)

    assert len(prepared["curves"]) == 2
    assert len(prepared["prefill"]) == 3
    assert prepared["scope"] == "priced held-out prefill anchors only"
    assert prepared["mtp_dependency"] == "COMP-72 resumable Merlin execution"
    assert prepared["decode_declared_step_ms"] == pytest.approx(28.60412)
    assert prepared["decode_implied_step_ms"] == pytest.approx(11.4890943362)


def test_run2_renderer_writes_pdf_and_png_without_a_watermark(tmp_path):
    pytest.importorskip("matplotlib")
    plot = _plot()
    result, anchors = _synthetic_result()
    prepared = plot.prepare_flagship_plot(result, anchors)

    pdf, png = plot.render_flagship_figure(prepared, tmp_path / "flagship-run2")

    assert pdf.read_bytes().startswith(b"%PDF")
    assert png.read_bytes().startswith(b"\x89PNG")


def test_run2_publication_projection_omits_bulk_requests():
    publisher = _publisher()
    result, _ = _synthetic_result()

    publication = publisher.build_publication_result(result, {})

    assert publication["held_out_score"] == result["held_out_score"]
    assert publication["decode_calibration_miss"] == result[
        "decode_calibration_miss"
    ]
    assert "stable_requests" not in publication["scored_session_summary"][0]


def test_run2_check_only_needs_no_frontend_or_unix_only_import(monkeypatch):
    original_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "resource" or name.startswith(("sglang", "matplotlib")):
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    runner = _runner()
    args = type(
        "Args",
        (),
        {"config": STUDY_DIR / "flagship_run2_config.json"},
    )()

    _, frozen, config, preservation = runner.check_registry(
        args,
        require_runtime=False,
    )

    assert frozen["schema"] == "simllm-deployment-curve-scored-run2-expectations-v1"
    assert config["schema"] == "simllm-deployment-curve-flagship-run2-config-v1"
    assert len(preservation) == 24
    assert runner.render_cli_path(PureWindowsPath("C:/runs/result.json")) == (
        "C:/runs/result.json"
    )
