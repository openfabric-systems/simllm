from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path, PureWindowsPath

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples" / "deployment_curve_v1"


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
    return _module("deployment_curve_flagship_run3_tools", "flagship_run3_tools.py")


def _runner():
    return _module("deployment_curve_flagship_run3_runner", "run_flagship_run3.py")


def _plot():
    return _module("deployment_curve_flagship_run3_plot", "plot_flagship_run3.py")


def _publisher():
    return _module(
        "deployment_curve_flagship_run3_publisher",
        "publish_flagship_run3.py",
    )


def _json(name: str):
    return json.loads((STUDY_DIR / name).read_text(encoding="utf-8"))


def _synthetic_inputs():
    return (
        {
            "sglang_prefill_1k": {"value": 50_000},
            "sglang_decode_standard": {"value": 9_000},
        },
        {
            "sglang_prefill_2k": {"value": 50_000},
            "sglang_prefill_4k": {"value": 48_000},
        },
    )


def _synthetic_result():
    tools = _tools()
    runner = _runner()
    frozen = _json("scored_run3_expectations.json")
    config = _json("flagship_run2_config.json")
    calibration, held_out = _synthetic_inputs()
    fit = tools.fit_constants(calibration, frozen)
    score = tools.score_frozen_fit(held_out, frozen, fit, fit_sha256="1" * 64)
    predictions = runner._anchor_predictions(frozen, fit, score)
    decode = next(
        row["layers"]["physics_plus_boundary_plus_attenuation"]["prediction"]
        for row in predictions
        if row["anchor_id"] == "sglang_decode_standard"
    )
    decode_fit = next(
        row
        for row in fit["calibration_rows"]
        if row["anchor_id"] == "sglang_decode_standard"
    )
    decode_comparison = decode_fit["layers"][
        "physics_plus_boundary_plus_attenuation"
    ]
    return {
        "schema": "simllm-deployment-curve-flagship-run3-result-v1",
        "status": score["status"],
        "verdict": f"SCORABLE_HELD_OUT_{score['status']}_MTP_BLOCKED",
        "classification": "THIRD_SCORED_FLAGSHIP",
        "scope": score["scope"],
        "core54_closure": False,
        "closure_reason": "synthetic test",
        "provenance": {},
        "allocation": {},
        "scale_mapping": {},
        "topology": {},
        "pricing_configuration": frozen["pricing_configuration"],
        "attenuation_layer": frozen["attenuation_layer"],
        "constant_fit": fit,
        "constant_fit_sha256": "1" * 64,
        "held_out_score": score,
        "held_out_score_sha256": "2" * 64,
        "anchor_access": {"status": "PASS"},
        "anchor_predictions": predictions,
        "curves": [
            tools.build_publication_curve(curve, decode)
            for curve in config["publication_curves"]
        ],
        "offered_load_sweep_requests_per_second": frozen[
            "offered_load_sweep_requests_per_second"
        ],
        "second_legend": config["second_legend"],
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
                "numerator": 256_000_000_000_000,
                "denominator": 9_000,
            },
            "absolute_relative_error": decode_comparison[
                "absolute_relative_error"
            ],
        },
        "dominant_held_out_contributor": {"anchor_id": "sglang_prefill_4k"},
        "preservation_lock": {"status": "PASS"},
        "residuals_required": [],
    }


def test_run3_config_is_the_byte_identical_second_run_configuration():
    tools = _tools()
    frozen = _json("scored_run3_expectations.json")
    config_path = STUDY_DIR / "flagship_run2_config.json"
    config = _json(config_path.name)

    tools.validate_execution_config(config, frozen)

    assert tools.sha256(config_path) == tools.RUN2_CONFIG_SHA256
    assert config["live_session"]["project_remote_kv_length"] is True
    assert config["model"]["weights_required"] is False
    assert [
        row["prompt_tokens"] * row["requests"]
        for row in config["exact_shape_observations"][:3]
    ] == [16_384, 16_384, 16_384]


def test_run3_field_reader_reads_only_calibration_spans(tmp_path):
    tools = _tools()
    access_log = tmp_path / "access.jsonl"

    rows = tools.read_anchor_subset(
        STUDY_DIR / "expectations.json",
        ("sglang_prefill_1k", "sglang_decode_standard"),
        access_log,
        classification="calibration",
    )
    access = tools.load_access_log(access_log)

    assert set(rows) == {"sglang_prefill_1k", "sglang_decode_standard"}
    assert [row["anchor_id"] for row in access] == [
        "sglang_prefill_1k",
        "sglang_decode_standard",
    ]
    assert all(row["whole_record_loaded"] is False for row in access)
    assert sum(row["length"] for row in access) < (
        STUDY_DIR / "expectations.json"
    ).stat().st_size


def test_run3_fit_reads_only_calibration_and_stays_inside_clean_envelope():
    tools = _tools()
    frozen = _json("scored_run3_expectations.json")
    calibration, _ = _synthetic_inputs()

    fit = tools.fit_constants(calibration, frozen)
    constants = {row["id"]: row for row in fit["constants"]}

    assert fit["status"] == "FROZEN"
    assert fit["accessed_anchor_ids"] == [
        "sglang_decode_standard",
        "sglang_prefill_1k",
    ]
    assert fit["forbidden_anchor_ids_accessed"] == []
    assert constants["intra_node_collective_surcharge_ps"]["fitted"] == 0
    exposed = Fraction(**constants["overlap_exposed_fraction"]["fitted"])
    assert 0 <= exposed <= Fraction(1, 2)
    assert fit["attenuation_factor_fitted"] is False


def test_run3_prediction_layers_propagate_all_named_sources():
    tools = _tools()
    frozen = _json("scored_run3_expectations.json")
    calibration, _ = _synthetic_inputs()
    fit = tools.fit_constants(calibration, frozen)
    constants = {row["id"]: row for row in fit["constants"]}
    exposed = Fraction(**constants["overlap_exposed_fraction"]["fitted"])
    rows = {
        row["anchor_id"]: row
        for row in frozen["pre_fit_prediction_layers"]
        if row.get("status") != "BLOCKED"
    }

    layers = tools.prediction_layers(rows["sglang_prefill_4k"], frozen, exposed, 0)

    assert set(layers) == {
        "physics_only",
        "physics_plus_boundary",
        "physics_plus_boundary_plus_attenuation",
    }
    assert [
        row["source_kind"]
        for row in layers["physics_plus_boundary_plus_attenuation"]["contributions"]
    ] == [
        "comp75-clean-composition-record",
        "constant-envelope",
        "distribution",
        "overlap-exposure-envelope",
        "benchmark-bias-attenuation",
    ]
    assert Fraction(**layers["physics_plus_boundary"]["lower"]) < Fraction(
        **layers["physics_only"]["lower"]
    )


def test_run3_score_requires_frozen_fit_address_and_publishes_three_layers():
    tools = _tools()
    frozen = _json("scored_run3_expectations.json")
    calibration, held_out = _synthetic_inputs()
    fit = tools.fit_constants(calibration, frozen)

    with pytest.raises(ValueError, match="serialized fit SHA-256"):
        tools.score_frozen_fit(held_out, frozen, fit, fit_sha256="not-addressed")

    score = tools.score_frozen_fit(held_out, frozen, fit, fit_sha256="1" * 64)

    assert score["status"] == "PASS"
    assert score["unattenuated_status"] == "REFUTED"
    assert score["accessed_anchor_ids"] == ["sglang_prefill_2k", "sglang_prefill_4k"]
    assert score["forbidden_anchor_ids_accessed"] == []
    assert all(len(row["layers"]) == 3 for row in score["rows"])
    assert score["blocked_rows"][0]["numeric_anchor_read"] is False
    assert score["blocked_rows"][0]["published"] is None


def test_run3_access_summary_requires_calibration_then_held_out():
    tools = _tools()
    rows = [
        {
            "classification": classification,
            "anchor_id": anchor_id,
            "whole_record_loaded": False,
        }
        for classification, anchor_id in (
            ("calibration", "sglang_prefill_1k"),
            ("calibration", "sglang_decode_standard"),
            ("held_out", "sglang_prefill_2k"),
            ("held_out", "sglang_prefill_4k"),
        )
    ]

    summary = tools.access_summary(rows)

    assert summary["status"] == "PASS"
    assert summary["calibration_access_count"] == 2
    assert summary["held_out_access_count"] == 2
    assert summary["mtp_numeric_access_count"] == 0
    with pytest.raises(ValueError, match="sequence disagrees"):
        tools.access_summary(list(reversed(rows)))


def test_run3_plot_preparation_carries_layers_scope_and_unattenuated_verdict():
    prepared = _plot().prepare_flagship_plot(_synthetic_result())

    assert len(prepared["curves"]) == 2
    assert len(prepared["prefill"]) == 3
    assert all(len(row["layers"]) == 3 for row in prepared["prefill"])
    assert prepared["verdict"] == "PASS"
    assert prepared["unattenuated_verdict"] == "REFUTED"
    assert "benchmark-bias model" in prepared["scope"]
    assert prepared["mtp_dependency"] == "COMP-72 resumable Merlin execution"


def test_run3_renderer_writes_pdf_and_png_without_a_watermark(tmp_path):
    pytest.importorskip("matplotlib")
    plot = _plot()
    prepared = plot.prepare_flagship_plot(_synthetic_result())

    pdf, png = plot.render_flagship_figure(prepared, tmp_path / "flagship-run3")

    assert pdf.read_bytes().startswith(b"%PDF")
    assert png.read_bytes().startswith(b"\x89PNG")
    assert b"watermark" not in pdf.read_bytes().lower()


def test_run3_publication_projection_omits_bulk_requests():
    publisher = _publisher()
    result = _synthetic_result()

    publication = publisher.build_publication_result(result, {})

    assert publication["held_out_score"] == result["held_out_score"]
    assert publication["attenuation_layer"] == result["attenuation_layer"]
    assert publication["anchor_access"] == result["anchor_access"]
    assert "stable_requests" not in publication["scored_session_summary"][0]


def test_run3_check_only_loads_no_anchor_frontend_or_unix_only_module(monkeypatch):
    original_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "resource" or name.startswith(("sglang", "matplotlib")):
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    runner = _runner()
    args = type("Args", (), {"config": STUDY_DIR / "flagship_run2_config.json"})()

    frozen, config, preservation = runner.check_registry(args, require_runtime=False)

    assert frozen["schema"] == "simllm-deployment-curve-scored-run3-expectations-v1"
    assert config["schema"] == "simllm-deployment-curve-flagship-run2-config-v1"
    assert len(preservation) >= 30
    assert runner.render_cli_path(PureWindowsPath("C:/runs/result.json")) == (
        "C:/runs/result.json"
    )
