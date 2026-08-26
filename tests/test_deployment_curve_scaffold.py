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
    return _module("deployment_curve_tools", "curve_tools.py")


def _study():
    return _module("deployment_curve_study", "run_study.py")


def _plot():
    return _module("deployment_curve_plot", "plot_results.py")


def _freeze():
    return json.loads((STUDY_DIR / "expectations.json").read_text())


def _config():
    return json.loads((STUDY_DIR / "dry_run_config.json").read_text())


def _fraction(value):
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    return {"numerator": value, "denominator": 1}


def _declaration(constant_id, selected, lower, upper):
    return {
        "id": constant_id,
        "tunable": True,
        "unit": "ps",
        "selected": selected,
        "envelope": {"lower": lower, "upper": upper},
        "provenance": {
            "source": "synthetic-test",
            "locator": "fixture",
            "physical_basis": "bounded synthetic arithmetic",
        },
    }


def test_anchor_freeze_transcribes_numeric_dossier_and_split():
    tools = _tools()
    freeze = _freeze()

    tools.validate_anchor_freeze(freeze)

    anchors = {anchor["id"]: anchor for anchor in freeze["anchors"]}
    assert {
        anchor_id: anchors[anchor_id]["value"]
        for anchor_id in (
            "sglang_prefill_1k",
            "sglang_prefill_2k",
            "sglang_prefill_4k",
            "sglang_decode_standard",
            "sglang_decode_simulated_mtp",
        )
    } == {
        "sglang_prefill_1k": 57_674,
        "sglang_prefill_2k": 54_543,
        "sglang_prefill_4k": 50_302,
        "sglang_decode_standard": 22_282,
        "sglang_decode_simulated_mtp": 17_373,
    }
    assert anchors["sglang_ttft_range"]["lower"] == 2
    assert anchors["sglang_ttft_range"]["upper"] == 5
    assert anchors["sglang_inter_token_latency"]["value"] == 100
    assert anchors["sglang_output_cost"]["value"] == 0.20
    assert anchors["sglang_prefill_delta"]["value"] == 5.6
    assert anchors["sglang_mtp_decode_delta"]["value"] == 6.6
    assert anchors["deepseek_production_prefill"]["value"] == 73_700
    assert anchors["deepseek_production_decode"]["value"] == 14_800
    assert all(anchor["source_id"] and anchor["source_locator"] for anchor in anchors.values())

    split = freeze["calibration_split"]
    assert split["calibration_anchor_ids"] == [
        "sglang_prefill_1k",
        "sglang_decode_standard",
    ]
    assert split["held_out_anchor_ids"] == [
        "sglang_prefill_2k",
        "sglang_prefill_4k",
        "sglang_decode_simulated_mtp",
    ]
    assert freeze["acceptance"]["maximum_absolute_relative_error"] == 0.05
    assert freeze["acceptance"]["pricing_dependent_bands_frozen_here"] is False


def test_anchor_freeze_declares_axis_and_deployment_shapes():
    freeze = _freeze()

    assert freeze["axis_contract"] == {
        "x": {
            "quantity": "aggregated_output_throughput",
            "unit": "tokens_per_second",
            "direction": "rightward",
            "definition": "all terminal output tokens divided by last terminal completion minus first admission",
        },
        "y": {
            "quantity": "inverse_per_token_request_delay",
            "unit": "tokens_per_second",
            "direction": "upward",
            "definition": "one divided by the arithmetic mean across requests of terminal completion minus admission divided by that request's output token count",
            "record_projection": "1000000000000 divided by per_token_request_delay_ps",
        },
        "optimal_corner": "upper-right",
        "sweep_parameter": "offered_load_requests_per_second",
        "one_curve_per_configuration": True,
    }
    sglang = freeze["disclosed_configurations"]["sglang_reproduction"]
    assert (sglang["cluster_nodes"], sglang["gpus_per_node"], sglang["cluster_gpus"]) == (12, 8, 96)
    assert (sglang["prefill"]["nodes"], sglang["prefill"]["expert_parallelism"]) == (4, 32)
    assert (sglang["decode"]["nodes"], sglang["decode"]["expert_parallelism"]) == (9, 72)
    production = freeze["disclosed_configurations"]["deepseek_production"]
    assert production["prefill"]["nodes"] == 4
    assert production["decode"]["nodes"] == 18
    what_if = freeze["disclosed_configurations"]["declared_what_if"]
    assert (what_if["prefill_nodes"], what_if["decode_nodes"], what_if["total_gpus"]) == (
        16,
        40,
        448,
    )


def test_constant_declarations_refuse_selected_value_outside_envelope():
    tools = _tools()
    declaration = _declaration("c", 11, 0, 10)

    with pytest.raises(ValueError, match="outside"):
        tools.validate_constant_declarations([declaration])


def test_fit_reads_only_calibration_anchors_and_discloses_parameters():
    tools = _tools()
    declarations = [
        _declaration("collective", 500, 0, 2_000),
        _declaration("pcie", 100, 0, 1_000),
        _declaration("handoff", 500, 0, 2_000),
    ]
    rows = [
        {
            "constant_id": "collective",
            "anchor_id": "sglang_prefill_1k",
            "baseline": 56_674,
            "sensitivity_per_ps": 1,
        },
        {
            "constant_id": "pcie",
            "anchor_id": "sglang_decode_standard",
            "baseline": 22_082,
            "sensitivity_per_ps": 1,
        },
        {
            "constant_id": "handoff",
            "anchor_id": "sglang_decode_standard",
            "baseline": 21_282,
            "sensitivity_per_ps": 1,
        },
    ]

    result = tools.fit_tunable_constants(_freeze(), declarations, rows)

    assert result["accessed_anchor_ids"] == [
        "sglang_decode_standard",
        "sglang_prefill_1k",
    ]
    assert result["forbidden_anchor_ids_accessed"] == []
    fitted = {row["constant_id"]: tools.as_fraction(row["fitted"], "fit") for row in result["fits"]}
    assert fitted == {"collective": 1_000, "pcie": 200, "handoff": 1_000}
    assert all(row["disposition"] == "fitted-parameter-not-measurement" for row in result["fits"])


def test_fit_refuses_held_out_anchor_and_out_of_envelope_value():
    tools = _tools()
    declaration = _declaration("handoff", 50, 0, 100)
    held_out = [
        {
            "constant_id": "handoff",
            "anchor_id": "sglang_prefill_2k",
            "baseline": 54_500,
            "sensitivity_per_ps": 1,
        }
    ]
    with pytest.raises(ValueError, match="not in the calibration split"):
        tools.fit_tunable_constants(_freeze(), [declaration], held_out)

    outside = [
        {
            "constant_id": "handoff",
            "anchor_id": "sglang_prefill_1k",
            "baseline": 56_674,
            "sensitivity_per_ps": 1,
        }
    ]
    with pytest.raises(ValueError, match="outside"):
        tools.fit_tunable_constants(_freeze(), [declaration], outside)


def test_scoring_reads_exactly_held_out_anchors():
    tools = _tools()
    freeze = _freeze()
    held_out = tools._anchor_map(freeze, "held-out")
    predictions = [
        {
            "anchor_id": anchor_id,
            "point": anchor["value"],
            "lower": anchor["value"],
            "upper": anchor["value"],
        }
        for anchor_id, anchor in held_out.items()
    ]

    result = tools.score_held_out_predictions(freeze, predictions)

    assert result["status"] == "PASS"
    assert result["accessed_anchor_ids"] == sorted(held_out)
    assert result["forbidden_anchor_ids_accessed"] == []

    predictions[0]["point"] = 1
    predictions[0]["lower"] = 1
    predictions[0]["upper"] = 1
    assert tools.score_held_out_predictions(freeze, predictions)["status"] == "REFUTED"

    predictions[0]["anchor_id"] = "sglang_prefill_1k"
    with pytest.raises(ValueError, match="exactly the frozen held-out"):
        tools.score_held_out_predictions(freeze, predictions)


def test_uncertainty_propagates_all_three_source_classes_exactly():
    tools = _tools()
    point = {
        "aggregated_output_throughput_tokens_per_second": _fraction(100),
        "per_token_request_delay_ps": _fraction(100),
    }
    declaration = _declaration("handoff", 5, 0, 10)
    uncertainty = {
        "record_bounds": {
            "source_id": "record",
            "aggregated_output_throughput_tokens_per_second": {
                "lower": _fraction(90),
                "upper": _fraction(110),
            },
            "per_token_request_delay_ps": {
                "lower": _fraction(80),
                "upper": _fraction(120),
            },
        },
        "distribution_spreads": [
            {
                "source_id": "distribution",
                "throughput_relative_half_width": 0.1,
                "delay_relative_half_width": 0.1,
            }
        ],
        "tuned_constant_envelopes": [
            {
                "constant_id": "handoff",
                "throughput_tokens_per_second_per_ps": 2,
                "delay_ps_per_ps": -1,
            }
        ],
    }

    interval = tools.propagate_curve_interval(point, uncertainty, [declaration])

    x = interval["aggregated_output_throughput_tokens_per_second"]
    delay = interval["per_token_request_delay_ps"]
    inverse = interval["inverse_per_token_request_delay_tokens_per_second"]
    assert tools.as_fraction(x["lower"], "x") == 70
    assert tools.as_fraction(x["upper"], "x") == 130
    assert tools.as_fraction(delay["lower"], "delay") == 65
    assert tools.as_fraction(delay["upper"], "delay") == 135
    assert tools.as_fraction(inverse["lower"], "inverse") == Fraction(10**12, 135)
    assert tools.as_fraction(inverse["upper"], "inverse") == Fraction(10**12, 65)
    assert [row["source_kind"] for row in interval["source_contributions"]] == [
        "record-bounds",
        "distribution-spread",
        "tuned-constant-envelope",
    ]


def test_dry_run_config_declares_complete_one_plus_one_session():
    study = _study()
    config = _config()

    study.validate_study_config(config)

    deployment = config["configurations"][0]
    assert deployment["framework"] == "vllm"
    assert deployment["pool"] == {
        "prefill_nodes": 1,
        "decode_nodes": 1,
        "gpus_per_node": 8,
    }
    assert deployment["model"]["column"] == "granite-roofline-bootstrap"
    assert deployment["pricing"]["mode"] == "bootstrap"
    assert deployment["requests"]["offered_load_requests_per_second"] == [
        8_000,
        16_000,
        32_000,
    ]
    assert config["study"] == {
        "classification": "dry-run",
        "label": "CORE-54 scaffold dry run",
        "scored_flagship": False,
    }


def test_check_only_consumes_anchor_freeze_without_frontend_import(monkeypatch):
    original_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "resource" or name.startswith(("vllm", "matplotlib")):
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    study = _study()

    freeze, config = study.check_registry(STUDY_DIR / "dry_run_config.json")

    assert freeze["schema"] == "simllm-deployment-curve-anchor-freeze-v1"
    assert config["schema"] == "simllm-deployment-curve-study-config-v1"
    assert study.render_cli_path(PureWindowsPath("C:/runs/result.json")) == ("C:/runs/result.json")


def _synthetic_curve(configuration_id, label):
    return {
        "schema": "simllm-deployment-curve-v1",
        "configuration_id": configuration_id,
        "configuration_label": label,
        "points": [
            {
                "offered_load_requests_per_second": _fraction(1),
                "uncertainty": {
                    "aggregated_output_throughput_tokens_per_second": {
                        "lower": _fraction(90),
                        "point": _fraction(100),
                        "upper": _fraction(110),
                    },
                    "inverse_per_token_request_delay_tokens_per_second": {
                        "lower": _fraction(8),
                        "point": _fraction(9),
                        "upper": _fraction(10),
                    },
                },
            }
        ],
    }


def test_plot_preparation_keeps_configurations_and_disclosures_separate():
    plot = _plot()
    result = {
        "classification": "dry-run",
        "curves": [
            _synthetic_curve("first", "First configuration"),
            _synthetic_curve("second", "Second configuration"),
        ],
    }

    prepared = plot.prepare_plot_data(result, _freeze())

    assert [curve["label"] for curve in prepared["curves"]] == [
        "First configuration",
        "Second configuration",
    ]
    assert len(prepared["disclosure_points"]) == 2
    assert all(anchor["paired_measurement"] is False for anchor in prepared["disclosure_points"])
    assert prepared["vertical_references"] == [
        {
            "anchor_id": "deepseek_production_decode",
            "label": "DeepSeek H800 decode average, delay undisclosed",
            "x": 266_400.0,
        }
    ]


def test_plot_renderer_writes_pdf_and_png_when_matplotlib_is_available(tmp_path):
    pytest.importorskip("matplotlib")
    plot = _plot()
    result = {
        "classification": "dry-run",
        "curves": [_synthetic_curve("first", "First configuration")],
    }
    prepared = plot.prepare_plot_data(result, _freeze())

    pdf_path, png_path = plot.render_figure(prepared, tmp_path / "figure")

    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert png_path.read_bytes().startswith(b"\x89PNG")
