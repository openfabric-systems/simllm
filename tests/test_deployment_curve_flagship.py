from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path, PureWindowsPath

import pytest

from simllm.core import DeclaredKvHandoffPolicy

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
    return _module("deployment_curve_flagship_tools", "flagship_tools.py")


def _runner():
    return _module("deployment_curve_flagship_runner", "run_flagship.py")


def _plot():
    return _module("deployment_curve_flagship_plot", "plot_flagship.py")


def _json(name: str):
    return json.loads((STUDY_DIR / name).read_text(encoding="utf-8"))


def test_scored_freeze_locks_separate_experiments_and_stable_identity():
    tools = _tools()
    frozen = _json("scored_expectations.json")

    tools.validate_scored_expectations(frozen)

    allocation = frozen["allocation_ruling"]
    assert allocation["prefill_experiment"]["ranks"] == 32
    assert allocation["decode_experiment"]["ranks"] == 72
    assert allocation["structural_comparator_only"] == {
        "prefill_plus_decode_nodes": 13,
        "ranks": 104,
        "may_be_called_96_gpu_system": False,
    }
    stable = frozen["stable_cross_run_identity_fields"]
    assert "complete serialized request-result bytes" in stable["excluded"]
    assert "prefill_internal_request_id" in stable["excluded"]
    assert "timeline.request_id" in stable["included"]


def test_pre_tuning_bands_are_exact_and_mtp_is_blocked():
    tools = _tools()
    frozen = _json("scored_expectations.json")
    rows = {row["anchor_id"]: row for row in frozen["pre_tuning_predicted_bands"]}
    constant = frozen["constants"]["tunable"][0]

    interval = tools.prediction_interval(
        rows["sglang_prefill_2k"],
        constant,
        constant["initial"],
    )

    assert interval["lower"] == rows["sglang_prefill_2k"]["lower"]
    assert interval["point"] == rows["sglang_prefill_2k"]["point"]
    assert interval["upper"] == rows["sglang_prefill_2k"]["upper"]
    assert rows["sglang_decode_simulated_mtp"] == {
        "anchor_id": "sglang_decode_simulated_mtp",
        "status": "BLOCKED",
        "prediction": None,
        "reason": "The candidate record has no EP72 MTP batch-16 KV-4000 cell.",
        "dependency": "COMP-72 resumable Merlin execution",
    }


def test_bounded_fit_reads_calibration_only_and_selects_physical_floor():
    tools = _tools()
    fit = tools.fit_frozen_constant(
        _json("expectations.json"),
        _json("scored_expectations.json"),
    )

    assert fit["status"] == "FROZEN"
    assert fit["fitted_ps"] == 0
    assert fit["envelope"] == {"lower": 0, "upper": 30_128_029}
    assert fit["accessed_anchor_ids"] == [
        "sglang_decode_standard",
        "sglang_prefill_1k",
    ]
    assert fit["forbidden_anchor_ids_accessed"] == []


def test_one_shot_score_reads_only_priced_held_out_and_refutes_point_error():
    tools = _tools()
    anchors = _json("expectations.json")
    frozen = _json("scored_expectations.json")
    fit = tools.fit_frozen_constant(anchors, frozen)

    score = tools.score_frozen_fit(anchors, frozen, fit)

    assert score["status"] == "REFUTED"
    assert score["accessed_anchor_ids"] == [
        "sglang_prefill_2k",
        "sglang_prefill_4k",
    ]
    assert score["forbidden_anchor_ids_accessed"] == []
    maximum = tools.as_fraction(score["maximum_absolute_relative_error"], "max")
    assert maximum > Fraction(1, 20)
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


def test_flagship_config_conserves_shapes_and_exact_interarrivals():
    tools = _tools()
    config = _json("flagship_config.json")

    tools.validate_flagship_config(config)

    assert config["model"]["weights_required"] is False
    assert [
        row["prompt_tokens"] * row["requests"]
        for row in config["exact_shape_observations"][:3]
    ] == [16_384, 16_384, 16_384]
    assert config["packet_observation"]["prefill_ranks"] == list(range(8))
    assert config["packet_observation"]["decode_ranks"] == list(range(32, 40))
    assert config["live_session"]["context_length"] > 4096


def test_runner_projects_handoff_event_fields_without_assuming_a_json_method():
    runner = _runner()
    event = DeclaredKvHandoffPolicy(20).schedule(
        submitted_at_ps=7,
        request_id="request",
        kv_bytes=8,
    )

    assert runner._handoff_event_json(event) == {
        "request_id": "request",
        "authority": "simllm-declared-kv-handoff-v1",
        "pricing_arm": "declared-constant",
        "kv_bytes": 8,
        "submitted_at_ps": 7,
        "eligible_at_ps": 7,
        "started_at_ps": 7,
        "finished_at_ps": 27,
        "completed_at_ps": 27,
    }


def test_binding_qualification_never_fits_or_scores(monkeypatch, tmp_path):
    runner = _runner()
    config = _json("flagship_config.json")

    def observation(config, model_path, run_dir, candidate, declared, *, suffix):
        del config, model_path, run_dir, candidate, suffix
        pool = declared["pool"]
        return {
            "anchor_id": declared["anchor_id"],
            "pool": pool,
            "pricing_provenance": {
                pool: {
                    "record_sha256": (
                        "ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52"
                    ),
                    "acceptance_status": "candidate",
                    "lookup_hits": 1,
                    "lookup_misses": 0,
                    "selected_entry_key_sha256s": ["1" * 64],
                }
            },
        }

    monkeypatch.setattr(runner, "_runtime_observation", observation)
    args = type(
        "Args",
        (),
        {
            "run_dir": tmp_path / "qualification",
            "model_path": tmp_path / "model",
            "config": STUDY_DIR / "flagship_config.json",
        },
    )()

    result = runner.run_binding_qualification(config, args)

    assert result["status"] == "PASS"
    assert result["fit_performed"] is False
    assert result["anchor_numeric_values_accessed"] is False
    assert result["held_out_score_performed"] is False


def test_shared_curve_records_keep_output_axis_and_target_scale():
    tools = _tools()
    frozen = _json("scored_expectations.json")
    row = next(
        value
        for value in frozen["pre_tuning_predicted_bands"]
        if value["anchor_id"] == "sglang_decode_standard"
    )
    interval = tools.prediction_interval(row, frozen["constants"]["tunable"][0], 0)
    config = _json("flagship_config.json")["publication_curves"][0]

    curve = tools.build_publication_curve(config, interval)

    assert curve["schema"] == "simllm-deployment-curve-v1"
    assert curve["decode_engines"] == 9
    assert curve["orientation"] == {
        "x": "aggregated-output-throughput-rightward",
        "y": "inverse-per-token-request-delay-upward",
    }
    assert len(curve["points"]) == 5
    assert all(
        point["uncertainty"]["method"] == "deterministic-additive-interval-v1"
        for point in curve["points"]
    )


def test_stable_projection_excludes_process_and_internal_request_ids():
    tools = _tools()
    value = {
        "request_id": "stable",
        "admitted_at_ps": 0,
        "prefill_eligible_at_ps": 0,
        "prefill_completed_at_ps": 10,
        "handoff": {"request_id": "stable", "kv_bytes": 8},
        "decode_eligible_at_ps": 20,
        "decode_token_completed_at_ps": [30],
        "prefill_engine_id": "prefill-0",
        "decode_engine_id": "decode-0",
        "prefill_internal_request_id": "random-a",
        "decode_internal_request_id": "random-b",
        "bootstrap_token_id": 1,
        "decode_token_ids": [2],
        "prefill_step_count": 1,
        "decode_step_count": 1,
        "join_metadata": {
            "schema": "join",
            "prefill_process_id": 10,
            "decode_process_id": 11,
        },
    }

    projected = tools.stable_request_projection(value)

    assert "prefill_internal_request_id" not in projected
    assert "decode_internal_request_id" not in projected
    assert "prefill_process_id" not in projected["join_metadata"]
    assert "decode_process_id" not in projected["join_metadata"]


def test_check_only_needs_no_frontend_or_unix_only_import(monkeypatch):
    original_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "resource" or name.startswith(("sglang", "matplotlib")):
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    runner = _runner()
    args = type("Args", (), {"config": STUDY_DIR / "flagship_config.json"})()

    anchors, scored, config = runner.check_registry(args, require_runtime=False)

    assert anchors["schema"] == "simllm-deployment-curve-anchor-freeze-v1"
    assert scored["schema"] == "simllm-deployment-curve-scored-expectations-v1"
    assert config["schema"] == "simllm-deployment-curve-flagship-config-v1"
    assert runner.render_cli_path(PureWindowsPath("C:/runs/result.json")) == (
        "C:/runs/result.json"
    )


def test_flagship_plot_preparation_keeps_two_legends_and_prefill_panel():
    tools = _tools()
    plot = _plot()
    anchors = _json("expectations.json")
    frozen = _json("scored_expectations.json")
    config = _json("flagship_config.json")
    fit = tools.fit_frozen_constant(anchors, frozen)
    score = tools.score_frozen_fit(anchors, frozen, fit)
    predictions = []
    for row in frozen["pre_tuning_predicted_bands"]:
        if row.get("status") == "BLOCKED":
            predictions.append(
                {"anchor_id": row["anchor_id"], "status": "BLOCKED", "prediction": None}
            )
        else:
            predictions.append(
                {
                    "anchor_id": row["anchor_id"],
                    "status": "PREDICTED",
                    "prediction": tools.prediction_interval(
                        row,
                        frozen["constants"]["tunable"][0],
                        fit["fitted_ps"],
                    ),
                }
            )
    capacity = next(
        row["prediction"] for row in predictions if row["anchor_id"] == "sglang_decode_standard"
    )
    result = {
        "curves": [
            tools.build_publication_curve(curve, capacity)
            for curve in config["publication_curves"]
        ],
        "anchor_predictions": predictions,
        "held_out_score": score,
    }

    prepared = plot.prepare_flagship_plot(result, anchors)

    assert len(prepared["curves"]) == 2
    assert len(prepared["prefill"]) == 3
    assert prepared["verdict"] == "REFUTED"
    assert prepared["mtp_dependency"] == "COMP-72 resumable Merlin execution"


def test_flagship_renderer_writes_pdf_and_png_when_matplotlib_is_available(tmp_path):
    pytest.importorskip("matplotlib")
    plot_module = _plot()
    tools = _tools()
    anchors = _json("expectations.json")
    frozen = _json("scored_expectations.json")
    config = _json("flagship_config.json")
    fit = tools.fit_frozen_constant(anchors, frozen)
    score = tools.score_frozen_fit(anchors, frozen, fit)
    predictions = [
        {
            "anchor_id": row["anchor_id"],
            "status": "PREDICTED",
            "prediction": tools.prediction_interval(
                row,
                frozen["constants"]["tunable"][0],
                fit["fitted_ps"],
            ),
        }
        for row in frozen["pre_tuning_predicted_bands"]
        if row.get("status") != "BLOCKED"
    ]
    capacity = next(
        row["prediction"] for row in predictions if row["anchor_id"] == "sglang_decode_standard"
    )
    result = {
        "curves": [
            tools.build_publication_curve(curve, capacity)
            for curve in config["publication_curves"]
        ],
        "anchor_predictions": predictions,
        "held_out_score": score,
    }
    prepared = plot_module.prepare_flagship_plot(result, anchors)

    pdf, png = plot_module.render_flagship_figure(prepared, tmp_path / "flagship")

    assert pdf.read_bytes().startswith(b"%PDF")
    assert png.read_bytes().startswith(b"\x89PNG")
