from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/matched_seam_frontier_v1"
RUNNER = STUDY / "run_agg_arm.py"
PLOTTER = STUDY / "plot_agg.py"
BASE_RECORD = STUDY / "record.json"
EXTERNAL_AGG = ROOT / "examples/frontier_comparison_v1/external/agg_pareto.csv"
EXPECTATIONS = STUDY / "expectations_agg.md"
ADJUSTMENTS = STUDY / "external_adjustments_agg.json"
AGG_RECORD = STUDY / "agg_record.json"
AGG_RESULTS_CSV = STUDY / "agg_results.csv"
AGG_RESULTS = STUDY / "AGG_RESULTS.md"
AGG_FIGURES = {
    STUDY / "figures/matched-seam-frontier-aggregate.pdf": (
        "89a52a3bd0282923332568bc2cfee8c54bb1c9f36b7e6194639a01df99e78b86"
    ),
    STUDY / "figures/matched-seam-frontier-aggregate.png": (
        "932177b1d9320f1f7b23834e6c44b8ddcbbb86e04a46f8294b22c6778d817fd9"
    ),
    STUDY / "figures/matched-seam-frontier-aggregate-publication.pdf": (
        "83dc1445826914df60e430bb67dfe8b756552dfcd8a6318c5c433ec3b3d32ff0"
    ),
    STUDY / "figures/matched-seam-frontier-aggregate-publication.png": (
        "92e8e76c96ddc3d9ed2ae42e035c6ee9e1f134b8bcb6793fc93ae851ef7f5252"
    ),
}
AGG_RECORD_SHA256 = "799e491cc474ac08e87767187d035e9205c348dbdc9c97e61e947e34163c752f"
AGG_RESULTS_CSV_SHA256 = (
    "44ac173646082aff8bf8e453b27edc9479850a444b33f5eeef696d085a6b1b98"
)
AGG_RESULTS_SHA256 = "32e00a2cd29322c6ecc46770a577132f5b5fbbbe947e429f5fca4c898f5b0643"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_record() -> dict[str, object]:
    with EXTERNAL_AGG.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    projection = [
        {
            "row": index,
            "point": {
                "configuration_id": f"aggregate-row-{index:02d}",
                "tokens_per_second_per_gpu": float(row["tokens/s/gpu"]).hex(),
                "tokens_per_second_per_user": float(row["tokens/s/user"]).hex(),
            },
        }
        for index, row in enumerate(rows, start=1)
    ]
    return {
        "schema": "simllm-matched-seam-aggregate-record-v1",
        "base_record_sha256": _sha256(BASE_RECORD),
        "families": {"AR": {"baseline_projection": projection}},
    }


def test_aggregate_freeze_adjustments_and_protected_inputs_are_locked() -> None:
    runner = _load(RUNNER, "matched_seam_aggregate_runner")

    assert _sha256(EXPECTATIONS) == runner.EXPECTATIONS_SHA256
    assert _sha256(ADJUSTMENTS) == runner.ADJUSTMENTS_SHA256
    assert _sha256(EXTERNAL_AGG) == runner.EXTERNAL_AGG_SHA256
    assert runner._protected_hashes() == runner.PROTECTED_PRIOR_SHA256
    rows = runner._adjustment_rows()
    assert len(rows) == 10
    assert {
        str(row["id"])
        for row in rows
        if row["aggregate_tpot_reachable"] or row["aggregate_ttft_reachable"]
    } == runner.EXPECTED_APPLIED_ADJUSTMENTS
    assert len(runner._source_references(rows)) == 23


def test_aggregate_result_is_locked_nonvoid_and_complete() -> None:
    assert _sha256(AGG_RECORD) == AGG_RECORD_SHA256
    assert _sha256(AGG_RESULTS_CSV) == AGG_RESULTS_CSV_SHA256
    assert _sha256(AGG_RESULTS) == AGG_RESULTS_SHA256
    assert all(_sha256(path) == expected for path, expected in AGG_FIGURES.items())

    record = json.loads(AGG_RECORD.read_text(encoding="utf-8"))
    assert record["schema"] == "simllm-matched-seam-aggregate-record-v1"
    assert record["run_state"] == "nonvoid"
    assert record["voiding_guards"] == []
    assert record["attempt"] == "attempt-0002"
    assert record["run_commit"] == "6450a656e826bea325ec1127d4b2f7ab9be910e0"
    assert len(record["fatal_guards"]) == 11
    assert all(record["fatal_guards"].values())
    assert record["family_tallies"] == {
        "AR": {"passed": 25, "denominator": 25},
        "W": {"passed": 1, "denominator": 1},
    }
    assert record["determinism"] == {
        "comparison": "byte-for-byte complete scored evaluation JSON",
        "equal": True,
        "evaluation_sha256": [
            "0f37ed4ce38ca2b985501fc3072be046917cd3c57f272f0cd10c5579e08ed16f",
            "0f37ed4ce38ca2b985501fc3072be046917cd3c57f272f0cd10c5579e08ed16f",
        ],
        "excluded_by_name": ["elapsed_seconds", "W-1"],
        "fresh_processes": 2,
    }

    aggregate_rows = record["families"]["AR"]["rows"]
    quotients = [row["quotient"]["decimal"] for row in aggregate_rows]
    assert len(aggregate_rows) == 25
    assert min(quotients) == pytest.approx(0.9999300628273019)
    assert max(quotients) == pytest.approx(1.000074535718821)
    assert all(row["passed"] for row in aggregate_rows)

    ttft_rows = record["families"]["TTFT"]["rows"]
    residuals = [row["publication_residual_ms"]["decimal"] for row in ttft_rows]
    assert len(ttft_rows) == 25
    assert min(residuals) == pytest.approx(-0.0004979317788524895)
    assert max(residuals) == pytest.approx(0.0004985500327734371)
    assert all(row["residual_within_rounding_bound"] for row in ttft_rows)

    sensitivity = {
        row["adjustment_id"]: row
        for row in record["families"]["ADJ"]["rows"]
    }
    assert len(sensitivity) == 10
    assert sensitivity["trtllm_tpot_mixed_step_reduction"][
        "tpot_quotient_maximum"
    ]["decimal"] == pytest.approx(1.1548670761254167)
    assert sensitivity["aggregate_ttft_queueing_heuristic"][
        "ttft_quotient_minimum"
    ]["decimal"] == pytest.approx(0.3508773075696615)
    unreachable = {
        "prefill_latency_correction",
        "decode_latency_correction",
        "prefill_rate_matching_degradation",
        "decode_rate_matching_degradation",
        "autoscale_ttft_correction",
    }
    assert all(
        sensitivity[factor]["complete_projection_byte_identical"]
        for factor in unreachable
    )

    arms = record["network_arms"]
    assert arms["unpriced"]["projection"] == arms["packet"]["projection"]
    assert arms["packet"]["native_process_invocations"] == 0
    assert arms["packet"]["handoff_bytes"] == 0
    assert arms["packet"]["handoff_flows"] == 0

    serialized = AGG_RECORD.read_text(encoding="utf-8")
    assert "/data3/" not in serialized
    assert "/home/" not in serialized
    assert "\N{EM DASH}" not in AGG_RESULTS.read_text(encoding="utf-8")
    for path in (AGG_RECORD, AGG_RESULTS_CSV, AGG_RESULTS):
        assert b"\r" not in path.read_bytes()
    with AGG_RESULTS_CSV.open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == len(record["rows"]) == 73


def test_aggregate_plot_contract_names_strategy_and_traffic() -> None:
    plotter = _load(PLOTTER, "matched_seam_aggregate_plotter")
    record = _synthetic_record()
    base_record = json.loads(BASE_RECORD.read_text(encoding="utf-8"))

    study = plotter.prepare_study_data(record, base_record)
    publication = plotter.prepare_publication_data(record, base_record)

    for projection in (study, publication):
        assert len(projection["series"]) == 6
        assert all(len(series["points"]) > 0 for series in projection["series"])
        labels = [series["label"] for series in projection["series"]]
        assert all("agg" in label or "disagg" in label for label in labels)
        assert all(
            any(
                word in label
                for word in (
                    "co-located mix",
                    "split P/D",
                    "unpriced P/D",
                    "packet P/D",
                    "zero-byte P/D",
                )
            )
            for label in labels
        )
    assert study["series"][-2]["points"] == study["series"][-1]["points"]
    assert publication["series"][-2]["points"] == publication["series"][-1]["points"]


def test_aggregate_plotter_renders_both_additive_pairs(tmp_path: Path) -> None:
    plotter = _load(PLOTTER, "matched_seam_aggregate_renderer")
    paths = {
        "study_pdf": tmp_path / "study.pdf",
        "study_png": tmp_path / "study.png",
        "publication_pdf": tmp_path / "publication.pdf",
        "publication_png": tmp_path / "publication.png",
    }

    plotter.render_all(_synthetic_record(), **paths)

    assert all(path.stat().st_size > 10_000 for path in paths.values())


def test_new_aggregate_scripts_run_standalone_without_pythonpath() -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    for script in (RUNNER, PLOTTER):
        completed = subprocess.run(
            [sys.executable, os.fspath(script), "--help"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout


def test_new_aggregate_artifacts_use_lf_and_no_personal_paths() -> None:
    for path in (RUNNER, PLOTTER, EXPECTATIONS, ADJUSTMENTS):
        payload = path.read_bytes()
        assert b"\r" not in payload
        assert b"/home/" not in payload
        assert b"/data3/" not in payload
