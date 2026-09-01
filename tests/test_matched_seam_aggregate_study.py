from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/matched_seam_frontier_v1"
RUNNER = STUDY / "run_agg_arm.py"
PLOTTER = STUDY / "plot_agg.py"
BASE_RECORD = STUDY / "record.json"
EXTERNAL_AGG = ROOT / "examples/frontier_comparison_v1/external/agg_pareto.csv"
EXPECTATIONS = STUDY / "expectations_agg.md"
ADJUSTMENTS = STUDY / "external_adjustments_agg.json"


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
            any(word in label for word in ("request mix", "traffic", "handoff"))
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
