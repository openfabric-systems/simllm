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
RUNNER = STUDY / "run_study.py"
PLOTTER = STUDY / "plot_study.py"
CONFIG = STUDY / "study_config.json"
EXPECTATIONS = STUDY / "expectations.md"
FIGURE_ADDENDUM = STUDY / "figure_addendum.md"
EXPECTATIONS_V2 = STUDY / "expectations_v2.md"
ADJUSTMENTS = STUDY / "external_adjustments.json"
RECORD = STUDY / "record.json"
RESULTS_CSV = STUDY / "results.csv"
PDF = STUDY / "figures/matched-seam-frontier.pdf"
PNG = STUDY / "figures/matched-seam-frontier.png"
EXTERNAL_VENV_ENV = "SIMLLM_EXTERNAL_AIC_VENV"
RECORD_SHA256 = "c08157f5b96f027dd522474f40b4d3159e057e47896a8f0603668c1915feb82d"
RESULTS_CSV_SHA256 = "fb577860e83f6b7b8dc5f21e44644a48d05730f5ddd450cde2381ab80ae98e8a"
PDF_SHA256 = "040ba924f343c763239bfabb93c6cb84c5a14a77a26508be01ad2a16577cc88b"
PNG_SHA256 = "831368ca0dcf1c5f9be9a5a83553c0a365be3ea80906fdc2d7b3cede020c9724"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location("matched_seam_frontier_run", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_plotter():
    spec = importlib.util.spec_from_file_location("matched_seam_frontier_plot", PLOTTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_published_record_and_figures_are_locked() -> None:
    assert _sha256(RECORD) == RECORD_SHA256
    assert _sha256(RESULTS_CSV) == RESULTS_CSV_SHA256
    assert _sha256(PDF) == PDF_SHA256
    assert _sha256(PNG) == PNG_SHA256
    assert b"\r" not in RECORD.read_bytes()
    assert b"\r" not in RESULTS_CSV.read_bytes()

    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["schema"] == "simllm-matched-seam-frontier-record-v1"
    assert record["run_state"] == "nonvoid"
    assert record["voiding_guards"] == []
    assert record["attempt"] == "attempt-0002"
    assert record["run_commit"] == "3e752d58c9e874f234110af69851384ea02873cd"
    assert all(record["fatal_guards"].values())
    assert record["family_tallies"] == {
        "S": {"passed": 13, "denominator": 13},
        "R": {"passed": 10, "denominator": 10},
        "F": {"passed": 12, "denominator": 13},
        "M": {"passed": 2, "denominator": 2},
        "W": {"passed": 1, "denominator": 1},
    }
    misses = [
        row
        for row in record["rows"]
        if row["kind"] == "scored" and not row["passed"]
    ]
    assert [(row["id"], row["observed"]) for row in misses] == [
        ("F-2-09", "0.607495219355")
    ]
    assert record["families"]["M"]["maximum_quotient"]["decimal"] == pytest.approx(
        1.0427153998047758
    )
    assert record["families"]["D"]["raw_prefill_pass_ms"] == pytest.approx(
        99.20380474486889
    )
    assert record["families"]["D"]["residual_from_raw_pass_ms"] == pytest.approx(
        97.21919525513111
    )
    assert len(record["families"]["candidate_grid"]["agg"]) == 25
    assert len(record["families"]["candidate_grid"]["disagg"]) == 10
    assert len(record["families"]["F"]["ideal_frontier"]) == 10
    assert len(record["families"]["F"]["packet_frontier"]) == 9
    serialized = RECORD.read_text(encoding="utf-8")
    assert "/data3/" not in serialized
    assert "/home/" not in serialized

    with RESULTS_CSV.open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(record["rows"])


def test_freezes_and_external_adjustment_table_are_locked() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frozen = config["frozen_inputs"]
    assert _sha256(EXPECTATIONS) == frozen["expectations"]["sha256"]
    assert _sha256(EXPECTATIONS_V2) == frozen["expectations_v2"]["sha256"]
    assert _sha256(ADJUSTMENTS) == frozen["external_adjustments"]["sha256"]
    assert _sha256(FIGURE_ADDENDUM) == (
        "cc4dcb8c82bbcd5e542457b56d91ddf172af2cbe05e6bac5c865535dcc307762"
    )
    table = json.loads(ADJUSTMENTS.read_text(encoding="utf-8"))
    adjustments = {row["id"]: row for row in table["adjustments"]}
    assert set(adjustments) == {
        "prefill_latency_correction",
        "decode_latency_correction",
        "prefill_rate_matching_degradation",
        "decode_rate_matching_degradation",
        "autoscale_ttft_correction",
        "memory_bandwidth_empirical_scale",
        "memory_empirical_constant_latency",
        "context_attention_extra_latency_correction",
    }
    assert adjustments["prefill_rate_matching_degradation"]["value"] == "0.9"
    assert adjustments["decode_rate_matching_degradation"]["value"] == "0.92"
    assert "transposes those phase names" in table["phase_assignment_finding"]
    assert all(row["source"]["start_line"] > 0 for row in adjustments.values())
    assert all(row["documentation"]["start_line"] > 0 for row in adjustments.values())


def test_scored_value_trace_rejects_reachable_simllm_fitted_value() -> None:
    runner = _load_runner()
    trace = {
        "nodes": [
            {
                "id": "external",
                "kind": "external_composed_service",
                "origin": "external-aiconfigurator",
                "value": 10,
                "dependencies": [],
            },
            {
                "id": "local-fit",
                "kind": "fitted_constant",
                "origin": "simllm-study",
                "value": 2,
                "dependencies": [],
            },
            {
                "id": "scored",
                "kind": "scored_coordinate",
                "origin": "simllm-projection",
                "value": 12,
                "dependencies": ["external", "local-fit"],
            },
        ],
        "scored_roots": ["scored"],
    }

    assert runner._validate_scored_value_trace(trace) == [
        "forbidden SimLLM-authored fitted_constant reaches scored root through local-fit"
    ]


def test_runner_starts_directly_without_pythonpath(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    external_venv = tmp_path / "external-venv"
    external_venv.mkdir()
    txt2bin = tmp_path / "txt2bin"
    txt2bin.touch()
    htsim_rnic = tmp_path / "htsim_rnic"
    htsim_rnic.touch()
    completed = subprocess.run(
        [
            sys.executable,
            os.fspath(RUNNER),
            "--bulk-root",
            os.fspath(tmp_path / "bulk"),
            "--external-venv",
            os.fspath(external_venv),
            "--txt2bin",
            os.fspath(txt2bin),
            "--htsim-rnic",
            os.fspath(htsim_rnic),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "ModuleNotFoundError" not in completed.stderr
    assert "SIMLLM_EXTERNAL_AIC_VENV has no Python interpreter" in completed.stderr


def test_tracked_external_grid_constructs_all_declared_candidates() -> None:
    runner = _load_runner()
    disagg = runner._csv_rows(runner.DISAGG_PATH)
    agg = runner._csv_rows(runner.AGG_PATH)
    inventory = "a" * 64

    disagg_candidates = [
        runner._candidate(
            row,
            row_number=index,
            disaggregated=True,
            inventory_sha256=inventory,
        )
        for index, row in enumerate(disagg, start=1)
    ]
    agg_candidates = [
        runner._candidate(
            row,
            row_number=index,
            disaggregated=False,
            inventory_sha256=inventory,
        )
        for index, row in enumerate(agg, start=1)
    ]

    assert len(disagg_candidates) == 10
    assert len(agg_candidates) == 25
    assert {candidate.pools[1].tensor_parallel for candidate in disagg_candidates} == {
        2,
        4,
        8,
    }
    assert {candidate.pools[0].tensor_parallel for candidate in agg_candidates} == {
        4,
        8,
    }
    assert all(
        sum(pool.engines * pool.gpus_per_engine for pool in candidate.pools)
        == candidate.budget.max_gpus
        for candidate in (*disagg_candidates, *agg_candidates)
    )


def test_results_csv_writer_is_lf_only() -> None:
    runner = _load_runner()
    payload = runner._csv_bytes(
        [
            runner._scored_row(
                "S",
                "S-test",
                True,
                expected="exact",
                observed="exact",
            )
        ]
    )

    assert b"\r" not in payload
    assert len(list(csv.DictReader(payload.decode().splitlines()))) == 1


def test_directed_figure_is_a_record_only_projection() -> None:
    plotter = _load_plotter()
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    plot = plotter.prepare_plot_data(record)

    assert [series["id"] for series in plot["series"]] == [
        "external-agg",
        "external-disagg",
        "simllm-ideal",
        "simllm-packet",
    ]
    assert plot["agreement"] == {
        "rows": [
            {
                "row": row["row"],
                "quotient": pytest.approx(row["quotient"]["decimal"]),
            }
            for row in record["families"]["R"]["rows"]
        ],
        "frozen_band": [0.98, 1.02],
        "minimum": "0.999946608534",
        "maximum": "1.000076344974",
    }
    assert plot["mechanism"] == {
        "arrow_enabled": True,
        "candidate_rows": [1, 3],
        "selected_row": 3,
        "x": pytest.approx(84.00604166008027),
        "ideal_y": pytest.approx(644.3347078275151),
        "packet_y": pytest.approx(617.9391883424295),
        "quotient": "1.042715399805",
        "label": (
            "Their planner class prices no network cost.\n"
            "Our unpriced-network arm charges zero network service.\n"
            "This workload: packet-priced / unpriced-network\n"
            "= 1.042715399805.\n"
            "Unpriced: MEASURED-EXTERNAL.\n"
            "Packet-priced: MEASURED-EXTERNAL + SIM-DERIVED."
        ),
    }
    assert plot["f209"] == {"quotient": "0.607495219355", "passed": False}
    assert "different schedule regime and is not plotted" in plot["caption"]
    assert "does not isolate receiver-side serialization" in plot["caption"]


def test_live_sdk_reproduces_frozen_service_oracles() -> None:
    raw_venv = os.environ.get(EXTERNAL_VENV_ENV)
    if raw_venv is None:
        pytest.skip(
            f"live Family S check requires {EXTERNAL_VENV_ENV}; the locked record remains covered"
        )
    venv = Path(raw_venv)
    python = next(
        (
            path
            for path in (venv / "bin/python", venv / "Scripts/python.exe")
            if path.is_file()
        ),
        None,
    )
    if python is None:
        pytest.skip(
            f"live Family S check requires a Python interpreter in {EXTERNAL_VENV_ENV}"
        )
    completed = subprocess.run(
        [os.fspath(python), os.fspath(RUNNER), "--worker", "live-sdk"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    observed = {
        row["id"]: row["service_ms_hex"]
        for row in json.loads(completed.stdout)["services"]
    }
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = {
        oracle["id"]: oracle[
            "expected_step_ms_hex" if phase == "decode" else "expected_service_ms_hex"
        ]
        for phase in ("decode", "prefill")
        for oracle in config["oracles"][phase]
    }

    assert observed == expected
