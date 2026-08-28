from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/matched_seam_frontier_v1"
RUNNER = STUDY / "run_study.py"
CONFIG = STUDY / "study_config.json"
EXTERNAL_VENV_ENV = "SIMLLM_EXTERNAL_AIC_VENV"


def _load_runner():
    spec = importlib.util.spec_from_file_location("matched_seam_frontier_run", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
