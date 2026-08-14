"""Regression checks for the frozen SGLang per-step host cost study.

The study's own ``--check-only`` gate re-derives every frozen literal from the
frozen constants, so the cheapest way to keep the freeze honest is to run that
derivation here. The second half checks the tracked results against the same
frozen file, so a later edit to either one cannot drift from the other
unnoticed. Neither half runs a backend and neither imports SGLang.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY / "examples/sglang_host_step_v1"
RUN_PATH = STUDY_DIR / "run_study.py"
RESULTS_PATH = STUDY_DIR / "results.json"


def _study_module():
    spec = importlib.util.spec_from_file_location("sglang_host_step_study", RUN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_check_only_gate_rederives_every_frozen_literal():
    completed = subprocess.run(
        [sys.executable, str(RUN_PATH), "--check-only"],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "every frozen literal re-derives" in completed.stdout


def test_the_frozen_crossover_is_one_launch_wide():
    module = _study_module()
    document = module.load_expectations()
    service = module.derived_provider_service_ps(document)
    floors = {cell["name"]: cell["launch_floor_ps"] for cell in document["cells"]}

    assert all(floors["graph122"] <= value for value in service)
    assert all(floors["graph123"] > value for value in service)
    assert all(floors["eager41"] <= value for value in service)
    assert all(floors["eager42"] > value for value in service)
    assert floors["graph123"] - floors["graph122"] == 809_306
    assert floors["eager42"] - floors["eager41"] == 2_364_255


def test_the_masked_cells_are_registered_as_not_silent():
    """A masked calibrated cell still moves the step, by under 24 ns."""

    module = _study_module()
    document = module.load_expectations()

    for cell in document["masked_cells"]:
        deltas = document["step_latency_delta_ps"][cell]
        assert all(0 < value <= 24_000 for value in deltas)


def test_the_transferred_bracket_reproduces_the_accepted_enclosures():
    module = _study_module()
    document = module.load_expectations()
    enclosures = module.derived_enclosures(document)

    assert set(enclosures["graph440"]) == {356_095}
    assert set(enclosures["eager567"]) == {1_340_533}


def test_the_tracked_results_agree_with_the_freeze():
    module = _study_module()
    document = module.load_expectations()
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    assert results["schema"] == module.RESULTS_SCHEMA
    assert results["records_sha256"] == document["input"]["sha256"]

    scoring = results["scoring"]
    assert scoring["void"] is False
    assert all(scoring["fatal_guards"].values())
    for name, expected in (("R0", 9), ("R1", 54), ("R2", 18)):
        rows = scoring[name]["rows"]
        assert len(rows) == expected
        assert all(row["pass"] for row in rows)

    ideal = results["cells"]["ideal"]["steps"]
    for name, deltas in document["step_latency_delta_ps"].items():
        measured = [
            row["step_latency_ps"] - base["step_latency_ps"]
            for row, base in zip(results["cells"][name]["steps"], ideal, strict=True)
        ]
        assert measured == deltas


def test_r2_records_both_halves_of_its_frozen_conjunction():
    """The freeze registered exposure AND the bound name; both must be recorded.

    The first run scored the exposure half only, because the bound string was
    dropped between ``HostStepEstimate`` and the recorded rows. This pins the
    conjunction so the same gap cannot reopen.
    """

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    rows = results["scoring"]["R2"]["rows"]

    assert rows
    for row in rows:
        assert row["exposure_half"] is True
        assert row["bound_half"] is True
        assert row["masked_bound"] == row["provider_bound"] == "memory"
        assert row["bound_cell_bound"] == "host-initiation"

    masked = {"graph122", "eager41"}
    for name, cell in results["cells"].items():
        recorded = {step["represented_bound"] for step in cell["steps"]}
        if name == "ideal" or name in masked:
            assert recorded == {"memory"}
        else:
            assert recorded == {"host-initiation"}


def test_the_tracked_results_carry_the_reducible_digest_summary():
    """The runner emits the tracked form; no untracked post-processing step."""

    module = _study_module()
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    assert "artifact_digests" not in results
    summary = results["artifact_digest_summary"]
    assert set(summary) == set(results["cells"]) | {"reference-preseam"}
    for cell in summary.values():
        assert set(cell) == {"artifact_count", "goal_files", "rollup_sha256"}
        assert cell["artifact_count"] == 1296
        assert cell["goal_files"] == 432

    rows = [("a.goal", "0" * 64), ("b.bin", "1" * 64)]
    derived = module.artifact_digest_summary(rows)
    assert derived["artifact_count"] == 2
    assert derived["goal_files"] == 1
    assert len(derived["rollup_sha256"]) == 64


def test_the_recorded_fabric_term_matches_the_hand_closed_form():
    """48 collectives of seven 2,048-byte pairs per token, plus propagation."""

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    new_tokens = [12, 1, 7, 2, 1, 11, 2, 1, 1]

    for cell in results["cells"].values():
        for row, tokens in zip(cell["steps"], new_tokens, strict=True):
            fabric = row["step_latency_ps"] - 1000 * row["enclosed_calc_ns"]
            predicted = 48 * (tokens * 2048 * 7 * 20 + 2_000_000)
            assert fabric - predicted == 48
