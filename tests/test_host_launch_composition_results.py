"""Lock the host_launch_composition_v1 result against its freeze.

The study is a pure model-side computation, so unlike the hardware studies
these checks re-run it and require the committed report to match exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

STUDY = Path(__file__).resolve().parents[1] / "examples" / "host_launch_composition_v1"
FREEZE_JSON = STUDY / "expectations.json"
RESULTS_JSON = STUDY / "results.json"
RESULTS_MD = STUDY / "RESULTS.md"
EXPECTATIONS_MD = STUDY / "expectations.md"

EM_DASH = "—"


@pytest.fixture(scope="module")
def freeze() -> dict:
    return json.loads(FREEZE_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def results() -> dict:
    return json.loads(RESULTS_JSON.read_text(encoding="utf-8"))


def test_the_study_documents_use_no_em_dash() -> None:
    for path in (RESULTS_MD, EXPECTATIONS_MD):
        assert EM_DASH not in path.read_text(encoding="utf-8")


def test_the_report_names_the_freeze_it_was_scored_against(
    freeze: dict,
    results: dict,
) -> None:
    import hashlib

    digest = hashlib.sha256(FREEZE_JSON.read_bytes()).hexdigest()
    assert results["freeze_sha256"] == digest
    assert results["study"] == freeze["study"]


def test_the_run_is_nonvoid_and_every_fatal_guard_held(results: dict) -> None:
    assert results["run_state"] == "nonvoid"
    assert results["voiding_guards"] == []
    for name, guard in results["fatal_guards"].items():
        assert guard["passed"], name


def test_the_scored_denominator_matches_the_freeze(freeze: dict, results: dict) -> None:
    assert results["scored_denominator"] == freeze["denominators"]["genuine_risk"]
    assert results["scored_passed"] == results["scored_denominator"]
    scored = {name for name, r in results["relations"].items() if r["scored"]}
    assert scored == {"R4", "R5"}


def test_the_probed_relation_is_never_added_to_a_denominator(results: dict) -> None:
    r2 = results["relations"]["R2"]
    assert r2["kind"] == "post-specified-regression"
    assert r2["scored"] is False


def test_rerunning_the_study_reproduces_the_committed_cells(results: dict) -> None:
    import sys

    sys.path.insert(0, str(STUDY))
    try:
        import run_study
    finally:
        sys.path.pop(0)

    assert run_study.sweep() == results["cells"]


def test_every_measured_regime_cell_predicts_exactly_zero(
    freeze: dict,
    results: dict,
) -> None:
    eager_point = freeze["constants_under_test"]["turing_eager_host_point_ps"]
    covered = 0
    for cell in results["cells"]:
        if cell["c1_ps"] >= eager_point:
            assert cell["delta_ps"] == 0, cell["cell_id"]
            assert cell["eager"]["exposed_ps"] == 0, cell["cell_id"]
            assert cell["graph"]["exposed_ps"] == 0, cell["cell_id"]
            covered += 1
    assert covered == results["relations"]["R2"]["cell_count"] == 20


def test_the_refutation_relations_carry_their_frozen_magnitudes(results: dict) -> None:
    r4 = results["relations"]["R4"]
    assert r4["measured_cv"] < 0.04
    assert r4["period_span"] > 10
    assert r4["modeled_per_kernel_delta_ps"] == [0, 0, 0]

    r5 = results["relations"]["R5"]
    assert r5["relative_errors"] == [1.0, 1.0, 1.0]
    assert all(miss > 990 for miss in r5["absolute_miss_in_gpu_cycles"])


def test_the_retained_a100_evidence_is_still_at_its_published_digest(
    freeze: dict,
    results: dict,
) -> None:
    import hashlib

    source = STUDY.parent / "a100_graph_launch_v1" / "measurements" / "results.json"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == freeze["retained_a100_evidence"]["results_json_sha256"]
    assert results["fatal_guards"]["G4"]["results_json_sha256"] == digest


def test_the_study_closes_nothing(freeze: dict) -> None:
    assert freeze["closes"] == []
    assert set(freeze["does_not_close"]) == {"COMP-44", "COMP-47", "COMP-48"}
