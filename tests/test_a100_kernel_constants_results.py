"""Lock the a100_kernel_constants_v1 result against its freeze and refreeze.

These checks read the committed artifacts only. They never run the study, need
no GPU, and assert exactly the relations the two expectations documents named,
so a later change that silently moves a published number fails here.

The study is VOID. These locks therefore protect the void verdict, the guard
identities and the retained evidence; they never assert that a scored fraction
means anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

STUDY = Path(__file__).resolve().parents[1] / "examples" / "a100_kernel_constants_v1"
RESULTS_JSON = STUDY / "measurements" / "results.json"
RESULTS_MD = STUDY / "RESULTS.md"
FREEZE_JSON = STUDY / "expectations.json"
REFREEZE_JSON = STUDY / "refreeze_expectations.json"

EM_DASH = "—"


@pytest.fixture(scope="module")
def results() -> dict:
    return json.loads(RESULTS_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def freeze() -> dict:
    return json.loads(FREEZE_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def refreeze() -> dict:
    return json.loads(REFREEZE_JSON.read_text(encoding="utf-8"))


def test_the_report_uses_no_em_dash() -> None:
    assert EM_DASH not in RESULTS_MD.read_text(encoding="utf-8")


def test_the_run_is_void_and_says_so(results: dict) -> None:
    assert results["verdict"] == "void"
    assert results["voiding_guards"] == ["G9R", "G10", "G11R"]
    assert "The reviewed study state is `VOID`" in RESULTS_MD.read_text(encoding="utf-8")


def test_every_published_guard_carries_its_frozen_claim_verbatim(
    results: dict,
    freeze: dict,
    refreeze: dict,
) -> None:
    """The one assertion that stops a weaker check being published as a guard.

    A result guard identity may only appear beside the exact claim string one of
    the two expectations documents wrote for it. What the scorer actually
    checked belongs in the separate ``evaluated`` field, which must be present
    and nonblank.
    """

    frozen = {guard["id"]: guard["claim"] for guard in freeze["fatal_guards"]}
    frozen.update({guard["id"]: guard["claim"] for guard in refreeze["fatal_guards"]})
    for guard in results["fatal_guards"]:
        assert guard["claim"] == frozen[guard["id"]], guard["id"]
        assert guard["evaluated"].strip(), guard["id"]


def test_every_published_expectation_carries_its_frozen_claim_verbatim(
    results: dict,
    freeze: dict,
) -> None:
    frozen = {row["id"]: (row["claim"], row["risk"]) for row in freeze["scored_expectations"]}
    assert len(results["scored"]) == freeze["scored_denominator"]
    for row in results["scored"]:
        claim, risk = frozen[row["id"]]
        assert row["claim"] == claim, row["id"]
        assert row["risk"] == risk, row["id"]
        assert row["evaluated"].strip(), row["id"]
        assert row["status"] in {"pass", "fail", "unevaluated"}


def test_the_guard_set_is_the_refreeze_guard_set(
    results: dict,
    freeze: dict,
    refreeze: dict,
) -> None:
    """The rerun evaluates G9R and G11R, never the superseded G9 and G11."""

    published = [guard["id"] for guard in results["fatal_guards"]]
    superseded = set(refreeze["supersedes_guards"])
    assert superseded.isdisjoint(published)
    expected = set(refreeze["retains_guards"]) | {
        guard["id"] for guard in refreeze["fatal_guards"]
    }
    assert set(published) == expected
    assert set(freeze["fatal_guards"][0].keys()) >= {"id", "claim"}


def test_only_the_declared_survivable_guard_survives_its_failure(
    results: dict,
    refreeze: dict,
) -> None:
    survivable = {g["id"] for g in refreeze["fatal_guards"] if g["survivable"]}
    assert results["survivable_guards"] == sorted(survivable)
    failed = {guard["id"] for guard in results["fatal_guards"] if not guard["held"]}
    assert set(results["voiding_guards"]) == failed - survivable
    assert "G13" in failed
    assert "G13" not in results["voiding_guards"]


def test_the_survived_guard_excluded_its_cell_from_scoring(results: dict) -> None:
    excluded = results["host_issue_bound_cells"]
    assert len(excluded) == 1
    row = excluded[0]
    assert row["id"] == "moe_expert_down_m54"
    assert row["arm"] == "base"
    assert row["host_ratio_max"] > 0.8


def test_the_scored_counts_are_reported_as_three_states_never_as_one_fraction(
    results: dict,
) -> None:
    total = results["scored_total"]
    assert results["scored_passed"] + results["scored_failed"] + results[
        "scored_unevaluated"
    ] == total
    assert total == 31
    assert "That 16 is not a\nscore." in RESULTS_MD.read_text(encoding="utf-8")


def test_the_measured_roof_is_below_nameplate_and_above_the_prior_study(
    results: dict,
    freeze: dict,
) -> None:
    roof = results["measured_hbm_roof_bytes_per_second"]
    assert roof < freeze["substrate"]["hbm_nameplate_bytes_per_second"]
    assert 1.80e12 < roof < 1.83e12
    assert 0.88 < results["measured_hbm_roof_fraction_of_nameplate"] < 0.90


def test_the_instrumentation_cost_is_measured_and_inside_its_frozen_band(
    results: dict,
    refreeze: dict,
) -> None:
    low, high = (value * 1e-6 for value in refreeze["instrumentation_cost_band_us"])
    cost = results["instrumentation_per_boundary_cost_s"]
    event_only = results["instrumentation_event_only_period_s"]
    assert low <= cost <= high
    assert low <= event_only <= high
    # Two independent methods for the same quantity must agree closely.
    assert abs(cost - event_only) / event_only < 0.05


def test_no_cell_beats_its_compulsory_traffic_floor(results: dict) -> None:
    """G5 as frozen, rechecked against the retained cells rather than trusted."""

    for arm in ("boosted", "base"):
        for cell in results["cells"][arm]:
            if cell["void_for_scoring"]:
                continue
            assert cell["constant_s"] >= cell["physical_floor_s"], cell["id"]


def test_memory_limited_constants_do_not_move_with_the_sm_clock(results: dict) -> None:
    """Finding: the clock moves compute constants and leaves memory ones alone."""

    boosted = {cell["id"]: cell for cell in results["cells"]["boosted"]}
    ratios = {}
    for cell in results["cells"]["base"]:
        peer = boosted.get(cell["id"])
        if cell["void_for_scoring"] or peer is None or peer["void_for_scoring"]:
            continue
        if cell["scored_state"] != 1275 or peer["scored_state"] != 1410:
            continue
        ratios.setdefault(peer["regime"], []).append(
            cell["constant_s"] / peer["constant_s"]
        )
    assert ratios["compute"], "no compute-regime pair survived"
    assert ratios["memory"], "no memory-regime pair survived"
    assert min(ratios["compute"]) > 1.05
    assert min(ratios["memory"]) < 1.05


def test_no_calibrated_profile_table_is_published(results: dict) -> None:
    """A void run must not ship a table a provider could load."""

    assert not (STUDY / "profile_table_a100.json").exists()
    assert not list(STUDY.glob("**/*profile_table*.json"))
    assert results["verdict"] == "void"
