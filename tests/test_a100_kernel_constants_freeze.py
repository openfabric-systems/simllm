"""Lock the a100_kernel_constants_v1 stage-1 freeze document.

The freeze is the contract the measurement is scored against, so it is locked
the moment it lands and before the harness that produces any number exists.
These checks read the frozen documents only. They never run a kernel, never
reach a GPU and never assert a measured value.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

STUDY = Path(__file__).resolve().parents[1] / "examples" / "a100_kernel_constants_v1"
EXPECTATIONS_JSON = STUDY / "expectations.json"
EXPECTATIONS_MD = STUDY / "expectations.md"

EM_DASH = "—"
TIMER_QUANTUM_NS = 1024
L2_BYTES = 41_943_040
HBM_NAMEPLATE = 2_039_040_000_000
PEAK_1410 = 311_869_440_000_000
PEAK_1275 = 282_009_600_000_000


@pytest.fixture(scope="module")
def freeze() -> dict:
    return json.loads(EXPECTATIONS_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def prose() -> str:
    return EXPECTATIONS_MD.read_text(encoding="utf-8")


def test_the_freeze_documents_are_present() -> None:
    assert EXPECTATIONS_MD.is_file()
    assert EXPECTATIONS_JSON.is_file()


def test_the_freeze_uses_no_em_dash(freeze: dict, prose: str) -> None:
    assert EM_DASH not in prose
    assert EM_DASH not in json.dumps(freeze)


def test_the_substrate_constants_are_the_derived_nameplate_values(freeze: dict) -> None:
    """Every roof in the freeze follows from the recorded device geometry."""

    substrate = freeze["substrate"]
    assert substrate["hbm_nameplate_bytes_per_second"] == HBM_NAMEPLATE
    assert HBM_NAMEPLATE == int(1593e6 * 2 * 5120 / 8)
    assert substrate["peak_flops_1410"] == PEAK_1410 == 108 * 1_410_000_000 * 2048
    assert substrate["peak_flops_1275"] == PEAK_1275 == 108 * 1_275_000_000 * 2048
    assert math.isclose(substrate["clock_ratio_1410_over_1275"], 1410 / 1275, rel_tol=1e-9)
    assert substrate["timer_quantum_ns"] == TIMER_QUANTUM_NS
    assert substrate["l2_bytes"] == L2_BYTES
    assert substrate["memory_clock_mhz"] == 1593
    assert substrate["clock_states_scored"] == [1275, 1410]


def test_the_frozen_knee_predictions_solve_the_frozen_knee_formula(freeze: dict) -> None:
    """M* is not a hand-entered number; it is the roofline crossover in rows."""

    for family in freeze["gemm_families"]:
        n, k = family["n"], family["k"]
        expected = PEAK_1410 * k * n / (n * k * HBM_NAMEPLATE - PEAK_1410 * (k + n))
        assert math.isclose(family["knee_nameplate"], expected, rel_tol=1e-3), family["id"]


def test_the_scored_denominator_equals_the_scored_expectation_count(freeze: dict) -> None:
    scored = freeze["scored_expectations"]
    assert freeze["scored_denominator"] == len(scored)
    ids = [row["id"] for row in scored]
    assert len(set(ids)) == len(ids)
    for row in scored:
        assert re.fullmatch(r"E-\d-\d+", row["id"]), row["id"]
        assert row["risk"] in {"genuine", "informed"}
        assert row["claim"].strip().endswith(".")
        assert len(row["claim"].strip()) > 25, row["id"]


def test_every_scored_expectation_appears_in_the_prose_freeze(
    freeze: dict,
    prose: str,
) -> None:
    for row in freeze["scored_expectations"]:
        assert f"**{row['id']}**" in prose, row["id"]
    for guard in freeze["fatal_guards"]:
        assert f"**{guard['id']}**" in prose, guard["id"]


def test_the_prose_and_machine_freeze_agree_on_the_denominator(prose: str, freeze: dict) -> None:
    assert f"scored behavioral denominator is {freeze['scored_denominator']}" in prose


def test_every_fatal_guard_carries_a_unique_nonblank_claim(freeze: dict) -> None:
    guards = freeze["fatal_guards"]
    assert [guard["id"] for guard in guards] == [f"G{index}" for index in range(1, 13)]
    claims = [guard["claim"] for guard in guards]
    assert len(set(claims)) == len(claims)
    assert all(claim.strip().endswith(".") for claim in claims)


def test_the_guard_ids_are_disjoint_from_the_scored_ids(freeze: dict) -> None:
    """A guard may never be counted in the behavioral denominator."""

    scored = {row["id"] for row in freeze["scored_expectations"]}
    guards = {guard["id"] for guard in freeze["fatal_guards"]}
    assert scored.isdisjoint(guards)
    assert "E-3-6" not in scored


def test_the_clock_policy_records_the_denial_rather_than_a_lock(freeze: dict) -> None:
    policy = freeze["clock_policy"]
    assert policy["lock_attempted"] is True
    assert policy["lock_denied"] is True
    assert "does not have permission to change clocks" in policy["denial_text"]
    assert policy["discovery_job"] == "195960"


def test_the_warm_state_definition_is_stated_not_implied(freeze: dict) -> None:
    definitions = freeze["definitions"]
    for key in (
        "warm_state",
        "rotated_state",
        "clock_stationary_batch",
        "clock_conditioned_constant",
        "R_hbm",
    ):
        assert definitions[key].strip(), key
    assert "back-to-back" in definitions["warm_state"]
    assert "8" in definitions["clock_conditioned_constant"]


def test_the_gemm_grid_and_holdout_factors_do_not_overlap(freeze: dict) -> None:
    """Held-out shapes must sit between grid knots, not on them."""

    grid = set(freeze["gemm_grid_knee_factors"])
    holdout = set(freeze["gemm_holdout_knee_factors"])
    assert grid.isdisjoint(holdout)
    assert min(holdout) > min(grid)
    assert max(holdout) < max(grid)
    for family in freeze["gemm_families"]:
        knee = family["knee_nameplate"]
        grid_m = {round(knee * f) for f in grid} | set(freeze["gemm_grid_fixed_m"])
        holdout_m = {round(knee * f) for f in holdout} - grid_m
        assert holdout_m, family["id"]


def test_the_captured_expert_grid_brackets_the_captured_population(freeze: dict) -> None:
    population = freeze["captured_expert_population"]
    grid = freeze["expert_load_grid"]
    balanced = population["scheduled_tokens"] * population["top_k"] / population["num_experts"]
    assert math.isclose(balanced, population["balanced_rows_per_expert"])
    assert min(grid) == 1
    assert max(grid) == population["scheduled_tokens"]
    assert any(abs(value - balanced) <= 1 for value in grid)


def test_the_freeze_closes_nothing(freeze: dict) -> None:
    assert freeze["closes"] == []
    assert set(freeze["does_not_close"]) == {"COMP-1", "COMP-5"}
    assert "It does not close COMP-1." in prose_text()
    assert "It does not close COMP-5." in prose_text()


def prose_text() -> str:
    return EXPECTATIONS_MD.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The rerun refreeze. It supersedes two guards and retains every scored claim.
# ---------------------------------------------------------------------------

REFREEZE_JSON = STUDY / "refreeze_expectations.json"
REFREEZE_MD = STUDY / "refreeze_expectations.md"


@pytest.fixture(scope="module")
def refreeze() -> dict:
    return json.loads(REFREEZE_JSON.read_text(encoding="utf-8"))


def test_the_refreeze_documents_are_present() -> None:
    assert REFREEZE_JSON.is_file()
    assert REFREEZE_MD.is_file()
    assert EM_DASH not in REFREEZE_MD.read_text(encoding="utf-8")


def test_the_refreeze_supersedes_only_the_two_instrumentation_guards(
    refreeze: dict,
    freeze: dict,
) -> None:
    superseded = set(refreeze["supersedes_guards"])
    retained = set(refreeze["retains_guards"])
    original = {guard["id"] for guard in freeze["fatal_guards"]}
    assert superseded == {"G9", "G11"}
    assert superseded | retained == original
    assert superseded.isdisjoint(retained)


def test_the_refreeze_changes_no_scored_claim(refreeze: dict, freeze: dict) -> None:
    """The rerun re-evaluates a pre-registered claim set, it does not refreeze it."""

    assert refreeze["retains_scored_expectations_unchanged"] is True
    assert refreeze["scored_denominator"] == freeze["scored_denominator"]


def test_exactly_one_refreeze_guard_is_declared_survivable(refreeze: dict) -> None:
    survivable = [g["id"] for g in refreeze["fatal_guards"] if g["survivable"]]
    assert survivable == ["G13"]
    remainder = refreeze["survivable_guard_remainder"]
    assert remainder["guard"] == "G13"
    assert remainder["on_failure"].strip()
    assert "never as a pass" in remainder["remains_interpretable"]


def test_the_instrumentation_band_brackets_the_run_one_samples(refreeze: dict) -> None:
    """The G11R band is derived from measured run-1 inflation, not invented."""

    low, high = refreeze["instrumentation_cost_band_us"]
    samples = refreeze["void_run"]["instrumentation_inflation_samples_ns"].values()
    assert low > 0.0
    for value in samples:
        assert low * 1000.0 < value < high * 1000.0


def test_the_void_run_is_recorded_as_void_with_its_violated_guards(refreeze: dict) -> None:
    void = refreeze["void_run"]
    assert void["verdict"] == "void"
    assert set(void["violated_guards"]) == {"G9", "G10", "G11"}
    assert void["job"] == "195964"


def test_every_refreeze_guard_and_repair_appears_in_the_prose(refreeze: dict) -> None:
    prose = REFREEZE_MD.read_text(encoding="utf-8")
    for guard in refreeze["fatal_guards"]:
        assert f"**{guard['id']}**" in prose, guard["id"]
    for repair in refreeze["repairs"]:
        assert f"**{repair['id']}**" in prose, repair["id"]
