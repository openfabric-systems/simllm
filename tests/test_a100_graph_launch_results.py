"""Lock the a100_graph_launch_v1 result against its freeze.

These checks read the committed artifacts only. They never run the study, need
no GPU, and assert exactly the relations the freeze named.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

STUDY = Path(__file__).resolve().parents[1] / "examples" / "a100_graph_launch_v1"
RESULTS_JSON = STUDY / "measurements" / "results.json"
RESULTS_MD = STUDY / "RESULTS.md"
FREEZE_JSON = STUDY / "expectations.json"

EM_DASH = "—"


@pytest.fixture(scope="module")
def results() -> dict:
    return json.loads(RESULTS_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def freeze() -> dict:
    return json.loads(FREEZE_JSON.read_text(encoding="utf-8"))


def test_the_report_uses_no_em_dash() -> None:
    assert EM_DASH not in RESULTS_MD.read_text(encoding="utf-8")


def test_every_published_guard_carries_its_frozen_claim_verbatim(
    results: dict,
    freeze: dict,
) -> None:
    """The one assertion that stops a weaker check being published as a guard."""

    frozen = {guard["id"]: guard["claim"] for guard in freeze["fatal_guards"]}
    assert [guard["id"] for guard in results["fatal_guards"]] == list(frozen)
    for guard in results["fatal_guards"]:
        assert guard["claim"] == frozen[guard["id"]], guard["id"]
        assert guard["evaluated"].strip(), guard["id"]


def test_every_published_expectation_carries_its_frozen_claim_verbatim(
    results: dict,
    freeze: dict,
) -> None:
    frozen = {row["id"]: (row["claim"], row["risk"], row["group"]) for row in freeze["scored_expectations"]}
    assert len(results["scored"]) == freeze["scored_denominator"] == 15
    for row in results["scored"]:
        claim, risk, group = frozen[row["id"]]
        assert row["claim"] == claim, row["id"]
        assert row["risk"] == risk, row["id"]
        assert row["group"] == group, row["id"]
        assert row["evaluated"].strip(), row["id"]
        assert row["status"] in {"pass", "fail", "unevaluated"}


def test_the_scored_counts_are_reported_as_three_states(results: dict) -> None:
    total = results["scored_total"]
    assert results["scored_passed"] + results["scored_failed"] + results[
        "scored_unevaluated"
    ] == total == 15


def test_the_verdict_follows_from_the_guards(results: dict) -> None:
    failed = [guard["id"] for guard in results["fatal_guards"] if not guard["held"]]
    assert results["voiding_guards"] == failed
    assert results["verdict"] == ("interpretable" if not failed else "void")


def test_the_falsifier_row_is_published_with_its_measured_ratio(results: dict) -> None:
    """The ruling's falsifier is reported plainly, whichever way it went."""

    row = next(r for r in results["scored"] if r["id"] == "F1")
    assert row["group"] == "falsifier"
    assert set(row["detail"]) == {"g1", "g2", "g4"}
    for tag, detail in row["detail"].items():
        assert detail["s_graph_s"] > 0, tag
        assert detail["s_eager_s"] > 0, tag
        assert detail["ratio"] > 0, tag


def test_the_two_timers_never_contain_each_other(results: dict) -> None:
    """GG4 as frozen, rechecked on the retained cells."""

    guard = next(g for g in results["fatal_guards"] if g["id"] == "GG4")
    assert guard["held"]
    for cell in results["cells"]:
        assert cell["host_before_sync"], cell["tag"]


def test_the_reserved_device_gap_is_recorded_and_wired_to_nothing(results: dict) -> None:
    """The seed constant is published with provenance and consumed by no code.

    The source side of this lock is AST-based rather than a substring scan: it
    walks every ``*.py`` module under ``simllm`` and looks for all nine
    published picosecond values as integer literals, and for any identifier or
    attribute a real wiring would use. A substring scan over the raw text would
    be fooled by a comment and would fire on one inside a docstring. The scan
    covers Python sources under ``simllm`` only, which RESULTS.md states.
    """

    reserved = results["reserved_device_gap"]
    assert reserved["wired_to"] == "nothing"
    graph_ps = reserved["in_graph_null_kernel_period_ps"]
    eager_ps = reserved["eager_null_kernel_period_ps"]
    assert graph_ps > 0
    assert eager_ps > 0
    assert graph_ps < eager_ps
    assert reserved["provenance"].startswith("examples/a100_graph_launch_v1")

    forbidden_values = {graph_ps, eager_ps}
    for profile in results["output_profiles"].values():
        for field in (
            "point_ps",
            "empirical_min_ps",
            "empirical_max_ps",
            "fixed_per_replay_ps",
        ):
            value = profile.get(field)
            if isinstance(value, int):
                forbidden_values.add(value)
    # Nine integers: the two reserved-gap constants and the seven the two
    # withheld host profiles carry.
    assert len(forbidden_values) == 9
    forbidden_names = (
        "null_kernel_period",
        "in_graph_gap",
        "graph_launch",
        "a100_epyc",
        "a100epyc",
        "epyc_eager",
        "epyc_cuda_graph",
        "graph_node_cost",
        "eager_host_cost",
        "front_end_gap",
    )
    package = Path(__file__).resolve().parents[1] / "simllm"
    scanned = 0
    for path in sorted(package.rglob("*.py")):
        scanned += 1
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                assert node.value not in forbidden_values, f"{path}:{node.lineno}"
            if isinstance(node, ast.Name):
                assert not any(part in node.id for part in forbidden_names), path
            if isinstance(node, ast.Attribute):
                assert not any(part in node.attr for part in forbidden_names), path
    # The scan scope is asserted rather than assumed, and RESULTS states it.
    assert scanned > 20
    report = (STUDY / "RESULTS.md").read_text(encoding="utf-8")
    assert "It does not cover the C++ under `simllm/backends/rnic`" in report


def test_the_registry_entry_carries_the_void_and_not_a_score_disclaimer() -> None:
    """COMP-47 must not quote 14 of 15 as if it were a score.

    Stage 1's disclaimer is locked as a string in its own report; this locks
    stage 2's in both places it appears, the report and the registry entry that
    cites it, so a later edit cannot quietly turn a void run into a fraction.
    """

    report = RESULTS_MD.read_text(encoding="utf-8")
    assert "The reviewed study state is `VOID`" in report
    assert "**That 14 is not a\nscore.**" in report

    registry = (
        Path(__file__).resolve().parents[1] / "docs" / "modules" / "compute.md"
    ).read_text(encoding="utf-8")
    entry_start = registry.index("- COMP-47 (Precision; P1; L)")
    entry = registry[entry_start : registry.index("\n- COMP-4", entry_start + 10)]
    assert "reviewed `VOID`" in entry
    assert "that 14 is not a score" in entry
    assert "uninterpretable" in entry
    assert "14 of its 15" not in entry


def test_the_contract_clause_points_at_the_finding_and_the_pending_ruling() -> None:
    """The refuted clause must be discoverable from the contract text."""

    registry = (
        Path(__file__).resolve().parents[1] / "docs" / "modules" / "compute.md"
    ).read_text(encoding="utf-8")
    clause = registry[registry.index("**CUDA-graph launch and eager launch differ") :][:2400]
    assert "examples/a100_graph_launch_v1/RESULTS.md" in clause
    assert "pending" in clause
    assert "COMP-48" in clause
    # The standing ruling text itself is untouched: the sentence still stands.
    assert (
        "The launch class never reaches kernel service time." in clause
    )


def test_the_output_profiles_carry_their_definition_and_range(results: dict) -> None:
    profiles = results["output_profiles"]
    assert set(profiles) == {"a100-epyc-eager-host", "a100-epyc-cuda-graph"}
    for name, profile in profiles.items():
        assert profile["point_ps"] > 0, name
        assert profile["empirical_min_ps"] > 0, name
        assert profile["empirical_max_ps"] >= profile["empirical_min_ps"], name
        assert profile["definition"].strip(), name
    assert profiles["a100-epyc-cuda-graph"]["fixed_per_replay_ps"] > 0
