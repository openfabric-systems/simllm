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
    walks every module under ``simllm`` and looks for the reserved picosecond
    values as integer literals and for any identifier that names the quantity.
    A substring scan over the raw text would be fooled by a comment and would
    fire on one inside a docstring.
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
    forbidden_names = ("null_kernel_period", "in_graph_gap", "graph_launch")
    package = Path(__file__).resolve().parents[1] / "simllm"
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                assert node.value not in forbidden_values, f"{path}:{node.lineno}"
            if isinstance(node, ast.Name):
                assert not any(part in node.id for part in forbidden_names), path
            if isinstance(node, ast.Attribute):
                assert not any(part in node.attr for part in forbidden_names), path


def test_the_output_profiles_carry_their_definition_and_range(results: dict) -> None:
    profiles = results["output_profiles"]
    assert set(profiles) == {"a100-epyc-eager-host", "a100-epyc-cuda-graph"}
    for name, profile in profiles.items():
        assert profile["point_ps"] > 0, name
        assert profile["empirical_min_ps"] > 0, name
        assert profile["empirical_max_ps"] >= profile["empirical_min_ps"], name
        assert profile["definition"].strip(), name
    assert profiles["a100-epyc-cuda-graph"]["fixed_per_replay_ps"] > 0
