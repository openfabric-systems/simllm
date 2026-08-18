"""Lock the a100_graph_launch_v1 stage-2 freeze document.

The freeze is the contract the measurement is scored against, so it is locked
the moment it lands and before the harness that produces any number exists.
These checks read the frozen documents only. They never run a kernel and never
assert a measured value.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

STUDY = Path(__file__).resolve().parents[1] / "examples" / "a100_graph_launch_v1"
EXPECTATIONS_JSON = STUDY / "expectations.json"
EXPECTATIONS_MD = STUDY / "expectations.md"
STAGE1 = Path(__file__).resolve().parents[1] / "examples" / "a100_kernel_constants_v1"

EM_DASH = "—"
TURING_EAGER_PS = 2_364_255
TURING_GRAPH_PS = 809_306


@pytest.fixture(scope="module")
def freeze() -> dict:
    return json.loads(EXPECTATIONS_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def prose() -> str:
    return EXPECTATIONS_MD.read_text(encoding="utf-8")


def test_the_freeze_documents_are_present_and_use_no_em_dash(prose: str) -> None:
    assert EXPECTATIONS_JSON.is_file()
    assert EM_DASH not in prose


def test_the_scored_denominator_equals_the_scored_expectation_count(freeze: dict) -> None:
    scored = freeze["scored_expectations"]
    assert freeze["scored_denominator"] == len(scored) == 15
    ids = [row["id"] for row in scored]
    assert len(set(ids)) == len(ids)
    for row in scored:
        assert re.fullmatch(r"[FHDM]\d", row["id"]), row["id"]
        assert row["risk"] in {"genuine", "informed"}
        assert row["claim"].strip().endswith(".")


def test_every_expectation_and_guard_appears_in_the_prose(freeze: dict, prose: str) -> None:
    for row in freeze["scored_expectations"]:
        assert f"**{row['id']}**" in prose, row["id"]
    for guard in freeze["fatal_guards"]:
        assert f"**{guard['id']}**" in prose, guard["id"]


def test_the_guard_ids_are_disjoint_from_the_scored_ids(freeze: dict) -> None:
    scored = {row["id"] for row in freeze["scored_expectations"]}
    guards = {guard["id"] for guard in freeze["fatal_guards"]}
    assert scored.isdisjoint(guards)
    assert [guard["id"] for guard in freeze["fatal_guards"]] == [
        f"GG{index}" for index in range(1, 9)
    ]


def test_the_falsifier_group_is_present_and_is_a_real_falsifier(freeze: dict, prose: str) -> None:
    """F1 must be able to fail, and the freeze must say what happens then."""

    falsifier = [row for row in freeze["scored_expectations"] if row["group"] == "falsifier"]
    assert {row["id"] for row in falsifier} == {"F1", "F2", "F3"}
    assert all(row["risk"] == "genuine" for row in falsifier)
    assert "If F1 fails, the constant is launch-mode conditioned" in prose
    assert "will not be folded" in prose


def test_the_stage_one_inputs_are_declared_void_and_used_only_for_bands(
    freeze: dict,
    prose: str,
) -> None:
    """A void run's numbers may set a band; they may not anchor a claim."""

    inputs = freeze["stage1_inputs"]
    assert inputs["status"] == "void"
    assert "never an anchor" in inputs["role"]
    assert "is reviewed `VOID`" in prose
    assert (STAGE1 / "RESULTS.md").is_file()


def test_the_band_widths_are_derived_from_the_stage_one_dispersion(freeze: dict) -> None:
    """The 5 percent falsifier band must exceed the measured dispersion."""

    worst_cv = freeze["stage1_inputs"]["worst_batch_cv"]
    assert 0.05 > worst_cv > 0.03
    assert freeze["fatal_guards"][6]["id"] == "GG7"
    assert "4 percent" in freeze["fatal_guards"][6]["claim"]


def test_the_turing_comparison_points_are_the_accepted_comp2_constants(freeze: dict) -> None:
    points = freeze["comparison_points"]
    assert points["turing_eager_host_ps"] == TURING_EAGER_PS
    assert points["turing_graph_node_ps" if False else "turing_cuda_graph_node_ps"] == (
        TURING_GRAPH_PS
    )


def test_the_output_profiles_target_a100_and_close_no_task(freeze: dict, prose: str) -> None:
    names = {profile["name"] for profile in freeze["output_profiles"]}
    assert names == {"a100-epyc-eager-host", "a100-epyc-cuda-graph"}
    classes = {profile["launch_class"] for profile in freeze["output_profiles"]}
    assert classes == {"eager-host-bound", "cuda-graph-node"}
    assert freeze["closes"] == []
    assert set(freeze["does_not_close"]) == {"COMP-1", "COMP-2"}
    assert "This study closes no COMP-2 clause." in prose


def test_the_graph_profile_declares_its_reference_chain_length(freeze: dict, prose: str) -> None:
    """A per-launch graph constant is K-scoped and the freeze must say so first."""

    assert freeze["substrate"]["reference_chain_length"] == 64
    assert "K_ref" in prose
    assert "K-scoped sensitivity constant" in prose


def test_the_kernel_set_is_drawn_from_the_stage_one_set(freeze: dict) -> None:
    tags = {kernel["tag"] for kernel in freeze["kernels"]}
    assert tags == {"nop", "g1", "g2", "g4", "mix"}
    cycle = next(k for k in freeze["kernels"] if k["tag"] == "mix")["cycle"]
    assert set(cycle) <= tags - {"mix"}


def test_the_host_and_device_timers_are_separated_by_construction(freeze: dict) -> None:
    definition = freeze["definitions"]["host_submission_cost"]
    assert "before any synchronization" in definition
    assert "containing none" in definition
    guard = next(g for g in freeze["fatal_guards"] if g["id"] == "GG4")
    assert "no synchronization call and no event record" in guard["claim"]
