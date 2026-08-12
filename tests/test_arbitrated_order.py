"""Regressions for the registered arbitrated-order forms (CORE-49, CORE-10).

The orders and step latencies pinned here were frozen in
`examples/arbitrated_order_v1/expectations.md` before the implementation
existed. The 132, 328 and 7 cycle controls are the measured rows the task-mix
and mixed-makespan studies already published; they belong to the synthetic
1 GHz mechanism fixture and are never a silicon claim.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

STUDY_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "arbitrated_order_v1"
    / "run_study.py"
)
SPEC = importlib.util.spec_from_file_location("arbitrated_order_v1_study", STUDY_PATH)
assert SPEC is not None
assert SPEC.loader is not None
study = importlib.util.module_from_spec(SPEC)
# Register before executing: the study defines dataclasses, and dataclasses
# resolves annotations through ``sys.modules`` while the class body runs.
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


def test_check_only_validates_the_frozen_registry(capsys):
    study.check_only()
    assert "check-only validated" in capsys.readouterr().out


def test_isolated_controls_match_the_published_rows():
    assert study.isolated_controls() == study.FROZEN_ISOLATED_CYCLES


@pytest.mark.parametrize("instance", sorted(study.FROZEN_FAMILY_A))
def test_family_a_arbitrated_order_reaches_the_compute_service(instance):
    spec, labels, budget, lanes, order, jct = study.FROZEN_FAMILY_A[instance]
    observed = study.run_live_case(
        study.LiveCase(
            f"f2-{instance}",
            study.FIXTURE_F2_ORDER,
            spec,
            dict(labels),
            budget,
            lanes,
        )
    )
    assert study._orders(observed) == [order] * study.LIVE_STEP_COUNT
    assert study._latencies(observed) == [jct] * study.LIVE_STEP_COUNT


@pytest.mark.parametrize("instance", sorted(study.FROZEN_FAMILY_B))
def test_family_b_class_aware_policies_order_by_their_own_contract(instance):
    spec, labels, orders = study.FROZEN_FAMILY_B[instance]
    observed = study.run_live_case(
        study.LiveCase(f"f3-{instance}", study.FIXTURE_F3_ORDER, spec, dict(labels))
    )
    assert study._orders(observed) == list(orders)


def test_identity_baseline_order_and_step_latency_are_pinned():
    for (budget, lanes), jct in study.FROZEN_F2_IDENTITY_JCT_PS.items():
        observed = study.run_live_case(
            study.LiveCase(
                f"f2-identity-{budget}{lanes}",
                study.FIXTURE_F2_ORDER,
                None,
                {},
                budget,
                lanes,
            )
        )
        assert study._orders(observed) == [
            study.FROZEN_F2_IDENTITY_ORDER
        ] * study.LIVE_STEP_COUNT
        assert study._latencies(observed) == [jct] * study.LIVE_STEP_COUNT


@pytest.mark.parametrize(
    "spec",
    [
        None,
        ("identity", {}),
        ("strict", {"class_aware": False}),
        ("wrr", {"weights": {1: 2, 2: 1}, "class_aware": False}),
    ],
)
@pytest.mark.parametrize(
    "labels",
    [{}, {"memory": 2, "network": 1, "compute": 1}, {"memory": 1, "network": 2, "compute": 3}],
)
def test_every_identity_setting_is_behaviorally_identical_under_label_permutation(
    spec,
    labels,
):
    baseline = study.run_live_case(
        study.LiveCase("case", study.FIXTURE_F3_ORDER, None, {})
    )
    observed = study.run_live_case(
        study.LiveCase("case", study.FIXTURE_F3_ORDER, spec, dict(labels))
    )
    assert study._orders(observed) == [
        study.FROZEN_F3_IDENTITY_ORDER
    ] * study.LIVE_STEP_COUNT
    assert study._latencies(observed) == study._latencies(baseline)
    assert [step["canonical"] for step in observed["steps"]] == [
        step["canonical"] for step in baseline["steps"]
    ]
    assert [step["total_class_service_bytes"] for step in observed["steps"]] == [
        step["total_class_service_bytes"] for step in baseline["steps"]
    ]


@pytest.mark.parametrize(
    "spec,labels",
    [
        (("identity", {}), {}),
        (("strict", {}), {"memory": 2, "network": 1}),
    ],
)
def test_scalar_compatibility_path_stays_order_invariant(spec, labels):
    observed = study.run_live_case(
        study.LiveCase(
            "f2-scalar",
            study.FIXTURE_F2_ORDER,
            spec,
            dict(labels),
            nominal_ps=dict(study.FROZEN_SCALAR_NOMINAL_PS),
        )
    )
    assert study._latencies(observed) == [
        study.FROZEN_SCALAR_JCT_PS
    ] * study.LIVE_STEP_COUNT


@pytest.mark.parametrize(
    "spec,labels",
    [
        (("identity", {}), {}),
        (("strict", {}), {"memory": 2, "network": 1}),
    ],
)
def test_a_dependency_is_mandatory_ordering_that_arbitration_cannot_move(spec, labels):
    observed = study.run_live_case(
        study.LiveCase(
            "f2-dependent",
            study.FIXTURE_F2_ORDER,
            spec,
            dict(labels),
            depends={"network": ("memory",)},
        )
    )
    for step in observed["steps"]:
        assert step["service_orders"] == [["memory"], ["network"]]
    assert study._latencies(observed) == [
        study.FROZEN_DEPENDENT_JCT_PS
    ] * study.LIVE_STEP_COUNT


def test_component_replay_reproduces_the_registered_issue_term():
    floor = max(
        study.FROZEN_ISOLATED_CYCLES["memory"],
        study.FROZEN_ISOLATED_CYCLES["network"],
    )
    memory_first = study.component_replay(("memory", "network"), 4, 4)
    network_first = study.component_replay(("network", "memory"), 4, 4)
    assert memory_first["duration_cycles"] == floor + 1
    assert network_first["duration_cycles"] == floor
    widened = study.component_replay(("memory", "network"), 8, 8)
    assert widened["duration_cycles"] == floor
