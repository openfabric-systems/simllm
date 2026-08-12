"""Regressions for the registered mixed-makespan forms (COMP-12).

The numbers pinned here are the measured rows of the task-mix study, which
this repository already published in `examples/gpu_task_mix/RESULTS.md`. They
belong to the synthetic 1 GHz mechanism fixture and are never a silicon claim.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from simllm.compute import (
    GpuTask,
    GpuTaskKind,
    MixedMakespanRegime,
    SmSchedulerModel,
    decompose_mixed_makespan,
)

STUDY_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "mixed_makespan_v1"
    / "run_study.py"
)
SPEC = importlib.util.spec_from_file_location("mixed_makespan_v1_study", STUDY_PATH)
assert SPEC is not None
assert SPEC.loader is not None
study = importlib.util.module_from_spec(SPEC)
# Register before executing: the study defines dataclasses, and dataclasses
# resolves annotations through ``sys.modules`` while the class body runs.
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)

HALF_SHARED = study.SHARED_PER_SM // 2


def _model(*, scheduler: int = 4, lanes: int = 4) -> SmSchedulerModel:
    return SmSchedulerModel(
        study.architecture(
            scheduler_count_per_sm=scheduler,
            load_store_issue_width_per_sm=lanes,
        )
    )


def _tasks(*ordered):
    return tuple(
        GpuTask(task_id=task_id, kind=kind, launch=launch)
        for task_id, kind, launch in ordered
    )


def test_isolated_controls_match_the_published_task_mix_rows():
    model = _model()
    assert model.estimate(study.memory_launch()).duration_cycles == 132
    assert model.estimate(study.egress_launch()).duration_cycles == 328
    assert (
        model.estimate(study.compute_launch(shared_bytes=HALF_SHARED)).duration_cycles
        == 14
    )
    assert (
        model.estimate(study.memory_launch(shared_bytes=HALF_SHARED)).duration_cycles
        == 229
    )
    assert model.estimate(study.compute_launch()).duration_cycles == 7


@pytest.mark.parametrize(
    "order,scheduler,lanes,delay,mixed",
    [
        (("memory", "network"), 4, 4, 1, 329),
        (("memory", "network"), 8, 4, 1, 329),
        (("memory", "network"), 4, 8, 1, 329),
        (("memory", "network"), 8, 8, 0, 328),
        (("network", "memory"), 4, 4, 0, 328),
    ],
)
def test_g1_issue_delay_follows_the_submitted_order(
    order, scheduler, lanes, delay, mixed
):
    model = _model(scheduler=scheduler, lanes=lanes)
    memory = study.memory_launch()
    egress = study.egress_launch()
    by_name = {
        "memory": ("memory", GpuTaskKind.MEMORY, memory),
        "network": ("network", GpuTaskKind.NETWORK, egress),
    }
    estimate = model.estimate_concurrent(_tasks(*(by_name[name] for name in order)))
    form = decompose_mixed_makespan(
        estimate,
        {
            "memory": model.estimate(memory).duration_cycles,
            "network": model.estimate(egress).duration_cycles,
        },
    )
    assert form.regime is MixedMakespanRegime.ISSUE_ORDER
    assert form.residency_gated_task_id is None
    assert form.residency_serialized_cycles is None
    assert form.concurrent_floor_cycles == 328
    assert form.serialized_ceiling_cycles == 460
    assert form.within_physical_interval
    assert form.mixed_cycles == mixed
    assert form.issue_delay_cycles == delay


def test_g2_half_shared_tasks_serialize_on_residency():
    model = _model()
    compute = study.compute_launch(shared_bytes=HALF_SHARED)
    memory = study.memory_launch(shared_bytes=HALF_SHARED)
    estimate = model.estimate_concurrent(
        _tasks(
            ("compute", GpuTaskKind.COMPUTE, compute),
            ("memory", GpuTaskKind.MEMORY, memory),
        )
    )
    form = decompose_mixed_makespan(estimate, {"compute": 14, "memory": 229})

    assert form.regime is MixedMakespanRegime.RESIDENCY_SERIALIZED
    assert form.residency_gated_task_id == "memory"
    assert form.admitted_cycles == (0, 14)
    assert form.residency_delay_cycles == (0, 14)
    assert form.mixed_cycles == 243
    assert form.residency_serialized_cycles == 243
    assert form.concurrent_floor_cycles == 229
    assert form.serialized_ceiling_cycles == 243
    assert form.within_physical_interval


def test_g2_zero_shared_control_restores_backfill():
    model = _model()
    compute = study.compute_launch()
    memory = study.memory_launch()
    estimate = model.estimate_concurrent(
        _tasks(
            ("compute", GpuTaskKind.COMPUTE, compute),
            ("memory", GpuTaskKind.MEMORY, memory),
        )
    )
    form = decompose_mixed_makespan(estimate, {"compute": 7, "memory": 132})

    assert form.regime is MixedMakespanRegime.ISSUE_ORDER
    assert form.mixed_cycles == 133
    assert form.issue_delay_cycles == 1
    assert form.within_physical_interval


def test_task_kind_relabelling_moves_labels_only():
    model = _model()
    memory = study.memory_launch()
    egress = study.egress_launch()
    labelled = model.estimate_concurrent(
        _tasks(
            ("memory", GpuTaskKind.MEMORY, memory),
            ("network", GpuTaskKind.NETWORK, egress),
        )
    )
    relabelled = model.estimate_concurrent(
        _tasks(
            ("memory", GpuTaskKind.NETWORK, memory),
            ("network", GpuTaskKind.COMPUTE, egress),
        )
    )
    assert labelled.duration_cycles == relabelled.duration_cycles
    assert [task.completion_cycle for task in labelled.tasks] == [
        task.completion_cycle for task in relabelled.tasks
    ]
    assert [task.kind for task in labelled.tasks] != [
        task.kind for task in relabelled.tasks
    ]


def test_decomposition_requires_an_isolated_control_for_every_task():
    model = _model()
    estimate = model.estimate_concurrent(
        _tasks(
            ("memory", GpuTaskKind.MEMORY, study.memory_launch()),
            ("network", GpuTaskKind.NETWORK, study.egress_launch()),
        )
    )
    with pytest.raises(KeyError):
        decompose_mixed_makespan(estimate, {"memory": 132})
    with pytest.raises(ValueError):
        decompose_mixed_makespan(estimate, {"memory": 132, "network": 0})
    with pytest.raises(TypeError):
        decompose_mixed_makespan(estimate, [132, 328])


@pytest.mark.parametrize(
    "case_id,order,scheduler,lanes,expected_ps",
    [
        ("live-memory-first-4-4", ("memory", "network"), 4, 4, 329_000),
        ("live-network-first-4-4", ("network", "memory"), 4, 4, 328_000),
        ("live-memory-first-8-8", ("memory", "network"), 8, 8, 328_000),
        ("live-network-first-8-8", ("network", "memory"), 8, 8, 328_000),
    ],
)
def test_live_core4_projects_the_issue_order_term(
    case_id, order, scheduler, lanes, expected_ps
):
    launches = {"memory": study.memory_launch(), "network": study.egress_launch()}
    observed = study.run_live_case(
        study.LiveCase(
            case_id,
            order,
            launches,
            scheduler_count_per_sm=scheduler,
            load_store_issue_width_per_sm=lanes,
        )
    )
    for step in observed["steps"]:
        assert step["step_latency_ps"] == expected_ps
        assert step["ttft_ps"] == expected_ps
        assert step["queued_matches_eligibility"]
        assert step["started_matches_grant"]
        assert step["completed_events_match_operations"]
        if step["step_index"] > 0:
            assert step["tpot_ps"] == expected_ps


def test_live_core4_projects_the_residency_term():
    half = study.run_live_case(
        study.LiveCase(
            "live-half-shared",
            ("compute", "memory"),
            {
                "compute": study.compute_launch(shared_bytes=HALF_SHARED),
                "memory": study.memory_launch(shared_bytes=HALF_SHARED),
            },
        )
    )
    zero = study.run_live_case(
        study.LiveCase(
            "live-zero-shared",
            ("compute", "memory"),
            {"compute": study.compute_launch(), "memory": study.memory_launch()},
        )
    )
    for half_step, zero_step in zip(half["steps"], zero["steps"]):
        assert half_step["step_latency_ps"] == 243_000
        assert zero_step["step_latency_ps"] == 133_000
        assert half_step["step_latency_ps"] - zero_step["step_latency_ps"] == 110_000


def test_live_identity_arbitration_and_priority_permutation_preserve_timing():
    launches = {"memory": study.memory_launch(), "network": study.egress_launch()}
    baseline = study.run_live_case(
        study.LiveCase("live-baseline", ("memory", "network"), launches)
    )
    explicit = study.run_live_case(
        study.LiveCase(
            "live-identity",
            ("memory", "network"),
            launches,
            explicit_identity_policy=True,
        )
    )
    permuted = study.run_live_case(
        study.LiveCase(
            "live-priority",
            ("memory", "network"),
            launches,
            priorities={"memory": 7, "network": -3},
        )
    )
    for other in (explicit, permuted):
        assert [step["step_latency_ps"] for step in other["steps"]] == [
            step["step_latency_ps"] for step in baseline["steps"]
        ]
        assert [step["step_completed_at_ps"] for step in other["steps"]] == [
            step["step_completed_at_ps"] for step in baseline["steps"]
        ]


@pytest.mark.parametrize("order", [("memory", "network"), ("network", "memory")])
def test_scalar_compatibility_path_stays_order_invariant(order):
    launches = {"memory": study.memory_launch(), "network": study.egress_launch()}
    observed = study.run_live_case(
        study.LiveCase(
            f"scalar-{'-'.join(order)}",
            order,
            launches,
            nominal_ps={"memory": 132_000, "network": 328_000},
        )
    )
    assert [step["step_latency_ps"] for step in observed["steps"]] == [328_000] * 3


def test_check_only_validates_the_frozen_registry(capsys):
    study.check_only()
    assert "check-only validated" in capsys.readouterr().out


def test_b100_roofline_context_is_the_configured_floor():
    observed = study.roofline_observations()
    assert observed["envelope_name"] == "b100"
    assert observed["envelope_mem_bandwidth"] == 8.0e12
    assert observed["roofline_estimate_ps"] == 99_366_034
    assert observed["b100_hardware_floor_ps"] == 69_556_224
    assert observed["h100_hardware_floor_ps"] == 166_104_415
    assert observed["host_initiation_delay_ps"] == 0
    assert observed["host_profile"] == "ideal"
