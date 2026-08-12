"""Replicate the corrected mixed-makespan forms registered for COMP-12.

The forms come from findings G1 and G2 of the task-mix study
(`examples/gpu_task_mix/RESULTS.md`). G1: a concurrent makespan is the longest
isolated control plus a submission-order issue delay. G2: tasks whose CTAs
exhaust an SM's shared memory serialize on residency instead of backfilling.

This runner replays the exact frozen fixtures of that study through the
component scheduler, then carries the same launches through the live CORE-4
runtime and the request-metric chain. Everything it asserts was frozen in
`expectations.md` before implementation. Nothing here fits a broader
scheduling law, and the synthetic 1 GHz profile is a mechanism fixture, never
a B100, H100 or Turing calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Frozen literals. Every value below is quoted from expectations.md and must
# never be edited to match an observation.
# ---------------------------------------------------------------------------

#: one cycle of the synthetic 1 GHz mechanism fixture
PS_PER_CYCLE = 1_000

ISSUE_BUDGET = 4
LANES = 4
SHARED_PER_SM = 65_536
ALU_LATENCY = 4
LOAD_STORE_LATENCY = 1
HBM_LATENCY = 100
NVLINK_LATENCY = 200
HBM_BYTES_PER_CYCLE = 64
NVLINK_BYTES_PER_CYCLE = 16
SM_COUNT = 2
TASK_BLOCKS = 8
TASK_INSTRUCTIONS_PER_WARP = 4
TRANSACTION_BYTES = 64

FROZEN_MEMORY_ISOLATED_CYCLES = 132
FROZEN_NETWORK_ISOLATED_CYCLES = 328

#: (submitted order, issue budget, load/store lanes, delta_issue, mixed cycles)
FROZEN_ISSUE_MATRIX: tuple[tuple[tuple[str, str], int, int, int, int], ...] = (
    (("memory", "network"), 4, 4, 1, 329),
    (("memory", "network"), 8, 4, 1, 329),
    (("memory", "network"), 4, 8, 1, 329),
    (("memory", "network"), 8, 8, 0, 328),
    (("network", "memory"), 4, 4, 0, 328),
)

FROZEN_HALF_COMPUTE_ISOLATED_CYCLES = 14
FROZEN_HALF_MEMORY_ISOLATED_CYCLES = 229
FROZEN_HALF_MEMORY_ADMISSION_CYCLE = 14
FROZEN_HALF_MIXED_CYCLES = 243
FROZEN_ZERO_COMPUTE_ISOLATED_CYCLES = 7
FROZEN_ZERO_MEMORY_ISOLATED_CYCLES = 132
FROZEN_ZERO_MIXED_CYCLES = 133

#: (issue budget, lanes) -> memory-first minus network-first step JCT, ps
FROZEN_LIVE_ISSUE_DELTA_PS: dict[tuple[int, int], int] = {(4, 4): 1_000, (8, 8): 0}
FROZEN_LIVE_RESIDENCY_DELTA_PS = 110_000

#: label -> (first-principles floor, serialized ceiling, expected location)
FROZEN_PHYSICAL_INTERVALS: dict[str, tuple[int, int, int]] = {
    "G1 baseline": (328, 460, 329),
    "G2 half shared": (229, 243, 243),
    "G2 zero shared": (132, 139, 133),
}

FROZEN_SCALAR_NOMINAL_PS = (132_000, 328_000)
FROZEN_SCALAR_JCT_PS = 328_000

#: Granite decode roofline context, kept apart from the synthetic fixture
ROOFLINE_MEMORY_BYTES = 556_449_792
B100_BANDWIDTH_BYTES_PER_S = 8.0e12
ROOFLINE_EFFICIENCY = 0.7
FROZEN_B100_HARDWARE_FLOOR_PS = 69_556_224
FROZEN_B100_CONFIGURED_FLOOR_PS = 99_366_034
FROZEN_H100_HARDWARE_FLOOR_PS = 166_104_415

T0_PS = 5_000
LIVE_REQUEST_ID = "mixed-makespan-request"
LIVE_STEP_COUNT = 3

SOURCE_STUDY_COMMIT = "0d9e2337eab6d5e49c112f3fbccb7d5e70a44f7f"
EXPECTATIONS_COMMIT = "3d079077ae91699a14c180eaba0e534bca7a7e91"

# Evidence classes. Only GENUINE_RISK rows enter a behavioral fraction.
GENUINE_RISK = "behavioral-relation"
FATAL_GUARD = "fatal-guard"
RAW_OBSERVATION = "raw-observation"
RUN_CONFIGURATION = "run-configuration"

FAMILY_G1_COMPONENT = "G1 component issue-delay matrix"
FAMILY_G2_COMPONENT = "G2 component residency and backfill form"
FAMILY_G1_LIVE = "live CORE-4 issue-order projection"
FAMILY_G2_LIVE = "live CORE-4 residency projection"
SCORED_FAMILIES = (
    FAMILY_G1_COMPONENT,
    FAMILY_G2_COMPONENT,
    FAMILY_G1_LIVE,
    FAMILY_G2_LIVE,
)


@dataclass(frozen=True)
class Row:
    """One evidence row. ``expected`` is frozen, ``measured`` is observed."""

    family: str
    evidence_class: str
    case: str
    expected: Any
    measured: Any

    @property
    def is_predicate(self) -> bool:
        """Raw observations and run configurations assert nothing."""

        return self.evidence_class in (GENUINE_RISK, FATAL_GUARD)

    @property
    def passed(self) -> bool:
        return not self.is_predicate or self.expected == self.measured

    @property
    def status(self) -> str:
        if not self.is_predicate:
            return "REPORTED"
        return "PASS" if self.passed else "FAIL"


def _row_payload(row: Row) -> dict[str, Any]:
    return {
        "family": row.family,
        "evidence_class": row.evidence_class,
        "case": row.case,
        "expected": row.expected,
        "measured": row.measured,
        "status": row.status,
    }


# ---------------------------------------------------------------------------
# Check-only path. Validates the frozen registry and its arithmetic without
# importing any SimLLM implementation, reading any input, or writing anything.
# ---------------------------------------------------------------------------


def check_only() -> None:
    """Validate the frozen literals against each other and exit."""

    floor = max(FROZEN_MEMORY_ISOLATED_CYCLES, FROZEN_NETWORK_ISOLATED_CYCLES)
    for order, budget, lanes, delta, mixed in FROZEN_ISSUE_MATRIX:
        if set(order) != {"memory", "network"}:
            raise ValueError(f"issue-matrix order {order} is not the frozen task pair")
        if budget not in (4, 8) or lanes not in (4, 8):
            raise ValueError("issue-matrix sweep leaves the frozen 4/8 grid")
        if floor + delta != mixed:
            raise ValueError(f"issue-matrix row {order} {budget} {lanes} is inconsistent")
        if delta not in (0, 1):
            raise ValueError("frozen issue delay is 0 or 1 cycle")

    widened = {
        (budget, lanes): delta
        for order, budget, lanes, delta, _ in FROZEN_ISSUE_MATRIX
        if order == ("memory", "network")
    }
    if widened != {(4, 4): 1, (8, 4): 1, (4, 8): 1, (8, 8): 0}:
        raise ValueError("frozen counterfactual sweep is not the registered grid")

    if (
        FROZEN_HALF_COMPUTE_ISOLATED_CYCLES + FROZEN_HALF_MEMORY_ISOLATED_CYCLES
        != FROZEN_HALF_MIXED_CYCLES
    ):
        raise ValueError("frozen G2 sum does not close")
    if FROZEN_HALF_MEMORY_ADMISSION_CYCLE != FROZEN_HALF_COMPUTE_ISOLATED_CYCLES:
        raise ValueError("frozen G2 admission does not equal the compute control")
    if (
        max(FROZEN_ZERO_COMPUTE_ISOLATED_CYCLES, FROZEN_ZERO_MEMORY_ISOLATED_CYCLES) + 1
        != FROZEN_ZERO_MIXED_CYCLES
    ):
        raise ValueError("frozen zero-shared control does not close")

    for label, (bound_floor, ceiling, location) in FROZEN_PHYSICAL_INTERVALS.items():
        if not bound_floor <= location <= ceiling:
            raise ValueError(f"frozen physical interval {label} excludes its location")

    live_floor = FROZEN_NETWORK_ISOLATED_CYCLES * PS_PER_CYCLE
    for (budget, lanes), delta_ps in FROZEN_LIVE_ISSUE_DELTA_PS.items():
        expected = widened[(budget, lanes)] * PS_PER_CYCLE
        if delta_ps != expected:
            raise ValueError(f"frozen live delta at {(budget, lanes)} contradicts the matrix")
    if live_floor != FROZEN_SCALAR_JCT_PS:
        raise ValueError("frozen scalar compatibility JCT is not the egress control")
    if max(FROZEN_SCALAR_NOMINAL_PS) != FROZEN_SCALAR_JCT_PS:
        raise ValueError("frozen scalar path is not the independent-resource maximum")
    residency_delta = (
        FROZEN_HALF_MIXED_CYCLES - FROZEN_ZERO_MIXED_CYCLES
    ) * PS_PER_CYCLE
    if residency_delta != FROZEN_LIVE_RESIDENCY_DELTA_PS:
        raise ValueError("frozen live residency delta contradicts the component rows")

    exact_floor = Fraction(ROOFLINE_MEMORY_BYTES) / (
        Fraction(8) * 10**12 * Fraction(7, 10)
    )
    if int(exact_floor * 10**12) != FROZEN_B100_CONFIGURED_FLOOR_PS:
        raise ValueError("frozen B100 configured floor is not the registered arithmetic")
    hardware_floor = Fraction(ROOFLINE_MEMORY_BYTES) / (Fraction(8) * 10**12)
    if int(hardware_floor * 10**12) != FROZEN_B100_HARDWARE_FLOOR_PS:
        raise ValueError("frozen B100 hardware floor is not the registered arithmetic")
    h100_floor = Fraction(ROOFLINE_MEMORY_BYTES) / (Fraction(335, 100) * 10**12)
    if int(h100_floor * 10**12) != FROZEN_H100_HARDWARE_FLOOR_PS:
        raise ValueError("frozen H100 floor is not the registered arithmetic")

    print(
        "check-only validated the frozen mixed-makespan registry and arithmetic; "
        "no SimLLM import, no input read, no artifact written"
    )


# ---------------------------------------------------------------------------
# Frozen synthetic fixture. Identical in shape to the task-mix study fixture,
# restated here so this study owns its own inventory and can guard it.
# ---------------------------------------------------------------------------


def architecture(
    *,
    scheduler_count_per_sm: int = ISSUE_BUDGET,
    load_store_issue_width_per_sm: int = LANES,
):
    """Return the frozen synthetic two-SM 1 GHz mechanism profile."""

    from simllm.compute import (
        GpuArchitectureProfile,
        GpuCalibrationProfile,
        GpuModelProvenance,
        MemoryHierarchyProfile,
        NvlinkProfile,
        PipelineKind,
        PipelineProfile,
        WarpSchedulerPolicy,
    )

    variant = f"sched{scheduler_count_per_sm}-ls{load_store_issue_width_per_sm}"
    profile_id = f"mixed-makespan-profile-{variant}"
    calibration = GpuCalibrationProfile(
        calibration_id=f"mixed-makespan-calibration-{variant}",
        target_architecture_profile_id=profile_id,
        provenance=GpuModelProvenance(
            source="synthetic study fixture, no silicon claim",
            version="1",
            gpu="mixed-makespan-synthetic",
            created="2026-08-12",
        ),
        core_clock_hz=1_000_000_000,
        target_memory_clock_hz=None,
        pipelines=(
            PipelineProfile(
                kind=PipelineKind.ALU,
                opcodes=("ALU",),
                latency_cycles=ALU_LATENCY,
                issue_width_per_sm=LANES,
                initiation_interval_cycles=1,
            ),
            PipelineProfile(
                kind=PipelineKind.LOAD_STORE,
                opcodes=("LDG", "STG"),
                latency_cycles=LOAD_STORE_LATENCY,
                issue_width_per_sm=load_store_issue_width_per_sm,
            ),
        ),
        memory=MemoryHierarchyProfile(
            hbm_latency_cycles=HBM_LATENCY,
            hbm_bandwidth_bytes_per_cycle=HBM_BYTES_PER_CYCLE,
            l2_latency_cycles=20,
            l1_latency_cycles=10,
            shared_latency_cycles=5,
        ),
        nvlink=NvlinkProfile(
            latency_cycles=NVLINK_LATENCY,
            bandwidth_bytes_per_cycle=NVLINK_BYTES_PER_CYCLE,
        ),
        copy_engines=(),
        warp_scheduler_policy=WarpSchedulerPolicy.LOOSE_ROUND_ROBIN,
        relative_uncertainty=0.0,
    )
    return GpuArchitectureProfile(
        profile_id=profile_id,
        gpu_name="mixed-makespan-synthetic",
        sm_count=SM_COUNT,
        warp_size=32,
        scheduler_count_per_sm=scheduler_count_per_sm,
        dispatch_width_per_scheduler=1,
        max_blocks_per_sm=16,
        max_warps_per_sm=64,
        max_threads_per_sm=2_048,
        max_threads_per_block=1_024,
        registers_per_sm=65_536,
        max_registers_per_thread=255,
        register_allocation_granularity_per_warp=1,
        shared_memory_per_sm=SHARED_PER_SM,
        max_static_shared_memory_per_block=SHARED_PER_SM,
        max_shared_memory_per_block=SHARED_PER_SM,
        shared_memory_allocation_granularity=1,
        calibration=calibration,
        aliases=(),
    )


def _launch(*, name: str, instructions, shared_bytes: int):
    from simllm.compute import CtaTrace, KernelLaunch, SassWarpTrace

    return KernelLaunch(
        implementation_id=name,
        trace_id=f"{name}-trace",
        grid_blocks=TASK_BLOCKS,
        threads_per_block=32,
        registers_per_thread=0,
        static_shared_memory_bytes=shared_bytes,
        dynamic_shared_memory_bytes=0,
        cta_traces=(
            CtaTrace(
                trace_class_id=f"{name}-cta",
                block_ids=tuple(range(TASK_BLOCKS)),
                warp_traces=(SassWarpTrace(warp_id=0, instructions=instructions),),
            ),
        ),
    )


def compute_launch(*, shared_bytes: int = 0):
    from simllm.compute import PipelineKind, SassInstruction

    instructions = tuple(
        SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU)
        for _ in range(TASK_INSTRUCTIONS_PER_WARP)
    )
    suffix = "half" if shared_bytes else "zero"
    return _launch(
        name=f"compute-{suffix}-shared",
        instructions=instructions,
        shared_bytes=shared_bytes,
    )


def memory_launch(*, shared_bytes: int = 0):
    from simllm.compute import MemorySpace, PipelineKind, SassInstruction

    instructions = tuple(
        SassInstruction(
            opcode="LDG",
            pipeline=PipelineKind.LOAD_STORE,
            memory_space=MemorySpace.HBM,
            requested_bytes=TRANSACTION_BYTES,
            transacted_bytes=TRANSACTION_BYTES,
        )
        for _ in range(TASK_INSTRUCTIONS_PER_WARP)
    )
    suffix = "half" if shared_bytes else "zero"
    return _launch(
        name=f"memory-{suffix}-shared",
        instructions=instructions,
        shared_bytes=shared_bytes,
    )


def egress_launch():
    from simllm.compute import MemorySpace, PipelineKind, SassInstruction

    instructions = tuple(
        SassInstruction(
            opcode="STG",
            pipeline=PipelineKind.LOAD_STORE,
            memory_space=MemorySpace.NVLINK,
            requested_bytes=TRANSACTION_BYTES,
            transacted_bytes=TRANSACTION_BYTES,
        )
        for _ in range(TASK_INSTRUCTIONS_PER_WARP)
    )
    return _launch(name="egress", instructions=instructions, shared_bytes=0)


# ---------------------------------------------------------------------------
# Component replay
# ---------------------------------------------------------------------------


def _concurrent(model, ordered):
    """Replay ``ordered`` as ``((task_id, kind, launch), ...)`` in that order."""

    from simllm.compute import GpuTask

    tasks = tuple(
        GpuTask(task_id=task_id, kind=kind, launch=launch)
        for task_id, kind, launch in ordered
    )
    return model.estimate_concurrent(tasks)


def component_observations() -> dict[str, Any]:
    """Return every raw component observation this study evaluates."""

    from simllm.compute import GpuTaskKind, SmSchedulerModel, decompose_mixed_makespan

    memory = memory_launch()
    egress = egress_launch()
    baseline = SmSchedulerModel(architecture())

    observations: dict[str, Any] = {
        "memory_isolated_cycles": baseline.estimate(memory).duration_cycles,
        "network_isolated_cycles": baseline.estimate(egress).duration_cycles,
        "issue_matrix": [],
        "conservation": [],
    }

    for order, budget, lanes, _, _ in FROZEN_ISSUE_MATRIX:
        model = SmSchedulerModel(
            architecture(
                scheduler_count_per_sm=budget,
                load_store_issue_width_per_sm=lanes,
            )
        )
        by_name = {
            "memory": ("memory", GpuTaskKind.MEMORY, memory),
            "network": ("network", GpuTaskKind.NETWORK, egress),
        }
        estimate = _concurrent(model, tuple(by_name[name] for name in order))
        isolated = {
            "memory": model.estimate(memory).duration_cycles,
            "network": model.estimate(egress).duration_cycles,
        }
        form = decompose_mixed_makespan(estimate, isolated)
        observations["issue_matrix"].append(
            {
                "order": list(order),
                "scheduler_count_per_sm": budget,
                "load_store_issue_width_per_sm": lanes,
                "mixed_cycles": estimate.duration_cycles,
                "issue_delay_cycles": form.issue_delay_cycles,
                "regime": form.regime.value,
                "task_ids": list(form.task_ids),
                "admitted_cycles": list(form.admitted_cycles),
                "eligible_cycles": list(form.eligible_cycles),
                "completion_cycles": list(form.completion_cycles),
                "concurrent_floor_cycles": form.concurrent_floor_cycles,
                "serialized_ceiling_cycles": form.serialized_ceiling_cycles,
                "within_physical_interval": form.within_physical_interval,
                "isolated_cycles": list(form.isolated_cycles),
            }
        )
        observations["conservation"].append(
            {
                "case": f"{'-'.join(order)}-{budget}-{lanes}",
                "issued_instructions": estimate.issued_instructions,
                "issued_instructions_by_task": sum(
                    task.issued_instructions for task in estimate.tasks
                ),
                "hbm_transacted_bytes": estimate.hbm_transacted_bytes,
                "hbm_transacted_bytes_by_task": sum(
                    task.hbm_transacted_bytes for task in estimate.tasks
                ),
                "nvlink_transacted_bytes": estimate.nvlink_transacted_bytes,
                "nvlink_transacted_bytes_by_task": sum(
                    task.nvlink_transacted_bytes for task in estimate.tasks
                ),
            }
        )

    half = SHARED_PER_SM // 2
    half_compute = compute_launch(shared_bytes=half)
    half_memory = memory_launch(shared_bytes=half)
    half_isolated = {
        "compute": baseline.estimate(half_compute).duration_cycles,
        "memory": baseline.estimate(half_memory).duration_cycles,
    }
    half_estimate = _concurrent(
        baseline,
        (
            ("compute", GpuTaskKind.COMPUTE, half_compute),
            ("memory", GpuTaskKind.MEMORY, half_memory),
        ),
    )
    half_form = decompose_mixed_makespan(half_estimate, half_isolated)

    zero_compute = compute_launch()
    zero_isolated = {
        "compute": baseline.estimate(zero_compute).duration_cycles,
        "memory": baseline.estimate(memory).duration_cycles,
    }
    zero_estimate = _concurrent(
        baseline,
        (
            ("compute", GpuTaskKind.COMPUTE, zero_compute),
            ("memory", GpuTaskKind.MEMORY, memory),
        ),
    )
    zero_form = decompose_mixed_makespan(zero_estimate, zero_isolated)

    observations["residency"] = {
        "half_compute_isolated_cycles": half_isolated["compute"],
        "half_memory_isolated_cycles": half_isolated["memory"],
        "half_mixed_cycles": half_estimate.duration_cycles,
        "half_regime": half_form.regime.value,
        "half_gated_task_id": half_form.residency_gated_task_id,
        "half_memory_admitted_cycle": half_form.admitted_cycles[
            half_form.task_ids.index("memory")
        ],
        "half_residency_serialized_cycles": half_form.residency_serialized_cycles,
        "half_within_physical_interval": half_form.within_physical_interval,
        "half_concurrent_floor_cycles": half_form.concurrent_floor_cycles,
        "half_serialized_ceiling_cycles": half_form.serialized_ceiling_cycles,
        "zero_compute_isolated_cycles": zero_isolated["compute"],
        "zero_memory_isolated_cycles": zero_isolated["memory"],
        "zero_mixed_cycles": zero_estimate.duration_cycles,
        "zero_regime": zero_form.regime.value,
        "zero_issue_delay_cycles": zero_form.issue_delay_cycles,
        "zero_within_physical_interval": zero_form.within_physical_interval,
        "zero_concurrent_floor_cycles": zero_form.concurrent_floor_cycles,
        "zero_serialized_ceiling_cycles": zero_form.serialized_ceiling_cycles,
    }
    observations["conservation"].extend(
        {
            "case": case,
            "issued_instructions": estimate.issued_instructions,
            "issued_instructions_by_task": sum(
                task.issued_instructions for task in estimate.tasks
            ),
            "hbm_transacted_bytes": estimate.hbm_transacted_bytes,
            "hbm_transacted_bytes_by_task": sum(
                task.hbm_transacted_bytes for task in estimate.tasks
            ),
            "nvlink_transacted_bytes": estimate.nvlink_transacted_bytes,
            "nvlink_transacted_bytes_by_task": sum(
                task.nvlink_transacted_bytes for task in estimate.tasks
            ),
        }
        for case, estimate in (("half-shared", half_estimate), ("zero-shared", zero_estimate))
    )
    observations["timestamps_nonnegative"] = all(
        task.admitted_cycle >= 0 and task.completion_cycle >= 0
        for estimate in (half_estimate, zero_estimate)
        for task in estimate.tasks
    )
    observations["kind_relabel"] = _kind_relabel_observation(memory, egress)
    observations["fixture"] = _fixture_inventory(
        {
            "memory": memory,
            "network": egress,
            "half-compute": half_compute,
            "half-memory": half_memory,
            "zero-compute": zero_compute,
        }
    )
    return observations


def _fixture_inventory(launches: dict[str, Any]) -> dict[str, Any]:
    """Report the constructed fixture's configuration, not its timing."""

    profile = architecture()
    calibration = profile.calibration
    load_store = next(
        pipeline
        for pipeline in calibration.pipelines
        if pipeline.opcodes == ("LDG", "STG")
    )
    inventory = {
        "sm_count": profile.sm_count,
        "scheduler_count_per_sm": profile.scheduler_count_per_sm,
        "load_store_issue_width_per_sm": load_store.issue_width_per_sm,
        "shared_memory_per_sm": profile.shared_memory_per_sm,
        "core_clock_hz": calibration.core_clock_hz,
        "hbm_latency_cycles": calibration.memory.hbm_latency_cycles,
        "hbm_bandwidth_bytes_per_cycle": calibration.memory.hbm_bandwidth_bytes_per_cycle,
        "nvlink_latency_cycles": calibration.nvlink.latency_cycles,
        "nvlink_bandwidth_bytes_per_cycle": calibration.nvlink.bandwidth_bytes_per_cycle,
        "launches": {},
    }
    for name, launch in sorted(launches.items()):
        warp_traces = launch.cta_traces[0].warp_traces
        instructions = warp_traces[0].instructions
        inventory["launches"][name] = {
            "grid_blocks": launch.grid_blocks,
            "warps_per_cta": len(warp_traces),
            "instructions_per_warp": len(instructions),
            "opcodes": sorted({instruction.opcode for instruction in instructions}),
            "transaction_bytes": sorted(
                {instruction.transacted_bytes for instruction in instructions}
            ),
            "static_shared_memory_bytes": launch.static_shared_memory_bytes,
        }
    return inventory


def _kind_relabel_observation(memory, egress) -> dict[str, Any]:
    """Swap only the attribution labels and record what moved."""

    from simllm.compute import GpuTaskKind, SmSchedulerModel

    model = SmSchedulerModel(architecture())
    labelled = _concurrent(
        model,
        (
            ("memory", GpuTaskKind.MEMORY, memory),
            ("network", GpuTaskKind.NETWORK, egress),
        ),
    )
    relabelled = _concurrent(
        model,
        (
            ("memory", GpuTaskKind.NETWORK, memory),
            ("network", GpuTaskKind.COMPUTE, egress),
        ),
    )
    return {
        "duration_unchanged": labelled.duration_cycles == relabelled.duration_cycles,
        "admission_unchanged": [task.admitted_cycle for task in labelled.tasks]
        == [task.admitted_cycle for task in relabelled.tasks],
        "completion_unchanged": [task.completion_cycle for task in labelled.tasks]
        == [task.completion_cycle for task in relabelled.tasks],
        "labels_moved": [task.kind.value for task in labelled.tasks]
        != [task.kind.value for task in relabelled.tasks],
    }


# ---------------------------------------------------------------------------
# Live CORE-4 replay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveCase:
    """One live configuration: ordered kernels on rank 0 of one request."""

    case_id: str
    order: tuple[str, str]
    launches: dict[str, Any]
    scheduler_count_per_sm: int = ISSUE_BUDGET
    load_store_issue_width_per_sm: int = LANES
    priorities: dict[str, int] | None = None
    explicit_identity_policy: bool = False
    nominal_ps: dict[str, int] | None = None


def _live_operations(case: LiveCase):
    """Build the ordered rank-0 compute operations of one live step."""

    from simllm.core import ComputeWork, ExecutionOperation, OperationCorrelation

    correlation = OperationCorrelation(request_ids=(LIVE_REQUEST_ID,))
    operations = []
    for name in case.order:
        nominal = None if case.nominal_ps is None else case.nominal_ps[name]
        operations.append(
            ExecutionOperation(
                name,
                0,
                f"cuda:0:{name}",
                ComputeWork(
                    name,
                    flops=0 if name == "memory" else 1,
                    hbm_bytes=(
                        TASK_BLOCKS * TASK_INSTRUCTIONS_PER_WARP * TRANSACTION_BYTES
                        if name == "memory"
                        else 0
                    ),
                    nominal_duration_ps=nominal,
                ),
                priority=0 if case.priorities is None else case.priorities[name],
                correlation=correlation,
            )
        )
    return tuple(operations)


def run_live_case(case: LiveCase) -> dict[str, Any]:
    """Execute one live case through CORE-4 and the request-metric chain."""

    from simllm.compute import SmSchedulerModel
    from simllm.core import (
        CoarseDeviceRuntime,
        CompletionReducer,
        EventPhase,
        ExecutionGraph,
        IdentityArbitrationPolicy,
        RequestPhase,
        ScheduledRequest,
        StepRecord,
        VirtualClock,
    )

    kwargs: dict[str, Any] = {}
    if case.nominal_ps is None:
        service = SmSchedulerModel(
            architecture(
                scheduler_count_per_sm=case.scheduler_count_per_sm,
                load_store_issue_width_per_sm=case.load_store_issue_width_per_sm,
            )
        )
        kwargs["kernel_services"] = {0: service}
        kwargs["kernel_launches"] = {name: case.launches[name] for name in case.order}
    if case.explicit_identity_policy:
        kwargs["arbitration_policy"] = IdentityArbitrationPolicy()
    runtime = CoarseDeviceRuntime(**kwargs)

    clock = VirtualClock(T0_PS)
    reducer = CompletionReducer(clock)
    steps: list[dict[str, Any]] = []
    for index in range(LIVE_STEP_COUNT):
        released_at_ps = clock.now_ps
        graph = ExecutionGraph(
            execution_id=f"{case.case_id}-{index}",
            step_index=index,
            released_at_ps=released_at_ps,
            operations=_live_operations(case),
        )
        record = StepRecord(
            step_index=index,
            virtual_time_ps=released_at_ps,
            scheduled=[
                ScheduledRequest(
                    LIVE_REQUEST_ID,
                    RequestPhase.PREFILL if index == 0 else RequestPhase.DECODE,
                    1,
                )
            ],
            num_sampled=1,
        )
        result = runtime.execute(graph)
        report = runtime.last_report
        assert report is not None
        step = reducer.reduce(record, graph, result, report)
        metric = step.request_metrics[0]

        visit_by_operation = {
            visit.operation_id: visit
            for visit in report.visits
            if visit.subject_object_id is None
        }
        queued_matches_eligibility = all(
            event.timestamp_ps == visit_by_operation[event.operation_id].eligible_at_ps
            for event in result.events
            if event.phase is EventPhase.QUEUED and event.subject_object_id is None
        )
        started_matches_grant = all(
            event.timestamp_ps == visit_by_operation[event.operation_id].started_at_ps
            for event in result.events
            if event.phase is EventPhase.STARTED and event.subject_object_id is None
        )
        completed_events = [
            event
            for event in result.events
            if event.phase is EventPhase.COMPLETED and event.subject_object_id is None
        ]
        operation_completion = {
            record_row.operation_id: record_row.completed_at_ps
            for record_row in report.operations
        }
        steps.append(
            {
                "step_index": index,
                "released_at_ps": released_at_ps,
                "graph_order": [
                    operation.operation_id for operation in graph.operations
                ],
                "report_operation_order": [
                    record_row.operation_id for record_row in report.operations
                ],
                "step_latency_ps": step.step_latency_ps,
                "step_completed_at_ps": step.completed_at_ps,
                "execution_completed_at_ps": result.completed_at_ps,
                "ttft_ps": metric.ttft_ps,
                "tpot_ps": None if metric.tpot_ps is None else int(metric.tpot_ps),
                "metric_completed_at_ps": metric.completed_at_ps,
                "metric_latency_ps": metric.latency_ps,
                "metric_attribution_total_ps": metric.attribution.total_ps,
                "queued_matches_eligibility": queued_matches_eligibility,
                "started_matches_grant": started_matches_grant,
                "completed_event_count": len(completed_events),
                "completed_events_match_operations": sorted(
                    (event.operation_id, event.timestamp_ps)
                    for event in completed_events
                )
                == sorted(operation_completion.items()),
            }
        )
    return {"case_id": case.case_id, "steps": steps}


def live_observations() -> dict[str, dict[str, Any]]:
    """Run every live configuration this study registered."""

    memory = memory_launch()
    egress = egress_launch()
    half = SHARED_PER_SM // 2
    issue_launches = {"memory": memory, "network": egress}
    half_launches = {
        "compute": compute_launch(shared_bytes=half),
        "memory": memory_launch(shared_bytes=half),
    }
    zero_launches = {"compute": compute_launch(), "memory": memory}

    cases = [
        LiveCase("issue-memory-first-4-4", ("memory", "network"), issue_launches),
        LiveCase("issue-network-first-4-4", ("network", "memory"), issue_launches),
        LiveCase(
            "issue-memory-first-8-8",
            ("memory", "network"),
            issue_launches,
            scheduler_count_per_sm=8,
            load_store_issue_width_per_sm=8,
        ),
        LiveCase(
            "issue-network-first-8-8",
            ("network", "memory"),
            issue_launches,
            scheduler_count_per_sm=8,
            load_store_issue_width_per_sm=8,
        ),
        LiveCase("residency-half-shared", ("compute", "memory"), half_launches),
        LiveCase("residency-zero-shared", ("compute", "memory"), zero_launches),
        LiveCase(
            "identity-policy-memory-first-4-4",
            ("memory", "network"),
            issue_launches,
            explicit_identity_policy=True,
        ),
        LiveCase(
            "priority-permuted-memory-first-4-4",
            ("memory", "network"),
            issue_launches,
            priorities={"memory": 7, "network": -3},
        ),
        LiveCase(
            "scalar-memory-first",
            ("memory", "network"),
            issue_launches,
            nominal_ps={"memory": FROZEN_SCALAR_NOMINAL_PS[0], "network": FROZEN_SCALAR_NOMINAL_PS[1]},
        ),
        LiveCase(
            "scalar-network-first",
            ("network", "memory"),
            issue_launches,
            nominal_ps={"memory": FROZEN_SCALAR_NOMINAL_PS[0], "network": FROZEN_SCALAR_NOMINAL_PS[1]},
        ),
    ]
    return {case.case_id: run_live_case(case) for case in cases}


def roofline_observations() -> dict[str, Any]:
    """Record the B100 production context kept apart from the fixture."""

    from simllm.compute import (
        GPU_ENVELOPES,
        HostInitiationModel,
        KernelSpec,
        RooflineProvider,
    )

    gpu = GPU_ENVELOPES["b100"]
    estimate = RooflineProvider(efficiency=ROOFLINE_EFFICIENCY).estimate(
        KernelSpec(
            name="granite-decode-step",
            flops=0,
            bytes_moved=ROOFLINE_MEMORY_BYTES,
        ),
        gpu,
    )
    host = HostInitiationModel()
    return {
        "envelope_name": gpu.name,
        "envelope_mem_bandwidth": gpu.mem_bandwidth,
        "roofline_estimate_ps": estimate.duration_ps,
        "roofline_bound": estimate.bound,
        "b100_hardware_floor_ps": int(
            Fraction(ROOFLINE_MEMORY_BYTES) / (Fraction(8) * 10**12) * 10**12
        ),
        "h100_hardware_floor_ps": int(
            Fraction(ROOFLINE_MEMORY_BYTES) / (Fraction(335, 100) * 10**12) * 10**12
        ),
        "host_initiation_delay_ps": host.initiation_delay_ps,
        "host_profile": host.profile,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_rows(
    component: dict[str, Any],
    live: dict[str, dict[str, Any]],
    roofline: dict[str, Any],
    worktree_clean: tuple[bool, bool],
) -> list[Row]:
    """Evaluate every registered relation and guard from raw observations."""

    rows: list[Row] = []

    # Family 1: G1 component issue-delay matrix, evaluated from raw cycles.
    for frozen, measured in zip(
        FROZEN_ISSUE_MATRIX, component["issue_matrix"], strict=True
    ):
        order, budget, lanes, delta, mixed = frozen
        case = f"{'-'.join(order)},sched={budget},ls={lanes}"
        rows.append(
            Row(
                FAMILY_G1_COMPONENT,
                GENUINE_RISK,
                case,
                (delta, mixed),
                (measured["issue_delay_cycles"], measured["mixed_cycles"]),
            )
        )

    # Family 2: G2 component residency and backfill form.
    residency = component["residency"]
    rows.append(
        Row(
            FAMILY_G2_COMPONENT,
            GENUINE_RISK,
            "half-shared tasks serialize on residency",
            FROZEN_HALF_MIXED_CYCLES,
            residency["half_mixed_cycles"],
        )
    )
    rows.append(
        Row(
            FAMILY_G2_COMPONENT,
            GENUINE_RISK,
            "memory admits when constrained compute finishes",
            FROZEN_HALF_MEMORY_ADMISSION_CYCLE,
            residency["half_memory_admitted_cycle"],
        )
    )
    rows.append(
        Row(
            FAMILY_G2_COMPONENT,
            GENUINE_RISK,
            "zero-shared control restores backfill plus the issue delay",
            (FROZEN_ZERO_MIXED_CYCLES, 1),
            (residency["zero_mixed_cycles"], residency["zero_issue_delay_cycles"]),
        )
    )

    # Family 3: live CORE-4 issue-order projection, from raw step latencies.
    for budget, lanes in sorted(FROZEN_LIVE_ISSUE_DELTA_PS):
        memory_first = live[f"issue-memory-first-{budget}-{lanes}"]["steps"]
        network_first = live[f"issue-network-first-{budget}-{lanes}"]["steps"]
        deltas = {
            first["step_latency_ps"] - second["step_latency_ps"]
            for first, second in zip(memory_first, network_first, strict=True)
        }
        rows.append(
            Row(
                FAMILY_G1_LIVE,
                GENUINE_RISK,
                f"memory-first minus network-first JCT,sched={budget},ls={lanes}",
                FROZEN_LIVE_ISSUE_DELTA_PS[(budget, lanes)],
                deltas.pop() if len(deltas) == 1 else sorted(deltas),
            )
        )

    # Family 4: live CORE-4 residency projection, from raw step latencies.
    half_steps = live["residency-half-shared"]["steps"]
    zero_steps = live["residency-zero-shared"]["steps"]
    residency_deltas = {
        half["step_latency_ps"] - zero["step_latency_ps"]
        for half, zero in zip(half_steps, zero_steps, strict=True)
    }
    rows.append(
        Row(
            FAMILY_G2_LIVE,
            GENUINE_RISK,
            "half-shared minus zero-shared JCT",
            FROZEN_LIVE_RESIDENCY_DELTA_PS,
            residency_deltas.pop() if len(residency_deltas) == 1 else sorted(residency_deltas),
        )
    )

    rows.extend(_raw_rows(component, live))
    rows.extend(_guard_rows(component, live, roofline, worktree_clean))
    return rows


def _raw_rows(
    component: dict[str, Any],
    live: dict[str, dict[str, Any]],
) -> list[Row]:
    """Raw observations. Reported for review, never scored and never fatal.

    These carry the numbers a reader needs to check the scored relations by
    hand: the isolated controls the deltas are taken against, the residency
    decomposition behind the G2 sum, and the absolute live step latencies. A
    difference-only relation cannot see a constant offset added to both sides,
    so the absolute values belong in the record.
    """

    residency = component["residency"]
    configurations = {
        (
            entry["scheduler_count_per_sm"],
            entry["load_store_issue_width_per_sm"],
            tuple(entry["order"]),
        )
        for entry in component["issue_matrix"]
    }
    rows = [
        Row(
            "run configuration",
            RUN_CONFIGURATION,
            "distinct component and live replays",
            "reported",
            {
                "component issue configurations": len(configurations),
                "component residency configurations": 2,
                "live cases": len(live),
                "live steps per case": LIVE_STEP_COUNT,
            },
        ),
        Row(
            "raw isolated controls",
            RAW_OBSERVATION,
            "measured single-task durations, cycles",
            "reported",
            {
                "memory": component["memory_isolated_cycles"],
                "network": component["network_isolated_cycles"],
                "half-shared compute": residency["half_compute_isolated_cycles"],
                "half-shared memory": residency["half_memory_isolated_cycles"],
                "zero-shared compute": residency["zero_compute_isolated_cycles"],
                "zero-shared memory": residency["zero_memory_isolated_cycles"],
            },
        ),
        Row(
            "raw residency decomposition",
            RAW_OBSERVATION,
            "measured regime, gated task and serialized form",
            "reported",
            {
                "half regime": residency["half_regime"],
                "half gated task": residency["half_gated_task_id"],
                "half admission": residency["half_memory_admitted_cycle"],
                "half admission plus isolated": residency[
                    "half_residency_serialized_cycles"
                ],
                "half makespan": residency["half_mixed_cycles"],
                "zero regime": residency["zero_regime"],
                "zero issue delay": residency["zero_issue_delay_cycles"],
            },
        ),
    ]
    for case_id, case in sorted(live.items()):
        rows.append(
            Row(
                "raw live step latency",
                RAW_OBSERVATION,
                f"absolute step JCT, ps,{case_id}",
                "reported",
                [step["step_latency_ps"] for step in case["steps"]],
            )
        )
    return rows


def _guard_rows(
    component: dict[str, Any],
    live: dict[str, dict[str, Any]],
    roofline: dict[str, Any],
    worktree_clean: tuple[bool, bool],
) -> list[Row]:
    """Fatal and by-construction guards. Never part of a behavioral score."""

    rows: list[Row] = []
    residency = component["residency"]

    # The frozen inventory guard fixes the CONFIGURATION, not any measured
    # duration. Pinning the measured isolated controls fatally would entail
    # the scored makespans and convert genuine risk into bookkeeping.
    fixture = component["fixture"]
    rows.append(
        Row(
            "fixture inventory",
            FATAL_GUARD,
            "frozen synthetic profile",
            (
                SM_COUNT,
                ISSUE_BUDGET,
                LANES,
                SHARED_PER_SM,
                1_000_000_000,
                HBM_LATENCY,
                HBM_BYTES_PER_CYCLE,
                NVLINK_LATENCY,
                NVLINK_BYTES_PER_CYCLE,
            ),
            (
                fixture["sm_count"],
                fixture["scheduler_count_per_sm"],
                fixture["load_store_issue_width_per_sm"],
                fixture["shared_memory_per_sm"],
                fixture["core_clock_hz"],
                fixture["hbm_latency_cycles"],
                fixture["hbm_bandwidth_bytes_per_cycle"],
                fixture["nvlink_latency_cycles"],
                fixture["nvlink_bandwidth_bytes_per_cycle"],
            ),
        )
    )
    half = SHARED_PER_SM // 2
    frozen_launches = {
        "memory": (["LDG"], [TRANSACTION_BYTES], 0),
        "network": (["STG"], [TRANSACTION_BYTES], 0),
        "half-compute": (["ALU"], [0], half),
        "half-memory": (["LDG"], [TRANSACTION_BYTES], half),
        "zero-compute": (["ALU"], [0], 0),
    }
    for name, (opcodes, transaction_bytes, shared_bytes) in sorted(
        frozen_launches.items()
    ):
        observed = fixture["launches"][name]
        rows.append(
            Row(
                "fixture inventory",
                FATAL_GUARD,
                f"frozen launch,{name}",
                (
                    TASK_BLOCKS,
                    1,
                    TASK_INSTRUCTIONS_PER_WARP,
                    opcodes,
                    transaction_bytes,
                    shared_bytes,
                ),
                (
                    observed["grid_blocks"],
                    observed["warps_per_cta"],
                    observed["instructions_per_warp"],
                    observed["opcodes"],
                    observed["transaction_bytes"],
                    observed["static_shared_memory_bytes"],
                ),
            )
        )

    # The frozen physical guard is interval membership. The exact location
    # inside the interval is a scored relation, so it is not pinned here.
    intervals = {
        "G1 baseline": (
            component["issue_matrix"][0]["within_physical_interval"],
            component["issue_matrix"][0]["concurrent_floor_cycles"],
            component["issue_matrix"][0]["serialized_ceiling_cycles"],
        ),
        "G2 half shared": (
            residency["half_within_physical_interval"],
            residency["half_concurrent_floor_cycles"],
            residency["half_serialized_ceiling_cycles"],
        ),
        "G2 zero shared": (
            residency["zero_within_physical_interval"],
            residency["zero_concurrent_floor_cycles"],
            residency["zero_serialized_ceiling_cycles"],
        ),
    }
    for label, (bound_floor, ceiling, _) in FROZEN_PHYSICAL_INTERVALS.items():
        rows.append(
            Row(
                "physical interval",
                FATAL_GUARD,
                f"{label} in [{bound_floor}, {ceiling}]",
                (True, bound_floor, ceiling),
                intervals[label],
            )
        )

    rows.append(
        Row(
            "component timestamps",
            FATAL_GUARD,
            "admission and completion are nonnegative",
            True,
            component["timestamps_nonnegative"],
        )
    )

    relabel = component["kind_relabel"]
    rows.append(
        Row(
            "task-kind relabel",
            FATAL_GUARD,
            "labels move, timing does not",
            (True, True, True, True),
            (
                relabel["duration_unchanged"],
                relabel["admission_unchanged"],
                relabel["completion_unchanged"],
                relabel["labels_moved"],
            ),
        )
    )

    for entry in component["conservation"]:
        rows.append(
            Row(
                "component conservation",
                FATAL_GUARD,
                f"per-task totals conserve,{entry['case']}",
                (
                    entry["issued_instructions"],
                    entry["hbm_transacted_bytes"],
                    entry["nvlink_transacted_bytes"],
                ),
                (
                    entry["issued_instructions_by_task"],
                    entry["hbm_transacted_bytes_by_task"],
                    entry["nvlink_transacted_bytes_by_task"],
                ),
            )
        )

    baseline = live["issue-memory-first-4-4"]["steps"]
    for label in (
        "identity-policy-memory-first-4-4",
        "priority-permuted-memory-first-4-4",
    ):
        rows.append(
            Row(
                "identity preservation",
                FATAL_GUARD,
                label,
                [
                    (
                        step["step_latency_ps"],
                        step["step_completed_at_ps"],
                        step["ttft_ps"],
                        step["tpot_ps"],
                    )
                    for step in baseline
                ],
                [
                    (
                        step["step_latency_ps"],
                        step["step_completed_at_ps"],
                        step["ttft_ps"],
                        step["tpot_ps"],
                    )
                    for step in live[label]["steps"]
                ],
            )
        )

    rows.append(
        Row(
            "scalar compatibility path",
            FATAL_GUARD,
            "order-invariant independent-resource maximum",
            [FROZEN_SCALAR_JCT_PS] * (2 * LIVE_STEP_COUNT),
            [
                step["step_latency_ps"]
                for label in ("scalar-memory-first", "scalar-network-first")
                for step in live[label]["steps"]
            ],
        )
    )

    for case_id, case in sorted(live.items()):
        for step in case["steps"]:
            rows.append(
                Row(
                    "live graph order and identity",
                    FATAL_GUARD,
                    f"{case_id},step={step['step_index']}",
                    (sorted(step["graph_order"]), len(step["graph_order"])),
                    (
                        sorted(step["report_operation_order"]),
                        len(set(step["report_operation_order"])),
                    ),
                )
            )
            rows.append(
                Row(
                    "queue-visit contract",
                    FATAL_GUARD,
                    f"{case_id},step={step['step_index']}",
                    (True, True, len(step["graph_order"]), True),
                    (
                        step["queued_matches_eligibility"],
                        step["started_matches_grant"],
                        step["completed_event_count"],
                        step["completed_events_match_operations"],
                    ),
                )
            )
            rows.append(
                Row(
                    "live metric conservation",
                    FATAL_GUARD,
                    f"{case_id},step={step['step_index']}",
                    (
                        step["execution_completed_at_ps"],
                        step["step_completed_at_ps"] - step["released_at_ps"],
                        step["step_completed_at_ps"],
                        step["metric_latency_ps"],
                    ),
                    (
                        step["step_completed_at_ps"],
                        step["step_latency_ps"],
                        step["metric_completed_at_ps"],
                        step["metric_attribution_total_ps"],
                    ),
                )
            )

    rows.append(
        Row(
            "B100 roofline context",
            FATAL_GUARD,
            "envelope name, bandwidth and configured floor",
            ("b100", B100_BANDWIDTH_BYTES_PER_S, FROZEN_B100_CONFIGURED_FLOOR_PS, "memory"),
            (
                roofline["envelope_name"],
                roofline["envelope_mem_bandwidth"],
                roofline["roofline_estimate_ps"],
                roofline["roofline_bound"],
            ),
        )
    )
    rows.append(
        Row(
            "B100 roofline context",
            FATAL_GUARD,
            "hardware floors and zero host initiation",
            (FROZEN_B100_HARDWARE_FLOOR_PS, FROZEN_H100_HARDWARE_FLOOR_PS, 0, "ideal"),
            (
                roofline["b100_hardware_floor_ps"],
                roofline["h100_hardware_floor_ps"],
                roofline["host_initiation_delay_ps"],
                roofline["host_profile"],
            ),
        )
    )
    rows.append(
        Row(
            "run hygiene",
            FATAL_GUARD,
            "clean worktree before the run and still clean after it writes",
            (True, True),
            worktree_clean,
        )
    )

    # By-construction reachability: repeated equal steps make TTFT and each
    # decode TPOT equal the step JCT. Entailed, so unscored.
    for case_id, case in sorted(live.items()):
        equalities = []
        for step in case["steps"]:
            equalities.append(step["ttft_ps"] == step["step_latency_ps"])
            if step["step_index"] > 0:
                equalities.append(step["tpot_ps"] == step["step_latency_ps"])
        rows.append(
            Row(
                "metric reachability",
                FATAL_GUARD,
                f"TTFT and TPOT equal the step JCT,{case_id}",
                [True] * len(equalities),
                equalities,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _worktree_clean() -> bool:
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip() == ""


def _observed_revision() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_rows_csv(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["family", "evidence_class", "case", "expected", "measured", "status"])
        for row in rows:
            payload = _row_payload(row)
            writer.writerow(
                [
                    payload["family"],
                    payload["evidence_class"],
                    payload["case"],
                    json.dumps(payload["expected"], sort_keys=True),
                    json.dumps(payload["measured"], sort_keys=True),
                    payload["status"],
                ]
            )


def run_study(out: Path) -> int:
    """Produce the registered result. Return the process exit code."""

    out.mkdir(parents=True, exist_ok=True)
    clean_before = _worktree_clean()
    component = component_observations()
    live = live_observations()
    roofline = roofline_observations()
    # Write the raw record first, then re-read the worktree. Every later write
    # takes the same code path into ``out``, so this pair demonstrates that the
    # run leaves the repository untouched.
    _write_json(out / "raw_observations.json", {"component": component, "live": live})
    clean_after = _worktree_clean()
    worktree_clean = (clean_before, clean_after)
    rows = score_rows(component, live, roofline, worktree_clean)

    failed_guards = [row for row in rows if row.evidence_class == FATAL_GUARD and not row.passed]
    scored = [row for row in rows if row.evidence_class == GENUINE_RISK]
    void = bool(failed_guards)

    family_scores = {
        family: {
            "passed": sum(1 for row in scored if row.family == family and row.passed),
            "instances": sum(1 for row in scored if row.family == family),
        }
        for family in SCORED_FAMILIES
    }
    summary: dict[str, Any] = {
        "void": void,
        "expectations_commit": EXPECTATIONS_COMMIT,
        "source_study_commit": SOURCE_STUDY_COMMIT,
        "observed_revision": _observed_revision(),
        "worktree_clean_before_run": clean_before,
        "worktree_clean_after_writes": clean_after,
        "fatal_guards_evaluated": sum(
            1 for row in rows if row.evidence_class == FATAL_GUARD
        ),
        "fatal_guards_failed": [row.case for row in failed_guards],
        "scored_families": family_scores if not void else None,
        "scored_instances": len(scored) if not void else None,
        "scored_passed": (
            sum(1 for row in scored if row.passed) if not void else None
        ),
        "roofline": roofline,
    }

    _write_rows_csv(out / "rows.csv", rows)
    _write_json(out / "summary.json", summary)

    print(f"wrote {out / 'rows.csv'}")
    print(f"wrote {out / 'raw_observations.json'}")
    print(f"wrote {out / 'summary.json'}")
    if void:
        print("VOID: a frozen fatal guard failed; the behavioral score is uninterpretable")
        for row in failed_guards:
            print(f"  fatal fail: {row.family} / {row.case}")
        return 1
    print(
        "scored "
        f"{summary['scored_passed']} of {summary['scored_instances']} "
        f"genuine-risk instances across {len(SCORED_FAMILIES)} families"
    )
    return 0 if summary["scored_passed"] == summary["scored_instances"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check_only:
        check_only()
        return 0
    if args.out is None:
        raise SystemExit(
            "--out is required; set the run root in local configuration, e.g. "
            "SIMLLM_MIXED_MAKESPAN_RUN_ROOT, and pass it explicitly"
        )
    return run_study(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
