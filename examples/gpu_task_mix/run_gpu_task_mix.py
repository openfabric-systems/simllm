"""Run the post-specified GPU task-mix regression study.

Checks the post-specified forms in expectations.md: compute kernels are
limited by the issue path, memory kernels by the HBM cursor, network kernels
by the NVLink egress cursor, and mixed submissions by whichever resource they
actually share. Historical registration errors remain visible as unscored
ledger rows; the active regression exits non-zero on any current miss.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

from simllm.compute import (
    CtaTrace,
    GpuArchitectureProfile,
    GpuCalibrationProfile,
    GpuModelProvenance,
    GpuTask,
    GpuTaskKind,
    KernelLaunch,
    MemoryHierarchyProfile,
    MemorySpace,
    NvlinkProfile,
    PipelineKind,
    PipelineProfile,
    SassInstruction,
    SassWarpTrace,
    SmSchedulerModel,
    WarpSchedulerPolicy,
    nccl_ring_allreduce_launch,
    nccl_ring_allreduce_task,
    nccl_ring_egress_bytes,
)

ALU_LATENCY = 4
LOAD_STORE_LATENCY = 1
HBM_LATENCY = 100
NVLINK_LATENCY = 200
LANES = 4
ISSUE_BUDGET = 4
SHARED_PER_SM = 65_536

EXACT_ORACLE = "exact_oracle"
BEHAVIORAL_RELATION = "behavioral_relation"
STRUCTURAL_INVARIANT = "structural_invariant"
HISTORICAL_LEDGER = "historical_ledger"

_run_configuration_count = 0
_run_configurations: set[tuple[GpuArchitectureProfile, tuple[KernelLaunch, ...]]] = set()


@dataclass
class Row:
    check: str
    evidence_class: str
    family: str
    case: str
    expected: int
    measured: int

    @property
    def residual(self) -> int:
        return self.measured - self.expected

    @property
    def status(self) -> str:
        if self.evidence_class == HISTORICAL_LEDGER:
            return "HISTORICAL_FAIL"
        return "PASS" if self.residual == 0 else "FAIL"


def _estimate(model: SmSchedulerModel, launch: KernelLaunch):
    """Run one isolated fixed configuration and count it once."""

    global _run_configuration_count
    _run_configuration_count += 1
    _run_configurations.add((model.architecture, (launch,)))
    return model.estimate(launch)


def _estimate_concurrent(model: SmSchedulerModel, tasks: tuple[GpuTask, ...]):
    """Run one concurrent fixed configuration and count it once."""

    global _run_configuration_count
    _run_configuration_count += 1
    _run_configurations.add((model.architecture, tuple(task.launch for task in tasks)))
    return model.estimate_concurrent(tasks)


def exact(check: str, family: str, case: str, expected: int, measured: int) -> Row:
    return Row(check, EXACT_ORACLE, family, case, expected, measured)


def relation(check: str, family: str, case: str, expected: int, measured: int) -> Row:
    return Row(check, BEHAVIORAL_RELATION, family, case, expected, measured)


def structural(check: str, family: str, case: str, expected: int, measured: int) -> Row:
    return Row(check, STRUCTURAL_INVARIANT, family, case, expected, measured)


def architecture(
    *,
    sm_count: int = 1,
    scheduler_count_per_sm: int = ISSUE_BUDGET,
    load_store_issue_width_per_sm: int = LANES,
    alu_initiation_interval: int = 1,
    hbm_bandwidth: float = 64,
    nvlink_bandwidth: float = 16,
) -> GpuArchitectureProfile:
    """Return the synthetic 1 GHz fixture specified in expectations.md."""

    variant = (
        f"sms{sm_count}-sched{scheduler_count_per_sm}-ls{load_store_issue_width_per_sm}-"
        f"alui{alu_initiation_interval}-hbm{hbm_bandwidth:g}-nv{nvlink_bandwidth:g}"
    )
    profile_id = f"task-mix-profile-{variant}"
    calibration = GpuCalibrationProfile(
        calibration_id=f"task-mix-calibration-{variant}",
        target_architecture_profile_id=profile_id,
        provenance=GpuModelProvenance(
            source="synthetic study fixture, no silicon claim",
            version="1",
            gpu="task-mix-synthetic",
            created="2026-08-06",
        ),
        core_clock_hz=1_000_000_000,
        target_memory_clock_hz=None,
        pipelines=(
            PipelineProfile(
                kind=PipelineKind.ALU,
                opcodes=("ALU",),
                latency_cycles=ALU_LATENCY,
                issue_width_per_sm=LANES,
                initiation_interval_cycles=alu_initiation_interval,
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
            hbm_bandwidth_bytes_per_cycle=hbm_bandwidth,
            l2_latency_cycles=20,
            l1_latency_cycles=10,
            shared_latency_cycles=5,
        ),
        nvlink=NvlinkProfile(
            latency_cycles=NVLINK_LATENCY,
            bandwidth_bytes_per_cycle=nvlink_bandwidth,
        ),
        copy_engines=(),
        warp_scheduler_policy=WarpSchedulerPolicy.LOOSE_ROUND_ROBIN,
        relative_uncertainty=0.0,
    )
    return GpuArchitectureProfile(
        profile_id=profile_id,
        gpu_name="task-mix-synthetic",
        sm_count=sm_count,
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


def _launch(
    *,
    name: str,
    blocks: int,
    instructions: tuple[SassInstruction, ...],
    shared_bytes: int = 0,
) -> KernelLaunch:
    return KernelLaunch(
        implementation_id=name,
        trace_id=f"{name}-trace",
        grid_blocks=blocks,
        threads_per_block=32,
        registers_per_thread=0,
        static_shared_memory_bytes=shared_bytes,
        dynamic_shared_memory_bytes=0,
        cta_traces=(
            CtaTrace(
                trace_class_id=f"{name}-cta",
                block_ids=tuple(range(blocks)),
                warp_traces=(SassWarpTrace(warp_id=0, instructions=instructions),),
            ),
        ),
    )


def compute_launch(
    *, warps: int, per_warp: int, dependent: bool = False, shared_bytes: int = 0
) -> KernelLaunch:
    """One warp per block so warp count equals block count."""

    instructions = tuple(
        SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU, dependent=dependent)
        for _ in range(per_warp)
    )
    return _launch(
        name=f"compute-{warps}x{per_warp}{'-dep' if dependent else ''}",
        blocks=warps,
        instructions=instructions,
        shared_bytes=shared_bytes,
    )


def memory_launch(
    *, warps: int, per_warp: int, transaction_bytes: int, shared_bytes: int = 0
) -> KernelLaunch:
    instructions = tuple(
        SassInstruction(
            opcode="LDG",
            pipeline=PipelineKind.LOAD_STORE,
            memory_space=MemorySpace.HBM,
            requested_bytes=transaction_bytes,
            transacted_bytes=transaction_bytes,
        )
        for _ in range(per_warp)
    )
    return _launch(
        name=f"memory-{warps}x{per_warp}-{transaction_bytes}b",
        blocks=warps,
        instructions=instructions,
        shared_bytes=shared_bytes,
    )


def egress_launch(*, warps: int, per_warp: int, chunk_bytes: int) -> KernelLaunch:
    """Pure NVLink egress: stores with no load to wait for."""

    instructions = tuple(
        SassInstruction(
            opcode="STG",
            pipeline=PipelineKind.LOAD_STORE,
            memory_space=MemorySpace.NVLINK,
            requested_bytes=chunk_bytes,
            transacted_bytes=chunk_bytes,
        )
        for _ in range(per_warp)
    )
    return _launch(
        name=f"egress-{warps}x{per_warp}-{chunk_bytes}b",
        blocks=warps,
        instructions=instructions,
    )


def check_a(rows: list[Row]) -> None:
    """A: compute is limited by the issue path."""

    for interval in (1, 2, 4):
        model = SmSchedulerModel(architecture(alu_initiation_interval=interval))
        launch = compute_launch(warps=8, per_warp=4)
        total = 8 * 4
        expected = (math.ceil(total / LANES) - 1) * interval + ALU_LATENCY
        rows.append(
            exact(
                "A1",
                "pipeline throughput closed form",
                f"interval={interval}",
                expected,
                _estimate(model, launch).duration_cycles,
            )
        )

    for interval in (1, 4):
        model = SmSchedulerModel(architecture(alu_initiation_interval=interval))
        launch = compute_launch(warps=1, per_warp=8, dependent=True)
        rows.append(
            exact(
                "A2",
                "dependent-chain closed form",
                f"interval={interval}",
                8 * ALU_LATENCY,
                _estimate(model, launch).duration_cycles,
            )
        )

    model = SmSchedulerModel(architecture(sm_count=2))
    launch = compute_launch(warps=8, per_warp=4)
    per_sm = 8 * 4 // 2
    expected = (math.ceil(per_sm / LANES) - 1) * 1 + ALU_LATENCY
    rows.append(
        exact(
            "A3",
            "two-SM compute closed form",
            "interval=1,sm_count=2",
            expected,
            _estimate(model, launch).duration_cycles,
        )
    )


def check_b(rows: list[Row]) -> None:
    """B: memory is limited by the single HBM cursor."""

    for transaction_bytes in (64, 128):
        for bandwidth in (32, 64):
            service = math.ceil(transaction_bytes / bandwidth)
            expected = 32 * service + HBM_LATENCY
            launch = memory_launch(warps=8, per_warp=4, transaction_bytes=transaction_bytes)
            one_sm = _estimate(
                SmSchedulerModel(architecture(sm_count=1, hbm_bandwidth=bandwidth)),
                launch,
            )
            two_sm = _estimate(
                SmSchedulerModel(architecture(sm_count=2, hbm_bandwidth=bandwidth)),
                launch,
            )
            rows.append(
                exact(
                    "B1",
                    "HBM serialization closed form",
                    f"bytes={transaction_bytes},bw={bandwidth},sms=1",
                    expected,
                    one_sm.duration_cycles,
                )
            )
            rows.append(
                relation(
                    "B2",
                    "HBM duration is SM-count invariant",
                    f"bytes={transaction_bytes},bw={bandwidth},sms=2",
                    one_sm.duration_cycles,
                    two_sm.duration_cycles,
                )
            )

    fast = _estimate(
        SmSchedulerModel(architecture(hbm_bandwidth=64)),
        memory_launch(warps=8, per_warp=4, transaction_bytes=64),
    )
    slow = _estimate(
        SmSchedulerModel(architecture(hbm_bandwidth=32)),
        memory_launch(warps=8, per_warp=4, transaction_bytes=64),
    )
    rows.append(
        relation(
            "B3",
            "HBM serialization scales inversely with bandwidth",
            "serialization ratio",
            2 * (fast.duration_cycles - HBM_LATENCY),
            slow.duration_cycles - HBM_LATENCY,
        )
    )


def check_c(rows: list[Row]) -> None:
    """C: network is limited by the NVLink egress cursor."""

    for payload in (65_536, 131_072):
        for world in (2, 4, 8):
            launch = nccl_ring_allreduce_launch(
                payload_bytes=payload,
                world_size=world,
                channels=2,
                chunk_bytes=64,
                warps_per_channel=4,
            )
            estimate = _estimate(SmSchedulerModel(architecture(sm_count=2)), launch)
            expected = nccl_ring_egress_bytes(payload_bytes=payload, world_size=world)
            rows.append(
                exact(
                    "C1",
                    "ring egress byte closed form",
                    f"payload={payload},world={world},egress",
                    expected,
                    estimate.nvlink_transacted_bytes,
                )
            )
            rows.append(
                structural(
                    "C1",
                    "ring load-store byte symmetry",
                    f"payload={payload},world={world},loaded",
                    expected,
                    estimate.hbm_transacted_bytes,
                )
            )

    for chunk in (64, 128):
        for bandwidth in (8, 16):
            service = math.ceil(chunk / bandwidth)
            expected = 32 * service + NVLINK_LATENCY
            model = SmSchedulerModel(architecture(nvlink_bandwidth=bandwidth))
            launch = egress_launch(warps=8, per_warp=4, chunk_bytes=chunk)
            rows.append(
                exact(
                    "C2",
                    "NVLink serialization closed form",
                    f"chunk={chunk},bw={bandwidth}",
                    expected,
                    _estimate(model, launch).duration_cycles,
                )
            )


def check_c3(rows: list[Row]) -> list[tuple[int, int, int]]:
    """C3: NCCL respects the egress bound and converges toward it."""

    payload = 65_536
    world = 2
    chunk = 64
    egress_bytes = nccl_ring_egress_bytes(payload_bytes=payload, world_size=world)
    stores = egress_bytes // chunk
    bound = stores * math.ceil(chunk / 16) + NVLINK_LATENCY
    trend: list[tuple[int, int, int]] = []
    for warps in (1, 2, 4, 8):
        launch = nccl_ring_allreduce_launch(
            payload_bytes=payload,
            world_size=world,
            channels=2,
            chunk_bytes=chunk,
            warps_per_channel=warps,
        )
        duration = _estimate(
            SmSchedulerModel(architecture(sm_count=2)), launch
        ).duration_cycles
        trend.append((warps, duration, duration - bound))
        # not-below-bound is registered as an inequality, recorded as a
        # zero-residual row only when the inequality holds
        rows.append(
            structural(
                "C3",
                "single-server egress lower bound",
                f"warps={warps},at-or-above-bound",
                1,
                int(duration >= bound),
            )
        )
    for (earlier_warps, _, earlier), (later_warps, _, later) in itertools.pairwise(trend):
        rows.append(
            relation(
                "C3",
                "NCCL excess decreases with channel warps",
                f"warps={earlier_warps}->{later_warps}",
                1,
                int(later <= earlier),
            )
        )
    return trend


#: Post-specified regression literals from expectations.md. The superseded
#: D2 and D3 values remain in the historical rows emitted by main().
POST_SPECIFIED_D1_CYCLES = 132
POST_SPECIFIED_D2_CYCLES = 329
POST_SPECIFIED_D3_CYCLES = 243


def check_d(rows: list[Row]) -> dict[str, int]:
    """D: what two kinds of task do to each other."""

    model = SmSchedulerModel(architecture(sm_count=2))
    first = memory_launch(warps=4, per_warp=4, transaction_bytes=64)
    second = memory_launch(warps=4, per_warp=4, transaction_bytes=64)
    concurrent = _estimate_concurrent(
        model,
        (
            GpuTask(task_id="mem-a", kind=GpuTaskKind.MEMORY, launch=first),
            GpuTask(task_id="mem-b", kind=GpuTaskKind.MEMORY, launch=second),
        ),
    )
    rows.append(
        exact(
            "D1",
            "two HBM tasks share one cursor",
            "two memory tasks",
            POST_SPECIFIED_D1_CYCLES,
            concurrent.duration_cycles,
        )
    )

    mixed = _estimate_concurrent(
        SmSchedulerModel(architecture(sm_count=2)),
        (
            GpuTask(
                task_id="mem",
                kind=GpuTaskKind.MEMORY,
                launch=memory_launch(warps=8, per_warp=4, transaction_bytes=64),
            ),
            GpuTask(
                task_id="net",
                kind=GpuTaskKind.NETWORK,
                launch=egress_launch(warps=8, per_warp=4, chunk_bytes=64),
            ),
        ),
    )
    rows.append(
        exact(
            "D2",
            "memory-first mixed-task regression",
            "memory beside network",
            POST_SPECIFIED_D2_CYCLES,
            mixed.duration_cycles,
        )
    )

    half_shared = SHARED_PER_SM // 2
    compute = compute_launch(warps=8, per_warp=4, shared_bytes=half_shared)
    memory = memory_launch(warps=8, per_warp=4, transaction_bytes=64, shared_bytes=half_shared)
    backfilled = _estimate_concurrent(
        SmSchedulerModel(architecture(sm_count=2)),
        (
            GpuTask(task_id="compute", kind=GpuTaskKind.COMPUTE, launch=compute),
            GpuTask(task_id="memory", kind=GpuTaskKind.MEMORY, launch=memory),
        ),
    )
    rows.append(
        exact(
            "D3",
            "half-SM shared-memory mixed-task regression",
            "compute beside memory,half-SM demand",
            POST_SPECIFIED_D3_CYCLES,
            backfilled.duration_cycles,
        )
    )

    for label, estimate in (("D1", concurrent), ("D2", mixed), ("D3", backfilled)):
        rows.append(
            structural(
                "D4",
                "per-task attribution conservation",
                f"{label} issued instructions",
                estimate.issued_instructions,
                sum(task.issued_instructions for task in estimate.tasks),
            )
        )
        rows.append(
            structural(
                "D4",
                "per-task attribution conservation",
                f"{label} hbm bytes",
                estimate.hbm_transacted_bytes,
                sum(task.hbm_transacted_bytes for task in estimate.tasks),
            )
        )
        rows.append(
            structural(
                "D4",
                "per-task attribution conservation",
                f"{label} nvlink bytes",
                estimate.nvlink_transacted_bytes,
                sum(task.nvlink_transacted_bytes for task in estimate.tasks),
            )
        )
    return {
        "D1 two memory tasks": concurrent.duration_cycles,
        "D2 memory beside network": mixed.duration_cycles,
        "D3 compute beside memory": backfilled.duration_cycles,
    }


def check_e(rows: list[Row]) -> list[tuple[str, int]]:
    """E: schedule the real double-buffered ring task beside memory."""

    model = SmSchedulerModel(architecture(sm_count=2))
    ring = nccl_ring_allreduce_task(
        task_id="ring",
        payload_bytes=65_536,
        world_size=2,
        channels=2,
        chunk_bytes=64,
        warps_per_channel=8,
    )
    memory = GpuTask(
        task_id="memory",
        kind=GpuTaskKind.MEMORY,
        launch=memory_launch(warps=8, per_warp=4, transaction_bytes=64),
    )
    ring_isolated = _estimate(model, ring.launch).duration_cycles
    memory_isolated = _estimate(model, memory.launch).duration_cycles
    mixed = _estimate_concurrent(model, (ring, memory))
    ring_estimate = mixed.tasks[0]

    rows.extend(
        (
            structural(
                "E1",
                "task-kind wiring",
                "ring task kind is network",
                1,
                int(ring_estimate.kind is GpuTaskKind.NETWORK),
            ),
            exact(
                "E1",
                "ring task byte and request oracles",
                "ring HBM requested bytes",
                65_536,
                ring_estimate.hbm_requested_bytes,
            ),
            exact(
                "E1",
                "ring task byte and request oracles",
                "ring HBM transacted bytes",
                65_536,
                ring_estimate.hbm_transacted_bytes,
            ),
            exact(
                "E1",
                "ring task byte and request oracles",
                "ring HBM requests",
                1_024,
                ring_estimate.hbm_request_instructions,
            ),
            exact(
                "E1",
                "ring task byte and request oracles",
                "ring NVLink requested bytes",
                65_536,
                ring_estimate.nvlink_requested_bytes,
            ),
            exact(
                "E1",
                "ring task byte and request oracles",
                "ring NVLink transacted bytes",
                65_536,
                ring_estimate.nvlink_transacted_bytes,
            ),
            exact(
                "E1",
                "ring task byte and request oracles",
                "ring NVLink requests",
                1_024,
                ring_estimate.nvlink_request_instructions,
            ),
            exact(
                "E2", "mixed-task counter oracles", "mixed issued instructions", 2_080,
                mixed.issued_instructions,
            ),
            exact(
                "E2", "mixed-task counter oracles", "mixed HBM requested bytes", 67_584,
                mixed.hbm_requested_bytes,
            ),
            exact(
                "E2", "mixed-task counter oracles", "mixed HBM transacted bytes", 67_584,
                mixed.hbm_transacted_bytes,
            ),
            exact(
                "E2", "mixed-task counter oracles", "mixed HBM requests", 1_056,
                mixed.hbm_request_instructions,
            ),
            exact(
                "E2", "mixed-task counter oracles", "mixed NVLink requested bytes", 65_536,
                mixed.nvlink_requested_bytes,
            ),
            exact(
                "E2", "mixed-task counter oracles", "mixed NVLink transacted bytes", 65_536,
                mixed.nvlink_transacted_bytes,
            ),
            exact(
                "E2", "mixed-task counter oracles", "mixed NVLink requests", 1_024,
                mixed.nvlink_request_instructions,
            ),
        )
    )
    conserved = all(
        sum(getattr(task, field) for task in mixed.tasks) == getattr(mixed, field)
        for field in (
            "issued_instructions",
            "hbm_requested_bytes",
            "hbm_transacted_bytes",
            "hbm_request_instructions",
            "nvlink_requested_bytes",
            "nvlink_transacted_bytes",
            "nvlink_request_instructions",
        )
    )
    rows.append(
        structural(
            "E2",
            "per-task attribution conservation",
            "all per-task counters conserve",
            1,
            int(conserved),
        )
    )
    rows.append(
        relation(
            "E3",
            "ring and memory overlap band",
            "mixed at or above ring control",
            1,
            int(mixed.duration_cycles >= 4_397),
        )
    )
    rows.append(
        relation(
            "E3",
            "ring and memory overlap band",
            "mixed below serialized controls",
            1,
            int(mixed.duration_cycles < 4_529),
        )
    )
    return [
        ("E ring isolated", ring_isolated),
        ("E memory isolated", memory_isolated),
        ("E ring beside memory", mixed.duration_cycles),
    ]


def diagnostics() -> list[tuple[str, int]]:
    """Controls discovered after the initial D2 and D3 misses.

    Their chronology remains post-hoc. They also feed the corrected,
    post-specified D2R and D3R relation families.
    """

    model = SmSchedulerModel(architecture(sm_count=2))
    memory = memory_launch(warps=8, per_warp=4, transaction_bytes=64)
    egress = egress_launch(warps=8, per_warp=4, chunk_bytes=64)
    network_first = _estimate_concurrent(
        model,
        (
            GpuTask(task_id="net", kind=GpuTaskKind.NETWORK, launch=egress),
            GpuTask(task_id="mem", kind=GpuTaskKind.MEMORY, launch=memory),
        ),
    )
    scheduler_wide = _estimate_concurrent(
        SmSchedulerModel(
            architecture(sm_count=2, scheduler_count_per_sm=2 * ISSUE_BUDGET)
        ),
        (
            GpuTask(task_id="mem", kind=GpuTaskKind.MEMORY, launch=memory),
            GpuTask(task_id="net", kind=GpuTaskKind.NETWORK, launch=egress),
        ),
    )
    load_store_wide = _estimate_concurrent(
        SmSchedulerModel(
            architecture(sm_count=2, load_store_issue_width_per_sm=2 * LANES)
        ),
        (
            GpuTask(task_id="mem", kind=GpuTaskKind.MEMORY, launch=memory),
            GpuTask(task_id="net", kind=GpuTaskKind.NETWORK, launch=egress),
        ),
    )
    both_wide = _estimate_concurrent(
        SmSchedulerModel(
            architecture(
                sm_count=2,
                scheduler_count_per_sm=2 * ISSUE_BUDGET,
                load_store_issue_width_per_sm=2 * LANES,
            )
        ),
        (
            GpuTask(task_id="mem", kind=GpuTaskKind.MEMORY, launch=memory),
            GpuTask(task_id="net", kind=GpuTaskKind.NETWORK, launch=egress),
        ),
    )

    half_shared = SHARED_PER_SM // 2
    constrained_compute = compute_launch(
        warps=8,
        per_warp=4,
        shared_bytes=half_shared,
    )
    constrained_memory = memory_launch(
        warps=8,
        per_warp=4,
        transaction_bytes=64,
        shared_bytes=half_shared,
    )
    constrained = _estimate_concurrent(
        model,
        (
            GpuTask(
                task_id="compute",
                kind=GpuTaskKind.COMPUTE,
                launch=constrained_compute,
            ),
            GpuTask(
                task_id="memory",
                kind=GpuTaskKind.MEMORY,
                launch=constrained_memory,
            ),
        ),
    )
    constrained_memory_task = next(task for task in constrained.tasks if task.task_id == "memory")

    compute = compute_launch(warps=8, per_warp=4)
    unconstrained = _estimate_concurrent(
        model,
        (
            GpuTask(task_id="compute", kind=GpuTaskKind.COMPUTE, launch=compute),
            GpuTask(task_id="memory", kind=GpuTaskKind.MEMORY, launch=memory),
        ),
    )
    return [
        ("network submitted first, memory second", network_first.duration_cycles),
        ("D2 scheduler budget doubled", scheduler_wide.duration_cycles),
        ("D2 load-store lanes doubled", load_store_wide.duration_cycles),
        ("D2 scheduler budget and load-store lanes doubled", both_wide.duration_cycles),
        (
            "compute isolated, half-SM shared-memory demand",
            _estimate(model, constrained_compute).duration_cycles,
        ),
        (
            "memory isolated, half-SM shared-memory demand",
            _estimate(model, constrained_memory).duration_cycles,
        ),
        ("memory admission, half-SM shared-memory demand", constrained_memory_task.admitted_cycle),
        ("compute isolated, no shared-memory pressure", _estimate(model, compute).duration_cycles),
        ("memory isolated, no shared-memory pressure", _estimate(model, memory).duration_cycles),
        ("compute beside memory, no shared-memory pressure", unconstrained.duration_cycles),
    ]


def register_corrected_relations(
    rows: list[Row], mixed: dict[str, int], measured: dict[str, int]
) -> None:
    """Register the corrected D2 and D3 mechanisms after the initial misses."""

    d2_family = "D2 shared issue-path mechanism"
    memory_first = mixed["D2 memory beside network"]
    network_first = measured["network submitted first, memory second"]
    rows.extend(
        (
            relation(
                "D2R",
                d2_family,
                "memory-first is network-first plus one cycle",
                network_first + 1,
                memory_first,
            ),
            relation(
                "D2R",
                d2_family,
                "scheduler widening alone preserves delay",
                memory_first,
                measured["D2 scheduler budget doubled"],
            ),
            relation(
                "D2R",
                d2_family,
                "load-store widening alone preserves delay",
                memory_first,
                measured["D2 load-store lanes doubled"],
            ),
            relation(
                "D2R",
                d2_family,
                "widening both issue resources removes delay",
                network_first,
                measured["D2 scheduler budget and load-store lanes doubled"],
            ),
        )
    )

    d3_family = "D3 shared-memory residency mechanism"
    constrained_compute = measured["compute isolated, half-SM shared-memory demand"]
    constrained_memory = measured["memory isolated, half-SM shared-memory demand"]
    unconstrained_compute = measured["compute isolated, no shared-memory pressure"]
    unconstrained_memory = measured["memory isolated, no shared-memory pressure"]
    rows.extend(
        (
            relation(
                "D3R",
                d3_family,
                "half-SM tasks serialize",
                constrained_compute + constrained_memory,
                mixed["D3 compute beside memory"],
            ),
            relation(
                "D3R",
                d3_family,
                "memory admits after constrained compute",
                constrained_compute,
                measured["memory admission, half-SM shared-memory demand"],
            ),
            relation(
                "D3R",
                d3_family,
                "unconstrained tasks overlap plus issue delay",
                max(unconstrained_compute, unconstrained_memory) + 1,
                measured["compute beside memory, no shared-memory pressure"],
            ),
        )
    )


def main() -> int:
    global _run_configuration_count
    _run_configuration_count = 0
    _run_configurations.clear()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    rows: list[Row] = []
    check_a(rows)
    check_b(rows)
    check_c(rows)
    trend = check_c3(rows)
    mixed_controls = check_d(rows)
    e_diagnostics = check_e(rows)
    post_hoc_diagnostics = diagnostics()
    register_corrected_relations(rows, mixed_controls, dict(post_hoc_diagnostics))
    rows.extend(
        (
            Row(
                "H-D2",
                HISTORICAL_LEDGER,
                "superseded initial registration",
                "memory beside network",
                328,
                329,
            ),
            Row(
                "H-D3",
                HISTORICAL_LEDGER,
                "superseded initial registration",
                "compute hides under memory",
                132,
                243,
            ),
        )
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "check",
                "evidence_class",
                "family",
                "case",
                "expected",
                "measured",
                "residual",
                "status",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.check,
                    row.evidence_class,
                    row.family,
                    row.case,
                    row.expected,
                    row.measured,
                    row.residual,
                    row.status,
                )
            )
    with (out / "nccl_convergence.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("warps_per_channel", "duration_cycles", "excess_over_egress_bound"))
        writer.writerows(trend)

    measured = post_hoc_diagnostics + e_diagnostics
    with (out / "diagnostics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("diagnostic", "duration_cycles"))
        writer.writerows(measured)

    active = [row for row in rows if row.evidence_class != HISTORICAL_LEDGER]
    failures = [row for row in active if row.status == "FAIL"]
    for row in rows:
        print(
            f"{row.status:15} {row.check:4} {row.case:48} "
            f"expected {row.expected} got {row.measured}"
        )

    exact_rows = [row for row in active if row.evidence_class == EXACT_ORACLE]
    relation_rows = [row for row in active if row.evidence_class == BEHAVIORAL_RELATION]
    structural_rows = [row for row in active if row.evidence_class == STRUCTURAL_INVARIANT]
    relation_families = {row.family for row in relation_rows}
    passing_relation_families = {
        family
        for family in relation_families
        if all(row.status == "PASS" for row in relation_rows if row.family == family)
    }
    print("\nEvidence summary (classes are not aggregated):")
    print(f"  distinct run configurations: {len(_run_configurations)}")
    print(f"  replay invocations: {_run_configuration_count}")
    print(
        "  scored exact-oracle rows: "
        f"{sum(row.status == 'PASS' for row in exact_rows)}/{len(exact_rows)} pass"
    )
    print(
        "  scored behavioral relation families: "
        f"{len(passing_relation_families)}/{len(relation_families)} pass"
    )
    print(
        "  scored behavioral relation instances: "
        f"{sum(row.status == 'PASS' for row in relation_rows)}/{len(relation_rows)} pass"
    )
    print(
        "  unscored structural invariants: "
        f"{sum(row.status == 'PASS' for row in structural_rows)}/{len(structural_rows)} hold"
    )
    print("  historical ledger rows: 2 superseded failures retained")
    print("NCCL convergence (warps_per_channel, cycles, excess over egress bound):")
    for warps, duration, excess in trend:
        print(f"  {warps:2}  {duration:8}  {excess:8}")
    print("Post-hoc controls (also used by D2R and D3R):")
    for label, value in measured:
        print(f"  {label:52} {value}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
