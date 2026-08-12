"""Check that the coarse runtime hands the arbitrated order to the GPU service.

CORE-49: `_compute_group` rebuilt the co-runnable set in `ExecutionGraph` tuple
order and never consulted the arbitration policy, so a class-aware policy could
win the ready seam while the compute service still replayed graph order.
CORE-10: strict priority and weighted round robin are the first two non-identity
policies, and they are what makes that difference observable at all.

Everything this runner asserts was frozen in `expectations.md` before any
implementation existed. The synthetic 1 GHz fixture is a mechanism fixture
replicated from the published COMP-12 registration, never a silicon claim.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Frozen literals. Every value below is quoted from expectations.md and must
# never be edited to match an observation.
# ---------------------------------------------------------------------------

PS_PER_CYCLE = 1_000

ISSUE_BUDGET = 4
LANES = 4
ALU_LANES = 4
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

#: launch name -> isolated duration, measured and published by COMP-12
FROZEN_ISOLATED_CYCLES: dict[str, int] = {"memory": 132, "network": 328, "compute": 7}

FIXTURE_F2_ORDER = ("memory", "network")
FIXTURE_F3_ORDER = ("memory", "network", "compute")

#: (issue budget, lanes) -> identity baseline ordered tuple and step JCT on F2
FROZEN_F2_IDENTITY_ORDER = FIXTURE_F2_ORDER
FROZEN_F2_IDENTITY_JCT_PS: dict[tuple[int, int], int] = {
    (4, 4): 329_000,
    (8, 8): 328_000,
}
FROZEN_F3_IDENTITY_ORDER = FIXTURE_F3_ORDER

#: family A, evaluated on fixture F2 in every step
#: instance -> (policy spec, labels, issue budget, lanes, ordered tuple, JCT ps)
FROZEN_FAMILY_A: dict[str, tuple[Any, dict[str, int], int, int, tuple[str, ...], int]] = {
    "A1": (
        ("strict", {}),
        {"memory": 2, "network": 1},
        4,
        4,
        ("network", "memory"),
        328_000,
    ),
    "A2": (
        ("strict", {}),
        {"memory": 1, "network": 2},
        4,
        4,
        ("memory", "network"),
        329_000,
    ),
    "A3": (
        ("strict", {}),
        {"memory": 2, "network": 1},
        8,
        8,
        ("network", "memory"),
        328_000,
    ),
    "A4": (
        ("wrr", {"weights": {1: 2, 2: 1}}),
        {"memory": 2, "network": 1},
        4,
        4,
        ("network", "memory"),
        328_000,
    ),
    "A5": (
        ("wrr", {"weights": {1: 2, 2: 1}}),
        {"memory": 1, "network": 2},
        4,
        4,
        ("memory", "network"),
        329_000,
    ),
}

#: family B, evaluated on fixture F3 at issue budget 4 and lanes 4
#: instance -> (policy spec, labels, per-step ordered tuples)
FROZEN_FAMILY_B: dict[str, tuple[Any, dict[str, int], tuple[tuple[str, ...], ...]]] = {
    "B1": (
        ("strict", {}),
        {"memory": 2, "network": 1, "compute": 1},
        (
            ("network", "compute", "memory"),
            ("network", "compute", "memory"),
            ("network", "compute", "memory"),
        ),
    ),
    "B2": (
        ("wrr", {"weights": {1: 2, 2: 1}}),
        {"memory": 2, "network": 1, "compute": 1},
        (
            ("network", "memory", "compute"),
            ("network", "compute", "memory"),
            ("network", "memory", "compute"),
        ),
    ),
    "B3": (
        ("strict", {}),
        {"memory": 3, "network": 2, "compute": 1},
        (
            ("compute", "network", "memory"),
            ("compute", "network", "memory"),
            ("compute", "network", "memory"),
        ),
    ),
}

#: label -> (first-principles floor cycles, serialized ceiling cycles)
FROZEN_PHYSICAL_INTERVALS: dict[str, tuple[int, int]] = {
    "F2 concurrent": (328, 460),
    "F3 concurrent": (328, 467),
    "F2 dependent": (460, 460),
}

FROZEN_SCALAR_NOMINAL_PS = {"memory": 132_000, "network": 328_000}
FROZEN_SCALAR_JCT_PS = 328_000
FROZEN_DEPENDENT_JCT_PS = 460_000

T0_PS = 5_000
LIVE_REQUEST_ID = "arbitrated-order-request"
LIVE_STEP_COUNT = 3

EXPECTATIONS_COMMIT = "9d89d513baec9785093e8d95671051c78447379a"
AUTHORED_AGAINST_COMMIT = "aeb40ac95cdd8163942297335948c94df0376e04"

# Evidence classes. Only GENUINE_RISK rows enter a behavioral fraction.
GENUINE_RISK = "behavioral-relation"
FATAL_GUARD = "fatal-guard"
RAW_OBSERVATION = "raw-observation"
RUN_CONFIGURATION = "run-configuration"

FAMILY_A = "A: the arbitrated order reaches the compute service"
FAMILY_B = "B: class-aware policies order by their own contract"
SCORED_FAMILIES = (FAMILY_A, FAMILY_B)


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


def _strict_priority_order(
    names: tuple[str, ...],
    labels: dict[str, int],
) -> tuple[str, ...]:
    """Reference strict-priority total order over one candidate set."""

    baseline = {name: index for index, name in enumerate(names)}
    return tuple(sorted(names, key=lambda name: (labels[name], baseline[name])))


class _ReferenceWeightedRoundRobin:
    """Reference weighted round robin used only to re-derive frozen literals."""

    def __init__(self, weights: dict[int, int], default_weight: int = 1) -> None:
        self._weights = dict(weights)
        self._default_weight = default_weight
        self._remaining: dict[int, int] = {}

    def weight(self, label: int) -> int:
        return self._weights.get(label, self._default_weight)

    def select(
        self,
        names: tuple[str, ...],
        labels: dict[str, int],
        baseline: dict[str, int],
    ) -> str:
        present = sorted({labels[name] for name in names})
        eligible = [
            label
            for label in present
            if self._remaining.get(label, self.weight(label)) > 0
        ]
        if not eligible:
            for label in present:
                self._remaining[label] = self.weight(label)
            eligible = list(present)
        winner = eligible[0]
        self._remaining[winner] = (
            self._remaining.get(winner, self.weight(winner)) - 1
        )
        return min(
            (name for name in names if labels[name] == winner),
            key=lambda name: baseline[name],
        )


def _reference_wrr_steps(
    graph_order: tuple[str, ...],
    labels: dict[str, int],
    weights: dict[int, int],
    steps: int,
) -> tuple[tuple[str, ...], ...]:
    """Re-derive the frozen weighted-round-robin orders from the grant model."""

    baseline = {name: index for index, name in enumerate(graph_order)}
    policy = _ReferenceWeightedRoundRobin(weights)
    observed: list[tuple[str, ...]] = []
    for _ in range(steps):
        # one ready-seam grant offering every ready operation
        policy.select(graph_order, labels, baseline)
        remaining = list(graph_order)
        order: list[str] = []
        while remaining:
            granted = policy.select(tuple(remaining), labels, baseline)
            remaining.remove(granted)
            order.append(granted)
        observed.append(tuple(order))
    return tuple(observed)


def check_only() -> None:
    """Validate the frozen literals against each other and exit."""

    for name in FIXTURE_F2_ORDER + FIXTURE_F3_ORDER:
        if name not in FROZEN_ISOLATED_CYCLES:
            raise ValueError(f"fixture names an unregistered launch {name!r}")

    f2_floor = max(FROZEN_ISOLATED_CYCLES[name] for name in FIXTURE_F2_ORDER)
    f2_ceiling = sum(FROZEN_ISOLATED_CYCLES[name] for name in FIXTURE_F2_ORDER)
    if FROZEN_PHYSICAL_INTERVALS["F2 concurrent"] != (f2_floor, f2_ceiling):
        raise ValueError("frozen F2 interval is not the isolated-control bound")
    f3_floor = max(FROZEN_ISOLATED_CYCLES[name] for name in FIXTURE_F3_ORDER)
    f3_ceiling = sum(FROZEN_ISOLATED_CYCLES[name] for name in FIXTURE_F3_ORDER)
    if FROZEN_PHYSICAL_INTERVALS["F3 concurrent"] != (f3_floor, f3_ceiling):
        raise ValueError("frozen F3 interval is not the isolated-control bound")
    if FROZEN_PHYSICAL_INTERVALS["F2 dependent"] != (f2_ceiling, f2_ceiling):
        raise ValueError("frozen dependent interval is not the serialized sum")
    if FROZEN_DEPENDENT_JCT_PS != f2_ceiling * PS_PER_CYCLE:
        raise ValueError("frozen dependent JCT is not the serialized sum")

    for key, jct in FROZEN_F2_IDENTITY_JCT_PS.items():
        cycles = jct // PS_PER_CYCLE
        if jct % PS_PER_CYCLE or not f2_floor <= cycles <= f2_ceiling:
            raise ValueError(f"frozen identity JCT at {key} leaves the F2 interval")
    if FROZEN_F2_IDENTITY_JCT_PS[(4, 4)] - FROZEN_F2_IDENTITY_JCT_PS[(8, 8)] != (
        PS_PER_CYCLE
    ):
        raise ValueError("frozen identity baseline does not carry one issue cycle")

    for instance, (spec, labels, budget, lanes, order, jct) in FROZEN_FAMILY_A.items():
        if tuple(sorted(order)) != tuple(sorted(FIXTURE_F2_ORDER)):
            raise ValueError(f"{instance} is not a permutation of fixture F2")
        if set(labels) != set(FIXTURE_F2_ORDER):
            raise ValueError(f"{instance} does not label every F2 operation")
        if budget not in (4, 8) or lanes not in (4, 8):
            raise ValueError(f"{instance} leaves the frozen 4/8 issue grid")
        cycles = jct // PS_PER_CYCLE
        if jct % PS_PER_CYCLE or not f2_floor <= cycles <= f2_ceiling:
            raise ValueError(f"{instance} JCT leaves the F2 physical interval")
        # a first-submitted `memory` costs the registered one-cycle issue delay
        # only while both per-SM issue currencies are narrow
        delay = 1 if (order[0] == "memory" and (budget, lanes) == (4, 4)) else 0
        if cycles != f2_floor + delay:
            raise ValueError(f"{instance} JCT contradicts the COMP-12 issue term")
        if spec[0] == "strict":
            if _strict_priority_order(FIXTURE_F2_ORDER, labels) != order:
                raise ValueError(f"{instance} contradicts strict priority")
        elif spec[0] == "wrr":
            derived = _reference_wrr_steps(
                FIXTURE_F2_ORDER,
                labels,
                spec[1]["weights"],
                LIVE_STEP_COUNT,
            )
            if set(derived) != {order}:
                raise ValueError(f"{instance} contradicts the weighted grant model")
        else:
            raise ValueError(f"{instance} names an unregistered policy {spec[0]!r}")

    if FROZEN_FAMILY_A["A1"][4] == FROZEN_F2_IDENTITY_ORDER:
        raise ValueError("A1 must differ from the identity order to discriminate")
    if FROZEN_FAMILY_A["A1"][5] == FROZEN_F2_IDENTITY_JCT_PS[(4, 4)]:
        raise ValueError("A1 must move the identity step JCT")
    if FROZEN_FAMILY_A["A3"][5] != FROZEN_F2_IDENTITY_JCT_PS[(8, 8)]:
        raise ValueError("A3 must leave the widened baseline unmoved")

    for instance, (spec, labels, orders) in FROZEN_FAMILY_B.items():
        if len(orders) != LIVE_STEP_COUNT:
            raise ValueError(f"{instance} does not cover every registered step")
        for order in orders:
            if tuple(sorted(order)) != tuple(sorted(FIXTURE_F3_ORDER)):
                raise ValueError(f"{instance} is not a permutation of fixture F3")
        if set(labels) != set(FIXTURE_F3_ORDER):
            raise ValueError(f"{instance} does not label every F3 operation")
        if spec[0] == "strict":
            expected = _strict_priority_order(FIXTURE_F3_ORDER, labels)
            if orders != (expected,) * LIVE_STEP_COUNT:
                raise ValueError(f"{instance} contradicts strict priority")
        elif spec[0] == "wrr":
            derived = _reference_wrr_steps(
                FIXTURE_F3_ORDER,
                labels,
                spec[1]["weights"],
                LIVE_STEP_COUNT,
            )
            if derived != orders:
                raise ValueError(f"{instance} contradicts the weighted grant model")
        else:
            raise ValueError(f"{instance} names an unregistered policy {spec[0]!r}")

    if FROZEN_FAMILY_B["B1"][2] == FROZEN_FAMILY_B["B2"][2]:
        raise ValueError("B1 and B2 must separate strict priority from weighting")
    if FROZEN_FAMILY_B["B2"][2][0] == FROZEN_FAMILY_B["B2"][2][1]:
        raise ValueError("B2 must show the credit carry across arbitration rounds")
    for instance in ("B1", "B2", "B3"):
        if FROZEN_F3_IDENTITY_ORDER in FROZEN_FAMILY_B[instance][2]:
            raise ValueError(f"{instance} must differ from the identity order")

    if max(FROZEN_SCALAR_NOMINAL_PS.values()) != FROZEN_SCALAR_JCT_PS:
        raise ValueError("frozen scalar path is not the independent-resource maximum")
    for name, nominal in FROZEN_SCALAR_NOMINAL_PS.items():
        if nominal != FROZEN_ISOLATED_CYCLES[name] * PS_PER_CYCLE:
            raise ValueError("frozen scalar nominal contradicts the isolated control")

    print(
        "check-only validated the frozen arbitrated-order registry and arithmetic; "
        "no SimLLM import, no input read, no artifact written"
    )


# ---------------------------------------------------------------------------
# Frozen synthetic fixture.
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
    profile_id = f"arbitrated-order-profile-{variant}"
    calibration = GpuCalibrationProfile(
        calibration_id=f"arbitrated-order-calibration-{variant}",
        target_architecture_profile_id=profile_id,
        provenance=GpuModelProvenance(
            source="synthetic study fixture, no silicon claim",
            version="1",
            gpu="arbitrated-order-synthetic",
            created="2026-08-13",
        ),
        core_clock_hz=1_000_000_000,
        target_memory_clock_hz=None,
        pipelines=(
            PipelineProfile(
                kind=PipelineKind.ALU,
                opcodes=("ALU",),
                latency_cycles=ALU_LATENCY,
                issue_width_per_sm=ALU_LANES,
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
        gpu_name="arbitrated-order-synthetic",
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


def _launch(*, name: str, instructions):
    from simllm.compute import CtaTrace, KernelLaunch, SassWarpTrace

    return KernelLaunch(
        implementation_id=name,
        trace_id=f"{name}-trace",
        grid_blocks=TASK_BLOCKS,
        threads_per_block=32,
        registers_per_thread=0,
        static_shared_memory_bytes=0,
        dynamic_shared_memory_bytes=0,
        cta_traces=(
            CtaTrace(
                trace_class_id=f"{name}-cta",
                block_ids=tuple(range(TASK_BLOCKS)),
                warp_traces=(SassWarpTrace(warp_id=0, instructions=instructions),),
            ),
        ),
    )


def compute_launch():
    from simllm.compute import PipelineKind, SassInstruction

    return _launch(
        name="compute",
        instructions=tuple(
            SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU)
            for _ in range(TASK_INSTRUCTIONS_PER_WARP)
        ),
    )


def memory_launch():
    from simllm.compute import MemorySpace, PipelineKind, SassInstruction

    return _launch(
        name="memory",
        instructions=tuple(
            SassInstruction(
                opcode="LDG",
                pipeline=PipelineKind.LOAD_STORE,
                memory_space=MemorySpace.HBM,
                requested_bytes=TRANSACTION_BYTES,
                transacted_bytes=TRANSACTION_BYTES,
            )
            for _ in range(TASK_INSTRUCTIONS_PER_WARP)
        ),
    )


def network_launch():
    from simllm.compute import MemorySpace, PipelineKind, SassInstruction

    return _launch(
        name="network",
        instructions=tuple(
            SassInstruction(
                opcode="STG",
                pipeline=PipelineKind.LOAD_STORE,
                memory_space=MemorySpace.NVLINK,
                requested_bytes=TRANSACTION_BYTES,
                transacted_bytes=TRANSACTION_BYTES,
            )
            for _ in range(TASK_INSTRUCTIONS_PER_WARP)
        ),
    )


def launches() -> dict[str, Any]:
    return {
        "memory": memory_launch(),
        "network": network_launch(),
        "compute": compute_launch(),
    }


# ---------------------------------------------------------------------------
# Observation instruments. Neither instrument changes runtime behavior: the
# service subclass delegates, and the policy wrapper delegates.
# ---------------------------------------------------------------------------


def recording_service(scheduler_count_per_sm: int, load_store_issue_width_per_sm: int):
    """Return an `SmSchedulerModel` that records every ordered tuple it gets."""

    from simllm.compute import SmSchedulerModel

    class _RecordingSmSchedulerModel(SmSchedulerModel):
        def __init__(self, arch) -> None:
            super().__init__(arch)
            self.received_orders: list[tuple[str, ...]] = []

        def estimate_concurrent(self, tasks):
            tasks = tuple(tasks)
            self.received_orders.append(tuple(task.task_id for task in tasks))
            return super().estimate_concurrent(tasks)

    return _RecordingSmSchedulerModel(
        architecture(
            scheduler_count_per_sm=scheduler_count_per_sm,
            load_store_issue_width_per_sm=load_store_issue_width_per_sm,
        )
    )


class RecordingPolicy:
    """Delegating `ArbitrationPolicy` that records every offer and grant."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.grants: list[tuple[tuple[str, ...], str]] = []

    def select(self, candidates):
        granted = self.inner.select(candidates)
        self.grants.append(
            (tuple(item.operation_id for item in candidates), granted.operation_id)
        )
        return granted


def build_policy(spec: tuple[str, dict[str, Any]] | None):
    """Return the policy object named by one frozen policy spec."""

    from simllm.core import (
        IdentityArbitrationPolicy,
        StrictPriorityArbitrationPolicy,
        WeightedRoundRobinArbitrationPolicy,
    )

    if spec is None:
        return None
    kind, kwargs = spec
    if kind == "identity":
        return IdentityArbitrationPolicy()
    if kind == "strict":
        return StrictPriorityArbitrationPolicy(**kwargs)
    if kind == "wrr":
        return WeightedRoundRobinArbitrationPolicy(**kwargs)
    raise ValueError(f"unregistered policy spec {kind!r}")


# ---------------------------------------------------------------------------
# Live cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveCase:
    """One live configuration: ordered compute operations on rank 0."""

    case_id: str
    order: tuple[str, ...]
    policy_spec: tuple[str, dict[str, Any]] | None = None
    labels: dict[str, int] = field(default_factory=dict)
    scheduler_count_per_sm: int = ISSUE_BUDGET
    load_store_issue_width_per_sm: int = LANES
    nominal_ps: dict[str, int] | None = None
    depends: dict[str, tuple[str, ...]] = field(default_factory=dict)


def _live_operations(case: LiveCase):
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
                depends_on=case.depends.get(name, ()),
                priority=case.labels.get(name, 0),
                correlation=correlation,
            )
        )
    return tuple(operations)


def _strip_label_echo(value: Any) -> Any:
    """Project out the passive echoes of the input class label.

    `RuntimeOperationRecord.class_label` repeats `ExecutionOperation.priority`
    and `RuntimeReport.class_service_bytes` attributes the same bytes per class
    label. Both are the input read back, not behavior, so amendment 1 removes
    them from the byte-identity comparison and conserves the byte total
    separately.
    """

    if isinstance(value, dict):
        return {
            key: _strip_label_echo(item)
            for key, item in value.items()
            if key not in ("class_label", "class_service_bytes")
        }
    if isinstance(value, list):
        return [_strip_label_echo(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_label_echo(item) for item in value]
    return value


def _canonical(result, report, case_id: str) -> str:
    """Serialize one execution deterministically for byte-identity checks.

    The case identifier is the only other value allowed to differ between two
    identity settings, because it names the run rather than its behavior, so it
    is replaced by a placeholder before the comparison.
    """

    from simllm.core import execution_result_to_json

    payload = {
        "result": execution_result_to_json(result),
        "report": _strip_label_echo(dataclasses.asdict(report)),
    }
    return json.dumps(payload, sort_keys=True, default=str).replace(case_id, "case")


def run_live_case(case: LiveCase) -> dict[str, Any]:
    """Execute one live case through CORE-4 and the request-metric chain."""

    from simllm.core import (
        CoarseDeviceRuntime,
        CompletionReducer,
        EventPhase,
        ExecutionGraph,
        RequestPhase,
        ResourceKind,
        ScheduledRequest,
        StepRecord,
        VirtualClock,
    )

    kwargs: dict[str, Any] = {}
    service = None
    if case.nominal_ps is None:
        service = recording_service(
            case.scheduler_count_per_sm,
            case.load_store_issue_width_per_sm,
        )
        kwargs["kernel_services"] = {0: service}
        kwargs["kernel_launches"] = {
            name: launch for name, launch in launches().items() if name in case.order
        }
    policy = build_policy(case.policy_spec)
    recorder = None
    if policy is not None:
        recorder = RecordingPolicy(policy)
        kwargs["arbitration_policy"] = recorder
    runtime = CoarseDeviceRuntime(**kwargs)

    clock = VirtualClock(T0_PS)
    reducer = CompletionReducer(clock)
    steps: list[dict[str, Any]] = []
    for index in range(LIVE_STEP_COUNT):
        released_at_ps = clock.now_ps
        grants_before = 0 if recorder is None else len(recorder.grants)
        orders_before = 0 if service is None else len(service.received_orders)
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

        gpu_visits = {
            visit.operation_id: visit
            for visit in report.visits
            if visit.resource.kind is ResourceKind.GPU_WORK_QUEUE
        }

        def _key(operation_id, resource, timestamp):
            return (operation_id, resource.kind, resource.resource_id, timestamp)

        eligible_keys = {
            _key(visit.operation_id, visit.resource, visit.eligible_at_ps)
            for visit in report.visits
            if visit.subject_object_id is None
        }
        granted_keys = {
            _key(visit.operation_id, visit.resource, visit.started_at_ps)
            for visit in report.visits
            if visit.subject_object_id is None
        }
        queued_matches_eligibility = all(
            _key(event.operation_id, event.resource, event.timestamp_ps)
            in eligible_keys
            for event in result.events
            if event.phase is EventPhase.QUEUED and event.subject_object_id is None
        )
        started_matches_grant = all(
            _key(event.operation_id, event.resource, event.timestamp_ps) in granted_keys
            for event in result.events
            if event.phase is EventPhase.STARTED and event.subject_object_id is None
        )
        completed_events = [
            event
            for event in result.events
            if event.phase is EventPhase.COMPLETED and event.subject_object_id is None
        ]
        operation_completion = {
            row.operation_id: row.completed_at_ps for row in report.operations
        }
        steps.append(
            {
                "step_index": index,
                "released_at_ps": released_at_ps,
                "graph_order": list(case.order),
                "service_orders": (
                    []
                    if service is None
                    else [list(row) for row in service.received_orders[orders_before:]]
                ),
                "grants": (
                    []
                    if recorder is None
                    else [
                        {"offered": list(offered), "granted": granted}
                        for offered, granted in recorder.grants[grants_before:]
                    ]
                ),
                "step_latency_ps": step.step_latency_ps,
                "step_completed_at_ps": step.completed_at_ps,
                "execution_completed_at_ps": result.completed_at_ps,
                "operation_started_at_ps": {
                    operation_id: visit.started_at_ps
                    for operation_id, visit in gpu_visits.items()
                },
                "operation_completed_at_ps": operation_completion,
                "ttft_ps": metric.ttft_ps,
                "tpot_ps": None if metric.tpot_ps is None else int(metric.tpot_ps),
                "metric_completed_at_ps": metric.completed_at_ps,
                "metric_latency_ps": metric.latency_ps,
                "queued_matches_eligibility": queued_matches_eligibility,
                "started_matches_grant": started_matches_grant,
                "completed_event_count": len(completed_events),
                "completed_events_match_operations": sorted(
                    (event.operation_id, event.timestamp_ps)
                    for event in completed_events
                )
                == sorted(operation_completion.items()),
                "canonical": _canonical(result, report, case.case_id),
                "total_class_service_bytes": sum(
                    byte_count for _, byte_count in report.class_service_bytes
                ),
            }
        )
    return {
        "case_id": case.case_id,
        "policy_spec": None if case.policy_spec is None else list(case.policy_spec[0:1]),
        "steps": steps,
    }


def _orders(observed: dict[str, Any]) -> list[tuple[str, ...]]:
    """Return the ordered tuple the compute service received in each step."""

    result: list[tuple[str, ...]] = []
    for step in observed["steps"]:
        received = step["service_orders"]
        if len(received) != 1:
            raise AssertionError(
                f"{observed['case_id']} step {step['step_index']} did not form "
                f"exactly one co-runnable group: {received}"
            )
        result.append(tuple(received[0]))
    return result


def _latencies(observed: dict[str, Any]) -> list[int]:
    return [step["step_latency_ps"] for step in observed["steps"]]


def component_replay(
    order: tuple[str, ...],
    scheduler_count_per_sm: int,
    load_store_issue_width_per_sm: int,
) -> dict[str, Any]:
    """Replay one ordered tuple directly through the compute service."""

    from simllm.compute import GpuTask, GpuTaskKind, SmSchedulerModel

    built = launches()
    model = SmSchedulerModel(
        architecture(
            scheduler_count_per_sm=scheduler_count_per_sm,
            load_store_issue_width_per_sm=load_store_issue_width_per_sm,
        )
    )
    tasks = tuple(
        GpuTask(
            task_id=name,
            kind=GpuTaskKind.MEMORY if name == "memory" else GpuTaskKind.COMPUTE,
            launch=built[name],
        )
        for name in order
    )
    estimate = model.estimate_concurrent(tasks)
    return {
        "order": list(order),
        "duration_cycles": estimate.duration_cycles,
        "admitted_cycle": {task.task_id: task.admitted_cycle for task in estimate.tasks},
        "completion_cycle": {
            task.task_id: task.completion_cycle for task in estimate.tasks
        },
    }


def isolated_controls() -> dict[str, int]:
    from simllm.compute import SmSchedulerModel

    model = SmSchedulerModel(architecture())
    built = launches()
    return {name: model.estimate(built[name]).duration_cycles for name in built}


# ---------------------------------------------------------------------------
# Observation plan
# ---------------------------------------------------------------------------


def observe() -> dict[str, Any]:
    """Run every registered live case and component replay."""

    observed: dict[str, Any] = {"isolated_cycles": isolated_controls(), "cases": {}}

    def add(case: LiveCase) -> None:
        observed["cases"][case.case_id] = run_live_case(case)

    # identity baselines and identity settings on fixture F2
    for budget, lanes in ((4, 4), (8, 8)):
        add(
            LiveCase(
                f"f2-omitted-{budget}{lanes}",
                FIXTURE_F2_ORDER,
                None,
                {},
                budget,
                lanes,
            )
        )
        add(
            LiveCase(
                f"f2-identity-{budget}{lanes}",
                FIXTURE_F2_ORDER,
                ("identity", {}),
                {},
                budget,
                lanes,
            )
        )
    for tag, labels in (
        ("plain", {}),
        ("labels", {"memory": 2, "network": 1}),
        ("permuted", {"memory": 1, "network": 2}),
    ):
        add(LiveCase(f"f2-omitted-{tag}", FIXTURE_F2_ORDER, None, dict(labels)))
        add(
            LiveCase(
                f"f2-identity-{tag}",
                FIXTURE_F2_ORDER,
                ("identity", {}),
                dict(labels),
            )
        )
        add(
            LiveCase(
                f"f2-strict-off-{tag}",
                FIXTURE_F2_ORDER,
                ("strict", {"class_aware": False}),
                dict(labels),
            )
        )
        add(
            LiveCase(
                f"f2-wrr-off-{tag}",
                FIXTURE_F2_ORDER,
                ("wrr", {"weights": {1: 2, 2: 1}, "class_aware": False}),
                dict(labels),
            )
        )

    # family A
    for instance, (spec, labels, budget, lanes, _, _) in FROZEN_FAMILY_A.items():
        add(
            LiveCase(
                f"f2-{instance}",
                FIXTURE_F2_ORDER,
                spec,
                dict(labels),
                budget,
                lanes,
            )
        )

    # fixture F3 identity and identity settings
    add(LiveCase("f3-omitted", FIXTURE_F3_ORDER, None, {}))
    add(LiveCase("f3-identity", FIXTURE_F3_ORDER, ("identity", {}), {}))
    f3_labels = {"memory": 2, "network": 1, "compute": 1}
    add(
        LiveCase(
            "f3-strict-off",
            FIXTURE_F3_ORDER,
            ("strict", {"class_aware": False}),
            dict(f3_labels),
        )
    )
    add(
        LiveCase(
            "f3-wrr-off",
            FIXTURE_F3_ORDER,
            ("wrr", {"weights": {1: 2, 2: 1}, "class_aware": False}),
            dict(f3_labels),
        )
    )

    # family B
    for instance, (spec, labels, _) in FROZEN_FAMILY_B.items():
        add(LiveCase(f"f3-{instance}", FIXTURE_F3_ORDER, spec, dict(labels)))

    # scalar compatibility path
    add(
        LiveCase(
            "f2-scalar-identity",
            FIXTURE_F2_ORDER,
            ("identity", {}),
            {},
            nominal_ps=dict(FROZEN_SCALAR_NOMINAL_PS),
        )
    )
    add(
        LiveCase(
            "f2-scalar-strict",
            FIXTURE_F2_ORDER,
            ("strict", {}),
            {"memory": 2, "network": 1},
            nominal_ps=dict(FROZEN_SCALAR_NOMINAL_PS),
        )
    )

    # mandatory ordering stays outside the policy
    add(
        LiveCase(
            "f2-dependent-identity",
            FIXTURE_F2_ORDER,
            ("identity", {}),
            {},
            depends={"network": ("memory",)},
        )
    )
    add(
        LiveCase(
            "f2-dependent-strict",
            FIXTURE_F2_ORDER,
            ("strict", {}),
            {"memory": 2, "network": 1},
            depends={"network": ("memory",)},
        )
    )

    observed["component"] = {
        "f2-44-memory-first": component_replay(("memory", "network"), 4, 4),
        "f2-44-network-first": component_replay(("network", "memory"), 4, 4),
        "f2-88-memory-first": component_replay(("memory", "network"), 8, 8),
        "f2-88-network-first": component_replay(("network", "memory"), 8, 8),
    }
    return observed


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _foreign_candidate_rejected() -> bool:
    """A policy granting an operation outside the offer must be refused."""

    from simllm.core import ArbitrationCandidate, CoarseDeviceRuntime

    class _ForeignPolicy:
        def select(self, candidates):
            return ArbitrationCandidate(
                operation_id="not-a-candidate",
                baseline_sequence=0,
                eligible_at_ps=0,
                class_label=0,
            )

    runtime = CoarseDeviceRuntime(
        kernel_services={0: recording_service(4, 4)},
        kernel_launches=launches(),
        arbitration_policy=_ForeignPolicy(),
    )
    case = LiveCase("foreign", FIXTURE_F2_ORDER)
    from simllm.core import ExecutionGraph

    graph = ExecutionGraph(
        execution_id="foreign",
        step_index=0,
        released_at_ps=T0_PS,
        operations=_live_operations(case),
    )
    try:
        runtime.execute(graph)
    except ValueError:
        return runtime.last_report is None
    return False


def score_rows(observed: dict[str, Any]) -> list[Row]:
    rows: list[Row] = []
    cases = observed["cases"]

    # ---- run configuration -------------------------------------------------
    rows.append(
        Row(
            "configuration",
            RUN_CONFIGURATION,
            "fixtures",
            None,
            {
                "F2": list(FIXTURE_F2_ORDER),
                "F3": list(FIXTURE_F3_ORDER),
                "steps": LIVE_STEP_COUNT,
                "t0_ps": T0_PS,
            },
        )
    )

    # ---- raw observations, recorded before any predicate -------------------
    for case_id, case in cases.items():
        rows.append(
            Row(
                "raw",
                RAW_OBSERVATION,
                f"{case_id} service orders",
                None,
                [step["service_orders"] for step in case["steps"]],
            )
        )
        rows.append(
            Row(
                "raw",
                RAW_OBSERVATION,
                f"{case_id} step latency ps",
                None,
                _latencies(case),
            )
        )
        rows.append(
            Row(
                "raw",
                RAW_OBSERVATION,
                f"{case_id} grants",
                None,
                [step["grants"] for step in case["steps"]],
            )
        )
    for key, payload in observed["component"].items():
        rows.append(Row("raw", RAW_OBSERVATION, f"component {key}", None, payload))

    # ---- family A ----------------------------------------------------------
    identity_orders = {
        (4, 4): _orders(cases["f2-omitted-44"]),
        (8, 8): _orders(cases["f2-omitted-88"]),
    }
    identity_latencies = {
        (4, 4): _latencies(cases["f2-omitted-44"]),
        (8, 8): _latencies(cases["f2-omitted-88"]),
    }
    for instance, (_, _, budget, lanes, order, jct) in FROZEN_FAMILY_A.items():
        case = cases[f"f2-{instance}"]
        rows.append(
            Row(
                FAMILY_A,
                GENUINE_RISK,
                f"{instance} ordered tuple in every step",
                [list(order)] * LIVE_STEP_COUNT,
                [list(item) for item in _orders(case)],
            )
        )
        rows.append(
            Row(
                FAMILY_A,
                GENUINE_RISK,
                f"{instance} step JCT ps in every step",
                [jct] * LIVE_STEP_COUNT,
                _latencies(case),
            )
        )
        rows.append(
            Row(
                FAMILY_A,
                GENUINE_RISK,
                f"{instance} policy-caused JCT move against identity",
                [jct - value for value in identity_latencies[(budget, lanes)]],
                [
                    measured - value
                    for measured, value in zip(
                        _latencies(case),
                        identity_latencies[(budget, lanes)],
                        strict=True,
                    )
                ],
            )
        )

    # ---- family B ----------------------------------------------------------
    for instance, (_, _, orders) in FROZEN_FAMILY_B.items():
        case = cases[f"f3-{instance}"]
        rows.append(
            Row(
                FAMILY_B,
                GENUINE_RISK,
                f"{instance} per-step ordered tuples",
                [list(item) for item in orders],
                [list(item) for item in _orders(case)],
            )
        )

    # ---- fatal guards ------------------------------------------------------
    rows.append(
        Row(
            "guards",
            FATAL_GUARD,
            "isolated controls",
            {name: FROZEN_ISOLATED_CYCLES[name] for name in sorted(FROZEN_ISOLATED_CYCLES)},
            {
                name: observed["isolated_cycles"][name]
                for name in sorted(observed["isolated_cycles"])
            },
        )
    )
    rows.append(
        Row(
            "guards",
            FATAL_GUARD,
            "identity ordered tuple is literally graph order",
            {
                "F2": [list(FROZEN_F2_IDENTITY_ORDER)] * LIVE_STEP_COUNT,
                "F3": [list(FROZEN_F3_IDENTITY_ORDER)] * LIVE_STEP_COUNT,
            },
            {
                "F2": [list(item) for item in identity_orders[(4, 4)]],
                "F3": [list(item) for item in _orders(cases["f3-omitted"])],
            },
        )
    )
    rows.append(
        Row(
            "guards",
            FATAL_GUARD,
            "identity baseline step JCT ps",
            {
                "44": [FROZEN_F2_IDENTITY_JCT_PS[(4, 4)]] * LIVE_STEP_COUNT,
                "88": [FROZEN_F2_IDENTITY_JCT_PS[(8, 8)]] * LIVE_STEP_COUNT,
            },
            {
                "44": identity_latencies[(4, 4)],
                "88": identity_latencies[(8, 8)],
            },
        )
    )

    # byte-identity of every identity setting, including label permutation
    for fixture, group in (
        (
            "F2",
            [
                "f2-omitted-plain",
                "f2-identity-plain",
                "f2-omitted-labels",
                "f2-identity-labels",
                "f2-omitted-permuted",
                "f2-identity-permuted",
                "f2-strict-off-plain",
                "f2-strict-off-labels",
                "f2-strict-off-permuted",
                "f2-wrr-off-plain",
                "f2-wrr-off-labels",
                "f2-wrr-off-permuted",
            ],
        ),
        ("F3", ["f3-omitted", "f3-identity", "f3-strict-off", "f3-wrr-off"]),
    ):
        reference = cases[group[0]]
        reference_canonical = [step["canonical"] for step in reference["steps"]]
        reference_orders = [list(item) for item in _orders(reference)]
        reference_bytes = [
            step["total_class_service_bytes"] for step in reference["steps"]
        ]
        for case_id in group[1:]:
            case = cases[case_id]
            rows.append(
                Row(
                    "guards",
                    FATAL_GUARD,
                    f"{fixture} identity setting {case_id} is byte-identical",
                    {
                        "canonical": reference_canonical,
                        "orders": reference_orders,
                        "latency": _latencies(reference),
                        "total_class_service_bytes": reference_bytes,
                    },
                    {
                        "canonical": [step["canonical"] for step in case["steps"]],
                        "orders": [list(item) for item in _orders(case)],
                        "latency": _latencies(case),
                        "total_class_service_bytes": [
                            step["total_class_service_bytes"] for step in case["steps"]
                        ],
                    },
                )
            )

    # membership is policy invariant, only the order changes
    for fixture, group, expected_members in (
        (
            "F2",
            ["f2-omitted-44"] + [f"f2-{name}" for name in ("A1", "A2", "A4", "A5")],
            sorted(FIXTURE_F2_ORDER),
        ),
        (
            "F3",
            ["f3-omitted"] + [f"f3-{name}" for name in FROZEN_FAMILY_B],
            sorted(FIXTURE_F3_ORDER),
        ),
    ):
        for case_id in group:
            rows.append(
                Row(
                    "guards",
                    FATAL_GUARD,
                    f"{fixture} co-runnable membership {case_id}",
                    [expected_members] * LIVE_STEP_COUNT,
                    [sorted(item) for item in _orders(cases[case_id])],
                )
            )

    # the frozen grant model
    for case_id, member_count in (
        ("f2-A1", len(FIXTURE_F2_ORDER)),
        ("f2-A4", len(FIXTURE_F2_ORDER)),
        ("f3-B1", len(FIXTURE_F3_ORDER)),
        ("f3-B2", len(FIXTURE_F3_ORDER)),
        ("f3-B3", len(FIXTURE_F3_ORDER)),
    ):
        case = cases[case_id]
        rows.append(
            Row(
                "guards",
                FATAL_GUARD,
                f"{case_id} grant model",
                [
                    {
                        "count": 1 + member_count,
                        "offers": [member_count] + list(range(member_count, 0, -1)),
                        "legal": True,
                    }
                ]
                * LIVE_STEP_COUNT,
                [
                    {
                        "count": len(step["grants"]),
                        "offers": [len(grant["offered"]) for grant in step["grants"]],
                        "legal": all(
                            grant["granted"] in grant["offered"]
                            for grant in step["grants"]
                        ),
                    }
                    for step in case["steps"]
                ],
            )
        )

    rows.append(
        Row(
            "guards",
            FATAL_GUARD,
            "a grant outside the offered set is rejected before state mutates",
            True,
            _foreign_candidate_rejected(),
        )
    )

    # scalar compatibility path
    for case_id in ("f2-scalar-identity", "f2-scalar-strict"):
        rows.append(
            Row(
                "guards",
                FATAL_GUARD,
                f"scalar path {case_id} stays order invariant",
                [FROZEN_SCALAR_JCT_PS] * LIVE_STEP_COUNT,
                _latencies(cases[case_id]),
            )
        )
    rows.append(
        Row(
            "guards",
            FATAL_GUARD,
            "scalar path still arbitrates the group order",
            [["network", "memory"]] * LIVE_STEP_COUNT,
            [
                [grant["granted"] for grant in step["grants"][1:]]
                for step in cases["f2-scalar-strict"]["steps"]
            ],
        )
    )

    # mandatory ordering before arbitration
    for case_id in ("f2-dependent-identity", "f2-dependent-strict"):
        case = cases[case_id]
        rows.append(
            Row(
                "guards",
                FATAL_GUARD,
                f"dependency keeps {case_id} serialized",
                {
                    "orders": [[["memory"], ["network"]]] * LIVE_STEP_COUNT,
                    "latency": [FROZEN_DEPENDENT_JCT_PS] * LIVE_STEP_COUNT,
                },
                {
                    "orders": [step["service_orders"] for step in case["steps"]],
                    "latency": _latencies(case),
                },
            )
        )

    # physical intervals
    interval_rows = {
        "F2 concurrent": [
            latency
            for case_id in ("f2-omitted-44", "f2-omitted-88") + tuple(
                f"f2-{name}" for name in FROZEN_FAMILY_A
            )
            for latency in _latencies(cases[case_id])
        ],
        "F3 concurrent": [
            latency
            for case_id in ("f3-omitted",) + tuple(f"f3-{name}" for name in FROZEN_FAMILY_B)
            for latency in _latencies(cases[case_id])
        ],
        "F2 dependent": [
            latency
            for case_id in ("f2-dependent-identity", "f2-dependent-strict")
            for latency in _latencies(cases[case_id])
        ],
    }
    for label, latencies in interval_rows.items():
        low, high = FROZEN_PHYSICAL_INTERVALS[label]
        rows.append(
            Row(
                "guards",
                FATAL_GUARD,
                f"physical interval {label}",
                True,
                all(
                    low * PS_PER_CYCLE <= latency <= high * PS_PER_CYCLE
                    for latency in latencies
                ),
            )
        )

    # queue-visit semantics and completion conservation
    rows.append(
        Row(
            "guards",
            FATAL_GUARD,
            "queue events, completion events and metric conservation",
            True,
            all(
                step["queued_matches_eligibility"]
                and step["started_matches_grant"]
                and step["completed_events_match_operations"]
                and step["completed_event_count"] == len(case["steps"][0]["graph_order"])
                and step["step_completed_at_ps"] == step["execution_completed_at_ps"]
                and step["metric_completed_at_ps"] == step["execution_completed_at_ps"]
                for case in cases.values()
                for step in case["steps"]
            ),
        )
    )
    rows.append(
        Row(
            "guards",
            FATAL_GUARD,
            "TTFT and each decode TPOT equal the step JCT",
            True,
            all(
                (
                    step["ttft_ps"] == step["step_latency_ps"]
                    if step["step_index"] == 0
                    else step["tpot_ps"] == step["step_latency_ps"]
                )
                for case in cases.values()
                for step in case["steps"]
            ),
        )
    )

    # the live projection agrees with the component observables
    component_by_order = {
        (tuple(payload["order"]), key.split("-")[1]): payload
        for key, payload in observed["component"].items()
    }
    live_versus_component: list[bool] = []
    for instance, (_, _, budget, lanes, order, _) in FROZEN_FAMILY_A.items():
        payload = component_by_order[(order, f"{budget}{lanes}")]
        for step in cases[f"f2-{instance}"]["steps"]:
            for name in order:
                started = step["operation_started_at_ps"][name]
                expected = started + payload["completion_cycle"][name] * PS_PER_CYCLE
                live_versus_component.append(
                    step["operation_completed_at_ps"][name] == expected
                )
    rows.append(
        Row(
            "guards",
            FATAL_GUARD,
            "live completions equal the component replay of the same order",
            True,
            all(live_versus_component),
        )
    )
    return rows


def _worktree_clean() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and not completed.stdout.strip()


def _observed_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_rows_csv(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", "evidence_class", "case", "expected", "measured", "status"])
        for row in rows:
            writer.writerow(
                [
                    row.family,
                    row.evidence_class,
                    row.case,
                    json.dumps(row.expected, sort_keys=True),
                    json.dumps(row.measured, sort_keys=True),
                    row.status,
                ]
            )


def run_study(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    clean = _worktree_clean()
    observed = observe()
    rows = score_rows(observed)
    rows.append(Row("guards", FATAL_GUARD, "clean worktree", True, clean))

    scored = [row for row in rows if row.evidence_class == GENUINE_RISK]
    guards = [row for row in rows if row.evidence_class == FATAL_GUARD]
    failed_guards = [row for row in guards if not row.passed]
    passed_scored = [row for row in scored if row.passed]

    summary = {
        "observed_revision": _observed_revision(),
        "authored_against_commit": AUTHORED_AGAINST_COMMIT,
        "expectations_commit": EXPECTATIONS_COMMIT,
        "void": bool(failed_guards),
        "fatal_guards": len(guards),
        "fatal_guard_failures": [row.case for row in failed_guards],
        "genuine_risk_rows": len(scored),
        "genuine_risk_passed": len(passed_scored),
        "families": {
            family: {
                "rows": sum(1 for row in scored if row.family == family),
                "passed": sum(
                    1 for row in scored if row.family == family and row.passed
                ),
                "failed": [
                    row.case
                    for row in scored
                    if row.family == family and not row.passed
                ],
            }
            for family in SCORED_FAMILIES
        },
    }

    _write_json(out / "observations.json", observed)
    _write_json(out / "summary.json", summary)
    _write_json(out / "rows.json", [_row_payload(row) for row in rows])
    _write_rows_csv(out / "rows.csv", rows)

    print(json.dumps(summary, indent=2, sort_keys=True))
    if failed_guards:
        print("VOID: a fatal guard failed, no behavioral fraction is reported")
        return 1
    return 0 if len(passed_scored) == len(scored) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="directory for generated artifacts")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate the frozen registry without importing SimLLM",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check_only:
        check_only()
        return 0
    if args.out is None:
        raise SystemExit("--out is required for the production run")
    return run_study(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
