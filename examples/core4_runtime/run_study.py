"""Run the four CORE-4 experiment families frozen before implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from simllm._local_config import path_from_env
from simllm.compute import (
    CopyDirection,
    CopyDirectionProfile,
    CopyEngineProfile,
    CopyEngineServiceModel,
    GpuArchitectureProfile,
    GpuCalibrationProfile,
    GpuModelProvenance,
    MemoryHierarchyProfile,
    PipelineKind,
    PipelineProfile,
    WarpSchedulerPolicy,
)
from simllm.core import (
    CoarseDeviceProfile,
    CoarseDeviceRuntime,
    CollectiveWork,
    ComputeWork,
    ControlMode,
    ControlWork,
    DmaWork,
    ExecutionGraph,
    ExecutionOperation,
    IdentityArbitrationPolicy,
    collective_goal_tags,
    execution_result_to_json,
)
from simllm.traffic import render_serial_execution_graph_goal

T0 = 5_000
B = 1_048_576
G = 1_000_000_000
S_400_PS = 20_971_520
S_200_PS = 41_943_040
FROZEN_A_JCT_PS = {
    (10_000_000, 40_000_000, False): 40_000_000,
    (10_000_000, 40_000_000, True): 50_000_000,
    (80_000_000, 40_000_000, False): 80_000_000,
    (80_000_000, 40_000_000, True): 120_000_000,
}
FROZEN_B_JCT_PS = {
    (1, 200): 83_886_080,
    (1, 400): 41_943_040,
    (8, 200): 83_886_080,
    (8, 400): 41_943_040,
}
FROZEN_B_THROUGHPUT_BPS = {
    (1, 200): 200_000_000_000,
    (1, 400): 400_000_000_000,
    (8, 200): 1_600_000_000_000,
    (8, 400): 3_200_000_000_000,
}
EXACT_ORACLE = "exact-oracle"
BEHAVIORAL_RELATION = "behavioral-relation"
STRUCTURAL = "fatal-structural"


@dataclass(frozen=True)
class EvidenceRow:
    family: str
    evidence_class: str
    case: str
    expected: int | str
    measured: int | str
    genuine_risk: bool

    @property
    def passed(self) -> bool:
        return self.expected == self.measured


def _architecture() -> GpuArchitectureProfile:
    engine = CopyEngineProfile(
        engine_id="ideal-copy",
        clock_hz=1_000_000_000,
        direction_profiles=tuple(
            CopyDirectionProfile(
                direction=direction,
                setup_cycles=0,
                bandwidth_bytes_per_cycle=1.0,
            )
            for direction in (
                CopyDirection.HOST_TO_DEVICE,
                CopyDirection.DEVICE_TO_HOST,
                CopyDirection.DEVICE_TO_DEVICE,
                CopyDirection.PEER_TO_PEER,
            )
        ),
    )
    profile_id = "core4-study-gpu"
    calibration = GpuCalibrationProfile(
        calibration_id="core4-study-calibration",
        target_architecture_profile_id=profile_id,
        provenance=GpuModelProvenance(
            source="synthetic ideal study fixture, no silicon claim",
            version="1",
            gpu="synthetic",
            created="2026-08-10",
        ),
        core_clock_hz=1_000_000_000,
        target_memory_clock_hz=None,
        pipelines=(
            PipelineProfile(
                kind=PipelineKind.ALU,
                opcodes=("ALU",),
                latency_cycles=1,
                issue_width_per_sm=4,
            ),
        ),
        memory=MemoryHierarchyProfile(
            hbm_latency_cycles=0,
            hbm_bandwidth_bytes_per_cycle=64,
        ),
        copy_engines=(engine,),
        warp_scheduler_policy=WarpSchedulerPolicy.LOOSE_ROUND_ROBIN,
        relative_uncertainty=0.0,
    )
    return GpuArchitectureProfile(
        profile_id=profile_id,
        gpu_name="core4-study-gpu-name",
        sm_count=1,
        warp_size=32,
        scheduler_count_per_sm=4,
        max_blocks_per_sm=16,
        max_warps_per_sm=64,
        max_threads_per_sm=2048,
        max_threads_per_block=1024,
        registers_per_sm=65536,
        max_registers_per_thread=255,
        register_allocation_granularity_per_warp=1,
        shared_memory_per_sm=65536,
        max_static_shared_memory_per_block=49152,
        max_shared_memory_per_block=65536,
        shared_memory_allocation_granularity=1,
        calibration=calibration,
    )


ARCHITECTURE = _architecture()


def profile(
    rate_gbps: int = 400,
    *,
    launch_service_ps: int = 0,
) -> CoarseDeviceProfile:
    return CoarseDeviceProfile(
        rnic_rate_bps=rate_gbps * G,
        launch_service_ps=launch_service_ps,
        copy_engines=(CopyEngineServiceModel(ARCHITECTURE, "ideal-copy"),),
    )


def overlap_graph(
    execution_id: str,
    compute_ps: int,
    dma_ps: int,
    *,
    dependency: bool,
    shared_hbm: bool = False,
) -> ExecutionGraph:
    return ExecutionGraph(
        execution_id,
        0,
        T0,
        (
            ExecutionOperation(
                "compute",
                0,
                "cuda:0:compute",
                ComputeWork(
                    "ideal-compute",
                    hbm_bytes=1 if shared_hbm else 0,
                    nominal_duration_ps=compute_ps,
                ),
            ),
            ExecutionOperation(
                "dma",
                0,
                "cuda:0:copy",
                DmaWork("ideal-dma", "host:pinned", "gpu:0:hbm", dma_ps // 1000),
                depends_on=("compute",) if dependency else (),
            ),
        ),
    )


def control_graph(
    execution_id: str,
    active_gpus: int,
    *,
    mode: ControlMode,
    class_label: int = 0,
    anchor: bool = False,
) -> ExecutionGraph:
    operations = []
    for rank in range(active_gpus):
        for sequence in range(2):
            operations.append(
                ExecutionOperation(
                    f"control-{rank}-{sequence}",
                    rank,
                    f"control-lane-{sequence}",
                    ControlWork(
                        "rail-control",
                        destination_ranks=(rank + 8,),
                        payload_bytes=B,
                        mode=mode,
                    ),
                    priority=class_label,
                )
            )
    if anchor:
        operations.append(
            ExecutionOperation(
                "compute-anchor",
                7,
                "anchor",
                ComputeWork("anchor", nominal_duration_ps=10_000_000),
            )
        )
    return ExecutionGraph(execution_id, 0, T0, tuple(operations))


def critical_launch_graph(launch_service_ps: int) -> ExecutionGraph:
    return ExecutionGraph(
        f"critical-launch-{launch_service_ps}",
        0,
        0,
        (
            ExecutionOperation(
                "a",
                0,
                "a",
                ComputeWork("a", nominal_duration_ps=100),
            ),
            ExecutionOperation(
                "filler",
                1,
                "filler",
                ComputeWork("filler", nominal_duration_ps=0),
                not_before_ps=100 + launch_service_ps,
            ),
            ExecutionOperation(
                "b",
                0,
                "b",
                ComputeWork("b", nominal_duration_ps=20),
                depends_on=("a",),
            ),
        ),
        completion_operation_ids=("b",),
    )


def add(
    rows: list[EvidenceRow],
    family: str,
    evidence_class: str,
    case: str,
    expected: int | str,
    measured: int | str,
    *,
    genuine_risk: bool,
) -> None:
    rows.append(
        EvidenceRow(
            family,
            evidence_class,
            case,
            expected,
            measured,
            genuine_risk,
        )
    )


def family_a(rows: list[EvidenceRow], configurations: list[dict[str, object]]) -> None:
    for compute_ps, dma_ps in ((10_000_000, 40_000_000), (80_000_000, 40_000_000)):
        measured = {}
        for dependency in (False, True):
            case = f"C={compute_ps},D={dma_ps},edge={int(dependency)}"
            graph = overlap_graph(case, compute_ps, dma_ps, dependency=dependency)
            result = CoarseDeviceRuntime(profile()).execute(graph)
            jct = result.completed_at_ps - graph.released_at_ps
            closed_form = (
                compute_ps + dma_ps if dependency else max(compute_ps, dma_ps)
            )
            expected = FROZEN_A_JCT_PS[(compute_ps, dma_ps, dependency)]
            assert closed_form == expected
            measured[dependency] = jct
            configurations.append(
                {
                    "family": "A",
                    "compute_ps": compute_ps,
                    "dma_ps": dma_ps,
                    "dependency": dependency,
                }
            )
            add(rows, "A", EXACT_ORACLE, case, expected, jct, genuine_risk=True)
            add(
                rows,
                "A",
                STRUCTURAL,
                f"origin:{case}",
                T0 + expected,
                result.completed_at_ps,
                genuine_risk=False,
            )
        add(
            rows,
            "A",
            BEHAVIORAL_RELATION,
            f"edge-penalty:C={compute_ps},D={dma_ps}",
            min(compute_ps, dma_ps),
            measured[True] - measured[False],
            genuine_risk=True,
        )
        shared_graph = overlap_graph(
            f"shared-hbm-{compute_ps}-{dma_ps}",
            compute_ps,
            dma_ps,
            dependency=False,
            shared_hbm=True,
        )
        shared = CoarseDeviceRuntime(profile()).execute(shared_graph)
        add(
            rows,
            "A",
            STRUCTURAL,
            f"shared-hbm:C={compute_ps},D={dma_ps}",
            compute_ps + dma_ps,
            shared.completed_at_ps - T0,
            genuine_risk=False,
        )


def family_b(rows: list[EvidenceRow], configurations: list[dict[str, object]]) -> None:
    jcts = {}
    for active_gpus in (1, 8):
        for rate_gbps in (200, 400):
            graph = control_graph(
                f"rails-N{active_gpus}-R{rate_gbps}",
                active_gpus,
                mode=ControlMode.SYNCHRONOUS,
            )
            runtime = CoarseDeviceRuntime(profile(rate_gbps))
            result = runtime.execute(graph)
            report = runtime.last_report
            assert report is not None
            closed_form = 16_000 * B // rate_gbps
            expected = FROZEN_B_JCT_PS[(active_gpus, rate_gbps)]
            assert closed_form == expected
            jct = result.completed_at_ps - T0
            jcts[(active_gpus, rate_gbps)] = jct
            useful_bytes = sum(wqe.payload_bytes for wqe in report.wqes)
            useful_bps = useful_bytes * 8 * 1_000_000_000_000 // jct
            configurations.append(
                {
                    "family": "B",
                    "active_gpus": active_gpus,
                    "rate_gbps": rate_gbps,
                    "payload_bytes": B,
                }
            )
            add(
                rows,
                "B",
                EXACT_ORACLE,
                f"jct:N={active_gpus},R={rate_gbps}",
                expected,
                jct,
                genuine_risk=True,
            )
            add(
                rows,
                "B",
                EXACT_ORACLE,
                f"throughput:N={active_gpus},R={rate_gbps}",
                FROZEN_B_THROUGHPUT_BPS[(active_gpus, rate_gbps)],
                useful_bps,
                genuine_risk=True,
            )
            assert (
                active_gpus * rate_gbps * G
                == FROZEN_B_THROUGHPUT_BPS[(active_gpus, rate_gbps)]
            )
            by_source = {}
            for wqe in report.wqes:
                by_source.setdefault(wqe.source_rank, []).append(wqe)
            fifo_ok = all(
                [wqe.sq_post_sequence for wqe in records] == [1, 2]
                and records[0].finished_at_ps == records[1].started_at_ps
                and {wqe.rnic_id for wqe in records}
                == {f"node-0:rnic-{source_rank}"}
                for source_rank, records in by_source.items()
            )
            add(
                rows,
                "B",
                STRUCTURAL,
                f"affinity-fifo:N={active_gpus},R={rate_gbps}",
                "PASS",
                "PASS" if fifo_ok and len(by_source) == active_gpus else "FAIL",
                genuine_risk=False,
            )
    for rate_gbps in (200, 400):
        add(
            rows,
            "B",
            BEHAVIORAL_RELATION,
            f"rail-independence:R={rate_gbps}",
            jcts[(1, rate_gbps)],
            jcts[(8, rate_gbps)],
            genuine_risk=True,
        )
    for active_gpus in (1, 8):
        add(
            rows,
            "B",
            BEHAVIORAL_RELATION,
            f"rate-halving:N={active_gpus}",
            2 * jcts[(active_gpus, 400)],
            jcts[(active_gpus, 200)],
            genuine_risk=True,
        )


def _schedule_canonical(result, report) -> bytes:
    visits = []
    for visit in report.visits:
        visits.append(
            {
                "execution_id": visit.execution_id,
                "operation_id": visit.operation_id,
                "resource_kind": visit.resource.kind.value,
                "resource_id": visit.resource.resource_id,
                "submitted_at_ps": visit.submitted_at_ps,
                "eligible_at_ps": visit.eligible_at_ps,
                "started_at_ps": visit.started_at_ps,
                "finished_at_ps": visit.finished_at_ps,
                "completed_at_ps": visit.completed_at_ps,
                "service_bytes": visit.service_bytes,
                "subject_object_id": visit.subject_object_id,
            }
        )
    operations = []
    for operation in report.operations:
        operations.append(
            {
                "operation_id": operation.operation_id,
                "submitted_at_ps": operation.submitted_at_ps,
                "eligible_at_ps": operation.eligible_at_ps,
                "completed_at_ps": operation.completed_at_ps,
                "physical_completed_at_ps": operation.physical_completed_at_ps,
                "participant_completed_at_ps": operation.participant_completed_at_ps,
                "breakdown": asdict(operation.breakdown),
                "sum_visit_wait_ps": operation.sum_visit_wait_ps,
            }
        )
    payload = {
        "result": execution_result_to_json(result),
        "authority": report.authority,
        "operations": operations,
        "visits": visits,
        "wqes": [asdict(wqe) for wqe in report.wqes],
        "sum_visit_wait_ps": report.sum_visit_wait_ps,
        "critical_path_queue_ps": report.critical_path_queue_ps,
        "critical_path": report.realized_critical_path_operation_ids,
        "random_draw_count": report.random_draw_count,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def family_c(rows: list[EvidenceRow], configurations: list[dict[str, object]]) -> None:
    outcomes = {}
    for mode in (ControlMode.ASYNCHRONOUS, ControlMode.SYNCHRONOUS):
        for class_label in (3, 9):
            graph = control_graph(
                "tail-control",
                8,
                mode=mode,
                class_label=class_label,
                anchor=True,
            )
            runtime = CoarseDeviceRuntime(profile())
            result = runtime.execute(graph)
            report = runtime.last_report
            assert report is not None
            expected_jct = (
                10_000_000
                if mode is ControlMode.ASYNCHRONOUS
                else 41_943_040
            )
            assert (
                10_000_000
                if mode is ControlMode.ASYNCHRONOUS
                else 2 * S_400_PS
            ) == expected_jct
            outcomes[(mode, class_label)] = (result, report)
            configurations.append(
                {
                    "family": "C",
                    "mode": mode.value,
                    "class_label": class_label,
                    "active_gpus": 8,
                }
            )
            add(
                rows,
                "C",
                EXACT_ORACLE,
                f"jct:{mode.value}:class={class_label}",
                expected_jct,
                result.completed_at_ps - T0,
                genuine_risk=True,
            )
            add(
                rows,
                "C",
                EXACT_ORACLE,
                f"quiescence:{mode.value}:class={class_label}",
                41_943_040,
                result.quiesced_at_ps - T0,
                genuine_risk=True,
            )
            assert 2 * S_400_PS == 41_943_040
            add(
                rows,
                "C",
                BEHAVIORAL_RELATION,
                f"additive-wait:{mode.value}:class={class_label}",
                167_772_160,
                report.sum_visit_wait_ps,
                genuine_risk=True,
            )
            assert 8 * S_400_PS == 167_772_160
            add(
                rows,
                "C",
                BEHAVIORAL_RELATION,
                f"critical-path-queue:{mode.value}:class={class_label}",
                0 if mode is ControlMode.ASYNCHRONOUS else 20_971_520,
                report.critical_path_queue_ps,
                genuine_risk=True,
            )
            conservation_ok = all(
                operation.breakdown.operation_latency_ps
                == operation.breakdown.launch_queue_ps
                + operation.breakdown.device_queue_ps
                + operation.breakdown.service_ps
                + operation.breakdown.completion_delivery_ps
                + operation.breakdown.external_dependency_ps
                for operation in report.operations
            )
            event_bounds_ok = (
                result.events[-1].timestamp_ps <= result.quiesced_at_ps
                and max(
                    operation.completed_at_ps
                    for operation in report.operations
                    if operation.operation_id in {
                        item.operation_id
                        for item in graph.operations
                    }
                )
                == result.quiesced_at_ps
                if mode is ControlMode.SYNCHRONOUS
                else result.events[-1].timestamp_ps == result.quiesced_at_ps
            )
            add(
                rows,
                "C",
                STRUCTURAL,
                f"conservation:{mode.value}:class={class_label}",
                "PASS",
                "PASS" if conservation_ok and event_bounds_ok else "FAIL",
                genuine_risk=False,
            )
    for class_label in (3, 9):
        async_result = outcomes[(ControlMode.ASYNCHRONOUS, class_label)][0]
        sync_result = outcomes[(ControlMode.SYNCHRONOUS, class_label)][0]
        add(
            rows,
            "C",
            BEHAVIORAL_RELATION,
            f"sync-penalty:class={class_label}",
            31_943_040,
            sync_result.completed_at_ps - async_result.completed_at_ps,
            genuine_risk=True,
        )

    for launch_service_ps, expected_jct_ps, expected_queue_ps in (
        (10, 150, 10),
        (20, 180, 20),
    ):
        graph = critical_launch_graph(launch_service_ps)
        runtime = CoarseDeviceRuntime(
            profile(launch_service_ps=launch_service_ps)
        )
        result = runtime.execute(graph)
        report = runtime.last_report
        assert report is not None
        configurations.append(
            {
                "family": "C",
                "mode": "critical-launch-chain",
                "launch_service_ps": launch_service_ps,
            }
        )
        add(
            rows,
            "C",
            EXACT_ORACLE,
            f"critical-launch-jct:L={launch_service_ps}",
            expected_jct_ps,
            result.completed_at_ps,
            genuine_risk=True,
        )
        add(
            rows,
            "C",
            BEHAVIORAL_RELATION,
            f"critical-launch-queue:L={launch_service_ps}",
            expected_queue_ps,
            report.critical_path_queue_ps,
            genuine_risk=True,
        )
        by_id = {record.operation_id: record for record in report.operations}
        chain_latency_ps = sum(
            by_id[operation_id].breakdown.operation_latency_ps
            for operation_id in report.realized_critical_path_operation_ids
        )
        add(
            rows,
            "C",
            STRUCTURAL,
            f"critical-launch-conservation:L={launch_service_ps}",
            expected_jct_ps,
            chain_latency_ps,
            genuine_risk=False,
        )
    for mode in (ControlMode.ASYNCHRONOUS, ControlMode.SYNCHRONOUS):
        left_result, left_report = outcomes[(mode, 3)]
        right_result, right_report = outcomes[(mode, 9)]
        add(
            rows,
            "C",
            BEHAVIORAL_RELATION,
            f"class-invariance:{mode.value}",
            hashlib.sha256(_schedule_canonical(left_result, left_report)).hexdigest(),
            hashlib.sha256(_schedule_canonical(right_result, right_report)).hexdigest(),
            genuine_risk=True,
        )


def mixed_identity_graph(labels: tuple[int, int, int, int]) -> ExecutionGraph:
    operations = (
        ExecutionOperation(
            "compute",
            0,
            "compute",
            ComputeWork("compute", nominal_duration_ps=10_000_000),
            priority=labels[0],
        ),
        ExecutionOperation(
            "dma",
            0,
            "copy",
            DmaWork("dma", "host", "gpu:0:hbm", 40_000),
            priority=labels[1],
        ),
        ExecutionOperation(
            "control",
            1,
            "control",
            ControlWork(
                "fabric",
                destination_ranks=(9,),
                payload_bytes=B,
                mode=ControlMode.SYNCHRONOUS,
            ),
            priority=labels[2],
        ),
        ExecutionOperation(
            "ring",
            0,
            "nccl",
            CollectiveWork("all-reduce", (0, 8), 1024, "ring", "tp"),
            priority=labels[3],
        ),
    )
    return ExecutionGraph("identity-mixed", 0, T0, operations)


def family_d(rows: list[EvidenceRow], configurations: list[dict[str, object]]) -> None:
    variants = (
        ("omitted-baseline", None, (1, 2, 3, 4)),
        ("identity-baseline", IdentityArbitrationPolicy(), (1, 2, 3, 4)),
        ("omitted-permuted", None, (4, 3, 2, 1)),
        ("identity-permuted", IdentityArbitrationPolicy(), (4, 3, 2, 1)),
    )
    canonical = {}
    for name, policy, labels in variants:
        graph = mixed_identity_graph(labels)
        runtime = CoarseDeviceRuntime(profile(), arbitration_policy=policy)
        result = runtime.execute(graph)
        report = runtime.last_report
        assert report is not None
        canonical[name] = _schedule_canonical(result, report)
        configurations.append(
            {
                "family": "D",
                "variant": name,
                "labels": labels,
            }
        )
    baseline_hash = hashlib.sha256(canonical["omitted-baseline"]).hexdigest()
    for name, _, _ in variants[1:]:
        add(
            rows,
            "D",
            BEHAVIORAL_RELATION,
            f"byte-equivalence:{name}",
            baseline_hash,
            hashlib.sha256(canonical[name]).hexdigest(),
            genuine_risk=True,
        )

    tag_graph = ExecutionGraph(
        "tag-referent",
        0,
        T0,
        (
            ExecutionOperation(
                "ring",
                0,
                "nccl",
                CollectiveWork("all-reduce", (0, 8), 1024, "ring"),
            ),
        ),
    )
    runtime = CoarseDeviceRuntime(profile())
    runtime.execute(tag_graph)
    report = runtime.last_report
    assert report is not None
    rendered = render_serial_execution_graph_goal(tag_graph, num_goal_ranks=9).render()
    rendered_tags = sorted(
        int(line.split(" tag ", 1)[1])
        for line in rendered.splitlines()
        if ": send " in line
    )
    add(
        rows,
        "D",
        STRUCTURAL,
        "rendered-goal-tag-projection",
        json.dumps(rendered_tags),
        json.dumps(sorted(wqe.goal_tag for wqe in report.wqes)),
        genuine_risk=False,
    )
    add(
        rows,
        "D",
        STRUCTURAL,
        "goal-tag-allocation",
        json.dumps({"ring": [1000, 1001]}, sort_keys=True),
        json.dumps(
            {key: list(value) for key, value in collective_goal_tags(tag_graph).items()},
            sort_keys=True,
        ),
        genuine_risk=False,
    )


def check_only() -> None:
    assert T0 > 0
    assert B % 4096 == 0
    assert 16_000 * B // 200 == 83_886_080
    assert 16_000 * B // 400 == 41_943_040
    assert 8 * S_400_PS == 167_772_160
    assert 2 * S_400_PS - 10_000_000 == 31_943_040
    assert FROZEN_A_JCT_PS[(10_000_000, 40_000_000, False)] == 40_000_000
    assert FROZEN_A_JCT_PS[(80_000_000, 40_000_000, True)] == 120_000_000
    assert FROZEN_B_JCT_PS[(8, 400)] == 41_943_040
    assert FROZEN_B_THROUGHPUT_BPS[(8, 400)] == 3_200_000_000_000
    assert 120 + 3 * 10 == 150
    assert 120 + 3 * 20 == 180
    assert profile().gpus_per_node == profile().rnics_per_node == 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="parse and validate frozen inputs without running the runtime",
    )
    args = parser.parse_args()
    check_only()
    if args.check_only:
        print("CORE-4 study inputs: parse-only PASS")
        return
    if args.output_dir is None:
        data_root = path_from_env("SIMLLM_DATA_ROOT")
        if data_root is None:
            parser.error("--output-dir is required when SIMLLM_DATA_ROOT is not set")
        args.output_dir = data_root / "core4_runtime"

    rows: list[EvidenceRow] = []
    configurations: list[dict[str, object]] = []
    family_a(rows, configurations)
    family_b(rows, configurations)
    family_c(rows, configurations)
    family_d(rows, configurations)
    failures = [row for row in rows if not row.passed]
    payload = {
        "expectation_commit": "d43cddb",
        "review_amendment_commit": "67cabda",
        "configurations": configurations,
        "evidence": [
            {
                **asdict(row),
                "passed": row.passed,
            }
            for row in rows
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    for evidence_class in (EXACT_ORACLE, BEHAVIORAL_RELATION, STRUCTURAL):
        selected = [row for row in rows if row.evidence_class == evidence_class]
        passed = sum(row.passed for row in selected)
        print(f"{evidence_class}: {passed}/{len(selected)} PASS")
    for family in "ABCD":
        scored = [
            row
            for row in rows
            if row.family == family and row.evidence_class != STRUCTURAL
        ]
        risky = [row for row in scored if row.genuine_risk]
        print(f"family {family} genuine-risk fraction: {len(risky)}/{len(scored)}")
    print(f"run configurations: {len(configurations)}")
    print(f"summary: {summary_path}")
    if failures:
        for row in failures:
            print(
                f"FAIL {row.family} {row.evidence_class} {row.case}: "
                f"expected={row.expected!r} measured={row.measured!r}"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
