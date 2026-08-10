"""Run the frozen CORE-5 completion-reduction study."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

from simllm._local_config import path_from_env

REPO_ROOT = Path(__file__).resolve().parents[2]
SHAPES = ("parallel", "serial")
RATES_GBPS = (200, 400)
JCT_PS = {
    ("parallel", 200): 356_680,
    ("serial", 200): 366_680,
    ("parallel", 400): 192_840,
    ("serial", 400): 202_840,
}
TIER_B_PAYLOADS = (4096, 1_048_576)
TIER_B_DOORBELL_PS = (0, 1000)
TIER_B_JCT_PS = {
    (4096, 200, 0): 163_840,
    (4096, 200, 1000): 164_840,
    (4096, 400, 0): 81_920,
    (4096, 400, 1000): 82_920,
    (1_048_576, 200, 0): 41_943_040,
    (1_048_576, 200, 1000): 41_944_040,
    (1_048_576, 400, 0): 20_971_520,
    (1_048_576, 400, 1000): 20_972_520,
}
T0 = 7_000
C_PS = 20_000
M_PS = 10_000
A_PS = 8_000
H_PS = 1_000
CONTROL_BYTES = 4096
EXPECTATIONS_COMMIT = "fc3836d"
REVIEW_EXPECTATIONS_COMMIT = "067cbfb"


def _validate_registry(out: Path, tier_b_only: bool, tier_b_producer: str | None) -> None:
    if not out.is_absolute():
        raise ValueError("study output path must be absolute")
    if len(JCT_PS) != len(SHAPES) * len(RATES_GBPS):
        raise AssertionError("CORE-5 matrix must contain four cells")
    for rate in RATES_GBPS:
        if JCT_PS[("serial", rate)] - JCT_PS[("parallel", rate)] != 10_000:
            raise AssertionError("frozen dependency delta drifted")
    for shape in SHAPES:
        if JCT_PS[(shape, 200)] - JCT_PS[(shape, 400)] != 163_840:
            raise AssertionError("frozen rate delta drifted")
    expected_tier_b_cells = (
        len(TIER_B_PAYLOADS) * len(RATES_GBPS) * len(TIER_B_DOORBELL_PS)
    )
    if len(TIER_B_JCT_PS) != expected_tier_b_cells:
        raise AssertionError("Tier B matrix must contain eight structural cells")
    for payload in TIER_B_PAYLOADS:
        for rate in RATES_GBPS:
            if (
                TIER_B_JCT_PS[(payload, rate, 1000)]
                - TIER_B_JCT_PS[(payload, rate, 0)]
                != 1000
            ):
                raise AssertionError("Tier B doorbell delta drifted")
    for relative in (
        "examples/core5_reduction/expectations.md",
        "examples/core5_reduction/review_expectations.md",
        "examples/rnic_live_v1/tier_b_expectations.md",
        "examples/rnic_live_v1/tier_b_review_expectations.json",
        "examples/rnic_live_v1/tier_b_review_supplement.md",
        "examples/rnic_live_v1/expectations.md",
        "examples/rnic_live_v1/tier_a_harness_expectations.md",
        "simllm/core/runtime.py",
        "simllm/core/step.py",
        "simllm/core/clock.py",
        "simllm/core/execution.py",
    ):
        if not (REPO_ROOT / relative).is_file():
            raise FileNotFoundError(f"frozen source-audit input is missing: {relative}")
    if tier_b_only and not tier_b_producer:
        raise ValueError("--tier-b-only requires --tier-b-producer")
    if tier_b_only:
        producer = Path(tier_b_producer).resolve(strict=False)
        resolved_out = out.resolve(strict=False)
        if not producer.is_relative_to(resolved_out):
            raise ValueError("Tier B producer must reside under its output directory")


def _profile(rate_gbps: int, *, control_service_ps: int = H_PS):
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
    from simllm.core import CoarseDeviceProfile

    engine = CopyEngineProfile(
        engine_id="copy",
        clock_hz=1_000_000_000,
        direction_profiles=(
            CopyDirectionProfile(
                direction=CopyDirection.HOST_TO_DEVICE,
                setup_cycles=0,
                bandwidth_bytes_per_cycle=1.0,
            ),
        ),
    )
    profile_id = "core5-study-gpu"
    architecture = GpuArchitectureProfile(
        profile_id=profile_id,
        gpu_name="core5-study",
        sm_count=1,
        warp_size=32,
        scheduler_count_per_sm=1,
        max_blocks_per_sm=1,
        max_warps_per_sm=1,
        max_threads_per_sm=32,
        max_threads_per_block=32,
        registers_per_sm=1024,
        max_registers_per_thread=32,
        register_allocation_granularity_per_warp=1,
        shared_memory_per_sm=1024,
        max_static_shared_memory_per_block=1024,
        max_shared_memory_per_block=1024,
        shared_memory_allocation_granularity=1,
        calibration=GpuCalibrationProfile(
            calibration_id="core5-study-calibration",
            target_architecture_profile_id=profile_id,
            provenance=GpuModelProvenance(
                source="synthetic CORE-5 study fixture",
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
                    issue_width_per_sm=1,
                ),
            ),
            memory=MemoryHierarchyProfile(
                hbm_latency_cycles=0,
                hbm_bandwidth_bytes_per_cycle=1,
            ),
            copy_engines=(engine,),
            warp_scheduler_policy=WarpSchedulerPolicy.LOOSE_ROUND_ROBIN,
            relative_uncertainty=0.0,
        ),
    )
    return CoarseDeviceProfile(
        rnic_rate_bps=rate_gbps * 1_000_000_000,
        nvlink_rate_bps=100_000_000_000,
        control_service_ps=control_service_ps,
        copy_engines=(CopyEngineServiceModel(architecture, engine.engine_id),),
    )


def _request_operations(request_id: str, base_rank: int, serial: bool):
    from simllm.core import (
        CollectiveWork,
        ComputeWork,
        ControlMode,
        ControlWork,
        DmaWork,
        ExecutionOperation,
        KvCacheAction,
        KvCacheWork,
        OperationCorrelation,
    )

    correlation = OperationCorrelation(request_ids=(request_id,))
    kv_id = f"{request_id}-kv"
    kernel_id = f"{request_id}-kernel"
    dma_id = f"{request_id}-dma"
    collective_id = f"{request_id}-collective"
    return (
        ExecutionOperation(
            kv_id,
            base_rank,
            f"{request_id}:kv",
            KvCacheWork(
                KvCacheAction.ALLOCATE,
                f"pool-{base_rank}",
                request_id=request_id,
            ),
            correlation=correlation,
        ),
        ExecutionOperation(
            kernel_id,
            base_rank,
            f"{request_id}:kernel",
            ComputeWork("kernel", nominal_duration_ps=C_PS),
            depends_on=(kv_id,),
            correlation=correlation,
        ),
        ExecutionOperation(
            dma_id,
            base_rank,
            f"{request_id}:dma",
            DmaWork("copy", "host", f"gpu:{base_rank}", M_PS // 1000),
            depends_on=(kernel_id,) if serial else (kv_id,),
            correlation=correlation,
        ),
        ExecutionOperation(
            collective_id,
            base_rank,
            f"{request_id}:collective",
            CollectiveWork(
                "all-reduce",
                (base_rank, base_rank + 1),
                100,
                "ring",
                request_id,
            ),
            depends_on=(kernel_id, dma_id),
            correlation=correlation,
        ),
        ExecutionOperation(
            f"{request_id}-control-0",
            base_rank,
            f"{request_id}:control-0",
            ControlWork(
                "feedback",
                (base_rank + 8,),
                CONTROL_BYTES,
                ControlMode.SYNCHRONOUS,
            ),
            depends_on=(collective_id,),
            correlation=correlation,
        ),
        ExecutionOperation(
            f"{request_id}-control-1",
            base_rank,
            f"{request_id}:control-1",
            ControlWork(
                "feedback",
                (base_rank + 8,),
                CONTROL_BYTES,
                ControlMode.SYNCHRONOUS,
            ),
            depends_on=(collective_id,),
            correlation=correlation,
        ),
    )


def _two_request_graph(step_index: int, release_ps: int, serial: bool):
    from simllm.core import ExecutionGraph

    return ExecutionGraph(
        execution_id=f"core5-study-{step_index}",
        step_index=step_index,
        released_at_ps=release_ps,
        operations=(
            *_request_operations("request-0", 0, serial),
            *_request_operations("request-1", 2, serial),
        ),
        completion_operation_ids=("request-0-control-1", "request-1-control-1"),
    )


def _step_record(step_index: int, release_ps: int):
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord

    phase = RequestPhase.PREFILL if step_index == 0 else RequestPhase.DECODE
    return StepRecord(
        step_index=step_index,
        virtual_time_ps=release_ps,
        scheduled=[
            ScheduledRequest("request-0", phase, 1),
            ScheduledRequest("request-1", phase, 1),
        ],
        num_sampled=2,
        sampled_request_ids=["request-0", "request-1"],
    )


def _fraction_json(value: Fraction | None) -> dict[str, int] | None:
    if value is None:
        return None
    return {"numerator": value.numerator, "denominator": value.denominator}


def _observation_check(
    records: list[dict[str, object]],
    check_id: str,
    observed: object,
    expected: object,
) -> None:
    passed = observed == expected
    records.append(
        {
            "check_id": check_id,
            "observed": observed,
            "expected": expected,
            "passed": passed,
        }
    )
    if not passed:
        raise AssertionError(f"structural observation failed: {check_id}")


def _scored_check(
    check_id: str,
    observed: object,
    expected: object,
    *,
    measurements: object | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "check_id": check_id,
        "observed": observed,
        "expected": expected,
        "passed": observed == expected,
        "genuine_risk": True,
    }
    if measurements is not None:
        record["measurements"] = measurements
    if isinstance(observed, int) and isinstance(expected, int):
        record["signed_residual"] = observed - expected
    return record


def _run_cell(shape: str, rate_gbps: int) -> dict[str, object]:
    from simllm.core import (
        AdditiveVisitTotals,
        CoarseDeviceRuntime,
        CompletionReducer,
        EventPhase,
        LatencyAttribution,
        VirtualClock,
    )

    serial = shape == "serial"
    network_ps = {200: 163_840, 400: 81_920}[rate_gbps]
    expected_jct = JCT_PS[(shape, rate_gbps)]
    expected_attribution = LatencyAttribution(
        queue_ps=network_ps,
        kernel_ps=C_PS,
        dma_ps=M_PS if serial else 0,
        collective_ps=A_PS,
        nic_ps=network_ps,
        control_ps=H_PS,
    )
    expected_additive = AdditiveVisitTotals(
        queue_wait_ps=network_ps,
        service_ps=C_PS + 2 * M_PS + 2 * A_PS + 2 * H_PS + 2 * network_ps,
        visit_count=21,
    )
    clock = VirtualClock(T0)
    reducer = CompletionReducer(clock)
    runtime = CoarseDeviceRuntime(_profile(rate_gbps))
    step_rows: list[dict[str, object]] = []
    structural_checks: list[dict[str, object]] = []

    for step_index in range(3):
        record = _step_record(step_index, clock.now_ps)
        graph = _two_request_graph(step_index, clock.now_ps, serial)
        streamed = []
        execution = runtime.execute(graph, on_event=streamed.append)
        report = runtime.last_report
        if report is None:
            raise AssertionError("runtime omitted its report")
        prefix = f"{shape}-{rate_gbps}-step-{step_index}"
        _observation_check(
            structural_checks,
            f"{prefix}-callback-count",
            len(streamed),
            len(execution.events),
        )
        _observation_check(
            structural_checks,
            f"{prefix}-callback-object-identity",
            all(
                left is right
                for left, right in zip(streamed, execution.events)
            ),
            True,
        )
        _observation_check(
            structural_checks,
            f"{prefix}-event-phase-membership",
            sorted(event.value for event in {row.phase for row in streamed}),
            sorted(event.value for event in EventPhase),
        )
        _observation_check(
            structural_checks,
            f"{prefix}-graph-additive-queue-wait",
            report.sum_visit_wait_ps,
            2 * network_ps,
        )
        _observation_check(
            structural_checks,
            f"{prefix}-critical-path-queue-wait",
            report.critical_path_queue_ps,
            network_ps,
        )

        step_result = reducer.reduce(record, graph, execution, report)
        if step_result.step_latency_ps != expected_jct:
            raise AssertionError("step latency missed the frozen oracle")
        if step_result.completed_at_ps != T0 + (step_index + 1) * expected_jct:
            raise AssertionError("absolute StepResult completion drifted")
        if step_result.additive_visit_totals != expected_additive + expected_additive:
            raise AssertionError("graph additive totals drifted")
        if len(step_result.request_metrics) != 2:
            raise AssertionError("two-request fixture lost a metric row")
        metric_rows = []
        for metric in step_result.request_metrics:
            expected_tpot = None if step_index == 0 else Fraction(expected_jct, 1)
            if metric.attribution != expected_attribution:
                raise AssertionError("request component row missed the frozen oracle")
            if metric.additive_visit_totals != expected_additive:
                raise AssertionError("request additive totals drifted")
            if (
                metric.latency_ps != expected_jct
                or metric.ttft_ps != expected_jct
                or metric.tpot_ps != expected_tpot
            ):
                raise AssertionError("request TTFT or TPOT drifted")
            if metric.additive_visit_totals.total_ps <= metric.latency_ps:
                raise AssertionError("additive work did not exceed request latency")
            metric_rows.append(
                {
                    "request_id": metric.request_id,
                    "token_index": metric.token_index,
                    "completed_at_ps": metric.completed_at_ps,
                    "latency_ps": metric.latency_ps,
                    "ttft_ps": metric.ttft_ps,
                    "tpot_ps": _fraction_json(metric.tpot_ps),
                    "attribution": asdict(metric.attribution),
                    "additive_visit_totals": asdict(metric.additive_visit_totals),
                }
            )
        step_rows.append(
            {
                "step_index": step_index,
                "step_latency_ps": step_result.step_latency_ps,
                "completed_at_ps": step_result.completed_at_ps,
                "request_metrics": metric_rows,
                "graph_additive_visit_totals": asdict(
                    step_result.additive_visit_totals
                ),
                "event_count": len(streamed),
            }
        )

    if clock.now_ps != T0 + 3 * expected_jct:
        raise AssertionError("final virtual clock missed the frozen closed form")
    measured_jcts = {row["step_latency_ps"] for row in step_rows}
    if len(measured_jcts) != 1:
        raise AssertionError("fixed cell produced unequal measured step latencies")
    measured_jct = measured_jcts.pop()
    return {
        "shape": shape,
        "rate_gbps": rate_gbps,
        "jct_ps": measured_jct,
        "expected_jct_ps": expected_jct,
        "network_ps": network_ps,
        "steps": step_rows,
        "final_clock_ps": clock.now_ps,
        "structural_checks": structural_checks,
    }


def _run_progress(kind: str, synchronous: bool) -> dict[str, object]:
    from simllm.core import (
        CoarseDeviceRuntime,
        CollectiveWork,
        CompletionReducer,
        ComputeWork,
        ControlMode,
        ControlWork,
        ExecutionGraph,
        ExecutionOperation,
        OperationCorrelation,
        RequestPhase,
        ScheduledRequest,
        StepRecord,
        VirtualClock,
    )

    correlation = OperationCorrelation(request_ids=("request",))
    if kind == "control":
        background = ExecutionOperation(
            "background",
            0,
            "control",
            ControlWork(
                "background",
                (8,),
                1_048_576,
                ControlMode.SYNCHRONOUS
                if synchronous
                else ControlMode.ASYNCHRONOUS,
            ),
            correlation=correlation,
        )
    else:
        background = ExecutionOperation(
            "background",
            0,
            "collective",
            CollectiveWork("all-reduce", (0, 8), 1_048_576, "ring"),
            correlation=correlation,
        )
    anchor = ExecutionOperation(
        "anchor",
        1,
        "compute",
        ComputeWork("anchor", nominal_duration_ps=10_000),
        correlation=correlation,
    )
    if synchronous or kind == "control":
        required = ("background", "anchor")
    else:
        required = ("anchor",)
    graph = ExecutionGraph("progress", 0, T0, (background, anchor), required)
    record = StepRecord(
        0,
        T0,
        [ScheduledRequest("request", RequestPhase.PREFILL, 1)],
        num_sampled=1,
        sampled_request_ids=["request"],
    )
    clock = VirtualClock(T0)
    runtime = CoarseDeviceRuntime(_profile(400, control_service_ps=0))
    execution = runtime.execute(graph)
    if runtime.last_report is None:
        raise AssertionError("runtime omitted progress report")
    step = CompletionReducer(clock).reduce(record, graph, execution, runtime.last_report)
    expected = 20_971_520 if synchronous else 10_000
    if step.step_latency_ps != expected or clock.now_ps != T0 + expected:
        raise AssertionError("progress boundary missed the frozen oracle")
    if execution.quiesced_at_ps != T0 + 20_971_520:
        raise AssertionError("progress quiescence missed the frozen oracle")
    if not synchronous and not any(
        event.timestamp_ps > step.completed_at_ps for event in execution.events
    ):
        raise AssertionError("asynchronous progress emitted no future event")
    return {
        "kind": kind,
        "synchronous": synchronous,
        "step_latency_ps": step.step_latency_ps,
        "completed_at_ps": step.completed_at_ps,
        "quiesced_at_ps": execution.quiesced_at_ps,
        "event_count": len(execution.events),
    }


def _family(
    checks: list[dict[str, object]], rationale: str
) -> dict[str, object]:
    passed = sum(check["passed"] is True for check in checks)
    total = len(checks)
    genuine_risk_instances = sum(
        check["genuine_risk"] is True for check in checks
    )
    if passed != total:
        failed = [check["check_id"] for check in checks if not check["passed"]]
        raise AssertionError(
            f"behavioral family failed ({rationale}): {failed}"
        )
    return {
        "passed": passed,
        "total": total,
        "genuine_risk_instances": genuine_risk_instances,
        "genuine_risk_fraction": f"{genuine_risk_instances}/{total}",
        "rationale": rationale,
        "checks": checks,
    }


def _unscored_summary(
    records: list[dict[str, object]],
) -> dict[str, object]:
    passed = sum(record["passed"] is True for record in records)
    total = len(records)
    if passed != total:
        failed = [record["check_id"] for record in records if not record["passed"]]
        raise AssertionError(f"unscored evidence failed: {failed}")
    return {
        "passed": passed,
        "total": total,
        "scored": False,
        "records": records,
    }


def _run_reducer_regression_probes() -> dict[str, list[dict[str, object]]]:
    from simllm.core import (
        CoarseDeviceRuntime,
        CompletionReducer,
        ComputeWork,
        ExecutionGraph,
        ExecutionOperation,
        OperationCorrelation,
        RequestPhase,
        ScheduledRequest,
        StepRecord,
        VirtualClock,
        step_record_from_json,
        step_record_to_json,
    )

    def operation(
        operation_id: str,
        request_ids: tuple[str, ...],
        duration_ps: int,
    ) -> ExecutionOperation:
        return ExecutionOperation(
            operation_id,
            0,
            operation_id,
            ComputeWork(operation_id, nominal_duration_ps=duration_ps),
            correlation=OperationCorrelation(request_ids=request_ids),
        )

    rejections: list[dict[str, object]] = []
    compatibility: list[dict[str, object]] = []

    replay_clock = VirtualClock(0)
    replay_reducer = CompletionReducer(replay_clock)
    replay_runtime = CoarseDeviceRuntime(_profile(400))
    replay_graph = ExecutionGraph(
        "probe-zero-replay",
        0,
        0,
        (operation("zero", ("request",), 0),),
        ("zero",),
    )
    replay_record = StepRecord(
        0,
        0,
        [ScheduledRequest("request", RequestPhase.PREFILL, 1)],
        num_sampled=1,
        sampled_request_ids=["request"],
    )
    replay_execution = replay_runtime.execute(replay_graph)
    if replay_runtime.last_report is None:
        raise AssertionError("replay probe runtime omitted its report")
    first_replay = replay_reducer.reduce(
        replay_record,
        replay_graph,
        replay_execution,
        replay_runtime.last_report,
    )
    replay_before = (
        replay_clock.now_ps,
        replay_reducer.latest_request_metrics,
    )
    replay_exception: Exception | None = None
    try:
        replay_reducer.reduce(
            replay_record,
            replay_graph,
            replay_execution,
            replay_runtime.last_report,
        )
    except Exception as error:  # noqa: BLE001 - exception is raw probe evidence
        replay_exception = error
    replay_after = (
        replay_clock.now_ps,
        replay_reducer.latest_request_metrics,
    )
    replay_passed = (
        isinstance(replay_exception, ValueError)
        and "execution ID has already been reduced" in str(replay_exception)
        and replay_before == replay_after
        and first_replay.request_metrics[0].token_index == 1
        and first_replay.request_metrics[0].tpot_ps is None
    )
    rejections.append(
        {
            "check_id": "duplicate-zero-latency-execution",
            "exception_type": (
                type(replay_exception).__name__
                if replay_exception is not None
                else None
            ),
            "exception_message": (
                str(replay_exception) if replay_exception is not None else None
            ),
            "state_unchanged": replay_before == replay_after,
            "passed": replay_passed,
        }
    )

    ambiguous_clock = VirtualClock(0)
    ambiguous_reducer = CompletionReducer(ambiguous_clock)
    ambiguous_runtime = CoarseDeviceRuntime(_profile(400))
    ambiguous_graph = ExecutionGraph(
        "probe-ambiguous-sample",
        0,
        0,
        (operation("shared", ("a", "b"), 100),),
        ("shared",),
    )
    ambiguous_record = StepRecord(
        0,
        0,
        [
            ScheduledRequest("a", RequestPhase.PREFILL, 1),
            ScheduledRequest("b", RequestPhase.PREFILL, 1),
        ],
        num_sampled=1,
    )
    ambiguous_execution = ambiguous_runtime.execute(ambiguous_graph)
    if ambiguous_runtime.last_report is None:
        raise AssertionError("ambiguous-sample probe runtime omitted its report")
    ambiguous_exception: Exception | None = None
    try:
        ambiguous_reducer.reduce(
            ambiguous_record,
            ambiguous_graph,
            ambiguous_execution,
            ambiguous_runtime.last_report,
        )
    except Exception as error:  # noqa: BLE001 - exception is raw probe evidence
        ambiguous_exception = error
    rejections.append(
        {
            "check_id": "ambiguous-partial-sample",
            "exception_type": (
                type(ambiguous_exception).__name__
                if ambiguous_exception is not None
                else None
            ),
            "exception_message": (
                str(ambiguous_exception)
                if ambiguous_exception is not None
                else None
            ),
            "state_unchanged": (
                ambiguous_clock.now_ps == 0
                and ambiguous_reducer.latest_request_metrics == ()
            ),
            "passed": (
                isinstance(ambiguous_exception, ValueError)
                and "CORE-17" in str(ambiguous_exception)
                and ambiguous_clock.now_ps == 0
                and ambiguous_reducer.latest_request_metrics == ()
            ),
        }
    )

    zero_clock = VirtualClock(0)
    zero_reducer = CompletionReducer(zero_clock)
    zero_runtime = CoarseDeviceRuntime(_profile(400))
    zero_graph = ExecutionGraph(
        "probe-zero-samples",
        0,
        0,
        (operation("decode-zero", ("request",), 100),),
        ("decode-zero",),
    )
    zero_record = StepRecord(
        0,
        0,
        [ScheduledRequest("request", RequestPhase.DECODE, 1)],
        num_sampled=0,
    )
    zero_execution = zero_runtime.execute(zero_graph)
    if zero_runtime.last_report is None:
        raise AssertionError("zero-sample probe runtime omitted its report")
    zero_step = zero_reducer.reduce(
        zero_record,
        zero_graph,
        zero_execution,
        zero_runtime.last_report,
    )
    sampled_graph = ExecutionGraph(
        "probe-after-zero-samples",
        1,
        100,
        (operation("decode-sampled", ("request",), 50),),
        ("decode-sampled",),
    )
    sampled_record = StepRecord(
        1,
        100,
        [ScheduledRequest("request", RequestPhase.DECODE, 1)],
        num_sampled=1,
        sampled_request_ids=["request"],
    )
    sampled_execution = zero_runtime.execute(sampled_graph)
    if zero_runtime.last_report is None:
        raise AssertionError("follow-up sample probe runtime omitted its report")
    sampled_step = zero_reducer.reduce(
        sampled_record,
        sampled_graph,
        sampled_execution,
        zero_runtime.last_report,
    )
    zero_passed = (
        zero_step.request_metrics == ()
        and len(sampled_step.request_metrics) == 1
        and sampled_step.request_metrics[0].latency_ps == 150
        and sampled_step.request_metrics[0].ttft_ps == 150
        and zero_clock.now_ps == 150
    )
    compatibility.append(
        {
            "check_id": "zero-sampled-decode-is-empty",
            "observed_first_metric_count": len(zero_step.request_metrics),
            "observed_follow_up_latency_ps": (
                sampled_step.request_metrics[0].latency_ps
            ),
            "observed_clock_ps": zero_clock.now_ps,
            "passed": zero_passed,
        }
    )

    null_payload = step_record_to_json(zero_record)
    null_payload["sampled_request_ids"] = None
    loaded = step_record_from_json(null_payload)
    rewritten = step_record_to_json(loaded)
    compatibility.append(
        {
            "check_id": "explicit-null-sampled-identities",
            "loaded_as_none": loaded.sampled_request_ids is None,
            "canonical_writer_omits_field": "sampled_request_ids" not in rewritten,
            "passed": (
                loaded.sampled_request_ids is None
                and "sampled_request_ids" not in rewritten
            ),
        }
    )
    return {
        "validator_rejections": rejections,
        "compatibility_acceptance": compatibility,
    }


def _run(out: Path) -> dict[str, object]:
    cells = [
        _run_cell(shape, rate)
        for shape in SHAPES
        for rate in RATES_GBPS
    ]
    by_cell = {(row["shape"], row["rate_gbps"]): row for row in cells}
    dependency_relations = []
    dependency_checks: list[dict[str, object]] = []
    for rate in RATES_GBPS:
        delta = (
            by_cell[("serial", rate)]["jct_ps"]
            - by_cell[("parallel", rate)]["jct_ps"]
        )
        dependency_relations.append({"rate_gbps": rate, "signed_delta_ps": delta})
        dependency_checks.append(
            _scored_check(f"rate-{rate}", delta, 10_000)
        )
    rate_relations = []
    rate_checks: list[dict[str, object]] = []
    for shape in SHAPES:
        delta = by_cell[(shape, 200)]["jct_ps"] - by_cell[(shape, 400)]["jct_ps"]
        rate_relations.append({"shape": shape, "signed_delta_ps": delta})
        rate_checks.append(
            _scored_check(shape, delta, 163_840)
        )

    component_checks: list[dict[str, object]] = []
    additive_checks: list[dict[str, object]] = []
    for cell in cells:
        shape = str(cell["shape"])
        rate = int(cell["rate_gbps"])
        expected_jct = JCT_PS[(shape, rate)]
        network_ps = int(cell["network_ps"])
        expected_attribution = {
            "queue_ps": network_ps,
            "kv_ps": 0,
            "kernel_ps": C_PS,
            "dma_ps": M_PS if shape == "serial" else 0,
            "collective_ps": A_PS,
            "nic_ps": network_ps,
            "control_ps": H_PS,
        }
        steps = list(cell["steps"])
        for request_id in ("request-0", "request-1"):
            observed_metrics = []
            expected_metrics = []
            for step in steps:
                metric = next(
                    row
                    for row in step["request_metrics"]
                    if row["request_id"] == request_id
                )
                observed_metrics.append(
                    {
                        "step_index": step["step_index"],
                        "token_index": metric["token_index"],
                        "latency_ps": metric["latency_ps"],
                        "ttft_ps": metric["ttft_ps"],
                        "tpot_ps": metric["tpot_ps"],
                        "attribution": metric["attribution"],
                    }
                )
                step_index = int(step["step_index"])
                expected_metrics.append(
                    {
                        "step_index": step_index,
                        "token_index": step_index + 1,
                        "latency_ps": expected_jct,
                        "ttft_ps": expected_jct,
                        "tpot_ps": (
                            None
                            if step_index == 0
                            else {"numerator": expected_jct, "denominator": 1}
                        ),
                        "attribution": expected_attribution,
                    }
                )
            component_checks.append(
                _scored_check(
                    f"{shape}-{rate}-{request_id}",
                    observed_metrics,
                    expected_metrics,
                )
            )

        additive_measurements = []
        for step in steps:
            graph_totals = step["graph_additive_visit_totals"]
            for metric in step["request_metrics"]:
                additive = metric["additive_visit_totals"]
                additive_measurements.append(
                    {
                        "step_index": step["step_index"],
                        "request_id": metric["request_id"],
                        "request_additive_total_ps": (
                            additive["queue_wait_ps"]
                            + additive["service_ps"]
                            + additive["visibility_ps"]
                        ),
                        "request_latency_ps": metric["latency_ps"],
                        "graph_queue_wait_ps": graph_totals["queue_wait_ps"],
                        "request_queue_component_ps": metric["attribution"]["queue_ps"],
                    }
                )
        additive_predicates = {
            "request_additive_exceeds_latency": all(
                row["request_additive_total_ps"] > row["request_latency_ps"]
                for row in additive_measurements
            ),
            "graph_additive_queue_is_twice_selected_queue": all(
                row["graph_queue_wait_ps"]
                == 2 * row["request_queue_component_ps"]
                for row in additive_measurements
            ),
        }
        additive_checks.append(
            _scored_check(
                f"{shape}-{rate}",
                additive_predicates,
                {
                    "request_additive_exceeds_latency": True,
                    "graph_additive_queue_is_twice_selected_queue": True,
                },
                measurements=additive_measurements,
            )
        )

    progress_rows = [
        _run_progress(kind, synchronous)
        for kind in ("control", "collective")
        for synchronous in (False, True)
    ]
    asynchronous_checks: list[dict[str, object]] = []
    for kind in ("control", "collective"):
        by_mode = {
            row["synchronous"]: row
            for row in progress_rows
            if row["kind"] == kind
        }
        observed_delta = (
            by_mode[True]["step_latency_ps"]
            - by_mode[False]["step_latency_ps"]
        )
        asynchronous_checks.append(
            _scored_check(kind, observed_delta, 20_961_520)
        )

    families = {
        "dependency_shape": _family(
            dependency_checks,
            "A missing realized predecessor segment can erase either rate's DMA penalty.",
        ),
        "inverse_rate_tail": _family(
            rate_checks,
            "A lost or duplicated NIC tail can change either shape's signed rate delta.",
        ),
        "request_metrics_and_components": _family(
            component_checks,
            "The total can remain correct while one of seven component owners is wrong.",
        ),
        "additive_work_separation": _family(
            additive_checks,
            "Substituting visit work for a selected path changes every nonzero cell.",
        ),
        "asynchronous_boundaries": _family(
            asynchronous_checks,
            "Using quiescence or ignoring a required boundary fails control or collective.",
        ),
    }
    structural_checks = [
        check
        for cell in cells
        for check in cell.pop("structural_checks")
    ]
    regression_probes = _run_reducer_regression_probes()
    report = {
        "schema": "simllm-core5-reduction-study-v1",
        "expectations_commit": REVIEW_EXPECTATIONS_COMMIT,
        "expectations_commits": [
            EXPECTATIONS_COMMIT,
            REVIEW_EXPECTATIONS_COMMIT,
        ],
        "run_configurations": [
            {"shape": shape, "rate_gbps": rate}
            for shape in SHAPES
            for rate in RATES_GBPS
        ],
        "exact_oracle_rows": cells,
        "behavioral_relations": {
            "dependency_shape": dependency_relations,
            "inverse_rate_tail": rate_relations,
            "asynchronous_progress": progress_rows,
        },
        "behavioral_families": families,
        "fatal_structural_guards": _unscored_summary(structural_checks),
        "validator_rejection_probes": _unscored_summary(
            regression_probes["validator_rejections"]
        ),
        "compatibility_acceptance_probes": _unscored_summary(
            regression_probes["compatibility_acceptance"]
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote CORE-5 study evidence to {path}")
    for name, family in families.items():
        print(f"{name} genuine-risk fraction: {family['genuine_risk_fraction']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--tier-b-only", action="store_true")
    parser.add_argument("--tier-b-producer")
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.out is None:
        data_root = path_from_env("SIMLLM_DATA_ROOT")
        if data_root is None:
            parser.error("--out is required when SIMLLM_DATA_ROOT is not set")
        arguments.out = data_root / "core5_reduction"
        if arguments.tier_b_only:
            arguments.out /= "tier_b"
    _validate_registry(arguments.out, arguments.tier_b_only, arguments.tier_b_producer)
    if not arguments.check_only:
        if arguments.tier_b_only:
            raise RuntimeError(
                "Tier B execution requires the composed HTSIM-9/CORE-15 producer; "
                "the frozen registry is ready but no producer contract is landed"
            )
        _run(arguments.out.resolve())
        return
    scope = "Tier B" if arguments.tier_b_only else "CORE-5"
    print(f"{scope} frozen registry check passed; no artifacts produced")


if __name__ == "__main__":
    main()
