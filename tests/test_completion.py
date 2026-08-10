from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

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
    AdditiveVisitTotals,
    CoarseDeviceProfile,
    CoarseDeviceRuntime,
    CollectiveWork,
    CompletionReducer,
    ComputeWork,
    ControlMode,
    ControlWork,
    DmaWork,
    EventPhase,
    ExecutionGraph,
    ExecutionOperation,
    KvCacheAction,
    KvCacheWork,
    LatencyAttribution,
    OperationCorrelation,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    VirtualClock,
    step_record_from_json,
    step_record_to_json,
)

T0 = 7_000
C_PS = 20_000
M_PS = 10_000
A_PS = 8_000
H_PS = 1_000
CONTROL_BYTES = 4096


def _profile(rate_gbps: int, *, control_service_ps: int = H_PS) -> CoarseDeviceProfile:
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
    profile_id = "core5-fixture-gpu"
    architecture = GpuArchitectureProfile(
        profile_id=profile_id,
        gpu_name="core5-fixture",
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
            calibration_id="core5-fixture-calibration",
            target_architecture_profile_id=profile_id,
            provenance=GpuModelProvenance(
                source="synthetic CORE-5 fixture",
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


def _correlation(request_id: str) -> OperationCorrelation:
    return OperationCorrelation(request_ids=(request_id,))


def _request_operations(
    request_id: str,
    base_rank: int,
    *,
    serial: bool,
) -> tuple[ExecutionOperation, ...]:
    prefix = request_id
    kv_id = f"{prefix}-kv"
    kernel_id = f"{prefix}-kernel"
    dma_id = f"{prefix}-dma"
    collective_id = f"{prefix}-collective"
    correlation = _correlation(request_id)
    return (
        ExecutionOperation(
            kv_id,
            base_rank,
            f"{prefix}:kv",
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
            f"{prefix}:kernel",
            ComputeWork("kernel", nominal_duration_ps=C_PS),
            depends_on=(kv_id,),
            correlation=correlation,
        ),
        ExecutionOperation(
            dma_id,
            base_rank,
            f"{prefix}:dma",
            DmaWork("copy", "host", f"gpu:{base_rank}", M_PS // 1000),
            depends_on=(kernel_id,) if serial else (kv_id,),
            correlation=correlation,
        ),
        ExecutionOperation(
            collective_id,
            base_rank,
            f"{prefix}:collective",
            CollectiveWork(
                "all-reduce",
                (base_rank, base_rank + 1),
                100,
                "ring",
                prefix,
            ),
            depends_on=(kernel_id, dma_id),
            correlation=correlation,
        ),
        ExecutionOperation(
            f"{prefix}-control-0",
            base_rank,
            f"{prefix}:control-0",
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
            f"{prefix}-control-1",
            base_rank,
            f"{prefix}:control-1",
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


def _two_request_graph(step_index: int, release_ps: int, serial: bool) -> ExecutionGraph:
    operations = (
        *_request_operations("request-0", 0, serial=serial),
        *_request_operations("request-1", 2, serial=serial),
    )
    return ExecutionGraph(
        execution_id=f"core5-{step_index}",
        step_index=step_index,
        released_at_ps=release_ps,
        operations=operations,
        completion_operation_ids=("request-0-control-1", "request-1-control-1"),
    )


def _step_record(step_index: int, release_ps: int) -> StepRecord:
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


@pytest.mark.parametrize("serial", [False, True], ids=["parallel", "serial"])
@pytest.mark.parametrize("rate_gbps", [200, 400])
def test_two_request_reduction_matches_frozen_closed_forms(serial, rate_gbps):
    network_ps = {200: 163_840, 400: 81_920}[rate_gbps]
    expected_jct = (
        C_PS + (M_PS if serial else 0) + A_PS + H_PS + 2 * network_ps
    )
    expected_attribution = LatencyAttribution(
        queue_ps=network_ps,
        kv_ps=0,
        kernel_ps=C_PS,
        dma_ps=M_PS if serial else 0,
        collective_ps=A_PS,
        nic_ps=network_ps,
        control_ps=H_PS,
    )
    expected_additive = AdditiveVisitTotals(
        queue_wait_ps=network_ps,
        service_ps=C_PS + 2 * M_PS + 2 * A_PS + 2 * H_PS + 2 * network_ps,
        visibility_ps=0,
        visit_count=21,
    )
    clock = VirtualClock(T0)
    reducer = CompletionReducer(clock)
    runtime = CoarseDeviceRuntime(_profile(rate_gbps))

    for step_index in range(3):
        record = _step_record(step_index, clock.now_ps)
        graph = _two_request_graph(step_index, clock.now_ps, serial)
        streamed = []
        execution_result = runtime.execute(graph, on_event=streamed.append)
        report = runtime.last_report
        assert report is not None
        assert all(
            left is right
            for left, right in zip(streamed, execution_result.events, strict=True)
        )
        assert {event.phase for event in streamed} == set(EventPhase)
        step_result = reducer.reduce(record, graph, execution_result, report)

        assert step_result.step_latency_ps == expected_jct
        assert step_result.completed_at_ps == T0 + (step_index + 1) * expected_jct
        assert step_result.additive_visit_totals == expected_additive + expected_additive
        assert report.sum_visit_wait_ps == 2 * network_ps
        assert report.critical_path_queue_ps == network_ps
        assert len(step_result.request_metrics) == 2
        for metric in step_result.request_metrics:
            assert metric.completed_at_ps == step_result.completed_at_ps
            assert metric.latency_ps == expected_jct
            assert metric.ttft_ps == expected_jct
            assert metric.tpot_ps == (
                None if step_index == 0 else Fraction(expected_jct, 1)
            )
            assert metric.attribution == expected_attribution
            assert metric.additive_visit_totals == expected_additive
            assert metric.additive_visit_totals.total_ps > metric.latency_ps

    assert clock.now_ps == T0 + 3 * expected_jct
    assert {metric.request_id for metric in reducer.latest_request_metrics} == {
        "request-0",
        "request-1",
    }


def _reduce_progress_graph(*, kind: str, synchronous: bool):
    correlation = _correlation("request")
    if kind == "control":
        background = ExecutionOperation(
            "background",
            0,
            "control",
            ControlWork(
                "background",
                (8,),
                1_048_576,
                (
                    ControlMode.SYNCHRONOUS
                    if synchronous
                    else ControlMode.ASYNCHRONOUS
                ),
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
    required = (
        ("background", "anchor")
        if synchronous
        else (("background", "anchor") if kind == "control" else ("anchor",))
    )
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
    execution_result = runtime.execute(graph)
    assert runtime.last_report is not None
    step_result = CompletionReducer(clock).reduce(
        record,
        graph,
        execution_result,
        runtime.last_report,
    )
    return clock, execution_result, step_result


@pytest.mark.parametrize("kind", ["control", "collective"])
def test_asynchronous_progress_stops_clock_at_framework_boundary(kind):
    async_clock, async_result, async_step = _reduce_progress_graph(
        kind=kind,
        synchronous=False,
    )
    sync_clock, sync_result, sync_step = _reduce_progress_graph(
        kind=kind,
        synchronous=True,
    )
    background_ps = 20_971_520
    assert async_step.step_latency_ps == 10_000
    assert async_clock.now_ps == T0 + 10_000
    assert async_result.quiesced_at_ps == T0 + background_ps
    assert any(event.timestamp_ps > async_step.completed_at_ps for event in async_result.events)
    assert sync_step.step_latency_ps == background_ps
    assert sync_clock.now_ps == T0 + background_ps
    assert sync_result.completed_at_ps == sync_result.quiesced_at_ps
    assert sync_step.step_latency_ps - async_step.step_latency_ps == 20_961_520


def test_partial_sample_identity_fails_atomically_until_explicit():
    graph = ExecutionGraph(
        "sample-identity",
        0,
        T0,
        (
            ExecutionOperation(
                "shared",
                0,
                "compute",
                ComputeWork("shared", nominal_duration_ps=100),
                correlation=OperationCorrelation(request_ids=("a", "b")),
            ),
        ),
        ("shared",),
    )
    runtime = CoarseDeviceRuntime(_profile(400))
    execution_result = runtime.execute(graph)
    assert runtime.last_report is not None
    ambiguous = StepRecord(
        0,
        T0,
        [
            ScheduledRequest("a", RequestPhase.PREFILL, 1),
            ScheduledRequest("b", RequestPhase.PREFILL, 1),
        ],
        num_sampled=1,
    )
    clock = VirtualClock(T0)
    reducer = CompletionReducer(clock)
    with pytest.raises(ValueError, match="CORE-17"):
        reducer.reduce(ambiguous, graph, execution_result, runtime.last_report)
    assert clock.now_ps == T0
    assert reducer.latest_request_metrics == ()

    explicit = replace(ambiguous, sampled_request_ids=["a"])
    step_result = reducer.reduce(explicit, graph, execution_result, runtime.last_report)
    assert [metric.request_id for metric in step_result.request_metrics] == ["a"]
    assert clock.now_ps == T0 + 100


def test_non_sampling_prefill_accumulates_into_first_token_ttft():
    correlation = _correlation("request")
    clock = VirtualClock(0)
    reducer = CompletionReducer(clock)
    runtime = CoarseDeviceRuntime(_profile(400))

    first_graph = ExecutionGraph(
        "chunk-0",
        0,
        0,
        (
            ExecutionOperation(
                "compute-0",
                0,
                "compute",
                ComputeWork("chunk", nominal_duration_ps=100),
                correlation=correlation,
            ),
        ),
        ("compute-0",),
    )
    first_record = StepRecord(
        0,
        0,
        [ScheduledRequest("request", RequestPhase.PREFILL, 1)],
        num_sampled=0,
        sampled_request_ids=[],
    )
    first_execution = runtime.execute(first_graph)
    assert runtime.last_report is not None
    first_step = reducer.reduce(
        first_record,
        first_graph,
        first_execution,
        runtime.last_report,
    )
    assert first_step.request_metrics == ()
    assert clock.now_ps == 100

    second_graph = ExecutionGraph(
        "chunk-1",
        1,
        100,
        (
            ExecutionOperation(
                "compute-1",
                0,
                "compute",
                ComputeWork("sample", nominal_duration_ps=200),
                correlation=correlation,
            ),
        ),
        ("compute-1",),
    )
    second_record = StepRecord(
        1,
        100,
        [ScheduledRequest("request", RequestPhase.PREFILL, 1)],
        num_sampled=1,
        sampled_request_ids=["request"],
    )
    second_execution = runtime.execute(second_graph)
    assert runtime.last_report is not None
    second_step = reducer.reduce(
        second_record,
        second_graph,
        second_execution,
        runtime.last_report,
    )
    metric = second_step.request_metrics[0]
    assert metric.ttft_ps == metric.latency_ps == 300
    assert metric.attribution == LatencyAttribution(kernel_ps=300)
    assert metric.additive_visit_totals.service_ps == 300


def test_request_tail_includes_wait_for_the_next_scheduler_release():
    clock = VirtualClock(0)
    reducer = CompletionReducer(clock)
    runtime = CoarseDeviceRuntime(_profile(400))

    for step_index, release_ps in enumerate((0, 200)):
        graph = ExecutionGraph(
            f"batch-{step_index}",
            step_index,
            release_ps,
            (
                ExecutionOperation(
                    f"short-{step_index}",
                    0,
                    "short",
                    ComputeWork("short", nominal_duration_ps=100 if step_index == 0 else 50),
                    correlation=_correlation("short"),
                ),
                ExecutionOperation(
                    f"long-{step_index}",
                    1,
                    "long",
                    ComputeWork("long", nominal_duration_ps=200),
                    correlation=_correlation("long"),
                ),
            ),
            (f"short-{step_index}", f"long-{step_index}"),
        )
        phase = RequestPhase.PREFILL if step_index == 0 else RequestPhase.DECODE
        record = StepRecord(
            step_index,
            release_ps,
            [
                ScheduledRequest("short", phase, 1),
                ScheduledRequest("long", phase, 1),
            ],
            num_sampled=2,
            sampled_request_ids=["short", "long"],
        )
        execution = runtime.execute(graph)
        assert runtime.last_report is not None
        step = reducer.reduce(record, graph, execution, runtime.last_report)

    short = next(metric for metric in step.request_metrics if metric.request_id == "short")
    assert short.latency_ps == 150
    assert short.attribution == LatencyAttribution(queue_ps=100, kernel_ps=50)
    assert short.tpot_ps == Fraction(150, 1)


def test_sampled_request_identity_is_optional_in_the_v1_record_reader():
    legacy = StepRecord(
        0,
        0,
        [ScheduledRequest("request", RequestPhase.PREFILL, 1)],
        num_sampled=1,
    )
    legacy_payload = step_record_to_json(legacy)
    assert "sampled_request_ids" not in legacy_payload
    assert step_record_from_json(legacy_payload) == legacy

    exact = replace(legacy, sampled_request_ids=["request"])
    payload = step_record_to_json(exact)
    assert payload["sampled_request_ids"] == ["request"]
    assert step_record_from_json(payload) == exact
