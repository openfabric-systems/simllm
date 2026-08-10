from __future__ import annotations

from dataclasses import replace

import pytest

from simllm.compute import (
    CopyDirection,
    CopyDirectionProfile,
    CopyEngineProfile,
    CopyEngineServiceModel,
    CtaTrace,
    GpuArchitectureProfile,
    GpuCalibrationProfile,
    GpuModelProvenance,
    KernelLaunch,
    MemoryHierarchyProfile,
    PipelineKind,
    PipelineProfile,
    SassInstruction,
    SassWarpTrace,
    SmSchedulerModel,
    WarpSchedulerPolicy,
)
from simllm.core import (
    AtlahsWqeLedger,
    CoarseDeviceProfile,
    CoarseDeviceRuntime,
    CollectiveWork,
    CompletionEvent,
    ComputeWork,
    ControlMode,
    ControlWork,
    CreatedObjectKind,
    CreatedObjectRecord,
    DmaWork,
    EventPhase,
    ExecutionGraph,
    ExecutionOperation,
    IdentityArbitrationPolicy,
    KvCacheAction,
    KvCacheWork,
    QueueVisit,
    RequestBookkeeper,
    ResourceKind,
    ResourceRef,
    RnicAuthorityMode,
    SemanticWqeSubmission,
    WqeLifecycleProjection,
    collective_goal_tags,
    validate_bookkeeping_ledger,
)
from simllm.traffic import render_serial_execution_graph_goal


def _architecture(
    engines: tuple[CopyEngineProfile, ...] = (),
) -> GpuArchitectureProfile:
    profile_id = "runtime-test-gpu"
    calibration = GpuCalibrationProfile(
        calibration_id="runtime-test-calibration",
        target_architecture_profile_id=profile_id,
        provenance=GpuModelProvenance(
            source="synthetic test fixture",
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
        copy_engines=engines,
        warp_scheduler_policy=WarpSchedulerPolicy.LOOSE_ROUND_ROBIN,
        relative_uncertainty=0.0,
    )
    return GpuArchitectureProfile(
        profile_id=profile_id,
        gpu_name="runtime-test-gpu-name",
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


def _engine(
    engine_id: str,
    directions: tuple[CopyDirection, ...],
    *,
    bytes_per_cycle: float = 1.0,
) -> CopyEngineProfile:
    return CopyEngineProfile(
        engine_id=engine_id,
        clock_hz=1_000_000_000,
        direction_profiles=tuple(
            CopyDirectionProfile(
                direction=direction,
                setup_cycles=0,
                bandwidth_bytes_per_cycle=bytes_per_cycle,
            )
            for direction in directions
        ),
    )


def _profile(
    *,
    rate_gbps: int = 400,
    engines: tuple[CopyEngineProfile, ...] | None = None,
) -> CoarseDeviceProfile:
    if engines is None:
        engines = (
            _engine(
                "copy-all",
                (
                    CopyDirection.HOST_TO_DEVICE,
                    CopyDirection.DEVICE_TO_HOST,
                    CopyDirection.DEVICE_TO_DEVICE,
                    CopyDirection.PEER_TO_PEER,
                ),
            ),
        )
    architecture = _architecture(engines)
    return CoarseDeviceProfile(
        rnic_rate_bps=rate_gbps * 1_000_000_000,
        copy_engines=tuple(
            CopyEngineServiceModel(architecture, engine.engine_id) for engine in engines
        ),
    )


def _launch(name: str) -> KernelLaunch:
    return KernelLaunch(
        implementation_id=name,
        trace_id=f"{name}-trace",
        grid_blocks=1,
        threads_per_block=32,
        registers_per_thread=0,
        static_shared_memory_bytes=0,
        dynamic_shared_memory_bytes=0,
        cta_traces=(
            CtaTrace(
                trace_class_id=f"{name}-cta",
                block_ids=(0,),
                warp_traces=(
                    SassWarpTrace(
                        warp_id=0,
                        instructions=(
                            SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU),
                        ),
                    ),
                ),
            ),
        ),
    )


def _overlap_graph(
    execution_id: str,
    *,
    compute_ps: int,
    dma_ps: int,
    dependent: bool,
    compute_hbm_bytes: int = 0,
) -> ExecutionGraph:
    return ExecutionGraph(
        execution_id,
        0,
        5000,
        (
            ExecutionOperation(
                "compute",
                0,
                "cuda:0:compute",
                ComputeWork(
                    "compute",
                    nominal_duration_ps=compute_ps,
                    hbm_bytes=compute_hbm_bytes,
                ),
            ),
            ExecutionOperation(
                "dma",
                0,
                "cuda:0:copy",
                DmaWork("dma", "host:pinned", "gpu:0:hbm", dma_ps // 1000),
                depends_on=("compute",) if dependent else (),
            ),
        ),
    )


def test_profile_and_authority_modes_are_explicit_and_exclusive():
    with pytest.raises(ValueError, match="exactly eight GPUs"):
        CoarseDeviceProfile(gpus_per_node=4)
    with pytest.raises(ValueError, match="requires a native"):
        CoarseDeviceRuntime(authority_mode=RnicAuthorityMode.STRUCTURAL)
    with pytest.raises(ValueError, match="cannot also supply"):
        CoarseDeviceRuntime(native_session=object())
    with pytest.raises(TypeError, match="ArbitrationPolicy"):
        CoarseDeviceRuntime(arbitration_policy=object())

    runtime = CoarseDeviceRuntime()
    assert isinstance(runtime.bypass_ledger, AtlahsWqeLedger)
    assert runtime.authority_name == "AtlahsWqeLedger"


def test_queue_visit_uses_eligibility_for_wait_and_keeps_visibility_separate():
    visit = QueueVisit(
        "exec",
        "op",
        ResourceRef(ResourceKind.COPY_ENGINE, "ce0"),
        submitted_at_ps=0,
        eligible_at_ps=100,
        started_at_ps=120,
        finished_at_ps=170,
        completed_at_ps=180,
    )
    assert visit.queue_wait_ps == 20
    assert visit.service_ps == 50
    assert visit.visibility_ps == 10
    with pytest.raises(ValueError, match="before submission"):
        replace(visit, submitted_at_ps=101)
    with pytest.raises(ValueError, match="before eligibility"):
        replace(visit, started_at_ps=99)


@pytest.mark.parametrize(
    "compute_ps,dma_ps",
    [(10_000_000, 40_000_000), (80_000_000, 40_000_000)],
)
def test_dependency_changes_max_overlap_to_additive_exactly(compute_ps, dma_ps):
    profile = _profile()
    independent = CoarseDeviceRuntime(profile).execute(
        _overlap_graph(
            "independent",
            compute_ps=compute_ps,
            dma_ps=dma_ps,
            dependent=False,
        )
    )
    dependent = CoarseDeviceRuntime(profile).execute(
        _overlap_graph(
            "dependent",
            compute_ps=compute_ps,
            dma_ps=dma_ps,
            dependent=True,
        )
    )
    assert independent.completed_at_ps - 5000 == max(compute_ps, dma_ps)
    assert dependent.completed_at_ps - 5000 == compute_ps + dma_ps
    assert dependent.completed_at_ps - independent.completed_at_ps == min(
        compute_ps, dma_ps
    )


def test_shared_hbm_serializes_kernel_and_dma_without_changing_copy_service():
    profile = _profile()
    runtime = CoarseDeviceRuntime(profile)
    result = runtime.execute(
        _overlap_graph(
            "shared-hbm",
            compute_ps=10_000_000,
            dma_ps=40_000_000,
            dependent=False,
            compute_hbm_bytes=1,
        )
    )
    assert result.completed_at_ps - 5000 == 50_000_000
    report = runtime.last_report
    assert report is not None
    hbm_visits = [visit for visit in report.visits if visit.resource.kind is ResourceKind.HBM_QUEUE]
    assert len(hbm_visits) == 2
    assert hbm_visits[1].started_at_ps == hbm_visits[0].finished_at_ps


def test_directional_copy_engines_are_selected_before_fifo_queueing():
    profile = _profile(
        engines=(
            _engine("h2d", (CopyDirection.HOST_TO_DEVICE,)),
            _engine("d2h", (CopyDirection.DEVICE_TO_HOST,)),
        )
    )
    graph = ExecutionGraph(
        "directions",
        0,
        0,
        (
            ExecutionOperation(
                "in",
                0,
                "copy-in",
                DmaWork("in", "host", "gpu:0:hbm", 4),
            ),
            ExecutionOperation(
                "out",
                0,
                "copy-out",
                DmaWork("out", "gpu:0:hbm", "host", 4),
            ),
        ),
    )
    runtime = CoarseDeviceRuntime(profile)
    runtime.execute(graph)
    assert runtime.last_report is not None
    resources = {
        visit.operation_id: visit.resource.resource_id
        for visit in runtime.last_report.visits
        if visit.resource.kind is ResourceKind.COPY_ENGINE
    }
    assert resources == {
        "in": "node-0:gpu-0:copy:h2d",
        "out": "node-0:gpu-0:copy:d2h",
    }


def test_co_runnable_kernels_are_one_concurrent_compute_service_dispatch():
    class RecordingScheduler(SmSchedulerModel):
        def __init__(self, architecture):
            super().__init__(architecture)
            self.calls = []

        def estimate_concurrent(self, tasks):
            self.calls.append(tuple(task.task_id for task in tasks))
            return super().estimate_concurrent(tasks)

    scheduler = RecordingScheduler(_architecture())
    graph = ExecutionGraph(
        "concurrent-kernels",
        0,
        0,
        (
            ExecutionOperation("a", 0, "stream-a", ComputeWork("a")),
            ExecutionOperation("b", 0, "stream-b", ComputeWork("b")),
        ),
    )
    runtime = CoarseDeviceRuntime(
        kernel_services={0: scheduler},
        kernel_launches={"a": _launch("a"), "b": _launch("b")},
    )
    runtime.execute(graph)
    assert scheduler.calls == [("a", "b")]


def _control_graph(execution_id: str, active_gpus: int, mode=ControlMode.SYNCHRONOUS):
    operations = []
    for rank in range(active_gpus):
        for sequence in range(2):
            operations.append(
                ExecutionOperation(
                    f"control-{rank}-{sequence}",
                    rank,
                    f"control-{sequence}",
                    ControlWork(
                        "rail",
                        destination_ranks=(rank + 8,),
                        payload_bytes=1_048_576,
                        mode=mode,
                    ),
                )
            )
    return ExecutionGraph(execution_id, 0, 5000, tuple(operations))


@pytest.mark.parametrize("active_gpus", [1, 8])
@pytest.mark.parametrize("rate_gbps,expected_ps", [(200, 83_886_080), (400, 41_943_040)])
def test_gpu_affine_rails_have_rate_scaling_and_per_qp_fifo(
    active_gpus, rate_gbps, expected_ps
):
    runtime = CoarseDeviceRuntime(_profile(rate_gbps=rate_gbps))
    result = runtime.execute(_control_graph(f"rails-{active_gpus}-{rate_gbps}", active_gpus))
    assert result.completed_at_ps - 5000 == expected_ps
    assert runtime.last_report is not None
    assert len(runtime.last_report.wqes) == 2 * active_gpus
    for rank in range(active_gpus):
        records = [wqe for wqe in runtime.last_report.wqes if wqe.source_rank == rank]
        assert [record.sq_post_sequence for record in records] == [1, 2]
        assert records[0].finished_at_ps == records[1].started_at_ps
        assert {record.rnic_id for record in records} == {f"node-0:rnic-{rank}"}


def test_async_control_changes_logical_boundary_but_not_physical_quiescence():
    graph = _control_graph("async-control", 1, ControlMode.ASYNCHRONOUS)
    anchor = ExecutionOperation(
        "anchor",
        1,
        "compute",
        ComputeWork("anchor", nominal_duration_ps=10_000_000),
    )
    graph = replace(graph, operations=(*graph.operations, anchor))
    runtime = CoarseDeviceRuntime(_profile())
    result = runtime.execute(graph)
    assert result.completed_at_ps - 5000 == 10_000_000
    assert result.quiesced_at_ps is not None
    assert result.quiesced_at_ps - 5000 == 41_943_040
    assert runtime.last_report is not None
    control_records = [
        record for record in runtime.last_report.operations if record.operation_id.startswith("control")
    ]
    assert all(record.completed_at_ps == 5000 for record in control_records)
    assert all(record.physical_completed_at_ps > record.completed_at_ps for record in control_records)


def test_tail_accounting_separates_visit_sum_from_critical_path():
    runtime = CoarseDeviceRuntime(_profile())
    result = runtime.execute(_control_graph("tail", 8))
    report = runtime.last_report
    assert report is not None
    assert result.completed_at_ps - 5000 == 41_943_040
    assert report.sum_visit_wait_ps == 167_772_160
    assert report.sum_visit_wait_ps == 4 * (result.completed_at_ps - 5000)
    assert report.critical_path_queue_ps == 20_971_520
    for operation in report.operations:
        breakdown = operation.breakdown
        assert (
            breakdown.launch_queue_ps
            + breakdown.device_queue_ps
            + breakdown.service_ps
            + breakdown.completion_delivery_ps
            + breakdown.external_dependency_ps
            == breakdown.operation_latency_ps
        )


@pytest.mark.parametrize(
    "launch_service_ps,expected_jct_ps,expected_queue_ps",
    [(10, 150, 10), (20, 180, 20)],
)
def test_critical_path_clips_launch_work_before_predecessor_completion(
    launch_service_ps,
    expected_jct_ps,
    expected_queue_ps,
):
    profile = replace(_profile(), launch_service_ps=launch_service_ps)
    graph = ExecutionGraph(
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
    runtime = CoarseDeviceRuntime(profile)
    result = runtime.execute(graph)
    report = runtime.last_report
    assert report is not None
    assert result.completed_at_ps == expected_jct_ps
    assert report.realized_critical_path_operation_ids == ("a", "b")
    assert report.critical_path_queue_ps == expected_queue_ps
    by_id = {record.operation_id: record for record in report.operations}
    assert sum(
        by_id[operation_id].breakdown.operation_latency_ps
        for operation_id in report.realized_critical_path_operation_ids
    ) == expected_jct_ps


def test_collective_goal_tags_and_bookkeeping_preserve_operation_identity():
    graph = ExecutionGraph(
        "collective-identity",
        4,
        5000,
        (
            ExecutionOperation(
                "ring-op",
                0,
                "nccl-ring",
                CollectiveWork("all-reduce", (0, 8), 1024, "ring", "tp"),
            ),
            ExecutionOperation(
                "pairwise-op",
                0,
                "nccl-pairwise",
                CollectiveWork("all-to-allv", (0, 8), 256, "pairwise", "ep"),
            ),
        ),
    )
    assert collective_goal_tags(graph) == {
        "ring-op": (1000, 1001),
        "pairwise-op": (1002,),
    }
    rendered = render_serial_execution_graph_goal(graph, num_goal_ranks=9).render()
    rendered_tags = sorted(
        int(line.split(" tag ", 1)[1])
        for line in rendered.splitlines()
        if ": send " in line
    )
    bookkeeper = RequestBookkeeper()
    runtime = CoarseDeviceRuntime(_profile())
    result = runtime.execute(graph, bookkeeping=bookkeeper)
    assert runtime.last_report is not None
    assert sorted(wqe.goal_tag for wqe in runtime.last_report.wqes) == rendered_tags
    validate_bookkeeping_ledger(bookkeeper.snapshot())
    wqe_objects = [
        entry.fact
        for entry in bookkeeper.snapshot().entries
        if isinstance(entry.fact, CreatedObjectRecord)
        and entry.fact.ref.kind is CreatedObjectKind.NETWORK_WQE
    ]
    assert len(wqe_objects) == len(runtime.last_report.wqes)
    assert {dict(record.metadata)["graph_operation_id"] for record in wqe_objects} == {
        "ring-op",
        "pairwise-op",
    }
    ledger_events = [
        entry.fact
        for entry in bookkeeper.snapshot().entries
        if isinstance(entry.fact, CompletionEvent)
    ]
    assert len(ledger_events) == len(result.events)
    assert all(left is right for left, right in zip(ledger_events, result.events, strict=True))


def test_intra_node_collective_uses_nvlink_and_creates_no_wqe():
    graph = ExecutionGraph(
        "intra-node",
        0,
        0,
        (
            ExecutionOperation(
                "ring",
                0,
                "nccl",
                CollectiveWork("all-reduce", (0, 1), 1024, "ring"),
            ),
        ),
    )
    runtime = CoarseDeviceRuntime(_profile())
    runtime.execute(graph)
    assert runtime.last_report is not None
    assert runtime.last_report.wqes == ()
    assert any(
        visit.resource.kind is ResourceKind.NVLINK for visit in runtime.last_report.visits
    )


def test_participant_local_collective_arrivals_remain_per_rank():
    graph = ExecutionGraph(
        "local-arrivals",
        0,
        0,
        (
            ExecutionOperation(
                "fast",
                0,
                "compute-0",
                ComputeWork("fast", nominal_duration_ps=10),
            ),
            ExecutionOperation(
                "slow",
                8,
                "compute-8",
                ComputeWork("slow", nominal_duration_ps=40),
            ),
            ExecutionOperation(
                "ring",
                0,
                "nccl",
                CollectiveWork("all-reduce", (0, 8), 100, "ring"),
                participant_local_depends_on=("fast", "slow"),
            ),
        ),
    )
    runtime = CoarseDeviceRuntime(_profile())
    runtime.execute(graph)
    assert runtime.last_report is not None
    first_by_source = {}
    for wqe in runtime.last_report.wqes:
        first_by_source.setdefault(wqe.source_rank, wqe.submitted_at_ps)
    assert first_by_source == {0: 10, 8: 40}


def test_async_control_publishes_destination_arrival_for_local_dependency():
    graph = ExecutionGraph(
        "control-destination-arrival",
        0,
        0,
        (
            ExecutionOperation(
                "control",
                0,
                "control",
                ControlWork(
                    "arrival",
                    destination_ranks=(8,),
                    payload_bytes=1,
                    mode=ControlMode.ASYNCHRONOUS,
                ),
            ),
            ExecutionOperation(
                "destination",
                8,
                "compute",
                ComputeWork("destination", nominal_duration_ps=1),
                participant_local_depends_on=("control",),
            ),
        ),
    )
    runtime = CoarseDeviceRuntime(_profile())
    runtime.execute(graph)
    assert runtime.last_report is not None
    by_id = {record.operation_id: record for record in runtime.last_report.operations}
    destination_arrival = dict(by_id["control"].participant_completed_at_ps)[8]
    assert by_id["control"].completed_at_ps == 0
    assert destination_arrival > by_id["control"].completed_at_ps
    assert by_id["destination"].eligible_at_ps == destination_arrival


def test_peer_dma_publishes_destination_arrival_for_local_dependency():
    graph = ExecutionGraph(
        "peer-dma-destination-arrival",
        0,
        0,
        (
            ExecutionOperation(
                "peer",
                0,
                "copy",
                DmaWork("peer", "gpu:0:hbm", "gpu:1:hbm", 4),
            ),
            ExecutionOperation(
                "destination",
                1,
                "compute",
                ComputeWork("destination", nominal_duration_ps=1),
                participant_local_depends_on=("peer",),
            ),
        ),
    )
    runtime = CoarseDeviceRuntime(_profile())
    runtime.execute(graph)
    assert runtime.last_report is not None
    by_id = {record.operation_id: record for record in runtime.last_report.operations}
    arrivals = dict(by_id["peer"].participant_completed_at_ps)
    assert arrivals[0] == by_id["peer"].completed_at_ps
    assert arrivals[1] == by_id["peer"].completed_at_ps
    assert by_id["destination"].eligible_at_ps == arrivals[1]


class _FakeNativeSession:
    authority_name = "fake-native-session"

    def __init__(self):
        self.submissions = []
        self.commit_count = 0
        self.abort_count = 0
        self.fail_second_projection = False
        self.random_draw_count = 7

    def begin_transaction(self):
        return _FakeNativeTransaction(self)


class _FakeNativeTransaction:
    authority_name = _FakeNativeSession.authority_name

    def __init__(self, session: _FakeNativeSession):
        self.session = session
        self.pending = []
        self.prepared = False
        self.closed = False

    @property
    def random_draw_count(self):
        return self.session.random_draw_count

    def submit(self, submission: SemanticWqeSubmission) -> WqeLifecycleProjection:
        self.pending.append(submission)
        sequence = len(self.session.submissions) + len(self.pending)
        operation_id = submission.operation_id
        if self.session.fail_second_projection and len(self.pending) == 2:
            operation_id = "invalid-second-projection"
        return WqeLifecycleProjection(
            authority=self.authority_name,
            execution_id=submission.execution_id,
            operation_id=operation_id,
            wqe_id=f"native-wqe:{sequence}",
            native_wqe_id=f"native:{sequence}",
            sq_id=f"native:sq:{submission.source_rank}",
            rq_id=f"native:rq:{submission.destination_rank}",
            cq_id=f"native:cq:{submission.source_rank}",
            qp_id=f"native:qp:{submission.source_rank}",
            rnic_id=f"native:rnic:{submission.source_rank % 8}",
            source_rank=submission.source_rank,
            destination_rank=submission.destination_rank,
            payload_bytes=submission.payload_bytes,
            goal_tag=submission.goal_tag,
            extent_index=submission.extent_index,
            sq_post_sequence=sequence,
            cq_post_sequence=sequence,
            submitted_at_ps=submission.submitted_at_ps,
            eligible_at_ps=submission.eligible_at_ps,
            started_at_ps=submission.eligible_at_ps,
            finished_at_ps=submission.eligible_at_ps + 123,
            completed_at_ps=submission.eligible_at_ps + 123,
            channel_id=submission.channel_id,
            nccl_command_id=submission.nccl_command_id,
        )

    def prepare(self):
        assert not self.closed
        self.prepared = True

    def commit(self):
        assert self.prepared
        assert not self.closed
        self.session.submissions.extend(self.pending)
        self.session.commit_count += 1
        self.closed = True

    def abort(self):
        assert not self.closed
        self.pending.clear()
        self.session.abort_count += 1
        self.closed = True


def test_structural_mode_delegates_without_constructing_bypass_authority():
    session = _FakeNativeSession()
    runtime = CoarseDeviceRuntime(
        _profile(),
        authority_mode=RnicAuthorityMode.STRUCTURAL,
        native_session=session,
    )
    result = runtime.execute(_control_graph("native", 1))
    assert runtime.bypass_ledger is None
    assert runtime.authority_name == session.authority_name
    assert len(session.submissions) == 2
    assert session.commit_count == 1
    assert session.abort_count == 0
    assert result.completed_at_ps - 5000 == 123
    assert runtime.last_report is not None
    assert runtime.last_report.random_draw_count == 7


def test_structural_native_failure_aborts_without_double_advancing_on_retry():
    session = _FakeNativeSession()
    session.fail_second_projection = True
    runtime = CoarseDeviceRuntime(
        _profile(),
        authority_mode=RnicAuthorityMode.STRUCTURAL,
        native_session=session,
    )
    graph = _control_graph("native-rollback", 1)

    with pytest.raises(ValueError, match="operation_id"):
        runtime.execute(graph)

    assert session.submissions == []
    assert session.commit_count == 0
    assert session.abort_count == 1
    assert runtime.last_report is None

    session.fail_second_projection = False
    runtime.execute(graph)
    assert len(session.submissions) == 2
    assert session.commit_count == 1
    assert session.abort_count == 1
    assert runtime.last_report is not None
    assert [wqe.sq_post_sequence for wqe in runtime.last_report.wqes] == [1, 2]


def _canonical_report(runtime: CoarseDeviceRuntime):
    report = runtime.last_report
    assert report is not None
    operations = tuple(replace(record, class_label=0) for record in report.operations)
    return (
        operations,
        report.visits,
        report.wqes,
        report.sum_visit_wait_ps,
        report.critical_path_queue_ps,
        report.realized_critical_path_operation_ids,
        report.random_draw_count,
    )


def _identity_graph(execution_id: str, labels: tuple[int, int]) -> ExecutionGraph:
    graph = _control_graph(execution_id, 1)
    return replace(
        graph,
        operations=tuple(
            replace(operation, priority=label)
            for operation, label in zip(graph.operations, labels, strict=True)
        ),
    )


def test_omitted_and_explicit_identity_ignore_class_label_permutation_exactly():
    outcomes = []
    for policy in (None, IdentityArbitrationPolicy()):
        for labels in ((3, 9), (9, 3)):
            runtime = CoarseDeviceRuntime(_profile(), arbitration_policy=policy)
            result = runtime.execute(_identity_graph("identity", labels))
            outcomes.append((result, _canonical_report(runtime)))
    assert all(outcome == outcomes[0] for outcome in outcomes[1:])


class _LastReadyPolicy:
    def __init__(self):
        self.calls = []

    def select(self, candidates):
        self.calls.append(tuple(candidate.operation_id for candidate in candidates))
        return max(candidates, key=lambda candidate: candidate.baseline_sequence)


def test_conforming_arbitration_policy_is_accepted_and_selects_from_legal_ready_set():
    policy = _LastReadyPolicy()
    runtime = CoarseDeviceRuntime(_profile(), arbitration_policy=policy)
    runtime.execute(_identity_graph("replaceable-policy", (3, 9)))
    assert policy.calls[0] == ("control-0-0", "control-0-1")
    assert runtime.last_report is not None
    assert [wqe.operation_id for wqe in runtime.last_report.wqes] == [
        "control-0-1",
        "control-0-0",
    ]


@pytest.mark.parametrize("action", [KvCacheAction.READ, KvCacheAction.WRITE])
def test_byte_carrying_kv_read_write_fails_closed_before_state_mutation(action):
    runtime = CoarseDeviceRuntime(_profile())
    graph = ExecutionGraph(
        f"kv-{action.value}",
        0,
        0,
        (
            ExecutionOperation(
                "kv",
                0,
                "kv",
                KvCacheWork(action, "pool", byte_count=1),
            ),
        ),
    )
    with pytest.raises(ValueError, match="CORE-3"):
        runtime.execute(graph)
    assert runtime.last_report is None
    assert runtime.bypass_ledger is not None
    assert runtime.bypass_ledger.records == ()


def test_zero_byte_kv_read_remains_a_timing_neutral_lifecycle_observation():
    graph = ExecutionGraph(
        "kv-zero-byte",
        0,
        7,
        (
            ExecutionOperation(
                "kv",
                0,
                "kv",
                KvCacheWork(KvCacheAction.READ, "pool", byte_count=0),
            ),
        ),
    )
    runtime = CoarseDeviceRuntime(_profile())
    result = runtime.execute(graph)
    assert result.completed_at_ps == 7
    assert runtime.last_report is not None
    assert runtime.last_report.wqes == ()


@pytest.mark.parametrize("payload_bytes", [0, 1, 3])
def test_ring_payload_without_exact_positive_chunks_fails_closed(payload_bytes):
    runtime = CoarseDeviceRuntime(_profile())
    graph = ExecutionGraph(
        f"ring-payload-{payload_bytes}",
        0,
        0,
        (
            ExecutionOperation(
                "ring",
                0,
                "nccl",
                CollectiveWork("all-reduce", (0, 8), payload_bytes, "ring"),
            ),
        ),
    )
    with pytest.raises(ValueError, match="CORE-16"):
        runtime.execute(graph)
    assert runtime.last_report is None
    assert runtime.bypass_ledger is not None
    assert runtime.bypass_ledger.records == ()


def test_control_destination_count_cannot_overflow_reserved_goal_tag_block():
    runtime = CoarseDeviceRuntime(_profile())
    graph = ExecutionGraph(
        "control-tag-overflow",
        0,
        0,
        (
            ExecutionOperation(
                "control",
                0,
                "control",
                ControlWork("overflow", tuple(range(1, 1026)), 1),
            ),
        ),
    )
    with pytest.raises(ValueError, match="at most 1024 destinations"):
        runtime.execute(graph)
    assert runtime.last_report is None


def test_failed_preflight_does_not_mutate_runtime_or_bookkeeper():
    runtime = CoarseDeviceRuntime(_profile())
    bookkeeper = RequestBookkeeper()
    invalid = ExecutionGraph(
        "bad-dma",
        0,
        0,
        (
            ExecutionOperation(
                "dma",
                0,
                "copy",
                DmaWork("dma", "mystery", "gpu:0", 1),
            ),
        ),
    )
    with pytest.raises(ValueError, match="DMA endpoint"):
        runtime.execute(invalid, bookkeeping=bookkeeper)
    assert runtime.last_report is None
    assert runtime.bypass_ledger is not None
    assert runtime.bypass_ledger.records == ()
    assert bookkeeper.snapshot().entries == ()


def test_completion_queue_events_project_eligibility_and_grant_timestamps():
    runtime = CoarseDeviceRuntime(_profile())
    result = runtime.execute(_control_graph("event-contract", 1))
    assert runtime.last_report is not None
    second = runtime.last_report.wqes[1]
    events = [event for event in result.events if event.subject_object_id == second.wqe_id]
    by_phase = {event.phase: event for event in events}
    assert by_phase[EventPhase.QUEUED].timestamp_ps == second.eligible_at_ps
    assert by_phase[EventPhase.STARTED].timestamp_ps == second.started_at_ps
    assert by_phase[EventPhase.COMPLETED].resource == ResourceRef(
        ResourceKind.COMPLETION_QUEUE,
        second.cq_id,
    )
