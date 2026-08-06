import math
from dataclasses import replace

import pytest

from simllm.compute.gpu_model import (
    CopyDirection,
    CopyDirectionProfile,
    CopyEngineProfile,
    CopyEngineServiceModel,
    CopyTransfer,
    CtaTrace,
    GpuArchitectureProfile,
    GpuCalibrationProfile,
    GpuModelProvenance,
    KernelLaunch,
    MemoryHierarchyProfile,
    MemorySpace,
    PipelineKind,
    PipelineProfile,
    SassInstruction,
    SassWarpTrace,
    SmSchedulerModel,
    TraceCalibratedGpuProvider,
    WarpSchedulerPolicy,
    a100_sxm_80gb_seed_profile,
    h100_sxm_80gb_seed_profile,
)
from simllm.compute.provider import GpuSpec, KernelSpec


def synthetic_architecture(
    *,
    sm_count: int = 1,
    schedulers: int = 1,
    hbm_bytes_per_cycle: float = 64,
    shared_memory_per_sm: int = 65_536,
    register_granularity_per_warp: int = 1,
) -> GpuArchitectureProfile:
    provenance = GpuModelProvenance(
        source="synthetic fixture",
        version="1",
        gpu="synthetic",
        created="2026-08-06",
    )
    return GpuArchitectureProfile(
        profile_id="synthetic-profile",
        gpu_name="synthetic",
        aliases=("synthetic-alias",),
        sm_count=sm_count,
        warp_size=32,
        scheduler_count_per_sm=schedulers,
        dispatch_width_per_scheduler=1,
        max_blocks_per_sm=16,
        max_warps_per_sm=64,
        max_threads_per_sm=2_048,
        max_threads_per_block=1_024,
        registers_per_sm=65_536,
        max_registers_per_thread=255,
        register_allocation_granularity_per_warp=register_granularity_per_warp,
        shared_memory_per_sm=shared_memory_per_sm,
        max_static_shared_memory_per_block=min(shared_memory_per_sm, 48 * 1_024),
        max_shared_memory_per_block=shared_memory_per_sm,
        shared_memory_allocation_granularity=1,
        calibration=GpuCalibrationProfile(
            calibration_id="synthetic-calibration",
            target_architecture_profile_id="synthetic-profile",
            provenance=provenance,
            core_clock_hz=1_000_000_000,
            target_memory_clock_hz=1_200_000_000,
            pipelines=(
                PipelineProfile(
                    kind=PipelineKind.ALU,
                    opcodes=("ALU", "ALU_SLOW"),
                    latency_cycles=4,
                    issue_width_per_sm=4,
                    opcode_latencies=(("ALU_SLOW", 8),),
                ),
                PipelineProfile(
                    kind=PipelineKind.LOAD_STORE,
                    opcodes=("MEMORY",),
                    latency_cycles=1,
                    issue_width_per_sm=4,
                ),
            ),
            memory=MemoryHierarchyProfile(
                hbm_latency_cycles=100,
                hbm_bandwidth_bytes_per_cycle=hbm_bytes_per_cycle,
                l2_latency_cycles=20,
                l1_latency_cycles=10,
                shared_latency_cycles=5,
            ),
            copy_engines=(
                CopyEngineProfile(
                    engine_id="ce0",
                    clock_hz=1_000_000_000,
                    direction_profiles=(
                        CopyDirectionProfile(
                            direction=CopyDirection.HOST_TO_DEVICE,
                            setup_cycles=20,
                            bandwidth_bytes_per_cycle=hbm_bytes_per_cycle,
                        ),
                        CopyDirectionProfile(
                            direction=CopyDirection.DEVICE_TO_HOST,
                            setup_cycles=30,
                            bandwidth_bytes_per_cycle=hbm_bytes_per_cycle / 2,
                        ),
                    ),
                ),
            ),
            warp_scheduler_policy=WarpSchedulerPolicy.LOOSE_ROUND_ROBIN,
            relative_uncertainty=0.0,
        ),
    )


def launch(
    *,
    blocks: int,
    threads: int,
    traces: tuple[SassWarpTrace, ...],
    registers_per_thread: int = 0,
    shared_bytes: int = 0,
) -> KernelLaunch:
    return KernelLaunch(
        implementation_id="kernel-sha256:fixture",
        trace_id="trace-sha256:fixture",
        grid_blocks=blocks,
        threads_per_block=threads,
        registers_per_thread=registers_per_thread,
        static_shared_memory_bytes=shared_bytes,
        dynamic_shared_memory_bytes=0,
        cta_traces=(
            CtaTrace(
                trace_class_id="uniform-cta",
                block_ids=tuple(range(blocks)),
                warp_traces=traces,
            ),
        ),
    )


def traces_for_warps(warps: int, instruction: SassInstruction) -> tuple[SassWarpTrace, ...]:
    return tuple(SassWarpTrace(warp_id=warp, instructions=(instruction,)) for warp in range(warps))


@pytest.mark.parametrize(
    ("blocks", "sm_count", "expected_cycles"),
    ((4, 1, 1_024), (4, 4, 256), (9, 1, 2_304), (9, 4, 768)),
)
def test_partial_cta_waves_match_frozen_closed_form(blocks, sm_count, expected_cycles):
    architecture = synthetic_architecture(sm_count=sm_count, shared_memory_per_sm=1)
    instruction = SassInstruction(
        opcode="ALU",
        pipeline=PipelineKind.ALU,
        repeat=64,
        dependent=True,
    )
    kernel = launch(
        blocks=blocks,
        threads=32,
        shared_bytes=1,
        traces=traces_for_warps(1, instruction),
    )

    estimate = SmSchedulerModel(architecture).estimate(kernel)

    assert estimate.duration_cycles == expected_cycles
    assert estimate.duration_ps == expected_cycles * 1_000
    assert estimate.completed_blocks == blocks
    assert estimate.resident_blocks_per_sm == 1
    assert estimate.cta_waves == math.ceil(blocks / sm_count)


def test_distinct_cta_trace_classes_preserve_edge_block_work():
    architecture = synthetic_architecture()
    short = SassWarpTrace(
        warp_id=0,
        instructions=(SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU),),
    )
    long = SassWarpTrace(
        warp_id=0,
        instructions=(
            SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU, repeat=2, dependent=True),
        ),
    )
    kernel = KernelLaunch(
        implementation_id="kernel-sha256:fixture",
        trace_id="trace-sha256:fixture",
        grid_blocks=2,
        threads_per_block=32,
        registers_per_thread=0,
        static_shared_memory_bytes=0,
        dynamic_shared_memory_bytes=0,
        cta_traces=(
            CtaTrace(trace_class_id="interior", block_ids=(0,), warp_traces=(short,)),
            CtaTrace(trace_class_id="edge", block_ids=(1,), warp_traces=(long,)),
        ),
    )

    assert SmSchedulerModel(architecture).estimate(kernel).duration_cycles == 9


def test_cta_trace_classes_require_exact_block_coverage():
    trace = traces_for_warps(1, SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU))
    base = {
        "implementation_id": "kernel-sha256:fixture",
        "trace_id": "trace-sha256:fixture",
        "grid_blocks": 2,
        "threads_per_block": 32,
        "registers_per_thread": 0,
        "static_shared_memory_bytes": 0,
        "dynamic_shared_memory_bytes": 0,
    }

    with pytest.raises(ValueError, match="cover every linear block ID"):
        KernelLaunch(
            **base,
            cta_traces=(CtaTrace(trace_class_id="missing", block_ids=(0,), warp_traces=trace),),
        )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        KernelLaunch(
            **base,
            cta_traces=(
                CtaTrace(trace_class_id="a", block_ids=(0, 1), warp_traces=trace),
                CtaTrace(trace_class_id="b", block_ids=(1,), warp_traces=trace),
            ),
        )


@pytest.mark.parametrize(
    ("warps", "schedulers", "expected_cycles"),
    ((4, 1, 7), (4, 4, 4), (16, 1, 19), (16, 4, 7)),
)
def test_warp_scheduler_width_matches_frozen_closed_form(warps, schedulers, expected_cycles):
    architecture = synthetic_architecture(schedulers=schedulers)
    instruction = SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU)
    kernel = launch(
        blocks=1,
        threads=warps * architecture.warp_size,
        traces=traces_for_warps(warps, instruction),
    )

    estimate = SmSchedulerModel(architecture).estimate(kernel)

    assert estimate.duration_cycles == expected_cycles
    assert estimate.issued_instructions == warps


@pytest.mark.parametrize("schedulers", (1, 4))
def test_register_scoreboard_preserves_a_true_dependency_chain(schedulers):
    architecture = synthetic_architecture(schedulers=schedulers)
    instruction = SassInstruction(
        opcode="ALU",
        pipeline=PipelineKind.ALU,
        repeat=8,
        source_registers=("R0",),
        destination_registers=("R0",),
    )
    kernel = launch(blocks=1, threads=32, traces=traces_for_warps(1, instruction))

    estimate = SmSchedulerModel(architecture).estimate(kernel)

    assert estimate.duration_cycles == 32
    assert estimate.dependency_stall_cycles == 21
    assert estimate.completion_drain_cycles == 3
    assert estimate.sm_dependency_idle_cycles == (21,)
    assert estimate.sm_completion_drain_cycles == (3,)


@pytest.mark.parametrize(
    ("threads", "registers", "expected_resident"),
    ((128, 32, 16), (128, 128, 4), (256, 32, 8), (256, 128, 2)),
)
def test_occupancy_is_the_minimum_resource_limit(threads, registers, expected_resident):
    architecture = synthetic_architecture()
    warps = math.ceil(threads / architecture.warp_size)
    instruction = SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU)
    kernel = launch(
        blocks=1,
        threads=threads,
        registers_per_thread=registers,
        traces=traces_for_warps(warps, instruction),
    )

    assert SmSchedulerModel(architecture).resident_blocks_per_sm(kernel) == expected_resident


def test_register_allocation_granularity_is_applied_to_each_warp():
    architecture = synthetic_architecture(register_granularity_per_warp=256)
    instruction = SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU)
    kernel = launch(
        blocks=1,
        threads=128,
        registers_per_thread=33,
        traces=traces_for_warps(4, instruction),
    )

    # ceil(33 * 32 / 256) * 256 = 1,280 registers per warp. Four warps
    # consume 5,120 registers, so registers cap residency at floor(65,536 / 5,120).
    assert SmSchedulerModel(architecture).resident_blocks_per_sm(kernel) == 12


@pytest.mark.parametrize(
    ("byte_count", "bandwidth", "expected_cycles"),
    ((4_096, 32, 228), (4_096, 64, 164), (8_192, 32, 356), (8_192, 64, 228)),
)
def test_hbm_latency_and_serialization_match_frozen_closed_form(
    byte_count, bandwidth, expected_cycles
):
    architecture = synthetic_architecture(hbm_bytes_per_cycle=bandwidth)
    instruction = SassInstruction(
        opcode="MEMORY",
        pipeline=PipelineKind.LOAD_STORE,
        memory_space=MemorySpace.HBM,
        requested_bytes=byte_count,
        transacted_bytes=byte_count,
        dependent=True,
    )
    kernel = launch(blocks=1, threads=32, traces=traces_for_warps(1, instruction))

    estimate = SmSchedulerModel(architecture).estimate(kernel)

    assert estimate.duration_cycles == expected_cycles
    assert estimate.hbm_requested_bytes == byte_count
    assert estimate.hbm_transacted_bytes == byte_count
    assert estimate.hbm_serviced_bytes == byte_count
    assert estimate.hbm_request_instructions == 1


def test_hbm_bandwidth_serializes_bytes_but_not_return_latency():
    architecture = synthetic_architecture(schedulers=2, hbm_bytes_per_cycle=64)
    instruction = SassInstruction(
        opcode="MEMORY",
        pipeline=PipelineKind.LOAD_STORE,
        memory_space=MemorySpace.HBM,
        requested_bytes=64,
        transacted_bytes=64,
    )
    kernel = launch(blocks=1, threads=64, traces=traces_for_warps(2, instruction))

    estimate = SmSchedulerModel(architecture).estimate(kernel)

    assert estimate.duration_cycles == 102
    assert estimate.hbm_requested_bytes == 128
    assert estimate.hbm_transacted_bytes == 128


def test_loose_round_robin_exposes_memory_latency_before_a_long_ready_warp():
    architecture = synthetic_architecture(schedulers=1, hbm_bytes_per_cycle=64)
    alu_trace = SassWarpTrace(
        warp_id=0,
        instructions=(SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU, repeat=4),),
    )
    memory_trace = SassWarpTrace(
        warp_id=1,
        instructions=(
            SassInstruction(
                opcode="MEMORY",
                pipeline=PipelineKind.LOAD_STORE,
                memory_space=MemorySpace.HBM,
                requested_bytes=64,
                transacted_bytes=64,
            ),
        ),
    )

    estimate = SmSchedulerModel(architecture).estimate(
        launch(blocks=1, threads=64, traces=(alu_trace, memory_trace))
    )
    greedy_calibration = replace(
        architecture.calibration,
        warp_scheduler_policy=WarpSchedulerPolicy.GREEDY_THEN_OLDEST,
    )
    greedy = SmSchedulerModel(replace(architecture, calibration=greedy_calibration)).estimate(
        launch(blocks=1, threads=64, traces=(alu_trace, memory_trace))
    )

    assert estimate.duration_cycles == 102
    assert greedy.duration_cycles == 105


@pytest.mark.parametrize(
    ("byte_count", "bandwidth", "expected_cycles"),
    ((4_096, 32, 148), (4_096, 64, 84), (8_192, 32, 276), (8_192, 64, 148)),
)
def test_copy_service_matches_frozen_closed_form(byte_count, bandwidth, expected_cycles):
    architecture = synthetic_architecture(hbm_bytes_per_cycle=bandwidth)
    transfer = CopyTransfer(
        transfer_id="copy-0",
        direction=CopyDirection.HOST_TO_DEVICE,
        bytes=byte_count,
        source="host:pinned",
        destination="gpu:0:hbm",
    )

    estimate = CopyEngineServiceModel(architecture, "ce0").estimate(transfer)

    assert estimate.duration_cycles == expected_cycles
    assert estimate.duration_ps == expected_cycles * 1_000
    assert estimate.setup_cycles == 20
    assert estimate.effective_bandwidth_bytes_per_cycle <= bandwidth
    assert estimate.source == transfer.source
    assert estimate.destination == transfer.destination


def test_copy_direction_is_explicit_and_incompatible_engines_fail():
    architecture = synthetic_architecture()
    transfer = CopyTransfer(
        transfer_id="copy-0",
        direction=CopyDirection.PEER_TO_PEER,
        bytes=4_096,
        source="gpu:0:hbm",
        destination="gpu:1:hbm",
    )

    with pytest.raises(ValueError, match="does not support"):
        CopyEngineServiceModel(architecture, "ce0").estimate(transfer)


def test_copy_engine_uses_its_own_clock_domain():
    architecture = synthetic_architecture()
    engine = replace(architecture.copy_engines[0], clock_hz=500_000_000)
    calibration = replace(architecture.calibration, copy_engines=(engine,))
    architecture = replace(architecture, calibration=calibration)
    transfer = CopyTransfer(
        transfer_id="copy-0",
        direction=CopyDirection.HOST_TO_DEVICE,
        bytes=4_096,
        source="host:pinned",
        destination="gpu:0:hbm",
    )

    estimate = CopyEngineServiceModel(architecture, "ce0").estimate(transfer)

    assert estimate.duration_cycles == 84
    assert estimate.duration_ps == 168_000


def test_copy_engine_calibrates_each_direction_independently():
    architecture = synthetic_architecture()
    transfer = CopyTransfer(
        transfer_id="copy-0",
        direction=CopyDirection.DEVICE_TO_HOST,
        bytes=4_096,
        source="gpu:0:hbm",
        destination="host:pinned",
    )

    estimate = CopyEngineServiceModel(architecture, "ce0").estimate(transfer)

    assert estimate.duration_cycles == 158


def test_replay_rejects_unknown_barrier_and_impossible_launches():
    architecture = synthetic_architecture(shared_memory_per_sm=1_024)
    unknown = SassInstruction(opcode="UNKNOWN", pipeline=PipelineKind.ALU)
    barrier = SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU, barrier=True)

    with pytest.raises(ValueError, match="unknown normalized opcode"):
        SmSchedulerModel(architecture).estimate(
            launch(blocks=1, threads=32, traces=traces_for_warps(1, unknown))
        )
    with pytest.raises(ValueError, match="barrier instructions"):
        SmSchedulerModel(architecture).estimate(
            launch(blocks=1, threads=32, traces=traces_for_warps(1, barrier))
        )
    with pytest.raises(ValueError, match="above the profile limit"):
        SmSchedulerModel(architecture).estimate(
            launch(
                blocks=1,
                threads=32,
                shared_bytes=1_025,
                traces=traces_for_warps(
                    1, SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU)
                ),
            )
        )
    with pytest.raises(ValueError, match="threads per CTA"):
        SmSchedulerModel(architecture).estimate(
            launch(
                blocks=1,
                threads=1_056,
                traces=traces_for_warps(
                    33, SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU)
                ),
            )
        )
    with pytest.raises(ValueError, match="registers per thread"):
        SmSchedulerModel(architecture).estimate(
            launch(
                blocks=1,
                threads=32,
                registers_per_thread=256,
                traces=traces_for_warps(
                    1, SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU)
                ),
            )
        )
    with pytest.raises(ValueError, match="static shared-memory"):
        SmSchedulerModel(synthetic_architecture()).estimate(
            launch(
                blocks=1,
                threads=32,
                shared_bytes=48 * 1_024 + 1,
                traces=traces_for_warps(
                    1, SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU)
                ),
            )
        )


@pytest.mark.parametrize(
    ("record", "field"),
    (
        (SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU), "dependent"),
        (SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU), "barrier"),
    ),
)
def test_instruction_flags_require_actual_booleans(record, field):
    with pytest.raises(TypeError, match="must be a boolean"):
        replace(record, **{field: 1})


def test_cooperative_flag_requires_an_actual_boolean():
    kernel = launch(
        blocks=1,
        threads=32,
        traces=traces_for_warps(1, SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU)),
    )

    with pytest.raises(TypeError, match="must be a boolean"):
        replace(kernel, cooperative=1)


def test_register_scoreboard_preserves_write_after_write_order():
    architecture = synthetic_architecture(schedulers=2)
    trace = SassWarpTrace(
        warp_id=0,
        instructions=(
            SassInstruction(
                opcode="MEMORY",
                pipeline=PipelineKind.LOAD_STORE,
                memory_space=MemorySpace.HBM,
                requested_bytes=64,
                transacted_bytes=64,
                destination_registers=("R0",),
            ),
            SassInstruction(
                opcode="ALU",
                pipeline=PipelineKind.ALU,
                destination_registers=("R0",),
            ),
            SassInstruction(
                opcode="ALU",
                pipeline=PipelineKind.ALU,
                source_registers=("R0",),
            ),
        ),
    )

    estimate = SmSchedulerModel(architecture).estimate(
        launch(blocks=1, threads=32, traces=(trace,))
    )

    assert estimate.duration_cycles == 109


def test_opcode_latency_override_is_observable():
    architecture = synthetic_architecture()
    instruction = SassInstruction(opcode="ALU_SLOW", pipeline=PipelineKind.ALU, dependent=True)
    kernel = launch(blocks=1, threads=32, traces=traces_for_warps(1, instruction))

    assert SmSchedulerModel(architecture).estimate(kernel).duration_cycles == 8


def test_trace_provider_preserves_the_existing_compute_provider_boundary():
    architecture = synthetic_architecture()
    instruction = SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU, dependent=True)
    kernel_launch = launch(blocks=1, threads=32, traces=traces_for_warps(1, instruction))
    kernel = KernelSpec(name="captured-kernel", flops=0, bytes_moved=0, config=(("n", 1),))
    provider = TraceCalibratedGpuProvider(
        (architecture,),
        {(kernel.name, kernel.config, "synthetic-alias"): kernel_launch},
    )

    estimate = provider.estimate(kernel, GpuSpec("synthetic", 1, 1))
    replay = provider.replay(kernel, GpuSpec("synthetic", 1, 1))

    assert estimate.duration_ps == 4_000
    assert estimate.bound == "sass-replay"
    assert estimate.uncertainty == 0.0
    assert replay.architecture_profile_id == architecture.profile_id
    assert replay.calibration_id == architecture.calibration.calibration_id
    assert replay.implementation_id == kernel_launch.implementation_id
    assert provider.replay(kernel, GpuSpec("synthetic", 1, 1)) is replay
    with pytest.raises(KeyError, match="no exact trace launch"):
        provider.estimate(replace(kernel, config=(("n", 2),)), GpuSpec("synthetic", 1, 1))


def test_public_seed_profiles_keep_facts_separate_from_capture_defined_copy_engines():
    a100 = a100_sxm_80gb_seed_profile()
    h100 = h100_sxm_80gb_seed_profile()

    assert (a100.sm_count, h100.sm_count) == (108, 132)
    assert (a100.max_warps_per_sm, h100.max_warps_per_sm) == (64, 64)
    assert (a100.shared_memory_per_sm, h100.shared_memory_per_sm) == (
        164 * 1_024,
        228 * 1_024,
    )
    assert (a100.max_shared_memory_per_block, h100.max_shared_memory_per_block) == (
        163 * 1_024,
        227 * 1_024,
    )
    assert a100.clock_hz == 1_410_000_000
    assert h100.clock_hz == 1_980_000_000
    assert a100.calibration.relative_uncertainty == 0.5
    assert h100.calibration.relative_uncertainty == 0.5
    assert (a100.memory.l1_latency_cycles, h100.memory.l1_latency_cycles) == (33, 32)
    assert a100.copy_engines == h100.copy_engines == ()
    assert "a100" in a100.all_names
    assert "h100" in h100.all_names
    assert "2208.11174" in " ".join(a100.calibration.provenance.references)
    assert "2402.13499" in " ".join(h100.calibration.provenance.references)


def test_calibration_is_bound_to_its_target_architecture_profile():
    architecture = synthetic_architecture()

    with pytest.raises(ValueError, match="calibration target architecture profile"):
        replace(architecture, profile_id="different-structure")

    a100 = a100_sxm_80gb_seed_profile()
    h100 = h100_sxm_80gb_seed_profile()
    with pytest.raises(ValueError, match="calibration target architecture profile"):
        replace(h100, calibration=a100.calibration)


def test_hopper_warpgroup_matrix_instructions_fail_until_semantics_exist():
    architecture = h100_sxm_80gb_seed_profile()
    instruction = SassInstruction(opcode="WGMMA", pipeline=PipelineKind.TENSOR)
    kernel = launch(blocks=1, threads=32, traces=traces_for_warps(1, instruction))

    with pytest.raises(ValueError, match="unknown normalized opcode"):
        SmSchedulerModel(architecture).estimate(kernel)


def test_replay_is_bit_identical_including_component_counters():
    architecture = synthetic_architecture(sm_count=2, schedulers=4)
    instruction = SassInstruction(
        opcode="ALU",
        pipeline=PipelineKind.ALU,
        repeat=11,
        dependent=True,
    )
    kernel = launch(blocks=5, threads=32, traces=traces_for_warps(1, instruction))
    scheduler = SmSchedulerModel(architecture)

    assert scheduler.estimate(kernel) == scheduler.estimate(kernel)
