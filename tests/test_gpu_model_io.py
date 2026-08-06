import json
import statistics
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
    PipelineKind,
    PipelineProfile,
    SassInstruction,
    SassWarpTrace,
    SmSchedulerModel,
    WarpSchedulerPolicy,
    a100_sxm_80gb_seed_profile,
    h100_sxm_80gb_seed_profile,
)
from simllm.compute.gpu_model_io import (
    CalibrationSplit,
    GpuCaptureEnvironment,
    GpuCopyReplayRecord,
    GpuKernelCatalogRecord,
    GpuMeasurementRecord,
    GpuModelArtifact,
    GpuReplayRecord,
    gpu_model_artifact_from_json,
    gpu_model_artifact_to_profile_table,
    load_gpu_model_artifact,
)
from simllm.compute.provider import GpuSpec, KernelSpec

FUNCTION_HASH = "1" * 64
TRACE_HASH = "2" * 64
BINARY_HASH = "3" * 64
CAPTURE_HASH = "4" * 64


def architecture() -> GpuArchitectureProfile:
    provenance = GpuModelProvenance(
        source="synthetic calibration",
        version="1",
        gpu="artifact-gpu",
        created="2026-08-06",
    )
    calibration = GpuCalibrationProfile(
        calibration_id="artifact-calibration-v1",
        target_architecture_profile_id="artifact-profile-v1",
        provenance=provenance,
        core_clock_hz=1_000_000_000,
        target_memory_clock_hz=1_200_000_000,
        pipelines=(
            PipelineProfile(
                kind=PipelineKind.ALU,
                opcodes=("ALU",),
                latency_cycles=4,
                issue_width_per_sm=2,
            ),
        ),
        memory=MemoryHierarchyProfile(
            hbm_latency_cycles=100,
            hbm_bandwidth_bytes_per_cycle=64,
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
                        bandwidth_bytes_per_cycle=64,
                    ),
                ),
            ),
        ),
        warp_scheduler_policy=WarpSchedulerPolicy.LOOSE_ROUND_ROBIN,
        relative_uncertainty=0.1,
    )
    return GpuArchitectureProfile(
        profile_id="artifact-profile-v1",
        gpu_name="artifact-gpu",
        aliases=("artifact-alias",),
        sm_count=1,
        warp_size=32,
        scheduler_count_per_sm=2,
        dispatch_width_per_scheduler=1,
        max_blocks_per_sm=16,
        max_warps_per_sm=64,
        max_threads_per_sm=2_048,
        max_threads_per_block=1_024,
        registers_per_sm=65_536,
        max_registers_per_thread=255,
        register_allocation_granularity_per_warp=256,
        shared_memory_per_sm=64 * 1_024,
        max_static_shared_memory_per_block=48 * 1_024,
        max_shared_memory_per_block=64 * 1_024,
        shared_memory_allocation_granularity=256,
        calibration=calibration,
    )


def kernel_launch(*, trace_hash: str = TRACE_HASH, repeat: int = 2) -> KernelLaunch:
    trace = SassWarpTrace(
        warp_id=0,
        instructions=(
            SassInstruction(
                opcode="ALU",
                pipeline=PipelineKind.ALU,
                repeat=repeat,
                dependent=True,
            ),
        ),
    )
    return KernelLaunch(
        implementation_id=FUNCTION_HASH,
        trace_id=trace_hash,
        grid_blocks=1,
        threads_per_block=32,
        registers_per_thread=32,
        static_shared_memory_bytes=0,
        dynamic_shared_memory_bytes=0,
        cta_traces=(CtaTrace(trace_class_id="uniform", block_ids=(0,), warp_traces=(trace,)),),
    )


def capture() -> GpuCaptureEnvironment:
    return GpuCaptureEnvironment(
        capture_id="capture-0",
        model_id="fixture-model@revision",
        gpu_name="artifact-gpu",
        gpu_uuid="GPU-fixture",
        framework="fixture-runtime",
        framework_version="abc123",
        driver_version="fixture-driver",
        cuda_version="fixture-cuda",
        execution_mode="eager",
        clock_policy="locked fixture clocks",
        observed_core_clock_hz=1_000_000_000,
        observed_memory_clock_hz=1_200_000_000,
        warmup_iterations=5,
        measured_iterations=3,
        tools=(("tracer", "1"), ("profiler", "2")),
        libraries=(("collective", "3"),),
        created="2026-08-06",
    )


def measurement(*, replay_id: str, duration_ps: int, measurement_id: str) -> GpuMeasurementRecord:
    samples = (duration_ps - 100, duration_ps, duration_ps + 100)
    mean = statistics.fmean(samples)
    return GpuMeasurementRecord(
        measurement_id=measurement_id,
        replay_id=replay_id,
        capture_id="capture-0",
        capture_sha256=CAPTURE_HASH,
        duration_samples_ps=samples,
        mean_duration_ps=mean,
        p50_duration_ps=float(statistics.median(samples)),
        coefficient_of_variation=statistics.pstdev(samples) / mean,
        calibration_split=CalibrationSplit.HELD_OUT,
        residual_fraction=(duration_ps - mean) / mean,
    )


def complete_artifact() -> GpuModelArtifact:
    arch = architecture()
    launch = kernel_launch()
    kernel_estimate = SmSchedulerModel(arch).estimate(launch)
    catalog = GpuKernelCatalogRecord(
        kernel="captured-kernel",
        config=(("tokens", 1),),
        gpu=arch.gpu_name,
        binary_sha256=BINARY_HASH,
        function_sha256=FUNCTION_HASH,
        trace_sha256=TRACE_HASH,
        launch=launch,
        capture_id="capture-0",
        launch_order=0,
        stream_id="stream-7",
        attributes=(("dtype", "bf16"), ("cache_protocol", "cold-l2")),
    )
    kernel_replay = GpuReplayRecord(
        replay_id="kernel-replay-0",
        implementation_id=FUNCTION_HASH,
        architecture_profile_id=arch.profile_id,
        calibration_id=arch.calibration.calibration_id,
        calibration_split=CalibrationSplit.HELD_OUT,
        relative_uncertainty=arch.calibration.relative_uncertainty,
        estimate=kernel_estimate,
    )
    transfer = CopyTransfer(
        transfer_id="copy-transfer-0",
        direction=CopyDirection.HOST_TO_DEVICE,
        bytes=4_096,
        source="host:pinned",
        destination="gpu:0:hbm",
    )
    copy_estimate = CopyEngineServiceModel(arch, "ce0").estimate(transfer)
    copy_replay = GpuCopyReplayRecord(
        replay_id="copy-replay-0",
        architecture_profile_id=arch.profile_id,
        calibration_id=arch.calibration.calibration_id,
        calibration_split=CalibrationSplit.HELD_OUT,
        relative_uncertainty=arch.calibration.relative_uncertainty,
        transfer=transfer,
        estimate=copy_estimate,
        capture_id="capture-0",
        launch_order=1,
        stream_id="stream-7",
        attributes=(("memory", "pinned"),),
    )
    return GpuModelArtifact(
        artifact_id="artifact-0",
        architecture=arch,
        captures=(capture(),),
        catalog=(catalog,),
        replays=(kernel_replay,),
        copy_replays=(copy_replay,),
        measurements=(
            measurement(
                replay_id=kernel_replay.replay_id,
                duration_ps=kernel_estimate.duration_ps,
                measurement_id="measurement-kernel",
            ),
            measurement(
                replay_id=copy_replay.replay_id,
                duration_ps=copy_estimate.duration_ps,
                measurement_id="measurement-copy",
            ),
        ),
    )


def test_complete_artifact_round_trips_kernel_copy_capture_and_measurements(tmp_path):
    artifact = complete_artifact()

    path = artifact.save(tmp_path / "gpu-model.json")

    loaded = load_gpu_model_artifact(path)

    assert loaded == artifact
    assert loaded.captures[0].observed_core_clock_hz == 1_000_000_000
    assert loaded.captures[0].observed_memory_clock_hz == 1_200_000_000
    assert json.loads(path.read_text())["schema"] == "simllm-gpu-model-artifact-v1"


def test_artifact_records_freeze_caller_owned_sequences():
    captures = [capture()]
    attributes = [["dtype", "bf16"]]
    catalog = replace(complete_artifact().catalog[0], attributes=attributes)
    artifact = GpuModelArtifact(
        artifact_id="immutable",
        architecture=architecture(),
        captures=captures,
        catalog=[catalog],
        replays=[],
        copy_replays=[],
        measurements=[],
    )

    captures.clear()
    attributes[0][1] = "fp32"

    assert artifact.captures == (capture(),)
    assert artifact.catalog[0].attributes == (("dtype", "bf16"),)


def test_artifact_compiles_to_constant_time_online_profile_table():
    artifact = complete_artifact()

    provider = gpu_model_artifact_to_profile_table(artifact)
    estimate = provider.estimate(
        KernelSpec(
            name="captured-kernel",
            flops=0,
            bytes_moved=0,
            config=(("tokens", 1),),
        ),
        GpuSpec(name="artifact-gpu", peak_flops=1, mem_bandwidth=1),
    )

    assert estimate.duration_ps == artifact.replays[0].estimate.duration_ps
    assert estimate.uncertainty == artifact.replays[0].relative_uncertainty
    assert provider.provenance.source == "gpu-model-artifact:artifact-0"


@pytest.mark.parametrize(
    "seed",
    (a100_sxm_80gb_seed_profile, h100_sxm_80gb_seed_profile),
)
def test_public_seed_profiles_round_trip_without_fabricated_captures(seed):
    profile = seed()
    artifact = GpuModelArtifact(
        artifact_id=f"{profile.profile_id}-artifact",
        architecture=profile,
        captures=(),
        catalog=(),
        replays=(),
        copy_replays=(),
    )

    assert GpuModelArtifact.from_json(artifact.to_json()) == artifact


def test_strict_loader_rejects_a_calibration_for_another_architecture():
    payload = complete_artifact().to_json()
    payload["architecture"]["calibration"]["target_architecture_profile_id"] = (
        "different-architecture"
    )

    with pytest.raises(ValueError, match="calibration target architecture profile"):
        gpu_model_artifact_from_json(payload)


def test_artifact_recomputes_measurement_statistics():
    payload = complete_artifact().to_json()
    payload["measurements"][0]["mean_duration_ps"] += 1

    with pytest.raises(ValueError, match="does not match duration samples"):
        gpu_model_artifact_from_json(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("observed_core_clock_hz", 900_000_000, "does not match calibration core clock"),
        (
            "observed_memory_clock_hz",
            1_100_000_000,
            "does not match calibration target memory clock",
        ),
    ),
)
def test_capture_clocks_must_match_the_calibration_target(field, value, message):
    artifact = complete_artifact()
    invalid_capture = replace(artifact.captures[0], **{field: value})

    with pytest.raises(ValueError, match=message):
        replace(artifact, captures=(invalid_capture,)).to_json()


def test_captured_artifact_requires_a_numeric_target_memory_clock():
    artifact = complete_artifact()
    calibration = replace(artifact.architecture.calibration, target_memory_clock_hz=None)
    architecture = replace(artifact.architecture, calibration=calibration)

    with pytest.raises(ValueError, match="captured artifacts require a numeric"):
        replace(artifact, architecture=architecture).to_json()


def test_artifact_recomputes_deterministic_kernel_replay():
    payload = complete_artifact().to_json()
    payload["replays"][0]["estimate"]["duration_cycles"] += 1
    payload["replays"][0]["estimate"]["duration_ps"] += 1_000

    with pytest.raises(ValueError, match="deterministic replay mismatch"):
        gpu_model_artifact_from_json(payload)


def test_artifact_recomputes_deterministic_copy_replay():
    payload = complete_artifact().to_json()
    payload["copy_replays"][0]["estimate"]["transfer_cycles"] += 1

    with pytest.raises(ValueError, match="deterministic copy replay mismatch"):
        gpu_model_artifact_from_json(payload)


def test_one_implementation_can_have_distinct_shape_traces():
    artifact = complete_artifact()
    second_trace_hash = "5" * 64
    second_launch = kernel_launch(trace_hash=second_trace_hash, repeat=3)
    second_catalog = replace(
        artifact.catalog[0],
        config=(("tokens", 2),),
        trace_sha256=second_trace_hash,
        launch=second_launch,
        launch_order=2,
    )
    second_replay = replace(
        artifact.replays[0],
        replay_id="kernel-replay-1",
        estimate=SmSchedulerModel(artifact.architecture).estimate(second_launch),
    )
    expanded = replace(
        artifact,
        catalog=(*artifact.catalog, second_catalog),
        replays=(*artifact.replays, second_replay),
    )

    assert GpuModelArtifact.from_json(expanded.to_json()) == expanded


def test_same_trace_cannot_cross_calibration_splits():
    artifact = complete_artifact()
    duplicate = replace(
        artifact.replays[0], replay_id="leak", calibration_split=CalibrationSplit.TRAIN
    )
    invalid = replace(artifact, replays=(*artifact.replays, duplicate))

    with pytest.raises(ValueError, match="kernel replay identities"):
        invalid.to_json()


def test_same_copy_service_cannot_cross_calibration_splits():
    artifact = complete_artifact()
    original = artifact.copy_replays[0]
    duplicate = replace(
        original,
        replay_id="copy-leak",
        calibration_split=CalibrationSplit.TRAIN,
        launch_order=2,
        transfer=replace(original.transfer, transfer_id="copy-transfer-leak"),
        estimate=replace(original.estimate, transfer_id="copy-transfer-leak"),
    )

    with pytest.raises(ValueError, match="copy service identities"):
        replace(artifact, copy_replays=(original, duplicate)).to_json()


def test_copy_attribute_order_cannot_bypass_split_identity():
    artifact = complete_artifact()
    original = replace(
        artifact.copy_replays[0],
        attributes=(("api", "cudaMemcpyAsync"), ("memory", "pinned")),
    )
    duplicate = replace(
        original,
        replay_id="copy-order-leak",
        calibration_split=CalibrationSplit.TRAIN,
        launch_order=2,
        attributes=(("memory", "pinned"), ("api", "cudaMemcpyAsync")),
        transfer=replace(original.transfer, transfer_id="copy-transfer-order-leak"),
        estimate=replace(original.estimate, transfer_id="copy-transfer-order-leak"),
    )

    with pytest.raises(ValueError, match="copy service identities"):
        replace(artifact, copy_replays=(original, duplicate)).to_json()


def test_hash_prefix_cannot_bypass_trace_identity_uniqueness():
    artifact = complete_artifact()
    prefixed_launch = replace(
        artifact.catalog[0].launch,
        implementation_id=f"sha256:{FUNCTION_HASH}",
        trace_id=f"sha256:{TRACE_HASH}",
    )
    duplicate_catalog = replace(
        artifact.catalog[0],
        config=(("tokens", 2),),
        function_sha256=f"sha256:{FUNCTION_HASH}",
        trace_sha256=f"sha256:{TRACE_HASH}",
        launch=prefixed_launch,
        launch_order=2,
    )
    invalid = replace(artifact, catalog=(*artifact.catalog, duplicate_catalog))

    with pytest.raises(ValueError, match="trace IDs"):
        invalid.to_json()


def test_gpu_alias_cannot_duplicate_a_semantic_catalog_key():
    artifact = complete_artifact()
    duplicate_catalog = replace(
        artifact.catalog[0],
        gpu="artifact-alias",
        launch_order=2,
        trace_sha256="5" * 64,
        launch=replace(artifact.catalog[0].launch, trace_id="5" * 64),
    )
    invalid = replace(artifact, catalog=(*artifact.catalog, duplicate_catalog))

    with pytest.raises(ValueError, match="catalog keys"):
        invalid.to_json()


def test_capture_stream_order_is_unique_across_kernel_and_copy():
    artifact = complete_artifact()
    duplicate_order = replace(artifact.copy_replays[0], launch_order=0)

    with pytest.raises(ValueError, match="captured stream launch orders"):
        replace(artifact, copy_replays=(duplicate_order,)).to_json()


def test_capture_order_metadata_is_all_or_none():
    artifact = complete_artifact()
    orphan_order = replace(artifact.catalog[0], capture_id=None)

    with pytest.raises(ValueError, match="all set or all absent"):
        replace(artifact, catalog=(orphan_order,)).to_json()


def test_dynamic_dependency_indices_survive_repeated_instruction_round_trip():
    artifact = complete_artifact()
    trace = SassWarpTrace(
        warp_id=0,
        instructions=(
            SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU, repeat=2),
            SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU, dependency_indices=(1,)),
        ),
    )
    launch = replace(
        artifact.catalog[0].launch,
        cta_traces=(CtaTrace(trace_class_id="uniform", block_ids=(0,), warp_traces=(trace,)),),
    )
    catalog = replace(artifact.catalog[0], launch=launch)
    replay = replace(
        artifact.replays[0], estimate=SmSchedulerModel(artifact.architecture).estimate(launch)
    )
    updated = replace(artifact, catalog=(catalog,), replays=(replay,), measurements=())

    assert GpuModelArtifact.from_json(updated.to_json()) == updated


def test_loader_rejects_duplicate_json_object_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema": "a", "schema": "b"}\n')

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_gpu_model_artifact(path)
