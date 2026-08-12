from __future__ import annotations

import csv

import pytest

from simllm.compute import (
    HELD_OUT_SPLIT,
    TRAIN_SPLIT,
    CapturePlanCell,
    ComputeCalibrationArtifact,
    ComputeCalibrationProvenance,
    GpuSpec,
    KernelCaptureCell,
    KernelLaunchMetadata,
    KernelSpec,
    ProfileTableProvenance,
    ProfileTableProvider,
    absolute_percentage_error,
    calibration_artifact_to_profile_table,
    nearest_rank,
    parse_nsight_cuda_gpu_trace_csv,
    physical_duration_bounds_ps,
)

GPU = GpuSpec("gtx1660-ti-sm75-fp32", 5.437e12, 288e9)


def _launch(grid_x: int = 1024) -> KernelLaunchMetadata:
    return KernelLaunchMetadata(
        grid=(grid_x, 1, 1),
        block=(256, 1, 1),
        registers_per_thread=20,
        static_shared_memory_bytes=0,
        dynamic_shared_memory_bytes=0,
        device="NVIDIA GeForce GTX 1660 Ti (0)",
        context_id=1,
        stream_id=7,
    )


def _provenance() -> ComputeCalibrationProvenance:
    return ComputeCalibrationProvenance(
        gpu_model="NVIDIA GeForce GTX 1660 Ti",
        gpu_uuid="GPU-test",
        compute_capability="7.5",
        driver_version="550.90.07",
        cuda_version="12.4.99",
        nsight_systems_version="2023.4.4.54",
        nsight_compute_version="2024.1.0.0",
        source_sha256="1" * 64,
        binary_sha256="2" * 64,
        static_sass_sha256="3" * 64,
        capture_sha256="4" * 64,
        creation_date="2026-08-12",
        warmup_policy="10 launches per cell",
        cache_policy="flush before each measured launch",
        clock_policy="unlocked",
        core_clock_before_mhz=300,
        core_clock_after_mhz=1500,
        memory_clock_before_mhz=405,
        memory_clock_after_mhz=6001,
        performance_counter_status="blocked: ERR_NVGPUCTRPERM",
        references=("https://docs.nvidia.com/nsight-systems/",),
    )


def _cell(shape: int, duration: int, split: str) -> KernelCaptureCell:
    return KernelCaptureCell(
        family="kv_read",
        dtype="fp32",
        gpu_profile=GPU.name,
        config=(("kv_tokens", shape),),
        split=split,
        kernel_name="void simllm_kv_read_kernel<float>(...)",
        work_items=shape * 262_144,
        source_flops=0,
        compulsory_input_bytes=shape * 262_144 * 4,
        total_bytes=shape * 262_144 * 8,
        launch=_launch(shape * 1024),
        durations_ps=(duration - 1_000, duration, duration + 1_000),
    )


def _artifact() -> ComputeCalibrationArtifact:
    return ComputeCalibrationArtifact(
        provenance=_provenance(),
        cells=(
            _cell(1, 100_000, TRAIN_SPLIT),
            _cell(4, 400_000, TRAIN_SPLIT),
            _cell(16, 1_600_000, TRAIN_SPLIT),
            _cell(2, 200_000, HELD_OUT_SPLIT),
            _cell(8, 800_000, HELD_OUT_SPLIT),
        ),
    )


def test_nsight_csv_parser_preserves_order_samples_and_launch(tmp_path):
    path = tmp_path / "trace.csv"
    headers = [
        "Start (ns)",
        "Duration (ns)",
        "CorrId",
        "GrdX",
        "GrdY",
        "GrdZ",
        "BlkX",
        "BlkY",
        "BlkZ",
        "Reg/Trd",
        "StcSMem (MB)",
        "DymSMem (MB)",
        "Bytes (MB)",
        "Throughput (MBps)",
        "SrcMemKd",
        "DstMemKd",
        "Device",
        "Ctx",
        "GreenCtx",
        "Strm",
        "Name",
    ]
    common = {
        "CorrId": "1",
        "GrdX": "1024",
        "GrdY": "1",
        "GrdZ": "1",
        "BlkX": "256",
        "BlkY": "1",
        "BlkZ": "1",
        "Reg/Trd": "20",
        "StcSMem (MB)": "0.000",
        "DymSMem (MB)": "0.000",
        "Bytes (MB)": "",
        "Throughput (MBps)": "",
        "SrcMemKd": "",
        "DstMemKd": "",
        "Device": "NVIDIA GeForce GTX 1660 Ti (0)",
        "Ctx": "1",
        "GreenCtx": "",
        "Strm": "7",
    }
    rows = [
        {
            **common,
            "Start (ns)": "1",
            "Duration (ns)": "10",
            "Name": "simllm_cache_flush_kernel(unsigned int*, unsigned long)",
        },
        {
            **common,
            "Start (ns)": "11",
            "Duration (ns)": "12.5",
            "Name": "void simllm_kv_read_kernel<float>(float*, float const*)",
        },
        {
            **common,
            "Start (ns)": "24",
            "Duration (ns)": "13.5",
            "Name": "void simllm_kv_read_kernel<float>(float*, float const*)",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        stream.write("Generating SQLite file from report\n")
        stream.write("Processing report with cuda_gpu_trace.py...\n")
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    plan = (
        CapturePlanCell(
            family="kv_read",
            dtype="fp32",
            gpu_profile=GPU.name,
            config=(("kv_tokens", 1),),
            split=TRAIN_SPLIT,
            sample_count=2,
            work_items=262_144,
            source_flops=0,
            compulsory_input_bytes=1_048_576,
            total_bytes=2_097_152,
            expected_grid_x=1024,
            expected_block_x=256,
        ),
    )

    cells = parse_nsight_cuda_gpu_trace_csv(path, plan)

    assert len(cells) == 1
    assert cells[0].durations_ps == (12_500, 13_500)
    assert cells[0].launch == _launch()
    assert cells[0].median_duration_ps == 13_000


def test_nsight_csv_parser_rejects_missing_target_rows(tmp_path):
    path = tmp_path / "trace.csv"
    path.write_text(
        "Start (ns),Duration (ns),GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,Reg/Trd,StcSMem,"
        "DymSMem,Device,Ctx,Strm,Name\n",
        encoding="utf-8",
    )
    plan = (
        CapturePlanCell(
            family="kv_read",
            dtype="fp32",
            gpu_profile=GPU.name,
            config=(("kv_tokens", 1),),
            split=TRAIN_SPLIT,
            sample_count=1,
            work_items=1,
            source_flops=0,
            compulsory_input_bytes=4,
            total_bytes=8,
            expected_grid_x=1,
            expected_block_x=256,
        ),
    )
    with pytest.raises(ValueError, match="0 target rows; expected 1"):
        parse_nsight_cuda_gpu_trace_csv(path, plan)


def test_calibration_artifact_round_trip_and_train_only_table(tmp_path):
    artifact = _artifact()
    path = artifact.save(tmp_path / "calibration.json")

    loaded = ComputeCalibrationArtifact.load(path)
    table = calibration_artifact_to_profile_table(loaded)

    assert loaded == artifact
    assert loaded.sha256 == artifact.sha256
    train = table.estimate(KernelSpec("kv_read", 0, 0, (("kv_tokens", 4),)), GPU)
    held_out = table.estimate(KernelSpec("kv_read", 0, 0, (("kv_tokens", 2),)), GPU)
    assert train.duration_ps == 400_000
    assert train.bound == "measured"
    assert held_out.duration_ps == 200_000
    assert held_out.bound == "interpolated"
    assert table.provenance is not None
    assert artifact.sha256 in table.provenance.references[0]


def test_calibration_artifact_rejects_changed_summary():
    payload = _artifact().to_json()
    payload["cells"][0]["summary"]["median_duration_ps"] += 1
    with pytest.raises(ValueError, match="summary does not match"):
        ComputeCalibrationArtifact.from_json(payload)


def test_profile_table_family_sum_is_explicit_and_conservative(tmp_path):
    provenance = ProfileTableProvenance(
        source="capture",
        version="test",
        gpu=GPU.name,
        created="2026-08-12",
    )
    entries = {
        ("attn_gemm", (("new_tokens", 1),), GPU.name): (100, 0.1),
        ("kv_read", (("kv_tokens", 1),), GPU.name): (300, 0.2),
    }
    fused = KernelSpec(
        "llm_step",
        0,
        0,
        family_kernels=(
            KernelSpec("attn_gemm", 0, 0, (("new_tokens", 1),)),
            KernelSpec("kv_read", 0, 0, (("kv_tokens", 1),)),
        ),
    )
    disabled = ProfileTableProvider(entries, provenance=provenance)
    enabled = ProfileTableProvider(
        entries,
        provenance=provenance,
        enable_family_sum=True,
    )

    with pytest.raises(KeyError):
        disabled.estimate(fused, GPU)
    estimate = enabled.estimate(fused, GPU)
    assert estimate.duration_ps == 400
    assert estimate.bound == "measured"
    assert estimate.uncertainty == pytest.approx((100 * 0.1 + 300 * 0.2) / 400)

    disabled_path = disabled.save(tmp_path / "disabled.json")
    enabled_path = enabled.save(tmp_path / "enabled.json")
    assert disabled_path.read_bytes() == enabled_path.read_bytes()
    loaded = ProfileTableProvider.load(enabled_path, enable_family_sum=True)
    assert loaded.estimate(fused, GPU) == estimate


def test_profile_table_family_sum_propagates_unsupported_family():
    provider = ProfileTableProvider({}, enable_family_sum=True)
    fused = KernelSpec(
        "llm_step",
        0,
        0,
        family_kernels=(KernelSpec("missing", 0, 0),),
    )
    with pytest.raises(KeyError, match="missing"):
        provider.estimate(fused, GPU)


def test_distribution_and_physical_helpers():
    cell = _cell(1, 100_000, TRAIN_SPLIT)
    assert nearest_rank([1, 2, 3, 4, 5], 0.95) == 5
    assert absolute_percentage_error(90, 100) == 10
    floor_ps, ceiling_ps = physical_duration_bounds_ps(
        cell,
        dtype_peak_flops_per_second=5_437_000_000_000,
        memory_bandwidth_bytes_per_second=288_000_000_000,
        serial_operations_per_second=1_500_000_000,
        serial_memory_bytes_per_second=48_000_000_000,
    )
    assert floor_ps == 3_640_889
    assert ceiling_ps == 43_690_667
    assert floor_ps < ceiling_ps
