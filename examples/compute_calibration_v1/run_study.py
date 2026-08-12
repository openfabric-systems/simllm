"""Run the frozen Turing compute-calibration study."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXPECTATIONS_PATH = HERE / "expectations.json"
EXPECTATION_COMMIT = "50c22105a34db7e645e4d8ecdce7982a3c640cdb"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _validate_registry(expectations: dict[str, Any], out: Path) -> None:
    if expectations["schema"] != "simllm-compute-calibration-v1-expectations-v1":
        raise AssertionError("expectation schema drifted")
    capture = expectations["capture"]
    if capture["families"] != [
        "attn_gemm",
        "attn_score",
        "mlp_gemm",
        "lm_head",
        "kv_read",
    ]:
        raise AssertionError("family matrix drifted")
    if capture["dtypes"] != ["fp32", "fp64"]:
        raise AssertionError("dtype matrix drifted")
    if capture["train_shapes"] != [1, 4, 16]:
        raise AssertionError("train split drifted")
    if capture["held_out_shapes"] != [2, 8]:
        raise AssertionError("held-out split drifted")
    cells = len(capture["families"]) * len(capture["dtypes"])
    if cells * len(capture["train_shapes"]) != 30:
        raise AssertionError("train denominator drifted")
    if cells * len(capture["held_out_shapes"]) != 20:
        raise AssertionError("held-out denominator drifted")
    if cells * 5 * capture["measured_launches_per_cell"] != 2050:
        raise AssertionError("target launch count drifted")
    if sum(expectations["scored_relations"].values()) != 67:
        raise AssertionError("scored instance inventory drifted")
    run_root = os.environ.get("SIMLLM_WAVE6_RUN_ROOT")
    if not run_root:
        raise RuntimeError("SIMLLM_WAVE6_RUN_ROOT must be configured")
    expected_parent = Path(run_root).resolve() / "codex" / "comp1_compute_calibration"
    try:
        out.resolve().relative_to(expected_parent)
    except ValueError as error:
        raise ValueError("output must remain under the branch wave-6 root") from error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _tool(path: Path, name: str) -> Path:
    candidate = path / "bin" / name
    if not candidate.is_file():
        raise FileNotFoundError(f"CUDA tool {name!r} is unavailable below --cuda-root: {candidate}")
    return candidate


def _run(
    command: Sequence[str | Path],
    *,
    timeout: int = 300,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [str(item) for item in command],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed with status {completed.returncode}: "
            f"{' '.join(str(item) for item in command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def _write_log(path: Path, completed: subprocess.CompletedProcess[str]) -> None:
    path.write_text(
        f"returncode: {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}\n",
        encoding="utf-8",
    )


def _git_revision() -> str:
    return _run(("git", "rev-parse", "HEAD")).stdout.strip()


def _version_line(completed: subprocess.CompletedProcess[str]) -> str:
    lines = [
        line.strip()
        for line in (completed.stdout + "\n" + completed.stderr).splitlines()
        if line.strip()
    ]
    if not lines:
        raise RuntimeError("version command produced no output")
    return " | ".join(lines)


def _device_snapshot(nvidia_smi: str) -> dict[str, Any]:
    fields = (
        "name",
        "uuid",
        "driver_version",
        "compute_cap",
        "clocks.current.sm",
        "clocks.current.memory",
        "clocks.max.sm",
        "clocks.max.memory",
    )
    completed = _run(
        (
            nvidia_smi,
            "--id=0",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        )
    )
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    if len(values) != len(fields):
        raise RuntimeError(f"unexpected nvidia-smi row: {completed.stdout!r}")
    snapshot = dict(zip(fields, values))
    for field in (
        "clocks.current.sm",
        "clocks.current.memory",
        "clocks.max.sm",
        "clocks.max.memory",
    ):
        snapshot[field] = int(snapshot[field])
    return snapshot


def _cell_config(family: str, shape: int) -> tuple[tuple[str, int], ...]:
    if family in ("attn_gemm", "mlp_gemm"):
        return (("new_tokens", shape),)
    if family == "attn_score":
        return (("new_tokens", 1), ("kv_tokens", shape))
    if family == "lm_head":
        return (("sampled", shape),)
    if family == "kv_read":
        return (("kv_tokens", shape),)
    raise ValueError(f"unknown family: {family}")


def _capture_plan(expectations: dict[str, Any]):
    from simllm.compute import HELD_OUT_SPLIT, TRAIN_SPLIT, CapturePlanCell

    capture = expectations["capture"]
    flops_per_item = {
        "attn_gemm": 16,
        "attn_score": 8,
        "mlp_gemm": 32,
        "lm_head": 24,
        "kv_read": 0,
    }
    input_arrays = {
        "attn_gemm": 2,
        "attn_score": 2,
        "mlp_gemm": 2,
        "lm_head": 2,
        "kv_read": 1,
    }
    total_arrays = {
        "attn_gemm": 3,
        "attn_score": 3,
        "mlp_gemm": 3,
        "lm_head": 3,
        "kv_read": 2,
    }
    shapes = [*capture["train_shapes"], *capture["held_out_shapes"]]
    plan = []
    for dtype in capture["dtypes"]:
        element_bytes = 4 if dtype == "fp32" else 8
        gpu_profile = f"gtx1660-ti-sm75-{dtype}"
        for family in capture["families"]:
            for shape in shapes:
                work_items = capture["work_items_per_shape_unit"] * shape
                plan.append(
                    CapturePlanCell(
                        family=family,
                        dtype=dtype,
                        gpu_profile=gpu_profile,
                        config=_cell_config(family, shape),
                        split=(TRAIN_SPLIT if shape in capture["train_shapes"] else HELD_OUT_SPLIT),
                        sample_count=capture["measured_launches_per_cell"],
                        work_items=work_items,
                        source_flops=flops_per_item[family] * work_items,
                        compulsory_input_bytes=(input_arrays[family] * element_bytes * work_items),
                        total_bytes=total_arrays[family] * element_bytes * work_items,
                        expected_grid_x=work_items // capture["block_threads"],
                        expected_block_x=capture["block_threads"],
                    )
                )
    return tuple(plan)


def _plan_json(plan) -> list[dict[str, Any]]:
    return [
        {
            "family": cell.family,
            "dtype": cell.dtype,
            "gpu_profile": cell.gpu_profile,
            "config": [[name, value] for name, value in cell.config],
            "split": cell.split,
            "sample_count": cell.sample_count,
            "work_items": cell.work_items,
            "source_flops": cell.source_flops,
            "compulsory_input_bytes": cell.compulsory_input_bytes,
            "total_bytes": cell.total_bytes,
            "expected_grid_x": cell.expected_grid_x,
            "expected_block_x": cell.expected_block_x,
        }
        for cell in plan
    ]


def _shape(cell) -> int:
    return cell.config[-1][1]


def _distribution(values: Sequence[float]) -> dict[str, float]:
    from simllm.compute import nearest_rank

    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "p95": nearest_rank(values, 0.95),
        "maximum": max(values),
    }


def _family_sum_rows(provider, artifact):
    from simllm.compute import GpuSpec, KernelSpec

    by_identity = {(cell.family, cell.dtype, _shape(cell)): cell for cell in artifact.cells}
    rows = []
    for shape in (4, 2):
        families = tuple(
            KernelSpec(
                family,
                float(by_identity[(family, "fp32", shape)].source_flops),
                float(by_identity[(family, "fp32", shape)].total_bytes),
                by_identity[(family, "fp32", shape)].config,
            )
            for family in (
                "attn_gemm",
                "attn_score",
                "mlp_gemm",
                "lm_head",
                "kv_read",
            )
        )
        fused = KernelSpec(
            "llm_step",
            sum(family.flops for family in families),
            sum(family.bytes_moved for family in families),
            family_kernels=families,
        )
        gpu = GpuSpec("gtx1660-ti-sm75-fp32", 5.437e12, 288e9)
        children = [provider.estimate(family, gpu) for family in families]
        estimate = provider.estimate(fused, gpu)
        rows.append(
            {
                "shape": shape,
                "split": "train" if shape == 4 else "held-out",
                "child_sum_ps": sum(child.duration_ps for child in children),
                "fused_duration_ps": estimate.duration_ps,
                "bound": estimate.bound,
                "uncertainty": estimate.uncertainty,
                "passed": estimate.duration_ps == sum(child.duration_ps for child in children),
            }
        )
    return rows


def _live_reachability(provider) -> dict[str, Any]:
    from simllm.backends import DeviceRuntimeStepSink, SerialStepLowererConfig
    from simllm.compute import GpuSpec, ModelDims
    from simllm.core import (
        RequestPhase,
        ScheduledRequest,
        StepRecord,
        VirtualClock,
    )

    dims = ModelDims(
        num_layers=1,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        num_kv_heads=4,
        head_size=16,
        vocab_size=256,
        dtype_bytes=4,
    )
    gpu = GpuSpec("gtx1660-ti-sm75-fp32", 5.437e12, 288e9)
    sink = DeviceRuntimeStepSink(SerialStepLowererConfig(dims, (0,), provider=provider, gpu=gpu))
    clock = VirtualClock()
    sink.bind_clock(clock)
    results = []
    for step_index, context_length in enumerate((2, 8)):
        record = StepRecord(
            step_index,
            clock.now_ps,
            [
                ScheduledRequest(
                    "decode-request",
                    RequestPhase.DECODE,
                    1,
                    context_length=context_length,
                )
            ],
            num_sampled=1,
            sampled_request_ids=["decode-request"],
        )
        result = sink(record, None)
        metric = result.request_metrics[0]
        results.append(
            {
                "step_index": step_index,
                "context_length": context_length,
                "step_latency_ps": result.step_latency_ps,
                "completed_at_ps": result.completed_at_ps,
                "ttft_ps": metric.ttft_ps,
                "tpot_numerator_ps": (None if metric.tpot_ps is None else metric.tpot_ps.numerator),
                "tpot_denominator": (
                    None if metric.tpot_ps is None else metric.tpot_ps.denominator
                ),
            }
        )
    return {
        "classification": "unscored component-to-live-chain reachability",
        "steps": results,
        "final_clock_ps": clock.now_ps,
        "graph_count": len(sink.outcomes),
    }


def _production(args: argparse.Namespace, expectations: dict[str, Any]) -> int:
    from simllm.compute import (
        ComputeCalibrationArtifact,
        ComputeCalibrationProvenance,
        GpuSpec,
        KernelSpec,
        ProfileTableProvider,
        RooflineProvider,
        calibration_artifact_to_profile_table,
        held_out_errors,
        parse_nsight_cuda_gpu_trace_csv,
        physical_duration_bounds_ps,
        sha256_file,
    )

    if args.out.exists():
        raise FileExistsError(f"immutable output directory already exists: {args.out}")
    args.out.mkdir(parents=True)
    build_dir = args.out / "build"
    capture_dir = args.out / "capture"
    build_dir.mkdir()
    capture_dir.mkdir()

    cuda_root = args.cuda_root.resolve()
    nvcc = _tool(cuda_root, "nvcc")
    nsys = _tool(cuda_root, "nsys")
    ncu = _tool(cuda_root, "ncu")
    cuobjdump = _tool(cuda_root, "cuobjdump")
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        raise FileNotFoundError("nvidia-smi must be available on PATH")

    plan = _capture_plan(expectations)
    (args.out / "capture_plan.json").write_text(
        json.dumps(_plan_json(plan), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    source = REPO_ROOT / "tools" / "compute_capture" / "compute_family_benchmark.cu"
    binary = build_dir / "compute_family_benchmark"
    compile_run = _run(
        (
            nvcc,
            "-std=c++17",
            "-O3",
            "-lineinfo",
            "-arch=sm_75",
            source,
            "-o",
            binary,
        ),
        timeout=300,
    )
    _write_log(build_dir / "nvcc.log", compile_run)

    nvcc_version = _version_line(_run((nvcc, "--version")))
    nsys_version = _version_line(_run((nsys, "--version")))
    ncu_version = _version_line(_run((ncu, "--version")))
    before = _device_snapshot(nvidia_smi)
    if before["name"] != expectations["gpu"]["device_model"]:
        raise RuntimeError(f"unexpected GPU model: {before['name']}")
    if before["compute_cap"] != expectations["gpu"]["compute_capability"]:
        raise RuntimeError(f"unexpected compute capability: {before['compute_cap']}")

    counter_run = _run(
        (ncu, "--set", "basic", "--launch-count", "1", binary, "--counter-smoke"),
        timeout=120,
        check=False,
    )
    _write_log(capture_dir / "ncu_counter_probe.log", counter_run)
    counter_output = counter_run.stdout + "\n" + counter_run.stderr
    if "ERR_NVGPUCTRPERM" in counter_output:
        counter_status = "blocked: ERR_NVGPUCTRPERM"
    elif counter_run.returncode == 0:
        counter_status = "available: Nsight Compute basic set collected"
    else:
        match = re.search(r"==ERROR==[^\n]*", counter_output)
        detail = match.group(0) if match else f"return code {counter_run.returncode}"
        counter_status = f"failed: {detail}"

    sass_run = _run((cuobjdump, "--dump-sass", binary), timeout=120)
    sass_path = build_dir / "compute_family_benchmark.sass"
    sass_path.write_text(sass_run.stdout, encoding="utf-8")
    for family in expectations["capture"]["families"]:
        if f"simllm_{family}_kernel" not in sass_run.stdout:
            raise AssertionError(f"static SASS is missing family {family}")

    report_prefix = capture_dir / "compute_family_capture"
    capture_run = _run(
        (
            nsys,
            "profile",
            "--trace=cuda",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--force-overwrite=true",
            "--output",
            report_prefix,
            binary,
        ),
        timeout=900,
    )
    _write_log(capture_dir / "nsys_profile.log", capture_run)
    report = report_prefix.with_suffix(".nsys-rep")
    if not report.is_file():
        raise RuntimeError(f"Nsight Systems did not produce {report}")
    stats_run = _run(
        (
            nsys,
            "stats",
            "--report",
            "cuda_gpu_trace",
            "--format",
            "csv",
            "--output",
            "-",
            report,
        ),
        timeout=300,
    )
    trace_csv = capture_dir / "cuda_gpu_trace.csv"
    trace_csv.write_text(stats_run.stdout, encoding="utf-8")
    (capture_dir / "nsys_stats.stderr.log").write_text(
        stats_run.stderr,
        encoding="utf-8",
    )
    cells = parse_nsight_cuda_gpu_trace_csv(trace_csv, plan)
    after = _device_snapshot(nvidia_smi)

    provenance = ComputeCalibrationProvenance(
        gpu_model=before["name"],
        gpu_uuid=before["uuid"],
        compute_capability=before["compute_cap"],
        driver_version=before["driver_version"],
        cuda_version=nvcc_version,
        nsight_systems_version=nsys_version,
        nsight_compute_version=ncu_version,
        source_sha256=sha256_file(source),
        binary_sha256=sha256_file(binary),
        static_sass_sha256=sha256_file(sass_path),
        capture_sha256=sha256_file(report),
        creation_date=expectations["capture"]["creation_date"],
        warmup_policy=(
            f"{expectations['capture']['warmup_launches_per_cell']} target launches "
            "per cell before cudaProfilerStart"
        ),
        cache_policy=(
            f"{expectations['capture']['cache_flush_bytes']} byte flush kernel "
            "before every measured target"
        ),
        clock_policy="unlocked display GPU; endpoint clocks recorded",
        core_clock_before_mhz=before["clocks.current.sm"],
        core_clock_after_mhz=after["clocks.current.sm"],
        memory_clock_before_mhz=before["clocks.current.memory"],
        memory_clock_after_mhz=after["clocks.current.memory"],
        performance_counter_status=counter_status,
        references=(
            "https://docs.nvidia.com/nsight-systems/2023.4/UserGuide/index.html",
            "https://docs.nvidia.com/nsight-compute/2024.1/ReleaseNotes/topics/gpu-support.html",
            "https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters",
        ),
    )
    artifact = ComputeCalibrationArtifact(provenance=provenance, cells=cells)
    calibration_path = args.out / "calibration.json"
    artifact.save(calibration_path)
    loaded_artifact = ComputeCalibrationArtifact.load(calibration_path)
    if loaded_artifact != artifact:
        raise AssertionError("calibration artifact roundtrip changed the record")

    table = calibration_artifact_to_profile_table(
        artifact,
        enable_family_sum=True,
    )
    table_path = args.out / "profile_table.json"
    table.save(table_path)
    disabled_table = ProfileTableProvider.load(table_path)
    calibrated = ProfileTableProvider.load(table_path, enable_family_sum=True)
    roundtrip_path = args.out / "profile_table_roundtrip.json"
    disabled_table.save(roundtrip_path)
    table_roundtrip_identical = table_path.read_bytes() == roundtrip_path.read_bytes()

    gpu_specs = {
        "gtx1660-ti-sm75-fp32": GpuSpec(
            "gtx1660-ti-sm75-fp32",
            expectations["gpu"]["fp32_peak_flops_per_second"],
            expectations["gpu"]["memory_bandwidth_bytes_per_second"],
        ),
        "gtx1660-ti-sm75-fp64": GpuSpec(
            "gtx1660-ti-sm75-fp64",
            expectations["gpu"]["fp64_peak_flops_per_second"],
            expectations["gpu"]["memory_bandwidth_bytes_per_second"],
        ),
    }
    rooflines = {name: RooflineProvider(efficiency=0.7) for name in gpu_specs}
    error_rows = held_out_errors(
        artifact,
        calibrated,
        rooflines,
        gpu_specs,
    )
    calibrated_errors = [row["calibrated_absolute_percentage_error"] for row in error_rows]
    roofline_errors = [row["roofline_absolute_percentage_error"] for row in error_rows]
    calibrated_distribution = _distribution(calibrated_errors)
    roofline_distribution = _distribution(roofline_errors)

    physical_rows = []
    physical_ok = True
    cv_ok = True
    for cell in artifact.cells:
        peak = (
            expectations["gpu"]["fp32_peak_flops_per_second"]
            if cell.dtype == "fp32"
            else expectations["gpu"]["fp64_peak_flops_per_second"]
        )
        floor_ps, ceiling_ps = physical_duration_bounds_ps(
            cell,
            dtype_peak_flops_per_second=peak,
            memory_bandwidth_bytes_per_second=expectations["gpu"][
                "memory_bandwidth_bytes_per_second"
            ],
            serial_operations_per_second=expectations["gpu"]["base_clock_hz"],
            serial_memory_bytes_per_second=expectations["gpu"]["memory_channel_bytes_per_second"],
        )
        bounds_pass = min(cell.durations_ps) >= floor_ps and max(cell.durations_ps) <= ceiling_ps
        cv_pass = (
            cell.coefficient_of_variation
            < expectations["accuracy_bands"]["cell_coefficient_of_variation_max"]
        )
        physical_ok = physical_ok and bounds_pass
        cv_ok = cv_ok and cv_pass
        physical_rows.append(
            {
                "family": cell.family,
                "dtype": cell.dtype,
                "config": [[name, value] for name, value in cell.config],
                "split": cell.split,
                "floor_ps": floor_ps,
                "ceiling_ps": ceiling_ps,
                "minimum_duration_ps": min(cell.durations_ps),
                "median_duration_ps": cell.median_duration_ps,
                "maximum_duration_ps": max(cell.durations_ps),
                "median_over_floor": cell.median_duration_ps / floor_ps,
                "ceiling_over_median": ceiling_ps / cell.median_duration_ps,
                "coefficient_of_variation": cell.coefficient_of_variation,
                "bounds_passed": bounds_pass,
                "variation_passed": cv_pass,
            }
        )

    by_identity = {(cell.family, cell.dtype, _shape(cell)): cell for cell in artifact.cells}
    monotonic_rows = []
    scaling_rows = []
    for family in expectations["capture"]["families"]:
        for dtype in expectations["capture"]["dtypes"]:
            medians = {
                shape: by_identity[(family, dtype, shape)].median_duration_ps
                for shape in (1, 2, 4, 8, 16)
            }
            monotonic = all(
                medians[left] < medians[right] for left, right in ((1, 2), (2, 4), (4, 8), (8, 16))
            )
            monotonic_rows.append(
                {
                    "family": family,
                    "dtype": dtype,
                    "medians_ps": medians,
                    "passed": monotonic,
                }
            )
            for lower, upper in ((1, 4), (4, 16)):
                ratio = medians[upper] / medians[lower]
                scaling_rows.append(
                    {
                        "family": family,
                        "dtype": dtype,
                        "lower_shape": lower,
                        "upper_shape": upper,
                        "duration_ratio": ratio,
                        "passed": (
                            expectations["physical_guards"]["minimum_shape_ratio"]
                            <= ratio
                            <= expectations["physical_guards"]["maximum_shape_ratio"]
                        ),
                    }
                )

    dtype_rows = []
    for family in expectations["capture"]["families"]:
        for shape in (1, 2, 4, 8, 16):
            fp32_ps = by_identity[(family, "fp32", shape)].median_duration_ps
            fp64_ps = by_identity[(family, "fp64", shape)].median_duration_ps
            ratio = fp64_ps / fp32_ps
            dtype_rows.append(
                {
                    "family": family,
                    "shape": shape,
                    "fp32_median_ps": fp32_ps,
                    "fp64_median_ps": fp64_ps,
                    "duration_ratio": ratio,
                    "passed": ratio
                    >= expectations["physical_guards"]["minimum_fp64_to_fp32_ratio"],
                }
            )

    family_sum_rows = _family_sum_rows(calibrated, artifact)
    try:
        disabled_table.estimate(
            KernelSpec(
                "llm_step",
                0,
                0,
                family_kernels=(KernelSpec("kv_read", 0, 0, (("kv_tokens", 4),)),),
            ),
            gpu_specs["gtx1660-ti-sm75-fp32"],
        )
    except KeyError:
        family_sum_disabled_identity = True
    else:
        family_sum_disabled_identity = False

    accuracy = expectations["accuracy_bands"]
    accuracy_global_pass = (
        calibrated_distribution["median"]
        <= accuracy["calibrated_median_absolute_percentage_error_max"]
        and calibrated_distribution["p95"]
        <= accuracy["calibrated_p95_absolute_percentage_error_max"]
    )
    roofline_missed = (
        roofline_distribution["median"]
        > accuracy["calibrated_median_absolute_percentage_error_max"]
        or roofline_distribution["p95"] > accuracy["calibrated_p95_absolute_percentage_error_max"]
    )
    calibrated_beats_roofline = (
        calibrated_distribution["median"] < roofline_distribution["median"]
        and calibrated_distribution["p95"] < roofline_distribution["p95"]
    )
    for row in error_rows:
        row["passed"] = (
            row["calibrated_absolute_percentage_error"]
            <= accuracy["calibrated_p95_absolute_percentage_error_max"]
        )

    scores = {
        "held_out_accuracy_distribution": {
            "passed": sum(row["passed"] for row in error_rows),
            "total": len(error_rows),
            "global_band_passed": accuracy_global_pass,
            "roofline_missed_same_band": roofline_missed,
            "calibrated_beats_roofline": calibrated_beats_roofline,
        },
        "train_shape_scaling": {
            "passed": sum(row["passed"] for row in scaling_rows),
            "total": len(scaling_rows),
        },
        "dtype_slowdown": {
            "passed": sum(row["passed"] for row in dtype_rows),
            "total": len(dtype_rows),
        },
        "family_sum_opt_in": {
            "passed": sum(row["passed"] for row in family_sum_rows),
            "total": len(family_sum_rows),
        },
    }

    live_reachability = _live_reachability(calibrated)
    results = {
        "schema": "simllm-compute-calibration-v1-results-v1",
        "provenance": {
            "expectation_commit": EXPECTATION_COMMIT,
            "observed_implementation_commit": _git_revision(),
            "authored_against": expectations["authored_against"],
            "calibration_sha256": artifact.sha256,
            "profile_table_sha256": sha256_file(table_path),
            "capture_sha256": provenance.capture_sha256,
            "source_sha256": provenance.source_sha256,
            "binary_sha256": provenance.binary_sha256,
            "static_sass_sha256": provenance.static_sass_sha256,
            "gpu_model": provenance.gpu_model,
            "gpu_uuid": provenance.gpu_uuid,
            "driver_version": provenance.driver_version,
            "compute_capability": provenance.compute_capability,
            "cuda_version": provenance.cuda_version,
            "nsight_systems_version": provenance.nsight_systems_version,
            "nsight_compute_version": provenance.nsight_compute_version,
            "performance_counter_status": counter_status,
            "counter_probe_returncode": counter_run.returncode,
            "clock_before": before,
            "clock_after": after,
        },
        "capture_inventory": {
            "cells": len(artifact.cells),
            "train_cells": sum(cell.split == "train" for cell in artifact.cells),
            "held_out_cells": sum(cell.split == "held-out" for cell in artifact.cells),
            "target_rows": sum(len(cell.durations_ps) for cell in artifact.cells),
            "profile_table_entries": len(_load_json(table_path)["entries"]),
        },
        "held_out_errors": error_rows,
        "calibrated_error_distribution_percent": calibrated_distribution,
        "roofline_error_distribution_percent": roofline_distribution,
        "physical_rows": physical_rows,
        "monotonic_rows": monotonic_rows,
        "train_shape_scaling_rows": scaling_rows,
        "dtype_rows": dtype_rows,
        "family_sum_rows": family_sum_rows,
        "live_reachability": live_reachability,
        "scores": scores,
        "fatal_unscored": {
            "physical_bounds_passed": physical_ok,
            "cell_variation_passed": cv_ok,
            "strict_all_shape_monotonicity_passed": all(row["passed"] for row in monotonic_rows),
            "table_roundtrip_byte_identical": table_roundtrip_identical,
            "family_sum_disabled_identity": family_sum_disabled_identity,
            "capture_row_count_exact": sum(len(cell.durations_ps) for cell in artifact.cells)
            == 2050,
            "train_held_out_split_exact": (
                sum(cell.split == "train" for cell in artifact.cells) == 30
                and sum(cell.split == "held-out" for cell in artifact.cells) == 20
            ),
        },
        "evidence_accounting": {
            "scored_families": 4,
            "scored_instances": sum(item["total"] for item in scores.values()),
            "run_configurations": 50,
            "raw_duration_samples": 2050,
            "physical_and_structural_guards": "fatal unscored",
            "tool_smokes": "separate evidence class",
            "live_reachability": "unscored successor evidence",
        },
    }
    results_path = args.out / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    all_scored = all(item["passed"] == item["total"] for item in scores.values())
    all_fatal = all(results["fatal_unscored"].values())
    if not (
        all_scored
        and all_fatal
        and accuracy_global_pass
        and roofline_missed
        and calibrated_beats_roofline
    ):
        raise AssertionError(
            f"compute calibration missed a frozen acceptance relation; inspect {results_path}"
        )
    print(
        json.dumps(
            {
                "calibrated_error_percent": calibrated_distribution,
                "roofline_error_percent": roofline_distribution,
                "scores": scores,
                "counter_status": counter_status,
                "results": str(results_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    args = _parse_args()
    expectations = _load_json(EXPECTATIONS_PATH)
    _validate_registry(expectations, args.out)
    if args.check_only:
        print("check-only: 50 cells, 2050 target rows, 67 scored instances")
        return 0
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return _production(args, expectations)


if __name__ == "__main__":
    raise SystemExit(main())
