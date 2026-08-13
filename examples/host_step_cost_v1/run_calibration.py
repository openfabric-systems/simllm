"""Repeat the Turing host-step calibration under the corrected fatal oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPECTATIONS = HERE / "expectations.json"
CALIBRATION = HERE / "calibration.json"
PROBE_SOURCE = ROOT / "tools" / "compute_capture" / "gpu_fixed_cost_probe.cu"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _run(command: Sequence[object], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in command],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_scalars(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("samples_begin") or line == "samples_end":
            continue
        fields = line.split(",")
        if fields[0] == "sample":
            continue
        if len(fields) != 2 or not fields[0] or fields[0] in values:
            raise ValueError(f"malformed or duplicate probe scalar: {line!r}")
        values[fields[0]] = fields[1]
    return values


def _picoseconds(value: str) -> int:
    result = Decimal(value) * 1000
    if result != result.to_integral_value():
        raise ValueError(f"nanosecond value has sub-picosecond precision: {value}")
    return int(result)


def _load_expectations() -> dict[str, Any]:
    return json.loads(EXPECTATIONS.read_text(encoding="utf-8"))


def _environment(args: argparse.Namespace) -> dict[str, Any]:
    values = _load_expectations()
    if values["schema"] != "simllm-host-step-cost-v1-expectations-v1":
        raise AssertionError("expectation schema drifted")
    if values["task"] != "COMP-2" or values["calibration_attempt"] != 2:
        raise AssertionError("calibration identity drifted")
    if sum(values["scored_calibration_relations"].values()) != 4:
        raise AssertionError("scored calibration inventory drifted")
    if sum(values["scored_live_relations"].values()) != 12:
        raise AssertionError("scored live inventory drifted")
    if len(values["fatal_unscored_guards"]) != 11:
        raise AssertionError("fatal guard inventory drifted")
    prior = values["prior_attempt"]
    if prior["disposition"] != "void_with_findings":
        raise AssertionError("prior void disposition drifted")
    if prior["integer_scaling_residual_ps"] != 1:
        raise AssertionError("prior integer residual drifted")
    prior_names = (
        "graph_replay_ps",
        "eager_host_bound_ps",
        "stamped_device_gap_ps",
        "serialized_launch_ps",
    )
    if any(len(prior[name]) != 3 for name in prior_names):
        raise AssertionError("prior observation inventory drifted")
    if any(len(value) != 64 for value in prior["launch_csv_sha256"].values()):
        raise AssertionError("prior launch content identity drifted")
    capture = values["capture"]
    if capture["device_key"] != "gtx1660-ti-sm75":
        raise AssertionError("capture device key drifted")
    if capture["graph_nodes"] != 512 or capture["graph_replays"] != 200:
        raise AssertionError("graph capture shape drifted")
    if capture["launch_iterations"] != 20_000:
        raise AssertionError("eager capture shape drifted")
    if capture["long_backtoback_iterations"] != 400:
        raise AssertionError("stamped capture shape drifted")
    if not all(
        len(bounds) == 2 and 0 < bounds[0] < bounds[1]
        for bounds in capture["acceptance_ps"].values()
    ):
        raise AssertionError("calibration bands drifted")
    representative = values["representative_step"]
    computed = tuple(
        1.0
        + representative["modeled_compute_ps"]
        / representative["modeled_decode_makespan_ps"]
        * multiple
        for multiple in representative["retained_omitted_multiple"]
    )
    if any(
        not math.isclose(observed, expected, rel_tol=0.0, abs_tol=5e-11)
        for observed, expected in zip(
            computed,
            representative["derived_decode_multiplier"],
            strict=True,
        )
    ):
        raise AssertionError("derived multiplier arithmetic drifted")
    root_value = os.environ.get("SIMLLM_WAVE12_RUN_ROOT")
    if not root_value:
        raise RuntimeError("SIMLLM_WAVE12_RUN_ROOT must be configured")
    args.out.resolve().relative_to(Path(root_value).resolve())
    if args.out.exists():
        raise FileExistsError(f"output already exists: {args.out}")
    nvcc = args.cuda_root / "bin" / "nvcc"
    if not nvcc.is_file() or not os.access(nvcc, os.X_OK):
        raise FileNotFoundError(f"CUDA compiler is missing or not executable: {nvcc}")
    nvcc_output = _run((nvcc, "--version")).stdout
    if "release 12.4" not in nvcc_output or "V12.4.99" not in nvcc_output:
        raise RuntimeError("CUDA compiler identity does not match the freeze")
    gpu_rows = _run(
        (
            "nvidia-smi",
            "--query-gpu=name,uuid,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        )
    ).stdout.strip().splitlines()
    expected_gpu = ", ".join(
        (
            capture["device_name"],
            capture["gpu_uuid"],
            capture["compute_capability"],
            capture["driver_version"],
        )
    )
    if gpu_rows != [expected_gpu]:
        raise RuntimeError("GPU identity does not match the freeze")
    cpu_model = ""
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    if cpu_model != capture["host_cpu"]:
        raise RuntimeError("host CPU identity does not match the freeze")
    return {
        "expectations": values,
        "nvcc": nvcc,
        "nvcc_output": nvcc_output,
        "gpu_query": gpu_rows[0],
        "cpu_model": cpu_model,
    }


def _clean_head() -> str:
    status = _run(("git", "status", "--porcelain", "--untracked-files=all")).stdout
    if status:
        raise RuntimeError(
            "capture requires a clean tracked and untracked worktree before output"
        )
    return _run(("git", "rev-parse", "HEAD")).stdout.strip()


def _expectation_commit() -> str:
    return _run(
        (
            "git",
            "log",
            "-1",
            "--format=%H",
            "--",
            EXPECTATIONS,
            HERE / "expectations.md",
        )
    ).stdout.strip()


def _zero_work_oracle() -> dict[str, int]:
    from simllm.compute import GPU_ENVELOPES, KernelSpec, RooflineProvider

    provider = RooflineProvider()
    gpu = GPU_ENVELOPES["b100"]
    zero = provider.estimate(
        KernelSpec(name="zero", flops=0.0, bytes_moved=0.0), gpu
    )
    single = provider.estimate(
        KernelSpec(name="positive", flops=1.0e12, bytes_moved=5.564e8), gpu
    )
    doubled = provider.estimate(
        KernelSpec(name="positive", flops=2.0e12, bytes_moved=1.1128e9), gpu
    )
    return {
        "zero_work_ps": zero.duration_ps,
        "single_ps": single.duration_ps,
        "doubled_ps": doubled.duration_ps,
        "positive_scaling_residual_ps": doubled.duration_ps - 2 * single.duration_ps,
    }


def _guard(identifier: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"id": identifier, "passed": bool(passed), "detail": detail}


def _score(
    identifier: str,
    observed_ps: int,
    bounds_ps: Sequence[int],
) -> dict[str, Any]:
    lower, upper = bounds_ps
    return {
        "id": identifier,
        "passed": lower <= observed_ps <= upper,
        "observed_ps": observed_ps,
        "acceptance_ps": [lower, upper],
    }


def _disposition(guards: Sequence[dict[str, Any]], scored: Sequence[dict[str, Any]]) -> str:
    if any(not row["passed"] for row in guards):
        return "void"
    if any(not row["passed"] for row in scored):
        return "not_accepted"
    return "accepted"


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8", newline="\n")


def _capture(args: argparse.Namespace, environment: dict[str, Any]) -> int:
    values = environment["expectations"]
    capture = values["capture"]
    observed_head = _clean_head()
    entry_hashes = {
        "capture_harness_sha256": _sha256(Path(__file__)),
        "probe_source_sha256": _sha256(PROBE_SOURCE),
    }
    args.out.mkdir(parents=True)
    build_dir = args.out / "build"
    capture_dir = args.out / "capture"
    build_dir.mkdir()
    capture_dir.mkdir()
    probe = build_dir / "gpu_fixed_cost_probe"
    compile_run = _run(
        (
            environment["nvcc"],
            "-std=c++17",
            "-O3",
            "-lineinfo",
            "-arch=sm_75",
            PROBE_SOURCE,
            "-o",
            probe,
        ),
        timeout=600,
    )
    _write_text(build_dir / "nvcc.stdout.log", compile_run.stdout)
    _write_text(build_dir / "nvcc.stderr.log", compile_run.stderr)

    identity_run = _run((probe, "--identity"))
    launch_run = _run(
        (
            probe,
            "--launch",
            capture["launch_iterations"],
            capture["graph_nodes"],
            capture["graph_replays"],
            capture["long_isolated_iterations"],
            capture["long_backtoback_iterations"],
        ),
        timeout=1800,
    )
    identity_path = capture_dir / "identity.csv"
    launch_path = capture_dir / "launch.csv"
    _write_text(identity_path, identity_run.stdout)
    _write_text(launch_path, launch_run.stdout)
    identity = _parse_scalars(identity_run.stdout)
    launch = _parse_scalars(launch_run.stdout)
    required = {
        "empty_stream_cpu_enqueue_ns",
        "empty_stream_pipelined_ns",
        "empty_stream_serialized_ns",
        "empty_graph_ns",
        "long_isolated_ns",
        "long_backtoback_ns",
        "stamped_batch_wall_ns",
        "stamped_kernel_service_ns",
        "stamped_device_gap_ns",
    }
    if not required.issubset(launch):
        missing = ", ".join(sorted(required - set(launch)))
        raise RuntimeError(f"launch probe omitted required values: {missing}")
    measured_ps = {
        "cpu_enqueue": _picoseconds(launch["empty_stream_cpu_enqueue_ns"]),
        "eager_host_bound": _picoseconds(launch["empty_stream_pipelined_ns"]),
        "serialized_launch": _picoseconds(launch["empty_stream_serialized_ns"]),
        "graph_replay": _picoseconds(launch["empty_graph_ns"]),
        "stamped_device_gap": _picoseconds(launch["stamped_device_gap_ns"]),
        "stamped_batch_wall": _picoseconds(launch["stamped_batch_wall_ns"]),
        "stamped_kernel_service": _picoseconds(launch["stamped_kernel_service_ns"]),
    }
    integers = {
        key: int(launch[key])
        for key in (
            "empty_iterations",
            "empty_serialized_iterations",
            "graph_nodes",
            "graph_replays",
            "long_isolated_iterations",
            "long_backtoback_iterations",
        )
    }
    oracle = _zero_work_oracle()
    end_head = _run(("git", "rev-parse", "HEAD")).stdout.strip()
    exit_hashes = {
        "capture_harness_sha256": _sha256(Path(__file__)),
        "probe_source_sha256": _sha256(PROBE_SOURCE),
    }

    guards = [
        _guard(
            "CAL-G1_device_identity",
            identity.get("device_name") == capture["device_name"]
            and identity.get("device_uuid") == capture["gpu_uuid"]
            and identity.get("compute_capability") == capture["compute_capability"],
            {
                "device_key": capture["device_key"],
                "device_name": identity.get("device_name"),
                "gpu_uuid": identity.get("device_uuid"),
                "compute_capability": identity.get("compute_capability"),
            },
        ),
        _guard(
            "CAL-G2_host_and_toolchain_identity",
            environment["cpu_model"] == capture["host_cpu"]
            and capture["driver_version"] in environment["gpu_query"]
            and "V12.4.99" in environment["nvcc_output"],
            {
                "host_cpu": environment["cpu_model"],
                "driver_version": capture["driver_version"],
                "cuda_version": capture["cuda_version"],
                "compile_target": "sm_75",
            },
        ),
        _guard(
            "CAL-G3_positive_measurements",
            all(value > 0 for value in measured_ps.values())
            and integers
            == {
                "empty_iterations": capture["launch_iterations"],
                "empty_serialized_iterations": min(
                    capture["launch_iterations"], 5000
                ),
                "graph_nodes": capture["graph_nodes"],
                "graph_replays": capture["graph_replays"],
                "long_isolated_iterations": capture["long_isolated_iterations"],
                "long_backtoback_iterations": capture[
                    "long_backtoback_iterations"
                ],
            },
            {"measurements_ps": measured_ps, "counts": integers},
        ),
        _guard(
            "CAL-G4_zero_work_is_exactly_zero",
            oracle["zero_work_ps"] == 0,
            oracle,
        ),
        _guard(
            "CAL-G5_causal_measurement_enclosures",
            0 < measured_ps["graph_replay"] <= measured_ps["serialized_launch"]
            and 0 < measured_ps["eager_host_bound"] <= measured_ps["serialized_launch"]
            and 0
            < measured_ps["stamped_device_gap"]
            <= measured_ps["stamped_batch_wall"]
            // capture["long_backtoback_iterations"],
            {
                "graph_replay_ps": measured_ps["graph_replay"],
                "eager_host_bound_ps": measured_ps["eager_host_bound"],
                "empty_serialized_ps": measured_ps["serialized_launch"],
                "stamped_device_gap_ps": measured_ps["stamped_device_gap"],
                "stamped_batch_wall_per_launch_ps": (
                    measured_ps["stamped_batch_wall"]
                    // capture["long_backtoback_iterations"]
                ),
            },
        ),
        _guard(
            "CAL-G6_clean_revision_and_stable_capture_sources",
            observed_head == end_head and entry_hashes == exit_hashes,
            {
                "repository_clean_before_output": True,
                "entry_head": observed_head,
                "exit_head": end_head,
                "entry_hashes": entry_hashes,
                "exit_hashes": exit_hashes,
            },
        ),
    ]
    bands = capture["acceptance_ps"]
    scored = [
        _score(
            "CAL-1_graph_replay_in_band",
            measured_ps["graph_replay"],
            bands["graph_replay"],
        ),
        _score(
            "CAL-2_eager_host_bound_in_band",
            measured_ps["eager_host_bound"],
            bands["eager_host_bound"],
        ),
        _score(
            "CAL-3_device_gap_in_band",
            measured_ps["stamped_device_gap"],
            bands["stamped_device_gap"],
        ),
        _score(
            "CAL-4_serialized_enclosure_in_band",
            measured_ps["serialized_launch"],
            bands["serialized_launch"],
        ),
    ]
    derived = {
        "id": "CAL-D1_graph_cheaper_than_eager",
        "passed": measured_ps["graph_replay"] < measured_ps["eager_host_bound"],
        "scored": False,
    }
    fatal_failures = [row["id"] for row in guards if not row["passed"]]
    scored_passed = sum(1 for row in scored if row["passed"])
    status = _disposition(guards, scored)

    prior = values["prior_attempt"]
    profile_specs = (
        (
            "turing-cuda-graph",
            "cuda-graph-node",
            "graph_replay",
            prior["graph_replay_ps"],
        ),
        (
            "turing-eager-host",
            "eager-host-bound",
            "eager_host_bound",
            prior["eager_host_bound_ps"],
        ),
    )
    profiles = {}
    for profile, launch_class, key, old_values in profile_specs:
        observations = [*old_values, measured_ps[key]]
        profiles[profile] = {
            "launch_class": launch_class,
            "point_ps_per_launch": measured_ps[key],
            "empirical_min_ps_per_launch": min(observations),
            "empirical_max_ps_per_launch": max(observations),
            "empirical_observations_ps": observations,
            "uncertainty_kind": "sample-limited empirical range, not a confidence interval",
        }
    result = {
        "schema": "simllm-host-step-calibration-v1",
        "study": "host_step_cost_v1",
        "task": "COMP-2",
        "attempt": 2,
        "expectation_commit": _expectation_commit(),
        "observed_commit": observed_head,
        "run_status": status,
        "behavioral_score_interpretable": not fatal_failures,
        "fatal_guard_failures": fatal_failures,
        "fatal_guards": guards,
        "scored_relations": scored,
        "derived_unscored_checks": [derived],
        "descriptive_positive_scaling_residual_ps": oracle[
            "positive_scaling_residual_ps"
        ],
        "device": {
            "device_key": capture["device_key"],
            "device_name": capture["device_name"],
            "gpu_uuid": capture["gpu_uuid"],
            "compute_capability": capture["compute_capability"],
            "host_cpu": capture["host_cpu"],
            "driver_version": capture["driver_version"],
            "cuda_version": capture["cuda_version"],
            "compile_target": "sm_75",
        },
        "capture_shape": {
            "launch_iterations": capture["launch_iterations"],
            "graph_nodes": capture["graph_nodes"],
            "graph_replays": capture["graph_replays"],
            "long_isolated_iterations": capture["long_isolated_iterations"],
            "long_backtoback_iterations": capture[
                "long_backtoback_iterations"
            ],
        },
        "measurements_ps": measured_ps,
        "profiles": profiles,
        "raw_artifacts": {
            "identity_sha256": _sha256(identity_path),
            "launch_sha256": _sha256(launch_path),
        },
        "source_identity": entry_hashes,
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    _write_text(args.out / "results.json", payload)
    _write_text(CALIBRATION, payload)

    if fatal_failures:
        print("run is VOID for closure purposes: fatal guard findings follow")
        for identifier in fatal_failures:
            print(f"  {identifier}")
        print("behavioral score suppressed")
        return 1
    print("all fatal calibration guards held")
    print(f"scored calibration relations: {scored_passed} of {len(scored)}")
    if scored_passed != len(scored):
        print("registered calibration acceptance was not met")
        return 1
    print("calibration attempt two accepted")
    return 0


def main() -> int:
    args = _parse_args()
    environment = _environment(args)
    if args.check_only:
        print(
            "check-only: calibration registry, arithmetic and environment valid; "
            "no CUDA workload invoked; no output created"
        )
        return 0
    return _capture(args, environment)


if __name__ == "__main__":
    raise SystemExit(main())
