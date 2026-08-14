"""Regression checks for the A100 profiler-environment qualification."""

from __future__ import annotations

import csv
import importlib.util
import inspect
import io
import os
import subprocess
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    REPOSITORY / "examples/a100_environment_qualification_v1/run_qualification.py"
)


def _runner_module():
    spec = importlib.util.spec_from_file_location(
        "a100_environment_qualification_v1", RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nsys_csv(kernel_name: str) -> str:
    stream = io.StringIO()
    stream.write("Processing report with cuda_gpu_trace.py...\n")
    fieldnames = [
        "Start (ns)",
        "Duration (ns)",
        "GrdX",
        "GrdY",
        "GrdZ",
        "BlkX",
        "BlkY",
        "BlkZ",
        "Device",
        "Ctx",
        "Strm",
        "Name",
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(
        {
            "Start (ns)": "100",
            "Duration (ns)": "25.5",
            "GrdX": "65536",
            "GrdY": "1",
            "GrdZ": "1",
            "BlkX": "256",
            "BlkY": "1",
            "BlkZ": "1",
            "Device": "NVIDIA A100-SXM4-80GB (0)",
            "Ctx": "1",
            "Strm": "7",
            "Name": kernel_name,
        }
    )
    return stream.getvalue()


def test_nsys_trace_accepts_public_cuda_gpu_trace_columns():
    runner = _runner_module()

    rows = runner._validate_nsys_trace(_nsys_csv(runner.TARGET_KERNEL))

    assert rows == [
        {
            "name": runner.TARGET_KERNEL,
            "start_ns": 100.0,
            "duration_ns": 25.5,
            "device": "NVIDIA A100-SXM4-80GB (0)",
            "context_id": 1,
            "stream_id": 7,
            "grid": [65536, 1, 1],
            "block": [256, 1, 1],
        }
    ]


def test_nsys_trace_refuses_missing_target_kernel():
    runner = _runner_module()

    with pytest.raises(RuntimeError, match="no target-kernel row"):
        runner._validate_nsys_trace(_nsys_csv("unrelated_kernel"))


def test_nsys_trace_refuses_geometry_drift():
    runner = _runner_module()
    trace = _nsys_csv(runner.TARGET_KERNEL).replace("65536,1,1", "65535,1,1")

    with pytest.raises(RuntimeError, match="target grid drifted"):
        runner._validate_nsys_trace(trace)


@pytest.mark.parametrize(
    ("job_gpus", "visible", "expected"),
    [
        ("1", "0", "0"),
        ("gpu:3", "2", "2"),
        (
            "GPU-acde0000-1111-2222-3333-444444444444",
            "GPU-acde9999-1111-2222-3333-444444444444",
            "GPU-acde9999-1111-2222-3333-444444444444",
        ),
    ],
)
def test_job_visible_gpu_selector_honors_device_remapping(
    job_gpus, visible, expected
):
    runner = _runner_module()

    assert (
        runner._job_visible_gpu_selector(
            {"job_gpus": job_gpus, "cuda_visible_devices": visible}
        )
        == expected
    )


@pytest.mark.parametrize("visible", ["unparseable", "0,1", ""])
def test_job_visible_gpu_selector_refuses_unparseable_device(visible):
    runner = _runner_module()

    with pytest.raises(RuntimeError, match="job-local GPU selector"):
        runner._job_visible_gpu_selector(
            {"job_gpus": "gpu:3", "cuda_visible_devices": visible}
        )


def test_child_environment_is_confined_to_result_root(tmp_path):
    runner = _runner_module()
    out = tmp_path / "result"
    out.mkdir()

    environment = runner._configure_child_environment(out)
    record = runner._child_environment_record(out, environment)

    assert set(environment) == {
        "HOME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "CUDA_CACHE_PATH",
    }
    assert record["TMPDIR"] == "tool-state/tmp"
    assert record["TMP"] == record["TMPDIR"]
    assert record["TEMP"] == record["TMPDIR"]
    assert all((out / relative).is_dir() for relative in record.values())
    if os.name == "posix":
        assert all(
            (out / relative).stat().st_mode & 0o777 == 0o700
            for relative in record.values()
        )


def test_run_propagates_configured_child_environment(monkeypatch):
    runner = _runner_module()
    observed = {}

    def fake_run(command, **kwargs):
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner._QUALIFICATION_ENVIRONMENT = {
        "HOME": "/qualified/home",
        "TMPDIR": "/qualified/tmp",
    }

    runner._run(("qualification-tool", "--version"))

    assert observed["env"]["HOME"] == "/qualified/home"
    assert observed["env"]["TMPDIR"] == "/qualified/tmp"
    assert observed["env"]["PATH"]


def test_nsys_output_paths_must_match_configured_roots(tmp_path):
    runner = _runner_module()
    out = tmp_path / "result"
    out.mkdir()
    temporary = out / "tool-state/tmp/nsys-report-abcd.qdstrm"
    report = out / "capture/a100_environment_probe.nsys-rep"
    output = f"Generating '{temporary}'\nGenerated:\n\t{report}\n"

    assert runner._validate_nsys_output_paths(output, out, report) == {
        "intermediate": "tool-state/tmp/nsys-report-abcd.qdstrm",
        "report": "capture/a100_environment_probe.nsys-rep",
    }

    escaped = (
        "Generating '/tmp/nsys-report-b336.qdstrm'\n"
        f"Generated:\n\t{report}\n"
    )
    with pytest.raises(RuntimeError, match="escaped the configured temporary root"):
        runner._validate_nsys_output_paths(
            escaped, out, report
        )


@pytest.mark.parametrize(
    "temporary",
    ["relative.qdstrm", "/tmp/nsys-report.qdstrm", "{sibling}/nsys-report.qdstrm"],
)
def test_nsys_output_paths_refuse_unscoped_temporary_path(temporary, tmp_path):
    runner = _runner_module()
    out = tmp_path / "result"
    out.mkdir()
    report = out / "capture/a100_environment_probe.nsys-rep"
    rendered = temporary.format(sibling=f"{out}-sibling")
    output = f"Generating '{rendered}'\nGenerated:\n\t{report}\n"

    with pytest.raises(RuntimeError, match="temporary"):
        runner._validate_nsys_output_paths(output, out, report)


def test_nsys_output_paths_require_exactly_one_expected_report(tmp_path):
    runner = _runner_module()
    out = tmp_path / "result"
    out.mkdir()
    temporary = out / "tool-state/tmp/nsys-report-abcd.qdstrm"
    expected = out / "capture/a100_environment_probe.nsys-rep"
    unexpected = out / "capture/unexpected.nsys-rep"

    with pytest.raises(RuntimeError, match="unexpected final report"):
        runner._validate_nsys_output_paths(
            f"Generating '{temporary}'\nGenerated:\n\t{unexpected}\n",
            out,
            expected,
        )

    with pytest.raises(RuntimeError, match="exactly one final report"):
        runner._validate_nsys_output_paths(
            f"Generating '{temporary}'\n", out, expected
        )

    with pytest.raises(RuntimeError, match="nonabsolute final report"):
        runner._validate_nsys_output_paths(
            f"Generating '{temporary}'\nGenerated:\n\trelative.nsys-rep\n",
            out,
            expected,
        )


def test_scratch_budget_and_manifest_include_tool_temporary_files(
    monkeypatch, tmp_path
):
    runner = _runner_module()
    scratch = tmp_path / "scratch"
    out = scratch / "result"
    temporary = out / "tool-state/tmp/retained.qdstrm"
    source = scratch / "source/source.bin"
    temporary.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    temporary.write_bytes(b"temp")
    source.write_bytes(b"source")

    assert runner._assert_scratch_budget(scratch) >= 10
    assert "tool-state/tmp/retained.qdstrm" in {
        row["path"] for row in runner._artifact_manifest(out)
    }

    monkeypatch.setattr(runner, "MAX_SCRATCH_BYTES", 5)
    with pytest.raises(RuntimeError, match="scratch exceeds"):
        runner._assert_scratch_budget(scratch)


def test_mig_state_uses_portable_full_query(monkeypatch):
    runner = _runner_module()
    calls = []

    def fake_run(command, **_kwargs):
        normalized = tuple(str(item) for item in command)
        calls.append(normalized)
        return subprocess.CompletedProcess(
            normalized,
            0,
            stdout=(
                "GPU 00000000:00:00.0\n"
                "    Unrelated Mode\n"
                "        Current                  : Enabled\n"
                "    MIG Mode\n"
                "        Current                  : Disabled\n"
                "        Pending                  : Disabled\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(runner, "_run", fake_run)

    assert runner._mig_state(Path("nvidia-smi"), "GPU-acde") == {
        "current": "Disabled",
        "pending": "Disabled",
    }
    assert calls == [("nvidia-smi", "--id=GPU-acde", "-q")]


def test_mig_state_refuses_missing_mode_section(monkeypatch):
    runner = _runner_module()

    monkeypatch.setattr(
        runner,
        "_run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, stdout="GPU query without MIG state\n", stderr=""
        ),
    )

    with pytest.raises(RuntimeError, match="no complete MIG mode section"):
        runner._mig_state(Path("nvidia-smi"), "GPU-acde")


@pytest.mark.parametrize("current", ["Enabled", "N/A"])
def test_mig_state_requires_disabled_current_mode(monkeypatch, current):
    runner = _runner_module()

    monkeypatch.setattr(
        runner,
        "_run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                f"    MIG Mode\n        Current : {current}\n"
                "        Pending : Disabled\n"
            ),
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="MIG is not stably disabled"):
        runner._mig_state(Path("nvidia-smi"), "GPU-acde")


@pytest.mark.parametrize("pending", [None, "Enabled", "N/A"])
def test_mig_state_requires_disabled_pending_mode(monkeypatch, pending):
    runner = _runner_module()
    pending_line = "" if pending is None else f"        Pending : {pending}\n"

    monkeypatch.setattr(
        runner,
        "_run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=f"    MIG Mode\n        Current : Disabled\n{pending_line}",
            stderr="",
        ),
    )

    expected = (
        "no complete MIG mode section"
        if pending is None
        else "MIG is not stably disabled"
    )
    with pytest.raises(RuntimeError, match=expected):
        runner._mig_state(Path("nvidia-smi"), "GPU-acde")


def test_scheduler_record_keeps_requested_and_allocated_tres():
    runner = _runner_module()
    output = (
        "JobId=24680 Account=merlin QOS=gpu_general JobState=RUNNING "
        "TimeLimit=00:20:00 Partition=a100-hourly NodeList=gpu101 "
        "NumNodes=1 NumCPUs=16 "
        "NumTasks=1 CPUs/Task=4 "
        "ReqTRES=cpu=4,mem=32G,node=1,billing=4,gres/gpu=1 "
        "AllocTRES=cpu=16,mem=32G,node=1,billing=16,gres/gpu=1,"
        "gres/gpu:nvidia_a100-sxm4-80gb=1 OverSubscribe=OK "
        "TresPerNode=gres/gpu:nvidia_a100-sxm4-80gb:1 TresPerTask=cpu=4\n"
    )

    record = runner._parse_scheduler_record(output, "24680")

    assert record["ReqTRES"].startswith("cpu=4,mem=32G")
    assert "gres/gpu:nvidia_a100-sxm4-80gb=1" in record["AllocTRES"]
    assert record["NumCPUs"] == "16"


@pytest.mark.parametrize(
    ("spelling", "seconds"),
    [("20:00", 1200), ("00:20:00", 1200), ("1-00:00:00", 86_400)],
)
def test_slurm_duration_normalization(spelling, seconds):
    runner = _runner_module()

    assert runner._slurm_duration_seconds(spelling) == seconds


def test_probe_output_requires_frozen_kernel_and_geometry():
    runner = _runner_module()
    output = "\n".join(
        [
            "probe=simllm-a100-environment-qualification-v1",
            f"target_kernel={runner.TARGET_KERNEL}",
            "device_name=NVIDIA A100-SXM4-80GB",
            "compute_capability=8.0",
            "element_count=16777216",
            "threads_per_block=256",
            "warmup_launches=5",
            "measured_launches=1",
            "output_checksum=0x1234",
            "correctness=PASS",
            "status=PASS",
        ]
    )

    assert runner._validate_probe_output(output)["output_checksum"] == "0x1234"


def test_ncu_metric_requires_target_and_finite_value():
    runner = _runner_module()
    header = (
        '"ID","Kernel Name","Metric Name","Metric Value",'
        '"Estimated Speedup"\n'
    )

    assert runner._has_numeric_ncu_metric(
        header + f'"0","{runner.TARGET_KERNEL}","metric","12.5","99"\n'
    )
    rejected_rows = [
        f'"0","{runner.TARGET_KERNEL}","metric","","99"\n',
        f'"0","{runner.TARGET_KERNEL}","metric","nan","99"\n',
        f'"0","{runner.TARGET_KERNEL}","metric","inf","99"\n',
        f'"0","prefix_{runner.TARGET_KERNEL}","metric","12.5","99"\n',
        '"0","unrelated_kernel","metric","12.5","99"\n',
        f'"0","{runner.TARGET_KERNEL}","","12.5","99"\n',
    ]
    for row in rejected_rows:
        assert not runner._has_numeric_ncu_metric(header + row)


def test_ncu_metric_requires_exactly_one_target_launch():
    runner = _runner_module()
    header = '"ID","Kernel Name","Metric Name","Metric Value"\n'
    output = header + "".join(
        [
            f'"0","{runner.TARGET_KERNEL}","duration","12.5"\n',
            f'"1","{runner.TARGET_KERNEL}","duration","12.7"\n',
        ]
    )

    assert not runner._has_numeric_ncu_metric(output)


def test_profiler_state_observations_bracket_profiler_commands():
    runner = _runner_module()
    source = inspect.getsource(runner._run_qualification)
    ordered_markers = (
        "unprofiled = _run(",
        "sass_run = _run(",
        "before = _gpu_snapshot(",
        "mig_before = _mig_state(",
        "supported_clocks_before, supported_clock_blocker_before = ",
        "processes_before = _foreign_processes(",
        "nsys_run = _run(",
        "ncu_run = _run(",
        "after = _gpu_snapshot(",
        "mig_after = _mig_state(",
        "supported_clocks_after, supported_clock_blocker_after = ",
        "processes_after = _foreign_processes(",
    )

    locations = [source.index(marker) for marker in ordered_markers]

    assert locations == sorted(locations)


def test_ncu_blocker_distinguishes_site_policy_from_tool_failure():
    runner = _runner_module()

    assert runner._ncu_capability_blocker(
        "==ERROR== ERR_NVGPUCTRPERM: permission denied"
    ) == "==ERROR== ERR_NVGPUCTRPERM: permission denied"
    assert runner._ncu_capability_blocker("==ERROR== unknown option") is None


def test_process_inventory_is_limited_to_allocated_gpu(monkeypatch):
    runner = _runner_module()
    calls = []

    def fake_run(command, **_kwargs):
        normalized = tuple(str(item) for item in command)
        calls.append(normalized)
        return subprocess.CompletedProcess(
            normalized,
            0,
            stdout="GPU-acde, 123, worker, 10\n",
            stderr="",
        )

    monkeypatch.setattr(runner, "_run", fake_run)

    assert runner._foreign_processes(Path("nvidia-smi"), "2", "GPU-acde") == [
        {
            "gpu_uuid": "GPU-acde",
            "pid": "123",
            "process_name": "worker",
            "used_gpu_memory": "10",
        }
    ]
    assert "--id=2" in calls[0]


def test_missing_clock_policy_becomes_capability_blocker():
    runner = _runner_module()
    snapshot = {
        "clocks.current.sm": "N/A",
        "clocks.current.memory": "1215",
        "clocks.max.sm": "1410",
        "clocks.max.memory": "1215",
        "power.limit": "400",
        "power.draw": "75",
        "temperature.gpu": "31",
        "persistence_mode": "Enabled",
        "compute_mode": "Default",
    }

    assert runner._telemetry_blockers(snapshot) == [
        "nvidia-smi did not expose a positive numeric clocks.current.sm"
    ]


def test_supported_clock_policy_is_scoped_and_nonempty(monkeypatch):
    runner = _runner_module()
    calls = []

    def fake_run(command, **_kwargs):
        normalized = tuple(str(item) for item in command)
        calls.append(normalized)
        return subprocess.CompletedProcess(
            normalized,
            0,
            stdout="1215, 1410\n1215, 1395\n",
            stderr="",
        )

    monkeypatch.setattr(runner, "_run", fake_run)

    clocks, blocker = runner._supported_clock_policy(Path("nvidia-smi"), "3")

    assert clocks == [
        {"memory_mhz": 1215, "graphics_mhz": 1410},
        {"memory_mhz": 1215, "graphics_mhz": 1395},
    ]
    assert blocker is None
    assert "--id=3" in calls[0]


def test_supported_clock_evidence_requires_stable_before_and_after_policy():
    runner = _runner_module()
    before = [
        {"memory_mhz": 1215, "graphics_mhz": 1410},
        {"memory_mhz": 1215, "graphics_mhz": 1395},
    ]

    assert runner._supported_clock_evidence_blockers(
        before, None, list(reversed(before)), None
    ) == []
    assert runner._supported_clock_evidence_blockers(
        before,
        "query denied",
        before,
        "query unavailable",
    ) == [
        "before profiling: query denied",
        "after profiling: query unavailable",
    ]
    assert runner._supported_clock_evidence_blockers(
        before,
        None,
        [{"memory_mhz": 1215, "graphics_mhz": 1410}],
        None,
    ) == ["supported clock policy changed during profiler probes"]


def test_tool_versions_require_frozen_cuda_and_nsight_identity():
    runner = _runner_module()
    versions = {
        "nvcc": "Cuda compilation tools, release 12.9, V12.9.86",
        "cuobjdump": "Cuda compilation tools, release 12.9, V12.9.86",
        "nsys": "NVIDIA Nsight Systems version 2025.1.3.120",
        "ncu": "NVIDIA Nsight Compute Version 2025.2.1.0",
    }

    runner._validate_tool_versions(versions)
    versions["ncu"] = "NVIDIA Nsight Compute Version 2025.1.0"
    with pytest.raises(RuntimeError, match="ncu identity"):
        runner._validate_tool_versions(versions)
