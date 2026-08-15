"""CPU-only regression checks for the frozen SGLang A100 kernel pilot."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY / "examples/sglang_a100_kernel_pilot_v1"
RUNNER_PATH = STUDY_DIR / "run_study.py"


def _runner_module():
    spec = importlib.util.spec_from_file_location("sglang_a100_kernel_pilot_v1", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expectations():
    return json.loads((STUDY_DIR / "expectations.json").read_text(encoding="utf-8"))


def _cuda_trace_csv(*, bad_duration: str | None = None) -> str:
    header = (
        "Start (ns),Duration (ns),CorrId,GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,"
        "Reg/Trd,StcSMem (byte),DymSMem (byte),Bytes (byte),Device,Ctx,Strm,Name\n"
    )
    rows = [
        (
            "100,30,11,2,1,1,64,1,1,24,0,0,,NVIDIA A100-SXM4-80GB (0),"
            "1,7,void moe::kernel<float>(float*)\n"
        ),
        ("110,50,12,3,1,1,32,1,1,18,0,16,,NVIDIA A100-SXM4-80GB (0),1,8,attention_kernel\n"),
        ("170,5,13,,,,,,,,,,4096,NVIDIA A100-SXM4-80GB (0),1,7,[CUDA memcpy HtoD]\n"),
        ("180,4,14,,,,,,,,,,1024,NVIDIA A100-SXM4-80GB (0),1,7,[CUDA memset]\n"),
    ]
    if bad_duration is not None:
        rows[0] = rows[0].replace("100,30,", f"100,{bad_duration},", 1)
    return "Processing report with cuda_gpu_trace.py...\n" + header + "".join(rows)


def _nvtx_projection_csv() -> str:
    return (
        "Name,Projected Start (ns),Projected Duration (ns),Orig Start (ns),"
        "Orig Duration (ns),Style,PID,TID,NumGPUOps,Lvl,NumChild,RangeId,"
        "ParentId,RangeStack\n"
        "simllm-pilot:decode-b4-c2048:step:00,100,10000,1000,900,PushPop,"
        "42,77,2,0,2,100,,step\n"
        "simllm-pilot:decode-b4-c2048:layer-0-fused-moe,3100,7000,1290,100,"
        "PushPop,42,77,1,1,0,300,100,step/moe\n"
    )


def _cuda_api_trace_csv() -> str:
    return (
        "Start (ns),Duration (ns),Name,Result,CorrID,PID,TID,T-Pri,Thread Name\n"
        "1100,10,cudaLaunchKernel,0,11,42,77,0,Main Thread\n"
        "1300,10,cudaLaunchKernel,0,12,42,77,0,Main Thread\n"
    )


def _joined_capture_fixture(runner, phase="decode-b4-c2048"):
    cuda_lines = [
        (
            "Start (ns),Duration (ns),CorrId,GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,"
            "Reg/Trd,StcSMem (byte),DymSMem (byte),Bytes (byte),Device,Ctx,Strm,Name"
        )
    ]
    api_lines = ["Start (ns),Duration (ns),Name,Result,CorrID,PID,TID,T-Pri,Thread Name"]
    nvtx_lines = [
        (
            "Name,Projected Start (ns),Projected Duration (ns),Orig Start (ns),"
            "Orig Duration (ns),Style,PID,TID,NumGPUOps,Lvl,NumChild,RangeId,"
            "ParentId,RangeStack"
        )
    ]
    for index in range(5):
        cpu_base = 10_000 * index
        device_base = 1_000_000 * index
        qkv_correlation = 10 * index + 1
        moe_correlation = 10 * index + 2
        cuda_lines.extend(
            (
                (
                    f"{device_base + 100},2000,{qkv_correlation},1,1,1,64,1,1,24,0,0,,"
                    "NVIDIA A100-SXM4-80GB (0),1,7,qkv_kernel"
                ),
                (
                    f"{device_base + 3100},7000,{moe_correlation},1,1,1,64,1,1,24,0,0,,"
                    "NVIDIA A100-SXM4-80GB (0),1,7,moe_kernel"
                ),
            )
        )
        api_lines.extend(
            (
                f"{cpu_base + 100},10,cudaLaunchKernel,0,{qkv_correlation},42,77,0,Main",
                f"{cpu_base + 300},10,cudaLaunchKernel,0,{moe_correlation},42,77,0,Main",
            )
        )
        step_id = 100 + index
        nvtx_lines.extend(
            (
                (
                    f"simllm-pilot:{phase}:step:{index:02d},{device_base + 100},10000,"
                    f"{cpu_base},900,PushPop,42,77,2,0,2,{step_id},,step"
                ),
                (
                    f"simllm-pilot:{phase}:layer-0-qkv,{device_base + 100},2000,"
                    f"{cpu_base + 90},100,PushPop,42,77,1,1,0,{200 + index},{step_id},"
                    "step/qkv"
                ),
                (
                    f"simllm-pilot:{phase}:layer-0-fused-moe,{device_base + 3100},7000,"
                    f"{cpu_base + 290},100,PushPop,42,77,1,1,0,{300 + index},{step_id},"
                    "step/moe"
                ),
            )
        )
    cuda_rows = runner._parse_nsys_cuda_trace("\n".join(cuda_lines) + "\n")
    api_rows = runner._parse_nsys_cuda_api_trace("\n".join(api_lines) + "\n")
    ranges = runner._parse_nvtx_projection("\n".join(nvtx_lines) + "\n")
    rows = runner._annotate_ranges(cuda_rows, ranges, api_rows, phase)
    child = {
        "measurements": [
            {"device_ms": 0.01, "host_ms": 0.02, "repetition": index} for index in range(5)
        ]
    }
    return rows, ranges, child


def test_check_only_writes_nothing_and_cannot_import_runtime_packages(tmp_path):
    blockers = tmp_path / "blockers"
    blockers.mkdir()
    for package in ("sglang", "torch"):
        package_dir = blockers / package
        package_dir.mkdir()
        (package_dir / "__init__.py").write_text(
            f"raise SystemExit('{package} imported during check-only')\n",
            encoding="utf-8",
        )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    output = scratch / "must-not-exist"
    sentinel = scratch / "sentinel"
    sentinel.write_text("unchanged", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in scratch.iterdir()}
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(blockers), environment.get("PYTHONPATH", "")))
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--expected-head",
            head,
            "--out",
            str(output),
            "--scratch-root",
            str(scratch),
            "--check-only",
        ],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CHECK_ONLY=PASS" in completed.stdout
    assert not output.exists()
    assert {path.name: path.read_bytes() for path in scratch.iterdir()} == before


def test_frozen_identities_shapes_and_token_formula_are_exact():
    runner = _runner_module()
    frozen = runner._load_expectations()

    runner._validate_expectations(frozen)
    assert frozen == _expectations()
    assert frozen["identity"]["sglang_commit"] == ("8f2a3ad6d7d68c58ae65b61a75bb2115449addca")
    assert frozen["identity"]["model_revision"] == ("ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445")
    assert frozen["measurement"] == {
        "capture_repetitions": 5,
        "ncu_set": "basic",
        "profiler_timeout_seconds": 300,
        "retained_timing_repetitions": 41,
        "step_timeout_seconds": 120,
        "warmups": 10,
    }
    assert frozen["workloads"]["prefill-t512-r4"]["required_extend_tokens"] == 512
    assert frozen["workloads"]["decode-b4-c2048"]["required_seq_lens"] == [
        2048,
        2048,
        2048,
        2048,
    ]
    for request in range(4):
        for position in (0, 1, 127, 2046):
            expected = 1 + ((173 + 257 * request + 31 * position) % 49_154)
            assert runner._token_id(request, position) == expected
            assert 1 <= expected <= 49_154


def test_model_revision_is_explicit_runtime_provenance():
    runner = _runner_module()

    assert (
        runner._validated_model_revision({"SIMLLM_MODEL_REVISION": runner.EXPECTED_MODEL_REVISION})
        == runner.EXPECTED_MODEL_REVISION
    )
    with pytest.raises(RuntimeError, match="model revision drifted"):
        runner._validated_model_revision({})


def test_child_audits_cache_config_state_and_data_roots():
    runner = _runner_module()

    assert {
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_STATE_HOME",
        "XDG_DATA_HOME",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "CUDA_CACHE_PATH",
        "HF_HOME",
        "TORCH_HOME",
    } == set(runner.AUDITED_CHILD_ROOTS)


def test_scheduler_records_site_qos_without_inventing_a_resource_constraint(monkeypatch):
    runner = _runner_module()
    record = (
        "JobId=42 Account=merlin QOS=gpu_general JobState=RUNNING TimeLimit=00:45:00 "
        "Partition=a100-hourly NodeList=node001 NumNodes=1 NumCPUs=16 NumTasks=1 "
        "CPUs/Task=8 ReqTRES=cpu=8,mem=64G,node=1,gres/gpu=1 "
        "AllocTRES=cpu=16,mem=64G,node=1,gres/gpu=1,"
        "gres/gpu:nvidia_a100-sxm4-80gb=1 OverSubscribe=NO"
    )
    monkeypatch.setattr(runner, "_required_tool", lambda name: Path(name))
    monkeypatch.setattr(
        runner,
        "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, record, ""),
    )

    snapshot = runner._scheduler_snapshot("42")

    assert snapshot["QOS"] == "gpu_general"
    assert snapshot["Partition"] == "a100-hourly"


def test_phase_step_watchdog_uses_frozen_two_minute_ceiling(monkeypatch):
    runner = _runner_module()
    created = []

    class FakeTimer:
        def __init__(self, interval, function, args=()):
            self.interval = interval
            self.function = function
            self.args = args
            self.daemon = False
            self.started = False
            self.cancelled = False

        def start(self):
            self.started = True

        def cancel(self):
            self.cancelled = True

        def join(self, timeout=None):
            assert timeout == 1

        def is_alive(self):
            return False

    def make_timer(interval, function, args=()):
        timer = FakeTimer(interval, function, args)
        created.append(timer)
        return timer

    monkeypatch.setattr(runner.threading, "Timer", make_timer)
    watchdog = runner._start_step_watchdog("prefill:timing:00")
    runner._cancel_step_watchdog(watchdog)

    assert len(created) == 1
    assert created[0].interval == 120
    assert created[0].function is runner._expire_phase_step
    assert created[0].args == ("prefill:timing:00",)
    assert created[0].daemon
    assert created[0].started
    assert created[0].cancelled


def test_child_progress_is_atomic_and_attached_to_blocked_lane(tmp_path):
    runner = _runner_module()
    child_output = tmp_path / "prefill.timing.json"

    runner._append_child_progress(
        child_output,
        mode="timing",
        phase="prefill-t512-r4",
        stage="model_load_started",
    )
    progress = runner._append_child_progress(
        child_output,
        mode="timing",
        phase="prefill-t512-r4",
        stage="warmups_started",
        count=10,
    )

    assert progress["schema"] == "simllm-sglang-a100-kernel-pilot-progress-v1"
    assert [row["stage"] for row in progress["history"]] == [
        "model_load_started",
        "warmups_started",
    ]
    assert progress["history"][-1]["count"] == 10
    progress_path = runner._child_progress_path(child_output)
    assert progress_path.is_file()
    assert not progress_path.with_name(progress_path.name + ".tmp").exists()

    lane = runner._lane_failure(runner.CapabilityBlocked("command timed out"), child_output)
    assert lane["state"] == "BLOCKED"
    assert lane["progress"] == progress


def test_child_rejects_escaping_output_before_writing_progress(tmp_path, monkeypatch):
    runner = _runner_module()
    scratch = tmp_path / "scratch"
    source = scratch / "source"
    temporary = scratch / "tmp"
    source.mkdir(parents=True)
    temporary.mkdir()
    monkeypatch.setenv("SIMLLM_SGLANG_ENABLE", "0")
    monkeypatch.setenv("SIMLLM_SGLANG_ORACLE_CAPTURE", "0")
    monkeypatch.setenv("SIMLLM_SGLANG_SOURCE", str(source))
    monkeypatch.setenv("SIMLLM_SCRATCH_ROOT", str(scratch))
    monkeypatch.setenv("TMPDIR", str(temporary))
    for index, name in enumerate(runner.AUDITED_CHILD_ROOTS):
        root = scratch / f"audited-{index}"
        root.mkdir()
        monkeypatch.setenv(name, str(root))

    child_output = tmp_path / "escape.json"
    args = runner.argparse.Namespace(
        child="timing",
        phase="prefill-t512-r4",
        child_out=child_output,
    )

    with pytest.raises(RuntimeError, match="child output escapes"):
        runner._run_child(args)

    assert not child_output.exists()
    assert not runner._child_progress_path(child_output).exists()


@pytest.mark.parametrize(("name", "expected"), [("EXTEND", "extend"), ("DECODE", "decode")])
def test_forward_mode_is_derived_from_live_batch(name, expected):
    runner = _runner_module()

    class Mode:
        def __init__(self, mode_name):
            self.name = mode_name

        def is_extend(self):
            return self.name == "EXTEND"

        def is_decode(self):
            return self.name == "DECODE"

    assert runner._forward_mode_label(Mode(name), expected) == expected
    with pytest.raises(RuntimeError, match="forward mode drifted"):
        runner._forward_mode_label(Mode("MIXED"), expected)


def test_cuda_end_event_precedes_device_settlement(monkeypatch):
    runner = _runner_module()
    operations = []

    class Event:
        def __init__(self, label):
            self.label = label

        def record(self):
            operations.append(f"{self.label}.record")

        def synchronize(self):
            operations.append(f"{self.label}.synchronize")

        def elapsed_time(self, other):
            assert self.label == "start"
            assert other.label == "end"
            return 1.0

    class Cuda:
        def __init__(self):
            self.events = iter((Event("start"), Event("end")))

        def Event(self, *, enable_timing):
            assert enable_timing
            return next(self.events)

        def synchronize(self):
            operations.append("cuda.synchronize")

    clock = iter((1_000_000, 3_000_000))
    monkeypatch.setattr(runner.time, "perf_counter_ns", lambda: next(clock))

    def target(instance):
        operations.append("target")
        return instance, {"live": True}

    output, workload, device_ms, host_ms = runner._time_cuda_target(Cuda(), target, "output")

    assert operations == [
        "start.record",
        "target",
        "end.record",
        "end.synchronize",
        "cuda.synchronize",
    ]
    assert output == "output"
    assert workload == {"live": True}
    assert device_ms == 1.0
    assert host_ms == 2.0


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("identity", "sglang_commit"), "0" * 40),
        (("identity", "pytorch_cuda"), "12.9"),
        (("measurement", "capture_repetitions"), 4),
        (("workloads", "prefill-t512-r4", "required_extend_tokens"), 511),
        (("workloads", "decode-b4-c2048", "required_seq_lens"), [2048] * 3),
        (("token_formula", "position_multiplier"), 30),
    ],
)
def test_expectation_validation_refuses_identity_or_geometry_drift(path, value):
    runner = _runner_module()
    frozen = copy.deepcopy(runner._load_expectations())
    cursor = frozen
    for component in path[:-1]:
        cursor = cursor[component]
    cursor[path[-1]] = value

    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        runner._validate_expectations(frozen)


def test_nsys_command_disables_event_trace_and_requests_exactly_five_ranges(tmp_path):
    runner = _runner_module()
    command = runner._nsys_profile_command(
        Path("nsys"), tmp_path / "capture", ("python", "run_study.py", "--child", "capture")
    )

    assert "--trace=cuda,nvtx" in command
    assert "--capture-range=cudaProfilerApi" in command
    assert "--capture-range-end=repeat:5" in command
    assert "--cuda-event-trace=false" in command
    assert command.count("--cuda-event-trace=false") == 1
    assert not any(argument.startswith("--cuda-event-trace=true") for argument in command)


def test_interval_union_counts_overlap_once_and_keeps_additive_work_separate():
    runner = _runner_module()
    rows = [
        {"start_ns": 100.0, "duration_ns": 30.0},
        {"start_ns": 110.0, "duration_ns": 50.0},
        {"start_ns": 160.0, "duration_ns": 5.0},
        {"start_ns": 200.0, "duration_ns": 10.0},
    ]

    assert runner._interval_union_ns(rows) == 75.0
    assert sum(row["duration_ns"] for row in rows) == 95.0


def test_nsys_csv_inventory_preserves_kernels_copies_memsets_and_conservation():
    runner = _runner_module()
    rows = runner._parse_nsys_cuda_trace(_cuda_trace_csv())

    assert [row["activity"] for row in rows] == [
        "kernel",
        "kernel",
        "memcpy",
        "memset",
    ]
    kernels = [row for row in rows if row["activity"] == "kernel"]
    assert len(kernels) == 2
    assert {row["stream_id"] for row in kernels} == {7, 8}
    assert sum(row["duration_ns"] for row in kernels) == 80.0
    assert runner._interval_union_ns(kernels) == 60.0
    assert len(rows) == sum(
        sum(row["activity"] == kind for row in rows) for kind in ("kernel", "memcpy", "memset")
    )


def test_official_nsys_api_and_nvtx_projection_schemas_are_parsed():
    runner = _runner_module()

    api_rows = runner._parse_nsys_cuda_api_trace(_cuda_api_trace_csv())
    ranges = runner._parse_nvtx_projection(_nvtx_projection_csv())

    assert [row["correlation_id"] for row in api_rows] == [11, 12]
    assert api_rows[0]["pid"] == 42
    assert api_rows[0]["tid"] == 77
    assert ranges[0]["projected_start_ns"] == 100.0
    assert ranges[0]["original_start_ns"] == 1000.0
    assert ranges[0]["num_gpu_ops"] == 2
    assert ranges[1]["parent_id"] == 100
    assert ranges[1]["range_stack"] == "step/moe"


def test_device_rows_join_via_cuda_api_and_conserve_every_nvtx_range():
    runner = _runner_module()
    rows, ranges, child = _joined_capture_fixture(runner)

    assert len(rows) == 10
    assert all(row["api"]["name"] == "cudaLaunchKernel" for row in rows)
    assert sum(row["semantic_family"] == "qkv_projection" for row in rows) == 5
    assert sum(row["semantic_family"] == "fused_moe" for row in rows) == 5
    summary = runner._capture_summary("decode-b4-c2048", child, rows, ranges)
    assert summary["kernel_count_min"] == 2
    assert summary["kernel_count_max"] == 2
    assert all(item["device_span_ns"] == 10_000.0 for item in summary["ranges"])
    assert all(item["kernel_busy_union_ns"] == 9_000.0 for item in summary["ranges"])
    assert all(item["exposed_activity_gap_ns"] == 1_000.0 for item in summary["ranges"])


def test_cuda_api_join_and_repeated_kernel_identity_guards_are_fatal():
    runner = _runner_module()
    rows, ranges, child = _joined_capture_fixture(runner)
    missing_api = [row["api"] for row in rows]
    assert missing_api

    cuda_rows = [dict(row) for row in rows]
    for row in cuda_rows:
        row.pop("api", None)
        row.pop("phase_range", None)
        row.pop("semantic_family", None)
        row.pop("inside_layer0_fused_moe", None)
        row.pop("inside_layer0_qkv", None)
    api_rows = []
    with pytest.raises(RuntimeError, match="CorrID"):
        runner._annotate_ranges(cuda_rows, ranges, api_rows, "decode-b4-c2048")

    rows[-1]["name"] = "alternate_moe_kernel"
    with pytest.raises(RuntimeError, match="identities"):
        runner._capture_summary("decode-b4-c2048", child, rows, ranges)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1"])
def test_nsys_csv_refuses_nonfinite_or_nonpositive_durations(value):
    runner = _runner_module()

    with pytest.raises((RuntimeError, ValueError), match="duration|finite|positive"):
        runner._parse_nsys_cuda_trace(_cuda_trace_csv(bad_duration=value))


def test_ncu_target_uses_escaped_anchored_name_and_global_same_name_ordinal():
    runner = _runner_module()
    kernel_name = "void moe::kernel<float>(float* input[3], int x+1?)"
    rows = [
        {"name": kernel_name, "inside_layer0_fused_moe": False},
        {"name": "unrelated", "inside_layer0_fused_moe": False},
        {"name": kernel_name, "inside_layer0_fused_moe": True},
        {"name": kernel_name, "inside_layer0_fused_moe": True},
    ]

    target = runner._ncu_target(rows)

    assert target["kernel_name"] == kernel_name
    assert target["launch_skip"] == 1
    assert target["kernel_regex"] == "^" + re.escape(kernel_name) + "$"
    assert re.fullmatch(target["kernel_regex"], kernel_name)
    assert re.fullmatch(target["kernel_regex"], "prefix" + kernel_name) is None

    command = runner._ncu_command(Path("ncu"), target, ("python", "run_study.py", "--child", "ncu"))
    kernel_option = command[command.index("--kernel-name") + 1]
    assert kernel_option == "regex:" + target["kernel_regex"]
    assert command[command.index("--launch-skip") + 1] == "1"
    assert command[command.index("--launch-count") + 1] == "1"
    assert command[command.index("--replay-mode") + 1] == "kernel"
    assert command[command.index("--clock-control") + 1] == "none"
    assert command[command.index("--print-metric-name") + 1] == "name"
    assert command[command.index("--print-units") + 1] == "base"
    assert command[command.index("--page") + 1] == "raw"


def test_artifact_manifest_hashes_regular_files_and_rejects_symlinks(tmp_path):
    runner = _runner_module()
    root = tmp_path / "result"
    root.mkdir()
    payload = root / "capture.csv"
    payload.write_bytes(b"payload")

    assert runner._artifact_manifest(root) == [
        {
            "path": "capture.csv",
            "sha256": "239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5",
            "size": 7,
        }
    ]
    link = root / "escape"
    link.symlink_to(tmp_path / "outside")
    with pytest.raises((RuntimeError, ValueError), match="symlink"):
        runner._assert_confined_tree(root)
    with pytest.raises((RuntimeError, ValueError), match="symlink"):
        runner._artifact_manifest(root)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unavailable")
def test_confined_tree_rejects_special_files(tmp_path):
    runner = _runner_module()
    root = tmp_path / "result"
    root.mkdir()
    os.mkfifo(root / "unexpected.fifo")

    with pytest.raises((RuntimeError, ValueError), match="special|regular"):
        runner._assert_confined_tree(root)
    with pytest.raises((RuntimeError, ValueError), match="special|regular"):
        runner._artifact_manifest(root)


def test_output_and_scratch_caps_are_hard_failures(monkeypatch, tmp_path):
    runner = _runner_module()
    scratch = tmp_path / "scratch"
    output = scratch / "result"
    output.mkdir(parents=True)
    (output / "four-bytes").write_bytes(b"1234")
    monkeypatch.setattr(runner, "MAX_OUTPUT_BYTES", 3)
    monkeypatch.setattr(runner, "MAX_SCRATCH_BYTES", 3)

    with pytest.raises(RuntimeError, match="output.*exceeds|retained.*exceeds"):
        runner._assert_output_budget(output)
    with pytest.raises(RuntimeError, match="scratch.*exceeds"):
        runner._assert_scratch_budget(scratch)


@pytest.mark.parametrize("value", ["", "nan", "inf", "-inf", "not-a-number"])
def test_ncu_metrics_require_finite_values(value):
    runner = _runner_module()
    output = (
        '"ID","Kernel Name","Metric Name","Metric Value"\n'
        f'"0","target_kernel","gpu__time_duration.sum","{value}"\n'
    )

    assert not runner._has_finite_ncu_metrics(output, "target_kernel")


def test_ncu_metrics_accept_one_exact_target_with_finite_values():
    runner = _runner_module()
    output = (
        '"ID","Kernel Name","Metric Name","Metric Value"\n'
        '"0","target_kernel","gpu__time_duration.sum","12.5"\n'
        '"0","target_kernel","dram__bytes.sum","1024"\n'
    )

    assert runner._has_finite_ncu_metrics(output, "target_kernel")
    assert not runner._has_finite_ncu_metrics(output, "target")


def test_ncu_metric_ledger_applies_dram_serialization_floor():
    runner = _runner_module()
    output = (
        '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
        '"0","target_kernel","gpu__time_duration.sum","ns","10"\n'
        '"0","target_kernel","dram__bytes.sum","byte","2039"\n'
    )

    metrics = runner._parse_ncu_metrics(output, "target_kernel")
    physical = runner._validate_ncu_physical_floor(metrics)
    assert physical == {
        "duration_ns": 10.0,
        "dram_bytes": 2039.0,
        "dram_peak_floor_ns": 1.0,
    }

    too_fast = copy.deepcopy(metrics)
    next(row for row in too_fast if row["metric_name"] == "dram__bytes.sum")["metric_value"] = (
        203_900
    )
    with pytest.raises(RuntimeError, match="DRAM serialization floor"):
        runner._validate_ncu_physical_floor(too_fast)

    without_bytes = [row for row in metrics if row["metric_name"] != "dram__bytes.sum"]
    with pytest.raises(runner.CapabilityBlocked, match="dram__bytes"):
        runner._validate_ncu_physical_floor(without_bytes)


def test_child_result_requires_exact_identity_determinism_and_pool_reset():
    runner = _runner_module()
    phase = "prefill-t512-r4"
    output_ids = [1, 2, 3, 4]
    checksum = hashlib.sha256(json.dumps(output_ids).encode("utf-8")).hexdigest()
    measurement = {
        "batch_size": 4,
        "device_ms": 2.0,
        "forward_mode": "extend",
        "host_ms": 2.1,
        "input_tokens_per_request": 128,
        "kv_slots_allocated": 512,
        "output_ids": output_ids,
        "output_sha256": checksum,
        "request_ids": [f"{phase}-r{index}" for index in range(4)],
        "request_pool_indices": [1, 2, 3, 4],
        "scheduled_tokens": 512,
        "seq_lens": [128] * 4,
    }
    value = {
        "schema": "simllm-sglang-a100-kernel-pilot-child-v1",
        "mode": "timing",
        "phase": phase,
        "versions": runner.EXPECTED_RUNTIME_VERSIONS,
        "source": {
            "commit": runner.EXPECTED_SGLANG_COMMIT,
            "tree": runner.EXPECTED_SGLANG_TREE,
            "package": "python/sglang/__init__.py",
            "one_batch": "python/sglang/benchmark/one_batch.py",
        },
        "backends": {
            "attention": "triton",
            "moe_a2a": "none",
            "moe_runner": "triton",
            "sampling": "pytorch",
        },
        "runner_identity": "sglang.srt.model_executor.model_runner.ModelRunner",
        "model_identity": "sglang.srt.models.granitemoe.GraniteMoeForCausalLM",
        "cuda_device_name": runner.EXPECTED_GPU_NAME,
        "cuda_device_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "measurements": [{**measurement, "repetition": index} for index in range(41)],
        "cache_inventory": {name: [] for name in runner.AUDITED_CHILD_ROOTS},
    }

    runner._validate_child_result(
        value, "timing", phase, "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    )
    drifted = copy.deepcopy(value)
    drifted["measurements"][-1]["output_ids"] = [1, 2, 3, 5]
    drifted["measurements"][-1]["output_sha256"] = hashlib.sha256(
        json.dumps([1, 2, 3, 5]).encode("utf-8")
    ).hexdigest()
    with pytest.raises(RuntimeError, match="deterministic"):
        runner._validate_child_result(
            drifted, "timing", phase, "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        )


def test_no_capture_compatibility_rejects_profiler_artifacts_or_enabled_env(tmp_path):
    runner = _runner_module()
    ideal = tmp_path / "ideal.json"
    ideal.write_bytes(b"frozen ideal")
    expected_sha256 = runner._sha256(ideal)
    output = tmp_path / "timing"
    output.mkdir()
    environment = {
        "SIMLLM_SGLANG_ENABLE": "0",
        "SIMLLM_SGLANG_ORACLE_CAPTURE": "0",
    }

    runner._validate_compatibility_control(output, environment, ideal, expected_sha256)
    (output / "unexpected.nsys-rep").write_bytes(b"profile")
    with pytest.raises((RuntimeError, ValueError), match="profiler"):
        runner._validate_compatibility_control(output, environment, ideal, expected_sha256)
    (output / "unexpected.nsys-rep").unlink()
    enabled = {**environment, "SIMLLM_SGLANG_ENABLE": "1"}
    with pytest.raises((RuntimeError, ValueError), match="SIMLLM_SGLANG_ENABLE"):
        runner._validate_compatibility_control(output, enabled, ideal, expected_sha256)


def test_only_explicit_capability_failures_are_blocked():
    runner = _runner_module()

    assert runner._classify_failure(runner.CapabilityBlocked("counter denied")) == ("BLOCKED")
    assert runner._classify_failure(RuntimeError("counter denied")) == "VOID"
    assert runner._classify_failure(ValueError("identity drift")) == "VOID"
