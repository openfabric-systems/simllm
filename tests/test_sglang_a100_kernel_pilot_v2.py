"""CPU-only regression checks for the frozen SGLang A100 CUDA 12.9 pilot."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY / "examples/sglang_a100_kernel_pilot_v2"
RUNNER_PATH = STUDY_DIR / "run_study.py"
V1_EXPECTATIONS_PATH = "examples/sglang_a100_kernel_pilot_v1/expectations.json"
V2_EXPECTATIONS_PATH = "examples/sglang_a100_kernel_pilot_v2/expectations.json"
V2_EXPECTATIONS_MARKDOWN_PATH = "examples/sglang_a100_kernel_pilot_v2/expectations.md"

FREEZE_COMMIT = "dee129923883d6b8cc394933b4944a4ff50193e5"
FREEZE_PARENT = "43064d6ae88d6380c86bd400d336a07aa504ccbd"
EXPECTATIONS_JSON_SHA256 = "f22ddd9a6a7223f3c2a7a836c0fd29d5a0ee66282212413c6ea5ddac22308894"
EXPECTATIONS_MARKDOWN_SHA256 = "6175ee55323a89ac027fdf4d81acb26dc4b49f8c2ae7569dbc95cc9410b563ed"


def _runner_module():
    spec = importlib.util.spec_from_file_location("sglang_a100_kernel_pilot_v2", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expectations():
    return json.loads((STUDY_DIR / "expectations.json").read_text(encoding="utf-8"))


def _git_show(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    ).stdout


def _path_environment(tmp_path: Path) -> dict[str, str]:
    del tmp_path
    runner = _runner_module()
    values = {}
    for name in runner.PATH_ENVIRONMENT_ROLES:
        values[name] = f"/opt/simllm/test-roles/{name.lower()}"
    values["PYTHONPATH"] = "/opt/simllm/test-source/python"
    return values


def _bind_rows(tmp_path: Path) -> list[dict[str, str]]:
    del tmp_path
    values = []
    modes = {
        "runner_projection": "ro",
        "source_projection": "ro",
        "model_projection": "ro",
        "nsys_install": "ro",
        "ncu_install": "ro",
        "result_and_cache_roots": "rw",
    }
    for index, (role, mode) in enumerate(modes.items()):
        values.append(
            {
                "role": role,
                "source": f"/opt/simllm-host/{index}",
                "destination": f"/simllm/{role}",
                "mode": mode,
            }
        )
    return values


def _source_fixture() -> tuple[dict[str, object], list[dict[str, object]], dict[str, list]]:
    source_root = "/simllm/source"
    package_root = f"{source_root}/python/sglang"
    files = [
        {
            "path": "python/sglang/__init__.py",
            "sha256": "a" * 64,
            "authority": "git_blob",
        },
        {
            "path": "python/sglang/runtime.py",
            "sha256": "b" * 64,
            "authority": "git_blob",
        },
        {
            "path": "python/sglang/_version.py",
            "sha256": "315a0924e5dde6902935235d5308bac9d76ae0b8ef44b4e2730891dba90fcceb",
            "authority": "generated_version",
        },
    ]
    manifest = {
        "root": source_root,
        "commit": "8f2a3ad6d7d68c58ae65b61a75bb2115449addca",
        "tree": "5be26db1f559064c0f9e724e78c1a8f619754867",
        "package_paths": [package_root],
        "files": files,
    }
    modules = [
        {
            "name": "sglang",
            "path": f"{package_root}/__init__.py",
            "sha256": "a" * 64,
            "package_paths": [package_root],
        },
        {
            "name": "sglang.runtime",
            "path": f"{package_root}/runtime.py",
            "sha256": "b" * 64,
        },
        {
            "name": "sglang._version",
            "path": f"{package_root}/_version.py",
            "sha256": "315a0924e5dde6902935235d5308bac9d76ae0b8ef44b4e2730891dba90fcceb",
        },
    ]
    entry_points = {"sglang.srt.platforms": [], "sglang.srt.plugins": []}
    return manifest, modules, entry_points


def _runtime_manifest(runner) -> dict[str, object]:
    layers = [
        {"digest": f"sha256:{index:064x}", "size": 1}
        for index in range(runner.EXPECTED_OCI_LAYER_COUNT - 1)
    ]
    layers.append(
        {
            "digest": f"sha256:{runner.EXPECTED_OCI_LAYER_COUNT - 1:064x}",
            "size": runner.EXPECTED_OCI_LAYER_BYTES - len(layers),
        }
    )
    return {
        "schema": "simllm-sglang-a100-environment-v2",
        "oci": {
            "manifest_digest": runner.EXPECTED_OCI_MANIFEST_DIGEST,
            "config_digest": runner.EXPECTED_OCI_CONFIG_DIGEST,
            "index_sha256": runner.EXPECTED_OCI_INDEX_SHA256,
            "index_bytes": runner.EXPECTED_OCI_INDEX_BYTES,
            "config_environment": ["PATH=/usr/bin"],
            "layers": layers,
        },
        "python": {"abi": "3.12", "executable": "/usr/bin/python3.12"},
        "distributions": [
            {"name": "torch", "version": "2.11.0+cu129", "direct_url": ""},
            {"name": "sglang", "version": "0.5.17", "direct_url": ""},
            {"name": "sglang-kernel", "version": "0.4.5+cu129", "direct_url": ""},
            {"name": "transformers", "version": "5.12.1", "direct_url": ""},
            {"name": "triton", "version": "3.6.0", "direct_url": ""},
        ],
        "critical_files": [
            {
                "path": "/opt/runtime/lib/libcudart.so.12",
                "sha256": "a" * 64,
                "authority": "oci",
                "device": 1,
                "inode": 1,
                "mtime_ns": 1,
                "ctime_ns": 1,
            }
        ],
        "source_projection": {
            "root": "/opt/simllm/source",
            "commit": runner.EXPECTED_SGLANG_COMMIT,
            "files": [],
        },
        "profilers": {
            "nsys": [
                {
                    "path": "/opt/nsys/bin/nsys",
                    "sha256": "b" * 64,
                    "authority": "nsys",
                    "kind": "launcher",
                    "device": 1,
                    "inode": 1,
                    "mtime_ns": 1,
                    "ctime_ns": 1,
                }
            ],
            "ncu": [
                {
                    "path": "/opt/ncu/bin/ncu",
                    "sha256": "c" * 64,
                    "authority": "ncu",
                    "kind": "launcher",
                    "device": 1,
                    "inode": 1,
                    "mtime_ns": 1,
                    "ctime_ns": 1,
                }
            ],
        },
        "environment": {
            "fixed_overrides": dict(runner.FIXED_CHILD_ENVIRONMENT),
            "path_overrides": {name: f"/opt/roles/{name.lower()}" for name in runner.PATH_ENVIRONMENT_ROLES},
            "required_unset": list(runner.REQUIRED_UNSET),
            "oci_config_environment": ["PATH=/usr/bin"],
            "observed_residual": {"PATH": "/usr/bin"},
        },
        "mount_policy": {
            "cleanenv": True,
            "contain": True,
            "no_home": True,
            "nv": True,
            "disabled_default_mounts": ["bind-paths", "cwd", "hostfs"],
            "optional_nvidia_file_mounts": sorted(
                runner.OPTIONAL_NVIDIA_FILE_MOUNT_POINTS
            ),
            "optional_nvidia_ipc_mounts": sorted(
                runner.OPTIONAL_NVIDIA_IPC_MOUNT_POINTS
            ),
        },
        "mount_contract": [
            {
                "mount_point": mount_point,
                "mount_options": [mode],
                "filesystem_type": "ext4",
                "super_options": [mode],
            }
            for mount_point, mode in runner.REQUIRED_CONTAINER_MOUNT_MODES.items()
        ],
    }


def _valid_lanes() -> dict[str, dict[str, str]]:
    return {
        "timing:prefill-t512-r4": {"state": "VALID"},
        "timing:decode-b4-c2048": {"state": "VALID"},
        "nsys:prefill-t512-r4": {"state": "VALID"},
        "nsys:decode-b4-c2048": {"state": "VALID"},
        "ncu:prefill-t512-r4": {"state": "VALID"},
        "ncu:decode-b4-c2048": {"state": "VALID"},
    }


def _mountinfo(
    *,
    writable_role: str | None = None,
    unexpected: str | None = None,
    additional: tuple[tuple[str, str], ...] = (),
) -> str:
    mounts = (
        ("runner", "/host/runner", "ro"),
        ("source", "/host/source", "ro"),
        ("model", "/host/model", "ro"),
        ("nsys", "/host/nsys", "ro"),
        ("ncu", "/host/ncu", "ro"),
        ("job", "/host/job", "rw"),
    )
    lines = []
    for index, (role, source, expected_mode) in enumerate(mounts, start=30):
        mode = "rw" if role == writable_role else expected_mode
        lines.append(
            f"{index} 20 0:{index} / /opt/simllm/{role} {mode},relatime "
            f"- ext4 {source} {mode},relatime"
        )
    if unexpected is not None:
        lines.append(
            f"99 20 0:99 / {unexpected} ro,relatime - ext4 /host/unexpected ro,relatime"
        )
    for index, (mount_point, mode) in enumerate(additional, start=100):
        lines.append(
            f"{index} 20 0:{index} / {mount_point} {mode},nosuid,nodev,relatime "
            f"- ext4 /host/nvidia/{index} rw,relatime"
        )
    return "\n".join(lines) + "\n"


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
        (
            "110,50,12,3,1,1,32,1,1,18,0,16,,NVIDIA A100-SXM4-80GB (0),"
            "1,8,attention_kernel\n"
        ),
        "170,5,13,,,,,,,,,,4096,NVIDIA A100-SXM4-80GB (0),1,7,[CUDA memcpy HtoD]\n",
        "180,4,14,,,,,,,,,,1024,NVIDIA A100-SXM4-80GB (0),1,7,[CUDA memset]\n",
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
            {"device_ms": 0.01, "host_ms": 0.02, "repetition": index}
            for index in range(5)
        ]
    }
    return rows, ranges, child


def test_check_only_writes_nothing_and_cannot_import_or_execute_runtime(tmp_path):
    blockers = tmp_path / "blockers"
    blockers.mkdir()
    for package in ("sglang", "torch"):
        package_dir = blockers / package
        package_dir.mkdir()
        (package_dir / "__init__.py").write_text(
            f"raise SystemExit('{package} imported during check-only')\n",
            encoding="utf-8",
        )
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_apptainer = fake_bin / "apptainer"
    fake_apptainer.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
    fake_apptainer.chmod(0o755)
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
    environment["PYTHONPATH"] = os.pathsep.join((str(blockers), environment.get("PYTHONPATH", "")))
    environment["PATH"] = os.pathsep.join((str(fake_bin), environment.get("PATH", "")))

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


def test_freeze_commit_has_exact_parent_paths_and_raw_hashes():
    parent = subprocess.run(
        ["git", "rev-parse", f"{FREEZE_COMMIT}^"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    paths = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", FREEZE_COMMIT],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    assert parent == FREEZE_PARENT
    assert paths == [V2_EXPECTATIONS_PATH, V2_EXPECTATIONS_MARKDOWN_PATH]
    assert hashlib.sha256(_git_show(FREEZE_COMMIT, V2_EXPECTATIONS_PATH)).hexdigest() == (
        EXPECTATIONS_JSON_SHA256
    )
    assert hashlib.sha256(_git_show(FREEZE_COMMIT, V2_EXPECTATIONS_MARKDOWN_PATH)).hexdigest() == (
        EXPECTATIONS_MARKDOWN_SHA256
    )


def test_current_expectations_still_match_frozen_raw_files():
    assert hashlib.sha256(_git_show("HEAD", V2_EXPECTATIONS_PATH)).hexdigest() == (
        EXPECTATIONS_JSON_SHA256
    )
    assert hashlib.sha256(_git_show("HEAD", V2_EXPECTATIONS_MARKDOWN_PATH)).hexdigest() == (
        EXPECTATIONS_MARKDOWN_SHA256
    )


def test_recorded_v1_chronology_hashes_the_original_commit_blobs():
    chronology = _expectations()["chronology"]

    assert chronology["v1_expectations_commit"] == (
        "b825285f24024a4ae41453c418672c89f03379cd"
    )
    assert chronology["v1_result_commit"] == FREEZE_PARENT
    assert hashlib.sha256(
        _git_show(chronology["v1_expectations_commit"], V1_EXPECTATIONS_PATH)
    ).hexdigest() == chronology["v1_expectations_json_sha256"]
    assert hashlib.sha256(
        _git_show(
            chronology["v1_expectations_commit"],
            "examples/sglang_a100_kernel_pilot_v1/expectations.md",
        )
    ).hexdigest() == chronology["v1_expectations_markdown_sha256"]
    assert hashlib.sha256(
        _git_show(
            chronology["v1_result_commit"],
            "examples/sglang_a100_kernel_pilot_v1/RESULTS.md",
        )
    ).hexdigest() == chronology["v1_result_sha256"]


def test_v2_inherits_physics_workloads_and_target_identity_from_v1_result():
    before = json.loads(_git_show(FREEZE_PARENT, V1_EXPECTATIONS_PATH))
    frozen = json.loads(_git_show(FREEZE_COMMIT, V2_EXPECTATIONS_PATH))
    for section in (
        "compatibility",
        "model_geometry",
        "physical",
        "source_invocations",
        "token_formula",
        "transferred_vllm_bracket",
        "workloads",
    ):
        assert frozen[section] == before[section]
    for key in (
        "attention_backend",
        "compute_capability",
        "config_sha256",
        "driver",
        "dtype",
        "gpu_name",
        "model",
        "model_revision",
        "moe_a2a_backend",
        "moe_runner_backend",
        "parallelism",
        "sampling_backend",
        "sglang_commit",
        "sglang_tree",
        "weight_bytes",
        "weight_sha256",
    ):
        assert frozen["identity"][key] == before["identity"][key]


def test_frozen_v2_identity_allocation_and_measurement_are_exact():
    runner = _runner_module()
    frozen = runner._load_expectations()

    runner._validate_expectations(frozen)
    assert frozen == _expectations()
    assert runner.EXPECTED_EXPECTATIONS_COMMIT == FREEZE_COMMIT
    assert frozen["runtime"]["pytorch_cuda"] == "12.9"
    assert frozen["identity"]["cuda_module"] == "cuda/12.9.1"
    assert frozen["runtime"]["cuda_13_allowed"] is False
    assert frozen["runtime"]["python_abi"] == "3.12"
    assert frozen["runtime"]["pytorch"] == "2.11.0+cu129"
    assert frozen["runtime"]["sglang_kernel_variant"] == "cu129"
    assert runner.EXPECTED_RUNTIME_VERSIONS["sglang_kernel"] == "0.4.5+cu129"
    assert frozen["measurement"]["required_lanes"] == [
        "timing:prefill-t512-r4",
        "timing:decode-b4-c2048",
        "nsys:prefill-t512-r4",
        "nsys:decode-b4-c2048",
    ]
    assert frozen["measurement"]["optional_lanes"] == [
        "ncu:prefill-t512-r4",
        "ncu:decode-b4-c2048",
    ]
    assert frozen["allocation"]["scratch_free_bytes_min"] == 180 * 1024**3
    assert frozen["allocation"]["scratch_bytes_max"] == 160 * 1024**3
    assert frozen["allocation"]["retained_bytes_max"] == 4 * 1024**3


def test_repository_gate_requires_freeze_commit_as_ancestor(monkeypatch):
    runner = _runner_module()
    head = "f" * 40
    merge_base_calls = []

    def fake_git_output(*arguments, cwd=runner.REPOSITORY_ROOT):
        del cwd
        if arguments == ("rev-parse", "HEAD"):
            return head
        if arguments == ("status", "--porcelain", "--untracked-files=all"):
            return ""
        raise AssertionError(arguments)

    def fake_run(command, **kwargs):
        del kwargs
        merge_base_calls.append(tuple(str(item) for item in command))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner, "_git_output", fake_git_output)
    monkeypatch.setattr(runner, "_run", fake_run)

    runner._check_repository(head, require_clean=False)

    assert any(
        call[-3:] == ("--is-ancestor", FREEZE_COMMIT, head) for call in merge_base_calls
    )


@pytest.mark.parametrize(
    "name",
    [
        "APPTAINER_BIND",
        "APPTAINERENV_PATH",
        "SINGULARITY_BINDPATH",
        "SINGULARITYENV_LD_PRELOAD",
    ],
)
@pytest.mark.parametrize("value", ["", "/host/injection"])
def test_host_launcher_environment_rejects_control_prefix_by_presence(name, value):
    runner = _runner_module()

    with pytest.raises(RuntimeError, match="inherited|launcher|control"):
        runner._validate_host_launcher_environment({name: value, "PATH": "/usr/bin"})


def test_container_environment_is_a_closed_allowlist(tmp_path):
    runner = _runner_module()
    paths = _path_environment(tmp_path)

    result = runner._container_environment(paths, "0")

    expected = {**runner.FIXED_CHILD_ENVIRONMENT, **paths, "CUDA_VISIBLE_DEVICES": "0"}
    assert result == expected
    assert "HOME" not in result
    assert not set(runner.REQUIRED_UNSET) & set(result)
    assert not any(key.startswith(runner.REJECTED_LAUNCHER_PREFIXES) for key in result)


@pytest.mark.parametrize("visible", ["", "0,1", "GPU-a,GPU-b", "all"])
def test_container_environment_requires_exactly_one_visible_gpu(tmp_path, visible):
    runner = _runner_module()

    with pytest.raises((RuntimeError, ValueError), match="GPU|CUDA_VISIBLE_DEVICES"):
        runner._container_environment(_path_environment(tmp_path), visible)


@pytest.mark.parametrize("mutation", ["missing", "extra", "relative"])
def test_container_environment_rejects_path_inventory_drift(tmp_path, mutation):
    runner = _runner_module()
    paths = _path_environment(tmp_path)
    if mutation == "missing":
        paths.pop("TRITON_CACHE_DIR")
    elif mutation == "extra":
        paths["LD_LIBRARY_PATH"] = str(tmp_path)
    else:
        paths["TRITON_CACHE_DIR"] = "relative/cache"

    with pytest.raises((RuntimeError, ValueError), match="path|role|absolute|normalized"):
        runner._container_environment(paths, "0")


def test_child_environment_rejects_unforwarded_scheduler_variables(monkeypatch, tmp_path):
    runner = _runner_module()
    environment = {
        **runner.FIXED_CHILD_ENVIRONMENT,
        **_path_environment(tmp_path),
        "CUDA_VISIBLE_DEVICES": "0",
        "SLURM_JOB_ID": "unexpected",
    }
    monkeypatch.setattr(runner.os, "environ", environment)

    with pytest.raises(RuntimeError, match="scheduler variable is forbidden"):
        runner._validate_child_environment_contract("timing")


def test_container_command_has_exact_isolation_bind_and_environment_contract(tmp_path):
    runner = _runner_module()
    apptainer = PurePosixPath("/host/bin/apptainer")
    sandbox = PurePosixPath("/host/sandbox")
    child_python = "/usr/bin/python3.12"
    binds = _bind_rows(tmp_path)
    result_destination = next(
        PurePosixPath(row["destination"])
        for row in binds
        if row["role"] == "result_and_cache_roots"
    )
    cwd = result_destination / "work"
    environment = runner._container_environment(_path_environment(tmp_path), "0")

    command = runner._container_command(
        apptainer,
        sandbox,
        cwd,
        binds,
        environment,
        (child_python, "/simllm/runner/run_study.py", "--child", "timing"),
    )

    assert command[0] == str(apptainer)
    assert command.count("--cleanenv") == 1
    assert command.count("--contain") == 1
    assert command.count("--no-home") == 1
    assert command.count("--nv") == 1
    assert command[command.index("--no-mount") + 1] == "bind-paths,cwd,hostfs"
    assert command[command.index("--pwd") + 1] == str(cwd)
    assert str(sandbox) in command
    child_index = command.index(child_python)
    assert command[child_index:] == (
        child_python,
        "/simllm/runner/run_study.py",
        "--child",
        "timing",
    )
    assert not {"--writable", "--overlay", "--fakeroot", "--userns", "--home", "--env-file"} & set(
        command
    )
    bind_values = [command[index + 1] for index, item in enumerate(command) if item == "--bind"]
    assert bind_values == [
        f"{row['source']}:{row['destination']}:{row['mode']}"
        for row in sorted(binds, key=lambda row: row["role"])
    ]
    env_values = [command[index + 1] for index, item in enumerate(command) if item == "--env"]
    assert env_values == [f"{name}={environment[name]}" for name in sorted(environment)]


def test_child_command_uses_runner_path_within_full_repository_projection(tmp_path):
    runner = _runner_module()

    command = runner._child_command(
        "timing", "prefill-t512-r4", tmp_path / "timing.json"
    )

    assert command[:4] == (
        "/usr/bin/python3",
        "/opt/simllm/runner/examples/sglang_a100_kernel_pilot_v2/run_study.py",
        "--child",
        "timing",
    )


@pytest.mark.parametrize(
    ("mutation", "pattern"),
    [
        (lambda rows: rows.__setitem__(0, {**rows[0], "mode": "rw"}), "mode|read.only"),
        (
            lambda rows: rows.__setitem__(1, {**rows[1], "destination": rows[0]["destination"]}),
            "duplicate|destination|overlap",
        ),
        (
            lambda rows: rows.__setitem__(1, {**rows[1], "destination": "/simllm/../escape"}),
            "destination|escape|traversal",
        ),
        (
            lambda rows: rows.__setitem__(1, {**rows[1], "source": rows[1]["source"] + ":rw"}),
            "bind|source|unsafe",
        ),
        (
            lambda rows: rows.__setitem__(1, {**rows[1], "source": rows[1]["source"] + ",rw"}),
            "bind|source|unsafe",
        ),
    ],
)
def test_container_command_rejects_hostile_bind_rows(tmp_path, mutation, pattern):
    runner = _runner_module()
    binds = _bind_rows(tmp_path)
    mutation(binds)

    with pytest.raises((RuntimeError, ValueError), match=pattern):
        runner._container_command(
            PurePosixPath("/host/apptainer"),
            PurePosixPath("/host/sandbox"),
            PurePosixPath("/simllm/result_and_cache_roots/work"),
            binds,
            runner._container_environment(_path_environment(tmp_path), "0"),
            ("/usr/bin/python3.12", "/simllm/runner/run_study.py"),
        )


def test_clean_source_projection_accepts_only_manifested_sglang_modules():
    runner = _runner_module()
    manifest, modules, entry_points = _source_fixture()

    runner._validate_source_projection(manifest, modules, entry_points)


def test_safe_git_archive_accepts_frozen_in_tree_parent_symlinks(tmp_path):
    runner = _runner_module()
    archive = tmp_path / "source.tar"
    payload = b"format\n"
    with tarfile.open(archive, "w:") as handle:
        target = tarfile.TarInfo("python/kernels/aot/.clang-format")
        target.size = len(payload)
        handle.addfile(target, io.BytesIO(payload))
        link = tarfile.TarInfo("python/sglang/srt/mem_cache/cpp_radix_tree/.clang-format")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../../../kernels/aot/.clang-format"
        handle.addfile(link)

    destination = tmp_path / "projection"
    destination.mkdir()
    runner._safe_extract_git_archive(archive, destination)

    extracted = destination / link.name
    assert extracted.is_symlink()
    assert extracted.resolve() == (destination / target.name).resolve()


@pytest.mark.parametrize(
    ("name", "target"),
    [
        ("outside", "../outside"),
        ("nested/outside", "../../outside"),
        ("absolute", "/outside"),
    ],
)
def test_safe_git_archive_rejects_symlinks_that_escape_root(tmp_path, name, target):
    runner = _runner_module()
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w:") as handle:
        link = tarfile.TarInfo(name)
        link.type = tarfile.SYMTYPE
        link.linkname = target
        handle.addfile(link)

    destination = tmp_path / "projection"
    destination.mkdir()
    with pytest.raises(RuntimeError, match="archive link is unsafe"):
        runner._safe_extract_git_archive(archive, destination)


def test_oci_archive_names_only_layout_children_and_omits_root_member(
    monkeypatch, tmp_path
):
    runner = _runner_module()
    layout = tmp_path / "oci"
    layout.mkdir()
    archive = tmp_path / "runtime.oci.tar"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((tuple(str(item) for item in command), kwargs))
        archive.write_bytes(b"archive")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner, "_required_tool", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(runner, "EXPECTED_OCI_LAYER_BYTES", 1)

    runner._create_oci_archive(layout, archive)

    assert calls == [
        (
            (
                "/usr/bin/tar",
                "--format=posix",
                "-cf",
                str(archive),
                "-C",
                str(layout),
                "oci-layout",
                "index.json",
                "blobs",
            ),
            {"timeout": 600},
        )
    ]
    assert "." not in calls[0][0]
    assert "./" not in calls[0][0]


@pytest.mark.parametrize(
    ("mutator", "pattern"),
    [
        (
            lambda manifest, modules, points: modules[1].update(
                {"path": "/usr/local/lib/python3.12/site-packages/sglang/runtime.py"}
            ),
            "source|projection|origin",
        ),
        (
            lambda manifest, modules, points: modules[1].update({"sha256": "c" * 64}),
            "hash|manifest",
        ),
        (
            lambda manifest, modules, points: modules.append(
                {
                    "name": "sglang.untracked",
                    "path": "/simllm/source/python/sglang/untracked.py",
                    "sha256": "d" * 64,
                }
            ),
            "manifest|untracked|unknown",
        ),
        (
            lambda manifest, modules, points: points["sglang.srt.plugins"].append(
                {"name": "replacement", "value": "outside:hook"}
            ),
            "entry.point|plugin",
        ),
        (
            lambda manifest, modules, points: manifest["files"][-1].update(
                {"sha256": "e" * 64}
            ),
            "version|hash",
        ),
    ],
)
def test_clean_source_projection_rejects_overlay_plugin_or_hash_drift(mutator, pattern):
    runner = _runner_module()
    manifest, modules, entry_points = _source_fixture()
    mutator(manifest, modules, entry_points)

    with pytest.raises((RuntimeError, ValueError), match=pattern):
        runner._validate_source_projection(manifest, modules, entry_points)


def test_runtime_manifest_digest_normalizes_maps_and_set_like_inventories():
    runner = _runner_module()
    manifest = _runtime_manifest(runner)
    reordered = copy.deepcopy(manifest)
    reordered["distributions"].reverse()
    reordered["mount_contract"].reverse()
    reordered["environment"] = dict(reversed(list(reordered["environment"].items())))

    digest = runner._environment_manifest_digest(manifest)

    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert runner._environment_manifest_digest(reordered) == digest
    tampered = copy.deepcopy(manifest)
    tampered["python"]["abi"] = "3.13"
    assert runner._environment_manifest_digest(tampered) != digest
    reordered_layers = copy.deepcopy(manifest)
    reordered_layers["oci"]["layers"].reverse()
    assert runner._environment_manifest_digest(reordered_layers) != digest


def test_runtime_manifest_validates_frozen_oci_and_cuda129_substrate():
    runner = _runner_module()
    manifest = _runtime_manifest(runner)

    assert runner._validate_runtime_environment_manifest(manifest) is None
    assert re.fullmatch(r"[0-9a-f]{64}", runner._environment_manifest_digest(manifest))


@pytest.mark.parametrize(
    ("mutator", "pattern"),
    [
        (
            lambda value: value["environment"].pop("observed_residual"),
            "environment contract inventory",
        ),
        (
            lambda value: value["environment"].update({"host_environment": {}}),
            "environment contract inventory",
        ),
        (
            lambda value: value["environment"].update({"observed_residual": []}),
            "residual environment.*malformed",
        ),
        (
            lambda value: value["environment"].update({"oci_config_environment": {}}),
            "OCI config environment.*malformed",
        ),
    ],
)
def test_runtime_manifest_residual_environment_structure_is_closed(mutator, pattern):
    runner = _runner_module()
    manifest = _runtime_manifest(runner)
    mutator(manifest)

    with pytest.raises((RuntimeError, TypeError, ValueError), match=pattern):
        runner._validate_runtime_environment_manifest(manifest)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("no_home", False),
        ("nv", False),
        ("optional_nvidia_file_mounts", ["/usr/share/nvidia/unexpected.bin"]),
        ("optional_nvidia_ipc_mounts", ["/run/nvidia-persistenced/unexpected"]),
    ],
)
def test_runtime_manifest_rejects_mount_policy_drift(key, value):
    runner = _runner_module()
    manifest = _runtime_manifest(runner)
    manifest["mount_policy"][key] = value

    with pytest.raises(RuntimeError, match="manifest.*policy|optional NVIDIA"):
        runner._validate_runtime_environment_manifest(manifest)


@pytest.mark.parametrize("authority", ["nsys", "ncu"])
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong-kind", "bad-hash"])
def test_runtime_manifest_requires_one_hashed_profiler_launcher(authority, mutation):
    runner = _runner_module()
    manifest = _runtime_manifest(runner)
    launcher = manifest["profilers"][authority][0]
    if mutation == "missing":
        manifest["profilers"][authority].clear()
    elif mutation == "duplicate":
        manifest["profilers"][authority].append(dict(launcher))
    elif mutation == "wrong-kind":
        launcher["kind"] = "file"
    else:
        launcher["sha256"] = "not-a-hash"

    with pytest.raises(RuntimeError, match=f"{authority} profiler (manifest|launcher)"):
        runner._validate_runtime_environment_manifest(manifest)


@pytest.mark.parametrize(
    ("mutator", "pattern"),
    [
        (
            lambda value: value["oci"].update({"manifest_digest": "sha256:" + "0" * 64}),
            "OCI|manifest|digest",
        ),
        (
            lambda value: value["distributions"][0].update({"version": "2.11.0+cu130"}),
            "CUDA 13|cu13|torch",
        ),
        (
            lambda value: value["distributions"].append(
                {"name": "cuda-python", "version": "13.0.0", "direct_url": ""}
            ),
            "CUDA 13|cuda.python|cu13",
        ),
        (
            lambda value: value["distributions"].append(
                {
                    "name": "addon",
                    "version": "1.0",
                    "direct_url": "file:///wheelhouse/cu13/addon.whl",
                }
            ),
            "CUDA 13|cu13|provenance",
        ),
        (
            lambda value: value["distributions"].append(
                {"name": "Torch", "version": "2.11.0+cu129", "direct_url": ""}
            ),
            "duplicate|distribution|torch",
        ),
    ],
)
def test_runtime_manifest_rejects_oci_cuda13_or_distribution_drift(mutator, pattern):
    runner = _runner_module()
    manifest = _runtime_manifest(runner)
    mutator(manifest)

    with pytest.raises((RuntimeError, ValueError), match=pattern):
        runner._validate_runtime_environment_manifest(manifest)


def test_loaded_object_ledger_accepts_authorized_cuda12_driver_and_profiler_objects():
    runner = _runner_module()
    rows = [
        {
            "path": "/opt/runtime/lib/libcudart.so.12",
            "sha256": "a" * 64,
            "authority": "oci",
        },
        {
            "path": "/usr/lib64/libcuda.so.565.57.01",
            "sha256": "b" * 64,
            "authority": "driver",
        },
        {
            "path": "/opt/nsys/lib/libToolsInjection64.so",
            "sha256": "c" * 64,
            "authority": "nsys",
        },
    ]

    assert runner._validate_loaded_object_ledger(rows) is None


@pytest.mark.parametrize(
    ("row", "pattern"),
    [
        (
            {"path": "/opt/runtime/lib/libcudart.so.13", "sha256": "a" * 64, "authority": "oci"},
            "CUDA 13|libcudart",
        ),
        (
            {
                "path": "/opt/runtime/lib/libnvrtc.so.13.0",
                "sha256": "a" * 64,
                "authority": "oci",
            },
            "CUDA 13|libnvrtc",
        ),
        (
            {"path": "/opt/runtime/lib/extension.so", "sha256": "", "authority": "oci"},
            "hash|SHA-256",
        ),
        (
            {"path": "/tmp/injected.so", "sha256": "a" * 64, "authority": "unknown"},
            "authority|unknown",
        ),
        (
            {
                "path": "/opt/runtime/lib/deleted.so (deleted)",
                "sha256": "a" * 64,
                "authority": "oci",
            },
            "deleted|path",
        ),
    ],
)
def test_loaded_object_ledger_rejects_cuda13_unhashed_or_unowned_rows(row, pattern):
    runner = _runner_module()

    with pytest.raises((RuntimeError, ValueError), match=pattern):
        runner._validate_loaded_object_ledger([row])


def test_loaded_object_ledger_rejects_conflicting_duplicate_path():
    runner = _runner_module()
    rows = [
        {"path": "/opt/runtime/lib/a.so", "sha256": "a" * 64, "authority": "oci"},
        {"path": "/opt/runtime/lib/a.so", "sha256": "b" * 64, "authority": "driver"},
    ]

    with pytest.raises((RuntimeError, ValueError), match="duplicate|conflict|authority"):
        runner._validate_loaded_object_ledger(rows)


def test_immutable_file_ledger_rehashes_direct_and_mapped_files(tmp_path):
    runner = _runner_module()
    actual_root = tmp_path / "host-profiler"
    actual = actual_root / "lib/tool.so"
    actual.parent.mkdir(parents=True)
    actual.write_bytes(b"exact profiler bytes")
    row = {
        "path": "/opt/simllm/nsys/lib/tool.so",
        "sha256": hashlib.sha256(actual.read_bytes()).hexdigest(),
        "size": actual.stat().st_size,
        "authority": "nsys",
        "kind": "file",
        **runner._stat_identity(actual),
    }

    runner._revalidate_file_ledger(
        [row], manifest_root=PurePosixPath("/opt/simllm/nsys"), actual_root=actual_root
    )
    actual.write_bytes(b"changed profiler bytes")
    with pytest.raises(RuntimeError, match="immutable file (identity|content) changed"):
        runner._revalidate_file_ledger(
            [row], manifest_root=PurePosixPath("/opt/simllm/nsys"), actual_root=actual_root
        )


def test_immutable_file_ledger_rehashes_symlink_targets(tmp_path):
    runner = _runner_module()
    target = "../lib/tool"
    link = tmp_path / "launcher"
    link.symlink_to(target)
    row = {
        "path": str(link),
        "sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
        "size": len(target),
        "authority": "nsys",
        "kind": "symlink",
        "target": target,
        **runner._stat_identity(link, follow_symlinks=False),
    }

    runner._revalidate_file_ledger([row])
    link.unlink()
    link.symlink_to("../other/tool")
    with pytest.raises(RuntimeError, match="symlink target changed"):
        runner._revalidate_file_ledger([row])


def test_generated_cache_loaded_object_joins_stable_post_run_inventory(tmp_path):
    runner = _runner_module()
    cache_root = tmp_path / "triton"
    generated = cache_root / "generated/kernel.so"
    generated.parent.mkdir(parents=True)
    generated.write_bytes(b"compiled kernel")
    digest = hashlib.sha256(generated.read_bytes()).hexdigest()
    row = {
        "path": str(generated.resolve()),
        "sha256": digest,
        "size": generated.stat().st_size,
        "authority": "generated_cache",
    }
    roots = {"TRITON_CACHE_DIR": cache_root}
    inventories = {
        "TRITON_CACHE_DIR": [
            {"path": "generated/kernel.so", "sha256": digest, "size": generated.stat().st_size}
        ]
    }

    runner._validate_loaded_objects_against_manifest([row], _runtime_manifest(runner))
    assert runner._validate_generated_object_authority([row], roots, inventories) is None


def test_generated_cache_loaded_object_must_match_one_exact_cache_entry(tmp_path):
    runner = _runner_module()
    outer = tmp_path / "cache"
    inner = outer / "nested"
    generated = inner / "kernel.so"
    generated.parent.mkdir(parents=True)
    generated.write_bytes(b"compiled kernel")
    digest = hashlib.sha256(generated.read_bytes()).hexdigest()
    row = {
        "path": str(generated.resolve()),
        "sha256": digest,
        "size": generated.stat().st_size,
        "authority": "generated_cache",
    }

    with pytest.raises(RuntimeError, match="ambiguous cache authority"):
        runner._validate_generated_object_authority(
            [row],
            {"outer": outer, "inner": inner},
            {"outer": [], "inner": []},
        )
    with pytest.raises(RuntimeError, match="absent from stable cache inventory"):
        runner._validate_generated_object_authority(
            [row],
            {"inner": inner},
            {"inner": [{"path": "kernel.so", "sha256": "0" * 64}]},
        )


def test_effective_mount_inventory_enforces_read_only_and_read_write_roles(monkeypatch):
    runner = _runner_module()

    class MountInfo:
        def read_text(self, *, encoding):
            assert encoding == "utf-8"
            return _mountinfo()

    monkeypatch.setattr(runner, "Path", lambda value: MountInfo())
    monkeypatch.setattr(
        runner,
        "_load_authority_seed",
        lambda: {"mount_policy": {"forbidden_host_paths": []}},
    )

    rows = runner._effective_mount_inventory()

    modes = {row["mount_point"]: set(row["mount_options"]) for row in rows}
    assert "rw" in modes["/opt/simllm/job"]
    assert all(
        "ro" in modes[f"/opt/simllm/{role}"]
        for role in ("runner", "source", "model", "nsys", "ncu")
    )


def test_effective_mount_inventory_accepts_exact_read_only_nvidia_binds(monkeypatch):
    runner = _runner_module()
    mountinfo = _mountinfo(
        additional=tuple(
            (mount_point, "ro")
            for mount_point in sorted(runner.OPTIONAL_NVIDIA_MOUNT_POINTS)
        )
    )

    class MountInfo:
        def read_text(self, *, encoding):
            assert encoding == "utf-8"
            return mountinfo

    monkeypatch.setattr(runner, "Path", lambda value: MountInfo())
    monkeypatch.setattr(
        runner,
        "_load_authority_seed",
        lambda: {"mount_policy": {"forbidden_host_paths": []}},
    )

    rows = runner._effective_mount_inventory()

    observed = {row["mount_point"] for row in rows}
    assert runner.OPTIONAL_NVIDIA_MOUNT_POINTS <= observed


@pytest.mark.parametrize(
    "unexpected",
    [
        str(PurePosixPath("/", "home", "synthetic-user")),
        str(PurePosixPath("/", "data", "user", "synthetic-user")),
        "/usr/share/nvidia/nvoptix.bin.bak",
        "/usr/share/nvidia/nvoptix.bin/child",
        "/usr/bin/nvidia-backdoor",
        "/run/nvidiarogue",
        "/run/nvidia-persistenced/socket.backup",
        "/run/nvidia-persistenced/unexpected",
        "/var/run/nvidia-malicious",
    ],
)
def test_effective_mount_inventory_rejects_home_and_nvidia_neighbors(
    monkeypatch, unexpected
):
    runner = _runner_module()

    class MountInfo:
        def read_text(self, *, encoding):
            assert encoding == "utf-8"
            return _mountinfo(unexpected=unexpected)

    monkeypatch.setattr(runner, "Path", lambda value: MountInfo())
    monkeypatch.setattr(
        runner,
        "_load_authority_seed",
        lambda: {"mount_policy": {"forbidden_host_paths": []}},
    )

    with pytest.raises(RuntimeError, match="unexpected user binds"):
        runner._effective_mount_inventory()


@pytest.mark.parametrize(
    "mount_point",
    [
        "/usr/share/nvidia/nvoptix.bin",
        "/run/nvidia-persistenced/socket",
    ],
)
@pytest.mark.parametrize("mode", ["rw", "ro"])
def test_effective_mount_inventory_rejects_writable_or_duplicate_nvidia_bind(
    monkeypatch, mount_point, mode
):
    runner = _runner_module()
    additional = ((mount_point, mode),)
    pattern = "writable"
    if mode == "ro":
        additional *= 2
        pattern = "duplicated"

    class MountInfo:
        def read_text(self, *, encoding):
            assert encoding == "utf-8"
            return _mountinfo(additional=additional)

    monkeypatch.setattr(runner, "Path", lambda value: MountInfo())
    monkeypatch.setattr(
        runner,
        "_load_authority_seed",
        lambda: {"mount_policy": {"forbidden_host_paths": []}},
    )

    with pytest.raises(RuntimeError, match=pattern):
        runner._effective_mount_inventory()


@pytest.mark.parametrize(
    ("mountinfo", "pattern"),
    [
        (_mountinfo(writable_role="source"), "read-only bind is writable"),
        (_mountinfo(unexpected="/opt/simllm/rogue"), "unexpected user binds"),
        (_mountinfo(unexpected="/mnt/rogue"), "unexpected user binds"),
        (
            _mountinfo().replace(
                "/opt/simllm/job rw,relatime - ext4 /host/job rw,relatime",
                "/opt/simllm/job ro,relatime - ext4 /host/job rw,relatime",
            ),
            "job result/cache bind is not writable",
        ),
    ],
)
def test_effective_mount_inventory_rejects_writable_or_unexpected_bind(
    monkeypatch, mountinfo, pattern
):
    runner = _runner_module()

    class MountInfo:
        def read_text(self, *, encoding):
            assert encoding == "utf-8"
            return mountinfo

    monkeypatch.setattr(runner, "Path", lambda value: MountInfo())
    monkeypatch.setattr(
        runner,
        "_load_authority_seed",
        lambda: {"mount_policy": {"forbidden_host_paths": []}},
    )

    with pytest.raises(RuntimeError, match=pattern):
        runner._effective_mount_inventory()


def test_optional_ncu_blocked_preserves_valid_required_evidence():
    runner = _runner_module()
    lanes = _valid_lanes()
    lanes["ncu:prefill-t512-r4"] = {"state": "BLOCKED", "error": "counter denied"}
    lanes["ncu:decode-b4-c2048"] = {"state": "BLOCKED", "error": "counter denied"}

    assert runner._aggregate_lane_state(lanes) == "VALID"


def test_ncu_blocker_does_not_mask_generic_child_connection_failure():
    runner = _runner_module()

    assert runner._ncu_blocker("==ERROR== Failed to connect to process") is None
    assert runner._ncu_blocker("ERR_NVGPUCTRPERM: counter access denied") is not None


@pytest.mark.parametrize(
    ("lane", "state", "expected"),
    [
        ("timing:prefill-t512-r4", "BLOCKED", "BLOCKED"),
        ("nsys:decode-b4-c2048", "BLOCKED", "BLOCKED"),
        ("ncu:prefill-t512-r4", "VOID", "VOID"),
        ("ncu:decode-b4-c2048", "VOID", "VOID"),
    ],
)
def test_lane_aggregation_required_and_fatal_truth_table(lane, state, expected):
    runner = _runner_module()
    lanes = _valid_lanes()
    lanes[lane] = {"state": state}

    assert runner._aggregate_lane_state(lanes) == expected


@pytest.mark.parametrize("mutation", ["missing", "unknown", "invalid"])
def test_lane_aggregation_fails_closed_for_malformed_inventory(mutation):
    runner = _runner_module()
    lanes = _valid_lanes()
    if mutation == "missing":
        lanes.pop("timing:prefill-t512-r4")
    elif mutation == "unknown":
        lanes["timing:unknown"] = {"state": "VALID"}
    else:
        lanes["timing:prefill-t512-r4"] = {"state": "SKIPPED"}

    with pytest.raises(RuntimeError, match="lane|inventory|state"):
        runner._aggregate_lane_state(lanes)


def test_budget_boundaries_are_inclusive_and_overages_are_fatal(monkeypatch, tmp_path):
    runner = _runner_module()
    scratch = tmp_path / "scratch"
    output = scratch / "output"
    output.mkdir(parents=True)

    monkeypatch.setattr(runner, "_tree_size", lambda path: runner.MAX_OUTPUT_BYTES)
    assert runner._assert_output_budget(output) == runner.MAX_OUTPUT_BYTES
    monkeypatch.setattr(runner, "_tree_size", lambda path: runner.MAX_OUTPUT_BYTES + 1)
    with pytest.raises(RuntimeError, match="output.*exceeds|retained.*exceeds"):
        runner._assert_output_budget(output)

    monkeypatch.setattr(runner, "_tree_size", lambda path: runner.MAX_SCRATCH_BYTES)
    assert runner._assert_scratch_budget(scratch) == runner.MAX_SCRATCH_BYTES
    monkeypatch.setattr(runner, "_tree_size", lambda path: runner.MAX_SCRATCH_BYTES + 1)
    with pytest.raises(RuntimeError, match="scratch.*exceeds"):
        runner._assert_scratch_budget(scratch)


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


def test_nsys_and_nvtx_csv_schemas_preserve_activity_and_range_semantics():
    runner = _runner_module()
    rows = runner._parse_nsys_cuda_trace(_cuda_trace_csv())
    api_rows = runner._parse_nsys_cuda_api_trace(_cuda_api_trace_csv())
    ranges = runner._parse_nvtx_projection(_nvtx_projection_csv())

    assert [row["activity"] for row in rows] == ["kernel", "kernel", "memcpy", "memset"]
    kernels = [row for row in rows if row["activity"] == "kernel"]
    assert sum(row["duration_ns"] for row in kernels) == 80.0
    assert runner._interval_union_ns(kernels) == 60.0
    assert [row["correlation_id"] for row in api_rows] == [11, 12]
    assert ranges[0]["projected_start_ns"] == 100.0
    assert ranges[1]["parent_id"] == 100


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
    command = runner._ncu_command(
        Path("ncu"), target, ("python", "run_study.py", "--child", "ncu")
    )
    assert command[command.index("--kernel-name") + 1] == "regex:" + target["kernel_regex"]
    assert command[command.index("--launch-skip") + 1] == "1"
    assert command[command.index("--launch-count") + 1] == "1"
    assert command[command.index("--replay-mode") + 1] == "kernel"
    assert command[command.index("--clock-control") + 1] == "none"


def test_ncu_metric_ledger_applies_dram_serialization_floor():
    runner = _runner_module()
    output = (
        '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
        '"0","target_kernel","gpu__time_duration.sum","ns","10"\n'
        '"0","target_kernel","dram__bytes.sum","byte","2039"\n'
    )

    metrics = runner._parse_ncu_metrics(output, "target_kernel")
    assert runner._validate_ncu_physical_floor(metrics) == {
        "duration_ns": 10.0,
        "dram_bytes": 2039.0,
        "dram_peak_floor_ns": 1.0,
    }
    too_fast = copy.deepcopy(metrics)
    next(row for row in too_fast if row["metric_name"] == "dram__bytes.sum")[
        "metric_value"
    ] = 203_900
    with pytest.raises(RuntimeError, match="DRAM serialization floor"):
        runner._validate_ncu_physical_floor(too_fast)


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


def test_only_explicit_capability_failures_are_blocked():
    runner = _runner_module()

    assert runner._classify_failure(runner.CapabilityBlocked("counter denied")) == "BLOCKED"
    assert runner._classify_failure(RuntimeError("CUDA initialization error")) == "VOID"
    assert runner._classify_failure(ValueError("identity drift")) == "VOID"
