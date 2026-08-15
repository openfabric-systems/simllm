"""Run the frozen CUDA 12.9 single-A100 SGLang production-kernel pilot.

The parent process is a standard-library-only orchestrator. SGLang, PyTorch,
model loading, CUDA initialization, timing and profiling are isolated in child
processes that may run only inside the reviewed Slurm allocation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import posixpath
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTATIONS_PATH = Path(__file__).with_name("expectations.json")
IDEAL_ARTIFACT = REPOSITORY_ROOT / "examples/sglang_host_step_v1/results.json"

EXPECTED_SCHEMA = "simllm-sglang-a100-kernel-pilot-expectations-v2"
EXPECTED_EXPECTATIONS_COMMIT = "dee129923883d6b8cc394933b4944a4ff50193e5"
EXPECTED_EXPECTATIONS_SHA256 = "f22ddd9a6a7223f3c2a7a836c0fd29d5a0ee66282212413c6ea5ddac22308894"
EXPECTED_EXPECTATIONS_MARKDOWN_SHA256 = (
    "6175ee55323a89ac027fdf4d81acb26dc4b49f8c2ae7569dbc95cc9410b563ed"
)
EXPECTED_NORMALIZED_EXPECTATIONS_SHA256 = (
    "5f9d94f5c18e2ec5a6515685ef2410c699306ead51b425b13a94e44850915cbd"
)
EXPECTED_BASE = "43064d6ae88d6380c86bd400d336a07aa504ccbd"
EXPECTED_SGLANG_COMMIT = "8f2a3ad6d7d68c58ae65b61a75bb2115449addca"
EXPECTED_SGLANG_TREE = "5be26db1f559064c0f9e724e78c1a8f619754867"
EXPECTED_MODEL_REVISION = "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
EXPECTED_CONFIG_SHA256 = "ca4bb3a5c1bdef988ab413e0d731640446da65316e4ed16de3666cd96ecc3a0b"
EXPECTED_WEIGHT_SHA256 = "f7ae1cee56a9ea6c5360437b1c0407f8d84816b2cc75470f4e7e5236fa2a07dc"
EXPECTED_WEIGHT_BYTES = 2_669_283_096
EXPECTED_IDEAL_SHA256 = "c021c55274e691fe609720045eec4441a8bb4828d248ca02b8561b63e2fddaff"
EXPECTED_GPU_NAME = "NVIDIA A100-SXM4-80GB"
EXPECTED_COMPUTE_CAPABILITY = "8.0"
EXPECTED_DRIVER = "565.57.01"
EXPECTED_RUNTIME_VERSIONS = {
    "python_abi": "3.12",
    "torch": "2.11.0+cu129",
    "torch_cuda": "12.9",
    "sglang_source": "0.0.0.dev1+g8f2a3ad6d",
    "sglang_substrate": "0.5.17",
    "sglang_kernel": "0.4.5+cu129",
    "transformers": "5.12.1",
    "triton": "3.6.0",
}

EXPECTED_OCI_MANIFEST_DIGEST = (
    "sha256:bbe1ec8694ab6b0e6ea0a567c2d492ed7c9bed13cb33ffe356ade0aaa8343c9f"
)
EXPECTED_OCI_CONFIG_DIGEST = (
    "sha256:0dbc9c00134e6c42b7d2cf291023b74fba3be2eaebd7a274603c869fae57188e"
)
EXPECTED_OCI_INDEX_SHA256 = "508033e7f177c186de2ac6ebf7f19008d1bb6345f42b0efdc042996f2dd1e9cd"
EXPECTED_OCI_INDEX_BYTES = 407
EXPECTED_OCI_LAYER_COUNT = 69
EXPECTED_OCI_LAYER_BYTES = 18_690_951_253
EXPECTED_VERSION_FILE_SHA256 = (
    "315a0924e5dde6902935235d5308bac9d76ae0b8ef44b4e2730891dba90fcceb"
)
EXPECTED_SOURCE_VERSION = "0.0.0.dev1+g8f2a3ad6d"
EXPECTED_SUBSTRATE_SGLANG_COMMIT = "29481685462732237d80d86076d6563e1f658102"

WARMUPS = 10
TIMING_REPETITIONS = 41
CAPTURE_REPETITIONS = 5
PROFILER_TIMEOUT_SECONDS = 300
STEP_TIMEOUT_SECONDS = 120
GLOBAL_DEADLINE_SECONDS = 2400
MAX_OUTPUT_BYTES = 4 * 1024**3
MAX_SCRATCH_BYTES = 160 * 1024**3
MIN_SCRATCH_FREE_BYTES = 180 * 1024**3

PHASES = ("prefill-t512-r4", "decode-b4-c2048")
PROFILER_SUFFIXES = (".nsys-rep", ".ncu-rep", ".qdrep", ".qdstrm")
SCHEDULER_FIELDS = (
    "JobId",
    "Account",
    "QOS",
    "JobState",
    "TimeLimit",
    "Partition",
    "NodeList",
    "NumNodes",
    "NumCPUs",
    "NumTasks",
    "CPUs/Task",
    "ReqTRES",
    "AllocTRES",
    "OverSubscribe",
    "TresPerNode",
    "TresPerTask",
)
REQUIRED_SCHEDULER_FIELDS = (
    "JobId",
    "Account",
    "QOS",
    "JobState",
    "TimeLimit",
    "Partition",
    "NodeList",
    "NumNodes",
    "NumCPUs",
    "NumTasks",
    "CPUs/Task",
    "ReqTRES",
    "AllocTRES",
    "OverSubscribe",
)
GPU_QUERY_FIELDS = (
    "index",
    "name",
    "uuid",
    "pci.bus_id",
    "compute_cap",
    "memory.total",
    "driver_version",
    "persistence_mode",
    "compute_mode",
    "clocks.current.sm",
    "clocks.current.memory",
    "clocks.max.sm",
    "clocks.max.memory",
    "power.limit",
    "power.draw",
    "temperature.gpu",
)
AUDITED_CHILD_ROOTS = (
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "CUDA_CACHE_PATH",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_STATE_HOME",
    "XDG_DATA_HOME",
    "HF_HOME",
    "TORCH_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "SGLANG_CACHE_DIR",
)

REJECTED_LAUNCHER_PREFIXES = (
    "APPTAINER_",
    "APPTAINERENV_",
    "SINGULARITY_",
    "SINGULARITYENV_",
)
FORWARDED_SCHEDULER_VARIABLES = ("CUDA_VISIBLE_DEVICES",)
REQUIRED_UNSET = (
    "ALL_PROXY",
    "CONDA_PREFIX",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "PYTHONHOME",
    "VIRTUAL_ENV",
    "all_proxy",
    "http_proxy",
    "https_proxy",
)
FIXED_CHILD_ENVIRONMENT = {
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "MKL_NUM_THREADS": "8",
    "NUMEXPR_NUM_THREADS": "8",
    "OMP_NUM_THREADS": "8",
    "OPENBLAS_NUM_THREADS": "8",
    "PIP_NO_INDEX": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "SIMLLM_PILOT_CUDA_MODULE": "cuda/12.9.1",
    "SIMLLM_SGLANG_ENABLE": "0",
    "SIMLLM_SGLANG_ORACLE_CAPTURE": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
    "UV_OFFLINE": "1",
}
PATH_ENVIRONMENT_ROLES = (
    "CUDA_CACHE_PATH",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "PYTHONPATH",
    "SGLANG_CACHE_DIR",
    "SIMLLM_MODEL_SNAPSHOT",
    "SIMLLM_NCU_INSTALL",
    "SIMLLM_NSYS_INSTALL",
    "SIMLLM_RUN_ROOT",
    "SIMLLM_SCRATCH_ROOT",
    "SIMLLM_SGLANG_SOURCE",
    "TEMP",
    "TMP",
    "TMPDIR",
    "TORCH_EXTENSIONS_DIR",
    "TORCH_HOME",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
)
REQUIRED_CONTAINER_MOUNT_MODES = {
    "/opt/simllm/runner": "ro",
    "/opt/simllm/source": "ro",
    "/opt/simllm/model": "ro",
    "/opt/simllm/nsys": "ro",
    "/opt/simllm/ncu": "ro",
    "/opt/simllm/job": "rw",
}
ALLOWED_SYSTEM_MOUNT_POINTS = {
    "/",
    "/dev",
    "/etc/group",
    "/etc/hostname",
    "/etc/hosts",
    "/etc/localtime",
    "/etc/passwd",
    "/etc/resolv.conf",
    "/home",
    "/proc",
    "/run",
    "/sys",
    "/tmp",
    "/var/tmp",
}
ALLOWED_SYSTEM_MOUNT_PREFIXES = (
    "/.singularity.d/libs/",
    "/dev/",
    "/home/",
    "/proc/",
    "/run/nvidia",
    "/sys/",
    "/usr/bin/nvidia-",
    "/var/run/nvidia",
)
STAT_IDENTITY_FIELDS = ("device", "inode", "mtime_ns", "ctime_ns")
MOUNT_CONTRACT_OPTIONS = {"ro", "rw", "nosuid", "nodev", "noexec"}
FORBIDDEN_CUDA13_SONAMES = (
    "libcublas.so.13",
    "libcupti.so.13",
    "libcudart.so.13",
    "libcufft.so.13",
    "libcurand.so.13",
    "libcusolver.so.13",
    "libcusparse.so.13",
    "libnvJitLink.so.13",
    "libnvfatbin.so.13",
    "libnvrtc.so.13",
)
_DEADLINE: float | None = None


class CapabilityBlocked(RuntimeError):
    """A frozen site or tool capability is unavailable."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--child",
        choices=("manifest", "cuda-gate", "import-gate", "timing", "capture", "ncu"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--phase", choices=PHASES, help=argparse.SUPPRESS)
    parser.add_argument("--child-out", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _load_expectations() -> dict[str, Any]:
    return json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))


def _expect(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise RuntimeError(f"frozen {label} drifted: {value!r} != {expected!r}")


def _validate_expectations(value: Mapping[str, Any]) -> None:
    normalized = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(normalized).hexdigest() != EXPECTED_NORMALIZED_EXPECTATIONS_SHA256:
        raise RuntimeError("frozen expectations content drifted")
    _expect(value.get("schema"), EXPECTED_SCHEMA, "schema")
    identity = value["identity"]
    _expect(identity["simllm_base"], EXPECTED_BASE, "SimLLM base")
    _expect(identity["sglang_commit"], EXPECTED_SGLANG_COMMIT, "SGLang commit")
    _expect(identity["sglang_tree"], EXPECTED_SGLANG_TREE, "SGLang tree")
    _expect(identity["model_revision"], EXPECTED_MODEL_REVISION, "model revision")
    _expect(identity["config_sha256"], EXPECTED_CONFIG_SHA256, "config hash")
    _expect(identity["weight_sha256"], EXPECTED_WEIGHT_SHA256, "weight hash")
    _expect(identity["weight_bytes"], EXPECTED_WEIGHT_BYTES, "weight size")
    _expect(identity["cuda_module"], "cuda/12.9.1", "CUDA module")
    _expect(identity["sglang_source_version"], EXPECTED_SOURCE_VERSION, "source version")
    _expect(identity["attention_backend"], "triton", "attention backend")
    _expect(identity["moe_runner_backend"], "triton", "MoE backend")
    _expect(identity["sampling_backend"], "pytorch", "sampling backend")
    _expect(identity["parallelism"], {"dp": 1, "ep": 1, "pp": 1, "tp": 1}, "parallelism")
    _expect(
        value["measurement"],
        {
            "capture_repetitions": CAPTURE_REPETITIONS,
            "cuda_gate_timeout_seconds": 60,
            "ncu_set": "basic",
            "optional_lanes": ["ncu:prefill-t512-r4", "ncu:decode-b4-c2048"],
            "profiler_launch_authority": (
                "inside_containment_from_read_only_host_install"
            ),
            "profiler_timeout_seconds": PROFILER_TIMEOUT_SECONDS,
            "required_lanes": [
                "timing:prefill-t512-r4",
                "timing:decode-b4-c2048",
                "nsys:prefill-t512-r4",
                "nsys:decode-b4-c2048",
            ],
            "retained_timing_repetitions": TIMING_REPETITIONS,
            "runtime_import_timeout_seconds": 120,
            "step_timeout_seconds": STEP_TIMEOUT_SECONDS,
            "warmups": WARMUPS,
        },
        "measurement protocol",
    )
    _expect(value["token_formula"]["seed"], 173, "token seed")
    _expect(value["token_formula"]["request_multiplier"], 257, "request multiplier")
    _expect(value["token_formula"]["position_multiplier"], 31, "position multiplier")
    _expect(value["token_formula"]["modulus"], 49_154, "token modulus")
    _expect(value["token_formula"]["shift"], 1, "token shift")
    prefill = value["workloads"]["prefill-t512-r4"]
    _expect(prefill["requests"], 4, "prefill request count")
    _expect(prefill["input_tokens_per_request"], 128, "prefill input length")
    _expect(prefill["required_extend_tokens"], 512, "prefill token count")
    decode = value["workloads"]["decode-b4-c2048"]
    _expect(decode["batch"], 4, "decode batch")
    _expect(decode["input_tokens_per_request"], 2047, "decode input length")
    _expect(decode["required_seq_lens"], [2048] * 4, "decode sequence lengths")
    _expect(value["transferred_vllm_bracket"], [440, 567], "vLLM bracket")
    _expect(
        value["compatibility"]["ideal_sha256"],
        EXPECTED_IDEAL_SHA256,
        "ideal artifact hash",
    )
    runtime = value["runtime"]
    _expect(runtime["pytorch_cuda"], "12.9", "PyTorch CUDA build")
    _expect(
        EXPECTED_RUNTIME_VERSIONS["sglang_kernel"],
        f"{runtime['sglang_kernel_release']}+{runtime['sglang_kernel_variant']}",
        "SGLang kernel distribution version",
    )
    _expect(runtime["oci_manifest_digest"], EXPECTED_OCI_MANIFEST_DIGEST, "OCI manifest")
    _expect(runtime["oci_config_digest"], EXPECTED_OCI_CONFIG_DIGEST, "OCI config")
    _expect(runtime["index_sha256"], EXPECTED_OCI_INDEX_SHA256, "OCI index hash")
    _expect(runtime["index_bytes"], EXPECTED_OCI_INDEX_BYTES, "OCI index size")
    _expect(runtime["layer_count"], EXPECTED_OCI_LAYER_COUNT, "OCI layer count")
    _expect(runtime["layer_payload_bytes"], EXPECTED_OCI_LAYER_BYTES, "OCI layer bytes")
    _expect(runtime["version_file_sha256"], EXPECTED_VERSION_FILE_SHA256, "version file")
    _expect(runtime["cuda_13_allowed"], False, "CUDA 13 prohibition")
    _expect(runtime["platform_entry_points"], [], "platform entry points")
    _expect(runtime["plugin_entry_points"], [], "plugin entry points")
    allocation = value["allocation"]
    _expect(allocation["scratch_free_bytes_min"], MIN_SCRATCH_FREE_BYTES, "free scratch")
    _expect(allocation["scratch_bytes_max"], MAX_SCRATCH_BYTES, "scratch ceiling")
    _expect(allocation["retained_bytes_max"], MAX_OUTPUT_BYTES, "retained ceiling")
    execution = value["execution_environment"]
    _expect(execution["fixed_overrides"], FIXED_CHILD_ENVIRONMENT, "fixed environment")
    _expect(execution["path_overrides"], list(PATH_ENVIRONMENT_ROLES), "path environment")
    _expect(execution["required_unset"], list(REQUIRED_UNSET), "unset environment")
    _expect(
        execution["rejected_inherited_host_control_prefixes"],
        list(REJECTED_LAUNCHER_PREFIXES),
        "rejected launcher prefixes",
    )


def _token_id(request: int, position: int) -> int:
    if request < 0 or position < 0:
        raise ValueError("token coordinates must be nonnegative")
    return 1 + ((173 + 257 * request + 31 * position) % 49_154)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_file_sha256(commit: str, path: Path) -> str:
    relative = path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    completed = subprocess.run(
        ("git", "-C", str(REPOSITORY_ROOT), "show", f"{commit}:{relative}"),
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.decode("utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"cannot read tracked Git blob {commit}:{relative}: {diagnostic}")
    return hashlib.sha256(completed.stdout).hexdigest()


def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _environment_manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Return the canonical digest of an already normalized immutable manifest."""
    normalized = json.loads(json.dumps(manifest))
    for key in ("distributions", "critical_files", "mount_contract"):
        rows = normalized.get(key)
        if isinstance(rows, list):
            rows.sort(key=lambda row: _canonical_json(row))
    profilers = normalized.get("profilers")
    if isinstance(profilers, dict):
        for value in profilers.values():
            if isinstance(value, list):
                value.sort(key=lambda row: _canonical_json(row))
    return hashlib.sha256(_canonical_json(normalized)).hexdigest()


def _validate_host_launcher_environment(environment: Mapping[str, str]) -> None:
    rejected = sorted(
        name
        for name in environment
        if any(name.startswith(prefix) for prefix in REJECTED_LAUNCHER_PREFIXES)
    )
    if rejected:
        raise RuntimeError(f"inherited container launcher controls are forbidden: {rejected}")


def _container_environment(paths: Mapping[str, str | Path], visible_gpu: str) -> dict[str, str]:
    """Construct the complete experiment override environment from an allowlist."""
    if not visible_gpu or "," in visible_gpu or visible_gpu.lower() in {"all", "none"}:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must identify exactly one GPU")
    if set(paths) != set(PATH_ENVIRONMENT_ROLES):
        missing = sorted(set(PATH_ENVIRONMENT_ROLES) - set(paths))
        extra = sorted(set(paths) - set(PATH_ENVIRONMENT_ROLES))
        raise RuntimeError(f"container path roles drifted: missing={missing}, extra={extra}")
    normalized = {name: str(PurePosixPath(str(paths[name]))) for name in PATH_ENVIRONMENT_ROLES}
    for name, value in normalized.items():
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"container path for {name} is not absolute and normalized")
    return {
        **FIXED_CHILD_ENVIRONMENT,
        **normalized,
        "CUDA_VISIBLE_DEVICES": visible_gpu,
    }


def _validate_bind_rows(binds: Sequence[Mapping[str, Any]]) -> None:
    expected_modes = {
        "runner_projection": "ro",
        "source_projection": "ro",
        "model_projection": "ro",
        "nsys_install": "ro",
        "ncu_install": "ro",
        "result_and_cache_roots": "rw",
    }
    if len(binds) != len(expected_modes):
        raise RuntimeError("container bind inventory must contain exactly six roles")
    roles = [str(row.get("role", "")) for row in binds]
    if set(roles) != set(expected_modes) or len(roles) != len(set(roles)):
        raise RuntimeError(f"container bind roles drifted: {roles}")
    destinations: list[PurePosixPath] = []
    for row in binds:
        role = str(row["role"])
        source = PurePosixPath(str(row.get("source", "")))
        destination = PurePosixPath(str(row.get("destination", "")))
        mode = str(row.get("mode", ""))
        if not source.is_absolute() or not destination.is_absolute():
            raise RuntimeError(f"container bind {role} must use absolute paths")
        if ".." in source.parts or ".." in destination.parts:
            raise RuntimeError(f"container bind {role} contains parent traversal")
        if any(character in str(source) + str(destination) for character in (",", ":")):
            raise RuntimeError(f"container bind {role} contains an unsafe delimiter")
        if mode != expected_modes[role]:
            raise RuntimeError(f"container bind {role} mode drifted: {mode!r}")
        destinations.append(destination)
    if len(destinations) != len(set(destinations)):
        raise RuntimeError("container bind destinations must be unique")
    for left, right in itertools.combinations(destinations, 2):
        if left in right.parents or right in left.parents:
            raise RuntimeError("container bind destinations must not overlap")


def _container_command(
    apptainer: Path | PurePosixPath,
    sandbox: Path | PurePosixPath,
    cwd: Path | PurePosixPath,
    binds: Sequence[Mapping[str, Any]],
    environment: Mapping[str, str],
    child_argv: Sequence[str | Path],
) -> tuple[str, ...]:
    """Build a fail-closed contained invocation without consulting host state."""
    _validate_bind_rows(binds)
    apptainer_path = PurePosixPath(str(apptainer))
    sandbox_path = PurePosixPath(str(sandbox))
    cwd_path = PurePosixPath(str(cwd))
    if not apptainer_path.is_absolute() or not sandbox_path.is_absolute() or not cwd_path.is_absolute():
        raise RuntimeError("Apptainer, sandbox and container cwd paths must be absolute")
    if not child_argv or not PurePosixPath(str(child_argv[0])).is_absolute():
        raise RuntimeError("contained child executable must be absolute")
    if "HOME" in environment:
        raise RuntimeError("HOME must not be overridden")
    expected_environment = set(FIXED_CHILD_ENVIRONMENT) | set(PATH_ENVIRONMENT_ROLES) | {
        "CUDA_VISIBLE_DEVICES"
    }
    if set(environment) != expected_environment:
        raise RuntimeError("contained environment does not match the frozen allowlist")
    result_destination = next(
        PurePosixPath(str(row["destination"]))
        for row in binds
        if row["role"] == "result_and_cache_roots"
    )
    source_destination = next(
        PurePosixPath(str(row["destination"]))
        for row in binds
        if row["role"] == "source_projection"
    )
    if (
        cwd_path != result_destination / "work"
        or cwd_path == source_destination
        or cwd_path in source_destination.parents
    ):
        raise RuntimeError("contained working directory is outside the frozen job role")
    command: list[str] = [
        str(apptainer),
        "exec",
        "--cleanenv",
        "--contain",
        "--nv",
        "--no-mount",
        "bind-paths,cwd,hostfs",
        "--pwd",
        str(cwd),
    ]
    for row in sorted(binds, key=lambda item: str(item["role"])):
        command.extend(
            (
                "--bind",
                f"{row['source']}:{row['destination']}:{row['mode']}",
            )
        )
    for name in sorted(environment):
        value = environment[name]
        if "\n" in value or "\x00" in value:
            raise RuntimeError(f"contained environment value for {name} is unsafe")
        command.extend(("--env", f"{name}={value}"))
    command.append(str(sandbox))
    command.extend(("/usr/bin/env",))
    for name in REQUIRED_UNSET:
        command.extend(("-u", name))
    command.extend(str(item) for item in child_argv)
    forbidden = {"--writable", "--writable-tmpfs", "--overlay", "--fakeroot", "--userns"}
    if forbidden.intersection(command):
        raise RuntimeError("container command includes a forbidden mutability or privilege flag")
    return tuple(command)


def _aggregate_lane_state(lanes: Mapping[str, Mapping[str, Any]]) -> str:
    required = {
        f"timing:{phase}" for phase in PHASES
    } | {f"nsys:{phase}" for phase in PHASES}
    optional = {f"ncu:{phase}" for phase in PHASES}
    if set(lanes) != required | optional:
        raise RuntimeError("lane inventory is incomplete or unknown")
    states = {name: str(value.get("state", "")) for name, value in lanes.items()}
    if any(state not in {"VALID", "BLOCKED", "VOID"} for state in states.values()):
        raise RuntimeError("lane state is unknown")
    if any(state == "VOID" for state in states.values()):
        return "VOID"
    if any(states[name] != "VALID" for name in required):
        return "BLOCKED"
    return "VALID"


def _validate_loaded_object_ledger(rows: Sequence[Mapping[str, Any]]) -> None:
    seen: dict[str, tuple[str, str]] = {}
    allowed_authorities = {
        "oci",
        "driver",
        "nsys",
        "ncu",
        "source",
        "runner",
        "model",
        "generated_cache",
    }
    for row in rows:
        path = str(row.get("path", ""))
        digest = str(row.get("sha256", ""))
        authority = str(row.get("authority", ""))
        logical_path = PurePosixPath(path)
        if not path or path.endswith(" (deleted)") or not logical_path.is_absolute():
            raise RuntimeError("loaded object path is not absolute")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError("loaded object has no exact SHA-256")
        if authority not in allowed_authorities:
            raise RuntimeError(f"loaded object has unowned authority: {authority!r}")
        if path in seen:
            raise RuntimeError(
                "loaded object ledger contains duplicate or conflicting path authority"
            )
        seen[path] = (digest, authority)
        basename = logical_path.name
        if any(
            basename == soname or basename.startswith(soname + ".")
            for soname in FORBIDDEN_CUDA13_SONAMES
        ):
            raise RuntimeError(f"CUDA 13 loaded object is forbidden: {basename}")


def _validate_runtime_environment_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "oci",
        "distributions",
        "critical_files",
        "profilers",
        "python",
        "source_projection",
        "environment",
        "mount_policy",
        "mount_contract",
    }
    if set(manifest) != required:
        raise RuntimeError("immutable environment manifest inventory drifted")
    _expect(manifest["schema"], "simllm-sglang-a100-environment-v2", "manifest schema")
    oci = manifest["oci"]
    _expect(oci.get("manifest_digest"), EXPECTED_OCI_MANIFEST_DIGEST, "manifest OCI digest")
    _expect(oci.get("config_digest"), EXPECTED_OCI_CONFIG_DIGEST, "manifest OCI config")
    distributions = manifest["distributions"]
    if not isinstance(distributions, list) or not distributions:
        raise RuntimeError("distribution inventory is empty")
    normalized_names: set[str] = set()
    substrate_seen = False
    for row in distributions:
        name = re.sub(r"[-_.]+", "-", str(row.get("name", "")).lower())
        version = str(row.get("version", ""))
        direct_url = str(row.get("direct_url", ""))
        if not name or not version or name in normalized_names:
            raise RuntimeError("distribution inventory is malformed or duplicated")
        normalized_names.add(name)
        identity = f"{name}=={version} {direct_url}".lower()
        if name.endswith("-cu13") or "cu13" in direct_url.lower() or "+cu13" in identity:
            raise RuntimeError(f"CUDA 13 distribution marker is forbidden: {identity}")
        if name == "cuda-python":
            major_text = re.match(r"(\d+)", version)
            if major_text is not None and int(major_text.group(1)) >= 13:
                raise RuntimeError("cuda-python 13 or newer is forbidden")
        if name == "sglang" and version == EXPECTED_RUNTIME_VERSIONS["sglang_substrate"]:
            substrate_seen = True
    if not substrate_seen:
        raise RuntimeError("frozen substrate SGLang distribution is absent")
    _validate_loaded_object_ledger(manifest["critical_files"])
    for row in manifest["critical_files"]:
        if any(not isinstance(row.get(field), int) for field in STAT_IDENTITY_FIELDS):
            raise RuntimeError("critical file identity seal is absent")
    profilers = manifest["profilers"]
    if set(profilers) != {"nsys", "ncu"}:
        raise RuntimeError("profiler manifest roles drifted")
    for authority, rows in profilers.items():
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"{authority} profiler manifest is empty")
        launchers = [row for row in rows if row.get("kind") == "launcher"]
        if len(launchers) != 1:
            raise RuntimeError(f"{authority} profiler launcher manifest drifted")
        if re.fullmatch(r"[0-9a-f]{64}", str(launchers[0].get("sha256", ""))) is None:
            raise RuntimeError(f"{authority} profiler launcher hash is absent")
        for row in rows:
            if any(not isinstance(row.get(field), int) for field in STAT_IDENTITY_FIELDS):
                raise RuntimeError(f"{authority} profiler file identity seal is absent")
    environment = manifest["environment"]
    if set(environment) != {
        "fixed_overrides",
        "path_overrides",
        "required_unset",
        "oci_config_environment",
        "observed_residual",
    }:
        raise RuntimeError("manifest environment contract inventory drifted")
    _expect(
        environment["fixed_overrides"], FIXED_CHILD_ENVIRONMENT, "manifest fixed environment"
    )
    _expect(
        set(environment["path_overrides"]),
        set(PATH_ENVIRONMENT_ROLES),
        "manifest path environment",
    )
    _expect(environment["required_unset"], list(REQUIRED_UNSET), "manifest unset environment")
    if not isinstance(environment["oci_config_environment"], list):
        raise TypeError("manifest OCI config environment is malformed")
    if not isinstance(environment["observed_residual"], dict):
        raise TypeError("manifest residual environment is malformed")
    mount_contract = manifest["mount_contract"]
    if not isinstance(mount_contract, list) or not mount_contract:
        raise RuntimeError("manifest mount contract is absent")
    mount_points = [str(row.get("mount_point", "")) for row in mount_contract]
    if not set(REQUIRED_CONTAINER_MOUNT_MODES).issubset(mount_points):
        raise RuntimeError("manifest mount contract lacks required role mounts")


def _validate_source_projection(
    manifest: Mapping[str, Any],
    imported_modules: Sequence[Mapping[str, Any]],
    entry_points: Mapping[str, Sequence[Any]],
) -> None:
    _expect(manifest.get("commit"), EXPECTED_SGLANG_COMMIT, "source projection commit")
    _expect(manifest.get("tree"), EXPECTED_SGLANG_TREE, "source projection tree")
    root = PurePosixPath(str(manifest.get("root", "")))
    if not root.is_absolute():
        raise RuntimeError("source projection root is not absolute")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("source projection file manifest is empty")
    expected_files: dict[str, str] = {}
    for row in files:
        path = str(row.get("path", ""))
        digest = str(row.get("sha256", ""))
        if not path or path.startswith("/") or ".." in PurePosixPath(path).parts:
            raise RuntimeError("source projection manifest path is unsafe")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None or path in expected_files:
            raise RuntimeError("source projection file inventory is malformed")
        expected_files[path] = digest
    version_path = "python/sglang/_version.py"
    _expect(expected_files.get(version_path), EXPECTED_VERSION_FILE_SHA256, "generated version")
    _expect(set(entry_points), {"sglang.srt.platforms", "sglang.srt.plugins"}, "entry points")
    if any(entry_points.values()):
        raise RuntimeError("external SGLang platform or plugin entry points are forbidden")
    paths = manifest.get("package_paths")
    expected_package_root = root / "python/sglang"
    if paths != [str(expected_package_root)]:
        raise RuntimeError("sglang.__path__ is not the sole clean projection")
    for row in imported_modules:
        name = str(row.get("name", ""))
        path = PurePosixPath(str(row.get("path", "")))
        digest = str(row.get("sha256", ""))
        if not name.startswith("sglang"):
            continue
        try:
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError("module path is not an absolute normalized POSIX path")
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise RuntimeError(f"imported SGLang module escaped projection: {name}") from error
        if expected_files.get(relative) != digest:
            raise RuntimeError(f"imported SGLang module hash drifted: {name}")


def _assert_output_budget(path: Path) -> int:
    size = _tree_size(path)
    if size > MAX_OUTPUT_BYTES:
        raise RuntimeError(f"retained output exceeds {MAX_OUTPUT_BYTES} bytes: {size}")
    return size


def _assert_scratch_budget(path: Path) -> int:
    size = _tree_size(path)
    if size > MAX_SCRATCH_BYTES:
        raise RuntimeError(f"scratch exceeds {MAX_SCRATCH_BYTES} bytes: {size}")
    return size


def _assert_confined_tree(root: Path) -> None:
    resolved_root = root.resolve()
    for entry in root.rglob("*"):
        if entry.is_symlink():
            raise RuntimeError(f"retained tree contains symlink: {entry}")
        if not entry.is_file() and not entry.is_dir():
            raise RuntimeError(f"retained tree contains special file: {entry}")
        try:
            entry.resolve().relative_to(resolved_root)
        except ValueError as error:
            raise RuntimeError(f"retained path escapes result root: {entry}") from error


def _artifact_manifest(root: Path) -> list[dict[str, Any]]:
    _assert_confined_tree(root)
    rows = []
    for path in sorted(entry for entry in root.rglob("*") if entry.is_file()):
        if path.name == "manifest.json":
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _normalized_header(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _find_csv_header(lines: Sequence[str], required: Sequence[str]) -> int:
    wanted = {_normalized_header(item) for item in required}
    for index, line in enumerate(lines):
        fields = next(csv.reader([line]))
        normalized = {_normalized_header(field) for field in fields}
        if wanted <= normalized:
            return index
    raise RuntimeError(f"CSV header lacks required fields: {required}")


def _trace_column(fieldnames: Sequence[str], *aliases: str) -> str:
    wanted = {_normalized_header(alias) for alias in aliases}
    matches = [field for field in fieldnames if _normalized_header(field) in wanted]
    if len(matches) != 1:
        raise RuntimeError(f"trace expected one of {aliases}, observed {matches}")
    return matches[0]


def _float_field(row: Mapping[str, str], column: str, label: str) -> float:
    try:
        value = float(row[column].strip())
    except (KeyError, ValueError) as error:
        raise RuntimeError(f"{label} is not numeric: {row.get(column)!r}") from error
    if not math.isfinite(value):
        raise RuntimeError(f"{label} must be finite")
    return value


def _optional_int(row: Mapping[str, str], column: str | None) -> int | None:
    if column is None or not row.get(column, "").strip():
        return None
    try:
        return int(row[column].strip())
    except ValueError as error:
        raise RuntimeError(f"trace integer field is invalid: {row[column]!r}") from error


def _optional_column(fieldnames: Sequence[str], *aliases: str) -> str | None:
    wanted = {_normalized_header(alias) for alias in aliases}
    matches = [field for field in fieldnames if _normalized_header(field) in wanted]
    if len(matches) > 1:
        raise RuntimeError(f"trace has ambiguous columns for {aliases}: {matches}")
    return matches[0] if matches else None


def _parse_nsys_cuda_trace(output: str) -> list[dict[str, Any]]:
    lines = output.splitlines()
    header_index = _find_csv_header(lines, ("Start (ns)", "Duration (ns)", "Name"))
    reader = csv.DictReader(lines[header_index:])
    if reader.fieldnames is None:
        raise RuntimeError("Nsight Systems CUDA trace has no header")
    names = reader.fieldnames
    start_column = _trace_column(names, "Start (ns)")
    duration_column = _trace_column(names, "Duration (ns)")
    name_column = _trace_column(names, "Name")
    context_column = _trace_column(names, "Ctx", "Context")
    stream_column = _trace_column(names, "Strm", "Stream")
    correlation_column = _trace_column(names, "CorrId", "CorrID", "Correlation ID")
    device_column = _trace_column(names, "Device")
    bytes_column = _optional_column(names, "Bytes (byte)", "Bytes")
    registers_column = _optional_column(names, "Reg/Trd", "Registers Per Thread")
    static_shared_column = _optional_column(names, "StcSMem (byte)", "Static Shared Memory (byte)")
    dynamic_shared_column = _optional_column(
        names, "DymSMem (byte)", "Dynamic Shared Memory (byte)"
    )
    grid_columns = tuple(
        _optional_column(names, short, long)
        for short, long in zip(
            ("GrdX", "GrdY", "GrdZ"), ("Grid X", "Grid Y", "Grid Z"), strict=True
        )
    )
    block_columns = tuple(
        _optional_column(names, short, long)
        for short, long in zip(
            ("BlkX", "BlkY", "BlkZ"), ("Block X", "Block Y", "Block Z"), strict=True
        )
    )
    rows: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(reader):
        name = raw.get(name_column, "").strip()
        if not name:
            continue
        start_ns = _float_field(raw, start_column, "trace start")
        duration_ns = _float_field(raw, duration_column, "trace duration")
        if start_ns < 0 or duration_ns <= 0:
            raise RuntimeError("trace duration must be positive and start nonnegative")
        lowered = name.lower()
        if "memcpy" in lowered:
            activity = "memcpy"
        elif "memset" in lowered:
            activity = "memset"
        else:
            activity = "kernel"
        context_id = _optional_int(raw, context_column)
        stream_id = _optional_int(raw, stream_column)
        correlation_id = _optional_int(raw, correlation_column)
        device = raw.get(device_column, "").strip()
        if (
            context_id is None
            or context_id < 0
            or stream_id is None
            or stream_id < 0
            or correlation_id is None
            or correlation_id < 0
            or not device
        ):
            raise RuntimeError("device activity lacks a valid device, context, stream or CorrID")
        grid = [_optional_int(raw, column) for column in grid_columns]
        block = [_optional_int(raw, column) for column in block_columns]
        if activity == "kernel":
            for label, dimensions in (("grid", grid), ("block", block)):
                if any(value is None or value <= 0 for value in dimensions):
                    raise RuntimeError(f"kernel {label} dimensions must be positive")
        rows.append(
            {
                "ordinal": ordinal,
                "activity": activity,
                "name": name,
                "start_ns": start_ns,
                "duration_ns": duration_ns,
                "end_ns": start_ns + duration_ns,
                "device": device,
                "context_id": context_id,
                "stream_id": stream_id,
                "correlation_id": correlation_id,
                "bytes": _optional_int(raw, bytes_column),
                "grid": grid,
                "block": block,
                "registers_per_thread": _optional_int(raw, registers_column),
                "static_shared_bytes": _optional_int(raw, static_shared_column),
                "dynamic_shared_bytes": _optional_int(raw, dynamic_shared_column),
            }
        )
    if not rows or not any(row["activity"] == "kernel" for row in rows):
        raise RuntimeError("Nsight Systems trace has no kernel activity")
    return rows


def _parse_nsys_cuda_api_trace(output: str) -> list[dict[str, Any]]:
    lines = output.splitlines()
    header_index = _find_csv_header(lines, ("Start (ns)", "Duration (ns)", "Name", "CorrID"))
    reader = csv.DictReader(lines[header_index:])
    if reader.fieldnames is None:
        raise RuntimeError("Nsight Systems CUDA API trace has no header")
    names = reader.fieldnames
    start_column = _trace_column(names, "Start (ns)")
    duration_column = _trace_column(names, "Duration (ns)")
    name_column = _trace_column(names, "Name")
    correlation_column = _trace_column(names, "CorrId", "CorrID", "Correlation ID")
    pid_column = _trace_column(names, "PID", "Pid")
    tid_column = _trace_column(names, "TID", "Tid")
    result_column = _optional_column(names, "Result")
    rows = []
    for ordinal, raw in enumerate(reader):
        name = raw.get(name_column, "").strip()
        if not name:
            continue
        start_ns = _float_field(raw, start_column, "CUDA API start")
        duration_ns = _float_field(raw, duration_column, "CUDA API duration")
        pid = _optional_int(raw, pid_column)
        tid = _optional_int(raw, tid_column)
        correlation_id = _optional_int(raw, correlation_column)
        if start_ns < 0 or duration_ns <= 0 or pid is None or pid < 0 or tid is None or tid < 0:
            raise RuntimeError("CUDA API row has invalid timing or process identity")
        if correlation_id is not None and correlation_id < 0:
            raise RuntimeError("CUDA API CorrID must be nonnegative")
        rows.append(
            {
                "ordinal": ordinal,
                "name": name,
                "start_ns": start_ns,
                "duration_ns": duration_ns,
                "end_ns": start_ns + duration_ns,
                "correlation_id": correlation_id,
                "pid": pid,
                "tid": tid,
                "result": raw.get(result_column, "").strip() if result_column else "",
            }
        )
    if not rows:
        raise RuntimeError("Nsight Systems CUDA API trace is empty")
    return rows


def _parse_nvtx_projection(output: str) -> list[dict[str, Any]]:
    lines = output.splitlines()
    header_index = _find_csv_header(
        lines, ("Name", "Projected Start (ns)", "Projected Duration (ns)", "Orig Start (ns)")
    )
    reader = csv.DictReader(lines[header_index:])
    if reader.fieldnames is None:
        raise RuntimeError("Nsight Systems NVTX projection has no header")
    names = reader.fieldnames
    projected_start_column = _trace_column(names, "Projected Start (ns)")
    projected_duration_column = _trace_column(names, "Projected Duration (ns)")
    original_start_column = _trace_column(names, "Orig Start (ns)", "Original Start (ns)")
    original_duration_column = _trace_column(names, "Orig Duration (ns)", "Original Duration (ns)")
    text_column = _trace_column(names, "Name")
    pid_column = _trace_column(names, "PID", "Pid")
    tid_column = _trace_column(names, "TID", "Tid")
    gpu_ops_column = _trace_column(names, "NumGPUOps", "Number of GPU Operations")
    level_column = _trace_column(names, "Lvl", "Level")
    range_id_column = _trace_column(names, "RangeId", "Range ID")
    parent_id_column = _trace_column(names, "ParentId", "Parent ID")
    range_stack_column = _optional_column(names, "RangeStack", "Range Stack")
    ranges = []
    for raw in reader:
        text = raw.get(text_column, "").strip()
        if not text:
            continue
        projected_start_ns = _float_field(raw, projected_start_column, "NVTX projected start")
        projected_duration_ns = _float_field(
            raw, projected_duration_column, "NVTX projected duration"
        )
        original_start_ns = _float_field(raw, original_start_column, "NVTX original start")
        original_duration_ns = _float_field(raw, original_duration_column, "NVTX original duration")
        pid = _optional_int(raw, pid_column)
        tid = _optional_int(raw, tid_column)
        gpu_ops = _optional_int(raw, gpu_ops_column)
        level = _optional_int(raw, level_column)
        range_id = _optional_int(raw, range_id_column)
        parent_id = _optional_int(raw, parent_id_column)
        if (
            projected_start_ns < 0
            or projected_duration_ns <= 0
            or original_start_ns < 0
            or original_duration_ns <= 0
            or pid is None
            or pid < 0
            or tid is None
            or tid < 0
            or gpu_ops is None
            or gpu_ops <= 0
            or level is None
            or level < 0
            or range_id is None
            or range_id < 0
        ):
            raise RuntimeError("projected NVTX range has invalid timing or identity")
        ranges.append(
            {
                "text": text,
                "projected_start_ns": projected_start_ns,
                "projected_duration_ns": projected_duration_ns,
                "projected_end_ns": projected_start_ns + projected_duration_ns,
                "original_start_ns": original_start_ns,
                "original_duration_ns": original_duration_ns,
                "original_end_ns": original_start_ns + original_duration_ns,
                "pid": pid,
                "tid": tid,
                "num_gpu_ops": gpu_ops,
                "level": level,
                "range_id": range_id,
                "parent_id": parent_id,
                "range_stack": (
                    raw.get(range_stack_column, "").strip() if range_stack_column else ""
                ),
            }
        )
    if not ranges:
        raise RuntimeError("Nsight Systems trace has no projected NVTX range")
    return ranges


def _api_inside_range(row: Mapping[str, Any], span: Mapping[str, Any]) -> bool:
    return (
        row["pid"] == span["pid"]
        and row["tid"] == span["tid"]
        and row["start_ns"] >= span["original_start_ns"]
        and row["end_ns"] <= span["original_end_ns"]
    )


def _annotate_ranges(
    rows: list[dict[str, Any]],
    ranges: Sequence[Mapping[str, Any]],
    api_rows: Sequence[Mapping[str, Any]],
    phase: str,
) -> list[dict[str, Any]]:
    expected_phase_names = {
        f"simllm-pilot:{phase}:step:{index:02d}" for index in range(CAPTURE_REPETITIONS)
    }
    phase_ranges = [span for span in ranges if span["text"] in expected_phase_names]
    if (
        len(phase_ranges) != CAPTURE_REPETITIONS
        or {str(span["text"]) for span in phase_ranges} != expected_phase_names
    ):
        raise RuntimeError("projected NVTX phase-range inventory drifted")
    phase_ranges.sort(key=lambda span: str(span["text"]))
    for earlier, later in itertools.pairwise(phase_ranges):
        if earlier["original_end_ns"] > later["original_start_ns"]:
            raise RuntimeError("captured phase ranges overlap or nest")
    moe_text = f"simllm-pilot:{phase}:layer-0-fused-moe"
    qkv_text = f"simllm-pilot:{phase}:layer-0-qkv"
    moe_ranges = [span for span in ranges if span["text"] == moe_text]
    qkv_ranges = [span for span in ranges if span["text"] == qkv_text]
    custom_ranges = [*phase_ranges, *moe_ranges, *qkv_ranges]
    range_ids = [int(span["range_id"]) for span in custom_ranges]
    if len(range_ids) != len(set(range_ids)):
        raise RuntimeError("custom NVTX ranges have duplicate RangeId values")
    for label, spans in (("fused MoE", moe_ranges), ("QKV", qkv_ranges)):
        if len(spans) != CAPTURE_REPETITIONS:
            raise RuntimeError(f"expected one layer-0 {label} range per retained step")
        for span in spans:
            parents = [
                parent
                for parent in phase_ranges
                if span["pid"] == parent["pid"]
                and span["tid"] == parent["tid"]
                and span["original_start_ns"] >= parent["original_start_ns"]
                and span["original_end_ns"] <= parent["original_end_ns"]
                and span["level"] > parent["level"]
            ]
            if len(parents) != 1:
                raise RuntimeError(f"layer-0 {label} range is not nested in exactly one step")
    for phase_range in phase_ranges:
        nested_qkv = [
            span
            for span in qkv_ranges
            if span["pid"] == phase_range["pid"]
            and span["tid"] == phase_range["tid"]
            and span["original_start_ns"] >= phase_range["original_start_ns"]
            and span["original_end_ns"] <= phase_range["original_end_ns"]
        ]
        nested_moe = [
            span
            for span in moe_ranges
            if span["pid"] == phase_range["pid"]
            and span["tid"] == phase_range["tid"]
            and span["original_start_ns"] >= phase_range["original_start_ns"]
            and span["original_end_ns"] <= phase_range["original_end_ns"]
        ]
        if len(nested_qkv) != 1 or len(nested_moe) != 1:
            raise RuntimeError(
                "each retained step must contain one layer-0 QKV and fused-MoE range"
            )
        if nested_qkv[0]["original_end_ns"] > nested_moe[0]["original_start_ns"]:
            raise RuntimeError("layer-0 QKV and fused-MoE ranges overlap or execute out of order")
    api_by_correlation: dict[int, list[Mapping[str, Any]]] = {}
    for api_row in api_rows:
        correlation_id = api_row.get("correlation_id")
        if correlation_id is not None:
            api_by_correlation.setdefault(int(correlation_id), []).append(api_row)
    device_correlations = [int(row["correlation_id"]) for row in rows]
    if len(device_correlations) != len(set(device_correlations)):
        raise RuntimeError("one CUDA API CorrID produced multiple device-ledger rows")
    for row in rows:
        candidates = []
        for api_row in api_by_correlation.get(int(row["correlation_id"]), []):
            containing = [span for span in phase_ranges if _api_inside_range(api_row, span)]
            if len(containing) == 1:
                candidates.append((api_row, containing[0]))
        if len(candidates) != 1:
            raise RuntimeError(
                "device row must join through CorrID to exactly one CUDA API row and step: "
                f"{row['name']} CorrID={row['correlation_id']}"
            )
        api_row, phase_range = candidates[0]
        row["api"] = {
            "ordinal": api_row["ordinal"],
            "name": api_row["name"],
            "start_ns": api_row["start_ns"],
            "duration_ns": api_row["duration_ns"],
            "end_ns": api_row["end_ns"],
            "pid": api_row["pid"],
            "tid": api_row["tid"],
            "result": api_row["result"],
        }
        row["phase_range"] = phase_range["text"]
        row["inside_layer0_fused_moe"] = any(
            _api_inside_range(api_row, span) for span in moe_ranges
        )
        row["inside_layer0_qkv"] = any(_api_inside_range(api_row, span) for span in qkv_ranges)
        if row["inside_layer0_fused_moe"] and row["inside_layer0_qkv"]:
            raise RuntimeError("one device row belongs to both layer-0 semantic ranges")
        if row["activity"] != "kernel":
            row["semantic_family"] = row["activity"]
            continue
        if row["inside_layer0_fused_moe"]:
            row["semantic_family"] = "fused_moe"
        elif row["inside_layer0_qkv"]:
            row["semantic_family"] = "qkv_projection"
        else:
            row["semantic_family"] = "unattributed"
    for span in (*phase_ranges, *moe_ranges, *qkv_ranges):
        observed = sum(1 for row in rows if _api_inside_range(row["api"], span))
        if observed != span["num_gpu_ops"]:
            raise RuntimeError(
                f"NVTX range {span['text']!r} reports {span['num_gpu_ops']} GPU ops, "
                f"joined {observed}"
            )
    return rows


def _interval_union_ns(rows: Sequence[Mapping[str, Any]]) -> float:
    intervals = sorted(
        (float(row["start_ns"]), float(row["start_ns"]) + float(row["duration_ns"])) for row in rows
    )
    if not intervals:
        return 0.0
    start, end = intervals[0]
    total = 0.0
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _ncu_target(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for index, row in enumerate(rows):
        if row.get("activity", "kernel") != "kernel":
            continue
        if not row.get("inside_layer0_fused_moe", False):
            continue
        name = str(row["name"])
        launch_skip = sum(
            1
            for earlier in rows[:index]
            if earlier.get("activity", "kernel") == "kernel" and str(earlier.get("name")) == name
        )
        return {
            "kernel_name": name,
            "kernel_regex": "^" + re.escape(name) + "$",
            "launch_skip": launch_skip,
            "source_ordinal": int(row.get("ordinal", index)),
        }
    raise CapabilityBlocked("no device kernel resolved inside layer 0 fused MoE")


def _nsys_profile_command(
    nsys: Path, output_prefix: Path, child_command: Sequence[str | Path]
) -> tuple[str, ...]:
    return tuple(
        str(item)
        for item in (
            nsys,
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            f"--capture-range-end=repeat:{CAPTURE_REPETITIONS}",
            "--cuda-event-trace=false",
            "--target-processes=all",
            "--force-overwrite=true",
            "--output",
            output_prefix,
            *child_command,
        )
    )


def _ncu_command(
    ncu: Path, target: Mapping[str, Any], child_command: Sequence[str | Path]
) -> tuple[str, ...]:
    return tuple(
        str(item)
        for item in (
            ncu,
            "--set",
            "basic",
            "--profile-from-start",
            "off",
            "--target-processes",
            "all",
            "--kernel-name-base",
            "demangled",
            "--print-metric-name",
            "name",
            "--print-units",
            "base",
            "--page",
            "raw",
            "--kernel-name",
            f"regex:{target['kernel_regex']}",
            "--launch-skip",
            str(target["launch_skip"]),
            "--launch-count",
            "1",
            "--replay-mode",
            "kernel",
            "--clock-control",
            "none",
            "--csv",
            *child_command,
        )
    )


def _has_finite_ncu_metrics(output: str, target_kernel: str) -> bool:
    try:
        _parse_ncu_metrics(output, target_kernel)
    except (CapabilityBlocked, RuntimeError, ValueError):
        return False
    return True


def _parse_ncu_metrics(output: str, target_kernel: str) -> list[dict[str, Any]]:
    lines = output.splitlines()
    try:
        header_index = _find_csv_header(lines, ("ID", "Kernel Name", "Metric Name", "Metric Value"))
    except RuntimeError as error:
        raise CapabilityBlocked("Nsight Compute returned no metric CSV header") from error
    reader = csv.DictReader(lines[header_index:])
    if reader.fieldnames is None:
        raise CapabilityBlocked("Nsight Compute returned no metric columns")
    names = reader.fieldnames
    id_column = _trace_column(names, "ID")
    kernel_column = _trace_column(names, "Kernel Name", "KernelName")
    metric_column = _trace_column(names, "Metric Name", "MetricName")
    value_column = _trace_column(names, "Metric Value", "MetricValue")
    unit_column = _optional_column(names, "Metric Unit", "MetricUnit")
    launch_ids = set()
    metrics = []
    for row in reader:
        if row.get(kernel_column, "").strip() != target_kernel:
            continue
        launch_id = row.get(id_column, "").strip()
        if not launch_id:
            raise RuntimeError("Nsight Compute target metric has no launch ID")
        launch_ids.add(launch_id)
        metric_name = row.get(metric_column, "").strip()
        if not metric_name:
            raise RuntimeError("Nsight Compute target row has no metric name")
        candidate = row.get(value_column, "").strip().replace(",", "")
        try:
            value = float(candidate)
        except ValueError as error:
            raise CapabilityBlocked(
                f"Nsight Compute metric {metric_name!r} is unsupported: {candidate!r}"
            ) from error
        if not math.isfinite(value):
            raise CapabilityBlocked(f"Nsight Compute metric {metric_name!r} is nonfinite")
        metrics.append(
            {
                "launch_id": launch_id,
                "kernel_name": target_kernel,
                "metric_name": metric_name,
                "metric_unit": row.get(unit_column, "").strip() if unit_column else "",
                "metric_value": value,
            }
        )
    if len(launch_ids) != 1 or not metrics:
        raise CapabilityBlocked("Nsight Compute did not return one exact target launch")
    metric_names = [row["metric_name"] for row in metrics]
    if len(metric_names) != len(set(metric_names)):
        raise RuntimeError("Nsight Compute returned duplicate target metric names")
    return metrics


def _metric_value_in_base_units(
    metrics: Sequence[Mapping[str, Any]], name: str, units: Mapping[str, float]
) -> float:
    matches = [row for row in metrics if row["metric_name"] == name]
    if len(matches) != 1:
        raise CapabilityBlocked(f"Nsight Compute basic set lacks exact metric {name!r}")
    unit = str(matches[0]["metric_unit"]).strip().lower()
    if unit not in units:
        raise CapabilityBlocked(f"Nsight Compute metric {name!r} has unsupported unit {unit!r}")
    return float(matches[0]["metric_value"]) * units[unit]


def _validate_ncu_physical_floor(metrics: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    duration_ns = _metric_value_in_base_units(
        metrics,
        "gpu__time_duration.sum",
        {
            "ns": 1.0,
            "nsecond": 1.0,
            "us": 1_000.0,
            "usecond": 1_000.0,
            "ms": 1_000_000.0,
            "msecond": 1_000_000.0,
            "s": 1_000_000_000.0,
            "second": 1_000_000_000.0,
        },
    )
    byte_units = {
        "byte": 1.0,
        "bytes": 1.0,
        "kbyte": 1_000.0,
        "mbyte": 1_000_000.0,
        "gbyte": 1_000_000_000.0,
    }
    try:
        dram_bytes = _metric_value_in_base_units(metrics, "dram__bytes.sum", byte_units)
    except CapabilityBlocked:
        dram_bytes = _metric_value_in_base_units(
            metrics, "dram__bytes_read.sum", byte_units
        ) + _metric_value_in_base_units(metrics, "dram__bytes_write.sum", byte_units)
    floor_ns = dram_bytes / 2_039_000_000_000.0 * 1_000_000_000.0
    if duration_ns + 1.0 < floor_ns:
        raise RuntimeError(
            f"NCU duration {duration_ns} ns is below DRAM serialization floor {floor_ns} ns"
        )
    return {
        "duration_ns": duration_ns,
        "dram_bytes": dram_bytes,
        "dram_peak_floor_ns": floor_ns,
    }


def _ncu_blocker(output: str) -> str | None:
    markers = (
        "err_nvgpuctrperm",
        "permission to access nvidia gpu performance counters",
        "profiling is not supported",
        "unsupported gpu",
        "unsupported cuda",
    )
    for line in output.splitlines():
        if any(marker in line.lower() for marker in markers):
            return line.strip()
    return None


def _validate_compatibility_control(
    output: Path,
    environment: Mapping[str, str],
    ideal: Path,
    expected_sha256: str,
) -> None:
    for name in ("SIMLLM_SGLANG_ENABLE", "SIMLLM_SGLANG_ORACLE_CAPTURE"):
        if environment.get(name) != "0":
            raise RuntimeError(f"{name} must be exactly 0")
    if _sha256(ideal) != expected_sha256:
        raise RuntimeError("ideal compatibility artifact drifted")
    if output.exists():
        artifacts = [
            path
            for path in output.rglob("*")
            if path.is_file() and path.name.endswith(PROFILER_SUFFIXES)
        ]
        if artifacts:
            raise RuntimeError(f"no-capture output contains profiler artifact: {artifacts}")


def _classify_failure(error: BaseException) -> str:
    return "BLOCKED" if isinstance(error, CapabilityBlocked) else "VOID"


def _remaining_timeout(requested: int) -> float:
    if _DEADLINE is None:
        return float(requested)
    remaining = _DEADLINE - time.monotonic()
    if remaining <= 0.5:
        raise TimeoutError("pilot reached its 40-minute deadline")
    return min(float(requested), remaining)


def _expire_phase_step(label: str) -> None:
    message = f"CapabilityBlocked: phase step {label} exceeded {STEP_TIMEOUT_SECONDS} seconds\n"
    os.write(2, message.encode("utf-8"))
    os._exit(124)


def _start_step_watchdog(label: str) -> threading.Timer:
    watchdog = threading.Timer(STEP_TIMEOUT_SECONDS, _expire_phase_step, args=(label,))
    watchdog.daemon = True
    watchdog.start()
    return watchdog


def _cancel_step_watchdog(watchdog: threading.Timer) -> None:
    watchdog.cancel()
    watchdog.join(timeout=1)
    if watchdog.is_alive():
        raise RuntimeError("phase-step watchdog did not terminate")


def _run(
    command: Sequence[str | Path],
    *,
    timeout: int,
    check: bool = True,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    normalized = [str(item) for item in command]
    try:
        process = subprocess.Popen(
            normalized,
            cwd=REPOSITORY_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=None if environment is None else dict(environment),
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=_remaining_timeout(timeout))
        except subprocess.TimeoutExpired as error:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
            diagnostic = ""
            if stdout:
                diagnostic += f"\npartial stdout:\n{stdout[-4000:]}"
            if stderr:
                diagnostic += f"\npartial stderr:\n{stderr[-4000:]}"
            raise CapabilityBlocked(
                f"command timed out: {' '.join(normalized)}{diagnostic}"
            ) from error
    except subprocess.TimeoutExpired as error:
        raise CapabilityBlocked(
            f"command group could not be terminated: {' '.join(normalized)}"
        ) from error
    except FileNotFoundError as error:
        raise CapabilityBlocked(f"required executable is unavailable: {normalized[0]}") from error
    completed = subprocess.CompletedProcess(normalized, process.returncode, stdout, stderr)
    if check and completed.returncode != 0:
        combined = completed.stdout + "\n" + completed.stderr
        if _looks_like_capability_blocker(combined):
            raise CapabilityBlocked(combined.strip()[-4000:])
        raise RuntimeError(
            f"command failed with status {completed.returncode}: {' '.join(normalized)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def _looks_like_capability_blocker(output: str) -> bool:
    markers = (
        "cuda driver version is insufficient",
        "no cuda gpus are available",
        "cuda initialization error",
        "unsupported cuda",
        "failed to initialize cuda",
        "err_nvgpuctrperm",
        "capabilityblocked",
    )
    return any(marker in output.lower() for marker in markers)


def _git_output(*arguments: str, cwd: Path = REPOSITORY_ROOT) -> str:
    return _run(("git", "-C", cwd, *arguments), timeout=60).stdout.strip()


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def _tracked_source_entries(checkout: Path) -> list[dict[str, str]]:
    completed = _run(
        ("git", "-C", checkout, "ls-tree", "-r", "-z", "--full-tree", EXPECTED_SGLANG_COMMIT),
        timeout=60,
    )
    entries = []
    for raw in completed.stdout.split("\0"):
        if not raw:
            continue
        metadata, path = raw.split("\t", 1)
        mode, kind, blob = metadata.split(" ", 2)
        if kind != "blob" or path.startswith("/") or ".." in PurePosixPath(path).parts:
            raise RuntimeError(f"unsupported tracked source entry: {raw!r}")
        entries.append({"path": path, "mode": mode, "git_blob": blob})
    if not entries:
        raise RuntimeError("frozen SGLang tree contains no tracked files")
    return entries


def _safe_extract_git_archive(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:") as handle:
        members = handle.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if member.name.startswith("/") or ".." in path.parts:
                raise RuntimeError(f"Git archive member is unsafe: {member.name!r}")
            if member.ischr() or member.isblk() or member.isfifo() or member.isdev():
                raise RuntimeError(f"Git archive member is special: {member.name!r}")
            if member.issym() or member.islnk():
                target = PurePosixPath(member.linkname)
                if target.is_absolute():
                    raise RuntimeError(f"Git archive link is unsafe: {member.name!r}")
                anchor = path.parent if member.issym() else PurePosixPath()
                normalized_target = PurePosixPath(
                    posixpath.normpath((anchor / target).as_posix())
                )
                if normalized_target.is_absolute() or (
                    normalized_target.parts and normalized_target.parts[0] == ".."
                ):
                    raise RuntimeError(f"Git archive link is unsafe: {member.name!r}")
        handle.extractall(destination, members=members)


def _source_projection_manifest(
    checkout: Path, projection: Path, *, manifest_root: PurePosixPath
) -> dict[str, Any]:
    tracked = _tracked_source_entries(checkout)
    files = []
    observed_paths: set[str] = set()
    for row in tracked:
        relative = row["path"]
        path = projection / relative
        mode = row["mode"]
        if mode == "120000":
            if not path.is_symlink():
                raise RuntimeError(f"tracked source symlink drifted: {relative}")
            payload = os.readlink(path).encode("utf-8")
            sha256 = hashlib.sha256(payload).hexdigest()
        else:
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"tracked source file is absent or unsafe: {relative}")
            payload = path.read_bytes()
            sha256 = hashlib.sha256(payload).hexdigest()
        if _git_blob_sha1(payload) != row["git_blob"]:
            raise RuntimeError(f"tracked source blob drifted: {relative}")
        files.append({**row, "sha256": sha256, "size": len(payload)})
        observed_paths.add(relative)
    version_relative = "python/sglang/_version.py"
    version_path = projection / version_relative
    if not version_path.is_file() or version_path.is_symlink():
        raise RuntimeError("frozen generated SGLang version file is absent")
    if _sha256(version_path) != EXPECTED_VERSION_FILE_SHA256:
        raise RuntimeError("generated SGLang version file hash drifted")
    files.append(
        {
            "path": version_relative,
            "mode": "100644",
            "git_blob": None,
            "sha256": EXPECTED_VERSION_FILE_SHA256,
            "size": version_path.stat().st_size,
            "generated": True,
        }
    )
    observed_paths.add(version_relative)
    actual_paths = set()
    for path in projection.rglob("*"):
        if path.is_dir():
            continue
        if not path.is_file() and not path.is_symlink():
            raise RuntimeError(f"source projection contains a special entry: {path}")
        actual_paths.add(path.relative_to(projection).as_posix())
    if actual_paths != observed_paths:
        raise RuntimeError(
            "source projection contains missing or extra entries: "
            f"missing={sorted(observed_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - observed_paths)}"
        )
    files.sort(key=lambda row: row["path"])
    return {
        "commit": EXPECTED_SGLANG_COMMIT,
        "tree": EXPECTED_SGLANG_TREE,
        "root": str(manifest_root),
        "package_paths": [str(manifest_root / "python/sglang")],
        "files": files,
    }


def _project_clean_source(checkout: Path, projection: Path) -> dict[str, Any]:
    _validate_sglang_source_unchanged(checkout)
    if projection.exists():
        raise RuntimeError(f"source projection already exists: {projection}")
    projection.mkdir(parents=True, mode=0o700)
    archive = projection.parent / "sglang-source.tar"
    if archive.exists():
        raise RuntimeError(f"source archive already exists: {archive}")
    _run(
        (
            "git",
            "-C",
            checkout,
            "archive",
            "--format=tar",
            f"--output={archive}",
            EXPECTED_SGLANG_COMMIT,
        ),
        timeout=120,
    )
    try:
        _safe_extract_git_archive(archive, projection)
    finally:
        archive.unlink(missing_ok=True)
    generated = checkout / "python/sglang/_version.py"
    if not generated.is_file() or generated.is_symlink():
        raise CapabilityBlocked("frozen generated SGLang version file is unavailable")
    if _sha256(generated) != EXPECTED_VERSION_FILE_SHA256:
        raise RuntimeError("input generated SGLang version file hash drifted")
    destination = projection / "python/sglang/_version.py"
    if destination.exists():
        raise RuntimeError("Git archive unexpectedly contains generated SGLang version file")
    shutil.copyfile(generated, destination)
    destination.chmod(0o644)
    return _source_projection_manifest(
        checkout, projection, manifest_root=PurePosixPath("/opt/simllm/source")
    )


def _digest_path(root: Path, digest: str) -> Path:
    algorithm, value = digest.split(":", 1)
    if algorithm != "sha256" or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"unsupported OCI digest: {digest!r}")
    return root / "blobs" / algorithm / value


def _validate_oci_layout(layout: Path) -> dict[str, Any]:
    index = layout / "index.json"
    if not index.is_file() or index.is_symlink():
        raise CapabilityBlocked("cached OCI index is unavailable")
    if index.stat().st_size != EXPECTED_OCI_INDEX_BYTES or _sha256(index) != (
        EXPECTED_OCI_INDEX_SHA256
    ):
        raise RuntimeError("cached OCI index identity drifted")
    index_value = json.loads(index.read_text(encoding="utf-8"))
    manifests = index_value.get("manifests")
    if not isinstance(manifests, list) or len(manifests) != 1:
        raise RuntimeError("cached OCI index must resolve one manifest")
    descriptor = manifests[0]
    _expect(descriptor.get("digest"), EXPECTED_OCI_MANIFEST_DIGEST, "OCI index manifest")
    manifest_path = _digest_path(layout, EXPECTED_OCI_MANIFEST_DIGEST)
    if not manifest_path.is_file() or _sha256(manifest_path) != EXPECTED_OCI_MANIFEST_DIGEST[7:]:
        raise RuntimeError("cached OCI manifest blob drifted")
    if descriptor.get("size") != manifest_path.stat().st_size:
        raise RuntimeError("cached OCI manifest descriptor size drifted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("config", {})
    _expect(config.get("digest"), EXPECTED_OCI_CONFIG_DIGEST, "OCI config descriptor")
    config_path = _digest_path(layout, EXPECTED_OCI_CONFIG_DIGEST)
    if not config_path.is_file() or _sha256(config_path) != EXPECTED_OCI_CONFIG_DIGEST[7:]:
        raise RuntimeError("cached OCI config blob drifted")
    if config.get("size") != config_path.stat().st_size:
        raise RuntimeError("cached OCI config descriptor size drifted")
    config_value = json.loads(config_path.read_text(encoding="utf-8"))
    config_environment = config_value.get("config", {}).get("Env")
    if not isinstance(config_environment, list) or not all(
        isinstance(item, str) and "=" in item for item in config_environment
    ):
        raise RuntimeError("cached OCI config environment is malformed")
    layers = manifest.get("layers")
    if not isinstance(layers, list) or len(layers) != EXPECTED_OCI_LAYER_COUNT:
        raise RuntimeError("cached OCI layer count drifted")
    if sum(int(row.get("size", -1)) for row in layers) != EXPECTED_OCI_LAYER_BYTES:
        raise RuntimeError("cached OCI layer byte count drifted")
    layer_rows = []
    for ordinal, row in enumerate(layers):
        digest = str(row.get("digest", ""))
        path = _digest_path(layout, digest)
        if not path.is_file() or path.is_symlink():
            raise CapabilityBlocked(f"cached OCI layer is unavailable: {digest}")
        if path.stat().st_size != int(row.get("size", -1)):
            raise RuntimeError(f"cached OCI layer size drifted: {digest}")
        if _sha256(path) != digest[7:]:
            raise RuntimeError(f"cached OCI layer hash drifted: {digest}")
        layer_rows.append({"ordinal": ordinal, "digest": digest, "size": path.stat().st_size})
    return {
        "index_sha256": EXPECTED_OCI_INDEX_SHA256,
        "index_bytes": EXPECTED_OCI_INDEX_BYTES,
        "manifest_digest": EXPECTED_OCI_MANIFEST_DIGEST,
        "config_digest": EXPECTED_OCI_CONFIG_DIGEST,
        "config_environment": config_environment,
        "layers": layer_rows,
    }


def _create_oci_archive(layout: Path, archive: Path) -> None:
    if archive.exists():
        raise RuntimeError(f"OCI archive already exists: {archive}")
    tar = _required_tool("tar")
    # Apptainer rejects a tar root member as an illegal extraction path.
    _run(
        (
            tar,
            "--format=posix",
            "-cf",
            archive,
            "-C",
            layout,
            "oci-layout",
            "index.json",
            "blobs",
        ),
        timeout=600,
    )
    if not archive.is_file() or archive.stat().st_size < EXPECTED_OCI_LAYER_BYTES:
        raise RuntimeError("OCI archive construction was incomplete")


def _construct_sandbox(
    apptainer: Path,
    layout: Path,
    sandbox: Path,
    launcher_environment: Mapping[str, str],
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    archive = sandbox.parent / "runtime.oci.tar"
    _create_oci_archive(layout, archive)
    if sandbox.exists():
        raise RuntimeError(f"runtime sandbox already exists: {sandbox}")
    try:
        _run(
            (apptainer, "build", "--sandbox", sandbox, f"oci-archive://{archive}"),
            timeout=900,
            environment=launcher_environment,
        )
        _assert_scratch_budget(sandbox.parent)
    finally:
        archive.unlink(missing_ok=True)
    if not (sandbox / "usr/bin/env").is_file():
        raise RuntimeError("constructed OCI sandbox has no /usr/bin/env")
    identity = _validate_oci_layout(layout)
    if identity != expected_identity:
        raise RuntimeError("cached OCI layout changed during sandbox construction")
    return identity


def _check_repository(expected_head: str, *, require_clean: bool = True) -> dict[str, str]:
    if re.fullmatch(r"[0-9a-f]{40}", expected_head or "") is None:
        raise ValueError("--expected-head must be a full lowercase Git SHA")
    head = _git_output("rev-parse", "HEAD")
    if head != expected_head:
        raise RuntimeError(f"SimLLM HEAD is {head}, expected {expected_head}")
    ancestor = _run(
        (
            "git",
            "-C",
            REPOSITORY_ROOT,
            "merge-base",
            "--is-ancestor",
            EXPECTED_EXPECTATIONS_COMMIT,
            head,
        ),
        timeout=60,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(
            f"expectations-only commit {EXPECTED_EXPECTATIONS_COMMIT} is not an ancestor"
        )
    status = _git_output("status", "--porcelain", "--untracked-files=all")
    if require_clean and status:
        raise RuntimeError(f"pilot source tree is dirty:\n{status}")
    return {"base": EXPECTED_BASE, "head": head}


def _check_slurm(environment: Mapping[str, str]) -> dict[str, str]:
    fields = {
        "job_id": environment.get("SLURM_JOB_ID", ""),
        "cluster": environment.get("SLURM_CLUSTER_NAME", ""),
        "partition": environment.get("SLURM_JOB_PARTITION", ""),
        "account": environment.get("SLURM_JOB_ACCOUNT", ""),
        "nodes": environment.get("SLURM_NNODES", ""),
        "tasks": environment.get("SLURM_NTASKS", ""),
        "cpus": environment.get("SLURM_CPUS_PER_TASK", ""),
        "memory_mib": environment.get("SLURM_MEM_PER_NODE", ""),
        "job_gpus": environment.get("SLURM_JOB_GPUS", ""),
        "visible_gpus": environment.get("CUDA_VISIBLE_DEVICES", ""),
    }
    expected = {
        "cluster": "gmerlin7",
        "partition": "a100-hourly",
        "account": "merlin",
        "nodes": "1",
        "tasks": "1",
        "cpus": "8",
        "memory_mib": "65536",
    }
    if not fields["job_id"]:
        raise RuntimeError("pilot is not running inside a Slurm job")
    for name, value in expected.items():
        if fields[name] != value:
            raise RuntimeError(f"Slurm {name} is {fields[name]!r}, expected {value!r}")
    for name in ("job_gpus", "visible_gpus"):
        values = [item for item in fields[name].split(",") if item]
        if len(values) != 1:
            raise RuntimeError(f"Slurm {name} does not identify one GPU: {fields[name]!r}")
    return fields


def _parse_scheduler_record(output: str, job_id: str) -> dict[str, str]:
    records = []
    for line in output.splitlines():
        values: dict[str, str] = {}
        for field in SCHEDULER_FIELDS:
            match = re.search(rf"(?:^|\s){re.escape(field)}=([^\s]*)", line)
            if match is not None:
                values[field] = match.group(1)
        if values.get("JobId") == job_id:
            records.append(values)
    if len(records) != 1:
        raise RuntimeError(f"scontrol returned {len(records)} exact records for Slurm job {job_id}")
    record = records[0]
    missing = [field for field in REQUIRED_SCHEDULER_FIELDS if not record.get(field)]
    if missing:
        raise RuntimeError(f"scontrol record lacks required fields: {missing}")
    return record


def _slurm_duration_seconds(value: str) -> int:
    day_count = 0
    clock = value
    if "-" in value:
        days, clock = value.split("-", 1)
        day_count = int(days)
    fields = [int(field) for field in clock.split(":")]
    if len(fields) == 3:
        hours, minutes, seconds = fields
    elif len(fields) == 2:
        hours = 0
        minutes, seconds = fields
    else:
        raise ValueError(f"unsupported Slurm duration: {value!r}")
    if min(day_count, hours, minutes, seconds) < 0 or minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid Slurm duration: {value!r}")
    return day_count * 86_400 + hours * 3_600 + minutes * 60 + seconds


def _scheduler_snapshot(job_id: str) -> dict[str, str]:
    scontrol = _required_tool("scontrol")
    completed = _run((scontrol, "--oneliner", "show", "job", job_id), timeout=60)
    record = _parse_scheduler_record(completed.stdout, job_id)
    expected = {
        "Account": "merlin",
        "Partition": "a100-hourly",
        "JobState": "RUNNING",
        "NumNodes": "1",
        "NumTasks": "1",
        "CPUs/Task": "8",
    }
    for field, value in expected.items():
        if record[field] != value:
            raise RuntimeError(f"Slurm scheduler {field} is {record[field]!r}, expected {value!r}")
    if _slurm_duration_seconds(record["TimeLimit"]) != 45 * 60:
        raise RuntimeError(f"Slurm time limit is {record['TimeLimit']!r}, expected 45 minutes")
    requested = set(record["ReqTRES"].split(","))
    for tres in ("cpu=8", "mem=64G", "node=1", "gres/gpu=1"):
        if tres not in requested:
            raise RuntimeError(f"Slurm requested TRES lacks {tres!r}: {record['ReqTRES']}")
    allocated = set(record["AllocTRES"].split(","))
    if "gres/gpu:nvidia_a100-sxm4-80gb=1" not in allocated:
        raise RuntimeError(
            f"Slurm allocated TRES lacks one typed A100 SXM4 80 GB: {record['AllocTRES']}"
        )
    if record["OverSubscribe"].upper() == "EXCLUSIVE":
        raise RuntimeError("pilot unexpectedly received an exclusive node")
    return record


def _job_visible_gpu_selector(slurm: Mapping[str, str]) -> str:
    visible = slurm["visible_gpus"].strip()
    if visible.isdigit() or re.fullmatch(r"GPU-[0-9A-Fa-f-]+", visible) is not None:
        return visible
    raise RuntimeError(
        f"cannot resolve the job-local GPU selector from CUDA_VISIBLE_DEVICES: {visible!r}"
    )


def _resolve_required_path(environment: Mapping[str, str], name: str) -> Path:
    raw = environment.get(name, "")
    if not raw:
        raise RuntimeError(f"{name} is required")
    path = Path(raw).resolve()
    if not path.exists():
        raise CapabilityBlocked(f"{name} is unavailable: {path}")
    return path


def _validated_model_revision(environment: Mapping[str, str]) -> str:
    revision = environment.get("SIMLLM_MODEL_REVISION", "")
    if revision != EXPECTED_MODEL_REVISION:
        raise RuntimeError(f"model revision drifted: {revision!r} != {EXPECTED_MODEL_REVISION!r}")
    return revision


def _validate_input_identities(environment: Mapping[str, str]) -> dict[str, Any]:
    source = _resolve_required_path(environment, "SIMLLM_PILOT_SGLANG_CHECKOUT")
    layout = _resolve_required_path(environment, "SIMLLM_PILOT_OCI_LAYOUT")
    model = _resolve_required_path(environment, "SIMLLM_PILOT_MODEL_SNAPSHOT")
    cuda_install = _resolve_required_path(environment, "SIMLLM_PILOT_CUDA_INSTALL")
    source_head = _git_output("rev-parse", "HEAD", cwd=source)
    source_tree = _git_output("rev-parse", "HEAD^{tree}", cwd=source)
    source_status = _git_output("status", "--porcelain", "--untracked-files=all", cwd=source)
    if source_head != EXPECTED_SGLANG_COMMIT or source_tree != EXPECTED_SGLANG_TREE:
        raise RuntimeError("SGLang source identity drifted")
    if source_status:
        raise RuntimeError(f"SGLang source tree is dirty:\n{source_status}")
    version_file = source / "python/sglang/_version.py"
    if not version_file.is_file() or _sha256(version_file) != EXPECTED_VERSION_FILE_SHA256:
        raise RuntimeError("generated SGLang source version file drifted")
    model_revision = _validated_model_revision(environment)
    config = model / "config.json"
    if _sha256(config) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("model config hash drifted")
    config_value = json.loads(config.read_text(encoding="utf-8"))
    expected_geometry = {
        "hidden_size": 1024,
        "intermediate_size": 512,
        "num_attention_heads": 16,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 24,
        "num_key_value_heads": 8,
        "num_local_experts": 32,
        "vocab_size": 49_155,
    }
    for name, expected in expected_geometry.items():
        if config_value.get(name) != expected:
            raise RuntimeError(f"model geometry drifted for {name}")
    weights = sorted(model.glob("*.safetensors"))
    if len(weights) != 1:
        raise RuntimeError(f"expected one safetensors file, observed {weights}")
    if weights[0].stat().st_size != EXPECTED_WEIGHT_BYTES:
        raise RuntimeError("model weight size drifted")
    if _sha256(weights[0]) != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("model weight hash drifted")
    profiler_tools = {}
    for name, expected_version in (("nsys", "2025.1.3"), ("ncu", "2025.2.1")):
        executable = cuda_install / "bin" / name
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise CapabilityBlocked(f"frozen {name} installation is unavailable")
        completed = _run((executable, "--version"), timeout=60)
        version = (completed.stdout + "\n" + completed.stderr).strip()
        if expected_version not in version:
            raise RuntimeError(f"{name} version drifted: {version!r}")
        profiler_tools[name] = {"path": str(executable), "version": version}
    oci = _validate_oci_layout(layout)
    return {
        "paths": {
            "source": str(source),
            "oci": str(layout),
            "model": str(model),
            "cuda_install": str(cuda_install),
        },
        "sglang_source": {"commit": source_head, "tree": source_tree},
        "oci": oci,
        "profiler_tools": profiler_tools,
        "model": {
            "revision": model_revision,
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "weight_sha256": EXPECTED_WEIGHT_SHA256,
            "weight_bytes": EXPECTED_WEIGHT_BYTES,
            "geometry": expected_geometry,
        },
    }


def _validate_job_environment(
    environment: Mapping[str, str], scratch: Path, out: Path
) -> dict[str, str]:
    _validate_host_launcher_environment(environment)
    expected_values = {
        "SIMLLM_PILOT_CUDA_MODULE": "cuda/12.9.1",
        "SIMLLM_SGLANG_ENABLE": "0",
        "SIMLLM_SGLANG_ORACLE_CAPTURE": "0",
    }
    for name, expected in expected_values.items():
        if environment.get(name) != expected:
            raise RuntimeError(f"{name} must be exactly {expected!r}")
    for name in REQUIRED_UNSET:
        if environment.get(name):
            raise RuntimeError(f"{name} must be unset for the offline pilot")
    scratch_environment = _resolve_required_path(environment, "SIMLLM_SCRATCH_ROOT")
    if scratch_environment != scratch:
        raise RuntimeError("SIMLLM_SCRATCH_ROOT does not match --scratch-root")
    run_root_raw = environment.get("SIMLLM_RUN_ROOT", "")
    if not run_root_raw or not Path(run_root_raw).is_absolute():
        raise RuntimeError("SIMLLM_RUN_ROOT must be an absolute path")
    if Path(run_root_raw).resolve() != out:
        raise RuntimeError("SIMLLM_RUN_ROOT does not match --out")
    input_roles = {}
    for name in (
        "SIMLLM_PILOT_OCI_LAYOUT",
        "SIMLLM_PILOT_MODEL_SNAPSHOT",
        "SIMLLM_PILOT_SGLANG_CHECKOUT",
        "SIMLLM_PILOT_CUDA_INSTALL",
    ):
        input_roles[name] = str(_resolve_required_path(environment, name))
    return {**expected_values, **input_roles}


def _validate_scratch_free(scratch: Path) -> int:
    free = shutil.disk_usage(scratch).free
    if free < MIN_SCRATCH_FREE_BYTES:
        raise CapabilityBlocked(
            f"job scratch has {free} free bytes, requires {MIN_SCRATCH_FREE_BYTES}"
        )
    return free


def _gpu_snapshot(nvidia_smi: Path, selector: str) -> dict[str, str]:
    completed = _run(
        (
            nvidia_smi,
            f"--id={selector}",
            f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ),
        timeout=60,
    )
    rows = list(csv.reader(completed.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != len(GPU_QUERY_FIELDS):
        raise RuntimeError(f"nvidia-smi did not return one GPU: {rows}")
    record = {name: value.strip() for name, value in zip(GPU_QUERY_FIELDS, rows[0], strict=True)}
    if record["name"] != EXPECTED_GPU_NAME:
        raise RuntimeError(f"GPU name drifted: {record['name']!r}")
    if record["compute_cap"] != EXPECTED_COMPUTE_CAPABILITY:
        raise RuntimeError(f"compute capability drifted: {record['compute_cap']!r}")
    if record["driver_version"] != EXPECTED_DRIVER:
        raise RuntimeError(f"driver drifted: {record['driver_version']!r}")
    if record["memory.total"] != "81920":
        raise RuntimeError(f"GPU memory size drifted: {record['memory.total']!r}")
    return record


def _mig_state(nvidia_smi: Path, selector: str) -> dict[str, str]:
    completed = _run((nvidia_smi, f"--id={selector}", "-q"), timeout=60)
    match = re.search(
        r"(?m)^[ \t]*MIG Mode[ \t]*\r?\n"
        r"[ \t]*Current[ \t]*:[ \t]*([^\r\n]+)"
        r"\r?\n[ \t]*Pending[ \t]*:[ \t]*([^\r\n]+)",
        completed.stdout,
    )
    if match is None:
        raise RuntimeError("nvidia-smi full query has no complete MIG mode section")
    state = {"current": match.group(1).strip(), "pending": match.group(2).strip()}
    if any(value.lower() != "disabled" for value in state.values()):
        raise RuntimeError(f"MIG is not stably disabled: {state}")
    return state


def _supported_clock_policy(
    nvidia_smi: Path, selector: str
) -> tuple[list[dict[str, int]], str | None]:
    completed = _run(
        (
            nvidia_smi,
            f"--id={selector}",
            "--query-supported-clocks=memory,graphics",
            "--format=csv,noheader,nounits",
        ),
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        diagnostic = (completed.stdout + "\n" + completed.stderr).strip()
        return [], f"supported clock query failed: {diagnostic}"
    clocks = []
    try:
        for row in csv.reader(completed.stdout.splitlines()):
            if len(row) != 2:
                raise ValueError(f"unexpected row {row!r}")
            memory_mhz, graphics_mhz = (int(value.strip()) for value in row)
            if memory_mhz <= 0 or graphics_mhz <= 0:
                raise ValueError(f"nonpositive row {row!r}")
            clocks.append({"memory_mhz": memory_mhz, "graphics_mhz": graphics_mhz})
    except ValueError as error:
        return [], f"supported clock query was unparseable: {error}"
    if not clocks:
        return [], "supported clock query returned no clock pairs"
    return clocks, None


def _foreign_processes(nvidia_smi: Path, selector: str, gpu_uuid: str) -> list[dict[str, str]]:
    fields = ("gpu_uuid", "pid", "process_name", "used_gpu_memory")
    processes = _run(
        (
            nvidia_smi,
            f"--id={selector}",
            f"--query-compute-apps={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ),
        timeout=60,
        check=False,
    )
    if processes.returncode != 0 and "No running processes found" not in (
        processes.stdout + processes.stderr
    ):
        raise RuntimeError(f"nvidia-smi process query failed: {processes.stderr}")
    rows = []
    for row in csv.reader(processes.stdout.splitlines()):
        if len(row) != len(fields):
            continue
        record = {field: value.strip() for field, value in zip(fields, row, strict=True)}
        if record["gpu_uuid"].lower() == gpu_uuid.lower():
            rows.append(record)
    return rows


def _gpu_state(nvidia_smi: Path, selector: str) -> dict[str, Any]:
    snapshot = _gpu_snapshot(nvidia_smi, selector)
    mig = _mig_state(nvidia_smi, selector)
    supported_clocks, clock_blocker = _supported_clock_policy(nvidia_smi, selector)
    if clock_blocker is not None:
        raise CapabilityBlocked(clock_blocker)
    processes = _foreign_processes(nvidia_smi, selector, snapshot["uuid"])
    if processes:
        raise RuntimeError(f"allocated GPU has foreign compute processes: {processes}")
    return {
        "snapshot": snapshot,
        "mig": mig,
        "supported_clocks": supported_clocks,
        "processes": processes,
    }


def _validate_gpu_state_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    stable_fields = (
        "index",
        "name",
        "uuid",
        "pci.bus_id",
        "compute_cap",
        "memory.total",
        "driver_version",
        "persistence_mode",
        "compute_mode",
        "clocks.max.sm",
        "clocks.max.memory",
        "power.limit",
    )
    for field in stable_fields:
        if before["snapshot"][field] != after["snapshot"][field]:
            raise RuntimeError(f"GPU state changed during pilot: {field}")
    if before["mig"] != after["mig"]:
        raise RuntimeError("MIG state changed during pilot")
    before_clocks = {
        (int(row["memory_mhz"]), int(row["graphics_mhz"])) for row in before["supported_clocks"]
    }
    after_clocks = {
        (int(row["memory_mhz"]), int(row["graphics_mhz"])) for row in after["supported_clocks"]
    }
    if before_clocks != after_clocks:
        raise RuntimeError("supported clock policy changed during pilot")


def _normalize_gpu_uuid(value: str) -> str:
    normalized = value.strip().lower().removeprefix("gpu-")
    normalized = normalized.replace("-", "")
    if re.fullmatch(r"[0-9a-f]{32}", normalized) is None:
        raise RuntimeError(f"GPU UUID has an unsupported representation: {value!r}")
    return normalized


def _cache_inventory(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise RuntimeError(f"cache contains unsafe entry: {path}")
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return rows


def _launcher_environment(
    host_environment: Mapping[str, str], scratch: Path
) -> dict[str, str]:
    _validate_host_launcher_environment(host_environment)
    visible = host_environment.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("launcher must forward exactly one CUDA_VISIBLE_DEVICES value")
    cache = scratch / "apptainer-cache"
    temporary = scratch / "apptainer-tmp"
    cache.mkdir(mode=0o700)
    temporary.mkdir(mode=0o700)
    return {
        "APPTAINER_CACHEDIR": str(cache),
        "APPTAINER_TMPDIR": str(temporary),
        "CUDA_VISIBLE_DEVICES": visible,
        "PATH": "/usr/bin:/bin",
    }


def _child_command(mode: str, phase: str | None, output: Path) -> tuple[str, ...]:
    command = [
        "/usr/bin/python3",
        "/opt/simllm/runner/examples/sglang_a100_kernel_pilot_v2/run_study.py",
        "--child",
        mode,
    ]
    if phase is not None:
        command.extend(("--phase", phase))
    command.extend(("--child-out", str(output)))
    return tuple(command)


def _required_tool(name: str) -> Path:
    resolved = shutil.which(name)
    if resolved is None:
        raise CapabilityBlocked(f"required profiler tool is unavailable: {name}")
    return Path(resolved)


def _profiler_tools() -> dict[str, dict[str, str]]:
    expected = {"nsys": "2025.1.3", "ncu": "2025.2.1"}
    tools = {}
    for name, version in expected.items():
        path = _required_tool(name)
        completed = _run((path, "--version"), timeout=60)
        output = (completed.stdout + "\n" + completed.stderr).strip()
        if version not in output:
            raise RuntimeError(f"{name} version drifted: {output!r}")
        tools[name] = {"path": str(path.resolve()), "version": output}
    return tools


def _validate_nsys_output_paths(
    output: str, scratch: Path, expected_report: Path, temporary_root: Path
) -> dict[str, str]:
    temporary_paths = re.findall(r"(?m)^Generating '([^'\r\n]+\.qdstrm)'[ \t]*$", output)
    report_paths = re.findall(
        r"(?m)^Generated:[ \t]*\r?\n[ \t]+([^\r\n]+?\.nsys-rep)[ \t]*$", output
    )
    if len(temporary_paths) != 1:
        raise RuntimeError(
            f"Nsight Systems did not report exactly one temporary stream path: {temporary_paths}"
        )
    if len(report_paths) != 1:
        raise RuntimeError(
            f"Nsight Systems did not report exactly one final report path: {report_paths}"
        )
    temporary = Path(temporary_paths[0])
    report = Path(report_paths[0])
    if not temporary.is_absolute() or not report.is_absolute():
        raise RuntimeError("Nsight Systems reported a nonabsolute output path")
    try:
        temporary_relative = temporary.resolve().relative_to(temporary_root.resolve())
        report_relative = report.resolve().relative_to(scratch.resolve())
    except ValueError as error:
        raise RuntimeError("Nsight Systems output escaped the configured scratch root") from error
    if report.resolve() != expected_report.resolve():
        raise RuntimeError(
            f"Nsight Systems reported {report.resolve()}, expected {expected_report.resolve()}"
        )
    return {
        "temporary": (Path(temporary_root.name) / temporary_relative).as_posix(),
        "report": report_relative.as_posix(),
    }


def _validate_sglang_source_unchanged(source: Path) -> None:
    if _git_output("rev-parse", "HEAD", cwd=source) != EXPECTED_SGLANG_COMMIT:
        raise RuntimeError("SGLang commit changed during the pilot")
    if _git_output("rev-parse", "HEAD^{tree}", cwd=source) != EXPECTED_SGLANG_TREE:
        raise RuntimeError("SGLang tree changed during the pilot")
    status = _git_output("status", "--porcelain", "--untracked-files=all", cwd=source)
    if status:
        raise RuntimeError(f"SGLang source tree changed during the pilot:\n{status}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _child_progress_path(child_output: Path) -> Path:
    return child_output.with_suffix(".progress.json")


def _append_child_progress(
    child_output: Path, *, mode: str, phase: str, stage: str, **details: Any
) -> dict[str, Any]:
    path = _child_progress_path(child_output)
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        _expect(
            value.get("schema"), "simllm-sglang-a100-kernel-pilot-progress-v2", "progress schema"
        )
        _expect(value.get("mode"), mode, "progress mode")
        _expect(value.get("phase"), phase, "progress phase")
        history = value.get("history")
        if not isinstance(history, list):
            raise RuntimeError("child progress history is malformed")
    else:
        value = {
            "schema": "simllm-sglang-a100-kernel-pilot-progress-v2",
            "mode": mode,
            "phase": phase,
            "history": [],
        }
        history = value["history"]
    history.append({"stage": stage, "at_unix_ns": time.time_ns(), **details})
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return value


def _optional_child_progress(child_output: Path) -> dict[str, Any] | None:
    path = _child_progress_path(child_output)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("child progress is not an object")
        return value
    except (OSError, TypeError, ValueError) as error:
        return {
            "schema": "simllm-sglang-a100-kernel-pilot-progress-unreadable-v2",
            "error": str(error),
        }


def _lane_failure(error: BaseException, child_output: Path) -> dict[str, Any]:
    value = {"state": _classify_failure(error), "error": str(error)}
    progress = _optional_child_progress(child_output)
    if progress is not None:
        value["progress"] = progress
    return value


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"child produced no JSON result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_child_result(
    value: Mapping[str, Any], mode: str, phase: str, expected_gpu_uuid: str
) -> dict[str, Any]:
    _expect(value.get("schema"), "simllm-sglang-a100-kernel-pilot-child-v2", "child schema")
    _expect(value.get("mode"), mode, "child mode")
    _expect(value.get("phase"), phase, "child phase")
    _expect(value.get("versions"), EXPECTED_RUNTIME_VERSIONS, "runtime packages")
    _expect(
        value.get("runner_identity"),
        "sglang.srt.model_executor.model_runner.ModelRunner",
        "runtime runner",
    )
    if not str(value.get("model_identity", "")).endswith("granitemoe.GraniteMoeForCausalLM"):
        raise RuntimeError("runtime model identity drifted")
    _expect(value.get("cuda_device_name"), EXPECTED_GPU_NAME, "CUDA device name")
    if _normalize_gpu_uuid(str(value.get("cuda_device_uuid", ""))) != _normalize_gpu_uuid(
        expected_gpu_uuid
    ):
        raise RuntimeError("CUDA and nvidia-smi GPU UUIDs disagree")
    source = value.get("source", {})
    _expect(source.get("commit"), EXPECTED_SGLANG_COMMIT, "child SGLang commit")
    _expect(source.get("tree"), EXPECTED_SGLANG_TREE, "child SGLang tree")
    for name in ("package", "one_batch"):
        path = source.get(name)
        if not isinstance(path, str) or path.startswith(("/", "..")):
            raise RuntimeError(f"child SGLang {name} path is not repository-relative")
    if re.fullmatch(r"[0-9a-f]{64}", str(value.get("environment_manifest_sha256", ""))) is None:
        raise RuntimeError("child environment manifest digest is absent")
    entry_points = value.get("entry_points")
    _expect(
        entry_points,
        {"sglang.srt.platforms": [], "sglang.srt.plugins": []},
        "child external entry points",
    )
    modules = value.get("modules")
    if not isinstance(modules, list) or not modules:
        raise RuntimeError("child module ledger is absent")
    loaded_objects = value.get("loaded_objects")
    if not isinstance(loaded_objects, list) or not loaded_objects:
        raise RuntimeError("child loaded-object ledger is absent")
    _validate_loaded_object_ledger(loaded_objects)
    _expect(
        value.get("backends"),
        {
            "attention": "triton",
            "moe_a2a": "none",
            "moe_runner": "triton",
            "sampling": "pytorch",
        },
        "runtime backends",
    )
    expected_repetitions = (
        TIMING_REPETITIONS
        if mode == "timing"
        else (CAPTURE_REPETITIONS if mode == "capture" else 1)
    )
    measurements = value.get("measurements", [])
    if len(measurements) != expected_repetitions:
        raise RuntimeError("child measurement cardinality drifted")
    expected_shape = {
        "batch_size": 4,
        "forward_mode": "extend" if phase == "prefill-t512-r4" else "decode",
        "input_tokens_per_request": 128 if phase == "prefill-t512-r4" else 2047,
        "request_ids": [f"{phase}-r{index}" for index in range(4)],
        "scheduled_tokens": 512 if phase == "prefill-t512-r4" else 4,
        "seq_lens": [128] * 4 if phase == "prefill-t512-r4" else [2048] * 4,
    }
    outputs = []
    pool_observations = []
    for index, row in enumerate(measurements):
        _expect(row.get("repetition"), index, "child repetition index")
        for name, expected in expected_shape.items():
            _expect(row.get(name), expected, f"child workload {name}")
        device_ms = float(row.get("device_ms", math.nan))
        host_ms = float(row.get("host_ms", math.nan))
        if not math.isfinite(device_ms) or device_ms <= 0:
            raise RuntimeError("child device duration must be finite and positive")
        if not math.isfinite(host_ms) or host_ms + 1e-9 < device_ms:
            raise RuntimeError("child host settlement is shorter than device span")
        output_ids = row.get("output_ids")
        if (
            not isinstance(output_ids, list)
            or len(output_ids) != 4
            or any(not isinstance(item, int) or item < 0 or item >= 49_155 for item in output_ids)
        ):
            raise RuntimeError("child output-token cardinality or domain drifted")
        checksum = hashlib.sha256(json.dumps(output_ids).encode("utf-8")).hexdigest()
        _expect(row.get("output_sha256"), checksum, "child output checksum")
        pool_indices = row.get("request_pool_indices")
        if (
            not isinstance(pool_indices, list)
            or len(pool_indices) != 4
            or len(set(pool_indices)) != 4
            or any(not isinstance(item, int) or item < 0 for item in pool_indices)
        ):
            raise RuntimeError("request-pool allocation did not reset to four unique slots")
        _expect(row.get("kv_slots_allocated"), expected_shape["scheduled_tokens"], "KV slots")
        pool_observations.append(tuple(pool_indices))
        outputs.append((tuple(output_ids), checksum))
    if len(set(outputs)) != 1:
        raise RuntimeError("deterministic output tokens changed across repetitions")
    if len(set(pool_observations)) != 1:
        raise RuntimeError("request-pool slot allocation changed across repetitions")
    cache_inventory = value.get("cache_inventory")
    expected_cache_roots = set(AUDITED_CHILD_ROOTS)
    if not isinstance(cache_inventory, dict) or set(cache_inventory) != expected_cache_roots:
        raise RuntimeError("child cache inventory is incomplete")
    return dict(value)


def _stats_csv(nsys: Path, report: Path, report_name: str) -> str:
    completed = _run(
        (nsys, "stats", "--report", report_name, "--format", "csv", "--output", "-", report),
        timeout=PROFILER_TIMEOUT_SECONDS,
    )
    return completed.stdout


def _capture_summary(
    phase: str,
    child: Mapping[str, Any],
    rows: list[dict[str, Any]],
    ranges: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["phase_range"]), []).append(row)
    if len(groups) != CAPTURE_REPETITIONS:
        raise RuntimeError(
            f"{phase} expected {CAPTURE_REPETITIONS} capture ranges, observed {len(groups)}"
        )
    phase_ranges = {
        str(span["text"]): span
        for span in ranges
        if str(span["text"]).startswith(f"simllm-pilot:{phase}:step:")
    }
    if set(phase_ranges) != set(groups):
        raise RuntimeError("device-ledger phase ranges do not match projected NVTX ranges")
    summaries = []
    event_spans = child.get("measurements", [])
    if len(event_spans) != CAPTURE_REPETITIONS:
        raise RuntimeError("capture child measurement cardinality drifted")
    for index, name in enumerate(sorted(groups)):
        group = sorted(groups[name], key=lambda row: (row["start_ns"], row["ordinal"]))
        kernels = [row for row in group if row["activity"] == "kernel"]
        if not kernels:
            raise RuntimeError(f"captured range {name!r} has no kernel activity")
        if any("nccl" in str(row["name"]).lower() for row in kernels):
            raise RuntimeError("TP=EP=PP=1 capture unexpectedly contains an NCCL kernel")
        devices = {str(row["device"]) for row in group}
        contexts = {int(row["context_id"]) for row in group}
        if len(devices) != 1 or EXPECTED_GPU_NAME not in next(iter(devices)):
            raise RuntimeError("device-ledger GPU identity drifted")
        if len(contexts) != 1:
            raise RuntimeError("one retained step spans multiple CUDA contexts")
        busy = _interval_union_ns(kernels)
        activity_union = _interval_union_ns(group)
        first_device_ns = min(row["start_ns"] for row in group)
        last_device_ns = max(row["end_ns"] for row in group)
        device_span = last_device_ns - first_device_ns
        additive = sum(row["duration_ns"] for row in kernels)
        if not 0 <= busy <= activity_union <= device_span:
            raise RuntimeError("activity unions lie outside the joined device span")
        projected = phase_ranges[name]
        projection_tolerance = max(10.0, 1e-6 * device_span)
        if (
            abs(float(projected["projected_start_ns"]) - first_device_ns) > projection_tolerance
            or abs(float(projected["projected_end_ns"]) - last_device_ns) > projection_tolerance
        ):
            raise RuntimeError("projected NVTX bounds disagree with joined device activity")
        event_ns = float(event_spans[index]["device_ms"]) * 1_000_000.0
        host_ns = float(event_spans[index]["host_ms"]) * 1_000_000.0
        tolerance = max(20_000.0, 0.05 * max(event_ns, device_span))
        if abs(event_ns - device_span) > tolerance:
            raise RuntimeError(
                f"CUDA-event and Nsight spans disagree for {phase}: {event_ns} vs {device_span} ns"
            )
        if host_ns + tolerance < device_span:
            raise RuntimeError("host settlement is shorter than the joined all-stream device span")
        hard_floor_ns = 1_238_933.0 if phase == "prefill-t512-r4" else 9_679.0
        if device_span + 1.0 < hard_floor_ns:
            raise RuntimeError(
                f"captured {phase} span is below its peak-compute floor: "
                f"{device_span} < {hard_floor_ns} ns"
            )
        family_counts = Counter(str(row["semantic_family"]) for row in kernels)
        if sum(family_counts.values()) != len(kernels):
            raise RuntimeError("semantic family counts do not conserve launches")
        summaries.append(
            {
                "range": name,
                "kernel_count": len(kernels),
                "device_activity_count": len(group),
                "device_span_ns": device_span,
                "device_activity_union_ns": activity_union,
                "kernel_busy_union_ns": busy,
                "kernel_duration_sum_ns": additive,
                "exposed_gap_ns": device_span - busy,
                "exposed_kernel_gap_ns": device_span - busy,
                "exposed_activity_gap_ns": device_span - activity_union,
                "family_counts": dict(sorted(family_counts.items())),
                "streams": sorted({row["stream_id"] for row in group}),
                "contexts": sorted(contexts),
                "memcpy_count": sum(row["activity"] == "memcpy" for row in group),
                "memset_count": sum(row["activity"] == "memset" for row in group),
                "memcpy_bytes": sum(
                    int(row["bytes"] or 0) for row in group if row["activity"] == "memcpy"
                ),
                "memset_bytes": sum(
                    int(row["bytes"] or 0) for row in group if row["activity"] == "memset"
                ),
            }
        )
    counts = [row["kernel_count"] for row in summaries]
    if len(set(counts)) != 1:
        raise RuntimeError("repeated launch counts differ and have no frozen explanation")
    identity_sequences = []
    for name in sorted(groups):
        identity_sequences.append(
            tuple(
                str(row["name"])
                for row in sorted(groups[name], key=lambda row: (row["start_ns"], row["ordinal"]))
                if row["activity"] == "kernel"
            )
        )
    if len(set(identity_sequences)) != 1:
        raise RuntimeError("ordered kernel identities differ across retained ranges")
    return {
        "phase": phase,
        "ranges": summaries,
        "kernel_count_min": min(counts),
        "kernel_count_max": max(counts),
        "kernel_count_stable": len(set(counts)) == 1,
        "memcpy_count_total": sum(row["activity"] == "memcpy" for row in rows),
        "memset_count_total": sum(row["activity"] == "memset" for row in rows),
    }


def _validate_static_contract(frozen: Mapping[str, Any], scratch: Path, out: Path) -> None:
    if _git_file_sha256("HEAD", EXPECTATIONS_PATH) != EXPECTED_EXPECTATIONS_SHA256:
        raise RuntimeError("frozen expectations file hash drifted")
    expectations_markdown = EXPECTATIONS_PATH.with_suffix(".md")
    if _git_file_sha256("HEAD", expectations_markdown) != EXPECTED_EXPECTATIONS_MARKDOWN_SHA256:
        raise RuntimeError("frozen expectations Markdown hash drifted")
    if _git_file_sha256("HEAD", IDEAL_ARTIFACT) != EXPECTED_IDEAL_SHA256:
        raise RuntimeError("ideal compatibility artifact drifted")
    try:
        out.relative_to(scratch)
    except ValueError as error:
        raise RuntimeError("--out must be beneath --scratch-root") from error
    for request, length in ((0, 128), (3, 2047)):
        tokens = [_token_id(request, position) for position in range(length)]
        if len(tokens) != length or not all(1 <= token < 49_155 for token in tokens):
            raise RuntimeError("frozen token formula escaped the model vocabulary")
    nsys_command = _nsys_profile_command(
        Path("nsys"), Path("capture"), ("python", "run_study.py", "--child", "capture")
    )
    for literal in (
        "--trace=cuda,nvtx",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=repeat:5",
        "--cuda-event-trace=false",
    ):
        if literal not in nsys_command:
            raise RuntimeError(f"frozen Nsight Systems command lacks {literal}")
    target = {"kernel_regex": "^target$", "launch_skip": 0}
    ncu_command = _ncu_command(Path("ncu"), target, ("python", "run_study.py", "--child", "ncu"))
    for literal in ("basic", "demangled", "regex:^target$", "kernel", "none"):
        if literal not in ncu_command:
            raise RuntimeError(f"frozen Nsight Compute command lacks {literal}")
    expected_sections = {
        "allocation",
        "chronology",
        "compatibility",
        "execution_environment",
        "identity",
        "measurement",
        "model_geometry",
        "physical",
        "runtime",
        "source_invocations",
        "token_formula",
        "transferred_vllm_bracket",
        "workloads",
    }
    if set(frozen) != expected_sections | {"schema"}:
        raise RuntimeError("frozen evidence inventory drifted")
    parent_record = _git_output("rev-list", "--parents", "-n", "1", EXPECTED_EXPECTATIONS_COMMIT)
    _expect(
        parent_record,
        f"{EXPECTED_EXPECTATIONS_COMMIT} {EXPECTED_BASE}",
        "expectations-only parent",
    )
    changed = _git_output(
        "diff-tree", "--no-commit-id", "--name-status", "-r", EXPECTED_EXPECTATIONS_COMMIT
    ).splitlines()
    _expect(
        changed,
        [
            "A\texamples/sglang_a100_kernel_pilot_v2/expectations.json",
            "A\texamples/sglang_a100_kernel_pilot_v2/expectations.md",
        ],
        "expectations-only diff",
    )
    chronology = frozen["chronology"]
    historical = {
        "v1_expectations_json_sha256": (
            EXPECTED_BASE,
            "examples/sglang_a100_kernel_pilot_v1/expectations.json",
        ),
        "v1_expectations_markdown_sha256": (
            EXPECTED_BASE,
            "examples/sglang_a100_kernel_pilot_v1/expectations.md",
        ),
        "v1_result_sha256": (
            EXPECTED_BASE,
            "examples/sglang_a100_kernel_pilot_v1/RESULTS.md",
        ),
    }
    for field, (commit, path) in historical.items():
        payload = _run(
            ("git", "-C", REPOSITORY_ROOT, "show", f"{commit}:{path}"), timeout=60
        ).stdout.encode("utf-8")
        _expect(hashlib.sha256(payload).hexdigest(), chronology[field], field)


def _project_repository(expected_head: str, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"runner projection already exists: {destination}")
    destination.mkdir(parents=True, mode=0o700)
    archive = destination.parent / "simllm-runner.tar"
    if archive.exists():
        raise RuntimeError(f"runner archive already exists: {archive}")
    _run(
        (
            "git",
            "-C",
            REPOSITORY_ROOT,
            "archive",
            "--format=tar",
            f"--output={archive}",
            expected_head,
        ),
        timeout=120,
    )
    try:
        _safe_extract_git_archive(archive, destination)
    finally:
        archive.unlink(missing_ok=True)
    runner = destination / "examples/sglang_a100_kernel_pilot_v2/run_study.py"
    expectations = destination / "examples/sglang_a100_kernel_pilot_v2/expectations.json"
    if not runner.is_file() or not expectations.is_file():
        raise RuntimeError("runner projection lacks the V2 study files")
    if _sha256(expectations) != EXPECTED_EXPECTATIONS_SHA256:
        raise RuntimeError("runner projection expectations drifted")


def _copy_model_projection(source: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise RuntimeError(f"model projection already exists: {destination}")
    shutil.copytree(source, destination, symlinks=False)
    config = destination / "config.json"
    weights = sorted(destination.glob("*.safetensors"))
    if _sha256(config) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("copied model config hash drifted")
    if len(weights) != 1 or weights[0].stat().st_size != EXPECTED_WEIGHT_BYTES:
        raise RuntimeError("copied model weight inventory drifted")
    if _sha256(weights[0]) != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("copied model weight hash drifted")
    for path in destination.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise RuntimeError(f"model projection contains an unsafe entry: {path}")
    return {
        "revision": EXPECTED_MODEL_REVISION,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "weight_sha256": EXPECTED_WEIGHT_SHA256,
        "weight_bytes": EXPECTED_WEIGHT_BYTES,
    }


def _container_paths() -> dict[str, str]:
    job = PurePosixPath("/opt/simllm/job")
    cache = job / "cache"
    return {
        "CUDA_CACHE_PATH": str(cache / "cuda"),
        "HF_HOME": str(cache / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(cache / "huggingface/hub"),
        "PYTHONPATH": "/opt/simllm/source/python",
        "SGLANG_CACHE_DIR": str(cache / "sglang"),
        "SIMLLM_MODEL_SNAPSHOT": "/opt/simllm/model",
        "SIMLLM_NCU_INSTALL": "/opt/simllm/ncu",
        "SIMLLM_NSYS_INSTALL": "/opt/simllm/nsys",
        "SIMLLM_RUN_ROOT": str(job / "results"),
        "SIMLLM_SCRATCH_ROOT": str(job),
        "SIMLLM_SGLANG_SOURCE": "/opt/simllm/source",
        "TEMP": str(job / "tmp"),
        "TMP": str(job / "tmp"),
        "TMPDIR": str(job / "tmp"),
        "TORCH_EXTENSIONS_DIR": str(cache / "torch-extensions"),
        "TORCH_HOME": str(cache / "torch"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "XDG_CONFIG_HOME": str(job / "xdg/config"),
        "XDG_DATA_HOME": str(job / "xdg/data"),
        "XDG_STATE_HOME": str(job / "xdg/state"),
    }


def _container_binds(
    runner: Path, source: Path, model: Path, cuda_install: Path, job: Path
) -> list[dict[str, str]]:
    return [
        {
            "role": "runner_projection",
            "source": str(runner),
            "destination": "/opt/simllm/runner",
            "mode": "ro",
        },
        {
            "role": "source_projection",
            "source": str(source),
            "destination": "/opt/simllm/source",
            "mode": "ro",
        },
        {
            "role": "model_projection",
            "source": str(model),
            "destination": "/opt/simllm/model",
            "mode": "ro",
        },
        {
            "role": "nsys_install",
            "source": str(cuda_install),
            "destination": "/opt/simllm/nsys",
            "mode": "ro",
        },
        {
            "role": "ncu_install",
            "source": str(cuda_install),
            "destination": "/opt/simllm/ncu",
            "mode": "ro",
        },
        {
            "role": "result_and_cache_roots",
            "source": str(job),
            "destination": "/opt/simllm/job",
            "mode": "rw",
        },
    ]


def _prepare_runtime(
    *,
    scratch: Path,
    out: Path,
    expected_head: str,
    inputs: Mapping[str, Any],
    gpu_uuid: str,
) -> dict[str, Any]:
    job = scratch / "job"
    projections = scratch / "projections"
    runtime = scratch / "runtime-sandbox"
    if out != job / "results":
        raise RuntimeError("--out must be exactly <scratch>/job/results for V2 containment")
    for path in (job, projections):
        if path.exists():
            raise RuntimeError(f"runtime staging path already exists: {path}")
        path.mkdir(mode=0o700)
    for relative in (
        "cache/cuda",
        "cache/huggingface/hub",
        "cache/sglang",
        "cache/torch-extensions",
        "cache/torch",
        "cache/torchinductor",
        "cache/triton",
        "cache/xdg",
        "tmp",
        "work",
        "xdg/config",
        "xdg/data",
        "xdg/state",
    ):
        (job / relative).mkdir(parents=True, mode=0o700)
    runner_projection = projections / "runner"
    source_projection = projections / "source"
    model_projection = projections / "model"
    _project_repository(expected_head, runner_projection)
    source_manifest = _project_clean_source(
        Path(inputs["paths"]["source"]), source_projection
    )
    model_manifest = _copy_model_projection(
        Path(inputs["paths"]["model"]), model_projection
    )
    apptainer = Path("/usr/bin/apptainer")
    if not apptainer.is_file() or not os.access(apptainer, os.X_OK):
        raise CapabilityBlocked("site Apptainer executable is unavailable")
    version = _run((apptainer, "--version"), timeout=60).stdout.strip()
    launcher = _launcher_environment(os.environ, scratch)
    oci = _construct_sandbox(
        apptainer,
        Path(inputs["paths"]["oci"]),
        runtime,
        launcher,
        inputs["oci"],
    )
    binds = _container_binds(
        runner_projection,
        source_projection,
        model_projection,
        Path(inputs["paths"]["cuda_install"]),
        job,
    )
    paths = _container_paths()
    container_environment = _container_environment(paths, os.environ["CUDA_VISIBLE_DEVICES"])
    mount_policy = {
        "cleanenv": True,
        "contain": True,
        "nv": True,
        "disabled_default_mounts": ["bind-paths", "cwd", "hostfs"],
        "binds": [
            {key: row[key] for key in ("role", "destination", "mode")} for row in binds
        ],
        "cwd": "/opt/simllm/job/work",
        "forbidden_host_paths": sorted(
            {
                str(Path(value).resolve())
                for value in (
                    os.environ.get("HOME", ""),
                    os.environ.get("TMPDIR", ""),
                    str(REPOSITORY_ROOT),
                    str(Path.cwd()),
                )
                if value
                and Path(value).is_absolute()
                and str(Path(value).resolve()) not in {"/tmp", "/var/tmp"}
            }
        ),
    }
    seed = {
        "schema": "simllm-sglang-a100-authority-seed-v2",
        "oci": oci,
        "source_projection": source_manifest,
        "model_projection": model_manifest,
        "mount_policy": mount_policy,
        "gpu_uuid": gpu_uuid,
    }
    _write_json(job / "authority_seed.json", seed)
    return {
        "job": job,
        "sandbox": runtime,
        "apptainer": apptainer,
        "apptainer_version": version,
        "launcher_environment": launcher,
        "container_environment": container_environment,
        "binds": binds,
        "seed": seed,
    }


def _contained_child_command(
    runtime: Mapping[str, Any], mode: str, phase: str | None, output: Path
) -> tuple[str, ...]:
    child = _child_command(mode, phase, output)
    return _container_command(
        runtime["apptainer"],
        runtime["sandbox"],
        Path("/opt/simllm/job/work"),
        runtime["binds"],
        runtime["container_environment"],
        child,
    )


def _contained_command(
    runtime: Mapping[str, Any], child_argv: Sequence[str | Path]
) -> tuple[str, ...]:
    return _container_command(
        runtime["apptainer"],
        runtime["sandbox"],
        Path("/opt/simllm/job/work"),
        runtime["binds"],
        runtime["container_environment"],
        child_argv,
    )


def _stats_csv_contained(
    runtime: Mapping[str, Any], report: Path, report_name: str
) -> str:
    command = _contained_command(
        runtime,
        (
            "/opt/simllm/nsys/bin/nsys",
            "stats",
            "--report",
            report_name,
            "--format",
            "csv",
            "--output",
            "-",
            report,
        ),
    )
    return _run(
        command,
        timeout=PROFILER_TIMEOUT_SECONDS,
        environment=runtime["launcher_environment"],
    ).stdout


def _validate_profiler_authority_before_launch(
    runtime: Mapping[str, Any], manifest: Mapping[str, Any], authority: str
) -> None:
    role = "nsys_install" if authority == "nsys" else "ncu_install"
    bind = next(row for row in runtime["binds"] if row["role"] == role)
    host_root = Path(str(bind["source"]))
    container_root = PurePosixPath(str(bind["destination"]))
    _revalidate_file_ledger(
        manifest["profilers"][authority],
        manifest_root=container_root,
        actual_root=host_root,
    )
    candidates = []
    for row in manifest["profilers"][authority]:
        path = PurePosixPath(str(row["path"]))
        if row.get("kind") == "launcher" or (
            row.get("kind") == "file" and path.name == authority
        ):
            candidates.append(row)
    if len(candidates) < 2:
        raise RuntimeError(f"{authority} profiler executable authority is incomplete")
    for row in candidates:
        try:
            relative = PurePosixPath(str(row["path"])).relative_to(container_root)
        except ValueError as error:
            raise RuntimeError(f"{authority} profiler path escaped its bind role") from error
        host_path = host_root.joinpath(*relative.parts)
        if not host_path.is_file() or host_path.is_symlink() or _sha256(host_path) != row["sha256"]:
            raise RuntimeError(f"{authority} profiler executable changed before launch")


def _cleanup_runtime(scratch: Path, out: Path) -> list[str]:
    removed = []
    candidates = (
        scratch / "runtime-sandbox",
        scratch / "projections",
        scratch / "apptainer-cache",
        scratch / "apptainer-tmp",
        scratch / "job/cache",
        scratch / "job/tmp",
        scratch / "job/work",
        scratch / "job/xdg",
    )
    for target in candidates:
        resolved = target.resolve()
        try:
            resolved.relative_to(scratch.resolve())
        except ValueError as error:
            raise RuntimeError(f"cleanup target escaped scratch: {resolved}") from error
        if resolved in {scratch.resolve(), out.resolve()} or resolved in out.resolve().parents:
            raise RuntimeError(f"cleanup target is unsafe: {resolved}")
        if target.exists():
            shutil.rmtree(target)
            removed.append(target.relative_to(scratch).as_posix())
    return removed


def _run_parent(args: argparse.Namespace) -> int:
    global _DEADLINE
    frozen = _load_expectations()
    _validate_expectations(frozen)
    if args.expected_head is None or args.out is None or args.scratch_root is None:
        raise ValueError("--expected-head, --out and --scratch-root are required")
    scratch = args.scratch_root.resolve()
    out = args.out.resolve()
    _validate_static_contract(frozen, scratch, out)
    repository = _check_repository(args.expected_head, require_clean=not args.check_only)
    if args.check_only:
        print("CHECK_ONLY=PASS")
        return 0
    if not scratch.is_dir():
        raise CapabilityBlocked(f"job scratch root is unavailable: {scratch}")
    if out.exists():
        raise RuntimeError(f"result path already exists: {out}")

    _DEADLINE = time.monotonic() + GLOBAL_DEADLINE_SECONDS
    context: dict[str, Any] = {
        "schema": "simllm-sglang-a100-kernel-pilot-result-v2",
        "state": "RUNNING",
        "repository": repository,
        "expectations_commit": EXPECTED_EXPECTATIONS_COMMIT,
        "expectations_sha256": EXPECTED_EXPECTATIONS_SHA256,
        "started_at_unix_ns": time.time_ns(),
        "lanes": {"shared": {}, "timing": {}, "nsys": {}, "ncu": {}},
    }
    nvidia_smi: Path | None = None
    gpu_selector: str | None = None
    gpu_before: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    manifest_record: dict[str, Any] | None = None
    cleanup: list[str] = []
    try:
        free_before = _validate_scratch_free(scratch)
        slurm = _check_slurm(os.environ)
        scheduler = _scheduler_snapshot(slurm["job_id"])
        environment_contract = _validate_job_environment(os.environ, scratch, out)
        nvidia_smi = _required_tool("nvidia-smi")
        gpu_selector = _job_visible_gpu_selector(slurm)
        gpu_before = _gpu_state(nvidia_smi, gpu_selector)
        inputs = _validate_input_identities(os.environ)
        runtime = _prepare_runtime(
            scratch=scratch,
            out=out,
            expected_head=args.expected_head,
            inputs=inputs,
            gpu_uuid=gpu_before["snapshot"]["uuid"],
        )
        _assert_scratch_budget(scratch)
        out.mkdir(mode=0o700)
        capture_dir = out / "capture"
        child_dir = out / "child"
        log_dir = out / "logs"
        authority_dir = out / "authority"
        for directory in (capture_dir, child_dir, log_dir, authority_dir):
            directory.mkdir(mode=0o700)
        context.update(
            {
                "slurm": slurm,
                "scheduler": scheduler,
                "environment": environment_contract,
                "inputs": inputs,
                "gpu_before": gpu_before,
                "scratch_free_bytes_before": free_before,
                "apptainer_version": runtime["apptainer_version"],
                "runtime_authority": runtime["seed"],
            }
        )
        _write_json(out / "partial_context.json", context)

        manifest_host = child_dir / "shared.manifest.json"
        manifest_container = Path("/opt/simllm/job/results/child/shared.manifest.json")
        context["lanes"]["shared"]["manifest"] = {"state": "RUNNING"}
        _write_json(out / "partial_context.json", context)
        try:
            completed = _run(
                _contained_child_command(runtime, "manifest", None, manifest_container),
                timeout=900,
                environment=runtime["launcher_environment"],
            )
            (log_dir / "shared.manifest.stdout.log").write_text(
                completed.stdout, encoding="utf-8"
            )
            (log_dir / "shared.manifest.stderr.log").write_text(
                completed.stderr, encoding="utf-8"
            )
            manifest_record = _read_json(manifest_host)
            _expect(
                manifest_record.get("schema"),
                "simllm-sglang-a100-environment-record-v2",
                "environment record schema",
            )
            manifest = manifest_record.get("manifest")
            manifest_digest = str(manifest_record.get("sha256", ""))
            if not isinstance(manifest, dict):
                raise TypeError("environment manifest child produced no manifest")
            _validate_runtime_environment_manifest(manifest)
            if _environment_manifest_digest(manifest) != manifest_digest:
                raise RuntimeError("environment manifest child digest drifted")
            _write_json(runtime["job"] / "environment_manifest.json", manifest_record)
            _write_json(authority_dir / "environment_manifest.json", manifest_record)
            _write_json(authority_dir / "authority_seed.json", runtime["seed"])
            context["environment_manifest_sha256"] = manifest_digest
            _assert_scratch_budget(scratch)
            context["lanes"]["shared"]["manifest"] = {"state": "VALID"}
            _write_json(out / "partial_context.json", context)
        except BaseException as error:
            context["lanes"]["shared"]["manifest"] = _lane_failure(error, manifest_host)
            raise

        for gate, timeout in (("cuda-gate", 60), ("import-gate", 120)):
            gate_host = child_dir / f"shared.{gate}.json"
            gate_container = Path(f"/opt/simllm/job/results/child/shared.{gate}.json")
            context["lanes"]["shared"][gate] = {"state": "RUNNING"}
            _write_json(out / "partial_context.json", context)
            try:
                completed = _run(
                    _contained_child_command(runtime, gate, None, gate_container),
                    timeout=timeout,
                    environment=runtime["launcher_environment"],
                )
                (log_dir / f"shared.{gate}.stdout.log").write_text(
                    completed.stdout, encoding="utf-8"
                )
                (log_dir / f"shared.{gate}.stderr.log").write_text(
                    completed.stderr, encoding="utf-8"
                )
                gate_value = _read_json(gate_host)
                _expect(gate_value.get("state"), "VALID", f"{gate} state")
                _expect(
                    gate_value.get("environment_manifest_sha256"),
                    manifest_digest,
                    f"{gate} environment manifest",
                )
                if gate == "cuda-gate":
                    _expect(
                        _normalize_gpu_uuid(str(gate_value.get("cuda_device_uuid", ""))),
                        _normalize_gpu_uuid(gpu_before["snapshot"]["uuid"]),
                        "CUDA gate GPU UUID",
                    )
                else:
                    _expect(
                        gate_value.get("source_version"),
                        EXPECTED_SOURCE_VERSION,
                        "import-gate source version",
                    )
                    _expect(
                        gate_value.get("entry_points"),
                        {"sglang.srt.platforms": [], "sglang.srt.plugins": []},
                        "import-gate entry points",
                    )
                context["lanes"]["shared"][gate] = {"state": "VALID"}
                context[gate.replace("-", "_")] = gate_value
                _write_json(out / "partial_context.json", context)
            except BaseException as error:
                context["lanes"]["shared"][gate] = _lane_failure(error, gate_host)
                raise

        container_root = Path("/opt/simllm/job/results")
        timing: dict[str, Any] = {}
        captures: dict[str, Any] = {}
        ledgers: dict[str, list[dict[str, Any]]] = {}
        ncu_results: dict[str, Any] = {}
        for phase in PHASES:
            context["lanes"]["timing"][phase] = {"state": "RUNNING"}
            _write_json(out / "partial_context.json", context)
            timing_path = child_dir / f"{phase}.timing.json"
            timing_container = container_root / "child" / f"{phase}.timing.json"
            try:
                completed = _run(
                    _contained_child_command(runtime, "timing", phase, timing_container),
                    timeout=PROFILER_TIMEOUT_SECONDS,
                    environment=runtime["launcher_environment"],
                )
                (log_dir / f"{phase}.timing.stdout.log").write_text(
                    completed.stdout, encoding="utf-8"
                )
                (log_dir / f"{phase}.timing.stderr.log").write_text(
                    completed.stderr, encoding="utf-8"
                )
                timing[phase] = _validate_child_result(
                    _read_json(timing_path),
                    phase=phase,
                    mode="timing",
                    expected_gpu_uuid=gpu_before["snapshot"]["uuid"],
                )
                _expect(
                    timing[phase]["environment_manifest_sha256"],
                    manifest_digest,
                    "timing environment manifest",
                )
                context["lanes"]["timing"][phase] = {"state": "VALID"}
                context["timing"] = timing
                _write_json(out / "partial_context.json", context)
            except BaseException as error:
                context["lanes"]["timing"][phase] = _lane_failure(error, timing_path)
                raise

        _validate_compatibility_control(
            out, runtime["container_environment"], IDEAL_ARTIFACT, EXPECTED_IDEAL_SHA256
        )
        prefill_median = statistics.median(
            float(row["device_ms"]) for row in timing["prefill-t512-r4"]["measurements"]
        )
        decode_median = statistics.median(
            float(row["device_ms"]) for row in timing["decode-b4-c2048"]["measurements"]
        )
        if prefill_median < 1.238933:
            raise RuntimeError("prefill device span is below the frozen peak-compute floor")
        if decode_median < 0.009679:
            raise RuntimeError("decode device span is below the frozen peak-compute floor")
        context["physical"] = {
            "prefill_median_ms": prefill_median,
            "prefill_peak_compute_floor_ms": 1.238933,
            "prefill_full_resident_hbm_reference_ms": 1.308280,
            "decode_median_ms": decode_median,
            "decode_peak_compute_floor_ms": 0.009679,
            "decode_selected_weight_kv_hbm_reference_ms": 0.617115,
            "decode_full_resident_hbm_reference_ms": 1.308280,
        }
        _write_json(out / "partial_context.json", context)

        nsys = Path("/opt/simllm/nsys/bin/nsys")
        for phase in PHASES:
            _validate_profiler_authority_before_launch(runtime, manifest, "nsys")
            context["lanes"]["nsys"][phase] = {"state": "RUNNING"}
            _write_json(out / "partial_context.json", context)
            capture_child_path = child_dir / f"{phase}.capture.json"
            capture_child_container = container_root / "child" / f"{phase}.capture.json"
            prefix_container = container_root / "capture" / phase
            report_container = prefix_container.with_suffix(".nsys-rep")
            report = capture_dir / f"{phase}.nsys-rep"
            try:
                profiler_argv = _nsys_profile_command(
                    nsys,
                    prefix_container,
                    _child_command("capture", phase, capture_child_container),
                )
                command = _contained_command(runtime, profiler_argv)
                completed = _run(
                    command,
                    timeout=PROFILER_TIMEOUT_SECONDS,
                    environment=runtime["launcher_environment"],
                )
                (log_dir / f"{phase}.nsys.stdout.log").write_text(
                    completed.stdout, encoding="utf-8"
                )
                (log_dir / f"{phase}.nsys.stderr.log").write_text(
                    completed.stderr, encoding="utf-8"
                )
                output_paths = _validate_nsys_output_paths(
                    completed.stdout + "\n" + completed.stderr,
                    Path("/opt/simllm/job"),
                    report_container,
                    Path("/opt/simllm/job/tmp"),
                )
                if not report.is_file() or report.stat().st_size == 0:
                    raise CapabilityBlocked(f"Nsight Systems produced no report for {phase}")
                cuda_csv = _stats_csv_contained(runtime, report_container, "cuda_gpu_trace")
                api_csv = _stats_csv_contained(runtime, report_container, "cuda_api_trace")
                nvtx_csv = _stats_csv_contained(runtime, report_container, "nvtx_gpu_proj_trace")
                cuda_path = capture_dir / f"{phase}.cuda_gpu_trace.csv"
                api_path = capture_dir / f"{phase}.cuda_api_trace.csv"
                nvtx_path = capture_dir / f"{phase}.nvtx_gpu_proj_trace.csv"
                cuda_path.write_text(cuda_csv, encoding="utf-8")
                api_path.write_text(api_csv, encoding="utf-8")
                nvtx_path.write_text(nvtx_csv, encoding="utf-8")
                api_rows = _parse_nsys_cuda_api_trace(api_csv)
                ranges = _parse_nvtx_projection(nvtx_csv)
                rows = _annotate_ranges(
                    _parse_nsys_cuda_trace(cuda_csv),
                    api_rows=api_rows,
                    ranges=ranges,
                    phase=phase,
                )
                rows.sort(key=lambda row: (row["start_ns"], row["ordinal"]))
                capture_child = _validate_child_result(
                    _read_json(capture_child_path),
                    phase=phase,
                    mode="capture",
                    expected_gpu_uuid=gpu_before["snapshot"]["uuid"],
                )
                _expect(
                    capture_child["environment_manifest_sha256"],
                    manifest_digest,
                    "Nsight Systems environment manifest",
                )
                ledgers[phase] = rows
                captures[phase] = {
                    **_capture_summary(phase, capture_child, rows, ranges),
                    "child": capture_child,
                    "command": list(command),
                    "nsys_output_paths": output_paths,
                    "report_sha256": _sha256(report),
                    "cuda_gpu_trace_sha256": _sha256(cuda_path),
                    "cuda_api_trace_sha256": _sha256(api_path),
                    "nvtx_gpu_projection_sha256": _sha256(nvtx_path),
                    "custom_nvtx_ranges": [
                        dict(span)
                        for span in ranges
                        if str(span["text"]).startswith(f"simllm-pilot:{phase}:")
                    ],
                }
                if (
                    timing[phase]["measurements"][0]["output_sha256"]
                    != capture_child["measurements"][0]["output_sha256"]
                ):
                    raise RuntimeError("profiled and no-capture outputs disagree")
                context["lanes"]["nsys"][phase] = {"state": "VALID"}
                context.update({"captures": captures, "ledgers": ledgers})
                _write_json(out / "partial_context.json", context)
                _assert_output_budget(out)
                _assert_scratch_budget(scratch)
            except BaseException as error:
                context["lanes"]["nsys"][phase] = _lane_failure(
                    error, capture_child_path
                )
                raise

        decode_bracket = [
            captures["decode-b4-c2048"]["kernel_count_min"],
            captures["decode-b4-c2048"]["kernel_count_max"],
        ]
        context.update(
            {
                "decode_launch_bracket": decode_bracket,
                "transfer_errors": {
                    "lower_signed_error": decode_bracket[0] - 440,
                    "upper_signed_error": decode_bracket[1] - 567,
                },
            }
        )
        _write_json(out / "partial_context.json", context)

        ncu = Path("/opt/simllm/ncu/bin/ncu")
        for phase in PHASES:
            _validate_profiler_authority_before_launch(runtime, manifest, "ncu")
            lane_evidence: dict[str, Any] = {}
            context["lanes"]["ncu"][phase] = {"state": "RUNNING"}
            _write_json(out / "partial_context.json", context)
            ncu_child_path = child_dir / f"{phase}.ncu.json"
            ncu_child_container = container_root / "child" / f"{phase}.ncu.json"
            try:
                first_range = f"simllm-pilot:{phase}:step:00"
                target_rows = [
                    row
                    for row in ledgers[phase]
                    if row.get("activity") == "kernel" and row.get("phase_range") == first_range
                ]
                target = _ncu_target(target_rows)
                profiler_argv = _ncu_command(
                    ncu,
                    target,
                    _child_command("ncu", phase, ncu_child_container),
                )
                command = _contained_command(runtime, profiler_argv)
                lane_evidence.update({"target": target, "command": list(command)})
                completed = _run(
                    command,
                    timeout=PROFILER_TIMEOUT_SECONDS,
                    check=False,
                    environment=runtime["launcher_environment"],
                )
                output = completed.stdout + "\n" + completed.stderr
                ncu_log = capture_dir / f"{phase}.ncu.csv.log"
                ncu_log.write_text(output, encoding="utf-8")
                lane_evidence["csv_sha256"] = _sha256(ncu_log)
                blocker = _ncu_blocker(output)
                if blocker is not None:
                    raise CapabilityBlocked(blocker)
                if completed.returncode != 0:
                    raise RuntimeError(f"Nsight Compute failed for {phase}: {output[-4000:]}")
                ncu_child = _validate_child_result(
                    _read_json(ncu_child_path),
                    phase=phase,
                    mode="ncu",
                    expected_gpu_uuid=gpu_before["snapshot"]["uuid"],
                )
                _expect(
                    ncu_child["environment_manifest_sha256"],
                    manifest_digest,
                    "Nsight Compute environment manifest",
                )
                if (
                    timing[phase]["measurements"][0]["output_sha256"]
                    != ncu_child["measurements"][0]["output_sha256"]
                ):
                    raise RuntimeError("NCU replay and no-capture outputs disagree")
                lane_evidence["child"] = ncu_child
                metrics = _parse_ncu_metrics(output, str(target["kernel_name"]))
                lane_evidence["metrics"] = metrics
                ncu_results[phase] = {
                    "state": "VALID",
                    "physical": _validate_ncu_physical_floor(metrics),
                    **lane_evidence,
                }
                context["lanes"]["ncu"][phase] = {"state": "VALID"}
            except BaseException as error:
                lane_state = _classify_failure(error)
                context["lanes"]["ncu"][phase] = _lane_failure(error, ncu_child_path)
                ncu_results[phase] = {
                    "state": lane_state,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    **lane_evidence,
                }
                if not isinstance(error, CapabilityBlocked):
                    raise
            context["ncu"] = ncu_results
            _write_json(out / "partial_context.json", context)
            _assert_output_budget(out)
            _assert_scratch_budget(scratch)

        flat_lanes = {
            f"{kind}:{phase}": context["lanes"][kind][phase]
            for kind in ("timing", "nsys", "ncu")
            for phase in PHASES
        }
        overall = _aggregate_lane_state(flat_lanes)
        if overall != "VALID":
            raise RuntimeError(f"required evidence aggregation returned {overall}")
        gpu_after = _gpu_state(nvidia_smi, gpu_selector)
        _validate_gpu_state_unchanged(gpu_before, gpu_after)
        inputs_after = _validate_input_identities(os.environ)
        if inputs_after != inputs:
            raise RuntimeError("pinned input identities changed during pilot")
        if _sha256(IDEAL_ARTIFACT) != EXPECTED_IDEAL_SHA256:
            raise RuntimeError("ideal compatibility artifact changed during pilot")
        context["hypotheses"] = {
            "prefill_device_span_exceeds_decode": prefill_median > decode_median,
            "prefill_busy_union_exceeds_decode": statistics.median(
                row["kernel_busy_union_ns"] for row in captures["prefill-t512-r4"]["ranges"]
            )
            > statistics.median(
                row["kernel_busy_union_ns"] for row in captures["decode-b4-c2048"]["ranges"]
            ),
        }
        context.update({"gpu_after": gpu_after, "inputs_after": inputs_after})
        _assert_scratch_budget(scratch)
        cleanup = _cleanup_runtime(scratch, out)
        context["cleanup"] = {"state": "VALID", "removed": cleanup}
        result = {**context, "state": "VALID", "finished_at_unix_ns": time.time_ns()}
        _write_json(out / "result.json", result)
        _write_json(out / "partial_context.json", result)
        _write_json(out / "manifest.json", _artifact_manifest(out))
        _assert_output_budget(out)
        print("PILOT_STATE=VALID")
        return 0
    except BaseException as error:  # noqa: BLE001 - persist frozen failure state
        state = _classify_failure(error)
        postcondition_errors = []
        try:
            out.mkdir(parents=True, exist_ok=True, mode=0o700)
            authority_dir = out / "authority"
            authority_dir.mkdir(exist_ok=True, mode=0o700)
            if runtime is not None:
                _write_json(authority_dir / "authority_seed.json", runtime["seed"])
            if manifest_record is not None:
                _write_json(authority_dir / "environment_manifest.json", manifest_record)
            if _sha256(IDEAL_ARTIFACT) != EXPECTED_IDEAL_SHA256:
                raise RuntimeError("ideal compatibility artifact changed during failed pilot")
        except BaseException as post_error:  # noqa: BLE001
            postcondition_errors.append(str(post_error))
        if nvidia_smi is not None and gpu_selector is not None and gpu_before is not None:
            try:
                gpu_after_failure = _gpu_state(nvidia_smi, gpu_selector)
                _validate_gpu_state_unchanged(gpu_before, gpu_after_failure)
                context["gpu_after"] = gpu_after_failure
            except BaseException as post_error:  # noqa: BLE001
                postcondition_errors.append(str(post_error))
        try:
            cleanup = _cleanup_runtime(scratch, out)
            context["cleanup"] = {"state": "VALID", "removed": cleanup}
        except BaseException as cleanup_error:  # noqa: BLE001
            postcondition_errors.append(f"cleanup failed: {cleanup_error}")
            context["cleanup"] = {"state": "VOID", "error": str(cleanup_error)}
        if postcondition_errors:
            state = "VOID"
        context.update(
            {
                "state": state,
                "error_type": type(error).__name__,
                "error": str(error),
                "postcondition_errors": postcondition_errors,
                "finished_at_unix_ns": time.time_ns(),
            }
        )
        _write_json(out / "result.json", context)
        _write_json(out / "partial_context.json", context)
        try:
            _write_json(out / "manifest.json", _artifact_manifest(out))
            _assert_output_budget(out)
        except BaseException as manifest_error:  # noqa: BLE001
            state = "VOID"
            postcondition_errors.append(f"artifact finalization failed: {manifest_error}")
            context["manifest_error"] = str(manifest_error)
            context["state"] = state
            context["postcondition_errors"] = postcondition_errors
            _write_json(out / "result.json", context)
            _write_json(out / "partial_context.json", context)
        print(f"PILOT_STATE={state}")
        print(f"PILOT_ERROR={error}", file=sys.stderr)
        return 2 if state == "BLOCKED" else 3


def _phase_tokens(phase: str) -> list[list[int]]:
    length = 128 if phase == "prefill-t512-r4" else 2047
    return [[_token_id(request, position) for position in range(length)] for request in range(4)]


def _forward_mode_label(value: Any, expected: str) -> str:
    name = getattr(value, "name", None)
    predicate = getattr(value, f"is_{expected}", None)
    observed = name.lower() if isinstance(name, str) else ""
    if observed != expected or not callable(predicate) or not predicate():
        raise RuntimeError(f"{expected} forward mode drifted: {value!r}")
    return observed


def _cuda_status(value: Any, operation: str) -> None:
    if value not in (None, 0):
        raise RuntimeError(f"{operation} returned CUDA status {value!r}")


def _time_cuda_target(
    cuda: Any, target: Any, instance: Any
) -> tuple[Any, dict[str, Any], float, float]:
    start = cuda.Event(enable_timing=True)
    end = cuda.Event(enable_timing=True)
    host_start = time.perf_counter_ns()
    start.record()
    output_ids, workload = target(instance)
    end.record()
    end.synchronize()
    cuda.synchronize()
    host_end = time.perf_counter_ns()
    device_ms = float(start.elapsed_time(end))
    if not math.isfinite(device_ms) or device_ms <= 0:
        raise RuntimeError("CUDA-event duration must be finite and positive")
    host_ms = (host_end - host_start) / 1_000_000.0
    if host_ms + 1e-9 < device_ms:
        raise RuntimeError("host settlement is shorter than device span")
    return output_ids, workload, device_ms, host_ms


def _distribution_inventory() -> list[dict[str, Any]]:
    rows = []
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name", "")).strip()
        version = str(distribution.version).strip()
        if not name or not version:
            raise RuntimeError("installed distribution lacks name or version")
        direct_url = distribution.read_text("direct_url.json") or ""
        if direct_url:
            try:
                direct_url = json.dumps(json.loads(direct_url), sort_keys=True, separators=(",", ":"))
            except ValueError as error:
                raise RuntimeError(f"distribution {name} has malformed direct_url.json") from error
        metadata_path = Path(str(getattr(distribution, "_path", ""))).resolve()
        rows.append(
            {
                "name": name,
                "normalized_name": re.sub(r"[-_.]+", "-", name.lower()),
                "version": version,
                "metadata_path": str(metadata_path),
                "direct_url": direct_url,
            }
        )
    rows.sort(key=lambda row: (row["normalized_name"], row["version"], row["metadata_path"]))
    return rows


def _distribution_version(rows: Sequence[Mapping[str, Any]], name: str) -> str:
    normalized = re.sub(r"[-_.]+", "-", name.lower())
    matches = [str(row["version"]) for row in rows if row["normalized_name"] == normalized]
    if len(matches) != 1:
        raise RuntimeError(f"expected one installed {name} distribution, observed {matches}")
    return matches[0]


def _stat_identity(path: Path, *, follow_symlinks: bool = True) -> dict[str, int]:
    status = path.stat() if follow_symlinks else path.lstat()
    return {
        "device": status.st_dev,
        "inode": status.st_ino,
        "mtime_ns": status.st_mtime_ns,
        "ctime_ns": status.st_ctime_ns,
    }


def _file_ledger(root: Path, authority: str) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise CapabilityBlocked(f"{authority} installation root is unavailable: {root}")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if path.is_symlink():
            target = os.readlink(path)
            payload = target.encode("utf-8")
            identity = _stat_identity(path, follow_symlinks=False)
            rows.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size": len(payload),
                    "authority": authority,
                    "kind": "symlink",
                    "target": target,
                    **identity,
                }
            )
            continue
        if not path.is_file():
            raise RuntimeError(f"{authority} installation contains a special file: {path}")
        identity = _stat_identity(path)
        rows.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "size": path.stat().st_size,
                "authority": authority,
                "kind": "file",
                **identity,
            }
        )
    if not rows:
        raise CapabilityBlocked(f"{authority} installation manifest is empty")
    return rows


def _revalidate_file_ledger(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest_root: PurePosixPath | None = None,
    actual_root: Path | None = None,
    hash_content: bool = True,
) -> None:
    if (manifest_root is None) != (actual_root is None):
        raise ValueError("manifest and actual roots must be supplied together")
    seen: set[str] = set()
    for row in rows:
        raw_manifest_path = str(row.get("path", ""))
        manifest_path = (
            Path(raw_manifest_path)
            if manifest_root is None
            else PurePosixPath(raw_manifest_path)
        )
        if not manifest_path.is_absolute() or raw_manifest_path in seen:
            raise RuntimeError("immutable file ledger path is malformed or duplicated")
        seen.add(raw_manifest_path)
        if manifest_root is None:
            actual_path = manifest_path
        else:
            try:
                relative = manifest_path.relative_to(manifest_root)
            except ValueError as error:
                raise RuntimeError("immutable file ledger escaped its manifest root") from error
            actual_path = actual_root.joinpath(*relative.parts)
        kind = str(row.get("kind", "file"))
        if kind == "symlink":
            if not actual_path.is_symlink():
                raise RuntimeError(f"immutable symlink changed: {manifest_path}")
            target = os.readlink(actual_path)
            payload = target.encode("utf-8")
            identity = _stat_identity(actual_path, follow_symlinks=False)
            if target != row.get("target"):
                raise RuntimeError(f"immutable symlink target changed: {manifest_path}")
        elif kind in {"file", "launcher"}:
            if not actual_path.is_file() or actual_path.is_symlink():
                raise RuntimeError(f"immutable file changed type: {manifest_path}")
            payload = None
            identity = _stat_identity(actual_path)
        else:
            raise RuntimeError(f"immutable file ledger kind is unknown: {kind!r}")
        size = len(payload) if payload is not None else actual_path.stat().st_size
        if size != row.get("size") or any(
            identity[field] != row.get(field) for field in STAT_IDENTITY_FIELDS
        ):
            raise RuntimeError(f"immutable file identity changed: {manifest_path}")
        digest = None
        if hash_content:
            digest = (
                hashlib.sha256(payload).hexdigest()
                if payload is not None
                else _sha256(actual_path)
            )
        if hash_content and digest != row.get("sha256"):
            raise RuntimeError(f"immutable file content changed: {manifest_path}")


def _critical_binary_ledger(distributions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    executable = Path(sys.executable).resolve()
    rows[str(executable)] = {
        "path": str(executable),
        "sha256": _sha256(executable),
        "size": executable.stat().st_size,
        "authority": "oci",
        **_stat_identity(executable),
    }
    for distribution in importlib.metadata.distributions():
        for relative in distribution.files or ():
            path = Path(distribution.locate_file(relative)).resolve()
            if not path.is_file() or path.suffix not in {".so", ".bin"} and ".so." not in path.name:
                continue
            rows[str(path)] = {
                "path": str(path),
                "sha256": _sha256(path),
                "size": path.stat().st_size,
                "authority": "oci",
                **_stat_identity(path),
            }
    required_distributions = {"torch", "triton", "sglang-kernel"}
    if not required_distributions.issubset(
        {str(row["normalized_name"]) for row in distributions}
    ):
        raise RuntimeError("critical installed distribution is absent")
    roots = (
        Path("/usr/lib/python3.12/lib-dynload"),
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/lib64"),
        Path("/usr/local/cuda/lib64"),
        Path("/usr/local/nvidia/lib"),
        Path("/usr/local/nvidia/lib64"),
        Path("/.singularity.d/libs"),
    )
    for root in roots:
        if not root.is_dir():
            continue
        for candidate in root.rglob("*.so*"):
            path = candidate.resolve()
            if not path.is_file():
                continue
            value = str(path)
            authority = (
                "driver"
                if path.name.startswith(("libcuda.", "libnvidia-", "nvidia-smi"))
                or value.startswith("/.singularity.d/libs/")
                else "oci"
            )
            rows[value] = {
                "path": value,
                "sha256": _sha256(path),
                "size": path.stat().st_size,
                "authority": authority,
                **_stat_identity(path),
            }
    for row in _loaded_object_ledger("manifest"):
        path = Path(str(row["path"]))
        rows[str(path)] = {**dict(row), **_stat_identity(path)}
    return sorted(rows.values(), key=lambda row: row["path"])


def _build_runtime_environment_manifest(seed: Mapping[str, Any]) -> dict[str, Any]:
    distributions = _distribution_inventory()
    python_version = ".".join(str(value) for value in sys.version_info[:3])
    if ".".join(str(value) for value in sys.version_info[:2]) != "3.12":
        raise RuntimeError(f"container Python ABI drifted: {python_version}")
    expected = {
        "torch": EXPECTED_RUNTIME_VERSIONS["torch"],
        "sglang": EXPECTED_RUNTIME_VERSIONS["sglang_substrate"],
        "sglang-kernel": EXPECTED_RUNTIME_VERSIONS["sglang_kernel"],
        "transformers": EXPECTED_RUNTIME_VERSIONS["transformers"],
        "triton": EXPECTED_RUNTIME_VERSIONS["triton"],
    }
    for name, version in expected.items():
        _expect(_distribution_version(distributions, name), version, f"{name} distribution")
    mount_contract = _normalized_mount_contract(_effective_mount_inventory())
    nsys_root = Path(os.environ["SIMLLM_NSYS_INSTALL"]) / "nsight-systems-2025.1.3"
    ncu_root = Path(os.environ["SIMLLM_NCU_INSTALL"]) / "nsight-compute-2025.2.1"
    nsys_files = _file_ledger(nsys_root, "nsys")
    ncu_files = _file_ledger(ncu_root, "ncu")
    for executable, authority, rows in (
        (Path(os.environ["SIMLLM_NSYS_INSTALL"]) / "bin/nsys", "nsys", nsys_files),
        (Path(os.environ["SIMLLM_NCU_INSTALL"]) / "bin/ncu", "ncu", ncu_files),
    ):
        if not executable.is_file() or executable.is_symlink():
            raise RuntimeError(f"{authority} launcher script authority drifted")
        rows.append(
            {
                "path": str(executable),
                "sha256": _sha256(executable),
                "size": executable.stat().st_size,
                "authority": authority,
                "kind": "launcher",
                **_stat_identity(executable),
            }
        )
        rows.sort(key=lambda row: row["path"])
    manifest = {
        "schema": "simllm-sglang-a100-environment-v2",
        "oci": seed["oci"],
        "distributions": distributions,
        "critical_files": _critical_binary_ledger(distributions),
        "profilers": {
            "nsys": nsys_files,
            "ncu": ncu_files,
        },
        "python": {
            "abi": "3.12",
            "version": python_version,
            "executable": str(Path(sys.executable).resolve()),
            "sha256": _sha256(Path(sys.executable).resolve()),
        },
        "source_projection": seed["source_projection"],
        "environment": {
            "fixed_overrides": dict(FIXED_CHILD_ENVIRONMENT),
            "path_overrides": {name: os.environ[name] for name in PATH_ENVIRONMENT_ROLES},
            "required_unset": list(REQUIRED_UNSET),
            "oci_config_environment": list(seed["oci"]["config_environment"]),
            "observed_residual": _normalized_residual_environment(),
        },
        "mount_policy": seed["mount_policy"],
        "mount_contract": mount_contract,
    }
    _validate_runtime_environment_manifest(manifest)
    return manifest


def _normalized_residual_environment() -> dict[str, str]:
    excluded = set(FIXED_CHILD_ENVIRONMENT) | set(PATH_ENVIRONMENT_ROLES) | {
        "CUDA_VISIBLE_DEVICES",
        "LD_PRELOAD",
        "SHLVL",
        "_",
    }
    profiler_prefixes = (
        "CUDA_INJECTION",
        "NCU_",
        "NSYS_",
        "NV_COMPUTE_PROFILER",
        "QUADD_",
    )
    return {
        name: value
        for name, value in sorted(os.environ.items())
        if name not in excluded
        and name not in REQUIRED_UNSET
        and not any(name.startswith(prefix) for prefix in profiler_prefixes)
    }


def _load_authority_seed() -> dict[str, Any]:
    path = Path(os.environ["SIMLLM_RUN_ROOT"]).parent / "authority_seed.json"
    return _read_json(path)


def _manifest_record_path() -> Path:
    return Path(os.environ["SIMLLM_RUN_ROOT"]).parent / "environment_manifest.json"


def _validate_live_environment_manifest(mode: str) -> tuple[dict[str, Any], str]:
    record = _read_json(_manifest_record_path())
    manifest = record.get("manifest")
    digest = str(record.get("sha256", ""))
    if not isinstance(manifest, dict) or _environment_manifest_digest(manifest) != digest:
        raise RuntimeError("immutable environment manifest digest drifted")
    _validate_runtime_environment_manifest(manifest)
    current = _distribution_inventory()
    if current != manifest["distributions"]:
        raise RuntimeError("installed distribution inventory changed across children")
    if _sha256(Path(sys.executable).resolve()) != manifest["python"]["sha256"]:
        raise RuntimeError("container Python executable changed across children")
    observed_mount_contract = _normalized_mount_contract(_effective_mount_inventory())
    if observed_mount_contract != manifest["mount_contract"]:
        raise RuntimeError("effective mount contract changed across children")
    _revalidate_file_ledger(manifest["critical_files"], hash_content=False)
    observed_residual = _normalized_residual_environment()
    if mode in {"capture", "ncu"} and "LD_LIBRARY_PATH" in observed_residual:
        baseline = str(
            manifest["environment"]["observed_residual"].get("LD_LIBRARY_PATH", "")
        )
        profiler_root = Path(
            os.environ["SIMLLM_NSYS_INSTALL" if mode == "capture" else "SIMLLM_NCU_INSTALL"]
        ).resolve()
        baseline_parts = [item for item in baseline.split(":") if item]
        observed_parts = [item for item in observed_residual["LD_LIBRARY_PATH"].split(":") if item]
        extras = [item for item in observed_parts if item not in baseline_parts]
        for item in extras:
            try:
                Path(item).resolve().relative_to(profiler_root)
            except ValueError as error:
                raise RuntimeError("profiler injected an unauthorized library path") from error
        observed_residual["LD_LIBRARY_PATH"] = baseline
    if observed_residual != manifest["environment"]["observed_residual"]:
        extra = sorted(set(observed_residual) - set(manifest["environment"]["observed_residual"]))
        missing = sorted(set(manifest["environment"]["observed_residual"]) - set(observed_residual))
        raise RuntimeError(
            f"closed residual environment changed across children: extra={extra}, missing={missing}"
        )
    if {name: os.environ[name] for name in PATH_ENVIRONMENT_ROLES} != manifest[
        "environment"
    ]["path_overrides"]:
        raise RuntimeError("path-role environment changed across children")
    source = manifest["source_projection"]
    root = Path(str(source["root"]))
    expected_paths = set()
    for row in source["files"]:
        path = root / str(row["path"])
        expected_paths.add(str(row["path"]))
        if row.get("mode") == "120000":
            if not path.is_symlink():
                raise RuntimeError(f"source projection symlink changed: {path}")
            digest = hashlib.sha256(os.readlink(path).encode("utf-8")).hexdigest()
        else:
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"source projection file changed: {path}")
            digest = _sha256(path)
        if digest != row["sha256"]:
            raise RuntimeError(f"source projection hash changed: {path}")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if not path.is_dir()
    }
    if actual_paths != expected_paths:
        raise RuntimeError("source projection inventory changed across children")
    for authority, install_variable, executable_name in (
        ("nsys", "SIMLLM_NSYS_INSTALL", "nsys"),
        ("ncu", "SIMLLM_NCU_INSTALL", "ncu"),
    ):
        launcher = Path(os.environ[install_variable]) / f"bin/{executable_name}"
        expected = [
            row
            for row in manifest["profilers"][authority]
            if row.get("kind") == "launcher"
        ]
        if len(expected) != 1 or str(launcher) != expected[0]["path"]:
            raise RuntimeError(f"{authority} profiler launcher path changed across children")
        if not launcher.is_file() or launcher.is_symlink() or _sha256(launcher) != expected[0][
            "sha256"
        ]:
            raise RuntimeError(f"{authority} profiler launcher changed across children")
    return manifest, digest


def _validate_loaded_objects_against_manifest(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> None:
    authorized: dict[str, tuple[str, str]] = {}
    for row in manifest["critical_files"]:
        authorized[str(row["path"])] = (str(row["sha256"]), str(row["authority"]))
    for authority, profiler_rows in manifest["profilers"].items():
        for row in profiler_rows:
            if row.get("kind") == "symlink":
                continue
            authorized[str(row["path"])] = (str(row["sha256"]), authority)
    source_root = PurePosixPath(str(manifest["source_projection"]["root"]))
    for row in manifest["source_projection"]["files"]:
        authorized[str(source_root / str(row["path"]))] = (str(row["sha256"]), "source")
    runner_path = Path(__file__).resolve()
    authorized[str(runner_path)] = (_sha256(runner_path), "runner")
    for row in rows:
        authority = str(row["authority"])
        if authority == "generated_cache":
            continue
        if authority == "model":
            path = Path(str(row["path"]))
            if path.suffix == ".safetensors" and row["sha256"] == EXPECTED_WEIGHT_SHA256:
                continue
            if path.name == "config.json" and row["sha256"] == EXPECTED_CONFIG_SHA256:
                continue
            raise RuntimeError(f"loaded model object has no frozen authority: {path}")
        expected = authorized.get(str(row["path"]))
        if expected != (str(row["sha256"]), authority):
            raise RuntimeError(
                f"loaded object does not join immutable authority manifest: {row['path']}"
            )


def _entry_point_inventory() -> dict[str, list[dict[str, str]]]:
    result = {}
    all_entry_points = importlib.metadata.entry_points()
    for group in ("sglang.srt.platforms", "sglang.srt.plugins"):
        selected = all_entry_points.select(group=group)
        result[group] = sorted(
            (
                {
                    "name": item.name,
                    "value": item.value,
                    "group": item.group,
                }
                for item in selected
            ),
            key=lambda row: (row["name"], row["value"]),
        )
    return result


def _module_ledger() -> list[dict[str, Any]]:
    rows = []
    for name, module in sorted(sys.modules.items()):
        raw = getattr(module, "__file__", None)
        if not raw:
            continue
        path = Path(str(raw)).resolve()
        if path.suffix in {".pyc", ".pyo"}:
            source = path.with_suffix("")
            if source.is_file():
                path = source
        if not path.is_file():
            continue
        rows.append({"name": name, "path": str(path), "sha256": _sha256(path)})
    return rows


def _loaded_object_ledger(mode: str) -> list[dict[str, Any]]:
    paths: set[str] = set()
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or "x" not in fields[1] or not fields[5].startswith("/"):
            continue
        paths.add(fields[5])
    rows = []
    for raw in sorted(paths):
        if raw.endswith(" (deleted)"):
            raise RuntimeError(f"loaded object was deleted while mapped: {raw}")
        path = Path(raw)
        if not path.is_file():
            continue
        value = str(path)
        if value.startswith("/opt/simllm/source/"):
            authority = "source"
        elif value.startswith("/opt/simllm/runner/"):
            authority = "runner"
        elif value.startswith("/opt/simllm/model/"):
            authority = "model"
        elif value.startswith("/opt/simllm/nsys/"):
            authority = "nsys"
        elif value.startswith("/opt/simllm/ncu/"):
            authority = "ncu"
        elif Path(value).name.startswith(("libcuda.", "libnvidia-", "nvidia-smi")):
            authority = "driver"
        elif value.startswith("/opt/simllm/job/cache/"):
            authority = "generated_cache"
        elif value.startswith("/opt/simllm/job/"):
            raise RuntimeError(f"loaded object came from unaudited mutable job state: {value}")
        else:
            authority = "oci"
        if authority == "nsys" and mode != "capture":
            raise RuntimeError("Nsight Systems object loaded outside capture child")
        if authority == "ncu" and mode != "ncu":
            raise RuntimeError("Nsight Compute object loaded outside NCU child")
        rows.append(
            {
                "path": value,
                "sha256": _sha256(path),
                "size": path.stat().st_size,
                "authority": authority,
            }
        )
    _validate_loaded_object_ledger(rows)
    return rows


def _validate_generated_object_authority(
    rows: Sequence[Mapping[str, Any]],
    cache_roots: Mapping[str, Path],
    inventories: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    expected = {
        (name, str(item["path"])): str(item["sha256"])
        for name, inventory in inventories.items()
        for item in inventory
    }
    for row in rows:
        if row.get("authority") != "generated_cache":
            continue
        path = Path(str(row["path"])).resolve()
        matches = []
        for name, root in cache_roots.items():
            try:
                relative = path.relative_to(root.resolve()).as_posix()
            except ValueError:
                continue
            matches.append((name, relative))
        if len(matches) != 1:
            raise RuntimeError(f"generated loaded object has ambiguous cache authority: {path}")
        if expected.get(matches[0]) != row["sha256"]:
            raise RuntimeError(f"generated loaded object is absent from stable cache inventory: {path}")


def _validate_child_environment_contract(mode: str) -> None:
    for name, expected in FIXED_CHILD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"contained {name} drifted")
    for name in REQUIRED_UNSET:
        if name in os.environ:
            raise RuntimeError(f"contained {name} must be absent")
    for name in PATH_ENVIRONMENT_ROLES:
        raw = os.environ.get(name, "")
        logical_path = PurePosixPath(raw)
        if not raw or not logical_path.is_absolute() or ".." in logical_path.parts:
            raise RuntimeError(f"contained path role {name} drifted")
    permitted_sglang = {
        "SGLANG_BUILD_COMMIT",
        "SGLANG_BUILD_URL",
        "SGLANG_CACHE_DIR",
        "SGLANG_IMAGE_TAG",
        "SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL",
    }
    for name in os.environ:
        upper = name.upper()
        if name.startswith("SLURM_"):
            raise RuntimeError(f"contained scheduler variable is forbidden: {name}")
        if (
            any(marker in upper for marker in ("AWS_", "SECRET", "CREDENTIAL", "AUTH_TOKEN"))
            or upper in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}
        ):
            raise RuntimeError(f"contained credential or preload variable is forbidden: {name}")
        if name.startswith("SGLANG_") and name not in permitted_sglang:
            raise RuntimeError(f"contained arbitrary SGLang control is forbidden: {name}")
    if "LD_PRELOAD" in os.environ and mode not in {"capture", "ncu"}:
        raise RuntimeError("contained LD_PRELOAD is forbidden outside profiler children")


def _profiler_injection_environment(
    mode: str, manifest: Mapping[str, Any]
) -> dict[str, str]:
    prefixes = (
        "CUDA_INJECTION",
        "NCU_",
        "NSYS_",
        "NV_COMPUTE_PROFILER",
        "QUADD_",
    )
    result = {
        name: value
        for name, value in os.environ.items()
        if name == "LD_PRELOAD" or any(name.startswith(prefix) for prefix in prefixes)
    }
    if mode not in {"capture", "ncu"}:
        if result:
            raise RuntimeError("profiler injection environment escaped profiler child")
        return {}
    authority = "nsys" if mode == "capture" else "ncu"
    install = Path(
        os.environ["SIMLLM_NSYS_INSTALL" if mode == "capture" else "SIMLLM_NCU_INSTALL"]
    ).resolve()
    authorized = {
        str(row["path"]): str(row["sha256"])
        for row in manifest["profilers"][authority]
        if row.get("kind") != "symlink"
    }
    preload = result.get("LD_PRELOAD", "")
    for item in filter(None, preload.split(":")):
        path = Path(item).resolve()
        try:
            path.relative_to(install)
        except ValueError as error:
            raise RuntimeError("profiler LD_PRELOAD escaped exact profiler install") from error
        if authorized.get(str(path)) != _sha256(path):
            raise RuntimeError("profiler LD_PRELOAD lacks exact manifest authority")
    return dict(sorted(result.items()))


def _mount_point_is_allowed(mount_point: str) -> bool:
    return (
        mount_point in REQUIRED_CONTAINER_MOUNT_MODES
        or mount_point in ALLOWED_SYSTEM_MOUNT_POINTS
        or mount_point == "/.singularity.d/libs"
        or any(mount_point.startswith(prefix) for prefix in ALLOWED_SYSTEM_MOUNT_PREFIXES)
    )


def _normalized_mount_contract(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "mount_point": str(row["mount_point"]),
                "mount_options": sorted(
                    str(item)
                    for item in row["mount_options"]
                    if item in MOUNT_CONTRACT_OPTIONS
                ),
                "filesystem_type": str(row["filesystem_type"]),
                "super_options": sorted(
                    str(item)
                    for item in row["super_options"]
                    if item in MOUNT_CONTRACT_OPTIONS
                ),
            }
            for row in rows
        ),
        key=lambda row: _canonical_json(row),
    )


def _effective_mount_inventory() -> list[dict[str, Any]]:
    rows = []
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            raise RuntimeError("contained mountinfo row is malformed")
        left_fields = left.split()
        right_fields = right.split()
        if len(left_fields) < 6 or len(right_fields) < 3:
            raise RuntimeError("contained mountinfo row lacks required fields")
        rows.append(
            {
                "mount_id": int(left_fields[0]),
                "parent_id": int(left_fields[1]),
                "major_minor": left_fields[2],
                "root": left_fields[3],
                "mount_point": left_fields[4],
                "mount_options": left_fields[5].split(","),
                "optional_fields": left_fields[6:],
                "filesystem_type": right_fields[0],
                "source": right_fields[1],
                "super_options": right_fields[2].split(","),
            }
        )
    observed = [str(row["mount_point"]) for row in rows]
    missing = sorted(set(REQUIRED_CONTAINER_MOUNT_MODES) - set(observed))
    if missing:
        raise RuntimeError(f"effective mount inventory lacks explicit roles: {missing}")
    unexpected = sorted(mount for mount in observed if not _mount_point_is_allowed(mount))
    if unexpected:
        raise RuntimeError(f"effective mount inventory contains unexpected user binds: {unexpected}")
    by_mount: dict[str, dict[str, Any]] = {}
    for mount in REQUIRED_CONTAINER_MOUNT_MODES:
        matches = [row for row in rows if row["mount_point"] == mount]
        if len(matches) != 1:
            raise RuntimeError(f"effective bind role is duplicated: {mount}")
        by_mount[mount] = matches[0]
    for mount, expected_mode in REQUIRED_CONTAINER_MOUNT_MODES.items():
        options = set(by_mount[mount]["mount_options"])
        opposite = "rw" if expected_mode == "ro" else "ro"
        if expected_mode not in options or opposite in options:
            if expected_mode == "rw":
                raise RuntimeError("job result/cache bind is not writable")
            raise RuntimeError(f"read-only bind is writable in effective mount inventory: {mount}")
    policy = _load_authority_seed()["mount_policy"]
    serialized = json.dumps(
        [{"root": row["root"], "source": row["source"]} for row in rows], sort_keys=True
    )
    leaked = [path for path in policy["forbidden_host_paths"] if path in serialized]
    if leaked:
        raise RuntimeError(f"effective mount inventory exposes forbidden host paths: {leaked}")
    return rows


def _run_cuda_gate(args: argparse.Namespace, record_progress: Any) -> int:
    manifest, digest = _validate_live_environment_manifest("cuda-gate")
    record_progress("runtime_import_started", package="torch")
    import torch

    record_progress("runtime_import_completed", package="torch")
    seed = _load_authority_seed()
    distributions = manifest["distributions"]
    versions = {
        "python": manifest["python"]["version"],
        "python_abi": manifest["python"]["abi"],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "sglang_substrate": _distribution_version(distributions, "sglang"),
        "sglang_kernel": _distribution_version(distributions, "sglang-kernel"),
        "transformers": _distribution_version(distributions, "transformers"),
        "triton": _distribution_version(distributions, "triton"),
    }
    expected = {key: value for key, value in EXPECTED_RUNTIME_VERSIONS.items() if key != "sglang_source"}
    _expect({key: versions[key] for key in expected}, expected, "CUDA gate packages")
    if not torch.cuda.is_available():
        raise CapabilityBlocked("exact CUDA 12.9 runtime cannot initialize the allocated GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"CUDA gate observed {torch.cuda.device_count()} GPUs")
    name = torch.cuda.get_device_name(0)
    capability = ".".join(str(value) for value in torch.cuda.get_device_capability(0))
    if name != EXPECTED_GPU_NAME or capability != EXPECTED_COMPUTE_CAPABILITY:
        raise RuntimeError(f"CUDA gate GPU identity drifted: {name}, {capability}")
    tensor = torch.arange(16, device="cuda", dtype=torch.float32)
    observed_sum = float(tensor.sum().item())
    torch.cuda.synchronize()
    if observed_sum != 120.0:
        raise RuntimeError(f"CUDA gate reduction drifted: {observed_sum}")
    cuda_uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    expected_uuid = str(seed["gpu_uuid"])
    if _normalize_gpu_uuid(cuda_uuid) != _normalize_gpu_uuid(expected_uuid):
        raise RuntimeError("CUDA gate and host GPU UUIDs disagree")
    loaded_objects = _loaded_object_ledger("cuda-gate")
    _validate_loaded_objects_against_manifest(loaded_objects, manifest)
    result = {
        "schema": "simllm-sglang-a100-cuda-gate-v2",
        "state": "VALID",
        "versions": versions,
        "cuda_device_name": name,
        "cuda_device_uuid": cuda_uuid,
        "compute_capability": capability,
        "reduction_sum": observed_sum,
        "environment_manifest_sha256": digest,
        "modules": _module_ledger(),
        "loaded_objects": loaded_objects,
        "mounts": _effective_mount_inventory(),
        "profiler_environment": _profiler_injection_environment("cuda-gate", manifest),
    }
    _write_json(args.child_out, result)
    record_progress("child_result_written")
    print("CHILD_MODE=cuda-gate")
    print("CHILD_STATE=PASS")
    return 0


def _run_manifest_child(args: argparse.Namespace, record_progress: Any) -> int:
    record_progress("environment_manifest_started")
    manifest = _build_runtime_environment_manifest(_load_authority_seed())
    digest = _environment_manifest_digest(manifest)
    result = {
        "schema": "simllm-sglang-a100-environment-record-v2",
        "manifest": manifest,
        "sha256": digest,
        "mounts": _effective_mount_inventory(),
    }
    _write_json(args.child_out, result)
    record_progress("environment_manifest_completed", sha256=digest)
    print("CHILD_MODE=manifest")
    print("CHILD_STATE=PASS")
    return 0


def _run_import_gate(args: argparse.Namespace, record_progress: Any) -> int:
    manifest, digest = _validate_live_environment_manifest("import-gate")
    preimport_objects = _loaded_object_ledger("import-gate")
    _validate_loaded_objects_against_manifest(preimport_objects, manifest)
    record_progress("runtime_import_started", package="sglang")
    try:
        import sglang
        import sglang.benchmark.one_batch as one_batch_module
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.models.granitemoe import GraniteMoeForCausalLM
    except (ImportError, OSError) as error:
        raise CapabilityBlocked(f"exact SGLang source/runtime ABI import failed: {error}") from error
    record_progress("runtime_import_completed", package="sglang")
    source_version = str(getattr(sglang, "__version__", ""))
    if source_version != EXPECTED_SOURCE_VERSION:
        raise RuntimeError(f"source SGLang version drifted: {source_version!r}")
    if ModelRunner.__module__ != "sglang.srt.model_executor.model_runner":
        raise RuntimeError("stock ModelRunner authority drifted")
    if GraniteMoeForCausalLM.__module__ != "sglang.srt.models.granitemoe":
        raise RuntimeError("stock GraniteMoe model authority drifted")
    modules = _module_ledger()
    entry_points = _entry_point_inventory()
    source_manifest = dict(manifest["source_projection"])
    source_manifest["package_paths"] = [str(Path(item).resolve()) for item in sglang.__path__]
    _validate_source_projection(source_manifest, modules, entry_points)
    one_batch_path = Path(str(getattr(one_batch_module, "__file__", ""))).resolve()
    try:
        one_batch_relative = one_batch_path.relative_to(
            Path(os.environ["SIMLLM_SGLANG_SOURCE"]).resolve()
        )
    except ValueError as error:
        raise RuntimeError("one-batch helper escaped the source projection") from error
    objects = _loaded_object_ledger("import-gate")
    _validate_loaded_objects_against_manifest(objects, manifest)
    result = {
        "schema": "simllm-sglang-a100-import-gate-v2",
        "state": "VALID",
        "environment_manifest_sha256": digest,
        "source_version": source_version,
        "source": {
            "commit": EXPECTED_SGLANG_COMMIT,
            "tree": EXPECTED_SGLANG_TREE,
            "package_paths": list(sglang.__path__),
            "one_batch": one_batch_relative.as_posix(),
        },
        "entry_points": entry_points,
        "modules": modules,
        "loaded_objects": objects,
        "mounts": _effective_mount_inventory(),
        "profiler_environment": _profiler_injection_environment("import-gate", manifest),
    }
    _write_json(args.child_out, result)
    record_progress("child_result_written")
    print("CHILD_MODE=import-gate")
    print("CHILD_STATE=PASS")
    return 0


def _run_child(args: argparse.Namespace) -> int:
    if args.child_out is None or args.child is None:
        raise ValueError("child mode requires --child-out")
    if args.child in {"timing", "capture", "ncu"} and args.phase is None:
        raise ValueError("measurement child mode requires --phase")
    progress_phase = args.phase or "shared"

    def record_progress(stage: str, **details: Any) -> None:
        _append_child_progress(
            args.child_out,
            mode=args.child,
            phase=progress_phase,
            stage=stage,
            **details,
        )

    _validate_child_environment_contract(args.child)
    source_path = _resolve_required_path(os.environ, "SIMLLM_SGLANG_SOURCE")
    scratch_root = _resolve_required_path(os.environ, "SIMLLM_SCRATCH_ROOT")
    temporary_root = _resolve_required_path(os.environ, "TMPDIR")
    cache_roots = {name: _resolve_required_path(os.environ, name) for name in AUDITED_CHILD_ROOTS}
    for name, path in (*cache_roots.items(), ("TMPDIR", temporary_root)):
        if not Path(os.environ[name]).is_absolute():
            raise RuntimeError(f"{name} must be absolute")
        try:
            path.relative_to(scratch_root)
        except ValueError as error:
            raise RuntimeError(f"{name} escapes the job scratch root") from error
    try:
        args.child_out.resolve().relative_to(scratch_root)
    except ValueError as error:
        raise RuntimeError("child output escapes the job scratch root") from error
    progress_path = _child_progress_path(args.child_out)
    try:
        progress_path.resolve().relative_to(scratch_root)
    except ValueError as error:
        raise RuntimeError("child progress escapes the job scratch root") from error
    if args.child_out.exists():
        raise RuntimeError(f"child output already exists: {args.child_out}")
    if progress_path.exists():
        raise RuntimeError(f"child progress already exists: {progress_path}")

    record_progress("entered")
    record_progress("preflight_validated")
    if args.child == "manifest":
        return _run_manifest_child(args, record_progress)
    if args.child == "cuda-gate":
        return _run_cuda_gate(args, record_progress)
    if args.child == "import-gate":
        return _run_import_gate(args, record_progress)
    environment_manifest, environment_manifest_digest = _validate_live_environment_manifest(
        args.child
    )

    record_progress("runtime_import_started")
    import sglang
    import sglang.benchmark.one_batch as one_batch_module
    import torch
    from sglang.benchmark.one_batch import (
        _set_envs_and_config,
        initialize_fp4_gemm_config,
        initialize_fp8_gemm_config,
        initialize_moe_config,
        load_model,
        prepare_synthetic_inputs_for_latency_test,
    )
    from sglang.srt.layers.moe import get_moe_a2a_backend, get_moe_runner_backend
    from sglang.srt.runtime_context import get_exec
    from sglang.srt.server_args import PortArgs, ServerArgs

    record_progress("runtime_import_completed")

    versions = {
        "python_abi": ".".join(str(item) for item in sys.version_info[:2]),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "sglang_source": str(getattr(sglang, "__version__", "")),
        "sglang_substrate": importlib.metadata.version("sglang"),
        "sglang_kernel": importlib.metadata.version("sglang-kernel"),
        "transformers": importlib.metadata.version("transformers"),
        "triton": importlib.metadata.version("triton"),
    }
    if versions != EXPECTED_RUNTIME_VERSIONS:
        raise RuntimeError(f"runtime package identity drifted: {versions!r}")
    entry_points = _entry_point_inventory()
    module_ledger = _module_ledger()
    source_manifest = dict(environment_manifest["source_projection"])
    source_manifest["package_paths"] = [str(Path(item).resolve()) for item in sglang.__path__]
    _validate_source_projection(source_manifest, module_ledger, entry_points)
    source_files = {}
    for name, imported_file in (
        ("package", getattr(sglang, "__file__", None)),
        ("one_batch", getattr(one_batch_module, "__file__", None)),
    ):
        if imported_file is None:
            raise RuntimeError(f"imported SGLang {name} has no source file")
        try:
            relative = Path(imported_file).resolve().relative_to(source_path)
        except ValueError as error:
            raise RuntimeError(f"imported SGLang {name} is outside the pinned checkout") from error
        source_files[name] = relative.as_posix()
    record_progress("runtime_identity_validated", versions=versions, source=source_files)
    model_path = _resolve_required_path(os.environ, "SIMLLM_MODEL_SNAPSHOT")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise CapabilityBlocked("the child does not see exactly one CUDA GPU")
    if torch.cuda.get_device_name(0) != EXPECTED_GPU_NAME:
        raise RuntimeError(f"CUDA device drifted: {torch.cuda.get_device_name(0)!r}")
    capability = ".".join(str(item) for item in torch.cuda.get_device_capability(0))
    if capability != EXPECTED_COMPUTE_CAPABILITY:
        raise RuntimeError(f"CUDA compute capability drifted: {capability}")
    record_progress(
        "cuda_identity_validated",
        cuda_device_name=torch.cuda.get_device_name(0),
        compute_capability=capability,
    )

    server_args = ServerArgs(
        model_path=str(model_path),
        tokenizer_path=str(model_path),
        load_format="safetensors",
        trust_remote_code=False,
        dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        quantization=None,
        tp_size=1,
        pp_size=1,
        dp_size=1,
        ep_size=1,
        moe_dp_size=1,
        mem_fraction_static=0.60,
        chunked_prefill_size=-1,
        page_size=1,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        disable_cuda_graph=True,
        attention_backend="triton",
        sampling_backend="pytorch",
        moe_a2a_backend="none",
        moe_runner_backend="triton",
        random_seed=173,
    )
    _set_envs_and_config(server_args)
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)
    record_progress("backend_configuration_initialized")
    if get_moe_runner_backend().value != "triton":
        raise RuntimeError("runtime MoE runner backend is not Triton")
    if get_moe_a2a_backend().value != "none":
        raise RuntimeError("runtime MoE A2A backend is not none")
    port_args = PortArgs.init_new(server_args)
    record_progress("model_load_started")
    model_runner, _tokenizer = load_model(server_args, port_args, 0, 0)
    record_progress("model_load_completed")
    torch_runner = getattr(model_runner, "torch_runner", None)
    if torch_runner is None:
        raise RuntimeError("one-batch helper did not return the stock Torch runner")
    runner_identity = f"{type(torch_runner).__module__}.{type(torch_runner).__name__}"
    if runner_identity != "sglang.srt.model_executor.model_runner.ModelRunner":
        raise RuntimeError(f"runtime runner identity drifted: {runner_identity}")
    model_identity = f"{type(torch_runner.model).__module__}.{type(torch_runner.model).__name__}"
    if not model_identity.endswith("granitemoe.GraniteMoeForCausalLM"):
        raise RuntimeError(f"runtime model identity drifted: {model_identity}")
    attention_identity = (
        f"{type(torch_runner.attn_backend).__module__}.{type(torch_runner.attn_backend).__name__}"
    )
    if attention_identity != "sglang.srt.layers.attention.triton_backend.TritonAttnBackend":
        raise RuntimeError(f"runtime attention backend drifted: {attention_identity}")
    if get_exec().kernel.sampling_backend != "pytorch":
        raise RuntimeError("runtime sampling backend is not PyTorch")
    backends = {
        "attention": "triton",
        "moe_a2a": get_moe_a2a_backend().value,
        "moe_runner": get_moe_runner_backend().value,
        "sampling": get_exec().kernel.sampling_backend,
    }
    record_progress("runtime_backend_validated", backends=backends)

    layer0 = torch_runner.model.model.layers[0]
    active_phase = args.phase

    def _pre_hook(label: str):
        def hook(_module: Any, _inputs: Any) -> None:
            torch.cuda.nvtx.range_push(f"simllm-pilot:{active_phase}:{label}")

        return hook

    def _post_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        torch.cuda.nvtx.range_pop()
        return output

    handles = (
        layer0.self_attn.qkv_proj.register_forward_pre_hook(_pre_hook("layer-0-qkv")),
        layer0.self_attn.qkv_proj.register_forward_hook(_post_hook),
        layer0.block_sparse_moe.experts.register_forward_pre_hook(_pre_hook("layer-0-fused-moe")),
        layer0.block_sparse_moe.experts.register_forward_hook(_post_hook),
    )

    tokens = _phase_tokens(args.phase)
    from sglang.srt.sampling.sampling_params import SamplingParams

    def prepare() -> tuple[Any, Any, Any]:
        model_runner.clear()
        reqs = prepare_synthetic_inputs_for_latency_test(4, len(tokens[0]), tokens)
        if len(reqs) != 4:
            raise RuntimeError(f"request count drifted: {len(reqs)}")
        for index, req in enumerate(reqs):
            req.rid = f"{active_phase}-r{index}"
            req.sampling_params = SamplingParams(
                temperature=0,
                max_new_tokens=1 if active_phase == "prefill-t512-r4" else 2,
                ignore_eos=True,
            )
        if active_phase == "prefill-t512-r4":
            return reqs, None, None
        next_token_ids, _logits, batch = model_runner.extend(reqs)
        torch.cuda.synchronize()
        return reqs, next_token_ids, batch

    def target(instance: tuple[Any, Any, Any]) -> tuple[Any, dict[str, Any]]:
        reqs, next_token_ids, batch = instance
        if active_phase == "prefill-t512-r4":
            output_ids, _logits, prepared_batch = model_runner.extend(reqs)
            forward_mode = _forward_mode_label(prepared_batch.forward_mode, "extend")
            extend_tokens = int(getattr(prepared_batch, "extend_num_tokens", -1))
            if extend_tokens != 512:
                raise RuntimeError(f"prefill extend token count drifted: {extend_tokens}")
            seq_lens = [int(item) for item in prepared_batch.seq_lens_cpu.tolist()]
            if seq_lens != [128] * 4:
                raise RuntimeError(f"prefill sequence lengths drifted: {seq_lens}")
            return output_ids, {
                "batch_size": 4,
                "forward_mode": forward_mode,
                "input_tokens_per_request": 128,
                "kv_slots_allocated": int(prepared_batch.out_cache_loc.numel()),
                "request_ids": [req.rid for req in reqs],
                "request_pool_indices": [int(req.req_pool_idx) for req in reqs],
                "scheduled_tokens": extend_tokens,
                "seq_lens": seq_lens,
            }
        output_ids, _logits = model_runner.decode(next_token_ids, batch)
        forward_mode = _forward_mode_label(batch.forward_mode, "decode")
        seq_lens = [int(item) for item in batch.seq_lens_cpu.tolist()]
        if seq_lens != [2048] * 4:
            raise RuntimeError(f"decode sequence lengths drifted: {seq_lens}")
        return output_ids, {
            "batch_size": 4,
            "forward_mode": forward_mode,
            "input_tokens_per_request": 2047,
            "kv_slots_allocated": int(batch.out_cache_loc.numel()),
            "request_ids": [req.rid for req in reqs],
            "request_pool_indices": [int(req.req_pool_idx) for req in reqs],
            "scheduled_tokens": 4,
            "seq_lens": seq_lens,
        }

    record_progress("warmups_started", count=WARMUPS)
    for warmup in range(WARMUPS):
        watchdog = _start_step_watchdog(f"{active_phase}:warmup:{warmup:02d}")
        try:
            target(prepare())
            torch.cuda.synchronize()
        finally:
            _cancel_step_watchdog(watchdog)
    record_progress("warmups_completed", count=WARMUPS)

    cache_before = {name: _cache_inventory(root) for name, root in cache_roots.items()}
    measurements = []
    repetitions = (
        TIMING_REPETITIONS
        if args.child == "timing"
        else (CAPTURE_REPETITIONS if args.child == "capture" else 1)
    )
    record_progress("retained_measurement_started", repetitions=repetitions)
    for repetition in range(repetitions):
        watchdog = _start_step_watchdog(f"{active_phase}:{args.child}:{repetition:02d}")
        try:
            instance = prepare()
            torch.cuda.synchronize()
            if args.child in {"capture", "ncu"}:
                _cuda_status(torch.cuda.cudart().cudaProfilerStart(), "cudaProfilerStart")
            torch.cuda.nvtx.range_push(f"simllm-pilot:{active_phase}:step:{repetition:02d}")
            output_ids, workload, device_ms, host_ms = _time_cuda_target(
                torch.cuda, target, instance
            )
            torch.cuda.nvtx.range_pop()
            if args.child in {"capture", "ncu"}:
                _cuda_status(torch.cuda.cudart().cudaProfilerStop(), "cudaProfilerStop")
        finally:
            _cancel_step_watchdog(watchdog)
        values = [int(item) for item in output_ids.detach().cpu().tolist()]
        checksum = hashlib.sha256(json.dumps(values).encode("utf-8")).hexdigest()
        measurements.append(
            {
                "repetition": repetition,
                "device_ms": device_ms,
                "host_ms": host_ms,
                "output_ids": values,
                "output_sha256": checksum,
                **workload,
            }
        )
    record_progress("retained_measurement_completed", repetitions=repetitions)
    cache_after = {name: _cache_inventory(root) for name, root in cache_roots.items()}
    if cache_before != cache_after:
        raise RuntimeError("Triton cache changed during retained measurements")
    for handle in handles:
        handle.remove()

    module_ledger = _module_ledger()
    source_manifest["package_paths"] = [str(Path(item).resolve()) for item in sglang.__path__]
    _validate_source_projection(source_manifest, module_ledger, entry_points)
    loaded_objects = _loaded_object_ledger(args.child)
    _validate_generated_object_authority(loaded_objects, cache_roots, cache_after)
    _validate_loaded_objects_against_manifest(loaded_objects, environment_manifest)
    cuda_uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
    if cuda_uuid is None:
        raise CapabilityBlocked("PyTorch does not expose the CUDA device UUID")
    result = {
        "schema": "simllm-sglang-a100-kernel-pilot-child-v2",
        "mode": args.child,
        "phase": args.phase,
        "versions": versions,
        "source": {
            "commit": EXPECTED_SGLANG_COMMIT,
            "tree": EXPECTED_SGLANG_TREE,
            **source_files,
        },
        "backends": backends,
        "runner_identity": runner_identity,
        "model_identity": model_identity,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_device_uuid": str(cuda_uuid),
        "environment_manifest_sha256": environment_manifest_digest,
        "entry_points": entry_points,
        "modules": module_ledger,
        "loaded_objects": loaded_objects,
        "mounts": _effective_mount_inventory(),
        "profiler_environment": _profiler_injection_environment(
            args.child, environment_manifest
        ),
        "measurements": measurements,
        "cache_inventory": cache_after,
    }
    _write_json(args.child_out, result)
    record_progress("child_result_written")
    print(f"CHILD_MODE={args.child}")
    print(f"CHILD_PHASE={args.phase}")
    print("CHILD_STATE=PASS")
    return 0


def main() -> int:
    args = parse_args()
    if args.child is not None:
        return _run_child(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
