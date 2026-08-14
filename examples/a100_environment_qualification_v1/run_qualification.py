"""Run the frozen A100 profiler-environment qualification.

This is capability evidence for COMP-5, not a compute calibration. All CUDA,
profiler and binary-tool commands must run inside the reviewed Slurm GPU job.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTATIONS_PATH = Path(__file__).with_name("expectations.md")
PROBE_SOURCE = REPOSITORY_ROOT / "tools" / "compute_capture" / "a100_environment_probe.cu"

BASE_SIMLLM_SHA = "dddf8fbf70e2b168dcd43ccf6799496d1ab9be11"
EXPECTED_GPU_NAME = "NVIDIA A100-SXM4-80GB"
EXPECTED_COMPUTE_CAPABILITY = "8.0"
EXPECTED_CLUSTER = "gmerlin7"
EXPECTED_PARTITION = "a100-hourly"
EXPECTED_CUDA_MODULE = "cuda/12.9.1"
TARGET_KERNEL = "simllm_a100_environment_probe_vector_add_fp32_v1"
EXPECTED_ELEMENTS = 16_777_216
EXPECTED_THREADS = 256
EXPECTED_WARMUPS = 5
EXPECTED_MEASURED = 1
PROFILER_TIMEOUT_SECONDS = 180
GLOBAL_DEADLINE_SECONDS = 17 * 60
MAX_OUTPUT_BYTES = 10 * 1024**3
MAX_SCRATCH_BYTES = 10 * 1024**3
_QUALIFICATION_DEADLINE: float | None = None
_QUALIFICATION_ENVIRONMENT: dict[str, str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _run(
    command: Sequence[str | Path],
    *,
    timeout: int = 300,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    effective_timeout = float(timeout)
    if _QUALIFICATION_DEADLINE is not None:
        remaining = _QUALIFICATION_DEADLINE - time.monotonic()
        if remaining <= 0.5:
            raise TimeoutError("A100 qualification reached its 17-minute deadline")
        effective_timeout = min(effective_timeout, remaining)
    normalized = [str(item) for item in command]
    environment = None
    if _QUALIFICATION_ENVIRONMENT is not None:
        environment = os.environ.copy()
        environment.update(_QUALIFICATION_ENVIRONMENT)
    try:
        completed = subprocess.run(
            normalized,
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            env=environment,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        raise RuntimeError(
            f"command timed out after {effective_timeout:.1f} seconds: "
            f"{' '.join(normalized)}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        ) from error
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_size(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _assert_output_budget(path: Path) -> int:
    size = _tree_size(path)
    if size > MAX_OUTPUT_BYTES:
        raise RuntimeError(f"qualification output exceeds 10 GiB: {size} bytes")
    return size


def _assert_scratch_budget(path: Path) -> int:
    size = _tree_size(path)
    if size > MAX_SCRATCH_BYTES:
        raise RuntimeError(f"qualification scratch exceeds 10 GiB: {size} bytes")
    return size


def _configure_child_environment(out: Path) -> dict[str, str]:
    tool_state = out / "tool-state"
    temporary = tool_state / "tmp"
    environment = {
        "HOME": tool_state / "home",
        "TMPDIR": temporary,
        "TMP": temporary,
        "TEMP": temporary,
        "XDG_CACHE_HOME": tool_state / "cache",
        "XDG_CONFIG_HOME": tool_state / "config",
        "CUDA_CACHE_PATH": tool_state / "cuda-cache",
    }
    for directory in set(environment.values()):
        directory.mkdir(parents=True)
        directory.chmod(0o700)
    return {name: str(path.resolve()) for name, path in environment.items()}


def _child_environment_record(
    out: Path, environment: Mapping[str, str]
) -> dict[str, str]:
    root = out.resolve()
    record = {}
    for name, raw_path in environment.items():
        path = Path(raw_path).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"child environment {name} escapes the result root: {path}"
            ) from error
        record[name] = relative.as_posix()
    return record


def _version(tool: Path, *arguments: str) -> str:
    completed = _run((tool, *arguments), timeout=60)
    lines = [
        line.strip()
        for line in (completed.stdout + "\n" + completed.stderr).splitlines()
        if line.strip()
    ]
    if not lines:
        raise RuntimeError(f"version command produced no output: {tool}")
    return " | ".join(lines)


def _validate_tool_versions(versions: dict[str, str]) -> None:
    expected = {
        "nvcc": "release 12.9,",
        "cuobjdump": "release 12.9,",
        "nsys": "version 2025.1.3",
        "ncu": "version 2025.2.1",
    }
    for tool, marker in expected.items():
        if marker not in versions[tool].lower():
            raise RuntimeError(
                f"{tool} identity does not match the frozen environment: "
                f"{versions[tool]!r}"
            )


def _cuda_tool(cuda_root: Path, name: str) -> Path:
    tool = cuda_root / "bin" / name
    if not tool.is_file() or not os.access(tool, os.X_OK):
        raise FileNotFoundError(f"CUDA tool is unavailable: {tool}")
    return tool


def _git_head() -> str:
    return _run(("git", "rev-parse", "HEAD"), timeout=30).stdout.strip()


def _check_repository(expected_head: str) -> dict[str, str]:
    if re.fullmatch(r"[0-9a-f]{40}", expected_head) is None:
        raise ValueError("--expected-head must be a full lowercase git SHA")
    head = _git_head()
    if head != expected_head:
        raise RuntimeError(f"SimLLM HEAD is {head}, expected {expected_head}")
    base_check = _run(
        ("git", "merge-base", "--is-ancestor", BASE_SIMLLM_SHA, head),
        timeout=30,
        check=False,
    )
    if base_check.returncode != 0:
        raise RuntimeError(f"frozen SimLLM base {BASE_SIMLLM_SHA} is not an ancestor")
    status = _run(("git", "status", "--porcelain", "--untracked-files=all"), timeout=60)
    if status.stdout.strip():
        raise RuntimeError(f"qualification source tree is dirty:\n{status.stdout}")
    return {"base": BASE_SIMLLM_SHA, "head": head}


def _check_slurm() -> dict[str, str]:
    required = {
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "cluster": os.environ.get("SLURM_CLUSTER_NAME", ""),
        "partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "account": os.environ.get("SLURM_JOB_ACCOUNT", ""),
        "nodes": os.environ.get("SLURM_NNODES", ""),
        "tasks": os.environ.get("SLURM_NTASKS", ""),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "memory_per_node": os.environ.get("SLURM_MEM_PER_NODE", ""),
        "job_gpus": os.environ.get("SLURM_JOB_GPUS", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if not required["job_id"]:
        raise RuntimeError("qualification is not running in a Slurm job")
    if required["cluster"] != EXPECTED_CLUSTER:
        raise RuntimeError(
            f"Slurm cluster is {required['cluster']!r}, expected {EXPECTED_CLUSTER!r}"
        )
    if required["partition"] != EXPECTED_PARTITION:
        raise RuntimeError(
            f"Slurm partition is {required['partition']!r}, expected {EXPECTED_PARTITION!r}"
        )
    if required["account"] != "merlin":
        raise RuntimeError(f"Slurm account is {required['account']!r}, expected 'merlin'")
    if required["nodes"] != "1" or required["tasks"] != "1":
        raise RuntimeError("qualification requires exactly one node and one task")
    if required["cpus_per_task"] != "4":
        raise RuntimeError("qualification requires exactly four CPUs per task")
    if required["memory_per_node"] != "32768":
        raise RuntimeError("qualification requires exactly 32 GiB of host memory")
    for name in ("job_gpus", "cuda_visible_devices"):
        values = [value for value in required[name].split(",") if value]
        if len(values) != 1:
            raise RuntimeError(f"{name} does not identify exactly one GPU: {required[name]!r}")
    return required


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
        raise RuntimeError(
            f"scontrol returned {len(records)} exact records for Slurm job {job_id}"
        )
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
    scontrol_name = shutil.which("scontrol")
    if scontrol_name is None:
        raise FileNotFoundError("scontrol is unavailable on PATH")
    completed = _run(
        (scontrol_name, "--oneliner", "show", "job", job_id), timeout=60
    )
    record = _parse_scheduler_record(completed.stdout, job_id)
    expected = {
        "Account": "merlin",
        "Partition": EXPECTED_PARTITION,
        "JobState": "RUNNING",
        "NumNodes": "1",
        "NumTasks": "1",
        "CPUs/Task": "4",
    }
    for field, value in expected.items():
        if record[field] != value:
            raise RuntimeError(
                f"Slurm scheduler {field} is {record[field]!r}, expected {value!r}"
            )
    if _slurm_duration_seconds(record["TimeLimit"]) != 20 * 60:
        raise RuntimeError(
            f"Slurm time limit is {record['TimeLimit']!r}, expected 20 minutes"
        )
    requested = record["ReqTRES"].split(",")
    for tres in ("cpu=4", "mem=32G", "node=1", "gres/gpu=1"):
        if tres not in requested:
            raise RuntimeError(f"Slurm requested TRES lacks {tres!r}: {record['ReqTRES']}")
    if "gres/gpu:nvidia_a100-sxm4-80gb=1" not in record["AllocTRES"].split(","):
        raise RuntimeError(
            "Slurm allocated TRES lacks one typed A100 SXM4 80 GB: "
            f"{record['AllocTRES']}"
        )
    if record["OverSubscribe"].upper() == "EXCLUSIVE":
        raise RuntimeError("qualification unexpectedly received an exclusive node")
    return record


def _job_visible_gpu_selector(slurm: dict[str, str]) -> str:
    visible = slurm["cuda_visible_devices"].strip()
    if visible.isdigit():
        return visible
    if re.fullmatch(r"GPU-[0-9A-Fa-f-]+", visible) is not None:
        return visible
    raise RuntimeError(
        "cannot resolve the job-local GPU selector from CUDA_VISIBLE_DEVICES: "
        f"{visible!r}; SLURM_JOB_GPUS is allocation provenance only: "
        f"{slurm['job_gpus']!r}"
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


def _gpu_snapshot(nvidia_smi: Path, gpu_selector: str) -> dict[str, str]:
    completed = _run(
        (
            nvidia_smi,
            f"--id={gpu_selector}",
            f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ),
        timeout=60,
    )
    rows = list(csv.reader(completed.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != len(GPU_QUERY_FIELDS):
        raise RuntimeError(f"unexpected nvidia-smi GPU row: {completed.stdout!r}")
    snapshot = {
        field: value.strip() for field, value in zip(GPU_QUERY_FIELDS, rows[0], strict=True)
    }
    if snapshot["name"] != EXPECTED_GPU_NAME:
        raise RuntimeError(
            f"visible GPU is {snapshot['name']!r}, expected {EXPECTED_GPU_NAME!r}"
        )
    if snapshot["compute_cap"] != EXPECTED_COMPUTE_CAPABILITY:
        raise RuntimeError(
            "visible GPU compute capability is "
            f"{snapshot['compute_cap']!r}, expected {EXPECTED_COMPUTE_CAPABILITY!r}"
        )
    return snapshot


def _mig_state(nvidia_smi: Path, gpu_selector: str) -> dict[str, str]:
    completed = _run((nvidia_smi, f"--id={gpu_selector}", "-q"), timeout=60)
    mig_mode = re.search(
        r"(?m)^[ \t]*MIG Mode[ \t]*\r?\n"
        r"[ \t]*Current[ \t]*:[ \t]*([^\r\n]+)"
        r"\r?\n[ \t]*Pending[ \t]*:[ \t]*([^\r\n]+)",
        completed.stdout,
    )
    if mig_mode is None:
        raise RuntimeError("nvidia-smi full query has no complete MIG mode section")
    state = {
        "current": mig_mode.group(1).strip(),
        "pending": mig_mode.group(2).strip(),
    }
    if any(value.lower() != "disabled" for value in state.values()):
        raise RuntimeError(f"MIG is not stably disabled: {state}")
    return state


def _supported_clock_policy(
    nvidia_smi: Path, gpu_selector: str
) -> tuple[list[dict[str, int]], str | None]:
    completed = _run(
        (
            nvidia_smi,
            f"--id={gpu_selector}",
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
            clocks.append(
                {"memory_mhz": memory_mhz, "graphics_mhz": graphics_mhz}
            )
    except ValueError as error:
        return [], f"supported clock query was unparseable: {error}"
    if not clocks:
        return [], "supported clock query returned no clock pairs"
    return clocks, None


def _supported_clock_evidence_blockers(
    before: Sequence[Mapping[str, int]],
    before_blocker: str | None,
    after: Sequence[Mapping[str, int]],
    after_blocker: str | None,
) -> list[str]:
    blockers = []
    if before_blocker is not None:
        blockers.append(f"before profiling: {before_blocker}")
    if after_blocker is not None:
        blockers.append(f"after profiling: {after_blocker}")
    if blockers:
        return blockers
    before_pairs = {
        (int(row["memory_mhz"]), int(row["graphics_mhz"])) for row in before
    }
    after_pairs = {
        (int(row["memory_mhz"]), int(row["graphics_mhz"])) for row in after
    }
    if before_pairs != after_pairs:
        blockers.append("supported clock policy changed during profiler probes")
    return blockers


def _foreign_processes(
    nvidia_smi: Path, gpu_selector: str, gpu_uuid: str
) -> list[dict[str, str]]:
    fields = ("gpu_uuid", "pid", "process_name", "used_gpu_memory")
    completed = _run(
        (
            nvidia_smi,
            f"--id={gpu_selector}",
            f"--query-compute-apps={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ),
        timeout=60,
        check=False,
    )
    if completed.returncode != 0 and "No running processes found" not in (
        completed.stdout + completed.stderr
    ):
        raise RuntimeError(f"nvidia-smi process query failed: {completed.stderr}")
    processes: list[dict[str, str]] = []
    for row in csv.reader(completed.stdout.splitlines()):
        if len(row) != len(fields):
            continue
        record = {field: value.strip() for field, value in zip(fields, row, strict=True)}
        if record["gpu_uuid"] == gpu_uuid:
            processes.append(record)
    return processes


def _probe_command(binary: Path) -> tuple[str | Path, ...]:
    return (
        binary,
        "--measured-launches",
        str(EXPECTED_MEASURED),
    )


def _validate_probe_output(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip()
    expected = {
        "probe": "simllm-a100-environment-qualification-v1",
        "target_kernel": TARGET_KERNEL,
        "device_name": EXPECTED_GPU_NAME,
        "compute_capability": EXPECTED_COMPUTE_CAPABILITY,
        "element_count": str(EXPECTED_ELEMENTS),
        "threads_per_block": str(EXPECTED_THREADS),
        "warmup_launches": str(EXPECTED_WARMUPS),
        "measured_launches": str(EXPECTED_MEASURED),
        "correctness": "PASS",
        "status": "PASS",
    }
    for name, value in expected.items():
        if values.get(name) != value:
            raise RuntimeError(f"probe output {name}={values.get(name)!r}, expected {value!r}")
    if not values.get("output_checksum"):
        raise RuntimeError("probe output has no checksum")
    return values


def _artifact_manifest(out: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in out.rglob("*") if item.is_file()):
        if path.name == "qualification.json":
            continue
        rows.append(
            {
                "path": path.relative_to(out).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _write_partial_context(out: Path, context: dict[str, Any]) -> None:
    (out / "partial_context.json").write_text(
        json.dumps(context, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _has_numeric_ncu_metric(output: str) -> bool:
    lines = output.splitlines()
    required_columns = ("id", "kernelname", "metricname", "metricvalue")
    column_indices = None
    header_index = None
    for index, line in enumerate(lines):
        header = next(csv.reader([line]))
        normalized = [_normalized_header(field) for field in header]
        if not all(normalized.count(column) == 1 for column in required_columns):
            continue
        column_indices = {
            column: normalized.index(column) for column in required_columns
        }
        header_index = index
        break
    if column_indices is None or header_index is None:
        return False

    target_launch_ids = set()
    has_numeric_metric = False
    maximum_index = max(column_indices.values())
    for row in csv.reader(lines[header_index + 1 :]):
        if len(row) <= maximum_index:
            continue
        kernel_name = row[column_indices["kernelname"]].strip()
        if kernel_name != TARGET_KERNEL:
            continue
        launch_id = row[column_indices["id"]].strip()
        if not launch_id:
            return False
        target_launch_ids.add(launch_id)
        metric_name = row[column_indices["metricname"]].strip()
        if not metric_name:
            continue
        candidate = row[column_indices["metricvalue"]].strip().replace(",", "")
        try:
            value = float(candidate)
        except ValueError:
            continue
        if math.isfinite(value):
            has_numeric_metric = True
    return has_numeric_metric and len(target_launch_ids) == 1


def _ncu_capability_blocker(output: str) -> str | None:
    markers = (
        "err_nvgpuctrperm",
        "permission to access nvidia gpu performance counters",
        "profiling is not supported on this device",
        "unsupported gpu",
    )
    for line in output.splitlines():
        if any(marker in line.lower() for marker in markers):
            return line.strip()
    return None


def _normalized_header(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _trace_column(fieldnames: Sequence[str], *aliases: str) -> str:
    wanted = {_normalized_header(alias) for alias in aliases}
    matches = [field for field in fieldnames if _normalized_header(field) in wanted]
    if len(matches) != 1:
        raise RuntimeError(
            f"Nsight Systems trace expected one of {aliases}, observed {matches}"
        )
    return matches[0]


def _parse_positive_trace_int(row: dict[str, str], column: str) -> int:
    try:
        value = int(row[column].strip())
    except (KeyError, ValueError) as error:
        raise RuntimeError(
            f"Nsight Systems trace {column!r} is not an integer: {row.get(column)!r}"
        ) from error
    if value <= 0:
        raise RuntimeError(f"Nsight Systems trace {column!r} must be positive")
    return value


def _validate_nsys_trace(output: str) -> list[dict[str, Any]]:
    lines = output.splitlines()
    header_index = None
    for index, line in enumerate(lines):
        candidate = next(csv.reader([line]))
        if len(candidate) >= 2 and (
            _normalized_header(candidate[0]) == "startns"
            and _normalized_header(candidate[1]) == "durationns"
        ):
            header_index = index
            break
    if header_index is None:
        raise RuntimeError("Nsight Systems trace has no CUDA GPU trace header")
    reader = csv.DictReader(lines[header_index:])
    if reader.fieldnames is None:
        raise RuntimeError("Nsight Systems trace has no CSV field names")
    name_column = _trace_column(reader.fieldnames, "Name")
    start_column = _trace_column(reader.fieldnames, "Start (ns)")
    duration_column = _trace_column(reader.fieldnames, "Duration (ns)")
    device_column = _trace_column(reader.fieldnames, "Device")
    context_column = _trace_column(reader.fieldnames, "Ctx", "Context")
    stream_column = _trace_column(reader.fieldnames, "Strm", "Stream")
    grid_columns = tuple(
        _trace_column(reader.fieldnames, short, long)
        for short, long in zip(
            ("GrdX", "GrdY", "GrdZ"),
            ("Grid X", "Grid Y", "Grid Z"),
            strict=True,
        )
    )
    block_columns = tuple(
        _trace_column(reader.fieldnames, short, long)
        for short, long in zip(
            ("BlkX", "BlkY", "BlkZ"),
            ("Block X", "Block Y", "Block Z"),
            strict=True,
        )
    )
    target_rows = []
    for row in reader:
        if TARGET_KERNEL not in row.get(name_column, ""):
            continue
        try:
            start_ns = float(row[start_column])
            duration_ns = float(row[duration_column])
            context_id = int(row[context_column])
            stream_id = int(row[stream_column])
        except (KeyError, ValueError) as error:
            raise RuntimeError("Nsight Systems target row has invalid numeric fields") from error
        if (
            not math.isfinite(start_ns)
            or not math.isfinite(duration_ns)
            or start_ns < 0
            or duration_ns <= 0
            or context_id < 0
            or stream_id < 0
        ):
            raise RuntimeError("Nsight Systems target row has invalid timing or identity")
        grid = tuple(_parse_positive_trace_int(row, column) for column in grid_columns)
        block = tuple(_parse_positive_trace_int(row, column) for column in block_columns)
        if grid != (EXPECTED_ELEMENTS // EXPECTED_THREADS, 1, 1):
            raise RuntimeError(f"Nsight Systems target grid drifted: {grid}")
        if block != (EXPECTED_THREADS, 1, 1):
            raise RuntimeError(f"Nsight Systems target block drifted: {block}")
        device = row.get(device_column, "").strip()
        if not device:
            raise RuntimeError("Nsight Systems target row has no device identity")
        target_rows.append(
            {
                "name": row[name_column],
                "start_ns": start_ns,
                "duration_ns": duration_ns,
                "device": device,
                "context_id": context_id,
                "stream_id": stream_id,
                "grid": list(grid),
                "block": list(block),
            }
        )
    if not target_rows:
        raise RuntimeError("Nsight Systems CUDA trace has no target-kernel row")
    return target_rows


def _validate_nsys_output_paths(
    output: str, out: Path, expected_report: Path
) -> dict[str, str]:
    temporary_paths = re.findall(
        r"(?m)^Generating '([^'\r\n]+\.qdstrm)'[ \t]*$", output
    )
    report_paths = re.findall(
        r"(?m)^Generated:[ \t]*\r?\n[ \t]+([^\r\n]+?\.nsys-rep)[ \t]*$",
        output,
    )
    if len(temporary_paths) != 1:
        raise RuntimeError(
            "Nsight Systems did not report exactly one temporary stream path: "
            f"{temporary_paths}"
        )
    if len(report_paths) != 1:
        raise RuntimeError(
            "Nsight Systems did not report exactly one final report path: "
            f"{report_paths}"
        )
    root = out.resolve()
    temporary_root = (out / "tool-state" / "tmp").resolve()
    temporary = Path(temporary_paths[0])
    if not temporary.is_absolute():
        raise RuntimeError(
            f"Nsight Systems reported a nonabsolute temporary path: {temporary}"
        )
    try:
        temporary_relative = temporary.resolve().relative_to(temporary_root)
    except ValueError as error:
        raise RuntimeError(
            "Nsight Systems temporary output escaped the configured temporary "
            f"root: {temporary}"
        ) from error
    report = Path(report_paths[0])
    if not report.is_absolute():
        raise RuntimeError(
            f"Nsight Systems reported a nonabsolute final report path: {report}"
        )
    if report.resolve() != expected_report.resolve():
        raise RuntimeError(
            f"Nsight Systems reported an unexpected final report path: {report}"
        )
    return {
        "intermediate": (Path("tool-state/tmp") / temporary_relative).as_posix(),
        "report": report.resolve().relative_to(root).as_posix(),
    }


def _telemetry_blockers(snapshot: dict[str, str]) -> list[str]:
    required_numeric = (
        "clocks.current.sm",
        "clocks.current.memory",
        "clocks.max.sm",
        "clocks.max.memory",
        "power.limit",
        "power.draw",
        "temperature.gpu",
    )
    blockers = []
    for field in required_numeric:
        try:
            if float(snapshot[field]) <= 0:
                raise ValueError
        except (KeyError, ValueError):
            blockers.append(f"nvidia-smi did not expose a positive numeric {field}")
    for field in ("persistence_mode", "compute_mode"):
        value = snapshot.get(field, "").strip().lower()
        if not value or value in {"n/a", "[n/a]", "unknown"}:
            blockers.append(f"nvidia-smi did not expose {field}")
    return blockers


def _run_qualification(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    revision = _check_repository(args.expected_head)
    if _QUALIFICATION_ENVIRONMENT is None:
        raise RuntimeError("qualification child environment is not configured")
    context: dict[str, Any] = {
        "revision": revision,
        "expectations_sha256": _sha256(EXPECTATIONS_PATH),
        "probe_source_sha256": _sha256(PROBE_SOURCE),
        "global_deadline_seconds": GLOBAL_DEADLINE_SECONDS,
        "child_environment": _child_environment_record(
            args.out, _QUALIFICATION_ENVIRONMENT
        ),
    }
    _write_partial_context(args.out, context)
    slurm = _check_slurm()
    scheduler = _scheduler_snapshot(slurm["job_id"])
    job_visible_gpu_selector = _job_visible_gpu_selector(slurm)
    context.update(
        {
            "slurm": {"environment": slurm, "scheduler": scheduler},
            "job_visible_gpu_selector": job_visible_gpu_selector,
        }
    )
    _write_partial_context(args.out, context)

    cuda_root = args.cuda_root.resolve()
    cuda_module = os.environ.get("SIMLLM_QUALIFICATION_CUDA_MODULE", "")
    if cuda_module != EXPECTED_CUDA_MODULE:
        raise RuntimeError(
            f"CUDA module is {cuda_module!r}, expected {EXPECTED_CUDA_MODULE!r}"
        )
    loaded_modules = os.environ.get("LOADEDMODULES", "").split(":")
    if EXPECTED_CUDA_MODULE not in loaded_modules:
        raise RuntimeError(
            f"loaded modules do not contain {EXPECTED_CUDA_MODULE!r}: {loaded_modules}"
        )
    tools = {
        name: _cuda_tool(cuda_root, name)
        for name in ("nvcc", "cuobjdump", "nsys", "ncu")
    }
    tool_versions = {
        "cuda_module": cuda_module,
        "cuda_root": str(cuda_root),
        "nvcc": _version(tools["nvcc"], "--version"),
        "cuobjdump": _version(tools["cuobjdump"], "--version"),
        "nsys": _version(tools["nsys"], "--version"),
        "ncu": _version(tools["ncu"], "--version"),
    }
    _validate_tool_versions(tool_versions)
    context["tool_versions"] = tool_versions
    _write_partial_context(args.out, context)
    nvidia_smi_name = shutil.which("nvidia-smi")
    if nvidia_smi_name is None:
        raise FileNotFoundError("nvidia-smi is unavailable on PATH")
    nvidia_smi = Path(nvidia_smi_name)

    build_dir = args.out / "build"
    capture_dir = args.out / "capture"
    build_dir.mkdir()
    capture_dir.mkdir()

    before = _gpu_snapshot(nvidia_smi, job_visible_gpu_selector)
    gpu_uuid = before["uuid"]
    mig_before = _mig_state(nvidia_smi, gpu_uuid)
    supported_clocks_before, supported_clock_blocker_before = _supported_clock_policy(
        nvidia_smi, gpu_uuid
    )
    processes_before = _foreign_processes(nvidia_smi, gpu_uuid, gpu_uuid)
    context.update(
        {
            "gpu_before": before,
            "mig_before": mig_before,
            "supported_clocks_before": supported_clocks_before,
            "supported_clock_blocker_before": supported_clock_blocker_before,
            "processes_before": processes_before,
        }
    )
    _write_partial_context(args.out, context)
    if processes_before:
        raise RuntimeError(f"allocated GPU has foreign compute processes: {processes_before}")

    binary = build_dir / "a100_environment_probe"
    compile_run = _run(
        (
            tools["nvcc"],
            "-std=c++17",
            "-O3",
            "-lineinfo",
            "-arch=sm_80",
            PROBE_SOURCE,
            "-o",
            binary,
        ),
        timeout=120,
    )
    _write_log(build_dir / "nvcc.log", compile_run)
    context["probe_binary_sha256"] = _sha256(binary)
    _write_partial_context(args.out, context)

    unprofiled = _run(_probe_command(binary), timeout=180)
    _write_log(capture_dir / "unprofiled.log", unprofiled)
    probe_values = _validate_probe_output(unprofiled.stdout)
    probe_uuid = probe_values.get("device_uuid", "")
    if probe_uuid.lower() != before["uuid"].lower():
        raise RuntimeError(
            "CUDA and nvidia-smi GPU UUIDs disagree: "
            f"{probe_uuid!r} != {before['uuid']!r}"
        )

    sass_run = _run((tools["cuobjdump"], "--dump-sass", binary), timeout=120)
    sass_path = build_dir / "a100_environment_probe.sass"
    sass_path.write_text(sass_run.stdout, encoding="utf-8")
    if TARGET_KERNEL not in sass_run.stdout or "sm_80" not in sass_run.stdout:
        raise RuntimeError("static SASS lacks the target kernel or sm_80 code object")
    context.update(
        {
            "probe": probe_values,
            "static_sass_sha256": _sha256(sass_path),
        }
    )
    _write_partial_context(args.out, context)

    nsys_prefix = capture_dir / "a100_environment_probe"
    nsys_run = _run(
        (
            tools["nsys"],
            "profile",
            "--trace=cuda",
            "--sample=none",
            "--cpuctxsw=none",
            "--force-overwrite=true",
            "--output",
            nsys_prefix,
            *_probe_command(binary),
        ),
        timeout=PROFILER_TIMEOUT_SECONDS,
    )
    _write_log(capture_dir / "nsys_profile.log", nsys_run)
    report = nsys_prefix.with_suffix(".nsys-rep")
    nsys_output_paths = _validate_nsys_output_paths(
        nsys_run.stdout + "\n" + nsys_run.stderr, args.out, report
    )
    if not report.is_file() or report.stat().st_size == 0:
        raise RuntimeError("Nsight Systems produced no report")
    nsys_stats = _run(
        (
            tools["nsys"],
            "stats",
            "--report",
            "cuda_gpu_trace",
            "--format",
            "csv",
            "--output",
            "-",
            report,
        ),
        timeout=PROFILER_TIMEOUT_SECONDS,
    )
    _write_log(capture_dir / "nsys_stats.log", nsys_stats)
    trace_csv = capture_dir / "cuda_gpu_trace.csv"
    trace_csv.write_text(nsys_stats.stdout, encoding="utf-8")
    nsys_target_rows = _validate_nsys_trace(nsys_stats.stdout)
    context.update(
        {
            "nsys_trace_sha256": _sha256(trace_csv),
            "nsys_report_sha256": _sha256(report),
            "nsys_target_rows": nsys_target_rows,
            "nsys_output_paths": nsys_output_paths,
        }
    )
    _write_partial_context(args.out, context)
    output_bytes_after_nsys = _assert_output_budget(args.out)
    scratch_bytes_after_nsys = _assert_scratch_budget(args.scratch_root)
    context.update(
        {
            "output_bytes_after_nsys": output_bytes_after_nsys,
            "scratch_bytes_after_nsys": scratch_bytes_after_nsys,
        }
    )
    _write_partial_context(args.out, context)
    _assert_output_budget(args.out)
    _assert_scratch_budget(args.scratch_root)

    ncu_run = _run(
        (
            tools["ncu"],
            "--set",
            "basic",
            "--profile-from-start",
            "yes",
            "--target-processes",
            "all",
            "--kernel-name",
            f"regex:{TARGET_KERNEL}",
            "--launch-skip",
            str(EXPECTED_WARMUPS),
            "--launch-count",
            "1",
            "--csv",
            *_probe_command(binary),
        ),
        timeout=PROFILER_TIMEOUT_SECONDS,
        check=False,
    )
    _write_log(capture_dir / "ncu_basic.log", ncu_run)
    ncu_output = ncu_run.stdout + "\n" + ncu_run.stderr
    counter_blocker = _ncu_capability_blocker(ncu_output)
    if ncu_run.returncode != 0 and counter_blocker is None:
        raise RuntimeError(
            f"Nsight Compute failed with status {ncu_run.returncode}; "
            "see capture/ncu_basic.log"
        )
    if ncu_run.returncode == 0 and not _has_numeric_ncu_metric(ncu_output):
        counter_blocker = "Nsight Compute returned no numeric target-kernel metric"

    after = _gpu_snapshot(nvidia_smi, gpu_uuid)
    mig_after = _mig_state(nvidia_smi, gpu_uuid)
    supported_clocks_after, supported_clock_blocker_after = _supported_clock_policy(
        nvidia_smi, gpu_uuid
    )
    processes_after = _foreign_processes(nvidia_smi, gpu_uuid, gpu_uuid)
    context.update(
        {
            "gpu_after": after,
            "mig_after": mig_after,
            "supported_clocks_after": supported_clocks_after,
            "supported_clock_blocker_after": supported_clock_blocker_after,
            "processes_after": processes_after,
            "counter_blocker": counter_blocker,
        }
    )
    _write_partial_context(args.out, context)
    if processes_after:
        raise RuntimeError(f"allocated GPU retains foreign compute processes: {processes_after}")
    output_bytes_after_ncu = _assert_output_budget(args.out)
    scratch_bytes_after_ncu = _assert_scratch_budget(args.scratch_root)

    capability_blockers = [
        *_telemetry_blockers(before),
        *_telemetry_blockers(after),
        *_supported_clock_evidence_blockers(
            supported_clocks_before,
            supported_clock_blocker_before,
            supported_clocks_after,
            supported_clock_blocker_after,
        ),
    ]
    if counter_blocker is not None:
        capability_blockers.append(counter_blocker)
    context.update(
        {
            "capability_blockers": capability_blockers,
            "output_bytes_after_ncu": output_bytes_after_ncu,
            "scratch_bytes_after_ncu": scratch_bytes_after_ncu,
        }
    )
    _write_partial_context(args.out, context)
    _assert_output_budget(args.out)
    _assert_scratch_budget(args.scratch_root)
    state = "BLOCKED" if capability_blockers else "QUALIFIED"
    result = {
        "schema": "simllm-a100-environment-qualification-v1",
        "state": state,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "revision": revision,
        "expectations_sha256": _sha256(EXPECTATIONS_PATH),
        "probe_source_sha256": _sha256(PROBE_SOURCE),
        "probe_binary_sha256": _sha256(binary),
        "static_sass_sha256": _sha256(sass_path),
        "slurm": {"environment": slurm, "scheduler": scheduler},
        "job_visible_gpu_selector": job_visible_gpu_selector,
        "gpu_before": before,
        "gpu_after": after,
        "mig_before": mig_before,
        "mig_after": mig_after,
        "supported_clocks_before": supported_clocks_before,
        "supported_clocks_after": supported_clocks_after,
        "processes_before": processes_before,
        "processes_after": processes_after,
        "child_environment": context["child_environment"],
        "tool_versions": tool_versions,
        "probe": probe_values,
        "nsys_trace_sha256": _sha256(trace_csv),
        "nsys_report_sha256": _sha256(report),
        "nsys_target_rows": nsys_target_rows,
        "nsys_output_paths": nsys_output_paths,
        "output_bytes_after_nsys": output_bytes_after_nsys,
        "output_bytes_after_ncu": output_bytes_after_ncu,
        "scratch_bytes_after_nsys": scratch_bytes_after_nsys,
        "scratch_bytes_after_ncu": scratch_bytes_after_ncu,
        "counter_status": "available" if counter_blocker is None else "blocked",
        "counter_blocker": counter_blocker,
        "capability_blockers": capability_blockers,
        "artifacts": _artifact_manifest(args.out),
        "nonclaims": [
            "not a compute calibration",
            "not a production kernel capture",
            "not dynamic SASS evidence",
            "not an SGLang TTFT or TPOT result",
        ],
    }
    return result, 2 if state == "BLOCKED" else 0


def check_only(args: argparse.Namespace) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", args.expected_head) is None:
        raise SystemExit("--expected-head must be a full lowercase git SHA")
    if not args.out.is_absolute():
        raise SystemExit("--out must be an explicit absolute path")
    if not args.scratch_root.is_absolute():
        raise SystemExit("--scratch-root must be an explicit absolute path")
    if args.out.resolve() == REPOSITORY_ROOT or REPOSITORY_ROOT in args.out.resolve().parents:
        raise SystemExit("--out must be outside the repository")
    if (
        args.scratch_root.resolve() == REPOSITORY_ROOT
        or REPOSITORY_ROOT in args.scratch_root.resolve().parents
    ):
        raise SystemExit("--scratch-root must be outside the repository")
    if EXPECTED_ELEMENTS != 16_777_216 or EXPECTED_THREADS != 256:
        raise AssertionError("frozen probe geometry changed")
    if EXPECTED_WARMUPS != 5 or EXPECTED_MEASURED != 1:
        raise AssertionError("frozen probe launch counts changed")
    if (
        PROFILER_TIMEOUT_SECONDS != 180
        or GLOBAL_DEADLINE_SECONDS != 17 * 60
        or MAX_OUTPUT_BYTES != 10 * 1024**3
        or MAX_SCRATCH_BYTES != 10 * 1024**3
    ):
        raise AssertionError("frozen qualification safety bounds changed")
    if not EXPECTATIONS_PATH.is_file() or not PROBE_SOURCE.is_file():
        raise SystemExit("qualification source files are incomplete")
    _check_repository(args.expected_head)
    print("A100_ENVIRONMENT_QUALIFICATION_CHECK_ONLY=PASS")


def main() -> int:
    global _QUALIFICATION_DEADLINE, _QUALIFICATION_ENVIRONMENT

    args = parse_args()
    if not args.out.is_absolute():
        raise SystemExit("--out must be an explicit absolute path")
    if not args.scratch_root.is_absolute():
        raise SystemExit("--scratch-root must be an explicit absolute path")
    args.cuda_root = args.cuda_root.resolve()
    args.out = args.out.resolve()
    args.scratch_root = args.scratch_root.resolve()
    if args.check_only:
        check_only(args)
        return 0
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite qualification output: {args.out}")
    if args.out == REPOSITORY_ROOT or REPOSITORY_ROOT in args.out.parents:
        raise SystemExit("qualification output must be outside the repository")
    if not args.scratch_root.is_dir():
        raise SystemExit(f"qualification scratch root is unavailable: {args.scratch_root}")
    if args.scratch_root not in args.out.parents:
        raise SystemExit("qualification output must be inside the scratch root")
    args.out.mkdir(parents=True)
    _QUALIFICATION_ENVIRONMENT = _configure_child_environment(args.out)
    _QUALIFICATION_DEADLINE = time.monotonic() + GLOBAL_DEADLINE_SECONDS

    result_path = args.out / "qualification.json"
    try:
        result, status = _run_qualification(args)
    except Exception as error:
        partial_context_path = args.out / "partial_context.json"
        partial_context = None
        if partial_context_path.is_file():
            partial_context = json.loads(partial_context_path.read_text(encoding="utf-8"))
        result = {
            "schema": "simllm-a100-environment-qualification-v1",
            "state": "VOID",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "expected_head": args.expected_head,
            "expectations_sha256": (
                _sha256(EXPECTATIONS_PATH) if EXPECTATIONS_PATH.is_file() else None
            ),
            "probe_source_sha256": (
                _sha256(PROBE_SOURCE) if PROBE_SOURCE.is_file() else None
            ),
            "partial_context": partial_context,
            "output_bytes": _tree_size(args.out),
            "scratch_bytes": _tree_size(args.scratch_root),
            "artifacts": _artifact_manifest(args.out),
        }
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        raise
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"A100_ENVIRONMENT_QUALIFICATION={result['state']}")
    print(f"A100_ENVIRONMENT_QUALIFICATION_RESULT={result_path}")
    return status


if __name__ == "__main__":
    sys.exit(main())
