#!/usr/bin/env python3
"""Validate and extract the exact Nsys service from a CORE-61 retry capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

MODEL = "deepseek-ai/DeepSeek-V3"
REVISION = "e815299b0bcbac849fa540c768ef21845365c9eb"
CONFIG_SHA256 = "cbf0b95dc614de208a109bb5fd4e7eed11385e9c68411d2c17db5319443035d9"
CELL = "decode_b32_c2000"
BOUNDARY = "execute_context_0(0)_generation_32(32)"
COLLECTIVE_MARKERS = (
    "nccl",
    "msccl",
    "customallreduce",
    "custom_all_reduce",
    "cross_device_reduce",
    "allreduce",
    "all_reduce",
    "reducescatter",
    "reduce_scatter",
    "allgather",
    "all_gather",
    "alltoall",
    "all_to_all",
    "sendrecv",
    "nvshmem",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_collective(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in COLLECTIVE_MARKERS)


def _validate_profile(run_dir: Path, profile: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "schema": "simllm-core61-depth8-retry-capture-v1",
        "model": MODEL,
        "revision": REVISION,
        "reduced_layers": 8,
        "startup_max_num_batched_tokens": 4096,
        "max_num_seqs": 32,
    }
    for field, value in expected.items():
        if profile.get(field) != value:
            raise ValueError(f"profile.{field}: expected {value!r}")
    config = profile.get("model_config")
    if not isinstance(config, dict) or config.get("config_sha256") != CONFIG_SHA256:
        raise ValueError("profile.model_config: frozen digest is absent")
    if profile.get("framework", {}).get("name") != "vllm":
        raise ValueError("profile.framework: expected vLLM")
    cases = profile.get("cases")
    if not isinstance(cases, list) or len(cases) != 1:
        raise ValueError("profile.cases: expected one exact decode case")
    case = cases[0]
    for field, value in {
        "cell": CELL,
        "pool": "decode",
        "batch_size": 32,
        "remote_kv_tokens_per_request": 2000,
        "decode_steps": 1,
    }.items():
        if case.get(field) != value:
            raise ValueError(f"profile.cases[0].{field}: expected {value!r}")
    scheduler = case.get("scheduler_marker", {}).get("scheduler", {})
    computed = scheduler.get("cached_num_computed_tokens_by_request", {})
    if (
        scheduler.get("is_decode") is not True
        or scheduler.get("num_requests") != 32
        or len(computed) != 32
        or any(int(value) != 2000 for value in computed.values())
    ):
        raise ValueError("profile scheduler marker is not exact batch-32 KV-2000")
    if (run_dir / "weight_files.txt").read_text(encoding="utf-8").strip():
        raise ValueError("dummy-weight isolation was violated")
    return case


def analyze(run_dir: Path) -> dict[str, Any]:
    profile_path = run_dir / "profile.json"
    database_path = run_dir / "profile.sqlite"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    case = _validate_profile(run_dir, profile)
    connection = sqlite3.connect(str(database_path))
    try:
        session_epoch_ns = connection.execute(
            "SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME"
        ).fetchone()[0]
        outer_start = int(case["started_epoch_ns"]) - session_epoch_ns
        outer_end = int(case["finished_epoch_ns"]) - session_epoch_ns
        ranges = list(
            connection.execute(
                """
                SELECT n.start, n.end, COALESCE(n.text, names.value)
                FROM NVTX_EVENTS AS n
                LEFT JOIN StringIds AS names ON n.textId = names.id
                WHERE n.end IS NOT NULL AND n.start >= ? AND n.end <= ?
                  AND COALESCE(n.text, names.value) = ?
                ORDER BY n.start
                """,
                (outer_start, outer_end, BOUNDARY),
            )
        )
        if len(ranges) != 1:
            raise ValueError(f"expected one exact decode boundary, found {len(ranges)}")
        start, end, _ = ranges[0]
        correlation_ids = {
            row[0]
            for row in connection.execute(
                """
                SELECT DISTINCT correlationId
                FROM CUPTI_ACTIVITY_KIND_RUNTIME
                WHERE correlationId IS NOT NULL AND start >= ? AND end <= ?
                """,
                (start, end),
            )
        }
        if not correlation_ids:
            raise ValueError("exact decode boundary has no runtime correlations")
        placeholders = ",".join("?" for _ in correlation_ids)
        kernels = list(
            connection.execute(
                f"""
                SELECT k.start, k.end, names.value
                FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
                JOIN StringIds AS names ON k.demangledName = names.id
                WHERE k.correlationId IN ({placeholders})
                ORDER BY k.start
                """,
                tuple(sorted(correlation_ids)),
            )
        )
    finally:
        connection.close()
    if not kernels:
        raise ValueError("exact decode boundary has no GPU kernels")
    noncollective_ns = sum(
        end - start for start, end, name in kernels if not _is_collective(name)
    )
    collective_ns = sum(
        end - start for start, end, name in kernels if _is_collective(name)
    )
    if noncollective_ns <= 0:
        raise ValueError("exact decode boundary has no positive noncollective service")
    sources = []
    for name in (
        "profile.json",
        "profile.sqlite",
        "profile.nsys-rep",
        "alignment.json",
        "harness_sha256.txt",
        "weight_files.txt",
        "job.log",
    ):
        path = run_dir / name
        sources.append(
            {"name": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    return {
        "schema": "simllm-core61-depth8-retry-measurement-v1",
        "evidence_class": "MEASURED",
        "job_id": profile["machine"]["slurm_job_id"],
        "shape": {
            "depth_layers": 8,
            "batch_size": 32,
            "remote_kv_tokens_per_request": 2000,
        },
        "boundary": {
            "basis": "exact-nvtx-runtime-correlation",
            "label": BOUNDARY,
            "runtime_correlation_count": len(correlation_ids),
            "kernel_record_count": len(kernels),
        },
        "measured_service_ps": noncollective_ns * 1000,
        "collective_service_ps": collective_ns * 1000,
        "scheduler_guard": case["scheduler_marker"]["scheduler"],
        "sources": sources,
        "status": "DIGEST_READY_MEASUREMENT",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_dir.resolve())
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
