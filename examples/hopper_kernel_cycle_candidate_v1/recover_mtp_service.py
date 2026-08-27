#!/usr/bin/env python3
"""Recover the registered DeepSeek MTP service from its exact NVTX boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = "simllm-hopper-mtp-nsys-service-v1"
EXPECTED_MODEL = "deepseek-ai/DeepSeek-V3"
EXPECTED_REVISION = "e815299b0bcbac849fa540c768ef21845365c9eb"
EXPECTED_CONFIG_SHA256 = "cbf0b95dc614de208a109bb5fd4e7eed11385e9c68411d2c17db5319443035d9"
EXPECTED_BASE_CELL = "decode_base_mtp_b16_c4000"
EXPECTED_DECODE_CELL = "decode_mtp_b16_c4000"
EXPECTED_NVTX_LABEL = "execute_context_0(0)_generation_16(32)"
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
SOURCE_PATHS = (
    "profile.json",
    "profile.sqlite",
    "profile.nsys-rep",
    "analysis/ordered-kernels.csv",
    "harness_sha256.txt",
    "sha256.txt",
    "weight_files.txt",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for relative in SOURCE_PATHS:
        path = run_dir / relative
        if not path.is_file():
            raise ValueError(f"registered source is absent: {relative}")
        rows.append(
            {
                "name": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _validate_profile(profile: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "model": EXPECTED_MODEL,
        "model_key": "deepseek-v3",
        "tensor_parallel_size": 1,
        "mode": "graph",
        "shape_set": "deepseek",
        "deepseek_suite": "mtp",
        "reduced_layers": 4,
        "phase": "profile",
    }
    for name, value in expected.items():
        if profile.get(name) != value:
            raise ValueError(f"profile.{name}: expected {value!r}, found {profile.get(name)!r}")
    config = profile.get("model_config")
    if not isinstance(config, dict):
        raise TypeError("profile.model_config: expected an object")
    for name, value in {
        "requested_revision": EXPECTED_REVISION,
        "resolved_revision": EXPECTED_REVISION,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "effective_num_hidden_layers": 4,
    }.items():
        if config.get(name) != value:
            raise ValueError(
                f"profile.model_config.{name}: expected {value!r}, found {config.get(name)!r}"
            )
    expected_cases = {
        EXPECTED_BASE_CELL: ("decode_base", 16, 4000, 1, 0),
        EXPECTED_DECODE_CELL: ("decode", 16, 4000, 2, 1),
    }
    cases = profile.get("cases")
    if not isinstance(cases, list) or len(cases) != 2:
        raise ValueError("profile.cases: expected the exact base and MTP decode pair")
    by_cell = {case.get("cell"): case for case in cases if isinstance(case, dict)}
    if set(by_cell) != set(expected_cases):
        raise ValueError("profile.cases: unexpected MTP cell identity")
    for cell, expected_case in expected_cases.items():
        case = by_cell[cell]
        actual = tuple(
            case.get(name)
            for name in ("pool", "batch_size", "input_len", "output_len", "decode_steps")
        )
        if actual != expected_case:
            raise ValueError(f"profile.cases[{cell}]: expected {expected_case!r}, found {actual!r}")
    return by_cell[EXPECTED_DECODE_CELL]


def _is_collective(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in COLLECTIVE_MARKERS)


def _nvtx_service(
    run_dir: Path,
    decode_case: dict[str, Any],
    expected_label: str,
) -> tuple[int, int, dict[str, Any]]:
    """Select one exact generation boundary and return separated service."""

    with sqlite3.connect(run_dir / "profile.sqlite") as connection:
        session_epoch_row = connection.execute(
            "SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME"
        ).fetchall()
        if len(session_epoch_row) != 1:
            raise ValueError("profile.sqlite: expected one session epoch")
        session_epoch_ns = int(session_epoch_row[0][0])
        outer_start = int(decode_case["started_epoch_ns"]) - session_epoch_ns
        outer_end = int(decode_case["finished_epoch_ns"]) - session_epoch_ns
        boundary_rows = connection.execute(
            """
            SELECT n.start, n.end, COALESCE(n.text, names.value)
            FROM NVTX_EVENTS AS n
            LEFT JOIN StringIds AS names ON n.textId = names.id
            WHERE n.end IS NOT NULL AND n.start >= ? AND n.end <= ?
              AND COALESCE(n.text, names.value) = ?
            ORDER BY n.start
            """,
            (outer_start, outer_end, expected_label),
        ).fetchall()
        if len(boundary_rows) != 1:
            raise ValueError(
                "profile.sqlite: expected one exact speculative-generation boundary, "
                f"found {len(boundary_rows)}"
            )
        boundary_start, boundary_end, boundary_label = boundary_rows[0]
        correlation_rows = connection.execute(
            """
            SELECT DISTINCT correlationId
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
            WHERE correlationId IS NOT NULL AND start >= ? AND end <= ?
            ORDER BY correlationId
            """,
            (boundary_start, boundary_end),
        ).fetchall()
        correlation_ids = [int(row[0]) for row in correlation_rows]
        if not correlation_ids:
            raise ValueError("profile.sqlite: generation boundary has no runtime correlations")
        placeholders = ",".join("?" for _ in correlation_ids)
        kernel_rows = connection.execute(
            f"""
            SELECT kernels.start, kernels.end, names.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
            JOIN StringIds AS names ON kernels.demangledName = names.id
            WHERE kernels.correlationId IN ({placeholders})
            ORDER BY kernels.start, kernels.deviceId, kernels.streamId
            """,
            correlation_ids,
        ).fetchall()
    if not kernel_rows:
        raise ValueError("profile.sqlite: generation boundary has no correlated GPU kernels")

    noncollective_ns = sum(
        int(end) - int(start)
        for start, end, name in kernel_rows
        if not _is_collective(str(name))
    )
    collective_ns = sum(
        int(end) - int(start)
        for start, end, name in kernel_rows
        if _is_collective(str(name))
    )
    if noncollective_ns <= 0:
        raise ValueError("profile.sqlite: noncollective generation service is not positive")
    return (
        noncollective_ns,
        collective_ns,
        {
            "basis": "exact-nvtx-runtime-correlation",
            "label": str(boundary_label),
            "runtime_correlation_count": len(correlation_ids),
            "kernel_record_count": len(kernel_rows),
        },
    )


def recover_mtp_service(run_dir: Path) -> dict[str, Any]:
    """Return one measured MTP service record or fail on any boundary ambiguity."""

    profile = _read_json(run_dir / "profile.json")
    decode_case = _validate_profile(profile)
    if (run_dir / "weight_files.txt").read_text(encoding="utf-8").strip():
        raise ValueError("weight_files.txt: dummy-weight isolation was violated")

    noncollective_ns, collective_ns, boundary = _nvtx_service(
        run_dir,
        decode_case,
        EXPECTED_NVTX_LABEL,
    )
    return {
        "schema": SCHEMA,
        "cell": "deepseek-ep72-mtp-decode-b16-c4000",
        "evidence_class": "MEASURED",
        "component_class": "DISCLOSED",
        "measured_service_ps": noncollective_ns * 1000,
        "collective_service_ps": collective_ns * 1000,
        "physical_key": {
            "model": EXPECTED_MODEL,
            "revision": EXPECTED_REVISION,
            "reduced_layers": 4,
            "tensor_parallel": 1,
            "batch_size": 16,
            "remote_kv_tokens_per_request": 4000,
            "speculative_tokens": 1,
        },
        "boundary": boundary,
        "sources": _source_manifest(run_dir),
        "original_compact_analysis": {
            "status": "BLOCKED",
            "reason": (
                "the staged analyzer expected generation_16(16), while one speculative "
                "token produces the exact runtime label generation_16(32)"
            ),
        },
        "lookup_pricing": "FORBIDDEN_BY_FREEZE",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = recover_mtp_service(args.run_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
