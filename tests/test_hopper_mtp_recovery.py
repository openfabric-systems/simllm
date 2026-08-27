from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/hopper_kernel_cycle_candidate_v1/recover_mtp_service.py"
SPEC = importlib.util.spec_from_file_location("hopper_mtp_recovery", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fixture(tmp_path: Path, *, label: str = MODULE.EXPECTED_NVTX_LABEL) -> Path:
    profile = {
        "model": MODULE.EXPECTED_MODEL,
        "model_key": "deepseek-v3",
        "tensor_parallel_size": 1,
        "mode": "graph",
        "shape_set": "deepseek",
        "deepseek_suite": "mtp",
        "reduced_layers": 4,
        "phase": "profile",
        "model_config": {
            "requested_revision": MODULE.EXPECTED_REVISION,
            "resolved_revision": MODULE.EXPECTED_REVISION,
            "config_sha256": MODULE.EXPECTED_CONFIG_SHA256,
            "effective_num_hidden_layers": 4,
        },
        "cases": [
            {
                "cell": MODULE.EXPECTED_BASE_CELL,
                "pool": "decode_base",
                "batch_size": 16,
                "input_len": 4000,
                "output_len": 1,
                "decode_steps": 0,
                "started_epoch_ns": 1_000,
                "finished_epoch_ns": 1_100,
            },
            {
                "cell": MODULE.EXPECTED_DECODE_CELL,
                "pool": "decode",
                "batch_size": 16,
                "input_len": 4000,
                "output_len": 2,
                "decode_steps": 1,
                "started_epoch_ns": 1_200,
                "finished_epoch_ns": 1_800,
            },
        ],
    }
    (tmp_path / "profile.json").write_text(
        json.dumps(profile) + "\n", encoding="utf-8", newline="\n"
    )
    (tmp_path / "analysis").mkdir()
    for relative in (
        "profile.nsys-rep",
        "analysis/ordered-kernels.csv",
        "harness_sha256.txt",
        "sha256.txt",
        "weight_files.txt",
    ):
        (tmp_path / relative).write_text("", encoding="utf-8", newline="\n")

    with sqlite3.connect(tmp_path / "profile.sqlite") as connection:
        connection.executescript(
            """
            CREATE TABLE TARGET_INFO_SESSION_START_TIME (utcEpochNs INTEGER);
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER, end INTEGER, correlationId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER,
                correlationId INTEGER, demangledName INTEGER
            );
            INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (1000);
            INSERT INTO StringIds VALUES (1, 'compute_kernel');
            INSERT INTO StringIds VALUES (2, 'nccl_all_reduce');
            """
        )
        connection.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, NULL)", (250, 500, label)
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?)",
            [(260, 270, 7), (280, 290, 8)],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0, 7, ?, ?)",
            [(300, 311, 7, 1), (320, 325, 8, 2)],
        )
    return tmp_path


def test_recovery_selects_only_the_exact_speculative_generation_boundary(
    tmp_path: Path,
) -> None:
    result = MODULE.recover_mtp_service(_fixture(tmp_path))

    assert result["measured_service_ps"] == 11_000
    assert result["collective_service_ps"] == 5_000
    assert result["boundary"] == {
        "basis": "exact-nvtx-runtime-correlation",
        "label": MODULE.EXPECTED_NVTX_LABEL,
        "runtime_correlation_count": 2,
        "kernel_record_count": 2,
    }
    assert result["evidence_class"] == "MEASURED"
    assert result["lookup_pricing"] == "FORBIDDEN_BY_FREEZE"
    assert len(result["sources"]) == len(MODULE.SOURCE_PATHS)


def test_recovery_rejects_the_staged_non_speculative_label(tmp_path: Path) -> None:
    run_dir = _fixture(tmp_path, label="execute_context_0(0)_generation_16(16)")

    with pytest.raises(ValueError, match="expected one exact speculative-generation"):
        MODULE.recover_mtp_service(run_dir)


def test_recovery_rejects_any_weight_file(tmp_path: Path) -> None:
    run_dir = _fixture(tmp_path)
    (run_dir / "weight_files.txt").write_text(
        "weights.safetensors\n", encoding="utf-8", newline="\n"
    )

    with pytest.raises(ValueError, match="dummy-weight isolation"):
        MODULE.recover_mtp_service(run_dir)
