"""Run the frozen BACK-20 submission-source component study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "simllm" / "backends" / "rnic"
RESULTS = Path(__file__).with_name("results.csv")
SIMLLM_BASE_COMMIT = "dba467984b9d82ba374dce5d64d687ca59074135"
NCCL_COMMIT = "5067397c2676d5aed50042fc39e5c8ee96eb0027"
PRODUCER_SHAPES = (
    "host_cpu_driver",
    "cpu_proxy",
    "gpu_initiated",
)
BATCH_SIZES = (1, 4)
FROZEN_ARTIFACT_DIGESTS = {
    "examples/rnic_wq_v1/results.csv": (
        "598f0e10ca4e5a83a9dfb8ed8289e25cdc4c80fc24f92f2f70db967724be5682"
    ),
    "examples/rnic_pcie_v1/results.csv": (
        "464b92fd5327287db6b5e71a5449add5b893285bc3c4bcdf6a4950355339a5e2"
    ),
    "examples/rnic_device_v1/results.csv": (
        "7a0b8423d0a99de9538047f307bb7fd2f20c8d19bd408ef90fe02199da868934"
    ),
    "examples/rnic_device_v1/native_tests.csv": (
        "969963477314bfb723770556a02e4f038c7220820d522ae60dfa8c80744a202d"
    ),
    "examples/rnic_session_records_v1/results.json": (
        "d83575d1c873d3375bc24819c4d6eca0b85ea3a414fe8578f30262268a39fdf6"
    ),
    "examples/rnic_hostmem_v1/results.csv": (
        "1bc7bcc8e72b7aef9fda1ed7e6ca2078d60c48a00377cbf8dfded75ff4d2fa53"
    ),
}
ROW_FIELDS = (
    "producer_shape",
    "batch_size",
    "producer_kind",
    "producer_id",
    "descriptor_writer_kind",
    "descriptor_writer_id",
    "descriptor_queue_endpoint",
    "sq_endpoint",
    "cq_endpoint",
    "doorbell_endpoint",
    "data_endpoint",
    "uar_mapping_owner",
    "cq_consumer_kind",
    "cq_consumer_id",
    "rnic_requester_id",
    "qpn",
    "submission_records",
    "cq_consumption_records",
    "completed_wqes",
    "qpc_fetches",
    "qpc_icm_transactions",
    "qpc_mkey_events",
    "qpc_mpt_events",
    "qpc_mtt_events",
    "data_mkey_events",
    "data_mpt_events",
    "data_mtt_events",
    "qpc_stays_host_icm",
    "exactly_one_cq_consumer",
    "identities_separate_from_qpn",
    "invariants_valid",
)
SHAPE_EXPECTATIONS = {
    "host_cpu_driver": {
        "producer_kind": "host_cpu_driver",
        "producer_id": "7101",
        "descriptor_writer_kind": "none",
        "descriptor_writer_id": "0",
        "descriptor_queue_endpoint": "none",
        "sq_endpoint": "host_pinned_memory",
        "cq_endpoint": "host_pinned_memory",
        "doorbell_endpoint": "host_pinned_memory",
        "data_endpoint": "host_pinned_memory",
        "uar_mapping_owner": "host_cpu",
        "cq_consumer_kind": "host_cpu_driver",
        "cq_consumer_id": "8101",
    },
    "cpu_proxy": {
        "producer_kind": "cpu_proxy",
        "producer_id": "7102",
        "descriptor_writer_kind": "gpu",
        "descriptor_writer_id": "7202",
        "descriptor_queue_endpoint": "host_pinned_memory",
        "sq_endpoint": "host_pinned_memory",
        "cq_endpoint": "host_pinned_memory",
        "doorbell_endpoint": "host_pinned_memory",
        "data_endpoint": "gpu_memory",
        "uar_mapping_owner": "host_cpu",
        "cq_consumer_kind": "cpu_proxy",
        "cq_consumer_id": "8102",
    },
    "gpu_initiated": {
        "producer_kind": "gpu",
        "producer_id": "7103",
        "descriptor_writer_kind": "none",
        "descriptor_writer_id": "0",
        "descriptor_queue_endpoint": "none",
        "sq_endpoint": "gpu_memory",
        "cq_endpoint": "gpu_memory",
        "doorbell_endpoint": "gpu_memory",
        "data_endpoint": "gpu_memory",
        "uar_mapping_owner": "gpu",
        "cq_consumer_kind": "gpu",
        "cq_consumer_id": "8103",
    },
}


def _wave3_root() -> Path:
    configured = os.environ.get("SIMLLM_WAVE3_RUN_ROOT")
    if not configured:
        raise RuntimeError(
            "SIMLLM_WAVE3_RUN_ROOT must name the external wave-3 run root"
        )
    return Path(configured).resolve()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_registry(out: Path) -> None:
    cells = {
        (shape, batch_size)
        for shape in PRODUCER_SHAPES
        for batch_size in BATCH_SIZES
    }
    if len(cells) != 6:
        raise AssertionError("BACK-20 registry must contain six unique cells")
    if len(PRODUCER_SHAPES) * sum(BATCH_SIZES) != 15:
        raise AssertionError("BACK-20 registry must contain fifteen QPC fetches")
    if set(SHAPE_EXPECTATIONS) != set(PRODUCER_SHAPES):
        raise AssertionError("BACK-20 shape registry is incomplete")
    if len(ROW_FIELDS) != len(set(ROW_FIELDS)):
        raise AssertionError("BACK-20 row fields must be unique")
    for commit in (SIMLLM_BASE_COMMIT, NCCL_COMMIT):
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise AssertionError("BACK-20 source audit commits must be full hashes")
    subprocess.run(
        ["git", "cat-file", "-e", f"{SIMLLM_BASE_COMMIT}^{{commit}}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    for relative, expected in FROZEN_ARTIFACT_DIGESTS.items():
        actual = _digest(REPO_ROOT / relative)
        if actual != expected:
            raise AssertionError(
                f"frozen artifact digest drifted for {relative}: {actual}"
            )
    try:
        out.resolve().relative_to(_wave3_root())
    except ValueError as error:
        raise ValueError(
            "BACK-20 output must remain under SIMLLM_WAVE3_RUN_ROOT"
        ) from error


def _native_executable(build_dir: Path) -> Path:
    name = "simllm_rnic_submission_test"
    candidates = (
        build_dir / name,
        build_dir / f"{name}.exe",
        build_dir / "Release" / name,
        build_dir / "Release" / f"{name}.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    joined = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(f"native executable not found; checked {joined}")


def _build(build_dir: Path) -> tuple[Path, dict[str, int]]:
    subprocess.run(
        [
            "cmake",
            "-S",
            str(SOURCE_DIR),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DSIMLLM_RNIC_BUILD_TESTS=ON",
            "-DSIMLLM_RNIC_BUILD_TOOLS=ON",
            "-DSIMLLM_RNIC_WARNINGS_AS_ERRORS=ON",
        ],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--parallel",
        ],
        check=True,
    )
    completed = subprocess.run(
        [
            "ctest",
            "--test-dir",
            str(build_dir),
            "-C",
            "Release",
            "--output-on-failure",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print(completed.stdout, end="")
    match = re.search(
        r"100% tests passed, (\d+) tests failed out of (\d+)",
        completed.stdout,
    )
    if match is None:
        raise RuntimeError("could not parse the native CTest summary")
    failed = int(match.group(1))
    total = int(match.group(2))
    return _native_executable(build_dir), {
        "passed": total - failed,
        "failed": failed,
        "total": total,
    }


def _read_native_rows(executable: Path) -> tuple[str, list[dict[str, str]]]:
    completed = subprocess.run(
        [str(executable), "--study-csv"],
        check=True,
        capture_output=True,
        text=True,
    )
    reader = csv.DictReader(io.StringIO(completed.stdout))
    if tuple(reader.fieldnames or ()) != ROW_FIELDS:
        raise RuntimeError("BACK-20 native row schema drifted")
    return completed.stdout, list(reader)


def _require_integer(row: dict[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, ValueError) as error:
        raise AssertionError(f"BACK-20 field {field} must be an integer") from error


def _validate_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    expected_cells = {
        (shape, str(batch_size))
        for shape in PRODUCER_SHAPES
        for batch_size in BATCH_SIZES
    }
    actual_cells = {
        (row["producer_shape"], row["batch_size"]) for row in rows
    }
    if len(rows) != 6 or actual_cells != expected_cells:
        raise AssertionError("BACK-20 native rows differ from the frozen grid")

    qpc_asymmetry_passed = 0
    structural_failures: list[str] = []
    normalized: list[dict[str, str | int]] = []
    for row in rows:
        shape = row["producer_shape"]
        batch_size = _require_integer(row, "batch_size")
        cell = f"S={shape},B={batch_size}"
        expected_strings = SHAPE_EXPECTATIONS[shape]
        for field, expected in expected_strings.items():
            if row[field] != expected:
                structural_failures.append(
                    f"{cell}:{field}={row[field]} expected {expected}"
                )
        expected_integers = {
            "rnic_requester_id": 9100,
            "qpn": 19,
            "submission_records": batch_size,
            "cq_consumption_records": batch_size,
            "completed_wqes": batch_size,
            "qpc_fetches": batch_size,
            "qpc_icm_transactions": batch_size,
            "qpc_mkey_events": 0,
            "qpc_mpt_events": 0,
            "qpc_mtt_events": 0,
            "data_mkey_events": batch_size,
            "data_mpt_events": batch_size,
            "data_mtt_events": batch_size,
            "qpc_stays_host_icm": 1,
            "exactly_one_cq_consumer": 1,
            "identities_separate_from_qpn": 1,
            "invariants_valid": 1,
        }
        values = {
            field: _require_integer(row, field) for field in expected_integers
        }
        qpc_ok = (
            values["qpc_fetches"] == batch_size
            and values["qpc_icm_transactions"] == batch_size
            and values["qpc_mkey_events"] == 0
            and values["qpc_mpt_events"] == 0
            and values["qpc_mtt_events"] == 0
        )
        qpc_asymmetry_passed += int(qpc_ok)
        for field, expected in expected_integers.items():
            if values[field] != expected:
                structural_failures.append(
                    f"{cell}:{field}={values[field]} expected {expected}"
                )
        normalized.append({**row, **values, "batch_size": batch_size})
    if qpc_asymmetry_passed != 6:
        raise AssertionError("BACK-20 QPC translation asymmetry failed")
    if structural_failures:
        raise AssertionError(
            "BACK-20 fatal structural failures: "
            + "; ".join(structural_failures)
        )
    return {
        "rows": normalized,
        "qpc_asymmetry_passed": qpc_asymmetry_passed,
        "qpc_asymmetry_total": 6,
        "active_qpc_fetches": len(PRODUCER_SHAPES) * sum(BATCH_SIZES),
    }


def _run(out: Path) -> dict[str, Any]:
    before = {
        relative: _digest(REPO_ROOT / relative)
        for relative in FROZEN_ARTIFACT_DIGESTS
    }
    out.mkdir(parents=True, exist_ok=True)
    executable, ctest = _build(out / "build")
    raw_csv, rows = _read_native_rows(executable)
    checked = _validate_rows(rows)
    (out / "raw_results.csv").write_bytes(raw_csv.encode("utf-8"))
    RESULTS.write_bytes(raw_csv.encode("utf-8"))
    after = {
        relative: _digest(REPO_ROOT / relative)
        for relative in FROZEN_ARTIFACT_DIGESTS
    }
    identity = {
        relative: before[relative] == after[relative] == expected
        for relative, expected in FROZEN_ARTIFACT_DIGESTS.items()
    }
    if not all(identity.values()):
        raise AssertionError("BACK-20 accepted artifact byte identity failed")
    return {
        **checked,
        "artifact_identity": identity,
        "artifact_identity_passed": sum(identity.values()),
        "artifact_identity_total": len(identity),
        "ctest": ctest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate the frozen registry without creating outputs",
    )
    arguments = parser.parse_args()
    _validate_registry(arguments.out)
    if arguments.check_only:
        print(
            "RNIC submission-source registry check passed; "
            "no artifacts were produced"
        )
        return
    summary = _run(arguments.out.resolve())
    print(
        "BACK-20 passed "
        f"{summary['qpc_asymmetry_passed']}/"
        f"{summary['qpc_asymmetry_total']} translation cells and "
        f"{summary['artifact_identity_passed']}/"
        f"{summary['artifact_identity_total']} artifact identities"
    )


if __name__ == "__main__":
    main()
