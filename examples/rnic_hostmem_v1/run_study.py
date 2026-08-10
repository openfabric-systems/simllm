"""Run the frozen BACK-19 host-memory component study."""

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
SIMLLM_BASE_COMMIT = "fc282efc91573638de7dcfae2befee1cf022011b"
RDMA_CORE_COMMIT = "c1c5bf1f480312c07ed4d23f0feecf8b5fd73289"
LINUX_COMMIT = "db2ddb87143519e20a95aa36c60b36107b736a58"
PAGE_SIZES = (4096, 2097152)
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
}
ROW_FIELDS = (
    "page_size_bytes",
    "batch_size",
    "qpc_fetches",
    "qpc_icm_transactions",
    "qpc_mkey_events",
    "qpc_mpt_events",
    "qpc_mtt_events",
    "sq_page_list_events",
    "data_mkey_events",
    "data_mpt_events",
    "data_mtt_events",
    "cq_page_list_events",
    "mtt_mpt_transactions",
    "wqe_read_transactions",
    "payload_read_transactions",
    "cqe_write_transactions",
    "doorbell_record_transactions",
    "uar_transactions",
    "registration_events",
    "teardown_events",
    "second_data_page_selected",
    "invariants_valid",
)


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
        (page_size, batch_size)
        for page_size in PAGE_SIZES
        for batch_size in BATCH_SIZES
    }
    if len(cells) != 4:
        raise AssertionError("BACK-19 registry must contain four unique cells")
    if sum(BATCH_SIZES) * len(PAGE_SIZES) != 10:
        raise AssertionError("BACK-19 registry must contain ten QPC fetches")
    if len(ROW_FIELDS) != len(set(ROW_FIELDS)):
        raise AssertionError("BACK-19 row fields must be unique")
    for commit in (SIMLLM_BASE_COMMIT, RDMA_CORE_COMMIT, LINUX_COMMIT):
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise AssertionError("BACK-19 source audit commits must be full hashes")
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
            "BACK-19 output must remain under SIMLLM_WAVE3_RUN_ROOT"
        ) from error


def _native_executable(build_dir: Path) -> Path:
    name = "simllm_rnic_host_memory_test"
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
        raise RuntimeError("BACK-19 native row schema drifted")
    return completed.stdout, list(reader)


def _require_integer(row: dict[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, ValueError) as error:
        raise AssertionError(f"BACK-19 field {field} must be an integer") from error


def _validate_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    expected_cells = {
        (str(page_size), str(batch_size))
        for page_size in PAGE_SIZES
        for batch_size in BATCH_SIZES
    }
    actual_cells = {
        (row["page_size_bytes"], row["batch_size"]) for row in rows
    }
    if len(rows) != 4 or actual_cells != expected_cells:
        raise AssertionError("BACK-19 native rows differ from the frozen grid")

    qpc_asymmetry_passed = 0
    structural_failures: list[str] = []
    normalized: list[dict[str, int]] = []
    for row in rows:
        page_size = _require_integer(row, "page_size_bytes")
        batch_size = _require_integer(row, "batch_size")
        cell = f"P={page_size},B={batch_size}"
        expected = {
            "qpc_fetches": batch_size,
            "qpc_icm_transactions": batch_size,
            "qpc_mkey_events": 0,
            "qpc_mpt_events": 0,
            "qpc_mtt_events": 0,
            "sq_page_list_events": batch_size,
            "data_mkey_events": batch_size,
            "data_mpt_events": batch_size,
            "data_mtt_events": batch_size,
            "cq_page_list_events": batch_size,
            "mtt_mpt_transactions": 4 * batch_size,
            "wqe_read_transactions": batch_size,
            "payload_read_transactions": batch_size,
            "cqe_write_transactions": batch_size,
            "doorbell_record_transactions": 1,
            "uar_transactions": 1,
            "registration_events": 6,
            "teardown_events": 6,
            "second_data_page_selected": 1,
            "invariants_valid": 1,
        }
        values = {field: _require_integer(row, field) for field in expected}
        qpc_ok = (
            values["qpc_fetches"] == batch_size
            and values["qpc_icm_transactions"] == batch_size
            and values["qpc_mkey_events"] == 0
            and values["qpc_mpt_events"] == 0
            and values["qpc_mtt_events"] == 0
        )
        qpc_asymmetry_passed += int(qpc_ok)
        for field, expected_value in expected.items():
            if values[field] != expected_value:
                structural_failures.append(
                    f"{cell}:{field}={values[field]} expected {expected_value}"
                )
        normalized.append(
            {
                "page_size_bytes": page_size,
                "batch_size": batch_size,
                **values,
            }
        )
    if qpc_asymmetry_passed != 4:
        raise AssertionError("BACK-19 QPC translation asymmetry failed")
    if structural_failures:
        raise AssertionError(
            "BACK-19 fatal structural failures: " + "; ".join(structural_failures)
        )
    return {
        "rows": normalized,
        "qpc_asymmetry_passed": qpc_asymmetry_passed,
        "qpc_asymmetry_total": 4,
        "active_qpc_fetches": sum(BATCH_SIZES) * len(PAGE_SIZES),
    }


def _run(out: Path) -> dict[str, Any]:
    before = {
        relative: _digest(REPO_ROOT / relative)
        for relative in FROZEN_ARTIFACT_DIGESTS
    }
    out.mkdir(parents=True, exist_ok=True)
    executable, ctest = _build(out / "build")
    raw_csv, checked = _read_native_rows(executable)
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
        raise AssertionError("BACK-19 accepted artifact byte identity failed")
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
            "RNIC host-memory registry check passed; "
            "no artifacts were produced"
        )
        return
    summary = _run(arguments.out.resolve())
    print(
        "BACK-19 passed "
        f"{summary['qpc_asymmetry_passed']}/"
        f"{summary['qpc_asymmetry_total']} translation cells and "
        f"{summary['artifact_identity_passed']}/"
        f"{summary['artifact_identity_total']} artifact identities"
    )


if __name__ == "__main__":
    main()
