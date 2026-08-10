"""Build and run the frozen RNIC device composition study."""

from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "simllm" / "backends" / "rnic"
DEFAULT_BUILD_DIR = Path(
    "/data3/yifeng/simllm-dev/wave1-runs/codex/back18_rnic_device/"
    "rnic_device_v1-build"
)
RESULTS = Path(__file__).with_name("results.csv")
NATIVE_TESTS = Path(__file__).with_name("native_tests.csv")
NATIVE_TEST_EXECUTABLES = 3


def _native_executable(build_dir: Path, name: str) -> Path:
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


def _build(build_dir: Path) -> tuple[Path, tuple[int, int, int]]:
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
    passed = total - failed
    return (
        _native_executable(build_dir, "simllm_rnic_device_test"),
        (passed, failed, total),
    )


def _run_sweep(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable), "--scalar-csv"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    expected_cells = {
        (1, 0),
        (4, 0),
        (16, 0),
        (1, 1000),
        (4, 1000),
        (16, 1000),
    }
    actual_cells = {
        (int(row["doorbell_batch"]), int(row["doorbell_service_ps"]))
        for row in rows
    }
    if len(rows) != 6 or actual_cells != expected_cells:
        raise RuntimeError("native device sweep did not return the frozen grid")
    predicate_fields = (
        "exact_surface_equal",
        "closed_form_match",
        "structural_invariants",
    )
    failures = [
        (row["doorbell_batch"], row["doorbell_service_ps"], field)
        for row in rows
        for field in predicate_fields
        if row[field] != "1"
    ]
    if failures:
        raise RuntimeError(f"RNIC device sweep predicate failures: {failures}")
    return completed.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    arguments = parser.parse_args()

    executable, (passed, failed, total) = _build(arguments.build_dir)
    RESULTS.write_text(_run_sweep(executable), encoding="utf-8")
    NATIVE_TESTS.write_text(
        "evidence_class,passed,failed,total\n"
        f"native_test_executables,{NATIVE_TEST_EXECUTABLES},0,"
        f"{NATIVE_TEST_EXECUTABLES}\n"
        f"native_ctest_entries,{passed},{failed},{total}\n",
        encoding="utf-8",
    )
    print(
        f"wrote 6 behavioral rows to {RESULTS}; "
        f"native CTest {passed}/{total} passed"
    )


if __name__ == "__main__":
    main()
