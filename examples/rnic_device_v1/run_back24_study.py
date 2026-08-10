"""Run the frozen BACK-24 RNIC device rejection study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "simllm" / "backends" / "rnic"
DEFAULT_OUT = Path(
    "/data3/yifeng/simllm-dev/wave2-runs/"
    "codex/back8_session_records/back24"
)
INVALID_TERMINALS = (
    "unknown_token",
    "duplicate_token",
    "cross_wqe",
)
FUTURE_EVENT_TIMES_PS = (110, 1010)
CONTINUATION_TIME_PS = 20
EXPECTED_EXCEPTIONS = {
    "unknown_token": (
        "std::logic_error",
        "unknown or duplicate RNIC network token",
    ),
    "duplicate_token": (
        "std::logic_error",
        "unknown or duplicate RNIC network token",
    ),
    "cross_wqe": (
        "std::logic_error",
        "RNIC network token/WQE mismatch",
    ),
}
SNAPSHOT_SURFACES = (
    "wqe_records",
    "counters",
    "evidence",
    "port_ledger",
    "next_event_time",
    "pending_physical_work",
    "fatal",
    "occupied_sq_entries",
    "completion_queue_depth",
    "unpublished_wqe_count",
    "pcie_state",
)
FROZEN_ARTIFACT_DIGESTS = {
    "results.csv": (
        "7a0b8423d0a99de9538047f307bb7fd2f20c8d19bd408ef90fe02199da868934"
    ),
    "native_tests.csv": (
        "969963477314bfb723770556a02e4f038c7220820d522ae60dfa8c80744a202d"
    ),
}
RESULT_NAME = "back24_results.json"
ROW_NAME = "back24_rows.csv"
ROW_FIELDS = (
    "invalid_terminal",
    "future_event_time_ps",
    "exception_type",
    "exception_message",
    "exception_identity",
    "pre_probe_zero",
    "wqe_records_equal",
    "counters_equal",
    "evidence_equal",
    "port_ledger_equal",
    "physical_state_equal",
    "pcie_state_equal",
    "post_probe_zero",
    "continuation_succeeded",
    "continuation_surface_equal",
    "terminal_delta_ps",
    "cqe_delta_ps",
    "invariants_valid",
)
FATAL_BOOLEAN_FIELDS = (
    "exception_identity",
    "pre_probe_zero",
    "wqe_records_equal",
    "counters_equal",
    "evidence_equal",
    "port_ledger_equal",
    "physical_state_equal",
    "pcie_state_equal",
    "post_probe_zero",
    "invariants_valid",
)
SCORED_BOOLEAN_FIELDS = (
    "continuation_succeeded",
    "continuation_surface_equal",
)


def _validate_registry(out: Path) -> None:
    cells = {
        (terminal, event_time_ps)
        for terminal in INVALID_TERMINALS
        for event_time_ps in FUTURE_EVENT_TIMES_PS
    }
    if len(cells) != 6:
        raise AssertionError("BACK-24 registry must contain six unique cells")
    if set(EXPECTED_EXCEPTIONS) != set(INVALID_TERMINALS):
        raise AssertionError("every invalid terminal needs an exception identity")
    if len(SNAPSHOT_SURFACES) != 11 or len(set(SNAPSHOT_SURFACES)) != 11:
        raise AssertionError("BACK-24 snapshot inventory drifted")
    if set(FROZEN_ARTIFACT_DIGESTS) != {"results.csv", "native_tests.csv"}:
        raise AssertionError("BACK-24 frozen artifact inventory drifted")
    if CONTINUATION_TIME_PS != 20:
        raise AssertionError("BACK-24 continuation time drifted")
    for name, expected_digest in FROZEN_ARTIFACT_DIGESTS.items():
        artifact = Path(__file__).with_name(name)
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise AssertionError(
                f"frozen artifact digest drifted for {name}: {digest}"
            )
    data_root = Path("/data3/yifeng").resolve()
    try:
        out.resolve().relative_to(data_root)
    except ValueError as error:
        raise ValueError("study output must remain under /data3/yifeng") from error


def _native_executable(build_dir: Path) -> Path:
    name = "simllm_rnic_device_test"
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


def _run_native(executable: Path) -> tuple[str, list[dict[str, str]]]:
    completed = subprocess.run(
        [str(executable), "--back24-csv"],
        check=True,
        capture_output=True,
        text=True,
    )
    reader = csv.DictReader(io.StringIO(completed.stdout))
    if tuple(reader.fieldnames or ()) != ROW_FIELDS:
        raise RuntimeError("BACK-24 native row schema drifted")
    return completed.stdout, list(reader)


def _validate_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    expected_cells = {
        (terminal, str(event_time_ps))
        for terminal in INVALID_TERMINALS
        for event_time_ps in FUTURE_EVENT_TIMES_PS
    }
    actual_cells = {
        (row["invalid_terminal"], row["future_event_time_ps"])
        for row in rows
    }
    if len(rows) != 6 or actual_cells != expected_cells:
        raise AssertionError("BACK-24 native study did not return the frozen grid")

    fatal_failures: list[str] = []
    scored_failures: list[str] = []
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        terminal = row["invalid_terminal"]
        cell = f"{terminal}@{row['future_event_time_ps']}"
        expected_type, expected_message = EXPECTED_EXCEPTIONS[terminal]
        if row["exception_type"] != expected_type:
            fatal_failures.append(f"{cell}:exception_type")
        if row["exception_message"] != expected_message:
            fatal_failures.append(f"{cell}:exception_message")
        for field in FATAL_BOOLEAN_FIELDS:
            if row[field] != "1":
                fatal_failures.append(f"{cell}:{field}")
        for field in SCORED_BOOLEAN_FIELDS:
            if row[field] != "1":
                scored_failures.append(f"{cell}:{field}")
        for field in ("terminal_delta_ps", "cqe_delta_ps"):
            if row[field] != "0":
                scored_failures.append(f"{cell}:{field}")
        normalized_rows.append(
            {
                "invalid_terminal": terminal,
                "future_event_time_ps": int(row["future_event_time_ps"]),
                "exception_type": row["exception_type"],
                "exception_message": row["exception_message"],
                "terminal_delta_ps": int(row["terminal_delta_ps"]),
                "cqe_delta_ps": int(row["cqe_delta_ps"]),
                "fatal_guards_passed": all(
                    row[field] == "1" for field in FATAL_BOOLEAN_FIELDS
                ),
                "continuation_equal": all(
                    row[field] == "1" for field in SCORED_BOOLEAN_FIELDS
                ),
            }
        )
    if fatal_failures:
        raise AssertionError(f"BACK-24 fatal guard failures: {fatal_failures}")
    if scored_failures:
        raise AssertionError(
            f"BACK-24 scored relation failures: {scored_failures}"
        )
    return {
        "family": "post_rejection_clock_continuity",
        "passed": len(rows),
        "total": len(rows),
        "timestamp_delta_band_ps": [0, 0],
        "rows": normalized_rows,
    }


def _artifact_digests() -> dict[str, str]:
    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in FROZEN_ARTIFACT_DIGESTS
    }


def _run(out: Path) -> dict[str, Any]:
    before = _artifact_digests()
    out.mkdir(parents=True, exist_ok=True)
    executable, native_ctest = _build(out / "build")
    native_csv, rows = _run_native(executable)
    (out / ROW_NAME).write_text(native_csv, encoding="utf-8")
    scored = _validate_rows(rows)
    after = _artifact_digests()
    if before != FROZEN_ARTIFACT_DIGESTS or after != before:
        raise AssertionError("accepted RNIC device artifacts changed during study")
    return {
        "schema": "simllm-rnic-back24-study-v1",
        "scored": scored,
        "fatal_guards": {
            "instances": len(rows),
            "fields_per_instance": len(FATAL_BOOLEAN_FIELDS) + 2,
            "passed": True,
        },
        "native_ctest": native_ctest,
        "tracked_artifact_sha256": after,
        "genuine_risk": {
            "plausible_failures": len(rows),
            "relations": len(rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate the frozen registry without creating outputs",
    )
    arguments = parser.parse_args()
    _validate_registry(arguments.out)
    if arguments.check_only:
        print("BACK-24 study registry check passed; no results produced")
        return
    report = _run(arguments.out.resolve())
    result_path = arguments.out.resolve() / RESULT_NAME
    result_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "BACK-24 study passed 6/6 scored clock-continuity relations; "
        f"wrote {result_path}"
    )


if __name__ == "__main__":
    main()
