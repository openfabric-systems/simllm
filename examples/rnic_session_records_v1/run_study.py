"""Run the frozen RNIC session-record component study."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "simllm" / "backends" / "rnic"
DEFAULT_OUT = Path(
    "/data3/yifeng/simllm-dev/wave2-runs/codex/back8_session_records"
)
RESULTS = Path(__file__).with_name("results.json")
HTSIM_COMMIT = "8c3f8b231a6a9311ffc1e7969a003dcba724b50d"
SIMLLM_BASE_COMMIT = "6aa3a76"
POLICIES = ("rnic-nn", "rnic-cn", "dcqcn")
SQ_DEPTHS = (32, 64)
DOORBELL_SERVICES_PS = (0, 1000)
ARTIFACT_NAMES = (
    "completion_csv",
    "canonical_completion",
    "step_results",
    "replay_summary",
)


def _validate_registry(out: Path) -> None:
    cells = {
        (sq_depth, doorbell_service_ps, policy)
        for sq_depth in SQ_DEPTHS
        for doorbell_service_ps in DOORBELL_SERVICES_PS
        for policy in POLICIES
    }
    if len(cells) != 12:
        raise AssertionError("frozen hash sweep must contain 12 rows")
    if len(ARTIFACT_NAMES) != 4 or len(set(ARTIFACT_NAMES)) != 4:
        raise AssertionError("frozen bypass inventory must contain four unique artifacts")
    data_root = Path("/data3/yifeng").resolve()
    try:
        out.resolve().relative_to(data_root)
    except ValueError as error:
        raise ValueError("study output must remain under /data3/yifeng") from error
    pinned_htsim = subprocess.run(
        ["git", "rev-parse", f"{SIMLLM_BASE_COMMIT}:third_party/htsim"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if pinned_htsim != HTSIM_COMMIT:
        raise AssertionError(
            f"pinned HTSim source audit drifted: expected {HTSIM_COMMIT}, got {pinned_htsim}"
        )
    for relative in (
        "simllm/backends/rnic/include/simllm/rnic/rnic_device.h",
        "simllm/backends/rnic/include/simllm/rnic/work_queue.h",
        "simllm/backends/rnic/include/simllm/rnic/pcie_fabric.h",
        "examples/rnic_live_v1/expectations.md",
    ):
        if not (REPO_ROOT / relative).is_file():
            raise FileNotFoundError(f"frozen source-audit input is missing: {relative}")


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


def _build(build_dir: Path) -> Path:
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
    subprocess.run(
        [
            "ctest",
            "--test-dir",
            str(build_dir),
            "-C",
            "Release",
            "--output-on-failure",
        ],
        check=True,
    )
    return _native_executable(build_dir, "simllm_rnic_session_record_test")


def _native_report(executable: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(executable), "--study-json"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise TypeError("native study report must be a JSON object")
    return value


def _run(out: Path) -> dict[str, Any]:
    from simllm.backends.rnic_records import validate_session_record_study

    out.mkdir(parents=True, exist_ok=True)
    native = _native_report(_build(out / "build"))
    return validate_session_record_study(
        native,
        policies=POLICIES,
        sq_depths=SQ_DEPTHS,
        doorbell_services_ps=DOORBELL_SERVICES_PS,
        artifact_names=ARTIFACT_NAMES,
    )


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
        print("RNIC session-record study registry check passed; no results produced")
        return
    report = _run(arguments.out.resolve())
    RESULTS.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote validated component evidence to {RESULTS}")


if __name__ == "__main__":
    main()
