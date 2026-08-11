"""Run the frozen VLLM-16 combined-isolation fix-round attempt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]
MODEL_RELATIVE_PATH = (
    Path("hub")
    / "models--ibm-granite--granite-3.0-1b-a400m-instruct"
    / "snapshots"
    / "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
)
SOURCE_HASHES = {
    "platforms/__init__.py": "a2bd800acc39b3215ccb78808d43317b351f137072b03e7f0f0ab3d069d91521",
    "platforms/cpu.py": "067f92d391b1c131e12a7ba9631921e4b9dd57d3c55b1d8724e9963e2fdc9c7d",
    "config/device.py": "7b82eee02ceb5842337451a27a3d5729920c47e25e8f6bf3997f5146f9330a9c",
}
REPOSITORY_HASHES = {
    "simllm/adapters/vllm/worker.py": "07e2d26213a1899aaf2604787cd85f47a67731d660b94fd473943831e7bccd2e",
    "examples/vllm_skeleton_v1/live_smoke.py": "a43d5e6987b0322bc0a6d05d3b7046de84980f7cfd3600eb4b94b8a7d56782cc",
}
BWRAP_SHA256 = "a87328fd969d4bc9fbc62e56b15a393b2b23c7b47aa092a3ac02955a68da19e4"
MECHANISM = "device-namespace-cpu-platform"
EXPECTED_OUTPUT_TOKENS = 2
EXPECTED_STEP_SCHEMA = "atlahs-closed-loop-step-v1"
EXPECTED_SCORED = 1
FREEZE_COMMIT = "9b7f854"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_only(args: argparse.Namespace) -> None:
    model = args.cache_dir / MODEL_RELATIVE_PATH
    if not model.is_dir() or not (model / "config.json").is_file():
        raise SystemExit(f"pinned model snapshot is missing: {model}")
    if importlib.metadata.version("vllm") != "0.26.0":
        raise SystemExit("VLLM-16 fix round requires vLLM 0.26.0")
    if importlib.metadata.version("torch") != "2.11.0":
        raise SystemExit("VLLM-16 fix round requires Torch 2.11.0")
    for relative, expected in SOURCE_HASHES.items():
        path = args.vllm_package_root / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise SystemExit(f"pinned vLLM source changed: {relative}")
    for relative, expected in REPOSITORY_HASHES.items():
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise SystemExit(f"pinned repository source changed: {relative}")
    if not args.bwrap.is_file() or not args.bwrap.stat().st_mode & 0o111:
        raise SystemExit(f"bubblewrap is missing or not executable: {args.bwrap}")
    if file_sha256(args.bwrap) != BWRAP_SHA256:
        raise SystemExit("bubblewrap executable identity changed")
    version = subprocess.run(
        [str(args.bwrap), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if version != "bubblewrap 0.4.1":
        raise SystemExit("bubblewrap version changed")
    if MECHANISM != "device-namespace-cpu-platform":
        raise AssertionError("combined mechanism changed")
    if EXPECTED_OUTPUT_TOKENS != 2:
        raise AssertionError("output count changed")
    if EXPECTED_STEP_SCHEMA != "atlahs-closed-loop-step-v1":
        raise AssertionError("step schema changed")
    if EXPECTED_SCORED != 1:
        raise AssertionError("evidence denominator changed")
    print(
        f"check-only run-dir={args.run_dir}; validated frozen VLLM-16 "
        "fix-round inputs and produced no artifacts"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--vllm-package-root", type=Path, required=True)
    parser.add_argument("--bwrap", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def run_study(args: argparse.Namespace) -> dict[str, object]:
    import vllm16_smoke

    if MECHANISM != vllm16_smoke.FIX_ROUND_MECHANISM:
        raise AssertionError("fix-round mechanism disagrees with the launcher")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    baseline = vllm16_smoke._device_probe()
    attempt = vllm16_smoke._launch_attempt(args, MECHANISM)
    if attempt["mechanism"] != MECHANISM:
        raise AssertionError("launcher returned the wrong mechanism")
    passed = int(attempt["joint_passed"])
    summary = {
        "freeze_commit": FREEZE_COMMIT,
        "baseline": baseline,
        "attempt_order": [MECHANISM],
        "attempts": [attempt],
        "scored": {
            "executed": EXPECTED_SCORED,
            "passed": passed,
            "genuine_risk_numerator": EXPECTED_SCORED,
            "genuine_risk_denominator": EXPECTED_SCORED,
        },
        "vllm16_complete": passed == EXPECTED_SCORED,
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    check_only(args)
    if args.check_only:
        return
    print(json.dumps(run_study(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
