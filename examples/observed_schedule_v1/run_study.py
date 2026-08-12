"""Qualify and measure the vLLM observed-schedule path."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

CAPTURE_SHA256 = "5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6"
CAPTURE_ROWS = 120
NUM_LAYERS = 24
EXPECTED_MOE_SITES = 48
EXPECTED_GENUINE_RISK_INSTANCES = 4
SERIAL_GRAPH_BYTES = 4_127
SERIAL_GRAPH_SHA256 = "aa3c836fe559973a7bf0940384c2e8a84e6af84e0fbd2c02d3b89774ee0c8e2d"
SERIAL_GOAL_BYTES = 1_880
SERIAL_GOAL_SHA256 = "7087db6780f7e34f5a559a6505eeccc15d984c7b478cd8f0bc5838053825d4b6"

REDUCTION_BANDS_PS = {
    "single-node": (1_000_000, 5_000_000),
    "cross-node": (20_000_000, 130_000_000),
}
PERTURBATION_MIN_PS = {
    "single-node": 100_000,
    "cross-node": 5_000_000,
}

SOURCE_HASHES = {
    "vllm/model_executor/models/granitemoe.py": (
        "b60e452c3f28b25aa104c88869daa25c06a7fb6ed45bd34e908fa6a8395efda1"
    ),
    "vllm/v1/worker/gpu_model_runner.py": (
        "81b7627fbe81f7aaa2f77b4bf085faa353c69d03662ebfe369536a9773bb70d0"
    ),
    "vllm/v1/worker/ubatching.py": (
        "40391241c564feb5f16c77898ae6ae152ed6e71a4682e2a406387785d8de02d7"
    ),
    "vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py": (
        "465cdf1d6cee91b2ee8c2e43abbea6e8408976e3048c10f44c089f34b415bc60"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def check_expectation_registry(capture: Path, vllm_source: Path) -> None:
    """Validate frozen inputs and literals without importing target code."""

    if not capture.is_file():
        raise SystemExit(f"capture is not a file: {capture}")
    if _sha256(capture) != CAPTURE_SHA256:
        raise SystemExit("capture SHA-256 does not match the frozen input")
    if _line_count(capture) != CAPTURE_ROWS:
        raise SystemExit("capture row count does not match the frozen input")

    if not vllm_source.is_dir():
        raise SystemExit(f"vLLM source is not a directory: {vllm_source}")
    for relative, expected in SOURCE_HASHES.items():
        source = vllm_source / relative
        if not source.is_file():
            raise SystemExit(f"audited vLLM source is missing: {relative}")
        if _sha256(source) != expected:
            raise SystemExit(f"audited vLLM source hash changed: {relative}")

    assert NUM_LAYERS == 24
    assert EXPECTED_MOE_SITES == 2 * NUM_LAYERS
    assert EXPECTED_GENUINE_RISK_INSTANCES == 2 * len(REDUCTION_BANDS_PS)
    assert set(REDUCTION_BANDS_PS) == set(PERTURBATION_MIN_PS)
    for low, high in REDUCTION_BANDS_PS.values():
        assert 0 < low <= high
    for placement, minimum in PERTURBATION_MIN_PS.items():
        assert 0 < minimum <= REDUCTION_BANDS_PS[placement][1]
    assert SERIAL_GRAPH_BYTES > SERIAL_GOAL_BYTES > 0
    assert len(SERIAL_GRAPH_SHA256) == len(SERIAL_GOAL_SHA256) == 64


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    check_expectation_registry(args.capture, args.vllm_source)
    if args.check_only:
        print("observed-schedule expectation registry: PASS")
        return 0
    raise SystemExit(
        "result mode is intentionally unavailable in the expectations-only freeze"
    )


if __name__ == "__main__":
    raise SystemExit(main())
