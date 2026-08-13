"""Run the frozen vLLM observed-overlap study.

This file currently holds only the frozen expectation registry. The three-arm
harness that measures anything has not been written, so the study can be
validated with ``--check-only`` and cannot produce a result.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from fractions import Fraction
from pathlib import Path

CAPTURE_SHA256 = "5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6"
CAPTURE_ROWS = 120
REPLAY_RUN_SHA256 = "b4d38a09011caf6de159c22133264d62a2727063496953f4337b17d79cfde93e"
ROUTED_EXPERTS_SHA256 = (
    "24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f"
)
VLLM_VERSION = "0.26.0"
VLLM_AUTHORED_AGAINST_COMMIT = "568afb3a13806beb53bb2e6bd518269357b237c0"
EXPECTATION_BOUNDARY_COMMIT = "4e1be35af5327c27db53ed002dc420e1de6f613b"

DP_SIZE = 8
TP_SIZE = 1
PP_SIZE = 1
NUM_LAYERS = 24
NUM_MICROBATCHES = 2
SEMANTIC_MOE_SITES = 48
DBO_MOE_INVOCATIONS = 96
DBO_DECODE_THRESHOLD = 2
DBO_PREFILL_THRESHOLD = 512

#: every nonempty translated step of the frozen replay
NONEMPTY_STEPS = 32
#: steps 1 through 23 carry r0's TPOT intervals and select DBO
R0_DECODE_STEP_INDICES = tuple(range(1, 24))
#: step 0 is the 54-token prefill; steps 24 through 31 are one-request decodes
SINGLE_BATCH_STEP_INDICES = (0, *range(24, 32))
#: 47 microbatch boundaries times eight EP ranks
CONTROL_EDGES_PER_DBO_STEP = 376
CONTROL_EDGES_TOTAL = CONTROL_EDGES_PER_DBO_STEP * len(R0_DECODE_STEP_INDICES)

NVLINK_RATE_BPS = 3_600_000_000_000
RNIC_RATE_BPS = 400_000_000_000

#: summed maximum per-source egress over every DBO collective invocation of
#: r0's 23 TPOT steps, from the frozen routed table under the landed TRAF-25
#: token-ownership correction
DBO_DECODE_EGRESS_BYTES = 15_071_232
#: summed maximum per-source egress of the 54-token prefill step
PREFILL_EGRESS_BYTES = 15_249_408

#: ceiling on the realizable overlap: a step cannot save more than the
#: communication it contains
COMM_CEILING_PS = {
    "single-node": 1_456_158,
    "cross-node": 13_105_420,
}
OVERLAP_BAND_LOW_FRACTION = Fraction(3, 4)
OVERLAP_BAND_PS = {
    "single-node": (1_092_119, 1_456_158),
    "cross-node": (9_829_065, 13_105_420),
}
OVERLAP_RATE_RATIO_BAND = (Fraction(15, 2), Fraction(21, 2))
TERMINAL_RATE_RATIO_BAND = (Fraction(19, 20), Fraction(21, 20))
OVERLAP_CEILING_ALLOWANCE = Fraction(21, 20)

#: 553,648,128 resident weight and LM-head bytes at 8 TB/s
COMPUTE_FLOOR_PS = 69_206_016
DECODE_TPOT_CEILING_PS = 150_000_000
PREFILL_TTFT_CEILING_PS = 500_000_000

EXPECTED_GENUINE_RISK_INSTANCES = 6

SERIAL_GRAPH_BYTES = 4_127
SERIAL_GRAPH_SHA256 = (
    "aa3c836fe559973a7bf0940384c2e8a84e6af84e0fbd2c02d3b89774ee0c8e2d"
)
SERIAL_GOAL_BYTES = 1_880
SERIAL_GOAL_SHA256 = (
    "7087db6780f7e34f5a559a6505eeccc15d984c7b478cd8f0bc5838053825d4b6"
)

SOURCE_HASHES = {
    "model_executor/models/granitemoe.py": (
        "b60e452c3f28b25aa104c88869daa25c06a7fb6ed45bd34e908fa6a8395efda1"
    ),
    "config/parallel.py": (
        "a6581c267ab265e24905d2f5caa514482c28359f71380c6f894ceab25aa22541"
    ),
    "v1/worker/gpu_model_runner.py": (
        "81b7627fbe81f7aaa2f77b4bf085faa353c69d03662ebfe369536a9773bb70d0"
    ),
    "v1/worker/dp_utils.py": (
        "2ba84bbf92a25e756576918bfb215c1fb387b006899885d811bdb2f774e843a9"
    ),
    "v1/worker/ubatch_utils.py": (
        "0b727aaa1c7072152e25f684ddc2fc9790c430eddd862e610c97a8e3e9febdc4"
    ),
    "v1/worker/gpu_ubatch_wrapper.py": (
        "4eae50c929f3ba873072c13291c7140be3dd00d4a5b623170dff44754519c021"
    ),
    "v1/worker/ubatching.py": (
        "40391241c564feb5f16c77898ae6ae152ed6e71a4682e2a406387785d8de02d7"
    ),
    "model_executor/layers/fused_moe/modular_kernel.py": (
        "f78ae626babfd69f3c6ba37eef9c8f5186f28cd9064f566e341ca0c9e0fdb9b9"
    ),
    "model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py": (
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


def _check_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} is not a file: {path}")
    if _sha256(path) != expected_sha256:
        raise SystemExit(f"{label} SHA-256 does not match the frozen input")


def _ceiling_ps(byte_count: int, steps: int, rate_bps: int) -> int:
    """Round the exact serialization ceiling up to whole picoseconds."""

    exact = Fraction(byte_count * 8, rate_bps * steps) * 10**12
    return -((-exact.numerator) // exact.denominator)


def check_expectation_registry(args: argparse.Namespace) -> None:
    """Validate frozen inputs and literals without importing target code."""

    _check_file(args.capture, CAPTURE_SHA256, "capture")
    if _line_count(args.capture) != CAPTURE_ROWS:
        raise SystemExit("capture row count does not match the frozen input")
    _check_file(args.replay_run, REPLAY_RUN_SHA256, "replay run")
    _check_file(args.routed_experts, ROUTED_EXPERTS_SHA256, "routed experts")

    if not args.vllm_source.is_dir():
        raise SystemExit(f"vLLM source is not a directory: {args.vllm_source}")
    for relative, expected in SOURCE_HASHES.items():
        _check_file(args.vllm_source / relative, expected, f"vLLM source {relative}")

    if not args.vllm_python.is_file():
        raise SystemExit(f"vLLM Python is not a file: {args.vllm_python}")
    observed_version = subprocess.run(
        [str(args.vllm_python), "-c", "import vllm; print(vllm.__version__)"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_version != VLLM_VERSION:
        raise SystemExit(
            f"vLLM version must be {VLLM_VERSION}, got {observed_version!r}"
        )

    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")

    driver = (
        Path(__file__).resolve().parents[1]
        / "vllm_observed_schedule_v1"
        / "live_driver.py"
    )
    if not driver.is_file():
        raise SystemExit(f"reused live driver is missing: {driver}")

    assert len(EXPECTATION_BOUNDARY_COMMIT) == 40
    assert len(VLLM_AUTHORED_AGAINST_COMMIT) == 40
    assert (TP_SIZE, DP_SIZE, PP_SIZE) == (1, 8, 1)
    assert SEMANTIC_MOE_SITES == 2 * NUM_LAYERS
    assert DBO_MOE_INVOCATIONS == SEMANTIC_MOE_SITES * NUM_MICROBATCHES
    assert DBO_DECODE_THRESHOLD == 2
    assert DBO_PREFILL_THRESHOLD > 54

    # step partition: 23 DBO steps plus 9 single-batch steps
    assert len(R0_DECODE_STEP_INDICES) == 23
    assert len(SINGLE_BATCH_STEP_INDICES) == 9
    assert not set(R0_DECODE_STEP_INDICES) & set(SINGLE_BATCH_STEP_INDICES)
    assert (
        len(R0_DECODE_STEP_INDICES) + len(SINGLE_BATCH_STEP_INDICES)
        == NONEMPTY_STEPS
    )

    # control edges: 47 microbatch boundaries times eight ranks
    assert CONTROL_EDGES_PER_DBO_STEP == (2 * NUM_LAYERS - 1) * DP_SIZE
    assert CONTROL_EDGES_TOTAL == 8_648

    # the ceilings are arithmetic over the frozen routed table, not fitted
    for placement, rate in (
        ("single-node", NVLINK_RATE_BPS),
        ("cross-node", RNIC_RATE_BPS),
    ):
        expected_ceiling = _ceiling_ps(
            DBO_DECODE_EGRESS_BYTES,
            len(R0_DECODE_STEP_INDICES),
            rate,
        )
        if expected_ceiling != COMM_CEILING_PS[placement]:
            raise SystemExit(
                f"{placement} communication ceiling literal "
                f"{COMM_CEILING_PS[placement]} disagrees with the arithmetic "
                f"{expected_ceiling}"
            )
        low, high = OVERLAP_BAND_PS[placement]
        assert high == COMM_CEILING_PS[placement]
        assert low == -(
            (-(OVERLAP_BAND_LOW_FRACTION * high).numerator)
            // (OVERLAP_BAND_LOW_FRACTION * high).denominator
        )
        assert 0 < low < high
    assert (
        _ceiling_ps(PREFILL_EGRESS_BYTES, 1, RNIC_RATE_BPS) == 304_988_160
    )
    assert Fraction(NVLINK_RATE_BPS, RNIC_RATE_BPS) == 9
    assert (
        OVERLAP_RATE_RATIO_BAND[0] < 9 < OVERLAP_RATE_RATIO_BAND[1]
    )
    assert TERMINAL_RATE_RATIO_BAND[0] < 1 < TERMINAL_RATE_RATIO_BAND[1]
    assert OVERLAP_CEILING_ALLOWANCE > 1

    assert 0 < COMPUTE_FLOOR_PS < DECODE_TPOT_CEILING_PS
    assert DECODE_TPOT_CEILING_PS < PREFILL_TTFT_CEILING_PS
    assert EXPECTED_GENUINE_RISK_INSTANCES == 6
    assert SERIAL_GRAPH_BYTES > SERIAL_GOAL_BYTES > 0
    assert len(SERIAL_GRAPH_SHA256) == len(SERIAL_GOAL_SHA256) == 64


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--replay-run", type=Path, required=True)
    parser.add_argument("--routed-experts", type=Path, required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--vllm-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    check_expectation_registry(args)
    if args.check_only:
        print("check-only: frozen expectation registry passed; no artifacts written")
        return
    raise SystemExit(
        "the three-arm harness has not landed yet; only --check-only is available"
    )


if __name__ == "__main__":
    main()
