"""Expectation registry for the simulated SGLang communicator study."""

from __future__ import annotations

import argparse
from pathlib import Path

EXPECTED_REFERENCE_COORDINATOR = (
    ("all_reduce", "tp", 4_096),
)

EXPECTED_REFERENCE_STACK = (
    "ncclAllReduce",
    "ncclEnqueueCheck",
    "scheduleCollTasksToPlan",
    "calcCollChunking",
    "ncclLaunchKernel",
    "ncclKernelMain",
    "runRing",
    "genericOp",
    "genericOp",
    "genericOp",
    "genericOp",
    "genericOp",
    "genericOp",
    "simllmKernelComplete",
)

EXPECTED_FULL_STACK_PREFIX = (
    "ncclCommInitRank",
    "ncclBuildRings",
    "initChannel",
)

EXPECTED_VLLM_BASE_SHA256 = (
    "9b7b4bf6e49d6b35979ef8532873a35b4321453ecb78e9d58aa5b97adf85475e"
)


def check_expectation_registry() -> None:
    """Validate frozen literals without executing target behavior."""

    assert EXPECTED_REFERENCE_COORDINATOR == (("all_reduce", "tp", 4_096),)
    assert len(EXPECTED_REFERENCE_STACK) == 14
    assert EXPECTED_REFERENCE_STACK.count("genericOp") == 6
    assert len(EXPECTED_FULL_STACK_PREFIX + EXPECTED_REFERENCE_STACK) == 17
    assert len(EXPECTED_VLLM_BASE_SHA256) == 64


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check_expectation_registry()
    if args.check_only:
        if args.run_dir is not None:
            parser.error("--check-only does not accept --run-dir")
        print("expectation registry check passed; no study artifact was produced")
        return
    if args.run_dir is None:
        parser.error("--check requires --run-dir")
    parser.error("the result-producing study is unavailable before the freeze commit")


if __name__ == "__main__":
    main()
