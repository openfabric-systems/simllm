"""Expectation registry for the simulated vLLM coordinator study."""

from __future__ import annotations

import argparse

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


def check_expectation_registry() -> None:
    """Validate frozen literals without executing the target behavior."""

    assert EXPECTED_REFERENCE_COORDINATOR == (("all_reduce", "tp", 4_096),)
    assert len(EXPECTED_REFERENCE_STACK) == 14
    assert EXPECTED_REFERENCE_STACK.count("genericOp") == 6
    assert len(EXPECTED_FULL_STACK_PREFIX + EXPECTED_REFERENCE_STACK) == 17


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check_expectation_registry()
    if args.check_only:
        print("expectation registry check passed; no study result was produced")
        return
    parser.error("the result-producing study is unavailable before the freeze commit")


if __name__ == "__main__":
    main()
