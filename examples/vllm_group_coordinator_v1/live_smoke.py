"""Expectation registry for the live simulated-coordinator smoke."""

from __future__ import annotations

import argparse
import ast
from importlib.metadata import version
from pathlib import Path

VLLM_SOURCE = Path(
    "/data3/yifeng/simllm-dev/venv-vllm/lib64/python3.12/"
    "site-packages/vllm/distributed/parallel_state.py"
)
MODEL = Path(
    "/home/yifeng/packages/vllm-rnic-capture/hf-cache/hub/"
    "models--ibm-granite--granite-3.0-1b-a400m-instruct/snapshots/"
    "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
)

EXPECTED_SIGNATURE_LINES = {
    "all_reduce": 641,
    "all_gather": 670,
    "broadcast": 745,
    "send": 1188,
    "recv": 1195,
}

EXPECTED_LIVE_COORDINATOR = (
    ("all_reduce", "dp", 64),
    ("all_reduce", "tp", 4_096),
    ("all_reduce", "dp", 64),
    ("all_reduce", "tp", 4_096),
)

EXPECTED_TP_STACK = (
    "ncclAllReduce",
    "ncclEnqueueCheck",
    "scheduleCollTasksToPlan",
    "calcCollChunking",
    "ncclLaunchKernel",
    "ncclKernelMain",
    "runRing",
    "genericOp",  # 1
    "genericOp",  # 2
    "genericOp",  # 3
    "genericOp",  # 4
    "genericOp",  # 5
    "genericOp",  # 6
    "simllmKernelComplete",
)

EXPECTED_DP_STACK = (
    "ncclAllReduce",
    "ncclEnqueueCheck",
    "scheduleCollTasksToPlan",
    "calcCollChunking",
    "ncclLaunchKernel",
    "ncclKernelMain",
    "runRing",
    "genericOp",  # 1
    "genericOp",  # 2
    "genericOp",  # 3
    "genericOp",  # 4
    "genericOp",  # 5
    "genericOp",  # 6
    "genericOp",  # 7
    "genericOp",  # 8
    "genericOp",  # 9
    "genericOp",  # 10
    "genericOp",  # 11
    "genericOp",  # 12
    "genericOp",  # 13
    "genericOp",  # 14
    "genericOp",  # 15
    "genericOp",  # 16
    "genericOp",  # 17
    "genericOp",  # 18
    "genericOp",  # 19
    "genericOp",  # 20
    "genericOp",  # 21
    "genericOp",  # 22
    "genericOp",  # 23
    "genericOp",  # 24
    "simllmKernelComplete",
)


def _audited_method_lines() -> dict[str, int]:
    tree = ast.parse(VLLM_SOURCE.read_text())
    coordinator = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GroupCoordinator"
    )
    return {
        node.name: node.lineno
        for node in coordinator.body
        if isinstance(node, ast.FunctionDef) and node.name in EXPECTED_SIGNATURE_LINES
    }


def check_expectation_registry() -> None:
    """Audit sources and literals without constructing a vLLM engine."""

    assert version("vllm") == "0.26.0"
    assert VLLM_SOURCE.is_file()
    assert MODEL.joinpath("config.json").is_file()
    assert _audited_method_lines() == EXPECTED_SIGNATURE_LINES
    assert len(EXPECTED_LIVE_COORDINATOR) == 4
    assert len(EXPECTED_TP_STACK) == 14
    assert len(EXPECTED_DP_STACK) == 32


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    check_expectation_registry()
    if args.check_only:
        print("live expectation registry check passed; no engine was constructed")
        return
    parser.error("the result-producing live smoke is unavailable before the freeze commit")


if __name__ == "__main__":
    main()
