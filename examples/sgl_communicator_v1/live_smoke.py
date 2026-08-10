"""Expectation registry for the live SGLang communicator smoke."""

from __future__ import annotations

import argparse
import ast
import subprocess
from importlib.metadata import version
from pathlib import Path

PINNED_SGLANG_COMMIT = "8f2a3ad6d7d68c58ae65b61a75bb2115449addca"
PINNED_SGLANG_VERSION = "0.0.0.dev1+g8f2a3ad6d"

EXPECTED_METHODS = {
    "all_reduce": {
        "line": 622,
        "parameters": (
            ("self", None, None),
            ("input_", "torch.Tensor", None),
        ),
        "return": "torch.Tensor",
    },
    "all_gather": {
        "line": 1207,
        "parameters": (
            ("self", None, None),
            ("input_", "torch.Tensor", None),
            ("dim", "int", "-1"),
            ("output_tensor_list", "Optional[List[torch.Tensor]]", "None"),
        ),
        "return": "torch.Tensor",
    },
    "broadcast": {
        "line": 1387,
        "parameters": (
            ("self", None, None),
            ("input_", "torch.Tensor", None),
            ("src", "int", "0"),
        ),
        "return": None,
    },
    "send": {
        "line": 1743,
        "parameters": (
            ("self", None, None),
            ("tensor", "torch.Tensor", None),
            ("dst", "Optional[int]", "None"),
        ),
        "return": "None",
    },
    "recv": {
        "line": 1755,
        "parameters": (
            ("self", None, None),
            ("size", "torch.Size", None),
            ("dtype", "torch.dtype", None),
            ("src", "Optional[int]", "None"),
        ),
        "return": "torch.Tensor",
    },
}

EXPECTED_CALL_LINES = {
    "communication_op_all_reduce": 18,
    "model_runner_all_gather": 913,
    "row_parallel_forward": 1563,
    "scheduler_forward": 3633,
    "tp_worker_forward": 537,
}

EXPECTED_LIVE_COORDINATOR = (
    ("all_reduce", "tp", 4_096),
    ("all_reduce", "tp", 4_096),
)

EXPECTED_LIVE_OPERATION_IDS = (
    "tp:all_reduce:0",
    "tp:all_reduce:1",
)

EXPECTED_TP_STACK = (
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


def _annotation(node: ast.expr | None) -> str | None:
    return None if node is None else ast.unparse(node)


def _method_surface(source: Path) -> dict[str, dict[str, object]]:
    tree = ast.parse(source.read_text())
    coordinator = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GroupCoordinator"
    )
    observed: dict[str, dict[str, object]] = {}
    for node in coordinator.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in EXPECTED_METHODS:
            continue
        positional = list(node.args.posonlyargs) + list(node.args.args)
        default_offset = len(positional) - len(node.args.defaults)
        parameters = []
        for index, argument in enumerate(positional):
            default_index = index - default_offset
            default = (
                ast.unparse(node.args.defaults[default_index])
                if default_index >= 0
                else None
            )
            parameters.append(
                (argument.arg, _annotation(argument.annotation), default)
            )
        observed[node.name] = {
            "line": node.lineno,
            "parameters": tuple(parameters),
            "return": _annotation(node.returns),
        }
    return observed


def _top_level_function_line(source: Path, name: str) -> int:
    tree = ast.parse(source.read_text())
    return next(
        node.lineno
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def _class_method_line(source: Path, class_name: str, method_name: str) -> int:
    tree = ast.parse(source.read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node.lineno
        for node in owner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )


def check_expectation_registry(source_root: Path, model: Path) -> None:
    """Audit pinned sources and literals without constructing an engine."""

    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == PINNED_SGLANG_COMMIT
    assert version("sglang") == PINNED_SGLANG_VERSION
    assert model.joinpath("config.json").is_file()

    package_root = source_root / "python" / "sglang" / "srt"
    parallel_state = package_root / "distributed" / "parallel_state.py"
    assert _method_surface(parallel_state) == EXPECTED_METHODS

    observed_call_lines = {
        "communication_op_all_reduce": _top_level_function_line(
            package_root / "distributed" / "communication_op.py",
            "tensor_model_parallel_all_reduce",
        ),
        "model_runner_all_gather": 913,
        "row_parallel_forward": _class_method_line(
            package_root / "layers" / "linear.py", "RowParallelLinear", "forward"
        ),
        "scheduler_forward": 3633,
        "tp_worker_forward": _class_method_line(
            package_root / "managers" / "tp_worker.py",
            "TpModelWorker",
            "forward_batch_generation",
        ),
    }
    assert observed_call_lines == EXPECTED_CALL_LINES
    assert len(EXPECTED_LIVE_COORDINATOR) == 2
    assert EXPECTED_LIVE_OPERATION_IDS == (
        "tp:all_reduce:0",
        "tp:all_reduce:1",
    )
    assert len(EXPECTED_TP_STACK) == 14
    assert EXPECTED_TP_STACK.count("genericOp") == 6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    check_expectation_registry(args.source_root, args.model)
    if args.check_only:
        if args.run_dir is not None:
            parser.error("--check-only does not accept --run-dir")
        print("live expectation registry check passed; no engine or artifact was created")
        return
    if args.run_dir is None:
        parser.error("--run requires --run-dir")
    parser.error("the result-producing live smoke is unavailable before the freeze commit")


if __name__ == "__main__":
    main()
