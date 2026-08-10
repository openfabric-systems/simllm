"""Expectation registry for the live SGLang communicator smoke."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

FREEZE_COMMIT = "b0c5b731dccfdf86e9a07c3425c95c60f9980f39"
PINNED_SGLANG_COMMIT = "8f2a3ad6d7d68c58ae65b61a75bb2115449addca"
PINNED_SGLANG_VERSION = "0.0.0.dev1+g8f2a3ad6d"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

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


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _run_case(model: Path, case_dir: Path, *, enabled: bool) -> dict[str, object]:
    case_dir.mkdir(parents=True, exist_ok=True)
    steps_path = case_dir / "steps.jsonl"
    events_path = case_dir / "events.jsonl"
    evidence_path = case_dir / "case_evidence.json"
    for path in (steps_path, events_path, evidence_path):
        path.unlink(missing_ok=True)

    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "SIMLLM_SGLANG_ENABLE": "1",
        "SIMLLM_SGLANG_GPU": "b100",
        "SIMLLM_SGLANG_MODE": "virtual",
        "SIMLLM_SGLANG_STEP_RECORDS": str(steps_path),
        "SIMLLM_SGLANG_TOKEN_ID": "512",
        "SGLANG_PLUGINS": "simllm",
        "SGLANG_USE_CPU_ENGINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    if enabled:
        environment.update(
            {
                "SIMLLM_SGLANG_COMMUNICATOR_EVENTS": str(events_path),
                "SIMLLM_SGLANG_COMMUNICATOR_TP_SIZE": "4",
            }
        )
    else:
        os.environ.pop("SIMLLM_SGLANG_COMMUNICATOR_EVENTS", None)
        os.environ.pop("SIMLLM_SGLANG_COMMUNICATOR_TP_SIZE", None)

    previous = {name: os.environ.get(name) for name in environment}
    os.environ.update(environment)
    engine = None
    try:
        import sglang as sgl

        engine = sgl.Engine(
            model_path=str(model),
            device="cpu",
            disable_overlap_schedule=True,
            context_length=64,
            max_running_requests=4,
            max_total_tokens=64,
            random_seed=0,
            tp_size=1,
        )
        output = engine.generate(
            prompt="The simulated SGLang communicator",
            sampling_params={
                "temperature": 0,
                "max_new_tokens": 2,
                "ignore_eos": True,
            },
            rid="sgl11-live",
        )
    finally:
        if engine is not None:
            engine.shutdown()
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    output_ids = list(output["output_ids"])
    records = _read_jsonl(steps_path)
    events = _read_jsonl(events_path) if events_path.is_file() else []
    assert output_ids == [512, 512]
    assert len(records) == 2
    assert [record["step_index"] for record in records] == [0, 1]
    assert {record["schema"] for record in records} == {
        "atlahs-closed-loop-step-v1"
    }

    if enabled:
        projection = tuple(
            (event["operation"], event["group"], event["payload_bytes"])
            for event in events
        )
        assert projection == EXPECTED_LIVE_COORDINATOR
        assert tuple(event["operation_id"] for event in events) == (
            EXPECTED_LIVE_OPERATION_IDS
        )
        assert [event["sequence"] for event in events] == [0, 1]
        assert {event["schema"] for event in events} == {
            "simllm-vllm-group-coordinator-event-v1"
        }
        assert all(event["ranks"] == [0, 1, 2, 3] for event in events)
        assert all(event["stack_disposition"] == "entered" for event in events)
        for event, record in zip(events, records, strict=True):
            assert event["timestamp_ps"] == record["virtual_time_ps"]
            nested = event["stack_events"]
            assert tuple(item["function"] for item in nested) == EXPECTED_TP_STACK
            assert all(
                item["timestamp_ps"] == event["timestamp_ps"] for item in nested
            )
            assert event["work"] == {
                "algorithm_hint": "ring",
                "channel_hint": None,
                "collective": "all-reduce",
                "payload_bytes": 4_096,
                "ranks": [0, 1, 2, 3],
            }
        live_relation = "PASS"
    else:
        assert not events_path.exists()
        projection = ()
        live_relation = "NOT_APPLICABLE"

    evidence = {
        "case": "enabled" if enabled else "baseline",
        "communicator_enabled": enabled,
        "output_ids": output_ids,
        "step_count": len(records),
        "step_sha256": hashlib.sha256(steps_path.read_bytes()).hexdigest(),
        "coordinator_projection": [list(item) for item in projection],
        "stack_event_counts": [len(event["stack_events"]) for event in events],
        "live_relation": live_relation,
    }
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return evidence


def _run_child_case(
    *,
    case: str,
    source_root: Path,
    model: Path,
    run_dir: Path,
) -> None:
    enabled = case == "enabled"
    evidence = _run_case(model, run_dir / case, enabled=enabled)
    print(f"CASE={case}")
    print(f"CASE_STEP_COUNT={evidence['step_count']}")
    print(f"CASE_COORDINATOR_EVENTS={len(evidence['coordinator_projection'])}")


def run_live_smoke(
    *,
    source_root: Path,
    model: Path,
    run_dir: Path,
) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    repository_path = str(REPOSITORY_ROOT)
    child_environment = os.environ.copy()
    existing_pythonpath = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = (
        repository_path
        if not existing_pythonpath
        else repository_path + os.pathsep + existing_pythonpath
    )
    for case in ("baseline", "enabled"):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run",
            "--case",
            case,
            "--source-root",
            str(source_root),
            "--model",
            str(model),
            "--run-dir",
            str(run_dir),
        ]
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=child_environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"{case} SGLang child failed with code {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        print(completed.stdout, end="")

    baseline_path = run_dir / "baseline" / "case_evidence.json"
    enabled_path = run_dir / "enabled" / "case_evidence.json"
    baseline = json.loads(baseline_path.read_text())
    enabled = json.loads(enabled_path.read_text())
    baseline_steps = run_dir / "baseline" / "steps.jsonl"
    enabled_steps = run_dir / "enabled" / "steps.jsonl"
    byte_identical = baseline_steps.read_bytes() == enabled_steps.read_bytes()
    assert byte_identical
    assert baseline["output_ids"] == enabled["output_ids"] == [512, 512]
    assert baseline["coordinator_projection"] == []
    assert tuple(tuple(item) for item in enabled["coordinator_projection"]) == (
        EXPECTED_LIVE_COORDINATOR
    )
    assert enabled["stack_event_counts"] == [14, 14]
    evidence = {
        "freeze_commit": FREEZE_COMMIT,
        "sglang_commit": PINNED_SGLANG_COMMIT,
        "sglang_version": PINNED_SGLANG_VERSION,
        "baseline": baseline,
        "enabled": enabled,
        "step_records_byte_identical": byte_identical,
        "live_relation": "PASS",
    }
    evidence_path = run_dir / "live_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--case", choices=("baseline", "enabled"), help=argparse.SUPPRESS)
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
    if args.case is not None:
        _run_child_case(
            case=args.case,
            source_root=args.source_root,
            model=args.model,
            run_dir=args.run_dir,
        )
        return
    evidence = run_live_smoke(
        source_root=args.source_root,
        model=args.model,
        run_dir=args.run_dir,
    )
    print("SMOKE_SIMWORKER_REACHED=True")
    print(
        "SMOKE_COORDINATOR_EVENTS="
        f"{len(evidence['enabled']['coordinator_projection'])}"
    )
    print(
        "SMOKE_STACK_EVENT_COUNTS="
        + ",".join(
            str(value) for value in evidence["enabled"]["stack_event_counts"]
        )
    )
    print("SMOKE_FLAG_OFF_BYTE_IDENTITY=PASS")
    print("SMOKE_LIVE_RELATION=PASS")


if __name__ == "__main__":
    main()
