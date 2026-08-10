"""Expectation registry for the live simulated-coordinator smoke."""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import traceback
from importlib.metadata import version
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from simllm._local_config import path_from_env

MODEL_CACHE_PATH = (
    Path("hub")
    / "models--ibm-granite--granite-3.0-1b-a400m-instruct"
    / "snapshots"
    / "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
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

POSTSPEC_EXPECTED_DP_PADDED_TOKENS = (4, 1)

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


def _required_env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set to an existing directory")
    path = Path(value).expanduser()
    if not path.is_dir():
        raise RuntimeError(f"{name} must name an existing directory: {path}")
    return path


def _model_path() -> Path:
    return _required_env_path("HF_HOME") / MODEL_CACHE_PATH


def _vllm_source_path() -> Path:
    configured = _required_env_path("SIMLLM_VLLM_PACKAGE_ROOT")
    relative = Path("distributed") / "parallel_state.py"
    candidate = configured / relative
    if candidate.is_file():
        return candidate
    raise RuntimeError(
        "SIMLLM_VLLM_PACKAGE_ROOT must name the installed vllm package directory"
    )


def _audited_method_lines() -> dict[str, int]:
    tree = ast.parse(_vllm_source_path().read_text())
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

    vllm_source = _vllm_source_path()
    model = _model_path()
    assert version("vllm") == "0.26.0"
    assert vllm_source.is_file()
    assert model.joinpath("config.json").is_file()
    assert _audited_method_lines() == EXPECTED_SIGNATURE_LINES
    assert len(EXPECTED_LIVE_COORDINATOR) == 4
    assert len(EXPECTED_TP_STACK) == 14
    assert len(EXPECTED_DP_STACK) == 32
    assert POSTSPEC_EXPECTED_DP_PADDED_TOKENS == (4, 1)


def worker_reached() -> bool:
    from simllm.adapters.vllm import latest_worker

    return latest_worker() is not None


def run_live_smoke(run_dir: Path) -> dict[str, object]:
    repository_path = str(REPOSITORY_ROOT)
    if sys.path[0] != repository_path:
        sys.path.insert(0, repository_path)
    from vllm import LLM, SamplingParams

    from simllm.adapters.vllm import (
        GroupCoordinatorObserver,
        SimGroupCoordinator,
        SimModelRunner,
        latest_worker,
    )
    from simllm.compute import NcclStackConfig

    run_dir.mkdir(parents=True, exist_ok=True)
    stream_path = run_dir / "live_steps.jsonl"
    evidence_path = run_dir / "live_evidence.json"
    stream_path.unlink(missing_ok=True)
    evidence_path.unlink(missing_ok=True)

    environment = {
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "SIMLLM_VLLM_WORKER_MODE": "skeleton",
        "SIMLLM_VLLM_MODE": "virtual",
        "SIMLLM_VLLM_STEP_RECORDS": str(stream_path),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(_required_env_path("HF_HOME")),
        "CUDA_VISIBLE_DEVICES": "",
    }
    previous = {name: os.environ.get(name) for name in environment}
    os.environ.update(environment)
    try:
        llm = LLM(
            model=str(_model_path()),
            worker_cls="simllm.adapters.vllm.SimWorker",
            enforce_eager=True,
            max_model_len=64,
            num_gpu_blocks_override=64,
            disable_log_stats=True,
        )
        worker = latest_worker()
        assert worker is not None, "SimWorker was not constructed"
        runner = worker.model_runner
        assert isinstance(runner, SimModelRunner), f"unexpected runner {type(runner)!r}"

        observer = GroupCoordinatorObserver(worker.clock)
        dp_group = SimGroupCoordinator(
            group_name="dp",
            ranks=(0, 1, 2, 3),
            rank=0,
            local_rank=0,
            clock=worker.clock,
            observer=observer,
            stack_config=NcclStackConfig(
                channel_count=1,
                chunk_bytes=4,
                fifo_slots_per_channel=2,
            ),
        )
        tp_group = SimGroupCoordinator(
            group_name="tp",
            ranks=(0, 1, 2, 3),
            rank=0,
            local_rank=0,
            clock=worker.clock,
            observer=observer,
            stack_config=NcclStackConfig(
                channel_count=1,
                chunk_bytes=1_024,
                fifo_slots_per_channel=2,
            ),
        )
        runner.bind_simulated_groups(tp_group=tp_group, dp_group=dp_group)

        outputs = llm.generate(
            ["The simulated coordinator"],
            SamplingParams(max_tokens=2, ignore_eos=True),
        )
        events = runner.coordinator_events
        projection = tuple(
            (event.operation, event.group, event.payload_bytes) for event in events
        )
        assert projection == EXPECTED_LIVE_COORDINATOR, (
            f"coordinator projection {projection!r} does not match the frozen oracle"
        )
        assert {event.schema for event in events} == {
            "simllm-vllm-group-coordinator-event-v1"
        }
        for event in events:
            expected_stack = EXPECTED_DP_STACK if event.group == "dp" else EXPECTED_TP_STACK
            measured_stack = tuple(item.function for item in event.stack_events)
            assert measured_stack == expected_stack, (
                f"{event.group} stack projection differs at event {event.sequence}"
            )
            assert event.timestamp_ps == worker.clock.now_ps
            assert all(item.timestamp_ps == worker.clock.now_ps for item in event.stack_events)

        assert len(outputs) == 1
        assert len(outputs[0].outputs) == 1
        sampled_token_ids = tuple(outputs[0].outputs[0].token_ids)
        assert sampled_token_ids == (worker.token_id, worker.token_id)
        records = [
            json.loads(line)
            for line in stream_path.read_text().splitlines()
            if line.strip()
        ]
        assert len(records) == 2
        assert {record.get("schema") for record in records} == {
            "atlahs-closed-loop-step-v1"
        }
        postspecified_dp_padding = tuple(
            record.get("num_tokens_after_padding") for record in records
        )
        assert postspecified_dp_padding == POSTSPEC_EXPECTED_DP_PADDED_TOKENS, (
            "post-specified DP padding regression differs: "
            f"{postspecified_dp_padding!r}"
        )

        evidence = {
            "freeze_commit": "29221e4",
            "vllm_version": version("vllm"),
            "worker_class": type(worker).__name__,
            "runner_class": type(runner).__name__,
            "coordinator_schema": events[0].schema,
            "coordinator_projection": [list(item) for item in projection],
            "stack_event_counts": [len(event.stack_events) for event in events],
            "sampled_token_ids": list(sampled_token_ids),
            "step_record_count": len(records),
            "step_schemas": sorted({record["schema"] for record in records}),
            "postspecified_dp_padding": {
                "expected": list(POSTSPEC_EXPECTED_DP_PADDED_TOKENS),
                "observed": list(postspecified_dp_padding),
                "regression": "PASS",
            },
            "final_clock_ps": worker.clock.now_ps,
            "live_relation": "PASS",
        }
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        return evidence
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    check_expectation_registry()
    if args.check_only:
        print("live expectation registry check passed; no engine was constructed")
        return
    if args.run_dir is None:
        data_root = path_from_env("SIMLLM_DATA_ROOT")
        if data_root is None:
            parser.error("--run-dir is required when SIMLLM_DATA_ROOT is not set")
        args.run_dir = data_root / "vllm_group_coordinator_v1" / "live"
    print(f"SMOKE_MODEL={_model_path()}")
    print("SMOKE_WORKER_CLS=simllm.adapters.vllm.SimWorker")
    try:
        evidence = run_live_smoke(args.run_dir)
    except BaseException:
        print(f"SMOKE_SIMWORKER_REACHED={worker_reached()}")
        traceback.print_exc()
        raise
    print(f"SMOKE_SIMWORKER_REACHED={worker_reached()}")
    print(f"SMOKE_COORDINATOR_EVENTS={len(evidence['coordinator_projection'])}")
    print(
        "SMOKE_STACK_EVENT_COUNTS="
        + ",".join(str(value) for value in evidence["stack_event_counts"])
    )
    print(f"SMOKE_STEP_RECORD_COUNT={evidence['step_record_count']}")
    print("SMOKE_POSTSPEC_DP_PADDING=PASS")
    print("SMOKE_LIVE_RELATION=PASS")


if __name__ == "__main__":
    main()
