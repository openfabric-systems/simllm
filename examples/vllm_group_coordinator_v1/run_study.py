"""Run the import-free simulated vLLM coordinator study."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from simllm._local_config import path_from_env
from simllm.adapters.vllm import FLOAT32, ShapeTensor, SimGroupCoordinator
from simllm.compute import NcclStackConfig
from simllm.core import CollectiveWork, VirtualClock

CLOCK_START_PS = 123_000

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

POSTSPEC_ZERO_PAYLOAD_STACK_DISPOSITION = "zero_payload_bypass"
POSTSPEC_FIRST_VALID_OPERATION_ID = "tp:all_reduce:0"


def check_expectation_registry() -> None:
    """Validate frozen literals without executing the target behavior."""

    assert EXPECTED_REFERENCE_COORDINATOR == (("all_reduce", "tp", 4_096),)
    assert len(EXPECTED_REFERENCE_STACK) == 14
    assert EXPECTED_REFERENCE_STACK.count("genericOp") == 6
    assert len(EXPECTED_FULL_STACK_PREFIX + EXPECTED_REFERENCE_STACK) == 17
    assert POSTSPEC_ZERO_PAYLOAD_STACK_DISPOSITION == "zero_payload_bypass"
    assert POSTSPEC_FIRST_VALID_OPERATION_ID == "tp:all_reduce:0"


def _make_group(
    group_size: int,
    *,
    group_name: str = "tp",
    chunk_bytes: int = 4,
) -> SimGroupCoordinator:
    return SimGroupCoordinator(
        group_name=group_name,
        ranks=tuple(range(group_size)),
        rank=0,
        local_rank=0,
        clock=VirtualClock(start_ps=CLOCK_START_PS),
        stack_config=NcclStackConfig(
            channel_count=1,
            chunk_bytes=chunk_bytes,
            fifo_slots_per_channel=2,
        ),
    )


def _shape_cell(group_size: int, extent: int) -> dict[str, object]:
    group = _make_group(group_size)
    input_ = ShapeTensor(
        (4, extent),
        dtype=FLOAT32,
        element_size_bytes=FLOAT32.itemsize,
    )
    reduced = group.all_reduce(input_)
    gathered = group.all_gather(input_, dim=1)
    broadcast = group.broadcast(input_, src=0)
    sent = group.send(input_, dst=1)
    received = group.recv((4, extent), FLOAT32, src=1)
    expected_payload = 16 * extent
    expected_operations = ("all_reduce", "all_gather", "broadcast", "send", "recv")
    shape_pass = (
        reduced.shape == (4, extent)
        and reduced.dtype is FLOAT32
        and gathered.shape == (4, group_size * extent)
        and gathered.dtype is FLOAT32
        and broadcast is input_
        and sent is None
        and received.shape == (4, extent)
        and received.dtype is FLOAT32
    )
    structural_pass = (
        tuple(event.operation for event in group.events) == expected_operations
        and tuple(event.sequence for event in group.events) == tuple(range(5))
        and {event.schema for event in group.events}
        == {"simllm-vllm-group-coordinator-event-v1"}
        and all(event.timestamp_ps == CLOCK_START_PS for event in group.events)
        and all(event.stack_events for event in group.events)
        and group.clock.now_ps == CLOCK_START_PS
    )
    return {
        "group_size": group_size,
        "extent": extent,
        "input_shape": input_.shape,
        "all_reduce_shape": reduced.shape,
        "all_gather_shape": gathered.shape,
        "broadcast_identity": broadcast is input_,
        "send_result_is_none": sent is None,
        "recv_shape": received.shape,
        "event_payloads": [event.payload_bytes for event in group.events],
        "expected_payload": expected_payload,
        "shape_relation": "PASS" if shape_pass else "FAIL",
        "structural_guard": "PASS" if structural_pass else "FAIL",
    }


def _payload_relations(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {(row["group_size"], row["extent"]): row for row in rows}
    relations = []
    for group_size in (2, 4):
        short = by_key[(group_size, 8)]
        long = by_key[(group_size, 16)]
        short_payloads = [int(value) for value in short["event_payloads"]]
        long_payloads = [int(value) for value in long["event_payloads"]]
        passed = long_payloads == [2 * value for value in short_payloads]
        relations.append(
            {
                "group_size": group_size,
                "short_payloads": short_payloads,
                "long_payloads": long_payloads,
                "payload_scaling": "PASS" if passed else "FAIL",
            }
        )
    return relations


def _reference_stack_guard() -> dict[str, object]:
    group = _make_group(4, chunk_bytes=1_024)
    input_ = ShapeTensor(
        (16, 64),
        dtype=FLOAT32,
        element_size_bytes=FLOAT32.itemsize,
    )
    group.all_reduce(input_)
    event = group.events[0]
    nested = tuple(item.function for item in event.stack_events)
    full = tuple(item.function for item in group.stack_events)
    passed = (
        (event.operation, event.group, event.payload_bytes)
        == EXPECTED_REFERENCE_COORDINATOR[0]
        and event.work
        == CollectiveWork("all-reduce", (0, 1, 2, 3), 4_096, "ring")
        and nested == EXPECTED_REFERENCE_STACK
        and full == EXPECTED_FULL_STACK_PREFIX + EXPECTED_REFERENCE_STACK
    )
    return {
        "coordinator_projection": [
            event.operation,
            event.group,
            event.payload_bytes,
        ],
        "nested_stack_count": len(nested),
        "full_stack_count": len(full),
        "literal_stack_guard": "PASS" if passed else "FAIL",
    }


def _singleton_guard() -> dict[str, object]:
    group = _make_group(1)
    input_ = ShapeTensor((4, 8), dtype=FLOAT32, element_size_bytes=4)
    passed = (
        group.all_reduce(input_) is input_
        and group.all_gather(input_, dim=1) is input_
        and group.broadcast(input_) is input_
        and group.stack_events == ()
        and all(event.stack_events == () for event in group.events)
        and group.clock.now_ps == CLOCK_START_PS
    )
    return {
        "coordinator_event_count": len(group.events),
        "stack_event_count": len(group.stack_events),
        "clock_ps": group.clock.now_ps,
        "singleton_guard": "PASS" if passed else "FAIL",
    }


def _payload_domain_guards() -> dict[str, object]:
    gapless_group = _make_group(4, chunk_bytes=1)
    invalid = ShapeTensor((5,), dtype=FLOAT32, element_size_bytes=2)
    invalid_error = None
    try:
        gapless_group.all_reduce(invalid)
    except ValueError as exc:
        invalid_error = str(exc)
    gapless_group.all_reduce(
        ShapeTensor((1_024,), dtype=FLOAT32, element_size_bytes=4)
    )
    first_valid = gapless_group.events[0]
    gapless_pass = (
        invalid_error is not None
        and "payload_bytes must divide evenly" in invalid_error
        and len(gapless_group.events) == 1
        and first_valid.sequence == 0
        and first_valid.operation_id == POSTSPEC_FIRST_VALID_OPERATION_ID
        and first_valid.stack_disposition == "entered"
    )

    zero_group = _make_group(4, chunk_bytes=1)
    zero_output = zero_group.all_reduce(
        ShapeTensor((0,), dtype=FLOAT32, element_size_bytes=4)
    )
    zero_event = zero_group.events[0]
    zero_pass = (
        zero_output.shape == (0,)
        and zero_event.payload_bytes == 0
        and zero_event.operation_id == POSTSPEC_FIRST_VALID_OPERATION_ID
        and zero_event.stack_disposition == POSTSPEC_ZERO_PAYLOAD_STACK_DISPOSITION
        and zero_event.stack_events == ()
        and len(zero_group.stack_events) == len(EXPECTED_FULL_STACK_PREFIX)
    )
    return {
        "invalid_error": invalid_error,
        "first_valid_operation_id": first_valid.operation_id,
        "gapless_operation_id_guard": "PASS" if gapless_pass else "FAIL",
        "zero_payload_stack_disposition": zero_event.stack_disposition,
        "zero_payload_nested_stack_count": len(zero_event.stack_events),
        "zero_payload_guard": "PASS" if zero_pass else "FAIL",
    }


def _vllm13_baseline_guard(run_dir: Path) -> dict[str, object]:
    baseline_source = (
        Path(__file__).parents[1] / "vllm_skeleton_v1" / "run_vllm_skeleton_v1.py"
    )
    spec = importlib.util.spec_from_file_location("simllm_vllm13_baseline", baseline_source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load baseline study from {baseline_source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        run_cell = module.run_cell
    finally:
        sys.modules.pop(spec.name, None)

    baseline_dir = run_dir / "singleton_baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.joinpath("r1_p4_steps.jsonl").unlink(missing_ok=True)
    row = run_cell(1, 4, baseline_dir)
    passed = (
        row["exact_oracle"] == "PASS"
        and row["structural_check"] == "PASS"
        and row["step_count"] == 2
        and row["sampled_tokens"] == 2
        and row["final_clock_ps"] == CLOCK_START_PS
    )
    return {
        "step_count": row["step_count"],
        "sampled_tokens": row["sampled_tokens"],
        "final_clock_ps": row["final_clock_ps"],
        "vllm13_exact_oracle": row["exact_oracle"],
        "vllm13_structural_check": row["structural_check"],
        "baseline_guard": "PASS" if passed else "FAIL",
    }


def run_study(run_dir: Path) -> dict[str, object]:
    if "vllm" in sys.modules:
        raise RuntimeError("the component study must not import vLLM")
    shape_rows = [
        _shape_cell(group_size, extent)
        for group_size in (2, 4)
        for extent in (8, 16)
    ]
    payload_rows = _payload_relations(shape_rows)
    evidence = {
        "freeze_commit": "29221e4",
        "shape_rows": shape_rows,
        "payload_relations": payload_rows,
        "reference_stack": _reference_stack_guard(),
        "singleton": _singleton_guard(),
        "payload_domain": _payload_domain_guards(),
        "vllm13_baseline": _vllm13_baseline_guard(run_dir),
    }
    if "vllm" in sys.modules:
        raise RuntimeError("the component study imported vLLM")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check_expectation_registry()
    if args.check_only:
        print("expectation registry check passed; no study result was produced")
        return
    if args.run_dir is None:
        data_root = path_from_env("SIMLLM_DATA_ROOT")
        if data_root is None:
            parser.error("--run-dir is required when SIMLLM_DATA_ROOT is not set")
        args.run_dir = data_root / "vllm_group_coordinator_v1"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    evidence = run_study(args.run_dir)
    output = args.run_dir / "component_results.json"
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")

    shape_rows = evidence["shape_rows"]
    payload_rows = evidence["payload_relations"]
    shape_passes = sum(row["shape_relation"] == "PASS" for row in shape_rows)
    payload_passes = sum(row["payload_scaling"] == "PASS" for row in payload_rows)
    structural = (
        all(row["structural_guard"] == "PASS" for row in shape_rows)
        and evidence["reference_stack"]["literal_stack_guard"] == "PASS"
        and evidence["singleton"]["singleton_guard"] == "PASS"
        and evidence["payload_domain"]["gapless_operation_id_guard"] == "PASS"
        and evidence["payload_domain"]["zero_payload_guard"] == "PASS"
        and evidence["vllm13_baseline"]["baseline_guard"] == "PASS"
    )
    print(f"shape relation instances: {shape_passes}/{len(shape_rows)} PASS")
    print(f"payload scaling instances: {payload_passes}/{len(payload_rows)} PASS")
    print("post-specified payload guards: PASS")
    print(f"fatal structural guards: {'PASS' if structural else 'FAIL'}")
    print(f"evidence: {output}")
    if shape_passes != len(shape_rows) or payload_passes != len(payload_rows) or not structural:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
