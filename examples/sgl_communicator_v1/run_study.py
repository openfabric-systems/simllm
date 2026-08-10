"""Run the import-free simulated SGLang communicator study."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from simllm._local_config import path_from_env
from simllm.adapters.sglang import FLOAT32, ShapeTensor, SimGroupCoordinator
from simllm.adapters.vllm import SimGroupCoordinator as VllmSimGroupCoordinator
from simllm.compute import NcclStackConfig
from simllm.core import CollectiveWork, VirtualClock

FREEZE_COMMIT = "b0c5b731dccfdf86e9a07c3425c95c60f9980f39"
CLOCK_START_PS = 123_000
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

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


def _make_group(
    group_size: int,
    *,
    chunk_bytes: int = 4,
) -> SimGroupCoordinator:
    return SimGroupCoordinator(
        group_name="tp",
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
    parts = [input_.new_empty(input_.shape) for _ in range(group_size)]
    reduced = group.all_reduce(input_)
    gathered = group.all_gather(input_, dim=1)
    gathered_into = group.all_gather(
        input_, dim=1, output_tensor_list=parts
    )
    broadcast = group.broadcast(input_, src=0)
    sent = group.send(input_, dst=1)
    received = group.recv((4, extent), FLOAT32, src=1)
    expected_payload = 16 * extent
    expected_operations = (
        "all_reduce",
        "all_gather",
        "all_gather",
        "broadcast",
        "send",
        "recv",
    )
    shape_pass = (
        reduced.shape == (4, extent)
        and reduced.dtype is FLOAT32
        and gathered.shape == (4, group_size * extent)
        and gathered.dtype is FLOAT32
        and gathered_into is None
        and len(parts) == group_size
        and all(part.shape == input_.shape and part.dtype is FLOAT32 for part in parts)
        and broadcast is input_
        and sent is None
        and received.shape == (4, extent)
        and received.dtype is FLOAT32
    )
    structural_pass = (
        tuple(event.operation for event in group.events) == expected_operations
        and tuple(event.sequence for event in group.events) == tuple(range(6))
        and {event.schema for event in group.events}
        == {"simllm-vllm-group-coordinator-event-v1"}
        and all(event.payload_bytes == expected_payload for event in group.events)
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
        "output_list_length": len(parts),
        "output_list_shapes": [part.shape for part in parts],
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
        (1, 1_024),
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
    parts = [input_.new_empty(input_.shape)]
    passed = (
        group.all_reduce(input_) is input_
        and group.all_gather(input_, dim=1) is input_
        and group.all_gather(input_, output_tensor_list=parts) is None
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


def _vllm_parity_guard() -> dict[str, object]:
    base_path = REPOSITORY_ROOT / "simllm" / "adapters" / "vllm" / "communicator.py"
    source_hash = hashlib.sha256(base_path.read_bytes()).hexdigest()
    kwargs = {
        "group_name": "tp",
        "ranks": (0, 1, 2, 3),
        "rank": 0,
        "local_rank": 0,
        "stack_config": NcclStackConfig(
            channel_count=1,
            chunk_bytes=1_024,
            fifo_slots_per_channel=2,
        ),
    }
    vllm = VllmSimGroupCoordinator(
        clock=VirtualClock(start_ps=CLOCK_START_PS), **kwargs
    )
    sglang = SimGroupCoordinator(
        clock=VirtualClock(start_ps=CLOCK_START_PS), **kwargs
    )
    input_ = ShapeTensor((1, 1_024), dtype=FLOAT32, element_size_bytes=4)
    vllm_output = vllm.all_reduce(input_)
    sglang_output = sglang.all_reduce(input_)
    passed = (
        source_hash == EXPECTED_VLLM_BASE_SHA256
        and vllm_output == sglang_output
        and vllm.events == sglang.events
        and vllm.stack_events == sglang.stack_events
    )
    return {
        "expected_source_sha256": EXPECTED_VLLM_BASE_SHA256,
        "observed_source_sha256": source_hash,
        "event_count": len(sglang.events),
        "stack_event_count": len(sglang.stack_events),
        "vllm_parity_guard": "PASS" if passed else "FAIL",
    }


def run_study() -> dict[str, object]:
    if "sglang" in sys.modules:
        raise RuntimeError("the component study must not import SGLang")
    shape_rows = [
        _shape_cell(group_size, extent)
        for group_size in (2, 4)
        for extent in (8, 16)
    ]
    evidence = {
        "freeze_commit": FREEZE_COMMIT,
        "shape_rows": shape_rows,
        "payload_relations": _payload_relations(shape_rows),
        "reference_stack": _reference_stack_guard(),
        "singleton": _singleton_guard(),
        "vllm_parity": _vllm_parity_guard(),
    }
    if "sglang" in sys.modules:
        raise RuntimeError("the component study imported SGLang")
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
        if args.run_dir is not None:
            parser.error("--check-only does not accept --run-dir")
        print("expectation registry check passed; no study artifact was produced")
        return
    if args.run_dir is None:
        data_root = path_from_env("SIMLLM_DATA_ROOT")
        if data_root is None:
            parser.error("--run-dir is required when SIMLLM_DATA_ROOT is not set")
        args.run_dir = data_root / "sgl_communicator_v1"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    evidence = run_study()
    output = args.run_dir / "component_results.json"
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")

    shape_rows = evidence["shape_rows"]
    payload_rows = evidence["payload_relations"]
    shape_passes = sum(row["shape_relation"] == "PASS" for row in shape_rows)
    payload_passes = sum(
        row["payload_scaling"] == "PASS" for row in payload_rows
    )
    structural = (
        all(row["structural_guard"] == "PASS" for row in shape_rows)
        and evidence["reference_stack"]["literal_stack_guard"] == "PASS"
        and evidence["singleton"]["singleton_guard"] == "PASS"
        and evidence["vllm_parity"]["vllm_parity_guard"] == "PASS"
    )
    print(f"shape relation instances: {shape_passes}/{len(shape_rows)} PASS")
    print(f"payload scaling instances: {payload_passes}/{len(payload_rows)} PASS")
    print(f"fatal structural guards: {'PASS' if structural else 'FAIL'}")
    print(f"evidence: {output}")
    if (
        shape_passes != len(shape_rows)
        or payload_passes != len(payload_rows)
        or not structural
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
