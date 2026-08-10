"""Run the frozen structural study for the COMP-15 NCCL stack skeleton."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

from simllm.compute import (
    NcclRoute,
    NcclStack,
    NcclStackConfig,
    NcclStackEvent,
    NcclStackEventKind,
    nccl_stack_events_from_json,
    nccl_stack_events_to_json,
    ncclAllReduce,
    ncclCommInitRank,
    validate_nccl_stack_events,
)
from simllm.core import VirtualClock

STUDY_DIR = Path(__file__).resolve().parent
TRACKED_RESULTS = STUDY_DIR / "results.csv"

REFERENCE_CONFIG = NcclStackConfig(
    channel_count=1,
    chunk_bytes=6_144,
    fifo_slots_per_channel=1,
)
PLANNER_EXPECTATIONS = {
    (4_096, 1): (6_144, 6, (6,)),
    (4_096, 2): (6_144, 6, (3, 3)),
    (4_096, 4): (6_144, 6, (2, 2, 1, 1)),
    (5_120, 1): (7_680, 8, (8,)),
    (5_120, 2): (7_680, 8, (4, 4)),
    (5_120, 4): (7_680, 8, (2, 2, 2, 2)),
    (8_192, 1): (12_288, 12, (12,)),
    (8_192, 2): (12_288, 12, (6, 6)),
    (8_192, 4): (12_288, 12, (3, 3, 3, 3)),
}

INTER_FUNCTIONS = (
    "ncclCommInitRank",
    "ncclBuildLogicalChannels",
    "ncclConstructLogicalChannel",
    "ncclAllReduce",
    "ncclPlanAllReduce",
    "ncclChunkPayload",
    "ncclAssignChunksToChannels",
    "ncclLaunchCollectiveKernel",
    "ncclCollectiveKernel",
    "ncclCopyChunkToFifo",
    "ncclStoreReadyFlag",
    "ncclStoreHead",
    "ncclProxyProgress",
    "ncclProxyPollHead",
    "ncclProxyPollReady",
    "ncclNet.isend",
    "ibverbs.post_send",
    "ibverbs.write_cqe",
    "ncclNet.test",
    "ibverbs.poll_cq",
    "ncclProxyStoreTail",
    "ncclKernelPollTail",
    "ncclReleaseFifoSlot",
    "ncclKernelComplete",
)
INTER_KINDS = (
    "call",
    "call",
    "call",
    "call",
    "call",
    "call",
    "call",
    "call",
    "call",
    "call",
    "signal_store",
    "signal_store",
    "call",
    "poll_observes",
    "poll_observes",
    "call",
    "call",
    "signal_store",
    "call",
    "poll_observes",
    "signal_store",
    "poll_observes",
    "signal_store",
    "signal_store",
)
INTER_LANES = (
    "cpu",
    "cpu",
    "cpu",
    "cpu",
    "cpu",
    "cpu",
    "cpu",
    "cpu",
    "gpu",
    "gpu",
    "gpu",
    "gpu",
    "cpu",
    "cpu",
    "cpu",
    "cpu",
    "cpu",
    "rnic",
    "cpu",
    "cpu",
    "cpu",
    "gpu",
    "gpu",
    "gpu",
)
INTER_SUBJECTS = (
    None,
    None,
    "logical_channel",
    None,
    None,
    None,
    None,
    None,
    None,
    "data_fifo_slot",
    "ready_flag",
    "head_counter",
    None,
    "head_counter",
    "ready_flag",
    None,
    None,
    "completion_queue_entry",
    None,
    "completion_queue_entry",
    "tail_counter",
    "tail_counter",
    "ready_flag",
    "kernel_completion",
)

INTRA_FUNCTIONS = INTER_FUNCTIONS[:9] + (
    "ncclNvlinkCollective",
    "ncclKernelComplete",
)
INTRA_KINDS = INTER_KINDS[:9] + ("call", "signal_store")
INTRA_LANES = INTER_LANES[:9] + ("gpu", "gpu")
INTRA_SUBJECTS = INTER_SUBJECTS[:9] + ("nvlink", "kernel_completion")


def _event_tuple(event: NcclStackEvent) -> tuple[Any, ...]:
    return (
        event.sequence,
        event.timestamp_ps,
        event.function,
        event.kind.value,
        event.lane.value,
        event.rank,
        event.communicator_id,
        event.operation_id,
        event.channel_id,
        event.chunk_id,
        event.peer_rank,
        event.subject,
        event.value,
        event.observed_signal_sequence,
    )


def _expected_inter(rank: int) -> tuple[tuple[Any, ...], ...]:
    send_peer = (rank + 1) % 4
    receive_peer = (rank - 1) % 4
    operation_id = f"inter-r{rank}"
    values = (
        4,
        1,
        receive_peer,
        4_096,
        6_144,
        1,
        1,
        1,
        1,
        6_144,
        1,
        1,
        None,
        1,
        1,
        6_144,
        6_144,
        6_144,
        6_144,
        6_144,
        1,
        1,
        0,
        1,
    )
    observed = {13: 11, 14: 10, 19: 17, 21: 20}
    expected = []
    for sequence, (function, kind, lane, subject, value) in enumerate(
        zip(
            INTER_FUNCTIONS,
            INTER_KINDS,
            INTER_LANES,
            INTER_SUBJECTS,
            values,
            strict=True,
        )
    ):
        initialized = sequence < 3
        channel_event = sequence == 2
        chunk_event = 9 <= sequence <= 22
        expected.append(
            (
                sequence,
                0,
                function,
                kind,
                lane,
                rank,
                "ring-4",
                None if initialized else operation_id,
                0 if channel_event or chunk_event else None,
                0 if chunk_event else None,
                send_peer if channel_event or chunk_event else None,
                subject,
                value,
                observed.get(sequence),
            )
        )
    return tuple(expected)


def _expected_intra(rank: int) -> tuple[tuple[Any, ...], ...]:
    send_peer = (rank + 1) % 4
    receive_peer = (rank - 1) % 4
    operation_id = f"intra-r{rank}"
    values = (4, 1, receive_peer, 4_096, 6_144, 1, 1, 1, 1, 6_144, 1)
    expected = []
    for sequence, (function, kind, lane, subject, value) in enumerate(
        zip(
            INTRA_FUNCTIONS,
            INTRA_KINDS,
            INTRA_LANES,
            INTRA_SUBJECTS,
            values,
            strict=True,
        )
    ):
        initialized = sequence < 3
        channel_event = sequence == 2
        nvlink_event = sequence == 9
        expected.append(
            (
                sequence,
                0,
                function,
                kind,
                lane,
                rank,
                "ring-4",
                None if initialized else operation_id,
                0 if channel_event or nvlink_event else None,
                0 if nvlink_event else None,
                send_peer if channel_event or nvlink_event else None,
                subject,
                value,
                None,
            )
        )
    return tuple(expected)


def _compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_compact(value).encode()).hexdigest()


def _row(
    evidence_class: str,
    family: str,
    check: str,
    parameters: dict[str, Any],
    expected: Any,
    measured: Any,
    passed: bool,
) -> dict[str, str]:
    if evidence_class == "run_configuration":
        status = "CONFIG"
    elif evidence_class == "structural_invariant":
        status = "HOLD" if passed else "FAIL"
    else:
        status = "PASS" if passed else "FAIL"
    return {
        "evidence_class": evidence_class,
        "family": family,
        "check": check,
        "parameters": _compact(parameters),
        "expected": _compact(expected),
        "measured": _compact(measured),
        "status": status,
    }


def _run_reference_route(
    route: NcclRoute,
    rank: int,
) -> tuple[VirtualClock, NcclStack, Any, Any]:
    clock = VirtualClock()
    stack = NcclStack(clock=clock, config=REFERENCE_CONFIG)
    communicator = ncclCommInitRank(
        stack,
        nranks=4,
        communicator_id="ring-4",
        rank=rank,
    )
    result = ncclAllReduce(
        communicator,
        payload_bytes=4_096,
        operation_id=f"{route.value.split('_')[0]}-r{rank}",
        route=route,
    )
    return clock, stack, communicator, result


def run_study() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    route_runs: dict[tuple[NcclRoute, int], tuple[VirtualClock, NcclStack, Any, Any]] = {}
    for route in (NcclRoute.INTER_NODE, NcclRoute.INTRA_NODE):
        rows.append(
            _row(
                "run_configuration",
                "route_reference",
                route.value,
                {
                    "world_size": 4,
                    "payload_bytes": 4_096,
                    "channels": 1,
                    "chunk_bytes": 6_144,
                    "fifo_slots": 1,
                    "ranks": [0, 1, 2, 3],
                },
                None,
                None,
                True,
            )
        )
        for rank in range(4):
            clock, stack, communicator, result = _run_reference_route(route, rank)
            route_runs[(route, rank)] = (clock, stack, communicator, result)
            measured = tuple(_event_tuple(event) for event in stack.events)
            expected = (
                _expected_inter(rank)
                if route is NcclRoute.INTER_NODE
                else _expected_intra(rank)
            )
            rows.append(
                _row(
                    "behavioral_relation",
                    f"{route.value}_event_sequence",
                    f"rank_{rank}",
                    {"rank": rank, "route": route.value},
                    {"event_count": len(expected), "sha256": _digest(expected)},
                    {"event_count": len(measured), "sha256": _digest(measured)},
                    measured == expected,
                )
            )

    planner_layout_holds: dict[tuple[int, int], bool] = {}
    rows.append(
        _row(
            "run_configuration",
            "planner_sweep",
            "payload_by_channel_count",
            {
                "world_size": 4,
                "payload_bytes": [4_096, 5_120, 8_192],
                "channel_count": [1, 2, 4],
                "chunk_bytes": 1_024,
            },
            None,
            None,
            True,
        )
    )
    for (payload_bytes, channel_count), planner_expectation in PLANNER_EXPECTATIONS.items():
        stack = NcclStack(
            clock=VirtualClock(),
            config=NcclStackConfig(
                channel_count=channel_count,
                chunk_bytes=1_024,
                fifo_slots_per_channel=1,
            ),
        )
        communicator = ncclCommInitRank(
            stack,
            nranks=4,
            communicator_id="planner-ring",
            rank=0,
        )
        plan = stack.planner.plan_all_reduce(
            communicator,
            payload_bytes=payload_bytes,
            operation_id=f"planner-p{payload_bytes}-c{channel_count}",
        )
        expected_wire, expected_chunks, expected_assignment = planner_expectation
        expected_last = 512 if payload_bytes == 5_120 else 1_024
        planner_layout_holds[(payload_bytes, channel_count)] = (
            sum(chunk.byte_count for chunk in plan.chunks) == plan.wire_bytes
            and [chunk.offset_bytes for chunk in plan.chunks]
            == [chunk_id * 1_024 for chunk_id in range(len(plan.chunks))]
            and all(chunk.byte_count == 1_024 for chunk in plan.chunks[:-1])
            and plan.chunks[-1].byte_count == expected_last
        )
        parameters = {
            "payload_bytes": payload_bytes,
            "channel_count": channel_count,
        }
        rows.extend(
            (
                _row(
                    "behavioral_relation",
                    "planner_chunk_count",
                    f"p{payload_bytes}_c{channel_count}",
                    parameters,
                    expected_chunks,
                    len(plan.chunks),
                    len(plan.chunks) == expected_chunks
                    and plan.wire_bytes == expected_wire,
                ),
                _row(
                    "behavioral_relation",
                    "planner_channel_count",
                    f"p{payload_bytes}_c{channel_count}",
                    parameters,
                    channel_count,
                    len(communicator.logical_channels),
                    len(communicator.logical_channels) == channel_count,
                ),
                _row(
                    "behavioral_relation",
                    "planner_channel_assignment",
                    f"p{payload_bytes}_c{channel_count}",
                    parameters,
                    expected_assignment,
                    plan.chunks_per_channel(),
                    plan.chunks_per_channel() == expected_assignment,
                ),
            )
        )

    all_route_runs = list(route_runs.values())
    round_trip_holds = all(
        nccl_stack_events_from_json(nccl_stack_events_to_json(stack.events))
        == stack.events
        for _, stack, _, _ in all_route_runs
    )
    monotonic_holds = True
    for _, stack, _, _ in all_route_runs:
        try:
            validate_nccl_stack_events(stack.events)
        except ValueError:
            monotonic_holds = False
    polls = [
        event
        for (route, _), (_, stack, _, _) in route_runs.items()
        if route is NcclRoute.INTER_NODE
        for event in stack.events
        if event.kind is NcclStackEventKind.POLL_OBSERVES
    ]
    causal_holds = len(polls) == 16 and all(
        event.observed_signal_sequence is not None
        and event.observed_signal_sequence < event.sequence
        for event in polls
    )
    intra_off_holds = all(
        not any(
            event.function.startswith(("ncclProxy", "ncclNet.", "ibverbs."))
            for event in route_runs[(NcclRoute.INTRA_NODE, rank)][1].events
        )
        for rank in range(4)
    )
    inter_quiescent = all(
        all(
            snapshot.head == snapshot.tail
            and not any(snapshot.ready_flags)
            and all(chunk_id is None for chunk_id in snapshot.slot_chunk_ids)
            and all(byte_count == 0 for byte_count in snapshot.slot_byte_counts)
            for snapshot in route_runs[(NcclRoute.INTER_NODE, rank)][3].channel_snapshots
        )
        for rank in range(4)
    )
    intra_untouched = all(
        all(
            snapshot.head == snapshot.tail == 0
            and not any(snapshot.ready_flags)
            and all(chunk_id is None for chunk_id in snapshot.slot_chunk_ids)
            and all(byte_count == 0 for byte_count in snapshot.slot_byte_counts)
            for snapshot in route_runs[(NcclRoute.INTRA_NODE, rank)][3].channel_snapshots
        )
        for rank in range(4)
    )
    zero_time_holds = all(
        clock.now_ps == 0 and all(event.timestamp_ps == 0 for event in stack.events)
        for clock, stack, _, _ in all_route_runs
    )
    planner_chunk_bytes_hold = all(planner_layout_holds.values())
    structural = (
        ("event_schema_round_trip", "8/8", "8/8" if round_trip_holds else "failure", round_trip_holds),
        ("monotonic_sequence_and_time", "8/8", "8/8" if monotonic_holds else "failure", monotonic_holds),
        ("signal_poll_causality", 16, len(polls), causal_holds),
        ("intra_proxy_and_net_off", "4/4", "4/4" if intra_off_holds else "failure", intra_off_holds),
        ("inter_fifo_quiescence", "4/4", "4/4" if inter_quiescent else "failure", inter_quiescent),
        ("intra_fifo_untouched", "4/4", "4/4" if intra_untouched else "failure", intra_untouched),
        ("zero_duration_clock", "8/8", "8/8" if zero_time_holds else "failure", zero_time_holds),
        (
            "planner_wire_byte_conservation",
            "9/9",
            "9/9" if planner_chunk_bytes_hold else "failure",
            planner_chunk_bytes_hold,
        ),
    )
    for check, expected, measured, holds in structural:
        rows.append(
            _row(
                "structural_invariant",
                "fatal_unscored",
                check,
                {},
                expected,
                measured,
                holds,
            )
        )
    return rows


def _render_csv(rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=(
            "evidence_class",
            "family",
            "check",
            "parameters",
            "expected",
            "measured",
            "status",
        ),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=STUDY_DIR,
        help="output directory for results.csv",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare generated bytes with the tracked results.csv",
    )
    arguments = parser.parse_args()

    rows = run_study()
    rendered = _render_csv(rows)
    output_path = arguments.out.resolve() / "results.csv"
    if arguments.check:
        if not TRACKED_RESULTS.is_file() or TRACKED_RESULTS.read_bytes() != rendered:
            raise SystemExit(f"generated study rows differ from tracked {TRACKED_RESULTS}")
        print(f"tracked results match {len(rows)} rows")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(rendered)
        print(f"wrote {len(rows)} rows to {output_path}")

    classes = Counter(row["evidence_class"] for row in rows)
    behavioral = [row for row in rows if row["evidence_class"] == "behavioral_relation"]
    structural = [row for row in rows if row["evidence_class"] == "structural_invariant"]
    family_status = {
        family: all(row["status"] == "PASS" for row in behavioral if row["family"] == family)
        for family in {row["family"] for row in behavioral}
    }
    passed_instances = sum(row["status"] == "PASS" for row in behavioral)
    held_structural = sum(row["status"] == "HOLD" for row in structural)
    print(f"run configurations: {classes['run_configuration']}")
    print(
        "scored behavioral relation families: "
        f"{sum(family_status.values())}/{len(family_status)} pass"
    )
    print(
        "scored behavioral relation instances: "
        f"{passed_instances}/{len(behavioral)} pass"
    )
    print(
        "fatal unscored structural invariants: "
        f"{held_structural}/{len(structural)} hold"
    )
    if not all(family_status.values()) or held_structural != len(structural):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
