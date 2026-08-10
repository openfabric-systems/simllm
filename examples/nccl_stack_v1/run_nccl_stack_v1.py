"""Run the amended structural study for the COMP-15 NCCL stack skeleton."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

from simllm import compute
from simllm.compute import (
    NcclRoute,
    NcclStack,
    NcclStackConfig,
    NcclStackEvent,
    NcclStackEventKind,
    NcclStackLane,
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
    chunk_bytes=1_024,
    fifo_slots_per_channel=2,
)
PLANNER_EXPECTATIONS = {
    (4_096, 1): (6_144, 24, (24,), 4),
    (4_096, 2): (6_144, 24, (12, 12), 4),
    (4_096, 4): (6_144, 24, (6, 6, 6, 6), 4),
    (8_192, 1): (12_288, 48, (48,), 8),
    (8_192, 2): (12_288, 48, (24, 24), 8),
    (8_192, 4): (12_288, 48, (12, 12, 12, 12), 8),
    (16_384, 1): (24_576, 96, (96,), 16),
    (16_384, 2): (24_576, 96, (48, 48), 16),
    (16_384, 4): (24_576, 96, (24, 24, 24, 24), 16),
}


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
        event.slot_id,
        event.send_peer_rank,
        event.receive_peer_rank,
        event.subject,
        event.value,
        event.observed_signal_sequence,
    )


def _expected_event(
    events: list[tuple[Any, ...]],
    *,
    function: str,
    kind: str,
    lane: str,
    rank: int,
    operation_id: str | None,
    channel_id: int | None = None,
    chunk_id: int | None = None,
    slot_id: int | None = None,
    send_peer_rank: int | None = None,
    receive_peer_rank: int | None = None,
    subject: str | None = None,
    value: int | None = None,
    observed_signal_sequence: int | None = None,
) -> int:
    sequence = len(events)
    events.append(
        (
            sequence,
            0,
            function,
            kind,
            lane,
            rank,
            "ring-4",
            operation_id,
            channel_id,
            chunk_id,
            slot_id,
            send_peer_rank,
            receive_peer_rank,
            subject,
            value,
            observed_signal_sequence,
        )
    )
    return sequence


def _expected_prefix(
    events: list[tuple[Any, ...]],
    *,
    rank: int,
    operation_id: str,
    include_proxy_save: bool,
) -> tuple[int, int]:
    send_peer = (rank + 1) % 4
    receive_peer = (rank - 1) % 4
    _expected_event(
        events,
        function="ncclCommInitRank",
        kind="call",
        lane="cpu",
        rank=rank,
        operation_id=None,
        value=4,
    )
    _expected_event(
        events,
        function="ncclBuildRings",
        kind="call",
        lane="cpu",
        rank=rank,
        operation_id=None,
        value=1,
    )
    _expected_event(
        events,
        function="initChannel",
        kind="call",
        lane="cpu",
        rank=rank,
        operation_id=None,
        channel_id=0,
        send_peer_rank=send_peer,
        receive_peer_rank=receive_peer,
        subject="logical_channel",
    )
    for function, value in (
        ("ncclAllReduce", 4_096),
        ("ncclEnqueueCheck", 4_096),
        ("scheduleCollTasksToPlan", 6),
        ("calcCollChunking", 6),
    ):
        _expected_event(
            events,
            function=function,
            kind="call",
            lane="cpu",
            rank=rank,
            operation_id=operation_id,
            value=value,
        )
    if include_proxy_save:
        _expected_event(
            events,
            function="ncclProxySaveOp",
            kind="call",
            lane="cpu",
            rank=rank,
            operation_id=operation_id,
            send_peer_rank=send_peer,
            subject="proxy_operation",
            value=6,
        )
    for function, lane, value in (
        ("ncclLaunchKernel", "cpu", 6),
        ("ncclKernelMain", "gpu", 6),
        ("runRing", "gpu", 6),
    ):
        _expected_event(
            events,
            function=function,
            kind="call",
            lane=lane,
            rank=rank,
            operation_id=operation_id,
            value=value,
        )
    return send_peer, receive_peer


def _expected_inter(rank: int) -> tuple[tuple[Any, ...], ...]:
    events: list[tuple[Any, ...]] = []
    operation_id = f"inter-r{rank}"
    send_peer, _ = _expected_prefix(
        events,
        rank=rank,
        operation_id=operation_id,
        include_proxy_save=True,
    )
    head_signals: dict[int, int] = {}
    for batch in range(3):
        ready_signals: dict[int, int] = {}
        tail_signals: dict[int, int] = {}
        chunks = range(2 * batch, 2 * batch + 2)
        for chunk_id in chunks:
            slot_id = chunk_id % 2
            if chunk_id >= 2:
                _expected_event(
                    events,
                    function="waitPeer",
                    kind="poll_observes",
                    lane="gpu",
                    rank=rank,
                    operation_id=operation_id,
                    channel_id=0,
                    chunk_id=chunk_id,
                    slot_id=slot_id,
                    send_peer_rank=send_peer,
                    subject="head_counter",
                    value=chunk_id - 1,
                    observed_signal_sequence=head_signals[slot_id],
                )
            ready_signals[chunk_id] = _expected_event(
                events,
                function="waitPeer",
                kind="signal_store",
                lane="gpu",
                rank=rank,
                operation_id=operation_id,
                channel_id=0,
                chunk_id=chunk_id,
                slot_id=slot_id,
                send_peer_rank=send_peer,
                subject="ready_flag",
                value=1,
            )
            _expected_event(
                events,
                function="genericOp",
                kind="call",
                lane="gpu",
                rank=rank,
                operation_id=operation_id,
                channel_id=0,
                chunk_id=chunk_id,
                slot_id=slot_id,
                send_peer_rank=send_peer,
                subject="data_fifo_slot",
                value=1_024,
            )
            tail_signals[chunk_id] = _expected_event(
                events,
                function="postPeer",
                kind="signal_store",
                lane="gpu",
                rank=rank,
                operation_id=operation_id,
                channel_id=0,
                chunk_id=chunk_id,
                slot_id=slot_id,
                send_peer_rank=send_peer,
                subject="tail_counter",
                value=chunk_id + 1,
            )
        for function in ("ncclProxyProgress", "sendProxyProgress"):
            _expected_event(
                events,
                function=function,
                kind="call",
                lane="cpu",
                rank=rank,
                operation_id=operation_id,
                send_peer_rank=send_peer,
            )
        for chunk_id in chunks:
            slot_id = chunk_id % 2
            for subject, producer in (
                ("tail_counter", tail_signals[chunk_id]),
                ("ready_flag", ready_signals[chunk_id]),
            ):
                _expected_event(
                    events,
                    function="sendProxyProgress",
                    kind="poll_observes",
                    lane="cpu",
                    rank=rank,
                    operation_id=operation_id,
                    channel_id=0,
                    chunk_id=chunk_id,
                    slot_id=slot_id,
                    send_peer_rank=send_peer,
                    subject=subject,
                    value=chunk_id + 1 if subject == "tail_counter" else 1,
                    observed_signal_sequence=producer,
                )
            for function in ("ncclNet.isend", "wrap_ibv_post_send"):
                _expected_event(
                    events,
                    function=function,
                    kind="call",
                    lane="cpu",
                    rank=rank,
                    operation_id=operation_id,
                    channel_id=0,
                    chunk_id=chunk_id,
                    slot_id=slot_id,
                    send_peer_rank=send_peer,
                    value=1_024,
                )
            _expected_event(
                events,
                function="simllmRnicRingDoorbell",
                kind="signal_store",
                lane="rnic",
                rank=rank,
                operation_id=operation_id,
                channel_id=0,
                chunk_id=chunk_id,
                slot_id=slot_id,
                send_peer_rank=send_peer,
                subject="doorbell",
                value=1,
            )
        completion_signals = {}
        for chunk_id in chunks:
            completion_signals[chunk_id] = _expected_event(
                events,
                function="simllmNetworkComplete",
                kind="signal_store",
                lane="rnic",
                rank=rank,
                operation_id=operation_id,
                channel_id=0,
                chunk_id=chunk_id,
                slot_id=chunk_id % 2,
                send_peer_rank=send_peer,
                subject="completion_queue_entry",
                value=1_024,
            )
        for function in ("ncclProxyProgress", "sendProxyProgress"):
            _expected_event(
                events,
                function=function,
                kind="call",
                lane="cpu",
                rank=rank,
                operation_id=operation_id,
                send_peer_rank=send_peer,
            )
        for chunk_id in chunks:
            slot_id = chunk_id % 2
            _expected_event(
                events,
                function="ncclNet.test",
                kind="call",
                lane="cpu",
                rank=rank,
                operation_id=operation_id,
                channel_id=0,
                chunk_id=chunk_id,
                slot_id=slot_id,
                send_peer_rank=send_peer,
                value=1_024,
            )
            _expected_event(
                events,
                function="wrap_ibv_poll_cq",
                kind="poll_observes",
                lane="cpu",
                rank=rank,
                operation_id=operation_id,
                channel_id=0,
                chunk_id=chunk_id,
                slot_id=slot_id,
                send_peer_rank=send_peer,
                subject="completion_queue_entry",
                value=1_024,
                observed_signal_sequence=completion_signals[chunk_id],
            )
            _expected_event(
                events,
                function="sendProxyProgress",
                kind="signal_store",
                lane="cpu",
                rank=rank,
                operation_id=operation_id,
                channel_id=0,
                chunk_id=chunk_id,
                slot_id=slot_id,
                send_peer_rank=send_peer,
                subject="ready_flag",
                value=0,
            )
            head_signals[slot_id] = _expected_event(
                events,
                function="sendProxyProgress",
                kind="signal_store",
                lane="cpu",
                rank=rank,
                operation_id=operation_id,
                channel_id=0,
                chunk_id=chunk_id,
                slot_id=slot_id,
                send_peer_rank=send_peer,
                subject="head_counter",
                value=chunk_id + 1,
            )
    _expected_event(
        events,
        function="simllmKernelComplete",
        kind="signal_store",
        lane="gpu",
        rank=rank,
        operation_id=operation_id,
        subject="kernel_completion",
        value=6,
    )
    return tuple(events)


def _expected_intra(rank: int) -> tuple[tuple[Any, ...], ...]:
    events: list[tuple[Any, ...]] = []
    operation_id = f"intra-r{rank}"
    send_peer, _ = _expected_prefix(
        events,
        rank=rank,
        operation_id=operation_id,
        include_proxy_save=False,
    )
    for chunk_id in range(6):
        _expected_event(
            events,
            function="genericOp",
            kind="call",
            lane="gpu",
            rank=rank,
            operation_id=operation_id,
            channel_id=0,
            chunk_id=chunk_id,
            send_peer_rank=send_peer,
            subject="nvlink",
            value=1_024,
        )
    _expected_event(
        events,
        function="simllmKernelComplete",
        kind="signal_store",
        lane="gpu",
        rank=rank,
        operation_id=operation_id,
        subject="kernel_completion",
        value=6,
    )
    return tuple(events)


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


def _run_reference_route(route: NcclRoute, rank: int):
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
    route_runs = {}
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
                    "warps_per_channel": 1,
                    "chunk_bytes": 1_024,
                    "fifo_slots": 2,
                    "ranks": [0, 1, 2, 3],
                },
                None,
                None,
                True,
            )
        )
        for rank in range(4):
            run = _run_reference_route(route, rank)
            route_runs[(route, rank)] = run
            measured = tuple(_event_tuple(event) for event in run[1].events)
            expected = _expected_inter(rank) if route is NcclRoute.INTER_NODE else _expected_intra(rank)
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

    rows.append(
        _row(
            "run_configuration",
            "planner_sweep",
            "payload_by_channel_count",
            {
                "world_size": 4,
                "payload_bytes": [4_096, 8_192, 16_384],
                "channel_count": [1, 2, 4],
                "warps_per_channel": 1,
                "chunk_bytes": 256,
            },
            None,
            None,
            True,
        )
    )
    planner_runs = {}
    for (payload_bytes, channel_count), expectation in PLANNER_EXPECTATIONS.items():
        stack = NcclStack(
            clock=VirtualClock(),
            config=NcclStackConfig(
                channel_count=channel_count,
                chunk_bytes=256,
                fifo_slots_per_channel=2,
            ),
        )
        communicator = ncclCommInitRank(
            stack,
            nranks=4,
            communicator_id="planner-ring",
            rank=0,
        )
        result = ncclAllReduce(
            communicator,
            payload_bytes=payload_bytes,
            operation_id=f"planner-p{payload_bytes}-c{channel_count}",
            route=NcclRoute.INTRA_NODE,
        )
        plan = result.plan
        planner_runs[(payload_bytes, channel_count)] = (communicator, plan)
        expected_wire, expected_chunks, expected_assignment, chunks_per_step = expectation
        parameters = {"payload_bytes": payload_bytes, "channel_count": channel_count}
        expected_steps = tuple(
            (step, "reduce_scatter" if step < 3 else "all_gather", chunks_per_step)
            for step in range(6)
        )
        measured_steps = tuple(
            (
                step,
                next(chunk.phase for chunk in plan.chunks if chunk.ring_step == step),
                plan.chunks_per_step()[step],
            )
            for step in range(plan.step_count)
        )
        rows.extend(
            (
                _row(
                    "behavioral_relation",
                    "planner_ring_step_structure",
                    f"p{payload_bytes}_c{channel_count}",
                    parameters,
                    expected_steps,
                    measured_steps,
                    measured_steps == expected_steps,
                ),
                _row(
                    "behavioral_relation",
                    "planner_chunk_count",
                    f"p{payload_bytes}_c{channel_count}",
                    parameters,
                    {"wire_bytes": expected_wire, "total_chunks": expected_chunks},
                    {"wire_bytes": plan.wire_bytes, "total_chunks": len(plan.chunks)},
                    plan.wire_bytes == expected_wire and len(plan.chunks) == expected_chunks,
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
    schema_and_order_holds = True
    for _, stack, _, _ in all_route_runs:
        try:
            validate_nccl_stack_events(stack.events)
            schema_and_order_holds &= (
                nccl_stack_events_from_json(nccl_stack_events_to_json(stack.events))
                == stack.events
            )
        except (TypeError, ValueError):
            schema_and_order_holds = False
    polls = [
        event
        for (route, _), (_, stack, _, _) in route_runs.items()
        if route is NcclRoute.INTER_NODE
        for event in stack.events
        if event.kind is NcclStackEventKind.POLL_OBSERVES
    ]
    causal_holds = len(polls) == 88 and all(
        event.observed_signal_sequence is not None
        and event.observed_signal_sequence < event.sequence
        and stack.events[event.observed_signal_sequence].kind
        is NcclStackEventKind.SIGNAL_STORE
        for (route, _), (_, stack, _, _) in route_runs.items()
        if route is NcclRoute.INTER_NODE
        for event in stack.events
        if event.kind is NcclStackEventKind.POLL_OBSERVES
    )
    first_stack = route_runs[(NcclRoute.INTER_NODE, 0)][1]
    second_stack = route_runs[(NcclRoute.INTER_NODE, 1)][1]
    before_foreign_probe = second_stack.events
    try:
        second_stack._observer.poll(
            "sendProxyProgress",
            NcclStackLane.CPU,
            first_stack.events[11],
        )
    except ValueError:
        foreign_scope_holds = second_stack.events == before_foreign_probe
    else:
        foreign_scope_holds = False
    inter_fifo_holds = all(
        all(
            snapshot.head == snapshot.tail == 6
            and snapshot.high_watermark == 2
            and not any(snapshot.ready_flags)
            and all(chunk_id is None for chunk_id in snapshot.slot_chunk_ids)
            and all(byte_count == 0 for byte_count in snapshot.slot_byte_counts)
            for snapshot in route_runs[(NcclRoute.INTER_NODE, rank)][3].channel_snapshots
        )
        for rank in range(4)
    )
    intra_inactive_holds = all(
        not any(
            event.function
            in {
                "ncclProxySaveOp",
                "ncclProxyProgress",
                "sendProxyProgress",
                "ncclNet.isend",
                "ncclNet.test",
                "wrap_ibv_post_send",
                "wrap_ibv_poll_cq",
                "simllmRnicRingDoorbell",
                "simllmNetworkComplete",
            }
            for event in route_runs[(NcclRoute.INTRA_NODE, rank)][1].events
        )
        and all(
            snapshot.head == snapshot.tail == snapshot.high_watermark == 0
            and not any(snapshot.ready_flags)
            for snapshot in route_runs[(NcclRoute.INTRA_NODE, rank)][3].channel_snapshots
        )
        for rank in range(4)
    )
    zero_time_holds = all(
        clock.now_ps == 0 and all(event.timestamp_ps == 0 for event in stack.events)
        for clock, stack, _, _ in all_route_runs
    )
    configured_channels_holds = all(
        len(communicator.logical_channels) == channel_count
        and plan.channel_count == channel_count
        for (_, channel_count), (communicator, plan) in planner_runs.items()
    )
    planner_conservation_holds = all(
        plan.step_count == 6
        and sum(plan.chunks_per_step()) == len(plan.chunks)
        and sum(chunk.byte_count for chunk in plan.chunks) == plan.wire_bytes
        and [chunk.offset_bytes for chunk in plan.chunks]
        == [chunk.chunk_id * 256 for chunk in plan.chunks]
        and all(chunk.byte_count == 256 for chunk in plan.chunks)
        for _, plan in planner_runs.values()
    )
    network_order_holds = all(
        all(
            stack.events[post].function == "wrap_ibv_post_send"
            and stack.events[doorbell].function == "simllmRnicRingDoorbell"
            and stack.events[completion].function == "simllmNetworkComplete"
            and post < doorbell < completion < head
            for post, doorbell, completion, head in zip(
                (22, 27, 54, 59, 86, 91),
                (23, 28, 55, 60, 87, 92),
                (29, 30, 61, 62, 93, 94),
                (36, 40, 68, 72, 100, 104),
                strict=True,
            )
        )
        for rank in range(4)
        for stack in (route_runs[(NcclRoute.INTER_NODE, rank)][1],)
    )
    forbidden_exports = {
        "Ibverbs",
        "IbverbsRequest",
        "NcclCommunicator",
        "NcclDataFifoSlot",
        "NcclGpuChannel",
        "NcclNetPlugin",
        "NcclProxyProgressEngine",
        "NcclTrafficPlanner",
    }
    receive_absent_holds = (
        forbidden_exports.isdisjoint(compute.__all__)
        and all(not hasattr(compute, name) for name in forbidden_exports)
        and not hasattr(first_stack._nccl_net, "irecv")
        and not hasattr(first_stack._ibverbs, "post_receive")
        and not any(
            "irecv" in event.function or "post_recv" in event.function
            for _, stack, _, _ in all_route_runs
            for event in stack.events
        )
    )
    structural = (
        (
            "configured_channel_count",
            "9/9",
            "9/9" if configured_channels_holds else "failure",
            configured_channels_holds,
        ),
        (
            "strict_schema_stream_order_and_time",
            "8/8",
            "8/8" if schema_and_order_holds else "failure",
            schema_and_order_holds,
        ),
        ("signal_poll_causality", 88, len(polls), causal_holds),
        (
            "foreign_observer_scope",
            "rejected before mutation",
            "rejected before mutation" if foreign_scope_holds else "failure",
            foreign_scope_holds,
        ),
        (
            "inter_fifo_depth_and_quiescence",
            "4/4",
            "4/4" if inter_fifo_holds else "failure",
            inter_fifo_holds,
        ),
        (
            "intra_stack_layers_inactive",
            "4/4",
            "4/4" if intra_inactive_holds else "failure",
            intra_inactive_holds,
        ),
        (
            "zero_duration_clock",
            "8/8",
            "8/8" if zero_time_holds else "failure",
            zero_time_holds,
        ),
        (
            "ring_step_and_byte_conservation",
            "9/9",
            "9/9" if planner_conservation_holds else "failure",
            planner_conservation_holds,
        ),
        (
            "post_doorbell_completion_head_order",
            "24/24",
            "24/24" if network_order_holds else "failure",
            network_order_holds,
        ),
        (
            "receive_leg_and_internal_surface_absent",
            "absent",
            "absent" if receive_absent_holds else "failure",
            receive_absent_holds,
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
    if any(row["status"] == "FAIL" for row in rows):
        raise SystemExit("one or more NCCL stack study checks failed")


if __name__ == "__main__":
    main()
