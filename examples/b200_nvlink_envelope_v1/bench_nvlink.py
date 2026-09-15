"""Run the three frozen lanes of the B200 NVLink envelope study.

The freeze lives in ``expectations.md`` and ``expectations.json`` next to this
file and was committed before the lane existed. Nothing here reads a measured
value back into a bound; the scorer does that separately.

Lane P1 times PyTorch device copies between a pinned pair (and, in stage 2,
across placements on an eight-GPU board) from rank 0 with every visible
device. Lane P2 times ``torch.distributed`` point-to-point over NCCL with an
8-byte completion token. Lane P3 times NCCL all-reduce, all-gather and
reduce-scatter at the frozen widths. Every timed block is bracketed by CUDA
events on the issuing stream and the reported time is the block time divided
by the iteration count.

Usage::

    python -m torch.distributed.run --standalone --nproc_per_node=2 \\
        bench_nvlink.py --stage 1 --output <dir>

``--mock`` runs the same control flow on CPU tensors over the gloo backend
with capped payloads and a host clock, so the flow can be exercised without a
GPU. Mock output is a flow check, never evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

STUDY = "b200_nvlink_envelope_v1"
SCHEMA = "simllm-b200-nvlink-envelope-result-v1"

#: the frozen payload sweep, 8 B and every power of two from 1 KiB to 1 GiB
PAYLOAD_BYTES: tuple[int, ...] = (8,) + tuple(1 << k for k in range(10, 31))
WARMUP_ITERATIONS = 5
TIMED_ITERATIONS = 20
TIMED_ITERATIONS_ABOVE_64MIB = 10
LARGE_PAYLOAD_BYTES = 64 * 1024 * 1024

MOCK_PAYLOAD_CAP_BYTES = 1024 * 1024
MOCK_WARMUP_ITERATIONS = 1
MOCK_TIMED_ITERATIONS = 2
MOCK_DEVICE_COUNT = {1: 2, 2: 8}

#: amendment of 2026-09-15: at or below this payload the row of record is
#: measured by CUDA graph replay and the timed count rises to 200, because
#: eager Python dispatch, not the link, bounds a smaller transfer.
GRAPH_ROW_MAX_BYTES = 1024 * 1024
TIMED_ITERATIONS_AT_OR_BELOW_1MIB = 200
METHOD_GRAPH = "graph"
METHOD_EAGER = "eager"
METHODS: tuple[str, ...] = (METHOD_GRAPH, METHOD_EAGER)
GRAPH_WARMUP_ITERATIONS = 3
GRAPH_PROBE_ELEMENTS = 8

#: one replay is thrown away before the timed one. The first replay of a fresh
#: graph pays its instantiation and, for a point to point graph, its first
#: channel use; the first captured payload of a lane must not carry that.
UNTIMED_REPLAYS = 1

STAGE2_ALL_PAIRS_PAYLOADS: tuple[int, ...] = (65536, 16777216, 1073741824)
STAGE2_CONCURRENT_PAYLOAD = 67108864
STAGE2_FANIN_DONORS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
STAGE2_P2_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (0, 7), (3, 4), (7, 0))
STAGE2_P2_PAYLOADS: tuple[int, ...] = (65536, 16777216, 1073741824)

WIDTHS: dict[int, tuple[int, ...]] = {1: (2,), 2: (2, 4, 8)}
COLLECTIVE_OPS: tuple[str, ...] = ("all_reduce", "all_gather", "reduce_scatter")
FLOAT32_BYTES = 4
TOKEN_BYTES = 8

EXIT_NO_PEER_ACCESS = 4
EXIT_TOO_FEW_DEVICES = 5

#: process group timeout. A collective that hangs must raise long before the
#: rental cap rather than spin, so the stage script can still write evidence.
PG_TIMEOUT_SECONDS = 120


def busbw_factor(op: str, width: int) -> float:
    """Return the nccl-tests bus bandwidth factor for one collective."""

    if op == "all_reduce":
        return 2.0 * (width - 1) / width
    return float(width - 1) / width


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _progress(message: str) -> None:
    """Print one flushed progress line, so a tail of the log shows the lane."""

    print(f"LANE {_utc_now()} {message}", flush=True)


def _init_process_group(mock: bool, local_rank: int, timeout_seconds: int) -> None:
    """Start the process group with the rank to device mapping pinned.

    Without ``device_id`` a communicator binds to whatever device happens to be
    current when it is first used, which is how a leaked ``set_device`` in one
    lane can give two ranks the same device and hang the next communicator.
    The timeout turns a stuck collective into an error instead of a spin.
    """

    timeout = timedelta(seconds=timeout_seconds)
    if mock:
        dist.init_process_group("gloo", timeout=timeout)
        return
    dist.init_process_group(
        "nccl",
        timeout=timeout,
        device_id=torch.device("cuda", local_rank),
    )


def _driver_version() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    line = out.stdout.strip().splitlines()
    return line[0].strip() if line else "unavailable"


def _pci_bus_ids(count: int) -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ["unavailable"] * count
    lines = [line.strip() for line in out.stdout.strip().splitlines() if line.strip()]
    if len(lines) < count:
        lines += ["unavailable"] * (count - len(lines))
    return lines[:count]


def _nccl_version() -> Any:
    if not torch.cuda.is_available():
        return None
    try:
        return list(torch.cuda.nccl.version())
    except (AttributeError, RuntimeError):
        return None


def _iterations(nbytes: int, mock: bool) -> tuple[int, int]:
    """Return ``(warmup, timed)`` for one payload.

    The freeze sets 5 warmup and 20 timed, 10 above 64 MiB. The amendment
    raises the timed count to 200 at or below 1 MiB, where one row was noisier
    than the payload term it was meant to measure.
    """

    if mock:
        return MOCK_WARMUP_ITERATIONS, MOCK_TIMED_ITERATIONS
    if nbytes <= GRAPH_ROW_MAX_BYTES:
        return WARMUP_ITERATIONS, TIMED_ITERATIONS_AT_OR_BELOW_1MIB
    if nbytes > LARGE_PAYLOAD_BYTES:
        return WARMUP_ITERATIONS, TIMED_ITERATIONS_ABOVE_64MIB
    return WARMUP_ITERATIONS, TIMED_ITERATIONS


def _of_record(method: str, requested_bytes: int) -> bool:
    """Return whether this method is the amendment's row of record."""

    expected = METHOD_GRAPH if requested_bytes <= GRAPH_ROW_MAX_BYTES else METHOD_EAGER
    return method == expected


def _graph_supported(mock: bool) -> bool:
    """Return whether this process can capture a CUDA graph at all."""

    return not mock and torch.cuda.is_available() and hasattr(torch.cuda, "CUDAGraph")


def _sweep(mock: bool) -> tuple[int, ...]:
    if mock:
        return tuple(size for size in PAYLOAD_BYTES if size <= MOCK_PAYLOAD_CAP_BYTES)
    return PAYLOAD_BYTES


def _capped(nbytes: int, mock: bool) -> int:
    return min(nbytes, MOCK_PAYLOAD_CAP_BYTES) if mock else nbytes


def _rate(nbytes: int, seconds: float) -> float:
    return float(nbytes) / seconds if seconds > 0.0 else float("inf")


def _device(index: int, mock: bool) -> torch.device:
    return torch.device("cpu") if mock else torch.device("cuda", index)


class BufferPool:
    """Cache device buffers so a long sweep does not churn the allocator."""

    def __init__(self, mock: bool) -> None:
        self._mock = mock
        self._buffers: dict[tuple[int, str, int], torch.Tensor] = {}

    def get(self, index: int, slot: str, nbytes: int) -> torch.Tensor:
        key = (index, slot, nbytes)
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = torch.empty(nbytes, dtype=torch.uint8, device=_device(index, self._mock))
            buffer.fill_(1 + (index % 7))
            self._buffers[key] = buffer
        return buffer

    def release(self) -> None:
        self._buffers.clear()
        if not self._mock and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _capture(issue: Any, timed: int, device_index: int | None = None) -> Any:
    """Capture ``timed`` issues of ``issue`` into a CUDA graph.

    The caller warms the work up first, on a side stream, with the
    communicator already initialized. The capture stream is created here on the
    device that owns the work: the context manager's default capture stream is
    a class level singleton bound to whichever device was current when it was
    first built, which is the wrong device as soon as a lane captures on two.
    """

    graph = torch.cuda.CUDAGraph()
    if device_index is None:
        capture_stream = torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=capture_stream, capture_error_mode="thread_local"):
            for _ in range(timed):
                issue()
        return graph
    with torch.cuda.device(device_index):
        capture_stream = torch.cuda.Stream(device=device_index)
        with torch.cuda.graph(graph, stream=capture_stream, capture_error_mode="thread_local"):
            for _ in range(timed):
                issue()
    return graph


def _capture_transfer(
    source: torch.Tensor,
    destination: torch.Tensor,
    src: int,
    dst: int,
    timed: int,
) -> Any:
    """Capture ``timed`` cross-device copies into one graph spanning both.

    A PyTorch cross-device ``copy_`` touches two streams: it records an event
    on the destination device's current stream and makes the source device's
    copy stream wait on it. Capturing only the source stream is therefore
    invalid, which is what the first graph run hit. The destination stream is
    forked into the capture from the capture stream and joined back before the
    capture ends, which is the legal shape for a multi-device graph.
    """

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.device(src):
        capture_stream = torch.cuda.Stream(device=src)
        destination_stream = torch.cuda.Stream(device=dst)
        fork = torch.cuda.Event()
        join = torch.cuda.Event()
        with torch.cuda.graph(graph, stream=capture_stream, capture_error_mode="thread_local"):
            fork.record(capture_stream)
            destination_stream.wait_event(fork)
            with torch.cuda.stream(destination_stream):
                for _ in range(timed):
                    destination.copy_(source, non_blocking=True)
                join.record(destination_stream)
            capture_stream.wait_event(join)
    return graph


def _time_transfers(
    pairs: list[tuple[int, int]],
    nbytes: int,
    warmup: int,
    timed: int,
    pool: BufferPool,
    mock: bool,
    method: str = METHOD_EAGER,
) -> tuple[list[float], str]:
    """Time one group of simultaneous copies and return per-pair seconds.

    Each copy runs on its own stream on its source device, which is the stream
    PyTorch issues a cross-device ``copy_`` on, so the events bracket the
    issuing stream. The makespan of the group is the maximum of the returned
    seconds. The second return value is empty on success and carries the
    reason when a graph row could not be captured.
    """

    sources = [
        pool.get(src, f"src{src}", nbytes) for src, _ in pairs
    ]
    destinations = [
        pool.get(dst, f"dst{index}", nbytes) for index, (_, dst) in enumerate(pairs)
    ]

    if mock:
        if method == METHOD_GRAPH:
            return [], "mock mode copies CPU tensors, which cannot be captured"
        for _ in range(warmup):
            for source, destination in zip(sources, destinations):
                destination.copy_(source)
        start = time.perf_counter_ns()
        for _ in range(timed):
            for source, destination in zip(sources, destinations):
                destination.copy_(source)
        elapsed = (time.perf_counter_ns() - start) * 1e-9 / timed
        return [elapsed] * len(pairs), ""

    if method == METHOD_GRAPH and not _graph_supported(mock):
        return [], "this process cannot capture a CUDA graph"

    # Every CUDA call below runs inside a torch.cuda.device context, which
    # restores the previous device on exit. The lane must never leave the
    # process pointed at another rank's device: a communicator created after
    # it would bind to that device and deadlock.
    streams = []
    starts = []
    stops = []
    for src, _ in pairs:
        with torch.cuda.device(src):
            streams.append(torch.cuda.Stream(device=src))
            starts.append(torch.cuda.Event(enable_timing=True))
            stops.append(torch.cuda.Event(enable_timing=True))
    devices = sorted({index for pair in pairs for index in pair})

    for _ in range(warmup):
        for index, (src, _) in enumerate(pairs):
            with torch.cuda.device(src), torch.cuda.stream(streams[index]):
                destinations[index].copy_(sources[index], non_blocking=True)
    for index in devices:
        torch.cuda.synchronize(index)

    graphs: list[Any] = []
    if method == METHOD_GRAPH:
        # One graph per pair, captured on that pair's source device and
        # spanning its destination device, so the replay keeps the concurrency
        # of the cell.
        for index, (src, dst) in enumerate(pairs):
            try:
                graphs.append(
                    _capture_transfer(sources[index], destinations[index], src, dst, timed)
                )
            except Exception as error:  # noqa: BLE001 - any capture failure falls back
                for device_index in devices:
                    torch.cuda.synchronize(device_index)
                return [], f"{type(error).__name__}: {error}"
        for _ in range(UNTIMED_REPLAYS):
            for index, (src, _) in enumerate(pairs):
                with torch.cuda.device(src), torch.cuda.stream(streams[index]):
                    graphs[index].replay()
            for device_index in devices:
                torch.cuda.synchronize(device_index)

    for index, (src, _) in enumerate(pairs):
        with torch.cuda.device(src):
            starts[index].record(streams[index])
    if graphs:
        for index, (src, _) in enumerate(pairs):
            with torch.cuda.device(src), torch.cuda.stream(streams[index]):
                graphs[index].replay()
    else:
        for _ in range(timed):
            for index, (src, _) in enumerate(pairs):
                with torch.cuda.device(src), torch.cuda.stream(streams[index]):
                    destinations[index].copy_(sources[index], non_blocking=True)
    for index, (src, _) in enumerate(pairs):
        with torch.cuda.device(src):
            stops[index].record(streams[index])
    for index in devices:
        torch.cuda.synchronize(index)

    return (
        [starts[index].elapsed_time(stops[index]) * 1e-3 / timed for index in range(len(pairs))],
        "",
    )


def _copy_row(
    cell: str,
    pairs: list[tuple[int, int]],
    requested_bytes: int,
    moved_bytes: int,
    timed: int,
    seconds: list[float],
    method: str,
) -> dict[str, Any]:
    makespan = max(seconds)
    aggregate = moved_bytes * len(pairs)
    row: dict[str, Any] = {
        "cell": cell,
        "method": method,
        "of_record": _of_record(method, requested_bytes),
        "pairs": [list(pair) for pair in pairs],
        "src": pairs[0][0],
        "dst": pairs[0][1],
        "direction": f"{pairs[0][0]}->{pairs[0][1]}",
        "requested_bytes": requested_bytes,
        "bytes": moved_bytes,
        "iterations": timed,
        "status": "measured",
        "time_ns": makespan * 1e9,
        "bytes_per_second": _rate(moved_bytes, makespan),
        "per_stream_time_ns": [value * 1e9 for value in seconds],
        "aggregate_bytes": aggregate,
        "aggregate_bytes_per_second": _rate(aggregate, makespan),
    }
    return row


def _copy_cell_rows(
    cell: str,
    pairs: list[tuple[int, int]],
    requested: int,
    pool: BufferPool,
    mock: bool,
    methods: tuple[str, ...] = (METHOD_EAGER,),
) -> list[dict[str, Any]]:
    """Time one copy cell by both methods and return the rows.

    The amendment keeps the eager measurement beside the graph replay at every
    payload, so the dispatch floor stays on the record.
    """

    moved = _capped(requested, mock)
    warmup, timed = _iterations(requested, mock)
    rows: list[dict[str, Any]] = []
    skip_reason = ""
    for method in methods:
        seconds, reason = _time_transfers(pairs, moved, warmup, timed, pool, mock, method)
        if not seconds:
            skip_reason = reason
            continue
        rows.append(_copy_row(cell, pairs, requested, moved, timed, seconds, method))
    for row in rows:
        if skip_reason:
            row["graph_skipped"] = True
            row["graph_skip_reason"] = skip_reason
        if mock and row["method"] == METHOD_EAGER and len(pairs) > 1:
            row["concurrency"] = "sequential_mock"
    return rows


def check_peer_access(devices: list[int], mock: bool) -> list[dict[str, Any]]:
    """Verify peer access both ways for every ordered pair, or fail closed."""

    if mock:
        return [{"src": -1, "dst": -1, "can_access": True, "note": "mock, no device pair"}]
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for src in devices:
        for dst in devices:
            if src == dst:
                continue
            can = bool(torch.cuda.can_device_access_peer(src, dst))
            rows.append({"src": src, "dst": dst, "can_access": can})
            if not can:
                missing.append(f"{src}->{dst}")
    if missing:
        print(
            "FATAL no peer access (and therefore no NVLink) on pairs: "
            + ", ".join(missing)
            + "; the study aborts before any timed lane",
            file=sys.stderr,
        )
        sys.exit(EXIT_NO_PEER_ACCESS)
    return rows


def run_lane_p1(
    stage: int,
    devices: list[int],
    mock: bool,
    restore_device: int = 0,
    methods: tuple[str, ...] = (METHOD_EAGER,),
) -> list[dict[str, Any]]:
    """Time the peer copy cells of lane P1 on rank 0.

    The lane touches every visible device, so it restores the process device
    on the way out even if a cell raises. ``restore_device`` is this rank's own
    device, the one every later communicator must keep using.
    """

    pool = BufferPool(mock)
    rows: list[dict[str, Any]] = []
    try:
        for requested in _sweep(mock):
            for src, dst in ((0, 1), (1, 0)):
                rows.extend(
                    _copy_cell_rows(
                        "unidirectional", [(src, dst)], requested, pool, mock, methods
                    )
                )
            for row in _copy_cell_rows(
                "bidirectional", [(0, 1), (1, 0)], requested, pool, mock, methods
            ):
                row["direction"] = "bidirectional"
                rows.append(row)

        if stage == 2 and len(devices) >= 8:
            rows.extend(_lane_p1_stage2(devices, pool, mock, methods))
    finally:
        pool.release()
        if not mock and torch.cuda.is_available():
            torch.cuda.set_device(restore_device)
    return rows


def _lane_p1_stage2(
    devices: list[int],
    pool: BufferPool,
    mock: bool,
    methods: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for requested in STAGE2_ALL_PAIRS_PAYLOADS:
        for src in devices:
            for dst in devices:
                if src == dst:
                    continue
                rows.extend(
                    _copy_cell_rows("all_pairs", [(src, dst)], requested, pool, mock, methods)
                )

    requested = STAGE2_CONCURRENT_PAYLOAD
    disjoint = [(0, 1), (2, 3), (4, 5), (6, 7)]
    for row in _copy_cell_rows("disjoint_pairs", disjoint, requested, pool, mock, methods):
        row["direction"] = "four disjoint pairs"
        rows.append(row)

    fanout = [(0, peer) for peer in devices if peer != 0]
    for row in _copy_cell_rows("fanout", fanout, requested, pool, mock, methods):
        row["direction"] = "0->all"
        rows.append(row)

    for donors in STAGE2_FANIN_DONORS:
        fanin = [(donor, 0) for donor in devices[1 : donors + 1]]
        for row in _copy_cell_rows("fanin", fanin, requested, pool, mock, methods):
            row["direction"] = f"{donors}->0"
            row["donors"] = donors
            rows.append(row)
    return rows


def _max_across(value: float, device: torch.device, group: Any = None) -> float:
    """Return the maximum of ``value`` over the ranks of ``group``."""

    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    return float(tensor.item())


def _all_agree(value: bool, device: torch.device, group: Any = None) -> bool:
    """Return whether every rank of ``group`` reports ``True``."""

    tensor = torch.tensor([1.0 if value else 0.0], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=group)
    return bool(tensor.item() > 0.5)


def probe_graph_support(mock: bool, device: torch.device) -> tuple[bool, str]:
    """Decide once, for every rank together, whether NCCL rows can be captured.

    Capture and replay have to happen in lockstep: a rank that replayed a graph
    while another issued eagerly would leave the communicator mismatched. One
    agreed decision for the whole run removes that risk, and a container that
    cannot capture simply records eager rows and says so.
    """

    if not _graph_supported(mock):
        return False, "mock mode or no CUDA graph support in this build"
    probe = torch.zeros(GRAPH_PROBE_ELEMENTS, dtype=torch.float32, device=device)

    def issue() -> None:
        dist.all_reduce(probe)

    reason = ""
    captured = True
    try:
        for _ in range(GRAPH_WARMUP_ITERATIONS):
            issue()
        torch.cuda.synchronize(device)
        graph = _capture(issue, 2)
        graph.replay()
        torch.cuda.synchronize(device)
        del graph
    except Exception as error:  # noqa: BLE001 - any capture failure disables the method
        captured = False
        reason = f"{type(error).__name__}: {error}"
        torch.cuda.synchronize(device)
    agreed = _all_agree(captured, device)
    if not agreed and not reason:
        reason = "another rank could not capture a CUDA graph"
    return agreed, reason


def _time_block(
    mock: bool,
    device: torch.device,
    timed: int,
    issue: Any,
    method: str = METHOD_EAGER,
) -> tuple[float, str]:
    """Time ``timed`` issues of ``issue`` and return ``(seconds, reason)``.

    The eager path issues every iteration from Python between the CUDA events,
    which is what the amendment keeps as a control. The graph path captures the
    iterations once and replays them inside the same bracket, so no dispatch
    sits between them.
    """

    if mock:
        if method == METHOD_GRAPH:
            return 0.0, "mock mode runs on CPU over gloo, which cannot be captured"
        start = time.perf_counter_ns()
        for _ in range(timed):
            issue()
        return (time.perf_counter_ns() - start) * 1e-9 / timed, ""

    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if method == METHOD_GRAPH:
        try:
            side = torch.cuda.Stream(device=device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                for _ in range(GRAPH_WARMUP_ITERATIONS):
                    issue()
            torch.cuda.current_stream(device).wait_stream(side)
            torch.cuda.synchronize(device)
            graph = _capture(issue, timed)
            for _ in range(UNTIMED_REPLAYS):
                graph.replay()
                torch.cuda.synchronize(device)
        except Exception as error:  # noqa: BLE001 - fall back to the eager row
            torch.cuda.synchronize(device)
            return 0.0, f"{type(error).__name__}: {error}"
        begin.record(torch.cuda.current_stream(device))
        graph.replay()
        end.record(torch.cuda.current_stream(device))
        torch.cuda.synchronize(device)
        del graph
        return begin.elapsed_time(end) * 1e-3 / timed, ""

    begin.record(torch.cuda.current_stream(device))
    for _ in range(timed):
        issue()
    end.record(torch.cuda.current_stream(device))
    torch.cuda.synchronize(device)
    return begin.elapsed_time(end) * 1e-3 / timed, ""


def _p2_pair_cell(
    src: int,
    dst: int,
    requested: int,
    mock: bool,
    rank: int,
    device: torch.device,
    method: str,
) -> dict[str, Any] | None:
    """Time one unidirectional point-to-point payload and its 8-byte reply.

    Both ranks of the pair run the identical block, so a captured graph stays
    in lockstep; only the source rank reports the row. Under the eager method
    the reply is the freeze's host-side completion token. Under the graph
    method the same exchange is captured, so the reply still proves completion
    but costs no host round trip per iteration.
    """

    moved = _capped(requested, mock)
    warmup, timed = _iterations(requested, mock)
    if rank not in (src, dst):
        return None
    payload = torch.empty(moved, dtype=torch.uint8, device=device)
    token = torch.empty(TOKEN_BYTES, dtype=torch.uint8, device=device)

    def issue() -> None:
        if rank == src:
            send = dist.isend(payload, dst)
            recv = dist.irecv(token, dst)
            send.wait()
            recv.wait()
        else:
            recv = dist.irecv(payload, src)
            send = dist.isend(token, src)
            recv.wait()
            send.wait()

    for _ in range(warmup):
        issue()
    if not mock:
        torch.cuda.synchronize(device)
    seconds, reason = _time_block(mock, device, timed, issue, method)
    if reason or rank != src:
        return None
    return {
        "cell": "unidirectional",
        "method": method,
        "of_record": _of_record(method, requested),
        "src": src,
        "dst": dst,
        "direction": f"{src}->{dst}",
        "requested_bytes": requested,
        "bytes": moved,
        "iterations": timed,
        "status": "measured",
        "time_ns": seconds * 1e9,
        "bytes_per_second": _rate(moved, seconds),
        "token_bytes": TOKEN_BYTES,
    }


def _p2_bidirectional_cell(
    src: int,
    dst: int,
    requested: int,
    mock: bool,
    rank: int,
    device: torch.device,
    method: str,
) -> dict[str, Any] | None:
    """Time one bidirectional exchange issued with ``batch_isend_irecv``."""

    moved = _capped(requested, mock)
    warmup, timed = _iterations(requested, mock)
    if rank not in (src, dst):
        return None
    peer = dst if rank == src else src
    send_buffer = torch.empty(moved, dtype=torch.uint8, device=device)
    recv_buffer = torch.empty(moved, dtype=torch.uint8, device=device)

    def issue() -> None:
        ops = [
            dist.P2POp(dist.isend, send_buffer, peer),
            dist.P2POp(dist.irecv, recv_buffer, peer),
        ]
        for work in dist.batch_isend_irecv(ops):
            work.wait()

    for _ in range(warmup):
        issue()
    if not mock:
        torch.cuda.synchronize(device)
    seconds, reason = _time_block(mock, device, timed, issue, method)
    if reason or rank != src:
        return None
    return {
        "cell": "bidirectional",
        "method": method,
        "of_record": _of_record(method, requested),
        "src": src,
        "dst": dst,
        "direction": "bidirectional",
        "requested_bytes": requested,
        "bytes": moved,
        "iterations": timed,
        "status": "measured",
        "time_ns": seconds * 1e9,
        "bytes_per_second": _rate(moved, seconds),
        "aggregate_bytes": 2 * moved,
        "aggregate_bytes_per_second": _rate(2 * moved, seconds),
    }


def run_lane_p2(
    stage: int,
    mock: bool,
    rank: int,
    world_size: int,
    device: torch.device,
    methods: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Time the NCCL point-to-point cells of lane P2."""

    rows: list[dict[str, Any]] = []
    if world_size < 2:
        return rows
    for requested in _sweep(mock):
        for method in methods:
            for row in (
                _p2_pair_cell(0, 1, requested, mock, rank, device, method),
                _p2_bidirectional_cell(0, 1, requested, mock, rank, device, method),
            ):
                if row is not None:
                    rows.append(row)
    if stage == 2:
        for src, dst in STAGE2_P2_PAIRS:
            if max(src, dst) >= world_size:
                continue
            for requested in STAGE2_P2_PAYLOADS:
                for method in methods:
                    # One pair at a time, so a placement cell measures an
                    # otherwise idle fabric.
                    dist.barrier()
                    row = _p2_pair_cell(src, dst, requested, mock, rank, device, method)
                    if row is not None:
                        row["cell"] = "stage2_pair"
                        rows.append(row)
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, rows)
    merged: list[dict[str, Any]] = []
    for part in gathered:
        merged.extend(part or [])
    return merged


def _collective_elements(op: str, nbytes: int, width: int) -> tuple[int, int] | None:
    """Return ``(input_elements, output_elements)`` or ``None`` when skipped."""

    if nbytes % FLOAT32_BYTES:
        return None
    elements = nbytes // FLOAT32_BYTES
    if op == "all_reduce":
        return elements, elements
    if elements % width:
        return None
    if op == "all_gather":
        return elements // width, elements
    return elements, elements // width


def _run_collective_point(
    op: str,
    width: int,
    requested: int,
    mock: bool,
    rank: int,
    device: torch.device,
    group: Any,
    send: torch.Tensor,
    recv: torch.Tensor,
    method: str,
) -> dict[str, Any]:
    moved = _capped(requested, mock)
    warmup, timed = _iterations(requested, mock)
    shape = _collective_elements(op, moved, width)
    if shape is None:
        return {
            "op": op,
            "width": width,
            "method": method,
            "of_record": _of_record(method, requested),
            "requested_bytes": requested,
            "bytes": moved,
            "status": "skipped_not_divisible",
        }
    input_elements, output_elements = shape
    source = send[:input_elements]
    target = recv[:output_elements]

    def issue() -> None:
        if op == "all_reduce":
            dist.all_reduce(target, op=dist.ReduceOp.SUM, group=group)
        elif op == "all_gather":
            dist.all_gather_into_tensor(target, source, group=group)
        else:
            dist.reduce_scatter_tensor(target, source, op=dist.ReduceOp.SUM, group=group)

    correct: bool | None = None
    if op == "all_reduce":
        target.fill_(float(rank + 1))
        dist.all_reduce(target, op=dist.ReduceOp.SUM, group=group)
        expected = float(width * (width + 1) // 2)
        probes = (0, output_elements // 2, output_elements - 1)
        correct = all(float(target[index].item()) == expected for index in probes)
        target.zero_()

    for _ in range(warmup):
        issue()
    if not mock:
        torch.cuda.synchronize(device)
    dist.barrier(group=group)
    seconds, reason = _time_block(mock, device, timed, issue, method)
    if reason:
        return {
            "op": op,
            "width": width,
            "method": method,
            "of_record": _of_record(method, requested),
            "requested_bytes": requested,
            "bytes": moved,
            "status": "graph_skipped",
            "graph_skipped": True,
            "graph_skip_reason": reason,
        }
    seconds = _max_across(seconds, device, group=group)

    factor = busbw_factor(op, width)
    algbw = _rate(moved, seconds)
    return {
        "op": op,
        "width": width,
        "method": method,
        "of_record": _of_record(method, requested),
        "requested_bytes": requested,
        "bytes": moved,
        "iterations": timed,
        "status": "measured",
        "time_ns": seconds * 1e9,
        "bytes_per_second": algbw,
        "algbw_bytes_per_second": algbw,
        "busbw_bytes_per_second": algbw * factor,
        "busbw_factor": factor,
        "all_reduce_correct": correct,
    }


def run_lane_p3(
    stage: int,
    mock: bool,
    rank: int,
    world_size: int,
    device: torch.device,
    methods: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Time the NCCL collectives of lane P3 at the frozen widths."""

    rows: list[dict[str, Any]] = []
    widths = [width for width in WIDTHS[stage] if width <= world_size]
    # Every rank builds every sub-communicator, including the ranks a group
    # leaves out, because new_group is itself collective over the world.
    groups = {width: dist.new_group(ranks=list(range(width))) for width in widths}
    sweep = _sweep(mock)
    max_bytes = _capped(max(sweep), mock)
    send = torch.empty(max_bytes // FLOAT32_BYTES, dtype=torch.float32, device=device)
    send.fill_(float(rank + 1))
    recv = torch.zeros(max_bytes // FLOAT32_BYTES, dtype=torch.float32, device=device)

    for width in widths:
        group = groups[width]
        for op in COLLECTIVE_OPS:
            for requested in sweep:
                for method in methods:
                    if rank < width:
                        rows.append(
                            _run_collective_point(
                                op,
                                width,
                                requested,
                                mock,
                                rank,
                                device,
                                group,
                                send,
                                recv,
                                method,
                            )
                        )
            dist.barrier()
    del send, recv
    if not mock and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def build_header(
    mock: bool,
    world_size: int,
    started: str,
    graph_enabled: bool = False,
    graph_reason: str = "",
) -> dict[str, Any]:
    """Return the header the freeze requires, with the hostname redacted."""

    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    devices: list[dict[str, Any]] = []
    if device_count:
        bus_ids = _pci_bus_ids(device_count)
        for index in range(device_count):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "ordinal": index,
                    "name": properties.name,
                    "pci_bus_id": bus_ids[index],
                    "multi_processor_count": properties.multi_processor_count,
                    "total_memory_bytes": properties.total_memory,
                }
            )
    return {
        "hostname": "redacted",
        "started_utc": started,
        "finished_utc": None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "nccl_version": _nccl_version(),
        "driver_version": _driver_version() if device_count else "unavailable",
        "visible_device_count": device_count,
        "world_size": world_size,
        "backend": "gloo" if mock else "nccl",
        "mock": mock,
        "mock_payload_cap_bytes": MOCK_PAYLOAD_CAP_BYTES if mock else None,
        "mock_timed_iterations": MOCK_TIMED_ITERATIONS if mock else None,
        "amendment": "expectations-amendment-2026-09-15",
        "graph_capture_enabled": graph_enabled,
        "graph_capture_reason": graph_reason,
        "graph_row_max_bytes": GRAPH_ROW_MAX_BYTES,
        "untimed_replays": UNTIMED_REPLAYS,
        "timed_iterations_at_or_below_1mib": (
            MOCK_TIMED_ITERATIONS if mock else TIMED_ITERATIONS_AT_OR_BELOW_1MIB
        ),
        "methods": list(METHODS if graph_enabled else (METHOD_EAGER,)),
        "devices": devices,
        "payload_bytes": list(_sweep(mock)),
        "frozen_payload_bytes": list(PAYLOAD_BYTES),
        "warmup_iterations": MOCK_WARMUP_ITERATIONS if mock else WARMUP_ITERATIONS,
        "timed_iterations": MOCK_TIMED_ITERATIONS if mock else TIMED_ITERATIONS,
        "timed_iterations_above_64mib": (
            MOCK_TIMED_ITERATIONS if mock else TIMED_ITERATIONS_ABOVE_64MIB
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B200 NVLink envelope benchmark lanes")
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="exercise the control flow on CPU tensors over gloo with capped payloads",
    )
    parser.add_argument(
        "--pg-timeout-seconds",
        type=int,
        default=PG_TIMEOUT_SECONDS,
        help="process group timeout; a stuck collective raises instead of spinning",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    started = _utc_now()

    if not args.mock:
        if not torch.cuda.is_available():
            print("FATAL no CUDA device visible; use --mock for a flow check", file=sys.stderr)
            return EXIT_TOO_FEW_DEVICES
        if torch.cuda.device_count() < 2:
            print(
                "FATAL the study needs at least two visible devices, found "
                f"{torch.cuda.device_count()}",
                file=sys.stderr,
            )
            return EXIT_TOO_FEW_DEVICES
        torch.cuda.set_device(local_rank)

    if args.mock:
        device_count = MOCK_DEVICE_COUNT[args.stage]
    else:
        device_count = torch.cuda.device_count()
    devices = list(range(device_count))
    peer_access = check_peer_access(devices, args.mock)
    if rank == 0:
        _progress(f"peer access verified on {len(devices)} devices")

    _init_process_group(args.mock, local_rank, args.pg_timeout_seconds)
    device = torch.device("cpu") if args.mock else torch.device("cuda", local_rank)
    if rank == 0:
        _progress(
            f"process group ready, backend {'gloo' if args.mock else 'nccl'}, "
            f"timeout {args.pg_timeout_seconds} s"
        )

    graph_enabled, graph_reason = probe_graph_support(args.mock, device)
    methods = METHODS if graph_enabled else (METHOD_EAGER,)
    if rank == 0:
        _progress(
            f"graph capture {'enabled' if graph_enabled else 'disabled'}"
            + (f": {graph_reason}" if graph_reason else "")
        )

    p1_rows: list[dict[str, Any]] = []
    if rank == 0:
        _progress("P1 start")
        started_lane = time.perf_counter()
        p1_rows = run_lane_p1(args.stage, devices, args.mock, local_rank, methods)
        _progress(f"P1 end, {len(p1_rows)} rows in {time.perf_counter() - started_lane:.1f} s")
    if not args.mock:
        # Belt and braces: no lane may leave this process on another device.
        torch.cuda.set_device(local_rank)
    dist.barrier()

    if rank == 0:
        _progress("P2 start")
        started_lane = time.perf_counter()
    p2_rows = run_lane_p2(args.stage, args.mock, rank, world_size, device, methods)
    if rank == 0:
        _progress(f"P2 end, {len(p2_rows)} rows in {time.perf_counter() - started_lane:.1f} s")
    dist.barrier()

    if rank == 0:
        _progress("P3 start")
        started_lane = time.perf_counter()
    p3_rows = run_lane_p3(args.stage, args.mock, rank, world_size, device, methods)
    if rank == 0:
        _progress(f"P3 end, {len(p3_rows)} rows in {time.perf_counter() - started_lane:.1f} s")
    dist.barrier()

    if not graph_enabled:
        # No row of record exists at or below 1 MiB when capture is off, so
        # every row says on its face that it is the eager control.
        for lane_rows in (p1_rows, p2_rows, p3_rows):
            for row in lane_rows:
                row["graph_skipped"] = True
                row.setdefault(
                    "graph_skip_reason", graph_reason or "graph rows disabled for this run"
                )

    if rank == 0:
        header = build_header(args.mock, world_size, started, graph_enabled, graph_reason)
        header["finished_utc"] = _utc_now()
        result = {
            "schema": SCHEMA,
            "study": STUDY,
            "stage": args.stage,
            "header": header,
            "peer_access": peer_access,
            "p1": p1_rows,
            "p2": p2_rows,
            "p3": p3_rows,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / f"stage{args.stage}_result.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"wrote {path}")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
