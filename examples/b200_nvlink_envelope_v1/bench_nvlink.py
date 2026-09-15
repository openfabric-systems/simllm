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
    """Return ``(warmup, timed)`` for one payload under the frozen rule."""

    if mock:
        return MOCK_WARMUP_ITERATIONS, MOCK_TIMED_ITERATIONS
    if nbytes > LARGE_PAYLOAD_BYTES:
        return WARMUP_ITERATIONS, TIMED_ITERATIONS_ABOVE_64MIB
    return WARMUP_ITERATIONS, TIMED_ITERATIONS


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


def _time_transfers(
    pairs: list[tuple[int, int]],
    nbytes: int,
    warmup: int,
    timed: int,
    pool: BufferPool,
    mock: bool,
) -> list[float]:
    """Time one group of simultaneous copies and return per-pair seconds.

    Each copy runs on its own stream on its source device, which is the stream
    PyTorch issues a cross-device ``copy_`` on, so the events bracket the
    issuing stream. The makespan of the group is the maximum of the returned
    seconds.
    """

    sources = [
        pool.get(src, f"src{src}", nbytes) for src, _ in pairs
    ]
    destinations = [
        pool.get(dst, f"dst{index}", nbytes) for index, (_, dst) in enumerate(pairs)
    ]

    if mock:
        for _ in range(warmup):
            for source, destination in zip(sources, destinations):
                destination.copy_(source)
        start = time.perf_counter_ns()
        for _ in range(timed):
            for source, destination in zip(sources, destinations):
                destination.copy_(source)
        elapsed = (time.perf_counter_ns() - start) * 1e-9 / timed
        return [elapsed] * len(pairs)

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

    for index, (src, _) in enumerate(pairs):
        with torch.cuda.device(src):
            starts[index].record(streams[index])
    for _ in range(timed):
        for index, (src, _) in enumerate(pairs):
            with torch.cuda.device(src), torch.cuda.stream(streams[index]):
                destinations[index].copy_(sources[index], non_blocking=True)
    for index, (src, _) in enumerate(pairs):
        with torch.cuda.device(src):
            stops[index].record(streams[index])
    for index in devices:
        torch.cuda.synchronize(index)

    return [
        starts[index].elapsed_time(stops[index]) * 1e-3 / timed for index in range(len(pairs))
    ]


def _copy_row(
    cell: str,
    pairs: list[tuple[int, int]],
    requested_bytes: int,
    moved_bytes: int,
    timed: int,
    seconds: list[float],
) -> dict[str, Any]:
    makespan = max(seconds)
    aggregate = moved_bytes * len(pairs)
    row: dict[str, Any] = {
        "cell": cell,
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
            moved = _capped(requested, mock)
            warmup, timed = _iterations(requested, mock)
            for src, dst in ((0, 1), (1, 0)):
                seconds = _time_transfers([(src, dst)], moved, warmup, timed, pool, mock)
                rows.append(
                    _copy_row("unidirectional", [(src, dst)], requested, moved, timed, seconds)
                )
            seconds = _time_transfers([(0, 1), (1, 0)], moved, warmup, timed, pool, mock)
            row = _copy_row("bidirectional", [(0, 1), (1, 0)], requested, moved, timed, seconds)
            row["direction"] = "bidirectional"
            if mock:
                row["concurrency"] = "sequential_mock"
            rows.append(row)

        if stage == 2 and len(devices) >= 8:
            rows.extend(_lane_p1_stage2(devices, pool, mock))
    finally:
        pool.release()
        if not mock and torch.cuda.is_available():
            torch.cuda.set_device(restore_device)
    return rows


def _lane_p1_stage2(
    devices: list[int],
    pool: BufferPool,
    mock: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for requested in STAGE2_ALL_PAIRS_PAYLOADS:
        moved = _capped(requested, mock)
        warmup, timed = _iterations(requested, mock)
        for src in devices:
            for dst in devices:
                if src == dst:
                    continue
                seconds = _time_transfers([(src, dst)], moved, warmup, timed, pool, mock)
                rows.append(
                    _copy_row("all_pairs", [(src, dst)], requested, moved, timed, seconds)
                )

    requested = STAGE2_CONCURRENT_PAYLOAD
    moved = _capped(requested, mock)
    warmup, timed = _iterations(requested, mock)

    disjoint = [(0, 1), (2, 3), (4, 5), (6, 7)]
    seconds = _time_transfers(disjoint, moved, warmup, timed, pool, mock)
    row = _copy_row("disjoint_pairs", disjoint, requested, moved, timed, seconds)
    row["direction"] = "four disjoint pairs"
    rows.append(row)

    fanout = [(0, peer) for peer in devices if peer != 0]
    seconds = _time_transfers(fanout, moved, warmup, timed, pool, mock)
    row = _copy_row("fanout", fanout, requested, moved, timed, seconds)
    row["direction"] = "0->all"
    rows.append(row)

    for donors in STAGE2_FANIN_DONORS:
        fanin = [(donor, 0) for donor in devices[1 : donors + 1]]
        seconds = _time_transfers(fanin, moved, warmup, timed, pool, mock)
        row = _copy_row("fanin", fanin, requested, moved, timed, seconds)
        row["direction"] = f"{donors}->0"
        row["donors"] = donors
        rows.append(row)
    return rows


def _max_across(value: float, device: torch.device, group: Any = None) -> float:
    """Return the maximum of ``value`` over the ranks of ``group``."""

    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    return float(tensor.item())


def _time_block(mock: bool, device: torch.device, timed: int, issue: Any) -> float:
    """Time ``timed`` issues of ``issue`` on the issuing stream."""

    if mock:
        start = time.perf_counter_ns()
        for _ in range(timed):
            issue()
        return (time.perf_counter_ns() - start) * 1e-9 / timed
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record(torch.cuda.current_stream(device))
    for _ in range(timed):
        issue()
    end.record(torch.cuda.current_stream(device))
    torch.cuda.synchronize(device)
    return begin.elapsed_time(end) * 1e-3 / timed


def _p2_pair_cell(
    src: int,
    dst: int,
    requested: int,
    mock: bool,
    rank: int,
    device: torch.device,
) -> dict[str, Any] | None:
    """Time one unidirectional point-to-point payload with its 8-byte token.

    Only the two ranks of the pair transfer anything. The source rank times the
    block and returns the row; every other rank returns ``None`` and the rows
    are gathered once at the end of the lane.
    """

    moved = _capped(requested, mock)
    warmup, timed = _iterations(requested, mock)
    payload = None
    token = None
    if rank in (src, dst):
        payload = torch.empty(moved, dtype=torch.uint8, device=device)
        token = torch.empty(TOKEN_BYTES, dtype=torch.uint8, device=device)

    def issue() -> None:
        if rank == src:
            send = dist.isend(payload, dst)
            recv = dist.irecv(token, dst)
            send.wait()
            recv.wait()
        elif rank == dst:
            recv = dist.irecv(payload, src)
            send = dist.isend(token, src)
            recv.wait()
            send.wait()

    if rank not in (src, dst):
        return None
    for _ in range(warmup):
        issue()
    if not mock:
        torch.cuda.synchronize(device)
    if rank == dst:
        for _ in range(timed):
            issue()
        if not mock:
            torch.cuda.synchronize(device)
        return None
    seconds = _time_block(mock, device, timed, issue)
    return {
        "cell": "unidirectional",
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
) -> dict[str, Any] | None:
    """Time one bidirectional exchange issued with ``batch_isend_irecv``."""

    moved = _capped(requested, mock)
    warmup, timed = _iterations(requested, mock)
    send_buffer = None
    recv_buffer = None
    peer = dst if rank == src else src
    if rank in (src, dst):
        send_buffer = torch.empty(moved, dtype=torch.uint8, device=device)
        recv_buffer = torch.empty(moved, dtype=torch.uint8, device=device)

    def issue() -> None:
        if rank not in (src, dst):
            return
        ops = [
            dist.P2POp(dist.isend, send_buffer, peer),
            dist.P2POp(dist.irecv, recv_buffer, peer),
        ]
        for work in dist.batch_isend_irecv(ops):
            work.wait()

    if rank not in (src, dst):
        return None
    for _ in range(warmup):
        issue()
    if not mock:
        torch.cuda.synchronize(device)
    if rank == dst:
        for _ in range(timed):
            issue()
        if not mock:
            torch.cuda.synchronize(device)
        return None
    seconds = _time_block(mock, device, timed, issue)
    return {
        "cell": "bidirectional",
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
) -> list[dict[str, Any]]:
    """Time the NCCL point-to-point cells of lane P2."""

    rows: list[dict[str, Any]] = []
    if world_size < 2:
        return rows
    for requested in _sweep(mock):
        for row in (
            _p2_pair_cell(0, 1, requested, mock, rank, device),
            _p2_bidirectional_cell(0, 1, requested, mock, rank, device),
        ):
            if row is not None:
                rows.append(row)
    if stage == 2:
        for src, dst in STAGE2_P2_PAIRS:
            if max(src, dst) >= world_size:
                continue
            for requested in STAGE2_P2_PAYLOADS:
                # One pair at a time, so a placement cell measures an
                # otherwise idle fabric.
                dist.barrier()
                row = _p2_pair_cell(src, dst, requested, mock, rank, device)
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
) -> dict[str, Any]:
    moved = _capped(requested, mock)
    warmup, timed = _iterations(requested, mock)
    shape = _collective_elements(op, moved, width)
    if shape is None:
        return {
            "op": op,
            "width": width,
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
    seconds = _time_block(mock, device, timed, issue)
    seconds = _max_across(seconds, device, group=group)

    factor = busbw_factor(op, width)
    algbw = _rate(moved, seconds)
    return {
        "op": op,
        "width": width,
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
                if rank < width:
                    rows.append(
                        _run_collective_point(
                            op, width, requested, mock, rank, device, group, send, recv
                        )
                    )
            dist.barrier()
    del send, recv
    if not mock and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def build_header(mock: bool, world_size: int, started: str) -> dict[str, Any]:
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

    p1_rows: list[dict[str, Any]] = []
    if rank == 0:
        _progress("P1 start")
        started_lane = time.perf_counter()
        p1_rows = run_lane_p1(args.stage, devices, args.mock, local_rank)
        _progress(f"P1 end, {len(p1_rows)} rows in {time.perf_counter() - started_lane:.1f} s")
    if not args.mock:
        # Belt and braces: no lane may leave this process on another device.
        torch.cuda.set_device(local_rank)
    dist.barrier()

    if rank == 0:
        _progress("P2 start")
        started_lane = time.perf_counter()
    p2_rows = run_lane_p2(args.stage, args.mock, rank, world_size, device)
    if rank == 0:
        _progress(f"P2 end, {len(p2_rows)} rows in {time.perf_counter() - started_lane:.1f} s")
    dist.barrier()

    if rank == 0:
        _progress("P3 start")
        started_lane = time.perf_counter()
    p3_rows = run_lane_p3(args.stage, args.mock, rank, world_size, device)
    if rank == 0:
        _progress(f"P3 end, {len(p3_rows)} rows in {time.perf_counter() - started_lane:.1f} s")
    dist.barrier()

    if rank == 0:
        header = build_header(args.mock, world_size, started)
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
