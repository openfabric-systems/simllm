"""NCCL-channel kernel builders over the GPU service model.

An NCCL collective on one GPU is a kernel like any other: its channels are
CTAs resident on SMs, its warps issue loads from HBM and stores onto the
NVLink egress serializer, and it competes with compute and memory kernels
for scheduler slots, pipelines and cursors. The builders here emit exactly
that shape so `SmSchedulerModel.estimate_concurrent` can price a network
task next to compute and memory tasks with no special cases.

The v1 builder models the egress half of a ring all-reduce: every chunk is
loaded from HBM and stored to NVLink in program order with double-buffered
registers, across as many channel CTAs and per-channel warps as the
caller declares. Ingress service, reduction lanes, per-peer link
topology and proxy operations are deliberately absent and are tracked as
COMP-11 in docs/modules/compute.md.
"""

from __future__ import annotations

from simllm.compute.gpu_model import (
    CtaTrace,
    GpuTask,
    GpuTaskKind,
    KernelLaunch,
    MemorySpace,
    PipelineKind,
    SassInstruction,
    SassWarpTrace,
)

NCCL_RING_EGRESS_IMPLEMENTATION = "simllm-nccl-ring-egress-v1"


def nccl_ring_allreduce_launch(
    *,
    payload_bytes: int,
    world_size: int,
    channels: int,
    chunk_bytes: int,
    warps_per_channel: int = 4,
    registers_per_thread: int = 32,
    implementation_id: str | None = None,
    trace_id: str | None = None,
) -> KernelLaunch:
    """Return the per-GPU egress kernel of one ring all-reduce.

    A ring all-reduce over ``world_size`` ranks runs ``2 * (world_size - 1)``
    steps and every rank sends ``payload_bytes / world_size`` bytes per step,
    split evenly over ``channels`` channel CTAs and the warps inside them, in
    ``chunk_bytes`` chunks. Divisibility is enforced so the emitted trace
    carries exactly the closed form
    ``2 * (world_size - 1) * payload_bytes / world_size`` egress bytes.

    Each warp loads a chunk from HBM and stores it onto the NVLink egress
    path, double buffering so a warp's next load does not wait on its own
    previous store data. Warps are the concurrency that lets the link
    saturate: one warp per channel is latency bound by the load-to-store
    chain, which is why real NCCL channels run hundreds of threads.
    """

    _require_positive("payload_bytes", payload_bytes)
    _require_positive("world_size", world_size)
    _require_positive("chunk_bytes", chunk_bytes)
    _require_positive("channels", channels)
    _require_positive("warps_per_channel", warps_per_channel)
    _require_positive("registers_per_thread", registers_per_thread)
    if registers_per_thread < 2:
        raise ValueError("registers_per_thread must be at least 2 for double buffering")
    if world_size < 2:
        raise ValueError("world_size must be at least 2 for a ring all-reduce")
    lanes = world_size * channels * warps_per_channel
    step_bytes, step_remainder = divmod(payload_bytes, lanes)
    if step_remainder:
        raise ValueError(
            "payload_bytes must divide evenly over world_size * channels * "
            f"warps_per_channel; {payload_bytes} bytes over {lanes} shares "
            f"leaves {step_remainder}"
        )
    chunks_per_step, chunk_remainder = divmod(step_bytes, chunk_bytes)
    if chunk_remainder:
        raise ValueError(
            "per-warp step bytes must divide evenly into chunk_bytes; "
            f"{step_bytes} bytes into {chunk_bytes}-byte chunks leaves "
            f"{chunk_remainder}"
        )
    if chunks_per_step == 0:
        raise ValueError("chunk_bytes exceeds the per-warp step share")

    def load_chunk(buffer_register: str) -> SassInstruction:
        return SassInstruction(
            opcode="LDG",
            pipeline=PipelineKind.LOAD_STORE,
            memory_space=MemorySpace.HBM,
            requested_bytes=chunk_bytes,
            transacted_bytes=chunk_bytes,
            destination_registers=(buffer_register,),
        )

    def store_chunk(buffer_register: str) -> SassInstruction:
        return SassInstruction(
            opcode="STG",
            pipeline=PipelineKind.LOAD_STORE,
            memory_space=MemorySpace.NVLINK,
            requested_bytes=chunk_bytes,
            transacted_bytes=chunk_bytes,
            source_registers=(buffer_register,),
        )

    steps = 2 * (world_size - 1)
    chunk_count = steps * chunks_per_step
    instructions = [load_chunk(f"r{index}") for index in range(min(2, chunk_count))]
    for chunk_index in range(chunk_count):
        buffer_register = f"r{chunk_index % 2}"
        instructions.append(store_chunk(buffer_register))
        next_chunk = chunk_index + 2
        if next_chunk < chunk_count:
            instructions.append(load_chunk(buffer_register))

    name = (
        f"{NCCL_RING_EGRESS_IMPLEMENTATION}-{payload_bytes}b-w{world_size}-c{channels}-"
        f"t{warps_per_channel}-k{chunk_bytes}-r{registers_per_thread}"
    )
    warp_instructions = tuple(instructions)
    return KernelLaunch(
        implementation_id=implementation_id or name,
        trace_id=trace_id or f"{name}-trace",
        grid_blocks=channels,
        threads_per_block=32 * warps_per_channel,
        registers_per_thread=registers_per_thread,
        static_shared_memory_bytes=0,
        dynamic_shared_memory_bytes=0,
        cta_traces=(
            CtaTrace(
                trace_class_id=f"{name}-channel",
                block_ids=tuple(range(channels)),
                warp_traces=tuple(
                    SassWarpTrace(warp_id=warp, instructions=warp_instructions)
                    for warp in range(warps_per_channel)
                ),
            ),
        ),
    )


def nccl_ring_allreduce_task(
    *,
    task_id: str,
    payload_bytes: int,
    world_size: int,
    channels: int,
    chunk_bytes: int,
    warps_per_channel: int = 4,
    registers_per_thread: int = 32,
) -> GpuTask:
    """Wrap the egress kernel as a network task for concurrent scheduling."""

    return GpuTask(
        task_id=task_id,
        kind=GpuTaskKind.NETWORK,
        launch=nccl_ring_allreduce_launch(
            payload_bytes=payload_bytes,
            world_size=world_size,
            channels=channels,
            chunk_bytes=chunk_bytes,
            warps_per_channel=warps_per_channel,
            registers_per_thread=registers_per_thread,
        ),
    )


def nccl_ring_egress_bytes(*, payload_bytes: int, world_size: int) -> int:
    """Return the exact per-GPU egress byte count of one ring all-reduce."""

    _require_positive("payload_bytes", payload_bytes)
    _require_positive("world_size", world_size)
    if world_size < 2:
        raise ValueError("world_size must be at least 2 for a ring all-reduce")
    per_rank_bytes, remainder = divmod(payload_bytes, world_size)
    if remainder:
        raise ValueError(
            "payload_bytes must divide evenly over world_size before computing "
            "an exact per-GPU egress byte count"
        )
    return 2 * (world_size - 1) * per_rank_bytes


def _require_positive(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "NCCL_RING_EGRESS_IMPLEMENTATION",
    "nccl_ring_allreduce_launch",
    "nccl_ring_allreduce_task",
    "nccl_ring_egress_bytes",
]
