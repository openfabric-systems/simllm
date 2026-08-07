#!/usr/bin/env python3
"""Run the frozen structural checks for the isolated GPU service model."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

from simllm.compute import (
    CopyDirection,
    CopyDirectionProfile,
    CopyEngineProfile,
    CopyEngineServiceModel,
    CopyTransfer,
    CtaTrace,
    GpuArchitectureProfile,
    GpuCalibrationProfile,
    GpuModelArtifact,
    GpuModelProvenance,
    KernelLaunch,
    MemoryHierarchyProfile,
    MemorySpace,
    PipelineKind,
    PipelineProfile,
    SassInstruction,
    SassWarpTrace,
    SmSchedulerModel,
    WarpSchedulerPolicy,
    a100_sxm_80gb_seed_profile,
    h100_sxm_80gb_seed_profile,
)

HERE = Path(__file__).resolve().parent

SM_WAVE_CASES = ((4, 1, 1_024), (4, 4, 256), (9, 1, 2_304), (9, 4, 768))
WARP_ISSUE_CASES = ((4, 1, 7), (4, 4, 4), (16, 1, 19), (16, 4, 7))
OCCUPANCY_CASES = ((128, 32, 16), (128, 128, 4), (256, 32, 8), (256, 128, 2))
HBM_CASES = ((4_096, 32, 228), (4_096, 64, 164), (8_192, 32, 356), (8_192, 64, 228))
COPY_CASES = ((4_096, 32, 148), (4_096, 64, 84), (8_192, 32, 276), (8_192, 64, 148))


def architecture(
    *,
    sm_count: int = 1,
    schedulers: int = 1,
    bandwidth_bytes_per_cycle: float = 64,
    shared_memory_per_sm: int = 65_536,
) -> GpuArchitectureProfile:
    provenance = GpuModelProvenance(
        source="synthetic structural fixture",
        version="1",
        gpu="synthetic-1ghz",
        created="2026-08-06",
    )
    calibration = GpuCalibrationProfile(
        calibration_id="synthetic-structural-calibration-v1",
        target_architecture_profile_id="synthetic-1ghz-v1",
        provenance=provenance,
        core_clock_hz=1_000_000_000,
        target_memory_clock_hz=1_000_000_000,
        pipelines=(
            PipelineProfile(
                kind=PipelineKind.ALU,
                opcodes=("ALU",),
                latency_cycles=4,
                issue_width_per_sm=4,
            ),
            PipelineProfile(
                kind=PipelineKind.LOAD_STORE,
                opcodes=("MEMORY",),
                latency_cycles=1,
                issue_width_per_sm=4,
            ),
        ),
        memory=MemoryHierarchyProfile(
            hbm_latency_cycles=100,
            hbm_bandwidth_bytes_per_cycle=bandwidth_bytes_per_cycle,
            l2_latency_cycles=20,
            l1_latency_cycles=10,
            shared_latency_cycles=5,
        ),
        copy_engines=(
            CopyEngineProfile(
                engine_id="ce0",
                clock_hz=1_000_000_000,
                direction_profiles=(
                    CopyDirectionProfile(
                        direction=CopyDirection.HOST_TO_DEVICE,
                        setup_cycles=20,
                        bandwidth_bytes_per_cycle=bandwidth_bytes_per_cycle,
                    ),
                ),
            ),
        ),
        warp_scheduler_policy=WarpSchedulerPolicy.LOOSE_ROUND_ROBIN,
        relative_uncertainty=0.0,
    )
    return GpuArchitectureProfile(
        profile_id="synthetic-1ghz-v1",
        gpu_name="synthetic-1ghz",
        aliases=("synthetic",),
        sm_count=sm_count,
        warp_size=32,
        scheduler_count_per_sm=schedulers,
        dispatch_width_per_scheduler=1,
        max_blocks_per_sm=16,
        max_warps_per_sm=64,
        max_threads_per_sm=2_048,
        max_threads_per_block=1_024,
        registers_per_sm=65_536,
        max_registers_per_thread=255,
        register_allocation_granularity_per_warp=1,
        shared_memory_per_sm=shared_memory_per_sm,
        max_static_shared_memory_per_block=min(shared_memory_per_sm, 48 * 1_024),
        max_shared_memory_per_block=shared_memory_per_sm,
        shared_memory_allocation_granularity=1,
        calibration=calibration,
    )


def traces(warps: int, instruction: SassInstruction) -> tuple[SassWarpTrace, ...]:
    return tuple(
        SassWarpTrace(warp_id=warp_id, instructions=(instruction,)) for warp_id in range(warps)
    )


def launch(
    *,
    blocks: int,
    threads: int,
    warp_traces: tuple[SassWarpTrace, ...],
    registers_per_thread: int = 0,
    shared_memory_bytes: int = 0,
) -> KernelLaunch:
    return KernelLaunch(
        implementation_id="synthetic-implementation",
        trace_id="synthetic-trace",
        grid_blocks=blocks,
        threads_per_block=threads,
        registers_per_thread=registers_per_thread,
        static_shared_memory_bytes=shared_memory_bytes,
        dynamic_shared_memory_bytes=0,
        cta_traces=(
            CtaTrace(
                trace_class_id="uniform",
                block_ids=tuple(range(blocks)),
                warp_traces=warp_traces,
            ),
        ),
    )


def add_row(
    rows: list[dict[str, str | int]],
    *,
    check: str,
    parameter_a_name: str,
    parameter_a: int,
    parameter_b_name: str,
    parameter_b: int,
    measured: int,
    expected: int,
    unit: str,
) -> None:
    residual = measured - expected
    if residual != 0:
        raise AssertionError(
            f"{check}({parameter_a}, {parameter_b}) measured {measured}, expected {expected}"
        )
    rows.append(
        {
            "check": check,
            "parameter_a_name": parameter_a_name,
            "parameter_a": parameter_a,
            "parameter_b_name": parameter_b_name,
            "parameter_b": parameter_b,
            "measured_value": measured,
            "expected_value": expected,
            "residual": residual,
            "unit": unit,
        }
    )


def run() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []

    dependent_alu = SassInstruction(
        opcode="ALU", pipeline=PipelineKind.ALU, repeat=64, dependent=True
    )
    for blocks, sm_count, expected in SM_WAVE_CASES:
        arch = architecture(sm_count=sm_count, shared_memory_per_sm=1)
        estimate = SmSchedulerModel(arch).estimate(
            launch(
                blocks=blocks,
                threads=32,
                shared_memory_bytes=1,
                warp_traces=traces(1, dependent_alu),
            )
        )
        add_row(
            rows,
            check="sm_waves",
            parameter_a_name="grid_blocks",
            parameter_a=blocks,
            parameter_b_name="sm_count",
            parameter_b=sm_count,
            measured=estimate.duration_cycles,
            expected=expected,
            unit="cycles",
        )

    ready_alu = SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU)
    for warps, schedulers, expected in WARP_ISSUE_CASES:
        arch = architecture(schedulers=schedulers)
        estimate = SmSchedulerModel(arch).estimate(
            launch(
                blocks=1,
                threads=warps * arch.warp_size,
                warp_traces=traces(warps, ready_alu),
            )
        )
        add_row(
            rows,
            check="warp_issue",
            parameter_a_name="ready_warps",
            parameter_a=warps,
            parameter_b_name="schedulers_per_sm",
            parameter_b=schedulers,
            measured=estimate.duration_cycles,
            expected=expected,
            unit="cycles",
        )

    dependency = SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU, repeat=8, dependent=True)
    for schedulers in (1, 4):
        estimate = SmSchedulerModel(architecture(schedulers=schedulers)).estimate(
            launch(blocks=1, threads=32, warp_traces=traces(1, dependency))
        )
        add_row(
            rows,
            check="dependency_chain",
            parameter_a_name="instructions",
            parameter_a=8,
            parameter_b_name="schedulers_per_sm",
            parameter_b=schedulers,
            measured=estimate.duration_cycles,
            expected=32,
            unit="cycles",
        )

    for threads_per_block, registers_per_thread, expected in OCCUPANCY_CASES:
        arch = architecture()
        resident = SmSchedulerModel(arch).resident_blocks_per_sm(
            launch(
                blocks=1,
                threads=threads_per_block,
                registers_per_thread=registers_per_thread,
                warp_traces=traces(
                    (threads_per_block + arch.warp_size - 1) // arch.warp_size,
                    ready_alu,
                ),
            )
        )
        add_row(
            rows,
            check="occupancy",
            parameter_a_name="threads_per_block",
            parameter_a=threads_per_block,
            parameter_b_name="registers_per_thread",
            parameter_b=registers_per_thread,
            measured=resident,
            expected=expected,
            unit="resident_ctas_per_sm",
        )

    for byte_count, bandwidth, expected in HBM_CASES:
        instruction = SassInstruction(
            opcode="MEMORY",
            pipeline=PipelineKind.LOAD_STORE,
            memory_space=MemorySpace.HBM,
            requested_bytes=byte_count,
            transacted_bytes=byte_count,
            dependent=True,
        )
        estimate = SmSchedulerModel(architecture(bandwidth_bytes_per_cycle=bandwidth)).estimate(
            launch(blocks=1, threads=32, warp_traces=traces(1, instruction))
        )
        if estimate.hbm_serviced_bytes != byte_count:
            raise AssertionError("isolated HBM replay did not conserve transacted bytes")
        add_row(
            rows,
            check="hbm",
            parameter_a_name="transacted_bytes",
            parameter_a=byte_count,
            parameter_b_name="bytes_per_cycle",
            parameter_b=bandwidth,
            measured=estimate.duration_cycles,
            expected=expected,
            unit="cycles",
        )

    throughput_by_bandwidth: dict[float, list[tuple[int, float]]] = {}
    for byte_count, bandwidth, expected in COPY_CASES:
        arch = architecture(bandwidth_bytes_per_cycle=bandwidth)
        estimate = CopyEngineServiceModel(arch, "ce0").estimate(
            CopyTransfer(
                transfer_id=f"copy-{byte_count}-{bandwidth}",
                direction=CopyDirection.HOST_TO_DEVICE,
                bytes=byte_count,
                source="host:pinned",
                destination="gpu:0:hbm",
            )
        )
        # Regression tripwire only: duration is setup + ceil(bytes / rate),
        # so this inequality holds by arithmetic and cannot fail today.
        if estimate.effective_bandwidth_bytes_per_cycle > bandwidth:
            raise AssertionError("effective copy bandwidth exceeded the configured service rate")
        throughput_by_bandwidth.setdefault(bandwidth, []).append(
            (byte_count, estimate.effective_bandwidth_bytes_per_cycle)
        )
        add_row(
            rows,
            check="copy",
            parameter_a_name="descriptor_bytes",
            parameter_a=byte_count,
            parameter_b_name="bytes_per_cycle",
            parameter_b=bandwidth,
            measured=estimate.duration_cycles,
            expected=expected,
            unit="copy_engine_cycles",
        )

    # The registered clause the tripwire above does not cover: at a fixed
    # service rate, effective throughput must strictly rise with size,
    # because the fixed setup cost is amortized over more bytes.
    for bandwidth, samples in throughput_by_bandwidth.items():
        ordered = sorted(samples)
        for (smaller, low), (larger, high) in itertools.pairwise(ordered):
            if high <= low:
                raise AssertionError(
                    f"effective copy throughput at {bandwidth} bytes per cycle did not rise "
                    f"from {smaller} to {larger} bytes: {low} then {high}"
                )

    replay_arch = architecture(sm_count=2, schedulers=4)
    replay_launch = launch(
        blocks=5,
        threads=32,
        warp_traces=traces(
            1,
            SassInstruction(opcode="ALU", pipeline=PipelineKind.ALU, repeat=11, dependent=True),
        ),
    )
    scheduler = SmSchedulerModel(replay_arch)
    if scheduler.estimate(replay_launch) != scheduler.estimate(replay_launch):
        raise AssertionError("exact replay was not deterministic")

    for seed in (a100_sxm_80gb_seed_profile(), h100_sxm_80gb_seed_profile()):
        artifact = GpuModelArtifact(
            artifact_id=f"{seed.profile_id}-roundtrip",
            architecture=seed,
            captures=(),
            catalog=(),
            replays=(),
            copy_replays=(),
        )
        if GpuModelArtifact.from_json(artifact.to_json()) != artifact:
            raise AssertionError(f"seed artifact {seed.profile_id} did not round-trip")
        if seed.calibration.relative_uncertainty != 0.5 or seed.copy_engines:
            raise AssertionError("public seed claimed capture precision or copy topology")

    return rows


def write_results(rows: list[dict[str, str | int]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "results.csv")
    args = parser.parse_args()
    rows = run()
    write_results(rows, args.out)
    print(f"wrote {len(rows)} exact rows to {args.out}; maximum residual = 0")


if __name__ == "__main__":
    main()
