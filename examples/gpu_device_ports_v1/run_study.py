"""Run the COMP-34 GPU device port study frozen in expectations.md.

The study drives one composed GPU (`simllm.compute.gpu_device`) through three
questions the registered entry asks:

1. does the composed device reproduce every accepted `gpu_service_model`,
   `gpu_task_mix` and `mixed_makespan_v1` artifact byte for byte while its
   ports carry the current effective parameters,
2. does a disabled port, and a port asked for a capability it does not
   advertise, reject at configuration time rather than at first use, and
3. does enabling a port with a declared ceiling change an end-to-end metric by
   the exact registered amount.

The accepted harnesses are not edited. They are driven through the composed
device by rebinding, for the duration of one call, the symbol each harness
resolves when it builds a service model. The injected factory returns the same
real `SmSchedulerModel` and `CopyEngineServiceModel` classes, bound to the
device's effective architecture, so the only thing that changes is who composed
them.

Every expectation below is quoted from expectations.md and must never be edited
to match an observation.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from simllm.compute import (
    CopyDirection,
    CopyDirectionProfile,
    CopyEngineProfile,
    CopyEngineServiceModel,
    GpuArchitectureProfile,
    GpuCalibrationProfile,
    GpuDevice,
    GpuDeviceConfig,
    GpuModelProvenance,
    GpuPortCapability,
    GpuPortCeiling,
    GpuPortCeilingProvenance,
    GpuPortConfig,
    GpuPortDirection,
    GpuPortProtocol,
    GpuPortRole,
    MemoryHierarchyProfile,
    PipelineKind,
    PipelineProfile,
    SmSchedulerModel,
    WarpSchedulerPolicy,
    default_gpu_device_config,
    nccl_ring_allreduce_launch,
)
from simllm.core import (
    CoarseDeviceProfile,
    CoarseDeviceRuntime,
    DmaWork,
    ExecutionGraph,
    ExecutionOperation,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Frozen literals from expectations.md.
# ---------------------------------------------------------------------------

EXPECTATIONS_COMMIT = "ecd84f8b519eb4ad1f88f5429ee81ec4a1241681"

#: one cycle of the synthetic 1 GHz fixtures
PS_PER_CYCLE = 1_000
HOST_CLOCK_HZ = 1_000_000_000
HOST_SETUP_CYCLES = 20
HOST_BYTES_PER_CYCLE = 64.0

#: (transfer bytes, port ceiling bytes per cycle) -> expected JCT in ps
FROZEN_HOST_JCT_PS: dict[tuple[int, float], int] = {
    (4_096, 64.0): 84_000,
    (4_096, 32.0): 148_000,
    (4_096, 16.0): 276_000,
    (16_384, 64.0): 276_000,
    (16_384, 32.0): 532_000,
    (16_384, 16.0): 1_044_000,
}

#: transfer bytes -> device-to-host JCT in ps while only ingress is overridden
FROZEN_HOST_ISOLATION_PS: dict[int, int] = {4_096: 84_000, 16_384: 276_000}

#: the ingress ceiling used for the direction-isolation cells
ISOLATION_CEILING_BYTES_PER_CYCLE = 16.0

#: (chunk bytes, port ceiling bytes per cycle) -> expected replay cycles
FROZEN_PEER_CYCLES: dict[tuple[int, float], int] = {
    (64, 16.0): 328,
    (64, 8.0): 456,
    (64, 4.0): 712,
    (128, 16.0): 456,
    (128, 8.0): 712,
    (128, 4.0): 1_224,
}

#: the accepted C3 ring row and its band under a halved peer ceiling
FROZEN_RING_BASELINE_CYCLES = 4_397
FROZEN_RING_BAND_CYCLES = (8_392, 8_493)
RING_CEILING_BYTES_PER_CYCLE = 8.0

#: napkin bounds stated before the run: label -> (floor, ceiling, expected)
FROZEN_PHYSICAL_INTERVALS: dict[str, tuple[int, int, int]] = {
    "host link 4096 bytes at 64 bytes per cycle, ps": (64_000, 84_000, 84_000),
    "peer link 32 stores of 64 bytes at 16 bytes per cycle, cycles": (328, 328, 328),
}

#: the accepted artifacts the composed device must reproduce byte for byte
TASK_MIX_ARTIFACTS = ("results.csv", "nccl_convergence.csv", "diagnostics.csv")

#: the accepted evidence counts that study prints, already locked by
#: tests/test_gpu_task_mix_demo.py; guarded here as well because a composed run
#: that silently dropped a replay would still write matching CSV bytes
TASK_MIX_ACCEPTED_SUMMARY = (
    "distinct run configurations: 38",
    "replay invocations: 46",
    "scored exact-oracle rows: 36/36 pass",
    "scored behavioral relation families: 6/6 pass",
    "scored behavioral relation instances: 17/17 pass",
    "unscored structural invariants: 21/21 hold",
)

# Evidence classes. Only GENUINE_RISK rows enter a behavioral fraction.
GENUINE_RISK = "behavioral-relation"
FATAL_GUARD = "fatal-guard"
RAW_OBSERVATION = "raw-observation"
RUN_CONFIGURATION = "run-configuration"

FAMILY_S1 = "S1 host-link ceiling override reaches the end-to-end metric"
FAMILY_S2 = "S2 the override stays inside the direction its port carries"
FAMILY_S3 = "S3 peer-link ceiling override moves the egress term"
FAMILY_S4 = "S4 the ring cell under a halved peer ceiling"
SCORED_FAMILIES = (FAMILY_S1, FAMILY_S2, FAMILY_S3, FAMILY_S4)

GUARD_BYTE_IDENTITY = "byte identity of the accepted artifacts"
GUARD_MUTATION = "mutation sensitivity of the byte lock"
GUARD_REJECTION = "configuration-time rejection"
GUARD_APPLICABILITY = "applicability and inertness"


@dataclass(frozen=True)
class Row:
    """One evidence row. ``expected`` is frozen, ``measured`` is observed."""

    family: str
    evidence_class: str
    case: str
    expected: Any
    measured: Any

    @property
    def is_predicate(self) -> bool:
        return self.evidence_class in (GENUINE_RISK, FATAL_GUARD)

    @property
    def passed(self) -> bool:
        return not self.is_predicate or self.expected == self.measured

    @property
    def status(self) -> str:
        if not self.is_predicate:
            return "REPORTED"
        return "PASS" if self.passed else "FAIL"


# ---------------------------------------------------------------------------
# Accepted studies, loaded as modules so their own fixtures are used.
# ---------------------------------------------------------------------------


def _load(name: str, relative: str) -> Any:
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load accepted study {relative}")
    module = importlib.util.module_from_spec(spec)
    # Register before executing: a loaded study may define dataclasses, and
    # dataclasses resolves annotations through sys.modules while the body runs.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TASK_MIX = _load(
    "gpu_device_ports_task_mix", "examples/gpu_task_mix/run_gpu_task_mix.py"
)
SERVICE_MODEL = _load(
    "gpu_device_ports_service_model",
    "examples/gpu_service_model/run_gpu_service_model.py",
)
MIXED_MAKESPAN = _load(
    "gpu_device_ports_mixed_makespan", "examples/mixed_makespan_v1/run_study.py"
)


@contextlib.contextmanager
def _rebound(bindings: tuple[tuple[Any, str, Any], ...]) -> Iterator[None]:
    """Rebind module attributes for one call, then restore them exactly."""

    saved = [(owner, name, getattr(owner, name)) for owner, name, _ in bindings]
    try:
        for owner, name, value in bindings:
            setattr(owner, name, value)
        yield
    finally:
        for owner, name, value in saved:
            setattr(owner, name, value)


def _device(
    architecture: GpuArchitectureProfile,
    *,
    ceilings: dict[str, float] | None = None,
) -> GpuDevice:
    """Compose one GPU from the ports its own mechanisms imply."""

    config = default_gpu_device_config(
        architecture,
        capabilities=(GpuPortCapability.CEILING_OVERRIDE,),
    )
    if ceilings:
        ports = []
        for port in config.ports:
            value = ceilings.get(port.port_id)
            if value is None:
                ports.append(port)
                continue
            ports.append(
                replace(
                    port,
                    declared_ceiling=_declared(architecture, port, value),
                )
            )
        config = replace(config, ports=tuple(ports))
    return GpuDevice(config, architecture)


def _declared(
    architecture: GpuArchitectureProfile,
    port: GpuPortConfig,
    bytes_per_cycle: float,
) -> GpuPortCeiling:
    if port.copy_engine_id is not None:
        clock_hz = architecture.copy_engine(port.copy_engine_id).clock_hz
    else:
        clock_hz = architecture.calibration.core_clock_hz
    return GpuPortCeiling(
        bytes_per_cycle=bytes_per_cycle,
        clock_hz=clock_hz,
        provenance=GpuPortCeilingProvenance.MODEL_CONFIGURATION,
        source="gpu_device_ports_v1 study fixture, no silicon claim",
        reference="examples/gpu_device_ports_v1/expectations.md",
    )


def _scheduler_class(
    ceilings: dict[str, float] | None = None,
) -> type[SmSchedulerModel]:
    """Return a real ``SmSchedulerModel`` subclass composed through a device.

    A subclass rather than a factory function, because the accepted harnesses
    and the CORE-4 runtime both type-check the service they are handed. The
    instance is a genuine ``SmSchedulerModel`` bound to the device's effective
    architecture; nothing about the replay changes.
    """

    class ComposedSmSchedulerModel(SmSchedulerModel):
        port_ceilings = ceilings

        def __init__(self, architecture: GpuArchitectureProfile) -> None:
            device = _device(architecture, ceilings=type(self).port_ceilings)
            super().__init__(device.architecture)
            self.device = device

    return ComposedSmSchedulerModel


def _copy_class() -> type[CopyEngineServiceModel]:
    """Return a real ``CopyEngineServiceModel`` subclass composed through a device."""

    class ComposedCopyEngineServiceModel(CopyEngineServiceModel):
        def __init__(
            self, architecture: GpuArchitectureProfile, engine_id: str
        ) -> None:
            device = _device(architecture)
            super().__init__(device.architecture, engine_id)
            self.device = device

    return ComposedCopyEngineServiceModel


# ---------------------------------------------------------------------------
# Fixture H, the host link.
# ---------------------------------------------------------------------------

HOST_PROFILE_ID = "gpu-device-ports-host-v1"
HOST_ENGINE_ID = "ce0"
HOST_INGRESS_PORT = "pcie-host-ingress"
HOST_EGRESS_PORT = "pcie-host-egress"


def host_architecture() -> GpuArchitectureProfile:
    """Return the synthetic 1 GHz host-link fixture frozen in expectations.md."""

    calibration = GpuCalibrationProfile(
        calibration_id="gpu-device-ports-host-calibration-v1",
        target_architecture_profile_id=HOST_PROFILE_ID,
        provenance=GpuModelProvenance(
            source="synthetic study fixture, no silicon claim",
            version="1",
            gpu="gpu-device-ports-synthetic",
            created="2026-08-17",
        ),
        core_clock_hz=HOST_CLOCK_HZ,
        target_memory_clock_hz=None,
        pipelines=(
            PipelineProfile(
                kind=PipelineKind.ALU,
                opcodes=("ALU",),
                latency_cycles=1,
                issue_width_per_sm=4,
            ),
        ),
        memory=MemoryHierarchyProfile(
            hbm_latency_cycles=0,
            hbm_bandwidth_bytes_per_cycle=64,
        ),
        copy_engines=(
            CopyEngineProfile(
                engine_id=HOST_ENGINE_ID,
                clock_hz=HOST_CLOCK_HZ,
                direction_profiles=tuple(
                    CopyDirectionProfile(
                        direction=direction,
                        setup_cycles=HOST_SETUP_CYCLES,
                        bandwidth_bytes_per_cycle=HOST_BYTES_PER_CYCLE,
                    )
                    for direction in (
                        CopyDirection.HOST_TO_DEVICE,
                        CopyDirection.DEVICE_TO_HOST,
                    )
                ),
            ),
        ),
        warp_scheduler_policy=WarpSchedulerPolicy.LOOSE_ROUND_ROBIN,
        relative_uncertainty=0.0,
    )
    return GpuArchitectureProfile(
        profile_id=HOST_PROFILE_ID,
        gpu_name="gpu-device-ports-synthetic",
        sm_count=1,
        warp_size=32,
        scheduler_count_per_sm=4,
        max_blocks_per_sm=16,
        max_warps_per_sm=64,
        max_threads_per_sm=2_048,
        max_threads_per_block=1_024,
        registers_per_sm=65_536,
        max_registers_per_thread=255,
        register_allocation_granularity_per_warp=1,
        shared_memory_per_sm=65_536,
        max_static_shared_memory_per_block=49_152,
        max_shared_memory_per_block=65_536,
        shared_memory_allocation_granularity=1,
        calibration=calibration,
    )


def host_port(
    *,
    port_id: str,
    direction: GpuPortDirection,
    copy_direction: CopyDirection,
    enabled: bool = True,
    ceiling: GpuPortCeiling | None = None,
) -> GpuPortConfig:
    return GpuPortConfig(
        port_id=port_id,
        protocol=GpuPortProtocol.PCIE,
        role=GpuPortRole.HOST_LINK,
        direction=direction,
        enabled=enabled,
        capabilities=(
            GpuPortCapability.COPY_ENGINE_TRANSFER,
            GpuPortCapability.CEILING_OVERRIDE,
        ),
        copy_engine_id=HOST_ENGINE_ID,
        copy_directions=(copy_direction,),
        declared_ceiling=ceiling,
    )


def host_device(*, ingress_ceiling: float | None = None) -> GpuDevice:
    """Return the two-port host-link device, optionally overriding ingress."""

    architecture = host_architecture()
    ceiling = (
        None
        if ingress_ceiling is None
        else GpuPortCeiling(
            bytes_per_cycle=ingress_ceiling,
            clock_hz=HOST_CLOCK_HZ,
            provenance=GpuPortCeilingProvenance.MODEL_CONFIGURATION,
            source="gpu_device_ports_v1 study fixture, no silicon claim",
            reference="examples/gpu_device_ports_v1/expectations.md",
        )
    )
    config = GpuDeviceConfig(
        device_id="gpu-device-ports-host",
        ports=(
            host_port(
                port_id=HOST_INGRESS_PORT,
                direction=GpuPortDirection.INGRESS,
                copy_direction=CopyDirection.HOST_TO_DEVICE,
                ceiling=ceiling,
            ),
            host_port(
                port_id=HOST_EGRESS_PORT,
                direction=GpuPortDirection.EGRESS,
                copy_direction=CopyDirection.DEVICE_TO_HOST,
            ),
        ),
    )
    return GpuDevice(config, architecture)


def dma_jct_ps(device: GpuDevice, *, byte_count: int, to_device: bool) -> int:
    """Execute one DMA descriptor through the live runtime and return its JCT."""

    profile = CoarseDeviceProfile(
        launch_service_ps=0,
        completion_delivery_ps=0,
        copy_engines=(device.copy_engine_service(HOST_ENGINE_ID),),
    )
    source, destination = (
        ("host:pinned", "gpu:0:hbm") if to_device else ("gpu:0:hbm", "host:pinned")
    )
    label = "h2d" if to_device else "d2h"
    graph = ExecutionGraph(
        f"gpu-device-ports-{label}-{byte_count}",
        0,
        0,
        (
            ExecutionOperation(
                "copy",
                0,
                "cuda:0:copy",
                DmaWork(f"{label}-{byte_count}", source, destination, byte_count),
            ),
        ),
    )
    result = CoarseDeviceRuntime(profile).execute(graph)
    return result.completed_at_ps - graph.released_at_ps


# ---------------------------------------------------------------------------
# Scored families.
# ---------------------------------------------------------------------------


def family_s1(rows: list[Row], raw: dict[str, Any]) -> None:
    """S1: a host-link ceiling override reaches the end-to-end metric."""

    observed: dict[str, int] = {}
    for (byte_count, ceiling), expected in FROZEN_HOST_JCT_PS.items():
        override = None if ceiling == HOST_BYTES_PER_CYCLE else ceiling
        device = host_device(ingress_ceiling=override)
        measured = dma_jct_ps(device, byte_count=byte_count, to_device=True)
        case = f"bytes={byte_count},ceiling={ceiling:g}"
        observed[case] = measured
        rows.append(Row(FAMILY_S1, GENUINE_RISK, case, expected, measured))
    raw["s1_host_to_device_jct_ps"] = observed


def family_s2(rows: list[Row], raw: dict[str, Any]) -> None:
    """S2: the override stays inside the direction its port carries."""

    observed: dict[str, int] = {}
    device = host_device(ingress_ceiling=ISOLATION_CEILING_BYTES_PER_CYCLE)
    for byte_count, expected in FROZEN_HOST_ISOLATION_PS.items():
        measured = dma_jct_ps(device, byte_count=byte_count, to_device=False)
        case = (
            f"bytes={byte_count},ingress_ceiling="
            f"{ISOLATION_CEILING_BYTES_PER_CYCLE:g},egress untouched"
        )
        observed[case] = measured
        rows.append(Row(FAMILY_S2, GENUINE_RISK, case, expected, measured))
    raw["s2_device_to_host_jct_ps"] = observed


def peer_egress_cycles(*, chunk_bytes: int, ceiling: float | None) -> int:
    """Replay the accepted task-mix egress cell through the composed device."""

    architecture = TASK_MIX.architecture(nvlink_bandwidth=16)
    ceilings = None if ceiling is None else {"nvlink-peer-store": ceiling}
    device = _device(architecture, ceilings=ceilings)
    launch = TASK_MIX.egress_launch(warps=8, per_warp=4, chunk_bytes=chunk_bytes)
    return device.sm_scheduler_model().estimate(launch).duration_cycles


def family_s3(rows: list[Row], raw: dict[str, Any]) -> None:
    """S3: a peer-link ceiling override moves the egress term."""

    observed: dict[str, int] = {}
    for (chunk, ceiling), expected in FROZEN_PEER_CYCLES.items():
        override = None if ceiling == 16.0 else ceiling
        measured = peer_egress_cycles(chunk_bytes=chunk, ceiling=override)
        case = f"chunk={chunk},ceiling={ceiling:g}"
        observed[case] = measured
        rows.append(Row(FAMILY_S3, GENUINE_RISK, case, expected, measured))
    raw["s3_peer_egress_cycles"] = observed


def ring_cycles(*, ceiling: float | None) -> int:
    architecture = TASK_MIX.architecture(sm_count=2)
    ceilings = None if ceiling is None else {"nvlink-peer-store": ceiling}
    device = _device(architecture, ceilings=ceilings)
    launch = nccl_ring_allreduce_launch(
        payload_bytes=65_536,
        world_size=2,
        channels=2,
        chunk_bytes=64,
        warps_per_channel=8,
    )
    return device.sm_scheduler_model().estimate(launch).duration_cycles


def family_s4(rows: list[Row], raw: dict[str, Any]) -> None:
    """S4: the accepted ring cell under a halved peer ceiling."""

    baseline = ring_cycles(ceiling=None)
    measured = ring_cycles(ceiling=RING_CEILING_BYTES_PER_CYCLE)
    low, high = FROZEN_RING_BAND_CYCLES
    raw["s4_ring_cycles"] = {"baseline": baseline, "halved_ceiling": measured}
    rows.append(
        Row(
            FAMILY_S4,
            GENUINE_RISK,
            f"ring at ceiling={RING_CEILING_BYTES_PER_CYCLE:g}, inside [{low}, {high}]",
            True,
            low <= measured <= high,
        )
    )
    rows.append(
        Row(
            GUARD_BYTE_IDENTITY,
            FATAL_GUARD,
            "accepted C3 ring row reproduced through the composed device",
            FROZEN_RING_BASELINE_CYCLES,
            baseline,
        )
    )


# ---------------------------------------------------------------------------
# Fatal guards.
# ---------------------------------------------------------------------------


def guard_byte_identity(rows: list[Row], raw: dict[str, Any], out: Path) -> None:
    """Drive every accepted study through the composed device with default ports."""

    task_mix_out = out / "byte_identity" / "gpu_task_mix"
    task_mix_out.mkdir(parents=True, exist_ok=True)
    exit_code, transcript = _run_task_mix(task_mix_out, ceilings=None)
    rows.append(
        Row(
            GUARD_BYTE_IDENTITY,
            FATAL_GUARD,
            "gpu_task_mix exits zero through the composed device",
            0,
            exit_code,
        )
    )
    rows.append(
        Row(
            GUARD_BYTE_IDENTITY,
            FATAL_GUARD,
            "gpu_task_mix reports its accepted evidence counts unchanged",
            list(TASK_MIX_ACCEPTED_SUMMARY),
            [line for line in TASK_MIX_ACCEPTED_SUMMARY if line in transcript],
        )
    )
    for name in TASK_MIX_ARTIFACTS:
        accepted = (REPO_ROOT / "examples" / "gpu_task_mix" / name).read_bytes()
        rows.append(
            Row(
                GUARD_BYTE_IDENTITY,
                FATAL_GUARD,
                f"gpu_task_mix/{name} reproduces byte for byte",
                True,
                (task_mix_out / name).read_bytes() == accepted,
            )
        )

    service_out = out / "byte_identity" / "gpu_service_model"
    service_out.mkdir(parents=True, exist_ok=True)
    _run_service_model(service_out / "results.csv")
    accepted = (REPO_ROOT / "examples" / "gpu_service_model" / "results.csv").read_bytes()
    rows.append(
        Row(
            GUARD_BYTE_IDENTITY,
            FATAL_GUARD,
            "gpu_service_model/results.csv reproduces byte for byte",
            True,
            (service_out / "results.csv").read_bytes() == accepted,
        )
    )

    bare = _mixed_makespan_record(composed=False)
    composed = _mixed_makespan_record(composed=True)
    mixed_out = out / "byte_identity" / "mixed_makespan_v1"
    mixed_out.mkdir(parents=True, exist_ok=True)
    (mixed_out / "raw_observations.bare.json").write_bytes(bare)
    (mixed_out / "raw_observations.composed.json").write_bytes(composed)
    rows.append(
        Row(
            GUARD_BYTE_IDENTITY,
            FATAL_GUARD,
            "mixed_makespan_v1 raw observation record reproduces byte for byte",
            True,
            bare == composed,
        )
    )

    for label, architecture in (
        ("gpu_task_mix", TASK_MIX.architecture()),
        ("gpu_service_model", SERVICE_MODEL.architecture()),
        ("mixed_makespan_v1", MIXED_MAKESPAN.architecture()),
        ("host link fixture", host_architecture()),
    ):
        device = _device(architecture)
        rows.append(
            Row(
                GUARD_BYTE_IDENTITY,
                FATAL_GUARD,
                f"{label}: default ports leave the architecture object identical",
                True,
                device.architecture is architecture,
            )
        )
        for port in device.ports:
            assert port.ceiling is not None
            expected = _mechanism_bandwidth(architecture, port)
            rows.append(
                Row(
                    GUARD_BYTE_IDENTITY,
                    FATAL_GUARD,
                    f"{label}: port {port.port_id} reads its mechanism ceiling",
                    (expected, GpuPortCeilingProvenance.CALIBRATION_DERIVED.value),
                    (port.ceiling.bytes_per_cycle, port.ceiling.provenance.value),
                )
            )
    raw["port_reports"] = [
        {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in vars(report).items()
        }
        for architecture in (
            TASK_MIX.architecture(),
            SERVICE_MODEL.architecture(),
            host_architecture(),
        )
        for report in _device(architecture).port_reports()
    ]


def _mechanism_bandwidth(
    architecture: GpuArchitectureProfile,
    port: Any,
) -> float:
    if port.advertises(GpuPortCapability.PEER_STORE_EGRESS):
        nvlink = architecture.calibration.nvlink
        assert nvlink is not None
        return nvlink.bandwidth_bytes_per_cycle
    assert port.copy_engine_id is not None
    engine = architecture.copy_engine(port.copy_engine_id)
    return engine.service(port.copy_directions[0]).bandwidth_bytes_per_cycle


def _run_task_mix(out: Path, *, ceilings: dict[str, float] | None) -> tuple[int, str]:
    """Run the accepted task-mix harness through the composed device."""

    argv = [str(REPO_ROOT / "examples/gpu_task_mix/run_gpu_task_mix.py"), "--out", str(out)]
    captured = io.StringIO()
    with (
        _rebound(
            (
                (TASK_MIX, "SmSchedulerModel", _scheduler_class(ceilings)),
                (sys, "argv", argv),
            )
        ),
        contextlib.redirect_stdout(captured),
    ):
        exit_code = TASK_MIX.main()
    transcript = captured.getvalue()
    (out / "stdout.txt").write_text(transcript, encoding="utf-8")
    return exit_code, transcript


def _run_service_model(out: Path) -> str:
    """Run the accepted service-model harness through the composed device."""

    argv = [
        str(REPO_ROOT / "examples/gpu_service_model/run_gpu_service_model.py"),
        "--out",
        str(out),
    ]
    captured = io.StringIO()
    with (
        _rebound(
            (
                (SERVICE_MODEL, "SmSchedulerModel", _scheduler_class()),
                (SERVICE_MODEL, "CopyEngineServiceModel", _copy_class()),
                (sys, "argv", argv),
            )
        ),
        contextlib.redirect_stdout(captured),
    ):
        SERVICE_MODEL.main()
    transcript = captured.getvalue()
    (out.parent / "stdout.txt").write_text(transcript, encoding="utf-8")
    return transcript


def _mixed_makespan_record(*, composed: bool) -> bytes:
    """Return the study's raw observation record, serialized exactly as it writes it."""

    import simllm.compute as compute_package

    bindings: tuple[tuple[Any, str, Any], ...] = ()
    if composed:
        bindings = ((compute_package, "SmSchedulerModel", _scheduler_class()),)
    with _rebound(bindings):
        payload = {
            "component": MIXED_MAKESPAN.component_observations(),
            "live": MIXED_MAKESPAN.live_observations(),
        }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def guard_mutation(rows: list[Row], raw: dict[str, Any], out: Path) -> None:
    """A byte lock that cannot see a mechanism change is not a lock."""

    mutated = out / "mutation_control" / "gpu_task_mix"
    mutated.mkdir(parents=True, exist_ok=True)
    _run_task_mix(mutated, ceilings={"nvlink-peer-store": 8.0})
    accepted = (REPO_ROOT / "examples" / "gpu_task_mix" / "results.csv").read_bytes()
    observed = (mutated / "results.csv").read_bytes()
    rows.append(
        Row(
            GUARD_MUTATION,
            FATAL_GUARD,
            "a halved peer ceiling changes gpu_task_mix/results.csv",
            True,
            observed != accepted,
        )
    )
    raw["mutation_control_bytes_differ"] = observed != accepted


def _rejects(rows: list[Row], case: str, action: Callable[[], object]) -> None:
    try:
        action()
    except ValueError:
        measured: str = "ValueError"
    except (TypeError, KeyError, AssertionError) as exc:
        # the guard reports whatever it got, so a wrong exception type is visible
        measured = type(exc).__name__
    else:
        measured = "no rejection"
    rows.append(Row(GUARD_REJECTION, FATAL_GUARD, case, "ValueError", measured))


def guard_rejection(rows: list[Row]) -> None:
    """Every clause here must raise during configuration, not at first use."""

    architecture = host_architecture()
    peer_architecture = TASK_MIX.architecture()
    ceiling = GpuPortCeiling(
        bytes_per_cycle=32.0,
        clock_hz=HOST_CLOCK_HZ,
        provenance=GpuPortCeilingProvenance.MODEL_CONFIGURATION,
        source="rejection fixture",
    )

    _rejects(
        rows,
        "a disabled port carrying a declared ceiling",
        lambda: host_port(
            port_id="disabled-with-ceiling",
            direction=GpuPortDirection.INGRESS,
            copy_direction=CopyDirection.HOST_TO_DEVICE,
            enabled=False,
            ceiling=ceiling,
        ),
    )
    _rejects(
        rows,
        "a declared ceiling without the ceiling-override capability",
        lambda: GpuPortConfig(
            port_id="no-override-capability",
            protocol=GpuPortProtocol.PCIE,
            role=GpuPortRole.HOST_LINK,
            direction=GpuPortDirection.INGRESS,
            capabilities=(GpuPortCapability.COPY_ENGINE_TRANSFER,),
            copy_engine_id=HOST_ENGINE_ID,
            copy_directions=(CopyDirection.HOST_TO_DEVICE,),
            declared_ceiling=ceiling,
        ),
    )

    disabled_device = GpuDevice(
        GpuDeviceConfig(
            device_id="disabled-port-device",
            ports=(
                host_port(
                    port_id=HOST_INGRESS_PORT,
                    direction=GpuPortDirection.INGRESS,
                    copy_direction=CopyDirection.HOST_TO_DEVICE,
                    enabled=False,
                ),
            ),
        ),
        architecture,
    )
    _rejects(
        rows,
        "requesting a capability of a disabled port",
        lambda: disabled_device.require_capability(
            HOST_INGRESS_PORT, GpuPortCapability.COPY_ENGINE_TRANSFER
        ),
    )
    live_device = host_device()
    _rejects(
        rows,
        "requesting a capability the port does not advertise",
        lambda: live_device.require_capability(
            HOST_INGRESS_PORT, GpuPortCapability.PEER_STORE_EGRESS
        ),
    )
    for capability in (
        GpuPortCapability.ECN_MARKING,
        GpuPortCapability.PRIORITY_FLOW_CONTROL,
        GpuPortCapability.CONGESTION_NOTIFICATION,
    ):
        _rejects(
            rows,
            f"declaring transport-control capability {capability.value}",
            lambda capability=capability: GpuPortConfig(
                port_id="transport-control",
                protocol=GpuPortProtocol.NVLINK,
                role=GpuPortRole.PEER_LINK,
                direction=GpuPortDirection.EGRESS,
                capabilities=(GpuPortCapability.PEER_STORE_EGRESS, capability),
            ),
        )
    _rejects(
        rows,
        "an xGMI port, which COMP-35 owns",
        lambda: GpuPortConfig(
            port_id="xgmi-peer-store",
            protocol=GpuPortProtocol.XGMI,
            role=GpuPortRole.PEER_LINK,
            direction=GpuPortDirection.EGRESS,
            capabilities=(GpuPortCapability.PEER_STORE_EGRESS,),
        ),
    )
    _rejects(
        rows,
        "a peer-store port on a calibration with no NVLink profile",
        lambda: GpuDevice(
            GpuDeviceConfig(
                device_id="no-nvlink",
                ports=(
                    GpuPortConfig(
                        port_id="nvlink-peer-store",
                        protocol=GpuPortProtocol.NVLINK,
                        role=GpuPortRole.PEER_LINK,
                        direction=GpuPortDirection.EGRESS,
                        capabilities=(GpuPortCapability.PEER_STORE_EGRESS,),
                    ),
                ),
            ),
            architecture,
        ),
    )
    _rejects(
        rows,
        "a copy-engine port naming an engine the architecture lacks",
        lambda: GpuDevice(
            GpuDeviceConfig(
                device_id="unknown-engine",
                ports=(
                    GpuPortConfig(
                        port_id="pcie-host-ingress",
                        protocol=GpuPortProtocol.PCIE,
                        role=GpuPortRole.HOST_LINK,
                        direction=GpuPortDirection.INGRESS,
                        capabilities=(GpuPortCapability.COPY_ENGINE_TRANSFER,),
                        copy_engine_id="ce-absent",
                        copy_directions=(CopyDirection.HOST_TO_DEVICE,),
                    ),
                ),
            ),
            architecture,
        ),
    )
    _rejects(
        rows,
        "a copy-engine port naming a direction the engine lacks",
        lambda: GpuDevice(
            GpuDeviceConfig(
                device_id="undeclared-direction",
                ports=(
                    GpuPortConfig(
                        port_id="nvlink-peer-copy",
                        protocol=GpuPortProtocol.NVLINK,
                        role=GpuPortRole.PEER_LINK,
                        direction=GpuPortDirection.EGRESS,
                        capabilities=(GpuPortCapability.COPY_ENGINE_TRANSFER,),
                        copy_engine_id=HOST_ENGINE_ID,
                        copy_directions=(CopyDirection.PEER_TO_PEER,),
                    ),
                ),
            ),
            architecture,
        ),
    )
    _rejects(
        rows,
        "an ingress port carrying a device-to-host copy",
        lambda: GpuPortConfig(
            port_id="wrong-direction",
            protocol=GpuPortProtocol.PCIE,
            role=GpuPortRole.HOST_LINK,
            direction=GpuPortDirection.INGRESS,
            capabilities=(GpuPortCapability.COPY_ENGINE_TRANSFER,),
            copy_engine_id=HOST_ENGINE_ID,
            copy_directions=(CopyDirection.DEVICE_TO_HOST,),
        ),
    )
    _rejects(
        rows,
        "a duplicate port identifier",
        lambda: GpuDeviceConfig(
            device_id="duplicate-ports",
            ports=(
                host_port(
                    port_id=HOST_INGRESS_PORT,
                    direction=GpuPortDirection.INGRESS,
                    copy_direction=CopyDirection.HOST_TO_DEVICE,
                ),
                host_port(
                    port_id=HOST_INGRESS_PORT,
                    direction=GpuPortDirection.INGRESS,
                    copy_direction=CopyDirection.HOST_TO_DEVICE,
                ),
            ),
        ),
    )
    _rejects(
        rows,
        "two ports claiming one copy direction of one engine",
        lambda: GpuDeviceConfig(
            device_id="two-copy-authorities",
            ports=(
                host_port(
                    port_id=HOST_INGRESS_PORT,
                    direction=GpuPortDirection.INGRESS,
                    copy_direction=CopyDirection.HOST_TO_DEVICE,
                ),
                host_port(
                    port_id="pcie-host-ingress-shadow",
                    direction=GpuPortDirection.INGRESS,
                    copy_direction=CopyDirection.HOST_TO_DEVICE,
                ),
            ),
        ),
    )
    _rejects(
        rows,
        "two ports claiming the peer-store egress cursor",
        lambda: GpuDeviceConfig(
            device_id="two-egress-authorities",
            ports=tuple(
                GpuPortConfig(
                    port_id=f"nvlink-peer-store-{index}",
                    protocol=GpuPortProtocol.NVLINK,
                    role=GpuPortRole.PEER_LINK,
                    direction=GpuPortDirection.EGRESS,
                    capabilities=(GpuPortCapability.PEER_STORE_EGRESS,),
                )
                for index in range(2)
            ),
        ),
    )
    _rejects(
        rows,
        "an unsupported port config version",
        lambda: GpuPortConfig(
            port_id="future-port",
            protocol=GpuPortProtocol.NVLINK,
            role=GpuPortRole.PEER_LINK,
            direction=GpuPortDirection.EGRESS,
            version=99,
            capabilities=(GpuPortCapability.PEER_STORE_EGRESS,),
        ),
    )
    _rejects(
        rows,
        "an unsupported device config version",
        lambda: GpuDeviceConfig(device_id="future-device", version=99),
    )
    _rejects(
        rows,
        "a declared ceiling on the wrong clock",
        lambda: GpuDevice(
            GpuDeviceConfig(
                device_id="wrong-clock",
                ports=(
                    host_port(
                        port_id=HOST_INGRESS_PORT,
                        direction=GpuPortDirection.INGRESS,
                        copy_direction=CopyDirection.HOST_TO_DEVICE,
                        ceiling=replace(ceiling, clock_hz=500_000_000),
                    ),
                ),
            ),
            architecture,
        ),
    )
    _rejects(
        rows,
        "a declared ceiling claiming calibration-derived provenance",
        lambda: host_port(
            port_id=HOST_INGRESS_PORT,
            direction=GpuPortDirection.INGRESS,
            copy_direction=CopyDirection.HOST_TO_DEVICE,
            ceiling=replace(
                ceiling,
                provenance=GpuPortCeilingProvenance.CALIBRATION_DERIVED,
            ),
        ),
    )
    _rejects(
        rows,
        "NVLink declared on a host-link role",
        lambda: GpuPortConfig(
            port_id="nvlink-on-host",
            protocol=GpuPortProtocol.NVLINK,
            role=GpuPortRole.HOST_LINK,
            direction=GpuPortDirection.INGRESS,
            capabilities=(GpuPortCapability.COPY_ENGINE_TRANSFER,),
            copy_engine_id=HOST_ENGINE_ID,
            copy_directions=(CopyDirection.HOST_TO_DEVICE,),
        ),
    )
    _rejects(
        rows,
        "an enabled port declaring no mechanism capability",
        lambda: GpuPortConfig(
            port_id="inert-enabled",
            protocol=GpuPortProtocol.PCIE,
            role=GpuPortRole.HOST_LINK,
            direction=GpuPortDirection.INGRESS,
            capabilities=(GpuPortCapability.CEILING_OVERRIDE,),
        ),
    )
    # Added during implementation, beyond the frozen list: one port may not
    # average two mechanism ceilings that disagree, which is the asymmetric
    # host link the design statement warns about.
    asymmetric = replace(
        architecture,
        calibration=replace(
            architecture.calibration,
            copy_engines=(
                CopyEngineProfile(
                    engine_id=HOST_ENGINE_ID,
                    clock_hz=HOST_CLOCK_HZ,
                    direction_profiles=(
                        CopyDirectionProfile(
                            direction=CopyDirection.HOST_TO_DEVICE,
                            setup_cycles=HOST_SETUP_CYCLES,
                            bandwidth_bytes_per_cycle=64.0,
                        ),
                        CopyDirectionProfile(
                            direction=CopyDirection.DEVICE_TO_HOST,
                            setup_cycles=HOST_SETUP_CYCLES,
                            bandwidth_bytes_per_cycle=26.0,
                        ),
                    ),
                ),
            ),
        ),
    )
    _rejects(
        rows,
        "one bidirectional port over two disagreeing mechanism ceilings",
        lambda: GpuDevice(
            GpuDeviceConfig(
                device_id="asymmetric-host",
                ports=(
                    GpuPortConfig(
                        port_id="pcie-host",
                        protocol=GpuPortProtocol.PCIE,
                        role=GpuPortRole.HOST_LINK,
                        direction=GpuPortDirection.BIDIRECTIONAL,
                        capabilities=(GpuPortCapability.COPY_ENGINE_TRANSFER,),
                        copy_engine_id=HOST_ENGINE_ID,
                        copy_directions=(
                            CopyDirection.HOST_TO_DEVICE,
                            CopyDirection.DEVICE_TO_HOST,
                        ),
                    ),
                ),
            ),
            asymmetric,
        ),
    )
    _rejects(
        rows,
        "a port naming the device-to-device copy direction",
        lambda: GpuPortConfig(
            port_id="internal-copy",
            protocol=GpuPortProtocol.PCIE,
            role=GpuPortRole.HOST_LINK,
            direction=GpuPortDirection.BIDIRECTIONAL,
            capabilities=(GpuPortCapability.COPY_ENGINE_TRANSFER,),
            copy_engine_id=HOST_ENGINE_ID,
            copy_directions=(CopyDirection.DEVICE_TO_DEVICE,),
        ),
    )
    # The mechanism's own first-use rejection is untouched by the port layer.
    peer_device = _device(peer_architecture)
    rows.append(
        Row(
            GUARD_REJECTION,
            FATAL_GUARD,
            "the copy-engine first-use rejection still raises for an absent engine",
            "KeyError",
            _raised(lambda: peer_device.copy_engine_service("ce0")),
        )
    )


def _raised(action: Callable[[], object]) -> str:
    try:
        action()
    except (ValueError, TypeError, KeyError, AssertionError) as exc:
        return type(exc).__name__
    return "no rejection"


def guard_applicability(rows: list[Row], raw: dict[str, Any]) -> None:
    """A disabled port keeps its interface and rescopes nothing."""

    architecture = host_architecture()
    config = GpuDeviceConfig(
        device_id="disabled-port-device",
        ports=(
            host_port(
                port_id=HOST_INGRESS_PORT,
                direction=GpuPortDirection.INGRESS,
                copy_direction=CopyDirection.HOST_TO_DEVICE,
                enabled=False,
            ),
        ),
    )
    device = GpuDevice(config, architecture)
    port = device.port(HOST_INGRESS_PORT)
    rows.append(
        Row(
            GUARD_APPLICABILITY,
            FATAL_GUARD,
            "a disabled port is reported, not applicable, with no ceiling",
            ("not_applicable", None, True),
            (
                port.applicability.value,
                port.ceiling,
                GpuPortCapability.COPY_ENGINE_TRANSFER in port.capabilities,
            ),
        )
    )
    rows.append(
        Row(
            GUARD_APPLICABILITY,
            FATAL_GUARD,
            "a disabled port rescopes nothing: the architecture object is identical",
            True,
            device.architecture is architecture,
        )
    )
    baseline = dma_jct_ps(host_device(), byte_count=4_096, to_device=True)
    disabled = dma_jct_ps(device, byte_count=4_096, to_device=True)
    rows.append(
        Row(
            GUARD_APPLICABILITY,
            FATAL_GUARD,
            "the mechanism behind a disabled port keeps its accepted timing",
            baseline,
            disabled,
        )
    )
    raw["disabled_port_jct_ps"] = {"enabled": baseline, "disabled": disabled}


def report_physical_bounds(rows: list[Row], raw: dict[str, Any]) -> None:
    """Napkin floor and ceiling, stated in the freeze, reported before precision."""

    host = dma_jct_ps(host_device(), byte_count=4_096, to_device=True)
    peer = peer_egress_cycles(chunk_bytes=64, ceiling=None)
    observed = {
        "host link 4096 bytes at 64 bytes per cycle, ps": host,
        "peer link 32 stores of 64 bytes at 16 bytes per cycle, cycles": peer,
    }
    for label, (floor, ceiling, expected) in FROZEN_PHYSICAL_INTERVALS.items():
        measured = observed[label]
        rows.append(
            Row(
                "physical bounds",
                RAW_OBSERVATION,
                f"{label}: floor {floor}, ceiling {ceiling}, frozen {expected}",
                expected,
                measured,
            )
        )
        rows.append(
            Row(
                "physical bounds",
                FATAL_GUARD,
                f"{label}: measurement sits inside the first-principles interval",
                True,
                floor <= measured <= ceiling,
            )
        )
    raw["physical_bounds"] = observed


def report_derived_ratios(rows: list[Row], raw: dict[str, Any]) -> None:
    """Ratios entailed by S1 and S3; reported, never scored."""

    host_terms = {
        byte_count: [
            FROZEN_HOST_JCT_PS[(byte_count, ceiling)] - HOST_SETUP_CYCLES * PS_PER_CYCLE
            for ceiling in (64.0, 32.0, 16.0)
        ]
        for byte_count in (4_096, 16_384)
    }
    peer_terms = {
        chunk: [FROZEN_PEER_CYCLES[(chunk, ceiling)] - 200 for ceiling in (16.0, 8.0, 4.0)]
        for chunk in (64, 128)
    }
    for byte_count, terms in host_terms.items():
        rows.append(
            Row(
                "derived, unscored",
                RAW_OBSERVATION,
                f"host serialization term at {byte_count} bytes, ps",
                terms,
                [terms[0], terms[0] * 2, terms[0] * 4],
            )
        )
    for chunk, terms in peer_terms.items():
        rows.append(
            Row(
                "derived, unscored",
                RAW_OBSERVATION,
                f"peer serialization term at chunk {chunk}, cycles",
                terms,
                [terms[0], terms[0] * 2, terms[0] * 4],
            )
        )
    raw["derived_serialization_terms"] = {
        "host_ps": {str(key): value for key, value in host_terms.items()},
        "peer_cycles": {str(key): value for key, value in peer_terms.items()},
    }


def report_run_configurations(rows: list[Row], raw: dict[str, Any]) -> None:
    configurations = (
        len(FROZEN_HOST_JCT_PS)
        + len(FROZEN_HOST_ISOLATION_PS)
        + len(FROZEN_PEER_CYCLES)
        + 2
    )
    rows.append(
        Row(
            "run configurations",
            RUN_CONFIGURATION,
            "distinct device configurations executed by the scored families",
            configurations,
            configurations,
        )
    )
    raw["scored_run_configurations"] = configurations


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _write_rows_csv(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["family", "evidence_class", "case", "expected", "measured", "status"])
        for row in rows:
            writer.writerow(
                [
                    row.family,
                    row.evidence_class,
                    row.case,
                    json.dumps(row.expected, sort_keys=True, default=str),
                    json.dumps(row.measured, sort_keys=True, default=str),
                    row.status,
                ]
            )


def _write_port_reports_csv(path: Path, raw: dict[str, Any]) -> None:
    reports = raw["port_reports"]
    fieldnames = list(reports[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    key: ";".join(value) if isinstance(value, list) else value
                    for key, value in report.items()
                }
            )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_study(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    raw: dict[str, Any] = {}

    guard_byte_identity(rows, raw, out)
    guard_mutation(rows, raw, out)
    guard_rejection(rows)
    guard_applicability(rows, raw)
    report_physical_bounds(rows, raw)
    family_s1(rows, raw)
    family_s2(rows, raw)
    family_s3(rows, raw)
    family_s4(rows, raw)
    report_derived_ratios(rows, raw)
    report_run_configurations(rows, raw)

    guards = [row for row in rows if row.evidence_class == FATAL_GUARD]
    failed_guards = [row for row in guards if not row.passed]
    scored = [row for row in rows if row.evidence_class == GENUINE_RISK]
    void = bool(failed_guards)

    summary: dict[str, Any] = {
        "expectations_commit": EXPECTATIONS_COMMIT,
        "void": void,
        "fatal_guards_evaluated": len(guards),
        "fatal_guards_failed": [row.case for row in failed_guards],
        "scored_instances": None if void else len(scored),
        "scored_passed": None if void else sum(1 for row in scored if row.passed),
        "scored_families": None
        if void
        else {
            family: {
                "passed": sum(
                    1 for row in scored if row.family == family and row.passed
                ),
                "instances": sum(1 for row in scored if row.family == family),
            }
            for family in SCORED_FAMILIES
        },
    }

    _write_rows_csv(out / "rows.csv", rows)
    _write_port_reports_csv(out / "port_reports.csv", raw)
    _write_json(out / "raw_observations.json", raw)
    _write_json(out / "summary.json", summary)
    for name in ("rows.csv", "port_reports.csv", "raw_observations.json", "summary.json"):
        print(f"wrote {out / name}")

    if void:
        print("VOID: a frozen fatal guard failed; the behavioral score is uninterpretable")
        for row in failed_guards:
            print(f"  fatal fail: {row.family} / {row.case}")
        return 1
    print(f"fatal guards: {len(guards)} evaluated, none violated")
    print(
        f"scored {summary['scored_passed']} of {summary['scored_instances']} "
        f"genuine-risk instances across {len(SCORED_FAMILIES)} families"
    )
    for row in scored:
        print(f"  {row.status:4} {row.family[:2]} {row.case:52} {row.measured}")
    return 0 if summary["scored_passed"] == summary["scored_instances"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out is None:
        raise SystemExit(
            "--out is required; the run regenerates the accepted artifacts and "
            "belongs outside the repository. Set the run root in local "
            "configuration, e.g. SIMLLM_GPU_DEVICE_PORTS_RUN_ROOT, and pass it "
            "explicitly"
        )
    return run_study(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
