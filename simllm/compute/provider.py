"""Compute-time provider interface and the first two implementations."""

from __future__ import annotations

import abc
from dataclasses import dataclass

PS_PER_SECOND = 1_000_000_000_000


@dataclass(frozen=True)
class GpuSpec:
    """Peak envelope of one GPU for analytical estimates."""

    name: str
    #: peak throughput for the relevant dtype/pipeline, FLOP/s
    peak_flops: float
    #: HBM bandwidth, bytes/s
    mem_bandwidth: float


@dataclass(frozen=True)
class KernelSpec:
    """One unit of GPU work whose duration the provider must estimate.

    ``flops`` and ``bytes_moved`` describe the work; providers that key on
    measured tables may instead match on ``name`` and shape parameters
    carried in ``config``.
    """

    name: str
    flops: float
    bytes_moved: float
    #: free-form shape/config key, e.g. {"batch_tokens": 512, "hidden": 7168}
    config: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class DurationEstimate:
    duration_ps: int
    #: which regime dominated: "compute", "memory", or "measured"
    bound: str
    #: relative uncertainty (0.1 = ±10%); providers must be honest here
    uncertainty: float = 0.0


class ComputeProvider(abc.ABC):
    """Maps a kernel (or fused region) to a simulated duration."""

    @abc.abstractmethod
    def estimate(self, kernel: KernelSpec, gpu: GpuSpec) -> DurationEstimate: ...


class RooflineProvider(ComputeProvider):
    """Analytical roofline: ``t = max(flops/peak_flops, bytes/mem_bw)``.

    Classifies each kernel as compute- or memory-bound from its arithmetic
    intensity against the GPU envelope. ``efficiency`` derates peak numbers
    (real kernels rarely reach 100% of either roof).
    """

    def __init__(self, efficiency: float = 0.7):
        if not 0.0 < efficiency <= 1.0:
            raise ValueError("efficiency must be in (0, 1]")
        self.efficiency = efficiency

    def estimate(self, kernel: KernelSpec, gpu: GpuSpec) -> DurationEstimate:
        t_compute = kernel.flops / (gpu.peak_flops * self.efficiency)
        t_memory = kernel.bytes_moved / (gpu.mem_bandwidth * self.efficiency)
        bound = "compute" if t_compute >= t_memory else "memory"
        duration = max(t_compute, t_memory)
        return DurationEstimate(
            duration_ps=int(duration * PS_PER_SECOND),
            bound=bound,
            uncertainty=0.5,
        )


class ProfileTableProvider(ComputeProvider):
    """Measured (kernel name, config, gpu) → duration lookups.

    Tables come from real captures (phase benchmarks, torch profiler) or from
    offline SASS-level simulation. Exact-match only for now; interpolation
    across configs is future work.
    """

    def __init__(self, table: dict[tuple[str, tuple[tuple[str, int], ...], str], int]):
        self._table = table

    def estimate(self, kernel: KernelSpec, gpu: GpuSpec) -> DurationEstimate:
        key = (kernel.name, kernel.config, gpu.name)
        if key not in self._table:
            raise KeyError(f"no profile entry for {key}")
        return DurationEstimate(
            duration_ps=self._table[key], bound="measured", uncertainty=0.05
        )
