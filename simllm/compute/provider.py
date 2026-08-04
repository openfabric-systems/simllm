"""Compute-time provider interface and the first two implementations."""

from __future__ import annotations

import abc
import json
import math
from dataclasses import dataclass
from pathlib import Path

PS_PER_SECOND = 1_000_000_000_000

#: versioned on-disk schema of a ProfileTableProvider table
PROFILE_TABLE_SCHEMA = "simllm-profile-table-v1"


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
    #: which regime dominated: "compute", "memory", "measured", or
    #: "interpolated" (a table lookup between measured neighbors)
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


#: table key: (kernel name, config tuple, gpu name)
ProfileKey = tuple[str, tuple[tuple[str, int], ...], str]


@dataclass(frozen=True)
class ProfileTableProvenance:
    """Where a profile table's numbers came from; mandatory in the artifact.

    ``source`` names the producer kind ("accel-sim" for offline SASS
    simulation, "capture" for silicon measurements); ``version`` is the
    simulator or capture-tool version string; ``gpu`` is the GPU the table
    describes; ``created`` is a caller-supplied date string. The library
    never reads the clock itself: the caller states when the table was
    made, so replayed builds stay byte-reproducible.
    """

    source: str
    version: str
    gpu: str
    created: str

    def to_json(self) -> dict[str, str]:
        return {
            "source": self.source,
            "version": self.version,
            "gpu": self.gpu,
            "created": self.created,
        }

    @classmethod
    def from_json(cls, payload: dict) -> ProfileTableProvenance:
        missing = [f for f in ("source", "version", "gpu", "created") if f not in payload]
        if missing:
            raise ValueError(f"profile-table provenance missing fields: {missing}")
        return cls(
            source=str(payload["source"]),
            version=str(payload["version"]),
            gpu=str(payload["gpu"]),
            created=str(payload["created"]),
        )


class ProfileTableProvider(ComputeProvider):
    """Measured (kernel name, config, gpu) → duration lookups.

    Tables come from real captures (phase benchmarks, torch profiler) or
    from offline SASS-level simulation (COMP-1). ``table`` values are the
    duration in ps, either bare (uncertainty defaults to
    :data:`EXACT_UNCERTAINTY`) or as a ``(duration_ps, uncertainty)`` pair;
    the plain-int form is the original constructor and keeps working
    unchanged. The versioned JSON artifact (:data:`PROFILE_TABLE_SCHEMA`)
    round-trips through :meth:`save` / :meth:`load` and requires
    :class:`ProfileTableProvenance`.

    Lookup rule: an exact key hit returns the measured entry. On a miss the
    provider interpolates log-linearly along ONE numeric config axis: among
    entries of the same kernel name and gpu whose configs agree with the
    query on every other axis, the nearest bracketing pair below and above
    the query value gives ``log(duration)`` linear in ``log(axis value)``.
    Interpolated estimates carry ``bound="interpolated"`` and an inflated
    uncertainty of ``max(0.15, neighbors' uncertainties)``: interpolation
    can never claim tighter error than its inputs, and never tighter than
    0.15 even between two confident points. A query outside the covered
    range of every axis, or one that differs from all covered points on
    more than one axis at once (multi-axis interpolation is COMP-4),
    raises ``KeyError`` exactly like the original exact-match miss.
    Log-linear interpolation needs positive axis values and durations;
    a non-positive value on a config axis removes that axis as an
    interpolation direction (the query can still interpolate along a
    different, positive axis).
    """

    EXACT_UNCERTAINTY = 0.05
    MIN_INTERPOLATED_UNCERTAINTY = 0.15

    def __init__(
        self,
        table: dict[ProfileKey, int | tuple[int, float]],
        provenance: ProfileTableProvenance | None = None,
    ):
        self._durations: dict[ProfileKey, int] = {}
        self._uncertainties: dict[ProfileKey, float] = {}
        for key, value in table.items():
            if isinstance(value, tuple):
                duration_ps, uncertainty = value
            else:
                duration_ps, uncertainty = value, self.EXACT_UNCERTAINTY
            self._durations[key] = int(duration_ps)
            self._uncertainties[key] = float(uncertainty)
        self.provenance = provenance

    def estimate(self, kernel: KernelSpec, gpu: GpuSpec) -> DurationEstimate:
        key = (kernel.name, kernel.config, gpu.name)
        if key in self._durations:
            return DurationEstimate(
                duration_ps=self._durations[key],
                bound="measured",
                uncertainty=self._uncertainties[key],
            )
        return self._interpolate(kernel, gpu)

    def _interpolate(self, kernel: KernelSpec, gpu: GpuSpec) -> DurationEstimate:
        axis_names = tuple(name for name, _ in kernel.config)
        query = dict(kernel.config)
        neighbors: list[tuple[ProfileKey, dict[str, int]]] = []
        for key in self._durations:
            name, config, gpu_name = key
            if name != kernel.name or gpu_name != gpu.name:
                continue
            if tuple(n for n, _ in config) != axis_names:
                continue
            neighbors.append((key, dict(config)))
        for axis in axis_names:
            q = query[axis]
            if q <= 0:
                continue
            below: tuple[ProfileKey, int] | None = None
            above: tuple[ProfileKey, int] | None = None
            for key, config in neighbors:
                if any(config[a] != query[a] for a in axis_names if a != axis):
                    continue
                v = config[axis]
                if 0 < v < q and (below is None or v > below[1]):
                    below = (key, v)
                elif v > q and (above is None or v < above[1]):
                    above = (key, v)
            if below is None or above is None:
                continue
            (lo_key, lo), (hi_key, hi) = below, above
            d_lo, d_hi = self._durations[lo_key], self._durations[hi_key]
            if d_lo <= 0 or d_hi <= 0:
                continue
            frac = (math.log(q) - math.log(lo)) / (math.log(hi) - math.log(lo))
            duration = math.exp(math.log(d_lo) + frac * (math.log(d_hi) - math.log(d_lo)))
            uncertainty = max(
                self.MIN_INTERPOLATED_UNCERTAINTY,
                self._uncertainties[lo_key],
                self._uncertainties[hi_key],
            )
            return DurationEstimate(
                duration_ps=round(duration),
                bound="interpolated",
                uncertainty=uncertainty,
            )
        raise KeyError(
            f"no profile entry for {(kernel.name, kernel.config, gpu.name)} and no "
            "same-kernel neighbors bracket it on a single config axis "
            "(out of covered range, or multi-axis interpolation: COMP-4)"
        )

    def save(self, path: str | Path) -> Path:
        """Write the versioned JSON artifact; requires provenance."""
        if self.provenance is None:
            raise ValueError("saving a profile table requires provenance")
        entries = []
        for key in sorted(self._durations):
            name, config, gpu_name = key
            entries.append(
                {
                    "kernel": name,
                    "config": [[axis, value] for axis, value in config],
                    "gpu": gpu_name,
                    "duration_ps": self._durations[key],
                    "uncertainty": self._uncertainties[key],
                }
            )
        payload = {
            "schema": PROFILE_TABLE_SCHEMA,
            "provenance": self.provenance.to_json(),
            "entries": entries,
        }
        path = Path(path)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> ProfileTableProvider:
        """Load a versioned JSON artifact; rejects any other schema loudly."""
        payload = json.loads(Path(path).read_text())
        schema = payload.get("schema")
        if schema != PROFILE_TABLE_SCHEMA:
            raise ValueError(
                f"unsupported profile-table schema {schema!r}; "
                f"this reader speaks {PROFILE_TABLE_SCHEMA!r}"
            )
        provenance = ProfileTableProvenance.from_json(payload.get("provenance", {}))
        table: dict[ProfileKey, int | tuple[int, float]] = {}
        for entry in payload.get("entries", []):
            key = (
                str(entry["kernel"]),
                tuple((str(axis), int(value)) for axis, value in entry["config"]),
                str(entry["gpu"]),
            )
            table[key] = (int(entry["duration_ps"]), float(entry["uncertainty"]))
        return cls(table, provenance=provenance)
