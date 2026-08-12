"""Compute-time provider interface and the first two implementations."""

from __future__ import annotations

import abc
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

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
    #: optional exact family decomposition of this fused kernel
    family_kernels: tuple[KernelSpec, ...] = ()


@dataclass(frozen=True)
class DurationEstimate:
    duration_ps: int
    #: estimate provenance/regime: "compute", "memory", "measured",
    #: "interpolated" (a table lookup between neighbors), or "sass-replay"
    bound: str
    #: relative uncertainty (0.1 = ±10%); providers must be honest here
    uncertainty: float = 0.0


class ComputeProvider(abc.ABC):
    """Maps a kernel (or fused region) to a simulated duration.

    A provider may also expose an ordered layer breakdown for a fused step
    through :meth:`estimate_layers`. The default returns ``None`` so existing
    providers and their callers retain the scalar estimate exactly.

    ``precision_compute_level`` names the compute seam level this provider
    implements, using the vocabulary of
    :class:`simllm.core.ComputeLevel`. It is a plain string so this module
    stays independent of the core package. ``None`` means the provider does
    not declare a level, and the unified precision surface then treats the
    compute seam as unconstrained by this spelling rather than assuming one.
    """

    #: declared compute-seam level, or None when the provider names no level
    precision_compute_level: str | None = None

    @abc.abstractmethod
    def estimate(self, kernel: KernelSpec, gpu: GpuSpec) -> DurationEstimate: ...

    def estimate_layers(
        self,
        kernel: KernelSpec,
        gpu: GpuSpec,
        num_layers: int,
    ) -> tuple[DurationEstimate, ...] | None:
        """Optional ordered layer durations for ``kernel``.

        When present, the result must contain ``num_layers`` nonnegative
        durations whose picosecond sum equals :meth:`estimate` exactly. A
        caller must validate that contract before using the breakdown.
        ``None`` selects the caller's scalar compatibility behavior.
        """
        return None


class RooflineProvider(ComputeProvider):
    """Analytical roofline: ``t = max(flops/peak_flops, bytes/mem_bw)``.

    Classifies each kernel as compute- or memory-bound from its arithmetic
    intensity against the GPU envelope. ``efficiency`` derates peak numbers
    (real kernels rarely reach 100% of either roof).
    """

    precision_compute_level = "roofline"

    def __init__(
        self,
        efficiency: float = 0.7,
        *,
        enable_layer_breakdown: bool = False,
    ):
        if not 0.0 < efficiency <= 1.0:
            raise ValueError("efficiency must be in (0, 1]")
        self.efficiency = efficiency
        self.enable_layer_breakdown = enable_layer_breakdown

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

    def estimate_layers(
        self,
        kernel: KernelSpec,
        gpu: GpuSpec,
        num_layers: int,
    ) -> tuple[DurationEstimate, ...] | None:
        """Split a fused estimate using work on its selected roof.

        Transformer families aggregate work over all layers except for the
        LM head. The repeated work is shared equally; the complete LM-head
        family belongs to the final layer. Integer cumulative boundaries make
        the returned durations sum to the fused estimate exactly.
        """
        if not self.enable_layer_breakdown or not kernel.family_kernels:
            return None
        if isinstance(num_layers, bool) or not isinstance(num_layers, int):
            raise TypeError("num_layers must be an integer")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        family_flops = sum(family.flops for family in kernel.family_kernels)
        family_bytes = sum(family.bytes_moved for family in kernel.family_kernels)
        if any(
            family.flops < 0 or family.bytes_moved < 0
            for family in kernel.family_kernels
        ):
            raise ValueError("family work must be nonnegative")
        if family_flops != kernel.flops or family_bytes != kernel.bytes_moved:
            raise ValueError("family kernels must conserve fused flops and bytes exactly")

        fused = self.estimate(kernel, gpu)
        selected = "flops" if fused.bound == "compute" else "bytes_moved"
        weighted_families = tuple(
            (family.name, Fraction(getattr(family, selected)))
            for family in kernel.family_kernels
        )
        head_weight = sum(
            (weight for name, weight in weighted_families if name == "lm_head"),
            start=Fraction(0),
        )
        repeated_weight = sum(
            (weight for name, weight in weighted_families if name != "lm_head"),
            start=Fraction(0),
        )
        layer_weights = [repeated_weight] * num_layers
        layer_weights[-1] += head_weight * num_layers
        total_weight = sum(layer_weights, start=Fraction(0))
        if total_weight == 0:
            if fused.duration_ps != 0:
                raise ValueError("zero family work cannot apportion a nonzero estimate")
            return tuple(
                DurationEstimate(0, fused.bound, fused.uncertainty)
                for _ in range(num_layers)
            )

        cumulative_weight = Fraction(0)
        previous_boundary_ps = 0
        estimates = []
        for index, weight in enumerate(layer_weights):
            cumulative_weight += weight
            boundary_ps = (
                fused.duration_ps
                if index == num_layers - 1
                else (fused.duration_ps * cumulative_weight) // total_weight
            )
            estimates.append(
                DurationEstimate(
                    duration_ps=boundary_ps - previous_boundary_ps,
                    bound=fused.bound,
                    uncertainty=fused.uncertainty,
                )
            )
            previous_boundary_ps = boundary_ps
        return tuple(estimates)


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
    made, so replayed builds stay byte-reproducible. ``references`` carries
    the audit links of the upstream evidence, so citations survive
    compilation from a richer artifact into this compact table.
    """

    source: str
    version: str
    gpu: str
    created: str
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "references", tuple(self.references))
        for reference in self.references:
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError("profile-table provenance references must be nonblank strings")
        if len(set(self.references)) != len(self.references):
            raise ValueError("profile-table provenance references must not contain duplicates")

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "version": self.version,
            "gpu": self.gpu,
            "created": self.created,
            "references": list(self.references),
        }

    @classmethod
    def from_json(cls, payload: dict) -> ProfileTableProvenance:
        missing = [f for f in ("source", "version", "gpu", "created") if f not in payload]
        if missing:
            raise ValueError(f"profile-table provenance missing fields: {missing}")
        references = payload.get("references", [])
        if not isinstance(references, list):
            raise TypeError("profile-table provenance references must be an array")
        parsed_references = []
        for index, reference in enumerate(references):
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(
                    f"profile-table provenance references[{index}] must be a nonblank string"
                )
            parsed_references.append(reference)
        return cls(
            source=str(payload["source"]),
            version=str(payload["version"]),
            gpu=str(payload["gpu"]),
            created=str(payload["created"]),
            references=tuple(parsed_references),
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

    precision_compute_level = "profile-table"

    EXACT_UNCERTAINTY = 0.05
    MIN_INTERPOLATED_UNCERTAINTY = 0.15

    def __init__(
        self,
        table: dict[ProfileKey, int | tuple[int, float]],
        provenance: ProfileTableProvenance | None = None,
        *,
        enable_family_sum: bool = False,
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
        self.enable_family_sum = enable_family_sum

    def estimate(self, kernel: KernelSpec, gpu: GpuSpec) -> DurationEstimate:
        key = (kernel.name, kernel.config, gpu.name)
        if key in self._durations:
            return DurationEstimate(
                duration_ps=self._durations[key],
                bound="measured",
                uncertainty=self._uncertainties[key],
            )
        if self.enable_family_sum and kernel.family_kernels:
            return self._estimate_family_sum(kernel, gpu)
        return self._interpolate(kernel, gpu)

    def _estimate_family_sum(
        self,
        kernel: KernelSpec,
        gpu: GpuSpec,
    ) -> DurationEstimate:
        """Sum independently keyed family entries for one fused kernel.

        This path is opt-in. Direct table entries retain precedence, and the
        default provider keeps the historical exact-match and interpolation
        behavior. Relative uncertainty is the conservative sum of each
        child's absolute uncertainty divided by the summed duration.
        """

        children = []
        for family in kernel.family_kernels:
            try:
                children.append(self.estimate(family, gpu))
            except KeyError as error:
                raise KeyError(
                    f"fused kernel {kernel.name!r} has no supported profile "
                    f"for family {family.name!r}: {error}"
                ) from error
        duration_ps = sum(child.duration_ps for child in children)
        if duration_ps == 0:
            uncertainty = max(
                (child.uncertainty for child in children),
                default=0.0,
            )
        else:
            uncertainty = sum(
                child.duration_ps * child.uncertainty for child in children
            ) / duration_ps
        bound = (
            "measured"
            if all(child.bound == "measured" for child in children)
            else "interpolated"
        )
        return DurationEstimate(
            duration_ps=duration_ps,
            bound=bound,
            uncertainty=uncertainty,
        )

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
        path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        enable_family_sum: bool = False,
    ) -> ProfileTableProvider:
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
        return cls(
            table,
            provenance=provenance,
            enable_family_sum=enable_family_sum,
        )
