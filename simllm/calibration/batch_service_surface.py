"""Pool-local batch-service surfaces imported from measured record rows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from simllm.compute.provider import (
    ComputeProvider,
    GpuSpec,
    KernelSpec,
    ProfileKey,
    ProfileTableProvenance,
    ProfileTableProvider,
    RooflineProvider,
)

BATCH_SERVICE_PROVENANCE_SCHEMA = "simllm-batch-service-pricing-provenance-v1"


@dataclass(frozen=True, slots=True)
class BatchServicePoint:
    """One measured batch point and its exact source-row identity."""

    batch_size: int
    duration_ps: int
    uncertainty_fraction: float
    entry_key_sha256: str
    evidence_class: str = "MEASURED"
    split: str = "calibration"

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or type(self.batch_size) is not int:
            raise TypeError("batch_size must be an integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if isinstance(self.duration_ps, bool) or type(self.duration_ps) is not int:
            raise TypeError("duration_ps must be an integer")
        if self.duration_ps <= 0:
            raise ValueError("duration_ps must be positive")
        if not 0.0 <= self.uncertainty_fraction <= 1.0:
            raise ValueError("uncertainty_fraction must be in [0, 1]")
        if len(self.entry_key_sha256) != 64:
            raise ValueError("entry_key_sha256 must be a SHA-256 hex digest")
        try:
            int(self.entry_key_sha256, 16)
        except ValueError as exc:
            raise ValueError("entry_key_sha256 must be hexadecimal") from exc
        if self.evidence_class != "MEASURED":
            raise ValueError("batch-service points must be MEASURED")
        if self.split != "calibration":
            raise ValueError("batch-service points must be calibration rows")


def interpolate_batch_service_ps(
    points: tuple[BatchServicePoint, ...],
    batch_size: int,
) -> int:
    """Power-law interpolate total service between bracketing batch points."""

    if isinstance(batch_size, bool) or type(batch_size) is not int:
        raise TypeError("batch_size must be an integer")
    ordered = tuple(sorted(points, key=lambda point: point.batch_size))
    if len(ordered) < 2 or len({point.batch_size for point in ordered}) != len(ordered):
        raise ValueError("batch-service surface needs unique bracketing points")
    exact = {point.batch_size: point.duration_ps for point in ordered}
    if batch_size in exact:
        return exact[batch_size]
    lower = max(
        (point for point in ordered if point.batch_size < batch_size),
        key=lambda point: point.batch_size,
        default=None,
    )
    upper = min(
        (point for point in ordered if point.batch_size > batch_size),
        key=lambda point: point.batch_size,
        default=None,
    )
    if lower is None or upper is None:
        raise ValueError("batch_size is outside the measured surface")
    fraction = (
        math.log(batch_size) - math.log(lower.batch_size)
    ) / (math.log(upper.batch_size) - math.log(lower.batch_size))
    value = math.exp(
        math.log(lower.duration_ps)
        + fraction * (math.log(upper.duration_ps) - math.log(lower.duration_ps))
    )
    return round(value)


class BatchSurfaceLookupBinding:
    """Bind live request counts to an imported profile-table batch axis.

    This is a binding for the existing :class:`ProfileTableProvider`, not a
    new timing provider. Exact endpoints and the table provider's power-law
    interpolation replace the comparator only inside the measured batch span.
    """

    def __init__(
        self,
        points: tuple[BatchServicePoint, ...],
        *,
        record_sha256: str,
        acceptance_status: str,
        campaign_id: str,
        coverage: str,
        record_device_kind_id: str,
        pool: str = "decode",
    ) -> None:
        self.points = tuple(sorted(points, key=lambda point: point.batch_size))
        if len(self.points) < 2:
            raise ValueError("batch-service surface needs at least two points")
        if len({point.batch_size for point in self.points}) != len(self.points):
            raise ValueError("batch-service point sizes must be unique")
        if len(record_sha256) != 64:
            raise ValueError("record_sha256 must be a SHA-256 hex digest")
        try:
            int(record_sha256, 16)
        except ValueError as exc:
            raise ValueError("record_sha256 must be hexadecimal") from exc
        if acceptance_status not in {"candidate", "validated"}:
            raise ValueError("acceptance_status must be candidate or validated")
        for name, value in (
            ("campaign_id", campaign_id),
            ("coverage", coverage),
            ("record_device_kind_id", record_device_kind_id),
            ("pool", pool),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonblank string")
        if pool != "decode":
            raise ValueError("the v1 batch surface prices only the decode pool")
        self.record_sha256 = record_sha256
        self.acceptance_status = acceptance_status
        self.campaign_id = campaign_id
        self.coverage = coverage
        self.record_device_kind_id = record_device_kind_id
        self.pool = pool
        self.record_gpu = GpuSpec(record_device_kind_id, 1.0, 1.0)
        self._lookup_hits = 0
        self._lookup_misses = 0
        self._exact_hits = 0
        self._interpolated_hits = 0
        self._selected_batch_sizes: list[int] = []
        self._pending_batch_size: int | None = None

    def profile_query(
        self,
        kernel: KernelSpec,
    ) -> tuple[KernelSpec, GpuSpec] | None:
        """Return one batch-axis table query, or the comparator miss signal."""

        if not kernel.request_shapes:
            self._pending_batch_size = None
            return None
        batch_size = len(kernel.request_shapes)
        if not self.points[0].batch_size <= batch_size <= self.points[-1].batch_size:
            self._pending_batch_size = None
            return None
        self._pending_batch_size = batch_size
        return (
            KernelSpec(
                name="pool-local-batch-service",
                flops=0,
                bytes_moved=0,
                config=(("batch_size", batch_size),),
            ),
            self.record_gpu,
        )

    def record_lookup(self, selected: bool) -> None:
        """Count one provider decision without counting layer probes."""

        batch_size = self._pending_batch_size
        if not selected or batch_size is None:
            self._lookup_misses += 1
            return
        self._lookup_hits += 1
        if batch_size not in self._selected_batch_sizes:
            self._selected_batch_sizes.append(batch_size)
        if batch_size in {point.batch_size for point in self.points}:
            self._exact_hits += 1
        else:
            self._interpolated_hits += 1

    def pricing_provenance(self) -> dict[str, Any]:
        """Expose the candidate source and separate exact/interpolated use."""

        return {
            "schema": BATCH_SERVICE_PROVENANCE_SCHEMA,
            "record_sha256": self.record_sha256,
            "acceptance_status": self.acceptance_status,
            "campaign_id": self.campaign_id,
            "coverage": self.coverage,
            "record_device_kind_id": self.record_device_kind_id,
            "pool": self.pool,
            "surface_entry_key_sha256s": [
                point.entry_key_sha256 for point in self.points
            ],
            "surface_batch_sizes": [point.batch_size for point in self.points],
            "surface_evidence_classes": [
                point.evidence_class for point in self.points
            ],
            "selected_batch_sizes": list(self._selected_batch_sizes),
            "lookup_hits": self._lookup_hits,
            "lookup_misses": self._lookup_misses,
            "exact_hits": self._exact_hits,
            "interpolated_hits": self._interpolated_hits,
            "calibration_claim": False,
        }


def compile_pool_local_batch_service_provider(
    points: tuple[BatchServicePoint, ...],
    *,
    record_sha256: str,
    acceptance_status: str,
    campaign_id: str,
    coverage: str,
    record_device_kind_id: str,
    pool: str = "decode",
    comparator: ComputeProvider | None = None,
) -> ProfileTableProvider:
    """Compile a content-addressed batch surface into the landed provider."""

    binding = BatchSurfaceLookupBinding(
        points,
        record_sha256=record_sha256,
        acceptance_status=acceptance_status,
        campaign_id=campaign_id,
        coverage=coverage,
        record_device_kind_id=record_device_kind_id,
        pool=pool,
    )
    fallback = RooflineProvider(efficiency=0.7) if comparator is None else comparator
    if not isinstance(fallback, ComputeProvider):
        raise TypeError("comparator must implement ComputeProvider")
    table: dict[ProfileKey, tuple[int, float]] = {
        (
            "pool-local-batch-service",
            (("batch_size", point.batch_size),),
            record_device_kind_id,
        ): (point.duration_ps, point.uncertainty_fraction)
        for point in binding.points
    }
    return ProfileTableProvider(
        table,
        ProfileTableProvenance(
            source="content-addressed-kernel-cycle-record",
            version=record_sha256,
            gpu=record_device_kind_id,
            created="2026-08-26",
        ),
        lookup_binding=binding,
        comparator=fallback,
    )


__all__ = [
    "BATCH_SERVICE_PROVENANCE_SCHEMA",
    "BatchServicePoint",
    "BatchSurfaceLookupBinding",
    "compile_pool_local_batch_service_provider",
    "interpolate_batch_service_ps",
]
