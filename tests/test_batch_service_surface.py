from __future__ import annotations

import pickle

from simllm.calibration.batch_service_surface import (
    BATCH_SERVICE_PROVENANCE_SCHEMA,
    BatchServicePoint,
    compile_pool_local_batch_service_provider,
    interpolate_batch_service_ps,
)
from simllm.compute import (
    GPU_ENVELOPES,
    ComputeProvider,
    DurationEstimate,
    KernelRequestShape,
    KernelSpec,
)

RECORD_SHA256 = "f" * 64
KEY_ONE = "1" * 64
KEY_EIGHT = "8" * 64


class FixedComparator(ComputeProvider):
    def estimate(self, kernel: KernelSpec, gpu) -> DurationEstimate:
        return DurationEstimate(77, "comparison", 0.25)


def _points() -> tuple[BatchServicePoint, ...]:
    return (
        BatchServicePoint(1, 1_110_576_000, 0.004232, KEY_ONE),
        BatchServicePoint(8, 1_892_831_500, 0.001538, KEY_EIGHT),
    )


def _provider():
    return compile_pool_local_batch_service_provider(
        _points(),
        record_sha256=RECORD_SHA256,
        acceptance_status="candidate",
        campaign_id="candidate-campaign",
        coverage="complete-kernel-stream",
        record_device_kind_id="nvidia-gh200-sm90",
        comparator=FixedComparator(),
    )


def _kernel(batch_size: int) -> KernelSpec:
    return KernelSpec(
        "llm_step",
        1,
        1,
        request_shapes=tuple(
            KernelRequestShape(1, 17) for _ in range(batch_size)
        ),
    )


def test_surface_preserves_endpoints_and_decreases_service_per_request() -> None:
    points = _points()

    assert interpolate_batch_service_ps(points, 1) == 1_110_576_000
    assert interpolate_batch_service_ps(points, 8) == 1_892_831_500
    values = [interpolate_batch_service_ps(points, batch) for batch in range(1, 9)]
    assert values == sorted(values)
    assert [value / batch for batch, value in enumerate(values, start=1)] == sorted(
        (value / batch for batch, value in enumerate(values, start=1)),
        reverse=True,
    )


def test_provider_uses_measured_endpoints_and_interpolated_interior() -> None:
    provider = _provider()
    gpu = GPU_ENVELOPES["b100"]

    assert provider.estimate(_kernel(1), gpu) == DurationEstimate(
        1_110_576_000,
        "measured",
        0.004232,
    )
    middle = provider.estimate(_kernel(4), gpu)
    assert middle.duration_ps == interpolate_batch_service_ps(_points(), 4)
    assert middle.bound == "interpolated"
    assert middle.uncertainty == 0.15
    assert provider.estimate(_kernel(9), gpu).duration_ps == 77


def test_candidate_status_and_surface_keys_are_visible_where_they_price() -> None:
    provider = _provider()
    provider.estimate(_kernel(1), GPU_ENVELOPES["b100"])
    provider.estimate(_kernel(4), GPU_ENVELOPES["b100"])
    provenance = provider.pricing_provenance()

    assert provenance == {
        "schema": BATCH_SERVICE_PROVENANCE_SCHEMA,
        "record_sha256": RECORD_SHA256,
        "acceptance_status": "candidate",
        "campaign_id": "candidate-campaign",
        "coverage": "complete-kernel-stream",
        "record_device_kind_id": "nvidia-gh200-sm90",
        "pool": "decode",
        "surface_entry_key_sha256s": [KEY_ONE, KEY_EIGHT],
        "surface_batch_sizes": [1, 8],
        "surface_evidence_classes": ["MEASURED", "MEASURED"],
        "selected_batch_sizes": [1, 4],
        "lookup_hits": 2,
        "lookup_misses": 0,
        "exact_hits": 1,
        "interpolated_hits": 1,
        "calibration_claim": False,
    }


def test_provider_round_trips_through_spawn_serialization() -> None:
    restored = pickle.loads(pickle.dumps(_provider()))

    assert restored.estimate(_kernel(8), GPU_ENVELOPES["b100"]).duration_ps == (
        1_892_831_500
    )
