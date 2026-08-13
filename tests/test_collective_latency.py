"""Focused tests for the calibrated collective latency profile."""

from dataclasses import FrozenInstanceError

import pytest

from simllm.core import CollectiveWork
from simllm.traffic.collective_latency import (
    B200_NCCL_2_27_LOCAL_PROFILE,
    LEGACY_COLLECTIVE_LATENCY_PROFILE,
    CollectiveLatencyProfile,
    critical_collective_endpoint_bytes,
    resolve_collective_latency_profile,
)


@pytest.mark.parametrize(
    ("participant_count", "endpoint_bytes", "serialization_ps", "service_ps"),
    (
        (2, 4_096, 58_492, 10_780_604),
        (4, 6_144, 87_738, 15_832_905),
        (8, 7_168, 102_361, 30_230_390),
    ),
)
def test_frozen_heldout_predictions_use_exact_ceil_picoseconds(
    participant_count: int,
    endpoint_bytes: int,
    serialization_ps: int,
    service_ps: int,
):
    profile = B200_NCCL_2_27_LOCAL_PROFILE

    assert profile.endpoint_serialization_ps(participant_count, endpoint_bytes) == (
        serialization_ps
    )
    assert profile.total_service_ps(participant_count, endpoint_bytes) == service_ps


def test_profile_exposes_frozen_identity_and_allreduce_endpoint_envelopes():
    profile = B200_NCCL_2_27_LOCAL_PROFILE

    assert profile.profile_id == "b200-nccl-2.27-local-v1"
    assert profile.bandwidth_bytes_per_second == 70_027_079_100
    assert profile.participant_latency_ps == (
        (2, 10_722_112),
        (4, 15_745_167),
        (8, 30_128_029),
    )
    assert profile.supported_participant_counts == (2, 4, 8)
    assert profile.propagation_reference_ps == 2_000_000
    assert profile.endpoint_byte_bounds(2) == (8, 262_144)
    assert profile.endpoint_byte_bounds(4) == (12, 393_216)
    assert profile.endpoint_byte_bounds(8) == (14, 458_752)


@pytest.mark.parametrize("endpoint_bytes", (True, 0, -1))
def test_profile_rejects_nonpositive_and_boolean_endpoint_bytes(endpoint_bytes: object):
    with pytest.raises(ValueError, match="endpoint_bytes"):
        B200_NCCL_2_27_LOCAL_PROFILE.endpoint_serialization_ps(8, endpoint_bytes)  # type: ignore[arg-type]


@pytest.mark.parametrize("participant_count", (True, 1, 3, 16))
def test_profile_rejects_invalid_or_unsupported_participant_widths(
    participant_count: object,
):
    with pytest.raises(ValueError, match="participant_count|participant count"):
        B200_NCCL_2_27_LOCAL_PROFILE.base_latency_ps(participant_count)  # type: ignore[arg-type]


@pytest.mark.parametrize("endpoint_bytes", (13, 458_753))
def test_profile_rejects_endpoint_bytes_outside_fitted_envelope(endpoint_bytes: int):
    with pytest.raises(ValueError, match="outside.*envelope"):
        B200_NCCL_2_27_LOCAL_PROFILE.total_service_ps(8, endpoint_bytes)


def test_critical_endpoint_bytes_follow_actual_ring_chunk_expansion():
    assert critical_collective_endpoint_bytes(
        CollectiveWork("all-reduce", tuple(range(8)), 4_096, "ring")
    ) == 7_168
    assert critical_collective_endpoint_bytes(
        CollectiveWork("all-reduce", tuple(range(8)), 9, "ring")
    ) == 14
    assert critical_collective_endpoint_bytes(
        CollectiveWork("all-reduce", (0, 1), 0, "ring")
    ) == 0


def test_critical_endpoint_bytes_cover_uniform_and_sparse_pairwise_work():
    assert critical_collective_endpoint_bytes(
        CollectiveWork("all-to-allv", (0, 1, 2, 3), 7, "pairwise")
    ) == 21
    assert critical_collective_endpoint_bytes(
        CollectiveWork(
            "all-to-allv",
            (0, 1, 2, 3),
            0,
            "pairwise",
            pair_payload_bytes=(
                (0, 1, 5),
                (0, 2, 7),
                (2, 0, 13),
                (3, 0, 11),
            ),
        )
    ) == 24
    assert critical_collective_endpoint_bytes(
        CollectiveWork("all-to-allv", (0, 1), 0, "pairwise")
    ) == 0


@pytest.mark.parametrize(
    "work",
    (
        CollectiveWork("broadcast", (0, 1), 8, "tree"),
        CollectiveWork("all-reduce", (0, 1), 8, "tree"),
        CollectiveWork("all-to-allv", (0, 1), 8, "ring"),
    ),
)
def test_critical_endpoint_bytes_refuse_unsupported_collectives_and_algorithms(
    work: CollectiveWork,
):
    with pytest.raises(ValueError, match="supported collectives"):
        critical_collective_endpoint_bytes(work)


def test_critical_endpoint_bytes_validates_sparse_pair_tables():
    with pytest.raises(ValueError, match="self-pairs"):
        critical_collective_endpoint_bytes(
            CollectiveWork(
                "all-to-allv",
                (0, 1),
                0,
                "pairwise",
                pair_payload_bytes=((0, 0, 8),),
            )
        )
    with pytest.raises(ValueError, match="source-major"):
        critical_collective_endpoint_bytes(
            CollectiveWork(
                "all-to-allv",
                (0, 1, 2),
                0,
                "pairwise",
                pair_payload_bytes=((1, 0, 8), (0, 1, 8)),
            )
        )


def test_profile_is_immutable_and_normalizes_nested_tuple_storage():
    profile = CollectiveLatencyProfile(
        profile_id="test-profile",
        bandwidth_bytes_per_second=1,
        participant_latency_ps=[(2, 3)],  # type: ignore[arg-type]
        source_payload_bytes_min=8,
        source_payload_bytes_max=16,
        propagation_reference_ps=0,
    )

    assert profile.participant_latency_ps == ((2, 3),)
    with pytest.raises(FrozenInstanceError):
        profile.profile_id = "changed"  # type: ignore[misc]


def test_profile_resolver_preserves_explicit_identity_and_legacy_off_path():
    profile = B200_NCCL_2_27_LOCAL_PROFILE

    assert resolve_collective_latency_profile(None) is None
    assert resolve_collective_latency_profile(LEGACY_COLLECTIVE_LATENCY_PROFILE) is None
    assert resolve_collective_latency_profile(profile.profile_id) is profile
    assert resolve_collective_latency_profile(profile) is profile
    with pytest.raises(ValueError, match="unknown collective latency profile"):
        resolve_collective_latency_profile("unknown")
    with pytest.raises(TypeError, match="must be None"):
        resolve_collective_latency_profile(1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    (
        {"bandwidth_bytes_per_second": True},
        {"participant_latency_ps": ((2, 1), (2, 2))},
        {"source_payload_bytes_min": 17, "source_payload_bytes_max": 16},
        {"propagation_reference_ps": -1},
    ),
)
def test_profile_constructor_rejects_invalid_calibration(overrides: dict[str, object]):
    values: dict[str, object] = {
        "profile_id": "invalid",
        "bandwidth_bytes_per_second": 1,
        "participant_latency_ps": ((2, 1),),
        "source_payload_bytes_min": 8,
        "source_payload_bytes_max": 16,
        "propagation_reference_ps": 0,
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        CollectiveLatencyProfile(**values)  # type: ignore[arg-type]
