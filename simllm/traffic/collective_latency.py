"""Calibrated fixed-latency and endpoint-serialization collective model."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from simllm.core.execution import CollectiveWork

PICOSECONDS_PER_SECOND = 1_000_000_000_000
LEGACY_COLLECTIVE_LATENCY_PROFILE = "legacy"


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _ring_endpoint_bytes(source_payload_bytes: int, participant_count: int) -> int:
    """Return one rank's bytes over the current ring plan expansion."""

    if source_payload_bytes == 0:
        return 0
    chunk_bytes = max(1, source_payload_bytes // participant_count)
    return 2 * (participant_count - 1) * chunk_bytes


@dataclass(frozen=True)
class CollectiveLatencyProfile:
    """One immutable calibration for semantic collective service.

    ``participant_latency_ps`` is a direct table because the source capture
    identifies participant widths 2, 4 and 8, but does not identify an
    interpolation law. Endpoint-byte validity is bounded by the endpoint
    loads produced by the source all-reduce payload interval.
    """

    profile_id: str
    bandwidth_bytes_per_second: int
    participant_latency_ps: tuple[tuple[int, int], ...]
    source_payload_bytes_min: int
    source_payload_bytes_max: int
    propagation_reference_ps: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "participant_latency_ps",
            tuple(tuple(entry) for entry in self.participant_latency_ps),
        )
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("profile_id must be a nonblank string")
        _require_int(
            "bandwidth_bytes_per_second",
            self.bandwidth_bytes_per_second,
            minimum=1,
        )
        source_min = _require_int(
            "source_payload_bytes_min",
            self.source_payload_bytes_min,
            minimum=1,
        )
        source_max = _require_int(
            "source_payload_bytes_max",
            self.source_payload_bytes_max,
            minimum=1,
        )
        if source_max < source_min:
            raise ValueError(
                "source_payload_bytes_max must not be smaller than "
                "source_payload_bytes_min"
            )
        _require_int(
            "propagation_reference_ps",
            self.propagation_reference_ps,
            minimum=0,
        )
        if not self.participant_latency_ps:
            raise ValueError("participant_latency_ps must not be empty")
        widths: list[int] = []
        for index, entry in enumerate(self.participant_latency_ps):
            if len(entry) != 2:
                raise ValueError(
                    f"participant_latency_ps[{index}] must be a two-item tuple"
                )
            width = _require_int(
                f"participant_latency_ps[{index}][0]",
                entry[0],
                minimum=2,
            )
            _require_int(
                f"participant_latency_ps[{index}][1]",
                entry[1],
                minimum=0,
            )
            widths.append(width)
        if widths != sorted(set(widths)):
            raise ValueError(
                "participant_latency_ps widths must be unique and increasing"
            )

    @property
    def supported_participant_counts(self) -> tuple[int, ...]:
        """Return the directly calibrated participant widths."""

        return tuple(width for width, _ in self.participant_latency_ps)

    def base_latency_ps(self, participant_count: int) -> int:
        """Return the non-serialization floor for ``participant_count``."""

        _require_int("participant_count", participant_count, minimum=2)
        for width, latency_ps in self.participant_latency_ps:
            if width == participant_count:
                return latency_ps
        supported = ", ".join(str(width) for width in self.supported_participant_counts)
        raise ValueError(
            f"profile {self.profile_id!r} does not support participant count "
            f"{participant_count}; supported counts are {supported}"
        )

    def endpoint_byte_bounds(self, participant_count: int) -> tuple[int, int]:
        """Return the fitted endpoint-byte envelope for one width."""

        self.base_latency_ps(participant_count)
        return (
            _ring_endpoint_bytes(self.source_payload_bytes_min, participant_count),
            _ring_endpoint_bytes(self.source_payload_bytes_max, participant_count),
        )

    def validate_endpoint_bytes(
        self,
        participant_count: int,
        endpoint_bytes: int,
    ) -> int:
        """Validate and return an endpoint load within the source envelope."""

        endpoint_bytes = _require_int("endpoint_bytes", endpoint_bytes, minimum=1)
        minimum, maximum = self.endpoint_byte_bounds(participant_count)
        if not minimum <= endpoint_bytes <= maximum:
            raise ValueError(
                f"endpoint_bytes {endpoint_bytes} is outside profile "
                f"{self.profile_id!r} envelope [{minimum}, {maximum}] for "
                f"participant count {participant_count}"
            )
        return endpoint_bytes

    def endpoint_serialization_ps(
        self,
        participant_count: int,
        endpoint_bytes: int,
    ) -> int:
        """Return exact upward-rounded endpoint serialization picoseconds."""

        endpoint_bytes = self.validate_endpoint_bytes(
            participant_count,
            endpoint_bytes,
        )
        return _ceil_div(
            endpoint_bytes * PICOSECONDS_PER_SECOND,
            self.bandwidth_bytes_per_second,
        )

    def total_service_ps(self, participant_count: int, endpoint_bytes: int) -> int:
        """Return one base latency plus exact endpoint serialization service."""

        return self.base_latency_ps(
            participant_count
        ) + self.endpoint_serialization_ps(participant_count, endpoint_bytes)


B200_NCCL_2_27_LOCAL_PROFILE = CollectiveLatencyProfile(
    profile_id="b200-nccl-2.27-local-v1",
    bandwidth_bytes_per_second=70_027_079_100,
    participant_latency_ps=(
        (2, 10_722_112),
        (4, 15_745_167),
        (8, 30_128_029),
    ),
    source_payload_bytes_min=8,
    source_payload_bytes_max=262_144,
    propagation_reference_ps=2_000_000,
)


def critical_collective_endpoint_bytes(work: CollectiveWork) -> int:
    """Return the largest full-operation endpoint load for supported work.

    Ring all-reduce follows the current traffic-owned chunk expansion exactly.
    Pairwise all-to-allv accumulates complete-operation egress and ingress per
    rank, so asymmetric sparse dispatch and combine tables are both covered.
    """

    if not isinstance(work, CollectiveWork):
        raise TypeError("work must be a CollectiveWork")
    if not isinstance(work.ranks, tuple):
        raise TypeError("work.ranks must be a tuple")
    if len(work.ranks) < 2:
        raise ValueError("collective work needs at least two ranks")
    for index, rank in enumerate(work.ranks):
        _require_int(f"work.ranks[{index}]", rank, minimum=0)
    if len(set(work.ranks)) != len(work.ranks):
        raise ValueError("collective work ranks must be unique")
    payload_bytes = _require_int("work.payload_bytes", work.payload_bytes, minimum=0)

    key = (work.collective, work.algorithm_hint)
    if key == ("all-reduce", "ring"):
        if work.pair_payload_bytes or work.request_pair_payload_bytes:
            raise ValueError("ring all-reduce does not accept pair payload tables")
        return _ring_endpoint_bytes(payload_bytes, len(work.ranks))
    if key != ("all-to-allv", "pairwise"):
        raise ValueError(
            "supported collectives are ring all-reduce and pairwise all-to-allv"
        )

    if not work.pair_payload_bytes:
        if work.request_pair_payload_bytes:
            raise ValueError(
                "request_pair_payload_bytes needs an aggregate pair payload table"
            )
        if payload_bytes == 0:
            return 0
        return (len(work.ranks) - 1) * payload_bytes
    if payload_bytes != 0:
        raise ValueError(
            "sparse pair_payload_bytes and uniform payload_bytes cannot both be set"
        )

    rank_set = set(work.ranks)
    egress: dict[int, int] = defaultdict(int)
    ingress: dict[int, int] = defaultdict(int)
    pair_keys: list[tuple[int, int]] = []
    for index, entry in enumerate(work.pair_payload_bytes):
        if not isinstance(entry, tuple) or len(entry) != 3:
            raise TypeError(f"work.pair_payload_bytes[{index}] must be a triple")
        source = _require_int(
            f"work.pair_payload_bytes[{index}][0]",
            entry[0],
            minimum=0,
        )
        destination = _require_int(
            f"work.pair_payload_bytes[{index}][1]",
            entry[1],
            minimum=0,
        )
        size = _require_int(
            f"work.pair_payload_bytes[{index}][2]",
            entry[2],
            minimum=1,
        )
        if source == destination:
            raise ValueError("sparse pair payloads cannot contain self-pairs")
        if source not in rank_set or destination not in rank_set:
            raise ValueError("sparse pair payload ranks must belong to work.ranks")
        pair_keys.append((source, destination))
        egress[source] += size
        ingress[destination] += size
    if pair_keys != sorted(set(pair_keys)):
        raise ValueError("sparse pair payloads must be unique and source-major")
    return max(
        max(egress.values(), default=0),
        max(ingress.values(), default=0),
    )


def resolve_collective_latency_profile(
    selector: str | CollectiveLatencyProfile | None,
) -> CollectiveLatencyProfile | None:
    """Resolve the explicit legacy off path or a calibrated profile."""

    if selector is None:
        return None
    if isinstance(selector, CollectiveLatencyProfile):
        return selector
    if isinstance(selector, str):
        if selector == LEGACY_COLLECTIVE_LATENCY_PROFILE:
            return None
        if selector == B200_NCCL_2_27_LOCAL_PROFILE.profile_id:
            return B200_NCCL_2_27_LOCAL_PROFILE
        raise ValueError(f"unknown collective latency profile {selector!r}")
    raise TypeError(
        "collective latency profile must be None, a selector string, or a "
        "CollectiveLatencyProfile"
    )


__all__ = [
    "B200_NCCL_2_27_LOCAL_PROFILE",
    "LEGACY_COLLECTIVE_LATENCY_PROFILE",
    "CollectiveLatencyProfile",
    "critical_collective_endpoint_bytes",
    "resolve_collective_latency_profile",
]
