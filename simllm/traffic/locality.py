"""Placement-backed split of directed collective segments by physical locality.

The placement manifest remains the sole authority for whether two semantic
global ranks share a node. This module classifies already expanded directed
segments before any fabric rendering. Local segments use a declared flat
per-source NVLink serializer; remote segments remain inputs to the existing
fabric backend.

The first-cut service is deliberately analytic and uncalibrated. GOAL calc
units are whole nanoseconds, so one source's phase service is

    ceil(local_egress_bytes * 1e9 / bandwidth_bytes_per_second)

nanoseconds. No propagation term is added. TRAF-11 owns calibration against
same-generation measurements.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from simllm.placement import RankMapper

DEFAULT_NVLINK_BANDWIDTH_BYTES_PER_SECOND = 450_000_000_000


def _require_nonnegative_rank(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _require_positive_int(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


@dataclass(frozen=True)
class DirectedCollectiveSegment:
    """One positive point-to-point transfer after collective expansion."""

    source_rank: int
    destination_rank: int
    payload_bytes: int
    tag: int

    def __post_init__(self) -> None:
        _require_nonnegative_rank("source_rank", self.source_rank)
        _require_nonnegative_rank("destination_rank", self.destination_rank)
        if self.source_rank == self.destination_rank:
            raise ValueError("a directed collective segment cannot be a self-pair")
        _require_positive_int("payload_bytes", self.payload_bytes)
        _require_nonnegative_rank("tag", self.tag)


@dataclass(frozen=True)
class CollectiveCommunicationPhase:
    """One serial collective phase whose directed segments start together."""

    phase_id: str
    layer: int
    participants: tuple[int, ...]
    segments: tuple[DirectedCollectiveSegment, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "participants", tuple(self.participants))
        object.__setattr__(self, "segments", tuple(self.segments))
        if not isinstance(self.phase_id, str) or not self.phase_id.strip():
            raise ValueError("phase_id must be a nonblank string")
        _require_nonnegative_rank("layer", self.layer)
        if len(self.participants) < 2:
            raise ValueError("a communication phase needs at least two participants")
        if len(set(self.participants)) != len(self.participants):
            raise ValueError("communication phase participants must be unique")
        for rank in self.participants:
            _require_nonnegative_rank("participant rank", rank)
        participant_set = set(self.participants)
        if not self.segments:
            raise ValueError("a communication phase needs at least one segment")
        if any(
            not isinstance(segment, DirectedCollectiveSegment)
            for segment in self.segments
        ):
            raise TypeError("communication phase segments must be directed segments")
        if any(
            segment.source_rank not in participant_set
            or segment.destination_rank not in participant_set
            for segment in self.segments
        ):
            raise ValueError("communication phase segment rank is not a participant")


@dataclass(frozen=True)
class ClassifiedCommunicationPhase:
    """One phase split into fabric segments and analytic source service."""

    phase: CollectiveCommunicationPhase
    fabric_segments: tuple[DirectedCollectiveSegment, ...]
    nvlink_segments: tuple[DirectedCollectiveSegment, ...]
    #: semantic source rank to egress service in whole nanoseconds
    nvlink_source_service_ns: tuple[tuple[int, int], ...]
    #: maximum source serializer duration for this serial phase
    nvlink_service_ps: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "fabric_segments", tuple(self.fabric_segments))
        object.__setattr__(self, "nvlink_segments", tuple(self.nvlink_segments))
        object.__setattr__(
            self,
            "nvlink_source_service_ns",
            tuple(self.nvlink_source_service_ns),
        )
        if not isinstance(self.phase, CollectiveCommunicationPhase):
            raise TypeError("phase must be a CollectiveCommunicationPhase")
        raw = Counter(self.phase.segments)
        partition = Counter(self.fabric_segments) + Counter(self.nvlink_segments)
        if raw != partition:
            raise ValueError("classified segments must exactly partition the phase")
        for subset, name in (
            (self.fabric_segments, "fabric"),
            (self.nvlink_segments, "NVLink"),
        ):
            cursor = iter(self.phase.segments)
            if any(not any(candidate == segment for candidate in cursor) for segment in subset):
                raise ValueError(f"{name} segments must retain original phase order")
        services = self.nvlink_source_service_ns
        if tuple(sorted(services)) != services:
            raise ValueError("NVLink source services must be sorted by source rank")
        service_ranks = [rank for rank, _ in services]
        if len(service_ranks) != len(set(service_ranks)):
            raise ValueError("NVLink source services contain duplicate ranks")
        local_sources = {segment.source_rank for segment in self.nvlink_segments}
        if set(service_ranks) != local_sources:
            raise ValueError("NVLink source services do not cover the local sources")
        for rank, duration_ns in services:
            _require_nonnegative_rank("NVLink source rank", rank)
            _require_positive_int("NVLink source service", duration_ns)
        _require_nonnegative_rank("nvlink_service_ps", self.nvlink_service_ps)
        expected_phase_ps = max((duration for _, duration in services), default=0) * 1000
        if self.nvlink_service_ps != expected_phase_ps:
            raise ValueError("nvlink_service_ps disagrees with source service")

    @property
    def fabric_bytes(self) -> int:
        return sum(segment.payload_bytes for segment in self.fabric_segments)

    @property
    def nvlink_bytes(self) -> int:
        return sum(segment.payload_bytes for segment in self.nvlink_segments)


@dataclass(frozen=True)
class StepLocalityPlan:
    """Immutable locality classification for every communication phase."""

    phases: tuple[ClassifiedCommunicationPhase, ...]
    authority: str
    nvlink_bandwidth_bytes_per_second: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "phases", tuple(self.phases))
        if any(
            not isinstance(phase, ClassifiedCommunicationPhase)
            for phase in self.phases
        ):
            raise TypeError("locality plan phases must be classified phases")
        if self.authority not in {"placement-manifest", "compatibility-all-remote"}:
            raise ValueError("unsupported locality authority")
        _require_positive_int(
            "nvlink_bandwidth_bytes_per_second",
            self.nvlink_bandwidth_bytes_per_second,
        )

    @property
    def total_directed_bytes(self) -> int:
        return sum(
            segment.payload_bytes
            for phase in self.phases
            for segment in phase.phase.segments
        )

    @property
    def fabric_bytes(self) -> int:
        return sum(phase.fabric_bytes for phase in self.phases)

    @property
    def nvlink_bytes(self) -> int:
        return sum(phase.nvlink_bytes for phase in self.phases)

    @property
    def fabric_segments(self) -> int:
        return sum(len(phase.fabric_segments) for phase in self.phases)

    @property
    def nvlink_segments(self) -> int:
        return sum(len(phase.nvlink_segments) for phase in self.phases)

    @property
    def nvlink_service_ps(self) -> int:
        return sum(phase.nvlink_service_ps for phase in self.phases)


def _classify_phase(
    phase: CollectiveCommunicationPhase,
    *,
    rank_mapper: RankMapper | None,
    bandwidth_bytes_per_second: int,
) -> ClassifiedCommunicationPhase:
    if rank_mapper is None:
        return ClassifiedCommunicationPhase(
            phase=phase,
            fabric_segments=phase.segments,
            nvlink_segments=(),
            nvlink_source_service_ns=(),
            nvlink_service_ps=0,
        )

    for rank in phase.participants:
        rank_mapper.goal_rank(rank)
    fabric = []
    local = []
    for segment in phase.segments:
        if rank_mapper.is_intra_node(
            segment.source_rank,
            segment.destination_rank,
        ):
            local.append(segment)
        else:
            fabric.append(segment)

    source_egress_bytes: dict[int, int] = {}
    for segment in local:
        source_egress_bytes[segment.source_rank] = (
            source_egress_bytes.get(segment.source_rank, 0) + segment.payload_bytes
        )
    source_service_ns = {
        rank: _ceil_div(payload_bytes * 1_000_000_000, bandwidth_bytes_per_second)
        for rank, payload_bytes in source_egress_bytes.items()
    }

    return ClassifiedCommunicationPhase(
        phase=phase,
        fabric_segments=tuple(fabric),
        nvlink_segments=tuple(local),
        nvlink_source_service_ns=tuple(sorted(source_service_ns.items())),
        nvlink_service_ps=max(source_service_ns.values(), default=0) * 1000,
    )


def classify_step_locality(
    phases: tuple[CollectiveCommunicationPhase, ...],
    *,
    rank_mapper: RankMapper | None,
    bandwidth_bytes_per_second: int = DEFAULT_NVLINK_BANDWIDTH_BYTES_PER_SECOND,
) -> StepLocalityPlan:
    """Classify phases without mutating a renderer or runtime output path."""

    _require_positive_int(
        "bandwidth_bytes_per_second",
        bandwidth_bytes_per_second,
    )
    if rank_mapper is not None and not isinstance(rank_mapper, RankMapper):
        raise TypeError("rank_mapper must be RankMapper or None")
    authority = (
        "placement-manifest"
        if rank_mapper is not None
        else "compatibility-all-remote"
    )
    plan = StepLocalityPlan(
        phases=tuple(
            _classify_phase(
                phase,
                rank_mapper=rank_mapper,
                bandwidth_bytes_per_second=bandwidth_bytes_per_second,
            )
            for phase in phases
        ),
        authority=authority,
        nvlink_bandwidth_bytes_per_second=bandwidth_bytes_per_second,
    )
    for classified in plan.phases:
        raw = classified.phase.segments
        fabric_ids = {id(segment) for segment in classified.fabric_segments}
        local_ids = {id(segment) for segment in classified.nvlink_segments}
        if fabric_ids & local_ids or fabric_ids | local_ids != {
            id(segment) for segment in raw
        }:
            raise AssertionError("locality classification is not an exact partition")
        if tuple(
            segment
            for segment in raw
            if id(segment) in fabric_ids
        ) != classified.fabric_segments:
            raise AssertionError("fabric classification changed segment order")
        if tuple(
            segment
            for segment in raw
            if id(segment) in local_ids
        ) != classified.nvlink_segments:
            raise AssertionError("NVLink classification changed segment order")
    if plan.fabric_bytes + plan.nvlink_bytes != plan.total_directed_bytes:
        raise AssertionError("locality classification does not conserve directed bytes")
    return plan


__all__ = [
    "DEFAULT_NVLINK_BANDWIDTH_BYTES_PER_SECOND",
    "ClassifiedCommunicationPhase",
    "CollectiveCommunicationPhase",
    "DirectedCollectiveSegment",
    "StepLocalityPlan",
    "classify_step_locality",
]
