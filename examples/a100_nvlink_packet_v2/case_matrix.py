"""Deterministic parameter points for the frozen TRAF-70 case catalog."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True, kw_only=True)
class SweepPoint:
    case_name: str
    point_id: str
    producer: str
    payload_bytes: int
    message_count: int
    source: int = 0
    destination: int = 1
    sources: str = "0"
    destinations: str = "1"
    source_alignment: int = 0
    destination_alignment: int = 0
    access_width: int = 16
    active_lanes: int = 32
    lane_mask: str = "contiguous"
    stride: int = 1
    stream_count: int = 1
    outstanding: int = 256
    burst_messages: int = 256
    gap_ns: int = 0
    offered_rate_percent: int = 100
    pattern: str = "unidirectional"

    def __post_init__(self) -> None:
        for name in ("case_name", "point_id", "producer", "sources", "destinations"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "payload_bytes",
            "message_count",
            "access_width",
            "active_lanes",
            "stride",
            "stream_count",
            "outstanding",
            "burst_messages",
            "offered_rate_percent",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("source_alignment", "destination_alignment", "gap_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


TSV_FIELDS = tuple(SweepPoint.__dataclass_fields__)


def points_for_case(case: dict[str, Any]) -> tuple[SweepPoint, ...]:
    """Return every frozen point for one case before producer expansion."""

    ordinal = int(case["ordinal"])
    name = str(case["stable_name"])
    base = SweepPoint(
        case_name=name,
        point_id=f"{name}:base",
        producer="unassigned",
        payload_bytes=256,
        message_count=_message_count(256),
    )

    if ordinal == 1:
        return _payload_points(base, range(1, 513))
    if ordinal == 2:
        payloads = sorted(
            {max(1, boundary + delta) for boundary in range(256, 4097, 256) for delta in (-1, 0, 1)}
        )
        return _payload_points(base, payloads)
    if ordinal == 3:
        payloads = sorted(
            {
                256 * multiple + residual
                for multiple in (1, 2, 4, 16, 256)
                for residual in (0, 1, 15, 16, 127, 255)
            }
        )
        return _payload_points(base, payloads)
    if ordinal in (4, 5):
        points = []
        field = "destination_alignment" if ordinal == 4 else "source_alignment"
        for alignment in (0, 1, 2, 4, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255):
            points.append(
                replace(
                    base,
                    point_id=f"{name}:{field}={alignment}",
                    **{field: alignment},
                )
            )
        return tuple(points)
    if ordinal == 6:
        return tuple(
            replace(base, point_id=f"{name}:access_width={width}", access_width=width)
            for width in (1, 2, 4, 8, 16)
        )
    if ordinal == 7:
        return tuple(
            replace(base, point_id=f"{name}:active_lanes={lanes}", active_lanes=lanes)
            for lanes in (1, 2, 4, 8, 16, 32)
        )
    if ordinal == 8:
        return tuple(
            replace(
                base,
                point_id=f"{name}:mask={mask}",
                active_lanes=16,
                lane_mask=mask,
            )
            for mask in ("contiguous", "alternating", "split", "seeded")
        )
    if ordinal == 9:
        return tuple(
            replace(base, point_id=f"{name}:stride={stride}", stride=stride)
            for stride in (1, 2, 4, 8, 16, 32, 64)
        )
    if ordinal == 10:
        return tuple(
            replace(
                base,
                point_id=f"{name}:message={payload}",
                payload_bytes=payload,
                message_count=max(1, (16 << 20) // payload),
            )
            for payload in (16, 64, 256, 1024, 4096, 65536, 1 << 20)
        )
    if ordinal == 11:
        return tuple(
            replace(
                base,
                point_id=f"{name}:count={count}",
                payload_bytes=256 << 10,
                message_count=count,
            )
            for count in (1, 2, 4, 16, 64, 256, 1024)
        )
    if ordinal == 12:
        return tuple(
            replace(base, point_id=f"{name}:{pattern}", pattern=pattern)
            for pattern in ("address_reuse", "address_separation")
        )
    if ordinal in (13, 14):
        points = []
        for payload in (1, 15, 16, 31, 32, 255, 256, 257, 4096):
            for alignment in (0, 1, 16, 255):
                points.append(
                    replace(
                        base,
                        point_id=f"{name}:bytes={payload}:alignment={alignment}",
                        payload_bytes=payload,
                        message_count=_message_count(payload),
                        destination_alignment=alignment,
                    )
                )
        return tuple(points)
    if ordinal == 15:
        return (base,)
    if ordinal == 16:
        rng = random.Random(650016)
        return tuple(
            replace(
                base,
                point_id=f"{name}:blind={index:02d}",
                payload_bytes=rng.randint(1, 4096),
                source_alignment=rng.randint(0, 255),
                destination_alignment=rng.randint(0, 255),
                active_lanes=rng.choice((1, 2, 4, 8, 16, 32)),
                lane_mask=rng.choice(("contiguous", "alternating", "split", "seeded")),
            )
            for index in range(32)
        )

    if ordinal == 17:
        return tuple(
            replace(
                base,
                point_id=f"{name}:{source}->{destination}",
                source=source,
                destination=destination,
                sources=str(source),
                destinations=str(destination),
            )
            for source in range(4)
            for destination in range(4)
            if source != destination
        )
    if ordinal == 18:
        return tuple(
            replace(
                base,
                point_id=f"{name}:pair=0->{destination}",
                destination=destination,
                destinations=str(destination),
            )
            for destination in (1, 2, 3)
        )
    if ordinal == 19:
        return _vary_int(base, "stream_count", (1, 2, 4, 8, 16))
    if ordinal == 20:
        return tuple(
            replace(
                base,
                point_id=f"{name}:sources={count}",
                sources=",".join(str(value) for value in range(count)),
                source=0,
                destination=3,
                destinations="3",
                pattern="producer_concurrency",
            )
            for count in (1, 2, 3)
        )
    if ordinal == 21:
        return _payload_points(base, (16, 64, 256, 1024, 4096, 65536, 1 << 20))
    if ordinal == 22:
        return _vary_int(base, "offered_rate_percent", (10, 25, 50, 75, 90, 95, 100))
    if ordinal == 23:
        return _vary_int(base, "burst_messages", (1, 2, 4, 16, 64, 256, 1024))
    if ordinal == 24:
        return tuple(
            replace(
                base,
                point_id=f"{name}:destinations={count}",
                destinations=",".join(str(value) for value in range(1, count + 1)),
                pattern="source_fanout",
            )
            for count in (1, 2, 3)
        )
    if ordinal == 25:
        return tuple(
            replace(
                base,
                point_id=f"{name}:sources={count}",
                sources=",".join(str(value) for value in range(1, count + 1)),
                source=1,
                destination=0,
                destinations="0",
                pattern="destination_fanin",
            )
            for count in (1, 2, 3)
        )
    if ordinal in (26, 27):
        return tuple(
            replace(
                base,
                point_id=f"{name}:ratio={ratio}",
                pattern="symmetric_bidirectional" if ordinal == 26 else "asymmetric_bidirectional",
                offered_rate_percent=ratio,
            )
            for ratio in (25, 50, 75, 100)
        )
    if ordinal in (28, 29):
        return (
            replace(
                base,
                point_id=f"{name}:pairs=0-1,2-3",
                sources="0,2",
                destinations="1,3",
                pattern="disjoint_bidirectional" if ordinal == 29 else "disjoint_unidirectional",
            ),
        )
    if ordinal == 30:
        return (
            replace(
                base,
                point_id=f"{name}:all",
                sources="0,1,2,3",
                destinations="0,1,2,3",
                pattern="full_mesh",
            ),
        )
    if ordinal == 31:
        return tuple(
            replace(base, point_id=f"{name}:{state}", pattern=state) for state in ("cold", "warm")
        )
    if ordinal == 32:
        return tuple(
            replace(base, point_id=f"{name}:repeat={repeat}", pattern="repeatability")
            for repeat in range(5)
        )

    if 33 <= ordinal <= 48:
        source_counts = (1,) if ordinal == 33 else (2,) if ordinal == 34 else (3,)
        if ordinal >= 36:
            source_counts = (1, 2, 3)
        points = []
        for count in source_counts:
            point = replace(
                base,
                point_id=f"{name}:sources={count}",
                source=1,
                destination=0,
                sources=",".join(str(value) for value in range(1, count + 1)),
                destinations="0",
                pattern=_incast_pattern(ordinal),
            )
            if ordinal == 37:
                for offered in (25, 50, 75, 100):
                    points.append(
                        replace(
                            point,
                            point_id=f"{point.point_id}:offered={offered}",
                            offered_rate_percent=offered,
                        )
                    )
            elif ordinal == 38:
                for gap in (0, 50, 100, 500, 1000):
                    points.append(
                        replace(point, point_id=f"{point.point_id}:skew={gap}", gap_ns=gap)
                    )
            elif ordinal == 40:
                for burst in (1, 4, 16, 64, 256, 1024):
                    points.append(
                        replace(
                            point,
                            point_id=f"{point.point_id}:burst={burst}",
                            burst_messages=burst,
                        )
                    )
            else:
                points.append(point)
        return tuple(points)

    if ordinal == 49:
        return _payload_points(base, (16, 32, 64, 128, 256, 512, 4096))
    if 50 <= ordinal <= 54:
        payload = (16, 32, 64, 128, 256)[ordinal - 50]
        return tuple(
            replace(
                base,
                point_id=f"{name}:outstanding={outstanding}",
                payload_bytes=payload,
                message_count=_message_count(payload),
                outstanding=outstanding,
            )
            for outstanding in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
        )
    if ordinal in (55, 56):
        return tuple(
            replace(
                base,
                point_id=f"{name}:outstanding={outstanding}",
                outstanding=outstanding,
                pattern="outstanding_read" if ordinal == 55 else "outstanding_atomic",
            )
            for outstanding in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        )
    if ordinal == 57:
        return _vary_int(base, "burst_messages", (1, 2, 4, 8, 16, 64, 256, 1024))
    if ordinal in (58, 59):
        return _vary_int(base, "gap_ns", (0, 20, 50, 100, 200, 500, 1000, 5000))
    if ordinal == 60:
        return _vary_int(base, "offered_rate_percent", (10, 25, 50, 75, 90, 95, 100))
    if ordinal == 61:
        return tuple(
            replace(
                base,
                point_id=f"{name}:outstanding={outstanding}",
                outstanding=outstanding,
                pattern="adaptive_knee_zoom",
            )
            for outstanding in range(1, 513)
        )
    if ordinal == 62:
        return _vary_int(base, "stream_count", (1, 2))
    if ordinal == 63:
        return tuple(
            replace(
                base,
                point_id=f"{name}:destinations={count}",
                destinations="1" if count == 1 else "1,2",
                pattern="one_source_two_peers",
            )
            for count in (1, 2)
        )
    if ordinal == 64:
        return (
            replace(base, point_id=f"{name}:one-way", pattern="unidirectional"),
            replace(base, point_id=f"{name}:opposite", pattern="opposite_directions"),
        )

    patterns = {
        65: "small_behind_large",
        66: "large_behind_small",
        67: "separate_streams",
        68: "alternating_sizes",
        69: "bimodal_mix",
        70: "same_pair_bulk",
        71: "other_peer_bulk",
        72: "remote_incast",
        73: "write_write",
        74: "read_read",
        75: "same_direction_read_write",
        76: "opposite_direction_read_write",
        77: "distinct_regions",
        78: "shared_cache_line",
        79: "post_burst_drain",
        80: "blind_mixed_soak",
    }
    if ordinal == 80:
        rng = random.Random(650080)
        points = []
        for index in range(32):
            source = rng.randrange(4)
            destination = (source + rng.randrange(1, 4)) % 4
            points.append(
                replace(
                    base,
                    point_id=f"{name}:blind={index:02d}",
                    payload_bytes=(payload := rng.choice((16, 64, 256, 4096, 1 << 20))),
                    message_count=_message_count(payload),
                    source=source,
                    destination=destination,
                    sources=str(source),
                    destinations=str(destination),
                    pattern=patterns[ordinal],
                )
            )
        return tuple(points)
    return tuple(
        replace(
            base,
            point_id=f"{name}:payload={payload}",
            payload_bytes=payload,
            message_count=_message_count(payload),
            pattern=patterns[ordinal],
            stream_count=2 if ordinal in (67, 70, 71, 72) else 1,
        )
        for payload in (64, 256, 4096, 1 << 20)
    )


def expand_producers(
    case: dict[str, Any], points: tuple[SweepPoint, ...]
) -> tuple[SweepPoint, ...]:
    """Cross one case's points with exactly its frozen producer classes."""

    return tuple(
        replace(point, producer=str(producer), point_id=f"{point.point_id}:{producer}")
        for producer in case["producer_classes"]
        for point in points
    )


def protocol_validation_points(case_name: str) -> tuple[SweepPoint, ...]:
    """NCCL send and receive checks, kept outside packet-format authority."""

    return tuple(
        SweepPoint(
            case_name=case_name,
            point_id=f"{case_name}:nccl_send_receive:{payload}",
            producer="nccl_send_receive_validation",
            payload_bytes=payload,
            message_count=32,
            pattern="protocol_validation_only",
        )
        for payload in (8, 256, 4096, 1 << 20)
    )


def point_to_tsv_row(point: SweepPoint) -> tuple[str, ...]:
    return tuple(str(getattr(point, field)) for field in TSV_FIELDS)


def _payload_points(base: SweepPoint, payloads: object) -> tuple[SweepPoint, ...]:
    return tuple(
        replace(
            base,
            point_id=f"{base.case_name}:bytes={int(payload)}",
            payload_bytes=int(payload),
            message_count=_message_count(int(payload)),
        )
        for payload in payloads
    )


def _vary_int(base: SweepPoint, field: str, values: tuple[int, ...]) -> tuple[SweepPoint, ...]:
    return tuple(
        replace(
            base,
            point_id=f"{base.case_name}:{field}={value}",
            **{field: value},
        )
        for value in values
    )


def _message_count(payload_bytes: int) -> int:
    return max(32, min(1 << 20, (1 << 20) // payload_bytes))


def _incast_pattern(ordinal: int) -> str:
    return {
        33: "one_source",
        34: "two_source_simultaneous",
        35: "three_source_simultaneous",
        36: "fixed_aggregate_rate",
        37: "per_source_rate",
        38: "start_skew",
        39: "join_leave",
        40: "burst_depth",
        41: "equal_message_size",
        42: "unequal_message_size",
        43: "one_elephant_two_mice",
        44: "two_elephants_one_mouse",
        45: "push_distinct_buffers",
        46: "pull_gather",
        47: "hot_destination",
        48: "long_soak",
    }[ordinal]
