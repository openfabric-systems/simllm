"""Calibrated fixed-latency and endpoint-serialization collective model.

A profile's ``participant_latency_ps`` table is a *surcharge* added on top of
whatever transport the backend prices, not a complete service time. The step
sink composes one artifact as ``base_latency + max(local_service,
fabric_transport)``, so a profile that charges nothing still leaves the
backend's own per-collective fixed cost in place, which for the fluid null
network is one propagation delay named by ``propagation_reference_ps``.

:class:`CollectiveFixedCostEnvelope` turns the choice of that surcharge from a
silent default into a named bracket with three arms: ``off`` charges no
surcharge at all, ``lower`` charges the envelope's lower-bound profile and
``upper`` charges its upper-bound profile. A study runs the arms it wants and
publishes the bracket instead of a single unattributed number.

Every profile that participates in an envelope carries a
:class:`CollectiveLatencyProvenance` record naming its source, its locator
inside that source, the transfer being made, and an uncertainty band per
participant width. Widths that no band anchors fail closed.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from simllm.core.execution import CollectiveWork

PICOSECONDS_PER_SECOND = 1_000_000_000_000
LEGACY_COLLECTIVE_LATENCY_PROFILE = "legacy"

#: how far a profile's intercept table may be trusted
#:
#: ``calibrated`` is fitted to a capture of the very thing it prices.
#: ``transferred-at-use`` is calibrated at its source but applied outside the
#: operation shape or topology that source captured, so the number is exact and
#: the application of it is not. ``provisional-transferred`` is derived from
#: other evidence and never measured. ``structural-floor`` holds by
#: construction rather than by measurement.
COLLECTIVE_EVIDENCE_CLASSES = (
    "calibrated",
    "transferred-at-use",
    "provisional-transferred",
    "structural-floor",
)

#: the three states a fixed-cost envelope can be selected in
COLLECTIVE_FIXED_COST_ARMS = ("off", "lower", "upper")


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
class CollectiveBandwidthCurve:
    """A payload-indexed serialization bandwidth for one participant width.

    The single-slope serializer charges one ``bytes_per_second`` at every
    payload. Two first-party captures, on an A100 `NV4` mesh and a GH200 `NV6`
    mesh, measured what that costs: anchored at the small-message floor and the
    asymptotic bandwidth, one slope is optimistic at every payload between
    those anchors, by up to 50.8 percent, because bus bandwidth is still
    climbing across that window. A curve replaces the constant with measured
    anchors and a declared interpolation law.

    ``points`` are ``(endpoint_bytes, bytes_per_second)`` pairs with strictly
    increasing endpoint bytes. The bandwidth is a *serialization* bandwidth:
    endpoint bytes over the measured time with the width's base latency already
    removed. Storing total algorithm bandwidth would double-count that floor
    once the caller adds ``base_latency_ps`` back on.

    Between anchors the logarithm of bandwidth is linear in the logarithm of
    endpoint bytes. Below the first anchor the first anchor's bandwidth holds,
    above the last the last one does. The quantity spans more than three
    decades, so a geometric law is the natural one; a linear one would be
    dominated by the largest anchor.

    The law is declared rather than derived. NCCL selects protocols and channel
    counts in discrete steps, so no smooth closed form describes the ramp. What
    this class defends is its held-out interpolation error, recorded in
    examples/collective_regime_curve_v1, not a claim about mechanism.
    """

    curve_id: str
    points: tuple[tuple[int, int], ...]
    provenance: CollectiveLatencyProvenance | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tuple(tuple(point) for point in self.points))
        if not isinstance(self.curve_id, str) or not self.curve_id.strip():
            raise ValueError("curve_id must be a nonblank string")
        if len(self.points) < 2:
            raise ValueError("a bandwidth curve needs at least two anchors")
        previous_bytes: int | None = None
        for index, point in enumerate(self.points):
            if len(point) != 2:
                raise ValueError(f"points[{index}] must be a two-item tuple")
            endpoint_bytes = _require_int(f"points[{index}][0]", point[0], minimum=1)
            _require_int(f"points[{index}][1]", point[1], minimum=1)
            if previous_bytes is not None and endpoint_bytes <= previous_bytes:
                raise ValueError(
                    "points must have strictly increasing endpoint bytes; "
                    f"points[{index}][0] is {endpoint_bytes} after {previous_bytes}"
                )
            previous_bytes = endpoint_bytes

    @property
    def anchored_endpoint_bytes(self) -> tuple[int, ...]:
        """Return the endpoint byte counts this curve was calibrated at."""

        return tuple(endpoint_bytes for endpoint_bytes, _ in self.points)

    def bandwidth_bytes_per_second(self, endpoint_bytes: int) -> Fraction:
        """Return the interpolated serialization bandwidth as an exact ratio.

        A :class:`~fractions.Fraction` is returned rather than a float so the
        caller's picosecond arithmetic stays exact and reproducible.
        """

        endpoint_bytes = _require_int("endpoint_bytes", endpoint_bytes, minimum=0)
        if endpoint_bytes <= self.points[0][0]:
            return Fraction(self.points[0][1])
        if endpoint_bytes >= self.points[-1][0]:
            return Fraction(self.points[-1][1])
        for (low_bytes, low_rate), (high_bytes, high_rate) in zip(
            self.points, self.points[1:]
        ):
            if low_bytes <= endpoint_bytes <= high_bytes:
                # Geometric interpolation: log(rate) is linear in log(bytes).
                # The exponent is computed in floating point and the result is
                # re-rationalized, so the returned bandwidth is an exact ratio
                # and every downstream picosecond stays integral.
                span = math.log(high_bytes / low_bytes)
                position = math.log(endpoint_bytes / low_bytes) / span
                rate = low_rate * (high_rate / low_rate) ** position
                return Fraction(rate).limit_denominator(1_000_000)
        raise AssertionError("unreachable: endpoint bytes fell outside every span")


@dataclass(frozen=True)
class CollectiveLatencyProvenance:
    """Where one profile's intercept table came from and how far it is trusted.

    ``evidence_class`` separates a direct capture from a stated transfer and
    from a bound that holds by construction. ``source`` and ``locator``
    together identify the exact evidence, and ``transfer`` states in one
    sentence what was done to it to reach this profile's table. Every
    participant width the profile supports must appear in
    ``participant_latency_band_ps`` as an inclusive ``(width, low, high)``
    triple bracketing the profile's own point value, so a consumer cannot read
    a transferred constant without its uncertainty attached.
    """

    evidence_class: str
    source: str
    locator: str
    transfer: str
    participant_latency_band_ps: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "participant_latency_band_ps",
            tuple(tuple(entry) for entry in self.participant_latency_band_ps),
        )
        if self.evidence_class not in COLLECTIVE_EVIDENCE_CLASSES:
            raise ValueError(
                "evidence_class must be one of "
                f"{COLLECTIVE_EVIDENCE_CLASSES}, got {self.evidence_class!r}"
            )
        for name in ("source", "locator", "transfer"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonblank string")
        if not self.participant_latency_band_ps:
            raise ValueError("participant_latency_band_ps must not be empty")
        widths: list[int] = []
        for index, entry in enumerate(self.participant_latency_band_ps):
            if len(entry) != 3:
                raise ValueError(
                    f"participant_latency_band_ps[{index}] must be a triple"
                )
            width = _require_int(
                f"participant_latency_band_ps[{index}][0]",
                entry[0],
                minimum=2,
            )
            low = _require_int(
                f"participant_latency_band_ps[{index}][1]",
                entry[1],
                minimum=0,
            )
            high = _require_int(
                f"participant_latency_band_ps[{index}][2]",
                entry[2],
                minimum=0,
            )
            if high < low:
                raise ValueError(
                    f"participant_latency_band_ps[{index}] upper edge is below "
                    "its lower edge"
                )
            widths.append(width)
        if widths != sorted(set(widths)):
            raise ValueError(
                "participant_latency_band_ps widths must be unique and increasing"
            )

    @property
    def banded_participant_counts(self) -> tuple[int, ...]:
        """Return the widths this provenance record anchors."""

        return tuple(width for width, _, _ in self.participant_latency_band_ps)

    def band_ps(self, participant_count: int) -> tuple[int, int]:
        """Return the inclusive uncertainty band for one participant width."""

        _require_int("participant_count", participant_count, minimum=2)
        for width, low, high in self.participant_latency_band_ps:
            if width == participant_count:
                return (low, high)
        anchored = ", ".join(str(width) for width in self.banded_participant_counts)
        raise ValueError(
            f"no uncertainty band anchors participant count {participant_count}; "
            f"anchored counts are {anchored}"
        )


@dataclass(frozen=True)
class CollectiveLatencyProfile:
    """One immutable calibration for semantic collective service.

    ``participant_latency_ps`` is a direct table because the source capture
    identifies participant widths 2, 4 and 8, but does not identify an
    interpolation law. Endpoint-byte validity is bounded by the endpoint
    loads produced by the source all-reduce payload interval.

    The table is a surcharge added to the backend's transport, so the fixed
    cost a run actually realizes is ``realized_fixed_cost_ps``, which adds the
    named propagation reference the backend already charges. ``provenance`` is
    optional so ad-hoc profiles stay constructible, but every profile that
    joins a :class:`CollectiveFixedCostEnvelope` must carry one.
    """

    profile_id: str
    bandwidth_bytes_per_second: int
    participant_latency_ps: tuple[tuple[int, int], ...]
    source_payload_bytes_min: int
    source_payload_bytes_max: int
    propagation_reference_ps: int
    provenance: CollectiveLatencyProvenance | None = None
    bandwidth_curves: tuple[tuple[int, CollectiveBandwidthCurve], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "participant_latency_ps",
            tuple(tuple(entry) for entry in self.participant_latency_ps),
        )
        object.__setattr__(
            self,
            "bandwidth_curves",
            tuple(tuple(entry) for entry in self.bandwidth_curves),
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
        curve_widths: list[int] = []
        for index, entry in enumerate(self.bandwidth_curves):
            if len(entry) != 2:
                raise ValueError(f"bandwidth_curves[{index}] must be a two-item tuple")
            curve_width = _require_int(
                f"bandwidth_curves[{index}][0]", entry[0], minimum=2
            )
            if not isinstance(entry[1], CollectiveBandwidthCurve):
                raise TypeError(
                    f"bandwidth_curves[{index}][1] must be a CollectiveBandwidthCurve"
                )
            if curve_width not in widths:
                raise ValueError(
                    f"profile {self.profile_id!r} carries a bandwidth curve for "
                    f"participant count {curve_width}, which its latency table "
                    f"does not support"
                )
            curve_widths.append(curve_width)
        if curve_widths != sorted(set(curve_widths)):
            raise ValueError(
                "bandwidth_curves widths must be unique and increasing"
            )
        if self.provenance is not None:
            if not isinstance(self.provenance, CollectiveLatencyProvenance):
                raise TypeError(
                    "provenance must be a CollectiveLatencyProvenance or None"
                )
            if self.provenance.banded_participant_counts != tuple(widths):
                raise ValueError(
                    f"profile {self.profile_id!r} provenance anchors widths "
                    f"{self.provenance.banded_participant_counts} but the table "
                    f"supports {tuple(widths)}"
                )
            for width, latency_ps in self.participant_latency_ps:
                low, high = self.provenance.band_ps(width)
                if not low <= latency_ps <= high:
                    raise ValueError(
                        f"profile {self.profile_id!r} charges {latency_ps} ps at "
                        f"participant count {width}, outside its own declared "
                        f"band [{low}, {high}]"
                    )

    @property
    def supported_participant_counts(self) -> tuple[int, ...]:
        """Return the directly calibrated participant widths."""

        return tuple(width for width, _ in self.participant_latency_ps)

    @property
    def evidence_class(self) -> str:
        """Return the declared evidence class, or ``'unattributed'``."""

        return "unattributed" if self.provenance is None else self.provenance.evidence_class

    def require_provenance(self) -> CollectiveLatencyProvenance:
        """Return the provenance record, refusing an unattributed profile."""

        if self.provenance is None:
            raise ValueError(
                f"profile {self.profile_id!r} carries no provenance record"
            )
        return self.provenance

    def base_latency_band_ps(self, participant_count: int) -> tuple[int, int]:
        """Return the declared uncertainty band around one surcharge."""

        self.base_latency_ps(participant_count)
        return self.require_provenance().band_ps(participant_count)

    def realized_fixed_cost_ps(self, participant_count: int) -> int:
        """Return the per-collective fixed cost a run actually charges.

        The surcharge sits on top of a transport that already contains one
        propagation delay, so the realized fixed cost is the sum of the two.
        For a profile whose source capture was itself a complete fixed cost,
        that sum over-counts by at most ``propagation_reference_ps``.
        """

        return self.base_latency_ps(participant_count) + self.propagation_reference_ps

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

    def bandwidth_curve(self, participant_count: int) -> CollectiveBandwidthCurve | None:
        """Return this width's regime curve, or ``None`` for the flat slope."""

        for width, curve in self.bandwidth_curves:
            if width == participant_count:
                return curve
        return None

    def endpoint_serialization_ps(
        self,
        participant_count: int,
        endpoint_bytes: int,
    ) -> int:
        """Return exact upward-rounded endpoint serialization picoseconds.

        A width with no calibrated curve keeps the flat slope this profile has
        always charged, so a profile carrying no curve is bit-for-bit what it
        was. A width that has one charges its interpolated serialization
        bandwidth instead, which is the whole point of TRAF-43: one slope is
        optimistic by up to 50.8 percent across the payload decade where bus
        bandwidth is still climbing.
        """

        endpoint_bytes = self.validate_endpoint_bytes(
            participant_count,
            endpoint_bytes,
        )
        curve = self.bandwidth_curve(participant_count)
        if curve is None:
            return _ceil_div(
                endpoint_bytes * PICOSECONDS_PER_SECOND,
                self.bandwidth_bytes_per_second,
            )
        rate = curve.bandwidth_bytes_per_second(endpoint_bytes)
        return _ceil_div(
            endpoint_bytes * PICOSECONDS_PER_SECOND * rate.denominator,
            rate.numerator,
        )

    def total_service_ps(self, participant_count: int, endpoint_bytes: int) -> int:
        """Return one base latency plus exact endpoint serialization service."""

        return self.base_latency_ps(
            participant_count
        ) + self.endpoint_serialization_ps(participant_count, endpoint_bytes)


#: fluid one-way propagation the backend already charges per collective, ps
COLLECTIVE_PROPAGATION_REFERENCE_PS = 2_000_000

#: NVLink ring-step cost implied by the source table's two extreme widths, ps
NVLINK_RING_STEP_PS = 1_617_160

#: fabric ring-step anchors used by the cross-node transfer, ps
FABRIC_RING_STEP_LOW_PS = 2_000_000
FABRIC_RING_STEP_POINT_PS = 3_000_000
FABRIC_RING_STEP_HIGH_PS = 5_000_000


def _ring_steps(participant_count: int) -> int:
    """Return the ring-step count of a ``participant_count``-wide collective."""

    return 2 * (participant_count - 1)


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
    propagation_reference_ps=COLLECTIVE_PROPAGATION_REFERENCE_PS,
    provenance=CollectiveLatencyProvenance(
        evidence_class="calibrated",
        source=(
            "nccl-tests issue 333 capture of a DGX B200 intra-node NVLink "
            "ALL-REDUCE under NCCL 2.27; the capture records completion times "
            "and does not name the algorithm NCCL selected"
        ),
        locator=(
            "the fitted intercepts of examples/collective_latency_floor_v1, "
            "held out at 4 KiB with errors of 0.261, 0.347 and 0.080 us at "
            "widths 2, 4 and 8"
        ),
        transfer=(
            "the intra-node ALL-REDUCE intercept is charged unchanged as the "
            "per-collective surcharge of any supported collective, including "
            "pairwise ALL-TO-ALLV"
        ),
        participant_latency_band_ps=(
            (2, 10_461_112, 10_983_112),
            (4, 15_398_167, 16_092_167),
            (8, 30_048_029, 30_208_029),
        ),
    ),
)

COLLECTIVE_FIXED_COST_FLOOR_PROFILE = CollectiveLatencyProfile(
    profile_id="collective-fixed-cost-floor-v1",
    bandwidth_bytes_per_second=B200_NCCL_2_27_LOCAL_PROFILE.bandwidth_bytes_per_second,
    participant_latency_ps=((2, 0), (4, 0), (8, 0)),
    source_payload_bytes_min=B200_NCCL_2_27_LOCAL_PROFILE.source_payload_bytes_min,
    source_payload_bytes_max=B200_NCCL_2_27_LOCAL_PROFILE.source_payload_bytes_max,
    propagation_reference_ps=COLLECTIVE_PROPAGATION_REFERENCE_PS,
    provenance=CollectiveLatencyProvenance(
        evidence_class="structural-floor",
        source="the rnic-nn-fluid backend's own measured per-collective propagation",
        locator=(
            "31 matched decode compositions where 2 * fabric(400G) - fabric(200G) "
            "returned 96,000,006 to 96,000,048 ps across 48 collectives, recorded "
            "in the traffic module status section"
        ),
        transfer=(
            "no surcharge is added, so the claimed per-collective fixed cost is "
            "exactly the 2.000 us propagation the backend already charges; this "
            "is a lower bound because no collective can complete faster than one "
            "propagation delay"
        ),
        participant_latency_band_ps=((2, 0, 0), (4, 0, 0), (8, 0, 0)),
    ),
)

B200_NCCL_2_27_CROSS_NODE_PROVISIONAL_PROFILE = CollectiveLatencyProfile(
    profile_id="b200-nccl-2.27-cross-node-provisional-v1",
    bandwidth_bytes_per_second=B200_NCCL_2_27_LOCAL_PROFILE.bandwidth_bytes_per_second,
    participant_latency_ps=tuple(
        (
            width,
            latency_ps
            + _ring_steps(width) * (FABRIC_RING_STEP_POINT_PS - NVLINK_RING_STEP_PS),
        )
        for width, latency_ps in B200_NCCL_2_27_LOCAL_PROFILE.participant_latency_ps
    ),
    source_payload_bytes_min=B200_NCCL_2_27_LOCAL_PROFILE.source_payload_bytes_min,
    source_payload_bytes_max=B200_NCCL_2_27_LOCAL_PROFILE.source_payload_bytes_max,
    propagation_reference_ps=COLLECTIVE_PROPAGATION_REFERENCE_PS,
    provenance=CollectiveLatencyProvenance(
        evidence_class="provisional-transferred",
        source=(
            "b200-nccl-2.27-local-v1 combined with the RDMA fixed-cost anchors "
            "recorded in docs/papers/msg-size-vs-bandwidth.md"
        ),
        locator=(
            "every fabric step is this repository's measured 2,000,000 ps fluid "
            "propagation reference plus a per-step initiation term. The lower "
            "edge adds nothing, so it is that 2,000,000 ps alone. The point "
            "estimate adds 1,000,000 ps, one half of the about 2 us commodity "
            "RDMA round-trip anchor of Kalia et al. ATC'16, giving 3,000,000 ps. "
            "The upper edge adds 3,000,000 ps, the top of the p50 ACK turnaround "
            "in UCCL Table 2, giving 5,000,000 ps. The UCCL figures are "
            "restricted to that table's Light columns, whose message sizes match "
            "this workload's 12 to 114 KiB collectives; the Heavy columns reach "
            "7.0 us of p50 ACK turnaround and would put the upper edge higher"
        ),
        transfer=(
            "the 2(W-1) ring-step decomposition is this repository's own "
            "collective expansion model rather than an attribute of the capture, "
            "which names no algorithm. Each such step, worth 1,617,160 ps by the "
            "two-point slope of the source table, is replaced by one fabric step "
            "worth 3,000,000 ps at the point estimate, 2,000,000 ps at the lower "
            "edge and 5,000,000 ps at the upper edge. NCCL 2.27 on an eight-GPU "
            "NVSwitch node ordinarily selects NVLS or a tree for a small "
            "ALL-REDUCE, and a 2 log2(W) tree decomposition at the same per-step "
            "delta would move the width-8 point estimate from 49.49 to 38.43 us, "
            "so the algorithm assumption is a first-order term rather than a "
            "detail. No cross-node measurement was taken, so this profile is "
            "provisional-transferred and never calibrated"
        ),
        participant_latency_band_ps=tuple(
            (
                width,
                latency_ps
                + _ring_steps(width) * (FABRIC_RING_STEP_LOW_PS - NVLINK_RING_STEP_PS),
                latency_ps
                + _ring_steps(width) * (FABRIC_RING_STEP_HIGH_PS - NVLINK_RING_STEP_PS),
            )
            for width, latency_ps in B200_NCCL_2_27_LOCAL_PROFILE.participant_latency_ps
        ),
    ),
)


#: one Cray Cassini 1 Slingshot 200Gb port, one direction, decimal bytes/s
CASSINI_SLINGSHOT_PORT_BYTES_PER_SECOND = 25_000_000_000

#: payload at which NCCL 2.31.2 switched this collective from LL to SIMPLE
#:
#: Measured, not assumed: the debug log names the protocol per call, and the
#: switch lands between 786,432 and 1,048,576 bytes. LL carries a 4-byte flag
#: for every 4 bytes of payload, so it puts twice the bytes on the wire, which
#: is nearly free on NVLink and is the dominant cost on a bandwidth-starved
#: kernel socket transport.
CROSS_NODE_SOCKET_LL_TO_SIMPLE_BYTES = 1_048_576

A100_NCCL_2_31_CROSS_NODE_SOCKET_PROFILE = CollectiveLatencyProfile(
    profile_id="a100-nccl-2.31-cross-node-socket-v1",
    # Never consulted: the one supported width carries a curve, so
    # endpoint_serialization_ps always takes the curve branch. It records the
    # asymptotic serialization bandwidth at the largest measured payload, which
    # is what a single-slope form would have charged everywhere.
    bandwidth_bytes_per_second=1_610_316_109,
    participant_latency_ps=((2, 40_140_799),),
    source_payload_bytes_min=8,
    source_payload_bytes_max=134_217_728,
    propagation_reference_ps=COLLECTIVE_PROPAGATION_REFERENCE_PS,
    provenance=CollectiveLatencyProvenance(
        evidence_class="calibrated",
        source=(
            "first-party measurement of two Merlin A100-SXM4-80GB nodes under "
            "NCCL 2.31.2, Slurm job 195648, where the inter-node path is four "
            "Cray Cassini 1 Slingshot 200Gb ports per node carrying NCCL's "
            "kernel socket transport with GPUDirect RDMA disabled, because no "
            "NCCL network plugin is installed and the nodes expose no "
            "InfiniBand device. This is not the 400 Gbit/s RDMA fabric the "
            "shipped cross-node envelope targets, and the transport is a "
            "first-order term rather than a caveat"
        ),
        locator=(
            "examples/crossnode_collective_envelope_v1, the 8-byte ring "
            "ALL-REDUCE back-to-back floor of the default interface arm, and "
            "that arm's measured serialization bandwidth at the four anchor "
            "payloads its freeze named"
        ),
        transfer=(
            "none for the intercept, which is measured at the width, on the "
            "interconnect, for the operation and inside the payload envelope it "
            "prices. The band is the back-to-back to isolated bracket: the "
            "charged value is the back-to-back reading, comparable to the "
            "intra-node lanes and to nccl-tests, and the upper edge is the "
            "isolated reading, which additionally pays the 6.07 us launch and "
            "synchronize roundtrip this node type measures"
        ),
        participant_latency_band_ps=((2, 40_140_799, 55_808_000),),
    ),
    bandwidth_curves=(
        (
            2,
            CollectiveBandwidthCurve(
                curve_id="a100-nccl-2.31-cross-node-socket-w2-v1",
                points=(
                    (64, 138_888_729),
                    (786_432, 555_254_304),
                    (1_048_576, 1_339_700_466),
                    (134_217_728, 1_610_316_109),
                ),
            ),
        ),
    ),
)


@dataclass(frozen=True)
class CollectiveFixedCostEnvelope:
    """A named bracket on the per-collective fixed cost, with three arms.

    ``off`` charges no surcharge and is the repository default. ``lower`` and
    ``upper`` charge the two named profiles. The two arms must share the
    endpoint bandwidth and the source payload interval so that switching arms
    changes the fixed cost and nothing else, and the lower arm must be strictly
    cheaper at every supported width so the bracket is a bracket.

    The bracket is over the arms a study can select, not over the physical
    value. An arm's profile may declare an uncertainty band that reaches past
    the arm above it, and the ``claim`` string is required to say what the
    bracket does and does not assert.

    ``applied_evidence_class`` downgrades an arm's reported evidence class at
    the point of use. A profile calibrated on one operation shape or topology
    is not calibrated when an envelope charges it for a different one, and the
    class the run record publishes has to say so. Only a ``calibrated`` profile
    can be downgraded, and every downgrade carries its reason.
    """

    envelope_id: str
    claim: str
    lower_profile: CollectiveLatencyProfile
    upper_profile: CollectiveLatencyProfile
    #: per-arm point-of-use downgrades, as ``(arm, evidence_class, reason)``
    applied_evidence_class: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "applied_evidence_class",
            tuple(tuple(entry) for entry in self.applied_evidence_class),
        )
        if not isinstance(self.envelope_id, str) or not self.envelope_id.strip():
            raise ValueError("envelope_id must be a nonblank string")
        if not isinstance(self.claim, str) or not self.claim.strip():
            raise ValueError("claim must be a nonblank string")
        for name in ("lower_profile", "upper_profile"):
            if not isinstance(getattr(self, name), CollectiveLatencyProfile):
                raise TypeError(f"{name} must be a CollectiveLatencyProfile")
        lower = self.lower_profile
        upper = self.upper_profile
        lower.require_provenance()
        upper.require_provenance()
        if lower.profile_id == upper.profile_id:
            raise ValueError("envelope arms must name two different profiles")
        if lower.bandwidth_bytes_per_second != upper.bandwidth_bytes_per_second:
            raise ValueError(
                "envelope arms must share bandwidth_bytes_per_second so the "
                "bracket isolates the fixed cost"
            )
        if lower.bandwidth_curves != upper.bandwidth_curves:
            raise ValueError(
                "envelope arms must share their bandwidth curves so the bracket "
                "isolates the fixed cost"
            )
        if (
            lower.source_payload_bytes_min != upper.source_payload_bytes_min
            or lower.source_payload_bytes_max != upper.source_payload_bytes_max
        ):
            raise ValueError(
                "envelope arms must share the source payload interval so both "
                "arms accept and reject exactly the same workloads"
            )
        if lower.propagation_reference_ps != upper.propagation_reference_ps:
            raise ValueError("envelope arms must share propagation_reference_ps")
        if lower.supported_participant_counts != upper.supported_participant_counts:
            raise ValueError("envelope arms must support the same participant counts")
        for width in lower.supported_participant_counts:
            if lower.base_latency_ps(width) >= upper.base_latency_ps(width):
                raise ValueError(
                    f"envelope {self.envelope_id!r} does not bracket participant "
                    f"count {width}: the lower arm is not strictly cheaper"
                )
        seen: list[str] = []
        for index, entry in enumerate(self.applied_evidence_class):
            if len(entry) != 3:
                raise ValueError(
                    f"applied_evidence_class[{index}] must be an "
                    "(arm, evidence_class, reason) triple"
                )
            arm, evidence_class, reason = entry
            if arm not in ("lower", "upper"):
                raise ValueError(
                    f"applied_evidence_class[{index}] arm must be 'lower' or 'upper'"
                )
            if arm in seen:
                raise ValueError("applied_evidence_class arms must be unique")
            seen.append(arm)
            if evidence_class not in COLLECTIVE_EVIDENCE_CLASSES:
                raise ValueError(
                    f"applied_evidence_class[{index}] class must be one of "
                    f"{COLLECTIVE_EVIDENCE_CLASSES}"
                )
            if evidence_class == "calibrated":
                raise ValueError(
                    "a point-of-use declaration may only downgrade, never "
                    "restore, the 'calibrated' class"
                )
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    f"applied_evidence_class[{index}] reason must be a nonblank string"
                )
            declared = (lower if arm == "lower" else upper).evidence_class
            if declared != "calibrated":
                raise ValueError(
                    f"envelope {self.envelope_id!r} cannot downgrade the {arm} arm: "
                    f"its profile already declares {declared!r}, not 'calibrated'"
                )

    def arm_evidence_class(self, arm: str) -> str | None:
        """Return the evidence class this envelope publishes for one arm."""

        profile = self.arm_profile(arm)
        if profile is None:
            return None
        for declared_arm, evidence_class, _ in self.applied_evidence_class:
            if declared_arm == arm:
                return evidence_class
        return profile.evidence_class

    def arm_evidence_note(self, arm: str) -> str:
        """Return why one arm's evidence class was downgraded, or an empty string."""

        self.arm_profile(arm)
        for declared_arm, _, reason in self.applied_evidence_class:
            if declared_arm == arm:
                return reason
        return ""

    @property
    def supported_participant_counts(self) -> tuple[int, ...]:
        """Return the widths both arms support."""

        return self.lower_profile.supported_participant_counts

    @property
    def arm_names(self) -> tuple[str, ...]:
        """Return the three selectable arm names."""

        return COLLECTIVE_FIXED_COST_ARMS

    def arm_profile(self, arm: str) -> CollectiveLatencyProfile | None:
        """Return the profile one arm selects; ``off`` selects none."""

        if arm == "off":
            return None
        if arm == "lower":
            return self.lower_profile
        if arm == "upper":
            return self.upper_profile
        raise ValueError(
            f"arm must be one of {COLLECTIVE_FIXED_COST_ARMS}, got {arm!r}"
        )

    def bracket_ps(self, participant_count: int) -> tuple[int, int]:
        """Return the surcharge bracket at one participant width."""

        return (
            self.lower_profile.base_latency_ps(participant_count),
            self.upper_profile.base_latency_ps(participant_count),
        )

    def realized_bracket_ps(self, participant_count: int) -> tuple[int, int]:
        """Return the realized fixed-cost bracket, propagation included."""

        return (
            self.lower_profile.realized_fixed_cost_ps(participant_count),
            self.upper_profile.realized_fixed_cost_ps(participant_count),
        )


_OPERATION_SHAPE_DOWNGRADE = (
    "the source capture is an ALL-REDUCE and this envelope charges its "
    "intercept for whatever collective the workload emits, including pairwise "
    "ALL-TO-ALLV, so the number is calibrated and its application here is not"
)

INTRA_NODE_COLLECTIVE_FIXED_COST_ENVELOPE = CollectiveFixedCostEnvelope(
    envelope_id="intra-node-fixed-cost-v1",
    claim=(
        "for a collective whose steps stay on NVLink, the selectable arms run "
        "from the modeled propagation delay, which is a floor no collective can "
        "beat, to the captured DGX B200 intra-node NCCL intercept transferred "
        "to whichever collective is being priced. The upper arm is the "
        "pessimistic selectable edge, not a ceiling on the physical value"
    ),
    lower_profile=COLLECTIVE_FIXED_COST_FLOOR_PROFILE,
    upper_profile=B200_NCCL_2_27_LOCAL_PROFILE,
    applied_evidence_class=(("upper", "transferred-at-use", _OPERATION_SHAPE_DOWNGRADE),),
)

CROSS_NODE_COLLECTIVE_FIXED_COST_ENVELOPE = CollectiveFixedCostEnvelope(
    envelope_id="cross-node-fixed-cost-provisional-v1",
    claim=(
        "for a collective whose steps cross the fabric, the selectable arms run "
        "from the captured intra-node intercept, which is a floor because a "
        "fabric hop cannot be cheaper than the NVLink hop it replaces, to the "
        "provisional-transferred cross-node estimate. The upper arm is the "
        "pessimistic selectable edge and not a ceiling on the physical value: "
        "its own declared band reaches 77,487,789 ps at width 8, 57 percent "
        "above the 49,487,789 ps it charges, and no evidence here establishes "
        "any ceiling at all"
    ),
    lower_profile=B200_NCCL_2_27_LOCAL_PROFILE,
    upper_profile=B200_NCCL_2_27_CROSS_NODE_PROVISIONAL_PROFILE,
    applied_evidence_class=(
        (
            "lower",
            "transferred-at-use",
            _OPERATION_SHAPE_DOWNGRADE
            + ", and this envelope additionally charges an intra-node capture "
            "for a collective whose steps cross the fabric",
        ),
    ),
)

_NAMED_COLLECTIVE_LATENCY_PROFILES = (
    B200_NCCL_2_27_LOCAL_PROFILE,
    COLLECTIVE_FIXED_COST_FLOOR_PROFILE,
    B200_NCCL_2_27_CROSS_NODE_PROVISIONAL_PROFILE,
    A100_NCCL_2_31_CROSS_NODE_SOCKET_PROFILE,
)

_NAMED_COLLECTIVE_FIXED_COST_ENVELOPES = (
    INTRA_NODE_COLLECTIVE_FIXED_COST_ENVELOPE,
    CROSS_NODE_COLLECTIVE_FIXED_COST_ENVELOPE,
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
        for profile in _NAMED_COLLECTIVE_LATENCY_PROFILES:
            if selector == profile.profile_id:
                return profile
        raise ValueError(f"unknown collective latency profile {selector!r}")
    raise TypeError(
        "collective latency profile must be None, a selector string, or a "
        "CollectiveLatencyProfile"
    )


def resolve_collective_fixed_cost_envelope(
    selector: str | CollectiveFixedCostEnvelope | None,
) -> CollectiveFixedCostEnvelope | None:
    """Resolve a named fixed-cost envelope, or ``None`` for no envelope."""

    if selector is None:
        return None
    if isinstance(selector, CollectiveFixedCostEnvelope):
        return selector
    if isinstance(selector, str):
        for envelope in _NAMED_COLLECTIVE_FIXED_COST_ENVELOPES:
            if selector == envelope.envelope_id:
                return envelope
        raise ValueError(f"unknown collective fixed cost envelope {selector!r}")
    raise TypeError(
        "collective fixed cost envelope must be None, a selector string, or a "
        "CollectiveFixedCostEnvelope"
    )


@dataclass(frozen=True)
class ArmRatioEnvelope:
    """One reported ratio bracketed across the named fixed-cost arms.

    This is the publishable form of an envelope study: instead of one ratio
    computed under one silently chosen fixed cost, the same ratio is reported
    once per arm together with the interval they span. ``brackets_unity`` says
    whether the evidence determines the sign of the comparison at all.
    """

    label: str
    numerator: str
    denominator: str
    arm_ratios: tuple[tuple[str, float], ...]
    minimum: float
    maximum: float
    brackets_unity: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "arm_ratios",
            tuple((arm, float(ratio)) for arm, ratio in self.arm_ratios),
        )
        for name in ("label", "numerator", "denominator"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonblank string")
        if len(self.arm_ratios) < 2:
            raise ValueError("an arm ratio envelope needs at least two arms")
        arms = [arm for arm, _ in self.arm_ratios]
        if len(set(arms)) != len(arms):
            raise ValueError("arm names must be unique")
        ratios = [ratio for _, ratio in self.arm_ratios]
        if self.minimum != min(ratios) or self.maximum != max(ratios):
            raise ValueError("envelope edges disagree with the reported arm ratios")
        if self.brackets_unity != (self.minimum <= 1.0 <= self.maximum):
            raise ValueError("brackets_unity disagrees with the reported edges")

    @property
    def width(self) -> float:
        """Return the multiplicative width of the bracket."""

        return self.maximum / self.minimum


def arm_ratio_envelope(
    label: str,
    numerator: str,
    denominator: str,
    arm_values: Sequence[tuple[str, int, int]],
) -> ArmRatioEnvelope:
    """Bracket one ratio across arms given ``(arm, numerator, denominator)`` rows.

    Each row carries the two picosecond values measured under that arm. The
    quotient is taken exactly and only then converted to a float, so a large
    fixed cost cannot lose the low-order digits of the ratio.
    """

    rows: list[tuple[str, float]] = []
    for index, entry in enumerate(arm_values):
        if not isinstance(entry, tuple) or len(entry) != 3:
            raise TypeError(f"arm_values[{index}] must be a triple")
        arm, numerator_ps, denominator_ps = entry
        if not isinstance(arm, str) or not arm.strip():
            raise ValueError(f"arm_values[{index}][0] must be a nonblank string")
        _require_int(f"arm_values[{index}][1]", numerator_ps, minimum=1)
        _require_int(f"arm_values[{index}][2]", denominator_ps, minimum=1)
        rows.append((arm, float(Fraction(numerator_ps, denominator_ps))))
    if len(rows) < 2:
        raise ValueError("an arm ratio envelope needs at least two arms")
    ratios = [ratio for _, ratio in rows]
    return ArmRatioEnvelope(
        label=label,
        numerator=numerator,
        denominator=denominator,
        arm_ratios=tuple(rows),
        minimum=min(ratios),
        maximum=max(ratios),
        brackets_unity=min(ratios) <= 1.0 <= max(ratios),
    )


__all__ = [
    "A100_NCCL_2_31_CROSS_NODE_SOCKET_PROFILE",
    "B200_NCCL_2_27_CROSS_NODE_PROVISIONAL_PROFILE",
    "B200_NCCL_2_27_LOCAL_PROFILE",
    "CASSINI_SLINGSHOT_PORT_BYTES_PER_SECOND",
    "COLLECTIVE_EVIDENCE_CLASSES",
    "COLLECTIVE_FIXED_COST_ARMS",
    "COLLECTIVE_FIXED_COST_FLOOR_PROFILE",
    "COLLECTIVE_PROPAGATION_REFERENCE_PS",
    "CROSS_NODE_COLLECTIVE_FIXED_COST_ENVELOPE",
    "FABRIC_RING_STEP_HIGH_PS",
    "FABRIC_RING_STEP_LOW_PS",
    "FABRIC_RING_STEP_POINT_PS",
    "INTRA_NODE_COLLECTIVE_FIXED_COST_ENVELOPE",
    "LEGACY_COLLECTIVE_LATENCY_PROFILE",
    "NVLINK_RING_STEP_PS",
    "ArmRatioEnvelope",
    "CollectiveBandwidthCurve",
    "CollectiveFixedCostEnvelope",
    "CollectiveLatencyProfile",
    "CollectiveLatencyProvenance",
    "arm_ratio_envelope",
    "critical_collective_endpoint_bytes",
    "resolve_collective_fixed_cost_envelope",
    "resolve_collective_latency_profile",
]
