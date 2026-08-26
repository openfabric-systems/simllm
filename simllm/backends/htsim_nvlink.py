"""Candidate packet-level NVLink service for htsim-style compositions.

The active SimLLM intra-node path remains analytical.  This module is an
additive candidate-profile handoff for TRAF-65: callers that provide no
profile receive their analytical result back by object identity.  Selecting a
profile composes three independently parameterized services in this order:

``TX -> switch -> RX``

The four-A100 NV4 profile uses an explicit pass-through switch.  Queues,
arbitration, FIFO placement, and head-of-line blocking belong to the switch
module for future NVSwitch profiles, but none is inferred from a direct mesh.
All shipped numbers in this module are declared candidates.  The only measured
values accepted by the validator are the already published A100 envelope rows.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Generic, TypeVar

NVLINK_CANDIDATE_PROFILE_SCHEMA = "simllm-htsim-nvlink-candidate-profile-v1"
NVLINK_CANDIDATE_PROFILE_IMPLEMENTATION = "simllm-htsim-nvlink-domain-v1"
NVLINK_CANDIDATE_EVIDENCE_CLASS = "declared_candidate_not_hardware_measurement"

_PS_PER_SECOND = 1_000_000_000_000
_AnalyticResult = TypeVar("_AnalyticResult")


class NvlinkOperation(str, Enum):
    """Logical peer operation whose payload enters the NVLink domain."""

    PEER_WRITE = "peer_write"
    PEER_READ = "peer_read"


class NvlinkPacketDirection(str, Enum):
    """Transaction direction, independent of endpoint orientation."""

    REQUEST = "request"
    RESPONSE = "response"


class NvlinkSwitchMode(str, Enum):
    """Whether the stage is inert or owns an explicit contention service."""

    PASS_THROUGH = "pass_through"
    QUEUED = "queued"


class NvlinkFifoPlacement(str, Enum):
    """Queue ownership for a non-pass-through switch profile."""

    INPUT = "input"
    OUTPUT = "output"
    SHARED = "shared"


@dataclass(frozen=True, kw_only=True)
class NvlinkTransfer:
    """One logical peer extent presented to a TX endpoint."""

    extent_id: str
    source: int
    destination: int
    payload_bytes: int
    operation: NvlinkOperation = NvlinkOperation.PEER_WRITE
    released_at_ps: int = 0

    def __post_init__(self) -> None:
        _require_text("extent_id", self.extent_id)
        _require_endpoint("source", self.source)
        _require_endpoint("destination", self.destination)
        if self.source == self.destination:
            raise ValueError("NVLink source and destination must differ")
        _require_positive_int("payload_bytes", self.payload_bytes)
        _require_enum("operation", self.operation, NvlinkOperation)
        _require_nonnegative_int("released_at_ps", self.released_at_ps)


@dataclass(frozen=True, kw_only=True)
class NvlinkPacket:
    """One candidate NVLink transaction packet and its module timestamps."""

    extent_id: str
    attempt_id: str
    sequence: int
    source: int
    destination: int
    direction: NvlinkPacketDirection
    payload_bytes: int
    header_bytes: int
    wire_bytes: int
    released_at_ps: int
    link_index: int | None = None
    tx_started_at_ps: int | None = None
    tx_finished_at_ps: int | None = None
    switch_started_at_ps: int | None = None
    switch_finished_at_ps: int | None = None
    rx_started_at_ps: int | None = None
    rx_finished_at_ps: int | None = None
    delivered_at_ps: int | None = None

    def __post_init__(self) -> None:
        for name in ("extent_id", "attempt_id"):
            _require_text(name, getattr(self, name))
        _require_nonnegative_int("sequence", self.sequence)
        _require_endpoint("source", self.source)
        _require_endpoint("destination", self.destination)
        if self.source == self.destination:
            raise ValueError("NVLink packet source and destination must differ")
        _require_enum("direction", self.direction, NvlinkPacketDirection)
        _require_nonnegative_int("payload_bytes", self.payload_bytes)
        _require_positive_int("header_bytes", self.header_bytes)
        _require_positive_int("wire_bytes", self.wire_bytes)
        if self.wire_bytes != self.payload_bytes + self.header_bytes:
            raise ValueError("wire_bytes must equal payload_bytes plus header_bytes")
        _require_nonnegative_int("released_at_ps", self.released_at_ps)
        if self.link_index is not None:
            _require_nonnegative_int("link_index", self.link_index)
        for name in (
            "tx_started_at_ps",
            "tx_finished_at_ps",
            "switch_started_at_ps",
            "switch_finished_at_ps",
            "rx_started_at_ps",
            "rx_finished_at_ps",
            "delivered_at_ps",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_nonnegative_int(name, value)


@dataclass(frozen=True, kw_only=True)
class NvlinkTxConfig:
    """Per-endpoint egress packetization and directional serializers."""

    max_payload_bytes: int
    header_bytes: int
    links_per_peer: int
    per_link_rate_bytes_per_second: int
    endpoint_egress_rate_bytes_per_second: int
    bond_policy: str
    credits_per_destination: int
    credit_unit_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "max_payload_bytes",
            "header_bytes",
            "links_per_peer",
            "per_link_rate_bytes_per_second",
            "endpoint_egress_rate_bytes_per_second",
            "credits_per_destination",
            "credit_unit_bytes",
        ):
            _require_positive_int(name, getattr(self, name))
        _require_text("bond_policy", self.bond_policy)
        if self.bond_policy != "earliest_available_packet_striping":
            raise ValueError("unsupported NVLink TX bond policy")
        if self.credit_unit_bytes < self.max_payload_bytes + self.header_bytes:
            raise ValueError("credit unit cannot be smaller than one maximum packet")


@dataclass(frozen=True, kw_only=True)
class NvlinkRxConfig:
    """Per-endpoint ingress, buffering, credit return, and delivery."""

    ingress_rate_bytes_per_second: int
    buffer_capacity_bytes: int
    credit_return_latency_ps: int
    reassembly_policy: str
    delivery_order: str

    def __post_init__(self) -> None:
        _require_positive_int("ingress_rate_bytes_per_second", self.ingress_rate_bytes_per_second)
        _require_positive_int("buffer_capacity_bytes", self.buffer_capacity_bytes)
        _require_nonnegative_int("credit_return_latency_ps", self.credit_return_latency_ps)
        _require_text("reassembly_policy", self.reassembly_policy)
        _require_text("delivery_order", self.delivery_order)
        if self.reassembly_policy != "extent_sequence":
            raise ValueError("unsupported NVLink RX reassembly policy")
        if self.delivery_order != "per_extent":
            raise ValueError("unsupported NVLink RX delivery order")


@dataclass(frozen=True, kw_only=True)
class NvlinkSwitchConfig:
    """Contention stage between TX and RX.

    Pass-through is an explicit configuration, not an omitted module.  Queue
    parameters are absent in that mode so a direct mesh cannot accidentally
    acquire switch delay.  A queued profile must declare every service term.
    """

    mode: NvlinkSwitchMode
    fifo_placement: NvlinkFifoPlacement | None = None
    service_rate_bytes_per_second: int | None = None
    buffer_capacity_bytes: int | None = None
    arbitration: str | None = None
    head_of_line_blocking: bool | None = None

    def __post_init__(self) -> None:
        _require_enum("mode", self.mode, NvlinkSwitchMode)
        queue_values = (
            self.fifo_placement,
            self.service_rate_bytes_per_second,
            self.buffer_capacity_bytes,
            self.arbitration,
            self.head_of_line_blocking,
        )
        if self.mode is NvlinkSwitchMode.PASS_THROUGH:
            if any(value is not None for value in queue_values):
                raise ValueError(
                    "pass-through NVLink switch must not declare FIFO, rate, buffer, "
                    "arbitration, or head-of-line parameters"
                )
            return
        if any(value is None for value in queue_values):
            raise ValueError("queued NVLink switch requires every queue parameter")
        _require_enum("fifo_placement", self.fifo_placement, NvlinkFifoPlacement)
        _require_positive_int("service_rate_bytes_per_second", self.service_rate_bytes_per_second)
        _require_positive_int("buffer_capacity_bytes", self.buffer_capacity_bytes)
        _require_text("arbitration", self.arbitration)
        if self.arbitration != "fifo":
            raise ValueError("the v1 NVLink switch supports FIFO arbitration only")
        if type(self.head_of_line_blocking) is not bool:
            raise TypeError("head_of_line_blocking must be a boolean")


@dataclass(frozen=True, kw_only=True)
class NvlinkCandidateProfile:
    """Versioned, unscored candidate profile handed to downstream tasks."""

    profile_id: str
    status: str
    evidence_class: str
    freeze_sha256: str
    tx: NvlinkTxConfig
    switch: NvlinkSwitchConfig
    rx: NvlinkRxConfig
    schema: str = NVLINK_CANDIDATE_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        for name in ("profile_id", "status", "evidence_class", "freeze_sha256"):
            _require_text(name, getattr(self, name))
        if self.schema != NVLINK_CANDIDATE_PROFILE_SCHEMA:
            raise ValueError(f"unsupported NVLink candidate schema {self.schema!r}")
        if self.status != "candidate":
            raise ValueError("TRAF-65 handoff must remain a candidate until hardware scoring")
        if self.evidence_class != NVLINK_CANDIDATE_EVIDENCE_CLASS:
            raise ValueError("candidate profile must not claim measured evidence")
        if len(self.freeze_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.freeze_sha256
        ):
            raise ValueError("freeze_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.tx, NvlinkTxConfig):
            raise TypeError("tx must be an NvlinkTxConfig")
        if not isinstance(self.switch, NvlinkSwitchConfig):
            raise TypeError("switch must be an NvlinkSwitchConfig")
        if not isinstance(self.rx, NvlinkRxConfig):
            raise TypeError("rx must be an NvlinkRxConfig")


@dataclass(frozen=True, kw_only=True)
class NvlinkDomainResult:
    """One composed candidate-profile result and its exact byte ledger."""

    implementation: str
    profile_id: str
    packets: tuple[NvlinkPacket, ...]
    logical_bytes: int
    request_payload_bytes: int
    response_payload_bytes: int
    request_wire_bytes: int
    response_wire_bytes: int
    completion_time_ps: int
    max_rx_buffer_occupancy_bytes: int

    def canonical_json_bytes(self) -> bytes:
        """Return the stable conformance representation."""

        payload = asdict(self)
        return (
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        ).encode("utf-8")


@dataclass(frozen=True, kw_only=True)
class NvlinkEnvelopeValidation:
    """Post-specified comparison to already published A100 measurements."""

    predicted_pair_payload_rate_gbps: float
    measured_pair_min_gbps: float
    measured_pair_max_gbps: float
    pair_worst_relative_error: float
    predicted_fanout_payload_rate_gbps: float
    measured_fanout_gbps: float
    fanout_relative_error: float
    within_registered_error: bool


class NvlinkTx:
    """Packetize transfers, consume destination credits, and bond links."""

    def __init__(self, config: NvlinkTxConfig) -> None:
        if not isinstance(config, NvlinkTxConfig):
            raise TypeError("config must be an NvlinkTxConfig")
        self.config = config

    def packetize(self, transfer: NvlinkTransfer) -> tuple[NvlinkPacket, ...]:
        """Build request and response packets without scheduling them."""

        if not isinstance(transfer, NvlinkTransfer):
            raise TypeError("transfer must be an NvlinkTransfer")
        packets: list[NvlinkPacket] = []
        if transfer.operation is NvlinkOperation.PEER_READ:
            packets.append(
                self._packet(
                    transfer=transfer,
                    sequence=0,
                    source=transfer.source,
                    destination=transfer.destination,
                    direction=NvlinkPacketDirection.REQUEST,
                    payload_bytes=0,
                )
            )
            data_source = transfer.destination
            data_destination = transfer.source
            data_direction = NvlinkPacketDirection.RESPONSE
            first_sequence = 1
        else:
            data_source = transfer.source
            data_destination = transfer.destination
            data_direction = NvlinkPacketDirection.REQUEST
            first_sequence = 0
        remaining = transfer.payload_bytes
        sequence = first_sequence
        while remaining:
            payload_bytes = min(remaining, self.config.max_payload_bytes)
            packets.append(
                self._packet(
                    transfer=transfer,
                    sequence=sequence,
                    source=data_source,
                    destination=data_destination,
                    direction=data_direction,
                    payload_bytes=payload_bytes,
                )
            )
            remaining -= payload_bytes
            sequence += 1
        return tuple(packets)

    def transmit(
        self,
        packets: Sequence[NvlinkPacket],
        *,
        credit_return_latency_ps: int,
    ) -> tuple[NvlinkPacket, ...]:
        """Schedule directional link and endpoint egress serializers."""

        _require_nonnegative_int("credit_return_latency_ps", credit_return_latency_ps)
        link_cursors: dict[tuple[int, int, int], int] = {}
        endpoint_cursors: dict[int, int] = {}
        credit_slots: dict[tuple[int, int], list[int]] = {}
        scheduled = []
        for packet in packets:
            if not isinstance(packet, NvlinkPacket):
                raise TypeError("packets must contain NvlinkPacket records")
            pair = (packet.source, packet.destination)
            slots = credit_slots.setdefault(pair, [0] * self.config.credits_per_destination)
            slot_index = packet.sequence % self.config.credits_per_destination
            links = [
                link_cursors.get((packet.source, packet.destination, link), 0)
                for link in range(self.config.links_per_peer)
            ]
            link_index = min(range(len(links)), key=lambda candidate: (links[candidate], candidate))
            link_key = (packet.source, packet.destination, link_index)
            started_at_ps = max(
                packet.released_at_ps,
                links[link_index],
                endpoint_cursors.get(packet.source, 0),
                slots[slot_index],
            )
            link_duration = _serialize_ps(
                packet.wire_bytes, self.config.per_link_rate_bytes_per_second
            )
            endpoint_duration = _serialize_ps(
                packet.wire_bytes, self.config.endpoint_egress_rate_bytes_per_second
            )
            finished_at_ps = started_at_ps + link_duration
            link_cursors[link_key] = finished_at_ps
            endpoint_cursors[packet.source] = started_at_ps + endpoint_duration
            slots[slot_index] = finished_at_ps + credit_return_latency_ps
            scheduled.append(
                replace(
                    packet,
                    link_index=link_index,
                    tx_started_at_ps=started_at_ps,
                    tx_finished_at_ps=finished_at_ps,
                )
            )
        return tuple(scheduled)

    def _packet(
        self,
        *,
        transfer: NvlinkTransfer,
        sequence: int,
        source: int,
        destination: int,
        direction: NvlinkPacketDirection,
        payload_bytes: int,
    ) -> NvlinkPacket:
        return NvlinkPacket(
            extent_id=transfer.extent_id,
            attempt_id=f"{transfer.extent_id}:attempt-0:packet-{sequence}",
            sequence=sequence,
            source=source,
            destination=destination,
            direction=direction,
            payload_bytes=payload_bytes,
            header_bytes=self.config.header_bytes,
            wire_bytes=payload_bytes + self.config.header_bytes,
            released_at_ps=transfer.released_at_ps,
        )


class NvlinkSwitch:
    """Own port contention, FIFO placement, and head-of-line behavior."""

    def __init__(self, config: NvlinkSwitchConfig) -> None:
        if not isinstance(config, NvlinkSwitchConfig):
            raise TypeError("config must be an NvlinkSwitchConfig")
        self.config = config

    def forward(self, packets: tuple[NvlinkPacket, ...]) -> tuple[NvlinkPacket, ...]:
        """Forward packets; pass-through returns the exact tuple unchanged."""

        if not isinstance(packets, tuple) or any(
            not isinstance(packet, NvlinkPacket) for packet in packets
        ):
            raise TypeError("packets must be a tuple of NvlinkPacket records")
        if self.config.mode is NvlinkSwitchMode.PASS_THROUGH:
            return packets
        cursors: dict[tuple[object, ...], int] = {}
        forwarded = []
        for packet in packets:
            if packet.tx_finished_at_ps is None:
                raise ValueError("switch input packet has no TX completion")
            if packet.wire_bytes > int(self.config.buffer_capacity_bytes or 0):
                raise ValueError("packet exceeds declared NVLink switch buffer")
            key = self._queue_key(packet)
            started_at_ps = max(packet.tx_finished_at_ps, cursors.get(key, 0))
            finished_at_ps = started_at_ps + _serialize_ps(
                packet.wire_bytes, int(self.config.service_rate_bytes_per_second or 0)
            )
            cursors[key] = finished_at_ps
            forwarded.append(
                replace(
                    packet,
                    switch_started_at_ps=started_at_ps,
                    switch_finished_at_ps=finished_at_ps,
                )
            )
        return tuple(forwarded)

    def _queue_key(self, packet: NvlinkPacket) -> tuple[object, ...]:
        if self.config.fifo_placement is NvlinkFifoPlacement.INPUT:
            key: tuple[object, ...] = ("input", packet.source)
        elif self.config.fifo_placement is NvlinkFifoPlacement.OUTPUT:
            key = ("output", packet.destination)
        else:
            key = ("shared",)
        if self.config.head_of_line_blocking is False:
            return (*key, packet.extent_id)
        return key


class NvlinkRx:
    """Own ingress buffering, reassembly, credit return, and delivery."""

    def __init__(self, config: NvlinkRxConfig) -> None:
        if not isinstance(config, NvlinkRxConfig):
            raise TypeError("config must be an NvlinkRxConfig")
        self.config = config

    def receive(self, packets: Sequence[NvlinkPacket]) -> tuple[tuple[NvlinkPacket, ...], int]:
        """Schedule per-destination ingress and return maximum occupancy."""

        cursors: dict[int, int] = {}
        buffered: dict[int, deque[tuple[int, int]]] = {}
        occupancy: dict[int, int] = {}
        delivered = []
        max_occupancy = 0
        last_sequence: dict[str, int] = {}
        for packet in packets:
            if not isinstance(packet, NvlinkPacket):
                raise TypeError("packets must contain NvlinkPacket records")
            arrival = (
                packet.switch_finished_at_ps
                if packet.switch_finished_at_ps is not None
                else packet.tx_finished_at_ps
            )
            if arrival is None:
                raise ValueError("RX input packet has no upstream completion")
            if packet.wire_bytes > self.config.buffer_capacity_bytes:
                raise ValueError("packet exceeds declared NVLink RX buffer")
            previous = last_sequence.get(packet.extent_id, -1)
            if packet.sequence <= previous:
                raise ValueError("NVLink RX sequence is not strictly increasing per extent")
            last_sequence[packet.extent_id] = packet.sequence
            queue = buffered.setdefault(packet.destination, deque())
            used = occupancy.get(packet.destination, 0)
            while queue and queue[0][0] <= arrival:
                _, released_bytes = queue.popleft()
                used -= released_bytes
            used += packet.wire_bytes
            if used > self.config.buffer_capacity_bytes:
                raise ValueError("packets exceed declared NVLink RX buffer occupancy")
            started_at_ps = max(arrival, cursors.get(packet.destination, 0))
            finished_at_ps = started_at_ps + _serialize_ps(
                packet.wire_bytes, self.config.ingress_rate_bytes_per_second
            )
            cursors[packet.destination] = finished_at_ps
            queue.append((finished_at_ps, packet.wire_bytes))
            occupancy[packet.destination] = used
            max_occupancy = max(max_occupancy, used)
            delivered.append(
                replace(
                    packet,
                    rx_started_at_ps=started_at_ps,
                    rx_finished_at_ps=finished_at_ps,
                    delivered_at_ps=finished_at_ps,
                )
            )
        return tuple(delivered), max_occupancy


class NvlinkDomainService(Generic[_AnalyticResult]):
    """Compose TX, switch, and RX, or preserve the analytical bypass exactly."""

    def __init__(self, profile: NvlinkCandidateProfile | None = None) -> None:
        if profile is not None and not isinstance(profile, NvlinkCandidateProfile):
            raise TypeError("profile must be an NvlinkCandidateProfile or None")
        self.profile = profile

    def serve(
        self,
        transfers: Sequence[NvlinkTransfer],
        *,
        analytic_result: _AnalyticResult,
        include_switch: bool = True,
    ) -> _AnalyticResult | NvlinkDomainResult:
        """Return the exact bypass object or the selected packet service."""

        if self.profile is None:
            return analytic_result
        if type(include_switch) is not bool:
            raise TypeError("include_switch must be a boolean")
        tx = NvlinkTx(self.profile.tx)
        packetized = _interleave_packetized(tx, transfers)
        transmitted = tx.transmit(
            packetized,
            credit_return_latency_ps=self.profile.rx.credit_return_latency_ps,
        )
        if include_switch:
            forwarded = NvlinkSwitch(self.profile.switch).forward(transmitted)
        else:
            forwarded = transmitted
        delivered, max_occupancy = NvlinkRx(self.profile.rx).receive(forwarded)
        request_packets = tuple(
            packet for packet in delivered if packet.direction is NvlinkPacketDirection.REQUEST
        )
        response_packets = tuple(
            packet for packet in delivered if packet.direction is NvlinkPacketDirection.RESPONSE
        )
        return NvlinkDomainResult(
            implementation=NVLINK_CANDIDATE_PROFILE_IMPLEMENTATION,
            profile_id=self.profile.profile_id,
            packets=delivered,
            logical_bytes=sum(transfer.payload_bytes for transfer in transfers),
            request_payload_bytes=sum(packet.payload_bytes for packet in request_packets),
            response_payload_bytes=sum(packet.payload_bytes for packet in response_packets),
            request_wire_bytes=sum(packet.wire_bytes for packet in request_packets),
            response_wire_bytes=sum(packet.wire_bytes for packet in response_packets),
            completion_time_ps=max(
                (packet.delivered_at_ps or 0 for packet in delivered), default=0
            ),
            max_rx_buffer_occupancy_bytes=max_occupancy,
        )


def load_nvlink_candidate_profile(path: str | Path) -> NvlinkCandidateProfile:
    """Load the versioned candidate-profile handoff."""

    with open(path, encoding="utf-8", newline="") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise TypeError("NVLink candidate profile must be a JSON object")
    tx_raw = _mapping_field(raw, "tx")
    switch_raw = _mapping_field(raw, "switch")
    rx_raw = _mapping_field(raw, "rx")
    fifo_value = switch_raw.get("fifo_placement")
    return NvlinkCandidateProfile(
        schema=str(raw.get("schema", "")),
        profile_id=str(raw.get("profile_id", "")),
        status=str(raw.get("status", "")),
        evidence_class=str(raw.get("evidence_class", "")),
        freeze_sha256=str(raw.get("freeze_sha256", "")),
        tx=NvlinkTxConfig(**tx_raw),
        switch=NvlinkSwitchConfig(
            mode=NvlinkSwitchMode(str(switch_raw.get("mode", ""))),
            fifo_placement=(None if fifo_value is None else NvlinkFifoPlacement(str(fifo_value))),
            service_rate_bytes_per_second=switch_raw.get("service_rate_bytes_per_second"),
            buffer_capacity_bytes=switch_raw.get("buffer_capacity_bytes"),
            arbitration=switch_raw.get("arbitration"),
            head_of_line_blocking=switch_raw.get("head_of_line_blocking"),
        ),
        rx=NvlinkRxConfig(**rx_raw),
    )


def validate_candidate_against_published_a100_envelope(
    profile: NvlinkCandidateProfile,
) -> NvlinkEnvelopeValidation:
    """Compare composed service timing only to measurements published before TRAF-65."""

    if not isinstance(profile, NvlinkCandidateProfile):
        raise TypeError("profile must be an NvlinkCandidateProfile")
    extent_bytes = 2048 * profile.tx.max_payload_bytes
    service = NvlinkDomainService(profile)
    pair = service.serve(
        [
            NvlinkTransfer(
                extent_id="published-pair", source=0, destination=1, payload_bytes=extent_bytes
            )
        ],
        analytic_result=None,
    )
    fanout = service.serve(
        [
            NvlinkTransfer(
                extent_id=f"published-fanout-{destination}",
                source=0,
                destination=destination,
                payload_bytes=extent_bytes,
            )
            for destination in (1, 2, 3)
        ],
        analytic_result=None,
    )
    if not isinstance(pair, NvlinkDomainResult) or not isinstance(fanout, NvlinkDomainResult):
        raise TypeError("candidate profile did not produce composed NVLink results")
    pair_gbps = _payload_rate_gbps(pair)
    fanout_gbps = _payload_rate_gbps(fanout)
    measured_min = 94.0
    measured_max = 94.07
    measured_fanout = 281.65
    pair_error = max(
        abs(pair_gbps - measured_min) / measured_min,
        abs(pair_gbps - measured_max) / measured_max,
    )
    fanout_error = abs(fanout_gbps - measured_fanout) / measured_fanout
    return NvlinkEnvelopeValidation(
        predicted_pair_payload_rate_gbps=pair_gbps,
        measured_pair_min_gbps=measured_min,
        measured_pair_max_gbps=measured_max,
        pair_worst_relative_error=pair_error,
        predicted_fanout_payload_rate_gbps=fanout_gbps,
        measured_fanout_gbps=measured_fanout,
        fanout_relative_error=fanout_error,
        within_registered_error=pair_error <= 0.10 and fanout_error <= 0.10,
    )


def sha256_file(path: str | Path) -> str:
    """Hash a file without interpreting it."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _serialize_ps(byte_count: int, rate_bytes_per_second: int) -> int:
    _require_positive_int("byte_count", byte_count)
    _require_positive_int("rate_bytes_per_second", rate_bytes_per_second)
    numerator = byte_count * _PS_PER_SECOND
    return (numerator + rate_bytes_per_second - 1) // rate_bytes_per_second


def _interleave_packetized(
    tx: NvlinkTx,
    transfers: Sequence[NvlinkTransfer],
) -> tuple[NvlinkPacket, ...]:
    """Round-robin ready extents so fan-out can use independent bonded pairs."""

    packet_groups = []
    for transfer in transfers:
        if not isinstance(transfer, NvlinkTransfer):
            raise TypeError("transfers must contain NvlinkTransfer records")
        packet_groups.append(tx.packetize(transfer))
    return tuple(
        group[sequence]
        for sequence in range(max((len(group) for group in packet_groups), default=0))
        for group in packet_groups
        if sequence < len(group)
    )


def _payload_rate_gbps(result: NvlinkDomainResult) -> float:
    if result.completion_time_ps <= 0:
        raise ValueError("composed NVLink result has no positive completion time")
    return result.logical_bytes * 1000 / result.completion_time_ps


def _mapping_field(raw: Mapping[object, object], name: str) -> dict[str, object]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"candidate profile field {name!r} must be an object")
    return {str(key): item for key, item in value.items()}


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")


def _require_enum(name: str, value: object, enum_type: type[Enum]) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{name} must be a {enum_type.__name__}")


def _require_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_endpoint(name: str, value: object) -> None:
    _require_nonnegative_int(name, value)
    if int(value) > 3:
        raise ValueError(f"{name} must identify one of four A100 endpoints")
