"""Packet-level NVLink service for htsim-style compositions.

The active SimLLM intra-node path remains analytical.  This module is an
additive versioned-profile handoff: callers that provide no
profile receive their analytical result back by object identity.  Selecting a
profile composes three independently parameterized services in this order:

``TX -> switch -> RX``

The four-A100 NV4 profile uses an explicit pass-through switch. Queues,
arbitration, FIFO placement, and head-of-line blocking belong to the switch
module for future NVSwitch profiles, but none is inferred from a direct mesh.
TRAF-70 publishes evidence per parameter: measured fields, unchanged declared
candidates, and the structural direct-mesh switch invariant remain distinct.
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
NVLINK_SCORED_PROFILE_STATUS = "scored_mixed_parameter_evidence"
NVLINK_SCORED_EVIDENCE_CLASS = "parameter_specific_evidence_see_traf70_score"
NVLINK_REQUEST_RESPONSE_DIRECTION = (
    "write payload travels as request; read control travels as request and "
    "read payload travels as response"
)

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


class NvlinkFlowPolicy(str, Enum):
    """How multiple extents enter the independently parameterized modules."""

    STATIC_INTERLEAVE = "static_interleave"
    RELEASE_AWARE_ROUND_ROBIN = "release_aware_round_robin"


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
class NvlinkParameterEvidence:
    """Evidence class and frozen decision-rule result for one parameter."""

    module: str
    parameter: str
    status: str
    value: object
    candidate_relation: str
    evidence_class: str
    rule_id: str
    reason: str

    def __post_init__(self) -> None:
        for name in (
            "module",
            "parameter",
            "status",
            "candidate_relation",
            "evidence_class",
            "rule_id",
            "reason",
        ):
            _require_text(name, getattr(self, name))
        if self.status == "IDENTIFIED":
            if not self.evidence_class.startswith("measured_"):
                raise ValueError("identified NVLink evidence must name its measured class")
            if self.candidate_relation not in {"CONFIRMED", "REFUTED_AND_REPLACED"}:
                raise ValueError("identified NVLink evidence has an invalid candidate relation")
        elif self.status == "INCONCLUSIVE":
            if self.evidence_class != NVLINK_CANDIDATE_EVIDENCE_CLASS:
                raise ValueError("inconclusive NVLink evidence must remain declared candidate")
            if self.candidate_relation != "UNCHANGED":
                raise ValueError("inconclusive NVLink evidence must remain unchanged")
        elif self.status == "STRUCTURAL":
            if self.evidence_class != "structural_direct_mesh_invariant_not_measurement":
                raise ValueError("structural NVLink evidence must not claim measurement")
            if self.candidate_relation != "RETAINED_STRUCTURAL":
                raise ValueError("structural NVLink evidence has an invalid candidate relation")
        else:
            raise ValueError(f"unsupported NVLink parameter evidence status {self.status!r}")


@dataclass(frozen=True, kw_only=True)
class NvlinkPublishedParameter:
    """One parameter explicitly carried by the score publication patch."""

    module: str
    parameter: str
    value: object
    candidate_relation: str
    evidence_class: str
    rule_id: str
    publication_surface: str

    def __post_init__(self) -> None:
        for name in (
            "module",
            "parameter",
            "candidate_relation",
            "evidence_class",
            "rule_id",
            "publication_surface",
        ):
            _require_text(name, getattr(self, name))
        if self.publication_surface not in {
            "runtime_profile",
            "existing_htsim_directional_packetization",
        }:
            raise ValueError("unsupported NVLink publication surface")


@dataclass(frozen=True, kw_only=True)
class NvlinkScorePublication:
    """Digest-bound TRAF-70 score metadata attached to a published profile."""

    score_sha256: str
    score_status: str
    protected_candidate_before_sha256: str
    scheduler_job: str
    runtime_changes: tuple[NvlinkPublishedParameter, ...]
    metadata_only_changes: tuple[NvlinkPublishedParameter, ...]
    unchanged_parameter_count: int

    def __post_init__(self) -> None:
        for name in (
            "score_sha256",
            "score_status",
            "protected_candidate_before_sha256",
            "scheduler_job",
        ):
            _require_text(name, getattr(self, name))
        for name in ("score_sha256", "protected_candidate_before_sha256"):
            _require_sha256(name, getattr(self, name))
        if self.score_status != "COMPLETE_VALID_86_OF_86":
            raise ValueError("NVLink publication must reference a complete valid score")
        if any(
            change.publication_surface != "runtime_profile"
            for change in self.runtime_changes
        ):
            raise ValueError("runtime NVLink changes have an invalid publication surface")
        if any(
            change.publication_surface != "existing_htsim_directional_packetization"
            for change in self.metadata_only_changes
        ):
            raise ValueError("metadata-only NVLink changes have an invalid publication surface")
        _require_nonnegative_int("unchanged_parameter_count", self.unchanged_parameter_count)


@dataclass(frozen=True, kw_only=True)
class NvlinkCandidateProfile:
    """Versioned packet profile with candidate or parameter-specific evidence."""

    profile_id: str
    status: str
    evidence_class: str
    freeze_sha256: str
    tx: NvlinkTxConfig
    switch: NvlinkSwitchConfig
    rx: NvlinkRxConfig
    schema: str = NVLINK_CANDIDATE_PROFILE_SCHEMA
    parameter_evidence: tuple[NvlinkParameterEvidence, ...] = ()
    score_publication: NvlinkScorePublication | None = None

    def __post_init__(self) -> None:
        for name in ("profile_id", "status", "evidence_class", "freeze_sha256"):
            _require_text(name, getattr(self, name))
        if self.schema != NVLINK_CANDIDATE_PROFILE_SCHEMA:
            raise ValueError(f"unsupported NVLink candidate schema {self.schema!r}")
        _require_sha256("freeze_sha256", self.freeze_sha256)
        if not isinstance(self.tx, NvlinkTxConfig):
            raise TypeError("tx must be an NvlinkTxConfig")
        if not isinstance(self.switch, NvlinkSwitchConfig):
            raise TypeError("switch must be an NvlinkSwitchConfig")
        if not isinstance(self.rx, NvlinkRxConfig):
            raise TypeError("rx must be an NvlinkRxConfig")
        if self.status == "candidate":
            if self.evidence_class != NVLINK_CANDIDATE_EVIDENCE_CLASS:
                raise ValueError("candidate profile must not claim measured evidence")
            if self.parameter_evidence or self.score_publication is not None:
                raise ValueError("unscored candidate must not contain score evidence")
            return
        if self.status != NVLINK_SCORED_PROFILE_STATUS:
            raise ValueError(f"unsupported NVLink profile status {self.status!r}")
        if self.evidence_class != NVLINK_SCORED_EVIDENCE_CLASS:
            raise ValueError("scored profile must declare parameter-specific evidence")
        if self.score_publication is None:
            raise ValueError("scored profile is missing its score publication")
        _validate_scored_profile(self)

    def evidence_for(self, module: str, parameter: str) -> NvlinkParameterEvidence:
        """Return the unique evidence record for a module parameter."""

        matches = tuple(
            evidence
            for evidence in self.parameter_evidence
            if evidence.module == module and evidence.parameter == parameter
        )
        if len(matches) != 1:
            raise KeyError(f"NVLink evidence is not uniquely defined for {module}.{parameter}")
        return matches[0]


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

    def transmit_flows(
        self,
        transfers: Sequence[NvlinkTransfer],
        *,
        credit_return_latency_ps: int,
    ) -> tuple[NvlinkPacket, ...]:
        """Schedule released extents round-robin without changing the static path."""

        _require_nonnegative_int("credit_return_latency_ps", credit_return_latency_ps)
        groups_by_source: dict[int, list[tuple[int, tuple[NvlinkPacket, ...]]]] = {}
        for transfer_order, transfer in enumerate(transfers):
            if not isinstance(transfer, NvlinkTransfer):
                raise TypeError("transfers must contain NvlinkTransfer records")
            packets_by_source: dict[int, list[NvlinkPacket]] = {}
            for packet in self.packetize(transfer):
                packets_by_source.setdefault(packet.source, []).append(packet)
            for source, packets in packets_by_source.items():
                groups_by_source.setdefault(source, []).append(
                    (transfer_order, tuple(packets))
                )

        scheduled: list[tuple[int, NvlinkPacket]] = []
        for groups in groups_by_source.values():
            scheduled.extend(
                self._transmit_source_flows(
                    groups,
                    credit_return_latency_ps=credit_return_latency_ps,
                )
            )
        scheduled.sort(
            key=lambda item: (
                item[1].tx_finished_at_ps,
                item[1].tx_started_at_ps,
                item[0],
                item[1].sequence,
                item[1].source,
                item[1].destination,
            )
        )
        return tuple(packet for _, packet in scheduled)

    def _transmit_source_flows(
        self,
        groups: list[tuple[int, tuple[NvlinkPacket, ...]]],
        *,
        credit_return_latency_ps: int,
    ) -> list[tuple[int, NvlinkPacket]]:
        ordered = sorted(
            range(len(groups)),
            key=lambda index: (
                groups[index][1][0].released_at_ps,
                groups[index][0],
                groups[index][1][0].sequence,
            ),
        )
        positions = [0] * len(groups)
        active: deque[int] = deque()
        inactive_index = 0
        now_ps = 0
        link_cursors: dict[tuple[int, int, int], int] = {}
        endpoint_cursors: dict[int, int] = {}
        credit_slots: dict[tuple[int, int], list[int]] = {}
        pair_visits: dict[tuple[int, int], int] = {}
        scheduled: list[tuple[int, NvlinkPacket]] = []

        def activate_ready() -> None:
            nonlocal inactive_index
            while inactive_index < len(ordered):
                group_index = ordered[inactive_index]
                first_packet = groups[group_index][1][0]
                if first_packet.released_at_ps > now_ps:
                    break
                active.append(group_index)
                inactive_index += 1

        while active or inactive_index < len(ordered):
            activate_ready()
            if not active:
                next_group = ordered[inactive_index]
                now_ps = max(now_ps, groups[next_group][1][0].released_at_ps)
                activate_ready()

            group_index = active.popleft()
            transfer_order, packets = groups[group_index]
            packet = packets[positions[group_index]]
            pair = (packet.source, packet.destination)
            slots = credit_slots.setdefault(
                pair, [0] * self.config.credits_per_destination
            )
            visit = pair_visits.get(pair, 0)
            slot_index = visit % self.config.credits_per_destination
            pair_visits[pair] = visit + 1
            links = [
                link_cursors.get((packet.source, packet.destination, link), 0)
                for link in range(self.config.links_per_peer)
            ]
            link_index = min(
                range(len(links)), key=lambda candidate: (links[candidate], candidate)
            )
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
                (
                    transfer_order,
                    replace(
                        packet,
                        link_index=link_index,
                        tx_started_at_ps=started_at_ps,
                        tx_finished_at_ps=finished_at_ps,
                    ),
                )
            )
            positions[group_index] += 1
            now_ps = endpoint_cursors[packet.source]
            activate_ready()
            if positions[group_index] < len(packets):
                active.append(group_index)
        return scheduled

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

    def receive_arrivals(
        self, packets: Sequence[NvlinkPacket]
    ) -> tuple[tuple[NvlinkPacket, ...], int]:
        """Drain physical arrivals, then expose each extent in sequence order."""

        indexed = []
        for index, packet in enumerate(packets):
            if not isinstance(packet, NvlinkPacket):
                raise TypeError("packets must contain NvlinkPacket records")
            arrival = (
                packet.switch_finished_at_ps
                if packet.switch_finished_at_ps is not None
                else packet.tx_finished_at_ps
            )
            if arrival is None:
                raise ValueError("RX input packet has no upstream completion")
            indexed.append((arrival, index, packet))
        indexed.sort(key=lambda item: (item[0], item[1]))

        cursors: dict[int, int] = {}
        buffered: dict[int, deque[tuple[int, int]]] = {}
        occupancy: dict[int, int] = {}
        ingressed: list[tuple[int, NvlinkPacket]] = []
        max_occupancy = 0
        for arrival, original_index, packet in indexed:
            if packet.wire_bytes > self.config.buffer_capacity_bytes:
                raise ValueError("packet exceeds declared NVLink RX buffer")
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
            ingressed.append(
                (
                    original_index,
                    replace(
                        packet,
                        rx_started_at_ps=started_at_ps,
                        rx_finished_at_ps=finished_at_ps,
                    ),
                )
            )

        by_extent: dict[str, list[tuple[int, NvlinkPacket]]] = {}
        for item in ingressed:
            by_extent.setdefault(item[1].extent_id, []).append(item)
        delivered = []
        for extent_packets in by_extent.values():
            extent_packets.sort(key=lambda item: item[1].sequence)
            extent_order = min(item[0] for item in extent_packets)
            previous_sequence = -1
            visible_at_ps = 0
            for original_index, packet in extent_packets:
                if packet.sequence <= previous_sequence:
                    raise ValueError("NVLink RX sequence is not strictly increasing per extent")
                previous_sequence = packet.sequence
                if packet.rx_finished_at_ps is None:
                    raise ValueError("NVLink RX packet has no ingress completion")
                visible_at_ps = max(visible_at_ps, packet.rx_finished_at_ps)
                delivered.append(
                    (
                        extent_order,
                        original_index,
                        replace(packet, delivered_at_ps=visible_at_ps),
                    )
                )
        delivered.sort(
            key=lambda item: (
                item[2].delivered_at_ps,
                item[0],
                item[2].sequence,
                item[1],
            )
        )
        return tuple(packet for _, _, packet in delivered), max_occupancy


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
        flow_policy: NvlinkFlowPolicy = NvlinkFlowPolicy.STATIC_INTERLEAVE,
    ) -> _AnalyticResult | NvlinkDomainResult:
        """Return the exact bypass object or the selected packet service."""

        if self.profile is None:
            return analytic_result
        if type(include_switch) is not bool:
            raise TypeError("include_switch must be a boolean")
        _require_enum("flow_policy", flow_policy, NvlinkFlowPolicy)
        tx = NvlinkTx(self.profile.tx)
        if flow_policy is NvlinkFlowPolicy.STATIC_INTERLEAVE:
            packetized = _interleave_packetized(tx, transfers)
            transmitted = tx.transmit(
                packetized,
                credit_return_latency_ps=self.profile.rx.credit_return_latency_ps,
            )
        else:
            transmitted = tx.transmit_flows(
                transfers,
                credit_return_latency_ps=self.profile.rx.credit_return_latency_ps,
            )
        if include_switch:
            forwarded = NvlinkSwitch(self.profile.switch).forward(transmitted)
        else:
            forwarded = transmitted
        rx = NvlinkRx(self.profile.rx)
        if flow_policy is NvlinkFlowPolicy.STATIC_INTERLEAVE:
            delivered, max_occupancy = rx.receive(forwarded)
        else:
            delivered, max_occupancy = rx.receive_arrivals(forwarded)
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
    """Load a versioned candidate or scored mixed-evidence profile."""

    with open(path, encoding="utf-8", newline="") as handle:
        raw = json.load(handle)
    return nvlink_candidate_profile_from_mapping(raw)


def nvlink_candidate_profile_from_mapping(
    raw: Mapping[object, object],
) -> NvlinkCandidateProfile:
    """Parse an already-loaded candidate or scored profile mapping."""

    if not isinstance(raw, Mapping):
        raise TypeError("NVLink candidate profile must be a JSON object")
    tx_raw = _mapping_field(raw, "tx")
    switch_raw = _mapping_field(raw, "switch")
    rx_raw = _mapping_field(raw, "rx")
    fifo_value = switch_raw.get("fifo_placement")
    parameter_evidence = _load_parameter_evidence(raw.get("parameter_evidence"))
    score_publication = _load_score_publication(raw.get("traf70_score_publication"))
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
        parameter_evidence=parameter_evidence,
        score_publication=score_publication,
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


def _load_parameter_evidence(raw: object) -> tuple[NvlinkParameterEvidence, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise TypeError("NVLink parameter_evidence must be an object")
    records = []
    for module, module_raw in raw.items():
        if not isinstance(module, str) or not isinstance(module_raw, Mapping):
            raise TypeError("NVLink parameter_evidence modules must be named objects")
        for parameter, evidence_raw in module_raw.items():
            if not isinstance(parameter, str) or not isinstance(evidence_raw, Mapping):
                raise TypeError("NVLink parameter evidence entries must be named objects")
            records.append(
                NvlinkParameterEvidence(
                    module=module,
                    parameter=parameter,
                    status=str(evidence_raw.get("status", "")),
                    value=evidence_raw.get("value"),
                    candidate_relation=str(evidence_raw.get("candidate_relation", "")),
                    evidence_class=str(evidence_raw.get("evidence_class", "")),
                    rule_id=str(evidence_raw.get("rule_id", "")),
                    reason=str(evidence_raw.get("reason", "")),
                )
            )
    return tuple(records)


def _load_score_publication(raw: object) -> NvlinkScorePublication | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError("NVLink traf70_score_publication must be an object")
    return NvlinkScorePublication(
        score_sha256=str(raw.get("score_sha256", "")),
        score_status=str(raw.get("score_status", "")),
        protected_candidate_before_sha256=str(
            raw.get("protected_candidate_before_sha256", "")
        ),
        scheduler_job=str(raw.get("scheduler_job", "")),
        runtime_changes=_load_published_changes(raw.get("runtime_changes")),
        metadata_only_changes=_load_published_changes(raw.get("metadata_only_changes")),
        unchanged_parameter_count=raw.get("unchanged_parameter_count", -1),
    )


def _load_published_changes(raw: object) -> tuple[NvlinkPublishedParameter, ...]:
    if not isinstance(raw, list):
        raise TypeError("NVLink published parameter changes must be a list")
    changes = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise TypeError("NVLink published parameter changes must be objects")
        changes.append(
            NvlinkPublishedParameter(
                module=str(item.get("module", "")),
                parameter=str(item.get("parameter", "")),
                value=item.get("value"),
                candidate_relation=str(item.get("candidate_relation", "")),
                evidence_class=str(item.get("evidence_class", "")),
                rule_id=str(item.get("rule_id", "")),
                publication_surface=str(item.get("publication_surface", "")),
            )
        )
    return tuple(changes)


def _validate_scored_profile(profile: NvlinkCandidateProfile) -> None:
    expected_values: dict[tuple[str, str], object] = {
        ("tx", "max_payload_bytes"): profile.tx.max_payload_bytes,
        ("tx", "header_bytes"): profile.tx.header_bytes,
        ("tx", "links_per_peer"): profile.tx.links_per_peer,
        ("tx", "per_link_rate_bytes_per_second"): (
            profile.tx.per_link_rate_bytes_per_second
        ),
        ("tx", "endpoint_egress_rate_bytes_per_second"): (
            profile.tx.endpoint_egress_rate_bytes_per_second
        ),
        ("tx", "bond_policy"): profile.tx.bond_policy,
        ("tx", "credits_per_destination"): profile.tx.credits_per_destination,
        ("tx", "credit_unit_bytes"): profile.tx.credit_unit_bytes,
        ("tx", "request_response_direction"): NVLINK_REQUEST_RESPONSE_DIRECTION,
        ("switch", "mode"): profile.switch.mode.value,
        ("rx", "ingress_rate_bytes_per_second"): profile.rx.ingress_rate_bytes_per_second,
        ("rx", "buffer_capacity_bytes"): profile.rx.buffer_capacity_bytes,
        ("rx", "credit_return_latency_ps"): profile.rx.credit_return_latency_ps,
        ("rx", "reassembly_policy"): profile.rx.reassembly_policy,
        ("rx", "delivery_order"): profile.rx.delivery_order,
        ("tx_rx", "queue_scope"): None,
    }
    evidence_by_key = {
        (evidence.module, evidence.parameter): evidence
        for evidence in profile.parameter_evidence
    }
    if len(evidence_by_key) != len(profile.parameter_evidence):
        raise ValueError("scored NVLink profile contains duplicate parameter evidence")
    if evidence_by_key.keys() != expected_values.keys():
        raise ValueError("scored NVLink profile does not cover the complete parameter catalog")
    for key, expected_value in expected_values.items():
        if evidence_by_key[key].value != expected_value:
            raise ValueError(f"scored evidence value does not match runtime parameter {key}")

    publication = profile.score_publication
    if publication is None:
        raise ValueError("scored profile is missing its score publication")
    published_changes = (*publication.runtime_changes, *publication.metadata_only_changes)
    published_by_key = {
        (change.module, change.parameter): change for change in published_changes
    }
    if len(published_by_key) != len(published_changes):
        raise ValueError("scored NVLink profile publishes a parameter more than once")
    identified_keys = {
        key for key, evidence in evidence_by_key.items() if evidence.status == "IDENTIFIED"
    }
    if published_by_key.keys() != identified_keys:
        raise ValueError("score publication does not exactly match identified parameters")
    if publication.unchanged_parameter_count != len(expected_values) - len(identified_keys):
        raise ValueError("score publication has an invalid unchanged parameter count")
    for key, change in published_by_key.items():
        evidence = evidence_by_key[key]
        for name in ("value", "candidate_relation", "evidence_class", "rule_id"):
            if getattr(change, name) != getattr(evidence, name):
                raise ValueError(f"published change does not match parameter evidence for {key}")
    runtime_keys = {(change.module, change.parameter) for change in publication.runtime_changes}
    if runtime_keys - (expected_values.keys() - {("tx", "request_response_direction")}):
        raise ValueError("score publication names an unknown runtime parameter")
    metadata_keys = {
        (change.module, change.parameter) for change in publication.metadata_only_changes
    }
    if metadata_keys - {("tx", "request_response_direction")}:
        raise ValueError("score publication has an invalid metadata-only parameter set")


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")


def _require_sha256(name: str, value: object) -> None:
    _require_text(name, value)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


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
