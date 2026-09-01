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
from heapq import heappop, heappush
from pathlib import Path
from typing import Generic, TypeVar

NVLINK_CANDIDATE_PROFILE_SCHEMA = "simllm-htsim-nvlink-candidate-profile-v1"
NVLINK_CANDIDATE_PROFILE_IMPLEMENTATION = "simllm-htsim-nvlink-domain-v1"
NVLINK_ALIGNED_PROFILE_IMPLEMENTATION = "simllm-htsim-nvlink-domain-v2"
NVLINK_CANDIDATE_EVIDENCE_CLASS = "declared_candidate_not_hardware_measurement"
NVLINK_PUBLIC_MECHANISM_EVIDENCE_CLASS = "public_document_generation_scoped_mechanism"
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


class NvlinkMechanismAuthority(str, Enum):
    """Select the one packet and timing authority for a domain run."""

    COMPATIBILITY = "compatibility_v1"
    ALIGNED = "aligned_v2"


class NvlinkTrafficClass(str, Enum):
    """Transaction role carried independently from virtual-channel identity."""

    POSTED_REQUEST = "posted_request"
    NON_POSTED_REQUEST = "non_posted_request"
    RESPONSE = "response"
    LINK_CONTROL = "link_control"


class NvlinkCreditPoolScope(str, Enum):
    """Configurable ownership scope for an unidentified product credit pool."""

    LINK_DESTINATION_VIRTUAL_CHANNEL = "link_destination_virtual_channel"


class NvlinkCreditAccounting(str, Enum):
    """Candidate mapping from variable packet occupancy to credit consumption."""

    PACKET_SLOT = "packet_slot"
    FIXED_BYTE_QUANTUM = "fixed_byte_quantum"


class NvlinkSwitchMode(str, Enum):
    """Whether the stage is inert or owns an explicit contention service."""

    PASS_THROUGH = "pass_through"
    QUEUED = "queued"


class NvlinkFlowPolicy(str, Enum):
    """Compatibility scheduling used by merged pre-TRAF-73 consumers."""

    STATIC_INTERLEAVE = "static_interleave"
    RELEASE_AWARE_ROUND_ROBIN = "release_aware_round_robin"


LEGACY_NVLINK_FLOW_POLICY = NvlinkFlowPolicy.STATIC_INTERLEAVE


class NvlinkArbitrationPolicy(str, Enum):
    """How a contended receiver chooses among independently credited links."""

    RELEASE_AWARE_ROUND_ROBIN = "release_aware_round_robin"
    STATIC_INTERLEAVE = "static_interleave"
    GREEDY_CAPTURE = "greedy_capture"


DEFAULT_NVLINK_ARBITRATION_POLICY = NvlinkArbitrationPolicy.RELEASE_AWARE_ROUND_ROBIN


class NvlinkSwitchArbitration(str, Enum):
    """Policy seam for legal virtual-output-queue heads."""

    IDENTITY = "identity"
    ROUND_ROBIN_CANDIDATE = "round_robin_candidate"


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
    topology_endpoint_count: int = 4
    offered_rate_bytes_per_second: int | None = None
    traffic_class: NvlinkTrafficClass | None = None
    virtual_channel: str = "vc0"
    ordering_domain: str | None = None
    address_extension_flits: int = 0
    byte_enable_flits: int = 0

    def __post_init__(self) -> None:
        _require_text("extent_id", self.extent_id)
        _require_positive_int("topology_endpoint_count", self.topology_endpoint_count)
        _require_endpoint("source", self.source, self.topology_endpoint_count)
        _require_endpoint("destination", self.destination, self.topology_endpoint_count)
        if self.source == self.destination:
            raise ValueError("NVLink source and destination must differ")
        _require_positive_int("payload_bytes", self.payload_bytes)
        _require_enum("operation", self.operation, NvlinkOperation)
        _require_nonnegative_int("released_at_ps", self.released_at_ps)
        if self.offered_rate_bytes_per_second is not None:
            _require_positive_int(
                "offered_rate_bytes_per_second",
                self.offered_rate_bytes_per_second,
            )
        if self.traffic_class is not None:
            _require_enum("traffic_class", self.traffic_class, NvlinkTrafficClass)
        _require_text("virtual_channel", self.virtual_channel)
        if self.ordering_domain is not None:
            _require_text("ordering_domain", self.ordering_domain)
        for name in ("address_extension_flits", "byte_enable_flits"):
            value = getattr(self, name)
            _require_nonnegative_int(name, value)
            if value > 1:
                raise ValueError(f"{name} must be zero or one")


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
        _require_nonnegative_int("source", self.source)
        _require_nonnegative_int("destination", self.destination)
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


@dataclass(frozen=True, kw_only=True)
class NvlinkPacketFormat:
    """Generation-scoped flit structure without product-field promotion."""

    generation_scope: str
    flit_bytes: int
    header_flits: int
    maximum_payload_flits: int
    maximum_packet_flits: int
    evidence_class: str
    provenance: str

    def __post_init__(self) -> None:
        for name in ("generation_scope", "evidence_class", "provenance"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        for name in (
            "flit_bytes",
            "header_flits",
            "maximum_payload_flits",
            "maximum_packet_flits",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.flit_bytes != 16:
            raise ValueError("the documented NVLink family flit is 16 bytes")
        if self.header_flits != 1:
            raise ValueError("the documented NVLink family header occupies one flit")
        if self.maximum_payload_flits > self.maximum_packet_flits - self.header_flits:
            raise ValueError("payload flits exceed the complete packet bound")


A100_NVLINK3_PACKET_FORMAT_CANDIDATE = NvlinkPacketFormat(
    generation_scope="a100_nvlink3_candidate_from_pascal_documented_family",
    flit_bytes=16,
    header_flits=1,
    maximum_payload_flits=16,
    maximum_packet_flits=18,
    evidence_class=NVLINK_CANDIDATE_EVIDENCE_CLASS,
    provenance=(
        "TRAF-79 confirms the 16-byte documented-family flit. A100 packet-field "
        "continuity, the 256-byte payload bound and optional-flit selection remain "
        "DECLARED CANDIDATES owned by TRAF-73."
    ),
)


@dataclass(frozen=True, kw_only=True)
class NvlinkFlowControlConfig:
    """Explicit credit and virtual-channel state with candidate provenance."""

    virtual_channels: tuple[str, ...]
    credits_per_pool: int
    pool_scope: NvlinkCreditPoolScope
    accounting: NvlinkCreditAccounting
    credit_quantum_bytes: int
    return_transport_latency_ps: int
    evidence_class: str
    provenance: str

    def __post_init__(self) -> None:
        if not self.virtual_channels:
            raise ValueError("at least one explicit NVLink virtual channel is required")
        if len(set(self.virtual_channels)) != len(self.virtual_channels):
            raise ValueError("NVLink virtual-channel identities must be unique")
        for virtual_channel in self.virtual_channels:
            _require_text("virtual_channel", virtual_channel)
        _require_positive_int("credits_per_pool", self.credits_per_pool)
        _require_enum("pool_scope", self.pool_scope, NvlinkCreditPoolScope)
        _require_enum("accounting", self.accounting, NvlinkCreditAccounting)
        _require_positive_int("credit_quantum_bytes", self.credit_quantum_bytes)
        _require_nonnegative_int(
            "return_transport_latency_ps", self.return_transport_latency_ps
        )
        for name in ("evidence_class", "provenance"):
            _require_text(name, getattr(self, name))

    @classmethod
    def from_candidate_profile(
        cls,
        profile: NvlinkCandidateProfile,
    ) -> NvlinkFlowControlConfig:
        """Project merged candidates into explicit state without promotion."""

        if not isinstance(profile, NvlinkCandidateProfile):
            raise TypeError("profile must be an NvlinkCandidateProfile")
        return cls(
            virtual_channels=("vc0",),
            credits_per_pool=profile.tx.credits_per_destination,
            pool_scope=NvlinkCreditPoolScope.LINK_DESTINATION_VIRTUAL_CHANNEL,
            accounting=NvlinkCreditAccounting.PACKET_SLOT,
            credit_quantum_bytes=profile.tx.credit_unit_bytes,
            return_transport_latency_ps=profile.rx.credit_return_latency_ps,
            evidence_class=NVLINK_CANDIDATE_EVIDENCE_CLASS,
            provenance=(
                "TRAF-79 leaves A100 credit quantum, pool scope, virtual-channel "
                "count, pool depth and return encoding unidentified. The merged "
                "values remain DECLARED CANDIDATES for TRAF-73."
            ),
        )


@dataclass(frozen=True, kw_only=True)
class NvlinkAlignedOptions:
    """Opt-in structural controls for the aligned domain authority."""

    packet_format: NvlinkPacketFormat = A100_NVLINK3_PACKET_FORMAT_CANDIDATE
    flow_control: NvlinkFlowControlConfig | None = None
    switch_arbitration: NvlinkSwitchArbitration = NvlinkSwitchArbitration.IDENTITY
    acknowledgement_latency_ps: int = 0
    replay_timeout_ps: int = 0
    replay_counts: tuple[tuple[str, int], ...] = ()
    arbitration_provenance: str = (
        "Identity is the off policy. Round robin is a DECLARED CANDIDATE policy "
        "because TRAF-79 does not identify a deployed NVSwitch product arbiter."
    )

    def __post_init__(self) -> None:
        if not isinstance(self.packet_format, NvlinkPacketFormat):
            raise TypeError("packet_format must be an NvlinkPacketFormat")
        if self.flow_control is not None and not isinstance(
            self.flow_control, NvlinkFlowControlConfig
        ):
            raise TypeError("flow_control must be an NvlinkFlowControlConfig or None")
        _require_enum(
            "switch_arbitration",
            self.switch_arbitration,
            NvlinkSwitchArbitration,
        )
        _require_nonnegative_int(
            "acknowledgement_latency_ps", self.acknowledgement_latency_ps
        )
        _require_nonnegative_int("replay_timeout_ps", self.replay_timeout_ps)
        _require_text("arbitration_provenance", self.arbitration_provenance)
        packet_ids = set()
        for packet_id, count in self.replay_counts:
            _require_text("replay packet_id", packet_id)
            _require_positive_int("replay count", count)
            if packet_id in packet_ids:
                raise ValueError("replay packet identities must be unique")
            packet_ids.add(packet_id)


@dataclass(frozen=True, kw_only=True)
class NvlinkFlitPacket:
    """One aligned packet with flit, reliability and visibility identity."""

    extent_id: str
    packet_id: str
    sequence: int
    source: int
    destination: int
    direction: NvlinkPacketDirection
    traffic_class: NvlinkTrafficClass
    virtual_channel: str
    ordering_domain: str
    generation_scope: str
    payload_bytes: int
    padding_bytes: int
    header_flits: int
    address_extension_flits: int
    byte_enable_flits: int
    payload_flits: int
    wire_flits: int
    wire_bytes: int
    credit_units: int
    released_at_ps: int
    link_index: int | None = None
    input_port: int | None = None
    output_port: int | None = None
    tx_started_at_ps: int | None = None
    tx_finished_at_ps: int | None = None
    acknowledged_at_ps: int | None = None
    replay_buffer_released_at_ps: int | None = None
    replay_count: int = 0
    replay_wire_bytes: int = 0
    replay_time_ps: int = 0
    switch_started_at_ps: int | None = None
    switch_finished_at_ps: int | None = None
    rx_buffer_accepted_at_ps: int | None = None
    rx_started_at_ps: int | None = None
    rx_finished_at_ps: int | None = None
    rx_buffer_released_at_ps: int | None = None
    credit_available_at_ps: int | None = None
    visible_at_ps: int | None = None
    random_draw_count: int = 0

    def __post_init__(self) -> None:
        for name in (
            "extent_id",
            "packet_id",
            "virtual_channel",
            "ordering_domain",
            "generation_scope",
        ):
            _require_text(name, getattr(self, name))
        _require_nonnegative_int("sequence", self.sequence)
        _require_nonnegative_int("source", self.source)
        _require_nonnegative_int("destination", self.destination)
        if self.source == self.destination:
            raise ValueError("NVLink packet source and destination must differ")
        _require_enum("direction", self.direction, NvlinkPacketDirection)
        _require_enum("traffic_class", self.traffic_class, NvlinkTrafficClass)
        for name in (
            "payload_bytes",
            "padding_bytes",
            "address_extension_flits",
            "byte_enable_flits",
            "payload_flits",
            "replay_count",
            "replay_wire_bytes",
            "replay_time_ps",
            "random_draw_count",
        ):
            _require_nonnegative_int(name, getattr(self, name))
        for name in ("header_flits", "wire_flits", "wire_bytes", "credit_units"):
            _require_positive_int(name, getattr(self, name))
        if self.address_extension_flits > 1 or self.byte_enable_flits > 1:
            raise ValueError("each documented optional packet field is zero or one flit")
        expected_flits = (
            self.header_flits
            + self.address_extension_flits
            + self.byte_enable_flits
            + self.payload_flits
        )
        if self.wire_flits != expected_flits:
            raise ValueError("wire_flits does not conserve packet field occupancy")
        if self.payload_bytes + self.padding_bytes != self.payload_flits * 16:
            raise ValueError("payload bytes and padding do not conserve payload flits")
        if self.wire_bytes != self.wire_flits * 16:
            raise ValueError("wire bytes do not conserve 16-byte flits")
        if self.replay_wire_bytes != self.replay_count * self.wire_bytes:
            raise ValueError("replay bytes do not conserve repeated packet occupancy")
        for name in (
            "link_index",
            "input_port",
            "output_port",
            "tx_started_at_ps",
            "tx_finished_at_ps",
            "acknowledged_at_ps",
            "replay_buffer_released_at_ps",
            "switch_started_at_ps",
            "switch_finished_at_ps",
            "rx_buffer_accepted_at_ps",
            "rx_started_at_ps",
            "rx_finished_at_ps",
            "rx_buffer_released_at_ps",
            "credit_available_at_ps",
            "visible_at_ps",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_nonnegative_int(name, value)


@dataclass(frozen=True, kw_only=True)
class NvlinkCreditRelease:
    """Receiver-owned capacity release and later sender-visible return."""

    packet_id: str
    source: int
    destination: int
    link_index: int
    virtual_channel: str
    credit_units: int
    buffer_released_at_ps: int
    credit_available_at_ps: int

    def __post_init__(self) -> None:
        _require_text("packet_id", self.packet_id)
        _require_text("virtual_channel", self.virtual_channel)
        for name in ("source", "destination", "link_index"):
            _require_nonnegative_int(name, getattr(self, name))
        _require_positive_int("credit_units", self.credit_units)
        _require_nonnegative_int("buffer_released_at_ps", self.buffer_released_at_ps)
        _require_nonnegative_int("credit_available_at_ps", self.credit_available_at_ps)
        if self.credit_available_at_ps < self.buffer_released_at_ps:
            raise ValueError("credit availability cannot precede receiver-buffer release")


@dataclass(frozen=True, kw_only=True)
class NvlinkSwitchGrant:
    """Read-only projection of one legal VOQ-to-crossbar grant."""

    packet_id: str
    input_port: int
    output_port: int
    virtual_channel: str
    started_at_ps: int
    finished_at_ps: int
    policy: NvlinkSwitchArbitration

    def __post_init__(self) -> None:
        _require_text("packet_id", self.packet_id)
        _require_text("virtual_channel", self.virtual_channel)
        for name in ("input_port", "output_port", "started_at_ps", "finished_at_ps"):
            _require_nonnegative_int(name, getattr(self, name))
        _require_enum("policy", self.policy, NvlinkSwitchArbitration)
        if self.finished_at_ps < self.started_at_ps:
            raise ValueError("NVLink switch grant finish cannot precede its start")


@dataclass(frozen=True, kw_only=True)
class NvlinkVisibilityEvent:
    """Read-only projection of ordered consumer visibility."""

    packet_id: str
    ordering_domain: str
    sequence: int
    rx_finished_at_ps: int
    visible_at_ps: int

    def __post_init__(self) -> None:
        _require_text("packet_id", self.packet_id)
        _require_text("ordering_domain", self.ordering_domain)
        for name in ("sequence", "rx_finished_at_ps", "visible_at_ps"):
            _require_nonnegative_int(name, getattr(self, name))
        if self.visible_at_ps < self.rx_finished_at_ps:
            raise ValueError("consumer visibility cannot precede RX completion")


@dataclass(frozen=True, kw_only=True)
class NvlinkAlignedDomainResult:
    """Aligned-domain result with exact conservation and authority ledgers."""

    implementation: str
    profile_id: str
    authority: NvlinkMechanismAuthority
    packets: tuple[NvlinkFlitPacket, ...]
    credit_releases: tuple[NvlinkCreditRelease, ...]
    switch_grants: tuple[NvlinkSwitchGrant, ...]
    visibility_events: tuple[NvlinkVisibilityEvent, ...]
    logical_bytes: int
    request_payload_bytes: int
    response_payload_bytes: int
    request_wire_bytes: int
    response_wire_bytes: int
    replay_wire_bytes: int
    total_wire_bytes: int
    acknowledgement_count: int
    replayed_packet_count: int
    replay_time_ps: int
    completion_time_ps: int
    max_rx_buffer_occupancy_bytes: int
    random_draw_count: int
    fixed_point_iterations: int

    def canonical_json_bytes(self) -> bytes:
        """Return the stable aligned conformance representation."""

        payload = asdict(self)
        return (
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            + "\n"
        ).encode("utf-8")


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
        offered_wire_bytes = 0
        if transfer.operation is NvlinkOperation.PEER_READ:
            request = self._packet(
                transfer=transfer,
                sequence=0,
                source=transfer.source,
                destination=transfer.destination,
                direction=NvlinkPacketDirection.REQUEST,
                payload_bytes=0,
                released_at_ps=self._offered_release_ps(
                    transfer, offered_wire_bytes
                ),
            )
            packets.append(request)
            offered_wire_bytes += request.wire_bytes
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
            packet = self._packet(
                transfer=transfer,
                sequence=sequence,
                source=data_source,
                destination=data_destination,
                direction=data_direction,
                payload_bytes=payload_bytes,
                released_at_ps=self._offered_release_ps(
                    transfer, offered_wire_bytes
                ),
            )
            packets.append(packet)
            offered_wire_bytes += packet.wire_bytes
            remaining -= payload_bytes
            sequence += 1
        return tuple(packets)

    def packetize_flits(
        self,
        transfer: NvlinkTransfer,
        *,
        packet_format: NvlinkPacketFormat = A100_NVLINK3_PACKET_FORMAT_CANDIDATE,
        flow_control: NvlinkFlowControlConfig,
    ) -> tuple[NvlinkFlitPacket, ...]:
        """Build generation-scoped packets for the aligned authority."""

        if not isinstance(transfer, NvlinkTransfer):
            raise TypeError("transfer must be an NvlinkTransfer")
        if not isinstance(packet_format, NvlinkPacketFormat):
            raise TypeError("packet_format must be an NvlinkPacketFormat")
        if not isinstance(flow_control, NvlinkFlowControlConfig):
            raise TypeError("flow_control must be an NvlinkFlowControlConfig")
        if transfer.virtual_channel not in flow_control.virtual_channels:
            raise ValueError("transfer selects an undeclared NVLink virtual channel")
        maximum_payload_bytes = min(
            self.config.max_payload_bytes,
            packet_format.maximum_payload_flits * packet_format.flit_bytes,
        )
        if maximum_payload_bytes <= 0:
            raise ValueError("aligned packet format has no payload capacity")

        packets: list[NvlinkFlitPacket] = []
        offered_wire_bytes = 0
        ordering_domain = transfer.ordering_domain or transfer.extent_id
        if transfer.operation is NvlinkOperation.PEER_READ:
            request = self._flit_packet(
                transfer=transfer,
                packet_format=packet_format,
                flow_control=flow_control,
                sequence=0,
                source=transfer.source,
                destination=transfer.destination,
                direction=NvlinkPacketDirection.REQUEST,
                traffic_class=(
                    transfer.traffic_class or NvlinkTrafficClass.NON_POSTED_REQUEST
                ),
                ordering_domain=ordering_domain,
                payload_bytes=0,
                released_at_ps=self._offered_release_ps(
                    transfer, offered_wire_bytes
                ),
            )
            packets.append(request)
            offered_wire_bytes += request.wire_bytes
            data_source = transfer.destination
            data_destination = transfer.source
            data_direction = NvlinkPacketDirection.RESPONSE
            data_class = NvlinkTrafficClass.RESPONSE
            first_sequence = 1
        else:
            data_source = transfer.source
            data_destination = transfer.destination
            data_direction = NvlinkPacketDirection.REQUEST
            data_class = transfer.traffic_class or NvlinkTrafficClass.POSTED_REQUEST
            first_sequence = 0

        remaining = transfer.payload_bytes
        sequence = first_sequence
        while remaining:
            payload_bytes = min(remaining, maximum_payload_bytes)
            packet = self._flit_packet(
                transfer=transfer,
                packet_format=packet_format,
                flow_control=flow_control,
                sequence=sequence,
                source=data_source,
                destination=data_destination,
                direction=data_direction,
                traffic_class=data_class,
                ordering_domain=ordering_domain,
                payload_bytes=payload_bytes,
                released_at_ps=self._offered_release_ps(
                    transfer, offered_wire_bytes
                ),
            )
            packets.append(packet)
            offered_wire_bytes += packet.wire_bytes
            remaining -= payload_bytes
            sequence += 1
        return tuple(packets)

    @staticmethod
    def _offered_release_ps(
        transfer: NvlinkTransfer,
        wire_bytes_before: int,
    ) -> int:
        if transfer.offered_rate_bytes_per_second is None or wire_bytes_before == 0:
            return transfer.released_at_ps
        return transfer.released_at_ps + _serialize_ps(
            wire_bytes_before,
            transfer.offered_rate_bytes_per_second,
        )

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
        credit_slots: dict[tuple[int, int, int], list[int]] = {}
        link_visits: dict[tuple[int, int, int], int] = {}
        scheduled = []
        for packet in packets:
            if not isinstance(packet, NvlinkPacket):
                raise TypeError("packets must contain NvlinkPacket records")
            pair = (packet.source, packet.destination)
            link_ready = []
            for link in range(self.config.links_per_peer):
                link_key = (*pair, link)
                slots = credit_slots.setdefault(
                    link_key,
                    [0] * self.config.credits_per_destination,
                )
                visit = link_visits.get(link_key, 0)
                slot_index = visit % self.config.credits_per_destination
                link_ready.append(
                    max(link_cursors.get(link_key, 0), slots[slot_index])
                )
            link_index = min(
                range(len(link_ready)),
                key=lambda candidate: (link_ready[candidate], candidate),
            )
            link_key = (packet.source, packet.destination, link_index)
            slots = credit_slots[link_key]
            visit = link_visits.get(link_key, 0)
            slot_index = visit % self.config.credits_per_destination
            link_visits[link_key] = visit + 1
            started_at_ps = max(
                packet.released_at_ps,
                link_ready[link_index],
                endpoint_cursors.get(packet.source, 0),
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
        credit_slots: dict[tuple[int, int, int], list[int]] = {}
        link_visits: dict[tuple[int, int, int], int] = {}
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
            link_ready = []
            for link in range(self.config.links_per_peer):
                link_key = (*pair, link)
                slots = credit_slots.setdefault(
                    link_key,
                    [0] * self.config.credits_per_destination,
                )
                visit = link_visits.get(link_key, 0)
                slot_index = visit % self.config.credits_per_destination
                link_ready.append(
                    max(link_cursors.get(link_key, 0), slots[slot_index])
                )
            link_index = min(
                range(len(link_ready)),
                key=lambda candidate: (link_ready[candidate], candidate),
            )
            link_key = (packet.source, packet.destination, link_index)
            slots = credit_slots[link_key]
            visit = link_visits.get(link_key, 0)
            slot_index = visit % self.config.credits_per_destination
            link_visits[link_key] = visit + 1
            started_at_ps = max(
                packet.released_at_ps,
                link_ready[link_index],
                endpoint_cursors.get(packet.source, 0),
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
        released_at_ps: int,
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
            released_at_ps=released_at_ps,
        )

    def _flit_packet(
        self,
        *,
        transfer: NvlinkTransfer,
        packet_format: NvlinkPacketFormat,
        flow_control: NvlinkFlowControlConfig,
        sequence: int,
        source: int,
        destination: int,
        direction: NvlinkPacketDirection,
        traffic_class: NvlinkTrafficClass,
        ordering_domain: str,
        payload_bytes: int,
        released_at_ps: int,
    ) -> NvlinkFlitPacket:
        payload_flits = (
            payload_bytes + packet_format.flit_bytes - 1
        ) // packet_format.flit_bytes
        padding_bytes = payload_flits * packet_format.flit_bytes - payload_bytes
        wire_flits = (
            packet_format.header_flits
            + transfer.address_extension_flits
            + transfer.byte_enable_flits
            + payload_flits
        )
        if wire_flits > packet_format.maximum_packet_flits:
            raise ValueError("optional fields exceed the generation-scoped packet bound")
        wire_bytes = wire_flits * packet_format.flit_bytes
        if flow_control.accounting is NvlinkCreditAccounting.PACKET_SLOT:
            credit_units = 1
        else:
            credit_units = (
                wire_bytes + flow_control.credit_quantum_bytes - 1
            ) // flow_control.credit_quantum_bytes
        return NvlinkFlitPacket(
            extent_id=transfer.extent_id,
            packet_id=f"{transfer.extent_id}:packet-{sequence}",
            sequence=sequence,
            source=source,
            destination=destination,
            direction=direction,
            traffic_class=traffic_class,
            virtual_channel=transfer.virtual_channel,
            ordering_domain=ordering_domain,
            generation_scope=packet_format.generation_scope,
            payload_bytes=payload_bytes,
            padding_bytes=padding_bytes,
            header_flits=packet_format.header_flits,
            address_extension_flits=transfer.address_extension_flits,
            byte_enable_flits=transfer.byte_enable_flits,
            payload_flits=payload_flits,
            wire_flits=wire_flits,
            wire_bytes=wire_bytes,
            credit_units=credit_units,
            released_at_ps=released_at_ps,
        )


@dataclass(frozen=True, kw_only=True)
class _NvlinkVoqHead:
    original_index: int
    packet: NvlinkFlitPacket


class NvlinkSwitchPolicy:
    """Select a legal maximal crossbar match from ready VOQ heads."""

    arbitration = NvlinkSwitchArbitration.IDENTITY

    def select(
        self,
        candidates: Sequence[_NvlinkVoqHead],
    ) -> tuple[_NvlinkVoqHead, ...]:
        used_inputs: set[int] = set()
        used_outputs: set[int] = set()
        selected = []
        for candidate in sorted(candidates, key=lambda item: item.original_index):
            packet = candidate.packet
            if packet.source in used_inputs or packet.destination in used_outputs:
                continue
            used_inputs.add(packet.source)
            used_outputs.add(packet.destination)
            selected.append(candidate)
        return tuple(selected)


class NvlinkIdentitySwitchPolicy(NvlinkSwitchPolicy):
    """Ignore class labels and retain deterministic baseline candidate order."""

    arbitration = NvlinkSwitchArbitration.IDENTITY


class NvlinkRoundRobinSwitchPolicy(NvlinkSwitchPolicy):
    """Declared candidate that rotates the baseline VOQ-head order."""

    arbitration = NvlinkSwitchArbitration.ROUND_ROBIN_CANDIDATE

    def __init__(self) -> None:
        self._cursor = 0

    def select(
        self,
        candidates: Sequence[_NvlinkVoqHead],
    ) -> tuple[_NvlinkVoqHead, ...]:
        ordered = sorted(candidates, key=lambda item: item.original_index)
        if not ordered:
            return ()
        offset = self._cursor % len(ordered)
        rotated = ordered[offset:] + ordered[:offset]
        used_inputs: set[int] = set()
        used_outputs: set[int] = set()
        selected = []
        for candidate in rotated:
            packet = candidate.packet
            if packet.source in used_inputs or packet.destination in used_outputs:
                continue
            used_inputs.add(packet.source)
            used_outputs.add(packet.destination)
            selected.append(candidate)
        self._cursor = (offset + max(1, len(selected))) % len(ordered)
        return tuple(selected)


class NvlinkSwitch:
    """Own pass-through identity or port, VOQ and crossbar contention."""

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

    def forward_flits(
        self,
        packets: tuple[NvlinkFlitPacket, ...],
        *,
        policy: NvlinkSwitchPolicy | None = None,
    ) -> tuple[tuple[NvlinkFlitPacket, ...], tuple[NvlinkSwitchGrant, ...]]:
        """Forward aligned packets through explicit ports, VOQs and outputs."""

        if not isinstance(packets, tuple) or any(
            not isinstance(packet, NvlinkFlitPacket) for packet in packets
        ):
            raise TypeError("packets must be a tuple of NvlinkFlitPacket records")
        selected_policy = policy or NvlinkIdentitySwitchPolicy()
        if not isinstance(selected_policy, NvlinkSwitchPolicy):
            raise TypeError("policy must be an NvlinkSwitchPolicy")
        if self.config.mode is NvlinkSwitchMode.PASS_THROUGH:
            return packets, ()
        service_rate = int(self.config.service_rate_bytes_per_second or 0)
        buffer_capacity = int(self.config.buffer_capacity_bytes or 0)
        if not packets:
            return (), ()

        arrivals = sorted(
            (
                (packet.tx_finished_at_ps, index, packet)
                for index, packet in enumerate(packets)
            ),
            key=lambda item: (
                -1 if item[0] is None else item[0],
                item[1],
            ),
        )
        if arrivals[0][0] is None:
            raise ValueError("switch input packet has no TX completion")
        voqs: dict[
            tuple[int, str, int], deque[tuple[int, NvlinkFlitPacket]]
        ] = {}
        voq_bytes: dict[tuple[int, str, int], int] = {}
        input_cursors: dict[int, int] = {}
        output_cursors: dict[int, int] = {}
        forwarded: list[tuple[int, NvlinkFlitPacket]] = []
        grants: list[NvlinkSwitchGrant] = []
        arrival_index = 0
        now_ps = int(arrivals[0][0] or 0)

        def enqueue(until_ps: int) -> None:
            nonlocal arrival_index
            while arrival_index < len(arrivals):
                arrival, original_index, packet = arrivals[arrival_index]
                if arrival is None:
                    raise ValueError("switch input packet has no TX completion")
                if arrival > until_ps:
                    break
                key = (packet.source, packet.virtual_channel, packet.destination)
                queued_bytes = voq_bytes.get(key, 0) + packet.wire_bytes
                if queued_bytes > buffer_capacity:
                    raise ValueError("packet exceeds declared NVLink switch VOQ capacity")
                voqs.setdefault(key, deque()).append((original_index, packet))
                voq_bytes[key] = queued_bytes
                arrival_index += 1

        while len(forwarded) < len(packets):
            enqueue(now_ps)
            candidates = []
            for key, queue in voqs.items():
                if not queue:
                    continue
                original_index, packet = queue[0]
                if input_cursors.get(packet.source, 0) > now_ps:
                    continue
                if output_cursors.get(packet.destination, 0) > now_ps:
                    continue
                if key != (
                    packet.source,
                    packet.virtual_channel,
                    packet.destination,
                ):
                    raise AssertionError("NVLink VOQ route identity changed")
                candidates.append(
                    _NvlinkVoqHead(
                        original_index=original_index,
                        packet=packet,
                    )
                )
            selected = selected_policy.select(candidates)
            if selected:
                for candidate in selected:
                    packet = candidate.packet
                    key = (
                        packet.source,
                        packet.virtual_channel,
                        packet.destination,
                    )
                    original_index, queued_packet = voqs[key].popleft()
                    if queued_packet.packet_id != packet.packet_id:
                        raise AssertionError("NVLink policy selected a non-head VOQ packet")
                    voq_bytes[key] -= packet.wire_bytes
                    finished_at_ps = now_ps + _serialize_ps(
                        packet.wire_bytes,
                        service_rate,
                    )
                    input_cursors[packet.source] = finished_at_ps
                    output_cursors[packet.destination] = finished_at_ps
                    scheduled = replace(
                        packet,
                        input_port=packet.source,
                        output_port=packet.destination,
                        switch_started_at_ps=now_ps,
                        switch_finished_at_ps=finished_at_ps,
                    )
                    forwarded.append((original_index, scheduled))
                    grants.append(
                        NvlinkSwitchGrant(
                            packet_id=packet.packet_id,
                            input_port=packet.source,
                            output_port=packet.destination,
                            virtual_channel=packet.virtual_channel,
                            started_at_ps=now_ps,
                            finished_at_ps=finished_at_ps,
                            policy=selected_policy.arbitration,
                        )
                    )
                continue

            next_times = [
                cursor
                for cursor in (*input_cursors.values(), *output_cursors.values())
                if cursor > now_ps
            ]
            if arrival_index < len(arrivals):
                arrival = arrivals[arrival_index][0]
                if arrival is None:
                    raise ValueError("switch input packet has no TX completion")
                next_times.append(arrival)
            if not next_times:
                raise AssertionError("NVLink switch has queued packets but no legal grant")
            now_ps = min(next_times)

        forwarded.sort(key=lambda item: item[0])
        grants.sort(key=lambda item: (item.started_at_ps, item.packet_id))
        return tuple(packet for _, packet in forwarded), tuple(grants)

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

    def receive_flits(
        self,
        packets: Sequence[NvlinkFlitPacket],
        *,
        flow_control: NvlinkFlowControlConfig,
    ) -> tuple[
        tuple[NvlinkFlitPacket, ...],
        int,
        tuple[NvlinkCreditRelease, ...],
        tuple[NvlinkVisibilityEvent, ...],
    ]:
        """Own aligned receive buffers, credit release and ordered visibility."""

        if not isinstance(flow_control, NvlinkFlowControlConfig):
            raise TypeError("flow_control must be an NvlinkFlowControlConfig")
        indexed = []
        for index, packet in enumerate(packets):
            if not isinstance(packet, NvlinkFlitPacket):
                raise TypeError("packets must contain NvlinkFlitPacket records")
            arrival = (
                packet.switch_finished_at_ps
                if packet.switch_finished_at_ps is not None
                else packet.tx_finished_at_ps
            )
            if arrival is None:
                raise ValueError("RX input packet has no upstream completion")
            if packet.link_index is None:
                raise ValueError("RX input packet has no link identity")
            if packet.virtual_channel not in flow_control.virtual_channels:
                raise ValueError("packet selects an undeclared NVLink virtual channel")
            indexed.append((arrival, index, packet))
        indexed.sort(key=lambda item: (item[0], item[1]))

        ingress_cursors: dict[int, int] = {}
        buffered: dict[tuple[int, str], deque[tuple[int, int]]] = {}
        occupancy: dict[tuple[int, str], int] = {}
        ingressed: list[tuple[int, NvlinkFlitPacket]] = []
        releases: list[NvlinkCreditRelease] = []
        max_occupancy = 0
        for physical_arrival, original_index, packet in indexed:
            if packet.wire_bytes > self.config.buffer_capacity_bytes:
                raise ValueError("packet exceeds declared NVLink RX buffer")
            key = (packet.destination, packet.virtual_channel)
            queue = buffered.setdefault(key, deque())
            used = occupancy.get(key, 0)
            admitted_at_ps = physical_arrival
            while queue and queue[0][0] <= admitted_at_ps:
                _, released_bytes = queue.popleft()
                used -= released_bytes
            while used + packet.wire_bytes > self.config.buffer_capacity_bytes:
                if not queue:
                    raise AssertionError("NVLink RX occupancy has no owning release")
                admitted_at_ps = max(admitted_at_ps, queue[0][0])
                while queue and queue[0][0] <= admitted_at_ps:
                    _, released_bytes = queue.popleft()
                    used -= released_bytes
            used += packet.wire_bytes
            started_at_ps = max(
                admitted_at_ps,
                ingress_cursors.get(packet.destination, 0),
            )
            finished_at_ps = started_at_ps + _serialize_ps(
                packet.wire_bytes,
                self.config.ingress_rate_bytes_per_second,
            )
            buffer_released_at_ps = finished_at_ps
            credit_available_at_ps = (
                buffer_released_at_ps + flow_control.return_transport_latency_ps
            )
            ingress_cursors[packet.destination] = finished_at_ps
            queue.append((buffer_released_at_ps, packet.wire_bytes))
            occupancy[key] = used
            max_occupancy = max(max_occupancy, used)
            scheduled = replace(
                packet,
                rx_buffer_accepted_at_ps=admitted_at_ps,
                rx_started_at_ps=started_at_ps,
                rx_finished_at_ps=finished_at_ps,
                rx_buffer_released_at_ps=buffer_released_at_ps,
                credit_available_at_ps=credit_available_at_ps,
            )
            ingressed.append((original_index, scheduled))
            releases.append(
                NvlinkCreditRelease(
                    packet_id=packet.packet_id,
                    source=packet.source,
                    destination=packet.destination,
                    link_index=packet.link_index,
                    virtual_channel=packet.virtual_channel,
                    credit_units=packet.credit_units,
                    buffer_released_at_ps=buffer_released_at_ps,
                    credit_available_at_ps=credit_available_at_ps,
                )
            )

        by_ordering_domain: dict[
            tuple[int, str], list[tuple[int, NvlinkFlitPacket]]
        ] = {}
        for item in ingressed:
            packet = item[1]
            by_ordering_domain.setdefault(
                (packet.destination, packet.ordering_domain), []
            ).append(item)

        visible_packets: list[tuple[int, NvlinkFlitPacket]] = []
        visibility_events: list[NvlinkVisibilityEvent] = []
        for domain_packets in by_ordering_domain.values():
            domain_packets.sort(key=lambda item: (item[1].sequence, item[0]))
            previous_sequence = -1
            visible_at_ps = 0
            for original_index, packet in domain_packets:
                if packet.sequence <= previous_sequence:
                    raise ValueError(
                        "NVLink visibility sequence is not strictly increasing per domain"
                    )
                previous_sequence = packet.sequence
                if packet.rx_finished_at_ps is None:
                    raise AssertionError("NVLink packet has no RX finish")
                visible_at_ps = max(visible_at_ps, packet.rx_finished_at_ps)
                visible = replace(packet, visible_at_ps=visible_at_ps)
                visible_packets.append((original_index, visible))
                visibility_events.append(
                    NvlinkVisibilityEvent(
                        packet_id=packet.packet_id,
                        ordering_domain=packet.ordering_domain,
                        sequence=packet.sequence,
                        rx_finished_at_ps=packet.rx_finished_at_ps,
                        visible_at_ps=visible_at_ps,
                    )
                )
        visible_packets.sort(
            key=lambda item: (
                item[1].visible_at_ps,
                item[0],
                item[1].sequence,
            )
        )
        visibility_events.sort(
            key=lambda item: (item.visible_at_ps, item.packet_id)
        )
        releases.sort(key=lambda item: item.packet_id)
        return (
            tuple(packet for _, packet in visible_packets),
            max_occupancy,
            tuple(releases),
            tuple(visibility_events),
        )

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
            while used + packet.wire_bytes > self.config.buffer_capacity_bytes:
                if not queue:
                    raise AssertionError("NVLink RX occupancy has no credit to return")
                arrival = max(arrival, queue[0][0])
                while queue and queue[0][0] <= arrival:
                    _, released_bytes = queue.popleft()
                    used -= released_bytes
            used += packet.wire_bytes
            started_at_ps = max(arrival, cursors.get(packet.destination, 0))
            finished_at_ps = started_at_ps + _serialize_ps(
                packet.wire_bytes, self.config.ingress_rate_bytes_per_second
            )
            cursors[packet.destination] = finished_at_ps
            queue.append(
                (
                    finished_at_ps + self.config.credit_return_latency_ps,
                    packet.wire_bytes,
                )
            )
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

    def receive_arbitrated(
        self,
        packets: Sequence[NvlinkPacket],
        *,
        policy: NvlinkArbitrationPolicy = DEFAULT_NVLINK_ARBITRATION_POLICY,
    ) -> tuple[tuple[NvlinkPacket, ...], int]:
        """Arbitrate independently credited input links at each destination."""

        _require_enum("policy", policy, NvlinkArbitrationPolicy)
        by_destination: dict[int, list[tuple[int, NvlinkPacket]]] = {}
        for index, packet in enumerate(packets):
            if not isinstance(packet, NvlinkPacket):
                raise TypeError("packets must contain NvlinkPacket records")
            if packet.switch_finished_at_ps is None and packet.tx_finished_at_ps is None:
                raise ValueError("RX input packet has no upstream completion")
            by_destination.setdefault(packet.destination, []).append((index, packet))

        delivered: list[tuple[int, NvlinkPacket]] = []
        max_occupancy = 0
        for destination_packets in by_destination.values():
            scheduled, occupancy = self._receive_one_destination(
                destination_packets,
                policy=policy,
            )
            delivered.extend(scheduled)
            max_occupancy = max(max_occupancy, occupancy)
        delivered.sort(
            key=lambda item: (
                item[1].delivered_at_ps,
                item[0],
                item[1].sequence,
            )
        )
        return tuple(packet for _, packet in delivered), max_occupancy

    def _receive_one_destination(
        self,
        indexed_packets: list[tuple[int, NvlinkPacket]],
        *,
        policy: NvlinkArbitrationPolicy,
    ) -> tuple[list[tuple[int, NvlinkPacket]], int]:
        arrivals = sorted(
            (
                (
                    _packet_arrival_ps(packet),
                    original_index,
                    packet,
                )
                for original_index, packet in indexed_packets
            ),
            key=lambda item: (item[0], item[1]),
        )
        source_order = list(dict.fromkeys(packet.source for _, _, packet in arrivals))
        queues: dict[int, deque[tuple[int, NvlinkPacket]]] = {
            source: deque() for source in source_order
        }
        future_arrivals: dict[int, deque[int]] = {source: deque() for source in source_order}
        for arrival, _, packet in arrivals:
            future_arrivals[packet.source].append(arrival)

        release_events: list[tuple[int, int, int]] = []
        arrival_index = 0
        event_index = 0
        occupancy = 0
        max_occupancy = 0
        now_ps = 0
        round_robin_index = 0
        static_index = 0
        greedy_index = 0
        captured_source = source_order[0]
        scheduled: list[tuple[int, NvlinkPacket]] = []

        def retire(until_ps: int) -> None:
            nonlocal occupancy
            while release_events and release_events[0][0] <= until_ps:
                _, _, released_bytes = heappop(release_events)
                occupancy -= released_bytes

        def enqueue(until_ps: int) -> None:
            nonlocal arrival_index, max_occupancy, occupancy
            while arrival_index < len(arrivals) and arrivals[arrival_index][0] <= until_ps:
                arrival, original_index, packet = arrivals[arrival_index]
                retire(arrival)
                future_arrivals[packet.source].popleft()
                queues[packet.source].append((original_index, packet))
                occupancy += packet.wire_bytes
                if occupancy > self.config.buffer_capacity_bytes:
                    raise ValueError("packets exceed declared NVLink RX buffer occupancy")
                max_occupancy = max(max_occupancy, occupancy)
                arrival_index += 1
            retire(until_ps)

        def source_is_active(source: int) -> bool:
            return bool(queues[source] or future_arrivals[source])

        def ready_round_robin(start_index: int, *, skip_source: int | None = None) -> int | None:
            for offset in range(len(source_order)):
                index = (start_index + offset) % len(source_order)
                source = source_order[index]
                if source == skip_source:
                    continue
                if queues[source]:
                    return index
            return None

        while len(scheduled) < len(arrivals):
            enqueue(now_ps)
            selected_index: int | None = None
            if policy is NvlinkArbitrationPolicy.STATIC_INTERLEAVE:
                while selected_index is None:
                    source = source_order[static_index]
                    if queues[source]:
                        selected_index = static_index
                        static_index = (static_index + 1) % len(source_order)
                        break
                    if future_arrivals[source]:
                        now_ps = max(now_ps, future_arrivals[source][0])
                        enqueue(now_ps)
                        continue
                    static_index = (static_index + 1) % len(source_order)
            elif policy is NvlinkArbitrationPolicy.GREEDY_CAPTURE:
                if queues[captured_source]:
                    selected_index = source_order.index(captured_source)
                else:
                    selected_index = ready_round_robin(
                        greedy_index,
                        skip_source=captured_source,
                    )
                    if selected_index is not None:
                        greedy_index = (selected_index + 1) % len(source_order)
            else:
                selected_index = ready_round_robin(round_robin_index)
                if selected_index is not None:
                    round_robin_index = (selected_index + 1) % len(source_order)

            if selected_index is None:
                if arrival_index >= len(arrivals):
                    raise AssertionError("NVLink arbitration has packets but no ready source")
                now_ps = max(now_ps, arrivals[arrival_index][0])
                continue

            source = source_order[selected_index]
            original_index, packet = queues[source].popleft()
            started_at_ps = max(now_ps, _packet_arrival_ps(packet))
            finished_at_ps = started_at_ps + _serialize_ps(
                packet.wire_bytes,
                self.config.ingress_rate_bytes_per_second,
            )
            event_index += 1
            heappush(
                release_events,
                (
                    finished_at_ps + self.config.credit_return_latency_ps,
                    event_index,
                    packet.wire_bytes,
                ),
            )
            scheduled.append(
                (
                    original_index,
                    replace(
                        packet,
                        rx_started_at_ps=started_at_ps,
                        rx_finished_at_ps=finished_at_ps,
                        delivered_at_ps=finished_at_ps,
                    ),
                )
            )
            now_ps = finished_at_ps

            if not source_is_active(captured_source):
                active = [source for source in source_order if source_is_active(source)]
                if active:
                    captured_source = active[0]

        return scheduled, max_occupancy


class _NvlinkAlignedEngine:
    """Couple link credits to receiver-owned releases until timing is stable."""

    def __init__(
        self,
        profile: NvlinkCandidateProfile,
        options: NvlinkAlignedOptions,
    ) -> None:
        if not isinstance(profile, NvlinkCandidateProfile):
            raise TypeError("profile must be an NvlinkCandidateProfile")
        if not isinstance(options, NvlinkAlignedOptions):
            raise TypeError("options must be an NvlinkAlignedOptions")
        self.profile = profile
        self.options = options
        self.flow_control = options.flow_control or NvlinkFlowControlConfig.from_candidate_profile(
            profile
        )

    def serve(
        self,
        transfers: Sequence[NvlinkTransfer],
        *,
        include_switch: bool,
    ) -> NvlinkAlignedDomainResult:
        tx = NvlinkTx(self.profile.tx)
        packetized = tuple(
            packet
            for transfer in transfers
            for packet in tx.packetize_flits(
                transfer,
                packet_format=self.options.packet_format,
                flow_control=self.flow_control,
            )
        )
        if not packetized:
            raise ValueError("aligned NVLink service requires at least one packet")
        if len({packet.packet_id for packet in packetized}) != len(packetized):
            raise ValueError("aligned NVLink packet identities must be globally unique")

        release_by_packet: dict[str, int] = {}
        previous_signature: tuple[tuple[object, ...], ...] | None = None
        delivered: tuple[NvlinkFlitPacket, ...] = ()
        credit_releases: tuple[NvlinkCreditRelease, ...] = ()
        switch_grants: tuple[NvlinkSwitchGrant, ...] = ()
        visibility_events: tuple[NvlinkVisibilityEvent, ...] = ()
        max_occupancy = 0
        fixed_point_iterations = 0
        for iteration in range(1, len(packetized) + 3):
            transmitted = self._transmit(
                packetized,
                credit_release_by_packet=release_by_packet,
            )
            if include_switch:
                forwarded, switch_grants = NvlinkSwitch(
                    self.profile.switch
                ).forward_flits(
                    transmitted,
                    policy=self._switch_policy(),
                )
            else:
                forwarded = transmitted
                switch_grants = ()
            (
                delivered,
                max_occupancy,
                credit_releases,
                visibility_events,
            ) = NvlinkRx(self.profile.rx).receive_flits(
                forwarded,
                flow_control=self.flow_control,
            )
            signature = self._timing_signature(delivered)
            fixed_point_iterations = iteration
            if signature == previous_signature:
                break
            previous_signature = signature
            release_by_packet = {
                release.packet_id: release.credit_available_at_ps
                for release in credit_releases
            }
        else:
            raise RuntimeError("NVLink receiver-owned credit timing did not converge")

        return self._result(
            transfers=transfers,
            delivered=delivered,
            credit_releases=credit_releases,
            switch_grants=switch_grants,
            visibility_events=visibility_events,
            max_occupancy=max_occupancy,
            fixed_point_iterations=fixed_point_iterations,
        )

    def _switch_policy(self) -> NvlinkSwitchPolicy:
        if self.options.switch_arbitration is NvlinkSwitchArbitration.IDENTITY:
            return NvlinkIdentitySwitchPolicy()
        return NvlinkRoundRobinSwitchPolicy()

    def _transmit(
        self,
        packets: tuple[NvlinkFlitPacket, ...],
        *,
        credit_release_by_packet: Mapping[str, int],
    ) -> tuple[NvlinkFlitPacket, ...]:
        link_cursors: dict[tuple[int, int, int], int] = {}
        endpoint_cursors: dict[int, int] = {}
        credit_slots: dict[tuple[int, int, int, str], list[int]] = {}
        credit_visits: dict[tuple[int, int, int, str], int] = {}
        replay_counts = dict(self.options.replay_counts)
        scheduled = []
        for packet in packets:
            pair = (packet.source, packet.destination)
            link_ready: list[tuple[int, tuple[int, ...]]] = []
            for link_index in range(self.profile.tx.links_per_peer):
                credit_key = (*pair, link_index, packet.virtual_channel)
                slots = credit_slots.setdefault(
                    credit_key,
                    [0] * self.flow_control.credits_per_pool,
                )
                visit = credit_visits.get(credit_key, 0)
                slot_indices = tuple(
                    (visit + offset) % self.flow_control.credits_per_pool
                    for offset in range(packet.credit_units)
                )
                if len(set(slot_indices)) != packet.credit_units:
                    raise ValueError("packet requires more credits than the declared pool")
                ready_at_ps = max(
                    link_cursors.get((*pair, link_index), 0),
                    *(slots[index] for index in slot_indices),
                )
                link_ready.append((ready_at_ps, slot_indices))
            link_index = min(
                range(len(link_ready)),
                key=lambda candidate: (link_ready[candidate][0], candidate),
            )
            ready_at_ps, slot_indices = link_ready[link_index]
            credit_key = (*pair, link_index, packet.virtual_channel)
            credit_visits[credit_key] = (
                credit_visits.get(credit_key, 0) + packet.credit_units
            )
            started_at_ps = max(
                packet.released_at_ps,
                ready_at_ps,
                endpoint_cursors.get(packet.source, 0),
            )
            replay_count = replay_counts.get(packet.packet_id, 0)
            total_link_bytes = packet.wire_bytes * (1 + replay_count)
            link_duration_ps = _serialize_ps(
                total_link_bytes,
                self.profile.tx.per_link_rate_bytes_per_second,
            ) + replay_count * self.options.replay_timeout_ps
            base_link_duration_ps = _serialize_ps(
                packet.wire_bytes,
                self.profile.tx.per_link_rate_bytes_per_second,
            )
            endpoint_duration_ps = _serialize_ps(
                total_link_bytes,
                self.profile.tx.endpoint_egress_rate_bytes_per_second,
            )
            finished_at_ps = started_at_ps + link_duration_ps
            acknowledged_at_ps = (
                finished_at_ps + self.options.acknowledgement_latency_ps
            )
            link_cursors[(*pair, link_index)] = finished_at_ps
            endpoint_cursors[packet.source] = started_at_ps + endpoint_duration_ps
            packet_credit_available = credit_release_by_packet.get(packet.packet_id, 0)
            slots = credit_slots[credit_key]
            for slot_index in slot_indices:
                slots[slot_index] = packet_credit_available
            scheduled.append(
                replace(
                    packet,
                    link_index=link_index,
                    tx_started_at_ps=started_at_ps,
                    tx_finished_at_ps=finished_at_ps,
                    acknowledged_at_ps=acknowledged_at_ps,
                    replay_buffer_released_at_ps=acknowledged_at_ps,
                    replay_count=replay_count,
                    replay_wire_bytes=replay_count * packet.wire_bytes,
                    replay_time_ps=link_duration_ps - base_link_duration_ps,
                )
            )
        return tuple(scheduled)

    @staticmethod
    def _timing_signature(
        packets: tuple[NvlinkFlitPacket, ...],
    ) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (
                packet.packet_id,
                packet.link_index,
                packet.tx_started_at_ps,
                packet.tx_finished_at_ps,
                packet.switch_started_at_ps,
                packet.switch_finished_at_ps,
                packet.rx_buffer_accepted_at_ps,
                packet.rx_started_at_ps,
                packet.rx_finished_at_ps,
                packet.rx_buffer_released_at_ps,
                packet.credit_available_at_ps,
                packet.visible_at_ps,
            )
            for packet in packets
        )

    def _result(
        self,
        *,
        transfers: Sequence[NvlinkTransfer],
        delivered: tuple[NvlinkFlitPacket, ...],
        credit_releases: tuple[NvlinkCreditRelease, ...],
        switch_grants: tuple[NvlinkSwitchGrant, ...],
        visibility_events: tuple[NvlinkVisibilityEvent, ...],
        max_occupancy: int,
        fixed_point_iterations: int,
    ) -> NvlinkAlignedDomainResult:
        request_packets = tuple(
            packet
            for packet in delivered
            if packet.direction is NvlinkPacketDirection.REQUEST
        )
        response_packets = tuple(
            packet
            for packet in delivered
            if packet.direction is NvlinkPacketDirection.RESPONSE
        )
        request_wire_bytes = sum(packet.wire_bytes for packet in request_packets)
        response_wire_bytes = sum(packet.wire_bytes for packet in response_packets)
        replay_wire_bytes = sum(packet.replay_wire_bytes for packet in delivered)
        return NvlinkAlignedDomainResult(
            implementation=NVLINK_ALIGNED_PROFILE_IMPLEMENTATION,
            profile_id=self.profile.profile_id,
            authority=NvlinkMechanismAuthority.ALIGNED,
            packets=delivered,
            credit_releases=credit_releases,
            switch_grants=switch_grants,
            visibility_events=visibility_events,
            logical_bytes=sum(transfer.payload_bytes for transfer in transfers),
            request_payload_bytes=sum(packet.payload_bytes for packet in request_packets),
            response_payload_bytes=sum(packet.payload_bytes for packet in response_packets),
            request_wire_bytes=request_wire_bytes,
            response_wire_bytes=response_wire_bytes,
            replay_wire_bytes=replay_wire_bytes,
            total_wire_bytes=request_wire_bytes + response_wire_bytes + replay_wire_bytes,
            acknowledgement_count=len(delivered),
            replayed_packet_count=sum(packet.replay_count > 0 for packet in delivered),
            replay_time_ps=sum(packet.replay_time_ps for packet in delivered),
            completion_time_ps=max(
                (packet.visible_at_ps or 0 for packet in delivered),
                default=0,
            ),
            max_rx_buffer_occupancy_bytes=max_occupancy,
            random_draw_count=sum(packet.random_draw_count for packet in delivered),
            fixed_point_iterations=fixed_point_iterations,
        )


class NvlinkDomainService(Generic[_AnalyticResult]):
    """Compose TX, switch, and RX, or preserve the analytical bypass exactly.

    ``serve`` retains the merged pre-TRAF-73 flow-policy behavior so preserved
    studies keep their exact bytes under the explicit compatibility authority.
    The aligned authority deepens the same three modules with flits, link
    reliability, receiver-owned credits, ordering and crossbar state.
    ``serve_arbitrated`` remains the compatibility contention entry point.
    """

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
        flow_policy: NvlinkFlowPolicy = LEGACY_NVLINK_FLOW_POLICY,
        authority: NvlinkMechanismAuthority = NvlinkMechanismAuthority.COMPATIBILITY,
        aligned_options: NvlinkAlignedOptions | None = None,
    ) -> _AnalyticResult | NvlinkDomainResult | NvlinkAlignedDomainResult:
        """Return the exact bypass object or the selected sole domain authority."""

        if self.profile is None:
            return analytic_result
        if type(include_switch) is not bool:
            raise TypeError("include_switch must be a boolean")
        _require_enum("flow_policy", flow_policy, NvlinkFlowPolicy)
        _require_enum("authority", authority, NvlinkMechanismAuthority)
        if authority is NvlinkMechanismAuthority.ALIGNED:
            if flow_policy is not LEGACY_NVLINK_FLOW_POLICY:
                raise ValueError(
                    "aligned authority does not accept a compatibility flow policy"
                )
            options = aligned_options or NvlinkAlignedOptions()
            return _NvlinkAlignedEngine(self.profile, options).serve(
                transfers,
                include_switch=include_switch,
            )
        if aligned_options is not None:
            raise ValueError("aligned_options require the aligned authority")
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
        return self._result(transfers, delivered, max_occupancy)

    def serve_aligned(
        self,
        transfers: Sequence[NvlinkTransfer],
        *,
        analytic_result: _AnalyticResult,
        include_switch: bool = True,
        options: NvlinkAlignedOptions | None = None,
    ) -> _AnalyticResult | NvlinkAlignedDomainResult:
        """Select the aligned authority without widening compatibility callers."""

        result = self.serve(
            transfers,
            analytic_result=analytic_result,
            include_switch=include_switch,
            authority=NvlinkMechanismAuthority.ALIGNED,
            aligned_options=options,
        )
        if isinstance(result, NvlinkDomainResult):
            raise TypeError("aligned authority returned a compatibility result")
        return result

    def serve_arbitrated(
        self,
        transfers: Sequence[NvlinkTransfer],
        *,
        analytic_result: _AnalyticResult,
        include_switch: bool = True,
        policy: NvlinkArbitrationPolicy = DEFAULT_NVLINK_ARBITRATION_POLICY,
    ) -> _AnalyticResult | NvlinkDomainResult:
        """Serve independently credited links through a selected RX arbiter."""

        if self.profile is None:
            return analytic_result
        if type(include_switch) is not bool:
            raise TypeError("include_switch must be a boolean")
        _require_enum("policy", policy, NvlinkArbitrationPolicy)
        tx = NvlinkTx(self.profile.tx)
        packetized = tuple(
            packet
            for transfer in transfers
            for packet in tx.packetize(transfer)
        )
        transmitted = tx.transmit(
            packetized,
            credit_return_latency_ps=self.profile.rx.credit_return_latency_ps,
        )
        if include_switch:
            forwarded = NvlinkSwitch(self.profile.switch).forward(transmitted)
        else:
            forwarded = transmitted
        delivered, max_occupancy = NvlinkRx(self.profile.rx).receive_arbitrated(
            forwarded,
            policy=policy,
        )
        return self._result(transfers, delivered, max_occupancy)

    def _result(
        self,
        transfers: Sequence[NvlinkTransfer],
        delivered: tuple[NvlinkPacket, ...],
        max_occupancy: int,
    ) -> NvlinkDomainResult:
        if self.profile is None:
            raise AssertionError("an NVLink domain result requires a profile")
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
        flow_policy=LEGACY_NVLINK_FLOW_POLICY,
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
        flow_policy=LEGACY_NVLINK_FLOW_POLICY,
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


def _packet_arrival_ps(packet: NvlinkPacket) -> int:
    arrival = (
        packet.switch_finished_at_ps
        if packet.switch_finished_at_ps is not None
        else packet.tx_finished_at_ps
    )
    if arrival is None:
        raise ValueError("RX input packet has no upstream completion")
    return arrival


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


def _require_endpoint(name: str, value: object, endpoint_count: int = 4) -> None:
    _require_nonnegative_int(name, value)
    if int(value) >= endpoint_count:
        raise ValueError(
            f"{name} must identify one of {endpoint_count} declared endpoints"
        )
