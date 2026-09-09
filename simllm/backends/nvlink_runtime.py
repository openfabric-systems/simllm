"""Causal packet service behind the existing NVLink domain interface.

The calendar owns every grant, in-flight buffer reservation, credit return and
consumer visibility event. A direct mesh has one credited hop. A queued route
has a switch-owned input hop and a separately reserved destination buffer.
Buffer scopes and zero additional read-target processing remain declared model
choices; this engine does not identify a deployed GPU's hidden parameters.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from heapq import heappop, heappush

from simllm.placement.peer_topology import PeerFabric, ResolvedPeerPath

from .htsim_nvlink import (
    NVLINK_ALIGNED_PROFILE_IMPLEMENTATION,
    NVLINK_PHYSICAL_PROFILE_IMPLEMENTATION,
    NvlinkAlignedDomainResult,
    NvlinkAlignedOptions,
    NvlinkBufferVisit,
    NvlinkCandidateProfile,
    NvlinkCreditRelease,
    NvlinkFlitPacket,
    NvlinkFlowControlConfig,
    NvlinkIdentitySwitchPolicy,
    NvlinkMechanismAuthority,
    NvlinkOperation,
    NvlinkPacketDirection,
    NvlinkRoundRobinSwitchPolicy,
    NvlinkSwitchArbitration,
    NvlinkSwitchGrant,
    NvlinkSwitchMode,
    NvlinkTransfer,
    NvlinkTx,
    NvlinkVisibilityEvent,
    _NvlinkVoqHead,
    _serialize_ps,
)

CreditKey = tuple[object, ...]
VoqKey = tuple[int, str, int]


@dataclass(frozen=True)
class NvlinkPhysicalBinding:
    """Explicit attachment authority and additional control processing delays.

    Link serialization and propagation come only from the fabric. Source feed,
    receive service and crossbar rate come from the candidate profile. Processing
    delays below add to the physical reverse propagation of the corresponding
    hop; they do not replace it. No value is a hardware measurement.
    """

    fabric: PeerFabric
    credit_return_processing_ps: int = 0
    acknowledgement_processing_ps: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.fabric, PeerFabric) or self.fabric.protocol != "nvlink":
            raise ValueError("NVLink binding requires an explicit NVLink peer fabric")
        for name in ("credit_return_processing_ps", "acknowledgement_processing_ps"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class NvlinkPhysicalPacketPath:
    """Read-only physical attachment join for one packet attempt."""

    packet_id: str
    path: ResolvedPeerPath
    source_feed_finished_at_ps: int
    first_hop_arrived_at_ps: int
    input_hop_acknowledged_at_ps: int
    output_hop_acknowledged_at_ps: int | None


@dataclass(frozen=True)
class NvlinkBufferClaimObservation:
    """A reservation projection that also represents an unfinished visit."""

    packet_id: str
    buffer_id: str
    capacity_bytes: int
    wire_bytes: int
    reserved_at_ps: int
    arrived_at_ps: int | None
    released_at_ps: int | None
    credit_available_at_ps: int | None
    returned: bool


@dataclass(slots=True)
class _Buffer:
    capacity: int
    available: int
    resident: int = 0


@dataclass(slots=True)
class _Claim:
    packet_index: int
    buffer_id: str
    reserved_at_ps: int
    credit_key: CreditKey | None
    arrived_at_ps: int | None = None
    released_at_ps: int | None = None
    credit_available_at_ps: int | None = None
    returned: bool = False


class NvlinkCausalEngine:
    """One finite, monotonically advancing domain transaction.

    The batch interface preserves the existing component contract. The retained
    interface admits complete ready phases and returns at consumer visibility,
    while acknowledgements and credit returns stay on the same calendar.
    """

    def __init__(
        self, profile: NvlinkCandidateProfile, options: NvlinkAlignedOptions,
        *, on_packet_event: Callable[[str, NvlinkFlitPacket, int], None] | None = None,
        physical: NvlinkPhysicalBinding | None = None,
    ) -> None:
        if not isinstance(profile, NvlinkCandidateProfile):
            raise TypeError("profile must be an NvlinkCandidateProfile")
        if not isinstance(options, NvlinkAlignedOptions):
            raise TypeError("options must be an NvlinkAlignedOptions")
        if on_packet_event is not None and not callable(on_packet_event):
            raise TypeError("packet observer must be callable")
        if physical is not None and not isinstance(physical, NvlinkPhysicalBinding):
            raise TypeError("physical must be an NvlinkPhysicalBinding")
        self.profile = profile
        self.options = options
        self.flow_control = options.flow_control or (
            NvlinkFlowControlConfig.from_candidate_profile(profile)
        )
        self.physical = physical
        self._routes = {} if physical is None else {
            (route.source_rank, route.destination_rank): physical.fabric.paths_between(
                route.source_rank, route.destination_rank
            ) for route in physical.fabric.routes
        }
        self._physical_ports = {} if physical is None else {
            port.port_id: index for index, port in enumerate(physical.fabric.ports)
        }
        self._packet_paths: dict[str, ResolvedPeerPath] = {}
        self._feed_finishes: dict[str, int] = {}
        if physical is not None:
            if options.replay_counts or options.replay_timeout_ps:
                raise ValueError("physical retained service does not select injected replays")
            if (self.flow_control.return_transport_latency_ps or profile.rx.credit_return_latency_ps
                    or options.acknowledgement_latency_ps):
                raise ValueError("physical control delays must use the explicit processing fields")
            if physical.fabric.switched != (profile.switch.mode is NvlinkSwitchMode.QUEUED):
                raise ValueError("physical route and switch service authorities disagree")
            if physical.fabric.switched and (
                profile.switch.buffer_capacity_bytes != physical.fabric.switch_input_buffer_bytes
            ):
                raise ValueError("physical input buffer and candidate capacity declarations disagree")
        self._used = False
        self._streaming = False
        self._observer = on_packet_event
        self._transfers: list[NvlinkTransfer] = []
        self._extent_packets: dict[str, tuple[int, ...]] = {}
        self._transport_retired: set[int] = set()
        self._now = 0
        self._last_physical_event_ps = 0
        self._event_order = 0
        self._events: list[tuple[int, int, str, int]] = []
        self._event_count = 0
        self._packets: list[NvlinkFlitPacket] = []
        self._tx_queues: list[deque[int]] = []
        self._read_requests: dict[str, int] = {}
        self._domains: dict[tuple[int, str], deque[int]] = {}
        self._rx_done: set[int] = set()
        self._visible: set[int] = set()
        self._acknowledged: set[int] = set()
        self._input_acknowledged: set[int] = set()
        self._output_acknowledged: set[int] = set()
        self._links: dict[tuple[object, ...], int] = {}
        self._endpoints: dict[int, int] = {}
        self._credits: dict[CreditKey, int] = {}
        self._buffers: dict[str, _Buffer] = {}
        self._claims: list[_Claim] = []
        self._first_hop_claims: dict[int, int] = {}
        self._rx_claims: dict[int, int] = {}
        self._voq: dict[VoqKey, list[tuple[int, int]]] = {}
        self._switch_inputs: dict[int, int] = {}
        self._switch_outputs: dict[int, int] = {}
        self._rx_queues: dict[int, list[tuple[int, int]]] = {}
        self._rx_cursors: dict[int, int] = {}
        self._credit_releases: list[NvlinkCreditRelease] = []
        self._switch_grants: list[NvlinkSwitchGrant] = []
        self._visibility: list[NvlinkVisibilityEvent] = []
        self._max_rx_occupancy = 0
        self._replays = dict(options.replay_counts)
        self._queued_switch = False
        self._switch_policy = (
            NvlinkIdentitySwitchPolicy()
            if options.switch_arbitration is NvlinkSwitchArbitration.IDENTITY
            else NvlinkRoundRobinSwitchPolicy()
        )

    def serve(
        self, transfers: Sequence[NvlinkTransfer], *, include_switch: bool
    ) -> NvlinkAlignedDomainResult:
        if self._used:
            raise RuntimeError("NVLink event engine cannot serve a second transaction")
        if self.physical is not None:
            raise ValueError("physical service uses admit, advance and drain on one retained session")
        if type(include_switch) is not bool:
            raise TypeError("include_switch must be a boolean")
        transfers = tuple(transfers)
        self._queued_switch = include_switch and self.profile.switch.mode is NvlinkSwitchMode.QUEUED
        self._prepare(transfers)
        self._used = True
        for release in sorted({p.released_at_ps for p in self._packets}):
            self._schedule(release, "wake", -1)
        self._drive()
        self._check_drain()
        return self._result(transfers)

    @property
    def now_ps(self) -> int:
        return self._now

    @property
    def has_pending_physical_work(self) -> bool:
        return bool(self._events)

    @property
    def packets(self) -> tuple[NvlinkFlitPacket, ...]:
        """Current immutable packet records, including pending scheduled fields."""

        return tuple(self._packets)

    @property
    def physical_paths(self) -> tuple[NvlinkPhysicalPacketPath, ...]:
        return tuple(
            NvlinkPhysicalPacketPath(
                packet.packet_id, self._packet_paths[packet.packet_id],
                self._feed_finishes[packet.packet_id],
                packet.tx_finished_at_ps
                + self._packet_paths[packet.packet_id].input_link.propagation_delay_ps,
                packet.tx_finished_at_ps
                + 2 * self._packet_paths[packet.packet_id].input_link.propagation_delay_ps
                + self.physical.acknowledgement_processing_ps,
                (None if packet.switch_finished_at_ps is None else
                 packet.switch_finished_at_ps
                 + 2 * self._packet_paths[packet.packet_id].output_link.propagation_delay_ps
                 + self.physical.acknowledgement_processing_ps),
            ) for packet in self._packets if packet.packet_id in self._packet_paths
        )

    @property
    def buffer_ownership(self) -> tuple[tuple[str, int, int, int], ...]:
        """Capacity, sender-visible availability and resident bytes at this instant."""
        return tuple((name, pool.capacity, pool.available, pool.resident)
                     for name, pool in sorted(self._buffers.items()))

    @property
    def buffer_claims(self) -> tuple[NvlinkBufferClaimObservation, ...]:
        """All reservations, including live credit ownership after visibility."""
        return tuple(NvlinkBufferClaimObservation(
            self._packets[claim.packet_index].packet_id, claim.buffer_id,
            self._buffers[claim.buffer_id].capacity,
            self._packets[claim.packet_index].wire_bytes,
            claim.reserved_at_ps, claim.arrived_at_ps, claim.released_at_ps,
            claim.credit_available_at_ps, claim.returned,
        ) for claim in self._claims)

    def _preflight(self, transfers: tuple[NvlinkTransfer, ...], include_switch: bool) -> NvlinkCausalEngine:
        """Build an isolated packetization only after checking the retained state."""

        if self._used and not self._streaming:
            raise RuntimeError("a completed batch engine cannot become a retained session")
        if type(include_switch) is not bool:
            raise TypeError("include_switch must be a boolean")
        if not transfers or any(not isinstance(t, NvlinkTransfer) for t in transfers):
            raise ValueError("a retained phase needs transfer records")
        if any(t.operation is not NvlinkOperation.PEER_WRITE for t in transfers):
            raise ValueError("retained local phases currently require peer writes")
        if self.options.replay_counts:
            raise ValueError("retained local phases do not select injected component replays")
        queued = include_switch and self.profile.switch.mode is NvlinkSwitchMode.QUEUED
        if self.physical is not None and queued != self.physical.fabric.switched:
            raise ValueError("physical admission cannot bypass its declared switch")
        if self._streaming and queued != self._queued_switch:
            raise ValueError("a retained domain cannot change its switch authority")
        ids = [transfer.extent_id for transfer in transfers]
        if len(set(ids)) != len(ids) or any(name in self._extent_packets for name in ids):
            raise ValueError("retained extent identities must be unique for the session")
        if any(transfer.released_at_ps < self._now for transfer in transfers):
            raise ValueError("retained phase eligibility precedes the current calendar")
        if self.physical is not None:
            self._check_physical_horizon(transfers)

        # This temporary packetizer has no callbacks or calendar execution. A
        # malformed later transfer cannot partially commit an earlier one.
        probe = NvlinkCausalEngine(self.profile, self.options, physical=self.physical)
        probe._queued_switch = queued
        probe._prepare(transfers)
        previous_sequences: dict[tuple[int, str], int] = {}
        for packet in self._packets:
            key = (packet.destination, packet.ordering_domain)
            previous_sequences[key] = max(previous_sequences.get(key, -1), packet.sequence)
        for key, queue in probe._domains.items():
            if probe._packets[queue[0]].sequence <= previous_sequences.get(key, -1):
                raise ValueError("retained visibility sequence reuses a prior domain position")
        return probe

    def preview(self, transfers: Sequence[NvlinkTransfer], *, include_switch: bool) -> tuple[NvlinkFlitPacket, ...]:
        """Validate a complete phase without tokens, callbacks or calendar mutation."""
        return self._preflight(tuple(transfers), include_switch).packets

    def admit(self, transfers: Sequence[NvlinkTransfer], *, include_switch: bool) -> None:
        """Preflight the complete phase, then append it without advancing time."""
        transfers = tuple(transfers)
        probe = self._preflight(transfers, include_switch)
        ids = [transfer.extent_id for transfer in transfers]

        offset = len(self._packets)
        self._packets.extend(probe._packets)
        self._transfers.extend(transfers)
        self._tx_queues.extend(deque(offset + i for i in queue) for queue in probe._tx_queues)
        for key, queue in probe._domains.items():
            self._domains.setdefault(key, deque()).extend(offset + i for i in queue)
        for extent_id in ids:
            self._extent_packets[extent_id] = tuple(
                offset + i for i, packet in enumerate(probe._packets) if packet.extent_id == extent_id
            )
        self._queued_switch = probe._queued_switch
        self._used = self._streaming = True
        for release in sorted({packet.released_at_ps for packet in probe._packets}):
            self._schedule(release, "wake", -1)

    def advance_until_visible(self, extent_ids: Sequence[str]) -> int:
        """Reach the requested logical boundary without draining later work."""

        ids = tuple(extent_ids)
        if not ids or len(set(ids)) != len(ids) or any(name not in self._extent_packets for name in ids):
            raise ValueError("visibility boundary requires unique admitted extent identities")
        targets = {i for name in ids for i in self._extent_packets[name]}
        if not targets <= self._visible:
            self._drive(targets)
        if not targets <= self._visible:
            raise RuntimeError("retained NVLink session stalled before consumer visibility")
        return max(self._packets[i].visible_at_ps or 0 for i in targets)

    def advance_to(self, at_ps: int) -> None:
        """Let existing physical tails progress while another graph resource runs."""

        if type(at_ps) is not int or at_ps < self._now:
            raise ValueError("retained calendar cannot move backwards")
        if self.physical is not None and at_ps >= 2**64:
            raise ValueError("physical packet clock must fit unsigned 64-bit picoseconds")
        self._drive(until_ps=at_ps)
        self._now = at_ps

    def drain(self) -> NvlinkAlignedDomainResult:
        if not self._streaming:
            raise RuntimeError("drain requires an admitted retained session")
        self._drive()
        self._check_drain()
        return self._result(tuple(self._transfers))

    def _drive(self, visible_targets: set[int] | None = None, *, until_ps: int | None = None) -> None:
        while self._events:
            if until_ps is not None and self._events[0][0] > until_ps:
                break
            self._now = self._events[0][0]
            while True:
                while self._events and self._events[0][0] == self._now:
                    _, _, kind, index = heappop(self._events)
                    self._event_count += 1
                    self._handle(kind, index)
                changed = self._grant_rx()
                changed = self._grant_switch() or changed
                changed = self._grant_tx() or changed
                if not changed and not (self._events and self._events[0][0] == self._now):
                    break
            if visible_targets is not None and visible_targets <= self._visible:
                break

    def _observe(self, kind: str, index: int) -> None:
        if self._observer is not None:
            self._observer(kind, self._packets[index], self._now)

    def _retire_if_complete(self, index: int) -> None:
        if index in self._visible and index in self._acknowledged and index not in self._transport_retired:
            self._transport_retired.add(index)
            self._observe("delivered", index)

    def _prepare(self, transfers: tuple[NvlinkTransfer, ...]) -> None:
        tx = NvlinkTx(self.profile.tx)
        for transfer in transfers:
            packets = tx.packetize_flits(
                transfer,
                packet_format=self.options.packet_format,
                flow_control=self.flow_control,
            )
            first = len(self._packets)
            self._packets.extend(packets)
            self._tx_queues.append(deque(range(first, len(self._packets))))
            if transfer.operation is NvlinkOperation.PEER_READ:
                self._read_requests[transfer.extent_id] = first
        if not self._packets:
            raise ValueError("aligned NVLink service requires at least one packet")
        ids = {p.packet_id for p in self._packets}
        if len(ids) != len(self._packets):
            raise ValueError("aligned NVLink packet identities must be globally unique")
        if not self._replays.keys() <= ids:
            raise ValueError("NVLink replay selects an unknown packet identity")
        domains: dict[tuple[int, str], list[int]] = {}
        for index, packet in enumerate(self._packets):
            if self.physical is not None and (packet.source, packet.destination) not in self._routes:
                raise ValueError("packet has no physical attachment route")
            if packet.credit_units > self.flow_control.credits_per_pool:
                raise ValueError("packet requires more credits than the declared pool")
            if packet.wire_bytes > self.profile.rx.buffer_capacity_bytes:
                raise ValueError("packet exceeds declared NVLink RX buffer")
            if self._queued_switch and packet.wire_bytes > int(
                self.profile.switch.buffer_capacity_bytes or 0
            ):
                raise ValueError("packet exceeds declared NVLink switch VOQ capacity")
            key = (packet.destination, packet.ordering_domain)
            domains.setdefault(key, []).append(index)
        for key, indices in domains.items():
            ordered = sorted(indices, key=lambda index: self._packets[index].sequence)
            sequences = [self._packets[index].sequence for index in ordered]
            if len(set(sequences)) != len(sequences):
                raise ValueError("NVLink visibility sequence is not strictly increasing per domain")
            self._domains[key] = deque(ordered)

    def _check_physical_horizon(self, transfers: tuple[NvlinkTransfer, ...]) -> None:
        """Reject overflow using a conservative fully serialized completion bound."""
        assert self.physical is not None
        maximum = 2**64 - 1
        if any(t.released_at_ps > maximum or t.payload_bytes > maximum for t in transfers):
            raise ValueError("physical input exceeds the unsigned 64-bit clock or byte contract")

        def packet_bound(source, destination, wire):
            paths = self._routes.get((source, destination))
            if not paths:
                raise ValueError("packet has no physical attachment route")
            bounds = []
            for path in paths:
                first = max(self._physical_serialize(wire, path.input_link.link_rate_bps),
                            _serialize_ps(wire, self.profile.tx.endpoint_egress_rate_bytes_per_second))
                receive = _serialize_ps(wire, self.profile.rx.ingress_rate_bytes_per_second)
                bound = first + receive + 3 * path.input_link.propagation_delay_ps
                bound += self.physical.credit_return_processing_ps + self.physical.acknowledgement_processing_ps
                if path.output_link is not None:
                    bound += max(self._physical_serialize(wire, path.output_link.link_rate_bps),
                                 _serialize_ps(wire, self.profile.switch.service_rate_bytes_per_second))
                    bound += 3 * path.output_link.propagation_delay_ps
                    bound += self.physical.credit_return_processing_ps + self.physical.acknowledgement_processing_ps
                bounds.append(bound)
            return max(bounds)

        horizon = max((self._now, *(event[0] for event in self._events),
                       *(transfer.released_at_ps for transfer in transfers)))
        for index, packet in enumerate(self._packets):
            if index not in self._transport_retired:
                horizon += packet_bound(packet.source, packet.destination, packet.wire_bytes)
        fmt = self.options.packet_format
        payload_limit = min(self.profile.tx.max_payload_bytes, fmt.maximum_payload_flits * 16)
        for transfer in transfers:
            count = (transfer.payload_bytes + payload_limit - 1) // payload_limit
            wire = 16 * (fmt.header_flits + transfer.address_extension_flits + transfer.byte_enable_flits
                         + (payload_limit + 15) // 16)
            horizon += count * packet_bound(transfer.source, transfer.destination, wire)
            if transfer.offered_rate_bytes_per_second is not None:
                horizon += _serialize_ps(count * wire, transfer.offered_rate_bytes_per_second)
        if horizon > maximum:
            raise ValueError("physical completion horizon exceeds unsigned 64-bit picoseconds")

    def _schedule(self, at_ps: int, kind: str, index: int) -> None:
        if at_ps < self._now:
            raise AssertionError("NVLink event calendar attempted to move backwards")
        self._event_order += 1
        heappush(self._events, (at_ps, self._event_order, kind, index))

    def _update(self, index: int, **fields) -> NvlinkFlitPacket:
        packet = replace(self._packets[index], **fields)
        self._packets[index] = packet
        return packet

    def _rx_buffer_id(self, packet: NvlinkFlitPacket) -> str:
        if self.physical is not None:
            return f"rx:{packet.destination}"
        return f"rx:{packet.destination}:{packet.virtual_channel}"

    def _switch_buffer_id(self, packet: NvlinkFlitPacket, link: int | None = None) -> str:
        if self.physical is not None:
            path = (self._packet_paths[packet.packet_id] if link is None
                    else self._routes[(packet.source, packet.destination)][link])
            return f"switch-input:{path.switch_input_port_id}"
        return f"switch:{packet.source}:{packet.destination}:{packet.virtual_channel}"

    def _link_key(self, packet: NvlinkFlitPacket, link: int) -> tuple[object, ...]:
        if self.physical is not None:
            return self._routes[(packet.source, packet.destination)][link].input_resource
        return packet.source, packet.destination, link

    def _credit_key(self, packet: NvlinkFlitPacket, link: int) -> CreditKey:
        return (*self._link_key(packet, link), packet.virtual_channel)

    def _switch_ports(self, packet: NvlinkFlitPacket) -> tuple[int, int]:
        if self.physical is None:
            return packet.source, packet.destination
        path = self._packet_paths[packet.packet_id]
        return (self._physical_ports[path.switch_input_port_id],
                self._physical_ports[path.switch_output_port_id])

    def _voq_key(self, packet: NvlinkFlitPacket) -> VoqKey:
        source, destination = self._switch_ports(packet)
        return source, packet.virtual_channel, destination

    @staticmethod
    def _physical_serialize(wire_bytes: int, rate_bps: int) -> int:
        return (wire_bytes * 8 * 10**12 + rate_bps - 1) // rate_bps

    def _buffer(self, buffer_id: str, capacity: int) -> _Buffer:
        pool = self._buffers.setdefault(buffer_id, _Buffer(capacity, capacity))
        if pool.capacity != capacity:
            raise AssertionError("NVLink buffer identity changed capacity")
        return pool

    def _reserve(
        self,
        index: int,
        buffer_id: str,
        capacity: int,
        credit_key: CreditKey | None,
    ) -> int:
        packet = self._packets[index]
        pool = self._buffer(buffer_id, capacity)
        if pool.available < packet.wire_bytes:
            raise AssertionError("NVLink granted a packet without downstream capacity")
        pool.available -= packet.wire_bytes
        claim_index = len(self._claims)
        self._claims.append(_Claim(index, buffer_id, self._now, credit_key))
        return claim_index

    def _arrive(self, claim_index: int) -> None:
        claim = self._claims[claim_index]
        if claim.arrived_at_ps is not None or claim.reserved_at_ps > self._now:
            raise AssertionError("NVLink arrival has no unique earlier reservation")
        claim.arrived_at_ps = self._now
        pool = self._buffers[claim.buffer_id]
        pool.resident += self._packets[claim.packet_index].wire_bytes
        if pool.resident > pool.capacity:
            raise AssertionError("NVLink resident bytes exceed buffer capacity")
        if claim.buffer_id.startswith("rx:"):
            self._max_rx_occupancy = max(self._max_rx_occupancy, pool.resident)

    def _release(self, claim_index: int) -> None:
        claim = self._claims[claim_index]
        if claim.arrived_at_ps is None or claim.released_at_ps is not None:
            raise AssertionError("NVLink buffer release has no unique resident owner")
        packet = self._packets[claim.packet_index]
        self._buffers[claim.buffer_id].resident -= packet.wire_bytes
        claim.released_at_ps = self._now
        return_delay = self.flow_control.return_transport_latency_ps
        if self.physical is not None:
            path = self._packet_paths[packet.packet_id]
            link = path.input_link if claim.credit_key is not None else path.output_link
            assert link is not None
            return_delay = link.propagation_delay_ps + self.physical.credit_return_processing_ps
        claim.credit_available_at_ps = self._now + return_delay
        self._schedule(claim.credit_available_at_ps, "return", claim_index)
        if claim.credit_key is not None:
            assert packet.link_index is not None
            self._credit_releases.append(
                NvlinkCreditRelease(
                    packet_id=packet.packet_id,
                    source=packet.source,
                    destination=packet.destination,
                    link_index=packet.link_index,
                    virtual_channel=packet.virtual_channel,
                    credit_units=packet.credit_units,
                    buffer_released_at_ps=self._now,
                    credit_available_at_ps=claim.credit_available_at_ps,
                    buffer_id=claim.buffer_id,
                )
            )
            self._update(
                claim.packet_index,
                credit_available_at_ps=claim.credit_available_at_ps,
            )

    def _return(self, claim_index: int) -> None:
        claim = self._claims[claim_index]
        if claim.returned or claim.credit_available_at_ps != self._now:
            raise AssertionError("NVLink credit return has no unique scheduled owner")
        packet = self._packets[claim.packet_index]
        pool = self._buffers[claim.buffer_id]
        pool.available += packet.wire_bytes
        if pool.available > pool.capacity:
            raise AssertionError("NVLink returned more buffer capacity than exists")
        if claim.credit_key is not None:
            key = claim.credit_key
            self._credits[key] += packet.credit_units
            if self._credits[key] > self.flow_control.credits_per_pool:
                raise AssertionError("NVLink returned more link credits than exist")
        claim.returned = True

    def _eligible_at(self, index: int) -> int | None:
        packet = self._packets[index]
        ready = packet.released_at_ps
        if packet.direction is NvlinkPacketDirection.RESPONSE:
            request_index = self._read_requests[packet.extent_id]
            if request_index not in self._visible:
                return None
            request_visible = self._packets[request_index].visible_at_ps
            if request_visible is None:
                raise AssertionError("NVLink read request has no visibility timestamp")
            ready = max(ready, request_visible)
        return ready

    def _tx_candidate(self, index: int) -> tuple[int, int, str, int] | None:
        packet = self._packets[index]
        ready = self._eligible_at(index)
        if ready is None or ready > self._now:
            return None
        if self._endpoints.get(packet.source, 0) > self._now:
            return None
        capacity = (
            int(self.profile.switch.buffer_capacity_bytes or 0)
            if self._queued_switch
            else self.profile.rx.buffer_capacity_bytes
        )
        link_count = (self.profile.tx.links_per_peer if self.physical is None else
                      len(self._routes[(packet.source, packet.destination)]))
        legal_links = []
        for link in range(link_count):
            buffer_id = (self._switch_buffer_id(packet, link) if self._queued_switch
                         else self._rx_buffer_id(packet))
            if self._buffer(buffer_id, capacity).available < packet.wire_bytes:
                continue
            credit_key = self._credit_key(packet, link)
            credits = self._credits.get(credit_key, self.flow_control.credits_per_pool)
            cursor = self._links.get(self._link_key(packet, link), 0)
            if cursor <= self._now and credits >= packet.credit_units:
                legal_links.append((cursor, link, buffer_id))
        if not legal_links:
            return None
        _, link, buffer_id = min(legal_links)
        return ready, link, buffer_id, capacity

    def _grant_tx(self) -> bool:
        changed = False
        # Queue order is the deterministic identity policy after legality checks.
        for queue in self._tx_queues:
            if not queue:
                continue
            index = queue[0]
            candidate = self._tx_candidate(index)
            if candidate is None:
                continue
            ready, link, buffer_id, capacity = candidate
            queue.popleft()
            packet = self._packets[index]
            key = self._credit_key(packet, link)
            if self.physical is not None:
                self._packet_paths[packet.packet_id] = self._routes[(packet.source, packet.destination)][link]
            self._credits[key] = (
                self._credits.get(key, self.flow_control.credits_per_pool) - packet.credit_units
            )
            claim = self._reserve(index, buffer_id, capacity, key)
            self._first_hop_claims[index] = claim
            if not self._queued_switch:
                self._rx_claims[index] = claim
            replay_count = self._replays.get(packet.packet_id, 0)
            link_service = _serialize_ps(
                packet.wire_bytes, self.profile.tx.per_link_rate_bytes_per_second
            )
            if self.physical is not None:
                path = self._packet_paths[packet.packet_id]
                link_service = self._physical_serialize(packet.wire_bytes, path.input_link.link_rate_bps)
            endpoint_service = _serialize_ps(
                packet.wire_bytes, self.profile.tx.endpoint_egress_rate_bytes_per_second
            )
            # Source feed and link service overlap, but the last byte cannot
            # leave before either service completes. Round each attempt once.
            base_duration = max(link_service, endpoint_service)
            duration = (
                1 + replay_count
            ) * base_duration + replay_count * self.options.replay_timeout_ps
            finished = self._now + duration
            # The declared replay policy blocks both link and total egress
            # through the retry train. Prepaying later retries at the first
            # grant would allow competing traffic to overrun endpoint capacity.
            endpoint_finished = self._now + (duration if replay_count else endpoint_service)
            acknowledgement = finished + self.options.acknowledgement_latency_ps
            if self.physical is not None:
                acknowledgement = (finished + 2 * path.input_link.propagation_delay_ps
                                   + self.physical.acknowledgement_processing_ps)
                self._feed_finishes[packet.packet_id] = endpoint_finished
            self._links[self._link_key(packet, link)] = finished
            self._endpoints[packet.source] = endpoint_finished
            self._update(
                index,
                link_index=link,
                tx_eligible_at_ps=ready,
                tx_started_at_ps=self._now,
                tx_finished_at_ps=finished,
                acknowledged_at_ps=acknowledgement,
                replay_buffer_released_at_ps=acknowledgement,
                replay_count=replay_count,
                replay_wire_bytes=replay_count * packet.wire_bytes,
                replay_time_ps=duration - base_duration,
            )
            self._observe("packet_tx_started", index)
            self._schedule(endpoint_finished, "wake", -1)
            self._schedule(finished, "tx_finish", index)
            self._schedule(acknowledgement, "ack", index)
            changed = True
        return changed

    def _grant_switch(self) -> bool:
        if not self._queued_switch:
            return False
        candidates = []
        for queue in self._voq.values():
            if not queue:
                continue
            _, index = queue[0]
            packet = self._packets[index]
            input_port, output_port = self._switch_ports(packet)
            if self._switch_inputs.get(input_port, 0) > self._now:
                continue
            if self._switch_outputs.get(output_port, 0) > self._now:
                continue
            if self.physical is not None:
                output_resource = self._packet_paths[packet.packet_id].output_resource
                if self._links.get(output_resource, 0) > self._now:
                    continue
            pool = self._buffer(self._rx_buffer_id(packet), self.profile.rx.buffer_capacity_bytes)
            if pool.available < packet.wire_bytes:
                continue
            # The policy sees actual port conflicts. Semantic ranks stay intact
            # on the authoritative packet; this temporary view owns no timing.
            policy_packet = packet if self.physical is None else replace(
                packet, source=input_port, destination=output_port
            )
            candidates.append(_NvlinkVoqHead(original_index=index, packet=policy_packet,
                                             shared_capacity_id=self._rx_buffer_id(packet) if self.physical is not None else None))
        selected = self._switch_policy.select(candidates, shared_capacities={
            name: pool.available for name, pool in self._buffers.items()
        } if self.physical is not None else None)
        for candidate in selected:
            index = candidate.original_index
            packet = self._packets[index]
            key = self._voq_key(packet)
            _, head = heappop(self._voq[key])
            if head != index:
                raise AssertionError("NVLink switch selected a non-head packet")
            self._rx_claims[index] = self._reserve(
                index,
                self._rx_buffer_id(packet),
                self.profile.rx.buffer_capacity_bytes,
                None,
            )
            duration = _serialize_ps(
                packet.wire_bytes, int(self.profile.switch.service_rate_bytes_per_second or 0)
            )
            if self.physical is not None:
                path = self._packet_paths[packet.packet_id]
                assert path.output_link is not None and path.output_resource is not None
                duration = max(duration, self._physical_serialize(packet.wire_bytes, path.output_link.link_rate_bps))
                self._links[path.output_resource] = self._now + duration
            finished = self._now + duration
            input_port, output_port = self._switch_ports(packet)
            self._switch_inputs[input_port] = finished
            self._switch_outputs[output_port] = finished
            self._update(
                index,
                input_port=input_port,
                output_port=output_port,
                switch_started_at_ps=self._now,
                switch_finished_at_ps=finished,
            )
            if self.physical is not None:
                acknowledgement = finished + 2 * path.output_link.propagation_delay_ps + self.physical.acknowledgement_processing_ps
                self._update(index, acknowledged_at_ps=max(packet.acknowledged_at_ps, acknowledgement),
                             replay_buffer_released_at_ps=max(packet.replay_buffer_released_at_ps, acknowledgement))
                self._schedule(acknowledgement, "ack_output", index)
            self._switch_grants.append(
                NvlinkSwitchGrant(
                    packet_id=packet.packet_id,
                    input_port=input_port,
                    output_port=output_port,
                    virtual_channel=packet.virtual_channel,
                    started_at_ps=self._now,
                    finished_at_ps=finished,
                    policy=self._switch_policy.arbitration,
                )
            )
            self._schedule(finished, "switch_finish", index)
        return bool(selected)

    def _rx_arrival(self, index: int) -> None:
        self._arrive(self._rx_claims[index])
        packet = self._update(index, rx_buffer_accepted_at_ps=self._now)
        self._observe("packet_rx_arrived", index)
        queue = self._rx_queues.setdefault(packet.destination, [])
        heappush(queue, (self._now, index))

    def _grant_rx(self) -> bool:
        changed = False
        for destination, queue in self._rx_queues.items():
            if not queue or self._rx_cursors.get(destination, 0) > self._now:
                continue
            _, index = heappop(queue)
            packet = self._packets[index]
            finished = self._now + _serialize_ps(
                packet.wire_bytes, self.profile.rx.ingress_rate_bytes_per_second
            )
            self._rx_cursors[destination] = finished
            self._update(index, rx_started_at_ps=self._now, rx_finished_at_ps=finished)
            self._schedule(finished, "rx_finish", index)
            changed = True
        return changed

    def _advance_visibility(self, index: int) -> None:
        packet = self._packets[index]
        queue = self._domains[(packet.destination, packet.ordering_domain)]
        while queue and queue[0] in self._rx_done:
            visible_index = queue.popleft()
            if visible_index in self._visible:
                raise AssertionError("NVLink packet became visible twice")
            visible_packet = self._update(visible_index, visible_at_ps=self._now)
            if visible_packet.rx_finished_at_ps is None:
                raise AssertionError("NVLink visible packet has no completed receive")
            self._visible.add(visible_index)
            self._visibility.append(
                NvlinkVisibilityEvent(
                    packet_id=visible_packet.packet_id,
                    ordering_domain=visible_packet.ordering_domain,
                    sequence=visible_packet.sequence,
                    rx_finished_at_ps=visible_packet.rx_finished_at_ps,
                    visible_at_ps=self._now,
                )
            )
            self._observe("consumer_visible", visible_index)
            self._retire_if_complete(visible_index)

    def _handle(self, kind: str, index: int) -> None:
        if kind == "wake":
            return
        self._last_physical_event_ps = self._now
        if kind == "return":
            self._return(index)
        elif kind in ("ack", "ack_output"):
            acknowledged = self._input_acknowledged if kind == "ack" else self._output_acknowledged
            if index in acknowledged:
                raise AssertionError("NVLink acknowledged a packet twice")
            acknowledged.add(index)
            if index in self._input_acknowledged and (
                self.physical is None or not self._queued_switch or index in self._output_acknowledged
            ):
                self._acknowledged.add(index)
                self._retire_if_complete(index)
        elif kind == "tx_finish":
            self._observe("packet_tx_finished", index)
            if self.physical is not None:
                path = self._packet_paths[self._packets[index].packet_id]
                self._schedule(self._now + path.input_link.propagation_delay_ps, "hop_arrive", index)
            else:
                self._first_hop_arrival(index)
        elif kind == "hop_arrive":
            self._first_hop_arrival(index)
        elif kind == "switch_finish":
            self._release(self._first_hop_claims[index])
            if self.physical is not None:
                path = self._packet_paths[self._packets[index].packet_id]
                assert path.output_link is not None
                self._schedule(self._now + path.output_link.propagation_delay_ps, "rx_arrive", index)
            else:
                self._rx_arrival(index)
        elif kind == "rx_arrive":
            self._rx_arrival(index)
        elif kind == "rx_finish":
            if index in self._rx_done:
                raise AssertionError("NVLink received a packet twice")
            self._release(self._rx_claims[index])
            self._update(index, rx_buffer_released_at_ps=self._now)
            self._rx_done.add(index)
            self._advance_visibility(index)
        else:
            raise AssertionError(f"unknown NVLink event kind {kind!r}")

    def _first_hop_arrival(self, index: int) -> None:
        if self._queued_switch:
            self._arrive(self._first_hop_claims[index])
            packet = self._packets[index]
            key = self._voq_key(packet)
            heappush(self._voq.setdefault(key, []), (self._now, index))
        else:
            self._rx_arrival(index)

    def _check_drain(self) -> None:
        count = len(self._packets)
        if len(self._visible) != count or len(self._acknowledged) != count:
            raise RuntimeError("NVLink domain stalled before every packet drained")
        if any(self._tx_queues) or any(self._voq.values()) or any(self._rx_queues.values()):
            raise AssertionError("NVLink completed with queued packets")
        if any(pool.resident or pool.available != pool.capacity for pool in self._buffers.values()):
            raise AssertionError("NVLink completed with outstanding buffer ownership")
        if any(value != self.flow_control.credits_per_pool for value in self._credits.values()):
            raise AssertionError("NVLink completed with outstanding link credits")
        if len(self._credit_releases) != count or not all(c.returned for c in self._claims):
            raise AssertionError("NVLink completed with missing credit returns")

    def _result(self, transfers: tuple[NvlinkTransfer, ...]) -> NvlinkAlignedDomainResult:
        packets = tuple(
            sorted(
                enumerate(self._packets),
                key=lambda item: (item[1].visible_at_ps, item[0], item[1].sequence),
            )
        )
        delivered = tuple(packet for _, packet in packets)
        requests = tuple(p for p in delivered if p.direction is NvlinkPacketDirection.REQUEST)
        responses = tuple(p for p in delivered if p.direction is NvlinkPacketDirection.RESPONSE)
        request_wire = sum(p.wire_bytes for p in requests)
        response_wire = sum(p.wire_bytes for p in responses)
        replay_wire = sum(p.replay_wire_bytes for p in delivered)
        visits = []
        for claim in self._claims:
            if (
                claim.arrived_at_ps is None
                or claim.released_at_ps is None
                or claim.credit_available_at_ps is None
            ):
                raise AssertionError("NVLink buffer claim has incomplete evidence")
            packet = self._packets[claim.packet_index]
            visits.append(
                NvlinkBufferVisit(
                    packet_id=packet.packet_id,
                    buffer_id=claim.buffer_id,
                    capacity_bytes=self._buffers[claim.buffer_id].capacity,
                    wire_bytes=packet.wire_bytes,
                    reserved_at_ps=claim.reserved_at_ps,
                    arrived_at_ps=claim.arrived_at_ps,
                    released_at_ps=claim.released_at_ps,
                    credit_available_at_ps=claim.credit_available_at_ps,
                )
            )
        return NvlinkAlignedDomainResult(
            implementation=(NVLINK_ALIGNED_PROFILE_IMPLEMENTATION if self.physical is None
                            else NVLINK_PHYSICAL_PROFILE_IMPLEMENTATION),
            profile_id=self.profile.profile_id,
            authority=(NvlinkMechanismAuthority.ALIGNED if self.physical is None
                       else NvlinkMechanismAuthority.PHYSICAL),
            packets=delivered,
            credit_releases=tuple(sorted(self._credit_releases, key=lambda r: r.packet_id)),
            switch_grants=tuple(
                sorted(self._switch_grants, key=lambda g: (g.started_at_ps, g.packet_id))
            ),
            visibility_events=tuple(
                sorted(self._visibility, key=lambda v: (v.visible_at_ps, v.packet_id))
            ),
            logical_bytes=sum(t.payload_bytes for t in transfers),
            request_payload_bytes=sum(p.payload_bytes for p in requests),
            response_payload_bytes=sum(p.payload_bytes for p in responses),
            request_wire_bytes=request_wire,
            response_wire_bytes=response_wire,
            replay_wire_bytes=replay_wire,
            total_wire_bytes=request_wire + response_wire + replay_wire,
            acknowledgement_count=len(self._acknowledged),
            replayed_packet_count=sum(p.replay_count > 0 for p in delivered),
            replay_time_ps=sum(p.replay_time_ps for p in delivered),
            completion_time_ps=max(p.visible_at_ps or 0 for p in delivered),
            max_rx_buffer_occupancy_bytes=self._max_rx_occupancy,
            random_draw_count=sum(p.random_draw_count for p in delivered),
            fixed_point_iterations=0,
            buffer_visits=tuple(visits),
            event_count=self._event_count,
            physical_drain_time_ps=self._now if self.physical is None else self._last_physical_event_ps,
        )
