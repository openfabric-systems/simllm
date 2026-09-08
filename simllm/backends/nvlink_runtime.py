"""Causal packet service behind the existing NVLink domain interface.

The calendar owns every grant, in-flight buffer reservation, credit return and
consumer visibility event. A direct mesh has one credited hop. A queued route
has a switch-owned input hop and a separately reserved destination buffer.
Buffer scopes and zero additional read-target processing remain declared model
choices; this engine does not identify a deployed GPU's hidden parameters.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from heapq import heappop, heappush

from .htsim_nvlink import (
    NVLINK_ALIGNED_PROFILE_IMPLEMENTATION,
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

CreditKey = tuple[int, int, int, str]
VoqKey = tuple[int, str, int]


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

    An engine instance serves exactly one batch. The containing domain service
    constructs it for each call, so failed preflight cannot mutate another run.
    Future streaming composition must keep the same calendar and ownership
    rather than precomputing an independent completion for each new extent.
    """

    def __init__(self, profile: NvlinkCandidateProfile, options: NvlinkAlignedOptions) -> None:
        if not isinstance(profile, NvlinkCandidateProfile):
            raise TypeError("profile must be an NvlinkCandidateProfile")
        if not isinstance(options, NvlinkAlignedOptions):
            raise TypeError("options must be an NvlinkAlignedOptions")
        self.profile = profile
        self.options = options
        self.flow_control = options.flow_control or (
            NvlinkFlowControlConfig.from_candidate_profile(profile)
        )
        self._used = False
        self._now = 0
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
        self._links: dict[tuple[int, int, int], int] = {}
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
        if type(include_switch) is not bool:
            raise TypeError("include_switch must be a boolean")
        transfers = tuple(transfers)
        self._queued_switch = include_switch and self.profile.switch.mode is NvlinkSwitchMode.QUEUED
        self._prepare(transfers)
        self._used = True
        for release in sorted({p.released_at_ps for p in self._packets}):
            self._schedule(release, "wake", -1)

        while self._events:
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

        self._check_drain()
        return self._result(transfers)

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

    def _schedule(self, at_ps: int, kind: str, index: int) -> None:
        if at_ps < self._now:
            raise AssertionError("NVLink event calendar attempted to move backwards")
        self._event_order += 1
        heappush(self._events, (at_ps, self._event_order, kind, index))

    def _update(self, index: int, **fields) -> NvlinkFlitPacket:
        packet = replace(self._packets[index], **fields)
        self._packets[index] = packet
        return packet

    @staticmethod
    def _rx_buffer_id(packet: NvlinkFlitPacket) -> str:
        return f"rx:{packet.destination}:{packet.virtual_channel}"

    @staticmethod
    def _switch_buffer_id(packet: NvlinkFlitPacket) -> str:
        return f"switch:{packet.source}:{packet.destination}:{packet.virtual_channel}"

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
        claim.credit_available_at_ps = self._now + self.flow_control.return_transport_latency_ps
        self._schedule(claim.credit_available_at_ps, "return", claim_index)
        if claim.credit_key is not None:
            source, destination, link_index, virtual_channel = claim.credit_key
            self._credit_releases.append(
                NvlinkCreditRelease(
                    packet_id=packet.packet_id,
                    source=source,
                    destination=destination,
                    link_index=link_index,
                    virtual_channel=virtual_channel,
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
        buffer_id = (
            self._switch_buffer_id(packet) if self._queued_switch else self._rx_buffer_id(packet)
        )
        capacity = (
            int(self.profile.switch.buffer_capacity_bytes or 0)
            if self._queued_switch
            else self.profile.rx.buffer_capacity_bytes
        )
        if self._buffer(buffer_id, capacity).available < packet.wire_bytes:
            return None
        pair = (packet.source, packet.destination)
        legal_links = []
        for link in range(self.profile.tx.links_per_peer):
            credit_key = (*pair, link, packet.virtual_channel)
            credits = self._credits.get(credit_key, self.flow_control.credits_per_pool)
            cursor = self._links.get((*pair, link), 0)
            if cursor <= self._now and credits >= packet.credit_units:
                legal_links.append((cursor, link))
        if not legal_links:
            return None
        _, link = min(legal_links)
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
            key = (packet.source, packet.destination, link, packet.virtual_channel)
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
            self._links[(packet.source, packet.destination, link)] = finished
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
            if self._switch_inputs.get(packet.source, 0) > self._now:
                continue
            if self._switch_outputs.get(packet.destination, 0) > self._now:
                continue
            pool = self._buffer(self._rx_buffer_id(packet), self.profile.rx.buffer_capacity_bytes)
            if pool.available < packet.wire_bytes:
                continue
            candidates.append(_NvlinkVoqHead(original_index=index, packet=packet))
        selected = self._switch_policy.select(candidates)
        for candidate in selected:
            index = candidate.original_index
            packet = self._packets[index]
            key = (packet.source, packet.virtual_channel, packet.destination)
            _, head = heappop(self._voq[key])
            if head != index:
                raise AssertionError("NVLink switch selected a non-head packet")
            self._rx_claims[index] = self._reserve(
                index,
                self._rx_buffer_id(packet),
                self.profile.rx.buffer_capacity_bytes,
                None,
            )
            finished = self._now + _serialize_ps(
                packet.wire_bytes, int(self.profile.switch.service_rate_bytes_per_second or 0)
            )
            self._switch_inputs[packet.source] = finished
            self._switch_outputs[packet.destination] = finished
            self._update(
                index,
                input_port=packet.source,
                output_port=packet.destination,
                switch_started_at_ps=self._now,
                switch_finished_at_ps=finished,
            )
            self._switch_grants.append(
                NvlinkSwitchGrant(
                    packet_id=packet.packet_id,
                    input_port=packet.source,
                    output_port=packet.destination,
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

    def _handle(self, kind: str, index: int) -> None:
        if kind == "wake":
            return
        if kind == "return":
            self._return(index)
        elif kind == "ack":
            if index in self._acknowledged:
                raise AssertionError("NVLink acknowledged a packet twice")
            self._acknowledged.add(index)
        elif kind == "tx_finish":
            if self._queued_switch:
                self._arrive(self._first_hop_claims[index])
                packet = self._packets[index]
                key = (packet.source, packet.virtual_channel, packet.destination)
                heappush(self._voq.setdefault(key, []), (self._now, index))
            else:
                self._rx_arrival(index)
        elif kind == "switch_finish":
            self._release(self._first_hop_claims[index])
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
            implementation=NVLINK_ALIGNED_PROFILE_IMPLEMENTATION,
            profile_id=self.profile.profile_id,
            authority=NvlinkMechanismAuthority.ALIGNED,
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
            physical_drain_time_ps=self._now,
        )
