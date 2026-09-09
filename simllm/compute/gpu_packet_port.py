"""GPU peer packet projections of one retained physical timing authority.

Each GPU exposes one logical peer port over its declared physical attachments.
The calendar owns bytes and time. The common version 2 ledger validates packet
observations; graph events carry only logical extent identities and visibility.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace

from simllm.backends.htsim_nvlink import (
    NvlinkAlignedDomainResult,
    NvlinkAlignedOptions,
    NvlinkCandidateProfile,
    NvlinkFlitPacket,
    NvlinkTransfer,
)
from simllm.backends.nvlink_runtime import NvlinkCausalEngine, NvlinkPhysicalBinding
from simllm.core.execution import CompletionEvent, EventPhase, ResourceKind, ResourceRef
from simllm.core.packet_port import (
    PacketAttemptEvent,
    PacketExtentAdmission,
    PacketExtentTerminal,
    PacketPortContext,
    PacketPortLedger,
    PacketPortSnapshot,
)
from simllm.core.runtime import QueueVisit

from .gpu_device import (
    GpuDeviceConfig,
    GpuPortCapability,
    GpuPortConfig,
    GpuPortDirection,
    GpuPortProtocol,
    GpuPortRole,
)


@dataclass(frozen=True)
class GpuPacketPortBinding:
    """One GPU's logical peer port and its exact physical attachment inventory."""

    gpu_rank: int
    config: GpuPortConfig
    physical_port_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.gpu_rank) is not int or self.gpu_rank < 0:
            raise ValueError("GPU packet port rank must be a nonnegative integer")
        if not isinstance(self.config, GpuPortConfig):
            raise TypeError("GPU packet port requires GpuPortConfig")
        if not self.config.advertises(GpuPortCapability.PEER_PACKET_SERVICE):
            raise ValueError("GPU packet binding requires the packet service capability")
        if not isinstance(self.physical_port_ids, (tuple, list)):
            raise TypeError("physical_port_ids must be an array")
        object.__setattr__(self, "physical_port_ids", tuple(self.physical_port_ids))
        if not self.physical_port_ids or not all(isinstance(x, str) and x.strip() for x in self.physical_port_ids):
            raise ValueError("GPU packet port needs named physical attachments")
        if len(set(self.physical_port_ids)) != len(self.physical_port_ids):
            raise ValueError("GPU packet port repeats a physical attachment")
        GpuDeviceConfig(device_id=f"gpu-{self.gpu_rank}", ports=(self.config,))


@dataclass(frozen=True)
class GpuPacketExtent:
    """An immutable join from one logical graph extent to admitted packet tokens."""

    execution_id: str
    operation_id: str
    subject_object_id: str
    transfer: NvlinkTransfer
    extent_token: int
    producer_id: int
    packet_ids: tuple[str, ...]
    attempt_tokens: tuple[int, ...]


class GpuPeerPacketSession:
    """One physical domain retained across ready phases and request steps.

    A missing selection is handled by the step sink's exact scalar bypass. A
    selected session admits only enabled, fully bound GPU peer ports. It never
    falls back to a scalar cursor after an invalid packet admission.
    """

    def __init__(
        self, session_id: str, profile: NvlinkCandidateProfile,
        binding: NvlinkPhysicalBinding, *, options: NvlinkAlignedOptions | None = None,
        ports: Sequence[GpuPacketPortBinding] | None = None,
    ) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("GPU packet session identity must be nonblank")
        if not isinstance(binding, NvlinkPhysicalBinding):
            raise TypeError("GPU packet session needs a physical binding")
        self.session_id = session_id
        self.binding = binding
        owners: dict[int, tuple[str, ...]] = {}
        for port in binding.fabric.ports:
            if port.gpu_rank is not None:
                owners[port.gpu_rank] = (*owners.get(port.gpu_rank, ()), port.port_id)
        if ports is None:
            ports = tuple(GpuPacketPortBinding(rank, GpuPortConfig(
                port_id=f"{binding.fabric.domain_id}:gpu-{rank}:peer",
                protocol=GpuPortProtocol.NVLINK, role=GpuPortRole.PEER_LINK,
                direction=GpuPortDirection.BIDIRECTIONAL,
                capabilities=(GpuPortCapability.PEER_PACKET_SERVICE,),
            ), physical_ids) for rank, physical_ids in owners.items())
        self.ports = tuple(ports)
        if any(not isinstance(port, GpuPacketPortBinding) for port in self.ports):
            raise TypeError("GPU packet ports must be typed bindings")
        by_rank = {port.gpu_rank: port for port in self.ports}
        if len(by_rank) != len(self.ports) or set(by_rank) != set(owners):
            raise ValueError("GPU packet ports must cover each physical GPU owner exactly once")
        if len({port.config.port_id for port in self.ports}) != len(self.ports):
            raise ValueError("GPU logical port identities must be unique")
        for port in self.ports:
            if set(port.physical_port_ids) != set(owners[port.gpu_rank]):
                raise ValueError("GPU port binding disagrees with its physical owner")
        self._ports = by_rank
        self._ledgers = {
            rank: PacketPortLedger(PacketPortContext(
                session_id, port.config.port_id, tx_start_boundary="link_grant",
                link_ids=tuple(link.link_id for link in binding.fabric.links
                               if link.endpoint_a in port.physical_port_ids or link.endpoint_b in port.physical_port_ids),
                packet_kinds=("data",),
            )) for rank, port in by_rank.items()
        }
        self._engine = NvlinkCausalEngine(profile, options or NvlinkAlignedOptions(),
                                          physical=binding, on_packet_event=self._observe)
        self._next_token = 1
        self._extents: dict[str, GpuPacketExtent] = {}
        self._packets: dict[str, PacketAttemptEvent] = {}
        self._started: set[str] = set()
        self._visible: set[str] = set()
        self._retired: set[str] = set()
        self._events: list[CompletionEvent] = []
        self._packet_events: list[PacketAttemptEvent] = []
        self._drained_result: NvlinkAlignedDomainResult | None = None

    @property
    def now_ps(self) -> int:
        return self._engine.now_ps

    @property
    def has_pending_physical_work(self) -> bool:
        return self._engine.has_pending_physical_work

    @property
    def extents(self) -> tuple[GpuPacketExtent, ...]:
        return tuple(self._extents.values())

    @property
    def events(self) -> tuple[CompletionEvent, ...]:
        return tuple(self._events)

    @property
    def packet_events(self) -> tuple[PacketAttemptEvent, ...]:
        return tuple(self._packet_events)

    @property
    def packets(self) -> tuple[NvlinkFlitPacket, ...]:
        return self._engine.packets

    @property
    def physical_paths(self):
        return self._engine.physical_paths

    @property
    def resource_visits(self) -> tuple[QueueVisit, ...]:
        """Completed resource visits projected from the physical calendar.

        A TX grant acquires source feed and an attachment together; a switch
        grant similarly acquires its input, output and output attachment.
        Each resource retains that coupled grant's ready boundary. Their waits
        and services overlap and are additive work, never a token-latency sum.
        Unfinished visits remain in packets, paths and buffer claims instead
        of inventing completion timestamps. Packet IDs stay in this sidecar.
        """
        paths = {row.packet_id: row for row in self.physical_paths}
        visits = []
        for packet in self.packets:
            observation = paths.get(packet.packet_id)
            if observation is None:
                continue
            extent = self._extents[packet.extent_id]
            path = observation.path

            def add(stage, resource_id, submitted, eligible, started, finished, completed,
                    *, extent=extent, packet=packet):
                if any(value is None for value in (eligible, started, finished, completed)) or completed > self.now_ps:
                    return
                visits.append(QueueVisit(
                    extent.execution_id, extent.operation_id,
                    ResourceRef(ResourceKind.NVLINK, f"{self.binding.fabric.domain_id}:{resource_id}"),
                    submitted, eligible, started, finished, completed,
                    packet.wire_bytes, packet.packet_id, stage,
                ))

            add("source_feed", f"gpu-{packet.source}:feed", packet.released_at_ps,
                packet.tx_eligible_at_ps, packet.tx_started_at_ps,
                observation.source_feed_finished_at_ps, observation.source_feed_finished_at_ps)
            add("input_attachment", f"link:{path.input_link.link_id}:{path.source_port_id}",
                packet.released_at_ps, packet.tx_eligible_at_ps, packet.tx_started_at_ps,
                packet.tx_finished_at_ps, observation.first_hop_arrived_at_ps)
            if path.output_link is not None:
                for stage, port_id in (("switch_input", path.switch_input_port_id),
                                       ("switch_output", path.switch_output_port_id)):
                    add(stage, f"port:{port_id}", observation.first_hop_arrived_at_ps,
                        observation.first_hop_arrived_at_ps, packet.switch_started_at_ps,
                        packet.switch_finished_at_ps, packet.switch_finished_at_ps)
                add("output_attachment", f"link:{path.output_link.link_id}:{path.switch_output_port_id}",
                    observation.first_hop_arrived_at_ps, observation.first_hop_arrived_at_ps,
                    packet.switch_started_at_ps, packet.switch_finished_at_ps, packet.rx_buffer_accepted_at_ps)
            add("receive_ingress", f"gpu-{packet.destination}:receive", packet.rx_buffer_accepted_at_ps,
                packet.rx_buffer_accepted_at_ps, packet.rx_started_at_ps, packet.rx_finished_at_ps, packet.visible_at_ps)
        return tuple(visits)

    def port_snapshots(self, *, final: bool = False) -> tuple[tuple[int, PacketPortSnapshot], ...]:
        return tuple((rank, ledger.snapshot(final=final)) for rank, ledger in self._ledgers.items())

    def _prepare(self, execution_id: str, operation_id: str, transfers: Sequence[NvlinkTransfer],
                 *, ready_at_ps: int | None = None):
        for name, value in (("execution_id", execution_id), ("operation_id", operation_id)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonblank")
        transfers = tuple(transfers)
        ready = self.now_ps if ready_at_ps is None else ready_at_ps
        if type(ready) is not int or not self.now_ps <= ready < 2**64:
            raise ValueError("GPU packet preflight needs a nonrewinding uint64 ready boundary")
        packets = self._engine.preview(transfers, include_switch=self.binding.fabric.switched)
        for transfer in transfers:
            if transfer.released_at_ps != ready:
                raise ValueError("GPU packet phase must be admitted at its current ready boundary")
            if not all(self._ports[rank].config.enabled for rank in (transfer.source, transfer.destination)):
                raise ValueError("GPU packet transfer selects a disabled port")
        next_token = self._next_token
        extents, descriptors, admissions = [], {}, []
        for transfer in transfers:
            extent_token = next_token
            next_token += 1
            children = tuple(packet for packet in packets if packet.extent_id == transfer.extent_id)
            attempt_tokens = tuple(range(next_token, next_token + len(children)))
            next_token += len(children)
            extent = GpuPacketExtent(execution_id, operation_id, f"peer-extent:{transfer.extent_id}",
                                     transfer, extent_token, extent_token,
                                     tuple(packet.packet_id for packet in children), attempt_tokens)
            admission = PacketExtentAdmission(extent_token, extent_token, transfer.payload_bytes, ready)
            admissions.append((transfer.source, admission))
            offset = 0
            for packet, token in zip(children, attempt_tokens, strict=True):
                descriptors[packet.packet_id] = PacketAttemptEvent(
                    token, extent_token, extent_token, "packet_tx_started", packet.released_at_ps,
                    0, packet.sequence, 0, offset, packet.payload_bytes, packet.wire_bytes, "data",
                )
                offset += packet.payload_bytes
            extents.append(extent)
        return transfers, tuple(extents), descriptors, tuple(admissions), next_token

    def validate_phase(self, execution_id: str, operation_id: str, transfers: Sequence[NvlinkTransfer],
                       *, ready_at_ps: int | None = None) -> None:
        """Dry preflight, including every token and immutable graph correlation."""
        self._prepare(execution_id, operation_id, transfers, ready_at_ps=ready_at_ps)

    def admit(self, execution_id: str, operation_id: str, transfers: Sequence[NvlinkTransfer]) -> tuple[GpuPacketExtent, ...]:
        transfers, extents, descriptors, admissions, next_token = self._prepare(execution_id, operation_id, transfers)
        # No callbacks run during calendar admission. All projection records and
        # capability checks are valid before this sole mutable authority commits.
        self._engine.admit(transfers, include_switch=self.binding.fabric.switched)
        self._drained_result = None
        for rank, admission in admissions:
            self._ledgers[rank].admit(admission)
        self._next_token = next_token
        self._packets.update(descriptors)
        for extent in extents:
            self._extents[extent.transfer.extent_id] = extent
            self._emit(extent, EventPhase.SUBMITTED, self.now_ps)
            self._emit(extent, EventPhase.QUEUED, self.now_ps)
        return extents

    def _emit(self, extent: GpuPacketExtent, phase: EventPhase, at_ps: int) -> None:
        self._events.append(CompletionEvent(
            extent.execution_id, extent.operation_id, phase, at_ps,
            ResourceRef(ResourceKind.NVLINK, self._ports[extent.transfer.source].config.port_id),
            extent.transfer.payload_bytes if phase is EventPhase.COMPLETED else 0,
            extent.subject_object_id,
        ))

    def _observe(self, kind: str, packet: NvlinkFlitPacket, at_ps: int) -> None:
        extent = self._extents[packet.extent_id]
        if kind == "consumer_visible":
            self._visible.add(packet.packet_id)
            if all(name in self._visible for name in extent.packet_ids):
                self._emit(extent, EventPhase.COMPLETED, at_ps)
            return
        row = replace(self._packets[packet.packet_id], event_kind=kind, event_time_ps=at_ps)
        ledger = self._ledgers[extent.transfer.source]
        ledger.observe(row)
        self._packet_events.append(row)
        if kind == "packet_tx_started" and packet.extent_id not in self._started:
            self._started.add(packet.extent_id)
            self._emit(extent, EventPhase.STARTED, at_ps)
        if kind == "delivered":
            self._retired.add(packet.packet_id)
            if all(name in self._retired for name in extent.packet_ids):
                ledger.retire(PacketExtentTerminal(extent.extent_token, extent.producer_id, "delivered", at_ps))

    def advance_to(self, at_ps: int) -> None:
        self._engine.advance_to(at_ps)

    def advance_until_visible(self, extent_ids: Sequence[str]) -> int:
        return self._engine.advance_until_visible(extent_ids)

    def drain(self) -> NvlinkAlignedDomainResult:
        result = self._engine.drain()
        self.port_snapshots(final=True)
        self._drained_result = result
        return result

    def evidence(self) -> dict:
        """A partial or final observation, with no hidden scalar timing owner."""
        return {
            "schema": "simllm-gpu-peer-packet-observation-v1",
            "session_id": self.session_id,
            "authority": "physical_retained_v1",
            "evidence_class": "declared_model_not_hardware_measurement",
            "now_ps": self.now_ps,
            "has_pending_physical_work": self.has_pending_physical_work,
            "binding": asdict(self.binding),
            "ports": [asdict(port) for port in self.ports],
            "contexts": [asdict(ledger.context) for ledger in self._ledgers.values()],
            "extents": [asdict(extent) for extent in self.extents],
            "packet_events": [row.to_row() for row in self.packet_events],
            "packets": [asdict(packet) for packet in self.packets],
            "physical_paths": [asdict(path) for path in self.physical_paths],
            "resource_visits": [{**asdict(visit), "resource": {
                "kind": visit.resource.kind.value, "resource_id": visit.resource.resource_id,
            }} for visit in self.resource_visits],
            "buffer_claims": [asdict(claim) for claim in self._engine.buffer_claims],
            "port_snapshots": [{"gpu_rank": rank, **asdict(snapshot)} for rank, snapshot in self.port_snapshots()],
            "buffer_ownership": self._engine.buffer_ownership,
            "drained_result": None if self._drained_result is None else asdict(self._drained_result),
        }
