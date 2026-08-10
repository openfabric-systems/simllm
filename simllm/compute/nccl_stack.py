"""Name-mirrored zero-time skeleton of the NCCL communication stack.

This module makes the main NCCL call structure executable before any timing
mechanism is attached. A caller supplies the one :class:`VirtualClock` used by
all layers. Calls never advance that clock. Instead, every modeled boundary,
producer signal, and successful consumer poll emits a strictly versioned
stack-internal event.

The intra-node route executes only the collective-kernel shape. The inter-node
route uses the default CPU-host proxy shape: a GPU kernel publishes a FIFO
slot, the proxy calls an ``ncclNet``-shaped plugin, the plugin calls an
ibverbs-shaped seam, and completion returns through the tail counter. No GPU
initiated submission leg or service-time approximation is present in this
slice.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from simllm.compute.nccl import nccl_ring_egress_bytes
from simllm.core._wire import (
    _array,
    _enum_value,
    _fields,
    _integer,
    _object,
    _optional_integer,
    _optional_string,
    _string,
)
from simllm.core.clock import VirtualClock

NCCL_STACK_EVENT_SCHEMA = "simllm-nccl-stack-event-v1"


class NcclStackEventKind(enum.Enum):
    """How one observable stack boundary transfers control or state."""

    CALL = "call"
    SIGNAL_STORE = "signal_store"
    POLL_OBSERVES = "poll_observes"


class NcclStackLane(enum.Enum):
    """Execution lane that emits one stack event."""

    CPU = "cpu"
    GPU = "gpu"
    RNIC = "rnic"


class NcclRoute(enum.Enum):
    """Physical route selected for a collective."""

    INTRA_NODE = "intra_node"
    INTER_NODE = "inter_node"


@dataclass(frozen=True)
class NcclStackEvent:
    """One immutable stack-internal observability event.

    ``sequence`` provides deterministic order when zero-time events share a
    timestamp. A successful poll names the earlier producer event in
    ``observed_signal_sequence``. This is deliberately separate from the core
    completion-event schema because later runtime work will project selected
    stack events onto that public metric chain.
    """

    sequence: int
    timestamp_ps: int
    kind: NcclStackEventKind
    lane: NcclStackLane
    function: str
    rank: int
    communicator_id: str
    operation_id: str | None = None
    channel_id: int | None = None
    chunk_id: int | None = None
    peer_rank: int | None = None
    subject: str | None = None
    value: int | None = None
    observed_signal_sequence: int | None = None

    def __post_init__(self) -> None:
        _validate_nccl_stack_event(self)

    def to_json(self) -> dict[str, Any]:
        """Return the canonical strict JSON form."""

        return nccl_stack_event_to_json(self)

    @classmethod
    def from_json(cls, value: Any) -> NcclStackEvent:
        """Parse one canonical strict JSON event."""

        return nccl_stack_event_from_json(value)


def _validate_nccl_stack_event(event: NcclStackEvent, path: str = "event") -> None:
    if not isinstance(event, NcclStackEvent):
        raise TypeError(f"{path}: expected NcclStackEvent")
    _integer(event.sequence, f"{path}.sequence", nonnegative=True)
    _integer(event.timestamp_ps, f"{path}.timestamp_ps", nonnegative=True)
    if not isinstance(event.kind, NcclStackEventKind):
        raise TypeError(f"{path}.kind: expected NcclStackEventKind")
    if not isinstance(event.lane, NcclStackLane):
        raise TypeError(f"{path}.lane: expected NcclStackLane")
    _string(event.function, f"{path}.function")
    _integer(event.rank, f"{path}.rank", nonnegative=True)
    _string(event.communicator_id, f"{path}.communicator_id")
    _optional_string(event.operation_id, f"{path}.operation_id")
    _optional_integer(event.channel_id, f"{path}.channel_id", nonnegative=True)
    _optional_integer(event.chunk_id, f"{path}.chunk_id", nonnegative=True)
    _optional_integer(event.peer_rank, f"{path}.peer_rank", nonnegative=True)
    _optional_string(event.subject, f"{path}.subject")
    _optional_integer(event.value, f"{path}.value", nonnegative=True)
    observed = _optional_integer(
        event.observed_signal_sequence,
        f"{path}.observed_signal_sequence",
        nonnegative=True,
    )
    if event.kind is NcclStackEventKind.CALL:
        if observed is not None:
            raise ValueError(f"{path}: a call cannot observe a signal")
    elif event.kind is NcclStackEventKind.SIGNAL_STORE:
        if event.subject is None:
            raise ValueError(f"{path}: a signal store requires a subject")
        if observed is not None:
            raise ValueError(f"{path}: a signal store cannot observe another signal")
    else:
        if event.subject is None:
            raise ValueError(f"{path}: a poll observation requires a subject")
        if observed is None:
            raise ValueError(f"{path}: a poll observation requires its producer sequence")
        if observed >= event.sequence:
            raise ValueError(f"{path}: a poll must observe an earlier signal")


def nccl_stack_event_to_json(event: NcclStackEvent) -> dict[str, Any]:
    """Serialize one stack event with its independent version tag."""

    _validate_nccl_stack_event(event)
    return {
        "schema": NCCL_STACK_EVENT_SCHEMA,
        "sequence": event.sequence,
        "timestamp_ps": event.timestamp_ps,
        "kind": event.kind.value,
        "lane": event.lane.value,
        "function": event.function,
        "rank": event.rank,
        "communicator_id": event.communicator_id,
        "operation_id": event.operation_id,
        "channel_id": event.channel_id,
        "chunk_id": event.chunk_id,
        "peer_rank": event.peer_rank,
        "subject": event.subject,
        "value": event.value,
        "observed_signal_sequence": event.observed_signal_sequence,
    }


def nccl_stack_event_from_json(value: Any) -> NcclStackEvent:
    """Parse one strict ``simllm-nccl-stack-event-v1`` object."""

    payload = _object(value, "event")
    required = {
        "schema",
        "sequence",
        "timestamp_ps",
        "kind",
        "lane",
        "function",
        "rank",
        "communicator_id",
        "operation_id",
        "channel_id",
        "chunk_id",
        "peer_rank",
        "subject",
        "value",
        "observed_signal_sequence",
    }
    _fields(payload, "event", required=required)
    schema = _string(payload["schema"], "event.schema")
    if schema != NCCL_STACK_EVENT_SCHEMA:
        raise ValueError(
            "event.schema: unsupported schema "
            f"{schema!r}; expected {NCCL_STACK_EVENT_SCHEMA!r}"
        )
    return NcclStackEvent(
        sequence=_integer(payload["sequence"], "event.sequence", nonnegative=True),
        timestamp_ps=_integer(
            payload["timestamp_ps"], "event.timestamp_ps", nonnegative=True
        ),
        kind=_enum_value(NcclStackEventKind, payload["kind"], "event.kind"),
        lane=_enum_value(NcclStackLane, payload["lane"], "event.lane"),
        function=_string(payload["function"], "event.function"),
        rank=_integer(payload["rank"], "event.rank", nonnegative=True),
        communicator_id=_string(payload["communicator_id"], "event.communicator_id"),
        operation_id=_optional_string(payload["operation_id"], "event.operation_id"),
        channel_id=_optional_integer(
            payload["channel_id"], "event.channel_id", nonnegative=True
        ),
        chunk_id=_optional_integer(payload["chunk_id"], "event.chunk_id", nonnegative=True),
        peer_rank=_optional_integer(
            payload["peer_rank"], "event.peer_rank", nonnegative=True
        ),
        subject=_optional_string(payload["subject"], "event.subject"),
        value=_optional_integer(payload["value"], "event.value", nonnegative=True),
        observed_signal_sequence=_optional_integer(
            payload["observed_signal_sequence"],
            "event.observed_signal_sequence",
            nonnegative=True,
        ),
    )


def validate_nccl_stack_events(events: Sequence[NcclStackEvent]) -> None:
    """Validate complete-stream order and producer-to-poll causality."""

    signals: dict[int, NcclStackEvent] = {}
    previous_timestamp = -1
    for expected_sequence, event in enumerate(events):
        _validate_nccl_stack_event(event, f"events[{expected_sequence}]")
        if event.sequence != expected_sequence:
            raise ValueError(
                f"events[{expected_sequence}].sequence: expected {expected_sequence}, "
                f"got {event.sequence}"
            )
        if event.timestamp_ps < previous_timestamp:
            raise ValueError(
                f"events[{expected_sequence}].timestamp_ps: event time moved backward"
            )
        previous_timestamp = event.timestamp_ps
        if event.kind is NcclStackEventKind.SIGNAL_STORE:
            signals[event.sequence] = event
            continue
        if event.kind is not NcclStackEventKind.POLL_OBSERVES:
            continue
        producer_sequence = event.observed_signal_sequence
        assert producer_sequence is not None
        producer = signals.get(producer_sequence)
        if producer is None:
            raise ValueError(
                f"events[{expected_sequence}]: producer sequence "
                f"{producer_sequence} is not an earlier signal store"
            )
        if event.timestamp_ps < producer.timestamp_ps:
            raise ValueError(
                f"events[{expected_sequence}]: poll time precedes its producer"
            )
        identity_fields = (
            "rank",
            "communicator_id",
            "operation_id",
            "channel_id",
            "chunk_id",
            "peer_rank",
            "subject",
            "value",
        )
        mismatched = [
            name for name in identity_fields if getattr(event, name) != getattr(producer, name)
        ]
        if mismatched:
            raise ValueError(
                f"events[{expected_sequence}]: poll and producer disagree on {mismatched}"
            )


def nccl_stack_events_to_json(events: Sequence[NcclStackEvent]) -> list[dict[str, Any]]:
    """Serialize a complete validated event stream."""

    validate_nccl_stack_events(events)
    return [nccl_stack_event_to_json(event) for event in events]


def nccl_stack_events_from_json(value: Any) -> tuple[NcclStackEvent, ...]:
    """Parse and validate a complete event stream."""

    events = tuple(
        nccl_stack_event_from_json(item)
        for item in _array(value, "events")
    )
    validate_nccl_stack_events(events)
    return events


class _NcclStackObserver:
    """The sole event sequence and timestamp issuer for one stack."""

    def __init__(self, clock: VirtualClock):
        self._clock = clock
        self._events: list[NcclStackEvent] = []

    @property
    def events(self) -> tuple[NcclStackEvent, ...]:
        return tuple(self._events)

    def call(
        self,
        function: str,
        lane: NcclStackLane,
        *,
        rank: int,
        communicator_id: str,
        operation_id: str | None = None,
        channel_id: int | None = None,
        chunk_id: int | None = None,
        peer_rank: int | None = None,
        subject: str | None = None,
        value: int | None = None,
    ) -> NcclStackEvent:
        return self._emit(
            kind=NcclStackEventKind.CALL,
            lane=lane,
            function=function,
            rank=rank,
            communicator_id=communicator_id,
            operation_id=operation_id,
            channel_id=channel_id,
            chunk_id=chunk_id,
            peer_rank=peer_rank,
            subject=subject,
            value=value,
        )

    def signal(
        self,
        function: str,
        lane: NcclStackLane,
        *,
        rank: int,
        communicator_id: str,
        operation_id: str,
        subject: str,
        value: int,
        channel_id: int | None = None,
        chunk_id: int | None = None,
        peer_rank: int | None = None,
    ) -> NcclStackEvent:
        return self._emit(
            kind=NcclStackEventKind.SIGNAL_STORE,
            lane=lane,
            function=function,
            rank=rank,
            communicator_id=communicator_id,
            operation_id=operation_id,
            channel_id=channel_id,
            chunk_id=chunk_id,
            peer_rank=peer_rank,
            subject=subject,
            value=value,
        )

    def poll(
        self,
        function: str,
        lane: NcclStackLane,
        producer: NcclStackEvent,
    ) -> NcclStackEvent:
        if producer.kind is not NcclStackEventKind.SIGNAL_STORE:
            raise ValueError("a poll producer must be a signal-store event")
        return self._emit(
            kind=NcclStackEventKind.POLL_OBSERVES,
            lane=lane,
            function=function,
            rank=producer.rank,
            communicator_id=producer.communicator_id,
            operation_id=producer.operation_id,
            channel_id=producer.channel_id,
            chunk_id=producer.chunk_id,
            peer_rank=producer.peer_rank,
            subject=producer.subject,
            value=producer.value,
            observed_signal_sequence=producer.sequence,
        )

    def _emit(
        self,
        *,
        kind: NcclStackEventKind,
        lane: NcclStackLane,
        function: str,
        rank: int,
        communicator_id: str,
        operation_id: str | None,
        channel_id: int | None,
        chunk_id: int | None,
        peer_rank: int | None,
        subject: str | None,
        value: int | None,
        observed_signal_sequence: int | None = None,
    ) -> NcclStackEvent:
        timestamp_ps = _integer(
            self._clock.now_ps,
            "clock.now_ps",
            nonnegative=True,
        )
        if self._events and timestamp_ps < self._events[-1].timestamp_ps:
            raise ValueError("the caller-supplied virtual clock moved backward")
        event = NcclStackEvent(
            sequence=len(self._events),
            timestamp_ps=timestamp_ps,
            kind=kind,
            lane=lane,
            function=function,
            rank=rank,
            communicator_id=communicator_id,
            operation_id=operation_id,
            channel_id=channel_id,
            chunk_id=chunk_id,
            peer_rank=peer_rank,
            subject=subject,
            value=value,
            observed_signal_sequence=observed_signal_sequence,
        )
        self._events.append(event)
        return event


def _require_positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class NcclStackConfig:
    """Structural planner and FIFO settings for the zero-time skeleton."""

    channel_count: int = 1
    chunk_bytes: int = 1 << 20
    fifo_slots_per_channel: int = 1

    def __post_init__(self) -> None:
        _require_positive_integer("channel_count", self.channel_count)
        _require_positive_integer("chunk_bytes", self.chunk_bytes)
        _require_positive_integer("fifo_slots_per_channel", self.fifo_slots_per_channel)


@dataclass(frozen=True)
class NcclLogicalChannel:
    """One ring channel with stable peers for a communicator rank."""

    channel_id: int
    send_peer: int
    receive_peer: int


@dataclass(frozen=True)
class NcclChunk:
    """One per-rank wire chunk assigned to a logical channel."""

    chunk_id: int
    offset_bytes: int
    byte_count: int
    channel_id: int


@dataclass(frozen=True)
class NcclCollectivePlan:
    """The traffic planner's immutable output for one ring all-reduce."""

    payload_bytes: int
    wire_bytes: int
    chunk_bytes: int
    channel_count: int
    chunks: tuple[NcclChunk, ...]

    def chunks_per_channel(self) -> tuple[int, ...]:
        counts = [0] * self.channel_count
        for chunk in self.chunks:
            counts[chunk.channel_id] += 1
        return tuple(counts)


class NcclTrafficPlanner:
    """Construct ring channels, chunks, and round-robin assignments."""

    def __init__(self, config: NcclStackConfig, observer: _NcclStackObserver):
        self._config = config
        self._observer = observer

    def build_logical_channels(
        self,
        *,
        rank: int,
        world_size: int,
        communicator_id: str,
    ) -> tuple[NcclLogicalChannel, ...]:
        send_peer = (rank + 1) % world_size
        receive_peer = (rank - 1) % world_size
        self._observer.call(
            "ncclBuildLogicalChannels",
            NcclStackLane.CPU,
            rank=rank,
            communicator_id=communicator_id,
            value=self._config.channel_count,
        )
        channels = []
        for channel_id in range(self._config.channel_count):
            self._observer.call(
                "ncclConstructLogicalChannel",
                NcclStackLane.CPU,
                rank=rank,
                communicator_id=communicator_id,
                channel_id=channel_id,
                peer_rank=send_peer,
                subject="logical_channel",
                value=receive_peer,
            )
            channels.append(
                NcclLogicalChannel(
                    channel_id=channel_id,
                    send_peer=send_peer,
                    receive_peer=receive_peer,
                )
            )
        return tuple(channels)

    def plan_all_reduce(
        self,
        communicator: NcclCommunicator,
        *,
        payload_bytes: int,
        operation_id: str,
    ) -> NcclCollectivePlan:
        payload_bytes = _require_positive_integer("payload_bytes", payload_bytes)
        _, remainder = divmod(payload_bytes, communicator.world_size)
        if remainder:
            raise ValueError(
                "payload_bytes must divide evenly by communicator world_size "
                "for an exact ring plan"
            )
        wire_bytes = nccl_ring_egress_bytes(
            payload_bytes=payload_bytes,
            world_size=communicator.world_size,
        )
        self._observer.call(
            "ncclPlanAllReduce",
            NcclStackLane.CPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            value=wire_bytes,
        )
        raw_chunks = self._chunk_payload(
            communicator,
            operation_id=operation_id,
            wire_bytes=wire_bytes,
        )
        chunks = self._assign_chunks_to_channels(
            communicator,
            operation_id=operation_id,
            raw_chunks=raw_chunks,
        )
        return NcclCollectivePlan(
            payload_bytes=payload_bytes,
            wire_bytes=wire_bytes,
            chunk_bytes=self._config.chunk_bytes,
            channel_count=self._config.channel_count,
            chunks=chunks,
        )

    def _chunk_payload(
        self,
        communicator: NcclCommunicator,
        *,
        operation_id: str,
        wire_bytes: int,
    ) -> tuple[tuple[int, int, int], ...]:
        chunk_count = (wire_bytes + self._config.chunk_bytes - 1) // self._config.chunk_bytes
        self._observer.call(
            "ncclChunkPayload",
            NcclStackLane.CPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            value=chunk_count,
        )
        chunks = []
        for chunk_id in range(chunk_count):
            offset = chunk_id * self._config.chunk_bytes
            byte_count = min(self._config.chunk_bytes, wire_bytes - offset)
            chunks.append((chunk_id, offset, byte_count))
        return tuple(chunks)

    def _assign_chunks_to_channels(
        self,
        communicator: NcclCommunicator,
        *,
        operation_id: str,
        raw_chunks: Sequence[tuple[int, int, int]],
    ) -> tuple[NcclChunk, ...]:
        self._observer.call(
            "ncclAssignChunksToChannels",
            NcclStackLane.CPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            value=self._config.channel_count,
        )
        return tuple(
            NcclChunk(
                chunk_id=chunk_id,
                offset_bytes=offset,
                byte_count=byte_count,
                channel_id=chunk_id % self._config.channel_count,
            )
            for chunk_id, offset, byte_count in raw_chunks
        )


@dataclass
class NcclDataFifoSlot:
    """One GPU-resident FIFO slot with metadata but no byte contents."""

    slot_id: int
    chunk_id: int | None = None
    byte_count: int = 0
    ready: bool = False


@dataclass(frozen=True)
class NcclChannelSnapshot:
    """Read-only terminal projection of one GPU channel."""

    channel_id: int
    head: int
    tail: int
    ready_flags: tuple[bool, ...]
    slot_chunk_ids: tuple[int | None, ...]
    slot_byte_counts: tuple[int, ...]


@dataclass(frozen=True)
class _PublishedSlot:
    slot_id: int
    ready_signal: NcclStackEvent
    head_signal: NcclStackEvent


class NcclGpuChannel:
    """Mutable authority for one channel FIFO, its counters, and flags."""

    def __init__(
        self,
        logical_channel: NcclLogicalChannel,
        fifo_slots: int,
        observer: _NcclStackObserver,
    ):
        self.logical_channel = logical_channel
        self.head = 0
        self.tail = 0
        self.slots = [NcclDataFifoSlot(slot_id=index) for index in range(fifo_slots)]
        self._observer = observer

    def copy_chunk_to_fifo(
        self,
        communicator: NcclCommunicator,
        *,
        operation_id: str,
        chunk: NcclChunk,
    ) -> _PublishedSlot:
        if chunk.channel_id != self.logical_channel.channel_id:
            raise ValueError("chunk assigned to the wrong GPU channel")
        if self.head - self.tail >= len(self.slots):
            raise RuntimeError("NCCL data FIFO is full")
        slot = self.slots[self.head % len(self.slots)]
        if slot.ready:
            raise RuntimeError("NCCL data FIFO slot is still ready")
        self._observer.call(
            "ncclCopyChunkToFifo",
            NcclStackLane.GPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            channel_id=chunk.channel_id,
            chunk_id=chunk.chunk_id,
            peer_rank=self.logical_channel.send_peer,
            subject="data_fifo_slot",
            value=chunk.byte_count,
        )
        slot.chunk_id = chunk.chunk_id
        slot.byte_count = chunk.byte_count
        slot.ready = True
        ready_signal = self._observer.signal(
            "ncclStoreReadyFlag",
            NcclStackLane.GPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            channel_id=chunk.channel_id,
            chunk_id=chunk.chunk_id,
            peer_rank=self.logical_channel.send_peer,
            subject="ready_flag",
            value=1,
        )
        self.head += 1
        head_signal = self._observer.signal(
            "ncclStoreHead",
            NcclStackLane.GPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            channel_id=chunk.channel_id,
            chunk_id=chunk.chunk_id,
            peer_rank=self.logical_channel.send_peer,
            subject="head_counter",
            value=self.head,
        )
        return _PublishedSlot(
            slot_id=slot.slot_id,
            ready_signal=ready_signal,
            head_signal=head_signal,
        )

    def store_tail(
        self,
        communicator: NcclCommunicator,
        *,
        operation_id: str,
        chunk: NcclChunk,
        published: _PublishedSlot,
    ) -> NcclStackEvent:
        slot = self.slots[published.slot_id]
        if not slot.ready or slot.chunk_id != chunk.chunk_id:
            raise RuntimeError("proxy completion does not match a ready FIFO slot")
        if self.tail >= self.head:
            raise RuntimeError("tail cannot advance beyond head")
        self.tail += 1
        return self._observer.signal(
            "ncclProxyStoreTail",
            NcclStackLane.CPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            channel_id=chunk.channel_id,
            chunk_id=chunk.chunk_id,
            peer_rank=self.logical_channel.send_peer,
            subject="tail_counter",
            value=self.tail,
        )

    def poll_tail_and_release(
        self,
        communicator: NcclCommunicator,
        *,
        operation_id: str,
        chunk: NcclChunk,
        published: _PublishedSlot,
        tail_signal: NcclStackEvent,
    ) -> None:
        self._observer.poll("ncclKernelPollTail", NcclStackLane.GPU, tail_signal)
        slot = self.slots[published.slot_id]
        if not slot.ready or slot.chunk_id != chunk.chunk_id:
            raise RuntimeError("kernel release does not match a ready FIFO slot")
        slot.ready = False
        slot.chunk_id = None
        slot.byte_count = 0
        self._observer.signal(
            "ncclReleaseFifoSlot",
            NcclStackLane.GPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            channel_id=chunk.channel_id,
            chunk_id=chunk.chunk_id,
            peer_rank=self.logical_channel.send_peer,
            subject="ready_flag",
            value=0,
        )

    def snapshot(self) -> NcclChannelSnapshot:
        return NcclChannelSnapshot(
            channel_id=self.logical_channel.channel_id,
            head=self.head,
            tail=self.tail,
            ready_flags=tuple(slot.ready for slot in self.slots),
            slot_chunk_ids=tuple(slot.chunk_id for slot in self.slots),
            slot_byte_counts=tuple(slot.byte_count for slot in self.slots),
        )


@dataclass
class IbverbsRequest:
    """One zero-time verbs request and its synthesized CQE signal."""

    request_id: str
    direction: str
    completion_signal: NcclStackEvent
    polled: bool = False


class Ibverbs:
    """Trimmed ibverbs-shaped post and completion-queue seam."""

    def __init__(self, observer: _NcclStackObserver):
        self._observer = observer

    def post_send(
        self,
        communicator: NcclCommunicator,
        *,
        operation_id: str,
        chunk: NcclChunk,
        peer_rank: int,
    ) -> IbverbsRequest:
        return self._post(
            "ibverbs.post_send",
            "send",
            communicator,
            operation_id=operation_id,
            chunk=chunk,
            peer_rank=peer_rank,
        )

    def post_receive(
        self,
        communicator: NcclCommunicator,
        *,
        operation_id: str,
        chunk: NcclChunk,
        peer_rank: int,
    ) -> IbverbsRequest:
        """Serve the named receive seam for callers outside the send loop."""

        return self._post(
            "ibverbs.post_recv",
            "receive",
            communicator,
            operation_id=operation_id,
            chunk=chunk,
            peer_rank=peer_rank,
        )

    def _post(
        self,
        function: str,
        direction: str,
        communicator: NcclCommunicator,
        *,
        operation_id: str,
        chunk: NcclChunk,
        peer_rank: int,
    ) -> IbverbsRequest:
        self._observer.call(
            function,
            NcclStackLane.CPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            channel_id=chunk.channel_id,
            chunk_id=chunk.chunk_id,
            peer_rank=peer_rank,
            value=chunk.byte_count,
        )
        completion_signal = self._observer.signal(
            "ibverbs.write_cqe",
            NcclStackLane.RNIC,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            channel_id=chunk.channel_id,
            chunk_id=chunk.chunk_id,
            peer_rank=peer_rank,
            subject="completion_queue_entry",
            value=chunk.byte_count,
        )
        return IbverbsRequest(
            request_id=(
                f"{operation_id}:c{chunk.channel_id}:k{chunk.chunk_id}:{direction}"
            ),
            direction=direction,
            completion_signal=completion_signal,
        )

    def poll_cq(self, request: IbverbsRequest) -> bool:
        if request.polled:
            raise RuntimeError(f"verbs request {request.request_id!r} was already polled")
        self._observer.poll(
            "ibverbs.poll_cq",
            NcclStackLane.CPU,
            request.completion_signal,
        )
        request.polled = True
        return True


@dataclass
class NcclNetRequest:
    """One request returned by the trimmed ``ncclNet`` plugin seam."""

    request_id: str
    direction: str
    verbs_request: IbverbsRequest


class NcclNetPlugin:
    """Name-mirrored ``ncclNet`` send, receive, and test surface."""

    def __init__(self, observer: _NcclStackObserver, verbs: Ibverbs):
        self._observer = observer
        self._verbs = verbs

    def isend(
        self,
        communicator: NcclCommunicator,
        *,
        operation_id: str,
        chunk: NcclChunk,
        peer_rank: int,
    ) -> NcclNetRequest:
        self._observer.call(
            "ncclNet.isend",
            NcclStackLane.CPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            channel_id=chunk.channel_id,
            chunk_id=chunk.chunk_id,
            peer_rank=peer_rank,
            value=chunk.byte_count,
        )
        request = self._verbs.post_send(
            communicator,
            operation_id=operation_id,
            chunk=chunk,
            peer_rank=peer_rank,
        )
        return NcclNetRequest(request.request_id, "send", request)

    def irecv(
        self,
        communicator: NcclCommunicator,
        *,
        operation_id: str,
        chunk: NcclChunk,
        peer_rank: int,
    ) -> NcclNetRequest:
        self._observer.call(
            "ncclNet.irecv",
            NcclStackLane.CPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            channel_id=chunk.channel_id,
            chunk_id=chunk.chunk_id,
            peer_rank=peer_rank,
            value=chunk.byte_count,
        )
        request = self._verbs.post_receive(
            communicator,
            operation_id=operation_id,
            chunk=chunk,
            peer_rank=peer_rank,
        )
        return NcclNetRequest(request.request_id, "receive", request)

    def test(self, request: NcclNetRequest) -> bool:
        signal = request.verbs_request.completion_signal
        self._observer.call(
            "ncclNet.test",
            NcclStackLane.CPU,
            rank=signal.rank,
            communicator_id=signal.communicator_id,
            operation_id=signal.operation_id,
            channel_id=signal.channel_id,
            chunk_id=signal.chunk_id,
            peer_rank=signal.peer_rank,
            value=signal.value,
        )
        return self._verbs.poll_cq(request.verbs_request)


class NcclProxyProgressEngine:
    """Default CPU-host proxy progression loop."""

    def __init__(self, observer: _NcclStackObserver, net: NcclNetPlugin):
        self._observer = observer
        self._net = net

    def progress_send(
        self,
        communicator: NcclCommunicator,
        *,
        operation_id: str,
        chunk: NcclChunk,
        channel: NcclGpuChannel,
        published: _PublishedSlot,
    ) -> NcclStackEvent:
        self._observer.call(
            "ncclProxyProgress",
            NcclStackLane.CPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            channel_id=chunk.channel_id,
            chunk_id=chunk.chunk_id,
            peer_rank=channel.logical_channel.send_peer,
        )
        self._observer.poll(
            "ncclProxyPollHead",
            NcclStackLane.CPU,
            published.head_signal,
        )
        self._observer.poll(
            "ncclProxyPollReady",
            NcclStackLane.CPU,
            published.ready_signal,
        )
        request = self._net.isend(
            communicator,
            operation_id=operation_id,
            chunk=chunk,
            peer_rank=channel.logical_channel.send_peer,
        )
        if not self._net.test(request):
            raise RuntimeError("zero-time ncclNet send did not complete")
        return channel.store_tail(
            communicator,
            operation_id=operation_id,
            chunk=chunk,
            published=published,
        )


class NcclCollectiveKernel:
    """GPU-side collective kernel over per-channel FIFO state."""

    def __init__(
        self,
        observer: _NcclStackObserver,
        proxy: NcclProxyProgressEngine,
    ):
        self._observer = observer
        self._proxy = proxy

    def launch(
        self,
        communicator: NcclCommunicator,
        *,
        operation_id: str,
        route: NcclRoute,
        plan: NcclCollectivePlan,
    ) -> NcclStackEvent:
        self._observer.call(
            "ncclLaunchCollectiveKernel",
            NcclStackLane.CPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            value=len(plan.chunks),
        )
        self._observer.call(
            "ncclCollectiveKernel",
            NcclStackLane.GPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            value=len(plan.chunks),
        )
        if route is NcclRoute.INTRA_NODE:
            self._run_intra_node(communicator, operation_id=operation_id, plan=plan)
        else:
            self._run_inter_node(communicator, operation_id=operation_id, plan=plan)
        return self._observer.signal(
            "ncclKernelComplete",
            NcclStackLane.GPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            subject="kernel_completion",
            value=len(plan.chunks),
        )

    def _run_intra_node(
        self,
        communicator: NcclCommunicator,
        *,
        operation_id: str,
        plan: NcclCollectivePlan,
    ) -> None:
        for chunk in plan.chunks:
            channel = communicator.gpu_channel(chunk.channel_id)
            self._observer.call(
                "ncclNvlinkCollective",
                NcclStackLane.GPU,
                rank=communicator.rank,
                communicator_id=communicator.communicator_id,
                operation_id=operation_id,
                channel_id=chunk.channel_id,
                chunk_id=chunk.chunk_id,
                peer_rank=channel.logical_channel.send_peer,
                subject="nvlink",
                value=chunk.byte_count,
            )

    def _run_inter_node(
        self,
        communicator: NcclCommunicator,
        *,
        operation_id: str,
        plan: NcclCollectivePlan,
    ) -> None:
        for chunk in plan.chunks:
            channel = communicator.gpu_channel(chunk.channel_id)
            published = channel.copy_chunk_to_fifo(
                communicator,
                operation_id=operation_id,
                chunk=chunk,
            )
            tail_signal = self._proxy.progress_send(
                communicator,
                operation_id=operation_id,
                chunk=chunk,
                channel=channel,
                published=published,
            )
            channel.poll_tail_and_release(
                communicator,
                operation_id=operation_id,
                chunk=chunk,
                published=published,
                tail_signal=tail_signal,
            )


class NcclStack:
    """One composed skeleton whose caller owns the virtual clock."""

    def __init__(self, *, clock: VirtualClock, config: NcclStackConfig | None = None):
        if not isinstance(clock, VirtualClock):
            raise TypeError("clock must be a simllm.core.VirtualClock")
        self.clock = clock
        self.config = config or NcclStackConfig()
        self._observer = _NcclStackObserver(clock)
        self.planner = NcclTrafficPlanner(self.config, self._observer)
        self.ibverbs = Ibverbs(self._observer)
        self.nccl_net = NcclNetPlugin(self._observer, self.ibverbs)
        self.proxy = NcclProxyProgressEngine(self._observer, self.nccl_net)
        self.kernel = NcclCollectiveKernel(self._observer, self.proxy)

    @property
    def events(self) -> tuple[NcclStackEvent, ...]:
        return self._observer.events


@dataclass
class NcclCommunicator:
    """One rank's communicator and GPU-resident channel objects."""

    stack: NcclStack
    communicator_id: str
    rank: int
    world_size: int
    logical_channels: tuple[NcclLogicalChannel, ...]
    gpu_channels: tuple[NcclGpuChannel, ...]

    def gpu_channel(self, channel_id: int) -> NcclGpuChannel:
        if channel_id < 0 or channel_id >= len(self.gpu_channels):
            raise ValueError(f"unknown NCCL channel {channel_id}")
        channel = self.gpu_channels[channel_id]
        if channel.logical_channel.channel_id != channel_id:
            raise RuntimeError("GPU channel order no longer matches channel identity")
        return channel

    def snapshots(self) -> tuple[NcclChannelSnapshot, ...]:
        return tuple(channel.snapshot() for channel in self.gpu_channels)


@dataclass(frozen=True)
class NcclCollectiveResult:
    """Structural result of one zero-time collective call."""

    operation_id: str
    route: NcclRoute
    plan: NcclCollectivePlan
    events: tuple[NcclStackEvent, ...]
    channel_snapshots: tuple[NcclChannelSnapshot, ...]
    completion_event: NcclStackEvent


def ncclCommInitRank(
    stack: NcclStack,
    *,
    nranks: int,
    communicator_id: str,
    rank: int,
) -> NcclCommunicator:
    """Create one rank-local communicator with the real API's call shape."""

    if not isinstance(stack, NcclStack):
        raise TypeError("stack must be an NcclStack")
    nranks = _require_positive_integer("nranks", nranks)
    if nranks < 2:
        raise ValueError("nranks must be at least 2 for a ring communicator")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < nranks:
        raise ValueError("rank must be an integer in [0, nranks)")
    communicator_id = _string(communicator_id, "communicator_id")
    stack._observer.call(
        "ncclCommInitRank",
        NcclStackLane.CPU,
        rank=rank,
        communicator_id=communicator_id,
        value=nranks,
    )
    logical_channels = stack.planner.build_logical_channels(
        rank=rank,
        world_size=nranks,
        communicator_id=communicator_id,
    )
    gpu_channels = tuple(
        NcclGpuChannel(
            channel,
            stack.config.fifo_slots_per_channel,
            stack._observer,
        )
        for channel in logical_channels
    )
    return NcclCommunicator(
        stack=stack,
        communicator_id=communicator_id,
        rank=rank,
        world_size=nranks,
        logical_channels=logical_channels,
        gpu_channels=gpu_channels,
    )


def ncclAllReduce(
    communicator: NcclCommunicator,
    *,
    payload_bytes: int,
    operation_id: str,
    route: NcclRoute,
) -> NcclCollectiveResult:
    """Plan and execute one name-mirrored, zero-time ring all-reduce."""

    if not isinstance(communicator, NcclCommunicator):
        raise TypeError("communicator must be an NcclCommunicator")
    payload_bytes = _require_positive_integer("payload_bytes", payload_bytes)
    if payload_bytes % communicator.world_size:
        raise ValueError(
            "payload_bytes must divide evenly by communicator world_size "
            "for an exact ring plan"
        )
    operation_id = _string(operation_id, "operation_id")
    if not isinstance(route, NcclRoute):
        raise TypeError("route must be an NcclRoute")
    start_event = len(communicator.stack.events)
    communicator.stack._observer.call(
        "ncclAllReduce",
        NcclStackLane.CPU,
        rank=communicator.rank,
        communicator_id=communicator.communicator_id,
        operation_id=operation_id,
        value=payload_bytes,
    )
    plan = communicator.stack.planner.plan_all_reduce(
        communicator,
        payload_bytes=payload_bytes,
        operation_id=operation_id,
    )
    completion = communicator.stack.kernel.launch(
        communicator,
        operation_id=operation_id,
        route=route,
        plan=plan,
    )
    events = communicator.stack.events[start_event:]
    validate_nccl_stack_events(communicator.stack.events)
    return NcclCollectiveResult(
        operation_id=operation_id,
        route=route,
        plan=plan,
        events=events,
        channel_snapshots=communicator.snapshots(),
        completion_event=completion,
    )


__all__ = [
    "NCCL_STACK_EVENT_SCHEMA",
    "Ibverbs",
    "IbverbsRequest",
    "NcclChannelSnapshot",
    "NcclChunk",
    "NcclCollectivePlan",
    "NcclCollectiveResult",
    "NcclCommunicator",
    "NcclDataFifoSlot",
    "NcclGpuChannel",
    "NcclLogicalChannel",
    "NcclNetPlugin",
    "NcclNetRequest",
    "NcclProxyProgressEngine",
    "NcclRoute",
    "NcclStack",
    "NcclStackConfig",
    "NcclStackEvent",
    "NcclStackEventKind",
    "NcclStackLane",
    "NcclTrafficPlanner",
    "ncclAllReduce",
    "ncclCommInitRank",
    "nccl_stack_event_from_json",
    "nccl_stack_event_to_json",
    "nccl_stack_events_from_json",
    "nccl_stack_events_to_json",
    "validate_nccl_stack_events",
]
