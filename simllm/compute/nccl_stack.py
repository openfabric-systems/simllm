"""Name-mirrored zero-time skeleton of the NCCL communication stack.

The caller supplies the only :class:`VirtualClock`. Every modeled call,
producer store, and successful consumer poll reads that clock without advancing
it. The resulting strictly versioned events are stack-internal observations,
not completion events and not a timing model.

The inter-node route follows the CPU-host proxy shape. The GPU may fill its
send FIFO before an independently invoked proxy progression loop consumes the
published work. The GPU advances send ``tail``; the proxy advances send
``head`` only after a separately injected network completion is observed. The
intra-node route stays in the collective kernel and never enters proxy, net,
verbs, doorbell, or network-completion code.

The audited NCCL name mapping is recorded in ``docs/modules/compute.md``. It is
based on NVIDIA NCCL release ``v2.30.7-1``. This module mirrors names and causal
shape only and contains no copied NCCL implementation.
"""

from __future__ import annotations

import enum
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
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

    Ring neighbors have dedicated fields. ``value`` is reserved for the
    observed counter, flag, byte count, or structural count and never encodes a
    peer. A successful poll identifies its producer by sequence.
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
    slot_id: int | None = None
    send_peer_rank: int | None = None
    receive_peer_rank: int | None = None
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
    _optional_integer(event.slot_id, f"{path}.slot_id", nonnegative=True)
    _optional_integer(event.send_peer_rank, f"{path}.send_peer_rank", nonnegative=True)
    _optional_integer(
        event.receive_peer_rank,
        f"{path}.receive_peer_rank",
        nonnegative=True,
    )
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


_EVENT_FIELDS = {
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
    "slot_id",
    "send_peer_rank",
    "receive_peer_rank",
    "subject",
    "value",
    "observed_signal_sequence",
}


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
        "slot_id": event.slot_id,
        "send_peer_rank": event.send_peer_rank,
        "receive_peer_rank": event.receive_peer_rank,
        "subject": event.subject,
        "value": event.value,
        "observed_signal_sequence": event.observed_signal_sequence,
    }


def nccl_stack_event_from_json(value: Any) -> NcclStackEvent:
    """Parse one strict simllm-nccl-stack-event-v1 object."""

    payload = _object(value, "event")
    _fields(payload, "event", required=_EVENT_FIELDS)
    schema = _string(payload["schema"], "event.schema")
    if schema != NCCL_STACK_EVENT_SCHEMA:
        raise ValueError(
            "event.schema: unsupported schema "
            f"{schema!r}; expected {NCCL_STACK_EVENT_SCHEMA!r}"
        )
    return NcclStackEvent(
        sequence=_integer(payload["sequence"], "event.sequence", nonnegative=True),
        timestamp_ps=_integer(
            payload["timestamp_ps"],
            "event.timestamp_ps",
            nonnegative=True,
        ),
        kind=_enum_value(NcclStackEventKind, payload["kind"], "event.kind"),
        lane=_enum_value(NcclStackLane, payload["lane"], "event.lane"),
        function=_string(payload["function"], "event.function"),
        rank=_integer(payload["rank"], "event.rank", nonnegative=True),
        communicator_id=_string(payload["communicator_id"], "event.communicator_id"),
        operation_id=_optional_string(payload["operation_id"], "event.operation_id"),
        channel_id=_optional_integer(
            payload["channel_id"],
            "event.channel_id",
            nonnegative=True,
        ),
        chunk_id=_optional_integer(payload["chunk_id"], "event.chunk_id", nonnegative=True),
        slot_id=_optional_integer(payload["slot_id"], "event.slot_id", nonnegative=True),
        send_peer_rank=_optional_integer(
            payload["send_peer_rank"],
            "event.send_peer_rank",
            nonnegative=True,
        ),
        receive_peer_rank=_optional_integer(
            payload["receive_peer_rank"],
            "event.receive_peer_rank",
            nonnegative=True,
        ),
        subject=_optional_string(payload["subject"], "event.subject"),
        value=_optional_integer(payload["value"], "event.value", nonnegative=True),
        observed_signal_sequence=_optional_integer(
            payload["observed_signal_sequence"],
            "event.observed_signal_sequence",
            nonnegative=True,
        ),
    )


def _poll_identity_mismatches(
    event: NcclStackEvent,
    producer: NcclStackEvent,
) -> list[str]:
    identity_fields = (
        "rank",
        "communicator_id",
        "operation_id",
        "channel_id",
        "slot_id",
        "send_peer_rank",
        "receive_peer_rank",
        "subject",
        "value",
    )
    mismatched = [
        name for name in identity_fields if getattr(event, name) != getattr(producer, name)
    ]
    if event.subject == "head_counter":
        if (
            event.chunk_id is None
            or producer.chunk_id is None
            or event.chunk_id <= producer.chunk_id
        ):
            mismatched.append("chunk_id")
    elif event.chunk_id != producer.chunk_id:
        mismatched.append("chunk_id")
    return mismatched


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
        mismatched = _poll_identity_mismatches(event, producer)
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

    events = tuple(nccl_stack_event_from_json(item) for item in _array(value, "events"))
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
        slot_id: int | None = None,
        send_peer_rank: int | None = None,
        receive_peer_rank: int | None = None,
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
            slot_id=slot_id,
            send_peer_rank=send_peer_rank,
            receive_peer_rank=receive_peer_rank,
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
        slot_id: int | None = None,
        send_peer_rank: int | None = None,
        receive_peer_rank: int | None = None,
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
            slot_id=slot_id,
            send_peer_rank=send_peer_rank,
            receive_peer_rank=receive_peer_rank,
            subject=subject,
            value=value,
        )

    def poll(
        self,
        function: str,
        lane: NcclStackLane,
        producer: NcclStackEvent,
        *,
        chunk_id: int | None = None,
    ) -> NcclStackEvent:
        sequence = producer.sequence
        if sequence >= len(self._events) or self._events[sequence] is not producer:
            raise ValueError("a poll producer must belong to this stack observer")
        if producer.kind is not NcclStackEventKind.SIGNAL_STORE:
            raise ValueError("a poll producer must be a signal-store event")
        observed_chunk = producer.chunk_id if chunk_id is None else chunk_id
        event = self._make_event(
            kind=NcclStackEventKind.POLL_OBSERVES,
            lane=lane,
            function=function,
            rank=producer.rank,
            communicator_id=producer.communicator_id,
            operation_id=producer.operation_id,
            channel_id=producer.channel_id,
            chunk_id=observed_chunk,
            slot_id=producer.slot_id,
            send_peer_rank=producer.send_peer_rank,
            receive_peer_rank=producer.receive_peer_rank,
            subject=producer.subject,
            value=producer.value,
            observed_signal_sequence=producer.sequence,
        )
        mismatched = _poll_identity_mismatches(event, producer)
        if mismatched:
            raise ValueError(f"poll and producer disagree on {mismatched}")
        self._events.append(event)
        return event

    def _make_event(
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
        slot_id: int | None,
        send_peer_rank: int | None,
        receive_peer_rank: int | None,
        subject: str | None,
        value: int | None,
        observed_signal_sequence: int | None = None,
    ) -> NcclStackEvent:
        timestamp_ps = _integer(self._clock.now_ps, "clock.now_ps", nonnegative=True)
        if self._events and timestamp_ps < self._events[-1].timestamp_ps:
            raise ValueError("the caller-supplied virtual clock moved backward")
        return NcclStackEvent(
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
            slot_id=slot_id,
            send_peer_rank=send_peer_rank,
            receive_peer_rank=receive_peer_rank,
            subject=subject,
            value=value,
            observed_signal_sequence=observed_signal_sequence,
        )

    def _emit(self, **fields: Any) -> NcclStackEvent:
        event = self._make_event(**fields)
        self._events.append(event)
        return event


def _require_positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class NcclStackConfig:
    """Structural ring, GPU-lane, and FIFO settings."""

    channel_count: int = 1
    chunk_bytes: int = 1 << 20
    fifo_slots_per_channel: int = 1
    warps_per_channel: int = 1

    def __post_init__(self) -> None:
        _require_positive_integer("channel_count", self.channel_count)
        _require_positive_integer("chunk_bytes", self.chunk_bytes)
        _require_positive_integer("fifo_slots_per_channel", self.fifo_slots_per_channel)
        _require_positive_integer("warps_per_channel", self.warps_per_channel)


@dataclass(frozen=True)
class _RingLayout:
    payload_bytes: int
    wire_bytes: int
    step_count: int
    step_bytes: int
    lane_bytes: int
    chunks_per_lane_per_step: int
    total_chunks: int


def _validated_ring_layout(
    *,
    payload_bytes: object,
    world_size: int,
    config: NcclStackConfig,
) -> _RingLayout:
    payload = _require_positive_integer("payload_bytes", payload_bytes)
    lanes = world_size * config.channel_count * config.warps_per_channel
    lane_bytes, lane_remainder = divmod(payload, lanes)
    if lane_remainder:
        raise ValueError(
            "payload_bytes must divide evenly over world_size * channel_count * "
            f"warps_per_channel; {payload} bytes over {lanes} shares leaves "
            f"{lane_remainder}"
        )
    chunks_per_lane, chunk_remainder = divmod(lane_bytes, config.chunk_bytes)
    if chunk_remainder:
        raise ValueError(
            "per-warp step bytes must divide evenly into chunk_bytes; "
            f"{lane_bytes} bytes into {config.chunk_bytes}-byte chunks leaves "
            f"{chunk_remainder}"
        )
    if chunks_per_lane == 0:
        raise ValueError("chunk_bytes exceeds the per-warp step share")
    step_count = 2 * (world_size - 1)
    total_chunks = (
        step_count
        * config.channel_count
        * config.warps_per_channel
        * chunks_per_lane
    )
    return _RingLayout(
        payload_bytes=payload,
        wire_bytes=nccl_ring_egress_bytes(
            payload_bytes=payload,
            world_size=world_size,
        ),
        step_count=step_count,
        step_bytes=payload // world_size,
        lane_bytes=lane_bytes,
        chunks_per_lane_per_step=chunks_per_lane,
        total_chunks=total_chunks,
    )


@dataclass(frozen=True)
class _NcclLogicalChannel:
    channel_id: int
    send_peer: int
    receive_peer: int


@dataclass(frozen=True)
class _NcclChunk:
    chunk_id: int
    ring_step: int
    phase: str
    channel_id: int
    warp_id: int
    lane_chunk_index: int
    offset_bytes: int
    byte_count: int


@dataclass(frozen=True)
class _NcclCollectivePlan:
    payload_bytes: int
    wire_bytes: int
    step_count: int
    step_bytes: int
    lane_bytes: int
    chunk_bytes: int
    channel_count: int
    warps_per_channel: int
    chunks: tuple[_NcclChunk, ...]

    def chunks_per_channel(self) -> tuple[int, ...]:
        counts = [0] * self.channel_count
        for chunk in self.chunks:
            counts[chunk.channel_id] += 1
        return tuple(counts)

    def chunks_per_step(self) -> tuple[int, ...]:
        counts = [0] * self.step_count
        for chunk in self.chunks:
            counts[chunk.ring_step] += 1
        return tuple(counts)


class _NcclTrafficPlanner:
    def __init__(self, config: NcclStackConfig, observer: _NcclStackObserver):
        self._config = config
        self._observer = observer

    def build_rings(
        self,
        *,
        rank: int,
        world_size: int,
        communicator_id: str,
    ) -> tuple[_NcclLogicalChannel, ...]:
        send_peer = (rank + 1) % world_size
        receive_peer = (rank - 1) % world_size
        self._observer.call(
            "ncclBuildRings",
            NcclStackLane.CPU,
            rank=rank,
            communicator_id=communicator_id,
            value=self._config.channel_count,
        )
        channels = []
        for channel_id in range(self._config.channel_count):
            self._observer.call(
                "initChannel",
                NcclStackLane.CPU,
                rank=rank,
                communicator_id=communicator_id,
                channel_id=channel_id,
                send_peer_rank=send_peer,
                receive_peer_rank=receive_peer,
                subject="logical_channel",
            )
            channels.append(_NcclLogicalChannel(channel_id, send_peer, receive_peer))
        return tuple(channels)

    def plan_all_reduce(
        self,
        communicator: _NcclCommunicator,
        *,
        operation_id: str,
        layout: _RingLayout,
    ) -> _NcclCollectivePlan:
        chunks = self._construct_chunks(communicator.world_size, layout)
        self._observer.call(
            "ncclEnqueueCheck",
            NcclStackLane.CPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            value=layout.payload_bytes,
        )
        self._observer.call(
            "scheduleCollTasksToPlan",
            NcclStackLane.CPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            value=layout.step_count,
        )
        self._observer.call(
            "calcCollChunking",
            NcclStackLane.CPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            value=len(chunks),
        )
        return _NcclCollectivePlan(
            payload_bytes=layout.payload_bytes,
            wire_bytes=layout.wire_bytes,
            step_count=layout.step_count,
            step_bytes=layout.step_bytes,
            lane_bytes=layout.lane_bytes,
            chunk_bytes=self._config.chunk_bytes,
            channel_count=self._config.channel_count,
            warps_per_channel=self._config.warps_per_channel,
            chunks=chunks,
        )

    def _construct_chunks(
        self,
        world_size: int,
        layout: _RingLayout,
    ) -> tuple[_NcclChunk, ...]:
        chunks = []
        chunk_id = 0
        for ring_step in range(layout.step_count):
            phase = "reduce_scatter" if ring_step < world_size - 1 else "all_gather"
            for channel_id in range(self._config.channel_count):
                for warp_id in range(self._config.warps_per_channel):
                    for lane_chunk_index in range(layout.chunks_per_lane_per_step):
                        chunks.append(
                            _NcclChunk(
                                chunk_id=chunk_id,
                                ring_step=ring_step,
                                phase=phase,
                                channel_id=channel_id,
                                warp_id=warp_id,
                                lane_chunk_index=lane_chunk_index,
                                offset_bytes=chunk_id * self._config.chunk_bytes,
                                byte_count=self._config.chunk_bytes,
                            )
                        )
                        chunk_id += 1
        if chunk_id != layout.total_chunks:
            raise RuntimeError("ring planner chunk count disagrees with validated layout")
        return tuple(chunks)


@dataclass
class _NcclDataFifoSlot:
    slot_id: int
    chunk_id: int | None = None
    byte_count: int = 0
    ready: bool = False
    head_signal: NcclStackEvent | None = None


@dataclass(frozen=True)
class _NcclChannelSnapshot:
    channel_id: int
    head: int
    tail: int
    high_watermark: int
    ready_flags: tuple[bool, ...]
    slot_chunk_ids: tuple[int | None, ...]
    slot_byte_counts: tuple[int, ...]


@dataclass(frozen=True)
class _PublishedChunk:
    chunk: _NcclChunk
    slot_id: int
    ready_signal: NcclStackEvent
    tail_signal: NcclStackEvent


class _NcclGpuChannel:
    """Mutable authority for one send connector and its FIFO slots."""

    def __init__(
        self,
        logical_channel: _NcclLogicalChannel,
        fifo_slots: int,
        observer: _NcclStackObserver,
    ):
        self.logical_channel = logical_channel
        self.head = 0
        self.tail = 0
        self.high_watermark = 0
        self.slots = [_NcclDataFifoSlot(index) for index in range(fifo_slots)]
        self._observer = observer

    @property
    def has_credit(self) -> bool:
        return self.tail - self.head < len(self.slots)

    def publish(
        self,
        communicator: _NcclCommunicator,
        *,
        operation_id: str,
        chunk: _NcclChunk,
    ) -> _PublishedChunk:
        if chunk.channel_id != self.logical_channel.channel_id:
            raise ValueError("chunk assigned to the wrong GPU channel")
        if not self.has_credit:
            raise RuntimeError("NCCL send FIFO is full")
        slot = self.slots[self.tail % len(self.slots)]
        if slot.ready:
            raise RuntimeError("NCCL send FIFO slot is still ready")
        if self.tail >= len(self.slots):
            if slot.head_signal is None:
                raise RuntimeError("reused NCCL send FIFO slot has no head credit")
            self._observer.poll(
                "waitPeer",
                NcclStackLane.GPU,
                slot.head_signal,
                chunk_id=chunk.chunk_id,
            )
        slot.chunk_id = chunk.chunk_id
        slot.byte_count = chunk.byte_count
        slot.ready = True
        ready_signal = self._observer.signal(
            "waitPeer",
            NcclStackLane.GPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            channel_id=chunk.channel_id,
            chunk_id=chunk.chunk_id,
            slot_id=slot.slot_id,
            send_peer_rank=self.logical_channel.send_peer,
            subject="ready_flag",
            value=1,
        )
        self._observer.call(
            "genericOp",
            NcclStackLane.GPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            channel_id=chunk.channel_id,
            chunk_id=chunk.chunk_id,
            slot_id=slot.slot_id,
            send_peer_rank=self.logical_channel.send_peer,
            subject="data_fifo_slot",
            value=chunk.byte_count,
        )
        self.tail += 1
        self.high_watermark = max(self.high_watermark, self.tail - self.head)
        tail_signal = self._observer.signal(
            "postPeer",
            NcclStackLane.GPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            channel_id=chunk.channel_id,
            chunk_id=chunk.chunk_id,
            slot_id=slot.slot_id,
            send_peer_rank=self.logical_channel.send_peer,
            subject="tail_counter",
            value=self.tail,
        )
        return _PublishedChunk(chunk, slot.slot_id, ready_signal, tail_signal)

    def complete(
        self,
        communicator: _NcclCommunicator,
        *,
        operation_id: str,
        published: _PublishedChunk,
    ) -> None:
        slot = self.slots[published.slot_id]
        if not slot.ready or slot.chunk_id != published.chunk.chunk_id:
            raise RuntimeError("proxy completion does not match the ready FIFO slot")
        if self.head >= self.tail:
            raise RuntimeError("send connector head cannot advance beyond tail")
        slot.ready = False
        slot.chunk_id = None
        slot.byte_count = 0
        self._observer.signal(
            "sendProxyProgress",
            NcclStackLane.CPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            channel_id=published.chunk.channel_id,
            chunk_id=published.chunk.chunk_id,
            slot_id=published.slot_id,
            send_peer_rank=self.logical_channel.send_peer,
            subject="ready_flag",
            value=0,
        )
        self.head += 1
        slot.head_signal = self._observer.signal(
            "sendProxyProgress",
            NcclStackLane.CPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            channel_id=published.chunk.channel_id,
            chunk_id=published.chunk.chunk_id,
            slot_id=published.slot_id,
            send_peer_rank=self.logical_channel.send_peer,
            subject="head_counter",
            value=self.head,
        )

    def snapshot(self) -> _NcclChannelSnapshot:
        return _NcclChannelSnapshot(
            channel_id=self.logical_channel.channel_id,
            head=self.head,
            tail=self.tail,
            high_watermark=self.high_watermark,
            ready_flags=tuple(slot.ready for slot in self.slots),
            slot_chunk_ids=tuple(slot.chunk_id for slot in self.slots),
            slot_byte_counts=tuple(slot.byte_count for slot in self.slots),
        )


@dataclass
class _IbverbsRequest:
    request_id: str
    communicator: _NcclCommunicator
    operation_id: str
    published: _PublishedChunk
    send_peer_rank: int
    doorbell_signal: NcclStackEvent
    completion_signal: NcclStackEvent | None = None
    polled: bool = False


class _NetworkCompletionSource:
    """Fake external completion source for the deliberate zero-time slice."""

    def __init__(self, observer: _NcclStackObserver):
        self._observer = observer
        self._posted: dict[str, _IbverbsRequest] = {}

    def register(self, request: _IbverbsRequest) -> None:
        if request.request_id in self._posted:
            raise RuntimeError(f"duplicate verbs request {request.request_id!r}")
        self._posted[request.request_id] = request

    def complete(self, request: _IbverbsRequest) -> NcclStackEvent:
        if self._posted.get(request.request_id) is not request:
            raise RuntimeError("network completion does not match a posted request")
        if request.completion_signal is not None:
            raise RuntimeError(f"verbs request {request.request_id!r} already completed")
        chunk = request.published.chunk
        signal = self._observer.signal(
            "simllmNetworkComplete",
            NcclStackLane.RNIC,
            rank=request.communicator.rank,
            communicator_id=request.communicator.communicator_id,
            operation_id=request.operation_id,
            channel_id=chunk.channel_id,
            chunk_id=chunk.chunk_id,
            slot_id=request.published.slot_id,
            send_peer_rank=request.send_peer_rank,
            subject="completion_queue_entry",
            value=chunk.byte_count,
        )
        request.completion_signal = signal
        return signal

    def poll(self, request: _IbverbsRequest) -> NcclStackEvent | None:
        if self._posted.get(request.request_id) is not request:
            raise RuntimeError("completion poll does not match a posted request")
        return request.completion_signal

    def retire(self, request: _IbverbsRequest) -> None:
        if self._posted.pop(request.request_id, None) is not request:
            raise RuntimeError("completion retirement does not match a posted request")


class _Ibverbs:
    def __init__(
        self,
        observer: _NcclStackObserver,
        completion_source: _NetworkCompletionSource,
    ):
        self._observer = observer
        self._completion_source = completion_source

    def post_send(
        self,
        communicator: _NcclCommunicator,
        *,
        operation_id: str,
        published: _PublishedChunk,
        send_peer_rank: int,
    ) -> _IbverbsRequest:
        chunk = published.chunk
        self._observer.call(
            "wrap_ibv_post_send",
            NcclStackLane.CPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            channel_id=chunk.channel_id,
            chunk_id=chunk.chunk_id,
            slot_id=published.slot_id,
            send_peer_rank=send_peer_rank,
            value=chunk.byte_count,
        )
        doorbell = self._observer.signal(
            "simllmRnicRingDoorbell",
            NcclStackLane.RNIC,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            channel_id=chunk.channel_id,
            chunk_id=chunk.chunk_id,
            slot_id=published.slot_id,
            send_peer_rank=send_peer_rank,
            subject="doorbell",
            value=1,
        )
        request = _IbverbsRequest(
            request_id=(
                f"{operation_id}:c{chunk.channel_id}:k{chunk.chunk_id}:send"
            ),
            communicator=communicator,
            operation_id=operation_id,
            published=published,
            send_peer_rank=send_peer_rank,
            doorbell_signal=doorbell,
        )
        self._completion_source.register(request)
        return request

    def poll_cq(self, request: _IbverbsRequest) -> bool:
        if request.polled:
            raise RuntimeError(f"verbs request {request.request_id!r} was already polled")
        completion = self._completion_source.poll(request)
        if completion is None:
            return False
        self._observer.poll("wrap_ibv_poll_cq", NcclStackLane.CPU, completion)
        request.polled = True
        self._completion_source.retire(request)
        return True


@dataclass(frozen=True)
class _NcclNetRequest:
    request_id: str
    verbs_request: _IbverbsRequest


class _NcclNetPlugin:
    def __init__(self, observer: _NcclStackObserver, verbs: _Ibverbs):
        self._observer = observer
        self._verbs = verbs

    def isend(
        self,
        communicator: _NcclCommunicator,
        *,
        operation_id: str,
        published: _PublishedChunk,
        send_peer_rank: int,
    ) -> _NcclNetRequest:
        chunk = published.chunk
        self._observer.call(
            "ncclNet.isend",
            NcclStackLane.CPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            channel_id=chunk.channel_id,
            chunk_id=chunk.chunk_id,
            slot_id=published.slot_id,
            send_peer_rank=send_peer_rank,
            value=chunk.byte_count,
        )
        request = self._verbs.post_send(
            communicator,
            operation_id=operation_id,
            published=published,
            send_peer_rank=send_peer_rank,
        )
        return _NcclNetRequest(request.request_id, request)

    def test(self, request: _NcclNetRequest) -> bool:
        verbs_request = request.verbs_request
        chunk = verbs_request.published.chunk
        self._observer.call(
            "ncclNet.test",
            NcclStackLane.CPU,
            rank=verbs_request.communicator.rank,
            communicator_id=verbs_request.communicator.communicator_id,
            operation_id=verbs_request.operation_id,
            channel_id=chunk.channel_id,
            chunk_id=chunk.chunk_id,
            slot_id=verbs_request.published.slot_id,
            send_peer_rank=verbs_request.send_peer_rank,
            value=chunk.byte_count,
        )
        return self._verbs.poll_cq(verbs_request)


@dataclass(frozen=True)
class _InflightSend:
    published: _PublishedChunk
    request: _NcclNetRequest


@dataclass
class _ProxyOperation:
    communicator: _NcclCommunicator
    operation_id: str
    total_chunks: int
    published: deque[_PublishedChunk] = field(default_factory=deque)
    inflight: deque[_InflightSend] = field(default_factory=deque)
    completed_chunks: int = 0


class _NcclProxyProgressEngine:
    """Default CPU-host proxy queue and independent progression loop."""

    def __init__(self, observer: _NcclStackObserver, net: _NcclNetPlugin):
        self._observer = observer
        self._net = net

    def save_op(
        self,
        communicator: _NcclCommunicator,
        *,
        operation_id: str,
        total_chunks: int,
    ) -> _ProxyOperation:
        self._observer.call(
            "ncclProxySaveOp",
            NcclStackLane.CPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            send_peer_rank=communicator.logical_channels[0].send_peer,
            subject="proxy_operation",
            value=total_chunks,
        )
        return _ProxyOperation(communicator, operation_id, total_chunks)

    def progress(self, operation: _ProxyOperation) -> int:
        ready_completions = sum(
            item.request.verbs_request.completion_signal is not None
            for item in operation.inflight
        )
        if not operation.published and ready_completions == 0:
            return 0
        communicator = operation.communicator
        send_peer = communicator.logical_channels[0].send_peer
        self._observer.call(
            "ncclProxyProgress",
            NcclStackLane.CPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation.operation_id,
            send_peer_rank=send_peer,
        )
        self._observer.call(
            "sendProxyProgress",
            NcclStackLane.CPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation.operation_id,
            send_peer_rank=send_peer,
        )
        actions = 0
        while operation.published:
            published = operation.published.popleft()
            channel = communicator.gpu_channel(published.chunk.channel_id)
            self._observer.poll(
                "sendProxyProgress",
                NcclStackLane.CPU,
                published.tail_signal,
            )
            self._observer.poll(
                "sendProxyProgress",
                NcclStackLane.CPU,
                published.ready_signal,
            )
            request = self._net.isend(
                communicator,
                operation_id=operation.operation_id,
                published=published,
                send_peer_rank=channel.logical_channel.send_peer,
            )
            operation.inflight.append(_InflightSend(published, request))
            actions += 1
        while operation.inflight:
            inflight = operation.inflight[0]
            if inflight.request.verbs_request.completion_signal is None:
                break
            if not self._net.test(inflight.request):
                break
            operation.inflight.popleft()
            channel = communicator.gpu_channel(inflight.published.chunk.channel_id)
            channel.complete(
                communicator,
                operation_id=operation.operation_id,
                published=inflight.published,
            )
            operation.completed_chunks += 1
            actions += 1
        return actions


class _KernelExecution:
    def __init__(
        self,
        observer: _NcclStackObserver,
        communicator: _NcclCommunicator,
        operation_id: str,
        plan: _NcclCollectivePlan,
        proxy_operation: _ProxyOperation | None,
    ):
        self._observer = observer
        self._communicator = communicator
        self._operation_id = operation_id
        self._plan = plan
        self._proxy_operation = proxy_operation
        self._next_chunk = 0

    @property
    def done(self) -> bool:
        return self._next_chunk == len(self._plan.chunks)

    def progress(self, route: NcclRoute) -> int:
        produced = 0
        while self._next_chunk < len(self._plan.chunks):
            chunk = self._plan.chunks[self._next_chunk]
            channel = self._communicator.gpu_channel(chunk.channel_id)
            if route is NcclRoute.INTER_NODE and not channel.has_credit:
                break
            if route is NcclRoute.INTRA_NODE:
                self._observer.call(
                    "genericOp",
                    NcclStackLane.GPU,
                    rank=self._communicator.rank,
                    communicator_id=self._communicator.communicator_id,
                    operation_id=self._operation_id,
                    channel_id=chunk.channel_id,
                    chunk_id=chunk.chunk_id,
                    send_peer_rank=channel.logical_channel.send_peer,
                    subject="nvlink",
                    value=chunk.byte_count,
                )
            else:
                if self._proxy_operation is None:
                    raise RuntimeError("inter-node kernel has no saved proxy operation")
                published = channel.publish(
                    self._communicator,
                    operation_id=self._operation_id,
                    chunk=chunk,
                )
                self._proxy_operation.published.append(published)
            self._next_chunk += 1
            produced += 1
        return produced


class _NcclCollectiveKernel:
    def __init__(self, observer: _NcclStackObserver):
        self._observer = observer

    def launch(
        self,
        communicator: _NcclCommunicator,
        *,
        operation_id: str,
        plan: _NcclCollectivePlan,
        proxy_operation: _ProxyOperation | None,
    ) -> _KernelExecution:
        self._observer.call(
            "ncclLaunchKernel",
            NcclStackLane.CPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            value=len(plan.chunks),
        )
        self._observer.call(
            "ncclKernelMain",
            NcclStackLane.GPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            value=len(plan.chunks),
        )
        self._observer.call(
            "runRing",
            NcclStackLane.GPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            value=plan.step_count,
        )
        return _KernelExecution(
            self._observer,
            communicator,
            operation_id,
            plan,
            proxy_operation,
        )

    def complete(
        self,
        communicator: _NcclCommunicator,
        *,
        operation_id: str,
        chunk_count: int,
    ) -> NcclStackEvent:
        return self._observer.signal(
            "simllmKernelComplete",
            NcclStackLane.GPU,
            rank=communicator.rank,
            communicator_id=communicator.communicator_id,
            operation_id=operation_id,
            subject="kernel_completion",
            value=chunk_count,
        )


class NcclStack:
    """One composed skeleton whose caller owns the virtual clock."""

    def __init__(self, *, clock: VirtualClock, config: NcclStackConfig | None = None):
        if not isinstance(clock, VirtualClock):
            raise TypeError("clock must be a simllm.core.VirtualClock")
        self.clock = clock
        self.config = config or NcclStackConfig()
        self._observer = _NcclStackObserver(clock)
        self._planner = _NcclTrafficPlanner(self.config, self._observer)
        self._completion_source = _NetworkCompletionSource(self._observer)
        self._ibverbs = _Ibverbs(self._observer, self._completion_source)
        self._nccl_net = _NcclNetPlugin(self._observer, self._ibverbs)
        self._proxy = _NcclProxyProgressEngine(self._observer, self._nccl_net)
        self._kernel = _NcclCollectiveKernel(self._observer)

    @property
    def events(self) -> tuple[NcclStackEvent, ...]:
        return self._observer.events


@dataclass
class _NcclCommunicator:
    stack: NcclStack
    communicator_id: str
    rank: int
    world_size: int
    logical_channels: tuple[_NcclLogicalChannel, ...]
    gpu_channels: tuple[_NcclGpuChannel, ...]

    def gpu_channel(self, channel_id: int) -> _NcclGpuChannel:
        if channel_id < 0 or channel_id >= len(self.gpu_channels):
            raise ValueError(f"unknown NCCL channel {channel_id}")
        channel = self.gpu_channels[channel_id]
        if channel.logical_channel.channel_id != channel_id:
            raise RuntimeError("GPU channel order no longer matches channel identity")
        return channel

    def snapshots(self) -> tuple[_NcclChannelSnapshot, ...]:
        return tuple(channel.snapshot() for channel in self.gpu_channels)


@dataclass(frozen=True)
class NcclCollectiveResult:
    """Structural result of one zero-time collective call."""

    operation_id: str
    route: NcclRoute
    plan: _NcclCollectivePlan
    events: tuple[NcclStackEvent, ...]
    channel_snapshots: tuple[_NcclChannelSnapshot, ...]
    completion_event: NcclStackEvent


def ncclCommInitRank(
    stack: NcclStack,
    *,
    nranks: int,
    communicator_id: str,
    rank: int,
) -> _NcclCommunicator:
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
    logical_channels = stack._planner.build_rings(
        rank=rank,
        world_size=nranks,
        communicator_id=communicator_id,
    )
    gpu_channels = tuple(
        _NcclGpuChannel(
            channel,
            stack.config.fifo_slots_per_channel,
            stack._observer,
        )
        for channel in logical_channels
    )
    return _NcclCommunicator(
        stack=stack,
        communicator_id=communicator_id,
        rank=rank,
        world_size=nranks,
        logical_channels=logical_channels,
        gpu_channels=gpu_channels,
    )


def _run_inter_node(
    communicator: _NcclCommunicator,
    *,
    operation_id: str,
    plan: _NcclCollectivePlan,
) -> NcclStackEvent:
    stack = communicator.stack
    proxy_operation = stack._proxy.save_op(
        communicator,
        operation_id=operation_id,
        total_chunks=len(plan.chunks),
    )
    execution = stack._kernel.launch(
        communicator,
        operation_id=operation_id,
        plan=plan,
        proxy_operation=proxy_operation,
    )
    while not execution.done:
        if execution.progress(NcclRoute.INTER_NODE) == 0:
            raise RuntimeError("kernel could not publish despite a drained proxy batch")
        if stack._proxy.progress(proxy_operation) == 0:
            raise RuntimeError("proxy did not consume a published kernel batch")
        requests = tuple(item.request.verbs_request for item in proxy_operation.inflight)
        if not requests:
            raise RuntimeError("proxy submission batch created no network requests")
        for request in requests:
            stack._completion_source.complete(request)
        if stack._proxy.progress(proxy_operation) == 0:
            raise RuntimeError("proxy did not consume a completed network batch")
    if proxy_operation.published or proxy_operation.inflight:
        raise RuntimeError("kernel completed with undrained proxy work")
    if proxy_operation.completed_chunks != len(plan.chunks):
        raise RuntimeError("proxy completion count disagrees with the plan")
    return stack._kernel.complete(
        communicator,
        operation_id=operation_id,
        chunk_count=len(plan.chunks),
    )


def _run_intra_node(
    communicator: _NcclCommunicator,
    *,
    operation_id: str,
    plan: _NcclCollectivePlan,
) -> NcclStackEvent:
    stack = communicator.stack
    execution = stack._kernel.launch(
        communicator,
        operation_id=operation_id,
        plan=plan,
        proxy_operation=None,
    )
    if execution.progress(NcclRoute.INTRA_NODE) != len(plan.chunks):
        raise RuntimeError("intra-node kernel did not consume the complete plan")
    return stack._kernel.complete(
        communicator,
        operation_id=operation_id,
        chunk_count=len(plan.chunks),
    )


def ncclAllReduce(
    communicator: _NcclCommunicator,
    *,
    payload_bytes: int,
    operation_id: str,
    route: NcclRoute,
) -> NcclCollectiveResult:
    """Plan and execute one name-mirrored, zero-time ring all-reduce."""

    if not isinstance(communicator, _NcclCommunicator):
        raise TypeError("communicator must be returned by ncclCommInitRank")
    operation_id = _string(operation_id, "operation_id")
    if not isinstance(route, NcclRoute):
        raise TypeError("route must be an NcclRoute")
    layout = _validated_ring_layout(
        payload_bytes=payload_bytes,
        world_size=communicator.world_size,
        config=communicator.stack.config,
    )
    start_event = len(communicator.stack.events)
    communicator.stack._observer.call(
        "ncclAllReduce",
        NcclStackLane.CPU,
        rank=communicator.rank,
        communicator_id=communicator.communicator_id,
        operation_id=operation_id,
        value=layout.payload_bytes,
    )
    plan = communicator.stack._planner.plan_all_reduce(
        communicator,
        operation_id=operation_id,
        layout=layout,
    )
    if route is NcclRoute.INTER_NODE:
        completion = _run_inter_node(
            communicator,
            operation_id=operation_id,
            plan=plan,
        )
    else:
        completion = _run_intra_node(
            communicator,
            operation_id=operation_id,
            plan=plan,
        )
    return NcclCollectiveResult(
        operation_id=operation_id,
        route=route,
        plan=plan,
        events=communicator.stack.events[start_event:],
        channel_snapshots=communicator.snapshots(),
        completion_event=completion,
    )


__all__ = [
    "NCCL_STACK_EVENT_SCHEMA",
    "NcclCollectiveResult",
    "NcclRoute",
    "NcclStack",
    "NcclStackConfig",
    "NcclStackEvent",
    "NcclStackEventKind",
    "NcclStackLane",
    "ncclAllReduce",
    "ncclCommInitRank",
    "nccl_stack_event_from_json",
    "nccl_stack_event_to_json",
    "nccl_stack_events_from_json",
    "nccl_stack_events_to_json",
    "validate_nccl_stack_events",
]
