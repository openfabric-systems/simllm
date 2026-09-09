"""Port-neutral observations of the existing version 2 packet vocabulary.

The ledger checks projections. It never grants a resource, advances simulation
time or retires a producer object. Destination visibility remains an explicit
observation of the producing runtime, separate from transport retirement.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any


class PacketPortError(ValueError):
    """A port observation violates its negotiated contract."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise PacketPortError(message)


def _integer(value: object, name: str, *, minimum: int = 0, maximum: int = 2**64 - 1) -> None:
    _check(
        type(value) is int and minimum <= value <= maximum,
        f"{name} must be an integer in [{minimum}, {maximum}]",
    )


class PacketEventKind(str, Enum):
    TX_STARTED = "packet_tx_started"
    TX_FINISHED = "packet_tx_finished"
    RX_ARRIVED = "packet_rx_arrived"
    DELIVERED = "delivered"
    DROPPED = "dropped"


PACKET_EVENT_FIELDS = (
    "attempt_token",
    "extent_token",
    "wqe_id",
    "event_kind",
    "event_time_ps",
    "extent_index",
    "packet_index",
    "transmission_attempt",
    "payload_offset_bytes",
    "payload_bytes",
    "wire_bytes",
    "packet_kind",
)
PACKET_EVENT_KEYS = frozenset(PACKET_EVENT_FIELDS)
PACKET_KINDS = frozenset(("data", "retransmission", "ack", "nak", "cnp", "pfc", "other_control"))


@dataclass(frozen=True)
class PacketPortContext:
    """Declared boundary meanings and capabilities, outside legacy wire rows.

    An empty link inventory explicitly means the observation has no physical link
    identity. A producer must not infer one from its correlation token. All times
    are integer picoseconds on the named simulation clock.
    """

    session_id: str
    port_id: str
    abi_version: int = 2
    packet_attempt_events: bool = True
    tx_start_boundary: str = "producer_issue"
    terminal_boundary: str = "transport_retirement"
    clock_id: str = "simulation"
    link_ids: tuple[str, ...] = ()
    event_order_evidence: str = "live_callbacks"
    packet_kinds: tuple[str, ...] = ("data", "retransmission", "ack", "nak", "other_control")

    def __post_init__(self) -> None:
        for name in ("session_id", "port_id", "clock_id"):
            value = getattr(self, name)
            _check(isinstance(value, str) and bool(value.strip()), f"{name} must be nonblank")
        _check(type(self.abi_version) is int and self.abi_version in (1, 2), "unsupported port ABI")
        _check(type(self.packet_attempt_events) is bool, "packet capability must be boolean")
        _check(
            self.tx_start_boundary in ("producer_issue", "link_grant"), "unknown TX-start boundary"
        )
        _check(self.terminal_boundary == "transport_retirement", "unknown terminal boundary")
        _check(
            self.event_order_evidence in ("live_callbacks", "separate_arrays"),
            "unknown event-order evidence",
        )
        _check(isinstance(self.link_ids, (list, tuple)), "link_ids must be an array")
        object.__setattr__(self, "link_ids", tuple(self.link_ids))
        _check(
            all(isinstance(x, str) and x.strip() for x in self.link_ids), "invalid link identity"
        )
        _check(len(set(self.link_ids)) == len(self.link_ids), "duplicate physical link identity")
        _check(isinstance(self.packet_kinds, (list, tuple)), "packet_kinds must be an array")
        object.__setattr__(self, "packet_kinds", tuple(self.packet_kinds))
        _check(
            bool(self.packet_kinds) and all(kind in PACKET_KINDS for kind in self.packet_kinds),
            "invalid declared packet kinds",
        )
        _check(len(set(self.packet_kinds)) == len(self.packet_kinds), "duplicate packet kind")

    def require(self, *capabilities: str) -> None:
        """Negotiate before constructing a ledger or registering any token."""

        _check(self.abi_version == 2, "packet consumer requires explicit ABI version 2")
        supported = {
            "packet_attempt_events": self.packet_attempt_events,
            "priority_flow_control": self.packet_attempt_events and "pfc" in self.packet_kinds,
            "congestion_notification": self.packet_attempt_events and "cnp" in self.packet_kinds,
        }
        for capability in capabilities:
            _check(
                supported.get(capability, False),
                f"port {self.port_id!r} does not support {capability!r}",
            )


@dataclass(frozen=True)
class PacketExtentAdmission:
    """An opaque producer extent admitted to a port, with its payload contract."""

    extent_token: int
    producer_id: int
    payload_bytes: int
    eligible_at_ps: int
    extent_index: int = 0

    def __post_init__(self) -> None:
        for name in ("extent_token", "producer_id", "payload_bytes"):
            _integer(getattr(self, name), name, minimum=1)
        _integer(self.eligible_at_ps, "eligible_at_ps")
        _integer(self.extent_index, "extent_index", maximum=2**32 - 1)


@dataclass(frozen=True)
class PacketAttemptEvent:
    """The exact native packet row, including its legacy opaque wqe_id name."""

    attempt_token: int
    extent_token: int
    wqe_id: int
    event_kind: str
    event_time_ps: int
    extent_index: int
    packet_index: int
    transmission_attempt: int
    payload_offset_bytes: int
    payload_bytes: int
    wire_bytes: int
    packet_kind: str

    def __post_init__(self) -> None:
        for name in ("attempt_token", "extent_token", "wqe_id", "wire_bytes"):
            _integer(getattr(self, name), name, minimum=1)
        for name in (
            "event_time_ps",
            "extent_index",
            "packet_index",
            "transmission_attempt",
            "payload_offset_bytes",
            "payload_bytes",
        ):
            _integer(getattr(self, name), name)
        _integer(self.extent_index, "extent_index", maximum=2**32 - 1)
        _integer(self.transmission_attempt, "transmission_attempt", maximum=2**32 - 1)
        _check(self.attempt_token != self.extent_token, "attempt token aliases its parent")
        _check(
            self.event_kind in {kind.value for kind in PacketEventKind},
            "unsupported packet event kind",
        )
        _check(self.packet_kind in PACKET_KINDS, "unsupported packet kind")
        _check(self.wire_bytes >= self.payload_bytes, "wire bytes are below payload bytes")

    @classmethod
    def from_row(cls, value: Mapping[str, Any]) -> PacketAttemptEvent:
        _check(isinstance(value, Mapping), "packet row must be an object")
        _check(set(value) == PACKET_EVENT_KEYS, "packet row does not match the ABI v2 fields")
        return cls(**value)

    def to_row(self) -> dict[str, Any]:
        # Dataclass declaration order is the native serializer's field order.
        return asdict(self)

    @property
    def geometry(self) -> tuple[object, ...]:
        return (
            self.extent_token,
            self.wqe_id,
            self.extent_index,
            self.packet_index,
            self.transmission_attempt,
            self.payload_offset_bytes,
            self.payload_bytes,
            self.wire_bytes,
            self.packet_kind,
        )


@dataclass(frozen=True)
class PacketExtentTerminal:
    extent_token: int
    producer_id: int
    kind: str
    at_ps: int

    def __post_init__(self) -> None:
        _integer(self.extent_token, "extent_token", minimum=1)
        _integer(self.producer_id, "producer_id", minimum=1)
        _integer(self.at_ps, "at_ps")
        _check(self.kind in ("delivered", "dropped"), "unsupported extent terminal")


@dataclass(frozen=True)
class PacketPortSnapshot:
    admitted_tokens: tuple[int, ...]
    retired_tokens: tuple[int, ...]
    live_extent_tokens: tuple[int, ...]
    started_attempt_tokens: tuple[int, ...]
    retired_attempt_tokens: tuple[int, ...]
    live_attempt_tokens: tuple[int, ...]
    delivered_payload_bytes: int


@dataclass(frozen=True)
class _Attempt:
    first: PacketAttemptEvent
    last_kind: str
    last_time_ps: int
    terminal: bool = False


class PacketPortLedger:
    """One correlation ledger shared by wire adapters and peer projections.

    Every rejected observation leaves this ledger unchanged. A live caller feeds
    the original callback order. A legacy batch adapter retains each source array
    and can prove timestamp bounds, but not absent cross-scope callback ordering.
    """

    def __init__(self, context: PacketPortContext) -> None:
        _check(isinstance(context, PacketPortContext), "context must be PacketPortContext")
        context.require("packet_attempt_events")
        self.context = context
        self._admissions: dict[int, PacketExtentAdmission] = {}
        self._terminals: dict[int, PacketExtentTerminal] = {}
        self._attempts: dict[int, _Attempt] = {}
        self._attempt_ids: set[tuple[int, int, int]] = set()
        self._logical_geometry: dict[tuple[int, int], tuple[int, int]] = {}
        self._events: list[PacketAttemptEvent] = []
        self._last_callback_at_ps = 0

    @property
    def events(self) -> tuple[PacketAttemptEvent, ...]:
        return tuple(self._events)

    @property
    def admissions(self) -> tuple[PacketExtentAdmission, ...]:
        return tuple(self._admissions.values())

    @property
    def terminals(self) -> tuple[PacketExtentTerminal, ...]:
        return tuple(self._terminals.values())

    def _check_callback_time(self, at_ps: int) -> None:
        if self.context.event_order_evidence == "live_callbacks":
            _check(at_ps >= self._last_callback_at_ps, "live callback time moved backwards")

    def admit(self, value: PacketExtentAdmission) -> None:
        _check(isinstance(value, PacketExtentAdmission), "invalid extent admission")
        token = value.extent_token
        _check(
            token not in self._admissions and token not in self._attempts,
            "extent token was already used in this session",
        )
        self._admissions[token] = value

    def observe(self, event: PacketAttemptEvent) -> None:
        _check(isinstance(event, PacketAttemptEvent), "invalid packet event")
        _check(event.packet_kind in self.context.packet_kinds, "port does not advertise this packet kind")
        self._check_callback_time(event.event_time_ps)
        _check(
            not self._events or event.event_time_ps >= self._events[-1].event_time_ps,
            "packet event array time moved backwards",
        )
        admission = self._admissions.get(event.extent_token)
        _check(admission is not None, "packet has no admitted parent")
        assert admission is not None
        _check(
            event.extent_token not in self._terminals, "packet follows parent transport retirement"
        )
        _check(
            event.wqe_id == admission.producer_id and event.extent_index == admission.extent_index,
            "packet changed its producer or extent identity",
        )
        _check(
            event.event_time_ps >= admission.eligible_at_ps, "packet precedes extent eligibility"
        )
        data = event.packet_kind in ("data", "retransmission")
        if data:
            _check(event.payload_bytes > 0, "data packet has no payload")
            _check(
                event.payload_offset_bytes + event.payload_bytes <= admission.payload_bytes,
                "packet range exceeds admitted payload",
            )
        else:
            _check(event.payload_bytes == 0, "control packets cannot carry logical extent payload")
        attempt = self._attempts.get(event.attempt_token)
        logical_id = (event.extent_token, event.packet_index)
        attempt_id = (*logical_id, event.transmission_attempt)
        geometry = (event.payload_offset_bytes, event.payload_bytes)
        if attempt is None:
            _check(
                event.event_kind == "packet_tx_started",
                "new attempt must start before other events",
            )
            _check(event.attempt_token not in self._admissions, "attempt token aliases an extent")
            _check(attempt_id not in self._attempt_ids, "attempt identity has a second token")
            previous = self._logical_geometry.get(logical_id)
            _check(
                previous is None or previous == geometry, "retry changed its logical packet range"
            )
            updated = _Attempt(event, event.event_kind, event.event_time_ps)
        else:
            _check(not attempt.terminal, "packet has a duplicate or post-terminal event")
            _check(attempt.first.geometry == event.geometry, "packet attempt geometry changed")
            _check(event.event_time_ps >= attempt.last_time_ps, "packet event time moved backwards")
            allowed = {
                "packet_tx_started": ("packet_tx_finished", "dropped"),
                "packet_tx_finished": ("packet_rx_arrived", "dropped"),
                "packet_rx_arrived": ("delivered", "dropped"),
            }
            _check(
                event.event_kind in allowed.get(attempt.last_kind, ()),
                "packet lifecycle order is invalid",
            )
            updated = replace(
                attempt,
                last_kind=event.event_kind,
                last_time_ps=event.event_time_ps,
                terminal=event.event_kind in ("delivered", "dropped"),
            )
        # Commit only after every precondition has passed.
        self._attempts[event.attempt_token] = updated
        self._attempt_ids.add(attempt_id)
        if data:
            self._logical_geometry[logical_id] = geometry
        self._events.append(event)
        self._last_callback_at_ps = max(self._last_callback_at_ps, event.event_time_ps)

    def retire(self, terminal: PacketExtentTerminal) -> None:
        _check(isinstance(terminal, PacketExtentTerminal), "invalid extent terminal")
        self._check_callback_time(terminal.at_ps)
        admission = self._admissions.get(terminal.extent_token)
        _check(admission is not None, "extent terminal has no admitted owner")
        assert admission is not None
        _check(terminal.extent_token not in self._terminals, "duplicate extent terminal")
        _check(
            terminal.producer_id == admission.producer_id, "terminal changed its producer identity"
        )
        _check(terminal.at_ps >= admission.eligible_at_ps, "terminal precedes eligibility")
        children = [
            x for x in self._attempts.values() if x.first.extent_token == terminal.extent_token
        ]
        _check(
            all(x.terminal and x.last_time_ps <= terminal.at_ps for x in children),
            "parent retires before an outstanding child",
        )
        if terminal.kind == "delivered":
            self._check_coverage(admission, children)
        self._terminals[terminal.extent_token] = terminal
        self._last_callback_at_ps = max(self._last_callback_at_ps, terminal.at_ps)

    @staticmethod
    def _check_coverage(admission: PacketExtentAdmission, children: list[_Attempt]) -> None:
        ranges = {
            (x.first.packet_index, x.first.payload_offset_bytes, x.first.payload_bytes)
            for x in children
            if x.last_kind == "delivered" and x.first.packet_kind in ("data", "retransmission")
        }
        cursor = 0
        for _, offset, length in sorted(ranges, key=lambda item: (item[1], item[0])):
            _check(
                offset == cursor, "successful payload ranges contain a gap or conflicting overlap"
            )
            cursor += length
        _check(
            cursor == admission.payload_bytes,
            "delivered extent lacks complete successful payload coverage",
        )

    def snapshot(self, *, final: bool = False) -> PacketPortSnapshot:
        admitted, retired = set(self._admissions), set(self._terminals)
        started = set(self._attempts)
        finished = {token for token, attempt in self._attempts.items() if attempt.terminal}
        live_extents, live_attempts = admitted - retired, started - finished
        if final:
            _check(
                not live_extents and not live_attempts,
                "final drain has missing transport terminals",
            )
        return PacketPortSnapshot(
            tuple(sorted(admitted)),
            tuple(sorted(retired)),
            tuple(sorted(live_extents)),
            tuple(sorted(started)),
            tuple(sorted(finished)),
            tuple(sorted(live_attempts)),
            sum(
                self._admissions[token].payload_bytes
                for token, row in self._terminals.items()
                if row.kind == "delivered"
            ),
        )


def validate_packet_observation(
    port: Mapping[str, Any],
    context: PacketPortContext,
) -> PacketPortLedger:
    """Check the legacy native arrays without inventing cross-scope ordering."""

    ledger = PacketPortLedger(replace(context, event_order_evidence="separate_arrays"))
    last_admission = last_terminal = 0
    for row in port["issued"]:
        _check(
            isinstance(row, Mapping)
            and set(row)
            == {
                "token",
                "wqe_id",
                "accepted_at_ps",
                "port_tx_at_ps",
                "payload_bytes",
            },
            "invalid native extent admission fields",
        )
        ledger.admit(
            PacketExtentAdmission(
                row["token"], row["wqe_id"], row["payload_bytes"], row["accepted_at_ps"]
            )
        )
        _check(row["accepted_at_ps"] >= last_admission, "native admission array time moved backwards")
        last_admission = row["accepted_at_ps"]
        _integer(row["port_tx_at_ps"], "port_tx_at_ps")
        _check(row["port_tx_at_ps"] >= row["accepted_at_ps"], "native port TX precedes admission")
    for row in port["packet_events"]:
        ledger.observe(PacketAttemptEvent.from_row(row))
    for row in port["terminals"]:
        _check(
            isinstance(row, Mapping) and set(row) == {"token", "wqe_id", "kind", "at_ps"},
            "invalid native extent terminal fields",
        )
        terminal = PacketExtentTerminal(row["token"], row["wqe_id"], row["kind"], row["at_ps"])
        _check(terminal.at_ps >= last_terminal, "native terminal array time moved backwards")
        ledger.retire(terminal)
        last_terminal = terminal.at_ps
    snapshot = ledger.snapshot(final=True)
    _check(
        tuple(port["live_tokens"]) == snapshot.live_extent_tokens,
        "native live-token projection disagrees",
    )
    return ledger
