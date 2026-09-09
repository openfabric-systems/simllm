"""Immutable causal observations of the retained physical packet calendar."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import asdict, dataclass, fields

SCHEMA = "simllm-peer-critical-path-v1"
INTERVAL_KINDS = frozenset({
    "source_feed", "input_link", "input_propagation", "switch_forward",
    "output_link", "output_propagation", "receive_ingress", "credit_return", "source_pacing",
})


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonblank text")


def _time(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class PeerCausalNode:
    node_id: str
    packet_id: str
    point: str
    at_ps: int
    parents: tuple[str, ...]
    resource_id: str
    interval_kind: str | None = None

    def __post_init__(self):
        for name in ("node_id", "packet_id", "point", "resource_id"):
            _text(getattr(self, name), name)
        _time(self.at_ps, "at_ps")
        if not isinstance(self.parents, tuple) or any(
            not isinstance(parent, str) or not parent for parent in self.parents
        ):
            raise ValueError("causal parents must be a tuple of identities")
        if len(set(self.parents)) != len(self.parents):
            raise ValueError("causal parents repeat an identity")
        if self.interval_kind is not None:
            if self.interval_kind not in INTERVAL_KINDS or len(self.parents) != 1:
                raise ValueError("a service interval needs one parent and a declared kind")
        elif not self.parents and self.point not in {"submitted", "eligible"}:
            raise ValueError("only packet submission or eligibility can be a causal root")


@dataclass(frozen=True)
class PeerCriticalInterval:
    node_id: str
    packet_id: str
    resource_id: str
    kind: str
    started_at_ps: int
    completed_at_ps: int
    original_started_at_ps: int


@dataclass(frozen=True)
class PeerCriticalPath:
    execution_id: str
    operation_id: str
    packet_ids: tuple[str, ...]
    released_at_ps: int
    visible_at_ps: int
    terminal_node_id: str
    node_ids: tuple[str, ...]
    intervals: tuple[PeerCriticalInterval, ...]

    @property
    def duration_ps(self):
        return self.visible_at_ps - self.released_at_ps


def validate_nodes(nodes: tuple[PeerCausalNode, ...]) -> dict[str, PeerCausalNode]:
    """Check causal append order and each exact maximum or service relation."""
    if not isinstance(nodes, tuple):
        raise TypeError("causal nodes must be an immutable tuple")
    seen, points = {}, set()
    for node in nodes:
        if not isinstance(node, PeerCausalNode):
            raise TypeError("causal nodes must be typed records")
        node.__post_init__()
        if node.node_id in seen or (node.packet_id, node.point) in points:
            raise ValueError("causal event identity is duplicated")
        if any(parent not in seen for parent in node.parents):
            raise ValueError("causal parent is missing or not earlier in append order")
        if node.parents:
            latest = max(seen[parent].at_ps for parent in node.parents)
            if node.interval_kind is None and latest != node.at_ps:
                raise ValueError("grant or visibility is not its latest causal gate")
            if node.interval_kind is not None and latest > node.at_ps:
                raise ValueError("service finishes before its causal start")
        seen[node.node_id] = node
        points.add((node.packet_id, node.point))
    return seen


class PeerCausalRecorder:
    """Append observations; resource markers never select or time a grant."""

    def __init__(self):
        self._nodes: list[PeerCausalNode] = []
        self._by_id: dict[str, PeerCausalNode] = {}
        self._points: dict[tuple[str, str], str] = {}
        self._markers: dict[tuple[str, Hashable], str] = {}
        self._capacity_gates: dict[tuple[str, str, Hashable], str | bool] = {}

    def point(self, packet_id, point):
        return self._points[(packet_id, point)]

    def previous(self, kind: str, identity: Hashable):
        return self._markers.get((kind, identity))

    def mark(self, kind: str, identity: Hashable, node_id: str):
        if node_id not in self._by_id:
            raise ValueError("resource observation has no causal event")
        self._markers[(kind, identity)] = node_id

    def capacity(self, packet_id, kind, identity, available):
        key = (packet_id, kind, identity)
        if not available:
            self._capacity_gates[key] = False
        elif self._capacity_gates.get(key) is False:
            released = self.previous(kind, identity)
            if released is None:
                raise ValueError("blocked capacity became available without a recorded return")
            self._capacity_gates[key] = released

    def capacity_parent(self, packet_id, kind, identity):
        value = self._capacity_gates.get((packet_id, kind, identity))
        if value is False:
            raise ValueError("grant still has an unresolved capacity gate")
        return value

    def add(self, packet_id, point, at_ps, parents=(), *, resource_id,
            interval_kind=None):
        parents = tuple(dict.fromkeys(parent for parent in parents if parent is not None))
        node = PeerCausalNode(
            f"event-{len(self._nodes):09d}", packet_id, point, at_ps,
            parents, resource_id, interval_kind,
        )
        if (packet_id, point) in self._points:
            raise ValueError("packet causal point was observed twice")
        if any(parent not in self._by_id for parent in parents):
            raise ValueError("causal event names an unrecorded parent")
        if parents:
            latest = max(self._by_id[parent].at_ps for parent in parents)
            if interval_kind is None and at_ps != latest:
                raise ValueError(f"unexplained {point} grant: observed {at_ps}, latest gate {latest}")
            if interval_kind is not None and at_ps < latest:
                raise ValueError("causal service starts after it finishes")
        self._nodes.append(node)
        self._by_id[node.node_id] = node
        self._points[(packet_id, point)] = node.node_id
        return node.node_id

    def observed_through(self, at_ps):
        """Omit scheduled future releases until the calendar reaches them."""
        _time(at_ps, "observation time")
        return tuple(node for node in self._nodes if node.at_ps <= at_ps)


def critical_path(nodes: tuple[PeerCausalNode, ...], *, execution_id: str,
                  operation_id: str, packet_ids: Iterable[str],
                  released_at_ps: int) -> PeerCriticalPath:
    """Follow one realized maximum chain, clipping retained predecessors."""
    _text(execution_id, "execution_id")
    _text(operation_id, "operation_id")
    _time(released_at_ps, "released_at_ps")
    packet_ids = tuple(packet_ids)
    if not packet_ids or len(set(packet_ids)) != len(packet_ids):
        raise ValueError("critical phase requires unique packet identities")
    by_id = validate_nodes(nodes)
    by_point = {(node.packet_id, node.point): node for node in nodes}
    try:
        terminals = tuple(by_point[(packet, "visible")] for packet in packet_ids)
        roots = tuple(by_point.get((packet, "submitted"), by_point[(packet, "eligible")])
                      for packet in packet_ids)
    except KeyError as error:
        raise ValueError("critical phase has a missing packet boundary") from error
    if any(root.at_ps != released_at_ps for root in roots):
        raise ValueError("phase release disagrees with its packet submission")
    # Stable first-in-parent-order tie selection retains every alternative edge.
    terminal = max(terminals, key=lambda node: node.at_ps)
    chain, intervals = [], []
    node = terminal
    while True:
        chain.append(node.node_id)
        if node.at_ps <= released_at_ps:
            break
        if not node.parents:
            raise ValueError("critical chain stops after the phase release")
        parent = max((by_id[name] for name in node.parents), key=lambda row: row.at_ps)
        if node.interval_kind is not None and node.at_ps > parent.at_ps:
            intervals.append(PeerCriticalInterval(
                node.node_id, node.packet_id, node.resource_id, node.interval_kind,
                max(parent.at_ps, released_at_ps), node.at_ps, parent.at_ps,
            ))
        node = parent
    chain.reverse()
    intervals.reverse()
    cursor = released_at_ps
    for interval in intervals:
        if interval.started_at_ps != cursor or interval.completed_at_ps <= cursor:
            raise ValueError("critical intervals have a gap, overlap or empty duration")
        cursor = interval.completed_at_ps
    if cursor != terminal.at_ps:
        raise ValueError("critical intervals do not cover packet visibility")
    return PeerCriticalPath(
        execution_id, operation_id, packet_ids, released_at_ps, terminal.at_ps,
        terminal.node_id, tuple(chain), tuple(intervals),
    )


def observation(nodes, phases, *, service_rates):
    """Publish only the typed graph and paths derived from its actual gates."""
    validate_nodes(nodes)
    return {
        "schema": SCHEMA,
        "service_rates": dict(service_rates),
        "nodes": [asdict(node) for node in nodes],
        "phases": [asdict(phase) for phase in phases],
    }


def validate_observation(value):
    """Read the closed diagnostic wire form and recompute every selected path."""
    if not isinstance(value, dict) or set(value) != {"schema", "nodes", "phases", "service_rates"}:
        raise ValueError("critical observation fields disagree with its schema")
    if value["schema"] != SCHEMA or not isinstance(value["nodes"], (tuple, list)):
        raise ValueError("unsupported critical observation schema")
    rates = value["service_rates"]
    if not isinstance(rates, dict) or set(rates) != {"feed", "switch", "receive"}:
        raise ValueError("critical service rates require the exact resource inventory")
    for name, rate in rates.items():
        if name == "switch" and rate is None:
            continue
        if type(rate) is not int or rate <= 0:
            raise ValueError("critical service rates must be positive integers")
    nodes = []
    for raw in value["nodes"]:
        if not isinstance(raw, dict) or set(raw) != {field.name for field in fields(PeerCausalNode)}:
            raise ValueError("causal node fields disagree with its schema")
        row = dict(raw)
        if not isinstance(row["parents"], (tuple, list)):
            raise TypeError("causal parent inventory is not an array")
        row["parents"] = tuple(row["parents"])
        nodes.append(PeerCausalNode(**row))
    nodes = tuple(nodes)
    validate_nodes(nodes)
    if not isinstance(value["phases"], (tuple, list)):
        raise TypeError("critical phase inventory is not an array")
    phases, seen = [], set()
    for raw in value["phases"]:
        if not isinstance(raw, dict) or set(raw) != {field.name for field in fields(PeerCriticalPath)}:
            raise ValueError("critical phase fields disagree with its schema")
        if not isinstance(raw["packet_ids"], (tuple, list)):
            raise TypeError("critical packet inventory is not an array")
        phase = critical_path(
            nodes, execution_id=raw["execution_id"], operation_id=raw["operation_id"],
            packet_ids=raw["packet_ids"], released_at_ps=raw["released_at_ps"],
        )
        key = (phase.execution_id, phase.operation_id, phase.released_at_ps)
        if key in seen:
            raise ValueError("critical phase identity is duplicated")
        seen.add(key)
        expected = asdict(phase)
        # Normalize array containers only, preserving scalar types exactly.
        def same(left, right):
            if isinstance(right, dict):
                return isinstance(left, dict) and set(left) == set(right) and all(
                    same(left[name], item) for name, item in right.items()
                )
            if isinstance(right, (tuple, list)):
                return isinstance(left, (tuple, list)) and len(left) == len(right) and all(
                    same(a, b) for a, b in zip(left, right, strict=True)
                )
            return type(left) is type(right) and left == right
        if not same(raw, expected):
            raise ValueError("recorded critical path differs from its causal graph")
        phases.append(phase)
    return nodes, tuple(phases)


def validate_packet_projection(value):
    """Join the diagnostic to original packets, paths, buffers and extents."""
    if "critical_path" not in value:
        return None
    nodes, phases = validate_observation(value["critical_path"])
    now = value["now_ps"]
    _time(now, "packet observation time")
    packets = {row["packet_id"]: row for row in value["packets"]}
    paths = {row["packet_id"]: row for row in value["physical_paths"]}
    if len(packets) != len(value["packets"]) or len(paths) != len(value["physical_paths"]):
        raise ValueError("packet observation repeats a physical identity")
    if not set(paths) <= set(packets):
        raise ValueError("physical path names an unknown packet")
    rates = value["critical_path"]["service_rates"]
    expected = {}
    submissions = {extent["transfer"]["extent_id"]: extent["transfer"]["released_at_ps"]
                   for extent in value["extents"]}
    if len(submissions) != len(value["extents"]):
        raise ValueError("original packet observation repeats an extent identity")

    def add(packet_id, point, at, resource, kind=None):
        if at is not None:
            _time(at, "original " + point)
        if at is not None and at <= now:
            key = (packet_id, point)
            if key in expected:
                raise ValueError("original observation repeats a causal point")
            expected[key] = (at, resource, kind)

    def duration(amount, rate):
        return (amount * 10**12 + rate - 1) // rate

    for packet_id, packet in packets.items():
        submitted = submissions[packet["extent_id"]]
        paced = packet["released_at_ps"] > submitted
        if packet["released_at_ps"] < submitted:
            raise ValueError("packet eligibility precedes its extent submission")
        if paced:
            add(packet_id, "submitted", submitted, f"gpu:{packet['source']}:submission")
        add(packet_id, "eligible", packet["released_at_ps"], f"gpu:{packet['source']}:submission",
            "source_pacing" if paced else None)
        observed = paths.get(packet_id)
        if observed is None:
            continue
        path = observed["path"]
        link = path["input_link"]
        resource = f"link:{link['link_id']}:{path['source_port_id']}"
        tx = packet["tx_started_at_ps"]
        add(packet_id, "tx-grant", tx, resource)
        add(packet_id, "source-feed", observed["source_feed_finished_at_ps"],
            f"gpu:{packet['source']}:feed", "source_feed")
        add(packet_id, "input-wire",
            tx + (packet["wire_bytes"] * 8 * 10**12 + link["link_rate_bps"] - 1) // link["link_rate_bps"],
            resource, "input_link")
        add(packet_id, "tx-finish", packet["tx_finished_at_ps"], resource)
        add(packet_id, "input-arrival", observed["first_hop_arrived_at_ps"], resource, "input_propagation")
        if observed["source_feed_finished_at_ps"] != tx + duration(packet["wire_bytes"], rates["feed"]):
            raise ValueError("source-feed observation disagrees with its declared service")
        switched = path["output_link"] is not None
        sw = packet["switch_started_at_ps"]
        if sw is not None:
            if not switched or rates["switch"] is None:
                raise ValueError("switch grant has no physical service binding")
            resource = f"switch:{path['switch_id']}"
            add(packet_id, "switch-grant", sw, resource)
            add(packet_id, "switch-service", sw + duration(packet["wire_bytes"], rates["switch"]),
                resource, "switch_forward")
            add(packet_id, "switch-finish", packet["switch_finished_at_ps"], resource)
            output = path["output_link"]
            add(packet_id, "output-wire",
                sw + (packet["wire_bytes"] * 8 * 10**12 + output["link_rate_bps"] - 1) // output["link_rate_bps"],
                f"link:{output['link_id']}:{path['switch_output_port_id']}", "output_link")
        add(packet_id, "rx-arrival", packet["rx_buffer_accepted_at_ps"],
            f"rx:{packet['destination']}", "output_propagation" if switched else None)
        resource = f"gpu:{packet['destination']}:receive"
        add(packet_id, "rx-grant", packet["rx_started_at_ps"], resource)
        add(packet_id, "rx-finish", packet["rx_finished_at_ps"], resource, "receive_ingress")
        if packet["rx_started_at_ps"] is not None and (
            packet["rx_finished_at_ps"] != packet["rx_started_at_ps"] + duration(packet["wire_bytes"], rates["receive"])
        ):
            raise ValueError("receiver observation disagrees with its declared service")
        add(packet_id, "visible", packet["visible_at_ps"],
            f"gpu:{packet['destination']}:visibility:{packet['ordering_domain']}")
    for claim in value["buffer_claims"]:
        packet_id = claim["packet_id"]
        if packet_id not in paths:
            raise ValueError("buffer claim has no physical packet path")
        switched = paths[packet_id]["path"]["output_link"] is not None
        point = ("receiver-credit" if switched and claim["buffer_id"].startswith("rx:")
                 else "input-credit")
        add(packet_id, point + "-return", claim["credit_available_at_ps"],
            claim["buffer_id"], "credit_return")
    actual = {(node.packet_id, node.point): (node.at_ps, node.resource_id, node.interval_kind) for node in nodes}
    if actual != expected:
        raise ValueError("causal event domain, packet, resource or timestamp disagrees with original observation")
    by_point = {(node.packet_id, node.point): node.node_id for node in nodes}
    receive, visibility = {}, {}
    for node in nodes:
        packet = packets[node.packet_id]
        switched = paths.get(node.packet_id, {}).get("path", {}).get("output_link") is not None
        fixed = {
            "submitted": (),
            "eligible": ("submitted",) if (node.packet_id, "submitted") in by_point else (),
            "source-feed": ("tx-grant",), "input-wire": ("tx-grant",),
            "tx-finish": ("source-feed", "input-wire"), "input-arrival": ("tx-finish",),
            "switch-service": ("switch-grant",), "output-wire": ("switch-grant",),
            "switch-finish": ("switch-service", "output-wire"),
            "rx-arrival": ("switch-finish" if switched else "input-arrival",),
            "rx-finish": ("rx-grant",),
            "input-credit-return": ("switch-finish" if switched else "rx-finish",),
            "receiver-credit-return": ("rx-finish",),
        }
        parents = None
        key = (packet["destination"], packet["ordering_domain"])
        if node.point in fixed:
            parents = tuple(by_point[(node.packet_id, point)] for point in fixed[node.point])
        elif node.point == "rx-grant":
            parents = tuple(parent for parent in (
                by_point[(node.packet_id, "rx-arrival")], receive.get(packet["destination"]),
            ) if parent is not None)
        elif node.point == "visible":
            parents = tuple(parent for parent in (
                by_point[(node.packet_id, "rx-finish")], visibility.get(key),
            ) if parent is not None)
        if parents is not None and node.parents != parents:
            raise ValueError("causal parent identity differs from its original packet service or ordering gate")
        if node.point == "rx-finish":
            receive[packet["destination"]] = node.node_id
        elif node.point == "visible":
            visibility[key] = node.node_id
    expected_phases = {}
    owned = set()
    for extent in value["extents"]:
        key = (extent["execution_id"], extent["operation_id"], extent["transfer"]["released_at_ps"])
        members = tuple(extent["packet_ids"])
        if not members or owned.intersection(members):
            raise ValueError("extent has missing or repeated packet ownership")
        if any(name not in packets or packets[name]["extent_id"] != extent["transfer"]["extent_id"] for name in members):
            raise ValueError("extent packet identity differs from original packet")
        owned.update(members)
        expected_phases[key] = (*expected_phases.get(key, ()), *members)
    if owned != set(packets):
        raise ValueError("extents do not own every observed packet")
    complete = {
        key: members for key, members in expected_phases.items()
        if all(packets[name]["visible_at_ps"] is not None and packets[name]["visible_at_ps"] <= now for name in members)
    }
    actual_phases = {
        (phase.execution_id, phase.operation_id, phase.released_at_ps): phase.packet_ids for phase in phases
    }
    if actual_phases != complete:
        raise ValueError("critical phase identity or packet membership differs from original extents")
    return nodes, phases
