"""Audit the frozen pipeline study's read-only physical packet observations."""

from __future__ import annotations

import heapq
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "rnic-cn-trace-v1"
COMMON = "schema,sequence,observed_at_ps,"
HEADERS = {
    "flows": COMMON + "flow_id,packet_flow_id,source,destination,tag,start_time_ps,"
    "payload_bytes,total_wire_bytes,total_data_packets",
    "packets": COMMON + "packet_id,lifecycle_id,flow_id,packet_flow_id,kind,source,"
    "destination,path_id,wire_bytes,packet_index,attempt,payload_offset,payload_bytes,"
    "header_bytes,eta_ps,source_start_ps,source_end_ps,duplicate_test_copy,nflow_ppm,"
    "membership_epoch,n_hat,wire_rate_bps,effective_time_ps,feedback_deadline_ps,"
    "lease_expiry_ps,requested_attempt,acknowledged_attempt,final_gap_detection_ps,"
    "total_payload_bytes,total_wire_bytes,total_data_packets",
    "queues": COMMON + "transition,time_ps,switch_type,switch_id,ingress_id,egress_id,"
    "priority,packet_flow_id,packet_id,lifecycle_id,flow_id,kind,wire_bytes,buffered_bytes,"
    "in_service_bytes,backlog_bytes,shared_buffer_bytes",
    "events": COMMON + "event,time_ps,flow_id,lifecycle_id,packet_id,packet_index,attempt,"
    "detail,logical_release_ps,service_start_ps,service_end_ps,trigger_lifecycle_id,"
    "predecessor_lifecycle_id,origin_attempt,deadline_ps",
}
MANIFEST_HEADER = (
    "schema,status,finalized,physical_quiescence,sequence_first,sequence_last,"
    "observation_count,flows_rows,packets_rows,queues_rows,events_rows,flow_count,"
    "data_packet_count,control_packet_count,physical_quiescence_time_ps"
)
TEXT_FIELDS = {
    "schema",
    "kind",
    "transition",
    "event",
    "detail",
    "status",
    "finalized",
    "physical_quiescence",
}
OPTIONAL = {
    "packets": {
        "packet_index",
        "attempt",
        "payload_offset",
        "payload_bytes",
        "header_bytes",
        "eta_ps",
        "nflow_ppm",
        "membership_epoch",
        "n_hat",
        "wire_rate_bps",
        "effective_time_ps",
        "feedback_deadline_ps",
        "lease_expiry_ps",
        "requested_attempt",
        "acknowledged_attempt",
        "final_gap_detection_ps",
        "total_payload_bytes",
        "total_wire_bytes",
        "total_data_packets",
    },
    "events": {
        "packet_id",
        "packet_index",
        "attempt",
        "logical_release_ps",
        "service_start_ps",
        "service_end_ps",
        "origin_attempt",
        "deadline_ps",
    },
}
KINDS = {
    "DATA",
    "DECLARE",
    "ACCEPT",
    "GRANT_UPDATE",
    "GAP_NACK",
    "GAP_RESOLVED",
    "RETIRE",
    "NFLOW_UPDATE",
}
PARTITIONS = ("ep_data", "ep_control", "pp_or_other_control", "other_data")
UINT = re.compile(r"0|[1-9][0-9]*", re.ASCII)


class TraceAuditError(ValueError):
    """A fatal observation or physical-accounting guard invalidates the trace."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TraceAuditError(message)


def _rows(path: Path, table: str, header: str):
    try:
        with path.open("r", encoding="ascii", newline="") as stream:
            require(stream.readline() == header + "\n", f"{table}: schema header mismatch")
            names = header.split(",")
            previous = 0
            for number, raw in enumerate(stream, 2):
                require(
                    raw.endswith("\n") and not any(c in raw for c in '\r"'),
                    f"{table}:{number}: truncated or noncanonical CSV",
                )
                cells = raw[:-1].split(",")
                require(len(cells) == len(names), f"{table}:{number}: CSV width mismatch")
                row = {}
                for key, value in zip(names, cells):
                    if key in TEXT_FIELDS:
                        row[key] = value
                    elif value == "" and key in OPTIONAL.get(table, set()):
                        row[key] = None
                    else:
                        require(
                            bool(UINT.fullmatch(value)), f"{table}:{number}: invalid integer {key}"
                        )
                        parsed = int(value)
                        require(parsed <= 2**64 - 1, f"{table}:{number}: integer overflow {key}")
                        row[key] = parsed
                require(row["schema"] == SCHEMA, f"{table}:{number}: unknown schema")
                if table != "manifest":
                    require(row["sequence"] > previous, f"{table}: nonincreasing sequence")
                    previous = row["sequence"]
                yield row
    except (OSError, UnicodeError) as error:
        raise TraceAuditError(f"{table}: unreadable trace: {error}") from error


def _merged_rows(directory: Path, counts: Counter):
    iterators = {
        name: iter(_rows(directory / f"{name}.csv", name, header))
        for name, header in HEADERS.items()
    }
    with ExitStack() as stack:
        for iterator in iterators.values():
            stack.callback(iterator.close)
        heap = []
        for name, iterator in iterators.items():
            row = next(iterator, None)
            if row is not None:
                heapq.heappush(heap, (row["sequence"], name, row))
        expected, observed = 1, 0
        while heap:
            sequence, name, row = heapq.heappop(heap)
            require(sequence == expected, "global observation sequence is missing or duplicated")
            require(row["observed_at_ps"] >= observed, "observation time moved backwards")
            observed = row["observed_at_ps"]
            expected += 1
            counts[name] += 1
            yield name, row
            following = next(iterators[name], None)
            if following is not None:
                heapq.heappush(heap, (following["sequence"], name, following))


@dataclass(slots=True)
class Packet:
    identity: int
    lifecycle: int
    flow: int
    kind: str
    source: int
    destination: int
    wire: int
    source_start: int
    source_end: int
    index: int | None
    attempt: int | None
    offset: int | None
    payload: int | None
    eta: int | None
    group: str
    route: tuple
    control_attempt: int | None = None
    created: bool = True
    terminal: str | None = None
    terminal_time: int | None = None
    arrival: int | None = None
    visits: list = field(default_factory=list)
    switch_count: int = 0
    last_switch_end: int | None = None
    current_queue: tuple | None = None
    dropped: bool = False
    admission: str | None = None
    release: int | None = None
    rx_start: int | None = None
    rx_end: int | None = None
    rx_predecessor: int = 0
    rx_completed: bool = False
    delivery: int | None = None
    delivery_predecessor: int = 0
    delivery_trigger: int = 0
    authorization: dict | None = None


@dataclass(slots=True)
class Visit:
    packet: Packet
    enqueue: int
    ingress: int
    priority: int
    sequence: int
    reserve: bool
    integral_at_enqueue: tuple
    earlier_ep_bytes: int = 0
    active_ep_residual: int = 0
    earlier_ep_packets: list = field(default_factory=list)
    competitors: list = field(default_factory=list)
    start: int | None = None


@dataclass(slots=True)
class Queue:
    buffered: dict = field(default_factory=dict)
    arbitration: list = field(default_factory=list)
    active: Visit | None = None
    now: int = 0
    integral: list = field(default_factory=lambda: [0, 0, 0, 0])
    buffered_bytes: int = 0
    reserve_bytes: int = 0
    peak_buffered_bytes: int = 0
    enqueues: int = 0
    services: int = 0
    drops: int = 0
    all_wait: int = 0
    pp_wait: int = 0
    pp_parts: list = field(default_factory=lambda: [0, 0, 0, 0])
    pp_bound: int = 0
    previous_service_lifecycle: int = 0
    previous_service_end: int = 0
    tagged_waiters: dict = field(default_factory=dict)

    def advance(self, time: int, ps_per_byte: int) -> None:
        require(time >= self.now, "physical queue time moved backwards")
        if self.active is not None:
            end = self.active.start + self.active.packet.wire * ps_per_byte
            require(time <= end, "missing queue serialization completion")
            self.integral[PARTITIONS.index(self.active.packet.group)] += time - self.now
            if time > self.now:
                for waiter in self.tagged_waiters.values():
                    packet = self.active.packet
                    if (
                        waiter.competitors
                        and waiter.competitors[-1]["lifecycle_id"] == packet.lifecycle
                    ):
                        interval = waiter.competitors[-1]
                        interval["overlap_end_ps"] = time
                        interval["overlap_ps"] += time - self.now
                    else:
                        waiter.competitors.append(
                            {
                                "packet_id": packet.identity,
                                "lifecycle_id": packet.lifecycle,
                                "flow_id": packet.flow,
                                "kind": packet.kind,
                                "class": packet.group,
                                "wire_bytes": packet.wire,
                                "enqueue_ps": self.active.enqueue,
                                "service_start_ps": self.active.start,
                                "service_end_ps": end,
                                "overlap_start_ps": self.now,
                                "overlap_end_ps": time,
                                "overlap_ps": time - self.now,
                            }
                        )
        elif self.buffered:
            require(time == self.now, "work-conserving egress was idle with ready packets")
        self.now = time


class Audit:
    def __init__(
        self,
        completion_rows,
        endpoint_by_rank,
        spines,
        delta,
        tick,
        rate,
        propagation,
        capacity,
        headroom,
        ring_capacity,
        max_retries,
        payload_limit,
        header,
        control,
    ):
        require(
            rate > 0 and 8_000_000_000_000 % rate == 0,
            "audit requires an exact integral picosecond wire-byte duration",
        )
        require(
            delta > 0
            and tick > 0
            and propagation >= 0
            and capacity > 0
            and headroom >= 0
            and ring_capacity > 0
            and max_retries >= 0,
            "invalid audit configuration",
        )
        require(
            payload_limit > 0 and header >= 0 and control > 0, "invalid packetization configuration"
        )
        self.ps_per_byte = 8_000_000_000_000 // rate
        self.delta, self.tick, self.propagation = delta, tick, propagation
        self.capacity, self.headroom, self.ring_capacity = capacity, headroom, ring_capacity
        self.max_retries, self.payload_limit, self.header, self.control = (
            max_retries,
            payload_limit,
            header,
            control,
        )
        self.rank_by_endpoint = {endpoint: rank for rank, endpoint in enumerate(endpoint_by_rank)}
        require(
            len(self.rank_by_endpoint) == len(endpoint_by_rank), "endpoint map is not bijective"
        )
        require(
            set(endpoint_by_rank) == set(range(64)) and spines in (2, 8),
            "trace requires the frozen 64-endpoint, two- or eight-spine fabric",
        )
        self.spines = spines
        self.completions = {}
        for row in completion_rows:
            fields = {
                "flow_id",
                "source",
                "destination",
                "tag",
                "payload_bytes",
                "start_time_ps",
                "completion_time_ps",
                "fct_ps",
            }
            require(fields <= row.keys(), "completion row lacks required native fields")
            require(
                all(type(row[name]) is int and row[name] >= 0 for name in fields),
                "completion row contains invalid native integer fields",
            )
            require(
                row.get("profile", "rnic-cn") == "rnic-cn", "trace completion uses another profile"
            )
            identity = row["flow_id"]
            require(identity not in self.completions, "duplicate completion flow identity")
            require(
                row["completion_time_ps"] - row["start_time_ps"] == row["fct_ps"]
                and row["fct_ps"] > 0,
                "completion timestamp/FCT mismatch",
            )
            self.completions[identity] = dict(row)
        require(bool(self.completions), "trace has no expected flow completions")
        self.flows = {}
        self.packet_flows = {}
        self.packets = {}
        self.lifecycles = {}
        self.creations = {}
        self.queues = {}
        self.switch_bytes = Counter()
        self.switch_reserve = Counter()
        self.originals = {}
        self.attempts = set()
        self.attempt_packet = {}
        self.accepted_logical = set()
        self.rejected_attempt = {}
        self.authorizations = {}
        self.resolutions = {}
        self.highest_authorized = Counter()
        self.delivered = Counter()
        self.delivered_bytes = Counter()
        self.delivered_wire = Counter()
        self.previous_delivery = {}
        self.previous_rx = {}
        self.ring = defaultdict(list)
        self.ring_bytes = Counter()
        self.ring_peak = Counter()
        self.source_end = Counter()
        self.completed = {}
        self.counts = Counter()
        self.latest_observed = 0
        self.previous_record = None

    def packet(self, row):
        identity, lifecycle, flow_id = row["packet_id"], row["lifecycle_id"], row["flow_id"]
        require(
            identity not in self.packets and lifecycle not in self.lifecycles,
            "duplicate physical packet or lifecycle identity",
        )
        require(flow_id in self.flows, "packet references unknown flow")
        flow = self.flows[flow_id]
        require(
            row["packet_flow_id"] == flow["packet_flow_id"], "packet PacketFlow mapping mismatch"
        )
        kind = row["kind"]
        require(kind in KINDS, "unknown packet kind")
        forward = kind in {"DATA", "DECLARE", "NFLOW_UPDATE", "RETIRE"}
        endpoints = (flow["source"], flow["destination"])
        require(
            (row["source"], row["destination"]) == (endpoints if forward else endpoints[::-1]),
            "packet physical direction disagrees with its control kind",
        )
        creation = self.creations.pop(lifecycle, None)
        require(
            creation is not None
            and creation["packet_id"] == identity
            and creation["flow_id"] == flow_id,
            "packet creation identity missing or misbound",
        )
        require(
            creation["time_ps"] == row["source_end_ps"] == row["observed_at_ps"],
            "packet allocation does not coincide with source service completion",
        )
        require(
            row["source_end_ps"] - row["source_start_ps"] == row["wire_bytes"] * self.ps_per_byte,
            "source serialization disagrees with wire bytes",
        )
        require(
            row["source_start_ps"] >= flow["start_time_ps"], "source service precedes flow release"
        )
        require(
            row["source_start_ps"] >= self.source_end[row["source"]], "source wire services overlap"
        )
        self.source_end[row["source"]] = row["source_end_ps"]
        require(row["duplicate_test_copy"] == 0, "test-copy injection is outside the frozen study")
        required = {
            "DATA": "packet_index attempt payload_offset payload_bytes header_bytes eta_ps "
            "total_payload_bytes total_wire_bytes total_data_packets",
            "GAP_NACK": "packet_index payload_offset payload_bytes header_bytes requested_attempt",
            "GAP_RESOLVED": "packet_index payload_offset payload_bytes header_bytes acknowledged_attempt",
            "DECLARE": "nflow_ppm",
            "NFLOW_UPDATE": "nflow_ppm",
            "ACCEPT": "membership_epoch n_hat wire_rate_bps effective_time_ps "
            "feedback_deadline_ps lease_expiry_ps",
            "GRANT_UPDATE": "membership_epoch n_hat wire_rate_bps effective_time_ps "
            "feedback_deadline_ps lease_expiry_ps",
            "RETIRE": "final_gap_detection_ps total_payload_bytes total_wire_bytes total_data_packets",
        }
        present = {name for name in OPTIONAL["packets"] if row[name] is not None}
        require(
            present == set(required[kind].split()),
            "packet kind-specific metadata is missing or inactive",
        )
        if kind in {"DATA", "GAP_NACK", "GAP_RESOLVED"}:
            index = row["packet_index"]
            require(index < flow["total_data_packets"], "logical packet index exceeds flow ledger")
            offset = index * self.payload_limit
            payload = min(self.payload_limit, flow["payload_bytes"] - offset)
            require(
                (row["payload_offset"], row["payload_bytes"], row["header_bytes"])
                == (offset, payload, self.header),
                "logical DATA extent is not the input packetization",
            )
        if kind in {"GAP_NACK", "GAP_RESOLVED"}:
            control_attempt = (
                row["requested_attempt"] if kind == "GAP_NACK" else row["acknowledged_attempt"]
            )
            require(
                1 <= control_attempt <= self.max_retries,
                "control attempt exceeds frozen retry guard",
            )
        if kind == "DATA":
            require(
                row["wire_bytes"] == row["payload_bytes"] + self.header,
                "DATA wire byte conservation failed",
            )
            key = flow_id, row["packet_index"], row["attempt"]
            require(
                key not in self.attempts and row["attempt"] <= self.max_retries,
                "logical attempt is duplicated or exceeds retry guard",
            )
            self.attempts.add(key)
            self.attempt_packet[key] = identity
            require(
                creation["packet_index"] == row["packet_index"], "creation logical index mismatch"
            )
            if row["attempt"] == 0:
                self.originals[(flow_id, row["packet_index"])] = identity
            else:
                require(
                    (flow_id, row["packet_index"]) in self.originals,
                    "retry has no original logical DATA",
                )
                require(key in self.authorizations, "retry source service lacks authorization")
                require(
                    self.authorizations[key]["time_ps"] <= row["source_start_ps"],
                    "retry source service precedes authorization",
                )
                resolved = self.resolutions.get((flow_id, row["packet_index"]))
                require(
                    resolved is None or row["source_start_ps"] < resolved,
                    "retry source service begins after terminal gap resolution",
                )
        else:
            require(row["wire_bytes"] == self.control, "control wire size changed")
            require(creation["packet_index"] is None, "control creation has a DATA index")
        if kind in {"DATA", "RETIRE"}:
            require(
                (row["total_payload_bytes"], row["total_wire_bytes"], row["total_data_packets"])
                == (flow["payload_bytes"], flow["total_wire_bytes"], flow["total_data_packets"]),
                "packet final ledger disagrees with flow",
            )
        group = (
            ("ep_data" if flow["class"] == "ep" else "other_data")
            if kind == "DATA"
            else ("ep_control" if flow["class"] == "ep" else "pp_or_other_control")
        )
        packet = Packet(
            identity,
            lifecycle,
            flow_id,
            kind,
            row["source"],
            row["destination"],
            row["wire_bytes"],
            row["source_start_ps"],
            row["source_end_ps"],
            row["packet_index"],
            row["attempt"],
            row["payload_offset"],
            row["payload_bytes"],
            row["eta_ps"],
            group,
            self.route(row["source"], row["destination"], row["path_id"]),
            row["requested_attempt"] if kind == "GAP_NACK" else row["acknowledged_attempt"],
        )
        if kind == "DATA" and packet.attempt:
            packet.authorization = self.authorizations[(flow_id, packet.index, packet.attempt)]
        self.packets[identity], self.lifecycles[lifecycle] = packet, identity
        self.counts["data_packets" if kind == "DATA" else "control_packets"] += 1

    def route(self, source, destination, path):
        source_leaf, source_port = divmod(source, 8)
        destination_leaf, destination_port = divmod(destination, 8)
        if source_leaf == destination_leaf:
            require(path == 0, "local route has a nonexistent path ID")
            return ((1, source_leaf, source_port, destination_port),)
        require(path < self.spines, "route selects a nonexistent spine")
        return (
            (1, source_leaf, source_port, 8 + path),
            (2, path, source_leaf, destination_leaf),
            (1, destination_leaf, 8 + path, destination_port),
        )

    def flow(self, row):
        identity = row["flow_id"]
        require(
            identity not in self.flows and row["packet_flow_id"] not in self.packet_flows,
            "duplicate flow or PacketFlow identity",
        )
        require(identity in self.completions, "trace flow missing from completion CSV")
        for field_name in ("source", "destination", "tag", "start_time_ps", "payload_bytes"):
            require(
                row[field_name] == self.completions[identity][field_name],
                f"flow {identity}: completion {field_name} mismatch",
            )
        require(row["start_time_ps"] == row["observed_at_ps"], "flow declaration time mismatch")
        require(
            row["source"] in self.rank_by_endpoint and row["destination"] in self.rank_by_endpoint,
            "flow endpoint missing from explicit rank map",
        )
        require(row["payload_bytes"] > 0, "zero-DATA flow is outside the frozen study")
        count = (row["payload_bytes"] + self.payload_limit - 1) // self.payload_limit
        require(
            (row["total_data_packets"], row["total_wire_bytes"])
            == (count, row["payload_bytes"] + count * self.header),
            "flow packetization ledger mismatch",
        )
        source, destination = (self.rank_by_endpoint[row[key]] for key in ("source", "destination"))
        pp = source % 8 == destination % 8 == 0
        ep = source % 8 in (1, 2, 3, 4) and destination % 8 in (1, 2, 3, 4)
        require(pp or ep, "flow is outside the frozen PP/EP rank inventory")
        if pp:
            require(destination == source + 8, "pipeline flow skips a stage")
        else:
            require(
                source // 8 != destination // 8, "expert fabric flow stays inside one semantic node"
            )
        row.update(
            {"class": "pp" if pp else "ep", "source_rank": source, "destination_rank": destination}
        )
        self.flows[identity] = row
        self.packet_flows[row["packet_flow_id"]] = identity

    def bound_packet(self, row):
        identity = row["packet_id"]
        require(identity in self.packets, "observation references unknown physical packet")
        packet = self.packets[identity]
        require(
            (row["lifecycle_id"], row["flow_id"]) == (packet.lifecycle, packet.flow),
            "observation physical/logical identity mismatch",
        )
        return packet

    def queue(self, row):
        packet = self.bound_packet(row)
        require(packet.terminal is None, "queue observation follows physical termination")
        require(
            (row["kind"], row["wire_bytes"], row["packet_flow_id"])
            == (packet.kind, packet.wire, self.flows[packet.flow]["packet_flow_id"]),
            "queue packet metadata mismatch",
        )
        require(row["priority"] == (0 if packet.kind == "DATA" else 2), "packet priority changed")
        require(row["switch_type"] in (1, 2), "queue is outside the frozen two-tier fabric")
        require(
            packet.switch_count < len(packet.route)
            and (row["switch_type"], row["switch_id"], row["ingress_id"], row["egress_id"])
            == packet.route[packet.switch_count],
            "physical switch or port visit disagrees with the declared packet route",
        )
        require(row["time_ps"] == row["observed_at_ps"], "queue physical/observation time mismatch")
        switch = row["switch_type"], row["switch_id"]
        key = (*switch, row["egress_id"])
        queue = self.queues.setdefault(key, Queue())
        time = row["time_ps"]
        queue.advance(time, self.ps_per_byte)
        transition = row["transition"]
        if transition in {"enqueue", "drop"}:
            require(
                packet.current_queue is None and not packet.dropped,
                "packet overlaps physical queue visits",
            )
            previous = (
                packet.last_switch_end if packet.last_switch_end is not None else packet.source_end
            )
            require(
                time == previous + self.propagation,
                "switch arrival violates previous service plus propagation",
            )
            base_switch = self.switch_bytes[switch] - self.switch_reserve[switch]
            base_egress = queue.buffered_bytes - queue.reserve_bytes
            full = (
                base_switch + packet.wire > self.capacity
                or base_egress + packet.wire > self.capacity
            )
            reserve = (
                full
                and packet.kind != "DATA"
                and queue.reserve_bytes + packet.wire <= self.headroom
            )
            if transition == "drop":
                require(
                    full and not reserve,
                    "fabric drop lacks a full buffer or exhausted control reserve",
                )
                packet.dropped = True
                queue.drops += 1
            else:
                require(not full or reserve, "buffer admission exceeds physical capacity")
                visit = Visit(
                    packet,
                    time,
                    row["ingress_id"],
                    row["priority"],
                    row["sequence"],
                    reserve,
                    tuple(queue.integral),
                )
                if self.is_pp_data(packet):
                    visit.earlier_ep_packets = [
                        {
                            "packet_id": v.packet.identity,
                            "lifecycle_id": v.packet.lifecycle,
                            "enqueue_ps": v.enqueue,
                            "wire_bytes": v.packet.wire,
                        }
                        for v in queue.buffered.values()
                        if v.packet.group == "ep_data"
                        and v.enqueue < time
                        and v.priority == row["priority"]
                    ]
                    visit.earlier_ep_bytes = sum(p["wire_bytes"] for p in visit.earlier_ep_packets)
                    active = queue.active
                    if active is not None and active.packet.group == "ep_data":
                        visit.active_ep_residual = (
                            active.start + active.packet.wire * self.ps_per_byte - time
                        )
                    queue.tagged_waiters[packet.identity] = visit
                queue.buffered[packet.identity] = visit
                heapq.heappush(
                    queue.arbitration,
                    (-row["priority"], time, row["ingress_id"], row["sequence"], packet.identity),
                )
                queue.buffered_bytes += packet.wire
                self.switch_bytes[switch] += packet.wire
                if reserve:
                    queue.reserve_bytes += packet.wire
                    self.switch_reserve[switch] += packet.wire
                packet.current_queue = key
                queue.enqueues += 1
                queue.peak_buffered_bytes = max(queue.peak_buffered_bytes, queue.buffered_bytes)
        elif transition == "service_start":
            require(
                queue.active is None and packet.identity in queue.buffered,
                "queue service has no eligible packet or overlaps active service",
            )
            require(packet.current_queue == key, "queue service moved to another egress")
            selected = heapq.heappop(queue.arbitration)
            require(
                selected[-1] == packet.identity,
                "queue service violates frozen oldest-head priority arbitration",
            )
            visit = queue.buffered.pop(packet.identity)
            require(visit.ingress == row["ingress_id"], "queue visit ingress identity changed")
            parts = [
                current - before
                for current, before in zip(queue.integral, visit.integral_at_enqueue)
            ]
            wait = time - visit.enqueue
            require(sum(parts) == wait, "queue wait is not exactly competing wire service")
            require(
                time == max(visit.enqueue, queue.previous_service_end),
                "queue service start does not select eligibility or preceding wire release",
            )
            bound = visit.earlier_ep_bytes * self.ps_per_byte + visit.active_ep_residual
            require(wait >= bound, "tagged queue wait beats independent earlier EP byte bound")
            queue.all_wait += wait
            if self.is_pp_data(packet):
                del queue.tagged_waiters[packet.identity]
                require(
                    sum(v["overlap_ps"] for v in visit.competitors) == wait,
                    "tagged service intersections do not reconstruct exact queue wait",
                )
                queue.pp_wait += wait
                queue.pp_bound += bound
                queue.pp_parts = [a + b for a, b in zip(queue.pp_parts, parts)]
                packet.visits.append(
                    {
                        "switch_type": key[0],
                        "switch_id": key[1],
                        "egress_id": key[2],
                        "ingress_id": visit.ingress,
                        "enqueue_ps": visit.enqueue,
                        "service_start_ps": time,
                        "service_end_ps": time + packet.wire * self.ps_per_byte,
                        "queue_wait_ps": wait,
                        "service_ps": packet.wire * self.ps_per_byte,
                        "service_ahead_ps": dict(zip(PARTITIONS, parts)),
                        "earlier_ep_data_wire_bytes": visit.earlier_ep_bytes,
                        "active_ep_data_residual_ps": visit.active_ep_residual,
                        "ep_data_bound_ps": bound,
                        "earlier_ep_data_packets": visit.earlier_ep_packets,
                        "competing_service_intersections": visit.competitors,
                        "service_predecessor_lifecycle_id": queue.previous_service_lifecycle,
                        "previous_service_end_ps": queue.previous_service_end,
                        "selected_service_start_inputs": self.selected_max_inputs(
                            visit.enqueue,
                            queue.previous_service_end,
                            "enqueue",
                            "previous_service_end",
                        ),
                    }
                )
            queue.buffered_bytes -= packet.wire
            self.switch_bytes[switch] -= packet.wire
            if visit.reserve:
                queue.reserve_bytes -= packet.wire
                self.switch_reserve[switch] -= packet.wire
            visit.start, queue.active = time, visit
        elif transition == "service_end":
            require(
                queue.active is not None and queue.active.packet.identity == packet.identity,
                "queue completion identity does not match active serialization",
            )
            require(row["ingress_id"] == queue.active.ingress, "queue completion ingress changed")
            require(
                time - queue.active.start == packet.wire * self.ps_per_byte,
                "queue serialization duration disagrees with wire bytes",
            )
            packet.current_queue, packet.last_switch_end = None, time
            packet.switch_count += 1
            queue.previous_service_lifecycle = packet.lifecycle
            queue.previous_service_end = time
            queue.active = None
            queue.services += 1
        else:
            raise TraceAuditError(f"unknown queue transition {transition}")
        in_service = queue.active.packet.wire if queue.active is not None else 0
        require(
            (
                row["buffered_bytes"],
                row["in_service_bytes"],
                row["backlog_bytes"],
                row["shared_buffer_bytes"],
            )
            == (
                queue.buffered_bytes,
                in_service,
                queue.buffered_bytes + in_service,
                self.switch_bytes[switch],
            ),
            "queue occupancy snapshot violates independent byte conservation",
        )

    def is_pp_data(self, packet):
        return packet.kind == "DATA" and self.flows[packet.flow]["class"] == "pp"

    def event(self, row):
        event, time, flow_id, lifecycle = (
            row[key] for key in ("event", "time_ps", "flow_id", "lifecycle_id")
        )
        require(flow_id in self.flows, "runtime event references unknown flow")
        require(time <= row["observed_at_ps"], "runtime event reports a future cause")
        timings = {"logical_release_ps", "service_start_ps", "service_end_ps"}
        active_timings = (
            timings
            if event == "rx_schedule"
            else ({"logical_release_ps"} if event == "admission" else set())
        )
        require(
            all(row[name] is None for name in timings - active_timings),
            "runtime event carries inactive receiver timing fields",
        )
        if event != "retry_authorized":
            require(
                row["origin_attempt"] is None and row["deadline_ps"] is None,
                "runtime event carries inactive retry timer fields",
            )
        if event not in {"delivery", "flow_complete", "retry_authorized"}:
            require(row["trigger_lifecycle_id"] == 0, "runtime event carries an inactive trigger")
        if event not in {"rx_schedule", "delivery"}:
            require(
                row["predecessor_lifecycle_id"] == 0,
                "runtime event carries an inactive predecessor",
            )
        if event not in {
            "packet_created",
            "endpoint_consumed",
            "fabric_drop",
            "admission",
            "retry_authorized",
        }:
            require(row["detail"] == "", "runtime event carries an inactive detail field")
        if event in {"packet_created", "endpoint_consumed", "fabric_drop"}:
            require(row["attempt"] is None, "lifecycle event carries an inactive attempt")
        if event == "packet_created":
            require(
                lifecycle > 0
                and lifecycle not in self.creations
                and lifecycle not in self.lifecycles,
                "physical lifecycle was created twice",
            )
            require(
                row["detail"] == "0" and time == row["observed_at_ps"],
                "packet creation has prior route service",
            )
            self.creations[lifecycle] = row
            return
        if event == "retry_authorized":
            self.retry(row)
            return
        packet = self.bound_packet(row)
        if event not in {"endpoint_consumed", "fabric_drop"}:
            require(
                (row["packet_index"], row["attempt"]) == (packet.index, packet.attempt)
                if packet.kind == "DATA"
                else row["packet_index"] is None and row["attempt"] is None,
                "runtime event logical DATA identity mismatch",
            )
        if event == "endpoint_arrival":
            require(
                packet.arrival is None and packet.terminal is None and packet.current_queue is None,
                "endpoint arrival is duplicated or precedes fabric service",
            )
            require(
                packet.switch_count == len(packet.route) and packet.last_switch_end is not None,
                "packet does not traverse a complete frozen Clos route",
            )
            require(
                time == packet.last_switch_end + self.propagation == row["observed_at_ps"],
                "endpoint arrival violates final serialization plus propagation",
            )
            packet.arrival = time
            if packet.kind == "GAP_RESOLVED":
                self.resolutions.setdefault((packet.flow, packet.index), time)
            if packet.kind == "DATA":
                no_queue = (
                    packet.source_end
                    + packet.switch_count * packet.wire * self.ps_per_byte
                    + (packet.switch_count + 1) * self.propagation
                )
                require(
                    packet.eta == no_queue and time >= packet.eta,
                    "packet ETA or physical arrival floor is wrong",
                )
        elif event in {"endpoint_consumed", "fabric_drop"}:
            require(
                packet.terminal is None and packet.current_queue is None,
                "physical terminal is duplicated or leaves a queue",
            )
            require(
                time == row["observed_at_ps"], "physical terminal time differs from observation"
            )
            require(
                row["packet_index"] == packet.index
                if packet.kind == "DATA"
                else row["packet_index"] is None,
                "terminal logical packet index mismatch",
            )
            if event == "endpoint_consumed":
                require(
                    packet.arrival == time and not packet.dropped,
                    "consumed packet lacks endpoint arrival",
                )
                require(
                    row["detail"] == str(2 + 3 * packet.switch_count),
                    "terminal route element count mismatch",
                )
            else:
                require(
                    packet.dropped and packet.arrival is None and packet.kind == "DATA",
                    "fabric terminal lacks a buffer drop or drops protected control",
                )
                require(
                    row["detail"] == str(2 + 3 * packet.switch_count),
                    "drop route element count mismatch",
                )
            packet.terminal, packet.terminal_time = event, time
        elif event == "admission":
            self.admission(packet, row)
        elif event == "rx_schedule":
            self.rx_schedule(packet, row)
        elif event == "rx_complete":
            require(
                packet.rx_end == time and not packet.rx_completed,
                "receiver completion missing, duplicated, or mistimed",
            )
            packet.rx_completed = True
        elif event == "delivery":
            self.delivery(packet, row)
        elif event == "flow_complete":
            require(
                flow_id not in self.completed and packet.delivery == time,
                "flow completion is duplicated or lacks final delivery",
            )
            require(
                self.delivered[flow_id] == self.flows[flow_id]["total_data_packets"],
                "flow completion precedes complete packet ledger",
            )
            require(
                row["trigger_lifecycle_id"] == packet.delivery_trigger,
                "flow completion trigger disagrees with actual delivery cause",
            )
            require(
                self.previous_record is not None
                and self.previous_record[0] == "events"
                and self.previous_record[1]["event"] == "delivery"
                and self.previous_record[1]["lifecycle_id"] == packet.lifecycle,
                "flow completion is not the final logical delivery boundary",
            )
            require(
                time == self.completions[flow_id]["completion_time_ps"],
                "flow completion differs from completion CSV",
            )
            self.completed[flow_id] = packet.identity
        else:
            raise TraceAuditError(f"unknown runtime event {event}")

    def admission(self, packet, row):
        require(
            packet.kind == "DATA"
            and packet.arrival == row["time_ps"]
            and packet.terminal == "endpoint_consumed"
            and packet.admission is None,
            "DATA admission is duplicated or not bound to consumed arrival",
        )
        admission = row["detail"]
        require(
            admission
            in {
                "admitted",
                "admitted_late_retry",
                "late",
                "early",
                "overflow",
                "discard_duplicate",
            },
            "unknown receiver admission decision",
        )
        packet.admission = admission
        heap = self.ring[packet.destination]
        while heap and heap[0][0] <= packet.arrival:
            _, _, wire = heapq.heappop(heap)
            self.ring_bytes[packet.destination] -= wire
        if admission in {"admitted", "admitted_late_retry"}:
            late = packet.arrival - packet.eta > self.delta
            require(packet.arrival >= packet.eta, "receiver admitted an early packet")
            require(
                late == (admission == "admitted_late_retry"),
                "receiver late-admission classification mismatch",
            )
            require(not late or packet.attempt > 0, "late original admitted as a retry")
            edge = packet.arrival if late else packet.eta + self.delta
            release = (edge + self.tick - 1) // self.tick * self.tick
            require(
                row["logical_release_ps"] == release,
                "receiver release violates rounded holding rule",
            )
            require(
                self.ring_bytes[packet.destination] + packet.wire <= self.ring_capacity,
                "Ring-CAM admission exceeds capacity",
            )
            self.ring_bytes[packet.destination] += packet.wire
            self.ring_peak[packet.destination] = max(
                self.ring_peak[packet.destination], self.ring_bytes[packet.destination]
            )
            heapq.heappush(heap, (release, packet.lifecycle, packet.wire))
            require(
                (packet.flow, packet.index) not in self.accepted_logical,
                "receiver admitted two physical copies of one logical packet",
            )
            self.accepted_logical.add((packet.flow, packet.index))
            packet.release = release
        else:
            require(row["logical_release_ps"] is None, "rejected DATA retains receiver release")
            if admission == "late":
                require(
                    packet.arrival - packet.eta > self.delta,
                    "late rejection inside accepted receiver window",
                )
                self.rejected_attempt[(packet.flow, packet.index)] = packet.attempt
            elif admission == "early":
                require(packet.arrival < packet.eta, "early rejection after ETA")
            elif admission == "overflow":
                require(
                    self.ring_bytes[packet.destination] + packet.wire > self.ring_capacity,
                    "receiver overflow without full Ring-CAM",
                )
                self.rejected_attempt[(packet.flow, packet.index)] = packet.attempt
            elif admission == "discard_duplicate":
                key = packet.flow, packet.index
                competing = key in self.accepted_logical
                require(
                    competing or self.rejected_attempt.get(key, -1) >= packet.attempt,
                    "duplicate rejection lacks an accepted or previously failed logical copy",
                )

    def rx_schedule(self, packet, row):
        require(
            packet.release is not None and packet.rx_start is None,
            "receiver service is duplicated or lacks admission",
        )
        require(
            row["logical_release_ps"] == row["time_ps"] == packet.release,
            "receiver service changed logical release",
        )
        predecessor = self.previous_rx.get(packet.destination)
        require(
            row["predecessor_lifecycle_id"] == (predecessor.lifecycle if predecessor else 0),
            "receiver serialization predecessor is missing or misbound",
        )
        previous_end = predecessor.rx_end if predecessor else 0
        start = max(packet.release, previous_end)
        require(
            row["service_start_ps"] == start
            and row["service_end_ps"] == start + packet.wire * self.ps_per_byte,
            "receiver service violates eligibility, predecessor, or wire duration",
        )
        packet.rx_start, packet.rx_end = row["service_start_ps"], row["service_end_ps"]
        packet.rx_predecessor = row["predecessor_lifecycle_id"]
        self.previous_rx[packet.destination] = packet

    def delivery(self, packet, row):
        require(
            packet.rx_completed and packet.delivery is None,
            "logical delivery precedes RX completion or is duplicated",
        )
        previous = self.previous_delivery.get(packet.flow)
        require(
            row["predecessor_lifecycle_id"] == (previous.lifecycle if previous else 0),
            "logical delivery predecessor missing or misbound",
        )
        require(
            packet.index == self.delivered[packet.flow],
            "logical DATA delivery skips or repeats an index",
        )
        trigger_id = self.lifecycles.get(row["trigger_lifecycle_id"])
        require(trigger_id is not None, "delivery trigger references unknown physical packet")
        trigger = self.packets[trigger_id]
        prior = self.previous_record
        require(
            prior is not None
            and prior[0] == "events"
            and (
                (
                    prior[1]["event"] == "rx_complete"
                    and prior[1]["lifecycle_id"] == trigger.lifecycle
                )
                or (
                    prior[1]["event"] == "delivery"
                    and prior[1]["flow_id"] == packet.flow
                    and prior[1]["trigger_lifecycle_id"] == trigger.lifecycle
                )
            ),
            "delivery trigger is not the immediately realized RX completion chain",
        )
        require(
            trigger.flow == packet.flow
            and trigger.rx_completed
            and trigger.rx_end <= row["time_ps"],
            "delivery trigger is not a completed receiver dependency",
        )
        require(
            row["time_ps"] == max(packet.rx_end, row["observed_at_ps"]),
            "delivery visibility time disagrees with authoritative observation",
        )
        require(
            trigger.rx_end >= packet.rx_end
            or (previous is not None and previous.delivery == row["time_ps"]),
            "delivery trigger cannot release this ready packet",
        )
        packet.delivery = row["time_ps"]
        packet.delivery_trigger, packet.delivery_predecessor = (
            row["trigger_lifecycle_id"],
            row["predecessor_lifecycle_id"],
        )
        self.previous_delivery[packet.flow] = packet
        self.delivered[packet.flow] += 1
        self.delivered_bytes[packet.flow] += packet.payload
        self.delivered_wire[packet.flow] += packet.wire

    def retry(self, row):
        require(
            row["lifecycle_id"] == 0 and row["packet_id"] is None,
            "retry authorization invents a physical packet",
        )
        flow_id, index, attempt = row["flow_id"], row["packet_index"], row["attempt"]
        require(
            index is not None and attempt is not None and (flow_id, index) in self.originals,
            "retry authorization lacks original logical DATA",
        )
        require(
            attempt == self.highest_authorized[(flow_id, index)] + 1
            and attempt <= self.max_retries,
            "retry authorization skips, duplicates, or exceeds an attempt",
        )
        require(
            (flow_id, index) not in self.resolutions,
            "retry authorization follows terminal gap resolution",
        )
        detail = row["detail"]
        if detail == "gap_nack":
            identity = self.lifecycles.get(row["trigger_lifecycle_id"])
            require(identity is not None, "retry authorization lacks control trigger")
            control = self.packets[identity]
            require(
                (control.kind, control.flow, control.index, control.control_attempt)
                == ("GAP_NACK", flow_id, index, attempt)
                and control.arrival == row["time_ps"],
                "retry authorization does not bind the arriving GAP_NACK",
            )
            require(
                row["origin_attempt"] is None and row["deadline_ps"] is None,
                "GAP_NACK authorization carries inactive timer metadata",
            )
        elif detail in {"probe_timeout", "legacy_timeout"}:
            require(
                row["trigger_lifecycle_id"] == 0
                and row["origin_attempt"] == attempt - 1
                and row["deadline_ps"] == row["time_ps"],
                "retry timeout origin or deadline mismatch",
            )
            require(
                (flow_id, index, attempt - 1) in self.attempts,
                "retry timeout has no prior transmitted attempt",
            )
            origin = self.packets[self.attempt_packet[(flow_id, index, attempt - 1)]]
            require(attempt > 1, "original DATA cannot trigger a sender retry timer")
            interval = (
                40_000_000 * 2 ** (attempt - 2) if detail == "probe_timeout" else 50_000_000_000
            )
            require(
                row["time_ps"] == origin.source_end + interval,
                "retry deadline disagrees with transmitted origin and frozen interval",
            )
        else:
            raise TraceAuditError("unknown retry authorization cause")
        self.highest_authorized[(flow_id, index)] = attempt
        self.authorizations[(flow_id, index, attempt)] = {
            "time_ps": row["time_ps"],
            "cause": detail,
            "trigger_lifecycle_id": row["trigger_lifecycle_id"],
            "origin_attempt": row["origin_attempt"],
            "deadline_ps": row["deadline_ps"],
            "attempt": attempt,
        }

    def finish(self, manifest):
        require(not self.creations, "unbound packet creation remains at trace completion")
        require(
            set(self.flows) == set(self.completions) == set(self.completed),
            "flow inventory is incomplete",
        )
        for queue in self.queues.values():
            require(
                not queue.buffered
                and queue.active is None
                and queue.buffered_bytes == queue.reserve_bytes == 0,
                "physical quiescence retains a queued or serializing packet",
            )
        require(
            not any(self.switch_bytes.values()) and not any(self.switch_reserve.values()),
            "switch byte ledger is not drained",
        )
        for packet in self.packets.values():
            require(
                packet.terminal is not None, "physical packet is missing its terminal observation"
            )
            if packet.kind == "DATA" and packet.arrival is not None:
                require(packet.admission is not None, "arrived DATA lacks admission observation")
            if packet.release is not None:
                require(
                    packet.rx_completed and packet.delivery is not None,
                    "admitted DATA lacks receiver service or logical delivery",
                )
        for flow_id, flow in self.flows.items():
            require(
                self.delivered_bytes[flow_id] == flow["payload_bytes"]
                and self.delivered_wire[flow_id] == flow["total_wire_bytes"],
                "delivered packet byte totals disagree with input",
            )
            require(
                all(
                    (flow_id, index) in self.originals
                    for index in range(flow["total_data_packets"])
                ),
                "input packetization lacks an original DATA attempt",
            )
        require(
            manifest["physical_quiescence_time_ps"] >= self.latest_observed,
            "trace verification boundary precedes observations",
        )
        pp = [
            self.pp_summary(flow_id)
            for flow_id, flow in self.flows.items()
            if flow["class"] == "pp"
        ]
        require(bool(pp), "trace contains no pipeline request")
        egresses = []
        parts = [0, 0, 0, 0]
        for key, queue in sorted(self.queues.items()):
            parts = [a + b for a, b in zip(parts, queue.pp_parts)]
            egresses.append(
                {
                    "switch_type": key[0],
                    "switch_id": key[1],
                    "egress_id": key[2],
                    "enqueues": queue.enqueues,
                    "serializations": queue.services,
                    "drops": queue.drops,
                    "buffer_high_watermark_bytes": queue.peak_buffered_bytes,
                    "all_packet_queue_wait_ps": queue.all_wait,
                    "pp_data_queue_wait_ps": queue.pp_wait,
                    "pp_service_ahead_ps": dict(zip(PARTITIONS, queue.pp_parts)),
                    "pp_ep_data_bound_ps": queue.pp_bound,
                }
            )
        return {
            "schema": "pp-queue-audit-v1",
            "status": "valid",
            "flow_count": len(self.flows),
            "data_packet_count": self.counts["data_packets"],
            "control_packet_count": self.counts["control_packets"],
            "physical_quiescence_time_ps": manifest["physical_quiescence_time_ps"],
            "pp_flows": sorted(pp, key=lambda row: row["source_rank"]),
            "queue_work": {
                "scope": "all physical PP DATA attempts and switch visits",
                "total_queue_wait_ps": sum(parts),
                **{name + "_service_ahead_ps": value for name, value in zip(PARTITIONS, parts)},
            },
            "egresses": egresses,
            "ring_high_watermark_bytes": dict(sorted(self.ring_peak.items())),
        }

    def pp_summary(self, flow_id):
        flow = self.flows[flow_id]
        final = self.packets[self.completed[flow_id]]
        trigger = self.packets[self.lifecycles[final.delivery_trigger]]
        packets = [
            packet
            for packet in self.packets.values()
            if packet.flow == flow_id and packet.kind == "DATA"
        ]
        rows = []
        for packet in sorted(packets, key=lambda p: (p.index, p.attempt)):
            row = {
                "packet_id": packet.identity,
                "lifecycle_id": packet.lifecycle,
                "packet_index": packet.index,
                "attempt": packet.attempt,
                "payload_bytes": packet.payload,
                "wire_bytes": packet.wire,
                "source_start_ps": packet.source_start,
                "source_end_ps": packet.source_end,
                "eta_ps": packet.eta,
                "arrival_ps": packet.arrival,
                "admission": packet.admission,
                "logical_release_ps": packet.release,
                "rx_service_start_ps": packet.rx_start,
                "rx_service_end_ps": packet.rx_end,
                "delivery_ps": packet.delivery,
                "terminal": packet.terminal,
                "ring_holding_ps": None
                if packet.release is None
                else packet.release - packet.arrival,
                "rx_queue_wait_ps": None
                if packet.rx_start is None
                else packet.rx_start - packet.release,
                "ordering_delay_ps": None
                if packet.delivery is None
                else packet.delivery - packet.rx_end,
                "rx_predecessor_lifecycle_id": packet.rx_predecessor,
                "delivery_predecessor_lifecycle_id": packet.delivery_predecessor,
                "delivery_trigger_lifecycle_id": packet.delivery_trigger,
                "retry_authorization": packet.authorization,
                "visits": packet.visits,
            }
            rows.append(row)
        path = {
            "dispatch_offset_ps": trigger.source_start - flow["start_time_ps"],
            "source_service_ps": trigger.source_end - trigger.source_start,
            "switch_queue_wait_ps": sum(v["queue_wait_ps"] for v in trigger.visits),
            "switch_service_ps": sum(v["service_ps"] for v in trigger.visits),
            "propagation_ps": (trigger.switch_count + 1) * self.propagation,
            "ring_holding_ps": trigger.release - trigger.arrival,
            "rx_queue_wait_ps": trigger.rx_start - trigger.release,
            "rx_service_ps": trigger.rx_end - trigger.rx_start,
            "delivery_visibility_ps": final.delivery - trigger.rx_end,
        }
        require(
            all(value >= 0 for value in path.values()),
            "completion causal path contains negative time",
        )
        path["total_ps"] = sum(path.values())
        require(
            path["total_ps"] == self.completions[flow_id]["fct_ps"],
            "trigger packet timeline does not reconstruct flow completion",
        )
        visits = [visit for packet in packets for visit in packet.visits]
        return {
            "flow_id": flow_id,
            "source_rank": flow["source_rank"],
            "destination_rank": flow["destination_rank"],
            "start_time_ps": flow["start_time_ps"],
            "completion_time_ps": final.delivery,
            "fct_ps": self.completions[flow_id]["fct_ps"],
            "completion_packet_lifecycle_id": final.lifecycle,
            "completion_trigger_lifecycle_id": trigger.lifecycle,
            "trigger_packet_timeline": path,
            "completion_dependency_witness": self.dependency_witness(final, trigger),
            "all_visit_queue_wait_ps": sum(v["queue_wait_ps"] for v in visits),
            "ep_data_service_ahead_ps": sum(v["service_ahead_ps"]["ep_data"] for v in visits),
            "ep_data_bound_ps": sum(v["ep_data_bound_ps"] for v in visits),
            "packets": rows,
        }

    @staticmethod
    def selected_max_inputs(left, right, left_name, right_name):
        value = max(left, right)
        return [
            name
            for name, candidate in ((left_name, left), (right_name, right))
            if candidate == value
        ]

    def dependency_witness(self, final, trigger):
        """Select receiver predecessors without promoting packet timelines to a DAG.

        Source eligibility and recursively selected competing switch service
        are not fully observed by this interface. The witness names those
        limits instead of assigning all time differences to tagged queues.
        """
        nodes = []
        packet = trigger
        seen = set()
        while packet is not None:
            require(packet.lifecycle not in seen, "receiver dependency witness contains a cycle")
            seen.add(packet.lifecycle)
            previous = (
                self.packets[self.lifecycles[packet.rx_predecessor]]
                if packet.rx_predecessor
                else None
            )
            previous_end = previous.rx_end if previous else 0
            selected = self.selected_max_inputs(
                packet.release, previous_end, "ring_release", "previous_rx_end"
            )
            late = packet.admission == "admitted_late_retry"
            base = packet.arrival if late else packet.eta + self.delta
            nodes.append(
                {
                    "lifecycle_id": packet.lifecycle,
                    "flow_id": packet.flow,
                    "packet_index": packet.index,
                    "attempt": packet.attempt,
                    "rx_service_start_ps": packet.rx_start,
                    "rx_service_end_ps": packet.rx_end,
                    "rx_wire_service_ps": packet.wire * self.ps_per_byte,
                    "ring_release_ps": packet.release,
                    "previous_rx_end_ps": previous_end,
                    "previous_rx_lifecycle_id": packet.rx_predecessor,
                    "selected_rx_start_inputs": selected,
                    "ring_release_rule": "arrival_plus_rounding"
                    if late
                    else "eta_plus_delta_plus_rounding",
                    "ring_release_base_ps": base,
                    "release_rounding_ps": packet.release - base,
                    "actual_arrival_ps": packet.arrival,
                    "eta_ps": packet.eta,
                    "arrival_selected_by_release": late,
                    "arrival_delay_masked_by_release_ps": 0
                    if late
                    else packet.arrival - packet.eta,
                    "source_boundary": {
                        "source_start_ps": packet.source_start,
                        "source_end_ps": packet.source_end,
                        "dispatch_offset_ps": packet.source_start
                        - self.flows[packet.flow]["start_time_ps"],
                        "eligibility_observed": False,
                        "retry_authorization": packet.authorization,
                    },
                }
            )
            packet = previous if "previous_rx_end" in selected and previous is not None else None
        return {
            "delivery_lifecycle_id": final.lifecycle,
            "delivery_trigger_lifecycle_id": trigger.lifecycle,
            "ordering_predecessor_lifecycle_id": final.delivery_predecessor,
            "trigger_rx_completed_at_ps": trigger.rx_end,
            "delivery_visible_at_ps": final.delivery,
            "delivery_visibility_delay_ps": final.delivery - trigger.rx_end,
            "receiver_selected_predecessors": nodes,
            "complete_resource_dependency_graph": False,
            "limitations": [
                "source pacing and eligibility remain an opaque dispatch boundary",
                "competing switch-service dependencies are not recursively expanded",
                "packet queue-work sums do not measure marginal latency",
            ],
        }


def audit_trace(
    trace_dir: str | Path,
    *,
    completion_rows: Iterable[Mapping],
    endpoint_by_rank: Sequence[int],
    spine_count: int,
    delta_ps: int,
    tick_ps: int = 16_000,
    link_rate_bps: int = 400_000_000_000,
    propagation_ps: int = 1_000_000,
    switch_buffer_bytes: int = 1_048_576,
    control_headroom_bytes: int = 131_072,
    ring_capacity_bytes: int = 1_048_576,
    max_retries: int = 8,
    packet_payload_bytes: int = 4096,
    data_header_bytes: int = 64,
    control_wire_bytes: int = 64,
) -> dict:
    """Return compact PP evidence only after every fatal trace guard succeeds.

    Completion rows use ``dataclasses.asdict(FlowCompletion)`` before endpoint
    unprojection. The explicit endpoint permutation classifies physical flows.
    CSV streams are merged in observation order; retained state grows with
    packet identities and active queues, never with the number of queue rows.
    """
    directory = Path(trace_dir)
    require(
        directory.is_dir() and not directory.name.endswith(".tmp"),
        "trace is not an atomically finalized directory",
    )
    require(
        {path.name for path in directory.iterdir()}
        == {f"{name}.csv" for name in HEADERS} | {"manifest.csv"},
        "trace directory has missing or unexpected tables",
    )
    manifests = list(_rows(directory / "manifest.csv", "manifest", MANIFEST_HEADER))
    require(len(manifests) == 1, "trace manifest must contain exactly one final record")
    manifest = manifests[0]
    require(
        (manifest["status"], manifest["finalized"], manifest["physical_quiescence"])
        == ("complete", "true", "verified"),
        "trace manifest is incomplete or not physically quiescent",
    )
    audit = Audit(
        completion_rows,
        endpoint_by_rank,
        spine_count,
        delta_ps,
        tick_ps,
        link_rate_bps,
        propagation_ps,
        switch_buffer_bytes,
        control_headroom_bytes,
        ring_capacity_bytes,
        max_retries,
        packet_payload_bytes,
        data_header_bytes,
        control_wire_bytes,
    )
    counts = Counter()
    for table, row in _merged_rows(directory, counts):
        audit.latest_observed = row["observed_at_ps"]
        {
            "flows": audit.flow,
            "packets": audit.packet,
            "queues": audit.queue,
            "events": audit.event,
        }[table](row)
        audit.previous_record = table, row
    total = sum(counts.values())
    require(
        (manifest["sequence_first"], manifest["sequence_last"], manifest["observation_count"])
        == (1 if total else 0, total, total),
        "manifest observation totals disagree with complete streams",
    )
    for name in HEADERS:
        require(manifest[f"{name}_rows"] == counts[name], f"manifest {name} row count mismatch")
    require(
        (manifest["flow_count"], manifest["data_packet_count"], manifest["control_packet_count"])
        == (len(audit.flows), audit.counts["data_packets"], audit.counts["control_packets"]),
        "manifest physical packet or flow totals mismatch",
    )
    result = audit.finish(manifest)
    result["observation_counts"] = {**{name: counts[name] for name in HEADERS}, "total": total}
    return result
