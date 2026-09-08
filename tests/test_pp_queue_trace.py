"""Synthetic physical traces exercise fatal attribution guards without simulations."""

import copy
from collections import Counter

import pytest

from examples.pp_rail_contention_v2 import trace_analysis as trace


class Fixture:
    def __init__(self, delta=16_000):
        self.delta = delta
        self.rows = []
        self.flows = {}
        self.packets = {}
        self.completions = []
        self.endpoint_by_rank = tuple(range(64))
        self.visit_counts = Counter()

    def add(self, table, observed, **values):
        row = dict.fromkeys(trace.HEADERS[table].split(","), None)
        row.update(schema=trace.SCHEMA, observed_at_ps=observed, **values)
        if table == "events":
            for name, value in {
                "detail": "",
                "trigger_lifecycle_id": 0,
                "predecessor_lifecycle_id": 0,
            }.items():
                if row[name] is None:
                    row[name] = value
        self.rows.append((observed, len(self.rows), table, row))
        return row

    def flow(self, identity=1, source=0, destination=8, payload=8192):
        count = (payload + 4095) // 4096
        self.flows[identity] = self.add(
            "flows",
            0,
            flow_id=identity,
            packet_flow_id=identity + 100,
            source=source,
            destination=destination,
            tag=7,
            start_time_ps=0,
            payload_bytes=payload,
            total_wire_bytes=payload + count * 64,
            total_data_packets=count,
        )

    def packet(self, identity, flow=1, index=0, source_start=0, hops=1, attempt=0):
        declared = self.flows[flow]
        payload = min(4096, declared["payload_bytes"] - 4096 * index)
        wire = payload + 64
        source_end = source_start + wire * 20
        self.add(
            "events",
            source_end,
            event="packet_created",
            time_ps=source_end,
            flow_id=flow,
            lifecycle_id=identity,
            packet_id=identity,
            packet_index=index,
            detail="0",
        )
        packet = self.add(
            "packets",
            source_end,
            packet_id=identity,
            lifecycle_id=identity,
            flow_id=flow,
            packet_flow_id=declared["packet_flow_id"],
            kind="DATA",
            source=declared["source"],
            destination=declared["destination"],
            path_id=0,
            wire_bytes=wire,
            packet_index=index,
            attempt=attempt,
            payload_offset=index * 4096,
            payload_bytes=payload,
            header_bytes=64,
            eta_ps=source_end + wire * 20 * hops + (hops + 1) * 1_000_000,
            source_start_ps=source_start,
            source_end_ps=source_end,
            duplicate_test_copy=0,
            total_payload_bytes=declared["payload_bytes"],
            total_wire_bytes=declared["total_wire_bytes"],
            total_data_packets=declared["total_data_packets"],
        )
        self.packets[identity] = packet
        return source_end

    def visit(self, packet_id, enqueue, start):
        packet = self.packets[packet_id]
        source_leaf, source_port = divmod(packet["source"], 8)
        destination_leaf, destination_port = divmod(packet["destination"], 8)
        routes = (
            ((1, source_leaf, source_port, destination_port),)
            if source_leaf == destination_leaf
            else (
                (1, source_leaf, source_port, 8),
                (2, 0, source_leaf, destination_leaf),
                (1, destination_leaf, 8, destination_port),
            )
        )
        switch_type, switch_id, ingress, egress = routes[self.visit_counts[packet_id]]
        self.visit_counts[packet_id] += 1
        end = start + packet["wire_bytes"] * 20
        for transition, time in (
            ("enqueue", enqueue),
            ("service_start", start),
            ("service_end", end),
        ):
            self.add(
                "queues",
                time,
                transition=transition,
                time_ps=time,
                switch_type=switch_type,
                switch_id=switch_id,
                egress_id=egress,
                ingress_id=ingress,
                priority=0 if packet["kind"] == "DATA" else 2,
                packet_flow_id=packet["packet_flow_id"],
                packet_id=packet_id,
                lifecycle_id=packet_id,
                flow_id=packet["flow_id"],
                kind=packet["kind"],
                wire_bytes=packet["wire_bytes"],
            )
        return end

    def runtime(self, packet_id, event, time, **values):
        packet = self.packets[packet_id]
        return self.add(
            "events",
            time,
            event=event,
            time_ps=time,
            flow_id=packet["flow_id"],
            lifecycle_id=packet_id,
            packet_id=packet_id,
            packet_index=packet["packet_index"] if packet["kind"] == "DATA" else None,
            attempt=packet["attempt"],
            **values,
        )

    def arrived(self, packet_id, arrival, hops=1):
        self.runtime(packet_id, "endpoint_arrival", arrival)
        row = self.runtime(packet_id, "endpoint_consumed", arrival, detail=str(2 + 3 * hops))
        row["attempt"] = None
        eta = self.packets[packet_id]["eta_ps"]
        release = (eta + self.delta + 15_999) // 16_000 * 16_000
        self.runtime(packet_id, "admission", arrival, detail="admitted", logical_release_ps=release)
        return release

    def service(self, packet_id, release, start, previous=0):
        end = start + self.packets[packet_id]["wire_bytes"] * 20
        self.runtime(
            packet_id,
            "rx_schedule",
            release,
            logical_release_ps=release,
            service_start_ps=start,
            service_end_ps=end,
            predecessor_lifecycle_id=previous,
        )
        self.runtime(packet_id, "rx_complete", end)
        return end

    def delivery(self, packet_id, time, *, previous=0, trigger=None, final=False):
        trigger = trigger or packet_id
        self.runtime(
            packet_id,
            "delivery",
            time,
            predecessor_lifecycle_id=previous,
            trigger_lifecycle_id=trigger,
        )
        if final:
            self.runtime(packet_id, "flow_complete", time, trigger_lifecycle_id=trigger)
            flow = self.flows[self.packets[packet_id]["flow_id"]]
            self.completions.append(
                {
                    key: flow[key]
                    for key in (
                        "flow_id",
                        "source",
                        "destination",
                        "tag",
                        "payload_bytes",
                        "start_time_ps",
                    )
                }
                | {"completion_time_ps": time, "fct_ps": time}
            )

    def ordered(self):
        rows = sorted(copy.deepcopy(self.rows))
        buffered, active, shared = Counter(), Counter(), Counter()
        for sequence, (_, _, table, row) in enumerate(rows, 1):
            row["sequence"] = sequence
            if table != "queues":
                continue
            switch = row["switch_type"], row["switch_id"]
            queue = *switch, row["egress_id"]
            if row["transition"] == "enqueue":
                buffered[queue] += row["wire_bytes"]
                shared[switch] += row["wire_bytes"]
            elif row["transition"] == "service_start":
                buffered[queue] -= row["wire_bytes"]
                shared[switch] -= row["wire_bytes"]
                active[queue] = row["wire_bytes"]
            elif row["transition"] == "service_end":
                active[queue] = 0
            row.update(
                buffered_bytes=buffered[queue],
                in_service_bytes=active[queue],
                backlog_bytes=buffered[queue] + active[queue],
                shared_buffer_bytes=shared[switch],
            )
        return [(table, row) for _, _, table, row in rows]

    def write(self, directory, rows=None):
        directory.mkdir()
        rows = self.ordered() if rows is None else rows
        for table, header in trace.HEADERS.items():
            names = header.split(",")
            lines = [header]
            for name, row in rows:
                if name == table:
                    lines.append(
                        ",".join("" if row[field] is None else str(row[field]) for field in names)
                    )
            (directory / f"{table}.csv").write_text(
                "\n".join(lines) + "\n", encoding="ascii", newline=""
            )
        counts = Counter(table for table, _ in rows)
        packet_counts = Counter(row["kind"] for table, row in rows if table == "packets")
        last = max(row["observed_at_ps"] for _, row in rows)
        values = [
            trace.SCHEMA,
            "complete",
            "true",
            "verified",
            1,
            len(rows),
            len(rows),
            *(counts[name] for name in trace.HEADERS),
            counts["flows"],
            packet_counts["DATA"],
            sum(packet_counts.values()) - packet_counts["DATA"],
            last,
        ]
        (directory / "manifest.csv").write_text(
            trace.MANIFEST_HEADER + "\n" + ",".join(map(str, values)) + "\n",
            encoding="ascii",
            newline="",
        )
        return directory

    def audit(self, directory, **kwargs):
        return trace.audit_trace(
            directory,
            completion_rows=self.completions,
            endpoint_by_rank=self.endpoint_by_rank,
            spine_count=2,
            delta_ps=self.delta,
            **kwargs,
        )


def unloaded(delta=16_000):
    fixture = Fixture(delta=delta)
    previous_end = 0
    fixture.endpoint_by_rank = tuple((rank % 8) * 8 + rank // 8 for rank in range(64))
    fixture.flow(destination=1)
    for identity, index, source_start, start in ((1, 0, 0, 1_083_200), (2, 1, 83_200, 1_166_400)):
        fixture.packet(identity, index=index, source_start=source_start)
        finish = fixture.visit(identity, start, start)
        release = fixture.arrived(identity, finish + 1_000_000)
        end = fixture.service(identity, release, max(release, previous_end), previous=identity - 1)
        fixture.delivery(identity, end, previous=identity - 1, final=identity == 2)
        previous_end = end
    return fixture


def contended():
    fixture = Fixture(delta=1_000_000)
    fixture.flow(1, 0, 8, 4096)
    fixture.flow(2, 1, 9, 4096)
    fixture.flow(3, 2, 10, 4096)
    # Two earlier expert packets occupy one selected 400G uplink. The tagged
    # packet sees 73,200 ps of residual work plus one complete 83,200 ps packet.
    for identity, source_start, enqueue, start in (
        (2, 90_000, 1_173_200, 1_173_200),
        (3, 95_000, 1_178_200, 1_256_400),
        (1, 100_000, 1_183_200, 1_339_600),
    ):
        fixture.packet(identity, flow=identity, source_start=source_start, hops=3)
        end = fixture.visit(identity, enqueue, start)
        end = fixture.visit(identity, end + 1_000_000, end + 1_000_000)
        end = fixture.visit(identity, end + 1_000_000, end + 1_000_000)
        release = fixture.arrived(identity, end + 1_000_000, hops=3)
        finish = fixture.service(identity, release, release)
        fixture.delivery(identity, finish, final=True)
    return fixture


def test_unloaded_exact_completion_and_receiver_predecessor(tmp_path):
    fixture = unloaded()
    result = fixture.audit(fixture.write(tmp_path / "trace"))
    flow = result["pp_flows"][0]
    assert flow["fct_ps"] == flow["trigger_packet_timeline"]["total_ps"] == 2_358_400
    assert flow["trigger_packet_timeline"]["rx_queue_wait_ps"] == 3_200
    assert flow["all_visit_queue_wait_ps"] == 0
    assert flow["packets"][1]["rx_predecessor_lifecycle_id"] == 1
    witness = flow["completion_dependency_witness"]["receiver_selected_predecessors"]
    assert witness[0]["selected_rx_start_inputs"] == ["previous_rx_end"]
    assert witness[1]["selected_rx_start_inputs"] == ["ring_release"]
    assert result["observation_counts"]["packets"] == 2


def test_independent_tagged_wire_bound_includes_only_remaining_active_service(tmp_path):
    fixture = contended()
    result = fixture.audit(fixture.write(tmp_path / "trace"))
    flow = result["pp_flows"][0]
    first = flow["packets"][0]["visits"][0]
    assert first["earlier_ep_data_wire_bytes"] == 4_160
    assert first["active_ep_data_residual_ps"] == 73_200
    assert first["ep_data_bound_ps"] == first["queue_wait_ps"] == 156_400
    assert first["service_ahead_ps"] == {
        "ep_data": 156_400,
        "ep_control": 0,
        "pp_or_other_control": 0,
        "other_data": 0,
    }
    assert result["queue_work"]["ep_data_service_ahead_ps"] == 156_400
    intersections = first["competing_service_intersections"]
    assert [(row["lifecycle_id"], row["overlap_ps"]) for row in intersections] == [
        (2, 73_200),
        (3, 83_200),
    ]
    # The expert packet's own waiting time is retained as separate work and
    # never becomes evidence of expert service ahead of the tagged pipeline.
    assert sum(row["all_packet_queue_wait_ps"] for row in result["egresses"]) > 156_400
    assert flow["trigger_packet_timeline"]["total_ps"] == flow["fct_ps"]


CORRUPTIONS = [
    ("packets", None, "flow_id", 999, "unknown flow"),
    ("packets", None, "packet_flow_id", 999, "PacketFlow"),
    ("packets", None, "lifecycle_id", 999, "creation identity"),
    ("packets", None, "payload_offset", 1, "packetization"),
    ("packets", None, "wire_bytes", 4000, "source serialization"),
    ("packets", None, "total_wire_bytes", 1, "final ledger"),
    ("packets", None, "duplicate_test_copy", 1, "test-copy"),
    ("packets", None, "requested_attempt", 1, "inactive"),
    ("queues", "enqueue", "packet_id", 999, "unknown physical"),
    ("queues", "enqueue", "wire_bytes", 4161, "metadata mismatch"),
    ("queues", "enqueue", "buffered_bytes", 1, "occupancy"),
    ("queues", "enqueue", "shared_buffer_bytes", 1, "occupancy"),
    ("queues", "enqueue", "in_service_bytes", 1, "occupancy"),
    ("queues", "enqueue", "backlog_bytes", 1, "occupancy"),
    ("queues", "service_start", "ingress_id", 99, "route"),
    ("queues", "service_start", "priority", 2, "priority"),
    ("queues", "service_end", "in_service_bytes", 4160, "occupancy"),
    ("queues", "service_end", "time_ps", 1_166_399, "physical/observation"),
    ("events", "packet_created", "detail", "2", "prior route"),
    ("events", "endpoint_consumed", "detail", "8", "route element"),
    ("events", "admission", "logical_release_ps", 2_208_000, "rounded holding"),
    ("events", "admission", "detail", "admitted_late_retry", "classification"),
    ("events", "rx_schedule", "predecessor_lifecycle_id", 999, "predecessor"),
    ("events", "rx_schedule", "service_start_ps", 2_192_001, "receiver service"),
    ("events", "rx_schedule", "service_end_ps", 2_275_199, "receiver service"),
    ("events", "rx_complete", "time_ps", 2_275_199, "receiver completion"),
    ("events", "delivery", "trigger_lifecycle_id", 999, "trigger"),
    ("events", "delivery", "predecessor_lifecycle_id", 999, "predecessor"),
    ("events", "flow_complete", "trigger_lifecycle_id", 1, "trigger"),
]


@pytest.mark.parametrize("table,event,field,value,message", CORRUPTIONS)
def test_independent_corruptions_are_fatal(tmp_path, table, event, field, value, message):
    fixture = unloaded()
    rows = fixture.ordered()
    row = next(
        row
        for name, row in rows
        if name == table and (event is None or row.get("event", row.get("transition")) == event)
    )
    row[field] = value
    with pytest.raises(trace.TraceAuditError, match=message):
        fixture.audit(fixture.write(tmp_path / "trace", rows))


@pytest.mark.parametrize(
    "table,event",
    [
        ("events", "endpoint_consumed"),
        ("events", "rx_complete"),
        ("events", "flow_complete"),
        ("queues", "service_end"),
    ],
)
def test_missing_terminal_or_service_cannot_be_hidden_by_recounting_manifest(
    tmp_path, table, event
):
    fixture = unloaded()
    rows = fixture.ordered()
    index = next(
        index
        for index, (name, row) in enumerate(rows)
        if name == table and row.get("event", row.get("transition")) == event
    )
    rows.pop(index)
    for index, (_, row) in enumerate(rows, 1):
        row["sequence"] = index
    with pytest.raises(trace.TraceAuditError):
        fixture.audit(fixture.write(tmp_path / "trace", rows))


@pytest.mark.parametrize(
    "name,value",
    [
        ("status", "partial"),
        ("finalized", "false"),
        ("physical_quiescence", "pending"),
        ("observation_count", "1"),
        ("packets_rows", "0"),
        ("data_packet_count", "0"),
        ("physical_quiescence_time_ps", "0"),
    ],
)
def test_final_manifest_is_a_fatal_guard(tmp_path, name, value):
    fixture = unloaded()
    directory = fixture.write(tmp_path / "trace")
    path = directory / "manifest.csv"
    lines = path.read_text().splitlines()
    cells = lines[1].split(",")
    cells[lines[0].split(",").index(name)] = value
    path.write_text(lines[0] + "\n" + ",".join(cells) + "\n", newline="")
    with pytest.raises(trace.TraceAuditError):
        fixture.audit(directory)


@pytest.mark.parametrize(
    "edit", ["gap", "duplicate", "backwards_time", "blank_integer", "extra_column", "missing_lf"]
)
def test_stream_codec_rejects_ambiguous_or_incomplete_observation_order(tmp_path, edit):
    fixture = unloaded()
    rows = fixture.ordered()
    if edit == "gap":
        rows[2][1]["sequence"] += 1
    elif edit == "duplicate":
        rows[2][1]["sequence"] = rows[1][1]["sequence"]
    elif edit == "backwards_time":
        rows[3][1]["observed_at_ps"] = 0
    elif edit == "blank_integer":
        rows[2][1]["source_start_ps"] = None
    directory = fixture.write(tmp_path / "trace", rows)
    path = directory / "packets.csv"
    if edit == "extra_column":
        path.write_text(path.read_text().replace("\n", ",0\n", 1), newline="")
    elif edit == "missing_lf":
        path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(trace.TraceAuditError):
        fixture.audit(directory)


def test_endpoint_map_and_completion_csv_bind_native_identity(tmp_path):
    fixture = unloaded()
    directory = fixture.write(tmp_path / "trace")
    with pytest.raises(trace.TraceAuditError, match="bijective"):
        trace.audit_trace(
            directory,
            completion_rows=fixture.completions,
            endpoint_by_rank=(0, 0),
            spine_count=2,
            delta_ps=fixture.delta,
        )
    fixture.completions[0]["tag"] += 1
    with pytest.raises(trace.TraceAuditError, match="tag mismatch"):
        fixture.audit(directory)


def test_queue_capacity_is_checked_independently_of_snapshot_consistency(tmp_path):
    fixture = contended()
    with pytest.raises(trace.TraceAuditError, match="capacity"):
        fixture.audit(fixture.write(tmp_path / "trace"), switch_buffer_bytes=4160)


def test_ring_capacity_is_checked_against_residency(tmp_path):
    fixture = unloaded(delta=1_000_000)
    with pytest.raises(trace.TraceAuditError, match="Ring-CAM admission exceeds"):
        fixture.audit(fixture.write(tmp_path / "trace"), ring_capacity_bytes=4160)


def recovered_hole():
    fixture = Fixture(delta=1_000_000)
    fixture.flow(1, 0, 8, 8192)
    fixture.flow(2, 1, 9, 4096)
    fixture.flow(3, 2, 10, 4096)

    def route(identity, first_enqueue, first_start):
        end = fixture.visit(identity, first_enqueue, first_start)
        end = fixture.visit(identity, end + 1_000_000, end + 1_000_000)
        return fixture.visit(identity, end + 1_000_000, end + 1_000_000) + 1_000_000

    for identity, source_start, enqueue, start in (
        (2, 90_000, 1_173_200, 1_173_200),
        (3, 95_000, 1_178_200, 1_256_400),
    ):
        fixture.packet(identity, flow=identity, source_start=source_start, hops=3)
        arrival = route(identity, enqueue, start)
        release = fixture.arrived(identity, arrival, hops=3)
        end = fixture.service(identity, release, release)
        fixture.delivery(identity, end, final=True)
    fixture.packet(1, index=0, source_start=100_000, hops=3)
    fixture.add(
        "queues",
        1_183_200,
        transition="drop",
        time_ps=1_183_200,
        switch_type=1,
        switch_id=0,
        ingress_id=0,
        egress_id=8,
        priority=0,
        packet_flow_id=101,
        packet_id=1,
        lifecycle_id=1,
        flow_id=1,
        kind="DATA",
        wire_bytes=4160,
    )
    row = fixture.runtime(1, "fabric_drop", 1_183_200, detail="2")
    row["attempt"] = None
    fixture.packet(4, index=1, source_start=183_200, hops=3)
    arrival = route(4, 1_266_400, 1_339_600)
    release = fixture.arrived(4, arrival, hops=3)
    fixture.service(4, release, release)
    # The later logical packet exposes the hole. Its returned NACK is a
    # separate control packet with its own reverse-route physical service.
    source_end = release + 1_280
    fixture.add(
        "events",
        source_end,
        event="packet_created",
        time_ps=source_end,
        flow_id=1,
        lifecycle_id=6,
        packet_id=6,
        detail="0",
    )
    packet = fixture.add(
        "packets",
        source_end,
        packet_id=6,
        lifecycle_id=6,
        flow_id=1,
        packet_flow_id=101,
        kind="GAP_NACK",
        source=8,
        destination=0,
        path_id=0,
        wire_bytes=64,
        packet_index=0,
        payload_offset=0,
        payload_bytes=4096,
        header_bytes=64,
        source_start_ps=release,
        source_end_ps=source_end,
        duplicate_test_copy=0,
        requested_attempt=1,
    )
    fixture.packets[6] = packet
    nack_arrival = route(6, source_end + 1_000_000, source_end + 1_000_000)
    fixture.runtime(6, "endpoint_arrival", nack_arrival)
    fixture.runtime(6, "endpoint_consumed", nack_arrival, detail="11")
    fixture.add(
        "events",
        nack_arrival,
        event="retry_authorized",
        time_ps=nack_arrival,
        flow_id=1,
        lifecycle_id=0,
        packet_index=0,
        attempt=1,
        detail="gap_nack",
        trigger_lifecycle_id=6,
    )
    retry_end = fixture.packet(5, index=0, source_start=nack_arrival, hops=3, attempt=1)
    arrival = route(5, retry_end + 1_000_000, retry_end + 1_000_000)
    release = fixture.arrived(5, arrival, hops=3)
    end = fixture.service(5, release, release, previous=4)
    fixture.delivery(5, end, trigger=5)
    fixture.delivery(4, end, previous=5, trigger=5, final=True)
    return fixture


def test_earlier_recovered_hole_is_actual_completion_cause(tmp_path):
    fixture = recovered_hole()
    result = fixture.audit(fixture.write(tmp_path / "trace"), switch_buffer_bytes=4160)
    flow = result["pp_flows"][0]
    assert flow["completion_packet_lifecycle_id"] == 4
    assert flow["completion_trigger_lifecycle_id"] == 5
    assert flow["trigger_packet_timeline"]["total_ps"] == flow["fct_ps"]
    assert flow["trigger_packet_timeline"]["dispatch_offset_ps"] == 9_525_120
    later = next(packet for packet in flow["packets"] if packet["lifecycle_id"] == 4)
    assert later["ordering_delay_ps"] > 9_000_000
    retry = next(packet for packet in flow["packets"] if packet["lifecycle_id"] == 5)
    assert retry["retry_authorization"]["trigger_lifecycle_id"] == 6


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("trigger_lifecycle_id", 2, "arriving GAP_NACK"),
        ("attempt", 2, "attempt"),
        ("origin_attempt", 0, "inactive timer"),
    ],
)
def test_retry_authorization_is_bound_to_physical_cause(tmp_path, field, value, message):
    fixture = recovered_hole()
    rows = fixture.ordered()
    row = next(
        row for table, row in rows if table == "events" and row["event"] == "retry_authorized"
    )
    row[field] = value
    with pytest.raises(trace.TraceAuditError, match=message):
        fixture.audit(fixture.write(tmp_path / "trace", rows), switch_buffer_bytes=4160)


def test_identical_release_absorbs_independently_measured_queue_service(tmp_path):
    loaded = contended()
    baseline = Fixture(delta=loaded.delta)
    baseline.flow(1, 0, 8, 4096)
    end = baseline.packet(1, source_start=100_000, hops=3)
    for _ in range(3):
        end = baseline.visit(1, end + 1_000_000, end + 1_000_000)
    release = baseline.arrived(1, end + 1_000_000, hops=3)
    end = baseline.service(1, release, release)
    baseline.delivery(1, end, final=True)
    low = baseline.audit(baseline.write(tmp_path / "baseline"))["pp_flows"][0]
    high = loaded.audit(loaded.write(tmp_path / "loaded"))["pp_flows"][0]
    assert high["ep_data_service_ahead_ps"] == 156_400
    assert high["fct_ps"] == low["fct_ps"]
    assert low["packets"][0]["ring_holding_ps"] - high["packets"][0]["ring_holding_ps"] == 156_400
    selected = high["completion_dependency_witness"]["receiver_selected_predecessors"][0]
    assert selected["arrival_selected_by_release"] is False
    assert selected["arrival_delay_masked_by_release_ps"] == 156_400


def add_control(fixture, identity, kind, source_start, attempt):
    end = source_start + 1_280
    fixture.add(
        "events",
        end,
        event="packet_created",
        time_ps=end,
        flow_id=1,
        lifecycle_id=identity,
        packet_id=identity,
        detail="0",
    )
    specific = {"requested_attempt" if kind == "GAP_NACK" else "acknowledged_attempt": attempt}
    fixture.packets[identity] = fixture.add(
        "packets",
        end,
        packet_id=identity,
        lifecycle_id=identity,
        flow_id=1,
        packet_flow_id=101,
        kind=kind,
        source=8,
        destination=0,
        path_id=0,
        wire_bytes=64,
        packet_index=0,
        payload_offset=0,
        payload_bytes=4096,
        header_bytes=64,
        source_start_ps=source_start,
        source_end_ps=end,
        duplicate_test_copy=0,
        **specific,
    )
    for _ in range(3):
        end = fixture.visit(identity, end + 1_000_000, end + 1_000_000)
    arrival = end + 1_000_000
    fixture.runtime(identity, "endpoint_arrival", arrival)
    fixture.runtime(identity, "endpoint_consumed", arrival, detail="11")
    return arrival


def resolved_with_extra_retry(source_start):
    fixture = recovered_hole()
    nack = add_control(fixture, 8, "GAP_NACK", 14_800_000, 2)
    fixture.add(
        "events",
        nack,
        event="retry_authorized",
        time_ps=nack,
        flow_id=1,
        lifecycle_id=0,
        packet_index=0,
        attempt=2,
        detail="gap_nack",
        trigger_lifecycle_id=8,
    )
    add_control(fixture, 7, "GAP_RESOLVED", 14_864_000, 1)
    end = fixture.packet(9, index=0, source_start=source_start, hops=3, attempt=2)
    for _ in range(3):
        end = fixture.visit(9, end + 1_000_000, end + 1_000_000)
    arrival = end + 1_000_000
    fixture.runtime(9, "endpoint_arrival", arrival)
    terminal = fixture.runtime(9, "endpoint_consumed", arrival, detail="11")
    terminal["attempt"] = None
    fixture.runtime(9, "admission", arrival, detail="discard_duplicate")
    return fixture


def test_gap_resolution_allows_already_serializing_copy_to_drain(tmp_path):
    fixture = resolved_with_extra_retry(18_850_000)
    result = fixture.audit(fixture.write(tmp_path / "trace"), switch_buffer_bytes=4160)
    retry = next(packet for packet in result["pp_flows"][0]["packets"] if packet["attempt"] == 2)
    assert retry["source_start_ps"] < 18_869_120 < retry["source_end_ps"]
    assert retry["admission"] == "discard_duplicate"


@pytest.mark.parametrize("start", [18_869_120, 18_869_121])
def test_gap_resolution_cancels_retry_that_has_not_started_source_service(tmp_path, start):
    fixture = resolved_with_extra_retry(start)
    with pytest.raises(trace.TraceAuditError, match="source service begins after terminal"):
        fixture.audit(fixture.write(tmp_path / "trace"), switch_buffer_bytes=4160)


def test_gap_resolution_cancels_previously_scheduled_retry_timeout(tmp_path):
    fixture = recovered_hole()
    add_control(fixture, 7, "GAP_RESOLVED", 14_864_000, 1)
    deadline = fixture.packets[5]["source_end_ps"] + 40_000_000
    fixture.add(
        "events",
        deadline,
        event="retry_authorized",
        time_ps=deadline,
        flow_id=1,
        lifecycle_id=0,
        packet_index=0,
        attempt=2,
        detail="probe_timeout",
        origin_attempt=1,
        deadline_ps=deadline,
    )
    with pytest.raises(trace.TraceAuditError, match="authorization follows terminal"):
        fixture.audit(fixture.write(tmp_path / "trace"), switch_buffer_bytes=4160)


@pytest.mark.parametrize("kind", ["GAP_NACK", "GAP_RESOLVED"])
def test_control_attempts_cannot_escape_frozen_retry_limit(tmp_path, kind):
    fixture = recovered_hole()
    add_control(fixture, 7, kind, 14_864_000, 9)
    with pytest.raises(trace.TraceAuditError, match="control attempt exceeds"):
        fixture.audit(fixture.write(tmp_path / "trace"), switch_buffer_bytes=4160)


@pytest.mark.parametrize(
    "field,value", [("switch_id", 999), ("egress_id", 999), ("ingress_id", 999)]
)
def test_self_consistent_but_nonexistent_physical_route_is_fatal(tmp_path, field, value):
    fixture = unloaded()
    rows = fixture.ordered()
    for table, row in rows:
        if table == "queues":
            row[field] = value
    with pytest.raises(trace.TraceAuditError, match="declared packet route"):
        fixture.audit(fixture.write(tmp_path / "trace", rows))


def test_changed_path_id_cannot_keep_the_old_physical_queue_sequence(tmp_path):
    fixture = contended()
    rows = fixture.ordered()
    next(row for table, row in rows if table == "packets")["path_id"] = 1
    with pytest.raises(trace.TraceAuditError, match="declared packet route"):
        fixture.audit(fixture.write(tmp_path / "trace", rows))


def test_fabric_cannot_choose_a_nonexistent_spine(tmp_path):
    fixture = contended()
    rows = fixture.ordered()
    next(row for table, row in rows if table == "packets")["path_id"] = 2
    with pytest.raises(trace.TraceAuditError, match="nonexistent spine"):
        fixture.audit(fixture.write(tmp_path / "trace", rows))


@pytest.mark.parametrize("mutation", ["missing", "boolean", "profile"])
def test_completion_input_is_a_strict_native_row(tmp_path, mutation):
    fixture = unloaded()
    if mutation == "missing":
        del fixture.completions[0]["tag"]
    elif mutation == "boolean":
        fixture.completions[0]["tag"] = True
    else:
        fixture.completions[0]["profile"] = "rnic-nn"
    with pytest.raises(trace.TraceAuditError, match="completion"):
        fixture.audit(fixture.write(tmp_path / "trace"))
