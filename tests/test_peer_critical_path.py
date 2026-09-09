"""Physical causal ancestry and exact reporting-off packet identities."""

import copy
import json
from dataclasses import asdict, replace
from enum import Enum
from itertools import product

import pytest
import test_nvswitch_native as native_helpers
from test_peer_packet_runtime import engine, transfer

from examples.dgx_nvlink_v1.run_study import domain_profile
from simllm.backends.htsim_nvlink import NvlinkOperation, NvlinkSwitchArbitration, NvlinkTransfer
from simllm.backends.nvlink_runtime import NvlinkCausalEngine, NvlinkPhysicalBinding
from simllm.backends.peer_critical_path import (
    critical_path,
    observation,
    validate_observation,
    validate_packet_projection,
)
from simllm.compute.gpu_packet_port import GpuPeerPacketSession


@pytest.fixture(scope="session")
def native_library(tmp_path_factory):
    return native_helpers.native_library.__wrapped__(tmp_path_factory)


def selected(base, enabled=True, *, options=None, library=None):
    return NvlinkCausalEngine(
        base.profile, options or base.options, physical=base.physical,
        capture_critical_path=enabled, native_switch_library=library,
    )


def check_path(runtime, extents, release):
    packets = tuple(p.packet_id for p in runtime.packets if p.extent_id in extents)
    path = critical_path(
        runtime.causal_nodes, execution_id="execution", operation_id="collective",
        packet_ids=packets, released_at_ps=release,
    )
    assert path.visible_at_ps == max(
        p.visible_at_ps for p in runtime.packets if p.packet_id in packets
    )
    assert sum(row.completed_at_ps - row.started_at_ps for row in path.intervals) == path.duration_ps
    wire = observation(runtime.causal_nodes, (path,), service_rates={
        "feed": runtime.profile.tx.endpoint_egress_rate_bytes_per_second,
        "switch": runtime.profile.switch.service_rate_bytes_per_second,
        "receive": runtime.profile.rx.ingress_rate_bytes_per_second,
    })
    assert validate_observation(wire) == (runtime.causal_nodes, (path,))
    return path


@pytest.mark.parametrize("switched", [False, True])
@pytest.mark.parametrize("capacity", [272, 544, 65536])
@pytest.mark.parametrize("processing", [0, 100000])
def test_finite_capacity_and_control_tails_keep_complete_critical_coverage(switched, capacity, processing):
    base = engine(switched=switched, ranks=4, capacity=capacity,
                  receiver_capacity=capacity, credits=1, processing=processing)
    enabled, absent = selected(base), selected(base, False)
    for phase in range(2):
        release = enabled.now_ps
        transfers = tuple(transfer(f"phase-{phase}:flow-{i}", i, 3, 1024, release) for i in range(3))
        for runtime in (enabled, absent):
            runtime.admit(transfers, include_switch=switched)
            runtime.advance_until_visible(tuple(row.extent_id for row in transfers))
        assert enabled.packets == absent.packets
        assert enabled.physical_paths == absent.physical_paths
        assert enabled.buffer_claims == absent.buffer_claims
        assert enabled.now_ps == absent.now_ps
        path = check_path(enabled, {row.extent_id for row in transfers}, release)
        if phase and processing and capacity == 272:
            assert any(row.original_started_at_ps <= release and row.kind == "credit_return"
                       and row.packet_id.startswith("phase-0:") for row in path.intervals)
    assert asdict(enabled.drain()) == asdict(absent.drain())


def test_prior_control_interval_is_clipped_at_the_next_phase_release():
    base = engine(ranks=2, credits=1, receiver_capacity=272, processing=100000)
    runtime = selected(base)
    runtime.admit((transfer("old"),), include_switch=False)
    first = runtime.advance_until_visible(("old",))
    runtime.advance_to(first + 5000)
    release = runtime.now_ps
    runtime.admit((transfer("new", release=release),), include_switch=False)
    runtime.advance_until_visible(("new",))
    path = check_path(runtime, {"new"}, release)
    interval = path.intervals[0]
    assert interval.kind == "credit_return"
    assert interval.original_started_at_ps == first
    assert interval.started_at_ps == release
    assert interval.completed_at_ps == first + 101000
    assert interval.packet_id.startswith("old")


@pytest.mark.parametrize("switched", [False, True])
@pytest.mark.parametrize("rate", [2500000000, 25000000000, 100000000000])
def test_slow_receiver_is_one_observed_service_owner(switched, rate):
    base = engine(switched=switched, ranks=4)
    profile = replace(base.profile, rx=replace(base.profile.rx, ingress_rate_bytes_per_second=rate))
    base = NvlinkCausalEngine(profile, base.options, physical=base.physical)
    enabled, absent = selected(base), selected(base, False)
    rows = tuple(transfer(f"donor-{source}", source, 3, 1024) for source in range(3))
    for runtime in (enabled, absent):
        runtime.admit(rows, include_switch=switched)
        runtime.advance_until_visible(tuple(row.extent_id for row in rows))
    check_path(enabled, {row.extent_id for row in rows}, 0)
    assert asdict(enabled.drain()) == asdict(absent.drain())


@pytest.mark.parametrize("switched", [False, True])
@pytest.mark.parametrize("rate", [2500000000, 25000000000, 100000000000])
def test_packet_pacing_has_an_explicit_phase_start_and_preserves_the_scheduler(switched, rate):
    base = engine(switched=switched, ranks=2)
    sessions = [GpuPeerPacketSession("paced", base.profile, base.physical,
                                    capture_critical_path=enabled) for enabled in (False, True)]
    for index in range(2):
        release = sessions[0].now_ps
        row = replace(transfer(f"paced-{index}", payload=1024, release=release),
                      offered_rate_bytes_per_second=rate)
        for session in sessions:
            session.admit(f"step-{index}", "dispatch", (row,))
            session.advance_until_visible((row.extent_id,))
        absent, enabled = (session.evidence() for session in sessions)
        _, phases = validate_packet_projection(enabled)
        assert phases[-1].released_at_ps == release
        assert {key: value for key, value in enabled.items() if key != "critical_path"} == absent
        if rate == 2500000000:
            assert any(interval.kind == "source_pacing" for interval in phases[-1].intervals)
    assert sessions[0].drain() == sessions[1].drain()


def test_out_of_order_wire_arrival_names_the_packet_that_unlocks_visibility():
    base = engine(ranks=2)
    domain = base.physical.fabric
    first, second = domain.ports
    original, = domain.links
    domain = replace(
        domain, ports=(*domain.ports, replace(first, port_id="parallel-source"),
                       replace(second, port_id="parallel-destination")),
        links=(replace(original, propagation_delay_ps=100000),
               replace(original, link_id="parallel", endpoint_a="parallel-source",
                       endpoint_b="parallel-destination")),
        routes=tuple(replace(route, paths=(*route.paths, ("parallel",))) for route in domain.routes),
    )
    enabled = GpuPeerPacketSession("ordered", base.profile, NvlinkPhysicalBinding(domain),
                                  capture_critical_path=True)
    absent = GpuPeerPacketSession("ordered", base.profile, NvlinkPhysicalBinding(domain))
    for session in (enabled, absent):
        session.admit("step", "dispatch", (transfer("striped", payload=512),))
        session.advance_until_visible(("striped",))
    row = enabled.evidence()
    nodes, _ = validate_packet_projection(row)
    first, second = row["packets"]
    assert second["rx_finished_at_ps"] < first["rx_finished_at_ps"]
    assert first["visible_at_ps"] == second["visible_at_ps"]
    first_visible = next(node for node in nodes if node.packet_id == first["packet_id"] and node.point == "visible")
    second_visible = next(node for node in nodes if node.packet_id == second["packet_id"] and node.point == "visible")
    assert first_visible.node_id in second_visible.parents
    assert enabled.drain() == absent.drain()


def test_enabled_preflight_failure_changes_no_causal_or_packet_state():
    base = engine(ranks=2)
    session = GpuPeerPacketSession("atomic", base.profile, base.physical, capture_critical_path=True)
    before = session.evidence()
    with pytest.raises(ValueError, match="no physical attachment route"):
        session.admit("step", "dispatch", (transfer("valid"), transfer("invalid", destination=3)))
    assert session.evidence() == before
    with pytest.raises(ValueError, match="write"):
        session.admit("step", "read", (replace(transfer("read"), operation=NvlinkOperation.PEER_READ),))
    assert session.evidence() == before


def test_unbound_and_replayed_critical_reporting_reject_before_execution():
    base = engine(ranks=2)
    with pytest.raises(ValueError, match="physical"):
        NvlinkCausalEngine(base.profile, base.options, capture_critical_path=True)
    with pytest.raises(ValueError, match="replays"):
        selected(base, options=replace(base.options, replay_timeout_ps=1))


@pytest.mark.parametrize("switched", [False, True])
@pytest.mark.parametrize("policy", list(NvlinkSwitchArbitration))
@pytest.mark.parametrize("feed", [12500000000, 100000000000])
def test_fanout_and_disjoint_paths_record_actual_shared_resource_owners(switched, policy, feed):
    base = engine(switched=switched, ranks=4, feed=feed)
    options = replace(base.options, switch_arbitration=policy)
    enabled, absent = selected(base, options=options), selected(base, False, options=options)
    transfers = (
        transfer("first", 0, 1, 1024), transfer("second", 0, 2, 1024),
        transfer("reverse", 1, 0, 1024), transfer("independent", 3, 2, 1024),
    )
    for runtime in (enabled, absent):
        runtime.admit(transfers, include_switch=switched)
        runtime.advance_until_visible(tuple(row.extent_id for row in transfers))
    check_path(enabled, {row.extent_id for row in transfers}, 0)
    assert asdict(enabled.drain()) == asdict(absent.drain())


def captured():
    base = engine(switched=True, ranks=3, credits=1, receiver_capacity=272, processing=50000)
    session = GpuPeerPacketSession("session", base.profile, base.physical, capture_critical_path=True)
    session.admit("step", "combine", (transfer("a", 0, 2, 1024), transfer("b", 1, 2, 1024)))
    session.advance_until_visible(("a", "b"))
    return session


def test_wire_roundtrip_and_partial_observation_bind_every_original_point():
    session = captured()
    snapshot = session.evidence()
    def encode(value):
        if isinstance(value, Enum):
            return value.value
        raise TypeError(type(value).__name__)
    wire = json.loads(json.dumps(snapshot, default=encode))
    assert validate_packet_projection(wire) == validate_packet_projection(snapshot)
    session.drain()
    assert validate_packet_projection(session.evidence())[0]


def test_equal_time_foreign_service_parent_cannot_forge_a_valid_critical_chain():
    raw = copy.deepcopy(captured().evidence())
    trace = raw["critical_path"]
    feed = next(row for row in trace["nodes"] if row["point"] == "source-feed")
    eligible = next(row for row in trace["nodes"] if row["point"] == "eligible" and row["packet_id"] == feed["packet_id"])
    feed["parents"] = (eligible["node_id"],)
    # Recompute the internally consistent path so only its physical join can reject.
    phase_rows, trace["phases"] = trace["phases"], []
    nodes, _ = validate_observation(trace)
    trace["phases"] = [asdict(critical_path(
        nodes, execution_id=phase["execution_id"], operation_id=phase["operation_id"],
        packet_ids=phase["packet_ids"], released_at_ps=phase["released_at_ps"],
    )) for phase in phase_rows]
    with pytest.raises(ValueError, match="original packet service"):
        validate_packet_projection(raw)


@pytest.mark.parametrize("kind", [
    "missing-node", "duplicate-node", "missing-parent", "forward-parent",
    "timestamp", "invented-grant", "packet", "resource", "phase",
    "missing-interval", "overlap", "terminal", "rate", "unknown-field",
])
def test_corrupted_critical_sidecar_rejects_against_original_observations(kind):
    raw = copy.deepcopy(captured().evidence())
    trace = raw["critical_path"]
    nodes, phase = trace["nodes"], trace["phases"][0]
    grant = next(row for row in nodes if row["point"] == "tx-grant")
    if kind == "missing-node":
        nodes.pop(0)
    elif kind == "duplicate-node":
        nodes.append(copy.deepcopy(nodes[0]))
    elif kind == "missing-parent":
        grant["parents"] = ("missing",)
    elif kind == "forward-parent":
        grant["parents"] = (nodes[-1]["node_id"],)
    elif kind == "timestamp":
        nodes[0]["at_ps"] += 1
    elif kind == "invented-grant":
        grant["at_ps"] += 1
    elif kind == "packet":
        nodes[-1]["packet_id"] = "foreign"
    elif kind == "resource":
        nodes[-1]["resource_id"] = "unbound-resource"
    elif kind == "phase":
        phase["execution_id"] = "foreign"
    elif kind == "missing-interval":
        phase["intervals"] = phase["intervals"][1:]
    elif kind == "overlap":
        phase["intervals"][1]["started_at_ps"] -= 1
    elif kind == "terminal":
        phase["visible_at_ps"] += 1
    elif kind == "rate":
        trace["service_rates"]["feed"] *= 2
    else:
        nodes[-1]["unobserved"] = True
    with pytest.raises((TypeError, ValueError)):
        validate_packet_projection(raw)


@pytest.mark.parametrize("generation", ["a100", "h100"])
@pytest.mark.parametrize("policy", list(NvlinkSwitchArbitration))
@pytest.mark.parametrize("capacity", [272, 65536])
def test_native_and_python_causal_paths_match_across_every_ordered_board_pair(
    native_library, generation, policy, capacity
):
    domain, profile = domain_profile(generation, 25_000_000_000,
                                     capacity=capacity, rx_capacity=capacity)
    base = NvlinkCausalEngine(
        profile, replace(engine().options, switch_arbitration=policy),
        physical=NvlinkPhysicalBinding(domain, 1000),
    )
    python = selected(base)
    native = selected(base, library=native_library)
    for phase in range(2):
        release = python.now_ps
        transfers = tuple(NvlinkTransfer(
            extent_id=f"phase-{phase}:{source}-{destination}", source=source, destination=destination,
            payload_bytes=256, released_at_ps=release, topology_endpoint_count=8)
                          for source, destination in product(range(8), repeat=2) if source != destination)
        for runtime in (python, native):
            runtime.admit(transfers, include_switch=True)
            runtime.advance_until_visible(tuple(row.extent_id for row in transfers))
        assert python.causal_nodes == native.causal_nodes
        assert python.packets == native.packets
        check_path(python, {row.extent_id for row in transfers}, release)
        check_path(native, {row.extent_id for row in transfers}, release)
    assert asdict(python.drain()) == asdict(native.drain())
