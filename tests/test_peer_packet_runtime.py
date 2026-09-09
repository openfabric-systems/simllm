"""Physical attachment, retained control tails and finite ownership tests."""

import json
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path

import pytest

from simllm.backends.htsim_nvlink import (
    NvlinkAlignedOptions,
    NvlinkFifoPlacement,
    NvlinkSwitchConfig,
    NvlinkSwitchMode,
    NvlinkTransfer,
    load_nvlink_candidate_profile,
)
from simllm.backends.nvlink_runtime import NvlinkCausalEngine, NvlinkPhysicalBinding
from simllm.compute.gpu_packet_port import GpuPeerPacketSession
from simllm.core.execution import EventPhase
from simllm.placement import (
    FabricLink,
    FabricNodePlacement,
    FabricTopologyManifest,
    GpuFabricPlacement,
    PeerFabric,
    PeerPortPlacement,
    PeerRoute,
)

PROFILE = Path(__file__).resolve().parents[1] / "examples/a100_nvlink_packet_v1/candidate-profile-pre-traf70.json"


def fabric(*, switched=False, ranks=3, rate=25_000_000_000, capacity=65536):
    ports, links, routes = [], [], []
    if switched:
        for rank in range(ranks):
            ports.extend((PeerPortPlacement(f"gpu{rank}-p", gpu_rank=rank),
                          PeerPortPlacement(f"switch-p{rank}", switch_id="peer-switch")))
            links.append(FabricLink(f"link{rank}", f"gpu{rank}-p", f"switch-p{rank}", rate * 8, 1000))
        for source in range(ranks):
            for destination in range(ranks):
                if source != destination:
                    routes.append(PeerRoute(source, destination, ((f"link{source}", f"link{destination}"),)))
    else:
        for source in range(ranks):
            for destination in range(source + 1, ranks):
                a, b, link = f"gpu{source}-to{destination}", f"gpu{destination}-to{source}", f"link{source}-{destination}"
                ports.extend((PeerPortPlacement(a, gpu_rank=source), PeerPortPlacement(b, gpu_rank=destination)))
                links.append(FabricLink(link, a, b, rate * 8, 1000))
                routes.extend((PeerRoute(source, destination, ((link,),)),
                               PeerRoute(destination, source, ((link,),))))
    return PeerFabric("peer-domain", "node-0", tuple(ports), tuple(links), tuple(routes),
                      switch_input_buffer_bytes=capacity)


def engine(*, switched=False, ranks=3, rate=25_000_000_000, capacity=65536,
           receiver_capacity=65536, credits=256, processing=0, acknowledgement=0,
           feed=25_000_000_000, observer=None):
    base = load_nvlink_candidate_profile(PROFILE)
    profile = replace(
        base,
        tx=replace(base.tx, links_per_peer=1, per_link_rate_bytes_per_second=rate,
                   endpoint_egress_rate_bytes_per_second=feed, credits_per_destination=credits),
        rx=replace(base.rx, ingress_rate_bytes_per_second=25_000_000_000,
                   buffer_capacity_bytes=receiver_capacity, credit_return_latency_ps=0),
        switch=NvlinkSwitchConfig(mode=NvlinkSwitchMode.QUEUED, fifo_placement=NvlinkFifoPlacement.INPUT,
                                  service_rate_bytes_per_second=25_000_000_000,
                                  buffer_capacity_bytes=capacity, arbitration="fifo", head_of_line_blocking=True)
        if switched else NvlinkSwitchConfig(mode=NvlinkSwitchMode.PASS_THROUGH),
    )
    return NvlinkCausalEngine(profile, NvlinkAlignedOptions(), on_packet_event=observer,
                             physical=NvlinkPhysicalBinding(
                                 fabric(switched=switched, ranks=ranks, rate=rate, capacity=capacity),
                                 processing, acknowledgement))


def transfer(name, source=0, destination=1, payload=256, release=0):
    return NvlinkTransfer(extent_id=name, source=source, destination=destination,
                          payload_bytes=payload, released_at_ps=release)


@pytest.mark.parametrize("processing", (0, 100000))
def test_next_phase_waits_for_retained_credit_and_final_drain_keeps_ack(processing):
    observations = []
    runtime = engine(credits=1, receiver_capacity=272, processing=processing,
                     acknowledgement=200000, observer=lambda kind, packet, at: observations.append((kind, packet.packet_id, at)))
    runtime.admit((transfer("first"),), include_switch=False)
    assert runtime.advance_until_visible(("first",)) == 22760
    assert runtime.has_pending_physical_work
    assert not any(kind == "delivered" for kind, _, _ in observations)
    runtime.admit((transfer("second", release=runtime.now_ps),), include_switch=False)
    assert runtime.advance_until_visible(("second",)) == 46520 + processing
    assert runtime.packets[1].tx_started_at_ps == 23760 + processing
    result = runtime.drain()
    assert result.physical_drain_time_ps == 236640 + processing
    assert result.authority.value == "physical_retained_v1"
    assert [at for _, _, at in observations] == sorted(at for _, _, at in observations)
    assert sum(kind == "delivered" for kind, _, _ in observations) == 2
    assert not runtime.has_pending_physical_work


@pytest.mark.parametrize("capacity", (272, 544))
@pytest.mark.parametrize("rate", (12_500_000_000, 25_000_000_000))
def test_fanout_shares_input_attachment_and_total_input_buffer(capacity, rate):
    runtime = engine(switched=True, rate=rate, capacity=capacity, feed=100_000_000_000)
    runtime.admit((transfer("one", payload=1024), transfer("two", destination=2, payload=1024)), include_switch=True)
    result = runtime.drain()
    packets = sorted(result.packets, key=lambda packet: packet.tx_started_at_ps)
    assert len({row.path.input_resource for row in runtime.physical_paths}) == 1
    assert all(a.tx_finished_at_ps <= b.tx_started_at_ps for a, b in pairwise(packets))
    assert all(row.source_feed_finished_at_ps - next(p.tx_started_at_ps for p in packets if p.packet_id == row.packet_id) == 2720
               for row in runtime.physical_paths)
    visits = [visit for visit in result.buffer_visits if visit.buffer_id.startswith("switch-input:")]
    assert len({visit.buffer_id for visit in visits}) == 1
    boundaries = sorted({time for visit in visits for time in (visit.reserved_at_ps, visit.credit_available_at_ps)})
    for time in boundaries:
        outstanding = sum(visit.wire_bytes for visit in visits
                          if visit.reserved_at_ps <= time < visit.credit_available_at_ps)
        assert outstanding <= capacity


def test_physical_link_directions_are_independent():
    runtime = engine(ranks=2)
    runtime.admit((transfer("forward"), transfer("reverse", source=1, destination=0)), include_switch=False)
    result = runtime.drain()
    assert {packet.tx_started_at_ps for packet in result.packets} == {0}
    assert {packet.visible_at_ps for packet in result.packets} == {22760}
    paths = runtime.physical_paths
    assert paths[0].path.input_link.link_id == paths[1].path.input_link.link_id
    assert paths[0].path.input_resource != paths[1].path.input_resource


def test_whole_phase_preflight_failure_preserves_calendar_and_callbacks():
    events = []
    runtime = engine(ranks=2, observer=lambda *args: events.append(args))
    runtime.admit((transfer("old"),), include_switch=False)
    runtime.advance_until_visible(("old",))
    before = runtime.now_ps, runtime.packets, runtime.buffer_ownership, tuple(events)
    with pytest.raises(ValueError, match="no physical attachment route"):
        runtime.admit((transfer("valid", release=runtime.now_ps),
                       transfer("invalid", destination=3, release=runtime.now_ps)), include_switch=False)
    assert (runtime.now_ps, runtime.packets, runtime.buffer_ownership, tuple(events)) == before
    runtime.admit((transfer("valid", release=runtime.now_ps),), include_switch=False)
    assert runtime.drain().logical_bytes == 512


@pytest.mark.parametrize("switched", (False, True))
def test_optional_manifest_roundtrip_preserves_physical_attachment(switched, tmp_path):
    domain = fabric(switched=switched)
    node = FabricNodePlacement("node-0", "serving", tuple(
        GpuFabricPlacement(rank, f"gpu{rank}", "node-0", f"pcie{rank}", f"nic{rank}") for rank in range(3)
    ), ())
    manifest = FabricTopologyManifest(nodes=[node], peer_fabrics=(domain,))
    manifest.validate()
    loaded = FabricTopologyManifest.load(manifest.save(tmp_path / "fabric.json"))
    assert loaded.peer_fabrics == (domain,)
    legacy = FabricTopologyManifest(nodes=[node])
    assert "peer_fabrics" not in legacy.to_dict()
    bad = asdict(domain)
    bad["routes"][0]["paths"] = ["link0-1"]
    with pytest.raises(TypeError, match="array of link arrays"):
        PeerFabric.from_dict(bad)


def test_a_gpu_feed_cannot_be_claimed_by_two_domains():
    domain = fabric()
    node = FabricNodePlacement("node-0", "serving", tuple(
        GpuFabricPlacement(rank, f"gpu{rank}", "node-0", f"pcie{rank}", f"nic{rank}") for rank in range(3)
    ), ())
    manifest = FabricTopologyManifest(nodes=[node], peer_fabrics=(domain, replace(domain, domain_id="second")))
    with pytest.raises(ValueError, match="two domain authorities"):
        manifest.validate()


def test_gpu_emitter_uses_one_common_consumer_and_retains_transport_tail():
    source = engine(ranks=2, acknowledgement=200000, processing=100000, receiver_capacity=272, credits=1)
    session = GpuPeerPacketSession("two-step", source.profile, source.physical)
    first = session.admit("step0", "dispatch", (transfer("first"),))
    assert session.advance_until_visible(("first",)) == 22760
    assert session.port_snapshots()[0][1].live_extent_tokens == (first[0].extent_token,)
    partial = session.evidence()
    assert partial["drained_result"] is None
    assert len(partial["resource_visits"]) == 3
    assert partial["buffer_claims"][0]["released_at_ps"] == 22760
    assert not partial["buffer_claims"][0]["returned"]
    assert [event.phase for event in session.events] == [EventPhase.SUBMITTED, EventPhase.QUEUED,
                                                       EventPhase.STARTED, EventPhase.COMPLETED]
    second = session.admit("step1", "dispatch", (transfer("second", release=session.now_ps),))
    assert session.advance_until_visible(("second",)) == 146520
    result = session.drain()
    assert result.physical_drain_time_ps == 336640
    tokens = [token for extent in (*first, *second) for token in (extent.extent_token, *extent.attempt_tokens)]
    assert len(set(tokens)) == len(tokens)
    snapshots = session.port_snapshots(final=True)
    assert sum(row.delivered_payload_bytes for _, row in snapshots) == 512
    assert [row.event_time_ps for row in session.packet_events] == sorted(row.event_time_ps for row in session.packet_events)
    evidence = session.evidence()
    json.dumps(evidence)
    assert len(evidence["resource_visits"]) == 6
    assert len(evidence["drained_result"]["buffer_visits"]) == 2
    assert all(row["returned"] for row in evidence["buffer_claims"])
    assert evidence["authority"] == "physical_retained_v1"
    assert evidence["contexts"][0]["tx_start_boundary"] == "link_grant"
    assert {row["event_kind"] for row in evidence["packet_events"]} == {
        "packet_tx_started", "packet_tx_finished", "packet_rx_arrived", "delivered"
    }


def test_partial_switched_resource_ledger_retains_unfinished_reservations():
    source = engine(switched=True)
    session = GpuPeerPacketSession("partial", source.profile, source.physical)
    session.admit("step0", "dispatch", (transfer("one"),))
    session.advance_to(5000)
    partial = session.evidence()
    assert partial["resource_visits"] == []
    assert len(partial["buffer_claims"]) == 1
    assert partial["buffer_claims"][0]["arrived_at_ps"] is None
    session.drain()
    final = session.evidence()
    json.dumps(final)
    assert len(final["resource_visits"]) == 6
    assert len(final["buffer_claims"]) == 2
    assert len(final["drained_result"]["buffer_visits"]) == 2
    for visit in session.resource_visits:
        assert visit.queue_wait_ps == visit.started_at_ps - visit.eligible_at_ps
        assert visit.service_ps == visit.finished_at_ps - visit.started_at_ps


def test_gpu_preflight_failure_does_not_consume_tokens_or_emit_events():
    source = engine(ranks=2)
    session = GpuPeerPacketSession("atomic", source.profile, source.physical)
    before = session.evidence()
    with pytest.raises(ValueError, match="no physical attachment route"):
        session.admit("step0", "dispatch", (transfer("valid"), transfer("invalid", destination=3)))
    assert session.evidence() == before
    session.validate_phase("step0", "dispatch", (transfer("valid"),))
    assert session.evidence() == before
    admitted = session.admit("step0", "dispatch", (transfer("valid"),))
    assert admitted[0].extent_token == 1
    assert session.drain().logical_bytes == 256


def test_disabled_gpu_packet_port_keeps_interface_and_rejects_before_admission():
    source = engine(ranks=2)
    enabled = GpuPeerPacketSession("enabled", source.profile, source.physical)
    ports = tuple(replace(port, config=replace(port.config, enabled=False)) for port in enabled.ports)
    disabled = GpuPeerPacketSession("disabled", source.profile, source.physical, ports=ports)
    before = disabled.evidence()
    with pytest.raises(ValueError, match="disabled port"):
        disabled.admit("step0", "dispatch", (transfer("blocked"),))
    assert disabled.evidence() == before


def test_two_switch_outputs_share_one_gpu_receive_budget_before_any_grant():
    source = engine(switched=True, receiver_capacity=272)
    owners = (("source0", 0), ("source1", 1), ("receiver-a", 2), ("receiver-b", 2))
    ports, links = [], []
    for name, rank in owners:
        ports.extend((PeerPortPlacement(name, gpu_rank=rank),
                      PeerPortPlacement(f"switch-{name}", switch_id="shared-switch")))
        links.append(FabricLink(f"link-{name}", name, f"switch-{name}", 200_000_000_000, 1000))
    domain = PeerFabric("multi-attachment", "node-0", tuple(ports), tuple(links), (
        PeerRoute(0, 2, (("link-source0", "link-receiver-a"),)),
        PeerRoute(1, 2, (("link-source1", "link-receiver-b"),)),
    ))
    runtime = NvlinkCausalEngine(source.profile, source.options, physical=NvlinkPhysicalBinding(domain))
    runtime.admit((transfer("first", destination=2), transfer("second", source=1, destination=2)), include_switch=True)
    result = runtime.drain()
    assert sorted(packet.visible_at_ps for packet in result.packets) == [34640, 58400]
    assert result.max_rx_buffer_occupancy_bytes == 272
    assert {visit.buffer_id for visit in result.buffer_visits if visit.buffer_id.startswith("rx:")} == {"rx:2"}


def test_switched_attempt_retains_acknowledgements_for_both_hops():
    source = engine(switched=True, ranks=2, acknowledgement=200000)
    domain = replace(source.physical.fabric, links=tuple(
        replace(link, propagation_delay_ps=2000) if link.link_id == "link1" else link
        for link in source.physical.fabric.links
    ))
    observations = []
    runtime = NvlinkCausalEngine(source.profile, source.options,
                                 physical=replace(source.physical, fabric=domain),
                                 on_packet_event=lambda kind, packet, at: observations.append((kind, at)))
    runtime.admit((transfer("two-hop"),), include_switch=True)
    assert runtime.advance_until_visible(("two-hop",)) == 35640
    assert not any(kind == "delivered" for kind, _ in observations)
    path, = runtime.physical_paths
    assert path.input_hop_acknowledged_at_ps == 212880
    assert path.output_hop_acknowledged_at_ps == 226760
    result = runtime.drain()
    assert result.physical_drain_time_ps == 226760
    assert observations[-1] == ("delivered", 226760)


def test_idle_observation_clock_cannot_inflate_the_physical_drain_boundary():
    runtime = engine(ranks=2)
    runtime.admit((transfer("one"),), include_switch=False)
    first = runtime.drain()
    runtime.advance_to(first.physical_drain_time_ps + 1000000)
    assert runtime.drain().physical_drain_time_ps == first.physical_drain_time_ps
    assert runtime.now_ps == first.physical_drain_time_ps + 1000000


def test_gpu_clock_and_future_horizon_overflow_reject_before_admission():
    source = engine(ranks=2)
    session = GpuPeerPacketSession("overflow", source.profile, source.physical)
    before = session.evidence()
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        session.advance_to(2**64)
    assert session.evidence() == before
    session.advance_to(2**64 - 101)
    before = session.evidence()
    with pytest.raises(ValueError, match="completion horizon"):
        session.admit("near-end", "dispatch", (transfer("too-late", release=session.now_ps),))
    assert session.evidence() == before
