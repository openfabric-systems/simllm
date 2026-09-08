"""Independent physical and causal checks of the NVLink calendar."""

from collections import defaultdict
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pytest

from simllm.backends.htsim_nvlink import (
    NvlinkAlignedOptions,
    NvlinkDomainService,
    NvlinkFifoPlacement,
    NvlinkFlowControlConfig,
    NvlinkOperation,
    NvlinkPacketDirection,
    NvlinkSwitchConfig,
    NvlinkSwitchMode,
    NvlinkTrafficClass,
    NvlinkTransfer,
    load_nvlink_candidate_profile,
)
from simllm.backends.nvlink_runtime import NvlinkCausalEngine

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "examples/a100_nvlink_packet_v1/candidate-profile-pre-traf70.json"


def profile(*, rate=25_000_000_000, credits=4, capacity=4, return_ps=0, queued=False):
    base = load_nvlink_candidate_profile(PROFILE_PATH)
    return replace(
        base,
        tx=replace(
            base.tx,
            links_per_peer=1,
            credits_per_destination=credits,
            per_link_rate_bytes_per_second=rate,
            endpoint_egress_rate_bytes_per_second=100_000_000_000,
        ),
        rx=replace(
            base.rx,
            ingress_rate_bytes_per_second=100_000_000_000,
            buffer_capacity_bytes=272 * capacity,
            credit_return_latency_ps=return_ps,
        ),
        switch=(
            NvlinkSwitchConfig(
                mode=NvlinkSwitchMode.QUEUED,
                fifo_placement=NvlinkFifoPlacement.INPUT,
                service_rate_bytes_per_second=25_000_000_000,
                buffer_capacity_bytes=272,
                arbitration="fifo",
                head_of_line_blocking=True,
            )
            if queued
            else base.switch
        ),
    )


def transfer(name="write", *, source=0, destination=1, size=1024, release=0, read=False):
    return NvlinkTransfer(
        extent_id=name,
        source=source,
        destination=destination,
        payload_bytes=size,
        released_at_ps=release,
        operation=NvlinkOperation.PEER_READ if read else NvlinkOperation.PEER_WRITE,
    )


def run(transfers, *, config=None, options=None):
    return NvlinkDomainService(config or profile()).serve_aligned(
        transfers,
        analytic_result=None,
        options=options,
    )


def packet_times(result, extent):
    return [
        (
            p.packet_id,
            p.tx_eligible_at_ps,
            p.tx_started_at_ps,
            p.tx_finished_at_ps,
            p.rx_started_at_ps,
            p.rx_finished_at_ps,
            p.visible_at_ps,
        )
        for p in result.packets
        if p.extent_id == extent
    ]


def assert_capacity_conservation(result):
    """Integrate independently reconstructed occupancy and advertised capacity."""
    pools = defaultdict(list)
    claims = set()
    for visit in result.buffer_visits:
        key = (visit.packet_id, visit.buffer_id)
        assert key not in claims
        claims.add(key)
        assert visit.reserved_at_ps <= visit.arrived_at_ps <= visit.released_at_ps
        assert visit.released_at_ps <= visit.credit_available_at_ps
        pools[visit.buffer_id].append(visit)
    for visits in pools.values():
        capacities = {v.capacity_bytes for v in visits}
        assert len(capacities) == 1
        capacity = capacities.pop()
        for start, end in (
            ("reserved_at_ps", "credit_available_at_ps"),
            ("reserved_at_ps", "released_at_ps"),
            ("arrived_at_ps", "released_at_ps"),
        ):
            deltas = []
            for visit in visits:
                deltas += [
                    (getattr(visit, start), visit.wire_bytes),
                    (getattr(visit, end), -visit.wire_bytes),
                ]
            occupied = 0
            # Releases at a boundary precede that boundary's new grants.
            for _, delta in sorted(deltas):
                occupied += delta
                assert 0 <= occupied <= capacity
            assert occupied == 0
    assert len(result.credit_releases) == len(result.packets)
    assert len(result.visibility_events) == len(result.packets)
    assert result.physical_drain_time_ps >= result.completion_time_ps


@pytest.mark.parametrize("rate", [12_500_000_000, 25_000_000_000])
@pytest.mark.parametrize("size", [256, 1024])
@pytest.mark.parametrize("read", [False, True])
def test_uncongested_job_matches_independent_serialization_oracle(rate, size, read):
    result = run([transfer(size=size, read=read)], config=profile(rate=rate))
    n = size // 256
    link = (272 * 10**12 + rate - 1) // rate
    request = (16 * 10**12 + rate - 1) // rate + 160
    assert result.completion_time_ps == n * link + 2720 + (request if read else 0)
    assert result.fixed_point_iterations == 0
    assert result.event_count > 0
    assert_capacity_conservation(result)
    requests = [p for p in result.packets if p.direction is NvlinkPacketDirection.REQUEST]
    if read:
        assert len(requests) == 1
        for packet in result.packets:
            if packet.direction is NvlinkPacketDirection.RESPONSE:
                assert packet.tx_eligible_at_ps >= requests[0].visible_at_ps
                assert packet.tx_started_at_ps >= requests[0].visible_at_ps


@pytest.mark.parametrize("rate", [12_500_000_000, 25_000_000_000])
@pytest.mark.parametrize("size", [256, 1024])
@pytest.mark.parametrize("credits,capacity", [(1, 4), (4, 1), (1, 1)])
@pytest.mark.parametrize("return_ps", [0, 10_000])
def test_credit_or_byte_capacity_enforces_one_packet_recurrence(
    rate,
    size,
    credits,
    capacity,
    return_ps,
):
    result = run(
        [transfer(size=size)],
        config=profile(
            rate=rate,
            credits=credits,
            capacity=capacity,
            return_ps=return_ps,
        ),
    )
    n = size // 256
    link = (272 * 10**12 + rate - 1) // rate
    assert result.completion_time_ps == n * (link + 2720) + (n - 1) * return_ps
    assert_capacity_conservation(result)


@pytest.mark.parametrize("offset", [1_000_000, 2_000_000])
@pytest.mark.parametrize("future_first", [False, True])
def test_future_extent_cannot_reserve_earlier_service(offset, future_first):
    early = transfer("early", size=256)
    future = transfer("future", size=256, release=offset)
    transfers = [future, early] if future_first else [early, future]
    baseline = run([early])
    combined = run(transfers)
    assert packet_times(combined, "early") == packet_times(baseline, "early")
    assert all(p.tx_started_at_ps >= offset for p in combined.packets if p.extent_id == "future")
    assert_capacity_conservation(combined)


def test_unready_read_response_does_not_block_an_independent_target_write():
    read = transfer("read", source=0, destination=1, read=True, size=256)
    write = transfer("write", source=1, destination=2, size=256)
    result = run([read, write])
    outgoing = next(p for p in result.packets if p.extent_id == "write")
    request = next(
        p
        for p in result.packets
        if p.extent_id == "read" and p.direction is NvlinkPacketDirection.REQUEST
    )
    response = next(p for p in result.packets if p.direction is NvlinkPacketDirection.RESPONSE)
    assert outgoing.tx_started_at_ps == 0
    assert response.tx_started_at_ps >= request.visible_at_ps
    assert_capacity_conservation(result)


@pytest.mark.parametrize("queued", [False, True])
def test_disjoint_resources_do_not_change_the_other_extent(queued):
    config = profile(queued=queued)
    first = transfer("first", source=0, destination=1)
    other = transfer("other", source=2, destination=3)
    baseline = run([first], config=config)
    combined = run([first, other], config=config)
    assert packet_times(combined, "first") == packet_times(baseline, "first")
    assert_capacity_conservation(combined)


def test_opposite_directions_have_independent_link_service():
    result = run(
        [
            transfer("forward", source=0, destination=1, size=256),
            transfer("reverse", source=1, destination=0, size=256),
        ]
    )
    assert [p.tx_started_at_ps for p in result.packets] == [0, 0]
    assert len({p.visible_at_ps for p in result.packets}) == 1
    assert_capacity_conservation(result)


@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("return_ps", [0, 10_000])
def test_fanin_reserves_every_arrival_and_drains_one_packet_buffers(queued, return_ps):
    config = profile(queued=queued, capacity=1, return_ps=return_ps)
    config = replace(
        config,
        rx=replace(
            config.rx,
            ingress_rate_bytes_per_second=10_000_000_000,
        ),
    )
    result = run(
        [transfer(f"sender-{i}", source=i, destination=3) for i in (0, 1, 2)], config=config
    )
    assert result.logical_bytes == 3 * 1024
    assert_capacity_conservation(result)
    packets = sorted(result.packets, key=lambda p: p.rx_started_at_ps)
    assert all(a.rx_finished_at_ps <= b.rx_started_at_ps for a, b in pairwise(packets))
    visits = {(v.packet_id, v.buffer_id): v for v in result.buffer_visits}
    for packet in result.packets:
        receive = visits[(packet.packet_id, f"rx:3:{packet.virtual_channel}")]
        assert receive.arrived_at_ps == packet.rx_buffer_accepted_at_ps
        assert receive.released_at_ps == packet.rx_buffer_released_at_ps
        release = next(r for r in result.credit_releases if r.packet_id == packet.packet_id)
        if queued:
            assert release.buffer_id.startswith("switch:")
            assert release.buffer_released_at_ps == packet.switch_finished_at_ps
            assert receive.reserved_at_ps == packet.switch_started_at_ps
        else:
            assert release.buffer_id == receive.buffer_id
            assert receive.reserved_at_ps == packet.tx_started_at_ps
        assert release.credit_available_at_ps == packet.credit_available_at_ps
    if queued:
        for field in ("input_port", "output_port"):
            for port in range(4):
                grants = sorted(
                    (g for g in result.switch_grants if getattr(g, field) == port),
                    key=lambda g: g.started_at_ps,
                )
                assert all(a.finished_at_ps <= b.started_at_ps for a, b in pairwise(grants))


def test_identity_arbitration_ignores_classes_under_backpressure():
    config = profile(queued=True, capacity=1)
    inputs = [transfer(f"source-{i}", source=i, destination=3) for i in (0, 1, 2)]
    relabeled = [replace(t, traffic_class=NvlinkTrafficClass.RESPONSE) for t in inputs]
    first = run(inputs, config=config)
    second = run(relabeled, config=config)
    assert first.buffer_visits == second.buffer_visits
    assert first.switch_grants == second.switch_grants
    assert first.visibility_events == second.visibility_events
    assert [replace(p, traffic_class=NvlinkTrafficClass.RESPONSE) for p in first.packets] == list(
        second.packets
    )


@pytest.mark.parametrize("queued", [False, True])
def test_replay_owns_one_reservation_per_hop_and_one_visibility(queued):
    config = profile(queued=queued, capacity=1, credits=1)
    clean = run([transfer(size=256)], config=config)
    packet_id = clean.packets[0].packet_id
    replayed = run(
        [transfer(size=256)],
        config=config,
        options=NvlinkAlignedOptions(
            replay_counts=((packet_id, 1),),
            replay_timeout_ps=100,
            acknowledgement_latency_ps=1_000_000,
        ),
    )
    assert replayed.total_wire_bytes == clean.total_wire_bytes + 272
    assert replayed.completion_time_ps == clean.completion_time_ps + 10880 + 100
    assert len(replayed.buffer_visits) == (2 if queued else 1)
    assert replayed.logical_bytes == clean.logical_bytes
    assert replayed.physical_drain_time_ps > replayed.completion_time_ps
    assert_capacity_conservation(replayed)


def test_invalid_requests_fail_before_engine_can_publish_evidence():
    config = profile()
    with pytest.raises(ValueError, match="unknown packet"):
        run(
            [transfer()],
            config=config,
            options=NvlinkAlignedOptions(
                replay_counts=(("absent", 1),),
            ),
        )
    with pytest.raises(ValueError, match="globally unique"):
        run([transfer(), transfer()])
    invalid = replace(config, rx=replace(config.rx, buffer_capacity_bytes=271))
    with pytest.raises(ValueError, match="exceeds declared NVLink RX buffer"):
        run([transfer()], config=invalid)
    with pytest.raises(ValueError, match="strictly increasing"):
        run(
            [
                replace(transfer("a"), ordering_domain="shared"),
                replace(transfer("b"), ordering_domain="shared"),
            ]
        )


def test_no_profile_is_exact_object_identity_and_engine_is_one_transaction():
    sentinel = object()
    assert NvlinkDomainService().serve_aligned([], analytic_result=sentinel) is sentinel
    config = profile()
    engine = NvlinkCausalEngine(
        config,
        NvlinkAlignedOptions(
            flow_control=NvlinkFlowControlConfig.from_candidate_profile(config),
        ),
    )
    result = engine.serve([transfer()], include_switch=True)
    assert_capacity_conservation(result)
    with pytest.raises(RuntimeError, match="second transaction"):
        engine.serve([transfer()], include_switch=True)


@pytest.mark.parametrize("endpoint_rate", [1_000_000_000, 25_000_000_000, 100_000_000_000])
def test_source_feed_cannot_finish_after_its_packet_becomes_visible(endpoint_rate):
    config = profile()
    config = replace(
        config,
        tx=replace(
            config.tx,
            endpoint_egress_rate_bytes_per_second=endpoint_rate,
        ),
    )
    result = run([transfer(size=256)], config=config)
    source_floor = (272 * 10**12 + endpoint_rate - 1) // endpoint_rate
    packet = result.packets[0]
    assert packet.tx_finished_at_ps == max(source_floor, 10880)
    assert packet.visible_at_ps == max(source_floor, 10880) + 2720
    assert_capacity_conservation(result)


def test_blocking_replay_retains_source_egress_through_actual_retry_interval():
    config = profile(rate=25_000_000_000)
    config = replace(
        config,
        tx=replace(
            config.tx,
            endpoint_egress_rate_bytes_per_second=25_000_000_000,
        ),
    )
    first = transfer("retry", size=256)
    packet_id = run([first], config=config).packets[0].packet_id
    result = run(
        [first, transfer("other", destination=2, size=256, release=110880)],
        config=config,
        options=NvlinkAlignedOptions(
            replay_counts=((packet_id, 1),),
            replay_timeout_ps=100000,
        ),
    )
    retry = next(p for p in result.packets if p.extent_id == "retry")
    other = next(p for p in result.packets if p.extent_id == "other")
    assert retry.tx_finished_at_ps == 121760
    assert other.tx_started_at_ps == retry.tx_finished_at_ps
    assert_capacity_conservation(result)


def test_replay_rounds_each_attempt_instead_of_rounding_the_combined_bytes():
    config = profile(rate=13_000_000_000)
    first = transfer(size=256)
    packet_id = run([first], config=config).packets[0].packet_id
    result = run(
        [first],
        config=config,
        options=NvlinkAlignedOptions(
            replay_counts=((packet_id, 1),),
            replay_timeout_ps=100,
        ),
    )
    assert (
        result.packets[0].tx_finished_at_ps
        == 2 * ((272 * 10**12 + 13_000_000_000 - 1) // 13_000_000_000) + 100
    )
    assert result.packets[0].tx_finished_at_ps == 41948


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_packet_eligibility_is_a_validated_integer_timestamp(value):
    packet = run([transfer(size=256)]).packets[0]
    with pytest.raises((TypeError, ValueError)):
        replace(packet, tx_eligible_at_ps=value)
    with pytest.raises(ValueError, match="eligibility cannot precede"):
        replace(packet, released_at_ps=1, tx_eligible_at_ps=0)
    with pytest.raises(ValueError, match="grant cannot precede"):
        replace(packet, tx_eligible_at_ps=2, tx_started_at_ps=1)


@pytest.fixture
def study_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "nvlink_causal_checker_mutations",
        ROOT / "examples/nvlink_causal_service_v1/run_study.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def study_checker(study_module):
    return study_module.guard_findings


@pytest.mark.parametrize("mutation", ["capacity", "reservation", "visibility", "credit-source"])
def test_evidence_checker_rejects_self_consistent_but_unbound_projections(study_checker, mutation):
    config = profile()
    inputs = [transfer(size=256)]
    result = run(inputs, config=config)
    assert study_checker(result, config, inputs) == []
    if mutation == "capacity":
        result = replace(
            result, buffer_visits=(replace(result.buffer_visits[0], capacity_bytes=999999),)
        )
    elif mutation == "reservation":
        result = replace(
            result, buffer_visits=(replace(result.buffer_visits[0], reserved_at_ps=1),)
        )
    elif mutation == "visibility":
        result = replace(
            result,
            visibility_events=(
                replace(result.visibility_events[0], visible_at_ps=result.completion_time_ps + 1),
            ),
        )
    else:
        result = replace(result, credit_releases=(replace(result.credit_releases[0], source=2),))
    assert study_checker(result, config, inputs)


def test_evidence_checker_reports_missing_read_control_without_raising(study_checker):
    config = profile()
    inputs = [transfer(size=256, read=True)]
    result = run(inputs, config=config)
    assert study_checker(result, config, inputs) == []
    result = replace(
        result,
        packets=tuple(p for p in result.packets if p.direction is NvlinkPacketDirection.RESPONSE),
    )
    findings = study_checker(result, config, inputs)
    assert "exact input packet identities" in findings
    assert any("read request present" in finding for finding in findings)


def test_fatal_cell_keeps_raw_evidence_and_writes_void_summary(study_module, tmp_path, monkeypatch):
    import json

    original_serve = study_module.NvlinkDomainService.serve_aligned
    original_git = study_module.git

    def remove_read_requests(self, inputs, **kwargs):
        result = original_serve(self, inputs, **kwargs)
        if not inputs:
            return result
        return replace(
            result,
            packets=tuple(
                p for p in result.packets if p.direction is not NvlinkPacketDirection.REQUEST
            ),
        )

    def local_test_source(*arguments):
        if arguments[0] == "show" and arguments[1].startswith("HEAD:"):
            return (ROOT / arguments[1].removeprefix("HEAD:")).read_bytes()
        return original_git(*arguments)

    monkeypatch.setattr(study_module.NvlinkDomainService, "serve_aligned", remove_read_requests)
    monkeypatch.setattr(study_module, "git", local_test_source)
    output = tmp_path / "malformed-cell"
    result = study_module.run_study(output)
    assert result["verdict"] == "VOID"
    assert result["behavioral_score"] is None
    assert any("read request present" in finding for finding in result["fatal_findings"])
    assert json.loads((output / "summary.json").read_text()) == result
    invalid = [row for row in result["configurations"] if row["fatal_verdict"] == "VOID"]
    assert invalid
    assert all((output / row["evidence_file"]).is_file() for row in invalid)
