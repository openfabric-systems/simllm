import dataclasses
from pathlib import Path

import pytest

from simllm.backends.htsim_nvlink import (
    NVLINK_CANDIDATE_EVIDENCE_CLASS,
    NvlinkCandidateProfile,
    NvlinkDomainResult,
    NvlinkDomainService,
    NvlinkFifoPlacement,
    NvlinkOperation,
    NvlinkPacketDirection,
    NvlinkRx,
    NvlinkSwitch,
    NvlinkSwitchConfig,
    NvlinkSwitchMode,
    NvlinkTransfer,
    NvlinkTx,
    load_nvlink_candidate_profile,
    sha256_file,
    validate_candidate_against_published_a100_envelope,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "a100_nvlink_packet_v1"
FREEZE_SHA256 = "212a7a26f54e444c9b18f1e528bd0d00b5a28e4f9e005b0dc137f477ad642571"


@pytest.fixture
def candidate() -> NvlinkCandidateProfile:
    return load_nvlink_candidate_profile(STUDY / "candidate-profile.json")


def test_candidate_handoff_is_digest_bound_and_not_measured(candidate):
    assert sha256_file(STUDY / "expectations.json") == FREEZE_SHA256
    assert candidate.freeze_sha256 == FREEZE_SHA256
    assert candidate.status == "candidate"
    assert candidate.evidence_class == NVLINK_CANDIDATE_EVIDENCE_CLASS
    assert candidate.switch.mode is NvlinkSwitchMode.PASS_THROUGH


def test_tx_packetizes_write_at_the_declared_boundary(candidate):
    packets = NvlinkTx(candidate.tx).packetize(
        NvlinkTransfer(extent_id="write", source=0, destination=1, payload_bytes=513)
    )

    assert [packet.payload_bytes for packet in packets] == [256, 256, 1]
    assert [packet.wire_bytes for packet in packets] == [272, 272, 17]
    assert {packet.direction for packet in packets} == {NvlinkPacketDirection.REQUEST}
    assert len({packet.attempt_id for packet in packets}) == 3


def test_tx_makes_peer_read_request_and_response_direction_explicit(candidate):
    packets = NvlinkTx(candidate.tx).packetize(
        NvlinkTransfer(
            extent_id="read",
            source=0,
            destination=2,
            payload_bytes=257,
            operation=NvlinkOperation.PEER_READ,
        )
    )

    assert (packets[0].source, packets[0].destination) == (0, 2)
    assert packets[0].direction is NvlinkPacketDirection.REQUEST
    assert packets[0].payload_bytes == 0
    assert [(packet.source, packet.destination) for packet in packets[1:]] == [
        (2, 0),
        (2, 0),
    ]
    assert {packet.direction for packet in packets[1:]} == {NvlinkPacketDirection.RESPONSE}
    assert [packet.payload_bytes for packet in packets[1:]] == [256, 1]


def test_tx_bonds_packets_over_four_directional_serializers(candidate):
    tx = NvlinkTx(candidate.tx)
    packets = tx.packetize(
        NvlinkTransfer(extent_id="bond", source=0, destination=1, payload_bytes=8 * 256)
    )
    sent = tx.transmit(packets, credit_return_latency_ps=candidate.rx.credit_return_latency_ps)

    assert [packet.link_index for packet in sent] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert all(
        packet.tx_finished_at_ps > packet.tx_started_at_ps
        for packet in sent
        if packet.tx_finished_at_ps is not None and packet.tx_started_at_ps is not None
    )


def test_pass_through_switch_returns_the_exact_packet_tuple(candidate):
    tx = NvlinkTx(candidate.tx)
    sent = tx.transmit(
        tx.packetize(NvlinkTransfer(extent_id="direct", source=0, destination=1, payload_bytes=64)),
        credit_return_latency_ps=candidate.rx.credit_return_latency_ps,
    )

    assert NvlinkSwitch(candidate.switch).forward(sent) is sent


def test_pass_through_composition_is_byte_identical_to_direct_tx_rx(candidate):
    transfers = [
        NvlinkTransfer(extent_id="a", source=0, destination=1, payload_bytes=4096),
        NvlinkTransfer(
            extent_id="b",
            source=2,
            destination=0,
            payload_bytes=513,
            operation=NvlinkOperation.PEER_READ,
        ),
    ]
    service = NvlinkDomainService(candidate)

    through_switch = service.serve(transfers, analytic_result=object(), include_switch=True)
    direct = service.serve(transfers, analytic_result=object(), include_switch=False)

    assert isinstance(through_switch, NvlinkDomainResult)
    assert isinstance(direct, NvlinkDomainResult)
    assert through_switch.canonical_json_bytes() == direct.canonical_json_bytes()


def test_queued_switch_owns_contention_and_head_of_line_delay(candidate):
    tx = NvlinkTx(candidate.tx)
    sent = tx.transmit(
        tx.packetize(
            NvlinkTransfer(extent_id="queued", source=0, destination=1, payload_bytes=512)
        ),
        credit_return_latency_ps=candidate.rx.credit_return_latency_ps,
    )
    switch = NvlinkSwitch(
        NvlinkSwitchConfig(
            mode=NvlinkSwitchMode.QUEUED,
            fifo_placement=NvlinkFifoPlacement.SHARED,
            service_rate_bytes_per_second=10_000_000_000,
            buffer_capacity_bytes=4096,
            arbitration="fifo",
            head_of_line_blocking=True,
        )
    )
    forwarded = switch.forward(sent)

    assert forwarded[0].switch_started_at_ps == sent[0].tx_finished_at_ps
    assert forwarded[1].switch_started_at_ps == forwarded[0].switch_finished_at_ps
    assert forwarded[-1].switch_finished_at_ps > sent[-1].tx_finished_at_ps


def test_pass_through_rejects_silent_fifo_parameters():
    with pytest.raises(ValueError, match="pass-through"):
        NvlinkSwitchConfig(
            mode=NvlinkSwitchMode.PASS_THROUGH,
            fifo_placement=NvlinkFifoPlacement.INPUT,
        )


def test_rx_rate_is_independently_parameterized(candidate):
    tx = NvlinkTx(candidate.tx)
    sent = tx.transmit(
        tx.packetize(NvlinkTransfer(extent_id="rx", source=0, destination=1, payload_bytes=4096)),
        credit_return_latency_ps=candidate.rx.credit_return_latency_ps,
    )
    fast_packets, _ = NvlinkRx(candidate.rx).receive(sent)
    slow_config = dataclasses.replace(candidate.rx, ingress_rate_bytes_per_second=50_000_000_000)
    slow_packets, _ = NvlinkRx(slow_config).receive(sent)

    assert slow_packets[-1].delivered_at_ps > fast_packets[-1].delivered_at_ps


def test_rx_buffer_tracks_backlog_and_enforces_its_own_capacity(candidate):
    tx = NvlinkTx(candidate.tx)
    sent = tx.transmit(
        tx.packetize(
            NvlinkTransfer(extent_id="backlog", source=0, destination=1, payload_bytes=4096)
        ),
        credit_return_latency_ps=candidate.rx.credit_return_latency_ps,
    )
    slow_config = dataclasses.replace(
        candidate.rx,
        ingress_rate_bytes_per_second=1_000_000_000,
    )

    _, max_occupancy = NvlinkRx(slow_config).receive(sent)

    assert max_occupancy > candidate.tx.max_payload_bytes + candidate.tx.header_bytes
    with pytest.raises(ValueError, match="buffer occupancy"):
        NvlinkRx(dataclasses.replace(slow_config, buffer_capacity_bytes=400)).receive(sent)


def test_analytic_bypass_returns_the_callers_object_by_identity():
    analytic = {"accepted": [1, 2, 3], "duration_ps": 17}
    result = NvlinkDomainService().serve(
        [NvlinkTransfer(extent_id="off", source=0, destination=1, payload_bytes=1)],
        analytic_result=analytic,
    )

    assert result is analytic


def test_composition_conserves_write_and_read_directions(candidate):
    result = NvlinkDomainService(candidate).serve(
        [
            NvlinkTransfer(extent_id="w", source=0, destination=1, payload_bytes=513),
            NvlinkTransfer(
                extent_id="r",
                source=0,
                destination=2,
                payload_bytes=257,
                operation=NvlinkOperation.PEER_READ,
            ),
        ],
        analytic_result=None,
    )

    assert isinstance(result, NvlinkDomainResult)
    assert result.logical_bytes == 770
    assert result.request_payload_bytes == 513
    assert result.response_payload_bytes == 257
    assert result.request_wire_bytes == 513 + 4 * 16
    assert result.response_wire_bytes == 257 + 2 * 16
    assert all(packet.delivered_at_ps is not None for packet in result.packets)


def test_declared_candidate_contains_the_published_envelope(candidate):
    validation = validate_candidate_against_published_a100_envelope(candidate)

    assert validation.within_registered_error
    assert 94.0 <= validation.predicted_pair_payload_rate_gbps <= 94.07
    assert validation.pair_worst_relative_error < 0.0008
    assert validation.predicted_fanout_payload_rate_gbps == pytest.approx(281.6991815868504)
    assert validation.fanout_relative_error < 0.0002
