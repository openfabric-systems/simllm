import copy
import dataclasses
<<<<<<< HEAD
import json
=======
import re
from collections import Counter
from itertools import pairwise
>>>>>>> origin/main
from pathlib import Path
from xml.etree import ElementTree

import pytest

from simllm.backends.htsim_nvlink import (
    NVLINK_SCORED_EVIDENCE_CLASS,
    NVLINK_SCORED_PROFILE_STATUS,
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
<<<<<<< HEAD
TRAF65_FREEZE_SHA256 = (
    "212a7a26f54e444c9b18f1e528bd0d00b5a28e4f9e005b0dc137f477ad642571"
)
TRAF70_FREEZE_SHA256 = (
    "f0ab026e054873a56614af63ab3a7ae3219dc0b045423808cb41522910fa6da6"
)
PUBLISHED_PROFILE_SHA256 = (
    "d33ef5b2c6fa87cc97e1e7b45a43a841a5da45f5462311e3981fbc903c56deb2"
)
=======
FIGURE = ROOT / "resources" / "figures" / "nvlink-domain-model.svg"
FREEZE_SHA256 = "212a7a26f54e444c9b18f1e528bd0d00b5a28e4f9e005b0dc137f477ad642571"
SVG = "{http://www.w3.org/2000/svg}"
PATH_TOKEN = re.compile(r"[MHV]|-?(?:\d+(?:\.\d*)?|\.\d+)")


def _orthogonal_points(path: ElementTree.Element) -> tuple[tuple[float, float], ...]:
    data = path.attrib["d"]
    tokens = PATH_TOKEN.findall(data)
    remainder = PATH_TOKEN.sub("", data)
    assert not remainder.replace(",", "").strip(), data
    assert tokens.count("M") == 1, data
    assert tokens[0] == "M", data
    assert all(token not in {"L", "C", "Q", "S", "T", "A", "Z"} for token in tokens)

    points: list[tuple[float, float]] = []
    index = 0
    current: tuple[float, float] | None = None
    while index < len(tokens):
        command = tokens[index]
        index += 1
        if command == "M":
            current = (float(tokens[index]), float(tokens[index + 1]))
            index += 2
        elif command == "H":
            assert current is not None
            current = (float(tokens[index]), current[1])
            index += 1
        elif command == "V":
            assert current is not None
            current = (current[0], float(tokens[index]))
            index += 1
        else:
            raise AssertionError(f"non-orthogonal SVG command {command!r} in {data!r}")
        points.append(current)
    return tuple(points)


def _rect_box(rect: ElementTree.Element) -> tuple[float, float, float, float]:
    left = float(rect.attrib["x"])
    top = float(rect.attrib["y"])
    return (
        left,
        top,
        left + float(rect.attrib["width"]),
        top + float(rect.attrib["height"]),
    )


def _text_box(text: ElementTree.Element) -> tuple[float, float, float, float]:
    """Return a conservative box without depending on an installed font."""

    content = "".join(text.itertext())
    size = float(text.attrib["font-size"])
    width = max(size, len(content) * size * 0.58)
    x = float(text.attrib["x"])
    y = float(text.attrib["y"])
    anchor = text.attrib.get("text-anchor", "start")
    if anchor == "middle":
        left = x - width / 2
    elif anchor == "end":
        left = x - width
    else:
        left = x
    if text.attrib.get("dominant-baseline") == "central":
        top = y - size * 0.6
    else:
        top = y - size
    return left, top, left + width, top + size * 1.2


def _segment_crosses_box(
    start: tuple[float, float],
    end: tuple[float, float],
    box: tuple[float, float, float, float],
) -> bool:
    left, top, right, bottom = box
    if start[0] == end[0]:
        low, high = sorted((start[1], end[1]))
        return left < start[0] < right and max(low, top) < min(high, bottom)
    if start[1] == end[1]:
        low, high = sorted((start[0], end[0]))
        return top < start[1] < bottom and max(low, left) < min(high, right)
    raise AssertionError(f"non-orthogonal segment {start!r} to {end!r}")


def _point_on_box_boundary(
    point: tuple[float, float], box: tuple[float, float, float, float]
) -> bool:
    left, top, right, bottom = box
    return (
        left <= point[0] <= right
        and top <= point[1] <= bottom
        and (point[0] in {left, right} or point[1] in {top, bottom})
    )
>>>>>>> origin/main


@pytest.fixture
def candidate() -> NvlinkCandidateProfile:
    return load_nvlink_candidate_profile(STUDY / "candidate-profile.json")


def test_scored_handoff_keeps_parameter_evidence_distinct(candidate):
    assert sha256_file(STUDY / "expectations.json") == TRAF65_FREEZE_SHA256
    assert sha256_file(STUDY / "candidate-profile.json") == PUBLISHED_PROFILE_SHA256
    assert candidate.freeze_sha256 == TRAF70_FREEZE_SHA256
    assert candidate.status == NVLINK_SCORED_PROFILE_STATUS
    assert candidate.evidence_class == NVLINK_SCORED_EVIDENCE_CLASS
    assert candidate.switch.mode is NvlinkSwitchMode.PASS_THROUGH
    assert candidate.tx.endpoint_egress_rate_bytes_per_second == 160_795_737_454
    assert candidate.rx.ingress_rate_bytes_per_second == 207_101_921_876

    tx_rate = candidate.evidence_for("tx", "endpoint_egress_rate_bytes_per_second")
    rx_rate = candidate.evidence_for("rx", "ingress_rate_bytes_per_second")
    payload = candidate.evidence_for("tx", "max_payload_bytes")
    switch = candidate.evidence_for("switch", "mode")
    assert (tx_rate.status, tx_rate.candidate_relation) == (
        "IDENTIFIED",
        "REFUTED_AND_REPLACED",
    )
    assert (rx_rate.status, rx_rate.candidate_relation) == (
        "IDENTIFIED",
        "REFUTED_AND_REPLACED",
    )
    assert payload.status == "INCONCLUSIVE"
    assert switch.status == "STRUCTURAL"


def test_scored_handoff_rejects_runtime_values_without_matching_evidence(tmp_path):
    raw = json.loads((STUDY / "candidate-profile.json").read_text(encoding="utf-8"))
    changed = copy.deepcopy(raw)
    changed["tx"]["endpoint_egress_rate_bytes_per_second"] += 1
    path = tmp_path / "changed-profile.json"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(changed, handle)
        handle.write("\n")

    with pytest.raises(ValueError, match="does not match runtime parameter"):
        load_nvlink_candidate_profile(path)


def test_scored_handoff_rejects_unpublished_identification(tmp_path):
    raw = json.loads((STUDY / "candidate-profile.json").read_text(encoding="utf-8"))
    raw["traf70_score_publication"]["runtime_changes"] = raw[
        "traf70_score_publication"
    ]["runtime_changes"][1:]
    path = tmp_path / "missing-publication.json"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(raw, handle)
        handle.write("\n")

    with pytest.raises(ValueError, match="exactly match identified parameters"):
        load_nvlink_candidate_profile(path)


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


def test_scored_profile_reports_its_derived_published_envelope_comparison(candidate):
    validation = validate_candidate_against_published_a100_envelope(candidate)

<<<<<<< HEAD
    assert not validation.within_registered_error
    assert validation.predicted_pair_payload_rate_gbps == pytest.approx(94.009808228512)
    assert validation.pair_worst_relative_error < 0.0007
    assert validation.predicted_fanout_payload_rate_gbps == pytest.approx(151.14754255896753)
    assert validation.fanout_relative_error == pytest.approx(0.4633497512552191)
=======
    assert validation.within_registered_error
    assert 94.0 <= validation.predicted_pair_payload_rate_gbps <= 94.07
    assert validation.pair_worst_relative_error < 0.0008
    assert validation.predicted_fanout_payload_rate_gbps == pytest.approx(281.6991815868504)
    assert validation.fanout_relative_error < 0.0002


def test_domain_figure_routes_are_continuous_and_clear_of_blocks_and_text():
    root = ElementTree.parse(FIGURE).getroot()
    routes = root.findall(f".//{SVG}path[@data-logical-path]")
    assert routes == root.findall(f"./{SVG}path")
    assert not root.findall(f".//{SVG}line")
    assert not root.findall(f".//{SVG}polyline")
    route_names = [route.attrib["data-logical-path"] for route in routes]
    assert Counter(route_names) == Counter(
        {
            "tx-staging-packetizer",
            "tx-packetizer-credit",
            "tx-credit-bond",
            "switch-pass",
            "switch-fifo-a",
            "switch-fifo-b",
            "switch-fifo-other",
            "switch-egress",
            "rx-ingress-reassembly",
            "rx-reassembly-delivery",
            "rx-delivery-credit",
            "packet-tx-switch",
            "packet-switch-rx",
            "credit-return",
            "request-direction",
            "response-direction",
        }
    )

    rects_by_id = {
        rect.attrib["id"]: rect
        for rect in root.findall(f".//{SVG}rect[@id]")
    }
    non_obstacle_kinds = {"canvas", "frame", "container", "lane", "decoration"}
    obstacle_rects = [
        rect
        for rect in root.findall(f".//{SVG}rect")
        if rect.attrib.get("data-geometry") not in non_obstacle_kinds
    ]
    text_boxes = [
        ("".join(text.itertext()), _text_box(text))
        for text in root.findall(f".//{SVG}text")
    ]

    for route in routes:
        points = _orthogonal_points(route)
        assert len(points) >= 2
        for attribute, endpoint in (
            ("data-start-rect", points[0]),
            ("data-end-rect", points[-1]),
        ):
            rect_id = route.attrib.get(attribute)
            if rect_id is not None:
                assert _point_on_box_boundary(endpoint, _rect_box(rects_by_id[rect_id]))

        for start, end in pairwise(points):
            for rect in obstacle_rects:
                assert not _segment_crosses_box(start, end, _rect_box(rect)), (
                    route.attrib["id"],
                    rect.attrib.get("id", "unnamed-rect"),
                )
            for content, box in text_boxes:
                assert not _segment_crosses_box(start, end, box), (
                    route.attrib["id"],
                    content,
                )

    credit_routes = [
        route for route in routes if route.attrib["data-logical-path"] == "credit-return"
    ]
    assert len(credit_routes) == 1
    assert _orthogonal_points(credit_routes[0]) == (
        (1250.0, 649.0),
        (1215.0, 649.0),
        (1215.0, 820.0),
        (625.0, 820.0),
        (625.0, 470.0),
        (550.0, 470.0),
    )
>>>>>>> origin/main
