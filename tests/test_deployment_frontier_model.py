import sys
from dataclasses import asdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.deployment_frontier_v1.frontier import (
    account_step,
    bottleneck_classification,
    compact_nvlink_ring_service,
    ideal_network_service,
    intra_node_attribution,
    kernel_service,
    load_expectations,
    operating_point,
    partition_network_bytes,
)
from simllm.backends.htsim_nvlink import (
    NvlinkDomainResult,
    NvlinkDomainService,
    NvlinkTransfer,
    load_nvlink_candidate_profile,
)

PROFILE = ROOT / "examples" / "a100_nvlink_packet_v1" / "candidate-profile.json"


@pytest.mark.parametrize("gpu_name", ["h100", "b100"])
@pytest.mark.parametrize("batch_per_gpu", [1, 2, 4, 8, 16, 32])
def test_roofline_provider_matches_frozen_floor_arithmetic(gpu_name, batch_per_gpu):
    service = kernel_service(
        load_expectations(),
        gpu_name=gpu_name,
        batch_per_gpu=batch_per_gpu,
    )

    assert service["kernel_floor_ps"] == max(
        service["compute_floor_ps"],
        service["memory_floor_ps"],
    )
    assert service["provider"] == "RooflineProvider"
    assert service["provider_efficiency"] == 1.0
    assert service["kernel_simulation_enabled"] is False


def test_every_frozen_byte_partition_is_exact():
    frozen = load_expectations()

    for configuration in frozen["configurations"]:
        for batch_per_gpu in frozen["batch_per_gpu_sweep"]:
            partition = partition_network_bytes(
                frozen,
                configuration=configuration,
                batch_per_gpu=batch_per_gpu,
            )
            assert (
                partition["local_logical_bytes_per_transfer"]
                + sum(partition["remote_flow_payload_bytes"])
                == partition["total_logical_bytes"]
            )
            ideal = ideal_network_service(frozen, partition)
            assert ideal["ideal_fabric_wire_ps"] >= 0
            assert ideal["ideal_intra_node_wire_ps"] > 0


@pytest.mark.parametrize("payload_bytes", [1, 257, 4097, 65_537])
def test_compact_candidate_ring_equals_packet_object_service(payload_bytes):
    profile = load_nvlink_candidate_profile(PROFILE)
    compact = compact_nvlink_ring_service(
        profile,
        payload_bytes_per_transfer=payload_bytes,
        transfer_count=4,
    )
    packet_result = NvlinkDomainService(profile).serve(
        [
            NvlinkTransfer(
                extent_id=f"ring-{source}",
                source=source,
                destination=(source + 1) % 4,
                payload_bytes=payload_bytes,
            )
            for source in range(4)
        ],
        analytic_result=None,
    )

    assert isinstance(packet_result, NvlinkDomainResult)
    assert compact.logical_bytes == packet_result.logical_bytes
    assert compact.request_payload_bytes == packet_result.request_payload_bytes
    assert compact.response_payload_bytes == packet_result.response_payload_bytes
    assert compact.request_wire_bytes == packet_result.request_wire_bytes
    assert compact.response_wire_bytes == packet_result.response_wire_bytes
    assert compact.max_packet_tx_finish_ps == max(
        packet.tx_finished_at_ps or 0 for packet in packet_result.packets
    )
    assert compact.max_packet_switch_finish_ps == compact.max_packet_tx_finish_ps
    assert compact.completion_time_ps == packet_result.completion_time_ps
    assert compact.max_rx_buffer_occupancy_bytes == packet_result.max_rx_buffer_occupancy_bytes


def test_eight_gpu_mapping_is_two_identical_candidate_domains():
    profile = load_nvlink_candidate_profile(PROFILE)
    compact = compact_nvlink_ring_service(
        profile,
        payload_bytes_per_transfer=4097,
        transfer_count=8,
    )
    four = compact_nvlink_ring_service(
        profile,
        payload_bytes_per_transfer=4097,
        transfer_count=4,
    )

    assert compact.domain_count == 2
    assert compact.completion_time_ps == four.completion_time_ps
    for field in (
        "logical_bytes",
        "request_payload_bytes",
        "request_wire_bytes",
    ):
        assert asdict(compact)[field] == 2 * asdict(four)[field]


def test_candidate_module_terms_sum_to_raw_excess():
    profile = load_nvlink_candidate_profile(PROFILE)
    payload = 65_537
    compact = compact_nvlink_ring_service(
        profile,
        payload_bytes_per_transfer=payload,
        transfer_count=8,
    )
    terms = intra_node_attribution(ideal_wire_ps=payload * 10, result=compact)

    assert terms["tx_credits_ps"] >= 0
    assert terms["switch_contention_ps"] == 0
    assert terms["rx_return_ps"] >= 0
    assert (
        terms["tx_credits_ps"]
        + terms["switch_contention_ps"]
        + terms["rx_return_ps"]
        == terms["raw_excess_ps"]
    )


@pytest.mark.parametrize(
    ("kernel", "fabric", "intra", "expected"),
    [
        (10, 9, 8, "neither"),
        (8, 10, 9, "inter-node"),
        (8, 9, 10, "intra-node"),
        (10, 10, 9, "co-critical"),
    ],
)
def test_bottleneck_classification(kernel, fabric, intra, expected):
    observed = bottleneck_classification(
        kernel_floor_ps=kernel,
        simulated_fabric_ps=fabric,
        simulated_intra_node_ps=intra,
    )

    assert observed["classification"] == expected


def test_accounting_identity_telescopes_exactly():
    accounting = account_step(
        kernel_floor_ps=100,
        ideal_fabric_wire_ps=70,
        ideal_intra_node_wire_ps=80,
        simulated_fabric_ps=130,
        simulated_intra_node_ps=170,
    )

    assert accounting == {
        "analytical_step_ps": 100,
        "after_inter_node_ps": 130,
        "simulated_step_ps": 170,
        "inter_node_attributed_ps": 30,
        "intra_node_attributed_ps": 40,
        "residual_ps": 0,
    }


def test_operating_point_preserves_exact_axis_ratio():
    point = operating_point(batch_per_gpu=32, step_time_ps=10_000_000_000)

    assert point["x_tokens_per_second_per_request"] == {
        "numerator": 100,
        "denominator": 1,
        "decimal": 100.0,
    }
    assert point["y_tokens_per_second_per_gpu"] == {
        "numerator": 3200,
        "denominator": 1,
        "decimal": 3200.0,
    }
