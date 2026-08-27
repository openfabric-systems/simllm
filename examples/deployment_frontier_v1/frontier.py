"""Exact arithmetic for the frozen CORE-62 and TRAF-68 frontier study."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

PICOSECONDS_PER_SECOND = 1_000_000_000_000
STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
EXPECTATIONS_PATH = STUDY_DIR / "expectations.json"


def load_expectations() -> dict[str, Any]:
    """Load the expectations committed before implementation and observation."""

    return json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def ceil_div(numerator: int, denominator: int) -> int:
    """Return an exact integer ceiling for positive arithmetic."""

    if numerator < 0 or denominator <= 0:
        raise ValueError("ceil_div requires a nonnegative numerator and positive denominator")
    return (numerator + denominator - 1) // denominator


def fraction_record(numerator: int, denominator: int) -> dict[str, int | float]:
    """Make an exact rational record with a display-only decimal projection."""

    value = Fraction(numerator, denominator)
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def kernel_service(
    frozen: dict[str, Any],
    *,
    gpu_name: str,
    batch_per_gpu: int,
) -> dict[str, Any]:
    """Price the declared per-rank work through the installed roofline provider."""

    from simllm.compute import GPU_ENVELOPES, KernelSpec, RooflineProvider

    inventory = frozen["model_inventory"]
    envelope = frozen["gpu_envelopes"][gpu_name]
    flops = inventory["flops_per_batch_item"] * batch_per_gpu
    logical_hbm_bytes = (
        inventory["static_logical_hbm_bytes"]
        + inventory["dynamic_hbm_bytes_per_batch_item"] * batch_per_gpu
    )
    provider = RooflineProvider(efficiency=frozen["gpu_envelopes"]["efficiency"])
    estimate = provider.estimate(
        KernelSpec(
            name="deepseek-v3-decode-rank-class-0",
            flops=flops,
            bytes_moved=logical_hbm_bytes,
            config=(("batch_per_gpu", batch_per_gpu),),
        ),
        GPU_ENVELOPES[gpu_name],
    )
    compute_floor_ps = int(
        flops
        / (envelope["peak_flops_per_second"] * frozen["gpu_envelopes"]["efficiency"])
        * PICOSECONDS_PER_SECOND
    )
    memory_floor_ps = int(
        logical_hbm_bytes
        / (envelope["hbm_bytes_per_second"] * frozen["gpu_envelopes"]["efficiency"])
        * PICOSECONDS_PER_SECOND
    )
    if estimate.duration_ps != max(compute_floor_ps, memory_floor_ps):
        raise AssertionError("RooflineProvider disagrees with the frozen envelope arithmetic")
    if estimate.bound not in {"compute", "memory"}:
        raise AssertionError("RooflineProvider returned an unexpected bound")
    return {
        "flops": flops,
        "logical_hbm_bytes": logical_hbm_bytes,
        "compute_floor_ps": compute_floor_ps,
        "memory_floor_ps": memory_floor_ps,
        "kernel_floor_ps": estimate.duration_ps,
        "kernel_bound": estimate.bound,
        "provider": "RooflineProvider",
        "provider_efficiency": provider.efficiency,
        "kernel_simulation_enabled": False,
    }


def partition_network_bytes(
    frozen: dict[str, Any],
    *,
    configuration: dict[str, Any],
    batch_per_gpu: int,
) -> dict[str, Any]:
    """Partition every declared logical byte into local and remote extents."""

    per_item = frozen["model_inventory"]["network_geometry"][
        "logical_collective_bytes_per_gpu_per_batch_item"
    ]
    total = batch_per_gpu * per_item
    denominator = configuration["local_fraction_denominator"]
    local = total // denominator
    remote = total - local
    fan_in = configuration["inter_node_fan_in"]
    if fan_in == 0:
        if remote != 0:
            raise AssertionError("a zero-fan-in configuration retained remote bytes")
        flows: list[int] = []
    else:
        quotient, remainder = divmod(remote, fan_in)
        flows = [quotient + (index < remainder) for index in range(fan_in)]
    if local + sum(flows) != total:
        raise AssertionError("logical network byte partition does not conserve")
    if any(payload <= 0 for payload in flows):
        raise AssertionError("the frozen remote split produced an empty flow")
    return {
        "total_logical_bytes": total,
        "local_logical_bytes_per_transfer": local,
        "remote_logical_bytes": remote,
        "remote_flow_payload_bytes": flows,
    }


@dataclass(frozen=True)
class CompactNvlinkResult:
    """Exact summary of independent four-GPU candidate ring domains."""

    transfer_count: int
    domain_count: int
    payload_bytes_per_transfer: int
    packets_per_transfer: int
    logical_bytes: int
    request_payload_bytes: int
    response_payload_bytes: int
    request_wire_bytes: int
    response_wire_bytes: int
    max_packet_tx_finish_ps: int
    max_packet_switch_finish_ps: int
    completion_time_ps: int
    max_rx_buffer_occupancy_bytes: int


def compact_nvlink_ring_service(
    profile: Any,
    *,
    payload_bytes_per_transfer: int,
    transfer_count: int,
) -> CompactNvlinkResult:
    """Replay the candidate packet rules without retaining packet objects.

    The frozen eight-GPU placement is represented by two independent copies of
    the candidate's four-endpoint direct-mesh domain. Each copy carries one
    equal-sized ordered ring transfer per endpoint. Those transfers have no
    shared source, destination, or link serializer, so one transfer is the
    exact timing representative. Logical and wire ledgers are multiplied by
    the declared transfer count.
    """

    from simllm.backends.htsim_nvlink import NvlinkSwitchMode

    if payload_bytes_per_transfer <= 0:
        raise ValueError("payload_bytes_per_transfer must be positive")
    if transfer_count <= 0 or transfer_count % 4:
        raise ValueError("transfer_count must be a positive multiple of four")
    if profile.switch.mode is not NvlinkSwitchMode.PASS_THROUGH:
        raise ValueError("compact ring replay requires the frozen pass-through switch")

    tx = profile.tx
    rx = profile.rx
    packet_count = ceil_div(payload_bytes_per_transfer, tx.max_payload_bytes)
    link_cursors = [0] * tx.links_per_peer
    endpoint_cursor = 0
    credit_slots = [0] * tx.credits_per_destination
    rx_cursor = 0
    buffered: deque[tuple[int, int]] = deque()
    occupancy = 0
    max_occupancy = 0
    total_wire_bytes = 0
    max_tx_finish = 0
    completion = 0

    for sequence in range(packet_count):
        remaining = payload_bytes_per_transfer - sequence * tx.max_payload_bytes
        payload_bytes = min(tx.max_payload_bytes, remaining)
        wire_bytes = payload_bytes + tx.header_bytes
        link_index = min(
            range(tx.links_per_peer),
            key=lambda candidate: (link_cursors[candidate], candidate),
        )
        started_at_ps = max(
            link_cursors[link_index],
            endpoint_cursor,
            credit_slots[sequence % tx.credits_per_destination],
        )
        tx_finished_at_ps = started_at_ps + ceil_div(
            wire_bytes * PICOSECONDS_PER_SECOND,
            tx.per_link_rate_bytes_per_second,
        )
        endpoint_cursor = started_at_ps + ceil_div(
            wire_bytes * PICOSECONDS_PER_SECOND,
            tx.endpoint_egress_rate_bytes_per_second,
        )
        link_cursors[link_index] = tx_finished_at_ps
        credit_slots[sequence % tx.credits_per_destination] = (
            tx_finished_at_ps + rx.credit_return_latency_ps
        )

        while buffered and buffered[0][0] <= tx_finished_at_ps:
            _, released_bytes = buffered.popleft()
            occupancy -= released_bytes
        occupancy += wire_bytes
        if occupancy > rx.buffer_capacity_bytes:
            raise AssertionError("compact candidate replay exceeded the RX buffer")
        rx_started_at_ps = max(tx_finished_at_ps, rx_cursor)
        completion = rx_started_at_ps + ceil_div(
            wire_bytes * PICOSECONDS_PER_SECOND,
            rx.ingress_rate_bytes_per_second,
        )
        rx_cursor = completion
        buffered.append((completion, wire_bytes))
        max_occupancy = max(max_occupancy, occupancy)
        max_tx_finish = max(max_tx_finish, tx_finished_at_ps)
        total_wire_bytes += wire_bytes

    return CompactNvlinkResult(
        transfer_count=transfer_count,
        domain_count=transfer_count // 4,
        payload_bytes_per_transfer=payload_bytes_per_transfer,
        packets_per_transfer=packet_count,
        logical_bytes=payload_bytes_per_transfer * transfer_count,
        request_payload_bytes=payload_bytes_per_transfer * transfer_count,
        response_payload_bytes=0,
        request_wire_bytes=total_wire_bytes * transfer_count,
        response_wire_bytes=0,
        max_packet_tx_finish_ps=max_tx_finish,
        max_packet_switch_finish_ps=max_tx_finish,
        completion_time_ps=completion,
        max_rx_buffer_occupancy_bytes=max_occupancy,
    )


def ideal_network_service(
    frozen: dict[str, Any],
    partition: dict[str, Any],
) -> dict[str, int]:
    """Return the frozen exact-byte, nominal-rate network floors."""

    flows = partition["remote_flow_payload_bytes"]
    max_flow = max(flows, default=0)
    fabric_rate = frozen["network_inputs"]["fabric"][
        "nominal_link_rate_bits_per_second"
    ]
    ideal_fabric = max_flow * 8 * PICOSECONDS_PER_SECOND // fabric_rate
    local = partition["local_logical_bytes_per_transfer"]
    intra_rate = frozen["network_inputs"]["intra_node"][
        "nominal_ideal_pair_rate_bytes_per_second"
    ]
    ideal_intra = local * PICOSECONDS_PER_SECOND // intra_rate
    return {
        "ideal_fabric_wire_ps": ideal_fabric,
        "ideal_intra_node_wire_ps": ideal_intra,
    }


def fabric_attribution(
    *,
    ideal_wire_ps: int,
    isolated_service_ps: int,
    concurrent_service_ps: int,
) -> dict[str, Any]:
    """Extract the frozen fabric protocol, serialization, and incast terms."""

    if min(ideal_wire_ps, isolated_service_ps, concurrent_service_ps) < 0:
        raise ValueError("fabric service times must be nonnegative")
    protocol_ps = min(max(0, isolated_service_ps - ideal_wire_ps), 2_000_000)
    serialization_ps = max(0, isolated_service_ps - ideal_wire_ps - protocol_ps)
    incast_ps = max(0, concurrent_service_ps - isolated_service_ps)
    terms = {
        "protocol": protocol_ps,
        "serialization": serialization_ps,
        "incast": incast_ps,
    }
    dominant = max(terms, key=lambda name: terms[name]) if any(terms.values()) else "none"
    return {
        "protocol_ps": protocol_ps,
        "serialization_ps": serialization_ps,
        "incast_ps": incast_ps,
        "dominant_mechanism": dominant,
        "raw_excess_ps": max(0, concurrent_service_ps - ideal_wire_ps),
    }


def intra_node_attribution(
    *,
    ideal_wire_ps: int,
    result: CompactNvlinkResult,
) -> dict[str, Any]:
    """Extract candidate TX, switch, and RX module terms."""

    tx_credits_ps = result.max_packet_tx_finish_ps - ideal_wire_ps
    switch_contention_ps = (
        result.max_packet_switch_finish_ps - result.max_packet_tx_finish_ps
    )
    rx_return_ps = result.completion_time_ps - result.max_packet_switch_finish_ps
    if min(tx_credits_ps, switch_contention_ps, rx_return_ps) < 0:
        raise AssertionError("candidate module attribution became negative")
    terms = {
        "TX credits and packetization": tx_credits_ps,
        "switch contention": switch_contention_ps,
        "RX return": rx_return_ps,
    }
    dominant = max(terms, key=lambda name: terms[name]) if any(terms.values()) else "none"
    return {
        "tx_credits_ps": tx_credits_ps,
        "switch_contention_ps": switch_contention_ps,
        "rx_return_ps": rx_return_ps,
        "dominant_module": dominant,
        "raw_excess_ps": result.completion_time_ps - ideal_wire_ps,
    }


def account_step(
    *,
    kernel_floor_ps: int,
    ideal_fabric_wire_ps: int,
    ideal_intra_node_wire_ps: int,
    simulated_fabric_ps: int,
    simulated_intra_node_ps: int,
) -> dict[str, int]:
    """Apply the frozen inter-then-intra telescoping accounting identity."""

    values = (
        kernel_floor_ps,
        ideal_fabric_wire_ps,
        ideal_intra_node_wire_ps,
        simulated_fabric_ps,
        simulated_intra_node_ps,
    )
    if min(values) < 0:
        raise ValueError("step services must be nonnegative")
    analytical = max(values[:3])
    after_inter = max(kernel_floor_ps, simulated_fabric_ps, ideal_intra_node_wire_ps)
    simulated = max(kernel_floor_ps, simulated_fabric_ps, simulated_intra_node_ps)
    inter = after_inter - analytical
    intra = simulated - after_inter
    residual = simulated - analytical - inter - intra
    return {
        "analytical_step_ps": analytical,
        "after_inter_node_ps": after_inter,
        "simulated_step_ps": simulated,
        "inter_node_attributed_ps": inter,
        "intra_node_attributed_ps": intra,
        "residual_ps": residual,
    }


def bottleneck_classification(
    *,
    kernel_floor_ps: int,
    simulated_fabric_ps: int,
    simulated_intra_node_ps: int,
) -> dict[str, Any]:
    """Name the strict binding owner or all co-critical owners."""

    services = {
        "roofline": kernel_floor_ps,
        "inter-node": simulated_fabric_ps,
        "intra-node": simulated_intra_node_ps,
    }
    maximum = max(services.values())
    owners = [name for name, value in services.items() if value == maximum]
    if len(owners) > 1:
        classification = "co-critical"
    elif owners[0] == "roofline":
        classification = "neither"
    else:
        classification = owners[0]
    return {"classification": classification, "critical_owners": owners}


def operating_point(*, batch_per_gpu: int, step_time_ps: int) -> dict[str, Any]:
    """Return exact coordinates for the versioned frontier axis contract."""

    if batch_per_gpu <= 0 or step_time_ps <= 0:
        raise ValueError("frontier coordinates must be positive")
    return {
        "x_tokens_per_second_per_request": fraction_record(
            PICOSECONDS_PER_SECOND,
            step_time_ps,
        ),
        "y_tokens_per_second_per_gpu": fraction_record(
            batch_per_gpu * PICOSECONDS_PER_SECOND,
            step_time_ps,
        ),
    }
