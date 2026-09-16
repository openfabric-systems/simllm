"""Run the frozen synthetic mechanism checks for Simple buffered peer reads.

This is deliberately not a hardware-calibration runner.  Every rate and cycle
cost comes from the expectations files, and the outputs establish causal and
accounting behavior only.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from simllm.backends.htsim_nvlink import (
    NvlinkAlignedOptions,
    NvlinkOperation,
    NvlinkPacketDirection,
    NvlinkSwitchConfig,
    NvlinkSwitchMode,
    NvlinkTransfer,
    load_nvlink_candidate_profile,
)
from simllm.backends.nccl_resources import NcclGpuProfile, NcclGpuResources
from simllm.backends.nccl_runtime import NcclChannelRuntime, NcclExecutionConfig
from simllm.backends.nvlink_runtime import NvlinkPhysicalBinding
from simllm.backends.peer_step import PeerPacketConfig
from simllm.backends.step_attribution import HtsimRequestMetricReducer
from simllm.backends.step_sink import HtsimStepSink, HtsimStepSinkConfig
from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
from simllm.compute.gpu_packet_port import GpuPeerPacketSession
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.placement import (
    FabricLink,
    FabricNodePlacement,
    FabricTopologyManifest,
    GpuFabricPlacement,
    PeerFabric,
    PeerPortPlacement,
    PeerRoute,
    PlacementManifest,
    RankPlacement,
)
from simllm.traffic.nccl_program import NcclRingProgram, balanced_channels


def _write_table(output: Path, name: str, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty study table {name}")
    with (output / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _physical_inputs(width: int, rate: int):
    """Construct the direct-mesh fixture frozen by the TRAF-93 expectations."""

    profile_path = ROOT / "examples/a100_nvlink_packet_v1/candidate-profile-pre-traf70.json"
    base = load_nvlink_candidate_profile(profile_path)
    profile = replace(
        base,
        tx=replace(
            base.tx,
            links_per_peer=1,
            per_link_rate_bytes_per_second=rate,
            endpoint_egress_rate_bytes_per_second=100_000_000_000,
            credits_per_destination=256,
        ),
        rx=replace(
            base.rx,
            ingress_rate_bytes_per_second=100_000_000_000,
            buffer_capacity_bytes=65536,
            credit_return_latency_ps=0,
        ),
        switch=NvlinkSwitchConfig(mode=NvlinkSwitchMode.PASS_THROUGH),
    )
    ports, links, routes = [], [], []
    for source in range(width):
        for destination in range(source + 1, width):
            left = f"p{source}-{destination}"
            right = f"p{destination}-{source}"
            link = f"l{source}-{destination}"
            ports.extend((
                PeerPortPlacement(left, gpu_rank=source),
                PeerPortPlacement(right, gpu_rank=destination),
            ))
            links.append(FabricLink(link, left, right, rate * 8, 1000))
            routes.extend((
                PeerRoute(source, destination, ((link,),)),
                PeerRoute(destination, source, ((link,),)),
            ))
    fabric = PeerFabric(
        "peer-domain",
        "node-0",
        tuple(ports),
        tuple(links),
        tuple(routes),
        switch_input_buffer_bytes=65536,
    )
    return profile, NvlinkPhysicalBinding(fabric)


def _session(width: int, rate: int) -> GpuPeerPacketSession:
    profile, binding = _physical_inputs(width, rate)
    return GpuPeerPacketSession(
        "simple-read-study", profile, binding, options=NvlinkAlignedOptions()
    )


def _gpu_profile(config: dict, sms: int, *, memory_rate: int | None = None) -> NcclGpuProfile:
    channel = json.loads(
        (HERE / config["gpu_inputs"]).read_text(encoding="utf-8")
    )["synthetic_gpu"]
    shared = json.loads(
        (HERE / config["shared_memory_inputs"]).read_text(encoding="utf-8")
    )
    values = {
        **channel,
        "shared_memory_bytes_per_second": shared["shared_memory_bytes_per_second"],
    }
    if memory_rate is not None:
        values["memory_bytes_per_second"] = memory_rate
    return NcclGpuProfile(available_sms=sms, **values)


def _read_timing(size: int, delay_ps: int, rate: int) -> dict:
    session = _session(2, rate)
    transfer = NvlinkTransfer(
        extent_id="read",
        source=1,
        destination=0,
        payload_bytes=size,
        operation=NvlinkOperation.PEER_READ,
        topology_endpoint_count=2,
    )
    observed = []

    def request_visible() -> None:
        observed.append(session.now_ps)
        session.schedule_callback(
            session.now_ps + delay_ps,
            lambda: session.release_read_response("read"),
        )

    session.admit_read("isolated", "read", transfer, request_visible)
    session.advance_until_visible(("read",))
    request = next(
        packet for packet in session.packets
        if packet.direction is NvlinkPacketDirection.REQUEST
    )
    responses = tuple(
        packet for packet in session.packets
        if packet.direction is NvlinkPacketDirection.RESPONSE
    )
    if observed != [request.visible_at_ps]:
        raise AssertionError("read request visibility callback did not fire exactly once")
    row = {
        "payload_bytes": size,
        "delay_ps": delay_ps,
        "request_visible_ps": request.visible_at_ps,
        "first_response_start_ps": min(packet.tx_started_at_ps for packet in responses),
        "final_visible_ps": max(packet.visible_at_ps for packet in responses),
        "request_wire_bytes": request.wire_bytes,
        "response_payload_bytes": sum(packet.payload_bytes for packet in responses),
    }
    session.drain()
    return row


def isolated_reads(config: dict, output: Path) -> int:
    rows = []
    for size in config["isolated_read_bytes"]:
        baseline = _read_timing(size, 0, config["link_rates_bytes_per_second"][0])
        for delay in config["read_service_delays_ps"]:
            row = _read_timing(size, delay, config["link_rates_bytes_per_second"][0])
            if row["request_visible_ps"] != baseline["request_visible_ps"]:
                raise AssertionError("external source service changed request transport")
            if row["first_response_start_ps"] - baseline["first_response_start_ps"] != delay:
                raise AssertionError("response start did not shift by the external service delay")
            if row["final_visible_ps"] - baseline["final_visible_ps"] != delay:
                raise AssertionError("response visibility did not shift by the external service delay")
            rows.append(row)
    _write_table(output, "isolated_reads.csv", rows)
    return len(rows)


def remote_memory(config: dict, output: Path) -> int:
    """Check owner-memory serialization without requiring an owner block."""

    rows = []
    for size, rate in itertools.product(
        config["isolated_read_bytes"], config["memory_rates_bytes_per_second"]
    ):
        session = _session(2, config["link_rates_bytes_per_second"][0])
        resources = NcclGpuResources(session, _gpu_profile(config, 1, memory_rate=rate))
        completed = []

        def issue(
            resources=resources,
            size=size,
            completed=completed,
            session=session,
        ) -> None:
            resources.service(
                "requester",
                "peer_buffer_read_source",
                size,
                lambda: completed.append(session.now_ps),
                memory=True,
                memory_rank=0,
            )

        resources.admit("requester", 1, 4, issue)
        session.advance_until(lambda completed=completed: bool(completed))
        oracle = (size * 10**12 + rate - 1) // rate
        if completed != [oracle]:
            raise AssertionError("remote memory service disagrees with its rate oracle")
        visit, = resources.visits
        if visit.rank != 0 or visit.block_id != "requester":
            raise AssertionError("remote memory service lost owner or requester attribution")
        rows.append({
            "payload_bytes": size,
            "memory_rate_bytes_per_second": rate,
            "service_ps": completed[0],
            "oracle_ps": oracle,
            "owner_rank": visit.rank,
            "requester_block": visit.block_id,
        })
    _write_table(output, "remote_memory.csv", rows)
    return len(rows)


def collectives(config: dict, output: Path) -> int:
    rows = []
    gpu_inputs = json.loads(
        (HERE / config["gpu_inputs"]).read_text(encoding="utf-8")
    )
    protocols = gpu_inputs["protocols"]
    for width, placement, channels, sms, rate, payload, protocol in itertools.product(
        config["widths"],
        config["placements"],
        config["channels"],
        config["available_sms"],
        config["link_rates_bytes_per_second"],
        config["payload_bytes"],
        protocols,
    ):
        session = _session(width, rate)
        runtime = NcclChannelRuntime(session, _gpu_profile(config, sms))
        warps = config["warps"] if protocol == "SIMPLE" else 4
        program = NcclRingProgram(
            payload,
            tuple(range(width)),
            protocol,
            balanced_channels(payload, channels),
            warps,
            connection_mode=placement,
        )
        result = runtime.run("collective", "sum", program)
        result.validate()
        packets = session.packets
        rows.append({
            "width": width,
            "placement": placement,
            "channels": channels,
            "available_sms": sms,
            "link_rate_bytes_per_second": rate,
            "payload_bytes": payload,
            "protocol": protocol,
            "duration_ps": result.duration_ps,
            "protocol_data_bytes": result.protocol_data_bytes,
            "read_requests": sum(
                packet.direction is NvlinkPacketDirection.REQUEST and not packet.payload_bytes
                for packet in packets
            ),
            "read_response_bytes": sum(
                packet.payload_bytes
                for packet in packets
                if packet.direction is NvlinkPacketDirection.RESPONSE
            ),
            "resource_visits": len(result.resource_visits),
        })
        session.drain()

    # LL and LL128 do not use Simple's placement-specific source branch.  Their
    # entire timing and work projection must therefore remain identical.
    comparable = {}
    for row in rows:
        key = tuple(
            row[name] for name in (
                "width", "channels", "available_sms",
                "link_rate_bytes_per_second", "payload_bytes", "protocol",
            )
        )
        if row["protocol"] != "SIMPLE":
            normalized = {key: value for key, value in row.items() if key != "placement"}
            previous = comparable.setdefault(key, normalized)
            if previous != normalized:
                raise AssertionError("LL/LL128 changed under read-capable placement")
    _write_table(output, "collectives.csv", rows)
    return len(rows)


class _DeclaredCompute(ComputeProvider):
    """Keep graph compute fixed so communication deltas remain exact."""

    def __init__(self, duration_ps: int) -> None:
        self.duration_ps = duration_ps

    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=self.duration_ps, bound="compute")


def graph_metrics(config: dict, output: Path) -> int:
    """Project both Simple placements through the original step/metric path."""

    rows = []
    fixed_compute = config["fixed_graph_compute_ps"]
    for width, placement, sms, rate in itertools.product(
        config["widths"],
        config["placements"],
        config["available_sms"],
        config["link_rates_bytes_per_second"],
    ):
        profile, binding = _physical_inputs(width, rate)
        ranks = tuple(range(width))
        placement_manifest = PlacementManifest(
            ranks=[RankPlacement(rank, "node-0", rank) for rank in ranks]
        )
        topology = FabricTopologyManifest(
            nodes=[FabricNodePlacement(
                "node-0",
                "serving",
                tuple(
                    GpuFabricPlacement(
                        rank, f"gpu{rank}", "node-0", f"pcie{rank}", f"nic{rank}"
                    )
                    for rank in ranks
                ),
                (),
            )],
            peer_fabrics=(binding.fabric,),
        )
        selected = NcclExecutionConfig(
            _gpu_profile(config, sms),
            protocol="SIMPLE",
            channels=2,
            warps=config["warps"],
            connection_mode=placement,
        )
        settings = HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=ranks,
            dims=ModelDims(
                num_layers=1,
                hidden_size=256,
                intermediate_size=512,
                num_heads=4,
                num_kv_heads=4,
                head_size=64,
                vocab_size=256,
                dtype_bytes=4,
            ),
            workdir=output / "graphs" / f"{width}-{placement}-{sms}-{rate}",
            provider=_DeclaredCompute(fixed_compute),
            placement_manifest=placement_manifest,
            peer_packet=PeerPacketConfig(
                topology,
                (("peer-domain", profile),),
                nccl=selected,
            ),
        )
        sink = HtsimStepSink(
            settings,
            request_metric_reducer=HtsimRequestMetricReducer({"request": 0}),
        )
        at_ps = 0
        for step in range(2):
            record = StepRecord(
                step_index=step,
                virtual_time_ps=at_ps,
                num_sampled=1,
                scheduled=[ScheduledRequest(
                    "request",
                    RequestPhase.PREFILL if step == 0 else RequestPhase.DECODE,
                    1,
                    context_length=step + 1,
                )],
            )
            result = sink(record)
            evidence = sink.peer_evidence[-1]
            evidence.validate_result(result)
            phases = [
                artifact.local_phase
                for artifact in evidence.artifacts
                if artifact.local_phase is not None
            ]
            communication = sum(phase.service_ps for phase in phases)
            metric = result.request_metrics[0]
            token_time = metric.ttft_ps if step == 0 else metric.tpot_ps
            if token_time - communication != fixed_compute:
                raise AssertionError("token metric lost the frozen communication delta")
            rows.append({
                "width": width,
                "placement": placement,
                "available_sms": sms,
                "link_rate_bytes_per_second": rate,
                "phase": "prefill" if step == 0 else "decode",
                "communication_ps": communication,
                "token_metric_ps": token_time,
                "fixed_compute_ps": token_time - communication,
            })
            at_ps = result.completed_at_ps
        sink.close_peer_packets()
    _write_table(output, "graph_metrics.csv", rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((HERE / "expectations.json").read_text(encoding="utf-8"))
    summary = {
        "schema": "simllm-nccl-simple-read-study-v1",
        "expectations": "expectations.json",
        "evidence_class": config["evidence_class"],
        "fatal_guards": "valid",
        "isolated_read_rows": isolated_reads(config, args.output),
        "remote_memory_rows": remote_memory(config, args.output),
        "collective_rows": collectives(config, args.output),
        "graph_metric_rows": graph_metrics(config, args.output),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
