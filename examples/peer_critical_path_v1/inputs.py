"""Frozen synthetic packet and serving inputs shared by both source trees."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from examples.local_peer_packet_runtime_v1.run_study import (
    live_config,
    physical_domain,
    profile_for,
)
from simllm.backends.htsim_nvlink import NvlinkAlignedOptions, NvlinkTransfer
from simllm.backends.nvlink_runtime import NvlinkPhysicalBinding
from simllm.backends.peer_step import PeerPacketConfig
from simllm.core import RequestPhase, ScheduledRequest, StepRecord


def inputs(topology, rate, donors, payload, workdir, *, reported, library=None):
    switched = topology == "switched"
    domain = physical_domain(donors + 1, switched, rate=rate)
    profile = profile_for(switched, rate=rate, feed=rate, rx_rate=rate)
    if switched:
        profile = replace(profile, switch=replace(profile.switch, service_rate_bytes_per_second=rate))
    profile = replace(
        profile, profile_id="declared-peer-critical-path-v1",
        freeze_sha256=sha256(Path(__file__).with_name("expectations.md").read_bytes()).hexdigest(),
    )
    config = live_config(workdir, donors, rate, switched, enabled=True)
    hidden = payload // 2
    config = replace(
        config, dims=replace(
            config.dims, hidden_size=hidden, intermediate_size=hidden,
            moe_intermediate_size=hidden, num_heads=hidden // 64,
            num_kv_heads=min(4, hidden // 64),
        ),
        peer_packet=PeerPacketConfig(
            replace(config.peer_packet.fabric, peer_fabrics=(domain,)),
            ((domain.domain_id, profile),), native_switch_library=library,
        ),
        emit_packet_breakdown=reported,
    )
    transfers = tuple(NvlinkTransfer(
        extent_id=f"donor-{source}", source=source, destination=donors,
        payload_bytes=payload, topology_endpoint_count=donors + 1,
    ) for source in range(donors))
    return config, profile, NvlinkPhysicalBinding(domain), NvlinkAlignedOptions(), transfers


def record(index, release):
    return StepRecord(
        step_index=index, virtual_time_ps=release, num_sampled=1,
        scheduled=[ScheduledRequest(
            "peer-star", RequestPhase.PREFILL if index == 0 else RequestPhase.DECODE,
            1, context_length=index + 1,
        )],
    )


def input_record(config, profile, binding, options, transfers, *, arm,
                 topology, rate, donors, payload):
    return {
        "evidence_kind": "declared_synthetic", "arm": arm,
        "topology": topology, "rate": rate, "donors": donors, "payload": payload,
        "profile": profile, "binding": binding, "options": options,
        "transfers": transfers, "dims": config.dims,
        "placement": config.placement_manifest,
        "fabric": config.peer_packet.fabric.to_dict(),
    }
