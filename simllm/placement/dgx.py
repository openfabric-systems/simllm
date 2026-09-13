"""Public DGX/HGX GPU wiring with explicitly declared timing and route choices.

NVIDIA's Fabric Manager User Guide (the NVSwitch systems section) gives the
attachment counts of each eight-GPU baseboard: six NVSwitch chips with two
NVLink lanes from every GPU to each on HGX A100, four chips with 4, 5, 5 and 4
lanes on HGX H100, and two chips with nine lanes each on HGX B200. A captured
board confirms only a GPU's total lane count and its all-pair reachability,
never how the lanes split across chips, so every bundle below, the B200 split
included, is declared from that public description rather than captured.

Port IDs here are stable logical attachment identities, not vendor register
indices or nvidia-smi device ordering. Rates are directional payload rates:
an NVLink 3 or 4 lane carries 200 Gbit/s, an NVLink 5 lane 400 Gbit/s (the
published 1.8 TB/s bidirectional per GPU over eighteen lanes). Signalling rates
reported by nvidia-smi are higher and are recorded by capture studies, not used.
"""

from __future__ import annotations

from itertools import permutations

from .manifest import FabricLink
from .peer_topology import PeerFabric, PeerPortPlacement, PeerRoute

DGX_NVLINK_BUNDLES = {"a100": (2, 2, 2, 2, 2, 2), "h100": (4, 5, 5, 4), "b200": (9, 9)}
#: Declared per-direction payload rate of one lane, by generation.
DGX_NVLINK_LINK_RATE_BPS = {
    "a100": 200_000_000_000,
    "h100": 200_000_000_000,
    "b200": 400_000_000_000,
}
#: PCI device ids and NCCL ``sm`` value (compute capability as one number) that
#: each generation's GPUs report: A100 40 and 80 GB; H100, H100 NVL and H200;
#: B200.
DGX_GPU_SILICON = {
    "a100": (("0x20b0", "0x20b2"), 80),
    "h100": (("0x2330", "0x2335", "0x2339"), 90),
    "b200": (("0x2901",), 100),
}
DGX_NVLINK_SOURCE = "https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/"


def dgx_peer_fabric(
    generation: str, *, node_id: str, ranks: tuple[int, ...],
    propagation_delay_ps: int, switch_input_buffer_bytes: int,
    domain_id: str | None = None,
) -> PeerFabric:
    """Build one eight-GPU board through the existing peer manifest contract.

    Timing and buffer capacity are required, declared inputs. No hardware
    calibration is inferred. Lane-matched routes within each
    switch cover every attachment; this is a declared striping policy.
    Endpoint and output arbitration still conserve each link's capacity.
    """
    if generation not in DGX_NVLINK_BUNDLES:
        raise ValueError(f"DGX generation must be one of {', '.join(DGX_NVLINK_BUNDLES)}")
    if not isinstance(ranks, (list, tuple)) or len(ranks) != 8 or any(
        type(rank) is not int or rank < 0 for rank in ranks
    ) or len(set(ranks)) != 8:
        raise ValueError("a DGX peer preset requires eight distinct physical GPU ranks")
    for name, value, minimum in (("propagation", propagation_delay_ps, 0),
                                 ("input buffer", switch_input_buffer_bytes, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"DGX {name} must be an integer >= {minimum}")
    domain_id = domain_id or f"{node_id}:hgx-{generation}-8"
    ports, links, attached = [], [], {}
    for switch, width in enumerate(DGX_NVLINK_BUNDLES[generation], 1):
        switch_id = f"{domain_id}:switch-{switch}"
        for slot, rank in enumerate(ranks):
            attached[rank, switch] = []
            for lane in range(width):
                gpu_port = f"{domain_id}:gpu-{slot}:switch-{switch}:lane-{lane}"
                switch_port = f"{switch_id}:gpu-{slot}:lane-{lane}"
                link_id = f"{domain_id}:link-{slot}-{switch}-{lane}"
                ports.extend((PeerPortPlacement(gpu_port, gpu_rank=rank),
                              PeerPortPlacement(switch_port, switch_id=switch_id)))
                links.append(FabricLink(link_id, gpu_port, switch_port,
                                        DGX_NVLINK_LINK_RATE_BPS[generation],
                                        propagation_delay_ps))
                attached[rank, switch].append(link_id)
    routes = tuple(PeerRoute(source, destination, tuple(
        (first, last)
        for switch in range(1, len(DGX_NVLINK_BUNDLES[generation]) + 1)
        for first, last in zip(attached[source, switch], attached[destination, switch])
    )) for source, destination in permutations(ranks, 2))
    return PeerFabric(domain_id, node_id, tuple(ports), tuple(links), routes,
                      switch_input_buffer_bytes=switch_input_buffer_bytes)
