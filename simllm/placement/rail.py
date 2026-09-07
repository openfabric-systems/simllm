"""Fixed 64-endpoint rail and node-local variants of the declared Clos.

Both variants use the existing v1 physical graph. Only NIC attachments change;
link rates, propagation, switch latency and spine wiring retain their declared
values. PLACE-1 owns general inventory and topology discovery.
"""

from __future__ import annotations

from dataclasses import replace

from simllm.placement.declared import declared_manifest
from simllm.placement.disaggregated import (
    DECLARED_CLOS_EVIDENCE_CLASS,
    DECLARED_CLOS_SWITCH_LATENCY_PS,
    _declared_clos_graph,
    _endpoint_link_id,
    _endpoint_port_id,
    _leaf_switch_id,
)
from simllm.placement.manifest import (
    FabricNodePlacement,
    FabricTopologyManifest,
    GpuFabricPlacement,
    NicFabricPlacement,
    PlacementManifest,
)

RAIL_FABRIC_VARIANTS = ("rail", "node-local")


def declared_pipeline_placement(pipeline_width: int = 1) -> PlacementManifest:
    """Place PP peers on consecutive nodes, on the same NIC index.

    The inventory always contains eight nodes of eight GPUs. Tensor groups
    fill one node, PP groups stride across nodes, and remaining node groups
    are data-parallel replicas. A study may select one rank per stage from
    those groups when measuring an isolated PP chain.
    """
    if type(pipeline_width) is not int or pipeline_width not in (1, 2, 4, 8):
        raise ValueError("pipeline_width must be one of 1, 2, 4, 8")
    return declared_manifest(tp=8, pp=pipeline_width, dp=8 // pipeline_width, nodes=8)


def declared_rail_fabric(
    placement: PlacementManifest,
    *,
    variant: str,
) -> FabricTopologyManifest:
    """Attach exactly eight nodes of eight affine NICs to the fixed Clos.

    Node order is defined by rank blocks in the declared placement. NIC index
    is local_rank, which must match the position within the eight-rank block.
    Inputs are never mutated. The variant is mandatory so existing builders
    and their default artifacts keep their original semantics and bytes.
    """
    if not isinstance(placement, PlacementManifest):
        raise TypeError("placement must be a PlacementManifest")
    if variant not in RAIL_FABRIC_VARIANTS:
        raise ValueError(f"variant must be one of {RAIL_FABRIC_VARIANTS}")
    if placement.source != "declared":
        raise ValueError("fixed rail fabrics require a declared placement")
    if [rank.global_rank for rank in placement.ranks] != list(range(64)):
        raise ValueError("fixed rail fabric requires 64 dense, ordered ranks")
    hosts = []
    for node in range(8):
        ranks = placement.ranks[8 * node:8 * (node + 1)]
        host = ranks[0].hostname
        if not host or any(rank.hostname != host for rank in ranks):
            raise ValueError("each eight-rank block must occupy exactly one node")
        if [rank.local_rank for rank in ranks] != list(range(8)):
            raise ValueError("each node must declare local NIC indices zero through seven")
        hosts.append(host)
    if len(set(hosts)) != 8:
        raise ValueError("the fabric requires eight distinct physical nodes")

    switches, original_links = _declared_clos_graph(64)
    endpoint_slots = {
        rank.global_rank: (rank.local_rank * 8 + rank.global_rank // 8)
        if variant == "rail" else rank.global_rank
        for rank in placement.ranks
    }
    rank_by_link = {_endpoint_link_id(rank): rank for rank in range(64)}
    links = tuple(
        replace(link, endpoint_b=_endpoint_port_id(endpoint_slots[rank]))
        if (rank := rank_by_link.get(link.link_id)) is not None else link
        for link in original_links
    )
    nodes = []
    for node, host in enumerate(hosts):
        gpus = []
        nics = []
        for rank in placement.ranks[8 * node:8 * (node + 1)]:
            global_rank = rank.global_rank
            slot = endpoint_slots[global_rank]
            nic_id = f"sim-nic-{global_rank:04d}"
            gpus.append(GpuFabricPlacement(
                global_rank=global_rank,
                gpu_id=rank.gpu_uuid or f"sim-gpu-{global_rank:04d}",
                node_id=host,
                pcie_location=f"{host}/pcie-{rank.local_rank}",
                nic_id=nic_id,
            ))
            nics.append(NicFabricPlacement(
                nic_id=nic_id,
                node_id=host,
                fabric_location=_endpoint_port_id(slot),
                affine_gpu_rank=global_rank,
                switch_id=_leaf_switch_id(slot // 8),
                switch_port_id=_endpoint_port_id(slot),
                link_id=_endpoint_link_id(global_rank),
            ))
        nodes.append(FabricNodePlacement(host, "pipeline", tuple(gpus), tuple(nics)))
    fabric = FabricTopologyManifest(
        nodes=nodes,
        physical_rendering_enabled=True,
        topology_name=f"simllm-64-{variant}-clos-declared-v1",
        evidence_class=DECLARED_CLOS_EVIDENCE_CLASS,
        switch_latency_ps=DECLARED_CLOS_SWITCH_LATENCY_PS,
        switches=switches,
        links=links,
    )
    fabric.validate()
    return fabric
