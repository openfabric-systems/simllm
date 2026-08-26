"""Concrete placement for prefill and decode engine pools.

Each serving node is one tensor-parallel engine. The builder is deliberately
specific to the disaggregated deployment, while its outputs remain the
repository-standard placement and fabric manifests.
"""

from __future__ import annotations

from dataclasses import dataclass

from simllm.core import ServingPoolRole
from simllm.placement.manifest import (
    FabricLink,
    FabricNodePlacement,
    FabricSwitchPlacement,
    FabricSwitchPort,
    FabricTopologyManifest,
    GpuFabricPlacement,
    GroupMembership,
    NicFabricPlacement,
    PlacementManifest,
    RankPlacement,
)

DECLARED_CLOS_TOPOLOGY_NAME = "simllm-disaggregated-448-clos-declared-v1"
DECLARED_CLOS_EVIDENCE_CLASS = "declared"
DECLARED_CLOS_ENDPOINTS_PER_LEAF = 8
DECLARED_CLOS_SPINE_SWITCHES = 8
DECLARED_CLOS_LINK_RATE_BPS = 400_000_000_000
DECLARED_CLOS_LINK_PROPAGATION_DELAY_PS = 1_000_000
DECLARED_CLOS_SWITCH_LATENCY_PS = 0


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class DisaggregatedDeploymentManifests:
    """The joined logical placement and concrete fabric inventory."""

    placement: PlacementManifest
    fabric: FabricTopologyManifest

    def validate(self) -> None:
        self.fabric.validate()
        ranks = self.placement.ranks
        fabric_gpus = tuple(gpu for node in self.fabric.nodes for gpu in node.gpus)
        fabric_nics = tuple(nic for node in self.fabric.nodes for nic in node.nics)
        if len(ranks) != len(fabric_gpus) or len(ranks) != len(fabric_nics):
            raise ValueError("placement, GPU, and NIC cardinalities must agree")
        rank_ids = [rank.global_rank for rank in ranks]
        if rank_ids != list(range(len(ranks))):
            raise ValueError("disaggregated placement ranks must be dense and ordered")
        gpu_ranks = [gpu.global_rank for gpu in fabric_gpus]
        if sorted(gpu_ranks) != rank_ids:
            raise ValueError("fabric GPU ranks must match the placement exactly")
        gpu_ids = [gpu.gpu_id for gpu in fabric_gpus]
        nic_ids = [nic.nic_id for nic in fabric_nics]
        if len(gpu_ids) != len(set(gpu_ids)):
            raise ValueError("fabric GPU identities must be unique")
        if len(nic_ids) != len(set(nic_ids)):
            raise ValueError("fabric NIC identities must be unique")
        nic_by_id = {nic.nic_id: nic for nic in fabric_nics}
        fabric_node_by_id = {node.node_id: node for node in self.fabric.nodes}
        if len(fabric_node_by_id) != len(self.fabric.nodes):
            raise ValueError("fabric node identities must be unique")
        for rank in ranks:
            gpu = self.fabric.by_rank(rank.global_rank)
            if gpu.node_id != rank.hostname or gpu.gpu_id != rank.gpu_uuid:
                raise ValueError("fabric GPU placement disagrees with logical placement")
            nic = nic_by_id.get(gpu.nic_id)
            if (
                nic is None
                or nic.affine_gpu_rank != rank.global_rank
                or nic.node_id != rank.hostname
            ):
                raise ValueError("GPU-to-NIC affinity is incomplete or inconsistent")
            if not nic.fabric_location.strip():
                raise ValueError("every NIC must have a physical fabric location")
            node = fabric_node_by_id.get(rank.hostname)
            if node is None or node.pool_role != rank.pool_role:
                raise ValueError("fabric node role disagrees with logical placement")


def _leaf_switch_id(leaf_index: int) -> str:
    return f"leaf-{leaf_index:03d}"


def _spine_switch_id(spine_index: int) -> str:
    return f"spine-{spine_index:03d}"


def _endpoint_port_id(global_rank: int) -> str:
    leaf_index = global_rank // DECLARED_CLOS_ENDPOINTS_PER_LEAF
    local_endpoint = global_rank % DECLARED_CLOS_ENDPOINTS_PER_LEAF
    return f"{_leaf_switch_id(leaf_index)}/endpoint-{local_endpoint:02d}"


def _endpoint_link_id(global_rank: int) -> str:
    return f"endpoint-link-{global_rank:04d}"


def _leaf_spine_port_id(leaf_index: int, spine_index: int) -> str:
    return f"{_leaf_switch_id(leaf_index)}/spine-{spine_index:03d}"


def _spine_leaf_port_id(spine_index: int, leaf_index: int) -> str:
    return f"{_spine_switch_id(spine_index)}/leaf-{leaf_index:03d}"


def _leaf_spine_link_id(leaf_index: int, spine_index: int) -> str:
    return f"fabric-link-leaf-{leaf_index:03d}-spine-{spine_index:03d}"


def _declared_clos_graph(
    rank_count: int,
) -> tuple[tuple[FabricSwitchPlacement, ...], tuple[FabricLink, ...]]:
    leaf_count = (
        rank_count + DECLARED_CLOS_ENDPOINTS_PER_LEAF - 1
    ) // DECLARED_CLOS_ENDPOINTS_PER_LEAF
    switches: list[FabricSwitchPlacement] = []
    links: list[FabricLink] = []

    for leaf_index in range(leaf_count):
        switch_id = _leaf_switch_id(leaf_index)
        first_rank = leaf_index * DECLARED_CLOS_ENDPOINTS_PER_LEAF
        endpoint_ranks = range(
            first_rank,
            min(first_rank + DECLARED_CLOS_ENDPOINTS_PER_LEAF, rank_count),
        )
        ports = [
            FabricSwitchPort(
                port_id=_endpoint_port_id(global_rank),
                switch_id=switch_id,
            )
            for global_rank in endpoint_ranks
        ]
        ports.extend(
            FabricSwitchPort(
                port_id=_leaf_spine_port_id(leaf_index, spine_index),
                switch_id=switch_id,
            )
            for spine_index in range(DECLARED_CLOS_SPINE_SWITCHES)
        )
        switches.append(
            FabricSwitchPlacement(
                switch_id=switch_id,
                tier=0,
                ports=tuple(ports),
            )
        )

    for spine_index in range(DECLARED_CLOS_SPINE_SWITCHES):
        switch_id = _spine_switch_id(spine_index)
        switches.append(
            FabricSwitchPlacement(
                switch_id=switch_id,
                tier=1,
                ports=tuple(
                    FabricSwitchPort(
                        port_id=_spine_leaf_port_id(spine_index, leaf_index),
                        switch_id=switch_id,
                    )
                    for leaf_index in range(leaf_count)
                ),
            )
        )

    for global_rank in range(rank_count):
        links.append(
            FabricLink(
                link_id=_endpoint_link_id(global_rank),
                endpoint_a=f"sim-nic-{global_rank:04d}",
                endpoint_b=_endpoint_port_id(global_rank),
                link_rate_bps=DECLARED_CLOS_LINK_RATE_BPS,
                propagation_delay_ps=(
                    DECLARED_CLOS_LINK_PROPAGATION_DELAY_PS
                ),
            )
        )
    for leaf_index in range(leaf_count):
        for spine_index in range(DECLARED_CLOS_SPINE_SWITCHES):
            links.append(
                FabricLink(
                    link_id=_leaf_spine_link_id(leaf_index, spine_index),
                    endpoint_a=_leaf_spine_port_id(leaf_index, spine_index),
                    endpoint_b=_spine_leaf_port_id(spine_index, leaf_index),
                    link_rate_bps=DECLARED_CLOS_LINK_RATE_BPS,
                    propagation_delay_ps=(
                        DECLARED_CLOS_LINK_PROPAGATION_DELAY_PS
                    ),
                )
            )
    return tuple(switches), tuple(links)


def _role_node_offsets(prefill_nodes: int, decode_nodes: int) -> tuple[tuple, ...]:
    return (
        (ServingPoolRole.PREFILL, prefill_nodes, 0),
        (ServingPoolRole.DECODE, decode_nodes, prefill_nodes),
    )


def disaggregated_manifests(
    *,
    prefill_nodes: int,
    decode_nodes: int,
    gpus_per_node: int = 8,
    framework: str | None = None,
    framework_version: str | None = None,
    render_physical_topology: bool = True,
) -> DisaggregatedDeploymentManifests:
    """Build role-aware manifests for a concrete P/D deployment.

    Prefill nodes precede decode nodes in global rank order. Data-parallel
    groups never cross the role boundary. Every GPU has one affine NIC and
    both are pinned to the declared two-tier Clos when physical rendering is
    enabled. The explicit disabled path retains the PLACE-4 compatibility
    inventory and does not alter the placement record.
    """

    prefill_nodes = _positive_int("prefill_nodes", prefill_nodes)
    decode_nodes = _positive_int("decode_nodes", decode_nodes)
    gpus_per_node = _positive_int("gpus_per_node", gpus_per_node)
    if type(render_physical_topology) is not bool:
        raise TypeError("render_physical_topology must be a boolean")
    if (
        render_physical_topology
        and gpus_per_node != DECLARED_CLOS_ENDPOINTS_PER_LEAF
    ):
        raise ValueError(
            "fixed physical topology requires exactly eight GPUs and NICs per node"
        )
    ranks: list[RankPlacement] = []
    fabric_nodes: list[FabricNodePlacement] = []

    for role, role_nodes, global_node_offset in _role_node_offsets(
        prefill_nodes, decode_nodes
    ):
        role_rank_start = global_node_offset * gpus_per_node
        role_rank_count = role_nodes * gpus_per_node
        pool_ranks = list(range(role_rank_start, role_rank_start + role_rank_count))
        for role_node_index in range(role_nodes):
            global_node_index = global_node_offset + role_node_index
            node_id = f"{role.value}-node-{role_node_index}"
            node_rank_start = global_node_index * gpus_per_node
            tp_ranks = list(
                range(node_rank_start, node_rank_start + gpus_per_node)
            )
            gpus: list[GpuFabricPlacement] = []
            nics: list[NicFabricPlacement] = []
            for local_rank in range(gpus_per_node):
                global_rank = node_rank_start + local_rank
                dp_ranks = [
                    role_rank_start + node_index * gpus_per_node + local_rank
                    for node_index in range(role_nodes)
                ]
                gpu_id = f"sim-gpu-{global_rank:04d}"
                nic_id = f"sim-nic-{global_rank:04d}"
                ranks.append(
                    RankPlacement(
                        global_rank=global_rank,
                        hostname=node_id,
                        local_rank=local_rank,
                        gpu_uuid=gpu_id,
                        pci_bus_id=(
                            f"0000:{global_node_index:02x}:{local_rank:02x}.0"
                        ),
                        groups={
                            "tp": GroupMembership(local_rank, tp_ranks),
                            "pp": GroupMembership(0, [global_rank]),
                            "dp": GroupMembership(role_node_index, dp_ranks),
                            "pool": GroupMembership(
                                role_node_index * gpus_per_node + local_rank,
                                pool_ranks,
                            ),
                        },
                        pool_role=role.value,
                    )
                )
                gpus.append(
                    GpuFabricPlacement(
                        global_rank=global_rank,
                        gpu_id=gpu_id,
                        node_id=node_id,
                        pcie_location=f"{node_id}/pcie-{local_rank}",
                        nic_id=nic_id,
                    )
                )
                nics.append(
                    NicFabricPlacement(
                        nic_id=nic_id,
                        node_id=node_id,
                        fabric_location=(
                            _endpoint_port_id(global_rank)
                            if render_physical_topology
                            else (
                                f"leaf-{global_node_index // 16}/"
                                f"port-{global_node_index % 16:02d}/"
                                f"nic-{local_rank}"
                            )
                        ),
                        affine_gpu_rank=global_rank,
                        switch_id=(
                            _leaf_switch_id(
                                global_rank // DECLARED_CLOS_ENDPOINTS_PER_LEAF
                            )
                            if render_physical_topology
                            else None
                        ),
                        switch_port_id=(
                            _endpoint_port_id(global_rank)
                            if render_physical_topology
                            else None
                        ),
                        link_id=(
                            _endpoint_link_id(global_rank)
                            if render_physical_topology
                            else None
                        ),
                    )
                )
            fabric_nodes.append(
                FabricNodePlacement(
                    node_id=node_id,
                    pool_role=role.value,
                    gpus=tuple(gpus),
                    nics=tuple(nics),
                )
            )

    placement = PlacementManifest(
        ranks=sorted(ranks, key=lambda rank: rank.global_rank),
        source="declared",
        framework=framework,
        framework_version=framework_version,
    )
    if render_physical_topology:
        switches, links = _declared_clos_graph(len(placement.ranks))
        fabric = FabricTopologyManifest(
            nodes=fabric_nodes,
            physical_rendering_enabled=True,
            topology_name=DECLARED_CLOS_TOPOLOGY_NAME,
            evidence_class=DECLARED_CLOS_EVIDENCE_CLASS,
            switch_latency_ps=DECLARED_CLOS_SWITCH_LATENCY_PS,
            switches=switches,
            links=links,
        )
    else:
        fabric = FabricTopologyManifest(nodes=fabric_nodes)
    result = DisaggregatedDeploymentManifests(placement=placement, fabric=fabric)
    result.validate()
    return result


__all__ = [
    "DECLARED_CLOS_ENDPOINTS_PER_LEAF",
    "DECLARED_CLOS_EVIDENCE_CLASS",
    "DECLARED_CLOS_LINK_PROPAGATION_DELAY_PS",
    "DECLARED_CLOS_LINK_RATE_BPS",
    "DECLARED_CLOS_SPINE_SWITCHES",
    "DECLARED_CLOS_SWITCH_LATENCY_PS",
    "DECLARED_CLOS_TOPOLOGY_NAME",
    "DisaggregatedDeploymentManifests",
    "disaggregated_manifests",
]
