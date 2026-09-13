"""Declared shared-NIC fabric inventory for ``unique-nic`` GOAL-rank mapping.

Several GPUs behind one network interface controller (NIC) share one fabric
endpoint. The builder gives every node of a declared placement
``gpus_per_node // gpus_per_nic`` NICs, named ``"<host>:nic-<j>"``, and pins
the GPU at ``local_rank`` ``l`` to NIC ``l // gpus_per_nic``. Fabric order,
i.e. node order by lowest global rank and then NIC index, is the GOAL rank
order :class:`simllm.placement.RankMapper` assigns under ``unique-nic``.

No switch graph is rendered, so the inventory runs on the null-network
profiles. One NIC per GPU reproduces the identity mapping, which keeps every
GOAL artifact byte identical to ``gpu-rank``.
"""

from __future__ import annotations

from simllm.placement.manifest import (
    FabricNodePlacement,
    FabricTopologyManifest,
    GpuFabricPlacement,
    NicFabricPlacement,
    PlacementManifest,
    RankPlacement,
)
from simllm.placement.mapper import RankMapper


def declared_shared_nic_fabric(
    placement: PlacementManifest,
    *,
    gpus_per_nic: int,
) -> FabricTopologyManifest:
    """Attach ``gpus_per_nic`` GPUs of every declared node to each of its NICs.

    Every node must declare dense local ranks from zero, and ``gpus_per_nic``
    must divide each node's width. A NIC's ``affine_gpu_rank`` names the
    lowest global rank it serves; the GPU records carry the full affinity.
    The returned manifest declares ``goal_rank_mapping="unique-nic"`` and has
    already been checked against ``placement`` by the mapper.
    """

    if not isinstance(placement, PlacementManifest):
        raise TypeError("placement must be a PlacementManifest")
    if placement.source != "declared":
        raise ValueError("a declared shared-NIC fabric requires a declared placement")
    if isinstance(gpus_per_nic, bool) or type(gpus_per_nic) is not int:
        raise TypeError("gpus_per_nic must be an integer")
    if gpus_per_nic < 1:
        raise ValueError("gpus_per_nic must be positive")

    ranks_by_host: dict[str, list[RankPlacement]] = {}
    for rank in sorted(placement.ranks, key=lambda item: item.global_rank):
        ranks_by_host.setdefault(rank.hostname, []).append(rank)
    if not ranks_by_host:
        raise ValueError("placement must contain at least one rank")

    nodes: list[FabricNodePlacement] = []
    for host, host_ranks in ranks_by_host.items():
        ranks = sorted(host_ranks, key=lambda item: item.local_rank)
        width = len(ranks)
        if [rank.local_rank for rank in ranks] != list(range(width)):
            raise ValueError(f"node {host!r} must declare dense local ranks from zero")
        if width % gpus_per_nic:
            raise ValueError(
                f"gpus_per_nic={gpus_per_nic} does not divide the width {width} "
                f"of node {host!r}"
            )
        roles = {rank.pool_role for rank in ranks}
        if len(roles) != 1:
            raise ValueError(f"node {host!r} mixes serving pool roles")
        role = roles.pop()
        nic_ids = tuple(f"{host}:nic-{index}" for index in range(width // gpus_per_nic))
        gpus = tuple(
            GpuFabricPlacement(
                global_rank=rank.global_rank,
                gpu_id=rank.gpu_uuid or f"sim-gpu-{rank.global_rank:04d}",
                node_id=host,
                pcie_location=f"{host}/pcie-{rank.local_rank}",
                nic_id=nic_ids[rank.local_rank // gpus_per_nic],
            )
            for rank in ranks
        )
        nics = tuple(
            NicFabricPlacement(
                nic_id=nic_id,
                node_id=host,
                fabric_location=f"{host}/nic-{index}",
                affine_gpu_rank=ranks[index * gpus_per_nic].global_rank,
            )
            for index, nic_id in enumerate(nic_ids)
        )
        nodes.append(
            FabricNodePlacement(
                node_id=host,
                pool_role=role if role is not None else "declared",
                gpus=gpus,
                nics=nics,
            )
        )

    fabric = FabricTopologyManifest(nodes=nodes, goal_rank_mapping="unique-nic")
    fabric.validate()
    RankMapper(placement, mode="unique-nic", fabric=fabric)
    return fabric


__all__ = ["declared_shared_nic_fabric"]
