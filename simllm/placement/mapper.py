"""The mapper: resolve logical ranks to physical endpoints and GOAL ranks.

Every communication event carries global ranks; the network backend wants
endpoints. Resolution is a join across the two manifests::

    endpoint = placement.by_rank(global_rank)        # node, GPU
    nic      = select_nic(endpoint, fabric)          # GPU→NIC affinity

GOAL rank assignment mirrors the htsim RNIC drivers'
``-goal_rank_mapping`` option:

- ``gpu-rank``: one GOAL rank per global rank (GPU). Intra-node traffic is
  visible to the simulator as ranks sharing a node.
- ``unique-nic``: one GOAL rank per (node, NIC); multiple GPUs behind one NIC
  share a GOAL rank and intra-node transfers stay off the fabric. Requires
  the fabric manifest (M4).
"""

from __future__ import annotations

from simllm.placement.manifest import (
    PLACEMENT_SCHEMA,
    FabricTopologyManifest,
    PlacementManifest,
    RankPlacement,
)

GOAL_RANK_MAPPINGS = ("gpu-rank", "unique-nic")


class RankMapper:
    #: fabric-backed state; a mapper built without a fabric keeps these class
    #: defaults, so its instance state is exactly the placement-only state
    fabric: FabricTopologyManifest | None = None
    _nic_by_rank: dict[int, str] | None = None
    _goal_rank_by_rank: dict[int, int] | None = None
    _nic_count = 0

    def __init__(
        self,
        placement: PlacementManifest,
        mode: str = "gpu-rank",
        fabric: FabricTopologyManifest | None = None,
    ):
        if not isinstance(placement, PlacementManifest):
            raise TypeError("placement must be a PlacementManifest")
        if placement.schema != PLACEMENT_SCHEMA:
            raise ValueError(f"unsupported placement schema: {placement.schema!r}")
        if mode not in GOAL_RANK_MAPPINGS:
            raise ValueError(f"mode must be one of {GOAL_RANK_MAPPINGS}")
        if fabric is not None and not isinstance(fabric, FabricTopologyManifest):
            raise TypeError("fabric must be a FabricTopologyManifest or None")
        if mode == "unique-nic":
            if fabric is None:
                raise ValueError("unique-nic mapping requires a fabric topology manifest")
            if fabric.goal_rank_mapping != "unique-nic":
                raise ValueError(
                    "unique-nic mapping requires a fabric whose goal_rank_mapping "
                    "is unique-nic"
                )
        self.placement = placement
        self.mode = mode
        host_by_rank: dict[int, str] = {}
        local_ranks: set[tuple[str, int]] = set()
        for index, rank in enumerate(placement.ranks):
            path = f"placement.ranks[{index}]"
            if not isinstance(rank, RankPlacement):
                raise TypeError(f"{path} must be a RankPlacement")
            if type(rank.global_rank) is not int or rank.global_rank < 0:
                raise ValueError(f"{path}.global_rank must be a nonnegative integer")
            if rank.global_rank in host_by_rank:
                raise ValueError(
                    f"placement contains duplicate global rank {rank.global_rank}"
                )
            if not isinstance(rank.hostname, str) or not rank.hostname.strip():
                raise ValueError(f"{path}.hostname must be a nonblank string")
            if type(rank.local_rank) is not int or rank.local_rank < 0:
                raise ValueError(f"{path}.local_rank must be a nonnegative integer")
            local_key = (rank.hostname, rank.local_rank)
            if local_key in local_ranks:
                raise ValueError(
                    "placement contains duplicate local rank "
                    f"{rank.local_rank} on host {rank.hostname!r}"
                )
            host_by_rank[rank.global_rank] = rank.hostname
            local_ranks.add(local_key)
        if not host_by_rank:
            raise ValueError("placement must contain at least one rank")
        # Locality is an immutable run projection. A caller that changes the
        # manifest must construct a new mapper rather than changing an active
        # sink's physical authority underneath it.
        self._host_by_rank = host_by_rank
        if fabric is not None:
            self.fabric = fabric
            self._join_fabric(fabric)

    def _join_fabric(self, fabric: FabricTopologyManifest) -> None:
        """Join GPU-to-NIC affinity to the placement; the fabric owns NIC order."""

        fabric.validate()
        nic_index: dict[str, int] = {}
        nic_node: dict[str, str] = {}
        nic_affine_rank: dict[str, int] = {}
        for node in fabric.nodes:
            for nic in node.nics:
                if nic.node_id != node.node_id:
                    raise ValueError(f"fabric NIC {nic.nic_id!r} disagrees with its node")
                nic_index[nic.nic_id] = len(nic_index)
                nic_node[nic.nic_id] = node.node_id
                nic_affine_rank[nic.nic_id] = nic.affine_gpu_rank
        nic_by_rank: dict[int, str] = {}
        for node in fabric.nodes:
            for gpu in node.gpus:
                rank = gpu.global_rank
                if type(rank) is not int or rank < 0:
                    raise ValueError("fabric GPU global_rank must be a nonnegative integer")
                if rank in nic_by_rank:
                    raise ValueError(f"fabric contains duplicate GPU rank {rank}")
                if gpu.node_id != node.node_id:
                    raise ValueError(f"fabric GPU {rank} disagrees with its node")
                if gpu.nic_id not in nic_index:
                    raise ValueError(f"fabric GPU {rank} names no NIC in the fabric")
                if nic_node[gpu.nic_id] != node.node_id:
                    raise ValueError(f"fabric GPU {rank} must name a NIC on its own node")
                if rank in self._host_by_rank and self._host_by_rank[rank] != node.node_id:
                    raise ValueError(
                        f"fabric node of GPU {rank} disagrees with the placement host"
                    )
                nic_by_rank[rank] = gpu.nic_id
        if set(nic_by_rank) != set(self._host_by_rank):
            raise ValueError("fabric GPU set must equal the placement rank set")
        served: dict[str, set[int]] = {}
        for rank, nic_id in nic_by_rank.items():
            served.setdefault(nic_id, set()).add(rank)
        for nic_id, affine_rank in nic_affine_rank.items():
            if nic_id not in served:
                raise ValueError(f"fabric NIC {nic_id!r} serves no GPU")
            if affine_rank not in served[nic_id]:
                raise ValueError(
                    f"fabric NIC {nic_id!r} names an affine GPU it does not serve"
                )
        self._nic_by_rank = nic_by_rank
        self._nic_count = len(nic_index)
        if self.mode == "unique-nic":
            self._goal_rank_by_rank = {
                rank: nic_index[nic_id] for rank, nic_id in nic_by_rank.items()
            }

    def goal_rank(self, global_rank: int) -> int:
        """GOAL rank for a global rank.

        The identity under ``gpu-rank``; under ``unique-nic``, the index of the
        rank's NIC in fabric order (node order, then NIC order within a node).
        """
        if global_rank not in self._host_by_rank:
            raise KeyError(f"global rank {global_rank} not in manifest")
        if self._goal_rank_by_rank is None:
            return global_rank
        return self._goal_rank_by_rank[global_rank]

    def num_goal_ranks(self) -> int:
        """GOAL rank count: max rank plus one, or the fabric NIC count."""
        if self.mode == "unique-nic":
            return self._nic_count
        return max(self._host_by_rank) + 1

    def nic_of(self, global_rank: int) -> str:
        """The affine ``nic_id`` of a global rank, read from the fabric manifest."""
        if self.fabric is None or self._nic_by_rank is None:
            raise ValueError("nic_of requires a fabric topology manifest")
        if global_rank not in self._nic_by_rank:
            raise KeyError(f"global rank {global_rank} not in manifest")
        return self._nic_by_rank[global_rank]

    def is_intra_node(self, rank_a: int, rank_b: int) -> bool:
        """Whether two ranks share a node (their traffic may bypass the fabric)."""
        try:
            return self._host_by_rank[rank_a] == self._host_by_rank[rank_b]
        except KeyError as exc:
            raise KeyError(f"global rank {exc.args[0]} not in manifest") from exc
