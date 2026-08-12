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

from simllm.placement.manifest import PLACEMENT_SCHEMA, PlacementManifest, RankPlacement

GOAL_RANK_MAPPINGS = ("gpu-rank", "unique-nic")


class RankMapper:
    def __init__(self, placement: PlacementManifest, mode: str = "gpu-rank"):
        if not isinstance(placement, PlacementManifest):
            raise TypeError("placement must be a PlacementManifest")
        if placement.schema != PLACEMENT_SCHEMA:
            raise ValueError(f"unsupported placement schema: {placement.schema!r}")
        if mode not in GOAL_RANK_MAPPINGS:
            raise ValueError(f"mode must be one of {GOAL_RANK_MAPPINGS}")
        if mode == "unique-nic":
            raise NotImplementedError(
                "unique-nic mapping is deferred behind the fabric manifest (PLACE-2)"
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

    def goal_rank(self, global_rank: int) -> int:
        """GOAL rank for a global rank (identity under gpu-rank mapping)."""
        if global_rank not in self._host_by_rank:
            raise KeyError(f"global rank {global_rank} not in manifest")
        return global_rank

    def num_goal_ranks(self) -> int:
        return max(self._host_by_rank) + 1

    def is_intra_node(self, rank_a: int, rank_b: int) -> bool:
        """Whether two ranks share a node (their traffic may bypass the fabric)."""
        try:
            return self._host_by_rank[rank_a] == self._host_by_rank[rank_b]
        except KeyError as exc:
            raise KeyError(f"global rank {exc.args[0]} not in manifest") from exc
