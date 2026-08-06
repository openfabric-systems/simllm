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

from simllm.placement.manifest import PlacementManifest

GOAL_RANK_MAPPINGS = ("gpu-rank", "unique-nic")


class RankMapper:
    def __init__(self, placement: PlacementManifest, mode: str = "gpu-rank"):
        if mode not in GOAL_RANK_MAPPINGS:
            raise ValueError(f"mode must be one of {GOAL_RANK_MAPPINGS}")
        if mode == "unique-nic":
            raise NotImplementedError(
                "unique-nic mapping is deferred behind the fabric manifest (PLACE-2)"
            )
        self.placement = placement
        self.mode = mode

    def goal_rank(self, global_rank: int) -> int:
        """GOAL rank for a global rank (identity under gpu-rank mapping)."""
        self.placement.by_rank(global_rank)  # raise on unknown ranks
        return global_rank

    def num_goal_ranks(self) -> int:
        return len(self.placement.ranks)

    def is_intra_node(self, rank_a: int, rank_b: int) -> bool:
        """Whether two ranks share a node (their traffic may bypass the fabric)."""
        return self.placement.by_rank(rank_a).hostname == self.placement.by_rank(rank_b).hostname
