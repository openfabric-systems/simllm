"""Declared placement manifests: what-if placements computed from a layout.

A declared manifest describes a deployment that does not exist yet, so its
group memberships are necessarily computed from the parallel layout rather
than exported from live process groups. This is the one place where that is
correct, and it is worth stating the distinction precisely because the
extraction rule (see :mod:`simllm.placement.manifest`) forbids exactly this
computation for live runs:

- **Extracted** manifests (live runs) must export the *actual*
  ``GroupCoordinator.ranks`` lists, never recompute them from the rank
  formula: external DP, elastic scaling or framework changes silently break
  derived layouts, and the manifest exists to record what really happened.
- **Declared** manifests (this module) have no live deployment to ask; the
  layout formula *is* the specification of the what-if placement being
  simulated. ``source="declared"`` marks the provenance so a consumer can
  always tell which kind it holds.

The layout is the standard DP x PP x TP nesting (TP innermost, matching
vLLM's default group construction)::

    global_rank = (dp_index * PP + pp_index) * TP + tp_index

so a TP group is a block of consecutive ranks, PP peers stride by TP, and
DP peers stride by PP * TP. Ranks fill nodes in global-rank order,
``gpus_per_node`` at a time.
"""

from __future__ import annotations

from simllm.placement.manifest import GroupMembership, PlacementManifest, RankPlacement


def declared_manifest(
    *,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    nodes: int | None = None,
    gpus_per_node: int = 8,
    hostname_pattern: str = "node-{}",
    framework: str | None = None,
    framework_version: str | None = None,
) -> PlacementManifest:
    """Build a ``source="declared"`` manifest for a DP x PP x TP layout.

    ``nodes`` defaults to the minimum node count that fits the world at
    ``gpus_per_node``; passing it explicitly (e.g. to model an
    under-populated cluster) must still fit the world. Hostnames come from
    ``hostname_pattern.format(node_index)``.
    """
    for name, value in (("tp", tp), ("pp", pp), ("dp", dp)):
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
    if gpus_per_node < 1:
        raise ValueError(f"gpus_per_node must be >= 1, got {gpus_per_node}")
    world = tp * pp * dp
    min_nodes = -(-world // gpus_per_node)
    if nodes is None:
        nodes = min_nodes
    elif nodes < min_nodes:
        raise ValueError(
            f"world size {world} does not fit on {nodes} nodes of {gpus_per_node} GPUs"
        )

    ranks: list[RankPlacement] = []
    for dp_index in range(dp):
        for pp_index in range(pp):
            for tp_index in range(tp):
                global_rank = (dp_index * pp + pp_index) * tp + tp_index
                tp_ranks = [(dp_index * pp + pp_index) * tp + t for t in range(tp)]
                pp_ranks = [(dp_index * pp + p) * tp + tp_index for p in range(pp)]
                dp_ranks = [(d * pp + pp_index) * tp + tp_index for d in range(dp)]
                ranks.append(
                    RankPlacement(
                        global_rank=global_rank,
                        hostname=hostname_pattern.format(global_rank // gpus_per_node),
                        local_rank=global_rank % gpus_per_node,
                        groups={
                            "tp": GroupMembership(tp_index, tp_ranks),
                            "pp": GroupMembership(pp_index, pp_ranks),
                            "dp": GroupMembership(dp_index, dp_ranks),
                        },
                    )
                )
    ranks.sort(key=lambda placement: placement.global_rank)
    return PlacementManifest(
        ranks=ranks,
        source="declared",
        framework=framework,
        framework_version=framework_version,
    )
