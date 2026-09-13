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

**Declared expert parallelism.** :class:`DeclaredExpertLayout` is the optional
mixture-of-experts (MoE) half of the same what-if statement: which pipeline
stage owns which hidden layers, and which global expert ids of each MoE layer
live on which rank. The rules below are read from the pinned vLLM 0.27.1
package, so a declared manifest and a manifest extracted from that release
describe the same deployment. File names are given relative to the installed
``vllm`` package.

- Rank tensor. ``distributed/parallel_state.py``
  (``initialize_model_parallel``) shapes ``arange(world)`` as
  ``(external DP, DP, PP, PCP, TP)``, so the global rank is
  ``(dp * PP + pp) * TP + tp``. Prefill context parallelism (PCP) is one
  throughout this module, which is why it does not appear in the formula.
- Expert-parallel (EP) group. That same function transposes the DP and PP
  axes and flattens ``DP * PCP * TP``, so for every pipeline index the EP
  group holds the ``DP * TP`` ranks of that stage, ordered data-parallel
  major and tensor-parallel minor, and a member's index inside the group is
  ``dp * TP + tp``. vLLM creates this group for MoE models only, which is why
  the manifest gains it only when an expert layout is declared.
- EP size. With expert parallelism in use the fused MoE layer
  (``model_executor/layers/fused_moe/config.py``) flattens the tensor group
  across data parallel, so ``ep_size = DP * TP`` and the experts are not
  tensor sharded at all.
- Expert map. ``model_executor/layers/fused_moe/expert_map_manager.py``
  (``determine_expert_map``) builds the per-rank expert map from the
  placement strategy. With ``base = num_experts // ep_size`` and
  ``remainder = num_experts % ep_size``, EP rank ``r`` owns ``base + 1``
  experts when ``r < remainder`` and ``base`` otherwise. ``linear`` gives it
  the contiguous block starting at ``r * base + min(r, remainder)``, and
  ``round_robin`` gives it every ``ep_size``-th expert from ``r`` upward,
  ``range(r, num_experts, ep_size)``, which yields the same counts.
- Pipeline partition. ``distributed/utils.py`` (``get_pp_indices``) splits
  ``L`` hidden layers into ``base = L // PP`` layers per stage and hands the
  ``L mod PP`` remainder to the stages indexed ``-2, -3, ...`` in that order,
  so the last stage never gains a layer and the first gains one only when the
  remainder exceeds ``PP - 2``. Stage ``p`` then owns
  ``[sum(partitions[:p]), sum(partitions[:p + 1]))``. The environment override
  ``VLLM_PP_LAYER_PARTITION`` is deliberately not modeled: a declared layout
  states its own partition through ``num_layers`` and ``pp``.

The ``round_robin`` strategy is the caller's declaration. vLLM falls back to
``linear`` on its own when the model has at most one expert group, has
redundant experts, enables expert-parallel load balancing, or uses an
all-to-all backend without round-robin routing tables; that fallback is not
modeled here, and an extracted manifest records the map that really ran.

Omitting ``experts`` is the explicit off path. It adds no group, no layer
range and no expert ownership, and every manifest built without it is byte
identical to the pre-expert output.
"""

from __future__ import annotations

from dataclasses import dataclass

from simllm.placement.manifest import GroupMembership, PlacementManifest, RankPlacement

#: Expert placement strategies the pinned vLLM fused MoE layer implements.
DECLARED_EXPERT_PLACEMENT_STRATEGIES = ("linear", "round_robin")


def _positive_int(name: str, value: object, *, minimum: int) -> int:
    """Return ``value`` as a plain integer at or above ``minimum``."""

    # Booleans are integers in Python; a layout field spelled True is a
    # caller mistake, never a width of one.
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class DeclaredExpertLayout:
    """The declared MoE half of a what-if placement.

    ``num_layers`` is the hidden-layer count the pipeline partition splits.
    ``moe_layers`` are the ascending layer indices carrying a routed MoE
    block; they are stored as a tuple so the layout stays hashable and
    cannot be mutated after validation. ``num_experts`` is the global expert
    count of one MoE layer, before any expert-parallel split.
    ``placement_strategy`` selects the pinned expert map, and
    ``placement_epoch`` is the expert-placement epoch stamped on every rank,
    so a consumer can join routed traffic to the placement it was routed
    under.

    Field-local rules are checked here. The one rule that depends on the
    surrounding parallel layout, ``num_layers >= pp``, is checked by
    :func:`declared_manifest`.
    """

    num_layers: int
    moe_layers: tuple[int, ...]
    num_experts: int
    placement_strategy: str = "linear"
    placement_epoch: int = 0

    def __post_init__(self) -> None:
        _positive_int("num_layers", self.num_layers, minimum=1)
        _positive_int("num_experts", self.num_experts, minimum=1)
        _positive_int("placement_epoch", self.placement_epoch, minimum=0)
        if self.placement_strategy not in DECLARED_EXPERT_PLACEMENT_STRATEGIES:
            raise ValueError(
                "placement_strategy must be one of "
                f"{DECLARED_EXPERT_PLACEMENT_STRATEGIES}, got "
                f"{self.placement_strategy!r}"
            )
        layers = tuple(self.moe_layers)
        if not layers:
            raise ValueError("moe_layers must name at least one layer")
        previous = -1
        for index, layer in enumerate(layers):
            _positive_int(f"moe_layers[{index}]", layer, minimum=0)
            if layer >= self.num_layers:
                raise ValueError(
                    f"moe_layers[{index}] must be < num_layers "
                    f"{self.num_layers}, got {layer}"
                )
            if layer <= previous:
                raise ValueError(
                    "moe_layers must be strictly ascending, got "
                    f"{layer} after {previous}"
                )
            previous = layer
        object.__setattr__(self, "moe_layers", layers)


def declared_pipeline_partition(num_layers: int, pp: int) -> tuple[tuple[int, int], ...]:
    """Return one ``[start, end)`` layer interval per pipeline stage.

    This is the pinned ``get_pp_indices`` rule of vLLM 0.27.1 with no
    environment override: every stage takes ``num_layers // pp`` layers, then
    the remainder is handed out one layer at a time to the stages indexed
    ``-2, -3, ...`` in that order. A stage with zero layers is refused rather
    than emitted, because a pipeline stage that owns no layer has no work and
    no meaningful expert ownership.
    """

    _positive_int("num_layers", num_layers, minimum=1)
    _positive_int("pp", pp, minimum=1)
    if num_layers < pp:
        raise ValueError(
            f"num_layers must be >= pp {pp} so every stage owns a layer, "
            f"got {num_layers}"
        )
    base, remainder = divmod(num_layers, pp)
    sizes = [base] * pp
    for offset in range(2, remainder + 2):
        sizes[-offset] += 1
    intervals: list[tuple[int, int]] = []
    start = 0
    for size in sizes:
        intervals.append((start, start + size))
        start += size
    return tuple(intervals)


def declared_local_expert_ids(
    num_experts: int,
    ep_size: int,
    ep_rank: int,
    strategy: str = "linear",
) -> tuple[int, ...]:
    """Return the global expert ids one EP rank owns, in ascending order.

    This is the pinned vLLM expert map, remainder included: EP rank ``r``
    owns ``num_experts // ep_size`` experts, plus one when ``r`` is below
    ``num_experts % ep_size``. ``linear`` hands out that many contiguous
    experts starting at ``r * base + min(r, remainder)``; ``round_robin``
    hands out every ``ep_size``-th expert starting at ``ep_rank``.
    """

    _positive_int("num_experts", num_experts, minimum=1)
    _positive_int("ep_size", ep_size, minimum=1)
    _positive_int("ep_rank", ep_rank, minimum=0)
    if ep_rank >= ep_size:
        raise ValueError(f"ep_rank must be < ep_size {ep_size}, got {ep_rank}")
    if strategy not in DECLARED_EXPERT_PLACEMENT_STRATEGIES:
        raise ValueError(
            "placement_strategy must be one of "
            f"{DECLARED_EXPERT_PLACEMENT_STRATEGIES}, got {strategy!r}"
        )
    base, remainder = divmod(num_experts, ep_size)
    if strategy == "linear":
        count = base + 1 if ep_rank < remainder else base
        start = ep_rank * base + min(ep_rank, remainder)
        return tuple(range(start, start + count))
    return tuple(range(ep_rank, num_experts, ep_size))


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
    experts: DeclaredExpertLayout | None = None,
) -> PlacementManifest:
    """Build a ``source="declared"`` manifest for a DP x PP x TP layout.

    ``nodes`` defaults to the minimum node count that fits the world at
    ``gpus_per_node``; passing it explicitly (e.g. to model an
    under-populated cluster) must still fit the world. Hostnames come from
    ``hostname_pattern.format(node_index)``.

    ``experts`` is the optional declared MoE layout. With it present every
    rank additionally carries an ``ep`` group membership (inserted after
    ``tp``, ``pp`` and ``dp``), the ``[start, end)`` hidden-layer interval of
    its pipeline stage, the global expert ids it owns in each MoE layer of
    that interval, and the declared placement epoch. With it absent the
    manifest is exactly the expert-free placement, byte for byte.
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

    # Resolve everything the expert layout implies before the first rank is
    # built, so a refusal never leaves a half-populated manifest behind.
    stage_intervals: tuple[tuple[int, int], ...] = ()
    stage_moe_layers: tuple[tuple[int, ...], ...] = ()
    ep_size = tp * dp
    if experts is not None:
        if not isinstance(experts, DeclaredExpertLayout):
            raise TypeError("experts must be a DeclaredExpertLayout or None")
        if experts.num_layers < pp:
            raise ValueError(
                f"num_layers must be >= pp {pp} so every stage owns a layer, "
                f"got {experts.num_layers}"
            )
        stage_intervals = declared_pipeline_partition(experts.num_layers, pp)
        stage_moe_layers = tuple(
            tuple(layer for layer in experts.moe_layers if start <= layer < end)
            for start, end in stage_intervals
        )

    ranks: list[RankPlacement] = []
    for dp_index in range(dp):
        for pp_index in range(pp):
            for tp_index in range(tp):
                global_rank = (dp_index * pp + pp_index) * tp + tp_index
                tp_ranks = [(dp_index * pp + pp_index) * tp + t for t in range(tp)]
                pp_ranks = [(dp_index * pp + p) * tp + tp_index for p in range(pp)]
                dp_ranks = [(d * pp + pp_index) * tp + tp_index for d in range(dp)]
                groups = {
                    "tp": GroupMembership(tp_index, tp_ranks),
                    "pp": GroupMembership(pp_index, pp_ranks),
                    "dp": GroupMembership(dp_index, dp_ranks),
                }
                layer_range: tuple[int, int] | None = None
                local_expert_ids: dict[int, list[int]] = {}
                placement_epoch = 0
                if experts is not None:
                    # The EP group of this rank is its whole pipeline stage
                    # flattened data-parallel major, tensor-parallel minor.
                    ep_ranks = [
                        (d * pp + pp_index) * tp + t
                        for d in range(dp)
                        for t in range(tp)
                    ]
                    ep_rank = dp_index * tp + tp_index
                    groups["ep"] = GroupMembership(ep_rank, ep_ranks)
                    layer_range = stage_intervals[pp_index]
                    owned = list(
                        declared_local_expert_ids(
                            experts.num_experts,
                            ep_size,
                            ep_rank,
                            experts.placement_strategy,
                        )
                    )
                    local_expert_ids = {
                        layer: list(owned) for layer in stage_moe_layers[pp_index]
                    }
                    placement_epoch = experts.placement_epoch
                ranks.append(
                    RankPlacement(
                        global_rank=global_rank,
                        hostname=hostname_pattern.format(global_rank // gpus_per_node),
                        local_rank=global_rank % gpus_per_node,
                        groups=groups,
                        pipeline_layer_range=layer_range,
                        local_expert_ids=local_expert_ids,
                        placement_epoch=placement_epoch,
                    )
                )
    ranks.sort(key=lambda placement: placement.global_rank)
    return PlacementManifest(
        ranks=ranks,
        source="declared",
        framework=framework,
        framework_version=framework_version,
    )
