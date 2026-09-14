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
  Under ``round_robin`` that ``torch.arange`` raises for an EP rank above
  the expert count, so a layout with ``num_experts < ep_size - 1`` is
  refused; a rank at the expert count owns nothing, as ranks beyond the
  remainder do under ``linear``.
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

**Declared SGLang layout.** :func:`declared_sglang_manifest` is the second
declared builder. It answers the same what-if question for a deployment that
runs SGLang instead of vLLM, and it is a separate entry point rather than an
option because the two frameworks disagree about the rank space itself. The
rules below are read from the installed SGLang package at the pinned commit
``bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3``, whose distribution reports
``0.5.6.post3.dev9406+gbfeae4e79``. File names are given relative to the
installed ``sglang/srt`` package. Attention data parallelism, attention and
decode context parallelism, and the elastic expert-parallel joiner offset are
one or zero throughout.

- Rank formula. ``distributed/bootstrap.py`` computes
  ``world_size = tp_size * pp_size`` and ``rank = tp_size * pp_rank + tp_rank``,
  and ``distributed/parallel_state.py`` (``initialize_model_parallel``) refuses
  any other world size. A tensor group is a contiguous block of ``tp`` ranks
  and a pipeline group strides by ``tp``. There is no data-parallel axis in the
  rank space: ``layers/dp_attention.py`` places attention data parallelism
  inside the tensor group, and ``managers/data_parallel_controller.py`` launches
  router-style replicas as separate worlds with their own GPU offset. The
  declared manifest describes one replica, so every rank carries a singleton
  ``dp`` membership.
- MoE sizes. ``initialize_model_parallel`` sets ``moe_ep_size`` from
  ``--ep-size``, ``moe_dp_size`` from ``--moe-dp-size``, and
  ``moe_tp_size = tp // moe_ep_size // moe_dp_size``. ``server_args.py``
  asserts, only when ``moe_dp_size > 1``, that ``tp % moe_dp_size == 0``,
  ``ep_size * moe_dp_size <= tp``, ``pp == 1``, and, when also ``ep_size > 1``,
  ``ep_size * moe_dp_size == tp``. The general divisibility of ``tp`` by
  ``ep_size * moe_dp_size`` is asserted by the framework only for quantized
  models; the declared builder refuses it for every layout, because the group
  formulas cover the tensor group exactly once only when it holds.
- EP group. For the tensor group with base ``b = pp_rank * tp``, each
  ``moe_dp_idx`` and each ``moe_tp_idx``, the group is
  ``range(s, s + ep * moe_tp, moe_tp)`` with
  ``s = b + moe_dp_idx * ep * moe_tp + moe_tp_idx``. A rank's index inside it
  is ``tp_rank % (tp // moe_dp) // moe_tp``, the formula the scheduler launcher
  prints. At ``ep_size == tp`` the group is the tensor group itself.
- MoE tensor group. For each combined ``ep_dp_idx`` in ``range(ep * moe_dp)``,
  the contiguous block ``range(b + ep_dp_idx * moe_tp, b + (ep_dp_idx + 1) *
  moe_tp)``; a rank's index inside it is ``tp_rank % moe_tp``.
- MoE data-parallel group. For each ``idx`` in ``range(moe_tp * ep)``, the
  strided set ``range(b + idx, b + tp + idx, moe_tp * ep)``; a rank's index
  inside it is ``tp_rank // (tp // moe_dp)``.
- Expert map. With the default ``--init-expert-location trivial``,
  ``eplb/expert_location.py`` maps physical expert ``i`` to logical expert
  ``i`` and asserts ``num_experts % ep_size == 0``, and
  ``layers/moe/fused_moe_triton/layer.py`` gives EP rank ``r`` the contiguous
  experts ``[r * L, (r + 1) * L)`` with ``L = num_experts // ep_size``. There
  is no round-robin placement, so the SGLang builder refuses that strategy and
  refuses an expert count the EP size does not divide. Under ``moe_tp > 1``
  every rank of one MoE tensor group owns the same expert ids and holds one
  shard of each; the shard is not represented, which is the gap PLACE-7
  already records for vLLM.
- Pipeline partition. ``distributed/utils.py`` (``get_pp_indices``) gives every
  stage ``base = L // PP`` layers and one extra layer to each of the *last*
  ``L mod PP`` stages, where vLLM hands the remainder to the stages indexed
  ``-2, -3, ...``. This is why a 61-layer model on four stages is partitioned
  differently by the two frameworks. The environment override
  ``SGLANG_PP_LAYER_PARTITION`` is deliberately not modeled, for the same
  reason its vLLM counterpart is not.
- GPU and node placement. ``managers/data_parallel_controller.py`` hosts
  ``pp // nnodes`` pipeline stages per node when ``nnodes <= pp``, and
  otherwise spreads one stage over ``nnodes // pp`` nodes carrying
  ``tp // (nnodes // pp)`` tensor ranks each. In both cases ranks fill nodes in
  global-rank order, ``world // nnodes`` at a time, so the declared builder
  uses that fill and refuses a node count that neither divides nor is divided
  by ``pp``, or that leaves ``tp`` indivisible by ``nnodes // pp``.

The two builders never share an output. :func:`declared_manifest` keeps its
vLLM rank space, its ``ep`` group and its partition exactly as they are, and
:func:`declared_sglang_manifest` writes ``framework="sglang"`` on everything it
emits, so a consumer can always tell which layout rules it is holding.
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


def _check_round_robin_expert_count(num_experts: int, ep_size: int, strategy: str) -> None:
    """Refuse a ``round_robin`` layout the pinned expert map cannot build.

    ``torch.arange(r, num_experts, ep_size)`` raises for every EP rank ``r``
    above ``num_experts``, so the highest rank ``ep_size - 1`` fails whenever
    ``num_experts < ep_size - 1``.
    """

    if strategy == "round_robin" and num_experts < ep_size - 1:
        raise ValueError(
            f"num_experts {num_experts} must be >= ep_size - 1 ({ep_size - 1}) "
            "under round_robin placement"
        )


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
    hands out every ``ep_size``-th expert starting at ``ep_rank``. A
    ``round_robin`` layout whose EP size exceeds the expert count by more than
    one is refused, because the framework's expert map fails for its highest
    ranks.
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
    _check_round_robin_expert_count(num_experts, ep_size, strategy)
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
        _check_round_robin_expert_count(
            experts.num_experts, ep_size, experts.placement_strategy
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


def declared_sglang_pipeline_partition(
    num_layers: int, pp: int
) -> tuple[tuple[int, int], ...]:
    """Return one ``[start, end)`` layer interval per SGLang pipeline stage.

    This is the pinned ``get_pp_indices`` rule of SGLang at commit
    ``bfeae4e7`` with no environment override: every stage takes
    ``num_layers // pp`` layers and each of the last ``num_layers mod pp``
    stages takes one extra layer, so stage ``p`` starts at
    ``p * (base + 1) - (pp - remainder)`` when it is one of those and at
    ``p * base`` otherwise. vLLM hands the same remainder to the stages
    indexed ``-2, -3, ...``, which is why :func:`declared_pipeline_partition`
    returns different intervals for the same ``(num_layers, pp)``. A stage
    with zero layers is refused rather than emitted, because a pipeline stage
    that owns no layer has no work and no meaningful expert ownership.
    """

    _positive_int("num_layers", num_layers, minimum=1)
    _positive_int("pp", pp, minimum=1)
    if num_layers < pp:
        raise ValueError(
            f"num_layers must be >= pp {pp} so every stage owns a layer, "
            f"got {num_layers}"
        )
    base, remainder = divmod(num_layers, pp)
    without_extra = pp - remainder
    intervals: list[tuple[int, int]] = []
    for stage in range(pp):
        if stage >= without_extra:
            start = stage * (base + 1) - without_extra
            intervals.append((start, start + base + 1))
        else:
            start = stage * base
            intervals.append((start, start + base))
    return tuple(intervals)


def _sglang_node_fill(
    tp: int, pp: int, nodes: int | None, gpus_per_node: int
) -> tuple[int, int]:
    """Resolve the node count and the ranks each node holds, or refuse.

    Ranks fill nodes in global-rank order, ``world // nodes`` at a time, which
    is what the SGLang launcher does in both of its arrangements. ``nodes``
    defaults to the fewest that fit the world at ``gpus_per_node``; the
    launcher's own constraints are then checked against whatever count is in
    force, so an explicit and a defaulted count are refused on the same terms.
    """

    world = tp * pp
    if nodes is None:
        nodes = -(-world // gpus_per_node)
    else:
        _positive_int("nodes", nodes, minimum=1)
    if world % nodes:
        raise ValueError(f"nodes {nodes} must divide the world size {world}")
    ranks_per_node = world // nodes
    if ranks_per_node > gpus_per_node:
        raise ValueError(
            f"nodes {nodes} leaves {ranks_per_node} ranks on a node of "
            f"{gpus_per_node} GPUs"
        )
    if pp % nodes and nodes % pp:
        raise ValueError(f"nodes {nodes} must divide or be divided by pp {pp}")
    if nodes > pp and tp % (nodes // pp):
        raise ValueError(
            f"nodes {nodes} spreads one pipeline stage over {nodes // pp} nodes, "
            f"which must divide tp {tp}"
        )
    return nodes, ranks_per_node


def declared_sglang_manifest(
    *,
    tp: int = 1,
    pp: int = 1,
    ep_size: int = 1,
    moe_dp_size: int = 1,
    nodes: int | None = None,
    gpus_per_node: int = 8,
    hostname_pattern: str = "node-{}",
    framework_version: str | None = None,
    experts: DeclaredExpertLayout | None = None,
) -> PlacementManifest:
    """Build a ``source="declared"``, ``framework="sglang"`` manifest.

    The world is ``tp * pp`` with no data-parallel term, so every rank carries
    a ``tp`` membership, a ``pp`` membership and a singleton ``dp`` membership,
    in that order. ``ep_size`` is SGLang's ``--ep-size`` and ``moe_dp_size``
    its ``--moe-dp-size``; the MoE tensor width follows as
    ``tp // ep_size // moe_dp_size``. ``nodes`` is ``--nnodes`` and defaults to
    the fewest that fit the world at ``gpus_per_node``.

    ``experts`` is the optional declared MoE layout, the same
    :class:`DeclaredExpertLayout` the vLLM builder takes. With it present every
    rank additionally carries the ``ep``, ``moe_tp`` and ``moe_dp``
    memberships, in that order after ``dp``, the ``[start, end)`` hidden-layer
    interval of its SGLang pipeline stage, the contiguous block of global
    expert ids its EP rank owns in each MoE layer of that interval, and the
    declared placement epoch. With it absent the manifest carries the three
    base memberships and nothing else.

    Every refusal is a :class:`ValueError` naming the field at fault, raised
    before the first rank is built so a refusal never leaves a half-populated
    manifest behind.
    """

    _positive_int("tp", tp, minimum=1)
    _positive_int("pp", pp, minimum=1)
    _positive_int("ep_size", ep_size, minimum=1)
    _positive_int("moe_dp_size", moe_dp_size, minimum=1)
    _positive_int("gpus_per_node", gpus_per_node, minimum=1)
    if tp % (ep_size * moe_dp_size):
        raise ValueError(
            f"tp {tp} must be divisible by ep_size * moe_dp_size "
            f"({ep_size * moe_dp_size})"
        )
    if moe_dp_size > 1:
        if pp > 1:
            raise ValueError(f"moe_dp_size {moe_dp_size} requires pp 1, got pp {pp}")
        if ep_size > 1 and ep_size * moe_dp_size != tp:
            raise ValueError(
                f"moe_dp_size {moe_dp_size} with ep_size {ep_size} requires "
                f"ep_size * moe_dp_size == tp {tp}"
            )
    nodes, ranks_per_node = _sglang_node_fill(tp, pp, nodes, gpus_per_node)
    moe_tp_size = tp // ep_size // moe_dp_size
    moe_dp_stride = tp // moe_dp_size

    # Resolve everything the expert layout implies before the first rank is
    # built, so a refusal never leaves a half-populated manifest behind.
    stage_intervals: tuple[tuple[int, int], ...] = ()
    stage_moe_layers: tuple[tuple[int, ...], ...] = ()
    if experts is not None:
        if not isinstance(experts, DeclaredExpertLayout):
            raise TypeError("experts must be a DeclaredExpertLayout or None")
        if experts.placement_strategy != "linear":
            raise ValueError(
                "placement_strategy must be 'linear' under the SGLang expert "
                f"map, got {experts.placement_strategy!r}"
            )
        if experts.num_experts % ep_size:
            raise ValueError(
                f"num_experts {experts.num_experts} must be divisible by "
                f"ep_size {ep_size}"
            )
        if experts.num_layers < pp:
            raise ValueError(
                f"num_layers must be >= pp {pp} so every stage owns a layer, "
                f"got {experts.num_layers}"
            )
        stage_intervals = declared_sglang_pipeline_partition(experts.num_layers, pp)
        stage_moe_layers = tuple(
            tuple(layer for layer in experts.moe_layers if start <= layer < end)
            for start, end in stage_intervals
        )

    ranks: list[RankPlacement] = []
    for pp_index in range(pp):
        base = pp_index * tp
        for tp_index in range(tp):
            global_rank = base + tp_index
            groups = {
                "tp": GroupMembership(tp_index, [base + t for t in range(tp)]),
                "pp": GroupMembership(pp_index, [p * tp + tp_index for p in range(pp)]),
                # One replica: SGLang's router-style data parallelism launches
                # a separate world, so this rank is its own dp group.
                "dp": GroupMembership(0, [global_rank]),
            }
            layer_range: tuple[int, int] | None = None
            local_expert_ids: dict[int, list[int]] = {}
            placement_epoch = 0
            if experts is not None:
                moe_dp_index = tp_index // moe_dp_stride
                moe_tp_index = tp_index % moe_tp_size
                ep_rank = tp_index % moe_dp_stride // moe_tp_size
                ep_start = base + moe_dp_index * ep_size * moe_tp_size + moe_tp_index
                groups["ep"] = GroupMembership(
                    ep_rank,
                    list(
                        range(
                            ep_start,
                            ep_start + ep_size * moe_tp_size,
                            moe_tp_size,
                        )
                    ),
                )
                moe_tp_block = tp_index // moe_tp_size
                groups["moe_tp"] = GroupMembership(
                    moe_tp_index,
                    list(
                        range(
                            base + moe_tp_block * moe_tp_size,
                            base + (moe_tp_block + 1) * moe_tp_size,
                        )
                    ),
                )
                moe_dp_offset = tp_index % moe_dp_stride
                groups["moe_dp"] = GroupMembership(
                    moe_dp_index,
                    list(
                        range(
                            base + moe_dp_offset,
                            base + tp + moe_dp_offset,
                            moe_dp_stride,
                        )
                    ),
                )
                layer_range = stage_intervals[pp_index]
                owned = list(
                    declared_local_expert_ids(
                        experts.num_experts, ep_size, ep_rank, "linear"
                    )
                )
                local_expert_ids = {
                    layer: list(owned) for layer in stage_moe_layers[pp_index]
                }
                placement_epoch = experts.placement_epoch
            ranks.append(
                RankPlacement(
                    global_rank=global_rank,
                    hostname=hostname_pattern.format(global_rank // ranks_per_node),
                    local_rank=global_rank % ranks_per_node,
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
        framework="sglang",
        framework_version=framework_version,
    )
