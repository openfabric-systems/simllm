"""Per-step collective work (TP allreduces, MoE all-to-alls) from a step record.

Maps one :class:`~simllm.core.StepRecord` plus a per-rank
:class:`~simllm.compute.ModelDims` plus GOAL rank groups to the collective
traffic of that engine step.

Tensor parallel (:func:`step_tp_allreduces`): per transformer layer, two
ring allreduces (the attention output projection and the MLP output
projection), each moving

    payload_bytes = record.total_new_tokens * dims.hidden_size * dims.dtype_bytes

which is the activation tensor of every token computed this step. A TP
world of size 1 (or a drain record with zero new tokens) produces no
operations.

Expert parallel (:func:`step_moe_alltoalls`): per MoE layer, a dispatch
pairwise all-to-allv (tokens to their experts' owner ranks) followed by a
combine pairwise all-to-allv (expert outputs back). An optional captured
routing supply selects per-token destinations at an explicit placement epoch;
its absence retains the uniform compatibility assumption documented on the
function.

This is a deliberately first-order model of step traffic; each
simplification is a numbered task in docs/modules/traffic.md:

- no sequence parallelism (which would replace each allreduce with a
  reduce-scatter plus allgather of 1/W the bytes framed around the
  norm/dropout regions): TRAF-6;
- no communication/compute overlap (each layer's compute and its
  collectives are a strict serial chain): TRAF-7;
- no pipeline-parallel activation traffic (records carry no PP stage
  information yet): TRAF-8;
- the MoE layer is rendered as one calc, then the TP allreduces, then
  dispatch and combine back to back, instead of splitting the layer's
  compute around the all-to-alls: TRAF-9.

Rendering reuses :func:`simllm.traffic.patterns.ring_allreduce` and
:func:`simllm.traffic.patterns.pairwise_all_to_allv` unchanged, so the GOAL
structures are exactly the patterns validated by the M1/M4/M5 studies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from simllm.compute import ModelDims
from simllm.core import RequestPhase, StepRecord
from simllm.goal import GoalTrace
from simllm.traffic.patterns import pairwise_all_to_allv, ring_allreduce
from simllm.traffic.routed_moe import RoutedMoeSupply

#: the two allreduce sites of one transformer layer, in execution order
TP_ALLREDUCE_SITES = ("attention", "mlp")

#: the two all-to-allv phases of one MoE layer, in execution order
MOE_A2A_PHASES = ("dispatch", "combine")


@dataclass(frozen=True)
class TpAllReduce:
    """One tensor-parallel allreduce of one layer of one step."""

    layer: int
    #: "attention" or "mlp"
    site: str
    ranks: tuple[int, ...]
    payload_bytes: int


def step_tp_allreduces(
    record: StepRecord, dims: ModelDims, tp_ranks: Sequence[int]
) -> list[TpAllReduce]:
    """The step's TP collectives, empty when the step produces no traffic.

    Empty means either a TP world of size 1 (nothing to reduce across) or a
    record with zero new tokens (a drain record carrying only completions).
    """
    ranks = tuple(tp_ranks)
    if len(ranks) < 2:
        return []
    payload = record.total_new_tokens * dims.hidden_size * dims.dtype_bytes
    if payload <= 0:
        return []
    return [
        TpAllReduce(layer=layer, site=site, ranks=ranks, payload_bytes=payload)
        for layer in range(dims.num_layers)
        for site in TP_ALLREDUCE_SITES
    ]


@dataclass(frozen=True)
class MoeAllToAll:
    """One MoE all-to-allv phase of one layer of one step."""

    layer: int
    #: "dispatch" or "combine"
    phase: str
    ranks: tuple[int, ...]
    #: bytes each rank sends to EACH other rank (uniform routing)
    per_pair_bytes: int
    #: sparse ordered-pair bytes when captured routing is authoritative
    pair_payload_bytes: tuple[tuple[int, int, int], ...] = ()
    #: expert ownership epoch used to derive the sparse table
    placement_epoch: int = 0


def _scheduled_routed_tokens(record: StepRecord, supply: RoutedMoeSupply):
    from simllm.preplay.routing import RoutedToken

    request_ids = [request.request_id for request in record.scheduled]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("record.scheduled: duplicate request identity")
    tokens: list[RoutedToken] = []
    for index, scheduled in enumerate(record.scheduled):
        path = f"record.scheduled[{index}]"
        if (
            isinstance(scheduled.num_new_tokens, bool)
            or not isinstance(scheduled.num_new_tokens, int)
        ):
            raise TypeError(f"{path}.num_new_tokens: expected an integer")
        if scheduled.num_new_tokens < 0:
            raise ValueError(f"{path}.num_new_tokens: expected a nonnegative integer")
        if (
            isinstance(scheduled.context_length, bool)
            or not isinstance(scheduled.context_length, int)
        ):
            raise TypeError(f"{path}.context_length: expected an integer")
        try:
            routed_request = supply.routed_experts.by_request_id(
                scheduled.request_id
            )
        except KeyError as exc:
            raise ValueError(
                f"{path}.request_id: absent from routed-experts projection"
            ) from exc
        if scheduled.num_new_tokens == 0:
            continue
        if scheduled.phase is RequestPhase.PREFILL:
            end = scheduled.context_length
            start = end - scheduled.num_new_tokens
            phase_tokens = routed_request.prefill_tokens
            phase = "prefill"
        elif scheduled.phase is RequestPhase.DECODE:
            start = (
                scheduled.context_length
                - scheduled.num_new_tokens
                - routed_request.prompt_token_count
            )
            end = start + scheduled.num_new_tokens
            phase_tokens = routed_request.decode_tokens
            phase = "decode"
        else:
            raise TypeError(f"{path}.phase: expected RequestPhase")
        if start < 0 or end > len(phase_tokens) or start >= end:
            raise ValueError(
                f"{path}: {phase} token slice [{start}, {end}) is outside "
                f"captured count {len(phase_tokens)}"
            )
        tokens.extend(phase_tokens[start:end])
    if len(tokens) != record.total_new_tokens:
        raise ValueError(
            "record.scheduled: captured token count disagrees with total_new_tokens"
        )
    return tuple(tokens)


def _routed_moe_alltoalls(
    record: StepRecord,
    dims: ModelDims,
    ranks: tuple[int, ...],
    supply: RoutedMoeSupply,
) -> list[MoeAllToAll]:
    if len(ranks) != len(set(ranks)):
        raise ValueError("ep_ranks: contains duplicate ranks")
    if any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 for rank in ranks):
        raise ValueError("ep_ranks: expected distinct nonnegative integer ranks")
    routing = supply.routed_experts
    if routing.expert_count != dims.num_experts:
        raise ValueError(
            "routed_experts.expert_count: disagrees with model num_experts"
        )
    if routing.top_k != dims.top_k:
        raise ValueError("routed_experts.top_k: disagrees with model top_k")
    expected_layers = tuple(range(dims.num_layers))
    if routing.moe_layer_indices != expected_layers:
        raise ValueError(
            "routed_experts.moe_layer_indices: disagree with model layers"
        )
    placement = supply.placement_for_step(record.step_index)
    owners = placement.owner_map()
    expected_keys = {
        (layer, expert)
        for layer in expected_layers
        for expert in range(dims.num_experts)
    }
    owner_keys = set(owners)
    missing = expected_keys - owner_keys
    extra = owner_keys - expected_keys
    if missing:
        layer, expert = min(missing)
        raise ValueError(
            f"placement epoch {placement.placement_epoch}: missing owner for "
            f"layer {layer} expert {expert}"
        )
    if extra:
        layer, expert = min(extra)
        raise ValueError(
            f"placement epoch {placement.placement_epoch}: unexpected owner for "
            f"layer {layer} expert {expert}"
        )
    invalid_ranks = set(owners.values()) - set(ranks)
    if invalid_ranks:
        raise ValueError(
            "placement snapshot: owner ranks outside ep_ranks: "
            + ", ".join(str(rank) for rank in sorted(invalid_ranks))
        )

    tokens = _scheduled_routed_tokens(record, supply)
    vector_bytes = dims.hidden_size * dims.dtype_bytes
    operations = []
    for layer in expected_layers:
        send_bytes: dict[tuple[int, int], int] = {}
        for source in ranks:
            for token in tokens:
                layer_routing = token.layers[layer]
                destinations = {
                    owners[(layer, expert)]
                    for expert in layer_routing.expert_ids
                }
                for destination in destinations:
                    if destination == source:
                        continue
                    pair = (source, destination)
                    send_bytes[pair] = send_bytes.get(pair, 0) + vector_bytes
        dispatch = tuple(
            (source, destination, size)
            for (source, destination), size in sorted(send_bytes.items())
        )
        if not dispatch:
            continue
        combine = tuple(
            sorted(
                (destination, source, size)
                for source, destination, size in dispatch
            )
        )
        operations.extend(
            (
                MoeAllToAll(
                    layer=layer,
                    phase="dispatch",
                    ranks=ranks,
                    per_pair_bytes=0,
                    pair_payload_bytes=dispatch,
                    placement_epoch=placement.placement_epoch,
                ),
                MoeAllToAll(
                    layer=layer,
                    phase="combine",
                    ranks=ranks,
                    per_pair_bytes=0,
                    pair_payload_bytes=combine,
                    placement_epoch=placement.placement_epoch,
                ),
            )
        )
    return operations


def step_moe_alltoalls(
    record: StepRecord,
    dims: ModelDims,
    ep_ranks: Sequence[int],
    *,
    routed_supply: RoutedMoeSupply | None = None,
) -> list[MoeAllToAll]:
    """The step's MoE all-to-alls, empty when the step produces none.

    Per MoE layer (every layer of an MoE ``dims``): a dispatch pairwise
    all-to-allv routing each token's activation to its ``top_k`` experts'
    owner ranks, then a combine pairwise all-to-allv returning the expert
    outputs, both over the expert-parallel group ``ep_ranks`` of W ranks.

    With ``routed_supply``, the scheduled request slices select exact captured
    input tokens. Expert ownership at the step's explicit placement epoch maps
    each token to destination ranks. Dispatch deduplicates several selected
    experts on the same destination into one hidden vector, and combine is the
    exact reverse table after owner-side pre-reduction.

    Without a routed supply, the compatibility assumption spreads the
    ``total_new_tokens * top_k`` (token, expert) assignments evenly over
    the W ranks, so each rank sends

        per_pair_bytes = total_new_tokens * top_k * hidden_size
                         * dtype_bytes // W

    to every OTHER rank in both phases; the 1/W share routed to a rank's
    own resident experts stays local and never touches the fabric. The floor
    division remains part of this explicit compatibility path.

    Empty means: dense dims (``num_experts == 0``), an EP world smaller
    than 2, or a zero-new-token drain record. The uniform path also returns
    empty when its per-pair share rounds to zero bytes.
    """
    ranks = tuple(ep_ranks)
    if dims.num_experts <= 0 or len(ranks) < 2:
        return []
    if record.total_new_tokens <= 0:
        return []
    if routed_supply is not None:
        return _routed_moe_alltoalls(record, dims, ranks, routed_supply)
    per_pair = (
        record.total_new_tokens * dims.top_k * dims.hidden_size * dims.dtype_bytes
    ) // len(ranks)
    if per_pair <= 0:
        return []
    return [
        MoeAllToAll(layer=layer, phase=phase, ranks=ranks, per_pair_bytes=per_pair)
        for layer in range(dims.num_layers)
        for phase in MOE_A2A_PHASES
    ]


def render_step_goal(
    record: StepRecord,
    dims: ModelDims,
    tp_ranks: Sequence[int],
    per_layer_calc_ns: int | Sequence[int],
    *,
    ep_ranks: Sequence[int] | None = None,
    routed_supply: RoutedMoeSupply | None = None,
    num_goal_ranks: int | None = None,
    base_tag: int = 1000,
) -> GoalTrace:
    """Render one step as a GOAL program over the TP (and optionally EP) groups.

    Every participating rank executes the serial chain over layers: ``calc``
    of the corresponding ``per_layer_calc_ns`` GOAL units (ns), then the
    layer's attention allreduce, then its MLP allreduce (both only when the TP
    world produces collectives), then, for MoE dims with ``ep_ranks`` given,
    the dispatch and combine all-to-allvs over the EP group; the next layer's
    calc waits for the previous layer's last collective (no overlap, TRAF-7;
    the fixed calc/allreduce/dispatch/combine order is TRAF-9). A scalar calc
    repeats on every layer for compatibility; a sequence supplies unequal
    layer costs. Participants are the TP ranks plus, when MoE all-to-alls
    exist, the EP ranks; other ranks below ``num_goal_ranks`` get one
    zero-cost calc so every rank block is populated.

    Tags: allreduce k (layer * 2 + site index) takes the disjoint block
    ``base_tag + k * 2(W-1)`` onward, one tag per round, exactly as before;
    MoE all-to-allvs take one tag each, ``base_tag + tp_tag_slots + j`` for
    all-to-all j (layer * 2 + phase index), starting right after the
    allreduce blocks. A step without MoE work renders byte-identically to
    the pre-MoE emitter (golden-tested).

    Raises ``ValueError`` when the step has neither TP collectives nor MoE
    all-to-alls (callers decide what "no network work" means; the
    closed-loop sink returns None so the frontend's own compute estimate
    stands).
    """
    tp_ops = step_tp_allreduces(record, dims, tp_ranks)
    moe_ops = step_moe_alltoalls(
        record,
        dims,
        ep_ranks if ep_ranks is not None else [],
        routed_supply=routed_supply,
    )
    if not tp_ops and not moe_ops:
        raise ValueError(
            "step has no tensor-parallel collectives and no MoE all-to-alls "
            "to render (TP world < 2 or zero new tokens, and no expert traffic)"
        )
    if isinstance(per_layer_calc_ns, int):
        layer_calc_ns = (per_layer_calc_ns,) * dims.num_layers
    else:
        layer_calc_ns = tuple(per_layer_calc_ns)
    if len(layer_calc_ns) != dims.num_layers:
        raise ValueError(
            f"received {len(layer_calc_ns)} layer calc values for "
            f"num_layers={dims.num_layers}"
        )
    if any(value < 0 for value in layer_calc_ns):
        raise ValueError("layer calc values must be nonnegative")
    ranks = list(tp_ranks)
    participants = list(ranks)
    if moe_ops:
        for rank in moe_ops[0].ranks:
            if rank not in participants:
                participants.append(rank)
    tag_stride = 2 * (len(ranks) - 1) if tp_ops else 0
    moe_base_tag = base_tag + len(tp_ops) * tag_stride
    moe_by_key = {(op.layer, op.phase): op for op in moe_ops}
    if num_goal_ranks is None:
        num_goal_ranks = max(participants) + 1
    minimum_ranks = max(participants) + 1
    if num_goal_ranks < minimum_ranks:
        raise ValueError(
            f"num_goal_ranks={num_goal_ranks} cannot contain rank {minimum_ranks - 1}"
        )
    trace = GoalTrace(num_goal_ranks)

    previous: dict[int, str] = {}
    for layer in range(dims.num_layers):
        # start of a layer: the per-layer compute, chained to the previous
        # layer's last collective
        calc_done: dict[int, str] = {}
        for rank in participants:
            calc = trace.rank(rank).calc(layer_calc_ns[layer])
            if rank in previous:
                trace.rank(rank).requires(calc, previous[rank])
            calc_done[rank] = calc
        previous = {**previous, **calc_done}
        if tp_ops:
            for site_index in range(len(TP_ALLREDUCE_SITES)):
                op_index = layer * len(TP_ALLREDUCE_SITES) + site_index
                op = tp_ops[op_index]
                done = ring_allreduce(
                    trace,
                    ranks=list(op.ranks),
                    size_bytes=op.payload_bytes,
                    base_tag=base_tag + op_index * tag_stride,
                    after=previous,
                )
                previous = {**previous, **done}
        if moe_ops:
            for phase_index in range(len(MOE_A2A_PHASES)):
                moe_index = layer * len(MOE_A2A_PHASES) + phase_index
                op = moe_by_key.get((layer, MOE_A2A_PHASES[phase_index]))
                if op is None:
                    continue
                if op.pair_payload_bytes:
                    send_bytes = {
                        (source, destination): size
                        for source, destination, size in op.pair_payload_bytes
                    }
                else:
                    send_bytes = {
                        (s, d): op.per_pair_bytes
                        for s in op.ranks
                        for d in op.ranks
                        if s != d
                    }
                done = pairwise_all_to_allv(
                    trace,
                    ranks=list(op.ranks),
                    send_bytes=send_bytes,
                    tag=moe_base_tag + moe_index,
                    after=previous,
                )
                previous = {**previous, **done}

    used = set(participants)
    for rank in range(num_goal_ranks):
        if rank not in used:
            trace.rank(rank).calc(0)
    return trace
