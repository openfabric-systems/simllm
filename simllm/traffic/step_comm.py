"""Per-step tensor-parallel collective work derived from a step record.

Maps one :class:`~simllm.core.StepRecord` plus a per-rank
:class:`~simllm.compute.ModelDims` plus a tensor-parallel group (a list of
GOAL ranks) to the TP collective traffic of that engine step: per
transformer layer, two ring allreduces (the attention output projection and
the MLP output projection), each moving

    payload_bytes = record.total_new_tokens * dims.hidden_size * dims.dtype_bytes

which is the activation tensor of every token computed this step. A TP
world of size 1 (or a drain record with zero new tokens) produces no
operations.

This is a deliberately first-order model of TP traffic; each simplification
is a numbered task in docs/modules/traffic.md:

- no sequence parallelism (which would replace each allreduce with a
  reduce-scatter plus allgather of 1/W the bytes framed around the
  norm/dropout regions): TRAF-6;
- no communication/compute overlap (each layer's compute and its two
  allreduces are a strict serial chain): TRAF-7;
- no pipeline-parallel activation traffic (records carry no PP stage
  information yet): TRAF-8.

Rendering reuses :func:`simllm.traffic.patterns.ring_allreduce` unchanged,
so the GOAL structure (2(W-1) chained rounds of S/W chunks) is exactly the
pattern validated by the M1/M4 studies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from simllm.compute import ModelDims
from simllm.core import StepRecord
from simllm.goal import GoalTrace
from simllm.traffic.patterns import ring_allreduce

#: the two allreduce sites of one transformer layer, in execution order
TP_ALLREDUCE_SITES = ("attention", "mlp")


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


def render_step_goal(
    record: StepRecord,
    dims: ModelDims,
    tp_ranks: Sequence[int],
    per_layer_calc_ns: int,
    *,
    num_goal_ranks: int | None = None,
    base_tag: int = 1000,
) -> GoalTrace:
    """Render one step as a GOAL program over the TP group.

    Every rank executes the serial chain over layers: ``calc`` of
    ``per_layer_calc_ns`` GOAL units (ns), then the layer's attention
    allreduce, then its MLP allreduce; the next layer's calc waits for the
    previous layer's last allreduce round (no overlap, TRAF-7). Tags are
    disjoint per allreduce: operation k (layer * 2 + site index) uses
    ``base_tag + k * 2(W-1)`` onward, one tag per round. Ranks outside the
    TP group get one zero-cost calc so every rank block is populated.

    Raises ``ValueError`` when the step has no TP collectives (callers
    decide what "no network work" means; the closed-loop sink returns None
    so the frontend's own compute estimate stands).
    """
    ops = step_tp_allreduces(record, dims, tp_ranks)
    if not ops:
        raise ValueError(
            "step has no tensor-parallel collectives to render "
            "(TP world < 2 or zero new tokens)"
        )
    ranks = list(tp_ranks)
    world = len(ranks)
    tag_stride = 2 * (world - 1)
    if num_goal_ranks is None:
        num_goal_ranks = max(ranks) + 1
    trace = GoalTrace(num_goal_ranks)

    previous: dict[int, str] = {}
    for op_index, op in enumerate(ops):
        if op.site == TP_ALLREDUCE_SITES[0]:
            # start of a layer: the per-layer compute, chained to the
            # previous layer's last allreduce round
            calc_done: dict[int, str] = {}
            for rank in ranks:
                calc = trace.rank(rank).calc(per_layer_calc_ns)
                if rank in previous:
                    trace.rank(rank).requires(calc, previous[rank])
                calc_done[rank] = calc
            previous = calc_done
        previous = ring_allreduce(
            trace,
            ranks=ranks,
            size_bytes=op.payload_bytes,
            base_tag=base_tag + op_index * tag_stride,
            after=previous,
        )

    used = set(ranks)
    for rank in range(num_goal_ranks):
        if rank not in used:
            trace.rank(rank).calc(0)
    return trace
