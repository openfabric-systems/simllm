"""Traffic model: semantic collectives → physical flows.

Consumes three inputs and produces the flow-level work the GOAL emitter
renders:

1. **Collective trace** (schema ``simllm-collective-trace-v1``, JSONL): one
   record per communication op, i.e.
   ``{step, layer, op, group_type, group_global_ranks, send_counts,
   element_bytes, hidden_size, placement_epoch, release_time_ns}``
   with ``op`` in ALL_REDUCE / ALL_GATHER / REDUCE_SCATTER / ALL_TO_ALLV /
   SEND_RECV (PP), plus KV-transfer records.
2. **Placement manifest**: resolves group members to nodes/GPUs; for MoE,
   ``expert_owners[layer][global_expert_id]`` turns routed tokens into
   all-to-allv destinations at the correct ``placement_epoch``.
3. **Fabric manifest**: NIC selection, intra- vs inter-node split.

Semantic collectives are then expanded into the algorithm actually used
(ring, tree, pairwise all-to-allv, or a custom collective-network schedule)
as chunked send/recv chains, so the network simulator sees real traffic
patterns rather than abstract ops.
"""

COLLECTIVE_TRACE_SCHEMA = "simllm-collective-trace-v1"

from simllm.traffic.patterns import (
    binomial_broadcast,
    gather,
    pairwise_all_to_allv,
    ring_allreduce,
    scatter,
)
from simllm.traffic.step_comm import (
    TP_ALLREDUCE_SITES,
    TpAllReduce,
    render_step_goal,
    step_tp_allreduces,
)

__all__ = [
    "COLLECTIVE_TRACE_SCHEMA",
    "TP_ALLREDUCE_SITES",
    "TpAllReduce",
    "binomial_broadcast",
    "gather",
    "pairwise_all_to_allv",
    "render_step_goal",
    "ring_allreduce",
    "scatter",
    "step_tp_allreduces",
]
