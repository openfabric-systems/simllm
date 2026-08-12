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

from simllm.traffic.collective_plan import (
    DEFAULT_COLLECTIVE_PLAN_BASE_TAG,
    collective_plan_by_operation,
    plan_execution_graph_collectives,
    render_collective_plan,
)
from simllm.traffic.execution_goal import (
    ExecutionGoalArtifact,
    ExecutionGoalArtifactBoundary,
    ExecutionGoalProjection,
    project_execution_graph_goal,
    render_serial_execution_graph_goal,
    verify_execution_goal_projection,
)
from simllm.traffic.locality import (
    DEFAULT_NVLINK_BANDWIDTH_BYTES_PER_SECOND,
    ClassifiedCommunicationPhase,
    CollectiveCommunicationPhase,
    DirectedCollectiveSegment,
    StepLocalityPlan,
    classify_step_locality,
)
from simllm.traffic.patterns import (
    binomial_broadcast,
    gather,
    ordered_pairwise_messages,
    pairwise_all_to_allv,
    ring_allreduce,
    scatter,
)
from simllm.traffic.request_fidelity import (
    RequestFidelityError,
    RequestFidelityReport,
    compare_goal_request_attribution,
)
from simllm.traffic.routed_moe import (
    ExpertPlacementSnapshot,
    RoutedMoeSupply,
    validate_expert_placement_snapshot,
    validate_routed_moe_supply,
)
from simllm.traffic.step_comm import (
    MOE_A2A_PHASES,
    TP_ALLREDUCE_SITES,
    MoeAllToAll,
    MoeMessageGrouping,
    MoeMessageSequence,
    RoutedMoeMessage,
    TpAllReduce,
    compare_request_moe_fidelity,
    lower_step_observations,
    plan_execution_graph_locality,
    plan_step_locality,
    render_fabric_phase_goal,
    render_sequenced_step_goal,
    render_step_goal,
    step_communication_phases,
    step_moe_alltoalls,
    step_moe_message_sequences,
    step_tp_allreduces,
    validate_execution_graph_locality_projection,
    validate_request_moe_fidelity,
)

__all__ = [
    "COLLECTIVE_TRACE_SCHEMA",
    "DEFAULT_COLLECTIVE_PLAN_BASE_TAG",
    "DEFAULT_NVLINK_BANDWIDTH_BYTES_PER_SECOND",
    "MOE_A2A_PHASES",
    "TP_ALLREDUCE_SITES",
    "ClassifiedCommunicationPhase",
    "CollectiveCommunicationPhase",
    "DirectedCollectiveSegment",
    "ExecutionGoalArtifact",
    "ExecutionGoalArtifactBoundary",
    "ExecutionGoalProjection",
    "ExpertPlacementSnapshot",
    "MoeAllToAll",
    "MoeMessageGrouping",
    "MoeMessageSequence",
    "RequestFidelityError",
    "RequestFidelityReport",
    "RoutedMoeMessage",
    "RoutedMoeSupply",
    "StepLocalityPlan",
    "TpAllReduce",
    "binomial_broadcast",
    "classify_step_locality",
    "collective_plan_by_operation",
    "compare_goal_request_attribution",
    "compare_request_moe_fidelity",
    "gather",
    "lower_step_observations",
    "ordered_pairwise_messages",
    "pairwise_all_to_allv",
    "plan_execution_graph_collectives",
    "plan_execution_graph_locality",
    "plan_step_locality",
    "project_execution_graph_goal",
    "render_collective_plan",
    "render_fabric_phase_goal",
    "render_sequenced_step_goal",
    "render_serial_execution_graph_goal",
    "render_step_goal",
    "ring_allreduce",
    "scatter",
    "step_communication_phases",
    "step_moe_alltoalls",
    "step_moe_message_sequences",
    "step_tp_allreduces",
    "validate_execution_graph_locality_projection",
    "validate_expert_placement_snapshot",
    "validate_request_moe_fidelity",
    "validate_routed_moe_supply",
    "verify_execution_goal_projection",
]
