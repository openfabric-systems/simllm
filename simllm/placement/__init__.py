"""Placement and fabric mapping: where logical ranks physically live.

Serving frameworks are *topology-light*, not topology-agnostic. Each layer of
the stack knows a different slice of the truth:

=========================  ====================================================
Layer                      What it knows
=========================  ====================================================
Model-parallel framework   Logical ranks, TP/PP/DP/EP groups, weight/expert
                           partitioning
Executor / launcher        Which global rank runs on which host and local GPU
NCCL / cluster runtime     PCIe/NVLink/NIC topology, chosen communication path
                           and collective algorithm
Network simulator          Switches, links, routing, queueing, congestion
                           control
=========================  ====================================================

SimLLM therefore joins two independent descriptions:

- a **placement manifest** (:mod:`simllm.placement.manifest`):
  global rank → node → GPU → model shard → process groups, including
  per-layer local expert ownership and the expert-placement epoch;
- a **fabric topology manifest** (schema pinned in
  :mod:`simllm.placement.manifest`): GPU to PCIe/NVLink to NIC to switch to
  link graph.

The **mapper** (:mod:`simllm.placement.mapper`) resolves every rank in a
collective to a physical endpoint and assigns GOAL ranks for the network
backend.
"""

from simllm.placement.declared import declared_manifest
from simllm.placement.disaggregated import (
    DECLARED_CLOS_ENDPOINTS_PER_LEAF,
    DECLARED_CLOS_EVIDENCE_CLASS,
    DECLARED_CLOS_LINK_PROPAGATION_DELAY_PS,
    DECLARED_CLOS_LINK_RATE_BPS,
    DECLARED_CLOS_SPINE_SWITCHES,
    DECLARED_CLOS_SWITCH_LATENCY_PS,
    DECLARED_CLOS_TOPOLOGY_NAME,
    DisaggregatedDeploymentManifests,
    disaggregated_manifests,
)
from simllm.placement.manifest import (
    FABRIC_SCHEMA,
    PLACEMENT_SCHEMA,
    FabricLink,
    FabricNodePlacement,
    FabricSwitchPlacement,
    FabricSwitchPort,
    FabricTopologyManifest,
    GpuFabricPlacement,
    GroupMembership,
    NicFabricPlacement,
    PlacementManifest,
    RankPlacement,
)
from simllm.placement.mapper import RankMapper
from simllm.placement.sglang_disaggregated import (
    SglangPoolArrangement,
    sglang_disaggregated_manifests,
)

__all__ = [
    "DECLARED_CLOS_ENDPOINTS_PER_LEAF",
    "DECLARED_CLOS_EVIDENCE_CLASS",
    "DECLARED_CLOS_LINK_PROPAGATION_DELAY_PS",
    "DECLARED_CLOS_LINK_RATE_BPS",
    "DECLARED_CLOS_SPINE_SWITCHES",
    "DECLARED_CLOS_SWITCH_LATENCY_PS",
    "DECLARED_CLOS_TOPOLOGY_NAME",
    "FABRIC_SCHEMA",
    "PLACEMENT_SCHEMA",
    "DisaggregatedDeploymentManifests",
    "FabricLink",
    "FabricNodePlacement",
    "FabricSwitchPlacement",
    "FabricSwitchPort",
    "FabricTopologyManifest",
    "GpuFabricPlacement",
    "GroupMembership",
    "NicFabricPlacement",
    "PlacementManifest",
    "RankMapper",
    "RankPlacement",
    "SglangPoolArrangement",
    "declared_manifest",
    "disaggregated_manifests",
    "sglang_disaggregated_manifests",
]
