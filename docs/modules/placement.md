# simllm.placement

The mapper: where logical ranks physically live. Serving frameworks are
topology-light (they know ranks and groups, not the fabric), so SimLLM joins
two independent descriptions and resolves every communication event through
both.

## Interface

- Placement manifest (`simllm-placement-manifest-v1`): per global rank the
  host, local rank, GPU UUID / PCI bus ID, group memberships with the actual
  global-rank lists, pipeline layer range, per-MoE-layer local expert IDs
  and the EPLB `placement_epoch`. `PlacementManifest` loads/saves JSON and
  answers `by_rank` / `group_ranks`. Manifests are either declared (what-if)
  or extracted from a live run; both share one schema.
- `declared_manifest(tp=..., pp=..., dp=..., nodes=..., gpus_per_node=...,
  hostname_pattern=...)`: builds a `source="declared"` manifest with
  tp/pp/dp memberships in the DP x PP x TP layout order
  (`global_rank = (dp*PP + pp)*TP + tp`, TP innermost). A declared
  manifest is a what-if placement *computed from the layout formula*,
  which is exactly what the extraction rule forbids for live runs and
  permits here: a live run must export the actual group lists because the
  manifest records what really happened, while a declared deployment has
  no live groups to ask and the formula is its specification. The
  `source` field keeps the two kinds distinguishable forever.
- `DeclaredExpertLayout(num_layers=..., moe_layers=..., num_experts=...,
  placement_strategy="linear" | "round_robin", placement_epoch=...)`, passed
  as `declared_manifest(..., experts=...)`: the mixture-of-experts (MoE) half
  of the same what-if statement, following the pinned vLLM 0.27.1 rules.
  Every rank gains the expert-parallel `ep` group of its pipeline stage (the
  DP x TP ranks, data-parallel major, `rank_in_group = dp * TP + tp`), the
  `[start, end)` layer range of the pinned pipeline partition, and the global
  expert ids it owns in each MoE layer of that range under the selected
  expert map; `declared_pipeline_partition` and `declared_local_expert_ids`
  expose the two rules on their own. When `DP x TP` does not divide the
  expert count, each EP rank below the remainder owns one extra expert, as in
  the framework's own expert map. `round_robin` is the caller's declaration;
  vLLM's own fallback to `linear` is not modeled, and an extracted manifest
  records the map that really ran. Omitting `experts`
  is the explicit off path and keeps every expert-free manifest byte
  identical.
- Fabric topology manifest (`simllm-fabric-topology-v1`): GPU to PCIe/NVLink
  to NIC to switch to link graph. `FabricTopologyManifest` round-trips node
  inventory, switch ports and physical links, validates one termination per
  NIC and port, proves endpoint connectivity and resolves structured GOAL
  messages to deterministic shortest paths.
- The optional `peer_fabrics` inventory extends the same fabric schema with
  `PeerFabric`, `PeerPortPlacement` and `PeerRoute`. Each GPU or switch port
  terminates exactly one physical attachment. A directed rank pair resolves to
  declared one-hop direct or two-hop switched paths; opposite full-duplex
  directions have independent link resources. Shared attachments and finite
  switch input storage remain shared across destinations and virtual channels.
  A GPU belongs to one timing domain. `to_dict()` and `save()` omit an empty
  peer inventory, preserving every accepted absent-peer serialized artifact.
- `dgx_peer_fabric("a100" | "h100", node_id=..., ranks=..., ... )`: the
  eight-GPU HGX/DGX peer inventory, with explicit timing and buffer inputs.
  A100 has six switch chips and 96 links; H100 has four chips and 144 links.
  Port identities are stable logical attachments and lane-matched routing is
  declared. The [DGX study](../../examples/dgx_nvlink_v1/RESULTS.md) exercises
  both through packet service and request latency.
- `RankMapper`: rank to GOAL-rank assignment mirroring the htsim drivers'
  `-goal_rank_mapping` (`gpu-rank` implemented; `unique-nic` needs the
  fabric manifest), plus `is_intra_node`. Construction validates and snapshots
  a unique global-rank-to-host projection with nonblank hostnames and unique
  local GPU endpoints, so an active traffic run cannot silently change its
  locality authority.
- `disaggregated_manifests(prefill_nodes=..., decode_nodes=...,
  gpus_per_node=..., render_physical_topology=...)`: builds the paired
  placement and fabric projections for the fixed prefill/decode deployment.
  Every rank carries its pool role and one GPU-affine NIC through the existing
  schemas. Physical rendering supplies the declared two-tier 400 Gbit/s Clos
  with explicit 1,000,000 ps link delays; disabling it preserves the accepted
  placement bytes and emits no physical graph.

## Status

Explicit local attachments feed the [live peer packet study](../../examples/local_peer_packet_runtime_v1/RESULTS.md) through
the existing manifest contract. The direct and switched route inventory agrees
with semantic placement before packet admission. Legacy rail topology and
arrival-study writers use the canonical manifest projection, retaining the
accepted bytes when peer packet service is absent.

Manifest round-trip and the gpu-rank mapper are implemented and tested
(including the DP=2 x PP=2 x TP=4 worked example). The extraction path is
implemented for vLLM: `simllm.adapters.vllm.PlacementExporter` is a worker
extension class whose one RPC returns this rank's entry, and
`manifest_from_worker_entries` assembles the per-worker dicts into a
`source="extracted"` manifest with the framework version recorded (see
[adapters-vllm](adapters-vllm.md)). The declared builder landed with the
M4 first slice (tested against the same DP=2 x PP=2 x TP=4 worked example,
exact group lists) and closes the placement half of VLLM-7; the M4 studies
and the live tp=8 closed-loop run drive `HtsimStepSink` off
`declared_manifest(tp=8).group_ranks(0, "tp")`.

TRAF-10 now consumes this existing placement authority directly: collective
segments are classified by semantic global rank before fabric GOAL-rank
projection, and no locality field is copied into the execution graph. The
captured locality study covers one-node, two-node and all-remote placements;
see [the results](../../examples/nvlink_locality_v1/RESULTS.md). This required
no manifest schema change. General `unique-nic` projection remains PLACE-2.

The declared EP layout is landed and validated by the
[declared expert placement study](../../examples/declared_expert_placement_v1/RESULTS.md):
the EP group read from `declared_manifest(tp=1, dp=W, experts=...)`
reproduces all six frozen m5 check-B makespans exactly with byte-identical
GOAL text against the hand-typed rank list, the worked example's round-robin
ownership matches the manifest tests' original hand-written row, a 64-rank
DeepSeek-class pipeline partitions 256 experts over four 16-rank EP groups
with every `(layer, expert)` pair owned exactly once, and the five reference
expert-free manifests stay byte identical. Expert-parallel studies can
read `manifest.group_ranks(rank, "ep")` and build
`ExpertPlacementSnapshot.from_manifest` from declared ownership.

The disaggregated builder supplies the one-prefill plus one-decode placement
used by the live CORE-51 session and the same fixed structure at 16 prefill
plus 40 decode nodes. The target contains exactly 448 unique ranks, GPUs and
NICs with role counts of 128 prefill and 320 decode. Its declared physical
graph has 56 leaf switches, eight spines, 896 links and 1,344 switch ports.
All 448 endpoints are reachable, all 448 cross-leaf GOAL witness messages
resolve to four-link paths, and disabling physical rendering reproduces both
pre-change placement records byte for byte. See the
[PLACE-5 result](../../examples/disaggregated_target_topology_v1/RESULTS.md).

## Open tasks

### Completeness

- PLACE-6 (Completeness; P1; M): bind the shipped HGX/DGX A100 and H100
  peer presets to a captured eight-GPU system's physical device and port
  inventory. `dgx_peer_fabric` supplies the public attachment counts and all
  56 ordered rank pairs, with A100's six parallel chips and H100's four;
  the [DGX study](../../examples/dgx_nvlink_v1/RESULTS.md) validates software
  routing and shared capacities. Remaining work is the mapping from the
  preset's logical ports to GPU module IDs, active NVLink indices, switch
  identities and actual routing evidence from a qualified capture. Preserve
  host and network attachments as separate identities: HGX is the GPU
  baseboard/platform and DGX the complete server. The archived public A100
  link table has an apparent `233` switch-port typo, so do not silently
  turn it into register-exact topology or infer device order from rank number.
  Acceptance: captured GPU/board identity, NVLink status and NVIDIA Collective
  Communications Library topology agree on all 56 pair paths; each path joins
  two ports of one switch; capacities conserve full-duplex physical links;
  and absent capture selection preserves current manifests exactly.
  Keep Merlin's four-A100 NV4 and four-GH200 NV6 direct meshes separate from
  the eight-GPU switched presets. Two four-GPU nodes are not one switched
  allocation. TRAF-92 owns product timing calibration; BACK-74 owns further
  native switch state. This remains P1 for realistic one-node parallelism;
  broader discovery remains PLACE-1.

- PLACE-1 (Completeness; P1; L): fabric topology schema contents and general
  NIC selection in the mapper, sourcing intra-node structure from NCCL
  topology dumps. This is no longer blocked: CORE-4 validated the first
  fixed resource profile of eight GPUs per node, one WQE submission queue or
  QP per GPU, all
  eight feeding their GPU-affine 400G RNICs, with intra-node transfers on an
  NVLink-class path. The fixed rail profile does not need general inventory
  discovery.
  P1 since 2026-09-07: TRAF-88's fabric variants opt in; the first slice is
  the rail-optimized and node-local leaf variants of the reference Clos under
  the existing `simllm-fabric-topology-v1` schema.
- PLACE-2 (Completeness; P2; M): `unique-nic` GOAL-rank mapping (depends on
  PLACE-1). Also deferred behind the fixed eight-GPU, eight-RNIC profile;
  `gpu-rank` and `unique-nic`
  happen to have the same cardinality there, but the general mapper must not
  assume that affinity.

- PLACE-7 (Completeness; P2; S): declared MoE ownership with expert
  parallelism disabled. vLLM 0.27.1 still creates the `ep` group for a MoE
  model when expert parallelism is off, but then tensor-shards every expert
  across the flattened DP x TP group so each rank owns every expert of its
  stage's MoE layers. Today that deployment is declared by omitting `experts`, which is
  the accepted all-experts-local geometry but records no `ep` group, so an
  extracted manifest from such a run and its declared counterpart differ in
  group inventory. Add an explicit selection that emits the group with
  all-expert ownership; omitting it must keep every current manifest byte
  identical, and the enabled variant must not change any step metric of an
  expert-free run.

- PLACE-8 (Completeness; P2; S): uneven per-rank expert ownership in the
  step-sink consumers. A declared layout whose expert count `DP x TP` does
  not divide is representable in the manifest, with the remainder experts on
  the lowest EP ranks, but its consumers assume uniform per-rank expert
  geometry: the vLLM step schedule refuses it
  (`local_num_experts * len(ep_ranks) == num_experts`), and `HtsimStepSink`
  carries one `ModelDims.local_num_experts` for every EP rank. The
  calibration side shares the assumption: `simllm/calibration/extraction.py`
  refuses a MoE communication case whose participant count does not divide
  the expert count before floor-dividing it, and the memory estimate in
  `simllm/calibration/external_db.py` floor-divides `num_experts` by
  `moe_ep_size`, dropping remainder experts silently, so both belong to the
  same path. Add an explicit uneven-ownership path; its absence must keep
  every current artifact byte identical.

- PLACE-11 (Completeness; P2; S): framework-side expert map exceptions the
  declared layout does not model. vLLM 0.27.1 falls back from `round_robin`
  to `linear` when the model has at most one expert group, has redundant
  experts, runs EPLB, or uses an all-to-all backend without round-robin
  routing tables, and some model implementations refuse expert counts their
  expert-parallel size does not divide. `DeclaredExpertLayout` emits the
  caller's declared strategy and the remainder rule as stated; add an explicit
  model-scoped selection that applies those exceptions, whose absence keeps
  every current manifest byte identical. An extracted manifest records the map
  that really ran.
