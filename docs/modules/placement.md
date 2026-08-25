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
- Fabric topology manifest (`simllm-fabric-topology-v1`): GPU to PCIe/NVLink
  to NIC to switch to link graph. Schema name pinned in `manifest.py`;
  contents land with M4.
- `RankMapper`: rank to GOAL-rank assignment mirroring the htsim drivers'
  `-goal_rank_mapping` (`gpu-rank` implemented; `unique-nic` needs the
  fabric manifest), plus `is_intra_node`. Construction validates and snapshots
  a unique global-rank-to-host projection with nonblank hostnames and unique
  local GPU endpoints, so an active traffic run cannot silently change its
  locality authority.
- `disaggregated_manifests(prefill_nodes=..., decode_nodes=...,
  gpus_per_node=...)`: builds the paired placement and fabric projections for
  the fixed prefill/decode deployment. Every rank carries its pool role and a
  deterministic GPU-affine NIC projection through the existing schema. The
  builder validates the one-plus-one live slice and the 16-plus-40 structural
  target without inventing a second manifest format.

## Status

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
`declared_manifest(tp=8).group_ranks(0, "tp")`. Fabric manifest and NIC
selection are design-only.

TRAF-10 now consumes this existing placement authority directly: collective
segments are classified by semantic global rank before fabric GOAL-rank
projection, and no locality field is copied into the execution graph. The
captured locality study covers one-node, two-node and all-remote placements;
see [the results](../../examples/nvlink_locality_v1/RESULTS.md). This required
no manifest schema change. General `unique-nic` projection remains PLACE-2.

The first PLACE-4 slice builds the one-prefill plus one-decode placement used
by the live CORE-51 session and structurally renders the same builder at 16
prefill plus 40 decode nodes. The target contains exactly 448 unique ranks,
GPUs and NIC projections with role counts of 128 prefill and 320 decode ranks.
The deterministic location labels are simulated projections rather than a
complete physical switch and link graph, so the physical target half remains
PLACE-5 and PLACE-4 stays open.

## Open tasks

### Completeness

- PLACE-4 (Completeness; P1; M): build the 448-rank disaggregated placement:
  placement and fabric manifests for 40 decode plus 16 prefill nodes of
  eight GPUs and one NIC each, with each rank's pool role carried in the
  manifest, every GPU and NIC pinned to a physical fabric location, and
  resolution through the existing manifest schemas and GOAL-rank mapping.
  Smaller instances of the same shape (one plus one node first) come from
  the same builder. The general-manifest halves stay PLACE-1 and PLACE-2;
  this task owns the concrete disaggregated target and its role field. The
  role-aware one-plus-one live manifest and 448-rank structural render are
  delivered. PLACE-5 owns the remaining physical-location half, so this task
  stays open until that projection is literal.
- PLACE-5 (Completeness; P1; M): replace the disaggregated target builder's
  deterministic simulated location labels with the complete fixed target
  topology. Bind every GPU-affine NIC to its port, switch, link rate and
  propagation delay through `simllm-fabric-topology-v1`, then validate the
  same authority through GOAL rendering. Acceptance requires exact rank, GPU,
  NIC and role conservation at one-plus-one and 16-plus-40, complete endpoint
  reachability, and byte-identical placement records when physical rendering
  is disabled. General inventory discovery and `unique-nic` mapping remain
  PLACE-1 and PLACE-2.
- PLACE-1 (Completeness; P2; L): fabric topology schema contents and general
  NIC selection in the mapper, sourcing intra-node structure from NCCL
  topology dumps. This is no longer blocked: CORE-4 validated the first
  fixed resource profile of eight GPUs per node, one WQE submission queue or
  QP per GPU, all
  eight feeding their GPU-affine 400G RNICs, with intra-node transfers on an
  NVLink-class path. The fixed rail profile does not need general inventory
  discovery.
- PLACE-2 (Completeness; P2; M): `unique-nic` GOAL-rank mapping (depends on
  PLACE-1). Also deferred behind the fixed eight-GPU, eight-RNIC profile;
  `gpu-rank` and `unique-nic`
  happen to have the same cardinality there, but the general mapper must not
  assume that affinity.

### Uncategorized

- PLACE-3: expert-parallel group memberships and declared expert ownership
  in `declared_manifest`. The builder emits tp/pp/dp groups only, so the
  M5 MoE studies pass an explicit `ep_ranks` list to `HtsimStepSink`
  instead of reading an EP group from a manifest; a declared EP layout
  (group lists plus per-layer `local_num_experts` ownership) would close
  the gap, and the extracted manifest's per-MoE-layer expert IDs already
  model the live half.
