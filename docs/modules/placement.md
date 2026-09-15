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
  the framework's own expert map. `round_robin` is the caller's declaration,
  and `DeclaredExpertMapExceptions`, passed as
  `DeclaredExpertLayout(..., exceptions=...)`, is how a layout says the
  framework would overrule it: `single_expert_group`, `redundant_experts`,
  `eplb` and `all2all_without_round_robin` resolve a declared `round_robin` to
  `linear` exactly as the pinned resolver does, `eplb` additionally emits the
  `eplb` group that shares the `ep` group's ranks, and
  `model_uniform_expert_blocks` and `model_refuses_tp_above_experts` carry the
  two model-scoped rules as refusals. `declared_resolved_placement_strategy`
  reports the resolution; the manifest gains no strategy field, because
  ownership already records the map that runs.
  `DeclaredExpertLayout(expert_parallel=False)` states the deployment vLLM runs
  with expert parallelism off: the `ep` group is unchanged and every rank of a
  stage owns every expert of that stage's MoE layers, the geometry an extracted
  manifest of such a run records. The tensor shard of one expert is not
  represented. Omitting `experts` is the explicit off path and keeps every
  expert-free manifest byte identical; omitting the two selections keeps every
  expert-bearing manifest byte identical.
- `declared_sglang_manifest(tp=..., pp=..., ep_size=..., moe_dp_size=...,
  nodes=..., gpus_per_node=..., hostname_pattern=..., framework_version=...,
  experts=...)` and `declared_sglang_pipeline_partition(num_layers, pp)`: the
  same declared statement under the pinned SGLang layout rules, emitting
  `source="declared"` and `framework="sglang"`. The world is `tp x pp` with no
  data-parallel term in the rank space (`global_rank = tp * pp_rank + tp_rank`,
  attention data parallelism living inside the tensor group and router-style
  replicas being separate worlds), so every rank carries `tp`, `pp` and a
  singleton `dp` membership. With `experts` present it also carries the `ep`
  group carved out of one tensor group per pipeline stage, with the MoE tensor
  index innermost and the MoE data-parallel index outermost, and the `moe_tp`
  and `moe_dp` groups that go with it at `moe_tp_size = tp // ep_size //
  moe_dp_size`. Ownership is the contiguous block of `num_experts // ep_size`
  experts, the only expert map the pinned framework builds, and the partition
  hands the layer remainder to the *last* stages where the vLLM rule hands it
  to the stages indexed `-2, -3, ...`. `DeclaredExpertLayout` is shared with
  the vLLM builder, and `round_robin`, an expert count `ep_size` does not
  divide, a node count the launcher could not produce, `expert_parallel=False`
  (SGLang spells that `ep_size=1`) and a non-default `exceptions` object (the
  pinned SGLang map has no `round_robin` to fall back from) are refused. The two
  builders are separate entry points and never share an output: every manifest
  the vLLM builder emits is byte identical to its pre-change bytes.
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
- `dgx_peer_fabric("a100" | "h100" | "b200", node_id=..., ranks=..., ... )`:
  the eight-GPU HGX/DGX peer inventory, with explicit timing and buffer
  inputs. A100 has six switch chips and 96 links; H100 has four chips and
  144 links; B200 has two chips and 144 links at the NVLink 5 payload rate of
  400 Gbit/s per lane. Chip bundles are declared from NVIDIA's Fabric Manager
  User Guide; a capture confirms a GPU's lane count and all-pair reachability,
  never the per-chip split. Port identities are stable logical attachments
  and lane-matched routing is declared. The
  [DGX study](../../examples/dgx_nvlink_v1/RESULTS.md) exercises A100 and
  H100 through packet service and request latency, and the
  [HGX B200 capture study](../../examples/hgx_b200_capture_v1/RESULTS.md)
  binds the GPU side of a rented B200 board to the `b200` generation.
- `NcclTopologyDump.load(path)` and `captured_fabric_node(dump, node_id=...,
  pool_role=..., global_rank_by_gpu_dev=..., nvlink_propagation_delay_ps=...)`:
  the strict reader for NCCL's `NCCL_TOPO_DUMP_FILE` system dump (version 1)
  and the join of one captured node into the existing schemas: a
  `FabricNodePlacement` whose GPUs each take the one NIC under their own NUMA
  node, and a direct-mesh `PeerFabric` wired with the bonded NVLink count the
  dump states. Reading is fail closed; a NUMA node with zero or several NICs,
  a shared NIC or a partial mesh is refused rather than approximated. The
  [capture study](../../examples/nccl_topology_capture_v1/RESULTS.md) proves
  a captured four-A100 node substitutes exactly for its literal twin. The
  reader also accepts the switched-board shape (GPUs behind PCIe switches,
  a socket NIC repeated under every NUMA node, and bridge-class NVLink rows
  naming the virtual fabric address), and `captured_switched_node(dump,
  generation=..., module_id_by_gpu_dev=..., ...)` joins such a board to the
  preset of its generation, placing each GPU at the slot its captured module
  id names and declaring the NICs absent. With `switch_ports_by_gpu_dev`,
  read from the `nvidia-smi nvlink -R` block by the parsers in
  `simllm.placement.nvidia_smi_inventory`, the join also binds the preset's
  chips and switch ports to the captured switches and ports (`dgx_peer_fabric`
  takes `switch_ids` and `switch_port_ids` for that), refusing a board whose
  per-switch lane counts differ from the generation's bundle.
- `RankMapper(placement, mode="gpu-rank" | "unique-nic", fabric=...)`: rank
  to GOAL-rank assignment mirroring the htsim drivers' `-goal_rank_mapping`.
  `gpu-rank` is the identity; `unique-nic` reads each GPU's affine NIC from
  the fabric manifest and assigns one GOAL rank per NIC in fabric order (node
  order, then NIC order within a node), so GPUs behind one NIC share one
  fabric endpoint while `is_intra_node` keeps their local traffic off the
  fabric. `nic_of(rank)` answers the affinity in either mode. Construction
  validates and snapshots a unique global-rank-to-host projection with
  nonblank hostnames and unique local GPU endpoints, joins the fabric (every
  GPU names one NIC on its own node, the GPU set equals the rank set), so an
  active traffic run cannot silently change its locality or endpoint
  authority.
- `declared_shared_nic_fabric(placement, gpus_per_nic=...)`: a fabric for a
  declared placement with `gpus_per_node // gpus_per_nic` NICs per node and
  GPU `local_rank` `l` affine to NIC `l // gpus_per_nic`; one GPU per NIC is
  the identity mapping. The fabric renderer gives segments that collapse onto
  one GOAL endpoint pair distinct tags by `tag * M + j` (`M` the largest
  collapse in the step, the identity at `M = 1`) and carries a projection
  table that joins every backend completion row to its semantic segment.
  `HtsimStepSinkConfig(goal_rank_mapping="unique-nic", fabric_manifest=...)`
  selects the mode on the null-network profiles; the
  [unique-NIC study](../../examples/unique_nic_mapping_v1/RESULTS.md) shows
  the fluid closed form scaling exactly with the GPUs per NIC.
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
no manifest schema change.

The `unique-nic` projection is landed and validated by the
[unique-NIC study](../../examples/unique_nic_mapping_v1/RESULTS.md): with
one NIC per GPU the mode is byte identical to `gpu-rank`, a reversed NIC
order permutes GOAL ranks with identical results, and two or four GPUs behind
one NIC give per-flow serialization times exactly two or four times the
one-per-NIC value after subtracting fixed propagation, with every completion row joined
to its semantic segment through the projection table. The seven reference
manifests and the six m5 makespans stay exact under the default mapping.

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

A declared manifest can follow either pinned framework. The
[SGLang declared layout study](../../examples/sglang_declared_layout_v1/RESULTS.md)
validates the second builder: the installed package's own `get_pp_indices`
agrees with `declared_sglang_pipeline_partition` on all six frozen rows, the
`tp=8, ep_size=8` manifest equals its vLLM twin field for field outside the
framework name and the two MoE side groups, expert sharding and MoE data
parallelism place every EP, `moe_tp` and `moe_dp` group on its frozen ranks, a
16-rank four-stage pipeline conserves all 3,712 ownership entries, and the EP
group read from `declared_sglang_manifest(tp=W, pp=1, ep_size=W, experts=...)`
reproduces all six frozen m5 check-B makespans exactly with byte-identical
GOAL text. The five vLLM reference manifests stay byte identical and the
PLACE-3 study still reproduces its tracked results.

A declared layout also states the two deployments where the framework
overrules the caller. The
[declared expert variant study](../../examples/declared_expert_variants_v1/RESULTS.md)
validates both: with `expert_parallel=False` every rank of a stage keeps its
`ep` membership and owns every expert of that stage's MoE layers, conserving
12,288 ownership entries at eight owners per pair in the worked example and
collapsing to the accepted bytes at a flattened width of one; and each of the
four framework conditions of `DeclaredExpertMapExceptions` resolves a declared
`round_robin` to `linear` on the frozen rows, with `eplb` also emitting the
group that shares the `ep` ranks, while a declared `linear` never moves. The
SGLang builder refuses both selections, all eight rejection controls refuse,
the five reference manifests stay byte identical, and the PLACE-3 and PLACE-13
studies both still reproduce their tracked results. The all-owner geometry is
representable but reaches no consumer yet, which PLACE-8 records.

The disaggregated builder supplies the one-prefill plus one-decode placement
used by the live CORE-51 session and the same fixed structure at 16 prefill
plus 40 decode nodes. The target contains exactly 448 unique ranks, GPUs and
NICs with role counts of 128 prefill and 320 decode. Its declared physical
graph has 56 leaf switches, eight spines, 896 links and 1,344 switch ports.
All 448 endpoints are reachable, all 448 cross-leaf GOAL witness messages
resolve to four-link paths, and disabling physical rendering reproduces both
pre-change placement records byte for byte. See the
[PLACE-5 result](../../examples/disaggregated_target_topology_v1/RESULTS.md).

A captured node joins the same schemas: the
[NCCL topology capture study](../../examples/nccl_topology_capture_v1/RESULTS.md)
reads one Perlmutter A100 node's dump (four GPUs in an NV4 mesh, four
Slingshot NICs, one per NUMA node), reproduces every frozen inventory literal,
cross-checks it against the node's `nvidia-smi` and `lspci` records, and
drives the live peer packet path with results equal to a literal twin on
every step, with the eight reference artifacts byte identical.

A rented HGX B200 board binds the GPU side of the switched preset family:
the [HGX B200 capture study](../../examples/hgx_b200_capture_v1/RESULTS.md)
reads its switched dump, confirms eighteen active lanes per GPU, `NV18` on
all 56 pairs, the PCIe pairing and NUMA split, eight distinct board ids and
the module-id slot order, and runs the DGX study's live cells for the `b200`
generation with Python and native agreement on every cell. The switch side
of that board is not observable from a rented container (every remote
device is the virtual fabric address and no NVSwitch PCI device is visible),
which is what PLACE-12 records.

A rented HGX H200 board, whose container passes its four NVSwitches
through, binds the H100 generation on both sides: the
[HGX H200 switch capture study](../../examples/hgx_h200_switch_capture_v1/RESULTS.md)
confirms the declared `(4, 5, 5, 4)` bundle on every GPU, maps all 144
preset switch ports onto captured switch ports, shows every route path
joining two ports of one captured switch, binds the GPU side with module-id
slots, and reproduces every live DGX cell identically with the bound fabric.

## Open tasks

### Completeness

- PLACE-6 (Completeness; P1; M): bind the shipped HGX/DGX A100 and H100
  peer presets to a captured eight-GPU system's GPU-side inventory.
  `dgx_peer_fabric` supplies the public attachment counts and all 56 ordered
  rank pairs for A100, H100 and B200; the
  [DGX study](../../examples/dgx_nvlink_v1/RESULTS.md) validates software
  routing and shared capacities, and the
  [HGX B200 capture study](../../examples/hgx_b200_capture_v1/RESULTS.md)
  binds the B200 generation's GPU side to a rented board: lane counts, the
  all-pair matrix, board identity, PCIe placement and the module-id slot
  order agree, through `captured_switched_node`, and the
  [HGX H200 switch capture study](../../examples/hgx_h200_switch_capture_v1/RESULTS.md)
  binds the H100 generation's GPU side the same way on a rented H200 board
  (the HGX H100 baseboard). Remaining: the same GPU-side binding for an A100
  board (no rentable A100 board exposed active NVLink; a bare-metal HGX A100
  rental is pending) and NIC affinity on switched boards (the join declares
  NICs absent and refuses a board exposing a GPU Direct RDMA NIC; the H200
  inventory shows one ConnectX virtual function per PCIe switch but NCCL's
  dump carries no RDMA row). Preserve host and network
  attachments as separate identities: HGX is the GPU baseboard/platform and
  DGX the complete server. The archived public A100 link table has an
  apparent `233` switch-port typo, so do not silently turn it into
  register-exact topology or infer device order from rank number.
  Acceptance for this task's remaining half: captured GPU/board identity,
  NVLink status and NCCL topology agree on all 56 pairs for one A100 board,
  with the module-id slot binding, and absent capture selection preserves
  current manifests exactly. The switch-side clause (each path
  joins two ports of one switch, switch identities) is PLACE-12. Keep
  Merlin's four-A100 NV4 and four-GH200 NV6 direct meshes separate from the
  eight-GPU switched presets. TRAF-92 owns product timing calibration;
  BACK-74 owns further native switch state. This remains P1 for realistic
  one-node parallelism; broader discovery remains PLACE-1.
- PLACE-12 (Completeness; P1; L): switch-side binding of the eight-GPU
  switched presets. A rented container under NVIDIA Fabric Manager hides
  the NVSwitches: every NVLink remote device reads as the virtual fabric
  address, no bridge-class PCI device is visible, and NCCL collapses the
  fabric to one target with the lane count, so no capture from such a host
  can say which switch port a lane lands on. This half needs a host that
  passes its NVSwitch devices through to the tenant. A rented HGX H200
  container did, and the
  [HGX H200 switch capture study](../../examples/hgx_h200_switch_capture_v1/RESULTS.md)
  binds the H100 generation's switch side: the declared `(4, 5, 5, 4)`
  bundle equals the captured per-switch lanes on every GPU, the 144 preset
  switch ports map bijectively onto the captured `(switch, port)` pairs,
  every path joins two ports of one captured switch, and the bound fabric
  gives identical live results to the declared preset. Remaining: the same
  binding for an A100 board (six switches, two lanes per GPU per switch)
  and a B200 board whose host exposes its two switches. Acceptance for each:
  `nvidia-smi nvlink -R` remote bus ids, the bridge-class PCI devices and
  NCCL's per-switch NVLink rows agree with the preset's per-chip bundle on
  all 56 pair paths, each path joining two ports of one switch, with
  capacities conserving full-duplex links, and absent capture selection
  preserves current manifests exactly. The live B200 cells equal
  the H100 cells at equal lane rate because the native crossbar model
  carries no per-chip capacity term; pricing the chip partition is BACK-74's
  model work, which this capture would calibrate.

- PLACE-1 (Completeness; P1; L): general fabric topology contents and NIC
  selection beyond the landed slices. Landed: the rail-optimized and
  node-local leaf variants of the reference Clos (`declared_rail_fabric`,
  TRAF-88's opt-in) and the NCCL topology dump reader with the captured-node
  join for one NIC per NUMA node
  ([capture study](../../examples/nccl_topology_capture_v1/RESULTS.md)), plus
  fabric-backed NIC selection and `unique-nic` mapping for declared shared-NIC
  nodes ([unique-NIC study](../../examples/unique_nic_mapping_v1/RESULTS.md)).
  Remaining: topology discovery for nodes the fixed eight-GPU rail profile
  does not describe; PLACE-9 owns the captured shapes the reader refuses and
  the rendering of a captured NIC inventory, and PLACE-10 owns the remaining
  shared-NIC execution paths. CORE-4's fixed profile
  (eight GPUs, one WQE queue per GPU, eight GPU-affine 400G RNICs, intra-node
  transfers on an NVLink-class path) needs none of this and stays the
  default.
- PLACE-9 (Completeness; P2; M): captured-node shapes the NCCL topology
  reader refuses. `captured_fabric_node` accepts one NIC under each GPU's
  NUMA node and a complete NVLink mesh, and `captured_switched_node` accepts
  a PCIe-switched board whose GPUs are switch-attached with a socket NIC; a
  NUMA node with several NICs or none, a NIC shared by several GPUs, a
  partial mesh, and a switched board exposing a GPU Direct RDMA NIC are
  refused fail closed, and the direct-mesh peer domain is named
  `<node_id>:nv4` for any lane count. Add explicit selections for those
  shapes and a lane-count-derived domain name, and render the captured NIC
  inventory (200 Gbit/s Slingshot ports on the Perlmutter fixture) into a
  physical fabric graph. Absent selection must keep every current manifest
  and study artifact byte identical. PLACE-6 owns the GPU-side capture of
  switched boards, PLACE-12 their switch side; PLACE-10 owns the remaining
  `unique-nic` execution paths.
- PLACE-10 (Completeness; P2; M): the `unique-nic` seams the first slice
  refuses. `HtsimStepSinkConfig` refuses `goal_rank_mapping="unique-nic"`
  together with `peer_packet`, `flow_session`, `dependency_cross_check`, a
  physical `topology` file or `num_goal_ranks` below the NIC count, because
  the physical Clos projection assumes 64 GPU-affine endpoints and the
  session, peer and cross-check paths key their rows by GPU rank. The
  completion join through the projection table is enforced under
  `unique-nic` only: typed backend doubles and the LogGOPSim sink return
  rows that are not one per GOAL message, so `gpu-rank` publishes no join
  outcome. The step sink's expert-parallel traffic is one-engine (uniform
  routing dispatches from the first EP rank), so a full-population all-pairs
  phase can be rendered only through the traffic layer today. Add explicit
  selections for each: the projection carried through the flow session and
  cross-check joins, a physical projection for non-64 endpoint counts, the
  join enforced under `gpu-rank` once every backend double returns one row
  per message, and a full-population expert-parallel option threaded through
  the lowerer and the sink. Absent selection must keep every current artifact
  byte identical. CORE-14 owns the coarse runtime's affinity.

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
  same path. The all-owner geometry that `expert_parallel=False` now declares
  belongs to this task as well: the
  [variant study](../../examples/declared_expert_variants_v1/RESULTS.md) found
  that `ExpertPlacementSnapshot.from_manifest` refuses it outright, with
  `snapshot.expert_owners: expert has multiple owners`, so a manifest in which
  every rank of a stage owns every expert is representable but reaches no
  consumer. Add an explicit uneven-ownership path and an all-owner projection;
  the absence of both must keep every current artifact byte identical.

- PLACE-14 (Completeness; P2; M): the SGLang layout variants the declared
  builder leaves out. `declared_sglang_manifest` states one replica of a
  `tp x pp` world with `--ep-size` and `--moe-dp-size`, and everything else
  the pinned framework can do is fixed at one, zero, trivial or absent:
  attention data parallelism and attention context parallelism inside the
  tensor group, decode context parallelism, the `SGLANG_PP_LAYER_PARTITION`
  environment override, fused shared experts and redundant physical expert
  copies, expert-parallel load balancing (EPLB) maps other than the trivial
  identity, the elastic expert-parallel joiner offset, and the `--base-gpu-id`
  and `--gpu-id-step` launcher options (the GPU a rank lands on is
  `base_gpu_id + (pp_rank % pp_per_node) * tp_per_node +
  (tp_rank % tp_per_node) * gpu_id_step`, so the emitted `local_rank` equals
  the launcher's `gpu_id` only at the defaults 0 and 1 the builder
  states). The unrepresented
  tensor shard under `moe_tp > 1`, where every rank of one MoE tensor group
  owns the same expert ids and holds one shard of each, is the same gap
  PLACE-15 now records for vLLM and is settled with it. Add an explicit selection
  for each, so the layout that opts in says so; the absence of any new
  selection must keep every current manifest byte identical, which the five
  vLLM reference digests and the
  [SGLang study](../../examples/sglang_declared_layout_v1/RESULTS.md) rows
  both lock. SGL-18 owns the extracted counterpart of these same mechanisms.

- PLACE-15 (Completeness; P2; M): the declared MoE dimensions that carry no
  manifest field. Three gaps survive PLACE-7 and PLACE-11, all of them
  representational rather than defects. The tensor shard of one expert: with
  expert parallelism off, vLLM cuts every expert's `intermediate_size` into
  `DP x TP` shards
  (`model_executor/layers/fused_moe/config.py`, `FusedMoEConfig.__post_init__`),
  and with `moe_tp > 1` SGLang does the same inside a MoE tensor group, but
  `local_expert_ids` states which global expert ids live on a rank and has no
  way to say which slice of one expert's weights does. The redundant physical
  experts EPLB creates: `DeclaredExpertMapExceptions.redundant_experts`
  declares the condition that overrules a `round_robin` map, while the extra
  physical copies, their logical-to-physical map and its rebalancing epochs
  are not represented at all. And the widths behind
  `all2all_without_round_robin`: the declared layout folds
  `use_all2all_kernels and not needs_round_robin_routing_tables` into one
  boolean instead of stating a sequence-parallel width, a prefill
  context-parallel width and an all-to-all backend name. Add a field for each,
  so a layout that opts in says so; the absence of every new field must keep
  the five reference manifest digests and the
  [variant study](../../examples/declared_expert_variants_v1/RESULTS.md) rows
  byte identical. PLACE-8 owns the consumers of uneven and all-owner
  ownership; PLACE-14 owns the SGLang side of the same shard question.
