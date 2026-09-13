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
- `RankMapper`: rank to GOAL-rank assignment mirroring the htsim drivers'
  `-goal_rank_mapping` (`gpu-rank` implemented; `unique-nic` is refused by
  the mapper and by the fabric validator, PLACE-2), plus `is_intra_node`. Construction validates and snapshots
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
  ([capture study](../../examples/nccl_topology_capture_v1/RESULTS.md)).
  Remaining: the mapper's NIC selection for a fabric whose NICs are not one
  per GPU (frozen with PLACE-2) and topology discovery for nodes the fixed
  eight-GPU rail profile does not describe; PLACE-9 owns the captured shapes
  the reader refuses and the rendering of a captured NIC inventory. CORE-4's fixed profile
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
  switched boards, PLACE-12 their switch side; PLACE-2 owns `unique-nic`.
- PLACE-2 (Completeness; P2; M): `unique-nic` GOAL-rank mapping (depends on
  PLACE-1). Also deferred behind the fixed eight-GPU, eight-RNIC profile;
  `gpu-rank` and `unique-nic`
  happen to have the same cardinality there, but the general mapper must not
  assume that affinity.

- PLACE-3 (Completeness; P1; M): expert-parallel group memberships and
  declared expert ownership in `declared_manifest`. The builder emits tp/pp/dp
  groups only, so the M5 MoE studies pass an explicit `ep_ranks` list to
  `HtsimStepSink` instead of reading an EP group from a manifest; a declared
  EP layout (group lists plus per-layer `local_num_experts` ownership) would
  close the gap, and the extracted manifest's per-MoE-layer expert IDs already
  model the live half.
  P1 since 2026-09-07: the expert-parallel tail studies read the EP group from
  the manifest.
