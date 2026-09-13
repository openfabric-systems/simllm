# HGX B200 capture expectations

Date: 2026-09-13

This is the expectations-only freeze for the first PLACE-6 slice: binding
the GPU side of an eight-GPU switched preset to a captured board. It
precedes the `b200` preset generation, the reader's switched-dump shape, the
captured switched-node join, their tests, the study harness and every
result-producing run. The capture was made before this freeze and is
tracked under `tests/fixtures/nccl_topology/vastai_hgx_b200_8x/` with its
digests in that directory's `PROVENANCE.md`; it is input evidence, not a
result. The registered PLACE-6 acceptance has two halves, the GPU side and
the switch side; this slice qualifies the GPU side and states in plain words
why the switch side is not observable from the captured host.

## Question

Can the shipped eight-GPU switched preset family gain a `b200` generation
whose GPU-side facts (links per GPU, ordered-pair path counts, per-link
rate, board and device identity, PCIe placement) equal what a real HGX B200
board reports through `nvidia-smi` and NCCL's topology dump, with the
reader accepting the switched dump shape fail closed, the live DGX study
path running the new generation, and every accepted A100 and H100 artifact
unchanged?

## Frozen source and compatibility identity

The implementation starts from commit
`64c7f5923fc0fe0142d537a3bd4a054ee43ca02c`, the tip of the PLACE-1 slice
(`claude/place1_nccl_topology`), because it extends that slice's reader. The
JSON registry records the pre-change SHA-256 identities of the DGX preset,
reader, peer topology, package entry, NVLink runtime, DGX study harness and
results, the two test modules, and the five fixture files.

The compatibility authority is the canonical JSON of the A100 and H100 peer
fabrics and the DGX study's tracked results:

| Record | Bytes | SHA-256 |
|---|---:|---|
| `dgx_peer_fabric("a100", node_id="node-0", ranks=(0..7), propagation_delay_ps=1000, switch_input_buffer_bytes=65536)` as sorted compact JSON | 84,797 | `47d66b66acd5a477f757ee30400256d39569df5a5fef5aea1655a937fc9fa219` |
| the same for `"h100"` | 125,741 | `4429ed2ada4ee4e7dc7180dbade0b0be5b60b0fc234918f7e307b5c0214d7278` |
| `examples/dgx_nvlink_v1/results.json` | tracked | `00d935fc4b26cfc50aa4c5f6b46ba20e3db856d091215c2e616b095fb577fdba` |

The A100 preset has 96 links, 192 ports, 56 routes of 12 paths; the H100
preset has 144 links, 288 ports, 56 routes of 18 paths. Both stay byte
identical, and the DGX study's `--check` mode must still reproduce its
tracked results with the A100 and H100 grid unchanged. These are fatal,
unscored guards.

## Captured board, stated before the implementation exists

- Eight NVIDIA B200 (`0x10de:0x2901`, `sm` 100), one board part number
  `692-2G525-0220-500`, eight distinct Board IDs, at bus ids `51`, `52`,
  `62`, `63`, `75`, `76`, `86`, `87`, two behind each of four PCIe switches
  (class `0x060400`, vendor `0x1000`) at `45`, `56`, `67`, `7a`, on NUMA nodes
  0, 0, 0, 0, 1, 1, 1, 1.
- Eighteen active NVLinks per GPU at 53.125 GB/s each in `nvidia-smi nvlink
  -s`; `NV18` on all 56 ordered pairs in `nvidia-smi topo -m`; every link's
  remote device is `FFFFFFFF:FF:FF.0`, no class `0x0680` PCI device is
  visible, no `Module ID` is exposed.
- NCCL 2.27.3 dump: one `<nvlink target="fffffff:ff:ff.0" count="18"
  tclass="0x068000"/>` row per GPU; the socket NIC `eth0` (10,000 megabit,
  `gdr="0"`, `guid="0x0"`, `port="0"`) listed once under each of the two
  `<cpu>` elements as a `<nic>` child of `<cpu>`; GPUs as `<pci>` children of
  the PCIe-switch `<pci>` elements.
- NCCL graph search: 16 ring channels and 16 tree channels of type NVL, 8
  NVLS channels.

## Declared preset

`dgx_peer_fabric("b200", ...)` adds the third generation to the existing
entry point. Its public wiring comes from NVIDIA's Fabric Manager User
Guide (the NVSwitch systems section, cited in the JSON registry): two
fourth-generation NVSwitches per eight-GPU HGX B200 baseboard with nine
NVLink 5 links from every GPU to each, declared as the bundle `(9, 9)`. The capture confirms the eighteen links per GPU and the all-pair
`NV18`; it cannot confirm the per-chip split, so the split is declared and
cited, exactly as the A100 `2 x 6` and H100 `4 + 5 + 5 + 4` bundles are.
The per-link payload rate is declared as 400 Gbit/s per direction (NVIDIA's
1.8 TB/s bidirectional per GPU over eighteen links gives 50 GB/s per link
per direction); the captured 53.125 GB/s is the signalling rate and is
recorded, not used, following the TRAF-44 rule that raw signalling rates
overstate payload. The preset therefore has 144 links, 288 ports and 56
routes of 18 paths, matching the H100 counts with a different chip
partition.

## Declared reader and join

`NcclTopologyDump.load` accepts, in addition to the PLACE-1 shape:

- `<pci>` nested under `<pci>` when the outer element has class `0x060400`
  (a PCIe switch); the GPU's PCIe location records the switch bus id;
- `<nic>` directly under `<cpu>`;
- an `<nvlink>` row whose `tclass` is `0x068000` and whose `target` is the
  virtual fabric address `fffffff:ff:ff.0`, which marks the GPU as
  switch-attached with `count` lanes; a GPU mixing switch-attached and
  peer-attached rows is refused;
- one socket `<net>` repeated under several `<cpu>` elements when every
  repetition is attribute-identical; differing repetitions are refused.

`captured_switched_node(dump, *, generation, node_id, pool_role,
global_rank_by_gpu_dev, propagation_delay_ps, switch_input_buffer_bytes)`
joins a switch-attached dump to the preset of the named generation: it
requires every GPU switch-attached with the preset's lanes per GPU (18 for
`b200`), builds the `FabricNodePlacement` with declared-absent NICs when the
dump carries no GPU Direct RDMA NIC (`nic_id` `"<node_id>:nic-absent-<dev>"`,
`nics=()`, so `unique-nic` mapping refuses it), and returns the preset's
`PeerFabric` for the captured ranks with `source="extracted"` on the fabric
manifest. A dump whose lane count differs from the generation's, or whose
GPU count is not eight, is refused.

## Frozen cells

Cell G1, parse: the fixture parses to 2 CPUs, 4 PCIe switches, 8 GPUs, 8
switch-attached NVLink rows of count 18, and 1 distinct socket NIC seen
twice; every literal in the captured-board section is exact.

Cell G2, preset: `dgx_peer_fabric("b200", ...)` has 2 chips, 144 links, 288
ports, 56 routes, 18 paths per route, 9 paths per route per chip, every link
at 400,000,000,000 bit/s, and validates inside a fabric manifest; its
canonical JSON digest is recorded by the run and pinned by the tests
thereafter.

Cell G3, GPU-side binding: for the captured board joined to the `b200`
preset, the preset's lanes per GPU (18) equal `nvidia-smi nvlink -s`'s
active link count for all eight GPUs; the preset's paths per ordered pair
(18) equal the `NV18` matrix on all 56 pairs and the dump's `count` on all
eight rows; the join's device order and bus ids equal the `nvidia-smi -L`
and `-q` order; the PCIe placement puts GPU pairs (0, 1), (2, 3), (4, 5),
(6, 7) behind one switch each and the NUMA split is 4 and 4; and the eight
Board IDs are distinct with one part number. The switch-side clause of
PLACE-6 (each path joins two ports of one switch, switch identities) is
recorded as not observable, with the exact evidence (virtual remote device,
absent `0x0680` devices, absent Module ID), and is not scored.

Cell G4, live: the DGX study's live cells at generation `b200` (widths 2,
4, 8 for tensor-parallel all-reduce and expert dispatch and combine, both
attachment rates 12.5 and 25 GB/s per lane, both Python and native switch
selection) run through the same harness functions with the same fixed
compute provider. Frozen relations: Python and native agree exactly on
request metrics and packet observations in every `b200` cell; every
rate pair changes completion time in at least one transport-limited cell
while compute stays identical; and, at equal lane rate, buffers and
policy, the `b200` and `h100` generations give identical request metrics
on every isolated single-pair component cell (same 18 lanes per pair, the
chip partition does not enter an uncontended path) and `b200` step latency
at or above `h100` on the fan-in and live cells (two chips concentrate
more contending inputs per crossbar than four; equality is allowed).

Cell G5, refusals: a dump with 17 lanes on one GPU against the `b200`
generation; a dump with seven GPUs; a GPU mixing a switch-attached row and
a peer-attached row; a socket NIC repeated with differing attributes; a
nested `<pci>` whose outer element is not class `0x060400`; and a request
for generation `"h100"` against the eighteen-lane B200 dump succeeding
(same lane count) while `"a100"` (twelve lanes) is refused.

Cell G6, wire identity and off path: the joined fabric manifest round-trips
through `save` and `load`; the A100 and H100 preset digests are unchanged;
the DGX study's `--check` reproduces its tracked results.

## Physical sanity before observation

Eighteen links at 400 Gbit/s payload give 900 GB/s per GPU per direction,
NVIDIA's published per-direction figure for NVLink 5; the captured 53.125
GB/s signalling per link gives 956 GB/s, 6.25 percent above payload, the
same ratio TRAF-44 recorded for NVLink 4. A tensor-parallel exchange on the
`b200` preset cannot serialize a rank's egress faster than 900 GB/s, nor a
pair faster than 900 GB/s; the G4 rows are read against those bounds first.

## Evidence accounting and closure

G4's Python-versus-native identity is an exact-oracle family, the rate
relation and the `b200` versus `h100` relations are behavioral instances;
G1, G2, G3 and G6 are structural exact guards; G5 is a rejection control
family; the three compatibility digests and the DGX `--check` are fatal
by-construction identities. Counts are never added. A violated fatal guard
voids the run.

This slice does not close PLACE-6. If every cell holds, PLACE-6's registry
text narrows to the switch side, with the container observation recorded
as the reason it is blocked on bare-metal access, and the `b200` generation
joins the preset family. PLACE-9 narrows by the switched-dump shape this
slice lands. The switch-side binding is registered as a new task with the
bare-metal requirement stated. TRAF-92 keeps product timing calibration.
