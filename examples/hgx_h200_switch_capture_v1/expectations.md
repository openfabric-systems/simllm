# HGX H200 switch capture expectations

Date: 2026-09-13

This is the expectations-only freeze for the switch-side binding of the
eight-GPU switched preset (PLACE-12) and the H100-generation half of
PLACE-6's GPU-side clause. It precedes the reader's per-switch rows, the
switch-side join, their tests, the study harness and every result-producing
run. The capture was made before this freeze and is tracked under
`tests/fixtures/nccl_topology/vastai_hgx_h200_8x/` with its digests in that
directory's `PROVENANCE.md`; it is input evidence, not a result.

## Question

Can the `h100` generation of the switched preset be bound, switch side
included, to a real HGX H200 board (the same NVLink 4 baseboard as HGX
H100): does the preset's declared per-chip bundle equal the captured
per-switch lane counts, does every preset switch port land on one distinct
captured switch port, does every route path join two ports of one captured
switch, and does the binding leave every request metric and every accepted
artifact unchanged?

## Frozen source and compatibility identity

The implementation starts from commit
`af5c0110ae74f69afa69cfec0b42281e123b3170`, the tip of the B200 slice
(`claude/place6_b200_gpu_side`), because it extends that slice's reader and
join. The JSON registry records the pre-change SHA-256 identities of the
DGX preset, reader, peer topology, package entry, NVLink runtime, the
B200 and DGX study results, the test modules and the five fixture files.

The compatibility authority is the canonical JSON of the three presets and
the two switched studies' tracked results:

| Record | SHA-256 |
|---|---|
| A100 peer fabric JSON (as in the B200 freeze) | `47d66b66acd5a477f757ee30400256d39569df5a5fef5aea1655a937fc9fa219` |
| H100 peer fabric JSON | `4429ed2ada4ee4e7dc7180dbade0b0be5b60b0fc234918f7e307b5c0214d7278` |
| B200 peer fabric JSON | the value the B200 tests pin |
| `examples/dgx_nvlink_v1/results.json` | `00d935fc4b26cfc50aa4c5f6b46ba20e3db856d091215c2e616b095fb577fdba` |
| `examples/hgx_b200_capture_v1/results.json` | the tracked file on the base commit |

All five stay identical, and both studies' `--check` modes reproduce.
These are fatal, unscored guards.

## Captured board, stated before the implementation exists

- Eight H200 (`0x10de:0x2335`, `sm` 90), part number `695-2G520-0280-001`,
  at bus ids `83`, `8b`, `93`, `9b`, `a3`, `ab`, `b3`, `bb`, one behind each
  of eight PCIe switches (`0x104c:0x8232`) at `81`, `89`, `91`, `99`, `a1`,
  `a9`, `b1`, `b9`, NUMA 0, 0, 0, 0, 1, 1, 1, 1; module ids 2, 4, 1, 3, 7,
  5, 6, 8 for device order 0 through 7, so slots 1, 3, 0, 2, 6, 4, 5, 7.
- Four NVSwitches (`0x10de:0x22a3`, class `0x068000`) at `c3`, `c4`, `c5`,
  `c6`.
- Eighteen active links per GPU at 26.562 GB/s signalling; `NV18` on all 56
  ordered pairs; per GPU 4 links to `c3`, 5 to `c4`, 5 to `c5`, 4 to `c6`;
  144 distinct captured `(switch, port)` pairs, 32 on `c3`, 40 on `c4`, 40
  on `c5`, 32 on `c6`.
- NCCL 2.27.3 dump: four `<nvlink>` rows per GPU of class `0x068000` naming
  the four switch bus ids with counts 4, 5, 5, 4; the socket NIC `eth0`
  once under each `<cpu>`; GPUs `gdr="1"`. Eight ConnectX virtual-function
  NICs sit one per PCIe switch in the inventory but not in the dump, and the
  run used the socket network.

## Declared reader and join

`NcclTopologyDump.load` accepts several switch-attached `<nvlink>` rows per
GPU whose targets are real bus ids of class `0x068000`; a GPU mixing a
virtual-address row with real-switch rows is refused; the row order in the
dump is preserved.

The harness and tests read three inventory blocks with dedicated parsers
that never guess: the `nvidia-smi nvlink -R` block into a per-GPU, per-link
`(switch bus id, switch port)` table; the class `0x0680` device list; and
the `Module Id` lines.

`captured_switched_node(..., switch_ports_by_gpu_dev=...)` gains the
switch-side binding: the preset's chips, in bundle order, are bound to the
captured switches in ascending bus-id order, and the binding is refused
unless every GPU's captured lane count on the `s`-th switch equals the
preset bundle's `s`-th width (for `h100`: 4, 5, 5, 4 on `c3`, `c4`, `c5`,
`c6`). Preset lane `k` of a GPU on chip `s` is bound to the `k`-th captured
link of that GPU landing on that switch, in link-index order. The returned
peer fabric names its switch ports by the captured identity,
`"<domain>:switch-<busid>:port-<n>"`, and its switches by
`"<domain>:switch-<busid>"`; GPU-side port and link identities keep the
slot-based names. The `PeerFabric` gets `evidence_class="declared"` still
(its timing inputs are declared), and the fabric manifest `source="extracted"`.
Without `switch_ports_by_gpu_dev` the join behaves exactly as the B200 slice
left it.

## Frozen cells

Cell H1, parse: the fixture parses to 2 CPUs, 8 PCIe switches, 8 GPUs, 32
switch-attached rows (counts 4, 5, 5, 4 per GPU in bus-id order `c3`, `c4`,
`c5`, `c6`), 1 distinct socket NIC seen twice, with every literal above
exact.

Cell H2, inventory: the `-R` parser yields 144 rows, 18 per GPU, per-switch
counts 4, 5, 5, 4 for all eight GPUs, 144 distinct `(switch, port)` pairs
with 32, 40, 40, 32 ports per switch; the `0x0680` list yields exactly the
four bus ids; the module ids yield slots 1, 3, 0, 2, 6, 4, 5, 7.

Cell H3, switch-side binding to `h100`: the preset bundle `(4, 5, 5, 4)`
equals the captured per-switch lane counts on every GPU; the 144 preset
switch ports map onto the 144 captured `(switch, port)` pairs bijectively;
every one of the 56 routes' 18 paths joins two captured ports of one switch
and the chip index of that switch equals the preset chip of the path; the
per-switch port totals conserve 32, 40, 40, 32 full-duplex links; and the
declared H100 bundle is confirmed by capture on this board.

Cell H4, GPU side for `h100` (the B200 slice's G3 on this board): 18 lanes
per GPU equal 18 active links; `NV18` equals 18 paths and the dump's
per-GPU total 18; device order, bus ids, one GPU per PCIe switch and the
NUMA split 4 and 4 match; the eight Board IDs (`0x8300`, `0x8b00`, `0x9300`,
`0x9b00`, `0xa300`, `0xab00`, `0xb300`, `0xbb00`) are distinct under one part
number; module ids give slots 1, 3, 0, 2, 6, 4, 5, 7 carried by every
GPU-side port and link.

Cell H5, live identity: the DGX study's live cells at generation `h100`,
run through the same helpers with the captured switch-bound fabric in place
of the declared preset, give request metrics, packet observations and
Python-versus-native agreement identical to the declared preset on every
cell: the binding changes identities, never capacities.

Cell H6, refusals: a GPU with 5 lanes on `c3` against the `h100` bundle; a
captured switch port used by two links; a GPU with one virtual-address row
beside real-switch rows; only three switches in the `0x0680` list; the
`b200` generation against this dump (two chips against four switches); a
`-R` table with 17 links for one GPU.

Cell H7, wire identity and off path: the switch-bound fabric manifest
round-trips through `save` and `load`; the three preset digests and both
study results are unchanged; both studies' `--check` reproduce.

## Physical sanity before observation

Eighteen lanes at 200 Gbit/s payload give 450 GB/s per GPU per direction,
NVIDIA's NVLink 4 figure; the captured 26.562 GB/s signalling gives 478 GB/s,
6.25 percent above payload, as TRAF-44 recorded. On the H5 cells every phase
sits above its lane serialization floor, and the identity with the declared
preset means no number moves.

## Evidence accounting and closure

H5's identity is the scored exact-oracle family; H1 through H4 and H7 are
structural exact guards; H6 is a rejection control family; the five
compatibility identities are fatal by-construction guards. Counts are never
added. A violated fatal guard voids the run.

If every cell holds: PLACE-12 closes for the H100 generation and narrows to
the A100 and B200 boards (its registry text keeps the passthrough
requirement, now with the H200 host as the existence proof); PLACE-6's H100
GPU-side clause becomes literal, leaving the A100 board and NIC affinity on
switched boards, for which this capture supplies the inventory (one
ConnectX virtual function per PCIe switch) but no dump row, registered as a
narrowed PLACE-6 clause rather than a new task. TRAF-92 and BACK-74 are
untouched.
