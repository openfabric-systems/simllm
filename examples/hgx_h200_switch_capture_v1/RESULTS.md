# HGX H200 switch capture result

## Outcome

What ran: `examples/hgx_h200_switch_capture_v1`, the frozen switch-side
slice, read the rented HGX H200 board's NCCL dump and `nvidia-smi`
inventory, bound the `h100` generation of the switched preset to the
board's four NVSwitches and 144 switch ports, and ran the DGX study's live
`h100` cells with the bound fabric in place of the declared preset.
Expectations-only commit `3d24b2b1`, implementation `7eeec912`, harness
`21aaf463`.

What came out: the result is `PASS` with no finding. The deciding rows are
H3: the preset's declared bundle `(4, 5, 5, 4)` equals the captured lane
counts on switches `c3`, `c4`, `c5`, `c6` for all eight GPUs; the 144 preset
switch ports map bijectively onto the 144 captured `(switch, port)` pairs;
every one of the 56 routes' 18 paths joins two captured ports of one switch
whose chip index equals the preset chip; and the per-switch link totals 32,
40, 40, 32 conserve. H5 held on all 18 live cells: request metrics, ordered
causal packet logs and Python-versus-native agreement are identical between
the declared preset and the bound fabric, so the binding changes identities
and rank-indexed fields (126 port rank fields and 49 route rank pairs move
with the module-id slots) and never a capacity or timing field. All five
compatibility identities held, and the DGX and B200 studies' checks
reproduced.

What it changes for the project: PLACE-12 is literal for the H100
generation, so the declared H100 bundle is now a captured one on this
board, and PLACE-12 narrows to the A100 and B200 boards. PLACE-6's H100
GPU-side clause is literal (lanes, `NV18`, order, PCIe, NUMA, board ids,
module-id slots 1, 3, 0, 2, 6, 4, 5, 7), leaving the A100 board and NIC
affinity on switched boards. The switch side turned out to need a host that
passes its NVSwitches through, not bare metal as such: this rented
container did.

What it does not change: no timing is calibrated (TRAF-92); the A100, H100
and B200 preset digests are byte identical; the DGX study's frozen grid is
untouched; NIC affinity on switched boards stays unmodeled (eight ConnectX
virtual functions sit one per PCIe switch in the inventory but NCCL's dump
carries no RDMA row); the A100 switch side has no capture yet (every
rentable A100 board so far had inactive NVLinks; a bare-metal HGX A100
rental is pending an account limit).

## Evidence

| Evidence class | Result |
|---|---|
| Exact-oracle family (H5 bound versus declared) | identical on 18 live cells, Python equals native for declared, fully bound and switch-only fabrics |
| Structural exact guards (H1, H2, H3, H4, H7) | 5 of 5 cells exact |
| Rejection controls (H6) | 12 of 12 refused (the six frozen controls plus the six the amendment added) |
| Fatal compatibility identities | 5 of 5 identical; both study checks reproduce |

Counts in different evidence classes are not added. The tracked
[results](results.json) hold every row.

H1 parses 2 CPUs, 8 PCIe switches, 8 GPUs, 32 switch-attached rows with
counts 4, 5, 5, 4 per GPU in switch bus-id order, and one socket NIC seen
twice. H2's inventory parsers yield 144 remote rows, 18 per GPU, the four
class `0x0680` devices and module ids giving slots 1, 3, 0, 2, 6, 4, 5, 7.
H4 binds the GPU side as the B200 slice did: 18 lanes, `NV18` on 56 pairs,
one GPU per PCIe switch, NUMA split 4 and 4, eight distinct Board IDs under
one part number. H5's tensor-parallel and expert cells span job completion
times from 268,080 to 2,241,240 ps, every one identical between the two
fabrics.

## Physical sanity

Eighteen lanes at 200 Gbit/s payload give 450 GB/s per GPU per direction,
NVIDIA's NVLink 4 figure; the captured 26.562 GB/s signalling per lane gives
478 GB/s, 6.25 percent above payload, as TRAF-44 recorded. The live cells
swept 12.5 and 25 GB/s per lane, below both, and every phase sits above its
lane serialization floor.

## Choices the freeze left open

Two snapshot families are compared as sets, and buffer ownership is sorted
by buffer id, because slot-ordered identities reorder numeric port indices;
a switch-ports-only arm is exact except for that ownership order. The join
cross-checks the inventory table against the dump's switch rows, so a table
supplied against a virtual-address dump is refused. `dgx_peer_fabric` gained
`switch_ids` and `switch_port_ids`, taken together or not at all, with
defaults that keep all three preset digests.

## Amendment

The amendment (`e3a94f34`), frozen after an independent review and the
second B200 amendment and before the corrected implementation and its
rerun, moved the B200 results identity to the regenerated file on the
stacked base and added rules the review found missing: the `nvlink -R`
table now binds to NCCL devices by bus id through each block's UUID and the
`nvidia-smi -q` mapping, so a permuted table is refused; the switch device
list requires class `0x0680` and vendor `0x10de`, a listed switch receiving
no lane and a dump switch absent from the list are refused; H2 asserts the
eight ConnectX virtual functions and the `PIX` GPU-to-NIC pairing; and the
silicon check applies (the board binds to `h100` on device `0x2335`, `sm`
90, and is refused for `b200`). The corrected implementation is `17703dd0`;
the tracked results were regenerated by it, with the stamp naming the
checked-out commit at run time.

## Reproduction

```bash
python examples/hgx_h200_switch_capture_v1/run_study.py --library <native switch library> --output <evidence directory>
python examples/hgx_h200_switch_capture_v1/run_study.py --library <native switch library> --output <evidence directory> --check
```

The native switch library is built as `tests/test_nvswitch_native.py` builds
it. The capture script and rental flow are the ones recorded in the fixture's
provenance.
