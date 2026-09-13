# HGX B200 capture result

## Outcome

What ran: `examples/hgx_b200_capture_v1`, the frozen first PLACE-6 slice,
added a `b200` generation to the eight-GPU switched preset, taught the NCCL
topology reader the switched-dump shape, joined the rented HGX B200 board's
capture to that preset, and ran the DGX study's live cells for the new
generation. Expectations-only commit `acf0e5e4`, amendment `bc1e5c8f`
(module ids are exposed and bind the slots; RDMA device wording corrected),
implementation `59cbb4ce`, harness `e8ded163`.

What came out: the result is `PASS` with no finding. The deciding rows are
G3: every GPU-side fact of the captured board equals the preset. Eighteen
active links per GPU equal the preset's eighteen lanes; `NV18` on all 56
ordered pairs equals the preset's eighteen paths per pair and NCCL's count;
the device order, bus ids, PCIe pairing and NUMA split match the join; the
eight Board IDs are distinct under one part number; and the captured
module ids place devices 0 through 7 at slots 3, 1, 0, 2, 7, 5, 4, 6, an
order that differs from enumeration, carried by every port and link
identity. The switch side is recorded as unobservable on evidence: all 144
remote-device rows are the virtual fabric address and no NVSwitch PCI device
is visible. All three compatibility digests held and the DGX study's own
check reproduced.

What it changes for the project: PLACE-6's GPU-side clause is literal for
the B200 board, including the module-id binding it asked for, and the
preset family gains `b200`; PLACE-9 narrows by the switched-dump shape.
PLACE-6's switch-side clause is registered as PLACE-12, which needs a host
that passes its NVSwitches through to the tenant; this B200 container did
not, and the slice that stacks on this one carries a capture that does. One
model finding is recorded for the switch owners: the live B200 cells equal
the H100 cells everywhere at equal lane rate, so the native crossbar model
carries no per-chip capacity term today; for that reason the frozen
"B200 at or above H100" family held vacuously, by equality in every cell,
and it is disclosed here as unscored rather than counted as a behavioral
pass.

What it does not change: no timing is calibrated (TRAF-92), the A100 and
H100 presets are byte identical, the DGX study's frozen grid is untouched,
NIC affinity on switched boards is not modeled, and the 400 Gbit/s B200 lane
rate is checked structurally only, because the DGX live helpers override
link rates with the swept lane rate.

## Evidence

| Evidence class | Result |
|---|---|
| Exact-oracle family (G4 Python versus native) | equal on 96 component and 18 live cells |
| Behavioral relations (G4 rate pair, G4 B200 versus H100) | rate change moved completion in 9 of 9 live pairs with compute identical; B200 equals H100 on all 16 isolated pair cells; at-or-above holds on 48 fan-in and 18 live cells, with equality in every cell |
| Structural exact guards (G1, G2, G3, G6) | 4 of 4 cells exact |
| Rejection controls (G5) | 7 of 7 refused, including a module-id map that is not a bijection |
| Fatal compatibility digests | 3 of 3 identical; the DGX study check reproduces |

Counts in different evidence classes are not added. The tracked
[results](results.json) hold every row.

G1 parses 2 CPUs, 4 PCIe switches, 8 GPUs, 8 switch-attached NVLink rows of
count 18 and one socket NIC seen twice. G2's preset has 2 chips, 144 links,
288 ports, 56 routes of 18 paths, 9 per chip, every link at 400 Gbit/s
payload, from NVIDIA's Fabric Manager User Guide description of two
fourth-generation NVSwitches with nine NVLink 5 links from each GPU to each.
G4's tensor-parallel width-8 step completes in 747,080 ps at 12.5 GB/s per
lane and 424,520 ps at 25 GB/s, identical for the `b200` and `h100`
generations because both give every pair eighteen lanes and the crossbar
model prices no per-chip capacity.

## Physical sanity

Eighteen lanes at 400 Gbit/s payload give 900 GB/s per GPU per direction,
NVIDIA's published NVLink 5 figure; the captured 53.125 GB/s signalling
gives 956 GB/s, 6.25 percent above payload, the same ratio TRAF-44 recorded
for NVLink 4. The live cells swept 12.5 and 25 GB/s per lane, below both,
and every phase sits above its lane serialization floor.

## What the container did not show

Every NVLink remote device is `FFFFFFFF:FF:FF.0`, no PCI device of class
`0x0680` is visible, and NCCL sees one bridge-class target with count 18.
The host's PCIe-switch listing also lacks the bus ids NCCL names, so the
PCIe placement rests on the dump. Two RDMA-capable devices are listed but
NCCL exposed no RDMA network. The join refuses a switched board that
exposes a GPU Direct RDMA NIC; NIC affinity on switched boards remains
PLACE-6's own clause.

## Amendments

The first amendment (`bc1e5c8f`) corrected two captured facts before the
harness ran: module ids are exposed and bind the slots, and two RDMA-capable
devices sit in the container. The second amendment (`989482e2`), frozen
after an independent review and before the corrected join and its rerun,
withdrew G5's last control (generation `h100` accepted against the
eighteen-lane board on lane count alone) and added a silicon check: the
join now refuses a board whose GPUs do not report the generation's PCI
device id and streaming-multiprocessor version, a GPU outside a PCIe switch,
and a PCIe switch nested in a PCIe switch. G5 keeps seven controls with the
`h100` request refused on silicon, and the tracked results were regenerated
by the corrected join (commit `f7d55479`; the results stamp names its
parent `989482e2`, the checked-out commit at run time).

## Reproduction

```bash
python examples/hgx_b200_capture_v1/run_study.py --library <native switch library> --output <evidence directory>
python examples/hgx_b200_capture_v1/run_study.py --library <native switch library> --output <evidence directory> --check
```

The native switch library is built as `tests/test_nvswitch_native.py` builds
it. The capture script and rental flow are the ones recorded in the fixture's
provenance.
