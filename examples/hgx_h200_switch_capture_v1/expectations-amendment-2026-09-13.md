# HGX H200 switch capture amendment, 2026-09-13

This expectations amendment is post-specified to an independent review of
the implemented slice and to the second B200 amendment, and is frozen
before the corrected implementation and its rerun. The original freeze
stays unchanged.

## Compatibility identity moved

The freeze pinned `examples/hgx_b200_capture_v1/results.json` at the base
commit. The second B200 amendment regenerates that file (its G5 last
control is withdrawn and its implementation stamp moves), so this slice's
fatal identity becomes the regenerated B200 results on the branch this
slice stacks on, and the B200 study's `--check` against that file. The
A100, H100 and B200 preset digests and the DGX results are unchanged.

## Added rules

- The `nvidia-smi nvlink -R` table binds to NCCL devices by bus id: each
  GPU block's heading UUID is kept, the `nvidia-smi -q` block maps UUID to
  bus id, and the join requires the table's bus id for every device to
  equal the dump's. A permuted table (two GPUs swapped, all rotated, two
  heading UUIDs swapped) is refused; cell H6 gains those three controls.
- The switch device list requires class `0x0680` and vendor `0x10de` on
  every row; a listed switch that receives no lane and a dump switch row
  naming a bus id absent from the supplied list are refused; H6 gains
  those controls.
- Cell H2 asserts the inventory's NIC facts the freeze records: eight
  ConnectX virtual functions in the `lspci` block and `PIX` from GPU `i` to
  NIC `i` in the `topo -m` block.
- The silicon check of the second B200 amendment applies: the H200 board
  binds to `"h100"` (device `0x2335`, `sm` 90) and is refused for `"b200"`
  on silicon as well as on lanes.

## What does not change

Every literal of H1 through H5 and H7, the chip binding order, the switch
port identities and the evidence accounting.
