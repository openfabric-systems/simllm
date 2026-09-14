# HGX B200 capture amendment, 2026-09-13

This expectations amendment is post-specified to a reading error found
during implementation review of the fixture and is frozen before the study
harness and its first run. The original freeze stays unchanged; this file
corrects two captured facts and adds one binding rule and one cell.

## Corrected facts

- The board does expose GPU module identities. `nvidia-smi -q` prints
  `Module Id` for all eight GPUs: device order 0 through 7 carries module
  ids 4, 2, 1, 3, 8, 6, 5, 7 (bus ids `51`, `52`, `62`, `63`, `75`, `76`,
  `86`, `87`). The freeze's statement that no module id is exposed, the JSON
  flag `module_id_visible: false` and the same sentence in the fixture's
  provenance are withdrawn. The two other unobservable facts stand: every
  NVLink remote device is the virtual address `FFFFFFFF:FF:FF.0` and no PCI
  device of class `0x0680` is visible, so the switch side remains
  unobservable.
- The container's InfiniBand class listing shows two RDMA-capable devices,
  `mlx5_0` and `mlx5_1`. NCCL's dump lists no RDMA network and the run used
  the socket network, so `rdma_nics: 0` is redefined as the count of
  NCCL-visible RDMA networks; the host device count is two. Nothing in this
  slice uses them.

## Added binding rule and cell

A module id is the baseboard slot of a GPU, which is exactly the "GPU module
IDs" clause of PLACE-6's acceptance. `captured_switched_node` therefore
accepts an optional `module_id_by_gpu_dev` mapping (read by the harness and
tests from the inventory's `nvidia-smi -q` block, never guessed); when
given, the preset's GPU slot for a device is `module_id - 1`, so port and
link identities follow the physical slot rather than the enumeration order,
and the mapping must be a bijection onto 1 through 8 or the join refuses.
When absent, device order is the slot order, as before.

Cell G3 gains: with the fixture's module ids, the join places devices 0
through 7 at slots 3, 1, 0, 2, 7, 5, 4, 6, and every port and link identity
of the resulting peer fabric carries those slots; the two orders differ, so
this is a real binding and not a relabeling. Cell G5 gains one refusal: a
module id map that is not a bijection onto 1 through 8.

## What does not change

Every other literal of G1 through G6, the three compatibility digests, the
`(9, 9)` bundle and its citation, the 400 Gbit/s declared payload rate and
the evidence accounting. The switch-side clause stays unobservable and
unscored, now with two supporting facts instead of three.
