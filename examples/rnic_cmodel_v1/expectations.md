# RNIC golden C model slice B expectations

This document is frozen before the transmit pipeline exists and before any
number is produced by it. It registers the sweep, the closed forms, the bands
and the fatal guards for slice B of the RNIC golden C model: the packetizer,
the outstanding-work window and the packet-rate pacer (BACK-55), driven
through the C facade (BACK-54). No result appears here; results go in
`RESULTS.md` and cite this file's commit hash.

## Scope

Slice B is the transmit half of the endpoint only. There is no ingress meter,
no receive processor, no requester transport, no retransmission, no rate
control and no internal arbiter, so the modelled wire is lossless and every
packet is acknowledged. That is a deliberate boundary, and it is the reason
one registered band below is expected to be missed: the measured depth-1024
point at 8 KiB sits in a loss equilibrium that only BACK-56 can produce.

The study drives the `extern "C"` facade, not the C++ classes, so the same
entry points an RTL testbench uses are the ones under test.

## Model configuration

One RC send queue on one QP, one destination, every work request signaled,
polling at every event. The device runs with the transmit pipeline enabled
(network ABI v2, packetization on) and an externally driven wire.

The profile supplies the hardware constants. The two profiles are
`cx5_100g` (measured) and `cx7_400g` (derived from it by scaling the link,
goodput, packet-rate and threshold fields by four and keeping the initiation,
MTU, header, transport and flow-control fields).

| constant | cx5_100g | cx7_400g |
|---|---:|---:|
| `link_bps` | 100e9 | 400e9 |
| `goodput_bps` (C) | 97.1e9 | 388.4e9 |
| `mtu_bytes` | 4096 | 4096 |
| `wire_header_bytes` | 64 | 64 |
| `t_eff_ps` (T_eff) | 4480000 | 4480000 |
| `wire_round_trip_floor_ps` | 2100000 | 2100000 |
| `tx_pps_per_qp` | 3.87e6 | 15.48e6 |

Two derived quantities are fixed here so they cannot be chosen after the fact:

1. The pacer runs at the effective wire rate
   `eff = goodput_bps * (mtu_bytes + wire_header_bytes) / mtu_bytes`, which is
   98.617190e9 bit/s for `cx5_100g`. A full-MTU packet then delivers exactly
   `goodput_bps` of payload, so the model's asymptote is C by construction and
   the interesting content of the depth-1 check is the segmentation and header
   arithmetic, not the asymptote.
2. T_eff is the complete measured fixed offset per message, so the modelled
   wire round trip is subtracted from it rather than added to it. The five
   work-queue service stages sum to
   `t_eff_ps - wire_round_trip_floor_ps = 2380000 ps`, split as 40000 ps of
   doorbell service, 40000 ps of serialized WQE fetch, 2220000 ps of pipelined
   context readiness, 40000 ps of serialized scheduling and 40000 ps of CQE
   write. The split is declared, not measured: the campaign fitted one lumped
   offset. It is constrained only by the requirement that no serialized stage
   binds before the pacer does at the highest WQE rate in the sweep
   (15.48e6 WQE/s, i.e. 64.6 ns per WQE, against 40 ns per serialized stage).

The wire is a fake network that serializes at `link_bps`, adds
`wire_round_trip_floor_ps / 2` of one-way latency, and acknowledges each
packet after the same one-way latency again. It drops nothing.

## Sweep

Primary grid, 36 cells: message size {4 KiB, 8 KiB, 16 KiB, 64 KiB, 256 KiB,
1 MiB} times queue depth {1, 16, 1024} times profile {cx5_100g, cx7_400g} at
MTU 4096. Queue depth sets the send-queue depth and the outstanding-work
window together.

Registered extra cells, 4 more: MTU 1024 at 1 MiB and depth 1024 for both
profiles (the MTU pair completes with the MTU 4096 cell already in the grid),
and 1 KiB at depth 1024 for both profiles (the packet-rate ceiling cell).

Message count per cell is fixed by the frozen rule
`n = max(4 * depth, 33554432 / size)`, reduced to `max(1, 262144 / packets per
message)` when that would exceed 262144 packets, so a cell is long enough to
amortize pipeline fill and drain below 0.1 percent and short enough to run.

Goodput is `total payload bytes * 8 / (last completion time - 0)`, where time
zero is the first post. Per-QP packet rate is `total packets / last completion
time`.

## Closed forms

The depth-1 law is `B = S * 8 / (T_eff + S * 8 / C)`, with T_eff in seconds
and C in bit/s. It predicts, per profile:

| size | cx5_100g depth-1 (Gb/s) | cx5_100g ceiling ratio | cx7_400g depth-1 (Gb/s) | cx7_400g ceiling ratio |
|---|---:|---:|---:|---:|
| 4 KiB | 6.802 | 14.275 | 7.179 | 54.102 |
| 8 KiB | 12.713 | 7.638 | 14.098 | 27.551 |
| 16 KiB | 22.483 | 4.319 | 27.208 | 14.275 |
| 64 KiB | 53.068 | 1.830 | 89.931 | 4.319 |
| 256 KiB | 80.419 | 1.207 | 212.274 | 1.830 |
| 1 MiB | 92.313 | 1.052 | 321.676 | 1.207 |

The ceiling ratio column is what the depth-1024 over depth-1 ratio must equal
if a lossless pipeline saturates at C, which is what this slice models. The
measured ratios are 5.9 at 8 KiB and 1.57 at 64 KiB on `cx5_100g`.

The MTU tax is `1 - goodput(mtu 1024) / goodput(mtu 4096)`. With a 64 B wire
header the packetizer predicts `1 - (1024 / 1088) / (4096 / 4160)`, i.e.
4.412 percentage points.

## Checks

Every check is one `summary.csv` row with its measured value, its reference
value, its band and a PASS or FAIL verdict.

1. `depth1_law` (12 rows, one per profile and size): measured depth-1 goodput
   is within 15 percent of the law value tabulated above.
2. `depth_ratio_law` (4 rows: both profiles at 8 KiB and 64 KiB): the measured
   depth-1024 over depth-1 ratio is within 2 percent of the ceiling ratio
   above. This is the model-internal check that the pipeline saturates at C.
3. `depth_ratio_measured` (2 rows, `cx5_100g` at 8 KiB and 64 KiB): the same
   ratio is within 20 percent of the measured 5.9 and 1.57. The registered
   prediction is that the 64 KiB row passes (1.830 against a band of 1.256 to
   1.884) and the 8 KiB row fails high (7.638 against a band of 4.720 to
   7.080). The 8 KiB miss is registered in advance as a BACK-56 residual: at
   depth 1024 and 8 KiB the silicon sat in the saturated loss equilibrium, and
   a slice with no ingress meter and no loss cannot reproduce it. If that row
   passes instead, the pipeline is not saturating and the failure is in slice
   B.
4. `mtu_tax` (2 rows, one per profile): the tax at 1 MiB is within 2
   percentage points of the measured 5.6, i.e. inside 3.6 to 7.6 percentage
   points. The packetizer's own prediction, 4.412, sits inside that band.
5. `cx7_scaling` (6 rows, one per size): the `cx7_400g` depth-1 goodput equals
   `S * 8 / (T_eff + S * 8 / (4 * C))` within 1 percent, i.e. the derived
   profile is the measured one with C scaled by four and T_eff unchanged.
6. `pps_ceiling` (2 rows, one per profile): at 1 KiB and depth 1024 the
   measured per-QP packet rate is within 5 percent of `tx_pps_per_qp`
   (3.87 Mpps and 15.48 Mpps), and goodput is correspondingly capped below C.
7. `depth_monotone` (12 rows, one per profile and size): goodput is
   non-decreasing from depth 1 to depth 16 to depth 1024.
8. `ceiling_bound` (2 rows, one per profile): no cell exceeds its profile's
   `goodput_bps`. A cell above the ceiling means the packetizer is not
   charging header bytes.

## Fatal guards

A fatal guard voids the run. It is never reported as a fraction of passing
checks.

- Deterministic replay identity: the designated replay cell (`cx5_100g`,
  64 KiB, depth 16, MTU 4096) is run twice in one process, and both the CSV
  row and the facade transaction trace must be byte-identical.
- One-authority conservation: in every cell, posted equals delivered equals
  reclaimed, the send queue drains to zero, the controlled evidence list is
  empty and the device never reports a fatal state.
- Packet conservation: the packets the packetizer emits sum to
  `ceil(size / mtu)` per message with `size` payload bytes in total, and every
  packet attempt reaches a terminal event before its parent extent does.
- Counter monotonicity: every device counter is non-decreasing across the run.

## What this study cannot show

It cannot show loss, retransmission, congestion response, receive-side
behavior, multi-QP arbitration or anything the fabric owns. It also cannot
independently confirm C or T_eff: both are inputs from the campaign, and the
depth-1 check tests that segmentation, header accounting, the window and the
pacer compose into the fitted law, not that the law is right.
