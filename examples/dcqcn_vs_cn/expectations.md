# DCQCN vs rnic-cn: pre-registered expectations

Written and frozen before any simulation run of this study. The question:
across the traffic classes that matter (incast, ECMP collision, mixed
all-to-all, and a contention-free tensor-parallel request), where is the
DCQCN comparator worse than rnic-cn, by how much, and where is it not.

The already-merged comparator arc (examples/cn_ladder, seeded study at
the 1 MiB lossy operating point) established the shape of the answer:
DCQCN can match or beat rnic-cn at the uncontended median, and it loses
catastrophically in the tails and under contention (there: median 1.52
vs 2.06, but p99 1902x vs 19.3x and max 3528x with stranded flows).
This study registers that structure across scenarios rather than
claiming a uniform "worse in all cases": the registered claims below say
exactly where DCQCN must lose, and the one scenario where it is expected
to WIN is registered as such, because it maps the boundary (DCQCN's
failure mode is contention, not baseline overhead) and hiding it would
be dishonest.

## Setup

Topology: `examples/m1/topologies/clos_64_400g.topo` (64 nodes, 2-tier,
400G) for every engine. rnic-nn-fluid provides the ideal per-flow
denominator: normalized FCT = FCT / fluid FCT of the identical flow
(matched by source, destination, tag). rnic-cn and fluid are
deterministic (single runs; cn repeated once to confirm determinism).
DCQCN (`htsim_dcqcn_atlahs`, mlx5-faithful GBN recovery) runs seeded
ensembles at the realistic 1 MiB buffer point
(`-shared_buffer_bytes 1048576 -egress_buffer_bytes 1048576`), modes
ECN-only (`-pfc off`) and ECN+PFC (`-pfc on`) where stated.

## Scenarios and registered claims

### S-INC: incast, fan-in {8, 32} x size {64 KiB, 2 MiB}

Sources ranks 1..F send one flow to rank 0. DCQCN: both modes, 5 seeds
each.

- R1a: in every cell and both modes, the DCQCN ensemble p99 normalized
  FCT exceeds the cn p99, and the ensemble max exceeds the cn max.
- R1b: at the 2 MiB x 32 cell in ECN-only mode, DCQCN drops packets
  (`ns_tm3_dropped_packets > 0`) in every seed; rnic-cn reports zero
  loss-recovery events in every cell (lossless by design).
- R1c: in ECN+PFC mode at 2 MiB x 32, PFC pause frames are emitted
  (`dcqcn_pfc_pause_frames > 0`); cn needs no PFC anywhere.

### S-ECMP: cross-leaf permutation, the collision scenario

16 flows of 8 MiB: nodes i (leaves i) send from ranks 8i and 8i+1 to
ranks 8j+2 and 8j+3 of node j = (i+1) mod 8. Every leaf carries exactly
two uplink flows and receives exactly two downlink flows, so per seed
each pair independently collides on a shared leaf uplink or spine
downlink with probability about 1/8 per opportunity; with 16
opportunities the chance a seed shows at least one collision is about
1 - (7/8)^16, roughly 0.88. A collision halves the colliding flows'
bandwidth, so their normalized FCT approaches 2. DCQCN: ECN+PFC, 8
seeds. rnic-cn sprays per packet and cannot collide.

- R2a: cn is deterministic (two runs bit-identical) and its max
  normalized FCT is below 1.5.
- R2b: at least 5 of the 8 DCQCN seeds have a max normalized FCT of at
  least 1.7 (the collision signature), and the ensemble max exceeds the
  cn max.

### S-A2A: mixed lognormal all-to-all, the established operating point

The cn_ladder `mixed_all_to_all_goal(seed=7)` (16 ranks, lognormal mean
256 KiB) at 1 MiB buffers. DCQCN: both modes, 8 seeds.

- R3a: in both modes, the DCQCN ensemble p99 normalized FCT is more than
  10x the cn p99, and the ensemble max is more than 20x the cn max
  (the merged study measured about 98x and 183x; the registered bars
  leave wide margin for seed and mode variation).
- R3b: DCQCN shows loss recovery (drops or silent RTOs > 0) in every
  ECN-only seed; cn reports zero recovery events.
- R3c (report only, no claim): the medians. The merged study saw the
  DCQCN median beat cn's; whichever way it lands here is reported as is.

### S-TP: contention-free tensor-parallel request (the boundary case)

The breakdown study's 8-step request (2048-token prefill plus 7
decodes, TP=8) with the TP group spread across nodes (one GOAL rank per
leaf: ranks 7, 15, ..., 63), rendered as the same serial
calc-plus-ring-allreduce GOAL and executed by all three engines. Every
ring round is one flow up and one flow down per leaf: no shared uplink,
no shared spine downlink, no oversubscription, so there is nothing to
congest and nothing for ECMP to collide.

- R4a (the honest boundary claim): DCQCN's request total is BELOW
  rnic-cn's here, because cn pays its deterministic per-round control
  overhead while DCQCN in a contention-free serial schedule has nothing
  to pay; and the DCQCN seed spread is below 1 percent of the total
  (no ECMP choice matters). Both totals are at or above the fluid
  total.
- R4b: cn is deterministic (two runs bit-identical).

If R4a fails with DCQCN above cn even here, that is reported as a
finding (it would mean DCQCN's baseline is worse than assumed, which
only strengthens the comparator conclusion but must be measured, not
asserted).

### R5: rnic-cn tracks rnic-nn on large aggregated flows (maintainer rule)

Added before the first simulation run of this study (the freeze point is
the first run, which had not happened; nothing below is fitted). The
maintainer's calibration rule: for flows at or above the fabric BDP
(about 200 KB at 400G and 2 us propagation), rnic-cn must be very close
to rnic-nn, because the cn control overhead is a per-round additive term
that large transfers amortize; a miss is a bug to debug, not a result to
report around.

- R5a: on the S-ECMP permutation (16 x 8 MiB, far above BDP), the cn job
  completion time is within 1.25x of the rnic-nn JCT for the same GOAL.
- R5b: on both 2 MiB incast cells, the cn p99 normalized FCT is within
  1.25x of the rnic-nn p99.
- R5c (report only): the same ratio for the a2a16 mixed workload, whose
  mean size sits at the BDP boundary; no bar is registered there (the
  sub-BDP tail of the lognormal is the known cn corner regime).

## Verdict rule

Every claim above gets an explicit PASS or FAIL in RESULTS.md; a FAIL
carries analysis, and this file is never edited after the first run.
