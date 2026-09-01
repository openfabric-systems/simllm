# Multi-node collective capture expectations

The TRAF-77 evidence slice: independently observed multi-node collective
behavior from the Merlin7 A100 cluster, at two routing concentrations
and two widths, frozen before capture. The registry's switch-occupancy
and buffer high-water clauses are NOT satisfiable from standard user
access (no switch telemetry); this campaign captures every observable
endpoint proxy and the narrowed TRAF-77 entry will say exactly which
clauses remain open. Closing TRAF-77 is not a permitted outcome.

## Frozen sweep

- System: Merlin7 gmerlin7, a100 partitions, gpu101-105 class nodes
  (4x A100-SXM4-80GB NV4 mesh per node, 4x 200 Gbit/s Cassini ports),
  cuda/12.2.2 module, nvidia-nccl-cu12, GDR status recorded as found.
- Widths: 2 (two nodes, one GPU each) and 8 (two nodes, four GPUs
  each), matching the existing crossnode_collective_envelope_v1
  geometry so its published anchors act as consistency checks
  (width-2 8-byte all-reduce 40,140,799 ps; width-8 8-byte all-reduce
  50.790 us; width-8 8-byte all-to-allv 89.805 us).
- Routing concentrations: one-port (all cross-node traffic pinned to a
  single Cassini port per node) and four-port (NCCL default spread),
  pinned via the interface environment controls, with the achieved
  distribution PROVEN from per-port TX/RX counters, not assumed.
- Operations: all-gather, reduce-scatter, all-reduce, pairwise
  all-to-allv. Payload ladder: 8 B to 128 MiB in the existing
  crossnode ladder steps. Repeats: enough for a stated median and p95
  per cell, count frozen in the study configuration.
- Captures per cell: phase release and completion wall times,
  per-chunk completion where NCCL debug exposes it, NCCL algorithm,
  protocol and channel selection from logs, per-port byte counters
  before and after, and every host or NIC queue counter readable
  without privilege, each named in the configuration with its source.

## Fatal guards

- FG-1 no compute on the login node; outputs under ~/simllm-data/,
  append-only attempt directories mirrored to
  /data3/yifeng/simllm-dev/planmode-runs/traf77-t2/. The capture is
  executed by the integrator over ssh because worker sandboxes have no
  network; the submitted scripts are the tracked ones, byte for byte,
  and the record proves it by hashing them on both sides.
- FG-2 the achieved routing concentration is proven from counters; a
  cell whose counters contradict its declared concentration voids that
  cell, never silently relabels it.
- FG-3 environment identity recorded: driver, CUDA, NCCL version,
  nodes, NUMA binding, GDR state; the CUDA-13 trap explicitly avoided
  and the check recorded.
- FG-4 the existing crossnode anchors reproduce within a stated
  consistency band at the matching cells before any new cell is read;
  a miss voids the campaign with the divergence published.
- FG-5 a100 submission only after the flagship ping per the recorded
  fence; the ping and reply are quoted in the record.
- FG-6 determinism of the ANALYSIS: scoring from the captured evidence
  reproduces byte for byte; the captures themselves are hardware and
  never expected to be bit-stable.
- FG-7 chronology: this freeze, with expected directions, precedes
  every capture.

## Expected directions (frozen, falsifiable)

- E1 one-port concentration is never faster than four-port at equal
  payload for width-8 collectives; the large-payload ratio is at least
  2.0 (four links funneled into one).
- E2 width-8 completion exceeds width-2 at every payload for
  all-reduce (more ranks, same fabric).
- E3 per-chunk completion under one-port concentration shows a
  serialization signature (monotone chunk completion spacing at large
  payloads) absent or weaker at four-port.
- E4 the 8-byte floors are within 2x of the existing crossnode
  anchors (same stack, same site).
Misses publish as refutations with the mechanism named.

## Scored families

- C consistency: the FG-4 anchor cells, scored.
- R routing: E1 through E4 as scored directional rows.
- L ladders: the full payload ladders published with median and p95;
  scored only for completeness (every frozen cell present or its
  absence named with the scheduler evidence).
- W wall: cluster wall time disclosed; the analysis completes in 600 s.

## What T3 consumes

The fitted cross-node transport surrogate (rnic-cn service and the
switch-buffer replacement as a declared, sourced input) trains only on
cells this freeze marks trainable and validates on the rest; that
protocol is T3's own freeze. TRAF-77 then narrows quoting this
campaign; its H200-target and switch-telemetry clauses stay open.
