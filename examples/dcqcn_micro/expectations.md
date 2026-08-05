# DCQCN micro-behavior validation: pre-registered expectations

Written and frozen before any simulation run of this study. Purpose
(maintainer direction, 2026-08-05): validate the DCQCN comparator against
the published micro-behaviors, message-size vs bandwidth, incast rate
drop-off (fair share), and join/exit convergence time to fairness, with
rnic-cn on the same axes. If the micro plots match the papers, the
collective-pattern comparisons (examples/dcqcn_vs_cn) are objective; a
mismatch is a calibration gap logged as HTSIM-5 evidence in
docs/modules/backends.md, not something to tune away silently. Paper
anchors and candidate parameter sets are in
docs/papers/msg-size-vs-bandwidth.md.

Checks are target-anchored: M-checks describe what the current comparator
is expected to do (its own mechanics), T-checks describe the published
target behavior; a T-check FAIL with an M-check PASS means the comparator
is self-consistent but uncalibrated, which is HTSIM-5's exact scope.

## E-MSG: message size vs goodput (400G, cross-leaf pair)

Sizes 4 KB to 4 MB, one WQE (Q = 1) and 16 concurrent independent WQEs
(Q = 16, the UCCL Fig. 14 in-flight analog), engines fluid, rnic-cn,
DCQCN (seed 1; this scenario is contention-free per path so seed
variance is not expected).

- M1: the current DCQCN Q = 1 curve follows the fixed-offset law
  B = S / (T_eff + S/C) with T_eff about 8.3 us (measured in the probe
  of the calibration note; the offset is topology store-and-forward, not
  a modeled WQE cost). Registered: every Q = 1 point within 10 percent
  of that law.
- M2: at Q = 16 the current model pipelines perfectly (16 independent
  flows on one path share the link with no per-WQE cost), so aggregate
  goodput at S >= 64 KB reaches at least 90 percent of the Q = 1
  asymptote, far above the UCCL measured curve at small sizes.
- T1 (target, expected FAIL today): the UCCL Fig. 14 no-loss anchors
  (about 28 GB/s at 32 KB, 45 at 64 KB, saturation only at 128 to
  256 KB, Q = 16) are NOT matched by the current model at small sizes;
  the model set A2 curve (T0 = 5.2 us at Q = 1, T0/Q pipelined) is the
  calibration target the HTSIM-5 landing must hit within 15 percent.

## E-INC: incast fair-share drop-off (40G, the DCQCN paper's regime)

N senders, N in {2, 4, 8, 16, 20}, each sending 40 MB to one receiver
at 40 Gbit/s link rate (the paper's testbed rate; DCQCN paper Figure 8
shows N = 4 senders each getting an equal ~9.5 of the 10 Gbit/s fair
share). Per-sender achieved throughput = size / FCT.

- M3: fairness: Jain's index across senders >= 0.95 at every N, both
  DCQCN (each of 3 seeds) and rnic-cn (deterministic). Both protocols
  are symmetric here; gross unfairness would be a bug.
- M4: drop-off shape: mean per-sender throughput tracks the fair share
  C/N within [0.5, 1.05] x C/N at every N (the wide lower bound admits
  recovery and pause overheads at the deeper fan-ins; the paper's
  testbed reached about 0.95 x C/N at N = 4).
- T2 (target): utilization: sum of per-sender throughputs >= 0.85 C at
  every N for DCQCN, the paper's qualitative claim that DCQCN sustains
  near-full utilization under incast without collapse. Below that is a
  calibration gap (e.g. RTO-dominated recovery at the configured
  buffers).

## E-JOIN: join/exit convergence to fairness (40G, paper Figure 10)

Flow A: 200 chained 1 MB chunks (a rate probe: each chunk's FCT gives
A's achieved rate in that interval, resolution 0.2 to 0.4 ms at 40G).
Flow B: starts after a 10 ms calc delay, 100 chained 1 MB chunks, same
bottleneck (cross-leaf pair sharing the receiver link). A finishes
first is NOT expected here (200 > 100 + 10 ms head start at shared
rate); instead B finishes first and A then recovers the free capacity:
the join transient at t = 10 ms and the exit transient at B's
completion are both visible in A's chunk-rate series.

- M5 (join): within 5 ms of B's first chunk starting, A's chunk rate
  falls below 0.65 C (the cut engages), and the two flows' smoothed
  rates converge to within 20 percent of each other (fairness reached)
  no later than 40 ms after the join. The DCQCN paper's fluid model and
  implementation (Figure 10) converge in roughly 20 to 40 ms at 40G
  with paper parameters; our comparator's increase timers are what this
  check probes.
- M6 (exit): after B's last chunk completes, A's chunk rate returns to
  at least 0.9 C. The registered band for the recovery duration is
  deliberately wide, 0.5 to 60 ms: the paper arithmetic gives about
  0.3 ms of FastRecovery halving plus up to 27.5 ms of additive
  increase from C/2 at 40 Mbps per 55 us, and the comparator's actual
  increase machinery is unknown fidelity (that is the measurement).
  Outside the band in either direction is a calibration finding.
- M7 (cn contrast): rnic-cn on the identical schedule reaches fairness
  within 2 windowed-feedforward RTTs of the join (the deterministic
  ledger re-partitions at window granularity, no search dynamics), and
  its post-exit recovery is likewise within 2 RTTs; both bounds are
  structural claims of the algorithm book, so a miss is a cn bug, not
  a calibration gap.

## Verdict rule

Every check gets an explicit PASS or FAIL in RESULTS.md; T-check FAILs
are expected and feed HTSIM-5's acceptance table; this file is never
edited after the first run.
