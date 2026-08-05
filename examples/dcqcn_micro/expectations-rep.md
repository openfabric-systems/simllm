# Addendum: repeated-WQE stream collapse, pre-registered expectations

Written and frozen before any simulation run of this addendum
(2026-08-05; the base study's expectations.md was already frozen, so
this experiment carries its own registration). Maintainer direction:
repeat the same message size 10 to 10k times as consecutive WQEs so the
aggregate overflows the pipe and induces loss or PFC; expected shape:
rnic-cn and fluid hold line rate (small messages at full stream speed),
DCQCN drops toward nothing beyond some repetition count.

Per the process fix recorded after the base study, every numeric bar
below carries its derivation inline, and the harness constants are
diffed against this file before the first run.

## Setup

Rank 0 posts n WQEs of size S to rank 15 (cross-leaf), all independent
(no requires), n in {10, 100, 1000, 10000}, S in {16 KiB, 64 KiB},
400G. In this comparator every send op is an independent flow starting
at line rate with no shared QP state (dcqcn_atlahs_runtime.cpp:398), so
n posted WQEs are n simultaneous line-rate flows into one path: the
maximal version of the burst a deep hardware WQE queue drains. DCQCN
runs both modes at the established 1 MiB lossy buffers
(shared_buffer_bytes = egress_buffer_bytes = 1,048,576), seed 1 plus a
2-seed spot check at n = 100; fluid and cn are deterministic single
runs. Metric: aggregate goodput = n S / JCT.

## Registered checks (derivations inline)

- P1 (fluid): aggregate goodput >= 0.95 C for every n >= 100 at both
  sizes. Derivation: JCT_fluid = n S / C + P with P = 2 us; at the
  smallest such cell (n = 100, S = 16 KiB) n S / C = 1.6 MB / 50 GB/s
  = 32.8 us, so goodput = 32.8 / 34.8 = 0.94 C. That cell rounds to
  the bar; every larger cell exceeds it (0.994 C at n = 1000). To keep
  the bar honest at the boundary cell the check uses >= 0.94 C for
  (n = 100, S = 16 KiB) and >= 0.95 C elsewhere.
- P2 (cn): aggregate goodput in [0.80, 1.00] C for every n >= 100 at
  both sizes, zero recovery counters (the real key names,
  rnic_cn_gap_nacks_dispatched, rnic_cn_late_data_packets,
  rnic_cn_deterministic_retransmissions,
  rnic_cn_maximum_retry_attempt_observed), and bit-determinism at
  (n = 1000, S = 64 KiB) across two runs. Derivation of the band: cn
  paces at a 0.9 C basis; the concurrent-stream aggregate amortizes the
  per-batch control cost (the base study's Q = 16 cells reached
  0.79 to 0.90 of the fluid value and the deficit shrinks with
  aggregate size; at n >= 100 the aggregate is >= 1.6 MB where the
  base-study cn aggregate law gives >= 0.83 C), so 0.80 C is the floor
  and the 0.9 C basis bounds above; small messages at full stream
  speed is the claim under test.
- P3 (DCQCN, buffer-absorbed cells): for n S <= 1 MiB (n = 10 at both
  sizes) both modes complete without drops or pauses and goodput is
  within 25 percent of the fluid value for the same cell (the base
  study's incast cell that fits the buffer behaved ideally; 640 KiB
  and 160 KiB bursts fit with headroom).
- P4 (DCQCN ECN-only, overflow cells): every cell with n S > 1 MiB
  drops packets, and aggregate goodput < 0.1 C at n = 100 for both
  sizes. Derivation: at n = 100, S = 64 KiB the data is 6.4 MB
  (128 us at line rate) but any silent-RTO tail adds 50 ms
  (silent_loss_rto_ps = 5e10), so goodput <= 6.4 MB / 50 ms =
  0.128 GB/s = 0.0026 C; the 0.1 C bar leaves two orders of margin in
  case recovery avoids the full RTO. "Drops toward almost nothing
  beyond 100 repetitions", as directed.
- P5 (DCQCN ECN-only, large-n shape): goodput at n = 10000 exceeds
  goodput at n = 100 for S = 64 KiB (the fixed RTO tail amortizes over
  640 MB of data: 640 MB / 50 GB/s = 12.8 ms of transfer against the
  same tail), i.e. the collapse curve is non-monotonic with its
  minimum at intermediate n. This is a mechanism prediction of the
  RTO-quantized tail, not a recovery claim; if instead repeated loss
  keeps goodput flat or falling, that is a finding about compounding
  loss worth reporting.
- P6 (DCQCN ECN+PFC, overflow cells): every overflow cell emits PFC
  pause frames (the lossless mode must pause under a burst it cannot
  buffer), and its goodput stays below the cn value at the same cell.
  No absolute band is registered: whether PFC mode survives with
  degraded goodput or deadlocks into its own collapse is exactly what
  the measurement is for.
- P7 (ordering): at every overflow cell, cn goodput > 2x the better
  DCQCN mode's goodput (the collective-study ordering restated at the
  stream level; derivation: P2 floor 0.80 C against P4's 0.1 C bar and
  PFC pause overheads, with the 2x margin absorbing a PFC mode that
  paces well).

## Verdict rule

As in the base study: every check gets an explicit PASS or FAIL in the
RESULTS addendum; this file is never edited after the first run.
