# Addendum 2: contended repeated-WQE streams, pre-registered expectations

Written and frozen before any simulation run of this variant
(2026-08-05). The single-pair repetition grid (expectations-rep.md)
produced a mechanism finding during its first execution: a single
source can never inject faster than its own access link, so same-pair
repetition alone cannot overflow any buffer in this topology model (the
funnel is the sender's own serializer). The maintainer's intended
overflow therefore needs convergence. This variant registers it: two
same-leaf senders each stream n repeated WQEs into one receiver, so the
offered load is 2 C into a C bottleneck for the duration of the burst.

Model premise under test (the maintainer's framing): each WQE is a new
flow starting. In this comparator every send op starts at line rate
with no cross-WQE QP state, so the 2 C overload never abates; a real
per-QP rate limiter would converge both senders toward C/2 and largely
stop the loss after a transient. This experiment therefore measures the
WQE-as-new-flow semantics, and its interpretation note must say so:
on this axis the comparator is harsher than a single-QP hardware
stream, which is the opposite bias from the timer axis, and choosing
between the semantics is exactly the HTSIM-5 modeling decision.

## Setup

Senders ranks 0 and 1 (one leaf), receiver rank 15 (another leaf), each
sender posts n independent WQEs of size S, n in {10, 100, 1000}, S in
{16 KiB, 64 KiB}, 400G, DCQCN both modes at 1 MiB buffers (seed 1, plus
seed 2 at n = 100), fluid and cn deterministic. Metric: aggregate
goodput = 2 n S / JCT.

Overflow condition, derived: both senders inject at C, service is C, so
the queue grows at rate C for the burst duration n S / C and peaks near
n S (one sender's worth of bytes). Overflow iff n S > 1 MiB per sender:
at 16 KiB that is n > 64 (cells n = 100, 1000), at 64 KiB n > 16
(cells n = 100, 1000); every n = 10 cell fits (640 KiB and 160 KiB
peaks) with headroom.

## Registered checks (derivations inline)

- Q1 (fluid): aggregate goodput >= 0.90 C at n >= 100 both sizes.
  Derivation: JCT = 2 n S / C + P; at the smallest such cell (n = 100,
  16 KiB) transfer is 65.5 us against P = 2 us, giving 0.97 C; the
  0.90 bar leaves room for the fluid serialization detail at the
  shared egress.
- Q2 (cn): aggregate goodput in [0.75, 1.00] C at n >= 100 both sizes,
  zero recovery counters (real key names as in addendum 1). Derivation:
  the 0.9 C pacing basis minus the concurrent-stream amortization
  deficit observed in addendum 1's aggregates (its floor 0.80 C,
  widened to 0.75 here because two competing senders split one
  reservation domain and the per-stream overheads do not halve).
- Q3 (DCQCN, absorbed cells): at n = 10, both modes and both sizes
  complete without drops or pauses.
- Q4 (DCQCN ECN-only, overflow cells): every overflow cell (n >= 100,
  both sizes) drops packets, and aggregate goodput < 0.1 C at n = 100
  for both sizes. Derivation: excess bytes at n = 100 are about n S
  minus buffer (0.6 MB at 16 KiB, 5.4 MB at 64 KiB), so drops are
  guaranteed; any silent-RTO tail adds 50 ms against a transfer time
  of 65 to 262 us, capping goodput at 3.2 MB / 50 ms = 0.0013 C to
  12.8 MB / 50 ms = 0.005 C; the 0.1 C bar leaves margin if recovery
  avoids the full RTO.
- Q5 (DCQCN ECN+PFC, overflow cells): pause frames > 0 in every
  overflow cell, and goodput below the cn value at the same cell. As
  in addendum 1, no absolute band: whether PFC rescues throughput at
  the price of pausing the senders (and how deep the pause cascade
  goes, reported via dcqcn_pfc_max_cascade_depth) is the measurement.
- Q6 (ordering): at every overflow cell, cn aggregate goodput > 2x the
  better DCQCN mode (P2 floor 0.75 C against Q4's 0.1 C bar and a PFC
  mode expected well below half of cn).
- Q7 (persistence of the overload): DCQCN ECN-only goodput at n = 1000
  is not better than 2x its n = 100 value at the same size, i.e. the
  collapse does not heal with a longer stream. Derivation: the
  overload is renewed by every fresh WQE, so losses recur throughout
  the stream rather than amortizing like addendum 1's single fixed
  RTO tail; if this check fails with goodput recovering at long n,
  that falsifies the renewal argument and is a finding about the
  recovery machinery worth reporting as such.

## Verdict rule

As before: every check gets an explicit PASS or FAIL in the RESULTS
addendum; this file is never edited after the first run.
