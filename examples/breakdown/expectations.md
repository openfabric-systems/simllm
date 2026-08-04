# Request-time breakdown: pre-registered expectations

Written and frozen before any simulation run of this study (repo
validation rule). The question is qualitative: for one canonical request,
how does its total latency decompose into compute-bound kernel time,
memory-bound kernel time and network time, expected vs actual, as
parallelism (TP), the link speed and the network profile/topology change.

## Request and decomposition

One request: a 2048-token prompt served as a single prefill chunk, then 8
generated tokens, i.e. 8 steps: `prefill 2048 @ context 2048`, then seven
`decode 1 @ context 2049..2055` steps (batch 1, the request alone).

Per step, the sink charges kernel time `K = L * c * 1000` ps (the GOAL
calc chain; `c = floor(E / (L * 1000))`, E the whole-step roofline
estimate at efficiency 0.7 on the b100 envelope) and network time
`N = makespan - K`. K is attributed to **compute** when the roofline
classifies the step's fused kernel compute-bound and to **memory** when
memory-bound (the roofline's `max()` means the binding resource owns the
step; the shadowed resource is not separately charged). Request components
are the sums over the 8 steps. Declared per-rank geometry: llama-8b-shaped
as in examples/m4 (L=32, hidden 4096, dtype 2 B, sharded by TP).

## Registered qualitative expectations

- Q1 (bound structure): at every TP in {2, 4, 8}, the prefill step is
  compute-bound and every decode step is memory-bound. So the compute
  component comes entirely from the prefill step and the memory component
  entirely from the decodes.
- Q2 (parallelism): the network share of the request total strictly
  increases with TP on every profile: per allreduce the round count
  2(W-1) grows, the decode rounds are propagation-dominated (chunk * 20 ps
  << P = 2 us), and the prefill's total wired bytes 2(W-1)S/W grow with W,
  while both kernel components shrink roughly as 1/TP.
- Q3 (link speed, fluid): dropping 400G to 100G multiplies only the wire
  term (80 vs 20 ps per byte), never the propagation term or the kernels.
  So the compute and memory components are bit-identical across link
  speeds, the prefill-step network time grows by more than 2x, and each
  decode step's network time grows by less than 10 percent (decode chunks
  are at most 4096 B, so wire is at most 4096 * 80 ps = 0.33 us per round
  against P = 2 us).
- Q4 (profile/topology, 400G): rnic-nn is never faster than fluid
  (packetization adds store-and-forward slots), and rnic-cn on the real
  two-tier Clos (`examples/m1/topologies/clos_64_400g.topo`, TP ranks all
  on node 0) is never faster than fluid. The interesting registered claim:
  the *decode-phase* network inflation factor of cn over fluid exceeds the
  *prefill-phase* inflation factor, because cn's control overhead is
  additive per flow (M1 finding F2, about 28.5 us per flow) and the decode
  rounds' useful work is tiny while the prefill rounds are wire-dominated.
  cn vs nn ordering is reported, not registered (different topologies).

## Frozen quantitative tables

Component sums per request, computed by closed form before any run.

Check F (rnic-nn-fluid, exact to 0 ps): kernel terms from the frozen
roofline estimates, network from
`sum over steps of 2L * 2(W-1) * (chunk * psb + P)` with psb = 20 (400G)
or 80 (100G):

| Check | TP | link | compute ps | memory ps | network ps | total ps |
|---|---|---|---|---|---|---|
| F1 | 2 | 400G | 11,781,312,000 | 10,205,472,000 | 23,596,236,800 | 45,583,020,800 |
| F2 | 2 | 100G | 11,781,312,000 | 10,205,472,000 | 88,240,947,200 | 110,227,731,200 |
| F3 | 4 | 400G | 5,891,072,000 | 5,759,360,000 | 38,466,355,200 | 50,116,787,200 |
| F4 | 4 | 100G | 5,891,072,000 | 5,759,360,000 | 135,433,420,800 | 147,083,852,800 |
| F5 | 8 | 400G | 2,945,952,000 | 3,536,288,000 | 52,045,414,400 | 58,527,654,400 |
| F6 | 8 | 100G | 2,945,952,000 | 3,536,288,000 | 165,173,657,600 | 171,655,897,600 |

Bar: every component and total exact to 0 ps. The harness compares
against these literals and separately checks the runtime roofline against
the frozen kernel terms (the post-M4-audit anchoring rule).

Check N (rnic-nn at 400G, banded point form): kernel terms as in F;
network from the generalized per-round point form
`round = wire_bytes * 20 + 83,200 + P` with
`wire_bytes = Nfull * 4160 + (rem + 64 if rem else 0)` per chunk, band
one slot (83,200 ps) per round in either direction, floor never below the
F network value. TP 4 and 8 have sub-packet decode chunks (2048 B and
1024 B), the regime where the M1 incast ladder saw slot-calendar effects;
the band is registered all the same and a miss is an honest FAIL with
analysis:

| Check | TP | network point ps | band +- ps | sub-packet decode chunks |
|---|---|---|---|---|
| N1 | 2 | 24,018,124,800 | 85,196,800 | no |
| N2 | 4 | 39,228,702,720 | 255,590,400 | yes |
| N3 | 8 | 53,237,022,720 | 596,377,600 | yes |

Check C (rnic-cn on the Clos at 400G, directional only):

- C-a: the cn request total is >= the F (fluid, 400G) total at the same
  TP.
- C-b: decode-phase network inflation (cn network / fluid network, decode
  steps only) > prefill-phase network inflation (same ratio, prefill
  step), at every TP.
- C-c (report only): cn vs nn totals and the per-flow overhead implied by
  the cn-minus-fluid delta divided by the flow count.
