# M4 closed-loop slice: pre-registered expectations

Written and frozen before any M4 simulation run (repo validation rule:
expectations first, then measurements; every deviation is explained or is a
bug). All numbers below are closed-form arithmetic evaluated before the
first `htsim_rnic` invocation of this milestone; nothing here is fitted to
a measurement.

## Reused M1 calibration constants (not rederived)

From examples/m1/RESULTS.md (C1, C2, C3), all confirmed to 0 ps there:

- GOAL `calc` unit u = 1000 ps per unit (ns), with a 1 ns floor on
  `calc 0` (a zero-cost calc still takes 1000 ps).
- `rnic-nn-fluid`: wire factor h = 1.0, fixed propagation P = 2,000,000 ps.
  Wire time at 400 Gbit/s is exactly 20 ps per payload byte.
- `rnic-nn`: h = 4160/4096 (4160 B wire packet per 4096 B payload), one
  wire slot = 83,200 ps at 400 Gbit/s, and a single chained flow completes
  at serialization + P + exactly one slot (the C3 store-and-forward
  intercept).

## System under test

`HtsimStepSink` (simllm.backends.step_sink) renders one GOAL program per
scheduler step over the tensor-parallel group of W GOAL ranks:

- Serial chain over L layers per rank: `calc(c)` then the layer's two ring
  allreduces (attention output, MLP output). No overlap, no sequence
  parallelism, no PP traffic (tasks TRAF-6, TRAF-7, TRAF-8).
- Each allreduce moves payload S = total_new_tokens * hidden_size *
  dtype_bytes through `simllm.traffic.ring_allreduce`: 2(W-1) chained
  rounds, each round every rank sends one chunk = S/W to its ring successor
  and receives one from its predecessor; a round starts when the previous
  round's receive completed on that rank.
- Per-layer calc cost c = floor(E / (L * 1000)) calc units (ns), where E =
  `estimate_step_latency_ps(dims, record, num_sampled=len(scheduled),
  RooflineProvider(efficiency=0.7), GPU_ENVELOPES["b100"],
  HostInitiationModel(0))` is the whole-step roofline estimate in ps (even
  split across layers, task BACK-5; num_sampled approximation BACK-6).

Model geometry (declared, llama-8b-shaped, per TP rank): L = 32 layers,
hidden 4096, head size 128, 32/TP heads, 8/TP KV heads, intermediate
14336/TP, vocab 128256 (LM head deliberately kept unsharded in the per-rank
dims), activation dtype 2 bytes. GPU envelope b100: 1.8e15 FLOP/s,
8e12 B/s, efficiency 0.7. Topology: none (generated null-network manifold),
endpoint links 400 Gbit/s.

Step shapes (the two step-shape parameter points; TP in {2, 4, 8} is the
other swept parameter, per the two-parameter sanity-study rule):

- `decode8x2048`: 8 requests, 1 new token each, context_length 2048.
  Total new tokens 8, so S = 8 * 4096 * 2 = 65,536 B.
- `prefill2048`: 1 request, one 2048-token prefill chunk, context 2048,
  no cached tokens. Total new tokens 2048, so S = 2048 * 4096 * 2 =
  16,777,216 B.

## Check A: standalone ring allreduce on rnic-nn-fluid (exact to 0 ps)

A ring allreduce alone (no calc), released at t = 0. Every round is W
disjoint full-rate flows (each NIC sends exactly one and receives exactly
one), so on the fluid manifold each round takes chunk * 20 + P ps and the
rounds chain strictly:

    JCT_fluid = 2 * (W - 1) * (chunk * 20 + 2,000,000)   [ps], chunk = S / W

| Check | S bytes | W | chunk | predicted JCT ps |
|---|---|---|---|---|
| A1 | 65,536 | 2 | 32,768 | 5,310,720 |
| A2 | 65,536 | 4 | 16,384 | 13,966,080 |
| A3 | 65,536 | 8 | 8,192 | 30,293,760 |
| A4 | 16,777,216 | 2 | 8,388,608 | 339,544,320 |
| A5 | 16,777,216 | 4 | 4,194,304 | 515,316,480 |
| A6 | 16,777,216 | 8 | 2,097,152 | 615,202,560 |

Bar: measured JCT equals the prediction with 0 ps residual (the fluid
backend is deterministic and exact, as in M1 C4/C5).

## Check B: full step makespan on rnic-nn-fluid (exact to 0 ps)

The step's GOAL is the serial chain above, so with c the per-layer calc
units (all c below are >= 1, so the calc-0 floor never binds):

    makespan_fluid = L * c * 1000 + 2L * [2(W-1) * (chunk * 20 + P)]

The whole-step roofline estimates E (pure arithmetic on the declared dims;
these are inputs to the form, frozen here so the grid cannot be fitted):

| Check | shape | TP | E ps | c ns | chunk | predicted makespan ps |
|---|---|---|---|---|---|---|
| B1 | decode8x2048 | 2 | 1,625,667,291 | 50,802 | 32,768 | 1,965,550,080 |
| B2 | decode8x2048 | 4 | 906,643,748 | 28,332 | 16,384 | 1,800,453,120 |
| B3 | decode8x2048 | 8 | 547,131,977 | 17,097 | 8,192 | 2,485,904,640 |
| B4 | prefill2048 | 2 | 11,781,315,593 | 368,166 | 8,388,608 | 33,512,148,480 |
| B5 | prefill2048 | 4 | 5,891,074,730 | 184,096 | 4,194,304 | 38,871,326,720 |
| B6 | prefill2048 | 8 | 2,945,954,299 | 92,061 | 2,097,152 | 42,318,915,840 |

Bar: 0 ps residual on every cell.

Registered qualitative shape (direction checks):

- B-q1: the decode makespan is non-monotonic in TP: TP=4 < TP=2 < TP=8.
  Compute halves with TP but the allreduce is propagation-dominated
  (chunk * 20 << P) and its round count 2(W-1) grows, so 64 allreduces at
  W=8 cost 64 * 14 * ~2.16 us ~ 1.94 ms of nearly pure latency.
- B-q2: the prefill makespan increases monotonically in TP: the wire term
  dominates each round and total wired bytes per allreduce 2(W-1) * S/W
  grow with W while propagation rounds also grow.

## Check C: the same grid on rnic-nn (bounded prediction, not exact)

Per round each flow is chunk/4096 full packets (every chunk below is a
multiple of 4096). Point form per round: serialization Npkt * 83,200 plus
P plus the C3 single-flow store-and-forward slot:

    round_nn = (Npkt + 1) * 83,200 + 2,000,000,  Npkt = chunk / 4096
    makespan_nn_point = L * c * 1000 + R * round_nn,  R = 2L * 2(W-1)

Band: M1 C7 measured final-packet pipelining effects of a few slots under
concurrent flows; register the band as at most one slot per round in
either direction, plus the hard floor that packetization never beats
fluid:

    makespan_fluid <= makespan_nn in [point - R * 83,200, point + R * 83,200]

| Check | shape | TP | R | round_nn ps | point ps | band +- ps |
|---|---|---|---|---|---|---|
| C1 | decode8x2048 | 2 | 128 | 2,748,800 | 1,977,510,400 | 10,649,600 |
| C2 | decode8x2048 | 4 | 384 | 2,416,000 | 1,834,368,000 | 31,948,800 |
| C3 | decode8x2048 | 8 | 896 | 2,249,600 | 2,562,745,600 | 74,547,200 |
| C4 | prefill2048 | 2 | 128 | 172,476,800 | 33,858,342,400 | 10,649,600 |
| C5 | prefill2048 | 4 | 384 | 87,280,000 | 39,406,592,000 | 31,948,800 |
| C6 | prefill2048 | 8 | 896 | 44,681,600 | 42,980,665,600 | 74,547,200 |

Bar: every measured nn makespan is >= its fluid counterpart and inside the
registered band. The exact deviation from the point form is reported per
cell; a deviation outside the band is a FAIL.

## Check D: TTFT and TPOT of a replayed step sequence (virtual mode)

Replay a synthetic 9-step sequence at TP = 8 on rnic-nn-fluid through the
sink: step 0 is `prefill2048`, steps 1..8 are `decode8x2048` (fixed
context 2048 by construction, so every decode step is identical). The
virtual clock advances by each step's makespan; sim-native metrics:

- D1: TTFT = the first (prefill) step's latency = B6 = 42,318,915,840 ps,
  exact to 0 ps.
- D2: every decode step's latency (hence TPOT, the mean decode-step delta
  on the virtual clock) = B3 = 2,485,904,640 ps, exact to 0 ps, all 8
  deltas identical.

## Check E: replay of the recorded adapter smoke JSONLs

Replay `/data3/yifeng/simllm-dev/m2-smoke-steps-v2.jsonl` (vLLM, 8
records) and `/data3/yifeng/simllm-dev/m3-smoke-steps.jsonl` (SGLang, 9
records), schema `atlahs-closed-loop-step-v1`, through the sink with a
declared tp=8 x pp=1 x dp=1 manifest (`simllm.placement.declared_manifest`)
and the TP=8 dims above, on rnic-nn-fluid. No closed form is registered
for these makespans (the recorded token counts vary per step); the
registered properties are:

- E1: the recorded virtual times are strictly monotonically increasing
  within each file, and every replayed step returns a StepResult with
  completed_at_ps = record.virtual_time_ps + step_latency_ps.
- E2: every record in both files schedules at least one token (verified in
  the M2/M3 smokes), so the sink simulates every step and each step's
  latency is strictly greater than the compute-only whole-step estimate E
  for that record (the network adds 64 allreduces of >= 14 * 2 us each,
  which dwarfs the <= L * 1000 ps flooring loss of the calc split).
- E3 (report only): the network share of each step, defined as
  1 - (L * c * 1000) / makespan, is reported per step for both frontends;
  no band is registered.
