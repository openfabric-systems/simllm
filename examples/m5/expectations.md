# M5 MoE slice: pre-registered expectations

Written and frozen before any M5 simulation run (repo validation rule:
expectations first, then measurements; every deviation is explained or is
a bug). All numbers below are closed-form arithmetic evaluated before the
first `htsim_rnic` invocation of this milestone; nothing here is fitted to
a measurement. The harness (`run_m5.py`) compares measured values against
the frozen constants of this file, never against runtime-derived values
(the M4 audit rule).

## Reused M1 calibration constants (not rederived)

From examples/m1/RESULTS.md (C1, C2, C3), all confirmed to 0 ps there:

- GOAL `calc` unit u = 1000 ps per unit (ns), with a 1 ns floor on
  `calc 0`.
- `rnic-nn-fluid`: wire factor h = 1.0, fixed propagation P = 2,000,000 ps.
  Wire time of a single full-rate flow at 400 Gbit/s is exactly 20 ps per
  payload byte.
- `rnic-nn`: h = 4160/4096, one wire slot = 83,200 ps at 400 Gbit/s, and a
  single chained flow completes at serialization + P + exactly one slot
  (the C3 store-and-forward intercept).

## Fluid sharing quantization (read from the backend source, not fitted)

M1 and M4 only ever shared a fluid port among n flows with n a divisor of
400e9 (n in {1, 2, 4, 8}), where the 20 ps/byte law is exact. A symmetric
pairwise all-to-allv shares each port among n = W - 1 flows, and 3 and 7
do not divide 400e9, so the exact integer arithmetic of the fluid manifold
becomes visible. From the backend source (registered here before any run):

- `rnic_max_min_allocator.cpp` computes the exact rational max-min water
  level and then floors each flow's grant to whole bps
  (`floorToRateBps`): rate(n) = floor(400e9 / n) bps.
- `rnic_fluid_manifold.cpp` tracks a service debt of
  `size_bytes * 8 * 1e12` bit-picoseconds and completes a flow after
  `ceil(debt / rate)` whole ps; delivery adds the fixed propagation P.

So the registered fluid completion of one flow of S bytes granted a
1/n share of a 400G port is

    T_share(S, n) = ceil(S * 8e12 / floor(400e9 / n)) + P   [ps]

which reduces to S * n * 20 + P exactly when n divides 400e9. For
n in {3, 7} at the sizes below the floor/ceil pair contributes exactly
+1 ps; the predictions below carry it.

## Check A: standalone pairwise all-to-allv on rnic-nn-fluid (exact to 0 ps)

Pattern source (`simllm.traffic.patterns.pairwise_all_to_allv`) dependency
structure, read before registration: for every ordered pair (s, d) with
s != d it emits one send on s and one recv on d, with NO dependencies
inside the phase (only the optional `after` chaining, unused here), so all
W(W-1) flows release simultaneously at t = 0; each rank's completion label
is its last-in-iteration-order recv, which is irrelevant here because the
symmetric phase makes all completions tie. Every NIC sends exactly W - 1
equal flows and receives exactly W - 1 equal flows, so the max-min water
level is B / (W - 1) at every endpoint and every flow gets it. With
per-pair payload S:

    JCT_a2av(S, W) = T_share(S, W - 1)
                   = ceil(S * 8e12 / floor(400e9 / (W - 1))) + 2,000,000

| Check | S bytes/pair | W | rate bps | predicted JCT ps |
|---|---|---|---|---|
| A1 | 65,536 | 2 | 400,000,000,000 | 3,310,720 |
| A2 | 65,536 | 4 | 133,333,333,333 | 5,932,161 |
| A3 | 65,536 | 8 | 57,142,857,142 | 11,175,041 |
| A4 | 16,777,216 | 2 | 400,000,000,000 | 337,544,320 |
| A5 | 16,777,216 | 4 | 133,333,333,333 | 1,008,632,961 |
| A6 | 16,777,216 | 8 | 57,142,857,142 | 2,350,810,241 |

Bar: measured JCT equals the prediction with 0 ps residual on every cell
(the fluid backend is deterministic and exact). Passing closes the
pairwise-a2av part of TRAF-4.

## Check B: MoE step makespan on rnic-nn-fluid (exact to 0 ps)

System under test: `HtsimStepSink` with MoE dims and `ep_ranks`, TP world
of 1 (EP-only step). Per layer the GOAL is: one `calc c` on every EP rank,
then the dispatch a2av, then the combine a2av, phases strictly chained
(TRAF-7/TRAF-9 serialization; BACK-5 even calc split; BACK-6
num_sampled = number of scheduled requests). Both a2av phases are the
symmetric pattern of check A with per-pair payload

    S_pair = total_new_tokens * top_k * hidden * dtype_bytes / W

(uniform routing, TRAF-2 first half; the 1/W self-share stays off the
fabric). All flows of a phase tie in the fluid manifold, so the phases
chain at integer ps and

    makespan = L * c * 1000 + 2L * JCT_a2av(S_pair, W)

Declared geometry (granite-3.0-1b-a400m-like MoE, per EP rank): L = 24
layers, hidden 1024, 16 heads x 64, 8 KV heads, vocab 49152, activation
dtype 2 B, num_experts 32, top_k 8, moe_intermediate_size 512 (per
expert), local_num_experts = 32 / W (each EP rank owns an equal slice;
the dense `intermediate_size` field is set to 512 and unused under MoE).
GPU envelope b100 (1.8e15 FLOP/s, 8e12 B/s), roofline efficiency 0.7,
host model 0. EP group = GOAL ranks 0..W-1.

The whole-step roofline estimates E (pure arithmetic on the declared
dims, frozen here so the grid cannot be fitted; c = floor(E / (L * 1000))):

| Check | shape | W | flops | bytes | E ps | bound | c ns |
|---|---|---|---|---|---|---|---|
| B1 | decode8x2048 | 2 | 8,454,930,432 | 2,264,924,160 | 404,450,742 | memory | 16,852 |
| B2 | decode8x2048 | 4 | 8,454,930,432 | 1,660,944,384 | 296,597,211 | memory | 12,358 |
| B3 | decode8x2048 | 8 | 8,454,930,432 | 1,358,954,496 | 242,670,445 | memory | 10,111 |
| B4 | prefill2048 | 2 | 1,752,447,320,064 | 1,560,281,088 | 1,390,831,206 | compute | 57,951 |
| B5 | prefill2048 | 4 | 1,752,447,320,064 | 956,301,312 | 1,390,831,206 | compute | 57,951 |
| B6 | prefill2048 | 8 | 1,752,447,320,064 | 654,311,424 | 1,390,831,206 | compute | 57,951 |

Step shapes (as in M4): `decode8x2048` is 8 requests, 1 new token each,
context 2048 (total 8 tokens, num_sampled 8); `prefill2048` is one
2048-token prefill chunk at context 2048 (num_sampled 1). Decode is
memory-bound (resident expert weights stream once per step and shrink
with W), prefill is compute-bound (E identical across W by construction:
the flops do not depend on expert placement).

Predicted makespans:

| Check | shape | W | S_pair | a2av phase ps | predicted makespan ps |
|---|---|---|---|---|---|
| B1 | decode8x2048 | 2 | 65,536 | 3,310,720 | 563,362,560 |
| B2 | decode8x2048 | 4 | 32,768 | 3,966,081 | 486,963,888 |
| B3 | decode8x2048 | 8 | 16,384 | 4,293,761 | 448,764,528 |
| B4 | prefill2048 | 2 | 16,777,216 | 337,544,320 | 17,592,951,360 |
| B5 | prefill2048 | 4 | 8,388,608 | 505,316,481 | 25,646,015,088 |
| B6 | prefill2048 | 8 | 4,194,304 | 589,202,561 | 29,672,546,928 |

Bar: 0 ps residual on every cell, and the runtime roofline estimate and
calc split must equal the frozen E and c per cell (reported as explicit
match columns; a mismatch is a FAIL even if the makespan residual is 0).

## Check C: registered qualitative directions

- C-q1 (a2av time vs EP width at fixed total payload): within each step
  shape the total per-rank payload `total_new_tokens * top_k * hidden *
  dtype` is fixed and S_pair = total / W, so the per-phase wire term is
  (W-1)/W of the total serialization and grows with W while P stays per
  phase. Registered: the measured network component of the check-B
  makespan (makespan minus the frozen L * c * 1000) increases strictly
  monotonically in W for both shapes.
- C-q2 (fluid vs nn ordering): the standalone check-A grid rerun on
  `rnic-nn` must satisfy measured_nn >= measured_fluid on every cell
  (packetization and store-and-forward can only add time on the
  null-network manifold, as in M1 C7 and M4 C). The raw nn values are
  reported; no exact nn closed form is registered for contending a2av
  flows (M1 C7 saw final-packet pipelining of a few slots under
  contention, so a point form would be a guess; deriving and validating
  one is left with the binomial-broadcast remainder of TRAF-4).
- C-q3 (implied by the frozen B table, checked on measurements): the
  decode makespan decreases strictly monotonically in W (resident-expert
  weight streaming dominates and shrinks with W faster than the a2av term
  grows), while the prefill makespan increases strictly monotonically in
  W (compute-bound c is W-invariant and the a2av term grows).
