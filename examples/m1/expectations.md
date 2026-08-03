# M1 sanity studies: pre-registered expectations

Written before any sweep was run (repo validation rule: expectations first,
then measurements; every deviation is explained or is a bug).

Notation: payload S bytes per flow, endpoint link capacity B bits/s, wire
factor h (bytes on the wire per payload byte), fixed propagation P ps, calc
cost C in GOAL units with unknown unit u ps/unit (resolved by probe P1),
worker count W. Wire time of one flow at full rate:

    T(S, B, h) = S * h * 8e12 / B   [ps]

Known model constants (htsim_rnic defaults): max wire packet 4160 B with
64 B header (so packetized h = 4160/4096 = 1.015625), rnic-nn propagation
P = 2e6 ps, rnic-cn margin = 0.9, control packet 64 B, generated Clos hop
latency 1e6 ps per hop.

## Probes

- P1 (calc unit): 2-rank echo with C in {0, 1e6}. Prediction: delta JCT =
  1e6 * u with u = 1000 (calc units are ns) or u = 1 (ps). No other value
  is acceptable.
- P2 (wire factor and offset): single flow, fluid and nn, S in {1 MiB,
  4 MiB} at B = 400G. Prediction: FCT is affine in S; the slope resolves h
  (fluid: h = 1 or 1.015625, undetermined pre-run; nn: h = 1.015625), the
  intercept resolves the constant offset (expected P = 2e6 ps, plus at most
  one packet slot).

## Workload A: scatter, compute, gather (validation workload)

Root 0 scatters S to each of W workers, workers calc C, workers send S
back. All scatter sends release simultaneously, so W flows share the root
egress; the gather is a W-to-1 incast into the root ingress.

Fluid manifold (exact max-min, no packetization):

    JCT_fluid = 2 * (W * T(S, B, h_fluid) + P) + C * u

1. Bandwidth sweep (W = 4, C = 0, S = 1 MiB, B in {100, 200, 400} G):
   JCT is linear in 1/B through the fitted intercept 2P; fitted slope =
   2 * W * S * h * 8e12 within 2 percent. Fluid is deterministic, so the
   linear fit must be essentially exact (relative residuals < 1e-6).
2. Parallelism sweep (B = 400G, C = 0, W in {2, 4, 8}): JCT linear in W;
   slope = 2 * T(S, B, h); intercept = 2P.
3. rnic-nn (packetized): same shape; JCT_nn >= JCT_fluid at equal h with
   the excess bounded by per-flow slot quantization, and h_nn = 1.015625
   exactly. Normalized per-flow FCT nn/fluid in [1.0, 1.03].
4. rnic-cn on a generated 32-node ns-tm3 Clos (K = 8): every normalized
   per-flow FCT (cn/nn, identical GOAL) >= 1. Scatter flows (1 flow per
   receiver, n_hat = 1) are granted margin * B and bottlenecked by the
   sender port. Gather flows (W-to-1 incast, n_hat = W at the root) are
   granted margin * B / W each, so for large S the gather-phase normalized
   FCT approaches 1/margin = 1.111...; we accept [1.05, 1.25] at
   S = 1 MiB (control handshakes and Clos hops add picoseconds that do not
   fully amortize). Monotonicity in W and B must match the nn direction.
   Additive control latency: at least one DECLARE/ACCEPT round trip before
   the first data byte, so cn JCT minus nn JCT >= 2 hops of latency even
   as S -> 0.

## Workload B: pipeline-parallel decode (signature metrics)

Default reference config: 8 nodes x 8 B100, one 400G NIC per GPU, 64 ranks
(gpu-rank), two-tier Clos (`topologies/clos_64_400g.topo`). A 70B-class
model runs PP = 8 (one stage per node) x TP = 8 (intra-node NVLink, off
fabric). Fabric traffic is only the stage-to-stage activation handoff on
one NIC rank per node: prefill activations S_pre = tokens * hidden * 2
bytes, decode activations S_dec = hidden * 2 bytes, hidden = 8192,
tokens = 1024.

Per-stage compute from the roofline provider (declared B100 envelope:
1.75e15 FLOP/s BF16, 8e12 B/s HBM, efficiency 0.7): prefill is
compute-bound, decode is memory-bound (weight streaming of
params * 2 / (PP * TP) bytes per GPU).

    TTFT = sum over 8 stages of (C_prefill * u + T(S_pre, B, h) + P + hops)
           + (first decode chain)
    TPOT = mean over decode steps d of (t_d - t_{d-1})
         = 8 * (C_decode * u + T(S_dec, B, h) + P + hops)

Predictions:
5. TPOT is deterministic and matches the closed form within 2 percent on
   fluid/nn; decode compute dominates (C_decode ~ 273 us per stage vs
   ~ 21 us of network per hop at S_dec = 16 KiB), so TPOT ~ 2.4 ms for the
   declared envelope.
6. Doubling B changes TPOT by less than 1 percent (decode is
   compute-dominated: qualitative "slightly changed"), but changes the
   network share of TTFT visibly (prefill activations are MiB-scale:
   "multiple changes"). This is the two-parameter qualitative matrix the
   validation rule asks for.
7. cn vs nn on the identical workload-B GOAL: chained single flows never
   contend, n_hat = 1 throughout, so normalized FCT stays in [1.0, 1.05]
   and TPOT_cn - TPOT_nn is a small additive control overhead per hop.
