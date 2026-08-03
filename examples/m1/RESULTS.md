# M1 sanity studies: results against pre-registered expectations

Runs of 2026-08-03, htsim submodule at `09b7c7cc`. All sections ran in one
`run_m1.py` invocation (60 rows in `summary.csv`, per-run CSVs beside it);
the backends are deterministic, so every number below reproduces exactly.
Checks C1-C18 enumerate the registered predictions of
[expectations.md](expectations.md) (kept frozen; deviations are disclosed
here, never edited there). An independent re-derivation audit was run
against this document; its corrections are incorporated below.

Verdict: **15 of 18 checks pass; 3 registered expectations are falsified
(C11, C15, C18), each traced to a mis-registration (findings F1-F3), none
to a simulator or simllm defect.** All six fluid workload-A configurations
and all four fluid/nn workload-B runs reproduce their closed forms with
zero picosecond residual (for workload B under the corrected forms
disclosed at C13/C14).

## Probes

- C1 (calc unit): delta JCT = 999,999,000 ps for delta C = 1e6 on both
  profiles: u = 1000 ps per unit (ns), with a 1 ns floor on `calc 0` (the
  registered "delta = 1e6 * u" is off by exactly that floor). Pass.
- C2 (fluid constants): FCT = 20.000000 ps/byte * S + 2,000,000 ps:
  h_fluid = 1.0, P = 2 us, exact. Caveat: two sizes cannot falsify
  affinity by themselves; the three-point 1/B sweep in C4 does.
- C3 (nn constants): slope 20.3125 = 20 * 4160/4096 (h_nn = 1.015625
  exact); intercept = P + 83,200 ps = P + exactly one 4160 B wire slot at
  400G. Pass.

## Workload A: scatter, compute, gather

- C4 (fluid bandwidth sweep) and C5 (fluid parallelism sweep):
  JCT = 2(W*T + P) + 1000 ps matches all six configurations with **0 ps
  residual** (e.g. W=4/400G: 171,773,160 predicted and measured; B=100G:
  675,089,640; W=8: 339,545,320). Slope vs 1/B = 2*W*S*h*8e12 exactly;
  W-slope = 2T exactly; intercepts = 2P + 1 ns. Pass.
- C6 (nn wire factor): h_nn = 1.015625 exact in the sweep slopes. Pass.
- C7 (nn quantization): deviations from the naive 2(W*T*h + P) + 1 ns form
  are exactly {+1, -1, -5} slots for W = {2, 4, 8} (and -1 slot at 100G
  and 200G), i.e. <= 0.121 percent. JCT_nn >= JCT_fluid everywhere. The
  negative deviations come from final-packet pipelining across concurrent
  flows (bounded by W-1 slots); the +1 at W=2 mirrors the single-flow
  store-and-forward slot of C3. Mechanism partially characterized; the
  magnitudes are exact in slot units. Pass within the registered bound.
- C8 (nn/fluid per-flow band [1.0, 1.03]): measured [1.0037, 1.0168]. Pass.
- C9 (cn scatter): every normalized per-flow FCT >= 1; ranges
  W=2 [1.5373, 1.5477], W=4 [1.2055, 1.2828], W=8 [1.0073, 1.1567],
  decreasing with W as the fixed ~10 us control handshake amortizes. Pass.
- C10 (cn JCT monotonicity): 152.2 / 251.9 / 442.9 us for W = 2/4/8, same
  direction as nn. Pass.
- C11 (cn gather per-flow band [1.05, 1.25]): **FAIL, finding F1.**
  Measured per-flow slowdowns: W=2 [1.4721, 1.8613], W=4 [0.8487, 1.6147],
  W=8 [0.4305, 1.4180]. The raw flows show why: collective membership is
  dynamic, so early joiners run at large grants (n_hat small; first flow
  FCT 73.36 us) while late joiners see n_hat = 8; the nn baseline instead
  starts all flows one slot apart and serves them fairly. Per-flow
  normalization is ill-posed when the two models stagger starts
  differently; the metric is now scoped in `simllm.backends.fct`. The
  reviewable quantity is the phase makespan (first start to last
  completion): cn 269.711 us vs nn 172.477 us, ratio 1.5638. Ledger:
  ideal quantized-grant incast 8*T*h/0.9 = 189.326 us plus up to one 10 us
  stale-grant window after each of the 8 retirements bounds the makespan
  at 269.326 us; measured exceeds the bound by 0.385 us (0.14 percent),
  plausibly Clos hop latency or handshake serialization, and that residual
  is unattributed. The ledger closes to 0.14 percent, not byte-exact.
  Disposition: mis-registered metric, effect real and explained.

## Workload B: pipeline-parallel decode (signature metrics)

- C12 (roofline classification): prefill compute-bound (1.829 ms/stage),
  decode memory-bound (390.625 us/stage) on the declared envelope. Pass.
- C13 (TPOT, registered bar: 2 percent): the registered form
  8*(C_dec + T(S_dec) + P) describes 8 activation hops, but the executed
  GOAL has 7 activation hops plus one 64 B token hop closing the loop.
  Registered form: 3,143,621,440 ps at 400G, +0.0104 percent vs
  measurement (inside the bar). Corrected form
  8*C_dec + 7*(T(S_dec) + P) + (T(64 B) + P) matches measurement with
  **0 ps residual** at both bandwidths (400G: 3,143,295,040; 800G:
  3,142,147,520), and rnic-nn adds exactly its header bytes plus one slot
  per message (decomposition verified to 0 ps). Pass, with the form
  correction disclosed.
- C14 (TTFT, same bar): registered form counts 8 S_pre hops plus a first
  decode chain; the executed GOAL's first token completes after 7 S_pre
  hops and the token hop, with no decode compute. Corrected form matches
  with **0 ps residual** at both bandwidths (16,993,379,520 and
  15,818,973,760). Pass, correction disclosed.
- C15 (registered numeric anchors): **FAIL, finding F3.** The registered
  "C_decode ~ 273 us", "~21 us of network per hop", and "TPOT ~ 2.4 ms"
  are pre-registration arithmetic slips: 273 us omits the declared 0.7
  efficiency derate (correct: 390.625 us); per-hop network is 2.33 us
  (0.33 us wire + 2 us propagation), not 21; hence the anchor TPOT is 31
  percent below the measured 3.143 ms. The slips are in the registration,
  not the model; the closed forms built from the declared envelope pass
  (C13). Registered anchors falsified and disclosed.
- C16 (two-parameter qualitative matrix): doubling B moves fluid TPOT by
  -0.0365 percent ("same / slightly changed", decode compute-dominated)
  and TTFT by -6.911 percent ("multiple changes", MiB-scale prefill wire
  time halves). Pass, exactly the registered shape.
- C17 (cn additive overhead): TPOT_cn - TPOT_nn = 228,466,240 ps = 28.56
  us per hop of additive control overhead on chained, non-contending
  flows. Matches the registered additive form. Pass.
- C18 (cn per-flow band [1.0, 1.05]): **FAIL, finding F2.** All 71 flows
  sit outside the band: prefill (16 MiB) [1.1876, 1.1947], decode
  activations (16 KiB) [12.8181, 12.8883], token (64 B)
  [14.5311, 14.5359]. The single model
  FCT_cn ~ T_wire/0.9 + P + K with K ~ 28.5 us of per-flow control
  handshake fits all three classes within about 1 percent (prefill:
  (340.8/0.9 + 2 + 28.5) / 342.9 = 1.20; decode:
  (0.37 + 2 + 28.5) / 2.42 = 12.8). The band was registered as
  multiplicative, but the overhead is additive, so the ratio diverges as
  the flow shrinks; the additive form is the invariant to register.
  Chained flows are aligned-start, so the per-flow metric itself is valid
  here (unlike F1); only the band shape was wrong.

## Findings

- F1 (C11): per-flow FCT normalization is ill-posed across models that
  stagger flow starts differently (dynamic collective membership vs slot
  calendar); use phase makespan ratios there. Metric scoped in
  `simllm.backends.fct`.
- F2 (C18): control overhead is additive per flow (~28.5 us here), so
  normalized-FCT bands for small flows must be registered in additive
  form; multiplicative bands only make sense when T_wire dominates.
- F3 (C15): registration arithmetic must be ledgered like everything
  else; two slips (dropped efficiency derate, per-hop estimate off 9x)
  produced anchors the correct closed forms could never hit.

Grant-adaptation lag (C11) and additive handshake overhead (C17/C18) are
exactly the effects a bytes/bandwidth or 1/margin estimate cannot
represent; surfacing them with closed ledgers is what this pipeline is
for.
