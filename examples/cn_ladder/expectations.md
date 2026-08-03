# rnic-cn incast ladder: pre-registered expectations

Written before any post-surgery run (algorithm-book implementation of
D1-D4 plus sender-side own-fraction rate scaling). Validation protocol per
the maintainer: step by step, fan-in 1, 2, 4, 8, 16, 32, 63 into one
receiver (63 is the 64-node topology's maximum; stands in for the
requested 64) crossed with sizes 16 MiB down to one MTU worth of payload
(4096 B), rnic-cn versus rnic-nn on the identical GOAL; only after this
ladder holds does the mixed-size all-to-all phase run.

Notation as in examples/m1: T(S) = wire time of one flow at full rate with
h = 4160/4096, P = 2 us propagation (rnic-nn), margin = 0.9, dwnd = the
one-way control deadline K (default 10 us, calibrated 4.5 us).

## Bars (maintainer, 2026-08-03)

- Hard bar: every per-flow FCT within 2x of rnic-nn.
- Target: within 20 percent (slowdown <= 1.2) once the short-flow
  algorithm works, judged on the additive-invariant lens for cells where
  wire time is small against fixed offsets.

## Expected shapes

1. Fan-in 1 (single flow): cn = declare-and-go, so FCT ~ T(S)/0.9 + P +
   (control serialization + Clos residency, sub-us). Slowdown ~ 1.11 at
   16 MiB decaying toward an additive few-us excess at 4 KiB; per-flow
   slowdown <= 1.2 for S >= 64 KiB.
2. Bulk incast (S well above the fractional budget): every flow declares
   whole, n_hat = W. Steady per-flow rate margin * C / W; aligned starts;
   FCT ~ W * T(S)/0.9 + P + settle, settle <= one round trip of feedback.
   Slowdown -> 1/0.9 = 1.11 as S grows, for every W. No lease machinery
   exists any more, so no expiry race at any K; the calibrated K = 4.5 us
   run must be deterministic and quiescent (5x identical).
3. Small-S incast (S below the fractional budget): flows declare
   fractionally; with own-fraction scaling the aggregate stays
   margin * C, so the fabric never oversends; per-flow FCT is dominated
   by P plus the sub-us control path, hence the additive excess over nn
   must be flat in S and bounded by a few microseconds, and must NOT grow
   with W (the receiver's control replies serialize on its uplink, 64 B
   each, 1.28 ns apiece at 400G: negligible).
4. Monotonicity: at fixed S, FCT increases with W (both models, same
   direction); at fixed W, FCT increases with S.
5. Cross-K: results at K = 10 us and K = 4.5 us differ only through the
   fractional budget threshold (which flows declare fractionally) and
   startup-window granularity; no cell may regress across the 2x bar
   because of K alone.

## Disqualifiers (any one is a stop-and-debug)

- Ring-CAM late-admission or GAP-NACK counts exploding at small S
  (would indicate the own-fraction scaling or dwnd accounting is wrong).
- Additive excess growing with decreasing S, or with W at fixed small S.
- Any nondeterminism across 5 repeats of the same cell.
- Any cell above 2x.
