# Routed-MoE byte conservation v1 results (VLLM-24)

Expectations were frozen in commit `20f6017` and amended in commit `1a4db9b`,
both before the study harness and every result-producing run. The measured run
below observed repository commit `88a57f7` with a clean working tree. The
evidence was authored against vLLM v0.26.0 commit
`568afb3a13806beb53bb2e6bd518269357b237c0`; no check requires the observed
package to equal it.

## Chronology, stated exactly

1. `20f6017` froze nine named conservation rules, the fault model, the cells
   and the closed-form bounds. Its check-only dry run produced no artifacts.
2. The guard was implemented in `dc8249c`. Its first pass over the existing
   test suite refuted two frozen rule STATEMENTS: both owner rules were written
   as if every routed byte leaves the owner, which is false for combine.
3. `1a4db9b` amended those two statements additively, before the study harness
   existed and before any result-producing run. Nothing about the fault model,
   the cells, the bounds or expectations E1 through E6 changed.
4. `88a57f7` added the harness. The run below is the only result-producing run.

Two harness assertions were corrected after a first execution, and neither
touched an expectation. The arm-A guard read the report's cross-phase
`source_ranks` field where the amended rule is phase-aware, and the corrupted
observation cell never actually corrupted anything because its guard variable
was the accumulating result list. Both were harness bugs; the frozen
statements they were meant to test are unchanged.

## Verdict

**VLLM-24 closes.** The run is not void: all 8 fatal guards held. Scored
behavioral instances: **5/5**, against a frozen denominator of 9. The missing 4
were not failures, they were not executed, because the frozen capture cannot
build them; see "What did not run" below.

## Physical sanity, checked before the digits

Floors and ceilings were stated in the freeze and are re-derived by the
harness from the record and the geometry alone.

| Cell | arm A hops | frozen floor | frozen ceiling | inside |
|---|---|---|---|---|
| `prefill/W2` | 1056 | 0 | 1056 | yes |
| `prefill/W8` | 5332 | 1056 | 7392 | yes |

`prefill/W2` sits exactly on its ceiling, which says something concrete: with
`top_k = 8` selected from 32 experts split into two blocks of 16, every one of
the 22 tokens has at least one selected expert on the peer rank, on all 24
layers, in both phases. `prefill/W8` sits at 5332 of a possible 7392, i.e. a
mean of 5.05 distinct remote owners per token-layer out of a maximum of 7.

Scaling companion: arm A moves 2,162,688 bytes at `W = 2` and 10,919,936 bytes
at `W = 8`. The ratio 5.05 is above 1, as required: a wider expert world can
only split a token's experts over more owners, never fewer.

## Fatal unscored guards, 8 of 8 held

| Guard | Cells | Held |
|---|---|---|
| `arm-a-conserves` | 4 | yes |
| `replication-multiplier` | 4 | yes |

`arm-a-conserves` required zero violated rules, a dispatch source set of
exactly `{0}`, and owner egress equal to owner ingress. `replication-multiplier`
required `hops_B(W) == W * hops_A(W)` and the same identity in bytes, so that a
detection result is attributable to the registered fault and not to some other
perturbation. Every cell satisfied both exactly:
`2112 = 2 * 1056`, `42656 = 8 * 5332`, `96 = 2 * 48`, `1968 = 8 * 246`.

## Scored behavioral instances, 5 of 5

| Family | Cell | Expectation | Observed | Pass |
|---|---|---|---|---|
| E4 | `prefill/W2` | `step-hop-bound` does NOT detect the replication | not detected | yes |
| E5 | `prefill/W2` | `source-attribution` detects it | detected | yes |
| E3 | `prefill/W8` | `step-hop-bound` detects the replication | detected | yes |
| E5 | `prefill/W8` | `source-attribution` detects it | detected | yes |
| E6 | `granite-marker-at-w8` | the semantic marker is named and the guard runs anyway | see below | yes |

The quantitative content of E3 and E4, in the numbers the run produced:

- `W = 8`: the step hop bound is `22 * 8 * 24 * 2 = 8448` hops. Arm B emitted
  42,656 hops, i.e. 5.05x the bound. Detected with a factor of five to spare.
- `W = 2`: the same bound is 8448 hops. Arm B emitted 2112, i.e. 0.25x the
  bound. Not detected, and not detectable: `hops_A(2) <= 1056` holds for any
  routing whatsoever at a two-rank world, so `2 * hops_A(2) <= 4224 < 8448`
  always. The bound has no power at EP world 2 by construction.

This is the whole reason the identity is evaluated at EP world 8. A run of this
study restricted to EP world 2 would have reported the replicated arm as clean
on the one rule that generalizes, which is exactly how the historical 8x defect
survived.

### E6 in detail

At EP world 8, the Granite observation producer's routed sites report
`evidence_mode = "no-byte-evidence"`: all 48 of them are zero-byte semantic
markers carrying no pair table. Lowering those observations with a captured
routed supply still bound all 48 sites to the traffic plan's byte tables and
still ran the conservation guard on the full-step plan. Separately, an
observation whose pair table disagrees with the plan is still rejected with a
`ValueError` exactly as before. The marker path is therefore an explicit
no-byte-evidence mode that neither satisfies nor substitutes for the byte
guard.

## What the run showed beyond its own expectations

The captured-routing-only rules are strictly stronger than the step hop bound
and caught the replication at BOTH worlds:

| Cell | rules violated by arm B |
|---|---|
| `prefill/W2` | `source-attribution`, `per-layer-hop-bound`, `per-request-hop-bound` |
| `prefill/W8` | the same three, plus `owner-egress` and `step-hop-bound` |

Two things follow. First, `owner-egress` fires only at `W = 8`: with two ranks
every directed row still touches rank 0, so the phase-agnostic rule has no
signal there while the phase-aware one does. Second, the per-layer bound would
have caught the historical defect at either world, but it needs deduplicated
captured routing and is therefore unavailable on the uniform destination
approximation. `step-hop-bound` is the only rule that applies to every routed
representation, and it needs a wide expert world to have any power. That is a
sharper statement of the freeze's claim than the freeze itself made.

## What did not run

The frozen `decode` cell (a DECODE step with one new token, 4 of the 9 scored
instances) was not executed. The frozen capture
`examples/preplay_trace_v1/granite_length_cap.jsonl` carries 22 prefill tokens
and zero decode tokens, so a DECODE step cannot be built from it at all. The
freeze did not check that property of its own input. This is a coverage gap in
the freeze, not a failed expectation, and it is reported as not executed rather
than folded into a pass fraction.

A post-specified substitute was run and is labeled as such in the report: a
one-token PREFILL chunk, `prefill-chunk-1`, whose token count matches the
frozen decode cell's `T = 1` and therefore shares its closed-form bounds. It
behaved exactly as the frozen decode cell predicted: 48 hops against a
384-hop bound at `W = 2` with the replication undetected by the bound, and 246
hops against the same bound at `W = 8` with the replication detected at 5.1x.
That is post-specified regression evidence, not pre-registered evidence, and it
does not enter the 5/5.

## Genuine-risk analysis

All 5 scored instances are genuine risk. None is entailed by an earlier fatal
oracle: `arm-a-conserves` constrains arm A only, and `replication-multiplier`
fixes the fault's byte multiplier without saying which rule notices it or
whether `W * hops_A(8)` clears the bound, which depends on the captured routing
distribution that the freeze did not fix. E4's direction is a first-principles
certainty about the bound, but its instance still tests whether the
implementation computes that bound correctly.

E4 is a deliberately negative expectation: it is scored as a pass for NOT
detecting the fault. That is not a weakened bar, it is the counter-cell that
gives the E3 result its meaning, and its outcome was frozen before the run.

## Registered IDs

Zero new IDs. Every registered VLLM-24 acceptance clause is demonstrated:
source-observed token ownership crosses the adapter seam as the record's
per-request token counts plus the declared engine rank; exact agreement is
required for source rank, destination rank, request identity, per-source
egress and total directed bytes; and the semantic-marker path is now an
explicitly named no-byte-evidence mode that does not satisfy the guard. The
unexecuted decode cell belongs to this study's own freeze, not to a registered
acceptance clause, so it is recorded here in prose rather than as a new ID.

Raw report: `report.json` under the wave-10 run root, outside the repository.
