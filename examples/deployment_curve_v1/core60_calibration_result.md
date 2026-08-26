# CORE-60 calibration-only result, protocol void

This record is **VOID** for CORE-60 acceptance. The official SGLang page used
for the overlap citation rendered its evaluation table during source
inspection and exposed the forbidden 2K and 4K prefill rows. Neither number
entered the mechanism, packet derivation or visible-row calculation, but the
literal no-held-out-access clause is not met. The physical result remains
useful evidence and is not promoted to an accepted calibration.

## Adopted physical contracts

1. **Per-rank token ownership is already correct.** The deployment projection
   fixes 16,384 new tokens on each EP32 rank and 131,072 routed expert visits
   per rank per layer. CORE-60 does not rescale that population.
2. **Dispatch is FP8 plus scales and combine is BF16.** Pinned SGLang commit
   `bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3` selects FP8 DeepEP dispatch,
   quantizes at group size 128 with float32 H100 scales, and produces BF16
   expert output for combine. The frozen wire vectors are therefore 7,392 and
   14,336 bytes per token-destination, respectively.
3. **Routing deduplicates expert selections on the same destination.** DeepEP
   dispatch consumes `is_token_in_rank`; under the frozen uniform top-8
   assumption, a destination is hit with probability
   `1 - C(248, 8) / C(256, 8)`. This gives 7.266799917 expected unique
   destinations per token and 7.039712420 expected remote destinations.
4. **Two-batch overlap composes compute and communication by a maximum.** The
   pinned SGLang prefill strategy splits two children and interleaves stages at
   explicit dispatch and combine yield points. The only hiding capacity is the
   independently measured 1,363,249,960,000 ps candidate compute component;
   no overlap fraction is fitted.

The detailed file-and-line citations, expected signs and rounding envelopes
were committed in the expectations-only freeze before this comparison. The
[SGLang large-scale EP report](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)
also identifies two-batch overlap of attention and MLP compute with DeepEP
dispatch and combine in this EP32 prefill configuration.

## Composed service

The selected ordered-pair floors are 27,502,686 dispatch bytes and 53,338,543
combine bytes. The existing four-node placement and pinned `rnic-nn` backend
produce 13,410,556,120 ps dispatch and 26,006,336,300 ps combine service per
MoE layer at 400 Gbit/s. Across 58 layers, communication is
2,286,179,760,360 ps and dominates candidate compute, so the max-composed total
is also 2,286,179,760,360 ps. The exposed increment above candidate compute is
922,929,800,360 ps.

The adjacent-byte rounding envelope changes the point total by only 2,320 ps.
The 200 Gbit/s sensitivity total is 4,572,127,520,720 to
4,572,127,525,360 ps. All eight endpoint runs reproduced the freeze, conserved
the existing local/fabric placement split and reached quiescence. Compact
evidence is under
`$SIMLLM_CORE60_RUN_ROOT/mechanism-evidence-v1/mechanism-evidence.json`.

## Signed calibration-only movement

| Row | Published | Candidate only | CORE-59 | CORE-60 point | Movement from candidate | Movement from CORE-59 | Remaining error |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sglang_prefill_1k` | 57,674.0000 | 96,146.7111 | 27,982.1912 | 57,332.3245 | -40.3700% | +104.8886% | -0.5924% |
| `sglang_decode_standard` | 22,282.0000 | 8,949.7597 | 8,949.7597 | 8,949.7597 | exactly 0 | exactly 0 | -59.8341% |

The 1K row moves in both preregistered directions: down from candidate-only
and up from CORE-59. The max-composed physical model remains 341.675450
tokens/s/node low, an exact signed relative error of
`-9764143737533 / 1648164143737533` or -0.592425 percent.

## Honest remainder

No unmeasured scale factor is applied. The residual corresponds to a modeled
service surplus of `390565749501320 / 28837` ps, or 13,543,910,583.670979 ps.
That is 1.009943992 frozen dispatch phases, but numerical proximity is not
evidence for subtracting one phase.

TRAF-66 owns the exact remainder: capture the finite two-batch-overlap
prologue, steady-state stage interleave and epilogue in the pinned framework,
then determine which boundary service is exposed. It must freeze that trace
before using this visible residual and must reject a fitted boundary fraction.

## Access and preservation

The comparison accessed only `sglang_prefill_1k` and
`sglang_decode_standard` from the structured anchor freeze, so that forbidden
ID access list is empty. Separately, source inspection exposed the 2K and 4K
evaluation values, which voids the literal access clause. No held-out scorer
ran and the scored flagship was not rerun. All four CORE-59 freeze/result
artifacts and all nine first-run artifacts remain byte-identical. Decode
pricing is unchanged and SGL-38 remains its sole owner.

CORE-60 therefore remains open. COMP-75 owns a clean independent repetition
whose source inspection excludes every evaluation table before the freeze.
TRAF-66 owns the finite-overlap physical remainder if that clean repetition
reproduces it. No parameter was fitted and this void run is retained honestly.
