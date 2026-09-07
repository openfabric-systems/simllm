# Attention pair successor expectations

This expectations-only commit precedes implementation, extraction and study
execution for COMP-81. The repository has no root AGENTS.md in this worktree;
the supplied consequence-wave contract governs this study.

## Contract and immutable predecessors

`step_shape` is the sole pair-count authority. For each scheduled sequence
with n new tokens and context c including those tokens, its analytical pair
count is `n * max(c - n, 0) + n * n // 2`. Sum after rounding each sequence.
This retains the existing half-square causal approximation, including zero
pairs for an uncached one-token sequence. It is not the inclusive discrete
triangle `n * (n + 1) // 2`. Decode at context c contributes c - 1 pairs.

The attention invocation schema v2 appends `attention_pairs`, unit `pairs`,
to (`new_tokens`, `kv_tokens`). The inventory envelope remains v1; the
invocation schema v1 and every existing inventory remain readable and valid.
One query-key pair includes QK and PV, all query heads and all represented
layers: coefficient `4 * layers * query_heads * head_size`. GQA reduces KV
storage, not query-head arithmetic. Qwen3-32B therefore costs exactly
2,097,152 FLOPs per pair in both phases. A zero-pair row costs zero FLOPs
and cannot independently identify a coefficient.

The old inventories c8832ba8ba21e49517b6b74e89554c2abdb0d9e76530f647a7849f3f8448ec56
and 51740b52625002a964e75fddb679e9f8394a08a7d7c62556d2535c3bc60515e3,
their suite, and every file under examples/frontier_comparison_v1 remain
byte-for-byte immutable. Digests accept raw or LF-normalized tracked text
bytes on Windows. New records name both successor inventory digests, the old
result digest and this expectations commit. New text artifacts use LF bytes.

## Independent work prediction and physical bounds

Before reading any successor output, derive attention projection parameters
as `64 * (5120 * (8192 + 2 * 1024) + 8192 * 5120)` = 6,039,797,760,
MLP parameters as `3 * 5120 * 25600 * 64` = 25,165,824,000, and LM-head
parameters as `5120 * 151936` = 777,912,320. Thus linear FLOPs per new token
are 62,411,243,520 and sampling costs 1,555,824,640 FLOPs per request.

At the unchanged v1 point (3,500 uncached tokens, 4,250 average decode
context, 500 output tokens), successor whole-model decode work per batch
item is exactly 72,877,867,008 FLOPs: 63,967,068,160 fixed plus
4,249 pairs times 2,097,152 = 8,910,798,848. TP4 owns 18,219,466,752 FLOPs
per item and batch 64 owns 1,166,045,872,128 FLOPs per rank.

Successor whole-model prefill work per request is exactly
231,285,964,144,640 FLOPs: 218,439,352,320,000 linear plus
6,125,000 pairs times 2,097,152 = 12,845,056,000,000 attention plus
1,555,824,640 sampling. TP4 owns 57,821,491,036,160 FLOPs per request.
At 2,048 tokens, prefill is exactly 132,217,829,064,704 FLOPs, of which
4,398,046,511,104 (between 3 and 4 percent) is attention. At 3,500 tokens
the attention share must be between 5 and 6 percent. These shares follow
from quadratic attention versus linear projections, not timing fitting.

The first-principles service floor is max(F / (TP * 1.979e15),
B / (TP * 4.8e12)) seconds. The e=0.4 envelope ceiling is 2.5 times that
floor. Static matrix bytes remain 31,983,534,080 and KV bytes per context
token remain 262,144. Decode batch 64 TP4 moves 25,821,675,520 bytes, so
its floor is between 5.3 and 5.5 ms and ceiling between 13.2 and 13.8 ms;
its compute floor is between 0.58 and 0.60 ms. Prefill TP4 moves
8,225,259,520 bytes; the compute floor must be between 29 and 30 ms and
its ceiling between 72.5 and 75 ms, with memory below 1.8 ms. Weight
storage per rank stays below 10 GB and the declared 141 GB H200 capacity.
These are analytical envelope bounds, not silicon measurements.

## Frozen sweep, directions and shapes

Sweep both uncached prompt length U in {128, 512, 2048, 3500, 4096} and
decode context C in {128, 2048, 4250, 8192}, at TP in {2, 4, 8} and
efficiency in {0.6, 0.8, 1.0}. Report exact whole-model work and service
for batch 64 decode and batch 1 prefill. Work is independent of efficiency;
decode FLOPs grow affinely with C and prefill FLOPs are quadratic in U.
At fixed work service is nonincreasing with TP and efficiency, up to one
picosecond rounding. Floor <= service <= floor / 0.6 + 1 ps on these arms.

Retain v1's 32-GPU candidate family (TP 2, 4, 8 per role, worker splits,
batch ladder 1, 2, 4, 8, 9, 16, 20, 26, 32, 48, 56, 64, 96, 112, 128),
three efficiency arms, service targets and ten external rows. Successor
frontiers should remain monotone; matched throughput cannot increase when
only compute work increases. The prefill correction is modest: about 4.35
percent at 3,500 tokens relative to the old total. Decode work increases
by exactly 2,254,848 FLOPs per item because c - 1 replaces the old rounded
per-context coefficient; its memory-bound service is unchanged.

X2a remains PASS at approximately 5.379516 ms; X2c-decode remains PASS at
e-star approximately 0.586068. X2b remains PASS, moving from approximately
28.000527 to between 29 and 30 ms. X2c-prefill remains FAIL, moving from
approximately 0.142552 to between 0.147 and 0.153, below the unchanged
[0.40, 1.00] band. All four old and successor rows are reported together.
No changed verdict is required for COMP-81 acceptance.

## Fatal guards, exact oracles and evidence classes

Fatal guards: chronology and frozen expectation bytes; immutable predecessor
digests and external input digests; same model revision
aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df and 15 suite cases; canonical
new inventory filenames; two framework-neutral inventories exactly equal;
pair axes matching step_shape and 2,097,152 FLOPs per nonzero pair in all
30 framework/case rows; exact family FLOP and byte conservation against
the fused kernel; no subprocess in the estimator sweep; explicit evidence
classes and no external calibration installed. Config-only extraction uses
HF_HUB_OFFLINE=1, checkpoint-root and bulk-output options, no GPU or weights.
Retained framework configuration projections may supply the existing
extraction path without installing either framework in the study process.

No fatal guard is intended to be survivable for scoring. A failure voids
all behavioral and X-family scores, which become null, never a zero score;
retained raw diagnostics identify the failed invariant only. Malformed
inputs may fail closed before producing a result. Negative tests must prove
the contract rejects a wrong pair count and detects nonconservation.

Exact integer work and conservation are structural oracles. Trend predicates
and unchanged X2 bands are behavioral comparisons. Estimator outputs are
ESTIMATE with DECLARED H200 envelopes; archived external rows remain
MEASURED-EXTERNAL operation-database estimates, not end-to-end serving
measurements. No network, silicon or timing calibration is claimed. The
external TTFT operating-point versus isolated-service confound remains
DEPLOY-12's scope. Acceptance closes only COMP-81's pair contract and its
successor projection. Wall time is diagnostic, not a new precision score.
