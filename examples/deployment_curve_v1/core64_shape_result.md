# CORE-64 attention, MLA and shared-expert decode shape result

Status: **PASS_NULL_SHAPE_MOVEMENT_EXACT_REMAINDER**. The preregistered component-backed shape test
produces an honest null movement and retains the full standard-decode
undercorrection.

## Enumerated shape mismatches

1. **MLA and attention: none.** The disclosed nine-node, eight-rank-per-node
   standard decode layout has attention DP72. Its 2,304 global requests divide
   to `9 * 256 / 72 = 32` requests per rank. KV 2,000
   therefore produces 64,000 aggregate KV-token
   references per rank. This exactly matches the TP1 `b32/c2000` capture, so
   every MLA and attention scale remains one.
2. **Shared expert and dense feed-forward: none.** Each rank runs both paths
   for its own 32-request local stream. Decode dense DP1 is a singleton group;
   it does not replicate the node's 256 requests onto each rank. Both scales
   remain one.
3. **Router and LM head: none.** Both see the same 32 local tokens and retain
   scale one.
4. **Routed experts: no new CORE-64 mismatch.** CORE-63's exact `1/9`
   `fused_moe_kernel` residency scale remains unchanged.
5. **Fixed and other retained physical rows: none.** The 489 ps fixed term is
   kept once and every nonmatching noncollective row remains at scale one.
6. **Physical launch identity: unresolved attribution, not a numeric shape
   mismatch.** No total EP72 physical-kernel-to-logical-family binding exists.
   Because all non-routed logical scales are one, this cannot change the null
   arithmetic, but it prevents inventing a finer semantic timing attribution.
7. **MTP: absent and unread.** It is not part of standard decode.

These derivations use the [published deployment disclosure](../../docs/papers/deepseek-deployment-disclosures.md),
the [frozen SGLang DP72 and dense-DP1 arrangement](../sglang_pd_session_v1/expectations.md),
and the [framework-neutral DeepSeek family projection](../model_extraction_deepseek_v3_v1/RESULTS.md).

## Derived correction and signed movement

All 13 standard-decode logical families were enumerated; the
MTP family is separately absent. Shape mismatches: **0**. The inherited
CORE-63 step therefore stays **26,821,286,365
ps**.

```text
CORE-64 prediction movement = 0.000000 tokens/s/node
final standard-decode prediction = 9544.657796 tokens/s/node
calibration anchor = 22282 tokens/s/node
signed difference = -12737.342204 tokens/s/node
signed residual movement = 0.000000 percentage points
final signed residual = -57.164268 percent
```

The result remains an **UNDERCORRECTION**. No constant was fitted and no
decode overlap term was introduced.

## Component locality and classification

MLA, compressed-KV read, dense early MLP, router, shared expert and LM head
all retain their exact rank-local standard-decode shape. The committed JSON
companion carries all 14 logical family rows and the complete inherited
physical classification ledger. Physical names are not guessed into semantic
families: `fused_moe_kernel` remains routed and every nonmatching row remains
retained.

## Access and preservation

All 3 field-addressed accesses have contemporaneous
BEGIN and END events, and every completed byte count is below its source size.
The standard-case selector returned before the forbidden MTP case. Whole-file
streams: **0**. The forbidden-access ledger is exactly `[]`.

All 134 prior artifacts are byte-identical:
93 inherited SHA-256 locks plus
41 merged CORE-63 Git blob locks.
Hash verification decoded no artifact values.

## Registry disposition

CORE-64's literal gap-resolution clause is not satisfied by a zero movement,
so CORE-64 remains open. CORE-65 receives the exact remaining physical
attribution gap: **-12737.342204
tokens/s/node**, or **-57.164268 percent**,
after the complete rank-local shape match. CORE-65 was free on base main.
