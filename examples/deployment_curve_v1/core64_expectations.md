# CORE-64 decode-family shape expectations

Status: **EXPECTATIONS ONLY**. These shape mismatches, component classes, and
directions are frozen before protected record access and before movement is
computed.

## Enumerated shape mismatches

1. **MLA and attention:** expected no mismatch. The disclosed standard decode
   has `9 * 8 = 72` ranks and attention DP72. Its `9 * 256 = 2,304` requests
   therefore divide to `2,304 / 72 = 32` requests per rank. At KV 2,000 this
   is 64,000 aggregate KV-token references per rank. The capture is TP1 at
   batch 32 and KV 2,000, so every MLA and attention scale stays exactly one.
2. **Shared expert and dense feed-forward:** expected no mismatch. Each rank
   runs these rank-local paths for the same 32 requests that enter its
   attention path. Decode dense DP1 denotes a singleton structural dense group,
   not the whole node's 256 requests on each rank. Both scales stay one.
3. **Router and output path:** expected no mismatch. The router sees the same
   32 local tokens before expert dispatch, and the LM head samples the same 32
   local tokens. Both scales stay one.
4. **Routed experts:** no new CORE-64 scale is allowed. The inherited CORE-63
   `fused_moe_kernel` classification and exact `1/9` residency scale remain
   byte-identical and are not amended.
5. **Fixed and unbound physical rows:** the 489 ps fixed term stays once. Every
   nonmatching physical noncollective row stays at scale one. The repository
   has no total physical-kernel-to-logical-family binding for DeepSeek EP72;
   SGL-34 and VLLM-38 own that launch-identity gap. Because every non-routed
   logical family has the same frozen scale, the missing binding cannot alter
   this study's arithmetic, but it remains the exact attribution limitation.
6. **MTP:** absent from standard decode and forbidden. It is not a component of
   this calibration-only movement.

## Frozen family classification

The standard decode logical families are MLA query compression, query
decompression, KV compression, KV decompression, rotary split, attention,
compressed-KV read and output projection; dense early MLP; MoE router; MoE
shared expert; routed experts; and LM head. All except routed experts have
the capture-matching per-rank shape and scale one. The MTP head is absent.

Physical kernel names are not guessed into semantic families. A name
containing `fused_moe_kernel`, case-insensitively, retains CORE-63's routed
classification. Every other retained row stays non-routed at scale one.

## Frozen signed direction

Expected CORE-64 movement is null:

- corrected step: no change;
- standard-decode prediction: no change;
- signed residual: no change;
- prediction movement from 9,544.657796 tokens/s/node: exactly zero;
- remaining signed difference from 22,282: remain negative.

No parameter may be fitted. A null result must publish honestly. CORE-64 may
close only if its literal gap-resolution entry is satisfied. Otherwise CORE-65
must receive the exact remaining standard-decode attribution gap; CORE-65 was
verified absent on base main `b8864f9`.
