# Surrogate conformance result

**NOT CERTIFIED: F1, F2, F3, F5, F7 pass their frozen rows; F4, F6, W prevent certification**

## What ran

The frozen F1 through F7 exact families and the W wall-time band ran against the in-process vLLM 0.27.1 scheduler and the framework-free surrogate from identical causal tuples and workloads.

## What came out

- F1: 4 passed, 0 failed, 4 total.
- F2: 8 passed, 0 failed, 8 total.
- F3: 4 passed, 0 failed, 4 total.
- F4: 0 passed, 3 failed, 3 total.
- F5: 4 passed, 0 failed, 4 total.
- F6: 0 passed, 3 failed, 3 total.
- F7: 5 passed, 0 failed, 5 total.
- W: 0 passed, 1 failed, 1 total.

W measured a live median of 139552358 ns and a surrogate median of 58711515 ns over 7 runs. The surrogate-to-live ratio was 0.420713171, or a 2.377 times speedup, against the frozen maximum ratio of 0.01.

F7 retained 15 surrogate RESERVE rows outside the amended scored alphabet: `f3-blocks3-seqs2` 4, `f3-blocks5-seqs2` 4, `f4-zero-full-prefix-blocks` 2, `f4-one-full-prefix-block` 2, `f4-several-full-prefix-blocks` 3.

The W baseline remains the pre-run choice: compare the complete framework-free Python surrogate loop with the in-process live vLLM scheduler loop on the same workload, excluding construction and capture from both timed regions. It is a deployment-value class rather than an equal-implementation-cost microbenchmark, and its frozen one-hundred-times bar remains missed.

The corrected qualified scope is limited to the complete frozen families F1, F2, F3, F5, F7. The families F4, F6, W still prevent certification of the loop as a whole.

## Post-specified scoring corrections

Adversarial review found one vacuous negative control, one oracle capture-order artifact, one non-authoritative F7 projection, and one omitted frozen F6 comparison. This corrected record supersedes `attempt-003` for scoring only. The original portable record remains preserved at SHA-256 `bfd9c185a9d4d87b1daa6244933a9aeaf57b298547a0a5c80c694418b6a9556c`. The frozen expectations, amendment, configuration, study cells, and attempt evidence are unchanged.

| Family | Before | Corrected | Change |
|---|---:|---:|---:|
| F1 | 4/4 | 4/4 | +0 passes |
| F2 | 8/8 | 8/8 | +0 passes |
| F3 | 0/4 | 4/4 | +4 passes |
| F4 | 0/3 | 0/3 | +0 passes |
| F5 | 4/4 | 4/4 | +0 passes |
| F6 | 2/3 | 0/3 | -2 passes |
| F7 | 0/5 | 5/5 | +5 passes |
| W | 0/1 | 0/1 | +0 passes |

| Review finding | Class | Disclosed correction |
|---|---|---|
| `kv-mutation-control` | BLOCKER | move the KV mutation to a corrected passing F3 row and require an observed PASS-to-FAIL transition |
| `F3-free-order` | MAJOR | capture each pinned manager group's actual reversed free order instead of its pre-free allocation order |
| `F7-free-authority` | MAJOR | exclude FREE only from cache-enabled F7 scoring because VLLM-47 records that the bridge cannot identify discarded content; retain every FREE divergence as an unscored observation |
| `F6-kv-accounting` | MAJOR | compare live and surrogate KV accounting exactly as the frozen clause requires |
| `W-baseline` | UNCHANGED | retain the frozen complete-loop baseline choice, band, and miss without rescoring |
| `registration-renumbering` | DISCLOSED | renumber the registered residuals VLLM-42 to VLLM-46 and VLLM-43 to VLLM-47 at integration, because the queue-onset publication's frozen bytes on main already reserve VLLM-42 and VLLM-43; the tasks themselves, the frozen expectations, the amendment, and every scored verdict are unchanged |

### Evaluated mutation control

The KV control is re-pointed to the corrected passing row `f3-blocks3-seqs2`. Its baseline has 0 mismatches and is PASS; after one KV action mutation it has 1 mismatch and is FAIL. The asserted PASS-to-FAIL transition is True. The independent record and pricing mutations were also detected.

### Frozen F6 accounting result

F6 is scored without relaxation and fails all three rows. The two prefix-free rows have identical priced StepResult values but the surrogate carries RESERVE and WRITE accounting absent from the live sidecar. The cache-enabled row has those differences, the existing step-index mismatch, and FREE-accounting divergence. VLLM-44, VLLM-46 and VLLM-47 own the missing native service, content state and pre-decision observations; none is dropped from F6.

### Cache-enabled FREE observations

These rows remain recorded but do not decide F7. VLLM-47 documents that the bridge projects release as FREE without observing whether cached content remains reclaimable.

| Cell | Live FREE rows | Surrogate FREE rows | Divergences |
|---|---:|---:|---:|
| `f4-zero-full-prefix-blocks` | 2 | 2 | 4 |
| `f4-one-full-prefix-block` | 2 | 2 | 4 |
| `f4-several-full-prefix-blocks` | 3 | 3 | 6 |

## Row-level findings

| Family | Cell | Status | Misses | Clause |
|---|---|---:|---:|---|
| F1 | `f1-budget16-seqs1` | PASS | 0 | F1 budget by sequence cap |
| F1 | `f1-budget16-seqs2` | PASS | 0 | F1 budget by sequence cap |
| F1 | `f1-budget24-seqs1` | PASS | 0 | F1 budget by sequence cap |
| F1 | `f1-budget24-seqs2` | PASS | 0 | F1 budget by sequence cap |
| F2 | `f2-budget-minus-one` | PASS | 0 | F2 prompt at budget minus one with chunking enabled |
| F2 | `f2-budget` | PASS | 0 | F2 prompt at budget with chunking enabled |
| F2 | `f2-budget-plus-one` | PASS | 0 | F2 prompt at budget plus one with chunking enabled |
| F2 | `f2-threshold-one` | PASS | 0 | F2 one long-prefill-threshold extent with chunking enabled |
| F2 | `f2-threshold-two` | PASS | 0 | F2 two long-prefill-threshold extents with chunking enabled |
| F2 | `f2-threshold-three` | PASS | 0 | F2 three long-prefill-threshold extents with chunking enabled |
| F2 | `f2-chunking-off-stop-at-head` | PASS | 0 | F2 disabled chunking stops at an over-budget waiting head |
| F2 | `f2-chunking-off-threshold` | PASS | 0 | F2 long-prefill threshold is applied before disabled chunking |
| F3 | `f3-blocks3-seqs1` | PASS | 0 | F3 three blocks with one concurrent sequence |
| F3 | `f3-blocks3-seqs2` | PASS | 0 | F3 three blocks with two concurrent sequences |
| F3 | `f3-blocks5-seqs1` | PASS | 0 | F3 five blocks with one concurrent sequence |
| F3 | `f3-blocks5-seqs2` | PASS | 0 | F3 five blocks with two concurrent sequences |
| F4 | `f4-zero-full-prefix-blocks` | FAIL | 6 | F4 repeated prefixes share zero full hash blocks |
| F4 finding | `$.decision_records[1].finished_request_ids` | FAIL | 1 | expected `1`, observed `0` |
| F4 finding | `$.native[1].finished_request_ids` | FAIL | 1 | expected `1`, observed `0` |
| F4 finding | `$.kv_operations[2].token_end` | FAIL | 1 | expected `32`, observed `16` |
| F4 finding | `$.kv_operations[2].block_ids` | FAIL | 1 | expected `2`, observed `1` |
| F4 finding | `$.kv_operations[5].token_end` | FAIL | 1 | expected `32`, observed `16` |
| F4 finding | `$.kv_operations[5].block_ids` | FAIL | 1 | expected `2`, observed `1` |
| F4 | `f4-one-full-prefix-block` | FAIL | 6 | F4 repeated prefixes share one full hash block |
| F4 finding | `$.decision_records[1].finished_request_ids` | FAIL | 1 | expected `1`, observed `0` |
| F4 finding | `$.native[1].finished_request_ids` | FAIL | 1 | expected `1`, observed `0` |
| F4 finding | `$.kv_operations[2].token_end` | FAIL | 1 | expected `32`, observed `16` |
| F4 finding | `$.kv_operations[2].block_ids` | FAIL | 1 | expected `2`, observed `1` |
| F4 finding | `$.kv_operations[7].token_end` | FAIL | 1 | expected `32`, observed `16` |
| F4 finding | `$.kv_operations[7].block_ids` | FAIL | 1 | expected `2`, observed `1` |
| F4 | `f4-several-full-prefix-blocks` | FAIL | 10 | F4 several full hash blocks, full-hit last-token recompute, and stable eviction order |
| F4 finding | `$.decision_records[1].finished_request_ids` | FAIL | 1 | expected `1`, observed `0` |
| F4 finding | `$.decision_records[2].finished_request_ids` | FAIL | 1 | expected `1`, observed `0` |
| F4 finding | `$.native[1].finished_request_ids` | FAIL | 1 | expected `1`, observed `0` |
| F4 finding | `$.native[2].finished_request_ids` | FAIL | 1 | expected `1`, observed `0` |
| F4 finding | `$.kv_operations[2].token_end` | FAIL | 1 | expected `64`, observed `16` |
| F4 finding | `$.kv_operations[2].block_ids` | FAIL | 1 | expected `4`, observed `1` |
| F4 finding | `$.kv_operations[7].token_end` | FAIL | 1 | expected `64`, observed `16` |
| F4 finding | `$.kv_operations[7].block_ids` | FAIL | 1 | expected `4`, observed `1` |
| F4 finding | `$.kv_operations[13].token_end` | FAIL | 1 | expected `64`, observed `16` |
| F4 finding | `$.kv_operations[13].block_ids` | FAIL | 1 | expected `4`, observed `1` |
| F5 | `f5-one-offset750000` | PASS | 0 | F5 one follower at 750000 ps |
| F5 | `f5-one-offset1250000` | PASS | 0 | F5 one follower at 1250000 ps |
| F5 | `f5-three-offset750000` | PASS | 0 | F5 three followers at 750000 ps |
| F5 | `f5-three-offset1250000` | PASS | 0 | F5 three followers at 1250000 ps |
| F6 | `f1-budget16-seqs2` | FAIL | 6 | F6 identical pricing chain and metric reachability |
| F6 finding | `$.kv-accounting.action_counts.reserve` | FAIL | 1 | expected `null`, observed `3` |
| F6 finding | `$.kv-accounting.action_counts.write` | FAIL | 1 | expected `null`, observed `7` |
| F6 finding | `$.kv-accounting.block_visits.reserve` | FAIL | 1 | expected `null`, observed `0` |
| F6 finding | `$.kv-accounting.block_visits.write` | FAIL | 1 | expected `null`, observed `7` |
| F6 finding | `$.kv-accounting.token_spans.reserve` | FAIL | 1 | expected `null`, observed `48` |
| F6 finding | `$.kv-accounting.token_spans.write` | FAIL | 1 | expected `null`, observed `30` |
| F6 | `f3-blocks3-seqs2` | FAIL | 6 | F6 identical pricing chain and metric reachability |
| F6 finding | `$.kv-accounting.action_counts.reserve` | FAIL | 1 | expected `null`, observed `4` |
| F6 finding | `$.kv-accounting.action_counts.write` | FAIL | 1 | expected `null`, observed `6` |
| F6 finding | `$.kv-accounting.block_visits.reserve` | FAIL | 1 | expected `null`, observed `0` |
| F6 finding | `$.kv-accounting.block_visits.write` | FAIL | 1 | expected `null`, observed `10` |
| F6 finding | `$.kv-accounting.token_spans.reserve` | FAIL | 1 | expected `null`, observed `80` |
| F6 finding | `$.kv-accounting.token_spans.write` | FAIL | 1 | expected `null`, observed `52` |
| F6 | `f4-one-full-prefix-block` | FAIL | 9 | F6 identical pricing chain and metric reachability |
| F6 finding | `$.surrogate-pricing[1].step_index` | FAIL | 1 | expected `1`, observed `2` |
| F6 finding | `$.kv-accounting.action_counts.reserve` | FAIL | 1 | expected `null`, observed `2` |
| F6 finding | `$.kv-accounting.action_counts.write` | FAIL | 1 | expected `null`, observed `2` |
| F6 finding | `$.kv-accounting.block_visits.free` | FAIL | 1 | expected `4`, observed `2` |
| F6 finding | `$.kv-accounting.block_visits.reserve` | FAIL | 1 | expected `null`, observed `0` |
| F6 finding | `$.kv-accounting.block_visits.write` | FAIL | 1 | expected `null`, observed `4` |
| F6 finding | `$.kv-accounting.token_spans.free` | FAIL | 1 | expected `64`, observed `32` |
| F6 finding | `$.kv-accounting.token_spans.reserve` | FAIL | 1 | expected `null`, observed `48` |
| F6 finding | `$.kv-accounting.token_spans.write` | FAIL | 1 | expected `null`, observed `18` |
| F7 | `f3-blocks3-seqs2` | PASS | 0 | F7 witnessed KV alphabet under one stable block bijection |
| F7 | `f3-blocks5-seqs2` | PASS | 0 | F7 witnessed KV alphabet under one stable block bijection |
| F7 | `f4-zero-full-prefix-blocks` | PASS | 0 | F7 witnessed KV alphabet under one stable block bijection |
| F7 | `f4-one-full-prefix-block` | PASS | 0 | F7 witnessed KV alphabet under one stable block bijection |
| F7 | `f4-several-full-prefix-blocks` | PASS | 0 | F7 witnessed KV alphabet under one stable block bijection |
| W | `w-largest-frozen-workload` | FAIL | 1 | steady-loop-one-hundred-times |
| W finding | `$.surrogate_to_live_ratio` | FAIL | 1 | expected `"<= 0.01"`, observed `0.4207131706079807` |

## Fatal guards

The run is nonvoid. All 78 fatal guards were evaluated: 78 fatal guards passed and 0 failed. Fatal guards are not part of any behavioral denominator.

| Guard | Status | Misses |
|---|---:|---:|
| `chronology` | PASS | 0 |
| `source-hash` | PASS | 0 |
| `identifier-uniqueness` | PASS | 0 |
| `F5-frozen-fixture` | PASS | 0 |
| `native-output:f1-budget16-seqs1` | PASS | 0 |
| `configuration:f1-budget16-seqs1` | PASS | 0 |
| `token-conservation:f1-budget16-seqs1` | PASS | 0 |
| `F1-allocation:f1-budget16-seqs1` | PASS | 0 |
| `native-output:f1-budget16-seqs2` | PASS | 0 |
| `configuration:f1-budget16-seqs2` | PASS | 0 |
| `token-conservation:f1-budget16-seqs2` | PASS | 0 |
| `F1-allocation:f1-budget16-seqs2` | PASS | 0 |
| `native-output:f1-budget24-seqs1` | PASS | 0 |
| `configuration:f1-budget24-seqs1` | PASS | 0 |
| `token-conservation:f1-budget24-seqs1` | PASS | 0 |
| `F1-allocation:f1-budget24-seqs1` | PASS | 0 |
| `native-output:f1-budget24-seqs2` | PASS | 0 |
| `configuration:f1-budget24-seqs2` | PASS | 0 |
| `token-conservation:f1-budget24-seqs2` | PASS | 0 |
| `F1-allocation:f1-budget24-seqs2` | PASS | 0 |
| `native-output:f2-budget-minus-one` | PASS | 0 |
| `configuration:f2-budget-minus-one` | PASS | 0 |
| `token-conservation:f2-budget-minus-one` | PASS | 0 |
| `native-output:f2-budget` | PASS | 0 |
| `configuration:f2-budget` | PASS | 0 |
| `token-conservation:f2-budget` | PASS | 0 |
| `native-output:f2-budget-plus-one` | PASS | 0 |
| `configuration:f2-budget-plus-one` | PASS | 0 |
| `token-conservation:f2-budget-plus-one` | PASS | 0 |
| `native-output:f2-threshold-one` | PASS | 0 |
| `configuration:f2-threshold-one` | PASS | 0 |
| `token-conservation:f2-threshold-one` | PASS | 0 |
| `native-output:f2-threshold-two` | PASS | 0 |
| `configuration:f2-threshold-two` | PASS | 0 |
| `token-conservation:f2-threshold-two` | PASS | 0 |
| `native-output:f2-threshold-three` | PASS | 0 |
| `configuration:f2-threshold-three` | PASS | 0 |
| `token-conservation:f2-threshold-three` | PASS | 0 |
| `native-output:f2-chunking-off-stop-at-head` | PASS | 0 |
| `configuration:f2-chunking-off-stop-at-head` | PASS | 0 |
| `token-conservation:f2-chunking-off-stop-at-head` | PASS | 0 |
| `native-output:f2-chunking-off-threshold` | PASS | 0 |
| `configuration:f2-chunking-off-threshold` | PASS | 0 |
| `token-conservation:f2-chunking-off-threshold` | PASS | 0 |
| `native-output:f3-blocks3-seqs1` | PASS | 0 |
| `configuration:f3-blocks3-seqs1` | PASS | 0 |
| `token-conservation:f3-blocks3-seqs1` | PASS | 0 |
| `native-output:f3-blocks3-seqs2` | PASS | 0 |
| `configuration:f3-blocks3-seqs2` | PASS | 0 |
| `token-conservation:f3-blocks3-seqs2` | PASS | 0 |
| `native-output:f3-blocks5-seqs1` | PASS | 0 |
| `configuration:f3-blocks5-seqs1` | PASS | 0 |
| `token-conservation:f3-blocks5-seqs1` | PASS | 0 |
| `native-output:f3-blocks5-seqs2` | PASS | 0 |
| `configuration:f3-blocks5-seqs2` | PASS | 0 |
| `token-conservation:f3-blocks5-seqs2` | PASS | 0 |
| `native-output:f4-zero-full-prefix-blocks` | PASS | 0 |
| `configuration:f4-zero-full-prefix-blocks` | PASS | 0 |
| `token-conservation:f4-zero-full-prefix-blocks` | PASS | 0 |
| `native-output:f4-one-full-prefix-block` | PASS | 0 |
| `configuration:f4-one-full-prefix-block` | PASS | 0 |
| `token-conservation:f4-one-full-prefix-block` | PASS | 0 |
| `native-output:f4-several-full-prefix-blocks` | PASS | 0 |
| `configuration:f4-several-full-prefix-blocks` | PASS | 0 |
| `token-conservation:f4-several-full-prefix-blocks` | PASS | 0 |
| `native-output:f5-one-offset750000` | PASS | 0 |
| `configuration:f5-one-offset750000` | PASS | 0 |
| `token-conservation:f5-one-offset750000` | PASS | 0 |
| `native-output:f5-one-offset1250000` | PASS | 0 |
| `configuration:f5-one-offset1250000` | PASS | 0 |
| `token-conservation:f5-one-offset1250000` | PASS | 0 |
| `native-output:f5-three-offset750000` | PASS | 0 |
| `configuration:f5-three-offset750000` | PASS | 0 |
| `token-conservation:f5-three-offset750000` | PASS | 0 |
| `native-output:f5-three-offset1250000` | PASS | 0 |
| `configuration:f5-three-offset1250000` | PASS | 0 |
| `token-conservation:f5-three-offset1250000` | PASS | 0 |
| `end-to-end-mutation-controls` | PASS | 0 |

## What it changes for the project

DEPLOY-18 closes because the review-traced oracle capture now records the pinned engine free order and all four F3 rows plus both F3-derived F7 rows pass exactly, withdrawing the phantom surrogate allocator defect. DEPLOY-19 stays open, narrowed to the genuine one-step-late finished-identity finding in F4. DEPLOY-20 stays open on the prefix decision-step mismatch and frozen KV-accounting equality. DEPLOY-21 stays open on W. VLLM-47 owns the recorded cache-enabled FREE projection limit, renumbered from VLLM-43 at integration because main's queue-onset publication holds frozen claims to VLLM-42 and VLLM-43; the per-layer KV byte residual is VLLM-46 for the same reason. No milestone advances, and the faithful-stand-in claim remains nonliteral. This publication supersedes the attempt-004 record with a verdict-equivalent independent integrator rerun carrying the renumbered registrations; every family tally, every per-cell status and every guard status is unchanged, and only wall-clock measurements and timestamps differ.

## What it does not change

The result does not change the accepted F1, F2 or F5 surfaces, and it newly qualifies only the frozen F3 and F7 surfaces. F4, F6 and W remain failed. Cache-enabled FREE divergences are retained as unscored F7 observations rather than attributed to surrogate behavior. The result does not claim silicon timing, asynchronous scheduling, speculative decoding, LoRA, multimodal input, pipeline parallelism, multi-pool serving or framework pins other than 0.27.1. The original attempt-003 record and all frozen inputs remain preserved.

## Scope and chronology

Certification, when earned, applies only to the frozen cells, the declared witnessed KV alphabet, the deterministic synthetic pricing chain, and vLLM 0.27.1 at scheduler source SHA-256 `c67bda2886b52865ddafabaae7d797c359e930752f374421a33e537d94a5f45a`. It is re-earned at every framework pin bump.

The final pre-run configuration commit is `d947cea75ace88f220d2c06f58c6939d5929c932`.

The native SchedulerOutput captures, their paired projections, KV sidecars, per-cell summaries, and every timing repetition remain in the append-only bulk attempt named in the tracked record.
