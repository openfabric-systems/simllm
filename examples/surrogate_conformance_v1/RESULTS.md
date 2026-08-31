# Surrogate conformance result

**NOT CERTIFIED: 14 frozen family rows missed and bound the surrogate envelope**

## What ran

The frozen F1 through F7 exact families and the W wall-time band ran against the in-process vLLM 0.27.1 scheduler and the framework-free surrogate from identical causal tuples and workloads.

## What came out

- F1: 4 passed, 0 failed, 4 total.
- F2: 8 passed, 0 failed, 8 total.
- F3: 0 passed, 4 failed, 4 total.
- F4: 0 passed, 3 failed, 3 total.
- F5: 4 passed, 0 failed, 4 total.
- F6: 2 passed, 1 failed, 3 total.
- F7: 0 passed, 5 failed, 5 total.
- W: 0 passed, 1 failed, 1 total.

W measured a live median of 178323150 ns and a surrogate median of 74259205 ns over 7 runs. The surrogate-to-live ratio was 0.416430536, or a 2.401 times speedup, against the frozen maximum ratio of 0.01.

F7 retained 15 surrogate RESERVE rows outside the amended scored alphabet: `f3-blocks3-seqs2` 4, `f3-blocks5-seqs2` 4, `f4-zero-full-prefix-blocks` 2, `f4-one-full-prefix-block` 2, `f4-several-full-prefix-blocks` 3.

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
| F3 | `f3-blocks3-seqs1` | FAIL | 8 | F3 three blocks with one concurrent sequence |
| F3 finding | `$.kv_operations[2].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F3 finding | `$.kv_operations[2].block_bijection` | FAIL | 1 | expected `"2"`, observed `"1"` |
| F3 finding | `$.kv_operations[3].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F3 finding | `$.kv_operations[3].block_bijection` | FAIL | 1 | expected `"2"`, observed `"1"` |
| F3 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"2"`, observed `"1"` |
| F3 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F3 finding | `$.kv_operations[7].block_bijection` | FAIL | 1 | expected `"2"`, observed `"1"` |
| F3 finding | `$.kv_operations[7].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F3 | `f3-blocks3-seqs2` | FAIL | 8 | F3 three blocks with two concurrent sequences |
| F3 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F3 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"2"`, observed `"1"` |
| F3 finding | `$.kv_operations[7].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F3 finding | `$.kv_operations[7].block_bijection` | FAIL | 1 | expected `"2"`, observed `"1"` |
| F3 finding | `$.kv_operations[9].block_bijection` | FAIL | 1 | expected `"2"`, observed `"1"` |
| F3 finding | `$.kv_operations[9].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F3 finding | `$.kv_operations[10].block_bijection` | FAIL | 1 | expected `"2"`, observed `"1"` |
| F3 finding | `$.kv_operations[10].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F3 | `f3-blocks5-seqs1` | FAIL | 8 | F3 five blocks with one concurrent sequence |
| F3 finding | `$.kv_operations[2].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F3 finding | `$.kv_operations[2].block_bijection` | FAIL | 1 | expected `"2"`, observed `"1"` |
| F3 finding | `$.kv_operations[3].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F3 finding | `$.kv_operations[3].block_bijection` | FAIL | 1 | expected `"2"`, observed `"1"` |
| F3 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"3"`, observed `"4"` |
| F3 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"4"`, observed `"3"` |
| F3 finding | `$.kv_operations[7].block_bijection` | FAIL | 1 | expected `"3"`, observed `"4"` |
| F3 finding | `$.kv_operations[7].block_bijection` | FAIL | 1 | expected `"4"`, observed `"3"` |
| F3 | `f3-blocks5-seqs2` | FAIL | 8 | F3 five blocks with two concurrent sequences |
| F3 finding | `$.kv_operations[4].block_bijection` | FAIL | 1 | expected `"1"`, observed `"3"` |
| F3 finding | `$.kv_operations[4].block_bijection` | FAIL | 1 | expected `"3"`, observed `"1"` |
| F3 finding | `$.kv_operations[5].block_bijection` | FAIL | 1 | expected `"1"`, observed `"3"` |
| F3 finding | `$.kv_operations[5].block_bijection` | FAIL | 1 | expected `"3"`, observed `"1"` |
| F3 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"2"`, observed `"4"` |
| F3 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"4"`, observed `"2"` |
| F3 finding | `$.kv_operations[7].block_bijection` | FAIL | 1 | expected `"2"`, observed `"4"` |
| F3 finding | `$.kv_operations[7].block_bijection` | FAIL | 1 | expected `"4"`, observed `"2"` |
| F4 | `f4-zero-full-prefix-blocks` | FAIL | 12 | F4 repeated prefixes share zero full hash blocks |
| F4 finding | `$.decision_records[1].finished_request_ids` | FAIL | 1 | expected `1`, observed `0` |
| F4 finding | `$.native[1].finished_request_ids` | FAIL | 1 | expected `1`, observed `0` |
| F4 finding | `$.kv_operations[1].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F4 finding | `$.kv_operations[1].block_bijection` | FAIL | 1 | expected `"2"`, observed `"1"` |
| F4 finding | `$.kv_operations[2].token_end` | FAIL | 1 | expected `32`, observed `16` |
| F4 finding | `$.kv_operations[2].block_ids` | FAIL | 1 | expected `2`, observed `1` |
| F4 finding | `$.kv_operations[2].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F4 finding | `$.kv_operations[4].block_bijection` | FAIL | 1 | expected `"2"`, observed `"3"` |
| F4 finding | `$.kv_operations[4].block_bijection` | FAIL | 1 | expected `"3"`, observed `"2"` |
| F4 finding | `$.kv_operations[5].token_end` | FAIL | 1 | expected `32`, observed `16` |
| F4 finding | `$.kv_operations[5].block_ids` | FAIL | 1 | expected `2`, observed `1` |
| F4 finding | `$.kv_operations[5].block_bijection` | FAIL | 1 | expected `"2"`, observed `"3"` |
| F4 | `f4-one-full-prefix-block` | FAIL | 12 | F4 repeated prefixes share one full hash block |
| F4 finding | `$.decision_records[1].finished_request_ids` | FAIL | 1 | expected `1`, observed `0` |
| F4 finding | `$.native[1].finished_request_ids` | FAIL | 1 | expected `1`, observed `0` |
| F4 finding | `$.kv_operations[1].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F4 finding | `$.kv_operations[1].block_bijection` | FAIL | 1 | expected `"2"`, observed `"1"` |
| F4 finding | `$.kv_operations[2].token_end` | FAIL | 1 | expected `32`, observed `16` |
| F4 finding | `$.kv_operations[2].block_ids` | FAIL | 1 | expected `2`, observed `1` |
| F4 finding | `$.kv_operations[2].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F4 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F4 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"2"`, observed `"1"` |
| F4 finding | `$.kv_operations[7].token_end` | FAIL | 1 | expected `32`, observed `16` |
| F4 finding | `$.kv_operations[7].block_ids` | FAIL | 1 | expected `2`, observed `1` |
| F4 finding | `$.kv_operations[7].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F4 | `f4-several-full-prefix-blocks` | FAIL | 25 | F4 several full hash blocks, full-hit last-token recompute, and stable eviction order |
| F4 finding | `$.decision_records[1].finished_request_ids` | FAIL | 1 | expected `1`, observed `0` |
| F4 finding | `$.decision_records[2].finished_request_ids` | FAIL | 1 | expected `1`, observed `0` |
| F4 finding | `$.native[1].finished_request_ids` | FAIL | 1 | expected `1`, observed `0` |
| F4 finding | `$.native[2].finished_request_ids` | FAIL | 1 | expected `1`, observed `0` |
| F4 finding | `$.kv_operations[1].block_bijection` | FAIL | 1 | expected `"1"`, observed `"4"` |
| F4 finding | `$.kv_operations[1].block_bijection` | FAIL | 1 | expected `"2"`, observed `"3"` |
| F4 finding | `$.kv_operations[1].block_bijection` | FAIL | 1 | expected `"3"`, observed `"2"` |
| F4 finding | `$.kv_operations[1].block_bijection` | FAIL | 1 | expected `"4"`, observed `"1"` |
| F4 finding | `$.kv_operations[2].token_end` | FAIL | 1 | expected `64`, observed `16` |
| F4 finding | `$.kv_operations[2].block_ids` | FAIL | 1 | expected `4`, observed `1` |
| F4 finding | `$.kv_operations[2].block_bijection` | FAIL | 1 | expected `"1"`, observed `"4"` |
| F4 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"1"`, observed `"4"` |
| F4 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"2"`, observed `"3"` |
| F4 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"3"`, observed `"2"` |
| F4 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"4"`, observed `"1"` |
| F4 finding | `$.kv_operations[7].token_end` | FAIL | 1 | expected `64`, observed `16` |
| F4 finding | `$.kv_operations[7].block_ids` | FAIL | 1 | expected `4`, observed `1` |
| F4 finding | `$.kv_operations[7].block_bijection` | FAIL | 1 | expected `"1"`, observed `"4"` |
| F4 finding | `$.kv_operations[12].block_bijection` | FAIL | 1 | expected `"4"`, observed `"1"` |
| F4 finding | `$.kv_operations[12].block_bijection` | FAIL | 1 | expected `"3"`, observed `"2"` |
| F4 finding | `$.kv_operations[12].block_bijection` | FAIL | 1 | expected `"2"`, observed `"3"` |
| F4 finding | `$.kv_operations[12].block_bijection` | FAIL | 1 | expected `"1"`, observed `"4"` |
| F4 finding | `$.kv_operations[13].token_end` | FAIL | 1 | expected `64`, observed `16` |
| F4 finding | `$.kv_operations[13].block_ids` | FAIL | 1 | expected `4`, observed `1` |
| F4 finding | `$.kv_operations[13].block_bijection` | FAIL | 1 | expected `"4"`, observed `"1"` |
| F5 | `f5-one-offset750000` | PASS | 0 | F5 one follower at 750000 ps |
| F5 | `f5-one-offset1250000` | PASS | 0 | F5 one follower at 1250000 ps |
| F5 | `f5-three-offset750000` | PASS | 0 | F5 three followers at 750000 ps |
| F5 | `f5-three-offset1250000` | PASS | 0 | F5 three followers at 1250000 ps |
| F6 | `f1-budget16-seqs2` | PASS | 0 | F6 identical pricing chain and metric reachability |
| F6 | `f3-blocks3-seqs2` | PASS | 0 | F6 identical pricing chain and metric reachability |
| F6 | `f4-one-full-prefix-block` | FAIL | 1 | F6 identical pricing chain and metric reachability |
| F6 finding | `$.surrogate-pricing[1].step_index` | FAIL | 1 | expected `1`, observed `2` |
| F7 | `f3-blocks3-seqs2` | FAIL | 8 | F7 witnessed KV alphabet under one stable block bijection |
| F7 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F7 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"2"`, observed `"1"` |
| F7 finding | `$.kv_operations[7].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F7 finding | `$.kv_operations[7].block_bijection` | FAIL | 1 | expected `"2"`, observed `"1"` |
| F7 finding | `$.kv_operations[9].block_bijection` | FAIL | 1 | expected `"2"`, observed `"1"` |
| F7 finding | `$.kv_operations[9].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F7 finding | `$.kv_operations[10].block_bijection` | FAIL | 1 | expected `"2"`, observed `"1"` |
| F7 finding | `$.kv_operations[10].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F7 | `f3-blocks5-seqs2` | FAIL | 8 | F7 witnessed KV alphabet under one stable block bijection |
| F7 finding | `$.kv_operations[4].block_bijection` | FAIL | 1 | expected `"1"`, observed `"3"` |
| F7 finding | `$.kv_operations[4].block_bijection` | FAIL | 1 | expected `"3"`, observed `"1"` |
| F7 finding | `$.kv_operations[5].block_bijection` | FAIL | 1 | expected `"1"`, observed `"3"` |
| F7 finding | `$.kv_operations[5].block_bijection` | FAIL | 1 | expected `"3"`, observed `"1"` |
| F7 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"2"`, observed `"4"` |
| F7 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"4"`, observed `"2"` |
| F7 finding | `$.kv_operations[7].block_bijection` | FAIL | 1 | expected `"2"`, observed `"4"` |
| F7 finding | `$.kv_operations[7].block_bijection` | FAIL | 1 | expected `"4"`, observed `"2"` |
| F7 | `f4-zero-full-prefix-blocks` | FAIL | 10 | F7 witnessed KV alphabet under one stable block bijection |
| F7 finding | `$.kv_operations[1].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F7 finding | `$.kv_operations[1].block_bijection` | FAIL | 1 | expected `"2"`, observed `"1"` |
| F7 finding | `$.kv_operations[2].token_end` | FAIL | 1 | expected `32`, observed `16` |
| F7 finding | `$.kv_operations[2].block_ids` | FAIL | 1 | expected `2`, observed `1` |
| F7 finding | `$.kv_operations[2].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F7 finding | `$.kv_operations[4].block_bijection` | FAIL | 1 | expected `"2"`, observed `"3"` |
| F7 finding | `$.kv_operations[4].block_bijection` | FAIL | 1 | expected `"3"`, observed `"2"` |
| F7 finding | `$.kv_operations[5].token_end` | FAIL | 1 | expected `32`, observed `16` |
| F7 finding | `$.kv_operations[5].block_ids` | FAIL | 1 | expected `2`, observed `1` |
| F7 finding | `$.kv_operations[5].block_bijection` | FAIL | 1 | expected `"2"`, observed `"3"` |
| F7 | `f4-one-full-prefix-block` | FAIL | 10 | F7 witnessed KV alphabet under one stable block bijection |
| F7 finding | `$.kv_operations[1].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F7 finding | `$.kv_operations[1].block_bijection` | FAIL | 1 | expected `"2"`, observed `"1"` |
| F7 finding | `$.kv_operations[2].token_end` | FAIL | 1 | expected `32`, observed `16` |
| F7 finding | `$.kv_operations[2].block_ids` | FAIL | 1 | expected `2`, observed `1` |
| F7 finding | `$.kv_operations[2].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F7 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F7 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"2"`, observed `"1"` |
| F7 finding | `$.kv_operations[7].token_end` | FAIL | 1 | expected `32`, observed `16` |
| F7 finding | `$.kv_operations[7].block_ids` | FAIL | 1 | expected `2`, observed `1` |
| F7 finding | `$.kv_operations[7].block_bijection` | FAIL | 1 | expected `"1"`, observed `"2"` |
| F7 | `f4-several-full-prefix-blocks` | FAIL | 21 | F7 witnessed KV alphabet under one stable block bijection |
| F7 finding | `$.kv_operations[1].block_bijection` | FAIL | 1 | expected `"1"`, observed `"4"` |
| F7 finding | `$.kv_operations[1].block_bijection` | FAIL | 1 | expected `"2"`, observed `"3"` |
| F7 finding | `$.kv_operations[1].block_bijection` | FAIL | 1 | expected `"3"`, observed `"2"` |
| F7 finding | `$.kv_operations[1].block_bijection` | FAIL | 1 | expected `"4"`, observed `"1"` |
| F7 finding | `$.kv_operations[2].token_end` | FAIL | 1 | expected `64`, observed `16` |
| F7 finding | `$.kv_operations[2].block_ids` | FAIL | 1 | expected `4`, observed `1` |
| F7 finding | `$.kv_operations[2].block_bijection` | FAIL | 1 | expected `"1"`, observed `"4"` |
| F7 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"1"`, observed `"4"` |
| F7 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"2"`, observed `"3"` |
| F7 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"3"`, observed `"2"` |
| F7 finding | `$.kv_operations[6].block_bijection` | FAIL | 1 | expected `"4"`, observed `"1"` |
| F7 finding | `$.kv_operations[7].token_end` | FAIL | 1 | expected `64`, observed `16` |
| F7 finding | `$.kv_operations[7].block_ids` | FAIL | 1 | expected `4`, observed `1` |
| F7 finding | `$.kv_operations[7].block_bijection` | FAIL | 1 | expected `"1"`, observed `"4"` |
| F7 finding | `$.kv_operations[12].block_bijection` | FAIL | 1 | expected `"4"`, observed `"1"` |
| F7 finding | `$.kv_operations[12].block_bijection` | FAIL | 1 | expected `"3"`, observed `"2"` |
| F7 finding | `$.kv_operations[12].block_bijection` | FAIL | 1 | expected `"2"`, observed `"3"` |
| F7 finding | `$.kv_operations[12].block_bijection` | FAIL | 1 | expected `"1"`, observed `"4"` |
| F7 finding | `$.kv_operations[13].token_end` | FAIL | 1 | expected `64`, observed `16` |
| F7 finding | `$.kv_operations[13].block_ids` | FAIL | 1 | expected `4`, observed `1` |
| F7 finding | `$.kv_operations[13].block_bijection` | FAIL | 1 | expected `"4"`, observed `"1"` |
| W | `w-largest-frozen-workload` | FAIL | 1 | steady-loop-one-hundred-times |
| W finding | `$.surrogate_to_live_ratio` | FAIL | 1 | expected `"<= 0.01"`, observed `0.41643053636053423` |

## Fatal guards

The run is nonvoid. 78 fatal guards passed and 0 failed. Fatal guards are not part of any behavioral denominator.

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

No task closes and no milestone advances. DEPLOY-18 owns recycled block-identity equivalence, DEPLOY-19 owns prefix-cache decision and lifecycle fidelity, DEPLOY-20 owns prefix-path metric-step identity, and DEPLOY-21 owns the frozen steady-loop wall-time relation. The passing F1, F2 and F5 surfaces unblock use only inside their frozen cells; the faithful-stand-in claim remains nonliteral.

## What it does not change

The result does not change the accepted F1, F2 or F5 exact surfaces, does not invalidate the native vLLM capture or KV projection, and does not claim silicon timing, asynchronous scheduling, speculative decoding, LoRA, multimodal input, pipeline parallelism, multi-pool serving or framework pins other than 0.27.1. RESERVE rows remain retained and unscored exactly as amended. VLLM-11 and VLLM-42 through VLLM-45 remain open at their existing scope.

## Scope and chronology

Certification, when earned, applies only to the frozen cells, the declared witnessed KV alphabet, the deterministic synthetic pricing chain, and vLLM 0.27.1 at scheduler source SHA-256 `c67bda2886b52865ddafabaae7d797c359e930752f374421a33e537d94a5f45a`. It is re-earned at every framework pin bump.

The final pre-run configuration commit is `d947cea75ace88f220d2c06f58c6939d5929c932`.

The native SchedulerOutput captures, their paired projections, KV sidecars, per-cell summaries, and every timing repetition remain in the append-only bulk attempt named in the tracked record.
