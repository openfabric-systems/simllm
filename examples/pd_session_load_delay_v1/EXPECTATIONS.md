# Concurrent-session load-delay expectations

Status: expectations only. No surface-priced concurrent curve has run, and no
curve value informed this freeze.

## Exposure ruling

The imported source is the candidate-status record at content address
`ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52`.
The successful reader returned five field selectors, consumed 45,043 of
57,417 bytes and loaded no whole record. It decoded only the two permitted
Granite rows. The source-held-out batch-32 row was raw-skipped without JSON
decoding or capture, as were the leading DeepSeek rows.

This attempt is not clean. Initial reconnaissance scanned the candidate path
family before a reader was committed and surfaced a DeepSeek summary row. Two
later field-reader attempts also rejected, the second after decoding entry 0.
None of that exposed material is usable for this freeze or score. The complete
ledger is published, VLLM-39 cannot close from this attempt, and VLLM-40 is the
frozen clean-repetition residual.

## Imported batch-service surface

The selected keys are exact canonical keys, not partial tuple matches. Both
rows are MEASURED calibration evidence with candidate acceptance status and no
calibration claim.

| Batch | Prior KV per request | Total service (ps) | Service/request (ps) | CV (ppm) | Key SHA-256 |
|---:|---:|---:|---:|---:|---|
| 1 | 16 | 1,110,576,000 | 1,110,576,000 | 4,232 | `d8fdebd7051f530bed3232cfeb2e3d5c87a1ab9a22fe68db9cca51436853a502` |
| 8 | 16 | 1,892,831,500 | 236,603,937.5 | 1,538 | `38978457b27e56dbc0e9ffbe2385b7d53538a2515c826284be4d6d414520c830` |

Interior batches 2 through 7 use a power law, linear in log batch and log total
service, between the two measured endpoints. This is batching gain: total
service rises with batch while service per request falls by 78.7 percent from
batch 1 to batch 8.

## Frozen sweep and held-outs

Every cell contains 64 requests and four output tokens per request. Offered
load is 250, 500, 1,000, 2,000, 4,000 and 8,000 requests/s. The prompt lengths
are 8 and 16 tokens, and pool ratios are 1:1, 1:2 and 2:1.

The calibration configurations are `(1,1,8)` and `(1,2,8)`. All prompt-16
curves are held out, and both prompt curves at the 2:1 pool ratio are held out.
The distinct held-out configurations are therefore `(1,1,16)`, `(1,2,16)`,
`(2,1,8)` and `(2,1,16)`, for 24 held-out curve points.

The surface-derived max-batch decode knees are exactly
`4000000000 / 3785663`, or 1,056.618 requests/s, for one decode engine and
`8000000000 / 3785663`, or 2,113.236 requests/s, for two decode engines.

## Service and wait decomposition

Batching gain is `4 * S(b) / b`: four decode steps times imported total batch
service divided over the requests sharing the batch. Scheduler queue wait is
observed separately as each request's `prefill_queue_ps` plus
`decode_admission_wait_ps`, both arrival-to-admission intervals from the exact
timeline. Provider service never includes those waits.

The pre-run queue prediction is a finite D/D/c overload model over 64 requests,
batch 8 and four decode steps. It adds only positive excess of max-batch service
interval over arrival interval. The central held-out prediction is comparator
prefill service plus the 100,000,000 ps handoff, imported decode service and
modeled queue wait, divided across four output tokens.

Each inclusive band adds and subtracts, before the divide by four, 15 percent
of decode service, one max-batch per-request service share, 25 percent of
modeled queue wait and 10 percent of comparator prefill service. This rule uses
the imported surface and queue model only.

## Frozen signed segment directions

The exact JSON freeze carries all 30 segment rows. The compact direction table
is:

| Configuration | 250 to 500 | 500 to 1K | 1K to 2K | 2K to 4K | 4K to 8K |
|---|---|---|---|---|---|
| `(1,1,8)` | decrease | decrease | increase | increase | increase |
| `(1,1,16)` | decrease | decrease | increase | increase | increase |
| `(1,2,8)` | decrease | decrease | decrease | increase | increase |
| `(1,2,16)` | decrease | decrease | decrease | increase | increase |
| `(2,1,8)` | decrease | decrease | increase | increase | increase |
| `(2,1,16)` | decrease | decrease | increase | increase | increase |

These directions withdraw the old globally monotonic claim in advance: the
expected mechanism is batch amortization before the queue-wait knee, then
increasing delay after the knee.

## Held-out quantitative bands

Values are per-token request delay in milliseconds. Every interval was frozen
before the concurrent run.

| Configuration | Load | Prediction | Inclusive band |
|---|---:|---:|---:|
| `(1,1,16)` | 250 | 0.717 | [0.556, 0.879] |
| `(1,1,16)` | 500 | 0.450 | [0.328, 0.571] |
| `(1,1,16)` | 1,000 | 0.290 | [0.193, 0.388] |
| `(1,1,16)` | 2,000 | 3.806 | [2.829, 4.782] |
| `(1,1,16)` | 4,000 | 5.775 | [4.306, 7.243] |
| `(1,1,16)` | 8,000 | 6.759 | [5.044, 8.474] |
| `(1,2,16)` | 250 | 1.164 | [0.936, 1.393] |
| `(1,2,16)` | 500 | 0.717 | [0.556, 0.879] |
| `(1,2,16)` | 1,000 | 0.450 | [0.328, 0.571] |
| `(1,2,16)` | 2,000 | 0.290 | [0.193, 0.388] |
| `(1,2,16)` | 4,000 | 2.048 | [1.511, 2.585] |
| `(1,2,16)` | 8,000 | 3.032 | [2.249, 3.816] |
| `(2,1,8)` | 250 | 0.712 | [0.551, 0.873] |
| `(2,1,8)` | 500 | 0.445 | [0.324, 0.566] |
| `(2,1,8)` | 1,000 | 0.285 | [0.188, 0.382] |
| `(2,1,8)` | 2,000 | 3.801 | [2.825, 4.777] |
| `(2,1,8)` | 4,000 | 5.770 | [4.302, 7.238] |
| `(2,1,8)` | 8,000 | 6.754 | [5.040, 8.468] |
| `(2,1,16)` | 250 | 0.717 | [0.556, 0.879] |
| `(2,1,16)` | 500 | 0.450 | [0.328, 0.571] |
| `(2,1,16)` | 1,000 | 0.290 | [0.193, 0.388] |
| `(2,1,16)` | 2,000 | 3.806 | [2.829, 4.782] |
| `(2,1,16)` | 4,000 | 5.775 | [4.306, 7.243] |
| `(2,1,16)` | 8,000 | 6.759 | [5.044, 8.474] |

## Preservation and decision rule

The freeze locks all six CORE-51 control files, all nine existing deterministic
concurrent comparator files and 17 scored flagship artifacts. A byte or count
change is fatal.

Every observed segment is compared with its frozen sign, and every held-out
point is compared with its inclusive band. No tolerance or direction changes
after observation. If all 30 observed segments increase, the monotonic claim
validates. Otherwise it is explicitly withdrawn in favor of the measured
batch-amortization then queue-wait-knee mechanism. Any band refutation is
published without widening.
