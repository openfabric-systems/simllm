# Imported-surface concurrent load-delay result

The frozen held-out bands are honestly refuted: 1 of 24 held. All six curves
increase on all five adjacent offered-load segments, so the monotonic-delay
claim is validated, but only 16 of the 30 pre-run signed segment predictions
match. The observed queue-wait onset is already present in the first 250 to
500 requests/s segment for every curve. That refutes the predicted knees of
1,056.619 requests/s with one decode engine and 2,113.237 requests/s with two.

The replacement mechanism is scheduler queue wait dominating every observed
segment. Measured batch service per token still falls strongly with load, from
about 0.98 ms to 0.24 ms on the one-decode curves, but mean scheduler queue
wait rises from about 1.50 ms to 26 to 27 ms. The batching gain is real and is
reported separately; it never outweighs queue growth in this sweep.

This scored freeze is `REFUTED`, not void. Every fatal guard held. However,
the source-access history is `CONTAMINATED` because a broad reconnaissance
command crossed the candidate-record directory before the field-addressed
reader was committed. No exposed value informed the freeze or score, but the
literal protocol was broken, so VLLM-39 stays open and VLLM-40 owns a clean
repetition.

## Frozen study and imported surface

The expectations-only freeze is commit
`121345e950b12a36018404084c7dcf9bd507f962`, with expectations SHA-256
`28cee81deffe771836b5c38d7fe605185f4dc31a953087c80288ceb7a3a84e22`.
It froze 250, 500, 1,000, 2,000, 4,000 and 8,000 offered requests/s, 64
requests per cell, 8-token and 16-token prompts, four output tokens and pool
ratios 1:1, 1:2 and 2:1. Prompt length 16 and the 2:1 pool ratio were held out,
yielding 24 unique held-out points.

The successful reader selected exactly two Granite CUDA-graph decode rows
from candidate record
`ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52`:

| Batch | Entry-key SHA-256 | Evidence | Replays | Service | Trimmed CV |
|---:|---|---|---:|---:|---:|
| 1 | `d8fdebd7051f530bed3232cfeb2e3d5c87a1ab9a22fe68db9cca51436853a502` | MEASURED, calibration split | 300 | 1,110,576,000 ps | 4,232 ppm |
| 8 | `38978457b27e56dbc0e9ffbe2385b7d53538a2515c826284be4d6d414520c830` | MEASURED, calibration split | 300 | 1,892,831,500 ps | 1,538 ppm |

The surface linearly interpolates in log batch and log total service. It
carries the record's `candidate` acceptance status and makes no calibration
claim. The successful reader consumed 45,043 of 57,417 bytes without loading
the whole record, did not decode the source batch-32 held-out row and did not
decode or capture a DeepSeek row. The complete ledger, including the disclosed
pre-protocol incident and two rejected reader attempts, has SHA-256
`0394d2789a11e8dc68c6d3a18c563d19f493d1d27c21d53b3ea74f37b3d14fec`.

The decomposition was also frozen. Batching gain is the imported decode
service divided by the number of scheduled token visits. Scheduler queue wait
is arrival-to-prefill admission plus handoff-completion-to-decode admission.
Prediction bands combined surface dispersion with the declared queue model;
they were not derived from the observed curves.

## Curve and direction verdicts

Per-token request delay, in milliseconds, is:

| Configuration | 250 | 500 | 1,000 | 2,000 | 4,000 | 8,000 |
|---|---:|---:|---:|---:|---:|---:|
| 1:1, prompt 8 | 1.622496 | 2.116871 | 2.763767 | 5.944352 | 7.652547 | 8.537305 |
| 1:1, prompt 16 | 1.636350 | 2.141553 | 2.815675 | 6.200720 | 7.948210 | 8.831093 |
| 1:2, prompt 8 | 2.691722 | 3.582877 | 4.429498 | 7.139448 | 8.707573 | 9.682686 |
| 1:2, prompt 16 | 2.710309 | 3.600478 | 4.558178 | 7.419431 | 8.990117 | 9.961276 |
| 2:1, prompt 8 | 1.622496 | 2.116871 | 2.952663 | 6.074741 | 7.667739 | 8.421157 |
| 2:1, prompt 16 | 1.636350 | 2.141553 | 2.992581 | 6.310878 | 7.967333 | 8.728289 |

Each arrow describes the adjacent segments 250 to 500, 500 to 1,000, 1,000
to 2,000, 2,000 to 4,000 and 4,000 to 8,000 requests/s:

| Configuration | Frozen directions | Observed directions | Matches |
|---|---|---|---:|
| 1:1, prompt 8 | down, down, up, up, up | up, up, up, up, up | 3/5 |
| 1:1, prompt 16 | down, down, up, up, up | up, up, up, up, up | 3/5 |
| 1:2, prompt 8 | down, down, down, up, up | up, up, up, up, up | 2/5 |
| 1:2, prompt 16 | down, down, down, up, up | up, up, up, up, up | 2/5 |
| 2:1, prompt 8 | down, down, up, up, up | up, up, up, up, up | 3/5 |
| 2:1, prompt 16 | down, down, up, up, up | up, up, up, up, up | 3/5 |

All 30 movements are positive, with no flat or decreasing segment. The
monotonic-delay claim therefore validates across the complete frozen grid.
The pre-run queue model is refuted because it placed a batching-dominated
region inside the sweep. The live stock schedulers instead accumulate enough
queue wait to dominate immediately.

The decomposition endpoints, again in milliseconds, make that separation
explicit:

| Configuration | Batch service/token, 250 to 8,000 | Scheduler queue wait, 250 to 8,000 |
|---|---:|---:|
| 1:1, prompt 8 | 0.981284 to 0.245349 | 1.496204 to 26.075373 |
| 1:1, prompt 16 | 0.977789 to 0.245349 | 1.506372 to 27.002328 |
| 1:2, prompt 8 | 0.991767 to 0.258692 | 2.526001 to 24.940673 |
| 1:2, prompt 16 | 0.988272 to 0.258692 | 2.524805 to 25.710394 |
| 2:1, prompt 8 | 0.981284 to 0.244783 | 1.496204 to 25.587938 |
| 2:1, prompt 16 | 0.977789 to 0.244783 | 1.506372 to 26.562144 |

## Held-out band verdicts

All values are per-token request delay in milliseconds. Bands remain exactly
as frozen; none was widened after observation.

| Configuration | Load | Frozen band | Observed | Verdict |
|---|---:|---:|---:|---|
| 1:1, prompt 16 | 250 | [0.555510, 0.878547] | 1.636350 | miss |
| 1:1, prompt 16 | 500 | [0.328441, 0.571336] | 2.141553 | miss |
| 1:1, prompt 16 | 1,000 | [0.192823, 0.387853] | 2.815675 | miss |
| 1:1, prompt 16 | 2,000 | [2.829466, 4.782258] | 6.200720 | miss |
| 1:1, prompt 16 | 4,000 | [4.306028, 7.243195] | 7.948210 | miss |
| 1:1, prompt 16 | 8,000 | [5.044310, 8.473664] | 8.831093 | miss |
| 1:2, prompt 16 | 250 | [0.935699, 1.392921] | 2.710309 | miss |
| 1:2, prompt 16 | 500 | [0.555510, 0.878547] | 3.600478 | miss |
| 1:2, prompt 16 | 1,000 | [0.328441, 0.571336] | 4.558178 | miss |
| 1:2, prompt 16 | 2,000 | [0.192823, 0.387853] | 7.419431 | miss |
| 1:2, prompt 16 | 4,000 | [1.511144, 2.585055] | 8.990117 | miss |
| 1:2, prompt 16 | 8,000 | [2.249426, 3.815524] | 9.961276 | miss |
| 2:1, prompt 8 | 250 | [0.551119, 0.873181] | 1.622496 | miss |
| 2:1, prompt 8 | 500 | [0.324050, 0.565970] | 2.116871 | miss |
| 2:1, prompt 8 | 1,000 | [0.188433, 0.382487] | 2.952663 | miss |
| 2:1, prompt 8 | 2,000 | [2.825076, 4.776892] | 6.074741 | miss |
| 2:1, prompt 8 | 4,000 | [4.301638, 7.237830] | 7.667739 | miss |
| 2:1, prompt 8 | 8,000 | [5.039920, 8.468298] | 8.421157 | held |
| 2:1, prompt 16 | 250 | [0.555510, 0.878547] | 1.636350 | miss |
| 2:1, prompt 16 | 500 | [0.328441, 0.571336] | 2.141553 | miss |
| 2:1, prompt 16 | 1,000 | [0.192823, 0.387853] | 2.992581 | miss |
| 2:1, prompt 16 | 2,000 | [2.829466, 4.782258] | 6.310878 | miss |
| 2:1, prompt 16 | 4,000 | [4.306028, 7.243195] | 7.967333 | miss |
| 2:1, prompt 16 | 8,000 | [5.044310, 8.473664] | 8.728289 | miss |

The sole held point is 2:1, prompt 8 at 8,000 requests/s. The 23 misses are
not treated as fatal guards. They refute the frozen predictive queue model and
motivate VLLM-41's lower-load onset identification.

## Conservation and preservation

All 36 cells conserve exactly: 2,304 admissions, 2,304 handoffs, 2,304
terminals and 9,216 terminal decode tokens, with a maximum TTFT decomposition
residual of 0 ps. The one-request CORE-51 live control remains exact at
273,376,000 ps TTFT and 77,952,000 ps TPOT, including 95,424,000 ps prefill,
100,000,000 ps handoff and 77,952,000 ps first-token decode service.

The frozen artifact locks all held:

| Preservation class | Artifacts | Bytes | Manifest SHA-256 |
|---|---:|---:|---|
| CORE-51 one-request control | 6 | 61,248 | `092d79c35c7632e87427804cda11bc6fe0890d2c98ceec23ff30af2e1143ad4d` |
| Deterministic concurrent comparator | 9 | 56,495 | `d09202846afeba9efc019d4f44f881fde6639457c0398f1ed4ae8da1e0c804c3` |
| Scored flagship artifacts | 17 | 1,198,680 | `7630ebdaf91a722ff5004184a03a38fac98bbf11f2adbbfd5e8e32838ff130d5` |

The request-heavy external raw result is 7,169,930 bytes with SHA-256
`1521181817ac942318a6fda589b980ee8a5bf523853f19e17a2cf345652dc583`.
The tracked compact record retains every curve point, decomposition row,
segment verdict, held-out comparison and provenance lock without the 2,304
request timelines.

## Registry ruling

VLLM-39 remains open because this attempt cannot satisfy the clean exposure
protocol, despite exact conservation and a validated monotonic direction.
VLLM-40 owns a fresh clean repetition with the already frozen reader and
surface. VLLM-41 owns the distinct model residual: extend the sweep below 250
requests/s and identify the stock-scheduler queue-wait onset that the frozen
queue model placed too high.

VLLM-35 gains the measured mechanism and a monotonic six-curve result, but it
does not close ahead of the clean VLLM-40 qualification. The deterministic
comparator remains historical, byte-identical evidence rather than being
rewritten to match this surface-priced run.
