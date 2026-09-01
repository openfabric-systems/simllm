# VLLM-41 lower-load queue-onset expectations

Status: expectations only. No VLLM-41 concurrent-session cell has run, and no
VLLM-41 curve value is an input to this freeze.

## Frozen offered-load ladder

The ladder is 50, 100, 150, 175, 200, 210, 220, 225, 230, 235, 240, 245 and
250 offered requests/s. Every cell contains 64 deterministic arrivals, four
decode output tokens, a maximum stock-scheduler batch of eight and prompt
lengths 8 and 16. The pool ratios are 1:1, 1:2 and 2:1, for 78 cells and 4,992
request lifecycles.

The 240 requests/s load is held out. The entire 2:1 pool ratio is also held
out. Their union contains 30 cells, including sub-250 loads and a pool ratio
that cannot influence any pre-run model choice.

## Imported measured surface

The committed field-addressed reader was reused without edits. Its fresh
five-event ledger is `access_ledger.jsonl`, SHA-256
`8a61a6e0b58a213259a19a593b8b3f4ec08cea6a7c854f5481ccbd7bc2dc5914`.
It returned acceptance status, campaign ID, device kind ID and exactly the
Granite CUDA-graph decode batch-1 and batch-8 rows. It stopped at byte 45,043
of 57,417. It did not load the whole record, decode or capture the Granite
batch-32 row, or decode or capture a DeepSeek row.

| Batch | Service (ps) | Trimmed CV | Replays | Evidence |
|---:|---:|---:|---:|---|
| 1 | 1,110,576,000 | 4,232 ppm | 300 | MEASURED, calibration split |
| 8 | 1,892,831,500 | 1,538 ppm | 300 | MEASURED, calibration split |

The reconstructed projection is byte-identical to the committed imported
surface with SHA-256
`26fc547d8b47ccec7108872e05fbedfe71ebb6229b88799ca254089d3f2b6e9d`.
Its acceptance status remains `candidate`, and its calibration claim remains
false.

## Queue-model derivation and quantitative bands

The only numerical inputs are the two logged measured service and CV fields
and the frozen deterministic arrival process. No prior or VLLM-41 delay curve,
fit coefficient, fitted knee or comparator service enters the model.

For offered load `lambda`, request `i` arrives at
`floor(10^12 / lambda) * i` ps. One shared virtual clock selects one nonempty
prefill or decode engine at a time in work-conserving round-robin order.
Prefill and handoff are zero-cost admission boundaries in this numerical
abstraction. A decode request retains four visits, each decode engine schedules
at most eight requests per visit, and batch service is the unchanged imported
power-law interpolation `S(b)`.

Before batching, queue demand first equals interarrival at

`lambda_0 = 10^12 / (4 * S(1)) = 15,625,000 / 69,411 = 225.108412 requests/s`.

The uncertainty calculation scales every interpolated service by plus or
minus three times the largest imported trimmed CV. The frozen 12,696 ppm
envelope gives an onset-rate band of 222.286266 to 228.003140 requests/s.
The central first queue-dominated segment is 225 to 230 requests/s. Surface
uncertainty admits exactly two first segments: 220 to 225 and 225 to 230.

For each point, the batch-service band is the min and max service-per-token
value across the lower, central and upper surface scenarios. The scheduler-wait
band takes the scenario min and max and adds at most one upper batch-8 service
residual at each of the two admission boundaries. All 30 exact held-out point
bands are frozen in `expectations.json`; none may be widened after execution.

At the held-out 2:1 ratio, representative prompt-8 predictions are:

| Load | Queue wait central (us) | Inclusive queue band (us) | Service/token central (us) | Inclusive service band (us) |
|---:|---:|---:|---:|---:|
| 220 | 0.000 | [0.000, 3,833.726] | 1,110.576 | [1,096.476, 1,124.676] |
| 225 | 0.000 | [0.000, 4,428.540] | 1,110.576 | [1,096.476, 1,114.060] |
| 230 | 607.574 | [0.000, 4,447.229] | 1,089.610 | [1,089.288, 1,089.610] |
| 240 | 581.352 | [0.000, 4,479.688] | 1,044.183 | [1,044.183, 1,046.823] |
| 250 | 524.686 | [0.000, 4,390.463] | 1,005.744 | [1,003.325, 1,005.744] |

## Frozen decomposition and verdict rule

Batching service per token is the sum of imported `S(batch)` over every
stock-scheduler decode batch divided by all scheduled request-token visits.
It is reported separately from both wait components.

Arrival-to-prefill wait is `prefill_eligible_at_ps - admitted_at_ps`.
Handoff-to-decode admission wait is
`decode_eligible_at_ps - handoff.completed_at_ps`. Scheduler queue wait is the
mean sum of those two fields. Provider service is never added to that wait.

For an adjacent segment, scheduler-wait change is divided by four output
tokens and compared with the separately reported batch-service-per-token
change. A segment is queue-dominated only when wait rises and

`delta(wait / 4) + delta(batch service per token) > 0`.

The first such segment is reported for every configuration. VLLM-41 closes
only if all six configurations identify a first queue-dominated segment below
250 and have at least one earlier non-queue-dominated segment. Any held-out
component-band miss is registered on VLLM-42. Differing onset segments across
ratios or prompt lengths, or an unresolved configuration, are registered on
VLLM-43. Total per-token delay directions are neither computed nor rescored;
the validated VLLM-40 direction over 250 to 8,000 requests/s remains closed.

## Preservation locks

| Class | Files | Bytes | Manifest SHA-256 |
|---|---:|---:|---|
| Prior VLLM-39/40 load-delay lineage | 17 | 279,928 | `ae964f9ccecc2554764f9ef69300ca06a84c4a8609682c678063f73c0d41538d` |
| CORE-51 one-request control | 6 | 61,248 | `092d79c35c7632e87427804cda11bc6fe0890d2c98ceec23ff30af2e1143ad4d` |
| Deterministic concurrent comparator | 9 | 56,495 | `d09202846afeba9efc019d4f44f881fde6639457c0398f1ed4ae8da1e0c804c3` |
| Current scored flagship artifacts | 26 | 1,715,149 | `375d2359e0c9dff9cae98c576eaf8a9e24b0c7621b0af0dcfde187662c57955b` |

Any byte, file count or selection change is fatal before execution.
