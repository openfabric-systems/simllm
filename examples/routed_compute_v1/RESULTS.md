# Routed expert computation and declared kernel minimums

**Routed expert load now changes request latency through causal GPU work.**
Twelve synthetic configurations run thirty-six actual model step calls through
the standard device runtime and request reducer. All thirty-six expert-service
vectors and sixteen behavioral instances pass, with no fatal violation.
This qualifies the mechanism slice of COMP-7 and COMP-43. Both tasks retain
their calibrated acceptance obligations.

With eight requests and no declared kernel minimum, balanced routing gives
73.728512 microseconds to the first token. Sending all expert rows to one
GPU gives 98.305024 microseconds. The extra 24.576512 microseconds consists of
24.576 microseconds of critical expert work and 0.000512 microseconds of extra
communication. Both parts are visible in the executed graph and runtime.

## Physical interpretation

An expert consumes each token routed to it. Several selected experts on one
GPU can reuse one transmitted input vector, but they still perform separate
matrix products. The shared routing projection therefore retains every expert
assignment before deduplicating communication destinations. It uses one
placement epoch for both the traffic and compute work.

For hidden width 16 and expert width 32, a grouped gate/up projection takes
`4 * 16 * 32 * rows` floating-point operations. The down projection takes
`2 * 16 * 32 * rows`. At the deliberately declared one-billion-operation-per-
second compute rate, four routed rows take 8.192 and 4.096 microseconds;
eight rows take 16.384 and 8.192 microseconds. Across two layers, concentrating
eight requests on one owner adds exactly 24.576 microseconds of critical
compute service. The independent byte check accounts for 512 extra bytes at
the declared one-byte-per-picosecond endpoint rate, adding 512 picoseconds.

Floor: no expert projection beats its arithmetic work, active-weight stream
or selected minimum, and decode output cannot precede its causal transfers.
Ceiling: serializing every nonempty compute visit and twice every directed
transfer byte bounds each step. All thirty-six executed steps satisfy these
frozen bounds. The end-to-end model check joins dispatch completion to expert
eligibility, expert completion to combine, and final combine to output. The
runtime, completion result and request metrics agree at the terminal boundary.

The minimum is charged once to the grouped gate/up kernel and once to the
down kernel on each active owner. It is a maximum with their individual
provider estimates. It is not multiplied by expert count or added to the
whole step. An idle owner runs no expert kernel and pays no minimum.

| Requests | Routing | Declared minimum per expert kernel | TTFT | Mean TPOT |
|---|---|---:|---:|---:|
| 8 | Balanced | 0 | 73.728512 | 75.264512 |
| 8 | Hot owner | 0 | 98.305024 | 99.841024 |
| 8 | Balanced | 5 | 75.536512 | 77.072512 |
| 8 | Hot owner | 5 | 98.305024 | 99.841024 |
| 8 | Balanced | 20 | 129.152512 | 130.688512 |
| 8 | Hot owner | 20 | 129.153024 | 130.689024 |
| 16 | Balanced | 0 | 147.457024 | 150.529024 |
| 16 | Hot owner | 0 | 196.610048 | 199.682048 |
| 16 | Balanced | 5 | 147.457024 | 150.529024 |
| 16 | Hot owner | 5 | 196.610048 | 199.682048 |
| 16 | Balanced | 20 | 178.305024 | 181.377024 |
| 16 | Hot owner | 20 | 203.842048 | 206.914048 |

All timing columns are microseconds. Time to first token (TTFT) comes from the
prefill sample; time per output token (TPOT) averages the two subsequent decode
intervals. Each configuration uses one prefill token and two decode tokens per
request. All requests share the step's output boundary.

At a twenty-microsecond minimum and eight requests, both routing choices pay
forty microseconds of expert service per layer. The work imbalance remains in
the row histogram but produces no extra compute delay in this declared regime.
The remaining 512-picosecond TTFT difference is communication. At sixteen
requests, the hot owner's gate/up work exceeds that minimum, and compute
imbalance becomes visible again. This is a mechanism demonstration, not a
measurement of a GPU's minimum kernel time.

## Evidence and reproduction

Expectation commit `fd97c1e627397aba6605742406a68c543fa6129c` precedes all
implementation and campaign execution. The executed source is
`a0ffab1668038caa66261947feb014f28f9dd4d7`. The first campaign passes without
changing the matrix, equations or acceptance bounds.

- Twelve declared configurations produce thirty-six real model-step outcomes.
- All 1,913 fatal guards hold. They cover source, routing, ownership, timing,
  byte, request and receipt integrity and are unscored.
- Thirty-six exact expert-service vectors agree with the frozen arithmetic
  and the actual GPU-work queue service intervals.
- Sixteen behavioral instances pass in three families: five load-imbalance,
  five minimum-service and six request-count comparisons.
- The focused software suite passes 123 cases, including packed/strict
  routing agreement on the existing Granite trace, unchanged fallback
  results, local-only routes and eleven recorded-output corruption controls.
  Software cases and forced equalities are not additional behavioral points.

The [portable publication](results.json) retains all 87 original file
receipts, totaling 3,774,920 bytes. Each data file receives an append-only,
flushed receipt before admission. Publication rehashes every retained file.
The raw summary SHA-256 is
`5a146e5cc5a00519ea58786d2399ff70c26394fca2b130a0a9c9b33add29d291`.

Reproduce from a clean checkout using external output storage:

```bash
python -m examples.routed_compute_v1.run_study --output "$SIMLLM_ROUTED_COMPUTE_RUN_ROOT"
```

## Project effect and limits

COMP-7 gains one routed demand authority and an executable attention, dispatch,
expert-compute, combine and output schedule. COMP-43 gains a declared minimum
per grouped expert invocation, bound to its complete GPU envelope, with an
exact zero/off path. An invalid provider estimate, conflicting observed schedule,
padded token count, non-ideal host model or tensor-parallel width above one
rejects before runtime mutation.

The graph uses whole-phase barriers and one token-owning engine. It models
each active expert's weights as one stream and retains the existing non-expert
arithmetic. COMP-7 and COMP-43 remain open for measured, architecture-qualified
acceptance on captured model work. COMP-45 owns a non-void calibration;
COMP-6 owns physical invocation, activation and fusion identities. TRAF-26
retains full peer-engine workload population. No GPU kernels or native
framework engines run in this study. CORE-51 and CORE-54 retain large-model
deployment and frontier qualification, including DeepSeek-V3 and Kimi K3.
