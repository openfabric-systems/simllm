# CORE-5 integration-review correction expectations

## Freeze status

This review-triggered expectations-only supplement precedes the correction of
the reducer and study harness. It adds regression requirements to
[`expectations.md`](expectations.md) without weakening the original CORE-5
relations. The probe observations motivating these corrections are already
known, so these checks are post-specified regressions, not new
pre-registrations. No corrected implementation or corrected measured result
is included in this commit.

## Reducer replay and sampling corrections

One `CompletionReducer` consumes an `ExecutionResult.execution_id` at most
once. A second reduction of that identity raises `ValueError` containing
`execution ID has already been reduced` before request history, token counts,
pending attribution, or `VirtualClock` changes. This applies when the first
step has positive latency and when it has zero latency. A failed first attempt
does not consume the identity.

If `StepRecord.sampled_request_ids` is absent and `num_sampled == 0`, the
sampled set is exactly empty even when scheduled rows are decode rows. The
step advances normally, emits no request metric, and retains the interval for
the request's next sampled boundary. This is the unambiguous zero-sample case
already described in the original freeze.

In the v1 JSON reader, an explicit JSON null for `sampled_request_ids` has the
same meaning as an absent field. It loads as Python `None`; the canonical
writer continues to omit the field when its value is `None`. A non-null value
must still be an array accepted by `StepRecord` validation.

## Measured scored evidence

Every one of the original 18 scored instances is evaluated from objects
returned by the executed runtime and reducer. Frozen tables are expected
values only. In particular:

- dependency-shape deltas subtract measured `StepResult.step_latency_ps`
  values from the serial and parallel cells;
- inverse-rate deltas subtract measured `StepResult.step_latency_ps` values
  from the 200 and 400 Gbit/s cells;
- component checks compare each measured request attribution vector with its
  frozen vector;
- additive separation compares measured additive totals with measured request
  latency and attribution; and
- asynchronous checks subtract the two measured completion boundaries.

Each scored check record contains its observed value, expected value or band,
pass result, and `genuine_risk` boolean. Family passed, total,
genuine-risk-instance count, and fraction are derived from those executed
records. No family count or risk numerator is passed as a literal to the
summary function.

## Evidence-class accounting

The report keeps these classes separate:

- exact-oracle row publication;
- scored behavioral check records;
- in-harness structural predicates evaluated on live runtime output;
- expected validator rejection probes;
- compatibility acceptance probes;
- repository unit tests; and
- repository gates.

The 48 original runtime-stream predicates are counted only if the harness
records each actual predicate evaluation: callback length, callback object
identity, complete event-phase membership, graph additive queue wait, and
realized critical-path queue wait. Combined expressions must be split into
their evaluated predicates, so the corrected total is derived from the record
list and need not remain 48.

Validator rejection probes report caught exception type and message and are
not added to the in-harness structural count. Compatibility acceptance probes
for zero sampled decode rows and JSON null likewise remain a separate,
unscored class. A producer-supplied or hardcoded pass total is not evidence.

## Registered check-only command

Before this supplement is committed, run:

```bash
.venv/bin/python examples/rnic_live_v1/tier_b_review_check.py \
  --out "$SIMLLM_CORE5_RUN_ROOT/tier_b" \
  --producer "$SIMLLM_RNIC_TIER_B_PRODUCER" \
  --check-only
```

At freeze time this harness may be untracked and may contain only frozen
literals and check-only validation. It prints a registry confirmation by
design and produces no artifacts.

## Post-specified filesystem portability note

This note was added after the supplementary freeze and changes no regression,
schema, chronology, or historical dry run. The one-off environment-variable
spellings above remain frozen text. After loading `.env.local.sh`, the current
portable rendering is:

```bash
.venv/bin/python examples/rnic_live_v1/tier_b_review_check.py \
  --out "${SIMLLM_DATA_ROOT:?configure SIMLLM_DATA_ROOT}/core5_reduction/tier_b" \
  --producer "${SIMLLM_DATA_ROOT}/core5_reduction/tier_b/build/htsim_rnic_tier_b" \
  --check-only
```

The resolved historical machine-local paths are intentionally omitted.
