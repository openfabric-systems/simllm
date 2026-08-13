# Congestion-bearing live chain expectations

This document freezes the BACK-38 and BRIDGE-2 acceptance study before the
online co-simulator client is implemented and before any result-producing
session invocation. The only command executed before this freeze is the
artifact-free `--check-only` command recorded below.

## Scope and chronology

The study replays the accepted case-A, EP-8, 100 Gbit/s mission schedule from
`end_to_end_replay_v1`. That source schedule came from a real vLLM scheduler
with the pinned CPU oracle, arrival gating, captured per-token routing and
three requests. Replaying its exact scheduler decisions keeps the GOAL and
arrival schedule identical between the accepted `rnic-nn-fluid` comparator and
the new `rnic-cn` treatment. The source artifacts are bulk inputs supplied by
`--mission-run`; this tracked study contains no site path.

The source `StepRecord` bytes remain immutable inputs, but their absolute
`virtual_time_ps` values are fluid-result timestamps and cannot be reused as
treatment completion timestamps. All 35 adjacent source records release
exactly at the preceding fluid completion. For treatment record `i`, the
effective release is therefore frozen as
`max(source[i].virtual_time_ps, treatment[i - 1].completed_at_ps)`, with the
first release equal to its source value. The projected record changes no field
other than `virtual_time_ps`. This is the closed-loop causal recurrence: it
preserves every declared arrival and scheduler decision, prevents an active
request from moving backward when congestion lengthens a step, and reduces to
the source record byte for byte when treatment and source completion agree.

This branch is gated on HTSIM-8 and HTSIM-25. The expectations-only commit is
allowed before that gate turns green. Behavioral implementation and the first
session invocation are not. A valid run must record evidence that the complete
default backend gate exited zero before implementation began. If that evidence
does not exist, the study stops and neither BACK-38 nor BRIDGE-2 can close.

The study claims the first congestion-controlled per-request TTFT and TPOT
reaching the supported metric chain from the accepted mission replay. It does
not claim hardware calibration, an NCCL latency calibration, a nonzero host
step cost, a different GPU envelope, or a framework scheduling response to the
new timing. The fixed replay deliberately holds framework decisions constant
so the network-profile comparison has identical GOAL bytes.

## Pre-freeze source audit

The source audit was completed before this file was written:

- `docs/modules/backends.md:851-858` registers BACK-38. It requires retained
  topology, RNG, transport, congestion-control and RNIC state, one checked
  graph projection with every artifact and completion reconciled, and the
  rejection plus stateless bytes as explicit off paths.
- `docs/modules/core.md:994-1019` registers BRIDGE-2 above HTSIM-18 and CORE-24.
  It requires graph dependency to horizon lowering, canonical lifecycle
  events, exact cursor-aware bookkeeping appends, `ExecutionResult`, full
  `StepResult`, transactional publication and explicit diagnostic off paths.
- `examples/persistent_session_v1/expectations.md:26-39` identifies the native
  event list, flow runtime, source calendar and RNIC authority whose lifetime
  discriminates a real session from process reset.
- `examples/persistent_session_v1/expectations.md:72-87` defines the inclusive
  `through_ps` causal boundary. Lines 114-162 freeze canonical framing and the
  `open`, `inject`, `advance`, `drain` and `close` verbs. The `advance`
  response owns newly visible accepted, queued, started and completed
  projections.
- `simllm/backends/step_sink.py:13-19` documents the current isolated process
  boundary. Lines 768-778 reject multi-artifact `rnic-cn` before backend
  execution.
- `simllm/backends/step_attribution.py:1-23` and lines 139-235 provide the
  accepted read-only projection from authoritative step timing to conserved
  per-request TTFT and TPOT. This study reuses it and does not grow another
  reducer.
- `examples/end_to_end_replay_v1/RESULTS.md:187-216` reports the accepted
  process baseline: 48 simulator invocations per step, about 0.36 seconds per
  invocation, and a practical scale ceiling near 100 steps. Lines 362-412
  report the absolute-timescale error budget and supported interpretation.

The backend flow session has no structural fluid authority, so
`rnic-nn-fluid` must remain outside the session. The treatment uses the
session-capable structural `rnic-cn` profile. A caller that selects online
session mode with `rnic-nn-fluid`, a custom topology the flow session cannot
identify exactly, a dependency cross-check requiring a second backend
authority, or any other uncomposable configuration must still fail before
backend mutation.

## Frozen source artifacts

`--mission-run` names the root containing `cells/a-ep8-100g` and `capture`.
The check-only gate requires these accepted source bytes:

| artifact | SHA-256 |
|---|---|
| `cells/a-ep8-100g/steps.jsonl` | `893fd939460556a6e0639572ef41db58b478c72850f3a43b62de626bbade5706` |
| `cells/a-ep8-100g/replay-run.json` | `61109794360613d04ffbc4ed4a2e4eee6fb06adbd005db68d50b84961bea8ef3` |
| `cells/a-ep8-100g/routed-experts.json` | `2c3e5c961ed432a6eb9633983d4ccedc2cd2c26ccb24415f9f4092fac7b58dba` |
| `capture/greedy.jsonl` | `ef570a67fd8bbbb6a8d73b8ad9f73171d3eaaf9a51efc04f676bd9f25c8988fe` |

The source carries 36 contiguous steps, 1,728 collective GOAL artifacts,
50,802,688 routed bytes and three requests. The accepted fluid run used 1,728
simulator processes and 600.23 seconds. Its per-request TTFT values are
`[903795888, 936896110, 1161066114]` ps in request order. Its exact rational
TPOT values are `7397037050/23`, `612085002/1` and `7934142022/31` ps. With the
nearest-rank percentile rule frozen below, fluid TTFT p50 is 936,896,110 ps and
p99 is 1,161,066,114 ps.

No backend commit literal is frozen. The result records the backend commit and
binary digest it observed, separately from the commit against which the Python
evidence was authored.

## Frozen cells and reductions

The session treatment has three cells:

| cell | profile | link rate | purpose |
|---|---|---:|---|
| `cn-100g-a` | `rnic-cn` | 100 Gbit/s | flagship comparison and first deterministic run |
| `cn-100g-b` | `rnic-cn` | 100 Gbit/s | identical deterministic repeat |
| `cn-200g` | `rnic-cn` | 200 Gbit/s | independent bandwidth sanity parameter |

Every cell consumes the same 36 source `StepRecord` values, arrivals,
routed-expert projection, expert placement and graph lowering. Each cell uses
the frozen causal recurrence above to derive its effective release time while
preserving every other `StepRecord` field. The two 100 Gbit/s cells use the
same seed and every other configuration byte. Each cell opens exactly one
backend process and one flow session for the whole replay. No retained packet
trace is requested.

For a sorted nonempty list of `n` per-request values, percentile `p` is the
nearest-rank element at one-based index `ceil(p * n)`. Thus p50 is the middle
request and p99 is the maximum for this three-request fixture. TPOT remains an
exact `Fraction` until the report renders it.

## Registered acceptance clauses

### BACK-38 clause B1: retained physical authority

The registered clause is:

> "preserve htsim topology, RNG, transport, congestion-control and RNIC state
> across ordered GOAL artifacts instead of starting a fresh process at every
> boundary."

One live process must accept all ordered collective artifacts in a step and all
36 steps. Session sequence numbers are contiguous over the whole replay.
Topology identity, seed, authority and policy are selected once at `open` and
never reconstructed per artifact. The result reports process count, open
count, injection count, sequence range and the session's retained authority
counters.

### BACK-38 clause B2: reconciliation and off paths

The registered clause is:

> "Acceptance must execute one checked graph projection in a state-preserving
> session, reconcile every artifact and completion identity, and retain the
> current rejection and stateless-profile bytes as the explicit off paths."

For every step, the checked graph projection's GOAL text must be byte-identical
to the corresponding accepted mission GOAL. Every projected physical message
maps one-to-one to a contiguous injected flow identity and one native
completion row. Source, destination, tag, bytes, operation identity and all
five lifecycle timestamps must agree. Missing, duplicate, foreign or regressed
rows abort before publication. Diagnostic `rnic-nn` and `rnic-nn-fluid`
planning bytes and timestamps stay byte-identical when online session mode is
off. Unsupported online configurations retain a pre-execution rejection.

### BRIDGE-2 clause C1: graph, lifecycle and result chain

The registered clause requires the client to:

> "lower live `ExecutionGraph` dependencies into flow injections and inclusive
> virtual-time horizons, translate the returned native lifecycle projections
> into canonical `CompletionEvent` values, append the exact object, stage and
> completion facts at the supplied bookkeeping cursor, construct
> `ExecutionResult`, reduce the full `StepResult`, and publish only after all
> identities, cursors, timestamps and quiescence evidence validate."

The study supplies a `RequestBookkeeper` and records its starting and ending
cursors per step. Each returned event and append fact is decoded through the
repository's canonical types. The candidate append is validated atomically on
a staged ledger. The live bookkeeper, session-visible outcomes and returned
full `simllm-step-result-v2` publish only after the candidate graph result,
event multiset, cursor, completion boundary and physical quiescence agree.

### BRIDGE-2 clause C2: framing, sequence and fail-closed publication

The registered clause requires contiguous framed inputs and rejection of:

> "loss, duplication, cursor disagreement, graph/event identity disagreement
> and timestamp regression before publishing a result."

Focused tests inject each fault into a fake framed backend. They must observe
the same pre-call ledger cursor, no published outcome and a terminal client
error. A separate pytest executes the accepted deterministic fixture twice,
serializes its per-request `simllm-step-result-v2` bytes and compares both runs
to one LF-stable byte-locked artifact. This pytest cannot require an initialized
submodule or a native binary.

## Behavioral relations

These relations are evaluated from raw per-request observations before any
fatal reconciliation oracle. None is entailed by the fatal guards: exact
identity and conservation do not constrain a congestion-sensitive percentile,
the sign of a bandwidth response or repeat determinism.

- **R1, median:** `cn-100g-a` TTFT p50 divided by the accepted fluid TTFT p50
  lies in `[1.0, 2.0]`. The lower bound reflects that the physical treatment
  must not beat the identical ideal baseline; the upper bound is the standing
  rnic-cn acceptance bar.
- **R2, tail:** `cn-100g-a` TTFT p99 is strictly greater than fluid TTFT p99.
  The upward side is deliberately open. Equality or a decrease is a finding
  that retained congestion state has not reached the metric chain.
- **R3, bandwidth:** for identical replay inputs, `cn-200g` TTFT p50 and p99
  are each no greater than `cn-100g-a`, and at least one is strictly lower.
  The same signed relation is checked for TPOT p50 and p99. This is one
  behavioral family with four genuine-risk instances.
- **R4, determinism:** the ordered per-request TTFT, exact TPOT numerator and
  denominator, token count and attribution partitions from `cn-100g-a` and
  `cn-100g-b` are byte-identical. Wall time and process identifiers are
  excluded. This is one genuine-risk instance.

R1 and R2 are separate families, so a passing median cannot carry a flat or
failed tail. R3 varies link rate independently of the profile comparison. R4
tests the retained RNG and tie-breaking path, not merely deterministic JSON
formatting.

## Fatal guards and structural invariants

Fatal means void. A single violation makes every behavioral score
uninterpretable and leaves both owning tasks open. Fatal guards are never
reported as a fraction and do not enter a behavioral denominator.

- the backend HTSIM-8 plus HTSIM-25 gate was green before implementation and
  before the first session invocation;
- all source hashes, graph identities, GOAL bytes, arrivals and routing bytes
  equal the frozen inputs, and every effective treatment release obeys the
  frozen causal recurrence with no other `StepRecord` field changed;
- one process and one session serve each treatment cell, with no sequence gap,
  duplicate injection, missing completion or foreign completion;
- canonical lifecycle timestamps are monotonic and agree exactly with native
  rows; `QUEUED` maps to eligibility and `STARTED` maps to the resource grant;
- every step's staged bookkeeping append starts at the supplied cursor, ends
  at the returned cursor and passes the existing ledger validator before the
  live ledger changes;
- `ExecutionResult` completion, physical quiescence and full `StepResult`
  completion agree with the authoritative native boundary;
- the sum of composed artifact services equals step latency, and every
  per-request attribution and exact TPOT relation conserves;
- all 36 source steps and all three request totals publish, and every backend
  drain is physically quiescent;
- online fluid mode and every other frozen uncomposable configuration reject
  before backend mutation;
- diagnostic `rnic-nn` and `rnic-nn-fluid` off-path bytes and timestamps match
  their pre-change controls;
- the tracked deterministic fixture is LF-stable and its pytest byte lock
  passes without `third_party/` content.

## Physical sanity bounds stated before the new values are read

These bounds are defect detectors. Being inside a band is not proof of
correctness; being outside is a defect in the model, harness or reading.

The three prompts have 19, 15 and 18 tokens. One token-layer phase sends one
2,048-byte vector to at least one remote owner. At 100 Gbit/s one byte takes
80 ps. Even the smallest prompt therefore needs at least
`15 * 24 * 2 * 2048 * 80 ps = 117,964,800 ps` of source serialization. Adding
the 60 us frozen compute floor gives a first-token floor of 177,964,800 ps.
Every measured TTFT must exceed 0.177 ms.

Across the whole replay there are at most 108 forwarded prompt and decode
tokens. Serializing the worst-case seven remote vectors for all 24 layers and
both phases on one 100 Gbit/s link costs at most
`108 * 24 * 2 * 7 * 2048 * 80 ps = 5,945,425,920 ps`. Charging every possible
directed message an additional 10 us, serializing all 36 compute steps at the
200 us frozen ceiling, and adding the 2 ms arrival span remains below 0.14 s.
The deliberately looser 0.25 s ceiling covers integer boundaries and control
events. Every TTFT and the entire replay makespan must stay below 0.25 s.

Each decode interval must read the resident weights. The accepted B100
envelope gives a 60 us compute floor and a 200 us compute ceiling per step.
Every positive TPOT must be at least 60 us and below 0.25 s.

For each native flow, `payload_bytes * 8 / link_rate` is a serialization floor
that its FCT cannot beat. At 200 Gbit/s that floor is exactly half its 100
Gbit/s value. R3 checks the end-to-end quantity that should scale in the same
direction, independently of the per-flow fatal bound.

The first accepted fluid 100 Gbit/s prefill completes near 0.9 ms, which lies
between its own serialization-plus-compute floor and the broad 0.25 s ceiling.
The treatment is plausible only if its median remains in R1 while a retained
queue or control history moves R2 upward. A 50,000 token/s implied decode rate
would still be implausible for a comparable real deployment even if internally
consistent.

Three independent sanity angles are reported: network serialization per flow,
compute and memory bounds per decode step, and end-to-end implied token rate
against a comparable deployment.

## Absolute-timescale budget

This task changes the congestion-bearing network distribution. It does not
change the other two dominant missing terms from the mission study:

- fixed host initiation remains 0 ps versus a plausible 0.3 to 3 ms;
- the uncalibrated flat 0.7 roofline derate and B100 default remain unchanged.

The ideal fluid collective term is the before value. The result reports, for
each request and at p50 and p99, fluid collective attribution, rnic-cn
collective attribution, their difference and the recomposed TTFT and TPOT
budget. It also reports whether the physical profile retains a 2 us minimum or
replaces it with a different observed floor. The previous and new composed
decode budgets must be stated explicitly rather than hiding the network change
inside total TTFT.

## Evidence classes

- **Run configuration:** source digests, observed backend and Python commits,
  binary digest, profile, link rate, seed, process/session counts and wall time.
- **Behavioral relations:** R1, R2, the four R3 instances and R4. The headline
  reports family outcomes and the `7` genuine-risk instances without adding
  any fatal or exact-oracle count.
- **Exact-oracle evidence:** per-artifact GOAL identity, flow/message identity,
  lifecycle/event identity, ledger cursor append and full result wire identity.
  These are reported separately and are fatal when violated.
- **Structural invariants:** gate chronology, source hashes, quiescence,
  conservation, rejection and off-path identity. They are fatal-unscored.
- **Native and Python tests:** reported independently and never added to the
  behavioral denominator.
- **Wall evidence:** process count and elapsed seconds before and after are
  measured and reported. No speedup band or scored wall relation is registered.

## Closure and residual discipline

BACK-38 and BRIDGE-2 are evaluated independently against the quoted clauses
above. A valid result may close either, both or neither. A strong BACK-38
retention result cannot close a weak BRIDGE-2 cursor or publication result.

Only a registered acceptance clause not demonstrated by the valid run may
move to BACK-40, BACK-41 or BRIDGE-4. Adjacent improvements remain RESULTS
prose. The report states how many residual IDs were registered and quotes the
clause requiring each one. Closure removes the owning task from its module doc,
adds it to `docs/task-ledger.json`, regenerates task progress and performs the
required contradiction sweep. Hits in `README.md`, `docs/README_PRO.md` and
`docs/architecture.md` are reported but not edited by this worker.

## Reproduction contract

The pre-freeze dry run is:

```bash
.venv/bin/python examples/congestion_chain_v1/run_study.py \
  --mission-run "$SIMLLM_MISSION_RUN" \
  --htsim-rnic "$SIMLLM_HTSIM_RNIC" \
  --run-dir "$SIMLLM_RUN_ROOT/congestion_chain_v1" \
  --check-only
```

It validates only frozen inputs and arithmetic, creates no directory, invokes
no backend and produces no measured value. After the expectations commit and
the green backend gate, removing `--check-only` runs the three cells. The final
implementation must refuse to run into an existing output directory.
