# Dispatch message sequence v1 results

TRAF-21 implements the explicit `captured-message-sequence` traffic level.
The strict v2 fixture preserved every framework-returned top-k position under
both declared grouping rules, retained exact per-request and ordered-pair byte
totals, and changed native packet completion in the required positive
direction. The accepted aggregate renderer and Granite GOAL stayed byte
identical.

The frozen magnitude predictions were partly falsified. The genuine-risk
headline is **1 of 3 family classes and 2 of 10 parameterized instances**.
Both inverse-rate instances passed. All four packet signed directions were
positive, but their byte-only bands omitted the backend's full-envelope packet
calendar. The four fluid comparisons also missed: per-token and expert-group
timing were identical to each other, but preserving capture order made both
faster than the ascending-pair aggregate order. These misses are retained as
findings, not rewritten into passes.

At Granite prefill scale, aggregate and per-expert-group rendering were
practical under the frozen limits. The per-token plan contained 101,318
messages. Planning completed in 1.86 s, but GOAL rendering exceeded the 30 s
limit and reached the controlled 60 s attempt boundary. TRAF-22 owns the
quadratic message-record validation bottleneck and the unexecuted Granite
per-token backend cells.

## Chronology and provenance

The expectations-only commit is
`7efd71e7e54fc6faecde17c5faebab9430a2e847`. Its registered command first ran
with `--check-only`; no SimLLM module or source artifact was read, no native
tool ran and no output path was created. Implementation followed in
`04eeb5c28472ce11cc5f421097a07a12aef8e28f`. Commit
`bd752cd8cbe3b6948c165a2e548b0c09e7cea9f2` added only a bounded Granite
render attempt after the first scale attempt exposed the performance problem.

The result chronology is preserved:

1. The registered converter path pointed to a stale 46,360-byte executable
   that segfaulted on the unchanged accepted Granite GOAL. It produced no
   backend observation. The partial directory is retained beside the result
   as `dispatch_sequence_v1-failed-stale-txt2bin`.
2. A converter from the same htsim source tree as the available RNIC build
   compiled the accepted GOAL. That run completed the full synthetic matrix
   and the aggregate and expert-group Granite cells. The per-token renderer
   was manually stopped after more than nine minutes in
   `GoalTrace.record_message`, which repeatedly rebuilt rank-label sets. Its
   directory is retained as
   `dispatch_sequence_v1-failed-unbounded-render`.
3. The final controlled run reproduced every synthetic completion exactly,
   kept the failed registered verdicts and stopped only the Granite per-token
   render at 60 s. It wrote `raw_observations.json` and `summary.json`, then
   exited nonzero because the frozen behavioral acceptance was not fully met.

No rerun changed a threshold or replaced a failed observation with a pass.
The final evidence lives under
`$SIMLLM_WAVE6_RUN_ROOT/dispatch_sequence_v1`.

| Provenance field | Observed value |
|---|---|
| Expectations commit | `7efd71e7e54fc6faecde17c5faebab9430a2e847` |
| SimLLM commit observed by final run | `bd752cd8cbe3b6948c165a2e548b0c09e7cea9f2` |
| htsim gitlink observed by final run | `fc4400e4ca619223481536632074045cb6af2756` |
| `htsim_rnic` SHA-256 | `cfb5014a663791f7619fe33309114a74e82878de860c14fc8a723713501f027d` |
| working `txt2bin` SHA-256 | `f3745f34ad86febe9c9eebef10aee5fae00b8865cb29943344fb75b0f142495b` |
| Final raw-observation SHA-256 | `82764f7a1e31dc32f31128d8647ebf548c83746e54911834dd2adae5882e312d` |
| Final summary SHA-256 | `634e1c51311c2e9330b4af6e94cbac8d4701a26a6cafe0820fd3cbdddf7b012b` |

The gitlink and executable hashes are observations. No test asserts that a
future live submodule pin equals a frozen literal.

## Implemented contract

`project_framework_routing` reads a strict `simllm-preplay-trace-v2` artifact
and copies request order, phase-local token order, layer order and each
framework-returned expert tuple without sorting. It records the source bytes'
SHA-256. It makes no claim about a later kernel, NCCL or RNIC issue order.

`step_moe_message_sequences` derives this base order for each layer:

1. scheduled request order;
2. selected phase-local token order;
3. returned top-k position, with the first position owning a repeated
   destination;
4. source-rank projection.

`per-token` emits one hidden vector per token and unique remote destination.
`per-expert-group` coalesces one request's whole-layer messages for one source
and destination, retaining the first contributing routing position. Combine
transposes every dispatch contribution without changing its routing ordinal.
The plan retains every contributing `(token_index, top_k_index)` tuple.

`render_sequenced_step_goal` emits the planned rows in tuple order and uses a
source-local `irequires` chain, so sources remain mutually unordered while
each source posts messages in capture-derived order. This is an explicit API,
not a global configuration switch. `render_step_goal` and
`step_moe_alltoalls` remain the aggregate default. CORE-36 remains the sole
owner of a future unified selector and fidelity record.

## Physical sanity before measured digits

The registered synthetic bounds were recorded before backend execution. Peak
incident payload is 18,432 bytes in each of two serial phases. Payload over
link rate gives floors of 1,474,560 ps at 200 Gbit/s and 737,280 ps at 400
Gbit/s. Serializing all per-token packet work at both endpoints gives the
conservative ceilings of 9,000,000 ps and 4,500,000 ps. Every one of the 12
cells lay inside its rate's bounds:

| Rate | Registered range ps | Measured range ps | Result |
|---:|---:|---:|---|
| 200 Gbit/s | [1,474,560, 9,000,000] | [1,477,000, 3,121,000] | pass |
| 400 Gbit/s | [737,280, 4,500,000] | [740,000, 1,562,000] | pass |

The independent rate angle also behaved physically: every serialization-only
completion was approximately halved at 400 Gbit/s, and both packet deltas
obeyed the registered two-to-one relation within 2,000 ps.

For Granite, 24 layers times 4,139 ns gives a 99.336 microsecond compute
floor. The measured 27.1 MB peak rank egress needs about 542 microseconds at
400 Gbit/s. A serial compute-plus-peak-egress floor is therefore about 641
microseconds. Serializing all 207.5 MB through one link plus compute gives a
coarse 4.25 ms ceiling. The completed Granite cells, 795 to 952 microseconds,
sit inside that independent range. The aggregate packet result of 951.890
microseconds is also close to the separately supplied 974.838 microsecond
capture context, so no cell implies an implausible order-of-magnitude speedup.

## Raw synthetic backend observations

The matrix used zero compute, zero propagation and rates 200 and 400 Gbit/s.
Packet cells used a 4,096-byte maximum wire packet and a 64-byte data header.
Every native manifest reported `physical_quiescence=verified`. Exact raw FCT
rows remain in each external completion CSV; the table reports their observed
range without replacing those rows with an aggregate.

| Renderer | Profile | Gbit/s | Flows | Min FCT ps | Max FCT ps | Completion ps |
|---|---|---:|---:|---:|---:|---:|
| aggregate | rnic-nn-fluid | 200 | 18 | 491,521 | 901,120 | 1,639,401 |
| aggregate | rnic-nn-fluid | 400 | 18 | 245,761 | 450,560 | 820,201 |
| aggregate | rnic-nn | 200 | 18 | 660,480 | 1,310,720 | 2,463,720 |
| aggregate | rnic-nn | 400 | 18 | 330,240 | 655,360 | 1,232,360 |
| per-expert-group | rnic-nn-fluid | 200 | 18 | 491,521 | 737,281 | 1,477,000 |
| per-expert-group | rnic-nn-fluid | 400 | 18 | 245,761 | 368,641 | 740,000 |
| per-expert-group | rnic-nn | 200 | 18 | 742,400 | 1,227,800 | 2,382,000 |
| per-expert-group | rnic-nn | 400 | 18 | 371,200 | 613,400 | 1,192,000 |
| per-token | rnic-nn-fluid | 200 | 48 | 491,521 | 737,281 | 1,477,000 |
| per-token | rnic-nn-fluid | 400 | 48 | 245,761 | 368,641 | 740,000 |
| per-token | rnic-nn | 200 | 48 | 248,320 | 1,559,040 | 3,121,000 |
| per-token | rnic-nn | 400 | 48 | 124,160 | 779,520 | 1,562,000 |

## Scored behavioral evidence

Evidence classes remain separate. A parameterized instance passes only when
its complete registered relation passes.

| Family class | Instances | Result | Raw finding |
|---|---:|---:|---|
| Packet signed band | 4 | 0/4 | Every direction was positive, but magnitudes exceeded the byte-only bands. |
| Packet inverse rate | 2 | 2/2 | Errors were 1,000 ps and 2,000 ps against the 2,000 ps limit. |
| Fluid grouping diagnostic | 4 | 0/4 | Both sequenced modes were faster than aggregate by 162,401 ps at 200G and 80,201 ps at 400G. |

The genuine-risk headline is **1/3 family classes and 2/10 instances**.

Packet signed deltas were:

| Comparison | 200 Gbit/s ps | Frozen band ps | 400 Gbit/s ps | Frozen band ps |
|---|---:|---:|---:|---:|
| per-token minus per-expert-group | 739,000 | [15,360, 61,440] | 370,000 | [7,680, 30,720] |
| per-token minus aggregate | 657,280 | [15,360, 61,440] | 329,640 | [7,680, 30,720] |

### Entailment check

The runner calculated all ten scored predicates from raw native completion
values and wrote them before invoking the explicit sequence, conservation and
physical-bound evaluators. Input hash checks and native quiescence validation
necessarily preceded meaningful execution, but neither pins a completion
delta. The renderer's fail-closed request conservation also does not entail a
packet or fluid completion because legal message order, grouping and backend
service can still produce any of the observed signs and magnitudes.

Sequence counts, authored destination lists, pair conservation, request
conservation, combine transpose, source hashes, default GOAL identity,
quiescence and physical bounds are fatal unscored evidence. They add nothing
to the genuine-risk numerator or denominator. No by-construction guard is
scored.

## Registered misses and post-specified diagnosis

The packet bands counted 768 extra wire bytes and assumed exact-tail wire-byte
serialization was the complete incremental service. The native manifest also
reported `calendar_reservation=full-envelope`. Raw 400G per-token FCT steps
include 81,920 ps intervals, exactly one 4,096-byte calendar envelope at 400
Gbit/s. Splitting a group creates extra scheduled packet envelopes even when
the final packet carries fewer wire bytes. The frozen upper bands are about
12 times smaller than the measured deltas. The nearly exact two-to-one rate
scaling shows that this is still serializer work, not an unrelated constant
offset. This diagnosis is post-specified and does not change the 0/4 verdict.

The fluid miss separates two questions. Per-token and per-expert-group cells
were exactly equal at each rate, so fluid correctly erased message
granularity for this fixture. Both preserved the same capture-derived issue
order, while aggregate used ascending pair order. That order alone improved
fluid completion by 80,201 ps at 400G and 162,401 ps at 200G. Fluid can answer
the grouping-only question here, but it cannot answer an order-sensitive
contention question by substituting the aggregate order. This diagnosis also
leaves the four frozen aggregate comparisons failed.

## Fatal exact and structural evidence

All fatal checks passed:

- the v2 projection retained routes `(3, 1)`, `(2, 1)`, `(3, 2)`, `(1, 3)`;
- per-source dispatch destinations matched all four frozen sequences;
- per-token emitted 24 dispatch and 24 combine messages;
- expert-group and aggregate each emitted 9 dispatch and 9 combine messages;
- every phase carried 49,152 bytes and the two phases carried 98,304 bytes;
- all nine positive dispatch pairs and every per-request projection matched
  the aggregate authority exactly;
- combine was the exact transpose with unchanged routing ordinals;
- the default generated Granite GOAL remained 334,432 bytes with SHA-256
  `08a0403af66ff8a9d6b18f93afd15ae0bc925cc85555acf8a0593438a3d7bc92`.

The unit and integration subset passed 55/55 before the study. The final
repository gates passed `ruff check .` and 1,040 pytest tests, with seven
environment-dependent tests skipped. These executable checks remain a
separate evidence class.

## Granite scale and practicality

The Granite input is `simllm-preplay-trace-v1`. Its tuple order is a
Transformers reconstruction, not a framework observation. It is used only for
scale and cost. The routed-experts and step hashes matched their authored
values exactly.

| Grouping | Messages | Bytes | Plan s | Render s | Compile s | Peak traced MiB | GOAL bytes | Practical |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| aggregate | 2,688 | 207,499,264 | 0.461 | 1.020 | 0.196 | 10.4 | 334,432 | yes |
| per-expert-group | 8,064 | 207,499,264 | 2.579 | 6.134 | 0.203 | 22.7 | 1,446,770 | yes |
| per-token | 101,318 | 207,499,264 | 1.856 | greater than 60 | not reached | 45.8 during attempt | not reached | no |

The 54-token, 24-layer step is the measured practicality boundary: per-token
planning is affordable, but current GOAL message recording is not. The
interrupted unbounded attempt and the controlled attempt both stopped inside
`GoalTrace.record_message`, whose per-message label validation scans the
growing rank program. The controlled attempt is a lower bound, not a claimed
completion time.

Completed Granite native costs were:

| Grouping | Profile | Backend wall s | Flows | Completion ps |
|---|---|---:|---:|---:|
| aggregate | rnic-nn | 2.541 | 2,688 | 951,889,600 |
| aggregate | rnic-nn-fluid | 2.125 | 2,688 | 933,833,788 |
| per-expert-group | rnic-nn | 12.663 | 8,064 | 857,093,000 |
| per-expert-group | rnic-nn-fluid | 13.311 | 8,064 | 794,966,000 |

Per-token packet and fluid runs were not attempted because no GOAL or binary
existed after the render boundary. This omission is explicit TRAF-22 work.

## TRAF-21 closure map

The registered clauses are quoted and mapped without weakening them:

1. **"`captured-message-sequence` preserves the v2 framework-returned
   request, token and top-k order under both declared grouping rules."** The
   strict v2 fixture, both plans, all four source sequences and retained
   routing ordinals passed exactly.
2. **"Every aggregate ordered-pair and per-request byte total matches the
   accepted aggregate authority exactly."** Both grouping projections matched
   every request, phase and pair, totaling 98,304 synthetic bytes and
   207,499,264 Granite bytes.
3. **"The default aggregate APIs and accepted artifacts remain
   byte-identical."** No selector was added to either default API; all existing
   goldens passed and the accepted Granite GOAL hash matched.
4. **"The registered packet and fluid matrix reaches a native backend and
   reports the frozen signed relations, raw FCT and completion."** All 12
   cells reached the native backend and quiesced. Raw FCT CSVs and completion
   values are retained. The inverse-rate relation passed; packet bands and
   aggregate-relative fluid bands are reported as genuine-risk failures with
   their post-specified diagnoses.
5. **"The Granite scale record reports cost and practicality without
   describing v1 reconstructed order as observed."** All plan counts and the
   completed render, compile and backend costs are reported. The per-token
   render lower bound and omitted backend cells are assigned to TRAF-22. Every
   Granite row says `reconstructed-v1`.
6. **"No repository-wide fidelity selector is added ahead of CORE-36."** The
   new renderer is an explicit traffic API. CORE-36 remains untouched.

TRAF-21 closes for the sequence generator, exact projections, compatibility
identity and executed decision fixture. TRAF-22 retains the uncompleted
Granite per-token scale path. PLAY-14 retains the unobserved kernel-to-wire
ordering question. CORE-37 and BACK-40 were not used because the evidence did
not identify a separate core-authority or backend-correctness defect.

## Contradiction sweep and integrator-owned omissions

The required post-closure sweep found no statement that directly contradicts
the implemented bytes or ordering authority, but it found three
integrator-owned omissions that this branch does not hand edit:

- `README.md` still describes captured routing as driving the MoE all-to-all
  without distinguishing the aggregate default from ordered per-token or
  expert-group messages.
- `docs/README_PRO.md` does not yet list `captured-message-sequence` in its
  fidelity matrix or add this study to its study index. Only its generated
  task-progress block and mechanically required module open counts changed in
  this closure.
- `docs/architecture.md` says dispatch and combine can use captured per-token
  expert routing, but does not state that the default coalesces one send per
  ordered pair or that the new explicit level preserves source-local issue
  order and declared granularity.

These are integration edits, not evidence gaps in TRAF-21. The module docs and
this result remain the authoritative detailed record until the integrator
updates those maps.
