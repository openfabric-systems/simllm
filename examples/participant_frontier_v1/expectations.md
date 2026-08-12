# Participant frontier v1 expectations

This expectations-only record freezes CORE-35 before implementation and before
the new study produces a result. The study requires the coarse runtime report
to conserve participant-local dependency frontiers without changing graph
scheduling, completion events or request routing lifetimes.

## Decision and representation contract

The decision-relevant operation is
`step-0:layer-1:rank-1:compute` in the three-request Granite replay. Its rank-1
frontier comes from rank 1 of `step-0:layer-0:ep-combine`, not from that
collective's whole-operation maximum. The participant-local graph must be
admitted without converting its dependencies to barriers.

The report will replace scalar predecessor accounting as its conservation
authority with participant-keyed critical segments, or an exactly equivalent
representation. A segment is identified by operation and canonical participant
rank. It records its completion, exact causal boundary, predecessor segment,
selected resource path, critical-path breakdown and latency attribution. The
following are fatal invariants:

- every operation has exactly one segment for every rank returned by
  `operation_participant_ranks`;
- each segment completion equals that rank's entry in
  `participant_completed_at_ps`;
- a participant-local edge links the same rank, while a whole-operation edge
  links a predecessor segment at the predecessor's logical maximum;
- a nonroot segment starts exactly at its referenced predecessor segment's
  completion, and a root starts at graph release;
- breakdown and attribution each sum to `completed_at_ps - started_at_ps`;
- every selected endpoint chain is acyclic and sums exactly from graph release
  to endpoint completion.

Legacy operation-level report fields may remain as explicit compatibility
projections, but they are not allowed to replace, relax or contradict the
participant-keyed conservation authority. If this representation cannot
distinguish rank 1 from the collective maximum, the design decision changes
and CORE-35 remains open.

## Pre-freeze source audit

The evidence was authored against SimLLM revision
`76223875557a552deb5aa2c2c529a07f000135ba`. The relevant source locations at
that revision are:

- `simllm/backends/step_lowerer.py:293-313` carries each rank's prior tail into
  compute through `participant_local_depends_on`;
- `simllm/backends/step_lowerer.py:340-366` carries per-rank compute and MoE
  collective tails through participant-local dependencies;
- `simllm/core/execution_io.py:62-78` gives a collective all declared ranks as
  canonical participants;
- `simllm/core/runtime.py:1260-1298` realizes a participant-local edge at that
  rank's predecessor completion;
- `simllm/core/runtime.py:291-313` and `simllm/core/runtime.py:2307-2431`
  reduce those frontiers to one scalar causal and additive predecessor;
- `examples/routing_lifetime_v1/run_study.py:491-524` documents the accepted
  compatibility projection that tightened every local frontier to a
  whole-operation barrier.

The source inputs are addressed through `SIMLLM_MOE_E2E_ROOT`, never through a
tracked absolute path. The study freezes these identities:

| Input | SHA-256 |
|---|---|
| `capture/granite-greedy.jsonl` | `5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6` |
| `replay-400g/steps.jsonl` | `824cd9557293328bb42b593ac893b6a067302e545b087c9219195ccb8031d755` |
| `replay-400g/routed-experts.json` | `24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f` |

The accepted routing-lifetime result was observed at SimLLM revision
`c2217356e5c456256557072ba9723945cb69f8a2` with result SHA-256
`2488cbe14004fe02a9c1ccda2ba539a67f05e68d26cc96c244318705a501fdb1`.
It closed one request after 25 steps and three requests after 32 steps plus an
explicit finish-only drain. That prior result is provenance, not a live file
dependency of this study.

Before this freeze, read-only diagnostics imported the existing SimLLM code,
opened the accepted read-only routing arenas and wrote no file. They observed
the scalar report's current compatibility behavior and the exact completion
values below. These observations motivate the frozen oracles and are excluded
from scored evidence.

## Frozen sweep and graph shapes

The two-parameter matrix is:

- request count `R in {1, 3}`;
- graph shape `S in {participant-local, whole-operation-barrier}`.

Both shapes use the same `SerialStepLowerer` output. The barrier arm applies
the previously accepted routing-lifetime projection by moving each explicit
participant-local predecessor into `depends_on`. It may not change operation
order, operation work, request correlation, layer identity or completion IDs.
The participant-local arm executes the lowerer's graph unchanged.

The one-request replay consumes steps 0 through 24. The three-request replay
consumes all 32 captured steps and one empty step 32 that carries only the
delayed `r2` finish. Both use 24 layers, 32 experts, top-k eight, EP ranks 0
through 7, 4,139,000 ps compute per layer and the default coarse device
profile.

## Exact completion preservation

Canonical result bytes are `execution_result_to_json` payloads for every
execution in order, encoded as one compact, key-sorted JSON array plus LF.
Completion bytes are compact, key-sorted JSON rows
`[step_index, operation_id, subject_object_id, timestamp_ps]` for every
`COMPLETED` event in stable identity order, plus LF. These are fatal exact
oracles and do not increase a behavioral denominator.

| Requests | Shape | Executions | All events | Completions | Result bytes | Result SHA-256 | Completion bytes | Completion SHA-256 |
|---:|---|---:|---:|---:|---:|---|---:|---|
| 1 | participant-local | 25 | 110,416 | 5,760 | 30,399,320 | `00cff9f56b550a166548e9c44e98d4dffe26c8102eb17b7a1fcdeda6e863fb94` | 288,300 | `73b7415729185e9b4481561da8e6caff23487b68bb6a962f401bfe7052beb8b4` |
| 1 | barrier | 25 | 110,416 | 5,760 | 30,399,320 | `38cb6503f5475f2acd8071771c09119ddfa7ae4dc7af875169612b9375347420` | 288,300 | `6f70c590af674dea6f9f24860e16fda3cf1f9a20eda4869ec0a34b027cb637af` |
| 3 | participant-local | 33 | 160,416 | 7,680 | 44,179,494 | `f58841e7747ae08fb41355e48a1aba30fdf9b12bb3b2e68642241550cf36115f` | 386,327 | `1fcaf34da306efac867c27d45d0e2d0ae8975c7c692a34cafbf650b68adec6c7` |
| 3 | barrier | 33 | 160,416 | 7,680 | 44,179,502 | `66668afa531ab34054d2e4a3b3dc476d539600cc945ceb6188b87cbeb233f1a5` | 386,327 | `dd2356365d657d9c0c1e4056b1677bf184d14060bac0827ac8c34cbbbb18125e` |

Within each request count, both shapes must have the same event-identity
multiset on every step and the same execution boundary on every step. For one
request, 4,455 of 5,760 completion identities have equal timestamps and 1,305
legitimately differ. For three requests, 5,127 of 7,680 are equal and 2,553
legitimately differ. The differing timestamps are not drift: the local shape
allows a rank to proceed from its own frontier, while the barrier shape waits
for the predecessor's slowest rank. No completion identity may be lost,
duplicated or invented.

The exact decision-relevant step-0 rows are:

| Requests | Shape | Rank-1 predecessor boundary ps | Target completion ps | Step completion ps |
|---:|---|---:|---:|---:|
| 1 | participant-local | 6,341,742 | 10,480,742 | 154,568,365 |
| 1 | barrier | 6,651,217 | 10,790,217 | 154,568,365 |
| 3 | participant-local | 9,673,156 | 13,812,156 | 234,886,380 |
| 3 | barrier | 10,346,720 | 14,485,720 | 234,886,380 |

Thus the target's barrier-minus-local completion gap must be exactly 309,475
ps for one request and 673,564 ps for three requests. Tightening the local arm
to make either gap zero fails the decision-relevant relation. Changing either
step endpoint also fails, because the slowest participant already controls the
accepted endpoint in these cells.

## Live metric and routing-lifetime relations

PF-B1 is the raw live-chain admission relation. Both participant-local cells
must complete through `CompletionReducer` before any exact digest is checked.
The one-request cell must exit with `(closed, live, views) = (1, 0, 0)`, and
the three-request cell with `(3, 0, 0)`. The three-request cell must include
the named rank-1 operation and preserve all 7,680 logical completions. Either
cell can genuinely fail if segment construction, predecessor selection or
request attribution is wrong.

PF-B2 is the signed graph-shape relation. The raw target barrier-minus-local
gaps must equal the two positive values above, while every step-level
`StepResult` boundary remains equal across shapes. This is evaluated before
the exact completion or result digests.

PF-B3 is request-count scaling on the live step-0 result. In the
participant-local arm, three requests must increase step latency by exactly
80,318,015 ps, from 154,568,365 ps to 234,886,380 ps. The ratio is
1.519627771. The later exact result hashes entail these values, so this raw
relation is evaluated and recorded first.

The barrier arm is the explicit accepted compatibility baseline. Its exact
results and unchanged routing-lifetime exits are the bypass-preserves-baseline
check required for this live metric-chain change.

## Negative control

PF-B4 mutates one otherwise valid three-request participant report after raw
PF-B1 through PF-B3 observations are captured. It changes only the target
segment's predecessor participant from rank 1 to rank 0 while retaining the
rank-1 start boundary, completion, breakdown and attribution. The referenced
rank-0 predecessor segment has a different completion timestamp.
`CompletionReducer` must reject this report with a participant-segment
predecessor timestamp diagnostic before mutating the virtual clock, request
metrics or lifetime registry. Acceptance of this construction means the fix
weakened conservation and CORE-35 remains open.

The mutation outcome is scored because a plausible implementation can expose
segments but fail to cross-check their references. The unchanged input and
the exact diagnostic wording are fatal-unscored guards.

## Physical sanity

The study's EP ranks 0 through 7 are on one eight-GPU node, so their transfers
use the profile's 900 Gbit/s NVLink-class resources, not the 400 Gbit/s RNIC.
Step 0 has these frozen raw graph totals:

| Requests | Directed pairs | Total bytes | Peak rank egress bytes | Peak-egress serialization floor ps | Compute floor ps | Serialized-work ceiling ps | Observed step ps |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 336 | 10,403,840 | 5,201,920 | 46,239,289 | 99,336,000 | 191,814,578 | 154,568,365 |
| 3 | 336 | 25,563,136 | 12,781,568 | 113,613,049 | 99,336,000 | 326,563,876 | 234,886,380 |

The floor is the larger of the independent compute and peak-egress floors.
The ceiling serializes all directed bytes through one 900 Gbit/s resource and
then adds all 24 compute layers. Both observed values lie strictly between
their floor and ceiling. Total bytes and peak egress scale by 2.457086614,
while step time scales by 1.519627771 because the unchanged compute term and
rank-parallel transfers prevent byte-proportional scaling. A step below either
floor or above the conservative ceiling voids the run before exact checks.

## Evidence classes, entailment and fatal semantics

The scored headline contains four genuine-risk families and seven instances:

1. PF-B1 participant-local admission and clean lifecycle, two request-count
   instances;
2. PF-B2 positive target gap with unchanged step boundary, two request-count
   instances;
3. PF-B3 live step-latency request scaling, two raw endpoint instances treated
   as one relation family;
4. PF-B4 malformed predecessor-rank rejection, one mutation instance.

PF-B1 through PF-B3 are evaluated from raw runtime and reducer observations
before exact timestamp, byte or hash oracles. PF-B4 is evaluated before the
unchanged-state audit. Therefore no earlier fatal oracle entails a scored
pass. The later exact digests deliberately pin the same completion surface,
but they are fatal-unscored preservation evidence and cannot increase the
headline.

Input identity, graph census, event identity, exact timestamps, segment
inventory, segment conservation, endpoint-chain conservation, physical
bounds, no-mutation checks, check-only behavior and repository tests are
separate fatal-unscored evidence classes. A single fatal failure makes the run
void. Fatal guards are never reported as a fraction or folded into a scored
denominator.

## Registered acceptance clauses

1. The unchanged Granite participant-local graph admits
   `step-0:layer-1:rank-1:compute` on rank 1 and completes both request-count
   replays through the live reducer and routing-lifetime registry.
2. Every participant-local and barrier completion matches its frozen digest,
   both shapes retain identical completion identities and step boundaries,
   and all clean lifetime exit states match.
3. The report exposes participant-keyed conserved segments with exact
   predecessor identity, rank and boundary; every segment and selected
   endpoint chain satisfies the registered conservation identities.
4. The accepted barrier projection matches every frozen baseline surface,
   while the participant-local projection differs only at the registered
   intermediate timestamps caused by its weaker legal ordering.
5. The malformed predecessor-rank control is rejected atomically with the
   participant and timestamp disagreement identified.
6. Physical floors, ceilings and request-count scaling hold, all fatal guards
   pass, and evidence classes remain separate.

## Registered command and check-only dry run

The production command is:

```bash
.venv/bin/python examples/participant_frontier_v1/run_study.py \
  --source-root "$SIMLLM_MOE_E2E_ROOT" \
  --out "$SIMLLM_PARTICIPANT_FRONTIER_RUN_ROOT"
```

Before this expectations-only commit, the same command is run with
`--check-only`. That path parses both paths and validates only frozen registry
shape, arithmetic, digests and evidence counts. It imports no SimLLM
implementation, reads neither path, invokes no native binary and creates no
artifact.
