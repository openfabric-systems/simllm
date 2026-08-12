# Collective plan v1 results

## Outcome

The run is **not void**. Every registered fatal guard passed, and the two
registered genuine-risk families passed all six instances:

| Family | Instances | Passed |
|---|---:|---:|
| Byte-conserving perturbation rejected before any work request | 2 | 2 |
| Explicit plan reaches TTFT and TPOT through the reported chain | 4 | 4 |
| **Total** | **6** | **6** |

Exact plan-versus-pattern rows, compatibility identity rows, integrity rows,
physical-bound rows and zero-work rows are fatal-unscored. They are reported
below with their own counts and are never added into the behavioral total.

TRAF-14 closes. Two residuals move to new IDs, TRAF-28 and CORE-48, for the
work this run demonstrated is still missing. TRAF-29 was not needed and stays
unused.

## Chronology and provenance

The expectations-only freeze precedes both the implementation and the first
result-producing run, so the assertions are pre-registered rather than
post-specified regression checks.

| Event | Commit |
|---|---|
| Base revision the evidence was authored against | `76223875557a552deb5aa2c2c529a07f000135ba` |
| Expectations-only freeze | `bf3cf400225872f89a153aa4e2a3bd8e0e05f838` |
| Final pre-run expectation correction | `f9f886dd9360d76e74f7b4498094221b5e4aa6f5` |
| Immutable plan implementation | `49af52d4e4f294e13ad06e6945bb435b81e6ecc9` |
| Acceptance tests and renderer correction | `9229311f8f83d1405a28f7532d025c8131f06e8c` |
| Study implementation | `8880c16a3ceacfc726002a144c98f52ce00ed253` |

The registered `--check-only` command passed before the freeze and again
before the production path landed. It parses the full CLI, validates only
frozen literal shape, arithmetic and physical bounds, imports no SimLLM
implementation and creates no output path. That was reconfirmed after the
production path was written.

| Provenance field | Observed value |
|---|---|
| Revision the run observed | `9229311f8f83d1405a28f7532d025c8131f06e8c` |
| htsim gitlink the run observed | `fc4400e4ca619223481536632074045cb6af2756` |

The reported run executed from the working tree at HEAD
`9229311f8f83d1405a28f7532d025c8131f06e8c`, before its own study file was
committed, which is why the observed revision precedes the study commit above.
The authored and observed revisions are recorded separately. No equality
between them, or between either and a live submodule pin, is asserted or
tested. No third-party simulator, initialized submodule or native binary is
needed for this study.

### One void run, reported

The first production run was **void**: two fatal wire guards failed. The cause
was in the harness, not the implementation. The study built its absent-plan
wire graph with execution identity `("exec", 3, 17)` while the frozen
559-byte oracle names the graph `("core6-uniform", 7, 11)`. The nine-byte
execution-id difference produced 550 bytes and a different digest. The harness
was corrected to reproduce the exact graph the oracle names; the frozen
literals were not touched, and no expectation was edited to match an
observation. The corrected rerun produced the result reported here.

## The decision-relevant relation

Two authorities for one quantity is the failure mode this task exists to
remove. The acceptance therefore has to distinguish an immutable plan from a
runtime-side reconstruction, and a byte-conserving change is the sharpest
instrument: a reconstruction re-derives the quantity from semantic work and
cannot see the change at all.

### Instance 1, a changed plan tag

One frozen round tag was moved from 1,000 to 1,500 while the plan kept its
original integrity identity. Total directed bytes stayed at 24. The change was
rejected by graph validation and again by runtime preflight:

```text
graph.collective_plans[0].integrity_sha256: collective plan integrity mismatch
```

Work requests submitted: 0.

### Instance 2, a changed semantic rank order

Semantic ring rank order was changed from `(0, 8, 16, 24)` to
`(0, 16, 8, 24)` while the original plan was retained. The participant set and
total directed bytes stayed identical. Rejected by validation and by runtime
preflight:

```text
graph.collective_plans[0]: rank order disagrees with semantic work
```

Work requests submitted: 0.

### The negative control that makes these genuine risk

The same rank-order change was applied to the absent-plan graph, where the
runtime reconstructs the expansion from semantic work. It was absorbed
silently:

| Arm | Completion | Directed bytes |
|---|---:|---:|
| Baseline `(0, 8, 16, 24)` | 120 ps | 24 |
| Reordered `(0, 16, 8, 24)` | 120 ps | 24 |

Identical completion, identical bytes, no error. That is the surrogate
behavior the plan must reject, and it is why these two instances are
behavioral evidence rather than restatements of an earlier oracle. The control
itself is unscored: it characterizes the comparator, it does not test the
plan.

A third check confirms that resealing does not create an escape hatch. A plan
whose rank order was changed and whose integrity digest was then recomputed
over the changed content is still rejected, because plan rank order is also
compared against the semantic work it joins to.

## Live metric family

The 3-byte four-rank ring is the drift sentinel. The accepted GOAL pattern
expands it to six rounds of one-byte messages with `max(1, payload // W)`. The
absent-plan compatibility runtime rejects the operation before its own
`payload // W` expansion can invent zero-byte work:

```text
ring all-reduce payload must provide at least one byte per rank;
CORE-16 owns remainder chunking
```

The explicit-plan arm carries the GOAL-valid declared extents through
`CoarseDeviceRuntime`, `CompletionEvent`, `RuntimeReport`,
`CompletionReducer`, `StepResult`, TTFT and TPOT over one prefill and two
decode steps:

| Rate | Frozen expectation | Observed TTFT | Observed TPOT | Step latencies |
|---:|---:|---:|---:|---|
| 400 Gbit/s | 120 ps | 120 ps | 120 ps | 120, 120, 120 |
| 200 Gbit/s | 240 ps | 240 ps | 240 ps | 240, 240, 240 |

All four instances match exactly. Twenty-four work requests of one byte each
were submitted per step, so no byte was lost or duplicated.

Harsh reading of this family: at a fixed rate the prefill and decode steps have
the same step latency, so TTFT and TPOT report the same underlying number
through two different reducer paths. The four instances are therefore two
independent rate points crossed with two metric paths, not four independent
measurements of the network model. They are counted as registered, and this
caveat is recorded rather than quietly folded in.

The registered inverse-rate relation (200 Gbit/s is exactly twice 400 Gbit/s)
holds and is reported fatal-unscored, because the four exact metric instances
already pin it.

## Physical sanity, three independent angles

Bounds were stated from first principles before the measured values were read.

### Angle one, serialization physics

Bytes over link rate is a floor no flow can beat. Every measured value sits at
or inside its interval:

| Case | Floor | Observed | Ceiling |
|---|---:|---:|---:|
| Sentinel ring, 400 Gbit/s | 120 ps | 120 ps | 480 ps |
| Sentinel ring, 200 Gbit/s | 240 ps | 240 ps | 960 ps |
| 4,096 B ring W=2, 400 Gbit/s | 81,920 ps | 81,920 ps | 163,840 ps |
| 4,096 B ring W=4, 400 Gbit/s | 122,880 ps | 122,880 ps | 491,520 ps |
| 4,096 B ring W=2, 200 Gbit/s | 163,840 ps | 163,840 ps | 327,680 ps |
| 4,096 B ring W=4, 200 Gbit/s | 245,760 ps | 245,760 ps | 983,040 ps |
| Routed dispatch, 400 Gbit/s | 160 ps | 160 ps | 160 ps |
| Routed dispatch, 200 Gbit/s | 320 ps | 320 ps | 320 ps |

Every ring sits exactly on its floor because the study sets every non-network
service constant to zero, places each source on a distinct coarse RNIC and
uses no propagation term. A value below the floor would prove byte loss; a
zero would prove reconstruction. Neither occurred.

The many-source combine is structural evidence only. The coarse model has no
destination-ingress serializer, so its 100 ps at 400 Gbit/s and 200 ps at
200 Gbit/s are the maximum single-source egress, not a physical oracle for a
converging pattern. It is reported and not accepted as precision evidence.
The empty semantic collective has no physical service at all; its zero is by
construction and unscored.

### Angle two, the relation that should scale with it

Halving the link rate doubles every serialization-bound value exactly:
120 to 240, 81,920 to 163,840, 122,880 to 245,760, 160 to 320, 100 to 200.
Exact factors of two are the correct answer here precisely because every
fixed additive term is configured to zero. A factor short of two would have
indicated a hidden constant; a factor above two would have indicated
contention the model does not claim.

### Angle three, plausibility against a real system

Deliberately out of scope, and stated so before the run. A 120 ps step latency
is not a deployment number and is not offered as one. This study qualifies
authority and conservation on a tiny sentinel where every non-network cost is
zero by configuration. Collective-service calibration against real hardware
remains TRAF-11. Reporting these picoseconds as deployment TTFT would be the
error the physical-sanity rule exists to prevent.

## Exact and fatal evidence

All of the following passed. They are fatal-unscored: a violation would have
made the whole run void, and none of them increases the behavioral
denominator.

| Guard family | Rows | Result |
|---|---:|---|
| Plan versus accepted pattern expansion, both frontier modes | 18 | identical |
| Explicit-versus-absent runtime identity | 16 | identical |
| Absent-plan v1 wire oracle | 1 | 559 bytes, digest matches, field omitted |
| Plan integrity, coverage, round trip, tag order, idempotence | 5 | pass |
| Physical bounds | 10 | inside |
| Zero-work and idle-rank conservation | 1 | pass |

### Plan versus the accepted pattern

The plan is compared with structured output from the shipped `ring_allreduce`
and `pairwise_all_to_allv` functions, not a second text parser. For six ring
cells (worlds 2 and 4 crossed with payloads 3, 4 and 4,096) and three routed
sparse cases, in both GOAL frontier modes, the rendered messages,
dependencies, per-rank frontiers and full GOAL text are identical:

| W | Payload | Rounds | Chunk | Messages | Directed bytes | Internal deps |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3 | 2 | 1 | 4 | 4 | 4 |
| 2 | 4 | 2 | 2 | 4 | 8 | 4 |
| 2 | 4,096 | 2 | 2,048 | 4 | 8,192 | 4 |
| 4 | 3 | 6 | 1 | 24 | 24 | 40 |
| 4 | 4 | 6 | 1 | 24 | 24 | 40 |
| 4 | 4,096 | 6 | 1,024 | 24 | 24,576 | 40 |

Every row equals its frozen registry entry. Tags are the consecutive block
starting at 1,000 in every case.

### Routed sparse pairs

| Case | Sources | Destinations | Messages | Directed bytes | Exact frontier ranks |
|---|---|---|---:|---:|---|
| dispatch | 0 | 8, 16 | 2 | 8 | 0, 8, 16, 24 |
| combine | 8, 16 | 0 | 2 | 8 | 0, 8, 16, 24 |
| all local | none | none | 0 | 0 | 0, 8, 16, 24 |

A routed dispatch has one source and a combine has many sources converging on
one destination, as TRAF-25 established. Rank 24 carries no message in any of
the three cases and still receives its zero-time frontier, so no traffic is
invented for an idle rank. The empty semantic collective keeps exactly one
semantic round with no positive extent, emits no GOAL message, submits no work
request and completes at 0 ps, while the absent-plan runtime rejects it
outright with `pairwise all-to-allv requires a nonzero payload`.

### Compatibility identity

For the divisible ring cells at both link rates, the explicit-plan and
absent-plan arms produced identical `ExecutionResult`, identical
`RuntimeReport`, identical streamed event sequences and identical work-request
rows. Each cell was run twice, once with zero collective channel service and
once with 7,000 ps, because a nonzero channel service is what exposes a
per-round channel resource split. That second configuration caught a real
defect during implementation and now guards it.

The absent-plan graph still serializes to 559 bytes with SHA-256
`f4a5a70f5bd4a0c2fed874baa88f3035266a54f386a59927e115872c2bcff0a3`, omits the
optional plan field entirely, and round-trips exactly.

### Tag block order

With one four-rank ring followed by dispatch, combine and the empty semantic
collective, the plan carries ring tags 1,000 through 1,005 and then 1,006,
1,007 and 1,008. `collective_goal_tags` returns the same mapping for the
planned and unplanned graphs, because the plan builder calls that accepted
allocator instead of reimplementing it, and the allocator reads the tags back
out of the plan once it is attached.

## Entailment analysis

The runner evaluates the two perturbation outcomes and the four live metric
values from raw validation, runtime and reducer observations before any fixed
plan, GOAL, wire, work-request or timing oracle runs. The evaluation order is
recorded in the emitted summary.

No earlier fatal guard pins a scored quantity:

- the integrity guard asserts that emitted plans are correctly sealed. It does
  not assert that an unsealed or semantically disagreeing plan is rejected
  before scheduling, which is what instance 1 and instance 2 test;
- the physical guard on the sentinel asserts `120 <= observed <= 480` at
  400 Gbit/s. That interval does not pin the exact 120 ps the metric instances
  require;
- the compatibility guards cover divisible cells only. The sentinel is
  indivisible and has no absent-plan arm at all, so no identity row can
  entail its value.

The later exact rows reuse some of the same values for regression and do not
count again. The registered headline is therefore two families and six
genuine-risk instances, not the sum of all exact and fatal rows.

## Registered acceptance clauses

Each clause is quoted and mapped separately.

1. *"move ring-round and pairwise-extent expansion from the coarse runtime's
   current semantic-work surrogate into one immutable traffic-owned collective
   plan carried through `ExecutionGraph`"*: **demonstrated**.
   `simllm/traffic/collective_plan.py` builds one `CollectivePlan` per
   collective operation; `ExecutionGraph.collective_plans` carries it as an
   optional all-or-nothing field; `CoarseDeviceRuntime._schedule_collective_plan`
   consumes it. Mixed planned and unplanned authority is not representable: a
   graph carrying plans for only some of its collectives is rejected.
2. *"The runtime may schedule those extents but may not choose or reconstruct
   their algorithm, chunk sizes, rank order or tags"*: **demonstrated for the
   explicit-plan path**. With a plan present the runtime iterates declared
   extents and actions only, `collective_goal_tags` reads plan tags instead of
   allocating, and the emitted work-request rows equal the declared extents
   exactly in every sweep cell. See the residual below for the absent-plan
   path, which by clause 4 must keep reconstructing.
3. *"Compare the plan against the existing GOAL pattern expansion over payload,
   world-size and routed sparse-pair sweeps with exact byte, round, dependency
   and tag conservation"*: **demonstrated**. Eighteen structured comparisons
   against the shipped pattern functions, all identical, plus the exact
   registry table above.
4. *"The absent explicit plan must preserve the accepted v1 wire bytes and
   serial timing exactly"*: **demonstrated**. The 559-byte digest, the omitted
   field, the exact round trip and sixteen runtime identity rows across two
   rates and two channel-service settings. A dedicated test pins it.
5. *"routed dispatch has one source, combine has many sources converging on one
   destination, and idle ranks remain idle"*: **demonstrated**, see the routed
   sparse table.
6. *"a zero-byte semantic collective with every destination local preserves all
   rank frontiers without inventing traffic"*: **demonstrated**, see the same
   table and the zero-work guard.

## Residuals moved to new IDs

Clause 2 holds for the explicit-plan path, and clause 4 requires the
absent-plan reconstruction to stay. The surrogate is therefore demoted to the
explicit-off path rather than deleted, which is the sanctioned compatibility
arrangement, but it means the two expansions still coexist in the tree. Two
things this run did not demonstrate move to new IDs:

- **TRAF-28** (Precision; P1; M): make the traffic-owned plan the default on
  the production lowering path so the runtime-side reconstruction can be
  retired. Today no shipped lowerer attaches a plan; the plan is reachable only
  when a caller opts in through `plan_execution_graph_collectives`. Until the
  default flips, the two expansions remain able to drift for graphs that carry
  no plan, which is the exact failure mode TRAF-14 was opened against.
- **CORE-48** (Precision; P1; M): give the cross-node coarse RNIC path a
  destination-ingress serializer so a many-source combine has a physical
  completion oracle. This run could only report the converging combine as
  structural evidence, because the current model serializes egress per source
  and nothing at the receiver, so an all-remote combine completes at the
  maximum single-source egress rather than at a contended arrival.

  The boundary against the existing CORE-41 was checked before opening a new
  ID. CORE-41 owns the analytic intra-node routed service and is explicitly
  required to "preserve symmetric and all-remote timestamps exactly", so the
  all-remote path this run exercised is outside it. Filing the cross-node gap
  under CORE-41 would have created the same duplicated-ownership condition
  TRAF-14 exists to remove, this time in the task registry. Both entries now
  name the other's scope.

TRAF-29 was pre-allocated and was not needed. It is not reused.

## Contradiction sweep

`README.md`, `docs/README_PRO.md` and `docs/architecture.md` were checked for
statements this closure contradicts. No hit requires an edit:

- `docs/architecture.md` already states that `CollectiveWork` is semantic and
  that "the traffic/NCCL planner selects algorithms, chunks and channels before
  WQEs reach the network backend". This closure implements that statement
  rather than contradicting it.
- `README.md` mentions ring all-reduce only in study summaries and in the
  compute module's per-GPU egress kernel, neither of which describes graph-level
  expansion ownership.
- `docs/README_PRO.md` carries per-module open-task counts, which the progress
  script regenerates from the module docs as part of this change.

`docs/modules/traffic.md` is the owning doc and is updated here.

## Reproducing

```bash
.venv/bin/python examples/collective_plan_v1/run_study.py \
  --out "${SIMLLM_COLLECTIVE_PLAN_RUN_ROOT:?configure SIMLLM_COLLECTIVE_PLAN_RUN_ROOT}"
```

The run needs no native binary, no initialized submodule and no captured
trace. It writes one `summary.json` of about 20 KB and nothing else.
