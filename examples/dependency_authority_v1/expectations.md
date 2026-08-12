# Dependency authority v1 expectations

This is the expectations-only record for TRAF-12. It freezes the authority
decision, graph and GOAL correspondence checks, raw timing relations and
accepted-artifact treatment before the implementation or any result-producing
run exists.

## Architectural decision and claim boundary

`ExecutionGraph` is the single semantic authority for operation identity,
logical queues and ordering. A device runtime or GOAL execution may realize
those constraints, but neither may reconstruct or weaken them. This follows
the existing authority statement in `docs/modules/core.md`: the graph owns
semantic work, logical release constraints and dependency identity, while a
runtime owns their realized timing.

The htsim GOAL compiler observed during the source audit resolves labels
inside one rank block. It cannot name another rank's operation in a
`requires` relation. Therefore a whole-operation edge between distributed
operations must be rendered as an ordered boundary between GOAL artifacts
unless and until the backend grammar gains a cross-rank barrier. Such a
boundary is part of the checked graph projection, not a second scheduler.
Rank-local graph edges become GOAL `requires` relations. Deterministic
collective-internal relations are kept distinct from inter-operation graph
edges and carry their owning graph operation in projection provenance.

The active serial step sink will lower a `StepRecord` once to an
`ExecutionGraph`, derive any placement split from the graph-owned collective
operations, check the complete projection, and execute it. The old direct
`StepRecord` renderer remains a diagnostic compatibility artifact, not an
ordering authority for the active sink.

## Pre-freeze source and observation audit

The evidence was authored against SimLLM commit
`dcbef8682b1d74fb059a95d5b8b6f0c4ae07c9eb`. The audit found four execution
shapes:

1. `HtsimStepSink` directly renders one all-remote GOAL from `StepRecord` and
   uses its htsim CSV makespan.
2. The placement-enabled sink independently creates isolated phase GOALs,
   runs them serially, and adds `max(local, fabric)` phase durations.
3. `CoarseDeviceRuntime` realizes an `ExecutionGraph` and emits completion
   and runtime-report projections.
4. `render_serial_execution_graph_goal` renders a diagnostic GOAL from an
   `ExecutionGraph` but converts implicit FIFO predecessors to participant
   local dependencies.

The relevant source locations at that revision are:

- `simllm/core/execution.py:209-262` defines logical-queue FIFO,
  whole-operation `depends_on`, participant-local dependencies, and graph
  completion operations.
- `simllm/core/execution_io.py:268-365` validates explicit dependencies and
  adds consecutive same-queue FIFO edges to the graph DAG.
- `simllm/core/runtime.py:1186-1220` realizes explicit and FIFO
  whole-operation edges at predecessor logical completion, but realizes a
  participant-local edge at the shared rank's completion.
- `simllm/traffic/execution_goal.py:36-129` instead appends an implicit FIFO
  predecessor to the participant-local set and rejects explicit cross-rank
  whole-operation barriers.
- `simllm/traffic/patterns.py:67-157` independently chooses collective
  internal dependencies and a syntactic per-rank completion label.
- `simllm/backends/step_sink.py:346-545` does not build an execution graph;
  it selects either the direct monolithic renderer or the independently
  serialized locality phases.
- `simllm/backends/step_lowerer.py:226-252` places all serial EP collectives
  on one shared logical queue, so all 47 adjacent EP operations in the frozen
  step carry whole-operation FIFO constraints.

The external htsim compiler source observed for this audit was commit
`034e2419f061f872ece400b7280319290c7589d9`, independently of the SimLLM
gitlink. `htsim/sim/lgs/txt2bin.re:80-142` resolves both labels through one
map, while `htsim/sim/lgs/txt2bin.cpp:2953-2996` clears that map after each
rank is serialized. The observed SimLLM htsim gitlink was
`fc4400e4ca619223481536632074045cb6af2756`. These are separate provenance
observations and are not frozen equalities against a future live pin.

The task supplied a two-operation experiment in which a GOAL `requires`
increased completion by the predecessor's exact 22,000 ns transfer time.
That demonstrates that htsim enforces a representable dependency. The source
audit above bounds the current compiler's cross-rank representability.

Before this freeze, read-only diagnostics imported the existing SimLLM code
but wrote no file and invoked no native tool. They observed the edge census,
direct-versus-graph GOAL identity and coarse-runtime failure below. Those
observations motivate the frozen relations but are excluded from every scored
result. The historical TRAF-10 CSV observations are likewise prior evidence,
not new scored rows.

## Frozen step and two-parameter sweep

The study reuses the tracked captured Granite routing step and exact supply
construction from `examples/nvlink_locality_v1`. It fixes 24 layers, EP ranks
`(0, 1, 2, 3)`, 24,000 ps of represented compute, the `rnic-nn-fluid`
profile, 400 Gbit/s and three controlled replays per cell. It varies two
parameters:

- hidden-vector bytes `V in {1,024, 2,048}`;
- node span `AAAA`, `AABB`, and `ABCD`, with the same placement meanings as
  the accepted TRAF-10 study.

The projection set also contains a minimal two-collective shared-queue graph
and one participant-local asymmetric-completion graph. These fixtures isolate
edge scope without depending on the captured traffic table.

## Frozen authority and artifact inventory

The result must report, for every current and reconciled path, its semantic
input, ordering authority, rendered artifacts, runtime, completion boundary
and live metric consumer. It must account for at least these pre-freeze
disagreements rather than treating 46 overlapping transitions as exhaustive:

- graph whole-operation FIFO versus rank-local GOAL entry;
- monolithic simulator state versus isolated phase resets;
- selected syntactic GOAL frontier versus maximum participant completion;
- graph completion-operation subset versus full GOAL quiescence;
- graph runtime report substituting global completion for a selected
  participant-local completion;
- unequal per-layer provider estimates in the sink versus uniform graph
  lowering;
- supported-domain differences for forward, multiple and cross-rank
  dependencies.

## Exact projection contract

For each frozen graph, the implementation must enumerate one typed effective
edge set. An edge records source and target operation IDs, whole-operation or
participant-local scope, applicable rank when local, and explicit or FIFO
origin. Validation, coarse realization and the GOAL execution projection must
consume that same set.

The checked expanded projection must satisfy all of these fatal invariants:

- every inter-operation GOAL `requires` has exactly one effective graph-edge
  provenance;
- every effective graph edge is represented exactly once at its required
  scope, either by rank-local `requires` relations or an ordered artifact
  boundary;
- every collective-internal `requires` belongs to exactly one graph operation
  and is distinguished from graph-edge provenance;
- no projected operation, edge, rank, payload, tag or request partition is
  lost or duplicated;
- the graph completion boundary is represented or the graph is rejected
  before an artifact is written.

A negative control removes one nonredundant FIFO boundary from an otherwise
unchanged projection. The checker must reject it with a missing-edge error
before backend execution. A second control checks the valid unmodified
projection. The raw mutation outcome is evaluated before exact edge counts;
the counts remain fatal-unscored.

## Signed timing reconciliation relation

The current graph's shared EP queue is authoritative, so all 47 adjacent
collective transitions are whole-operation barriers. Reconciliation must
increase the all-remote live `StepResult` from the historical observations
into the previously registered global-phase bands:

| Vector bytes | Historical JCT ps | Reconciled JCT band ps | Signed change ps |
|---:|---:|---:|---:|
| 1,024 | 156,569,755 | [160,781,760, 160,781,808] | [+4,212,005, +4,212,053] |
| 2,048 | 217,222,486 | [225,539,520, 225,539,568] | [+8,317,034, +8,317,082] |

The signed change is the decision-relevant genuine-risk family. Both payload
instances must be positive and fall in their frozen bands. A zero change is a
failed finding and must be explained, not accepted.

The complete `StepResult` sweep must reproduce:

| Vector bytes | `AAAA` JCT ps | `AABB` JCT ps | `ABCD` JCT band ps |
|---:|---:|---:|---:|
| 1,024 | 7,121,000 | 139,195,840 | [160,781,760, 160,781,808] |
| 2,048 | 14,180,000 | 182,367,680 | [225,539,520, 225,539,568] |

The raw JCT observations are evaluated before projection inventories, exact
cell oracles, conservation, hashes or quiescence. Controlled TTFT is the
first replay latency and TPOT is the mean of the next two equal-step virtual
clock deltas. They demonstrate the supported metric chain but are entailed by
the fixed-step JCT and do not add scored evidence.

## Raw causal relation

For each reconciled all-remote payload, derive every tag's minimum start and
maximum completion directly from the new htsim CSV before checking graph-edge
or exact-timing oracles. For all 47 adjacent phase pairs:

```text
next_tag_min_start_ps - prior_tag_max_completion_ps >= 0
```

This is a second genuine-risk family over two payload instances. The raw
relation can fail despite correct bytes and a valid-looking graph projection.
The exact zero-early count that follows is fatal-unscored because it is
entailed by all 47 raw inequalities.

The minimal shared-queue graph must also produce no early successor entry in
both the coarse schedule and its GOAL execution projection. The asymmetric
participant-local fixture must preserve the selected rank timestamp through
`RuntimeReport` and `CompletionReducer` without substituting the predecessor's
global maximum.

## Accepted artifacts

The diagnostic direct renderer must preserve the accepted historical
all-remote GOAL bytes:

| Vector bytes | Bytes | SHA-256 |
|---:|---:|---|
| 1,024 | 72,819 | `0417832c8788a0477d48b414cf2d8456b87215abd1d0193ba46fb8db46185d8a` |
| 2,048 | 72,819 | `bcd72e63546d03efaddd48c16e160457d1e28f19795036d1f871788d78cf5a02` |

The active all-remote sink artifact changes from one rank-local GOAL to a
checked graph projection with ordered boundaries. Its artifact manifest,
GOAL bytes and timing are therefore explicitly re-accepted after observation,
with a labelled post-specified note. This is not presented as a preregistered
digest. Existing historical expectation and result files remain unchanged.

Other accepted serial GOAL and graph locks must remain unchanged wherever
their graph contains no unsupported whole-operation distributed edge. Any
unavoidable lock change is listed separately with old and new values and the
semantic cause.

## Evidence classes and entailment

The scored headline contains exactly three genuine-risk families:

1. signed all-remote JCT reconciliation, two payload instances;
2. raw adjacent-tag causal gaps, two payload instances;
3. negative-control discrimination, one mutated-edge instance.

Each is evaluated from its raw observation before a later exact oracle can
pin it. The signed JCT uses live `StepResult` values before bands and hashes.
The causal family uses raw CSV timestamps before zero-early and projection
checks. The negative control uses the checker's raw accept/reject result
before exact inventories.

Run configurations, exact-oracle rows, structured invariants, fatal guards,
focused Python tests, the full Python suite and native executables are kept as
separate evidence classes. Byte conservation, exact edge inventories,
identity paths, completion conservation, check-only behavior, authority
labels, configured barrier counts and physical quiescence are fatal-unscored.
They never increase the behavioral denominator. TTFT and TPOT are live metric
reachability only because the fixed replay makes them algebraically equal to
JCT.

## Registered command and pre-freeze dry run

The result-producing command is:

```text
.venv/bin/python examples/dependency_authority_v1/run_study.py --out "$SIMLLM_DEPENDENCY_AUTHORITY_RUN_ROOT"
```

Before the expectations commit, that exact CLI is run with `--check-only`.
Check-only parses the production arguments and validates only frozen literal
shapes and arithmetic. It imports no SimLLM implementation, inspects no input
or output path, invokes no native binary, and creates no artifact by design.
The result report records the final expectations-only commit and the observed
SimLLM and backend revisions separately.
