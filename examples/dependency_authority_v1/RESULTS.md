# Dependency authority v1 results

TRAF-12 closes for the demonstrated serial step-sink scope. `ExecutionGraph`
is now the single semantic authority for operation identity, logical queues,
dependency scope and completion identity. The locality plan, GOAL artifacts,
backend ordering and coarse runtime are checked projections of that graph.

Both genuine-risk families passed, with **2/2 families and 3/3 parameterized
instances**. Reconciliation increased the all-remote live JCT by exactly
4,212,053 ps at 1,024 vector bytes and 8,317,082 ps at 2,048 vector bytes.
Those positive changes land at the upper endpoints of the frozen bands and
replace the historical rank-local phase overlap with graph-authored ordering.

The expectations document originally listed raw adjacent-tag gaps as a third
genuine-risk family. The implemented backend projection executes ordered
causal-level artifacts in separate processes, and the study constructs their
absolute timestamps by adding prior artifact service. Nonnegative
cross-artifact gaps are therefore true by construction. The raw observations
were still checked before the later exact inventories, but this family is
correctly classified as fatal-unscored. It contributes neither a family nor
its two payload instances to the genuine-risk fraction.

## Chronology and provenance

The expectations-only commit is
`d39dfdc2951e147187446e27c46d9ed3f1a6816a`. Before that commit, the registered
command was exercised with `--check-only`; it validated frozen literals,
imported no SimLLM implementation, invoked no native binary and produced no
artifact. Implementation landed in `a1cb70f`. The first production invocation
then exposed a direct-script import error before creating the output directory
or invoking a native binary. Commit `aed6de2` fixed the registered entry point,
and the successful production record observed that revision.

| Provenance field | Revision or value |
|---|---|
| SimLLM evidence authored against | `dcbef8682b1d74fb059a95d5b8b6f0c4ae07c9eb` |
| Expectations-only commit | `d39dfdc2951e147187446e27c46d9ed3f1a6816a` |
| SimLLM revision observed by the run | `aed6de2e64743c46c377426139ae59aa5349600a` |
| htsim compiler evidence authored against | `034e2419f061f872ece400b7280319290c7589d9` |
| htsim gitlink observed by the run | `fc4400e4ca619223481536632074045cb6af2756` |
| `txt2bin` SHA-256 observed by the run | `f3745f34ad86febe9c9eebef10aee5fae00b8865cb29943344fb75b0f142495b` |
| `htsim_rnic` SHA-256 observed by the run | `cfb5014a663791f7619fe33309114a74e82878de860c14fc8a723713501f027d` |
| Captured trace SHA-256 | `36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341` |
| Production `summary.json` SHA-256 | `c50bdd716d9d99c2f5c658ad63a1a66360202d70b41d45e2a5b24212477ec560` |
| Runtime | Python 3.12.12 on Linux x86-64 |

The compiler revision used to author the source evidence and the gitlink
observed during production are deliberately separate provenance facts. This
report does not assert that either must equal a future live submodule pin.

Reproduce the production run from the repository root with:

```bash
.venv/bin/python examples/dependency_authority_v1/run_study.py \
  --out "${SIMLLM_DEPENDENCY_AUTHORITY_RUN_ROOT:?configure this variable}"
```

## Declared sweep

The study held one captured 24-layer Granite routing step, TP rank 0, EP ranks
0 through 3, four GOAL ranks, 24,000 ps of represented compute, 400 Gbit/s
fabric links, 450,000,000,000 bytes/s analytic NVLink service and
`rnic-nn-fluid` fixed. It swept vector bytes 1,024 and 2,048 across one-node
`AAAA`, two-node `AABB` and all-remote `ABCD` placements. Each of the six cells
ran three controlled replays. The first replay latency is TTFT and the mean of
the next two virtual clock deltas is TPOT.

## Authority decision

`ExecutionGraph` is the architectural choice, not a convenient interchange
format. It already owns stable operation IDs, logical queue FIFO, explicit
whole-operation and participant-local edge scopes, request partitions and the
declared completion boundary. `CoarseDeviceRuntime` already realizes those
semantics into timestamps and completion records. In contrast, GOAL labels
are rank-block local in the observed compiler, and a GOAL run naturally ends
at full simulator quiescence. Letting the emitter independently reconstruct
ordering would therefore discard graph scope and completion information.

The reconciled design enumerates one canonical effective edge set from the
graph. Validation, the coarse runtime, locality classification and GOAL
projection consume that same set. GOAL is a read-only rendering. Distributed
whole-operation edges become required ordered artifact boundaries when the
grammar cannot name a cross-rank predecessor. All other cross-artifact edges
remain in a scope-preserving serialized inventory, while collective-internal
relations retain separate operation-owned provenance.

### Path inventory

| Path | Semantic input and ordering authority | Rendered or runtime artifacts | Completion boundary and metric consumer |
|---|---|---|---|
| Historical direct all-remote step sink | `StepRecord`; `render_step_goal` and collective patterns independently chose rank-local order | One monolithic GOAL, one htsim process and one completion CSV | Full GOAL quiescence became `StepResult`, TTFT and TPOT |
| Historical placement-enabled step sink | `StepRecord`; the locality phase loop independently imposed phase order | Isolated analytic local phases plus per-phase fabric GOALs, each in a fresh htsim process; phase durations composed as `max(local, fabric)` and then summed | The accumulated phase sum became `StepResult`, TTFT and TPOT |
| Coarse graph runtime | `ExecutionGraph`; graph edge scopes were realized by `CoarseDeviceRuntime` | `RuntimeReport`, queue visits and `CompletionEvent` rows rather than GOAL bytes | `completion_operation_ids` selected the graph boundary, then `CompletionReducer` produced `ExecutionResult`, `StepResult`, TTFT and TPOT |
| Historical diagnostic graph renderer | `ExecutionGraph` input, but the renderer reconstructed implicit FIFO as participant-local order | One diagnostic GOAL | GOAL quiescence if executed; no supported live step-sink metric consumer |
| Reconciled live step sink | `StepRecord` is lowered once to the authoritative `ExecutionGraph`; locality and backend execution are projections | 72 causal graph artifacts for the frozen step. In `ABCD`, 24 compute artifacts stay analytic and 48 collective artifacts become GOAL files and htsim runs | The graph completion boundary must be representable or projection fails before artifact output. Ordered artifact service produces the live `StepResult`, TTFT and TPOT |
| Reconciled coarse graph runtime | `ExecutionGraph` and the same canonical effective edge set | Scope-correct readiness, `RuntimeReport`, queue visits and completion rows | The graph-selected completion frontier remains the reducer input |

The legacy direct renderer remains available only as a diagnostic accepted
artifact. It is no longer an ordering authority in the active sink.

### Prechange disagreement census

The 46 overlapping transitions were one observable, not the full problem.
The source audit and frozen fixtures found all of the following disagreements:

| Observable | Prechange disagreement | Reconciled disposition |
|---|---|---|
| Adjacent distributed FIFO | The graph required whole-operation completion, while the monolithic GOAL admitted the next operation from rank-local entry. TRAF-10 observed early starts in 46 of 47 transitions. | The 47 distributed FIFO edges are required artifact boundaries. Both all-remote JCTs increased into their frozen global-order bands. |
| Simulator state lifetime | One monolithic GOAL retained network and controller state, while phase-by-phase execution restarted htsim for every phase. | Ordering now comes only from the graph, but cross-artifact physical state is still reset. BACK-38 owns stateful multi-artifact physical execution; `rnic-cn` fails closed for this unsupported scope. |
| Collective completion frontier | The direct renderer selected a syntactic per-rank final label, while phase composition waited for the maximum participant completion. | Canonical artifact grouping and full artifact quiescence avoid a falsely selected label. Process sequencing can strengthen participant-local frontiers, so TRAF-16 owns exact frontier fidelity. |
| Declared graph completion | A graph could name a completion-operation subset, while a GOAL process always ran to complete quiescence. | Projection proves that the declared completion boundary dominates rendered work or rejects the graph before writing. CORE-29 owns early or background completion subsets. |
| Runtime-report timestamp | Participant-local readiness could be reported using the predecessor's global completion rather than the selected participant's timestamp. | Runtime readiness and causal reports now carry the selected rank timestamp through `RuntimeReport` and `CompletionReducer`. |
| Compute estimates | The old sink could use unequal per-layer provider estimates while independent graph lowering used a uniform surrogate. | The active sink lowers once and uses the graph operations and their nominal durations for artifact service. |
| Supported dependency domain | The coarse runtime and diagnostic GOAL renderer accepted different combinations of forward, multiple and cross-rank dependencies. | One validator and projection fail closed on unsupported forms. TRAF-15, CORE-29 and CORE-30 own the deliberately rejected residual domains. |

## Exact graph projection

Both payloads produced the same structural census:

- 144 graph operations;
- 423 effective dependency edges;
- 284 participant-local edges, expanded from 212 explicit dependency
  references at their shared ranks;
- 139 whole-operation logical-queue FIFO edges;
- 72 causal graph artifacts;
- 47 distributed whole-operation FIFO edges represented as required artifact
  boundaries;
- 376 remaining cross-artifact edges in the serialized inventory, comprising
  284 participant-local edges and 92 whole-operation FIFO edges;
- 48 all-remote GOAL backend artifacts after the 24 analytic compute artifacts
  are removed from the backend set.

Every effective edge occurs exactly once in one projection mechanism. Every
inter-operation GOAL `requires`, if present, must carry one graph-edge
provenance. Every collective-internal `requires` is instead identified as
collective-internal and names its owning graph operation. In this frozen graph
all graph edges advance causal artifacts, so the 423 graph edges are accounted
by 47 required boundaries plus 376 serialized edges rather than by
inter-operation GOAL `requires` lines. Canonical comparison also checked every
operation, rank, payload, global tag and routed request partition without loss
or duplication.

The serialized inventory preserves the original edge scope for checking, but
ordered process execution can strengthen participant-local timing to an
artifact-wide barrier. That limitation is not described as exact physical
frontier fidelity. It is the active precision residual under TRAF-16.

The valid minimal shared-queue projection was accepted. The negative control
then removed the nonredundant whole-operation FIFO edge from `first` to
`second`. The checker rejected it before backend execution with:

```text
GOAL projection edge mismatch: missing=[('first', 'second',
'whole-operation', 'logical-queue-fifo', None)], extra=[]
```

This diagnostic proves that rejection came from the specifically perturbed
edge, rather than from an unrelated count or malformed artifact.

## Live timing observations

### Decision-relevant signed change

| Vector bytes | Historical all-remote JCT ps | Reconciled JCT ps | Signed change ps | Frozen signed band ps | Result |
|---:|---:|---:|---:|---:|---|
| 1,024 | 156,569,755 | 160,781,808 | +4,212,053 | [+4,212,005, +4,212,053] | pass |
| 2,048 | 217,222,486 | 225,539,568 | +8,317,082 | [+8,317,034, +8,317,082] | pass |

The nonzero positive changes are the required decision-relevant result. A null
result would have contradicted the TRAF-10 overlap evidence.

### Exact six-cell sweep

| Vector bytes | Placement | StepResult JCT ps | TTFT ps | TPOT ps | Frozen JCT band ps | Result |
|---:|---|---:|---:|---:|---:|---|
| 1,024 | `AAAA` | 7,121,000 | 7,121,000 | 7,121,000 | [7,121,000, 7,121,000] | pass |
| 1,024 | `AABB` | 139,195,840 | 139,195,840 | 139,195,840 | [139,195,840, 139,195,840] | pass |
| 1,024 | `ABCD` | 160,781,808 | 160,781,808 | 160,781,808 | [160,781,760, 160,781,808] | pass |
| 2,048 | `AAAA` | 14,180,000 | 14,180,000 | 14,180,000 | [14,180,000, 14,180,000] | pass |
| 2,048 | `AABB` | 182,367,680 | 182,367,680 | 182,367,680 | [182,367,680, 182,367,680] | pass |
| 2,048 | `ABCD` | 225,539,568 | 225,539,568 | 225,539,568 | [225,539,520, 225,539,568] | pass |

The associated locality observations also matched all six frozen cells
exactly. They are exact-oracle evidence, not additional genuine-risk
instances.

| Vector bytes | Placement | Total directed bytes | Fabric bytes | NVLink bytes | NVLink service ps |
|---:|---|---:|---:|---:|---:|
| 1,024 | `AAAA` | 11,870,208 | 0 | 11,870,208 | 7,097,000 |
| 1,024 | `AABB` | 11,870,208 | 7,913,472 | 3,956,736 | 2,442,000 |
| 1,024 | `ABCD` | 11,870,208 | 11,870,208 | 0 | 0 |
| 2,048 | `AAAA` | 23,740,416 | 0 | 23,740,416 | 14,156,000 |
| 2,048 | `AABB` | 23,740,416 | 15,826,944 | 7,913,472 | 4,838,000 |
| 2,048 | `ABCD` | 23,740,416 | 23,740,416 | 0 | 0 |

TTFT and TPOT equal JCT because all three replays used the same controlled
step. They establish reachability through the live metric chain but are
algebraically entailed here and add no scored evidence.

For each all-remote payload, the composed CSV audit found 47 adjacent tag
gaps, zero early transitions and a minimum gap of 0 ps. The observed gaps were
alternating 0 ps and 1,000 ps. These are required causal guards, but the
ordered process offsets make their sign true by construction, so they remain
fatal-unscored.

## Artifact acceptance

The diagnostic direct renderer preserved the previously accepted all-remote
GOAL bytes exactly:

| Vector bytes | Bytes | SHA-256 | Disposition |
|---:|---:|---|---|
| 1,024 | 72,819 | `0417832c8788a0477d48b414cf2d8456b87215abd1d0193ba46fb8db46185d8a` | accepted diagnostic identity preserved |
| 2,048 | 72,819 | `bcd72e63546d03efaddd48c16e160457d1e28f19795036d1f871788d78cf5a02` | accepted diagnostic identity preserved |

The active all-remote sink intentionally changed from one independently
ordered GOAL to 48 graph-projected backend artifacts. Their aggregate
manifests were not frozen before observation and are explicitly re-accepted
as **post-specified** locks:

| Vector bytes | GOAL artifacts | Total bytes | Artifact sizes | Post-specified aggregate SHA-256 | Semantic cause |
|---:|---:|---:|---|---|---|
| 1,024 | 48 | 41,844 | 46 at 872 bytes, 2 at 866 bytes | `b6236b6a94203b3a0e595587d1745557b66a2f65bb4ba53510c4a89d7324c47d` | graph whole-operation FIFO replaces rank-local phase entry; JCT +4,212,053 ps |
| 2,048 | 48 | 41,844 | 46 at 872 bytes, 2 at 866 bytes | `e00730b7142901a96a90062b7675cca18b317e142485875f040658419781e8aa` | graph whole-operation FIFO replaces rank-local phase entry; JCT +8,317,082 ps |

Each aggregate digest covers every artifact name and exact payload in sorted
manifest order, so it locks the complete changed step artifact set rather than
only a count or a timing outcome.

The smaller tracked regression fixtures were audited separately. Their
post-specified reacceptances are not production evidence and do not enter any
score:

| Fixture | Previous accepted form | Reconciled accepted form and effect |
|---|---|---|
| Absent-observation overlap fallback | The 4,127-byte graph wire, SHA-256 `aa3c836fe559973a7bf0940384c2e8a84e6af84e0fbd2c02d3b89774ee0c8e2d`, plus one 1,880-byte GOAL, SHA-256 `7087db6780f7e34f5a559a6505eeccc15d984c7b478cd8f0bc5838053825d4b6` | Graph wire unchanged. The active GOAL becomes six causal artifacts listed below; the old GOAL hash remains a legacy direct-renderer diagnostic. |
| Default dense step sink | One direct GOAL with SHA-256 `f8aade109ba8e3a581b7d965b3a0c76c1247016a1e37491fa84efbbf377677a5` | The direct hash remains a diagnostic. The active sink accepts four 398-byte backend artifacts listed below, two analytic compute artifacts and the post-specified 352,072,160 ps stub makespan. |
| MoE graph diagnostic | One direct GOAL with SHA-256 `2b3b73320cf02ffa11fc8a513c22edd0b450f8cedf520b28e2156fd2e60a8c0c` | Direct bytes unchanged. The active projection is reaccepted structurally as six causal artifacts, three required boundaries, 24 serialized edges and the same 48 physical sends. |
| Captured routed-MoE sink | One 123,456 ps stub backend result | Two analytic compute artifacts plus four backend collective artifacts produce 495,824 ps, a post-specified increase of 372,368 ps because the old sink charged only one independently rendered backend result. Routed pair bytes remain exact. |
| All-intra and mixed-locality sink fixtures | Eight analytic phase services, or 24 backend phase services plus a separate compute term | The graph projection exposes ten all-intra artifacts or 26 mixed artifacts. JCT remains 10,000 ps and 2,964,944 ps respectively; only the causal artifact ownership changes. |

The absent-observation active manifest is:

| Artifact | Bytes | SHA-256 |
|---:|---:|---|
| 0 | 66 | `3141fbcf0c9670b212a2f271c6514030bc827754cc86958154a48aecd6eeee1e` |
| 1 | 374 | `b50c42be665e86528de008b744d131c30d7e98d66420ebf5a46208c9ca9ce1c3` |
| 2 | 374 | `2102961315a4031b965a6a66c5406baeafc192d9100c7ea74ecde25c39d9db7e` |
| 3 | 66 | `3141fbcf0c9670b212a2f271c6514030bc827754cc86958154a48aecd6eeee1e` |
| 4 | 374 | `6449f6faa1cca2836d615a66ed7dfa9a9efceab0de2a1f22827c871f3eebd8b5` |
| 5 | 374 | `5abe15a421948ffa0e5756c3e61369f8034a964a39a76615c6c7ae140ca2483d` |

The default dense active backend manifest is:

| Artifact | Bytes | SHA-256 |
|---:|---:|---|
| 1 | 398 | `74122635377dab6e0e605c88ac7e745d126ec6da58f9a6bf2b352d7dedad1d29` |
| 2 | 398 | `3e6ee8835bc50284cba1f20fb85250f5563b118fc9b121c41a236dad026b8e62` |
| 4 | 398 | `ffd1d5e61243f5c5d3ffd07d867ed71ae769a26757d7ceb4bc0630fea26d78e2` |
| 5 | 398 | `c5520d64fe8df27809dd1df8dc2d0e8b0e700657122bb081f51333ec6daa2e76` |

The historical expectation and result files were not rewritten. The prior
direct artifact was wrong as the live ordering authority, not as a diagnostic
byte fixture.

## Evidence classes and entailment

Evidence classes remain separate and no counts below are added together.

| Evidence class | Count | Outcome and scoring |
|---|---:|---|
| Run configurations | 6 cells, 3 replays each | All completed; configuration evidence only |
| Signed-JCT behavioral family | 2 instances | 2/2 passed and scored |
| Edge-mutation behavioral family | 1 instance | 1/1 passed and scored |
| Exact JCT and locality oracles | 6 rows | 6/6 passed; fatal-unscored |
| Structural projection census | 2 payload rows | 2/2 passed; fatal-unscored |
| Direct and omitted-placement identity | 2 payload rows | 2/2 passed; fatal-unscored |
| Composed causal gaps | 2 payload rows, 47 transitions each | Both passed; by-construction and fatal-unscored |
| Native simulator execution and quiescence | Fabric-bearing cells | Passed; native execution evidence, not a behavioral denominator |
| Clean pre-production Python regression | 951 collected | 944 passed, 7 skipped; test evidence only |
| Final post-closure Python regression | 952 collected | 945 passed, 7 skipped; test evidence only |

The genuine-risk headline is therefore **2/2 families and 3/3 instances**.
The runner evaluated live `StepResult` values before applying exact bands,
inventories or hashes. It evaluated the mutation from the checker's specific
accept/reject outcome before exact edge counts. Neither scored relation was
pinned by an earlier fatal oracle.

The causal gaps were also read from the composed CSV rows before checking the
later exact zero-early count. That evaluation order does not make their sign
genuine risk: absolute cross-artifact time was constructed by adding the
already ordered artifact services. The family was therefore removed from the
score instead of using entailed evidence to inflate it. Exact counts, byte
conservation, authority labels, completion conservation, identity paths,
check-only behavior and physical quiescence remain fatal-unscored.

## TRAF-12 closure map

Every sentence in the registered TRAF-12 entry is quoted and mapped below.

> "reconcile the active causal phase semantics of the accepted monolithic
> all-remote GOAL with the localized phase-by-phase path."

The active sink now lowers once to an `ExecutionGraph` and derives locality,
causal artifacts and backend GOALs from it. The monolithic legacy bytes remain
a diagnostic identity only. The live all-remote path and localized cells now
use the same graph-authored order.

> "The current all-remote compatibility renderer advances ranks from
> rank-local completion frontiers, while localized execution imposes the
> global serial phase barrier registered under the TRAF-7 off path."

The audit reproduced those two prechange authorities. The accepted direct
renderer still preserves the rank-local bytes, but it no longer controls live
ordering. The 47 distributed FIFO graph edges now require ordered artifact
boundaries. Exact participant-local frontier fidelity across processes remains
TRAF-16.

> "In the captured TRAF-10 run, 46 of 47 adjacent phase transitions overlap
> in the all-remote CSV, invalidating its registered phase-additive JCT bands."

The historical JCTs and overlap finding remain unchanged. Reconciliation
moved both all-remote results upward by the registered positive amounts and
into the phase-additive bands. No historical artifact or result was edited to
manufacture that movement.

> "Establish one timing authority, prove from raw per-tag starts and
> completions that no phase enters early when overlap is disabled, and
> reproduce exact StepResult timing over node-span and payload sweeps."

`ExecutionGraph` is the one semantic ordering authority. The complete edge
projection and negative control establish enforcement. The composed raw tag
rows had zero early transitions at both payloads, reported as fatal-unscored
because process ordering constructs their sign. All six `StepResult` cells
matched their frozen exact values or bands.

> "Retain the frozen all-remote GOAL-byte identity; label and re-accept any
> unavoidable timestamp change."

Both 72,819-byte diagnostic GOAL hashes are unchanged. The active 48-artifact
aggregate hashes are labeled post-specified above, together with the exact
positive JCT changes and the reason the prior authority was wrong.

TRAF-12 is closed only for this demonstrated domain. Every unproved clause or
deliberately rejected extension is registered under an allocated residual ID:

| Residual | Scope moved out of TRAF-12 |
|---|---|
| TRAF-15 (Completeness; P2; M) | Project arbitrary forward, non-monotone and multi-predecessor execution DAGs. The current projector fails closed outside its supported causal partition. |
| TRAF-16 (Precision; P1; L) | Preserve participant-local and exact completion-frontier timing across ordered artifacts. Current process sequencing can strengthen the 284 participant-local serialized edges. |
| CORE-29 (Completeness; P2; M) | Support an early or background graph completion subset. Current projection rejects a boundary that does not dominate all rendered work. |
| CORE-30 (Completeness; P2; M) | Realize asynchronous control work whose destination-local readiness is not covered by the supported runtime form. Current runtime rejects it. |
| BACK-38 (Precision; P1; L) | Preserve congestion-control and network state across multi-artifact backend execution. Physical `rnic-cn` multi-artifact use fails closed until one stateful session can execute the projection. |

## Contradiction sweep

The required post-closure sweep reviewed all three integrator-owned documents.
They were reported here and were not edited for these findings in this branch.

- `README.md:71-73` explicitly describes the offline mode, so its one-GOAL
  statement does not contradict the reconciled closed-loop sink.
  `README.md:78-83` already agrees that every scheduler step lowers to an
  execution graph. No contradictory README statement was found.
- `docs/README_PRO.md:219-223` still presents TRAF-7 as future open work even
  though TRAF-7 is closed and this study connects graph ordering to the live
  serial sink. `docs/README_PRO.md:418` remains broadly true but omits the
  checked ordered-artifact projection and the `rnic-cn` fail-closed scope.
  The `core2_lowering` statements at lines 182 and 442 are historical and
  remain true for their fixtures.
- `docs/README_PRO.md:260-280` and `335-340` place intra-node collective
  service in the NCCL compute path. The active sink demonstrated here still
  uses the mutually exclusive traffic-owned analytic term.
- `docs/architecture.md:279-280` correctly describes the direct diagnostic
  renderer but omits the checked ordered-artifact projector. Lines 453-455
  are explicitly offline and remain true.
- `docs/architecture.md:456-459` says closed-loop completion always projects
  through `CoarseDeviceRuntime`. The active `HtsimStepSink` instead composes
  checked artifact service directly into `StepResult`, so this broad wording
  is contradictory.
- `docs/architecture.md:527-529` assigns intra-node NVLink service solely to
  the compute model, while the active sink still has a mutually exclusive
  traffic-owned analytic authority.

The semantic authority statement at `docs/architecture.md:156-165` is
consistent with this result: graph edges declare order and the runtime
realizes timing without traffic or network providers rewriting that order.
