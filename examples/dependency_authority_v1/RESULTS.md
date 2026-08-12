# Dependency authority v1 results

## TRAF-27 corrected ownership refreeze

TRAF-27 is complete for the corrected routed-token ownership scope. The
single-home-rank renderer preserved the dependency architecture: the
`ExecutionGraph` projection remains the sole ordering and completion authority,
and the independently selected ATLAHS GOAL remains a diagnostic cross-check
that cannot change `StepResult`. Both genuine-risk families passed, with
**2/2 families and 3/3 parameterized instances**. Every fatal guard passed.
Fatal guards are not reported as a fraction.

The ownership correction changed population bytes, direct artifacts, native
flow count, cross-check frontiers and both direct-versus-graph timing gaps. It
did not change the graph operation or dependency inventory. The corrected
all-remote graph JCTs are 155,702,768 ps and 215,381,488 ps. The corrected
direct completions are 150,838,767 ps and 205,653,487 ps, leaving positive
graph-minus-direct gaps of 4,864,001 ps and 9,728,001 ps.

### Chronology, command and provenance

The corrected expectations were frozen in
`bf6780f21c3029b3dbc06c1ea1868c1eeb03ec97` before the result-producing run.
The predictions and check-only literals genuinely preceded both production
runs, but that commit's message omitted common4's required precise worktree
status. Immediately before the commit, all five intended expectation paths
were staged, with no unstaged or untracked path remaining. The dry run used
the tracked `run_study.py --check-only` path; no separate untracked harness
existed. It validated only frozen literals and arithmetic and created no
artifact. This paragraph is a post-run process disclosure. It does not
retroactively alter the freeze or repair the commit-message metadata
nonconformance, so this report does not describe the commit as fully
contract-compliant preregistration. The frozen predictions still preceded all
observations and can fail independently of the later fatal guards.

The runner preparation landed at
`7618c7106d3e34703a615725776aa633f93f934f`. Its first clean completed run
produced summary SHA-256
`0cd050e6cff44439cb47d1836c0bde88198050b4fef166cad71310d5aaa2ab25`, but
that run is rejected as acceptance evidence. TRAF-B1 subtracted the frozen
direct-JCT prediction before executing the direct cross-check, so the direct
side of the scored relation was not a raw observation. The run and its passing
fatal checks are retained, but it contributes no genuine-risk score.

Commit `536185687bfebb35b4ccbb86ad665ab6fa079155` corrected only that evaluation
order. It executes both mechanisms, records the raw graph and direct
completions, scores their signed difference, and only then applies registered
completion, artifact and comparator guards. The final clean production run
observed that revision and produced summary SHA-256
`ade89d9c3180bf71a778ed7b68ceb9ce01fccef5c2f5ec1a73ec40f977a4eab5`.

The production command was:

```bash
.venv/bin/python examples/dependency_authority_v1/run_study.py \
  --source-root "$SIMLLM_MOE_E2E_ROOT" \
  --out "$SIMLLM_DEPENDENCY_AUTHORITY_RUN_ROOT"
```

The tracked 22-token Granite `length-cap` trace remained the dependency-study
workload and matched SHA-256
`36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341`.
The separate full Granite capture was supplied through
`SIMLLM_MOE_E2E_ROOT` as provenance only. The run recorded relative names,
sizes and the three frozen hashes for `capture/granite-greedy.jsonl`,
`replay-400g/steps.jsonl` and `replay-400g/routed-experts.json`; it stored no
external absolute path. This separation avoids mixing the 54-token,
three-request full capture with the historical 22-token dependency sweep.

| Provenance field | Observed value |
|---|---|
| TRAF-27 expectations-only commit | `bf6780f21c3029b3dbc06c1ea1868c1eeb03ec97` |
| Initial completed but rejected run revision | `7618c7106d3e34703a615725776aa633f93f934f` |
| Rejected `summary.json` SHA-256 | `0cd050e6cff44439cb47d1836c0bde88198050b4fef166cad71310d5aaa2ab25` |
| Raw-relation correction and final run revision | `536185687bfebb35b4ccbb86ad665ab6fa079155` |
| htsim gitlink observed by the run | `fc4400e4ca619223481536632074045cb6af2756` |
| `txt2bin` SHA-256 | `f3745f34ad86febe9c9eebef10aee5fae00b8865cb29943344fb75b0f142495b` |
| `htsim_rnic` SHA-256 | `cfb5014a663791f7619fe33309114a74e82878de860c14fc8a723713501f027d` |
| Accepted `summary.json` SHA-256 | `ade89d9c3180bf71a778ed7b68ceb9ce01fccef5c2f5ec1a73ec40f977a4eab5` |
| Runtime | Python 3.12.12 on Linux x86-64 |

The two native executable hashes are inherited historical accepted
provenance, not new scored evidence. The clean run recorded them in its
summary, and both match the historical accepted hashes.

The authored-against revisions, observed SimLLM revision and observed htsim
gitlink are separate provenance facts. No frozen equality constrains a future
live submodule pin.

### Physical sanity before exact comparison

The measured values first passed the frozen physical bounds:

| Vector bytes | Peak-egress serialization floor ps | Direct JCT ps | Graph phase-chain floor ps | Graph JCT ps | Conservative ceiling ps |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 29,839,360 | 150,838,767 | 155,702,720 | 155,702,768 | 347,702,720 |
| 2,048 | 59,678,720 | 205,653,487 | 215,381,440 | 215,381,488 | 407,381,440 |

The direct result lies above the critical-rank serialization floor and below
the conservative all-flow ceiling in both cells. The graph result is exactly
48 ps above its stricter phase-chain floor in each cell and remains far below
the ceiling. Doubling the vector adds 59,678,720 ps to graph JCT, exactly the
additional serialization term, while the 96,000,000 ps propagation total and
24,000 ps represented compute remain fixed. This independent scaling check
agrees with the network bound.

The graph JCT fell only about 3.2 percent and 4.5 percent from the
source-multiplied observations. It did not fall by either the 3.978 total-byte
ratio or the 2.007 peak-egress ratio. Fixed per-phase propagation and the
realized critical port, rather than aggregate group bytes alone, control the
result. As end-to-end context, the separate corrected 54-token EP-width-eight
study is slower than this smaller 22-token EP-width-four fixture, as its larger
traffic population and represented compute require. It is context, not an
oracle for these cells.

### Corrected six-cell sweep

All six cells matched the corrected `nvlink_locality_v1` byte and timing rows.
Each cell ran three identical controlled replays, so TTFT and TPOT equal the
listed JCT. Those metric projections establish live reachability but are
algebraically entailed by the fixed replay and add no scored evidence.

| Vector bytes | Placement | Total bytes | Fabric bytes | NVLink bytes | NVLink service ps | StepResult JCT ps | Native flows | Precision status |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1,024 | `AAAA` | 2,983,936 | 0 | 2,983,936 | 6,652,000 | 6,676,000 | 0 | accepted, CORE-41 refreeze |
| 1,024 | `AABB` | 2,983,936 | 2,011,136 | 972,800 | 2,194,000 | 136,246,720 | 96 | accepted |
| 1,024 | `ABCD` | 2,983,936 | 2,983,936 | 0 | 0 | 155,702,768 | 144 | accepted |
| 2,048 | `AAAA` | 5,967,872 | 0 | 5,967,872 | 13,286,000 | 13,310,000 | 0 | accepted, CORE-41 refreeze |
| 2,048 | `AABB` | 5,967,872 | 4,022,272 | 1,945,600 | 4,358,000 | 176,469,440 | 96 | accepted |
| 2,048 | `ABCD` | 5,967,872 | 5,967,872 | 0 | 0 | 215,381,488 | 144 | accepted |

#### CORE-41 refreeze of the two single-node rows

The TRAF-27 run recorded the two `AAAA` rows as reproducible baseline
observations rather than accepted precision, because the analytic locality
service then charged maximum source egress and omitted maximum destination
ingress, which undercharges the corrected combine star. That is now fixed. The
service charges the maximum endpoint load over both directions, and exactly
those two rows moved:

| Vector bytes | Service old ps | Service new ps | Signed change ps | JCT, TTFT, TPOT old ps | JCT, TTFT, TPOT new ps |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 4,538,000 | 6,652,000 | +2,114,000 | 4,562,000 | 6,676,000 |
| 2,048 | 9,047,000 | 13,286,000 | +4,239,000 | 9,071,000 | 13,310,000 |

Both new values were predicted before the correction existed, by this study's
own refreeze expectations and again by the CORE-41 expectations-only commit
`3879fb01a7249bbe92fe4342ad9e163570c2da1d`. The refreeze commit
`43ffeb87b3d4877f9a491d55a83ddd33254b3923` recorded them and preceded the rerun
that tested them. The rerun observed exactly those values at revision
`43ffeb87b3d4877f9a491d55a83ddd33254b3923`, produced `summary.json` SHA-256
`95286d67fa033bc66e2e054b4aab9c53976a2bf90ada7e7e31501dbe2586eee4`, matched
every unaffected row, and left this study at 2/2 families and 3/3 instances
with all fatal guards passing. These rows are exact fatal-unscored consumer
regression evidence; no scored family, instance or denominator changed.

The `AABB` cells have one local pair in each direction per phase, so every
local endpoint's egress equals its ingress and its maximum is unchanged. The
`ABCD` cells have no local service. Both were measured unchanged, as predicted.
See [the endpoint service results](../endpoint_service_v1/RESULTS.md).

### Structural and artifact result

Ownership changed sparse payloads without changing declared graph
participation or dependencies. Both payloads reproduced the frozen structural
inventory:

| Quantity | Observed value |
|---|---:|
| Operations | 144 |
| Effective dependency edges | 423 |
| Participant-local edges | 284 |
| Whole-operation FIFO edges | 139 |
| Causal graph artifacts | 72 |
| Required distributed FIFO boundaries | 47 |
| Other serialized edges | 376 |
| Backend GOAL artifacts | 48 |
| All-remote physical flows | 144 |

Explicit and omitted all-remote placement produced the same 48 GOAL payloads,
423 effective edges and 72 graph artifacts at both vector sizes. The active
manifest aggregate SHA-256 values are
`1d18818582c79ac428cb521378412e7b3cf1568da2a4ea8f07ef5136863bfd35`
and
`7ff0a45824b0d3aea1c1e99add16c0d973b089ea6ad1e2847d54f30b48641eb9`.
These post-run manifest identities are fatal-unscored locks, not scored
behavioral evidence.

The independently rendered direct GOALs matched the corrected accepted
artifacts:

| Vector bytes | Bytes | SHA-256 |
|---:|---:|---|
| 1,024 | 20,392 | `917961edf996753223857d64010fc61e4f6b08672f18dcadf42c70d60ee36c4a` |
| 2,048 | 20,392 | `16ee686eda4634886b117788b3893c893f5e12ea819736e0afdbdf63bab0e826` |

### Corrected authority cross-check

The ownership correction changed the diagnostic disagreement census but not
its conclusion. `execution-graph-projection` remained authoritative and
`atlahs-independent-goal` remained the selected cross-check. Selecting it
preserved the graph completion and authoritative artifacts exactly. Both
mechanisms executed the same 144-message inventory and reached quiescence.

| Vector bytes | Edges audited | Ordering differences | Unequal and negative frontiers | First direct gap ps | Minimum direct gap ps | Direct completion ps | Graph completion ps | Graph minus direct ps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 423 | 94 (47 participant-local, 47 whole-operation FIFO) | 32/47 | -81,920 | -716,800 | 150,838,767 | 155,702,768 | +4,864,001 |
| 2,048 | 423 | 94 (47 participant-local, 47 whole-operation FIFO) | 32/47 | -163,840 | -1,433,600 | 205,653,487 | 215,381,488 | +9,728,001 |

The 47 whole-operation FIFO differences survive because direct rank-local
entry is still weaker than the graph's distributed barrier. Participant-local
syntactic differences fell from 188 to 47 because the single-source star
removed the old replicated source frontiers. The numerical disagreements are
useful findings. They do not authorize the direct mechanism to replace or
modify the graph result.

### Evidence classes and entailment

Evidence classes remain separate:

| Evidence class | Outcome | Scoring |
|---|---|---|
| Signed graph-minus-direct JCT | 2 payload instances passed | scored genuine risk |
| Missing-edge negative control | 1 mutated-edge instance passed | scored genuine risk |
| Exact six-cell rows | all matched | fatal-unscored |
| Structural, artifact and identity inventories | all matched | fatal-unscored |
| Cross-check completeness and registered findings | both payload rows matched | fatal-unscored diagnostic evidence |
| Physical bounds and quiescence | all passed | fatal-unscored |
| Composed causal gaps | all passed | by-construction and fatal-unscored |
| External source identity and revision provenance | source hashes matched; revisions recorded separately | configuration evidence |

The genuine-risk headline is **2/2 families and 3/3 instances**. The runner
evaluated each signed difference from the raw graph `StepResult` and direct
completion before exact timing, artifact or comparator guards. Prior corrected
graph observations constrain the absolute graph value, but they do not entail
the corrected direct-versus-graph gap in this consumer. The missing-edge
mutation was likewise evaluated from the checker's raw rejection before exact
edge counts. No earlier fatal oracle pins either scored family.

All exact cells, inventories, hashes, authority labels, cross-check
completeness, identity paths, source hashes and quiescence checks passed. They
are prerequisites for interpreting the score, so they remain fatal and
unscored. Had any one failed, this run would have been void rather than losing
a point.

### Repository gates

The final tracked state passed these independent gates:

| Gate | Result |
|---|---|
| Registered production CLI with `--check-only` | Passed; validated the frozen registry without reading either path, invoking a native binary or creating output |
| `.venv/bin/ruff check .` | Passed |
| `.venv/bin/pytest -q` | 1,050 passed, 7 skipped |
| `python3 scripts/task_progress.py --check` | Passed |
| `git diff --check` | Passed |
| `git ls-files -s third_party/htsim` | Preserved gitlink `fc4400e4ca619223481536632074045cb6af2756` |

The `scripts/check_docs_format.py` command named by the local agent rules is
not present in this checkout. The complete available Python suite passed, and
the task-progress drift checker passed separately.

### TRAF-27 closure map

Every registered acceptance clause is quoted and mapped to evidence below.

1. Input identity

   > "The tracked dependency workload and all three external provenance inputs
   > match their frozen hashes, and no external absolute path is stored."

   The summary records the tracked trace SHA-256 and all three external
   relative paths, sizes and SHA-256 values. All matched. Configuration names
   the tracked trace as the workload and the full capture as provenance only;
   no absolute source path appears in the record.

2. Ownership and structure

   > "Corrected ownership yields 144 positive flows and the frozen byte rows,
   > while graph operations, effective edges, artifacts, boundaries and
   > serialized edges remain in their exact singleton bands."

   Both all-remote cells produced 144 flows. All six byte rows matched, and
   both payloads retained 144 operations, 423 edges, 72 graph artifacts, 47
   boundaries, 376 serialized edges and 48 backend artifacts.

3. Timing and physics

   > "Both graph JCTs and both raw graph-minus-direct relations land in the
   > registered bands and within the physical floors and ceilings."

   The graph JCTs were 155,702,768 ps and 215,381,488 ps. Their raw positive
   gaps were 4,864,001 ps and 9,728,001 ps. Every value landed inside its
   frozen band and the physical table above.

4. Sole authority and comparison

   > "`ExecutionGraph` remains authoritative, selecting ATLAHS changes no graph
   > result or authoritative artifact, and the comparator reports the registered
   > 423-edge inventory and 94 disagreements."

   Both typed reports name the graph projection as authority, preserve its
   completion and artifact manifest, audit all 423 edges, and report exactly
   94 differences split 47 plus 47. The direct result remains diagnostic.

5. Fatal acceptance

   > "Exact cells, causal projection, negative-control acceptance, quiescence,
   > identity and provenance guards all pass. Any failure makes the run void;
   > fatal guards are not reported as fractions."

   Every fatal field in the summary is true. The valid projection was accepted,
   the removed FIFO boundary was rejected with the named missing edge, both
   mechanisms were quiescent, and all identity and provenance checks passed.
   The result reports that outcome without converting fatal guards into a
   fraction.

6. Precision boundary and publication seam

   > "Single-node analytic values are explicitly pending CORE-41, all stale
   > published current-value surfaces are corrected, historical consumers point
   > to the corrected table, and every remaining contradiction is reported."

   The single-node rows above and the owning module text explicitly mark
   CORE-41 pending. Every listed current-value surface in `README.md`,
   `docs/README_PRO.md`, `docs/modules/traffic.md`, `docs/modules/core.md` and
   `examples/routed_supply_v1/RESULTS.md` is corrected, and the historical
   consumer results point to this corrected section. This file labels the
   lower TRAF-12 values as historical. Residual contradictions in adjacent
   non-owning module documents and the architecture wording hits required by
   the closure sweep are reported below and were not edited here.

### Contradiction sweep

The post-closure sweep found no stale ownership or dependency numeric claim in
the now-corrected `README.md` and `docs/README_PRO.md`. The latter now points
the historical preplay result to its corrected 48-flow replay. It found no
numeric ownership contradiction in `docs/architecture.md`.

Two adjacent module summaries outside TRAF-27's owning documents still state
the historical comparator result as current:

- `docs/modules/backends.md:410-414` reports 235 differences, including 188
  participant-local mismatches, and 46/47 unequal early frontiers;
- `docs/modules/goal.md:62-66` repeats the 188 participant-local mismatches and
  46/47 frontier result.

For the corrected renderer, both should read 94 differences, split 47
participant-local and 47 whole-operation FIFO, with 32/47 unequal early
frontiers. They are reported here rather than edited because the task owns
only the traffic and core module documents.

The sweep also retained two older architecture wording hits unrelated to the
ownership numbers. `docs/architecture.md:456-459` describes a closed-loop path
through `CoarseDeviceRuntime`, while the demonstrated `HtsimStepSink` composes
checked artifact service directly into `StepResult`.
`docs/architecture.md:503-508` still attributes the already closed
reconciliation to TRAF-12. The one-authority semantic statement agrees with
this run; only its task-status wording is stale.

Everything below this point is retained TRAF-12 chronology. Its
source-multiplied bytes, 576-flow inventory, 235 disagreement count and older
timings remain true for that historical workload, but they are not current
acceptance values. The corrected TRAF-27 tables above are authoritative for
the single-home-rank renderer.

## Historical TRAF-12 post-closure amendment: selectable independent cross-check

TRAF-12 remains closed for the demonstrated serial step-sink scope, with a
more precise per-run authority contract. Exactly one mechanism decides
ordering in any run. The active `HtsimStepSink` uses `ExecutionGraph` as its
authority for operation identity, logical queues, dependency scope,
completion identity and the `StepResult`. Its locality plan, authoritative
GOAL artifacts and backend ordering are checked graph projections.

The independent direct ATLAHS GOAL path is retained. With the default
`dependency_cross_check=None`, only the graph-authoritative path executes.
Selecting `dependency_cross_check="atlahs-goal"` additionally renders and
executes the same all-remote operation and physical-message schedule through
the independent direct GOAL dependency mechanism. This study set
`dependency_cross_check_tolerance_ps=0`. The graph result remains
authoritative. The cross-check result is never averaged with it, never
silently preferred, and never allowed to override it. A disagreement is a
successful diagnostic finding rather than an API failure. Malformed or
incomplete comparison evidence still fails because no valid comparison can
then be made.

The selected comparison first proves equal operation identity and the same
576-message physical inventory. It then reports three disagreement classes:

- an **ordering-scope difference** exists when a graph ordering edge
  requires predecessor terminal labels that do not reach every applicable
  direct-GOAL target entry;
- a **phase-frontier difference** exists when the evaluated signed frontier
  gap differs between mechanisms. A negative direct gap while the graph
  authority has a nonnegative gap is the registered early-entry subtype;
- a **completion-time difference** exists when the absolute direct-minus-
  authority completion difference exceeds the registered tolerance.

The structural report retains each predecessor and target operation ID,
rank, terminal label, entry label, and the missing predecessor ranks for each
target. It therefore reports why a boundary differs, rather than only
returning a count. Top-level fields name `execution-graph-projection` as the
authority and `atlahs-independent-goal` as the cross-check, then retain both
completion times, their signed difference and tolerance, artifact identities,
quiescence and flow counts. The structural comparison audits all 423
canonical effective graph edges. The frozen registered subset contains 47
distributed whole-operation FIFO boundaries, and all 47 are deficient in the
independent direct `requires` reachability. The expanded comparator also found
188 participant-local syntactic-frontier mismatches, for 235 structural
differences in total. Those additional 188 were observed only after expanding
beyond the frozen subset and are explicitly **post-specified** diagnostic,
unscored findings. Raw frontier comparison remains scoped to the 47
distributed boundaries. It showed early direct phase entry at 46 boundaries
for each payload, while the graph authority had a minimum boundary gap of
0 ps. The remaining boundary had the same 1,000 ps gap in both mechanisms,
so 46 of 47 evaluated frontier gaps differed in these cells.

| Vector bytes | Structural edges audited | Ordering-scope differences | Unequal frontiers | Negative direct frontiers | First direct gap ps | Minimum direct gap ps | Direct completion ps | Graph completion ps | Direct minus graph ps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 423 | 235 (47/47 registered distributed, 188 post-specified participant-local) | 46/47 | 46/47 | -368,640 | -1,413,120 | 156,569,755 | 160,781,808 | -4,212,053 |
| 2,048 | 423 | 235 (47/47 registered distributed, 188 post-specified participant-local) | 46/47 | 46/47 | -737,280 | -3,675,091 | 217,222,486 | 225,539,568 | -8,317,082 |

The completion tolerance was 0 ps, so both nonzero completion differences
are reported disagreements. Both executions reached quiescence and each
reported 576 flows. The direct completion is shorter in both cells because
rank-local direct dependencies admit overlap that the graph's
whole-operation boundaries prohibit. These are findings, not a reason to
discard the independent mechanism.

The supported cross-check scope in this amendment is an all-remote schedule.
A selected cross-check with local NVLink segments rejects before writing a
cross-check artifact because that would compare different physical
schedules. TRAF-16 owns exact participant-local frontier fidelity. The local
`dependency_cross_check` option is the existing dependency seam's selector,
not a second repository-wide configuration design. CORE-36 owns the future
unified fidelity-selection surface.

### Amendment chronology and provenance

The original closure and results below were not rewritten to manufacture
this later architecture direction. Expectations for the selectable
cross-check were frozen in
`69a7ada2ec192b3d7eec81b53529a5662371e3b1` after the original closure and
before its implementation or production run. Initial implementation landed
at `ca167cfdd3da0a74c9ee6a531e31e5d0f61f0eee`. Its first completed amendment
run produced summary SHA-256
`a9778a8e2147446555a0a61a13d9a1730fe9ae18f072b074ca8c4faafc13ad80`, but
that run is rejected as acceptance evidence. Its fatal-unscored cross-check
equality and signed-band guards ran before TRAF-B1 scoring, so they pinned the
same signed completion relation before it entered the scored denominator.
The run is recorded here rather than silently discarded.

The correction evaluates TRAF-B1 from raw live `StepResult` values and
TRAF-B3 from the raw mutation-checker result before any entailing cross-check
guard. Only afterward does it apply exact diagnostic expectations. That
correction landed in `6470c31f90434c8255cbdf1258fa199662a86cfd`. The final
clean rerun observed `6470c31f90434c8255cbdf1258fa199662a86cfd` and produced
summary SHA-256
`8203f424a26da82eadc74414f5789187e7f2695a1989a693e1fb428b0fc06123`.
The fidelity contract read from the
integrator branch was PR 40 commit
`d41ff1a8e59d1669a03a7bd1501242704eb39d72`.

| Amendment provenance field | Revision or value |
|---|---|
| Cross-check expectations-only commit | `69a7ada2ec192b3d7eec81b53529a5662371e3b1` |
| Initial implementation and rejected completed run | `ca167cfdd3da0a74c9ee6a531e31e5d0f61f0eee` |
| Rejected run `summary.json` SHA-256 | `a9778a8e2147446555a0a61a13d9a1730fe9ae18f072b074ca8c4faafc13ad80` |
| Entailment-order correction commit | `6470c31f90434c8255cbdf1258fa199662a86cfd` |
| Corrected SimLLM revision observed by the final run | `6470c31f90434c8255cbdf1258fa199662a86cfd` |
| Integrator PR 40 contract commit | `d41ff1a8e59d1669a03a7bd1501242704eb39d72` |
| Accepted amendment `summary.json` SHA-256 | `8203f424a26da82eadc74414f5789187e7f2695a1989a693e1fb428b0fc06123` |
| `txt2bin` SHA-256 | `f3745f34ad86febe9c9eebef10aee5fae00b8865cb29943344fb75b0f142495b` |
| `htsim_rnic` SHA-256 | `cfb5014a663791f7619fe33309114a74e82878de860c14fc8a723713501f027d` |

The native hashes are unchanged from the original production record. At that
historical revision, the amended study command was:

```bash
.venv/bin/python examples/dependency_authority_v1/run_study.py \
  --out "${SIMLLM_DEPENDENCY_AUTHORITY_AMENDMENT_RUN_ROOT:?configure this variable}"
```

That command requires the historical revision. Current HEAD uses the TRAF-27
command with `--source-root` shown above.

### Amendment artifact acceptance and scoring

Selecting the cross-check adds artifacts only below `cross-check/`. It does
not change the authoritative graph artifacts or `StepResult`, and the
unselected path creates no cross-check directory. The independently rendered
artifacts remain byte-identical to the accepted legacy direct fixtures:

| Vector bytes | Bytes | SHA-256 | Disposition |
|---:|---:|---|---|
| 1,024 | 72,819 | `0417832c8788a0477d48b414cf2d8456b87215abd1d0193ba46fb8db46185d8a` | accepted legacy identity preserved |
| 2,048 | 72,819 | `bcd72e63546d03efaddd48c16e160457d1e28f19795036d1f871788d78cf5a02` | accepted legacy identity preserved |

The comparison's complete inventories, quiescence, authority preservation
and typed report are fatal-unscored evidence. The differences themselves are
diagnostic findings. The amended runner records the raw three-class report
and signed observations before applying the registered exact and bounded
checks. That observation order does not turn diagnostic replication into
genuine risk. The completion values and their signed differences are also
algebraically entailed by the already accepted direct and graph JCTs, so the
amendment adds no genuine-risk numerator or denominator. The original score
remains **2/2 families and 3/3 parameterized instances**.

A programmatic comparison with the original production summary found exact
equality for `cells`, `behavioral`, `behavioral_score`, `exact_oracle_rows`,
`structural_invariants`, and `all_remote_identity`. Both summaries retained a
true fatal-check result. This is byte- and value-level preservation evidence,
not another scored relation.

## Original TRAF-12 closure result

TRAF-12 closed for the demonstrated serial step-sink scope. In the
authoritative live sink configuration, `ExecutionGraph` became the single
semantic authority for operation identity, logical queues, dependency scope
and completion identity. The locality plan, GOAL artifacts, backend ordering
and coarse runtime became checked projections of that graph.

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

At that historical revision, the production command was:

```bash
.venv/bin/python examples/dependency_authority_v1/run_study.py \
  --out "${SIMLLM_DEPENDENCY_AUTHORITY_RUN_ROOT:?configure this variable}"
```

Current HEAD requires the TRAF-27 command with `--source-root` shown above.

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
projection consume that same set. On the authoritative path, GOAL is a
read-only graph rendering. Distributed whole-operation edges become required
ordered artifact boundaries when the grammar cannot name a cross-rank
predecessor. All other cross-artifact edges remain in a scope-preserving
serialized inventory, while collective-internal relations retain separate
operation-owned provenance. The separately selected direct ATLAHS GOAL
cross-check remains free to construct and execute its own dependencies, but
its timings cannot control the graph-authoritative `StepResult`.

### Path inventory

| Path | Semantic input and ordering authority | Rendered or runtime artifacts | Completion boundary and metric consumer |
|---|---|---|---|
| Historical direct all-remote step sink | `StepRecord`; `render_step_goal` and collective patterns independently chose rank-local order | One monolithic GOAL, one htsim process and one completion CSV | Full GOAL quiescence became `StepResult`, TTFT and TPOT |
| Historical placement-enabled step sink | `StepRecord`; the locality phase loop independently imposed phase order | Isolated analytic local phases plus per-phase fabric GOALs, each in a fresh htsim process; phase durations composed as `max(local, fabric)` and then summed | The accumulated phase sum became `StepResult`, TTFT and TPOT |
| Coarse graph runtime | `ExecutionGraph`; graph edge scopes were realized by `CoarseDeviceRuntime` | `RuntimeReport`, queue visits and `CompletionEvent` rows rather than GOAL bytes | `completion_operation_ids` selected the graph boundary, then `CompletionReducer` produced `ExecutionResult`, `StepResult`, TTFT and TPOT |
| Historical diagnostic graph renderer | `ExecutionGraph` input, but the renderer reconstructed implicit FIFO as participant-local order | One diagnostic GOAL | GOAL quiescence if executed; no supported live step-sink metric consumer |
| Reconciled live step sink | `StepRecord` is lowered once to the authoritative `ExecutionGraph`; locality and backend execution are projections | 72 causal graph artifacts for the frozen step. In `ABCD`, 24 compute artifacts stay analytic and 48 collective artifacts become GOAL files and htsim runs | The graph completion boundary must be representable or projection fails before artifact output. Ordered artifact service produces the live `StepResult`, TTFT and TPOT |
| Selected independent ATLAHS cross-check | The direct renderer independently constructs GOAL `requires` while `ExecutionGraph` remains the live sink authority | One additional monolithic GOAL and htsim completion CSV under `cross-check/`, plus a typed comparison report | Direct quiescence is diagnostic only. It reports structural, frontier and tolerance-qualified completion disagreements without changing `StepResult` |
| Reconciled coarse graph runtime | `ExecutionGraph` and the same canonical effective edge set | Scope-correct readiness, `RuntimeReport`, queue visits and completion rows | The graph-selected completion frontier remains the reducer input |

The legacy direct renderer remains independently executable as a diagnostic
cross-check and accepted artifact. It is not the ordering authority for the
active graph-sink run, but its independence is deliberately preserved so its
disagreements remain observable.

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

The independent direct cross-check renderer preserved the previously
accepted all-remote GOAL bytes exactly:

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
direct artifact was wrong as the silently competing live ordering authority,
not as an independent cross-check or diagnostic byte fixture.

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
| Selectable ATLAHS cross-check report | 2 payload rows, 423 structural edges and 3 typed disagreement classes each | Complete and preserved; fatal-unscored report evidence. The frozen 47 distributed differences and 188 post-specified participant-local differences are diagnostic findings. |
| Cross-check boundary timestamps | 2 payload rows, 47 transitions each | Raw observations retained; diagnostic evidence, not a behavioral denominator |
| Composed causal gaps | 2 payload rows, 47 transitions each | Both passed; by-construction and fatal-unscored |
| Native simulator execution and quiescence | Fabric-bearing cells | Passed; native execution evidence, not a behavioral denominator |
| Clean pre-production Python regression | 951 collected | 944 passed, 7 skipped; test evidence only |
| Final post-closure Python regression | 952 collected | 945 passed, 7 skipped; test evidence only |
| Post-amendment Python regression | 971 collected | 964 passed, 7 skipped; test evidence only |

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

`ExecutionGraph` is the one semantic ordering authority for the authoritative
live execution. The independently selected ATLAHS run reports its differences
but cannot replace that result. The complete edge projection and negative
control establish enforcement. The composed raw tag rows had zero early
transitions at both payloads, reported as fatal-unscored because process
ordering constructs their sign. All six `StepResult` cells matched their
frozen exact values or bands.

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

### Post-amendment sweep

No new contradiction was found in `README.md`. This branch predates PR 40, so
its local `docs/README_PRO.md` and `docs/architecture.md` copies do not yet
contain the integrator-owned fidelity sections. The sections were therefore
read from `origin/main` at
`d41ff1a8e59d1669a03a7bd1501242704eb39d72`. Their per-run single-authority,
explicit independent cross-check, reported-disagreement, and CORE-36
selection-surface requirements agree with this amendment. Both integrator
documents were left untouched. The older broad architecture statement that
all closed-loop completion projects through `CoarseDeviceRuntime` remains a
genuine contradiction for direct `HtsimStepSink` composition, as recorded in
the original sweep below; the amendment introduced no additional hit.

### Original closure sweep

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
