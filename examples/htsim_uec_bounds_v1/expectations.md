# HTSIM-25 and HTSIM-8 UEC bound reconciliation expectations

The original expectations-only record for this UEC validation-bound
reconciliation preceded the first result-producing command and every backend
implementation change made for HTSIM-25. This tracked record now includes the
disclosed attempt-five refreeze after four void attempts. The study does not
claim a TTFT or TPOT relation.
Its decision-relevant result is whether every known bound miss has a causal
classification strong enough to make the native backend gate citable.

If any case remains causally ambiguous, if any fatal guard fails, or if the
final default gate is not green, HTSIM-25 remains open. HTSIM-8 is judged
separately and remains open unless the final eight-plan gate exits zero and a
deliberate bound mutant is rejected and then removed.

## Attempt chronology and attempt-five refreeze

Attempt one used the unmodified bound-authorship binary. It is void. Before any
historical FCT could be observed, all three completed plans violated the frozen
historical-observability guard: the packet summaries showed that every
simulation ran, but the common validator saw zero completions. Source review
then found both completion-print blocks commented out at authorship commit
`896cc765aabce04b0707a42eacf8774275a5d771` in
`htsim/sim/uec.cpp:823-868`. The interrupted evidence is retained outside Git
and is not scored or used for closure.

Attempt two is also void. Its transport projection was still print-only, but a
post-start source-diff count found two redundant empty lines removed at
end-of-file where the attempt-two refreeze permitted one. No result from that
attempt is scored or used for closure. Its interrupted evidence is retained
outside Git.

This section refreezes attempt three before its first run. Attempt three builds an
authorship transport projection with exactly the four comment-delimiter edits
from commit `77943e48aec31cc5c7cff4e93e0296fce5a50097` ("Re-enable UEC
flow completion prints"). That commit changes four lines in
`htsim/sim/uec.cpp:834-882`: it removes the opening and closing comment tokens
around the two already-authored `cout` blocks. It changes no event, state,
packet, route, random draw or timing expression. The projection therefore
exposes the authorship binary's existing completion times without changing its
modeled behavior.

Attempt three is also void. Later independent review found that the plan
projector appended a final LF to two tracked plans that have none:
`validate_load_balancing_snd.txt` and
`validate_uec_connreuse.txt`. The frozen plan-byte guard permits no EOF
normalization. Attempt three's evidence is retained, unscored and unused for
closure.

This section refreezes attempt four before the plan projector is repaired and
before its first invocation. The workload, candidate binaries, 17 target
identities, causal classes, bound-update rules, paired relations, gate and
mutant remain unchanged from the expectations frozen before any measured
attempt. The only evidence-method change is that projected plans must preserve
each source plan's exact line-ending sequence and final-newline state while
making only the already-authorized matrix, topology, binary and external-log
path substitutions. The result summary must record source and projected plan
digests and final-newline states. Any source or projected plan outside that
exact rule voids attempt four.

The projector reads bytes and retains each line terminator. It replaces only
the body of a matrix row, `!Binary` row or `!Param -topo` row. It inserts
exactly one `!Param -o` immediately after a terminated Binary row, using that
row's original terminator, and rejects a pre-existing log parameter or an
unterminated Binary row. Every other byte and the source EOF state stay exact.
An independently implemented synchronized audit consumes exactly one
projected row per source row plus the one authorized log row after each
Binary, reconstructs the source bytes, rejects an extra or missing projected
row, and requires matching source/projected EOF states. The audit runs in
memory during post-implementation check-only, before every simulator child and
again after every plan run. Per plan, the summary records source and projected
SHA-256, EOF state, replacement and insertion counts, and successful exact
round trip. It also records the SHA-256 of every unique referenced matrix and
topology input.

Attempt four uses the same authorship transport projection as attempt three.
It is independently admissible only if a source diff proves the four
print-only edits are the projection's only executable-source change from
`896cc765` and the projected binary emits the completion and packet-summary
vocabulary. The source file-editing path removes its two redundant trailing
empty lines; the source diff must report exactly four changed code lines plus
those two empty-line deletions, i.e. four insertions and six deletions. Any
other source difference voids attempt four.

Because attempt three exposed numbers before this evidence-method repair,
attempt four is disclosed as a post-start rerun. No outcome expectation or
bound rule is changed to fit those numbers. The classification and relation
expectations already precede the first measured attempt; the only new
assertions are plan-byte integrity and the source-derived horizon-unit
correction below.

Attempt four is void. Its first measured invocation used `python3.10` as the
study driver instead of the registered `.venv/bin/python`. Independent review
caught the command deviation after the sender plan completed and while the
receiver plan was running. The process was interrupted, and its partial
artifacts are retained outside Git as off-protocol findings. They are unscored
and unused for closure.

This section refreezes attempt five before its first measured invocation.
Attempt five uses the already-frozen candidate binaries, workload, target
identities, causal classes, bound rules, relations, fatal guards, projection
implementation and exact source/input manifests without change. It uses the
literal registered `.venv/bin/python` driver and a fresh `attempt5-baseline`
output path. No backend code, modeled behavior, harness implementation,
expectation or measured-value rule changed after the interrupted attempt-four
invocation. Attempt five remains a disclosed post-start rerun because earlier
void attempts exposed findings.

## Prior evidence and external-source audit

The starting failure inventory is prior published evidence, not a result of
this run. `examples/htsim_commit_gate_v1/RESULTS.md:133-174` reports 17 misses
among 95 experiments in eight default plans. That report also establishes that
every miss was an FCT-bound failure, not a completion-count, input, child-status
or output-shape failure. This study must reproduce that pre-change inventory
before it may classify it.

The evidence is authored against the backend gate repair at HTSIM commit
`1f2c124c9738edcfa0f6044b4667c230e75a542c`. The result report records every
backend commit actually observed. No equality with a live SimLLM submodule pin
is assumed.

Source and history were audited before this freeze:

- The eight-plan gate and its plan order are in
  `htsim/sim/datacenter/commit_check.sh:8-28` at the authored-against commit.
  `set -euo pipefail` makes a validator failure terminate the gate.
- The common comparator is `htsim/sim/datacenter/validate.py:40-229`. It parses
  the inline `tailFCT` and per-flow `FCT` authorities, runs every experiment in
  a plan, checks the completion count and packet summary, and exits nonzero if
  any experiment fails. This study uses that comparator for current and
  historical binaries rather than implementing another bound checker.
- `git log --follow` shows that the seven UEC and load-balancing plan files were
  introduced by `896cc765aabce04b0707a42eacf8774275a5d771` and have no later
  plan-content commit. That commit is therefore the bound-authorship
  reference, not an assertion that its transport behavior is correct.
- `htsim/sim/datacenter/main.h:9` fixes the host NIC at 100 Gbit/s. The default
  leaf-spine topology also declares 100 Gbit/s links and 1 us link latency in
  `htsim/sim/datacenter/topologies/leaf_spine_1024.topo:5-15`.
- Failed links are multiplied by `_failed_link_ratio = 0.25` in
  `htsim/sim/datacenter/fat_tree_topology.cpp:123-124,1036-1061,1313-1314`.
  The two failed-link plans instead label those links as running at 10 percent.
  The generator independently uses `failed / 4` in its capacity calculation at
  `htsim/sim/datacenter/generate_permutation_experiments.py:11-12`, confirming
  that the executable mechanism is 25 percent, not 10 percent.
- The same generator computes an ideal serialization term plus 9 us, then
  applies a 1.20 tail allowance at
  `htsim/sim/datacenter/generate_permutation_experiments.py:11-12,32`. Its
  failure adjustment is aggregate-capacity arithmetic. The validator checks
  the maximum completed-flow FCT, so an aggregate adjustment cannot by itself
  bound an individual flow that uses a degraded path.
- The completion vocabulary and its environment gate are in
  `htsim/sim/uec.cpp:21-24,875-934`. Every measured invocation sets
  `HTSIM_TRACE_FLOW_COMPLETIONS=1`.

The source audit changes the decision if it is refuted. In particular, a
failed-link observation may not be called a simulator regression merely
because it exceeds a bound derived from aggregate capacity. Conversely, a
bound may not be relaxed merely because the current binary exceeds it.

## Frozen inventory and sweep

The default gate contains 95 experiments with the frozen per-plan counts 15,
15, 15, 10, 10, 9, 9 and 12. The 17 prior misses are the frozen target
inventory:

1. sender: `1024 node incast`;
2. sender: `3 to 1 incast with long running flow`;
3. sender: `Small permutation, INC (16 nodes)`;
4. receiver: `outcast incast`, per-flow `Uec_1_0`;
5. receiver: `Small permutation, INC (16 nodes)`;
6. both: `Small permutation, INC (16 nodes)`;
7. sender load balancing: `Large permutation, oblivious, large queues (1024 nodes)`;
8. receiver load balancing: `Large permutation, oblivious, large queues (1024 nodes)`;
9. failed sender load balancing: oblivious, default queues;
10. failed sender load balancing: oblivious, large queues;
11. failed sender load balancing: bitmap, default queues;
12. failed sender load balancing: bitmap, large queues;
13. failed sender load balancing: REPS with 4 SACKs per packet;
14. failed receiver load balancing: oblivious, default queues;
15. failed receiver load balancing: oblivious, large queues;
16. failed receiver load balancing: bitmap, large queues;
17. failed receiver load balancing: REPS with 1 SACK per packet and large queues.

The pre-change and final binaries run all 95 experiments. Historical binaries
run at least the 17 target cases. History search starts at the authorship commit
and follows the mechanically selected first-parent or ancestry path to the
authored-against gate parent. When endpoint observations differ, ordinary
binary search locates the first commit that changes the relevant tail. The
selection rule is frozen here; midpoint commits are not chosen to obtain a
preferred classification.

The parameter sweep varies at least two modeled inputs: failed-link count
(`0`, `1`, `8`) and queue capacity (automatic versus `q = 500` packets). It
also observes sender-only, receiver-only and combined congestion control.

## C1: causal classification, 17 scored instances

Each target case is one scored instance and must receive exactly one of these
classifications:

1. `wrong-at-authorship`: the authorship binary misses its own bound, or source
   arithmetic proves that the bound reduces the wrong quantity, such as using
   aggregate capacity to limit a maximum per-flow statistic.
2. `stale-after-intended-change`: the authorship binary passes, history locates
   the first change in the observation, and that commit explicitly repairs or
   intentionally changes the mechanism that controls the FCT. The new result
   must also satisfy the physical and paired relations below.
3. `simulator-regression`: the authorship binary passes, the first changing
   commit has no intended mechanism change that explains the direction, or the
   current result violates a frozen physical or paired relation. The simulator
   must be repaired; its bound is not relaxed.

Repository history, source behavior and independent physical relations are the
required causal evidence. Current disagreement alone is never sufficient for
the first two classifications. If more than one classification remains
plausible, the instance is unresolved, C1 fails for that instance and HTSIM-25
stays open. It does not void otherwise interpretable evidence.

Planned genuine-risk fraction: `17/17`. A competent investigation could find
an unexplained onset, a source/result disagreement, or a physical relation
failure in any case. No earlier exact oracle pins a causal classification.

### Bound update rule

An unchanged bound is preferred. A simulator-regression case keeps its bound.
For a stale bound, preserve the author's fractional slack across the identified
intentional semantic change: multiply the first post-change reference tail by
`old_bound / authorship_tail`, then round upward to the next 5 us. This uses the
authorship relation and the causal change boundary, not the final current
observation.

For a bound that was wrong at authorship, use the first applicable independent
rule:

1. derive it from a matched plan that differs in one active modeled resource,
   while preserving that matched plan's fractional slack; an observation
   cutoff may differ only when source semantics and the pre-stated physical
   range prove both cutoff values causally inert for the matched observations;
2. for a permutation path, use 1.20 times serialization plus the source's 9 us
   fixed allowance, with the actual slowest eligible link rate for a tail
   statistic, rounded upward to the next 5 us;
3. otherwise leave the case unresolved.

No bound is set to the current observation plus an invented margin. Any bound
edit must name its rule and operands in the result report.

## Physical sanity stated before measurement

All serialization floors below were stated before this study read a measured
value. Crossing a floor is a fatal finding, not a scored loss. The first three
attempt freezes mislabeled `-end` literals as microseconds. Source review for
attempt four establishes that `main_uec.cpp` passes `-end` through
`timeFromMs`, so the actual horizons are 1,000 times those labels. This
source-derived unit correction is refrozen before attempt four. Attempt four
also retains each original numeric inequality as a separate, deliberately
tighter fatal study cap: every FCT must be no greater in microseconds than its
plan's numeric `-end` literal. That legacy cap is not called a physical model
horizon, and it is preserved without reference to a prior observation.

- The 1024-node 100 KB incast sends 1023 flows into one 100 Gbit/s receiver.
  Its receiver-serialization floor is
  `1023 * 100,000 * 8 / 100e9 = 8,184 us`; its configured 15,000 ms experiment
  horizon is 15,000,000 us.
- The overlapped 3-to-1 matrix delivers 2,500,000 bytes to node 0. Its receiver
  floor is `2,500,000 * 8 / 100e9 = 200 us`; its configured 1,000 ms horizon is
  1,000,000 us.
- Every reported 2 MB permutation flow has a healthy-link serialization floor
  of `2,000,000 * 8 / 100e9 = 160 us`. A flow serialized wholly on a configured
  25 Gbit/s degraded link needs 640 us before propagation and protocol work.
  The small failed permutation has a 2,000 ms, or 2,000,000 us, horizon and the
  large failed permutations have a 3,000 ms, or 3,000,000 us, horizon.
- The outcast per-flow value is bounded below by that flow's own bytes divided
  by 100 Gbit/s, read from the matrix before its observation is read, and above
  by the plan's 3,000 ms, or 3,000,000 us, horizon.

The real-system plausibility check is deliberately modest: 2 MB cannot cross a
100 Gbit/s NIC in less than 160 us, and a path actually held at 25 Gbit/s cannot
serialize it in less than 640 us. Hundreds of microseconds are plausible for
these synthetic transfers; this gate provides no absolute prediction for a
deployed collective library.

## P1: paired scaling relations

These are behavioral support, separate from the causal classification family:

- For each congestion-control mode, the one-failed-link 16-node permutation
  tail must not be below its otherwise matched healthy permutation tail.
- For oblivious load balancing, the eight-failed-link tail must not be below
  the matched healthy-link tail. Oblivious selection has no feedback mechanism
  that can turn link degradation into a speedup.
- Within each matched oblivious configuration, `q = 500` must not produce a
  tail below the automatic-queue case. The larger buffer may leave the result
  unchanged or add queue residence, but cannot increase link service rate.

These relations are evaluated directly on raw tails before gate acceptance is
checked. Planned genuine-risk instances are determined by the available
matched pairs and reported without being combined with C1. A relation already
pinned by an exact fatal oracle is withdrawn rather than scored.

## Fatal and unscored guards

The following guards must all hold. One violation voids the run and leaves both
tasks open; they are never reported as a pass fraction.

- The plan registry contains exactly the frozen eight plans, 95 experiments and
  17 target identities, with no duplicate target.
- The plan and matrix inputs used for current and historical binaries are
  byte-identical projections of the authored files. Only `Binary`, absolute
  matrix or topology input location and external log location may change in a
  projected plan. Every source line ending and the exact final-newline state
  must be preserved.
- The authorship source projection differs from `896cc765` only by the four
  print-enabling comment-delimiter edits and the permitted redundant empty-line
  normalization audited above.
- Every simulator process exits zero, every declared connection completes and
  every experiment emits the packet summary expected by the common validator.
- Every reported FCT lies within its pre-stated physical floor, corrected model
  horizon and deliberately tighter legacy study cap.
- Every frozen target has exactly one observed decision row. This structural
  completeness guard does not decide which causal classification is correct.
- The deliberate mutant is absent from the final backend tree.

Plan counts, digests and fixed command strings are by-construction change-set
guards. They are unscored.

## G1: final default gate acceptance

The final real `commit_check.sh`, invoked with no plan override, must run all
eight tracked plans and exit zero. Its raw return code is evaluated before any
diagnostic text. This is one genuine-risk support instance for HTSIM-8 and is
not entailed by C1, because a complete classification does not force a green
gate.

Planned genuine-risk fraction: `1/1`.

## M1: deliberate mutant rejection

After a clean final gate pass, tighten the first tail bound in the tracked
connection-reuse plan from 18 us to 5 us, invoke the real gate on that plan, and
record its raw nonzero status before reading the diagnostic. Restore the exact
tracked content, prove the plan diff is empty, then rerun the same plan and
require exit zero.

The mutant input itself is author-defined and unscored. The final gate's raw
rejection is one genuine-risk instance because a gate that ignores validator
status can accept it. The clean positive controls are fatal-unscored.

Planned genuine-risk fraction: `1/1`.

## Entailment analysis

C1 evaluates causal provenance, not whether the final numeric bounds pass, so
the final gate cannot entail it. The structural target-row guard establishes
only that C1 has an input; it does not pin a causal class. P1 reads raw current
observations before the gate status and is not entailed by that status. G1 is
the one authoritative full-gate acceptance relation; per-experiment final
passes are not scored again. M1 reads the mutant gate return code before its
expected failure text, and a clean-gate pass cannot entail rejection of a
different input.

## Closure scope

HTSIM-25 registers: "reconcile the authored UEC validation bounds with current
backend behavior" and "Decide per experiment, with evidence, whether the
transport regressed or the authored bound is stale. Never relax a bound to
match an observation without that evidence." C1 plus the bound-update rule and
structural completeness guard must demonstrate both clauses for all 17 cases.

HTSIM-8 registers the remaining acceptance clause that the repaired full
default gate exits zero and is citable as release evidence. G1 and M1 must
demonstrate that clause independently of HTSIM-25's classification result.

Any registered clause not demonstrated remains open under one of the allocated
IDs HTSIM-26, HTSIM-27 or BACK-42. No ID is created for an adjacent improvement
that was not part of these clauses.

## Registered command and check-only dry run

Bulk outputs remain outside Git. The registered study command is:

```bash
HTSIM_SOURCE_ROOT="${HTSIM_PRECHANGE_SOURCE_ROOT:?configure a detached HTSIM worktree at 1ae1215}" \
.venv/bin/python examples/htsim_uec_bounds_v1/run_study.py \
  --out "${SIMLLM_DATA_ROOT:?configure the data root}/attempt5-baseline" \
  --candidate "authorship-print-projection=${HTSIM_AUTHORSHIP_BINARY:?build the print-only authorship projection}" \
  --candidate "prechange=${HTSIM_PRECHANGE_BINARY:?build the pre-change binary}"
```

The same command with `--check-only` must run before this attempt-five
expectations commit against a detached Git worktree at `1ae1215`. Check-only
invokes the backend comparator's `-dryrun` path for all eight plans, validates
the frozen inventory and candidate syntax, prints the external plan, and
creates no artifacts or measurements. A separate read-only EOF inventory must
confirm that exactly the sender load-balancing and connection-reuse plans lack
a final LF before implementation. After implementation, check-only must also
exercise the exact in-memory projection and independent audit without creating
an artifact. Historical build commands and any mechanically selected
bisection commits are recorded in RESULTS with compiler provenance; they do
not change the registered workload or comparator.

The untracked harness present at this refreeze is the already-reviewed
attempt-four implementation. It contains the frozen plan and target
registries, comparator orchestration, raw observation parsing, byte-preserving
projection and independent audit, source/input snapshots, check-only
validation and a portability-only switch from a literal `python3` child to
the current interpreter. It contains no backend fix, classification outcome
or observed timing, and it is unchanged after the interrupted attempt-four
invocation.
