# HTSIM-25 and HTSIM-8 UEC bound reconciliation results

Run on 2026-08-13. HTSIM-25 closes: all 17 previously out-of-bounds
experiments are classified one at a time, and all 17 were already outside
their bounds under the bound-authorship transport. None is a current
simulator regression. Five corrected authorities come from matched authored
plans, and the other twelve target authorities come from the configured
25 Gbit/s degraded-link serialization floor. No bound was set to a current
observation plus an invented margin. The five matched-plan authorities use the
failing authorship FCT as their measured base and preserve fractional slack
from an independently authored matched plan.

HTSIM-8 also closes, judged on its own clauses. The exact backend commit runs
all eight default plans and all 95 experiments with raw gate status zero. A
deliberate tracked-plan mutant returns raw status one, and byte-exact
restoration returns zero. The two tasks therefore close independently rather
than borrowing evidence from one another.

## Freeze, attempts and two-sided integrity

The original causal classes, target inventory, numeric formulas, failed-link
rule and relations were frozen in expectations-only commit
`132e05adcf2296a49423cf19e50def75f7e2089f`, before the first measurement and
every HTSIM-25 bound implementation change. The gate repair was already
preregistered by `htsim_commit_gate_v1` in expectations-only commit
`522f1fdc7830fd378b15cc9177b764b299d21fec` and landed as `1f2c124c`; this
branch's `1ae1215` has the identical parent and tree. The final qualifying
evidence-method and registered-command expectations-only commit is
`fe45c95b51232eac56bac072c9a67aa6e4bf8f07`. It precedes the valid attempt-six
run and follows the bound-only backend commit
`9800ea3296ff4bb017ac4d60df94917e4c8c3f0d`. The attempt-six check-only command
ran before and after that freeze; the two records are byte-identical. It
enumerated the exact eight plans, 95 experiments, 17 target identities and two
candidate labels, and created no artifact or measurement.

One matching criterion was clarified after attempt three exposed values.
Commit `f832a7c` changed “one declared resource” to “one active modeled
resource” and explicitly allowed a different observation cutoff only when
source semantics and the frozen physical range prove both cutoffs inert. This
post-start, post-specified clarification is relevant to case 1, whose two
plans use different `-end` literals. The fractional-slack formula and every
numeric operand rule stayed unchanged. Attempts one through five are void,
and attempt six reran the unchanged candidates after this criterion was
refrozen. Case 1's authority is therefore reported under that post-specified
criterion rather than presented as part of the original preregistration.

Attempts one through five are retained but void. They are not reported as
partial scores and support no closure:

1. Attempt one used the unmodified authorship binary. All three completed
   plans violated the historical-observability guard because completion
   printing was commented out. No historical FCT was readable.
2. Attempt two enabled only the authored completion prints, but its source
   projection removed two trailing empty lines where the then-current freeze
   allowed one. That source-integrity guard voided the attempt before its
   result could be used.
3. Attempt three exposed all 95 observations, but its plan projector appended
   a final LF to `validate_load_balancing_snd.txt` and
   `validate_uec_connreuse.txt`, whose tracked sources have none. That violated
   the frozen plan-byte fatal guard, so the run is void even though its findings
   were useful.
4. Attempt four used the repaired byte projector, but the measured driver was
   `python3.10` instead of the registered `.venv/bin/python`. Independent
   review caught the command deviation after one plan completed. The process
   was interrupted and its partial off-protocol findings were retained.
5. Attempt five used the repaired byte projector and literal registered
   driver, but a strict post-run method audit found that its snapshots ran at
   plan/comparator boundaries rather than once per individual simulator child.
   That deviation from the attempt-four phrase “before every simulator child”
   voids the run even though its findings are retained.
6. Attempt six was refrozen in `fe45c95` with the implemented plan-boundary
   cadence and a fresh output path. Its source projection differs from `896cc765`
   on exactly four code lines, the frozen removal of comment delimiters around
   two existing print blocks, plus two redundant trailing empty lines. The
   diff reports four insertions and six deletions. No event, state, route,
   random draw or timing expression changed. This is the valid authorship run.

Every commit after the first measured attempt is classified here for the
two-sided freeze-integrity rule:

| commit | what it fixes | modeled behavior change | measured value before and after |
|---|---|---|---|
| `c864b02` | historical completions were not observable | no, it refroze a print-only evidence projection | attempt one had no readable FCT; no modeled number changed |
| `3e92a1f` | the projection description did not identify its exact print-only boundary | no | no modeled number changed |
| `9301b94` | the frozen source-diff count omitted one redundant trailing empty line | no | attempt two stayed void; no modeled number changed |
| backend `9800ea3` | five matched-plan authorities, 24 failed-link authorities and 18 false capacity labels | no, only validation plans changed | all 17 current raw FCTs in the table below are identical before and after the commit |
| `f832a7c` | attempt three normalized final-newline bytes; it also corrected the source-derived horizon units, removed fatal-oracle entailment from C1 and post-specified case 1's inert-cutoff matching criterion | no, it changed only the evidence method and evidence semantics | attempt three was void; its retained values were not used for closure, and attempt six reran the unchanged candidates under the clarified criterion |
| `aee262e` | attempt four used the wrong driver and was interrupted | no, it changed chronology and command/output registration only | the partial attempt-four finding is unused; attempt five reran with unchanged outcome expectations and candidates |
| `fe45c95` | attempt five's snapshot cadence was plan-level, not individually executed per simulator child | no, it refroze the implemented plan-boundary evidence guarantee and a fresh output path only | attempt five is void; attempt six reran unchanged candidates and reproduced its complete observation structures exactly |
| this closure commit | the byte-exact study harness, results, module status and task ledger | no | no simulator input or modeled number changed |

There was no post-observation execution-graph, transport or other modeled
behavior change. The backend patch makes the comparator authority agree with
independent authored-plan and physical arguments; it does not make the
simulator generate a desired number.

## Provenance and artifact integrity

The valid attempt-six study observed backend commit
`1ae1215758b96a52c1709a538204f5a73a05c5a9`. That commit and the
authored-against repair commit `1f2c124c9738edcfa0f6044b4667c230e75a542c`
have the same tree and parent. All eight validation plans are byte-identical
from the bound-authorship commit through that observed commit. The source was
a clean detached worktree at exact `1ae1215`. A 24-row manifest reconciles the
eight plans, 15 unique matrices or topologies and `validate.py` to their Git
blobs. The projected plans change only binary, matrix, topology and external
log path bodies, plus one log row after each Binary. Independent before/after
audits reconstruct every source byte. Final-newline state is preserved: it is
absent only for the sender load-balancing and connection-reuse plans.

Both candidate binaries are Release builds made with GCC 12.2.1. The
authorship binary's CMake source is the independently audited print-projection
tree. The pre-change binary's CMake source is the live backend checkout: its
reflog placed HEAD at exact `1ae1215` before the build, and the next backend
commit changed only seven validation plans, not compiled source. The measured
plans and all referenced inputs came from the separate clean detached evidence
worktree. Commit `fe45c95` explicitly refroze the implemented audit cadence:
once immediately before each plan/comparator invocation, which temporally
precedes all of that plan's simulator children, and again after the comparator
returns. The measured run followed all 16 plan invocations and all 32 boundary
audits in order. It does not
rehash separately between individual simulator children and does not claim to
detect a transient change restored before comparator return. The after-boundary
audit confirms that no change persisted through that return.

The final gate used an archive of exact backend commit `9800ea3`, not the
backend worktree's ignored binary link. Its `htsim_uec` was built with GCC
12.2.1 and `ENABLE_TESTS=ON` from parent `1ae1215`, before the bound-only
commit. The exact `1ae1215..9800ea3` diff changes only seven validation plans
and no compiled source, so this is the compiled-code binary for `9800ea3`.
The backend worktree was clean after both local commits. The SimLLM
`third_party/htsim` pin was not moved.

| artifact | SHA-256 |
|---|---|
| valid attempt-six summary | `0b91d5559c30de531e040c041f057e23ebe52f6fb4b8f5d891174af0a42c78a8` |
| attempt-six prefreeze and postfreeze check-only record | `3892875177110e0d7f2f84e519ad0b3f217d2bf2b837253ab071a327b11b6bb7` |
| attempt-six driver stdout | `81fc67fdd24db46b7b8a0ea433cb6e48132ea9ba39b6dbbf3ad618fc3b526495` |
| attempt-six driver stderr, empty | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| attempt-six raw-status record | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| attempt-six live-driver provenance | `80de81929e17993699cd759c5c4d698505eaf5561fbdc8ccf6a9ceeed7fb00e9` |
| study harness | `3615a44713367d574062edc2da68eda413e12594f9ff786c08ede55323183ce2` |
| authorship print-projection binary | `d45a2d2c1af47e171fd036fca511b5233cb74a3e65126067eee0a32767e83e0d` |
| pre-change binary | `f243bb383a0141666abc9bdca37cf85ab4857ddeb77d4b0d30f8f4a1c2262247` |
| final gate binary | `4b5e703c761c83a68dce645845007d5c8adb70998af2aa15c1a98680c6ef8bb1` |
| final `commit_check.sh` | `ed555486ea5830ed88dc3226e989018d8ea06e10e207824e5e77217cc79fe64a` |
| final `validate.py` | `f57a51011a44a7b1e419ee060152a023861791c5519853e210f4881b9dfbb42b` |
| full gate log | `73f3dce11d73a37c0af4adaf4931d995541cd1ee11b61215b8514c8a07e77333` |
| immediate full-gate status | `53b1ad529e7394fa66b712fd5ac2ab571dd7f362a1caa8482729ebe70f60a689` |
| immediate full-gate process provenance | `9b1579ed5ffec981b5d77e6507293f122c734f20c0f7d76a9545880738c6269f` |
| mutant gate log | `187396cc48fe31bf40d2322831c2341a83d961da77f4627313e526e077995f8b` |
| mutant raw-status record | `0f02e2a540787aa58bf32052c964b3d934612c8ec1ba8da3b76e00740cd09634` |
| restored gate log | `af037f9d6d097d1ae3c17a553c7036b8fe9e8c738cb8f705eded829e275765cd` |
| restored raw-status record | `17d07168b51909f56da457365f99fd186a49dd3c8a357b44932925b6fa3e93ef` |
| exact-restoration record | `6b98e043d8faf31f3d0547ac2af7b5c6e1994c9008bee37069ac81b141ef02ae` |
| final native-test log | `88a93ad4356d8da511b0bc91b11905bbf3c012e08de3fd257a22949898d879cb` |

Raw artifacts remain outside Git under
`${SIMLLM_DATA_ROOT}/attempt6-baseline/` and
`${SIMLLM_DATA_ROOT}/gate-evidence-9800ea3/`. The complete wave output is
1.8 GB, below the stated few-GB limit; attempt six itself is 2.3 MB, and no
retained per-packet trace was needed. A live process capture independently
records the literal `.venv/bin/python` command, CPython 3.12.12 executable and
full attempt-six arguments.

## Physical sanity before the measurements

The serialization floors were frozen before the first measured value. Crossing
one would have voided the run. Later review found that the first three freezes
mislabeled each plan's `-end` literal as microseconds: `main_uec.cpp` passes it
through `timeFromMs`, so the actual observation horizons are 1,000 times those
numeric literals. Commit `f832a7c` disclosed and refroze the source-derived
unit correction before attempts four through six. To avoid result-dependent
loosening, it retained every original numeric literal as a separate, tighter
fatal cap in microseconds. Every modeled FCT reported below as a claim in this
public result satisfies its floor, tighter legacy cap and corrected model
horizon.

| quantity | first-principles floor | retained legacy cap | corrected model ceiling | observed authorship values | result |
|---|---:|---:|---:|---:|---|
| 1023 x 100 KB incast into 100 Gbit/s | `1023 x 100,000 x 8 / 100e9 = 8,184 us` | 10,000 or 15,000 us | 10,000,000 or 15,000,000 us | 10,556.2, 8,902.25, 8,802.62 us | inside |
| 2,500,000 bytes delivered by the 3-to-1 matrix | `2,500,000 x 8 / 100e9 = 200 us` | 1,000 us | 1,000,000 us | 231.768, 217.096, 217.096 us | inside |
| each 2 MB permutation flow on a healthy 100 Gbit/s link | `2,000,000 x 8 / 100e9 = 160 us` | each experiment's 400 to 3,000 us cap | each experiment's 400,000 to 3,000,000 us horizon | 62 observed spreads cover 167.312 to 1,479.39 us | inside |
| outcast `Uec_1_0`, 2 MB at 100 Gbit/s | 160 us | 3,000 us | 3,000,000 us | 208.019, 235.068, 200.108 us | inside |

A 2 MB flow serialized wholly on a configured 25 Gbit/s degraded link has a
separate conditional floor of 640 us. It is not a universal floor for every
failed-link experiment because a particular realized flow may not traverse a
degraded link. It is the correct slowest eligible serialization term for an
authority that must bound the maximum per-flow FCT.

The same quantities in the pre-change candidate and current full-gate run
remain in range. The three 1024-incast tails are 10,563.2, 8,901.87 and
8,800.48 us; the three 3-to-1 tails are unchanged; and all observed permutation
spreads remain above 160 us and below their horizons. Hundreds of microseconds
for the individual 2 MB transfers and roughly 10 ms for the 1024-way incast are
plausible against a 100 Gbit/s NIC. This is a simulator validation gate, not an
absolute prediction for a deployed collective library.

## C1: all 17 cases classified one at a time

The authorship column is the valid attempt-six print projection. The current
column is the attempt-six pre-change binary and is also the raw value produced
after the bound-only backend commit. `F` denotes the independent failed-link
physical rule described below.

| # | plan and experiment | authored bound, us | authorship FCT, us | current FCT, us | new bound, us | cause and rule | classification |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | sender, 1024 node incast | 8,950 | 10,556.2 | 10,563.2 | 10,615 | receiver-only matched plan | wrong-at-authorship |
| 2 | sender, 3-to-1 incast | 220 | 231.768 | 231.768 | 235 | receiver-only matched plan | wrong-at-authorship |
| 3 | sender, small permutation with one failed link | 210 | 238.218 | 238.218 | 780 | F | wrong-at-authorship |
| 4 | receiver, outcast `Uec_1_0` | 215 | 235.068 | 235.068 | 245 | sender-only matched plan | wrong-at-authorship |
| 5 | receiver, small permutation with one failed link | 210 | 229.764 | 229.764 | 780 | F | wrong-at-authorship |
| 6 | both CCs, small permutation with one failed link | 210 | 238.253 | 238.253 | 780 | F | wrong-at-authorship |
| 7 | sender LB, oblivious, `q = 500` | 220 | 221.895 | 221.895 | 230 | auto-queue matched plan | wrong-at-authorship |
| 8 | receiver LB, oblivious, `q = 500` | 220 | 225.113 | 225.113 | 240 | auto-queue matched plan | wrong-at-authorship |
| 9 | failed sender LB, oblivious, auto queue | 220 | 280.775 | 280.775 | 780 | F | wrong-at-authorship |
| 10 | failed sender LB, oblivious, `q = 500` | 220 | 564.701 | 564.701 | 780 | F | wrong-at-authorship |
| 11 | failed sender LB, bitmap, auto queue | 220 | 220.269 | 220.269 | 780 | F | wrong-at-authorship |
| 12 | failed sender LB, bitmap, `q = 500` | 220 | 478.668 | 478.668 | 780 | F | wrong-at-authorship |
| 13 | failed sender LB, REPS, 4 SACK | 220 | 245.821 | 243.257 | 780 | F | wrong-at-authorship |
| 14 | failed receiver LB, oblivious, auto queue | 220 | 224.755 | 224.755 | 780 | F | wrong-at-authorship |
| 15 | failed receiver LB, oblivious, `q = 500` | 220 | 374.592 | 374.592 | 780 | F | wrong-at-authorship |
| 16 | failed receiver LB, bitmap, `q = 500` | 220 | 356.935 | 356.935 | 780 | F | wrong-at-authorship |
| 17 | failed receiver LB, REPS, 1 SACK, `q = 500` | 220 | 338.083 | 333.29 | 780 | F | wrong-at-authorship |

Classification counts are 17 wrong-at-authorship, zero
stale-after-intended-change, zero simulator regressions and zero unresolved.
C1 therefore passes `17/17` genuine-risk instances. Each causal classification
could have been refuted by a passing authorship result, an unexplained onset or
a physical or paired-relation failure. No earlier fatal oracle pins the causal
class.

### Five matched-plan derivations

The first applicable frozen rule preserves the matched authored plan's
fractional slack and rounds upward to 5 us:

| case | matched authored operands | calculation | authority |
|---:|---|---|---:|
| 1 | receiver-only tail 8,902.25 under bound 8,950; sender target 10,556.2 | `ceil5(10,556.2 x 8,950 / 8,902.25) = ceil5(10,612.82)` | 10,615 us |
| 2 | receiver-only tail 217.096 under bound 220; sender target 231.768 | `ceil5(231.768 x 220 / 217.096) = ceil5(234.868)` | 235 us |
| 4 | sender-only `Uec_1_0` 208.019 under bound 215; receiver target 235.068 | `ceil5(235.068 x 215 / 208.019) = ceil5(242.957)` | 245 us |
| 7 | sender auto-queue tail 212.794 under bound 220; `q = 500` target 221.895 | `ceil5(221.895 x 220 / 212.794) = ceil5(229.409)` | 230 us |
| 8 | receiver auto-queue tail 208.781 under bound 220; `q = 500` target 225.113 | `ceil5(225.113 x 220 / 208.781) = ceil5(237.210)` | 240 us |

For case 1, the two plans declare 10,000 and 15,000 millisecond observation
cutoffs. Source inspection shows `-end` is passed through `timeFromMs`; both
10 s and 15 s cutoffs are inert for 8.9 ms and 10.6 ms completions. Congestion
control ownership is the only active modeled resource difference. The
allowance for this inert non-resource directive was post-specified in
`f832a7c`, then rerun in valid attempt six, as disclosed above.

### Failed-link derivation and coherent family

The authored 220 us failed-link authorities were wrong for two independent
reasons, not because current code happened to miss them:

1. `_failed_link_ratio` is 0.25 in the executable topology, and the generator
   independently uses `failed / 4`. Both tracked failed-link plan labels said
   10 percent. The labels, not the mechanism, were wrong.
2. The generator adjusts aggregate fabric capacity while `validate.py` checks
   the maximum individual-flow FCT. Aggregate capacity cannot bound a tail
   whose slowest eligible path is a 25 Gbit/s link.

The frozen physical rule gives
`ceil5(1.20 x (2,000,000 x 8 / 25e9 + 9 us)) =
ceil5(778.8 us) = 780 us`. It is independent of every measured target value.

Twelve target cases use that rule. A target-only edit would have left twelve
other default-gate failed-link experiments under the same invalid authority.
The backend commit therefore changes all 24 members of that exact family and
all 18 false capacity labels. The twelve coherent siblings are not additional
scored cases. Apart from the two matched healthy `q = 500` authorities in
cases 7 and 8, every other healthy permutation authority remains
byte-unchanged.

## P1: paired scaling support, 11/11 genuine risk

These behavioral relations are evaluated on raw authorship tails and remain
separate from C1:

| relation | observed pair, us | result |
|---|---:|---|
| sender small failed link >= healthy | 238.218 >= 194.022 | PASS |
| receiver small failed link >= healthy | 229.764 >= 195.673 | PASS |
| both-CC small failed link >= healthy | 238.253 >= 196.774 | PASS |
| sender failed-8 oblivious auto >= healthy | 280.775 >= 212.794 | PASS |
| sender failed-8 oblivious `q = 500` >= healthy | 564.701 >= 221.895 | PASS |
| receiver failed-8 oblivious auto >= healthy | 224.755 >= 208.781 | PASS |
| receiver failed-8 oblivious `q = 500` >= healthy | 374.592 >= 225.113 | PASS |
| healthy sender `q = 500` >= auto | 221.895 >= 212.794 | PASS |
| failed sender `q = 500` >= auto | 564.701 >= 280.775 | PASS |
| healthy receiver `q = 500` >= auto | 225.113 >= 208.781 | PASS |
| failed receiver `q = 500` >= auto | 374.592 >= 224.755 | PASS |

P1 passes `11/11` genuine-risk instances. These are direction checks on
distinct raw measurements, not exact oracles and not by-construction guards.

## G1: the final default gate is green

The real `commit_check.sh` from exact backend commit `9800ea3` was invoked
with no plan argument. After mutant restoration, a fresh repeat wrote its raw
return code immediately after the process with no intervening command, before
post-run acceptance interpretation. A live process capture records the exact
wrapper ordering. Its complete log is byte-identical to the initial clean
pre-mutant no-argument gate log.

| evidence | observed |
|---|---:|
| raw gate status | 0 |
| default plans completed | 8 |
| experiments completed | 95 |
| connection-count passes | 95 |
| packet summaries present | 95 |
| failure lines | 0 |

G1 passes `1/1` genuine-risk instance. Per-experiment final passes are not
scored again because the authoritative full-gate status already entails them.

## M1: the gate rejects a deliberate mutant

After the initial clean no-argument positive control, the first `!tailFCT 18`
in the external committed-tree copy of `validate_uec_connreuse.txt` was changed
to `!tailFCT 5`. Gate output was redirected while the raw return code was
recorded, so the status was known before its diagnostic was read. The fresh G1
repeat above then reconfirmed the full final state after restoration.

| step | raw gate status | observation |
|---|---:|---|
| deliberate mutant | 1 | first tail exceeded the 5 us mutant bound; 1 of 12 experiments failed |
| byte-exact restoration | not a score | SHA-256 restored to `34d7f8761782685c1bd08b3f4fd3a2719fd92d4abc14e955b80143a4cb48d54c` |
| restored positive control | 0 | 12 of 12 experiments passed |

The first restoration attempt was stopped by the byte-identity guard because
the edit tool had added a final LF to a tracked file that has none. No positive
control ran from that nonidentical copy. Removing only that LF restored the
exact committed digest, after which the positive control ran and passed. The
backend worktree itself was never mutated and is clean.

M1 passes `1/1` genuine-risk instance. The mutant contents are author-defined
and unscored; propagation of the validator failure through the real gate is
the live behavior at risk.

## Fatal guards and entailment

All fatal guards held in the valid attempt-six and final-gate evidence. The
exact plan registry and target identities held. The 24 source/input/comparator
authorities and all 16 measured plan projections reconcile byte-for-byte; the
two missing final LFs stay missing. The print-only source projection is exact.
All 190 simulator children exited zero, which the common comparator establishes
before it parses a complete observation; all 190 observations have a passing
connection count and packet summary, and all 16 captured comparator-process
stderr artifacts are empty. Successful simulator-child stderr is not retained
separately. Seven plan comparator processes per candidate return one because
they correctly reject the authored FCT bounds; connection reuse returns zero.
Those expected plan statuses are not simulator-child failures. Every modeled
FCT reported as a claim in this public result is covered by the pre-stated
physical table: C1 and its derivations map to the 1024-incast, 3-to-1, 2 MB
permutation or outcast rows; all P1 pairs are 2 MB permutations. Each is
inside its frozen floor, legacy cap and corrected horizon. The remaining raw
summary values are retained diagnostics, not modeled numbers claimed in this
report; as an additional broad audit, all 576
tail, spread and named-per-flow readings across both candidates are below their
experiment's legacy cap and corrected horizon. Every target has exactly one
observed decision row, and the deliberate mutant is absent from the final tree.
These guards are unscored. A single violation would have voided the run rather
than becoming a lost point.

Causal ambiguity is a C1 failure that keeps HTSIM-25 open, not a fatal oracle;
the structural guard supplies its 17 inputs without pinning their classes. C1
classifies causal provenance and does not force a green final gate. P1 compares
raw tails and is not entailed by either classification or gate status.
G1 is the sole scored full-gate acceptance relation. M1 uses a different input
and reads raw failure status before diagnostics, so G1 does not entail it.
The four evidence classes are not combined into one headline fraction.

## Native verification

The backend external build completes, and CTest passes 350 of 350 native
tests. The final SimLLM closure change passes `ruff check .`, the full pytest
suite with 1,430 passed and 7 skipped, the module-doc format checker, the
task-progress ledger tests and the generated task-progress checker.

## Mission error budget

This task changes validation authorities, not the timing model. Every dominant
term named by `end_to_end_replay_v1` is unchanged:

| budget term | before | after |
|---|---:|---:|
| fixed per-step host cost | 0 ps, `initiation_delay_ps = 0`, profile `ideal` | unchanged at 0 ps |
| collective latency floor | 2.000 us per collective | unchanged at 2.000 us |
| compute calibration | flat 0.7 derate, omitted `gpu=` selects B100 | unchanged |
| composed case-A decode step | 0.205 ms | unchanged at 0.205 ms |

The composed plausible budget remains 1.1 to 4.5 ms and the simulated
per-request decode latency remains roughly 5x to 22x optimistic. This task has
no TTFT or TPOT relation of its own and does not invent one.

## Closure scope, task by task

| task | registered acceptance clause | evidence | decision |
|---|---|---|---|
| HTSIM-25 | “reconcile the authored UEC validation bounds with current backend behavior” | exact authorship and current runs, the 17-row C1 table, five matched derivations, the physical failed-link derivation and coherent family audit | CLOSED |
| HTSIM-25 | “Decide per experiment, with evidence, whether the transport regressed or the authored bound is stale. Never relax a bound to match an observation without that evidence.” | all 17 are wrong-at-authorship; zero are stale, regressed or unresolved; every new authority names an independent frozen rule and operands | CLOSED |
| HTSIM-8 | “make the backend `commit_check.sh` gate citable as release evidence” | G1 raw zero over the exact committed eight-plan gate plus M1 raw rejection and restored positive control | CLOSED |
| HTSIM-8 | “This closes when HTSIM-25 resolves the bound drift and the full default gate exits zero.” | HTSIM-25 closes above; the no-argument gate returns zero over all 95 experiments | CLOSED |

Each P0 closes on its own registered clauses. No strong result is carried from
one task to excuse a weak result on the other.

Zero new IDs were registered. Every registered clause was demonstrated, so
none may move to HTSIM-26, HTSIM-27 or BACK-42. The generator's generic
aggregate-capacity estimator is an adjacent improvement idea, not an
undemonstrated registered clause; it is therefore recorded here without a new
task under residual discipline.

## Contradiction sweep

Hits are reported rather than silently repaired:

- `README.md`: no closure-specific contradiction.
- `docs/README_PRO.md:543`: the backend status still says the gate “rejects
  the backend checkout on 17 of 95 authored bounds, tracked as HTSIM-25”. That
  sentence is stale after this closure. The mechanically generated progress
  block and open-count cell are reconciled, but the narrative sentence is not
  edited by this task.
- `docs/architecture.md`: no closure-specific contradiction.

## Reproduction

The frozen authorship and pre-change inventory is reproduced with:

```bash
HTSIM_SOURCE_ROOT="${HTSIM_PRECHANGE_SOURCE_ROOT:?configure a detached HTSIM worktree at 1ae1215}" \
.venv/bin/python examples/htsim_uec_bounds_v1/run_study.py \
  --out "${SIMLLM_DATA_ROOT:?configure the data root}/attempt6-baseline" \
  --candidate "authorship-print-projection=${HTSIM_AUTHORSHIP_BINARY:?build the print-only authorship projection}" \
  --candidate "prechange=${HTSIM_PRECHANGE_BINARY:?build the pre-change binary}"
```

Add `--check-only` to validate the frozen registry and command shape without
creating artifacts or reading a measurement. `HTSIM_PRECHANGE_SOURCE_ROOT`
must name a clean detached worktree at exact `1ae1215`, because the frozen
target identities include the pre-correction 10 percent labels. The harness
also verifies every plan, referenced input and comparator blob at each
plan/comparator boundary, immediately before invocation and again after it
returns. The pre-invocation audit precedes all simulator children in that plan;
there is no separate rehash between children and no claim about a transient
change restored before return. The external evidence retains the CMake
configuration, raw gate logs, raw statuses and digests. The final acceptance
command is the backend
`htsim/sim/datacenter/commit_check.sh` with no arguments.
