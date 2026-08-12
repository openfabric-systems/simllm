# HTSIM-8 backend validation gate results

Run on 2026-08-13. The three code defects named by HTSIM-8 are repaired in the
backend repository and the repaired gate is proved able to reject. The scored
rejection family passes `2/2` genuine-risk instances and the fatal positive
control holds, so the rejection evidence is valid.

The frozen native acceptance run is refuted. On its first honest execution the
repaired gate rejects the backend checkout it was repaired on: 17 of 95
experiments miss their authored FCT bounds, and 7 of the 8 default plans fail.
Under the freeze as written that makes the repair incomplete, so HTSIM-8 stays
open with its remaining scope narrowed to that acceptance. The exposed bound
drift is registered separately as HTSIM-25.

## Chronology and provenance

The expectations-only commit is
`522f1fdc7830fd378b15cc9177b764b299d21fec`. It precedes every backend edit
kept in this study and every result-producing command. Both registered command
shapes ran with `--check-only` before it.

The evidence was authored against HTSIM commit
`fc4400e4ca619223481536632074045cb6af2756`. The repair landed in the backend
repository at `1f2c124c9738edcfa0f6044b4667c230e75a542c` on branch
`codex/htsim8_commit_check`. The study ran twice with byte-identical gate
scripts: once with the repair uncommitted (`observed_htsim_commit` recorded as
`fc4400e4`) and once after it landed (`observed_htsim_commit` recorded as
`1f2c124c`). Both runs passed `2/2`. No equality with any live
`third_party/htsim` pin is asserted, and the pin was not moved.

| artifact | SHA-256 |
|---|---|
| first control summary | `7f453c6d8fe117963b9fbef79e8cb75b3d2a1181a730f8ebf44648e478bfaab9` |
| landed control summary | `ae646e44f8034918c38994c3a5463aba57a3411c3c032fd2faf8eb1673ac10e5` |
| `commit_check.sh` as exercised | `ed555486ea5830ed88dc3226e989018d8ea06e10e207824e5e77217cc79fe64a` |
| `validate.py` as exercised | `f57a51011a44a7b1e419ee060152a023861791c5519853e210f4881b9dfbb42b` |
| `htsim_uec` as exercised | `2f3d9ad9f802c5ad8ecf100dd5d118b7405c8ccece6da6e6eec8424772399be0` |

External artifacts stay outside Git under
`${SIMLLM_DATA_ROOT}/htsim_commit_gate_v1/`,
`${SIMLLM_DATA_ROOT}/htsim_commit_gate_v1-landed/` and
`${SIMLLM_DATA_ROOT}/plan_*.log`. The host is Linux x86-64 with Python 3.10 for
the gate and Python 3.12 for the harness.

## What was broken, and the root cause of the zero division

`validate.py` initialized both FCT extrema to zero, left them there when no
completion row was parsed, and then divided them. The reason no completion row
was ever parsed is exact and confirmed by git: commit
`bf83fa2150c95faf794eb8384b347be7ad796730` ("Integrate Spritz source routing
artifact") put the UEC flow-completion prints behind the
`HTSIM_TRACE_FLOW_COMPLETIONS` environment flag, and `commit_check.sh` never
set it. Every experiment after that commit produced zero completion lines, so
every experiment reached the division. The same commit inserted `flowId <id>`
into the completion line, which shifted the positional token indices the old
parser relied on, so even with the flag set the old parser would have raised on
`float('total')`.

`commit_check.sh` then discarded each failure: it ran `validate.py`, `git show`
and `check_regressions.py` in sequence without `set -e` and without testing any
status, so the loop kept only its final command's status and the script exited
zero. `check_regressions.py` additionally exits successfully when a baseline
file is absent, and no baseline was ever present.

## Baseline decision: the comparison is removed

`git ls-tree -r fc4400e4 -- htsim/sim/datacenter/validate_outputs` is empty and
`.gitignore` ignores `*.out`, so there is no checked-in baseline authority. The
repair removes the historical output comparison and the remote fetch outright
rather than adding baselines. Generating baselines now would mean deriving them
from the very code whose behavior is in question, after the failure was
discovered, which encodes the current numbers as correct by definition. The
authored absolute FCT and completion-count bounds already present in every
validation plan are now the only comparison authority, and they are enforced.
`check_regressions.py` stays in the tree as a standalone tool for hand-driven
comparisons and is no longer wired into the gate.

The gate now also exports `HTSIM_TRACE_FLOW_COMPLETIONS=1`, without which its
authored FCT checks have no input at all.

## F1: positive control, fatal and unscored

| check | result |
|---|---|
| real gate exits zero on the passing fixture | PASS |
| gate reports one completed connection | PASS |

F1 holds in both runs, so the rejection results below are interpretable rather
than the output of a gate that rejects everything.

## R1: raw gate rejection family, 2/2 genuine risk

Each row invokes the real `commit_check.sh` entry point on one temporary plan
and records the raw gate return code before any diagnostic text is parsed.

| fixture | gate return code | predicates | result |
|---|---:|---|---|
| `child-exit-defect` | 1 | nonzero status, child status 23 named | PASS |
| `zero-completion-defect` | 1 | nonzero status, empty case named, no `ZeroDivisionError`, no traceback | PASS |

Genuine-risk fraction: `2/2`.

The `zero-completion-defect` row is the direct regression lock on the original
division: the gate now prints `zero flows completed while 1 were expected` and
exits nonzero instead of raising `ZeroDivisionError`.

### Entailment analysis

Each raw gate return code is evaluated before its explanatory text. No earlier
exact oracle pins those return codes. The positive control constrains a
different plan and does not entail either rejection. The fixture inputs are
fixed by construction, but propagation of a child failure through the shell
loop and of an empty completion set through the validator is live behavior and
was exactly the defect under test, so both instances are genuine risk rather
than by-construction guards. The fixture removal after the run is a cleanup
step, not evidence.

## Second rejection proof on a tracked plan

The fixtures are author-written, so the failure path was also exercised on real
tracked content. `validate_uec_connreuse.txt` is the one default plan that
passes clean at the landed commit.

| step | gate command | gate exit | gate output |
|---|---|---:|---|
| clean | `commit_check.sh validate_uec_connreuse.txt` | 0 | `[PASS] 12 experiments passed` |
| defect: first `!tailFCT` tightened from `18` to `5` | same | 1 | `[FAIL] Tail FCT 17.0107 us is above the target of 5.0 us`, `[FAIL] 1 of 12 experiments failed`, `[FAIL]   Single Flow, w/o Conn Reuse, NSCC` |
| defect removed with `git checkout --`, tree clean | same | 0 | `[PASS] 12 experiments passed` |

No deliberate defect remains. `git status` shows the plan file unmodified, and
the landed commit contains no change to any validation plan.

## Native acceptance run, frozen support class: REFUTED

The default gate stops at its first failing plan, so each of the eight plans
was also run individually to get the complete inventory. All 95 experiments in
the eight default plans were attempted.

| plan | experiments | failed |
|---|---:|---:|
| `validate_uec_sender.txt` | 15 | 3 |
| `validate_uec_rcv.txt` | 15 | 2 |
| `validate_uec_both.txt` | 15 | 1 |
| `validate_load_balancing_snd.txt` | 10 | 1 |
| `validate_load_balancing_rcv.txt` | 10 | 1 |
| `validate_load_balancing_failed_snd.txt` | 9 | 5 |
| `validate_load_balancing_failed_rcv.txt` | 9 | 4 |
| `validate_uec_connreuse.txt` | 12 | 0 |
| total | 95 | 17 |

Every failure is a missed FCT bound. No experiment failed on completion count,
missing input, simulator status or malformed output, so the transport does
finish all declared connections everywhere; it is only slower than the authored
targets.

| plan | experiment | bound, us | observed, us | over bound |
|---|---|---:|---:|---:|
| sender | 1024 node incast | 8950 | 10563.2 | +18.0% |
| sender | 3 to 1 incast with long running flow | 220 | 231.768 | +5.3% |
| sender | Small permutation, INC (16 nodes) | 210 | 238.218 | +13.4% |
| rcv | outcast incast, per-flow `Uec_1_0` | 215 | 235.068 | +9.3% |
| rcv | Small permutation, INC (16 nodes) | 210 | 229.764 | +9.4% |
| both | Small permutation, INC (16 nodes) | 210 | 238.253 | +13.5% |
| lb snd | Large permutation, oblivious, large queues | 220 | 221.895 | +0.9% |
| lb rcv | Large permutation, oblivious, large queues | 220 | 225.113 | +2.3% |
| lb failed snd | oblivious, 8 core links at 10% | 220 | 280.775 | +27.6% |
| lb failed snd | oblivious, large queues, 8 core links at 10% | 220 | 564.701 | +156.7% |
| lb failed snd | bitmap, 8 core links at 10% | 220 | 220.269 | +0.1% |
| lb failed snd | bitmap, large queues, 8 core links at 10% | 220 | 478.668 | +117.6% |
| lb failed snd | REPS 4SACK, 8 core links at 10% | 220 | 243.257 | +10.6% |
| lb failed rcv | oblivious, 8 core links at 10% | 220 | 224.755 | +2.2% |
| lb failed rcv | oblivious, large queues, 8 core links at 10% | 220 | 374.592 | +70.3% |
| lb failed rcv | bitmap, large queues, 8 core links at 10% | 220 | 356.935 | +62.2% |
| lb failed rcv | REPS 1SACK, large queues, 8 core links at 10% | 220 | 333.29 | +51.5% |

### Physical sanity of the rejections

Before reading whether the digits agree, each rejected number is placed against
a first-principles floor. The default host NIC is 100 Gbit/s
(`htsim/sim/datacenter/main.h`).

- 1024 node incast: 1023 flows of 100,000 bytes all terminate on one receiver,
  so the receiver serialization floor is `102,300,000 x 8 / 100e9 = 8184 us`.
  The authored bound of 8950 us allows 9.4% above that floor; the observed
  10563.2 us sits 29.1% above it.
- 3 to 1 incast with long running flow: 2,500,000 bytes terminate on node 0, a
  200 us receiver floor. The bound allows 10% above it; the observation is
  15.9% above it.
- 16-node and 1024-node permutations: each flow is 2,000,000 bytes, a 160 us
  per-flow serialization floor. The 210 and 220 us bounds allow 31% and 37.5%
  above it. The worst failed-link observation, 564.701 us, is 253% above it,
  which is still physically reachable because 8 core links run at 10% capacity
  and a flow pinned to one of them would need 1600 us.

Every observation is above its own floor and below the trivially unreachable
regime, so none is physically impossible. The gate is therefore reporting a
genuine behavior-versus-bound gap, not a modeling impossibility or a harness
artifact.

Two further checks argue the same way. First, the tail statistic is not the
cause: for the 3 to 1 incast the ordered completions are 72.5312, 73.5926 and
231.768 us, so the last completion and the maximum agree and the old
last-completion reading would have produced the same number. Second, the
failure pattern is structured rather than random: `Small permutation, INC (16
nodes)` fails in all three congestion-control modes with nearly identical
values (238.218, 229.764, 238.253), and 9 of the 17 failures are in the two
`-failed 8` load-balancing plans where the degraded-link path dominates. A
harness bug would not concentrate this way.

## Closure scope

| registered acceptance clause | evidence | status |
|---|---|---|
| "Add checked-in baselines or remove that compare" | Source audit, removal of the absent-baseline comparison and the remote fetch, README statement of the new comparison authority | DEMONSTRATED |
| "fix zero-flow diagnostics" | `zero-completion-defect` rejects with the empty-case diagnostic, no exception and no traceback | DEMONSTRATED |
| "make every failed command fail the gate" | `child-exit-defect`, the tracked-plan defect and its removal, the F1 positive control, and the real rejection of the backend checkout | DEMONSTRATED |
| frozen support clause: the default gate runs its eight tracked plans and exits zero | 17 of 95 experiments miss their authored bounds; 7 of 8 plans fail | REFUTED |

HTSIM-8 therefore stays open. Its three registered code clauses are closed by
evidence and its entry now carries only the acceptance residual. The bound
reconciliation that acceptance depends on is registered as HTSIM-25, because it
is transport and bound-authorship work rather than validation-infrastructure
work.

Refuting the frozen support expectation is the useful outcome here. A gate that
had passed on the first honest run would have proved much less than one that
immediately found 17 real bound misses that the false success had been hiding
since `bf83fa2`.

## Contradiction sweep

`README.md`, `docs/README_PRO.md` and `docs/architecture.md` make no claim about
the backend `commit_check.sh` gate, about the UEC validation plans, or about
backend release-gate status, so nothing in them contradicts this result. No edit
was made to those files.
