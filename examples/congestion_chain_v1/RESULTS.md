# Congestion-bearing live chain results

BACK-38 and BRIDGE-2 remain open. The registered check-only gate passed and
produced no artifacts, but no result-producing execution ran. The behavioral
report is therefore `0/0, blocked before behavioral execution`, not a void
run. No `rnic-cn` TTFT, TPOT, process-count or wall-time treatment value was
measured.

Two independent preconditions block behavioral implementation. First, the
required HTSIM-8 plus HTSIM-25 backend gate has not produced a citable full
default eight-plan zero exit. Second, the delivered HTSIM-18 flow-session
protocol cannot release an online dependent artifact at the exact completion
time it discovers. This second blocker requires a backend protocol change.
The task contract assigns backend source work to the backend gate branch and
requires this branch to stop at the SimLLM side, so no backend source or
submodule pin was changed.

## Chronology and provenance

The original expectations-only commit is
`b666bdc89e6ef1dcd14713a1c1ae28cb6f49239d`. Its registered command passed
with `--check-only` before the commit. It validated frozen source hashes,
counts, arithmetic, cells, relations and evidence classes; it created no
output directory and did not invoke a backend session.

Commit `a60d031c5ce967698b50de709d3a76f611640231` then froze the treatment's
causal release recurrence after a source-only audit showed that all 35
adjacent source records release exactly at the preceding fluid completion.
That commit preceded every modeled measurement and every backend session
invocation. The amended check-only command also passed without producing an
artifact.

There was no first measured run. Consequently no commit landed after a first
measured run, no post-measurement defect fix changed modeled behavior, and
there is no before-and-after treatment measurement to disclose. The causal
release amendment did change the semantics planned for the future treatment,
but it did so before any treatment value was observed: only
`virtual_time_ps` would become
`max(source_release, previous_treatment_completion)`.

The protocol audit observed HTSIM commit
`1f2c124c9738edcfa0f6044b4667c230e75a542c`. Its
`htsim/sim/datacenter/rnic_flow_session.cpp` is Git blob
`43e9a2b4e8cbc1549fb59bfc6ca7eeb06b88cfa6`. The same blob is present at the
SimLLM-pinned gitlink `fc4400e4ca619223481536632074045cb6af2756` and at the
current HTSIM tip audited on this date,
`9800ea3296ff4bb017ac4d60df94917e4c8c3f0d`. The limitation therefore holds at
the revision this repository ships and at the backend tip, rather than being
an artifact of an old checkout. The session-capable binary inspected without
invocation had SHA-256
`500c5afb8c29335d19bd7b77a166168c05d56f744fcf3d9df16be49ff2ee304f`.
The check-only command used the supplied executable with SHA-256
`32035c778e40e9b11dd32d081350a36a92872855a97dc4b5f217c634420c0816`;
check-only verifies only that the path is executable and never starts it.

The SimLLM `third_party/htsim` gitlink remains
`fc4400e4ca619223481536632074045cb6af2756`, exactly as it was at branch
creation. The submodule remained uninitialized.

## Backend gate precondition

The freeze requires evidence that HTSIM-8 and HTSIM-25 were green before
behavioral implementation and before the first session invocation. That
evidence does not exist in the locally available records.

The latest completed gate report remains
[`htsim_commit_gate_v1`](../htsim_commit_gate_v1/RESULTS.md). It records 17
authored FCT-bound misses among 95 experiments and failure of 7 of the 8
default plans. It leaves HTSIM-8 open behind HTSIM-25. The concurrently
prepared HTSIM-25 branch, observed at
`9301b94ac02b3a48d8d2540e4ecd353cba52b6b1` through 2026-08-13 12:37 CEST,
had not published a completed `RESULTS.md` or a full default eight-plan zero
exit when this branch stopped. No behavioral code was implemented while that
mandatory gate was unresolved.

## Exact online-boundary blocker

The static protocol audit covered all five accepted verbs, `open`, `inject`,
`advance`, `drain` and `close`, in the delivered HTSIM-18 implementation. The
relevant state transitions are:

| Backend behavior | Observed implementation |
|---|---|
| accept an injection | `eligible_at_ps` must be strictly greater than the last advanced horizon; equality and earlier time return `stale_eligibility` |
| advance | the caller supplies `through_ps`; the server stores that horizon, executes every event at or before it, then returns newly visible lifecycle rows |
| discover a completion | the completion becomes visible only after an advance whose horizon is at least the native completion time |
| drain | succeeds only after every accepted injection has fired and physical work is quiescent, then marks the session drained |
| inject after drain | returns `post_drain_inject` |

At the observed commit these checks are in
`htsim/sim/datacenter/rnic_flow_session.cpp:850-929`, the advance loop is at
lines 1099-1139, and terminal drain state is at lines 1170-1230.

Let `T` be the native completion boundary of artifact `i`, which is the exact
release required for dependent artifact `i + 1`. Let `H` be the caller-chosen
advance horizon. The client can observe `T` only after an advance with
`H >= T`. At that point the session accepts a later injection only when its
eligibility is greater than `H`. The exact eligibility `T` is therefore
illegal, including in the best case where the caller guessed `H = T`.

The protocol exposes neither the next event time nor an operation that runs
until a named accepted sequence completes. `drain` returns completion rows
only at terminal physical quiescence and prohibits the next injection. The
client therefore cannot learn a native completion and then use that same
timestamp as the next graph boundary while retaining the session.

The following apparent SimLLM-only workarounds are unsound:

- advancing beyond `T` and injecting after `H` adds model-dependent idle time,
  lets transport and congestion-control timers advance without the dependent
  traffic, and changes TTFT and TPOT;
- adding one picosecond after an exact completion changes the accepted graph's
  exact dependency and the existing sum-of-artifact-services semantics;
- pre-injecting the dependent artifact needs its unknown native release time
  and the protocol carries no dependency sequence;
- running a predictor or replay session to discover `T` creates a second
  timing authority and violates the one-authority contract;
- draining or restarting at every artifact discards the retained queue,
  transport, congestion-control and RNG state that BACK-38 requires.

The minimum sound backend extension is an atomic event-boundary operation.
For example, it may advance until a named accepted sequence set completes,
return the exact current simulator time, and permit a dependency-gated
injection at that same time without replaying already processed events.
Equivalently, the backend may accept explicit injection dependencies and
schedule them natively at predecessor completion. Either form must retain the
current fail-before-mutation checks, contiguous sequence contract and
one-session authority. A caller-selected polling quantum is not an equivalent
repair.

## What did not run

The tracked runner intentionally retains its expectations-only
`NotImplementedError`. No production session client, multi-artifact
`rnic-cn` enablement, completion translation, ledger append, transactional
publication or deterministic byte-lock fixture was added. In particular:

| Evidence class | Registered | Executed | Outcome |
|---|---:|---:|---|
| Treatment configurations | 3 | 0 | blocked before behavioral execution |
| Behavioral families | 4 | 0 | `0/0, blocked before behavioral execution` |
| Genuine-risk instances | 7 | 0 | `0/0, blocked before behavioral execution` |
| Fatal guard set | 1 set | 0 | not evaluated; no run exists to void |
| Exact-oracle reconciliation | per artifact and completion | 0 | not evaluated |
| Native or Python implementation tests | required after implementation | 0 | not applicable to an absent implementation |

The existing `rnic-cn` multi-artifact refusal remains unchanged and still
fails closed before backend execution. Diagnostic `rnic-nn` and
`rnic-nn-fluid` behavior was not edited. No claim is made for their off-path
identity beyond the fact that this branch changed no production code.

## Physical sanity and absolute-timescale budget

The expectations state the physical bounds before any new value is read: a
TTFT floor of 177,964,800 ps, an end-to-end ceiling of 250,000,000,000 ps, a
positive TPOT floor of 60,000,000 ps, and per-flow serialization floors of
`payload_bytes * 8 / link_rate`. No treatment value exists to place inside or
outside those bounds. The registered 100 to 200 Gbit/s scaling check also did
not execute.

The mission source remains the only wall observation: 1,728 isolated backend
processes over 36 steps and 600.23 seconds. There is no after value, so no
speedup or process-count reduction is claimed.

This branch moves none of the three dominant absolute-timescale terms. Fixed
host initiation remains 0 ps, the accepted fluid collective floor remains
2.000 us per collective, and compute remains the flat 0.7 roofline derate on
the default B100 envelope. The mission study's 5x to 22x optimism budget is
therefore unchanged. No congestion-controlled collective attribution or
recomposed per-request budget exists to report.

## Acceptance and closure map

BACK-38 clause B1 is:

> "preserve htsim topology, RNG, transport, congestion-control and RNIC state
> across ordered GOAL artifacts instead of starting a fresh process at every
> boundary."

Not demonstrated. No live session ran, and the exact dependent-artifact
release cannot be expressed by the delivered protocol.

BACK-38 clause B2 is:

> "Acceptance must execute one checked graph projection in a state-preserving
> session, reconcile every artifact and completion identity, and retain the
> current rejection and stateless-profile bytes as the explicit off paths."

Not demonstrated. No graph projection executed in a session, no returned row
was reconciled, and no post-implementation off-path byte comparison exists.
The pre-existing rejection remains in place but cannot carry the rest of the
clause.

BRIDGE-2 clause C1 is:

> "lower live `ExecutionGraph` dependencies into flow injections and inclusive
> virtual-time horizons, translate the returned native lifecycle projections
> into canonical `CompletionEvent` values, append the exact object, stage and
> completion facts at the supplied bookkeeping cursor, construct
> `ExecutionResult`, reduce the full `StepResult`, and publish only after all
> identities, cursors, timestamps and quiescence evidence validate."

Not demonstrated. The interface audit stops at the dependency-to-horizon
contradiction before a mutable client or publication path can be soundly
built.

BRIDGE-2 clause C2 is:

> "Reject loss, duplication, cursor disagreement, graph/event identity
> disagreement and timestamp regression before publishing a result."

Not demonstrated. No publication implementation or fault-injection suite was
added while the live execution contract was unavailable.

BACK-38 and BRIDGE-2 each remain open on their original full acceptance. No
ledger closure or module-status edit is made. Zero new residual IDs are
registered: no valid run split a demonstrated clause from an undemonstrated
remainder, so BACK-40, BACK-41 and BRIDGE-4 remain unused. The required
backend protocol capability is a prerequisite to the existing tasks, not an
adjacent SimLLM feature disguised as a new residual.

## Verification

Verification is separate from behavioral evidence and does not increase a
genuine-risk denominator.

- the registered check-only command passed and left its selected output path
  absent;
- `ruff check .` passed;
- `pytest -q` passed with 1,403 tests and 7 skips, without an initialized
  submodule;
- `python3 scripts/check_docs_format.py` passed for all 10 module documents;
- `git diff --check` passed.

## Contradiction sweep

The integrator-owned files were inspected and not edited. These statements
need reconciliation with the still-blocked live congestion chain:

- `README.md:17-28` says SimLLM predicts serving metrics with a packet-level
  network underneath and says congestion reshapes TTFT and TPOT. The accepted
  mission run used `rnic-nn-fluid`, and no congestion state yet reaches its
  per-request metrics.
- `README.md:63-76` depicts every closed-loop step flowing through the
  packet-level simulator and says congestion changes batching. The current
  multi-artifact `rnic-cn` path still rejects, so this is not a supported
  general closed-loop chain.
- `README.md:202-206` says the packet-level model reproduces congestion. That
  is true for standalone packet studies but, read with the broad serving
  claims above, does not disclose that persistent congestion is unreachable
  from the mission TTFT and TPOT chain.
- `docs/README_PRO.md:596` describes `end_to_end_replay_v1` as using the
  packet-level fabric. Its accepted cells used the explicit fluid bypass and
  spawned one fresh process per artifact.
- `docs/README_PRO.md:642-643` and `docs/architecture.md:472-473` say the online
  session remains BRIDGE-2, CORE-24 and HTSIM-18. CORE-24 and HTSIM-18 are
  delivered foundations; BRIDGE-2 remains open, and the newly identified
  exact-boundary backend prerequisite is not named.

`README.md:230-231`, which says the demonstrated native chain says nothing yet
about congestion or multi-request contention, remains accurate.

## Reproduction of the completed evidence

Only the artifact-free check is reproducible on this branch:

```bash
.venv/bin/python examples/congestion_chain_v1/run_study.py \
  --mission-run "$SIMLLM_MISSION_RUN" \
  --htsim-rnic "$SIMLLM_HTSIM_RNIC" \
  --run-dir "$SIMLLM_RUN_ROOT/congestion_chain_v1" \
  --check-only
```

It verifies that the selected run directory does not exist, validates the
frozen source and arithmetic, and exits without creating it. Removing
`--check-only` deliberately reaches `NotImplementedError`; it is not a hidden
partial study command.
