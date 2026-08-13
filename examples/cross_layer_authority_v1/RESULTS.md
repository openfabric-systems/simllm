# Cross-layer authority v1 results

CORE-8 asked for the cross-layer authority and the mechanism that enforces it:
one loss-checked projection from each authoritative runtime object into
`CompletionEvent` and `RequestBookkeeper`. This run establishes that mechanism
and applies it to those two projections. It does not close CORE-8, whose
cross-language half is untouched.

The useful deliverable is not this document. It is a check that fails when two
layers disagree about a quantity one of them owns, and thirteen hand-built
disagreements that were accepted before the change and are refused after it.

## Provenance

- Expectations frozen in `36499fc`, before any implementation and before the
  first run. The registered command was run with `--check-only` before that
  commit.
- Implementation in `138234d`.
- Evidence authored against SimLLM revision
  `4e1be35af5327c27db53ed002dc420e1de6f613b`. The run observed revision
  `138234d161f1f25f8979b662a28d1eeaeb0c79e0`. No equality is asserted between
  either revision and any submodule pin, and no pin literal is frozen here.
- Command:
  `python examples/cross_layer_authority_v1/run_study.py --out <run root>`.

## The duplicated-quantity inventory

Every serious defect this project found in the last two waves had one shape:
two authorities for one quantity, agreeing under the common case and drifting
where nobody looked. The inventory below is that class made visible. File and
line references are as of revision `4e1be35`, the revision the survey was taken
at; symbol names are given so they survive line drift.

Three entries are resolved by this change. The rest are recorded, unresolved,
because a working enforcement on a few quantities is worth more than a plan for
all of them.

### Resolved here

| # | Quantity | Representations | Status before |
|---|---|---|---|
| 1 | Queue-visit five-point timing | `QueueVisit` (`simllm/core/runtime.py:319`) and the four phase events rendered from it (`simllm/core/runtime.py:2713`) | unchecked |
| 2 | One WQE's lifecycle facts | `WqeLifecycleProjection` (`simllm/core/runtime.py:518`), its `QueueVisit`s (`simllm/core/runtime.py:2604`), its five subject-keyed events (`simllm/core/runtime.py:2735`), its bookkeeping `CreatedObjectRecord` and metadata (`simllm/core/runtime.py:3328`) | structure checked at `simllm/core/bookkeeping.py:488` and `:564`, bytes, tags and timestamps unchecked |
| 3 | Operation completion bytes and the class byte reduction | `_work_completed_bytes` (`simllm/core/runtime.py:2683`) into the logical completion event (`simllm/core/runtime.py:2807`), and `class_service_bytes` (`simllm/core/runtime.py:3092`) | unchecked |

Entry 3 also had to settle an ownership question rather than only add a check.
The operation-level `completed_bytes` and the `PROGRESS` and WQE byte counts
are not two copies of one number: the first projects the graph's declared
semantic payload and the second projects the bytes a resource served. For a
ring all-reduce they differ by construction. They now have one owner each, keep
distinct names, and are never summed.

### Recorded and not resolved

| # | Quantity | Representations | Checked? |
|---|---|---|---|
| 4 | Ring all-reduce chunk bytes | `simllm/traffic/patterns.py:104`, `simllm/traffic/collective_plan.py:119`, `simllm/traffic/step_comm.py:1655`, `simllm/core/runtime.py:2102` | unchecked, and already divergent: the runtime uses `payload_bytes // world` with no `max(1, ...)`, so a payload smaller than the rank count yields 0 bytes there and 1 byte in the other three |
| 5 | Same-node test and the NVLink rate that follows | `simllm/core/runtime.py:167` and `:2549` against `simllm/placement/declared.py:77` read at `simllm/traffic/locality.py:366`; rates `simllm/core/runtime.py:74` (bits/s) against `simllm/traffic/locality.py:40` (bytes/s) | unchecked; `CoarseDeviceProfile` never sees a `PlacementManifest` |
| 6 | Uniform pairwise all-to-allv expansion | `simllm/traffic/collective_plan.py:223` (filters positive payloads), `simllm/core/runtime.py:2208` (no filter), `simllm/traffic/execution_goal.py:339`, `simllm/traffic/step_comm.py:1458` and `:1691` | partial, only through `simllm/traffic/request_fidelity.py:76` and only when a request partition exists |
| 7 | Composite object-ID escaping | `simllm/core/runtime.py:108`, `simllm/core/bookkeeping.py:602` (identical), `simllm/backends/composed_rnic.py:652` (escapes `:` but not `%`, so a literal `%3A` in an operation ID collides) | unchecked across modules |
| 8 | GOAL tag allocation for a step | `simllm/core/runtime.py:1018` against `simllm/traffic/step_comm.py:1405`, `:1537`, `:1741` and `simllm/traffic/patterns.py:107` | partial, `simllm/traffic/execution_goal.py:311` reconciles the plan only |
| 9 | Step operation identity strings | `simllm/traffic/step_comm.py:1247` and `simllm/backends/step_lowerer.py:322`, plus inline copies at `simllm/traffic/step_comm.py:1585` and `:1620` | unchecked; a drift surfaces as a spurious missing boundary in `simllm/backends/dependency_cross_check.py:111` |
| 10 | Per-request arrival timestamp | `simllm/preplay/join.py:43`, `:70`, `:403` and its metadata copy `:413`, `simllm/core/bookkeeping.py:388`, `simllm/core/request_lifetime.py:107` | the metadata copy is documented as not consulted (`simllm/core/bookkeeping.py:349`); only a bound is checked at `simllm/core/completion.py:539` |
| 11 | `CompletionEvent` wire encoding | `simllm/core/execution_io.py:1523` (optional fields, carries a schema) against `simllm/core/bookkeeping_io.py:149` and `:211` (required fields, no schema) | unchecked; no round-trip equivalence test |
| 12 | Per-pair request partition ordering | `simllm/traffic/collective_plan.py:198` (insertion order) against `simllm/traffic/execution_goal.py:46` and `simllm/traffic/step_comm.py:1259` (sorted); order is load-bearing at `simllm/core/execution_io.py:606` | partial, sums only |
| 13 | Default GOAL base tag 1000 | constant at `simllm/core/runtime.py:75`; bare literals at `simllm/traffic/execution_goal.py:159` and `:677`, `simllm/traffic/step_comm.py:1347`, `:1496`, `:1727`, `:1905`, `:1942`, `:1969`, `simllm/backends/step_sink.py:155` | unchecked |
| 14 | Graph `execution_id` for a step | `simllm/backends/step_lowerer.py:117` and `simllm/core/execution_io.py:840`, with a differently padded artifact name at `simllm/backends/step_sink.py:485` | unchecked |
| 15 | Which requests a step sampled | `simllm/backends/step_lowerer.py:181`, `simllm/backends/step_sink.py:403`, and the stricter rule at `simllm/core/completion.py:194` | cardinality only |
| 16 | Node and GPU resource identity strings | `simllm/core/runtime.py:173`, `:788`, `:1967`, `:2522`, `:2568` and a disjoint scheme at `simllm/backends/composed_rnic.py:665` | unchecked |
| 17 | `nvlink_bandwidth_bytes_per_second` | `simllm/traffic/locality.py:182` (per phase) and `:284` (per plan) | unchecked between the two |
| 18 | Per-operation visit wait sum | `simllm/core/runtime.py:3008` (per operation) and `:3103` (graph wide) | only the graph-wide value is reconciled, at `simllm/core/completion.py:390` |
| 19 | Calibration cell identity | `simllm/compute/calibration.py:158` and `:206`, where the capture cell rebuilds a plan cell for validation and discards it at `:233` | checked only at parse time |

Entry 4 is the one worth acting on next. It is not a latent risk but a live
disagreement between four implementations of one rule, and the divergent copy
is in the runtime.

## What was found before the check existed

Three read-only diagnostics ran before the freeze and are disclosed in it
rather than presented as results.

- Every registered clause held on 719 runtime reports, 1,663 operations, 8,989
  queue visits, 2,777 WQEs and 38,540 completion events, with zero violations.
- Every clause also held on the 593 hand-built triples the tests feed the
  reducer directly rather than through the coarse runtime.
- All thirteen registered contradictions were accepted by the checking surfaces
  that existed then.

One pre-freeze guess was refuted and is recorded because it changed a clause:
`class_service_bytes` was assumed to reduce WQE payload bytes by class, which
failed on 116 reports. It reduces control-work bytes only, and clause A6 states
the rule the source actually implements. That is the inventory's own lesson
applied to itself: a name that suggests a general service-byte reduction while
covering one work kind is the shape a second authority hides behind.

## Scored evidence

The freeze registered three families over eighteen instances. Two corrections
are reported here rather than absorbed silently.

- CLA-B2 was registered with four cells, of which the 400 Gbit/s, 700 ps cell
  was already disclosed as a preservation baseline. Three cells are scored.
- CLA-B3 is reclassified from scored to a disclosed structural invariant. Both
  of its counts were observed while designing the fixture before the freeze, so
  reporting them as a scored prediction would inflate the denominator.

The scored headline is therefore two families over sixteen instances, all
passed, with genuine-risk fraction 16/16: every scored instance was measured to
fail before the change or was predicted from arithmetic before being run.

### CLA-B1: registered contradictions rejected, 13/13

Each was accepted before `138234d` and is refused after it. The message names
the violated join rather than reporting a generic mismatch.

| ID | Verdict after the change |
|---|---|
| C1 queued at submission | expected queued event of `xfer-fifo` on `nccl-channel` at 89,620 ps, the stream reports 3,000 ps |
| C2 dropped submitted event | completion stream lost a projection |
| C3 duplicated started event | stream carries a projection the authority does not |
| C4 progress bytes raised | expected 128 bytes, the stream reports 129 |
| C5 WQE completion moved 1 ps | expected 89,620 ps, the stream reports 89,619 ps |
| C6 phantom subject object | stream carries a projection the authority does not |
| C7 eligibility and grant swapped | expected queued event of `xfer` on `host-launch-queue` at 0 ps, the stream reports 1,000 ps |
| C8 invented class bytes | report class service bytes disagree with the graph's control-work reduction |
| D1 WQE created 1 ps early | ledger creates the WQE at 6,999 ps, the authority submits it at 7,000 ps |
| D2 WQE bytes raised | ledger records 4,097, the authority owns 4,096 |
| D3 WQE absent from ledger | ledger lost a WQE the authority reports |
| D4 ledger-only event | ledger carries an event the result does not |
| D5 completion stage moved | ledger completes at 296,981 ps, the result at 296,980 ps |

Thirteen mutations exercise nine distinct rejection paths: timestamp
disagreement, loss, duplication and byte disagreement on the event join, the
scalar class reduction, and creation time, metadata, loss and invention on the
ledger join. That is the honest count of independent mechanisms; the thirteen
count independent ways a producer can drift.

C1 and C7 are the pair the fixture exists for. On a visit with zero queue wait
and immediate eligibility, logical submission, eligibility and the resource
grant are the same number, every rendering agrees, and a confusion between them
is invisible. The fixture carries one subjectless visit whose eligibility is
89,620 ps against a 3,000 ps submission, and five with strictly positive queue
wait, so the three quantities are distinguishable.

### CLA-B2: the two-parameter sweep, 3/3 scored cells exact

| Cell | Rate Gbit/s | Delivery ps | Frozen JCT ps | Measured JCT ps | Role |
|---|---:|---:|---:|---:|---|
| baseline | 400 | 700 | 296,980 | 296,980 | preservation baseline |
| slow link | 200 | 700 | 586,260 | 586,260 | scored |
| slow delivery | 400 | 1,500 | 297,780 | 297,780 | scored |
| both | 200 | 1,500 | 587,060 | 587,060 | scored |

`result.completed_at_ps`, `result.quiesced_at_ps`, `StepResult.step_latency_ps`
and the single request's TTFT are equal in every cell, so the chain from graph
to runtime authority to `CompletionEvent` to `StepResult` to TTFT is live and
the enforced projection sits on it rather than beside it.

Both registered relations hold exactly:

- halving the link rate adds 289,280 ps at both delivery values, which is the
  serialization of 14,464 bytes at 200 against 400 Gbit/s;
- adding 800 ps of visibility adds exactly 800 ps at both rates.

Every cell also passed A1, A5, A6, B1, B1b, B2 and B3 on its own evidence.

## Fatal and unscored evidence

None of the following enters the behavioral denominator. All held; a violation
would have voided the run.

- The full existing suite passes unchanged with the enforcement live: 1,304
  tests before, 1,325 after with the 21 new ones, 0 failures. The producer's
  event renderer, its emission order and its arithmetic are untouched, so no
  accepted timestamp, digest or completion identity moves.
- The unmutated evidence of every cell is accepted by the reducer, by the
  event projection and by the ledger projection.
- A refused stream leaves the virtual clock at 0 with no request metric, and
  the same reducer then accepts the unmutated stream and returns the registered
  boundary.
- A runtime whose ledger claims one byte more than its WQE authority is refused
  during `execute`, and the caller's bookkeeper is still empty afterwards, so
  the staged append rolls back rather than committing a disagreeing ledger.
- The fixture carries 15 visits, 4 WQEs, 69 events and 87 ledger entries in
  every cell, and `class_service_bytes` is `[[0, 128]]` in every cell.
- `ruff check .` passes.

## Entailment

The scored relations are not entailed by any earlier fatal oracle in this run.
The contradictions are evaluated against the raw runtime report and the raw
ledger, before and independently of the reducer's other validation, and each
mutation was measured to pass the full pre-change validation chain. The sweep
cells are compared against arithmetic frozen before the run, not against the
runtime's own previous output.

The disclosed diagnostics are the opposite case and are kept out of the score
for exactly that reason: after the change they would be entailed by the check
being added, so counting 38,540 events as passing instances would be counting
the implementation against itself.

## Physical sanity

Three independent angles, per the maintainer rule, before any digit was
compared.

**Network and serialization physics.** Four WQEs carry 14,464 bytes on one
RNIC. At 400 Gbit/s that is 289,280 ps of serialization, a floor no schedule
can beat; at 200 Gbit/s, 578,560 ps. The first grant cannot precede 7,000 ps,
because five launch-queue services of 1,000 ps place `xfer` on its channel at
2,000 ps and 5,000 ps of channel service releases it at 7,000 ps. The measured
296,980 ps sits exactly on 7,000 plus the floor plus the 700 ps visibility
term, i.e. the RNIC is saturated with no idle gap from the first grant to the
last release. The ceiling, if nothing overlapped and each transfer also paid
its own channel service serially, is 311,980 ps; the measurement is below it,
which is the overlap the runtime claims. The scaling check gives 1.974, not 2,
because the 7,700 ps of fixed terms does not scale; a ratio of exactly 2 would
have meant the fixed terms were being scaled too, and a ratio near 1 would have
meant the transfers were not serialization bound.

**Compute and memory physics.** `compute-a` declares 100,000 ps and 1,024 HBM
bytes. At any plausible HBM bandwidth those bytes are far under a nanosecond,
so the kernel is issue bound rather than memory bound and its nominal duration
dominates, as the model intends. It completes at 101,700 ps, inside the graph
and far from the 296,980 ps boundary, so it never gates the result and cannot
be silently absorbing the discrepancy the network terms would otherwise show.

**System plausibility.** 297 ns for 14 KB across one 400 Gbit/s rail is the
right order for pure serialization, and that is all this fixture models. A real
NCCL collective of this size is dominated by launch and protocol overhead in
the microsecond range, not by 14 KB of wire time. This study makes no fidelity
claim about collective latency; it is a projection-checking harness whose
numbers exist to be predictable, and its service constants are chosen to
separate eligibility from the grant rather than to imitate a device. Reading
296,980 ps as a modeled NCCL latency would be wrong.

## Contradiction sweep

`README.md`, `docs/README_PRO.md` and `docs/architecture.md` were swept for
statements this change contradicts. No hits; two near-misses are worth
recording rather than editing.

- `docs/architecture.md:230-234` states that the bookkeeping stream "is a
  public projection, not a second WQE lifecycle implementation". That was an
  unenforced assertion until this change and is now checked, so the text
  becomes true rather than false.
- `docs/README_PRO.md:205-207` says CORE-8 "fixes one cross-language queue-visit
  meaning and identity arbitration baseline". That remains open and accurate:
  this run did not touch the cross-language half.

`docs/modules/core.md` is this module's own doc and was updated in the same
change, both in the authority section and in the status narrative.

## CORE-8 closure map

CORE-8 does not close. Each registered clause is quoted and mapped.

1. "establish the cross-layer authority and queue-visit contract above before
   residual-driven calibration": partly demonstrated. The cross-layer authority
   is established and enforced for the two projections the next clause names.
   The queue-visit contract is enforced at the projection boundary, i.e.
   `QUEUED` must be eligibility and `STARTED` must be the grant, but its
   cross-language conformance is not.
2. "Define one loss-checked projection from each authoritative runtime object
   into `CompletionEvent` and `RequestBookkeeper`": demonstrated. Queue visits,
   WQE lifecycles and operation completions each project into both, joined by
   stable identity and checked for loss, duplication and timestamp
   disagreement.
3. "use a versioned bookkeeping or completion-event extension only where v1
   cannot represent that projection without ambiguity": demonstrated by not
   needing one. No schema changed; `simllm-completion-event-v1` and
   `simllm-request-bookkeeping-v1` represent every projection above.
4. "the native side extracts a protocol-neutral exact reservation timeline and
   finite-capacity resource from the PCIe implementation, while Python uses a
   reference serial and capacity resource for GPU and runtime queues": not
   demonstrated. No C++ was touched.
5. "Shared golden fixtures must prove isolated zero wait, one external
   contention wait without triangular self-charging, predecessor service
   excluded from queue wait, finite-capacity release, overflow rollback and
   identical reductions in both languages": not demonstrated. Those fixtures
   are frozen in
   [queue_contract_v1](../queue_contract_v1/expectations.md) and remain unrun.
6. "Existing PCIe, WQ and GPU studies must remain byte-identical under identity
   arbitration before any non-identity class policy is enabled": partly
   demonstrated. The producer is unchanged and the full suite passes, so
   nothing in this change can move a byte; the clause's own study-level
   verification under a non-identity policy is out of this run's scope.

Clauses 4, 5 and 6 keep CORE-8 open. They stay on CORE-8 rather than moving to
new IDs, because they are that task's own registered clauses and moving them
would only rename the same open work.

## Task IDs registered

Zero. Both IDs allocated to this branch, CORE-52 and CORE-53, are unused.

The rule is that a new ID is registered only for a registered acceptance clause
a run did not demonstrate. This run's own registered clauses 1 through 5 are
all demonstrated, and clause 6 is satisfied by leaving undemonstrated CORE-8
clauses on CORE-8. The unresolved inventory entries are findings, not clauses:
no acceptance clause claimed them, so per the wave-10 residual discipline they
belong in this report rather than in the registry. Entry 4, the ring chunk
divergence, is the one a future change should take first.

## What a reader should not conclude

- Not that every duplicated quantity is now checked. Sixteen of the nineteen
  inventory entries are unresolved and named above.
- Not that the coarse runtime is now verified. The check verifies that its
  projections agree with its own report; it does not verify that the report is
  physically right. That is what the sanity bounds and the separate studies are
  for.
- Not that the sweep numbers model a real collective. See the system
  plausibility paragraph.
