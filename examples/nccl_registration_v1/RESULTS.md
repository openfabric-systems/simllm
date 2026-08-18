# nccl_registration_v1: results

Verdict: **interpretable**. All nine fatal guards held as frozen, all six
exact-oracle rows passed, and all three behavioral families passed over their
seven instances. The two evidence classes are reported separately and are
never summed.

This study measures the interim collective-completion contract of the
maintainer's 2026-08-18 kernel-time determinism ruling: collectives complete
through the deterministic ATLAHS and htsim chain with a no-tail constant
completion, gated on the destination memory being registered, with the
NCCL/RCCL buffer-channel registration carrying an explicit modeled time cost.

This document was corrected after an adversarial review. Every correction is
listed in [Corrections](#corrections) at the end, including the claims the
first version made that did not survive.

## Chronology

The freeze is commit `aaea1ec`, "Freeze the NCCL registration cost
expectations". It contains `expectations.md`, `expectations.json`, the
`.gitattributes` rules and a lock on the freeze document, and nothing else: no
mechanism, no wiring, no runner and no measured value. Both frozen files are
byte-identical today.

The implementation landed after it, in commit `07e4fd7` ("Model NCCL and RCCL
buffer registration as a one-time cost") and commit `2d15725` ("Charge
collective registration on the live metric chain"). This runner and these
results landed after both. Every relation below was written down before the
code that produces it existed, so the pre-registration claim is genuine rather
than reconstructed. The correction round that follows changed how the frozen
guards are evaluated and what this report claims; it changed nothing in the
freeze.

## What was run

Two frozen cells, both through the repository's own metric chain: `StepRecord`
into `HtsimStepSink`, an authoritative `ExecutionGraph`, the locality
projection, `StepResult`, `attribute_step_detail`, then
`HtsimRequestMetricReducer` into per-request TTFT and TPOT. One post-specified
cell was added in the correction round and is labeled as such everywhere.

| | `local-tp2` | `mixed-tp4` | `slow-fabric-tp4` (post-specified) |
|---|---|---|---|
| geometry | 32 layers, hidden 4096, inter 11008 | 2 layers, hidden 1024, inter 4096 | same as `mixed-tp4` |
| ranks and hosts | 2 ranks on one host | 4 ranks across two hosts | 4 ranks across two hosts |
| link rate | not used, no fabric segment | 400 Gbit/s | 10 Gbit/s |
| steps | prefill of 512 tokens, then two decodes | prefill of 64 tokens, then one decode | prefill of 64 tokens, then one decode |
| collectives per step | 64 | 4 | 4 |
| executed artifacts per step | 160 (128 collective phases plus 32 compute) | 26 (24 collective phases plus 2 compute) | 26 |
| phases per collective | 2 | 6 | 6 |
| backend processes per step | 0 | 24 | 24 |
| GOAL artifacts written | 0 | 48 across two steps | 48 across two steps |
| arms | `off`, `on`, `on-2ch`, `on-rebuild` | `off`, `on` | `off`, `on` |

Each frozen cell additionally runs a **feature-absent** replay whose
configuration omits the registration parameter entirely. The `off` arm passes
`collective_registration=None` explicitly, so the two are different
constructions and their comparison is not vacuous.

Reproduce with the backend binaries on the path the module documents:

```bash
python examples/nccl_registration_v1/run_study.py --workdir <a directory on a data volume>
```

## Headline numbers

| cell | arm | TTFT (ps) | step makespans (ps) | registration charged per step (ps) |
|---|---|---|---|---|
| `local-tp2` | `off` | 5,914,368,000 | 5,914,368,000 / 2,408,960,000 / 2,409,056,000 | 0 / 0 / 0 |
| `local-tp2` | `on` | 7,194,368,000 | 7,194,368,000 / 2,408,960,000 / 2,409,056,000 | 1,280,000,000 / 0 / 0 |
| `local-tp2` | `on-2ch` | 8,474,368,000 | 8,474,368,000 / 2,408,960,000 / 2,409,056,000 | 2,560,000,000 / 0 / 0 |
| `local-tp2` | `on-rebuild` | 7,194,368,000 | 7,194,368,000 / 2,408,960,000 / 3,689,056,000 | 1,280,000,000 / 0 / 1,280,000,000 |
| `mixed-tp4` | `off` | 87,508,640 | 87,508,640 / 72,025,760 | 0 / 0 |
| `mixed-tp4` | `on` | 167,508,640 | 167,508,640 / 72,025,760 | 80,000,000 / 0 |
| `slow-fabric-tp4` | `off` | 700,925,600 | 700,925,600 / 81,610,400 | 0 / 0 |
| `slow-fabric-tp4` | `on` | 780,925,600 | 780,925,600 / 81,610,400 | 80,000,000 / 0 |

The opt-in moves TTFT by exactly the registered amount and by nothing else.

## Fatal guards, unscored

A violated guard would void the run. None was violated, and none is reported
as a fraction. The claim column is the frozen text verbatim; the evaluation
column says what the runner actually checked, and the runner publishes the
frozen claim string beside its own evaluation sentence so a weaker check
cannot appear under a frozen identity.

| guard | frozen claim | how it was evaluated | held |
|---|---|---|---|
| G1 | the default-constructed arm and the explicitly disabled arm agree field by field | the off arm passes `collective_registration=None` explicitly while the feature-absent arm omits the parameter, so two different constructions are compared, on step makespans, TTFT, the GOAL digest list, the registration outcome count and the emptiness of the per-artifact registration projection | yes |
| G2 | every GOAL artifact SHA-256 is identical between the off and on arms | every arm's digest list against its cell's off arm. Vacuous in `local-tp2`, which writes no GOAL artifact; the 48 `mixed-tp4` artifacts carry the whole guard | yes |
| G3 | the off arm reproduces the accepted baseline timings exactly | field by field against the feature-absent computation. For all 16 evaluated steps: makespan, completion, per-artifact fabric service, per-artifact local service, per-artifact base latency, per-artifact composed service, per-artifact medium and the complete medium partition | yes |
| G4 | every step and every request partition conserves its span exactly | both halves. 16 evaluated step partitions conserve their makespans, and 6 per-request rows conserve TTFT and the decode span in both the coarse and the medium view | yes |
| G5 | a calibrated request against a declared cost fails closed | all three clauses: `calibrated_cost_ps` raises on the declared cost, a `calibrated` provenance without a measurement locator cannot be constructed, and an unknown model selector raises | yes |
| G6 | no collective is charged registration on more than one artifact | for every enabled arm and step, the nonzero per-artifact charges sum to the ledger charge and their count times the per-identity cost times the channel count reproduces it. 4 of 26 artifacts in `mixed-tp4`, 64 of 160 in `local-tp2` | yes |
| G7 | the mirrored nccl_stack event stream is unchanged when no registration is requested | against the accepted sequences. All 8 tracked per-rank event-sequence rows of `examples/nccl_stack_v1/results.csv` were regenerated through an ungated communicator with the accepted study's own digest helpers and reproduce the tracked event count and SHA-256 exactly; a default and an explicitly ungated communicator also produce identical JSON | yes |
| G8 | the results identity block equals this freeze | the geometry read back off the executed `HtsimStepSinkConfig` objects, plus the cost and evidence class read off the shipped model, against `expectations.json`. The two sides are derived independently | yes |
| G9 | ledger construction facts | all six re-derived: disabled charges zero, first charge pays, second charges zero, new buffer re-pays, new peer re-pays, rebuild re-pays | yes |

No guard is partially evaluated. Two vacuity disclosures remain: G2 in
`local-tp2` and the GOAL conjunct of B3 in the same cell. Both are stated
rather than counted as coverage.

## Scored relations

### Exact-oracle rows: 6 of 6

| row | expected | observed | passed |
|---|---|---|---|
| O1 `local-tp2` TTFT delta | 1,280,000,000 ps | 1,280,000,000 ps | yes |
| O2 `local-tp2` decode makespan delta | 0 ps | 0 and 0 ps | yes |
| O3 `local-tp2` two-channel TTFT delta | 2,560,000,000 ps | 2,560,000,000 ps | yes |
| O4 `local-tp2` rebuild re-pay on step 2, TTFT unchanged | 1,280,000,000 ps and equal TTFT | 1,280,000,000 ps and equal TTFT | yes |
| O5 `mixed-tp4` TTFT delta across a phase split with a real backend | 80,000,000 ps | 80,000,000 ps | yes |
| O6 published registration component equals the ledger charge | equality on every step of every enabled arm | equality | yes |

### Behavioral families: 3 of 3 over 7 instances

- **B1 linearity in identities**, 3 instances at 4, 64 and 128 identities.
  Charged registration was 80,000,000, 1,280,000,000 and 2,560,000,000 ps,
  exactly `identities x 20,000,000 ps` in each.
- **B2 the charge is additive, not absorbed**, 2 instances. In both frozen
  cells the first step's makespan under `on` equals its makespan under `off`
  plus the charged total exactly. The counterfactual section below states how
  much that discriminates, and adds the post-specified cell where it
  discriminates completely.
- **B3 later steps are untouched**, 2 instances. Every step after the
  registering step has an identical makespan under `on` and `off`, and an
  identical GOAL digest list. The digest conjunct is vacuous in `local-tp2`,
  which writes no GOAL artifact, so only the makespan conjunct carries that
  instance.

### What the scored set really covers

Six exact-oracle rows and three behavioral families over seven instances are
separate counts and are never added. They are also not seven independent
observations. O1, B1's 64-identity instance, B2's `local-tp2` instance and
O6's step-0 equality are four views of the same event, the 64 charges of
`local-tp2` step 0; once G9 holds, `identity count x declared cost` is
entailed, so those four do not multiply the evidence. The genuinely
independent risks the set covers are the charging site, the phase-zero gate,
the dedup across steps, the channel multiplicity and the rebuild.

## The counterfactual B2 exists to reject

The composition is per executed artifact. The alternative worth rejecting is
therefore `max(local, fabric, registration)` on each charged artifact instead
of `registration + max(local, fabric)`. Under that fold a charged artifact
contributes `max(realized, 20,000,000)` rather than `realized + 20,000,000`,
so the visible delta shrinks by the realized service of the charged artifacts.

| cell | charged artifacts | realized service each (ps) | observed delta (ps) | delta if folded (ps) | discrimination |
|---|---|---|---|---|---|
| `local-tp2` | 64 | 4,661,000 | 1,280,000,000 | 981,696,000 | 298,304,000, i.e. 23.3 percent |
| `mixed-tp4` | 4 | 2,655,360 | 80,000,000 | 69,378,560 | 10,621,440, i.e. 13.3 percent |
| `slow-fabric-tp4` (post-specified) | 4 | 28,214,400 | 80,000,000 | 0 | complete, the fold hides the charge entirely |

The two frozen cells discriminate by 23.3 and 13.3 percent, which is real but
weak: neither exercises the case the family exists to detect, where a
per-artifact realized service exceeds the charge and the fold becomes
invisible. The post-specified `slow-fabric-tp4` cell supplies it. At 10 Gbit/s
one 32,768-byte flow serializes for 26,214,400 ps on top of the 2,000,000 ps
propagation reference, giving 28,214,400 ps per artifact against a 20,000,000
ps charge, so a folded charge would produce a delta of exactly zero. The
observed delta is 80,000,000 ps.

## Physical sanity, checked against bounds stated before the run

**`local-tp2` compute floor.** The frozen geometry is about 6.5 G parameters,
so a 512-token prefill costs about 6.64 TFLOP, halved by tensor parallelism to
3.32 TFLOP per rank. Against the B100 envelope of 1.8 PFLOP/s derated to 0.7
that is a floor near 2.6 ms. The measured kernel component is 5,317,760,000
ps, i.e. 5.32 ms, which sits above its floor by a factor near two. That gap is
the roofline provider charging attention, the LM head and memory-bound terms
on top of the dense GEMM floor, and it is the right side of the bound.

**`local-tp2` serialization, and where the exactness ends.** Each all-reduce
moves 512 x 4096 x 2 bytes; a two-rank ring puts the same 4,194,304 bytes
through each endpoint, split across the two ring phases the locality plan
emits, so each phase carries 2,097,152 endpoint bytes. The measured NVLink
component is 596,608,000 ps over 128 phases, i.e. exactly 4,661,000 ps per
phase. That reproduces the model's own closed form exactly, and the closed
form contains a quantization: `classify_step_locality` charges
`ceil(endpoint_load x 1e9 / bandwidth) x 1000` picoseconds per phase, because
GOAL calc units are whole nanoseconds. Against continuous physics, 2,097,152
bytes at 450 GB/s is 4,660,337.8 ps, so the model is high by 662.2 ps per
phase, 84,764 ps over the step, or 0.0142 percent. On a decode step the same
rule costs proportionally more: 4,096 endpoint bytes per phase round from
9,102.2 ps up to 10,000 ps, so the measured 1,280,000 ps exceeds the
continuous 1,165,084 ps by 9.86 percent. The model form is exact; the physics
form is not, and the whole deviation is the nanosecond ceiling.

**`mixed-tp4` fabric, exact both ways.** Each of the 24 backend runs carries
one 32,768-byte flow. At 400 Gbit/s that is 655,360 ps of serialization on top
of the fluid profile's 2,000,000 ps propagation reference, so 2,655,360 ps per
run and 63,728,640 ps for 24. The measured fabric component is 63,728,640 ps
and the measured per-artifact term is 2,655,360 ps. No quantization enters
here, so this side is exact against continuous physics as well as against the
model.

**The scaling check.** Doubling the channel count doubled the charge exactly,
from 1,280,000,000 to 2,560,000,000 ps, and moved nothing else: the decode
makespans, the TPOT and the GOAL digests are identical across `off`, `on` and
`on-2ch`. A relation that moved by 1.05 or by 40 would have refuted the model
regardless of how exactly O1 matched, which is why the second measurement was
taken.

**Plausibility, and why one cell is a toy.** In `local-tp2` the registration
adds 1.28 ms to a 5.91 ms TTFT, a 21.6 percent increase, which is the "visible
minority of TTFT" the freeze predicted. `mixed-tp4` is a different matter and
the freeze said so in advance. Its per-rank prefill is about 1.07 GFLOP,
against a FLOP floor near 1.7 us and a weight-read floor near 4.1 us, while
the measured compute term is 23,780,000 ps across two artifacts: the cell is
dominated by fixed and memory terms rather than by its own arithmetic, its
whole step is 87.5 us, and the 80 us charge nearly doubles it. That is a
statement about the size of the cell, not evidence about a production
deployment, and it is why the cell exists only to exercise the phase split and
a real backend. Real NCCL registers user buffers around communicator setup and
the whole setup costs tens of milliseconds, so a per-buffer cost of tens of
microseconds is the right order of magnitude. It remains a declared constant
and not a measurement.

**The decode side.** `local-tp2` TPOT is 2,409,008,000 ps, i.e. 415 tokens per
second for one request. Weight traffic per rank per decode step is about
6.5 GB, which against the 8 TB/s B100 memory envelope is a floor near 0.81 ms,
so the measured 2.41 ms is above its floor by about three. The number is on
the optimistic side of a real 7B deployment but inside the physical envelope,
and no scored relation depends on it.

## Reproduction, post-specified

The freeze did not ask for this, so it is recorded as a post-specified check.
The study was executed twice into fresh working directories, with the same
backend binaries, and the two `results.json` documents were byte-identical:
every makespan, every TTFT, every fabric term decided by one of the real
`htsim_rnic` subprocesses, and every registration charge. That is the
constant-completion half of the interim contract observed directly rather than
asserted.

The document this report describes has
`sha256(results.json) =`
`c31bd288030fc3c99e916a4bced6f1f87367a6fc62bfe2c6aef9801688b1482a`, recorded
so the claim above is checkable rather than merely stated.

## What this run does not establish

- **Almost none of the model is measured.** The net plugin ABI establishes two
  things: a registration entry point exists at the seam, and one seam serves
  NCCL and RCCL. That the cost is paid once rather than per call, that the
  identity is scoped to a buffer, that a channel belongs to that identity,
  that exactly three events force a re-registration, and the 20,000,000 ps
  duration are all declared model choices with no measurement behind them.
  `TRAF-56` is the calibration and has to measure the model choices as well as
  the constant; `calibrated_cost_ps` refuses to serve the declared number in
  the meantime.
- **No packetized handshake.** The charge is a serialized constant. It occupies
  no link, contends with nothing and appears in no GOAL artifact. `TRAF-55`
  makes it port traffic; `TRAF-54` packetizes the collective path it would run
  over; `TRAF-57` makes that path port-kind independent across NVLink, xGMI and
  UALink.
- **No live new-peer re-registration.** All cells have a fixed rank set, so
  the new-peer event is covered only by the by-construction ledger guard G9.
- **Two registration states, not one.** The traffic-owned ledger and the
  mirrored seam's `require_buffer_registration` gate keep separate state; the
  gate carries no generation and the live chain never consults it. This study
  exercises them separately and never together. `TRAF-58` unifies them.
- **Two compositions untested.** Selecting `dependency_cross_check` with a
  registration model would shift the authority flow timestamps by the charge
  and report a spurious disagreement (`TRAF-59`), and
  `HtsimPersistentStepSink.prepare` charges the ledger for every prepared
  record while outcomes advance at publish (`TRAF-60`). Neither combination is
  exercised here.
- **No deregistration, no cache eviction, no size dependence.** The cost is
  flat in buffer size, which is certainly wrong for a mechanism dominated by
  page pinning.
- **Nothing about whether a no-tail constant completion is the right model.**

## Two things that were at genuine risk

The freeze asked the entailment question per relation, and two of the answers
were tested by the implementation rather than assumed.

First, the additive composition, quantified in the counterfactual section
above. Second, conservation: the attribution partition asserts that the
executed artifact services reproduce the step makespan exactly, and it
rejected the first version of this change until `MediumAttribution` gained a
named registration component and `_artifact_ownership` learned the new
identity. That rejection is why the registration is reported apart from
`collective_base_ps` rather than folded into it: a once-per-identity setup
cost and a per-call fixed cost are different claims and a reader must be able
to tell them apart.

## Corrections

An adversarial review of the first published version found the mechanism sound
and the study record not. Every finding below changed this document, the
runner, or both. The freeze was not touched.

1. **G3 was substituted.** The frozen G3 requires the off arm's step
   latencies, makespans, fabric services, local services, base latencies and
   attribution partitions to equal a feature-absent computation. The first
   runner checked only that the ledger charged zero and the projection was
   empty, and this report printed that weaker claim under the frozen
   identity. G3 is now evaluated field by field against the feature-absent
   arm, and the runner publishes the frozen claim string beside a separate
   sentence describing what it checked, so a substitution cannot recur
   silently.
2. **G1 was vacuous.** The first off arm omitted the parameter exactly like
   the feature-absent arm, so G1 compared one construction with itself. The
   off arm now passes `collective_registration=None` explicitly.
3. **Three more vacuities are fixed and one more is disclosed.** G8 derived
   both sides of its geometry comparison from the freeze; it now reads the
   geometry back off the executed sinks. G7 compared two ad-hoc equal streams
   instead of the accepted sequences; it now regenerates all 8 tracked
   `nccl_stack_v1` per-rank sequences and compares their event counts and
   digests. G4 evaluated only the step half of its claim; the per-request half
   is now evaluated too. G5's third clause is now evaluated in the runner
   rather than only in pytest. B3's GOAL conjunct is vacuous in `local-tp2`
   and is disclosed beside the same corner in G2.
4. **The counterfactual arithmetic was wrong.** The first version claimed that
   folding the charge into the max would have shown 16.3 us instead of 80 us
   in `mixed-tp4`, and framed the charge as something the fabric term could
   hide. The composition is per artifact, the per-artifact fabric term is
   2,655,360 ps, and the correct folded delta is 69,378,560 ps, so B2
   discriminates there by 13.3 percent rather than by a factor of five. The
   16.3 us figure corresponded to a per-collective maximum that does not exist
   in the code. The post-specified `slow-fabric-tp4` cell was added because no
   frozen configuration exercises the case B2 exists to detect.
5. **The evidence attribution was too broad.** The first version credited the
   ABI with the one-time nature and the per-buffer scope of the registration.
   The ABI supports only the existence of the entry point and the cross-stack
   identity of the seam. Corrected here, in the mechanism module, in the
   traffic contract, in the compute cross-reference and in TRAF-56.
6. **The NVLink check claimed exactness it did not have.** It reproduces the
   model exactly and continuous physics to 0.0142 percent on the prefill step
   and 9.86 percent on a decode step, both entirely from the per-phase
   nanosecond ceiling. Stated above with the rule named.
7. **A step count was wrong.** G4 was reported over "22 executed steps"; the
   correct figure is 16 evaluated frozen steps, or 21 including the
   feature-absent replays that G1 and G3 consume but G4 does not.
8. **The reproduction claim was uncheckable.** The results digest is now
   recorded.
9. **The scored views overlap.** Disclosed above rather than left to the
   reader.
10. **The layering claim overstated.** The seam and the ledger are two
    registration states, not one authority. Corrected in the module docstring
    and the module docs, and registered as TRAF-58.
11. **Two untested compositions are now registered**, as TRAF-59 and TRAF-60.

The narrative correction that came with finding 4: `mixed-tp4` splits each
collective into 6 phases, not 2, and its `nvlink_ps` is zero in every step
because the 2,655,360 ps fabric term wins the maximum against a 73,000 ps
local term, which the medium projection reports as masked NVLink service
rather than as owned time.
