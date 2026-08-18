# nccl_registration_v1: results

Verdict: **interpretable**. All nine fatal guards held, all six exact-oracle
rows passed, and all three behavioral families passed over their seven
instances. The two evidence classes are reported separately and are never
summed.

This study measures the interim collective-completion contract of the
maintainer's 2026-08-18 kernel-time determinism ruling: collectives complete
through the deterministic ATLAHS and htsim chain with a no-tail constant
completion, gated on the destination memory being registered, with the
NCCL/RCCL buffer-channel registration carrying an explicit modeled time cost.

## Chronology

The freeze is commit `aaea1ec`, "Freeze the NCCL registration cost
expectations". It contains `expectations.md`, `expectations.json`, the
`.gitattributes` rules and a lock on the freeze document, and nothing else: no
mechanism, no wiring, no runner and no measured value.

The implementation landed after it, in commit `07e4fd7` ("Model NCCL and RCCL
buffer registration as a one-time cost") and commit `2d15725` ("Charge
collective registration on the live metric chain"). This runner and these
results landed after both. Every relation below was written down before the
code that produces it existed, so the pre-registration claim is genuine rather
than reconstructed.

## What was run

Two cells, both through the repository's own metric chain: `StepRecord` into
`HtsimStepSink`, an authoritative `ExecutionGraph`, the locality projection,
`StepResult`, `attribute_step_detail`, then `HtsimRequestMetricReducer` into
per-request TTFT and TPOT.

| | `local-tp2` | `mixed-tp4` |
|---|---|---|
| geometry | 32 layers, hidden 4096, inter 11008 | 2 layers, hidden 1024, inter 4096 |
| ranks and hosts | 2 ranks on one host | 4 ranks across two hosts |
| steps | prefill of 512 tokens, then two decodes | prefill of 64 tokens, then one decode |
| collectives per step | 64 | 4 |
| executed artifacts per step | 160 | 26 |
| backend processes per step | 0 | 24 |
| GOAL artifacts written | 0 | 48 across two steps |
| arms | `off`, `on`, `on-2ch`, `on-rebuild` | `off`, `on` |

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

The opt-in moves TTFT by exactly the registered amount and by nothing else.
In `local-tp2` that amount is 64 identities times 20,000,000 ps; in
`mixed-tp4` it is 4 identities times the same cost, even though every
collective there is split into several executed artifacts and a real
`htsim_rnic` process decides each fabric term.

## Fatal guards, unscored

A violated guard would void the run. None was violated, and none is reported
as a fraction.

| guard | claim | held | evidence |
|---|---|---|---|
| G1 | the default-constructed arm equals the explicitly disabled arm | yes | both cells agree on every step makespan, on TTFT, on the GOAL digest list, on publishing no registration outcome and on leaving the per-artifact registration projection empty |
| G2 | every GOAL artifact SHA-256 is identical between the off and on arms | yes | 48 artifacts in `mixed-tp4` match across `off` and `on`. This guard is vacuous in `local-tp2`, which writes no GOAL artifact at all, and that is stated rather than counted |
| G3 | the off arm charges nothing and publishes no registration term | yes | ledger charge 0, empty per-artifact projection, `collective_registration_ps` 0 on every step of both cells |
| G4 | every step's medium partition conserves its makespan | yes | all 22 executed steps across both cells and every arm |
| G5 | a calibrated request against a declared cost fails closed | yes | `calibrated_cost_ps` raises, naming the declared class and TRAF-56; an unknown model selector raises at configuration time |
| G6 | no collective is charged registration on more than one artifact | yes | in `mixed-tp4` exactly 4 of the 26 artifacts carry a nonzero charge of 20,000,000 ps each; in `local-tp2` exactly 64 of 160 do |
| G7 | the mirrored seam is unchanged when no registration is requested | yes | the 394-event ungated stream is byte-identical between the default communicator and an explicitly ungated one; the gated communicator emits 2 registration events, refuses an unregistered buffer, and leaves the caller's clock at 17 ps |
| G8 | the results identity block equals the freeze | yes | declared cost, evidence class and both cell geometries match `expectations.json` |
| G9 | ledger construction facts hold | yes | disabled charges zero, first charge pays, second charges zero, new buffer re-pays, new peer re-pays, rebuild re-pays |

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
- **B2 the charge is additive, not absorbed**, 2 instances. In both cells the
  first step's makespan under `on` equals its makespan under `off` plus the
  charged total exactly. This is the instance that matters in `mixed-tp4`,
  where the fabric term is 63,728,640 ps and a charge folded inside the
  existing `max(local, fabric)` composition would have been hidden by it.
- **B3 later steps are untouched**, 2 instances. Every step after the
  registering step has an identical makespan under `on` and `off`, and an
  identical GOAL digest list.

The two classes are not added. Six exact-oracle rows and three behavioral
families over seven instances are separate counts.

## Physical sanity, checked against bounds stated before the run

**`local-tp2` compute floor.** The frozen geometry is about 6.5 G parameters,
so a 512-token prefill costs about 6.64 TFLOP, halved by tensor parallelism to
3.32 TFLOP per rank. Against a B100 envelope derated to 0.7 that is a floor
near 2.6 ms. The measured kernel component is 5,317,760,000 ps, i.e. 5.32 ms,
which sits above its floor by a factor near two. That gap is the roofline
provider charging attention, the LM head and memory-bound terms on top of the
dense GEMM floor, and it is the right side of the bound.

**`local-tp2` serialization.** Each all-reduce moves 512 x 4096 x 2 bytes,
which is 4,194,304 payload bytes; a two-rank ring puts the same 4,194,304
bytes through each endpoint. At the declared 450 GB/s NVLink rate that is
9.32 us per collective, and 64 collectives give 596.6 us. The measured NVLink
component is 596,608,000 ps. The model reproduces its own closed form exactly,
and the closed form is the one physics implies.

**`mixed-tp4` fabric.** Each of the 24 backend runs carries one 32,768-byte
flow. At 400 Gb/s that is 0.655 us of serialization on top of the fluid
profile's 2.000 us propagation reference, so 2.655 us per run and 63.7 us for
24. The measured fabric component is 63,728,640 ps, i.e. 2.655 us per run.
Again exact against the physical decomposition rather than against the
simulator's own previous number.

**The scaling check.** Doubling the channel count doubled the charge exactly,
from 1,280,000,000 to 2,560,000,000 ps, and moved nothing else: the decode
makespans, the TPOT and the GOAL digests are identical across `off`, `on` and
`on-2ch`. A relation that moved by 1.05 or by 40 would have refuted the model
regardless of how exactly O1 matched, which is why the second measurement was
taken.

**Plausibility against the real system.** In `local-tp2` the registration adds
1.28 ms to a 5.91 ms TTFT, a 21.6 percent increase, which is the "visible
minority of TTFT" the freeze predicted. In `mixed-tp4` it adds 80 us to an
87.5 us TTFT and nearly doubles it. The freeze called that in advance and said
why: `mixed-tp4` is a two-layer toy whose real work is microseconds, so the
comparison is a statement about the size of that cell and not evidence about a
production deployment. Real NCCL registers user buffers around communicator
setup and the whole setup costs tens of milliseconds, so a per-buffer cost of
tens of microseconds is the right order of magnitude. It remains a declared
constant and not a measurement.

**The decode side.** `local-tp2` TPOT is 2,409,008,000 ps, i.e. 415 tokens per
second for one request. Weight traffic per rank per decode step is about
6.5 GB, which against a B100-class HBM bound is a floor near 0.8 ms, so the
measured 2.41 ms is above its floor by about three. The number is on the
optimistic side of a real 7B deployment but inside the physical envelope, and
no scored relation depends on it.

## Reproduction, post-specified

The freeze did not ask for this, so it is recorded as a post-specified check
rather than as a scored relation. The study was executed a second time into a
fresh working directory, with the same backend binaries, and the two
`results.json` documents are byte-identical: every makespan, every TTFT, every
fabric term decided by one of the 48 real `htsim_rnic` subprocesses, and every
registration charge. That is the constant-completion half of the interim
contract observed directly rather than asserted: for a given traffic and a
given fabric state, nothing in this chain draws a different number twice.

## What this run does not establish

- **The cost is not calibrated.** 20,000,000 ps is configuration. The ABI
  recorded in `docs/papers/amd-gpu-fabric.md` establishes that a registration
  exists at the plugin seam, happens once per buffer and precedes the
  transfer; it establishes nothing about how long one takes. `TRAF-56` is the
  calibration and `calibrated_cost_ps` refuses to serve the declared number in
  the meantime.
- **No packetized handshake.** The charge is a serialized constant. It occupies
  no link, contends with nothing and appears in no GOAL artifact. `TRAF-55`
  makes it port traffic; `TRAF-54` packetizes the collective path it would run
  over; `TRAF-57` makes that path port-kind independent across NVLink, xGMI and
  UALink.
- **No live new-peer re-registration.** Both cells have a fixed rank set, so
  the new-peer event is covered only by the by-construction ledger guard G9.
  A composed run whose participant set changes mid-replay would be the live
  form and is not in this study.
- **No deregistration, no cache eviction, no size dependence.** The cost is
  flat in buffer size, which is certainly wrong for a mechanism dominated by
  page pinning, and is one of the things TRAF-56 must measure rather than
  assume.
- **Nothing about whether a no-tail constant completion is the right model.**
  This study implements the interim contract the maintainer ruled and measures
  the registration inside it. The question of whether collective completion
  should be constant at all is what the packetized destiny answers.

## Two things that were at genuine risk

The freeze asked the entailment question per relation, and two of the answers
were tested by the implementation rather than assumed.

First, the additive composition. The artifact composition already contained a
`max(local_service, fabric_transport)` term, and the obvious way to add a
registration is inside it, where it would vanish whenever the fabric term is
larger. `mixed-tp4` has a fabric term of 63.7 us and a charge of 80 us, so
that mistake would have shown as a 16.3 us delta instead of 80. B2 is the
relation that detects it.

Second, conservation. The attribution partition asserts that the executed
artifact services reproduce the step makespan exactly, and it rejected the
first version of this change until `MediumAttribution` gained a named
registration component and `_artifact_ownership` learned the new identity.
That rejection is the reason the registration is reported apart from
`collective_base_ps` rather than folded into it: a once-per-identity setup
cost and a per-call fixed cost are different claims and a reader must be able
to tell them apart.
