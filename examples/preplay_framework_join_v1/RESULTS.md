# Framework trace v2 join results

Contract: [expectations.md](expectations.md), frozen before the version 2 join
existed and before the first cell ran.

## Chronology

1. Expectations commit, with the accepted-path baselines captured from the
   code as it then stood and the pytest that locks them. The registered
   command had already been dry run with `--check-only` against an
   argument-parser skeleton that resolved and hash-checked all three inputs and
   executed no cell; that skeleton was never committed.
2. Implementation commit: the version 2 join, the KV reconciliation record, the
   schema-dispatched oracle reader behind the vLLM replay seam, the joined
   version 2 routing projection, and their tests. No expectation changed.
3. One study run, on the sixteen frozen cells. Nothing was re-run and no
   expectation, bound, constant or policy was touched after it.

No commit after the first measured run changes a modeled behavior. The only
post-run changes are this results file and the module-doc and ledger updates
that close the task.

## Headline

- 24 of 24 exact-oracle relations passed.
- 62 of 63 registered scored instances passed. The single failure is KV5 on the
  `preempt` cell, where the frozen prediction was refuted in the capture's
  favor: see below.
- 0 fatal guard violations, so the behavioral result is interpretable.

## Physical sanity, checked against the frozen bounds

The frozen floor for one step was `855,638,016 / 8.0e12` equals `107 us` on the
`b100` envelope, and the frozen ceiling was `W / 5.6e12` plus the KV term, i.e.
about `476.4 us`. The measured first step of the `short` cell is
`476,423,314 ps`, which is `476.42 us`: above the model-independent floor by a
factor of 4.5 and exactly at the modeled ceiling, which is where the freeze
said it had to sit because this geometry streams all 32 resident experts every
step. The hand-derived closed form predicts `int(476,353,097.14 +
8,777.14 * 8)` equals `476,423,314 ps` for that step, matching to the
picosecond. The `long` cell's 96-token prefill measures `477,195,702 ps`,
against `int(476,353,097.14 + 8,777.14 * 96)` equals `477,195,702 ps`.

Every one of the 392 step records across all sixteen runs was classified
memory bound by the roofline, so the frozen closed form applies everywhere and
guard G1 held. The 96-token prefill, the largest step in the sweep, needs
`7.3e10 FLOP` against the `5.5e11 FLOP` the memory roof pays for.

The absolute latency remains what the freeze said it was: optimistic. The
configuration has no host launch cost and no collective, and it over-streams
expert weights by four times against a batch-1 top-k-8 kernel. A real Granite
3.0 1B A400M decode step on a single modern GPU is roughly 1 ms to 2 ms, so the
modeled `476 us` is about two to four times fast. No absolute claim is made,
and every scored relation is a ratio or a difference inside one cost model.

## What the join does to TTFT and TPOT

`preempt4` on the `b100` envelope, all times in picoseconds:

| request | admitted (declared) | admitted (joined) | TTFT declared | TTFT joined | ratio |
|---|---|---|---|---|---|
| `p0` to `p3` | 0,1,2,3 | 0,1,2,3 | 476,423,314 to 1,906,202,330 | identical | 1.000000 |
| `p4` | 24 | 20 | 11,924,520,950 | 10,015,071,077 | 0.839872 |
| `p5` | 25 | 21 | 12,401,558,664 | 10,492,038,574 | 0.846026 |
| `p6` | 26 | 22 | 12,878,420,835 | 10,968,865,636 | 0.851724 |
| `p7` | 27 | 23 | 13,355,107,463 | 11,445,552,264 | 0.857017 |

Every ratio lands inside its frozen band, and the four bands were the exact
step-count ratios 21/25, 22/26, 23/27 and 24/28 plus or minus 0.5 percent.
`p4` starts `1,909,449,873 ps` earlier, which is 1.909 ms, i.e. the four steps
the declared arm spends finishing `p0` before a slot frees. The whole
`preempt4` makespan drops from 24.327 ms to 20.508 ms, exactly the eight steps
the frozen schedule table predicted (51 against 43).

TPOT moved almost not at all, in the predicted direction, in every one of the
36 registered instances: the largest relative change is 0.035 percent
(`preempt8` `p7`, ratio 0.999654) and the smallest is 0.004 percent (`short`,
ratio 0.999963). This is the physically interesting half of the result. Because
the modeled step is dominated by streaming 2.567 GB of resident weights, an
inter-token interval barely depends on how many tokens the step carries, so
pinning an output length moves a later request's TTFT by whole steps while
leaving the inter-token rate essentially untouched. The frozen 1 percent bound
was therefore not tight: the measurements sit about 30 times inside it, and a
reviewer should read B2 as confirming the direction and the order of magnitude
rather than as a demanding test.

Every request admitted at the same step under both treatments kept its TTFT to
the exact picosecond, across all 28 such request and envelope pairs (guard G4).
The join pins lengths and changes nothing about a first token that its own
length cannot reach, which is the defect detector the task asked for and which
found no defect.

## The closed form against the live chain

Relation E4 compares each cell's measured step-latency series against an
independent recomputation: the admission steps from the closed-form recurrence,
the running set and context sum from the frozen policy, and the duration from
the frozen byte constants, all in plain arithmetic that never calls
`step_kernel` or the schedule generator. All 376 scheduled-step latencies over
the sixteen runs matched exactly. That is a real cross-check on two things at
once: the hand-derived attention, resident-MoE and LM-head byte counts agree
with the module's own geometry arithmetic to the byte, and the generated
schedule's step compositions agree with the closed-form running set.

The frozen admission steps and step counts were also checked as a fatal guard
and matched literally in all eight cell and treatment pairs.

## KV reconciliation

The captures reconcile with their own oracle records almost perfectly.

| cell | requests | allocated per request | agreement | peak occupancy | capacity |
|---|---|---|---|---|---|
| `short` | `s0` | 11 | exact | 11 | 256 |
| `long` | `l0` | 103 | exact | 103 | 256 |
| `preempt` | `p0`, `p1`, `p2`, `p4` to `p7` | 27 | exact | 96 | 96 |
| `preempt` | `p3` | 51 | recompute surplus | | |

- KV1 passed on the scored `preempt` cell: all seven requests with neither a
  preemption nor a prefix hit allocated exactly `8 + 20 - 1` equals 27 tokens,
  the same number of forward passes the capture records as dispatch rows. The
  `short` and `long` values (11 and 103) are reported as post-specified
  confirmation, for the reason disclosed in the freeze.
- KV2 passed: `p3`, the one preempted request, allocated 51 tokens and released
  24. That is exactly the admissible recompute pattern the freeze described: it
  gave back its 24 live tokens on preemption and re-allocated an 8-token prompt
  plus 16 decode slots to catch up, so 27 plus 24.
- KV3 passed in all three cells. The one preemption event names `p3`, and `p3`
  is the one request reporting `framework_preemption_count` equal to 1.
- KV4 passed in all three cells. Every one of the 14 prefix-hit events in the
  `preempt` capture carries a zero token count, agreeing with every request's
  reported `framework_cached_tokens` of zero.
- KV5 failed on `preempt`, and the failure is the frozen prediction being
  refuted rather than a defect. The freeze predicted that reconstructed
  occupancy would exceed the 96-token pool, because eight requests allocate
  about 216 tokens and the capture carries only one release event. It does not:
  the peak is exactly 96 tokens, at capacity and never above it. The ten
  eviction events, all unattributed and therefore admissible by the frozen
  rule, remove 135 tokens between them, and `p3`'s single release removes 24.
  The SGLang version 2 capture does record enough free and eviction
  observations to reconstruct pool occupancy under pressure, and the
  reconstruction saturates the declared capacity exactly. No schema capability
  gap is reported, because none was found.
- No defect of any code fired in any cell.

One mechanism note, not registered because no clause claimed it: the `preempt`
capture ends with 81 tokens still live, since it records no terminal release
for the requests that ran to completion. Occupancy can therefore be
reconstructed and bounded but not shown to drain, and a future capture that
wants a clean drain check would have to record the terminal frees.

## Authority

The KV event stream never decides anything. Rewriting each capture with an
empty KV event list and re-joining left every joined request field identical
(identity, arrival, output length, stop reason, output token IDs, routing
reference request ID, bookkeeping object ID) in all three captures, and only
the trace hash and the reconciliation record changed. The reconciliation is a
read-only projection with no mutable state, and a defect in it is reported
rather than raised, precisely because a stream with no authority cannot
invalidate an output length the same capture states outright.

The declared arm provably ran the accepted absent-replay path rather than a
silently disabled replay: it enters the adapter with `max_tokens` equal to the
declared length (24, 8 and 16) while the oracle lengths are 20, 4 and 8, and
the replay seam rejects any admission whose `max_tokens` differs from its
oracle length. Had replay been engaged there, every cell would have raised.

## Evidence classes and the genuine-risk fraction

Counts from different classes are not added.

- Exact oracle, 24 of 24: sixteen E4 closed-form series (376 step latencies),
  three E2 join bindings, three E3 routing projections (264, 2,472 and 5,184
  token-layer rows copied in capture order), and two post-specified KV1
  confirmations.
- Scored behavioral, 62 of 63 registered instances.
- Fatal unscored, 0 violations: G1 over 392 step records, G2 in pytest against
  the tracked baselines, G3 over every joined request in every joined cell, G4
  over 28 request and envelope pairs, G5 to G8 in pytest, plus the frozen
  admission and step-count table checked as a guard.

The registered 63 overstates independence, and the honest subset is smaller:

- the `h200` arm reproduces the `b100` arm exactly at 5/3 once G1 holds, so its
  eight B1 and eighteen B2 instances add no independent risk;
- B3 (8 instances) is close to entailed by G1 for the same reason. Its only
  residual content is that integer truncation does not accumulate past one part
  in a million, which it did not (worst deviation 2.1e-9). A reviewer should
  discount B3.

Dropping the duplicated envelope and B3 leaves 33 genuinely independent scored
instances, of which 32 passed: KV1 (1), KV2 (1), KV3 (3), KV4 (3), KV5 (3, one
failed), B1 (4) and B2 (18).

## What this does and does not establish

Established: a serving framework's own version 2 capture pins each replayed
request's arrival, output length, stop reason, output token IDs and expert
routing through the accepted replay seam; the framework scheduler remains the
only KV authority and its event stream is evidence that reconciles with its own
oracle record under a stated operational rule; and the accepted version 1 join
and absent-replay paths are byte- and timestamp-identical, proved in pytest
against baselines captured before the implementation existed.

Not established, and not claimed:

- SGLang replay, which is PLAY-7. These are SGLang captures consumed through
  the vLLM replay seam, which is what shows the run record is framework
  neutral; nothing under the SGLang adapter changed.
- A real serving scheduler choosing the schedule. The step schedule comes from
  the registered deterministic policy in the freeze. The evidence that a real
  vLLM scheduler drives this same seam and changes scheduler-visible completion
  is PLAY-3's, and the object it consumes is the same
  `simllm-preplay-replay-run-v1` record; only the named trace schema differs.
- Any fabric effect. This study runs one rank with no collective, so the
  captured routing reaches the routed-expert projection but not a packet-level
  fabric. That chain is PLAY-4, PLAY-5 and PLAY-11's.
- Any absolute-latency claim, for the reasons in the physical-sanity section.
- A routing arena over a version 2 capture. The arena stays a version 1 form
  and the version 2 join offers no arena parameter, so nothing is deferred
  behind a flag; a version 2 arena would be new work with its own contract.
