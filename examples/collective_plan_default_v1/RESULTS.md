# Collective plan lowering default results

## Outcome

The corrected run **passed**: every fatal guard held and all 20 registered
scored instances in all 4 families passed. TRAF-28 closes on this run.

The first result-producing run is **void**. Its record is below, unmodified,
and it publishes no behavioral fraction.

| Evidence class | Void run | Corrected run |
|---|---|---|
| Fatal guards | physical bounds violated | 3 of 3 groups held |
| Scored instances | withheld (18 of 20 retained as findings) | 20 of 20 |
| Scored families | withheld (3 of 4 retained as findings) | 4 of 4 |

## Chronology

| Event | Commit |
|---|---|
| Expectations-only freeze | `68d8a8d` |
| Lowering default implementation | `2befccc` |
| Void run, observed revision | `2befccc1df01d20baffde5f646ea87f781aa2989` |
| Transport refreeze, expectations only | `eff0a94` |
| Corrected run, observed revision | `eff0a9423bdb7a1d36aa51274c2ed866776b0a55` |

Both the original and the refrozen commands passed `--check-only` before their
commits and produced no artifacts. The corrected run wrote to a separate output
directory.

## The void run and what it refuted

The frozen bound charged every directed collective byte of the live arm to the
swept `rnic_rate_bps`. The live arm placed the eight-rank group at semantic
ranks 0 through 7. `CoarseDeviceProfile` maps a global rank to
`(node, gpu) = divmod(rank, 8)`, so all eight ranks sat on node 0, and
`CoarseDeviceRuntime` serves a same-node semantic send over NVLink at a fixed
900 Gbit/s that the sweep never varied.

| Refuted fact | Frozen | Observed |
|---|---:|---:|
| Prefill floor at 400 Gbit/s | 656,719,680 ps | 370,655,040 ps measured |
| Prefill inverse-rate ratio | [1.95, 2.05] | 1.0 |
| Decode inverse-rate ratio | [1.95, 2.05] | 1.0 |

Both failures have the same single cause. Against the transport the model
actually used, the 27,869,184-byte peak endpoint load costs 247,726,080 ps and
the measured network term of 271,319,040 ps sits above it, so the mechanism was
never in question: the bound was.

The void run's other families were evaluated from raw records before the bound
was consulted, and their truth values are retained as findings rather than as a
score: default and bypass runtime identity held on all 8 instances,
perturbation rejection held on all 6, and surrogate unreachability held on all
4. Every exact oracle E1 through E6 held. A void run closes nothing regardless.

## Corrected live arm

The refreeze pinned the live placement to one rank per node,
`(0, 8, 16, 24, 32, 40, 48, 56)`, which is the standing reference
configuration, and made the bound charge each directed extent to the link the
model selects for it. With that placement every extent is cross-node, so the
NVLink terms are zero.

| Mode | Gbit/s | TTFT ps | TPOT ps |
|---|---:|---:|---:|
| Default | 200 | 1,320,271,680 | 166,709,760 |
| Default | 400 | 709,803,840 | 132,794,880 |
| Bypass | 200 | 1,320,271,680 | 166,709,760 |
| Bypass | 400 | 709,803,840 | 132,794,880 |

The replayed step carries 5,712 directed messages, 111,476,736 fabric bytes on
the 54-token prefill and 6,193,152 on each decode step, all cross-node.

| Cell | Latency ps | Floor ps | Ceiling ps |
|---|---:|---:|---:|
| Prefill, 200 Gbit/s | 1,320,271,680 | 1,214,103,360 | 4,564,117,440 |
| Prefill, 400 Gbit/s | 709,803,840 | 656,719,680 | 2,334,582,720 |
| Decode, 200 Gbit/s | 166,709,760 | 160,811,520 | 352,318,080 |
| Decode, 400 Gbit/s | 132,794,880 | 129,845,760 | 228,455,040 |

Every cell sits above its floor by 3.7 to 8.7 percent and far below its
ceiling. The prefill sits close to the floor because the coarse runtime's
critical rank is exactly the one the floor charges.

## Scored relations

**A, default and bypass runtime identity: 8 of 8.** Two rates times two
tensor-parallel widths times two lowering paths. For every cell the completion
time, the quiescence time, the full `CompletionEvent` tuple and every WQE tuple
(operation, source, destination, payload, tag, channel, and all five
timestamps) were identical between the planned and the bypassed graph. The
explicit-plan scheduler and the absent-plan reconstruction are separate
implementations, so this equality is a result rather than a construction.

**B, perturbation rejection on the default path: 6 of 6.**

| Perturbation | Lowering path | Outcome |
|---|---|---|
| One plan round tag changed | serial, observed | rejected: `collective plan integrity mismatch`, zero work requests |
| Semantic rank order `(0,8,16,24)` to `(0,16,8,24)` | serial, observed | rejected: `rank order disagrees with semantic work`, zero work requests |
| Same rank-order change on the bypass graph | serial, observed | absorbed silently at unchanged 196,608 bytes and unchanged 4,730,040 ps |

The absorption control is the point of the family: the surrogate cannot see a
byte-conserving rank-order change, and the default path now can.

**C, surrogate unreachability: 4 of 4.** With
`CoarseDeviceRuntime._schedule_collective` replaced by a sentinel that raises
whenever it is entered without a plan, both default-path cells executed to
completion and both bypass cells raised
`absent-plan collective reconstruction was reached`.

**D, live inverse-rate relation: 2 of 2.** Both ratios came out at exactly
2.0000 against the registered `[1.95, 2.05]` band:

| Step | 200 Gbit/s network ps | 400 Gbit/s network ps | Ratio |
|---|---:|---:|---:|
| Prefill | 1,220,935,680 | 610,467,840 | 2.0000 |
| Decode | 67,829,760 | 33,914,880 | 2.0000 |

## Fatal guards, all held

- **E1 coverage.** Every default-lowered graph carried one plan per collective
  operation, four of four in each of the four lowering configurations.
- **E2 bypass emptiness.** Every bypass graph had `collective_plans == ()` and
  its v1 wire JSON omitted the key.
- **E3 plan is the only difference.**
  `replace(default, collective_plans=()) == bypass` in every configuration.
- **E4 equivalence and idempotence.**
  `plan_execution_graph_collectives(bypass) == default`, and attaching to an
  already planned graph returns it unchanged.
- **E5 integrity.** Every attached plan's canonical SHA-256 equalled its
  recomputed identity.
- **E6 legacy wire anchor.** The accepted absent-plan graph still serializes to
  559 bytes with SHA-256
  `f4a5a70f5bd4a0c2fed874baa88f3035266a54f386a59927e115872c2bcff0a3`, still
  omits the plan field and still round-trips.
- **E7 physical bounds.** All 12 live cells inside their transport-aware floors
  and ceilings.
- **Live metric identity.** Default and bypass TTFT, TPOT and every step
  latency agreed exactly at both rates.

## Can the runtime reconstruction be deleted

Not yet, and the reason is the compatibility clause of this very task rather
than any gap in the evidence.

Family C establishes that no default-path lowering reaches the absent-plan
branch: with the branch made fatal, every default cell completes. What still
depends on it:

1. **The explicit bypass.** `attach_collective_plan=False` exists precisely to
   preserve the accepted 559-byte absent-plan wire form and its serial timing,
   and the bypass path is that branch. Deleting the branch means retiring the
   bypass, which this task's own acceptance clause requires to stay.
2. **Deserialized v1 graphs.** A graph read back from an accepted v1 wire
   record that predates the plan has no `collective_plans` field, so it reaches
   the branch. Every archived artifact in this repository is such a graph.
3. **Directly constructed graphs.** `ExecutionGraph` is a public type and
   several tests and studies build collectives without going through a lowerer.
   Those graphs are unplanned by construction.

The honest statement is therefore: the reconstruction is now dead code on the
production path and live code on the compatibility path. Deleting it becomes a
separate decision that must first retire the absent-plan wire form, and that
decision is not claimed here. This is recorded as prose rather than a new task
because no registered acceptance clause asked for the deletion itself, only for
the statement about it.

## Physical sanity

Floor and ceiling for a live cell are

```text
floor_ps   = compute_ps + max over endpoints of the busier direction, charged
             to the link the model selects for it
ceiling_ps = compute_ps + every fabric byte serialized once + every NVLink byte
             serialized once + 1 ns per message
```

Represented compute is 99,336,000 ps on the prefill step and 98,880,000 ps on
each decode step, read from the lowered graph's own `ComputeWork` nominal
durations rather than a literal. The 400 Gbit/s prefill spends 610,467,840 ps
of its 709,803,840 ps on the network, which is what a 111 MB all-remote
expert-parallel step should look like at that rate. The decode steps carry
1/18 of the prefill's bytes and land at 33,914,880 ps of network, a ratio of
18.0 against the 18.0 byte ratio.

## Registered acceptance clauses

- *"Attach the plan in `SerialStepLowerer` and `lower_step_observations`"*:
  **demonstrated**. Both paths carry it by default and E1 checks coverage.
- *"keep an explicit bypass that preserves the accepted 559-byte v1 wire form
  and its serial timing"*: **demonstrated** by E2, E3, E6 and family A, and
  pinned by tests in `tests/test_step_lowerer.py` and
  `tests/test_compute_comm_overlap.py`.
- *"requalify with the TRAF-14 perturbation family plus a live TTFT and TPOT
  arm on a real replayed step rather than the tiny sentinel"*:
  **demonstrated** by families B and D on the 54-token, 24-layer replayed
  Granite step.
- *"Acceptance must show the runtime reconstruction unreachable on the default
  path and removable without changing any accepted number"*: **partially
  demonstrated**. Unreachability is shown by family C. Removability is not
  claimed: the branch is still the implementation of the explicit bypass this
  task requires, and of every deserialized v1 graph. The section above
  enumerates what still depends on it.

Zero new IDs were registered. The one clause that is not fully demonstrated is
the removability half, and it is not an undemonstrated deferral: the same
registry entry requires the bypass that keeps the branch alive, so a deletion
task would contradict the clause that created the bypass. The statement the
clause asks for is given above.

## Verification and contradiction sweep

`ruff check .` passed and the full suite reported 1,213 passed with 7
environment-dependent skips, 14 more than before this change.

The sweep of `README.md`, `docs/README_PRO.md` and `docs/architecture.md` found
no statement contradicted by this result. None of the three names the runtime's
collective reconstruction or claims that the plan is opt-in, so none needs an
edit and this branch makes none.
