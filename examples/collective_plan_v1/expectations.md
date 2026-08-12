# Collective plan v1 expectations

This expectations-only record freezes the TRAF-14 qualification before the
immutable collective plan, its wire form, or its runtime consumer is
implemented. The study is a mechanism qualification. It does not claim that
the coarse runtime is calibrated to a current GPU or collective library.

## Source audit and authority boundary

The evidence was authored against SimLLM revision
`76223875557a552deb5aa2c2c529a07f000135ba`. The run will record the revision
it observes separately. No equality between that observed revision and a live
submodule pin is assumed or tested.

No third-party simulator source or initialized submodule is needed. The
pre-freeze source audit found these independent in-repository surfaces:

- `simllm/traffic/patterns.py:84-169` defines the accepted ring expansion as
  `2(W-1)` rounds, one successor message per rank and round, chunk size
  `max(1, payload // W)`, consecutive tags, and two rank-local dependency
  edges per rank after the first round.
- `simllm/traffic/patterns.py:173-235` defines the accepted source-major
  positive pair expansion and preserves request partitions on one physical
  message per ordered pair.
- `simllm/core/runtime.py:839-878` currently allocates collective tags from
  semantic work, while `simllm/core/runtime.py:1792-1964` independently
  reconstructs ring chunks, successors, rounds and pairwise extents. In
  particular, line 1793 uses `payload // W` without the traffic pattern's
  one-byte floor.
- `simllm/traffic/execution_goal.py:160-173` states the accepted empty sparse
  collective frontier and tag order, while lines 286-330 reconstruct the two
  supported patterns for graph rendering.
- `simllm/core/runtime.py:90-94` is the whole-picosecond serialization formula
  used for the physical bounds below.
- `tests/test_execution_io.py:212-230` pins the accepted absent-plan v1 graph
  at 559 bytes with SHA-256
  `f4a5a70f5bd4a0c2fed874baa88f3035266a54f386a59927e115872c2bcff0a3`.

The new immutable record must be authored in `simllm.traffic`, carried by
`ExecutionGraph`, and consumed as data by the explicit-plan runtime path. The
runtime may arbitrate and schedule declared extents. It may not infer a
successor, chunk, round, tag or sparse ordered pair in that path. The existing
semantic reconstruction remains only when the graph has no explicit plan.

## Frozen configurations

Ring ranks are `(0, 8)` at world size two and `(0, 8, 16, 24)` at world size
four. These ranks place every source on a distinct coarse RNIC. Payloads are
3, 4 and 4,096 bytes. Link rates are 200 and 400 Gbit/s. Every standalone ring
starts at tag 1,000.

For world size `W`, the GOAL oracle predicts:

```text
rounds = 2 * (W - 1)
chunk_bytes = max(1, payload_bytes // W)
message_count = W * rounds
directed_bytes = message_count * chunk_bytes
internal_dependency_count = 2 * W * (rounds - 1)
tags = range(1000, 1000 + rounds)
```

The complete exact ring registry is:

| W | Payload bytes | Rounds | Chunk bytes | Messages | Directed bytes | Internal dependencies |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3 | 2 | 1 | 4 | 4 | 4 |
| 2 | 4 | 2 | 2 | 4 | 8 | 4 |
| 2 | 4,096 | 2 | 2,048 | 4 | 8,192 | 4 |
| 4 | 3 | 6 | 1 | 24 | 24 | 40 |
| 4 | 4 | 6 | 1 | 24 | 24 | 40 |
| 4 | 4,096 | 6 | 1,024 | 24 | 24,576 | 40 |

The routed sparse registry uses ranks `(0, 8, 16, 24)` and three semantic
collectives:

| Case | Positive ordered pairs | Directed bytes | Required idle ranks |
|---|---|---:|---|
| dispatch | `(0,8,3)`, `(0,16,5)` | 8 | 24 |
| combine | `(8,0,3)`, `(16,0,5)` | 8 | 24 |
| all local | none | 0 | 0, 8, 16, 24 |

Dispatch request partitions are `alpha` on `(0,8)` for 3 bytes and `beta` on
`(0,16)` for 5 bytes. The pairwise plan has exactly one semantic round even
when that round contains no positive extent. When combined after one
four-rank ring, dispatch, combine and the empty semantic collective receive
tags 1,006, 1,007 and 1,008 after ring tags 1,000 through 1,005.

## Decision-relevant perturbations

The negative-control family is evaluated before any exact plan-to-GOAL
oracle. Both changes conserve total bytes and would be silently absorbed by
the current runtime reconstruction:

1. Change one frozen plan tag while retaining the immutable plan's original
   integrity identity. Graph validation and runtime preflight must reject it
   before any WQE is submitted.
2. Change semantic ring rank order from `(0, 8, 16, 24)` to
   `(0, 16, 8, 24)` while retaining the original plan. The participant set
   and total bytes stay constant, but validation must reject the disagreement
   before any WQE is submitted.

These are two genuine-risk behavioral instances. A runtime that ignores the
plan and reconstructs from semantic work reaches execution instead, so either
instance can fail independently of the later exact oracle.

## Live metric relation

The 3-byte, four-rank ring is the drift sentinel. The accepted GOAL pattern
has six rounds of one-byte messages. The current absent-plan compatibility
runtime floors `3 // 4` to zero bytes. With all non-network service constants
zero, distinct source RNICs and no propagation term, the explicit plan must
therefore produce these raw relations before any exact row is checked:

| Rate | Explicit-plan step latency | Absent-plan step latency | Signed explicit minus absent |
|---:|---:|---:|---:|
| 400 Gbit/s | 120 ps | 0 ps | +120 ps |
| 200 Gbit/s | 240 ps | 0 ps | +240 ps |

Three consecutive request steps, one prefill then two decode, pass through
`CoarseDeviceRuntime`, `CompletionEvent`, `RuntimeReport`,
`CompletionReducer`, `StepResult`, TTFT and TPOT. The explicit-plan arm must
increase TTFT and TPOT by exactly 120 ps at 400 Gbit/s and 240 ps at
200 Gbit/s relative to the absent-plan arm. These four metric instances form
the second genuine-risk family. The 200 Gbit/s explicit metric must be exactly
twice the 400 Gbit/s metric; this scaling check is fatal-unscored because the
four exact signed metric instances already pin it.

For every divisible ring cell and both link rates, explicit-plan and
absent-plan runtime WQE rows, completion order and serial timing must be
identical. Those identity rows are fatal-unscored. They do not enter the
behavioral fraction.

## Physical sanity bounds

Before reading a runtime value, the per-ring floor is the bytes sent by one
source over its link rate. The ceiling serializes all directed bytes over one
link. Integer ceilings are applied once per message, exactly as the coarse
runtime does.

For the 3-byte, four-rank sentinel, the explicit-plan floor and predicted
completion are 120 ps at 400 Gbit/s and 240 ps at 200 Gbit/s. The corresponding
single-link ceilings are 480 ps and 960 ps. A zero explicit-plan result is
below the floor and proves reconstruction or byte loss.

For the 4,096-byte cells at 400 Gbit/s, the two-rank floor and prediction are
81,920 ps with a 163,840 ps ceiling. The four-rank floor and prediction are
122,880 ps with a 491,520 ps ceiling. Every value doubles at 200 Gbit/s.

The sparse dispatch has an 8-byte source-egress floor, 160 ps at 400 Gbit/s
and 320 ps at 200 Gbit/s, which is also its predicted completion. Its
single-link ceilings are identical because all bytes share one source. The
many-source combine is structural evidence only: the current coarse model has
no destination-ingress serializer, so its timing cannot be accepted as a
physical precision oracle. The empty sparse collective has no physical
service; its zero is by construction and unscored.

Three independent angles are retained:

1. serialization physics bounds every measured runtime value;
2. structured GOAL messages and dependencies independently identify bytes,
   tags, rank order and causal rounds;
3. live TTFT and TPOT prove that the selected plan reaches the reported metric
   chain rather than only a component probe.

The tiny sentinel is deliberately not compared with a deployed LLM. It
qualifies authority and conservation only. TRAF-11 retains collective-service
calibration against real hardware.

## Exact and fatal evidence

The plan is compared with structured output from the existing
`ring_allreduce` and `pairwise_all_to_allv` functions, not a second text
parser. For every registered cell, exact message rows include operation ID,
round, source, destination, payload, tag and request partition. Exact
dependency rows preserve action role, participant rank, relation and
predecessor round. The plan and its canonical JSON must round-trip exactly.

The following are fatal-unscored guards. Any failure makes the run void and
no behavioral fraction is reported:

- every explicit plan has one immutable integrity identity and exactly covers
  the graph's collective operations;
- plan semantic identity, rank order, algorithms, actions, extents, tags,
  request partitions, entry actions and terminal frontiers are lossless;
- all ring and sparse rows match the existing GOAL pattern exactly;
- the empty sparse plan retains all four participant frontiers and creates no
  WQE or invented traffic;
- explicit-plan runtime WQE rows equal plan extents exactly, with no loss,
  duplication or tag change;
- divisible explicit and absent arms have exact timing identity;
- the absent-plan graph remains 559 bytes with the frozen SHA-256, omits the
  optional plan field, round-trips exactly, and retains its frozen timing;
- configuration echoes, event order, attribution, quiescence and output
  provenance conserve exactly;
- the check-only path creates no output directory or artifact.

Author-defined ranks, payloads, sparse tables, rates and expected counts are
configuration rows, not evidence. Integrity, round-trip, conservation,
identity and zero-work assertions are fatal but unscored.

## Entailment analysis

The runner must evaluate the two perturbation outcomes and four signed live
metric changes from raw validation and runtime observations before checking
any fixed plan, GOAL, wire, WQE or timing oracle. No earlier fatal check pins a
scored quantity. The later exact rows reuse some values for regression and do
not count again. The registered headline is therefore two families and six
genuine-risk instances, not the sum of all exact and fatal rows.

## Registered TRAF-14 acceptance clauses

The result report must quote and map each clause separately:

1. "move ring-round and pairwise-extent expansion from the coarse runtime's
   current semantic-work surrogate into one immutable traffic-owned collective
   plan carried through `ExecutionGraph`";
2. "The runtime may schedule those extents but may not choose or reconstruct
   their algorithm, chunk sizes, rank order or tags";
3. "Compare the plan against the existing GOAL pattern expansion over payload,
   world-size and routed sparse-pair sweeps with exact byte, round, dependency
   and tag conservation";
4. "The absent explicit plan must preserve the accepted v1 wire bytes and
   serial timing exactly";
5. routed dispatch has one source, combine has many sources converging on one
   destination, and idle ranks remain idle;
6. a zero-byte semantic collective with every destination local preserves all
   rank frontiers without inventing traffic.

If any clause is not demonstrated, TRAF-14 stays open or the residual moves to
TRAF-28, TRAF-29 or CORE-48 with the required category, priority and difficulty
tag. A fatal failure makes the whole run void rather than one failed point.

## Registered command and check-only dry run

The production command is:

```bash
.venv/bin/python examples/collective_plan_v1/run_study.py \
  --out "${SIMLLM_COLLECTIVE_PLAN_RUN_ROOT:?configure SIMLLM_COLLECTIVE_PLAN_RUN_ROOT}"
```

Before this expectations-only commit, the same command is run with
`--check-only`. That path parses the complete CLI and validates only frozen
literal shape, arithmetic, physical bounds and evidence-class counts. It
imports no SimLLM implementation, reads no input, invokes no native binary and
creates no output artifact.
