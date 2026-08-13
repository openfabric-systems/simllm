# Framework trace v2 join expectations

This document freezes the PLAY-8 validation contract before the version 2 join
exists and before any study cell runs. PLAY-8 is registered as:

> PLAY-8 (Completeness; P1; L): join `simllm-preplay-trace-v2` into the live
> replay path. Bind its observed per-request outputs and expert routing to the
> existing replay identities, retain the framework scheduler as the sole KV
> authority, and reconcile its per-request KV event stream with the oracle
> record. The explicit v1 join and absent-replay paths must remain byte- and
> timestamp-identical when v2 is not selected.

Every clause is mapped below to the evidence that is supposed to carry it.

## Scope and boundary

The joined artifact is consumed through the already accepted vLLM replay seam
(`ReplayTokenSource`, `SimExecutorConfig.replay_run_path`), because that is the
only live replay path on main today. SGLang replay is PLAY-7 and is owned by
other work; nothing under `simllm/adapters/sglang/` changes here. The captures
this study joins were produced by SGLang, so this study also shows that the
joined run record is framework neutral: an SGLang capture drives the vLLM
replay seam without either side learning about the other.

The routing arena (`simllm-routing-arena-index-v1`) stays a version 1 form. The
version 2 join offers no arena parameter at all rather than a stubbed one, so
nothing is deferred behind a flag.

## Frozen inputs

Three tracked-by-hash version 2 captures of
`ibm-granite/granite-3.0-1b-a400m-instruct` at revision
`ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`, greedy sampling, captured by
SGLang `0.0.0.dev1+g8f2a3ad6d` on CPU with `dispatch_layer_mapping` equal to
`framework-layer-id`, 24 MoE layers, 32 experts, top-k 8. Bulk rows stay
outside Git; the study takes each path as a command-line argument and verifies
its SHA-256 before anything else.

| cell key | SHA-256 | requests | prompt tokens | output tokens | stop | KV page, capacity | KV events |
|---|---|---|---|---|---|---|---|
| `short` | `7d979e20516bec19062d55bd4ce5ba6512c71a729c20f45e5f827fb28b6be298` | `s0` | 8 | 4 | length-cap | 1, 256 | 5 |
| `long` | `1c310fae51e98f309e805cd088c5c6869f8a32d9966eeeb601d8cd370184536e` | `l0` | 96 | 8 | length-cap | 1, 256 | 9 |
| `preempt` | `da0096b696564f365003d11565f4cbc5bed33ab47f7236c52b3ca7c989fd4982` | `p0` to `p7` | 8 each | 20 each | length-cap | 1, 96 | 186 |

### Disclosure of what was inspected before this freeze

Freeze integrity requires saying what was already seen. While scoping, the
following were read from the three captures and are therefore inputs, not
outcomes: provenance fields, per-request prompt and output lengths, stop
reasons, `framework_cached_tokens` and `framework_preemption_count`, the KV
event kind histogram per cell (`short`: 4 allocation and 1 prefix-hit;
`long`: 8 allocation and 1 prefix-hit; `preempt`: 160 allocation, 14
prefix-hit, 10 eviction, 1 preemption, 1 release), and the first six KV events
of each cell. Nothing else was read. Consequences, stated once and honored in
the scoring below:

- the per-request allocation identity for `short` and `long` is derivable from
  what was already seen, so relation KV1 is scored on the `preempt` cell only
  and is reported as post-specified confirmation for `short` and `long`;
- no per-request attribution of the `preempt` cell's prefix-hit, eviction,
  release or preemption events was read, so KV2 to KV5 are genuine risk there;
- no output token IDs, no routing rows and no timing were read at all.

## The two joins and what the replay consumes

The version 2 join produces the same `simllm-preplay-replay-run-v1` record as
the version 1 join, over the same replay identities: the trace reference gains
`simllm-preplay-trace-v2` as an accepted schema, and each request's
`simllm-preplay-routing-reference-v1` names that same version 2 artifact. This
is deliberate: the replay seam, the bookkeeping projection, the routed-expert
projection and every downstream consumer keep one run schema, and the trace
schema is the only thing that varies.

The KV event stream is projected into a separate, read-only
`simllm-preplay-kv-reconciliation-v1` record. It is evidence. It is never
consulted by the join to decide an arrival, an output length, a stop reason, a
token ID or a routing row, and no replay decision reads it.

## Reconciliation, stated operationally

For a joined request with `P` prompt tokens and oracle output length `L`, the
capture itself carries exactly `P + L - 1` forwarded-token dispatch rows. That
count is the reference the KV event stream is reconciled against.

Admissible disagreements, i.e. not defects:

- a request with a prefix hit allocates fewer tokens than `P + L - 1`, short by
  at most the hit;
- a request that the framework preempted allocates more than `P + L - 1`,
  because a recomputed request re-allocates slots it already held;
- eviction events that name no request: eviction is a pool-level decision about
  tokens no live request owns;
- `framework_step` absent on admission-time events;
- a capture with no KV events at all, which yields an empty reconciliation and
  no claim.

Defects, i.e. the capture and its own oracle record disagree:

- the number of preemption events naming a request differs from that request's
  `framework_preemption_count`;
- the prefix-hit token total for a request differs from its
  `framework_cached_tokens`;
- a request allocates fewer than `P` tokens, which no framework can do and
  still run the prefill;
- a joined request with no allocation event at all;
- a request with no preemption event and no prefix hit whose allocation total
  is not exactly `P + L - 1`.

## Frozen cost model

One rank, no expert parallelism, no fabric. The step cost is the repository's
analytical roofline over the real Granite geometry, so the numbers have a
physical floor instead of an invented constant:

```
ModelDims(num_layers=24, hidden_size=1024, intermediate_size=512,
          num_heads=16, num_kv_heads=8, head_size=64, vocab_size=49152,
          dtype_bytes=2, num_experts=32, top_k=8,
          moe_intermediate_size=512, local_num_experts=32)
RooflineProvider(efficiency=0.7), HostInitiationModel.ideal()
```

Derived byte counts, all exact integers:

- attention parameters, all layers: `24 * (1024 * 2048 + 1024 * 1024)` equals
  `75,497,472`;
- resident MoE parameters, all layers: `3 * 1024 * 512 * 32 * 24` equals
  `1,207,959,552`;
- weight bytes: `(75,497,472 + 1,207,959,552) * 2` equals `2,566,914,048`;
- LM head bytes: `1024 * 49,152 * 2` equals `100,663,296`;
- constant bytes per step `W`: `2,667,577,344`;
- KV bytes per context token: `2 * 24 * 8 * 64 * 2` equals `49,152`.

With `K(s)` the sum of `context_length` over the requests scheduled in step
`s`, and a memory-bound step, the step latency in picoseconds is exactly

```
T(s) = int( (W + 49152 * K(s)) / (bandwidth * 0.7) * 1e12 )
```

which is `int(476,353,097.142857 + 8,777.142857 * K(s))` on the `b100`
envelope (8.0 TB/s) and `int(793,921,828.571429 + 14,628.571429 * K(s))` on
the `h200` envelope (4.8 TB/s).

## Frozen scheduling policy

The step schedule is not hand-authored per cell. It is generated by one
registered deterministic policy whose only per-treatment input is the serving
length, so declared and joined runs differ in exactly that one variable:

1. every request arrives at virtual time zero and is admitted first come first
   served in capture order;
2. at most one request is admitted per step, and only while the running set is
   below the capacity `B`; an admitted request schedules its whole prompt as
   one unchunked prefill and samples its first token in that same step;
3. every other running request schedules exactly one decode token per step,
   reported with `num_computed_tokens = P + k - 1` and `num_output_tokens = k`
   after `k` produced tokens;
4. a request leaves the running set at the end of the step in which it produces
   its serving-length token, and its identity is reported in
   `finished_req_ids` on the following step, which is vLLM's own attribution
   asymmetry;
5. a final zero-length drain step carries the last completions.

This policy is a registered deterministic stand-in for a serving scheduler, not
a model of one. The evidence that a real vLLM scheduler consumes the same
replay seam and changes scheduler-visible completion is PLAY-3's, and this
study does not restate it.

## Frozen sweep

Four workload cells, two memory-bandwidth envelopes, two treatments, sixteen
runs:

| cell | trace | requests | `P` | oracle `L` | declared `D` | capacity `B` |
|---|---|---|---|---|---|---|
| `short` | short | `s0` | 8 | 4 | 8 | 1 |
| `long` | long | `l0` | 96 | 8 | 16 | 1 |
| `preempt4` | preempt | `p0` to `p7` | 8 | 20 | 24 | 4 |
| `preempt8` | preempt | `p0` to `p7` | 8 | 20 | 24 | 8 |

`D` is the workload's declared output length, a fixed round number above the
capture's cap, chosen here and never revised. The `joined` treatment replaces
`D` with the capture's `L` through the version 2 join; the `declared`
treatment runs the accepted absent-replay path with fabricated token 512.
Bandwidth envelopes are `b100` at 8.0 TB/s and `h200` at 4.8 TB/s, an exact
5 to 3 ratio.

Derived admission steps and step counts, all literal:

| cell | treatment | admission steps | scheduled steps |
|---|---|---|---|
| `short` | joined | `s0` at 0 | 4 |
| `short` | declared | `s0` at 0 | 8 |
| `long` | joined | `l0` at 0 | 8 |
| `long` | declared | `l0` at 0 | 16 |
| `preempt8` | joined | `p0..p7` at 0,1,2,3,4,5,6,7 | 27 |
| `preempt8` | declared | `p0..p7` at 0,1,2,3,4,5,6,7 | 31 |
| `preempt4` | joined | `p0..p3` at 0,1,2,3; `p4..p7` at 20,21,22,23 | 43 |
| `preempt4` | declared | `p0..p3` at 0,1,2,3; `p4..p7` at 24,25,26,27 | 51 |

## Physical sanity, written before any measurement

One line each, before reading a number.

- Floor, model independent: a decode step must stream at least the attention
  weights, the top-k activated expert weights and the LM head, which is
  `(75,497,472 + 301,989,888) * 2 + 100,663,296` equals `855,638,016` bytes, so
  no step can beat `855,638,016 / 8.0e12` equals `107 us` on `b100`.
- Ceiling, model internal: the roofline streams all 32 resident experts every
  step at a 0.7 derate, so no step can exceed `W / 5.6e12` plus the KV term,
  which is `476.4 us` plus at most `1 us` for the context sizes in this sweep.
- The measurement must therefore sit at `476 us` to `478 us` per step on
  `b100`, i.e. at the ceiling, because this geometry is deeply memory bound:
  the largest step here forwards 96 tokens for `7.3e10` FLOP against a
  `5.5e11` FLOP budget at the memory roof.
- Scaling check: halving memory bandwidth must move every step latency by
  exactly the bandwidth ratio, because the step is memory bound and the FLOP
  term never becomes active.
- Against the real system: this configuration models neither host launch cost
  (`HostInitiationModel.ideal()`) nor any collective (one rank), and it streams
  four times the expert bytes a batch-1 top-k-8 kernel would move. A real
  Granite 3.0 1B A400M decode step on a single modern GPU is roughly 1 ms to
  2 ms, dominated by per-kernel launch and small-GEMM inefficiency across 24
  layers. The simulated `476 us` is therefore expected to be optimistic by
  roughly two to four times, and no absolute-latency claim is made. Every
  scored relation below is a ratio or a difference within one cost model.

## Scored behavioral relations

Each carries its answer to the pre-freeze entailment question: given the fatal
guards and identity checks already registered, and given how the fixture is
built, can this relation fail?

### B1: the join moves a later request's TTFT by the frozen step count

In `preempt4`, `p4` to `p7` are admitted four steps earlier under the join.
For each of those four requests and each bandwidth envelope, the frozen
prediction is

```
TTFT_joined / TTFT_declared in [0.836, 0.844]   for p4   (step ratio 21/25)
                             in [0.841, 0.849]   for p5   (step ratio 22/26)
                             in [0.847, 0.855]   for p6   (step ratio 23/27)
                             in [0.853, 0.861]   for p7   (step ratio 24/28)
```

Each band is the exact step-count ratio plus or minus 0.5 percent, which
comfortably covers the KV term (at most 0.2 percent of a step).

Can it fail? Yes. The step-count ratio follows from the frozen policy, but the
measured TTFT comes out of the live adapter chain: a wrong admission index, a
lost or duplicated step, a clock that advances on the drain step, or a replay
path that changes step composition all move it out of band. Eight instances.

### B2: the join barely moves TPOT, and never upward

For every request in every cell and envelope,

```
TPOT_joined <= TPOT_declared    and    1 - TPOT_joined / TPOT_declared < 0.01
```

The direction is the growing KV term: the declared series is the longer one and
its extra intervals carry the largest contexts. The magnitude is small because
the step is weight-bound, so an inter-token interval is nearly independent of
how many tokens the step carries.

Can it fail? Yes, in both halves. The bound fails if co-scheduling composition
turns out to dominate the KV term, which is exactly the mechanism PLAY-15
records for the fabric-timed path. The direction fails if a request's declared
and joined interval sets are not nested the way the policy implies. Counted per
request and envelope: 2 for `short` and `long` together times 2 envelopes,
plus 8 requests times 2 capacities times 2 envelopes, i.e. 36 instances.

### B3: bandwidth scaling is exactly the memory-roof ratio

For every cell and treatment, and for every request's TTFT and TPOT,

```
| value_h200 / value_b100 - 5/3 | < 1e-6
```

Can it fail? Yes. It fails if any step is compute bound (the FLOP roof does not
scale with memory bandwidth), if the host model adds a bandwidth-independent
term, or if integer truncation accumulates faster than 1 part in a million.

### KV1: allocation reconciles with the forwarded-token count

For every `preempt` request with no preemption event and no prefix-hit tokens,
the allocation token total equals `P + L - 1`, i.e. `8 + 20 - 1` equals `27`.

Can it fail? Yes. Nothing already registered constrains the per-request
allocation attribution of that capture, and eight requests share a 96-token
pool. Scored on the `preempt` cell only; the `short` and `long` values are
reported as post-specified confirmation for the reason disclosed above.

### KV2: a preempted request allocates at least the forwarded-token count

For every `preempt` request with at least one preemption event, the allocation
token total is at least `P + L - 1`. Exactly one request is expected to be in
this class, because the capture reports one preemption event.

Can it fail? Yes: a recompute that reuses slots without re-allocating them
would land below the bound and would mean the event stream cannot account for
recomputation.

### KV3: preemption counters reconcile with preemption events

For every request in all three cells, the number of preemption events naming it
equals its `framework_preemption_count`.

Can it fail? Yes. The aggregate is consistent by inspection (one event, one
request reporting one preemption) but the per-request attribution was not read,
so a mismatched request identity fails this.

### KV4: prefix-hit tokens reconcile with the cached-token counter

For every request in all three cells, the sum of prefix-hit token counts naming
it equals its `framework_cached_tokens`. Every request in all three captures
reports zero cached tokens, so this asserts that all 14 `preempt` prefix-hit
events carry a zero token count.

Can it fail? Yes, and it is the most likely failure in this set: a nonzero
prefix hit anywhere in the `preempt` cell breaks it and would mean the scalar
counter and the event stream disagree about the same decision.

### KV5: occupancy reconstructed from the event stream

Replaying the event stream in sequence order, adding allocation token counts
and subtracting release and eviction token counts, the peak live occupancy is
predicted to stay within `kv_token_capacity` in `short` and `long` (256, with
11 and 103 tokens allocated in total) and to exceed `kv_token_capacity` in
`preempt` (96, with about 216 tokens allocated across eight requests and a
single release event in the whole capture).

Can it fail? Yes: either cell can come out the other way. A `preempt` peak
within capacity would mean the release and eviction records do account for the
pool; the predicted overshoot instead identifies a missing free or release
observation in the version 2 writer. If the overshoot happens it is reported as
a schema capability gap, not fixed here, because other branches are consuming
that artifact this wave.

## Exact-oracle relations, reported separately

- E1: the version 2 joined replay-run record and the KV reconciliation record
  each survive write, strict read and write again with identical UTF-8 bytes,
  and their readers reject unknown fields, an unsupported schema, a duplicate
  request identity, a routing reference naming a different trace, and a routing
  reference whose trace schema disagrees with the run's.
- E2: every joined request's output length, stop reason and output token IDs
  equal the capture's, element by element, and every served token stream in a
  joined run equals the capture's. This is a round trip through the artifact
  the join just read, not a prediction, so it is exact and unscored. It is
  still fatal when violated.
- E3: the joined routing projection reproduces the capture's per-token,
  per-layer top-k tuples in capture order without sorting, restricted to the
  joined requests and in joined request order. Also a round trip: exact and
  unscored, fatal when violated.

## Fatal unscored guards

Fatal means void, not a lost point. A single violation voids the run for the
purpose of closing PLAY-8, and no behavioral fraction from that run remains
interpretable. None of these enters any scored denominator.

- G1: every step of every run is classified memory bound by the roofline. If
  any step is compute bound, the frozen closed form does not apply and every
  latency relation above is void.
- G2: the accepted version 1 join and the absent-replay path stay byte- and
  timestamp-identical. Proved in pytest against the tracked baselines under
  `tests/fixtures/preplay`, captured from the accepted code in this same
  expectations commit, before any version 2 join exists.
- G3: in a joined run, every request's served token stream equals the capture's
  output token IDs and the scheduler-visible drain happens at exactly the
  oracle length.
- G4: every request admitted at the same step under both treatments has exactly
  equal TTFT in integer picoseconds. This is by construction, since the step
  compositions before that admission are identical, so it cannot fail without a
  defect: it is the detector the task asks for, and it is fatal, not scored.
  It covers `s0`, `l0`, `p0` to `p3` in both capacities and all eight requests
  in `preempt8`.
- G5: the version 2 join rejects, before any bookkeeping mutation, an arrival
  naming a request absent from the capture, a duplicate arrival identity, a
  negative or boolean timestamp, an empty arrival set, and a bookkeeping object
  collision.
- G6: the version 2 join refuses a version 1 trace and the version 1 join
  refuses a version 2 trace, each by schema and not by guesswork.
- G7: the KV event stream is not an authority. Rewriting each capture with an
  empty KV event list and re-joining leaves every joined request field
  identical (identity, arrival, output length, stop reason, output token IDs,
  routing reference request ID); only the trace hash and the reconciliation
  record change.
- G8: every joined request appears exactly once in core bookkeeping as a
  framework-owned `FRAMEWORK_REQUEST` correlated to exactly its own request ID,
  with no loss and no duplication.

## Registered commands and pre-freeze dry run

The registered study command is:

```text
.venv/bin/python examples/preplay_framework_join_v1/run_study.py \
    --short-trace <path> --long-trace <path> --preempt-trace <path> --check-only
```

Before this freeze that exact command was executed against an argument-parser
skeleton. It exited zero after resolving the three trace arguments and
verifying their frozen SHA-256 values, and it produced no result rows and no
output files. The skeleton was then removed, so this expectations commit
carries no study harness and no version 2 implementation.

The scored invocation replaces `--check-only` with a fresh `--run-dir` outside
Git. Gate commands are the repository-standard ruff, pytest, docs-format and
task-progress invocations and are not study commands.

## What a pass establishes, and what it does not

A pass establishes that a framework's own capture can pin each replayed
request's arrival, output length, stop reason, output token IDs and expert
routing through the live replay seam, that the framework scheduler stays the
only KV authority, that the capture's KV event stream reconciles with its own
oracle record under a stated operational rule, and that the accepted version 1
and absent-replay paths are untouched.

It does not establish SGLang replay (PLAY-7), a fabric-timed end-to-end
latency, per-request time attribution, or any absolute-accuracy claim. The
cost model is a single-rank roofline with an ideal host model, and its absolute
step latency is expected to be optimistic against a real deployment by roughly
two to four times.
