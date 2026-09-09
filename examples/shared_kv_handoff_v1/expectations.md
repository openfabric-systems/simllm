# Shared cache-transfer completion expectations

This expectations-only change precedes the implementation and first execution
of the shared handoff path. CORE-71 owns this bounded extraction from CORE-70.
Its source base is `72a395677ec56e9814a85518473940f2f95313c3`; CORE-68's
independent native-engine qualification is a prerequisite. This study does not
reuse, rescore or replace any earlier VOID campaign. The final qualifying
expectations commit must be recorded before the first implementation commit.

## Physical model and accepted scope

A completed producer has a model-sized key/value cache. Its eight tensor
parallel workers own disjoint shards. Each shard passes through its source
network port and its assigned destination port. Two producers directed to the
same decode engine compete for the same eight destination ports. Decode can
start only after its complete cache arrives. Request wall time follows the
last required shard, not a sum of durations over parallel shards.

The initial path uses the current pinned HTSIM source
`6efec16c54595e8905d37eb45ed9fc9fd7650be2` and its `rnic-nn` packetized ideal
endpoint-sharing profile. The profile has finite source and destination rates,
4,096 payload bytes and 64 header bytes per full packet, and a declared
2,000,000 ps propagation term. It has no switch graph. Logical deployment
placement does not establish physical Clos fidelity. The accepted effective
hardware SHA-256 is
`252f6a502fe8f0ced3b8848a924a03f27a53acaa9123eea4b692ef7e71c668ae`:
network enabled, DMA and queue-pair-context paths disabled, and zero modeled
doorbell, fetch, scheduler and completion-write costs. The actual native open
response must admit that identity; a copied configuration label is insufficient.

The known pinned executable receipt has SHA-256
`16a111a2a7e5ab573189739ba701bc8c10ec4ad4f8d0ef5b142c496964a5663c`.
The study receives its path explicitly and checks source, build and binary
identity before and after execution. No backend source or protocol change is
part of this slice.

Use actual in-process vLLM 0.27.1 with the same virtual-worker, model-revision,
tokenizer, source-file and selected-compute identity boundary as
`independent_engine_completion_v1`. Tensor parallel width is eight, pipeline
and data parallel width within each engine are one, scheduling is synchronous
and first-come first-served, and each engine admits one sequence at a time.
The provider is deterministic roofline pricing with efficiency 0.7, the B100
envelope, ideal host initiation and the already selected fixed local
collective profile. No GPU, model weights or hardware allocation is used.

The connector remains tensor-free. This campaign connects model-sized shard
bytes to native serving timing; it does not establish producer-buffer pinning,
real tensor copies or cache-lifetime safety while a transfer is pending.
Shared compute collectives, registration, peer traffic, request-reducer
composition, cancellation, replay, paced execution, batching extensions and
new native versions remain with CORE-70, CORE-69 and their existing owners.
All unsupported combinations reject before scheduling or child mutation.

The shared join identifies its authority as
`simllm-shared-packet-kv-handoff-v1` and its pricing arm as `shared-packet`.
Engine-local compute and collective timing keep their existing authorities.

## One authority and a pending capability

Preserve the synchronous `KvHandoffPolicy.apply/schedule` interface and every
field of the existing `KvHandoffEvent`. Add a separate pending capability
through the existing policy selection boundary. A shared policy cannot return
a priced completion before the native network produces one.

At prefill completion F, validate the complete request, engine binding, byte
partition and flow identity inventory. Immediately register every shard with
the same native `FlowSession`, at future eligibility F + delta. Delta is
strictly positive and is 20,000,000 ps in the accepted native sweep. Injection
is nonadvancing. The native event list owns the future eligibility events;
there is no Python eligibility heap and no second packet or queue authority.
An engine completion is not a native predecessor and must never authorize
`inject_at_boundary`.

The pending receipt is an opaque, immutable projection privately bound to its
exact owner, request, actual producer and consumer engine identities, byte
partition and accepted sequences. It has no guessed completion timestamp.
Each engine's network binding comes from the repository-standard deployment
placement and its GPU/NIC inventory. Per-engine local ranks starting at zero
are local compute projections and cannot be reused as global endpoints.
Validate dense endpoint domains, distinct producer/consumer pools, exactly one
mapping for every actual engine, tensor-parallel group membership, local-rank
order, and a bijection between placement ranks and GPU-affine NICs before
mutation. Freeze or copy mutable placement inputs before use. A later caller
mutation cannot silently reroute an accepted request.

Keep one native session, policy owner, sequence history and identity inventory
for the whole serving session, including repeated `run_requests` calls. Shared
policy replacement or per-call switching is rejected before mutation. Existing
legacy per-call overrides retain their accepted behavior.

## Completion joins and exact event order

Use a distinct immutable all-shard join. It carries request and engine
identities, aggregate and shard bytes, submitted and eligible times, all
native flow rows and lifecycle events, completed time, and the identities of
all shards attaining the latest completion. Complete it exactly once only
after every declared shard succeeds:

```
completed = max(shard.completion_time_ps)
handoff_wall = completed - submitted
```

Retain the native accepted, queued, started and completed timestamps with
their original meanings. The flow row's `start_time_ps` is injection
eligibility; its flow completion time includes everything until completion.
The protocol exposes no resource-release timestamp. Do not fabricate a
FINISHED event, physical `QueueVisit`, resource service time or visibility
tail. Per-flow work sums have separate names and never enter an additive
request wall-time decomposition.

`EngineStepRuntime` remains the sole public-clock writer and the owner of its
engine-completion heap. A native completion discovered at C can be staged
privately, but its request join cannot become public before the clock reaches
C. A publication after C is a failure, not a delayed timestamp repair.

At each public time T:

1. Retire all existing engine completions at T, including producer outputs
   that register future-dated shards.
2. Drain all required native packet callbacks through T. A native completion
   response can leave other callbacks at T pending; keep awaiting all remaining
   logical sequences through T until horizon T or no logical pending flow.
3. Publish every complete request join at exactly T. Partial joins remain
   pending. Then admit new arrivals, admit eligible decode requests, and submit
   ready engines in their established deterministic order.
4. Repeat zero-service retirement at T before advancing.

Between local events, await native progress only through the next arrival or
engine deadline minus one. If a native completion C is earlier, advance the
sole public clock to C and use the same loop. If there is no local deadline,
await the next logical completion. Never request a horizon below an already
established exclusion floor. With no pending native targets, skip await;
quiescence is not a substitute for progress and may exceed a future local
deadline. Do not insert packet events into the engine-owned heap.

Every native flow must join by stable identity to its one handoff, that
handoff's exact completion must gate the decode `StepRecord` release, and the
existing `CompletionEvent`, `StepResult` and request timeline must carry the
resulting absolute times. Do not manufacture a synthetic engine step for a
request-level handoff.

## Prospective native sweep

Use eight fresh shared-mode processes: two prefill engines, decode-engine
count one or two, prompt length eight or sixteen, and endpoint rate 200 or
400 gigabits per second. Each process runs exactly two requests arriving
together at zero and emitting four decode tokens each. Engine routing uses
the driver's normal round-robin assignments. There are sixteen completed
request vectors, not sixteen independent statistical samples.

The geometry is 24 layers, eight key/value heads, head size 64 and two bytes
per cache element. A prompt carries 393,216 or 786,432 bytes in total. Each
of eight rank pairs carries 49,152 or 98,304 bytes, i.e. exactly twelve or
twenty-four full packets. Every byte and shard is represented once.

Selected deterministic service is frozen at:

| Prompt tokens | Prefill service (ps) | Each decode service (ps) |
|---|---|---|
| 8 | 95424000 | 77952000 |
| 16 | 114936000 | 77976000 |

Those values are conditional on the unchanged selected compute model. Verify
actual selection, complete step inputs and all resulting records. They are
model inputs, not measured GPU performance. The existing resident-weight
surrogate and its limits remain owned by COMP-7.

Run two additional fresh shared-mode processes, one at each rate, with two
prefill engines and one decode engine. Each performs two calls on the same
serving session, each call containing two eight-token prompts. First admission
is zero; second admission is 1,664,000,000 ps, an integer multiple of both full
packet serialization quanta. The second call preserves the native child and
continues its sequence/identity history. Require four complete request vectors
per process, exact relative-time agreement where specified by the final
packet-calendar relations, and full byte, binding and lifecycle conservation.
Native aliases are distinct identities, not fields to erase for equality.

The primary eight cells declare 128 flows, 512 native lifecycle events,
2,304 full DATA packets and 9,584,640 wire bytes. Those structural inventories
are fatal guards, separate from behavioral evidence.

Use eight fresh compatibility processes: source before/after, serialized or
independent engine timing, and declared off or constant 100,000,000 ps handoff.
Each uses two prefill and two decode engines with two simultaneous eight-token
requests and four decode tokens. Compare the complete request results through
`to_comparison_json`, which excludes only its two declared opaque root native
request identifiers, plus every engine record, result, publication collection,
event, visit, clock advance, batch and selected profile. No new shared-mode
field appears in these legacy results. No native network child is opened.

Each native process has a 1,800-second wall limit and a 16-GiB resident-memory
limit sampled every 0.25 seconds. Every native packet progress call uses the
existing bounded protocol and a 60-second wall limit, one million callbacks
per call, and a ten-billion-picosecond session budget. First command, process,
raw-file and failure receipts are retained before admission on every exit,
including startup failure, nonzero exit and timeout. Partial rows are retained
but cannot substitute for completed process admission. These limits do not
change after a failed run.

## Physical floors, ceilings and behavioral relations

Let R be the endpoint rate, q = 4160 * 8 * 10^12 / R picoseconds, n the twelve
or twenty-four full packets per shard, delta = 20,000,000 ps, and p = 2,000,000
ps propagation. Both selected rates divide the full-packet wire extent
exactly: q is 166,400 ps at 200G and 83,200 ps at 400G.

Floor: an isolated shard cannot arrive before (n + 1) * q + p after
eligibility, because the destination serializes its last packet after the
source has serialized it. For any receiver's k earliest completions, elapsed
time from the common eligibility is at least q + p plus the cumulative wire
bytes of those k flows divided by that receiver's link rate. Apply the floor
to every prefix, not just the whole phase. Packet headers count as wire bytes.

Ceiling: serializing every wire byte in the two-request cell through one link,
plus one startup packet interval and one propagation delay, is a conservative
upper bound for this finite, loss-free endpoint graph with no other work.
The topology-free profile does not claim additional switch or routing delay.

The stronger source-derived accepted envelopes for this aligned equal-size
graph are:

| Destination sharing | Each shard completion after eligibility |
|---|---|
| Two decode engines, disjoint destinations | Exactly (n + 1) * q + p |
| One decode engine, degree-two sharing | Between 2 * n * q + p and (2 * n + 1) * q + p, inclusive |

The same bounds apply to a request's all-shard maximum. Equal half-rate
credits alternate packet service at a saturated shared receiver. They need
not give the same individual completion order as a fluid fair-share model.
The extra source/destination startup interval follows from the pinned packet
calendar; omitting it would compare against an incorrect one-port model.

Before reading outcomes, check each declared service against the conditional
40,108,032 ps resident-weight floor of the unchanged compute surrogate. For
this two-request envelope, prefill plus submission plus the conservative
network ceiling plus eight decode intervals bounds final request completion
from above. These are three distinct checks: compute/memory constraints,
packet/port constraints, and the complete causal serving schedule. Passing
them establishes internal model plausibility in this declared envelope, not
agreement with a measured GPU deployment.

Freeze twelve behavioral instances in three families. Each rate or size
instance is one paired-cell conjunction over both complete request rows; its
two row checks are not two additional instances. Write H = join completion
minus prefill completion for the request's handoff wall time.

- Rate family, four instances (decode count one/two by prompt eight/sixteen):
  H at 200G is strictly larger than H at 400G. The absolute residual
  `(H200 - delta - p) - 2 * (H400 - delta - p)` is at most `2 * q200`.
  Propagation and declared submission do not scale with rate.
- Size family, four instances (decode count one/two by rate 200/400G): H for
  sixteen tokens is strictly larger than H for eight. The absolute residual
  `(H16 - delta - p) - 2 * (H8 - delta - p)` is at most `2 * q`.
  The packet startup term remains additive when payload doubles.
- Sharing family, four instances (prompt eight/sixteen by rate 200/400G):
  the earliest request time to first token is strictly larger with one decode
  engine's shared endpoints than with two disjoint decode engines. The
  increase is between `(n - 1) * q` and `n * q`, inclusive, and equals the
  difference of earliest handoff joins exactly. The first decode service is
  unchanged; later requests' decode-engine queue waits are not network cost.

Keep exact oracle classes separate from those twelve relation instances:

- Four complete source-pair compatibility vectors, one per timing mode and
  declared handoff arm, including both request rows and all retained fields.
- Four disjoint-destination cell vectors, each covering all sixteen native
  shard durations at the exact `(n + 1) * q + p` value.
- Eight shared-mode cell schedule vectors and four repeated-batch schedule
  vectors. Each compares both requests' complete decode release, token,
  time-to-first-token, time-per-output-token and final completion times with
  the deterministic oracle below. These oracle vectors never increase the
  behavioral denominator.

For two decode engines, request i starts decode at its admitted all-shard
completion C_i. Its four token times are C_i + j * Cd for j = 1..4. For one
decode engine, order ready requests by admitted join time and the driver's
established deterministic tie order. The first starts at its C_i; the second
starts at max(its own C_i, first decode start + 4 * Cd). Apply the same four
token offsets. Time to first token is the first token minus request admission;
time per output token is exactly Cd. Validate the actual native `StepRecord`
eligibility, engine events and `StepResult` times against that oracle, not
only the final scalar request metrics. The oracle is fed only admitted native
flow completions, never the request's own reported completion or an isolated
network prediction.

The repeated-batch processes add no behavioral instances. Both batches have
the same prompt and isolated request inventory; the gap is an integer number
of packet intervals and exceeds the first batch's conservative completion
ceiling. Their relative packet and request times must match exactly, while
accepted sequence numbers, request identities, work-queue identities and
native aliases retain their complete distinct values and cross-layer joins.
The first batch's accepted network sequences are 1..16 and the second's are
17..32. A second child, repeated sequence, reused request or changed owner is
a fatal failure even when the resulting scalar times happen to agree.

## Structural and failure controls

These are fatal, unscored guards. They are not behavioral points.

- Exercise before, exact and after completion visibility; separate shard
  completions and separate native callbacks at the same time; an engine due
  event, request arrival and existing packet completion tied at one time; and
  a previously registered injection whose eligibility equals that completion.
- Reject zero submission delay, foreign/duplicate request identities, invalid
  or mutable endpoint bindings, changed shard bytes, wrong-owner or forged
  pending receipts, missing or duplicate sequences, foreign completion rows,
  partial joins, early/late/reused publications, and policy replacement.
- Retain the first accepted shard inventory and native transcript when a
  later shard injection fails. Poison both owners and abort/reap the native
  child, preserving the original exception without rollback or retry.
- A successful shutdown requires no pending engine work, handoffs or staged
  unpublished joins. Pending close must not advance packets behind the public
  clock. On failure, child cleanup must still run if engine-runtime close
  rejects pending service. Do not drain or close between request batches.
- Inspect the complete source, native framework, model-cache, backend binary,
  selected-profile and import-origin identities before and after execution.
  Preserve complete raw lifecycle and request records with first and final
  byte/hash receipts. A changed receipt, missing file, duplicate row or broken
  timestamp/byte/identity conservation voids the run.

Software fixtures may script protocol responses to isolate rejection and tie
semantics. They are component evidence, not native packet or serving outcomes.
The result must list their semantic cases separately from software test
executables, native configurations, exact oracles, behavioral families and
parameterized relation instances. Corrupting a complete retained observation
must be caught at admission rather than silently ignored by a partial reader.

## Qualification and limits

The prospective invocation is:

```bash
python -m examples.shared_kv_handoff_v1.run_study \
  --before-code-root "$SIMLLM_BEFORE_CODE_ROOT" \
  --native-python "$SIMLLM_NATIVE_PYTHON" \
  --vllm-source "$SIMLLM_VLLM_SOURCE" \
  --hf-hub-cache "$SIMLLM_HF_HUB_CACHE" \
  --known-native-receipt "$SIMLLM_KNOWN_NATIVE_RECEIPT" \
  --backend-code-root "$SIMLLM_BACKEND_CODE_ROOT" \
  --htsim-rnic "$SIMLLM_HTSIM_RNIC" \
  --known-backend-receipt "$SIMLLM_KNOWN_BACKEND_RECEIPT" \
  --output-root "$SIMLLM_SHARED_KV_RUN_ROOT"
```

All machine paths are explicit local configuration. The before checkout is
the exact source base named above; the executing after checkout is clean and
recorded before launch. The known receipts authenticate inputs only. No prior
outcome is imported as a result of this campaign.

Any violated fatal guard makes the run VOID and its behavioral score null.
Retain its first receipts and partial evidence; do not report a percentage or
rescore it with outcome-dependent checks. Correcting a frozen assumption
requires an explicit prospective amendment and a new output root. A software
reader correction records its actual chronology and preserves the failed run.

Only full native completion, defended packet/compute bounds, exact downstream
causality, full compatibility, failure controls and normal repository gates
can close CORE-71. CORE-68 must separately qualify its independent-engine
envelope. This slice moves shared modeled transfer time into live request
time-to-first-token and time-per-output-token evidence. It does not close
CORE-70's other resource compositions, CORE-52's large live session, CORE-51
or CORE-54's large-model deployment frontier, or GPU and physical-network
calibration tasks.
