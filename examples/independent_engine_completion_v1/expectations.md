# Independent native engine completion

This prospective contract owns CORE-68. Freeze its final expectations-only
commit before implementing the independent completion path or running this
study. Existing single-request service prices and compatibility identities
are known regression evidence. The independent schedules and their live
native request outcomes are prospective assertions. No GPU is allocated.

## Physical mechanism and authority

Two separate engines represent separate groups of GPUs. Each group can work
while another group is busy. A request still waits for its prompt computation,
cache transfer and preceding output token. Finishing a long step on one group
must not hide an earlier completion on another group.

The serialized compatibility driver completes one whole engine step before
selecting another. The new explicit independent mode retains one pending
native scheduler step per engine and publishes its result only when the shared
virtual clock reaches its completion. A single core authority owns pending
visits, event ordering and completion. There are no private engine clocks,
clock rewinds, post-hoc timestamp corrections or speed multipliers.

Use the existing `StepRecord`, `QueueVisit`, `CompletionEvent` and `StepResult`
contracts. A visit names an engine-qualified `GPU_WORK_QUEUE` resource and a
stable engine/step identity. It represents declared whole-engine service,
including the selected local collective cost. It does not claim measured
kernel occupancy or identify physical hardware scheduling inside the group.
The native scheduler owns request admission and its queue. At step submission,
that engine is free: submitted, eligible and started timestamps coincide.
Finished and completed timestamps coincide at the sole due event. These forced
zero visit waits and visibility delays are fatal unscored identities. Native
request queue waits remain distinct and may be positive.

Only completed visits enter completed result projections. Input records may
be retained at dispatch, with their exact release timestamp, so an arrival
during a pending step does not incorrectly include that earlier input in its
new request slice. A sink may prepare a private deterministic price but must
delay outcome publication until consumption at the completion event. The
price receipt is immutable, bound to its exact input and owner, and consumed
once. No second layer independently completes the same step.

## Native execution boundary

Admit the pinned in-process vLLM 0.27.1 core. Replace only the engine instance's
step callback for the execution split. Preserve native `InprocClient.get_output`
and `LLMEngine.step`, including their output processor, post-step handling and
request-visible results. Submission calls the native scheduler once and retains
its exact scheduler output and pending executor receipt. Scheduling reserves
native in-flight tokens; a second same-engine submission is forbidden until
that receipt retires.

At the due event, drive native `LLMEngine.step`. Its bound callback consumes
the exact receipt and calls native scheduler output processing. Preserve the
grammar lookup, error and iteration contexts, abort-queue processing and
iteration attachment. Native wall-clock statistics are disabled in the
supported envelope. Producer cache-handoff parameters and emitted tokens
become visible through native processing at this boundary. Fabricating a native
output privately cannot make it visible to the scheduler or frontend early.

Admit tensor parallelism with one pipeline stage, one data-parallel replica,
synchronous scheduling, no speculative decoding, no structured output and no
real GPU workers. The initial session uses the existing isolated declared local
collective arm. Reject a shared packet/fabric session, retained peer packet
runtime, collective registration authority, request-metric sink reducer,
replay token source or paced wall execution in this mode. Existing owners
retain their broader support obligations; enabling independent engine service
does not silently emulate those paths. An opt-in unsupported configuration
fails before the first native scheduler mutation.

Reject early or duplicate retirement, a foreign receipt, replacement of the
bound core/executor/sink, an occupied engine slot, changed input or changed
selected configuration. A runtime error poisons the pending session. Do not
retry half-mutated native state. Supported abort/reset entry points reject
while a step is pending, before releasing native cache or request state.
Mid-flight cancellation and recovery are a separate completeness obligation.

A genuine zero-service drain must carry native finished or preempted identities
and retire exactly once at the current time. An empty phantom step cannot
create work, advance time or produce an infinite scheduling loop. Completion
callbacks at one timestamp run in deterministic insertion order before new
admissions and submissions at that timestamp. A newly submitted zero-service
drain is processed before time advances further. A later short step may complete
before an earlier long step without changing the long step's due time.

## Fixed component and native sweep

The finite case list and source identities are in `expectations.json`.

Component evidence includes eight equal-work configurations: two or four work
items, service 1,000 or 3,000 ps, and independent resources or one serial
resource. Four unequal cases start a long step at zero and a short step at
1,000 ps, crossing long service 10,000 or 20,000 ps with short service 1,000 or
2,000 ps. One tied-completion case and one genuine zero-service drain complete
the 14 configurations. Check intermediate state immediately after submission,
just before a due event and just after retirement. No early completion is
permitted at any checkpoint.

The native sweep uses the accepted config-only Granite checkpoint, eight
simulated workers per engine, one request scheduled per engine step and four
output tokens per request. It runs six fresh processes: serialized or
independent timing, crossed with equal prefill/decode pool sizes 1, 2 and 4.
Every process retains its actual native engines for all of its cells. It runs
four simultaneous bursts of four requests each: prompt lengths 8 or 16,
crossed with declared handoff durations 100 or 200 microseconds.

The width-one processes also run two two-request arrival cells, prompt 8 and
100-microsecond handoff. The first request arrives at the cell origin; the
second arrives at half or twice the known prompt service. The width-two
processes also run one mixed cell: prompt 16 at the origin and prompt 8 one
microsecond later, both with a 100-microsecond handoff. The width-one serialized
process first repeats the four original CORE-58 requests under their original
labels and ordering. This is 112 native requests across 34 cells, including
the four historical controls. The fixed four-request bursts already exercise
native queueing at widths one and two.

The per-process host limit is 1,800 seconds and 16 GiB sampled resident memory.
There is no fitted host performance expectation. If a limit is reached, retain
the exact stopping point and mark the entire campaign VOID. Do not promote a
partial smaller configuration to a full pass. The separate CORE-52 target
campaign and its original timeout remain unchanged.

## Independent schedule oracle and bounds

For the declared component, a work item's completion cannot precede its arrival
plus service. All work can finish no later than the last arrival plus the sum
of services under the finite nonblocking component envelope. Equal ready work
on separate resources finishes in one service interval; equal work on one
resource requires two or four intervals. The later short step finishes at
1,000 plus its service, strictly before the long step. These are mechanism
relations, not measured GPU prices.

For the native compatibility price, prompt service `Cp` is 95,424,000 ps at
length 8 and 114,936,000 ps at length 16. Each of the four decode services `Cd`
is respectively 77,952,000 or 77,976,000 ps. These individual intervals were
read from retained CORE-58 evidence before this freeze. The first decode
engine step has native prefill phase with one new token and a cached prompt;
the three later steps have decode phase. Do not rename the first step to make
a phase assertion pass.

Cache transfer cannot beat bytes over the declared link rate. At 49,152 bytes
per prompt token, cache sizes are 393,216 and 786,432 bytes. A 400-Gbit/s route
gives floors 7,864,320 and 15,728,640 ps. A declared 10-Gbit/s route plus
50 microseconds gives ceilings 364,572,800 and 679,145,600 ps. Both handoff
choices lie within these declared bounds. The service model charges all
resident experts, so resident weight bytes divided by memory bandwidth is only
a conditional floor for that surrogate. It is not a hardware floor for the
subset of experts actually selected by one token; COMP-7 retains that work.

For each independent burst, let `P` be both pool widths, `H` the handoff and
`i` a zero-based request index. Round-robin routing assigns engine `i mod P`.
Native first-come scheduling at one request per engine gives:

```text
prefill_start(i) = floor(i / P) * Cp
prefill_end(i) = prefill_start(i) + Cp
decode_start(i) = Cp + H + floor(i / P) * 4 * Cd
token_end(i, k) = decode_start(i) + (k + 1) * Cd, k = 0..3
last_completion = Cp + H + (4 / P) * 4 * Cd
```

Here four decode services exceed one prompt service, so queued decode work
remains the bottleneck. Width one to two reduces last completion by exactly
eight decode services; width two to four reduces it by four. Completed request
throughput is exactly four divided by this makespan, with exact rational
units. Startup prompt work and handoff remain, so a fourfold hardware width
does not imply a fourfold whole-cell throughput. Adding 100 microseconds to
handoff shifts every token completion and request first-token time by exactly
100 microseconds. Inter-token time stays `Cd`, an unscored fixed-service check.

In the two later-arrival cells, the second request's prompt starts at the
maximum of its arrival and `Cp`. Its cache is ready before the first request's
decode finishes. Its absolute decode token times therefore equal those of the
second request in the width-one burst. Its time to first token, measured from
its own arrival, decreases by that arrival offset. This distinguishes real
admission time from a long pending step's completion. The mixed independent
cell must finish the later short prompt, and its first decode token, before
the earlier long request; the serialized comparator has the opposite order.

An independent finite reference scheduler derives every native request's full
prefill, handoff and four-token timeline from these frozen service values,
actual declared arrivals and routing. The serialized reference separately
implements its documented ready-engine round-robin rule. It does not call the
new runtime or infer expected values from observed completions. Retain both
full references. A changed native scheduling premise voids the run instead of
being repaired after observing the result.

The request makespan floor is the largest causal request chain and every
engine's charged work; neither can exceed the observed cell makespan. A loose
ceiling is the last arrival plus all unique step service plus all handoffs.
These are bounds on the declared service model. Check actual observed intervals
against them before exact agreement. The sum of work across independent
engines may exceed wall time. Only realized per-request causal intervals enter
time to first token (TTFT), time per output token (TPOT) and job completion
time (JCT). Never sum overlapping request record slices.

## Admission, evidence and acceptance

Keep complete native request results, scheduler input records, per-engine step
results, completion events, queue visits, native source receipts, actual worker
identities and process observations. The run records its selected timing
authority. Join every completed step to its input, sole receipt and native
retirement. Capture scheduler/frontend token and finished state before submit,
after submit, before retirement and after retirement. No request output,
completion-driven cache release or producer handoff may appear before due time.
Native scheduling reservations at submission are explicitly allowed.

Every declared engine is the actual session-owned object, bound to the actual
frontend executor and shared clock. Observer references do not keep engines
alive on the session's behalf. The GPU context remains uninitialized, model
weights remain unloaded and packet backend invocation count remains zero.
These facts and all excluded modes are fatal unscored guards.

The finite behavioral score contains 25 joint instances in six families:
four component independence, four unequal completion order, eight native width,
six native handoff, two native arrival and one native mixed-order instance.
Exact native timelines, original compatibility hashes, tied completion and
drain oracles, forced zero fields, source/identity checks and corruption
controls remain separate evidence classes. Never increase a denominator by
counting matched metrics, requests, timestamps or guards as extra behavior.

Admission rejects missing, duplicate, mistyped, noncanonical or foreign rows,
wrong family assignment, missing phases, unexplained idle, duplicate authority,
early visibility and any changed source or retained raw receipt. Ordinary
exceptions retain a traceback. One fatal guard, failed required relation or
incomplete stage makes the entire result VOID with a null behavioral score.
The original four complete CORE-58 comparisons exclude only their two declared
opaque native request IDs. The serialized disabled path preserves every other
field and all retained historical study bytes exactly.

A complete reviewed pass plus software gates closes CORE-68 for the explicit
isolated engine-service envelope, because its completion events reach native
request TTFT, TPOT and completed-token throughput. Component evidence alone
cannot close it. CORE-52 retains target host feasibility; CORE-51 and CORE-54
retain large deployment and public frontier qualification. Physical GPU service,
shared packet-fabric contention and mid-flight cancellation remain outside this
acceptance. This study establishes resource independence and causal visibility,
not calibrated GPU throughput.
