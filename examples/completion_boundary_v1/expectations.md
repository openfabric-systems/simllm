# Exact completion boundaries and persistent artifact expectations

This expectations-only change precedes implementation of the new native verbs,
the production session client and every new backend measurement in this study.
HTSIM-28 owns exact completion-to-injection continuation; BACK-38 owns one
checked step whose ordered artifacts retain a physical backend session. The
older `congestion_chain_v1` freeze and blocked report remain unchanged. This
successor does not adopt its equality between logical completion and physical
quiescence, and does not claim its larger BRIDGE-2 framed serving contract.

## Source audit and scope

The source bases are SimLLM `b7d64e1ced5a9e52b71ab89285932a17498a1998`
and HTSIM `cc1c80f434600ec977fdb5916fc5ff74be63231d`. The existing
`rnic_flow_session.cpp` blob is
`43e9a2b4e8cbc1549fb59bfc6ca7eeb06b88cfa6`. Its inclusive `advance(H)`
exposes a completion T only when H >= T; its later ordinary injection requires
eligibility greater than H. A client cannot use the discovered T for an exact
dependent release. The repair extends this session, preserving its single
event list, topology, random state, transport, congestion controller and native
remote direct memory access network interface controller (RNIC) authority.

The supported installed path is an explicit session option on `HtsimStepSink`.
One owned child lives from the first through the last artifact of one checked
step. Its absence retains the existing multi-artifact physical-profile refusal
and the accepted stateless ideal profiles. Retention across steps, arbitrary
online graph frames, full per-rank bookkeeping and cursor publication remain
BRIDGE-2. A diagnostic fresh-session comparison is not a supported physical
production mode.

Use the existing checked execution-graph projection, structured GOAL operation,
message and dependency records, native lifecycle observations, `CompletionEvent`,
`ExecutionResult`, `StepResult` and request reducer. Preserve an immutable
snapshot of each planned trace. The new executor must not invent a parallel
message inventory or feed an artifact-level synthetic report to the core as if
it were the original graph's per-rank runtime report.

## Native continuation contract

Keep the schema `simllm-htsim-flow-session-v1`, canonical length-prefixed JSON,
the one-mebibyte frame limit, contiguous accepted sequence and all existing
`open`, `inject`, `advance`, `drain` and `close` request/response forms.
New behavior is selected only by two new verbs:

- `await_completion` validates `through_sequence`, a strictly increasing list
  of pending `completion_sequences`, boolean `until_quiescent`, nullable normal
  scheduling horizon `through_ps`, positive hard `max_time_ps` and positive
  `max_events`. Completion mode requires at least one still-pending accepted
  target and `until_quiescent=false`. Quiescence mode requires an empty target
  list and `until_quiescent=true`. The hard time bound is absolute simulated
  time; the event bound limits callbacks in that call. A nonnull scheduling
  horizon cannot exceed the hard time bound or move backward.
- `inject_at_boundary` carries all ordinary injection fields plus a
  `boundary_id` and a strictly increasing, nonempty `predecessor_sequences`
  list. Every predecessor must have completed successfully. Eligibility must
  equal the still-open boundary's exact timestamp. Its accepted sequence and
  identity checks remain the ordinary injection checks.

Completion mode stops immediately after the first atomic native callback that
completes ANY named target. It returns every newly available completion row,
including other completions produced by that callback, and opens a unique
boundary token at T. Remaining callbacks at T and all later callbacks retain
their native order. Multiple legal injections at T are accepted in contiguous
sequence order without advancing the event list. A subsequent successful
advancing operation invalidates the old token.

Keep three concepts distinct: the fully processed inclusive horizon; the
ordinary-injection exclusion floor, which includes the exposed T; and the open
token permitting exact-time dependent injection. Stopping after one callback
at T does not mean all callbacks at T executed. Ordinary injection at T remains
illegal. Ordinary `advance(T)` processes the remaining callbacks at T and closes
that opportunity. Time zero must work without unsigned subtraction.

An await response distinguishes `completion`, `horizon` and `quiescence`,
reports the current event-list time and fully processed horizon separately,
and carries new lifecycle events, new completion rows, accepted cursor and
exclusive authority counters. A completion response additionally carries the
exact boundary timestamp and token. Every new completion row exposes native
success or transport-error status; error completion cannot become a request
metric. Existing legacy completion rows and CSV bytes retain their old form.

A normal scheduling horizon is a successful yield. For a pending local action
at L while native flows remain pending, the client awaits through L-1, so it
can still discover an earlier native completion and inject its dependents
exactly. It then processes the local action at L. Ready work at the current
time is handled before waiting, including time zero. A hard budget failure,
stalled unfinished runtime, transport failure or framing error is terminal;
the client discards staged metrics and reaps its child. There is no rollback
claim after native callbacks have executed. Validate requests before advancing,
posting, consuming a sequence or scheduling a partial injection.

Quiescence mode stops at the existing physical authority's verified empty-work
condition, after every accepted injection has fired. It enables the existing
terminal `drain` and `close`; neither is used between artifacts. Logical
completion T and physical quiescence Q remain distinct, with Q >= T. The
logical result carries no quiescence timestamp before that proof is available.

## GOAL and ownership semantics

The executor preserves `requires` completion dependencies, per-rank compute
ordering and the existing cost `max(calc_ns,1)*1000` ps. A physical send becomes
eligible from its source action. Its native completion timestamp is immutable;
logical send and matching receive completion also wait for receive availability.
An unexpected early physical arrival does not manufacture an early logical
receive. A ring successor waits for its declared predecessors. Waiting for one
native completion must not skip an earlier local compute completion.

Support `irequires` start dependencies only with explicit start evidence;
otherwise reject them before opening a child and identify the supported
completion-dependency subset. Reject unsupported CPU or NIC selectors,
self/local messages, grammar, message matching, topology and projection inputs
before native mutation. Optional calibrated surcharges may be accepted only
when their declared cost moves the absolute network release before injection.
Any deliberately unsupported composition retains an owning registry task.

Use the existing owned-child registry, process group or Windows Job, ownership
markers and launcher handshake. Read exactly one unbuffered handshake byte so
the first frame cannot be prefetched and lost across execution. Keep stdin open,
drain stderr, bound frame reads and wall time, and reap descendants after EOF,
timeout, protocol failure or cancellation. The existing communicate-style
process helper remains an exact compatibility path.

Join each accepted native sequence, flow, work queue entry, lifecycle event and
completion to its checked artifact and original graph operation. Preserve
absolute native evidence and derive relative artifact rows by subtracting the
network release. Never change flow completion time to hide a late receive.
Compute operation boundaries follow their own declared action; collective
boundaries follow the completed action graph. Reconcile graph completion with
the `StepResult` returned by the existing sink and attach the existing request
reducer's metrics to that result. Packet observations are read-only projections.

## Frozen native populations and physical bounds

Use unchanged default packetization: 4,096 payload bytes and 64 header bytes per
full frame. Define `q=4160*8*10^12/R` ps at R in {200,400} Gbit/s. No packet
or policy defaults change in this task.

The exact-boundary population has eight ideal `rnic-nn` configurations: payload
B in {4096,8192} bytes, chain length K in {2,4}, and both rates. Use two
endpoints, alternate direction, release the first flow at zero, then release
each successor at its predecessor's exact native completion. For n=B/4096,
the independent one-flow oracle is `F=(n+1)*q+2,000,000` ps and the chain
boundary is exactly K*F. The extra full frame is destination serialization;
the two-microsecond term is the ideal profile's propagation.

Floor: each flow needs B*8/R seconds of endpoint service and two microseconds
of propagation. Ceiling: this unloaded fixed-packet path equals the stated
finite F, with no overlapping same-direction work. Four bandwidth relations
halve K*F minus propagation exactly; four payload relations add K*q when B
doubles. Exact-oracle rows and relation instances remain separate evidence.

The retention population has four ideal configurations: successor B in
{8192,16384} bytes and both rates. At zero release a 4,096-byte trigger 0->1
and a 1,048,576-byte background flow 1->0. Release the successor 1->0 at the
trigger's exact completion, using its boundary token. Compare with one fresh
diagnostic session containing only the same successor at the same absolute
release. The retained successor FCT must be strictly larger, and its source
send-queue high-water mark must be two rather than one. Native identities and
post counts must conserve, with one constructed authority and no legacy posts
or mutations. An absolute clock retained after resetting the queue is not
acceptance.

Floor: every flow respects its endpoint payload service plus two microseconds
propagation, and the background cannot complete before its 256 full frames
leave the source. Ceiling: all finite wire frames serialized without overlap
at both endpoints plus four propagation allowances bound the whole retention
experiment by `2*(1+256+B/4096)*q+8,000,000` ps from time zero. The probe has
no loss, retry or control mechanism that can justify exceeding that bound.
Each of these four configurations runs one retained and one fresh session.

Protect the old native verb transcript for the four B/rate combinations of the
unloaded two-node case, once on the unchanged base binary and once on the new
binary. Use the same open/inject/advance/drain/close frames, advance through
10,000,000 ps and compare every response byte exactly. These are four unscored
compatibility controls, not new behavioral samples.

## Frozen live request population

Use generated eight-endpoint topology and active tensor-parallel ranks (0,4),
two dense layers, hidden width 256, intermediate width 1024, four query and
key/value heads of width 64, vocabulary 1024 and two-byte values. Batch size
is 16 or 32. Each request starts with one prefill token and then receives two
one-token decode steps. Their context lengths are one, two and three. The
first step releases at zero; each following step releases at the preceding
returned completion. The same request IDs persist through the three steps.

Declare a deterministic 16 microseconds compute per layer with the ideal host
launch model. This is a synthetic network study, not GPU service calibration.
Use no placement remapping, calibrated collective surcharge, registration,
aggregate floor or dependency cross-check. The session seed is one; custom
topology and profile overrides are absent. The two active endpoints sit on
different leaves of the generated physical topology.

There are four batch/rate configurations per profile, with three checked
steps each, for each of session-enabled `rnic-nn` and `rnic-cn`. Pure lowering
inspection before this freeze finds eight graph operations, six ordered
artifacts (two compute and four collective), 16 messages and eight network
rounds per step. Ring chunks are 4096 and 8192 bytes respectively. The isolated
collective artifacts have no active-rank `calc 0` joins; inactive padded ranks
finish their one-nanosecond no-op off the critical path. No join surcharge is
added to the supported sink oracle.

The ideal step oracle is exactly `L_NN=32,000,000+8*F` ps. Every prefill time to
first token (TTFT) and every decode interval equals its checked step latency;
time per output token (TPOT) uses the existing exact request-history reduction.
The expected four ideal latencies are 50,662,400 and 51,993,600 ps at 200G,
and 49,331,200 and 49,996,800 ps at 400G, in ascending batch order. These are
arithmetic predictions, not observations. Removing 32 microseconds compute
and 16 microseconds propagation leaves an exactly inverse-rate term. Doubling
the batch adds exactly 8*q. Check both parameter relations for all three
steps, six instances per family.

Physical floor: each cross-leaf round needs at least B*8/R seconds plus four
one-microsecond link pipes, giving
`L_CN >= 32,000,000+8*(B*8*10^12/R+4,000,000)` ps. This independently forces
CN above the matched ideal step by at least
`8*(2,000,000-q-(B/4096)*64*8*10^12/R)` ps. The smallest bound in the grid is
14,627,840 ps. The twelve physical-versus-ideal step comparisons must respect
their own positive bound and publish that effect through TTFT or TPOT.
No arbitrary CN bandwidth monotonicity is predicted.

Engineering ceiling: each checked step completes within 1,000,000,000 ps and
its native physical work drains within 10,000,000,000 ps after step release.
These are finite operational guards, not physical theorems. The generated
eight-endpoint receiver window is derived as 15*q, not the unused 4.096 us
option default. Release tick is 16,000 ps. Keep all control, recovery, routing
and arbitration defaults unchanged; do not enlarge a bound after observing it.

Run the identical three-step input grid with the session option absent for
both ideal profiles, `rnic-nn` and `rnic-nn-fluid`, on base and candidate code.
Compare all emitted GOAL, binary, completion CSV and serialized `StepResult`
bytes, plus artifact inventories. Run the four absent-option physical cases
on both versions and require the same refusal before any native child starts.
These eight ideal scenario pairs and four rejection pairs are unscored controls.
The baseline code and binary are independently retained before implementation.

## Interpretation and acceptance

Store all input, source, freeze, runner and executable identities before native
execution. Keep raw frames, inputs, completion rows, logs and full graph/result
evidence outside Git. The public result retains all verdicts, metrics,
relations, identity counts and hashes without rescoring omitted evidence.
Retain failed executions with their original chronology. No added shape, load,
rate, policy or timing offset is allowed after the first run under this freeze.

The fatal set includes frame and lifecycle integrity, complete unique graph
and message joins, exact dependency timing, successful completion status,
single authority, physical floors, finite budgets, final physical quiescence,
unchanged off controls and exact result conservation. A single fatal violation
voids the aggregate and removes its behavioral score. A valid failure of the
positive retention hypothesis is a refutation and leaves BACK-38 open.

Keep configuration counts, exact-oracle rows, behavioral families and their
parameter instances, fatal guards and native test executables separate. Native
tests must cover same-time pending callbacks, multiple completions in one
callback, stale and reused tokens, sequence/cursor rejection before mutation,
local action before/equal to completion, late receive availability, dependency
semantics, handshake-plus-frame delivery, child failure before publication and
logical completion preceding quiescence. Run the complete native suite with
the native RNIC composition enabled, the full eight-plan legacy gate, and the
full Python and lint gates before push. Merge history preserves this freeze.

HTSIM-28 closes after exact native continuation and protected old transcripts
are demonstrated. BACK-38 closes only with those prerequisites, the four
retention discriminators, all supported checked live metrics and every fatal
control satisfied. The literal claim is retained physical execution across one
step's ordered artifacts. This does not close BRIDGE-2's cross-step framed
session, TRAF-8's captured pipeline serving, hardware calibration, or the older
blocked congestion-chain experiment.
