# Coarse receiver ingress serialization

This prospective contract freezes CORE-48 before implementation or its first
study run. The as-of simllm commit is
`429acf0fbd733006c8978d00cf18c0090584dc86`. Existing collective-plan and
packet studies retain their original outcomes. This study uses CPU simulation
only and makes no hardware calibration claim.

Pre-run clarification chronology: the chunk notation and sentinel wording
below are clarified after the first receiver implementation edit, before any
unit execution or study run. The mechanism, matrix, accepted ring payload
rule and numerical sentinel values were already frozen at `5de4db2`. The
explicit rewritten ring-floor and sentinel assertions are post-specified
regression checks; this clarification is not claimed to precede implementation.

## Model and authority

The explicit coarse `receiver_ingress` selection extends the existing
`AtlahsWqeLedger` authority. It defaults to false, preserving the accepted
source-only path exactly. The enabled path applies only to cross-node
semantic sends. Intra-node service remains with its existing authority;
structural native RNIC selection rejects this additional coarse selection
before execution. Both source and receiver state clone and commit as part of
the same existing transaction.

For a payload B, link rate R, semantic eligibility e and completion visibility
delay D, define d=ceil(B*8*10^12/R) picoseconds. The source becomes ready to
offer the transfer at t=max(e, TX_available[source]). Joint transmission
starts at s=max(t, RX_available[destination]), finishes at f=s+d and becomes
visible at f+D. Both directional port cursors become f. Source and receiver
maps are separate, so opposite directions on one network interface remain
independent. Port identities include node and GPU rail.

The interval [s,f] is one indivisible reservation of both ports. This is a
coarse circuit with receiver backpressure and deterministic submission order.
A blocked extent delays later extents from that source. Already reserved
future intervals retain their order, even if a later submission has earlier
eligibility. The model does not claim independent buffered egress, packet
interleaving, work-conserving arbitration or finite switch-buffer behavior.
No traffic-plan reordering is allowed to improve an observed result.

The immutable WQE projection retains physical source start s, finish f and
one completion f+D. Enabled metadata names the destination and source-ready
boundary t. It must not change the disabled record's serialized fields.
Endpoint byte and occupancy ledgers are read-only projections of these
reservations, joined by WQE identity; no counter or diagnostic advances
another timing authority.

## Queue visits and the live metric chain

Project source ordering as eligibility e, grant t, zero service at t and
zero bytes. Project destination admission and joint service as eligibility t,
grant s, finish f, completion f+D and B service bytes. Source wait is t-e;
receiver wait is s-t; network service is f-s exactly once. Physical source
occupancy is the read-only [s,f] reservation, not another additive service.

Completion events expose both admission boundaries and receiver byte progress,
followed by one WQE completion on its existing completion queue. The authority
checker must reject lost, duplicated or mistimed projections. A source-NIC
queue pair that already includes receiver blocking must not remain as an
additional queue-wait attribution.

The supported chain is ExecutionGraph, the coarse runtime authority,
CompletionEvent, StepResult and CompletionReducer, then time to first token
(TTFT), time per output token (TPOT) and job completion time (JCT). Each study
configuration executes one prefill and two decode steps for one request.
The network work is identical in each step, with zero compute and launch
service unless a fixture explicitly declares channel service. With a
per-step elapsed time L and request arrival zero, expect TTFT=L, TPOT=L and
JCT=3*L. A step is released when its predecessor completes. These are network
isolation experiments, not estimates of a deployed model's token speed.

## Parameter matrix and predictions

The main matrix uses payloads B in {3,4096,65536} bytes, fan-in F in
{1,2,4,8}, rates R in {200,400} Gbit/s and both selection states. Every rank
is on a distinct node, using rank IDs separated by eight. Include:

- Combine star: F independent sources send B bytes to one receiver.
  Source-only latency is d; receiver-enabled latency is F*d. The increase
  is (F-1)*d. The enabled last transfer waits (F-1)*d at the receiver and
  serves for d. Total receiver visit wait is F*(F-1)*d/2, an additive work
  sum that must not replace the selected-path wait in TTFT or TPOT.
- Dispatch star: one source sends B bytes to F distinct receivers.
  Both modes take F*d, with identical WQE physical timestamps. Receiver
  wait is zero; source-order wait determines the tail.
- Symmetric duplex: F disjoint pairs each send B bytes in both directions.
  Both modes take d. Directional ledgers charge 2*F*B bytes in total at
  each endpoint-role reduction, with no receiver contention.

All chosen rates yield integral picoseconds per byte, so doubling payload
or halving bandwidth scales the serialization contribution exactly. The
4096-to-65536-byte change multiplies serialization by sixteen. Doubling F
doubles enabled combine and dispatch time; duplex duration is unchanged.
Compare quantitative relations before exact oracles. Zero effects forced by
the topology or disabled selection are fatal identity guards, not behavioral
score points.

Retain these additional fixtures:

- The accepted ring grid: widths {2,4}, payloads {4,4096}, rates {200,400}
  Gbit/s and per-channel service {0,7000} ps. There are 2*(W-1) rounds,
  each with q=max(1,floor(B/W)) bytes per send. Define chunk service
  d_q=ceil(q*8*10^12/R); ring latency is 2*(W-1)*(C+d_q+D), with C the
  configured per-channel service. Both modes keep the accepted
  physical timestamps and metric durations, including channel service.
- The four-rank, three-byte ring sentinel at both rates and zero channel
  service keeps TTFT=TPOT=120 ps at 400 Gbit/s and TTFT=TPOT=240 ps at
  200 Gbit/s.
- The asymmetric three-byte plus five-byte stars at both rates. Dispatch
  remains 160/320 ps. Combine changes from 100/200 ps to 160/320 ps. The
  original source-only combine values were structural evidence, not
  protected physical receiver timing.
- A source-major complete remote all-to-all at widths {2,4}, payloads
  {3,4096} and both rates. Width two stays d. Width four changes from 3*d
  to 5*d under the declared joint reservation order. No accepted remote
  complete all-to-all timing oracle was found in the retained qualification;
  its 559-byte wire oracle uses local ranks and remains unchanged.

This defines 102 distinct workload configurations: 72 main, 16 ring,
two sentinels, four asymmetric and eight complete all-to-all. Run each once
against the immutable pre-change source-only runtime to capture its canonical
snapshot, then run disabled and enabled on the candidate: 306 executions in
total, each containing three request steps. The baseline is read-only evidence;
its known absent receiver limit is not asserted to obey receiver physics.
The current default and explicit false selection must reproduce its report,
events, bookkeeping, graph and request outcomes exactly. Enabled identity
checks compare physical timestamps and metrics, allowing only the declared
additional receiver projections. Preserve graph and GOAL bytes in both modes.

## Bounds before measurements

Floor: every enabled phase takes at least the maximum total payload sent by
one source or received by one destination, divided by R, with any declared
causal ring rounds strengthening that bound. Every transmission takes at
least d and no dependent step completes before its predecessor.

Ceiling: for an isolated finite fixture with zero external arrivals, fully
serializing every declared extent plus every declared channel service and
visibility delay is a conservative finite ceiling. There is no unconditional
ceiling under external contention. Record physical floors and the fixture
ceiling before reading outcomes, separately from exact model predictions.

The equal star predictions meet their capacity floors. The four-rank complete
all-to-all has a 3*d receiver/source floor, model prediction 5*d and fully
serial ceiling 12*d. Its gap above the floor is the declared reservation
schedule, not evidence of a calibrated switch queue. For ring fixtures, the
causal lower bound is 2*(W-1)*d_q; channel service and completion visibility
are explicit extra terms.

## Fatal guards and regression boundaries

Every enabled cross-node extent charges exactly B bytes once to its source
and once to its receiver, with unique WQE identity and one completion. Both
endpoint interval projections are nonoverlapping on each directional port.
Receive service never starts before source readiness; completion never occurs
before release plus visibility. No source/destination identity is merged across
GPU rails. Duplex send/receive directions do not share one cursor.

Exercise zero-byte extents, no-send collectives, same-node bypass, independent
receiver rails, nonzero completion delivery, nonzero channel service, source
head-of-line blocking and out-of-order semantic eligibility under the declared
reservation order. A late validation failure must leave source cursors,
receiver cursors, sequences, records, request bookkeeping and clock unchanged;
the following valid execution must match a fresh runtime exactly. Invalid
selection types and structural/coarse double ownership reject before mutation.

Compare the exact default versus explicit false path, including serialized
report fields, across the retained fixtures. Run the existing collective-plan,
authority, device-runtime and completion-chain regressions unchanged. Identity
arbitration under class-label permutation must keep timestamps and bytes exact.
Do not replace a published baseline or amend a prediction after observing it.

One violated fatal guard voids this study for closure, with findings retained
and no behavioral score. Keep executions, exact oracles, behavioral relation
families and instances, and fatal invariants separate. Raw observations must
be saved before exact-oracle checking. Source, script, input and expectation
digests belong in immutable bulk records; only compact results enter Git.

## Project effect and limits

Successful receiver-bound scaling, live TTFT/TPOT effects, exact disabled
compatibility and full gates close CORE-48 for this declared coarse model.
The active source-only surrogate remains available as an explicitly selected
compatibility baseline, with no receiver-physics claim. CORE-8 retains general
cross-layer queue contracts and calibration; BACK-38 and TRAF-8 retain physical
serving integration. This task does not modify CORE-41's intra-node model,
the packet backend, HTSIM-40/41's completed evidence or hardware calibration.
