# Pipeline queue service and receiver holding: frozen expectations

This expectations-only commit precedes the new trace implementation and every
successor simulation. TRAF-88 owns the study. The prior contention study stays
void in aggregate; the valid DATA recovery successor establishes completion,
not a tagged pipeline queue penalty. No old result is rescored here.

## Mechanisms and fixed inputs

The physical workload, semantic graphs, placement, endpoint permutation and
GOAL come from pp_rail_contention_v1 and pp_rail_topology_v1. Stage s is rank
8*s on NIC zero, computes for 1,000,000 ps and sends 65,536 payload bytes to
the next stage. Expert-parallel (EP) traffic starts at zero on separate NICs;
each remote ordered pair sends 1,048,576 bytes. The pipeline (PP) has P stages
and P-1 forward hops. This is a forward-only isolated request with background
EP work, not fixed-model speedup, captured microbatches or GPU calibration.

Keep 64 endpoints, eight leaves, 400 Gbit/s endpoint and uplink rates,
1,000,000 ps link propagation, zero added switch delay, seed one, 4,096-byte
maximum DATA payload and 64-byte header. The ns-tm3 physical output uses the
existing oldest-head-first same-class policy and the existing deterministic
ingress tie arbitration. Priority flow control is disabled. Buffer capacities,
control parameters, packet routing and retry limits remain those of the
validated recovery implementation. No new scheduling or transport policy lands.

Every primary physical run uses 131,072 bytes of control headroom,
exponential DATA probes, four control windows per base probe, eight retries
and the retained final 50 ms timeout. Fix the control deadline at 10,000,000 ps,
margin at 900,000 parts per million and control wire size at 64 bytes. The
base switch buffer and Ring-CAM capacities are each 1,048,576 bytes.
Initial-window budgeting is disabled in every primary cell.
A load-dependent initial budget would confound sender grant gating with
fabric queueing; it must not enter this study. Recovery-only completion of
the formerly void cells is established by the prior published recovery study.

Hold the Ring-CAM delay window Delta at 10,566,400 ps in the main grid and
its release tick at 16,000 ps. Ring-CAM is the finite receiver store that
releases on-time originals on the rounded expected-arrival-plus-window edge.
Use an explicit window argument, so spine count cannot change this parameter.
Four unloaded crossover cells instead use Delta=6,572,800 ps. The production
automatic formula for 64 endpoints and S spines is
`(63 + 2*ceil(64/S))*4160*20` ps, giving those two values at S=8 and S=2.
The formula is source-derived before this successor runs.

## Finite configuration matrix

The physical core contains exactly 24 configurations:

- Twelve primary cells: P=2, attachment in {rail, node-local}, spine count
  S in {8,2}, EP width W in {0,8,32}, Delta=10,566,400 ps.
- Eight depth guards: P in {4,8}, both attachments, S=2, W in {0,32},
  Delta=10,566,400 ps. Together with the primary grid these cover all six
  formerly void two-spine W=32 pipeline configurations and unloaded companions.
- Four window crossover cells: P=2, both attachments, S in {8,2}, W=0,
  Delta=6,572,800 ps. The corresponding high-window cells already belong
  to the primary grid.

Run every physical cell once with tracing disabled and once with tracing
fully enabled. The 48 native executions are 24 paired configurations; they
are not 48 independent behavioral samples. Exact completion CSVs, GOAL bytes,
physical drain and request metrics must match in each trace pair. A trace
file is an observation, never a second authority or a scheduling input.

Keep two separate unscored control populations. Twelve legacy physical cells
cover both attachments, P in {2,4,8}, S=8 and W in {0,8}, with all added
recovery selections disabled and automatic window selection. Their accepted
completion CSVs, declared full-bisection fabric manifests, topology and endpoint permutation
must remain exact against the retained original records. Thirty-six ideal
cells cover both attachments, S in {8,2}, P in {2,4,8}, W in {0,8,32}, using
rnic-nn and the accepted inputs. Each must match the corresponding retained
ideal completion CSV. Ideal results are topology-bypass controls, never
physical switch evidence. The total is 96 native executions, with 24 physical
comparisons and 48 separate compatibility controls.
Declared fabric identity means `fabric.json`; native diagnostic manifest text
may include the already-shipped recovery rows and is not an old byte oracle.

No re-seeding, shifted traffic release, altered payload, alternative load
ladder, lowered window or additional sensitivity cell is allowed after the
first run under this freeze. A necessary successor requires a new freeze.

## Bounds written before observations

Floor: one PP payload needs 1,310,720 ps of wire service at 400 Gbit/s. Its
forward-link floor is 3,310,720 ps on rail and 5,310,720 ps on node-local.
The PP final-stage floor is `P*1,000,000 + (P-1)*hop_floor` ps. These path and
cut floors apply only to the physical profile, since rnic-nn bypasses the
topology. A finite
receiver window or control exchange can add time but cannot beat these floors.

Ceiling: with no competing packets, data-only store-and-forward traversal is
bounded by `H*1,310,720 + H*1,000,000` ps for H=2 or H=4. This is not a bound
on receiver-visible completion with Ring-CAM holding, nor on retry recovery.
There is no unconditional finite FCT ceiling from payload bytes alone. Retain
the DATA recovery study's per-cell engineering completion budget as a named
operational guard, not a physical theorem, and publish its calculation before
execution. Its metric is the complete-flow phase, `max(completion)-min(start)`,
including all PP and EP flows. Let S_work be the maximum of receiver payload
service over all input flows, EP endpoint/cut serialization without propagation,
and the PP final-stage floor. The unchanged formula at 400G is
`9*S_work + 8*(40,000,000 + Q_allowance)` ps, where
`Q_allowance = 3*1,048,576*20 + 8*1,000,000 + 4,160*20 = 70,997,760` ps.
Apply it to every declared cell, with the fixed window reported separately;
do not enlarge any inherited formerly-void-cell budget. TTFT and physical
drain are distinct quantities and never substituted for this phase metric.

EP endpoint byte floors are 146,800,640 ps at W=8 and 587,202,560 ps at W=32.
The busiest directional leaf cut carries 0/176,160,768 bytes for rail W=8/32,
and 7,340,032/117,440,512 bytes for node-local W=8/32. Cut service is
`bytes*20/S` ps. Add two propagation hops to a nonempty endpoint bound and
four to a nonempty cut bound, then take the maximum for the EP phase floor.
These counts come from actual directed input pairs and are checked again
against the emitted traffic. The aggregate cut does not identify the queue
seen by a particular routed PP packet.

For a tagged packet on one chosen 400G physical egress, q wire bytes ahead
require q*20 ps, not the two-uplink aggregate q*10 ps. An already serializing
packet contributes only its remaining service, not its full bytes. Without
independent queue observations, the guaranteed extra tagged delay remains zero.
A flow completion below a byte or causal floor voids its configuration.

## Read-only trace and identity contract

Connect the existing ns-tm3 queue observer to the supported direct htsim_rnic
execution path. Preserve the runtime and switch as the only mutable owners.
Record ordered queue enqueue, service-start, service-end and drop observations
with timestamp, physical switch and egress, ingress, priority, packet kind,
wire size and stable packet lifecycle identity. Join each original and retry
copy to flow ID, source, destination, message tag, logical packet index and
attempt; control packets remain distinguishable from DATA. Raw traces retain
all competing traffic needed to reconstruct a selected PP egress interval.
Capture all egresses from time zero. Each physical packet is joined by run and
HTSIM packet ID to its lifecycle ID and 64-bit Atlahs flow ID; the switch's
internal 32-bit PacketFlow ID has a separate explicit map. A retry is a new
physical packet, while logical DATA identity is flow ID plus packet index.
Controls retain their own physical identity and the existing gap, resolution
or retirement metadata they reference. Do not allocate packet IDs or select
routes early merely to generate observations.

Also project the existing collective runtime's source service, expected arrival,
actual destination arrival, admission decision, logical Ring-CAM release,
receiver service start/end, packet delivery and flow completion. Preserve
original versus retry identity, including late admission and discarded copies.
Record retry authorization time, logical packet, authorized attempt and cause:
received GAP_NACK, probe timeout or legacy timeout, with the existing control
lifecycle or timer-origin attempt and deadline. Identify the packet that
triggers delivery progress; a late earlier hole may release already-finished
later packets, so the last logical index is not automatically the predecessor.
Record the flow request map and actual packetization so classification never
comes from a guessed numeric range or a pointer address. Stable IDs and an
explicit observation order resolve repeated timestamps. Missing, duplicated,
misbound or impossible observations are fatal. Failed trace writes fail closed.

Trace selection is rejected on unsupported profiles. Absence preserves existing
CLI/configuration behavior and exact completion artifacts. Trace output cannot
change random draws, event insertion order, byte service, admission, recovery,
flow timestamps or the quiescence boundary. It must finish and close its files
before successful process exit; partial files never count as complete evidence.

## Exact queue and receiver accounting

Switch submitted and eligible times equal enqueue; start is dequeue and finish
is serialization completion. Downstream visibility is recorded at the next
consumer after propagation. At the receiver, submitted means actual arrival,
eligible means logical Ring release, and start/end are RX service boundaries.
Ring residence is protocol holding; RX queue wait is RX start minus logical
release, and subsequent ordering delay is logical delivery minus RX end.
Source flow release to packet transmission is a dispatch offset containing
pacing and eligibility effects. Without a separate eligibility observation it
is not labeled queue wait.

For one PP packet, enqueue a and egress service start b give queue wait W=b-a.
Reconstruct every other packet's service interval [s_j,f_j) on that egress.
For PFC-disabled work-conserving service, W equals the sum of intersections
of those intervals with [a,b), exactly. Partition the sum into EP DATA,
EP controls, PP/other controls and other DATA. Include any packet already
in service at a through its residual intersection. Intervals do not overlap;
service duration is the exact wire-byte serialization of that egress.

Independently identify EP DATA still buffered at a with strictly earlier
enqueue times and the same DATA priority. Their total wire bytes q, plus the
remaining service r of an active EP DATA packet, give the conservative bound
`W >= q*20 + r` ps under the frozen oldest-head-first policy. Same-time
arrivals are excluded from this lower-bound subset unless their earlier
arbitration order is explicitly proved. Do not substitute backlog-after-enqueue:
it contains the tagged packet and overcounts an active packet's remaining work.

For each admitted on-time original, logical receiver release must equal
`ceil((ETA+Delta)/tick)*tick`, with admission observed at actual arrival.
For a late admitted retry it equals `ceil(arrival/tick)*tick`. Check the
actual receiver service and subsequent delivery from their authoritative rows.
A larger actual arrival within an unchanged release window can reduce holding
by exactly that amount, leaving release and completion unchanged. Report this
absorption explicitly. Do not subtract Delta blindly from flow completion.

The packet that determines a PP flow's delivery must be identified, including
its actual causal predecessor where receiver or packet ordering matters.
Only waits on the realized completion path may explain additive flow or
request latency. Per-packet and per-queue sums remain separately named work
accounting; they may exceed elapsed time and never substitute for PP FCT.
Flow CSV completion on this path is receiver delivery. RETIRE and other
physical cleanup continue separately to verified quiescence.

## Behavioral relations and decision

R1, unloaded topology/window crossover: the four P2/W0 low-versus-high pairs
must obey the rounded Ring-CAM release rule at the packet level. The direction
of receiver holding is nondecreasing with Delta. Any whole-flow difference
must be reconstructed from authoritative arrival, release and receiver-service
rows; no unconditional exact FCT subtraction is asserted. Rail's PP data
uses no uplink, so a spine-count effect at fixed Delta and W=0 requires a
separately identified control or arbitration mechanism, not a shared-cut claim.

R2, actual expert service ahead: compare W=0,8,32 at P2/S2/node-local and
W=0,32 for P4/P8. Hypothesis: W32 puts positive EP DATA service ahead of at
least one PP packet for each P. The lower-bound and exact service-accounting
relations above decide this; merely overlapping EP and PP wall-time intervals
is insufficient. Report the zero-load and disjoint rail controls separately.

R3, visible pipeline penalty: hypothesis, at each P on S2/node-local with
fixed Delta and recovery, W32 increases PP hop maximum and final-stage request
latency above W0. Each p99 is the maximum of P-1 hops, not a population tail.
Positive expert queue service may instead be absorbed by receiver holding.
Such a null result is interpretable and refutes R3; it does not satisfy the
existing TRAF-88 positive request-penalty acceptance. Never tune Delta to
manufacture a positive result after observing absorption.

R4, cut response: compare S2 versus S8 at P2/W32 for each attachment. EP
phase floors strengthen by the declared cut-capacity relation, but the same
multiplicative relation is not asserted for observed phases that include
additive control and recovery. Report observed phase ratios and their floors;
explain every violation of a physical bound. These are diagnostic ratios,
not extra scored behavioral successes by construction.

Use the existing final-stage CompletionEvent and singleton StepResult
projection. Request time to first token (TTFT) must equal the PP final-stage
boundary and conserve stage compute, hop FCT and declared GOAL quantization
exactly. This study does not measure time per output token (TPOT). Background-
inclusive job completion, EP phase duration and physical drain remain distinct.

## Fatal guards and publication

All configuration populations, input hashes, packet/message identities and
bytes, directional capacity, service conservation, queue reconstruction,
receiver release, request causality, exact off-path comparisons, retry limits,
engineering completion budgets and physical quiescence are fatal unscored
guards. A violated guard voids that configuration and any relation depending
on it. Other configurations remain interpretable as explicitly permitted
here. The aggregate run is void if any required configuration is void. Failed
or incomplete configurations have null request metrics, never partial estimates.

Keep configurations, exact-oracle rows, behavioral families and instances,
structural guards, native executables and unit cases in separate evidence
classes. An identity or forced zero never improves a behavioral denominator.
Publish raw observations before evaluating acceptance, retain every attempt,
and freeze source/binary/runner identities before execution. Bulk traces stay
outside Git; only compact audited projections and the report are tracked.

TRAF-88 closes only with valid completion of every required configuration,
independent positive tagged-queue evidence and the registered positive live
request penalty. Otherwise retain and narrow its exact registered remainder,
with the refuted assumption and roadmap consequence stated plainly. CORE-8,
BACK-38, TRAF-8 and hardware calibration do not close through this study.
