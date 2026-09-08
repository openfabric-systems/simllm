# Early-burst pipeline arrival qualification: frozen expectations

This expectations-only commit precedes the successor runner and every new
simulation. TRAF-88 owns the qualification. The valid but refuted all-depth
queue study in `examples/pp_rail_contention_v2` supplies prior evidence for
this narrower hypothesis. Its results and freeze remain unchanged. The new
arrival phases have not been measured. This is prospective qualification
informed by published prior results, not an independent discovery of the
retry regime or a new public pre-registration of the old cases.

## Physical scope and workload

Keep the predecessor's 64 endpoints, eight leaves, 400 Gbit/s links, one
microsecond propagation per link, seed one, 4,096-byte DATA payload and
64-byte header. A pipeline stage occupies rank 8*s on NIC zero. Each forward
handoff sends 65,536 payload bytes. Expert-parallel (EP) background traffic
uses NICs one through four, starts at zero and sends 1,048,576 bytes per
remote ordered pair. Pipeline depth P is four or eight in the held-out cells;
EP width W is zero or 32. Both rail and node-local attachments use two spines.
The topology, placement, message inventory and immutable collective plans
retain the predecessor's interfaces and bytes wherever the input is unchanged.

Vary the first stage's declared compute duration C0=1,000,000+A ps, with
A in {16,000,80,000} ps. Every other stage still computes for 1,000,000 ps.
This is genuine declared synthetic stage work, not a timing gate disguised
as compute and not an implementation of external request arrival. The first
communication starts at A+1,001,000 ps, including the existing one-nanosecond
GOAL gate. The two offsets probe one receiver tick and just less than one
83,200 ps full DATA frame. They qualify only this early-burst, seed-one
neighborhood. No monotone law over arbitrary arrival phases, seeds, loads,
payloads or pipeline depths is asserted. Matched loaded and unloaded cells
have identical stage compute, so its direct cost cancels in their difference.

Keep the explicit receiver window Delta=10,566,400 ps and tick=16,000 ps,
1,048,576-byte switch and Ring-CAM capacities, 131,072-byte control headroom,
10,000,000 ps control deadline, 900,000 ppm margin, 64-byte control wire size,
exponential DATA recovery, four control windows per base probe, eight retries
and the final 50,000,000,000 ps timeout. Initial-window budgeting is disabled.
Priority flow control remains disabled; routing, oldest-head-first switch
arbitration and every sender/receiver policy remain unchanged.

## Finite populations

The new physical population has exactly 16 configurations: both attachments,
P in {4,8}, W in {0,32}, and A in {16,000,80,000} ps, all with two spines.
Run each once with trace off and once with trace on, giving 32 executions and
16 exact observation pairs. Trace selection must preserve the entire native
completion CSV, request metrics, flow inventory and physical quiescence time.

Run 16 corresponding ideal `rnic-nn` controls. For every PP flow, start and
completion must equal the predecessor's identical P/W/attachment/S2 ideal
row plus A; FCT is identical. Every EP row keeps its start, completion, bytes
and FCT exactly. Stable endpoint/tag/payload identity joins the rows and must
be complete and unique. The expected transformation is written before the
run and does not copy an observed physical timestamp.

Run twelve exact v2 regressions with A=0, both attachments, P in {2,4,8},
W in {0,32}, S2, fixed window and the same recovery. Trace remains off.
Compare their entire completion CSVs and request metrics against published
v2 locks, including the two-stage absorption and deeper recovery cases.
Retain and hash the full predecessor trace audits as prior evidence; those
observations do not become new independent behavioral samples.

Run twelve exact historical full-bisection physical controls with A=0,
both attachments, P in {2,4,8}, W in {0,8}, eight spines, automatic receiver
window and added recovery disabled. These are the predecessor's protected
legacy configurations, not new fixed-window penalty comparisons. Default
fabric manifests, topology and endpoint permutation must remain exact.

The total is 72 native executions: 16 paired physical configurations,
16 ideal controls, twelve fixed-window regressions and twelve legacy controls.
Evidence classes remain separate. No added phase, load, window, seed or
recovery sensitivity cell is allowed after the first run under this freeze.

## Bounds before observations

Floor: each PP payload needs 1,310,720 ps of 400G wire service, plus at least
two propagation hops on rail or four on node-local. The request floor is
`P*1,000,000 + A + (P-1)*hop_floor` ps. Every flow and every receiver's
ordered completion prefix must respect the corresponding byte and propagation
floor. The complete physical flow phase must not beat its identical ideal
phase. Shared-receiver per-flow ratios remain diagnostic, not fatal floors.

Ceiling: unloaded data-only store-and-forward traversal is bounded by
`H*(1,310,720+1,000,000)` ps at H=2 or H=4, excluding receiver holding and
recovery. Payload alone gives no unconditional receiver-visible FCT ceiling.
Retain the predecessor engineering complete-flow phase budget,
`9*S_work + 8*(40,000,000 + 70,997,760)` ps, with S_work the maximum of
receiver payload service, EP endpoint/cut serialization and the request
floor including A. This operational guard is not a physical theorem.
The W32 S2 budgets remain 11,457,628,160 ps for node-local and
16,742,451,200 ps for rail, since EP service dominates the small compute
shift. Do not enlarge either loaded budget after observations.

A physical 400G egress charges 20 ps per wire byte. A 4,160-byte DATA frame
therefore occupies 83,200 ps. The two-spine aggregate is not the service rate
of a tagged packet's selected egress. Reconstruct competing service over the
tagged packet's actual wait interval, including only the residual service of
an already active packet. The independently buffered, strictly earlier EP
DATA subset supplies a conservative bound. Aggregate cut bytes and summed
visit waits are work accounting, not additive request delays.

## Prospective behavioral hypotheses

B1, visible early-burst penalty, has four instances, one for each P/A pair on
node-local S2. Comparing W32 with its matched W0, both the largest PP-hop FCT
and time to first token (TTFT) increase by between 2,400,000,000 and
2,800,000,000 ps inclusive. This is an engineering acceptance band informed
by the predecessor, not a theorem that all delays lie between those limits.
No monotonicity with A or P is predicted. Each largest-hop statistic is a
maximum over P-1 hops, not a sampled population tail.

B2, independent contention, has the same four instances. At least one PP
packet has positive EP DATA service ahead on an observed physical egress,
with a strictly positive independent earlier-enqueue/residual lower bound.
All visits must also pass exact competing-service accounting. A wall-time
overlap or a shared leaf alone is insufficient evidence.

B3, retry mechanism, has four instances. In the loaded cell's slowest PP hop,
the delivery-triggering packet is attempt seven of one logical DATA packet;
its original and attempts one through six are fabric-dropped. Attempt one
is authorized by a physical GAP_NACK. Attempts two through seven follow
probe deadlines at 40,80,160,320,640 and 1280 microseconds after the preceding
transmission end. These six windows sum to exactly 2,520,000,000 ps. They
explain the hypothesized scale; they are not themselves the entire request
penalty. Additional authorization-to-transmission offsets stay explicitly
named dispatch offsets because source eligibility is unobserved.

For every positive cell, join that successful transmission and its expected
arrival to actual arrival, receiver release, service and ordered delivery.
Select the receiver maximum and identify any earlier logical hole releasing
already received later packets. Reconstruct the flow's trigger timeline and
the final request boundary exactly. Unchanged or shifted expected-arrival
release is an admissible mechanism; actual-arrival selection is not required.
At each of those seven drops, the trace records the base shared-switch and
target-egress buffered bytes partitioned into EP DATA, EP controls and other
traffic, excluding additive control reserve and packets already in service.
The existing capacity rule must explain rejection. B3 additionally predicts
positive EP DATA occupancy and that removing just those EP DATA bytes at the
same observed instant would permit the rejected frame at both admission
limits. This is a local admission counterfactual, not a replay without EP.
It connects background occupancy to the losses in the selected retry chain.
The optional audit projection changes no native observations or timing;
with the option absent, the old audit output remains exact.

A different valid retry mechanism refutes B3 even if B1 holds. Missing or
contradictory observations are fatal rather than a mechanism refutation.

For the four disjoint rail P/A pairs, require exact equality of every PP
flow's start/completion/FCT and all original PP source, expected-arrival,
arrival, receiver-release, service and delivery times between W0 and W32.
Require zero EP DATA service in PP egress waits. This off-path identity is an
unscored fatal guard. No whole-background-phase equality is expected.

## Request projection and fatal guards

Use the existing semantic graph, serial GOAL renderer and direct physical
backend. The final-stage CompletionEvent is the authoritative last PP receive
followed by the existing one-nanosecond frontier and one-microsecond compute,
with the printed GOAL boundary rounded down to a nanosecond. Successive PP
starts equal preceding delivery plus 1,002,000 ps. A singleton StepResult
projects this event to TTFT. Its exact decomposition is
`P*1,000,000 + A + sum(PP FCT) + gates_and_quantization`, where the last term
is `2,000*(P-1) - last_delivery%1,000` ps. No queue-work sum is added again.
The study measures one forward prefill request and no time per output token.
Whole-job completion, EP phase and final physical quiescence remain separate.

Strict v2 trace reconciliation remains authoritative for identities, bytes,
source/switch/receiver serialization, buffer capacity, original/retry lifecycle,
control authorization, Ring-CAM rules, receiver ordering and quiescence.
Every native run records its inputs and logs before interpreting outcomes.
Missing/duplicate population members, changed frozen inputs, incomplete traces,
identity or causal mismatches, physical-floor or engineering-budget violations,
failed native execution and changed exact controls void the aggregate run.
No void run has a behavioral score or closes a task. Individually valid rows
may be described as findings but cannot rescue aggregate closure.

Freeze source, binary, runner and expectation identities before execution.
Raw traces stay outside Git. A compact public projection retains verdict,
all metrics, relations, identities, causal witnesses and hashes of omitted
bulk evidence without rescoring it. The old v2 controls keep their original
chronology and evidence classification.

TRAF-88 closes only if all 72 executions are valid, every exact and physical
guard holds, and all four instances of each of B1, B2 and B3 hold. Closure
means this explicitly scoped early-burst pipeline penalty is qualified.
The general all-depth contention claim remains refuted. Captured serving,
microbatches and general request metrics remain TRAF-8; persistent physical
execution remains BACK-38. No hardware calibration or milestone beyond this
narrow declared-pipeline qualification is inferred.
