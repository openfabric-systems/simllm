# Oversubscribed pipeline rail: frozen expectations

This file is committed alone before implementation, tests or native runs.
TRAF-88 owns this contention follow-up to pp_rail_topology_v1. All evidence
is from a declared packet simulator, without hardware calibration. The root
AGENTS.md is absent in this worktree; the supplied wave contract applies.

## What will run

The complete grid has 72 cells: attachment {rail, node-local}, spine count
{8, 2}, pipeline width P {2, 4, 8}, concurrent expert-parallel participant
count W {0, 8, 32}, and profile {rnic-nn, rnic-cn}. Eight spines are 1:1;
two spines are 4:1 at the frozen equal 400 Gbit/s endpoint and uplink rates.
Production additionally supports four spines and a separate uniform uplink
rate, with unchanged defaults. Each leaf connects once to every spine.
There are 64 endpoints, eight leaves and 64+8S links. Link propagation is
1,000 ns and switch latency is zero. Endpoint numbering must not change.

Use the accepted study's exact graph, seed 1, placement and packet settings.
Stage s occupies node s, semantic rank 8s, on NIC zero. Each stage computes
for a declared 1,000 ns. One forward boundary sends B=65,536 payload bytes,
from eight tokens, hidden width 4,096 and two bytes per element. P stages
produce P-1 transfers. This adds stage service with P, not fixed-model speedup.
TP is one, with zero makespan, unscored. EP starts at time zero, on NICs
1 through 4 in NIC-major order across eight nodes. Each remote directed pair
sends M=1,048,576 bytes; node-local pairs are excluded. W=8 has 56 messages
and seven remote peers per rank; W=32 has 896 and 28. PP and EP endpoint
links are disjoint. No backward or combine traffic is implied.

Use HtsimStepSink's native GOAL execution boundary with the accepted single
concurrent graph renderer. Its normal ordered-artifact planner would alter
overlap, so a study-local adapter will retain the one-program schedule. A
read-only final-stage completion projection will return StepResult and a
one-token request metric from the packet timestamps. Do not feed those times
into a second scheduler. This is a narrow declared singleton-stage projection,
not a general packet runtime or adapter-captured PP integration.

## Physical sanity before reading results

At 400 Gbit/s, serialization costs 20 ps/byte. A PP payload needs 1,310,720 ps.
Adding H link delays and H-1 switch latencies gives data-arrival floors of
3,310,720 ps (rail H=2) and 5,310,720 ps (node-local H=4). Unloaded data-only
store-and-forward ceilings are H*1,310,720 + H*1,000,000 ps, respectively
4,621,440 and 9,242,880 ps. These are not sender-acknowledged FCT ceilings.
One 4,096-byte payload packet plus 64-byte header costs 83,200 ps. The null
profile bypasses the switch graph and is tested only against serialization.

Two 400G uplinks have a shared-throughput ceiling of 800 Gbit/s, or
100,000,000,000 bytes/s; eight have 3,200 Gbit/s. For Q payload bytes that
must cross one leaf's uplink cut, drain time is at least Q*20/S ps. The
following per-leaf counts and drain floors are fixed for every P and profile;
only rnic-cn can test physical topology. "PP leaf" means a leaf traversed
by the chain. Rail PP's leaf has no EP traffic and PP uses no uplink.

| Attachment | W | EP bytes on PP leaf's uplinks | EP bytes on busiest leaf's uplinks | S=8 drain floor (ps) | S=2 drain floor (ps) |
|---|---:|---:|---:|---:|---:|
| Rail | 0 | 0 | 0 | 0 | 0 |
| Rail | 8 | 0 | 0 | 0 | 0 |
| Rail | 32 | 0 | 176,160,768 | 440,401,920 | 1,761,607,680 |
| Node-local | 0 | 0 | 0 | 0 | 0 |
| Node-local | 8 | 7,340,032 | 7,340,032 | 18,350,080 | 73,400,320 |
| Node-local | 32 | 117,440,512 | 117,440,512 | 293,601,280 | 1,174,405,120 |

These are directional counts, not sums of ingress and egress. At W=32 a
rail EP leaf sends 8*21 MiB across rails; a node-local leaf sends 4*28 MiB.
At W=8 all rail EP peers share one leaf, so no EP data uses a rail uplink.
EP endpoint payload floors are 146,800,640 ps at W=8 and 587,202,560 ps at
W=32. The EP phase floor is the maximum of endpoint service plus two link
delays and uplink-cut drain plus four link delays (omit the empty cut).
The effective serialization-floor ratio S=2 versus S=8 is one at W=8,
three for rail W=32 and two for node-local W=32. The raw cut ratio is four
only for a nonempty cut; neither ratio is a theorem about measured FCT ratios.

For each PP hop, total potentially competing EP bytes are the PP-leaf count
above. However, release at time zero does not imply all those bytes precede
the PP hop. With no packet queue trace, the guaranteed EP bytes ahead of
the tagged PP packet are zero in every cell, and the unconditional added
queue-delay floor is zero. The PP data-arrival floor therefore stays
3,310,720 or 5,310,720 ps for every P,W,S cell. The study must publish those
per-cell bounds before launching the backend, and publish PP/EP temporal
overlap afterward. It must not relabel a whole-phase drain floor as a tagged
hop FCT floor. The conditional bound, if q bytes are independently known to
precede the tagged packet on the two-uplink cut, is q*10 ps, plus its own
serialization and path delay. No positive q is asserted by this freeze.

The step floor is P*1,000,000 + (P-1)*hop_data_floor ps. Critical-path shares
lie in [0,1]; non-compute share is at least 1-P*1,000,000/step_floor for
physical cells. EP is concurrent background and is never added to this PP
chain. A loose payload-work scale is 4*(total_EP_bytes+(P-1)*B)*20 plus
four link delays. It is not a control-inclusive FCT ceiling. There is no
finite unconditional sender-completion ceiling from payload bytes alone.

## Exact checks and behavioral hypotheses

R1: at W=0, all four fabrics agree to 0 ps on rnic-nn for each P. This is
topology bypass, not proof of physical equality.

R2: the 1:1 rnic-cn cells at W=0 reproduce every accepted p50 and p99 exactly,
including P=8 p99 10,335,600 ps rail and 12,585,200 ps node-local. Lock default
manifest bytes, topology text and endpoint permutation before modification.

R3: on 4:1 node-local rnic-cn, PP p99 is nondecreasing from W=0 to 8 to 32
and strictly larger at W=32 than W=0, for every P. Its non-compute critical
share follows the same direction. This is a hypothesis about scheduling,
not a consequence of whole-phase byte conservation.

R4: at each P,W, 4:1 rail rnic-cn PP p99 stays within one packet serialization
(83,200 ps) of the corresponding 1:1 value, because PP uses no uplink.

R5: for W>0, EP makespan on 4:1 is at least its 1:1 value times the effective
serialization-floor ratio above, on each physical attachment and P. Also
report the requested stronger raw-cut-ratio candidate of four when the cut
is nonempty. Both are hypotheses; fixed control cost and an endpoint-limited
baseline can refute them. W=0 has no phase and is unscored. Null-profile
results are controls, never physical oversubscription evidence.

All refutations remain visible. No re-seeding, changed load, changed routing,
or silently revised hypothesis is allowed after the first run.

## Evidence, guards and artifacts

Each cell reports nearest-rank PP FCT p50 and p99 of P-1 messages (p99 is a
finite-chain maximum, not a population tail), EP/TP phase makespans, PP final
stage completion, background-inclusive job completion, and two PP shares:
sum(hop_FCT)/step and (step-P*compute)/step. Their difference is the declared
GOAL gates/quantization, separately recorded. Final stage completes at
floor(last_PP_completion/1000)*1000 + 1,001,000 ps, the accepted calc(0)
gate projection. With no EP this must equal native job completion exactly.

Fatal guards, void-not-score: manifest/wire validity and bijection, exact
message identities/counts/bytes, no duplicate completions, integer positive
flow times, receive-before-next-stage causality (the accepted 1 ns rounding
tolerance), physical data and EP cut floors, quiescence, isolated final-stage
projection, live StepResult/request-metric conservation, unchanged default
artifacts, and accepted reference input identity. Native exceptions/timeouts
void that cell. No fatal guard is intended to be survivable for its cell's
timing claims. Other clear cells and static byte accounting stay interpretable
when one cell fails; relations involving a void cell are unscored. Relation
refutations do not void cells. Native runs and deterministic tests are separate
evidence classes with no pooled score.

Bulk evidence lives under SIMLLM_DATA_ROOT or --out, outside the repository.
Executables come from SIMLLM_HTSIM_BUILD, SIMLLM_HTSIM_RNIC and SIMLLM_TXT2BIN.
Track only a compact result table and plain labeled matplotlib PNG/PDF figures.
Digest tracked text using raw or LF-normalized bytes, write artifacts as LF
bytes, and add text eol=lf attributes. RESULTS.md cites this freeze commit.

TRAF-88 closes only if all acceptance is met, including an honestly justified
shared-byte floor and non-void load-dependent physical penalty. Otherwise its
existing entry is narrowed to the exact remainder. TRAF-8, general discovery,
microbatch overlap, backward traffic, calibrated TTFT/TPOT and flow-hash routing
remain outside this work. No backend source or default path is changed.
