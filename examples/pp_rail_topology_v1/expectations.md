# Pipeline activations on a declared rail fabric: frozen expectations

This freeze precedes implementation and the first run. TRAF-8 owns the forward
pipeline-parallel (PP) traffic slice; PLACE-1 owns the fixed manifest variants.
Neither task closes. The orchestrator registers the topology-study residual
against merged main. All results are simulator evidence, without GPU calibration.

## Inputs and sweep

Run 36 configurations: fabric `{rail, node-local}` times PP width `{2, 4, 8}`
times expert-parallel (EP) participant count `{0, 8, 32}` times packet profile
`{rnic-nn, rnic-cn}`. The EP count is the requested fan-in sweep label: a group
of W ranks has W-1 logical peers per rank, not W incoming messages. Intra-node
pairs stay off the external fabric. Report actual remote peer counts separately.

Reference: eight nodes, eight graphics processors (GPUs) per node, one affine
400 Gbit/s network interface controller (NIC) each. Eight leaves connect to
eight spines with one 400 Gbit/s link per leaf-spine pair. Every link has
1,000 ns propagation and switch latency is zero, preserving the declared Clos.
Semantic rank is `8 * node + NIC index`. Rail: NIC i on every node attaches to
leaf i, port node. Node-local: all NICs on node n attach to leaf n, port i.
Use the existing fabric schema and validate every physical attachment.

One PP chain uses semantic ranks `0, 8, ..., 8*(P-1)`. Stage s is on node s,
on NIC zero throughout. Each stage computes for a declared 1,000 ns. This is
an explicitly synthetic service input, not a transformer calibration. Its
fixed service is identical across fabrics and profiles. Tensor parallelism
(TP) is one; its phase makespan is therefore zero and is unscored.

One step schedules eight new tokens, hidden width 4,096, two bytes per element:
each boundary sends exactly B = 65,536 payload bytes, forward only. Exactly
P-1 messages and `(P-1)*B` bytes exist. Stage compute precedes its send;
receiving the activation precedes the next stage's compute. Stage service
is not divided by P: increasing P adds stages to this declared task.

Background EP starts at time zero on NICs other than zero. W=8 uses NIC 1 on
all eight nodes. W=32 uses NICs 1 through 4 on all eight nodes, in NIC-major
order. Each directed remote pair sends 1,048,576 bytes in one sparse pairwise
all-to-allv; no reverse combine is implied. There are 56 and 896 remote
messages respectively, with seven and 28 remote peers per endpoint. Local
pairs are excluded before fabric rendering. The workload is a declared EP
network-contention probe, not a complete mixture-of-experts layer. This choice
keeps PP endpoint links disjoint from EP endpoint links in both fabrics.

Run the pinned htsim build identified in the result provenance with a fixed
seed and identical settings across the grid. Configure executables and bulk
output using `SIMLLM_HTSIM_RNIC`, `SIMLLM_TXT2BIN` and `SIMLLM_DATA_ROOT`.
Do not put bulk GOAL programs, logs or completion tables in Git.

## Physical bounds, stated before measurement

At R = 400 Gbit/s, payload serialization costs exactly 20 ps per byte.
A PP hop cannot beat B/R = 1,310,720 ps of serialization. A physical one-leaf
path has two links, giving a propagation floor of 2,000,000 ps; a spine path
has four links, giving 4,000,000 ps. The joint cut-through data-arrival lower
bounds are 3,310,720 ps and 5,310,720 ps. Packet headers, store-and-forward
switches, acknowledgments and endpoint control can only add service or delay.
A null-network profile that bypasses the topology must be identified as such;
it cannot validate those topology-specific floors.

The data-only unloaded cut-through ceiling equals the joint floor. With full
packet store-and-forward, an intentionally loose data-arrival ceiling is
H*B/R + H*D + (H-1)*L, where H is the link count, D is link delay and L is
switch delay. These are data-arrival bounds, not promised sender-acknowledged
flow completion times (FCTs). Sender completion may need a return path.

For one work-conserving shared directed link, additional queue delay lies
between zero and Q/R, where Q is all other bytes that can precede the tagged
flow on that link. A conservative payload budget is
`Q <= W_remote_messages * 1,048,576 + (P-1)*65,536`.
Summing this budget over four links gives a conservative payload-work ceiling
for the spine path. Report this envelope separately from measured FCT.
Control messages, retransmissions and deliberately idle congestion-control
service invalidate a payload-only FCT ceiling; do not call that envelope a
physical bound on such a transport. There is no finite unconditional FCT
ceiling from payload bytes alone without progress and loss assumptions. A
120-second process timeout is an operational failure bound, not a physics claim.

The declared PP step data-arrival floor is
`P*1,000,000 + (P-1)*(B/R + H*D)` ps. The serial data-arrival ceiling sums
the hop store-and-forward/shared-link envelopes and stage compute. Actual
sender-visible runtime completion includes control delays and nanosecond GOAL
quantization. Never report the sum of overlapping EP waits as PP critical time.

## Requested assertions and predicted limitations

R1, exact-oracle candidate: on the rail, PP FCT equals B/R + 2D + L,
3,310,720 ps, independently of EP width. Test equality in integer picoseconds
for each profile and all PP widths. This claim assumes data-arrival completion
and no packet overhead. It is intentionally tested without correcting the
requested oracle after observing the transport's completion definition.

R2, behavioral family: rail PP p99 is independent of EP width, exact to 0 ps
for the fixed P/profile. Its reserved leaf and endpoint links carry no EP data.

R3, behavioral family: node-local PP p99 is nondecreasing from W=0 to 8 to 32,
with a strict increase between 0 and 32. Compare the change to the shared-link
payload budget above and explain any failure. A fully provisioned Clos plus
adaptive routing need not expose a collision on every finite flow schedule;
this expectation is a hypothesis, not a conservation identity.

R4, exact-oracle candidate requested in the brief: with W=0 the two fabrics'
PP hop FCTs agree to exactly 0 ps on rnic-nn. A null network can satisfy this
by bypassing the topology. A topology-sensitive physical transport cannot
simultaneously have exact R1 on both paths and equal propagation: the spine
path adds two link delays. Such equality is not evidence of physical equality.

R5, behavioral family: in an isolated rail chain, adding one PP stage adds
one stage compute and one hop service, within 2 ns of cumulative whole-ns
GOAL rounding per boundary. For a fixed P, increasing background load is
expected to increase the node-local PP critical-path fraction. Report any
nonmonotonicity; do not silently filter or re-seed runs.

## Metrics and evidence classes

Per configuration report PP hop FCT p50 and p99, using nearest-rank empirical
quantiles of the P-1 forward messages. The p99 is a finite-chain maximum for
these sample sizes, not a measured latency-tail distribution. Retain every
flow's start, completion and FCT in the bulk table. Report EP phase makespan
from its time-zero release, TP phase makespan zero, PP step completion and
PP communication critical-path share. The PP request completes at the final
stage, independently of background EP draining; report background-inclusive
job completion separately. Do not rename the latter PP step completion.

Compute share from the realized serial PP chain: final stage completion
minus the sum of stage computes, divided by final stage completion. The
complement is compute service, including declared GOAL quantization where
applicable. Request time to first token (TTFT) uses the step's final-stage
completion. Time per output token (TPOT) needs a multi-step request and is
outside this one-step probe. A separate coarse-runtime integration exercise
must demonstrate graph to CompletionEvent to StepResult to request metrics,
without passing packet FCT through a second timing authority.

Normalize physical-profile FCT against rnic-nn only for aligned starts. Later
PP boundaries start at model-dependent times, so compare phase makespans and
explicitly omit invalid per-flow ratios there. The first boundary is an
aligned-start candidate and must be joined by semantic source/destination,
tag and bytes after undoing any endpoint permutation.

Fatal, unscored guards: bijective endpoint projection, exact inventory and
bytes, positive integer times, causal stage order, completion-row identity
and no duplicates, backend physical quiescence, serialization floor, graph
and wire validation, and exact P=1/off-path artifacts. A violated guard voids
the run; preserve findings and close nothing. Native executables and unit
tests are reported separately. There is no aggregate denominator mixing
configuration counts, oracle candidates, relation families and fatal guards.
A false R1/R4 oracle or R2/R3/R5 hypothesis is a refutation with tasks open;
it is not scored as a fatal-guard fraction or treated as successful closure.

## Off path and scope

The PP feature is explicit opt-in. P=1 emits no activation and preserves the
existing step graph, GOAL and serialized records byte for byte. Existing
manifest builders keep their default bytes. Test malformed widths, membership,
rank reuse and causality deterministically, without native binaries or a GPU.
The first slice declares stage attribution externally; scheduler-captured PP
stages, microbatch overlap, backward traffic, general NIC discovery and a
calibrated end-to-end serving deployment remain outside this slice.
