# Collective width tail: frozen expectations

S3 studies COMP-9, latency-tail fidelity in the network, batching and queueing
chain. These expectations precede implementation and every study run. The
result report cites this expectations-only commit. Predictions below are
analytical, not measured values. Backend source pin: `617ce20`.

## BACK-68 guard amendment before the fresh rerun

The controlled aligned_baseline_v1 experiment frozen at `7078cc5` supports
scheduler-dependent per-flow completion under receiver contention. This
expectations-only amendment changes only the baseline fatal guard and the
survivability of the known width-64 control-loss exit. All 64 configurations,
placements, payloads, exact points, behavioral bands and unsupported-step
handling below remain frozen. The original standalone TRAF-89 oracle misses
remain misses. Results cite both expectations commits.

For every completed standalone receiver phase, sort completion rows by time
and retain every k-prefix byte floor: elapsed from first start must be at
least the bytes of those k completed messages over the receiver link rate,
plus the common path propagation (2 us ideal, 4 us remote physical). This
also applies conservatively across chained ring rounds. Match the complete
identical GOAL and normalize each physical phase's first-start-to-last-
completion makespan to the ideal phase. Both prefix and phase-baseline
floors are fatal, along with the original individual physical floors.

Only the width-64 rnic-cn standalone all-to-all exit explicitly reporting
`fabric dropped control lifecycle` (HTSIM-40) is a survivable fatal. Those
two cells remain void and unscored; independently completed cells, their
exact oracles and the existing behavioral relation instances remain
interpretable. Other crashes, missing cells or failed floors still prevent
acceptance. BACK-38 rejected steps retain null timing and share fields.

## Sweep and placement

Run 32 standalone configurations: pattern in {ring, all-to-all}, participating
width in {8, 16, 32, 64}, rate in {400, 200} Gbit/s, and profile in {rnic-nn,
rnic-cn}. Use 64 GOAL endpoints on eight nodes of eight GPUs, with one network
interface per GPU. Placement starts with `declared_manifest(tp=64, nodes=8,
gpus_per_node=8)`. Select ranks in rail-major order: 0, 8, ..., 56, 1, 9, ...,
63, taking the first W or F ranks. Every ring edge crosses nodes, including the
wraparound. This order exercises all eight nodes even at width eight.

Ring payload S is 1,048,576 bytes, split evenly into W chunks, using existing
`ring_allreduce`. All-to-all uses existing `pairwise_all_to_allv`, with B =
65,536 bytes per directed remote pair, released together. Same-node pairs are
excluded from this fabric component experiment. F denotes participants and
candidate destinations including self, not the number of remote peers. Each
receiver has D = 7F/8 remote sources: 7, 14, 28, 56. There are F*D flows.
A claim of 64 remote senders in this 64-rank reference deployment would be false.

Physical profile uses `examples/m1/topologies/clos_64_400g.topo`; the 200G arm
copies it into the bulk output directory and changes both tier speeds to 200.
Both arms keep each hop at 1,000 ns and switch latency zero. Null-network
profile is topology-free, with 2,000,000 ps propagation, not a simulated Clos.
Its endpoint rates and rank identities match the physical arm. Cross-leaf
physical paths have four 1,000 ns link traversals. Keep that extra propagation
visible when interpreting physical-to-ideal inflation.

## Physical bounds and predictions before measurement

Use ps throughout: 400G is 20 ps/byte; 200G is 40 ps/byte. A flow carrying b
payload bytes has a hard floor b*ps_per_byte. The floor is printed next to every
median (p50). Propagation strengthens this to b*ps_per_byte + P. Floors do not
include all header or control work, so satisfying them alone proves little.
Ceiling: physics alone gives no finite upper bound in a queued, flow-controlled
network. Report that explicitly for physical tails. The ideal profile has a
conditional finite bound below, assuming a work-conserving reservation calendar.
No arbitrary timeout is a physical ceiling.

Ring payload per rank is exactly 2(W-1)S/W. Its payload serialization is
T_ser = 2(W-1)S*ps_per_byte/W, approaching 2S*ps_per_byte. It is not exactly flat:
from W=8 to W=64 it grows by 9/8. Latency is exactly 2(W-1)P in the ideal model.
The packetized model has 4,096 payload bytes plus 64 header bytes per full
packet, so slot q is 83,200 ps at 400G and 166,400 ps at 200G. Here n=S/(4096W)
is integral. One isolated synchronized round predicts t_round=(n+1)q+P, and a
ring predicts T_nn=2(W-1)*t_round. The extra slot is destination serialization.
This yields the following exact point predictions in ps:

| W | Payload serialization 400G | Ideal latency | NN ring 400G | NN ring 200G |
|---|---|---|---|---|
| 8 | 36,700,160 | 28,000,000 | 66,438,400 | 104,876,800 |
| 16 | 39,321,600 | 60,000,000 | 102,432,000 | 144,864,000 |
| 32 | 40,632,320 | 124,000,000 | 170,425,600 | 216,851,200 |
| 64 | 41,287,680 | 252,000,000 | 304,416,000 | 356,832,000 |

Exact-oracle checks compare these points to 0 ps. Separately, a conservative
calendar envelope allows one additional full slot per round; it is not used to
turn a failed exact oracle into a pass. The dependency-depth angle gives a
ring floor 2(W-1)*(S/W*ps_per_byte+P); the envelope ceiling is T_nn+2(W-1)q.
The standalone round is synchronized initially; later physical round starts
may spread. Such flow ratios are not interpreted as aligned-start evidence.

All-to-all has a receiver floor D*B*ps_per_byte+P. Its ideal upper envelope is
(F*D*16+1)*q+P, serializing every packet globally with no idle slots. This is
intentionally loose; a tighter directional hypothesis is that phase makespan
increases strictly with F at fixed B, rate and profile. The ideal lower floors
at 400G are 11,175,040; 20,350,080; 38,700,160; 75,400,320 ps. At 200G subtract
2,000,000, double the rest, then restore 2,000,000. The all-to-all bottleneck
is growing per-endpoint byte service, while the ring's growing term is its
serial round depth. Neither flow-percentile population is a request-tail
population sampled across independent workloads.

Rate relation: the payload and header serialization terms double exactly,
while the propagation terms remain unchanged. Do not demand a 2x total FCT.
For the ideal ring, T_200-P_total = 2*(T_400-P_total), exact to 0 ps. For the
ideal all-to-all, predict a 200G/400G makespan ratio in [1,2], allowing 2 slots
for calendar boundary effects. Physical makespans should not decrease when
capacity halves; there is no registered exact scaling for congestion control.

Physical comparator: match identical GOAL messages by source, destination and
tag, validate equal payload and multiplicity, then retain ratios only when
start_time_ps matches exactly across profiles. Use `simllm.backends.fct` for
those rows. Require aligned physical FCT >= ideal FCT only where no other
aligned flow shares the bottleneck link (the initial ring round). Shared
receiver per-flow ratios below one are diagnostic, not fatal. Require every
receiver completion-prefix byte floor and physical phase makespan >= ideal
phase makespan as fatal guards. Score the <=2x upper band as a behavioral hypothesis, and report <=1.2x
as a separate tighter target with no extra pass denominator. Unaligned flows
retain raw FCT only; compare first-start-to-last-completion phase makespans.
A 2x miss is a real result, never repaired by relaxing the band after the run.

## Supported step path and explicit reachability limits

Run 32 one-step attempts across the same axes, with one layer and two
collectives, no host launch cost, and a fixed 100,000,000 ps compute service.
This is a declared synthetic control interval, not a calibrated GPU model or a
production model's token rate. Its floor and ceiling are both 100,000,000 ps
by input. Geometry is hidden=1024, intermediate=4096, heads=16, KV heads=8,
head size=64, vocabulary=4096, dtype=2 bytes. Ring uses 512 new tokens so its
activation payload is S. Expert mode uses 64 experts, top-k=1, F*32 new tokens,
and 64/F local experts so uniform per-remote-destination payload stays B.
The existing step expert path is one engine's dispatch and reverse combine,
not the full-population standalone all-to-all. Preserve and label that distinction.

Use `HtsimStepSink` with the placement manifest. Local expert transfers use the
existing analytic local-link service and stay off the fabric. Pass the real
`StepResult` and locality outcome to `HtsimRequestMetricReducer` and
`attribute_step_detail`. The collective share is collective_ps/step_latency_ps;
the fabric share is fabric_ps/step_latency_ps. Both have floor 0 and ceiling 1.
For this one sampled prefill step, time to first token (TTFT) equals the step
latency. No time per output token (TPOT) distribution is claimed. Ring step
prediction is K+2*T_nn, with exact K and its corresponding network share.
For expert steps, the network floor is 2*(D*B*ps_per_byte+P), with K added for
the step floor; the share floor is that network floor divided by K plus itself.
Step ideal ceilings follow twice the corresponding standalone envelope plus K.

Source inspection identifies two limits before the run. First, `HtsimStepSink`
rejects multi-artifact rnic-cn execution because restarting the process loses
network state (BACK-38). Record rejected cells, keep their shares and TTFT null,
and do not disable the guard or substitute standalone times as measured steps.
Second, `CriticalPathBreakdown` in `simllm/core/runtime.py` is emitted by the
coarse device runtime, not by this packet-level sink. The supported sink reducer
partitions ordered artifacts by their realized local/fabric maximum; it does
not publish internal packet queues or `CompletionEvent` segments. Do not
construct a counterfeit `CriticalPathBreakdown` from a makespan. These limits
prevent closure of COMP-9 and of the full S3 requested attribution claim even
if every executable component relation passes. They are coverage findings,
not fatal failures of separately valid standalone component measurements.

## Evidence and reproducibility

Separate configuration counts, exact-oracle rows, behavioral relation families
and instances, and fatal structural guards. Quantiles use nearest rank:
sorted_values[ceil(p*N)-1]. No interpolation or pooled configurations. Save all
raw completion rows and aligned ratios in bulk CSV, and small summaries and
check records in tracked JSON. Include backend executable and topology hashes,
GOAL digests, manifest lines and the expectation hash. Write outputs through
`SIMLLM_DATA_ROOT` or `--out`; require an external bulk directory.

Fatal guards are physical quiescence, exact flow identities/counts/bytes,
nonnegative consistent timestamps, no local flow on fabric, physical floors,
no unshared aligned ratio below one, receiver completion-prefix byte floors,
phase-baseline floors, and conservation of every supported step's reducer
partition and TTFT. Any non-survivable violation makes the entire study void
for closure, retains all evidence, and leaves COMP-9 open. Never mix fatal
guards into a behavioral score. Only the specific HTSIM-40 exit declared in
the amendment is survivable, retaining void cells separately. Other backend
crashes/timeouts record incomplete cells and prevent acceptance, with their
diagnostics retained.

An independent external check uses NVIDIA's published all-reduce bandwidth
accounting: per-rank bus bandwidth is S/T times 2(W-1)/W and cannot exceed
endpoint bandwidth. This checks units and physical interpretation rather than
repeating the simulator oracle. Source: [NCCL tests performance documentation](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md).
It validates the transfer factor, not a measured deployment latency. No held-out
hardware tail or workload calibration is supplied here; that part of COMP-9
remains open. Tests are deterministic and use synthetic completions, no GPU,
network access or native binaries. Required gates: ruff and full pytest.
