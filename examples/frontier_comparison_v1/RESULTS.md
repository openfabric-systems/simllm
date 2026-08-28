# Frontier comparison result

## Verdict

**PASS, non-void.** The frozen comparison behaves in the expected direction
without fitting to the external data. The exact candidate and matched-point
families pass, all three estimator frontiers plus the external frontier are
monotone, the efficiency-1.0 frontier dominates all 10 external rows at their
required per-user speeds, and 9 of 10 rows fall inside the frozen 0.6 to 1.0
matched-configuration throughput bracket. Row 10 is the published exception:
its **157.234 tokens/s/GPU** is below our efficiency-0.6 estimate of
**167.408 tokens/s/GPU**.

The result is non-void because all five fatal guards held. Evidence classes
remain separate and are not added into one score.

| Family | Tally | Acceptance | Verdict |
|---|---:|---:|---|
| X1 | 3 / 3 | all | PASS |
| X2 | 4 / 4 | all | PASS |
| X3a | 4 / 4 | all | PASS |
| X3b | 10 / 10 | all | PASS |
| X3c | 9 / 10 | >= 8 | PASS |
| W | 1 / 1 | <= 120 s | PASS |

## Qwen3-32B FP8 extraction column

The config-only extraction binds `Qwen/Qwen3-32B-FP8` to exact Hugging Face
revision `aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`. The vLLM 0.27.1 inventory is
`c8832ba8ba21e49517b6b74e89554c2abdb0d9e76530f647a7849f3f8448ec56` and the companion SGLang
inventory is `51740b52625002a964e75fddb679e9f8394a08a7d7c62556d2535c3bc60515e3`. Both contain
15 cases, 5 logical families
and 257 visits per case. Repeated runs are
byte identical, and their framework-neutral content agrees exactly after
source provenance is removed.

FG-2 confirms the fatal architecture literals exactly: 64 layers, hidden size
5120, intermediate size 25600, 64 attention heads, 8 key-value heads, head
dimension 128 and vocabulary size 151936. The FP8 checkpoint has
32,762,123,264 logical parameters, so TP4 owns
8,190,530,816 weight bytes per rank at one byte per parameter.
That is 5.8 percent of the declared 141 GB H200 capacity.

## Work derivation and matched-point pricing

The pricing record derives its work from the inventory's per-layer
projections. At the frozen TP4 mapping, decode carries
72,875,612,160 FLOPs and
1,114,112,000 logical key-value
cache bytes per batch item at the average 4,250-token context. A 3,500-token
uncached prefill carries 221,652,172,144,640
FLOPs and 917,504,000 logical
key-value bytes per request. Matrix weights use FP8 at one byte per parameter;
logical weight and key-value bytes shard by tensor-parallel width. TP2 and TP8
FLOPs scale relative to the frozen TP4 comparison mapping.

At efficiency 1.0, the exact external-best topology prices decode batch 64 to
**5.379516 ms**, below the external
9.179 ms, and prices the uncached prefill request to
**112.002108 ms**, below the external
196.423 ms. The implied efficiencies are **0.586068 for decode** and
**0.570209 for prefill**. Both sit inside the frozen [0.40, 1.00] band.
They are reported comparison results only and are never installed as model
parameters.

## Frontier overlay

[PDF](figures/frontier-comparison.pdf) and
[PNG](figures/frontier-comparison.png) show the three SimLLM ESTIMATE
frontiers and all 10 MEASURED-EXTERNAL rows on logarithmic per-user speed and
per-GPU throughput axes. The upper-right corner is better. External row labels
match the table below.

X3b compares the efficiency-1.0 service-feasible frontier at or above each
external per-user speed. X3c compares each external row's throughput with the
0.6 and 1.0 estimates for that row's exact prefill/decode topology and batch,
including comparison points that the 10 ms frontier filter excludes. This
keeps the external database out of pricing while making every row-level
predicate explicit.

| Row | External user tok/s | e=0.6 tok/s/GPU | External tok/s/GPU | e=1.0 tok/s/GPU | X3b | X3c |
|---:|---:|---:|---:|---:|---|---|
| 1 | 56.032 | 502.223 | 773.212 | 837.038 | PASS | PASS |
| 2 | 59.324 | 502.223 | 765.613 | 837.038 | PASS | PASS |
| 3 | 84.006 | 418.519 | 644.344 | 697.532 | PASS | PASS |
| 4 | 108.944 | 418.519 | 602.586 | 697.532 | PASS | PASS |
| 5 | 111.906 | 418.519 | 541.607 | 697.532 | PASS | PASS |
| 6 | 119.638 | 334.815 | 441.140 | 558.025 | PASS | PASS |
| 7 | 127.064 | 251.111 | 386.606 | 418.519 | PASS | PASS |
| 8 | 146.002 | 223.210 | 343.650 | 372.017 | PASS | PASS |
| 9 | 168.131 | 223.210 | 258.328 | 372.017 | PASS | PASS |
| 10 | 202.168 | 167.408 | 157.234 | 279.013 | PASS | MISS, below e=0.6 |

The X4 scope comes from the
[frontier ladder study](../frontier_ladder_v1/RESULTS.md). Its ideal-network
class tracks the packet rung within about 1.6 percent on contention-free
point-to-point legs and is about 8x optimistic at the frozen eight-into-one
fan-in cell. This workload uses intra-node tensor parallel and one
prefill-to-decode transfer, so its declared regime is contention-free. No
packet run executes here, and the 1.6 percent statement is a regime-scoped
mechanism result rather than an absolute-accuracy claim.

## Honesty and version drift

The external rows interpolate a measured per-operation database for real H200
silicon. Our rows use a declared roofline and declared envelopes until the
calibration campaigns close. **On absolute kernel throughput, their side is
better calibrated today.** The defensible precision claims are exactly the X4
network-mechanism envelope, the evidence-class label on every number and the
exact accounting gates. Nothing broader is claimed.

The local run used aiconfigurator 0.11.0 and TensorRT-LLM database
h200_sxm 1.3.0rc10. Its best disaggregated row is 602.586 tokens/s/GPU at
108.944 tokens/s/user, with five TP4 prefill workers and three TP4 decode
workers at batch 64. The published README snapshot reports 684.79 at 100.31
with four replicas of a different TP2/TP4 topology and decode batch 68. This is
external-tool version drift. Neither anchor is preferred, and neither was used
to fit any SimLLM parameter.

## Physical sanity

Before reading the decode estimate, 25,821,675,520 bytes over 4.8
TB/s set a memory floor of 5.379516 ms. The TP4 compute
floor is lower, so the observed 5.379516 ms estimate sits
exactly on the physical memory floor. At the frozen lower plausibility edge of
efficiency 0.4, the same term is 13.448789 ms, which brackets the external
9.179 ms rather than making it physically impossible.

Before reading the prefill estimate,
221,652,172,144,640 FLOPs over 1.979 PFLOP/s set
a compute floor of 112.002108 ms. At efficiency 0.4 the
same work takes 280.005271 ms, so the external 196.423 ms sits between the
declared ideal and low-efficiency bounds. This is an independent compute
check, not another pass over the memory arithmetic.

At system level, the external best point and our matched point both fit 32
GPUs and the same five-plus-three TP4 pool structure. Our ideal decode and
prefill services are 58.6 and 57.0 percent of their measured values. Those
ratios have the required optimistic sign and are of the same order, but their
roughly 1.7x residual is exactly why DEPLOY-11 owns silicon calibration.

## Fatal guards

| Guard | Outcome | Predicate |
|---|---|---|
| FG-1 | PASS | tracked external rows match external.sha256 |
| FG-2 | PASS | inventory geometry matches all seven frozen literals |
| FG-3 | PASS | external rows are display and comparison inputs only |
| FG-4 | PASS | pricing-lane process interception count is zero |
| FG-5 | PASS | 83bb281 is an ancestor and the frozen bytes match |

The pricing lane called only `simllm.deploy` and triggered zero process
interceptions. The external program did not run. All three efficiency arms
scanned 5,070 candidates each in one process
and completed in 19.024 seconds, below the frozen 120 s
limit. The external tool's observed 11 s search remains unscored context.

## Project consequence

What ran: `examples/frontier_comparison_v1`, a config-only extraction-backed,
binary-free comparison of the SimLLM deployment estimator against the 10
frozen aiconfigurator 0.11.0 disaggregated H200 rows.

What came out: the run is non-void and passes every family acceptance bar;
the deciding overlay result is X3c at 9 of 10 rows, with row 10 below the
declared 0.6 arm by 10.174 tokens/s/GPU. Decode and prefill implied
efficiencies are 0.586068 and 0.570209, and remain report-only.

What it changes for the project: the maintained frontier comparison is now a
literal binary-free study, while DEPLOY-9 registers wider candidate
enumeration, DEPLOY-10 registers additional external systems and DEPLOY-11
registers H200 silicon calibration. Those tasks name the remaining
completeness and precision work rather than letting the declared roofline read
as a calibrated model.

What it does not change: no existing calibration task closes, COMP-54 remains
open for Kimi K3, TRAF-20 remains open on its separate speed qualification, no
packet-level validity claim expands, no H200 efficiency is installed and
neither tool is validated against a live serving deployment.
