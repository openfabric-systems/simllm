# CORE-66 decode kernel ladder result

Status: **PASS**.

What ran: a four-rung, single-GH200 vLLM decode study at batch 32 and KV
length 2,000, followed by one expectations-frozen four-layer confirmation.
All cells used dummy weights, one GPU, one rank and no network.

What came out: the published four-layer graph measured 1,290.176 us of cold
native kernel service against the frozen 1,343.872 us composition. Measured
minus predicted is -53.696 us, or -3.996 percent, inside the frozen
+/-67.1936 us band. All eight fatal guards, all three behavioral relations and
all seven preservation locks pass.

What it changes: the compute half of the standard-decode model now uses
measured dense and mixture-of-experts (MoE) layer services instead of a bucket
inference. The calibration-only compute prediction moves upward by
+7,162.605199 to +8,458.827426 tokens/s/node from the inherited prediction.
The decode kernel ladder is complete under CORE-66.

What it does not change: CORE-66 remains open on the broader exact SGLang EP72
physical capture. This single-rank study contains no expert-parallel dispatch
or combine traffic, so the nonnegative communication term is absent and
unpriced. Adding that term moves throughput downward and can reduce, erase or
reverse the compute-only upward movement. No single deployed expert-parallel
prediction is published, and no SGLang identity claim is made.

## Rung 0: individual kernels

Each trial first writes a separate 256 MiB buffer to evict cache. The first
correlation-complete CUDA kernel sum is the cold service and composition
authority; later calls give the resident diagnostic. The raw time is the
median device-event interval divided by N. The subtracted time removes the
median empty device-event interval before division by N. Raw and subtracted
times include gaps while the host launches native work, so they are not used
as GPU service.

The speed-of-light (SOL) comparison uses an optimistic 4 TB/s HBM roof, 989
TFLOP/s for FP8 and 494 TFLOP/s for BF16. Its lower bound is the larger of
minimum bytes divided by HBM bandwidth and operations divided by the applicable
math peak. SOL is that lower-bound time divided by cold measured service.

| Kernel family | Raw us | Subtracted us | Cold us | Resident us | TB/s | TFLOP/s | SOL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q/KV compression | 169.580 | 169.432 | 13.600 | 11.104 | 1.157 | 71.241 | 28.9% |
| Q decompression | 167.196 | 167.043 | 21.984 | 12.160 | 1.793 | 109.894 | 44.8% |
| KV write | 48.238 | 48.084 | 2.432 | 1.536 | 0.030 | 0 | 0.8% |
| MLA attention core and KV read | 274.314 | 274.162 | 81.152 | 79.360 | 1.322 | 206.738 | 41.8% |
| Attention output | 161.797 | 161.647 | 48.576 | 41.280 | 2.449 | 154.731 | 61.2% |
| Dense gate/up | 161.581 | 161.430 | 89.600 | 80.384 | 2.981 | 188.744 | 74.5% |
| Dense activation | 49.307 | 49.155 | 5.792 | 4.416 | 0.611 | n/a | 15.3% |
| Dense down | 158.591 | 158.440 | 53.184 | 45.696 | 2.515 | 158.990 | 62.9% |
| Router | 80.545 | 80.399 | 7.488 | 5.792 | 0.554 | 15.684 | 13.8% |
| Shared expert | 344.694 | 344.545 | 31.264 | 22.944 | 1.436 | 90.154 | 35.9% |
| Routed expert, one assignment | 456.065 | 455.914 | 45.440 | 32.576 | 0.970 | 1.938 | 24.3% |
| LM head | 529.389 | 529.233 | 533.824 | 525.632 | 3.488 | 111.099 | 87.2% |

Every cold service clears its independently written roofline floor and loose
physical ceiling. The LM head is closest to saturating HBM. The large dense
and attention projections are bandwidth-heavy. KV write, router and activation
are dominated by short work and launch structure. The MLA core presses both
BF16 math and compressed-KV movement.

The Q/KV row comes from the reversed-order job `202260`, which replaced a
discarded 7.3 percent repeat-order separation with a 2.41 percent separation.
The other rows come from job `202255`. The isolated eager activation identity
differs from the full graph's fused Triton identity; rung 1 carries that
replacement as fusion structure.

## Rung 1: framework-fused layers

The chip executes each full layer in framework order. Although the trace uses
one CUDA stream, Hopper's programmatic dependent launch lets the attention
main and combine kernels overlap by about 5.6 us. Full layers also add norm,
quantization, rotary and routing work that a one-call-per-family sum omits.

| Layer | Rung-0 cold sum us | Full cold us | Native fusion delta us | Raw full us | Subtracted full us |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 316.320 | 308.256 | -8.064 (-2.55%) | 1,495.288 | 1,495.124 |
| MoE | 251.936 | 415.648 | +163.712 | 2,588.876 | 2,588.723 |

For the dense layer, dependent launch, fused activation and cache state remove
8.064 us of native work relative to isolated addition. Its raw and subtracted
launch-structure deltas are instead +304.684 and +305.731 us because the full
layer includes additional launches and their host gaps.

For the MoE layer, 216.448 us is non-routed work and 199.200 us is the grouped
routed path. The latter executes 256 assignment positions through two grouped
GEMMs plus quantization, alignment, activation and sum. It is not 256 serial
copies of the rung-0 45.440 us one-assignment operation. The MoE raw and
subtracted launch-structure deltas are +886.447 and +887.497 us. Job `202277`
provides both layer records on one stream.

## Rung 2: independent-stream contention

Each factor is per-copy native kernel service at width two or four divided by
the same cell's width-one resident service. These are a diagnostic
counterfactual: the retained framework graph uses one stream, so no factor is
multiplied into rung 3.

| Kernel family | Width-1 resident us | Width 2 | Width 4 | Timing inference |
| --- | ---: | ---: | ---: | --- |
| Q/KV compression | 12.064 | 0.939 | 0.915 | spare occupancy or cache reuse |
| Q decompression | 13.120 | 1.005 | 0.966 | spare occupancy or cache reuse |
| KV write | 1.504 | 1.000 | 1.000 | short memory work |
| MLA attention core | 76.096 | 1.686 to 1.715 | 1.893 to 1.914 | streaming multiprocessor and compressed-KV pressure |
| Attention output | 42.048 | 1.468 | 1.799 | HBM-heavy matrix work |
| Dense gate/up | 81.600 | 1.414 | 1.494 | HBM and tensor work |
| Dense activation | 5.408 | 0.888 | 0.828 | short work and spare occupancy |
| Dense down | 45.376 | 1.493 | 1.888 | HBM-heavy matrix work |
| Router | 6.144 | 0.979 | 0.958 | short matrix work |
| Shared expert | 23.008 | 0.999 | 0.974 | cache-resident matrix work |
| Routed expert, one assignment | 34.080 | 1.007 | 1.005 | cache-resident matrix work |
| LM head | 519.232 | 2.004 | 2.033 | HBM saturated at width two |

Job `202300` proves exact one-, two- and four-stream placement and every shape
guard. The hardware counter probe was denied with `ERR_NVGPUCTRPERM`, so the
resource column is timing inference rather than a direct HBM, L2 or streaming
multiprocessor counter result. The ordering still matches the independent
rung-0 roofline classes: the 87.2 percent-SOL LM head takes the full two-copy
penalty; large dense and attention work slows; short and cache-resident work
stays near one.

The stronger claim that concurrent MLA has one repeat-independent absolute
service is refuted. The targeted job `202420` still separated N=20 and N=50 by
more than five percent. Its sustained N=50 ratios independently reproduce job
`202300` within 1.7 percent at width two and 1.1 percent at width four, so the
published MLA result is a ratio band rather than a selected endpoint.

## Rung 3: mega-kernel composition

| Graph | Prediction us | Cold measured us | Measured minus predicted | Repeat separation | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| Four-layer scratch basis, 3 dense + 1 MoE | 1,343.872 | 1,285.056 | -58.816 us (-4.377%) | 0.224% | kept |
| Five-layer all-MoE diagnostic | 1,894.208 | 1,909.279 | +15.071 us (+0.796%) | 0.102% | kept |
| Four-layer published confirmation | 1,343.872 | 1,290.176 | -53.696 us (-3.996%) | 0.040% | PASS |

The four-layer prediction is

`3 * 308.256 us + 1 * 415.648 us + 3.456 us = 1,343.872 us`.

The independent-stream contention coefficient is exactly one. The scratch
basis in job `202431` fit without an outcome-selected correction, so that
smallest composition was frozen in commit `7919f7b`. Job `202449` then ran the
independent confirmation from implementation commit `ae158a2` and measured a
100-node, one-stream graph with three dense layers, one MoE layer and a 32 by
7,168 BF16 output.

For the published N=50 authority, raw service is 1,306.654 us and service after
the 3.264 us empty-event subtraction is 1,306.589 us. The cold
correlation-complete native sum is 1,290.176 us; the resident sum is 1,275.743
us. The cold result is 2.46 times the 0.524 ms optimistic HBM floor and 6.45
percent of the 20 ms ceiling. The all-MoE job `202446` independently shows that
root-separated in-graph MoE service does not acquire a depth-growing residual.

## Established per-layer compute model

The measured layer identities are:

- common service `C = 173.184 us` per layer;
- dense-specific service `D = 135.072 us`, so `C + D = 308.256 us`;
- MoE non-routed specific service `N = 43.264 us`;
- grouped routed service `G = 199.200 us` at 256 resident experts and 256
  assignments, so `C + N + G = 415.648 us`;
- fixed service `F = 560.192 us` once, comprising 3.456 us graph-fixed,
  22.912 us non-LM step/output and the rung-0 measured 533.824 us LM head.

For resident experts `R`, assignments `A` and unresolved split `alpha` from
zero to one:

`G(R,A,alpha) = 199.200 us * (alpha * R/256 + (1-alpha) * A/256)`.

No interior `alpha` is fitted. The assignment endpoint is `alpha = 0`; the
resident-expert endpoint is `alpha = 1`. The standard-decode compute model is

`S_compute = 61*C + 3*D + 58*(N + G(4,256/9,alpha)) + F`.

This reconstructs the CORE-65 multiplier structure from measured layer types:
common work 61 times, dense-specific work three times, MoE-specific work 58
times, and fixed step/output work once. Every term carries its rung provenance
in the machine-readable result.

## Calibration-only signed movement

With the absent communication term held at its literally measured single-rank
value of zero, the assignment endpoint gives 15,322,677,333.333 ps per step
and 16,707.262995 tokens/s/node. The resident endpoint gives 14,219,469,000 ps
and 18,003.485222 tokens/s/node. Relative to the inherited 9,544.657796
prediction, the compute-only movement is therefore +7,162.605199 to
+8,458.827426 tokens/s/node. Both endpoints remain below the 22,282 calibration
anchor, by -25.019015 and -19.201664 percent respectively.

Expert-parallel communication remains `E_ep >= 0`, absent and unpriced:

`throughput = 256 * 10^12 / (S_compute + E_ep)`.

The derivative with respect to `E_ep` is negative. The compute correction moves
upward, while any real dispatch/combine term moves downward. Because this cell
does not identify `E_ep`, it cannot say which direction wins in a deployed
EP72 system.

## Evidence and limits

The published run keeps evidence classes separate. The three behavioral
relations pass: residual, repeat stability and physical bounds. The eight
fatal structural guards pass: process, weights, scheduler shape, layer order,
output shape, one native stream, 100 nodes and native identities. Seven
preservation locks pass. No combined score is formed from those classes.

Two scratch-layer incidental exposures are disclosed. One broad historical
search exposed a held-out numeric field before the field boundary was noticed;
one later search exposed only its split label and record name. No held-out
value entered arithmetic, comparison, fitting or published reproduction. The
standard-decode 22,282 value is the allowed calibration anchor.

The scratch ladder used 3.085 GPU-hours. The published cell used 0.121
GPU-hours, for 3.206 GPU-hours total. Full profiler databases and logs remain
in campaign storage; this repository carries the compact result, exact
protocol, preservation manifest and reproducible harness.
