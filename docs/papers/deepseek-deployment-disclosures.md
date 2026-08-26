# DeepSeek-V3 public deployment disclosures

The evidence anchors for the deployment-curve reproduction flagship
(CORE-54): two independent public disclosures of DeepSeek-V3 serving
performance at disaggregated scale. This file records the published
numbers verbatim with their sources; the flagship study freezes its
acceptance bands against these values and never edits them after a run.

## Disclosure 1: SGLang on 96 H100 GPUs (the reproduction target)

Source: LMSYS Org blog, "Deploying DeepSeek with PD Disaggregation and
Large-Scale Expert Parallelism on 96 H100 GPUs", 2025-05-05
(https://www.lmsys.org/blog/2025-05-05-large-scale-ep/), with
open-source configurations in the sglang repository.

Cluster: 12 nodes of 8 H100 GPUs, InfiniBand interconnect, Atlas Cloud.

Prefill benchmark (4 nodes, expert parallelism 32, data-parallel
attention, dense feed-forward layers data-parallel, 16,384 tokens per
device):

| Input length | Throughput per node (tokens/s) | Versus TP16 baseline |
|---|---:|---:|
| 1K | 57,674 | 3.3x |
| 2K | 54,543 | 3.3x |
| 4K | 50,302 | 3.3x |

Decode benchmark (9 nodes, expert parallelism 72, data-parallel
attention):

| Scenario | Batch per node | KV length | Throughput per node (tokens/s) |
|---|---:|---:|---:|
| Standard | 256 | 2,000 | 22,282 |
| Simulated MTP | 128 | 4,000 | 17,373 |

Headline: 52.3k input and 22.3k output tokens per second per node at
2,000-token inputs; time to first token 2 to 5 seconds; inter-token
latency around 100 ms; output cost estimated at 0.20 dollars per
million tokens. Stated deltas against the official DeepSeek profile:
prefill within 5.6 percent (50,302 versus about 62,713 normalized) and
simulated-MTP decode 6.6 percent below (17,373 versus about 18,598) at
half DeepSeek's decode node count. Stated caveats: MTP integration
incomplete for data-parallel attention, in-distribution expert balance
data, Hopper-only kernels, 288 total experts (256 plus 32 redundant)
under the expert load balancer.

## Disclosure 2: DeepSeek's own production system (the second legend)

Source: DeepSeek open-infra-index, Open Source Week day 6,
"DeepSeek-V3/R1 Inference System Overview", 2025-03-01
(https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md).

Production hardware: H800 nodes of 8 GPUs. Prefill deployment unit: 4
nodes, routed-expert EP32, MLA and shared-expert DP32, 32 redundant
routed experts, 9 routed plus 1 shared expert per GPU. Decode
deployment unit: 18 nodes, EP144. Disclosed averages across all
production traffic: about 73.7k input tokens per second per node
(including cache hits) in prefill and about 14.8k output tokens per
second per node in decode.

## How the flagship consumes these numbers

- The reproduction axis set: aggregated output throughput against
  per-token delay of a request, oriented so the upper-right corner is
  optimal (throughput rightward, inverse delay upward). Each deployment
  configuration traces a curve as offered load sweeps; the published
  points above are anchors on those curves.
- The acceptance bar: the simulated SGLang-configuration curve passes
  within 5 percent of every anchored published point, with propagated
  error bars from the calibrated component uncertainties.
- The second legend: DeepSeek's own H800 production profile, plus
  declared what-if configurations (including the 16-prefill plus
  40-decode target), each as its own curve.

## Calibration constants policy (maintainer, 2026-08-25)

The declared constants (intra-node collective cost, PCIe work-queue
submission, KV-handoff terms) may be tuned per configuration to match
the disclosures, under three binding rules. Moderation: every tuned
value stays inside a physically justified envelope stated next to it
(measured floors and ceilings from the calibration campaign, link
arithmetic, or published hardware limits), and a value that cannot be
justified by later mechanistic modeling is out of bounds. Honesty: tuned
constants are disclosed as fitted parameters with their envelopes, never
presented as measurements. Non-circularity: constants are fitted on a
declared calibration subset of the published anchors and the 5 percent
bar is scored on the held-out remainder, so the claim cannot be
manufactured by the fit.

## Per-anchor benchmark-bias attenuation (maintainer, 2026-08-26)

The published anchors are benchmark numbers, and the disclosure itself
states the conditions that make them favorable: in-distribution expert
balance, fixed exact-length synthetic inputs packed to 16,384 tokens per
device, and incomplete MTP integration. Where a published point is biased
by such a benchmark simplification relative to the physics simLLM models,
a per-anchor attenuation factor may be applied to the comparison, under
five binding rules that extend, and never relax, the constants policy
above.

1. Named mechanism: each factor names the specific benchmark
   simplification it corrects (for example, balanced expert routing
   versus the uniform-routing dispatch incidence simLLM derives, or
   perfect length packing versus a realistic length distribution), and
   the correction applies only to the anchors that simplification
   touches.
2. Independent quantification: the factor's magnitude is derived from
   mechanism arithmetic or published evidence that does not include the
   anchor's own numeric value. A factor whose only support is that it
   makes the line fit is out of bounds; with one free factor per point,
   agreement is vacuous, so admissible factors must be fewer than the
   anchors they touch or derived with zero anchor input.
3. Freeze first: attenuation factors and their derivations are frozen
   in the expectations-only commit before any scored comparison, like
   every other constant.
4. Dual publication: the unattenuated prediction is always published
   next to the attenuated one, the figure states that the scored claim
   is under the declared benchmark-bias model, and each factor's
   uncertainty propagates into the error bars.
5. Scope: attenuation addresses benchmark-condition bias in the
   published point, never gaps in simLLM's own mechanism. A miss whose
   dominant contributor is a registered modeling residual (for example
   the decode depth-extrapolation gap) is resolved by that registered
   work, not by attenuating the anchor.
