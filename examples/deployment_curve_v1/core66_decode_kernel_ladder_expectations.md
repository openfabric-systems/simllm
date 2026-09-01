# CORE-66 decode kernel ladder expectations

Status: **EXPECTATIONS ONLY**. This freeze precedes the independent published
confirmation. It contains the composition selected in the unpublished scratch
layer and no confirmatory measurement.

## Runtime and scope

The ladder uses vLLM 0.27.1 on the validated CUDA 12.9, Python 3.11 aarch64,
Torch cu129 GH200 lane. vLLM's native operations are the physical kernel
identities of this record. This is the maintainer-directed substitution under
CORE-66. It does not bind those operations to SGLang or price a SGLang DeepEP
path; that physical binding remains a declared limitation.

The cell is one GPU, tensor parallel one, pipeline parallel one and expert
parallel one. Network and cross-rank contention are absent. Weights are dummy
and must remain absent from the model cache. The exact decode boundary is batch
32 with 2,000 prior KV tokens per request.

## Confirmatory graph

The captured root forward graph contains three dense layers followed by one
mixture-of-experts (MoE) layer. The frozen one-stream composition is

`3 * 308.256 us + 1 * 415.648 us + 3.456 us = 1,343.872 us`.

Independent-stream contention is diagnostic only and has coefficient one in
this graph because the retained framework path uses one stream. The
confirmation measures N=50 and N=20 over seven trials after 20 warmup replays
and a 256 MiB cache flush before each trial. Raw service is the device-event
interval divided by N. Subtracted service first removes the independently
measured empty event interval, then divides by N. The composition authority is
the sum of every correlation-complete native CUDA kernel duration in one graph
replay.

The native service must remain between the 0.524 ms optimistic HBM floor and
the 20 ms layer-ceiling bound. N=50 and N=20 must differ by no more than five
percent. Measured minus predicted must have absolute magnitude no larger than
67.1936 us. The residual may have either sign.

Process success, an empty and unchanged weight snapshot, the exact scheduler
marker, three dense plus one MoE layer, a 32 by 7,168 BF16 output, one native
CUDA stream, a stable 100-node graph and every dense, MoE, MLA and embedding
identity are fatal guards. A violation voids the run rather than lowering a
score. Run configuration, preservation and fatal structural guards remain
separate from the three behavioral relations.

## Per-layer compute model

The layer model preserves the full measured services while exposing the
common and type-specific terms needed by the CORE-65 multiplier structure:

- common service `C = 173.184 us` per layer;
- dense-specific service `D = 135.072 us`, so `C + D = 308.256 us`;
- MoE non-routed specific service `N = 43.264 us`;
- routed grouped service `G = 199.200 us` at 256 resident experts and 256
  assignments, so `C + N + G = 415.648 us`;
- fixed service `F = 560.192 us` once, comprising the retained graph-fixed and
  non-LM step/output work plus the rung-0 measured LM head.

For resident experts `R`, assignments `A` and an unresolved routed-service
split `alpha` from zero to one, the frozen routed function is

`G(R,A,alpha) = 199.200 us * (alpha * R/256 + (1-alpha) * A/256)`.

The endpoints keep the two physical interpretations honest. `alpha = 0`
prices all grouped service by assignments. `alpha = 1` prices it by resident
experts. No point inside the interval is selected or fitted.

The standard-decode compute equation is

`S_compute = 61*C + 3*D + 58*(N + G(4,256/9,alpha)) + F`.

This is the CORE-65 multiplier structure written from measured layer types:
common `61/4` over the measured four-layer common basis, dense `1` over the
three-layer dense basis, MoE `58` over the one-layer MoE basis, and graph/step
and output terms once.

## Signed-direction freeze

With expert-parallel communication set to the literal measured value of zero,
the assignment endpoint predicts 16,707.262995 tokens/s/node and the resident
endpoint predicts 18,003.485222. Both are below the 22,282 calibration anchor
and both move upward from the inherited 9,544.657796 prediction, by
+7,162.605199 to +8,458.827426 tokens/s/node.

Real expert parallelism adds an unmeasured nonnegative communication service
`E_ep`. This study does not price it because the single-rank vLLM cell contains
no dispatch or combine traffic. Adding `E_ep` always moves throughput downward.
It can reduce, erase or reverse the compute-only upward movement, so the study
publishes no single deployed EP prediction and does not call the compute-only
interval a SGLang result.

## Preservation and access

The seven-file preservation manifest protects the inherited CORE-65 inputs and
the CORE-66 protocol and remainder records. Automated tests may read their
allowlisted calibration fields. A held-out simulated-MTP value may not enter
arithmetic, comparison, fitting or reproduction. The standard-decode anchor is
the sole calibration comparison.
