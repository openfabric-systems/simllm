# CORE-61 depth-8 retry expectations

## What is frozen

This supplement is a pre-scoring harness amendment. No depth-8 decode boundary
had produced a measured service when it was committed. The original
3,751,359,511 ps held-out prediction, signed `measured minus predicted`
residual and 5 percent acceptance rule are inherited unchanged. They were never
at risk of outcome-driven adjustment because no measured depth-8 number
existed.

The measured contract remains DeepSeek-V3 at revision
`e815299b0bcbac849fa540c768ef21845365c9eb`, reduced to eight layers, in
CUDA-graph mode and the TP1/DP1/EP1 physical envelope. The selected step has 32
requests and every request has exactly 2,000 cached KV tokens. Service remains
the additive noncollective GPU kernel duration inside exactly one
`execute_context_0(0)_generation_32(32)` boundary, selected through Nsight
Systems runtime correlations.

## Exact retained failure diagnosis

The original decode command passed `MAX_NUM_BATCHED_TOKENS=65536`. vLLM used
that scheduler maximum as the token count of its dummy startup profile.
`REDUCED_LAYERS=8` only reduced model depth, so it did not reduce the dummy
token dimension.

The two pinned attempt manifests verify different final allocation sites:

- Job `200123`, manifest SHA-256 `b7d23c47...`, fails in the modular mixture of
  experts output allocation at `torch.empty_like(hidden_states)`. The request
  is exactly 939,524,096 bytes, or 896 MiB:
  `65,536 x 7,168 x 2` bytes for a BF16 hidden-state tensor.
- Warm-cache job `200128`, manifest SHA-256 `3345fc82...`, reaches the compiled
  FlashInfer DeepGEMM path and fails allocating its output. The generated
  shape is `65,536 x 24,576` BF16 values, exactly 3,221,225,472 bytes, or
  3 GiB.

The merged COMP-78 summary described both attempts as the same 896 MiB
allocation. The pinned logs are more specific: both are startup-only out-of-
memory failures caused by the 65,536-token dummy run, but the warm-cache final
allocation is 3 GiB rather than 896 MiB. The historical result remains
unchanged as chronology; this supplement carries the correction.

## Harness amendment

The retry starts vLLM with a 4,096-token scheduler and dummy-profile cap. That
cap alone would split equal 2,000-token prompts and would not preserve batch
32. The amended harness therefore uses calibrated staggered prompts, keeps the
earliest requests alive while later requests finish prefill, and stops only at
the first full-batch decode state whose scheduler record proves all 32 cached
token counts equal 2,000.

This changes startup and request-construction scaffolding. It does not change
the selected decode step, model depth, batch, KV lengths, framework, revision,
parallelism, launch mode, service definition, prediction or comparison rule.
Any run without the exact full-batch scheduler marker and exact NVTX boundary
is void and remains unscored.

## Registered execution

The base submission remains the cache-warming eight-layer base capture at a
16,384-token startup cap:

```text
ssh merlin sbatch -M gmerlin7 --partition=gh-hourly --time=00:25:00 --job-name=gh-core61r-d8-base --export=ALL,MODEL=deepseek-ai/DeepSeek-V3,MODEL_KEY=deepseek-v3,SHAPE_SET=deepseek,REVISION=e815299b0bcbac849fa540c768ef21845365c9eb,REDUCED_LAYERS=8,GPU_MEMORY_UTILIZATION=0.88,MODE=graph,DEEPSEEK_SUITE=base,MAX_MODEL_LEN=8192,MAX_NUM_BATCHED_TOKENS=16384,RUN_WALL=0 $SIMLLM_MERLIN_STAGE_ROOT/gh200lane/run_vllm_capture.sbatch
```

The amended decode submission is:

```text
ssh merlin sbatch -M gmerlin7 --partition=gh-hourly --time=00:20:00 --job-name=gh-core61r-d8-decode --export=ALL,MODEL=deepseek-ai/DeepSeek-V3,REVISION=e815299b0bcbac849fa540c768ef21845365c9eb,REDUCED_LAYERS=8,GPU_MEMORY_UTILIZATION=0.88,MAX_MODEL_LEN=8192,STARTUP_MAX_NUM_BATCHED_TOKENS=4096,MAX_NUM_SEQS=32,BATCH_SIZE=32,REMOTE_KV_TOKENS=2000 $SIMLLM_CORE61_RETRY_STAGE_ROOT/run_core61_depth_retry.sbatch
```

Inspect `gh-hourly` before each submission, keep at most one task-owned job
live, wait at least 60 seconds between state checks, and submit decode only
after base is terminal and retained. Never overwrite an attempt. On SSH or MFA
loss, retain the last job identifier and state and park resumably.

## Scoring and interpretation

The signed residual is

```text
R = measured_service_ps - 3,751,359,511 ps
```

and the frozen relative error is `abs(R) / measured_service_ps`. A value at or
below 5 percent validates linear depth scaling for this decode family. In that
case the larger decode-family gap does not live in depth scaling; it lives in
expert-parallel residency shape or decode-side overlap. A miss publishes the
signed direction and magnitude of the non-linearity. CORE-63 is registered only
if that measured miss leaves residual work.

TRAF-66's finite compute and communication overlap remains a separate ledger
term and is not recomputed. COMP-76 and every scored artifact remain untouched.
