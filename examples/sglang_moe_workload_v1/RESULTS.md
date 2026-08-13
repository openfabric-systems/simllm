# SGLang MoE workload v1 results

The frozen study is **void with findings**. All 10 scored geometry, workload,
and external timing-reducer rows passed, but one frozen fatal guard failed:
`TraceLengths` deliberately cycles a short trace, while the expectation file
said every short trace would be rejected. A fatal failure voids the behavioral
score, so 10/10 is retained for diagnosis and is not reported as a pass rate.

No live SGLang process or GPU was run. The retained study rows are diagnostic,
not qualifying evidence. Separate import-free tests cover MoE geometry,
deterministic request plans, native payloads, streaming validation, open-loop
transport, and the external timing reducer. Neither source is evidence that the
SGLang simulated worker produces request-level TTFT or TPOT.

## Chronology and command

Original expectations-only commit `c48e785` preceded the implementation and
the first run. Its content remains unchanged and that commit remains an
ancestor of this result. Later integration merged current `main` without
rewriting that original chronology.

The first working-tree harness execution printed `PASS`, 10/10 scored rows,
and 18 guards. That was a false-green harness defect: it had not executed the
frozen short-length-trace guard or several other frozen fatal-family instances.
Review expanded the harness to 42 guard instances. The corrected run retained
10/10 matching rows but became void at 41/42. The original false-green output
does not qualify any behavior and is disclosed here rather than removed from
the chronology.

Implementation commit `0fae22e38e890c8a364ea3c467b35917f3eb2e3e` was then
replayed from a clean worktree. The corrected output was `VOID`, with 10/10
diagnostic matches retained and 41/42 fatal guard instances. The external
`summary.json` has SHA-256
`20f900418c813290b06fd652c0c85756d697f7fbbe399b14f10ea4ecb59865a2`.

The audited study ran with:

```bash
PYTHONPATH=. python examples/sglang_moe_workload_v1/run_study.py \
  --run-dir "${SIMLLM_RUN_ROOT:?configure SIMLLM_RUN_ROOT}/sglang-moe-workload-v1"
```

The command exits nonzero because the run is void. Bulk rows remain outside
Git; the tracked report records the compact findings.

## Retained findings

| Evidence class | Retained matches | Total | Interpretation |
|---|---:|---:|---|
| exact MoE geometry rows | 4 | 4 | Granite, Mixtral, Qwen3 MoE, dense TP identity |
| deterministic workload rows | 4 | 4 | trace/Poisson arrivals crossed with fixed/trace lengths |
| external timing-reducer rows | 2 | 2 | exact arrival-origin TTFT, TPOT, and submission lateness |
| fatal guard instances | 41 | 42 | one failure voids the scored rows |

The failed guard is `workload-short-length-trace-rejected`. The established
`TraceLengths.sample(n)` contract cycles its file when the requested count is
larger than the file. The new realization seam correctly consumes that public
provider output without inspecting private storage. Short `TraceArrivals`
still fails cardinality validation because its public `times(n)` returns fewer
rows. The frozen statement treated the two provider contracts as identical;
the result refutes that assumption rather than changing either contract after
observation.

The geometry rows map exactly to:

| Cell | Experts | Top-k | Expert width | Resident experts |
|---|---:|---:|---:|---:|
| Granite pilot | 32 | 8 | 512 | 32 |
| Mixtral | 8 | 2 | 14336 | 8 |
| Qwen3 MoE | 64 | 8 | 1408 | 64 |
| dense Llama TP4 | 0 | 0 | none | 0 |

Unknown and contradictory MoE identities, missing or invalid fields, top-k
bounds, shared experts, mixed routed/dense layers, MLA, next-token-prediction
layers, redundant experts, multimodal wrappers, quantized weights, and
distributed MoE geometry fail before a dense estimate can be emitted.

The trace arrivals `[0, 1.5e-12, 1e-6]` seconds became exactly
`[0, 2, 1000000]` ps. The Poisson seed 31 produced
`[13660228419, 14799185939, 35696822367]` ps. Fresh providers reproduced the
same plans; changing only the prompt seed changed token IDs without moving
request identity, arrival, or length.

The external timing fixture, which is distinct from core `RequestMetric`,
reduced to:

| Request | TTFT | TPOT | Submission lateness |
|---|---:|---:|---:|
| `r0` | 1100 ps | 200 ps/token | 100 ps |
| `r1` | 1000 ps | undefined for one token | 200 ps |

## Post-specified physical sanity

These checks were added after the frozen guard was refuted, so they are not
part of the frozen denominator. The pilot geometry has 32 resident experts and
top-k 8 across 24 routed layers. In BF16 its MLP contributes 1,207,959,552
resident parameters, or 2,415,919,104 resident bytes. One token activates
301,989,888 MLP parameters. This separates logical active work from resident
storage; it does not assert that every resident byte reaches DRAM.

For one decode token at context 2048, the current fused analytical surrogate
moves 2,692,743,168 bytes and performs 1,031,700,480 FLOPs. Before reading the
model duration, the compute floor and the model-implied all-resident memory
term are:

| Declared envelope | Compute floor | All-resident memory term | Roofline result |
|---|---:|---:|---:|
| 1.8 PFLOP/s, 8.0 TB/s | 0.573 us | 336.593 us | 336.593 us, memory-bound |
| 1.8 PFLOP/s, 4.0 TB/s | 0.573 us | 673.186 us | 673.186 us, memory-bound |

Halving the declared bandwidth therefore doubles this component duration
exactly. Increasing decode batch from 1 to 8 at fixed context moves the 8 TB/s
row from 336.593 us to 380.633 us, because the current surrogate streams the
same resident weights once per step and adds token-dependent KV and work. This
is a component relation, not request TTFT or TPOT.

The frozen factor of four is guaranteed for one routed token and can persist
for multiple tokens when all route to the same hot-8 set. For `N` routed
tokens the number of unique selected experts is between 8 and
`min(32, 8N)`, so the resident-to-selected logical weight ratio lies between
1 and 4. A GPU campaign must keep four quantities distinct: logical active
FLOPs, actually executed or padded FLOPs, selected logical weight bytes, and
measured DRAM bytes. COMP-7 owns replacement of the balanced, all-resident
surrogate with routed-expert supply.

## What is usable now

- The SGLang adapter recognizes strict TP=EP=MoE-DP=1 Granite, Mixtral, and
  Qwen3 MoE geometry rather than silently pricing them as dense.
- A deterministic plan can be built from existing arrival and length
  providers, mapped to native streaming `/generate`, paced open-loop against
  one monotonic origin, and reduced from validated cumulative token frontiers.
- One simulated GPU may use a built-in A100, H100, H200, B100, or B200
  analytical envelope, or a caller-injected `GpuSpec`. These are not
  production-calibrated latency tables.

The current path is not yet an end-to-end SGLang TTFT/TPOT simulation. The
worker's SGLang `StepRecord` still stops before per-request identity reaches a
metric-bearing `StepResult`; SGL-12 and SGL-13 own those missing links. HTTP
wall time is external observation and does not become virtual model time;
WORK-4 owns that server ingress coordinator. SGL-4 owns the live comparison.
SGL-18 owns the unsupported MoE geometry and mechanisms.

Absolute simulated TTFT and TPOT are also not defensible yet. The existing
[compute fidelity result](../compute_fidelity_v1/RESULTS.md) found that an
omitted fixed per-step cost measured on a non-production Turing anchor was
1.79 to 12.31 times the whole modeled decode compute for a 24-layer top-8 MoE
step. That run was itself void and transfers no duration to a production GPU,
but it identifies the dominant precision risk. COMP-1 and COMP-5 remain open.

## Critical cloud-GPU measurement

The first rental should measure the complete worker-step settlement, because a
DRAM counter alone cannot recover omitted host, launch, graph, synchronization,
or multi-stream frontier gaps.

Use two aligned observables:

1. Host monotonic time from `forward_batch_generation` entry until its return
   value is visible to the SGLang scheduler. Client SSE visibility is a
   separate end-to-end boundary and must not be substituted for this span.
2. Device CUDA-event time from the first submitted device work to one joined
   completion-frontier event that waits on every participating stream.

Cross-check the device interval and kernel-busy union with CUPTI or Nsight,
then retain DRAM bytes, kernel launch count, exposed gaps, graph/eager mode,
and per-layer routed-token histograms. DRAM bytes are the highest-value single
explanatory counter, but the host-to-visible settlement span is the most
critical target metric.

Before renting, freeze the target SKU/architecture and allocation criteria.
After an allocation-only feasibility probe, but before any timed or profiler
capture, add a target-specific expectations-only amendment. It must record the
actual GPU UUID and freeze checkpoint and revision, driver/CUDA/PyTorch/SGLang
and kernel hashes, weight and KV dtypes, clocks and power, eager versus decode
graph versus prefill mode, warmup count, cache protocol, routed-token
construction, repetitions, train/holdout cells, and error denominators. It
must also freeze the requested arrival, prompt/output length, offered-load,
and burst sweep. State a controlled ceiling from declared minimum sustained
compute and HBM service plus a maximum host/launch-gap budget, alongside the
physical floors. An out-of-memory result after the matrix is frozen voids the
run instead of justifying an outcome-dependent matrix reduction.
