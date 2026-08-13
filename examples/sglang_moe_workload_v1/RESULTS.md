# SGLang MoE workload v1 results

The frozen study is **void with findings**. Frozen fatal guard
`workload-short-length-trace-rejected` was violated, so no behavioral pass
fraction is interpretable. It failed because `TraceLengths` deliberately cycles
a short trace while the expectation file said every short trace would be
rejected, i.e. the freeze's own assumption was refuted. The 9 scored geometry,
workload and external timing-reducer rows all matched and are retained for
diagnosis rather than reported as a pass rate.

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

The first working-tree harness execution printed `PASS` and a full scored
sweep. That was a false-green harness defect: it had not executed the frozen
short-length-trace guard or several other frozen fatal-family instances.
Review expanded the harness to cover every fatal clause the freeze enumerates
as fatal, and the corrected run is void. Two frozen behavior statements outside
that enumeration still have no guard instance, namely that a first Poisson
arrival is not shifted to zero and that equal arrival times retain caller
order, the latter having no tie in either arrival family. The original false-green output does not qualify any
behavior and is disclosed here rather than removed from the chronology. Guard
instance counts are deliberately not published as a ratio: the fatal class has
no near-miss, one violated guard removes the precondition under which every
scored number means what it claims, and the instantiation count is a property
of the harness rather than a frozen denominator.

Implementation commit `0fae22e38e890c8a364ea3c467b35917f3eb2e3e` was then
replayed from a clean worktree, with the post-review reclassification below
applied. The corrected output is `VOID`, with 9 diagnostic matches retained.
The external `summary.json` has SHA-256
`6055c4511a91ec7781a441bfacd01719e223de8403342fa6dc19651c782c2176`, and it
carries the guard totals as separate integers rather than as a fraction.

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
| exact MoE geometry rows | 3 | 3 | Granite, Mixtral and Qwen3-shaped routed geometry |
| deterministic workload rows | 4 | 4 | trace/Poisson arrivals crossed with fixed/trace lengths |
| external timing-reducer rows | 2 | 2 | exact arrival-origin TTFT, TPOT, and submission lateness |

The fatal guards are not tabulated here, because they are not a scored class
and a ratio over them would read as a degree of voidness. One frozen fatal
guard was violated and the run is void; the remaining guards held.

Two post-review reclassifications, applied before the replay above and
disclosed rather than folded in silently:

- The `llama-dense-tp4` geometry cell is now a fatal-unscored guard,
  `geometry-configuration-forced-llama-dense-tp4`, not a scored row. Its
  expected tuple is all zeros because a dense config cannot report routed-MoE
  fields, so it cannot fail for any correct reader, and AGENTS.md keeps
  configuration-forced assertions out of every behavioral denominator. Its
  zero-expert fact was already asserted by the
  `geometry-dense-identity-and-granite-ratios` guard, which evaluates a dense
  llama at `tp_size=1` and `intermediate_size=512`; the two are not the same
  assertion, since the cell also pins `top_k`, `moe_intermediate_size` and
  `local_num_experts` at `tp_size=4` and `intermediate_size=14336`. The new
  guard keeps the full tuple, so nothing is lost, and the reclassification
  rests on the configuration-forced rule rather than on duplication. Geometry
  therefore reports 3 scored rows, not 4.
- The `trace_quantization_ok` conjunct is inert for the two Poisson cells,
  where the arrival name is not `trace`. Those two workload rows therefore
  carry only determinism and seed orthogonality, and are strictly weaker
  evidence than the two trace cells.

One further disclosure, post-specified and deliberately not applied to the
frozen expectations, since a freeze is never rewritten after a run: the
`qwen3-moe` cell's 64 experts, top-8 and expert width 1408 do not reproduce any
published Qwen3 MoE checkpoint. The pinned SGLang tree defines no `qwen3_moe`
config of its own: it imports Hugging Face's `Qwen3MoeConfig`, whose defaults
are 128 experts, top-8 and width 768. Every geometry cell in this study is a
synthetic config assembled on one shared 1024-hidden, 24-layer backbone, so the
cell tests that the reader consumes the `num_experts` / `num_experts_per_tok` /
`moe_intermediate_size` field names of that family, not that it reproduces a
checkpoint. Nothing downstream consumes the tuple, so the reclassified counts
above are unaffected. The rows below are labeled "shaped" for that reason, and
a cell carrying real checkpoint geometry belongs to SGL-4's calibrated
comparison rather than here.

The failed guard is `workload-short-length-trace-rejected`. The established
`TraceLengths.sample(n)` contract cycles its file when the requested count is
larger than the file. The new realization seam correctly consumes that public
provider output without inspecting private storage. Short `TraceArrivals`
still fails cardinality validation because its public `times(n)` returns fewer
rows. The frozen statement treated the two provider contracts as identical;
the result refutes that assumption rather than changing either contract after
observation.

The geometry rows map exactly to the synthetic cells below. Every one is a
config assembled on the shared 1024-hidden, 24-layer backbone, so the names
identify which family's field names are being consumed, not a checkpoint:

| Cell | Experts | Top-k | Expert width | Resident experts | Class |
|---|---:|---:|---:|---:|---|
| Granite-pilot-shaped | 32 | 8 | 512 | 32 | scored |
| Mixtral-shaped | 8 | 2 | 14336 | 8 | scored |
| Qwen3-shaped | 64 | 8 | 1408 | 64 | scored, and see the checkpoint disclosure above |
| dense Llama TP4 | 0 | 0 | none | 0 | fatal-unscored, configuration-forced |

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
