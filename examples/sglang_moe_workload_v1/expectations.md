# SGLang MoE workload v1 expectations

Frozen before the geometry fix, workload driver, tests, or study run in this
change. Nothing below is derived from a result observed after this file was
written.

## Scope and existing authorities

This slice makes a small routed MoE model and an open-loop request workload
representable at the SGLang boundary without importing SGLang in the test
process. It does not add another scheduler, virtual clock, or request metric
authority.

- SGLang remains the batching, prefix-cache, preemption, and token-generation
  authority when its scheduler is run.
- `RequestAdmissionGate` remains the in-process virtual admission authority.
- `StepResult` remains the simulated per-request TTFT and TPOT authority.
- The new transport reducer reports client-observed TTFT and TPOT from streamed
  token timestamps. Submission lateness is a separate diagnostic and is never
  silently relabeled as framework queueing.
- WORK-4 still owns a server-mode virtual ingress coordinator. SGL-12 still
  owns exact sampled-token attribution at the SGLang worker seam. SGL-4 still
  owns the live GPU comparison.

## Surrogate being replaced

`simllm.adapters.sglang.worker.model_dims_from_sglang` currently reads only
dense geometry. The pinned Granite pilot therefore reaches the shared compute
provider with `num_experts=0`, even though every one of its 24 MLP layers is a
32-expert routed layer with top-k 8 and expert width 512. Relative to a dense
MLP of width 512, this undercounts active MLP parameters by 8 and resident MLP
parameters by 32.

The source identity used to define the replacement is SGLang commit
`8f2a3ad6d7d68c58ae65b61a75bb2115449addca`.

Pinned-source mappings:

- Granite MoE passes `num_local_experts`, `num_experts_per_tok`, and
  `intermediate_size` to every routed block.
- Mixtral uses the same three fields.
- Qwen3 MoE passes `num_experts`, `num_experts_per_tok`, and
  `moe_intermediate_size` to every routed block.
- Qwen2 MoE includes a shared expert. DeepSeek-style models include shared
  experts, mixed dense and routed layers, and MLA attention. The current
  `ModelDims` cannot represent those mechanisms.

The Hugging Face field `num_local_experts` is the global expert count for
Granite and Mixtral. It must not be copied to a per-rank field without applying
the framework's expert-parallel ownership rule.

## Frozen geometry cells

The first supported MoE slice is exactly TP=EP=MoE-DP=1. Dense TP behavior is
the accepted identity baseline and remains available at larger TP sizes.

| Cell | Family | Experts | Top-k | Expert width | Resident experts | Expected behavior |
|---|---|---|---|---|---|---|
| `granite-pilot` | Granite MoE | 32 | 8 | 512 | 32 | accept exactly |
| `mixtral` | Mixtral | 8 | 2 | 14336 | 8 | accept exactly |
| `qwen3-moe` | Qwen3 MoE | 64 | 8 | 1408 | 64 | accept exactly |
| `llama-dense-tp4` | dense Llama | 0 | 0 | none | 0 | preserve accepted dense projection |

Fatal, unscored validation guards:

- Any MoE sentinel on an unknown architecture raises `NotImplementedError`.
  It never becomes a dense model.
- A supported family with a missing, noninteger, nonpositive, or disagreeing
  expert field raises `ValueError`.
- Top-k must be between one and the global routed expert count, inclusive.
- Positive shared-expert fields, a mixed dense/routed layer schedule, MLA,
  or next-token-prediction layers are rejected explicitly.
- Any supported MoE cell with TP, EP, or MoE-DP greater than one is rejected
  by this first slice. SGL-18 owns that deliberate carve-out.
- A dense config with no MoE sentinel produces the exact pre-change
  `ModelDims`, including its current loud fallback behavior.

Expected scored geometry outcome: 4 of 4 exact cells. A fatal guard failure
voids the geometry evidence.

## Frozen workload realization

The workload is realized before submission. Each immutable request carries a
stable ordinal, caller-supplied request ID, integer-picosecond arrival, prompt
token IDs, and requested output token count.

- Arrival providers use their existing `times(n)` contract in seconds.
  Conversion is decimal round-half-up to the nearest picosecond. The first
  Poisson arrival is not shifted to zero.
- Prompt and output length providers use their existing `sample(n)` contract.
- The caller-supplied prompt builder must return exactly the sampled number of
  nonnegative integer token IDs.
- Equal arrival times retain caller order. Request IDs are unique.
- Short traces are rejected instead of being silently cycled or padded.

Two arrival families and two length families form four realization cells:

| Family | A | B |
|---|---|---|
| arrivals | trace `[0, 1.5e-12, 1e-6]` seconds | Poisson 100 requests/s, seed 31 |
| prompt lengths | fixed 8 | trace `[3, 7, 11]` |
| output lengths | fixed 3 | trace `[3, 1, 2]` |

The trace arrival cell must become exactly `(0, 2, 1000000)` ps. Repeating a
cell with fresh providers and the same seed is byte-for-byte identical at the
request-field level. Changing only the prompt seed changes at least one token
ID but no request ID, arrival, or length.

Expected scored realization outcome: 4 of 4 family crossings. Cardinality,
identity, ordering, positivity, and token-bound checks are fatal and unscored.

## Frozen SGLang generate mapping

Each realized request maps to one native `/generate` payload:

```json
{
  "rid": "r0",
  "input_ids": [1, 2, 3],
  "sampling_params": {
    "temperature": 0.0,
    "max_new_tokens": 3,
    "ignore_eos": true
  },
  "stream": true,
  "return_logprob": false
}
```

The concrete HTTP transport paces requests against one monotonic origin,
submits equal-time requests in ordinal order, and allows earlier requests to
remain in flight while later arrivals are injected. It uses only the Python
standard library. SGLang and a GPU are not required to test it.

Native SGLang chunks report cumulative completion counts. When one observed
chunk advances the count by more than one, every newly visible token receives
that chunk's timestamp. This preserves the observable completion frontier and
does not invent within-chunk timing.

## Frozen observed metrics

For each request:

```text
submission_lateness = submitted_at - arrived_at
TTFT                = first_token_at - arrived_at
TPOT                = (last_token_at - first_token_at) / (output_tokens - 1)
```

TPOT is an exact `Fraction` and is `None` for one output token. The reducer
accepts transport observations in any completion order and restores request
ordinal order.

Exact fixture:

| Request | Arrival | Submit | Token completions | TTFT | TPOT | Submit lateness |
|---|---|---|---|---|---|---|
| `r0` | 0 | 100 | `(1100, 1300, 1500)` | 1100 | 200 | 100 |
| `r1` | 1000 | 1200 | `(2000,)` | 1000 | none | 200 |

Missing, duplicate, or foreign observations; wrong output counts; submission
before arrival; token completion before submission; and decreasing token
timestamps are fatal. Expected scored metric outcome: 2 of 2 exact rows.

## Physical sanity and the future GPU measurement

The analytical model must state both independent floors before reading a
measured step:

```text
compute floor = active FLOPs / declared peak FLOP/s
memory floor  = measured DRAM bytes / declared peak HBM byte/s
step floor    = max(compute floor, memory floor)
```

The current roofline streams every resident expert once per step. That is an
explicit conservative surrogate, not a fact about SGLang's fused MoE kernels.
For the Granite pilot, a cold selected-expert working set and an all-resident
working set differ by four because top-k is 8 while resident experts are 32.
Only DRAM counters under a declared cache protocol can identify which byte
model applies in steady state.

The most critical cloud-GPU measurement is the full SGLang step-settlement
span from forward submission to the true completion frontier, using CUDA
events and a CUPTI or profiler cross-check. If only one explanatory hardware
counter is affordable, record DRAM bytes. Also retain kernel launch count,
the union of kernel-busy intervals, exposed launch or graph gaps, and the
per-layer routed-token histogram. Those observables separate kernel service
from host gaps and identify whether expert weights are fetched as selected or
resident sets.

The initial single-GPU BF16 capture matrix, reduced only if the rented GPU
runs out of memory before the matrix is frozen, is:

- decode batch `B={1,2,4,8,16,32,64}` at context 2048;
- decode context `C={128,256,512,1024,2048,4096,8192}` at batch 4;
- cross-axis holdouts `(B,C)={(2,256),(8,1024),(32,4096)}`;
- prefill tokens `T={16,32,64,128,256,512,1024,2048}`;
- fixed `T=512` split across `R={1,4,8,16}` requests;
- MoE routed rows `N={8,64,512}` under uniform, captured, and legal hot-8
  expert histograms.

Record 41 repetitions after warmup at fixed clocks, exact SGLang and model
commits, graph mode, dtype, and TP=EP=PP=1. Use powers-of-four anchors and
geometric-midpoint holdouts. The target acceptance is controlled-cell CV
below 2 percent, full-step error below 5 percent, phase median and p95 error
below 5 and 10 percent, and kernel median and p95 error below 10 and 20
percent. Unsupported multi-axis shapes must miss loudly rather than
extrapolate.

No live-GPU latency is claimed by this slice.

## Registry discipline

SGL-18 owns distributed and hybrid MoE geometry after this single-GPU slice.
The already registered SGL-4, SGL-12, and WORK-4 clauses own live calibration,
exact worker sampled-token attribution, and virtual server ingress
respectively. This change does not duplicate those authorities.
