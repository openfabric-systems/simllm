# Step-sink latent knobs: pre-registered expectations

Written and frozen before the COMP-16 and VLLM-15 implementation and before
any run of this study. The immutable pre-change reference is SimLLM commit
`6aa3a76`. No value below is fitted to an implementation or study result.

The frozen requirement is that raw outputs remain outside Git in a
machine-local external directory. The resolved historical target is
intentionally omitted. As a post-freeze portability convention, new runs
default to `${SIMLLM_DATA_ROOT}/step_sink_latent_knobs/`.

## Freeze audit and registered commands

Both registered runner modes were executed with `--check-only` before this
file was frozen. Check-only mode parses the complete command, validates its
input paths and version pin, does not construct a provider, adapter, sink, or
vLLM engine, and does not create the output directory or any result file.

The historical dry runs used the same executable basenames, scripts, options
and pinned inputs; resolved machine-local paths are intentionally omitted. The
following blocks are portable post-freeze renderings, not verbatim
transcripts. Source the local configuration first.

Deterministic study:

```bash
SIMLLM_HTSIM_RNIC="${SIMLLM_HTSIM_BUILD:?configure SIMLLM_HTSIM_BUILD}/datacenter/htsim_rnic" \
SIMLLM_TXT2BIN="${SIMLLM_TXT2BIN:?configure SIMLLM_TXT2BIN}" \
.venv/bin/python examples/step_sink_latent_knobs/run_study.py \
  --mode deterministic \
  --out "${SIMLLM_DATA_ROOT:?configure SIMLLM_DATA_ROOT}/step_sink_latent_knobs"
```

Pinned-runtime smoke:

```bash
env PYTHONPATH=. VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  VLLM_USE_V2_MODEL_RUNNER=0 SIMLLM_VLLM_WORKER_MODE=skeleton \
  SIMLLM_VLLM_MODE=virtual HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_HOME="${HF_HOME:?configure HF_HOME}" \
  CUDA_VISIBLE_DEVICES= \
  "${SIMLLM_VLLM_PYTHON:?configure SIMLLM_VLLM_PYTHON}" \
  examples/step_sink_latent_knobs/run_study.py \
  --mode live-vllm \
  --out "${SIMLLM_DATA_ROOT:?configure SIMLLM_DATA_ROOT}/step_sink_latent_knobs"
```

The dry runs used the historical resolved forms of these commands with
`--check-only` appended.

## External-source audit before freeze

The adapter expectation mirrors vLLM, so its source was audited before this
freeze. SimLLM pins vLLM `0.26.0` at
`simllm/adapters/vllm/_version.py:8-9`. The installed pinned sources had these
digests during the audit:

| vLLM v0.26.0 source | SHA-256 | Audited lines and meaning |
|---|---|---|
| `vllm/v1/core/sched/scheduler.py` | `2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941` | `502-510`: cap each request's scheduled work by the token budget; `1246-1262`: add scheduled tokens to the computed count and classify an unfinished prefill chunk; `1670-1673`: read the sampled-token row by request index. |
| `vllm/v1/outputs.py` | `1e87bf44162452c1908d3a5003685937dbdc56f5634e35e11ed7b6a5322a1c15` | `231-244`: `ModelRunnerOutput.sampled_token_ids` has one variable-length generated-token list per request. |
| `vllm/inputs/llm.py` | audited from the same installed v0.26.0 package | `106-113`: `TokensPrompt.prompt_token_ids` accepts an explicit token-id list. |

The live smoke therefore supplies three explicit prompt token IDs rather than
depending on tokenizer output. With a token budget of two, the source-backed
schedule is a two-token unfinished prefill, a one-token prompt-completing
prefill, then one decode step for the second requested output token. The
expected exact sample-count sequence is `(0, 1, 1)`.

The compute matrix uses `GPU_ENVELOPES["b100"]` from pre-change SimLLM commit
`6aa3a76`, `simllm/compute/transformer.py:35-41`, as a declared model input:
`1.8e15` FLOP/s and `8.0e12` byte/s. It does not assert that these internal
envelope values are an independently validated hardware specification.

## Evidence classes

- Check A has four scored exact analytical rows over layer count and TP width.
- Check B1 is one scored adapter-to-TTFT relation. Check B2 has five scored
  sample-attribution instances.
- Check C is one scored live vLLM v0.26.0 relation.
- Check D and every conservation, schema, byte-lock, and invalid-input check
  are fatal structural guards. They are unscored and do not increase a
  behavioral denominator.
- Run configurations, provider outputs repeated from the configured inputs,
  and source-version echoes are unscored.

No author-defined call or event sequence is scored.

## Check A: roofline family split reaches first-token latency

The matrix varies transformer layer count `L` in `{2, 4}` and TP width `W` in
`{2, 4}`. Every cell schedules one decode token at post-step context four
with exact sample count one. Per-rank geometry is hidden size 64,
intermediate size 128, four attention and KV heads of width 16, vocabulary
size 256, and two-byte elements. The provider is
`RooflineProvider(efficiency=0.7)`. The backend is `rnic-nn-fluid` at 400
Gbit/s, and host initiation is zero.

The scalar roofline remains

```text
E = floor(10^12 * max(flops / (peak_flops * 0.7),
                      bytes / (mem_bandwidth * 0.7))).
```

Both registered shapes are memory-bound. The family weights are therefore
the bytes of the existing `step_kernels` decomposition. All families except
`lm_head` are divided equally over `L` transformer layers. The complete
`lm_head` family, including its sampling projection, is added to the last
layer. If `q_i` is the resulting nonnegative layer weight, integer layer
boundaries are

```text
p_i = floor(E * sum(q_0 .. q_i) / sum(q_0 .. q_(L-1)))
d_0 = p_0
d_i = p_i - p_(i-1)
p_(L-1) = E exactly
```

The final assignment is explicit, so the returned nonnegative durations sum
to the scalar estimate exactly despite integer rounding. GOAL then applies
its already accepted cumulative picosecond-to-nanosecond rule:

```text
c_0 = floor(d_0 / 1000)
c_i = floor(sum(d_0 .. d_i) / 1000)
      - floor(sum(d_0 .. d_(i-1)) / 1000).
```

The frozen provider and rendering values are:

| L | E ps | fused bound | non-head family bytes | LM-head bytes | layer duration ps | enabled calc ns | disabled calc ns per layer |
|---:|---:|---|---:|---:|---|---|---:|
| 2 | 35,474 | memory | 165,888 | 32,768 | 14,811; 20,663 | 14; 21 | 17 |
| 4 | 65,097 | memory | 331,776 | 32,768 | 14,811; 14,811; 14,812; 20,663 | 14; 15; 15; 21 | 16 |

Each layer's two 128-byte tensor-parallel ring allreduces remain serial. With
`P = 2,000,000 ps`, 20 ps per payload byte, and exact chunk `128/W`, the
closed forms are

```text
N(L, W) = 2L * 2(W-1) * ((128/W) * 20 + P)
J_disabled = N(L, W) + 1000L * floor(E / (1000L))
J_enabled  = N(L, W) + 1000 * floor(E / 1000).
```

This is the first scheduled step, so its `StepResult.step_latency_ps`, JCT,
and TTFT are the same quantity. The registered end-to-end rows are:

| L | W | disabled JCT/TTFT ps | enabled JCT/TTFT ps | signed delta ps | flows |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 16,044,240 | 16,045,240 | +1,000 | 16 |
| 2 | 4 | 48,049,360 | 48,050,360 | +1,000 | 96 |
| 4 | 2 | 32,084,480 | 32,085,480 | +1,000 | 32 |
| 4 | 4 | 96,094,720 | 96,095,720 | +1,000 | 192 |

Enabling the split moves work into the last layer and preserves one more
whole nanosecond than the independently truncated even split. The final
allreduce boundary and TTFT must therefore move later by exactly 1,000 ps in
all four cells. Every measured residual against the closed form must be zero.
At fixed `W`, increasing `L` increases TTFT. At fixed `L`, increasing `W`
increases TTFT.

This relation is decision-relevant. If the selected-resource family weights
do not conserve the fused work, or if any returned layer vector cannot be
nonnegative and sum to `E` exactly, the `estimate_layers(kernel, gpu,
num_layers)` contract is insufficient. The design decision would change to
carry an explicit layer-work object rather than ship this provider
implementation. Conservation itself remains a fatal unscored structural
guard, as required by the evidence rules.

## Check B: adapter-produced exact samples reach TTFT

### B1. Chunked-prefill metric relation

The actual vLLM `StepTranslator` receives one scheduler-shaped batch:

- request `p` has a 12-token prompt, four cached tokens, and four newly
  scheduled tokens, so post-step context is eight and it remains mid-prompt;
- request `d` is attached mid-flight with 31 computed tokens and one output
  token, then schedules one token, so post-step context is 32 and it samples.

The produced `StepRecord` must carry `num_sampled=1`. The same record with
only that optional field removed is the explicit compatibility bypass and
therefore falls back to two scheduled rows.

This check reuses the independently frozen BACK-6 analytical provider: one
modeled FLOP costs one picosecond and bytes do not contribute. With `L=2`,
`W=2`, the Check A geometry, and `rnic-nn-fluid`, the exact relations are:

| quantity | absent-field bypass | adapter exact field | exact minus bypass |
|---|---:|---:|---:|
| sample count | 2 | 1 | -1 |
| fused estimate ps | 912,896 | 880,128 | -32,768 |
| rendered calc ns per layer | 456 | 440 | -16 |
| JCT/TTFT ps | 16,963,200 | 16,931,200 | -32,000 |

The live adapter path must decrease TTFT by exactly 32,000 ps with zero
residual. If the record count disagrees with the fabricated output rows, the
design decision changes: attribution moves from the translator to the
post-fabrication output seam instead of closing VLLM-15 here.

### B2. Attribution matrix

The same translator must populate these exact record counts. In every row the
count must equal `sum(produces_token)` and the number of nonempty fabricated
`ModelRunnerOutput.sampled_token_ids` rows.

| case | scheduled rows | expected `num_sampled` |
|---|---:|---:|
| mid-prompt chunk | 1 | 0 |
| prompt-completing chunk after a prefix hit | 1 | 1 |
| prefix-cache completion on admission | 1 | 1 |
| decode | 1 | 1 |
| attach-mid-flight fallback | 1 | 1 |

These are scored attribution instances because a plausible translator could
incorrectly price every scheduled row or omit the exact field.

## Check C: real vLLM v0.26.0 smoke

The pinned-runtime command constructs the real in-process vLLM engine with
the dotted `SimWorker`, the cached Granite snapshot already used by the
accepted skeleton smoke, one explicit three-token `TokensPrompt`,
`max_num_batched_tokens=2`, `max_num_seqs=1`, and two requested output
tokens. It must generate exactly two copies of the worker's fabricated token
and stream exactly three records.

The streamed records must have scheduled-token counts `(2, 1, 1)` and exact
sample counts `(0, 1, 1)`. The sum of the exact counts must equal the two live
generated tokens. This whole relation is scored live integration evidence,
not a structural appendix. A runtime that ignores the token budget, a worker
path that bypasses translation, or an attribution error can fail it.

## Check D: bypass and compatibility guards

- `RooflineProvider(efficiency=0.7)` with no opt-in must return no layer
  breakdown. The pre-change default sink case must retain GOAL SHA-256
  `f8aade109ba8e3a581b7d965b3a0c76c1247016a1e37491fa84efbbf377677a5`
  and the existing 12,030 ns even layer value.
- A provider with no breakdown keeps the byte-identical even split.
- A manually constructed `StepRecord` with absent `num_sampled` omits the
  JSON field, round-trips through the v1 reader as absent, and retains the
  scheduled-row fallback exactly. A present zero remains present.
- An enabled roofline returns `None` for a kernel with no family metadata.
  Invalid layer counts or nonconserving family metadata are rejected.
- No profile-table or trace-calibrated breakdown is claimed before COMP-6
  supplies per-layer captured shapes. The owning registry must retain that
  residual work under a new COMP identifier.

All Check D items are fatal and unscored.

## Acceptance

- Check A passes 4/4 exact end-to-end rows and both signed monotonic
  directions, with zero timing residual.
- Check B1 passes its exact `-32,000 ps` adapter-to-TTFT relation. Check B2
  passes 5/5 exact attribution instances.
- Check C passes its scored live `(0, 1, 1)` sample-count relation.
- Check D and the decision-relevant exact-sum contract pass without being
  added to a behavioral score.
- Any unexplained residual, nonzero source mismatch, byte-lock change,
  schema regression, invalid provider acceptance, or unverified backend
  quiescence fails the study.
