# SGL-16 expectations: SGLang-supplied dispatch layer identity

Frozen before the implementation and before every run reported by this study.
Nothing below is derived from a measurement taken after this file was written.

## What is being replaced

`simllm.adapters.sglang.oracle._observe_routed_capture` currently invents the
dispatch layer label that SGLang's post-selection capturer never receives. The
pinned SGLang Granite MoE block builds its router without a layer identity:

- `python/sglang/srt/models/granitemoe.py:46` takes `layer_id: int` in
  `GraniteMoeMoE.__init__`.
- `python/sglang/srt/models/granitemoe.py:65-68` constructs
  `TopK(top_k=..., renormalize=True)` and does not forward `layer_id`.
- `python/sglang/srt/models/granitemoe.py:71-79` constructs
  `FusedMoE(..., layer_id=layer_id, ...)`, so the sibling expert module does
  receive the explicit identity.
- `python/sglang/srt/layers/moe/topk.py:432` stores `self.layer_id`, which is
  therefore `None` for every Granite MoE block.
- `python/sglang/srt/layers/moe/topk.py:486` and `:567` pass that `None`
  through `select_experts(layer_id=...)`.
- `python/sglang/srt/layers/moe/topk.py:1864` forwards it to
  `capture_routed_experts_if_allowed(topk_config, layer_id, topk_ids)`,
  defined at `:1829-1845`.
- `python/sglang/srt/state_capturer/base.py:38-40` writes
  `self.buffer[:batch, layer_id, :] = topk_indices`, so a `None` label is a
  hard failure rather than a silent one.

The surrogate closes that gap by cycling a per-capturer counter modulo the
model's 24 MoE modules. It is correct only while every forward pass visits
every MoE module exactly once in registration order, which nothing in SGLang
guarantees and which a hybrid or partially dense MoE model breaks.

## The replacement

SGLang's own explicit layer identity is bound to the capture site by module
identity, mirroring the vLLM oracle's `_cpu_layer_ids` map
(`simllm/adapters/vllm/oracle.py:334-352`):

1. An AROUND hook on
   `sglang.srt.state_capturer.routed_experts.RoutedExpertsCapturer.create`
   receives the constructed model and walks `model.named_modules()`.
2. For every `TopK` module it resolves an explicit framework layer id: the
   module's own `layer_id` when SGLang set one, otherwise the unique integer
   `layer_id` carried by a sibling module of the same parent (the
   `FusedMoE` built at `granitemoe.py:71-79`).
3. The resolved id is cross-checked against the layer index in the module's
   own registered name before it is accepted.
4. An AROUND hook on
   `sglang.srt.layers.moe.topk.capture_routed_experts_if_allowed` substitutes
   the resolved id when and only when SGLang passes `None`, keyed on the
   identity of the per-module `TopKConfig`.
5. The hook on `RoutedExpertsCapturer.capture` no longer infers anything. A
   `None` label there is a hard error.

No selected expert id, routing tensor, sampling decision or scheduler state is
touched. `layer_id` reaches only three places in the pinned source
(`topk.py:1877` LP solver, guarded by `_is_cuda` and an `lp` dispatch
algorithm; `topk.py:2226` round-robin benchmark override, guarded by
`simulate_round_robin_experts`; and the capture label itself), none of which
is reachable on the CPU engine this oracle selects.

## Provenance

- SGLang commit this evidence was authored against:
  `8f2a3ad6d7d68c58ae65b61a75bb2115449addca`.
- The commit each run actually observed is recorded next to it in
  `summary.json` with no equality assumed between the two.
- Model `ibm-granite/granite-3.0-1b-a400m-instruct`, revision
  `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`.
- Engine: `sglang.Engine`, `device="cpu"`, `dtype="float32"`, `tp_size=1`,
  `page_size=1`, `disable_overlap_schedule=True`, `random_seed=173`,
  `temperature=0`, 8 torch threads, `enable_return_routed_experts=True`.
- Machine-local paths come from the ignored local environment
  (`SIMLLM_SGLANG_SOURCE`, `SIMLLM_SGLANG_PYTHON`, `SIMLLM_HF_CACHE`,
  `SIMLLM_WAVE11_RUN_ROOT`). None is written into a tracked file.

## Frozen parameter families

Two prompt shapes and one decode-retraction cell, exactly as required by the
registered SGL-16 clause. Prompt token ids are frozen formulas, not text.

| cell | requests | prompt tokens per request | `max_new_tokens` | `max_total_tokens` | `context_length` | `max_running_requests` |
|---|---|---|---|---|---|---|
| `short` | 1 | 8 | 4 | 256 | 512 | 2 |
| `long` | 1 | 96 | 8 | 256 | 512 | 2 |
| `preempt` | 8 | 8 | 20 | 96 | 512 | 8 |

Prompt token ids:

- `short`: request `s0` gets `100 * (i + 1)` for `i` in `range(8)`.
- `long`: request `l0` gets `1500 + i` for `i` in `range(96)`.
- `preempt`: request `p<k>` gets `1000 + 100 * k + i` for `i` in `range(8)`,
  `k` in `range(8)`.

The `preempt` capacity was chosen by a pre-freeze feasibility probe whose only
observed quantity was whether a `retract_decode` occurred at all. No expert id,
output token, label or byte comparison was read before this file was written.
That probe also established two facts about the pinned scheduler that the
frozen design depends on and that are recorded here as context, not as
results: SGLang clamps `max_new_tokens` to the token-pool capacity rather than
retracting a lone request, and its prefill admission control reserves decode
headroom, so a retraction needs more concurrent requests than the pool can
hold rather than one oversized request.

## Two phases

Both phases run the identical harness and the identical cells. They differ
only in which commit's `simllm` supplies the dispatch layer label.

- `baseline`: the model-order surrogate, i.e. the code as of this commit,
  which is the currently qualified Granite fallback the registered clause
  names.
- `treatment`: SGLang's explicit layer identity.

## Fatal, unscored guards

A violation voids the run. These are configuration-forced or by-construction
facts, so they never enter a behavioral fraction.

- G1: each cell's `worker-qualified` row reports `TpModelWorker`,
  `ModelRunner`, `GraniteMoeForCausalLM` and parameter devices exactly
  `["cpu"]`, in both phases.
- G2: each cell's `capture-storage-qualified` row reports an unpinned CPU
  buffer, in both phases.
- G3: the `baseline` trace provenance carries
  `dispatch_layer_mapping="granite-model-order"` and the `treatment` trace
  carries `dispatch_layer_mapping="framework-layer-id"`, the same value the
  vLLM runner already writes at `simllm/preplay/framework_runner.py:1229`.
- G4: the `treatment` `dispatch-layer-qualified` row reports
  `mapping="framework-layer-id"`, `selected_experts_unchanged=true`, and a
  `layer_ids` list that is exactly the integers `0` through `23`, one per
  SGLang MoE module, each resolved from SGLang's own explicit identity and
  each agreeing with the layer index in its module's registered name.
- G5: the `preempt` cell observes at least one `preemption` KV event, the
  named request reports `framework_preemption_count == 1`, and that request
  still reaches its length cap, i.e. it resumed rather than aborted. This must
  hold in both phases.
- G6: zero layer-label disagreements. With the audit enabled, every capture in
  the `treatment` phase reports the framework-supplied label equal to the
  label the model-order surrogate would have produced. This guard is fatal and
  unscored rather than scored because R1 below entails it: the routed-expert
  buffer is indexed by the label, so identical response bytes cannot coexist
  with a disagreeing label on an exercised layer.
- G7: no run writes any file outside its run directory, and `--check-only`
  reports `artifacts_written == 0`.

## Scored behavioral relations

Evaluated against raw observations first, per the entailment rule. Three
families, three instances each, nine instances total.

- R1, raw framework response identity (3 instances, one per cell). The raw
  SGLang response JSON for every request, including the base64
  `routed_experts` payload, the output token ids, the finish reason and the
  cached-token and preemption counters, is byte-identical between `baseline`
  and `treatment`. This is the raw observation. The derived v2 trace's
  `request` and `observed-dispatch` rows are entailed by it and are therefore
  checked as fatal-unscored consistency rather than scored again.
- R2, KV event identity (3 instances, one per cell). The `kv-event` rows of
  the v2 trace, which come from the allocation, prefix, eviction, retraction
  and release hooks rather than from the response, are byte-identical between
  phases. This channel is independent of R1: nothing about the response bytes
  constrains the paged-allocator sidecar.
- R3, discriminating power (3 instances, one per cell). The observed per-token
  layer-to-expert map is not invariant under a one-layer cyclic rotation of
  the layer labels. Without this, R1 and R2 would be satisfied by a degenerate
  model that routes identically at every layer, and the comparison would carry
  no information. Expected direction: strictly more than 90 percent of
  forwarded tokens change at least one layer's expert tuple under the
  rotation, in every cell.

Expected outcome, stated before the run: 9 of 9. R1 and R2 are expected to
hold exactly because the replacement changes only which integer labels an
already-computed expert tuple is filed under, and the frozen guard G6 asserts
those integers are the same ones for this model. R3 is expected to hold
because a 32-expert top-8 router over 24 layers with independently trained
gates has no reason to select the same tuple at two layers.

If R1 or R2 fails, the replacement changed framework behavior and SGL-16 does
not close. If R3 fails, R1 and R2 are vacuous for that cell and SGL-16 does
not close on that cell's evidence.

## What this study does not establish

- No timing is introduced, so the physical-sanity bounds have no target here.
  This study reports no latency, no bandwidth and no derived rate.
- The oracle remains a capture path. This study does not connect the SGLang
  adapter's step records to `StepResult`, `CompletionEvent`, TTFT or TPOT, and
  it makes no claim about the simulated SGLang worker or its communicator.
- Only one model exercises the sibling-`layer_id` resolution path. A model
  whose MoE blocks pass `layer_id` into `TopK` directly, or whose MoE blocks
  are a strict subset of its decoder layers, is not covered.

## Reproduction

```bash
python examples/sglang_layer_id_v1/run_study.py \
  --run-dir "${SIMLLM_WAVE11_RUN_ROOT:?configure SIMLLM_WAVE11_RUN_ROOT}/sgl16" \
  --sglang-source "${SIMLLM_SGLANG_SOURCE:?configure SIMLLM_SGLANG_SOURCE}" \
  --sglang-python "${SIMLLM_SGLANG_PYTHON:?configure SIMLLM_SGLANG_PYTHON}" \
  --model-path "${SIMLLM_GRANITE_MOE_PATH:?configure SIMLLM_GRANITE_MOE_PATH}" \
  --check-only
```

Then `--phase baseline` at this commit, `--phase treatment` after the
implementation lands, and `--phase compare` to score the frozen relations.
