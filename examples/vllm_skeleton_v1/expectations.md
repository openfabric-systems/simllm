# vLLM worker skeleton v1 expectations

The original expectations in this specification were frozen in commit
`582d3de` before the VLLM-13 worker skeleton implementation and every initial
study run. The review-triggered amendment at the end is intentionally later:
it follows integration review but precedes the fix-round implementation and
every fix-round run. It is not a retroactive pre-registration claim for the
initial implementation. The study targets the pinned vLLM v0.26.0 source and
the first, GPU-less slice only. It makes no claim about the later GPU-present
rebound runner, data-parallel coordination, simulated communicators, or CQ
consumers.

## Source-frozen boundary

The pinned source establishes these control-flow facts:

- `WorkerWrapperBase.init_worker` loads general plugins, requires
  `parallel_config.worker_cls` to be a dotted string, resolves it, and then
  constructs the class (`vllm/v1/worker/worker_base.py:245-259,317-320`).
- The in-process executor calls `init_device` and then `load_model`
  (`vllm/v1/executor/uniproc_executor.py:62-69`). The stock worker constructs
  its GPU model runner at the end of `init_device`
  (`vllm/v1/worker/gpu_worker.py:397-416`).
- Engine initialization requests the KV specification, profiles available
  memory when a KV cache exists, allocates the cache, and then compiles or
  warms the model (`vllm/v1/engine/core.py:243-324` and
  `vllm/v1/executor/abstract.py:118-136`).
- In-process `LLM` construction then resets the worker multimodal cache and
  queries the supported tasks (`vllm/v1/engine/llm_engine.py:123-142,205-210`
  and `vllm/entrypoints/llm.py:338-348`).
- The ordinary engine step calls `execute_model`; when the V1 runner returns
  `None`, it immediately calls `sample_tokens`
  (`vllm/v1/engine/core.py:576-606`). The stock worker delegates both calls
  through `self.model_runner` (`vllm/v1/worker/gpu_worker.py:1078-1178`).

The skeleton must therefore preserve this exact required worker-level
initialization sequence for the study fixture:

1. `init_device`
2. `load_model`
3. `get_kv_cache_spec`
4. `determine_available_memory`
5. `initialize_from_config`
6. `compile_or_warm_up_model`
7. `reset_mm_cache`
8. `get_supported_tasks`

The first six calls are the engine-core initialization prefix. Calls 7 and 8
complete the ordinary in-process `LLM` construction sequence. The fixture
does not use `max_model_len=-1`, KV transfer, speculative
decoding, structured output, pooling, pipeline parallelism, data parallelism,
LoRA, profiling, sleep, encoder reset, or prefix reset. Their worker names
must remain callable or fail explicitly as documented, but they do not enter
the scored sequence. In particular, the conditional `update_max_model_len`
call is absent from this fixed configuration.

For each non-empty generation step, the exact mirrored call sequence is:

1. `worker.execute_model`
2. `runner.execute_model`
3. `runner._update_states`
4. `runner._prepare_inputs`
5. `runner._determine_batch_execution_and_padding`
6. `runner._build_attention_metadata`
7. `runner._preprocess`
8. `runner._model_forward`
9. `worker.sample_tokens`
10. `runner.sample_tokens`
11. `runner._sample`
12. `runner._update_states_after_model_execute`
13. `runner._bookkeeping_sync`
14. `runner.eplb_step`

This is the selected copied V1 model-runner algorithm shape at
`vllm/v1/worker/gpu_model_runner.py:4111-4479,4497-4736`, trimmed to the
ordinary text-generation path. `_model_forward` is deliberately empty in
this slice. Sampling preserves the existing SimExecutor fabricated-token
contract so the real CPU scheduler can continue, but no model computation or
device state is created. The live Granite smoke sets
`VLLM_USE_V2_MODEL_RUNNER=0` so the upstream selector agrees with this copied
path. Supporting or rebinding both stock runner variants under the later
GPU-present mode remains outside this slice.

## Entry-gate expectations

The high-level gate is exactly
`SIMLLM_VLLM_WORKER_MODE=skeleton`.

- Constructing `simllm.adapters.vllm.SimWorker` with the variable absent,
  empty, or set to any other value must raise before stock `init_device` can
  run. The error must name `SIMLLM_VLLM_WORKER_MODE=skeleton`.
- With the exact value `skeleton`, `init_device` must leave the physical
  device unset and construct the adapter's mirrored model runner, never the
  stock `GPUModelRunner`.
- Selecting the class continues to require the dotted CLI value
  `--worker-cls simllm.adapters.vllm.SimWorker`; a class object is not an
  accepted vLLM seam.

These are fatal configuration invariants. They do not contribute to the
behavioral pass denominator.

## Scripted sweep

Use one deterministic two-step request script for every Cartesian cell:

- request count `R` in `{1, 3}`;
- prompt tokens per request `P` in `{4, 16}`;
- step 0 admits all requests and schedules all `P` prompt tokens;
- step 1 schedules one decode token for every request;
- fabricated token id is fixed and all requests use distinct stable ids.

This is four run configurations and two parameter dimensions. The exact
record relations are:

| Step | Phase | Scheduled entries | Total new tokens |
|---:|---|---:|---:|
| 0 | prefill | `R` | `R * P` |
| 1 | decode | `R` | `R` |

Increasing `R` from 1 to 3 triples both scheduled-entry counts and both token
totals. Increasing `P` from 4 to 16 multiplies only the prefill token total by
four; it leaves the decode total and step count unchanged. Each cell must emit
exactly two records through the existing schema-tagged streaming path, in
step-index order `{0, 1}` with schema `atlahs-closed-loop-step-v1`.

## Exact virtual-clock relation

Let `T_i` be the SimLLM core `VirtualClock.now_ps` when step `i` is released,
and let `L_i` be the skeleton result latency. The scripted fixture injects one
central clock starting at `T_0 = 123000 ps`, which makes a hardcoded zero
timestamp observable. With deliberate model compute empty and no downstream
sink in this study:

    T_0 = 123000 ps
    L_i = 0 ps
    completed_at_i = T_i + L_i
    T_(i+1) = completed_at_i

Therefore both steps in all four cells must have:

- `StepRecord.virtual_time_ps = 123000`;
- `StepResult.step_latency_ps = 0`;
- `StepResult.completed_at_ps = 123000`;
- final virtual-clock time `123000 ps`.

Every mirrored call record must obtain its start and completion timestamps
from that same `VirtualClock`. Since the bodies are empty, both timestamps are
exactly `123000 ps` and completion never precedes start. Request count and
prompt length may change bookkeeping volume, but cannot change skeleton
latency.

## Evidence accounting

Behavioral exact-oracle rows are the four sweep cells. A cell passes only if
its two record shapes, both zero-latency relations, final clock, and both
parameter-scaling relations match exactly. The call-sequence comparison is a
separate exact structural check and is reported separately from the four
behavioral rows.

Flag refusal, no physical device or stock runner, schema identity, monotonic
timestamps, unique request ids, and record/result count equality are fatal
unscored invariants. They can fail the study but never increase its behavioral
pass count. Unit-test counts and the live vLLM smoke outcome are separate
evidence classes and are not added to the four-row headline.

## Review-triggered fix-round amendment

This amendment freezes the expectations raised by integration review before
their implementation or any fix-round execution.

### Independent call-sequence oracle

The deterministic study harness must define the eight initialization names
and fourteen non-empty step names above as literal tuples in the harness. It
must not import either sequence constant from `simllm.adapters.vllm.worker`.
The recorded implementation calls are compared against those independent
literals in every sweep cell. The implementation may retain exported
constants for consumers, but they are not study oracles.

### Dual-environment worker construction

The same SimWorker mirror tests must execute in both environments:

- the repository `.venv`, where vLLM is absent and the transcribed worker
  stand-in is active;
- the pinned `venv-vllm`, where `SimWorker` subclasses the real v0.26.0
  `Worker` and its real constructor runs.

The fake configuration and fallback constructor must expose equivalent
constructor-facing attributes. No mirror test may be skipped merely because
vLLM is installed. Flag refusal, V1 selection, backend refusal, source-order
initialization, non-empty and drain steps, central clock timestamps, record
streaming, and rank-zero authority must keep the same asserted outcomes in
both environments.

### Structured-output refusal and worker surface

Both simulation entry paths must reject a scheduler output whose
`has_structured_output_requests` value is true before fabricating a token. The
executor guard must not depend only on `structured_output_request_ids`, which
is not the v0.26.0 scheduler-output signal. The refusal is a fatal VLLM-8
correctness gate and remains unscored.

The mirrored worker surface must contain only source-supported worker methods.
`reset_prefix_cache` is scheduler-only in v0.26.0 and must not be exposed as a
worker mirror without a pinned worker call site.

### Strengthened live smoke

Exactly one live in-process smoke is attempted in this fix round with the same
cached Granite model, offline environment, dotted worker class, V1 selection,
and multiprocessing disabled. A successful smoke must assert all of the
following before it exits zero:

- exactly one generated request with exactly two sampled token ids;
- every sampled token id equals the constructed worker's configured
  fabricated token id;
- the reached worker owns a `SimModelRunner`, not a stock runner;
- the configured step-record JSONL exists and contains exactly two objects;
- both objects carry schema `atlahs-closed-loop-step-v1`.

These live assertions are separate integration evidence. They do not change
the four-row behavioral denominator.

### Remaining GPU-invisible validation

The owning registry must add the next stable task, VLLM-15
`(Completeness; P1; M)`, for one equivalent in-process skeleton smoke on a
host where CUDA platform selection is genuinely unavailable and no GPU is
discoverable. Masking a host that still exposes a GPU does not close that
task. The task must require the worker, runner, token, record-count, and schema
assertions above.
