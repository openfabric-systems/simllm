# simllm.adapters.vllm

vLLM frontend adapter, pinned to **vLLM v0.26.0**. No fork required: the v1
engine resolves its executor class from a dotted import path, and injects an
arbitrary worker-extension class for the capture side.

## Interface

Simulated execution (`simllm/adapters/vllm/executor.py`):

```
SIMLLM_VLLM_MODE=virtual SIMLLM_VLLM_GPU=b100 \
SIMLLM_VLLM_STEP_RECORDS=/data3/yifeng/simllm/steps.jsonl \
vllm serve meta-llama/Llama-3.1-8B \
    --distributed-executor-backend simllm.adapters.vllm.SimExecutor \
    --num-gpu-blocks-override 8192
```

`SimExecutor` subclasses `vllm.v1.executor.abstract.Executor` and implements
the three abstract methods (`_init_executor`, `collective_rpc`,
`check_health`). It

- services every init-path RPC in order (`get_kv_cache_spec`,
  `determine_available_memory`, `update_max_model_len`,
  `initialize_from_config`, `compile_or_warm_up_model`,
  `get_kv_connector_handshake_metadata`, `get_supported_tasks`) with
  model-derived values, plus the per-step and control RPCs (`execute_model`,
  `sample_tokens`, `take_draft_token_ids`, `execute_dummy_batch`,
  `profile(is_start, profile_prefix)`, `reset_encoder_cache`,
  `reset_mm_cache`, LoRA and sleep/wake calls). An unknown method name
  returns `None` per worker and is counted in `unhandled_rpcs` instead of
  raising, because vLLM calls optional RPCs a simulated executor has nothing
  to say about;
- reports one `FullAttentionSpec` per layer the rank owns, taking the
  pipeline split from vLLM's own `get_pp_indices`, and a fixed
  `determine_available_memory`, so `--num-gpu-blocks-override` pins the KV
  pool to an exact block count (v0.26.0 back-propagates the override into
  available memory before auto-fit);
- returns `CompilationTimes(0.0, 0.0)` per worker from
  `compile_or_warm_up_model` (a list of `None` crashes the engine's
  `max(t.language_model ...)` reduction) and reads `cache_config.num_gpu_blocks`
  instead of the removed `initialize_cache` RPC;
- fabricates `ModelRunnerOutput(req_ids, req_id_to_index, sampled_token_ids)`
  per step: one fake mid-vocabulary token for every request whose prompt is
  complete this step, an empty list for a request still mid-prefill.
  `execute_model` returns an already-completed `Future` when `non_block=True`
  (`EngineCore.step()` always calls it that way and immediately reads
  `.result()`); `sample_tokens` is served defensively (it raises if no
  output is pending) but the engine never takes it in supported configs;
- keeps `supports_async_scheduling()` False, which is what makes vLLM's
  config post-init auto-disable async scheduling, and `supports_pp` False:
  the PP > 1 batch-queue loop interleaves `execute_model` and
  `sample_tokens` across in-flight steps, which needs a pending-output FIFO
  the executor does not have yet (VLLM-10). The CLI dotted-path spelling
  could never reach PP anyway: vLLM reads `supports_pp` off the string
  before resolving it, so `--pipeline-parallel-size > 1` fails in
  `EngineArgs` regardless;
- refuses configurations the fabricated token would silently corrupt:
  speculative decoding raises at construction (every draft would be
  rejected, i.e. an unstated 0% acceptance rate) and structured output
  raises at the first step that schedules one (the grammar would reject the
  fabricated id and kill every such request at its first token). Both are
  VLLM-8;
- translates each step into a `simllm.core.StepRecord` (phase, new tokens,
  prefix-cache hit at admission, context length), hands it to an injected
  sink, and accumulates it on `step_records` for the offline GOAL emission
  (VLLM-9). An empty-batch step that carries completions is recorded as a
  zero-cost drain record rather than dropped: under the `EngineCore` busy
  loop (`vllm serve`) the scheduler stays live while its finished set is
  non-empty, so the last requests' completions arrive on exactly such a
  step. The in-process `LLM.generate` loop stops stepping before that
  drain step and fires no teardown RPC (confirmed empirically on v0.26.0),
  so on that path the final completions are never reported by vLLM at all;
  a consumer infers a request's completion from the last record that
  schedules it. Attribution follows vLLM's own reporting:
  `finished_request_ids` on record N completed during step N-1 (the
  scheduler rebinds its finished set after constructing the step), while
  `preempted_request_ids` is same-step; consumers join a finished id to the
  preceding record's virtual time. Step latency comes from a
  `ComputeProvider` (default `RooflineProvider` over a per-rank
  `ModelDims`) plus a `HostInitiationModel`; a sink that returns a
  `StepResult` overrides the estimate, which is the closed-loop seam. The
  roofline sizes weights, KV cache and activations independently:
  `--kv-cache-dtype fp8` halves the KV read that dominates decode, a
  quantized checkpoint narrows the weight read (heuristic on the
  quantization method name), and any geometry field that fell back to its
  Llama-8B-shaped default is warned once and stamped on
  `ModelDims.defaulted_fields`.

Timing has two modes: `paced` (sleep the simulated latency, stock vLLM
metrics stay meaningful) and `virtual` (return immediately, report sim-native
metrics). Configuration that no vLLM flag carries comes from `SIMLLM_VLLM_*`
environment variables (`MODE`, `KV_MEMORY_BYTES`, `GPU`, `PEAK_FLOPS`,
`MEM_BANDWIDTH`, `EFFICIENCY`, `HOST_INIT_PS`, `TOKEN_ID`, `STEP_RECORDS`),
documented in the executor module docstring. Objects (a provider, a host
model, a sink) go through `configure()`, which reaches the executor only when
the engine core runs in the same process (`LLM(...)`, or
`VLLM_ENABLE_V1_MULTIPROCESSING=0`).

Placement capture (`simllm/adapters/vllm/worker_ext.py`), used on *real* runs:

```
vllm serve <model> -tp 8 \
    --worker-extension-cls simllm.adapters.vllm.PlacementExporter

entries = llm.collective_rpc("simllm_placement_entry")
manifest_from_worker_entries(entries).save("placement.json")
```

`PlacementExporter` exposes exactly one non-dunder name, because vLLM asserts
that no attribute of the extension class collides with the worker class. Each
rank returns a plain dict matching `simllm.placement.RankPlacement`: hostname,
local rank, GPU UUID and PCI bus id, the actual `GroupCoordinator.ranks` lists
for tp/pp/dp/ep/pcp/dcp/eplb (dcp matters: decode context parallelism
genuinely shards the KV cache across those ranks), the model's own
`start_layer`/`end_layer`, per-MoE-layer local expert ids from the
`expert_map`, and the EPLB `placement_epoch`.
Discovery is getattr-based throughout and never raises: a renamed internal
costs one optional field, it does not fail the capture run.

The v1 scheduler, KV-cache manager, block pool and prefix hashing are CPU-side
bookkeeping in the scheduler process and run unmodified.

## Status

The pure surfaces (step translation, drain records, output fabrication, the
roofline cost model with independent weight/KV dtype sizing, record
serialization, manifest assembly, the discovery helpers) are unit-tested
without importing vLLM in `tests/test_adapters_vllm.py`. The executor class
itself, its RPC table and the streaming JSONL dump are exercised by a real
end-to-end run, not by unit tests (VLLM-5 tracks the CI stand-in harness):
on 2026-08-04 a live vLLM v0.26.0 (`/data3/yifeng/simllm-dev/venv-vllm`,
in-process `LLM(...)` with `VLLM_ENABLE_V1_MULTIPROCESSING=0`) drove
`SimExecutor` in virtual mode with granite-3.0-1b-a400m-instruct: engine
init served every init RPC, `num_gpu_blocks_override=2048` pinned the KV
pool, 8 steps produced 24 scheduled entries and 35 new tokens, and the step
records streamed to the configured JSONL. That run is also what proved the
incremental dump necessary: vLLM never routed the in-process teardown
through the shutdown RPC, so each record is appended the moment its step
completes and `shutdown` only logs. The record JSON is the schema-tagged
form from `simllm.core.step` (`atlahs-closed-loop-step-v1`), shared with
the closed-loop wire format by construction.

The closed-loop seam is validated live as of the M4 first slice
(examples/m4/RESULTS.md): the same in-process pattern with
`tensor_parallel_size=8` was accepted on a single-GPU box (the executor
fabricates 8 workers and touches no device; v0.26.0's config validation
raised no device-count objection and the engine served all 8 workers'
init RPCs), and `configure(step_sink=HtsimStepSink(...))` put the
packet-level simulator inside the live step loop: 8 steps whose simulated
latencies each match the pre-registered closed form to 0 ps, with
sim-native TTFT/TPOT reported off the virtual clock. The recorded smoke
JSONL also replays through the same sink offline
(`simllm.core.step_records_from_jsonl`), reproducing the live latencies
row-for-row.

## Open tasks

- VLLM-3: sim-native metrics export via a `vllm.stat_logger_plugins` stat
  logger for virtual-time runs.
- VLLM-4 (remaining half): a paced-mode run whose TTFT/TPOT are compared
  with a real capture, a `vllm serve` run confirming the drain record lands
  under the `EngineCore` busy loop (source-verified only; the in-process
  loop is confirmed to never issue it), and the scheduler-side invariants
  under the fabricated executor: prefix-cache hit accounting with shared
  prefixes longer than one KV block (the 2026-08-04 smoke's shared prefix
  was shorter than the 16-token block, so hits were legitimately zero), and
  preemption behavior under KV pressure. Run only after the calibrated
  compute table and CORE-3/4/5 are ready. Use the identical vLLM commit,
  model, parallel configuration, request trace, seed and warm-up policy in
  simulation and silicon. Stage the comparison as single-GPU compute,
  eight-GPU intra-node, two-node rail-RNIC, offered-load sweep, KV pressure,
  chunked prefill/preemption, and mixed/bursty arrivals. Report p50, p90, p99
  and p99.9 TTFT/TPOT plus request queue, KV wait, kernel, collective, DMA,
  WQE/NIC, flow-completion and control-delay components. Calibrate only the
  early stages; hold out later stages, and choose the next accuracy task from
  their attributed residuals rather than aggregate error alone.
- VLLM-5: CI harness with transcribed stand-ins for the vLLM types
  (`Executor`, `ModelRunnerOutput`, `FullAttentionSpec`, `CompilationTimes`)
  so the init-RPC sequence and the step loop run end to end without a GPU
  stack installed.
- VLLM-6 (rescoped after the M5 slice landed MoE `ModelDims` and
  `step_moe_alltoalls` in the shared modules): the adapter half remains,
  `model_dims_from_vllm_config` does not yet read the MoE geometry
  (num_experts, top_k, per-expert intermediate size, local experts) off a
  vLLM MoE config, and `SimExecutor` passes no `ep_ranks` to a sink.
- VLLM-7 (placement half closed with M4): the placement-side builder
  exists, `simllm.placement.declared_manifest` computes a
  `source="declared"` manifest from tp/pp/dp in the DP x PP x TP layout
  order, and the M4 closed-loop runs drive the sink off it. Remaining
  half: `SimExecutor` deriving that declared manifest from its own
  `ParallelConfig` automatically (today the caller constructs it by hand
  and must keep the sizes in sync with the vLLM flags).
- VLLM-8: generate-only today. Speculative decoding and structured output
  are refused with explicit errors (fabricated tokens would silently model
  0% draft acceptance or first-token grammar deaths); pooling models
  (`pooler_output`) and encoder/multimodal inputs are serviced with empty
  or `None` answers rather than fabricated outputs.
- VLLM-9: render the accumulated `step_records` into a
  `simllm.core.GoalTrace` (the offline open-loop mode's second half; the
  records already carry phases, token counts and completions).
- VLLM-10: pipeline parallelism. Needs a pending-output FIFO so the
  batch-queue loop's interleaved `execute_model`/`sample_tokens` pairs map
  to the right steps, plus per-stage step accounting; until then
  `supports_pp` stays False and vLLM rejects PP > 1 up front.
- VLLM-11: observe the real vLLM KV manager and block-pool lifecycle for
  CORE-3 without replacing its policy. Emit stable pool/block/request IDs,
  token intervals, layer/dtype/bytes, allocation epoch, reference count,
  cause and correlation ID for reserve/allocation, prefix binding/touch,
  reads/writes, release/free, eviction, swap/transfer and preemption-driven
  recompute. Do not reconstruct allocation or eviction decisions from token
  deltas when the framework can report the actual event.
- VLLM-12: capture and replay the supported model runner's device schedule as
  an `ExecutionGraph` template keyed with the same identity envelope as the
  compute profile. Preserve CUDA stream order, event waits, kernel launches,
  NCCL launch/chunk boundaries and synchronous/asynchronous completion points.
  The simulated executor binds step shapes and framework KV events to this
  template; it does not invent concurrency from aggregate phase timings.
