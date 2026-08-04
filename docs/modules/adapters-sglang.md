# simllm.adapters.sglang

SGLang frontend adapter, pinned to SGLang main commit **8f2a3ad**
(2026-08-04; SGLang moves fast, so the pin is a commit, not a release). The
seam is the TP worker, installed without a fork through SGLang's plugin
framework.

## Interface

Simulated execution (`simllm/adapters/sglang/worker.py`):

```
SIMLLM_SGLANG_ENABLE=1 SIMLLM_SGLANG_MODE=virtual SIMLLM_SGLANG_GPU=b100 \
SIMLLM_SGLANG_STEP_RECORDS=/data3/yifeng/simllm/steps.jsonl \
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B \
    --disable-overlap-schedule --max-total-tokens 32768
```

Selection: simllm declares a `sglang.srt.plugins` entry point
(pyproject.toml). SGLang runs it in `run_scheduler_process` before the
`Scheduler` is constructed, and it applies a `REPLACE` hook on
`Scheduler.init_tp_model_worker`, the exact construction point where SGLang
swaps in its own MLX worker. Two gates keep it inert by default: the plugin
is a no-op unless `SIMLLM_SGLANG_ENABLE=1` (SGLang loads every discovered
plugin when its `SGLANG_PLUGINS` allowlist is unset, so an installed simllm
must never hijack a real run), and `SGLANG_PLUGINS=simllm` works as an
additional opt-in. In-process drivers call
`simllm.adapters.sglang.install()` instead.

`SimTpModelWorker` subclasses SGLang's `TpModelWorker` (all scheduler
bookkeeping inherited, the MLX-worker pattern) and

- replaces the model runner with `SimModelRunnerStub`, which loads no
  weights and allocates no KV tensors: `ModelRunner.__init__` still runs
  for real (device selection, `init_torch_distributed`, the
  `forward_stream` the scheduler adopts), while `initialize` fabricates
  CPU-resident `ReqToTokenPool` / `TokenToKVPoolAllocator` pools around a
  bufferless KV cache. Pool sizing honors `--max-total-tokens` (falling
  back to the model context length) and `--max-running-requests`, so KV
  pressure, radix eviction and retraction respond exactly as in
  production. Unlike SGLang's own MLX stub it also sets
  `full_max_total_num_tokens` / `swa_max_total_num_tokens` (the base
  worker's `get_tokens_per_layer_info` reads them unconditionally);
- fabricates one `GenerationBatchResult` per `forward_batch_generation`:
  `LogitsProcessorOutput(next_token_logits=None)` plus `next_token_ids` as
  an int64 tensor of one mid-vocabulary token per request, on the batch
  device. Both halves are load-bearing at the pinned commit: the
  scheduler's FutureMap relay checks `isinstance(..., torch.Tensor)` and
  writes into a scheduler-device buffer without a device cast, so a list
  or a wrong-device tensor corrupts the next decode step silently;
- translates each batch into a `simllm.core.StepRecord` via a pure
  observation layer (`observe_schedule_batch` reads `reqs`,
  `forward_mode`, `extend_lens`, `seq_lens_cpu`, `decoding_reqs` and
  `Req.cached_tokens` with getattr, so stubs exercise it without SGLang).
  Phases: extend rows are PREFILL, decode batches and the mixed-batch rows
  listed in `decoding_reqs` are DECODE (their `prefix_lens` entries are
  synthetic and never read as radix hits). The admission-time radix hit
  (`Req.cached_tokens`, final by forward time) is reported once, on the
  request's first prefill record; a decode row never consumes that
  one-time report, so a retracted request that resumes as a prefill still
  reports its hit. Step latency comes from the shared
  `simllm.compute` roofline (`ModelDims` built by
  `model_dims_from_sglang`, geometry fallbacks warned and stamped on
  `defaulted_fields`); a sink that returns a `StepResult` overrides the
  estimate, and records stream to the `SIMLLM_SGLANG_STEP_RECORDS` JSONL
  as each step completes (schema-tagged, shared with the vLLM adapter);
- refuses what the fabricated token would silently corrupt: a batch with
  `return_logprob` raises (SGLang dereferences
  `logits_output.next_token_logprobs` unguarded at the pinned commit, so a
  fabricated `None` would crash mid-batch anyway, only later and less
  clearly). Speculative decoding never reaches this worker: the scheduler
  only routes `spec_algorithm.is_none()` runs through the plain TP worker.

Timing has the same two modes as the vLLM adapter: `paced` (sleep the
simulated latency) and `virtual` (return immediately). Objects (a provider,
a host model, a sink) go through `configure()`, which reaches the worker
when the scheduler runs in this process.

Completion visibility: the worker never sees finish decisions (EOS and
`max_new_tokens` are applied in `process_batch_result` after the forward
returns) and there is no drain step at this seam, so
`finished_request_ids` is always empty and a record consumer infers a
request's completion from the last record that schedules it, the same
convention as the vLLM adapter's in-process path.

RadixCache prefix matching, eviction and the token/request pool accounting
are scheduler-side index bookkeeping and stay real, so radix hit rates and
vRAM pressure respond to the workload exactly as in production.

## Status

The pure surfaces (batch observation for extend/decode/mixed/idle, the
report-once radix-hit translation, the SGLang geometry reader, the config)
are unit-tested without importing SGLang in `tests/test_adapters_sglang.py`.
The worker, runner stub and plugin hook are validated by a live end-to-end
run: on 2026-08-04 a real SGLang at the pinned commit
(`/data3/yifeng/simllm-dev/venv-sglang`, editable install of the fresh
clone, `SGLANG_BUILD_RUST_EXTS=none`) ran the offline `Engine` on the CPU
engine (`device="cpu"`, torch_native attention selection, gloo process
groups) with the plugin active via its entry point: the scheduler
subprocess constructed `SimTpModelWorker`, three requests generated 8
fabricated tokens each, and the streamed JSONL held 9 schema-tagged records
with exactly 3 prefill and 21 decode rows and monotonic virtual time.
RadixCache ran live (0 hits, correctly: first-contact prompts shorter than
any reusable prefix). The overlap path is out of scope for the first
iteration: run with `--disable-overlap-schedule` (nothing forces overlap
on, and PP asserts it off anyway).

## Open tasks

- SGL-3: RadixCache-aware studies: prefix-hit rate and re-prefill traffic
  vs shared-prefix workload structure.
- SGL-4 (remaining half): a paced-mode run checked against SGLang's own
  wall-clock metrics, a workload that actually exercises radix hits
  (repeated shared prefixes) and retraction under KV pressure, and a
  `launch_server` (HTTP) run in addition to the offline `Engine` smoke.
- SGL-5: logprobs, speculative decoding and the dLLM/hybrid modes are
  refused or unreachable rather than fabricated.
- SGL-6: overlap-schedule support (the scheduler-side dual-stream loop with
  its result queue; needs delayed-sample semantics in the fabricated
  result).
- SGL-7: mamba/hybrid-attention models need the auxiliary-state pool the
  stub does not build; the stub currently builds a plain `ReqToTokenPool`
  only.

Closed this milestone: SGL-1 (the worker, this module). SGL-2 (upstream
worker-class selection flag) closed as moot 2026-08-04: SGLang's plugin
framework (`sglang.srt.plugins` entry points plus `HookRegistry` `REPLACE`
hooks, run before scheduler construction) is a supported non-fork selection
seam, so no upstream flag is needed.
