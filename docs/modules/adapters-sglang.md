# simllm.adapters.sglang

SGLang frontend adapter, pinned to SGLang main commit **8f2a3ad**
(2026-08-04; SGLang moves fast, so the pin is a commit, not a release). The
seam is the TP worker, installed without a fork through SGLang's plugin
framework.

## Interface

Simulated execution (`simllm/adapters/sglang/worker.py`):

```
SIMLLM_SGLANG_ENABLE=1 SIMLLM_SGLANG_MODE=virtual SIMLLM_SGLANG_GPU=b100 \
SIMLLM_SGLANG_STEP_RECORDS="$SIMLLM_RUN_DIR/steps.jsonl" \
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

Simulated communication (`simllm/adapters/sglang/communicator.py`) is a
separate opt-in. `SIMLLM_SGLANG_COMMUNICATOR_TP_SIZE` binds one logical TP
group to `SimModelRunnerStub`; `SIMLLM_SGLANG_COMMUNICATOR_EVENTS` optionally
streams its immutable events from the scheduler process. The public surface
mirrors the pinned SGLang `all_reduce`, `all_gather`, `broadcast`, `send`, and
`recv` signatures, including SGLang's caller-owned
`all_gather(..., output_tensor_list=...)` form. It subclasses the unchanged
VLLM-14 torch-optional base, so both adapters share shape values, the
historical event schema, `CollectiveWork`, and the COMP-15 compatibility
stack.

The event sidecar truncates its target on first append so stale events cannot
mix with a new run. Only one stream instance may reach first append for a
resolved path in one process; a second instance raises before truncation. This
is stricter than silent multi-writer append and preserves a single event-order
authority per sidecar.

Each nonempty simulated model step currently observes one fixed 4,096-byte TP
all-reduce before compute settlement. The observation and all nested stack
events read the step's starting virtual time without advancing it. No
communication service time, runtime projection, `CompletionEvent`, or
`StepResult` contribution exists in this slice. With the TP-size flag absent,
no group or event sidecar is created and `observe_tp_step` is the identity
bypass.

## Status

The pure surfaces (batch observation for extend/decode/mixed/idle, the
report-once radix-hit translation, the SGLang geometry reader, the config)
are unit-tested without importing SGLang in `tests/test_adapters_sglang.py`.
The worker, runner stub and plugin hook are validated by a live end-to-end
run: on 2026-08-04 a real SGLang at the pinned commit
(`$SIMLLM_SGLANG_PYTHON`, editable install of the pinned checkout,
`SGLANG_BUILD_RUST_EXTS=none`) ran the offline `Engine` on the CPU
engine (`device="cpu"`, torch_native attention selection, gloo process
groups) with the plugin active via its entry point: the scheduler
subprocess constructed `SimTpModelWorker`, three requests generated 8
fabricated tokens each, and the streamed JSONL held 9 schema-tagged records
with exactly 3 prefill and 21 decode rows and monotonic virtual time.
RadixCache ran live (0 hits, correctly: first-contact prompts shorter than
any reusable prefix). The overlap path is out of scope for the first
iteration: run with `--disable-overlap-schedule` (nothing forces overlap
on, and PP asserts it off anyway).

Joined pre-play token replay is not implemented in this adapter. It remains
the explicit PLAY-7 follow-up in [preplay.md](preplay.md#open-tasks), including
the fabricated-token identity off path and a real in-process smoke.

The recorded smoke JSONL is exercised against the closed-loop sink as of
the M4 first slice: all 9 records load through
`simllm.core.step_records_from_jsonl` and replay through
`simllm.backends.HtsimStepSink` behind a declared tp=8 manifest, with
monotonic virtual time and every step's simulated latency above the
compute-only estimate (examples/m4/RESULTS.md check E). The live
closed-loop run of that slice used the vLLM adapter; the SGLang worker's
sink seam is the same contract but has not driven htsim live yet (SGL-8).

The SGL-11 zero-time communicator slice is frozen by expectations-only commit
`b0c5b73` and reported in
`examples/sgl_communicator_v1/RESULTS.md`. On 2026-08-10 the import-free study
passed 4/4 shape instances, 2/2 payload-scaling instances, the literal 14-name
nested stack, singleton identity, and unchanged VLLM-14 parity. A paired
offline CPU-engine run at the pinned SGLang commit produced two model steps in
both configurations. The flag-off and enabled step JSONL files were
byte-identical; only the enabled run emitted the frozen two-event TP order,
with 14 nested stack events per call and timestamps equal to the corresponding
step starts.

Post-specified integration review added a tracked LF byte fixture under
`tests/fixtures/sglang`. A CI-runnable test drives `SglStepTranslator`,
`observe_tp_step`, and `StepRecordStream` on one shared clock in both flag
states; both streams must equal the fixture exactly. The pinned call-site
audit now derives every observed row from AST, and the correction supplement
in `examples/sgl_communicator_v1/RESULTS.md` identifies the actual
`output_tensor_list` callers without rewriting the frozen expectations file.

## Open tasks

- SGL-3: RadixCache-aware studies: prefix-hit rate and re-prefill traffic
  vs shared-prefix workload structure.
- SGL-4 (Precision; P1; L) (remaining half): a paced-mode run checked against
  SGLang's own wall-clock metrics, a workload that actually exercises radix
  hits
  (repeated shared prefixes) and retraction under KV pressure, and a
  `launch_server` (HTTP) run in addition to the offline `Engine` smoke. Run
  after the calibrated compute table and CORE-3/4/5, using the same commit,
  model, parallel configuration, request trace, seed and warm-up policy on
  silicon and in simulation. Stage single-GPU compute, eight-GPU intra-node,
  two-node rail-RNIC, offered-load, KV-pressure, chunked/retraction and
  mixed/bursty cases. Report p50 through p99.9 TTFT/TPOT and attributed queue,
  KV, kernel, collective, DMA, WQE/NIC, flow and control residuals. Calibrate
  early stages and reserve later stages as holdouts.
- SGL-5: logprobs, speculative decoding and the dLLM/hybrid modes are
  refused or unreachable rather than fabricated.
- SGL-6: overlap-schedule support (the scheduler-side dual-stream loop with
  its result queue; needs delayed-sample semantics in the fabricated
  result). Its observed host-side order and completion waits lower to graph
  dependencies; device overlap itself remains owned by CORE-4/TRAF-7.
- SGL-7: mamba/hybrid-attention models need the auxiliary-state pool the
  stub does not build; the stub currently builds a plain `ReqToTokenPool`
  only.
- SGL-8: a live closed-loop run with `HtsimStepSink` installed via
  `configure(step_sink=...)` on the CPU-engine smoke path, mirroring the
  vLLM tp=8 run of examples/m4 (the M4 slice covered this adapter by
  JSONL replay only).
- SGL-9: observe RadixCache, token-pool and request-pool lifecycle events for
  CORE-3 without replacing SGLang's policy. Emit stable pool/block/request
  IDs, token intervals, layer/dtype/bytes, epoch, reference count, cause and
  correlation ID for allocation, prefix bind/touch, reads/writes,
  release/free, eviction, transfer and retraction-driven recompute. The
  dedicated KV study compares these actual decisions with VLLM-11.
- SGL-10: capture and replay the supported model runner's CUDA stream/event,
  kernel and NCCL schedule as an `ExecutionGraph` template keyed by the same
  identity envelope as its compute table. Bind batch shapes, radix events and
  overlap-scheduler dependencies at runtime; never infer device concurrency
  from a single elapsed phase duration.
- SGL-11 (Completeness; P1; L) (remaining after the zero-time first slice):
  extend the SGLang-shaped mirror only as accepted adapter modes make further
  pinned `GroupCoordinator` calls reachable. In particular, DCP attention and
  MoE paths still invoke `all_gather_into_tensor`, `reduce_scatter_tensor`,
  `all_gatherv`, and `reduce_scatterv`; the current dense TP worker neither
  claims nor silently fabricates them. Freeze each supported call site's
  signature, shape contract, enabled behavior, and exact disabled baseline
  before adding it. The landed slice mirrors `all_reduce`, `all_gather`,
  `broadcast`, `send`, and `recv`, including SGLang's added
  `all_gather(..., output_tensor_list=None)` form, on the shared VLLM-14
  zero-time event base. SGL-13 owns runtime projection, SGL-14 owns native
  operation-specific lowerings, and SGL-15 preserves the real-call
  bottleneck-study clause.
- SGL-12 (Precision; P1; M): source and populate exact
  `StepRecord.num_sampled` at the worker seam. Distinguish a mid-prompt extend
  row from the extend step that reaches `origin_input_ids`, including radix
  hits, retracted prefills and MIXED batches; prove the count matches the rows
  for which SGLang consumes a generated token. Keep the absent field as the
  explicit compatibility path.
- SGL-13 (Completeness; P1; L): after CORE-4 and CORE-5 land, project each
  simulated SGLang communicator `CollectiveWork` through the single runtime
  authority into `CompletionEvent`, `StepResult`, and TTFT/TPOT. Freeze a
  fixed-workload signed metric relation and quantitative band first. The
  disabled projection must preserve every accepted SGLang worker timestamp,
  token, record byte, and completion order exactly.
- SGL-14 (Precision; P1; M): replace the zero-time
  `ncclAllReduce`-shaped compatibility call for SGLang all-gather, broadcast,
  send, and receive with native COMP stack entries when those entries exist.
  Identify the replacement with operation names, peer roles, payloads, shape
  results, and the SGLang-only output-list form. Remove the ring-layout
  servable-domain restriction while preserving the compatibility off path
  byte for byte and timestamp for timestamp.
- SGL-15 (Precision; P1; L): complete the SGLang half of the communicator
  bottleneck study after SGL-13 makes the call metric-live. The zero-time
  surrogate is the explicit baseline. Measure pinned-SGLang Python dispatch,
  custom-op routing, device-communicator selection, and synchronization stalls
  over a frozen payload, group-size, and call-mode matrix. Hold out at least
  one model and group size, require modeled median and p95 call cost within a
  pre-registered band, then verify the signed TTFT/TPOT effect and exact
  zero-cost bypass.

Closed this milestone: SGL-1 (the worker, this module). SGL-2 (upstream
worker-class selection flag) closed as moot 2026-08-04: SGLang's plugin
framework (`sglang.srt.plugins` entry points plus `HookRegistry` `REPLACE`
hooks, run before scheduler construction) is a supported non-fork selection
seam, so no upstream flag is needed.
