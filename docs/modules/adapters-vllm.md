# simllm.adapters.vllm

vLLM frontend adapter, pinned to **vLLM v0.26.0**. No fork required: the v1
engine resolves its executor class from a dotted import path, and injects an
arbitrary worker-extension class for the capture side. The engine also
resolves the worker class itself from a dotted path, which is the seam for
the flagged model-runner skeleton and the later coupling modes (VLLM-13).

## Interface

Simulated execution (`simllm/adapters/vllm/executor.py`):

```
SIMLLM_VLLM_MODE=virtual SIMLLM_VLLM_GPU=b100 \
SIMLLM_VLLM_STEP_RECORDS="${SIMLLM_DATA_ROOT:?configure SIMLLM_DATA_ROOT}/simllm/steps.jsonl" \
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
- returns `ModelRunnerOutput(req_ids, req_id_to_index, sampled_token_ids)` per
  step. Without a joined replay run, it uses the accepted fake
  mid-vocabulary token for every request whose prompt is complete and an empty
  list for a request still mid-prefill. With `SIMLLM_VLLM_REPLAY_RUN` set, it
  verifies the joined trace hash, requires an exact joined scheduler request
  ID and serves the oracle token selected by the scheduler-reported output
  index. Replay admission requires `max_tokens` to equal the joined oracle
  length and rejects an early EOS or stop token and a
  prompt-plus-oracle length beyond `max_model_len`. The complete replay batch
  validates before the sink, record stream or virtual clock changes;
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
  raises at the first step whose
  `SchedulerOutput.has_structured_output_requests` is true (the grammar would
  reject the fabricated id and kill every such request at its first token).
  The scheduler sets that signal before executor dispatch at
  `vllm/v1/core/sched/scheduler.py:1236-1259`; both refusals are VLLM-8;
- translates each step into a `simllm.core.StepRecord` (phase, new tokens,
  prefix-cache hit at admission, context length, and exact `num_sampled`),
  hands it to an injected sink, and accumulates it on `step_records` for the
  offline GOAL emission (VLLM-9). The exact count is the sum of the same
  `produces_token` flags used to fabricate `ModelRunnerOutput` rows, including
  zero for a mid-prompt chunk and a drain record. An empty-batch step that
  carries completions is recorded as a
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
`MEM_BANDWIDTH`, `EFFICIENCY`, `HOST_INIT_PS`, `TOKEN_ID`, `STEP_RECORDS`,
`REPLAY_RUN`), documented in the executor module docstring. Objects (a
provider, a host model, a sink) go through `configure()`, which reaches the
executor only when the engine core runs in the same process (`LLM(...)`, or
`VLLM_ENABLE_V1_MULTIPROCESSING=0`). Call `reset_configuration()` between
independent in-process engines to clear every accumulated hook. An explicit
constructor config takes priority over hooks and the environment.

Flagged worker-boundary skeleton
(`simllm/adapters/vllm/worker.py`):

```
SIMLLM_VLLM_WORKER_MODE=skeleton \
SIMLLM_VLLM_MODE=virtual \
VLLM_USE_V2_MODEL_RUNNER=0 \
vllm serve <model> \
    --no-async-scheduling \
    --worker-cls simllm.adapters.vllm.SimWorker
```

`SimWorker` is a real subclass of v0.26.0's GPU `Worker` when vLLM is
installed, selected through the same dotted worker-class seam the stock
executor uses. `WorkerWrapperBase.init_worker` loads general plugins, accepts
only a string, resolves it, and rejects a class object
(`vllm/v1/worker/worker_base.py:245-259,317-320`). Construction requires the
exact high-level flag `SIMLLM_VLLM_WORKER_MODE=skeleton`; an absent, empty, or
different value raises before the stock worker can initialize a device.

In skeleton mode, the override of `init_device` does not call the stock body.
It leaves `device` unset and constructs `SimModelRunner`, while the stock body
would select and construct either its hardcoded V2 or V1 runner at the end of
device initialization (`vllm/v1/worker/gpu_worker.py:297-416`; there is no
model-runner class parameter). This first copied path mirrors the V1 runner
algorithm, so live validation pins `VLLM_USE_V2_MODEL_RUNNER=0`; respecting
both upstream runner variants belongs to the later GPU-present rebind mode.
A V2-selected configuration is rejected before stock worker construction.

This device-free slice also rejects Ray and external-launch executors. A
multiprocess worker must use `--no-async-scheduling`, because v0.26.0's async
output thread calls `current_platform.set_device(self.worker.device)` at
`vllm/v1/executor/multiproc_executor.py:968-980`. In-process execution with
`VLLM_ENABLE_V1_MULTIPROCESSING=0` may retain async scheduling because it does
not start that device-setting worker thread. Ray's compiled-DAG path likewise
requires a non-null worker device at `vllm/v1/executor/ray_utils.py:109-145`.
These combinations fail at construction with a direct remediation message.

The ordinary construction surface is mirrored in source order:
`init_device`, `load_model`, `get_kv_cache_spec`,
`determine_available_memory`, `initialize_from_config`,
`compile_or_warm_up_model`, `reset_mm_cache`, and `get_supported_tasks`.
The order comes from
`vllm/v1/executor/uniproc_executor.py:48-69`,
`vllm/v1/engine/core.py:243-324`,
`vllm/v1/executor/abstract.py:118-150`,
`vllm/v1/engine/llm_engine.py:123-142,205-210`, and
`vllm/entrypoints/llm.py:338-348`. Conditional and control methods are also
served: max-length update, KV handshake, multimodal and encoder cache resets,
dummy batch, profile, LoRA, sleep/wake, health, draft-token query, and
shutdown. Prefix-cache reset remains scheduler-only, matching
`vllm/v1/engine/core.py:779-784`.

The model runner keeps the selected V1 algorithm names and order from state
update through input/attention preparation, empty `_model_forward`, sampling,
bookkeeping, and EPLB update
(`vllm/v1/worker/gpu_model_runner.py:4111-4479,4497-4736`). As in the stock
path, nonempty `execute_model` returns `None` and the engine immediately calls
`sample_tokens` (`vllm/v1/engine/core.py:576-606`). Worker RPCs reach the
runner through `self.model_runner` in the stock source as well
(`vllm/v1/worker/gpu_worker.py:701-713,923-927,955-956,1080-1178`).

`SimExecutor` and `SimWorker` share the same model-derived KV specification,
configured available-memory answer, compilation-time answer, task answer,
conditional replay or token fabrication, translation, settlement, and
streaming helpers. The worker and runner share exactly one core
`VirtualClock`; empty model compute
has zero fallback latency, while a configured closed-loop sink can still
provide a nonzero `StepResult`. Only global rank zero is the mutable time and
stream authority, including when every process is locally marked as a driver.
Records use the unchanged
`atlahs-closed-loop-step-v1` JSONL path.

Simulated communication (`simllm/adapters/vllm/communicator.py`) is a separate
trimmed layer. `SimGroupCoordinator` mirrors the pinned v0.26.0 signatures for
`all_reduce`, `all_gather`, `broadcast`, `send`, and `recv`, plus `rank`,
`ranks`, `world_size`, `local_rank`, `rank_in_group`, and the six rank
navigation properties. Its constructor accepts resolved ranks and the
runner-owned `VirtualClock`; it never constructs a torch process group. The
module is torch-optional: it remains importable without torch, but `recv`
uses a guarded runtime torch import when the caller supplies a real
`torch.dtype`. The copied runner uses `ShapeTensor` for its deliberate
empty-computation calls.

Every successful boundary produces one immutable event with operation, group,
rank membership, payload bytes, virtual timestamp, semantic `CollectiveWork`,
and the nested COMP-15 events. A multi-rank call enters the landed
`ncclAllReduce`-shaped stack skeleton. A singleton emits the upper observation
but takes the exact identity path and emits no ring event. The V1 runner issues
one shape-only TP call during `_model_forward`; when DP size exceeds one it
first issues the pinned runner helper's `(4, dp_size)` int32 coordination
all-reduce. Both groups share one observer and clock, so their zero-time call
order stays deterministic.

The COMP-15 compatibility entry constrains the currently servable nonzero
payloads. Payload bytes must divide evenly over world size, channels and warps,
and each per-lane share must contain an integral, nonzero number of configured
chunks. An unservable call raises before consuming its operation ID. A
zero-byte call emits an upper event with
`stack_disposition="zero_payload_bypass"` and no nested stack event. VLLM-20
owns removal of this compatibility-domain restriction.

This first slice is observability only. It does not create a runtime authority,
emit a `CompletionEvent`, change a `StepResult`, or model communication time.
It therefore makes no TTFT or TPOT claim. VLLM-19, VLLM-20, and VLLM-21 own
the explicit residuals. CORE-4 and CORE-5 have landed, so runtime projection
is unblocked and is VLLM-19's remaining work.

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
from transcribed inputs in `tests/test_adapters_vllm.py`. The mirror tests use
those same inputs against both the no-vLLM stand-in and the installed real
v0.26.0 `Worker`; the test module never imports vLLM directly. The executor
class itself, its RPC table and the streaming JSONL dump are exercised by a
real end-to-end run, not by a complete unit stand-in (VLLM-5 tracks that CI
harness):
on 2026-08-04 a live vLLM v0.26.0 from a machine-local pinned environment,
whose resolved historical path is intentionally omitted, used in-process
`LLM(...)` with `VLLM_ENABLE_V1_MULTIPROCESSING=0` and drove
`SimExecutor` in virtual mode with granite-3.0-1b-a400m-instruct: engine
init served every init RPC, `num_gpu_blocks_override=2048` pinned the KV
pool, 8 steps produced 24 scheduled entries and 35 new tokens, and the step
records streamed to the configured JSONL. That run is also what proved the
incremental dump necessary: vLLM never routed the in-process teardown
through the shutdown RPC, so each record is appended the moment its step
completes and `shutdown` only logs. The record JSON is the schema-tagged
form from `simllm.core.step` (`atlahs-closed-loop-step-v1`), shared with
the closed-loop wire format by construction.

For a current reproduction, `SIMLLM_VLLM_PYTHON` selects the compatible vLLM
interpreter.

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

The VLLM-13 flagged skeleton first slice is implemented and live-reachable as
of 2026-08-10. The expectations were frozen in commit `582d3de` before code or
runs. The four-cell request-count by prompt-length study in
`examples/vllm_skeleton_v1/RESULTS.md` passes 4/4 exact-oracle rows and 4/4
behavioral relation instances. All mirrored calls, records, and results use
one injected nonzero virtual clock; the deliberate zero-compute and schema
checks pass as fatal unscored invariants.

Exactly one initial in-process vLLM v0.26.0 smoke used the cached Granite model,
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, offline Hugging Face mode, the dotted
`worker_cls`, and a 64-block logical KV pool. It reached `SimWorker`, completed
engine initialization in 0.00 seconds, generated one request with two output
tokens, and streamed two schema-tagged step records. The host was not actually
GPU-invisible: extension setup warned that no CUDA runtime was found, but
vLLM then identified a GTX 1660 Ti and selected its CUDA platform. There was
no pre-worker platform blocker, and no retry was made.

The review-triggered expectations were frozen in commit `17b7bd1` before the
fix implementation and review-round runs. The same mirror test file now
passes 37/37 tests without vLLM and all 35 applicable tests against the real
v0.26.0 worker, with two absence-only tests skipped in that environment. The
deterministic study uses literal call-sequence oracles rather than importing
implementation constants and still passes 4/4 rows and 4/4 behavioral
relation instances. The executor's documented VLLM-8 refusal now keys on the
real `SchedulerOutput.has_structured_output_requests` signal, and the phantom
worker `reset_prefix_cache` projection is removed.

Exactly one strengthened smoke ran in the review round. It reached
`SimWorker`, asserted that the runner was `SimModelRunner`, generated token id
`24577` twice to match the worker's fabricated id, and asserted exactly two
`atlahs-closed-loop-step-v1` JSONL records. The host still exposed the GTX
1660 Ti despite masking, so the genuinely GPU-invisible version of this
asserted smoke remained open at that boundary.

The VLLM-16 three-mechanism isolation study ran on 2026-08-11 from expectations
frozen in commit `25e79be`. The invalid UUID left NVML and five NVIDIA
character nodes visible. A bubblewrap device namespace achieved genuine GPU
invisibility, but this CUDA-tagged vLLM package selected
`UnspecifiedPlatform` and failed device configuration before resolving
`SimWorker`. A forced `CpuPlatform` reached the skeleton and passed every
runner, token, record-count and schema assertion, but NVML and the physical
device nodes remained visible. No original row combined both requirements, so
that diagnostic headline remains 0/3 with genuine-risk fraction 3/3.

VLLM-16 then closed in the post-specified fix round. Additive expectations
were frozen in commit `9b7f854` before implementation and the one combined
attempt. The unchanged device namespace plus the unchanged `CpuPlatform`
override passed 1/1 with genuine-risk fraction 1/1 in the same child. Before
import it had no NVIDIA entry; after the smoke, `nvidia-caps` was preserved as
a directory with mode `0755` and the character-device count remained zero.
NVML was unavailable, Torch reported zero CUDA devices and zero allocated
bytes, and the exact worker, runner, two-token and two-record smoke passed. A
CPU-tagged vLLM build is therefore not a necessary host requirement. See
[the VLLM-16 results](../../examples/vllm_skeleton_v1/vllm16_RESULTS.md).

PLAY-3 joined-token replay is implemented in `SimExecutor` and the flagged
skeleton as of 2026-08-10. Expectations were frozen in commit `edcb2b9`
before implementation or any replay run. A joined
`simllm-preplay-replay-run-v1` is selected with
`SIMLLM_VLLM_REPLAY_RUN`; construction verifies the named trace bytes, and
sampling maps each scheduler-reported output index to the exact oracle token.
The adapter accepts only an exact joined scheduler request ID. Live replay
sets vLLM's audited request-ID no-randomization mode, and a suffix-shaped
lookalike fails as unjoined. Unknown IDs, cursor gaps, exhaustion, early stop
channels, model-length overflow and admission lengths that differ from the
oracle all fail before settlement. `reset_configuration()` prevents replay
state from leaking into a later in-process engine.

The four-cell metric study served exact oracle sequences through both adapter
paths. A review-amendment study then submitted the same two requests to the
real in-process vLLM scheduler in baseline and replay modes. The scheduler
itself moved `r0` completion from step 3 to step 0; the engine-produced records
changed TTFT and TPOT by every frozen exact relation. The absent-replay path
has a tracked LF-locked JSONL pytest fixture. A final in-process vLLM v0.26.0
Granite smoke asserted external and internal identity `length-cap`, returned
token ID 38 and retained a zero-latency completion drain. Earlier live
attempts exposed internal-ID randomization and the offline wrapper's
integer-only output sort; both remain explicit in
[the PLAY-3 results](../../examples/preplay_adapter_replay_v1/RESULTS.md),
along with their post-specified regression status. Speculative decoding and
structured output remain refused. SGLang replay is not implied by this status
and remains PLAY-7 in [preplay.md](preplay.md#open-tasks).
The VLLM-14 zero-time coordinator slice is implemented as of 2026-08-10. Its
expectations-only commit is `29221e4`, which precedes implementation and every
result-producing target run. The import-free study in
`examples/vllm_group_coordinator_v1/RESULTS.md` passes all 4 shape cells and
both payload-scaling instances. The fixed 4,096-byte all-reduce emits one
coordinator event, 14 nested COMP-15 events, and the frozen 17-event full stack
including communicator setup. Singleton identity and the accepted VLLM-13
step/token/clock baseline both pass as fatal unscored guards.

The scored in-process vLLM v0.26.0 smoke reached `SimWorker` and
`SimModelRunner` without a vLLM fork. Two model steps emitted the frozen
coordinator order `DP, TP, DP, TP` with payloads `64, 4096, 64, 4096` bytes and
nested stack counts `32, 14, 32, 14`. The request returned token id `24577`
twice and retained exactly two `atlahs-closed-loop-step-v1` records. As in the
earlier skeleton smoke, this host exposed a GTX 1660 Ti despite
`CUDA_VISIBLE_DEVICES=`, so the run is external-runtime seam evidence but not
GPU-invisible-host evidence.

VLLM-15 is complete. Expectations were frozen at commit `25d098c` before the
implementation and runs. The
[latent-knob study](../../examples/step_sink_latent_knobs/RESULTS.md) covers
mid-prompt and prompt-completing chunks, prefix-cache completion, decode, and
attach-mid-flight translation. Every record count equals the nonempty
fabricated output rows. The adapter-produced mixed batch reduces live fluid
TTFT by the frozen 32,000 ps relative to the absent-field bypass, and a real
vLLM v0.26.0 chunked-prefill smoke emits sample counts `(0, 1, 1)` for
scheduled-token counts `(2, 1, 1)`. Manually constructed records may still
omit the optional field; v1 readers and that compatibility path are unchanged.

## Open tasks

- VLLM-3: sim-native metrics export via a `vllm.stat_logger_plugins` stat
  logger for virtual-time runs.
- VLLM-4 (Precision; P1; L) (remaining half): a paced-mode run whose TTFT/TPOT
  are compared with a real capture, a `vllm serve` run confirming the drain
  record lands under the `EngineCore` busy loop (source-verified only; the
  in-process loop is confirmed to never issue it), and the scheduler-side
  invariants
  under the fabricated executor: prefix-cache hit accounting with shared
  prefixes longer than one KV block (the 2026-08-04 smoke's shared prefix
  was shorter than the 16-token block, so hits were legitimately zero), and
  preemption behavior under KV pressure. Run only after the calibrated
  compute table and CORE-3 are ready, since CORE-4 and CORE-5 have landed.
  Use the identical vLLM commit,
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
- VLLM-13 (Completeness; P1; L) (remaining GPU-present half after the flagged
  skeleton): the skeleton DP coordination half has landed through
  `SimGroupCoordinator`, including consumption of its local padded-token
  projection into `StepRecord.num_tokens_after_padding`. Add the GPU-present
  mode that runs stock `Worker.init_device`, preserves its
  distributed groups and memory snapshot, respects the upstream V1/V2 runner
  selection, and then rebinds `self.model_runner`. Couple runner work to the
  simulated GPU service and NCCL path, including BACK-20 submission in
  GPU-initiated mode, then serve runner-internal DP coordination through those
  preserved real groups. Enable and validate device-free async multiprocessing,
  Ray, and external-launch execution, which this first slice rejects before
  their device or ownership assumptions can run. Every run must declare the
  CQ consumer and how completion reaches the model runner through BACK-20 and
  CORE-5. The executor-level `SimExecutor` and the gated skeleton remain
  supported without behavior changes. VLLM-12 device-schedule capture uses the
  same seam.
- VLLM-14 (Completeness; P1; L) (remaining after the zero-time first slice):
  the name-mirrored `SimGroupCoordinator`, shape-only results, rank-membership
  surface, boundary observations, `CollectiveWork` lowering, COMP-15 call, and
  copied-runner TP and DP calls have landed. Keep that narrow interface aligned
  with the pinned supported model paths and bind it into VLLM-13's later
  GPU-present runner mode without changing the skeleton or executor bypasses.
  Custom-allreduce, symmetric-memory fast paths, and off-main-path calls remain
  omitted or inert unless a supported study opts into them. SGL-11 remains the
  untouched SGLang half and should reuse this torch-optional shape/event base.
  This ID explicitly excludes runtime projection and every timing claim:
  VLLM-19, VLLM-20, and VLLM-21 own those residuals on the landed CORE-4 and
  CORE-5 runtime and reduction path.
- VLLM-19 (Completeness; P1; L): now that CORE-4 and CORE-5 have landed,
  project each
  coordinator `CollectiveWork` through the single runtime authority into
  `CompletionEvent`, `StepResult`, and TTFT/TPOT. The current component event is
  not metric-live and must not be timed in parallel with another authority.
  Freeze a fixed-workload signed TTFT/TPOT relation and quantitative band before
  implementation. The disabled projection must preserve every accepted
  VLLM-13 timestamp, token, record, and completion order exactly.
- VLLM-20 (Precision; P1; M): replace the current `ncclAllReduce`-shaped
  compatibility lowering for `all_gather`, `broadcast`, `send`, and `recv`
  with native COMP stack entries when those entries exist. The surrogate is an
  all-reduce-shaped zero-time trace; the identifying observables are the
  operation-specific stack names, peer roles, byte counts, and shape results.
  Remove the current ring-layout servable-domain restriction: native entries
  must represent zero payloads explicitly and accept operation-legal nonzero
  byte counts without requiring even all-reduce lane or chunk division.
  Acceptance requires exact semantic operation identity for every enabled call
  while the compatibility off path remains byte-for-byte and timestamp-for-
  timestamp identical to this slice.
- VLLM-21 (Precision; P1; L): calibrate real coordinator dispatch cost after
  VLLM-19 makes it metric-live. The current surrogate is exactly zero dispatch
  time. Measure pinned-vLLM Python dispatch, custom-op indirection, and
  synchronization stalls over a frozen payload, group-size, and call-mode
  matrix. Hold out at least one model and group size; require modeled median
  and p95 call cost within a pre-registered relative or additive band, then
  verify the signed TTFT/TPOT effect and the exact zero-cost bypass baseline.
