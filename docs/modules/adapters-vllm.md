# simllm.adapters.vllm

vLLM frontend adapter, pinned to **vLLM v0.27.1**. No fork required: the v1
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
  pool to an exact block count (v0.27.1 back-propagates the override into
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
  prefix-cache hit at admission, context length, exact `num_sampled`, and the
  exact sampled request identities when
  `SIMLLM_VLLM_SAMPLED_REQUEST_IDS=1`),
  hands it to an injected sink, and accumulates it on `step_records` for the
  offline GOAL emission (VLLM-9). The exact count is the sum of the same
  `produces_token` flags used to fabricate `ModelRunnerOutput` rows, including
  zero for a mid-prompt chunk and a drain record. An empty-batch step that
  carries completions is recorded as a
  zero-cost drain record rather than dropped: under the `EngineCore` busy
  loop (`vllm serve`) the scheduler stays live while its finished set is
  non-empty, so the last requests' completions arrive on exactly such a
  step. The in-process `LLM.generate` loop stops stepping before that
  drain step and fires no teardown RPC (confirmed empirically on v0.27.1),
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

When `SIMLLM_VLLM_NATIVE_STEPS` names a path, the adapter appends the native
ordered request IDs, per-request scheduled tokens, native total, preempted IDs
and finished IDs beside the independently translated `StepRecord`. This
capture and exact sampled identities are both default-off, so the accepted
count-only step streams retain their bytes unless a harness selects the new
surface.

`model_dims_from_vllm_config` reads the MoE geometry off vLLM's own resolved
MoE parallel shape rather than off the raw flags. `expert_parallel_geometry`
resolves that shape for the config's global rank (the flattened
`dp * pcp * tp` device set, whether expert parallelism is actually in use, the
resulting MoE tensor-parallel size, and the rank's expert group and index), and
`expert_group_ranks` returns the group itself, or `None` for a dense model
because vLLM builds no expert-parallel group for one. When expert parallelism
is in use, `SimExecutor` binds that group once, before any step, to a sink
implementing `ExpertGroupStepSink.bind_expert_group`; with it disabled the
executor binds nothing and the sink keeps the group its own configuration
declared.

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

Disaggregated session driver (`simllm/adapters/vllm/pd_session.py`):

- `VllmDisaggregatedSession` constructs separate in-process prefill and decode
  engines with pool-local adapter configuration, calls
  `reset_configuration()` between constructions, and injects one shared
  virtual clock into every engine. The session configuration may select one
  provider per role while retaining its original shared provider as the exact
  default. A provider that publishes pricing provenance is projected into the
  request result; a provider with no provenance leaves that member absent.
- The pinned vLLM v0.27.1 scheduler-side KV connector is the real control
  seam. It gates producer completion and consumer external-token admission,
  while an explicit core KV-handoff event is the sole transfer-time authority.
  Pool-local request IDs stay distinct and one stable session request ID is
  carried in connector metadata. Simulated workers have no paged KV tensors,
  so worker tensor transfer is explicitly false rather than fabricated.
- Each engine's scheduler remains its only batching authority. The delivered
  concurrent path admits several stable session requests, releases each decode
  consumer from its own completed producer handoff and records the pool-local
  scheduler batches without assembling them in the driver. Exact deployment
  curve records reduce completed requests to aggregate output throughput and
  per-token request delay.

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

`SimWorker` is a real subclass of v0.27.1's GPU `Worker` when vLLM is
installed, selected through the same dotted worker-class seam the stock
executor uses. `WorkerWrapperBase.init_worker` loads general plugins, accepts
only a string, resolves it, and rejects a class object
(`vllm/v1/worker/worker_base.py:230-259`). Construction requires the
exact high-level flag `SIMLLM_VLLM_WORKER_MODE=skeleton`; an absent, empty, or
different value raises before the stock worker can initialize a device.

In skeleton mode, the override of `init_device` does not call the stock body.
It leaves `device` unset and constructs `SimModelRunner`, while the stock body
would select and construct either its hardcoded V2 or V1 runner at the end of
device initialization (`vllm/v1/worker/gpu_worker.py:304-426`; there is no
model-runner class parameter). This first copied path mirrors the V1 runner
algorithm, so live validation pins `VLLM_USE_V2_MODEL_RUNNER=0`; respecting
both upstream runner variants belongs to the later GPU-present rebind mode.
A V2-selected configuration is rejected before stock worker construction.

This device-free slice also rejects Ray and external-launch executors. A
multiprocess worker must use `--no-async-scheduling`, preserving the qualified
device-free contract. v0.27.1's async output thread now guards its
`current_platform.set_device(self.worker.device)` call with `hasattr` at
`vllm/v1/executor/multiproc_executor.py:974-996`, but that source change alone
does not qualify asynchronous multiprocess execution. In-process execution with
`VLLM_ENABLE_V1_MULTIPROCESSING=0` may retain async scheduling because it does
not start that device-setting worker thread. Ray's compiled-DAG path likewise
requires a non-null worker device at `vllm/v1/executor/ray_utils.py:105-123`.
These combinations fail at construction with a direct remediation message.

The ordinary construction surface is mirrored in source order:
`init_device`, `load_model`, `get_kv_cache_spec`,
`determine_available_memory`, `initialize_from_config`,
`compile_or_warm_up_model`, `reset_mm_cache`, and `get_supported_tasks`.
The order comes from
`vllm/v1/executor/uniproc_executor.py:48-69`,
`vllm/v1/engine/core.py:257-332`,
`vllm/v1/executor/abstract.py:118-150`,
`vllm/v1/engine/llm_engine.py:123-142,205-210`, and
`vllm/entrypoints/llm.py:348`. Conditional and control methods are also
served: max-length update, KV handshake, multimodal and encoder cache resets,
dummy batch, profile, LoRA, sleep/wake, health, draft-token query, and
shutdown. Prefix-cache reset remains scheduler-only, matching
`vllm/v1/engine/core.py:787-790`.

The model runner keeps the selected V1 algorithm names and order from state
update through input/attention preparation, empty `_model_forward`, sampling,
bookkeeping, and EPLB update
(`vllm/v1/worker/gpu_model_runner.py:4166-4760`). As in the stock
path, nonempty `execute_model` returns `None` and the engine immediately calls
`sample_tokens` (`vllm/v1/engine/core.py:584-604`). Worker RPCs reach the
runner through `self.model_runner` in the stock source as well
(`vllm/v1/worker/gpu_worker.py:645-717,856-897,1012-1110`).

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
trimmed layer. `SimGroupCoordinator` mirrors the pinned v0.27.1 signatures for
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
v0.27.1 `Worker`; the test module never imports vLLM directly. The executor
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
structured output remain refused. SGLang replay is not implied by this status;
the SGLang adapter has since landed its own replay token source, whose live
in-process smoke is PLAY-16 in [preplay.md](preplay.md#open-tasks).
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

The TRAF-13 component slice added the observation-aware step-sink handoff on
2026-08-12. `_SimStepRuntime` binds its single virtual clock into an explicitly
observation-capable sink and passes that sink the translated `StepRecord` plus
optional `ExecutionObservations`; legacy one-argument sinks retain their exact
call contract. `DeviceRuntimeStepSink` owns observed lowering, the coarse
device runtime, completion reduction, and the returned `StepResult`. A focused
component fixture routed supplied observations through that chain, produced
200 completion events and request metrics for both scheduled requests, and
preserved the frozen serial graph and GOAL digests. At that commit these were
component and fatal-unscored identity results only.

The first registered source and component qualification stopped before every
behavioral relation, as predicted by the expectations-only commit `409b4ad`.
The four-layer skeleton invoked the observation-capable sink once with no
`ExecutionObservations` and exposed one fixed 4,096-byte `all_reduce`, zero of
the required 48 semantic Granite MoE sites, and no layer, logical-stream,
dependency, request, or completion-frontier fields. The audited active source
had not yet been translated. No placement or TTFT/TPOT row ran, so that
historical genuine-risk result remains `0/0, blocked before behavioral
execution`.

A separate pre-VLLM-22 diagnostic reached a real vLLM 0.26.0 `LLM` using the
Granite skeleton runner. Its prefill and decode steps each exposed one 64-byte
DP event and one fixed 4,096-byte TP event, still with no
`ExecutionObservations`, per-layer operation, or EP dispatch and combine site.
That historical component evidence does not change the earlier blocked
behavioral denominator.

The 2026-08-12 VLLM-22 qualification was void. The opt-in `granite-dbo`
producer at the `SimWorker` model-forward boundary translates each real vLLM
v0.26.0 scheduler step into a Granite per-layer schedule. It records source
submission order, per-rank compute and shared communication queues,
participant-local edges, microbatch request correlation and request-visible
completion frontiers. The producer derives no overlap percentage or edge from
the serial compatibility graph. Unsupported model families, TP or PP shapes,
explicit microbatch sizes, padded DBO and multi-token DBO splits fail
explicitly; VLLM-23 owns those optional shapes. The audited vLLM wrapper
supplies the event-wait argument and contains no wrapper-level global barrier,
but `deep_ep` itself was not installed. Rank-local behavior below that wrapper
is inferred rather than directly source-backed.

That live eight-rank replay emitted observations on all 32 nonempty steps. All
steps carried 24 layers and 48 unique dispatch/combine sites; 23 DBO steps
carried 96 invocations. The run is void with findings because the fatal
`ttft_exact_single_batch` guard was violated. That guard tested the same
serial-versus-observed equivalence premise required to attribute the two
in-band TPOT reductions to DBO. DBO-off steps 24 through 31 bound the measured
non-DBO residual at 1.231 percent of the mean DBO reduction on both placements,
but the failure is not orthogonal and no behavioral pass fraction is reported.

Both arms use participant-local frontiers. Their structural differences are
the open TRAF-9 whole-layer MoE ordering approximation and the observed arm's
terminal logits plus `requests-visible` fan-in. TRAF-23 owns measured
frontiers only. The retained 440,115,200 directed bytes are a pre-TRAF-25
conservation identity over the source-multiplied table and are not portable.
The adapter emits zero-byte semantic all-to-allv markers. Those are an explicit
`no-byte-evidence` observation mode, named by
`simllm.traffic.observed_routed_byte_evidence`, and they no longer stand in for
a byte check: VLLM-24 closed that P0 gap. The
producer-disabled component path passed all 64 per-step direct serial
comparisons and both accepted graph and legacy diagnostic GOAL hashes. See
[the observed-schedule results](../../examples/vllm_observed_schedule_v1/RESULTS.md).

The 2026-08-13 TRAF-13 requalification drove the same unmodified producer
through a third, structure-matched arm and measured what the void run could
only name. Adding cross-microbatch serialization edges and nothing else
isolates DBO at 1,450,472.652 ps per decode step on one node and
13,051,993.043 ps across nodes, 99.6 percent of the communication ceiling that
the frozen routed table allows. The producer's structural signature is a
+17.97 microsecond layer-ordering shift against a -17.97 microsecond terminal
frontier, which cancel because the LM-head compute is relocated rather than
added; the remainder is the microbatch split's byte accounting. On the
single-batch prefill the serial and observed arms are now exactly equal on both
placements. That study covered the traffic-side decomposition but deliberately
left the adapter-owned qualification to a later closure. See
[the observed-overlap results](../../examples/vllm_observed_overlap_v1/RESULTS.md).

VLLM-22 is complete. Expectations were frozen at commit `6459c3c` before
implementation and the first measured run. The closing qualification drove
the real eight-rank vLLM v0.26.0 replay through all 32 nonempty steps and the
supported `ExecutionObservations` to `CompletionEvent`, `StepResult`, TTFT and
TPOT chain. Its three genuine-risk instances passed: the producer reduced
per-request TPOT by 1.436193 percent on one node and 11.587805 percent across
nodes, and the absolute reductions scaled by 8.999138 across the ninefold
link-rate change. No fatal guard was violated.

All 32 steps passed independently derived submission-order, dependency,
correlation, rank, logical-stream, completion-frontier and original-request
identity checks. Sequential single-batch comparison preserved full reducer
history before requiring exact graphs, execution events, timestamps,
completion order, `StepResult` and request metrics. The source audit names
the cooperative DBO wrapper, shared compute and communication streams and
DeepEP event waits, and found no overlap knob or compatibility-derived edge.
The disabled real replay made one one-argument legacy sink call per step and
matched the independent serial reference through the standard exact artifact
comparator for record, graph, diagnostic and graph-derived GOAL, execution,
completion and request-metric bytes. A tracked test calls that actual wrapper
without vLLM or `third_party` and proves the lock fails under one-byte
mutations. See
[the producer qualification results](../../examples/vllm_producer_qualification_v1/RESULTS.md).

VLLM-24 is complete. Expectations were frozen at commit `20f6017` and amended
at `1a4db9b`, both before the harness and every result-producing run; the
amendment corrected two frozen rule statements that were written as if every
routed byte leaves the owner, which is false for combine. The independent guard
lives in `simllm.traffic.routed_conservation` and runs on the full-step routed
plan from both the observation-aware lowerer and the serial renderer. Its
ownership side is built from the record's per-request scheduled token counts,
the declared `RoutedMoeSupply.engine_rank` and the model geometry, none of which
comes from the per-token routing walk that produces the byte table it inspects.
Five rules apply to every routed representation and four more need
deduplicated captured routing; the uniform destination approximation is
deliberately exempt from those four, because it never merges experts that share
a destination.

The study ran the captured Granite routing at EP worlds 2 and 8 against a
source-replicated arm that reproduces the pre-TRAF-25 shape. All 8 fatal guards
held and 5 of 5 executed scored instances passed, against a frozen denominator
of 9. At EP world 8 the replicated arm emitted 42,656 hops against an 8,448-hop
bound and was detected; at EP world 2 it emitted 2,112 against the same bound
and was not, which is a first-principles certainty rather than a measurement,
since a two-rank world admits at most one remote owner per token-layer. The
four unexecuted instances are the frozen decode cell: the frozen capture carries
22 prefill tokens and zero decode tokens, so that cell cannot be built from it
at all. A labeled post-specified one-token prefill chunk behaved as the frozen
decode cell predicted and is reported separately. See
[the conservation results](../../examples/routed_byte_conservation_v1/RESULTS.md).

VLLM-6 is complete. Expectations were frozen at commit `20f6017` before the
implementation and every run, replacing the earlier post-specified component
result that could not close the clause. The geometry reader now follows vLLM's
own resolved MoE parallel shape: the expert world is the flattened
`dp * pcp * tp` rather than `dp` alone, expert parallelism leaves the expert
weights entirely un-tensor-sharded while its absence shards them across that
whole flattened set, the global expert count includes the EPLB redundant
copies, and the per-rank local count follows vLLM's uneven remainder
distribution. An expert world wider than the expert count is refused rather
than handing a rank no experts. `SimExecutor` derives its expert group in
vLLM's `ExternalDP x DP x PP x PCP x TP` layout order, which excludes the other
pipeline stage, and binds it once to an expert-group-capable sink only when
expert parallelism is actually in use.

That study passed 22/22 scored instances with 5 of 5 fatal guard cases held,
over 12 geometry cells, 5 rank layout cells, 3 corrected directions and the two
binding cells. The parallel side of every cell was re-derived against the real
pinned `vllm.config.ParallelConfig` in a separate interpreter; 13 of 17 cells
constructed and agreed exactly, while vLLM v0.26.0 itself refuses prefill
context parallelism combined with data parallelism and the probe could not
supply the launcher convention the ExternalDP cell needs. See
[the geometry results](../../examples/vllm_moe_geometry_v1/RESULTS.md).

The pre-play framework oracle adds a separate, inert-by-default vLLM general
plugin. `VllmCpuRunner` enables it only inside an isolated CPU process. The
plugin observes the stock `CPUWorker`, `CPUModelRunner`, v1 KV manager, block
pool and scheduler. It records allocation, prefix-hit, eviction, preemption
and release decisions after the owning framework method decides them. It also
measures CUDA allocator bytes before and after model load and rejects any
increase.

The vLLM 0.26.0 CPU MoE path exposed one important capture gap: setting
`enable_return_routed_experts` allocated a response tensor, but the monolithic
`CPUFusedMOE` path left that tensor filled with zero because the built-in
callback is attached to the modular route. The plugin does not use those
zeros. It observes the exact expert IDs returned by
`cpu_fused_moe.select_experts` immediately before the unchanged expert kernel
uses them and passes the same IDs into the stock request capturer. The
framework remains the sampling, scheduling, dispatch and KV authority.

The 2026-08-12 study built vLLM 0.26.0+cpu from source, reached
`CpuPlatform`, `CPUWorker`, `CPUModelRunner` and Granite on CPU, and observed a
zero-byte CUDA allocation delta in both capacity cells. All three framework
outputs matched the Transformers oracle exactly. Every one of 1,512 aligned
routing rows was an order-only difference with the same selected expert set
and zero changed all-to-all bytes. The detailed evidence is in
[the framework-oracle results](../../examples/framework_oracle_v1/RESULTS.md).

VLLM-30 is complete. The 2026-08-25 qualification moved the pin to vLLM
0.27.1 at source commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`.
The native registry retains Kimi K3, Qwen3.5 and every required Granite
family. Executor, worker, communicator, serializer, placement and extraction
tests pass against the installed release, and both CPU-only in-process live
paths reach their intended seams. The worker mirror now carries the renamed
sleep fields and refuses vLLM fault tolerance before its device-backed worker
sentinel can start. Repeated Granite extraction produced the new canonical
inventory recorded in
[the pin-bump results](../../examples/framework_pin_bump_v1/RESULTS.md); the
v0.26.0 inventory and its study remain unchanged historical evidence.

The CORE-51 first slice reaches the real vLLM v0.27.1 disaggregated-prefill
control seam under `SimExecutor`. One eight-worker prefill engine hands its
stable session request to one eight-worker decode engine through the
scheduler-side connector and the shared virtual clock. All four frozen TTFT
decompositions have 0 ps residual, and all six behavioral relations pass.
The workers deliberately move no KV tensor because simulated GPUs allocate no
paged KV storage. The concurrent extension admits multiple requests across
one-plus-one, one-plus-two and two-plus-one pool ratios, conserves all 144
request lifecycles and exposes stock-scheduler batches as wide as eight. Its
exact throughput curves are live, while the frozen nondecreasing delay claim
is refuted and remains VLLM-35 through VLLM-39; see
[the concurrent-session results](../../examples/pd_session_concurrent_v1/RESULTS.md).

The imported-surface follow-up prices only the decode pool from two MEASURED
Granite rows in the merged Hopper candidate record and keeps batching service
separate from live scheduler queue wait. It conserves 2,304 request
lifecycles across 36 frozen cells and observes increasing per-token request
delay on all 30 segments. Scheduler queue wait dominates from the first 250
to 500 requests/s segment even while imported batch service per token falls.
Only 1 of 24 frozen held-out bands holds, refuting the queue model and its
1,056.6 and 2,113.2 requests/s predicted knees. The monotonic-delay claim is
validated, but the attempt is exposure-contaminated by a disclosed
pre-reader reconnaissance command, so VLLM-39 and VLLM-35 remain open on the
clean VLLM-40 repetition. VLLM-41 owns the distinct lower-load queue-onset
residual; see
[the imported-surface result](../../examples/pd_session_load_delay_v1/RESULTS.md).

The clean VLLM-40 repetition subsequently validated increasing delay on all
30 segments from 250 to 8,000 requests/s and closed VLLM-35, VLLM-39 and
VLLM-40 without changing the frozen model. VLLM-41 then froze and ran 78
lower-load cells from 50 to 250 requests/s. Its surface-and-arrival-only model
predicted a central onset at 225 to 230 requests/s, with 220 to 225 also
admitted by the measured-surface uncertainty. All six observed configurations
instead share the earlier 210 to 220 requests/s first queue-dominated segment,
after five non-queue-dominated segments. The run conserves 4,992 admissions,
handoffs and terminals plus 19,968 decode tokens at zero TTFT residual. All 30
held-out scheduler-wait bands hold, while only 14 of 30 separately scored
batching-service bands hold. VLLM-41 closes on the common sub-250 onset;
the successor phase-complete, service-only predictor advances the shared clock
through independently frozen prompt service and handoff before pricing decode
batch membership from the independent measured service surface. All 48
non-held-out and all 30 held-out bands hold, conserving 4,992 admissions,
handoffs and terminals plus 19,968 decode tokens at zero TTFT residual. This
qualifies VLLM-42 for the integrator's atomic registry and index closure with
no VLLM-50 residual. The common 210 to 220 requests/s onset and validated 250
to 8,000 requests/s direction remain settled and were not rescored; see
[the batching-service result](../../examples/pd_session_batching_service_v1/RESULTS.md).

The same live driver now accepts pool-specific content-addressed lookup
bindings for CORE-53. The retained candidate study selected its exact decode
row twice and surfaced candidate status without a calibration claim. Both
record-absent runs retained all accepted KV bytes and timestamps, but the
frozen complete-result byte guard was voided by vLLM's fresh pool-local request
identifier suffixes. CORE-58 owns the next frozen identity boundary; see
[the session kernel-cycle result](../../examples/pd_session_kernel_cycle_v1/RESULTS.md).

The pinned DeepSeek-V3 configuration surface now publishes a complete logical
inventory beside SGLang's structurally identical record. Its physical code
objects and observed launches remain absent by design and name VLLM-38 as the
framework-owned join; see
[the DeepSeek inventory results](../../examples/model_extraction_deepseek_v3_v1/RESULTS.md).

The first VLLM-11 normalization slice is complete. The ordered bridge
qualifies one stock manager against a pinned uniform geometry, maps native to
stable logical request IDs, and emits only sidecar-witnessed aggregate-pool
`KvCacheWork`: exact new block IDs, prefix bind then touch, release then free,
eviction before the allocation that reuses a cached block, and a
capacity-shaped recompute interval after preemption. The bridge fails closed
on an unknown sidecar kind. The pool block byte shape is derived exactly as
two times layers times KV heads times head size times dtype bytes times block
tokens. Every operation is a read-only projection of the CPU-oracle sidecar
and has zero service bytes; vLLM remains the cache authority.
`SIMLLM_VLLM_ORACLE_SCOPE=kv` installs only the same stock manager, block-pool
and preemption observers for deterministic simulated-worker capture; the
unset scope retains the accepted full CPU-oracle plugin bytes. The supported
vLLM v0.27.1 pin has no `max_num_partial_prefills` or
`max_long_partial_prefills` scheduler fields, so neither appears in adapter
configuration or causal provenance. VLLM-44 through VLLM-47 own the facts that
the sidecar does not yet identify, and VLLM-11 stays open on their precision
join rather than treating this projection as complete lifecycle evidence.

The nonvoid
[surrogate conformance study](../../examples/surrogate_conformance_v1/RESULTS.md)
runs the pinned vLLM 0.27.1 engine in process and retains each native
`SchedulerOutput` beside the adapter projection. All native-versus-projection
guards pass, as do the source hash, causal tuple, token conservation and
end-to-end mutation guards. The bridge therefore supplies a valid oracle for
this frozen comparison, but the framework-free surrogate is not certified:
14 family rows miss, including block-lifetime identity and prefix-cache token
intervals. The bridge's pre-decision `RESERVE` limitation remains explicitly
unscored under the governing amendment and remains VLLM-44 work; VLLM-11 and
VLLM-44 through VLLM-47 neither close nor widen from this result. Certification
is scoped to this scheduler pin and must be re-earned after every pin bump.

## Open tasks

### Precision

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
- VLLM-11 (Precision; P1; L) (remaining after the ordered normalization
  slice): join the enriched lifecycle stream into the CORE-3 ledger and the
  live metric chain after VLLM-46, VLLM-47 and VLLM-44 provide per-layer
  service bytes, native ownership state and native capacity correlations. The
  landed slice qualifies one uniform manager and projects aggregate pool
  geometry, token intervals, exact block and request IDs, causes available in
  the sidecar, prefix bind/touch, release/free, eviction and recompute in
  source order. It is observation-only and is not sufficient evidence for
  reference-sensitive lifecycle replay or KV latency. Closure requires exact
  event cardinality and identity against the owning manager and pool objects,
  successful transactional `KvLifecycleLedger` consumption, and a signed
  time to first token (TTFT) or time per output token (TPOT) effect through
  the supported `StepResult` chain. The first-slice projection and the
  oracle-disabled path remain byte-identical. Optional swap and transfer stay
  outside this precision closure under VLLM-45.
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
- VLLM-44 (Precision; P1; M): add framework-native correlation and capacity
  decision identity beyond the sidecar's row ordinal. The first slice gives
  each projected operation a deterministic source-row ID, leaves
  `correlation_id` unset, stamps one configured placement epoch, and sees only
  successful allocations after the capacity decision. Instrument the owning
  manager and pool objects before every reserve decision and carry the attempt,
  allocation epoch, outcome and native correlation through allocation,
  eviction, release and preemption. Acceptance requires a total one-to-one
  join for successful and refused attempts, including the exact preemption
  consequence, without changing the first-slice or oracle-disabled bytes.

- VLLM-46 (Precision; P1; L): replace the aggregate metadata-only KV byte
  shape with native per-layer reads and writes on the metric-live path. The
  current bridge identifies the whole-pool block size as
  `2 * layers * kv_heads * head_size * dtype_bytes * block_tokens` and emits
  zero service bytes. Observe each model-runner layer's exact request, block
  and token interval, then emit `READ` and `WRITE` with the layer index and
  exact bytes. Acceptance requires the per-layer sum to equal the pinned
  aggregate shape exactly, exact cardinality against untouched runner calls,
  a signed KV-service effect through `StepResult`, time to first token (TTFT)
  or time per output token (TPOT), and byte-identical metadata-only and
  oracle-disabled baselines.
 This task was numbered VLLM-42 on the surrogate-loop branch before
  integration; it is renumbered here because the queue-onset publication's
  frozen bytes on main already reserve VLLM-42 for the batching-service
  residual. The surrogate-loop publication cites the new number in a
  disclosed registration correction.
- VLLM-47 (Precision; P1; M): replace the bridge's forced prefix `TOUCH` and
  release-then-`FREE` projection with native ownership and content state. The
  current sidecar omits block reference counts and hash residency, so it
  cannot distinguish `TOUCH` from `RETAIN`, or a reclaimable cached block from
  content that is discarded. Observe the owning pool object's reference count
  and hash state before and after each action. Acceptance requires exact
  reference counts, exact `TOUCH` versus `RETAIN`, `FREE` only for discarded
  content, and successful `KvLifecycleLedger` consumption for one exclusive
  prefix, one shared live prefix and one released cached prefix, while the
  current projection and oracle-disabled artifacts remain byte-identical.
 This task was numbered VLLM-43 on the surrogate-loop branch before
  integration. VLLM-43 itself stays permanently reserved and unassigned: the
  queue-onset freeze pre-registered it for differing onset segments and the
  run resolved it unused, so its frozen bytes keep the name and no new task
  may take it.
- VLLM-48 (Precision; P1; M): observe the wall service of every stock vLLM
  communicator invocation executed inside a live model-runner step. Bracket
  GPU calls with CUDA events and CPU calls with a monotonic host clock, then
  attach the collective kind, payload bytes, world size, group tag, layer
  metadata, timer and environment identity through an optional versioned
  `StepRecord` field. An absent field must preserve every accepted capture and
  byte-locked fixture exactly. Compare captured service with the aggregate
  collective-floor authority only at matching kind, byte and rank coordinates;
  refuse a capture/calibration environment mismatch by default and stamp an
  explicit acknowledgement into every deliberately accepted mismatch. Local
  closure requires a pinned vLLM 0.27.1 CPU source build running tensor
  parallel size two over gloo, exact captured-call and metadata conservation
  against the independently frozen Granite step population, exact shape
  reproduction over two fresh live runs, strict old-load and new-round-trip
  schema evidence, mutation controls for every scored family, and both the
  comparator refusal and acknowledgement paths. Local service values remain
  environment-labeled and unscored. The retained campaign includes attempt 2,
  in which both live runs completed and captured 100 calls but the original
  request-identity guards failed, making that attempt VOID with no behavioral
  score. A post-attempt-2 harness-reality amendment at `ad98074` pins vLLM's
  exact logical-ID plus eight-hex-character internal suffix rule from
  `vllm/v1/engine/input_processor.py:249` and requires fresh evidence. Attempt
  3 predates that amendment and is diagnostic only. Fresh attempt 4 passed
  every fatal guard under the amendment but scored five of seven: both M1
  instances expected a final logits `gather`, while standard vLLM executed
  `all_gather`. `LogitsProcessor` stores the platform's all-gather preference
  (`vllm/model_executor/layers/logits_processor.py:55`), the platform interface
  default returns true (`vllm/platforms/interface.py:1102`), and CPU inherits
  that default (`vllm/platforms/cpu.py:42`), so the freeze was wrong about the
  standard cross-platform logits path rather than this CPU build. The retained
  refutation leaves this task open. Closing the kind family requires successor
  expectations committed before another pair of fresh runs; no observed run
  may be relabeled as closure.
- VLLM-49 (Precision; P1; L): run the VLLM-48 in-situ seam on the A100
  multi-GPU lane and score real collective service against an aggregate-floor
  calibration from the same A100 environment. The standard tensor-parallel
  logits path is expected to use `all_gather`, following the pinned platform
  default identified by VLLM-48, unless the A100 source pin explicitly
  overrides it. Freeze payload, rank and collective-kind sweeps before capture;
  state serialization and topology bounds before reading service values;
  require exact coordinate coverage and explain every floor violation or
  residual outside the registered band. Exercise the deferred CUDA event
  resolver, confirm that the model-step thread does not wait for event
  synchronization, and prove ordered record flush at process shutdown. A
  cross-environment acknowledgement is diagnostic only and cannot close this
  task. Carry the accepted service comparison into at least one signed time to
  first token or time per output token consequence while the capture-disabled
  baseline remains byte-identical.
### Completeness

- VLLM-45 (Completeness; P2; L): normalize stock vLLM offload connector swap
  and transfer events into the same pool and request identity envelope. The
  first bridge deliberately covers the recompute-only stock scheduler and
  emits no `SWAP` or `TRANSFER`. Observe source and destination pools, tier,
  blocks, token interval and exact bytes from the connector objects, reject a
  partial cross-pool join, and require the connector-disabled bridge,
  scheduler records and accepted oracle sidecar to remain byte-identical.

- VLLM-38 (Completeness; P2; L): join the published DeepSeek-V3 logical
  inventory to vLLM's physical MLA, routed-expert, shared-expert, dense and
  MTP launches on supported target silicon. Extend VLLM-12's source-backed
  producer with the DeepSeek operation identities and bind every launch by
  COMP-6's exact graph, operation and ordinal keys. Reject a missing or
  ambiguous family before emitting a partial physical envelope. With the
  DeepSeek capture absent or disabled, preserve both published logical
  inventory bytes, every StepRecord and every existing physical capture byte
  exactly.

- VLLM-25 (Completeness; P2; M): support shared-expert and mixed dense and
  routed MoE geometries in the config reader. `model_dims_from_vllm_config`
  refuses them instead of pricing them as one whole-model routed geometry,
  reaching parity with the SGLang reader on the shared-expert and mixed
  dense-and-routed families with the same per-field predicates, though the two
  lists are not identical in either direction: this one adds
  `num_shared_experts` and `shared_intermediate_size`, which the vLLM model
  definitions spell, while the SGLang reader also refuses MLA, speculative and
  quantization fields that are compute and sampling concerns outside this
  guard's reduction-inventory scope. The refused values are a positive
  `n_shared_experts`, `num_shared_experts`,
  `shared_expert_intermediate_size`, `moe_shared_expert_intermediate_size`,
  `shared_intermediate_size`, `first_k_dense_replace` or `num_dense_layers`; a
  `moe_layer_freq` or `decoder_sparse_step` other than 1, since 1 is the only
  fully routed stride; and a non-empty `mlp_only_layers`. The refusal exists
  because the collective inventory would be wrong, not only the FLOP count. A
  shared expert's output is all-reduced over the tensor-parallel group even
  when the combine kernel already reduced the routed output (pinned vLLM
  0.26.0, `model_executor/layers/fused_moe/runner/moe_runner.py:416-433`), and
  the shared MLP itself rides a row-parallel projection with
  `reduce_results=True` (`model_executor/models/granitemoeshared.py:48` and
  `:108`), so the layer keeps an mlp-site allreduce that
  `layer_tp_allreduce_sites` drops for a routed all-to-all layer. A mixed
  schedule leaves some layers with two allreduce sites and no all-to-all at
  all (`model_executor/models/qwen2_moe.py:310-316`,
  `qwen3_moe.py:385-391`). Acceptance needs shared-expert weight bytes and
  active FLOPs in `ModelDims`, the retained mlp-site allreduce for those
  layers, a per-layer routed schedule shared with TRAF-34, and a
  byte-identical fully routed baseline plus the preserved refusal for every
  geometry still unsupported.

- VLLM-12 (Completeness; P1; L): add a thin source-backed producer for the
  supported model runner's physical device schedule. Emit one concrete
  `ExecutionGraph` instance, a total set of observed
  `OperationImplementationBinding` records for noncollective launches and a
  total set of `CollectiveDeviceStageBinding` records for supported
  GPU-resident NCCL/RCCL stages under the same identity envelope as the compute
  profile. Preserve
  CUDA or ROCm stream and program order, event waits, physical kernel launches,
  supported NCCL/RCCL launch and chunk boundaries, and synchronous or
  asynchronous completion frontiers. Bind per-step shapes and framework KV
  events without inferring concurrency from aggregate phase timings. This
  adapter owns
  framework observation only: COMP-6 owns the generic identity projection and
  totality checks, while compute and runtime own replay, service timing,
  `CompletionEvent`, `StepResult`, TTFT and TPOT. VLLM-23 owns expansion beyond
  the currently accepted model and shape modes. When this producer is disabled
  or absent, preserve every accepted executor and worker record, event sidecar,
  sink call, timestamp, token and completion order exactly. Reject unsupported
  or incomplete capture before emitting a partial graph or either binding set.
  COMP-6 alone owns the separate topology projection and template hash.
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
- VLLM-23 (Completeness; P2; L): extend the source-backed observation producer
  beyond the implemented Granite TP1, PP1, uniform one-token decode boundary.
  Add exact request and token-interval correlation for multi-token or padded
  DBO splits, explicit `ubatch_size`, TP greater than one and PP greater than
  one. The current path rejects each shape before emitting a plausible
  schedule. Acceptance must exercise every enabled shape through traffic
  rebinding and request completion while preserving those explicit refusals,
  the implemented Granite schedule and the producer-disabled serial path
  exactly.
### Uncategorized

- VLLM-3: sim-native metrics export via a `vllm.stat_logger_plugins` stat
  logger for virtual-time runs.
- VLLM-5: CI harness with transcribed stand-ins for the vLLM types
  (`Executor`, `ModelRunnerOutput`, `FullAttentionSpec`, `CompilationTimes`)
  so the init-RPC sequence and the step loop run end to end without a GPU
  stack installed.
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
