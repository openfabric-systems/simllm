# simllm.adapters.sglang

SGLang frontend adapter, pinned to SGLang main commit **8f2a3ad**
(2026-08-04; SGLang moves fast, so the pin is a commit, not a release). The
seam is the TP worker, installed without a fork through SGLang's plugin
framework.

## Interface

Simulated execution (`simllm/adapters/sglang/worker.py`):

```
SIMLLM_SGLANG_ENABLE=1 SIMLLM_SGLANG_MODE=virtual SIMLLM_SGLANG_GPU=b100 \
SIMLLM_SGLANG_STEP_RECORDS="${SIMLLM_DATA_ROOT:?configure SIMLLM_DATA_ROOT}/simllm/steps.jsonl" \
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
  reports its hit. Each record also carries the exact sampled count and the
  identity of the rows SGLang consumes a generated token from: an extend or
  mixed row counts only when its in-flight middle chunks are drained and it
  is neither finished nor retracted, and every decode row counts. Both fields
  are emitted together, because a count alone is refused as ambiguous
  whenever the sampling rows are not exactly the scheduled decode set, which
  two concurrent prefills violate. `SIMLLM_SGLANG_SAMPLE_IDENTITY=0` is the
  explicit compatibility path: both fields stay absent, every consumer reads
  the whole scheduled batch as sampled, and the records are byte-identical to
  the pre-SGL-12 stream. Step latency comes from the shared
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
when the scheduler runs in this process. `reset_configuration()` clears every
hook and is the boundary between independent in-process runs, with the same
semantics as the vLLM adapter's: without it a multi-cell driver leaks one
cell's sink, device or replay configuration into the next.

Per-step host cost is chosen through `select_sglang_host_model`
(`simllm/adapters/sglang/host.py`). It resolves a profile name (`ideal`,
`turing-cuda-graph`, `turing-eager-host`) and a launch count into the three
objects that must travel together: the `HostInitiationModel`, the
`gtx1660-ti-sm75` device key its calibrated constants demand, and a provider
pinned to the accepted `b100` compute envelope so the device key can move
without the compute moving with it. The two consumers spell the provider
differently, so there are two accessors and not one: `worker_overrides()`
carries `configure`'s `compute_provider` and `sink_overrides()` carries
`HtsimStepSinkConfig`'s `provider`. Splatting either one into the other's
consumer raises `TypeError`, and a test pins both spellings against the real
signatures. `ideal` is the default and returns exactly
what a study built by hand before this seam existed, which is what keeps every
accepted artifact identical. A calibrated selection carries
`SGLANG_HOST_TRANSFER_DISCLOSURE`, because none of its constants was measured
on SGLang: the per-launch point is a GTX 1660 Ti capture from
`examples/host_step_cost_v1`, and compute stays on `b100`. The launch count
defaults to `SGLANG_TRANSFERRED_LAUNCH_COUNTS`, which is a study convention
rather than a per-class measurement: `examples/compute_fidelity_v1` enumerated
one eager decode step from vLLM 0.26.0 sources and froze its minimum and
maximum as the bracket `[440, 567]`, both endpoints apply to both launch
classes, and pairing 440 with CUDA graphs and 567 with eager launching only
reproduces the composed vLLM study's two headline cells. SGL-24 owns the
SGLang-side count that would replace the borrowed bracket. Every enabled row is
a disclosed three-source device hybrid and never a calibration.

Token serving has two paths. The default fabricates one mid-vocabulary token
for every row. `SIMLLM_SGLANG_REPLAY_RUN` instead names a joined pre-play
replay run, and `SglReplayTokenSource` then verifies the trace against its
recorded digest, requires each request to enter SGLang with
`max_new_tokens` equal to its oracle length, refuses stop strings, stop
regexes, an oracle token that would match a stop or EOS id before the last
position, structured output, speculative batches and logprob batches, and
serves each request's predefined token at the output index the scheduler
itself reports (`len(Req.output_ids)` at forward time). A mid-prompt chunk,
whose token SGLang discards, keeps the fabricated token and does not move the
oracle index. With no replay run configured the fabricated path is unchanged.

Completion visibility: the worker never sees finish decisions (EOS and
`max_new_tokens` are applied in `process_batch_result` after the forward
returns) and there is no drain step at this seam, so
`finished_request_ids` is always empty and a record consumer infers a
request's completion from the last record that schedules it, the same
convention as the vLLM adapter's in-process path.

In-process driving (`simllm/adapters/sglang/pump.py`) is what makes
`configure()` reachable at all. SGLang builds its `Scheduler` inside an
`mp.Process` and loads plugins in `run_scheduler_process`, so hooks set in a
parent process never reach the worker. `build_in_process_scheduler` removes
the boundary instead of reaching across it: it constructs the pinned
`Scheduler` in the calling process after `install()` and `configure(...)`
have run, mirroring `run_scheduler_process` minus the parts that only make
sense in a child (`publish(server_args, role="scheduler")` is required,
because the constructor reads the process-global config bags; the
parent-death watchdog and the process-title rewrite are not applied). SGLang
carries its own in-tree precedent for this in `srt/ray/scheduler_actor.py`.
`SglangSchedulerPump` then unrolls the body of `event_loop_normal` into one
synchronous `step()`: plan, run and settle a batch when the plan carries
one, call the scheduler's idle handler when it does not, publish
`last_batch`, all under `torch.no_grad()` because the real loop is decorated
`@DynamicGradMode()`, and with `cur_batch_for_debug` assigned because the
watchdog reads it. Ingress is the caller's own list through
`process_input_requests`, so an arrival gate on the worker's `VirtualClock`
decides when a request enters and the framework alone decides what to run.

The pump's only mutation of scheduler state is the egress socket. Generation
results leave through `scheduler.output_streamer.send_to_detokenizer`, a
`SenderWrapper` around a `zmq.PUSH` socket with no listener in this
configuration; the pump replaces that one object with
`SchedulerOutputCollector` and reads finished requests, their finish reason,
their token counts, their radix hit and their retraction count out of the
payloads. That is the completion signal the worker seam cannot report.
`attach_output_collector=False` is the exact off path and mutates nothing.
The pump admits chunked prefill on the default record path and refuses it only
on the compatibility stream. `chunked_prefill_refusal` is that gate: SGL-12
made every record carry `num_sampled` and `sampled_request_ids`, so a
mid-prompt extend row is excluded from the sampled set, while
`SIMLLM_SGLANG_SAMPLE_IDENTITY=0` restores the pre-SGL-12 stream in which every
scheduled row is read as having produced a token and a mid-prompt chunk would
be scored as a generated token. The gate reads that state through
`active_sample_identity`, whose authority is the constructed worker: the
scheduler builds the worker before the gate runs and the worker latches
`sample_identity` into its translator, so a later `configure` call or
environment change never reaches it, and the hook-or-environment derivation is
the fallback used only when no worker exists yet. Admitting chunked prefill is
not a safety certificate: the sampled-row rule behind it is source
transcription plus stub batches, the gate claims no live-scheduler agreement,
which stays SGL-22, and hazards outside that rule are outside what it
inspects. The module imports without SGLang and without torch, and
its ordering contract is tested against a stub scheduler.

RadixCache prefix matching, eviction and the token/request pool accounting
are scheduler-side index bookkeeping and stay real, so radix hit rates and
vRAM pressure respond to the workload exactly as in production.

The geometry reader accepts strict single-GPU routed-MoE projections for
Granite MoE, Mixtral, and all-layer Qwen3 MoE. It maps global routed experts,
top-k, per-expert width, and resident experts into `ModelDims`; the shared
compute provider then uses top-k for active FLOPs and resident experts for
weight bytes. Any MoE sentinel on an unknown family, shared experts, a mixed
dense/routed schedule, MLA, next-token-predict layers, or TP/EP/MoE-DP greater
than one fails before a dense estimate can be produced. Quantized MoE weights
also fail until their element width is sourced explicitly. Dense geometry and
its loud fallback behavior are unchanged.

External workload driving (`simllm/adapters/sglang/client.py`) is a separate
standard-library surface. `sglang_generate_payload` maps one immutable
workload request to the pinned native streaming `/generate` schema.
`SglangOpenLoopDriver` paces every request against one monotonic origin while
prior requests remain in flight, and `SglangHttpSubmitter` records cumulative
streamed-token visibility. Client submission lateness stays separate from
TTFT and from simulated framework queueing. No SGLang import or GPU is needed
for payload, pacing, chunk, or metric tests. The standard-library transport is
bounded to 64 requests by default and rejects a larger plan unless the caller
explicitly provisions one response-drain thread per request; larger studies
need a selector or async transport so delayed reads do not distort visibility.
SGL-4's offered-load HTTP campaign owns that scalable transport requirement.

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
run: on 2026-08-04 a real SGLang at pinned commit `8f2a3ad`, installed as an
editable fresh clone in a machine-local environment whose resolved historical
path is intentionally omitted, ran the offline `Engine` on the CPU engine
(`device="cpu"`, torch_native attention selection, gloo process groups,
`SGLANG_BUILD_RUST_EXTS=none`) with the plugin active via its entry point:
the scheduler subprocess constructed `SimTpModelWorker`, three requests generated 8
fabricated tokens each, and the streamed JSONL held 9 schema-tagged records
with exactly 3 prefill and 21 decode rows and monotonic virtual time.
RadixCache ran live (0 hits, correctly: first-contact prompts shorter than
any reusable prefix). The overlap path is out of scope for the first
iteration: run with `--disable-overlap-schedule` (nothing forces overlap
on, and PP asserts it off anyway).

For a current reproduction, `SIMLLM_SGLANG_ENV` may document a compatible
environment in local configuration, but it does not define the identity of the
recorded run.

SGL-12 is closed. Every record now carries the exact sampled count and
identity, transcribed from the pinned `process_batch_result_prefill` and
`process_batch_result_decode` rules rather than modeled, and the absent-field
stream stays selectable and byte-identical. The frozen twelve-cell study in
[examples/sglang_worker_seam_v1/RESULTS.md](../../examples/sglang_worker_seam_v1/RESULTS.md)
drove one stub batch stream per cell through the adapter seam and the shared
metric chain in both states: all 82 scored exact-oracle rows matched to the
picosecond and no fatal guard was violated. Without the fields a chunked
request's reported TTFT was the completion of its first extend step, i.e. 49.9
percent of its true TTFT with two chunks and 33.2 percent with three, and its
token count and TPOT were inflated by the mid-prompt steps. No live SGLang
scheduler was in that loop: the batch stubs carry the pinned attribute names,
so observed agreement with a running scheduler is SGL-22.

Joined pre-play token replay is implemented as `SglReplayTokenSource` with its
fabricated-token identity off path, both import-free tested. It has not been
driven by a live in-process SGLang scheduler and has not reached a reported
metric, so its live smoke and live reachability are PLAY-16 in
[preplay.md](preplay.md#open-tasks).

The recorded smoke JSONL is exercised against the closed-loop sink as of
the M4 first slice: all 9 records load through
`simllm.core.step_records_from_jsonl` and replay through
`simllm.backends.HtsimStepSink` behind a declared tp=8 manifest, with
monotonic virtual time and every step's simulated latency above the
compute-only estimate (examples/m4/RESULTS.md check E). The live
closed-loop run of that slice used the vLLM adapter. SGL-8 closed the
remaining half: the sink seam now drives `htsim_rnic` live from a real
scheduler in one process, reported in
[examples/sglang_end_to_end_v1/RESULTS.md](../../examples/sglang_end_to_end_v1/RESULTS.md).

That slice is frozen by expectations-only commit `8907c53`. On 2026-08-13 four
requests entered a real `Scheduler` through the arrival gate on the worker's
own clock, across four fabric cells and one sink-free control, for 95 simulated
steps and 4,560 `htsim_rnic` invocations. All 11 fatal guards held, so the run
is not void; 5 of 5 scored exact relations and 4 of 4 scored behavioral
relations passed, in two classes that are never summed. Per-request TTFT and
TPOT match an independent standard-library recomputation exactly over 192
intervals, the 40,596 per-request directed-byte rows and 24,136 executed GOAL
rows agree with a recomputation straight from the trace, and the measured
per-step compute service inverts to within 0.1 percent of the hand-counted
resident weight bytes at both expert-parallel widths. The closed loop is
demonstrated rather than assumed: the framework took 26, 24 and 21 scheduler
steps at 400, 200 and 100 Gbit/s for the identical workload, while the
sink-free control took 16. This is the first SGLang run in this repository to
drive `htsim_rnic` at all, and the first routed study driven by an SGLang trace
rather than a vLLM one.

SGL-23 is closed. The chain now owns its per-step host cost instead of leaving
every study to build the zero model by hand, and the seam is frozen by
expectations-only commit `79b03da` and reported in
[examples/sglang_host_step_v1/RESULTS.md](../../examples/sglang_host_step_v1/RESULTS.md).
On 2026-08-14 the tracked nine-record SGLang smoke capture replayed through
`HtsimStepSink` in seven selector states, for 3,024 `htsim_rnic` invocations
plus 432 more for a hand-built pre-seam reference sink. All 7 fatal guards
held, so the run is not void; 63 of 63 scored exact-oracle rows and 18 of 18
scored behavioral instances passed, in two classes that are never summed. The
regime flip is one launch wide exactly where the closed form puts it: at 122
CUDA-graph launches every record records zero exposed host time and a
`represented_bound` equal to the `memory` bound its own provider reported, and
at 123 every record records `represented_bound == "host-initiation"` with a
positive exposure. That bound half was carried into the recorded rows only in
the fix round, after the first run scored the exposure half alone; the study
report states that chronology and the rerun changed no timing value. The
pre-registered warning that a fully masked calibrated cell is still not
identical to the ideal arm was confirmed: it runs 5 to 24 ns longer per step
because the two arms quantize the whole-nanosecond enclosure differently, so
`ideal` is the only exact off path. At the transferred vLLM bracket the
composed decode step is 76.44 percent (CUDA graph, 440) or 92.43 percent
(eager, 567) one transferred constant and the modeled B100 compute contributes
exactly zero, because the launch floor masks it. The fabric term is entirely
simulated packets and reproduces a hand closed form to 1 ps per collective.
Nothing in the run was measured on SGLang and no live scheduler was in the
loop: SGL-24 owns the launch count and SGL-26 owns live selection.

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
states; both streams must equal the fixture exactly. SGL-12 added the sampled
count and identity to every record, so that original fixture is now the
compatibility baseline and the current default has a second fixture beside it.
The communicator flag must move neither. The pinned call-site
audit now derives every observed row from AST, and the correction supplement
in `examples/sgl_communicator_v1/RESULTS.md` identifies the actual
`output_tensor_list` callers without rewriting the frozen expectations file.

The TRAF-13 observation-aware backend handoff does not add an SGLang schedule
producer. The current communicator still emits its zero-time component event
sidecar rather than `ExecutionObservations`, and no source qualification,
runtime projection, TTFT, or TPOT relation was run for this adapter. SGL-17
owns that optional producer while preserving the current exact flag-off path.
The framework-oracle fallback is separate from the simulated worker and is
also inert by default. `SglangCpuRunner` selects the stock CPU engine,
`TpModelWorker`, `ModelRunner`, Granite model, sampler and
`return_routed_experts` response path. Observation hooks project the real
paged allocator, `RadixCache` matcher and eviction boundary, and decode
retraction decisions. SGLang's host capturer unconditionally requests pinned
memory, which initializes a CUDA pinned allocator even in its CPU engine. The
fallback substitutes an ordinary CPU tensor only for that capture buffer; it
retains every selected expert ID and does not replace model dispatch or
scheduler state. A one-request qualification reached the stock worker,
returned the selected expert IDs, observed an allocation and prefix lookup,
and wrote a strict v2 trace. The main 2026-08-12 study did not select this
fallback because the higher-fidelity vLLM CPU build qualified first. SGLang
therefore remains component evidence in
[the framework-oracle results](../../examples/framework_oracle_v1/RESULTS.md),
not part of its eight-instance behavioral headline.

The fallback's dispatch layer label now comes from SGLang. The pinned Granite
MoE block builds its router without a layer identity (`TopK(top_k=...,
renormalize=True)` at `models/granitemoe.py:65-68`) while handing the same
block's explicit `layer_id` to `FusedMoE` at `:71-79`, so the `None` reaches
SGLang's single capture gate (`layers/moe/topk.py:1829-1845`) and would index
the capture buffer at `state_capturer/base.py:38-40`. An AROUND hook on
`RoutedExpertsCapturer.create` reads the identity off the constructed model,
takes each router's own `layer_id` when a model forwards one and otherwise the
unique sibling `layer_id`, refuses any router whose resolved id disagrees with
the layer index in its registered module name, and binds the result by
`TopKConfig` identity. The gate hook substitutes that id only where SGLang
passes `None`; the capturer hook infers nothing and rejects an unlabeled
capture, which also covers models that bypass the gate
(`models/inkling_common/moe.py:450`). Provenance is `framework-layer-id`, the
same value the vLLM runner writes. With `SIMLLM_SGLANG_ORACLE_LAYER_AUDIT=1`
every capture is additionally compared against the replaced model-order
surrogate and a disagreement is fatal; the audit is off by default and adds no
sidecar row when off.

The SGL-16 replacement is frozen by expectations-only commit `f786510` and
reported in
[examples/sglang_layer_id_v1/RESULTS.md](../../examples/sglang_layer_id_v1/RESULTS.md).
Two prompt shapes and one real decode-retraction cell ran twice on a live CPU
engine at the pinned commit, once with the surrogate and once with the
framework identity. All nine frozen relation instances passed, of which 3 of 9
are genuine-risk behavioral evidence; the run is not void, every frozen fatal
guard held. The remaining six are retained as three R2 allocator-orthogonality
checks and three R3 treatment-trace validity controls. The study recorded 1,752
audited captures with zero label disagreements, byte-identical raw framework
responses, KV events and per-token dispatch rows, and one retraction of `p3` at
framework step 18 that resumed to its length cap in both phases.
The pressure probing behind that cell also established two properties of the
pinned scheduler worth recording: it clamps `max_new_tokens` to the token-pool
capacity rather than retracting a lone oversized request, and its prefill
admission reserves decode headroom, so a retraction needs more concurrent
requests than the pool can hold.

### Distance from the vLLM path

The mission names two frameworks. This is where the second one actually
stands, measured against what the vLLM path already does rather than against
this module's own task list.

- A real engine drives the schedule: partial. SGLang's real `Scheduler`,
  `RadixCache` and pools do run, and the plugin replaces only the TP worker.
  But the seam sits below the scheduler's decision record. The emitted
  `StepRecord` carries `scheduled` and nothing else; `finished_request_ids` is
  always empty and there is no preempted set, while the vLLM executor ingests
  vLLM's own `SchedulerOutput` including completions and preemptions and adds
  a drain step.
- Per-request identity reaches TTFT and TPOT: present, by the same route the
  vLLM path uses. `request_id` reaches `StepRecord`, the record now carries
  exactly which requests produced a token, and `HtsimRequestMetricReducer`
  projects the sink's own per-step outcomes into per-request TTFT and TPOT
  with the seven-component partition. What is still absent is a runtime
  `CompletionEvent` path out of the adapter itself: the SGLang sink alias
  takes one argument, so the two-argument observation sink cannot be
  attached. SGL-13 owns that piece.
- Per-token routing reaches traffic: demonstrated. An SGLang v2 trace projects
  through `RoutedExperts` into `RoutedMoeSupply`, drives the per-layer MoE
  all-to-alls, is emitted as GOAL, executed by `htsim_rnic` and reaches TTFT
  and TPOT. Before SGL-8 every routed study in this repository used a vLLM
  trace.
- The observed schedule comes from the framework: absent. The only
  `ExecutionObservations` producer in the repository is on the vLLM side, so
  the lowering is serial. SGL-10 and SGL-17 own the SGLang equivalents.

One gap is worth naming because it is not visible from the task list.
`configure()` is only reachable from an in-process scheduler driver, so a
normal `launch_server` run, where SGLang builds the worker inside its own
scheduler subprocess, still cannot install a sink and falls back to the JSONL
sidecar. The pump does not change that; it makes the in-process driver a
supported surface instead of a study-local one.

The honest summary is that SGLang is now a real frontend whose decisions reach
the reported metric on one supported path, the in-process pump, and a real
frontend whose decisions are only observed and recorded on every other path.
The routing authority the SGL-8 run consumed is the SGL-16 framework-layer-id
trace, so the live-reachability precondition SGL-16 recorded is met by that
run. SGL-16 is not closed here: this study registered no SGL-16 acceptance
clause and a demo cannot close a precision task by association.

### Why SGL-14 is blocked rather than deferred

SGL-14 asks for native COMP stack entries for all-gather, broadcast, send and
receive "when those entries exist". They do not. `simllm.compute.nccl_stack`
exports exactly one collective entry point, `ncclAllReduce`, alongside
`ncclCommInitRank`; there is no `ncclAllGather`, `ncclBroadcast`, `ncclSend`
or `ncclRecv` anywhere in `simllm/compute`. Creating them is COMP work: COMP-14
owns the non-ring algorithm builders and COMP-15 owns the absent receive leg.
The ring-layout servable-domain restriction SGL-14 also asks to remove lives
in the same COMP module, in the layout validator every positive-payload call
passes through, and the `ncclAllReduce`-shaped lowering itself lives in the
shared VLLM-14 base that both adapters subclass, so removing it there would
change the vLLM path too. No adapter-side change can satisfy the clause, and
SGL-14 stays open with that precondition recorded rather than partially
closed.

While it stays open, the cost the SGLang communicator publishes for every
mirrored collective is zero. That is a declared modeling choice, not a
measurement and not a claim that the cost is negligible: the named alternative
is the calibrated dispatch, custom-op routing, device-communicator selection
and synchronization measurement that SGL-15 registers, with SGL-13 projecting
the result onto `CompletionEvent`, `StepResult` and TTFT or TPOT. Until those
land, any figure computed with the SGLang communicator enabled understates
communication by exactly the whole of it.

The single-GPU MoE and workload-driver slice is frozen by expectations-only
commit `c48e785` and reported in
[examples/sglang_moe_workload_v1/RESULTS.md](../../examples/sglang_moe_workload_v1/RESULTS.md).
The import-free study retains diagnostic geometry, request-realization,
payload, and stream-reduction rows. It qualifies nothing because the frozen
run is void: its short length-trace guard contradicted the established
`TraceLengths` cycling contract. Separate import-free tests cover those seams
and native open-loop submission. No live SGLang or GPU ran, and the change does
not move the adapter onto the simulated metric chain described above. SGL-4
remains the live comparison, SGL-13 remains the missing metric-chain link
(SGL-12 has since closed), WORK-4 remains virtual server ingress, and SGL-18
owns unsupported MoE geometry and mechanisms.

The composed realistic-deployment study is frozen by expectations-only commit
`dd026c0` and reported in
[examples/sglang_composed_deployment_v1/RESULTS.md](../../examples/sglang_composed_deployment_v1/RESULTS.md).
On 2026-08-14 the live in-process chain ran the same four arrival-gated
requests through 18 cells: two declared deployments, one whose eight ranks
share a host and whose 24 attention allreduces and 48 MoE all-to-alls stay on
NVLink, and one whose eight ranks are eight hosts at 400 and 100 Gbit/s, each
under the `off`, `lower` and `upper` arms of a named fixed-cost envelope and
under the `ideal` and `turing-cuda-graph` host arms. 357 scheduler steps, 864
reduced intervals and 9,312 `htsim_rnic` runs, all of them cross-node because
an all-intra-node step invokes no backend. All 11 fatal guards held, so the run
is not void; 2 of 2 scored exact relations and 5 of 6 scored behavioral
relations passed, in classes that are never summed, and all 8 scored relations
were genuine risk. The `cross400-off-ideal` cell reproduced the accepted
SGL-8 `ep8-400g` artifact to every published digit despite a declared
placement manifest, the w14d selection seam and a different driver, which is
what makes the other 17 cells readable as changes from a known baseline. The
headline is a bracket: in the ten cells whose arm charges a nonzero surcharge,
70.6 to 95.3 percent of the upper median step is one per-collective constant
never measured on this chain, 57.6 to 95.3 percent per individual step, and
moving from `off` to `upper` multiplies summed TTFT by 4.95x to 37.1x. Two of
the twelve enabled cells charge nothing for it, because the intra-node `lower`
arm resolves to a zero surcharge and moves only the NVLink endpoint rate. B4 failed on its declared-risky half, the clause that the live TTFT
sensitivity to link rate falls as the arm constant grows: it holds on the
per-step quantity the closed form governs under both hosts and breaks on the
live one under `turing`, because at the `off` arm the 100 Gbit/s cell runs 17
steps against the 400 Gbit/s cell's 19 and the extra batching removes more
queueing than the slower link adds. The intra-node against cross-node ordering
is undetermined: matched by arm name the intra-node deployment looks cheaper
everywhere, but the two envelopes do not share arms, and matched on the
per-collective constant the ratio brackets one and inverts at 30,128,029 ps
because the corrected inventory makes the intra-node cell pay 72 surcharges
per step against 48. Nothing was calibrated and nothing was closed. SGL-27
registers the declared tensor-parallel shard against the width SGLang actually
executed, and TRAF-42 registers the fixed-cost surface's fabric-shaped
self-description. SGL-26 stays open: this is the first live in-process run to
select a nonideal host profile and carry it to TTFT and TPOT, but the matrix
makes SGL-26's second clause, the fallback path a collective-free step would
take, unreachable rather than settled.

## Open tasks

Closed this milestone: SGL-1 (the worker, this module). SGL-2 (upstream
worker-class selection flag) closed as moot 2026-08-04: SGLang's plugin
framework (`sglang.srt.plugins` entry points plus `HookRegistry` `REPLACE`
hooks, run before scheduler construction) is a supported non-fork selection
seam, so no upstream flag is needed. SGL-23 (the owned per-step host cost
selector) closed 2026-08-14 with `select_sglang_host_model` and the frozen
replay study; its residuals are registered as SGL-24 (SGLang's own launch
count) and SGL-26 (live in-process selection) rather than kept open under the
closed id.

### Precision
- SGL-4 (Precision; P1; L) (remaining half): a paced-mode run checked against
  SGLang's own wall-clock metrics, a workload that actually exercises radix
  hits
  (repeated shared prefixes) and retraction under KV pressure, and a
  `launch_server` (HTTP) run in addition to the offline `Engine` smoke. Run
  after the calibrated compute table and CORE-3, since CORE-4 and CORE-5
  have landed, using the same commit, model, parallel configuration,
  request trace, seed and warm-up policy on
  silicon and in simulation. Stage single-GPU compute, eight-GPU intra-node,
  two-node rail-RNIC, offered-load, KV-pressure, chunked/retraction and
  mixed/bursty cases. Report p50 through p99.9 TTFT/TPOT and attributed queue,
  KV, kernel, collective, DMA, WQE/NIC, flow and control residuals. Calibrate
  early stages and reserve later stages as holdouts.
- SGL-9 (Precision; P1; L) (remaining after the CPU-oracle lifecycle slice):
  the v2 fallback now observes stock Radix prefix matching, token-slot
  allocation, radix eviction and decode retraction with request and slot IDs.
  Its current projection is slot IDs plus token counts. Extend it for CORE-3
  with stable pool identity, token intervals, layer/dtype/bytes, epoch,
  reference count, cause and correlation ID, prefix bind/touch, reads/writes,
  release/free and transfer. The identifying observables are the owning
  `RadixCache`, token-pool and request-pool objects. Acceptance requires exact
  event cardinality, identity and cause agreement against those objects, a
  direct comparison with VLLM-11, and byte preservation of the current v2 and
  oracle-disabled paths.
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
- SGL-16 (Precision; P1; M): replace the framework-oracle fallback's Granite
  model-order layer inference with stable layer IDs supplied by SGLang. The
  current surrogate cycles missing capture labels through the model's 24 MoE
  modules in execution order; the identifying observable is an explicit
  framework layer ID at the post-selection capturer. Freeze at least two
  prompt shapes and a preemption resume before changing it. Acceptance
  requires zero layer-label disagreements and byte-identical expert IDs,
  request outputs and KV events relative to the current qualified Granite
  fallback.
- SGL-22 (Precision; P1; M): confirm the transcribed sampled-row rule against
  a live SGLang scheduler. SGL-12 landed the rule and its exact
  `num_sampled` and `sampled_request_ids`, but the evidence is source
  transcription plus stub batches carrying the pinned attribute names; no
  running scheduler has been observed agreeing with it. The identifying
  observation is SGLang's own consumption of the worker's `next_token_ids`,
  i.e. the per-request `output_ids` growth in `process_batch_result`.
  Acceptance requires a live in-process run that exercises chunked prefill, a
  MIXED batch and a decode retraction, exact per-step agreement between the
  emitted sampled identity and the requests whose `output_ids` grew, and an
  unchanged compatibility stream.
- SGL-24 (Precision; P1; M): measure the per-step device-visible launch demand
  of SGLang's own model step, so the SGLang chain stops borrowing vLLM's. The
  current surrogate is the `[440, 567]` bracket enumerated statically from
  vLLM 0.26.0 sources for the pinned Granite MoE geometry in
  [examples/compute_fidelity_v1](../../examples/compute_fidelity_v1/expectations.md);
  SGLang's own model runner, its fused MoE path and the pump's unrolled
  `event_loop_normal` issue their own launches and nobody has counted them.
  The identifying observable is the count of device-visible kernel launches
  per model step at the pinned commit for one fixed geometry, enumerated from
  SGLang sources and confirmed against a CUPTI or Nsight Systems capture of a
  real decode step. Acceptance requires an SGLang-specific bracket, the signed
  error of the transferred vLLM bracket against it, and an unchanged ideal
  path.
- SGL-25 (Precision; P1; S): price the end-to-end study's sink-free control
  cell on the same model its sink cells price. The sink cells declare the
  2-byte, 4-resident-expert per-rank geometry
  (`examples/sglang_end_to_end_v1/run_study.py`, `_dims`), while the control
  cell falls back to the worker's own reader, which sees the run's
  `dtype="float32"` and, because expert parallelism is refused under SGL-18,
  all 32 experts resident (`simllm/adapters/sglang/worker.py`,
  `model_dims_from_sglang`). The identifying observables are the two
  `ModelDims` the two arms actually use and their resident weight bytes:
  553,654,272 bytes against 5,335,166,976 bytes, a 9.6x step-compute gap that
  makes the control's scheduler-step count incomparable with the sink cells'.
  A `dims` override hook on `configure` is the smallest candidate fix and it
  is a new seam; the expert-residency half belongs to SGL-18. Acceptance
  requires both arms to report identical per-rank geometry and resident bytes,
  with the accepted sink-cell artifacts unchanged.
- SGL-27 (Precision; P1; M): make a composed SGLang study's declared
  tensor-parallel shard equal the width the framework actually executed. The
  intra-node cell of
  [examples/sglang_composed_deployment_v1](../../examples/sglang_composed_deployment_v1/RESULTS.md)
  declares `tp_ranks=(0..7)` with the matching per-rank geometry, 2 attention
  heads, 1 KV head and dense intermediate 64, while SGLang itself ran
  `tp_size=1`, so every batching decision, every radix-cache outcome and the
  whole captured routing come from a width-one engine and only the pricing is
  width eight. This is the tensor-parallel analogue of SGL-25's
  expert-residency gap, and it is why that study reports its two topologies as
  two deployments rather than as one controlled locality experiment. The
  identifying observables are SGLang's own `tp_size` and the per-rank resident
  weight bytes the sink prices, 421,582,848 at the declared width against
  554,047,488 at the executed one, a difference that is exactly the
  132,120,576 bytes of tensor-parallel attention weight plus the 344,064 bytes
  of KV the narrower head count removes. Acceptance requires one live run
  whose declared shard equals the framework's own parallel configuration, with
  the accepted single-rank artifacts byte-identical when the declaration
  matches.

### Completeness

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
- SGL-13 (Completeness; P1; L): now that CORE-4 and CORE-5 have landed,
  project each
  simulated SGLang communicator `CollectiveWork` through the single runtime
  authority into `CompletionEvent`, `StepResult`, and TTFT/TPOT. Freeze a
  fixed-workload signed metric relation and quantitative band first. The
  disabled projection must preserve every accepted SGLang worker timestamp,
  token, record byte, and completion order exactly.
- SGL-17 (Completeness; P2; L): add the SGLang communicator's source-backed
  observed schedule as `ExecutionObservations`, including exact per-layer
  semantic sites, logical streams, submission and program order, event-wait
  dependencies, request correlation, and completion frontier for every
  supported call. Derive concurrency only from the active communicator and
  scheduler source, with no overlap percentage or compatibility-schedule
  inference. When the producer is disabled or absent, preserve the accepted
  worker records, event sidecar, sink calls, timestamps, tokens, and completion
  order exactly.
- SGL-18 (Completeness; P2; L): extend the strict single-GPU routed-MoE
  geometry reader to SGLang's distributed, redundant-copy, and unsupported
  single-GPU MoE families, including DBRX and QuantMixtral. Source per-rank
  expert ownership and expert tensor sharding from the active `moe_ep_size`,
  `moe_dp_size`, and `moe_tp_size`; represent shared experts, redundant
  physical copies, mixed dense/routed layer schedules, MLA, DBRX's nested
  `ffn_config`, multimodal wrapper compute, and quantized MoE weight widths
  before enabling families that require them. The single-GPU Granite,
  Mixtral, and Qwen3 MoE path remains the explicit supported baseline. Every
  unsupported sentinel combination must keep failing before it can be priced
  as dense. Acceptance requires exact per-rank active FLOPs and resident
  bytes, a supported end-to-end TTFT/TPOT change, and byte-identical dense and
  single-GPU baselines.
- SGL-26 (Completeness; P1; M): select a nonideal host profile in a live
  in-process SGLang run and carry it to TTFT and TPOT. `configure` already
  accepts a host model and `_validate_host_model_selection` already requires
  the adapter and the sink to agree
  (`simllm/adapters/sglang/worker.py`), but no live scheduler run has ever
  selected anything but `ideal`, so the nonideal branch of that agreement
  check is exercised only by fixture replay and unit tests. The identifying
  observation is one live pump run at the pinned commit whose emitted
  `StepResult` values carry the launch floor. Acceptance requires the live
  run to reproduce the replay study's per-step composition for the same
  records and the ideal arm of the same run to stay byte-identical to the
  accepted live artifacts. One divergence the seam newly makes reachable must
  be settled by that run rather than discovered in it: when a step carries no
  collective, `HtsimStepSink._plan_step` returns `None`, the worker's `_settle`
  falls back to `estimate_step_latency_ps`, and that path charges
  `max(C, N * g)` in raw picoseconds with no whole-nanosecond enclosure, so it
  disagrees with the sink's enclosed value by up to one nanosecond per step. A
  single-rank run takes the fallback on every step. Acceptance must state which
  of the two is authoritative and make the other agree or refuse.

### Uncategorized

- SGL-3: RadixCache-aware studies: prefix-hit rate and re-prefill traffic
  vs shared-prefix workload structure.
- SGL-5: logprobs, speculative decoding and the dLLM/hybrid modes are
  refused or unreachable rather than fabricated.
- SGL-6: overlap-schedule support (the scheduler-side dual-stream loop with
  its result queue; needs delayed-sample semantics in the fabricated
  result). Its observed host-side order and completion waits lower to graph
  dependencies; device overlap itself remains owned by CORE-4/TRAF-7.
- SGL-7: mamba/hybrid-attention models need the auxiliary-state pool the
  stub does not build; the stub currently builds a plain `ReqToTokenPool`
  only.
- SGL-10: capture and replay the supported model runner's CUDA stream/event,
  kernel and NCCL schedule as an `ExecutionGraph` template keyed by the same
  identity envelope as its compute table. Bind batch shapes, radix events and
  overlap-scheduler dependencies at runtime; never infer device concurrency
  from a single elapsed phase duration.
