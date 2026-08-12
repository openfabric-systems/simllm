# Serving-framework CPU oracle expectations

This document freezes the framework-oracle study before its implementation,
first source build, first serving-framework model execution, or first
Transformers comparison run. CPU time is not an outcome in this study. The
decision is whether a real serving framework can supply request outcomes,
post-selection expert dispatch, and scheduler-owned paged-KV events without a
reconstruction standing in for any of those authorities.

The repository source at the freeze boundary is commit
`6973bd0e3ed6091c403c7055ee01c2d8ae0ae970`. The vLLM source is the official
`v0.26.0` archive, whose observed SHA-256 is
`c1ded7e5af7e7fd27c6abcce1ecfaf795bc927a0f747f10f1e5a7a13f2c4a6a9` and
whose observed tag commit is `568afb3a13806beb53bb2e6bd518269357b237c0`.
The SGLang evidence is authored against source commit
`8f2a3ad6d7d68c58ae65b61a75bb2115449addca`. The result record will report the
source commit it actually observed without requiring a live checkout to equal
that authored-against value.

## Source audit and authority seams

The following sources were read before this freeze. File hashes are checked by
the registered harness, but source identity is a fatal unscored qualification
guard rather than behavioral evidence.

### vLLM 0.26.0

- `docs/getting_started/installation/cpu.x86.inc.md:4-15,69-127,149-153`
  documents the x86 CPU target, the `VLLM_TARGET_DEVICE=cpu` source-build
  procedure, GCC 12.3 or newer, limited AVX2 support, and the AMD Zen 4 or newer
  runtime recommendation.
- `cmake/cpu_extension.cmake:123-144` makes GNU C++ 12.3 a configure-time
  requirement and defines both AVX-512 and AVX2 extension variants. The host
  audit observed GCC 12.2.1, an AMD Zen 2 processor, AVX2, and no AVX-512.
- `vllm/v1/worker/cpu_worker.py:33-104,120-183` calls
  `torch.ops._C.init_cpu_memory_env`, selects the CPU device, initializes the
  distributed environment, seeds execution, and constructs `CPUModelRunner`.
  The installed CUDA wheel is vLLM 0.26.0 with Torch 2.11.0+cu130 and does not
  export that operator. This is a pre-run environment observation, not a
  source-build result.
- `vllm/model_executor/models/granitemoe.py:70-136` passes Granite router
  logits to the model's fused expert layer.
- `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py:122-166,242-285,
  398-405,430-435` returns expert IDs and weights from `select_experts` and
  passes those exact tensors to the CPU expert implementation. A capture after
  `select_experts` therefore observes the dispatch used by the expert kernel.
  A hook on router logits followed by a second top-k operation does not.
- `vllm/v1/core/kv_cache_manager.py:207-281,283-514,548-554` owns prefix
  matching, slot allocation, free, and eviction. `vllm/v1/core/block_pool.py:
  198-225,702-761` owns cached-block touch, free, and eviction. The scheduler's
  preemption path frees the request's blocks, resets computed-token state, and
  emits the preempted request identity. These are the only admissible vLLM KV
  observation seams.

### SGLang

- `python/sglang/srt/utils/common.py:188-210` and
  `python/sglang/srt/overrides.py:1980-1990` define the CPU engine selection and
  the Torch-native fallback used when Intel AMX is unavailable.
- `python/sglang/srt/models/granitemoe.py:32-89` passes the `TopKOutput` made by
  the model's `TopK` layer directly to `FusedMoE`.
- `python/sglang/srt/layers/moe/topk.py:557-573,1829-1845` runs
  `select_experts` on CPU and captures the resulting expert IDs at the common
  post-selection site. `python/sglang/srt/state_capturer/routed_experts.py:
  24-143` stores those actual IDs in scheduler token slots, and
  `python/sglang/srt/managers/scheduler_components/batch_result_processor.py:
  115-158` returns them per request. The public `return_routed_experts` path is
  therefore preferred over a new router-logit hook.
- `python/sglang/srt/managers/tp_worker.py:537-624` constructs the real forward
  batch, calls the stock model runner, and leaves token sampling to that
  runner. A qualifying CPU oracle must reach this worker, not SimLLM's
  `SimTpModelWorker` replacement.
- `python/sglang/srt/mem_cache/allocation.py:303-370,539-590` performs the
  paged extend and decode allocations and writes their real locations into the
  request-to-token pool. `python/sglang/srt/mem_cache/radix_cache.py:352-432,
  562-590` performs prefix matching, insertion, and eviction.
  `python/sglang/srt/managers/schedule_batch.py:2734-2782` returns the exact
  requests retracted under decode pressure. These calls are the admissible
  detailed KV observation seams. Response fields for cached tokens and
  retraction counts are independent scheduler projections used for
  reconciliation, not alternate mutable authorities.

## Runtime fallback ladder

The ladder is ordered by semantic fidelity and stops at the first qualifying
runtime.

1. Create an isolated Python environment below the configured external run
   directory and attempt the official vLLM 0.26.0 CPU source-build procedure
   with `VLLM_TARGET_DEVICE=cpu`. The first attempt uses the audited system
   compiler so its exact result is preserved. If the only blocker is the
   documented compiler floor, create a newer compiler toolchain below that
   same run directory and retry without changing the source. No command may
   modify the repository's installed vLLM environment.
2. If the build, import, CPU worker initialization, model load, or authority
   qualification fails, retain the first and deepest failure texts and try
   SGLang's stock CPU engine from the separately configured environment.
3. Only if both framework paths fail may the report survey another CPU MoE
   runtime. Producing tokens is insufficient. A candidate qualifies only if it
   exposes its actual post-selection expert dispatch and scheduler-owned paged
   KV allocation, prefix hits, evictions, and preemptions per request.

The system-compiler vLLM attempt is predicted to stop at the GCC 12.3 floor.
The local-toolchain retry is at genuine risk of a later Zen 2 or AVX2 runtime
failure. SGLang is predicted to pass its CPU-engine import and Torch-native
MoE path because its audited fallback does not require AMX. These predictions
do not turn a failure into a pass. The report must preserve the exact command,
exit status, deepest reached source boundary, exception type, failure text,
and whether any partial trace existed.

## Framework qualification

A vLLM observation qualifies only if all of these fatal gates pass:

- the built distribution reports version 0.26.0 and target device CPU;
- `torch.ops._C.init_cpu_memory_env` exists;
- the engine reaches `CpuPlatform`, stock `CPUWorker`, `CPUModelRunner`, and
  vLLM's `GraniteMoeForCausalLM` with all model parameters on CPU;
- expert IDs come from the `select_experts` result handed to the CPU expert
  implementation, never from a second top-k operation;
- prefix, allocation, eviction, and preemption events come from the v1 KV
  manager, block pool, and scheduler objects that own those decisions;
- vLLM remains the authority for sampling, output token IDs, and finish reason.

An SGLang observation qualifies only if all of these fatal gates pass:

- `SGLANG_USE_CPU_ENGINE=1`, device CPU, and the stock serving engine are
  active;
- the engine reaches stock `TpModelWorker`, the stock model runner, and
  SGLang's Granite MoE model with no `SimTpModelWorker` replacement enabled;
- returned expert IDs originate from SGLang's post-selection
  `RoutedExpertsCapturer` and reconcile with one complete row per executed
  input token and MoE layer;
- allocation, prefix-hit, eviction, and retraction observations come from the
  stock allocation, `RadixCache`, and `ScheduleBatch` calls listed above;
- SGLang remains the authority for sampling, output token IDs, and finish
  reason.

Qualification failure is a blocker, not a behavioral pass and not an oracle
divergence. Partial output from a disqualified runtime is retained as debug
evidence but never compared or written as a valid trace.

## Frozen model, requests, and parameters

- Model: `ibm-granite/granite-3.0-1b-a400m-instruct`
- Revision: `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`
- CPU dtype: float32
- Torch threads: eight
- Engine seed: 173
- Sampling: greedy argmax. The seed is supplied identically but consumes no
  random draw in this mode.
- Offline controls: `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
- Router shape: 24 ordered layers, top-k 8, 32 experts
- KV page size: the framework's reported page size, never a SimLLM substitute

The comparison uses the existing three PLAY-5 requests and the same tokenizer
and chat formatting on both runners.

| Request | Prompt | Maximum output | Stop string | Source rank |
|---|---|---:|---|---:|
| `eos-brief` | `Reply with exactly one word: OK` | 16 | none | 0 |
| `length-cap` | `Continue this sequence with ten more integers: 1 2 3` | 1 | none | 1 |
| `stop-string` | `Reply with exactly SIMLLM_STOP and no other text` | 16 | `SIMLLM_STOP` | 0 |

The Transformers oracle executes first and supplies no framework-side state.
The qualifying framework receives the exact resulting prompt token IDs, not a
second independently tokenized prompt. The accepted earlier greedy trace has
output lengths `(3, 1, 5)` and normalized finish reasons
`(eos, length-cap, stop-string)`. Those values are prior context, not frozen
observations for this run. The live framework and live Transformers outputs
must establish the comparison independently.

The KV workload has two additional families:

- Cold then warm prefix: submit the exact prompt
  `Summarize in one word: cached framework oracle prefix.` twice in sequence
  under request IDs `prefix-cold` and `prefix-warm`, with maximum output one.
- Capacity pressure: run four requests concurrently with prompts
  `Count upward slowly from zero for pressure request A:`, then B, C, and D,
  each with maximum output 16. Repeat the workload with framework token
  capacities 64 and 256, page-aligning only through the framework's own
  configuration.

Request policy and KV capacity are the two varied parameter families. The
prefix state adds a cold versus warm within-family comparison. The framework
engine is recreated between the 64-token and 256-token cells, so no cache
state crosses cells.

## New trace contract

The implementation is expected to add `simllm-preplay-trace-v2` as a separate,
strict, canonical JSONL contract. It records:

- framework name, framework version, observed source identity,
  authored-against source identity, model and tokenizer provenance, CPU dtype,
  seed, sampling, page size, top-k, expert count, and MoE layers;
- each request's input token IDs, output token IDs, explicit output length,
  output text, normalized finish reason, matched stop string, and request
  policy;
- each executed prefill and nonterminal decode input token, with one ordered
  list of post-selection expert IDs per layer and an explicit
  `observed-dispatch` source marker;
- one globally sequenced KV event stream with request identity where the
  framework supplies it, event kind, framework step or call sequence, page or
  slot identities, and token count. Event kinds cover allocation, prefix hit,
  eviction, preemption, and release. A framework-global eviction may have no
  request identity, but it may not be silently assigned to a request;
- a footer whose request, dispatch, and event counts reconcile exactly.

The common contract deliberately omits reconstructed gate weights. An
observed framework may add actual post-selection weights only through a later
versioned field. The traffic decision in this study depends on selected
destinations, not weights.

The v2 reader rejects unknown fields, missing or duplicate identities,
nonmonotonic event sequences, impossible action-specific fields, output-length
disagreement, missing prompt or decode dispatch rows, duplicate layer rows,
out-of-range experts, and footer disagreement. These are fatal structural
guards.

The existing `simllm-preplay-trace-v1` parser, writer, public constants, and
tracked byte fixture stay byte-identical. The new reader does not reinterpret
v1 router-logit reconstruction as observed dispatch. Missing optional
framework dependencies and unsupported CPU backends are rejected before a v2
writer opens.

## Outcome relation

Each raw framework response is compared with the raw Transformers observation
before either side is admitted through the new strict reader. For each of the
three requests, the complete output-token sequence and normalized finish
reason must agree exactly. Output length is reported but not scored separately
because exact token-sequence equality already entails it. A difference fails
the row even if its text detokenizes identically.

Routing is aligned by request, phase, input-token index, input-token ID, and
layer. Differences use this exhaustive taxonomy:

- `order-only`: expert tuples differ but their sets are equal;
- `expert-id-only`: selected sets differ but the exact destination-byte vector
  under the frozen placement is unchanged;
- `byte-changing`: the destination-byte vector changes;
- `output-cascade`: the routing row occurs only after an earlier output-token
  disagreement changed the causal input context;
- `unaligned`: either oracle lacks the matching token or layer row.

`unaligned` is always fatal. `output-cascade` does not excuse the output-row
failure. `order-only` and `expert-id-only` are real dispatch differences and
are reported even though they do not change this traffic projection. Every
non-exact row must have exactly one class, and classification coverage is a
fatal unscored guard rather than a behavioral pass.

For the traffic projection, experts 0 through 15 belong to EP rank 0 and
experts 16 through 31 belong to EP rank 1 at every layer. Hidden width is
1,024 and dtype width is two bytes, so one activation vector is 2,048 bytes.
For source rank `s`, token `t`, layer `l`, and destination `d`, the independent
projection is:

```text
dispatch_bytes(s,d,t,l) = 2048 *
    indicator(s != d and any selected expert is owned by d)
combine_bytes(d,s,t,l) = dispatch_bytes(s,d,t,l)
```

Changed all-to-all bytes are the sum of absolute framework versus Transformers
differences at token-layer-pair granularity across dispatch and combine. This
granularity prevents opposite changes from cancelling in an aggregate table.
The smallest nonzero quantum is 4,096 bytes, one 2,048-byte dispatch plus its
combine. Any nonzero changed-byte result is therefore material. The frozen
expected outcome is zero changed bytes for each request. A measured value of
4,096 bytes or more fails that expected relation and is reported as a material
finding, not softened into an accepted warning.

## KV relations

For the repeated prefix request, the cold hit is expected to be zero. The warm
hit must be greater than the cold hit and at least the largest whole-page
prefix not exceeding `prompt_token_count - 1`. This is one scored live
relation. Allocation identities and exact page numbers are observations, not
frozen values.

For the pressure workload, the 64-token cell must observe at least one real
cache eviction and at least one real request preemption or SGLang retraction.
The 256-token cell must observe zero preemptions, and its eviction count must
not exceed the 64-token count. This composite is one scored live relation.
The framework response's per-request cached-token and preemption counters must
reconcile with the detailed event stream, but that conservation check is fatal
and unscored.

These direction and shape expectations can fail if admission serializes all
four requests, if the framework reserves more KV than its reported capacity,
if finished requests are not cacheable, or if the capture is attached to the
wrong scheduler boundary. The result must explain the actual mechanism rather
than editing the expected direction after observation.

## Evidence accounting and entailment check

If a framework qualifies, the behavioral headline has eight scored live
instances:

- three complete per-request outcome comparisons;
- three per-request changed-all-to-all byte relations;
- one cold-versus-warm prefix relation;
- one 64-versus-256 capacity-pressure relation.

The expected genuine-risk fraction is `8/8 = 100%`. All eight are evaluated
against raw framework and Transformers observations before source hashes,
schema validation, footer reconciliation, or other fatal gates. No earlier
fatal oracle entails any scored result.

Output length is not an additional scored instance because output-token
equality entails it. Exact expert-ID fractions, routing taxonomy counts, and
per-field output agreement are reported as diagnostic fractions but are not
added to the headline. Source hashes, framework qualification, fixed model and
placement, configured capacities, page alignment, request and token
conservation, classification coverage, event-counter reconciliation, schema
strictness, canonical round trips, v1 byte preservation, and no writer on
rejection are fatal unscored guards. Author-defined prompts, seed, capacity
values, and ownership are run configuration. Unit tests and native build
executables remain a separate evidence class.

If no framework qualifies, zero scored instances execute. Blocked rows stay
outside the denominator, and the report names the exact missing host or runtime
requirement. A runtime that returns only output tokens earns no partial pass.

## Registered command and dry run

Source machine-local configuration first. The single registered invocation
is:

```text
"${SIMLLM_VLLM_PYTHON:?configure SIMLLM_VLLM_PYTHON}" \
  examples/framework_oracle_v1/run_study.py \
  --cache-dir "${HF_HOME:?configure HF_HOME}" \
  --vllm-source "${SIMLLM_VLLM_026_SOURCE:?configure SIMLLM_VLLM_026_SOURCE}" \
  --sglang-source "${SIMLLM_SGLANG_SOURCE:?configure SIMLLM_SGLANG_SOURCE}" \
  --sglang-python "${SIMLLM_SGLANG_PYTHON:?configure SIMLLM_SGLANG_PYTHON}" \
  --run-dir "${SIMLLM_FRAMEWORK_ORACLE_RUN_ROOT:?configure SIMLLM_FRAMEWORK_ORACLE_RUN_ROOT}"
```

Before this freeze, the same command with `--check-only` ran against the
untracked parser and literal-audit harness. It checked the complete option
surface, external run location, model snapshot, both source trees and hashes,
SGLang Python, source provenance shape, model and seed literals, evidence
denominator, capacity family, and the 4,096-byte materiality quantum. It
reported GCC 12.2.1 and wrote zero artifacts. It imported no SimLLM target
implementation, constructed no model, initialized no serving engine, compiled
nothing, and executed no behavioral relation.

Build trees, virtual environments, logs, framework sidecars, traces, and result
JSON remain below the configured external run directory. Only the expectation,
implementation, tests, concise result report, and intentionally small fixtures
are tracked.

## Closure scope frozen before observation

PLAY-5 currently requires the same requests and seed through a vLLM 0.26.0 CPU
build, with lengths, stop reasons, and routing compared and all divergences
classified. A qualifying vLLM result maps directly to that clause. A blocked
vLLM attempt plus a qualifying SGLang result does not by itself demonstrate the
registered vLLM-specific clause, even though it can implement the maintainer's
reframed framework-oracle goal. Residual vLLM work must remain registered under
an allocated ID rather than being silently called complete.

PLAY-6 requires an optional vLLM or SGLang CPU runner, the Transformers runner
as a byte-identical baseline, and rejection before a writer opens. The new v2
trace, qualifying runner, dependency rejection, and v1 fixture map to those
clauses. Dispatch and KV claims map only if positive real-framework events are
captured and reconciled.

Any acceptance clause not demonstrated by raw evidence remains open or moves
to PLAY-8, PLAY-9, PLAY-10, VLLM-22, VLLM-23, or SGL-16 with category, priority,
difficulty, surrogate, identifying observable, and a quantitative acceptance
test. After any closure, the result round must reconcile the task ledger and
run the contradiction sweep without editing the top-level index documents.
