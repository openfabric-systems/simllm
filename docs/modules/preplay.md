# simllm.preplay

Offline CPU inference oracle, a separate module beside the simulator: run
the actual model, slowly and exactly, to pre-compute each request's
data-dependent outcome, then replay those outcomes so the fast simulation
never has to fabricate them.

## Why

The simulation replaces model execution, so every decision the real model
would make is otherwise fabricated: output token ids (one fixed
mid-vocabulary id on the absent-replay path), output length (drawn from a
workload distribution unless a joined replay run pins it), and MoE expert
routing (uniform only when no `RoutedMoeSupply` is attached). Those
decisions feed
back into exactly the things the simulation promises to keep real: the stop
position drives scheduler-visible completion and batch composition, token
identity drives prefix-cache hits, and routing drives the all-to-all
traffic shape. The pre-play oracle runs the real model once on CPU (very
slow, but exact and GPU-free) and freezes those decisions per request.
Replay then keeps the frameworks' real control behavior while every
request's route, length and output are predefined, which is what lets the
skeleton coupling mode (VLLM-13) run its name-mirrored virtual functions
and still produce a real simulation.

## Design

- **Runner.** Given the model identity (name, revision, dtype, tokenizer
  hash) and a request set, run inference on CPU with greedy decoding or
  seeded sampling. Capture per request: the output token ids, the output
  length and stop reason, plus per forwarded input token, per MoE layer, the
  top-k expert assignments. This includes every prompt token in the prefill
  forward and every nonterminal generated token forwarded during decode.
  `TransformersCpuRunner` remains the baseline. `VllmCpuRunner` captures the
  stock vLLM sampler, post-selection CPU expert dispatch, and scheduler-owned
  paged-KV decisions. `SglangCpuRunner` is the qualified fallback. Framework
  packages load lazily only when their runner executes.
- **Artifacts.** `simllm-preplay-trace-v1` remains the Transformers artifact.
  It is versioned and provenance-carrying, keyed by a stable request identity,
  and protected by a byte fixture. `simllm-preplay-trace-v2` is a separate
  strict framework-observation contract. It records explicit per-request
  outputs and finish reasons, post-selection expert IDs, framework page or
  slot identities, globally sequenced allocation, prefix-hit, eviction,
  preemption and release events, and independent observed versus
  authored-against framework source identities. Its canonical reader rejects
  unknown fields, incomplete or duplicate identities, inconsistent routing,
  nonmonotonic event sequences and footer disagreement. Bulk rows remain
  outside Git; the repository carries schemas and small fixtures only.
- **Join.** A workload arrival realization assigns each request its
  arrival timestamp at the framework entry point. The join produces
  per-request tracking records in the core bookkeeping (request identity,
  arrival time, predefined route, length and output), so a replay run
  knows every request's outcome up front without recomputing it. The strict
  `simllm-preplay-replay-run-v1` record names the resolved trace path and
  SHA-256, while each request carries a
  `simllm-preplay-routing-reference-v1` pointer to its trace row. One atomic
  `RequestBookkeeper.extend` call pins all framework-request objects only
  after the complete join validates.
- **Replay.** The vLLM adapter serves v1 predefined token ids instead of a
  fabricated token and honors the oracle's stop position, so the scheduler
  sees the true completion step; the traffic layer consumes the captured
  routing for non-uniform all-to-all. Each scheduled request identity remains
  attached to its routed token slice and becomes a read-only partition of the
  aggregate physical pair table. Direct and execution-graph GOAL renderers
  fail closed unless every request, layer, phase and directed pair agrees with
  that routed authority under the selected placement. Joining the observed v2
  request and KV records is PLAY-8. SGLang replay is PLAY-7.
- **Honesty rule.** A CPU run is one realization, not the deployment's
  exact token stream: CPU and GPU numerics differ, so sampled ids can
  diverge between the oracle and silicon. Greedy or fixed-seed sampling is
  the default and every trace records which was used. The claim is
  structural realism (lengths, stop reasons, routing pattern, cache-hit
  structure), never bit-exact parity with a GPU serve.

Routing is attributed to the executed forward pass that takes token `t` as
input and produces logits for token `t+1`. Prefill records exactly mirror the
prompt input token sequence. Decode records exactly mirror
`output_token_ids[:-1]`: generated token `i` is forwarded to produce token
`i+1`, while the terminal generated token is never forwarded and therefore
has no routing record. Traffic consumers must use this convention rather
than treating routing as a property of the token just produced.

The Transformers capture recomputes assignments with top-k selection and
softmax over hooked router logits. It does not observe the model's expert
dispatch. Discovery keys on Transformers-internal layer and router module
names, so both discovery and recomputation are version-sensitive. They have
been source-verified only for Transformers 5.14.1 and the pinned Granite
snapshot used by the PLAY-1 study.

The framework v2 path does observe dispatch. For vLLM 0.26.0 on CPU, the
stock response capturer allocates its routed-expert result but the monolithic
CPU MoE path does not populate it. The observation plugin therefore captures
the exact expert-ID tensor returned by `cpu_fused_moe.select_experts`
immediately before the unchanged expert kernel consumes it, then passes that
same tensor into vLLM's stock request capturer. KV hooks wrap the stock v1
manager, block pool and scheduler methods after their decisions; they never
replace allocation, eviction or preemption policy.

Module rules: nothing in the simulator core imports this module; the heavy
dependencies (torch, transformers or a framework CPU backend) import
lazily and only here; adapters and traffic consume the artifact through
versioned schemas rather than this module's internals, extending the core
forms where a representation is missing (CORE-6 carries only per-pair
sizes, so the per-token routing projection needs its own versioned form,
defined by PLAY-2 and PLAY-4).

## Status

PLAY-1, PLAY-2 and PLAY-3 are implemented. `simllm.preplay` provides the strict
`simllm-preplay-trace-v1` schema, a request-streaming writer, a strict reader
and a pinned Transformers CPU runner with greedy and seeded sampling. The
runner records EOS, length-cap and stop-string termination plus every
prompt token's prefill routing and every executed nonterminal decode token's
top-k expert IDs and normalized gate weights at each MoE layer. The Granite
3.0 1B A400M study passed seeded byte determinism, all three stop modes,
exact schema round trips and fatal routing-shape and attribution checks; its
evidence is recorded in [the PLAY-1 results](../../examples/preplay_trace_v1/RESULTS.md).

The arrival join adds strict replay-run and routing-reference schemas,
canonical run-record I/O and an atomic projection into existing core
bookkeeping. Its one-request and two-request study passed exact field
projection, a 7,000 ps arrival shift, trace-hash authority, cardinality scaling
and rollback gates; the evidence is recorded in
[the PLAY-2 results](../../examples/preplay_arrival_join_v1/RESULTS.md).

The joined routing supply now has its own strict
`simllm-routed-experts-v1` projection. It preserves joined request order,
prefill and decode input-token attribution, source-trace schema and hash,
and the exact per-layer expert identities while deliberately omitting gate
weights that the traffic expansion does not consume. Its canonical reader
rejects unknown fields and inconsistent phase, token, layer or expert shapes.
The captured-routing Granite oracle passed both tracked and full sources,
including six real decode forwards, exact canonical hashes and stable
request association under reversed join order. The same tracked assignments
then reached exact live traffic and JCT, closing PLAY-4; see
[the routing supply results](../../examples/routed_supply_v1/RESULTS.md).

The vLLM replay adapter consumes that joined run in both `SimExecutor` and the
flagged skeleton worker. It validates the named trace bytes, binds vLLM's
exact scheduler request identity to one joined request, serves tokens by the
scheduler-reported output index and requires the scheduler admission limit to
equal the oracle length. Replay also rejects an early EOS or stop token and a
prompt-plus-oracle length beyond `max_model_len` before a step settles. Its
real-scheduler study let vLLM choose both schedules: replay moved `r0`'s finish
from step 3 to step 0 and changed TTFT and TPOT by the frozen exact relations.
The live Granite smoke returned token ID 38 under the same external and
internal request ID. The absent-replay path is protected by a tracked JSONL
byte fixture in pytest, and `reset_configuration()` separates independent
in-process runs. The chronology and evidence are recorded in
[the PLAY-3 results](../../examples/preplay_adapter_replay_v1/RESULTS.md).

PLAY-4 is complete. The PLAY-5 routed replay half was validated as of
2026-08-11: both live bandwidth cells returned exact oracle completions, every
captured all-to-all table matched its closed form, and all 13 executed
completion, routed-stream, TTFT and TPOT relations passed. A post-specified
raw-trace recomputation then derived all ten GOAL tables without reading the
routing projection and matched 10/10. This corrects the primary evaluator's
overstated independence claim while leaving its result unchanged. The
chronology and exact evidence are recorded in
[the PLAY-5 results](../../examples/preplay_validation_v1/RESULTS.md).

PLAY-5 and PLAY-6 are complete as of 2026-08-12. An isolated vLLM 0.26.0 CPU
source build qualified on the AVX2 host with task-local GCC 13.4, CMake 4.4.2
and numactl headers. It selected `CpuPlatform`, exported
`init_cpu_memory_env`, constructed the stock CPU worker and model runner,
loaded Granite entirely on CPU, and showed a zero-byte CUDA allocation delta
in both study cells. All three requests matched the Transformers baseline on
complete output token sequence, length and finish reason. All 1,512 aligned
token-layer routing rows differed in tuple order but not expert set, producing
zero changed all-to-all bytes. The prefix relation passed. The registered
pressure relation failed because vLLM admitted or serialized the four pressure
requests without preempting any of them, even though the 64-token cell evicted
eight cached blocks. The exact build ladder, `7/8` genuine-risk result and
strict v1/v2 evidence are in
[the framework-oracle results](../../examples/framework_oracle_v1/RESULTS.md).

The independent `VllmCpuRunner` and fallback `SglangCpuRunner` are opt-in.
Missing packages and unsupported worker, model, device, dispatch and KV seams
are rejected before the canonical v2 writer runs. The Transformers runner and
v1 fixture remain byte-identical. SGLang replay remains PLAY-7.

PLAY-11 is complete. Captured MoE traffic retains each scheduled request ID as
a read-only partition of the aggregate directed-pair demand through direct and
execution-graph GOAL rendering. The gate checks every request, layer, dispatch
or combine phase and directed pair before backend execution. Across one, two
and three co-scheduled requests and two placements, all exact tables and
physical GOAL hashes matched. Four synthetic and one Granite
aggregate-preserving request permutations left the aggregate check unchanged
but were rejected by the per-request gate. The result remains byte fidelity,
not per-request time attribution; see
[the PLAY-11 results](../../examples/per_request_fidelity_v1/RESULTS.md).

## Open tasks

Tags follow the legend in [backends.md](backends.md#open-tasks).

- PLAY-7 (Completeness; P2; M): consume joined replay runs in the SGLang
  adapter. Serve each request's predefined token IDs, pin scheduler-visible
  completion to the oracle length, retain the speculative and structured
  refusal boundaries that apply there, and prove an identity off mode against
  the accepted fabricated-token baseline. Add an in-process live smoke before
  claiming the path is live-reachable.
- PLAY-8 (Completeness; P1; L): join `simllm-preplay-trace-v2` into the live
  replay path. Bind its observed per-request outputs and expert routing to the
  existing replay identities, retain the framework scheduler as the sole KV
  authority, and reconcile its per-request KV event stream with the oracle
  record. The explicit v1 join and absent-replay paths must remain byte- and
  timestamp-identical when v2 is not selected.
- PLAY-9 (Precision; P1; M): replace the first pressure-study workload, which
  vLLM admitted or serialized without pressure-group preemption, with a
  pre-registered family that distinguishes admission deferral from true
  scheduler recompute. The identifying observations are scheduler preemption
  counters and the detailed event ledger at two reported capacities.
  Acceptance requires at least one low-capacity pressure-group preemption,
  none at high capacity, exact event-to-counter reconciliation, and retention
  of the published negative `64` versus `256` result as chronology.
