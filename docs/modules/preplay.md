# simllm.preplay

Offline CPU inference oracle, a separate module beside the simulator: run
the actual model, slowly and exactly, to pre-compute each request's
data-dependent outcome, then replay those outcomes so the fast simulation
never has to fabricate them.

## Why

The simulation replaces model execution, so every decision the real model
would make is otherwise fabricated: output token ids (today one fixed
mid-vocabulary id), output length (today drawn from a workload
distribution), and MoE expert routing (today uniform). Those decisions feed
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
  `TransformersCpuRunner` is the implemented runner. Torch and Transformers
  load only when that execution entry is constructed.
- **Artifact.** `simllm-preplay-trace-v1`: versioned and
  provenance-carrying (model identity, sampling configuration and seed,
  capture host, schema version), keyed by a stable request identity. Bulk
  rows live off-repo, since large generated artifacts never live in this
  repo; the repo carries the schema and small fixtures only. Its JSONL
  header, request, forward-token and footer rows stream one completed request
  at a time. The writer protects existing paths by default and replaces one
  only when its caller passes `overwrite=True`. The strict reader rejects
  unknown fields, incomplete requests, duplicate identities, inconsistent
  route shapes and missing footers.
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
- **Replay.** The adapters serve the predefined token ids instead of a
  fabricated token and honor the oracle's stop position, so the scheduler
  sees the true completion step; the traffic layer consumes the captured
  routing for non-uniform all-to-all.
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

The vLLM replay adapter consumes that joined run in both `SimExecutor` and the
flagged skeleton worker. It validates the named trace bytes, binds vLLM's
version-pinned internal request identity to exactly one joined request, serves
tokens by scheduler-reported output index and requires the scheduler admission
limit to equal the oracle length. Its two-parameter study changed TTFT and TPOT
by the frozen exact relations, its live Granite smoke returned token ID 38 and
its absent-replay path reproduced all four accepted VLLM-13 JSONL hashes. The
chronology and evidence are recorded in
[the PLAY-3 results](../../examples/preplay_adapter_replay_v1/RESULTS.md).

Traffic projection remains open under PLAY-4. The independent framework CPU
runner is optional follow-up PLAY-6, and SGLang replay is the explicit PLAY-7
follow-up.

## Open tasks

Tags follow the legend in [backends.md](backends.md#open-tasks).

- PLAY-4 (Completeness; P1; M): supply captured routing to the traffic
  half. Define the versioned per-token expert-assignment projection of the
  trace and join it per request, in the form TRAF-2's expansion consumes.
  The projection must preserve the trace's prefill and decode phases and its
  input-token attribution, including the absence of a terminal-token
  forward pass, so traffic volume is derived only from executed routing.
  The boundary is explicit: this task owns the capture-side supply, TRAF-2
  keeps the traffic-side expansion that replaces uniform routing
  (including its EPLB placement-epoch handling), CORE-6 owns the graph
  representation of the resulting non-uniform per-pair sizes, and COMP-7
  consumes the same assignments for routed-load compute imbalance.
- PLAY-5 (Completeness; P1; M): pre-registered validation study. First,
  oracle consistency: the same requests and seed through the PLAY-1 runner
  and through an independent framework CPU run must agree on lengths, stop
  reasons and routing, with every divergence classified and none silently
  accepted; admissible causes are the recorded sampler difference and the
  numerics divergence the honesty rule predicts (kernel fusion and
  reduction order can flip a near-tie argmax, cascading into length and
  routing changes). Second, replay end-to-end: a replayed run's
  scheduler-visible completions must land exactly at the oracle lengths,
  and its all-to-all sizes must match the captured routing; freeze the
  expectations in their own commit before implementation per the
  development process.
- PLAY-6 (Completeness; P2; L): add an optional framework CPU backend runner
  that captures the same artifact through vLLM or SGLang on CPU, exercising
  the deployment framework's sampler. The Transformers runner remains the
  supported baseline and must stay byte-identical when no framework runner
  is selected. Missing framework dependencies and unsupported CPU backends
  must be rejected before a trace writer opens.
- PLAY-7 (Completeness; P2; M): consume joined replay runs in the SGLang
  adapter. Serve each request's predefined token IDs, pin scheduler-visible
  completion to the oracle length, retain the speculative and structured
  refusal boundaries that apply there, and prove an identity off mode against
  the accepted fabricated-token baseline. Add an in-process live smoke before
  claiming the path is live-reachable.
