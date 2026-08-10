# simllm.preplay (design)

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

## Design (this module is design-only today; PLAY-1 lands the first slice)

- **Runner.** Given the model identity (name, revision, dtype, tokenizer
  hash) and a request set, run inference on CPU with greedy decoding or
  seeded sampling. Capture per request: the output token ids, the output
  length and stop reason, and per token, per MoE layer, the top-k expert
  assignments.
- **Artifact.** `simllm-preplay-trace-v1`: versioned and
  provenance-carrying (model identity, sampling configuration and seed,
  capture host, schema version), keyed by a stable request identity. Bulk
  rows live off-repo, since large generated artifacts never live in this
  repo; the repo carries the schema and small fixtures only.
- **Join.** A workload arrival realization assigns each request its
  arrival timestamp at the framework entry point. The join produces
  per-request tracking records in the core bookkeeping (request identity,
  arrival time, predefined route, length and output), so a replay run
  knows every request's outcome up front without recomputing it.
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

Module rules: nothing in the simulator core imports this module; the heavy
dependencies (torch, transformers or a framework CPU backend) import
lazily and only here; adapters and traffic consume the artifact through
versioned schemas rather than this module's internals, extending the core
forms where a representation is missing (CORE-6 carries only per-pair
sizes, so the per-token routing projection needs its own versioned form,
defined by PLAY-2 and PLAY-4).

## Status

Design-only: this doc specifies the oracle, the trace artifact and the
replay join. No code exists yet.

## Open tasks

Tags follow the legend in [backends.md](backends.md#open-tasks).

- PLAY-1 (Completeness; P1; L): implement the CPU inference runner and the
  `simllm-preplay-trace-v1` artifact. A transformers-backed CPU run with
  greedy and seeded-sampling modes captures output token ids, stop reason
  (EOS, length cap, stop string) and per-token per-MoE-layer top-k expert
  ids; the writer streams rows to disk so a long request set never holds
  the trace in memory (bulk rows live off-repo, since large generated
  artifacts never live in this repo; the repo carries the schema and small
  fixtures only), and the strict reader validates schema, provenance
  and per-request completeness. Record the sampling mode, seed and model
  identity in every trace. An optional framework-CPU-backend runner (the
  same capture through vLLM or SGLang on CPU) is a later variant of this
  task, valuable because it exercises the deployment's own sampler.
- PLAY-2 (Completeness; P1; M): join arrivals with the trace into the core
  bookkeeping. Given a workload arrival realization and a trace, emit
  per-request tracking records (request identity, arrival timestamp at the
  framework entry point, predefined output length, stop reason, token ids
  and routing reference) into `RequestBookkeeper`, so the run's request
  futures are pinned before the first scheduler step. Joining must fail
  loudly on a request missing from the trace, and the run record names the
  trace it replayed.
- PLAY-3 (Completeness; P1; M): replay predefined outputs through the
  adapters. `SimExecutor`, the skeleton coupling mode and
  `SimTpModelWorker` serve the trace's token ids instead of the fabricated
  mid-vocabulary token and stop each request at the oracle's stop
  position, so scheduler-visible completion, batch composition and
  prefix-cache content match the real model's behavior. The
  fabricated-token baseline is the preserved off path: without a joined
  trace, behavior stays byte-identical to today's accepted runs. With a
  joined trace, plain generation no longer depends on a fabricated id; the
  speculative-decoding and structured-output refusals (VLLM-8) stay in
  place until their semantics are modeled explicitly.
- PLAY-4 (Completeness; P1; M): supply captured routing to the traffic
  half. Define the versioned per-token expert-assignment projection of the
  trace and join it per request, in the form TRAF-2's expansion consumes.
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
