# simllm.workload

Request generation as a queueing model: when requests arrive and what they
look like. Prefix-hit probability, cache-miss re-prefill traffic and vRAM
pressure must be emergent from the workload, never assumed, so prompts are
synthetic token-ID sequences with controllable shared-prefix structure.

## Interface

- `PoissonArrivals(rate_rps, seed)`: open-loop Poisson arrivals, reproducible
  by seed; `times(n)` or infinite iteration.
- `TraceArrivals(path)`: replay absolute arrival times from a file.
- `FixedLengths` / `LogNormalLengths(mean, sigma, seed)` / `TraceLengths`:
  request length samplers (token or byte counts), reproducible by seed.
- `RequestAdmissionGate(clock, bookkeeper, mode)`: expose framework requests
  in bookkeeping order. `ARRIVAL_GATED` withholds the next request until its
  creation timestamp is eligible on the shared virtual clock;
  `ALL_AT_ONCE` is the default identity mode and exposes every request without
  advancing time or mutating bookkeeping. The supplied framework callback
  remains the only handoff into `add_request`.
- `realize_generation_requests(...)`: freeze caller IDs, arrival providers,
  prompt/output length providers, and a prompt builder into immutable
  `GenerationRequest` rows. Arrival seconds use decimal round-half-up to
  integer picoseconds; ties retain caller order and the first Poisson arrival
  is not shifted.
- `HashedTokenPrompts(vocab_size, seed, first_token_id)`: deterministic private
  synthetic prompts for length/load studies. It deliberately adds no shared
  prefixes; WORK-1 remains the authority for controlled prefix structure.
- `reduce_transport_observations(requests, observations)`: validate one
  streamed-token observation per request and report client-observed TTFT,
  exact TPOT, and submission lateness. These `ObservedRequestTiming` rows are
  external evidence and never replace core `RequestMetric` values.

## Status

Arrival processes and length distributions are implemented and tested
(lengths landed with M1). WORK-3 is complete: the opt-in in-process gate now
consumes framework-request creation timestamps while leaving batching to the
framework. The live vLLM sweep passed all 8/8 genuine-risk arrival and offered
load instances, and the all-at-once mode retained every compared artifact
byte; see [the WORK-3 results](../../examples/arrival_admission_v1/RESULTS.md).
Shared-prefix prompt structure is not started.

The deterministic generation-request seam is also implemented and tested
without a framework dependency. The SGLang client maps it to native streaming
payloads and preserves the distinction between logical arrival, actual client
submission, and token visibility. The frozen study is void because its short
length-trace guard contradicted the established `TraceLengths` cycling
contract; its 6/6 matching workload and timing rows remain diagnostic only. See
[the SGLang MoE workload results](../../examples/sglang_moe_workload_v1/RESULTS.md).

## Open tasks

- WORK-1: shared-prefix prompt structure (system-prompt pools, multi-turn
  sessions) emitting token-ID sequences; the length-distribution part of
  this task landed with M1.
- WORK-2: bursty/MMPP arrival process for congestion-sensitive studies.
- WORK-4 (Completeness; P2; L): add a server-mode ingress coordinator that
  maps external request injection to simulated time without using wall-clock
  sleeps as model time. It must retain the in-process gate and ungated server
  path as explicit identity modes, preserve framework scheduler authority,
  and measure the coordinator's queue contribution separately from framework
  queueing.
