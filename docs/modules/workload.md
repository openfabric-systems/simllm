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

## Status

Arrival processes and length distributions are implemented and tested
(lengths landed with M1). Shared-prefix prompt structure is not started.

## Open tasks

- WORK-1: shared-prefix prompt structure (system-prompt pools, multi-turn
  sessions) emitting token-ID sequences; the length-distribution part of
  this task landed with M1.
- WORK-2: bursty/MMPP arrival process for congestion-sensitive studies.
- WORK-3 (Completeness; P1; M): consume framework-request creation timestamps
  through an opt-in in-process admission gate. Hold each request outside
  `add_request` until the shared virtual clock reaches its arrival, retain
  stable bookkeeping order for ties, and leave batching entirely to the
  framework scheduler. Acceptance must sweep arrival offset and a burst that
  exceeds one available scheduler slot, reproduce exact per-request queue and
  TTFT movement, and keep the all-at-once path byte-identical.
- WORK-4 (Completeness; P2; L): add a server-mode ingress coordinator that
  maps external request injection to simulated time without using wall-clock
  sleeps as model time. It must retain the in-process gate and ungated server
  path as explicit identity modes, preserve framework scheduler authority,
  and measure the coordinator's queue contribution separately from framework
  queueing.
