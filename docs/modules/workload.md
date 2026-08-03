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
