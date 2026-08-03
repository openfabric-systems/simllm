# simllm.workload

Request generation as a queueing model: when requests arrive and what they
look like. Prefix-hit probability, cache-miss re-prefill traffic and vRAM
pressure must be emergent from the workload, never assumed, so prompts are
synthetic token-ID sequences with controllable shared-prefix structure.

## Interface

- `PoissonArrivals(rate_rps, seed)`: open-loop Poisson arrivals, reproducible
  by seed; `times(n)` or infinite iteration.
- `TraceArrivals(path)`: replay absolute arrival times from a file.

## Status

Arrival processes are implemented and tested. Length distributions and
shared-prefix prompt structure are not started.

## Open tasks

- WORK-1: prompt/output length distributions (fixed, lognormal, trace) and
  shared-prefix prompt structure (system-prompt pools, multi-turn sessions)
  emitting token-ID sequences.
- WORK-2: bursty/MMPP arrival process for congestion-sensitive studies.
