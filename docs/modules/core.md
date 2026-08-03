# simllm.core

Framework-agnostic heart of the simulator: the virtual clock and the
scheduler-step contract every frontend adapter translates into.

## Interface

- `StepRecord`: what a framework scheduler decided to run in one engine step
  (per-request phase, new tokens, cached tokens, preemptions, finishes).
- `StepResult`: the simulated outcome (step latency, completion time on the
  virtual clock).
- `RequestPhase`, `ScheduledRequest`: the per-request vocabulary.

Nothing in this package may import vLLM or SGLang.

## Status

Step records are implemented and tested. The virtual clock is not yet
implemented; it lands with milestone M1.

## Open tasks

- CORE-1: implement the virtual clock (event ordering for arrivals, step
  releases and completions) as the M1 backbone.
