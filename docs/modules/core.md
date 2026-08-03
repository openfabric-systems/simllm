# simllm.core

Framework-agnostic heart of the simulator: the virtual clock and the
scheduler-step contract every frontend adapter translates into.

## Interface

- `StepRecord`: what a framework scheduler decided to run in one engine step
  (per-request phase, new tokens, cached tokens, preemptions, finishes).
- `StepResult`: the simulated outcome (step latency, completion time on the
  virtual clock).
- `RequestPhase`, `ScheduledRequest`: the per-request vocabulary.
- Closed-loop wire schemas (absorbed from the former bridge module): the
  step manifest `atlahs-closed-loop-step-v1` and the result manifest
  `atlahs-closed-loop-result-v1` (`STEP_SCHEMA` / `RESULT_SCHEMA`) are the
  JSON forms of `StepRecord` / `StepResult` exchanged with the simulator per
  scheduler step. Per-step subprocess invocation is the diagnostic mode.

Nothing in this package may import vLLM or SGLang.

## Status

Step records and the virtual clock (`VirtualClock`: heap-ordered events,
monotonic picosecond time, deterministic tie-breaking) are implemented and
tested; CORE-1 closed with M1. The closed-loop schemas are pinned names
only, first exercised in milestone M4.

## Open tasks

- BRIDGE-1 (inherited from the folded bridge module): persistent
  co-simulator process for closed loop, replacing per-step subprocess
  spawns; needs the incremental flow-injection mode on the htsim side
  (milestone M4).
