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
- Record serialization both ways: `step_record_to_json` /
  `step_records_to_json` / `write_step_records` out,
  `step_record_from_json` / `step_records_from_jsonl` back in (schema tag
  validated, unknown schemas rejected loudly, bad JSONL lines named by
  line number). The adapters' streamed JSONL dumps are therefore replayable
  through any step sink.

Nothing in this package may import vLLM or SGLang.

## Status

Step records and the virtual clock (`VirtualClock`: heap-ordered events,
monotonic picosecond time, deterministic tie-breaking) are implemented and
tested; CORE-1 closed with M1. The step-record JSON readers landed with the
M4 first slice, which also exercised the step schema for real: the recorded
M2/M3 smoke JSONLs load, round-trip and replay through the closed-loop
sink (`simllm.backends.HtsimStepSink`, examples/m4). The result schema
remains exercised in-process only (the `StepResult` side crosses no process
boundary until BRIDGE-1).

## Open tasks

- BRIDGE-1 (inherited from the folded bridge module): persistent
  co-simulator process for closed loop, replacing per-step subprocess
  spawns; needs the incremental flow-injection mode on the htsim side.
  The M4 first slice implemented the per-step-subprocess diagnostic mode
  (`simllm.backends.HtsimStepSink`, ~8 s wall per step in the live tp=8
  run), which is exactly the overhead this task removes.
