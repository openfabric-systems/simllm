# simllm.bridge

Closed-loop bridge schemas: the versioned JSON manifests the frontend
adapter and the simulator exchange per scheduler step (or window of steps).

## Interface

- Step manifest, schema `atlahs-closed-loop-step-v1`: what the scheduler ran
  (request ids, phases, token counts, cache hits) and the virtual time.
- Result manifest, schema `atlahs-closed-loop-result-v1`: the simulated time
  (`simulated_time_us`) and per-flow completions.

Per-step subprocess invocation is the diagnostic mode; a persistent
co-simulator process is planned for scale.

## Status

Schema names pinned (adopted from the pre-existing capture prototype's
blocking bridge contract). No implementation yet; first exercised in
milestone M4.

Flagged for consolidation: this module is two constants and may fold into
`simllm.core` next to `StepRecord`/`StepResult` (the manifests are their
wire form). Decision pending with the maintainer.

## Open tasks

- BRIDGE-1: persistent co-simulator process for closed loop (replacing
  per-step subprocess spawns; needs the incremental flow-injection mode on
  the htsim side), milestone M4.
- BRIDGE-2: decide the consolidation flag above (fold into core, or keep as
  its own module when BRIDGE-1 grows real code).
