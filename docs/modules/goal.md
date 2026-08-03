# simllm.goal

GOAL (Group Operation Assembly Language) trace emission: the dependency-graph
schedule format consumed by LogGOPSim and htsim.

## Interface

`GoalTrace(num_ranks)` builds per-rank programs of `calc` / `send` / `recv`
operations with `requires` (start after finish) and `irequires` (start after
start) edges, and renders the text grammar accepted by `txt2bin`:

```
num_ranks 2
rank 0 {
  r0op0: calc 5000
  r0op1: send 8192b to 1 tag 0
  r0op1 requires r0op0
}
rank 1 {
  r1op0: recv 8192b from 0 tag 0
}
```

Optional `cpu` / `nic` clauses pin operations to resources.

## Status

Implemented and tested. Validated end to end against the real toolchain: an
emitter-generated 8-rank chain converted with `txt2bin` and executed by both
`htsim_uec` and all three wired `htsim_rnic` profiles (2026-08-03 smoke).

## Open tasks

- GOAL-1: `txt2bin` invocation helper and a binary round-trip test wired
  into CI once the toolchain location is standardized (milestone M1).
