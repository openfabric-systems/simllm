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

Conversion: `to_binary(goal_path)` runs `txt2bin`, discovered via
`SIMLLM_TXT2BIN`, the checked-in binary in the htsim submodule, then `PATH`.

## Status

Implemented and tested; GOAL-1 closed with M1. The `txt2bin` helper and an
end-to-end round-trip test landed (`tests/test_htsim_rnic.py`); the test
self-skips where the backend toolchain is absent (e.g. CI without
submodules) and runs for real otherwise. Validated end to end by the M1
sanity studies across all three wired `htsim_rnic` profiles.

## Open tasks

None currently.
