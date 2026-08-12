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

`GoalOperation`, `GoalMessage` and `GoalDependency` retain read-only semantic
metadata beside those text rows. Operations name their graph owner; messages
record source, destination, payload, tag and optional request partition; and
dependencies distinguish execution-graph provenance from deterministic
collective-internal ordering. The metadata is not emitted into GOAL text.
Checked graph projections use `requires` for representable graph completion
edges, while collective expansion may use both grammar relations for its own
operation-internal schedule.

Optional `cpu` / `nic` clauses pin operations to resources.

Conversion: `to_binary(goal_path)` runs `txt2bin`, discovered via
`SIMLLM_TXT2BIN`, the CMake build tree (including MSVC configuration
directories), the legacy checked-in source-tree executable on Unix, then
`PATH`.

## Status

Implemented and tested; GOAL-1 closed with M1. The `txt2bin` helper and an
end-to-end round-trip test landed (`tests/test_htsim_rnic.py`); the test
self-skips where the backend toolchain is absent (e.g. CI without
submodules) and runs for real otherwise. Validated end to end by the M1
sanity studies across all three wired `htsim_rnic` profiles.
For the supported serial sink, the execution graph is the ordering authority.
The checked projector either renders a graph edge at its exact rank-local scope
or records it as an ordered artifact boundary or serialized edge; the verifier
rejects loss, duplication and unsupported completion boundaries before backend
execution. The legacy direct trace remains a byte-locked diagnostic.

## Open tasks

None currently.
