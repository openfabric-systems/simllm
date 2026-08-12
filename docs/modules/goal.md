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

The same grammar also carries the independently constructed ATLAHS schedule
from `render_step_goal`. In a standalone direct-GOAL run, those emitted
relations are the selected ordering authority. In the serial sink's explicit
`dependency_cross_check="atlahs-goal"` mode, that schedule remains independent
but observational: the graph-projected GOAL determines the sink result, and
the direct execution supplies ordering-scope, raw phase-frontier and
completion-time differences without overriding or averaging the result.

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
execution. The byte-locked direct trace remains independently executable as an
explicit ATLAHS cross-check rather than a second authority inside that run.
With the cross-check disabled, accepted GOAL artifacts and sink results remain
unchanged. The all-remote comparator audits all 423 canonical effective edges:
the frozen set has 47/47 whole-operation FIFO differences, and its expanded
post-specified, unscored diagnostic finds another 188 participant-local
syntactic-frontier mismatches. The separate raw timing subset retains the 47
frozen boundaries and finds 46/47 unequal, early gaps. Completion-time
differences remain separate findings; see
[the dependency authority results](../../examples/dependency_authority_v1/RESULTS.md).

## Open tasks

None currently.
