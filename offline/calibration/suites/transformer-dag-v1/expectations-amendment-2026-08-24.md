# Transformer DAG device calibration v1 freeze amendment

## Freeze scope and chronology

This is the expectations-only amendment to
`simllm-transformer-dag-expectations-v1`. The base freeze landed in commit
`45665fd`. Commit `60e3a83` registered its physical-sanity defects as COMP-53
on 2026-08-21, before any campaign cell governed by the suite was observed.
As of this amendment on 2026-08-24, no campaign-cell timing or result has been
observed or read. This amendment therefore precedes the first campaign cell
and contains no generated or measured value.

The original `expectations.json` and `expectations.md` remain unchanged. This
record supersedes only their incomplete finite-envelope field and evidence
clauses, EQ2 through EQ6, and G11. It clarifies the denominator inherited by a
scalar compute-memory envelope. Every other base-freeze clause remains in
force.

## Closed preflight states

The amendment retains the exact closed state set from the base freeze:

- `ready`: all required capabilities exist and the authored cell may run;
- `blocked`: a named site or tool capability is absent and no observation is
  produced;
- `not-applicable`: an optional mode is outside the declared target envelope;
- `rejected`: the backend must refuse an unsupported target or feature.

No other preflight state is valid.

## Campaign envelope and evidence independence

Every campaign envelope is content-addressed and frozen before its first
observation. The ceiling side retains minimum compute, HBM, peer-port and
transport rates plus maximum host-launch, device-fixed and per-transport-action
fixed times. The floor side declares the authored FLOP, compulsory-HBM-byte and
directional-peer-byte quantities, maximum compute, HBM and directional
peer-port rates, the per-implementation kernel-floor table, applicable stages
and their dependency edges.

Every field on both sides cites preexisting qualified evidence. A field is
rejected when its direct or transitive evidence lineage includes a timing or
result from the cell it bounds. This makes a floor derived from its bounded
measurement impossible by construction. Rates and per-kernel floors match the
exact target architecture and operating envelope. A missing, unqualified or
target-mismatched floor input leaves the affected cell `blocked` before any
timing is read.

Minimum and maximum rates are positive reduced rationals in their declared
base units per second. Fixed terms and kernel-floor values are nonnegative
integer picoseconds. All existing signed-128 arithmetic and one-final-ceiling
rules remain in force.

## Terms and operators

EQ2 through EQ5 evaluate independently for each applicable stage.

- `flops` is the stage-authored nonnegative floating-point operation count.
- `hbm_bytes` is the stage-authored nonnegative compulsory HBM byte count.
- `peer_bytes` is the stage-authored nonnegative directional peer-port byte
  count charged once.
- `picoseconds_per_second` is exactly 1,000,000,000,000.
- Each `maximum_*_rate_num` and `maximum_*_rate_den` is the positive numerator
  or denominator of the named preexisting qualified maximum rate.
- `compute_floor_ps`, `memory_floor_ps` and `peer_floor_ps` are the outputs of
  EQ2, EQ3 and EQ4 for one applicable stage.
- `kernel_floor_ps` is zero for a nonkernel stage. For a kernel stage it is the
  exact target- and implementation-matched nonnegative fixed per-kernel floor
  owned by COMP-43, obtained from preexisting nonvoid qualified evidence and
  resolved from `kernel_floor_ps_by_implementation` before timing is read.
- `isolated_floor_ps` is the EQ5 maximum for one applicable stage.
- `applicable_stages` is the pre-observation authored stage set for the cell,
  with each stage resolved to one exact implementation.
- `dependency_edges` is the pre-observation authored acyclic dependency graph
  over those stages.
- `applicable_stage_floors_ps` is the EQ5A map containing one
  `isolated_floor_ps` for every and only applicable stage.
- `graph_floor_ps` is the EQ6 longest dependency-path sum through that map.

`ceil` returns the least integer no smaller than its exact rational argument.
`max` returns the greatest exact integer argument. `map_applicable_stages`
applies EQ2 through EQ5 once per applicable stage and preserves stage identity.
`longest_dependency_path_sum` takes the maximum, over all directed dependency
paths, of the sum of each visited stage floor exactly once.

## Replacement equations

- **EQ2** `compute_floor_ps = ceil(flops * maximum_compute_rate_den * picoseconds_per_second / maximum_compute_rate_num)`.
- **EQ3** `memory_floor_ps = ceil(hbm_bytes * maximum_hbm_rate_den * picoseconds_per_second / maximum_hbm_rate_num)`.
- **EQ4** `peer_floor_ps = ceil(peer_bytes * maximum_peer_port_rate_den * picoseconds_per_second / maximum_peer_port_rate_num)`.
- **EQ5** `isolated_floor_ps = max(compute_floor_ps, memory_floor_ps, peer_floor_ps, kernel_floor_ps)`.
- **EQ5A** `applicable_stage_floors_ps = map_applicable_stages(applicable_stages, isolated_floor_ps)`.
- **EQ6** `graph_floor_ps = longest_dependency_path_sum(dependency_edges, applicable_stage_floors_ps)`.

## Replacement fatal guard

- **G11 Pass separation.** Timeline, counter, dynamic-instruction and mixed
  passes remain separate. Counter replay is never used as the concurrency
  timeline, and no pass output substitutes for another pass's evidence.

All four declared measurement passes are covered. G11 remains fatal and
unscored.

## Scalar compute-memory mixed denominator

The full authored mixed topology remains 28 cells. A scalar compute-memory
envelope has compute and memory ready and declares communication unsupported.
The suite's `mixed_rule` therefore includes all four widths for the
`mix-compute`, `mix-memory` and `mix-compute-memory` arms. Its reduced mixed
denominator is exactly 3 arms times 4 widths, or 12 cells. The 16 cells with a
communication member are excluded because that member capability is not
ready. The 20-cell communication denominator also remains unsupported.

## What this amendment does not change

This record changes no authored suite topology, scored acceptance bar,
expected behavioral relation, full communication-envelope denominator,
implementation or campaign result. It closes no campaign task and authorizes
no observation before all remaining preflight requirements are `ready`.
