# Dependency authority cross-check amendment expectations

## Chronology and architecture decision

The original TRAF-12 expectations were frozen in commit `d39dfdc` and the
registered run was closed in commit `a358f57`. Maintainer direction arrived
after that closure, following the fidelity-level contract merged in PR 40 at
commit `d41ff1a`. This file is a new expectations-only supplement. It does not
rewrite the original freeze or claim that the cross-check mode was registered
before the first TRAF-12 implementation or run.

The semantic decision is narrower than one permanent implementation:

- exactly one dependency mechanism owns ordering in any enabled run;
- the active serial step-sink configuration selects `ExecutionGraph` as that
  authority and executes its checked GOAL projection;
- the independent ATLAHS direct-GOAL mechanism remains selectable as
  `atlahs-goal` cross-check evidence;
- selecting the cross-check may add diagnostics, but it must not change the
  authoritative `StepResult`, graph projection, active artifact bytes or
  scheduler-visible completion;
- a disagreement is reported as a finding. It is never averaged with the
  authoritative result, used as a silent override or treated as a reason for
  the cross-check API itself to fail.

CORE-36 owns the eventual unified fidelity selection and provenance surface.
This amendment may expose only the current dependency-seam switch and must
not introduce a competing global configuration scheme.

## Frozen cross-check cells

The amendment reruns the captured Granite step for vector bytes 1,024 and
2,048 under the all-remote `ABCD` placement. Both executions use the same
`StepRecord`, model dimensions, ranks, captured routed supply, per-layer
compute durations, GOAL rank count, 400 Gbit/s link rate and deterministic
`rnic-nn-fluid` profile.

The graph projection is authoritative. The cross-check independently renders
one direct GOAL from the same semantic inputs and runs it through ATLAHS and
htsim. Completion tolerance is exactly 0 ps. The supported amendment scope is
the all-remote compatibility level, where both mechanisms represent the same
physical messages. A selected cross-check with local NVLink segments must
reject before writing cross-check artifacts; TRAF-16 owns exact per-rank
frontiers across placement and process boundaries.

The default is cross-check disabled. The disabled path must preserve every
accepted graph artifact and result byte for byte. The explicit cross-check
must use separately named GOAL, binary and completion CSV artifacts.

## Disagreement definitions

Every finding carries both mechanism names, the graph execution identity and
the step identity.

### Ordering scope

For every distributed whole-operation boundary in the authoritative graph,
inspect the independently rendered direct GOAL dependency reachability. An
ordering disagreement exists when at least one target-rank entry lacks a
causal path from a predecessor-rank terminal required by the graph scope.
The report identifies the predecessor and target operation IDs, edge origin,
target rank and missing predecessor ranks.

The frozen graph contains 47 distributed logical-queue FIFO boundaries. The
direct GOAL is expected to realize all 47 only as rank-local chains, so the
expected structural ordering disagreement is 47 of 47 at both payloads.

### Phase frontier

For each adjacent predecessor and target collective, calculate directly from
the cross-check completion CSV:

```text
gap_ps = minimum target-tag start - maximum predecessor-tag completion
```

A negative gap is a phase-frontier disagreement with the authoritative
whole-operation boundary. Record all 47 signed gaps with the two operation
IDs, tags and raw timestamps. The registered historical findings are:

| Vector bytes | Adjacent frontiers | Negative gaps | First gap ps | Minimum gap ps |
|---:|---:|---:|---:|---:|
| 1,024 | 47 | 46 | -368,640 | -1,413,120 |
| 2,048 | 47 | 46 | -737,280 | -3,675,091 |

The graph artifact path has zero early entries across these boundaries by
construction. That guard remains fatal-unscored and cannot make the raw
cross-check finding pass.

### Completion time

The signed comparison is:

```text
signed_completion_difference_ps = cross_check_completion_ps
                                  - authoritative_completion_ps
```

A completion disagreement exists when the absolute difference exceeds the
registered 0 ps tolerance. The direct-GOAL completion stays at the historical
all-remote value and the signed bands are the exact negatives of the original
TRAF-12 reconciliation bands:

| Vector bytes | Direct completion ps | Signed cross-check minus authority ps |
|---:|---:|---:|
| 1,024 | 156,569,755 | [-4,212,053, -4,212,005] |
| 2,048 | 217,222,486 | [-8,317,082, -8,317,034] |

A null completion result contradicts the already observed ordering and
frontier differences and must be explained rather than accepted.

## Evidence classification and acceptance

The existing genuine-risk denominator remains exactly two families and three
instances: two signed-JCT instances plus the one missing-edge mutation. The
new cross-check findings are not added to that score because the prior
TRAF-10 and TRAF-12 results already pin their direction and magnitude.

The amendment evidence classes remain separate:

- the two original scored behavioral families;
- two cross-check finding rows with three typed disagreement classes each;
- exact active and direct artifact inventories;
- raw per-boundary cross-check timestamps;
- fatal-unscored selection neutrality, report completeness, quiescence and
  supported-domain guards;
- unit, full-suite and native execution evidence.

The cross-check API succeeds when it produces a complete report, including
when that report says the mechanisms disagree. Missing artifacts, incomplete
edge or timestamp inventories, a cross-check that changes the authoritative
result, or an unknown selection value are fatal failures. The registered
study may still reject an unexpected numerical drift after recording it; the
runtime comparison API itself must not assert equality.

## Registered command

The production supplement uses a new bulk-output directory and never
overwrites the original run:

```bash
.venv/bin/python examples/dependency_authority_v1/run_study.py \
  --out "$SIMLLM_DEPENDENCY_AUTHORITY_AMENDMENT_RUN_ROOT"
```

Before this expectations commit, the same command is exercised with
`--check-only`. That path validates the amendment literals and arithmetic,
does not inspect native tools or tracked traces, and creates no output.
