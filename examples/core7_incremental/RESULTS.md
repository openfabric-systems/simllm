# CORE-7 incremental bookkeeping validation results

Date: 2026-08-10

The study passed every expectation frozen before implementation and execution
in [expectations.md](expectations.md) at commit `d487a69`. The implementation
first landed at `e18200d`; integration review subsequently consolidated its
rule engine before the rerun reported here.

## Reproduction

The behavioral equivalence cases are part of the normal test suite:

```bash
.venv/bin/pytest -q tests/test_bookkeeping.py \
  tests/test_bookkeeping_incremental.py
```

The scaling study is reproduced with:

```bash
.venv/bin/python -m examples.core7_incremental.run_study \
  --out /data3/yifeng/simllm-dev/wave1-runs/core7_incremental_ledger
```

The raw file is
`/data3/yifeng/simllm-dev/wave1-runs/core7_incremental_ledger/measurements.json`.
Its SHA-256 digest for this run is
`5a7035c60ece0dfe54eb22fc30550df45cdcd2f27cced0071b29280e8e2a6d0b`.
It contains 48 measured rows plus the configuration, host disclosure,
medians, relation checks and structural guards. Raw timing data is not Git
content.

## E1: seeded behavioral equivalence

All six registered seeds passed at valid stream lengths 8, 32 and 128. The 18
valid streams were each exercised through individual `append` calls and
seeded `extend` partitions, giving 36 valid stream-mode instances. Every
accept decision, returned entry or tuple, and final snapshot matched a fresh
full-candidate `validate_bookkeeping_ledger` call exactly.

Protocol deviation: the frozen plan says batch widths are drawn from
`{0, 1, 2, 7, 32}`. The harness draws each nonzero partition width from
`(1, 2, 7, 32)` and exercises width 0 once deterministically before the
partition loop. Every registered width is exercised, and this scheduling
difference does not change any candidate, decision or acceptance relation.

Each seed also exercised all 42 registered invalid mutation kinds through
both modes, giving 504 invalid stream-mode instances. Invalid facts were
injected at seed-selected positions. For every candidate prefix or atomic
batch, the incremental path matched the reference accept or reject decision
and exception class. Rejections retained the last accepted ledger exactly.

The mutation catalog covered fact and timestamp types, enum and scalar types,
scope and correlation rules, metadata, duplicate and malformed references,
causal lineage and request narrowing, WQE queue and transport shape, stage
references, completion subject scope and timestamp order, completion-queue
identity, duplicate WQE completion and strict post-completion terminality.
The integration audit added 20 direct parameterized cases for parent-field
inheritance, duplicate parent and object refs, excessive WQE queue cardinality,
blank identities and non-integer correlation fields. The focused command
completed with `77 passed in 12.18s`; 45 of those tests are the new seeded,
initialization and direct audit cases, while the existing 32 tests keep the
deterministic and wire sentinels.

This is the E1 behavioral relation family. Its instances are not added to the
timing relations below.

## E2: append scaling

The run used Python 3.12.12 on Linux 5.14 x86-64. Garbage collection was
disabled only within each timed call. One complete warmup sweep preceded five
recorded incremental repetitions and three recorded reference repetitions.
The tables report medians, not exact cross-machine expectations.

### Incremental path

| fact mix | 1,000 facts ms | 4,000 facts ms | 16,000 facts ms | 4,000 / 1,000 | 16,000 / 4,000 |
|---|---:|---:|---:|---:|---:|
| stage | 3.861 | 16.173 | 69.056 | 4.188 | 4.270 |
| WQE | 7.504 | 30.480 | 122.503 | 4.062 | 4.019 |

All four quadrupling relations passed the registered upper bound of 6. The
observed ratios stay near the ideal linear value of 4 for both the stateless
stage mix and the cross-entry WQE mix.

### Reproduced former full-candidate path

| fact mix | 1,000 facts s | 2,000 facts s | 4,000 facts s | 4,000 / 1,000 |
|---|---:|---:|---:|---:|
| stage | 1.047 | 4.203 | 17.301 | 16.523 |
| WQE | 2.509 | 10.593 | 40.379 | 16.092 |

Both endpoint growth relations passed the registered lower bound of 8. The
ratios expose the former quadratic trend and are not treated as stable speedup
claims.

## Structural guards and interpretation

The by-construction guards all passed: both timing modes consumed the same
immutable fact tuples, both complete generated streams passed the reference
validator, generated stream lengths matched the requested maximum, every
measured ledger reached the requested length, rejected equivalence cases
retained their prior snapshots, and API-assigned sequences remained contiguous
plain integers. The explicit malformed-initial-ledger test retained
Boolean-sequence rejection through the full validator. These guards are fatal
but unscored and do not increase either behavioral relation count.

`RequestBookkeeper` now retains private indexes for object records, latest
subject timestamps and terminal WQEs. Each `extend` call validates against a
copy-on-write overlay and commits it only after the complete batch succeeds.
The full scan and incremental transaction both call one shared per-entry rule
engine, so future invariant edits have one implementation. The public ledger
and v1 wire schema are unchanged. The full-ledger validator remains the
complete-scan reference used for constructor inputs, explicit immutable-ledger
validation and both wire directions.
