# Compute and communication overlap v1 results

TRAF-7 is complete for the generic observation-driven step path. All 16
scored behavioral relations, 22 exact-oracle rows and 12 fatal unscored guards
passed. Independent work realized `max(C, D)`, serial work realized `C + D`,
and the registered two-stage pipeline landed strictly between them at its exact
closed form. The same movement reached live TTFT and TPOT. This is evidence for
dependency-driven overlap in the accepted coarse runtime, not a calibrated
claim about a current framework or GPU-resident NCCL execution.

## Chronology and reproduction

The expectations were frozen in commit `cf3ae9c` before implementation or a
result-producing run. The pre-freeze check-only invocation parsed the complete
CLI, validated only frozen literals and created no directory or artifact.
Implementation landed in `255c5a0`. The first run passed every evidence row,
but its JSON rendered exact integral `Fraction` values as strings ending in
`/1`. Commit `5c9ea1f` normalized that report representation without changing
any modeled value, evidence comparison or decision, then the study was rerun.

The final result record observed repository commit
`5c9ea1fea4b5402d1ac91d10236b38b723c8e6d5`. This is run provenance, not an
assertion that a live repository or submodule pin must continue to equal that
commit. The record is external to Git at
`$SIMLLM_OVERLAP_RUN_ROOT/run-5c9ea1f/results.json` with SHA-256
`c6647cd5a4b0da6ff7b820de4ec92a3315ef5211a5fabfa69d7f5c4f7fde1c6a`.
The earlier representation-only record has SHA-256
`c7ed6d5cc0e360dbabe872792194ac26fd8529b9e720ed011109b1ae8df397b8`.

Reproduce the final run from the repository root with:

```bash
.venv/bin/python examples/compute_comm_overlap_v1/run_study.py \
  --output-dir "${SIMLLM_OVERLAP_RUN_ROOT:?configure SIMLLM_OVERLAP_RUN_ROOT}/run-5c9ea1f"
```

The final runner printed:

```text
study passed: exact_oracle=22/22, behavioral_relation=16/16, fatal_guard=12/12
```

## Evidence classes

The classes stay separate because they answer different questions.

| Evidence class | Count | Outcome |
|---|---:|---|
| Run configurations | 8 | Six dependency-shape cells plus shared and split NCCL-channel cells |
| Exact-oracle rows | 22 | 22 passed |
| Scored behavioral instances | 16 | 16 passed |
| Fatal structural and identity guards | 12 | 12 passed, unscored |
| Focused Python regressions before the run | 116 | 116 passed |
| Full repository Python regression | 853 collected | 846 passed, 7 skipped |
| Repository lint | 1 invocation | Passed |

No count in this table is added to another class. No native executable was
changed or used as evidence in this Python-only landing.

## A. Dependency shape and ratio crossover

The fixed communication term was `D = 41,943,040 ps`. Compute crossed it at
`C/D = 1/2` and `C/D = 2`. Every row below is the raw runtime observation and
equals its pre-registered closed form exactly.

| C (ps) | C/D | Independent (ps) | Pipeline (ps) | Serial (ps) |
|---:|---:|---:|---:|---:|
| 20,971,520 | 1/2 | 41,943,040 | 52,428,800 | 62,914,560 |
| 83,886,080 | 2 | 83,886,080 | 104,857,600 | 125,829,120 |

For `c = C/2` and `d = D/2`, the registered forms were:

```text
independent(C, D) = max(C, D)
pipeline(C, D) = c + max(c, d) + d
serial(C, D) = C + D
```

At the low ratio, both gaps around the pipeline result were 10,485,760 ps. At
the high ratio, both were 20,971,520 ps. Thus the partial-dependency result was
strictly between the independent and serial results on both sides of the
crossover. A global serializer, a missing completion frontier or a weakened
participant-local edge could have failed these relations.

Each of the six configurations ran three consecutive steps through one
stateful runtime. Every step emitted 200 completion events, and each request's
seven-component critical-path attribution summed exactly to its step latency.
Those event and conservation checks are fatal guards, not scored behavior.

## B. Live TTFT and TPOT

Step 0 was prefill and steps 1 and 2 were decode for the same request. With no
scheduler gap, the first-step TTFT and each decode TPOT equaled the raw step
latency. Changing only the schedule from serial to pipeline produced:

| C/D | Serial metric (ps) | Pipeline metric (ps) | Signed reduction (ps) | Pipeline/serial |
|---:|---:|---:|---:|---:|
| 1/2 | 62,914,560 | 52,428,800 | 10,485,760 | 5/6 |
| 2 | 125,829,120 | 104,857,600 | 20,971,520 | 5/6 |

The table applies independently to TTFT and TPOT, giving four scored metric
instances. The graph movement therefore remains live-reachable through
`ExecutionResult`, `CompletionEvent`, `CompletionReducer`, `StepResult`, TTFT
and exact rational TPOT.

## C. First added contention mechanism

The contention fixture submitted two dependency-independent two-rank rings on
distinct logical queues. Changing only the physical NCCL channel identity gave:

| Channel shape | Second operation first grant (ps) | JCT (ps) |
|---|---:|---:|
| Shared `shared, shared` | 2,001 | 4,003 |
| Split `a, b` | 0 | 3,004 |

On the shared channel, the second operation's first grant equaled the first
operation's last channel release at 2,001 ps. Splitting the channels exposed
legal concurrency, while RNIC serialization still remained, and reduced JCT by
exactly 999 ps. The two raw shared-versus-split relations were scored before
the four fixed channel literals were checked.

## D. Serial identity off path

With observations omitted, `ObservedStepLowerer` delegates directly to
`SerialStepLowerer`. It does not reconstruct the serial schedule. The accepted
compatibility fixture remained exact:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Canonical execution graph JSON | 4,127 | `aa3c836fe559973a7bf0940384c2e8a84e6af84e0fbd2c02d3b89774ee0c8e2d` |
| Serial graph-only GOAL | 1,880 | `7087db6780f7e34f5a559a6505eeccc15d984c7b478cd8f0bc5838053825d4b6` |

These hashes, the absent observation selection, configuration echoes, event
cardinality and attribution conservation are fatal but unscored. They do not
inflate the behavioral denominator.

## Entailment and genuine-risk accounting

The runner called `_family_a_relations`, `_family_b_relations` and
`_family_c_relations` on raw runtime observations before `_exact_rows` applied
any fixed literal. No exact-oracle or fatal guard gated those raw comparisons.
The serialized record states `raw relations before entailing exact oracles` as
the evaluation order.

The exact rows later pin the same individual values for regression use, so they
are not counted again as genuine-risk evidence. Likewise, the serial hashes,
field-preservation checks, by-construction configuration values, event
existence and conservation identities are fatal-unscored.

| Scored family | Genuine-risk instances | Passed | Fraction |
|---|---:|---:|---:|
| A, dependency shapes and crossover | 10 | 10 | 100% |
| B, live TTFT and TPOT movement | 4 | 4 | 100% |
| C, NCCL-channel contention | 2 | 2 | 100% |

The aggregate behavioral fraction is 16/16, but the three families share the
same implementation and are not statistically independent experiments.

## TRAF-7 closure map

The original acceptance clauses map as follows.

> "lowering compute and collective work onto the framework-observed logical streams with explicit event/dependency edges in `ExecutionGraph`"

`lower_step_observations` preserves the observation tuple order, queue, both
edge kinds, gates, priorities, correlation and completion frontier. Compute
work passes through unchanged. Traffic validates each semantic collective and
binds its algorithm, payload, routed pair table and placement epoch. Focused
tests and eight fatal runner guards cover this seam. A current adapter does not
yet produce such a schedule, so TRAF-13 owns that live integration.

> "The adapter owns observed program order and legal concurrency; the traffic planner owns collective algorithm/chunk expansion; `DeviceRuntime` owns realized overlap after CUDA-stream, GPU, HBM, copy-engine, NCCL-channel, WQE and NIC contention."

The demonstrated portion preserves adapter-authored order and concurrency,
selects traffic-owned collective semantics, and lets `CoarseDeviceRuntime`
realize the schedule through its compute, logical launch, NCCL-channel, WQE and
RNIC resources. Family C isolates NCCL-channel contention. The coarse runtime
still reconstructs physical ring rounds and pairwise extents from semantic
work, so TRAF-14 now owns the explicit traffic-plan projection. GPU-resident
collective work and any captured copy/GPUDirect path remain CORE-26, CORE-27
and COMP-22. They are not approximated here.

> "No layer stores or learns an overlap percentage."

The only enabled input is an observation graph. The public lowering signature
has no overlap, percentage or duration-discount parameter, and its fatal guard
passed. All realized overlap emerges from missing dependency paths and the
runtime's actual resource cursors.

> "First validate an ideal independent-resource graph at exact `max(compute, communication)` versus the serial graph at exact `compute + communication`, then add one resource contention mechanism at a time."

CORE-4 supplied the accepted component baseline. Family A reproduced both
forms with step-derived ring traffic, added the exact registered pipeline
shape, and crossed the resource-dominance point. Family C then changed one
resource identity at a time to isolate NCCL-channel contention.

> "The strictly serial graph is your identity off path: every accepted artifact that assumed serial chaining stays byte-identical when overlap is not enabled."

Family D pins the canonical graph and GOAL byte counts and hashes. Existing
dense, TP=1, MoE, routed-supply and legacy serial regressions passed in the
focused 116-test suite.

## Physical scope and residuals

The physical result is limited to the accepted coarse model: compute work has
zero HBM demand in the ideal family; communication uses semantic ring service,
the coarse per-rank NCCL-channel cursor, WQEs and 400 Gbit/s RNIC serialization.
The study does not establish a schedule captured from vLLM or SGLang. It does
not model or calibrate NCCL kernel residency, shared SM or HBM pressure,
ingress, reduction lanes, proxy progress, copy-engine occupancy, GPUDirect DMA
or online packet-level overlap. TRAF-13, TRAF-14, CORE-26, CORE-27 and COMP-22
state the surrogate, evidence source and acceptance bar for these residuals.

No code or expectation from TRAF-10 was used, and no traffic-locality logic was
changed.

## Contradiction sweep

The required post-closure sweep found no matching overlap or TRAF-7 statement
in `README.md` or `docs/architecture.md`. It found one integrator-owned roadmap
statement at `docs/README_PRO.md:219-222` that still links TRAF-7 as an open
dependency-driven-overlap task. That prose was not edited here. Only the
generated progress block and mechanically checked module open counts changed
in that file for ledger reconciliation.

The broader sweep also found `docs/modules/adapters-sglang.md:192`, which still
says device overlap remains CORE-4/TRAF-7. That adapter-owned wording should be
reconciled with TRAF-13 during integration. Historical frozen expectations and
results under `examples/m4`, `examples/m5`, `examples/breakdown` and
`examples/routed_supply_v1` describe the state at their run chronology and were
left unchanged.
