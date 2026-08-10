# CORE-4 coarse device runtime results

The first coarse `DeviceRuntime` passes all four experiment families frozen
before implementation. Across 16 run configurations, 20 of 20 exact-oracle
rows, 17 of 17 scored behavioral relations, and 16 of 16 fatal structural
guards passed. The runtime changed `ExecutionGraph` JCT in the signed direction
and exact amount predicted by dependency, RNIC rate, and synchronous control
delivery. Omitted and explicit identity arbitration produced the same canonical
bytes under class-label permutation.

## Expectations and chronology

The older experiment families were first recorded in `docs/modules/core.md` at
commits `ea3961b` and `37357cc`. The final task-specific expectations-only
ancestor is commit `d43cddb` (`Freeze the coarse device runtime expectations`).
That commit contains only
[expectations.md](expectations.md), follows its registered parse-only dry run,
and precedes the runtime, tests, study runner, and first measured result. These
are therefore pre-registered relations rather than post-specified regression
checks.

The freeze cited repository-native ring and pairwise source referents before
implementation. No scored relation imports a vLLM, NCCL, or hardware-spec
expectation. Rates and durations are ideal fixture inputs with no silicon
claim.

Reproduce from the repository root:

```bash
.venv/bin/python examples/core4_runtime/run_study.py --check-only
.venv/bin/python examples/core4_runtime/run_study.py \
  --output-dir /data3/yifeng/simllm-dev/wave2-runs/codex/core4_device_runtime
```

The parse-only command produces no result. The measured summary remains on the
data volume at the path above. Its SHA-256 is
`c89da4abfaa3cc8e6f00037e950887758c89c1af5a6bc9600ed353b7962ea53d`.
No bulk output is tracked in the repository.

## Evidence accounting

Evidence classes remain separate. Exact-oracle rows and behavioral relation
instances are both scored but are not added into one synthetic headline.
Conservation, authority exclusivity, frozen event/tag structure, and
configuration-forced zeros are fatal and unscored. Unit tests and the full
repository regression are separate executables.

| Evidence class | Result | Meaning |
|---|---:|---|
| Run configurations | 16 | Unscored parameter records |
| Exact-oracle rows | 20/20 pass | Graph JCT, quiescence, and useful throughput |
| Behavioral relation instances | 17/17 pass | Signed dependency, rail, rate, control, wait, and identity relations |
| Fatal structural guards | 16/16 pass | Authority, FIFO, conservation, origin, HBM, and GOAL-tag invariants |
| Focused runtime tests | 20/20 pass | Separate Python unit-test executable |

The structural GOAL comparison used the frozen renderer as its independent
referent. For a two-rank ring, both rendered sends in each round and all four
runtime WQE projections carried tags `[1000, 1000, 1001, 1001]`. Exact internal
event or method sequences without such a frozen referent were not scored.

## A. Dependency versus legal overlap

The two resource-independent no-edge graphs matched `max(C, D)`. Adding the
compute-to-DMA dependency changed JCT to `C + D` and added exactly `min(C, D)`.

| C (ps) | D (ps) | No-edge JCT (ps) | Edge JCT (ps) | Edge penalty (ps) |
|---:|---:|---:|---:|---:|
| 10,000,000 | 40,000,000 | 40,000,000 | 50,000,000 | 10,000,000 |
| 80,000,000 | 40,000,000 | 80,000,000 | 120,000,000 | 40,000,000 |

Every `ExecutionResult.completed_at_ps` also retained the nonzero
`T0 = 5,000 ps` origin. Changing only the compute operation from zero to
nonzero HBM demand activated the shared coarse HBM arbiter and changed both
no-edge JCTs to `C + D`. That HBM check is a fatal structural guard, not an
extra behavioral score.

This is the live graph-to-result metric relation. CORE-5 remains the successor
that reduces the completion stream through `StepResult` to TTFT and TPOT. This
study does not claim that later reduction.

## B. Eight GPU-affine RNICs

Each active GPU submitted two 1 MiB WQEs through its own QP and affine RNIC.
The second WQE made FIFO order observable. Changing active GPUs from one to
eight did not change phase JCT, while aggregate useful throughput changed from
`R` to `8R`. Doubling the per-port rate halved JCT exactly.

| Active GPUs | Rate (Gbit/s) | JCT (ps) | Useful throughput (Gbit/s) |
|---:|---:|---:|---:|
| 1 | 200 | 83,886,080 | 200 |
| 1 | 400 | 41,943,040 | 400 |
| 8 | 200 | 83,886,080 | 1,600 |
| 8 | 400 | 41,943,040 | 3,200 |

All four affinity/FIFO guards passed. Every source used only
`node-0:rnic-<gpu>`, SQ post sequences were `[1, 2]`, and the second grant
equaled the first completion. A node-global serialization cursor would have
failed both `N=8` cells by eight times; that defect was not excluded by graph
construction.

## C. Tail attribution and asynchronous control

At 400 Gbit/s, the two-WQE physical control phase quiesced at
`41,943,040 ps` in every cell. Asynchronous mode released the graph at the
independent `10,000,000 ps` compute anchor, while synchronous mode waited for
control delivery. The signed synchronous penalty was exactly
`31,943,040 ps` under both class labels.

| Mode | Class | JCT (ps) | Quiescence (ps) | Additive visit wait (ps) |
|---|---:|---:|---:|---:|
| asynchronous | 3 | 10,000,000 | 41,943,040 | 167,772,160 |
| asynchronous | 9 | 10,000,000 | 41,943,040 | 167,772,160 |
| synchronous | 3 | 41,943,040 | 41,943,040 | 167,772,160 |
| synchronous | 9 | 41,943,040 | 41,943,040 | 167,772,160 |

The additive wait is eight simultaneous second-WQE waits. It is exactly four
times synchronous wall JCT, so treating it as a critical-path delay would have
failed conspicuously. Every operation-level selected path conserved launch,
device queue, service, delivery, and separately named external dependency
time. `QUEUED` events matched eligibility, `STARTED` events matched grants, and
no returned event exceeded physical quiescence.

Class-label permutations retained the same canonical schedule hashes:
`a6441a9f...b76964` for asynchronous mode and
`b796f75e...05a7435` for synchronous mode. The omitted label echo is
accounting input, not scheduling evidence.

## D. Identity arbitration

The mixed compute, DMA, control, and ring graph produced canonical SHA-256
`5ffcaf893631c4195146369dac70ef9a9ec83c3fc6cdc69e038b2e9e86bd9de8`
in all four cells:

- arbitration omitted with baseline labels;
- explicit identity with baseline labels;
- arbitration omitted with permuted labels;
- explicit identity with permuted labels.

Canonical bytes included event order and timestamps, every queue visit, wait
and service-byte counter, WQE identity and sequence, GOAL tag, JCT,
quiescence, and the zero random-draw count. They excluded only the input label
echo. This passes the policy-seam bypass-preserves-baseline check.

## Genuine-risk fraction

All 37 scored rows or relations were judged genuinely at risk, but for
different mechanisms. The fraction is reported per family rather than as a
claim that all rows are statistically independent.

| Family | At-risk scored checks | Scored checks | Fraction | Plausible competent failure |
|---|---:|---:|---:|---|
| A | 6 | 6 | 100% | A valid DAG executor could serialize independent lanes or release an edge at the wrong boundary. |
| B | 12 | 12 | 100% | A coherent first implementation could use one node-global NIC cursor, lose a WQE, or scale bytes instead of ports. |
| C | 16 | 16 | 100% | Logical completion could be conflated with quiescence, or additive visits could be charged to wall time. |
| D | 3 | 3 | 100% | An otherwise legal policy seam could consult priority when identity is selected or reorder equal-ready work. |

The throughput rows share the JCT denominator but also use the actually
projected WQE cardinality and byte total; duplication or loss could fail them
even when the rate relation passed. Exact-oracle and relational counts are
still disclosed separately above so this dependence is not hidden.

## Scope and residual work

The structural native-session test proves mode exclusivity and semantic
delegation through a deterministic session double. It does not claim the
composed htsim path, which remains HTSIM-9. The bypass path is the sole live
cross-node authority in this slice. CORE-5 owns `StepResult`, virtual-clock,
TTFT, and TPOT reduction. CORE-9 owns the structural bookkeeping schema that
removes the v1 remote-RQ compatibility parent.

The owning module registry records four explicit residuals: CORE-11 replaces
whole-operation HBM exclusion with calibrated shared-bandwidth service;
CORE-12 admits kernels that become ready during an active batch; CORE-13
replaces the flat intra-node serializer with calibrated compute-owned NVLink
service; CORE-14 generalizes beyond the fixed eight-GPU/eight-RNIC mapping.
None is silently claimed by this coarse profile.
