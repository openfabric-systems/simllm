# CORE-4 coarse device runtime results

The corrected coarse `DeviceRuntime` passes all four experiment families.
Across 18 run configurations, 22 of 22 exact-oracle rows, 23 of 23 scored
behavioral relations, and 18 of 18 fatal structural guards passed. The runtime
changed `ExecutionGraph` JCT in the signed direction and exact amount predicted
by dependency, RNIC rate, synchronous control delivery, and predecessor-clipped
launch queueing. Omitted and explicit identity arbitration produced the same
canonical bytes under class-label permutation.

## Expectations and chronology

Commit `ea3961b` first recorded the dependency, original GPU-affine RNIC, and
tail-attribution families in `docs/modules/core.md`. Commit `37357cc` corrected
the RNIC scaling form, refined tail attribution, and added identity
arbitration. The final task-specific expectations-only ancestor is commit
`d43cddb` (`Freeze the coarse device runtime expectations`). That commit
contains only [expectations.md](expectations.md), follows its registered
parse-only dry run, and precedes the runtime, tests, study runner, and first
measured result. Its dry run explicitly asserted that `run_study.py` did not
exist, so no tracked or untracked study harness was present at freeze.

Integration review then identified the transactional and critical-path defects.
Commit `67cabda` (`Amend the runtime expectations after review`) appended the
two exact critical-chain cells and correction guards after a new check-only dry
run, but before any corrective implementation edit or corrected study run. The
original freeze remains unchanged in history. Original relations retain their
pre-registered status; the correction relations are review-triggered,
pre-correction expectations.

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
`1772189585247b2353683b25e0f6fd2de15ba36bf1e8b6d23c157a4ab0e16855`.
No bulk output is tracked in the repository.

## Evidence accounting

Evidence classes remain separate. Exact-oracle rows and behavioral relation
instances are both scored but are not added into one synthetic headline.
Conservation, authority exclusivity, frozen event/tag structure, and
configuration-forced zeros are fatal and unscored. Unit tests and the full
repository regression are separate executables.

| Evidence class | Result | Meaning |
|---|---:|---|
| Run configurations | 18 | Unscored parameter records |
| Exact-oracle rows | 22/22 pass | Graph JCT, quiescence, and useful throughput |
| Behavioral relation instances | 23/23 pass | Signed dependency, rail, rate, control, critical-path wait, and identity relations |
| Fatal structural guards | 18/18 pass | Authority, FIFO, conservation, origin, HBM, and GOAL-tag invariants |
| Focused runtime tests | 33/33 pass | Separate Python unit-test executable |
| Full repository regression | 480 passed, 3 skipped | Separate repository validation executable |

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

The harness compares each measured cell directly with these frozen integer
literals and separately asserts that the closed form recomputed from the sweep
inputs equals the same literal.

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
equaled the first resource release at `finished_at_ps`. A node-global
serialization cursor would have
failed both `N=8` cells by eight times; that defect was not excluded by graph
construction.

JCT and aggregate throughput were compared directly with the frozen integer
literals above. Their independently recomputed serialization and `N * R`
forms were also required to equal those literals.

## C. Tail attribution and asynchronous control

At 400 Gbit/s, the two-WQE physical control phase quiesced at
`41,943,040 ps` in every cell. Asynchronous mode released the graph at the
independent `10,000,000 ps` compute anchor, while synchronous mode waited for
control delivery. The signed synchronous penalty was exactly
`31,943,040 ps` under both class labels.

| Mode | Class | JCT (ps) | Quiescence (ps) | Additive visit wait (ps) | Critical-path queue (ps) |
|---|---:|---:|---:|---:|---:|
| asynchronous | 3 | 10,000,000 | 41,943,040 | 167,772,160 | 0 |
| asynchronous | 9 | 10,000,000 | 41,943,040 | 167,772,160 | 0 |
| synchronous | 3 | 41,943,040 | 41,943,040 | 167,772,160 | 20,971,520 |
| synchronous | 9 | 41,943,040 | 41,943,040 | 167,772,160 | 20,971,520 |

The additive wait is eight simultaneous second-WQE waits. It is exactly four
times synchronous wall JCT, so treating it as a critical-path delay would have
failed conspicuously. Every operation-level selected path conserved launch,
device queue, service, delivery, and separately named external dependency
time. `QUEUED` events matched eligibility, `STARTED` events matched grants, and
no returned event exceeded physical quiescence.

The review-triggered dependency-chain cells distinguish additive launch wait
from realized critical-path wait. In each graph, B's launch is submitted at
release, but only the tail after A completes contributes to the critical path.

| Launch service L (ps) | Graph JCT (ps) | Critical-path queue (ps) | Unclipped B launch wait (ps) |
|---:|---:|---:|---:|
| 10 | 150 | 10 | 120 |
| 20 | 180 | 20 | 140 |

The reported chain was exactly `(a, b)`. Summing its clipped operation segment
latencies produced `150 ps` and `180 ps`, respectively, so the corrected
decomposition conserved graph JCT without charging launch intervals concurrent
with predecessor service.

Class-label permutations retained the same canonical schedule hashes:
`3dcb49e9...9354e99` for asynchronous mode and
`52024d98...8738c4` for synchronous mode. The omitted label echo is
accounting input, not scheduling evidence.

## D. Identity arbitration

The mixed compute, DMA, control, and ring graph produced canonical SHA-256
`442c925c4fa2fe3e865e9b2932854bbe9f973765ef640578ad0aefbf59e4a6c3`
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

All 45 scored rows or relations were judged genuinely at risk, but for
different mechanisms. The fraction is reported per family rather than as a
claim that all rows are statistically independent.

| Family | At-risk scored checks | Scored checks | Fraction | Plausible competent failure |
|---|---:|---:|---:|---|
| A | 6 | 6 | 100% | A valid DAG executor could serialize independent lanes or release an edge at the wrong boundary. |
| B | 12 | 12 | 100% | A coherent first implementation could use one node-global NIC cursor, lose a WQE, or scale bytes instead of ports. |
| C | 24 | 24 | 100% | Logical completion could be conflated with quiescence, additive visits could be charged to wall time, or predecessor-overlapped launch work could be double-counted. |
| D | 3 | 3 | 100% | An otherwise legal policy seam could consult priority when identity is selected or reorder equal-ready work. |

The throughput rows share the JCT denominator but also use the actually
projected WQE cardinality and byte total; duplication or loss could fail them
even when the rate relation passed. Exact-oracle and relational counts are
still disclosed separately above so this dependence is not hidden.

## Scope and residual work

The structural native-session tests prove mode exclusivity, isolated staging,
atomic prepared commit, and rollback before retry through a deterministic
session double. They do not claim the composed htsim path, which remains gated
on HTSIM-9 and CORE-15. The bypass path is the sole live cross-node authority
in this slice. CORE-5 owns `StepResult`, virtual-clock, TTFT, and TPOT
reduction. CORE-9 owns the structural bookkeeping schema that removes the v1
remote-RQ compatibility parent. CORE-3 owns byte-carrying KV HBM lowering;
CORE-4 now rejects such READ/WRITE operations instead of reporting zero cost.

The owning module registry records six explicit CORE-4 residuals. CORE-11
replaces whole-operation HBM exclusion with calibrated shared-bandwidth
service; CORE-12 admits kernels that become ready during an active batch;
CORE-13 replaces the flat intra-node serializer with calibrated compute-owned
NVLink service; CORE-14 generalizes beyond the fixed eight-GPU/eight-RNIC
mapping; CORE-15 makes the composed native path live-reachable to changed
completion time; CORE-16 adds exact ring remainder chunking and wider
collision-free control-tag allocation. None is silently claimed by this coarse
profile.
