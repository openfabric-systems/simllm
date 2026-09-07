# Controlled-host pre-run amendment for VLLM-51

This expectations-only amendment precedes implementation and measurement of
one fresh controlled attempt. The original expectations and CPU-distribution
amendment remain authorities for the workload, source pins and evidence
reductions. This amendment changes only host control and its provenance.
The result must cite this commit and both earlier freeze commits.

## Host protocol and repetition plan

The orchestrator supplied affinity mask `ff000000`, CPUs 24 through 31.
Before this freeze, `taskset -p $$` reported that mask and `nproc` reported
8. The worker must inherit this mask without changing it. Other wave workers
are assigned different cores; background jobs from other users still run on
unpinned cores. Affinity is not CPU exclusivity or control of shared caches,
memory bandwidth, frequency, power limits or operating-system preemption.

Record `taskset -p` for the measurement process, `nproc`, the affinity CPU
list and one-, five- and fifteen-minute load averages before and after the
timing sweep, including the diagnostic profile. Save the before snapshot
before any attribute microbenchmark or timed loop. Require the same frozen
eight-core mask at both boundaries. Configure only the child process with
one thread for OpenMP, MKL, OpenBLAS and NumExpr, and require PyTorch intra-op
and inter-op thread counts of one. The driver calls remain serial. Do not
alter the supplied interpreter, environment installation or CPU affinity.
No local test suite, plot renderer or other local study may run concurrently
with timing. Run deterministic tests before or after the entire sweep.

Retain the exact attempt-004 grid: reference cap 16, budget 512, plus the nine
cells in caps {32, 64, 128} by budgets {256, 512, 1024}. These already contain
the requested 32-, 64- and 128-running-request coordinates; use actual
all-decode steps reaching each cap, not configured caps as observed counts.
If a cell never reaches that population, report its comparison unevaluated.
Keep the original 128 requests, arrivals, token identities, prompt lengths,
output lengths and all remaining configuration fields unchanged.

For each of the ten cells, run two discarded whole-workload warmups then
seven measured loops in the plain arm, followed by two warmups and seven
measured loops in the instrumented arm. This is 40 warmups and 140 measured
loops. Then run exactly one additional reference cProfile pass. Preserve
every repetition, including slow ones, in a fresh directory selected under
`SIMLLM_DATA_ROOT`. No selective discard, speed-based retry, extra warmup,
arm reordering or timing-loop implementation change is allowed.

## Frozen relations and physical sanity before measurement

- R1: at fixed token budget, inclusive scheduler time on all-decode steps
  at the cap grows with running requests. A doubling from 32 to 64 or 64 to
  128 has expected cost ratio in (1, 3]. Missing populations are unevaluated.
- R2: KV allocation cost grows with actual nonzero blocks touched per call.
  Compare groups separated by at least twofold block count; the cost ratio
  lies in (1, twice the block-count ratio]. Report median nanoseconds per
  block for each group. Apply the original same rule to free operations,
  while constant block counts leave free-cost growth unidentifiable.
- R3: each exclusive phase and the two residuals are nonnegative, contained
  in their parent interval and reconstruct the measured instrumented loop
  within 5%. Report the separate sum-of-phase-medians residual. The absolute
  difference between instrumented and plain medians must also be at most
  5% before treating instrumented absolute costs as the plain-loop partition.
- R4: within each seven-run arm, each of the first-three and last-three
  medians differs from the whole median by at most 10%, and their mutual
  difference divided by the whole median is at most 10%. This unchanged
  stability guard is fatal and unscored.

Floor: a scheduler pass visiting N running requests needs at least N Python
attribute accesses. Before reading scheduler timings, benchmark seven batches
of one million accesses and use the minimum elapsed nanoseconds per access
as the operational floor. State N times that floor before reporting observed
per-request cost. The benchmark includes Python loop overhead, so this is
a sanity reference, not a universal hardware lower bound.

Every phase and KV operation has zero as its wall-time floor and its enclosing
nonoverlapping interval as its ceiling; per-block cost is bounded by that
interval divided by actual blocks touched. Profile self times are nonnegative
and sum within the full profiled interval; nested cumulative times must not
be summed. No finite host-loop ceiling follows from first principles under
arbitrary preemption. The workload emits exactly 4,096 output tokens: at
cap 16 at least 256 nonempty steps are needed, and the 12,160 total scheduled
prompt and continuation tokens give a loose 12,160-step ceiling with positive
progress and sufficient cache capacity. The maximum final context allocation
is 128 times six blocks, or 768, below the unchanged 2,048-block capacity.

## Fatal outcomes and publication

Keep every original source, configuration, identity, completion, device and
exclusive-accounting guard. Host-affinity or thread-count violations are
additional fatal guards. A stability or boundary host-control failure is
survivable only for completing and retaining diagnostic evidence: finish the
planned grid and profile, emit VOID with no behavioral score, and state that
the guard could not be met on this host under this protocol. No valid cell,
matching phase ranking or intact structural identity rescues a void attempt.
An unavailable supplied environment stops before measurement; do not repair
it. An instrumentation difference outside 5% separately prevents absolute
attribution even if the stability guard holds.

The saved monotonic intervals are the sole phase authority. The compact
publication, phase table, figure and per-request/per-block quantities are
explicit projections of those intervals; the independent profile only names
hot functions. A void run still supports source arithmetic, conservation,
diagnostic phase ordering and identification of candidate inefficiencies,
but cannot certify inherent cost, an optimization speedup or metric-live GPU
service. Retain the prior published void record and figures byte-for-byte,
publish the controlled record beside them, and rewrite RESULTS.md around
what ran, what came out, what changes for VLLM-51 and what does not change.
Do not edit DEPLOY-21. Remove VLLM-51 only if every acceptance clause is met;
otherwise replace only its entry with the exact remaining scope.
