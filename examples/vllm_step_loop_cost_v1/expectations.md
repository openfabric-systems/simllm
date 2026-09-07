# vLLM step-loop cost attribution expectations

This pre-run freeze attributes the CPU cost of the pinned vLLM 0.27.1
conformance driver. DEPLOY-21 owns the deployment cost question. This study
does not optimize either loop or certify the surrogate.

## Source and workload contract

Use the unchanged `wall_cell`, `_construct_live_engine`, `_engine_environment`
and `_drive_live` functions from
`examples/surrogate_conformance_v1/run_study.py`. Read its unchanged
`study_config.json`, including the governing 0.27.1 amendment. Record their
SHA-256 hashes and verify the scheduler source hash
`c67bda2886b52865ddafabaae7d797c359e930752f374421a33e537d94a5f45a`.
The model configuration hash and revision must match that configuration.

Retain all 128 frozen request identities, arrival times, priorities, prompt
token IDs and output lengths byte-for-byte. Each prompt has 64 tokens and
each request emits 32 tokens. Do not reconstruct or shorten the population.
The reference has `max_num_seqs=16`, `max_num_batched_tokens=512`, block size
16, 2,048 blocks, prefix caching off and maximum model length 128.
Construction, shutdown and file capture are outside the timed region.
Admission, driver bookkeeping and output collection are inside it, exactly
as in the conformance wall-time method. A complete-loop duration is not a
single-step duration. Count actual nonempty engine steps separately.

The driver disables detokenization. Its zero detokenization work is an
inactive-path observation, never evidence of fast text decoding. The
simulated executor/model-runner stub and synthetic pricing remain unchanged.
There is no device computation or allocation in this study.

## Sweep, repetitions and clock

Run the reference plus the Cartesian product of concurrency caps
`max_num_seqs={32,64,128}` and token budgets
`max_num_batched_tokens={256,512,1024}`. Set the resolved scheduled-token
budget to the same value. Keep the entire request population and every
other input fixed. These caps are configuration coordinates, not assertions
that every step runs that many requests. Record actual scheduled and running
counts and identify all-decode steps reaching the cap.

For every cell and timing arm, construct fresh engines for two discarded
whole-workload warmups followed by seven measured whole-workload repetitions.
Run the uninstrumented arm first and then the phase-timed arm. No selective
discard, additional warmup or retry based on measured speed is allowed.
Use `time.perf_counter_ns`, a monotonic host clock, around the complete
driver and around phase calls. Preserve each repetition and every step in
append-only bulk evidence. Record clock resolution, Python version, CPU
model, available processor count and affinity without personal paths.

Run exactly one additional cProfile pass on the original reference after
the timing sweep, with construction and shutdown outside the profile.
Retain function self time, cumulative time and call counts. Profiling is
diagnostic and does not enter phase medians or stability checks.

## Exclusive attribution

Measure these phases, with nested calls removed from their parent phase
instead of counted twice:

1. `Scheduler.schedule`, excluding nested cache allocation/free work.
2. Key/value (KV) cache manager allocation and free operations, including
   their nested pool work exactly once. Count actual blocks allocated and
   freed as well as calls. Separate allocation from free in supporting data.
3. Simulated executor/model-runner execution and sampling, including its
   existing record translation and synthetic pricing.
4. Scheduler output updates and frontend output processing. Report their
   separate components and the disabled detokenization path.
5. Engine-step residual outside these calls, measured through the enclosing
   frontend step; driver residual outside engine steps is a separate row.

Use elapsed call intervals and an exclusive nesting stack. Preserve inclusive
diagnostics but never add them to exclusive phase totals. Time request
submission and driver output collection separately if the profile identifies
them as material. Report a ranked table for complete-loop cost and a separate
per-step view, retaining the distinction between weighted means and medians.

## Expected relations and sanity bounds

- R1, running requests: at each fixed budget, median inclusive scheduler
  cost on all-decode steps at the cap increases from 32 to 64 to 128 requests.
  Each doubling is expected to cost between 1 and 3 times as much. Report
  unavailable all-decode populations as an unevaluated relation, not a pass.
- R2, block work: allocation/free cost grows with blocks handled per step.
  Compare nonzero block-count groups within each operation. For groups whose
  block counts differ by at least twofold, the larger group's median call
  cost should be greater, with ratio between 1 and twice the block-count
  ratio. Use actual new/freed blocks, not reserve capacity or token counts.
  A larger budget may batch more prompt blocks into fewer steps; no strict
  complete-loop speedup follows from that alone. Report unidentifiable or
  empty comparisons as such rather than inventing a positive slope.
- R3, attribution: the exclusive phase sum and both residuals reconstruct
  each instrumented complete loop within 5 percent. This partition check is
  structural and unscored. Independently compare the median instrumented
  total with the median uninstrumented total; an overhead above 5 percent
  prevents interpreting instrumented absolute times as unchanged loop costs.
- R4, stability: within each seven-repetition arm and cell, the median of
  repetitions 1 to 3 and the median of repetitions 5 to 7 must each be within
  10 percent of the seven-run median, and within 10 percent of each other
  using the seven-run median as denominator. This is a fatal guard.

Physical floor before reading timings: a scheduler visiting N requests must
perform at least N Python attribute accesses; use N times an independently
measured attribute-access cost as an operational lower-bound sanity check.
Measure seven batches of one million repeated attribute accesses, report
the minimum per-access elapsed cost and the measured scheduler nanoseconds
per visited request. A Python microbenchmark includes loop overhead and is
not a universal hardware lower bound; disclose that limitation.

Physical ceiling before reading timings: a nonoverlapping CPU phase cannot
exceed its enclosing elapsed interval; its per-request average cannot exceed
that interval divided by the actual visited count. There is no finite
first-principles host wall-time ceiling under arbitrary operating-system
preemption. Do not invent one. Bound the loop from below by the sum of its
nonoverlapping phases and evaluate timing plausibility using source call
counts, request/block scaling and the independent function profile.

## Fatal guards and reporting

Source/workload hashes, exact resolved configuration, unique identities,
4,096 emitted tokens with the frozen token identity, unchanged step-record
and output bytes between timing arms, no premature stop, absence of device
work, nonnegative exclusive intervals, phase nesting and interval containment,
and R4 stability are fatal and unscored. A violation makes the run void,
retains all evidence, leaves DEPLOY-21 open and prevents a behavioral score.
Separate run configurations, relation families/instances, structural guards
and deterministic unit tests. Do not combine their denominators.

Select the supplied interpreter through `SIMLLM_VLLM_PYTHON` in gitignored
local configuration. Do not change its environment. If it cannot import,
stop after this freeze and report the exact failing import. Select existing
model assets explicitly and keep logs, caches and profiles under
`SIMLLM_DATA_ROOT`. Do not fetch assets or repair the supplied environment.

The result cites this expectations-only commit, reports the ranked phases
and names the dominant function supported by the profile. It distinguishes
pinned scheduler cost from driver or simulation-stub inefficiency. It states
what ran, what came out, the consequence for DEPLOY-21 and what remains
unproven. No new stable task ID is registered by this worker.
