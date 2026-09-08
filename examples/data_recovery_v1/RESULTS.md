# Wide-incast DATA recovery

The repeated 96-configuration packet study establishes prompt recovery for
the width-64 all-to-all and completion of all six formerly failing pipeline
cases. With both new selections enabled, width 64 completes in **178.8352
microseconds at 400 Gbit/s** and **328.8672 microseconds at 200 Gbit/s**, below
the frozen one-millisecond target. Node-local pipeline phases take 3.68 to
3.69 milliseconds; rail phases take 6.45 to 6.47 milliseconds. All retain the
same eight-retry limit, buffer capacities, inputs and engineering budgets.

This evidence, the paired backend pin and the native/Python gates close
HTSIM-41 and HTSIM-40's DATA-recovery remainder. TRAF-88 becomes unblocked
for its separate topology and queue-attribution study. The original control
study and the first constant-probe study remain **void**. This result does
not close BACK-38's physical request execution or TRAF-8's serving integration,
does not calibrate request time to first token (TTFT) or time per output token
(TPOT), and does not change the default recovery selection.

## Why the retry schedule matters

A receiver requests a missing original packet when later DATA or RETIRE
exposes its sequence gap. If that retry is also lost, another packet need
not reveal the loss. The sender's optional timer therefore probes that exact
missing extent using another physical retry. Every copy traverses the normal
route, source and receiver serializers, and finite switch and receive storage.

Before the receiver's first physical nonempty grant arrives, an optional
per-flow wire-byte budget U limits both original and retry DATA. The declared
fan-in F must satisfy F*U <= B for the unchanged shared leaf-pool capacity B.
This is an input envelope for the named pool, not a measurement of occupancy
or a proof for every pool along the path. Maximum wire packets are 4160
bytes; B is 1048576 bytes. At width 64, F=448 gives U=2340, so even the first
maximum-size packet waits for an actual grant. The pipeline declarations are
F=113, U=9279 for node-local attachment and F=224, U=4681 for rail attachment.

The first implementation used a constant 40-microsecond probe interval.
That study is void: the twelve recovery-only and combined pipeline runs
exhaust their retry allowance. A diagnostic-only replay of one node-local
case finds all eight retry copies of one extent lost before usable receiver
admission. Exhaustion occurs at 375.7952 microseconds, while the original
117440512 payload bytes need at least 1.17440512 milliseconds to cross the
busiest 800 Gbit/s uplink cut. The fixed intervals consume the retry allowance
while the original burst is still draining. This finding applies to the
diagnosed extent; the other eleven failures are not assigned that detailed
cause without matching diagnostics.

The prospective successor adds `exponential` beside `none` and `deadline`.
Its seven probe intervals are 40, 80, 160, 320, 640, 1280 and 2560
microseconds, measured from each preceding retry's physical serialization
end. Their sum is 5.08 milliseconds. A receiver negative acknowledgement
still requests recovery immediately. The final legal retry keeps the original
50-millisecond timeout, allowing its physical resolution to arrive; no ninth
retry is created. Late authenticated retries use actual-arrival release in
the same finite receive store. A physical resolution closes only its own
extent, cancels queued copies and timers, and lets already routed copies drain.

## Frozen chronology and provenance

The initial expectations-only commit is
`7904e8f09a4174f7320a0107ac676660f3dfcf02`, before implementation and execution.
[Its contract](expectations.md) and [constant-probe results](constant_results.json)
are retained without rescoring. Backend implementation `27504a0` produced that
void study. Diagnostic-only commit `c75b6a6` adds state reporting and precedes
the separate diagnostic replay; it changes no recovery timing.

The final successor expectations-only commit is
`fc8614599e51a05d23535f566382368639f52043`, before exponential implementation
and its first run. [The successor contract](expectations_backoff.md) keeps
every original physical floor, engineering budget, identity boundary and
retry limit. It records the longer cumulative probe allowance separately,
without increasing the acceptance budget. No history or earlier result was
rewritten to create this chronology.

The executed backend source is
`96d5aa821f96c86901cce6f3b81609740d285558`, binary SHA-256
`4e6e75d47a603289747d7c83a09ea30db0d7a15e781394bb3a00504e21ac1ac0`.
The executed simllm source is `3da33dfab1bd5db0610262abcde382ad50aacd81`.
The publication pin `ac3c9fdc1f2d2621ce63545f8d0ddbcfd320c018` follows the
design-only commit `3bd3ac3b71dc833a9d149820094085d9f5e6127f` with a Windows
test-portability repair: two file readers close before deletion and two
hash-locked topology fixtures retain LF checkout bytes. Production sources,
numerical oracles and the rebuilt executable SHA-256 remain identical to the
executed study. The complete post-repair native suite passes all 497 tests;
the study's original run record and binary identity remain unchanged.
[results.json](results.json) records input, reference, runner, wrapper and
binary hashes plus every configuration outcome. Raw logs and completion CSVs
remain outside Git. Published JSON replaces local artifact roots with project
environment-variable references; numeric fields and hashes retain their values.

## Physical bounds and observed phases

The following bounds precede reading the new observations. At width 64,
56 incoming flows of 65536 payload bytes per receiver require at least
77.400320 microseconds at 400 Gbit/s and 150.800640 microseconds at 200 Gbit/s,
including four microseconds of propagation. Pipeline cut serialization plus
propagation gives 1.17840512 milliseconds for node-local attachment and
1.76560768 milliseconds for rail attachment. The unconditional ceiling is
unbounded: finite storage and a finite retry allowance do not guarantee success.

The frozen engineering budget is a separate workload acceptance condition,
9*S + 8*(4*K + Q), with S the receiver/cut serialization floor and Q three
shared-buffer drains plus eight propagation hops and one maximum-packet source
serialization. Its pipeline values are 11.45762816 and 16.7424512 milliseconds.
Width 64 also has the stricter one-millisecond target. Every complete combined
phase sits above its physical floor and below its unchanged engineering budget.
Physical draining ends another 3.83 to 3.92 microseconds later, also below the
budgets; width-64 draining is below 182.758 and 332.699 microseconds.

Times in the tables are microseconds. Flow completion time (FCT) percentiles
describe flows within one configuration, not requests. CN/NN is the ratio of
the complete physical phase to its identical ideal-profile GOAL phase.

| Width | Gbit/s | Phase | FCT p50 | FCT p99 | CN/NN phase | Physical probes |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 400 | 26.0832 | 20.9024 | 26.0832 | 2.287679 | 0 |
| 8 | 200 | 47.3664 | 37.5008 | 47.3664 | 2.276880 | 0 |
| 16 | 400 | 49.6832 | 34.4384 | 49.5072 | 2.397838 | 0 |
| 16 | 200 | 91.2864 | 59.5616 | 87.6064 | 2.314564 | 0 |
| 32 | 400 | 89.3952 | 60.5664 | 83.6032 | 2.271404 | 0 |
| 32 | 200 | 154.6144 | 104.9568 | 142.5344 | 2.015476 | 0 |
| 64 | 400 | 178.8352 | 121.9904 | 159.1392 | 2.333737 | 12 |
| 64 | 200 | 328.8672 | 222.9344 | 293.4784 | 2.174173 | 54 |

| Attachment | Pipeline depth | Phase | FCT p50 | FCT p99 | CN/NN phase |
|---|---:|---:|---:|---:|---:|
| Node-local | 2 | 3689.0912 | 3397.6032 | 3612.2912 | 6.164299 |
| Node-local | 4 | 3678.9472 | 3396.8832 | 3629.3472 | 6.147349 |
| Node-local | 8 | 3694.0192 | 3399.3952 | 3599.9776 | 6.172533 |
| Rail | 2 | 6455.4272 | 6133.5232 | 6384.4832 | 10.786717 |
| Rail | 4 | 6468.4416 | 6114.6912 | 6358.2112 | 10.808463 |
| Rail | 8 | 6453.1392 | 6128.0352 | 6380.5312 | 10.782894 |

Halving bandwidth increases combined phase duration by 1.73 to 1.84 times
across widths 8 to 64. Serialization terms double, while propagation and probe
intervals stay fixed; changed packet scheduling also changes retry work.
The contract does not require exact twofold total scaling. Increasing pipeline
depth need not monotonically lengthen a phase dominated by the same 896 expert
transfers. Every pipeline send follows its predecessor receive by exactly
1.002 microseconds, including the configured one-microsecond compute step.

The receiver-prefix byte floors hold for every combined flow population, with
a minimum ratio of 1.54819 over all configurations. Some shared-phase per-flow
CN/NN ratios are below one, down to 0.55537. Those ratios remain diagnostic:
the physical scheduler and ideal fair allocator may finish shared flows in
different orders. No incomplete phase is reconstructed from its surviving
rows. Complete-phase ratios and receiver-prefix byte floors retain their
fatal physical meaning.

## Conservation and evidence classes

The successor executes 96 configurations from 52 retained inputs: 88 complete
and eight legacy controls reproduce their expected diagnostic failures.
There are no fatal findings or behavioral budget findings. The initial
constant study separately has 76 completed configurations, eight expected
diagnostic failures and twelve fatal pipeline failures; its verdict is void.

The exact CSV oracles are 44 protected legacy configurations with the new
mechanisms disabled and two predeclared dormant-recovery configurations.
All are byte-identical to their corresponding references. Native dormant
initial-window checks additionally preserve event order, timestamps, random
state and exact bytes. These identity guards are unscored.

The consumer has two behavioral relation families: fourteen combined
engineering-budget instances and two width-64 submillisecond instances.
All are within their frozen bands. Four bandwidth contrasts and mechanism-arm
contrasts remain explanatory diagnostics, not additional acceptance points.
Native test counts, configurations, exact oracles and fatal guards are not
added into one score.

An independent review of raw GOAL inputs and completed CSVs checks the exact
message multiset, unique flow identities, payload conservation and timestamp
causality. In all fourteen combined configurations, physical retransmissions
equal fabric drops plus rejected late packets plus ignored duplicates.
Probe wire bytes equal physically dispatched probe count times 4160 and are
a subset of total retransmission bytes. Width 64 dispatches 12 probes at
400G and 54 at 200G. Pipeline probe counts range from 295895 to 555820, so the
completion improvement is not a claim of lossless or inexpensive recovery.
The maximum observed retry count is two at width 64, seven on node-local
pipelines and eight on rail. Control reserve occupancy peaks at 3328 of
131072 bytes and returns to zero. All combined configurations physically
quiesce with no violated finite-storage or lifecycle guard.

The native schedule checks independently vary the control deadline, probe
multiple and consecutive lost retry count. They verify exact exponential
intervals, checked overflow, legacy-timeout independence on successful paths,
physical resolution around the final nominal probe boundary, permanent-loss
failure at the final long timeout, and no extra retry. Earlier constant-policy
race fixtures cover stale controls, queued cancellation, routed duplicates and
independent gaps. The broader consumer supplies the separate packet-outcome
evidence; native fixtures alone do not establish workload completion.

## Reproduction and gates

Configure `SIMLLM_HTSIM_SOURCE`, `SIMLLM_HTSIM_BUILD`, `SIMLLM_HTSIM_RNIC`,
`SIMLLM_HTSIM_RNIC_BASE`, `SIMLLM_DATA_ROOT`,
`SIMLLM_CONTROL_COLLECTIVE_REFERENCE` and `SIMLLM_CONTROL_PIPELINE_REFERENCE`
in ignored local configuration. Use a fresh external output directory:

```sh
. ./.env.local.sh
python examples/data_recovery_v1/run_study.py --probe-policy exponential \
  --out "$SIMLLM_DATA_ROOT/data_recovery_backoff_v1" --workers 4 --timeout-s 3600
```

The default `constant` study selection preserves the original experiment.
`--resume` requires matching locked inputs, selected policy, expectations and
execution digests; interrupted attempts require a fresh output directory.
Observations are written before exact-oracle guards. Exit zero means the
consumer contract is valid; it does not stand in for integration gates.

Final gates:

```text
ctest --test-dir "$SIMLLM_HTSIM_BUILD" --output-on-failure -j8
100% tests passed, 0 tests failed out of 497
ruff check .
All checks passed!
pytest -q
4629 passed, 29 skipped in 1482.90s (0:24:42)
# Current wrapper with the executed native binary configured
pytest -q tests/test_htsim_rnic.py
70 passed in 0.42s
# Final documentation, portability, task registry and authorship checks
29 passed in 2.76s
python scripts/check_docs_format.py
OK: 11 module doc(s) match docs/modules/FORMAT.md
python scripts/task_progress.py --check
task-progress block and module-status open counts are current
```

The full Python suite leaves optional external environments unconfigured;
its skips are declared optional integrations. The separate wrapper run uses
this study's actual native binary. The final backend documentation commit
changes no executed C++ source; the final simllm documentation and gitlink
changes are checked separately after the full code suite.

The gates use CPU simulation only. The packet profile remains a modeled
mechanism with deterministic controller timing, not a calibrated network
interface firmware policy. TRAF-88 still owns load-dependent tagged-hop queue
evidence and separation of contention from topology-dependent control timing.
BACK-38 and TRAF-8 still own physical request-metric reachability. CORE-48 is
the next separate coarse-runtime task; this study makes no change to it.
