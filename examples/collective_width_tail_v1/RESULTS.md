# Collective width tail: void study with findings

S3 ran the frozen tensor-parallel ring and expert-parallel all-to-all sweep,
then attempted the corresponding steps through `HtsimStepSink` and its request
completion reducer. The study is **void**: 122 aligned physical flows beat the
frozen ideal per-flow lower bound, and both width-64 physical all-to-all runs
terminate on fatal control-message loss.

COMP-9 remains open. The supported ideal steps expose a useful bottleneck:
at width 64 and 400 Gbit/s, fabric service owns **85.892%** of the two-ring
step and **60.515%** of the single-engine expert step. BACK-38 still blocks all
physical step shares. These findings locate candidate mechanisms and failures;
they close no task, validate no held-out request tail, and move no milestone.
The deterministic compute contract and shipped defaults remain unchanged.

## Frozen record and evidence classes

The pre-run expectations-only commit is
`aae9adf0d29c875bdf2c7aeb49dbf2f8c2ec966a`.
The full hash is also stored in `results.json`. The freeze precedes both the study implementation
and its first run. No measured result was added to that commit and no history
was rewritten. Backend pin is `617ce20`; executable hashes are in the result
provenance. The final report distinguishes the original simulation script hash
on each cell from the current analysis-script hash.

- Run configurations: 32 standalone attempts, with 30 completed and two fatal
  backend exits; 32 step attempts, with 16 completed ideal steps and 16 physical
  steps rejected by BACK-38. Rejected steps have null timing and share fields.
- Raw completion evidence: 54,432 standalone flow rows and 42,980 step-artifact
  flow rows in separate bulk files. These are flow populations, not independent
  workload samples or request-percentile observations.
- Exact oracles: 24 rows. Eight standalone ring points and four standalone rate
  relations disagree with their frozen zero-tolerance prediction. The eight
  ideal ring step points and four corresponding rate relations agree exactly.
- Behavioral relations: five families, 95 parameterized instances. The
  behavioral score is uninterpretable because fatal guards failed. Individual
  diagnostic check records remain available; no pass fraction is reported.
- Fatal findings: the aligned-baseline guard fails at F=32 at both rates;
  physical quiescence is not established at F=64 at either rate because of
  control lifecycle loss. No fatal guard was declared survivable.
- Offline tests exercise quantiles, identity matching, physical floors, void
  handling, canonical step pair ordering, topology rate changes and the
  state-preserving execution guard. They do not run a simulator or validate
  backend physics.

The first attempt supplied the selected expert ranks in rail-major order to
the step interface. At widths above eight, its generated pair table violated
the required source-major order before any backend started. The runner now
sorts the same selected expert rank set for the step interface. Membership,
per-destination bytes, physical placement and engine rank remain identical.
The twelve rejected inputs and their diagnostics remain in bulk as
`original-attempt.json` and `original-error.txt`. Only those inputs were retried.
No standalone measurement was rerun or replaced. This is an input-encoding fix,
not a change to a frozen behavioral band.

## Physical bounds before the measured numbers

Flow floor: payload bytes divided by endpoint rate, strengthened by 2 us
propagation on the ideal profile or 4 us across leaves on the physical Clos.
Flow ceiling: no finite physical upper bound follows from link rate when
control, arbitration and queueing can delay completion; ideal conditional
ceilings are recorded per configuration in `results.json`.

Ring phase floor: `2(W-1)*(S/W*ps_per_byte + propagation_ps)`.
Ring ideal ceiling: the frozen packetized point plus one full packet slot per
round. At W=64 and 400G these bounds are 293.287680 and 314.899200 us; the
measured ideal phase is 304.541000 us, inside the interval. For the physical
arm the corresponding floor is 545.287680 us and the ceiling is unbounded;
its measured phase is 1429.404800 us.

All-to-all phase floor: `D*65536*ps_per_byte + propagation_ps`, where
`D=7F/8` after excluding same-node pairs. At F=64 and 400G the ideal floor is
75.400320 us; the conditional global-serialization ceiling is 4773.104000 us.
The ideal phase is 76.630400 us, just above the floor. The physical lower bound
is 77.400320 us; no valid completion exists to compare with it.

Step floor: the fixed 100 us compute interval plus two corresponding ideal
phase floors. Step ceiling: 100 us plus two conditional phase ceilings.
At width 64 and 400G the ring interval is [686.575360, 729.798400] us and the
expert interval is [250.800640, 9646.208000] us. Their measured step latencies,
708.832000 and 253.260800 us, lie inside those bounds. Collective-share floors
are respectively 85.435% and 60.128%, with ceiling 100%; measured shares are
85.892% and 60.515%. The compute floor and ceiling are both 100 us by input.
This denominator is a synthetic control, not a calibrated model's GPU service.

Three independent checks frame interpretation. Endpoint serialization rules out
impossible goodput; all completed flow minima exceed their payload floor, with
the tightest ratio 1.429. Dependency depth explains why narrower ring chunks
can complete faster individually while the entire collective becomes slower.
NVIDIA's [NCCL bandwidth accounting](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
provides the independent collective transfer factor `2(W-1)/W`: the ideal
width-64 400G ring implies about 6.779 GB/s per-rank bus bandwidth, below the
50 GB/s endpoint limit. This external check validates units and transfer work,
not a deployment latency or held-out tail prediction.

## Standalone raw flow completion times

All numbers below are microseconds. FCT means flow completion time. The p50
and p99 are nearest-rank percentiles within one configuration. The payload
floor applies to both profiles in that row and is printed beside every median.
NN is the packetized ideal endpoint manifold. CN is the physical two-tier Clos.
The NN propagation is 2 us; CN cross-leaf propagation is 4 us. Thus inflation
includes the different path propagation as well as physical control and queues.

| Ring W | Gbit/s | Payload floor | NN p50 / p99 | CN p50 / p99 | NN phase | CN phase |
|---|---|---|---|---|---|---|
| 8 | 400 | 2.621440 | 4.745600 / 4.745600 | 13.832600 / 14.331800 | 66.451400 | 196.387200 |
| 8 | 200 | 5.242880 | 7.491200 / 7.491200 | 23.653400 / 24.642200 | 104.889800 | 336.604800 |
| 16 | 400 | 1.310720 | 3.414400 / 3.414400 | 12.411800 / 12.667800 | 102.461000 | 375.027200 |
| 16 | 200 | 2.621440 | 4.828800 / 4.828800 | 20.821400 / 21.487000 | 144.893000 | 630.035200 |
| 32 | 400 | 0.655360 | 2.748800 / 2.748800 | 11.666200 / 11.912600 | 170.486600 | 727.532800 |
| 32 | 200 | 1.310720 | 3.497600 / 3.497600 | 19.324800 / 19.823000 | 216.912200 | 1206.534400 |
| 64 | 400 | 0.327680 | 2.416000 / 2.416000 | 11.253400 / 11.496600 | 304.541000 | 1429.404800 |
| 64 | 200 | 0.655360 | 2.832000 / 2.832000 | 18.495000 / 18.991000 | 356.957000 | 2354.476800 |

| All-to-all F | Gbit/s | Payload floor | NN p50 / p99 | CN p50 / p99 | NN phase | CN phase |
|---|---|---|---|---|---|---|
| 8 | 400 | 1.310720 | 11.152000 / 11.401600 | 20.902400 / 26.083200 | 11.401600 | 26.083200 |
| 8 | 200 | 2.621440 | 20.304000 / 20.803200 | 37.500800 / 47.366400 | 20.803200 | 47.366400 |
| 16 | 400 | 1.310720 | 20.137600 / 20.720000 | 31.324800 / 45.203200 | 20.720000 | 51.827200 |
| 16 | 200 | 2.621440 | 38.275200 / 39.440000 | 58.390400 / 84.934400 | 39.440000 | 89.078400 |
| 32 | 400 | 1.310720 | 38.192000 / 39.356800 | 51.718400 / 75.555200 | 39.356800 | 79.891200 |
| 32 | 200 | 2.621440 | 74.384000 / 76.713600 | 100.310400 / 141.302400 | 76.713600 | 150.342400 |
| 64 | 400 | 1.310720 | 74.300800 / 76.630400 | fatal loss | 76.630400 | unavailable |
| 64 | 200 | 2.621440 | 146.601600 / 151.260800 | fatal loss | 151.260800 | unavailable |

## Refutations and their limits

Aligned flow normalization retains 2,592 flow pairs. For a ring, only the first
round has matching start times, so later rounds retain raw FCT and compare via
phase makespan. Every completed all-to-all has matching zero-time releases.
At F=32, 47 physical flows at 400G and 75 at 200G violate the frozen baseline
floor. The minimum ratios are 0.759090 and 0.575342. The latter is flow
0 to 8, tag 1000, if identified by the raw table rather than by percentile.
No payload-serialization floor fails. A fair allocation is not in general a
lower bound on each flow under a different scheduler: some flows can finish
earlier while the phase finishes later. That is a possible explanation of the
refuted assumption, not a proved backend root cause or permission to relax the
frozen guard. The phase ratios at F=32 are 2.029921 and 1.959788.

The physical 2x target also misses: 304 retained aligned pairs exceed 2x, and
the largest ratio is 6.646328 at ring W=64, 200G. The 1.2x tighter target is
reported separately in JSON. These counts are diagnostic populations, not a
behavioral score after the fatal guard has made the run void.

At F=64, the backend reports `fabric dropped control lifecycle` and exits 2
at both rates. Its manifest declares fatal control loss without control
recovery; the buffer is the unchanged 1,048,576-byte default. No valid FCT or
phase makespan is manufactured from a partial completion CSV, and no buffer
increase or recovery-policy change is used to obtain a favorable result.
The captured diagnostics identify the failing lifecycle and flow in bulk.

The standalone exact ring formula omits a schedule detail. Exact-frontier
rendering inserts `calc 0` joins, and the pinned backend's
`htsim/sim/logsim-interface.cpp` executes them as 1 ns. There are `2W-3` such
inter-round joins before the final flow completes. The observed residual is
therefore exactly `(2W-3)*1000` ps: 13,000; 29,000; 61,000; 125,000 at either
rate. This source-based explanation is post-specified and does not convert the
eight failed exact points into passes. It also explains the four negative
rate-relation residuals: subtracting only propagation leaves a rate-independent
join term. The supported sink runs rounds as ordered artifacts and has no
inter-round GOAL join cost, so its frozen ring step values and serialization
scaling agree exactly. No backend timing was edited.

## Supported critical-path contribution

Before reading these shares, their bounds are [0,1], strengthened by the
per-cell floor and ceiling above and in JSON. The completion reducer selects
each ordered artifact's realized local/fabric maximum, then conserves the
step's elapsed time. Its collective and fabric shares coincide in all 16
completed steps. Local expert transfers have positive work above width eight,
but finish behind fabric service and contribute zero to the selected path.
Their masked work remains separately named and is never added to step latency.

| Width | Gbit/s | Ring step / TTFT us | Ring fabric share | Expert step / TTFT us | Expert fabric share |
|---|---|---|---|---|---|
| 8 | 400 | 232.876800 | 57.059% | 122.803200 | 18.569% |
| 8 | 200 | 309.753600 | 67.716% | 141.606400 | 29.382% |
| 16 | 400 | 304.864000 | 67.198% | 141.440000 | 29.299% |
| 16 | 200 | 389.728000 | 74.341% | 178.880000 | 44.097% |
| 32 | 400 | 440.851200 | 77.317% | 178.713600 | 44.045% |
| 32 | 200 | 533.702400 | 81.263% | 253.427200 | 60.541% |
| 64 | 400 | 708.832000 | 85.892% | 253.260800 | 60.515% |
| 64 | 200 | 813.664000 | 87.710% | 402.521600 | 75.157% |

TTFT means time to first token. Here it is one synthetic sampled prefill step,
so TTFT equals that step's latency. There is no sampled time-per-output-token
(TPOT) distribution. The expert step is the supported one-engine dispatch and
reverse combine, not the standalone full-population exchange. Its fan-in and
per-destination bytes still match the frozen endpoint-load axis.

This is `HtsimRequestMetricReducer` with `attribute_step_detail`, not the
`CriticalPathBreakdown` emitted by the coarse device runtime. The packet-level
sink does not supply that object, per-visit packet queue waits or matching
`CompletionEvent` segments. Their absence is explicit in the JSON. A whole
collective share is not the share of excess p99 latency; neither can identify
how much of a physical tail belongs to control, switch queues or endpoint work.
All physical step cells stay null because restarting the backend between
artifacts loses state. BACK-38 must provide that execution before this sweep
can report a physical step fraction without a second timing authority.

![Phase makespans, the per-flow tail, ideal step shares and the physical-over-ideal ratio](figures/collective_tail.png)

The figure shows diagnostic measurements from a void study. Top row: the
ring and all-to-all phase makespans against width for both profiles and both
link rates, with the frozen phase floors as gray lines (the ideal profile
sits on its floor; the missing physical width-64 all-to-all points are the
fatal control-loss exits), then the per-flow completion-time tail at 400G:
p50 and p99 for both profiles above the payload floor. Bottom row: the
supported ideal step shares for the two-ring and expert steps, then the
physical-over-ideal phase makespan ratio against the 2x comparator target;
the ring stays between 3x and 6.6x above the ideal at every width, the
all-to-all between 2x and 2.5x. PNG and vector PDF are generated from
`results.json`; only the analysis-script hash in that record changes when
the figure code changes.

## Reproduction and residual work

Configure gitignored `.env.local.sh` with `SIMLLM_HTSIM_BUILD`,
`SIMLLM_HTSIM_RNIC`, `SIMLLM_TXT2BIN` and external bulk `SIMLLM_DATA_ROOT`.
Then run:

```sh
. ./.env.local.sh
.venv/bin/python examples/collective_width_tail_v1/run_study.py --publish
```

Exit 2 means the study is void or incomplete, not that its evidence should be
discarded. A fresh external `--out` reproduces the full matrix. `--resume`
reuses saved cells with the same expectation digest; `--summarize-only`
rebuilds the compact report and figure without rerunning the simulator. Cell
hashes preserve the executed source version. Raw manifests retain local paths
only in bulk; the published projection uses relative artifact names.

COMP-9 retains held-out request-tail prediction and removal-of-contention
validation. The orchestrator must register or route the aligned-baseline
refutation, width-64 control loss and unavailable packet-level critical-path
segments against merged main; S3 creates no new task IDs. BACK-38 retains the
physical multi-artifact execution prerequisite. The 1 ns join finding needs a
future expectation that explicitly accounts for the chosen schedule boundary;
this run does not retroactively preregister that explanation. No production
code, module registry, README, backend pin or backend source changed.

## Validation

The final worktree gates completed without a GPU or network dependency:

```text
.venv/bin/ruff check .
All checks passed!
.venv/bin/pytest -q
4186 passed, 26 skipped in 325.21s (0:05:25)
.venv/bin/python scripts/check_docs_format.py
OK: 11 module doc(s) match docs/modules/FORMAT.md
```

Green code gates establish harness integrity; the numerical study remains void.
