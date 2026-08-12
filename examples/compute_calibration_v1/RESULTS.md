# Turing compute calibration v1 results

Run on 2026-08-12. The calibrated provider decisively beats the roofline
bootstrap on held-out GTX 1660 Ti microkernels, but the study is an overall
failure because the frozen per-cell variation guard did not pass. The final
registered run passed all 67 genuine-risk instances and 6 of 7 fatal unscored
guard families. Three of 50 cells had coefficient of variation above the
frozen 2 percent ceiling. COMP-1 therefore remains open.

This is a Turing method anchor. It is not an H100, B100 or production Granite
table.

## Chronology and provenance

The expectations-only commit is
`50c22105a34db7e645e4d8ecdce7982a3c640cdb`. It precedes the implementation
and every target-kernel timing run. The check-only command ran before that
commit, validated 50 cells, 2,050 target rows and 67 scored instances, invoked
no CUDA or profiler tool, and created no output directory.

The final registered run observed implementation commit
`1e744ec9a085d2c84d8a3b0921a3a671e096fe09`. The full chronology is retained
because each failed harness assumption changed the capture pipeline rather
than the frozen matrix or acceptance bands:

| external output suffix | outcome |
|---|---|
| `compute-calibration-v1-failed-csv-preamble` | Nsight Systems captured the matrix; the parser rejected the public report's generation preamble before producing a table. |
| `compute-calibration-v1-failed-global-warmup` | The first parsed result exposed that warmups were separated from later cells by the rest of the matrix. One FP64 cell changed clock state during its measured samples and reached 9.04 percent CV. |
| `compute-calibration-v1-failed-repeat-report-discovery` | Immediate per-cell warmups and 50 repeated capture ranges succeeded; the runner incorrectly expected one unnumbered report instead of 50 numbered reports. |
| `compute-calibration-v1-failed-cell-variation` | Ordered report collation completed. Accuracy passed, but 2 of 50 CV guards failed. A final score-key typo raised only after `results.json` had been written. |
| `compute-calibration-v1` | The literal registered command reached the intended final acceptance assertion. Accuracy passed, but 3 of 50 CV guards failed, so the command exited nonzero. |

The official bulk run is below
`${SIMLLM_WAVE6_RUN_ROOT}/codex/comp1_compute_calibration/compute-calibration-v1`.
The compact tracked artifacts are:

| artifact | SHA-256 | role |
|---|---|---|
| [calibration.json](calibration.json) | `0be6dad653ff32a0f4667b5cb05f7ddaefdc06e8f6f1cea032bb8d285b42023f` | strict provenance, launch metadata, immutable split and all 2,050 raw target durations |
| [profile_table.json](profile_table.json) | `de7d91c373798f00714a0efa13f29fef8a2aeeea3a0a0ac208966f19a2f213c6` | 30 train-only family entries used by `ProfileTableProvider` |
| [results.json](results.json) | `88cd3dc341405b194af0b27d5d86dc341314666a8eceb39595d461f82cdbefde` | raw relation rows, distributions, guards and live-reachability evidence |
| external capture manifest | `2415883405a96aa8152d9a465254881bcc73c4e7af10ba59f1711cfb8699d6f9` | ordered hashes for 50 profiler reports and 50 public CSV exports |

The device was an NVIDIA GeForce GTX 1660 Ti, UUID recorded in the calibration
artifact, compute capability 7.5, driver 550.90.07 and 6 GiB memory. The tools
were CUDA 12.4.99, Nsight Systems 2023.4.4.54 and Nsight Compute 2024.1.0.0.
The benchmark and static SASS hashes are recorded in each artifact. Static
disassembly found both FP32 and FP64 implementations of all five named
families.

## What profiling works on this host

Nsight Systems attached successfully through the CUDA profiler API and CUPTI
activity tracing. It produced 50 nonempty reports, one for each registered
cell, and the public `cuda_gpu_trace` export accounted for exactly 2,050 target
kernel durations together with start time, launch dimensions, registers,
shared memory, device, context, stream and function identity.

Nsight Compute also attached to the benchmark process, but its basic counter
probe returned status 1 with `ERR_NVGPUCTRPERM`, reported that this user lacks
permission to access GPU performance counters, and profiled no kernels. The
loaded driver reports `RmProfilingAdminOnly: 1`. The exact unmet local
requirement is an administrator disabling that restriction or granting the
documented profiling capability, followed by a successful counter probe.
Changing the CUDA benchmark or pretending the absent counters are zero would
not satisfy it.

The activity-timing pipeline, immutable split, provenance manifest, table
compiler, interpolation and provider seam transfer to another CUDA GPU. The
numbers do not. TU116 has no Tensor Cores, and its `sm_75` SASS, scheduler,
cache behavior, throughput and latency do not represent Hopper. This device
cannot exercise Hopper FP8 tensor paths, TMA, warpgroup operations or
thread-block clusters. Nsight Systems activity rows also do not replace the
dynamic NVBit trace and architecture-specific counter ledger needed for
Accel-Sim correlation.

## Physical sanity before accuracy

The expectations commit froze these bounds before any target timing:

```text
floor = max(source FLOPs / dtype peak,
            compulsory input bytes / 288 GB/s)

ceiling = source operations / 1.5 Gop/s
          + total logical bytes / 48 GB/s
```

All 2,050 raw durations in all 50 cells lie between their cell's floor and
ceiling. The cell nearest its floor was FP64 `mlp_gemm` at shape 16: the floor
was 789.952 us, the median was 1,106.664 us and the ceiling was 91,575.637 us.
The median was 1.401 times the floor. The cell nearest its ceiling was FP32
`kv_read` at shape 1: the floor was 3.641 us, the median was 10.528 us and the
ceiling was 43.691 us. The ceiling was 4.150 times the median. These wide
ceilings detect impossible results; they are not evidence of calibration
accuracy.

The second independent angle was scaling. All 10 family and dtype sequences
increased strictly across shapes 1, 2, 4, 8 and 16. All 20 adjacent fourfold
train-shape ratios stayed in the frozen `[1, 8]` band, and FP64 was no faster
than FP32 in all 25 family and shape pairs.

The third angle was end-to-end plausibility. The table reached the supported
runtime chain in an unscored one-layer synthetic decode: context lengths 2
and 8 produced 89.420 us TTFT and 215.243 us TPOT respectively. That only
proves reachability. Its tiny model has 229,376 weight plus LM-head bytes, so
its 288 GB/s weight-read floor is about 0.797 us. The much larger measured
latency is above that floor, but the synthetic benchmark work is not the same
work as those model dimensions. It cannot validate a production decode.

For context, the motivating 400 million active-parameter BF16 decode has at
least 800 MB of active weight bytes. At the repository's 8 TB/s B100 roof its
weight-read floor is 100 us. A 99.4 us bootstrap value sits essentially on
that ideal floor, which is a reason to demand target-silicon calibration, not
evidence that this Turing table transfers to B100.

## Held-out accuracy

Shapes 1, 4 and 16 were fitted. Shapes 2 and 8 were excluded from the fit and
predicted by the existing one-axis log-linear interpolation rule for all five
families and both dtypes.

| predictor | minimum APE | median APE | p95 APE | maximum APE | frozen result |
|---|---:|---:|---:|---:|---|
| calibrated table | 0.119% | 0.674% | 1.773% | 1.909% | PASS, median <= 10% and p95 <= 20% |
| roofline bootstrap | 0.004% | 17.782% | 25.069% | 25.412% | MISS, p95 > 20% |

The calibrated median and p95 are both strictly below the corresponding
roofline values. Every held-out cell individually stayed below 20 percent.
The preceding post-fix replication measured calibrated median/p95 errors of
0.510%/1.768% and roofline median/p95 errors of 17.711%/25.069%, so the
decision-relevant separation repeated.

## Genuine-risk evidence

The genuine-risk fractions are kept by relation family:

| scored family | passed | total | why it could fail independently |
|---|---:|---:|---|
| held-out accuracy cells | 20 | 20 | activity timing can succeed while interpolation misses a family or roofline already fits |
| train-shape scaling | 20 | 20 | raw medians can scale too weakly or too strongly despite valid row counts |
| dtype slowdown | 25 | 25 | an implementation or identity error can make FP64 faster than FP32 |
| enabled family sum | 2 | 2 | all child lookups can succeed while opt-in reduction or uncertainty propagation is wrong |

Accuracy, scaling and dtype relations were evaluated from raw observations
before exact inventory, artifact and roundtrip checks. The physical bound
guard constrains only an interval and does not entail any of those scored
directions or errors. Family-sum observations were evaluated before the exact
table roundtrip. Therefore no earlier fatal oracle pins a scored result.

The following are fatal unscored guards. They are structural, compatibility
or by-construction evidence and do not increase a behavioral denominator:

| guard family | result |
|---|---|
| exact capture row count | PASS |
| exact train and held-out split | PASS |
| all raw durations within physical bounds | PASS |
| strict all-shape monotonicity | PASS |
| table roundtrip byte identity | PASS |
| disabled family-sum identity | PASS |
| every cell CV below 2 percent | **FAIL** |

The final CV failures were:

| family | dtype | shape | minimum | median | maximum | CV |
|---|---|---:|---:|---:|---:|---:|
| `attn_gemm` | FP32 | 8 | 99.585 us | 99.809 us | 115.392 us | 2.395% |
| `lm_head` | FP32 | 4 | 50.784 us | 50.912 us | 58.657 us | 2.343% |
| `attn_score` | FP64 | 1 | 26.657 us | 27.073 us | 31.168 us | 2.432% |

Each miss is driven by an isolated high-duration sample, but the frozen guard
applies to all samples. Removing outliers after observing them or substituting
the preceding 2-of-50 failure would manufacture a pass, so the study remains
failed.

## Compatibility and provider seam

`ProfileTableProvider` retains its exact default behavior. Family summation is
keyword-only and disabled by default. On the disabled path a fused miss still
raises `KeyError`, legacy lookup and interpolation remain unchanged, and the
same table serializes byte for byte. On the enabled path the all-train shape-4
query equaled its five children at 240.577 us, and the held-out shape-2 query
equaled its five children at 127.449222 us with conservative interpolated
uncertainty. An unsupported child fails the fused query.

The live reachability probe passed the calibrated provider through
`DeviceRuntimeStepSink`, `ExecutionGraph`, completion handling, `StepResult`,
TTFT and TPOT for two synthetic steps. It is deliberately unscored and is not
a production Granite accuracy claim.

## COMP-1 closure scope

The registered COMP-1 clauses map to evidence as follows:

| registered clause | evidence and disposition |
|---|---|
| "Pin a support envelope for every table" | PASS for this Turing benchmark envelope; no production framework envelope was captured. |
| "Capture the exact production run first" | NOT DEMONSTRATED. The anchor uses five framework-neutral synthetic CUDA kernels. |
| "NVBit supplies the SASS traces required by Accel-Sim" | NOT DEMONSTRATED. Static `sm_75` disassembly is present; no dynamic NVBit trace was fabricated. |
| "Build one replayable microbenchmark per captured kernel implementation" | PARTIAL. Five replayable family microkernels exist, but they were not derived from captured Granite kernels. |
| "Replay traces offline with a pinned Accel-Sim/GPGPU-Sim configuration" | NOT DEMONSTRATED. |
| "Populate simllm-gpu-model-artifact-v2" | NOT DEMONSTRATED. This method anchor uses the narrower measured calibration record and compact table. |
| "100 percent kernel identity coverage for the supported run" | PASS for all ten benchmark family and dtype functions; no production-run coverage claim. |
| "measured coefficient of variation below 2 percent" | FAIL in 3 of 50 final cells and 2 of 50 post-fix replication cells. |
| "held-out per-kernel median absolute percentage error below 10 percent and p95 below 20 percent" | PASS at 0.674 percent median and 1.773 percent p95. |
| "per-phase median below 5 percent and p95 below 10 percent" | NOT DEMONSTRATED. |
| "compute-only step error below 5 percent" | NOT DEMONSTRATED. |
| B100 efficiency-surface transfer | NOT DEMONSTRATED. Turing numbers are explicitly non-transferable. |

COMP-1 stays open because a fatal guard failed and the production clauses are
not satisfied. COMP-5 is rewritten in the owning module registry with the
specific remaining hardware requirements: counter permission, a stable
non-display or exclusive capture environment, and target-architecture
allocation. No task closes, so `docs/task-ledger.json` and the generated task
progress block intentionally do not change.

## Contradiction sweep

No integrator-owned overview file was edited. The sweep found one statement
that now needs nuance:

- `docs/README_PRO.md` says production SASS calibration and populated profile
  tables remain wholly blocked on capture hardware. A populated Turing method
  table now exists, while the production SASS and target-architecture table
  remain blocked.

`README.md` still correctly calls SASS offline calibration planned, and
`docs/architecture.md` still correctly says COMP-1 and COMP-5 remain open.

## Validation

The CUDA benchmark compiled for `sm_75`; its counter-smoke kernel completed.
The focused calibration suite passed 9 of 9 tests. `ruff check .` passed,
`python3 scripts/task_progress.py --check` reported no drift, and the full
suite passed 1,044 tests with 7 skips. The registered study command reached
its final assertion and exited nonzero solely because the frozen fatal
variation guard failed.
