# TRAF-81 collective-floor extrapolation results

## Outcome

What ran: the frozen TRAF-81 CUDA and NCCL harness measured aggregate
all-gather and reduce-scatter at ranks 2, 4 and 8 on Merlin A100 GPUs over all
24 operation-buffer sizes, with 10 warmups and the observed median of 31
maximum-over-rank samples per point. Rank 16 did not run.

What came out: the formal verdict is **BLOCKED**, because Merlin's configured
quality-of-service limits admit at most eight GPUs to one job and the frozen
rank-16 cell needs 16. The available wide-rank evidence is nevertheless
decisive at the first locality crossing. Rank-4 donor extrapolation misses the
rank-8 all-gather curve by 62.889 percent median and 155.593 percent p95, and
misses reduce-scatter by 61.111 percent median and 144.544 percent p95. Both
curves fail the frozen 25 and 50 percent bands. The floor-fraction comparison
also fails both operations. Small-rank normalized shape therefore does not
transfer accurately from one A100 NV4 node to two nodes in this experiment.

What it changes: TRAF-81 stays open only for the blocked rank-16 cell and the
resulting width-growth family. The rank-8 result already refutes the two
families that ask whether latency shape and the floor-versus-slope
decomposition transfer across the first locality boundary. TRAF-76's rank-8
donor transfers at expert-parallel widths 32 and 128 remain deliberate,
acknowledged extrapolations, but they no longer carry unqualified shape
confidence from the fit-small-extrapolate-wide premise. One scope caveat
belongs beside that consequence: the A100 boundary crossed here changed the
transport stack as well as the locality (NVLink NV4 inside the node, NCCL
Socket over Cray Slingshot with GPU Direct RDMA disabled between nodes), so
the measured 61 to 63 percent medians quantify extrapolation failure across
this deployment's combined boundary, not a pure locality effect on an
RDMA-capable fabric.

What it does not change: this A100 study does not calibrate an H200 absolute
latency, bandwidth, floor or slope. It does not install a profile, change the
aggregate authority, rerun MiniMax, or change any time-to-first-token (TTFT)
or time-per-output-token (TPOT) result. The exact H200 rank-8 measurements
remain measured evidence. Only their transfer to wider ranks is implicated.

## Chronology and cell disposition

The expectations-only freeze is commit
`3f0aa24ea16573e3fc2ca030d541009cf308d12f`, based on `aee8cb5`. It precedes
the harness commit `d92053314a2da4d6a638f20a4cb3d49d1a5f2a01`, the guard-only
repair `dc211d81d86a422d6f79aa41cbb52c30038797e3`, and every measurement used
here. The freeze fixes the boundary-selection rule (Bayesian information
criterion over training ranks) and the donor-scale formula; the selected
boundary locations and the numeric scales 3.5 and 3.75 are deterministic
deductions from frozen inputs that first materialize in the record, not
literals present before measurement. The frozen JSON has SHA-256
`e38a8f4a4f4a812663e251c76d56c107a6b8872c7babb70d7bf74066b2b129ea`.

| Rank | Job | Nodes x GPUs | State | Start | End | Disposition |
|---:|---:|---:|---|---|---|---|
| 2 | 202445 | 1 x 2 | completed | 12:02:29 | 12:02:39 | measured |
| 4 | 202444 | 1 x 4 | completed | 12:02:13 | 12:02:28 | measured |
| 8 | 202443 | 2 x 4 | completed | 13:19:06 | 13:19:33 | measured |
| 16 | 202442 | 4 x 4 | cancelled from pending | not started | 14:00:16 | blocked |

Times are local on 2026-09-01. Rank 16 remained pending with
`QOSMaxGRESPerJob`. The `gpu_hourly` and `gpu_daily` quality-of-service rules
cap one job at eight GPUs. `gpu_general` caps one job at four GPUs and exposes
only three nodes. No permitted partition can start the frozen 16-GPU
allocation. Job 202442 remained pending for the complete 120-minute window,
was cancelled at 14:00:16 with zero elapsed run time and no assigned node, and
produced no target timing. No rank-16 median or fit is substituted.

An earlier submission at 11:53:20 carried a mistyped, nonexistent full harness
commit identifier even though its staged byte hashes were correct. Jobs 202435
and 202436 were cancelled before execution. Job 202437 reached rank-4 timing
and is retained as VOID under FG-1; its timing values were never inspected or
scored. Job 202438 failed before timing because the original topology guard
incorrectly required four visible GPU rows for a two-GPU allocation. The
guard-only commit `dc211d8` changed the check to require the allocated direct
NV4 submesh, two rows at rank 2 and four rows otherwise. It changed no frozen
coordinate, fitting rule, score or measurement code. The valid campaign was
then submitted at 11:59:44.

The first evidence audit consumed a compact transport summary that omitted
uppercase `CUMEM` lines. A post-specified audit repair reads the retained NCCL
debug logs directly and records each log hash. This repair changes no raw
measurement, fit or scored rule. The scored record uses NCCL 2.31.2, which identifies
itself as built against CUDA 12.9; the harness itself was compiled with the
site CUDA 12.2 toolchain. The multi-node cell used the NCCL Socket network transport with
GPU Direct RDMA disabled (`GDR 0`); intra-node channels used direct CUDA memory
and peer paths, and their initialization record reported Socket with `GDR 1`.

## Fatal guards and physical sanity

FG-1 through FG-7 all held for the measured rank-2, rank-4 and rank-8 cells.
Each measured operation has the exact 24-point grid, 31 finite positive
samples per point, an observed median, zero value mismatches, the frozen
placement and a complete before-and-after GPU ownership record. No fatal guard
is converted into a scored point.

Before fitting, bus bandwidth was bounded by
`((rank - 1) / rank) * bytes / time`. Bytes over the frozen link ceiling is the
completion-time floor. A100 software has no useful finite completion-time
ceiling, so the upper bound remains unbounded. Every observed maximum is below
the applicable link envelope:

| Operation | Rank | Maximum bus bandwidth (GB/s) | Frozen ceiling (GB/s) | Fraction | At bytes |
|---|---:|---:|---:|---:|---:|
| all-gather | 2 | 43.7490 | 100 | 0.4375 | 67,108,864 |
| reduce-scatter | 2 | 52.3450 | 100 | 0.5235 | 67,108,864 |
| all-gather | 4 | 135.4050 | 300 | 0.4513 | 67,108,864 |
| reduce-scatter | 4 | 142.0578 | 300 | 0.4735 | 67,108,864 |
| all-gather | 8 | 5.2667 | 100 | 0.0527 | 67,108,864 |
| reduce-scatter | 8 | 5.2876 | 100 | 0.0529 | 67,108,864 |

The intra-node maxima are 43.7 to 52.3 percent of their direct NV4 envelopes.
The rank-8 maxima are only about 5.3 percent of the frozen four-port cut. That
large change is physically plausible because NCCL selected host Socket
transport without GPU Direct RDMA for the inter-node path. It is not evidence
that A100 and H200 absolute bandwidths transfer.

## Frozen training fit

The fit is `T(S) = floor + S * slope`. Rank 2 and rank 4 alone choose up to
three positive regimes by the frozen weighted-relative-error Bayesian
information criterion. Effective bandwidth is `1 / slope`; it is a fit
diagnostic and is not the collective bus bandwidth above.

| Operation | Rank | Byte interval | Floor (us) | Slope (ps/B) | Effective bandwidth (GB/s) |
|---|---:|---:|---:|---:|---:|
| all-gather | 2 | 512 to 65,535 | 60.287528 | 129.432538 | 7.726033 |
| all-gather | 2 | 65,536 to 786,431 | 64.096683 | 21.929760 | 45.600135 |
| all-gather | 2 | 786,432 to 67,108,864 | 82.406152 | 10.191248 | 98.123411 |
| all-gather | 4 | 512 to 196,607 | 73.210184 | 35.831135 | 27.908689 |
| all-gather | 4 | 196,608 to 2,097,151 | 68.902365 | 8.810002 | 113.507358 |
| all-gather | 4 | 2,097,152 to 67,108,864 | 96.986950 | 4.127161 | 242.297290 |
| reduce-scatter | 2 | 512 to 65,535 | 61.311543 | 129.431371 | 7.726102 |
| reduce-scatter | 2 | 65,536 to 786,431 | 62.185175 | 22.885516 | 43.695759 |
| reduce-scatter | 2 | 786,432 to 67,108,864 | 78.847035 | 8.296648 | 120.530603 |
| reduce-scatter | 4 | 512 to 2,097,151 | 65.803981 | 9.264895 | 107.934301 |
| reduce-scatter | 4 | 2,097,152 to 67,108,864 | 91.299171 | 3.943182 | 253.602299 |

For rank 8, the frozen source-form scale is exactly 3.5: the rank-4 donor's
floor and slope are both multiplied by `q(8) / q(4)` and by the ratio of the
300 GB/s rank-4 ceiling to the 100 GB/s rank-8 ceiling. That produces these
predicted regimes:

| Operation | Byte interval | Predicted floor (us) | Predicted slope (ps/B) |
|---|---:|---:|---:|
| all-gather | 512 to 196,607 | 256.235643 | 125.408972 |
| all-gather | 196,608 to 2,097,151 | 241.158279 | 30.835005 |
| all-gather | 2,097,152 to 67,108,864 | 339.454327 | 14.445065 |
| reduce-scatter | 512 to 2,097,151 | 230.313932 | 32.427134 |
| reduce-scatter | 2,097,152 to 67,108,864 | 319.547099 | 13.801137 |

The untouched rank-8 measurements were then fit descriptively using those
fixed rank-4 boundaries:

| Operation | Rank | Byte interval | Measured floor (us) | Measured slope (ps/B) | Effective bandwidth (GB/s) |
|---|---:|---:|---:|---:|---:|
| all-gather | 8 | 512 to 196,607 | 96.434880 | 1172.381748 | 0.852964 |
| all-gather | 8 | 196,608 to 2,097,151 | 445.443118 | 218.704076 | 4.572388 |
| all-gather | 8 | 2,097,152 to 67,108,864 | 388.629004 | 160.937528 | 6.213591 |
| reduce-scatter | 8 | 512 to 2,097,151 | 103.559747 | 531.516461 | 1.881409 |
| reduce-scatter | 8 | 2,097,152 to 67,108,864 | 369.837647 | 160.955079 | 6.212914 |

Every rank-8 descriptive regime has a positive floor and slope. The donor has
the wrong floor and slope in opposing directions over parts of both curves,
so total-latency agreement at an isolated byte point would be cancellation,
not a correct decomposition.

The frozen rank-16 donor scale is 3.75. With no measured rank-16 denominator,
neither operation has an extrapolation-error value and the prediction remains
unscored.

## Scored families

| Family | Operation | Rank | Median | p95 | Frozen band | Result |
|---|---|---:|---:|---:|---|---|
| S1 absolute latency error | all-gather | 8 | 62.889% | 155.593% | at most 25%, 50% | fail |
| S1 absolute latency error | reduce-scatter | 8 | 61.111% | 144.544% | at most 25%, 50% | fail |
| S2 floor-fraction difference | all-gather | 8 | 0.2302 | 0.5134 | at most 0.20, 0.35 | fail |
| S2 floor-fraction difference | reduce-scatter | 8 | 0.4840 | 0.7148 | at most 0.20, 0.35 | fail |

The S1 signed medians are -57.355 percent for all-gather and -55.406 percent
for reduce-scatter. A negative sign means the donor prediction is optimistic.
The maxima are 181.228 and 181.165 percent. Every byte-level prediction,
measurement and error is retained in [extrapolation.csv](extrapolation.csv).

S3 is unevaluated because it requires rank 16 to compare p95 growth and error
sign against rank 8. Per the frozen overall rule, that makes the formal verdict
BLOCKED even though S1 and S2 already fail at rank 8.

S4 also fails its independent plausibility statement. The rank-2 all-gather
dip is 8.57 percent at 786,432 bytes, and the rank-2 reduce-scatter dip is
14.29 percent at 786,432 bytes. Neither reaches the frozen 15 percent depth.
The dense region still exposes protocol shape. Rank-4 all-gather has local
dips of 11.11 percent at 1,048,576 bytes and 33.33 percent at 2,097,152 bytes.
Rank-4 reduce-scatter has local dips of 20.00 percent at 524,288 bytes, 3.75
percent at 917,504 bytes, 10.71 percent at 1,310,720 bytes and 29.52 percent at
2,097,152 bytes. Rank 8 has no local dip under the frozen definition inside
512 KiB through 1.5 MiB: its strongest all-gather and reduce-scatter dips are
3.55 and 0.32 percent at 262,144 bytes. It instead has completion-time speedup
steps, reported under the freeze's pre-specified descriptive rule and
unscored. Rank-8 all-gather completion falls by 21.84 percent from 1,179,648 to
1,310,720 bytes; rank-8 reduce-scatter falls by 12.61 percent over the same
step and by 10.48 percent from 917,504 to 1,048,576 bytes. Thus the expected
rank-2 dip does not reproduce at the frozen threshold, while the dense grid
does reveal non-monotone cross-node transitions near 1 MiB.

## Reproduction and evidence boundary

The tracked compact authority is [record.json](record.json), with 144 measured
medians in [measurements.csv](measurements.csv), 48 rank-8 holdout rows in
[extrapolation.csv](extrapolation.csv), and 16 training or descriptive regimes
in [fits.csv](fits.csv). The measurement-set SHA-256 is
`f40dbf7147f0a99457bd1506f5b42dc97ac7350ab876c28e41d5cdbacef41a70`.
The tracked medians reproduce every boundary, fit and score without access to
the external raw samples:

```bash
env -u PYTHONPATH .venv/bin/python \
  examples/collective_floor_extrapolation_v1/score_expectations.py --check
```

The append-only external evidence retains the 31-sample vectors, compiled
binaries, source and library hashes, topology and ownership guards, complete
NCCL logs, Slurm output, invalid attempts and scheduler diagnostics. Its root
is supplied through the local run-root configuration; no machine-specific
path is embedded in tracked code. The external `final-evidence.sha256`
manifest locks 165 files, including the terminal rank-16 scheduler record.

## Validation

Before the result commit, the standalone reconstruction passed with
`PYTHONPATH` removed, Ruff reported no violations, and full Pytest completed
with 3,870 passed and 33 skipped. `scripts/check_docs_format.py` accepted all
11 module documents. `scripts/task_progress.py --check` confirmed that the
generated progress block and module open counts are current. The repository's
diff whitespace check passed. Git reports `eol=lf` for every new tracked
result table and record.

No residual task ID was consumed. TRAF-85 and TRAF-86 remain available.
