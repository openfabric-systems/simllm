# TRAF-81 collective-floor extrapolation freeze

## Freeze scope and chronology

This is the expectations-only authority for TRAF-81. It is committed before
the measurement harness exists, before any source is staged for this study,
before any result-producing Slurm job is submitted, and before any collective
in this study is timed. No observed value may be written back into this file.

The base commit is `aee8cb5`. The study changes no SimLLM timing authority,
profile, execution path, time to first token (TTFT), or time per output token
(TPOT). It measures whether one active extrapolation assumption is defensible.

## Question and evidence boundary

The fitted H200 aggregate authority has separate all-gather and reduce-scatter
curves at ranks 2, 4 and 8. Its consumers deliberately transfer the rank-8
curve to wider expert-parallel groups. The external NCCL resolver underneath
the supported MiniMax pass also holds the largest measured rank fixed and
rescales it by the collective rank factor and a topology bandwidth tier above
that rank. Those transfers matter at expert-parallel widths 32 and 128, but no
independent wide-rank measurement currently tests the assumption.

This study recreates the same epistemic shape on A100 hardware:

1. ranks 2 and 4 are the only training ranks and remain inside one four-GPU
   `NV4` node;
2. ranks 8 and 16 are untouched scored ranks on two and four nodes;
3. the rank-4 curve is the donor, and the wide prediction rescales its fitted
   floor and byte slope by the same rank-factor and topology-ceiling form as
   the source resolver;
4. rank-8 and rank-16 measurements are never inputs to the predictive fit.

The A100 experiment does not calibrate an H200 absolute number. Prior A100 and
GH200 work found that efficiency relative to each architecture's own links is
similar while absolute bandwidth is not. Therefore every scored quantity here
is dimensionless or a within-A100 shape comparison. No H200 latency,
bandwidth, floor, slope, or uncertainty band enters a score.

## Frozen hardware and software substrate

| Item | Frozen value |
|---|---|
| cluster and account | `gmerlin7`, account `merlin` |
| partitions | `a100-hourly` first; `a100-daily` or `a100-general` only as a recorded retry when the hourly limit or availability requires it |
| GPU | A100-SXM4-80GB, four GPUs per node |
| intra-node topology | every ordered pair reports `NV4`, four 25 GB/s links per pair |
| cross-node topology | four 200 Gbit/s Cassini ports per node, with the exact NCCL network transport and GPUDirect status recorded rather than assumed |
| CUDA | site module `cuda/12.2.2`, compiler target `sm_80` |
| NCCL | x86_64 `nvidia-nccl-cu12` wheel, expected version 2.31.2; the exact library hash and runtime version are recorded |
| dtype | half precision, two bytes per element |
| launch | one process per GPU, one CUDA stream per rank, no graph capture |

CUDA 13 is forbidden. The installed driver exposes driver API 12.7, so a
CUDA-13-built executable is a capability mismatch rather than a measurement.

## Frozen harness and aggregation

The harness is a minimal self-built CUDA and NCCL executable. It is selected
instead of `nccl-tests` so the result can retain every per-repetition maximum,
the exact operation-buffer coordinate and the bootstrap diagnostics in one
bounded record without patching an external source tree. The implementation
lands only after this freeze.

For each operation, rank and byte point:

1. initialize deterministic half-precision buffers;
2. execute ten unmeasured warmups;
3. execute 31 measured repetitions;
4. before each measured repetition, synchronize all ranks with an untimed
   one-element NCCL barrier on the same communicator;
5. bracket exactly one target collective with CUDA events on the target
   stream, synchronize the stop event, then reduce the elapsed value with
   `ncclMax` outside the timed region;
6. report the median of the 31 maximum-over-rank repetitions.

The repetition count is odd, so the median is one observed sample rather than
an average of two. The raw 31-sample vectors and logs remain in the append-only
run root. The compact tracked measurement table carries the median, minimum,
maximum, nearest-rank p05 and nearest-rank p95, source and binary hashes, job
identity and exact environment provenance.

The harness runs out of place. For an operation-buffer coordinate `S` bytes and
rank count `n`:

- all-gather sends `S / n` bytes per rank and receives `S` bytes per rank;
- reduce-scatter sends `S` bytes per rank and receives `S / n` bytes per rank.

This is the `nccl-tests` total-buffer convention and matches the authority's
one byte coordinate. Every frozen size is divisible by `16 * 2`, so no cell
rounds an element count at the widest rank.

## Frozen byte and rank grid

The 24 operation-buffer sizes, in bytes, are:

```text
512          2048         8192         32768        65536        131072
196608       262144       393216       524288       655360       786432
917504       1048576      1179648      1310720      1572864      2097152
3145728      4194304      8388608      16777216     33554432     67108864
```

The grid places nine points from 512 KiB through 2 MiB, including 896 KiB,
1 MiB, 1.125 MiB and 1.25 MiB. That density is frozen because the prior A100
width-2 curve had a serialization-efficiency minimum about 26 percent below
its local peak at exactly 1 MiB. The new grid observes the transition rather
than interpolating across it.

| Rank | Nodes x GPUs per node | Locality | Role |
|---:|---:|---|---|
| 2 | 1 x 2 | intra-node NV4 | training |
| 4 | 1 x 4 | intra-node NV4 | training and donor |
| 8 | 2 x 4 | mixed NV4 and fabric | scored holdout |
| 16 | 4 x 4 | mixed NV4 and fabric | scored holdout |

Each rank runs both operations over all 24 sizes. There are 192 measured
medians and 5,952 retained timed repetitions when every cell is available.

## Frozen physical ceilings and napkin bounds

The `nccl-tests` bus-byte factor for both operations is

```text
q(n) = (n - 1) / n
```

For median completion `T(S,n)`, bus bandwidth is `q(n) * S / T`. The frozen
ceiling is the aggregate egress the realized topology can put on the limiting
cut:

| Rank | Ceiling | Physical basis |
|---:|---:|---|
| 2 | 100 GB/s | one NV4 peer bundle, four 25 GB/s links |
| 4 | 300 GB/s | three NV4 peer bundles per GPU |
| 8 | 100 GB/s | four 25 GB/s Cassini ports per two-node cut |
| 16 | 100 GB/s | four 25 GB/s Cassini ports per node cut |

These are A100 and Merlin bounds, not H200 transfers.

- Floor: no row may beat `q(n) * S / ceiling(n)`. A faster value is a harness,
  coordinate or topology defect.
- Ceiling: the software stack has no first-principles progress guarantee, so
  the honest completion-time ceiling is unbounded. No arbitrary upper latency
  is scored.
- Covariate: doubling `S` inside one stable slope regime doubles only the byte
  term, not the floor. If the fitted floor doubles with `S`, the decomposition
  is not identifying the authority's stated mechanism.
- System plausibility: the intra-node asymptote must remain below the measured
  NV4 egress envelope and the cross-node asymptote below the realized port
  envelope. An internally exact curve outside either envelope is still void.

## Frozen fit and extrapolation rule

Every curve uses the aggregate authority's positive piecewise form:

```text
T_ps(S) = floor_ps + S * slope_ps_per_byte
```

The fit uses exact-rational weighted relative least squares, with weight
`1 / measured_ps^2`, exactly as `fit_collective_floor_calibration` does.
Training-only Bayesian information criterion selection chooses one, two or
three regimes, with at least two cells per regime and a boundary only at a
frozen byte-grid point. Rank 2 and rank 4 choose boundaries independently for
each operation. No rank-8 or rank-16 value participates in boundary selection,
parameter fitting or model choice.

The rank-4 regimes are the donor. For scored rank `r`, every donor floor and
slope is multiplied by

```text
scale(r) = q(r) / q(4) * ceiling(4) / ceiling(r)
```

This is the source resolver's fit-largest-rank then rescale-by-rank-and-link
form, instantiated only with A100 and Merlin ceilings. It predicts that
normalized efficiency

```text
eta(S,n) = q(n) * S / (T(S,n) * ceiling(n))
```

is invariant with width. This dimensionless prediction is the core question.

After a scored verdict is fixed, ranks 8 and 16 are also fit descriptively with
the rank-4 operation's frozen regime boundaries. Those fits report the
measured floor-versus-slope decomposition. They never alter the donor,
prediction, score band or verdict. A wide regime that cannot produce positive
floor and slope is `UNREPRESENTABLE` and refutes the decomposition family; it
does not void sound measurements.

## Fatal guards

Fatal guards are reported as held or violated, never as a fraction. Any
violation voids the affected run and prevents every behavioral score that uses
it.

- **FG-1 chronology and identity.** This freeze commit precedes the harness
  commit and every result-producing job. Frozen files, harness source, binary,
  CUDA and NCCL library all have retained SHA-256 identities.
- **FG-2 target and placement.** Every rank is an A100-SXM4-80GB, each job has
  exactly the frozen rank and node shape, local rank maps to one distinct GPU,
  and the same GPU UUID is retained before and after.
- **FG-3 topology and transport.** Ranks 2 and 4 observe the full direct NV4
  mesh. Ranks 8 and 16 observe four Cassini ports per node. The exact NCCL
  network transport and GPUDirect status are complete, and ranks 8 and 16 use
  the same transport class. A changed but internally consistent transport is
  retained as a different configuration and is void for this comparison.
- **FG-4 coordinate and sample completeness.** Every available cell contains
  exactly the frozen 24 sizes, ten warmups, 31 finite positive timed samples,
  the maximum-over-rank reduction and one median per operation.
- **FG-5 value conservation.** All-gather returns each source rank's exact
  deterministic pattern at three probes per source. Reduce-scatter returns the
  exact half-precision sum at three probes. Any mismatch is fatal.
- **FG-6 physical ceiling.** No bus bandwidth exceeds its rank's frozen
  ceiling. The intra-node check is performed before any fitted digit is
  trusted.
- **FG-7 isolated ownership.** No foreign process occupies an allocated GPU,
  no result path escapes the configured run root, and no raw binary or bulk log
  is tracked in Git.

No guard has a survivable branch.

## Scored shape families

The held-out error is `(predicted_ps - measured_ps) / measured_ps`. A positive
sign means the extrapolation is pessimistic; a negative sign means it is
optimistic. All percentiles use deterministic nearest-rank selection.

### S1, normalized-efficiency reproduction

For each of the four `(operation, scored rank)` curves, median absolute
relative error is at most 25 percent and p95 absolute relative error is at
most 50 percent over all 24 byte points. Since the prediction and measurement
use the same A100 ceiling for that rank, this is exactly the error of the
dimensionless efficiency curve, not an H200 latency comparison.

### S2, floor-versus-slope decomposition

For every representable scored regime, compare the observed wide descriptive
fit with the scaled donor at every grid point in that regime. The absolute
difference in floor fraction,

```text
floor / (floor + S * slope)
```

has median at most 0.20 and p95 at most 0.35 for each operation and scored
rank. An unrepresentable regime fails this family. This tests whether an
apparently acceptable total error hides cancellation between a wrong floor and
a wrong slope.

### S3, error sign and growth with width

For each operation, rank 16 must not worsen p95 absolute error by more than ten
percentage points relative to rank 8. Median signed error must not flip across
zero unless at least one median lies in the neutral interval `[-0.05, 0.05]`.
The family therefore passes a small stable error and fails a locality error
that grows or reverses as width increases.

### S4, the 1 MiB transition

For each measured curve, define the small-message floor as the median of the
512 B, 2 KiB and 8 KiB medians. Remove that floor and compute normalized
serialization efficiency at larger sizes. A local dip at grid point `S` means
its efficiency is below both adjacent grid points. Its depth is one minus the
point divided by the smaller adjacent efficiency.

The prior-shape plausibility statement is that at rank 2 at least one of the
two operations has a dip of at least 15 percent between 512 KiB and 1.5 MiB.
This is scored. Dip location, depth and sign are reported for all eight curves,
including a speedup step, where completion falls as bytes rise, on a
cross-node protocol transition.

### Overall rule verdict

`FIT-SMALL-EXTRAPOLATE-WIDE HOLDS` requires S1, S2 and S3 to pass in full.
S4 has its own shape verdict and does not rescue or void the extrapolation
verdict. A partial pass is not closure by averaging.

## Blocked and void publication

Each rank cell has one state:

- `MEASURED`: all fatal guards hold and its rows are available;
- `BLOCKED`: the scheduler cannot start the allocation inside the campaign
  window, or the frozen CUDA/NCCL environment cannot initialize before a
  target collective is timed;
- `VOID`: a fatal guard fails after a measurement starts.

The campaign window is 120 minutes from the first submission of a rank cell.
A still-pending study-owned job is cancelled at that boundary and its exact
Slurm state and reason are retained. A multi-node NCCL initialization failure
is `BLOCKED` only when no target collective was timed and structural target
guards remain decidable; otherwise the run is `VOID`.

A blocked cell has no synthetic median, no fitted row and no score. Ranks 2
and 4 still publish their fit when either wide cell blocks. Rank 8 scores by
itself when rank 16 blocks, but S3 remains unevaluated and the overall rule
verdict is `BLOCKED`. TRAF-81 then remains open, narrowed to the named missing
cell. A void cell leaves all dependent claims void.

## Evidence classes and publication

The result keeps these classes separate:

1. four run configurations and their cell states;
2. fatal guards, reported only as held or violated;
3. training fits at ranks 2 and 4;
4. descriptive wide fits, never used as prediction inputs;
5. four scored shape families;
6. unscored physical bounds, environment provenance and protocol diagnostics.

The publication reports measured floors and slopes per available rank and
operation, the complete rank-2 and rank-4 fits, every byte-level rank-8 and
rank-16 error, median and p95 error per operation and rank, floor-fraction
error, error sign and growth, dip locations, and every blocked or void cell.
It states what the finding changes for rank-8 donor transfers and what remains
unchanged. No result is allowed to silently install a new production profile.
