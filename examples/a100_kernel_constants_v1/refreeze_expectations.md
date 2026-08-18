# A100 kernel constants v1 refreeze expectations

Date: 2026-08-18

This expectations-only supplement freezes the stage-1 rerun after the first
measurement run was reviewed VOID. It is written before the harness repairs it
describes exist and before any new result-producing job is submitted. The
original [expectations](expectations.md) stay byte-identical as the chronology
record and remain the source of every scored claim.

## Why the rerun exists

Slurm job `195964` executed the frozen protocol and completed. Its review state
is `VOID`: three fatal guards failed. Under the repository's fatal-void rule a
violated fatal guard makes the behavioral score uninterpretable, so run 1
contributes no fraction, closes nothing, and is retained as findings.

The three failures, and what each one turned out to be:

- **G11 failed on 131 of 246 cells.** The per-repetition event chain and the
  batched protocol disagreed by up to 61.8 percent. The cause is measured, not
  guessed: inserting one `cudaEventRecord` between two consecutive kernel
  launches costs about 2.1 to 2.6 microseconds of device time on this device.
  The batched protocol amortizes two events over `G` launches; the diagnostic
  chain pays one event per launch. For a 4.5 microsecond kernel that is a 58
  percent inflation, and for a 3.7 millisecond kernel it is 0.06 percent. The
  guard did exactly its job: the instrumentation is not free, and the freeze
  had assumed it was.
- **G9 failed on 49 cells.** G9 bounds the per-repetition coefficient of
  variation of the diagnostic chain. Because that chain is the contaminated
  series, its dispersion is not the kernel's dispersion for short cells.
- **G10 failed on 13 cells.** The batch-mean coefficient of variation exceeded
  2 percent, concentrated in cells of a few microseconds. Two mechanisms are
  visible in the run-1 data and neither was anticipated by the freeze: a batch
  of `G` = 1 begins immediately after a host synchronization, so it carries one
  post-synchronization launch latency inside the timed region, and a cell whose
  kernel is shorter than the host cost of issuing it measures the host issue
  rate rather than the kernel.

Run 1 also produced three results that are NOT guard failures and are carried
into the rerun unchanged as findings, because they say something about the
measurement rather than about the hardware:

1. The dense GEMM was issued as `C[M,N] = A[M,K] * B[K,N]` in column-major with
   leading dimensions `M`, `K` and `M`. The leading dimension of two of the
   three operands therefore depended on the token count, so an odd `M` broke
   vectorized access and cuBLAS selected a different kernel. Measured times
   swung by up to a factor 2.9 between neighbouring `M` values, for example
   264.9 microseconds at `M` = 87 against 110.2 microseconds at `M` = 64 in
   family G4. A serving engine does not issue that layout: its weight matrix is
   fixed and the token count is the free dimension, so its leading dimensions
   are `N` and `K` and do not move with `M`.
2. The decode attention kernel reached 3.6 to 13.1 percent of the measured HBM
   roof. One warp walked the cache with a serial online-softmax dependency, so
   the kernel was load-latency bound rather than bandwidth bound.
3. The measured HBM roof was 1809.45 GB/s, 88.7 percent of nameplate.

## Harness repairs, all post-specified and disclosed

These are repairs to the instrument, not to the expectations. Each one is named
here before it is written.

- **R1** The dense GEMM is issued in the engine-natural layout,
  `Out[N,M] = W[N,K] * X[K,M]` in column-major with leading dimensions `N`, `K`
  and `N`. No leading dimension depends on the token count. The work model,
  the shape families, the grid and the held-out set are unchanged.
- **R2** Every timed batch is preceded by one untimed priming launch, so the
  timed region begins with the stream already executing and measures
  back-to-back kernels rather than one post-synchronization launch.
- **R3** Every timed batch records the host wall time of its launch loop with a
  steady-clock pair taken around the loop and before any synchronization, so a
  host-issue-bound cell is identified instead of being published as a kernel
  constant.
- **R4** An instrumentation control measures the per-boundary cost of the event
  chain directly: the same 64 launches are timed with an event every 1, 4, 16
  and 64 repetitions, and an event-only chain with no kernels between the events
  is timed as well.
- **R5** The decode attention kernel carries four independent online-softmax
  accumulators per warp, so four cache rows are in flight per warp and the
  kernel is not held by one load latency at a time.

## What is retained without change

Every scored expectation E-1-1 through E-5-4 keeps its identifier, its band and
its exact claim string. The scored denominator stays 31. The substrate, the
arms, the definitions of warm state, rotated state, clock-stationary batch and
clock-conditioned constant, the shape families, the grids, the held-out set,
the expert-load grid and the attention grids are unchanged. Fatal guards G1
through G8, G10 and G12 are unchanged.

The scored expectations were frozen before run 1 and are re-evaluated after run
1 on repaired instrumentation. That is a rerun of a pre-registered claim set on
a corrected instrument, not a refreeze of the claims: no band moves and no
claim text changes. Run 1's evaluation of each one is published beside run 2's
so a reader can see both.

## What changes

### G9 becomes G9R

**G9R** Every scored cell whose instrumented chain median exceeds 60
microseconds has a per-repetition coefficient of variation, over the
clock-stationary part of its diagnostic chain, of at most
`0.02 + q / (sqrt(12) * median)`. A cell below 60 microseconds reports its
chain coefficient of variation as an unscored diagnostic, because the measured
per-boundary instrumentation cost is then more than 3 percent of the cell.

The 60 microsecond threshold is not chosen here for convenience. It is the
threshold the accepted
[A100 hardware envelope](../a100_hardware_envelope_v1/RESULTS.md) already
registered against COMP-5: any cell whose kernel is shorter than roughly 60
microseconds measures the launch path as much as the kernel.

### G11 becomes G11R

**G11R** The instrumentation control returns a per-boundary event cost in
[1.0, 4.0] microseconds, and for every scored cell the batch-mean constant
agrees with the instrumentation-corrected per-repetition mean, that is the
chain mean minus the measured per-boundary cost, within 3 percent.

The band [1.0, 4.0] microseconds is derived from run 1: the observed
inflation of the chain over the batched constant was 2.08 microseconds at
`gemm_G3_m1`, 2.54 microseconds at `gemm_G1_m1` and 2.62 microseconds at
`elem_scale_4mib_warm`. The band brackets that range with margin on both sides
and excludes zero, which is the assumption the original G11 made and which run
1 refuted.

### G13 is added, and is the one declared survivable guard

**G13** No scored cell is host-issue bound: in every scored batch the recorded
host launch-loop wall time is at most 0.8 times the device elapsed time of the
same batch.

G13 is declared SURVIVABLE in exactly one way, stated here before the run. A
cell that fails G13 is marked `host_issue_bound`, is excluded from every scored
expectation, from every other guard that quantifies over scored cells, and from
the published profile table, and is reported with both its host and its device
times. What remains interpretable is every expectation none of whose cells were
excluded, evaluated unchanged; an expectation that loses a cell is reported as
evaluated over a REDUCED SCOPE with the excluded cells named, and it may pass
only on the reduced scope, which is stated in its result row. An expectation
whose entire scope is excluded is reported as unevaluated, not as a pass. No
other guard in this study is survivable.

The reason G13 is survivable and the others are not: a host-bound cell is a
measurement of the host, and excluding it removes a wrong number rather than
hiding a failure. Its exclusion is visible in the result, and the constants of
the cells that remain are unaffected by it.

### G14 is added

**G14** The instrumentation control executed and reported a per-boundary cost,
a per-stride table over strides 1, 4, 16 and 64, and an event-only chain
period. A run without that control cannot evaluate G11R and is void.

## Unscored records added by the rerun

- the per-batch host launch-loop wall time of every cell;
- the instrumentation control's per-stride table and event-only chain period;
- the run-1 evaluation of every scored expectation, published beside run 2's;
- the list of cells excluded by G13 with their host and device times.

## What the rerun still will not claim

Unchanged from the original freeze: it closes neither COMP-1 nor COMP-5, it
measures no launch cost, it captures no framework kernel, and its constants are
standalone-kernel constants whose launch-mode independence is stage 2's
falsifier and is not assumed here.
