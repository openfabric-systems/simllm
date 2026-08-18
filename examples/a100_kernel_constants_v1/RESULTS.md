# A100 kernel constants v1 results

The reviewed study state is `VOID`. Both measurement runs violated fatal
guards, so neither behavioral score is interpretable, no fraction is reported
as a result, and no calibrated artifact is published. The evidence is retained
and the findings below are the product of this study.

This is a standalone-kernel microbenchmark on one A100 SXM4 80 GB. It runs no
framework, loads no model, replays no SASS and reports no TTFT or TPOT. It
closes nothing, and it never claimed it would.

## Why the study is void, stated plainly

Run 2 (Slurm job `195982`) violated three fatal guards:

| Guard | Outcome |
|---|---|
| `G9R` | 16 of 97 scored cells above the 60 microsecond threshold exceeded the quantum-aware per-repetition CV ceiling. |
| `G10` | 2 scored cells exceeded the 2 percent batch-mean CV ceiling: `hbm_write_512mib` at 2.305 percent and `attn_decode_b256_l2048` at 3.315 percent, both in the boosted arm. |
| `G11R` | The measured per-boundary event cost held its band, but 21 scored cells still disagreed with their batched constant by more than 3 percent after the correction. |

`G13`, the one guard the refreeze declared survivable, also fired once and was
survived exactly as the refreeze said it would be. `moe_expert_down_m54` in the
BASE arm was host-issue bound, so it was excluded from every scored
expectation, from the other guards and from any published table. The refreeze
promised it would be reported with both its host and its device times, and it
is: over its 12 batches of 16 launches each, the host launch loop took a mean
of 86.375 microseconds against a mean device elapsed time of 130.304
microseconds, a mean ratio of 0.653 and a worst-batch ratio of 0.870 against
the 0.8 ceiling. The per-batch arrays of both are in
`measurements/results.json` under `host_issue_bound_cells`.

Sixteen of the 31 scored expectations passed and 15 failed. **That 16 is not a
score.** A violated fatal guard asserts that the precondition under which the
scored numbers mean what they claim did not hold, so the fraction is
uninterpretable and is written here only so a reader can see which relations
survived and which did not.

## Chronology and integrity

| Event | Commit or job |
|---|---|
| unscored discovery pass | job `195960` |
| expectations-only freeze | `3df16b4` |
| measurement harness | `4e7c939` |
| run 1, reviewed `VOID` | job `195964` |
| rerun refreeze, expectations only | `d7cbc94` |
| repaired harness and scorer | `85ac01f` |
| run 2, reviewed `VOID` | job `195982` |

The freeze preceded the harness, the harness preceded run 1, and the refreeze
preceded the repaired harness, which preceded run 2. One deviation is recorded
rather than smoothed over: run 1 was submitted a few minutes before the harness
commit landed. The submitted source is byte-identical to the committed file,
verified by SHA-256 `b5139a4cfba76a645a45ac8a143cdb5245b062c9b255ce9ba7dd2a7d236ec11f`
recorded by the job itself. Run 2's submitted source digest is
`9d3edae604fa886534b92267567bb475657034f279ef6a753ea42e5554180134`, also
matching its commit.

| Artifact | SHA-256 |
|---|---|
| `expectations.md` | `4e803cb548be6df6ca5fe459aa9718b4ff735c6566abf6d320950a246b0d6a6b` |
| `refreeze_expectations.md` | `d4e2b0ffd6a941249d46d1c245446cb0f2641967c6f7199571e758c9e6206e7e` |
| run 2 boosted raw | `bdfc2efaac77018fbdc641249f4195475b6013a0e08deaf3ae88c5fff05232ad` |
| run 2 base raw | `822c8ffeb07674625dd9d87d795b351be65ae1b6f82afaa7dc449cf2fae0e12e` |
| `measurements/results.json` | `c2233dbd86a2acec858c0b9b2f599e0d64212e33d8b354d51ad41a0159cb0577` |

The results file was regenerated once after first publication, to add the three
records the refreeze had promised and this report had not delivered: the run-1
evaluation beside run 2's, the per-cell mean host launch-loop and device batch
times rather than only their ratio, and the excluded cell's two series. Its
earlier digest was
`bef749281f2767ef81e9906561c3334e70dad5918cb016893b142c66e6b4fd15`. No measured
value, bound, verdict or guard outcome changed in that regeneration; the two
raw files it is derived from are unchanged and their digests above still hold.

Neither expectations document was edited after any run. Two post-run repairs
were made to the scoring script only and are disclosed here: it learned to read
guard claims from the refreeze as well as the freeze, and it was corrected to
apply `G9R`, `G10` and `G11R` to cells that participate in at least one scored
expectation, which is what those guards say, rather than to every measured
cell. The correction removed three rotated-variant cells at 4 and 16 MiB from
`G10`, which no scored expectation names. It changed no measured value and no
bound, and the verdict was void before and after it.

## Run 1 beside run 2, as the refreeze promised

The refreeze listed "the run-1 evaluation of every scored expectation,
published beside run 2's" as an unscored record. It is in
`measurements/results.json` under `previous_run`, produced by re-scoring run
1's retained raw output with the identical scorer and its own identity record,
so the two columns are the same instrument applied twice. Run 1 is void under
that instrument too, on `G9R`, `G10`, `G11R` and `G14`, the last because run 1
predates the instrumentation control the refreeze added.

Run 1 scored 15 pass and 16 fail; run 2 scored 16 and 15. Five rows moved, and
the direction of each is the interesting part:

| Expectation | Run 1 | Run 2 | What moved it |
|---|---|---|---|
| E-1-7 | pass | fail | the 1024 MiB write ratio drifted to 1.0206, just outside [0.98, 1.02] |
| E-2-9 | fail | pass | held-out interpolation median error fell from 32.66 to 0.70 percent under repair R1 |
| E-3-2 | fail | pass | synthetic prefill rose from 39.85 to 40.44 percent of peak, across a 40 percent threshold |
| E-3-5 | pass | fail | the repaired decode kernel scales differently in batch, 3.024 against a [3.4, 4.6] band |
| E-4-2 | fail | pass | the expert-load plateau tightened from 1.81 and 2.09 to 1.55 and 1.50 under repair R1 |

Twenty-six rows did not move at all. Two of the five that did, E-1-7 and
E-3-5, moved from pass to fail, so the repairs did not simply buy passes.

## The permission denial, recorded

Application clock control is denied on this allocation. Both runs attempted it
and both recorded the same refusal verbatim:

```text
The current user does not have permission to change clocks for GPU 00000000:C1:00.0.
Terminating early due to previous errors.
rc=4
```

`nvidia-smi -ac 1593,1410` is refused identically. This is why the study
publishes clock-conditioned constants over clock-stationary batches instead of
locked-clock constants, and it is why COMP-5's controlled-environment stability
form remains out of reach here.

## Clock behavior, measured

The SM clock on this device is not continuous. Under load it takes exactly two
values, 1275 MHz (the default applications clock) and 1410 MHz (the maximum
boost), with 210 MHz at idle. The transition needs 283 to 432 milliseconds of
sustained load. The memory clock is 1593 MHz in every sample of both the
in-process NVML series and an independent `nvidia-smi dmon` series, and the
NVML throttle-reason word was zero throughout, through 300 W and 44 C.

The consequence for the contract is direct and it is measured:

| Cell | 1275 MHz | 1410 MHz | ratio | regime |
|---|---:|---:|---:|---|
| `gemm_G4_m1024` | 578.0 us | 523.5 us | 1.1040 | compute |
| `gemm_G4_m256` | 170.2 us | 154.2 us | 1.1032 | compute |
| `gemm_G1_m512` | 21.1 us | 18.9 us | 1.1121 | compute |
| `gemm_G4_m1` | 90.3 us | 90.0 us | 1.0026 | memory |
| `hbm_read_1024mib` | 613.2 us | 611.4 us | 1.0029 | memory |
| `hbm_triad_1024mib` | 1821.0 us | 1834.7 us | 0.9926 | memory |

A compute-limited constant moves with the SM clock by the clock ratio 1.10588.
A memory-limited constant does not move at all, because HBM service is pinned
to a memory clock that never changes. A clock-blind average over a cell that
crosses the boost boundary is a mixture of two different constants and is not
the constant of anything.

## The measured HBM roof

`R_hbm` is 1818.21 GB/s, 89.17 percent of the 2039.04 GB/s memory-clock
nameplate. It is the maximum over every lane-1 cell at 256 MiB and above in the
boosted arm and is the denominator of every memory-bound statement here.

| Kernel | 256 MiB | 512 MiB | 1024 MiB | 2048 MiB |
|---|---:|---:|---:|---:|
| read | 1693.5 | 1732.7 | 1756.2 | 1766.4 |
| write | 1733.2 | 1776.2 | 1800.1 | 1818.2 |
| copy | 1633.7 | 1665.3 | 1673.8 | 1678.3 |
| triad | 1726.8 | 1747.5 | 1755.8 | 1759.1 |

All values GB/s. Between 1024 and 2048 MiB the rate moves by 0.19 percent for
triad, 0.27 for copy, 0.58 for read and 0.99 for write, taking the larger value
as the denominator. Against the smaller value the write figure is 1.004
percent, so "under one percent" holds only on the first convention and the
convention is named here rather than assumed. The prior accepted
[A100 hardware envelope](../a100_hardware_envelope_v1/RESULTS.md) measured
1770.5 GB/s read and 1672.4 GB/s copy at 4 GiB on this hardware class. This
study's largest size is 2048 MiB, so the comparison is across sizes rather than
at one: read measures 1766.4 GB/s here against 1770.5 GB/s there, 0.23 percent
apart, and copy measures 1678.3 against 1672.4, 0.36 percent apart. Two
separately written harnesses at two different sizes agree to better than 0.4
percent.

## One entailment the freeze should have caught

E-1-6 asserts that `R_hbm` lies in [1700, 1937] GB/s. `R_hbm` is defined as the
maximum achieved rate over the lane-1 cells at 256 MiB and above, and the
maximum turned out to be the same 2048 MiB write cell that E-1-2 scores against
the identical band. On this run the two expectations are therefore the same
measurement checked twice, and E-1-6 carried almost no independent risk. It is
not exactly entailed, because a cell below 2048 MiB could in principle have
exceeded 1937 GB/s and broken E-1-6 while E-1-2 held, but that is a thin
distinction and it should have been noticed when the freeze was written rather
than after the run. It is disclosed here rather than quietly left in the
denominator. The denominator is uninterpretable anyway, since the run is void.

## Finding 1: per-kernel event instrumentation is not free

This is the reason run 1 was void and it is the most transferable result of the
study. Inserting one `cudaEventRecord` between two consecutive kernel launches
costs device time. Run 2 measured it two independent ways and they agree:

| Method | Value |
|---|---:|
| event-only chain, 64 event records with no kernels between them | 2.352 us per event |
| stride sweep, per-kernel time at event stride 1 minus at stride 64, median over five control kernels | 2.336 us per boundary |

The stride sweep is the clearer picture, because it also recovers the
uninstrumented back-to-back period:

| Control | stride 1 | stride 4 | stride 16 | stride 64 |
|---|---:|---:|---:|---:|
| empty kernel | 4.352 | 2.464 | 2.016 | 1.904 |
| `gemm_G1_m64` | 11.408 | 9.664 | 9.216 | 9.072 |
| `gemm_G2_m1024` | 21.344 | 19.408 | 18.944 | 18.816 |
| `gemm_G4_m1` | 91.344 | 89.952 | 89.728 | 89.648 |
| `scale_64mib` | 89.648 | 87.840 | 87.360 | 87.424 |

All values microseconds per kernel. The uninstrumented empty-kernel period of
1.904 us reproduces the 1.806 us pipelined eager launch the A100 hardware
envelope measured with a different harness, which is a second independent
cross-check.

Two consequences follow and both are load-bearing for the rest of the campaign.
First, a per-kernel event chain overstates a 5 microsecond kernel by about 45
percent and a 90 microsecond kernel by about 2 percent, so any per-kernel
timing of short kernels must either amortize the events or subtract a measured
boundary cost. Second, stage 2's falsifier, that in-graph per-kernel execution
equals the standalone constant, cannot be tested with naive per-kernel event
instrumentation: the instrumentation alone would refute it.

Run 1's freeze asserted this cost was zero (original `G11`). It is not. That
refutation is the guard doing its job.

## Finding 2: the GEMM operand layout, not the shape, produced a factor 2.9

Run 1 issued the dense GEMM as `C[M,N] = A[M,K] * B[K,N]` with leading
dimensions `M`, `K` and `M`, so two of the three leading dimensions moved with
the token count and an odd `M` broke vectorized access. Run 2 issued the layout
a serving engine issues, `Out[N,M] = W[N,K] * X[K,M]`, whose leading dimensions
are fixed at `N` and `K`.

| Family G4 cell | run 1 | run 2 | change |
|---|---:|---:|---:|
| `M` = 64 | 110.2 us | 93.6 us | 0.85 |
| `M` = 87 | 264.9 us | 104.4 us | 0.39 |
| `M` = 111 | 270.3 us | 104.8 us | 0.39 |
| `M` = 413, held out | 755.6 us | 313.7 us | 0.42 |
| `M` = 8192 | 3727.6 us | 3707.4 us | 0.99 |

The effect is confined to shapes whose token count is not a convenient
multiple, and it vanishes at large `M`. The size quoted here is the run-1
shape-to-shape swing between two neighbouring grid points of family G4, `M` = 87
at 264.9 us against `M` = 64 at 110.2 us, a factor **2.40**, recomputable from
the retained run-1 file. An earlier draft of this report said 2.9; that figure
came from comparing non-neighbouring points and is withdrawn. Its practical
size is what matters:
under the wrong layout, log-linear interpolation of the held-out shapes carried
a median absolute percentage error of 32.66 percent and a p95 of 130.83
percent; under the engine-natural layout the same interpolation over the same
grid and the same held-out shapes carries a median of 0.70 percent and a p95 of
18.53 percent. Both of COMP-1's registered held-out bars, median below 10
percent and p95 below 20 percent, are met by run 2's surface. That is a
measurement about interpolation on a corrected instrument, not a closure: the
run that produced it is void.

## Finding 3: no single efficiency constant describes a family

The surrogate this study exists to bound is `RooflineProvider(efficiency=0.7)`.
Measured roofline efficiency in the boosted arm:

| Family | `N`, `K` | eff at `M` = 1 | eff at largest grid `M` | span |
|---|---|---:|---:|---:|
| G1 granite QKV | 2048, 1024 | 0.315 | 0.763 | 2.4x |
| G2 granite output and expert gate/up | 1024, 1024 | 0.205 | 0.729 | 3.6x |
| G3 granite expert down | 1024, 512 | 0.125 | 0.630 | 5.0x |
| G4 synthetic 8192 | 8192, 8192 | 0.820 | 0.951 | 1.2x |
| G5 synthetic 4096 | 4096, 4096 | 0.834 | 0.873 | 1.0x |

A flat 0.7 is roughly right for the large synthetic shapes and wrong by a
factor 3 to 6 for the small granite shapes at low token counts. The frozen
expectation E-2-7 predicted efficiency below 0.62 at `M` = 1 for every family
and was refuted by G4 and G5, which sit near their memory roof at `M` = 1
because a tall-thin GEMM against an 8192-squared weight matrix is pure weight
streaming. The direction of the error therefore depends on the shape, which is
precisely why a surface and not a constant is the right object.

The largest GEMM reached 296.57 TFLOP/s at 8192 cubed, 95.10 percent of the
311.87 TFLOP/s clock-derived peak.

## Finding 4: at captured expert loads the expert GEMM is fixed-cost bound

Every one of the 18 captured MoE expert cells is memory-limited under the
roofline, exactly as the campaign brief signed: the A100 machine balance is
152.9 FLOP/B, the per-shape knees are 218 and 277 rows, and the captured
granite population puts 13.5 rows on a balanced expert with 54 as the absolute
ceiling. E-4-1, E-4-2 and E-4-3 all passed.

What the roofline gets wrong is the magnitude, by an order of magnitude. Over
all 18 captured cells the measured time is 5.17 to 12.20 times `t_mem`, with
the minimum at `expert_gate_up` `M_e` = 1 and the maximum at `expert_down`
`M_e` = 4. Four representative cells:

| Cell | measured | `t_mem` at `R_hbm` | measured over `t_mem` |
|---|---:|---:|---:|
| `expert_gate_up`, `M_e` = 14 | 8.845 us | 1.185 us | 7.47 |
| `expert_down`, `M_e` = 14 | 6.640 us | 0.600 us | 11.06 |
| `expert_gate_up`, `M_e` = 54 | 8.979 us | 1.275 us | 7.04 |
| `expert_down`, `M_e` = 1 | 4.725 us | 0.578 us | 8.18 |

E-4-4 predicted a factor in [1.0, 3.0] and was refuted. At these loads the
expert GEMM is not bound by bandwidth and not bound by arithmetic. It is bound
by fixed cost: the uninstrumented back-to-back period of an empty kernel on
this device is 1.904 us, and a granite expert kernel at any captured load
costs 4.7 to 9.2 us. Pricing a captured expert GEMM at its memory roof
understates it by 7 to 11 times. For a 24-layer top-8 granite step this is not
a rounding error, and it is the strongest argument in this study for measuring
a fixed per-kernel cost rather than deriving one.

## Finding 5: the roofline knee is real but sits far above its prediction

The measured knee, the smallest grid `M` whose time exceeds 1.5 times the
small-`M` plateau median, against the crossover computed from the measured roof:

| Family | measured knee `M` | `M*` at measured roof | band [0.7 `M*`, 4 `M*`] | inside |
|---|---:|---:|---|---|
| G1 | 434 | 229.1 | [160.4, 916.3] | yes |
| G2 | 872 | 257.9 | [180.6, 1031.8] | yes |
| G3 | 1024 | 344.8 | [241.4, 1379.1] | yes |
| G4 | 183 | 179.0 | [125.3, 716.1] | yes |
| G5 | 91 | 187.2 | [131.0, 748.8] | **no** |

E-2-4 failed on G5 alone, whose measured knee of 91 sits below the 131 floor.
Four of five families bracket their prediction. The systematic pattern is that
the small granite shapes cross far above their roofline knee, which follows
from finding 4: their small-`M` end is fixed-cost bound rather than memory
bound, so the plateau is higher than the roofline plateau and the crossing
happens later.

## Finding 6: warm and rotated agree, so the warm-state definition is benign

The freeze defines the contract's constant as the warm steady state and
promised the warm-versus-rotated delta as an unscored definition note. It is
small everywhere it was measured: the worst warm-against-rotated difference at
64 MiB and above is 0.687 percent, over the six paired cells E-5-3 scores,
which are `scale`, `add` and `rmsnorm` at 64 and at 256 MiB. The rotation pools
exceeded the 40 MiB L2 by at least a factor 8, so the agreement is not a
failure to cool the cache. On this device, for streaming kernels of this class,
choosing the warm definition costs nothing.

The related prediction E-5-4 also held. E-5-4 bounds how much FASTER than its
own bytes-over-`R_hbm` time the cell may run, so the scored quantity is
`t_mem / t_measured` and its ceiling is 1.15. A 4 MiB scale, whose working set
fits the L2 comfortably, measured 1.0024, meaning it ran 0.24 percent faster
than the roof rather than the 15 percent the bound allowed. It did not convert
L2 residency into bandwidth. A small streaming kernel on this device does not convert L2
residency into bandwidth.

## Finding 7: two lanes measured the harness rather than the hardware

Two lanes produced numbers that describe the microbenchmark and must not be
read as A100 properties.

The decode attention kernel reached 5.5 to 13.3 percent of `R_hbm` over the six
cells E-3-3 scores, which are the ones carrying at least 160 MiB of KV traffic,
and 0.5 to 13.3 percent over all 20 decode cells including the small-batch ones
E-3-3 does not reach. That is after repair R5 put four independent
online-softmax accumulators in every warp. E-3-3
predicted 55 to 100 percent and is refuted for this kernel. The scaling
diagnostics say why: time grew by 3.02 between batch 64 and batch 256 at
`L` = 8192 where the KV bytes grew by 4, so the kernel is still gaining
efficiency with occupancy at batch 256 rather than sitting on the roof. A
production paged or flash decoding kernel is a different program, and this lane
does not bound it.

The granite prefill lane at short sequences is fixed-cost dominated: `S` = 128
takes 11.2 us against a 1.9 us empty-kernel period, and the `S` = 4096 over
`S` = 2048 ratio is 2.56 rather than the quadratic 3.2 to 4.4 that E-3-1
predicted. The synthetic geometry, whose cells are large enough to escape the
fixed cost, measured 4.02 and would have passed on its own.

## Physical sanity review

Three independent framings, as the local rules require.

**Memory physics.** `R_hbm` of 1818.21 GB/s is 89.17 percent of the
memory-clock-derived 2039.04 GB/s ceiling, no cell exceeded that ceiling, and
no cell completed below its compulsory-traffic floor with the full 40 MiB L2
credited. The one cell whose apparent bandwidth exceeds `R_hbm` is the two-pass
RMS normalization at 2020.8 GB/s, and that is arithmetic, not physics: its byte
model charges two activation reads while its second pass is served from cache.
The overshoot of 11.1 percent is therefore an upper bound on how much of a
second pass the caches absorbed, which is a sensible number.

**Compute physics.** The largest GEMM reached 95.10 percent of the
clock-derived FLOP peak with the clock observed at 1410 MHz on both sides of
every timed batch, so comparing against the 1410 MHz peak is legitimate. The
compute-limited cells scale with the SM clock by 1.103 to 1.112 against a clock
ratio of 1.10588, and the memory-limited cells do not scale at all. Two
mechanisms, two signatures, both as physics requires.

**System plausibility.** Take the granite geometry, 24 layers, hidden 1024,
top-8 of 32 experts, one decode step at 54 tokens. The measured expert kernels
alone, at 8.8 us for gate/up and 6.6 us for down per expert per layer, give
24 layers times 8 experts times 15.4 us, which is 2.96 ms of expert GEMM per
step. The roofline prices the same work at about 0.36 ms. A 400M-active-
parameter model at 2.96 ms per step is roughly 340 tokens per second per
request, which is high but not absurd for a tiny model; at the roofline's
0.36 ms it would be 2,800 tokens per second per request, which is not a rate any
published deployment of a 1B-class MoE reaches on one A100. The measured number
is the plausible one, and the gap is finding 4.

## What this study delivers and what it withholds

Delivered, as retained evidence from a void run:

- the measured HBM roof and its size dependence;
- the clock-conditioned constants of 246 boosted cells and 18 base cells, with
  their per-batch clock states, batch-mean spreads, per-repetition series
  summaries and per-cell mean host launch-loop and device batch times, in
  `measurements/results.json`;
- the per-family roofline efficiency surfaces over the knee-anchored grids;
- the measured per-boundary event-instrumentation cost and the uninstrumented
  back-to-back kernel period;
- the warm-versus-rotated deltas;
- the run-1 evaluation of every scored expectation beside run 2's, and the
  excluded cell's per-batch host and device times.

Withheld deliberately:

- **no `simllm-profile-table-v1` artifact is published.** Publishing a loadable
  calibrated table from a void run would let a later study consume constants
  whose stability precondition failed. The constants are in the results file,
  which no provider loads.
- no host-launch profile: that is stage 2.
- no change to any provider, envelope or default. `GPU_ENVELOPES["a100"]` and
  `RooflineProvider(efficiency=0.7)` are untouched.

## What stays open

Three tasks are registered from these findings and are named here so a reader
can trace each one to the measurement that produced it: **COMP-43** owns the
fixed per-kernel cost that finding 4 measures, **COMP-45** owns reaching a
non-void run under a protocol whose stability precondition survives the absent
clock control, and **COMP-46** owns replacing the decode attention kernel of
finding 7.

- **COMP-1 stays open** on both of its blockers. This study captured no
  production framework kernel, replayed no SASS and calibrated no Accel-Sim
  configuration, so its first blocker is untouched. It measured no launch,
  host-delay or queueing term, so its second blocker is untouched here and is
  stage 2's subject. What it adds is a target-architecture efficiency surface
  and, more usefully, the finding that the flat 0.7 derate is wrong in
  different directions for different shapes and that captured MoE expert loads
  are fixed-cost bound rather than memory bound.
- **COMP-5 stays open.** Clocks cannot be locked on this allocation, so the
  controlled-environment stability form cannot be met here at all. This study
  substituted a clock-conditioned form and that form itself failed on 16 of 97
  cells, which is evidence about the environment rather than about the kernels.
- Everything here is one A100 SXM4 80 GB, one node, one toolchain. None of it
  transfers to H100, GH200, B100 or B200.
