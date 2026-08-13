# Compute fidelity v1 results

Run on 2026-08-13. All 12 fatal guards held and 101 of 102 genuine-risk
instances passed. The one refuted relation, FIX-2, was refuted for a reason
worth keeping: the inequality it registered had the wrong sign for the
quantity it meant to capture, and the correctly measured quantity is reported
beside it.

Three results, in the order that decides how much of SimLLM's serving numbers
can be defended:

1. The 2 percent coefficient-of-variation ceiling that kept COMP-1 open was
   failed by 7 samples out of 2,050. Every one of the 50 cells has an
   excursion-trimmed CV below 1.06 percent, and a fresh device probe attributes
   the excursions to SM sharing and to clock-state drops on a display GPU,
   neither of which is kernel service-time variation. The ceiling is refrozen
   in an explicitly environment-scoped form and the original bar is retained
   unchanged for the production capture.
2. A Turing anchor transfers the pipeline, the artifact format, the
   interpolation rule and the provider seam. It transfers no numbers, and the
   seam is machine-checked to fail closed on a B100 query rather than borrow
   one.
3. The fixed per-step cost that the modeled compute path omits entirely is
   worth between 2.8 and 13.3 times the whole modeled decode compute of a
   24-layer top-8 MoE step. This is the largest identified error in the
   project's serving numbers, and it is larger than the network corrections
   that preceded it.

Neither COMP-1 nor COMP-5 closes. No table is produced, no Turing number is
transferred onto a production envelope, and no new task ID is registered.

## Chronology and provenance

The expectations-only commit is
`62c088ecef78e5609a9a92c1830e0a6445375f74`. It precedes the implementation, the
probe source and every measurement. It names the three published failing cells
as prior observations, so the genuine-risk denominator is the 47 cells whose
raw samples had never been read, and it writes out the proposed stability
refreeze in full before any statistic could shape it.

The implementation commit is `7de897701e93d16873bf9350139fa9f71d870703`, which
is the commit every registered run observed. The check-only command ran after
it, validated the 50-cell inventory, the 47-cell denominator, the 102 scored
instances, the 12 fatal guards and the `[440, 567]` launch bracket, invoked no
CUDA and created no output directory.

Three captures ran. Each change between them was a harness defect, never a
frozen band, and all three replicate the same scored outcome:

| external output suffix | outcome |
|---|---|
| `compute-fidelity-v1` | VOID. 101 of 102 scored instances passed but the fatal guard XFER-G4 failed by 1 picosecond, because it asserted exact integer equality between a doubled kernel and twice a single kernel whose true value is 793,650,793.65 ps and is rounded once per call. |
| `compute-fidelity-v1-final` | All 12 fatal guards held, 101 of 102 scored. The TTFT projection still added the whole per-step fixed cost to a makespan that already contains the compute term. |
| `compute-fidelity-v1-registered` | The registered run. All 12 fatal guards held, 101 of 102 scored, and the projection replaces the compute term instead of adding to it. |

The XFER-G4 correction did not weaken the claim. The claim is that the roofline
provider carries no additive launch, scheduling or sampling constant, and it is
now tested in a form integer rounding cannot reach: a kernel with zero flops
and zero bytes returns exactly 0 ps. The proportionality check keeps its
original form with one picosecond of rounding allowed, which is six orders of
magnitude below the smallest per-launch cost this study measures.

The compact tracked artifacts are:

| artifact | SHA-256 | role |
|---|---|---|
| [results.json](results.json) | `a85df887f588d498b954bcf316ca83c8e1dc85f382d91e55c9d002db524aa66d` | per-cell statistics, per-launch attribution rows, launch measurements, bounds and every guard |
| [expectations.json](expectations.json) | frozen at `62c088e` | matrix, bands, launch enumeration and inventory |
| probe source | `51fe1b90460d8f9755ffbc0b6b7835215acd7117c04a332e78dcfb39040dd170` | `tools/compute_capture/gpu_fixed_cost_probe.cu` |
| input calibration artifact | `0be6dad653ff32a0f4667b5cb05f7ddaefdc06e8f6f1cea032bb8d285b42023f` | the immutable Turing capture this study re-reads, unchanged |

Bulk outputs live below `${SIMLLM_WAVE11_RUN_ROOT}/compute-fidelity-v1-registered`.
The device was the same NVIDIA GeForce GTX 1660 Ti at compute capability 7.5,
24 SMs, 1,800 MHz reported clock and 6,001 MHz memory clock on a 192-bit bus,
i.e. 288.0 GB/s peak. The toolchain was CUDA 12.4 `nvcc` targeting `sm_75`.
This study needs no profiler: it times with CUDA events, with the GPU's own
nanosecond global timer read from inside the kernel, and with the host clock.

## Physical sanity before accuracy

Every number is named against its envelope. The repository default is
`GPU_ENVELOPES["b100"]` at 8.0e12 bytes/s. A caller that omits `gpu=` gets
B100, not H100. Guard XFER-G5 reproduces every value below from
`GPU_ENVELOPES` to within 1e-9 relative.

For the motivating decode step at 556,449,792 active weight bytes:

| bound | envelope | value | where the modeled 99.366 us sits |
|---|---|---:|---|
| weight-read floor at full bandwidth | b100, 8.0e12 B/s | 69.556224 us | 1.4286 times above it, i.e. exactly the flat 0.7 derate |
| weight-read floor at full bandwidth | h100, 3.35e12 B/s | 166.104416 us | 1.6716 times **below** it |
| weight-read floor at full bandwidth | a100, 2.039e12 B/s | 272.903282 us | 2.7465 times below it |

The published 99.4 us step is exactly `556,449,792 / (8.0e12 * 0.7)`. It is a
B100 number sitting on a B100 floor at a flat 70 percent efficiency that no
measurement supports. Attributing it to H100 would put it 1.67 times faster
than the H100 hundred-percent-bandwidth floor, which is impossible.

The probe's own measurements sit inside their bounds:

| measurement | floor | measured | where it sits |
|---|---:|---:|---|
| `attn_gemm` FP32 shape 8 kernel span | 25,165,824 B over 288.0 GB/s = 87.367 us | 100.352 us | 1.1486 times the floor, i.e. 87.1 percent of peak DRAM bandwidth |
| per-launch cost | 0 ns | 630 to 2,332 ns | inside `[0, 5,053.5]`, the measured launch-plus-synchronize latency |

Two independent timing paths agree on the same kernel. The tracked Nsight
Systems median for that cell is 99.809 us and the probe's in-kernel global
timer span is 100.352 us, a 0.54 percent difference (guard G6 allowed 25
percent). CUPTI activity timing and the GPU's global timer are separate
mechanisms, so this is a real cross-validation and not a restatement.

Third angle, end-to-end plausibility. The modeled 224.0 us decode step implies
4,464 tokens per second for a single request. Real single-request decode of a
model this size is reported in milliseconds per token, not hundreds of
microseconds. The bound in part FIX below moves the modeled step to between
0.40 and 1.45 ms, which is the scale at which such deployments are actually
reported, and it identifies the missing term rather than tuning a derate to
match.

## Part VAR: what the 2 percent ceiling actually measured

### The census

Across all 50 cells and all 2,050 samples of the tracked Turing capture, using
the frozen 1.05 excursion threshold:

| statistic | value |
|---|---:|
| samples above 1.05 times their cell median | 7 of 2,050, i.e. 0.341 percent |
| cells containing at least one excursion | 7 of 50, each containing exactly one |
| worst all-sample coefficient of variation | 2.432 percent |
| median all-sample coefficient of variation | 0.320 percent |
| worst excursion-trimmed coefficient of variation | 1.054 percent |
| median excursion-trimmed coefficient of variation | 0.228 percent |
| worst quartile relative spread | 1.241 percent |
| worst maximum excursion ratio | 1.1561 |

Seven samples. The 2 percent all-sample ceiling was failed by seven samples out
of two thousand and fifty, one in each of three cells.

The three published failing cells, recomputed from raw samples (guard G2
reproduces their published CVs to better than 1e-3 percentage points):

| family | dtype | shape | all-sample CV | trimmed CV | quartile spread | excursions | max ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| `attn_gemm` | FP32 | 8 | 2.395% | **0.172%** | 0.192% | 1 of 41 | 1.1561 |
| `lm_head` | FP32 | 4 | 2.343% | **0.212%** | 0.251% | 1 of 41 | 1.1521 |
| `attn_score` | FP64 | 1 | 2.432% | **0.842%** | 1.064% | 1 of 41 | 1.1513 |

A cell whose 41 samples agree to 0.17 percent failed a 2 percent dispersion bar.
That is not a statement about the kernel; it is a statement about the estimator.

The worst never-observed cells tell the same story. `kv_read` FP64 shape 2 has
an all-sample CV of 1.662 percent and a trimmed CV of 0.423 percent, again from
one sample in 41. The worst trimmed CV anywhere in the capture is `kv_read`
FP64 shape 1 at 1.054 percent, and that cell has **zero** excursions, so its
trimmed and all-sample CV are identical. The genuine kernel dispersion ceiling
on this device is therefore about 1.05 percent, comfortably inside the 2
percent bar, and every failure of that bar came from the tail.

### The mechanism, measured directly

The re-analysis above shows the failures are tail-driven. It does not show
what produces the tail, and a study that stopped there would be trimming
outliers after seeing them. The probe measures the cause on the device.

`gpu_fixed_cost_probe --stability 4000` replays the exact failing cell
(`attn_gemm` FP32 shape 8, 2,097,152 work items, 8,192 blocks of 256 threads,
one 32 MiB L2 flush between samples) and records per launch both the wall
duration and, from each block, its own `clock64()` cycle span and its own
global-timer residency span. Those two together give the effective SM clock,
which separates causes that a duration alone cannot:

| observation | registered run |
|---|---:|
| launches | 4,000 |
| median kernel span | 100.352 us |
| run effective SM clock | 1,869.3 MHz |
| excursions above 1.05 times the median | 61, i.e. 1.53 percent |
| attributed to longer block residency by the frozen rule | 57 of 61, i.e. 93.4 percent |
| launches whose effective SM clock fell below 0.9 of the run clock | 3 |
| lowest effective clock ratio observed | 0.7695 |

The same attribution was computed independently on the CUDA-event duration of
every launch as well as on the in-kernel span, because choosing one timing
source after seeing the other would be a choice this study should not get to
make. The event view finds 60 excursions with 90.0 percent attributed to
residency and the same 3 clock drops; both views name the same dominant cause,
which is what PROBE-2 required.

Both physical causes are present and both are environmental:

- **SM sharing**, 57 of 61 excursions. The block's own cycle span rises with
  its wall time and the effective clock stays at about 1.87 GHz. The block was
  resident longer because something else was on the SM. On a GPU driving a
  display, that something else is the desktop.
- **Clock-state drops**, 3 launches. The effective clock falls to 76.9 percent
  of the run clock. These were left unattributed by the frozen rule, which
  expected a clock drop to leave the cycle count flat. It does not, for a
  memory-bound kernel: the memory clock is unchanged, so at a lower SM clock
  the same wait costs fewer SM cycles. The frozen rule is reported exactly as
  registered and the effective-clock measurement is reported beside it as
  descriptive evidence, not folded into any score.

Neither cause is kernel service-time variation, and neither can be removed
without administrator action: locking clocks needs `nvidia-smi -lgc`, and
removing the display work needs the GPU not to be the display device. Both are
outside this study's permissions.

The three captures replicate:

| quantity | run 1 | run 2 | registered run |
|---|---:|---:|---:|
| probe excursion fraction | 1.75% | 1.93% | 1.53% |
| fraction attributed to residency | 94.3% | 94.8% | 93.4% |
| launches with an effective clock drop | 4 | 4 | 3 |
| lowest effective clock ratio | 0.787 | 0.787 | 0.770 |

### Scored variation relations

| relation | passed | total | worst observed |
|---|---:|---:|---:|
| VAR-S1 trimmed CV below 2 percent | 47 | 47 | 1.054 percent |
| VAR-S3 maximum excursion ratio at most 1.35 | 47 | 47 | 1.1049 |
| VAR-S2 aggregate excursion fraction below 5 percent | 1 | 1 | 0.341 percent |

Denominators are the 47 cells whose raw samples had never been read, fixed
before the run. The three published cells are reported above as post-specified
confirmations and are not added to those 47.

### The refreeze

Every condition the freeze attached to the refreeze was met: VAR-S1, VAR-S2 and
VAR-S3 all passed, and PROBE-2 attributed 93.4 percent of excursions to one
identified environmental cause. The COMP-1 stability clause is therefore
replaced by the form written down in `expectations.md` before any statistic was
computed, and reproduced verbatim in `docs/modules/compute.md`:

> A capture cell is stable when its excursion-trimmed coefficient of variation
> is below 2 percent, its excursion fraction is below 10 percent of the cell's
> samples, and its maximum excursion ratio is below 1.35, where an excursion is
> a sample above 1.05 times the cell median. Every cell additionally reports
> its all-sample coefficient of variation and its full excursion census; no
> sample is ever discarded from the artifact.
>
> This form applies only to a capture environment explicitly declared as a
> shared display GPU without clock control. The original all-sample 2 percent
> ceiling remains the bar for a controlled environment, defined as a
> non-display device with locked application clocks and exclusive compute
> access, and remains the bar that the production target-architecture capture
> must meet.

This is not a weakening. The production bar is unchanged and still unmet. What
changed is that the reason COMP-1 was open is no longer "the numbers were
unstable", because they were not: it is "the capture environment is a shared
display GPU", which is COMP-5's requirement and is where it now sits alone.
Under the refrozen form the tracked Turing capture passes in all 50 cells, and
its 7 excursions remain in the artifact, in the census and in this report.

## Part XFER: what a Turing anchor transfers

**Transfers.** The capture pipeline: per-cell immediate warmup, per-cell
profiler capture ranges, ordered report collation and `cuda_gpu_trace` CSV
parsing. The `simllm-compute-calibration-v1` artifact and its strict
provenance. The immutable train and held-out split. The table compiler that
turns train medians into `simllm-profile-table-v1` with held-out error setting
family uncertainty. The one-axis log-linear interpolation rule. The
`ProfileTableProvider` seam into `ComputeProvider.estimate`. This study adds
two more transferable pieces: the excursion census and the in-kernel effective
clock measurement, both of which work on any CUDA device and require no
profiler and no counter permission.

**Does not transfer.** Every duration. Every derived uncertainty. The
per-family efficiency-versus-shape surface. The measured separation between the
calibrated table and the roofline bootstrap. The physical-bound constants of
288 GB/s, 48 GB/s and 1.5 Gop/s, which are TU116 numbers. TU116 has no tensor
cores, no TMA, no warpgroup operations and no thread-block clusters, so it
cannot exercise the paths a Hopper or Blackwell serving kernel spends its time
in.

The boundary is enforced, not merely asserted. Guard XFER-G1 confirms the
tracked table declares `NVIDIA GeForce GTX 1660 Ti` provenance with entry GPU
keys `gtx1660-ti-sm75-fp32` and `gtx1660-ti-sm75-fp64`, disjoint from every key
in `GPU_ENVELOPES`. Guard XFER-G2 confirms that asking that table for
`attn_gemm` at `new_tokens=1` on `b100` raises `KeyError` rather than returning
the Turing number. A Turing table cannot leak into a production envelope by
accident; it has to be deliberately rewritten, which is what COMP-1 requires.

## Part FIX: bounding the omitted fixed per-step cost

### There is no fixed term anywhere in the compute path

Guard XFER-G4 establishes it structurally: `RooflineProvider` returns exactly
0 ps for a kernel with zero flops and zero bytes, and doubling the work doubles
the duration to within 1 ps of integer rounding. A modeled step is exactly its
kernel service time. Guard XFER-G3 confirms `HostInitiationModel()` defaults to
`initiation_delay_ps = 0` with profile `ideal`, and that model is in any case a
per-send network initiation delay, not a per-kernel launch cost. Nothing in the
compute path prices kernel launch, scheduling or sampling.

### Launches per eager decode step

Enumerated statically from vLLM 0.26.0 for the pinned granite MoE geometry (24
layers, 32 experts, top-8), from `GraniteMoeDecoderLayer.forward`,
`GraniteMoeAttention.forward`, `GraniteMoeMoE.forward` and
`fused_experts_impl`. The full per-op table is frozen in `expectations.json` and
the harness re-derives the totals from it, so the bracket cannot drift from its
enumeration:

- 18 to 23 device-visible launches per decoder layer,
- 8 to 15 outside the layers (input copies, embedding, final norm, LM head,
  greedy sampling),
- **440 to 567 launches per eager decode step.**

Tensor-parallel all-reduces are excluded because the reference configuration
puts that collective on the fabric backend, which the network model already
prices.

### Measured per-launch cost, Turing device and this host

| quantity | measured |
|---|---:|
| host enqueue of one empty launch, no synchronization | 2,331.76 ns |
| eager stream throughput, 20,000 launches plus one synchronization | 2,331.96 ns |
| launch plus `cudaDeviceSynchronize` latency | 5,053.54 ns |
| CUDA-graph replay, 512 nodes replayed 200 times | 630.12 ns per node |
| device-side inter-kernel gap of a real 1,024-block kernel | 1,602.56 ns |
| in-kernel service of that same kernel | 13,169.28 ns |

The eager stream is entirely host-bound: enqueue and pipelined throughput
differ by 0.199 ns per launch over 20,000 launches, so the device finished
essentially when the host stopped enqueuing. Graph replay is 3.70 times
cheaper per launch than eager stream launching. The device-side gap is 10.8
percent of the total time of a back-to-back 13.2 us kernel, and it is a cost
the model omits even when the host is infinitely fast.

FIX-2 was refuted, and the refutation is informative. It registered
`long_backtoback_ns > long_isolated_ns`, expecting the back-to-back batch to
be slower per launch than an isolated launch. The observed difference is
-1,919.84 ns. The reason is that an isolated event window necessarily contains
the empty-queue launch latency, while a back-to-back batch hides it behind the
previous kernel, so the isolated measurement is the larger of the two and the
registered inequality has the wrong sign for the quantity it meant to capture.
The correctly signed measurement is the `stamped_device_gap_ns` row above, taken
by subtracting the summed in-kernel global-timer spans from the wall time of the
whole batch, and it is strictly positive at 1,602.56 ns. The registered relation
is reported as failed; it is not rewritten.

### The bound

A step is not the sum of its service time and its launch cost. With the host
running ahead the two overlap, so the step floor is
`max(kernel service, launches * per-launch cost)`, and the amount the model
omits is `max(0, launches * per-launch cost - modeled compute)`. Against the
modeled 99.36603 us B100 decode compute, over the frozen `[440, 567]` bracket:

| per-launch regime | ns per launch | step fixed cost | omitted | multiple of modeled compute |
|---|---:|---:|---:|---:|
| CUDA-graph replay | 630.12 | 277.3 to 357.3 us | 177.9 to 257.9 us | 2.79x to 3.60x |
| device-side gap only | 1,602.56 | 705.1 to 908.7 us | 605.8 to 809.3 us | 7.10x to 9.14x |
| eager, host-bound | 2,331.96 | 1,026.1 to 1,322.2 us | 926.7 to 1,222.9 us | 10.33x to 13.31x |

The lower end of that bracket is the honest floor for a production deployment,
because vLLM captures decode steps into CUDA graphs. Even there, the omitted
fixed cost is 2.8 to 3.6 times the entire modeled compute of the step. The
eager figure is the upper bound and the one that applies to any step a
deployment cannot capture.

### What it does to a modeled TTFT and TPOT

Applied to the published end-to-end run, replacing the compute term rather than
adding to it:

| regime | modeled TTFT | TTFT with the fixed cost | increase |
|---|---:|---:|---:|
| CUDA-graph replay | 974.8 us | 1,152.7 to 1,232.8 us | +18.2% to +26.5% |
| device-side gap only | 974.8 us | 1,580.6 to 1,784.1 us | +62.1% to +83.0% |
| eager, host-bound | 974.8 us | 1,901.5 to 2,197.7 us | +95.1% to +125.4% |

The decode step moves further, because it is smaller. The published 224.0 us
step becomes 401.9 to 481.9 us under graph replay and 1,150.7 to 1,446.9 us
eager, i.e. a modeled 4,464 tokens per second per request becomes 2,075 to
2,488 under graph replay and 691 to 869 eager.

That is the mission-relevant conclusion. After the network corrections, a
decode step was compute bound at roughly 99 us of modeled compute against a few
microseconds of communication. The fixed per-step cost the model omits is
between 2.8 and 13.3 times that compute, so the step is not compute bound at
all: it is launch bound, and the binding constraint on every reported TTFT,
TPOT and goodput number is a term the model does not have a place to put.

### What this bound is and is not

It is a Turing measurement on this host's CPU. The per-launch cost is dominated
by CUDA runtime and driver work on the host, which is why enqueue and pipelined
throughput coincide, so a faster server CPU and a newer driver would lower it,
and a B100 host would very likely land below 2.3 us per eager launch and below
630 ns per graph node. It is not a B100 launch model and this study does not
turn it into one.

What survives the transfer is the structure, not the constant. The launch count
is a property of the model geometry and the framework, not of the GPU. The
conclusion that the omitted term is a multiple of the modeled compute rather
than a correction to it survives any plausible per-launch cost: it would take a
per-launch cost below 225.8 ns for the graph-replay bound to fall under the
modeled 99.4 us even at the low end of the launch bracket, which is 36 percent
of what this device measures for a graph node. Establishing the production
constant needs the target architecture, which is COMP-5.

## Genuine-risk evidence

| scored family | passed | total | why it could fail independently |
|---|---:|---:|---|
| VAR-S1 trimmed CV below ceiling | 47 | 47 | a genuinely dispersed kernel spreads its central mass and fails regardless of the tail |
| VAR-S3 maximum excursion ratio bounded | 47 | 47 | a heavy or unbounded tail exceeds the bound even when the core is tight; a statement about the tail, not the core |
| VAR-S2 aggregate excursion fraction | 1 | 1 | a frequent rather than sparse disturbance |
| PROBE reproduction, attribution and sparsity | 3 | 3 | the device could have been clean, or the excursions could have refused to attribute to one cause |
| FIX launch bracket and bound | 3 | 4 | FIX-2 refuted; the registered inequality had the wrong sign for its intent |
| **total** | **101** | **102** | |

Entailment analysis. Every VAR relation was computed from raw samples before
any artifact-identity, inventory or reproduction guard ran. Those guards assert
sample counts, published summary reproduction and split identity; the published
summaries carry minimum, median, maximum and all-sample CV only, so none of them
constrains a trimmed CV, an excursion ratio or an excursion fraction. VAR-S1 and
VAR-S3 are computed from the same sample vector but neither entails the other:
S1 can pass with an arbitrarily large single outlier and S3 can pass with a
uniformly dispersed core. PROBE and FIX are evaluated on data that did not exist
at freeze time. Therefore no earlier fatal oracle pins a scored result.

The following are fatal unscored guards. They are structural, compatibility or
by-construction evidence and never increase a behavioral denominator:

| guard | result |
|---|---|
| G1 artifact inventory, 50 cells and 2,050 durations | PASS |
| G2 published CV reproduction to 1e-3 percentage points | PASS |
| G3 exactly 3 of 50 cells at or above the 2 percent all-sample ceiling | PASS |
| G4 47-cell genuine-risk denominator | PASS |
| G5 probe device identity matches the tracked capture provenance | PASS |
| G6 probe median within 25 percent of the tracked cell median (observed 0.54 percent) | PASS |
| G7 every probe launch positive, all 8,192 blocks reporting | PASS |
| XFER-G1 tracked table declares the Turing GPU and no envelope key | PASS |
| XFER-G2 table raises `KeyError` on a `b100` query | PASS |
| XFER-G3 `HostInitiationModel()` defaults to zero delay, profile `ideal` | PASS |
| XFER-G4 roofline is exactly proportional with no additive term | PASS |
| XFER-G5 envelope floors and ratios reproduce to 1e-9 relative | PASS |

## Closure scope

### COMP-1

| registered clause | evidence and disposition |
|---|---|
| "measured coefficient of variation below 2 percent for controlled microbenchmarks" | REFROZEN, not dropped. The original all-sample bar stands unchanged for a controlled environment and is still unmet, because no controlled environment exists here. An environment-scoped form is added for a declared shared display GPU, and the tracked Turing capture passes it in all 50 cells with a worst trimmed CV of 1.054 percent. The evidence is 7 excursions in 2,050 samples plus a device probe attributing 93.4 percent of a fresh excursion population to SM sharing and the rest to measured clock drops to 76.9 percent of the run clock. |
| "Launch overhead, host delay and queueing are measured separately from kernel service" | PARTIAL. Launch overhead is now measured separately, on Turing, in four regimes, and its magnitude is bounded against the modeled step. No production-architecture measurement exists and no seam carries the term into a modeled step. |
| "Pin a support envelope for every table" | PASS for the Turing benchmark envelope only, unchanged from the previous study. |
| "Capture the exact production run first" | NOT DEMONSTRATED. |
| "NVBit supplies the SASS traces required by Accel-Sim" | NOT DEMONSTRATED. |
| "Build one replayable microbenchmark per captured kernel implementation" | PARTIAL, unchanged. |
| "Replay traces offline with a pinned Accel-Sim/GPGPU-Sim configuration" | NOT DEMONSTRATED. |
| "Populate `simllm-gpu-model-artifact-v2`" | NOT DEMONSTRATED. |
| "100 percent kernel identity coverage for the supported run" | PASS for the benchmark families only, unchanged. |
| "held-out per-kernel median error below 10 percent and p95 below 20 percent" | PASS at 0.674 and 1.773 percent, unchanged from the previous study. |
| "per-phase median below 5 percent and p95 below 10 percent" | NOT DEMONSTRATED. |
| "compute-only step error below 5 percent" | NOT DEMONSTRATED, and now known to be unreachable while the fixed per-step cost is absent: the omitted term alone is 2.8 to 13.3 times the modeled compute. |
| B100 efficiency-surface transfer | NOT DEMONSTRATED. Machine-checked to fail closed instead. |

COMP-1 stays open. What changed is the reason: it is no longer open because
its numbers were unstable, and it is now open because it has no
target-architecture capture, no dynamic SASS ledger, no Accel-Sim replay, and
no seam for the fixed per-step cost this study bounded.

### COMP-5

| registered clause | evidence and disposition |
|---|---|
| "a nonempty activity trace" | PASS, already demonstrated by the previous study. |
| "successful required-counter probe" | NOT DEMONSTRATED. The loaded driver reports `RmProfilingAdminOnly: 1` and Nsight Compute returns `ERR_NVGPUCTRPERM`. This needs an administrator and was not attempted. |
| "exact tool and GPU provenance" | PASS. |
| "every registered cell below the stability ceiling" | PASS under the refrozen environment-scoped form for the shared display GPU, in all 50 cells. NOT DEMONSTRATED under the controlled-environment form, which requires locked clocks and a non-display device. |
| "a stable non-display or exclusive capture environment, controlled clocks" | NOT DEMONSTRATED, and now characterized rather than suspected: 93.4 percent of excursions are SM sharing and the remainder are clock-state drops to 76.9 percent of the run clock. Both need permissions this study does not have. |
| "allocation on the exact A100, H100 or B100 target with compatible dynamic NVBit tracing and Accel-Sim support" | NOT DEMONSTRATED. |

COMP-5 stays open. Its stability requirement is now backed by a measured
mechanism instead of an inference from outliers, which makes it concrete: the
GPU must not be driving a display and its clocks must be lockable.

## Residuals and registered IDs

**Zero IDs registered.** The wave-10 residual rule permits a new ID only for a
registered acceptance clause a run did not demonstrate. Neither COMP-1 nor
COMP-5 closes, so every clause either of them registered stays with the task
that registered it and none is dropped. The refreeze adds an
environment-scoped form beside the original ceiling rather than removing it,
so nothing is lost there either. FIX-2's refutation concerns this study's own
measurement design, not a COMP clause, and the correctly measured quantity is
reported.

Two findings are recorded as prose here and in the module doc's narrative,
which is where the rule says they belong:

- There is no seam anywhere in the compute path that can carry a fixed
  per-step cost. `RooflineProvider` is exactly proportional,
  `ProfileTableProvider` returns a measured kernel duration, and
  `HostInitiationModel` is a per-send network initiation delay rather than a
  per-kernel launch cost. Adding an uncalibrated knob would have been the
  fabricated number this study exists to avoid, so none was added. COMP-1
  already registers launch overhead as in scope and carries this.
- The frozen clock-attribution rule expected a clock drop to leave the cycle
  count flat. For a memory-bound kernel it does not, because the memory clock
  is unchanged. The effective-clock measurement that resolves this is in the
  probe and in `results.json`, and any future capture using this rule should
  use effective clock rather than cycle count.

## Contradiction sweep

No integrator-owned overview file was edited. The sweep found two statements
that now need nuance:

- `docs/README_PRO.md` says production SASS calibration and populated profile
  tables remain blocked on capture hardware under COMP-5. Half of that blocker
  is now resolved: the stability half is characterized and the tracked capture
  passes the environment-scoped form. What remains blocking is counter
  permission and target-architecture allocation.
- `docs/architecture.md` describes `initiation_delay_ps` as the analytical
  fallback for launch-path studies. That is true for the network launch path,
  per send. It does not cover the per-kernel launch path inside a step, which
  this study bounds at 2.8 to 13.3 times the whole modeled compute of a decode
  step and for which no seam exists.

`README.md` still correctly calls SASS offline calibration planned under
COMP-1 and COMP-5.

## Validation

The probe compiled for `sm_75` and both modes ran to completion on all three
captures. `ruff check .` passed and the full suite passed 1,304 tests with 7
skips. `python3 scripts/task_progress.py --check` reports no drift, and no task
closes, so `docs/task-ledger.json` and the generated progress block
intentionally do not change. The registered study command reached its final
acceptance assertion and exited nonzero solely because the scored relation
FIX-2 was refuted.
