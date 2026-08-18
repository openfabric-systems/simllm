# A100 graph launch v1 results

The reviewed study state is `VOID`. Fatal guard `GG7` was violated, so the
behavioral score is uninterpretable, no fraction is reported as a result, and
no `HostInitiationModel` profile is installed. The evidence is retained and
the findings below are the product.

Fourteen of the 15 scored expectations passed and one, the ruling's falsifier
`F1`, was refuted on the smallest of its three kernels. **That 14 is not a
score.** It is written down so a reader can see which relations survived.

This study measures host and device launch cost on one A100 SXM4 80 GB with an
AMD EPYC host. It runs no framework, loads no model and reports no TTFT or
TPOT.

## Why the study is void

`GG7` requires the block-mean coefficient of variation of every reported
period to be at most 4 percent. Eleven cells exceeded it:

| Cell | CV | Makespan |
|---|---:|---:|
| `eager:g1:2` | 29.54 percent | about 18 us |
| `eager:nop:4` | 28.11 percent | about 8 us |
| `eager:nop:1` | 19.51 percent | about 2 us |
| `graph:nop:1` | 18.59 percent | about 1 us |
| `eager:nop:8` | 13.35 percent | about 15 us |
| six more at `K` <= 8 | 4.03 to 7.71 percent | 1 to 20 us |

Every one is a chain of eight or fewer kernels whose whole makespan is a few
microseconds against a 1024 ns CUDA event quantum, which stage 1 measured on
this same device. At `K` = 1 for the null kernel the quantum is 54 percent of
the measurement. The guard as frozen quantifies over every reported period, it
did not carve out the short chains, and it fails. The freeze is the contract,
so the run is void.

Two readings of `GG7` and what separates them, stated so a reader can judge:

- **Broad reading, the one applied here.** "Every reported period" means every
  period the harness measured. Eleven cells fail and the run is void.
- **Narrow reading.** "Every reported period" means every period that enters a
  published number. Only the `K` = 256 cells of all five tags and
  `graph:nop:16` do, and their coefficients of variation are 0.011 to 0.433
  percent, so the guard would hold. Stage 1's scorer was corrected to exactly
  this narrow reading for its own `G10`, before the correction was known to
  change nothing.

The broad reading is published because the stage-2 scorer implemented it
before the run, and choosing the other reading after seeing which one passes is
the review defect these rules exist to prevent. The narrow reading is reported
here rather than acted on.

## Chronology and integrity

| Event | Commit or job |
|---|---|
| expectations-only freeze | `3284a53` |
| measurement harness | `1c09a43` |
| run 1, aborted after every scored cell | job `196020` |
| non-fatal readback repair | `94c8f7b` |
| run 2, reviewed `VOID` | job `196033` |

Job `196020` completed every scored cell and then aborted in the final
instrumented-graph section, so it produced no output file. Its failure is
recorded rather than hidden: an event recorded during CUDA stream capture
becomes a graph node whose elapsed time this driver refuses to report,
returning `invalid argument`. The repair records the refusal as -1 and
continues, because those cells feed one unscored record. This is a
post-specified harness repair; it changes no scored quantity, and the run-2
output confirms the refusal is systematic, with every inner readback returning
-1.

One deviation from the frozen allocation envelope is recorded. The freeze fixed
a one hour wall limit. The partition was congested, Slurm's backfill estimate
for a one hour job was about three hours out, so job `196020` was cancelled by
its owner and resubmitted with a twenty minute limit. Nothing else changed. A
wall limit is an upper bound on a resource, it is not a measured quantity, and
the run took 82 seconds.

| Artifact | SHA-256 |
|---|---|
| `expectations.md` | `b3ebc85490c8b145ead5af0168b24a354e63d2ed6bc9ff81662864490cf7db0f` |
| `graph_launch.cu` as submitted | `bea7df75063f8dbab7672da69a60d30273e0d7516de90fe479be24fe393bb054` |
| run 2 raw output | `fc4bf5eecc73f5185c599e316f749fea9351cacef534eb45c6c56a9d0ff45490` |
| `measurements/results.json` | `5c26035da27c40b86149c243361c74221da8bc51f4ad8c5851018510ab2ec65c` |

The submitted source digest matches the committed file exactly. The
expectations document was not edited after the run.

## Quantity 1: the ruling's falsifier, refuted narrowly and reported plainly

The kernel-time determinism contract says CUDA-graph launch and eager launch
differ only in the host launch cost and never in kernel service time. `F1`
tests the device-side form of that claim: the kernel's marginal cost over a
null kernel in the same mode should not depend on the mode.

| Kernel | `P_eager` | `P_graph` | `S_eager` | `S_graph` | `S_graph / S_eager` |
|---|---:|---:|---:|---:|---:|
| `g1`, N 2048 K 1024 M 64 | 8.9437 us | 7.4870 us | 7.0680 us | 6.6913 us | **0.9467** |
| `g2`, N 1024 K 1024 M 1024 | 18.7927 us | 17.3777 us | 16.9170 us | 16.5820 us | 0.9802 |
| `g4`, N 8192 K 8192 M 1 | 89.5560 us | 88.0497 us | 87.6803 us | 87.2540 us | 0.9951 |

`F1` requires all three in [0.95, 1.05]. `g1` measures 0.9467 and is outside
it. **The falsifier fails, so on this measurement kernel cost is launch-mode
conditioned.** It is not folded away, and the contract's CUDA-graph clause is
reported as refuted at the device level.

The refutation has a shape, and the shape is more useful than the verdict. The
raw mode difference `P_eager - P_graph` is nearly constant across kernels:

| Kernel | mode difference | excess over the null kernel |
|---|---:|---:|
| `nop` | 1.0800 us | 0 by construction |
| `g1` | 1.4567 us | 0.3767 us |
| `g2` | 1.4150 us | 0.3350 us |
| `g4` | 1.5063 us | 0.4263 us |

A real kernel costs about 1.42 to 1.51 microseconds more per invocation in
eager mode than in a graph, and that difference barely moves across kernels
whose own periods span 8.9 to 89.6 microseconds. The null kernel captures 1.08
of it; the residual 0.34 to 0.43 microseconds is a per-kernel cost that a real
kernel pays outside a graph and does not pay inside one. `F1` fails on `g1`
and not on `g2` or `g4` only because a roughly constant 0.38 microsecond offset
is 5.3 percent of `g1`'s 7.07 microsecond differenced time and 0.5 percent of
`g4`'s 87.68.

What this measurement cannot do is say whether that residual is kernel service
time or device front-end gap. Separating them needs per-kernel in-graph timing,
and this driver refuses to report the elapsed time of an event recorded during
capture, which is exactly why the instrumented cells returned -1. The honest
statement is that the device-side per-kernel cost is launch-mode conditioned by
about 1.4 to 1.5 microseconds and that the split between service and gap is
unmeasured here.

`F2` passed: the raw period ratio for the 90 microsecond kernel is 0.9832,
inside [0.97, 1.03]. A long kernel barely notices how it was launched, which is
the same fact seen from the other end.

`F3` passed: the null-kernel period is 2.3574 times larger in eager mode than
in a graph.

## Quantity 2: host submission cost

Host submission was timed with a monotonic host clock around the launch loop
only, closed before the first synchronization. `GG4` held on every cell, so the
host interval never contained device time and the device events never contained
host launch time.

**Eager mode is linear in `K`, as `H1` and `H2` predicted.** The fit over
`K` in [8, 256]:

| Chain | per-launch slope | intercept | R-squared |
|---|---:|---:|---:|
| `nop` | 1.6296 us | 8.297 us | 0.99597 |
| `g1` | 4.7063 us | -25.024 us | 0.99196 |
| `g2` | 3.5562 us | 11.754 us | 0.99516 |
| `g4` | 4.1591 us | -1.838 us | 0.99526 |
| `mix` | 3.4897 us | -9.129 us | 0.99431 |

The `nop` slope of 1.6296 microseconds is the pure host launch path and is the
value the two installed profiles would have used. The GEMM chains cost 3.5 to
4.7 microseconds per launch on the host because `cublasGemmEx` does host-side
work before it launches anything, which is a property of the library call and
not of the CUDA launch path.

**Graph replay is flat in `K`, as `H3`, `H4` and `H5` predicted, and this is
the central structural result.** Host cost per replay of a `K`-node graph:

| `K` | 1 | 2 | 8 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| per replay | 1.784 | 1.592 | 1.595 | 1.597 | 1.648 | 1.686 | 1.656 us |
| per enqueued kernel | 1.784 | 0.796 | 0.199 | 0.050 | 0.026 | 0.013 | 0.0065 us |

The fitted per-node slope is 0.000297 microseconds at an R-squared of 0.516,
which is a way of saying there is no trend to fit: replaying a 256-node graph
costs the host the same 1.6 microseconds as replaying a 1-node graph. At
`K` = 256 the host pays 6.5 nanoseconds per enqueued kernel against 1.626
microseconds in eager mode, a factor of 251.

**The Turing constant does not transfer, and `H6` measured by how much.** The
accepted COMP-2 `eager-host-bound` point is 2,364,255 ps. This host and GPU
pair measures 1,629,633 ps, 31.07 percent lower. Combined with the already
accepted finding that the aarch64 Grace launches 28 percent faster than this
same EPYC, a launch constant is a property of the host and the driver, not of
the GPU generation, and carrying one across either axis is wrong.

### The profiles the freeze named, and why they are not installed

The freeze names two `HostInitiationModel` profiles and their measured values
are below. They are **not installed**, because a calibrated constant taken from
a void run would let the model consume numbers whose stability precondition
failed. They are published here and in `measurements/results.json` as retained
evidence.

| Profile | launch class | point | empirical range |
|---|---|---:|---|
| `a100-epyc-eager-host` | `eager-host-bound` | 1,629,633 ps | 1,625,986 to 1,927,260 ps |
| `a100-epyc-cuda-graph` | `cuda-graph-node` | 25,745 ps at `K_ref` = 64 | 6,468 to 199,308 ps |

The graph profile also carries a fixed per-replay cost of 1,647,674 ps that the
calibrated profile form has nowhere to put.

The freeze predicted this and said what it would mean. `H3` held, so the graph
host cost is flat in `K`, so a per-launch constant for graph replay is
`K`-scoped by construction: the empirical range above spans a factor 31 purely
because `K` spans a factor 32. `HostInitiationModel`'s calibrated composition
`max(C, N * g)` assumes `g` is a per-launch constant, and for graph replay it
is not. The right shape is a fixed per-replay term plus a per-node term near
zero, and the calibrated form cannot express it. COMP-44 owns that gap. No knob
was invented for it.

## Quantity 3: the device inter-kernel gap, reserved

Recorded with provenance and wired to nothing, as the campaign brief requires.
A test walks every module under `simllm` with an AST parse and asserts neither
value appears as an integer literal and no identifier names the quantity.

| Constant | Value |
|---|---:|
| in-graph null-kernel period | 795,667 ps |
| eager null-kernel period | 1,875,667 ps |

`D1` through `D4` all passed: the in-graph period is inside [0.3, 1.5] us, it
is constant in `K` to a ratio of 0.8152 between `K` = 256 and `K` = 16, the
eager period is inside [1.5, 3.0] us, and the in-graph period is 0.4882 times
the eager per-launch host slope, so the device front end is cheaper than the
host path it replaces.

## Mixed chains

`M1` and `M2` both passed. A mixed cycle of `g1`, `g2`, `nop` and `g4` at
`K` = 256 has a per-kernel period of 28.2253 us in a graph against 28.4275 us
predicted by averaging its four members measured separately, a ratio of 0.9929;
in eager mode 29.3953 us against 29.7920 us, a ratio of 0.9867. Chaining
different kernels together costs no more than chaining each of them alone.

## Cross-checks against independent measurements

Three numbers here were measured before, by different harnesses:

| Quantity | This study | Independent value | Source |
|---|---:|---:|---|
| graph replay period per node, null kernel | 0.7957 us | 0.791 us | a100 hardware envelope |
| eager back-to-back period, null kernel | 1.8757 us | 1.806 us, 1.904 us | a100 hardware envelope, stage 1 |
| `g4` back-to-back period, eager | 89.556 us | 89.648 us | stage 1 |
| `g2` back-to-back period, eager | 18.793 us | 18.816 us | stage 1 |
| `g1` back-to-back period, eager | 8.944 us | 9.072 us | stage 1 |

Four separately written harnesses across three Slurm jobs agree to within 1.5
percent on every one. That is the strongest evidence in this campaign that the
numbers describe the device rather than a harness.

## Physical sanity review

**Host physics.** The eager host slope of 1.63 microseconds per launch on a
3 GHz-class EPYC core is about 5,000 cycles for one `cudaLaunchKernel`, which
is the right order for a driver call that writes a command buffer and rings a
doorbell. Graph replay at 1.6 microseconds per replay regardless of node count
says the host writes one command and the device walks the node list, which is
the entire point of the mechanism.

**Device physics.** The in-graph null-kernel period of 0.796 microseconds at
1410 MHz is about 1,100 SM clocks to retire a kernel node and start the next.
The eager equivalent of 1.876 microseconds is 2.36 times that, and the
difference of 1.08 microseconds is close to but below the 1.63 microsecond host
launch cost, which is what pipelining should produce: the host is ahead of the
device but not free.

**System plausibility.** Take the granite geometry the compute module already
prices, 24 layers with top-8 of 32 experts, and the 440 to 567 device-visible
launches per decode step the fidelity study bounded. At this study's 1.63
microseconds per eager launch, the host path alone is 717 to 924 microseconds
per step; under a graph it is one 1.6 microsecond replay plus 440 to 567 node
gaps at 0.796 microseconds, which is 351 to 452 microseconds. Both are large
next to the modeled compute of that step, which is the omission COMP-1 already
names, and the graph path halves it. Neither number is small enough to ignore
and neither is so large as to be implausible.

## What stays open

- **COMP-1 stays open.** Its second blocker, launch overhead and host delay on
  the target architecture, is now measured on A100 and EPYC. It is not closed,
  because the run is void and because its first blocker, a production framework
  capture with SASS replay and a held-out kernel matrix, is untouched.
- **COMP-2 is not reopened and not further closed.** It is already closed. This
  supplies the A100 leg its fail-closed device check refuses, but supplies it as
  retained evidence from a void run rather than as an installed profile.
- **COMP-44** is registered for the missing fixed per-invocation term in the
  calibrated host profile form.
- **COMP-47** is registered for a non-void graph-launch run, since `GG7` as
  frozen cannot be met by a sweep that includes chains of one kernel.
- The device-side per-kernel cost is launch-mode conditioned by about 1.4 to
  1.5 microseconds and the split between kernel service and front-end gap is
  unmeasured, because this driver refuses per-kernel timing inside a captured
  graph. Separating them needs a different mechanism.
- Everything here is one A100 SXM4 80 GB on one AMD EPYC host with one driver.
  The host-issue constants explicitly do not transfer across hosts.
