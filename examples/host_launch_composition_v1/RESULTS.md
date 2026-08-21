# Host launch composition v1 results

## Outcome

The run is **nonvoid**. All five fatal guards held, both genuinely risky
relations passed, and every exact closed-form identity held on all 32 cells.

The result is a **refutation**, not a closure. Under the composition the
shipped calibrated host profiles actually use, COMP-48's acceptance bar is
unreachable: the model predicts a launch-mode delta of exactly zero in every
regime where the A100 graph launch study measured a delta of 1.415 to 1.506
microseconds. The miss is 100 percent relative and about 2,000 GPU cycles
absolute, against a bar of "the larger of two GPU cycles or 10 percent".

The defect is in the composition operator, not in the calibrated constants.
No choice of per-launch constants can repair it, and this study proves that
generally rather than for the two constants that happen to be installed.

## What was and was not pre-registered

| Evidence class | Members | Outcome | Scored? |
|---|---|---|---|
| Genuinely risky behavioral relations | `R4`, `R5` | 2 of 2 passed | Yes, denominator 2 |
| Exact closed-form identities | `R1`, `R3`, `R6`, `R7` | 4 of 4 held | No, fatal when violated |
| Post-specified regression check | `R2` | Held on all 20 cells | Reported separately, in no denominator |
| Projected arithmetic | `R8` | Reported below | No, retained-evidence arithmetic |
| Fatal guards | `G1` to `G5` | 5 of 5 held | No, preconditions are never a fraction |

The freeze disclosed that a four-cell probe ran during the review that
motivated this study, before the freeze was written. Those four cells are
`c-g1`, `c-g2`, `c-g4` at launch count 1 and one 28.2 microsecond cell that is
not in the final grid. `R2` is therefore a post-specified regression check and
is excluded from every denominator. The remaining 16 `R2` cells, all launch
counts above 1, the whole sub-eager region of the grid, the additive contrast
and the generalization were pre-registered.

## Physical sanity registered before precision

The freeze stated the closed form before any cell ran. The composition in
`simllm/compute/host.py` is `max(provider_duration_ps, launch_count * point)`,
so the per-kernel delta must be

```text
C1 <= 809,306 ps               ->  1,554,949 ps
809,306 < C1 <= 2,364,255 ps   ->  2,364,255 - C1
C1 > 2,364,255 ps              ->  0
```

Every one of the 32 cells matched that form exactly, in integer picoseconds.
Being inside the form is necessary, not proof of correctness; the interesting
part is where the form sends the answer.

## The measured regime is entirely inside the zero branch

| Cell | `C1` | Launch count 1 delta | Launch count 256 delta | Per-kernel delta |
|---|---:|---:|---:|---:|
| `c-below-graph` | 0.4000 us | 1,554,949 ps | 398,066,944 ps | 1,554,949 ps |
| `c-at-graph` | 0.8093 us | 1,554,949 ps | 398,066,944 ps | 1,554,949 ps |
| `c-between` | 1.5000 us | 864,255 ps | 221,249,280 ps | 864,255 ps |
| `c-at-eager` | 2.3643 us | 0 ps | 0 ps | 0 ps |
| `c-above-eager` | 4.0000 us | 0 ps | 0 ps | 0 ps |
| `c-g1` | 8.9437 us | 0 ps | 0 ps | 0 ps |
| `c-g2` | 18.7927 us | 0 ps | 0 ps | 0 ps |
| `c-g4` | 89.5560 us | 0 ps | 0 ps | 0 ps |

The per-kernel delta is independent of launch count, as `R3` predicted, so
scaling the chain does not move a cell out of the zero branch.

Every real kernel the A100 study measured sits in the last three rows.

## `R4`: the shapes disagree in direction, not by a margin

| Kernel | Measured eager period | Measured delta | Model delta |
|---|---:|---:|---:|
| `g1` | 8,943,667 ps | 1,456,667 ps | 0 ps |
| `g2` | 18,792,667 ps | 1,415,000 ps | 0 ps |
| `g4` | 89,555,999 ps | 1,506,331 ps | 0 ps |

The measured delta has a coefficient of variation of 0.0313 while the kernel
period spans a factor of 10.01. The measurement is a flat positive constant in
the kernel period. The model is an exact zero over that whole range.

This is the signature of an **additive** per-kernel term. A `max` composition
can only ever produce a delta by lifting a short kernel up to a host floor, so
its delta must shrink to zero as the kernel grows. The measurement does the
opposite: it stays put.

## `R5`: both halves of the COMP-48 bar are missed by construction

| Kernel | Relative error | Absolute miss |
|---|---:|---:|
| `g1` | 1.000 | 2,054.5 GPU cycles |
| `g2` | 1.000 | 1,995.8 GPU cycles |
| `g4` | 1.000 | 2,124.6 GPU cycles |

COMP-48 requires agreement within the larger of two GPU cycles or 10 percent.
At the 1410 MHz clock that study observed, two GPU cycles is 1,418 ps. The
observed miss is about 1,000 times that, and the relative error is exactly 1.0
because the prediction is exactly zero.

An error of exactly 1.0 is the tell. It is not a calibration that is off; it
is a term the model has no way to express.

## `R7`: the finding does not depend on the installed constants

For any nonnegative pair `g_e >= g_g`, `max(C1, g_e) - max(C1, g_g)` is
exactly 0 whenever `C1 >= g_e`. The A100 study measured its own eager
per-launch host cost at 1,629,633 ps on that allocation, and its smallest
real-kernel period is 8,943,667 ps, so `C1 >= g_e` holds in every one of its
real-kernel cells by a factor above 5.

Installing the A100 host constants under COMP-47 therefore changes nothing
here. Neither does any future recalibration. The zero is structural.

## `R6`: the shipped additive profile has the right shape

The `legacy-fixed-step` profile composes as `provider + delay` and returned an
exposed contribution of exactly 1,400,000 ps in all 32 cells, independent of
both device service and launch count. That is the shape the measurement has.

So the repository already contains a composition that could carry this term.
What it does not have is a calibrated profile allowed to use one: the
calibrated branch is `max` only, and `HostInitiationModel` accepts calibrated
constants exclusively from its two named Turing factories.

## `R8`: projected step magnitude, arithmetic on retained evidence

For a decode step of `N` launches in eager mode, the model exposes exactly
zero host contribution in the measured regime, while the retained measurement
implies about `N * 1.4` microseconds of additional device time. At 300
launches per step that is about 0.42 ms per step.

This is arithmetic on retained evidence from a void study. No step was run to
obtain it and it is not a measurement of TPOT. It is stated only to show that
the missing term is not negligible at step scale, which is why it matters that
COMP-48's bar is currently unreachable rather than merely tight.

## Fatal guards

| Guard | Claim | Outcome |
|---|---|---|
| `G1` | The `ideal` profile is exactly zero in every cell | Held |
| `G2` | Calibrated constants are 809,306 and 2,364,255 ps and reject every other GPU key | Held |
| `G3` | Integer picoseconds and a bit-identical second evaluation | Held |
| `G4` | The A100 deltas are read verbatim from the committed results file at its published digest | Held, digest `5c26035d...ab2ec65c` |
| `G5` | No shipped behavior mutated, nothing written outside this directory | Held |

## What this closes

Nothing. COMP-44, COMP-47 and COMP-48 all stay open.

What it establishes is a dependency that was not stated before: COMP-48 cannot
meet its own acceptance bar until a calibrated host profile is permitted a
non-overlappable per-launch term, which is COMP-44's shape question widened
from "which term" to "which operator". The registry text is corrected in the
same change that publishes this report.

The kernel-time determinism ruling is untouched. This study says nothing about
whether the residual is kernel service; it says only that assigning it to the
host launch path, as the standing ruling requires, needs a host composition
able to represent it.

## Reproducing

```bash
.venv/bin/python examples/host_launch_composition_v1/run_study.py
```

The study needs no GPU and no allocation. It reads only committed artifacts
and writes only `results.json` in its own directory.
