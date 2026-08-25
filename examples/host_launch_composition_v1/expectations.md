# Host launch composition v1 expectations

## Freeze scope and chronology

This is the expectations-only record for a model-side study of one frozen
claim in `docs/modules/compute.md`. It is committed before the harness exists
and before the full sweep runs. No number produced by this study may be
written back into this file.

The study needs no GPU, no allocation and no profiler. It exercises the
shipped `simllm.compute.host.HostInitiationModel` through its public API and
compares its predicted CUDA-graph versus eager launch-mode delta against
first-party A100 measurements that are already committed in this repository.

### Prior-probe disclosure

During the review that motivated this study, a four-cell probe was run against
the same public API before this freeze was written. The probe used launch
count 1 and per-kernel provider service of 8.9, 18.793, 28.2253 and 89.556
microseconds against the two calibrated Turing profiles, and it returned a
zero launch-mode delta in all four cells.

Relation `R2` is therefore a **post-specified regression check** for those four
cells and is reported as such. Every other cell of the grid below, every
relation other than `R2`, the uncertainty-bound behavior, the additive-
composition contrast and the generalization argument are pre-registered by
this freeze and had not been evaluated when it was written. This study never
claims public pre-registration for the probed cells.

## The claim under test

`docs/modules/compute.md` states that CUDA-graph launch and eager launch
differ only in the host launch cost, that both calibrated profiles compose as
`max(C, N * g)` over an unchanged provider service `C`, and that COMP-48
"identifies the host term without changing kernel service or adding a
device-front-end stage".

COMP-48's acceptance bar requires measured host initiation to predict the
observed launch-mode delta within the larger of two GPU cycles or 10 percent
in every supported cell.

This study asks one question. Under the composition the shipped calibrated
profiles actually use, is that bar reachable at all?

## Definitions, stated rather than implied

**Provider service `C1`.** The per-kernel duration a compute provider returns,
in integer picoseconds, before any host profile is applied. A chain of `N`
kernels has total provider service `N * C1`.

**Per-launch host point `g`.** The calibrated constant a profile carries.
`turing-cuda-graph` carries 809,306 ps and `turing-eager-host` carries
2,364,255 ps, both read from `simllm/compute/host.py`.

**Model launch-mode delta.** For a chain of `N` kernels,
`delta_model(C1, N) = D_eager - D_graph`, where `D_profile` is the
`duration_ps` field returned by `represented_estimate` for that profile on a
provider estimate of `N * C1`.

**Measured launch-mode delta.** The device-side back-to-back per-kernel period
in eager mode minus the same period under graph replay, taken verbatim from
the committed
[A100 graph launch study](../a100_graph_launch_v1/RESULTS.md). That study is
reviewed `VOID`, so these are retained evidence used as a comparison target
and never as an anchor, a fit input or a closure claim.

**Exposed host contribution.** The `exposed_ps` field, defined by the shipped
model as composed duration minus provider duration.

## Parameter sweep

Two axes vary independently, as the repository's validation discipline
requires, and a third axis selects the composition under test.

Axis 1, per-kernel provider service `C1` in picoseconds. The grid straddles
both calibrated constants so all three regimes of the composition are covered:

| Cell | `C1` (ps) | Position |
|---|---:|---|
| `c-below-graph` | 400,000 | below the graph point |
| `c-at-graph` | 809,306 | exactly the graph point |
| `c-between` | 1,500,000 | between the two points |
| `c-at-eager` | 2,364,255 | exactly the eager point |
| `c-above-eager` | 4,000,000 | above the eager point |
| `c-g1` | 8,943,667 | measured A100 `g1` eager period |
| `c-g2` | 18,792,667 | measured A100 `g2` eager period |
| `c-g4` | 89,555,999 | measured A100 `g4` eager period |

Axis 2, launch count `N`: 1, 8, 64 and 256.

Axis 3, composition: the two calibrated `max` profiles as one pair, the
`ideal` profile, and the shipped `legacy-fixed-step` additive profile.

The full calibrated grid is 8 by 4, i.e. 32 cells per profile.

## Expected relations, written before the run

`R1` (exact identity). For every cell,
`D_profile = max(N * C1, N * g_profile)` exactly, in integer picoseconds.

`R2` (exact, the refutation). For every `C1` greater than or equal to
2,364,255 and every `N`, `delta_model(C1, N)` is exactly 0, and `exposed_ps`
is exactly 0 for both calibrated profiles. This covers `c-at-eager`,
`c-above-eager`, `c-g1`, `c-g2` and `c-g4`, i.e. 20 of the 32 cells.

`R3` (exact piecewise shape). `delta_model(C1, N) / N` depends only on `C1`
and is nonincreasing in `C1`:

```text
C1 <= 809,306                 ->  1,554,949 ps
809,306 < C1 <= 2,364,255     ->  2,364,255 - C1 ps
C1 > 2,364,255                ->  0 ps
```

`R4` (shape disagreement). The three measured A100 deltas are 1,456,667 ps at
a per-kernel period of 8,943,667 ps, 1,415,000 ps at 18,792,667 ps and
1,506,331 ps at 89,555,999 ps. Their coefficient of variation is below 4
percent while the period varies by a factor above 10, so the measured delta is
flat in the kernel period. The model's delta over that same range is exactly
zero at every point. The two shapes are a positive constant against an exact
zero and therefore disagree in both direction and shape.

`R5` (bar miss, quantitative). The relative error of the max-composed
prediction against each measured delta is exactly 1, i.e. 100 percent, in all
three A100 cells. Two A100 GPU cycles at the 1410 MHz clock that study
observed is 1,418 ps, so the absolute miss of 1,415,000 to 1,506,331 ps is
above 990 cycles. Both halves of COMP-48's "larger of two GPU cycles or 10
percent" bar are missed by construction, not by a margin that better constants
could close.

`R6` (constructive contrast). The shipped `legacy-fixed-step` additive profile
with `initiation_delay_ps = d` returns `exposed_ps = d` for every `C1` and
every `N`. Its exposed contribution is a per-invocation constant independent
of device service, which is the shape the measurement has. The defect is
therefore in the composition operator and not in the calibrated constants.

`R7` (generalization, exact). For any nonnegative pair `g_e >= g_g`, the max
composition yields a per-kernel delta of exactly 0 whenever `C1 >= g_e`. The
A100 study measured its own eager per-launch host cost at 1,629,633 ps on that
allocation, and its smallest real-kernel period is 8,943,667 ps, so `C1 >= g_e`
holds in every one of its real-kernel cells. No max-composed host profile
calibrated on that allocation can predict a nonzero delta there. The finding
does not depend on the Turing constants being the ones installed today.

`R8` (projected step magnitude, arithmetic on retained evidence). For a decode
step of `N` launches, the model's exposed host contribution in the measured
regime is exactly 0 while the retained measurement implies about
`N * 1.4` microseconds of additional device time in eager mode. This is stated
as arithmetic on retained evidence, not as a measured step, and no step was
run to obtain it.

## Fatal guards

A violated fatal guard voids the run. It is never reported as a fraction.

`G1`. The `ideal` profile returns `duration_ps` equal to the provider duration
and `exposed_ps` exactly 0 in every cell.

`G2`. The two calibrated profiles carry exactly 809,306 ps and 2,364,255 ps,
and each rejects every `GpuSpec.name` except `gtx1660-ti-sm75`.

`G3`. Every returned duration is an integer picosecond, and evaluating the
whole grid twice in one process returns bit-identical results.

`G4`. No measured A100 value is recomputed, refitted or adjusted here. The
three deltas are read verbatim from the committed
`examples/a100_graph_launch_v1/measurements/results.json`, whose SHA-256 must
equal `5c26035da27c40b86149c243361c74221da8bc51f4ad8c5851018510ab2ec65c`, the
digest that study published.

`G5`. The study mutates no shipped behavior and writes no file outside its own
directory.

## Survivable fatal guards

None. Every guard above is a precondition for the run meaning what it claims,
so any violation voids the run.

## Evidence classes and denominators

| Evidence class | Members | Scored? |
|---|---|---|
| Exact closed-form identities | `R1`, `R3`, `R6`, `R7` | No. By-construction identities, fatal when violated |
| Genuinely risky behavioral relations | `R4`, `R5` | Yes. Denominator 2 |
| Post-specified regression check | `R2` | Reported separately, never added to the denominator |
| Projected arithmetic | `R8` | No. Retained-evidence arithmetic, not a measurement |
| Fatal guards | `G1` to `G5` | No. Preconditions, never a fraction |

The scored denominator is 2. The four probed `R2` cells are named in the
report and excluded from every denominator.

## What this study can and cannot close

It cannot close COMP-48. It measures no host initiation and rules on no
maintainer question.

It can establish one thing: whether COMP-48's acceptance bar is reachable
under the composition the shipped calibrated profiles use. If `R2`, `R4` and
`R5` hold, the bar is unreachable without an additive per-launch term in the
host composition, and COMP-48 needs that dependency stated before anyone
attempts its measurement campaign. If they fail, this freeze is wrong and the
frozen claim in `docs/modules/compute.md` stands as written.

Refuting a frozen assumption is a legitimate result. It is still not a closure.
