# TRAF-73 NV4 long-flow incast validation freeze

## Expectations-only status

This record freezes the hardware matrix, simulator predictions, physical
bounds, acceptance band and miss attribution before the first TRAF-73
hardware cell. It contains no TRAF-73 hardware observation or result.
A miss remains a published model finding and never widens this freeze.

## Physical mechanism and long-flow choice

Each sender writes one long byte stream into GPU 0. Sender launch writes
are issued sequentially over PCIe, so their starts cannot be truly
simultaneous at nanosecond scale. The corrected TRAF-70 persistent
peer-write producer is reused unchanged. Its accepted one-source 1 MiB
completion was 416.768014 us. Scaling
that duration linearly gives conservative 104.192004 us and 208.384007 us
transfer times for 256 KiB and 512 KiB. The frozen budget is 5 us for
each later sender, or at most 10 us at degree 3. The worst ratios are
10 / 104.192004 = 9.598 percent and 10 / 208.384007 = 4.799 percent.
Both are below the frozen 10 percent negligibility ceiling. The scored
run recomputes the ratio from its own minimum per-flow completion; a
larger ratio is fatal and voids the comparison.

The physical floor is the larger of one flow's wire bytes divided by
the 100 GB/s ordered-pair raw capacity and all receiver wire bytes divided
by the measured 207.101921876 GB/s RX ingress plateau. One millisecond is
the conservative per-flow ceiling inherited from the accepted producer
scale. A point outside these bounds is a defect finding before precision
is discussed.

## Frozen simulation predictions

The simulator uses the scored three-module NVLink domain, simultaneous
release at 0 ps, the measured TX and RX endpoint plateaus, all declared
candidate internals unchanged, and the structural pass-through switch.

| Degree | Flow size | Per-flow completion us by source | Aggregate GB/s | Physical floor us | Binding parameter |
|---:|---:|---|---:|---:|---|
| 1 | 256 KiB | 2.791670 | 93.902216 | 2.785280 | `tx_egress_plateau` |
| 2 | 256 KiB | 2.794478, 2.795792 | 187.527541 | 2.785280 | `tx_egress_plateau` |
| 3 | 256 KiB | 4.044860, 4.046174, 4.047488 | 194.301255 | 4.034652 | `rx_ingress_plateau` |
| 1 | 512 KiB | 5.576950 | 94.009808 | 5.570560 | `tx_egress_plateau` |
| 2 | 512 KiB | 5.579758, 5.581072 | 187.880751 | 5.570560 | `tx_egress_plateau` |
| 3 | 512 KiB | 8.081468, 8.082782, 8.084096 | 194.562756 | 8.069303 | `rx_ingress_plateau` |

For every per-flow completion and aggregate goodput, signed relative
error is `(simulation - hardware) / hardware`. The frozen acceptance
band is [-0.15, +0.15]. Ten percentage points cover the maximum allowed
launch-skew fraction and five cover guarded endpoint repeatability. A cell
passes only when its aggregate and every source median are inside the band
and every fatal guard passes.

## Frozen attribution of a miss

A topology or pass-through identity failure names the pass-through switch
identity and voids the run. Otherwise, a size-dependent miss that shrinks
by more than five percentage points at 512 KiB names packetization. A
size-independent additive completion residual within 1 us names the credit
round. Remaining degree-3 misses name the RX ingress plateau; remaining
degree-1 or degree-2 misses name the TX egress plateau. These rules are
applied in that order and are not edited after hardware is observed.

## Scope and preservation

Only degrees 1, 2 and 3 have a hardware arm, and only for these long
flows. Degrees 4, 8 and 16 remain DECLARED SIMULATION with no hardware
counterpart on this node class. Agreement at degrees 1 to 3 supports but
does not prove that higher-degree extrapolation. True-sync small-flow
incast remains a model prediction.

All 59 frozen artifacts are
locked. They cover TRAF-69, TRAF-70 and TRAF-72 plus the scored profile
and runtime source. They must remain byte-identical. Raw hardware evidence
stays outside Git and this study publishes its own compact records.

## Evidence classes

Run configuration, model predictions, measured hardware rows, behavioral
comparisons, structural invariants and fatal guards remain separate. A
fatal failure makes the result void; it is never counted as a lost point.
