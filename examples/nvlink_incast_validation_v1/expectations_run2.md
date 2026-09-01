# TRAF-74 NV4 long-flow incast second-capture freeze

## Expectations-only status

This record freezes the second hardware matrix, simulator predictions,
physical bounds, acceptance band and miss attribution before the second
TRAF-74 hardware cell. It contains no second-capture hardware observation
or result. The retained first result remains byte-identical and void.

## Physical mechanism and long-flow choice

Each sender writes one long byte stream into GPU 0. The unchanged TRAF-70
producer starts sender work through sequential PCIe launch writes, so the
starts cannot be simultaneous at nanosecond scale. The first capture showed
that 256 KiB to 512 KiB flows still measured launch overhead: per-source
apparent goodput ranged from about 2.2 to 3.5 GB/s against 94.117647 GB/s
of packetized wire payload. The second capture therefore uses 4 MiB and
8 MiB flows so observed completion is on the millisecond scale.

The retained TRAF-70 one-source 1 MiB completion was 416.768014 us.
Linear byte scaling gives conservative 1667.072056 us and 3334.144112 us
for 4 MiB and 8 MiB. At degree 3, the maximum 10 us sequential launch
offset is 0.600 percent and 0.300 percent of those values. The respective
margins below the ten percent fatal ceiling are 9.400 and 9.700 percentage
points. The scored run recomputes the fraction from every observed minimum
per-source completion; a value above ten percent voids the entire run.

The physical floor is the larger of one flow's packetized wire bytes divided
by the 100 GB/s ordered-pair raw capacity and all receiver wire bytes divided
by the measured 207.101921876 GB/s RX ingress plateau. The five millisecond
ceiling rounds outward from 8 MiB divided by the retained slow 2.2260869 GB/s
apparent producer goodput, which is 3.768 milliseconds. A point outside the
frozen range is a defect finding before precision is discussed.

## Frozen launch-skew margins

| Degree | Flow size | Scaled completion us | Launch offset us | Fraction | Margin to budget |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 MiB | 1667.072056 | 0.000000 | 0.000% | 10.000 percentage points |
| 2 | 4 MiB | 1667.072056 | 5.000000 | 0.300% | 9.700 percentage points |
| 3 | 4 MiB | 1667.072056 | 10.000000 | 0.600% | 9.400 percentage points |
| 1 | 8 MiB | 3334.144112 | 0.000000 | 0.000% | 10.000 percentage points |
| 2 | 8 MiB | 3334.144112 | 5.000000 | 0.150% | 9.850 percentage points |
| 3 | 8 MiB | 3334.144112 | 10.000000 | 0.300% | 9.700 percentage points |

## Frozen simulation predictions

The simulator is exactly `simllm-htsim-nvlink-domain-v1` from base commit
`65593131a0448d2b33f51018d5972c918dad3493`. Every source releases at 0 ps. The flow policy is
explicitly `release_aware_round_robin`. The scored TX egress and RX ingress
plateaus, declared packetization and credit round, and structural
pass-through switch identity are unchanged.

| Degree | Flow size | Per-source completion us | Aggregate GB/s | Physical floor us | Binding parameter |
|---:|---:|---|---:|---:|---|
| 1 | 4 MiB | 44.570870 | 94.104154 | 44.564480 | `tx_egress_plateau` |
| 2 | 4 MiB | 44.573678, 44.574992 | 188.190903 | 44.564480 | `tx_egress_plateau` |
| 3 | 4 MiB | 64.593980, 64.595294, 64.596608 | 194.792148 | 64.554418 | `rx_ingress_plateau` |
| 1 | 8 MiB | 89.135350 | 94.110900 | 89.128960 | `tx_egress_plateau` |
| 2 | 8 MiB | 89.138158, 89.139472 | 188.213096 | 89.128960 | `tx_egress_plateau` |
| 3 | 8 MiB | 129.179708, 129.181022, 129.182336 | 194.808553 | 129.108836 | `rx_ingress_plateau` |

For every per-source completion median and aggregate receiver goodput,
signed relative error is `(simulation - hardware) / hardware`. The band is
plus or minus 16 percent. It is fixed from the retained ten percent TRAF-70
endpoint-repeatability allowance, 5.263 percent maximum per-source spread
on the first capture's 512 KiB rung, and 0.600 percent worst pre-run skew
fraction, whose 15.863 percent sum is rounded outward. A cell passes only
when its aggregate and every source median are inside the band and every
fatal guard passes.

## Frozen attribution of a miss

A topology or pass-through identity failure names the pass-through switch
identity and voids the run. Otherwise, a size-dependent miss that shrinks
by more than five percentage points at 8 MiB names packetization. A
size-independent additive completion residual within 1 us names the credit
round. Remaining degree-3 misses name the RX ingress plateau; remaining
degree-1 or degree-2 misses name the TX egress plateau. These rules are
applied in that order and are not edited after hardware is observed.

## Scope and preservation

Only degrees 1, 2 and 3 have a hardware arm, and only for these long flows.
Degrees 4, 8 and 16 remain DECLARED SIMULATION with no hardware counterpart
on this node class. Agreement at degrees 1 to 3 supports but does not prove
that higher-degree extrapolation. True-sync small-flow incast remains a
model prediction.

All 71 frozen artifacts are
locked. They cover TRAF-69, TRAF-70 and TRAF-72, the scored model and profile,
and every first-capture record. They must remain byte-identical. Raw hardware
evidence stays outside Git and the second capture publishes separate records.

## Evidence classes

Run configuration, frozen model predictions, measured hardware rows,
behavioral comparisons, structural invariants and fatal guards remain
separate. A fatal failure makes the result void; it is never counted as a
lost behavioral point.
