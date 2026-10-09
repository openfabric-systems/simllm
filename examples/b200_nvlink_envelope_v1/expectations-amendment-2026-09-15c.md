# B200 NVLink envelope amendment c, 2026-09-15

This third amendment is frozen after the first complete stage-2 capture (an
eight-GPU B200 board, machine 150403, every lane completed) and before any
further run. The freeze of 2026-09-15 and amendments a and b stay binding
except where this file says otherwise.

## What the stage-2 capture showed

Every lane produced graph-replay rows with no capture skipped, and the six
scored holdout rows of both stages are inside their bands: the refit now
carries widths 2, 4 and 8 under one shared slope, and each width's 4 KiB
holdout lands within its tolerance. The public profile is refuted at all
three widths, low by up to 13.4 microseconds at width 8.

The placement cells of E7 read two different boards depending on the timing
method. The graph-replay rows are physical everywhere: over the 56 ordered
pairs at 16 MiB the slowest pair is within 5 percent of the fastest, the 1
GiB rows agree to 0.2 percent across all pairs, four disjoint simultaneous
pairs at 64 MiB complete within 7 percent of the isolated pair, the fan-out
from one source to seven peers completes at 715 GB/s aggregate, and the
seven-donor fan-in completes at 791 GB/s aggregate into one receiver. The
eager rows above 1 MiB carry the artifact amendment b anticipated, now with
its mechanism confirmed: seven idle ranks spin in an NCCL barrier on their
own devices while rank 0 drives lane P1, and every eager copy whose
destination hosts a spinning rank reads a flat 2.33 milliseconds regardless
of payload, at 16 MiB and 64 MiB as at 1 GiB, which is why E7 evaluated on
eager rows reported an 83-fold spread and a 22-fold disjoint-pair slowdown.
Where an eager row is clean, it agrees with the graph row within 1.2
percent (1 GiB: 1,390 to 1,396 microseconds eager against 1,379 to 1,381
graph).

The freeze's fan-in rule is also wrong as written. It required the
seven-donor fan-in to take no less than seven isolated 64 MiB copy times
divided by the pair fraction `beta_pair / 900 GB/s`, which comes to 636
microseconds on this board; the fan-in completed in 594 microseconds. That
is not a violation of physics: seven donors together can drive a receiver's
lanes closer to the 900 GB/s nameplate than one source can (791 against 781
GB/s), and the nameplate floor for 7 times 64 MiB into one receiver is 522
microseconds, which the observation respects. The rule inverted the
fraction.

## Rows of record for the placement cells

For the stage-2 placement cells (the 56 ordered pairs, the disjoint pairs,
the fan-out and the fan-ins) the rows of record are the graph-replay rows at
every payload, not only at or below 1 MiB. The eager rows are retained as
the control and reported beside them, with every row whose destination
device hosted a spinning rank marked not meaningful. E7 is evaluated on the
graph rows: the all-pairs spread and disjoint-pair slowdown bands are
unchanged, and the fan-in rule becomes: the seven-donor fan-in completes in
no less than the nameplate floor of `7 * 64 MiB / 900 GB/s` and its
aggregate receive rate is reported as a fraction of the nameplate. The
pinned-pair rows of E2 keep the freeze's selection (graph at or below 1 MiB,
eager above), because in stage 1 the reverse direction's eager rows were
clean and the forward direction's rows of record were clean from 4 KiB up;
stage 1's scored outcomes do not move.

## Inter-lane barriers

Any further run holds its inter-lane and inter-cell barriers in a CPU-backed
process group (the `gloo` backend) so that idle ranks wait on the host and
never spin on a device another rank is timing. The NCCL group is used only
inside the timed lanes P2 and P3. One confirming rerun of stage 2 under this
rule is permitted within the budget; its eager rows are expected to agree
with the graph rows within 5 percent above 1 MiB, and that agreement is
reported as a structural check, not scored. If the rerun's graph rows move
any E5 intercept by more than its band, the wider band is recorded and both
runs are retained.

## What this amendment does not change

The scored denominator (two E2 holdouts, one E3 holdout, one E5 holdout per
measured width), the bands, the fit windows, the payload set, the nameplate
ceilings, the refit form, the profile name and the closure rule. The new
profile carries the widths measured, now 2, 4 and 8, each refusing nothing
it measured and refusing every width it did not.
