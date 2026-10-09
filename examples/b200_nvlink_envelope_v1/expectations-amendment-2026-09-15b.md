# B200 NVLink envelope amendment b, 2026-09-15

This second amendment is frozen after stage-1 rental attempt 5 (the first
capture with graph-replay rows in every lane) and before the eager copy path
it describes is run for the record. The freeze of 2026-09-15 and its first
amendment stay binding except where this file says otherwise.

## What attempt 5 showed

Every lane produced graph-replay rows with no capture skipped: the peer copy
at 8 B reads 6.8 microseconds in both directions, the NCCL exchange 23.7
microseconds, and the width-2 all-reduce refit over the frozen window gives a
9.13 microsecond intercept with a 69.4 GB/s slope, its 4 KiB holdout within
0.66 microseconds. The reverse copy and the NCCL exchange fit their large-
payload asymptotes at 779 and 697 GB/s with holdout errors of 2.6 and 2.1
percent. The forward eager copy rows above 1 MiB, however, read a flat 2.2
milliseconds at every payload, while the same direction's graph rows are
physical. Attempt 3, which captured nothing, had a clean forward sweep, so
the eager row was contaminated by the capture that preceded it inside the
cell, and only in the direction whose destination device also hosts the
waiting peer rank.

## Eager copy method

For the eager rows of lane P1 the order inside a cell is now eager first,
then capture, with both devices synchronized before the first timing event
and fresh streams and events per method. The copy is issued with the source
and destination streams both entered so the readiness event PyTorch records
on the destination device lands on the study's stream, not the device
default stream. The timing bracket is recorded on the destination stream:
a start event, the copies, a completion event on the source stream that the
destination stream waits on, and a stop event, so the interval covers the
transfers and nothing the destination device does on its own. CUDA refuses
to measure between events on different devices, which is why both timing
events sit on the destination; the graph-replay rows keep the source-stream
bracket of the freeze, because a replayed graph rejoins its destination
stream internally. Every row records `timed_on` as `destination` or
`source` so the two brackets are distinguishable in the result.

This is the one place where "bracketed by CUDA events on the issuing stream"
in the freeze no longer describes the eager cross-device copy literally; the
graph rows, lanes P2 and P3, and every band, holdout and tolerance are
unchanged.

## Precheck

The stage script's NVLink precheck now requires every visible GPU to report
eighteen active links and every pair the stage uses to read `NV18` in the
topology matrix; it names the offending GPUs and exits 3 before timing. The
eight-GPU host rented as stage-2 attempt 1 had one GPU with every link
inactive and was refused by the benchmark's own peer-access check at no
timed cost; that host is excluded from further rentals and the stage
records the refusal as evidence.

## If the forward eager rows stay flat

If the rerun still shows the forward eager rows above 1 MiB flat near 2
milliseconds while the graph rows are physical, the remaining explanation is
the peer rank spinning in an NCCL barrier on the destination device during
lane P1. In that case the eager control for that direction is recorded as
not meaningful and the direction is reported as unscored with the reason;
substituting the other direction's eager rows into its fit is not permitted.
A third amendment would then move the inter-lane barriers to a CPU-backed
group before any further run. No such change is made by this amendment.
