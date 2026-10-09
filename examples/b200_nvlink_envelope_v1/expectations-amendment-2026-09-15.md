# B200 NVLink envelope amendment, 2026-09-15

This amendment is frozen after the first complete stage-1 capture (rental
attempt 3 on a two-GPU B200 slice, driver 595.84, NCCL 2.27.3 through
PyTorch 2.8.0) and before the timing method it introduces is implemented or
run. The freeze of 2026-09-15 stays binding except where this file says
otherwise. The attempt-3 result is retained as evidence and is not scored as
the result of record.

## What the first capture showed

Every lane completed and the large-payload rows are physical: a 1 GiB peer
copy at 767 GB/s per direction, NCCL send and receive of 1 GiB at 660 GB/s,
a width-2 all-gather of 1 GiB at 899 GB/s algorithm bandwidth. Below 1 MiB
every row sits on a flat floor that does not move with payload: about 70
microseconds per width-2 all-reduce from 8 B through 1 MiB, 50 to 100
microseconds per peer copy, and 120 to 180 microseconds per NCCL point-to-
point exchange. The public profile the freeze compares against predicts
10.7 to 14.5 microseconds across that window, and the freeze's own bands
expected 1 to 30 microseconds for an 8-byte copy and 3 to 60 for an 8-byte
exchange. Those floors did not fail the freeze's 100-microsecond physical
sanity ceiling, but a window that is flat from 8 B to 1 MiB has no
serialization term to fit, so the refit slope in cell E5 is not finite.

The cause is the timing method, not the fabric. Each timed iteration is one
eager PyTorch call issued from Python, and the issue cost of that call (tens
of microseconds of dispatch, plus a host round trip per iteration in the
point-to-point lane's completion token) exceeds the device time of a
sub-megabyte transfer, so the CUDA events bracket the issue rate of the host,
not the completion rate of the link. The nccl-tests capture the profile was
fitted from issues its iterations back to back from a compiled loop and does
not pay that cost.

## Timing method of record

For every row at or below 1 MiB, in all three lanes, the row of record is
measured by CUDA graph replay: the timed iterations of one point are captured
once into a CUDA graph (after the warmup iterations, on a side stream, with
the NCCL communicator already initialized) and the graph is replayed under
the same CUDA event bracket, so no Python dispatch sits between iterations.
The eager measurement of the same rows is retained beside it as a control,
recorded and disclosed but not scored. Rows above 1 MiB keep the eager method
of the freeze, since dispatch is small against their device time, and both
methods are recorded there too so the crossover is visible. The timed
iteration count for rows at or below 1 MiB rises from 20 to 200 to reduce the
per-row noise the first capture showed; warmup stays at 5.

The NCCL point-to-point lane replaces its host-side completion token with a
captured exchange: rank 0 sends the payload and receives an 8-byte reply,
rank 1 receives the payload and sends the reply, both inside the graph, so
the timed block still measures completed transfers without a host round trip.

## Cells affected

- E2, E3, E5 and E6 read the graph-replay rows for every point at or below
  1 MiB and the eager rows above it. Their bands, holdouts, tolerances and
  fit windows are unchanged.
- E4 reports the before error from the graph-replay rows and, beside it, the
  same rows from the eager control, so the size of the dispatch floor is on
  the record.
- The scorer must report a non-positive or non-finite fitted slope as a
  degenerate fit that fails the affected holdout rather than stop; that is a
  harness rule, not a change of any band.
- E1 applies to both methods; a graph-replay row above a ceiling is as fatal
  as an eager one.
- The small-payload bands of E2 and E3 (1 to 30 and 3 to 60 microseconds at
  8 B) apply to the graph-replay rows. They stay recorded, not scored.

## What this amendment does not change

The substrate, the stages and their caps, the budget guard, the payload set,
the nameplate ceilings, the refit form, the holdout rows, the new profile's
name and the closure rule are as frozen. The attempt-3 eager capture is kept
under the measurements directory as `stage1_eager_attempt3_result.json` so
the dispatch floor it documents is reproducible from the tracked file.
