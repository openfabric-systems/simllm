# B200 NVLink envelope result, stage 1

## Outcome

What ran: `examples/b200_nvlink_envelope_v1`, the first TRAF-31 slice, timed
peer copies, NCCL point-to-point and NCCL collectives between two NVIDIA B200
GPUs joined by `NV18`, inside a rented container, and scored the result
against the frozen cells. Expectations-only commit `4c695ea0`, amendment a
`1217af41` (CUDA graph replay becomes the timing method of record at or below
1 MiB), amendment b `82686615` (the eager cross-device copy is measured before
the capture in its cell and bracketed on the destination device). Stage 1 ran
six times; attempt 6 is the result of record and the earlier attempts are
retained as evidence, with every rental named in
[the measurement provenance](measurements/PROVENANCE.md).

What came out: the run is not void, every fatal ceiling held, and all four
scored holdout rows are inside their bands. The deciding number is the width-2
refit: an intercept of 9,088,984 ps over a shared slope of 68,888,931,479
bytes per second, whose held-out 4 KiB row misses by 0.615 microseconds
against a 1 microsecond tolerance. The public profile under test is refuted at
this width: four of its five before-error rows are outside the larger of 10
percent and 1 microsecond, all four low, by up to 2.569 microseconds, so the
study refits rather than validates.

What it changes for the project: TRAF-31's same-generation point-to-point
capture exists and is first party, and the refit is registered as
`b200-nccl-2.27-local-firstparty-v1`, carrying width 2 only and refusing every
other width. `b200-nccl-2.27-local-v1` keeps its constants and its name, and
nothing that resolves it changes. PLACE-6's NIC clause gains a recorded
negative: this container exposed no usable RDMA device on either host.

What it does not change: no width other than 2 is measured, so the collective
floor study and everything downstream of widths 4 and 8 still rest on the
public fit; no switch-side fact is observed; no cross-node path, no NVSwitch
counter, no protocol identification and no timing under compute contention are
touched. TRAF-31 stays open, narrowed to those. This is a rented container on
a marketplace host, not a bare-metal DGX B200, and the profile's provenance
says so.

## Amendment chronology

The freeze of 2026-09-15 (`4c695ea0`) preceded the benchmark lane, the capture
script, the scorer and every run. Amendment a (`1217af41`) was frozen after the
first complete capture, attempt 3, which showed a flat sub-megabyte floor in
all three lanes, and before any graph-replay row existed. Amendment b
(`82686615`) was frozen after attempt 5 and before attempt 6, the only run of
record.

One ordering is disclosed rather than claimed as clean: amendment b was
committed one commit after `c3dd0333`, the harness change it describes, not
before it. No run of record had happened when either landed, and attempt 6 is
the first and only run under that method, so the amendment still precedes its
evidence; it does not precede its own implementation. Amendment a has no such
gap.

## Evidence

| Evidence class | Result |
|---|---|
| Scored holdout rows (E2 in each direction, E3, E5 at width 2) | 4 of 4 inside their bands |
| Fatal physical ceilings (E1) | every ceiling held; peak unidirectional 777.53 GB/s, peak bidirectional aggregate 1,545.19 GB/s, peak bus bandwidth 572.40 GB/s, fastest timed row 6,218 ns, every all-reduce correctness probe exact |
| Fatal identity guards (E8) | run separately, see Reproduction: the five PLACE-13 reference manifest digests, the collective floor study's own check, the public profile by name, and the full test suite |
| Reported, decides refit or validation (E4) | 4 of 5 rows outside the tolerance, all low; the profile is not validated at width 2 |
| Structural, unscored (E6) | the window slope is 0.1199 of the large-payload asymptote, below the frozen quarter |
| Structural, only with stage 2 (E7) | not evaluated, stage 2 did not run |

Counts in different evidence classes are never added. The tracked
[result](measurements/stage1_result.json) and
[scored report](measurements/scored.json) hold every row.

## The current profile's before error, width 2

Observed rows are the graph-replay rows of record; the eager control of the
same payload is shown beside them, because the gap between the two columns is
the host dispatch cost that amendment a removed from the record.

| Payload | Observed | Public prediction | Delta | Delta over observed | Eager control |
|---:|---:|---:|---:|---:|---:|
| 8 B | 8.153 us | 10.722 us | -2.569 us | -31.5 percent | 18.0 us |
| 1 KiB | 8.253 us | 10.737 us | -2.484 us | -30.1 percent | 19.1 us |
| 4 KiB | 8.534 us | 10.781 us | -2.247 us | -26.3 percent | 18.4 us |
| 64 KiB | 11.698 us | 11.658 us | +0.040 us | +0.3 percent | 18.1 us |
| 256 KiB | 11.965 us | 14.466 us | -2.501 us | -20.9 percent | 18.9 us |

The public profile is slower than this board everywhere except at 64 KiB,
where the two cross. Its source is a public eight-GPU capture whose width-2
intercept it carries; this board's width-2 intercept is 1.6 microseconds
lower.

## The refit

Ordinary least squares over the width-2 all-reduce rows from 8 B to 256 KiB,
excluding the 4 KiB holdout, in the floor study's form
`t = intercept + E / bandwidth` with `E = S` at width 2.

| Constant | Value |
|---|---|
| Intercept, width 2 | 9,088,984 ps |
| Shared slope | 68,888,931,479 bytes per second |
| Fit rows | 9 |
| R squared | 0.624 |
| Band, width 2 | 8,152,843 to 10,746,592 ps |
| Holdout, 4 KiB | predicted 9,148,443 ps against 8,533,600 ps observed, error 0.615 us, allowed 1.000 us |

The band is the inclusive minimum and maximum of the intercept plus the fit
residuals, which run from -0.936 to +1.658 microseconds; the holdout error is
smaller than both extremes and does not widen it. The large-payload cells sit
beside the refit: the peer copy asymptote is 780.54 and 781.66 GB/s per
direction with R squared 0.99977 and 0.99974 and holdout errors of 1.24 and
3.03 percent, and the NCCL point-to-point asymptote is 697.50 GB/s with
R squared 0.99589 and a holdout error of 2.16 percent. The width-2 all-reduce
asymptote is 574.66 GB/s, and bus bandwidth first reaches 90 percent of it at
512 MiB.

## Physical sanity

Eighteen lanes at 400 Gbit/s payload give 900 GB/s per GPU per direction, so a
1 GiB copy cannot complete below 1,193,047 ns; the measured graph-replay copy
is 1,381,000 ns in both directions, 15.8 percent above the floor, and the
eager rows agree at 1,396,000 and 1,395,000 ns. No unidirectional rate reaches
the ceiling, the bidirectional aggregate of 1,545 GB/s stays under the 1,800
GB/s pair ceiling, and the fastest timed row in the run is 6.2 microseconds,
far above the 1 microsecond harness floor.

Two numbers need stating rather than defending. The width-2 all-gather reports
903.8 and 905.8 GB/s of algorithm bandwidth at 1 GiB, above the 900 GB/s
per-direction ceiling, which is correct arithmetic and not a violation: the
nccl-tests convention divides the gathered output by the time while only half
of it crosses the link, so the physically bounded quantity is the 452 GB/s bus
bandwidth, and that is what cell E1 guards. The width-2 all-reduce of 256 KiB
moves 256 KiB per endpoint direction and cannot complete below 291 ns of pure
serialization; it completes in 11.965 microseconds, inside the freeze's window
of 5 to 100 microseconds, so no explanation is owed.

## Choices the freeze left open

- **Where the eager copy is timed.** The freeze says the timed block is
  bracketed by CUDA events on the issuing stream. For a cross-device copy
  PyTorch issues on the source stream, but CUDA refuses to measure between
  events on two devices, and the source-side bracket left the forward
  direction reading a flat 2.2 milliseconds at every payload while the
  captured rows of the same cell were physical. Amendment b moved the eager
  bracket to the destination device around an explicit completion event; the
  graph rows keep the source-stream bracket. Every row records `timed_on`, so
  the two are distinguishable.
- **What R squared of 0.624 means here.** Over 8 B to 256 KiB the serialization
  term the fit is trying to find is under 4 microseconds against a 9
  microsecond intercept, so most of the variance in the window is not payload
  dependent and no fit of this form can explain it. The number is not evidence
  that the constants are wrong: the held-out row lands within 0.615
  microseconds, and an independent rental one attempt earlier returned
  9,126,016 ps and 69,422,515,624 bytes per second, which is 0.4 and 0.8
  percent from the record. It is evidence that the intercept, not the slope,
  is what this window identifies, which is the same thing the A100 envelope
  warned about and what cell E6 reports.
- **One untimed replay per row.** The first replay of a fresh graph pays its
  instantiation, and in attempt 5 that landed entirely in the 8 B
  point-to-point row, which read 132 microseconds against 32 for its
  neighbours. Every captured row now replays once untimed before the timed
  replay; the same row reads 23.64 microseconds in the record.
- **One row, one block.** Each row is a single timed block with no repetition
  and no median over samples, as frozen. A host hiccup therefore lands whole
  in one row, which is visible in the residuals: the 64 KiB row sits 1.658
  microseconds above the fit and is the largest single contributor to the R
  squared above.
- **Trailing charges.** The rental cost below is the credit ledger's reading,
  which includes storage charges that accrue after an instance is destroyed,
  so the per-attempt figures carry about a dollar of tail across the day that
  cannot be attributed to a single rental.

## What the container did not show

No NVSwitch: no PCI device of class `0x0680` is visible, the same limit the
eight-GPU topology fixture records, so nothing here observes the switch side.
No RDMA: both hosts list InfiniBand class devices, `ibv_devinfo` reports no
device, `ibdev2netdev` is absent from the image, and NCCL logs `NET/IB : No
device found` before using the socket network, so no GPU Direct RDMA path was
available. NCCL also selected no NVLS at width 2 on this board, reporting
`0 nvls channels` against 32 collective channels, and the image carries no
tuner plugin, so the internal tuner chose every algorithm. The protocol NCCL
selected per call is not on the record: the debug subsystems the freeze names
are `INIT` and `NET`, which do not print it.

## Stage 2

Stage 2 did not run. One eight-GPU board was rented and refused before any
timed lane: GPU 4 reported every NVLink inactive and `SYS` rather than `NV18`
on all fourteen of its ordered pairs, so the benchmark's peer-access check
aborted at boot cost only. Its evidence is under
[the refusal directory](measurements/stage2_attempt1_refusal/), and the stage
script's precheck now refuses the same board itself rather than leaving it to
the benchmark. No healthy eight-GPU offer was listed while the study was open.

Stage 2 would add the widths 4 and 8 intercepts under the same shared slope,
each with its own 4 KiB holdout, the all-pairs placement spread at 16 MiB, the
four disjoint pairs and the fan-in and fan-out cells. Without it the new
profile carries width 2 alone, cell E7 is unevaluated, and TRAF-31 keeps the
eight-GPU placements it asked for.

## Reproduction

Inside the provider's container, from a checkout of the repository, with the
output directory given as an absolute path so the inventory script can cd into
it:

```bash
bash examples/b200_nvlink_envelope_v1/capture_stage.sh <output directory> 1
bash examples/b200_nvlink_envelope_v1/capture_stage.sh <output directory> 2
```

The script runs the topology inventory, the RDMA evidence and the NVLink
precheck, then launches the lanes under `torch.distributed.run` with one
process per visible GPU, and ends `listing.txt` with `STAGE_DONE`. Scoring
runs on any host, needs no GPU and imports no torch:

```bash
python examples/b200_nvlink_envelope_v1/score_expectations.py
```

The identity guards of cell E8 are run separately:
`pytest tests/test_b200_nvlink_envelope_profile.py` covers the five reference
manifest digests and the public profile by name,
`python examples/collective_latency_floor_v1/run_study.py --check-only` with
the htsim binary and its run directory reproduces the floor study's frozen
calibration, and the full `pytest -q` covers the rest.

Rental cost, from the provider's credit ledger: 15.11 US dollars across the
six stage 1 attempts, 1.27 for the refused stage 2 board, and about 1 dollar
of trailing storage charges.
