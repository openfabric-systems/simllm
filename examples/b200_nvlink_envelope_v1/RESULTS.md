# B200 NVLink envelope result

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

What came out: neither stage is void, every fatal ceiling held, and all six
scored holdout rows are inside their bands. The deciding numbers are the
refit's, over both stages: one shared slope of 75,888,107,438 bytes per second
with intercepts of 9,236,136, 12,822,835 and 23,087,092 ps at widths 2, 4 and
8, whose held-out 4 KiB rows miss by 0.638, 0.097 and 0.674 microseconds. The
public profile under test is refuted at every width, always low, by up to
3.487 microseconds at width 2, 5.443 at width 4 and 13.179 at width 8, so the
study refits rather than validates.

What it changes for the project: TRAF-31's same-generation point-to-point
capture exists and is first party, and the refit is registered as
`b200-nccl-2.27-local-firstparty-v1`, carrying widths 2, 4 and 8 and refusing
every other width. `b200-nccl-2.27-local-v1` keeps its constants and its name, and
nothing that resolves it changes. PLACE-6's NIC clause gains a recorded
negative: this container exposed no usable RDMA device on either host.

What it does not change: no switch-side fact is observed, because no NVSwitch
is visible to a container; no cross-node path, no NVSwitch counter, no per-call
protocol identification and no timing under compute contention are touched; and
nothing here is bare metal. TRAF-31 stays open, narrowed to those. This is a rented container on
a marketplace host, not a bare-metal DGX B200, and the profile's provenance
says so.

## Amendment chronology

The freeze of 2026-09-15 (`4c695ea0`) preceded the benchmark lane, the capture
script, the scorer and every run. Amendment a (`1217af41`) was frozen after the
first complete capture, attempt 3, which showed a flat sub-megabyte floor in
all three lanes, and before any graph-replay row existed. Amendment b
(`82686615`) was frozen after stage 1 attempt 5 and before stage 1 attempt 6.
Amendment c (`d855ec55`) was frozen after the first complete stage 2 capture,
which measured widths 4 and 8 but timed its eager rows against seven spinning
peers, and before the stage 2 rerun that is the result of record.

One ordering is disclosed rather than claimed as clean: amendment b was
committed one commit after `c3dd0333`, the harness change it describes, not
before it. No run of record had happened when either landed, and attempt 6 is
the first and only run under that method, so the amendment still precedes its
evidence; it does not precede its own implementation. Amendment a has no such
gap.

## Evidence

| Evidence class | Result |
|---|---|
| Scored holdout rows (E2 in each direction, E3, E5 at each measured width) | 6 of 6 inside their bands |
| Fatal physical ceilings (E1) | every ceiling held; peak unidirectional 777.53 GB/s, peak bidirectional aggregate 1,545.19 GB/s, peak bus bandwidth 572.40 GB/s, fastest timed row 6,218 ns, every all-reduce correctness probe exact |
| Fatal identity guards (E8) | run separately, see Reproduction: the five PLACE-13 reference manifest digests, the collective floor study's own check, the public profile by name, and the full test suite |
| Reported, decides refit or validation (E4) | at width 2, 4 of 5 rows outside the tolerance; at widths 4 and 8, 5 of 5; always low, so the profile is validated at no measured width |
| Structural, unscored (E6) | the window slope is 0.1319 of the large-payload asymptote, below the frozen quarter |
| Structural, stage 2 placement (E7) | the three banded cells hold on the rows of record; the eager agreement check fails as frozen, for the reason below |

Counts in different evidence classes are never added. The tracked
[result](measurements/stage1_result.json) and
[scored report](measurements/scored.json) hold every row, and the
[provenance](measurements/PROVENANCE.md) registers the two earlier attempts
kept beside them.

## The current profile's before error

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

Stage 2 measures the same five payloads at widths 4 and 8, from the board of
record.

| Payload | Width 4 observed | Predicted | Delta | Width 8 observed | Predicted | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 8 B | 10.302 us | 15.745 us | -5.443 us | 16.949 us | 30.128 us | -13.179 us |
| 1 KiB | 12.603 us | 15.767 us | -3.164 us | 23.025 us | 30.154 us | -7.128 us |
| 4 KiB | 13.000 us | 15.833 us | -2.832 us | 23.856 us | 30.230 us | -6.375 us |
| 64 KiB | 14.912 us | 17.149 us | -2.237 us | 26.274 us | 31.766 us | -5.492 us |
| 256 KiB | 17.306 us | 21.360 us | -4.054 us | 27.922 us | 36.679 us | -8.757 us |

Every row at widths 4 and 8 is outside the tolerance and every one is low, so
the profile is validated at no measured width. The public profile is slower
than these boards everywhere except the width-2 64 KiB row, where the two
cross, and the gap widens with width: its source is a public eight-GPU capture
whose intercepts it carries unchanged.

## The refit

Ordinary least squares over the width-2 all-reduce rows from 8 B to 256 KiB,
excluding the 4 KiB holdout, in the floor study's form
`t = intercept + E / bandwidth` with `E = S` at width 2.

| Constant | Value |
|---|---|
| Shared slope | 75,888,107,438 bytes per second |
| Intercept, width 2 | 9,236,136 ps, band 7,235,575 to 10,901,213 |
| Intercept, width 4 | 12,822,835 ps, band 10,301,762 to 14,075,670 |
| Intercept, width 8 | 23,087,092 ps, band 16,949,256 to 24,957,971 |
| Fit rows | 27, nine per width |
| R squared | 0.937 |
| Holdout, width 2 | predicted 9,290,111 ps against 8,651,680 observed, error 0.638 us, allowed 1.000 us |
| Holdout, width 4 | predicted 12,903,797 ps against 13,191,839 observed, error 0.097 us, allowed 1.319 us |
| Holdout, width 8 | predicted 23,181,547 ps against 23,956,480 observed, error 0.674 us, allowed 2.396 us |

The first stage 2 run, on the same board with its idle ranks spinning, returns
74,361,308,462 bytes per second with intercepts of 9,185,311, 12,788,419 and
22,780,276 ps. The intercepts agree with the record within 1.35 percent and
the slope within 2.05 percent, which is the cross-rental reproduction; the
record is the run whose control rows are also usable.

Each band is the inclusive minimum and maximum of that width's intercept plus
its fit residuals, widened to the holdout error where that reaches further.
The large-payload cells of stage 1 sit beside the refit: the peer copy asymptote is 780.54 and 781.66 GB/s per
direction with R squared 0.99977 and 0.99974 and holdout errors of 1.24 and
3.03 percent, and the NCCL point-to-point asymptote is 697.50 GB/s with
R squared 0.99589 and a holdout error of 2.16 percent. The width-2 all-reduce
asymptote is 575.47 GB/s over the whole 1 MiB to 1 GiB window, and bus
bandwidth first reaches 90 percent of it at 512 MiB.

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
- **What the fit's R squared means here.** Over 8 B to 256 KiB the
  serialization term the fit is trying to find is a few microseconds against
  intercepts of 9 to 23, so much of the variance in the window is not payload
  dependent and no fit of this form can explain it. Across three widths the
  two-stage fit reaches 0.937, because the widths separate the intercepts;
  the width-2 fit of stage 1 alone reached 0.624. Neither number is evidence
  that the constants are wrong: every held-out row lands well inside its
  tolerance, and three independent rentals reproduce the constants. Stage 1
  attempt 5, tracked as
  [the attempt 5 result](measurements/stage1_graph_attempt5_result.json),
  returned 9,126,016 ps and 69,422,515,624 bytes per second; the first stage 2
  run, tracked beside the record, returned intercepts within 1.35 percent and
  a slope within 2.05 percent of it. What this window identifies well is the
  intercept, not the slope, which is what cell E6 reports and what the A100
  envelope warned about. It is evidence that the intercept, not the slope,
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
- **Part of the eager control is not meaningful, and is marked so.** Amendment
  b's clause for a forward eager sweep that stays flat applies to four rows of
  the record. Lane P1's eager forward rows at 8 B, 1 KiB and 2 KiB read 2,332
  microseconds each, and the entire eager bidirectional sweep is flat between
  2,331 and 2,342 microseconds at every payload including 1 GiB, where the
  captured row of the same cell reads 1,390 microseconds. The remaining eager
  rows are physical: the forward direction from 4 KiB up, the reverse direction
  throughout, and both at 1 GiB within 1.1 percent of their captured rows. The
  likely mechanism is the one amendment b named: during lane P1 the peer rank
  waits in an NCCL barrier that spins on the destination device
  (`bench_nvlink.py`, the barrier after the lane), so an eager iteration that
  records its readiness event on that device interleaves with it, and a graph
  replay does not because it makes no per-iteration call there. Those rows are
  recorded as not meaningful and are not substituted by the other direction.
  Nothing scored reads them: below 1 MiB the row of record is the captured row,
  the E2 fit window starts at 1 MiB, and cell E1's peak bidirectional aggregate
  comes from the captured row, so the flat rows only understate a ceiling they
  cannot breach.
- **The 900 GB/s ceiling assumes this host's link rate.** Cell E1's ceilings
  come from eighteen lanes at 400 Gbit/s of payload, which is the NVLink 5
  nameplate, and both two-GPU hosts signalled 53.125 GB/s per link. The
  eight-GPU board refused for stage 2 signalled 50 GB/s, so a board whose links
  run slower would be judged against a ceiling it cannot reach, and the ceiling
  would stop being a tight guard. Link rate is recorded per host in the
  provenance for that reason, and a stage 2 result from a 50 GB/s board would
  need its own ceiling rather than this one.
- **The correctness check of the tracked run read three elements.** The freeze
  asks for one correctness check per all-reduce point; the run of record
  compared the first, middle and last element of the result against the
  expected sum, which cannot see a reduction that is wrong only in between. The
  lane now compares every element, and the scorer now treats a row with no
  correctness result as a violation rather than silence, but the tracked result
  predates both, so its correctness evidence is the three-element form.
- **The tracked header names one amendment.** The result header of the record
  carries `amendment` naming amendment a only, because the field predates
  amendment b; the lane now writes an `amendments` list. The method the run
  actually used is amendment b's, which the `timed_on` field on every row
  shows directly.
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
tuner plugin, so the internal tuner chose every algorithm. The capture sets the NCCL debug
subsystems to `INIT`, `NET` and `TUNING`, so the log carries the tuner's
algorithm and protocol bandwidth table for this board, listing Tree, Ring,
CollNet, NVLS, NVLSTree and PAT against LL, LL128 and Simple. What it does not
carry is the algorithm and protocol NCCL chose for each individual call, which
needs a finer debug level than the freeze asks for, so no per-call protocol
claim is made here.

## Stage 2

Stage 2 ran on the third rental. The first eight-GPU board was refused before
any timed lane, because GPU 4 reported every NVLink inactive and `SYS` rather
than `NV18` on all fourteen of its ordered pairs; its evidence is under
[the refusal directory](measurements/stage2_attempt1_refusal/). The second and
third rentals are the same board, machine 150403, on the same offer; they
differ only in where the idle ranks waited, which is the whole story of this
stage.

The placement cells, on the rows of record:

| Cell | Result | Band |
|---|---|---|
| 56 ordered pairs at 16 MiB | slowest 5.77 percent above the fastest | within 15 percent |
| Four disjoint pairs at 64 MiB | 1.0757 times the isolated pair | at most 1.15 |
| Seven-donor fan-in at 64 MiB | 0.594 ms, receiving 790.3 GB/s, 87.8 percent of nameplate | at or above the 0.522 ms nameplate floor |
| Fan-out to seven peers at 64 MiB | 0.657 ms at 715.4 GB/s aggregate, 79.5 percent of nameplate | reported, no band |

### The spinning peer, and why the fan-in was clean

In the second rental every eager copy whose destination device hosted a
waiting rank read a flat 2.33 milliseconds whatever the payload, at 16 MiB as
at 1 GiB, while the captured rows of the same cells were physical. The cause
is that lane P1 runs on rank 0 across every device while the other seven ranks
wait in an NCCL barrier, and an NCCL barrier spins on its own device. The
asymmetry is what makes this more than a plausible story: the fan-*in* rows
were clean at every donor count, because their destination is GPU 0, the
device of the rank that is driving the lane and therefore not spinning, while
the fan-*out* and disjoint rows, whose destinations are the waiting ranks'
devices, sat on the 2.33 ms floor. Amendment c moved the inter-lane and
inter-cell barriers into a CPU-backed group, and in the rerun no eager row is
left on that floor: the scorer marks zero rows not meaningful, against 73 of
177 in the run before it.

### The fan-in rule was wrong as frozen

The freeze required the seven-donor fan-in to take no less than seven isolated
64 MiB copy times divided by `beta_pair / 900 GB/s`, which comes to 636
microseconds on this board against an observation of 594. That is not a
physical violation, it is an inverted fraction: seven donors together drive the
receiver's lanes closer to the nameplate than one source can, 790.3 against
781 GB/s, and the nameplate floor for seven 64 MiB transfers into one receiver
is 522 microseconds, which the observation respects with 14 percent to spare.
Amendment c replaces the rule with that floor and asks for the aggregate
receive rate as a fraction of the nameplate.

### The eager agreement check fails as frozen

Amendment c expected the two timing methods to agree within 5 percent above
1 MiB once the idle ranks were off the devices. They do not: 77 of 172
compared rows are inside 5 percent. The criterion, not the harness, is what
fails here, and the offsets say why.

| Payload | Rows | Median eager minus graph | Median gap | Largest gap |
|---:|---:|---:|---:|---:|
| 2 MiB | 5 | 26.6 us | 377.9 percent | 600.5 percent |
| 4 MiB | 5 | 25.0 us | 248.1 percent | 429.9 percent |
| 8 MiB | 5 | 18.5 us | 120.4 percent | 239.8 percent |
| 16 MiB | 61 | 12.4 us | 44.8 percent | 121.3 percent |
| 32 MiB | 5 | 9.2 us | 17.9 percent | 24.9 percent |
| 64 MiB | 14 | 10.2 us | 9.5 percent | 41.7 percent |
| 128 MiB | 5 | 10.8 us | 6.0 percent | 7.1 percent |
| 256 MiB | 5 | 12.0 us | 3.4 percent | 5.2 percent |
| 512 MiB | 5 | 13.3 us | 1.9 percent | 2.2 percent |
| 1 GiB | 62 | 14.7 us | 1.1 percent | 68.6 percent |

The offset is roughly constant between 9 and 27 microseconds across three
decades of payload, which is the per-iteration host dispatch that graph replay
removes and nothing else; expressed as a fraction it is 378 percent of a 7
microsecond captured row at 2 MiB and 1 percent of a 1.4 millisecond one at
1 GiB. A fractional criterion therefore cannot hold across this band, and 5
percent is met from 256 MiB up. The outcome is recorded as a failed
structural, unscored check rather than reinterpreted, because the criterion
was frozen before the rerun; what it establishes is that the eager control is
usable as a control at large payloads and not at small ones, which is the same
conclusion amendment a reached for the sub-megabyte window by a different
route.

Two residues are disclosed rather than tidied. Nine rows were excluded from
the comparison as not meaningful, all of them stage 1's contaminated
bidirectional rows; the same cell's 1 GiB row is not excluded, because at that
payload the artifact no longer doubles the captured time and the marking rule
is a row-level test, so one known-bad row contributes the 68.6 percent maximum
in the last line of the table. And the largest gap in the record itself is a
bidirectional 2 MiB row at 66.3 microseconds eager against 9.5 captured, which
is the dispatch offset on the smallest captured time in the band.

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
six stage 1 attempts and 5.74 across the three stage 2 attempts (1.27 for the
refused board, 2.65 for the first capture, 1.82 for the rerun of record). The
study's total against the ledger is 32.34 dollars, which also carries the
day's earlier smoke tests and the storage charges that accrue after an
instance is destroyed.
