# B200 NVLink envelope expectations

Date: 2026-09-15

This is the expectations-only freeze for the first slice of TRAF-31. It
precedes the benchmark lane, the rental capture, the scorer, every measurement
file and every result-producing run. Nothing below was measured on a B200 by
this project before the freeze; the only B200 numbers the project holds are
the public issue-333 all-reduce rows that fitted `b200-nccl-2.27-local-v1` and
the GPU-side inventory of the rented eight-GPU board captured for PLACE-6.

## Question

Does a first-party B200 NVLink point-to-point and collective capture, taken
inside a rented container, validate the endpoint serializer and the width
intercepts of `b200-nccl-2.27-local-v1` within the TRAF-31 acceptance band,
and if not, does a refit from the same rows meet that band on held-out
payloads while every accepted artifact stays byte identical?

## Registry motivation

TRAF-31 asks for the same-generation point-to-point payload capture that the
public-fitted profile lacks: pinned B200 NVLink point-to-point completion
across the profile's 8-byte to 256-KiB window and beyond the payload where
bus bandwidth flattens, representative peer placements on the eight-GPU node,
at least one held-out payload, the current profile's before error, and a rerun
of the collective holdouts after any refit, with held-out completion error no
larger than 10 percent or 1 microsecond, whichever is larger. The A100
envelope's warning applies: a slope fitted inside the latency-dominated regime
is not a fabric bandwidth, so the asymptote is fitted separately over large
payloads.

## Frozen substrate and rental envelope

The substrate is a verified vast.ai host with NVIDIA B200 GPUs (PCI device
`0x10de:0x2901`, `sm` 100), running the provider's container from the image
`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` (PyTorch 2.8.0, CUDA 12.8,
NCCL 2.27.3 as `torch.cuda.nccl.version()` reports it). No compiler is
required: every timed operation is a PyTorch device copy or a
`torch.distributed` NCCL call, and timing uses CUDA events on the issuing
stream. Bandwidth is decimal (1 GB/s is 1e9 bytes per second); payload sizes
are binary.

Two stages, each a separate rental with its own wall-clock cap enforced by the
job-local rental script, which destroys the instance on exit:

| Stage | Board | Cap | Purpose |
|---|---|---|---|
| 1 | two B200 GPUs on one board with NVLink between them (`NV18` in `nvidia-smi topo -m`) | 20 minutes | pinned pair, both directions, full payload sweep; width-2 collectives; inventory and RDMA evidence |
| 2 | eight B200 GPUs on one board | 20 minutes | all ordered pairs, disjoint pairs, fan-out and fan-in, widths 4 and 8 |

Stage 2 runs only if an eight-GPU offer is listed while the study is open and
the combined rental cost of both stages stays under 30 US dollars; otherwise
the result records stage 2 as not run and the widths it would have measured
stay unsupported by the new profile. The budget is a process guard on the
rental script, not an evidence class. A stage whose precheck finds inactive
NVLinks (`nvidia-smi nvlink -s` reporting every link inactive, or a pair
without `NV` in the topology matrix) is aborted before any timed lane and
costs the boot time only.

## Pre-freeze facts

From the PLACE-6 capture of a rented eight-B200 board (`tests/fixtures/
nccl_topology/vastai_hgx_b200_8x/`): eighteen active NVLinks per GPU at
53.125 GB/s signalling, `NV18` on all 56 ordered pairs, two GPUs behind each
of four PCIe switches, NVSwitches hidden from the container, NCCL 2.27.3
through PyTorch 2.8.0 with a socket network only and no GPU Direct RDMA.

From `b200-nccl-2.27-local-v1` (`simllm/traffic/collective_latency.py`):
endpoint serializer 70,027,079,100 bytes per second, width intercepts
10,722,112 ps (2), 15,745,167 ps (4) and 30,128,029 ps (8), source payload
window 8 B to 262,144 B, bands (10,461,112, 10,983,112), (15,398,167,
16,092,167) and (30,048,029, 30,208,029) ps. Its predictions for a width-2
all-reduce of per-rank payload `S` (endpoint bytes `E = S` at width 2) are
frozen here so the before error is computed against literals:

| S (bytes) | Predicted service (ps) |
|---:|---:|
| 8 | 10,722,227 |
| 1,024 | 10,736,735 |
| 4,096 | 10,780,604 |
| 65,536 | 11,657,979 |
| 262,144 | 14,465,579 |

At width 4 the predictions are 15,745,339, 15,767,102, 15,832,905,
17,148,967 and 21,360,367 ps for the same five payloads (endpoint bytes
`1.5 S`), and at width 8 they are 30,128,229, 30,153,620, 30,230,390,
31,765,796 and 36,679,095 ps (endpoint bytes `1.75 S`).

## Nameplate constants and derived floors

- NVLink 5, eighteen lanes at 400 Gbit/s payload each: 900 GB/s per GPU per
  direction, the one-direction ceiling for any single pair and for any fan-in
  or fan-out at one GPU; 1,800 GB/s for one pair driven in both directions.
- A 1 GiB copy therefore cannot complete in less than 1,193,047 ns; a 1 GiB
  fan-in of seven donors into one GPU cannot complete in less than
  8,351,326 ns.
- No timed copy or collective can complete in less than 1 microsecond; a
  reported time below that is a harness defect.
- Bus bandwidth uses the nccl-tests factors: `2(n-1)/n` for all-reduce,
  `(n-1)/n` for all-gather and reduce-scatter, applied to `S / t`. No bus
  bandwidth may exceed 900 GB/s.

## Lanes

### Lane P1, peer copies

Device-to-device copies between a pinned pair, issued as PyTorch
`copy_` between tensors on the two devices with peer access verified by
`torch.cuda.can_device_access_peer` in both directions before timing. Payload
sizes `S` are 8 B and every power of two from 1 KiB to 1 GiB, 22 points. Five
warmup and twenty timed iterations per point, ten above 64 MiB; the timed
block is bracketed by CUDA events on the issuing stream and the reported time
is the block time divided by the iteration count. Directions 0 to 1 and 1 to
0 are timed separately; the bidirectional cell issues both directions on two
streams and reports the wall time of the slower stream. Stage 2 repeats the
unidirectional 64 KiB, 16 MiB and 1 GiB points on all 56 ordered pairs, then
at 64 MiB times four disjoint simultaneous pairs, a fan-out from GPU 0 to all
seven peers, and fan-ins of one through seven donors into GPU 0.

### Lane P2, NCCL point to point

Two processes, one per GPU, `torch.distributed` with the NCCL backend. Each
point sends `S` bytes from rank 0 to rank 1 with `isend` and `irecv` (the
same sweep and iteration rule as P1), timed on rank 0's stream from issue to
completion of the matching receive acknowledgement pattern: rank 1 returns an
8-byte token after each receive so the timed block measures completed
transfers. The bidirectional cell issues both directions with
`batch_isend_irecv`. Stage 2 repeats the 64 KiB, 16 MiB and 1 GiB points on
the ordered pairs (0, 1), (0, 7), (3, 4) and (7, 0).

### Lane P3, NCCL collectives

One process per GPU, one communicator per width. Widths are 2 in stage 1 and
2, 4 and 8 in stage 2. Operations are all-reduce, all-gather and
reduce-scatter, 32-bit float, sum. `S` follows the nccl-tests convention: the
per-rank buffer for all-reduce, the gathered output for all-gather, the
scattered input for reduce-scatter. Sizes are 8 B and every power of two from
1 KiB to 1 GiB, with the all-gather and reduce-scatter points that do not
divide evenly by the width skipped and recorded as skipped. Iterations as in
P1. Every rank times the block on its own stream and the reported time is the
maximum across ranks. Correctness is checked once per point on the all-reduce
result (every element equals the sum of rank indices plus the width).

### Evidence capture

Before any timed lane the container records the inventory
`examples/nccl_topology_capture_v1/capture_container.sh` records, plus the
RDMA evidence PLACE-6's NIC clause needs: `ls /sys/class/infiniband`,
`ibdev2netdev` and `ibv_devinfo` when present, `NCCL_DEBUG=INFO` NET lines,
and `nvidia-smi topo -m` including NIC columns. These files are evidence for
the fixture and for PLACE-6; they do not change this study's scoring.

## Frozen cells

Cell E1, physical ceilings, fatal: no P1 or P2 unidirectional rate exceeds
900 GB/s; no bidirectional aggregate exceeds 1,800 GB/s; no P3 bus bandwidth
exceeds 900 GB/s; no fan-in aggregate exceeds 900 GB/s into GPU 0; no timed
row is below 1 microsecond; every all-reduce correctness check passes.

Cell E2, peer copy asymptote: for each direction, an ordinary least squares
fit of `t = alpha + S / beta` over the points from 1 MiB to 1 GiB excluding
the 64 MiB holdout, with `R^2` at or above 0.99. The fitted `beta` is
expected between 500 and 900 GB/s and the 64 MiB holdout prediction within
10 percent of its observation. The small-payload copy time at 8 B is
expected between 1 and 30 microseconds and is recorded, not scored.

Cell E3, NCCL point-to-point envelope: the same fit for P2 unidirectional
rows, `beta` expected between 400 and 900 GB/s, 64 MiB holdout within 10
percent; the 8 B completion time expected between 3 and 60 microseconds,
recorded. The ratio of the P2 to the P1 asymptote is reported.

Cell E4, the current profile's before error: for the width-2 all-reduce rows
at 8 B, 1 KiB, 4 KiB, 64 KiB and 256 KiB, the observed completion minus the
frozen prediction above, as a signed time and as a fraction of the observed
value. The profile is validated at width 2 if every row is within the larger
of 10 percent and 1 microsecond. With stage 2, the same rows at widths 4 and
8 against their frozen predictions.

Cell E5, refit and holdout: from the width-2 all-reduce rows 8 B through
256 KiB excluding 4 KiB, fit one intercept and one slope by ordinary least
squares (the floor study's form, `t = intercept + E / bandwidth` with
`E = S` at width 2). The 4 KiB row is the holdout and its error must be
within the larger of 10 percent and 1 microsecond for the refit to be
eligible. With stage 2, widths 4 and 8 add their intercepts under one shared
slope, each with its own 4 KiB holdout, and `E = 2(W-1)S/W`. Whether E4
validates or E5 refits, the outcome is registered as a new named profile
`b200-nccl-2.27-local-firstparty-v1` carrying only the widths measured,
refusing every other width, with a provenance record naming this study, the
marketplace and dates, the driver and NCCL versions, and the per-width band
from the fit residuals. The existing `b200-nccl-2.27-local-v1` constants are
not modified.

Cell E6, the serializer question: the E5 slope (the profile's serializer as
refitted inside the latency-dominated window), the collective asymptote from
an ordinary least squares fit of the width-2 all-reduce over 1 MiB to 1 GiB,
the payload at which the all-reduce bus bandwidth first reaches 90 percent of
that asymptote, and the P1 asymptote from E2 are reported side by side. The
frozen expectation, from the A100 envelope's finding, is that the window
slope is below one quarter of the large-payload asymptote; the cell is
structural and unscored.

Cell E7, stage 2 placement structure, only if stage 2 runs: over the 56
ordered pairs at 16 MiB, the maximum unidirectional time is within 15 percent
of the minimum (every pair crosses the same switch fabric on an eight-GPU
board); four disjoint simultaneous pairs at 64 MiB each complete within 15
percent of the isolated pair time; the seven-donor fan-in at 64 MiB completes
in no less than seven times the isolated 64 MiB copy time divided by the
fraction `beta_pair / 900 GB/s`, since one receiver's lanes are the bottleneck.

Cell E8, wire and identity guards, fatal: the five vLLM reference manifest
digests of the PLACE-13 freeze, the tracked results of
`examples/collective_latency_floor_v1` (its `--check`), and every existing
test pass unchanged; resolving `b200-nccl-2.27-local-v1` by name returns the
same constants as before; the new profile is reachable only by its own name.

## Physical sanity before observation

The 8-byte copy floor of 1 microsecond and the 1 GiB floor of 1,193,047 ns
bracket every P1 row. A width-2 all-reduce of 256 KiB moves 256 KiB per
endpoint direction and cannot complete below 291 ns of pure serialization;
the public profile predicts 14.47 microseconds for it, so any first-party
observation under 5 microseconds at that payload, or above 100 microseconds,
is a harness or environment defect to be explained before scoring.

## Evidence accounting and closure

The scored denominator is the set of holdout rows: the 64 MiB holdout of E2
in each direction (2 rows), the 64 MiB holdout of E3 (1 row), and the 4 KiB
holdout of E5 per measured width (1 row in stage 1, 3 rows with stage 2).
E1 and E8 are fatal, unscored guards; E4 is the reported before error and
decides between validation and refit rather than scoring; E6 and E7 are
structural. Counts in different classes are never added. A violated fatal
guard voids the run and TRAF-31 stays open.

This slice closes when the stage-1 lanes have run, every scored holdout is
within its band, the new profile is registered with its provenance and the
widths it measured, the before error is reported, and the collective
holdouts of the floor study still reproduce. TRAF-31 narrows to the widths
stage 2 did not measure and to the switch-side observations a container
cannot make; it does not close with stage 1 alone. The result records the
rental cost of each stage from the provider's credit ledger.

## What this study does not claim

No switch arbitration, no NIC or cross-node path, no NVSwitch counters, no
protocol identification (TRAF-54, TRAF-94), no timing under compute
contention, and no claim that a rented container matches a bare-metal DGX
B200: the provenance names the marketplace, the image and the driver, and the
profile is scoped to them.
