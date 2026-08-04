# rnic-cn incast ladder: results

Runs of 2026-08-03 against the algorithm-book implementation (htsim branch
`wip/cn-fractional-nflow`, commit 2548562: no leases, no waits, windowed
k to k+2 snapshots, ppm-native fractional declarations, own-fraction
scaling, deterministic reservation ledger). 98 cells (fan-in 1, 2, 4, 8,
16, 32, 63 crossed with 16 MiB down to 4096 B, at K = 10 us and 4.5 us),
each against the rnic-nn baseline on the identical GOAL. Raw data:
`/data3/yifeng/simllm-dev/cn-ladder/ladder.csv`.

## Invariants (frozen disqualifiers): all hold

- Zero Ring-CAM late admissions and zero gap NACKs in every cell,
  including startup, at the design D. The per-dwnd sent <= reserved
  invariant held everywhere.
- Full determinism: repeat runs bit-identical in every checked cell.
- The previously fatal lease race is structurally gone (no lease exists);
  K = 4.5 us and K = 10 us behave identically up to the fractional budget
  threshold, as registered.

## Steady state: exact

Bulk incast at 16 MiB: median slowdown 1.124 / 1.106 / 1.109 at fan-in
1 / 8 / 63, against the registered prediction 1/0.9 = 1.111. The
allocation, windowing and scaling machinery is quantitatively correct at
every fan-in.

## The additive constant, and the maintainer's BJP ruling

As registered, small-size cells failed the raw 2x bar through one flat,
size- and K-independent additive. Parameter isolation attributed it
entirely to the resequencing window D: with D = 64 ns a single 4 KiB flow
measures slowdown 1.072. Shrinking D under load, however, produces
recovery events (16,650 at D = 1.024 us in the 63 x 16 MiB cell): the
system correctly refuses a violated jitter bound. D is now DERIVED from
the topology (book section on the bandwidth jitter product): upstream
FIFO depths are bounded by (S_max - 1) wire packets at the bottleneck
egress plus ceil(S_max / paths) per intermediate stage, giving
D = 6.5728 us on the reference Clos; the driver computes it whenever the
flag is not supplied, and the manifest reports the derived value.

Maintainer ruling (recorded in the algorithm book, section on the
bandwidth jitter product): D is not a tuning knob. It is the BJP constant
Q_upstream / C (upstream FIFO depth over bandwidth, generation-invariant
per switch family); the PIFO is sized at or above the upstream queue;
early arrivals waiting longest in the resequencer is the deterministic
release contract working. Judging cn against a baseline that carries no
resequencing discipline therefore treats D as a known constant offset:
the re-registered per-flow metric is cn_fct / (nn_fct + D). This is a
lens re-registration by the design authority, disclosed here in full, not
a silent goalpost move; the raw-lens numbers stay in the CSV.

## Re-judged grid (derived D = 6.5728 us, BJP lens, K = 10 us)

- 46 of 49 cells inside the 20 percent target (median <= 1.2); the
  1.2 to 2.0 band is empty.
- 3 cells fail the 2x bar, all in the sub-packet-share corner:
  in32-s4096 (max 2.39), in63-s16384 (2.01), in63-s4096 (3.35). In these
  cells a flow's per-window fair share is smaller than one 4160 B packet,
  whole-packet launches ride the deficit carry across several windows,
  and the uncoordinated carry completion order spreads the tail. This is
  the rotating discrete slot calendar already specified in the algorithm
  book's mechanism backlog; the corner is isolated, bounded, and has a
  designed fix.
- Zero recovery events and full determinism in all 98 cells at derived D.

## Phase 2: mixed lognormal all-to-all (16 ranks, mean 256 KiB)

The mixed pattern did its job and exposed the next mechanism gap
deterministically (repeats bit-identical):

- K = 4.5 us: zero recovery events; median D-adjusted behavior congested
  but lossless.
- K = 10 us: 335 late admissions and 335 gap NACKs, an invariant
  violation. The inverted K-dependence is the fingerprint: the fractional
  budget B0 grows with K, so at K = 10 most flows declare fractionally,
  and the failure is the known corner of multiple per-receiver ledgers
  contending at one sender egress. Each receiver's members sum to
  margin * C correctly, but a sender holding 15 flows granted by 15
  independent receiver ledgers can be granted more than its own port,
  launches late, and lands packets beyond ETA + D. The incast ladder
  cannot see this (single receiver); the all-to-all finds it immediately.
  Mechanism definition pending (sender-egress calendar composed with the
  per-receiver ledgers; all inputs are deterministic self-knowledge).

## Verdict

Ladder phase: the algorithm meets the maintainer's bar everywhere except
three specified sub-packet corner cells. Mixed phase: blocked on the
sender-egress composition mechanism, now precisely characterized. Both
gaps live in the algorithm book's backlog with designed fixes.

## Plots

`plots/a2a16_flow_size_pdf.png`: the mixed workload's flow-size PDF
(lognormal, mean 256 KiB, 240 flows). `plots/a2a16_normalized_fct_cdf.png`:
per-flow FCT CDFs of rnic-nn-fluid / rnic-nn / rnic-cn / dcqcn on the
identical GOAL and 64-node 400G Clos, normalized to rnic-nn-fluid.
Medians: nn 1.025, dcqcn 1.336, cn 2.062 (p99 18.3). The cn tail bloat
under this heavily oversubscribed all-to-all is the accepted price of the
deliberate RTT rebalancer (maintainer ruling): cn stays lossless and
deterministic with zero recovery events while dcqcn trades losslessness
and determinism for shorter congested FCTs. Removing destination
oversubscription by construction is the recorded open design point.

## Comparator realism round (maintainer correction)

The first CDF was too kind to DCQCN: default 33.5 MiB buffers meant zero
drops, two pauses and zero RTOs, so it never paid its own costs. With
mlx5-faithful loss recovery landed in the backend (go-back-N default with
the CNP-equivalent rate cut on every recovery event; optional limited
selective repeat with GBN fallback modeling ConnectX-6 Dx tracking;
ECN-only mode with drops at overflow; PFC storm metrics with cascade
depth), the seeded study at the 1 MiB lossy operating point
(`plots/a2a16_lossy_fct_cdf_seeded.png`, 8 seeds per DCQCN mode) shows the
true comparison: DCQCN median 1.52 but p99 at 1902x and worst-seed max
3528x (RTO-dominated tail; about 12 percent of flows strand beyond 100x),
against rnic-cn's median 2.06 with p99 19.3x and max 24.9x, lossless and
deterministic. ECN plus PFC at these thresholds barely engages (up to 7
pauses, cascade depth 3) and does not rescue the tail. Two harness bugs of
this study (a silently failed buffer edit and mismatched counter keys)
were caught by cross-checking against the backend acceptance numbers and
are fixed in run_dcqcn_modes.py.
