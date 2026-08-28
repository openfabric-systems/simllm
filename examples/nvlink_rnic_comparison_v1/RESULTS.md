# TRAF-71 NVLink credit versus rnic-nn on one physical mapping

## Outcome

What came out: `PASS_WITH_HONEST_MISSES`. rnic-nn is tighter in
8/21 rung-degree cells, NVLink is tighter in
11/21 and 2/21 tie. At 64 KiB and above,
rnic-nn is tighter in 5/9 cells. Frozen nonfatal misses: E3, E5.

What ran: the byte-identical seven-rung, three-degree, nine-seed staggered FCT
ladder from the merged NVLink study ran once through its scored three-module
credit domain and once through pinned htsim rnic-nn. Both arms received the
same release tuples and the frozen NVLink physical mapping.

All 16 fatal guards passed. The regenerated NVLink
sample and CDF projections exactly match the merged raw SHA-256 values. The
authority is expectations commit `6224d90fea2eed788b8e6ba876787fe7f0e52319`
with SHA-256 `4b60365d8251b5fd3c7627dbe38c66ad1fc1c096b21fdfada4fc744320a5bdfa`.

What it changes: TRAF-71 closes with a direct algorithm comparison and explicit
dispersion evidence.

What it does not change: the merged flow-dynamics study and its scored artifacts
remain byte-identical. This is a scored-profile simulation, not new hardware
evidence. The zero-fit homogeneous rnic-nn mapping limitation remains explicit.

## Per-rung dispersion comparison

Each cell reports `NVLink dispersion, rnic-nn dispersion; tighter transport by
absolute percentage-point difference`. Dispersion is the cross-seed p50 band
width divided by the median seed p50. Lower is tighter.

| Rung | Degree 1 | Degree 2 | Degree 3 |
|---:|---|---|---|
| 256 B | NV 0.000%, RN 0.000%; tie | NV 0.000%, RN 0.000%; tie | NV 8.143%, RN 37.762%; NVLink by 29.618 pp |
| 1 KiB | NV 115.956%, RN 138.977%; NVLink by 23.021 pp | NV 133.267%, RN 246.307%; NVLink by 113.040 pp | NV 16.983%, RN 39.129%; NVLink by 22.147 pp |
| 4 KiB | NV 62.436%, RN 63.287%; NVLink by 0.852 pp | NV 70.860%, RN 74.854%; NVLink by 3.993 pp | NV 31.995%, RN 49.563%; NVLink by 17.568 pp |
| 16 KiB | NV 11.803%, RN 11.423%; rnic by 0.380 pp | NV 6.316%, RN 6.122%; rnic by 0.194 pp | NV 6.805%, RN 5.253%; rnic by 1.552 pp |
| 64 KiB | NV 1.904%, RN 1.867%; rnic by 0.037 pp | NV 2.648%, RN 2.669%; NVLink by 0.021 pp | NV 1.593%, RN 1.806%; NVLink by 0.213 pp |
| 256 KiB | NV 0.609%, RN 0.668%; NVLink by 0.059 pp | NV 0.655%, RN 0.621%; rnic by 0.033 pp | NV 0.481%, RN 0.420%; rnic by 0.061 pp |
| 512 KiB | NV 0.419%, RN 0.404%; rnic by 0.015 pp | NV 0.414%, RN 0.419%; NVLink by 0.005 pp | NV 0.191%, RN 0.186%; rnic by 0.005 pp |

## Mechanism diagnosis

- Credit window: positive reconstructed credit wait appeared in
  0/21 NVLink cells, covering 0 packets and
  0 ps in aggregate. This is the direct test for credit-window
  stalls. The first credit returns after 10.880 + 200 = 210.880 ns, while one
  256-packet bonded-link credit round spans 696.320 ns, so credits recycle
  before exhaustion. rnic-nn has no credit or congestion window.
- Pacing: the pinned rnic-nn arm is a central progressive max-min allocator
  feeding deterministic full-packet slots. It emitted DATA events only, with
  zero ACK events and zero reverse bytes. Any smoothness is max-min slot pacing,
  not ACK pacing.
- Packetization: both arms use 256 payload plus 16 header bytes, exactly
  5.882353 percent header at a full packet. The 256 B intercept is therefore
  serializer composition and slot phase: 12.194 ns for NVLink versus 5.440,
  2.720 and 2.628 ns for mapped rnic-nn degrees 1, 2 and 3.
- Incast-3 arbitration: positive reconstructed RX admission wait appeared in
  14/21 NVLink cells and covered 1,733,130 packets. NVLink uses
  release-aware per-source packet round robin followed by stable tied-arrival
  order at RX; rnic-nn uses deterministic max-min grants and packet slots.
- Degree-3-left-of-degree-1 oddity: NVLink reproduced it on
  1 KiB, 4 KiB, 16 KiB, 64 KiB, 256 KiB, 512 KiB; rnic-nn
  reproduced it on 1 KiB.
  The common 1 KiB sign is assigned to the staggered release pattern.
  The NVLink-only 4 KiB, 16 KiB, 64 KiB, 256 KiB, 512 KiB signs
  remain transport effects; with zero credit wait, the named mechanisms
  are release-aware packet round robin and stable RX admission order,
  not credit-window stalls.
- CDF roughness: each seed contains only 12, 24 or 36 flows at degree 1, 2 or
  3, so a single observation moves its empirical CDF by 8.333, 4.167 or 2.778
  percentage points. At 1 KiB the median-relative seed widths are 16.983 to
  246.307 percent; at 512 KiB they fall to 0.186 to 0.419 percent. The visible
  small-rung steps and bands are finite-sample and stagger-alignment effects,
  not evidence of credit-window exhaustion.

The homogeneous rnic-nn adapter accepts one endpoint capacity. Its
degree-specific mapping is exact at full incast membership, but a temporarily
single active sender can exceed one 100 GB/s ordered-pair cap. That declared
limitation can only move rnic-nn FCT left and reduce its apparent dispersion;
it is not counted as an algorithm win.

## FCT location by rung and degree

The signed shift is rnic-nn minus NVLink. Negative is left of NVLink.

| Rung | Degree | NVLink mean seed p50 | rnic-nn mean seed p50 | Signed shift |
|---:|---:|---:|---:|---:|
| 256 B | 1 | 0.012194 us | 0.005440 us | -0.006754 us |
| 256 B | 2 | 0.012194 us | 0.002720 us | -0.009474 us |
| 256 B | 3 | 0.012381 us | 0.002801 us | -0.009580 us |
| 1 KiB | 1 | 0.023717 us | 0.019505 us | -0.004212 us |
| 1 KiB | 2 | 0.024938 us | 0.014809 us | -0.010129 us |
| 1 KiB | 3 | 0.022114 us | 0.012403 us | -0.009711 us |
| 4 KiB | 1 | 0.127262 us | 0.123344 us | -0.003918 us |
| 4 KiB | 2 | 0.135003 us | 0.125617 us | -0.009385 us |
| 4 KiB | 3 | 0.103118 us | 0.128853 us | +0.025735 us |
| 16 KiB | 1 | 0.606642 us | 0.602932 us | -0.003710 us |
| 16 KiB | 2 | 0.617702 us | 0.607613 us | -0.010089 us |
| 16 KiB | 3 | 0.522291 us | 0.840140 us | +0.317848 us |
| 64 KiB | 1 | 2.582308 us | 2.578859 us | -0.003449 us |
| 64 KiB | 2 | 2.589703 us | 2.581119 us | -0.008584 us |
| 64 KiB | 3 | 2.237491 us | 3.690433 us | +1.452943 us |
| 256 KiB | 1 | 10.404909 us | 10.403099 us | -0.001810 us |
| 256 KiB | 2 | 10.414850 us | 10.404711 us | -0.010139 us |
| 256 KiB | 3 | 9.044894 us | 15.043407 us | +5.998513 us |
| 512 KiB | 1 | 20.862777 us | 20.858694 us | -0.004084 us |
| 512 KiB | 2 | 20.876967 us | 20.869802 us | -0.007165 us |
| 512 KiB | 3 | 18.145397 us | 30.203976 us | +12.058579 us |

## Frozen expected directions

| Freeze ID | Passed | Required | Verdict |
|---|---:|---:|---|
| E1 | 3/3 | 3 | PASS |
| E2 | 6/6 | 6 | PASS |
| E3 | 5/9 | 7 | REFUTED |
| E4 | 7/7 | 7 | PASS |
| E5 | 1/6 | 4 | REFUTED |
| E6 | 1/1 | 1 | PASS |

Honest misses remain published without changing a threshold or mapping.

## Figures

- [`figures/nvlink-rnic-fct-cdf.pdf`](figures/nvlink-rnic-fct-cdf.pdf)
- [`figures/nvlink-rnic-dispersion.pdf`](figures/nvlink-rnic-dispersion.pdf)

Every PDF has a matching PNG. The final PNGs were inspected at publication
size for clipping, overlap, readable log ticks, visible min-max bands, legend
crossings and border contact. Compact numeric dispersion evidence is in
[`dispersion.csv`](dispersion.csv).
