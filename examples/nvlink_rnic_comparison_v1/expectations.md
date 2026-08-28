# TRAF-71 NVLink credit domain versus rnic-nn: frozen expectations

This is the expectations-only authority for the controlled transport comparison
directed on 2026-08-28. It is committed before the study adapter, runner, any
transport execution, raw sample, measured dispersion, or result-dependent edit.
The machine-readable authority is [`expectations.json`](expectations.json).
Observed values never widen a band, change a mapping, or rewrite a direction.
An honest miss is published as a miss.

## Question and source boundary

The study asks whether the visibly rough flow-completion-time (FCT) cumulative
distribution functions (CDFs) in `nvlink_flow_dynamics_v1` come from the
staggered workload, the scored NVLink credit domain, or the comparator's packet
scheduler. Both transports receive the same seven sizes, incast degrees one
through three, nine seeds, twelve waves per sender, exact picosecond release
tuples, sources, destination and packet geometry.

The NVLink arm is the scored three-module domain exactly as the merged study ran
it. The comparator is htsim commit
`1dcbfec36a33753bf978cf6323bade1a6645fe4f`, profile `rnic-nn`, through
`RnicPacketizedManifoldRuntime`. The study builds a small adapter from an
immutable export of that commit because GOAL text quantizes releases to whole
nanoseconds while the source workload contains arbitrary picoseconds. The
adapter changes no htsim source and supplies the same exact release tuple to
both transports.

Source inspection freezes one important correction before the run. The pinned
`rnic-nn` profile is not ACK-paced. It is a topology-free central progressive
max-min allocator feeding a deterministic packet-slot calendar. Its own header
says that it has no route, queue, loss, backpressure, acknowledgement, PRBS
pacer or Ring-CAM. It therefore emits zero ACK bytes and zero reverse-control
bytes. The diagnosis must name max-min packet-slot pacing, not ACK pacing.

## One physical link and the zero-fit mapping

The scored physical values are measured TX endpoint egress at
160,795,737,454 byte/s, measured RX ingress at 207,101,921,876 byte/s, four
declared 25,000,000,000 byte/s links per ordered pair, zero explicit transit
propagation in the scored direct-link model, and packets of 256 payload bytes
plus 16 header bytes. The packet has 272 wire bytes, a 5.882352941 percent
header fraction of wire bytes and a 6.25 percent overhead relative to payload.

The pinned rnic runtime accepts one homogeneous endpoint rate. For an incast of
degree `d`, the frozen mapping is

`C_d = 8 * min(d * 100 GB/s, d * 160.795737454 GB/s, 207.101921876 GB/s)`.

No constant is fitted. The resulting runtime rates are 800,000,000,000,
1,600,000,000,000 and 1,656,815,375,008 bit/s for degrees one, two and three.
They match the full-membership physical aggregate: degree one and two are
ordered-pair limited, while degree three is RX limited.

| Physical field | Scored NVLink credit domain | Pinned rnic-nn mapping | Expected signed effect |
|---|---|---|---|
| TX endpoint | 160,795,737,454 byte/s shared by one source | Enters the degree-specific minimum; one 100 GB/s pair is lower | None at full incast membership |
| RX endpoint | 207,101,921,876 byte/s shared at destination 3 | Caps `C_d` | Degree 3 is RX limited in both arms |
| Ordered pair | Four links times 25 GB/s, or 100 GB/s raw | `d * 100 GB/s` enters `C_d` | Degrees 1 and 2 are pair limited at full membership |
| Propagation | No explicit transit term, 0 ps | `propagation_delay_ps=0` | Neither arm receives additive flight delay |
| Packet | 256 payload plus 16 header, 272 wire bytes | 272 maximum wire bytes and 16 DATA header bytes | Packet overhead is identical |
| Window | 256 credits of 272 bytes, 200,000 ps return | No credit, congestion window or backpressure | rnic-nn cannot suffer credit-window stalls |
| Reverse control | Credit return is a timestamped release, not a packet | 0 ACK and control bytes | rnic-nn is biased left relative to an ACK-carrying design |
| Arbitration | Per-source extent round robin, then stable topology order at RX | Central max-min grants and a deterministic packet-slot calendar | rnic-nn removes credit burstiness but can retain tie-order steps |

The homogeneous rnic capacity cannot simultaneously express a 100 GB/s
per-pair ceiling and a 207.101921876 GB/s shared destination when fewer than
`d` members are active. The degree-specific mapping is exact at full incast
membership, but it permits transient source service above one ordered-pair rate
when other members have not joined or have drained. That limitation can only
bias rnic-nn FCT left and dispersion downward. It must remain on the figure and
in the diagnosis. A transient over one ordered-pair rate is a mapping bias, not
an algorithm win.

## Small-rung physical arithmetic

The scored NVLink no-queue one-packet time is

`max(1,692, 10,880) + 1,314 = 12,194 ps`.

The endpoint-admission serializer overlaps the longer bonded-link serializer;
the measured RX serializer follows it. One credit round covers 256 packets,
69,632 wire bytes or 65,536 payload bytes, and returns after 200,000 ps. A
256-byte flow is one packet and a 64 KiB flow is one full credit round.

The rnic full-wire slot durations after the degree mapping are 2,720, 1,360 and
1,314 ps. A single packet crossing its source and destination serializers costs
two slots, or 5,440, 2,720 and 2,628 ps before queueing. The figure must state
these numbers because one packet is the whole 256-byte flow and fixed packet or
credit phases are a large fraction of the smallest FCTs.

## Identical workload

Sizes are 256 B, 1 KiB, 4 KiB, 16 KiB, 64 KiB, 256 KiB and 512 KiB. Degrees
are one, two and three. Destination is GPU 3. The frozen seeds are 1103, 1907,
2801, 3691, 4513, 5381, 6271, 7159 and 8053. Every seed has twelve waves per
sender.

The exact source generator is retained. It seeds Python's deterministic random
stream with `seed * 1,000,003 + degree * 10,007 + size_bytes`. Every later wave
adds the source study's cell-specific release interval plus one integer draw
from minus 10,880 through plus 10,880 ps. Every source then adds one integer
draw from 0 through 10,880 ps. `expectations.json` freezes the interval and a
SHA-256 of all nine release lists for every one of the 21 cells. The runner
must independently regenerate those lists and hand the identical tuples to
both transports.

The source study's intervals remain unchanged. They equal three quarters of
the independently derived wave service and range from 9,145 ps at the smallest
degree-one and degree-two cell to 6,060,137 ps at degree three and 512 KiB.
Neither transport receives a rescaled arrival process.

## CDF and dispersion definitions

FCT is completion time minus the exact picosecond release. For each transport,
degree and size, each seed supplies its own empirical CDF. The curve is the
pointwise arithmetic mean across nine seeds on the sorted union of observed
FCT values for that transport and cell. The shaded band is the pointwise
minimum through maximum across those nine seed CDFs. Every CDF must be
monotone and terminate at one.

Dispersion uses the same nearest-rank seed median as the source study. For each
seed, `q_s` is the nearest-rank p50. Across the nine seed medians, let `m` be
their ordinary median. The plotted dimensionless width is

`D = (max_s(q_s) - min_s(q_s)) / m`.

It is rendered as a percentage. This is an FCT width divided by an FCT, not a
vertical CDF-probability width.

## Expected directions before the run

1. At 256 bytes, rnic-nn p50 is left of NVLink at every degree because its
   mapped two-serializer costs are 5,440, 2,720 and 2,628 ps versus the
   NVLink no-queue 12,194 ps.
2. For each transport and degree, `D` at 512 KiB is below `D` at 1 KiB because
   fixed 10,880 ps release jitter occupies a smaller fraction of the longer
   FCT.
3. At 64 KiB and above, rnic-nn is no wider than NVLink in at least seven of
   the nine rung-degree cells because it has no credit-return stalls.
4. The regenerated NVLink arm reproduces the merged degree-3-left-of-degree-1
   p50 ordering at 1 KiB through 512 KiB, but not at 256 bytes. This is an
   entailed preservation check, not a scored transport prediction.
5. rnic-nn reproduces degree 3 left of degree 1 on at least four of those six
   rungs. If it misses, the oddity points to NVLink credit or RX arbitration
   rather than the shared stagger schedule.
6. The ACK-pacing claim is not applicable. The pinned rnic-nn ledger contains
   DATA only and zero reverse bytes. This source-semantics check is fatal and
   unscored.

Each miss is publishable. None authorizes a new rate, packet size, ACK size,
window or jitter.

## Mechanism decision rules

- Credit-window pressure may explain NVLink-only widening above 64 KiB only
  when the packet ledger reuses the 256-credit round and shows positive credit
  or RX admission delay.
- rnic-nn smoothness is attributed to central max-min packet-slot pacing. It is
  never attributed to ACK pacing.
- Both arms pay exactly 16 header bytes per 256 payload bytes. Small-rung
  intercept differences therefore come from serializer composition and slot
  phase, not unequal packet overhead.
- If both arms retain degree 3 left of degree 1, the shared release pattern gets
  the sign. If only NVLink retains it, credit and stable RX arbitration get the
  difference. If neither retains it, the report says that reproduction failed
  before interpreting the sign.
- Any rnic-nn transient above one ordered-pair raw rate is labeled as the
  homogeneous-capacity mapping bias.

## Figure contract

`nvlink-rnic-fct-cdf` contains seven size panels. Every panel overlays both
transports and all three degrees, uses a logarithmic FCT axis, and shades a
separate pointwise nine-seed min-max band for every curve. Degree owns color;
transport owns solid versus dashed line style.

`nvlink-rnic-dispersion` contains one panel per degree. Each rung shows the two
transport widths side by side. The exact dispersion formula and the small-rung
packet and credit numbers appear on the canvas. Both figures render as PDF and
PNG with POSIX relative paths. Final PNGs are inspected at publication size for
clipping, overlap, readable log ticks, visible bands, legend crossings and
border contact.

## Preservation and verdict

The comparison inherits the source study's 60-artifact preservation class and
directly locks all 18 tracked files in `nvlink_flow_dynamics_v1`, including its
runner, records, PDFs and PNGs. No prior runner is executed and no prior result
or figure is rewritten.

The run is void if any fatal guard fails. TRAF-71 closes only after every cell
and both transports publish, both figures render in both formats, visual
inspection passes, preservation holds, all fatal guards pass and each expected
direction receives a verdict.
