# TRAF-73 NVLink credit ownership and arbitration freeze

## Expectations-only status

This is the expectations-only record for TRAF-73. It is committed before the
credit-scope correction, before the arbitration policy implementation, and
before the first TRAF-73 simulation run. The hardware cells are registered but
not executed here. No cluster time is requested by this study.

The public architecture descriptions are background evidence, not measurements
of this repository's NV4 node. The candidate credit unit, credit count, return
latency, receive capacity, and arbitration policy remain declared until the
registered hardware cells identify them.

## Physical structure and evidence classes

An NVLink receiver does not expose one credit bucket that every remote GPU can
drain. Receive buffers are hard allocated per physical link and per link-layer
virtual channel. The transmitter facing that link tracks the available receive
buffers for that link and virtual channel. A sender therefore cannot consume a
different sender's credits.

Incast contention occurs after those independent credit domains. On a direct
NV4 mesh it occurs in the destination ingress and memory-acceptance path. On an
NVSwitch path it can also occur at the crossbar output port. Round robin and
dual round robin are physically plausible crossbar policies, so release-aware
round robin remains the declared default candidate. Static interleave and
greedy capture remain explicit alternatives. None is called measured.

The evidence classes stay separate:

| Claim | Evidence class before the run |
|---|---|
| Per-link, per-virtual-channel hard allocation | PUBLIC ARCHITECTURE BACKGROUND, vendor and encyclopedic description, not our measurement |
| One implicit modeled virtual channel | MODEL SCOPE, not a virtual-channel count claim |
| 256 credits of 272 wire bytes on each modeled link | DECLARED CANDIDATE |
| 200,000 ps credit return | DECLARED CANDIDATE |
| 1 MiB receive capacity | DECLARED CANDIDATE |
| Release-aware round robin | DECLARED DEFAULT CANDIDATE |
| Static interleave | DECLARED ALTERNATIVE |
| Greedy capture | DECLARED UNFAIR ALTERNATIVE |
| A100 NV4 pass-through switch | STRUCTURAL DIRECT-MESH INVARIANT, not measurement |

Background references are NVIDIA's vendor overview of NVLink and NVLink Switch
and its NVSwitch technical overview, plus WikiChip's encyclopedic NVLink
description. They support the architecture prior only. No number from those
pages is copied into a captured-value field:

- NVIDIA, *NVLink and NVLink Switch System*, vendor overview,
  <https://www.nvidia.com/en-us/data-center/nvlink/>.
- NVIDIA, *NVIDIA NVSwitch Technical Overview*, vendor technical overview,
  <https://images.nvidia.com/content/pdf/nvswitch-technical-overview.pdf>.
- WikiChip, *NVLink*, encyclopedic architecture description,
  <https://en.wikichip.org/wiki/nvidia/nvlink>.

## Physical sanity before measurement

The candidate uses four 25 GB/s links per ordered pair. One maximum packet is
272 wire bytes carrying 256 payload bytes. A link serializes one packet in
10,880 ps; the four-link bond launches one packet every 2,720 ps when full.

The corrected candidate window per ordered pair is four links times 256
credits times 272 wire bytes, or 278,528 outstanding wire bytes. Its payload
counterpart is 262,144 bytes. One physical link takes 2,785,280 ps to serialize
its own 256-credit window. That is longer than the declared 200,000 ps return,
so the candidate predicts no credit stall at its nominal window. A missing knee
is therefore a valid result and cannot be rewritten into a measured window.

For every hardware transfer, the floor is the larger of ordered-pair wire
serialization and any independently measured destination-ingress floor. No
completion may beat bytes over the applicable nameplate link rate. A rate over
25 GB/s on one physical link or 100 GB/s on one ordered pair is fatal. The
credit knee, if any, is interpreted only after those floors hold.

## Hardware cell H1: effective credit window and return

Use one sender and one receiver on an otherwise idle directed pair. Execute all
12 directed pairs of the four-GPU NV4 mesh. The payload sweep in bytes is:

`4096, 8192, 16384, 32768, 65536, 131072, 196608, 229376, 245760, 253952,
258048, 260096, 261120, 261632, 261888, 262144, 262400, 262656, 263168,
264192, 266240, 270336, 278528, 294912, 327680, 393216, 524288, 1048576,
2097152, 4194304, 8388608`.

Each pair and size has 32 untimed warmups and 200 device-timed repetitions.
Operations are queued without a host synchronization between messages, then
one final synchronization drains the batch. The deterministic randomized size
order uses seed 7301. Record device completion time, logical bytes, all
available per-link and per-direction byte counters, replay and recovery
counters, clocks, throttling, competing processes, and destination checksum.

Fit a continuous baseline below each candidate break and a second line above
it. The first break is a knee only when its positive residual exceeds five
median absolute deviations and persists for three consecutive sweep points on
every repeated pass of that pair. The payload position estimates the effective
bonded window. The intercept increase estimates an effective return delay only
when it is positive, repeated, and not explained by a packet, launch, or memory
service boundary.

Implications:

- A repeated knee near 262,144 payload bytes supports the four-link candidate
  window, but still identifies an effective window rather than a literal
  register count or virtual-channel count.
- A knee elsewhere refutes and replaces only the effective window candidate.
- No knee through 8 MiB leaves the window and return unidentifiable. It yields
  only a lower bound or shows that return overlaps serialization.

## Hardware cell H2: per-link versus shared pool

Use receiver 3 with source sets `{0}`, `{0,1}`, and `{0,1,2}`. Repeat H1 at
payload sizes `131072, 196608, 229376, 245760, 253952, 258048, 260096, 261120,
261632, 261888, 262144, 262400, 262656, 263168, 264192, 266240, 270336,
278528, 294912, 327680, 393216, 524288, 1048576` bytes per sender. Use 64
untimed warmups and 200 timed repetitions per source count and size. All
senders remain active for the whole timed batch; start skew is recorded and
the first and last 10 percent of operations are excluded from the steady
window.

For every source, apply the H1 knee rule independently. Report the sum of the
per-sender outstanding payload at the knees.

Implications:

- If each sender's knee stays within one sweep interval of its single-sender
  knee and aggregate outstanding bytes grow in proportion to sender count,
  the result supports independent per-link pools.
- If the per-sender knee moves to approximately one half and one third while
  aggregate outstanding bytes remain within one sweep interval of the
  single-sender value, the result supports a shared destination pool.
- Missing or inconsistent knees are inconclusive. Architecture background
  remains background and no pool structure is promoted from an inconclusive
  run.

## Hardware cell H3: sustained unequal-rate arbitration

Use three senders into receiver 3. Rotate the greedy role across sources 0, 1,
and 2, for three physical cells. The greedy sender offers 100 GB/s raw and each
of the other two offers 60 GB/s raw. The total offer is 220 GB/s, above the
scored 207.101921876 GB/s receive plateau; using a smaller fraction would not
contend on this NV4 node and could not distinguish work-conserving policies.

Each stream cycles through an 8 MiB ring buffer. It runs for 50 ms of warmup,
500 ms of measurement, and 50 ms of drain. Launches are queued in separate
streams without per-chunk synchronization. The measurement interval begins
only after all three streams are active, so sequential PCIe launch skew is
outside the scored window. Record completed bytes per sender in the device
window and cross-check them against per-link counters.

The achieved sender rate is completed bytes divided by the common 500 ms
window. The physical signatures are:

- fair and work conserving: both small senders achieve 57 to 63 GB/s and the
  greedy sender receives the remainder, centered at 87.101921876 GB/s;
- greedy capture and work conserving: the greedy sender is at least 95 GB/s
  and at least one small sender is below 57 GB/s;
- static non-borrowing interleave: all three senders are between 57 and
  63 GB/s and aggregate achieved rate is at most 189 GB/s;
- any other shape is mixed or inconclusive and promotes no policy.

A checksum failure, counter disagreement above one 8 MiB chunk, throttling,
foreign traffic, a non-NV4 route, or any rate above a physical link ceiling
voids the run.

## Simulation arbitration matrix

The simulation uses the scored mixed-evidence candidate profile without
changing a numeric value. It sweeps degrees `2, 3, 4, 8, 16` under policies
`release_aware_round_robin`, `static_interleave`, and `greedy_capture`. Degrees
4, 8, and 16 are labeled **SIMULATED MESH EXTRAPOLATION** in every output. They
have no NV4 hardware counterpart.

At every degree, source 0 offers 100 GB/s raw and every other source offers
60 GB/s raw to the final endpoint. Each sender emits 240 maximum-payload
packets, with release times paced by cumulative wire bytes at its offered
rate. The scored steady window excludes the first and last 40 packets of the
greedy stream. Logical and wire bytes, source releases, packet delivery, and
the receive-capacity ceiling are fatal conservation guards.

Expected directions are frozen before the run:

- release-aware round robin is work conserving and max-min fair. At degree 2
  it carries 100 and 60 GB/s because the aggregate is below the receiver. At
  degree 3 it gives each small sender its full 60 GB/s and gives the greedy
  sender the remainder. At degrees 4, 8, and 16 all active senders approach
  equal receiver share because each small offer exceeds equal share.
- static interleave reserves turns in a fixed source cycle. At degrees 2 and 3
  the paced senders leave unborrowed turns, so the greedy sender falls toward
  60 GB/s and aggregate use is below the release-aware arm. At degrees 4, 8,
  and 16 all sources approach equal share because the receiver is continuously
  busy.
- greedy capture is work conserving but gives the first full-rate input
  priority whenever it is ready. Degree 2 matches the uncongested arm. At
  degree 3 and above the greedy sender stays near 100 GB/s and the remaining
  receiver rate is divided among the small senders, pushing each below the
  release-aware share.

The run publishes achieved GB/s and fraction of receiver service per sender,
plus aggregate utilization, for every policy and degree. A directional miss is
an honest refutation and is not repaired by changing the frozen window.

## Preservation and closure

The 89 tracked files under the two A100 packet studies, the flow-dynamics
study, and both NVLink versus rnic comparison studies total 6,429,838 bytes.
Their path-and-content digest is
`61af15faf7c7080f40a33f8f9d5503b3b0278f15be15997e90c6895cddf85c72`.
Every file stays byte-identical. The scored candidate profile remains exactly
`d33ef5b2c6fa87cc97e1e7b45a43a841a5da45f5462311e3981fbc903c56deb2`.

The structural credit-key correction and new policy seam must also preserve
the existing canonical result fixture and pass the full test suite. Any
preservation mismatch stops the task rather than publishing simulation rows.

TRAF-73 remains open after the simulation publication because the three
hardware families are only registered. It closes only after a nonvoid NV4 run
classifies the effective window, pool scope, and arbitration policy under the
rules above, or reports a frozen inconclusive result without promoting any
candidate.
