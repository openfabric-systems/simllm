# TRAF-73 aligned NVLink identification freeze

## Expectations-only status

This record is committed after TRAF-80 aligned the mechanism and before
the producer extension, the aligned policy check and every H1, H2 or H3
hardware observation. It replaces only the pre-alignment mechanism
predictions. The original workloads, candidates and simulation chronology
remain preserved except for one pre-run H2 sampling correction described
below. No cluster time has been requested.

Every candidate stays declared until a non-void hardware cell decides it.
A candidate the data cannot separate is published as unseparated. The
module and candidate profile remain unchanged during identification.

## Aligned physical basis

The aligned authority packetizes the candidate maximum payload into sixteen
16-byte payload flits plus one 16-byte header flit. One maximum packet is
therefore 256 payload bytes, 272 wire bytes and one declared credit unit.
Credits are returned only after the receiver releases the owning buffer.
The declared scope is one pool per link, destination and virtual channel; a
shared destination pool remains the H2 alternative rather than a fact.

The ordered-pair raw ceiling is 100 GB/s.
The measured receiver raw ceiling carried by the candidate profile is 207.101921876000 GB/s.
The 17-flit payload ceiling is 94.117647058824 GB/s per ordered pair.
For H1, the floor is packetized wire bytes divided by the ordered-pair
ceiling. The deliberately loose ceiling is fully serialized packet service
plus one declared return latency per packet. A value outside those bounds is
a defect before any knee fit is interpreted.

## H1: credit window and return

H1 runs all 12 directed pairs over the registered 31 payload sizes from 4 KiB through 8 MiB.
Each pair and size has 32 warmups and 200 device-timed repetitions in seed-7301 order.
A break must exceed five median absolute deviations and persist across three
consecutive sizes on every repeated pass of that directed pair.

The aligned declared candidate predicts **no break**: its 200,000 ps return is shorter than one link's 2,785,280 ps window serialization.
No repeated break through 8 MiB is therefore INCONCLUSIVE for both window
and return. It never confirms the declared values. A repeated break near
262,144 payload bytes supports the effective bonded window. A repeated break
elsewhere refutes that candidate and assigns its exact pair and break cell to
TRAF-85 for later promotion.

## H2: pool scope

H2 uses receiver 3, source sets {0}, {0,1} and {0,1,2}, and the same full
31-size H1 sweep. Using the full sweep before hardware keeps the shared
three-sender prediction, about 87,381 payload bytes per sender, inside the
sampled range. Each source-count and size has 64 warmups and 200 timed
repetitions. The H1 knee rule is applied per sender.

| Senders | Per-link knee per sender, B | Per-link aggregate, B | Shared knee per sender, B | Shared aggregate, B |
|---:|---:|---:|---:|---:|
| 1 | 262144 | 262144 | 262144.000000 | 262144 |
| 2 | 262144 | 524288 | 131072.000000 | 262144 |
| 3 | 262144 | 786432 | 87381.333333 | 262144 |

Stable per-sender knees with aggregate outstanding bytes growing with
sender count select per-link pools. Knees near one half and one third with
constant aggregate outstanding bytes select a shared destination pool.
Missing or inconsistent knees are INCONCLUSIVE and promote no scope.

## H3: downstream arbitration

H3 rotates the greedy role across sources 0, 1 and 2. The greedy stream
offers 100 GB/s raw and each small stream offers 60 GB/s raw. Each stream
cycles through an 8 MiB ring for 50 ms warmup, one common 500 ms device
measurement window and 50 ms drain. The window opens only after every
stream is active, so sequential PCIe launch skew is outside the score.

| Policy | Greedy center, GB/s | Small center, GB/s each | Aggregate center, GB/s | Hardware selector |
|---|---:|---:|---:|---|
| release-aware round robin | 87.101921876000 | 60.000000000000 | 207.101921876000 | both small senders 57 to 63 GB/s; greedy gets the remainder |
| greedy capture | 100.000000000000 | 53.550960938000 | 207.101921876000 | greedy at least 95 GB/s; at least one small sender below 57 GB/s |
| static interleave | 60.000000000000 | 60.000000000000 | 180.000000000000 | every sender 57 to 63 GB/s; aggregate at most 189 GB/s |

Any other non-void shape is mixed or inconclusive and promotes no policy.
Degrees 4, 8 and 16 in the JSON matrix are SIMULATED MESH EXTRAPOLATION
with no NV4 hardware counterpart.

## Producer lineage and fatal guards

The hardware cells extend the corrected TRAF-70 producer in place and
retain its checksum, ordering, byte, counter, replay, recovery, clock,
throttle, topology and competing-process observables. No new CUDA capture
harness is allowed. H1 and H2 add per-repetition device completions. H3
adds per-source offered rates and completed bytes inside the common device
window. No `ip link stats64` field is an NVLink wire authority.

Every fatal guard in `aligned_expectations.json` must be decidable and
pass. One violation makes the complete TRAF-73 hardware result VOID. A
void run reports findings, keeps TRAF-73 open, selects no candidate and
does not publish a behavioral pass fraction.

## Promotion boundary

TRAF-85 is free at the base commit. It is used only when a non-void cell
identifies a value or policy that contradicts the declared aligned module
or candidate profile. The residual names the exact deciding cell. This
identification wave does not edit the module, profile or any README.
