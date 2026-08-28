# NVLink credit and arbitration result

What ran: the frozen TRAF-73 simulation matrix exercised release-aware round
robin, static interleave and greedy capture under unequal offered rates at
incast degrees 2, 3, 4, 8 and 16; no hardware cell ran.

What came out: all 15 policy and degree instances matched their frozen
per-sender share predictions, and all 105 fatal conservation and physical
ceiling guards passed. At physical degree 3, release-aware round robin
delivered 87.159, 59.921 and 59.921 GB/s of raw wire traffic, static interleave
delivered 60.000 GB/s to each source and left 13.1 percent of the receiver
idle, and greedy capture delivered 99.760, 53.621 and 53.621 GB/s.

What it changes for the project: TRAF-73's simulation slice is complete and
the three registered hardware families now have frozen policy predictions to
discriminate. The qualified-NV4 hardware gate is unblocked, but TRAF-73 stays
open until a nonvoid capture classifies the effective credit window, pool
scope and arbitration policy.

What it does not change: no candidate value or evidence class is promoted, no
NV4 hardware claim is made, the candidate profile is byte-identical, all 89
files in the five merged packet and comparison studies are byte-identical, and
no TTFT or TPOT timestamp moves.

## Physical structure and evidence classes

The public architecture background describes receive buffers as hard
allocated per physical link and per link-layer virtual channel. The
transmitter tracks the corresponding credits on that link. A sender therefore
cannot consume a different sender's link credits. The model represents one
implicit virtual channel and keeps the numeric unit, window and return delay as
declared candidates rather than captured values.

Incast contention is downstream, at destination ingress and memory acceptance
on the direct NV4 mesh, plus a crossbar output on a path containing NVSwitch.
Release-aware round robin is the declared default because a fair crossbar
scheduler is physically plausible at that shared service. Static interleave
and greedy capture are selectable declared alternatives. Public vendor and
encyclopedic descriptions support only this structural hypothesis. They are
not this project's hardware measurement.

The evidence classes remain separate:

- Structural background: per-link and per-virtual-channel credit ownership.
- Declared numeric candidates: 256 credits of 272 wire bytes on each of four
  links, 200,000 ps credit return, and one modeled virtual channel.
- Declared policy candidates: release-aware round robin, static interleave and
  greedy capture.
- Behavioral simulation evidence: the 15 bounded achieved-share predictions.
- Fatal evidence: 105 conservation, visibility and rate-ceiling guards.
- Hardware evidence: registered, not run.

The four-link candidate window is 278,528 wire bytes or 262,144 payload bytes.
One link needs 2,785,280 ps to serialize its 256 maximum packets, while the
declared return is 200,000 ps. The declared candidate therefore predicts no
nominal stall knee. Observing no knee on hardware would only place a lower
bound on the effective window or an upper bound on return latency; it would
not confirm the candidate numbers.

## Registered NV4 measurements

| Cell | Exact registration | Result discriminator |
|---|---|---|
| H1 credit window and return | All 12 directed NV4 pairs; 31 payload sizes from 4 KiB through 8 MiB, including a dense zoom around 262,144 bytes; 32 warmups; 200 timed repetitions; randomized with seed 7301 | A persistent completion-time break identifies an effective window; the added delay estimates return latency. No break is inconclusive under the physical bound above. |
| H2 per-link or shared pool | Receiver 3; source sets `{0}`, `{0,1}` and `{0,1,2}`; 23 sizes from 128 KiB through 1 MiB with the same dense window zoom; 64 warmups; 200 timed repetitions | Stable per-sender knees and aggregate outstanding bytes growing with sender count support per-link pools. Knees near the single-sender window divided by sender count and constant aggregate bytes support one shared pool. |
| H3 arbitration | Receiver 3; sources 0, 1 and 2; rotate the greedy role; raw offers 100, 60 and 60 GB/s; 8 MiB ring chunks; 50 ms warmup, 500 ms steady measurement and 50 ms drain | Small senders near 60 GB/s with the greedy sender receiving the remainder support fair arbitration. Greedy at least 95 GB/s and any small sender below 57 GB/s support capture. Equal 57 to 63 GB/s shares plus unused service support static non-borrowing interleave. |

H3 uses sustained streams so sequential PCIe launch skew is negligible. It
does not claim synchronized short-flow co-arrival.

## Simulated achieved shares

The table reports raw wire GB/s in source order. Source 0 offers 100 GB/s and
every other source offers 60 GB/s. Degrees 4, 8 and 16 are simulated mesh
extrapolations with no physical NV4 counterpart.

| Degree | Scope | Policy | Achieved per source, GB/s | Aggregate, GB/s | Receiver use | Jain index | Verdict |
|---:|---|---|---|---:|---:|---:|---|
| 2 | physical NV4 | release-aware round robin | 100.000, 60.000 | 160.000 | 0.772566 | 0.941176 | PASS |
| 2 | physical NV4 | static interleave | 60.000, 60.000 | 120.000 | 0.579424 | 1.000000 | PASS |
| 2 | physical NV4 | greedy capture | 100.000, 60.000 | 160.000 | 0.772566 | 0.941176 | PASS |
| 3 | physical NV4 | release-aware round robin | 87.159, 59.921, 59.921 | 207.002 | 0.999515 | 0.966533 | PASS |
| 3 | physical NV4 | static interleave | 60.000, 60.000, 60.000 | 180.000 | 0.869136 | 1.000000 | PASS |
| 3 | physical NV4 | greedy capture | 99.760, 53.621, 53.621 | 207.002 | 0.999515 | 0.909619 | PASS |
| 4 | extrapolation | release-aware round robin | 51.750 each | 207.002 | 0.999515 | 1.000000 | PASS |
| 4 | extrapolation | static interleave | 51.750 each | 207.002 | 0.999515 | 1.000000 | PASS |
| 4 | extrapolation | greedy capture | 99.760, 35.539, 36.163, 35.539 | 207.002 | 0.999515 | 0.777057 | PASS |
| 8 | extrapolation | release-aware round robin | 25.875 each | 207.002 | 0.999515 | 1.000000 | PASS |
| 8 | extrapolation | static interleave | 25.875 each | 207.002 | 0.999515 | 1.000000 | PASS |
| 8 | extrapolation | greedy capture | 99.760; others 14.964 to 15.587 | 207.002 | 0.999515 | 0.461915 | PASS |
| 16 | extrapolation | release-aware round robin | 12.938 each | 207.002 | 0.999515 | 1.000000 | PASS |
| 16 | extrapolation | static interleave | 12.938 each | 207.002 | 0.999515 | 1.000000 | PASS |
| 16 | extrapolation | greedy capture | 99.760; others 6.858 to 7.482 | 207.002 | 0.999515 | 0.249819 | PASS |

At degree 2 the combined offer is below the receiver ceiling, so fair and
greedy policies are observationally identical. At degree 3 the frozen H3 cell
separates all three policies. At degree 4 and above every fixed static turn is
backlogged, so static interleave and round robin both converge to equal share;
greedy capture retains almost 100 GB/s while the small-source shares fall.

## Evidence and chronology audit

The committed result is a compact projection of the bulk result. The complete
record is addressed by `SIMLLM_NVFAIR_BULK_ROOT`, child
`traf73-simulation-e9a52e4/results.json`, with SHA-256
`570d37ceef74559a878d880be1eaee5b71be2fc2ca2b7f6c36c09b976591ef61`.
Its authority is the expectations-only commit
`15e68c26e81f155dfa475122ad867882a5735287`, whose expectations file has
SHA-256 `d127597dbeab23ae29f18214c583e4b958de9c57bf398d0a8308ad614f5cd7a0`.

The first run at implementation commit `0888344` produced the same workloads,
windows and per-sender values. Its aggregate scorer allowed one packet of
quantization for a sum over all sources, so it post-specified two false
aggregate refutations at degree 16. Commit `e9a52e4` corrected only the
aggregate bound to degree times the already frozen per-source one-packet
bound. The first record is retained at SHA-256
`b5713abb3902795cffd8e86ef1a9a4b40bd420a253928f36ab2d2c33442143ad`.
This is a post-specified scorer correction, not a rewritten expectation.

The preservation projection contains 89 tracked files and 6,429,838 bytes,
with path-content SHA-256
`61af15faf7c7080f40a33f8f9d5503b3b0278f15be15997e90c6895cddf85c72`.
The candidate-profile SHA-256 remains
`d33ef5b2c6fa87cc97e1e7b45a43a841a5da45f5462311e3981fbc903c56deb2`.
The legacy compatibility ledger remains byte-identical at canonical SHA-256
`2f2af64619ed3c6341b209d877d9f1e6984a67e44b97b5eb176a157294a6c252`.
