# TRAF-72 transport comparison round two freeze

## Expectations-only status

This is the expectations-only record for the mapping audit, corrected
ordered-pair mapping, fluid null reference, and incast mesh extension. It is
committed before the TRAF-72 implementation and before any TRAF-72 simulation
run. The frozen hypotheses and direction signs do not change after results are
seen. An honest refutation remains published with its original bar.

## Pre-run mapping-audit verdict

The merged TRAF-71 capacity values are not deficient at degree 3. Its frozen
formula is `min(degree*pair_raw, degree*tx_plateau, rx_plateau)`:

| Degree | Ordered-pair aggregate | TX aggregate | RX plateau | Receiver grant | Binding value |
|---:|---:|---:|---:|---:|---|
| 1 | 100.000 GB/s | 160.796 GB/s | 207.102 GB/s | 100.000 GB/s | ordered-pair class cap |
| 2 | 200.000 GB/s | 321.591 GB/s | 207.102 GB/s | 200.000 GB/s | ordered-pair class cap |
| 3 | 300.000 GB/s | 482.387 GB/s | 207.102 GB/s | 207.102 GB/s | RX ingress plateau |

At degree 3 the max-min allocator therefore divided the full
207.101921876 GB/s receiver plateau. The raw capacity ratio against the NVLink
composition is exactly 1.000000. It cannot explain the reported 512 KiB p50
ratio `30.203976 / 18.145397 = 1.664553`.

The deficit is the fair-share entity mapping. TRAF-71 admitted every released
transfer as a separate max-min flow, including overlapping transfers on one
ordered source-destination pair. The NVLink arm instead queued extents within
each source class. Normalize one incast wave to service `S`; the frozen release
interval is `3S/4`. A class-queued nearest-rank median consumes `9S/4`. The
legacy per-transfer processor-sharing median consumes `601S/160`. Their ratio
is `(601/160)/(9/4) = 601/360 = 1.669444`, within 0.3 percent of the observed
1.664553 after packet slots and release jitter. The verdict is **mapping
deficit**, specifically the mapping of transfers to fair-share entities, not a
genuine packet-transport tail and not a capacity-value deficit.

The correction keeps one active htsim fair-share entity per ordered pair.
Later transfers wait in that class. Each source class is capped at 100 GB/s;
all classes into one receiver share 207.101921876 GB/s. No fitted constant is
introduced.

## Frozen topology and workload

The seven rungs are 256 B, 1 KiB, 4 KiB, 16 KiB, 64 KiB, 256 KiB, and
512 KiB. Degrees are 1, 2, 3, 4, 8, and 16. Every cell has twelve releases per
sender under each of the nine inherited seeds. Degrees 1 through 3 reproduce
the TRAF-69 release tuples byte for byte.

Degrees 4, 8, and 16 are a **SIMULATED MESH EXTRAPOLATION**. They instantiate
the same scored link geometry, endpoint plateaus, packet geometry, and credits
on more endpoints. They have no hardware counterpart on an NV4 node. An
NVSwitch-class configuration is the physical route to higher incast degrees.
Every figure that contains these degrees carries that disclosure.

## Frozen measurement caveat

Hardware incast identification is **LONG-FLOW ONLY**. Sender launches on the
real node serialize through sequential PCIe writes, so nanosecond-scale true
synchronous small-flow co-arrival cannot be constructed. Every simulated
small-flow incast result is a model prediction with no direct hardware check.
Every figure containing a small-flow rung carries that caveat.

## Frozen transports

- `nvlink-credit`: the scored three-module NVLink packet and credit domain.
- `rnic-nn`: htsim max-min grants and packet slots on the corrected
  ordered-pair mapping.
- `rnic-nn-fluid`: the htsim continuous-byte, perfectly fair fluid manifold on
  the same corrected asymmetric capacity maps. It has no packetization,
  header, ACK, control traffic, or propagation term.

## Frozen metrics

Each transport, rung, degree, and seed reports nearest-rank p50, p99, and
worst-flow flow-completion time (FCT). Publication reports the mean and the
minimum-to-maximum seed band. The empirical cumulative distribution function
(CDF) is the pointwise mean across nine seeds with a pointwise seed min-max
band.

Concurrent-flow fairness is measured within each release wave. For sender
goodput `g_i = payload_bytes / FCT_ps`, Jain fairness is
`J = (sum g_i)^2 / (n * sum g_i^2)`. The cell reports the mean across waves
and seeds with the seed min-max band. Degree 1 is exactly one by definition.

## Frozen hypotheses

1. H1: at degree 3 and 512 KiB, corrected rnic-nn moves left of TRAF-71 with
   no raw capacity change. The legacy-to-corrected p50 ratio is within 5
   percent of `601/360`, and corrected rnic-nn is at or left of corrected
   NVLink.
2. H2: rnic-nn-fluid has no tail beyond its exact class-service capacity
   oracle and sits at or left of both packet transports for p50, p99, and
   worst FCT in all 42 cells. Any miss is a harness finding unless a mechanism
   is mechanically identified.
3. H3: at degrees 4, 8, and 16, rnic-nn and fluid increasingly beat NVLink on
   256 B, 1 KiB, and 4 KiB p99 and worst-flow FCT. Both references are
   strictly left at every mesh degree and their relative advantages are
   nondecreasing with degree.
4. H4: on the same small-flow mesh cells, rnic-nn and fluid Jain fairness are
   no lower than NVLink and the fairness gap is nondecreasing with degree.
5. H5: long-flow service stays within the mapped source and destination
   capacities. Packet wire-byte and fluid payload-byte ledgers are exact, and
   every flow completes.

## Figures and evidence classes

The CDF publication splits physical degrees 1 to 3 from extrapolated degrees
4, 8, and 16, one panel per rung. Separate tail and fairness figures use one
panel per rung. The mapping-audit figure directly overlays the merged TRAF-71
degree-3 curves with the corrected degree-3 curves. All figures are emitted as
PDF and PNG, use POSIX-rendered repository paths, identify all point classes,
and state the relevant extrapolation and measurement disclosures.

Run configuration, exact oracle, behavioral, structural, and simulated
evidence remain separate. Fatal guards are never added to a behavioral pass
fraction. The merged `examples/nvlink_rnic_comparison_v1/` study is protected
by per-file byte hashes and a recursive Git object-list digest. It is never
edited; TRAF-72 supersedes only its degree-3 interpretation by reference.
