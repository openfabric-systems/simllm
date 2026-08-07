# RNIC PCIe v1 expectations

Sweeps A through D and directed requirements 1 through 10 first entered public
history together with their results in commit `447a962`. Sweep E and
requirements 11 and 12 first entered public history together with their
implementation and results in commit `09030c9`. No earlier public commit
contains either expectation tranche alone. This document is therefore a
frozen regression specification from those landing commits onward; it does
not claim publicly auditable preregistration.

The ordering, link-queue accounting and evidence-class corrections below are
committed as an expectations-only child of `09030c9`, before their corrective
implementation, tests or regenerated results. That parent-child order is the
public freeze audit trail for the correction. Future study extensions must use
the same expectations-only commit boundary before implementation or execution.

An adversarial review of that frozen correction found three uncovered edges:
link contention reached only after a credit stall, a future completion that a
ready posted request cannot fit before, and the two dependency-horizon arms
that make reads wait for prior posted visibility and prior read completion.
Their exact expectations are added here in a second expectations-only commit,
before the corresponding implementation, tests or execution.

## Scope and evidence boundary

This study is the closure evidence for BACK-10: deterministic, full-duplex
PCIe transaction scheduling shared by RNIC clients. It covers host
stores, posted Memory Writes, non-posted Memory Read requests and Completion
with Data packets. Semantic service class remains separate from PCIe
operation, so UAR, BlueFlame, doorbell-record, WQE, context, translation,
payload-read, payload-write, CQE, command, interrupt and fault traffic have
independent accounting labels. Most labels do not yet have producers or
class-specific hardware queues.

The model parameters are synthetic unless a captured profile says otherwise.
PCIe generation, negotiated width, MPS, MRRS and topology are observable.
Credit depths, outstanding-read limits and completion buffering are
calibrated-opaque. Internal DMA arbitration is not modeled in v1. NUMA, IOMMU,
ACS, switch, DDIO-miss and GPU Direct penalties are labeled analytical path
profiles. Each profile is explicitly disabled or has a nonzero incidence
probability and is fixed, a bounded discrete Gaussian, or a bounded
two-Gaussian tail mixture. Zero incidence is rejected for active profiles.
The incidence draw decides when the analytical penalty is applied. It does
not model topology, IOTLB or ATC state, cache residency, fault handling or
route contention. The tail mixture is a finite rare-event surrogate, not a
mathematically long-tailed distribution. No value in this study is asserted
as a ConnectX-7 internal constant.

The submission order follows the public mlx5 userspace fast path: the provider
builds WQEs, updates the SQ doorbell record in host memory, then publishes the
batch through a UAR or BlueFlame write. A doorbell record is therefore a host
store, not an assumed device DMA read. Its default send-queue store is 4 bytes;
the regular UAR doorbell is 8 bytes. The regular UAR path permits subsequent
WQE DMA reads. BlueFlame is represented as its own class but its WorkQueue
producer remains later completeness work.

## Byte and time equations

For these aligned probes, posted writes split at MPS, reads split at MRRS and
read completions split at MPS. The model also enforces the 4 KiB request
boundary, DWORD payload alignment and the configured Read Completion Boundary
for unaligned directed tests. The v1 completer uses an eager split at the first
RCB crossing. This is a legal, conservative policy, not the only behavior PCIe
permits; a measured completer policy remains open calibration work.

V1 uses one fixed service-latency sample per host-store, posted-visibility and
read-response profile. Path penalties are sampled separately: once for a host
store, once for each posted-write fragment and once for each memory-read
fragment. A read's sampled path delay applies to request-to-response
availability, not to each CplD. TLP reservations are FIFO in transaction
scheduling order on one serializer per direction. Within a visibility domain,
posted publication and non-posted completion use separate dependency horizons:
a later posted request cannot inherit the completion wait of an earlier
non-posted request. This implements the mandatory posted-over-blocked-
non-posted dependency rule. The serializer also lets a ready posted TLP fill an
idle gap before an already scheduled, resource-blocked non-posted request. If
correct arbitration would have to displace a result already returned by the
eager v1 API, the model must fail transactionally instead of reporting an
illegal order. Fully deferred cross-class arbitration, Relaxed Ordering,
ID-based ordering, Traffic Classes and Virtual Channels remain precision work.

The analytical sampler is counter based and uses only specified unsigned
integer operations. A fabric seed, path ID, component ID and component-local
draw index select independent incidence, mixture and quantile streams through
SplitMix64. A probability draw maps the high 32 bits into `[0, 1,000,000)`.
Gaussian values use a checked-in 64-entry signed Q20 table containing the
midpoint quantiles `Phi^-1((i + 0.5) / 64)`, rounded to nearest. Sigma scaling
also rounds to nearest; negative samples clamp to zero. Configuration
validation proves that every positive sample and the aggregate path maximum
fit in the asserted timestamp range. A failed or discarded plan cannot
consume a shared draw.

With logical transferred span `S` (equal to useful bytes in these aligned
sweeps), posted/read-request overhead `H = 24 B` and completion overhead
`C = 20 B`:

```text
N_MWr       = ceil(S / MPS)
B_MWr       = S + N_MWr * H
N_MRd       = ceil(S / MRRS)
N_CplD      = sum over MRd requests ceil(request_bytes / MPS)
B_MRd       = N_MRd * H
B_CplD      = S + N_CplD * C
```

MRd request packets carry headers only. Useful read data appears only in the
opposite-direction CplD ledger. A host store carries host-store bytes but zero
PCIe modeled-link bytes.

For `b` modeled-link bytes, lane count `w`, transfer rate `r` in MT/s and
encoding ratio `e_num/e_den`, continuous link serialization is:

```text
T_ser(b) = ceil(8 * b * e_den * 1,000,000 / (r * w * e_num)) ps
```

Gen1/2 use 8/10 encoding. Gen3/4/5 use 128/130 encoding. Each directional
serializer retains its rational cursor, so a stream is rounded once rather
than once per TLP. Modeled-link bytes include configured TLP/DLL/framing
overhead only. DLLPs, UpdateFC packets, replay, SKP, FEC and other lower-layer
traffic remain explicit future events, so these are not raw physical-wire byte
counts.

`link_queue_ps` measures contention from previously scheduled transactions,
not serialization by earlier TLPs of the same transaction. Let `R[k]` be a
fragment's root readiness, `P[k]` the end of the preceding same-direction TLP
in this public transaction, `E[k] = max(R[k], P[k])` its accounting
eligibility, and `L[k]` the directional link cursor. The link-ready time before
any separately accounted credit or finite-resource stall is
`I[k] = max(R[k], L[k])`. The transaction accumulates `I[k] - E[k]` and then
updates `P` from the actual reservation end:

```text
Q_link          = sum over fragments (I[k] - E[k])
R_request[k]    = transaction eligibility
E_request[k]    = max(R_request[k], preceding request end)
R_cpld[k]       = response_ready for the parent MRd
E_cpld[k]       = max(R_cpld[k], preceding CplD end in this transaction)
```

The completion chain spans all MRd fragments belonging to one logical read.
Root readiness, rather than the integer accounting eligibility, continues to
drive the rational serializer reservation so that one continuous stream is
rounded only once.
Therefore a single uncontended transaction has zero link-queue wait regardless
of its MPS, MRRS or CplD count. A later transaction submitted against an
occupied serializer charges its external wait once, on the first affected
fragment. Credit, outstanding-read and completion-buffer stalls retain their
own ledgers and are not relabeled as link queueing.

If a separately accounted resource stall moves a TLP beyond `I[k]`, the link
is arbitrated again at resource-ready time `A[k]`. Let `S[k]` be the resulting
reservation start. The additional external queue wait is
`S[k] - A[k]`; it is zero unless another reservation owns the link after the
resource becomes available. Thus the complete per-fragment charge is:

```text
Q_link[k] = I[k] - E[k] + S[k] - A[k]
```

The second term must not be dropped or folded into credit wait.

## Evidence accounting

The study reports evidence classes separately. Counts from different classes
are never added into one headline denominator.

| Cohort | Run rows | Exact-oracle rows | Relation families | Predicate instances |
|---|---:|---:|---:|---:|
| A through D | 29 | 29 | 4 | 10 |
| E | 6 | 6 | 6 | 8 |
| Total | 35 | 35 | 10 | 18 |

The ten behavioral relation families are generation equivalence, lane scaling,
read-window improvement, MPS ordering, Gaussian mean straddling, sigma-range
widening, tail selection, tail-count monotonicity, aggregate-delay
monotonicity and intermittent incidence. Conservation identities,
operation-inactive fields, disabled-profile fields, configured zero penalties
and uncontended waits are fatal structural invariants, but are unscored and do
not increase a behavioral denominator. The directed native requirements and
the three CTest executables are reported separately from both row and relation
counts.

## Sweep A: MPS and MRRS byte conservation

Issue one aligned 512-byte DMA read for each MPS and MRRS in
`{128, 256, 512}`. MPS changes CplD count; MRRS changes MRd count. Neither may
change useful or transferred bytes. Exact total modeled-link bytes are:

| MRRS \ MPS | 128 | 256 | 512 |
|---|---:|---:|---:|
| 128 | 688 | 688 | 688 |
| 256 | 640 | 600 | 600 |
| 512 | 616 | 576 | 556 |

Every row must also satisfy these conservation identities exactly:

```text
H2D payload + D2H payload = 512 B
H2D modeled bytes + D2H modeled bytes = table entry
modeled bytes = payload bytes + overhead bytes in each direction
```

## Sweep B: generation, width and MPS serialization

Issue one aligned 4096-byte posted DMA write with MPS in `{128, 256, 512}`
for Gen4/Gen5 and x8/x16. The modeled-link byte count is respectively 4864,
4480 and 4288 bytes. With zero path and posted-visibility delay:

- x16 takes exactly the continuous-stream serialization equation above;
- x8 takes exactly twice x16 before final integer rounding;
- Gen5 x8 equals Gen4 x16 for identical MPS and overhead;
- increasing MPS cannot increase bytes or completion time.

## Sweep C: outstanding-read queue

Submit 16 aligned 512-byte DMA reads at time zero with fixed 1,000,000 ps read
response latency. Sweep the outstanding-request limit in `{1, 4}` and MPS in
`{128, 512}`, with MRRS 512. An independent recurrence applies one MRd request
serializer, one opposite-direction CplD serializer and release of a read slot
after the final CplD. Each completion starts at the greater of its response
ready time and the rational completion-link cursor. Accumulated outstanding
wait is the sum of `slot_ready - request_link_ready`:

```text
request_issue[i] = max(request_link_free, completion_end[i - Q]) for i >= Q
request_end[i]   = request_issue[i] + serialized MRd header
response_ready   = request_end[i] + 1,000,000 ps
completion_end   = serialize all CplDs after response_ready
```

The measured JCT and outstanding-read wait must match the recurrence exactly.
Four slots must reduce JCT relative to one slot for both MPS values. MPS 512
must not be slower than MPS 128 because it emits fewer CplD headers.

## Sweep D: path attribution

Issue one 512-byte DMA read on four otherwise identical path profiles:
local, remote NUMA, IOMMU and remote plus IOMMU. Configure a 100,000 ps NUMA
penalty and a 200,000 ps IOMMU penalty, with MPS = MRRS = 512, zero read
response latency and all other path penalties zero.

Fixed regression expectations:

| Path | NUMA attribution (ps) | IOMMU attribution (ps) | JCT delta (ps) |
|---|---:|---:|---:|
| local | 0 | 0 | 0 |
| remote | 100000 | 0 | 100000 |
| IOMMU | 0 | 200000 | 200000 |
| remote + IOMMU | 100000 | 200000 | 300000 |

The penalties are path delay, not extra modeled-link bytes. No positive
remote or translated path may outperform its otherwise identical local path.

## Sweep E: analytical penalty profiles

Issue 4,096 independent 8-byte host stores at time zero. Host stores isolate
the sampled path distribution from PCIe serializer arbitration. Apply the
profile only to the NUMA component and keep fixed service and other path delay
at zero. Sweep these fixed regression profiles:

| Profile | Incidence | Body mean | Body sigma | Tail probability | Tail mean | Tail sigma |
|---|---:|---:|---:|---:|---:|---:|
| fixed | 100% | 100000 ps | 0 | 0 | 0 | 0 |
| Gaussian narrow | 100% | 100000 ps | 10000 ps | 0 | 0 | 0 |
| Gaussian wide | 100% | 100000 ps | 40000 ps | 0 | 0 | 0 |
| tail rare | 100% | 100000 ps | 10000 ps | 1% | 500000 ps | 50000 ps |
| tail frequent | 100% | 100000 ps | 10000 ps | 10% | 500000 ps | 50000 ps |
| intermittent | 25% | 100000 ps | 10000 ps | 0 | 0 | 0 |

The Python oracle independently reproduces every integer draw. For each row,
the aggregate realized NUMA delay, occurrence count, tail count and maximum
completion time must match the oracle exactly. Every row has exactly 4,096
NUMA profile evaluations. Fixed has exactly 4,096 occurrences, zero tail
draws, aggregate delay 409,600,000 ps and JCT 100,000 ps. Both Gaussian rows
must contain samples above and below 100,000 ps; widening sigma from 10,000 to
40,000 ps must increase the observed range. Both mixture rows must select at
least one tail draw; increasing tail probability from 1 to 10 percent must
increase both tail count and aggregate realized delay for the frozen seed.
The intermittent row must have fewer than 4,096 occurrences and zero delay on
at least one transaction. Sampling changes neither transferred bytes nor
modeled-link bytes.

## Directed boundary checks

The native harness must additionally prove:

1. A posted write crossing MPS emits another header, and a transaction crossing
   4 KiB is split even when MPS/MRRS would otherwise permit one TLP.
2. Unaligned first and last DWORDs charge directional TLP payload and
   modeled-link bytes without changing logical useful or transferred bytes.
3. A read request uses MRRS in the request direction and MPS plus RCB in the
   completion direction.
4. Posted header/data, non-posted header and completion header/data credits
   stall at the first exhausted pool and return deterministically. MRd carries
   no non-posted data payload.
5. A completion buffer reserves one MRd fragment's DWORD-padded response span
   until its final CplD, and an undersized buffer is rejected instead of
   deadlocking.
6. Visibility-dependency domains preserve simulator publication dependencies
   without violating PCIe forward progress. Host stores and posted writes
   advance the posted-publication horizon. A non-posted read waits for that
   horizon and advances a separate non-posted-completion horizon. A later
   posted write never waits for the non-posted horizon, has zero ordering wait
   in the directed long-read case and completes before that read's delayed
   completion. Posted-to-posted and non-posted-to-non-posted order remains
   conservative. In the directed Gen5 x16, MPS = MRRS = 128 dependency case,
   MWr(4) completes at 445 ps, the following MRd(128) attributes 445 ps of
   ordering wait and completes at 3,175 ps, and a second MRd(128) attributes
   3,175 ps of ordering wait and completes at 5,905 ps. Domain zero is the
   generic global domain; independent work bypasses both horizons.
7. Invalid versions, paths, generation/width, MPS/MRRS/RCB, sizes and address
   overflow fail before transaction IDs, counters, credits or time mutate.
   Per-class and cross-class aggregate overflow obey the same rule.
8. A discarded or stale transaction plan cannot partially mutate the shared
   fabric.
9. The WorkQueue regular-doorbell path accounts one host DB-record store and
   one UAR posted write per batch, one WQE DMA read per WQE and one CQE DMA
   write per required completion. Failed doorbell and CQE plans preserve both
   WorkQueue ownership and fabric state. Existing fixed-profile timing remains
   bit-for-bit unchanged.
10. Per-class totals sum exactly to the global useful, transferred,
    host-store, directional payload, overhead, modeled-link, wait and path
    fields, as well as service-delay, service-sample and analytical-profile
    accounting fields.
11. Every path component accepts fixed, Gaussian and two-Gaussian tail-mixture
    profiles. Invalid kinds, versions, probabilities, inactive fields,
    standard deviations and worst-case aggregate delay fail at construction.
12. Per-component profile evaluations, occurrences and tail selections sum
    exactly from transactions into class and global ledgers. A failed,
    discarded or stale plan consumes no shared analytical sample.
13. `link_queue_ps` follows the fragment-eligibility recurrence above. The
    single-transaction rows in sweeps A, B and D have exactly zero link-queue
    wait, as do host-store rows in sweep E. Every read-window row matches an
    independent external-contention oracle. Directed multi-MWr, multi-MRd and
    multi-CplD cases prove that adding fragments to one uncontended transaction
    cannot create the triangular `N * (N - 1) / 2` overcount.
14. A ready posted request can fill a serializer gap before a resource-blocked
    non-posted request. The directed Gen5 x16 case uses MPS = MRRS = 128, one
    outstanding read, a 128-byte completion buffer, 1,000,000 ps response
    latency and ample credits. Independent D2H transactions A = MRd(128),
    B = MRd(128) and P = MWr(4) are submitted at time zero in that call order.
    A's request ends at 381 ps and A completes at 1,002,730 ps. B cannot issue
    before 1,002,730 ps. P starts at 381 ps, completes at 826 ps, attributes
    381 ps of external link queueing and completes before B starts. A posted
    request too large for such a pre-reserved gap must raise an asserted model
    error before IDs, reservations, counters or time commit; it must never be
    silently serialized behind the blocked non-posted request. The same
    transactional rule applies to a future completion: with MPS = MRRS = 128
    and 10,000 ps response latency, MRd(128) in H2D schedules its D2H CplD from
    10,381 to 12,730 ps. A subsequently submitted D2H MWr(1,024) cannot fit
    before that returned reservation and must fail without committing any of
    its fragments. A following MWr(4) uses transaction ID 2 and the idle D2H
    interval from 0 to 445 ps.
15. Link contention reached after a posted-credit stall is attributed exactly.
    The directed Gen5 x16 case uses MPS = MRRS = 128, one D2H posted-header
    credit, one outstanding read, a 128-byte completion buffer, 1,000,000 ps
    response latency and 1,003,000 ps credit-return latency. Submit independent
    D2H transactions P0 = MWr(4), A = MRd(128), B = MRd(128), P1 = MWr(4), all
    at time zero. P0 releases its posted credit at 1,003,445 ps. Before credit
    arbitration, P1 is link-ready at 826 ps. Its credit wait is 1,002,619 ps,
    which makes it ready during B's request reservation. P1 therefore starts
    at 1,003,556 ps, completes at 1,004,001 ps and charges 937 ps of link queue:
    826 ps before the credit stall plus 111 ps after credit availability.
    Arbitration must also apply the ready qualifier after credit availability,
    not at transaction eligibility. In a companion case with zero response
    latency, 3,500 ps credit return and P1 = MWr(128), the initial 826 to 3,175
    ps gap is too short for P1's TLP, but its credit does not return until 3,945
    ps, after B's request ends at 3,556 ps. P1 must therefore succeed from
    3,945 to 6,358 ps, attributing 826 ps of link queue and 3,119 ps of credit
    wait; it must not be rejected based on the gap seen before its credit is
    available.
