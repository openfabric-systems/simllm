# RNIC PCIe v1 expectations

Frozen before the first native PCIe-model run on 2026-08-07.

## Scope and evidence boundary

This study validates the first executable slice of BACK-10: deterministic,
full-duplex PCIe transaction scheduling shared by RNIC clients. It covers host
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
ACS, DDIO and GPU Direct fields are labeled additive path profiles, not yet
topology, IOTLB, cache-residency or route-contention models. No value in this
study is asserted as a ConnectX-7 internal constant.

The submission order follows the public mlx5 userspace fast path: the provider
builds WQEs, updates the SQ doorbell record in host memory, then publishes the
batch through a UAR or BlueFlame write. A doorbell record is therefore a host
store, not an assumed device DMA read. Its default send-queue store is 4 bytes;
the regular UAR doorbell is 8 bytes. The regular UAR path permits subsequent
WQE DMA reads. BlueFlame is represented as its own class but its WorkQueue
producer remains later BACK-9/BACK-10 scope.

## Byte and time equations

For these aligned probes, posted writes split at MPS, reads split at MRRS and
read completions split at MPS. The model also enforces the 4 KiB request
boundary, DWORD payload alignment and the configured Read Completion Boundary
for unaligned directed tests. The v1 completer uses an eager split at the first
RCB crossing. This is a legal, conservative policy, not the only behavior PCIe
permits; a measured completer policy remains open calibration work.

V1 uses one fixed latency sample per host-store, posted-visibility and read-
response profile. TLP reservations are FIFO in transaction scheduling order
on one serializer per direction. Multi-sample latency replay, chronological
cross-class arbitration and PCIe Relaxed Ordering remain open BACK-10 scope.

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

Pre-registered exact expectations:

| Path | NUMA attribution (ps) | IOMMU attribution (ps) | JCT delta (ps) |
|---|---:|---:|---:|
| local | 0 | 0 | 0 |
| remote | 100000 | 0 | 100000 |
| IOMMU | 0 | 200000 | 200000 |
| remote + IOMMU | 100000 | 200000 | 300000 |

The penalties are path delay, not extra modeled-link bytes. No positive
remote or translated path may outperform its otherwise identical local path.

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
6. Visibility-dependency domains preserve simulator publication dependencies;
   they are not a claim of full PCIe RO/IDO/TC/VC ordering. Domain zero is the
   generic global conservative domain. Marking an unrelated transaction
   independent never removes an explicit WQ publication dependency.
7. Invalid versions, paths, generation/width, MPS/MRRS/RCB, sizes and address
   overflow fail before transaction IDs, counters, credits or time mutate.
   Per-class and cross-class aggregate overflow obey the same rule.
8. A discarded or stale transaction plan cannot partially mutate the shared
   fabric.
9. The WorkQueue regular-doorbell path accounts one host DB-record store and
   one UAR posted write per batch, one WQE DMA read per WQE and one CQE DMA
   write per required completion. Failed doorbell and CQE plans preserve both
   WorkQueue ownership and fabric state. Existing scalar-mode timing remains
   bit-for-bit unchanged.
10. Per-class totals sum exactly to the global useful, transferred,
    host-store, directional payload, overhead, modeled-link, wait and path
    fields, as well as service-delay and latency-sample fields.
