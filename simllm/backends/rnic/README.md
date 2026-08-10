# Native RNIC queue core

This directory contains the SimLLM-owned C++17 RNIC hardware core. The module
design, status and open-task registry remain in
[`docs/modules/backends.md`](../../../docs/modules/backends.md).

The implemented v1 slice is one finite SQ and CQ bound to one QP. It models
accepted-prefix WR posting, explicit doorbell batches, serialized fetch and
CQE-write service, ordered retirement, signaled and unsignaled reclaim, CQ
owner wrap, polling, network retry gates and controlled queue failures.

## Device composition

`RnicDeviceConfig` and `RnicDevice` are the native composition entry point.
The versioned config retains device identity and the independently versioned
work-queue, QPC, DMA and network sub-configs. The QP number and opaque policy
context token are device-level authorities; the matching `WorkQueueConfig`
fields are checked projections. Every sub-config version is checked even when
its module is disabled.

The QPC v1 module is the existing scalar lookup stage. Disabling it requires
zero scalar lookup service and leaves `qpc_ready_at_ps` unset. DMA disabled
uses the accepted scalar doorbell, WQE-fetch and CQE-write services. DMA
enabled makes those scalar stages not applicable and binds the queue to one
`PcieFabric`; the existing double-charge validation is enforced before any
fabric transaction can mutate. An owned fabric is heap allocated. An external
fabric is accepted only as a `shared_ptr`, so its stable address outlives every
bound queue and plan.

Shared-fabric bindings use a nonzero device namespace to derive distinct
submission and completion ordering domains when the binding leaves either
domain at zero. Explicit domains pass through unchanged. Live domain pairs are
claimed on the shared fabric, so a collision is rejected and a failed device
construction releases its claim without changing fabric generation or
accounting. A device with an owned fabric retains the accepted SQ/CQ-derived
defaults.

Network enabled requires an injected external `NetworkPort*`, which is the
future HTSIM-9 seam. Network disabled rejects that pointer and owns an inert
port. The inert port accepts each descriptor with a fresh token and returns a
delivery on `RnicDevice::progress` at the caller's timestamp. It does not
invent packet-issue timestamps. `RnicDeviceStageReport` records scalar,
fabric, QPC, external-network and inert-network applicability explicitly.

The caller remains the sole clock authority. Deliver external network events
through `onNetworkEvent` before `progress` at the same timestamp, then choose
device-first or host-first CQ priority by ordering `progress` and
`pollCompletionQueue` as documented below. Both validation probes construct
through `RnicDevice`. The composition test retains direct module construction
only as an exact oracle and compares every public field, timestamp, counter
and PCIe accounting record. Evidence is in
[`examples/rnic_device_v1`](../../../examples/rnic_device_v1/RESULTS.md).

## PCIe fabric boundary

`PcieFabric` is the shared transaction-level PCIe resource used by WorkQueue
and later RNIC clients. Semantic class is independent of operation. Its v1
inventory has separate labels for UAR, BlueFlame, doorbell records, WQE,
QPC/ICM, MTT/MPT, payload reads, payload writes, CQE, command, interrupt and
ODP/IOMMU-fault traffic.

The executable v1 operations are a CPU host-store dependency, posted Memory
Write and non-posted Memory Read with Completion with Data. MWr, MRd and CplD
segmentation accounts for DWORD padding, 4 KiB boundaries, MPS, MRRS and an
eager Read Completion Boundary policy. Full-duplex Gen1 through Gen5 link
serializers retain rational time. Standard posted, non-posted and completion
credit pools, read tags and completion-buffer reservations are finite. Every
transaction returns class-attributed useful/transfer/host-store bytes,
directional TLP payload/overhead/modeled-link bytes, queue waits, fixed service
delay and realized analytical path delay. NUMA, IOMMU, ACS, switch, DDIO-miss
and GPU Direct penalties each accept an explicit disabled state or fixed,
discrete Gaussian and rare-tail two-Gaussian-mixture profiles with a nonzero
incidence probability. Results separately count profile evaluations,
occurrences and tail selections.

Planning is transactional. `beginPlan`, `schedule` and `commit` let a client
calculate a complete state transition against a private snapshot. A failed or
discarded plan changes no shared IDs, credits, counters or link time. The
fabric is single-threaded; multiple clients share it through deterministic
event-loop call order. Its address is stable in v1: a fabric cannot be copied
or moved and must outlive every plan and WorkQueue bound to it.
Visibility-dependency domains keep separate posted-publication and non-posted-
completion horizons. A posted request never inherits an earlier read's
completion wait. Domain zero is the generic global domain; clients use nonzero
namespaces to separate unrelated queues.

All default queue depths and delay values are synthetic. The default envelope
charges 24 B for MWr/MRd and 20 B for CplD, and calls the resulting total
modeled-link bytes. It is not a raw physical-wire byte count and does not
include unmodeled DLLPs, UpdateFC, replay, SKP or FEC traffic. V1 accepts one
fixed sample per service-latency profile and reserves one FIFO serializer per
direction. The path sampler is deterministic, counter based and integer only;
failed or discarded plans consume no shared draw. Its incidence probability
is an analytical surrogate, not a topology, translation-cache, DDIO-cache or
fault mechanism.

`link_queue_ps` counts contention from earlier public transactions, not a
transaction's own preceding TLPs. One transaction carries a same-direction
accounting eligibility chain across all MWr fragments, MRd requests and CplDs;
the rational reservation roots remain unchanged. The serializer calendar lets
a ready posted TLP fill an idle gap before a resource-blocked non-posted
request. Posted placement is recomputed after credit availability, and any
external link delay reached after that credit stall remains link queueing. If
mandatory posted-over-non-posted ordering would require displacing a result
already returned by the eager API, planning fails transactionally rather than
reporting an illegal order. Posted traffic may legally remain behind a
completion, with that delay charged to the link ledger. Deferred chronological
arbitration, class-specific queues, the remaining PCIe ordering matrix and
mechanism-driven occurrence remain BACK-16 precision work.
Optional BlueFlame, ATS/ATC and MSI-X behavior remains BACK-17 completeness
work.

The PCIe WorkQueue overload takes a separately versioned
`WorkQueuePcieBinding`. In the regular mlx5 submission path it records one
4-byte SQ doorbell-record host store and one 8-byte UAR posted write per batch,
then one WQE MRd/CplD transaction per WQE. A required completion emits one CQE
posted write. `doorbell_seen_at`, WQE-fetch begin/end and CQE visibility come
from these transactions. QPC lookup and scheduler service remain local RNIC
stages. BlueFlame production and its WQE-fetch bypass are not yet connected.
The provenance, correction freeze, equations, raw configuration and measured
sweeps are in
[`examples/rnic_pcie_v1`](../../../examples/rnic_pcie_v1/RESULTS.md).

## Network boundary

`NetworkPort` is independent of Python, htsim and any congestion-control
algorithm. A submitted descriptor carries:

- opaque WQE/WR correlation IDs;
- GOAL flow ID and tag;
- one stable opaque policy-context token;
- source, destination, traffic class, payload extent and eligibility time.

The port returns a network-owned token and later returns one delivery or drop
event for that token. A Busy result retains the SQ head until its advertised
retry time. Completion of another token does not revoke that deadline.

This first port admits one flow extent per WQE. Network acceptance and outcome
times are real v1 observations. First/last packet timestamps stay unset until
HTSIM-9 adds explicit packet-issue events, so flow admission is never mislabeled
as NIC packet start.

At one timestamp, deliver network events to `onNetworkEvent` before retrying
the SQ with `progress`. CQ priority is then explicit call order. Calling
`progress(t)` before `pollCompletionQueue(t)` gives device CQE publication
priority. Polling first gives host consumption priority and sees only CQEs
strictly older than the timestamp; CQEs due exactly at that timestamp remain
host-first. A fatal CQ overrun remains non-quiescent but exposes no next event,
so event loops must test `fatal()` and abort rather than spin.

## Standalone build

```bash
cmake -S simllm/backends/rnic -B build/rnic \
  -DCMAKE_BUILD_TYPE=Debug \
  -DSIMLLM_RNIC_WARNINGS_AS_ERRORS=ON
cmake --build build/rnic --parallel
ctest --test-dir build/rnic --output-on-failure
```

When this directory is consumed with `add_subdirectory`, tests and validation
tools default off. The link target is `simllm::rnic`.
