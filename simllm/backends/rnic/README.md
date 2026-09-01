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
bound queue and plan. Its effective `PcieFabricConfig` must equal the embedded
device fabric config field by field, so the retained device config never
misreports a silently substituted shared fabric.

Shared-fabric bindings use a nonzero device namespace to derive distinct
submission and completion ordering domains when the binding leaves either
domain at zero. Explicit domains pass through unchanged. Live domain pairs are
claimed on the shared fabric, so a collision is rejected and a failed device
construction releases its claim without changing fabric generation or
accounting. Standalone submissions through a device may use its own claimed
pair or an unclaimed domain, but reject a domain claimed by another live
device before changing the fabric. A device with an owned fabric retains the
accepted SQ/CQ-derived defaults. The namespace field is inert with DMA off,
must be zero for an owned fabric, and is accepted for a shared fabric only
when at least one domain must be derived.

Network enabled requires an injected external `NetworkPort*`, which the
HTSIM-9 composition binds to the directly invoked simulator. Network disabled
rejects that pointer and owns an inert port. The inert port accepts each
descriptor with a fresh token and returns a delivery on
`RnicDevice::progress` at the caller's timestamp. It does not invent
packet-issue timestamps. `RnicDeviceStageReport` records scalar, fabric, QPC,
external-network and inert-network applicability explicitly.

The caller remains the sole clock authority. Deliver external network events
through `onNetworkEvent` before `progress` at the same timestamp, then choose
device-first or host-first CQ priority by ordering `progress` and
`pollCompletionQueue` as documented below. A rejected standalone PCIe request
does not advance the device clock. Both validation probes construct through
`RnicDevice`. The composition test retains direct module construction only as
an exact oracle and compares every public field, timestamp, counter and PCIe
accounting record. Evidence is in
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
times are real ABI-v1 observations, where first and last packet timestamps stay
unset. ABI v2 adds explicit packet-attempt events; the native timeline accepts
first and last packet issue only from data or retransmission TX-start events,
so flow admission is never mislabeled as NIC packet start.

At one timestamp, deliver network events to `onNetworkEvent` before retrying
the SQ with `progress`. CQ priority is then explicit call order. Calling
`progress(t)` before `pollCompletionQueue(t)` gives device CQE publication
priority. Polling first gives host consumption priority and sees only CQEs
strictly older than the timestamp; CQEs due exactly at that timestamp remain
host-first. A fatal CQ overrun remains non-quiescent but exposes no next event,
so event loops must test `fatal()` and abort rather than spin.

## Session records and projections

`session_record.h` defines schema-tagged configuration, result and
bookkeeping records for one composed RNIC session. Structural mode names
`SimllmNativeRnicSession` as the sole WQE authority and carries the canonical
effective-hardware object plus its SHA-256 digest. Bypass mode names
`AtlahsWqeLedger`, omits native hardware and its hash, and keeps the legacy
ledger as the only lifecycle authority. `RnicAuthorityAudit` rejects both or
neither authority at construction and records native posts or legacy
mutations only in the selected mode.

The effective-hardware hash includes active queue capacities and services,
module selection, resolved PCIe domains and bindings, fabric limits, active
paths and active analytical-profile parameters. It excludes session, policy,
QP correlation and disabled-module payload. Path declarations are sorted by
path ID. Disabled-path payload and the PCIe non-posted data-credit placeholder
are omitted because neither can affect a supported transaction. Shared and
owned fabrics remain distinct resource scopes, while shared-device hashes use
the attached fabric configuration and resolved ordering domains.

At quiescence, `projectStructuralSessionResult` reads native WQE records and
returned CQ entries without advancing either object. It rejects projection
loss, duplicate identities, counter disagreement, timestamp drift and
nonterminal records. The stable send key is the session, source endpoint,
`send`, SQ ID and post sequence. A send projection names its SQ and completion
queue and leaves `rq_id` absent. The completion renderer preserves the pinned
legacy CSV prefix, column order and LF bytes.

`simllm-rnic-bookkeeping-v1` is the public structural WQE projection of that
same result, not another mutable ledger. It intentionally preserves send
cardinality without fabricating the receive-queue parent required by the older
`simllm-request-bookkeeping-v1` compatibility rule. `ComposedRnicSession`
joins the native projection to request and execution scope, including
`CompletionEvent` and step metrics. The HTSIM-9 composition supplies the
concrete network port and live token reconciliation.

The Python readers in `simllm.backends.rnic_records` strictly validate the
schemas, recompute hashes, freeze nested configuration objects and reconcile
bookkeeping and completion projections. Their reusable bypass checker guards
GOAL and topology bytes, profile, seed and canonical semantic baseline
parameters, then
compares completion CSV, canonical completions, `StepResult` tuples and replay
TTFT or TPOT summaries byte for byte.

## Transmit pipeline

`rnic_tx_pipeline.h` is the opt-in transmit slice of the golden model. It is
selected by `RnicNetworkConfig::abi_version = 2` together with an enabled
`packetization` block; the two must agree, and a contradiction is refused at
construction. The default configuration is ABI v1 with packetization off,
which is not merely equivalent to the old path but literally the same code:
the work queue binds straight to the injected port, so every accepted v1
timestamp, counter and completion order is unchanged.

With the pipeline selected, it becomes the port the work queue binds to and
the injected port becomes its downstream packet face. The queue keeps
submitting one flow extent per WQE and never learns about packets. The
pipeline has three parts:

- the **packetizer** segments an extent at the MTU, charges the wire header
  bytes per packet, assigns a per-QP PSN, and submits one
  `NetworkTxDescriptor` per packet with `extent_index` and `extent_count`
  carrying the packet index and count. The PSN stays inside the endpoint and
  the facade, because it is transport state the fabric does not need;
- the **outstanding-work window** bounds in-flight WQEs, bytes and packets per
  QP. A WQE is in flight from the issue of its first packet to the terminal of
  its last, so the window is what is on the wire, not what the send queue
  holds. It gates packet issue rather than admission, which is why the
  pipeline never returns Busy: it has no way to promise a retry time for an
  acknowledgement it has not seen, and admission is a real, separate instant
  from first packet issue;
- the **pacer** applies per-QP and per-NIC bits-per-second and message-rate
  ceilings, shared across the QPs of one device. Rate arithmetic is exact
  rational: a remainder carries the fractional picoseconds forward, so a
  million-packet run has bounded error rather than one truncation per packet.
  The bit rate is charged on wire bytes at the effective wire rate, which is
  the rate at which a full calibration-MTU packet delivers the profile's
  goodput. The measured small-message ceiling is charged once per work
  request, not once per wire packet, because what the campaign measured is a
  host-bound message rate; at or below the MTU the two readings coincide,
  which is where it was measured.

The downstream packet-port contract is narrow: the port returns one token per
accepted attempt and later reports TX finish, RX arrival and one terminal for
it. The pipeline stamps the TX start itself at the paced issue instant,
because the packetizer is the transmit authority, and it is that event that
fills `first_packet_at_ps` and `last_packet_at_ps`. `tests/fake_network.h`
carries `FakeV2NetworkPort`, which serializes at a link rate, adds a fixed
one-way latency and acknowledges per packet.

The caller must step to the times `nextEventTime()` reports. A release forced
later than an announced paced instant is counted as a late release, and a
study treats a nonzero count as a voided run rather than a measurement.
Evidence is in
[`examples/rnic_cmodel_v1`](../../../examples/rnic_cmodel_v1/RESULTS.md).

## Receive pipeline and requester transport

`rnic_rx_pipeline.h` is the receive half, selected by an enabled receive block
on the same ABI v2 network configuration and off by default. It is two blocks
in series. The ingress meter admits wire bytes into a finite buffer drained at
a service rate and discards the overflow at the PHY with no transport signal
at all, which is what makes the measured loss silent. The receive processor
then applies the per-QP receive packet-rate ceilings and the per-NIC one, and,
for a reliable connection, checks the responder's sequence number and emits an
ACK or a NAK. The sequence check runs at line rate on arrival rather than at
the drain instant, because on real silicon the transport parser sits in the
receive path while the buffer stages payload toward the host; delaying the NAK
by the standing queue depth would collapse go-back-N at any loss rate.

A packet the responder throws away keeps its bytes charged against the meter.
It was still received, parsed and sequence-checked, so it consumed the ingress
service its bytes were metered for. Refunding it would make go-back-N free at
the receiver and pin the equilibrium goodput to the drain rate.

The requester transport lives in the transmit pipeline behind
`transport_enabled`, whose off path is the unchanged slice-B code. It keeps
per-QP sequence and acknowledgement state and recovers by go-back-N: a NAK
opens one recovery episode at its sequence number, every attempt at or above
it that is still on the wire is closed as dropped and requeued, and the issue
queue is ordered by sequence number so a replay goes back where it belongs
rather than ahead of a lower number still waiting. A packet the responder
never saw draws no NAK, so the retransmission timer is the only way out of a
tail loss; firmware 16.31 counts that on `local_ack_timeout_err` and firmware
16.32 counts zero for it.

`rnic_nic_counters.h` is the observable-state facade, spelled the way the real
NIC spells it. Three groups are inert because silicon reports them inert:
`np_ecn_marked_roce_packets`, the two receive-pause counters, and
`rx_out_of_buffer` with the two `outbound_pci_stalled_*` counters.
`tests/fake_network.h` gains `FakeV2Fabric`, a two-endpoint wire with
per-direction links, configurable propagation and a reproducible loss
injector, and the facade gains `rnic_cm_rx_packet` with
`rnic_cm_nic_counters`. Evidence is in
[`examples/rnic_cmodel_rx_v1`](../../../examples/rnic_cmodel_rx_v1/RESULTS.md).

## Hardware profile, anomaly table and C facade

`rnic_hw_profile.h` carries the hardware parameter set with one evidence class
per field: `documented`, `driver-inferred`, `calibrated-opaque` or `declared`.
`kConnectX5_100G` is the measured mlx5-campaign set. `kConnectX7_400G` is
`scaleProfile(kConnectX5_100G, 4)`, which scales the link, goodput,
packet-rate and threshold fields, keeps the initiation, MTU, header,
outstanding-work, transport and flow-control fields, and marks every scaled
field `declared`. Rates are integers of bits or packets per second, so scaling
is exact and a rendered profile has no floating-point spelling.

The lumped fixed offset is the profile invariant worth knowing: the five
work-queue service stages plus `wire_round_trip_floor_ps` sum to `t_eff_ps`.
The campaign fitted one offset that already contains the wire round trip, so a
model that charges the round trip explicitly subtracts it from the lump rather
than adding to it. The split across the five stages is declared, and it is
constrained by the requirement that no serialized stage binds before the
transmit pacer does.

The profile is its own versioned record: `simllm-rnic-hw-profile-v1` with
`renderRnicHwProfileJson` and `rnicHwProfileSha256`. It is deliberately not
part of the effective-hardware schemas or their hash inputs, which identify a
composed device so a policy comparison cannot silently change hardware.

`rnic_anomaly_table.h` carries the measured performance anomalies as a
`constexpr` array: identity, trigger, the rendered effect cell, a short
machine-readable magnitude handle, the mechanism kind and the campaign
evidence. `renderRnicAnomalyTableMarkdown` projects it to
[`docs/design/rnic-anomaly-table.md`](../../../docs/design/rnic-anomaly-table.md),
and the native test compares the render to that file byte for byte and checks
that the design document still carries every row.

`rnic_cmodel_c.h` is the `extern "C"` facade an RTL testbench drives through
DPI-C: create, post, doorbell, receive, event, progress, next-event, poll,
transmit drain, counters, trace and destroy over plain fixed-width structs
with picosecond timestamps. No exception crosses the boundary; every entry
point returns a status code. Two struct typedefs are spelled
`rnic_cm_event_info` and `rnic_cm_counter_set` because C gives a typedef and a
function the same name space and the entry points `rnic_cm_event` and
`rnic_cm_counters` are the contract.

Determinism is the contract that makes the facade a golden reference: the same
stimulus sequence against the same profile and configuration produces a
byte-identical trace, one line per stimulus and per observed transition. The
native test proves the facade reproduces the C++ device's completion
timestamps exactly and that two identical stimulus sequences trace
identically. The receive entry point and the control-event kinds fail closed
with `RNIC_CM_ERROR_UNSUPPORTED` rather than pretending to model a path that
is not landed.

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
