# The pluggable packet-device model

## Status and scope boundary

This document states the model SimLLM is being built toward: the GPU is modeled
the same way the NIC already is, as a device with typed ports that carry
packets, and the software stack above it is a packet producer rather than a
bandwidth constant. The statement is architectural direction, not a landed
mechanism.

Nothing here changes behavior. Every default, artifact, timestamp and reported
number in the repository at this commit is unchanged by this document. What it
adds is the vocabulary that later implementation waves are held to, and the
numbered tasks that own the gaps: COMP-34 and COMP-35 in
[compute](../modules/compute.md), BACK-46, BACK-47 and BACK-48 in
[backends](../modules/backends.md), and TRAF-45 in
[traffic](../modules/traffic.md).

Evidence discipline follows the repository's usual split. A number called
measured here is first-party, from a study in `examples/` with a frozen
expectation record. A number called nameplate is a vendor figure this project
has not measured, and is labeled as such at every use. External claims about
NCCL and RCCL internals carry their source in the
[sources](#sources-for-external-claims) table at the end.

## The model statement

Four sentences carry the whole model.

1. **A device is a set of typed ports plus a service model.** The port carries
   the protocol identity, the direction, the ceiling and the provenance of that
   ceiling. The service model decides when a packet leaves and when it lands.
   An NVIDIA GPU has PCIe ports and NVLink ports; an AMD ROCm GPU has PCIe
   ports and xGMI (Infinity Fabric) ports; a Grace Hopper superchip replaces
   the GPU's host-side PCIe port with an NVLink-C2C port; an RNIC has PCIe
   ports and wire ports. These are the same kind of object with different
   parameters, not four different modeling techniques.
2. **Software stacks are the packet producers.** NCCL and RCCL are not a rate;
   they are the thing that decides how many bytes cross which port in what
   order. A collective becomes ring steps, ring steps become chunks, chunks
   become either peer stores over NVLink or xGMI, or descriptors, doorbells and
   DMA over PCIe followed by wire packets.
3. **Links carry packets between ports.** Host to GPU, NIC to GPU, GPU to GPU
   inside a node and NIC to NIC across the fabric are all packet flows over
   port-to-port links. They differ in ceiling, latency, ordering and control
   vocabulary, not in kind.
4. **Every packetized leg keeps a byte-identical analytic off path.** The
   closed forms this repository validated (the fluid fabric serializer, the
   flat intra-node NVLink rate, the calibrated collective profiles) stay
   selectable and stay exact. Packetizing a leg is a precision level, and the
   contract in [architecture.md](../architecture.md#precision-levels-and-their-contract)
   applies to it: a level may change a duration, never what happened.

The asymmetry this model removes is visible in the current code. The NIC is
already a device with typed ports: `RnicDevice` is composed from a work-queue
core, an optional PCIe fabric, an optional host-memory registry and either an
injected `NetworkPort` or an owned inert one, and a disabled module keeps its
interface with parameters inert or rejected
(`simllm/backends/rnic/include/simllm/rnic/rnic_device.h`). The GPU is not. It
is a calibration profile carrying two separate link mechanisms: one flat
per-GPU egress cursor for SM stores (`NvlinkProfile`, whose egress opcode set
is `ST` and `STG` only) and a set of per-direction copy-engine profiles that
already price host-to-device, device-to-host, device-to-device and peer
transfers with their own setup cost and bandwidth
(`CopyDirectionProfile` and `CopyEngineProfile`, both in
`simllm/compute/gpu_model.py`). Both are real service models with measured
cells behind them. What neither has is a port object: no protocol identity, no
declared capabilities, no ceiling provenance field, and no shared packet
vocabulary.

## Port taxonomy

Every port the model needs to name, with the ceiling it is held to and where
that ceiling comes from. Measured entries are verified against
[a100_hardware_envelope_v1](../../examples/a100_hardware_envelope_v1/RESULTS.md)
and
[gh200_hardware_envelope_v1](../../examples/gh200_hardware_envelope_v1/RESULTS.md).

| Port | Protocol | Where it appears | Ceiling and measurement | Provenance |
|---|---|---|---|---|
| Host link, PCIe | PCIe generation 4 by 16 on the measured node | x86 host to A100, over the same kind of fabric the RNIC sits on, though the modeled RNIC fabric defaults to generation 5 by 16, so this ceiling is the measured node's and not the modeled fabric's | 31.5 GB/s per direction nameplate; measured 26.78 GB/s host to device and 26.19 GB/s device to host on pinned 256 MiB transfers, i.e. 85.0 and 83.1 percent | first-party measured, A100 envelope |
| Host link, C2C | NVLink-C2C | Grace host to Hopper GPU, replacing the PCIe host port | 450 GB/s per direction specification; measured 419.93 GB/s inbound (93.3 percent) and 169.96 GB/s outbound (37.8 percent), asymmetric by a factor 2.47 | first-party measured, GH200 envelope |
| Peer link, NVLink3 | NVLink3, `NV4` mesh, 4 bonded links per ordered pair | A100 GPU to GPU inside a node | 100 GB/s per ordered pair and 300 GB/s per-GPU egress; measured 94.00 to 94.07 GB/s per pair (copy-engine wire efficiency 94.0 percent) and 281.65 GB/s on the three-way fan-out (93.9 percent) | first-party measured, A100 envelope |
| Peer link, NVLink4 | NVLink4, `NV6` mesh, 6 bonded links per ordered pair | GH200 GPU to GPU inside a node | 150 GB/s per ordered pair and 450 GB/s per-GPU egress; measured 133.24 to 133.27 GB/s per pair (copy-engine wire efficiency 88.8 percent) and 398.71 GB/s on the fan-out (88.6 percent) | first-party measured, GH200 envelope |
| Peer link, xGMI | xGMI over Infinity Fabric | AMD Instinct GPU to GPU inside a node | vendor nameplate only: AMD states up to 64 GB/s per point-to-point link and 448 GB/s aggregate per GPU over seven links, with 45 to 48 GB/s per link and 315 to 336 GB/s aggregate reported as realized | **not first-party**, AMD ROCm blog, no SimLLM measurement exists |
| Wire port, Ethernet | 400 Gbit/s RoCE-class over a two-tier Clos | RNIC to fabric, the default reference configuration | 400 Gbit/s, i.e. 50 GB/s per port per direction | model configuration, executed by the htsim packet models |

Three rules follow directly from the table and are load bearing for everything
below.

- **A ceiling belongs to a port on an architecture, never to the model.** The
  repository's `DEFAULT_NVLINK_BANDWIDTH_BYTES_PER_SECOND` of 450 GB/s is
  exactly the H100 and GH200 per-GPU NVLink4 payload egress nameplate, and it
  is exactly 1.5 times the A100's 300 GB/s. One number cannot serve both.
- **A nameplate is not a payload rate until it is checked.** The GH200 freeze
  took 26.5625 GB/s per NVLink4 link from `nvidia-smi nvlink -s`, which is the
  raw signalling rate; the payload rate after the 17/16 encoding overhead is
  25.0 GB/s, and taking the report at face value overstates a Hopper ceiling by
  6.25 percent. NVLink3 is unaffected because both figures are 25 GB/s there.
- **The host port is not symmetric everywhere.** A100 PCIe was symmetric within
  2 percent; Grace C2C is asymmetric by 2.47 times, so a single bidirectional
  host-link rate is wrong on Grace Hopper in one direction, and which direction
  matters depends on whether a workload stages weights in or reads results out.

## Producer taxonomy

The producers are what turn a collective into traffic on the ports above. Each
entry names what it emits, on which port, and where SimLLM stands today.

### NCCL host proxy

The CPU-side progression loop that drives inter-node transfers: it polls the
GPU-published tail counter, calls the network plugin, and advances the head
counter after network completion so the collective kernel can reuse a FIFO
slot. It emits PCIe traffic (descriptor and counter accesses, doorbell stores)
and causes wire packets, but it moves no payload itself. SimLLM mirrors the
send leg with real symbol names in `simllm.compute.nccl_stack`
(`ncclProxySaveOp`, `ncclProxyProgress`, `sendProxyProgress`), audited against
NCCL release `v2.30.7-1`, commit
`73cf112295c33aee2b895f329f592f2a9b4b0f97`, in the name-audit table in
[compute.md](../modules/compute.md#nccl-stack-name-audit). The receive leg is
explicitly absent and is part of COMP-15.

### The ncclNet plugin ABI

NCCL's network transport is a dynamically loaded plugin, not a built-in. NVIDIA
documents the contract in the NCCL repository's net plugin README: NCCL looks
for a `libnccl-net.so` library and loads it dynamically, communication "will be
done using the functions `isend`, `irecv` and `test`", and prior to calling
`isend` or `irecv` NCCL calls `regMr` on all buffers "to allow RDMA NICs to
prepare buffers". The README is written against `ncclNet_v11` and states that
plugins are encouraged to export several versioned symbols so one plugin spans
a range of NCCL versions. The NCCL release this repository audited its mirrored
names against carries the same member names at
`src/include/plugin/net/net_v12.h`, and SimLLM mirrors `ncclNet.isend` and
`ncclNet.test` at that boundary today.

This ABI is the exact seam where a packet producer meets a device. The buffer
that `regMr` prepares is what makes the next rule possible.

### GPUDirect RDMA, the leg that skips the host

NVIDIA defines GPUDirect RDMA as "a direct path for data exchange between the
GPU and a third-party peer device using standard features of PCI Express". In
model terms this is a PCIe transaction whose requester is the NIC and whose
completer is GPU memory, with no host bounce buffer in between. The repository
already describes this placement in prose: the per-channel data FIFO lives in
GPU memory and the NIC's payload DMA reads it directly over PCIe, while
counters and flags stay host visible (see the full call loop in
[README_PRO.md](../README_PRO.md#full-call-loop-default-setup)). That placement
is not only prose: the accepted BACK-20 artifact
(`examples/rnic_submission_v1/results.csv`) carries `data_endpoint` as
`gpu_memory` under both the CPU-proxy and the GPU-initiated shape, with the
proxy shape keeping SQ, CQ and doorbell records in host pinned memory, which is
exactly the GPUDirect placement. The data region is the endpoint that matters
for a NIC payload read, and the payload read really is issued against it, as a
`PayloadRead` non-posted read on the shared fabric. Every PCIe path
configuration also carries a `gpu_direct` analytical delay component.

What is missing is the second device. The GPU-memory label today is a property
of an allocation the posting RNIC device owns, and a WQE data descriptor must
resolve to a `DataRegion` whose `device_owner_id` equals that device's, so
there is no separately modeled GPU that owns the region, claims it on the
fabric, or has its transactions accounted apart from the NIC's. The GPU-direct
term is also an analytical penalty whose occurrence is not yet mechanism-driven,
which is BACK-16 precision scope. BACK-46 owns the composition half: a GPU
attached to the same fabric as an endpoint in its own right.

### NCCL P2P and SHM intra-node transports

NVIDIA documents `NCCL_P2P_DISABLE` as disabling "the peer to peer (P2P)
transport, which uses CUDA direct access between GPUs, using NVLink or PCI",
and `NCCL_SHM_DISABLE` as disabling the shared-memory transport, which "is used
between devices when peer-to-peer cannot happen, therefore, host memory is
used". Both matter to this model because they select which port carries an
intra-node chunk: a peer store on the NVLink port, a peer store over PCIe when
no NVLink path exists, or a two-hop staging through host memory that crosses
the host PCIe port twice.

The A100 envelope study supplies first-party evidence of which one ran there:
NCCL 2.31.2 built 8 channels on the width-2 communicator and 24 on the width-4
one, two channels per physical NVLink in both cases, and every connection was
reported `via P2P/direct` with no proxy or copy-engine hop, with NVLS
unavailable on that board.

### GPU-initiated submission

The device can post its own network work instead of asking a host proxy to do
it. SimLLM has landed three submission shapes with independently named
producer, requester and sole CQ consumer (BACK-20): a host CPU driver, a CPU
proxy fed by one GPU-written host-visible descriptor queue, and GPU-initiated
rings that require SQ, CQ and doorbell records in GPU memory and mark the MMIO
UAR mapping GPU owned. BACK-27 then made the CPU-proxy and GPU-initiated
producers explicit timed tasks in the concurrent compute service, disabled by
default and byte-identical when off. On the NCCL side the same idea appears in
the plugin ABI as device offload: the README describes a connection where
`*sendDevComm` or `*recvDevComm` points at a valid object as NCCL "requesting
device offload for this connection". BACK-37 retains the GPU-owned CQ consumer
and its runner callback.

### RCCL on AMD

AMD's ROCm Communication Collectives Library is documented as "a stand-alone
library that provides multi-GPU and multi-node collective communication
primitives optimized for AMD GPUs. It uses PCIe and xGMI high-speed
interconnects" (RCCL 2.30.7 documentation). Its network plugin page is titled
"Using the NCCL Net plugin API" and describes the same asynchronous `isend`,
`irecv` and `test` operations with the same `regMr` buffer registration to
"allow RDMA NICs to prepare the buffers", with the plugin packaged as a shared
library named `librccl-net.so`.

Read conservatively, that is one substitution in this model and no new
structure: the intra-node peer port becomes xGMI instead of NVLink, and the
inter-node producer keeps the same plugin-shaped boundary under a different
library name. This document deliberately claims nothing further about RCCL
internals, protocol selection, or how its intra-node transports map onto xGMI
link groups; the sourced evidence dossier for the AMD side, RCCL's lineage
and transports, the xGMI nameplate tables and the ROCm RDMA path, is
[amd-gpu-fabric.md](../papers/amd-gpu-fabric.md). No first-party AMD
measurement exists in this repository, so COMP-35 is registered to fail
closed rather than to guess.

## Everything is packets

The generalization is not new machinery. It is the observation that the packet
and transaction vocabulary this repository already landed is currently
NIC-private, and that nothing in it is NIC-specific.

**The NetworkPort ABI v2 vocabulary** (BACK-25 and BACK-26, closed) already
separates the three scopes a packetized leg needs:
`NetworkEventScope::FlowExtent` for the logical operation,
`NetworkEventScope::PacketAttempt` for one attempt at moving a piece of it, and
`NetworkEventScope::TransportControl` for feedback that is not payload. Its
event kinds are `PacketTxStarted`, `PacketTxFinished`, `PacketRxArrived`,
`Delivered`, `Dropped`, `EcnMarked`, `CnpReceived`, `EligibilityUpdated`,
`RateUpdated`, `PfcFrameSubmitted`, `PfcPaused`, `PfcResumed` and
`LinkStateChanged`, with `NetworkPacketKind` distinguishing data,
retransmission and control packets and typed drop location, reason and evidence
provenance
(`simllm/backends/rnic/include/simllm/rnic/network_port.h`). An NVLink or xGMI
peer transfer needs `PacketTxStarted`, `PacketTxFinished`, `PacketRxArrived`,
`Delivered` and `Dropped`, and it needs the same extent and attempt identity.
It does not need ECN, CNP or PFC, and the correct way to express that is a
capability-gated port that rejects an unsupported request explicitly, exactly
as the ABI already rejects a v2 consumer paired with a v1-only producer.

**The PCIe transaction model** (BACK-10, closed; BACK-16 and BACK-17 open)
already carries the endpoint and class vocabulary a shared fabric needs:
`PcieOperation` distinguishes `HostStore`, `PostedWrite` and `NonPostedRead`;
`PcieEndpointKind` distinguishes `MmioBar`, `HostPinnedMemory`, `GpuMemory` and
`DeviceMemory`; and twelve service classes name what the transaction is for,
including `UarDoorbell`, `DoorbellRecord`, `WqeRead`, `QpcIcm`, `PayloadRead`,
`PayloadWrite` and `CqeWrite`
(`simllm/backends/rnic/include/simllm/rnic/pcie_fabric.h`). The vocabulary
therefore already expresses the GPUDirect leg, and an accepted study already
exercises it. What is missing is composition: several RNIC devices may share one
fabric, but a GPU cannot attach to one, so the GPU-memory region a payload read
names is owned by the reading NIC rather than by a modeled GPU.

Four rules make the generalization concrete.

1. **One packet vocabulary, several ports.** Port kind is a property of the
   port object and its capabilities, not a second event language. A peer port
   that cannot mark ECN advertises that and rejects a request for it.
2. **A port is a service, not a wire constant.** Ceiling times efficiency is
   the envelope of the service, not a substitute for it. Ordering, credit and
   concurrency belong to the port.
3. **Identity is per extent and per attempt.** Session-unique extent and
   attempt tokens are what make a retry, a partial delivery and a
   double-charged byte detectable. That requirement does not weaken because a
   link is inside a node.
4. **Direction is explicit at both endpoints.** A modeled port has egress and
   ingress, and a leg that charges only the source is a known defect class in
   this repository, not a simplification: the intra-node path was corrected to
   charge `max(egress_bytes, ingress_bytes)` per endpoint under CORE-41, while
   the cross-node path still has no destination-ingress serializer (CORE-48).

## Mapping from existing assets to model roles

| Model role | What exists today | Where | Owning tasks |
|---|---|---|---|
| Device composition entry point, NIC | Versioned `RnicDeviceConfig` and `RnicDevice`, modular, disabled modules keep the interface with parameters inert or rejected | `simllm/backends/rnic/include/simllm/rnic/rnic_device.h` | BACK-18 closed; COMP-34 mirrors the pattern for the GPU |
| PCIe fabric | Shared transaction model with twelve service classes, finite credits, tags and buffers, analytical path penalties | `simllm/backends/rnic/include/simllm/rnic/pcie_fabric.h` | BACK-10 closed; BACK-16 and BACK-17 open; BACK-46 makes it multi-device |
| Virtual host memory | Tracked QPC, ring, doorbell-record and data allocations with the QPC translation asymmetry | `simllm/backends/rnic/include/simllm/rnic/host_memory.h` | BACK-19 closed |
| Submission shapes | Host CPU, CPU proxy from a GPU-written descriptor queue, and GPU-initiated rings with a GPU-owned CQ; producer work as timed GPU tasks | `simllm/backends/rnic/include/simllm/rnic/submission.h` | BACK-20 and BACK-27 closed; BACK-37 open for the GPU CQ consumer |
| GPU service model and NVLink egress cursor | One flat per-GPU egress serializer shared by every NVLINK store (opcodes `ST` and `STG`), plus the ring-collective egress kernel | `NvlinkProfile` in `simllm/compute/gpu_model.py`, launcher `nccl_ring_allreduce_launch` in `simllm/compute/nccl.py` | COMP-11 closed; COMP-31 open for peer topology, ingress and reduction lanes; COMP-34 adds the port objects |
| GPU copy-engine service | Per-direction profiles with their own setup cost and bandwidth for host to device, device to host, device to device and peer transfers, and an estimate that rejects a direction the engine does not declare; this is the mechanism the measured copy-engine peer efficiencies calibrate | `CopyEngineProfile`, `CopyDirectionProfile` and `CopyEngineServiceModel` in `simllm/compute/gpu_model.py` | COMP-34 adds typed ports over these directions rather than replacing them |
| NCCL stack skeleton | Name-mirrored communicator, planner, GPU FIFO, proxy, `ncclNet.isend` and `test`, verbs and doorbell, on one virtual clock | `simllm/compute/nccl_stack.py` | COMP-15 open; BACK-47 names the plugin seam as the producer boundary |
| Host initiation | `HostInitiationModel` with the exact-zero ideal profile and calibrated launch-throughput profiles | `simllm/compute/host.py` | COMP-2 closed; COMP-28 open for the analytical submission fallback |
| Analytic intra-node locality | Placement-driven local versus remote split, per-endpoint byte ledger, `max(egress, ingress)` endpoint load | `classify_step_locality` in `simllm/traffic/locality.py` | TRAF-10 and CORE-41 closed; CORE-48 open for cross-node ingress; TRAF-45 adds the packetized leg |
| Collective envelope arms and regime curve | `CollectiveLatencyProfile`, `CollectiveFixedCostEnvelope` and the inert `CollectiveBandwidthCurve` | `simllm/traffic/collective_latency.py` | TRAF-36, TRAF-42, TRAF-43 and TRAF-44 open |

## Calibration doctrine

The doctrine is one line:

```text
envelope = ceiling(port, architecture) x efficiency(stack, transfers across architectures)
```

The two hardware envelope studies were run on different NVLink generations,
link counts, channel counts and host architectures, and they separate the two
factors cleanly.

- **Ceilings do not transfer.** Per-GPU egress moves from 300 GB/s on the A100
  `NV4` mesh to 450 GB/s on the GH200 `NV6` mesh, exactly 1.5 times. The host
  port moves from 31.5 GB/s of PCIe generation 4 by 16 to 450 GB/s of C2C, and
  changes from symmetric to asymmetric by a factor 2.47.
- **Efficiencies largely transfer.** Ring all-reduce reaches 71.0 percent of
  per-GPU egress at width 4 on the A100 and 74.9 percent on the GH200, 3.9
  percentage points apart across a link generation. Width scaling is closer
  still: widening from two ranks to four multiplies bus bandwidth by 2.925 on
  the A100 and 2.926 on the GH200. Copy-engine wire efficiency is the one that
  moved, 94.0 percent on NVLink3 against 88.8 percent on NVLink4, and it moved
  uniformly across all three measured patterns, so it is a property of the link
  generation rather than of one measurement.
- **Small-message floors are launch bound, not wire bound.** An 8-byte
  all-reduce carries no meaningful serialization at any wire rate, and its
  1.53 times improvement from A100 to GH200 tracks the 1.38 times faster kernel
  launch rather than the link.

The second part of the doctrine is that stack efficiency is a curve, not a
scalar, and that the repository has already refuted its own first two attempts
at compressing it.

- A two-parameter model anchored at the measured 8-byte floor and the 1 GiB
  algorithm bandwidth is exact at both anchors and optimistic at every payload
  between them on both machines, worst at -50.8 percent (A100, width 2, 1 MiB)
  and -48.1 percent (GH200, width 2, 1 MiB). A near-perfect R-squared
  accompanies the artifact both times, so fit quality does not detect it.
- The first regime-aware candidate,
  [collective_regime_curve_v1](../../examples/collective_regime_curve_v1/RESULTS.md),
  scored `16 of 20` and is refuted. The cause is a genuine finding: measured
  serialization bandwidth is **not monotone in payload**. It dips 26 percent on
  the A100 and 22 percent on the GH200 at exactly 1 MiB at width 2, and about
  7 percent at 2 MiB at width 4, so any interpolation between anchors that
  straddle the dip predicts a faster collective than the hardware delivers, in
  exactly the payload band where tensor-parallel activation exchanges live.
  `CollectiveBandwidthCurve` landed as the substrate and is inert: no shipped
  profile carries a curve, so no reported TTFT or TPOT moved.

The third part, which is really the safety rule, is that a port with no
measured or declared profile fails closed. The repository already enforces this
shape elsewhere: calibrated B100 and H100 host-cost requests are rejected during
configuration rather than borrowing the measured Turing constant. A modeled
xGMI port must behave the same way, because the only figures available for it
today are vendor nameplate.

## What exists today, and what is registered as a gap

Landed and usable now: the modular RNIC device with typed ports (BACK-18), the
shared PCIe transaction model (BACK-10), tracked virtual host memory (BACK-19),
three submission shapes with GPU producer coupling (BACK-20, BACK-27), the ABI
v2 packet and transport-control vocabulary (BACK-25, BACK-26), the GPUDirect
data-region placement exercised by the accepted BACK-20 artifact, the flat
NVLink egress cursor and ring egress kernel (COMP-11), the per-direction
copy-engine service that the measured peer efficiencies calibrate, the analytic
locality split
(TRAF-10) with `max(egress, ingress)` endpoint load (CORE-41), and the
name-mirrored NCCL stack skeleton on one virtual clock (first slice of
COMP-15).

Registered by this document:

| Task | Gap it owns |
|---|---|
| COMP-34 | The GPU has no device composition entry point and no typed port objects over its existing link mechanisms, so no protocol identity, declared capabilities, ceiling provenance or shared packet vocabulary. Peer topology, ingress service and reduction lanes stay with COMP-31. |
| COMP-35 | No vendor port instantiation exists, so an AMD ROCm GPU cannot be expressed at all and an xGMI ceiling has no first-party or declared profile to fail closed against. |
| BACK-46 | The GPUDirect leg is already expressible and already exercised, with `data_endpoint` as `gpu_memory` in the accepted BACK-20 artifact, but the region is owned by the reading NIC: no separately modeled GPU attaches to the shared fabric, claims its own regions, or has its transactions accounted apart from the NIC's. |
| BACK-47 | The mirrored NCCL stack boundary is not named as the plugin ABI seam, and its packet-emission half toward the GPU is unregistered while the half toward the NIC stops at zero-time events. |
| BACK-48 | The ABI v2 packet vocabulary is reachable only through a wire port, so a non-wire port cannot emit an attempt, a TX boundary or an arrival in the same language. |
| TRAF-45 | The intra-node leg has no packetized path behind the analytic locality off path, and the ingress term of a converging combine is still owned elsewhere (CORE-48 cross-node, COMP-31 local mechanism). |

## Open decision point, deliberately not decided here

**Does a dedicated `simllm.device` module get carved out when this lands?**

This document does not decide it, and the decision must be made explicitly in
the change that implements COMP-34 rather than by accident.

The case for carving it out: port, packet and device-composition types would
then have one home instead of being duplicated between
`simllm/backends/rnic/` (C++, NIC-side, already device shaped) and
`simllm/compute/` (Python, GPU-side, currently profile shaped). A shared home
is also the natural place for the capability negotiation that lets one packet
vocabulary serve ports with different control features.

The case against: this repository's module rule is deep implementation behind a
narrow existing interface rather than a new parallel surface, and the working
seams already exist (`ComputeProvider`, the placement and fabric manifests,
`NetworkPort`, `PcieFabric`). A new top-level module is a wide shallow surface
until at least two ports of different kinds actually need it, and the
framework-independence ground rule in
[CONTRIBUTING.md](../../CONTRIBUTING.md) constrains where the types may live.

What decides it: whether COMP-34's port objects can be expressed inside the
existing compute and backend surfaces without either duplicating the packet
vocabulary or importing across the Python and C++ boundary in a new direction.
That question is answerable only with the first port implementation in hand, so
the implementing change records the answer with its reasoning, and this section
is updated to state which way it went.

## Sources for external claims

Every external claim in this document, with the source it rests on. Repository
claims cite the study or header inline above instead.

| Claim | Source |
|---|---|
| NCCL loads `libnccl-net.so`; communication uses `isend`, `irecv` and `test`; `regMr` is called on all buffers "to allow RDMA NICs to prepare buffers"; the README is written against `ncclNet_v11`; device offload is requested through a valid `*sendDevComm` or `*recvDevComm` | NVIDIA, NCCL net plugin README, https://github.com/NVIDIA/nccl/blob/master/plugins/net/README.md |
| The audited NCCL release checkout carries the same plugin member names at `src/include/plugin/net/net_v12.h`; NCCL is not a submodule of this repository | NCCL release `v2.30.7-1`, commit `73cf112295c33aee2b895f329f592f2a9b4b0f97`, audited in [compute.md](../modules/compute.md#nccl-stack-name-audit) |
| `NCCL_P2P_DISABLE` disables "the peer to peer (P2P) transport, which uses CUDA direct access between GPUs, using NVLink or PCI"; `NCCL_SHM_DISABLE` disables the shared-memory transport, which "is used between devices when peer-to-peer cannot happen, therefore, host memory is used" | NVIDIA, NCCL environment variables, https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html |
| NCCL "supports a variety of interconnect technologies including PCIe, NVLINK, InfiniBand Verbs, and IP sockets" | NVIDIA, NCCL overview, https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html |
| GPUDirect RDMA "enables a direct path for data exchange between the GPU and a third-party peer device using standard features of PCI Express" | NVIDIA, GPUDirect RDMA documentation, https://docs.nvidia.com/cuda/gpudirect-rdma/ |
| RCCL "is a stand-alone library that provides multi-GPU and multi-node collective communication primitives optimized for AMD GPUs. It uses PCIe and xGMI high-speed interconnects" (documented version 2.30.7) | AMD, RCCL documentation, https://rocm.docs.amd.com/projects/rccl/en/develop/index.html |
| RCCL's plugin page is titled "Using the NCCL Net plugin API", describes asynchronous `isend`, `irecv` and `test` with `regMr` buffer registration, and names the plugin library `librccl-net.so` | AMD, RCCL, Using the NCCL Net plugin API, https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/using-nccl.html |
| xGMI nameplate figures for MI300X: up to 64 GB/s per point-to-point link and 448 GB/s aggregate per GPU over seven links, with 45 to 48 GB/s per link and 315 to 336 GB/s aggregate reported as realized | AMD ROCm blog, "Understanding RCCL Bandwidth and xGMI Performance on AMD Instinct MI300X", Kolla, Alizadeh and Lee, 2 March 2025, https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html |
