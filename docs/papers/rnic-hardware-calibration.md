# RNIC hardware calibration and boundary plan

This is the public evidence and calibration plan for the SimLLM RNIC hardware
model. The numbered implementation tasks are BACK-8 through BACK-15 and
HTSIM-5, HTSIM-6 and HTSIM-9 in
[the backend module registry](../modules/backends.md). Source PDFs are kept in
the gitignored `papers/` directory; this file records what may be claimed from
them and how measurements will be used.

## Architecture decision

RNIC hardware and transport/congestion control are independent model axes.
The structural hardware model is SimLLM-owned C++ under
`simllm/backends/rnic/`. It is compiled into the directly invoked simulator
process so the packet event loop has no Python callback. htsim retains the
selectable `rnic-nn`, `rnic-cn` and DCQCN policies and the packet fabric.

```text
ibverbs capture or GOAL lowering
  -> SimLLM WR/WQE, WQ, QP/QPC, PCIe/DMA and TX hardware
  -> versioned htsim policy/fabric port
  -> htsim CC decision, packet queues, ECN, loss and PFC transport
  -> SimLLM RX, reliability, payload DMA and CQ/CQE hardware
  -> poll, completion channel or interrupt
```

Every full-RNIC comparison uses the same hardware configuration hash and
changes only the policy. `rnic-nn-fluid` retains an explicit hardware bypass
for closed-form validation. The boundary carries opaque flow and packet
tokens, packet metadata, transmit eligibility, delivery, ECN/CNP, drop,
pause and link events. WQ, CQ, QP, QPC, PCIe and DMA objects do not cross it.

State ownership is deliberately narrow:

- SimLLM owns WR/WQE/CQE contents, SQ/RQ/SRQ/CQ service, QP lifecycle and
  pairing, PSN and reliability state, context and translation caches,
  MMIO/PCIe/DMA, packetization and reassembly, TX/RX arbitration, the
  hardware rate gate, PFC watermarks/gates and completion delivery.
- An htsim policy owns only its algorithm state. DCQCN owns alpha,
  current/target rate, CNP suppression and recovery timers. `rnic-cn` owns
  reservation, control-slot and predeclaration state. Policy updates drive
  the SimLLM hardware rate gate.
- The htsim fabric owns links, switch queues, ECN marking, propagation,
  wire/switch drops and PFC-frame transport. A SimLLM RNIC originates or
  consumes a PFC frame from its modeled per-priority buffers.

This split prevents a DCQCN result from paying a different doorbell, cache or
DMA cost than `rnic-nn` or `rnic-cn`. It also permits one hardware calibration
to be reused when a later policy is added.

## Corrections to the initial mental model

Four refinements are required by the public evidence:

1. **Per-WQE start is a timeline, not a constant.** `ibv_post_send` may
   publish a linked WR list with one doorbell, the provider may inline a WQE
   through BlueFlame, and the NIC may fetch and overlap several WQEs. The
   structural model records posting, publication, observation, fetch,
   context readiness, admission and packet issue separately. A fitted fixed
   latency remains useful only as a reduced-form diagnostic.
2. **TCP is a pairing option, not RoCE data transport.** A manual program may
   exchange QPN, PSN, GID and path attributes over a TCP socket before QP
   transitions, while `rdma_cm` or IB-CM provides another control path. TCP
   carries RDMA data only for iWARP. Both control-plane pairers feed the same
   QP state machine.
3. **The three-tier context hierarchy is generic.** The model exposes
   `on_die_sram`, optional `device_memory` and `host_pinned_memory`. Public
   evidence supports an internal context cache with host ICM backing over
   PCIe, but does not establish a ConnectX-7 HBM tier or its geometry. The
   optional middle tier is disabled in the CX-7 profile until a documented
   or measured source justifies it.
4. **Collie supplies search dimensions and reproducer seeds.** Its public
   Mellanox cases are based on ConnectX-6, omit switch-loss experiments and
   cannot publish several NDA diagnostic counters. Its anomaly thresholds
   are not copied into a CX-7 profile as constants.

## Truth and observability contract

Every CX-7 model field carries one evidence class:

| Class | Meaning | Permitted use |
|---|---|---|
| `documented` | Defined in a public specification, driver interface or vendor manual | Reproduce the named field, units, access and documented semantics |
| `driver-inferred` | Behavior follows from reviewed Linux mlx5 or rdma-core source but lacks a public silicon contract | Implement the observed software-visible behavior and retain source/version provenance |
| `calibrated-opaque` | Internal stage is not public but its aggregate effect is identifiable from controlled measurements | Use a named latent parameter with confidence bounds; do not expose a fabricated physical register |

The register model is therefore a versioned observable-state facade, not a
claim to reproduce a private CX-7 register map. Each field records logical
name, units, width, access, reset/snapshot behavior, collection command,
firmware/PSID, kernel, rdma-core and MFT versions, PCIe topology, timestamp and
evidence class. Unsupported physical addresses, cache associativity,
scheduler registers and firmware-private state remain absent.

| Surface | What can anchor the model | Important limit |
|---|---|---|
| rdma-core mlx5 provider | WR validation, WQE construction, DB record, UAR/BlueFlame publication, CQ polling and CQE decoding | This is user-space fast-path evidence, not a silicon timing specification |
| Linux mlx5 RDMA driver | QP/CQ/MR creation and modification, firmware command fields, resource dumps and hardware-counter plumbing | Normal send/receive posting bypasses the kernel |
| Linux network/devlink/DCB | `ethtool -S`, RDMA hardware counters, resource/statistic views, health reporters, PFC configuration and pause counters | Availability depends on kernel, firmware and enabled counter groups |
| PCIe and platform telemetry | Link generation/width, MPS/MRRS, NUMA path, IOMMU state, AER and device locality | It exposes the path, not private RNIC scheduling |
| NVIDIA MFT/DOCA telemetry | Supported named registers, resource dumps and device counters | Only fields actually returned on the tested PSID/firmware may enter the facade |

Primary software references are the rdma-core mlx5
[QP posting path](https://github.com/linux-rdma/rdma-core/blob/master/providers/mlx5/qp.c)
and [CQ polling path](https://github.com/linux-rdma/rdma-core/blob/master/providers/mlx5/cq.c),
the Linux mlx5 [RDMA QP control path](https://github.com/torvalds/linux/blob/master/drivers/infiniband/hw/mlx5/qp.c),
[counter documentation](https://docs.kernel.org/networking/device_drivers/ethernet/mellanox/mlx5/counters.html)
and [devlink interface](https://docs.kernel.org/networking/devlink/mlx5.html).

## ibverbs capture and replay hook

The capture point must reflect how mlx5 applications really submit work:

1. Capture QP, CQ, SRQ, PD and MR creation/modification at the provider and
   kernel control paths. Record returned resource identities, QP transitions,
   path attributes, retry/RNR settings and memory mappings.
2. Instrument the rdma-core mlx5 provider data path around send/receive WR
   validation, WQE construction, the publication barrier, DB-record update,
   UAR/BlueFlame doorbell and CQ polling. This sees linked WR lists and
   provider-specific fast paths that a kernel probe cannot see.
3. Keep a generic `LD_PRELOAD` verbs wrapper as a convenient application
   experiment, not the signoff oracle. Direct operation tables, inlining and
   provider extensions can bypass it.
4. Normalize capture and SimLLM lowering into one BACK-9 schema. Preserve WR
   ID, opcode, SGEs, flags, queue identity, QP state, batch position and all
   observable timestamps. Payload bytes are excluded by default.
5. Correlate CQ polling to the normalized CQE. `wr_id` is normally recovered
   by the provider from its WQ metadata rather than read as a literal raw CQE
   field, so the schema records both the raw-source fields and provider-derived
   fields.

The manual pairer follows the normal RC example pattern: establish a host TCP
connection, exchange QPN/PSN/GID/path data, then apply RESET to INIT to RTR to
RTS transitions. The alternate pairer records `rdma_cm` events. Both produce
the same normalized pairing record and both cover timeout, rejection,
attribute mismatch and teardown.

## Structural RDMA Work Queue

The landed `AtlahsWqeLedger` is only a timing-neutral compatibility view. The
replacement is a queueing model with explicit transactions and credits:

| Object/stage | Required behavior |
|---|---|
| WR and WQE | Linked WR lists, opcode/SGE/fence/inline/signaling semantics, WQEBB use, batch position and stable transaction identity |
| SQ | Finite host ring, producer and NIC consumer, wrap/reclamation, doorbell batching, DB records, UAR and BlueFlame paths |
| RQ and SRQ | Posted receive buffers, matching/consumption, RNR behavior, finite depth, replenishment and sharing |
| CQ | Finite host-memory ring, producer/consumer and owner phase, requester/responder/error formats, 64/128-byte profiles, moderation, compression, overrun, polling and completion channels |
| PCIe and DMA | Separate queues and credits for MMIO, WQE/context/translation/payload reads, payload/CQE writes, completions, commands and interrupts |
| TX and RX | Context readiness, packetization/reassembly, arbitration, policy rate gate, PFC gate, port queues, ACK/NAK/retry and error transition |

The normalized CQE contains WR ID, QPN or source QP, opcode, status, byte
count, immediate/invalidate data, flags, syndrome and vendor syndrome, with
provenance for every provider-derived field. Unsignaled sends retire SQ space
without requiring one CQE per WQE; receives and errors still follow their
documented completion rules.

Each WQE records at least:

```text
posted_at
doorbelled_at
doorbell_seen_at
wqe_fetch_begin_at
wqe_fetch_end_at
qpc_ready_at
admitted_at
first_packet_at
last_packet_at
transport_retired_at
cqe_visible_at
polled_at
```

Stages that a selected path bypasses, such as a WQE DMA fetch after a
BlueFlame copy, are explicitly `not_applicable`, never silently zero.
`first_packet_at` is the defined NIC start. The other timestamps allow a
calibration to identify whether a boundary came from CPU posting, MMIO, DMA,
context locality, hardware scheduling, transport or CQ service instead of
fitting all of them into one number.

## Evidence corpus

| Source | Local PDF and SHA-256 | Use in this model |
|---|---|---|
| [Collie, NSDI 2022](https://www.usenix.org/conference/nsdi22/presentation/kong) | `papers/collie-nsdi22.pdf`, `6ad0c5418193ad65cbe08b9472a5631506c6a9ec1ac5a20abdd1d8433172039e` | Anomaly-search dimensions and Mellanox reproducer seeds |
| [Understanding the Microarchitecture of Modern RNICs, NSDI 2023](https://www.usenix.org/system/files/nsdi23-kong.pdf) | `papers/rdma-microarchitecture-nsdi23.pdf`, `ce026cfaa7a25eaa3b585edaa26078e4300bb145837f8fc48641c9502b1ef2d9` | Context/WQE/translation caches, data-path boundaries and cache-pressure experiments |
| [SRNIC, NSDI 2023](https://www.usenix.org/system/files/nsdi23-wang-zilong.pdf) | `papers/srnic-nsdi23.pdf`, `6c03f49a630babef65cf8a42bd3d671b3af9edff77b408bd26c96eb6691808c4` | Structural QP/WQE/PCIe model and scale thought experiments, not CX-7 constants |
| [Design Guidelines for High Performance RDMA Systems, ATC 2016](https://www.usenix.org/system/files/conference/atc16/atc16_paper-kalia.pdf) | `papers/kalia-atc16-rdma-guidelines.pdf`, `f69540e83bfc79e1885fd4ed9b64239fb409c5618627694452cf3fbe03c918fb` | Doorbell/BlueFlame, batching, inline and completion-signaling mechanics |
| [Understanding PCIe performance for end host networking, SIGCOMM 2018](https://web.stanford.edu/class/cs244/papers/neugebauer-sigcomm18.pdf) | `papers/pcie-end-host-sigcomm18.pdf`, `f535d231cbe334f8ee136ec9a124de743b1f76b025c558d9cff756561ed0322e` | PCIe transaction, concurrency, MPS/MRRS, NUMA and IOMMU experiments |
| [DCQCN, SIGCOMM 2015](https://www.microsoft.com/en-us/research/publication/congestion-control-for-large-scale-rdma-deployments/) | `papers/dcqcn-sigcomm15.pdf`, `879074a33b78ceb93f9f66d59b217fba9e7621320cc68bc47914747cf5cf31f8` | Persistent per-QP DCQCN state and recovery parameters |
| [HPCC, SIGCOMM 2019](https://arxiv.org/abs/1903.10924) | `papers/hpcc-sigcomm19.pdf`, `8199b81f7325b8797623b6c44fad90eb2664b4bc6a8e0f9bdbad7e043b02fe8a` | DCQCN timer/ECN sensitivity and PFC tradeoffs |
| [UCCL](https://arxiv.org/abs/2504.17307) | `papers/uccl-2504.17307.pdf`, `5a6fe7e2735f972bfcc3c5d1c501098989c908a5c8adf9171fd017829d178994` | ConnectX-7-class message-size, outstanding-work and loss anchors |

Collie varies operation, payload, MTU, QP count, WQ depth, batch, SGE/MR
layout, direction, concurrency, host memory and PCIe placement. Those
dimensions seed the campaign below. They are expanded with explicit loss,
control-plane pairing, CQ service, GPU Direct and CX-7 provenance.

## Pre-registered boundary campaign

Each campaign gets its own frozen `expectations.md` before execution. Every
claim sweeps at least two parameters, retains raw results outside Git under
`/data3/yifeng/`, and reports both median behavior and tail/first-failure
evidence.

| Campaign | Minimum sweep | Expected invariant or boundary |
|---|---|---|
| WR/WQE publication | payload `{32 B, 4 KiB, 32 KiB, 256 KiB, 4 MiB}` x outstanding `{1, 4, 16}` x doorbell batch `{1, 4, 16}` | MMIO events per WQE fall with batching; useful goodput approaches line rate with enough independent work; a queue/cache knee must be localized rather than hidden in one latency |
| CQ service | CQ depth `{64, 1024, 16384}` x signaling interval `{1, 16, 64}` x busy-poll cadence | Unsignaled operation reduces CQE DMA and polling work; producer never overwrites an owned entry without an explicit overrun/error; normalized completions preserve WR identity |
| QP/QPC and translation locality | active QPs `{1, 16, 256, 4096, 16384}` x sequential/round-robin reuse x MR page size `{4 KiB, 2 MiB}` | Hot locality is flat until capacity pressure; misses add attributable PCIe/context service; QPC and MTT/MPT knees need not coincide |
| PCIe/MMIO/DMA | local/remote NUMA x CPU/GPU memory x observed MPS/MRRS and outstanding-read settings | Bytes and latency reconcile per transaction class; remote/IOMMU paths cannot outperform an otherwise identical local/bypass path without measured evidence |
| TX/RX loss | injection location `{pre-TX, wire/switch, pre-RX}` x deterministic `1/256`, Bernoulli `{1e-6, 1e-4, 1e-2}` and burst `{1, 8, 64}` | Controlled drop count is exact for deterministic runs; first missing packet and first affected WQE remain identifiable; retry, timeout and completion status depend on location and opcode |
| PFC | incast fan-in `{2, 8, 32}` x headroom `{0.5, 1, 2}` times post-XOFF in-flight bytes x path RTT `{1, 5, 20 us}` | Adequate headroom prevents priority-buffer loss; insufficient headroom produces an attributed RX drop; pause duty cycle and blocked unrelated traffic are reported separately from DCQCN |
| DCQCN first | fan-in and background load x ECN thresholds x timer/rate parameter sets | A new QP begins at line/local-QoS rate before feedback; a CNP changes persistent per-QP state seen by later WQEs; changing CC never changes hardware MMIO/DMA/cache accounting |
| `rnic-cn` lookahead | same-destination queue depth x available grant x payload | Established sufficient grants avoid repeated setup; bounded lookahead hides later declaration behind current transfer without moving WQ/QPC state into htsim |

The loss ledger uses three evidence tiers: `controlled` when the injector names
the transaction, `asserted` when an invariant or model assertion identifies
the first loss, and `inferred` only when counters bracket the event. A run is
not accepted if it reports only aggregate goodput while losing transaction
identity.

Core metrics are WQE stage latency, per-flow FCT, phase makespan/JCT,
TTFT/TPOT, useful and raw PCIe/wire bytes, queue occupancy, context and
translation misses, retry/RNR/timeout, CQE count and age, CNP/ECN state, pause
duration and drops by named boundary. Physical-policy FCT is also normalized
to the identical `rnic-nn` GOAL where starts are aligned.

## Work order and closure

1. BACK-13 and BACK-14 establish the provenance schema and capture path early,
   so implementation parameters come from evidence rather than retrospective
   fitting.
2. BACK-8 and HTSIM-9 establish the C++ hardware/policy ABI and prove that all
   full-RNIC profiles share one hardware configuration.
3. BACK-9 and BACK-10 implement WQ/CQ plus PCIe/MMIO/DMA queues and close the
   no-loss message-size and batching anchors.
4. BACK-11 adds QP pairing, lifecycle and separate QPC/WQE/translation
   locality; its cache tiers remain generic and evidence-labelled.
5. BACK-12 adds TX/RX reliability, named loss boundaries and PFC hardware.
   HTSIM-5 closes persistent DCQCN state first, then HTSIM-6 closes `rnic-cn`
   lookahead.
6. BACK-15 runs the full cross-layer campaign. Closure requires defended
   queue knees and first-failure evidence, not only passing unit tests or a
   visually similar bandwidth curve.
