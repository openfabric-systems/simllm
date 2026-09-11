# GPU communication article: source review for the channel FIFO model

The linked article provides the architectural direction for the next NCCL
model: represent dependent GPU work and finite communication buffers explicitly.
Its illustrative constants do not calibrate A100 or GH200. This review connects
that direction to the pinned source and the
[implementation and identification plan](../design/nccl-channel-fifo-model.md).

## Article and retrieval

Source: [NVLink、NCCL、NVSHMEM 和 GIN：GPU 之间到底是怎么说话的](https://zhuanlan.zhihu.com/p/2081533393758699788),
retrieved on 2026-09-11. The direct page returned an anti-bot response; the
article body was read through a public text reader. Author and publication
date were not exposed in that response and are not inferred. The external
evidence archive retains the response and retrieval manifest, with SHA-256
`d4646ce2a074aba37e143e65d3bfd348564a55b93614f50613eb49c41b170472`.
The article is secondary explanation, not a controlled hardware experiment.

## Summary of the article

The article separates the physical interconnect from the software organizing
communication. NVLink carries peer-memory operations; NVSwitch connects those
links. NVIDIA Collective Communication Library (NCCL) chooses collective
algorithms, protocols, and parallel channels. NVSHMEM exposes one-sided
operations from GPU code. GPU-Initiated Networking (GIN) adds network operations
to NCCL's device interface, with either a direct or CPU-assisted backend.

It proposes expanding collectives into a directed acyclic graph of dependent
transfers, with protocol formatting, finite first-in, first-out (FIFO) buffers,
and distinct initiation paths. Channels consume GPU resources, affecting both
communication and overlapping computation. Directional bandwidth and benchmark
accounting need separate treatment. Suggested diagnostics include latency,
bandwidth, protocol selection, FIFO occupancy, slot-reuse waits, and GPU
resource use. Dynamic expert communication adds network request-rate limits.

## Source authority and scope

The implementation authority for the Merlin follow-up is NCCL 2.31.2, commit
`7b83616df3ae082a1f32bb74c27458bfe8153a13`. The benchmark authority is nccl-tests
commit `b4d5beebca8a76cf01335f724d154b9b9d394d96`. These are the retained capture
identities, not a claim about whichever versions are newest. The earlier
[protocol source audit](../../examples/nccl_protocol_model_v1/source_audit.md)
records the existing model and transport boundaries.

| Adopted mechanism | Pinned primary authority | Modeling consequence |
|---|---|---|
| Ring send, reduce-forward, copy-forward, final receive | [`all_reduce.h`, `runRing`](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/device/all_reduce.h) | Expand actual primitive dependencies and partial final chunks |
| Protocol line layout and eight software steps | [`device.h`](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/include/device.h) | Keep useful bytes, readiness bytes and slot sequence distinct |
| Configured buffer sizes | [`init.cc`, `computeBuffSizes`](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/init.cc) | Read effective communicator sizes, including overrides |
| LL readiness and reuse | [`prims_ll.h`](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/device/prims_ll.h) | Represent flag polling, head-counter gating and wrap cleanup |
| LL128 work and readiness | [`prims_ll128.h`](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/device/prims_ll128.h) | Preserve active-lane work, barriers and conditional publication |
| Simple wait, data movement and publication | [`prims_simple.h`](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/device/prims_simple.h) | Separate data readiness, buffer reuse, fence and counter operations |
| Work partition and thread/channel selection | [`enqueue.cc`](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/enqueue/enqueue.cc), [`tuning_general.cc`](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/tuning/tuning_general.cc) | Reproduce selected geometry before fitting any service cost |
| Peer-buffer location and direct access | [`p2p.cc`](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/transport/p2p.cc) | Capture actual connection mode; do not invent an extra buffer copy |

LL means low latency. In the pinned source, one LL software line contains two
4-byte data words and two 4-byte flags: 16 encoded bytes for 8 useful bytes.
LL128 carries 120 useful bytes and an 8-byte flag in a 128-byte software line.
Their encoding efficiencies are exactly 1/2 and 15/16 before padding and other
traffic. These are software-buffer formats, not NVLink packet headers.

Default sizes derived from `device.h` and `computeBuffSizes` are:

| Protocol | Encoded allocation per connection buffer | Encoded bytes per software slot | Useful capacity per slot |
|---|---:|---:|---:|
| LL | 512 KiB | 64 KiB | 32 KiB |
| LL128 | 4,800 KiB | 600 KiB | 576,000 bytes, or 562.5 KiB |
| Simple | 4 MiB | 512 KiB | 512 KiB |

For example, LL allocates `8 * 512 * 8 * 16 = 524288` bytes. Its whole FIFO
holds 256 KiB of useful data. A statement of "256 KiB LL buffer" therefore
needs its version and useful-versus-encoded convention; it is not the encoded
allocation in our pinned source. A slot, a protocol slice, and an algorithm
chunk are separate units. Simple's Ring specialization uses two slices per
chunk and advances two software steps per slice; the last slice can be empty.
Source: [`all_reduce.h`](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/device/all_reduce.h)
and [`collectives.h`](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/include/collectives.h).

## Corrections before adopting the article's examples

**Selection estimates and execution times have separate roles.** NCCL's
software cost table selects a path; it does not identify a physical round trip
or a GPU polling cost. Replay the pinned
[`ring.cc`](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/tuning/ring.cc)
selection with captured communicator inputs. The article's approximate
1-MB table labels Tree/LL128 at 30 microseconds as the winner while listing
Ring/LL at 27 microseconds. Even its illustrative winner is therefore not an
exact minimum of the displayed row. No payload threshold or latency from that
example becomes a profile value.

**Logical channels do not identify physical planes or occupied SMs.** A
streaming multiprocessor (SM) is a GPU execution unit. CUDA thread blocks are
scheduled onto available SM resources; channel count alone does not establish
their placement, residency or physical-link mapping. The pinned
[`common.h`](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/device/common.h)
maps block IDs to logical channels; selection changes with message size.
Record the block resource footprint and actual topology separately. NVIDIA's
[CUDA execution guidance](https://docs.nvidia.com/cuda/archive/12.1.0/ampere-tuning-guide/index.html)
describes the finite warp, register and shared-memory limits. A one-channel to
one-NVLink-plane assumption is not a valid default.

**Buffer location and protocol support depend on transport.** The pinned
`p2pSendConnect` maps device buffers for the peer path, including LL; LL is not
universally a host-memory protocol. Do not transfer a network-transport rule
to local NVLink. Likewise, validate LL128 support against the selected library
and connection: the pinned
[`cost_model.cc`](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/tuning/cost_model.cc)
uses platform and topology checks. The software's 128-byte line size alone
does not establish general 128-byte atomicity.

**Waiting is conditional and runs inside the existing GPU kernel.** LL and
LL128 check reusable-buffer head counters as well as inline readiness flags.
Simple checks peer progress, executes barriers, and publishes counters with
the source's fence predicate. Empty slice bodies still advance protocol state.
Some LL128 tail/fence work is conditional on the connection pointer and GPU
architecture. Price the selected branch, not every possible branch. There is
no unconditional new round trip per word and no separately launched polling
kernel in these primitive paths.

**Memory semantics still use packets underneath.** NVLink peer loads and stores
become hardware transactions. The repository's
[public NVLink reconstruction](../design/nvlink-mechanism-reverse-engineering.md)
distinguishes the documented Pascal flit format from unconfirmed later-product
details. A software FIFO slot is not a hardware receive-credit unit. Do not
copy an old packet-efficiency estimate into an A100 physical profile without
generation-specific evidence.

**Benchmark bandwidth is an accounting convention.** For Ring all-reduce with
payload `S`, participants `n` and time `T`, nccl-tests reports application
bandwidth `S/T` and adjusted bus bandwidth `S/T * 2(n-1)/n`. The latter does
not count every flag, control request or physical header. For other collectives,
first establish whether `S` is the input, output or per-rank contribution.
Source: [NVIDIA benchmark performance definitions](https://github.com/NVIDIA/nccl-tests/blob/b4d5beebca8a76cf01335f724d154b9b9d394d96/doc/PERFORMANCE.md).

**Device initiation does not always remove the CPU from progress.** GIN has
both direct and proxy backends. The current
[NCCL device API documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html)
distinguishes the device API's 2.28 introduction from GIN availability since
2.28.7 and gives version-specific backend requirements. Network queue and
doorbell timing do not belong on the local peer-memory path merely because
the host launched the collective. NVSHMEM and GIN are architectural context
for this task, not alternative calibration targets for A100 all-reduce.

## Check the cited experiments before using their numbers

| Primary reference | Controls actually stated | Remaining limit and use here |
|---|---|---|
| Hu et al., [Demystifying NCCL, v3](https://arxiv.org/html/2507.04786v3), introduction, Figure 6 and section V-F | Source analysis targets NCCL 2.19.1. Alps GH200 system, intra/inter-node arms, algorithm/protocol comparisons, 20 runs per point and a warm-up phase are stated. | Benchmark binary digest, exact warm-up count, GPU clock control and the timing boundary are not supplied in those descriptions. The text identifies Slingshot but later invokes RoCE in an explanation; that explanation needs transport verification. Useful mechanism and qualitative reference, not a matched A100 curve. |
| Bachan et al., [GPU-Initiated Networking for NCCL](https://arxiv.org/html/2511.15076), section V and Table III | EOS eight-H100 nodes, 400-Gbit/s InfiniBand interfaces, NCCL 2.28, NVSHMEM 3.4.5, DeepEP 1.2.1 and 24 SMs for the DeepEP experiments are stated. Small-message timing is a put-with-signal ping-pong round trip. | GPU clock locking, exact repetition/warm-up counts and full timer configuration are not stated in the inspected evaluation. Its 16.7-microsecond result concerns H100 network round trips, not A100 NVLink or a Simple fence. |
| [NVIDIA NCCL tuning example](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/) | The article exposes a measured-data-to-tuner workflow and explicit algorithm/protocol overrides. | Optimizing the chooser and modeling the selected implementation are different experiments. The example does not establish Merlin's default latency curve. |

Source transparency determines what a paper or vendor example can support;
affiliation alone does not settle experimental comparability. The two papers
identify useful mechanisms and some controls. Neither supplies the matched
A100 four-GPU conditions needed to diagnose our remaining error. The existing
[public comparison ledger](../../examples/nccl_protocol_model_v1/public_comparisons.md)
also separates an NVIDIA benchmark run locally from a result published by
NVIDIA on another machine.

## Effect on the project

TRAF-54 owns the source-derived per-channel FIFO execution path and shared GPU
resource service. TRAF-43 owns its measured collective accuracy and uncertainty
band. COMP-44 retains independently identified host initiation; TRAF-92 retains
product and topology qualification. The linked plan makes those acceptance
obligations concrete. This documentation review runs no new hardware campaign,
changes no runtime behavior and closes none of those tasks.
