# NCCL stack skeleton v1 integration-review amendment

## Amendment status

This expectations-only amendment is triggered by the integration review's
contract-conformance findings, not by an observed study result. It is written
after the original freeze in commit `92888d5`, but before the corrected
implementation and before any corrected study run. The original freeze stays
unchanged in history. The result report must cite both expectation commits and
label the corrected study re-registered after this amendment.

The name and protocol audit uses the shallow, non-recursive NVIDIA NCCL source
checkout at release tag `v2.30.7-1`, commit
`73cf112295c33aee2b895f329f592f2a9b4b0f97`. SimLLM mirrors names and call
shape only. It copies no NCCL source code.

## Frozen name mapping

Every function identity in the corrected event stream is either a symbol from
the audited release or explicitly prefixed `simllm` and documented as an
invented observation boundary.

| Event function | NCCL source or explicit SimLLM reason |
|---|---|
| `ncclCommInitRank` | `src/init.cc`, `ncclCommInitRank` |
| `ncclBuildRings` | `src/graph/rings.cc`, `ncclBuildRings` |
| `initChannel` | `src/channel.cc`, `initChannel` |
| `ncclAllReduce` | `src/collectives.cc`, `ncclAllReduce` |
| `ncclEnqueueCheck` | `src/enqueue.cc`, `ncclEnqueueCheck` |
| `scheduleCollTasksToPlan` | `src/enqueue.cc`, `scheduleCollTasksToPlan` |
| `calcCollChunking` | `src/enqueue.cc`, `calcCollChunking` |
| `ncclProxySaveOp` | `src/proxy.cc`, `ncclProxySaveOp`; upload call in `src/enqueue.cc` |
| `ncclLaunchKernel` | `src/enqueue.cc`, `ncclLaunchKernel` |
| `ncclKernelMain` | `src/device/common.h`, `ncclKernelMain` |
| `runRing` | `src/device/all_reduce.h`, `runRing` |
| `waitPeer` | `src/device/prims_simple.h`, `waitPeer` |
| `genericOp` | `src/device/prims_simple.h`, `genericOp` |
| `postPeer` | `src/device/prims_simple.h`, `postPeer` |
| `ncclProxyProgress` | `src/proxy.cc`, `ncclProxyProgress` |
| `sendProxyProgress` | `src/transport/net.cc`, `sendProxyProgress` |
| `ncclNet.isend` | `src/include/plugin/net/net_v12.h`, `isend` member; called in `src/transport/net.cc` |
| `ncclNet.test` | `src/include/plugin/net/net_v12.h`, `test` member; called in `src/transport/net.cc` |
| `wrap_ibv_post_send` | `src/include/ibvwrap.h`, `wrap_ibv_post_send`; called by `src/transport/net_ib/p2p.cc` |
| `wrap_ibv_poll_cq` | `src/include/ibvwrap.h`, `wrap_ibv_poll_cq`; called by `ncclIbTest` in `src/transport/net_ib/p2p.cc` |
| `simllmRnicRingDoorbell` | simllm-invented: exposes the RNIC notification hidden inside the verbs provider's post operation |
| `simllmNetworkComplete` | simllm-invented: deterministic external completion injection until the native RNIC session supplies CQEs |
| `simllmKernelComplete` | simllm-invented: stack-internal observation of GPU kernel completion until runtime projection lands |

The previous names `ncclBuildLogicalChannels`,
`ncclConstructLogicalChannel`, `ncclPlanAllReduce`, `ncclChunkPayload`,
`ncclAssignChunksToChannels`, `ncclLaunchCollectiveKernel`,
`ncclCollectiveKernel`, `ncclCopyChunkToFifo`, `ncclStoreReadyFlag`,
`ncclStoreHead`, `ncclProxyPollHead`, `ncclProxyPollReady`,
`ibverbs.post_send`, `ibverbs.write_cqe`, `ibverbs.poll_cq`,
`ncclProxyStoreTail`, `ncclKernelPollTail`, `ncclReleaseFifoSlot`, and
`ncclKernelComplete` are rejected by the corrected exact sequence.

## Corrected send-connector contract

The send connector follows `src/device/prims_simple.h` and
`sendProxyProgress` in `src/transport/net.cc`:

1. The GPU `waitPeer` path waits on the send connector's `head` before reusing
   a FIFO slot. The first `fifo_slots_per_channel` steps have initial credit
   and do not emit a successful head poll.
2. `waitPeer` publishes the FIFO size or ready state, `genericOp` represents
   the deliberate data-copy boundary, and GPU `postPeer` advances `tail`.
3. The independent send proxy observes `tail` and ready state, calls
   `ncclNet.isend`, posts the send, and rings the RNIC doorbell.
4. Posting never creates a CQE. The separate `simllmNetworkComplete` source
   makes completion visible only after the post and doorbell.
5. A later proxy progression calls `ncclNet.test`, whose verbs path polls the
   CQ. Only after that completion does `sendProxyProgress` clear the ready state
   and advance `head`.
6. The next GPU reuse of that slot emits `waitPeer` as `poll_observes` linked
   to the earlier proxy head signal.

The launch path calls `ncclProxySaveOp` before `ncclLaunchKernel`. Kernel
publication and proxy progression are separate calls. With FIFO depth two, two
GPU tail publications must be observable before the first proxy head advance,
and the channel's measured high watermark must equal two.

Signal-to-poll causal links are enforced by the observer emission API. Like
the unchanged zero-duration clock, this is a by-construction fatal invariant
and is never counted as a behavioral pass.

## Peer encoding and observer scope

The event schema replaces ambiguous `peer_rank` with two nullable fields:
`send_peer_rank` and `receive_peer_rank`. `initChannel` carries both ring
neighbors. Send-connector, proxy, network, and verbs events carry only
`send_peer_rank`. No numeric `value` field encodes either peer.

Each observer accepts a poll producer only when the exact producer event object
was issued by that observer at that sequence. A foreign stack's producer is
rejected even when every serialized identity field happens to match.

## Explicit ring-step planner

The corrected planner matches the decomposition used by
`simllm.compute.nccl.nccl_ring_allreduce_launch`. For world size `W`, payload
`P`, channels `C`, warps per channel `R`, and chunk bytes `K`:

```text
ring_steps = 2 * (W - 1)
step_bytes = P / W
lane_bytes = P / (W * C * R)
chunks_per_lane_per_step = lane_bytes / K
total_chunks = ring_steps * C * R * chunks_per_lane_per_step
wire_bytes = ring_steps * step_bytes
```

The first `W - 1` steps are `reduce_scatter`; the final `W - 1` steps are
`all_gather`. Each chunk records ring step, phase, channel, warp, lane-local
chunk index, byte offset, and byte count. The planner rejects any configuration
where `P` is not divisible by `W * C * R` or `lane_bytes` is not divisible by
`K`. It never creates a short final chunk.

The reference event cases use `W = 4`, `P = 4,096`, `C = 1`, `R = 1`,
`K = 1,024`, and FIFO depth 2. They therefore contain six ring steps and six
chunks, one chunk per step.

The planner sweep fixes `W = 4`, `R = 1`, and `K = 256`, then crosses payload
sizes 4,096, 8,192, and 16,384 bytes with channel counts 1, 2, and 4:

| Payload | Channels | Wire bytes | Steps | Total chunks | Chunks by channel | Chunks per step |
|---:|---:|---:|---:|---:|---|---:|
| 4,096 | 1 | 6,144 | 6 | 24 | `(24)` | 4 |
| 4,096 | 2 | 6,144 | 6 | 24 | `(12, 12)` | 4 |
| 4,096 | 4 | 6,144 | 6 | 24 | `(6, 6, 6, 6)` | 4 |
| 8,192 | 1 | 12,288 | 6 | 48 | `(48)` | 8 |
| 8,192 | 2 | 12,288 | 6 | 48 | `(24, 24)` | 8 |
| 8,192 | 4 | 12,288 | 6 | 48 | `(12, 12, 12, 12)` | 8 |
| 16,384 | 1 | 24,576 | 6 | 96 | `(96)` | 16 |
| 16,384 | 2 | 24,576 | 6 | 96 | `(48, 48)` | 16 |
| 16,384 | 4 | 24,576 | 6 | 96 | `(24, 24, 24, 24)` | 16 |

Configured channel count is configuration-forced. Its nine checks move to the
fatal unscored structural class. The five scored behavioral families remain:

1. exact inter-node sequence, four rank instances;
2. exact intra-node sequence, four rank instances;
3. explicit ring-step structure, nine planner instances;
4. exact total chunk count, nine planner instances; and
5. exact per-channel chunk distribution, nine planner instances.

All five families and all 35 instances must pass.

## Exact inter-node event sequence

Event kinds below use `C` for `call`, `S` for `signal_store`, and `P` for
`poll_observes`. The reference rank `r` sends to `(r + 1) mod 4` and receives
from `(r - 1) mod 4`. Events 0 through 10 are exact:

| Sequence | Function | Kind | Lane | Subject |
|---:|---|---|---|---|
| 0 | `ncclCommInitRank` | C | cpu | null |
| 1 | `ncclBuildRings` | C | cpu | null |
| 2 | `initChannel` | C | cpu | `logical_channel` |
| 3 | `ncclAllReduce` | C | cpu | null |
| 4 | `ncclEnqueueCheck` | C | cpu | null |
| 5 | `scheduleCollTasksToPlan` | C | cpu | null |
| 6 | `calcCollChunking` | C | cpu | null |
| 7 | `ncclProxySaveOp` | C | cpu | `proxy_operation` |
| 8 | `ncclLaunchKernel` | C | cpu | null |
| 9 | `ncclKernelMain` | C | gpu | null |
| 10 | `runRing` | C | gpu | null |

The remaining exact order is frozen below. `cN` identifies chunk and ring step
`N`; `sN` is slot `N mod 2`. Parentheses give subject and value where useful.

```text
11 waitPeer:S c0 s0 (ready_flag=1)
12 genericOp:C c0 s0 (data_fifo_slot=1024)
13 postPeer:S c0 s0 (tail_counter=1)
14 waitPeer:S c1 s1 (ready_flag=1)
15 genericOp:C c1 s1 (data_fifo_slot=1024)
16 postPeer:S c1 s1 (tail_counter=2)
17 ncclProxyProgress:C
18 sendProxyProgress:C
19 sendProxyProgress:P c0 s0 (tail_counter, observes 13)
20 sendProxyProgress:P c0 s0 (ready_flag, observes 11)
21 ncclNet.isend:C c0 s0
22 wrap_ibv_post_send:C c0 s0
23 simllmRnicRingDoorbell:S c0 s0 (doorbell=1)
24 sendProxyProgress:P c1 s1 (tail_counter, observes 16)
25 sendProxyProgress:P c1 s1 (ready_flag, observes 14)
26 ncclNet.isend:C c1 s1
27 wrap_ibv_post_send:C c1 s1
28 simllmRnicRingDoorbell:S c1 s1 (doorbell=1)
29 simllmNetworkComplete:S c0 s0 (completion_queue_entry=1024)
30 simllmNetworkComplete:S c1 s1 (completion_queue_entry=1024)
31 ncclProxyProgress:C
32 sendProxyProgress:C
33 ncclNet.test:C c0 s0
34 wrap_ibv_poll_cq:P c0 s0 (completion_queue_entry, observes 29)
35 sendProxyProgress:S c0 s0 (ready_flag=0)
36 sendProxyProgress:S c0 s0 (head_counter=1)
37 ncclNet.test:C c1 s1
38 wrap_ibv_poll_cq:P c1 s1 (completion_queue_entry, observes 30)
39 sendProxyProgress:S c1 s1 (ready_flag=0)
40 sendProxyProgress:S c1 s1 (head_counter=2)
41 waitPeer:P c2 s0 (head_counter, observes 36)
42 waitPeer:S c2 s0 (ready_flag=1)
43 genericOp:C c2 s0 (data_fifo_slot=1024)
44 postPeer:S c2 s0 (tail_counter=3)
45 waitPeer:P c3 s1 (head_counter, observes 40)
46 waitPeer:S c3 s1 (ready_flag=1)
47 genericOp:C c3 s1 (data_fifo_slot=1024)
48 postPeer:S c3 s1 (tail_counter=4)
49 ncclProxyProgress:C
50 sendProxyProgress:C
51 sendProxyProgress:P c2 s0 (tail_counter, observes 44)
52 sendProxyProgress:P c2 s0 (ready_flag, observes 42)
53 ncclNet.isend:C c2 s0
54 wrap_ibv_post_send:C c2 s0
55 simllmRnicRingDoorbell:S c2 s0 (doorbell=1)
56 sendProxyProgress:P c3 s1 (tail_counter, observes 48)
57 sendProxyProgress:P c3 s1 (ready_flag, observes 46)
58 ncclNet.isend:C c3 s1
59 wrap_ibv_post_send:C c3 s1
60 simllmRnicRingDoorbell:S c3 s1 (doorbell=1)
61 simllmNetworkComplete:S c2 s0 (completion_queue_entry=1024)
62 simllmNetworkComplete:S c3 s1 (completion_queue_entry=1024)
63 ncclProxyProgress:C
64 sendProxyProgress:C
65 ncclNet.test:C c2 s0
66 wrap_ibv_poll_cq:P c2 s0 (completion_queue_entry, observes 61)
67 sendProxyProgress:S c2 s0 (ready_flag=0)
68 sendProxyProgress:S c2 s0 (head_counter=3)
69 ncclNet.test:C c3 s1
70 wrap_ibv_poll_cq:P c3 s1 (completion_queue_entry, observes 62)
71 sendProxyProgress:S c3 s1 (ready_flag=0)
72 sendProxyProgress:S c3 s1 (head_counter=4)
73 waitPeer:P c4 s0 (head_counter, observes 68)
74 waitPeer:S c4 s0 (ready_flag=1)
75 genericOp:C c4 s0 (data_fifo_slot=1024)
76 postPeer:S c4 s0 (tail_counter=5)
77 waitPeer:P c5 s1 (head_counter, observes 72)
78 waitPeer:S c5 s1 (ready_flag=1)
79 genericOp:C c5 s1 (data_fifo_slot=1024)
80 postPeer:S c5 s1 (tail_counter=6)
81 ncclProxyProgress:C
82 sendProxyProgress:C
83 sendProxyProgress:P c4 s0 (tail_counter, observes 76)
84 sendProxyProgress:P c4 s0 (ready_flag, observes 74)
85 ncclNet.isend:C c4 s0
86 wrap_ibv_post_send:C c4 s0
87 simllmRnicRingDoorbell:S c4 s0 (doorbell=1)
88 sendProxyProgress:P c5 s1 (tail_counter, observes 80)
89 sendProxyProgress:P c5 s1 (ready_flag, observes 78)
90 ncclNet.isend:C c5 s1
91 wrap_ibv_post_send:C c5 s1
92 simllmRnicRingDoorbell:S c5 s1 (doorbell=1)
93 simllmNetworkComplete:S c4 s0 (completion_queue_entry=1024)
94 simllmNetworkComplete:S c5 s1 (completion_queue_entry=1024)
95 ncclProxyProgress:C
96 sendProxyProgress:C
97 ncclNet.test:C c4 s0
98 wrap_ibv_poll_cq:P c4 s0 (completion_queue_entry, observes 93)
99 sendProxyProgress:S c4 s0 (ready_flag=0)
100 sendProxyProgress:S c4 s0 (head_counter=5)
101 ncclNet.test:C c5 s1
102 wrap_ibv_poll_cq:P c5 s1 (completion_queue_entry, observes 94)
103 sendProxyProgress:S c5 s1 (ready_flag=0)
104 sendProxyProgress:S c5 s1 (head_counter=6)
105 simllmKernelComplete:S (kernel_completion=6)
```

Every rank emits exactly 106 events. Its FIFO reaches occupancy two before
sequence 36 advances head, then ends with `head = tail = 6`, high watermark 2,
all ready flags false, and all slots empty. Posting events 22, 27, 54, 59, 86,
and 91 precede their doorbells, and every CQE comes from a later
`simllmNetworkComplete`, never from the post call.

## Exact intra-node event sequence

The intra-node reference uses the same six-step plan. Every rank emits exactly
17 events:

| Sequence | Function | Kind | Lane | Subject |
|---:|---|---|---|---|
| 0 | `ncclCommInitRank` | C | cpu | null |
| 1 | `ncclBuildRings` | C | cpu | null |
| 2 | `initChannel` | C | cpu | `logical_channel` |
| 3 | `ncclAllReduce` | C | cpu | null |
| 4 | `ncclEnqueueCheck` | C | cpu | null |
| 5 | `scheduleCollTasksToPlan` | C | cpu | null |
| 6 | `calcCollChunking` | C | cpu | null |
| 7 | `ncclLaunchKernel` | C | cpu | null |
| 8 | `ncclKernelMain` | C | gpu | null |
| 9 | `runRing` | C | gpu | null |
| 10 through 15 | `genericOp` | C | gpu | `nvlink` for chunks 0 through 5 |
| 16 | `simllmKernelComplete` | S | gpu | `kernel_completion` |

It emits no `ncclProxySaveOp`, `ncclProxyProgress`, `sendProxyProgress`,
`ncclNet.*`, verbs, doorbell, or network-completion event. Its FIFO remains
untouched.

## Receive leg and package boundary

The corrected first slice deliberately excludes the receive leg. No reachable
`irecv` or receive-post stub may remain. COMP-15's remaining scope must name
the real wiring still required: `recvProxyProgress` in `src/transport/net.cc`,
`ncclNet.irecv`, `ncclIbIrecv` in `src/transport/net_ib/p2p.cc`,
`wrap_ibv_post_recv`, receive completion through `ncclNet.test` and
`wrap_ibv_poll_cq`, receive-connector tail publication, and GPU `waitPeer` plus
`postPeer` head-credit return.

The package root exports only adapter-facing entry points, stack configuration,
route, result, event types, codecs, and stream validator. Mutable FIFO, channel,
proxy, net, verbs, request, and planner implementation classes remain internal
to `simllm.compute.nccl_stack`.

## Fatal unscored invariants

The corrected study keeps these outside every behavioral denominator:

- configured channel count equals the nine requested values;
- strict event-schema round trip, contiguous sequence, and monotonic time;
- all 88 reference signal-to-poll links are emitted by the owning observer and
  obey producer-before-consumer time and sequence;
- a foreign observer producer is rejected before event mutation;
- the inter-node FIFO reaches depth two and finishes quiescent;
- the intra-node FIFO and all proxy/network layers stay inactive;
- all reference timestamps remain zero and the caller clocks do not advance;
- ring-step, phase, byte, chunk, channel, and warp conservation hold;
- every doorbell follows its post, every CQE comes from the external completion
  source, and every head advance follows CQ observation; and
- no receive-leg symbol is reachable from the package or main-path event stream.

These invariants are fatal when violated but never increase the five-family or
35-instance behavioral pass count.
