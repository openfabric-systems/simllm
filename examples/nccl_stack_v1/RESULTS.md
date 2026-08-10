# NCCL stack skeleton v1 results

All five frozen behavioral relation families pass, covering all 35
parameterized instances. All eight fatal structural invariants also hold. The
expectations were frozen in commit `92888d5` before the skeleton implementation
and before the first study run.

This study validates call structure, identities, and state handoff. It does not
measure or claim NCCL, NVLink, proxy, PCIe, RNIC, or fabric timing accuracy.

## Method

The study instantiates one independent stack and caller-owned `VirtualClock`
for each rank and route. It calls the name-mirrored `ncclCommInitRank` and
`ncclAllReduce` entry points, then compares every event field against the exact
sequence in [expectations.md](expectations.md). A separate planner sweep crosses
three payload sizes with three logical-channel counts.

Reproduce the tracked ledger from the repository root:

```bash
python examples/nccl_stack_v1/run_nccl_stack_v1.py --check
```

The 46-row evidence ledger is [results.csv](results.csv). Its three run
configuration rows are unscored. It keeps the 35 behavioral instances and 8
structural invariants in separate evidence classes.

## Behavioral results

| Family | Instances | Result |
|---|---:|---:|
| Exact inter-node event sequence | 4 ranks | 4/4 pass |
| Exact intra-node event sequence | 4 ranks | 4/4 pass |
| Planner chunk-count equation | 9 payload by channel cells | 9/9 pass |
| Configured logical-channel count | 9 payload by channel cells | 9/9 pass |
| Round-robin channel assignment | 9 payload by channel cells | 9/9 pass |

The inter-node route emits exactly 24 events per rank. The sequence starts at
communicator construction, passes through planning and kernel launch, then
closes one FIFO and proxy loop:

```text
GPU FIFO copy
  -> ready and head signal stores
  -> proxy head and ready polls
  -> ncclNet.isend
  -> ibverbs.post_send
  -> CQE signal store
  -> ncclNet.test and CQ poll
  -> proxy tail signal store
  -> kernel tail poll and slot release
  -> kernel completion signal store
```

All event fields match, including rank, communicator, operation, channel,
chunk, peer, subject, value, and causal producer sequence. Ring peers are
exactly `(send, receive) = (1, 3), (2, 0), (3, 1), (0, 2)` for ranks 0 through
3. Each rank's four polls observe the frozen producers: head sequence 11,
ready sequence 10, CQE sequence 17, and tail sequence 20.

The intra-node route emits exactly 11 events per rank. It shares communicator,
planner, launch, and kernel entry calls with the inter-node route, then emits
one `ncclNvlinkCollective` call and the kernel-completion signal. It emits no
proxy, `ncclNet`, or ibverbs calls.

## Planner results

For a four-rank ring and 1,024-byte chunk limit, all nine cells match

`wire_bytes = 2 * (world_size - 1) * payload_bytes / world_size`

and `chunk_count = ceil(wire_bytes / chunk_bytes)` exactly.

| Payload bytes | Channels | Wire bytes | Chunks | Measured chunks by channel |
|---:|---:|---:|---:|---|
| 4,096 | 1 | 6,144 | 6 | `(6)` |
| 4,096 | 2 | 6,144 | 6 | `(3, 3)` |
| 4,096 | 4 | 6,144 | 6 | `(2, 2, 1, 1)` |
| 5,120 | 1 | 7,680 | 8 | `(8)` |
| 5,120 | 2 | 7,680 | 8 | `(4, 4)` |
| 5,120 | 4 | 7,680 | 8 | `(2, 2, 2, 2)` |
| 8,192 | 1 | 12,288 | 12 | `(12)` |
| 8,192 | 2 | 12,288 | 12 | `(6, 6)` |
| 8,192 | 4 | 12,288 | 12 | `(3, 3, 3, 3)` |

Chunk IDs map to `chunk_id mod channel_count`. The 5,120-byte payload ends in
one 512-byte chunk; every other measured chunk is 1,024 bytes. Configured
logical channels remain present even when a channel receives fewer chunks.

## Fatal unscored invariants

| Check | Result |
|---|---:|
| Strict `simllm-nccl-stack-event-v1` round trip | 8/8 route and rank streams hold |
| Contiguous sequence and monotonic virtual time | 8/8 streams hold |
| Signal-to-poll causal links | 16/16 polls hold |
| Intra-node proxy and net off path | 4/4 ranks hold |
| Inter-node FIFO quiescence | 4/4 ranks hold |
| Intra-node FIFO remains untouched | 4/4 ranks hold |
| Zero-duration caller clock | 8/8 streams hold |
| Planner wire-byte conservation | 9/9 cells hold |

The zero-duration result is by construction. All events in the reference run
read 0 ps from the same caller-supplied clock, and the stack does not advance
that clock. This is a fatal invariant because any nonzero or divergent time
would reveal a hidden timing authority. It is unscored because no service-time
mechanism is enabled.

The event schema is independent of `simllm-completion-event-v1`. Strict JSON
loading rejects missing fields, unknown fields, wrong schema tags, invalid
enum values, and malformed poll references. Complete-stream validation also
rejects noncontiguous ordering, backward time, missing producers, and identity
disagreement between a poll and its signal.

## Scope and remaining work

This lands the COMP-15 first slice only. FIFO slots retain chunk identity and
byte count but no data contents. The zero-time ibverbs seam makes a CQE visible
immediately after a post so the completion call loop is executable. The
`ncclNet.irecv` and receive-post names are present and component-tested, while
the frozen default proxy loop follows the outgoing `isend` path specified by
the full call-loop graph.

The stack is not yet connected to `ExecutionGraph`, `CompletionEvent`,
`StepResult`, TTFT, or TPOT, so this is component evidence rather than a
live-reachable metric claim. COMP-15 remains open for service-time mechanisms,
calibration, the GPU-initiated leg, runtime projection, and the VLLM-14 and
SGL-11 adapter callers. The existing `simllm.compute.nccl` ring builder and GPU
service model are unchanged by this slice.
