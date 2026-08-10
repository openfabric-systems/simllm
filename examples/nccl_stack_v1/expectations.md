# NCCL stack skeleton v1 expectations

## Freeze status

This is the expectations-only record for the first COMP-15 slice. It is
written before the stack skeleton, its tests, or any run of this study. It
contains no measured values. The result report must cite this commit as the
pre-run expectation freeze.

The slice tests call structure and state handoff, not service time or physical
performance. Every modeled call, signal store, and successful poll observation
reads one caller-supplied `VirtualClock`. The reference clock starts at 0 ps and
the skeleton never advances it. Equal timestamps are therefore a fatal
by-construction invariant, but they are not a scored behavioral result.

## Reference configuration

Both route cases use one four-rank ring with ranks 0 through 3, a 4,096-byte
all-reduce payload, one logical channel, a 6,144-byte chunk limit, and one FIFO
slot per channel. The exact per-rank ring traffic is

`2 * (world_size - 1) * payload_bytes / world_size = 6,144 bytes`,

so the reference plan contains exactly one chunk on channel 0. Rank `r` sends
to `(r + 1) mod 4` and receives from `(r - 1) mod 4`:

| Rank | Send peer | Receive peer |
|---:|---:|---:|
| 0 | 1 | 3 |
| 1 | 2 | 0 |
| 2 | 3 | 1 |
| 3 | 0 | 2 |

The event record uses zero-based `sequence` values. `call` records entry into
a named functional boundary. `signal_store` records a proactive producer
store. `poll_observes` records the successful observation by a polling
consumer. There are no synthetic return records in this trimmed call trace.

## Exact inter-node sequence

Each of the four ranks must emit the following ordered `(function, kind, lane,
subject)` projection. Rank, communicator, operation, channel, chunk, peer,
value, and causal-link fields must also match the reference configuration.
Only subjects relevant to a state object are shown; other subject fields are
null.

| Sequence | Function | Kind | Lane | Subject | Exact causal detail |
|---:|---|---|---|---|---|
| 0 | `ncclCommInitRank` | `call` | `cpu` | null | rank and communicator identity |
| 1 | `ncclBuildLogicalChannels` | `call` | `cpu` | null | configured channel count is 1 |
| 2 | `ncclConstructLogicalChannel` | `call` | `cpu` | `logical_channel` | channel 0 and the rank's ring peers |
| 3 | `ncclAllReduce` | `call` | `cpu` | null | payload is 4,096 bytes |
| 4 | `ncclPlanAllReduce` | `call` | `cpu` | null | ring wire bytes are 6,144 |
| 5 | `ncclChunkPayload` | `call` | `cpu` | null | one 6,144-byte chunk |
| 6 | `ncclAssignChunksToChannels` | `call` | `cpu` | null | chunk 0 maps to channel 0 |
| 7 | `ncclLaunchCollectiveKernel` | `call` | `cpu` | null | CPU-host proxy is the only inter-node mode |
| 8 | `ncclCollectiveKernel` | `call` | `gpu` | null | one planned chunk |
| 9 | `ncclCopyChunkToFifo` | `call` | `gpu` | `data_fifo_slot` | slot 0 records 6,144 bytes, but no byte contents |
| 10 | `ncclStoreReadyFlag` | `signal_store` | `gpu` | `ready_flag` | value becomes 1 |
| 11 | `ncclStoreHead` | `signal_store` | `gpu` | `head_counter` | value becomes 1 |
| 12 | `ncclProxyProgress` | `call` | `cpu` | null | progress for channel 0, chunk 0 |
| 13 | `ncclProxyPollHead` | `poll_observes` | `cpu` | `head_counter` | observes sequence 11 |
| 14 | `ncclProxyPollReady` | `poll_observes` | `cpu` | `ready_flag` | observes sequence 10 |
| 15 | `ncclNet.isend` | `call` | `cpu` | null | destination is this rank's send peer |
| 16 | `ibverbs.post_send` | `call` | `cpu` | null | destination is this rank's send peer |
| 17 | `ibverbs.write_cqe` | `signal_store` | `rnic` | `completion_queue_entry` | zero-time send completion becomes visible |
| 18 | `ncclNet.test` | `call` | `cpu` | null | tests the send request |
| 19 | `ibverbs.poll_cq` | `poll_observes` | `cpu` | `completion_queue_entry` | observes sequence 17 |
| 20 | `ncclProxyStoreTail` | `signal_store` | `cpu` | `tail_counter` | value becomes 1 |
| 21 | `ncclKernelPollTail` | `poll_observes` | `gpu` | `tail_counter` | observes sequence 20 |
| 22 | `ncclReleaseFifoSlot` | `signal_store` | `gpu` | `ready_flag` | value becomes 0 |
| 23 | `ncclKernelComplete` | `signal_store` | `gpu` | `kernel_completion` | one chunk is complete |

For every `poll_observes` event, `observed_signal_sequence` must identify the
matching earlier `signal_store`. A poll may not precede its producer in event
sequence or virtual time. Subject, rank, communicator, operation, channel, and
chunk identities must agree across the causal pair.

## Exact intra-node sequence

The intra-node case uses the same four ranks and plan, but each rank emits only
this ordered projection:

| Sequence | Function | Kind | Lane | Subject |
|---:|---|---|---|---|
| 0 | `ncclCommInitRank` | `call` | `cpu` | null |
| 1 | `ncclBuildLogicalChannels` | `call` | `cpu` | null |
| 2 | `ncclConstructLogicalChannel` | `call` | `cpu` | `logical_channel` |
| 3 | `ncclAllReduce` | `call` | `cpu` | null |
| 4 | `ncclPlanAllReduce` | `call` | `cpu` | null |
| 5 | `ncclChunkPayload` | `call` | `cpu` | null |
| 6 | `ncclAssignChunksToChannels` | `call` | `cpu` | null |
| 7 | `ncclLaunchCollectiveKernel` | `call` | `cpu` | null |
| 8 | `ncclCollectiveKernel` | `call` | `gpu` | null |
| 9 | `ncclNvlinkCollective` | `call` | `gpu` | `nvlink` |
| 10 | `ncclKernelComplete` | `signal_store` | `gpu` | `kernel_completion` |

It has exactly zero events whose function starts with `ncclProxy`, `ncclNet.`,
or `ibverbs.`. Its FIFO head and tail remain zero, every ready flag remains
false, and no FIFO slot retains a chunk. These are fatal structural invariants,
not extra scored instances.

## Planner sweep and exact count relations

The planner sweep fixes `world_size = 4` and `chunk_bytes = 1,024`, then crosses
payload sizes 4,096, 5,120, and 8,192 bytes with configured channel counts 1,
2, and 4. The configured logical channel count is always preserved, including
idle channels. Let

```text
wire_bytes(P) = 2 * (world_size - 1) * P / world_size
chunk_count(P) = ceil(wire_bytes(P) / chunk_bytes)
channel(chunk_id) = chunk_id mod configured_channel_count
```

The payload must divide evenly by the world size. Chunks are contiguous in
wire-byte space, all chunks except the last have exactly `chunk_bytes`, and the
last chunk contains the nonzero remainder or `chunk_bytes` when division is
exact.

| Payload bytes | Channels | Wire bytes | Chunk count | Chunk counts by channel |
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

For the 5,120-byte payload, the final chunk is exactly 512 bytes. Every other
sweep chunk is exactly 1,024 bytes.

## Evidence classes and acceptance

Run configurations are reported without scoring. The study scores five
behavioral relation families, with parameterized instances kept visible:

1. exact inter-node event sequence, four rank instances;
2. exact intra-node event sequence, four rank instances;
3. planner chunk-count equation, nine sweep instances;
4. configured logical-channel count, nine sweep instances; and
5. round-robin per-channel chunk counts, nine sweep instances.

All five families and all 35 instances must pass. Fatal unscored structural
invariants cover strict event-schema round trips, monotonic event sequence and
timestamps, causal signal-to-poll identity, the intra-node proxy and net-seam
off path, quiescent FIFO state, and unchanged zero-duration clock state. These
checks never increase the behavioral denominator. Unit tests remain component
evidence and are reported separately.
