# NCCL stack skeleton v1 results

The re-registered study passes all 5 scored behavioral relation families and
all 35 parameterized instances. All 10 fatal unscored structural invariants
also hold.

The original expectations were frozen in commit
`92888d500daafd33223d11475c95ec31541140b7` before the first implementation or
run. Integration review then found contract-conformance defects in permanent
function names, the send-connector head/tail convention, and the proxy and
completion shape. Those findings triggered the expectations-only amendment in
commit `96477df9bfb49010b5ee4e5acc4f694348353021`, not an observed study result.
That amendment preceded the corrected implementation and corrected run. This
report therefore labels the evidence as re-registered after the naming and
protocol amendment and cites both frozen commits.

This study validates call structure, identities, planning, and state handoff.
It does not measure or claim NCCL, NVLink, proxy, PCIe, RNIC, or fabric timing
accuracy.

## Method

The study instantiates one independent stack and caller-owned `VirtualClock`
for each rank and route. It calls the name-mirrored `ncclCommInitRank` and
`ncclAllReduce` entry points, then compares every event field against the exact
sequence in
[expectations-review-amendment.md](expectations-review-amendment.md). The
original [expectations.md](expectations.md) remains unchanged as the historical
first freeze. A separate planner sweep crosses three payload sizes with three
logical-channel counts.

Reproduce the tracked ledger from the repository root:

```bash
python examples/nccl_stack_v1/run_nccl_stack_v1.py --check
```

The 48-row evidence ledger is [results.csv](results.csv), with SHA-256
`324374c2959ce52598ce97cbbfd34b977a66b1e2e55f52895bd25c62142fccf7`.
Its three run-configuration rows are unscored. It keeps the 35 behavioral
instances and 10 structural invariants in separate evidence classes.

## Behavioral results

| Family | Instances | Result |
|---|---:|---:|
| Exact inter-node event sequence | 4 ranks | 4/4 pass |
| Exact intra-node event sequence | 4 ranks | 4/4 pass |
| Explicit ring-step structure | 9 payload by channel cells | 9/9 pass |
| Exact total chunk count | 9 payload by channel cells | 9/9 pass |
| Exact per-channel chunk distribution | 9 payload by channel cells | 9/9 pass |

The inter-node route emits exactly 106 events per rank. The six planned chunks
move in three FIFO-depth-two batches through this causal loop:

```text
ncclProxySaveOp
  -> ncclLaunchKernel, ncclKernelMain, runRing
  -> waitPeer ready publication, genericOp copy, postPeer tail publication
  -> ncclProxyProgress and sendProxyProgress tail and ready polls
  -> ncclNet.isend, wrap_ibv_post_send, simllmRnicRingDoorbell
  -> later simllmNetworkComplete CQE publication
  -> ncclNet.test and wrap_ibv_poll_cq
  -> sendProxyProgress ready clear and head-credit publication
  -> later waitPeer head poll before slot reuse
```

Two GPU tail publications occur before the first proxy head advance. Each
rank's FIFO reaches occupancy two and finishes with `head = tail = 6`, all
ready flags clear, and all slot metadata empty. The 22 polls per rank observe
the exact frozen producer sequences, including the four later `waitPeer` head
polls used for slot reuse.

The intra-node route emits exactly 17 events per rank. It shares communicator,
planner, launch, `ncclKernelMain`, and `runRing` calls with the inter-node route,
then emits six `genericOp` calls with `nvlink` subjects and the kernel completion
signal. It emits no saved proxy operation, proxy progression, `ncclNet`, verbs,
doorbell, or network-completion event.

Every event field matches the amended oracle, including rank, communicator,
operation, channel, chunk, FIFO slot, dedicated send and receive peers, subject,
value, and causal producer sequence. `initChannel` carries both ring neighbors.
Send-path events carry only `send_peer_rank`, and no numeric value encodes a
peer.

## Planner results

The amended planner uses the explicit ring decomposition shared with
`simllm.compute.nccl`:

```text
ring_steps = 2 * (world_size - 1)
step_bytes = payload_bytes / world_size
lane_bytes = payload_bytes / (world_size * channels * warps_per_channel)
chunks_per_lane_per_step = lane_bytes / chunk_bytes
total_chunks = ring_steps * channels * warps_per_channel
               * chunks_per_lane_per_step
```

For the four-rank sweep with one warp per channel and 256-byte chunks, all nine
cells match exactly:

| Payload bytes | Channels | Wire bytes | Ring steps | Chunks | Measured chunks by channel | Chunks per step |
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

The first three steps are `reduce_scatter`; the final three are `all_gather`.
Every chunk is exactly 256 bytes, so no short final chunk exists. Configured
channel count is configuration-forced and is therefore a fatal unscored check,
not a behavioral family.

## Fatal unscored invariants

| Check | Result |
|---|---:|
| Configured channel count | 9/9 cells hold |
| Strict schema, contiguous sequence, and monotonic time | 8/8 streams hold |
| Signal-to-poll causal links | 88/88 links hold |
| Foreign observer producer scope | rejected before mutation |
| Inter-node FIFO depth and terminal quiescence | 4/4 ranks hold |
| Intra-node proxy, network, and FIFO off path | 4/4 ranks hold |
| Zero-duration caller clock | 8/8 streams hold |
| Ring-step and byte conservation | 9/9 cells hold |
| Post, doorbell, completion, and head order | 24/24 sends hold |
| Receive leg and internal package surface | absent |

Two properties are enforced by construction and are deliberately unscored.
First, all reference events read 0 ps from one caller-supplied clock, and the
stack never advances that clock. Second, a poll can be emitted only from the
owning observer with the exact producer event object; the API checks ownership,
producer kind, producer sequence, time, and identity before appending the poll.
Both properties are fatal because a violation would reveal a hidden timing
authority or a broken causal link. Neither increases the behavioral numerator
or denominator.

The event schema remains independent of `simllm-completion-event-v1`. Strict
JSON loading rejects missing fields, unknown fields, wrong schema tags, invalid
enum values, and malformed poll references. Complete-stream validation rejects
noncontiguous ordering, backward time, missing producers, and poll identity
disagreement. The main call path validates each emitted event incrementally;
serialization and explicit audit calls perform complete-stream validation.

## Scope and remaining work

This lands the corrected COMP-15 first slice only. FIFO slots retain chunk
identity and byte count but no data contents. The fake network completion
source makes CQEs visible after the verbs post and doorbell, with no service
duration. The receive leg is deliberately absent rather than exposed as a
self-completing stub.

The stack is not yet connected to `ExecutionGraph`, `CompletionEvent`,
`StepResult`, TTFT, or TPOT, so this is component evidence rather than a
live-reachable metric claim. COMP-15 remains open for calibrated GPU, PCIe,
RNIC and fabric service mechanisms, the receive progression leg, the
GPU-initiated leg, runtime projection, and the VLLM-14 and SGL-11 adapter
callers. The precise receive symbols and connector handoffs are recorded in
[the compute module registry](../../docs/modules/compute.md). Existing behavior
in `simllm.compute.nccl` and `simllm.compute.gpu_model` remains unchanged.
