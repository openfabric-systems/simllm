# RNIC WQ v1 expectations

The sweep equations, implementation and results first entered public history
together in commit `98746ff`. Directed requirements 7 through 11 were added
after initial runs, as recorded below, but before that same landing commit.
There is no expectations-only public ancestor. This document is therefore a
post-specified frozen regression specification, not publicly auditable
preregistration.

## Scope

This study validates the first structural slice of BACK-8/BACK-9: one finite
send queue and one finite completion queue bound to one QP, a batched
doorbell, ordered WQE retirement, signaled/unsignaled completion semantics and
an opaque network-side transfer port. One accepted network descriptor is one
WQE extent in this slice. Packetization, receive queues, retries, QPC caching
and PCIe service remain later slices.

All runs use deterministic picosecond time. CQ polling occurs at every event,
CQE-write/QPC/scheduler latency is zero, the final WQE is signaled, and the
network neither drops nor rejects a transfer.

## Sweep A: doorbell batching and signaling

Post `N = 32` WQEs at time zero into an SQ/CQ of depth 64. Sweep doorbell
batch size `B` in `{1, 4, 16}` and signaling interval `S` in `{1, 4, 16}`.
Use doorbell service `D = 1000 ps`, WQE-fetch service `F = 10 ps`, network
capacity 32 and zero network latency. All doorbells are issued at time zero.
Because `D >= B * F`, each batch's fetches finish before the next doorbell is
observed.

Fixed regression expectations:

- doorbell count is `N / B`;
- CQE count is `N / S`;
- JCT is `(N / B) * D + B * F`;
- signaling changes CQE traffic but not JCT in this poll-fast, non-overrunning
  configuration;
- all 32 WQEs are accepted, delivered and reclaimed, with zero drops, SQ-full
  rejections, network-busy attempts and CQ overruns.

| B | Expected doorbells | Expected JCT (ps) |
|---|---:|---:|
| 1 | 32 | 32010 |
| 4 | 8 | 8040 |
| 16 | 2 | 2160 |

| S | Expected CQEs |
|---|---:|
| 1 | 32 |
| 4 | 8 |
| 16 | 2 |

## Sweep B: network backpressure

Post `N = 16` signaled WQEs as one batch into an SQ/CQ of depth 32. Use
`D = 100 ps`, `F = 10 ps`, fixed network latency `L = 1000 ps`, and sweep
network in-flight capacity `C` in `{1, 4}`. Since C divides N, the completion
lanes remain regular.

Fixed regression expectations:

- JCT is `D + C * F + (N / C) * L`;
- network-busy attempts are `N - C`;
- doorbell count is 1 and CQE count is 16;
- all WQEs are delivered and reclaimed with no loss or overrun.

| C | Expected busy attempts | Expected JCT (ps) |
|---|---:|---:|
| 1 | 15 | 16110 |
| 4 | 12 | 4140 |

## Directed boundary checks

The native unit harness must additionally prove:

1. Posting past SQ depth is a controlled rejection naming the WR, not a dark
   drop.
2. A successful unsignaled WQE retires but retains SQ occupancy until a later
   signaled CQE is polled.
3. Out-of-order network callbacks retire in SQ order.
4. A controlled network drop creates an error CQE even for an unsignaled WQE
   and preserves drop location/reason.
5. Publishing a CQE into a full CQ creates a controlled fatal CQ-overrun event
   naming the WQE.
6. A busy network port stalls the SQ head and retries at the advertised time
   without reordering.

Before the final validation rerun, the native review added these adversarial
checks without changing either sweep equation:

7. The first lost CQE makes CQ overrun terminal. A later network completion
   cannot publish a CQE or reclaim through the failed WQE.
8. A completion for another network token cannot revoke a policy or PFC retry
   gate advertised for the SQ head.
9. The flow-level port records network acceptance and outcome but leaves first
   and last packet timestamps unset until a packetized adapter supplies real
   TX issue events.
10. CQE writes serialize, and same-timestamp host-poll versus device-progress
    priority is selected explicitly by call order without skipping any CQE
    scheduled strictly before the poll.
11. Contradictory delivery/drop evidence and timestamp overflow fail as
    asserted model errors before they can corrupt counters or time.
