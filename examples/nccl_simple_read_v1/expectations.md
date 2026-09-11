# Simple buffered-read expectations

This freeze precedes the new implementation and its first run. It covers
float32 Ring on a direct two/four-GPU mesh, using the pinned NCCL source
`7b83616df3ae082a1f32bb74c27458bfe8153a13`. All cycle, memory and transport
inputs in this study are declared synthetic inputs, not hardware calibration.
The sibling JSON freezes the parameter matrix.

## Source and causal invariants

The explicit `buffered_read` placement leaves LL/LL128 receiver-side writes
unchanged. For Simple, every useful sending stripe stores into the sender's
local FIFO. Publication follows completed stores. A matching receiver sees
the tail before issuing a peer-read request. The existing physical calendar
carries the request to the sender; only then does source-memory service
become eligible. Response transmission waits for that service, and consumer
work waits for response visibility. Head return follows consumer work.

The source-memory visit belongs to the data-owning GPU but is attributed to
the requesting block. It requires no resident sender block. Read request,
response, source-memory visit and consumer must share the original clock.
A response gate is installed before request visibility, releases exactly once
after the request arrives, and does not permit clock reentry. Unsupported
placement, missing routes, duplicate gates and changing a communicator's
placement after admission reject explicitly. Packet-only critical-path
capture rejects an external service gate until it can represent that edge.

Empty slices preserve tail/head progress and issue no payload transaction or
payload-memory visit. Each read extent counts useful data exactly once in
its response, and carries a separate zero-payload request. Source bindings
identify requester, buffer owner, connection sequence and stripe. A consumer
cannot return FIFO capacity before its required responses and output work.
Repeated operations retain absolute steps and slot reuse. Switching among
LL, Simple and LL128 on one read-capable communicator preserves legality.

## Physical bounds and behavioral expectations

Floor: a nonempty read cannot finish before request serialization and
propagation, source-memory service, and response serialization and
propagation; endpoint receive work can strengthen this bound.
Ceiling: the isolated no-loss one-read fixture is bounded by the sum of its
serialized request, both ingress services, source-memory service, response
packets and their propagation. General contended completion has no claimed
finite ceiling without bounded stalls.

1. Isolated gated read: adding D picoseconds of source service shifts every
   response start and final visibility by exactly D, for D in the JSON grid.
   The request timestamps and bytes stay identical. This is an exact oracle.
2. Isolated remote-memory visit: doubling its declared bandwidth halves
   service, subject only to integer-picosecond rounding. Concurrent local
   work on the owner shares the same memory cursor. This is a second oracle.
3. Collective and original serial graph: sweep link rate and available SMs
   independently at two/four ranks. Higher link rate and more available SMs
   are expected not to increase completion on this declared symmetric grid.
   Compare both placements; no universal read-versus-write timing ordering
   is asserted because placement changes overlap and local memory work.
4. The serial graph's time to first token and time per output token equal
   its modeled communication time plus exactly 200,000 ps fixed compute.
   Every change in either metric equals the changed collective duration.
5. Explicit `buffered` retains prior writes and timestamps. LL/LL128
   `buffered_read` executes the identical transfers and resource visits as
   `buffered`; placement metadata may differ. The existing source-off path
   preserves its accepted packet baseline exactly.

Fatal guards are separate from behavioral relations: identity, byte and
output conservation, causal order, slot legality, no duplicate service,
unchanged bypass, complete trace joins and physical floors. Any violation
voids the affected study scope. Retain its evidence; do not turn it into a
lost score or rewrite the freeze. A refuted directional expectation is
reported explicitly. Counts of configurations, oracle rows and guards are
never added into one success denominator.

## Qualification

This establishes executable mechanism and live token-metric reachability
under TRAF-54. It does not identify physical read granularity, cache behavior,
GPU cycles or hardware accuracy. TRAF-43 retains independent calibration and
untouched validation. The standardized primitive experiment has a separate
versioned contract and is not claimed to have run with this model study.
