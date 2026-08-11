# HTSIM-9 AtlahsFlowRuntime wrapper design and approval package

## Status and source boundary

This note originally prepared the ABI-v1 HTSIM-9 checkpoint. Its source
citations and gap assignments remain the historical audit used by the
follow-on ABI-v2 work. The htsim source referent for that audit is the SimLLM
gitlink
`8c3f8b231a6a9311ffc1e7969a003dcba724b50d`. Every htsim citation below is
against that exact object. The SimLLM source referent is wave-2 base
`6aa3a7622f57b63c35e030667bad24948c6a0e0e`, including the landed
`RnicDevice` review corrections. No file under `third_party/htsim` is changed
by this package.

Post-landing status, 2026-08-11: the ABI-v1 wrapper is on the pinned backend
main, and the frozen Tier A and Tier B gates pass for the isolated flow-level
fixture. BACK-8 and the demonstrated CORE-15 live-seam clauses closed on that
evidence. CORE-21 retains the same-graph authority comparison, BACK-31 retains
the unlinked-native executable negative. BACK-25 and BACK-26 later closed at
the vocabulary and relay boundary, so HTSIM-9 remains open only for a composed
run demonstrating packet-issue evidence through the live chain.

At the preparation freeze, the htsim outer contract accepted a flow request
containing flow, endpoint, payload, start-time and tag fields, then reported
completion by flow ID
(`third_party/htsim/htsim/sim/atlahs_flow_runtime.h:20-27,37-57`). The
then-current SimLLM `NetworkPort` was ABI v1 and admitted one flow extent
through `trySubmit`
(`simllm/backends/rnic/include/simllm/rnic/network_port.h:14-25,42-61,112-122`).
Its only terminal events are `Delivered` and `Dropped` with an optional ECN
bit and typed drop evidence
(`simllm/backends/rnic/include/simllm/rnic/network_port.h:22-40,101-110`).

The preparation did not silently broaden that ABI. Missing packet-attempt and
transport-control vocabulary was assigned to BACK-25 and BACK-26. The
follow-on completion landed those versioned surfaces while retaining the
explicit ABI-v1 compatibility subset.

## Authority and construction seam

Structural mode constructs one outer `AtlahsFlowRuntime` implementation that
owns the SimLLM RNIC devices and an htsim-backed port for each endpoint. A
native `RnicDevice` is the sole mutable authority for WR, WQE, SQ, CQ, CQE and
their timestamps. The htsim implementation owns transport-policy and fabric
state only. The port's token, flow and packet ledgers are correlation
projections and contain no WQ, CQ, QP, QPC, PCIe or DMA object.

Bypass mode retains the existing `AtlahsFlowRuntime` assembly and
`AtlahsWqeLedger` as the sole timing-neutral WQE authority. Mode selection is
made once in the runtime factory before `AtlahsHtsimApi::Setup()`. Structural
mode must take a setup path that never constructs the legacy ledger. The
pinned implementation currently constructs that ledger whenever a flow
runtime is present and posts it before calling `send`
(`third_party/htsim/htsim/sim/atlahs_htsim_api.cpp:287-305,141-184`), so
HTSIM-9 must split this path explicitly. A session that requests both modes
rejects before either authority is constructed.

The acceptance harness uses this narrow driven-port seam:

```text
Tier-A scenario runner
  -> PortFactory(factory configuration)
       -> NetworkPort implementation
       -> next-event query and event pump
       -> read-only issued and terminal trace
  -> RnicDevice with the returned NetworkPort
```

The fake factory and the future htsim factory implement the same seam. Device
configuration, WQE construction, scalar doorbell service D, event-loop order,
CQ polling and all acceptance checks stay outside the factory. Swapping
`fake` for `htsim` is the only harness change. The event driver delivers all
external events due at time `t` before calling `RnicDevice::progress(t)`, as
required by the device contract
(`simllm/backends/rnic/include/simllm/rnic/rnic_device.h:122-128`).

## Descriptor mapping

The table freezes the ABI-v1 request projection. `AtlahsFlowRequest` fields
refer to `third_party/htsim/htsim/sim/atlahs_flow_runtime.h:20-27`.
`NetworkTxDescriptor` fields refer to
`simllm/backends/rnic/include/simllm/rnic/network_port.h:42-61`.

| NetworkPort input | htsim projection | ABI-v1 rule | Gap owner |
|---|---|---|---|
| `abi_version` | No htsim field | Require exactly `kNetworkPortAbiVersion`; reject another version before htsim mutation. | None |
| `wqe_id`, `wr_id` | No htsim field | Retain only in the adapter's immutable correlation record. They never grant htsim ownership of native objects. | BACK-25 adds packet-attempt correlation below one WQE. |
| `flow_id` | `AtlahsFlowRequest::flow_id` | Project exactly. The pinned helper makes a GOAL host and node offset collision-free at `third_party/htsim/htsim/sim/atlahs_flow_runtime.h:11-18`. | BACK-25 permits several extents or attempts under one logical flow without overloading flow identity. |
| `flow_tag` | `AtlahsFlowRequest::tag` | Project exactly. | None |
| `source`, `destination` | Same-named request fields | Project exactly after endpoint-range validation. | None |
| `payload_bytes` | Same-named request field | Project exactly in the zero-header Tier-A fixture. | BACK-25 carries packet payload and wire extents when packetization is enabled. |
| `extent_index`, `extent_count` | No htsim field | ABI-v1 compatibility accepts only index 0 and count 1. Any other shape rejects explicitly. | BACK-25 |
| `eligible_at_ps`, `trySubmit(..., now_ps)` | `start_time_ps` | Require `now_ps >= eligible_at_ps`; the actual htsim flow start is `now_ps`. Eligibility remains a native timestamp, not a second htsim queue. | None |
| `policy_context_token`, `qpn` | No request field | Keep opaque in the adapter correlation record. The policy never dereferences a native QP object. | BACK-26 adds policy-context-keyed eligibility and rate updates. |
| `traffic_class` | No request field | Tier A accepts only its frozen class. Unsupported class-sensitive behavior rejects rather than becoming inert accidentally. | BACK-26 adds class and priority semantics for feedback and PFC. |

`NetworkSubmitResult` already represents Accepted, Busy and Rejected, including
a future retry time and typed rejection evidence
(`simllm/backends/rnic/include/simllm/rnic/network_port.h:63-99`). The pinned
`AtlahsFlowRuntime::send` returns `void`
(`third_party/htsim/htsim/sim/atlahs_flow_runtime.h:43-46`). HTSIM-9 therefore
owns the mechanical adapter protocol: install the token and flow correlation
before a potentially re-entrant `send`, return Accepted only after ownership
has transferred, return Busy only with an adapter-known future capacity event,
and unwind the provisional correlation if `send` throws. This mismatch needs
backend adapter code, not a wider NetworkPort vocabulary.

## Event mapping

The mapping distinguishes a packet observation from the one terminal outcome
of an admitted logical extent. A recoverable packet drop does not become a
WQE error. It terminates one packet attempt, and a later successful attempt
may still deliver the logical extent. ABI v1 cannot express that distinction,
which is why its compatibility path admits one whole-flow extent and the full
mapping waits for BACK-25.

| htsim-side event and pinned source | NetworkPort projection | Authority and ordering rule | Missing vocabulary |
|---|---|---|---|
| Flow transmit request through `AtlahsFlowRuntime::send` at `third_party/htsim/htsim/sim/atlahs_flow_runtime.h:37-57` | Construct the request from `NetworkTxDescriptor` and bind one nonzero `NetworkToken` before calling `send`; the port call and descriptor are at `simllm/backends/rnic/include/simllm/rnic/network_port.h:42-61,112-122`. | Native admission and SQ order remain in `RnicDevice`; htsim receives only an eligible flow. | Several extents and attempts under one flow require BACK-25. |
| Source DATA selection and transmit interval in `RnicTxPacket::{packet_index, extent, dispatch_start_ps, dispatch_end_ps, eta_ps}` at `third_party/htsim/htsim/sim/rnic_port.h:17-44`, committed at `third_party/htsim/htsim/sim/rnic_port.cpp:222-289` | No ABI-v1 event exists in the two-kind enum or terminal record at `simllm/backends/rnic/include/simllm/rnic/network_port.h:22-25,101-110`. Do not substitute acceptance for TX start or delivery for TX finish. | htsim owns serializer grant and service. SimLLM consumes a read-only observation when the vocabulary exists. | BACK-25 adds TX-start and TX-finish observations with extent, packet and attempt identity. |
| Route injection by `Packet::sendOn()` at `third_party/htsim/htsim/sim/network.cpp:55-84` | No ABI-v1 injection event exists at `simllm/backends/rnic/include/simllm/rnic/network_port.h:22-25,101-110`. | This is fabric progression, not WQE admission or completion. | BACK-25 adds the packet-attempt observation that can correlate injection without exposing route objects. |
| Endpoint arrival staged with flow, endpoints, packet kind and arrival time at `third_party/htsim/htsim/sim/datacenter/rnic_collective_network_runtime.cpp:925-975` | No ABI-v1 receive event exists at `simllm/backends/rnic/include/simllm/rnic/network_port.h:22-25,101-110`. For the one-extent compatibility path, aggregate through transport completion and emit one Delivered terminal. | Packet arrival belongs to htsim. Logical extent retirement and CQE production belong to the native device after the terminal event. | BACK-25 adds native-RX arrival and packet identity. |
| Whole-flow completion callback, typed only by `AtlahsFlowId`, at `third_party/htsim/htsim/sim/atlahs_flow_runtime.h:39-45` | Look up the still-live adapter token by flow correlation and emit exactly one `NetworkEventKind::Delivered` terminal using `simllm/backends/rnic/include/simllm/rnic/network_port.h:22-25,101-110`. | Remove the live token only after the native terminal call commits. Duplicate or unknown callbacks are fatal. | BACK-25 makes terminal identity unambiguous for multiple extents and retries and requires session-wide token uniqueness. |
| Packet endpoint consumption versus fabric drop, defined as exactly one terminal lifecycle at `third_party/htsim/htsim/sim/rnic_collective_packet.h:91-117` and implemented at `third_party/htsim/htsim/sim/rnic_collective_packet.cpp:497-541` | `ENDPOINT_CONSUMED` contributes to delivery aggregation. `FABRIC_DROP` contributes to an attempt drop. An unrecoverable logical-extent drop emits the Dropped terminal defined at `simllm/backends/rnic/include/simllm/rnic/network_port.h:22-25,101-110`. | A packet terminal never directly mutates a native WQE. The adapter reduces packet attempts to one logical terminal only after policy recovery is resolved. | BACK-25 adds explicit attempt terminals and stable drop provenance. |
| Queue overflow calls `Packet::free()` at `third_party/htsim/htsim/sim/queue.cpp:170-181`; the controlled rnic-cn injection also frees the selected packet at `third_party/htsim/htsim/sim/datacenter/rnic_collective_network_runtime.cpp:1238-1256` | When the logical extent is unrecoverable, emit Dropped with `DropLocation::Fabric`. Use `QueueOverflow` only for an observed queue-overflow source and `Injected` only for the controlled injection. Never infer one from a generic `free()`. | Drop classification is a read-only fact. Native error retirement and CQE status follow from the terminal event. | ABI v1 has coarse location and reason enums at `simllm/backends/rnic/include/simllm/rnic/network_port.h:27-40`; BACK-25 adds resource identity and evidence provenance. |
| ECN mark set on dequeue at `third_party/htsim/htsim/sim/compositequeue.cpp:78-102` | ABI v1 may OR the mark into the eventual terminal's `ecn_marked` bit at `simllm/backends/rnic/include/simllm/rnic/network_port.h:101-110`. | The bit is diagnostic only in v1. It cannot independently schedule hardware eligibility or a rate gate. | BACK-26 adds packet-keyed ECN feedback. |
| DCQCN receiver observes CE, emits CNP and the sender cuts rate at `third_party/htsim/htsim/sim/dcqcn.cpp:366-403,156-158` | The complete ABI-v1 event enum at `simllm/backends/rnic/include/simllm/rnic/network_port.h:22-25` has no CNP or rate update. Do not encode CNP as Delivered, Dropped or Busy. | htsim owns DCQCN policy state. The native hardware rate gate applies a versioned policy decision only after BACK-26. | BACK-26 adds CNP, rate and effective-time vocabulary keyed by packet and policy context. |
| Pause frame carries sleep time and sender identity at `third_party/htsim/htsim/sim/eth_pause_packet.h:15-53`; a lossless queue maps positive sleep to pause and zero to resume at `third_party/htsim/htsim/sim/queue_lossless.cpp:40-68` | The complete ABI-v1 event enum at `simllm/backends/rnic/include/simllm/rnic/network_port.h:22-25` has no pause or resume. Do not overload `traffic_class`, Busy or a terminal event. | htsim transports the frame. SimLLM owns RNIC watermarks and the paused priority gate once the control vocabulary exists. | BACK-26 adds PFC submit, pause and resume with endpoint or link, priority and duration or quanta. |
| Static failed-link count and route removal at `third_party/htsim/htsim/sim/datacenter/fat_tree_topology.h:70-73` and `third_party/htsim/htsim/sim/datacenter/fat_tree_topology.cpp:1392-1408` | ABI v1 can label an affected terminal with `DropReason::LinkDown` at `simllm/backends/rnic/include/simllm/rnic/network_port.h:34-40,101-110`, but it has no link-state event. | The pinned source configures failure before the run; it does not publish a timestamped transition. Unsupported dynamic transitions reject explicitly. | BACK-26 adds stable link identity, up or down state, transition time and optional effective rate. |

## Terminal validation limitation

The adapter must prevalidate nonzero session-unique tokens, token ownership,
WQE correlation and single termination before forwarding a terminal to the
device. This makes duplicate, unknown and cross-WQE negative controls atomic
at the wrapper seam. It does not repair the direct device path.

The current `RnicDevice::onNetworkEvent` commits the caller timestamp before
the work queue validates the terminal
(`simllm/backends/rnic/src/rnic_device.cpp:407-413`). The work queue itself
then rejects unknown, duplicate and cross-WQE terminals before queue mutation
(`simllm/backends/rnic/src/work_queue.cpp:551-615`). A rejected future event
can therefore ratchet the device clock even though queue state is unchanged.
BACK-24 owns transactional validation at the direct `RnicDevice` boundary.
HTSIM-9 acceptance must keep BACK-24 open until the direct-device negative
test passes.

## Maintainer approval package

### Requested backend branch and commit contents

After explicit maintainer approval, create `2026_08_10/simllm-addon` from the
maintainer-approved backend main tip. Record that full base commit in the
expectations-only commit before implementation starts. Backend main is not a
development target and is not merged without separate maintainer approval.
Once a SimLLM pin references any commit on the addon branch, the branch is
append-only: no referenced commit may be rebased, squashed, amended or
removed.

The approved backend sequence contains exactly:

1. An expectations-only commit that names the approved base commit, freezes
   the exact Tier-A command, event ordering, authority counters, token ledger,
   drop case, D and rate grids, bypass artifacts and expected relations. It
   contains no implementation or observed result.
2. A structural `AtlahsFlowRuntime` wrapper and htsim `NetworkPort`
   implementation behind the existing runtime factory. The wrapper links the
   SimLLM C++ RNIC library, owns no duplicate hardware scheduler, and uses the
   mapping above.
3. An explicit structural versus bypass selection in `AtlahsHtsimApi`.
   Structural setup does not construct or mutate `AtlahsWqeLedger`; bypass
   preserves the current ledger and accepted artifacts.
4. A versioned configuration and run record with hardware mode, authority,
   SimLLM RNIC configuration hash, transport profile, seed, topology identity
   and source revisions. The same hardware hash is used for `rnic-nn`,
   `rnic-cn` and DCQCN structural comparisons.
5. Native tests for factory selection, same-time event ordering, token
   conservation, terminal rejection, quiescence, controlled delivery and
   drop, one completion boundary, physical-work drainage and bypass identity.
6. A producer target named `htsim_rnic_tier_a` that implements the frozen
   harness port-factory seam. The first compatibility checkpoint keeps rich
   packet, ECN/CNP, PFC and dynamic-link modes disabled or rejecting until
   BACK-25 and BACK-26 land. Later append-only commits on the same branch map
   those versioned surfaces and add their directed tests. The checkpoint does
   not close HTSIM-9.
7. A results commit that cites the backend expectation commit and records raw
   evidence outside Git. It does not weaken or amend expectations after the
   run.

This package does not request changes in ATLAHS, backend main, a second addon
branch or either NetworkPort header. Any required SimLLM ABI change returns to
BACK-25 or BACK-26 before backend implementation continues.

### Required acceptance before a compatibility checkpoint pin

The backend tip must pass all of the following:

- the backend's repaired fail-fast build and CTest gate, including every
  existing test;
- the same Tier-A acceptance checker used by the fake factory, with only
  `--factory htsim` and the producer path changed;
- structural and bypass authority exclusivity, including a dual-authority
  construction negative control with exact pre-state and post-state equality;
- one issued token per frozen v1 WQE, one terminal per issued token, no token
  reuse in the session and zero live tokens at quiescence;
- the exact single-WQE D-additivity, inverse-rate and two-WQE FIFO relations,
  plus the wrapper-bypass mutant rejected by the same D predicate;
- a controlled htsim drop that produces one native TransportError completion,
  no Success CQE, one controlled drop record and no live physical work;
- direct binary evidence that htsim network service changes the native WQE
  terminal, CQE, JCT and completion boundary exactly once;
- byte-for-byte comparison of every retained bypass completion CSV, canonical
  completion rows, final JCT, step result sequence and replay TTFT/TPOT
  summary against gitlink `8c3f8b231a6a9311ffc1e7969a003dcba724b50d`;
- one hardware configuration hash across the three physical transport-policy
  rows, with profile identity kept separate; and
- explicit rejection of packet/control features that need BACK-25 or BACK-26,
  rather than silent no-op behavior.

BACK-24 must pass its direct-device atomicity test in SimLLM before the
composed gate claims terminal rejection at the device boundary. Wrapper-only
prevalidation is useful integration defense but is not task closure.

### ABI-v2 completion addendum

The follow-on vocabulary provides session-unique flow-extent and
packet-attempt tokens, explicit TX start and finish, RX arrival, attempt
delivery or drop, stable drop provenance, ECN/CNP, effective eligibility and
rate updates, PFC submit, pause and resume, and capability-gated link-state
events. `RnicPacketizedManifoldRuntime` emits packet observations from its
committed source and destination serializer boundaries. The wrapper schedules
those immutable observations at their event timestamps and never substitutes
flow acceptance for TX issue. The native WQE timeline consumes only explicit
data or retransmission TX-start events.

ABI v1 remains the default and its frozen raw and summary artifacts retain
their exact bytes. ABI v2 passes the same Tier A relations and adds exact
packet timeline relations. Static failed-link configuration still has no
timestamped transition source, so requested dynamic capability rejects and
HTSIM-15 owns the optional enabled producer. HTSIM-9 now remains open only for
the Tier B live-metric run through CORE-15.

Version negotiation is exact for the whole session. A v2 consumer paired with
a producer that advertises only v1 rejects before event-handler installation,
runtime setup, submission or authority mutation. There is no implicit
v2-to-v1 projection. A caller that requires compatibility constructs an
explicit v1 session, which retains the accepted baseline bytes.

Packet-keyed ECN and CNP correlation survives packet delivery or drop in a
bounded completed-attempt tombstone while the parent extent remains live. The
extent terminal purges that tombstone. This permits physical feedback to trail
the data packet without retaining packet state beyond the logical operation.

The physical packetized manifold currently advertises packet-attempt events
only. The full control vocabulary is exercised by a test runtime, not by a
physical policy or fabric producer. HTSIM-16 owns ECN/CNP, effective-rate, PFC
and link-state producers. HTSIM-15 separately owns the timestamped dynamic
link-transition source that HTSIM-16 can project. BACK-34 owns a registered
4,096-byte-quantum partial-final-packet cell; the frozen packet matrix and the
composed 8,192-byte directed test contain only full quanta.

### SimLLM pin-bump procedure

After the maintainer approves the backend results and names the immutable
backend tip, the SimLLM integrator performs this procedure in a separate
SimLLM change:

1. Record the full approved HTSim commit, its branch
   `2026_08_10/simllm-addon`, the backend expectation commit and all backend
   gate summaries. Confirm the object is reachable from the append-only
   branch. Do not initialize or fetch submodules recursively.
2. Move only the `third_party/htsim` gitlink to that full commit. Do not edit
   backend files from the SimLLM checkout and do not merge backend main.
3. Build the directly invoked htsim binaries with the repository helper, run
   backend CTest, then run the frozen Tier-A checker with `--factory htsim`.
   Store the htsim build under the root configured by
   `SIMLLM_HTSIM_BUILD` and study output under the external root configured
   by `SIMLLM_DATA_ROOT`, not in Git.
4. Run `.venv/bin/ruff check .` and `.venv/bin/pytest -q`, including the live
   backend tests when the toolchain is available. Compare bypass artifacts
   against the old gitlink before accepting the new run record.
5. Update `docs/modules/backends.md` in the same SimLLM change. The ABI-v2
   pin closes BACK-25 and BACK-26 at the vocabulary and relay boundary, and
   the Tier B results close the demonstrated BACK-8 and CORE-15 clauses,
   while every undemonstrated clause keeps a distinct residual owner
   (CORE-21, BACK-31, HTSIM-15 for the dynamic-link producer, HTSIM-16 for
   the physical control-event producers). Close HTSIM-9 only after a
   composed run demonstrates packet-issue evidence populating the native
   timeline through the live metric chain.
6. Commit the gitlink, module status and reproducible evidence together using
   the maintainer's identity. Never rewrite or delete the backend addon branch
   after SimLLM points at it.
