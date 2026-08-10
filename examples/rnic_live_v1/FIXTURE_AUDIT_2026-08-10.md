# Post-specified audit of the live RNIC v1 fixture

## Audit status and chronology

This is a post-specified fixture audit written on 2026-08-10. It is not an
expectations freeze, does not claim pre-registration, and records no new
acceptance result. The frozen
[`expectations.md`](expectations.md) is intentionally unchanged. Any proposed
wording at the end of this file is a draft for maintainer approval and has no
normative force.

The audited chronology is:

| Event | Commit | Time in Europe/Zurich |
|---|---|---|
| Standalone native WorkQueue implementation | `98746ff6974d6ffb6752a3bc678d6d18820dcf73` | 2026-08-07 12:15:14 |
| Original live-composition freeze | `65b56097d0409488e274b83eb2d0d2e6cb34a2f9` | 2026-08-07 16:39:05 |
| Retry-identity clarification | `facb26d1224fd358b3f5d5cc55f880786d624bc9` | 2026-08-07 16:58:55 |
| Drain and audit wording | `947399ce57616574bc6c1afbeaffd6342a25b921` | 2026-08-07 17:02:42 |
| Final pre-composition amendment | `d5d98a29c0bb5a3e1f61abb5eacdf33f27258f61` | 2026-08-10 11:10:50 |
| Initial `RnicDevice` composition implementation | `d1ed7dba23f3cd0b94b9157bd071f11be1213d91` | 2026-08-10 13:35:06 |
| Review-correction expectation freeze | `110e503491f0aee19b13b9b5893bf1ac4099d026` | 2026-08-10 13:56:37 |
| Review-correction implementation | `11b5fa2f7a05d74919a4450a3c3af2526a274dae` | 2026-08-10 14:04:11 |
| Wave-2 base audited here | `6aa3a7622f57b63c35e030667bad24948c6a0e0e` | 2026-08-10 14:23:05 |

The original freeze therefore followed the standalone WorkQueue slice. Its
truthful chronology is that it preceded the `RnicDevice` composition and any
SimLLM-plus-htsim combined run, not that it preceded every native WQ source
line. The final amendment `d5d98a2` still preceded `RnicDevice` implementation.
No HTSIM-9 composed result exists at this audit point.

The review-correction freeze `110e503` covered exact shared-fabric
configuration authority, ordering-domain claims, scalar-versus-DMA
validation, `submitPcie` clock atomicity, namespace applicability and evidence
separation. Those corrections landed in `11b5fa2` and are present in the
audited base. They did not cover external `NetworkEvent` validation.

## Landed evidence anchors

The audit uses only landed source and the registered task remainder. It does
not infer completion from run-record code being developed by another wave-2
worker.

- The ABI-v1 port carries one flow extent, Accepted, Busy or Rejected submit
  outcomes, and only Delivered or Dropped terminal events at
  `simllm/backends/rnic/include/simllm/rnic/network_port.h:14-25,42-110`.
  Its contract requires an accepted token to be unique only until its terminal
  is returned at `simllm/backends/rnic/include/simllm/rnic/network_port.h:112-122`.
- The current native queue implements only SEND and one embedded SQ/CQ pair at
  `simllm/backends/rnic/include/simllm/rnic/work_queue.h:20-41,67-85,109-151`.
  It records admission, network acceptance and outcome, retirement, CQE and
  poll timestamps, but explicitly leaves first and last packet issue unset in
  ABI v1 at `simllm/backends/rnic/include/simllm/rnic/work_queue.h:120-139`.
- The queue admits only the ready SQ head, handles Busy and binds a live token
  at `simllm/backends/rnic/src/work_queue.cpp:428-545`. It validates terminal
  kind, token and WQE before queue-state mutation at
  `simllm/backends/rnic/src/work_queue.cpp:551-615`, then performs ordered
  retirement and error CQE construction at
  `simllm/backends/rnic/src/work_queue.cpp:969-1039`.
- `RnicDevice` injects an externally owned port, reports module applicability
  and rejects DMA plus scalar doorbell, fetch or CQE service at
  `simllm/backends/rnic/include/simllm/rnic/rnic_device.h:47-98` and
  `simllm/backends/rnic/src/rnic_device.cpp:207-309`. Same-time external events
  precede progress by contract at
  `simllm/backends/rnic/include/simllm/rnic/rnic_device.h:122-128`.
- The `110e503` corrections are visible in shared-fabric equality and domain
  validation at `simllm/backends/rnic/src/rnic_device.cpp:253-279,311-374` and
  in transactional `submitPcie` validation at
  `simllm/backends/rnic/src/rnic_device.cpp:447-461`.
- A separate direct-device defect remains: `RnicDevice::onNetworkEvent`
  observes caller time before delegating validation at
  `simllm/backends/rnic/src/rnic_device.cpp:407-413`. BACK-24 registers the
  correction and its exact negative tests.
- The registered BACK-8 remainder is authoritative for this audit: run
  records, hardware-configuration hash, sole-authority projection and bypass
  equivalence remain SimLLM work; HTSIM-9 owns the outer wrapper and concrete
  htsim port; CORE-4 and CORE-5 own graph invocation and completion reduction.
  No in-flight implementation changes that status.

## Statement-by-statement audit

Status meanings are: **landed** for a capability present in the audited base,
**preparable** for a requirement executable with the deterministic fake port
without claiming htsim composition, **deferred** for an unchanged final gate
owned by a registered successor, and **divergent surface** where the frozen
wording needs an approved scope clarification because ABI v1 cannot represent
the claimed observation.

The rows below cover every normative paragraph, numbered relation, invariant
and reporting rule in the frozen file.

| ID | Frozen statement inventory | Audit against the landed surface | Status and owner |
|---|---|---|---|
| F01 | The file preceded native-plus-htsim composition and measured runs; results cite original and final freezes. | True for commits `65b5609` and `d5d98a2` relative to `d1ed7db` and HTSIM-9. The standalone WorkQueue already existed, so “before implementation” must mean composition implementation. | Landed chronology; results must cite `65b5609` and `d5d98a2`. |
| F02 | Tier A covers BACK-8 plus HTSIM-9 and step-sink replay; Tier B requires `ExecutionGraph -> DeviceRuntime -> CompletionEvent -> ExecutionResult -> StepResult -> TTFT/TPOT`; this study creates no `CompletionEvent` or `ExecutionResult`; BACK-8 stays open until Tier B. | The native device is component-only. There is no htsim adapter or Tier-B chain. Existing `HtsimStepSink` does not make the new device live-reachable. | Deferred to HTSIM-9, BACK-8, CORE-4 and CORE-5. |
| A01 | Structural mode has one mutable WQE lifecycle; native SimLLM owns WR/WQE/SQ/RQ/SRQ/CQ/CQE and timestamps; htsim owns policy/fabric; output rows are projections. | `RnicDevice` owns the landed one-SQ/one-CQ SEND slice. It has no RQ, SRQ, receive matching or public projection join. No composed htsim owner split exists yet. | Partly landed; remaining BACK-9, BACK-8 and HTSIM-9. The RQ/SRQ wording is a final-state requirement, not a current Tier-A capability. |
| A02 | Bypass retains `AtlahsWqeLedger`; structural and bypass are exclusive; dual mutation fails. | The pinned htsim path still constructs its ledger whenever a flow runtime is selected. SimLLM `RnicDevice` has no bypass-mode selector or authority counters. | Preparable as a factory negative control; real acceptance deferred to HTSIM-9 and BACK-8. |
| S01 | Sweep one signaled SEND over payload 4 KiB and 1 MiB, rates 200 and 400 Gbit/s, D 0 and 1,000 ps, and structural versus bypass mode. | Payload and scalar D are native fields. Rate and capacity are port/fabric inputs, not device config. The fake factory can cover structural cells; no real bypass comparison exists in the native surface. | Structural subset preparable; bypass deferred to HTSIM-9 and BACK-8. |
| S02 | All other native service, propagation and congestion are zero; wire serialization remains; direct, sink, StepResult, TTFT and TPOT use identical GOAL and replay. | DMA-off scalar service can be zeroed and the fake port can supply exact serialization. No GOAL-to-native wrapper, htsim serializer or downstream replay join exists. | Component fixture preparable; composed and metric portions deferred to HTSIM-9, BACK-8, CORE-4 and CORE-5. |
| S03 | FIFO posts two signaled 4 KiB SENDs W0 then W1 at time 0, one doorbell, one SQ and capacity-one egress. | One SQ, batched doorbell, ordered ready head and an injected capacity-one fake port are landed. | Preparable without backend changes. |
| S04 | FIFO boundaries are `D`, `D+L`, `D+L`, `D+2L`; W1 wait is L; CQE order is W0 then W1; JCT is `D+2L`; D adds 1,000 ps; `L(200)=2L(400)`; W1 never bypasses W0. | Scalar D is serialized at the doorbell, queue admission is ordered, Busy retains the head and retirement follows SQ sequence. A deterministic factory can provide exact L. Native `first_packet_at_ps` remains unset, so the port's TX trace must not be mislabeled as that native field. | Timing and order preparable; native first-packet claim is divergent surface, BACK-25. |
| B01 | Increasing D shifts fetch eligibility, first packet, CQE, poll, flow completion, direct JCT, StepResult and replay by exactly 1,000 ps without changing serialization. | Doorbell observation and zero-service admission shift natively. A fake TX probe, terminal, CQE and poll can shift. Native first-packet is unavailable and no downstream metric path exists. | Component subset preparable; first-packet BACK-25; downstream HTSIM-9, CORE-4 and CORE-5. |
| B02 | Doubling rate halves only wire serialization; D remains additive at both payloads. | Rate is outside `RnicDevice`; exact behavior can be imposed and checked at the fake port. A real htsim result does not exist. | Preparable, then HTSIM-9 successor evidence. |
| B03 | Composed htsim completion is the sole result boundary; StepResult adds neither probe JCT nor a second WQE-start constant. | No composed boundary exists, so this has not passed or failed. | Deferred to HTSIM-9, BACK-8, CORE-4 and CORE-5. |
| B04 | Native FIFO follows all four equations at both rates and D values. | All native and fake-port prerequisites exist except the native first-packet vocabulary. | Preparable using a separately labeled port TX observation; full claim needs BACK-25. |
| B05 | `rnic-nn`, `rnic-cn` and DCQCN structural rows share one hardware hash; only policy identity and effects differ. | `RnicDeviceConfig` is versioned, but no canonical hardware hash or three-profile composed run record is landed. | Deferred to BACK-8 and HTSIM-9. |
| P01 | Bypass reference is HTSim `8c3f8b2` with identical GOAL, topology, profile, seed and baseline argv. | The gitlink is exactly that commit. No new structural binary exists to compare. | Reference landed; comparison deferred to HTSIM-9 and BACK-8. |
| P02 | Completion CSV, canonical completion rows and JCT, StepResult tuples, and replay TTFT/TPOT are byte-identical for retained bypass profiles; GOAL text and binary are identity guards. | These are existing backend artifacts, outside `RnicDevice`. No bypass rerun has been made for this study. | Deferred to HTSIM-9 pin candidate and BACK-8. |
| P03 | The new config and audit record are excluded from byte comparison but must name bypass and `AtlahsWqeLedger`; diagnostic paths, IDs, wall time and command spelling are excluded. | BACK-8 explicitly retains the run record and authority projection. The landed device has neither field. | Deferred to BACK-8. |
| P04 | Default bypass adds no stdout line; native-only knobs reject or are omitted in bypass rather than silently acting. | No new CLI or mode selector is landed. | Deferred to HTSIM-9 and BACK-8. |
| I01 | Structural counters say one native session, N native posts, no legacy construction or mutations. | Native post counters exist, but cross-authority construction counters do not. | Preparable in the harness wrapper; production counters deferred to BACK-8 and HTSIM-9. |
| I02 | Bypass counters say no native construction or posts and one legacy ledger with N posts. | No SimLLM bypass run record is landed. | Preparable as a synthetic authority case; production evidence deferred to BACK-8 and HTSIM-9. |
| I03 | Every post has a stable WQE key; extent keys are unique; attempt tokens are issued once and terminate once; invalid terminals reject atomically; quiescence has no live token; retry changes only attempt index and token. | Stable WQE keys and live token-to-WQE checks are landed. ABI v1 defaults to one extent and guarantees token uniqueness only while live. It has no attempt index, session token history or transport retry record. Queue validation is atomic, but the device caller clock is not. | Divergent surface: BACK-25 for extent, attempt and session identity; BACK-24 for direct-device atomicity; BACK-12 for reliability behavior. |
| I04 | Native terminals, WQ/CQ sequences, projections and timestamps reconcile at quiescence. | Native records, counters and invariant validation exist. No public structural projection or htsim terminal join exists. | Native subset landed; projection deferred to BACK-8 and composed join to HTSIM-9. |
| I05 | SEND names local SQ and send CQ, not remote RQ; receive names one RQ or SRQ plus CQ; matching is later; one-sided operations create no receive WQE. | The landed WQE is SEND-only with local SQ/CQ and does not fabricate RQ. RQ, SRQ, receive matching and one-sided opcodes do not exist. | SEND subset landed; final cardinalities deferred to BACK-9 and BACK-12. |
| I06 | Signaled success completes at poll; successful unsignaled emits no CQE and reclaims through later signaled completion or explicit drain. | Signaled poll and unsignaled no-CQE behavior are landed, including later-signaled retirement. The registered BACK-9 remainder retains explicit all-unsignaled drain or teardown. Error completions are emitted even for unsignaled WQEs. | Partly landed; drain remains BACK-9. |
| I07 | First packet is an explicit issue event; acceptance and flow delivery are not substitutes. | The landed timeline comment states exactly this and leaves both packet fields unset. ABI v1 has no issue event. | Requirement aligned but unrepresentable today; BACK-25. |
| N01 | Unlinked structural mode, dual authority, duplicate/unknown/cross-WQE terminal, live token at quiescence, policy hash mismatch and wrapper bypass must all fail before mutation. | A preparation harness can exercise unlinked/factory, dual authority, wrapper bypass and its own token ledger. Direct WorkQueue rejects invalid terminals, but `RnicDevice` ratchets caller time first. No profile hash exists. | Preparable subset; BACK-24, BACK-8 and HTSIM-9 retain the real gates. |
| N02 | Controlled htsim drop yields a modeled error completion and never a Success CQE. | A Dropped `NetworkEvent` becomes `TransportError`, records controlled evidence and produces an error CQE even for an unsignaled WQE. There is no htsim source yet. | Native behavior landed and fake case preparable; real drop deferred to HTSIM-9. |
| E01 | Configuration is unscored; D, rate and FIFO families are scored separately; exact rows are separate; authority, conservation, reconciliation, quiescence, inactive and zero checks are fatal unscored; native executables are component evidence. | This is reporting discipline, not a device capability. The new checker can preserve these classes. | Preparable and remains mandatory for HTSIM-9 results. |
| E02 | Tier A reports native timeline, raw FCT, phase JCT, StepResult boundaries and replay TTFT/TPOT; probe JCT remains component-only. | Native WQE records and component JCT are available. First/last packet, htsim FCT, phase JCT and the joined replay do not exist. Existing `HtsimStepSink` output alone is not evidence for the new native path. | Divergent surface for a “complete” native timeline until BACK-25; remaining metrics deferred to HTSIM-9, BACK-8, CORE-4 and CORE-5. |
| E03 | No `CompletionEvent`, `ExecutionResult` or Tier-B claim appears before CORE-4 and CORE-5 validate that path. | No such claim or path is present. | Landed scope guard; keep it unchanged. |

## Divergence findings

The frozen final acceptance intent remains sound, but four distinctions must
be explicit before anyone treats the preparation harness as the full Tier-A
gate:

1. **Token lifetime:** ABI v1 permits reuse after a token's terminal. The
   frozen session-wide issue-once rule is stronger. The preparation adapter
   can enforce it locally, but HTSIM-9 cannot claim the native contract until
   BACK-25 lands or the composed adapter supplies an equally versioned ledger.
2. **Packet boundaries:** ABI v1 has no TX-start, TX-finish or native-RX event.
   A fake-port `port_tx_at_ps` observation is not
   `WqeTimeline::first_packet_at_ps`. BACK-25 owns the missing vocabulary.
3. **Terminal atomicity:** `110e503` repaired failed `submitPcie` clock
   atomicity, not failed external terminals. BACK-24 remains a P0 correction
   because `RnicDevice::onNetworkEvent` observes the future time before the
   queue rejects the event.
4. **Feature cardinality:** current v1 is one SEND-only SQ/CQ and one flow
   extent. RQ, SRQ, receive matching, one-sided operations, transport retries
   and all-unsignaled drain remain registered work. They must not be reported
   as landed merely because the final frozen invariant names them.

These findings do not justify weakening the frozen acceptance bar. They
require scope wording that separates the executable SimLLM-side preparation
subset from the final composed gate.

## Draft amendment for maintainer approval

**DRAFT ONLY. DO NOT APPLY WITHOUT MAINTAINER APPROVAL.**

Proposed text to add after the frozen file's “Freeze status” section:

> The executable SimLLM-side Tier-A preparation harness exercises the landed
> SEND-only, one-SQ/one-CQ, one-flow-extent `RnicDevice` slice with DMA off and
> an injected deterministic port. Its port TX trace is component evidence and
> does not populate `first_packet_at_ps` or `last_packet_at_ps`. It enforces
> session token uniqueness and terminal prevalidation at its wrapper seam,
> but those checks do not close BACK-25 or the direct-device BACK-24 defect.
> RQ/SRQ, receive matching, one-sided operations, packet attempts, transport
> retry and all-unsignaled drain remain under their registered tasks. The full
> Tier-A claim still requires HTSIM-9 to run the same checker with only the
> port factory replaced, BACK-8 to provide the run record, hardware hash,
> authority projection and bypass comparison, and BACK-25 to supply explicit
> packet issue vocabulary before a complete native packet timeline is
> reported. Tier B remains unchanged.

Proposed text to append to the frozen terminal negative-control paragraph:

> The review-correction freeze `110e503` and its implementation cover
> `submitPcie` validation atomicity only. External terminal atomicity remains
> BACK-24 until duplicate, unknown and cross-WQE future terminals are rejected
> by `RnicDevice` without advancing its caller clock. Wrapper prevalidation is
> defense in depth and is not a substitute for that direct-device gate.

If approved, the maintainer should land the amendment in a new explicit
expectations-amendment commit before the first real HTSIM-9 run. The results
must cite `65b5609`, `d5d98a2` and that new amendment commit and describe the
chronology exactly. If it is not approved before the run, this audit remains
post-specified context only and the original frozen wording continues to
govern.
