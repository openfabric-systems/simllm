# simllm.backends and third_party

Invocation and result parsing for the network simulators, plus the pinned
backend submodules.

## Interface

- `HtsimRnicConfig` + `build_htsim_rnic_command` + `run_htsim_rnic`: direct
  GOAL-driven `htsim_rnic` runs (profiles `rnic-nn`, `rnic-nn-fluid`,
  `rnic-cn`; a run is valid only with `physical_quiescence=verified`),
  binary discovered via `SIMLLM_HTSIM_RNIC`, the README build location,
  then `PATH`.
- `FlowCompletion` + `parse_completion_csv`: completion-CSV parsing
  with a stable legacy prefix
  (`profile,flow_id,source,destination,tag,payload_bytes,start_time_ps,completion_time_ps,fct_ps`)
  followed by optional WQE bookkeeping (`wqe_id`, SQ/RQ/CQ identities and
  sequences, transport kind and transport-object ID);
  `RnicRunResult.job_completion_time_ps()` takes the maximum of exact WQE
  completion rows and the driver's whole-nanosecond GOAL completion summary.
  This covers compute-only schedules and trailing compute after the last WQE.
- `simllm::rnic` session records + `simllm.backends.rnic_records`: strict
  `simllm-rnic-session-config-v1`, `simllm-rnic-session-result-v1` and
  structural-bookkeeping records. Structural records carry canonical
  effective hardware and its SHA-256; bypass records explicitly name
  `AtlahsWqeLedger` and carry no native hash. Native WQE state projects into
  immutable bookkeeping and the accepted completion CSV without a second
  lifecycle authority. The reusable bypass checker guards the full reference
  input tuple and compares the four frozen behavioral artifact classes byte
  for byte.
- `simllm.backends.fct.normalized_fct`: per-flow FCT normalized to the
  `rnic-nn` baseline of the identical GOAL, matched by
  (source, destination, tag). Valid for aligned-start flows; for phases
  with model-dependent start stagger use the phase makespan ratio
  (M1 finding F1).
- `HtsimDcqcnConfig` + `run_htsim_dcqcn`: GOAL-driven RoCEv2 DCQCN runs
  over a topology-file ns-tm3 Clos (`htsim_dcqcn_atlahs`, landed via the
  backend DCQCN PR); same completion-CSV schema and quiescence contract.
- `HtsimUecConfig` + `build_htsim_uec_command`: argv construction for
  GOAL-driven `htsim_uec` runs.
- `HtsimStepSink` + `HtsimStepSinkConfig` (M4): the closed-loop step sink,
  a callable `StepRecord -> StepResult | None` matching the adapters' sink
  contract. Per step it renders the TP serial chain
  (`simllm.traffic.render_step_goal`: per layer one `calc` then the two ring
  allreduces, plus the MoE
  dispatch/combine all-to-alls when the config declares `ep_ranks` and
  the dims declare experts, landed with the M5 slice). A provider may return
  an optional exact duration per layer; the sink validates count,
  nonnegativity and the fused sum, then truncates cumulative boundaries to
  GOAL ns. Providers without the hook retain the original even scalar split
  byte for byte. An optional `StepRecord.num_sampled` prices the LM head from
  exact attribution; absence retains `len(scheduled)`. The config's optional
  `num_goal_ranks` pads topology-sized GOALs without moving the active group
  to the highest rank. The sink converts
  with `txt2bin`, runs `htsim_rnic` on the configured profile/topology,
  parses the completion CSV and returns the simulated makespan as the
  step latency with `completed_at_ps = record.virtual_time_ps + makespan`.
  A step with no TP collectives (TP world of 1, or a zero-token drain
  record) returns `None`, so the adapter's own compute-only estimate
  stands. Per-step subprocess invocation is the documented diagnostic
  mode; the persistent co-simulator is BRIDGE-1 (core.md).
  `StepNetworkOutcome` keeps per-step bookkeeping (compute estimate, sample
  count and exactness, ordered layer calcs, makespan and network share) for
  reporting.
- `SerialStepLowerer` + `SerialStepLowererConfig`: CORE-2 diagnostic lowering
  from a `StepRecord` to per-layer compute plus semantic TP/EP collective
  operations. Explicit framework observations bypass the fallback schedule and
  are enveloped without reconstructing framework policy. JSON-round-tripped
  graphs replay through `render_serial_execution_graph_goal`.

## Pinned submodules

| Submodule | Repo | Ref | Provides |
|---|---|---|---|
| `third_party/atlahs` | [ATLAHS-rnic-private](https://github.com/yifeng-ethz/ATLAHS-rnic-private) | `main` | GOAL toolchain (txt2bin, LogGOPSim, goal_gen), validated `htsim_rnic` launcher (`atlahs_entry.py`) |
| `third_party/htsim` | [HTSIM-rnic-private](https://github.com/yifeng-ethz/HTSIM-rnic-private) | `2026_08_05/simllm-addon` | UEC htsim, RNIC model series, `htsim_rnic` executable and WQE bookkeeping |

As of 2026-08-03 the launcher, the RNIC wiring, the DCQCN comparator
(mlx5-faithful loss recovery, ECN-only and ECN plus PFC modes, storm
metrics) and the full rnic-cn algorithm-book implementation
(deterministic reservation ledger, windowed feedforward snapshots,
fractional nflow, sender egress composition, BJP-derived resequencing
window) are merged. The SimLLM pin for HTSim is now on the append-only
`2026_08_05/simllm-addon` branch because the WQE bookkeeping commit has not
been merged into backend main. A submodule pin to an addon branch is an
intentional supported state. The same HTSIM sources build on Linux with
GCC/Clang and on Windows with MSVC. From the SimLLM root, the supported
helper commands are:

```bash
./scripts/build_htsim.sh build/htsim --test
build/htsim/datacenter/htsim_rnic -goal trace.bin -linkspeed_bps 400000000000 -rnic_profile rnic-cn
```

```powershell
.\scripts\build_htsim.ps1 -BuildDirectory build\htsim -RunTests
build\htsim\datacenter\Release\htsim_rnic.exe -goal trace.bin -linkspeed_bps 400000000000 -rnic_profile rnic-cn
```

Binary discovery checks `SIMLLM_HTSIM_RNIC`, `SIMLLM_HTSIM_DCQCN`, or
`SIMLLM_TXT2BIN` first, then both the single-configuration build layout
and the MSVC `Release`/`RelWithDebInfo`/`Debug`/`MinSizeRel` layouts,
then `PATH`. The framework adapters and traffic-model layer stay in
Python and use this platform-neutral discovery path.

Changes to the backends go through their own repos on
`<YYYY_MM_DD>/simllm-addon` branches; SimLLM only bumps pins.

## RNIC hardware and transport-policy split

RNIC hardware and transport/congestion control are independent model axes.
The reusable hardware model is SimLLM-owned C++ under
`simllm/backends/rnic/`; htsim continues to own the fabric and the selectable
`rnic-nn`, `rnic-cn` and DCQCN policies. Full-RNIC comparisons must hold one
hardware configuration fixed while swapping only the policy. The
`rnic-nn-fluid` closed-form path retains an explicit hardware bypass for the
existing zero-residual validation anchor.

The composed direct-simulator path is:

```text
GOAL Send
  -> SimLLM RDMA Work Queue and RNIC hardware
       WR/WQE -> SQ/RQ -> doorbell -> PCIe/QPC/DMA -> TX
  -> htsim transport/CC policy and packet fabric
  -> SimLLM RNIC RX -> payload DMA -> CQE -> poll or interrupt
  -> GOAL completion
```

The target composition links the SimLLM C++ library into the directly invoked
htsim binary, with no Python callback in the packet event loop. The composed
runtime will present `AtlahsFlowRuntime` to `AtlahsHtsimApi`. That link is not
live today: current htsim binaries do not contain `simllm::rnic`. HTSIM-9 owns
the combined outer wrapper and backend extension through which the SimLLM
hardware runtime calls an htsim policy and fabric using opaque flow and packet
tokens. QP, WQE, CQ, QPC, PCIe and DMA objects never cross that boundary.

State ownership is explicit:

- SimLLM RNIC hardware owns WR/WQE/CQE contents, SQ/RQ/SRQ/CQ, QP state and
  pairing, PSN and reliability state, context and translation caches, PCIe,
  MMIO, DMA, packetization/reassembly, TX/RX queues, the hardware rate gate,
  PFC gates, counters and completion delivery.
- htsim transport/CC policies own policy state such as DCQCN alpha,
  current/target rate and recovery timers, or the `rnic-cn` reservation and
  predeclaration ledger. The hardware applies their decisions at its rate
  gate.
- htsim fabric owns links, switch queues, ECN marking, propagation, wire and
  switch drops, and PFC-frame transport. SimLLM owns the RNIC buffer
  watermarks that originate PFC and the paused priority state that consumes
  it.

### Modular construction

The native device is assembled through the versioned `RnicDeviceConfig` and
`RnicDevice` composition entry point. It joins the work-queue core with the
scalar QPC compatibility module, optional DMA (`PcieFabric` plus
`WorkQueuePcieBinding`) and either an injected versioned `NetworkPort` or an
owned inert port. The QP number and policy-context token remain device-level
identity, including when QPC is disabled. Both native probes and every
composed-session test construct through this entry point; direct module
construction remains only in component tests and exact oracle pairs.

A disabled module keeps the interface identical: its parameters are inert or
rejected, never silently rescoped; its module stages report `not_applicable`;
and its off state preserves the accepted baseline artifacts byte for byte.
DMA-on rejects scalar doorbell, WQE-fetch and CQE-write service before fabric
state can mutate. The resulting queue timestamps are mirrors of committed
fabric results, not a second scheduler. One caller-driven clock and the
documented event, progress and CQ-poll order apply to the whole device.

An owned fabric is heap-stable and an external fabric is retained by explicit
shared ownership. The embedded fabric config must equal the attached fabric's
effective config field by field. Shared devices derive missing ordering
domains from a nonzero device namespace and claim the resolved pair on the
fabric, so equal SQ/CQ defaults cannot collide silently. Device submissions
reject domains claimed by another live device, while accepting either own
claimed domain or an unclaimed domain. Failed construction and failed
submission leave claims, caller time, transactional generation and accounting
unchanged.
The absent-network path owns an inert port that accepts with a fresh token and
delivers on the device progress pump; HTSIM-9 supplies the future concrete
external port. BACK-20 adds selection of the queue submission source and CQ
consumer when DMA is present.

### WQE authority and projection contract

One session has one mutable WQE authority. Accounting records are projections
of that authority, not peer implementations of the lifecycle:

| Surface | Contract | May mutate WQ/WQE/CQ state? |
|---|---|---|
| SimLLM native C++ RNIC session | Sole structural authority for WR/WQE/CQE contents, WQ and CQ occupancy, identities, lifecycle and timestamps | Yes |
| htsim policy and fabric port | Network service behind opaque extent and policy-context tokens; returns admission, delivery, drop and feedback events | No |
| `simllm-request-bookkeeping-*` | Immutable public correlation projection of the selected session result into request and execution facts | No |
| `AtlahsWqeLedger` | Sole timing-neutral compatibility authority in explicit hardware-bypass mode | Yes, only while the native structural RNIC is disabled |
| Backend result and legacy CSV columns | Immutable output projection of the selected structural or bypass authority | No |

Structural and bypass modes are mutually exclusive. In structural mode the
native RNIC allocates every queue and WQE identity and supplies every WQE
timestamp; `AtlahsWqeLedger` is neither constructed nor mutated. In bypass
mode no structural RNIC object exists. The run record sets
`hardware_mode=bypass` and names the timing-neutral ledger as authority. A run
must never merge two independently produced lifecycle records or choose
between their timestamps after simulation.

The stable reconciliation key is the session and endpoint plus the owning WQ
kind, WQ identity and post sequence. A provider WR ID, GOAL flow ID, local
implementation index and htsim token are correlations, not substitute WQE
identities. One WQE may produce several logical network extents. Every extent
has a stable extent index; every transmission or retry has a distinct attempt
index and opaque token that terminates in one delivery or drop event. A dropped
attempt does not terminate its logical extent if reliability schedules a retry.
At quiescence, native posts and terminal states, WQ and CQ producer/consumer
sequences, all network attempt tokens, public bookkeeping facts and result rows
must reconcile exactly under the WQE and logical-extent keys. Applicable
timestamps are monotonic through post, publication, fetch, QPC readiness,
admission, first and last packet, transport retirement, CQE visibility and
poll. A bypassed stage is `not_applicable`, never an invented zero.

A send WQE belongs to its local SQ and send CQ. It does not own or parent the
remote RQ. A receive WQE is posted separately to exactly one RQ or SRQ and is
associated with a receive CQ; RX matching later links the send and receive
WQE keys. An SRQ receive WQE is not QP-specific until that match, and
one-sided operations do not invent a receive WQE. A successful signaled send
produces its requested CQE; a successful unsignaled send produces none.
Transport retirement advances the NIC consumer, while provider-visible WR-slot
reclamation follows a later signaled completion or an explicit modeled drain
or teardown rule. Error and receive completions follow their own documented
rules. The current bookkeeping-v1 rule and legacy CSV `rq_id` are compatibility
forms, not structural semantics. A structural public projection must use a
versioned schema with these cardinalities while preserving a v1 reader.

The current `AtlahsWqeLedger` remains the bypass authority until BACK-8,
BACK-9, BACK-12 and HTSIM-9 connect the structural path. A WQE has no single
scheduled start constant. The model records post, doorbell publication and
observation, WQE fetch or BlueFlame transfer, QPC readiness, scheduler
admission, first and last packet, transport retirement, CQE visibility and CQ
polling separately.
NIC start is first-packet issue. A reduced per-WQE start latency is derived
from the native timeline for calibration and never charged again by htsim.
The pre-implementation composition expectations were first frozen in
[examples/rnic_live_v1](../../examples/rnic_live_v1/expectations.md) at commit
`65b5609`; commit `facb26d` clarified retry identity, and commit `947399c`
records the final pre-run drain and audit wording.
The evidence classes, mlx5 hook and boundary-test matrix are recorded in
[the RNIC hardware calibration plan](../papers/rnic-hardware-calibration.md).

## Status

`htsim_rnic` invocation, completion parsing and FCT normalization landed
with M1 (BACK-1, BACK-3 closed). The end-to-end test runs them for real
wherever the backend toolchain is built (it self-skips otherwise), and the
M1 sanity studies exercise the full pipeline: 15 of 18 pre-registered
checks pass, the six fluid workload-A configurations and four workload-B
runs to zero picosecond residual, and the three failures are traced to
mis-registrations, not defects (findings F1-F3 in examples/m1/RESULTS.md).

`HtsimStepSink` landed with the M4 first slice and is validated by the
examples/m4 pre-registered studies (every check passes: fluid step
makespans exact to 0 ps across TP x step-shape, packetized nn inside its
registered band and in fact on its point form, replayed TTFT/TPOT exact)
plus a live closed loop: vLLM v0.26.0 in-process at tp=8 under
`SimExecutor` with the sink drove `htsim_rnic` inside the engine step
loop, every step latency matching the closed form to 0 ps
(examples/m4/RESULTS.md).

On 2026-08-10 BACK-5, BACK-6 and BACK-7 closed. The sink now consumes an
optional exact provider layer breakdown, an optional exact step sample count
and an explicit GOAL-rank count while preserving the default M4 and CORE-2
GOAL bytes. The precision study matched all four unequal-layer closed forms,
both sample-attribution relations and the default digest exactly. The shipped
roofline provider now supplies real per-layer values when its breakdown is
enabled. COMP-17 owns the remaining profile-table and trace-calibrated
breakdowns after COMP-6 supplies per-layer kernel shapes. The serial replay
lowerer uses the same optional exact sample count as the live sink and retains
the scheduled-row fallback when the field is absent. The study's
registered fluid-plus-topology command was invalid because htsim accepts
physical topology files only for physical profiles. The expectation was not
rewritten: post-specified checks instead showed 0 ps residual and exact
normalized flow ledgers for both a 64-rank fluid comparison and the actual
64-node `rnic-cn` topology comparison at TP widths 2 and 4. See
[examples/step_sink_precision/RESULTS.md](../../examples/step_sink_precision/RESULTS.md).

On 2026-08-05 HTSIM commit `d778326` added one timing-neutral WQE lifecycle
layer shared by the injected runtimes. It creates deterministic per-node
SQ/RQ/CQ identities, posts and FIFO-dispatches the SQ at the existing send
timestamp, retains RQ as an identity-only placeholder, and posts plus consumes
the CQ at the existing completion timestamp. DCQCN rows carry a stable
directed-pair QP identity; `rnic-cn` rows carry a stable directed L2 link-pair
identity; null profiles explicitly carry `none`. Packets remain private.
The complete backend suite passed 344 of 344 tests. Separate reproducible
manual driver smokes checked both physical transport fields. The frozen
lowering study retained every JCT and combined flow/WQE row exactly; see
[examples/core2_lowering/RESULTS.md](../../examples/core2_lowering/RESULTS.md).

On 2026-08-07 the first SimLLM-owned native RNIC slice landed under
`simllm/backends/rnic/` as a dependency-free C++17 library. One QP-bound SQ/CQ
pair now has finite capacity, accepted-prefix WR posting, batched doorbells,
ordered transport retirement, signaled/unsignaled reclamation, CQ owner wrap,
polling, network would-block and controlled SQ-full, network-drop and
CQ-overrun evidence. Its versioned `NetworkPort` passes opaque transfer tokens
plus flow/tag and policy-context identity without transferring WQ/QP/CQ
ownership. Flow-level acceptance/outcome timestamps remain separate from the
packet issue timestamps that HTSIM-9 must supply. The htsim wrapper is not yet
connected, so the old HTSIM ledger remains the live compatibility path.
The post-specified native regression study passes all 11 cells exactly; see
[examples/rnic_wq_v1/RESULTS.md](../../examples/rnic_wq_v1/RESULTS.md).

On 2026-08-07 BACK-10 closed at its accepted deterministic transaction-level
boundary. The shared `PcieFabric` has distinct semantic service classes,
transactional plan/commit, MWr/MRd/CplD segmentation, configured modeled-link
overhead, Gen1 through Gen5 directional serialization, DWORD, 4 KiB, MPS,
MRRS and eager-RCB splitting, typed credit pools, read-tag and completion-
buffer limits, fixed service latency, and per-class byte, wait, service-delay
and path-delay accounting. Every NUMA, IOMMU, ACS, switch, DDIO-miss and GPU
Direct penalty accepts an explicit disabled state or fixed, nonnegative
discrete-Gaussian and rare-tail two-Gaussian-mixture profiles with nonzero
analytical incidence. Results record realized delay plus evaluation,
occurrence and tail-selection counts.
The regular mlx5 Work Queue path emits its 4-byte
DB-record host store, 8-byte UAR write, WQE reads and CQE writes through that
fabric. All 35 deterministic row oracles pass; ten behavioral relation
families pass across 18 instances, while structural invariants remain fatal but
unscored. The review correction chains link-queue eligibility across one
transaction, separates posted and non-posted dependency horizons, and lets a
ready posted TLP fill an idle gap before a resource-blocked non-posted request;
posted placement is recomputed after credit availability so post-credit link
contention stays in the link ledger rather than disappearing or becoming a
false displacement error. Posted-after-completion remains legal and separately
accounted. See
[examples/rnic_pcie_v1/RESULTS.md](../../examples/rnic_pcie_v1/RESULTS.md).
The incidence draws are independent analytical surrogates: they do not claim
that the model detects a NUMA route, IOMMU or DDIO miss, ACS redirect or GPU
Direct event. Defaults remain synthetic, not a ConnectX-7 profile.
Service class is an accounting label in this closed slice; it does not affect
scheduling. The existing deterministic reservation order, including mandatory
posted forward progress, is the baseline that CORE-8's identity policy must
preserve. BACK-16 adds event-time mechanism precision without class-based
reordering. Optional class-aware policies remain CORE-10 completeness work,
and selecting identity must reproduce the accepted BACK-10 rows byte for byte.
BACK-16 owns active-path timing precision and calibration; BACK-17 owns
optional PCIe feature completeness.

On 2026-08-10 BACK-18 closed with the versioned `RnicDevice` composition
surface. The device owns or explicitly shares a stable-address fabric, owns an
inert network stub or accepts an external port pointer, preserves device
identity with QPC off, reports module-stage applicability and enforces scalar
versus fabric service exclusivity before state can mutate. A shared fabric's
config remains truthful at the device surface, and its ordering-domain claims
are enforced at construction and submission. Failed submissions do not
advance caller time. The commit-granular, post-specified
`B x doorbell-service` regression study passes all 6 direct-versus-composed
cells with exact field, timestamp and counter equality; separate PCIe and
inert-network directed scenarios also pass exactly. The predecessor artifact
gates remain byte identical through the composed probes: 11 of 11
`rnic_wq_v1` rows and 35 of 35 `rnic_pcie_v1` exact-oracle rows. Native CTest
passes all 4 entries. Evidence classes and reproduction commands are in
[examples/rnic_device_v1/RESULTS.md](../../examples/rnic_device_v1/RESULTS.md).

BACK-4 was retracted on 2026-08-03. Multi-QP striping as a DCQCN mitigation
was withdrawn by maintainer decision: DCQCN is the expected-fail comparator,
and its ECMP-collision and slow-start behavior is the phenomenon under study.

## Open tasks

Every task is labeled `(Category; priority; difficulty)`. P0 is a correctness,
state-integrity or validation-gate failure and outranks both categories. P1 is
active-path precision or completeness required by an accepted study or
milestone. P2 is deliberately disabled or bypassed feature coverage. Active-
path precision normally precedes P2 completeness. A disabled completeness
path must preserve the exact accepted baseline. Once a study enables that
path, errors in its behavior or calibration are precision work.

Difficulty is S for a localized change with local evidence, M for a change
that crosses one interface or needs one reproducible calibration, and L for
cross-layer work, hardware evidence or a multi-repository campaign. Difficulty
does not override priority, and correctness is never deferred because a fix
is difficult.

### Precision

- BACK-13 (Precision; P1; L): build a versioned CX-7 observable-state model
  and capture schema. Inventory only public Linux mlx5, rdma-core, NVIDIA
  MFT/DOCA and device-reported fields. Tag each as `documented`,
  `driver-inferred` or `calibrated-opaque`, with PSID, firmware, kernel,
  rdma-core, MFT, PCIe and topology provenance. Capture supported named
  registers, resource dumps, queue/counter snapshots, `ethtool -S`, RDMA
  hardware counters, `rdma resource`/`rdma statistic`, devlink health,
  DCB/PFC state, PCIe/AER/telemetry and tracepoints. Do not invent physical
  addresses, internal cache geometry, scheduler registers or firmware-
  private behavior.
- BACK-14 (Precision; P1; L): add an ibverbs capture/replay bridge for
  controlled calibration. Capture control verbs at QP/CQ/MR creation and
  modification, then capture data-path WR chains and CQ polls at the
  rdma-core mlx5 provider boundary, because the fast path bypasses the kernel
  and generic wrappers can be inlined or bypassed. Normalize both live
  capture and SimLLM lowering into the BACK-9 WR/WQE schema. An optional
  preload wrapper is a convenience path, not the signoff oracle. Preserve WR
  chains, SGEs, flags, queue identities, QP state and timestamps without
  recording payload contents by default.
- BACK-15 (Precision; P1; L): run the pre-registered RNIC calibration and
  boundary campaign. Start with DCQCN, then WQ/CQ and PCIe, QPC/cache, port
  loss and PFC. Sweep at least two dimensions per claim: WQ
  depth/batch/SGE/payload/signaling; QP and MR working sets; page size and
  context locality; PCIe width/NUMA/ordering; CQ depth/poll cadence;
  MTU/direction/loopback; loss location/rate/burst; DCQCN timers/rates/ECN;
  and PFC headroom/incast/RTT. Use Collie cases as reproducer seeds, not CX-7
  truth, since its Mellanox results are CX-6 and omit packet-loss,
  control-path and NDA diagnostic-counter details. Match transaction identity
  through the first loss or queue knee, classify every drop by evidence tier,
  and defend WQE latency, FCT/JCT, useful/raw bytes, queue depth, cache miss,
  retry, CQE, CNP and pause metrics.
- BACK-16 (Precision; P1; L): advance BACK-10's reproducible analytical
  profiles and generic FIFO approximations into mechanism-driven occurrence,
  correlation and measured calibration. Topology selects NUMA, ACS and GPU
  Direct routes; cache and translation state decide DDIO and IOMMU events,
  consuming ATS/ATC events from BACK-17 when that optional feature is enabled.
  Add event-time DMA/MMIO resource arbitration and occupancy so chronological
  arrivals can affect pending reservations. Reuse CORE-8's exact
  reservation-timeline and finite-capacity semantics. Apply PCIe legality and
  forward-progress rules before baseline selection; a resource-blocked
  non-posted read is not a legal ready candidate, so an eligible posted write
  can use the idle link. Identity ignores service class and must preserve every
  accepted BACK-10 row, timestamp, counter and random draw exactly. CORE-10
  owns optional non-identity class reordering.
  Add variable measured replay, the remaining PCIe RO/IDO/TC/VC ordering
  matrix and provenance-bearing CX-7 calibration. Calibrate tag-capacity knees
  for every mode enabled by BACK-17. Preserve deterministic replay and
  transactional sample state; extend run records with calibration provenance
  and exact draw ranges.
  Acceptance includes per-class attribution, calibrated queue and tag knees,
  and defended p50 through p99.9 latency. Until those mechanisms land,
  analytical incidence must not be described as detected hardware behavior.

### Completeness

- BACK-2 (Completeness; P2; S): LogGOPSim invocation helper for fast
  flow-level sweeps.
- BACK-8 (Completeness; P1; L): create the protocol-neutral SimLLM RNIC
  hardware extension under
  `simllm/backends/rnic/`. Its C++ event core must be independent of Python
  and of any one CC policy, compose with htsim through HTSIM-9, and preserve
  direct binary invocation. Define versioned configuration and result
  records, deterministic event ordering, opaque policy/fabric tokens and a
  hardware-bypass mode. The native session is the sole mutable WQE authority
  in structural mode and emits the versioned records from which bookkeeping
  and compatibility rows are projected. Acceptance requires the same hardware
  configuration hash across `rnic-nn`, `rnic-cn` and DCQCN comparison rows.
  Every bypass profile retained after composition, including packetized
  profiles, must preserve its accepted artifacts byte for byte; an
  intentionally unsupported legacy bypass must fail configuration explicitly.
  A directly invoked htsim run and a step-level run must prove live
  reachability: changing one nonzero native hardware service parameter changes
  the corresponding WQE timeline, per-flow FCT, JCT, step latency and at least
  one TTFT or TPOT outcome by the frozen relation. The test must fail if the
  native library is unlinked, the wrapper is bypassed or a second lifecycle
  authority is active.
  The standalone C++17 library, opaque flow-level `NetworkPort`, strict native
  build and deterministic fake adapter are complete. The SimLLM-owned
  component record layer is also complete: schema-tagged structural and bypass
  configuration/result records, canonical effective-hardware SHA-256,
  sole-authority bookkeeping and completion-CSV projections, exact authority
  counters, and the reusable bypass byte checker are complete. The frozen
  component study is
  [rnic_session_records_v1](../../examples/rnic_session_records_v1/RESULTS.md).
  BACK-8 remains open for the frozen live-reachability gate. HTSIM-9 owns the
  outer `AtlahsFlowRuntime` wrapper and concrete htsim-side adapter; CORE-4 and
  CORE-5 own graph invocation, `CompletionEvent`, step-result and TTFT/TPOT
  reduction. The modular composition entry point and external port injection
  seam are complete.
- BACK-9 (Completeness; P1; L): replace the timing-neutral WQE ledger with
  the structural **RDMA
  Work Queue**, merging the old WQE lifecycle and per-WQE-start work. Model
  verbs WR chains, WQE construction, SQ/RQ/SRQ rings, many-WQ CQ sharing,
  doorbell batches, WQEBB and WR indices, fences, inline data, signaled and
  unsignaled sends, receive consumption, finite depth, wrap and reclamation.
  The native RNIC session owns a registry of SQ, RQ, SRQ and standalone CQ
  objects. A send WQE has one SQ and send CQ; a receive WQE has one RQ or SRQ
  and receive CQ. Matching is a later event, not a remote-RQ parent on the send
  WQE. Multiple WQs may share one CQ, so CQ state must not remain embedded in
  one SQ object. Canonical result records use the stable endpoint, owning-WQ
  and post-sequence key and project exactly into the public bookkeeping
  schema. Successful unsignaled sends emit no CQE. One-sided operations emit
  no receive WQE, while SEND consumes one posted receive WQE or produces the
  modeled RNR outcome.
  CQ is a real host-memory queue with requester/responder/error CQEs, owner
  phase, producer/consumer indices, 64/128-byte format profiles, compression,
  moderation policy, polling, completion-channel notification requests and
  overrun. BACK-17 owns optional BlueFlame transport and MSI-X delivery, not
  the CQ's logical moderation, arming or polling policy.
  Normalized CQE content includes WR ID, QPN/source QP, opcode, status,
  opcode-valid byte count, immediate/invalidate data, flags, syndrome and
  vendor syndrome; provider-derived fields and valid bits stay explicit.
  Record optional capture-provenance `ibverbs_entry_at`, then native
  `posted_at`, `doorbelled_at`, `doorbell_seen_at`, WQE-fetch begin/end,
  `qpc_ready_at`, `admitted_at`, first/last packet, transport retirement, CQE
  visibility and poll time. Define NIC start as first-packet issue, never as
  `ibv_post_send` return. Reported per-WQE start latency is a derived difference
  over available timestamps, not a separately scheduled constant. The native
  model never fabricates an `ibverbs_entry_at` value when no capture is joined.
  The first one-SQ/one-CQ send slice is complete, including prefix acceptance,
  finite depth, batching, ordered retirement, signaling, poll-time reclaim,
  CQ wrap/owner generation and controlled first-failure evidence. Remaining
  scope includes RQ/SRQ, multiple WQs and shared CQs, WQEBB encoding, fences,
  inline WQE encodings, CQE format profiles, compression, moderation and
  completion-channel notification semantics, including an explicit modeled
  drain or teardown rule for an all-unsignaled tail. Acceptance includes two
  WQs sharing one CQ, RQ and SRQ receive matching, a one-sided no-RQ case, an
  unsignaled no-CQE case, later-signaled and modeled-drain or teardown
  reclamation, and exact native-result to public-projection reconciliation at
  quiescence.
- BACK-11 (Completeness; P1; L): implement QP lifecycle, RNIC pairing and
  context placement. Cover
  RESET, INIT, RTR, RTS, SQD/SQE, ERR and teardown; PD/MR/MPT/MTT ownership;
  peer QPN/PSN/GID/path exchange; retry/RNR parameters; and failed or timed-out
  pairing. Provide both manual out-of-band TCP pairing and `rdma_cm`/IB-CM
  pairing, with TCP treated as host control for RoCE/InfiniBand and as data
  transport only for iWARP. The generic memory hierarchy is `on_die_sram`, an
  optional `device_memory` tier and `host_pinned_memory`. The CX-7 default is
  an internal context cache plus host ICM over PCIe; the middle tier stays
  disabled until public evidence or measurements justify it. Every full-RNIC
  policy uses the same hardware QP objects; a separate opaque policy context
  carries DCQCN or `rnic-cn` identity. Migrate the current compatibility
  ledger without breaking its reader. Pair two RNIC endpoints explicitly;
  model TCP connect and attribute exchange, CM events and QP firmware-command
  time as control-path events. Model QPC, WQE-cache and MTT/MPT locality
  separately. QPC registration, ring page lists and data-region registration
  land in the BACK-19 host-memory model; QPC fetch never takes a per-access
  MKey/MTT translation while WQE rings and data buffers do.
- BACK-12 (Completeness; P1; L): implement the TX/RX hardware pipelines and
  cross-layer fault
  boundary. Include WQE decode, context/translation lookup, opcode-specific
  DMA, packetization, per-QP eligibility, arbitration, rate and PFC gates, MAC
  queues, RX matching/reassembly, SEND-to-RQ consumption, one-sided access,
  ACK/NAK/RNR, retry/timeout, error transition and CQE production. BACK-9 owns
  CQ and poll semantics; BACK-17 owns optional MSI-X delivery.
  Add deterministic, Bernoulli and burst injection at named TX,
  wire/switch and RX boundaries; every loss reports location, reason and
  controlled/asserted/inferred evidence. RNIC PFC covers per-priority
  headroom, XOFF/XON hysteresis, pause quanta/refresh, paused-egress gating and
  insufficient-headroom drops; HTSIM-9 transports the frames through the
  fabric. The DCQCN policy adapter is delivered and calibrated before wider
  PFC and programmable-CC work.
- BACK-17 (Completeness; P2; L): add optional PCIe mechanisms behind explicit
  enable, disable and rejection profiles. Cover mlx5 BlueFlame write-combining
  semantics and WQE-fetch bypass; ATS negotiation, ATC translation caching and
  fault production; negotiated read-tag capacity including optional 10-bit tag
  scaling; MSI-X vector routing, interrupt-side coalescing and interrupt writes
  that execute BACK-9's logical notification policy;
  QPC/ICM, MTT/MPT, payload, command and fault transaction adapters;
  and lower-layer DLLP, UpdateFC, replay, SKP and FEC events. Every disabled
  mode must preserve the accepted BACK-10 baseline exactly. Once enabled,
  timing, occurrence and calibration defects move to BACK-16 precision scope.
  BACK-11 and BACK-12 own when semantic lookup, DMA, CQE and fault events
  occur; BACK-17 only lowers still-unconnected events into their shared-fabric
  PCIe service classes.
- BACK-19 (Completeness; P1; L): add the virtual host-memory model that the
  QPC and DMA modules register into. Track every device-visible host object
  explicitly: QPC/ICM regions, SQ/RQ/CQ rings, doorbell records and data
  memory regions, each with owner, endpoint kind (host-pinned or GPU
  memory), page geometry and registration and teardown events, so BACK-11
  link building registers the QPC as a tracked allocation rather than a
  scalar lookup latency. Encode the translation asymmetry: QPC fetch is a
  context read on the QPC/ICM service class over device-managed ICM pages
  and never takes a per-access MKey/MPT/MTT translation; WQE rings are
  reached through the per-queue page list recorded in the QPC at creation;
  data buffers take the full MKey to MPT to MTT path. Translation and
  ATS/ATC events (BACK-16, BACK-17) therefore apply to rings and data but
  never to the QPC itself. The doorbell record is a located host-memory
  object whose address is registered at queue creation. Grounding: the
  public ConnectX PRM ICM and posting chapters, the upstream mlx5 QPC
  layout and the device-cache taxonomy recorded in
  [the RNIC hardware calibration plan](../papers/rnic-hardware-calibration.md).
  BACK-11 keeps QP lifecycle, pairing and cache residency; this model gives
  those caches their backing store and miss targets.
- BACK-20 (Completeness; P1; L): model the WQE submission source and the CQ
  consumer when the DMA module is enabled. Three producer shapes, selected
  per queue: a host CPU driver thread that writes WQEs into host-pinned
  rings, updates the doorbell record and rings the UAR register (the landed
  BACK-10 path); a CPU proxy fed by GPU-written descriptor queues in
  host-visible memory (the classic NCCL shape: the GPU never touches the
  NIC and completions return through host-mapped counters); and
  GPU-initiated submission (NCCL GIN, GDAKI/IBGDA), where GPU threads write
  WQEs and the doorbell record into GPU-memory rings, ring the GPU-mapped
  UAR register, and the NIC fetches WQEs and data through GPUDirect reads
  and writes CQEs into GPU memory. Every CQ names exactly one owning
  consumer, host driver or GPU, and the completion callback into the model
  runner is charged on that owner's path (VLLM-13 and CORE-5 consume the
  decision). Requires relaxing the current host-pinned-only endpoint
  validation for the SQ, CQ and doorbell-record paths to GPU memory,
  attributing initiator identity separately from the QP number, and driving
  the GPU-side producer as an explicitly submitted GPU task through the
  compute model's concurrent service, the same shape as its NCCL egress
  kernels (COMP-11 deepens that surrounding NCCL/NVLink model but does not
  own this producer). The QPC stays host ICM in every mode. COMP-2's fixed
  CPU-proxy versus GPU-initiated constants remain the analytical fallback
  while this structural path is disabled.
## Backend-repo follow-ups (tracked here, executed in their repos)

- HTSIM-1 (Completeness; P2; L): `rnic-ss` (Slingshot-like) profile wiring;
  the runtime factory
  rejects it with a clear error until the slingshot runtime lands. Its CLI
  options are already parsed so the flag ABI is stable. Out of simllm's
  scope by maintainer decision; tracked here for the backend repo only.
- HTSIM-2 (Precision; P1; M): goodput/state/queue trace flags for `rnic-cn`;
  they need trace
  hooks in the reviewed runtime first.
- HTSIM-4 (Completeness; P2; M): GOAL parser hardening and the checked-in
  `txt2bin` build target.
- HTSIM-5 (Precision; P1; L): persistent DCQCN policy state across hardware
  WQEs. On
  2026-08-07 the former hardware-specific per-WQE-start scope was merged into
  the BACK-9 RDMA Work Queue; this stable ID remains open for the unfinished
  CC behavior. One QP's alpha, current/target rate, CNP suppression, byte and
  timer recovery state must survive across its WQEs and reset only with the
  modeled QP lifecycle. A new QP starts at its configured line/local-QoS rate;
  HTSIM-9 carries CNP/ECN feedback to this policy and carries its rate update
  back to the SimLLM hardware gate. The policy never owns that gate. Doorbell,
  DMA and CQ costs are common across all policies and must not be charged only
  to DCQCN. Calibrate policy parameters against
  [docs/papers/msg-size-vs-bandwidth.md](../papers/msg-size-vs-bandwidth.md)
  using the DCQCN algorithm and vendor timer sets plus post-CNP repeated-WQE
  traces. The UCCL no-loss curve and 256 KB half-rate datum now calibrate the
  landed BACK-10 shared fabric plus BACK-9 Work Queue, and must not be fitted
  again in the policy. The
  existing micro-behavior anchors are in examples/dcqcn_micro. Source-level
  findings from the micro study's
  review, now the concrete work items: every send op constructs a fresh
  DCQCN source at line rate with no cross-WQE rate state
  (dcqcn_atlahs_runtime.cpp:398), the additive/hyper increase is
  R_AI = C/20 and C/10 (dcqcn.cpp:48-49) against the paper's fixed
  40 Mbps, and the ECN defaults are fixed bytes (Kmin 64 KB, Kmax
  640 KB, Pmax 0.25) independent of the link rate.
- HTSIM-7 (Precision; P1; L): rnic-cn concurrent same-pair flow scaling.
  10,000
  simultaneous flows between one source-destination pair make no visible
  progress within a 600 s wall-time budget (progress 0 percent, request
  queue 10,000; examples/dcqcn_micro addendum 1), far beyond the
  algorithm book's S_max regime but reachable by WQE-flood workloads;
  the measured per-flow control cost also scales with flow count, not
  bytes (16 KiB flood streams cap at 0.36 to 0.46 C). Both are adjacent to
  HTSIM-6 and BACK-9: policy lookahead removes the repeated declare cost,
  structural WQ backpressure limits how much work can be exposed, and the
  event-loop scaling needs its own look.
- HTSIM-6 (Precision; P1; L): `rnic-cn` policy lookahead (maintainer design
  2026-08-05). The
  established-pair fast path must not wait when granted bandwidth suffices,
  and the policy receives bounded lookahead from BACK-9 so it can pre-declare
  one RTT ahead for queued work toward the same destination. The WQ, WQE and
  QPC remain SimLLM hardware state; htsim retains only link-pair reservation,
  control-slot and predeclaration state. The timing-neutral SQ and directed
  link-pair identity in `d778326` remain the compatibility ledger until the
  adapter lands.
- HTSIM-8 (Precision; P0; M): repair the backend `commit_check.sh` validation
  gate. Current
  `origin/main` has no `validate_outputs` baselines, `validate.py` divides by
  zero in every attempted case, and the script lacks fail-fast handling, so
  it reports a false success. Add checked-in baselines or remove that compare,
  fix zero-flow diagnostics, and make every failed command fail the gate.
- HTSIM-9 (Completeness; P1; L): add the htsim side of the SimLLM RNIC
  extension. The combined
  session still implements `AtlahsFlowRuntime`, while the inner versioned port
  carries only opaque flow/packet tokens, transmit descriptors, delivery,
  drop/ECN, receive, pause and link-state events. Hardware submits an opaque
  CC-context token plus packet metadata; the policy returns eligibility/rate
  updates, and htsim returns delivery or feedback to the hardware. It must use
  the same SimLLM hardware implementation for `rnic-nn`, `rnic-cn` and DCQCN,
  transport PFC frames through htsim queues, and keep the fluid bypass
  explicit. No WQ, CQ, QP, QPC, PCIe, DMA or hardware scheduling state may
  live in this adapter. In structural mode `AtlahsHtsimApi` must not construct
  or mutate `AtlahsWqeLedger`; it delegates WR/WQE progression and completion
  to the SimLLM wrapper and returns only opaque network events. The legacy
  ledger remains available only in an explicitly labeled bypass run. Native
  and legacy WQE counters must never both advance in one session. Acceptance
  includes a mode-exclusivity assertion, exact token conservation and a
  directly invoked binary test in which a controlled htsim delay, drop or
  rate update reaches the native WQE timeline and final reported metric.
  Develop it only in the HTSIM repo's dated append-only addon branch, then
  update the SimLLM submodule pin.
- ATLAHS-1 (Completeness; P2; S): correct the vendored-fallback wording (the
  vendored htsim tree
  cannot satisfy the resolver) and pin a known-good HTSIM commit.
