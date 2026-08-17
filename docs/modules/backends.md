# simllm.backends and third_party

Invocation and result parsing for the network simulators, plus the pinned
backend submodules.

## Interface

- `HtsimRnicConfig` + `build_htsim_rnic_command` + `run_htsim_rnic`: direct
  GOAL-driven `htsim_rnic` runs (profiles `rnic-nn`, `rnic-nn-fluid`,
  `rnic-cn`; a run is valid only with `physical_quiescence=verified`),
  binary discovered via `SIMLLM_HTSIM_RNIC`, the README build location,
  then `PATH`.
- `simllm-htsim-flow-session-v1` (HTSIM-18): the opt-in framed stdin/stdout
  interface of the composed `htsim_rnic` binary. A 32-bit big-endian length
  prefixes each canonical JSON object. `open`, `inject`, inclusive
  virtual-time `advance`, `drain` and `close` retain one event list, topology,
  native RNIC authority and transport policy. Structural `rnic-nn` and
  generated `rnic-cn` are supported; the explicit nonstructural fluid mode is
  rejected. The unchanged one-GOAL command remains the exact default off path.
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
  lifecycle authority. Host-memory-enabled devices use the strict
  `simllm-rnic-effective-hardware-v3` projection, including allocation and
  page geometry plus the resolved submission producer, requester and CQ
  consumer. The native reader retains strict v2 compatibility, and disabled
  devices retain the accepted v1 bytes. The Python reader ingests and
  recursively freezes strict v1, v2 and v3 objects with native-matched
  allocation, page, submission, ownership and canonical-hash rejection. The
  reusable bypass checker guards the full reference input tuple and compares
  the four frozen behavioral artifact classes byte for byte.
- `ComposedRnicObservations` + `ComposedRnicSession`: strict validation and
  transactional projection of the frozen composed native rows into the core
  structural RNIC seam. The external native session owns WQE lifecycle and
  timing. The adapter tracks only atomic observation consumption and cannot
  advance a WQE or simulator resource.
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
- `LogGopsimConfig` + `build_loggopsim_command` + `run_loggopsim` +
  `parse_loggopsim_stdout` (BACK-2): the flow-level analytical seam. The same
  binary GOAL is costed with the LogGOPS model instead of a packet fabric, so
  a sweep that only needs a schedule completion time does not pay for a
  packet-level run. LogGOPS parameters keep the tool's own units, `L`, `o`,
  `g` and `O` in whole nanoseconds and `G` in nanoseconds per byte, under
  explicit `_ns` field names; parsed times convert to picoseconds by exactly
  1000. The parser reads both output shapes the tool can print, the per-host
  block and the batch-mode maximum, and treats a nonfinite `Average FCT` as
  absent. Discovery is `SIMLLM_LOGGOPSIM`, the `build/loggopsim` CMake
  layout, the ATLAHS submodule's own make output, then `PATH`; with none of
  those present the runner raises and names the environment variable. This is
  an invocation helper only, and TRAF-20 still owns the fluid fast fidelity
  level.
- `HtsimStepSink` + `HtsimStepSinkConfig` (M4): the closed-loop step sink,
  a callable `StepRecord -> StepResult | None` matching the adapters' sink
  contract. Per step its serial lowerer builds one `ExecutionGraph`; that
  graph's effective dependency inventory is the semantic ordering authority.
  The checked graph projector renders causal GOAL artifacts and htsim enforces
  exactly that projected order. A provider may return an optional exact
  duration per layer; the sink validates count, nonnegativity and the fused
  sum, then truncates cumulative boundaries to GOAL ns. Providers without the
  hook retain the original even scalar split byte for byte. An optional
  `StepRecord.num_sampled` prices the LM head from exact attribution; absence
  retains `len(scheduled)`. The config's optional `num_goal_ranks` pads
  topology-sized GOALs without moving the active group to the highest rank.
  The sink converts with `txt2bin`, runs `htsim_rnic` on the configured
  profile/topology, parses the completion CSV and returns the authoritative
  simulated makespan as the step latency with
  `completed_at_ps = record.virtual_time_ps + makespan`. A step with no fabric
  collectives returns `None`, so the adapter's own compute-only estimate
  stands. Per-step subprocess invocation is the documented diagnostic mode and
  remains the default.

  The seam-local `dependency_cross_check="atlahs-goal"` option independently
  renders and executes the same all-remote schedule through the direct ATLAHS
  GOAL path. The graph-projected execution remains the sole authority for the
  returned result. Its diagnostic report inspects every canonical effective
  edge for direct-GOAL syntactic reachability, identifies whole-operation and
  participant-local ordering-scope differences, and separately records raw
  phase-frontier gaps and the signed direct minus graph completion delta beyond
  the study-registered `dependency_cross_check_tolerance_ps`.
  Disagreement is reported with operation, phase and timing detail; it is not
  averaged, used to override the result or treated as an equality assertion.
  The default-off value preserves accepted artifacts and results exactly. The
  current cross-check rejects placement configurations with local NVLink work;
  TRAF-16 owns that frontier precision. `simllm.core.PrecisionConfig` owns the
  unified fidelity selection and `RunProvenance` its record, so this option is
  not a second global configuration scheme.
  `HtsimStepSinkConfig.precision` accepts an explicit surface and
  `selected_precision_levels` reports the compute, dependency, locality and
  network levels this configuration selects. A contradiction is refused during
  configuration validation, before the workdir, any GOAL artifact or any
  backend process exists. `HtsimRnicConfig` does the same for its profile
  spelling alone.
  `StepNetworkOutcome` keeps per-step bookkeeping (compute estimate, sample
  count and exactness, ordered layer calcs, makespan and network share) for
  reporting.
- `HtsimPersistentStepSink` (BRIDGE-1): the opt-in prepared-replay form of
  the same sink for a finite record sequence known before consumption.
  `prepare` copies and lowers the records serially, then a persistent local
  thread pool pipelines `txt2bin` and the unchanged isolated one-GOAL
  `htsim_rnic` invocations. Results remain unpublished until the complete
  batch succeeds and are served only for dataclass value-equal records in
  their original order. The pool can serve another batch after the first is
  fully consumed.
  This preserves the diagnostic path's reset semantics with a fresh process
  and local state for every GOAL artifact and step. This mode does not claim a
  stateful online backend session, and ordered `rnic-cn` multi-artifact runs
  are rejected before backend execution. The backend flow session and full
  result codec are now delivered; BRIDGE-2 owns their graph-level client.
- `SerialStepLowerer` + `SerialStepLowererConfig`: CORE-2 diagnostic lowering
  from a `StepRecord` to per-layer compute plus semantic TP/EP collective
  operations. Explicit framework observations bypass the fallback schedule and
  are enveloped without reconstructing framework policy. JSON-round-tripped
  graphs replay through `render_serial_execution_graph_goal`.
- `attribute_step_detail` + `HtsimRequestMetricReducer`: the read-only
  projection from executed steps to per-request TTFT and TPOT. Artifacts run
  serially and each composes as base plus the maximum of its local and fabric
  service, so the step's realized interval is one disjoint subinterval per
  artifact and the resource whose own service equals that maximum owns it.
  `MediumAttribution` names `kernel_ps`, `nvlink_ps`, `fabric_ps`,
  `co_critical_ps` and `collective_base_ps` separately, alongside `queue_ps`
  and `control_ps`, and totals the same picoseconds as the coarse
  `LatencyAttribution` it rolls up into. `MaskedMediumService` reports what the
  losing medium ran concurrently; it is a work sum, has no total, and never
  enters a latency partition. `attribute_step` returns the coarse partition
  alone. The evidence comes from `StepLocalityOutcome`'s per-artifact
  `local_phase_service_ps`, `base_phase_latency_ps` and `local_phase_medium`;
  an outcome that carries NVLink work without them is refused rather than
  approximated, and an all-remote outcome without them keeps its exact
  historical partition.

## Pinned submodules

| Submodule | Repo | Ref | Provides |
|---|---|---|---|
| `third_party/atlahs` | [ATLAHS-rnic-private](https://github.com/yifeng-ethz/ATLAHS-rnic-private) | `main` | GOAL toolchain (txt2bin, LogGOPSim, goal_gen), validated `htsim_rnic` launcher (`atlahs_entry.py`) |
| `third_party/htsim` | [HTSIM-rnic-private](https://github.com/yifeng-ethz/HTSIM-rnic-private) | `main` | UEC htsim, the composed SimLLM RNIC wrapper behind `HTSIM_ENABLE_SIMLLM_RNIC`, `htsim_rnic`, WQE bookkeeping, the ABI-v2 event relay with its physical control producers, and the persistent flow session |

As of 2026-08-03 the launcher, the RNIC wiring, the DCQCN comparator
(mlx5-faithful loss recovery, ECN-only and ECN plus PFC modes, storm
metrics) and the full rnic-cn algorithm-book implementation
(deterministic reservation ledger, windowed feedforward snapshots,
fractional nflow, sender egress composition, BJP-derived resequencing
window) are merged. The SimLLM pin for HTSim is on backend main, which now
carries the WQE bookkeeping commit, the composed SimLLM RNIC wrapper and the
ABI-v2 event relay. A pin to an append-only `<date>/simllm-addon` branch
remains an intentional supported state while backend work is in review, but
it is an intermediate state rather than the steady one. The same HTSIM
sources build on Linux with
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

The composition links the SimLLM C++ library into the directly invoked htsim
binary, with no Python callback in the packet event loop. The composed runtime
presents `AtlahsFlowRuntime` to `AtlahsHtsimApi`; the pinned backend main
contains this link and passed frozen Tier A and Tier B at ABI v1. The wrapper
and versioned flow, packet-attempt and transport-control event relay are
component-live. The qualifying Tier C run carries ABI-v2 explicit TX-start
evidence through the native timeline and live metric chain while its separate
link-OFF binaries preserve the frozen ABI-v1 bypass artifacts. HTSIM-9 is
closed on that run. The SimLLM hardware runtime calls an htsim policy and
fabric using opaque flow and packet tokens. QP, WQE, CQ, QPC, PCIe and DMA
objects never cross that boundary.

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

### HTSIM-9 wrapper seam

The SimLLM-side executable preparation package is in the
[Tier A harness results](../../examples/rnic_live_v1/tier_a_harness_results.md).
Its generic scenario runner receives a `PortFactory` that supplies the
versioned `NetworkPort`, an external-event pump and read-only issue and
terminal traces.
The physical factory configuration contains ABI version, capacity, link rate,
packet geometry, zero-propagation fixture controls and controlled-drop
selection. It does not contain native doorbell service D or a precomputed
serialization time.
The fake implementation composes the existing deterministic fake port. The
HTSIM-9 binary compiles the same runner and main and replaces only the factory
translation unit.

Preparation behavior was frozen at `35c2ee4` before implementation. A first
nonfinal smoke exposed a two-expression Python `Counter` defect; the
post-specified machinery-only correction is `21f9a4c`, and its chronology is
[recorded separately](../../examples/rnic_live_v1/tier_a_checker_correction.md).
Implementation `f8eeb34` and the subsequent registered fake run pass all four
instances in each of the D-additivity, inverse-rate and FIFO families, eight
separate exact rows and all fatal unscored families. This remains component
evidence. It creates no htsim composition, `CompletionEvent`, `StepResult`,
TTFT or TPOT result.

The complete source-cited event map, ABI gap assignment and requested addon
branch procedure are in the
[HTSIM-9 wrapper design and approval package](../design/htsim9-atlahs-flow-runtime-wrapper.md).
The original frozen gate is unchanged; its landed-surface review and proposed
maintainer-only clarification are in the
[post-specified fixture audit](../../examples/rnic_live_v1/FIXTURE_AUDIT_2026-08-10.md).

### Modular construction

The native device is assembled through the versioned `RnicDeviceConfig` and
`RnicDevice` composition entry point. It joins the work-queue core with the
scalar QPC compatibility module, optional DMA (`PcieFabric` plus
`WorkQueuePcieBinding`), optional `VirtualHostMemory`, and either an injected
versioned `NetworkPort` or an owned inert port. DMA composition also resolves
one versioned submission profile for the queue. The QP number and
policy-context token remain device-level identity, including when QPC is
disabled. Both native probes and every composed-session test construct
through this entry point; direct module construction remains only in
component tests and exact oracle pairs.

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
An enabled host-memory module transactionally registers the configured QPC,
SQ, RQ, CQ, doorbell-record and data allocations before the device becomes
callable. An owned registry is heap-stable; an attached registry is retained
by shared ownership and must have the same effective registry config. Queue
and data accesses commit their read-only access records with the same PCIe
plan that supplies their timestamps. Explicit teardown requires a quiescent
queue, records one teardown event per live device-owned allocation and makes
later device operations reject. Default construction does not allocate a
registry and preserves the accepted device and session-record bytes.
Each enabled device claims its nonzero `device_owner_id` in the registry
before planning any registration. A shared registry rejects a duplicate live
claim or a claim over pre-existing live allocations. Claimed registrations
and teardown require the same device identity, and a failed foreign operation
leaves allocations, lifecycle evidence and generation unchanged. WQE data
descriptors must resolve to a `DataRegion` owned by the posting device even
when another device uses the same numeric MKey in its own namespace.
The submission profile selects a host CPU driver, a CPU proxy fed by one
GPU-written host-visible descriptor queue, or a GPU-initiated producer. It
names the producer, RNIC requester and sole CQ consumer independently from the
QP number. Host and proxy shapes keep SQ, CQ and doorbell records in pinned
host memory. The GPU shape requires those objects in GPU memory and marks the
MMIO UAR mapping as GPU-owned. QPC/ICM remains host-pinned in every shape.
Successful doorbells and CQ polls append read-only submission and consumption
records joined to the existing WQE and CQ lifecycle; they never become a
second authority.
The required RQ allocation is only a typed registration placeholder in this
one-SQ/one-CQ SEND slice. There is no active RQ identity, fetch path or receive
consumer yet, so its nonzero `owner_id` and host/GPU endpoint are recorded but
are deliberately not matched to the send producer shape. The allocation must
still name the device owner and use the `ReceiveQueue` owner kind. BACK-9 owns
the RQ/SRQ registry, active receive path and the endpoint and identity checks
that become mandatory when receive execution is enabled.
The absent-network path owns an inert port that accepts with a fresh token and
delivers on the device progress pump. The composed path injects the concrete
htsim port landed by HTSIM-9; BACK-25 and BACK-26 added its ABI-v2 event
vocabulary. BACK-27 now connects CPU-proxy descriptor production and
GPU-initiated WQE production to timed tasks in the concurrent compute service.
The compute scheduler is the sole producer-task timing authority. Each native
submission record carries only its validated immutable task identity and queue
timestamps. The coupling is disabled by default, the caller-timestamp path
remains an explicit bypass for non-host shapes, and host-CPU submission stays
compute-free. The
[GPU producer study](../../examples/rnic_gpu_producer_v1/RESULTS.md) measures
exact issue-sharing and residency-delay relations while retaining all accepted
default bytes. BACK-37 owns the remaining GPU CQ-consumer and runner-callback
work. VLLM-13 and CORE-5 consume the recorded CQ-owner decision once that path
is live. COMP-28's fixed CPU-proxy and GPU-initiated constants remain the
analytical fallback while structural submission is disabled.

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

The current `AtlahsWqeLedger` remains the sole authority only in explicit
hardware-bypass mode. The composed structural path instead selects the native
session as sole WQE authority. BACK-9 and BACK-12 deepen that structural path
without changing this exclusivity. The HTSIM-9 Tier C projection consumes its
explicit packet events without creating another lifecycle. A WQE has no
single scheduled start constant. The model records post, doorbell publication
and observation, WQE fetch or BlueFlame transfer, QPC readiness, scheduler
admission, first and last packet, transport retirement, CQE visibility and CQ
polling separately.
NIC start is first-packet issue. A reduced per-WQE start latency is derived
from the native timeline for calibration and never charged again by htsim.
Request routing lifetime stops at the semantic collective and its expanded
flow or WQE granularity. ABI-v2 packet-attempt events remain backend-private
and are not joined to request identity. BACK-39 records the canonical
per-request byte extent, boundary packetization, attempt, retry and terminal
reconciliation required before that boundary may move.
The pre-implementation composition expectations were first frozen in
[examples/rnic_live_v1](../../examples/rnic_live_v1/expectations.md) at commit
`65b5609`; commit `facb26d` clarified retry identity, commit `947399c`
recorded the drain and audit wording, and commit `d5d98a2` is the final pre-run
amendment to that gate.
The evidence classes, mlx5 hook and boundary-test matrix are recorded in
[the RNIC hardware calibration plan](../papers/rnic-hardware-calibration.md).

## Status

On 2026-08-14 BACK-43 closed. Per-request attribution used to refuse every
step whose locality projection carried NVLink bytes or NVLink service, so any
placement that co-located two ranks took the reducer offline. The sink now
publishes each artifact's local service, semantic base latency and owning
medium, and attribution charges the artifact's realized service to the
resource whose own service equals the composed maximum, keeping the NVLink and
fabric components under separate names and reporting the losing medium's
masked service as a work sum outside every total. Its
[frozen study](../../examples/mixed_attribution_v1/RESULTS.md) held all 8
fatal guards, passed its scored exact relation 1 of 1 and passed its scored
behavioral relations 3 of 4 as written. The one miss is a mis-registration in
the freeze rather than a measurement: F1 attaches a single absolute NVLink
interval to both all-local cells while deriving it at the full rate, so the
half-rate cell cannot meet it, and F3's frozen relative bracket covers that
cell instead. A single two-node step reached per-request TTFT
with 24 NVLink-owned and 24 fabric-owned artifacts whose components total the
TTFT exactly, halving the NVLink rate moved that TTFT by exactly the 120,000
ps doubling of the NVLink-owned service while the fabric component stayed
identical to the picosecond, and the all-remote path stayed byte-identical
against both a pytest regression lock and an in-run replay of the pre-BACK-43
input shape. Measured fabric and NVLink services reproduce their closed forms
to the picosecond. Every artifact carrying a fabric segment was fabric owned,
because the model's 2.000 us per-phase propagation term is 150x to 400x above
the local serialization at these payloads, so BACK-45 owns qualification near
the ownership crossing point and BACK-44 owns the tensor-parallel plus
expert-parallel graph the study could not plan.

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

The TRAF-12 follow-up keeps the independently rendered ATLAHS GOAL execution
available behind the serial sink's explicit dependency cross-check. The
authoritative graph-projected execution still supplies the returned makespan;
the second execution reports its ordering-scope, raw phase-frontier and
completion-time disagreements for diagnosis. The all-remote structural audit
checked all 423 canonical effective edges and found 235 differences: the
frozen 47/47 whole-operation logical-queue FIFO differences plus 188
participant-local syntactic-frontier mismatches added as a post-specified,
unscored diagnostic. Raw timing remained scoped to the 47 frozen boundaries,
with 46/47 unequal, early gaps. The default-off path retained the accepted
artifacts and results exactly; see
[the dependency authority results](../../examples/dependency_authority_v1/RESULTS.md).

On 2026-08-11 BRIDGE-1 closed for finite known replays. The opt-in
`HtsimPersistentStepSink` reuses a local worker pool and concurrently executes
the unchanged isolated one-GOAL path. Its
[frozen study](../../examples/bridge_persistent_v1/RESULTS.md) retained every
step result, outcome, GOAL text, GOAL binary and completion CSV byte for byte
across both recorded M4 TP 8 replays. Four and eight workers reduced wall time
by 3.36x to 5.43x across the four scored cells. Diagnostic invocation remains
the default; BRIDGE-2 remains the online graph-level client.

On 2026-08-11 HTSIM-18 closed with paired backend commit
`f8e1ee923a9c108cd698786c1824b9722d22d0e1`. The opt-in
`simllm-htsim-flow-session-v1` process retains native event, topology, RNIC and
transport state from open through close. Its
[frozen study](../../examples/persistent_session_v1/RESULTS.md) matched both
stateless-equivalent latency streams byte for byte, while overlapping
same-source flows raised the second FCT and source SQ high-water mark in both
scored state cells. Both measured wall-clock cells were faster than isolated
one-GOAL runs. Their corrected bands are diagnostic only because the wall-only
amendment followed a precommit session smoke; HTSIM-24 owns a clean held-out
wall study. The one-GOAL stdout, stderr, completion CSV and help bytes remained
identical to the base binary. CORE-24 supplies the paired full result codec;
BRIDGE-2 remains above this lower-level flow interface.

On 2026-08-13 HTSIM-24 closed. The
[held-out wall study](../../examples/persistent_session_wall_v1/RESULTS.md)
requalified the wall-clock family on two bidirectional-ring replays generated
by a topology rule frozen before any local timing command, with two-sided bands
materialized mechanically from a base-CLI-only calibration and committed as a
band lock before the session option was invoked. Both replays pass every band
and the signed speedup instance, `2/2` genuine risk: the complete persistent
boundary is 6.16x faster than the complete isolated boundary on the 6-flow
replay and 5.96x faster on the 10-flow replay, against a predeclared 1.1x
minimum. Both sit below the 12x and 20x process-count ratios that bound what
retaining one process can save, and both boundaries scale near-linearly with
flow count. Every fatal guard held, including byte-identical ordered FCT lists
between the isolated and persistent paths, so the run is valid rather than
void. The diagnostic wave-5 bands are superseded.

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
ownership. Flow-level acceptance/outcome timestamps remain separate from
packet issue timestamps. At that checkpoint the htsim wrapper was not yet
connected and the old HTSIM ledger remained the live compatibility path. The
later BACK-8 closure below records the ABI-v1 composition and Tier B
evidence, and the BACK-25/26 closure records the packet vocabulary.
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
reordering. The class-aware strict-priority and weighted-round-robin policies
that CORE-10 landed live at the core graph-operation seam and no PCIe
reservation consults them, so selecting identity must still reproduce the
accepted BACK-10 rows byte for byte and no PCIe row moves.
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

On 2026-08-10 BACK-19 closed with the versioned `VirtualHostMemory` registry
and its `RnicDevice` composition path. QPC/ICM, SQ, RQ, CQ, doorbell-record and
data allocations carry typed ownership, endpoint and path, virtual extent,
page geometry and transactional registration and teardown evidence. QPC
fetches issue direct `QpcIcm` reads with no MKey, MPT or MTT stage. SQ and CQ
accesses resolve their recorded queue page lists, while data reads use the
MKey, MPT and MTT chain; all physical transactions commit through the shared
`PcieFabric` plan. The doorbell record is addressed through its allocation,
and explicit teardown rejects live queue state and all later device use.
Enabled configurations projected every allocation into strict effective-
hardware v2 bytes at that commit. BACK-20 supersedes newly rendered enabled
records with strict v3 bytes while retaining v2 validation. The disabled path
retains all five predecessor artifacts exactly. The frozen study passes 4 of
4 translation-asymmetry cells, ten translation-free QPC fetches, 5 of 5
byte-identity instances and 5 of 5 native CTest entries. Evidence classes and
reproduction commands are in
[examples/rnic_hostmem_v1/RESULTS.md](../../examples/rnic_hostmem_v1/RESULTS.md).
The integration-review correction adds an exclusive registry claim for every
live device owner and rejects cross-device data allocations even when their
numeric MKeys match. Directed tests preserve the registry generation and all
allocations on duplicate claims and foreign teardown, then exercise explicit
teardown followed by destruction without termination.

On 2026-08-10 BACK-20 closed with the versioned submission profile and its
read-only submission and CQ-consumption ledgers. The profile selects host CPU
driver, CPU proxy or GPU-initiated ownership per composed queue. CPU proxy
mode registers the GPU writer's host-visible descriptor queue. GPU-initiated
mode accepts GPU-memory SQ, CQ and doorbell allocations and a GPU-owned UAR
mapping; QPC/ICM remains host-pinned and direct in all modes. Producer, RNIC
requester, CQ consumer and QP identities are independent fields. The default
host CPU shape resolves zero compatibility identities to the QP number, so
existing PCIe requester bytes and all six accepted predecessor artifacts stay
unchanged. Enabled host-memory devices render strict effective-hardware v3
records with the resolved profile, while the native parser retains strict v2
compatibility. The frozen `producer-shape x batch-size` study passes 6 of 6
translation-asymmetry cells, fifteen translation-free QPC fetches, 6 of 6
byte-identity instances and 6 of 6 native CTest entries. Evidence classes and
reproduction commands are in
[examples/rnic_submission_v1/RESULTS.md](../../examples/rnic_submission_v1/RESULTS.md).
The post-specified integration-review correction makes the CSV
`producer_kind` field project the producer agent taxonomy, so GPU-initiated
rows now record kind `gpu` while retaining shape `gpu_initiated`.

On 2026-08-11 BACK-28 closed strict Python ingestion of the native
effective-hardware v2 and v3 objects. Four native-emitted v2/v3 controls are
accepted, retain every projected field and array value, and are recursively
immutable. The frozen rejection corpus covers 100 native branches across
schema, fabric, path, submission, sole-CQ-consumer, host-memory allocation,
page, binding, descriptor-ownership, work-queue and canonical-hash checks.
Native and Python readers both rejected all 100, with exact acceptance-bit
agreement in every case. The v1 structural object and complete config plus the
bypass config retain their frozen hashes and parsed identities. Evidence,
entailment analysis and reproduction commands are in
[examples/rnic_records_v3/RESULTS.md](../../examples/rnic_records_v3/RESULTS.md).

On 2026-08-11 BACK-8 closed for the clauses demonstrated across its component,
Tier A and Tier B gates. The session-record study established versioned
records, policy-invariant hardware hashes, authority counters, projection
identity and bypass comparison machinery. Tier A established the directly
invoked composed binary, native WQE and per-flow completion movement, sole
structural authority, exact single-WQE and FIFO relations, and step-sink
replay. Tier B projected immutable native observations through
`ExecutionGraph -> CoarseDeviceRuntime -> CompletionEvent -> ExecutionResult
-> StepResult -> TTFT/TPOT`. Its six genuine-risk families passed 4/4 D
additivity, 4/4 inverse-rate serialization, 8/8 live metric forms, 8/8
seven-component rows, 4/4 FIFO contention and 4/4 bypass artifact identity.
The W1 queue wait was exactly L, the selected `nic_owner` attribution conserved
every request latency, and all four protected bypass profiles matched the
frozen reference. The bypass family's discriminating backend artifacts are
the completion CSV and canonical completion rows; its scalar-derived
StepResult and request-summary arrays are weaker projections. The review fix
routes the comparison through the repository `BypassArtifacts` comparator.
See the
[Tier B results](../../examples/rnic_live_v1/RESULTS.md#tier-b-live-reachability).

Tier B kept failed adapter transaction atomicity as unit-test evidence and did
not run its same-graph or link-disabled residuals. The subsequent
[RNIC authority comparison](../../examples/rnic_authority_v1/RESULTS.md)
closed both residuals as CORE-21 and BACK-31. One canonical graph traversed
the timing-neutral and composed authorities through the deployed reducer,
passing the signed metric family 6/6 and the inverse-rate family 12/12. Each
live structural cell recorded the failed 0/0 transaction and one two-WQE
retry. A fresh build from the same pinned htsim source set the SimLLM native
link OFF, ran its unconditional RNIC main through the registered producer,
and was rejected before observations or results existed. The separate
positive binaries and repository-standard bypass bundle remained exact. The
result ledger quotes and maps every registered CORE-21 and BACK-31 clause; no
residual remains. HTSIM-1 retains explicit rejection of the unsupported
`rnic-ss` legacy profile. At the Tier B checkpoint HTSIM-9 remained open for a
composed run showing first-packet and last-packet issue, since ABI-v1 network
acceptance and whole-flow terminal events are not substitutes for packet
issue; the Tier C update below records its closure and the corrected
binary-role diagnosis.

On 2026-08-11 BACK-25 and BACK-26 closed at the versioned vocabulary and
relay boundary. NetworkPort ABI v2 carries session-unique packet-attempt
identity, explicit TX start and finish, RX arrival, attempt terminals, typed
drop evidence, ECN/CNP, effective eligibility and rate updates, PFC and
link-state forms. ABI v1 remains the exact default compatibility path, and a
v2 consumer rejects a v1-only producer rather than silently degrading. The
unbound Tier A serializer populates the packet-study rows; the physical
packetized manifold independently emits packet observations from committed
serializer boundaries in the directed composition test. At that checkpoint,
enabled control-form relay evidence came from a test runtime, while the
packetized manifold advertised packet attempts alone. Evidence and the
labeled post-specified review corrections are in
[rnic_packet_v2](../../examples/rnic_packet_v2/RESULTS.md).

On 2026-08-11 HTSIM-15, HTSIM-16 and BACK-34 closed at their registered
component scopes. The physical DCQCN runtime now emits packet-correlated ECN
and CNP, policy-context rate and eligibility updates, real lossless-fabric PFC
submission, pause and resume, and timestamped dynamic endpoint-link state.
Capabilities are present only when each physical producer is enabled. The
registered six-condition study scores 15 of 15 genuine-risk relations before
its fatal exact oracles: 2 of 2 signed CNP rate changes, 2 of 2 PFC intervals,
2 of 2 dynamic-link completion changes, 1 of 1 hold-duration spacing, 6 of 6
control-disabled physical identities and 2 of 2 ABI-v1 byte identities. Late
CNPs retain packet correlation after delivery while the extent remains live.
See the [physical control results](../../examples/rnic_control_v2/RESULTS.md).

The paired BACK-34 cell uses a 5,000-byte payload at the 4,096-byte wire
quantum. Tier A and the directed composed runtime both observe a 968-byte
payload tail in a 1,032-byte wire packet with exact committed TX and RX
boundaries. Its 3 of 3 compatibility relations preserve the accepted
full-quantum ABI-v2 projection and both ABI-v1 artifacts. The tail's exact
geometry and times remain fatal unscored component oracles. See the
[BACK-34 results](../../examples/rnic_packet_v2/BACK34_RESULTS.md).

On 2026-08-11 the HTSIM-9 Tier C implementation connected ABI-v2 data and
retransmission TX-start events to native `first_packet_at_ps` and
`last_packet_at_ps`, then projected first-packet issue through
`ExecutionGraph -> CompletionEvent -> StepResult -> TTFT/TPOT`. The qualifying
registered run used audited htsim commit `4885c64` in two explicit roles: a
link-ON composed binary for the live chain and link-OFF RNIC and DCQCN binaries
for the frozen Tier B bypass rows. All accepted ABI-v1 Tier A and Tier B files
were byte-identical. Tier B passed every family, including bypass identity
4 of 4. Ruff, 686 pytest tests with 5 skips, all 370 htsim CTest cases and all
6 standalone native CTest cases passed.

The run passed 4 of 4 doorbell packet-to-live instances and 4 of 4 link-rate
packet-to-live instances with the frozen signs and exact magnitudes. The
checker evaluated raw cross-cell observations before its packet exact oracle
and inherited Tier B checker, so neither scored family was entailed. The
acceptance-surrogate, producer-constant and missing-TX-start controls failed
as required and remain fatal-unscored. The 1 MiB cells placed last-packet
issue strictly after acceptance and strictly before whole-flow terminal time.

HTSIM-9 closes against each registered clause. "one composed run of the Tier
B class passes" is supported by the single outer invocation and its complete
Tier B result. "ABI-v2 packet-issue evidence populating the native timeline
through `ExecutionGraph` to `CompletionEvent`, `StepResult`, TTFT and TPOT" is
supported by both 4 of 4 live-chain families and the exact event projection.
"Network acceptance and whole-flow terminal events do not satisfy that
evidence" is supported by the separation cells, explicit TX-start origin and
rejected acceptance surrogate. No closure clause remains, so no residual task
was registered. See the
[Tier C results](../../examples/rnic_live_v1/RESULTS.md#tier-c-abi-v2-packet-chain-chronology-and-closure).

HTSIM-19 is retired without a backend change and its ID will not be reused.
The earlier P0 entry incorrectly treated a 2 of 4 Tier B bypass result as a
backend-main regression. Three unchanged-command reproductions showed 4 of 4
with the then-current `4885c64` link OFF, 2 of 4 with a pre-v2 link-ON build and
4 of 4
with the frozen wave-4 link-OFF build. The signature follows the link setting.
A link-ON binary selects the structural session for `rnic-nn` and `rnic-cn`
by design and is not the legacy bypass candidate. The harness now keeps those
binary roles separate; no HTSIM residual survives.

BACK-4 was retracted on 2026-08-03. Multi-QP striping as a DCQCN mitigation
was withdrawn by maintainer decision: DCQCN is the expected-fail comparator,
and its ECMP-collision and slow-start behavior is the phenomenon under study.

HTSIM-2 closed on 2026-08-13. `rnic-cn` now carries
`-rnic_cn_goodput_trace_csv` with `-rnic_cn_goodput_trace_bin_ps`,
`-rnic_cn_state_trace_csv`, and `-rnic_cn_queue_trace_csv` with
`-rnic_cn_queue_trace_max_rows`. Every flag is off by default, each pair is
all or nothing, and all five are rejected for the profiles that cannot produce
them. The two shared trace components were already profile neutral; the new
`AtlahsQueueTrace` is the third, consuming the ns-tm3 switch observation
boundary that no profile previously read. Seven recording points sit in the
reviewed runtime, goodput at the receiver's in-order release and sender state
at declare, rate activation, immediate feedback, nflow raise, retirement and
delivery completion. The untraced binary is byte-identical to the pre-change
binary and the traced run differs only by one observation manifest line
([rnic_cn_trace_v1](../../examples/rnic_cn_trace_v1/RESULTS.md), 22 of 27
scored instances, backend ctest 358 of 358).

Two frozen relations in that study were refuted and are recorded here because
they bear on how the traces may be read. Goodput is binned on the receiver's
in-order release, not on wire arrival, so a bin total may exceed the link's
per-bin byte budget by the resequencing burst; the observed excess never
exceeded one maximum DATA payload. And the `rnic-cn` control-packet population
is time driven and therefore rate dependent, while the DATA population is not:
the same GOAL produced exactly 1024 DATA switch enqueues at both 400 and
200 Gbit/s and 242 against 772 control enqueues. Comparisons across link rates
must separate the two.

BACK-2 closed on 2026-08-13. `simllm/backends/loggopsim.py` drives the
unmodified LogGOPSim binary over the same binary GOAL the htsim helpers use.
Fifteen exact argument and parse oracles pin the option grammar, both output
shapes and the picosecond conversion, and a live two-by-two sweep over message
size and per-byte gap reproduces the LogGOPS cost model on four of four scored
instances with an invariant 6500 ns constant, every cell above its own
serialization floor
([loggopsim_helper_v1](../../examples/loggopsim_helper_v1/RESULTS.md)). The
helper is the invocation seam only; TRAF-20 still owns the fluid fast level.

HTSIM-25 and HTSIM-8 closed on 2026-08-13, each against its own acceptance
clauses. An exact bound-authorship reproduction classified all 17 previously
out-of-bounds experiments as wrong at authorship, with zero stale bounds, zero
simulator regressions and zero unresolved cases. Five corrected authorities
preserve fractional slack from plans matched on one active modeled resource;
the failed-link family uses the actual 25 Gbit/s serialization floor rather
than aggregate capacity to bound a maximum per-flow statistic. The exact final
backend commit runs all eight default plans and all 95 experiments with raw
gate status zero.
A deliberate tracked-plan mutant returns nonzero, byte-exact restoration is
proved, and the restored plan returns zero
([htsim_uec_bounds_v1](../../examples/htsim_uec_bounds_v1/RESULTS.md), C1
17 of 17, P1 11 of 11, G1 1 of 1 and M1 1 of 1 kept as separate evidence
classes).

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

ID note: the BACK-34 partial-final-packet record reserved BACK-46, BACK-47 and
BACK-48 for residuals it then reported as not created, so no registry ever
carried them. The packet-device model change is their first registration and
gives them the meanings below; the earlier record's "no residual entry is
created" statement stands and refers to different, never-registered work.

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
  accepted BACK-10 row, timestamp, counter and random draw exactly. Optional
  non-identity class reordering here would reuse the landed core policies
  rather than growing a second policy surface.
  Add variable measured replay, the remaining PCIe RO/IDO/TC/VC ordering
  matrix and provenance-bearing CX-7 calibration. Calibrate tag-capacity knees
  for every mode enabled by BACK-17. Preserve deterministic replay and
  transactional sample state; extend run records with calibration provenance
  and exact draw ranges.
  Acceptance includes per-class attribution, calibrated queue and tag knees,
  and defended p50 through p99.9 latency. Until those mechanisms land,
  analytical incidence must not be described as detected hardware behavior.
- BACK-38 (Precision; P1; L): preserve htsim topology, RNG,
  transport, congestion-control and RNIC state across ordered GOAL artifacts
  instead of starting a fresh process at every boundary. Multi-artifact
  `rnic-cn` currently fails before backend execution, while `rnic-nn` and
  `rnic-nn-fluid` remain accepted. Acceptance must execute one checked graph
  projection in a state-preserving session, reconcile every artifact and
  completion identity, and retain the current rejection and stateless-profile
  bytes as the explicit off paths.
  BACK-38 is blocked behind HTSIM-28 because the delivered session cannot
  reuse a completion time it has just exposed as the dependent injection
  boundary; see [the protocol audit](../../examples/congestion_chain_v1/RESULTS.md).
- BACK-45 (Precision; P1; M): qualify per-artifact ownership near the crossing
  point where the NVLink and fabric services of one artifact are comparable.
  Every artifact `examples/mixed_attribution_v1` measured sat 150x to 400x away
  from that boundary, so the argmax rule is evidenced only in its extremes and
  the `co_critical_ps` component has unit evidence alone. The comparison is
  also biased: the local term charges the maximum of endpoint egress and
  ingress since CORE-41, while the cross-node term still has no
  destination-ingress serializer (CORE-48), so a converging combine is
  under-charged and a near-boundary artifact can be assigned to NVLink that a
  fully modeled fabric would own. Acceptance: a cell whose two media sit within
  a small factor, run with the ingress-aware fabric term, shows ownership
  flipping in the registered direction and reports the flip through the
  per-request components, while the far-from-crossing cells keep their measured
  values exactly.

### Completeness

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
  pairing. Provide both manual out-of-band TCP pairing and `rdma_cm` or IB-CM
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
  use the landed `VirtualHostMemory` model. QPC fetch never takes a per-access
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
  that execute BACK-9's logical notification policy; optional cache-hit bypass
  and ATS/ATC behavior around the landed QPC/ICM, queue-page-list and MTT/MPT
  transaction adapters; command and fault transaction adapters; and
  lower-layer DLLP, UpdateFC, replay, SKP and FEC events. Every disabled
  mode must preserve the accepted BACK-10 baseline exactly. Once enabled,
  timing, occurrence and calibration defects move to BACK-16 precision scope.
  BACK-11 and BACK-12 own when semantic lookup, DMA, CQE and fault events
  occur; BACK-17 only lowers optional events not already represented by the
  landed base transaction path into shared-fabric PCIe service classes.
- BACK-37 (Completeness; P1; L): connect the GPU-owned CQ consumer and its
  runner callback to explicitly submitted work on the concurrent compute
  service. The current enabled producer path stops at the immutable submission
  task link; CQ polling and callback work still use caller-supplied native
  timestamps. Join compute-owned consumption timing to the existing CQ
  consumption record by stable CQE identity, charge callback work to the
  configured consumer, and project the resulting completion through
  `CompletionEvent`, `StepResult`, TTFT and TPOT. The host-CPU consumer and an
  explicit caller-timestamp bypass must preserve the accepted BACK-20 rows,
  predecessor bytes and random draws exactly. Enabled GPU consumption must
  change an end-to-end metric in the registered direction and must never
  advance CQE lifecycle state independently of the native RNIC authority.

- BACK-44 (Completeness; P1; L): let one step carry tensor-parallel
  collectives inside a node and expert-parallel collectives across nodes. The
  graph projection refuses `tp_ranks=(0, 1)` together with
  `ep_ranks=(0, 1, 2, 3)` with "graph cannot be represented by ordered GOAL
  artifacts", because the tensor-parallel collective of a layer does not
  depend on the per-rank compute of the expert-parallel ranks, so no ordered
  artifact sequence represents the graph. That is the canonical realistic
  composition, and until it plans, an intra-node collective can only be
  produced from fully intra-node MoE phases under a declared expert layout, as
  `examples/mixed_attribution_v1` had to do. Acceptance: the mixed
  configuration plans, executes and reaches per-request TTFT with the
  tensor-parallel artifacts NVLink owned and the expert-parallel artifacts
  fabric owned, while every currently accepted single-parallelism graph keeps
  its artifacts, ordering and timestamps exactly.
- BACK-39 (Completeness; P2; L): join ABI-v2 packet attempts to request
  identity only if a future study needs packet-level request attribution. The
  current request dispatch lifetime intentionally stops at collective flow
  and WQE granularity. A packet join must first carry canonical per-request
  byte extents through aggregate GOAL flow submission, define packetization
  across extent boundaries, and reconcile every packet attempt and retry with
  its operation, WQE, byte range and terminal delivery or drop. The disabled
  path keeps packet identities backend-private and must preserve every
  accepted routing-lifetime, GOAL, completion and metric byte exactly.
- BACK-46 (Completeness; P2; L): attach a separately modeled GPU to the shared
  PCIe fabric, so NIC, GPU and host are endpoints of one fabric with their own
  identities. The GPUDirect placement itself is not the gap: `GpuMemory` is a
  legal endpoint for any allocation kind including `DataRegion`, the accepted
  BACK-20 artifact already carries `data_endpoint` as `gpu_memory` under both
  the CPU-proxy and the GPU-initiated shape, the payload read really is issued
  against that allocation as a `PayloadRead` non-posted read, and every path
  configuration carries a `gpu_direct` analytical delay component. What is
  missing is the second device. The GPU-memory label is a property of an
  allocation the posting RNIC device owns, since a WQE data descriptor must
  resolve to a `DataRegion` whose `device_owner_id` equals that device's, so no
  modeled GPU owns the region, claims it on the fabric, or has its transactions
  accounted apart from the NIC's. Land that device: ownership and claim rules
  for its regions, cross-device claim rejection, and per-endpoint accounting
  that keeps the two devices' transactions distinguishable. Acceptance: a
  payload read whose completer is a region owned by the modeled GPU is charged
  on the shared fabric under that device's endpoint identity; the default
  fabric configuration (`defaultPcieFabricConfig`, host stores and host-pinned
  paths) stays the selected baseline and every accepted BACK-10, BACK-19 and
  BACK-20 artifact stays byte-identical, including the rows whose data regions
  are already labeled GPU memory; a foreign-device region claim is rejected
  transactionally with unchanged state; and the enabled two-device leg changes
  an end-to-end metric in the registered direction. Timing, occurrence and
  calibration defects of the enabled leg become BACK-16 precision scope. The
  design statement is
  [the packet-device model](../design/packet-device-model.md).
- BACK-47 (Completeness; P2; M): name the mirrored NCCL stack boundary as the
  ncclNet-shaped plugin ABI seam and register its packet-emission half.
  `simllm.compute.nccl_stack` already mirrors `ncclNet.isend` and `ncclNet.test`
  under audited names, and NVIDIA documents the contract those names sit in: a
  dynamically loaded `libnccl-net.so` exporting `isend`, `irecv` and `test`,
  with `regMr` registering buffers so RDMA NICs can prepare them, and device
  offload requested through a valid `*sendDevComm` or `*recvDevComm`. AMD's RCCL
  documents the same ABI under `librccl-net.so`, so one seam serves both stacks.
  What is missing is the declaration that this boundary is where a producer
  hands packets to a device, plus the emission contract on both sides of it:
  toward the NIC, the descriptor, doorbell and payload DMA the call causes;
  toward the GPU, the peer stores an intra-node transport issues instead.
  Acceptance: every emission at the seam carries the extent and attempt identity
  of the device port it targets, a call that would emit onto an absent or
  disabled port is rejected rather than silently dropped, and the current
  zero-time skeleton stays the exact off path with its frozen call sequences and
  event streams byte-identical. COMP-15 keeps the stack's own calibrated
  service, its receive leg and its metric projection; this task owns only the
  device-facing emission contract at the plugin boundary.
- BACK-48 (Completeness; P2; M): make the ABI v2 packet vocabulary usable by
  non-wire ports. The vocabulary is reachable only through `NetworkPort`, so a
  GPU peer port cannot emit `PacketTxStarted`, `PacketTxFinished`,
  `PacketRxArrived` or an attempt terminal in the same language, and a consumer
  would have to learn a second event grammar per port kind. Make scope, event
  kind, packet identity and terminal semantics port-kind independent, with
  capability gating deciding which kinds a port may emit: a peer port that
  cannot mark ECN or transport PFC advertises that and rejects a request for it
  explicitly, exactly as a v2 consumer paired with a v1-only producer already
  rejects before any handler installation. Acceptance: one consumer reads wire
  and peer attempts through the same vocabulary without a port-kind switch, an
  unsupported capability request is rejected before any state mutation, and both
  ABI v1 and the accepted ABI v2 wire artifacts stay byte-identical.

## Backend-repo follow-ups (tracked here, executed in their repos)

### Precision

- HTSIM-5 (Precision; P1; L): persistent DCQCN policy state across hardware
  WQEs. On
  2026-08-07 the former hardware-specific per-WQE-start scope was merged into
  the BACK-9 RDMA Work Queue; this stable ID remains open for the unfinished
  CC behavior. One QP's alpha, current/target rate, CNP suppression, byte and
  timer recovery state must survive across its WQEs and reset only with the
  modeled QP lifecycle. A new QP starts at its configured line/local-QoS rate;
  HTSIM-16 carries physical CNP/ECN observations and effective rate updates
  across the landed vocabulary to the SimLLM hardware gate. With control
  observations disabled, those capabilities remain absent rather than
  fabricating feedback. The policy never owns the hardware gate. Doorbell,
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
- HTSIM-6 (Precision; P1; L): `rnic-cn` policy lookahead (maintainer design
  2026-08-05). The
  established-pair fast path must not wait when granted bandwidth suffices,
  and the policy receives bounded lookahead from BACK-9 so it can pre-declare
  one RTT ahead for queued work toward the same destination. The WQ, WQE and
  QPC remain SimLLM hardware state; htsim retains only link-pair reservation,
  control-slot and predeclaration state. The timing-neutral SQ and directed
  link-pair identity in `d778326` remain the compatibility ledger until the
  adapter lands.
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
### Completeness

- HTSIM-1 (Completeness; P2; L): `rnic-ss` (Slingshot-like) profile wiring;
  the runtime factory
  rejects it with a clear error until the slingshot runtime lands. Its CLI
  options are already parsed so the flag ABI is stable. Out of simllm's
  scope by maintainer decision; tracked here for the backend repo only.
- HTSIM-4 (Completeness; P2; M): GOAL parser hardening and the checked-in
  `txt2bin` build target.
- ATLAHS-1 (Completeness; P2; S): correct the vendored-fallback wording (the
  vendored htsim tree
  cannot satisfy the resolver) and pin a known-good HTSIM commit. Audited on
  2026-08-13 at the pinned ATLAHS commit: the registered description is
  accurate and the defect is in the ATLAHS sources, so the fix belongs in that
  repo and this entry stays open. `scripts/build.py` resolves an HTSIM source
  directory only when a candidate has both a `CMakeLists.txt` file and a
  `datacenter` directory. The vendored tree at `sim/htsim-backend/sim` has the
  directory and no `CMakeLists.txt` at any of the three candidate spellings;
  it is upstream Broadcom csg-htsim with a Makefile build and zero `rnic`
  sources, so it could not produce `htsim_rnic` even with CMake. Two strings
  nonetheless advertise it as a working default: the `resolve_htsim_sim_dir`
  docstring in `scripts/build.py` calls the in-tree backend "the compatibility
  fallback", and the `--htsim-root` help in `atlahs_entry.py` promises "then
  the vendored compatibility tree by default". The sibling preference ahead of
  it also looks for a directory named `HTSIM`, which no case-sensitive
  checkout of this layout provides. None of this affects SimLLM runs, which
  invoke the simulators directly rather than through the launcher.
