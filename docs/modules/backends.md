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
- `RnicHwProfile` + the RNIC anomaly table + `rnic_cmodel_c.h`: the
  golden-model surface of the native endpoint, specified in
  [the golden-model design](../design/rnic-cmodel.md). The profile is the
  versioned hardware-parameter object carrying link, initiation,
  outstanding-work, packet-rate, ingress, transport, congestion-control,
  flow-control and counter fields with one evidence class per field. It has a
  measured `cx5_100g` set and a `cx7_400g` set produced by `scaleProfile`,
  which scales the link, goodput, packet-rate and threshold fields, keeps the
  initiation, MTU, header, outstanding-work, transport and flow-control
  fields, and marks every scaled field `declared`. Its five work-queue service
  stages plus the wire round-trip floor sum to the lumped measured `t_eff_ps`.
  It is a separate versioned record, `simllm-rnic-hw-profile-v1`, with its own
  hash, and it is not mixed into the effective-hardware schemas or their hash
  inputs. The anomaly table is the measured performance-anomaly list carried
  as a `constexpr` array with a generated Markdown projection, so a reviewer
  can see which silicon behavior is reproduced by a named mechanism and which
  is injected by rule. The C facade is the `extern "C"` entry set over plain
  fixed-width structs and picosecond timestamps that lets an RTL testbench
  drive the same stimulus through DPI-C and compare timestamps, counters and a
  replayable transaction trace; it reproduces the C++ device's completion
  timestamps exactly, and two identical stimulus sequences trace identically.
  Its receive entry point and control-event kinds fail closed until BACK-57
  and BACK-58 land.
- `RnicTxPipeline` (BACK-56): the opt-in transmit slice, selected by
  `RnicNetworkConfig::abi_version = 2` with an enabled packetization block.
  The default stays ABI v1 with packetization off, which is the same code path
  as before the field existed, so every accepted v1 timestamp, counter and
  completion order is unchanged. Selected, it becomes the port the work queue
  binds to and the injected port becomes its downstream packet face: the queue
  still submits one flow extent per WQE, and the pipeline segments it at the
  MTU with wire header bytes and a per-QP PSN, bounds in-flight WQEs, bytes
  and packets per QP from first packet issue to last packet terminal, and
  paces issue against per-QP and per-NIC bit-rate and message-rate ceilings
  with exact rational arithmetic. It stamps the TX start at the paced issue
  instant, which is what fills `first_packet_at_ps` and `last_packet_at_ps`,
  and it emits the one extent terminal when the last packet of a WQE retires.
  Its measured behavior is in
  [the golden-model slice-B study](../../examples/rnic_cmodel_v1/RESULTS.md).
- `RnicRxPipeline` (BACK-57): the opt-in receive slice, selected by an enabled
  receive block on the same ABI v2 network configuration. The default leaves it
  off, which is the slice-B code path unchanged. Selected, it is three blocks
  in series: an ingress meter that admits wire bytes into a finite buffer
  drained at a service rate and discards the overflow at the PHY with no
  transport signal, a receive processor that applies per-QP RC and UD receive
  packet-rate ceilings and a per-NIC one, checks the RC responder's PSN and
  emits an ACK or a NAK, and delivers UD with a silent drop beyond its
  ceiling, and a requester transport that tracks PSNs and ACKs and recovers by
  go-back-N on a NAK or on the retransmission timeout. Its sweep, bands and
  fatal guards are frozen in
  [the slice-C expectations](../../examples/rnic_cmodel_rx_v1/expectations.md).
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
- `NicProfile` + `CX5_100G` + `CX7_400G` + `scale_profile` + `dcqcn_flags` +
  `dcqcn_link_bps` + `gap_fields` (BACK-54): the NIC hardware profile carrier,
  separate from transport and congestion-control policy and independent of any
  serving framework. A frozen profile holds rates, packetization, the message
  offset, queue depth, packet-rate ceilings, the responder ingress pool, PFC,
  recovery mode, timeout and ECN thresholds, and carries one evidence class per
  model field (`documented`, `driver-inferred`, `calibrated-opaque` or
  `declared`) plus a provenance string naming the campaign records.
  `CX5_100G` is the measured ConnectX-5 Ex 100 GbE profile; `CX7_400G` is
  `scale_profile(CX5_100G, 4.0)`, which multiplies the rate-carrying fields and
  the ECN byte thresholds (`SCALED_FIELDS`), carries the offset, packetization,
  depth, recovery, timeout and ingress pool across unchanged, and marks every
  field `declared` because no ConnectX-7 silicon was measured. `dcqcn_flags`
  is a pure renderer of the comparator flags a profile can fill, with the rate
  taken from `goodput_bps` and rounded to whole Gb/s by `dcqcn_link_bps`,
  because the fat-tree loader parses whole Gb/s and the runtime rejects a
  topology whose rate differs from `-link_bps`. `gap_fields` returns the
  complement, the model's gap ledger, and `GAP_TASKS` maps each gap field to
  the registry task that owns it: `link_bps` and `t_eff_ps` to BACK-54,
  `sq_depth` to HTSIM-34, `rx_ingress_meter_bytes` to HTSIM-35, and the two
  packet-rate ceilings to HTSIM-36. Switch buffers, seeds, the
  selective-repeat window and the DCQCN rate floor are fabric or policy
  parameters and stay with the study, except the switch buffers and the
  marking policy, which `FabricProfile` now owns.
- `FabricProfile` + `HACC_LEAF_4X100G` + `render_dcqcn` + `render_topology` +
  `dcqcn_port_gbps` + `fabric_gap_fields` (BACK-60): the fabric constants
  carrier, the counterpart of `NicProfile` for the switching fabric rather
  than the endpoint. A frozen profile holds the switch and host-port counts,
  the port rate, the per-pipe latency and the pipe count that set the latency
  floor, the per-port tail-drop egress buffer, the marking policy, PFC, whether
  the switch acts on a pause frame it receives, and the path count, with one
  evidence class per field (`documented`, `inferred` or `declared`) and a
  provenance string naming the campaign records. `inferred` is the class for a
  constant bracketed by measurement taken somewhere other than the device it
  describes, which is what endpoint probing of a switch produces.
  `HACC_LEAF_4X100G` is the measured HACC leaf: one non-blocking switch, four
  100 G ports, 515 ns per pipe over a four-pipe path, a 5.2 MB per-port
  tail-drop buffer, `ecn = "none"`, PFC off, pause emitted by the hosts and
  ignored by the switch, and one path. `render_dcqcn(nic, fabric)` returns the
  topology file's text and the comparator flag dict together, reusing
  `dcqcn_flags` for the NIC half and then letting the fabric override the rate,
  the buffers and the marking policy. `-link_bps` renders the fabric's port
  rate rather than the NIC's goodput asymptote, because the comparator has one
  rate and the asymptote is exactly the `link_bps` gap the NIC profile already
  registers; a NIC and a fabric that disagree about PFC are refused, since
  `-pfc` is one flag for both ends. `render_topology` expresses a single-switch
  fabric as the degenerate two-tier Clos whose whole node set hangs off one
  leaf, which is what makes every pair one hop apart and fixes the path count
  at 1. `fabric_gap_fields` is the fabric gap ledger and has exactly one
  possible member: a drop-only switch is inexpressible, because the runtime
  requires `0 <= Kmin < Kmax < egress buffer` with a nonzero Pmax, so the
  renderer parks the marking band in the last two bytes of the buffer at one
  part per million and `FABRIC_GAP_TASKS` points at HTSIM-38. The module's own
  `EVIDENCE_CLASSES` and `MODEL_FIELDS` stay module-local, because
  `nic_profile` owns those names in the package namespace.
- `HtsimUecConfig` + `build_htsim_uec_command`: argv construction for
  GOAL-driven `htsim_uec` runs.
- `LogGopsimConfig` + `build_loggopsim_command` + `run_loggopsim` +
  `parse_loggopsim_stdout` + `derive_loggp_params` (BACK-2, TRAF-20): the
  flow-level analytical seam. The same binary GOAL is costed with the LogGOPS
  model instead of a packet fabric. LogGOPS parameters keep the tool's own
  units, `L`, `o`, `g` and `O` in whole nanoseconds and `G` in nanoseconds per
  byte, under explicit `_ns` field names; parsed times convert to picoseconds
  by exactly 1000. The ideal mapping derives `G = 8e9 / rate_bits_per_second`
  and preserves its shortest binary64 decimal string. All six values carry a
  `DECLARED` evidence record. The parser reads both output shapes the tool can
  print, the per-host block and the batch-mode maximum, and treats a
  nonfinite `Average FCT` as absent. Discovery is `SIMLLM_LOGGOPSIM`, the
  `build/loggopsim` CMake layout, the ATLAHS submodule's own make output, then
  `PATH`; with none present the runner raises and names the environment
  variable.
- `LogGopsimStepSink` + `LogGopsimStepSinkConfig` (TRAF-20): the selectable
  ideal-network sibling of `HtsimStepSink`. It reuses the serial lowerer,
  graph authority, GOAL rendering and analytic local path unchanged, then
  runs each remote fabric artifact through LogGOPSim and returns the standard
  `StepResult`. Provenance records the binary SHA-256 and, for every native
  invocation, the full argv, exact `G` string, input hashes and maximum host
  finish. Its declared `S` keeps every rendered payload eager, with a
  pre-invocation rejection if that contract is violated.
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
  serially and each composes as registration plus base plus the maximum of its
  local and fabric service, so the step's realized interval is one disjoint
  subinterval per artifact and the resource whose own service equals that
  maximum owns it. `MediumAttribution` names `kernel_ps`, `nvlink_ps`,
  `fabric_ps`, `co_critical_ps`, `collective_base_ps` and
  `collective_registration_ps` separately, alongside `queue_ps` and
  `control_ps`, and totals the same picoseconds as the coarse
  `LatencyAttribution` it rolls up into. `MaskedMediumService` reports what the
  losing medium ran concurrently; it is a work sum, has no total, and never
  enters a latency partition. `attribute_step` returns the coarse partition
  alone. The evidence comes from `StepLocalityOutcome`'s per-artifact
  `local_phase_service_ps`, `base_phase_latency_ps`,
  `registration_phase_cost_ps` and `local_phase_medium`; an outcome that
  carries NVLink work without them is refused rather than approximated, and an
  all-remote outcome without them keeps its exact historical partition. A
  registration cost without the medium projection is refused for the same
  reason.
- `HtsimStepSinkConfig.collective_registration` selects the one-time
  channel-and-buffer registration model of the interim collective-completion
  contract, which the traffic module states in full. `None` is the exact off
  path: the sink's ledger charges zero, publishes no
  `StepCollectiveRegistrationOutcome`, leaves `registration_phase_cost_ps`
  empty and reproduces every accepted timestamp and artifact byte for byte.
  With a model named, the sink charges each
  `(communicator, generation, channel, buffer)` identity once at first use,
  serialized ahead of that collective's first executed artifact, and
  `rebuild_collective_communicators` models the destroy-and-init cycle that
  forces a re-registration. The charge changes no emitted byte: it is time, not
  traffic.

## Pinned submodules

| Submodule | Repo | Ref | Provides |
|---|---|---|---|
| `third_party/atlahs` | [ATLAHS-rnic-private](https://github.com/yifeng-ethz/ATLAHS-rnic-private) | `main` | GOAL toolchain (txt2bin, LogGOPSim, goal_gen), validated `htsim_rnic` launcher (`atlahs_entry.py`) |
| `third_party/htsim` | [HTSIM-rnic-private](https://github.com/yifeng-ethz/HTSIM-rnic-private) | `main` | UEC htsim, the composed SimLLM RNIC wrapper behind `HTSIM_ENABLE_SIMLLM_RNIC`, `htsim_rnic`, WQE bookkeeping, the ABI-v2 event relay with its physical control producers, the persistent flow session, and the Slingshot-class ss-dragonfly fabric wave (dragonfly geometry over ns-rosetta switches, progressive adaptive routing, the `htsim_ss_dragonfly` harness, and the `rnic-ss` endpoint hosted on the controlled Clos) |

As of 2026-08-03 the launcher, the RNIC wiring, the DCQCN comparator
(mlx5-faithful loss recovery, ECN-only and ECN plus PFC modes, storm
metrics) and the full rnic-cn algorithm-book implementation
(deterministic reservation ledger, windowed feedforward snapshots,
fractional nflow, sender egress composition, BJP-derived resequencing
window) are merged. The SimLLM pin for HTSim is on backend main at the
load-harness merge (`1dcbfec`), which carries the WQE bookkeeping
commit, the composed SimLLM RNIC wrapper, the ABI-v2 event relay, the
Slingshot-class dragonfly fabric wave (the physical ss-dragonfly fabric with
Rosetta-style switches and progressive adaptive routing, its deterministic
fixtures and sanity studies, the `htsim_ss_dragonfly` harness, and
the `rnic-ss` endpoint hosted on the controlled two-tier ns-rosetta Clos),
and the load harness that closed HTSIM-29 and HTSIM-30: paced sources at
declared sub-line-rate offered load, closed-loop sources with a declared
endpoint think-time seam for the measured Merlin per-chunk floors, explicit
per-flow cells with distinct destination ports, and a single-switch-capable
join, all default-off with the legacy invocations locked to golden bytes.
Its pre-registered discrimination experiment showed two buffer
configurations byte-identical at capture-shaped load and separated under a
saturating shared-egress cell (13 genuine-risk rows, 8 entailed rows
recorded but not counted), with the claim narrowed by review to what the
new capabilities add: expressing the capture-shaped regime at all.
The backend design note (`docs/ss-dragonfly-fabric/README.md` in the
submodule) labels the wave "hosted, calibration pending"; the rnic driver
rejects dragonfly `-topo` files at its own seam until a calibration ruling.
The wave-19 TRAF-51 comparison
([merlin_ss_fabric_calibration_v1](../../examples/merlin_ss_fabric_calibration_v1/RESULTS.md))
has since validated the declared single-switch Merlin instance's exact
serialization arithmetic and a frozen composition rule over measured
endpoint host-stack floors for the captured steady-state families; the
captured loads (each stack under a fifth of a port) do not discriminate
between fabric models, which the study states as its own limit. The
wave-21 load-bearing recalibration
([merlin_ss_fabric_loadbearing_v1](../../examples/merlin_ss_fabric_loadbearing_v1/RESULTS.md))
then made the fabric carry genuine risk through the load harness's
think-time seam: the captured x4 shared-egress aggregate is reproduced
within its frozen band with the sharing waits simulated (composed
10.63 against measured 11.10 GB/s), two buffer configurations that are
byte-identical at capture-shaped load produce opposite registered
verdicts on the composed x4 cell plus banded saturating-arm
separations, and the p50-static endpoint floor is refuted for skewed
shared-port families. None of this claims which buffer value the
Merlin switch physically has: the closed-loop abstraction carries no
loss recovery while the real transport does. Endpoint dynamics, the
tranche-2 families,
multi-switch routing and the source-shared x4 mapping remain open
(TRAF-52 and TRAF-53 in [traffic.md](traffic.md), HTSIM-31 to
HTSIM-33 below), with the backend note's own wording update as
HTSIM-31. A pin to an
append-only `<date>/simllm-addon` branch
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
leaves allocations, lifecycle evidence and generation unchanged. A WQE data
descriptor resolves to a `DataRegion` owned by the posting device unless it
names a peer owner explicitly, and a peer region is legal only when that owner
granted this device read access, so a numeric MKey reused in another device's
namespace still cannot be reached by accident.

### Second device on one fabric

`GpuDevice` is the second device kind that attaches to a `PcieFabric`. Its
composition rules mirror the RNIC's and its differences from one are deliberate.

- **Endpoint identity.** `PcieFabricConfig::host_endpoint_id` names the fabric's
  host endpoint and is owned by the fabric itself. A device claims its own
  identity for its lifetime; the identity is retired on release and can never be
  reclaimed, in any device kind, because the per-endpoint ledger is keyed by
  identity and a reclaimed row would mean two devices. A released row stays
  readable under a released label so a run record survives teardown, and
  `knownEndpoints()` enumerates attached plus released identities so the
  conservation identity stays checkable from outside.
- **Charging.** An endpoint pair names the two ends of one link traversal. A
  `HostStore` moves no link bytes and carries no pair; the two identities must
  differ; and a transfer inside any device's own memory is rejected rather than
  charged, including a granted GPU-to-GPU device-local transfer, which is the
  unmodeled peer-to-peer leg BACK-51 registers. An endpoint-attributed RNIC
  therefore cannot own device-local memory at all: that region belongs to a
  modeled GPU.
- **No MKey path for a copy engine.** A `GpuDevice` transfer deliberately
  bypasses `VirtualHostMemory::scheduleAccess`. A copy engine addresses the
  region directly, so the transfer emits no MKey, MPT or MTT read and never
  appears in an RNIC's `memoryAccesses()`. Its records are the device's own
  `transferRecords()` and the fabric's endpoint ledger. Translation stays a
  property of the RNIC's registered memory path.
- **Ordering domains.** A GPU claims one raw domain in the same flat namespace
  an RNIC claims into, where a shared RNIC derives the pair
  `(2 * namespace, 2 * namespace + 1)`. A GPU domain must avoid both halves of
  every RNIC namespace on the same fabric, and because it is one domain rather
  than a pair its reads and writes share one horizon.
- **One shared caller clock.** The fabric is a single contended resource whose
  reservation calendar only moves forward, so every device entry point that can
  schedule fabric work validates against a fabric-level caller clock and
  advances it only on success. An operation stamped before a peer device's last
  operation is rejected instead of silently absorbing that peer's backlog. Reuse
  of a released ordering domain is the one remaining inheritance of this class
  and is registered as BACK-52.
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

On 2026-08-17 the second device landed on the shared PCIe fabric. `GpuDevice`
attaches to the same `PcieFabric` an RNIC uses, claims its own endpoint
identity and ordering domain, owns data regions in a shared `VirtualHostMemory`,
grants named peer device owners read access to them, and issues its own payload
transfers; the fabric now keeps a per-endpoint requester and completer ledger
beside its per-service-class ledger, and a WQE data descriptor may name a
peer-owned region only when that peer granted the reading device. An endpoint
pair names the two ends of one link traversal, so host stores stay
unattributed by construction, the two identities must differ, and a transfer
inside any device's own memory is refused rather than charged. Its
[frozen study](../../examples/rnic_gpu_endpoint_v1/RESULTS.md) publishes 10 of
10 scored relation instances with no fatal guard violated: the payload read of a
GPU-owned region is charged to the GPU endpoint, the host-bounce arm completes
later by exactly the staged serialization in all four cells, the staged
transfers hit their closed form to the picosecond with zero credit and
link-queue wait, thirteen cross-device rejections leave fabric and registry state
unchanged, and every accepted BACK-10, BACK-19 and BACK-20 artifact reproduces
byte for byte from a rebuilt library. Every new field is inert at zero, so no
version constant moved and `defaultPcieFabricConfig()` is unchanged. Six defect
fixes to the enabled paths landed after the study's first publication, all
disclosed in its record, and the frozen rows reproduce byte for byte through
every one of them, which is the evidence that the study never exercised the
paths they close. BACK-46 stays open for its last clause, the end-to-end metric,
whose schema prerequisite is BACK-49; BACK-50 records that the effective-hardware
projection describes none of the second-device composition, BACK-51 the unmodeled
peer-to-peer leg and BACK-52 released ordering-domain reuse.

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
helper remains the generic invocation seam.

The TRAF-20 ideal-network slice passes 30 of 30 exact LogGOPS arithmetic
observables, 3 of 3 live metric-chain identities and 3 of 3 generous wall-time
ceilings in separate evidence classes. The sink's 202,000 ps network
makespan equals an independent execution of its emitted artifact, and the
remote step's TTFT exceeds the zero-collective control by exactly the same
202,000 ps
([loggopsim_ideal_v1](../../examples/loggopsim_ideal_v1/RESULTS.md)). All six
fatal guards held. The
[frontier ladder](../../examples/frontier_ladder_v1/RESULTS.md) measures the
modeled-error half against pinned packet observations: batch-32 packet over
ideal is 1.015637 for serialized traffic, 8.110405 for incast and 1.015682 for
the isolated incast control. It executes no packet reference and therefore
measures no packet wall clock. The level refuses overlapping multi-source
receiver fan-in by default because its receiver per-byte gap is unmodeled; an
explicit acknowledgment permits the run and is stamped in provenance.
The separately frozen
[acceptance study](../../examples/loggopsim_acceptance_v1/RESULTS.md) executes
both repository runners seven times on each of the same twelve GOAL binaries.
All twelve packet completions exactly reproduce the pinned reference, and all
three enforcement cells pass. The full acceptance is nevertheless REFUTED:
the packet total is 1.088866981 seconds, the ideal total is 0.029767114
seconds, and their 36.579528x ratio misses the frozen 50x floor. All four
fatal guards hold, so TRAF-20 remains open specifically on the speed
qualification. The backend evidence does not extend packet or silicon
fidelity beyond the pinned frontier record.

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
- BACK-54 (Precision; P1; M): calibrate the RoCEv2 DCQCN endpoint against the
  measured ConnectX-5 Ex 100 GbE campaign, using `simllm.backends.nic_profile`
  as the carrier. The profile's `link_bps` and `t_eff_ps` are the two fields
  the comparator's CLI cannot express: the packet path has one rate rather than
  a wire rate and a goodput rate, so the measured 100 G line rate and the
  97.1 Gb/s goodput asymptote collapse into one flag; and its message offset is
  entirely topology propagation plus store-and-forward, so a configuration can
  match either the measured 2.08 us latency floor or the measured 4.48 us
  message offset, never both. The registered evidence is in
  [examples/cx5_msgsize_v1](../../examples/cx5_msgsize_v1/expectations.md).
  Acceptance: with an endpoint initiation cost drawn from the RDMA Work Queue's
  own service stages rather than an added per-message sleep, one configuration
  reproduces the measured depth-1 WRITE curve within 15 percent at every size
  at or below 256 KiB, the 2 B latency floor within 15 percent, and the
  measured MTU-1024 tax within 2 percentage points, with the fitted `C` and
  `T_eff` reported per profile and the ConnectX-7 arm still derived by scaling
  alone. The declared ECN thresholds stay declared until an endpoint that
  exposes them is measured.
- BACK-60 (Precision; P1; M): validate the measured HACC leaf fabric end to
  end, using `simllm.backends.fabric_profile` as the carrier. The profile
  itself lands by configuration and its runnable subset is already scored in
  [examples/hacc_fabric_v1](../../examples/hacc_fabric_v1/expectations.md): the
  2.08 us latency floor, the rendered link rate, the per-port buffer identity
  read from the first go-back-N NACK, and fair sharing under fan-in. What is
  not scorable there is everything that needs a modeled endpoint, because the
  measured fabric never marks and never pauses, so every congestion signal on
  it originates at a NIC. Acceptance, all four bars frozen in that
  registration: 2 to 1 RC receiver goodput 74 to 78 Gb/s within 15 percent
  while the wire carries 99.3 Gb/s; lone-flow receiver ingress loss
  0.18 percent within 30 percent above about 94 Gb/s, in bursts of 50 to 100
  packets; DCQCN return to at least 95 percent of the pre-congestion rate in
  447 ms within 25 percent, with additive increase near 0.1 Gb/s per ms; and
  283 CNP per second per congested queue pair within 30 percent. Each bar is
  blocked on golden-model endpoint work that is not merged here: the transport
  slice for per-queue-pair sender state across messages, the receive slice for
  a responder ingress meter, the rate-control slice for the DCQCN timer
  constants, and HTSIM-38 for the notification origin. The distance between
  the study's comparator baseline and the first bar is the size of that gap.
- BACK-61 (Precision; P2; S): calibrate queue depth against delay with
  synchronised clocks. The fabric campaign estimated the per-port buffer from
  `t_drop x excess` and cross-checked it against a drain tail, but its
  third estimator, the rise in one-way delay as the queue builds, is void: the
  two-sender runs mix two unsynchronised sender monotonic clocks inside one
  bin, and a single sender never builds more than 58 KB of queue before its
  own send queue becomes the thing the delay column measures. Acceptance: with
  hardware timestamps taken on both ends of the same packet, the delay trend
  over a sweep of offered rates yields a queue-depth estimate that agrees with
  the 5.2 MB fill estimate within 25 percent, and the disagreement between the
  three estimators is reported rather than averaged away.

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
- BACK-46 (Completeness; P2; M): close the last acceptance clause of the
  second-device composition, which is the end-to-end metric. The four clauses
  this entry was registered with are quoted here so the numbering stays
  recoverable without git: (1) a payload read whose completer is a region owned
  by the modeled GPU is charged on the shared fabric under that device's
  endpoint identity; (2) the default fabric configuration stays the selected
  baseline and every accepted BACK-10, BACK-19 and BACK-20 artifact stays
  byte-identical, including the rows whose data regions are already labeled GPU
  memory; (3) a foreign-device region claim is rejected transactionally with
  unchanged state; (4) the enabled two-device leg changes an end-to-end metric
  in the registered direction.
  Clauses 1 to 3 landed: `GpuDevice` attaches to a shared `PcieFabric` with its
  own endpoint identity and ordering domain, owns its regions in a shared
  registry, grants named peers read access, and issues its own fabric transfers;
  the fabric keeps a per-endpoint ledger beside its per-service-class ledger; and
  a WQE data descriptor may name a peer-owned region only when that peer granted
  the reader. The
  [frozen study](../../examples/rnic_gpu_endpoint_v1/RESULTS.md) scored 10 of 10
  relation instances in its published form with no fatal guard violated: a
  payload read whose completer is a GPU-owned region is charged under endpoint
  identity 4002 while the host-bounce arm charges the host endpoint, the
  host-bounce arm's WQE completes later by exactly the staged serialization in
  all four cells, ten cross-device rejections leave the fabric and registry state
  unchanged, and every accepted BACK-10, BACK-19 and BACK-20 artifact reproduces
  byte for byte from a rebuilt library.
  Clause 4 is unmet: the relations above are native WQE completion times, not a
  projected TTFT or TPOT. This entry closes when that projection lands with the
  registered direction met, which needs the schema prerequisite BACK-49 owns.
  Timing, occurrence and calibration defects of the enabled leg are BACK-16
  precision scope; the unmodeled peer-to-peer leg is BACK-51 and released
  ordering-domain reuse is BACK-52. The design statement is
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
- BACK-49 (Completeness; P2; L): teach the composed-observation contract a
  DMA-mode cell family, which is the prerequisite that currently blocks any
  fabric-attached device from reaching the reported metric chain. The contract
  that feeds `CoarseDeviceRuntime` in structural mode requires
  `eligible_at_ps == doorbell_service_ps` for every WQE, where `eligible_at_ps`
  is the producer's projection of `admitted_at_ps`. That equality holds for the
  scalar-service fixture and cannot hold for a DMA-mode device, which rejects a
  nonzero scalar doorbell service and derives admission from PCIe transactions
  instead: the accepted GPU-endpoint rows admit at 80,811 ps in their smallest
  cell against a required zero. Land the second cell family, or an equivalent
  producer, so a `ComposedRnicSession` can ingest a cell whose eligibility comes
  from fabric transactions. Acceptance: a `ComposedRnicSession` ingests a
  DMA-mode cell and drives `CoarseDeviceRuntime` in structural mode to a
  `StepResult`, the scalar family stays selectable, and every accepted Tier A,
  Tier B and Tier C `rnic_live_v1` artifact stays byte-identical with the new
  family unselected. The difficulty is L rather than M because the span crosses
  the native producer, the Python observation contract, the runtime authority
  and the reducer, and it must hold four accepted artifact families still. This
  task carries the prerequisite only; BACK-46 clause 4 owns the frozen
  two-arm projection built on it and closes on the registered direction.
- BACK-50 (Completeness; P1; M): project the second-device composition into the
  effective-hardware snapshot, which today omits it entirely on an active path.
  `renderEffectiveHardwareConfigJson` builds its `dma` and `host_memory` objects
  from exact key sets that predate endpoint identities, so
  `PcieFabricConfig::host_endpoint_id`, `RnicDmaConfig::fabric_endpoint_id` and
  `RnicHostMemoryConfig::peer_read_grants` are all absent, and `GpuDeviceConfig`
  has no projection at all: its GPU number, PCIe requester identity, endpoint
  identity, ordering domain, regions and peer grants are invisible, as is the
  fact that a GPU shares the fabric. Two consequences make this P1 rather than
  disabled-feature coverage. The hash is the hardware identity of a run, and the
  landed study drives all of these fields, so two runs with different fabric
  compositions hash identically. Worse, `peer_read_grants` decides WQE legality,
  so a config that accepts a peer-region WQE and one that rejects it share one
  `hardware_config_sha256`. Extend the emitted keys and the matching reader in
  `simllm.backends`, keeping every new key shape dependent so an unattributed
  single-device configuration emits exactly today's bytes. Acceptance: two
  configs differing only in endpoint identity hash differently, two differing
  only in peer grants hash differently, a shared-fabric GPU composition is
  described in the snapshot, the reader rejects a malformed block, and every
  accepted effective-hardware digest and rejection-corpus row stays
  byte-identical for the unattributed shape.
- BACK-51 (Completeness; P2; M): model the GPU-to-GPU peer leg over the fabric.
  A `GpuDevice` transfer whose completer is another device's device-local memory
  is rejected today, because charging it on the host link would invent a
  traversal that never reaches the host: the request direction, the two ports it
  crosses and the switch path between them are all different from a host-pinned
  staging write. A granted peer region in device-local memory therefore fails
  closed with an explicit rejection rather than a wrong number. Land the leg with
  its own direction and path handling, and decide explicitly whether it belongs
  on the PCIe fabric at all or on the NVLink or xGMI peer port that COMP-34
  registers. Acceptance: a peer device-local transfer is charged with both
  endpoints named and a direction that matches the modeled route, the rejection
  stays the off path for any route not modeled, and every accepted endpoint-ledger
  row stays byte-identical.
- BACK-52 (Completeness; P2; S): retire released PCIe ordering domains the way
  endpoint identities are retired. `OrderingDomainClaims::release` erases the
  claim, but `PcieFabric` keeps that domain's posted-visibility and
  non-posted-completion cursors keyed by domain value, so a later device that
  claims the same value inherits a horizon it did not earn. The shared fabric
  caller clock bounds the exposure, since a new operation cannot be stamped
  before the last one, but a completion horizon can still sit ahead of the
  current caller time, so an inherited wait of up to that gap is reachable.
  Refuse reuse of a released domain, or retire its cursors with the claim.
  Acceptance: a reclaimed domain cannot inherit a cursor, the rejection or
  retirement is tested, and every accepted timestamp stays byte-identical.
- BACK-53 (Completeness; P1; L): reify the RNIC's inter-subsystem
  communication as a signal-slot event bus behind the common interface. The
  README states the device at the top level as three pluggable subsystems,
  the congestion-control algorithm (CCA), the PCIe engine with the DMA
  controller, and QPC management, communicating over one common interface.
  Today the landed modular entry point composes those modules, but their
  interactions ride direct calls with no reified interconnect object. Add a
  Qt-style signal-slot bus: each subsystem declares named signals and slots
  at its boundary, connections are declared at composition time, and every
  crossing emits an observability event on the existing stream. Shape the
  bus like a NoC so a later precision task can price arbitration and
  contention; this landing is contention-free by construction with zero bus
  cost, and the default composition must preserve every accepted artifact
  byte for byte. COMP-49 owns the xPU counterpart, a streaming crossbar.
- BACK-55 (Completeness; P1; S) (remaining half): complete the golden model's
  C facade on the receive side. The profile, the anomaly table with its
  generated projection, and the transmit and control halves of the facade are
  landed: the facade reproduces the C++ device timestamps exactly, two
  identical stimulus sequences trace byte-identically, the rendered anomaly
  table equals the committed projection byte for byte, and the profile record
  hashes reproducibly and separately from the effective-hardware record.
  `rnic_cm_rx_packet` now lands a data packet on the receive pipeline and an
  ACK or a NAK on the requester transport, and `rnic_cm_nic_counters` reads
  the NIC-named observable state without moving a field of the existing
  counter set. What remains is the control-event kinds, which still fail
  closed with an unsupported status because the rate control behind them is
  BACK-58. Acceptance: a control event reaches the rate-control gate under the
  same trace-determinism guard the receive entry point already runs under.
- BACK-56 (Completeness; P1; M) (remaining half): reconcile the transmit
  pipeline's depth-ratio residual and extend it past one QP. The packetizer,
  the outstanding-work window and the pacer are landed behind an opt-in whose
  off path is the unchanged v1 code, and
  [the slice-B study](../../examples/rnic_cmodel_v1/RESULTS.md) meets every
  registered band except one: the depth-1024 over depth-1 ratio at 8 KiB is
  7.62 against the measured 5.9, because a lossless pipeline saturates at the
  goodput ceiling where the silicon sat in the loss equilibrium BACK-57 owns.
  The model-internal ratio check passes at the same cell, which localizes the
  disagreement to the missing mechanism. The ingress meter has since landed
  and
  [the slice-C study](../../examples/rnic_cmodel_rx_v1/RESULTS.md) closes that
  band: the ratio is 6.23 with the meter enabled, and the slice-B rows are
  byte-identical with it disabled. What remains here is the second half only:
  give the pacer real per-NIC arbitration across several QPs, which one QP
  cannot exercise. Acceptance: a multi-QP cell shows the per-NIC ceiling
  binding while each per-QP ceiling does not.
- BACK-57 (Completeness; P1; L) (remaining half): place the responder's
  discard threshold. The ingress meter, the receive processor and the go-back-N
  requester transport are landed behind an opt-in whose off path is the
  unchanged slice-B code, and
  [the slice-C study](../../examples/rnic_cmodel_rx_v1/RESULTS.md) meets most
  of the bar with one fitted drain rate: the saturated single RC QP settles at
  79.25 Gb/s inside the measured 78 to 92 window, the gap sweep reproduces the
  measured clean and dirty pattern at 8 KiB and 64 KiB exactly with paced
  goodput inside 1 percent, a single UD QP delivers 3.070 Mpps against a
  measured 3.07 with the excess discarded silently and no transport counter
  moving, and the two sequence counters agree exactly in every cell. Two
  clauses remain. The first is the bidirectional pair: the measured 93.4
  against 91.8 split implies a discard threshold between 93.23 and 94.86 Gb/s
  of wire, which is incompatible with the 95.7 to 97.9 the equilibrium window
  requires, so the model reports 93.4 Gb/s clean where the silicon reported
  43040 discards. That needs a second limiter in the receive path, and the
  candidate is the per-QP receive packet-rate ceiling the internal arbiter of
  BACK-58 also feeds. The second is the incast, which BACK-58 owns outright:
  without a reaction point a go-back-N requester answers loss by raising its
  own offered load, so two senders sharing a saturated bottleneck collapse
  rather than settling, while the measured run was congestion-controlled
  (78058 CNPs sent, 179746 handled). Acceptance: the unidirectional cell
  reports a nonzero discard counter at 93.4 Gb/s while the duplex cell at 91.8
  stays clean, without moving the saturated equilibrium outside its window.
  HTSIM-35 and HTSIM-36 are the fabric-model counterparts of this native
  receive pipeline, expressing the same finite outstanding work, responder
  ingress meter and packets-per-second resource in htsim rather than in the
  endpoint.
- BACK-58 (Completeness; P2; L): land rate control and the internal arbiter.
  Rate control is the DCQCN notification point that emits a CNP on a
  congestion experienced mark, the reaction point whose per-QP state persists
  across WQEs, and the ECT(0) stamp the silicon applies to every RoCEv2
  transmit regardless of the requested ECN bits. The internal arbiter is one
  processing budget shared by loopback ingress and wire ingress with wire
  priority. Acceptance: the wire and loopback shares under a combined offered
  load above the internal budget are within 5 percent of the measured split;
  the marking counter stays inert while CNPs are generated, as silicon does;
  and, with fabric loss supplied by the composed port, the incast tax identity
  holds within 25 percent and the fair-share split within 2 percentage points.
  HTSIM-5 remains the owner of the policy-side DCQCN state in the backend
  repo; this task owns only the hardware notification point, reaction gate and
  stamp. The slice-C study makes the incast half of that acceptance a
  prerequisite rather than a nicety: with the reaction point absent, two
  senders into one responder collapse to a 98.8 percent tax instead of the
  measured 26.9, while the same transport with headroom recovers correctly at
  62 recovery episodes for 45 losses.
- BACK-59 (Completeness; P2; M): make the golden model usable as an RTL
  reference from a UVM testbench. The facade trace is the expected-result file
  and the DPI-C import declarations plus a stimulus reader are the missing
  half. Acceptance: a recorded trace replays through the DPI-C boundary and
  compares timestamps and counters transaction by transaction, a divergence
  localizes to the first differing line, and the comparison runs without a
  SystemVerilog simulator in the native gate by driving the same reader from
  C. The trace format stays append-only per stimulus and per observed
  transition so a longer run is a prefix-compatible extension.

## Backend-repo follow-ups (tracked here, executed in their repos)

Scope note for the ConnectX-5 calibration (BACK-54): the DCQCN comparator
treats a send whose source and destination are the same node as a fatal error
and exits, and there is no in-NIC loopback datapath anywhere in the packet
path. The measured in-NIC contention budget, where a loopback flow and a wire
flow share one internal ceiling and the wire flow wins, is therefore out of
scope for this backend and carries no task here; a study that needs it must
model the two flows as separate nodes and say so.

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
- HTSIM-34 (Precision; P1; M): finite outstanding work at the RoCE sender.
  `RoceSrc` is a rate-paced open-loop sender with no window, no send queue and
  no outstanding-bytes cap, so it approximates an infinitely deep pipeline and
  the only way a study can vary queue depth today is to issue independent
  flows, which amortize perfectly and therefore measure nothing. The observable
  that identifies the replacement is the measured send-queue-depth ratio on
  ConnectX-5: 8 KiB messages run 5.9x faster at depth 1024 than at depth 1, and
  64 KiB messages 1.57x, while the deep-pipeline arm saturates against a
  separate loss-induced ceiling rather than against line rate. Drive the cap
  from `NicProfile.sq_depth` (in-flight WQEs) or its byte equivalent, checked
  where the sender decides to emit the next packet. Acceptance: the depth-1 and
  depth-1024 pairs at 8 KiB and 64 KiB reproduce within 20 percent on the
  ratio, and an unset cap preserves every accepted result byte for byte.
- HTSIM-39 (Precision; P1; M): admission fairness in the ns-tm3 egress buffer.
  When the buffer is full the switch drops whatever arrives, and with several
  equal-rate sources whose pacing is deterministic the same source wins the
  race for every freed slot, so the loss lands entirely on the others. The
  [hacc_fabric_v1](../../examples/hacc_fabric_v1/RESULTS.md) study measured it:
  two symmetric 32 MiB senders into one port put **100 percent** of the
  retransmissions on one sender while the other finished with zero, and three
  senders produced a strict 0, 7689, 13943 ordering. The measurement this is
  compared against split the loss evenly, within 0.5 percent across eight
  concurrent streams on a real tail-drop switch. The consequence is not only
  unfair: the starved flow receives nothing after its first hole, so its
  receiver never sees an out-of-order packet, never generates a NACK, and the
  sender learns of the loss only when the queue drains, which makes any
  first-drop timing instrument unusable. Acceptance: with several equal-rate
  sources and a full buffer, the loss splits within 10 percent across sources
  and the first NACK arrives about one buffer drain after the first drop, while
  a single-source run preserves every accepted result byte for byte.
### Completeness

- HTSIM-1 (Completeness; P2; L): `rnic-ss` (Slingshot-like) profile
  exercise from simllm. At the current pin the backend factory accepts
  `rnic-ss` and hosts it on the controlled two-tier ns-rosetta Clos
  (hosted, calibration pending per the backend design note); the earlier
  out-of-scope ruling was reversed by the maintainer on 2026-08-17. What
  remains open here: no simllm study has driven `rnic-ss` through the
  supported metric chain, and its validity claim stays exactly the backend
  label until one does. The TRAF-51 calibration study exercises the
  `htsim_ss_dragonfly` fabric harness, not this endpoint.
- HTSIM-4 (Completeness; P2; M): GOAL parser hardening and the checked-in
  `txt2bin` build target.
- HTSIM-31 (Completeness; P2; S): update the backend design note's
  status section (`docs/ss-dragonfly-fabric/README.md`) from "hosted,
  calibration pending" to the calibrated-for-what wording the TRAF-51
  studies established: steady-state solo, incast and staggered-join
  behavior at the captured distinct-port mappings and loads on the
  declared single-switch Merlin instance, endpoint floor separate, plus
  the wave-21 load-bearing additions (the shared-egress x4 aggregate
  within its frozen band, composed-level buffer-configuration
  discrimination, the p50-static-floor refutation); endpoint dynamics
  and multi-switch routing uncalibrated. Backend-repo work, registered
  here; the simllm-side wording landed with the studies.
- HTSIM-32 (Completeness; P2; M): flow-identity-keyed delivery dispatch
  in the ss-dragonfly load harness, so several flows can share one
  (source, destination) host pair. `SsDragonflyLoadDispatch` routes
  deliveries by the (source, destination) pair and rejects duplicates,
  and the harness's explicit pattern enforces pairwise-distinct pairs,
  so the captured x4 family's true mapping (four same-node stacks whose
  combined traffic leaves one source port for one destination port) is
  inexpressible; the wave-21 recalibration
  ([merlin ss fabric loadbearing](../../examples/merlin_ss_fabric_loadbearing_v1/RESULTS.md))
  had to model it as four source hosts into one shared destination
  egress, a declared abstraction whose cost its freeze states. Packets
  already carry a distinct flow id, so keying the dispatch by flow id
  is the natural fix. Acceptance: an explicit cell with two or more
  flows on one (src, dst) pair delivers per-flow chunk accounting
  correctly, legacy invocations stay byte-identical, and a
  source-shared x4 mirror cell becomes runnable.
- HTSIM-33 (Completeness; P2; S): make the host injection queue depth a
  topology parameter. The ss-dragonfly fabric hardcodes each host's
  injection queue at 64 wire packets, which overflows (fatally, under
  closed-loop sources) within tens of microseconds of any burst overlap
  from co-hosted sources, so no source-side sharing study can run until
  the depth is declarable; registered by the wave-21 recalibration
  alongside HTSIM-32, which it gates.
- HTSIM-35 (Completeness; P1; L): a responder ingress meter at the DCQCN
  endpoint. The packet path drops only at switch queues; the endpoint host
  queue is egress-only and has no discard path, so a measured responder-side
  PHY discard can only be reproduced today by mis-attributing it to a switch
  buffer. This one absent object is the mechanism behind three separate
  measured behaviors: the 78 to 92 Gb/s single-flow saturated equilibrium, the
  drain-window effect where a 4 us inter-burst gap at 8 KiB removes the loss
  entirely and raises goodput 13.8 percent, and the bidirectional case being
  counter-clean at 91.8 Gb/s per direction while the unidirectional arm at
  93.4 Gb/s drops. Model a finite receive-side buffer with a service rate,
  mirroring the existing egress host queue, sized from
  `NicProfile.rx_ingress_meter_bytes` and dropping when full. The existing
  `RnicRxPort` and its ring-CAM ingress in the `rnic-cn` family are a tested
  implementation of exactly this object in another profile and should be read
  before a new one is written. Acceptance: the gap sweep at 8 and 64 KiB shows
  the counter-clean transition across the measured threshold, and the off path
  (no meter configured) keeps every accepted result byte for byte.
- HTSIM-36 (Completeness; P2; M): a packets-per-second resource at the NIC,
  separate from bits per second and applied at both transmit and receive, with
  a per-QP and a per-NIC level. Every rate limit in both backend families is in
  bits per second, and one flow is one QP with no per-NIC message-rate resource
  shared across QPs, so the entire measured small-message regime is
  inexpressible: a single UD receive QP caps at about 3.07 Mpps and silently
  discards beyond it, sixteen QPs take 9.65 Mpps clean, and 512 B RC WRITE
  reaches 16.7 Mmsg/s per sender with a 20.5 Mmsg/s aggregate under 2 to 1
  fan-in. Source the two levels from `NicProfile.pps_ceiling_per_qp` and
  `pps_ceiling_per_nic`. Acceptance: the 512 B and 4 KiB multi-QP rows are
  reproduced within 20 percent, the single-QP knee appears at the configured
  ceiling with the excess discarded and no sender-visible signal, and an unset
  ceiling preserves every accepted result byte for byte.
- HTSIM-37 (Completeness; P2; M): accept the golden model's per-packet
  attempts on the composed port. BACK-56 emits one descriptor per packet with
  an extent index and count, and expects the TX-start, TX-finish, RX-arrival
  and terminal events per packet that the endpoint's timeline and window are
  clocked by. The composed wrapper currently relays the ABI v2 vocabulary for
  flow extents; per-packet attempts have never been driven end to end through
  it. Acceptance: a composed run segments one WQE into MTU-sized attempts,
  returns the four-event lifecycle per attempt with stable tokens, terminates
  the parent extent only after the last attempt, and reproduces the frozen
  fake-network study numbers within their registered bands. The current
  flow-extent relay stays the exact off path.
- HTSIM-38 (Completeness; P2; M): an endpoint-side congestion-notification
  hook in the DCQCN runtime, and an explicit drop-only switch mode beside it.
  Every congestion notification the packet path can originate comes from a
  switch RED mark, but the measured HACC fabric marks nothing at all: 0
  CE-marked packets in 670 M, with the egress buffer full and dropping, while
  the receiving NIC's own `np_cnp_sent` rises from 38 to 2262 per second under
  fan-in. The notification point is an endpoint, not a switch. Two things are
  missing. First, a sink-side hook that can emit a CNP from its own ingress
  state, so a rate-control study can run on a fabric that never marks. Second,
  a way to say "this switch does not mark": the configuration guard requires
  `0 <= Kmin < Kmax < egress buffer` and `0 < Pmax`, so a drop-only switch can
  only be spelled by parking the threshold two bytes below the tail-drop limit
  at a probability of 1e-6, which
  [examples/hacc_fabric_v1](../../examples/hacc_fabric_v1/expectations.md)
  does and then has to check empirically. Acceptance: a fabric configured
  drop-only marks nothing by construction rather than by arithmetic, an
  endpoint-generated notification reproduces the measured 283 CNP per second
  per congested queue pair within 30 percent, and both paths off preserve
  every accepted result byte for byte.
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
