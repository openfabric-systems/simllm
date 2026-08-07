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
  (`simllm.traffic.render_step_goal`: per layer one
  `calc(per_layer_compute_ns)` then the two ring allreduces, plus the MoE
  dispatch/combine all-to-alls when the config declares `ep_ranks` and
  the dims declare experts, landed with the M5 slice), converts
  with `txt2bin`, runs `htsim_rnic` on the configured profile/topology,
  parses the completion CSV and returns the simulated makespan as the
  step latency with `completed_at_ps = record.virtual_time_ps + makespan`.
  A step with no TP collectives (TP world of 1, or a zero-token drain
  record) returns `None`, so the adapter's own compute-only estimate
  stands. Per-step subprocess invocation is the documented diagnostic
  mode; the persistent co-simulator is BRIDGE-1 (core.md).
  `StepNetworkOutcome` keeps per-step bookkeeping (compute estimate,
  per-layer calc, makespan, network share) for reporting.
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
intentional supported state:

```
cmake -S third_party/htsim/htsim/sim -B build/htsim -DCMAKE_BUILD_TYPE=Release
cmake --build build/htsim --parallel
build/htsim/datacenter/htsim_rnic -goal trace.bin -linkspeed_bps 400000000000 -rnic_profile rnic-cn
```

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

The SimLLM sources build a C++ library linked into the directly invoked htsim
binary; there is no Python callback in the packet event loop. The composed
runtime still presents `AtlahsFlowRuntime` to `AtlahsHtsimApi`. HTSIM-9 owns
the backend extension that lets a SimLLM hardware runtime call an htsim
policy and fabric through opaque flow and packet tokens. QP, WQE, CQ, QPC,
PCIe and DMA objects never cross that boundary.

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

The current `AtlahsWqeLedger` remains a compatibility accounting view until
BACK-9 replaces its timing-neutral transitions with the structural RDMA Work
Queue. A WQE will no longer have one ambiguous start time. The model records
post, doorbell publication and observation, WQE fetch or BlueFlame transfer,
QPC readiness, scheduler admission, first and last packet, transport
retirement, CQE visibility and CQ polling separately.
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
The pre-registered native study passes all 11 cells exactly; see
[examples/rnic_wq_v1/RESULTS.md](../../examples/rnic_wq_v1/RESULTS.md).

BACK-4 was retracted on 2026-08-03. Multi-QP striping as a DCQCN mitigation
was withdrawn by maintainer decision: DCQCN is the expected-fail comparator,
and its ECMP-collision and slow-start behavior is the phenomenon under study.

## Open tasks

- BACK-2: LogGOPSim invocation helper for fast flow-level sweeps.
- BACK-5: `HtsimStepSink` splits the whole-step compute estimate evenly
  across layers (`estimate_step_latency_ps(...) // (L * 1000)`, which
  also truncates to whole GOAL ns units). Real per-layer durations differ
  (LM head and sampling live in the last layer's share); a per-layer
  provider breakdown would replace the even split.
- BACK-6: `HtsimStepSink` approximates `num_sampled` as the number of
  scheduled requests; a mid-prompt chunked-prefill request does not
  actually sample. The inflated LM-head term is small against the step
  total; exact sampling attribution needs prompt-completion knowledge in
  the record.
- BACK-7: `HtsimStepSinkConfig` has no explicit GOAL-rank padding knob.
  `rnic-cn` enforces that the resolved GOAL layout matches the topology's
  node count, so a topology run today must place its TP group on the
  highest-numbered node's GPUs to pad the GOAL implicitly (see
  examples/breakdown/RESULTS.md method notes); the sink should pass
  `num_goal_ranks` through to `render_step_goal` when a topology is set.
- BACK-8: create the protocol-neutral SimLLM RNIC hardware extension under
  `simllm/backends/rnic/`. Its C++ event core must be independent of Python
  and of any one CC policy, compose with htsim through HTSIM-9, and preserve
  direct binary invocation. Define versioned configuration and result
  records, deterministic event ordering, opaque policy/fabric tokens and a
  hardware-bypass mode. Acceptance requires the same hardware configuration
  hash across `rnic-nn`, `rnic-cn` and DCQCN comparison rows, plus exact
  preservation of the current null/fluid closed forms in bypass mode.
  The standalone C++17 library, opaque flow-level `NetworkPort`, strict native
  build and deterministic fake adapter are complete. Remaining scope is the
  outer `AtlahsFlowRuntime` wrapper, live htsim composition, run records,
  configuration hash and bypass equivalence.
- BACK-9: replace the timing-neutral WQE ledger with the structural **RDMA
  Work Queue**, merging the old WQE lifecycle and per-WQE-start work. Model
  verbs WR chains, WQE construction, SQ/RQ/SRQ rings, many-WQ CQ sharing,
  doorbell batches, WQEBB and WR indices, fences, inline data, signaled and
  unsignaled sends, receive consumption, finite depth, wrap and reclamation.
  CQ is a real host-memory queue with requester/responder/error CQEs, owner
  phase, producer/consumer indices, 64/128-byte format profiles, compression,
  moderation, poll and completion-channel paths, interrupts and overrun.
  Normalized CQE content includes WR ID, QPN/source QP, opcode, status,
  opcode-valid byte count, immediate/invalidate data, flags, syndrome and
  vendor syndrome; provider-derived fields and valid bits stay explicit.
  Record `posted_at`, `doorbelled_at`, `doorbell_seen_at`, WQE-fetch begin/end,
  `qpc_ready_at`, `admitted_at`, first/last packet, transport retirement,
  CQE visibility and poll time. Define NIC start as first-packet issue, never
  as `ibv_post_send` return.
  The first one-SQ/one-CQ send slice is complete, including prefix acceptance,
  finite depth, batching, ordered retirement, signaling, poll-time reclaim,
  CQ wrap/owner generation and controlled first-failure evidence. Remaining
  scope includes RQ/SRQ, multiple WQs and shared CQs, WQEBB encoding, fences,
  inline/BlueFlame paths, CQE format profiles, compression, moderation,
  completion channels and interrupts.
- BACK-10: implement the shared PCIe/MMIO/DMA queueing model. Keep distinct
  service classes for UAR doorbells, BlueFlame write-combining copies, DB
  records, WQE reads, QPC/ICM and MTT/MPT fetches, payload reads/writes, CQE
  writes, command queues, interrupts and ODP/IOMMU faults. Parameters include
  PCIe generation/width, TLP overhead, MPS/MRRS, posted-write and read
  completion latency distributions, outstanding-read limits, credits,
  completion buffering, relaxed ordering, ACS routing, IOMMU, DDIO, NUMA and
  GPU Direct topology. Every byte and wait must be attributed to one class.
- BACK-11: implement QP lifecycle, RNIC pairing and context placement. Cover
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
  separately.
- BACK-12: implement the TX/RX hardware pipelines and cross-layer fault
  boundary. Include WQE decode, context/translation lookup, opcode-specific
  DMA, packetization, per-QP eligibility, arbitration, rate and PFC gates, MAC
  queues, RX matching/reassembly, SEND-to-RQ consumption, one-sided access,
  ACK/NAK/RNR, retry/timeout, error transition, CQE DMA and interrupt/poll
  delivery. Add deterministic, Bernoulli and burst injection at named TX,
  wire/switch and RX boundaries; every loss reports location, reason and
  controlled/asserted/inferred evidence. RNIC PFC covers per-priority
  headroom, XOFF/XON hysteresis, pause quanta/refresh, paused-egress gating and
  insufficient-headroom drops; HTSIM-9 transports the frames through the
  fabric. The DCQCN policy adapter is delivered and calibrated before wider
  PFC and programmable-CC work.
- BACK-13: build a versioned CX-7 observable-state model and capture schema.
  Inventory only public Linux mlx5, rdma-core, NVIDIA MFT/DOCA and device-
  reported fields. Tag each as `documented`, `driver-inferred` or
  `calibrated-opaque`, with PSID, firmware, kernel, rdma-core, MFT, PCIe and
  topology provenance. Capture supported named registers, resource dumps,
  queue/counter snapshots, `ethtool -S`, RDMA hardware counters,
  `rdma resource`/`rdma statistic`, devlink health, DCB/PFC state,
  PCIe/AER/telemetry and
  tracepoints. Do not invent physical addresses, internal cache geometry,
  scheduler registers or firmware-private behavior.
- BACK-14: add an ibverbs capture/replay bridge for controlled calibration.
  Capture control verbs at QP/CQ/MR creation and modification, then capture
  data-path WR chains and CQ polls at the rdma-core mlx5 provider boundary,
  because the fast path bypasses the kernel and generic wrappers can be
  inlined or bypassed. Normalize both live capture and SimLLM lowering into
  the BACK-9 WR/WQE schema. An optional preload wrapper is a convenience path,
  not the signoff oracle. Preserve WR chains, SGEs, flags, queue identities,
  QP state and timestamps without recording payload contents by default.
- BACK-15: run the pre-registered RNIC calibration and boundary campaign.
  Start with DCQCN, then WQ/CQ and PCIe, QPC/cache, port loss and PFC. Sweep at
  least two dimensions per claim: WQ depth/batch/SGE/payload/signaling; QP and
  MR working sets; page size and context locality; PCIe width/NUMA/ordering;
  CQ depth/poll cadence; MTU/direction/loopback; loss location/rate/burst;
  DCQCN timers/rates/ECN; and PFC headroom/incast/RTT. Use Collie cases as
  reproducer seeds, not CX-7 truth, since its Mellanox results are CX-6 and
  omit packet-loss, control-path and NDA diagnostic-counter details. Match
  transaction identity through the first loss or queue knee, classify every
  drop by evidence tier, and defend WQE latency, FCT/JCT, useful/raw bytes,
  queue depth, cache miss, retry, CQE, CNP and pause metrics.

## Backend-repo follow-ups (tracked here, executed in their repos)

- HTSIM-1: `rnic-ss` (Slingshot-like) profile wiring; the runtime factory
  rejects it with a clear error until the slingshot runtime lands. Its CLI
  options are already parsed so the flag ABI is stable. Out of simllm's
  scope by maintainer decision; tracked here for the backend repo only.
- HTSIM-2: goodput/state/queue trace flags for `rnic-cn`; they need trace
  hooks in the reviewed runtime first.
- HTSIM-4: GOAL parser hardening and the checked-in `txt2bin` build target.
- HTSIM-5: persistent DCQCN policy state across hardware WQEs. On
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
  traces. The UCCL no-loss curve and 256 KB half-rate datum now calibrate
  BACK-9/BACK-10 hardware and must not be fitted again in the policy. The
  existing micro-behavior anchors are in examples/dcqcn_micro. Source-level
  findings from the micro study's
  review, now the concrete work items: every send op constructs a fresh
  DCQCN source at line rate with no cross-WQE rate state
  (dcqcn_atlahs_runtime.cpp:398), the additive/hyper increase is
  R_AI = C/20 and C/10 (dcqcn.cpp:48-49) against the paper's fixed
  40 Mbps, and the ECN defaults are fixed bytes (Kmin 64 KB, Kmax
  640 KB, Pmax 0.25) independent of the link rate.
- HTSIM-7: rnic-cn concurrent same-pair flow scaling. 10,000
  simultaneous flows between one source-destination pair make no visible
  progress within a 600 s wall-time budget (progress 0 percent, request
  queue 10,000; examples/dcqcn_micro addendum 1), far beyond the
  algorithm book's S_max regime but reachable by WQE-flood workloads;
  the measured per-flow control cost also scales with flow count, not
  bytes (16 KiB flood streams cap at 0.36 to 0.46 C). Both are adjacent to
  HTSIM-6 and BACK-9: policy lookahead removes the repeated declare cost,
  structural WQ backpressure limits how much work can be exposed, and the
  event-loop scaling needs its own look.
- HTSIM-6: `rnic-cn` policy lookahead (maintainer design 2026-08-05). The
  established-pair fast path must not wait when granted bandwidth suffices,
  and the policy receives bounded lookahead from BACK-9 so it can pre-declare
  one RTT ahead for queued work toward the same destination. The WQ, WQE and
  QPC remain SimLLM hardware state; htsim retains only link-pair reservation,
  control-slot and predeclaration state. The timing-neutral SQ and directed
  link-pair identity in `d778326` remain the compatibility ledger until the
  adapter lands.
- HTSIM-8: repair the backend `commit_check.sh` validation gate. Current
  `origin/main` has no `validate_outputs` baselines, `validate.py` divides by
  zero in every attempted case, and the script lacks fail-fast handling, so
  it reports a false success. Add checked-in baselines or remove that compare,
  fix zero-flow diagnostics, and make every failed command fail the gate.
- HTSIM-9: add the htsim side of the SimLLM RNIC extension. The combined
  session still implements `AtlahsFlowRuntime`, while the inner versioned port
  carries only opaque flow/packet tokens, transmit descriptors, delivery,
  drop/ECN, receive, pause and link-state events. Hardware submits an opaque
  CC-context token plus packet metadata; the policy returns eligibility/rate
  updates, and htsim returns delivery or feedback to the hardware. It must use
  the same SimLLM hardware implementation for `rnic-nn`, `rnic-cn` and DCQCN,
  transport PFC frames through htsim queues, and keep the fluid bypass
  explicit. No WQ, CQ, QP, QPC, PCIe, DMA or hardware scheduling state may
  live in this adapter.
  Develop it only in the HTSIM repo's dated append-only addon branch, then
  update the SimLLM submodule pin.
- ATLAHS-1: correct the vendored-fallback wording (the vendored htsim tree
  cannot satisfy the resolver) and pin a known-good HTSIM commit.
