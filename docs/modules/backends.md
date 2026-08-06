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
## Backend-repo follow-ups (tracked here, executed in their repos)

- HTSIM-1: `rnic-ss` (Slingshot-like) profile wiring; the runtime factory
  rejects it with a clear error until the slingshot runtime lands. Its CLI
  options are already parsed so the flag ABI is stable. Out of simllm's
  scope by maintainer decision; tracked here for the backend repo only.
- HTSIM-2: goodput/state/queue trace flags for `rnic-cn`; they need trace
  hooks in the reviewed runtime first.
- HTSIM-4: GOAL parser hardening and the checked-in `txt2bin` build target.
- HTSIM-5: behavioral per-WQE starting for the DCQCN comparator (maintainer
  direction 2026-08-05). The common SQ/RQ/CQ ledger and directed-pair QP
  identity landed in `d778326`; fixed WQE initiation latency, DCQCN rate state
  shared by WQEs of one pair, and behavioral SQ pipelining remain. Calibrate
  them against the message-size-vs-bandwidth
  anchors and candidate parameter sets in
  [docs/papers/msg-size-vs-bandwidth.md](../papers/msg-size-vs-bandwidth.md)
  (UCCL Fig. 14/15a, the 256 KB half-rate datum, DCQCN paper and vendor
  timer sets). Acceptance bars recorded there, micro-behavior anchors in
  examples/dcqcn_micro. Source-level findings from the micro study's
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
  HTSIM-6's remaining behavioral work: queue lookahead removes the per-WQE
  declare cost, and the event-loop scaling needs its own look.
- HTSIM-6: rnic-cn WQE-queue lookahead (maintainer design 2026-08-05): a
  timing-neutral SQ and directed link-pair identity landed in `d778326`.
  The established-pair fast path still must avoid waiting when granted
  bandwidth suffices, and the endpoint must pre-declare one RTT ahead for
  queued WQEs of the same destination to hide later declare latency. Design
  and expected effects are in the same doc; the behavior belongs in the htsim
  algorithm book alongside the bootstrap control slots.
- HTSIM-8: repair the backend `commit_check.sh` validation gate. Current
  `origin/main` has no `validate_outputs` baselines, `validate.py` divides by
  zero in every attempted case, and the script lacks fail-fast handling, so
  it reports a false success. Add checked-in baselines or remove that compare,
  fix zero-flow diagnostics, and make every failed command fail the gate.
- ATLAHS-1: correct the vendored-fallback wording (the vendored htsim tree
  cannot satisfy the resolver) and pin a known-good HTSIM commit.
