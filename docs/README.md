# SimLLM Developer Guide

The documentation has two layers. The top-level [README](../README.md)
is the beginner layer: what SimLLM is, how to run it, what it models,
and where the project stands. This file is the developer layer: how the
docs are organized, how development works, in what order fidelity is
being built, and where every module and study stands. The per-module
docs under [modules/](modules/) are the source of truth for design,
status and open tasks; nothing here overrides them.

## Documentation map

| Layer | Where | What it holds |
|---|---|---|
| Beginner map | [../README.md](../README.md) | About, quick start, demo studies, model inventory, milestones |
| Developer map | this file | Process, fidelity order, module status, study index |
| Full design | [architecture.md](architecture.md) | Components, vLLM/SGLang integration seams, manifest schemas, execution/resource boundary, GOAL trace format, coupling modes, metrics |
| Module truth | [modules/*.md](modules/) | Per-module design, current status, numbered open tasks |
| Calibration sources | [papers/](papers/) | Literature anchors and evidence plans, including [message-size parameters](papers/msg-size-vs-bandwidth.md) and the [RNIC hardware/CX-7 boundary campaign](papers/rnic-hardware-calibration.md) |
| Studies | [../examples/](../examples/) | Expectation provenance, run scripts, audited results, plots |

## Development process

Every behavioral change follows the same discipline:

1. **Commit expectations before implementation or execution.** Each new or
   extended study first lands an expectations-only commit. Its
   `expectations.md` freezes the swept parameters (at least two), the expected
   direction and shape of every effect, and exact or bounded closed forms
   where they exist. That commit contains no implementation of the behavior,
   generated results, measured values or outcome-dependent edits. The final
   pre-run expectation commit must precede both the implementation that
   satisfies it and the first run, and `RESULTS.md` cites its hash.
2. **Run once the ledger is auditable.** The run script executes the frozen
   sweep and `RESULTS.md` defends every number: it matches the registered form
   exactly, the deviation is explained, or the bug is found. Misses are kept
   and ledgered, not silently re-registered. If no qualifying earlier commit
   exists, call the assertions post-specified regression checks, never
   pre-registration. Do not rewrite or split history after observing outcomes
   to manufacture an earlier expectation commit.
3. **Keep evidence classes separate.** Report run configurations, exact-oracle
   rows, behavioral relation families and parameterized instances, structural
   invariants, and native test executables separately. Conservation identities,
   inactive fields, disabled paths and other configuration-forced or
   by-construction zero assertions remain fatal when violated, but they are
   unscored and never increase a behavioral pass denominator. Do not sum
   counts from different evidence classes into one headline total.
4. **Independent audit.** Studies and landings are reviewed by
   independent passes (math, API conformance, house rules); audit
   findings are folded before a milestone is declared done.
5. **Numbered deferrals.** Whenever a change intentionally defers work
   (a carve-out, a stubbed mode, a `NotImplementedError`), the same
   change adds a numbered task to the owning module doc using the
   module's stable prefix (CORE-, WORK-, COMP-, PLACE-, TRAF-, GOAL-,
   BACK-, VLLM-, SGL-, BRIDGE-, and HTSIM-/ATLAHS- for backend-repo
   follow-ups). IDs are never renumbered or reused; the change that
   completes a task removes its entry. Nothing is deferred silently.
6. **Backends are pinned, not edited here.** `third_party/atlahs` and
   `third_party/htsim` are submodules. Changes to them are developed in
   their own repos on dated `<YYYY_MM_DD>/simllm-addon` branches cut
   from main, never directly on the backend main, and simllm re-pins the
   reviewed branch commit. A pinned addon branch is append-only: it is
   never rebased or deleted, so pinned commits stay reachable for fresh
   clones; merges into the backend main happen separately with the
   maintainer's approval.

Gates before every push: `ruff check .` and `pytest -q` pass, and CI
stays green.

## Open task registry

Open tasks are tracked in each module's doc with stable numbered IDs
(`PREFIX-N`, e.g. `PLACE-1`, `HTSIM-2`); an ID is added in the change
that defers the work and closed by the change that completes it, never
renumbered or reused. This list is only the index; the module docs carry
the task text:

- [core](modules/core.md): CORE-*, plus BRIDGE-* inherited from the
  folded bridge module
- [workload](modules/workload.md): WORK-*
- [compute](modules/compute.md): COMP-*
- [placement](modules/placement.md): PLACE-*
- [traffic](modules/traffic.md): TRAF-*
- [goal](modules/goal.md): GOAL-*
- [backends](modules/backends.md): BACK-*, plus backend-repo follow-ups
  HTSIM-* and ATLAHS-*
- [adapters-vllm](modules/adapters-vllm.md): VLLM-*
- [adapters-sglang](modules/adapters-sglang.md): SGL-*

## Execution-fidelity order

The implementation order is architectural, not just a feature ranking:
each stage supplies the evidence needed to calibrate the next one. The
linked task IDs own the detail.

1. **Execution and completion boundary (complete).** Strict lowering,
   validation, JSON round trips, central request/object bookkeeping and
   graph-only serial replay implement `simllm-execution-graph-v1`,
   `simllm-completion-event-v1`, `simllm-execution-result-v1` and
   `simllm-request-bookkeeping-v1`. Exact lowering and WQE results:
   [examples/core2_lowering](../examples/core2_lowering/RESULTS.md).
   Actual device-schedule capture is owned by
   [VLLM-12](modules/adapters-vllm.md#open-tasks) and
   [SGL-10](modules/adapters-sglang.md#open-tasks).
2. **Hybrid measured plus SASS compute.** Capture real framework
   kernels, calibrate offline SASS replay against silicon, and populate
   provenance-carrying tables. The first slice supplies a replaceable
   intra-kernel scheduler, SM-residency, HBM and isolated-copy service model
   plus exact synthetic validation. Its A100/H100 parameters are bootstrap
   seeds, not COMP-1 closure:
   [COMP-1, COMP-5, COMP-6, COMP-10](modules/compute.md#open-tasks).
3. **Explicit KV lifecycle.** Capture the framework's KV decisions
   (allocation through eviction, swap, transfer, recompute):
   [CORE-3](modules/core.md#open-tasks),
   [VLLM-11](modules/adapters-vllm.md#open-tasks),
   [SGL-9](modules/adapters-sglang.md#open-tasks), validated in a
   dedicated `examples/kv_cache_strategies/` study before KV bytes
   couple to resource contention.
4. **Resource queues and data movement.** The first coarse
   `DeviceRuntime`: launch and stream queues, GPU/HBM, copy engines and
   DMA, NCCL channels, per-GPU WQE queues, GPU-affine RNICs, completion
   and control queues: [CORE-4](modules/core.md#open-tasks). Structural
   RNIC service then lands under BACK-8 through BACK-12.
5. **Dependency-driven overlap.** Replace the serial step chain only
   after KV and resource queues exist; framework lowering declares
   dependencies, runtime arbitration determines realized overlap:
   [TRAF-7](modules/traffic.md#open-tasks).
6. **Paced comparison and residual closure.** Compare p50 through p99.9
   TTFT/TPOT against real vLLM and SGLang in increasing-complexity
   stages; the largest attributed residual selects the next fidelity
   investment: [VLLM-4](modules/adapters-vllm.md#open-tasks),
   [SGL-4](modules/adapters-sglang.md#open-tasks).

## Module status

One line per module; the linked doc is the source of truth.

| Module | Status | Open tasks |
|---|---|---|
| [core](modules/core.md) | Implemented: virtual clock, step records, execution-graph/completion/result/bookkeeping contracts with strict JSON, serial lowerer, graph-only replay | CORE-3/4/5/6/7, BRIDGE-1 |
| [workload](modules/workload.md) | Partial: Poisson/trace arrivals, fixed/lognormal/trace lengths | WORK-1 (shared prefixes), WORK-2 (bursty/MMPP) |
| [compute](modules/compute.md) | Implemented: roofline + profile tables, kernel families, dense/MoE geometry, host initiation model, trace-driven isolated-kernel and copy service with A100/H100 bootstrap profiles | COMP-1/2/4/5/6/7/8/9/10 |
| [placement](modules/placement.md) | Implemented: placement manifest round trip, declared placements, gpu-rank mapping, vLLM extraction; fabric manifest design-only | PLACE-1/2/3 |
| [traffic](modules/traffic.md) | Implemented: collective patterns, TP step mapping, MoE all-to-all, GOAL renderers for steps and execution graphs | TRAF-2/3/4/5/6/7/8/9/10 |
| [goal](modules/goal.md) | Implemented: GOAL trace + txt2bin helper | none |
| [backends](modules/backends.md) | Implemented: htsim invocation/parsing plus native C++ RNIC SQ/CQ, network-port and shared PCIe transaction slices; htsim composition remains open | BACK-2/5-9/11-17; backend-repo HTSIM-1/2/4-9, ATLAHS-1 |
| [adapters-vllm](modules/adapters-vllm.md) | Implemented: SimExecutor on pinned v0.26.0, full RPC surface, step-record streaming, placement exporter, live tp=8 closed loop | VLLM-3 through VLLM-12 |
| [adapters-sglang](modules/adapters-sglang.md) | Implemented: SimTpModelWorker via plugin entry point at pinned commit, live CPU-engine smoke | SGL-3 through SGL-10 |

## Study index

Every study ships `expectations.md` with explicit provenance, a run script and
an audited `RESULTS.md`. New studies use the expectations-only commit rule
above; older studies without that public ancestor are labeled post-specified.
Reproduce with
`python examples/<study>/run_*.py` after the quick-start build.

| Study | What it validates | Outcome |
|---|---|---|
| [m1](../examples/m1/RESULTS.md) | The standalone spine: workload to GOAL to `htsim_rnic` to TTFT/TPOT with per-flow FCT debugging, over bandwidth/parallelism sweeps | 15/18 pre-registered checks pass; all ten fluid/nn runs exact to 0 ps; 3 misses traced to mis-registered expectations, ledgered as findings F1-F3 |
| [m4](../examples/m4/RESULTS.md) | The closed loop: live vLLM v0.26.0 tp=8 under `SimExecutor` with `htsim_rnic` inside the engine step loop | 36/36 checks pass; ring-allreduce and full-step makespans exact to 0 ps; live per-step residual 0 ps |
| [m5](../examples/m5/RESULTS.md) | MoE expert-parallel all-to-all plus the SASS calibration groundwork | All cells pass; pairwise all-to-allv closed form (fluid quantization) exact to 0 ps across size x width |
| [breakdown](../examples/breakdown/RESULTS.md) | Per-request compute/memory/network decomposition, expected vs actual, TP {2,4,8} x {100G,400G} | 21/22 pass; network share 52% (TP=2) to 89% (TP=8) at 400G, 96% at 100G |
| [cn_ladder](../examples/cn_ladder/RESULTS.md) | `rnic-cn` acceptance: incast ladder and mixed all-to-all against the ideal baselines and DCQCN | 46/49 ladder cells within the 20% target; lossy a2a16: DCQCN median 1.52 but p99 1902x vs rnic-cn median 2.06, p99 19.3x, lossless |
| [dcqcn_vs_cn](../examples/dcqcn_vs_cn/RESULTS.md) | Mechanism-level scenarios: incast above/below buffer, ECMP collisions, cross-node TP ring | 18/20 checked rows pass; DCQCN collapses 2 to 3 orders of magnitude past the buffer, wins the buffer-absorbed cell (1.07 vs 1.68) and the path-disjoint ring |
| [dcqcn_micro](../examples/dcqcn_micro/RESULTS.md) | NIC micro-behavior calibration: message-size law, incast fair share, join/exit convergence, repeated-WQE streams | Jain fairness 0.993 to 1.000; the timing-neutral WQ undershoots real-NIC anchors at 64 to 256 KB, now a BACK-9 WQ and BACK-16 PCIe-calibration target atop the landed BACK-10 fabric; persistent post-CNP rate state remains HTSIM-5; contended repeated-WQE collapse is reproduced to the derived digit |
| [core2_lowering](../examples/core2_lowering/RESULTS.md) | Execution lowering, graph-only JSON replay and WQE bookkeeping | Legacy sink, graph replay and frozen closed form agree to 0 ps on all five rows (including the MoE sentinel); flow and WQE ledgers field-identical; backend WQE layer timing-neutral (344/344 backend tests) |
| [rnic_wq_v1](../examples/rnic_wq_v1/RESULTS.md) | Native RNIC SQ/CQ structure, doorbell batching, signaling and network-credit backpressure | 11/11 post-specified cells exact; controlled SQ-full, drop and CQ-overrun boundaries pass in the native harness |
| [rnic_pcie_v1](../examples/rnic_pcie_v1/RESULTS.md) | Shared PCIe transactions, finite credits/tags/buffers and deterministic analytical path penalties | 35/35 deterministic row oracles and 10/10 behavioral relation families over 18 instances pass; structural invariants are unscored, corrected link-queue accounting leaves JCT unchanged, and posted traffic fills the frozen blocked-read gap |
| [gpu_service_model](../examples/gpu_service_model/RESULTS.md) | Isolated CTA/SM/warp scheduling, occupancy, HBM and direction-specific copy service, plus strict capture/replay artifacts | 22/22 frozen structural cells exact to zero cycles; A100/H100 timing remains an explicitly uncertain, unvalidated bootstrap |

## Milestones

- **M0 (done).** Repo scaffold, backend submodules, CI, per-module docs.
  The backend side landed ahead of schedule: `htsim_rnic` with the
  `rnic-nn`, `rnic-nn-fluid` and `rnic-cn` fidelity profiles and the
  validated ATLAHS launcher merged and pinned (2026-08-03), so
  packet-level RNIC runs work from a fresh clone.
- **M1 (done).** Standalone core, no frontend: virtual clock, length
  distributions, collective patterns, `txt2bin`, direct `htsim_rnic`
  invocation with FCT parsing and nn-normalized FCT. Validated by the
  pre-registered [m1](../examples/m1/RESULTS.md) sweeps, independently
  audited.
- **M2 (done).** vLLM adapter, pinned v0.26.0. `SimExecutor` services
  the full init and step RPC surface, streams schema-tagged step
  records, refuses what fabricated tokens would silently corrupt
  (speculative decoding, structured output); `PlacementExporter`
  extracts placement manifests from real runs. Independently audited
  (19 findings folded), validated against a live engine (2026-08-04).
- **M3 (done).** SGLang adapter, pinned to a main-branch commit.
  `SimTpModelWorker` installs through SGLang's plugin framework with no
  fork (inert unless `SIMLLM_SGLANG_ENABLE=1`), fabricates CPU-resident
  pools so RadixCache and retraction stay real; live CPU-engine smoke
  passed (2026-08-04).
- **M4 (in progress).** Closed loop validating M2/M3. Landed and
  validated: `HtsimStepSink` (per-step diagnostic mode) with a live
  vLLM tp=8 run at 0 ps residual ([m4](../examples/m4/RESULTS.md)),
  and the execution/completion boundary (stage 1 above) with
  [core2_lowering](../examples/core2_lowering/RESULTS.md). Remaining:
  the persistent co-simulator (BRIDGE-1), KV lifecycle and the resource
  runtime (CORE-3 through CORE-5), calibration against real captures.
  General fabric manifests (PLACE-1/2) stay deferred behind the fixed
  eight-GPU profile with one 400G RNIC per GPU.
- **M5 (in progress).** All-to-all traffic studies (MoE expert
  parallelism) landed ([m5](../examples/m5/RESULTS.md)). The trace-driven
  isolated-kernel and copy-service mechanisms plus A100/H100 bootstrap
  profiles are now available; production SASS calibration and populated
  profile tables remain blocked on capture hardware under COMP-5 (plan in
  [modules/compute.md](modules/compute.md)). Training workloads are pending.
  Slingshot is out of simllm scope (`rnic-ss` remains a backend-repo
  follow-up only).
- **M6 (not started).** PD-disaggregation and KV-transfer traffic
  modeling.
