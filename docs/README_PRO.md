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
| Developer map | this file | Process, fidelity order, simulated communication stack, module status, study index |
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
   PLAY-, BACK-, VLLM-, SGL-, BRIDGE-, HTSIM-, and ATLAHS- for backend-repo
   follow-ups). IDs are never renumbered or reused; the change that
   completes a task removes its entry. Nothing is deferred silently.
6. **Backends are pinned, not edited here.** `third_party/atlahs` and
   `third_party/htsim` are submodules. Changes to them are developed in
   their own repos on dated `<YYYY_MM_DD>/simllm-addon` branches cut
   from main, never directly on the backend main, and simllm re-pins the
   reviewed branch commit. A pinned addon branch is append-only: it is
   never rebased or deleted, so pinned commits stay reachable for fresh
   clones; merges into the backend main happen separately with the
   maintainer's approval. The current `third_party/htsim` pin is a
   backend-main commit; an addon-branch pin is a supported intermediate
   state, not the steady state.

Gates before every push: `ruff check .` and `pytest -q` pass, and CI
stays green.

## Local path configuration

Machine-specific paths belong only in the gitignored `.env.local.sh` on
Linux or `.env.local.ps1` on Windows. Configure the variables needed by the
selected workflow, for example:

```bash
export SIMLLM_DATA_ROOT='<configure-me>'
export SIMLLM_HTSIM_BUILD='<configure-me>'
export SIMLLM_VLLM_PYTHON='<configure-me>'
export SIMLLM_VLLM_ENV='<configure-me>'
export SIMLLM_SGLANG_ENV='<configure-me>'
export HF_HOME='<configure-me>'
export SIMLLM_VLLM_PACKAGE_ROOT='<configure-me>'
export SIMLLM_TIER_A_RUN_ROOT='<configure-me>'
export SIMLLM_TXT2BIN='<configure-me>'
```

Load the Linux file from the repository root with
`source .env.local.sh`. The equivalent PowerShell file uses aligned names:

```powershell
$env:SIMLLM_DATA_ROOT = '<configure-me>'
$env:SIMLLM_HTSIM_BUILD = '<configure-me>'
$env:SIMLLM_VLLM_PYTHON = '<configure-me>'
$env:SIMLLM_VLLM_ENV = '<configure-me>'
$env:SIMLLM_SGLANG_ENV = '<configure-me>'
$env:HF_HOME = '<configure-me>'
$env:SIMLLM_VLLM_PACKAGE_ROOT = '<configure-me>'
$env:SIMLLM_TIER_A_RUN_ROOT = '<configure-me>'
$env:SIMLLM_TXT2BIN = '<configure-me>'
```

Dot-source it from the repository root with `. .\.env.local.ps1`.
`SIMLLM_DATA_ROOT` owns external study inputs and outputs and must resolve to
an absolute directory outside the checkout;
`SIMLLM_HTSIM_BUILD` is the htsim build root; `SIMLLM_VLLM_PYTHON` selects
the vLLM interpreter; `SIMLLM_VLLM_ENV` and `SIMLLM_SGLANG_ENV` identify the
framework environments; `HF_HOME` owns the model cache;
`SIMLLM_VLLM_PACKAGE_ROOT` identifies the installed `vllm` package directory;
`SIMLLM_TIER_A_RUN_ROOT` owns Tier A artifacts; and `SIMLLM_TXT2BIN` selects the
GOAL converter. A CLI that needs an unset variable must prompt for it or fail
with an actionable message naming that variable. It must never guess a
machine-specific fallback.

Historical expectation and result records redact resolved machine-local path
spellings. A displayed environment-variable command is a portable rendering
made after the recorded run unless the record explicitly says otherwise. This
presentation change does not relocate archived artifacts, alter executable
options or parameters, or retroactively become part of a pre-registration.
When the original artifact location matters, the record keeps its immutable
identity by content hash and intentionally omits the resolved local path.

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
- [preplay](modules/preplay.md): PLAY-*
- [backends](modules/backends.md): BACK-*, plus backend-repo follow-ups
  HTSIM-* and ATLAHS-*
- [adapters-vllm](modules/adapters-vllm.md): VLLM-*
- [adapters-sglang](modules/adapters-sglang.md): SGL-*

### Progress

<!-- task-progress: generated by scripts/task_progress.py, do not hand edit -->

**61 of 172 registered tasks closed.** 111 remain open across ten modules.

`██████████░░░░░░░░░░░░░░░░░░` 35 percent

| Module | Closed | Registered | Progress |
|---|---:|---:|---|
| [core](modules/core.md) | 14 | 32 | `████████░░░░░░░░░░` |
| [workload](modules/workload.md) | 1 | 4 | `████░░░░░░░░░░░░░░` |
| [compute](modules/compute.md) | 2 | 20 | `██░░░░░░░░░░░░░░░░` |
| [placement](modules/placement.md) | 0 | 3 | `░░░░░░░░░░░░░░░░░░` |
| [traffic](modules/traffic.md) | 6 | 20 | `█████░░░░░░░░░░░░░` |
| [goal](modules/goal.md) | 1 | 1 | `██████████████████` |
| [preplay](modules/preplay.md) | 9 | 12 | `██████████████░░░░` |
| [backends](modules/backends.md) | 22 | 43 | `█████████░░░░░░░░░` |
| [adapters-vllm](modules/adapters-vllm.md) | 4 | 20 | `████░░░░░░░░░░░░░░` |
| [adapters-sglang](modules/adapters-sglang.md) | 2 | 17 | `██░░░░░░░░░░░░░░░░` |

Counted from the open-task sections of the module docs plus the closed list in [task-ledger.json](task-ledger.json); regenerate with `python3 scripts/task_progress.py`. A task is registered when a change defers work and closed when the completing change removes its entry, so the denominator grows as the build-out discovers work: a rising open count is not a regression. BACK-4 was retracted by maintainer decision rather than completed, and is counted as closed.

<!-- end task-progress -->

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
   DMA, NCCL channels, GPU-affine RNIC submission, completion-event plumbing
   and a labeled control queue: CORE-4, landed
   ([core status](modules/core.md#status),
   [examples/core4_runtime](../examples/core4_runtime/RESULTS.md)). CORE-8
   fixes one cross-language queue-visit meaning and identity arbitration
   baseline;
   CORE-9 corrects the structural WQE projection. BACK-8 and the ABI-v1 part
   of HTSIM-9 now compose native RNIC timing with htsim; CORE-4 invokes the
   frozen path from the graph and CORE-5 reduces its completion into
   `ExecutionResult`, `StepResult` and TTFT/TPOT. CORE-21 retains the
   same-contended-graph authority comparison. BACK-9, BACK-11 and BACK-12
   complete the remaining RNIC mechanisms behind that path (BACK-10 is closed,
   and BACK-18 has landed the modular entry point), and HTSIM-9 now
   closes only on one composed Tier B-class run whose ABI-v2 packet-issue
   evidence populates the native timeline through the live metric chain;
   BACK-25 and BACK-26 landed that vocabulary, leaving HTSIM-15, HTSIM-16 and
   BACK-34 as its producer-side residuals.
5. **Dependency-driven overlap.** Replace the serial step chain only
   after KV and resource queues exist; framework lowering declares
   dependencies, runtime arbitration determines realized overlap:
   [TRAF-7](modules/traffic.md#open-tasks).
6. **Paced comparison and residual closure.** Compare p50 through p99.9
   TTFT/TPOT against real vLLM and SGLang in increasing-complexity
   stages; the largest attributed residual selects the next fidelity
   investment: [VLLM-4](modules/adapters-vllm.md#open-tasks),
   [SGL-4](modules/adapters-sglang.md#open-tasks).
7. **Model-runner coupling and GPU-initiated networking.** Move the vLLM
   seam from the executor RPC surface to the model runner, first as a
   flagged skeleton and later under a real GPU worker (the SGLang adapter
   already couples at that boundary) and let the
   simulated GPU launch the NCCL work. The tracked host-memory registration of
   the QPC and the rings (BACK-19) and the three per-queue submission shapes
   with owned CQ consumers (BACK-20) have landed, leaving GPU-owned CQ
   consumption and producer calibration open. The framework communicators
   and the NCCL stack are simulated behind their real interfaces, trimmed
   to the main path with observability inserted (the stack graph below):
   [VLLM-13](modules/adapters-vllm.md#open-tasks),
   [VLLM-14](modules/adapters-vllm.md#open-tasks),
   [SGL-11](modules/adapters-sglang.md#open-tasks),
   [COMP-15](modules/compute.md#open-tasks),
   [BACK-37](modules/backends.md#open-tasks),
   [COMP-21](modules/compute.md#open-tasks),
   [COMP-11](modules/compute.md#open-tasks). This deepens the visibility
   available to the stage 6 comparisons; the executor-level mode stays the
   GPU-less path.

## Fidelity levels and switches

The order above is what gets built next. This section is the orthogonal
axis: at each seam of the stack, what precision levels exist today, what
is registered, and how a run selects one. The whole point is that a study
should be able to trade accuracy for speed deliberately and say in its
record exactly which trade it made, instead of every run silently paying
for the most detailed model available.

Two properties hold at every seam, and both are testable rather than
aspirational:

- **Switching a level changes fidelity, never semantics.** The same
  request set produces the same tokens, the same stop reasons, the same
  scheduler decisions and the same collective *shape* at every level. A
  level may change how long something takes and how much variance it has.
  It may not change what happened. Every seam therefore names one
  compatibility level whose accepted artifacts stay byte-identical.
- **A level is either deterministic or calibrated.** A deterministic
  level returns one value for one input, which is what makes byte
  identity and regression locking possible. A calibrated level returns a
  *distribution* fitted offline against captured evidence, which is how a
  study asks for tail behavior instead of a mean. A calibrated level must
  carry its provenance and its seed, so a run is reproducible even though
  the model is stochastic.

### The seams

Legend: **landed** is usable today, **registered** names the owning task.

| Seam | Level | Kind | State |
|---|---|---|---|
| Workload | fixed trace | deterministic | landed |
| | Poisson arrivals, lognormal or trace lengths | calibrated | landed |
| | bursty and MMPP arrivals | calibrated | registered, WORK-2 |
| Request outcome | fabricated token and sampled length | deterministic | landed |
| | pre-play oracle: real model fixes length, stop reason and routing | deterministic replay of a real realization | landed, PLAY-1 through PLAY-4 |
| | framework CPU oracle with observed dispatch and KV | deterministic replay | in flight, PLAY-5 |
| Framework seam | recorded step records | deterministic | landed |
| | live engine at the executor RPC surface | deterministic | landed |
| | live engine at the model-runner seam | deterministic | landed flagged, GPU-present half is VLLM-13 |
| Compute | fixed per-step constant | deterministic | landed |
| | roofline over `ModelDims` | deterministic analytic | landed |
| | profile table and trace-calibrated service model | calibrated, mean only | landed as bootstrap, COMP-1 owns real calibration |
| | per-kernel latency distribution | calibrated, distribution | registered, COMP-23 |
| Dependency | serial per-layer chain | deterministic | landed, compatibility level |
| | observed framework schedule with realized overlap | deterministic | landed, TRAF-7; live producer is TRAF-13 |
| Locality | all segments remote on the fabric | deterministic | landed, compatibility level |
| | intra-node NVLink split, flat analytic rate | deterministic analytic | landed, TRAF-10 |
| | measured NVLink bandwidth, latency and concurrency | calibrated | registered, TRAF-11 |
| Network | fluid closed form (`rnic-nn-fluid`) | deterministic analytic | landed |
| | packet-level event driven (`rnic-nn`, `rnic-cn`, DCQCN) | deterministic given seed | landed |
| | statistical flow model: tail latency, ECMP collision, link failure | calibrated, distribution | registered, TRAF-19 |
| | fluid LogGOPSim fast path | deterministic analytic | registered, TRAF-20 |
| RNIC hardware | timing-neutral bypass ledger | deterministic | landed, compatibility level |
| | composed native RNIC with PCIe, QPC, DMA and packet issue | deterministic | landed, BACK-8 and HTSIM-9 |

### Why a statistical network level is not a retreat

Bypassing the packet simulation usually means pretending the network is
an infinite pipe, which deletes exactly the effects the project exists to
study. TRAF-19 is the opposite: fit a flow-completion distribution
offline from packet-level runs over a given topology and load, then draw
from it. An ECMP hash collision or a failed link does not disappear, it
becomes a tail with a measured shape. A sweep that needs thousands of
configurations can then keep the network side effect while paying an
analytic cost, and a study that needs to explain one tail can drop to the
packet level for that configuration only.

The honest limit of that approach belongs in the record: a fitted
distribution reproduces the marginal behavior it was fitted on and does
not reproduce correlations it never saw. A calibrated level is valid for
the topology, load and collective shape of its calibration, and TRAF-19
must carry that envelope with the fit.

### The ATLAHS GOAL path

The GOAL that the ATLAHS toolchain compiles is retained as a
cross-checking path, not removed. It genuinely enforces ordering today: a
rendered step carries thousands of `requires` edges and htsim honors
them, which a controlled two-rank experiment confirms (a calc gated on a
1 MB receive finishes 22,000 ns later than the same calc left free, which
is exactly the transfer time). Keeping that path gives an independent
execution of the same schedule to compare against the runtime's own
dependency realization.

What must not persist is two authorities disagreeing inside one run.
TRAF-12 owns the reconciliation: one authority decides order for a given
run, the other is available as an explicitly selected cross-check whose
disagreements are reported rather than averaged away.

### Selecting a level

Today each seam is selected separately, through a provider object, a
profile string, the presence of a placement manifest, a build option or
an environment variable. That works but it is not one surface, and a run
record cannot yet state its full fidelity configuration in one place.
CORE-36 owns a single selection surface: one configuration that names the
level of every seam, validated up front, refused when two seams are
incompatible, and stamped into the run provenance so any result can be
read back with the precision it was produced at.

## Simulated communication stack

The coupling modes reproduce the frameworks' real communication stack with
the same functional names and interfaces, trimmed to the main path: the
implementations are reduced, side calls off the main path are omitted or
served inertly, and every boundary crossing emits an observability event on
the virtual clock. Both frameworks now have a simulated communicator layer:
vLLM's `GroupCoordinator` (landed zero-time; residual
[VLLM-14](modules/adapters-vllm.md#open-tasks)) and SGLang's vendored
`GroupCoordinator` with its device communicators (landed zero-time; residual
[SGL-11](modules/adapters-sglang.md#open-tasks)). Both feed the
interface-faithful NCCL stack model
([COMP-15](modules/compute.md#open-tasks)): intra-node collectives run as
NCCL collective kernels on the NVLink-class egress model and stay off the
fabric, while inter-node transfers follow the proxy model down to the NIC
(the landed BACK-20 submission profile owns the submission source;
[BACK-37](modules/backends.md#open-tasks) retains GPU-owned CQ consumption).

```text
vLLM model runner                    SGLang model runner
  simulated GroupCoordinator           simulated vendored GroupCoordinator
  (same interface, trimmed)            and device communicators
               \                          /
                v                        v
          NCCL model (same functional names, trimmed)
               |                            |
   intra-node collective kernel   inter-node proxy
   on the NVLink-class egress       -> ncclNet (isend, irecv, test)
   model (stays off the fabric)     -> ibverbs (post send, poll CQ)
                                    -> modular RNIC device
                                       (WQ core + QPC + DMA/PCIe
                                        + network port)
                                    -> htsim transport/CC policy
                                       and packet fabric
```

In GPU-initiated mode (the landed BACK-20 shape) the proxy leg is replaced by
GPU-written rings behind the same upper interface; the CQ consumer is declared
per queue today, and moving its polling and callback work onto the GPU compute
service remains [BACK-37](modules/backends.md#open-tasks).

The first runnable slice is deliberately a skeleton: the complete mental
model exists as name-mirrored empty function calls. A high-level entry flag
tells the adapter it is simulating, so no physical GPU worker or GPU model
runner is constructed; the adapter's copied Python path executes the same
algorithm and call order with the deliberate computation left empty, and
every timestamp is issued centrally by the simllm core virtual clock. The
real communicator call path is also a study subject in its own right: the
communication function itself (Python dispatch, custom-op indirection,
synchronization stalls) can bottleneck vLLM and SGLang, so its measured
cost is compared against the simulated path (VLLM-14, SGL-11). The
skeleton's data-dependent outcomes (token ids, stop positions, expert
routing) come predefined from the CPU pre-play oracle when a trace is joined
([preplay](modules/preplay.md): capture, arrival join, adapter replay and
routed supply are landed, and the routed replay chain is validated 13/13 in
[preplay_validation_v1](../examples/preplay_validation_v1/RESULTS.md)), so
the empty calls still drive a real simulation.

### Full call loop (default setup)

The default setup is NCCL with a CPU-host proxy, not GPU-initiated
networking. Placement is explicit: the proxy is fed by GPU-written
descriptors and tail counters in host-visible pinned memory (the BACK-20
CPU-proxy shape), while the per-channel data FIFO (the net staging buffer)
is hosted in GPU memory, registered so the NIC's payload DMA reads it
directly over PCIe (GPUDirect-style); the head counters and ready flags
are host-visible as well. The counter roles follow real NCCL's
`prims_simple` convention: the GPU producer advances the send TAIL when
it publishes work, and the proxy advances the send HEAD after network
completion, which the kernel polls to reuse slots. In the graph, "signal"
is a proactive store by the producer and "poll" is a consumer that spins
until the value changes. The lanes show how one upper-layer call from the
worker side closes its full loop.

```text
CPU host lane                      GPU device lane                RNIC and fabric lane
-------------                      ---------------                --------------------
EngineCore.step()
 Executor.execute_model()
 Worker.execute_model()
  GPUModelRunner.execute_model()
   layer forward ................>  compute kernels on the
   GroupCoordinator.all_reduce()    simulated GPU (SM scheduler,
    pynccl.all_reduce()             HBM cursor, NVLink egress)
     traffic planner:
     logical channels, chunks
     kernel launch .............>  NCCL collective kernel
                                    intra-node share: NVLink
                                    path only, no proxy
                                    inter-node share: copy chunk
                                    into the GPU-memory data
                                    FIFO, then signal: advance
                                    the tail counter (postPeer)
 proxy loop: poll tail <..........  (tail counter is host-visible)
  ncclNet.isend()
  ibverbs post send,
  ring doorbell ..................................................>  WQE fetch (DMA read),
                                                                     payload DMA read from
                                                                     the GPU-memory FIFO,
                                                                     packets to the fabric
 proxy loop: poll CQ <.............................................  CQE written at
  ncclNet.test() done                                                completion
  signal: advance the head
  counter in host-mapped
  memory .......................>  collective kernel: poll head
                                   (waitPeer), release the FIFO
                                   slot, complete the kernel
 stream event sync
 (worker polls or waits) <........  kernel completion event
 StepResult on the simllm
 virtual clock
```

The SGLang lane is identical with `TpModelWorker.forward_batch_generation`
and the vendored `GroupCoordinator` in place of the vLLM names. The GPU
lane is served by the simulated GPU model in `simllm.compute` (CTA/SM
scheduling, HBM cursor, NVLink egress; the collective kernel is an
explicitly submitted GPU task).

### Sizing plan

The build-out is sized module by module and filled step by step. The
numbers are planning estimates, not commitments; each slice lands with its
own study per the development process. Every first slice below has landed
except the receive-leg and GPU-initiated halves of the proxy/`ncclNet` row;
the sizes are kept as the original planning estimates for provenance.

| Piece | Owner | First slice | Estimated size |
|---|---|---|---|
| vLLM adapter mirrored path (entry flag, no physical worker or runner) | [VLLM-13](modules/adapters-vllm.md#open-tasks) | empty calls, centralized timestamps | ~1,000 lines |
| Simulated vLLM `GroupCoordinator` and device-communicator stubs | [VLLM-14](modules/adapters-vllm.md#open-tasks) | interface plus observability events | ~500 lines |
| SGLang communicator half (shared base with vLLM's) | [SGL-11](modules/adapters-sglang.md#open-tasks) | interface plus observability events | ~400 lines |
| NCCL model: communicator setup, logical channels, traffic planner | [COMP-15](modules/compute.md#open-tasks) | empty calls over the ring builder | ~800 lines |
| GPU buffers and signals (data FIFO, flags, head/tail counters) | [COMP-15](modules/compute.md#open-tasks) | counters as events, no data contents | ~500 lines |
| Proxy, `ncclNet`-shaped and ibverbs-shaped seams | [COMP-15](modules/compute.md#open-tasks) | isend/irecv/test plus post/poll stubs | ~700 lines |
| Observability and centralized timestamps in the core | core (CORE-4 and CORE-5 landed; residuals [CORE-17](modules/core.md#open-tasks), [CORE-21](modules/core.md#open-tasks)) | reuse the completion-event schema | ~300 lines |

### Function inventory

The living list of mirrored functions, their simulation status and the
study that validated them (linked when performed). Statuses update as
slices land.

| Function (mirrored name) | Lane | Status | Owner | Study |
|---|---|---|---|---|
| vLLM `Executor.execute_model` RPC surface | CPU | implemented, executor-level `SimExecutor` | [adapters-vllm](modules/adapters-vllm.md) | [m4](../examples/m4/RESULTS.md) |
| vLLM `Worker` init and step surface | CPU | mirrored skeleton landed behind the entry flag; GPUs-present variant open | [VLLM-13](modules/adapters-vllm.md#open-tasks) | [vllm_skeleton_v1](../examples/vllm_skeleton_v1/RESULTS.md), [vllm16 GPU-invisible smoke](../examples/vllm_skeleton_v1/vllm16_RESULTS.md), [m4](../examples/m4/RESULTS.md) |
| vLLM `GPUModelRunner.execute_model` | CPU | skeleton mirror landed, driven by a live engine | [VLLM-13](modules/adapters-vllm.md#open-tasks) | [vllm_skeleton_v1](../examples/vllm_skeleton_v1/RESULTS.md) |
| vLLM `GroupCoordinator.all_reduce` and peers | CPU | zero-time simulated coordinator landed, wired to the NCCL stack; projection and timing open | [VLLM-14](modules/adapters-vllm.md#open-tasks) | [vllm_group_coordinator_v1](../examples/vllm_group_coordinator_v1/RESULTS.md) |
| SGLang `TpModelWorker.forward_batch_generation` | CPU | implemented, `SimTpModelWorker` | [adapters-sglang](modules/adapters-sglang.md) | live CPU-engine smoke (module doc) |
| SGLang vendored `GroupCoordinator` and device communicators | CPU | zero-time simulated vendored coordinator landed on the shared VLLM-14 base; projection and timing open | [SGL-11](modules/adapters-sglang.md#open-tasks) | [sgl_communicator_v1](../examples/sgl_communicator_v1/RESULTS.md) |
| `ncclCommInitRank` and channel setup | CPU | skeleton landed (`ncclBuildRings`, `initChannel`) | [COMP-15](modules/compute.md#open-tasks) | [nccl_stack_v1](../examples/nccl_stack_v1/RESULTS.md) |
| `ncclAllReduce` and the traffic planner | CPU | ring builder first cut; planner skeleton landed (`ncclEnqueueCheck`, `scheduleCollTasksToPlan`, `calcCollChunking`) | [COMP-15](modules/compute.md#open-tasks) | [nccl_stack_v1](../examples/nccl_stack_v1/RESULTS.md), [gpu_task_mix](../examples/gpu_task_mix/RESULTS.md) |
| NCCL collective kernel (FIFO copy, counters) | GPU | egress half first cut; FIFO, tail publication and head-credit polling skeleton landed (`runRing`, `postPeer`, `waitPeer`) | [COMP-15](modules/compute.md#open-tasks) | [nccl_stack_v1](../examples/nccl_stack_v1/RESULTS.md), [gpu_task_mix](../examples/gpu_task_mix/RESULTS.md) |
| Compute kernel service (SM, HBM, NVLink) | GPU | implemented bootstrap | [compute](modules/compute.md) | [gpu_service_model](../examples/gpu_service_model/RESULTS.md) |
| Proxy progression loop | CPU | send-leg skeleton landed (`ncclProxySaveOp`, `ncclProxyProgress`, `sendProxyProgress`); receive leg open | [COMP-15](modules/compute.md#open-tasks) | [nccl_stack_v1](../examples/nccl_stack_v1/RESULTS.md) |
| `ncclNet` isend, irecv, test | CPU | isend and test skeleton landed with the verbs seam (`wrap_ibv_post_send`, `wrap_ibv_poll_cq`); irecv open | [COMP-15](modules/compute.md#open-tasks) | [nccl_stack_v1](../examples/nccl_stack_v1/RESULTS.md) |
| ibverbs post send, poll CQ | CPU | native WQ/CQ slice, live inside the composed session through the Tier B chain; RQ/SRQ, shared CQs and moderation open | [BACK-9](modules/backends.md#open-tasks), [BACK-37](modules/backends.md#open-tasks) | [rnic_wq_v1](../examples/rnic_wq_v1/RESULTS.md), [rnic_live_v1 Tier B](../examples/rnic_live_v1/RESULTS.md) |
| NIC WQE fetch, payload DMA, CQE write | NIC | composed htsim session live: Tier B drives the chain into request TTFT/TPOT; the ABI v2 packet-issue vocabulary is landed and the composed packet-issue run through the live chain remains open | [HTSIM-9](modules/backends.md#open-tasks) | [rnic_live_v1 Tier B](../examples/rnic_live_v1/RESULTS.md), [rnic_packet_v2](../examples/rnic_packet_v2/RESULTS.md) |

## Module status

One line per module; the linked doc is the source of truth.

| Module | Status | Open |
|---|---|---|
| [core](modules/core.md) | Implemented: virtual clock, step records, execution contracts, incremental append validation, the coarse DeviceRuntime, the completion reduction to StepResult and per-request TTFT/TPOT with seven-component critical-path attribution, and the structural RNIC network seam consuming composed native observations; the demonstrated CORE-15 live-seam clauses closed on Tier B, with the same-graph comparison retained as CORE-21; BRIDGE-1 closed for the pinned-binary prepared-replay scope with the online stateful session moved to BRIDGE-2, CORE-24 and HTSIM-18 | [18 open](modules/core.md#open-tasks) |
| [workload](modules/workload.md) | Partial: Poisson/trace arrivals, fixed/lognormal/trace lengths | [3 open](modules/workload.md#open-tasks) |
| [compute](modules/compute.md) | Implemented: roofline + profile tables, kernel families, dense/MoE geometry, host initiation model, trace-driven GPU service primitive with concurrent compute/memory/NCCL scheduling and A100/H100 bootstrap profiles, the audited zero-time NCCL stack skeleton with real-source-verified names, plus the optional GPU-side RNIC producer task coupling that makes submission cadence compete for SM residency and issue budget with the surrounding kernels | [18 open](modules/compute.md#open-tasks) |
| [placement](modules/placement.md) | Implemented: placement manifest round trip, declared placements, gpu-rank mapping, vLLM extraction; fabric manifest design-only | [3 open](modules/placement.md#open-tasks) |
| [traffic](modules/traffic.md) | Implemented: collective patterns, TP step mapping, MoE all-to-all with captured non-uniform routing expansion, GOAL renderers for steps and execution graphs | [14 open](modules/traffic.md#open-tasks) |
| [goal](modules/goal.md) | Implemented: GOAL trace + txt2bin helper | none |
| [preplay](modules/preplay.md) | Capture, replay and routing supply implemented: CPU oracle traces join arrivals into the bookkeeping, the vLLM adapters serve predefined tokens, and the per-token routing feeds the traffic expansion through the packed arena, with the projection retained as the validation-time form; the routed replay chain is validated end to end with scheduler-visible completions at oracle lengths and a raw-trace GOAL recomputation, while the independent-framework oracle comparison stays blocked on a CPU-operator-bearing vLLM build | [3 open](modules/preplay.md#open-tasks) |
| [backends](modules/backends.md) | Implemented: htsim invocation/parsing with per-layer, exact-sampling and GOAL-padding step-sink precision, plus native C++ RNIC SQ/CQ, network-port and shared PCIe transaction slices; the modular device entry point and versioned session run records with the policy-invariant hardware hash are landed; the virtual host-memory model with the QPC translation asymmetry and the three submission-source shapes with owned CQ consumers are landed; the composed htsim wrapper is on the backend main and the frozen Tier A gate passes; the GPU-initiated and CPU-proxy producers are coupled to the compute model with the frozen submission timeline byte-identical on defaults; the prepared worker-reuse step sink cuts recorded-replay wall time 3.4x to 5.4x with byte-identical simulated results while the per-step diagnostic sink stays the default; Tier B passed with the first TTFT and TPOT claim through the composed native RNIC chain, scoped to the frozen isolated fixture, and BACK-8 closes for its demonstrated clauses; the NetworkPort ABI v2 packet-attempt and transport-control vocabulary is landed with the htsim relay emitting genuine committed TX/RX boundaries, closing BACK-25 and BACK-26 at the vocabulary boundary with the physical control producers deferred to HTSIM-15 and HTSIM-16 | [21 open](modules/backends.md#open-tasks) |
| [adapters-vllm](modules/adapters-vllm.md) | Implemented: SimExecutor on pinned v0.26.0, full RPC surface, step-record streaming, placement exporter, live tp=8 closed loop and exact sample attribution and the simulated GroupCoordinator, plus the flagged SimWorker skeleton through the worker-cls seam with a live engine smoke; VLLM-16 closed with a genuinely GPU-invisible in-process skeleton smoke via the combined device-namespace and CPU-platform mechanism | [16 open](modules/adapters-vllm.md#open-tasks) |
| [adapters-sglang](modules/adapters-sglang.md) | Implemented: SimTpModelWorker via plugin entry point at pinned commit, live CPU-engine smoke, exact sample attribution, and the simulated vendored GroupCoordinator on the shared communicator base | [15 open](modules/adapters-sglang.md#open-tasks) |

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
| [gpu_service_model](../examples/gpu_service_model/RESULTS.md) | Isolated CTA/SM/warp scheduling, occupancy, HBM and direction-specific copy service, plus strict capture/replay artifacts | 22/22 post-specified exact-oracle rows match to zero cycles; A100/H100 timing remains an explicitly uncertain, unvalidated bootstrap |
| [gpu_task_mix](../examples/gpu_task_mix/RESULTS.md) | What limits a compute, a memory and an NCCL/NVLink kernel, and what two of them do to each other when scheduled together | 36/36 exact-oracle rows and 6/6 behavioral relation families over 17 instances pass; 21 structural invariants are unscored; the historical D2/D3 misses remain visible and their corrected shared-issue and residency families pass |
| [rnic_wq_v1](../examples/rnic_wq_v1/RESULTS.md) | Native RNIC SQ/CQ structure, doorbell batching, signaling and network-credit backpressure | 11/11 post-specified cells exact; controlled SQ-full, drop and CQ-overrun boundaries pass in the native harness |
| [rnic_pcie_v1](../examples/rnic_pcie_v1/RESULTS.md) | Shared PCIe transactions, finite credits/tags/buffers and deterministic analytical path penalties | 35/35 deterministic row oracles and 10/10 behavioral relation families over 18 instances pass; structural invariants are unscored, corrected link-queue accounting leaves JCT unchanged, and posted traffic fills the frozen blocked-read gap |
| [step_sink_precision](../examples/step_sink_precision/RESULTS.md) | Step-sink precision: per-layer provider breakdown, exact sample attribution and the explicit GOAL-rank padding knob | 4/4 unequal-layer oracle rows exact; the registered 32,768 ps fused and 32,000 ps rendered deltas hold; the padding knob reproduces the historical workaround to 0 ps; the default path is locked byte-identical; the frozen fluid-plus-topology cell is disclosed as a pre-registration defect with post-specified replacements |
| [core7_incremental](../examples/core7_incremental/RESULTS.md) | Incremental bookkeeping-ledger validation: equivalence with the reference validator and amortized append scaling | Seeded valid and invalid stream families match the reference on decisions, exception classes and final state; incremental quadrupling ratios stay near 4 within the frozen bound of 6 while the reproduced reference path grows about 16x, above the frozen quadratic bound of 8 |
| [preplay_trace_v1](../examples/preplay_trace_v1/RESULTS.md) | The pre-play oracle's first slice: live CPU capture on the pinned granite MoE model with routing, stop-reason and determinism evidence | Same-seed captures byte-identical and independently reproduced; EOS, length-cap and stop-string terminations exact; every prompt and executed decode token carries top-8 routing across 24 MoE layers with ids in range and normalized weights; strict round trips and the frozen-byte writer fixture exact |
| [nccl_stack_v1](../examples/nccl_stack_v1/RESULTS.md) | The NCCL stack skeleton: real-source-verified mirrored names, proxy-op enqueue and progression, tail/head counter causality, poll versus signal events | 5/5 scored families over 35 instances pass and the frozen call sequences reproduce exactly; 10 fatal invariant families (including FIFO depth-two reachability, foreign-observer rejection and all 88 causal poll links) hold unscored; the naming amendment is a separate review-triggered record with the original freeze untouched |
| [bridge_persistent_v1](../examples/bridge_persistent_v1/RESULTS.md) | Prepared worker reuse for the htsim step sink: recorded-replay wall time against the per-step diagnostic baseline with byte-identity as the fatal invariant | 34/34 byte-identity across StepResult, outcomes, GOAL text, binaries and CSV; wall-clock speedups 3.36x to 5.43x against the frozen floors of 1.5x and 2.0x; baseline variance of roughly 15 percent disclosed post-specified with every pairing still passing; the SIGTERM child-lifetime defect is registered as BRIDGE-3 |
| [rnic_device_v1](../examples/rnic_device_v1/RESULTS.md) | The modular RNIC composition entry point: direct-versus-composed equality and cross-module contract enforcement | 6/6 direct-versus-composed sweep cells exact plus directed PCIe-bound and inert-network equality; rnic_wq_v1 and rnic_pcie_v1 tracked rows byte-identical through the entry point; shared-fabric config equality, foreign-domain rejection and all three scalar double-charge rejections enforced; the initial study is labeled post-specified and commit-granular with the correction round separately frozen |
| [vllm_skeleton_v1](../examples/vllm_skeleton_v1/RESULTS.md) | The flagged vLLM skeleton mode: mirrored worker and runner behind the worker-cls seam with centralized virtual timestamps | 4/4 exact-oracle rows and relation instances with frozen literal call sequences; 3/3 flag-gate negative controls; one live engine smoke reaches SimWorker, the runner mirror answers, sampled tokens equal the fabricated id and two schema-tagged step records stream; the mirror tests pass against the real pinned install; the repaired VLLM-8 structured-output refusal is regression-locked |
| [step_sink_latent_knobs](../examples/step_sink_latent_knobs/RESULTS.md) | The step-sink latent knobs go live: roofline per-layer breakdown and adapter-populated exact sampling | 11 deterministic rows exact including the cumulative-truncation relation and the -32,000 ps exact-sampling TTFT delta; a real vLLM v0.26.0 skeleton run attributes chunked prefill exactly; the default GOAL SHA lock holds; the honest genuine-risk fraction is 63.6 percent after review |
| [vllm_group_coordinator_v1](../examples/vllm_group_coordinator_v1/RESULTS.md) | The simulated GroupCoordinator: mirrored signatures audited against the pinned vLLM source, zero-time events wired into the NCCL stack | 4/4 shape and 2/2 payload-scaling instances pass; a real vLLM v0.26.0 skeleton run emits the frozen DP, TP, DP, TP order with nested stack counts 32/14 and the serialized DP padding is engine-consumed; review regressions (gapless ids, zero-payload skip events, DP-consumption assertion) are labeled post-specified |
| [preplay_arrival_join_v1](../examples/preplay_arrival_join_v1/RESULTS.md) | The arrival join: trace futures pinned into the bookkeeping before the first scheduler step | Exact projection, trace authority and cardinality families pass; joins fail loudly on missing requests |
| [preplay_adapter_replay_v1](../examples/preplay_adapter_replay_v1/RESULTS.md) | Adapter replay: a real vLLM engine decides completion from oracle lengths in both baseline and replay runs | Engine-produced record streams give the frozen TTFT deltas (-130 and -230 ps) and TPOT deltas exactly; the no-replay stream is byte-locked in the test suite; the three live attempts and post-specified fixes are disclosed |
| [core4_runtime](../examples/core4_runtime/RESULTS.md) | The coarse DeviceRuntime: dependency versus overlap, eight affine RNICs, tail-attribution conservation, identity arbitration | 22/22 exact-oracle rows against frozen integer literals, 23/23 behavioral instances and 18 unscored structural cells across 18 configurations; the review round made every authority transactional and re-froze the corrected critical-path reduction before the fix |
| [rnic_session_records_v1](../examples/rnic_session_records_v1/RESULTS.md) | Session run records: the policy-invariant hardware hash, mode-exclusivity counters, structural projection and the bypass byte-identity checker | 12/12 policy-invariant hashes, 4/4 distinct and 4/4 adjacent hardware cells, a 72-field effective census, bypass artifacts equal accepted with all mutations rejected; the review round made terminal rejection transactional with a pre-fix diagnostic and 6/6 exact clock continuations |
| [rnic_live_v1 Tier A preparation](../examples/rnic_live_v1/tier_a_harness_results.md) | The composed-session acceptance harness against the deterministic fake port, with the wrapper-bypass negative control | 8 exact-oracle rows and 4/4/4 scored family instances pass; all seven fatal invariant families hold including terminal atomicity and wrapper-bypass sensitivity; the composed Tier A run has since executed and passed against the real htsim binary (next row) |
| [rnic_live_v1 Tier A](../examples/rnic_live_v1/RESULTS.md) | The frozen composition gate against the real composed binary | All scored families and fatal invariants pass with the htsim factory under two independent fresh builds; 363/363 backend tests; legacy byte-identity holds; the full m4 sink-replay suite reproduces with 0 ps residuals through the composed binary; at that gate Tier B and TTFT/TPOT were explicitly unclaimed, and Tier B has since passed (next row) |
| [rnic_live_v1 Tier B](../examples/rnic_live_v1/RESULTS.md) | Live reachability: composed native timing flows through ExecutionGraph, the DeviceRuntime, CompletionEvent and StepResult into the first TTFT and TPOT claim through the composed native RNIC chain | All 8 single-WQE and 4 FIFO rows exact with six scored families at full genuine-risk fractions; D adds exactly 1,000 ps to live TTFT and decode TPOT, doubling the rate halves the wire term, FIFO W1 wait equals L through the live chain; four bypass profiles match the protected reference, with artifact discrimination resting on completion CSV and canonical completion rows; the claim stays scoped to the frozen isolated fixture with packet-issue timestamps and the CORE-21 same-graph comparison outside it |
| [rnic_packet_v2](../examples/rnic_packet_v2/RESULTS.md) | The ABI v2 packet vocabulary: packet-attempt TX/RX issue timestamps, ECN/CNP, PFC, rate-update and link-state forms with the htsim relay emitting committed serializer boundaries | Ten packet relations pass evaluated against raw observations before any entailing oracle, after the original oracle-first arrangement was withdrawn as 0/10 genuine-risk post-specified; TX and RX doorbell additivity exact at +1,000 ps and inverse-rate spans exact; ABI v1 stays byte-identical and a v2 consumer rejects a v1-only producer; the physical control producers remain open under HTSIM-15 and HTSIM-16, BACK-34 owns the missing partial-final-packet cell at the 4,096-byte wire quantum, and six of eight frozen relation-1 boundaries are scored |
| [preplay_validation_v1](../examples/preplay_validation_v1/RESULTS.md) | Pre-registered replay validation: scheduler-visible completions land exactly at oracle lengths and all-to-all sizes match the captured routing through the routed supply chain | 13/13 scored relations exact with a preserved failed run proving early-termination and stop-reason discrimination; the post-specified raw-trace GOAL recomputation passes 10/10 bypassing the projected routing artifact; the independent-framework CPU oracle half is honestly blocked (the CUDA vLLM build lacks the CPU memory operator) and PLAY-5 stays open for exactly that half |
| [vllm16 GPU-invisible smoke](../examples/vllm_skeleton_v1/vllm16_RESULTS.md) | Genuine GPU invisibility for the in-process skeleton smoke: three frozen single mechanisms then the frozen combined attempt | The three registered single mechanisms score 0/3 as frozen (each predicted to fail one half); the fix-round-frozen combined device-namespace plus CPU-platform attempt passes 1/1 with zero NVIDIA character devices visible and the exact skeleton smoke completing, closing VLLM-16 |
| [routed_supply_v1](../examples/routed_supply_v1/RESULTS.md) | Captured MoE routing drives the all-to-all: graph contract, pre-play projection and traffic expansion in one chain | Per-pair tables and GOAL phases exact for both placement epochs; real granite routing moves fluid JCT by the frozen closed forms (about -59 percent at 200G and -48 percent at 400G versus uniform); the uniform path stays byte-locked when no capture is joined |
| [core5_reduction](../examples/core5_reduction/RESULTS.md) | Completion feedback and tail attribution: events reduce to StepResult, the clock advances, and seven attributed components conserve end-to-end latency exactly | Exact JCTs and the conservation identity hold across dependency shape and rate sweeps with measured-versus-frozen comparisons; the Tier B run expectations, including the producer invocation contract and machine-checkable observation schema, were frozen for the composed path and the registered Tier B run subsequently passed all six scored families |
| [sgl_communicator_v1](../examples/sgl_communicator_v1/RESULTS.md) | The SGLang simulated communicator on the shared base, audited against the pinned vendored sources | Shape and payload families pass; a live CPU-engine run shows the frozen event counts with flag-off byte identity locked in the test suite; the vendored three-argument all_gather surface is exercised through a real SGLang-shaped call |
| [rnic_hostmem_v1](../examples/rnic_hostmem_v1/RESULTS.md) | The virtual host-memory model: QPC/ICM, rings, doorbell records and data regions as tracked allocations with the translation asymmetry | Every QPC fetch rides the QPC/ICM class with zero MKey/MPT/MTT events while data buffers take the full translation path; artifact identities exact; ownership claims reject duplicates and foreign teardown after review |
| [rnic_submission_v1](../examples/rnic_submission_v1/RESULTS.md) | Three submission-source shapes with per-queue CQ consumers and initiator identity separate from the QP number | Translation and identity families exact across all shapes; the default host-CPU shape preserves every accepted baseline byte for byte; GPU-memory rings admit only under the GPU-initiated shape; the producer-kind column correction is disclosed post-specified |
| [rnic_gpu_producer_v1](../examples/rnic_gpu_producer_v1/RESULTS.md) | The GPU-side producer coupling: WQE writing and UAR ringing run as explicit GpuTasks in the concurrent compute service for the GPU-initiated and CPU-proxy shapes | Half-occupancy and saturated cells shift submission cadence by the frozen closed forms (+20/+23 cycles saturated) while an idle GPU and the uncoupled default keep the frozen submission timeline and all accepted artifacts byte-identical; the artifact-identity family is reclassified fatal-unscored post-specified |

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
  the composed native RNIC and htsim session (Tier A and Tier B PASSED
  against the composed binary with the first TTFT/TPOT claim through the
  composed native RNIC chain; BACK-8 and the demonstrated CORE-15 clauses
  closed, CORE-21 retains the same-graph comparison and BACK-31 the
  unlinked-native executable negative control, and HTSIM-9 stays open for one
  composed Tier B-class run carrying ABI-v2 packet-issue evidence through the
  live metric chain now that BACK-25 and BACK-26 have landed that vocabulary;
  the frozen expectations live in
  [examples/rnic_live_v1](../examples/rnic_live_v1/expectations.md)),
  the online stateful co-simulator session (BRIDGE-2 and HTSIM-18, with
  BRIDGE-1's prepared worker reuse landed for recorded replays), KV
  lifecycle (CORE-3; the coarse resource runtime CORE-4 and the completion
  reduction CORE-5 have landed), and
  calibration against real captures.
  General fabric manifests (PLACE-1/2) stay deferred behind the fixed
  eight-GPU profile with one 400G RNIC per GPU.
- **M5 (in progress).** All-to-all traffic studies (MoE expert
  parallelism) landed ([m5](../examples/m5/RESULTS.md)). The trace-driven
  isolated-kernel and copy-service mechanisms plus A100/H100 bootstrap
  profiles are now available; production SASS calibration and populated
  profile tables remain blocked on capture hardware under COMP-5 (plan in
  [modules/compute.md](modules/compute.md)). A second offline calibration
  axis joins here: the CPU pre-play oracle
  ([modules/preplay.md](modules/preplay.md)) runs
  the real model slowly on CPU to fix each request's true output length,
  stop reason and expert routing, then replays them against the workload
  arrival model with every request's outcome pinned in the bookkeeping.
  The capture half is live
  ([preplay_trace_v1](../examples/preplay_trace_v1/RESULTS.md)). PLAY-2,
  PLAY-3 and PLAY-4 closed with the arrival join, vLLM adapter replay and
  routed supply, and the routed replay chain passed 13/13 in
  [preplay_validation_v1](../examples/preplay_validation_v1/RESULTS.md).
  Remaining: PLAY-5's independent-framework CPU oracle comparison (blocked on
  a vLLM build exporting `torch.ops._C.init_cpu_memory_env`), the optional
  framework CPU runner (PLAY-6) and SGLang replay (PLAY-7).
  Training workloads are pending.
  Slingshot is out of simllm scope (`rnic-ss` remains a backend-repo
  follow-up only).
- **M6 (not started).** PD-disaggregation and KV-transfer traffic
  modeling.
- **M7 (in progress).** Deep coupling: the full modular RNIC device and
  GPU-initiated networking. Landed: the composition entry point populated
  with the DMA, QPC and network modules (BACK-18); the tracked virtual
  host-memory registration, where the WQE rings and data buffers are reached
  through tracked pages and the QPC itself is not (BACK-19); the submission
  source made explicit per queue as a host CPU driver, a CPU proxy fed from
  GPU descriptor queues, or GPU-initiated rings with a GPU-owned CQ, each
  with an owned CQ consumer (BACK-20); and the GPU-initiated and CPU-proxy
  producer coupling to the compute model (BACK-27). On the framework side the
  flagged vLLM skeleton behind the worker-cls seam, the zero-time vLLM and
  SGLang communicators and the NCCL stack skeleton have all landed with
  studies. Remaining: QP lifecycle and pairing (BACK-11), the TX/RX hardware
  pipelines (BACK-12), the GPU-owned CQ consumer and its runner callback
  (BACK-37), producer calibration (COMP-21), the GPU-present runner mode
  (VLLM-13), and metric-live projection and timing for the landed
  communicator and NCCL slices (VLLM-14, VLLM-19, VLLM-20, VLLM-21, SGL-11,
  SGL-13, SGL-14, SGL-15, COMP-15; the simulated communication stack section
  above), with completion returning to the model runner through the declared
  CQ consumer.
