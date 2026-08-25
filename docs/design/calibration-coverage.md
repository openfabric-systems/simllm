# Calibration coverage matrix

SimLLM's device models are calibrated from direct measurement on real
silicon first. This document is the living coverage record of that
campaign: one matrix per serving framework, whose rows are GPU targets
and whose columns are models, where a cell is filled when that model's
kernel workload under that framework has been captured and measured on
that target through the qualified chain. Direct measurement on the
reachable cluster GPUs is the primary evidence for every cell. The
Accel-Sim sidecar serves only an explicitly missing exact A100 point
inside its qualified envelope, per the source precedence frozen in
[offline device calibration](offline-device-calibration.md): a kernel
that can be measured is measured, never simulated. Targets without
reachable silicon carry candidate-only architecture-derived entries
(COMP-52) and never validated status.

This file records status and links, never measured numbers; the numbers
live in study `RESULTS.md` files and device-release evidence ledgers.
The module registries under [modules](../modules/) own every task named
here.

## What fills a cell

A cell is one `(target, framework, model)` triple. It is filled by this
chain, each link owned by a registered task:

1. **Extraction** ([COMP-54](../modules/compute.md#open-tasks)): the
   model's kernel inventory is extracted from the framework offline,
   producing the content-addressed execution-graph template, kernel
   families, typed invocation shapes and per-phase launch counts that
   define the cell's denominators. No GPU is required for this link.
2. **Capture** ([VLLM-12](../modules/adapters-vllm.md#open-tasks) or
   [SGL-10](../modules/adapters-sglang.md#open-tasks), joined by
   [COMP-6](../modules/compute.md#open-tasks)): the physical device
   schedule with observed implementation identities is captured on the
   target.
3. **Qualification and measurement**
   ([COMP-5](../modules/compute.md#open-tasks)): the environment
   qualifies and the campaign measures under the frozen protocol.
4. **Acceptance and release**
   ([COMP-1](../modules/compute.md#open-tasks), COMP-50): the evidence
   compiles into a `simllm-device-model-v1` release whose ledger cites
   the study.

Cell states:

| State | Meaning |
|---|---|
| measured | The full chain landed and a device release cites the study |
| partial | Some strata, phases or launch modes are measured; the cell names the gap |
| planned | Inside the fill order below; no capture yet |
| gap-fill | An explicitly missing exact A100 point served by the qualified Accel-Sim sidecar between real anchors; legal on A100 rows only |
| derived | Candidate-only architecture-derived entry (COMP-52); never validated |
| blocked | A named freeze or environment gate must land first |
| anchor | Method evidence from a non-qualifying device; transfers pipeline and seams, never numbers |

## Targets (rows)

| Target | Where | Row status |
|---|---|---|
| NVIDIA A100-SXM4-80GB | Merlin `gmerlin7` cluster, `a100-*` partitions: five nodes of four GPUs in an NV4 all-pair NVLink3 mesh, EPYC host | Capture-capable: environment qualified by [a100_environment_qualification_v1](../../examples/a100_environment_qualification_v1/RESULTS.md); the production protocol is gated on the COMP-53 freeze amendment and the COMP-45 cycle-normalized publication form |
| NVIDIA GH200 | Merlin `gmerlin7` cluster, `gh-*` partitions: three nodes of four GPUs, Grace aarch64 host | Envelope measured by [gh200_hardware_envelope_v1](../../examples/gh200_hardware_envelope_v1/RESULTS.md); framework capture needs a qualified CUDA 12 aarch64 environment lane first (COMP-5 scope) |
| NVIDIA GTX 1660 Ti (TU116) | Local workstation | Anchor only: profiler counters are denied and display sharing breaks stability, so it can never qualify (COMP-5); its calibrated host profiles and fidelity study transfer method, not numbers |
| NVIDIA H100, B100, B200 | No reachable silicon | Derived lane only; calibrated requests fail closed today |
| AMD (`amd-rocm-target` slot) | No reachable silicon | Campaign slot unbound; binds to one immutable target identity when silicon is reachable |

Two standing facts shape the rows. The Hopper lane's first reachable
silicon is GH200, which is its own target identity, not an H100: SKU,
memory system and host differ, and host-coupled constants (launch cost,
host step cost) are measured per host and never transferred across
hosts. And the Accel-Sim sidecar is qualified only for a declared SM80
compute and memory region, so it is unreachable for GH200 and every
other non-A100 row.

## Models (columns)

A column's identity is the exact checkpoint: name, revision, config and
weight hashes, as recorded in its suite file. Columns are added by
COMP-54 extraction, which content-addresses the model's kernel inventory
before any cell of that column runs, and whose freeze verifies that the
pinned framework versions execute the architecture. A column the pinned
frameworks cannot execute is blocked pending a maintainer pin decision,
never silently bumped past the pins.

| Column | Identity | State |
|---|---|---|
| granite-3.0-1b-a400m-instruct | `ibm-granite/granite-3.0-1b-a400m-instruct`, revision `ffec3c35`, pinned with hashes in the [transformer-dag-v1 suite](../../offline/calibration/suites/transformer-dag-v1/suite.json) | Offline denominators published by [model_extraction_v1](../../examples/model_extraction_v1/RESULTS.md) and the [framework pin bump](../../examples/framework_pin_bump_v1/RESULTS.md): vLLM [v0.26.0](../../offline/calibration/model-inventories/e74e995a89588a304aa852593d3505cfab9a94d2c068c82dbe9c776da7119af9.json) and [v0.27.1](../../offline/calibration/model-inventories/33758c3c71d5dacae8f6a82cb937f5e70b0d28eaa7c2358c13baccbd8d8725e2.json); SGLang [`8f2a3ad`](../../offline/calibration/model-inventories/147fe4398d5615afe7954c9199134de37f706da2cecda8fc37d6514ad936c54c.json) and [`bfeae4e`](../../offline/calibration/model-inventories/3998b208bef6498709a9a4b6b2ca2e1825a9db54918186f4fc4387a9ee2b9b9a.json). This is the anchored first column and the small-MoE control for the beyond-node-memory route |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B` (open Apache 2.0 dense multimodal checkpoint, 27.78B parameters, architecture `Qwen3_5ForConditionalGeneration`); exact revision and hashes pinned at its extraction freeze | Planned (COMP-54). The architecture is present in both pinned framework registries (pin-bump verification 2026-08-25); the extraction freeze owns the binding verification |
| Kimi K3 | `moonshotai/Kimi-K3` (open-weight 2.8T-parameter MoE, 104B active, 896 experts with 16 selected, hybrid linear attention, native MXFP4 checkpoint, released 2026-07); exact revision and hashes pinned at its extraction freeze | Planned behind COMP-59. Both pinned registries resolve `KimiK3ForConditionalGeneration` through their native Kimi K3 implementations (pin-bump verification 2026-08-25); COMP-54 still owns exact checkpoint binding, and no Kimi weights were downloaded or executed by the pin bump |

Closed-weight models are out of scope regardless of interest: a column
requires open weights because capture executes the real checkpoint, so the
Qwen3.8-Max preview has no column until its weights are open. Initial
declared grids are text-only; multimodal encoder kernel families enter as
later grid extensions declared in their own extraction records.

## Launch modes

Every cell's denominators span the launch modes the backend supports: eager
and captured CUDA graph, per the frozen measurement design. vLLM serves
decode under CUDA graphs by default, so the captured-graph lane is the
deployment-realistic one. Launch mode never participates in kernel
dispatch; the launch-mode-conditioned host residual is owned by COMP-48,
and in-graph instrumentation follows the constraint the
[a100_graph_launch_v1](../../examples/a100_graph_launch_v1/RESULTS.md)
study measured: the driver refuses CUDA-event timing on capture-recorded
events, so in-graph evidence rides the profiler lane.

Capture additionally splits by serving pool, matching disaggregated
deployment: the prefill pool's eager or piecewise-compiled kernel stream
and the decode pool's replayed CUDA graph are separate capture
denominators for every cell, owned by the same VLLM-12 and SGL-10
producers (maintainer direction 2026-08-24).

## Models beyond node memory

A frontier MoE column is filled without hosting the full checkpoint. The
standing kernel-time determinism ruling makes a kernel constant a pure
function of kernel family, phase, shape inputs and architecture profile,
so the column needs the model's kernel families at their deployment
shapes, not its full working set. The fleet arithmetic decides the route:
a 2.8T-parameter MXFP4 checkpoint is roughly 1.4 TB of weights, against
1.6 TB of total HBM across all five A100 nodes (twenty GPUs of 80 GB) and
about 1.15 TB across all three GH200 nodes (twelve GPUs of 96 GB), before
any KV or activation memory and atop an inter-node path that runs NCCL
over the kernel socket transport. Full-model serving capture for that
class is therefore out on this fleet.

Physical capture never requires the real checkpoint: the frameworks
instantiate the architecture with dummy-weight loading, so no download
and no real inference is involved, and only the resident parameter bytes
of the instantiated depth bound the target (maintainer direction
2026-08-24). The route: COMP-54 extracts the structure half from the
model configuration alone, and COMP-59 owns the reduced-depth
same-geometry physical capture envelope, now with dummy weights, that
supplies observed implementation identities and per-expert load grids on
the reachable targets. That
envelope is its own declared identity: it never claims the full
checkpoint's routing population, weights or end-to-end makespans. Numeric
format keeps cells honest per target: a native MXFP4 checkpoint executes
different kernel implementations on SM80 than on SM90, and the dispatch
signature separates them.

## vLLM matrix (pinned v0.27.1)

| Target | granite-3.0-1b-a400m-instruct | Qwen3.8-27B | Kimi K3 |
|---|---|---|---|
| A100-SXM4-80GB | **blocked**: COMP-53 amendment, then the COMP-45 protocol, then capture (VLLM-12, COMP-6). Retained non-filling evidence exists: the void [a100_kernel_constants_v1](../../examples/a100_kernel_constants_v1/RESULTS.md) and [a100_graph_launch_v1](../../examples/a100_graph_launch_v1/RESULTS.md) measured granite kernel families as microbenchmarks outside the framework chain | **planned**: follows the granite cells; single-GPU fit (about 56 GB of BF16 weights on an 80 GB part) with TP 1, 2 and 4 sweeps on the NV4 mesh; pin support verified at the extraction freeze | **blocked**: COMP-59 must qualify the reduced-depth envelope on the MXFP4-on-SM80 lane |
| GH200 | **planned**: after the A100 cell and the GH200 environment qualification | **planned**: after the environment qualification; single-GPU fit with headroom | **blocked**: COMP-59 must qualify the reduced-depth envelope on SM90 |
| GTX 1660 Ti (TU116) | **anchor**: [compute_fidelity_v1](../../examples/compute_fidelity_v1/RESULTS.md) and [host_step_cost_v1](../../examples/host_step_cost_v1/RESULTS.md); never fills | not planned | not planned |
| H100, B100, B200 | **derived** (COMP-52); fail closed today | **derived** (COMP-52) | **derived** (COMP-52) |
| AMD slot | unbound | unbound | unbound |

## SGLang matrix (pinned main commit bfeae4e)

| Target | granite-3.0-1b-a400m-instruct | Qwen3.8-27B | Kimi K3 |
|---|---|---|---|
| A100-SXM4-80GB | **planned**: the SGL-10 producer follows the vLLM cell on the same qualified environment | **planned**: follows the vLLM cell; pin support verified at the extraction freeze | **blocked**: COMP-59 must qualify the reduced-depth envelope after the vLLM route lands |
| GH200 | **planned**: after the A100 cell and the GH200 environment qualification | **planned**: after the environment qualification | **blocked**: COMP-59 must qualify the reduced-depth envelope on SM90 |
| GTX 1660 Ti (TU116) | **anchor**: [sglang_host_step_v1](../../examples/sglang_host_step_v1/RESULTS.md) Turing host profiles; never fills | not planned | not planned |
| H100, B100, B200 | **derived** (COMP-52); fail closed today | **derived** (COMP-52) | **derived** (COMP-52) |
| AMD slot | unbound | unbound | unbound |

## Fill order

Maintainer direction (2026-08-24): measured silicon leads on the
reachable GPUs, and the sidecar is demand-driven.

1. COMP-54 extraction enumerates the granite column offline for both
   frameworks; every later column enters the same way.
2. A100 granite cells: land the COMP-53 amendment, run the COMP-45
   cycle-normalized protocol, then the vLLM capture and measurement
   campaign, then the SGLang cell on the same environment.
3. GH200 lane: qualify the CUDA 12 aarch64 environment under COMP-5,
   then repeat the granite cells.
4. Qwen3.8-27B cells follow the granite cells on each qualified target,
   after its extraction freeze verifies pinned-framework support.
5. Kimi K3 cells enter through the beyond-node-memory route after COMP-59
   lands, with the granite reduced-depth control validated first.
6. Further columns by nomination: extraction first, then rows in the
   same order.
7. The Accel-Sim sidecar (COMP-51, Wave 1B) proceeds only when a
   measured A100 column exposes an explicitly missing exact point that
   measurement cannot serve; it never substitutes for a measurable
   kernel, and it never appears on a non-A100 row.

## Update discipline

Every cell transition lands in the same change as the evidence it
cites: a study `RESULTS.md`, a dataset manifest or a device release. A
transition to measured requires a non-void study; a void run leaves the
cell state unchanged and links the retained findings. Cells never carry
numbers. The matrix is reconciled at integration time together with the
registry, so a cell claim without its owning task or evidence link is a
violation of the same kind as an unregistered deferral.
