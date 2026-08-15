# SGLang A100 kernel pilot v1 results

The reviewed study state is `BLOCKED`. Slurm job `195365` held the reached
parent-observable allocation, source, model, ideal-artifact, confinement,
GPU-state and postcondition guards, then stopped before the first timing
observation because the frozen PyTorch CUDA 13.0 runtime is incompatible with
Merlin driver 565.57.01. The child did not reach its imported-source, runtime
package, backend or CUDA identity checks; those identities are unobserved, not
passed. The behavioral result is therefore `0/0, blocked before behavioral
execution`, not a failed score and not a void run.

No device duration, Nsight Systems range, Nsight Compute counter, SGLang
launch bracket or transferred-vLLM signed error was measured. SGL-24 remains
open. This run closes no task and makes no claim for COMP-1, COMP-5, COMP-6,
SGL-10 or SGL-4.

## Freeze integrity and chronology

The expectations-only commit
`b825285f24024a4ae41453c418672c89f03379cd` preceded the implementation and
all three Slurm attempts. The Markdown freeze remained at SHA-256
`dcb7446682e0f4dbc5ed524bba66c87ccd01e0b9136758498cad48a4962abb8f`.
The machine-readable freeze remained at SHA-256
`8773e8e090b5ed939b4ed17d3fa932e266ee946b80bf82dff1e9af5968aa2edd`.
No observed value was written back into either file.

The final diagnostic run observed repository commit
`b165da24b9501e2d49ad61e937fac7ff5bcf302f`. Its complete source bundle had
SHA-256
`fbe39886e5b7c149079e8818ba503f9f3d1ea1a5cb4f5e333e1a4a64c927fa1b`,
and the submitted Slurm wrapper had SHA-256
`f7802a2df98c562b309937a26e13cea19235c6cfd797a960a556147e57ae50c9`.
The reconstructed bundle was clean at the expected commit, contained the
frozen base as an ancestor and passed `--check-only` without creating output.

Three attempts are retained. Harness corrections after the first two attempts
are post-specified diagnostic repairs. They did not alter the frozen runtime,
model, workload, profiler, allocation or acceptance clauses.

| Job | Observed commit | Slurm outcome | Reviewed state | Finding |
|---|---|---|---|---|
| `195360` | `9af374777ee50b271026c2d545cb7926e4ac3c8d` | exit 127 after 24 seconds | wrapper failure | The staged virtual environment could not locate its shared Python 3.12 library. No Python child, CUDA context, model step or profiler ran. |
| `195362` | `9af374777ee50b271026c2d545cb7926e4ac3c8d` | exit 2 after 5 minutes 30 seconds | `BLOCKED` | The first prefill timing child reached the frozen 300-second command timeout. The runner retained the clean shared context but did not yet retain child progress or partial stderr, so the internal phase was unknown. |
| `195365` | `b165da24b9501e2d49ad61e937fac7ff5bcf302f` | exit 2 after 5 minutes 29 seconds | `BLOCKED` | The child journal and partial stderr localize the stop to runtime import and identify the frozen CUDA incompatibility. |

Commit `9af374777ee50b271026c2d545cb7926e4ac3c8d` confined temporary, cache,
configuration and state roots; enforced the phase watchdog; validated live
forward modes and model provenance; placed the CUDA end event before global
device settlement; and made persistence and cleanup fail closed. The wrapper
repair for job `195362` resolved and recorded the shared Python library without
changing the runtime. Commit
`b165da24b9501e2d49ad61e937fac7ff5bcf302f` then added the atomic child
progress journal and bounded partial-output retention. Those repairs explain
the failure more precisely but do not make the blocked study a qualifying
measurement.

## Blocker evidence

The final child progress ledger contains exactly these stages:

1. `entered`
2. `preflight_validated`
3. `runtime_import_started`

It never records `runtime_import_completed`, CUDA identity, model load,
warmup or retained measurement. The parent then terminated the process group
at the frozen 300-second command timeout.

The retained child stderr reports that PyTorch was built for CUDA 13.0 and
that the NVIDIA driver exposes driver API level 12.7. PyTorch consequently
reported that the driver was too old while probing CUDA device availability.
This is the incompatibility that the expectations explicitly classified as
`BLOCKED`; the harness did not switch to another runtime, profiler, SGLang
checkout or privilege level.

The final compact evidence is independently reproducible by these hashes:

| Evidence | SHA-256 |
|---|---|
| wrapper manifest | `1e42e70adbfca817849a26b38998bcb61d493ff070e5cf1d3a4bab1f84ef1da4` |
| wrapper ledger | `c0cc3f319da46a7b4ecfdeb44977505e70d1f9ad9a5eec42a4a5f4961e87b540` |
| result | `f8021f3ab43072b289cf797f7a6af031ef447964189917e29cc2c48ca937d857` |
| child progress | `3339397cd31fe5c802cf8b173b0bae5c77b3c39ee8e57e084531c8bee637d834` |
| retained child diagnostic | `ca96624dd7142604b3c0d81d1e5a20ac65edd8b34d25085c015c5d5e3015213e` |

Every entry in the retained wrapper manifest verifies. The result and partial
context are byte-identical, the result state is `BLOCKED`, and
`postcondition_errors` is empty. The initial static contract also rechecked
the exact ideal artifact before the child launched.

## Allocation and safety outcome

All model loading, imports, CUDA initialization attempts and process waiting
ran inside the Slurm allocation. Login-node activity was limited to small
source and metadata reads, hashing, staging, `sbatch`, sparse scheduler and
accounting queries, and compact evidence transfer.

For final job `195365`, Slurm reported the following envelope:

| Resource | Frozen request | Observed allocation or use |
|---|---:|---:|
| node and task | 1 node, 1 task | 1 node, 1 task |
| GPU | 1 A100 SXM4 80 GB | 1 A100 SXM4 80 GB |
| task CPU | 8 CPUs for the task | `CPUs/Task=8`; site allocation accounted 16 logical CPUs |
| host memory | 64 GiB | batch maximum RSS 3,072,344 KiB |
| wall limit | 45 minutes | 5 minutes 29 seconds |
| scratch hard ceiling | 20 GiB | 2,708,726,951 bytes before cleanup |
| retained hard ceiling | 4 GiB | 71,684 bytes |

The job ran on the nonexclusive `a100-hourly` partition under site QOS
`gpu_general`; scheduler `OverSubscribe` was `OK`. The runner observed one
visible A100, disabled MIG, no foreign compute process and the frozen driver.
The stable GPU identity, mode, maximum-clock, supported-clock, power-limit and
MIG invariants held after the blocked child. Allowed live telemetry such as
current clocks, power and temperature was not treated as immutable. The exact
job scratch directory was removed by the cleanup trap.

The job made no clock, power, persistence, compute-mode, MIG, MPS, driver or
other-user process change. It performed no download, install, image pull or
dependency build.

## Physical and behavioral interpretation

No measured step reached the physical review. The frozen compute and HBM
bounds therefore have no observed value to compare against. Reporting those
bounds as passed would be false, as would reporting a launch-count score with
a zero denominator.

Likewise, the absence of Nsight rows is not evidence of zero launches. It is
evidence that the required runtime could not initialize far enough to produce
an interpretable capture. The transferred vLLM bracket `[440, 567]` remains
the active SGLang surrogate and its signed error remains unknown.

## Next gate

An earlier, separate deployment smoke on the same Merlin A100 provides a
candidate compatibility substrate: job `195220` ran an immutable OCI image at
manifest digest
`sha256:bbe1ec8694ab6b0e6ea0a567c2d492ed7c9bed13cb33ffe356ade0aaa8343c9f`
with SGLang 0.5.17 and PyTorch `2.11.0+cu129`, reported CUDA 12.9 available and
completed a real `sgl.Engine` generation. The image labels identify upstream
SGLang commit `29481685462732237d80d86076d6563e1f658102`. That smoke is
compatibility evidence only. It is not the SGL-24 measurement because it does
not import the exact SGLang source frozen by this pilot.

The smallest valid continuation is a separate v2 expectations-only commit
that retains the exact SGLang commit and tree, model, workload, backends,
profilers, controls and launch-count acceptance while selecting a new,
versioned CUDA 12.9 runtime. The CUDA 13.0 environment remains immutable v1
evidence. Runtime construction must occur in a bounded Slurm job, never on a
login node, and must record the resolved CUDA-specific wheel or installed
binary hashes. A short CUDA initialization gate should fail before importing
SGLang if the new runtime is not compatible.

Only a valid decode capture may close SGL-24. COMP-5 and COMP-1 still require
controlled-clock production cells, dynamic SASS, pinned Accel-Sim replay,
stability and immutable holdouts after that launch-demand gate succeeds.
