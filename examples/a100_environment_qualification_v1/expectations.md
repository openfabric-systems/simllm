# A100 environment qualification v1 expectations

## Freeze scope and chronology

This is the expectations-only record for the first Merlin A100 qualification
run supporting COMP-5. It is committed before the probe implementation, the
Slurm job, any profiler invocation in this study, and any result-producing
run. No result from this study may be used to edit this file afterward.

This qualification is deliberately narrower than compute calibration. It
tests whether one allocated A100 can support the activity and counter evidence
that COMP-5 requires. It does not capture a model, fit a timing parameter,
publish a profile table, close COMP-1, or claim SGLang TTFT or TPOT accuracy.

If the environment qualifies, a separate expectations-only record will freeze
the production SGLang kernel matrix before that matrix is implemented or run.
That later record will own COMP-1, SGL-24, COMP-6 and SGL-10 evidence.

## Registry motivation

COMP-5 is the first blocker on the execution-fidelity path. COMP-1 cannot
replace the A100 bootstrap or flat roofline surrogate until a controlled target
environment supplies activity timing, required counters, exact provenance and
dynamic SASS evidence. SGL-24 then needs SGLang's own device-visible launch
count rather than the current bracket transferred from vLLM.

The qualification result can establish only the first part of COMP-5:

- exact target and tool provenance;
- basic CUDA execution on the allocated GPU;
- a nonempty Nsight Systems CUDA activity trace;
- successful access to a basic Nsight Compute counter set;
- nonempty static SASS for a known probe binary;
- the clock, power, MIG, MPS and process state surrounding those probes.

Pinned NVBit tracing, Accel-Sim compatibility, controlled-cell stability and
production framework kernels remain outside this run. Static SASS is not a
substitute for a dynamic NVBit trace.

## Pre-freeze facts

These facts were observed before this freeze and are context, not scored
outcomes:

- SimLLM main is
  `dddf8fbf70e2b168dcd43ccf6799496d1ab9be11`.
- The adapter pins SGLang
  `8f2a3ad6d7d68c58ae65b61a75bb2115449addca`.
- The future production study pins Granite
  `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`.
- The scheduler exposes A100 SXM4 80 GB GPUs with compute capability 8.0.
- An unrelated image smoke already observed an A100 SXM4 80 GB, driver
  565.57.01 and working CUDA execution. It did not run the exact pinned
  SGLang source, Nsight Systems, Nsight Compute, NVBit or calibration code.
- The CUDA 12.9.1 environment exposes `nvcc`, `cuobjdump`, Nsight Systems
  2025.1.3 and Nsight Compute 2025.2.1 on the login host. Tool presence there
  does not establish that profiling works in a GPU allocation.
- No NVBit or Accel-Sim source is staged in the project environment.
- The existing `compute_family_benchmark.cu`,
  `gpu_fixed_cost_probe.cu` and Turing calibration runner refuse non-TU116
  targets. They are method anchors and will not be run unchanged on A100.

No kernel timing, activity row or performance counter from the proposed probe
has been observed before this commit.

## Frozen allocation envelope

The qualification uses one batch allocation:

| Resource | Frozen request |
|---|---:|
| cluster | `gmerlin7` |
| partition | `a100-hourly` |
| account | `merlin` |
| nodes and tasks | 1 node, 1 task |
| GPU | 1 `nvidia_a100-sxm4-80gb` |
| CPU | 4 CPUs for the one task |
| host memory | 32 GiB |
| wall limit | 20 minutes |
| expected runtime | 5 to 15 minutes |
| planned scratch | below 5 GiB |
| hard scratch ceiling | 10 GiB |

The job does not request an exclusive node. Slurm must grant the selected GPU
to this job alone, and the probe verifies that no foreign process uses that
GPU. The job performs no download, package installation, image construction,
model load, SGLang serving, full calibration matrix or Accel-Sim replay.

Compilation, CUDA execution, profiling and report export run only inside the
allocation. Login-host activity is limited to submission, scheduler queries
and small log reads.

## Frozen probe

The implementation will add one small, repository-tracked CUDA source after
this commit. It has one uniquely named FP32 vector-add kernel, deterministic
input initialization, a correctness check and explicit warmup. It is compiled
inside the allocation with CUDA 12.9.1, optimization, line information and an
`sm_80` target. The source and binary SHA-256 values are recorded.

The probe is intentionally not a production kernel and its duration does not
calibrate any SimLLM provider. It exists only to test the capture tools against
a known launch. Nsight Systems and Nsight Compute run in separate processes so
counter replay cannot contaminate the activity trace.

The fixed launch contains 16,777,216 FP32 elements, 256 threads per block and
the minimum grid covering all elements. The target kernel performs one add per
element, two FP32 reads and one FP32 write. The result reports these authored
work counts, but it does not turn profiler duration into a bandwidth or FLOP
claim because this qualification does not freeze a cache-state experiment.

## Required observations

The result records only an allowlisted provenance surface, never a complete
environment dump:

1. Slurm job, cluster, partition, node, requested TRES and allocated TRES.
2. `CUDA_VISIBLE_DEVICES`, CUDA-visible device count, exact GPU name, UUID,
   PCI identity, compute capability and total memory.
3. Driver, runtime, compiler, `cuobjdump`, Nsight Systems and Nsight Compute
   versions.
4. GPU compute mode, MIG state, MPS or compute-process inventory, persistence
   state, current and allowed SM and memory clocks, power limit, power draw and
   temperature immediately before and after the profiler probes.
5. Unprofiled probe exit status and exact output checksum.
6. A CUDA-only Nsight Systems report and exported `cuda_gpu_trace` CSV with at
   least one row for the target kernel, including device, context, stream,
   start, duration and launch dimensions.
7. A bounded Nsight Compute basic-set pass over one target launch, with a
   profiled target kernel and at least one numeric metric.
8. Nonempty `cuobjdump --dump-sass` output containing the target kernel and an
   `sm_80` code object.
9. Scratch usage after each profiler and the hashes and sizes of every retained
   compact artifact.

Each profiler has a three-minute timeout. Raw profiler databases remain under
the configured run root. The repository receives no raw profiler database,
binary or bulk trace.

## Decision states and fatal guards

This capability study has no scored behavioral denominator. It produces one
of three states:

- `QUALIFIED`: every fatal guard holds and both profiler probes succeed.
- `BLOCKED`: structural guards hold, but counter permission, profiler support,
  clock-policy evidence or another named site capability is unavailable.
- `VOID`: a structural guard fails, so the profiler evidence cannot be
  interpreted for the declared target.

The following failures make the run void:

- the SimLLM revision or probe source differs from the frozen identities;
- Slurm exposes zero or more than one GPU to the task;
- the visible GPU is not an A100 SXM4 80 GB at compute capability 8.0;
- MIG is enabled, another process uses the allocated GPU, or CUDA and
  `nvidia-smi` disagree on identity;
- the unprofiled launch or its correctness check fails;
- the Nsight Systems report or exported target-kernel activity row is empty;
- the static SASS or target code object is empty or mismatched;
- scratch exceeds 10 GiB, a profiler exceeds its timeout, an artifact hash is
  missing, or output escapes the configured run root.

`ERR_NVGPUCTRPERM`, an empty metric set, or another explicit counter-policy
denial yields `BLOCKED`, not `QUALIFIED` and not a fabricated zero. The same is
true if the site does not expose sufficient clock-policy evidence for the
later controlled capture. The result retains the exact diagnostic and names
the administrator capability needed next.

## Safety and compatibility rules

The job queries GPU state but does not set clocks, reset a GPU, change compute
mode, enable or disable MIG, alter MPS, use elevated privileges, change driver
profiling policy or inspect another user's data. It does not attempt to work
around a denied counter capability.

No existing SimLLM provider, artifact, test baseline, timestamp or source file
is changed by this study. The qualification result cannot populate an A100,
H100 or B100 profile table. A later A100 profile must reject H100 and B100
queries rather than transfer its measurements silently.

## Next gate

Only a `QUALIFIED` result permits preparation of the production kernel study.
That separate expectations-only record will reuse the already published BF16
single-GPU matrix in `sglang_moe_workload_v1`, resolve its exact train and
held-out membership, and freeze:

- exact SGLang, model, PyTorch, CUDA and kernel binary identities;
- eager or CUDA-graph mode, cache protocol, warmup and 41 repetitions;
- complete worker-step settlement, device completion frontier, launch count,
  kernel-busy union and exposed gaps;
- per-invocation layer and semantic mapping, streams, events and NCCL calls;
- activity, counters and dynamic SASS evidence;
- physical floors and ceilings;
- held-out kernel median and p95 error below 10 and 20 percent;
- per-phase median and p95 error below 5 and 10 percent;
- compute-only full-step error below 5 percent;
- controlled-cell coefficient of variation below 2 percent;
- exact calibration-off compatibility.

A `BLOCKED` result instead ends the campaign at the named capability until the
site state changes. It does not justify a synthetic replacement.
