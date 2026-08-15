# SGLang A100 kernel pilot v2 expectations

## Freeze scope and chronology

This is the expectations-only record for the CUDA 12.9 continuation of the
first production-kernel pilot on one Merlin A100. It precedes the v2 runtime
qualification, harness implementation, exact-target SGLang import, CUDA
initialization, profiler invocation and every v2 result-producing run. No v2
observation may be used to edit this file afterward.

This is a new experiment, not a rewrite of v1. The v1 expectations-only
commit was `b825285f24024a4ae41453c418672c89f03379cd`; its Markdown and JSON
SHA-256 values are respectively
`dcb7446682e0f4dbc5ed524bba66c87ccd01e0b9136758498cad48a4962abb8f`
and `8773e8e090b5ed939b4ed17d3fa932e266ee946b80bf82dff1e9af5968aa2edd`.
The corrected [v1 result](../sglang_a100_kernel_pilot_v1/RESULTS.md), at
commit `43064d6ae88d6380c86bd400d336a07aa504ccbd`, has SHA-256
`2f3c6933ae0ed4a1487add53ee07f2b3a79bea9b145d8ab65bc0db83b54f9133`.
Job `195365` observed no model step or launch count and remains `0/0,
blocked before behavioral execution`.

The runtime choice is informed by prior evidence. Separate job `195220`
used OCI manifest
`sha256:bbe1ec8694ab6b0e6ea0a567c2d492ed7c9bed13cb33ffe356ade0aaa8343c9f`
and showed that its PyTorch 2.11.0 CUDA 12.9 build could initialize the same
A100 and complete an Engine generation. That image contains SGLang 0.5.17
from commit `29481685462732237d80d86076d6563e1f658102`, not the exact source
owned by SGL-24. The smoke therefore supports the compatibility substrate
choice only. It is neither SGL-24 evidence nor a previously unobserved
prediction. The v2 launch counts, family ledger, timing relations and signed
transfer errors remain unobserved.

## Task ownership and closure boundary

SGL-24 owns the immediate result. Its active surrogate is the vLLM 0.26.0
device-launch bracket `[440, 567]`. V2 must produce a SGLang-specific
device-visible launch bracket for one fixed decode geometry, report its signed
distance from both transferred endpoints and prove that the ideal path is
unchanged.

No other task may close from this pilot. COMP-1 and COMP-5 still require
controlled clocks, the registered matrix, dynamic SASS, pinned Accel-Sim
replay, immutable train and holdout cells, stability and held-out error checks.
COMP-6 still requires usable per-invocation trace and table keys. SGL-10 still
requires an identity-keyed `ExecutionGraph`. SGL-4 still requires paced Engine
and HTTP TTFT and TPOT evidence.

SGL-24 closes only if the overall required-evidence state is `VALID`, both
timing and Nsight Systems lanes are `VALID` for both anchors, the measured
bracket and signed transfer errors are published, every launch is conserved
in the device ledger and the ideal artifact remains byte-identical. Nsight
Compute is an optional diagnostic lane; its explicit `BLOCKED` state does not
prevent closure when every required lane and shared guard is valid.

## Frozen target and runtime authorities

The target remains the clean SGLang checkout at commit
`8f2a3ad6d7d68c58ae65b61a75bb2115449addca` and tree
`5be26db1f559064c0f9e724e78c1a8f619754867`. Its source version is
`0.0.0.dev1+g8f2a3ad6d`. The stock `ModelRunner` path remains:

```text
ModelRunner.forward
  -> ModelRunner._forward_raw
  -> EagerRunner.execute
  -> EagerRunner._execute_extend or EagerRunner._execute_decode
  -> GraniteMoeForCausalLM.forward
  -> LogitsProcessor
  -> ModelRunner.sample
```

The study uses the pinned `sglang.benchmark.one_batch` helpers and does not
select the SimLLM worker or plugin. Inside the allocation, `git archive`
reconstructs a clean projection of the frozen commit in job-local scratch.
Only tracked blobs enter that projection. One generated file is then added:
`python/sglang/_version.py` at SHA-256
`315a0924e5dde6902935235d5308bac9d76ae0b8ef44b4e2730891dba90fcceb`.
Its frozen version string is `0.0.0.dev1+g8f2a3ad6d`. No ignored bytecode,
cache or other checkout artifact enters the projection, and Python bytecode
writes are disabled.

Every loaded `sglang` source file must match either its Git blob at the frozen
commit or that one generated version file. `sglang.__path__` and every package
search location contain only the clean projection. The container's installed
SGLang 0.5.17 distribution is substrate metadata only and may not supply an
imported module. Discovered `sglang.srt.platforms` and `sglang.srt.plugins`
entry-point inventories are both exactly empty, and no non-core plugin hook may
replace a class or method. Per-child loaded-module origin and hash ledgers are
retained. A wrong source blob, search location, plugin inventory or applied
non-core hook voids the run.

The model-step target is unchanged from v1. Its runtime is now this explicit,
immutable CUDA 12.9 dependency substrate; packaging identities are not assumed
equal to the v1 virtual environment:

| Identity | Frozen value |
|---|---|
| SimLLM base | `43064d6ae88d6380c86bd400d336a07aa504ccbd` |
| target SGLang commit | `8f2a3ad6d7d68c58ae65b61a75bb2115449addca` |
| target SGLang tree | `5be26db1f559064c0f9e724e78c1a8f619754867` |
| target source version | `0.0.0.dev1+g8f2a3ad6d` |
| generated version file SHA-256 | `315a0924e5dde6902935235d5308bac9d76ae0b8ef44b4e2730891dba90fcceb` |
| model | `ibm-granite/granite-3.0-1b-a400m-instruct` |
| model revision | `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445` |
| config SHA-256 | `ca4bb3a5c1bdef988ab413e0d731640446da65316e4ed16de3666cd96ecc3a0b` |
| weight object SHA-256 | `f7ae1cee56a9ea6c5360437b1c0407f8d84816b2cc75470f4e7e5236fa2a07dc` |
| weight file size | 2,669,283,096 bytes |
| OCI manifest digest | `sha256:bbe1ec8694ab6b0e6ea0a567c2d492ed7c9bed13cb33ffe356ade0aaa8343c9f` |
| OCI config digest | `sha256:0dbc9c00134e6c42b7d2cf291023b74fba3be2eaebd7a274603c869fae57188e` |
| OCI index SHA-256 | `508033e7f177c186de2ac6ebf7f19008d1bb6345f42b0efdc042996f2dd1e9cd` |
| Python | 3.12 ABI; exact patch recorded from the immutable OCI image |
| PyTorch | `2.11.0+cu129` |
| PyTorch CUDA build | 12.9 |
| CUDA userspace | 12.9.1 |
| SGLang kernel package | 0.4.5, CUDA 12.9 wheel variant |
| Triton | 3.6.0 |
| Transformers | 5.12.1 |
| profiler module | `cuda/12.9.1` |
| Nsight Systems | 2025.1.3 |
| Nsight Compute | 2025.2.1 |
| driver | 565.57.01 |
| GPU | `NVIDIA A100-SXM4-80GB`, compute capability 8.0 |
| execution | BF16, unquantized, eager |
| parallelism | TP=EP=PP=DP=1 |
| attention backend | `triton` |
| MoE runner backend | `triton` |
| MoE A2A backend | `none` |
| sampling backend | `pytorch` |

The v1 CUDA 13 environment is immutable evidence and is never activated,
modified or repaired by v2. The guard rejects a normalized distribution name
ending in `-cu13`, a local version or wheel/direct-URL identity containing
`cu13`, `cuda-python` major version 13 or newer, and CUDA 13 user-space SONAMEs
such as `libcudart.so.13`, `libnvrtc.so.13`, `libcupti.so.13` and
`libcublas.so.13`. Every loaded CUDA object must also belong to the frozen OCI
binary manifest or the explicit host-driver and host-profiler injection
allowlists. A forbidden or unowned distribution, extension or shared object
voids the run.

## Runtime construction and qualification

The job reconstructs a job-local Apptainer sandbox from the cached,
content-addressed OCI layout. The layout has 69 compressed layers totaling
18,690,951,253 bytes. Construction is offline and occurs only inside the
Slurm allocation. The source OCI index, manifests, configs and blobs are
hashed before and after extraction and must remain byte-identical. The
407-byte index has SHA-256
`508033e7f177c186de2ac6ebf7f19008d1bb6345f42b0efdc042996f2dd1e9cd`
and must resolve exactly the frozen manifest digest. No image pull, package
install or dependency build is permitted. The input checkout is never
modified, and the clean projection receives no patch beyond the one frozen
generated version file.

The job reads the exact checkout only to verify Git identity and produce the
clean tracked projection. That projection is then bound read-only and placed
before site packages for Python imports. The model snapshot is copied into
job-local scratch before model loading. Temporary files, caches,
configuration, compiled artifacts and profiler outputs are confined to the
job scratch root. Machine-specific locations are supplied through
project-prefixed environment variables and are never embedded in tracked
files.

Before invoking Apptainer, the launcher rejects every inherited
`APPTAINER_`, `APPTAINERENV_`, `SINGULARITY_` and `SINGULARITYENV_` control or
container-injection variable. It then sets only audited scratch-local cache
and temporary roots. Every container launch uses
`--cleanenv`, `--contain`, disables configured bind paths, cwd and hostfs
automatic mounts, and sets a working directory in job scratch outside both
the clean target projection and the substrate image's packaged source. The
user data binds are the frozen allowlist: read-only runner, clean source,
model and exact host-profiler installations plus read-write result and cache
roots. Apptainer's minimal contained system mounts and `--nv` driver injection
are separate frozen mount authorities. The effective mount inventory is
retained, and any other host-backed bind is fatal. Host home, host temporary
directories and the launch cwd are never visible, and `HOME` is not
overridden.

The only forwarded scheduler value is `CUDA_VISIBLE_DEVICES`. Fixed overrides
set all Hugging Face and dataset offline controls, disable telemetry, package
networking and Python user-site access and bytecode writes, fix tokenizer
parallelism, set all four thread controls to 8, and require
`SIMLLM_SGLANG_ENABLE=0` plus `SIMLLM_SGLANG_ORACLE_CAPTURE=0`. Explicit
project-prefixed variables provide the model, source, result and scratch
roles. Explicit path variables route temporary, Hugging Face, SGLang, Triton,
TorchInductor, Torch extension, CUDA and all XDG state to scratch. Python home,
virtual and Conda environments, and upper- and lower-case proxy variables are
unset. No host Python path, SGLang option, credential or library path is
inherited. The OCI-config environment plus this normalized allowlist, its
role-valued paths and its mount policy are part of the immutable environment
manifest.

Before importing SGLang, a separate child with a 60-second timeout imports
only PyTorch and checks the exact Python, PyTorch and CUDA build identities. It
must see exactly one A100, correlate its UUID and name with `nvidia-smi`,
allocate and reduce a small CUDA tensor and synchronize successfully. Matching
runtime identity with unavailable CUDA is `BLOCKED`; a package, driver, GPU or
allocation identity mismatch is `VOID`.

The subsequent SGLang import gate has a 120-second timeout. It verifies the
target commit and tree, target blob hashes and package search locations,
dependency versions, empty external entry-point inventories, absence of
non-core replacement hooks, backend selection and absence of CUDA 13 packages
and loaded objects. Exact artifacts that fail to import or expose an ABI
incompatibility yield `BLOCKED`; a wrong artifact or authority yields `VOID`.

The immutable environment manifest contains the OCI identities, complete
installed-distribution inventory and provenance, installed CUDA extension and
critical shared-object SHA-256 values, exact profiler-install manifests,
Python executable identity, clean source-projection manifest, normalized
environment policy and source status.
Every gate, timing and profiler child must reproduce that manifest digest.
Each child separately records the modules and shared objects it actually
loads. Those lazy-load ledgers are checked against permitted origins and
installed hashes but are not required to be identical across unlike children.

## Frozen allocation and storage envelope

The pilot uses one bounded, nonexclusive batch job:

| Resource | Frozen request or bound |
|---|---:|
| cluster | `gmerlin7` |
| account | `merlin` |
| partition | `a100-hourly` |
| nodes and tasks | 1 node, 1 task |
| GPU | 1 `nvidia_a100-sxm4-80gb` |
| CPU | 8 CPUs for the task |
| host memory | 64 GiB |
| wall limit | 45 minutes |
| expected runtime | 18 to 35 minutes |
| internal deadline | 40 minutes |
| free-scratch preflight | at least 180 GiB |
| scratch hard ceiling | 160 GiB |
| retained compact result ceiling | 4 GiB |

There is no exclusive-node request, array or fanout. Login-node work is
limited to small source and metadata reads, hashes, staging, `sbatch`, sparse
scheduler queries and compact log reads. Runtime extraction, source and model
staging, imports, CUDA work, timing, profiling and parsing execute only in the
allocation. Insufficient scratch or a missing cached input yields `BLOCKED`
without expanding the resource envelope.

## Frozen source ledger, model and workload

The semantic invocation ledger is unchanged from v1:

| Semantic site | Multiplicity per step |
|---|---:|
| token embedding and embedding scale | 1 each |
| decoder layer | 24 |
| layer RMSNorm | 48 |
| QKV projection | 24 |
| Q/K/V split views | 72 metadata views, no launch contract |
| rotary embedding | 24 |
| Triton radix attention | 24 |
| output projection | 24 |
| scaled residual expression | 48 |
| router projection | 24 |
| top-k selection | 24 |
| fused MoE dispatch, experts and combine | 24 |
| final RMSNorm | 1 |
| logits processor and LM head | 1 each |
| greedy sample | 1 |

This is not a fabricated raw CUDA count. Nsight Systems identifies the raw
bracket. Every observed kernel maps to a semantic family or explicit
`unattributed` family, and family counts sum exactly to the captured total.

The model geometry remains 24 layers, hidden size 1,024, 16 attention heads,
8 KV heads, head dimension 64, 32 experts per layer, top-k 8, expert width 512
and vocabulary size 49,155. Weights and KV cache are BF16. Quantization, CUDA
graphs, overlap scheduling, radix reuse, chunked prefill and speculative
decoding are disabled. Page size is 1 and memory fraction is 0.60.

Token IDs are generated without a tokenizer:

```text
token_id(request, position) = 1 + ((173 + 257*request + 31*position) mod 49154)
```

Sampling is greedy with temperature 0, `ignore_eos=true`, and no grammar or
logprobs.

| Anchor | Frozen construction | Required runtime assertion |
|---|---|---|
| `prefill-t512-r4` | four 128-token requests, one generated token | `extend_num_tokens == 512`, four requests |
| `decode-b4-c2048` | four 2,047-token requests, uncaptured prefill, then one decode | `seq_lens_cpu == [2048, 2048, 2048, 2048]` after `prepare_for_decode` |

Before each retained instance, request and KV pools are cleared and rebuilt.
JIT and autotuning state remain warm within each phase. There are 10
shape-identical warmups, 41 retained CUDA-event timing repetitions and 5
separate Nsight Systems ranges per anchor. Cache inventory must not change
during a retained or captured step.

## Frozen instrumentation and observables

Observation-only NVTX ranges bracket each complete anchor, layer 0 QKV and
layer 0 fused MoE. Hooks only push and pop ranges. They do not replace a
module, tensor, kernel, stream, allocator, batch or sample.

The exact host Nsight Systems 2025.1.3 installation is bound read-only into
the contained namespace. Its target-side `nsys` executable is invoked there
by an absolute path and wraps the exact container Python child with these
fixed settings: `--trace=cuda,nvtx`, `--sample=none`, `--cpuctxsw=none`,
`--capture-range=cudaProfilerApi`, `--capture-range-end=repeat:5`,
`--cuda-event-trace=false`, `--target-processes=all` and
`--force-overwrite=true`. Each anchor has a separate process. The child calls
`cudaProfilerStart` immediately before and `cudaProfilerStop` immediately
after each synchronized target step.

The exact host Nsight Compute 2025.2.1 installation is likewise bound
read-only, and its target-side executable runs inside containment. It uses the
first kernel inside the layer 0 fused MoE range, mechanically selected from
Nsight Systems, with one anchored escaped demangled-name regex, matching
launch skip, `--launch-count 1`,
`--replay-mode kernel`, `--clock-control none`, `--profile-from-start off`,
`--target-processes all` and the bounded `basic` set. Each profiler command
has a five-minute timeout and each phase step has a two-minute watchdog.

For every retained repetition the result records host settlement, joined
CUDA-event span, shape, output IDs and checksum, cache inventory and GPU
state. Captures additionally retain the ordered kernel ledger, family, raw
count, busy-interval union, additive duration sum, exposed device gap,
stream/context inventory, memory-operation inventory and NCU target evidence.

The empirical decode bracket and signed endpoint errors are unchanged:

```text
N_sgl_low  = minimum raw kernel count over the 5 valid decode ranges
N_sgl_high = maximum raw kernel count over the 5 valid decode ranges
lower_signed_error = N_sgl_low  - 440
upper_signed_error = N_sgl_high - 567
```

`kernel_busy_union` is wall time. `kernel_duration_sum` is additive work and
may exceed wall time. Neither is queue wait.

## Physical sanity before observations

The A100 references remain 312 TFLOP/s BF16 and 2.039 TB/s HBM bandwidth:

- Selected weight plus LM-head payload is at least 855,644,160 bytes, a
  419.639 microsecond peak-HBM reference.
- Full-resident weight plus LM-head payload is 2,667,583,488 bytes, a
  1,308.280 microsecond peak-HBM reference.
- Decode K/V payload is at least 402,653,184 bytes; selected weights plus K/V
  give a 617.115 microsecond peak-HBM reference.
- Decode projection and selected-expert work is at least 3,019,898,880 FLOPs,
  a 9.679 microsecond peak-compute floor before attention and sampling.
- Prefill projection and selected-expert work is at least 386,547,056,640
  FLOPs, a 1,238.933 microsecond peak-compute floor. It is exactly 128 times
  the decode token-linear lower bound.
- Prefill has 33,024 causal query-key pairs and decode has 8,192, a ratio of
  4.03125.

A device span below its conservative compute floor voids that lane. Measured
DRAM bytes divided by peak HBM bandwidth provide an additional NCU floor.
The HBM values are cold-traffic references, not runtime ceilings. A valid
result must locate each duration against the independent compute, memory and
end-to-end plausibility references before comparing exact digits.

## Frozen relations and evidence classes

Evidence classes stay separate: source/device correspondence, five decode and
five prefill conservation instances, two span-agreement aggregates, one phase
direction relation, and native/focused/repository executable evidence. Fatal
guards are unscored. One fatal violation voids the affected evidence rather
than lowering a pass fraction.

Frozen behavioral relations are:

- Every valid decode range has a finite positive count. Counts should be
  identical after warmup; any alternate dynamic identity must be explained.
- Each captured total exactly equals the sum of semantic-family counts,
  including `unattributed`, with no lost or duplicated kernel.
- CUDA-event and enclosing device-activity spans agree within the larger of
  5 percent or 20 microseconds for each anchor.
- Host settlement is not shorter than joined device work.
- `0 <= kernel_busy_union <= device_span`, and exposed gap is exactly device
  span minus the union at exported resolution.
- Prefill median device span and busy union exceed decode. A contrary valid
  result refutes the hypothesis without voiding the run.
- The no-capture path executes the same stock workload and emits no profiler
  artifact.
- `examples/sglang_host_step_v1/results.json` remains byte-identical at
  SHA-256 `c021c55274e691fe609720045eec4441a8bb4828d248ca02b8561b63e2fddaff`.

## States, fatal guards and safety

The overall and per-lane states remain `VALID`, `BLOCKED` and `VOID`.
`BLOCKED` means exact identities and structural guards hold but a permitted
site capability is unavailable. `VOID` means a fatal precondition failed.
Timing and Nsight Systems are required for both anchors. The overall state is
`VALID` only when those four required lanes and every shared guard are valid.
Nsight Compute is explicitly optional: a blocked NCU lane remains visible but
does not change an otherwise valid overall state. A void NCU lane still voids
the run because it refutes a shared or lane-specific fatal precondition.

All v1 source, model, workload, profiler, launch-conservation, physical-floor,
GPU-state, zero-NCCL and ideal-control semantics remain fatal. V2 explicitly
supersedes the v1 CUDA 13 runtime identity, its prohibition on image
extraction and its 20 GiB scratch bound. Output confinement remains fatal under
the v2 job-local roots and 160 GiB ceiling. V2 adds these fatal guards:

- the OCI index, manifest, config, layer inventory or before/after hashes
  differ from the frozen content-addressed image;
- the clean source projection contains a blob not matching the Git tree or the
  one frozen generated version file, or any SGLang search location, imported
  source, entry point or applied hook violates its authority;
- Python, PyTorch, CUDA, SGLang kernel, Triton, Transformers, driver, GPU,
  backend or parallelism identity differs;
- a CUDA 13 distribution or shared object is installed or loaded;
- the clean-environment policy, normalized override map or immutable
  environment-manifest digest differs across a gate, timing or profiler child;
- an inherited Apptainer or Singularity control, default host mount,
  unauthorized bind, Python user site, proxy or unconfined state path enters a
  child;
- a per-child module or shared-object ledger contains an origin or hash that
  is absent from its source, OCI, host-driver or host-profiler authority;
- extraction writes outside job scratch, source inputs change, a package is
  installed or downloaded, scratch exceeds 160 GiB, retained data exceeds
  4 GiB, or cleanup fails.

Explicit counter denial or an unavailable exact NCU target yields `BLOCKED`
for the optional NCU lane only. An Nsight Systems capability limitation or an
unavailable required timing lane yields overall `BLOCKED`. The runner never
substitutes another image, runtime, source, profiler, kernel or privilege.

The job never changes clocks, power, persistence, compute mode, MIG, MPS,
driver policy or another user's process. It uses no elevated privilege.
Optional profiler clock control stays disabled. Protected GPU identity and
mode fields must match before and after; live telemetry may vary and is not
misreported as immutable.

This pilot does not claim a production A100 profile table, controlled-clock
stability, held-out accuracy, CUDA-graph behavior, multi-GPU or fabric
behavior, scheduler semantics, Engine/HTTP coverage, TTFT or TPOT. A valid v2
launch bracket applies only to this exact CUDA 12.9 runtime and workload. It
does not transfer to CUDA 13 or another wheel and image set.
