# SGLang A100 kernel pilot v1 expectations

## Freeze scope and chronology

This is the expectations-only record for the first production-kernel pilot on
one Merlin A100. It is committed before the pilot harness, any SGLang import
or CUDA initialization in this study, any profiler invocation, and any
result-producing run. No observation from this study may be used to edit this
file afterward.

The study starts at the GPU-kernel boundary. It exercises the pinned SGLang
`ModelRunner` and its real Granite weights with fixed eager prefill and decode
shapes. A later study will move the same evidence boundary through the stock
Engine and then through the server path. This pilot does not report TTFT or
TPOT.

Static source inspection and package-metadata inspection occurred before this
freeze. No SGLang model was loaded, no Python package was imported, no CUDA
context was created, and no timed or profiled SGLang step was observed during
that inspection.

## Task ownership and closure boundary

SGL-24 owns the immediate result. Its current surrogate is the vLLM 0.26.0
device-launch bracket `[440, 567]`. The pilot must produce a SGLang-specific
device-visible launch bracket for one fixed decode geometry, report its signed
distance from both transferred endpoints, and prove that the existing ideal
path is unchanged.

The same capture may advance other tasks without closing them:

- COMP-5 and COMP-1 still require controlled clocks, the full registered
  matrix, dynamic SASS, pinned Accel-Sim replay, immutable train and holdout
  cells, stability below the registered ceiling, and held-out error checks.
- COMP-6 still requires per-invocation shapes to become usable trace and table
  keys. Capturing those shapes alone does not close it.
- SGL-10 still requires an identity-keyed `ExecutionGraph` template and replay
  with stream, event, kernel, NCCL, shape, radix and dependency binding.
- SGL-4 still requires paced live wall-clock TTFT and TPOT evidence, including
  the Engine and HTTP paths and the larger workload matrix.

No task other than SGL-24 may close from this pilot. SGL-24 closes only if its
decode lane is `VALID`, the measured bracket and signed transfer errors are
published, every launch is conserved in the device ledger, and the ideal
artifact remains byte-identical.

## Pre-freeze source audit

The exact SGLang checkout is clean at commit
`8f2a3ad6d7d68c58ae65b61a75bb2115449addca` and tree
`5be26db1f559064c0f9e724e78c1a8f619754867`. The relevant source path is:

```text
ModelRunner.forward
  -> ModelRunner._forward_raw
  -> EagerRunner.execute
  -> EagerRunner._execute_extend or EagerRunner._execute_decode
  -> GraniteMoeForCausalLM.forward
  -> LogitsProcessor
  -> ModelRunner.sample
```

The study calls the pinned `sglang.benchmark.one_batch` `extend` and `decode`
helpers around the stock `ModelRunner`. Those helpers construct a real
`ForwardBatch`, invoke `ModelRunner.forward`, and sample the next token. The
SimLLM SGLang worker and plugin are not selected.

The source fixes the following semantic invocation ledger for either anchor.
The two anchors have the same call multiplicities but different tensor and
attention shapes.

| Semantic site | Multiplicity per step | Source contract |
|---|---:|---|
| token embedding | 1 | `GraniteMoeModel.get_input_embeddings` |
| embedding scalar multiply | 1 | `GraniteMoeModel.forward` |
| decoder layer | 24 | `GraniteMoeModel.layers` |
| layer RMSNorm | 48 | input and post-attention norm in each layer |
| QKV projection | 24 | `GraniteMoeAttention.qkv_proj` |
| Q/K/V split views | 72 | metadata views, no device-launch count promised |
| rotary embedding | 24 | `GraniteMoeAttention.rotary_emb` |
| Triton radix attention | 24 | extend or decode backend path plus KV update |
| output projection | 24 | `GraniteMoeAttention.o_proj` |
| scaled residual expression | 48 | two expressions in each layer |
| router projection | 24 | `GraniteMoeMoE.gate` |
| top-k selection | 24 | `GraniteMoeMoE.topk` |
| fused MoE dispatch, experts and combine | 24 | `GraniteMoeMoE.experts` |
| final RMSNorm | 1 | `GraniteMoeModel.norm` |
| logits processor and LM head | 1 each | `GraniteMoeForCausalLM` |
| greedy sample | 1 | `ModelRunner.sample`, outside the model span |

This is an invocation ledger, not a fabricated CUDA-kernel count. ATen,
cuBLAS and Triton lower one semantic call to zero, one or many launches based
on shape, algorithm, JIT and backend branches. Static Python inspection cannot
honestly predict the raw count. The study therefore freezes the semantic
ledger before execution and identifies the raw bracket with CUPTI-backed
Nsight Systems. Every observed kernel must map to one of these families or to
an explicit `unattributed` family, and all family counts must sum exactly to
the captured total.

## Frozen identity envelope

| Identity | Frozen value |
|---|---|
| SimLLM base | `64b35512156bd589427c0f9bc2713df7d6088bdc` |
| SGLang commit | `8f2a3ad6d7d68c58ae65b61a75bb2115449addca` |
| SGLang tree | `5be26db1f559064c0f9e724e78c1a8f619754867` |
| SGLang package | `0.0.0.dev1+g8f2a3ad6d` |
| SGLang kernel package | `0.4.5` |
| model | `ibm-granite/granite-3.0-1b-a400m-instruct` |
| model revision | `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445` |
| config SHA-256 | `ca4bb3a5c1bdef988ab413e0d731640446da65316e4ed16de3666cd96ecc3a0b` |
| weight object SHA-256 | `f7ae1cee56a9ea6c5360437b1c0407f8d84816b2cc75470f4e7e5236fa2a07dc` |
| weight file size | 2,669,283,096 bytes |
| Python | 3.12.9 |
| PyTorch | `2.11.0+cu130` |
| PyTorch CUDA build | 13.0 |
| Triton | 3.6.0 |
| Transformers | 5.12.1 |
| profiler module | `cuda/12.9.1` |
| Nsight Systems | 2025.1.3 |
| Nsight Compute | 2025.2.1 |
| expected driver | 565.57.01 |
| GPU | `NVIDIA A100-SXM4-80GB`, compute capability 8.0 |
| execution | BF16, unquantized, eager |
| parallelism | TP=EP=PP=DP=1 |
| attention backend | `triton` |
| MoE runner backend | `triton` |
| MoE A2A backend | `none` |
| sampling backend | `pytorch` |

The runtime and profiler CUDA versions are deliberately not assumed
compatible. Failure to profile the CUDA 13.0 PyTorch runtime with the frozen
Nsight tools is `BLOCKED`. The run must not silently switch to the available
SGLang 0.5.17 container, another CUDA build, another SGLang checkout, or
another profiler.

Machine-specific locations are supplied through
`SIMLLM_SGLANG_SOURCE`, `SIMLLM_SGLANG_ENV`, `SIMLLM_MODEL_SNAPSHOT`,
`SIMLLM_RUN_ROOT` and `SIMLLM_SCRATCH_ROOT`. They are never embedded in the
tracked study.

## Frozen allocation and storage envelope

The pilot uses one batch job:

| Resource | Frozen request |
|---|---:|
| cluster | `gmerlin7` |
| account | `merlin` |
| partition | `a100-hourly` |
| nodes and tasks | 1 node, 1 task |
| GPU | 1 `nvidia_a100-sxm4-80gb` |
| CPU | 8 CPUs for the task |
| host memory | 64 GiB |
| wall limit | 45 minutes |
| expected runtime | 15 to 30 minutes |
| internal deadline | 40 minutes |
| scratch soft target | below 16 GiB |
| scratch hard ceiling | 20 GiB |
| retained compact result ceiling | 4 GiB |

The job is nonexclusive and has no array or fanout. Model load, imports, JIT,
CUDA initialization, timing, profiling, report export and parsing run only in
the allocation. Login-host work is limited to small source and metadata reads,
hash verification, staging, `sbatch`, sparse scheduler queries and small log
reads.

The job is offline. It performs no download, install, container pull, image
extraction or dependency build. Triton JIT may populate a job-local cache
during warmup on the compute node. Missing source, model, runtime or tool
inputs yield `BLOCKED` rather than a network access.

## Frozen model and workload

The model geometry is 24 layers, hidden size 1,024, 16 attention heads, 8 KV
heads, head dimension 64, 32 experts per layer, top-k 8, expert width 512 and
vocabulary size 49,155. Weights and KV cache are BF16. Quantization, CUDA
graphs, overlap scheduling, radix reuse, chunked prefill and speculative
decoding are disabled. Page size is 1 and memory fraction is 0.60.

Token IDs are generated without a tokenizer:

```text
token_id(request, position) = 1 + ((173 + 257*request + 31*position) mod 49154)
```

This keeps every token in `[1, 49154]` and excludes EOS token 0. Request IDs
are `<phase>-r0` through `<phase>-r3`. Sampling is greedy with temperature 0,
`ignore_eos=true`, and no grammar or logprobs.

| Anchor | Frozen construction | Required runtime assertion |
|---|---|---|
| `prefill-t512-r4` | four 128-token requests, one generated token | `extend_num_tokens == 512`, four requests |
| `decode-b4-c2048` | four 2,047-token requests, prefill outside the measured range, then one decode | `seq_lens_cpu == [2048, 2048, 2048, 2048]` immediately after `prepare_for_decode` |

The decode context-building prefill is never part of the decode span or decode
launch count. The prefill lane brackets `extend`; the decode lane brackets
exactly one `decode`. Both boundaries include SGLang's logits processing and
greedy sample.

Before each retained instance, request and KV pools are cleared and rebuilt.
JIT and autotuning state remain warm within the phase process. There are 10
shape-identical warmups, 41 retained CUDA-event timing repetitions and 5
separate Nsight Systems capture ranges per anchor. The Triton cache inventory
must not change during any retained or captured step.

## Frozen instrumentation

The harness adds observation-only NVTX ranges around each complete anchor and
around layer 0 QKV projection and layer 0 fused MoE. It uses module hooks only
to push and pop those ranges. It does not replace a module, tensor, kernel,
stream, allocator, batch or sample.

The Nsight Systems command is fixed to the equivalent of:

```text
nsys profile
  --trace=cuda,nvtx
  --sample=none
  --cpuctxsw=none
  --capture-range=cudaProfilerApi
  --capture-range-end=repeat:5
  --cuda-event-trace=false
  --target-processes=all
  --force-overwrite=true
```

Each anchor executes in a separate Nsight Systems process. The harness calls
`cudaProfilerStart` immediately before, and `cudaProfilerStop` immediately
after, each synchronized target step. The explicit event-tracing option is
mandatory because the qualification run showed that device-side event tracing
can add overhead or false cross-stream dependencies.

The result retains the CUDA GPU activity table, CUDA API table, NVTX GPU
projection, report identity and compact logs. It records function name,
correlation identity, device, context, stream, start, duration, grid, block,
registers and shared memory wherever the tool exposes them. Memcpy and memset
rows are separate from kernel launches.

Nsight Compute runs in a separate replay process. For each anchor, the target
is the first kernel launched inside the layer 0 fused-MoE NVTX range. The
kernel name and number of earlier same-name launches are derived mechanically
from Nsight Systems, never from duration or counter value. The replay uses one
fully anchored escaped demangled-name regex, the matching `--launch-skip`,
`--launch-count 1`, `--replay-mode kernel`, `--clock-control none`,
`--profile-from-start off`, `--target-processes all`, and the bounded `basic`
set. A name that cannot be resolved to that semantic target is `BLOCKED`.

Each command has a five-minute timeout. Each phase step has a two-minute
timeout. A global 40-minute deadline stops the study cleanly.

## Frozen observables

For every retained repetition the result records:

- host entry-to-return span;
- CUDA-event first-work-to-joined-frontier span;
- phase, batch shape, sequence lengths and scheduled token count;
- output token IDs and a deterministic output checksum;
- GPU state and cache inventory immediately around the measurement.

For every captured repetition the result additionally records:

- raw device kernel launch count;
- ordered kernel ledger and semantic family;
- kernel-busy interval union;
- additive kernel-duration sum;
- exposed device gap, exactly device span minus activity union;
- stream and context inventory;
- memcpy and memset inventories;
- NCU target identity and finite basic metrics when the lane is available.

`kernel_busy_union` is a wall-time union. `kernel_duration_sum` is additive
work accounting and may exceed wall time when streams overlap. Neither is
reported as queue wait.

The empirical SGLang decode bracket is:

```text
N_sgl_low  = minimum raw kernel count over the 5 valid decode ranges
N_sgl_high = maximum raw kernel count over the 5 valid decode ranges
```

The signed transferred-endpoint errors are:

```text
lower_signed_error = N_sgl_low  - 440
upper_signed_error = N_sgl_high - 567
```

Positive means the transferred vLLM endpoint undercounts SGLang. Negative
means it overcounts SGLang.

## Physical sanity before observations

The physical review uses the real BF16 Granite geometry and the repository's
A100 envelope of 312 TFLOP/s and 2.039 TB/s. These values are frozen before a
step duration is read.

- The minimum selected weight and LM-head payload is 855,644,160 bytes. Its
  zero-cache peak-HBM serialization reference is 419.639 microseconds.
- The full-resident weight and LM-head reference is 2,667,583,488 bytes. Its
  peak-HBM serialization time is 1,308.280 microseconds. This is a reference,
  not an upper runtime bound.
- Decode consumes at least 402,653,184 bytes of BF16 K/V payload for four
  2,048-token contexts. Selected weights plus this payload give a 617.115
  microsecond zero-cache peak-HBM reference.
- Decode projection and selected-expert work is at least 3,019,898,880 FLOPs,
  a 9.679 microsecond peak-compute floor before attention and sampling.
- Prefill projection and selected-expert work is at least
  386,547,056,640 FLOPs, a 1,238.933 microsecond peak-compute floor before
  attention and sampling. It is exactly 128 times the decode token-linear
  lower bound.
- Prefill has 33,024 causal query-key pairs and decode has 8,192, a ratio of
  4.03125.

A device span below the conservative peak-compute floor voids the lane. For
an NCU target, measured DRAM bytes divided by peak HBM bandwidth are an
additional hard floor. The whole-step HBM figures above are cold-traffic
references because the profiler has not yet measured cache service or DRAM
bytes. There is no honest physical runtime ceiling without assuming a minimum
sustained rate. The two-minute per-step timeout is therefore an operational
ceiling: reaching it yields `BLOCKED`, not a fabricated slow measurement. A
valid result must explain where its measured value sits relative to the
selected-weight, full-resident, K/V and compute references.

## Frozen evidence classes and relations

Evidence classes remain separate and are never summed into one score:

1. Source and device launch correspondence: one aggregate decode relation.
2. Repeated device-ledger conservation: five decode and five prefill
   instances.
3. Cross-boundary span agreement: two anchor aggregates.
4. Phase direction: one aggregate relation.
5. Native profiler execution, focused tests and repository tests: separate
   executable evidence.

Provenance, workload shape, artifact hashes, source-ledger inventory, output
confinement, physical floors, GPU identity, zero foreign processes, disabled
MIG, acyclic ordering, zero NCCL at TP=EP=PP=1 and exact compatibility are
fatal and unscored.

Frozen behavioral relations:

- Every valid decode range has a finite positive kernel count. The five counts
  should be identical after warmup. If they differ, every alternate dynamic
  kernel identity must be explained before the min/max bracket is accepted.
- Each captured total equals exactly the sum of its semantic-family counts,
  including `unattributed`.
- Each kernel appears once in the ordered ledger. No kernel may be lost or
  duplicated across nested NVTX projections.
- The CUDA-event span and the enclosing Nsight Systems device activity span
  agree within the larger of 5 percent or 20 microseconds for each anchor.
- Host settlement is not shorter than the joined device span.
- `0 <= kernel_busy_union <= device_span` and exposed gap equals
  `device_span - kernel_busy_union` exactly at the exported timestamp
  resolution.
- Prefill median device span and median kernel-busy union exceed decode. A
  contrary valid result is retained as a refuted hypothesis, not made void.
- The no-capture path executes the same stock workload and emits no profiler
  artifact.
- The tracked ideal artifact
  `examples/sglang_host_step_v1/results.json` remains byte-identical at
  SHA-256 `c021c55274e691fe609720045eec4441a8bb4828d248ca02b8561b63e2fddaff`.

## Decision states and fatal guards

The study produces one overall state and one state per profiler lane:

- `VALID`: every fatal guard holds and the required ledger is interpretable.
- `BLOCKED`: identities and structural guards hold, but the frozen runtime and
  toolchain are incompatible or a permitted site capability is unavailable.
- `VOID`: a fatal guard fails, so the affected evidence cannot be interpreted.

A refuted nonfatal hypothesis remains a valid finding. It is never converted
into a pass by editing these expectations.

The following failures void the affected lane, and a shared-identity failure
voids the whole run:

- SimLLM, SGLang, model, config, weight object, package, GPU, driver, profiler,
  dtype, backend, parallelism, allocation or workload identity differs;
- zero or more than one GPU is visible, the target is not the frozen A100,
  MIG is enabled, a foreign process uses the GPU, or CUDA and `nvidia-smi`
  disagree on identity;
- the source tree is dirty, the model weight hash differs, a dependency is
  downloaded or installed, or a profiler silently falls back;
- a phase has the wrong request count, token count, sequence length, forward
  mode, output cardinality or pool-reset behavior;
- retained work observes a Triton cache mutation, compilation or autotuning;
- a capture range is missing, duplicated, nested incorrectly, contains a
  context-building prefill in the decode lane, or cannot be joined to its
  CUDA API and device rows;
- a contributing stream is not joined, a kernel is lost or duplicated, a
  timestamp is nonfinite, or interval-union arithmetic fails;
- a measured duration is below a hard physical floor;
- output escapes the configured roots, a symlink or special file enters the
  retained manifest, a hash is missing, scratch exceeds 20 GiB, retained data
  exceeds 4 GiB, or the global deadline is exceeded without a clean state.

Explicit counter permission denial, inability of the frozen Nsight release to
trace the CUDA 13.0 runtime, an unavailable exact NCU target, or an unsupported
metric yields `BLOCKED`. The harness records the diagnostic and does not try a
different runtime, profiler, kernel or privilege.

## Safety, compatibility and nonclaims

The job queries GPU state but never changes clocks, power, persistence,
compute mode, MIG, MPS, driver policy or another user's process. It uses no
elevated privilege and never attempts to work around a site denial.

The harness is study-local. It does not modify a SimLLM provider, SGLang,
Granite weights or a backend submodule. `SIMLLM_SGLANG_ENABLE=0` and
`SIMLLM_SGLANG_ORACLE_CAPTURE=0` are required, and the runtime class identity
must remain the stock `ModelRunner`. The exact no-capture path and ideal
artifact are the compatibility controls.

This pilot makes no claim of:

- a calibrated A100 profile table or production SimLLM duration;
- COMP-1, COMP-5, COMP-6, SGL-10 or SGL-4 closure;
- controlled-clock stability, coefficient of variation below 2 percent, or
  the full registered production matrix;
- held-out kernel, phase or compute-only step accuracy;
- NVBit dynamic SASS, Accel-Sim compatibility or CUDA-graph behavior;
- multi-GPU, TP, EP, PP, NCCL, NVLink, RNIC or fabric behavior;
- scheduler queueing, radix hits, retraction, chunked prefill, overlap,
  Engine, HTTP, TTFT or TPOT behavior;
- device-side CUDA-event dependency evidence, because Nsight Systems event
  tracing is explicitly disabled;
- transfer to another GPU, model, SGLang commit, kernel image, backend, cache
  mode or runtime.

Configuration-forced zero NCCL is a fatal invariant but never a scored
success. A captured elapsed phase never establishes device concurrency by
itself.

## Pre-run gate

The harness must expose a `--check-only` mode that validates the frozen JSON,
formulas, command literals, identities supplied as strings, output
confinement and expected evidence inventory. It must not import SGLang or
PyTorch, initialize CUDA, create output, or run a model.

Before submission, the exact committed source bundle is reconstructed into a
clean staging tree, `--check-only` passes against it, all focused tests pass,
and the repository's `ruff check .` and `pytest -q` gates pass. The Slurm job
then verifies the staged commit and every frozen input again before importing
the runtime.
