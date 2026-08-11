# VLLM-16 GPU-invisible smoke expectations

This additive specification freezes the VLLM-16 isolation study before its
result-producing harness or any of the three registered isolation attempts.
It does not edit or reinterpret the frozen VLLM-13 study. A successful
skeleton smoke is not sufficient by itself: the process must also prove that
the physical GPU is undiscoverable before vLLM import, during worker
construction, and after generation.

## Audited source and host state

The repository source is commit `832442f748ff6b3c74ab55251c0186afe7686424`.
The external runtime is vLLM 0.26.0 with Torch 2.11.0+cu130. The following
sources were read before this freeze:

- `vllm/platforms/__init__.py:59-107` selects the CUDA platform from the NVML
  device count and the absence of a CPU tag in the vLLM package version. It
  does not consult `CUDA_VISIBLE_DEVICES` in this decision.
  `vllm/platforms/__init__.py:235-280` selects `UnspecifiedPlatform` when no
  plugin activates and lazily retains the selected platform. The file hash is
  `a2bd800acc39b3215ccb78808d43317b351f137072b03e7f0f0ab3d069d91521`.
- `vllm/platforms/interface.py:1293-1295` gives `UnspecifiedPlatform` an empty
  device type. `vllm/config/device.py:49-78` rejects an automatically selected
  platform with no device type. The device-config hash is
  `7b82eee02ceb5842337451a27a3d5729920c47e25e8f6bf3997f5146f9330a9c`.
- `vllm/platforms/cpu.py:130-178,219-239` configures the CPU executor and
  requires a CPU device type. Its hash is
  `067f92d391b1c131e12a7ba9631921e4b9dd57d3c55b1d8724e9963e2fdc9c7d`.
- `vllm/v1/worker/worker_base.py:245-259,317-320` resolves and constructs the
  dotted worker class only after platform and engine configuration succeed.
  Its hash is
  `7da44338c2645ebf03d23394e452b31a8e3da1011fd1b42fcfcccfe99551b3fe`.
- `simllm/adapters/vllm/worker.py:570-602,691-738` enforces the skeleton flag,
  delegates only the generic stock-worker constructor, leaves physical device
  state unset in `init_device`, and constructs `SimModelRunner`. Its hash is
  `07e2d26213a1899aaf2604787cd85f47a67731d660b94fd473943831e7bccd2e`.
- `examples/vllm_skeleton_v1/live_smoke.py:40-87` is the already accepted
  strengthened smoke boundary: dotted `SimWorker`, one request, two exact
  fabricated tokens, `SimModelRunner`, two records, and schema
  `atlahs-closed-loop-step-v1`. Its hash is
  `a43d5e6987b0322bc0a6d05d3b7046de84980f7cfd3600eb4b94b8a7d56782cc`.
- The bubblewrap 0.4.1 manual at lines 70-121 defines the namespace options,
  lines 194-261 state that mount operations apply in argument order, and
  `--dev` mounts a new device filesystem. The installed manual and executable
  hashes are respectively
  `5ea76295cb43a8f93e8a51814396585649dbebe998844c4d6932c3dc99697ccf`
  and `a87328fd969d4bc9fbc62e56b15a393b2b23c7b47aa092a3ac02955a68da19e4`.

The audited host exposes one NVIDIA GeForce GTX 1660 Ti with UUID
`GPU-a90a812a-41bf-4f2f-c96d-d83e6eae6bd0` and driver 550.90.07. Before any
isolation attempt, NVML and Torch each report one device, Torch reports CUDA
available, allocated CUDA memory is zero, and five NVIDIA character nodes are
present. The invoking user can open those nodes through the video group. The
v1 devices cgroup hierarchy is root-owned and not writable by the invoking
user, so the registered namespace mechanism uses bubblewrap rather than
claiming an unavailable direct cgroup edit.

These observations are the pre-run host audit. None is an isolation attempt
or a skeleton smoke.

## Frozen smoke and invisibility gates

Every mechanism receives the same cached Granite snapshot, offline controls,
in-process vLLM 0.26.0 execution, dotted
`simllm.adapters.vllm.SimWorker`, disabled V1 multiprocessing, the V1 runner,
the exact skeleton worker flag, one prompt, two requested output tokens, and a
fresh mechanism-local step-record path.

A mechanism is genuinely GPU-invisible only if all of these fatal gates hold
in the child process:

- the NVIDIA character-node count is exactly zero before vLLM import and
  remains zero after the smoke;
- direct NVML initialization is unavailable, or it succeeds and reports
  exactly zero devices, both before and after;
- `torch.cuda.is_available()` is false, `torch.cuda.device_count()` is zero,
  and CUDA allocated bytes remain exactly zero;
- vLLM does not select `CudaPlatform` at any point;
- no log line identifies a physical GPU or a CUDA device configuration.

The smoke half passes only if vLLM reaches the dotted `SimWorker`, constructs
`SimModelRunner`, leaves `worker.device` unset, generates exactly the worker's
configured fabricated token twice, and writes exactly two
`atlahs-closed-loop-step-v1` records. A pre-worker platform refusal is an
honest environmental outcome, not a smoke pass. A successful skeleton under a
process that still exposes the GPU is likewise not an invisibility pass.

VLLM-16 closes only if at least one registered mechanism passes both halves in
the same child. Outcomes from different mechanisms cannot be combined.

## Ordered one-attempt mechanisms

The driver attempts these mechanisms exactly once and in this order. It does
not retry a failed child under a weaker definition.

### M1: invalid CUDA UUID sentinel

The direct child receives
`CUDA_VISIBLE_DEVICES=GPU-00000000-0000-0000-0000-000000000000`.
The expected relation is that Torch CUDA visibility falls from one device to
zero, while NVML and the character-node count remain at their host values
because vLLM's audited platform detector uses NVML. The expected disposition
is therefore an invisibility failure even if the skeleton smoke succeeds.

### M2: device-free namespace

Bubblewrap receives `--unshare-all`, a read-only root bind, then a new device
filesystem at the process device mount, a fresh procfs, a private temporary
filesystem, and the registered child command. No host device is rebound. The
mount ordering is material: the new device filesystem follows the root bind
and replaces the inherited host device mount.

The exact isolation expectation is zero NVIDIA nodes, NVML unavailable or
zero, and Torch CUDA count zero. Because this is a CUDA-tagged vLLM package,
the expected platform is `UnspecifiedPlatform`, followed by a device-type
refusal before `SimWorker`. This mechanism is expected to pass isolation but
fail the smoke. A namespace setup refusal is recorded instead if the host
kernel or bubblewrap policy prevents construction.

### M3: forced CPU platform

The direct child replaces vLLM's lazy current platform with `CpuPlatform`
before importing `LLM` or any module that binds `current_platform`. It then
uses the same dotted skeleton worker and smoke.

The exact platform expectation is `CpuPlatform`, and the skeleton may reach
`SimWorker` without constructing `CPUWorker`. The expected physical outcome
is still one NVML device and the unchanged NVIDIA character nodes. This
mechanism therefore fails genuine invisibility even if the smoke succeeds.
The outcome must not be described as GPU-invisible merely because vLLM's
selected execution platform is CPU.

## Evidence accounting

There are three scored live isolation-and-smoke rows, one per mechanism. A row
passes only if its invisibility gates and exact smoke assertions both pass.
All three are genuine-risk observations, so the genuine-risk fraction is
`3/3 = 100%` regardless of how many pass. If none passes, the task remains
open and the report names the closest reached boundary and exact missing host
or runtime capability.

The ordered-attempt count, source hashes, model revision, offline mode, fresh
paths, no CUDA allocation, no stock device initialization, worker device
state, exact token identity, record count, schema identity, and absence of
cross-mechanism evidence mixing are fatal unscored guards. Host baseline,
mechanism definitions, and the two-token request are run configuration, not
behavioral passes. Logs and blocker records are diagnostic evidence and do not
increase the denominator.

This validation is a platform/component boundary and has no TTFT or TPOT
claim. Its successor is the GPU-present VLLM-13 work, whose metric-live timing
requires the separate completion-authority tasks already named in the owning
registry.

## Registered command and pre-freeze dry run

Source local configuration first. The single registered invocation is:

```text
"${SIMLLM_VLLM_PYTHON:?configure SIMLLM_VLLM_PYTHON}" \
  examples/vllm_skeleton_v1/vllm16_smoke.py \
  --cache-dir "${HF_HOME:?configure HF_HOME}" \
  --vllm-package-root "${SIMLLM_VLLM_PACKAGE_ROOT:?configure SIMLLM_VLLM_PACKAGE_ROOT}" \
  --bwrap "${SIMLLM_BWRAP:?configure SIMLLM_BWRAP}" \
  --run-dir "${SIMLLM_VLLM16_RUN_ROOT:?configure SIMLLM_VLLM16_RUN_ROOT}"
```

Before this freeze, the same command with `--check-only` was run against an
untracked parser and literal-audit harness. It checked the complete option
surface, model snapshot, external and repository source hashes, vLLM version,
bubblewrap version and executable identity, mechanism order, invalid UUID
sentinel, evidence denominator, and exact smoke literals. It printed one
confirmation line by design. It imported no SimLLM target module, constructed
no namespace, platform, engine, worker, or model, attempted no isolation
mechanism, and produced no artifacts. The untracked harness encoded only
literals frozen in this document.

Runtime JSON, JSONL, logs, and model artifacts stay below the configured
external run directory. The result report and owning registry update are the
only later tracked additions to the existing VLLM-13 example directory.

## Deliberate omissions

This study does not edit the frozen VLLM-13 expectations, result report,
tracked CSV, deterministic harness, or historical live smoke. It does not add
a production platform plugin, weaken vLLM platform checks, modify cgroups,
allocate a GPU, run a real model, validate GPU-present rebound mode, or claim
TTFT or TPOT. It records the environment honestly if all three mechanisms
fail the joint invisibility-and-smoke bar.
