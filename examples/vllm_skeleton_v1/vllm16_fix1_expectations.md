# VLLM-16 combined-isolation fix-round expectations

This is a labeled post-specified review study. The original three-row
expectations in `vllm16_expectations.md` remain frozen and retain their 0/3
interpretation. This additive expectation freezes one previously untried
composition before its implementation and only result-producing invocation:
the exact M2 device namespace and the exact M3 CPU-platform override in the
same child.

## Audited basis

The repository source is commit `fdc59619cf739f5796fdb5f47f0f701cc878599f`.
The external runtime remains vLLM 0.26.0 with Torch 2.11.0+cu130, the pinned
Granite snapshot, and bubblewrap 0.4.1. The following sources were re-read
before this freeze:

- `vllm/platforms/__init__.py:59-107` shows that automatic CUDA selection uses
  NVML, while `vllm/platforms/__init__.py:254-280` shows that assigning the
  lazy platform before its first resolution controls `current_platform`. Its
  SHA-256 is
  `a2bd800acc39b3215ccb78808d43317b351f137072b03e7f0f0ab3d069d91521`.
- `vllm/config/device.py:49-78` derives the device type from
  `current_platform` and leaves host-handled CPU device state unset. Its
  SHA-256 is
  `7b82eee02ceb5842337451a27a3d5729920c47e25e8f6bf3997f5146f9330a9c`.
- `vllm/platforms/cpu.py:142-166,219-238` retains a caller-supplied worker
  class, requires the CPU device type, and disables CUDA-oriented runtime
  settings. Its SHA-256 is
  `067f92d391b1c131e12a7ba9631921e4b9dd57d3c55b1d8724e9963e2fdc9c7d`.
- `examples/vllm_skeleton_v1/vllm16_smoke.py:140-251` contains the already run
  CPU override and exact skeleton assertions. Lines 306-339 contain the
  already run namespace construction. This fix composes those two operations
  without weakening either gate.

The bubblewrap executable SHA-256 remains
`a87328fd969d4bc9fbc62e56b15a393b2b23c7b47aa092a3ac02955a68da19e4`.
The worker and accepted live-smoke source hashes remain respectively
`07e2d26213a1899aaf2604787cd85f47a67731d660b94fd473943831e7bccd2e`
and `a43d5e6987b0322bc0a6d05d3b7046de84980f7cfd3600eb4b94b8a7d56782cc`.

## One scored composition

The driver attempts `device-namespace-cpu-platform` exactly once. Bubblewrap
receives the frozen M2 sequence: an unshared namespace, read-only root bind,
fresh device filesystem, fresh procfs and temporary filesystem, and no host
device rebind. Inside that child, before importing `LLM`, the driver assigns
the same `CpuPlatform` instance used by M3. It does not run either constituent
alone again.

The expected direction is constructive: M2 supplies physical invisibility
and M3 supplies a valid CPU device type, so the combined row is expected to
pass both halves. Interaction between namespace construction, platform
initialization, imported extensions, and worker construction remains a live
risk, which is why the composed row is executed rather than inferred.

The invisibility half passes only if both the pre-import and post-smoke probes
show zero NVIDIA character devices, NVML unavailable or reporting zero
devices, Torch CUDA unavailable with device count and allocated bytes both
zero, no CUDA platform, and no GPU-bearing log marker. Each probe also records
every matching device entry's `lstat` type and mode. A path named
`nvidia-caps` passes the character-device gate only when that preserved
evidence identifies it as a directory or another non-character type.

The smoke half retains the original exact gates: `CpuPlatform`, reached
`SimWorker`, `SimModelRunner`, unset worker device state, one output whose two
token IDs equal the configured fabricated token, and exactly two
`atlahs-closed-loop-step-v1` records. Both halves must pass in this one child.
The scored headline is therefore either 1/1 or 0/1. This row is genuine-risk,
so the genuine-risk fraction is `1/1 = 100%` in either outcome.

A pass closes VLLM-16 on this host and disproves the published claim that a
CPU-tagged vLLM build with matching CPU Torch is necessary. A failure does not
establish that build as necessary. It leaves the exact observed boundary and
the untried CPU-tagged build, a `VLLM_TARGET_DEVICE=cpu` source build against
the current Torch installation, and any supported non-CUDA platform plugin as
hypotheses only.

This component validation has no TTFT or TPOT claim. Its successor remains
the GPU-present VLLM-13 timing path and its separately registered completion
authority work.

## Registered command and pre-freeze dry run

Source local configuration first. The only result-producing command is:

```text
"${SIMLLM_VLLM_PYTHON:?configure SIMLLM_VLLM_PYTHON}" \
  examples/vllm_skeleton_v1/vllm16_fix1.py \
  --cache-dir "${HF_HOME:?configure HF_HOME}" \
  --vllm-package-root "${SIMLLM_VLLM_PACKAGE_ROOT:?configure SIMLLM_VLLM_PACKAGE_ROOT}" \
  --bwrap "${SIMLLM_BWRAP:?configure SIMLLM_BWRAP}" \
  --run-dir "${SIMLLM_VLLM16_FIX1_RUN_ROOT:?configure SIMLLM_VLLM16_FIX1_RUN_ROOT}"
```

Before this freeze, the same command with `--check-only` was run against the
untracked parser and literal-audit harness. It checked the full option
surface, pinned model, runtime versions, external and repository hashes,
bubblewrap identity, one-mechanism identity, output count, schema, and 1/1
denominator. It printed one confirmation line by design and produced no
artifacts. It imported no SimLLM target module and constructed no namespace,
platform, engine, worker, or model. The untracked harness encoded only the
literals frozen here.

## Deliberate omissions

This study does not edit or rerun the original three rows, change the frozen
M2 namespace or M3 override, build vLLM, install CPU Torch, add a platform
plugin, allocate a GPU, load model weights, or weaken an invisibility or smoke
gate. The single composition is intentionally not a parameter sweep: the
review question is whether these two already observed halves coexist in one
process, and one attempt preserves the requested mechanism accounting.
