# VLLM-16 GPU-invisible smoke results

The VLLM-16 expectations were frozen in commit `25e79be` before the
result-producing harness or any isolation attempt. Implementation commit
`1017d8a` then ran the three registered mechanisms exactly once, in frozen
order, on 2026-08-11. No mechanism passed the joint GPU-invisibility and
skeleton-smoke bar at that study boundary. The labeled fix-round row below
later closes VLLM-16 without changing the original 0/3 result.

## Headline

| Mechanism | Physical invisibility | Skeleton smoke | Joint row |
|---|---|---|---|
| Invalid CUDA UUID | fail | blocked before worker | fail |
| Device-free namespace | pass | blocked before worker | fail |
| Forced CPU platform | fail | pass | fail |

The behavioral headline is 0/3 scored live rows. All three mechanisms were
executed observations that could genuinely fail, so the genuine-risk fraction
is `3/3 = 100%`. Evidence from different rows was not combined: the namespace
isolation pass and CPU-platform smoke pass do not form one passing row.

The frozen ladder predicted that each of these three separate rungs would
fail one half while closure required one row to pass both halves. It was a
diagnostic ladder, not a closure-capable set under its own frozen predictions.
The 0/3 headline therefore confirms the frozen prediction rather than
reporting a surprising failure.

## Post-specified fix round 1 correction

The combined-row expectations were frozen in commit `9b7f854` after the
original result but before the additive implementation or attempt.
Implementation commit `08ef998` composed the exact M2 namespace with the exact
M3 `CpuPlatform` override. That mechanism ran once on 2026-08-11 and passed
1/1 scored row. Its genuine-risk fraction is `1/1 = 100%`.

| Mechanism | Physical invisibility | Skeleton smoke | Joint row |
|---|---|---|---|
| Device namespace plus CPU platform | pass | pass | pass |

The same child had no NVIDIA-named entry before vLLM import. After the smoke,
its sole matching entry was `nvidia-caps` in the process device mount. Its
preserved `lstat` evidence records `kind="directory"` and mode `0755`, not a
character device.
Both probes had zero NVIDIA character devices, unavailable NVML, zero Torch
CUDA devices, zero allocated CUDA bytes, and no CUDA platform or GPU-bearing
log marker. This re-establishes the factual premise of the earlier
post-specified M2 character-device reclassification under the same namespace
construction. The original M2 artifact and 0/3 summary remain untouched.

In that physically isolated process, vLLM selected `CpuPlatform`, reached
`SimWorker`, constructed `SimModelRunner`, left `worker.device` unset,
returned fabricated token `24577` twice, and emitted exactly two
`atlahs-closed-loop-step-v1` records. VLLM-16 is therefore complete on this
host.

## M1: invalid CUDA UUID

The invalid UUID made Torch report CUDA unavailable and zero CUDA devices, but
it left all five NVIDIA character nodes and the one-device NVML result visible.
vLLM retained `NvmlCudaPlatform`. Importing the skeleton's stock-worker base
then asked NVML to resolve the nonexistent UUID and stopped with
`NVMLError_NotFound` before `SimWorker` construction. CUDA allocated bytes
remained zero.

This is the frozen expected direction: `CUDA_VISIBLE_DEVICES` changed Torch's
logical view but did not make the process physically GPU-invisible.

## M2: device-free namespace

Bubblewrap successfully created the namespace. Before vLLM import, the child
had zero NVIDIA character nodes, NVML returned `Driver Not Loaded`, Torch
reported no CUDA device, and allocated bytes were zero. vLLM selected
`UnspecifiedPlatform`, then stopped in `vllm/config/device.py:78` with
`RuntimeError: Device string must not be empty`. The dotted worker was never
resolved or constructed.

The raw evaluator conservatively counted every post-probe name matching
`nvidia*` as a device and therefore marked the post-smoke probe false when a
path named `nvidia-caps` appeared. That path is the NVIDIA capability
directory, not a character node. The frozen gate explicitly counts character
nodes. The final harness now uses the filesystem node type, and this report
applies that post-specified structural correction to the preserved raw row.
NVML remained unavailable, Torch remained at zero devices, no CUDA memory was
allocated, and no GPU or CUDA platform appeared in the log. The correction
does not turn the scored row into a pass because the smoke still failed before
`SimWorker`. At the original 0/3 boundary, no isolation mechanism had been
rerun.

This was the closest valid boundary: genuine process isolation succeeded, but
the CUDA-tagged vLLM package had no usable non-CUDA platform after the GPU was
removed.

## M3: forced CPU platform

The override selected `CpuPlatform` before `LLM` import and vLLM recorded
`device_config=cpu`. It reached `SimWorker`, constructed `SimModelRunner` with
`worker.device` unset, returned fabricated token `24577` twice, and emitted
exactly two `atlahs-closed-loop-step-v1` records. No CUDA memory was allocated.

The physical visibility gates nevertheless failed before and after the smoke:
NVML still reported the GTX 1660 Ti and the five NVIDIA character nodes
remained accessible. Selecting a CPU platform did not make the process
GPU-invisible, so this successful skeleton execution is not VLLM-16 closure.

## Post-specified correction to host and runtime disposition

The earlier claim that a CPU-tagged vLLM build with matching CPU Torch was the
exact remaining requirement is withdrawn. The original three rows established
only that each constituent failed separately; they did not establish
necessity. The combined result disproves that claim: the existing CUDA-tagged
vLLM and Torch installation closes the task when the M2 namespace is composed
with the M3 platform override.

The attempted mechanisms are now the original invalid UUID, device namespace,
and forced CPU platform rows, followed by one additive namespace plus CPU
platform row. The following combinations remain explicitly untried, but none
is required for VLLM-16 closure:

- invalid UUID plus the CPU-platform override, which would retain the
  constituent UUID path's visible NVML device and character nodes;
- a `VLLM_TARGET_DEVICE=cpu` source build against the existing Torch
  installation, inside or outside the namespace;
- a CPU-tagged vLLM build with matching CPU Torch inside the namespace;
- an out-of-tree supported non-CUDA platform plugin inside the namespace.

Before the combined attempt, these were hypotheses rather than established
requirements. The passing row removes the residual entirely.

## Chronology and artifacts

The registered command printed its input confirmation by design, then ran
`invalid-uuid`, `device-namespace`, and `cpu-platform` once each. Each child
wrote one diagnostic JSON object and log below the configured
`SIMLLM_VLLM16_RUN_ROOT`. The CPU-platform child alone wrote a two-row step
stream. No model weights were loaded and no physical GPU allocation occurred.

The preserved raw `summary.json` SHA-256 is
`9bf7aa66ae42ed98cc4b14f2d8980347f3e36425841973fca2b834f075a96b09`.
Its three log hashes are:

- invalid UUID:
  `47c684d1ab07d49501797d9175f73b6a216e6fa4545c74d1884a12cabda332b3`;
- device namespace:
  `ae1777181e4e2a598514ad3cbd2c7e2e03ac3422e93e1628eabb01355d86232a`;
- CPU platform:
  `a3172bcb672fb02b444ecb7779767359a9f99e87856b9a4d03788006783a0d68`.

The post-specified command printed its check-only confirmation by design,
then produced one combined attempt directory below the configured
`SIMLLM_VLLM16_FIX1_RUN_ROOT`. It did not rerun any original row. The fix-round
hashes are:

- summary:
  `ec351407024a966f304f1a1f5b26c8343322c4f764b4144c3e2e85391616f7c1`;
- attempt JSON:
  `b98f79d4bb1592ad57a37ab754eb6c0e11521c836caa100036c39f8ad4f4f2bf`;
- attempt log:
  `da966121bab3424e4041f5c05521f272311ba62e4477b3a46a3a8fbcc307c6a0`;
- two-row step stream:
  `659975fca4d23951eedb817cb1e638f766638cfde6043f6891a32830acd0c936`.

The parent Torch availability probe differed from the pre-freeze audit because
Torch 2.11.0+cu130 warned that driver 550.90.07 exposed an older CUDA driver
interface during the result run. Device count, NVML count, character nodes,
and allocated bytes retained their audited physical values, so this difference
does not change any row disposition.

The existing VLLM-13 expectations, result report, deterministic script,
tracked CSV, and historical live smoke remain byte-unchanged. This study adds
no platform plugin and makes no TTFT or TPOT claim.
