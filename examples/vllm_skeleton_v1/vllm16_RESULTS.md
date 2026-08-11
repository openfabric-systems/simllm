# VLLM-16 GPU-invisible smoke results

The VLLM-16 expectations were frozen in commit `25e79be` before the
result-producing harness or any isolation attempt. Implementation commit
`1017d8a` then ran the three registered mechanisms exactly once, in frozen
order, on 2026-08-11. No mechanism passed the joint GPU-invisibility and
skeleton-smoke bar, so VLLM-16 remains open.

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
`SimWorker`; no isolation mechanism was rerun.

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

## Host and runtime disposition

The exact remaining requirement is a single process that combines both
properties:

- a host, container, or device namespace with zero NVIDIA character nodes,
  NVML unavailable or reporting zero devices, and Torch reporting zero CUDA
  devices before and after worker construction;
- a vLLM 0.26.0 runtime with a valid non-CUDA device type before dotted-worker
  resolution, concretely a CPU-tagged vLLM build with its matching CPU Torch
  runtime, capable of constructing the existing skeleton seam.

On that environment, the existing exact worker, runner, token, record-count,
schema, no-device-state, and zero-allocation assertions must pass together.
A GPU-visible CPU override and a GPU-free CUDA build that falls to
`UnspecifiedPlatform` remain explicit non-closing paths.

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

The parent Torch availability probe differed from the pre-freeze audit because
Torch 2.11.0+cu130 warned that driver 550.90.07 exposed an older CUDA driver
interface during the result run. Device count, NVML count, character nodes,
and allocated bytes retained their audited physical values, so this difference
does not change any row disposition.

The existing VLLM-13 expectations, result report, deterministic script,
tracked CSV, and historical live smoke remain byte-unchanged. This study adds
no platform plugin and makes no TTFT or TPOT claim.
