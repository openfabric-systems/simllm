# GPU service model: structural validation results

The 2026-08-06 run passed all 22 pre-registered cells with zero residual.
These are mechanism checks on a synthetic 1 GHz profile. They validate the
event model's equations and boundaries, not A100 or H100 timing accuracy.

Reproduce from the repository root:

```bash
.venv/bin/python examples/gpu_service_model/run_gpu_service_model.py
MPLCONFIGDIR=/tmp/simllm-gpu-service-mpl \
  python3 examples/gpu_service_model/plot_gpu_service_model.py
```

The raw table is [results.csv](results.csv). The renderer produces
[PNG](plots/gpu_service_structural_checks.png) and
[PDF](plots/gpu_service_structural_checks.pdf) versions of the same four
sweeps.

## Exact results

| Check | Parameter A | Parameter B | replay | frozen | residual |
|---|---:|---:|---:|---:|---:|
| CTA waves | 4 CTAs | 1 SM | 1,024 cycles | 1,024 | 0 |
| CTA waves | 4 CTAs | 4 SMs | 256 cycles | 256 | 0 |
| CTA waves | 9 CTAs | 1 SM | 2,304 cycles | 2,304 | 0 |
| CTA waves | 9 CTAs | 4 SMs | 768 cycles | 768 | 0 |
| warp issue | 4 warps | 1 scheduler | 7 cycles | 7 | 0 |
| warp issue | 4 warps | 4 schedulers | 4 cycles | 4 | 0 |
| warp issue | 16 warps | 1 scheduler | 19 cycles | 19 | 0 |
| warp issue | 16 warps | 4 schedulers | 7 cycles | 7 | 0 |
| dependency chain | 8 instructions | 1 scheduler | 32 cycles | 32 | 0 |
| dependency chain | 8 instructions | 4 schedulers | 32 cycles | 32 | 0 |
| occupancy | 128 threads | 32 registers/thread | 16 CTAs/SM | 16 | 0 |
| occupancy | 128 threads | 128 registers/thread | 4 CTAs/SM | 4 | 0 |
| occupancy | 256 threads | 32 registers/thread | 8 CTAs/SM | 8 | 0 |
| occupancy | 256 threads | 128 registers/thread | 2 CTAs/SM | 2 | 0 |
| HBM | 4,096 bytes | 32 bytes/cycle | 228 cycles | 228 | 0 |
| HBM | 4,096 bytes | 64 bytes/cycle | 164 cycles | 164 | 0 |
| HBM | 8,192 bytes | 32 bytes/cycle | 356 cycles | 356 | 0 |
| HBM | 8,192 bytes | 64 bytes/cycle | 228 cycles | 228 | 0 |
| copy | 4,096 bytes | 32 bytes/cycle | 148 engine cycles | 148 | 0 |
| copy | 4,096 bytes | 64 bytes/cycle | 84 engine cycles | 84 | 0 |
| copy | 8,192 bytes | 32 bytes/cycle | 276 engine cycles | 276 | 0 |
| copy | 8,192 bytes | 64 bytes/cycle | 148 engine cycles | 148 | 0 |

The partial CTA wave remains visible: nine one-CTA waves take three rounds on
four SMs, not the 2.25-round continuous approximation. Four schedulers hide
independent issue pressure but do not shorten the eight-instruction true
dependency chain. After subtracting fixed latency or setup, the HBM and copy
service terms scale exactly with bytes and inverse bandwidth.

## Boundary and artifact checks

The focused test suite also exercises behavior that has no useful scalar plot:

- heterogeneous CTA trace classes preserve distinct edge-block work and must
  cover every linear block ID exactly once;
- per-warp register allocation, per-block thread limits, per-thread register
  limits, static/total shared-memory limits, RAW/WAW scoreboards and final
  completion drain are checked independently;
- logical requested memory bytes are distinct from physical transacted and
  serviced bytes;
- one physical copy engine can carry asymmetric per-direction calibration in
  an independent clock domain;
- unsupported barrier, cooperative, cluster and Hopper warpgroup forms fail
  rather than receiving an invented scalar latency;
- the strict `simllm-gpu-model-artifact-v1` codec freezes caller-owned
  sequences, binds timing calibration to one target architecture, records and
  validates observed core/memory clocks, normalizes hash identities, enforces
  train/held-out isolation and stream order, recomputes sample statistics, and
  reruns deterministic kernel and copy estimates;
- a validated artifact compiles into `simllm-profile-table-v1`, keeping the
  online `ComputeProvider` path an O(1) lookup.

## A100 and H100 bootstrap status

| Profile | SMs | warps/SM | registers/SM | shared memory/SM | peak HBM | timing uncertainty | seeded copy engines |
|---|---:|---:|---:|---:|---:|---:|---:|
| A100 SXM 80 GB | 108 | 64 | 65,536 | 164 KiB | 2,039 GB/s | 50% | 0 |
| H100 SXM 80 GB | 132 | 64 | 65,536 | 228 KiB | 3,350 GB/s | 50% | 0 |

Structural limits and SKU bandwidths come from NVIDIA's
[Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html),
[Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html),
[A100 data sheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf), and
[H100 product specifications](https://www.nvidia.com/en-us/data-center/h100/).
Timing context comes from the open
[Ampere study](https://arxiv.org/abs/2208.11174),
[Hopper/H800 study](https://arxiv.org/abs/2402.13499), and
[A100/H800 study](https://arxiv.org/abs/2501.12084). The numeric memory priors
are transferred from the last paper. In particular, H800 PCIe measurements are
not H100 SXM calibration. The 50 percent uncertainty and empty copy-engine
lists make that limitation machine-visible.

No silicon capture, framework-kernel coverage, TTFT or TPOT claim is made in
this study. COMP-1, COMP-5, COMP-6 and COMP-10 retain production capture,
per-invocation mapping and advanced instruction/cache semantics. CORE-4 retains
inter-operation scheduling, copy selection/queueing, overlap and shared-HBM
arbitration.
