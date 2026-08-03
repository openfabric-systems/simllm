# simllm.compute

Pluggable compute-time providers plus the host initiation model. The core
needs one number per GOAL `calc` node: how long a rank computes before it
hands data over. Providers answer at different fidelity/cost points; the
step loop always reads tables or analytical estimates, never a cycle-level
simulator.

## Interface

- `ComputeProvider.estimate(kernel: KernelSpec, gpu: GpuSpec) -> DurationEstimate`
- `ProfileTableProvider`: measured (kernel, config, GPU) duration tables from
  real captures; exact-match lookups.
- `RooflineProvider`: analytical `max(flops/peak, bytes/bw)` with an
  efficiency derate; classifies compute- vs memory-bound from the kernel
  configuration alone.
- `HostInitiationModel`: constant per-operation delay between "ready" and
  "on the wire" (default 0, profile-labeled). The doorbell packet itself is
  modeled in-band on the fabric; host/PCIe/RNIC launch effects default to
  zero delay and zero jitter so network attribution stays clean.

Every estimate carries an honest uncertainty so results can report error
bounds.

## Status

Both providers and the host model are implemented and tested. The offline
SASS-level provider is design-only.

## Open tasks

- COMP-1: offline SASS-level table generator (Accel-Sim / GPGPU-Sim) that
  populates `ProfileTableProvider` tables for unmeasured configurations
  (milestone M5).
- COMP-2: calibrated host-initiation profiles (GPU-initiated vs CPU-proxy
  constants) for launch-path sensitivity studies.
- COMP-3: interpolation across uncovered configs in `ProfileTableProvider`.
