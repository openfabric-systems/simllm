"""Pluggable compute-time providers.

The core needs one number per GOAL ``calc`` node: how long a rank computes
before it hands data over and releases its successor. Different fidelity
levels answer that question at very different cost, so the provider is a
plugin interface (:class:`ComputeProvider`) rather than a fixed model:

============================  ================================================
Provider                      Fidelity / cost
============================  ================================================
:class:`ProfileTableProvider` Measured (kernel, config) → duration tables from
                              real captures. Cheap, accurate on covered points.
:class:`RooflineProvider`     Analytical ``max(flops/peak, bytes/bw)``:
                              classifies compute- vs memory-bound from the
                              kernel configuration alone. Cheap, coarse.
``AccelSimProvider`` (M5+)    SASS-level cycle simulation (Accel-Sim /
                              GPGPU-Sim). Far too slow to sit in the step
                              loop; it runs *offline* to populate profile
                              tables for configurations nobody measured.
============================  ================================================

The DP dependency chain the user-visible simulation executes (receive data
plus a small start packet, compute, hand data over, write a small packet to
release the next rank) is exactly GOAL's ``recv``/``calc``/``send`` chain
with ``requires`` edges; providers only supply the ``calc`` durations.

Host-side initiation (doorbells) is modeled separately in
:mod:`simllm.compute.host`.
"""

from simllm.compute.host import HostInitiationModel
from simllm.compute.provider import (
    PS_PER_SECOND,
    ComputeProvider,
    DurationEstimate,
    GpuSpec,
    KernelSpec,
    ProfileTableProvider,
    RooflineProvider,
)
from simllm.compute.transformer import (
    GPU_ENVELOPES,
    ModelDims,
    estimate_step_latency_ps,
    step_kernel,
)

__all__ = [
    "GPU_ENVELOPES",
    "PS_PER_SECOND",
    "ComputeProvider",
    "DurationEstimate",
    "GpuSpec",
    "HostInitiationModel",
    "KernelSpec",
    "ModelDims",
    "ProfileTableProvider",
    "RooflineProvider",
    "estimate_step_latency_ps",
    "step_kernel",
]
