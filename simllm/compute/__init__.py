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
:class:`TraceCalibratedGpuProvider`
                              Deterministic CTA, warp, pipeline, memory, and
                              copy-service replay. Replays once at provider
                              construction; online estimates are cached.
External Accel-Sim            Offline SASS trace replay used to calibrate and
                              cross-check immutable profile tables.
============================  ================================================

The DP dependency chain the user-visible simulation executes (receive data
plus a small start packet, compute, hand data over, write a small packet to
release the next rank) is exactly GOAL's ``recv``/``calc``/``send`` chain
with ``requires`` edges; providers only supply the ``calc`` durations.

Host-side initiation (doorbells) is modeled separately in
:mod:`simllm.compute.host`.
"""

from simllm.compute.gpu_model import (
    GPU_MODEL_IMPLEMENTATION,
    CopyDirection,
    CopyDirectionProfile,
    CopyEngineProfile,
    CopyEngineServiceModel,
    CopyServiceEstimate,
    CopyTransfer,
    CtaTrace,
    GpuArchitectureProfile,
    GpuCalibrationProfile,
    GpuConcurrentEstimate,
    GpuKernelEstimate,
    GpuModelProvenance,
    GpuTask,
    GpuTaskEstimate,
    GpuTaskKind,
    KernelLaunch,
    MemoryHierarchyProfile,
    MemorySpace,
    NvlinkProfile,
    PipelineKind,
    PipelineProfile,
    SassInstruction,
    SassWarpTrace,
    SmSchedulerModel,
    TraceCalibratedGpuProvider,
    WarpSchedulerPolicy,
    a100_sxm_80gb_seed_profile,
    h100_sxm_80gb_seed_profile,
)
from simllm.compute.gpu_model_io import (
    GPU_MODEL_ARTIFACT_SCHEMA,
    CalibrationSplit,
    GpuCaptureEnvironment,
    GpuCopyReplayRecord,
    GpuKernelCatalogRecord,
    GpuMeasurementRecord,
    GpuModelArtifact,
    GpuReplayRecord,
    gpu_model_artifact_from_json,
    gpu_model_artifact_to_json,
    gpu_model_artifact_to_profile_table,
    load_gpu_model_artifact,
    save_gpu_model_artifact,
)
from simllm.compute.host import HostInitiationModel
from simllm.compute.nccl import (
    NCCL_RING_EGRESS_IMPLEMENTATION,
    nccl_ring_allreduce_launch,
    nccl_ring_allreduce_task,
    nccl_ring_egress_bytes,
)
from simllm.compute.provider import (
    PROFILE_TABLE_SCHEMA,
    PS_PER_SECOND,
    ComputeProvider,
    DurationEstimate,
    GpuSpec,
    KernelSpec,
    ProfileTableProvenance,
    ProfileTableProvider,
    RooflineProvider,
)
from simllm.compute.transformer import (
    GPU_ENVELOPES,
    ModelDims,
    estimate_step_latency_ps,
    step_kernel,
    step_kernels,
)

__all__ = [
    "GPU_ENVELOPES",
    "GPU_MODEL_ARTIFACT_SCHEMA",
    "GPU_MODEL_IMPLEMENTATION",
    "NCCL_RING_EGRESS_IMPLEMENTATION",
    "PROFILE_TABLE_SCHEMA",
    "PS_PER_SECOND",
    "CalibrationSplit",
    "ComputeProvider",
    "CopyDirection",
    "CopyDirectionProfile",
    "CopyEngineProfile",
    "CopyEngineServiceModel",
    "CopyServiceEstimate",
    "CopyTransfer",
    "CtaTrace",
    "DurationEstimate",
    "GpuArchitectureProfile",
    "GpuCalibrationProfile",
    "GpuCaptureEnvironment",
    "GpuConcurrentEstimate",
    "GpuCopyReplayRecord",
    "GpuKernelCatalogRecord",
    "GpuKernelEstimate",
    "GpuMeasurementRecord",
    "GpuModelArtifact",
    "GpuModelProvenance",
    "GpuReplayRecord",
    "GpuSpec",
    "GpuTask",
    "GpuTaskEstimate",
    "GpuTaskKind",
    "HostInitiationModel",
    "KernelLaunch",
    "KernelSpec",
    "MemoryHierarchyProfile",
    "MemorySpace",
    "ModelDims",
    "NvlinkProfile",
    "PipelineKind",
    "PipelineProfile",
    "ProfileTableProvenance",
    "ProfileTableProvider",
    "RooflineProvider",
    "SassInstruction",
    "SassWarpTrace",
    "SmSchedulerModel",
    "TraceCalibratedGpuProvider",
    "WarpSchedulerPolicy",
    "a100_sxm_80gb_seed_profile",
    "estimate_step_latency_ps",
    "gpu_model_artifact_from_json",
    "gpu_model_artifact_to_json",
    "gpu_model_artifact_to_profile_table",
    "h100_sxm_80gb_seed_profile",
    "load_gpu_model_artifact",
    "nccl_ring_allreduce_launch",
    "nccl_ring_allreduce_task",
    "nccl_ring_egress_bytes",
    "save_gpu_model_artifact",
    "step_kernel",
    "step_kernels",
]
