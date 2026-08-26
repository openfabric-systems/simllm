"""vLLM adapter, pinned to vLLM **v0.27.1** (milestone M2).

vLLM's v1 engine resolves its executor class from a dotted import path, so
SimLLM plugs in without a fork::

    SIMLLM_VLLM_MODE=virtual SIMLLM_VLLM_GPU=b100 \\
    SIMLLM_VLLM_STEP_RECORDS=... \\
    vllm serve meta-llama/Llama-3.1-8B \\
        --distributed-executor-backend simllm.adapters.vllm.SimExecutor \\
        --num-gpu-blocks-override 8192

``SimExecutor`` subclasses ``vllm.v1.executor.abstract.Executor``: it serves
the init-time RPCs with model-derived values, pins the simulated KV pool via
``CacheConfig.num_gpu_blocks_override``, and fabricates a
``ModelRunnerOutput`` per step while translating that step into a
:class:`simllm.core.StepRecord`. The v1 scheduler, KV-cache manager, block
pool and prefix hashing run unmodified.

``PlacementExporter`` is the capture-side companion, used on *real* vLLM runs
to extract a placement manifest without a fork::

    vllm serve <model> -tp 8 \\
        --worker-extension-cls simllm.adapters.vllm.PlacementExporter

``SimWorker`` is the explicitly gated model-runner-boundary skeleton::

    SIMLLM_VLLM_WORKER_MODE=skeleton \\
    VLLM_USE_V2_MODEL_RUNNER=0 \\
    vllm serve <model> --no-async-scheduling \\
        --worker-cls simllm.adapters.vllm.SimWorker

It skips stock ``Worker.init_device``, creates no physical GPU state, and
uses one core virtual clock for mirrored calls and step records.

``SimGroupCoordinator`` is the shape-only communication companion for that
skeleton. It mirrors vLLM's all-reduce, all-gather, broadcast, send, receive,
and rank-membership surface without importing vLLM or requiring torch. Each
successful call emits a zero-time boundary event, lowers to ``CollectiveWork``,
and enters the COMP-15 stack skeleton for multi-rank groups. Runtime completion
projection and communication timing are intentionally not part of this slice.

Environment variables read by the executor (full table in
:mod:`simllm.adapters.vllm.executor`): ``SIMLLM_VLLM_MODE``,
``SIMLLM_VLLM_KV_MEMORY_BYTES``, ``SIMLLM_VLLM_GPU``,
``SIMLLM_VLLM_PEAK_FLOPS``, ``SIMLLM_VLLM_MEM_BANDWIDTH``,
``SIMLLM_VLLM_EFFICIENCY``, ``SIMLLM_VLLM_HOST_INIT_PS``,
``SIMLLM_VLLM_TOKEN_ID``, ``SIMLLM_VLLM_STEP_RECORDS``,
``SIMLLM_VLLM_REPLAY_RUN``.
The worker-only entry gate is ``SIMLLM_VLLM_WORKER_MODE=skeleton``.

Exports are resolved lazily through the module ``__getattr__``, so importing
this package pulls in neither the executor module nor vLLM until a name is
actually used. vLLM itself only ever asks for one attribute
(``resolve_obj_by_qualname`` does ``import_module`` plus ``getattr``), which
is exactly what this indirection serves.
"""

from typing import Any

from simllm.adapters.vllm._version import PINNED_VLLM_VERSION

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "FLOAT32": ("communicator", "FLOAT32"),
    "GROUP_COORDINATOR_EVENT_SCHEMA": (
        "communicator",
        "GROUP_COORDINATOR_EVENT_SCHEMA",
    ),
    "INT32": ("communicator", "INT32"),
    "ORACLE_ENABLE_ENV": ("oracle", "ORACLE_ENABLE_ENV"),
    "ORACLE_LOG_ENV": ("oracle", "ORACLE_LOG_ENV"),
    "ORACLE_OBSERVATION_SCHEMA": ("oracle", "ORACLE_OBSERVATION_SCHEMA"),
    "GroupCoordinatorEvent": ("communicator", "GroupCoordinatorEvent"),
    "GroupCoordinatorObserver": ("communicator", "GroupCoordinatorObserver"),
    "ShapeDType": ("communicator", "ShapeDType"),
    "ShapeTensor": ("communicator", "ShapeTensor"),
    "SimGroupCoordinator": ("communicator", "SimGroupCoordinator"),
    "GPU_ENVELOPES": ("executor", "GPU_ENVELOPES"),
    "HostModelStepSink": ("executor", "HostModelStepSink"),
    "ModelDims": ("executor", "ModelDims"),
    "ObservationStepSink": ("executor", "ObservationStepSink"),
    "SimExecutor": ("executor", "SimExecutor"),
    "SimExecutorConfig": ("executor", "SimExecutorConfig"),
    "SimExecutorHooks": ("executor", "SimExecutorHooks"),
    "StepTranslator": ("executor", "StepTranslator"),
    "TranslatedStep": ("executor", "TranslatedStep"),
    "configure": ("executor", "configure"),
    "reset_configuration": ("executor", "reset_configuration"),
    "estimate_step_latency_ps": ("executor", "estimate_step_latency_ps"),
    "fabricate_sampled_tokens": ("executor", "fabricate_sampled_tokens"),
    "latest_executor": ("executor", "latest_executor"),
    "observe_scheduler_output": ("executor", "observe_scheduler_output"),
    "mark_oracle_capture_start": ("oracle", "mark_oracle_capture_start"),
    "mark_oracle_request_mapping": ("oracle", "mark_oracle_request_mapping"),
    "mark_oracle_submission_group": ("oracle", "mark_oracle_submission_group"),
    "step_kernel": ("executor", "step_kernel"),
    "translate_scheduler_output": ("executor", "translate_scheduler_output"),
    "step_records_to_json": ("executor", "step_records_to_json"),
    "vllm_is_available": ("executor", "vllm_is_available"),
    "ExpertGroupStepSink": ("executor", "ExpertGroupStepSink"),
    "ExpertParallelGeometry": ("executor", "ExpertParallelGeometry"),
    "expert_group_ranks": ("executor", "expert_group_ranks"),
    "expert_parallel_geometry": ("executor", "expert_parallel_geometry"),
    "model_dims_from_vllm_config": ("executor", "model_dims_from_vllm_config"),
    "write_step_records": ("executor", "write_step_records"),
    "ReplayServingSnapshot": ("replay", "ReplayServingSnapshot"),
    "ReplayTokenSource": ("replay", "ReplayTokenSource"),
    "sample_adapter_tokens": ("replay", "sample_adapter_tokens"),
    "GRANITE_ALL2ALL_BACKEND": ("schedule", "GRANITE_ALL2ALL_BACKEND"),
    "OBSERVED_SCHEDULE_GRANITE_DBO": (
        "schedule",
        "OBSERVED_SCHEDULE_GRANITE_DBO",
    ),
    "OBSERVED_SCHEDULE_OFF": ("schedule", "OBSERVED_SCHEDULE_OFF"),
    "GraniteScheduleTiming": ("schedule", "GraniteScheduleTiming"),
    "VllmBatchSlice": ("schedule", "VllmBatchSlice"),
    "build_granite_execution_observations": (
        "schedule",
        "build_granite_execution_observations",
    ),
    "observations_from_vllm_step": ("schedule", "observations_from_vllm_step"),
    "vllm_batch_slices": ("schedule", "vllm_batch_slices"),
    "MirroredCall": ("worker", "MirroredCall"),
    "SKELETON_EMPTY_STEP_CALL_SEQUENCE": (
        "worker",
        "SKELETON_EMPTY_STEP_CALL_SEQUENCE",
    ),
    "SKELETON_INIT_CALL_SEQUENCE": ("worker", "SKELETON_INIT_CALL_SEQUENCE"),
    "SKELETON_STEP_CALL_SEQUENCE": ("worker", "SKELETON_STEP_CALL_SEQUENCE"),
    "SKELETON_WORKER_MODE": ("worker", "SKELETON_WORKER_MODE"),
    "SimModelRunner": ("worker", "SimModelRunner"),
    "SimModelRunnerOutput": ("worker", "SimModelRunnerOutput"),
    "SimWorker": ("worker", "SimWorker"),
    "WORKER_MODE_ENV": ("worker", "WORKER_MODE_ENV"),
    "latest_worker": ("worker", "latest_worker"),
    "skeleton_mode_enabled": ("worker", "skeleton_mode_enabled"),
    "PlacementExporter": ("worker_ext", "PlacementExporter"),
    "PD_CONNECTOR_MODULE": ("pd_session", "PD_CONNECTOR_MODULE"),
    "PD_CONNECTOR_NAME": ("pd_session", "PD_CONNECTOR_NAME"),
    "PD_KV_PARAMS_SCHEMA": ("pd_session", "PD_KV_PARAMS_SCHEMA"),
    "DEPLOYMENT_CURVE_POINT_SCHEMA": (
        "pd_session",
        "DEPLOYMENT_CURVE_POINT_SCHEMA",
    ),
    "DEPLOYMENT_CURVE_SCHEMA": ("pd_session", "DEPLOYMENT_CURVE_SCHEMA"),
    "VllmDisaggregatedSession": ("pd_session", "VllmDisaggregatedSession"),
    "VllmPdConcurrentResult": ("pd_session", "VllmPdConcurrentResult"),
    "VllmPdCurvePoint": ("pd_session", "VllmPdCurvePoint"),
    "VllmPdCurveRecord": ("pd_session", "VllmPdCurveRecord"),
    "VllmPdRequest": ("pd_session", "VllmPdRequest"),
    "VllmPdRequestResult": ("pd_session", "VllmPdRequestResult"),
    "VllmPdSessionConfig": ("pd_session", "VllmPdSessionConfig"),
    "VllmPoolEngine": ("pd_session", "VllmPoolEngine"),
    "manifest_from_worker_entries": ("worker_ext", "manifest_from_worker_entries"),
    "placement_entry": ("worker_ext", "placement_entry"),
    "vllm_version": ("worker_ext", "vllm_version"),
}

__all__ = [
    "DEPLOYMENT_CURVE_POINT_SCHEMA",
    "DEPLOYMENT_CURVE_SCHEMA",
    "FLOAT32",
    "GPU_ENVELOPES",
    "GRANITE_ALL2ALL_BACKEND",
    "GROUP_COORDINATOR_EVENT_SCHEMA",
    "INT32",
    "OBSERVED_SCHEDULE_GRANITE_DBO",
    "OBSERVED_SCHEDULE_OFF",
    "ORACLE_ENABLE_ENV",
    "ORACLE_LOG_ENV",
    "ORACLE_OBSERVATION_SCHEMA",
    "PD_CONNECTOR_MODULE",
    "PD_CONNECTOR_NAME",
    "PD_KV_PARAMS_SCHEMA",
    "PINNED_VLLM_VERSION",
    "SKELETON_EMPTY_STEP_CALL_SEQUENCE",
    "SKELETON_INIT_CALL_SEQUENCE",
    "SKELETON_STEP_CALL_SEQUENCE",
    "SKELETON_WORKER_MODE",
    "WORKER_MODE_ENV",
    "ExpertGroupStepSink",
    "ExpertParallelGeometry",
    "GraniteScheduleTiming",
    "GroupCoordinatorEvent",
    "GroupCoordinatorObserver",
    "HostModelStepSink",
    "MirroredCall",
    "ModelDims",
    "ObservationStepSink",
    "PlacementExporter",
    "ReplayServingSnapshot",
    "ReplayTokenSource",
    "ShapeDType",
    "ShapeTensor",
    "SimExecutor",
    "SimExecutorConfig",
    "SimExecutorHooks",
    "SimGroupCoordinator",
    "SimModelRunner",
    "SimModelRunnerOutput",
    "SimWorker",
    "StepTranslator",
    "TranslatedStep",
    "VllmBatchSlice",
    "VllmDisaggregatedSession",
    "VllmPdConcurrentResult",
    "VllmPdCurvePoint",
    "VllmPdCurveRecord",
    "VllmPdRequest",
    "VllmPdRequestResult",
    "VllmPdSessionConfig",
    "VllmPoolEngine",
    "build_granite_execution_observations",
    "configure",
    "estimate_step_latency_ps",
    "expert_group_ranks",
    "expert_parallel_geometry",
    "fabricate_sampled_tokens",
    "latest_executor",
    "latest_worker",
    "manifest_from_worker_entries",
    "mark_oracle_capture_start",
    "mark_oracle_request_mapping",
    "mark_oracle_submission_group",
    "model_dims_from_vllm_config",
    "observations_from_vllm_step",
    "observe_scheduler_output",
    "placement_entry",
    "reset_configuration",
    "sample_adapter_tokens",
    "skeleton_mode_enabled",
    "step_kernel",
    "step_records_to_json",
    "translate_scheduler_output",
    "vllm_batch_slices",
    "vllm_is_available",
    "vllm_version",
    "write_step_records",
]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    from importlib import import_module

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
