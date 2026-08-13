"""SGLang adapter, pinned to SGLang main commit **8f2a3ad** (milestone M3).

The seam is the TP worker: ``SimTpModelWorker`` subclasses SGLang's
``TpModelWorker`` (the same pattern as SGLang's own MLX worker) and replaces
model-runner construction and the forward pass with a simulated GPU, while
the scheduler, RadixCache and pool accounting run for real. It is installed
without a fork through SGLang's plugin framework::

    SIMLLM_SGLANG_ENABLE=1 SIMLLM_SGLANG_MODE=virtual \\
    python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B \\
        --disable-overlap-schedule

Environment variables read by the worker (full table in
:mod:`simllm.adapters.sglang.worker`): ``SIMLLM_SGLANG_ENABLE``,
``SIMLLM_SGLANG_MODE``, ``SIMLLM_SGLANG_GPU``, ``SIMLLM_SGLANG_PEAK_FLOPS``,
``SIMLLM_SGLANG_MEM_BANDWIDTH``, ``SIMLLM_SGLANG_EFFICIENCY``,
``SIMLLM_SGLANG_HOST_INIT_PS``, ``SIMLLM_SGLANG_TOKEN_ID``,
``SIMLLM_SGLANG_STEP_RECORDS``, ``SIMLLM_SGLANG_COMMUNICATOR_TP_SIZE``,
``SIMLLM_SGLANG_COMMUNICATOR_EVENTS``.

``SimGroupCoordinator`` is the SGLang-shaped communication companion. It
reuses VLLM-14's torch-optional shape and immutable event base, adds SGLang's
``output_tensor_list`` all-gather form, and emits zero-time semantic
``CollectiveWork`` through the COMP-15 compatibility stack.

Exports are resolved lazily through the module ``__getattr__``, so importing
this package pulls in neither the worker module nor SGLang until a name is
actually used.
"""

from typing import Any

from simllm.adapters.sglang._version import PINNED_SGLANG_COMMIT

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BatchRow": ("worker", "BatchRow"),
    "FLOAT32": ("communicator", "FLOAT32"),
    "GROUP_COORDINATOR_EVENT_SCHEMA": (
        "communicator",
        "GROUP_COORDINATOR_EVENT_SCHEMA",
    ),
    "INT32": ("communicator", "INT32"),
    "ORACLE_ENABLE_ENV": ("oracle", "ORACLE_ENABLE_ENV"),
    "ORACLE_LOG_ENV": ("oracle", "ORACLE_LOG_ENV"),
    "ORACLE_OBSERVATION_SCHEMA": ("oracle", "ORACLE_OBSERVATION_SCHEMA"),
    "PreparedSglangSubmission": ("client", "PreparedSglangSubmission"),
    "SglangHttpSubmitter": ("client", "SglangHttpSubmitter"),
    "SglangOpenLoopDriver": ("client", "SglangOpenLoopDriver"),
    "GroupCoordinatorEvent": ("communicator", "GroupCoordinatorEvent"),
    "GroupCoordinatorEventStream": ("communicator", "GroupCoordinatorEventStream"),
    "GroupCoordinatorObserver": ("communicator", "GroupCoordinatorObserver"),
    "SGLANG_TP_PAYLOAD_BYTES": ("communicator", "SGLANG_TP_PAYLOAD_BYTES"),
    "SglStepTranslator": ("worker", "SglStepTranslator"),
    "ShapeDType": ("communicator", "ShapeDType"),
    "ShapeTensor": ("communicator", "ShapeTensor"),
    "SimGroupCoordinator": ("communicator", "SimGroupCoordinator"),
    "SimTpModelWorker": ("worker", "SimTpModelWorker"),
    "SimWorkerConfig": ("worker", "SimWorkerConfig"),
    "SimWorkerHooks": ("worker", "SimWorkerHooks"),
    "sglang_generate_payload": ("client", "sglang_generate_payload"),
    "token_completion_times_from_sglang_chunks": (
        "client",
        "token_completion_times_from_sglang_chunks",
    ),
    "configure": ("worker", "configure"),
    "latest_worker": ("worker", "latest_worker"),
    "model_dims_from_sglang": ("worker", "model_dims_from_sglang"),
    "mark_oracle_capture_start": ("oracle", "mark_oracle_capture_start"),
    "mark_oracle_submission_group": ("oracle", "mark_oracle_submission_group"),
    "observe_schedule_batch": ("worker", "observe_schedule_batch"),
    "sglang_is_available": ("worker", "sglang_is_available"),
    "install": ("plugin", "install"),
    "register": ("plugin", "register"),
}

__all__ = [
    "FLOAT32",
    "GROUP_COORDINATOR_EVENT_SCHEMA",
    "INT32",
    "ORACLE_ENABLE_ENV",
    "ORACLE_LOG_ENV",
    "ORACLE_OBSERVATION_SCHEMA",
    "PINNED_SGLANG_COMMIT",
    "SGLANG_TP_PAYLOAD_BYTES",
    "BatchRow",
    "GroupCoordinatorEvent",
    "GroupCoordinatorEventStream",
    "GroupCoordinatorObserver",
    "PreparedSglangSubmission",
    "SglStepTranslator",
    "SglangHttpSubmitter",
    "SglangOpenLoopDriver",
    "ShapeDType",
    "ShapeTensor",
    "SimGroupCoordinator",
    "SimTpModelWorker",
    "SimWorkerConfig",
    "SimWorkerHooks",
    "configure",
    "install",
    "latest_worker",
    "mark_oracle_capture_start",
    "mark_oracle_submission_group",
    "model_dims_from_sglang",
    "observe_schedule_batch",
    "register",
    "sglang_generate_payload",
    "sglang_is_available",
    "token_completion_times_from_sglang_chunks",
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
