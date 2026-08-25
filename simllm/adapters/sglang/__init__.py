"""SGLang adapter, pinned to SGLang main commit **bfeae4e** (milestone M3).

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

:mod:`simllm.adapters.sglang.pump` is the in-process driver seam. It builds an
SGLang ``Scheduler`` in the calling process and unrolls ``event_loop_normal``
into a synchronous step, which is what lets ``configure(step_sink=...)`` reach
the worker at all: the hooks are process local and SGLang normally builds the
scheduler in a child process.

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
    "PumpCompletion": ("pump", "PumpCompletion"),
    "PumpStepOutcome": ("pump", "PumpStepOutcome"),
    "SGLANG_HOST_PROFILES": ("host", "SGLANG_HOST_PROFILES"),
    "SGLANG_HOST_TRANSFER_DISCLOSURE": ("host", "SGLANG_HOST_TRANSFER_DISCLOSURE"),
    "SGLANG_TRANSFERRED_LAUNCH_COUNTS": (
        "host",
        "SGLANG_TRANSFERRED_LAUNCH_COUNTS",
    ),
    "SchedulerOutputCollector": ("pump", "SchedulerOutputCollector"),
    "SglangHostSelection": ("host", "SglangHostSelection"),
    "SglangSchedulerPump": ("pump", "SglangSchedulerPump"),
    "active_sample_identity": ("worker", "active_sample_identity"),
    "build_in_process_scheduler": ("pump", "build_in_process_scheduler"),
    "chunked_prefill_refusal": ("pump", "chunked_prefill_refusal"),
    "select_sglang_host_model": ("host", "select_sglang_host_model"),
    "read_output_batch": ("pump", "read_output_batch"),
    "tokenized_generate_request": ("pump", "tokenized_generate_request"),
    "SglangHttpSubmitter": ("client", "SglangHttpSubmitter"),
    "SglangOpenLoopDriver": ("client", "SglangOpenLoopDriver"),
    "GroupCoordinatorEvent": ("communicator", "GroupCoordinatorEvent"),
    "GroupCoordinatorEventStream": ("communicator", "GroupCoordinatorEventStream"),
    "GroupCoordinatorObserver": ("communicator", "GroupCoordinatorObserver"),
    "SGLANG_TP_PAYLOAD_BYTES": ("communicator", "SGLANG_TP_PAYLOAD_BYTES"),
    "SglReplayServingSnapshot": ("replay", "SglReplayServingSnapshot"),
    "SglReplayTokenSource": ("replay", "SglReplayTokenSource"),
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
    "reset_configuration": ("worker", "reset_configuration"),
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
    "SGLANG_HOST_PROFILES",
    "SGLANG_HOST_TRANSFER_DISCLOSURE",
    "SGLANG_TP_PAYLOAD_BYTES",
    "SGLANG_TRANSFERRED_LAUNCH_COUNTS",
    "BatchRow",
    "GroupCoordinatorEvent",
    "GroupCoordinatorEventStream",
    "GroupCoordinatorObserver",
    "PreparedSglangSubmission",
    "PumpCompletion",
    "PumpStepOutcome",
    "SchedulerOutputCollector",
    "SglReplayServingSnapshot",
    "SglReplayTokenSource",
    "SglStepTranslator",
    "SglangHostSelection",
    "SglangHttpSubmitter",
    "SglangOpenLoopDriver",
    "SglangSchedulerPump",
    "ShapeDType",
    "ShapeTensor",
    "SimGroupCoordinator",
    "SimTpModelWorker",
    "SimWorkerConfig",
    "SimWorkerHooks",
    "active_sample_identity",
    "build_in_process_scheduler",
    "chunked_prefill_refusal",
    "configure",
    "install",
    "latest_worker",
    "mark_oracle_capture_start",
    "mark_oracle_submission_group",
    "model_dims_from_sglang",
    "observe_schedule_batch",
    "read_output_batch",
    "register",
    "reset_configuration",
    "select_sglang_host_model",
    "sglang_generate_payload",
    "sglang_is_available",
    "token_completion_times_from_sglang_chunks",
    "tokenized_generate_request",
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
