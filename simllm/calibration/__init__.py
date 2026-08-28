"""Lazy, offline-only device calibration interfaces.

Importing :mod:`simllm` does not import this package. Importing this package in
turn exposes only lazy attributes, so command help and record validation never
load a hardware runtime or an optional simulator.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "KERNEL_CYCLE_INPUT_SCHEMA",
    "KERNEL_CYCLE_LUT_SCHEMA",
    "KERNEL_CYCLE_PRICING_PROVENANCE_SCHEMA",
    "LOCAL_SHARD_REQUEST_SCHEMA",
    "LOCAL_SHARD_RESULT_SCHEMA",
    "LOCAL_SHARD_RUN_SCHEMA",
    "BatchServicePoint",
    "CalibrationCompiler",
    "CalibrationValidationError",
    "DeviceServiceCompilation",
    "ExternalCompositionLedger",
    "ExternalDatabaseError",
    "ExternalDatabaseGapError",
    "ExternalDatabaseIdentityError",
    "ExternalLatency",
    "ExternalModelConfig",
    "ExternalNcclDatabase",
    "ExternalOperationDatabase",
    "ExternalPassModel",
    "ExternalPassResult",
    "ExternalQwen32BPassModel",
    "ExternalSourceIdentity",
    "HardwareCollector",
    "KernelCycleLookupBinding",
    "LocalShardCaptureError",
    "LocalShardCaptureRequest",
    "LocalShardCaptureResult",
    "LocalShardCaptureRun",
    "ModelKernelInventory",
    "OfflineKernelSimulator",
    "RootKind",
    "RootResolutionError",
    "RootSelection",
    "RootSource",
    "ValidationResult",
    "analyze_kernel_cycle_capture",
    "compile_device_service_entries",
    "compile_pool_local_batch_service_provider",
    "compile_profile_table",
    "compile_session_profile_provider",
    "external_artifact_licensing_findings",
    "import_external_database",
    "interpolate_batch_service_ps",
    "resolve_registry_root",
    "resolve_suite_root",
    "run_local_shard_capture",
    "synthetic_input_sha256",
    "synthetic_token_rows",
    "validate_kernel_cycle_lut",
    "validate_local_shard_request",
    "validate_local_shard_result",
    "validate_path",
]

_EXPORTS = {
    "LOCAL_SHARD_REQUEST_SCHEMA": (
        "simllm.calibration.local_shard",
        "LOCAL_SHARD_REQUEST_SCHEMA",
    ),
    "LOCAL_SHARD_RESULT_SCHEMA": (
        "simllm.calibration.local_shard",
        "LOCAL_SHARD_RESULT_SCHEMA",
    ),
    "LOCAL_SHARD_RUN_SCHEMA": (
        "simllm.calibration.local_shard",
        "LOCAL_SHARD_RUN_SCHEMA",
    ),
    "LocalShardCaptureError": (
        "simllm.calibration.local_shard",
        "LocalShardCaptureError",
    ),
    "LocalShardCaptureRequest": (
        "simllm.calibration.local_shard",
        "LocalShardCaptureRequest",
    ),
    "LocalShardCaptureResult": (
        "simllm.calibration.local_shard",
        "LocalShardCaptureResult",
    ),
    "LocalShardCaptureRun": (
        "simllm.calibration.local_shard",
        "LocalShardCaptureRun",
    ),
    "run_local_shard_capture": (
        "simllm.calibration.local_shard",
        "run_local_shard_capture",
    ),
    "synthetic_input_sha256": (
        "simllm.calibration.local_shard",
        "synthetic_input_sha256",
    ),
    "synthetic_token_rows": (
        "simllm.calibration.local_shard",
        "synthetic_token_rows",
    ),
    "validate_local_shard_request": (
        "simllm.calibration.local_shard",
        "validate_local_shard_request",
    ),
    "validate_local_shard_result": (
        "simllm.calibration.local_shard",
        "validate_local_shard_result",
    ),
    "CalibrationCompiler": (
        "simllm.calibration.protocols",
        "CalibrationCompiler",
    ),
    "CalibrationValidationError": (
        "simllm.calibration.validation",
        "CalibrationValidationError",
    ),
    "DeviceServiceCompilation": (
        "simllm.calibration.kernel_cycle_lut",
        "DeviceServiceCompilation",
    ),
    "HardwareCollector": ("simllm.calibration.protocols", "HardwareCollector"),
    "ExternalCompositionLedger": (
        "simllm.calibration.external_db",
        "ExternalCompositionLedger",
    ),
    "ExternalDatabaseError": (
        "simllm.calibration.external_db",
        "ExternalDatabaseError",
    ),
    "ExternalDatabaseGapError": (
        "simllm.calibration.external_db",
        "ExternalDatabaseGapError",
    ),
    "ExternalDatabaseIdentityError": (
        "simllm.calibration.external_db",
        "ExternalDatabaseIdentityError",
    ),
    "ExternalLatency": ("simllm.calibration.external_db", "ExternalLatency"),
    "ExternalModelConfig": (
        "simllm.calibration.external_pass",
        "ExternalModelConfig",
    ),
    "ExternalNcclDatabase": (
        "simllm.calibration.external_nccl",
        "ExternalNcclDatabase",
    ),
    "ExternalOperationDatabase": (
        "simllm.calibration.external_db",
        "ExternalOperationDatabase",
    ),
    "ExternalPassModel": (
        "simllm.calibration.external_pass",
        "ExternalPassModel",
    ),
    "ExternalPassResult": (
        "simllm.calibration.external_db",
        "ExternalPassResult",
    ),
    "ExternalQwen32BPassModel": (
        "simllm.calibration.external_db",
        "ExternalQwen32BPassModel",
    ),
    "ExternalSourceIdentity": (
        "simllm.calibration.external_db",
        "ExternalSourceIdentity",
    ),
    "external_artifact_licensing_findings": (
        "simllm.calibration.external_db",
        "external_artifact_licensing_findings",
    ),
    "KERNEL_CYCLE_INPUT_SCHEMA": (
        "simllm.calibration.kernel_cycle_lut",
        "KERNEL_CYCLE_INPUT_SCHEMA",
    ),
    "KERNEL_CYCLE_LUT_SCHEMA": (
        "simllm.calibration.kernel_cycle_lut",
        "KERNEL_CYCLE_LUT_SCHEMA",
    ),
    "KERNEL_CYCLE_PRICING_PROVENANCE_SCHEMA": (
        "simllm.calibration.kernel_cycle_lut",
        "KERNEL_CYCLE_PRICING_PROVENANCE_SCHEMA",
    ),
    "KernelCycleLookupBinding": (
        "simllm.calibration.kernel_cycle_lut",
        "KernelCycleLookupBinding",
    ),
    "ModelKernelInventory": (
        "simllm.calibration.model_inventory",
        "ModelKernelInventory",
    ),
    "OfflineKernelSimulator": (
        "simllm.calibration.protocols",
        "OfflineKernelSimulator",
    ),
    "RootKind": ("simllm.calibration.registry", "RootKind"),
    "RootResolutionError": (
        "simllm.calibration.registry",
        "RootResolutionError",
    ),
    "RootSelection": ("simllm.calibration.registry", "RootSelection"),
    "RootSource": ("simllm.calibration.registry", "RootSource"),
    "ValidationResult": (
        "simllm.calibration.validation",
        "ValidationResult",
    ),
    "analyze_kernel_cycle_capture": (
        "simllm.calibration.kernel_cycle_lut",
        "analyze_kernel_cycle_capture",
    ),
    "compile_device_service_entries": (
        "simllm.calibration.kernel_cycle_lut",
        "compile_device_service_entries",
    ),
    "compile_profile_table": (
        "simllm.calibration.kernel_cycle_lut",
        "compile_profile_table",
    ),
    "compile_session_profile_provider": (
        "simllm.calibration.kernel_cycle_lut",
        "compile_session_profile_provider",
    ),
    "BatchServicePoint": (
        "simllm.calibration.batch_service_surface",
        "BatchServicePoint",
    ),
    "compile_pool_local_batch_service_provider": (
        "simllm.calibration.batch_service_surface",
        "compile_pool_local_batch_service_provider",
    ),
    "interpolate_batch_service_ps": (
        "simllm.calibration.batch_service_surface",
        "interpolate_batch_service_ps",
    ),
    "import_external_database": (
        "simllm.calibration.external_db",
        "import_external_database",
    ),
    "resolve_registry_root": (
        "simllm.calibration.registry",
        "resolve_registry_root",
    ),
    "resolve_suite_root": ("simllm.calibration.registry", "resolve_suite_root"),
    "validate_path": ("simllm.calibration.validation", "validate_path"),
    "validate_kernel_cycle_lut": (
        "simllm.calibration.kernel_cycle_lut",
        "validate_kernel_cycle_lut",
    ),
}


def __getattr__(name: str) -> Any:
    """Load a public offline interface on first access."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public attributes in interactive discovery."""

    return sorted(set(globals()) | set(__all__))
