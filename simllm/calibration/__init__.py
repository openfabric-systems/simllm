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
    "BatchServicePoint",
    "CalibrationCompiler",
    "CalibrationValidationError",
    "DeviceServiceCompilation",
    "HardwareCollector",
    "KernelCycleLookupBinding",
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
    "interpolate_batch_service_ps",
    "resolve_registry_root",
    "resolve_suite_root",
    "validate_kernel_cycle_lut",
    "validate_path",
]

_EXPORTS = {
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
