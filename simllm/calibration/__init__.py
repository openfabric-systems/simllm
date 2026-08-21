"""Lazy, offline-only device calibration interfaces.

Importing :mod:`simllm` does not import this package. Importing this package in
turn exposes only lazy attributes, so command help and record validation never
load a hardware runtime or an optional simulator.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CalibrationCompiler",
    "CalibrationValidationError",
    "HardwareCollector",
    "OfflineKernelSimulator",
    "RootKind",
    "RootResolutionError",
    "RootSelection",
    "RootSource",
    "ValidationResult",
    "resolve_registry_root",
    "resolve_suite_root",
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
    "HardwareCollector": ("simllm.calibration.protocols", "HardwareCollector"),
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
    "resolve_registry_root": (
        "simllm.calibration.registry",
        "resolve_registry_root",
    ),
    "resolve_suite_root": ("simllm.calibration.registry", "resolve_suite_root"),
    "validate_path": ("simllm.calibration.validation", "validate_path"),
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
