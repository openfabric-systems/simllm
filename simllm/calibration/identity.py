"""Offline-facing aliases for the serving-safe device identity records.

The defining authority lives in :mod:`simllm.compute.device_model` so online
model loading never imports the offline calibration package.
"""

from simllm.compute.device_model import (
    SHAPE_SCHEMA_SCHEMA,
    AnalyticalImplementationRef,
    BinaryImplementationRef,
    ImplementationRef,
    ShapeAxis,
    ShapeSchema,
    ShapeVector,
    implementation_ref_from_obj,
    validate_shape_schemas,
)

__all__ = [
    "SHAPE_SCHEMA_SCHEMA",
    "AnalyticalImplementationRef",
    "BinaryImplementationRef",
    "ImplementationRef",
    "ShapeAxis",
    "ShapeSchema",
    "ShapeVector",
    "implementation_ref_from_obj",
    "validate_shape_schemas",
]
