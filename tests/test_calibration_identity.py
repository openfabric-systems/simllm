from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from simllm.calibration.identity import (
    AnalyticalImplementationRef,
    BinaryImplementationRef,
    ShapeAxis,
    ShapeSchema,
    ShapeVector,
    implementation_ref_from_obj,
    validate_shape_schemas,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def shape_schema() -> ShapeSchema:
    return ShapeSchema(
        shape_schema_id="gemm-v1",
        axes=(
            ShapeAxis(axis_id="m", unit="elements", minimum=1, maximum=4096),
            ShapeAxis(axis_id="n", unit="elements", minimum=1, maximum=8192),
        ),
    )


def binary_ref() -> BinaryImplementationRef:
    return BinaryImplementationRef(
        implementation_id="cublas-gemm-sm80",
        vendor_id="nvidia",
        device_isa="sm_80",
        module_sha256=SHA_A,
        function_sha256=SHA_B,
        function_symbol=None,
        backend_id="cuda",
        algorithm_id="cublas-algo-7",
        launch_formula_id="gemm-grid-v1",
    )


def analytical_ref() -> AnalyticalImplementationRef:
    return AnalyticalImplementationRef(
        implementation_id="roofline-gemm-gfx942",
        model_sha256=SHA_A,
        target_vendor_id="amd",
        target_architecture="mi300x",
        target_isa="gfx942",
        applicability_sha256=SHA_B,
        trusted_evaluator_id="simllm-roofline-v1",
        parameter_sha256=SHA_C,
        anchor_evidence_sha256=SHA_D,
        delta_evidence_sha256=SHA_E,
    )


def test_shape_schema_and_vector_strict_round_trip() -> None:
    schema = shape_schema()
    parsed = ShapeSchema.from_obj(schema.to_obj())
    vector = ShapeVector(shape_schema_id="gemm-v1", values=(128, 4096))

    assert parsed == schema
    assert ShapeVector.from_obj(vector.to_obj()) == vector
    parsed.validate_vector(vector)
    assert ShapeSchema.__module__ == "simllm.compute.device_model"


def test_shape_domain_and_identity_are_enforced() -> None:
    schema = shape_schema()

    with pytest.raises(ValueError, match="outside 'm' domain"):
        schema.validate_vector(ShapeVector(shape_schema_id="gemm-v1", values=(0, 1)))
    with pytest.raises(ValueError, match="shape_schema_id"):
        schema.validate_vector(ShapeVector(shape_schema_id="other", values=(1, 1)))
    with pytest.raises(ValueError, match="expected 2 values"):
        schema.validate_vector(ShapeVector(shape_schema_id="gemm-v1", values=(1,)))


def test_shape_reader_rejects_unknown_fields_and_bool_axes() -> None:
    payload = shape_schema().to_obj()
    payload["extra"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        ShapeSchema.from_obj(payload)

    axis = ShapeAxis(axis_id="x", unit="items", minimum=0, maximum=1).to_obj()
    axis["minimum"] = False
    with pytest.raises(ValueError, match="expected an integer"):
        ShapeAxis.from_obj(axis)


def test_shape_schema_rejects_duplicate_axis_ids_and_model_ordering() -> None:
    duplicate = ShapeAxis(axis_id="x", unit="items", minimum=0, maximum=1)
    with pytest.raises(ValueError, match="axis IDs must be unique"):
        ShapeSchema(shape_schema_id="bad", axes=(duplicate, duplicate))

    first = ShapeSchema(shape_schema_id="z", axes=())
    second = ShapeSchema(shape_schema_id="a", axes=())
    with pytest.raises(ValueError, match="must be sorted"):
        validate_shape_schemas((first, second))


def test_binary_implementation_ref_is_exact_and_round_trips() -> None:
    reference = binary_ref()

    assert implementation_ref_from_obj(reference.to_obj()) == reference
    assert reference.to_obj()["kind"] == "binary"

    payload = reference.to_obj()
    payload["launch_mode"] = "graph"
    with pytest.raises(ValueError, match="unknown fields"):
        implementation_ref_from_obj(payload)

    payload = reference.to_obj()
    payload["launch_formula_id"] = "grid=ceil(m/128)"
    with pytest.raises(ValueError, match="trusted data identifier"):
        implementation_ref_from_obj(payload)

    payload = reference.to_obj()
    payload["launch_formula"] = payload.pop("launch_formula_id")
    with pytest.raises(ValueError, match="missing fields"):
        implementation_ref_from_obj(payload)


def test_binary_ref_requires_one_function_identity_and_backend_or_algorithm() -> None:
    kwargs = binary_ref().to_obj()
    kwargs.pop("kind")
    kwargs["function_symbol"] = "kernel"
    with pytest.raises(ValueError, match="exactly one"):
        BinaryImplementationRef(**kwargs)

    kwargs = binary_ref().to_obj()
    kwargs.pop("kind")
    kwargs["backend_id"] = None
    kwargs["algorithm_id"] = None
    with pytest.raises(ValueError, match="at least one"):
        BinaryImplementationRef(**kwargs)


def test_analytical_ref_is_content_addressed_and_exact() -> None:
    reference = analytical_ref()

    assert implementation_ref_from_obj(reference.to_obj()) == reference
    malformed = reference.to_obj()
    malformed["parameter_sha256"] = "sha256:" + SHA_C
    with pytest.raises(ValueError, match="64 lowercase"):
        implementation_ref_from_obj(malformed)


def test_implementation_union_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="binary.*analytical"):
        implementation_ref_from_obj({"kind": "kernel-name"})


def test_identity_records_are_immutable() -> None:
    vector = ShapeVector(shape_schema_id="fixed", values=())
    with pytest.raises(FrozenInstanceError):
        vector.shape_schema_id = "changed"  # type: ignore[misc]
