"""Strict object readers for serving-safe compact device model components."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from simllm.compute.device_model import (
    DEVICE_MODEL_SCHEMA,
    DEVICE_RESOURCE_REGISTRY_SCHEMA,
    SHAPE_SCHEMA_SCHEMA,
    DeviceModel,
    DeviceModelAcceptanceStatus,
    DeviceModelLimits,
    DeviceModelTargetBasis,
    DeviceResourceAxis,
    DeviceResourceRegistry,
    DeviceResourceVector,
    DeviceServiceEntry,
    DeviceServiceEntryRecord,
    ExactRate,
    ResourceAxisClass,
    ResourceInteractionContract,
    ResourceServiceScope,
    ServiceEntryEvidence,
    ServiceEntrySourceSelection,
    ServiceEpochDefinition,
    ShapeSchema,
    ShapeVector,
    _array,
    _bounded_integer,
    _fields,
    _git_object_id,
    _object,
    _optional_sha256,
    _optional_string,
    _require_schema,
    _sha256,
    _string,
)


def _enum_value(enum_type: type[Any], value: object, path: str) -> Any:
    raw = _string(value, path)
    try:
        return enum_type(raw)
    except ValueError as exc:
        choices = ", ".join(repr(member.value) for member in enum_type)
        raise ValueError(f"{path}: expected one of {choices}") from exc


def _nullable_bounded_integer(
    value: object,
    path: str,
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> int | None:
    if value is None:
        return None
    return _bounded_integer(
        value,
        path,
        nonnegative=nonnegative,
        positive=positive,
    )


def exact_rate_from_obj(value: object, path: str = "rate") -> ExactRate:
    payload = _object(value, path)
    _fields(payload, path, {"numerator", "denominator"})
    return ExactRate(
        numerator=_bounded_integer(
            payload["numerator"], f"{path}.numerator", nonnegative=True
        ),
        denominator=_bounded_integer(
            payload["denominator"], f"{path}.denominator", positive=True
        ),
    )


def resource_axis_from_obj(
    value: object, path: str = "resource_axis"
) -> DeviceResourceAxis:
    payload = _object(value, path)
    expected = {
        "axis_id",
        "axis_class",
        "service_scope",
        "base_unit",
        "clock_domain_id",
        "capacity_source_id",
        "rate",
        "residency_capacity",
        "exclusive_capacity",
    }
    _fields(payload, path, expected)
    raw_rate = payload["rate"]
    return DeviceResourceAxis(
        axis_id=_string(payload["axis_id"], f"{path}.axis_id"),
        axis_class=_enum_value(
            ResourceAxisClass, payload["axis_class"], f"{path}.axis_class"
        ),
        service_scope=_enum_value(
            ResourceServiceScope,
            payload["service_scope"],
            f"{path}.service_scope",
        ),
        base_unit=_string(payload["base_unit"], f"{path}.base_unit"),
        clock_domain_id=_optional_string(
            payload["clock_domain_id"], f"{path}.clock_domain_id"
        ),
        capacity_source_id=_string(
            payload["capacity_source_id"], f"{path}.capacity_source_id"
        ),
        rate=None if raw_rate is None else exact_rate_from_obj(raw_rate, f"{path}.rate"),
        residency_capacity=_nullable_bounded_integer(
            payload["residency_capacity"],
            f"{path}.residency_capacity",
            nonnegative=True,
        ),
        exclusive_capacity=_nullable_bounded_integer(
            payload["exclusive_capacity"],
            f"{path}.exclusive_capacity",
            positive=True,
        ),
    )


def resource_registry_from_obj(
    value: object, path: str = "resource_registry"
) -> DeviceResourceRegistry:
    payload = _object(value, path)
    expected = {"schema", "device_kind_id", "active_axis_ids", "axes"}
    _fields(payload, path, expected)
    _require_schema(
        payload["schema"], DEVICE_RESOURCE_REGISTRY_SCHEMA, f"{path}.schema"
    )
    return DeviceResourceRegistry(
        device_kind_id=_string(
            payload["device_kind_id"], f"{path}.device_kind_id"
        ),
        active_axis_ids=tuple(
            _string(item, f"{path}.active_axis_ids[{index}]")
            for index, item in enumerate(
                _array(payload["active_axis_ids"], f"{path}.active_axis_ids")
            )
        ),
        axes=tuple(
            resource_axis_from_obj(item, f"{path}.axes[{index}]")
            for index, item in enumerate(_array(payload["axes"], f"{path}.axes"))
        ),
    )


def resource_vector_from_obj(
    value: object, path: str = "resource_vector"
) -> DeviceResourceVector:
    payload = _object(value, path)
    expected = {"registry_sha256", "device_kind_id", "values", "known"}
    _fields(payload, path, expected)
    known: list[bool] = []
    for index, item in enumerate(_array(payload["known"], f"{path}.known")):
        if type(item) is not bool:
            raise ValueError(f"{path}.known[{index}]: expected a boolean")
        known.append(item)
    return DeviceResourceVector(
        registry_sha256=_sha256(
            payload["registry_sha256"], f"{path}.registry_sha256"
        ),
        device_kind_id=_string(
            payload["device_kind_id"], f"{path}.device_kind_id"
        ),
        values=tuple(
            _bounded_integer(item, f"{path}.values[{index}]", nonnegative=True)
            for index, item in enumerate(
                _array(payload["values"], f"{path}.values")
            )
        ),
        known=tuple(known),
    )


def service_epoch_from_obj(
    value: object, path: str = "service_epoch"
) -> ServiceEpochDefinition:
    payload = _object(value, path)
    _fields(payload, path, {"resource_vector", "fixed_floor_ps"})
    return ServiceEpochDefinition(
        resource_vector=resource_vector_from_obj(
            payload["resource_vector"], f"{path}.resource_vector"
        ),
        fixed_floor_ps=_nullable_bounded_integer(
            payload["fixed_floor_ps"], f"{path}.fixed_floor_ps", nonnegative=True
        ),
    )


def service_entry_from_obj(
    value: object, path: str = "service_entry"
) -> DeviceServiceEntry:
    payload = _object(value, path)
    _fields(payload, path, {"implementation_id", "shape_vector", "epochs"})
    return DeviceServiceEntry(
        implementation_id=_string(
            payload["implementation_id"], f"{path}.implementation_id"
        ),
        shape_vector=ShapeVector.from_obj(
            payload["shape_vector"], f"{path}.shape_vector"
        ),
        epochs=tuple(
            service_epoch_from_obj(item, f"{path}.epochs[{index}]")
            for index, item in enumerate(
                _array(payload["epochs"], f"{path}.epochs")
            )
        ),
    )


def service_entry_record_from_obj(
    value: object, path: str = "service_entry_record"
) -> DeviceServiceEntryRecord:
    payload = _object(value, path)
    _fields(payload, path, {"service_entry_id", "entry"})
    return DeviceServiceEntryRecord(
        service_entry_id=_string(
            payload["service_entry_id"], f"{path}.service_entry_id"
        ),
        entry=service_entry_from_obj(payload["entry"], f"{path}.entry"),
    )


def service_entry_evidence_from_obj(
    value: object, path: str = "service_entry_evidence"
) -> ServiceEntryEvidence:
    payload = _object(value, path)
    expected = {
        "service_entry_id",
        "source_selection",
        "source_record_sha256s",
        "residual_record_sha256",
        "support_envelope_sha256",
        "operating_envelope_sha256",
        "isolated_duration_ps",
        "uncertainty_bound",
    }
    _fields(payload, path, expected)
    return ServiceEntryEvidence(
        service_entry_id=_string(
            payload["service_entry_id"], f"{path}.service_entry_id"
        ),
        source_selection=_enum_value(
            ServiceEntrySourceSelection,
            payload["source_selection"],
            f"{path}.source_selection",
        ),
        source_record_sha256s=tuple(
            _sha256(item, f"{path}.source_record_sha256s[{index}]")
            for index, item in enumerate(
                _array(
                    payload["source_record_sha256s"],
                    f"{path}.source_record_sha256s",
                )
            )
        ),
        residual_record_sha256=_sha256(
            payload["residual_record_sha256"],
            f"{path}.residual_record_sha256",
        ),
        support_envelope_sha256=_sha256(
            payload["support_envelope_sha256"],
            f"{path}.support_envelope_sha256",
        ),
        operating_envelope_sha256=_sha256(
            payload["operating_envelope_sha256"],
            f"{path}.operating_envelope_sha256",
        ),
        isolated_duration_ps=_bounded_integer(
            payload["isolated_duration_ps"],
            f"{path}.isolated_duration_ps",
            nonnegative=True,
        ),
        uncertainty_bound=exact_rate_from_obj(
            payload["uncertainty_bound"], f"{path}.uncertainty_bound"
        ),
    )


def device_model_limits_from_obj(
    value: object, path: str = "model_limits"
) -> DeviceModelLimits:
    payload = _object(value, path)
    expected = {
        "max_shape_schemas",
        "max_shape_axes_per_schema",
        "max_resource_axes",
        "max_service_entries",
        "max_epochs_per_entry",
        "max_resident_entries",
    }
    _fields(payload, path, expected)
    return DeviceModelLimits(
        max_shape_schemas=_bounded_integer(
            payload["max_shape_schemas"],
            f"{path}.max_shape_schemas",
            positive=True,
        ),
        max_shape_axes_per_schema=_bounded_integer(
            payload["max_shape_axes_per_schema"],
            f"{path}.max_shape_axes_per_schema",
            positive=True,
        ),
        max_resource_axes=_bounded_integer(
            payload["max_resource_axes"],
            f"{path}.max_resource_axes",
            positive=True,
        ),
        max_service_entries=_bounded_integer(
            payload["max_service_entries"],
            f"{path}.max_service_entries",
            positive=True,
        ),
        max_epochs_per_entry=_bounded_integer(
            payload["max_epochs_per_entry"],
            f"{path}.max_epochs_per_entry",
            positive=True,
        ),
        max_resident_entries=_bounded_integer(
            payload["max_resident_entries"],
            f"{path}.max_resident_entries",
            positive=True,
        ),
    )


def interaction_contract_from_obj(
    value: object, path: str = "interaction_contract"
) -> ResourceInteractionContract:
    payload = _object(value, path)
    _fields(payload, path, {"interaction_law", "interaction_terms"})
    return ResourceInteractionContract(
        interaction_law=_string(
            payload["interaction_law"], f"{path}.interaction_law"
        ),
        interaction_terms=tuple(
            _array(payload["interaction_terms"], f"{path}.interaction_terms")
        ),
    )


def device_model_from_obj(
    value: object, path: str = "device_model"
) -> DeviceModel:
    payload = _object(value, path)
    expected = {
        "schema",
        "device_model_id",
        "device_kind_id",
        "acceptance_status",
        "target_basis",
        "device_identity_sha256",
        "operating_envelope_sha256",
        "support_envelope_sha256",
        "evidence_manifest_sha256",
        "fit_sha256",
        "expectations_commit",
        "dispatch_signature_sha256s",
        "shape_schemas",
        "implementation_selector_sha256",
        "collective_stage_selector_sha256",
        "resource_registry",
        "interaction_contract",
        "host_initiation_profile_sha256",
        "service_entries",
        "service_entry_evidence",
        "scalar_profile_table_sha256",
        "gpu_spec_sha256",
        "gpu_architecture_profile_sha256",
        "gpu_device_config_sha256",
        "validation_record_sha256",
        "validation_summary_sha256",
        "acceptance_bars_sha256",
        "model_limits",
    }
    _fields(payload, path, expected)
    _require_schema(payload["schema"], DEVICE_MODEL_SCHEMA, f"{path}.schema")
    return DeviceModel(
        device_model_id=_string(
            payload["device_model_id"], f"{path}.device_model_id"
        ),
        device_kind_id=_string(
            payload["device_kind_id"], f"{path}.device_kind_id"
        ),
        acceptance_status=_enum_value(
            DeviceModelAcceptanceStatus,
            payload["acceptance_status"],
            f"{path}.acceptance_status",
        ),
        target_basis=_enum_value(
            DeviceModelTargetBasis, payload["target_basis"], f"{path}.target_basis"
        ),
        device_identity_sha256=_sha256(
            payload["device_identity_sha256"], f"{path}.device_identity_sha256"
        ),
        operating_envelope_sha256=_sha256(
            payload["operating_envelope_sha256"],
            f"{path}.operating_envelope_sha256",
        ),
        support_envelope_sha256=_sha256(
            payload["support_envelope_sha256"],
            f"{path}.support_envelope_sha256",
        ),
        evidence_manifest_sha256=_sha256(
            payload["evidence_manifest_sha256"],
            f"{path}.evidence_manifest_sha256",
        ),
        fit_sha256=_sha256(payload["fit_sha256"], f"{path}.fit_sha256"),
        expectations_commit=_git_object_id(
            payload["expectations_commit"], f"{path}.expectations_commit"
        ),
        dispatch_signature_sha256s=tuple(
            _sha256(item, f"{path}.dispatch_signature_sha256s[{index}]")
            for index, item in enumerate(
                _array(
                    payload["dispatch_signature_sha256s"],
                    f"{path}.dispatch_signature_sha256s",
                )
            )
        ),
        shape_schemas=tuple(
            ShapeSchema.from_obj(item, f"{path}.shape_schemas[{index}]")
            for index, item in enumerate(
                _array(payload["shape_schemas"], f"{path}.shape_schemas")
            )
        ),
        implementation_selector_sha256=_sha256(
            payload["implementation_selector_sha256"],
            f"{path}.implementation_selector_sha256",
        ),
        collective_stage_selector_sha256=_optional_sha256(
            payload["collective_stage_selector_sha256"],
            f"{path}.collective_stage_selector_sha256",
        ),
        resource_registry=resource_registry_from_obj(
            payload["resource_registry"], f"{path}.resource_registry"
        ),
        interaction_contract=interaction_contract_from_obj(
            payload["interaction_contract"], f"{path}.interaction_contract"
        ),
        host_initiation_profile_sha256=_optional_sha256(
            payload["host_initiation_profile_sha256"],
            f"{path}.host_initiation_profile_sha256",
        ),
        service_entries=tuple(
            service_entry_record_from_obj(item, f"{path}.service_entries[{index}]")
            for index, item in enumerate(
                _array(payload["service_entries"], f"{path}.service_entries")
            )
        ),
        service_entry_evidence=tuple(
            service_entry_evidence_from_obj(
                item, f"{path}.service_entry_evidence[{index}]"
            )
            for index, item in enumerate(
                _array(
                    payload["service_entry_evidence"],
                    f"{path}.service_entry_evidence",
                )
            )
        ),
        scalar_profile_table_sha256=_optional_sha256(
            payload["scalar_profile_table_sha256"],
            f"{path}.scalar_profile_table_sha256",
        ),
        gpu_spec_sha256=_optional_sha256(
            payload["gpu_spec_sha256"], f"{path}.gpu_spec_sha256"
        ),
        gpu_architecture_profile_sha256=_optional_sha256(
            payload["gpu_architecture_profile_sha256"],
            f"{path}.gpu_architecture_profile_sha256",
        ),
        gpu_device_config_sha256=_optional_sha256(
            payload["gpu_device_config_sha256"],
            f"{path}.gpu_device_config_sha256",
        ),
        validation_record_sha256=_sha256(
            payload["validation_record_sha256"],
            f"{path}.validation_record_sha256",
        ),
        validation_summary_sha256=_sha256(
            payload["validation_summary_sha256"],
            f"{path}.validation_summary_sha256",
        ),
        acceptance_bars_sha256=_sha256(
            payload["acceptance_bars_sha256"],
            f"{path}.acceptance_bars_sha256",
        ),
        model_limits=device_model_limits_from_obj(
            payload["model_limits"], f"{path}.model_limits"
        ),
    )


def device_model_record_from_obj(
    value: object,
) -> DeviceModel | DeviceResourceRegistry | ShapeSchema:
    """Dispatch the schema-bearing component records frozen for Wave 1A."""

    payload: Mapping[str, Any] = _object(value, "record")
    schema = payload.get("schema")
    if schema == SHAPE_SCHEMA_SCHEMA:
        return ShapeSchema.from_obj(payload, "record")
    if schema == DEVICE_RESOURCE_REGISTRY_SCHEMA:
        return resource_registry_from_obj(payload, "record")
    if schema == DEVICE_MODEL_SCHEMA:
        return device_model_from_obj(payload, "record")
    raise ValueError(f"record.schema: unsupported compact device record {schema!r}")


TYPED_RECORD_READERS = {
    DEVICE_MODEL_SCHEMA: device_model_from_obj,
    DEVICE_RESOURCE_REGISTRY_SCHEMA: resource_registry_from_obj,
    SHAPE_SCHEMA_SCHEMA: ShapeSchema.from_obj,
}


def validate_typed_record(
    value: object,
) -> DeviceModel | DeviceResourceRegistry | ShapeSchema:
    """Validate one schema-bearing compact-model component record."""

    return device_model_record_from_obj(value)


__all__ = [
    "TYPED_RECORD_READERS",
    "device_model_from_obj",
    "device_model_limits_from_obj",
    "device_model_record_from_obj",
    "exact_rate_from_obj",
    "interaction_contract_from_obj",
    "resource_axis_from_obj",
    "resource_registry_from_obj",
    "resource_vector_from_obj",
    "service_entry_evidence_from_obj",
    "service_entry_from_obj",
    "service_entry_record_from_obj",
    "service_epoch_from_obj",
    "validate_typed_record",
]
