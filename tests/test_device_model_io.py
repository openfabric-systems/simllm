from __future__ import annotations

import subprocess
import sys

import pytest

from simllm.compute.device_model import (
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
    ShapeAxis,
    ShapeSchema,
    ShapeVector,
)
from simllm.compute.device_model_io import (
    device_model_from_obj,
    device_model_limits_from_obj,
    device_model_record_from_obj,
    interaction_contract_from_obj,
    resource_axis_from_obj,
    resource_registry_from_obj,
    resource_vector_from_obj,
    service_entry_evidence_from_obj,
    service_entry_from_obj,
    service_entry_record_from_obj,
)

REGISTRY = "a" * 64
ENVELOPE = "b" * 64
SIGNATURE = "c" * 64
SUPPORT = "d" * 64


def axis() -> DeviceResourceAxis:
    return DeviceResourceAxis(
        axis_id="sm-cycles",
        axis_class=ResourceAxisClass.THROUGHPUT,
        service_scope=ResourceServiceScope.DEVICE_INTERNAL,
        base_unit="cycles",
        clock_domain_id="sm-clock",
        capacity_source_id="sm80-clock-envelope",
        rate=ExactRate(4, 1),
        residency_capacity=None,
        exclusive_capacity=None,
    )


def registry() -> DeviceResourceRegistry:
    return DeviceResourceRegistry(
        device_kind_id="nvidia-a100-sm80",
        active_axis_ids=("sm-cycles",),
        axes=(axis(),),
    )


def vector() -> DeviceResourceVector:
    return DeviceResourceVector(
        registry_sha256=REGISTRY,
        device_kind_id="nvidia-a100-sm80",
        values=(1024,),
        known=(True,),
    )


def service_entry() -> DeviceServiceEntry:
    return DeviceServiceEntry(
        implementation_id="gemm-sm80",
        shape_vector=ShapeVector(shape_schema_id="gemm-v1", values=(128,)),
        epochs=(ServiceEpochDefinition(resource_vector=vector(), fixed_floor_ps=None),),
    )


def shape_schema() -> ShapeSchema:
    return ShapeSchema(
        shape_schema_id="gemm-v1",
        axes=(ShapeAxis(axis_id="m", unit="elements", minimum=1, maximum=1024),),
    )


def model() -> DeviceModel:
    record = DeviceServiceEntryRecord(
        service_entry_id="gemm-sm80-128",
        entry=service_entry(),
    )
    evidence = ServiceEntryEvidence(
        service_entry_id=record.service_entry_id,
        source_selection=ServiceEntrySourceSelection.ACCEL_SIM,
        source_record_sha256s=("1" * 64,),
        residual_record_sha256="2" * 64,
        support_envelope_sha256=SUPPORT,
        operating_envelope_sha256=ENVELOPE,
        isolated_duration_ps=4096,
        uncertainty_bound=ExactRate(1, 10),
    )
    return DeviceModel(
        device_model_id="a100-v1",
        device_kind_id="nvidia-a100-sm80",
        acceptance_status=DeviceModelAcceptanceStatus.CANDIDATE,
        target_basis=DeviceModelTargetBasis.TARGET_SILICON,
        device_identity_sha256="3" * 64,
        operating_envelope_sha256=ENVELOPE,
        support_envelope_sha256=SUPPORT,
        evidence_manifest_sha256="4" * 64,
        fit_sha256="5" * 64,
        expectations_commit="a" * 40,
        dispatch_signature_sha256s=(SIGNATURE,),
        shape_schemas=(shape_schema(),),
        implementation_selector_sha256="6" * 64,
        collective_stage_selector_sha256=None,
        resource_registry=registry(),
        interaction_contract=ResourceInteractionContract(
            interaction_law="independent-resource-v1", interaction_terms=()
        ),
        host_initiation_profile_sha256=None,
        service_entries=(record,),
        service_entry_evidence=(evidence,),
        scalar_profile_table_sha256=None,
        gpu_spec_sha256=None,
        gpu_architecture_profile_sha256=None,
        gpu_device_config_sha256=None,
        validation_record_sha256="7" * 64,
        validation_summary_sha256="8" * 64,
        acceptance_bars_sha256="9" * 64,
        model_limits=DeviceModelLimits(
            max_shape_schemas=1,
            max_shape_axes_per_schema=1,
            max_resource_axes=1,
            max_service_entries=1,
            max_epochs_per_entry=1,
            max_resident_entries=8,
        ),
    )


def test_strict_resource_component_round_trips() -> None:
    assert resource_axis_from_obj(axis().to_obj()) == axis()
    assert resource_registry_from_obj(registry().to_obj()) == registry()
    assert resource_vector_from_obj(vector().to_obj()) == vector()
    assert service_entry_from_obj(service_entry().to_obj()) == service_entry()
    contract = ResourceInteractionContract(
        interaction_law="independent-resource-v1", interaction_terms=()
    )
    assert interaction_contract_from_obj(contract.to_obj()) == contract
    assert service_entry_record_from_obj(model().service_entries[0].to_obj()) == (
        model().service_entries[0]
    )
    assert service_entry_evidence_from_obj(
        model().service_entry_evidence[0].to_obj()
    ) == model().service_entry_evidence[0]
    assert device_model_limits_from_obj(model().model_limits.to_obj()) == (
        model().model_limits
    )
    assert device_model_from_obj(model().to_obj()) == model()


def test_schema_dispatcher_accepts_only_schema_bearing_components() -> None:
    schema = shape_schema()
    assert device_model_record_from_obj(schema.to_obj()) == schema
    assert device_model_record_from_obj(registry().to_obj()) == registry()
    assert device_model_record_from_obj(model().to_obj()) == model()
    with pytest.raises(ValueError, match="unsupported compact device record"):
        device_model_record_from_obj({"schema": "future-model-v2"})


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload.update({"unknown": 1}), "unknown fields"),
        (lambda payload: payload.update({"axis_class": "latency"}), "expected one of"),
        (lambda payload: payload.update({"rate": {"numerator": 2, "denominator": 2}}), "reduced"),
        (lambda payload: payload.update({"clock_domain_id": 7}), "expected a string"),
    ],
)
def test_resource_axis_reader_rejects_malformed_objects(mutator, message: str) -> None:
    payload = axis().to_obj()
    mutator(payload)
    with pytest.raises(ValueError, match=message):
        resource_axis_from_obj(payload)


def test_resource_vector_reader_does_not_treat_integer_as_boolean() -> None:
    payload = vector().to_obj()
    payload["known"] = [1]
    with pytest.raises(ValueError, match="expected a boolean"):
        resource_vector_from_obj(payload)


def test_model_reader_rejects_architecture_derived_validation() -> None:
    payload = model().to_obj()
    payload["acceptance_status"] = "validated"
    payload["target_basis"] = "architecture-derived"
    with pytest.raises(ValueError, match="must remain candidate"):
        device_model_from_obj(payload)


def test_service_reader_rejects_empty_epochs_and_unknown_fields() -> None:
    payload = service_entry().to_obj()
    payload["epochs"] = []
    with pytest.raises(ValueError, match="must not be empty"):
        service_entry_from_obj(payload)

    payload = service_entry().to_obj()
    payload["start_time_ps"] = 0
    with pytest.raises(ValueError, match="unknown fields"):
        service_entry_from_obj(payload)


@pytest.mark.parametrize(
    "expectations_commit",
    ["A" * 40, "a" * 39, "g" * 40, "a" * 41, "a" * 63],
)
def test_model_reader_rejects_noncanonical_expectations_commit(
    expectations_commit: str,
) -> None:
    payload = model().to_obj()
    payload["expectations_commit"] = expectations_commit
    with pytest.raises(ValueError, match="40 or 64 lowercase hexadecimal"):
        device_model_from_obj(payload)


@pytest.mark.parametrize(
    ("digests", "message"),
    [
        ([], "must not be empty"),
        (["c" * 64, "c" * 64], "duplicates"),
        (["c" * 64, "b" * 64], "must be sorted"),
    ],
)
def test_model_reader_rejects_invalid_dispatch_digest_sets(
    digests: list[str], message: str
) -> None:
    payload = model().to_obj()
    payload["dispatch_signature_sha256s"] = digests
    with pytest.raises(ValueError, match=message):
        device_model_from_obj(payload)


def test_model_reader_rejects_unknown_fields_at_every_new_nesting() -> None:
    payload = model().to_obj()
    payload["raw_trace"] = []
    with pytest.raises(ValueError, match="unknown fields"):
        device_model_from_obj(payload)

    payload = model().to_obj()
    payload["service_entries"][0]["service_ns"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown fields"):
        device_model_from_obj(payload)

    payload = model().to_obj()
    payload["service_entry_evidence"][0]["profiler_rows"] = []  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown fields"):
        device_model_from_obj(payload)

    payload = model().to_obj()
    payload["model_limits"]["max_raw_rows"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown fields"):
        device_model_from_obj(payload)


def test_model_reader_rejects_evidence_ledger_and_source_violations() -> None:
    payload = model().to_obj()
    payload["service_entry_evidence"][0]["service_entry_id"] = "other"  # type: ignore[index]
    with pytest.raises(ValueError, match="sorted one-to-one"):
        device_model_from_obj(payload)

    payload = model().to_obj()
    payload["service_entry_evidence"][0]["source_selection"] = "estimated"  # type: ignore[index]
    with pytest.raises(ValueError, match="expected one of"):
        device_model_from_obj(payload)

    payload = model().to_obj()
    payload["service_entry_evidence"][0]["source_record_sha256s"] = []  # type: ignore[index]
    with pytest.raises(ValueError, match="must not be empty"):
        device_model_from_obj(payload)


def test_model_reader_rejects_shape_device_limit_and_reference_splices() -> None:
    payload = model().to_obj()
    payload["shape_schemas"] = []
    with pytest.raises(ValueError, match="must not be empty"):
        device_model_from_obj(payload)

    payload = model().to_obj()
    payload["resource_registry"]["device_kind_id"] = "amd-mi300x-gfx942"  # type: ignore[index]
    with pytest.raises(ValueError, match="model device kind"):
        device_model_from_obj(payload)

    payload = model().to_obj()
    payload["model_limits"]["max_resource_axes"] = 0  # type: ignore[index]
    with pytest.raises(ValueError, match="must be positive"):
        device_model_from_obj(payload)

    payload = model().to_obj()
    payload["gpu_spec_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        device_model_from_obj(payload)


def test_compute_model_import_does_not_initialize_offline_calibration() -> None:
    source = (
        "import sys; import simllm.compute.device_model_io; "
        "assert not any(name.startswith('simllm.calibration') for name in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
