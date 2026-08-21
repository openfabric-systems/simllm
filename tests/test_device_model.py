from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from simllm.compute.device_model import (
    SIGNED_128_MAX,
    SIGNED_128_MIN,
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
    ThroughputRateTimebase,
    validate_collective_stage_service_entries,
    validate_device_service_entries,
    validate_shape_schemas,
)

REGISTRY = "1" * 64
ENVELOPE = "2" * 64
SIGNATURE = "3" * 64
ENTRY = "4" * 64
SUPPORT_ENVELOPE = "5" * 64
EXPECTATIONS_COMMIT = "d" * 40


def shape_schema() -> ShapeSchema:
    return ShapeSchema(
        shape_schema_id="kernel-v1",
        axes=(ShapeAxis(axis_id="tokens", unit="tokens", minimum=1, maximum=4096),),
    )


def registry(*, throughput_numerator: int = 128) -> DeviceResourceRegistry:
    return DeviceResourceRegistry(
        device_kind_id="nvidia-a100-sm80",
        active_axis_ids=("copy-engine", "hbm-bytes", "resident-ctas"),
        axes=(
            DeviceResourceAxis(
                axis_id="copy-engine",
                axis_class=ResourceAxisClass.EXCLUSIVE,
                service_scope=ResourceServiceScope.DEVICE_INTERNAL,
                base_unit="slots",
                clock_domain_id=None,
                capacity_source_id="a100-copy-engines",
                rate=None,
                residency_capacity=None,
                exclusive_capacity=2,
            ),
            DeviceResourceAxis(
                axis_id="hbm-bytes",
                axis_class=ResourceAxisClass.THROUGHPUT,
                service_scope=ResourceServiceScope.DEVICE_INTERNAL,
                base_unit="bytes",
                clock_domain_id=None,
                capacity_source_id="a100-hbm-rate",
                rate=ExactRate(numerator=throughput_numerator, denominator=1),
                residency_capacity=None,
                exclusive_capacity=None,
            ),
            DeviceResourceAxis(
                axis_id="resident-ctas",
                axis_class=ResourceAxisClass.RESIDENCY,
                service_scope=ResourceServiceScope.DEVICE_INTERNAL,
                base_unit="ctas",
                clock_domain_id=None,
                capacity_source_id="a100-residency",
                rate=None,
                residency_capacity=108,
                exclusive_capacity=None,
            ),
        ),
    )


def vector(*, values: tuple[int, ...] = (1, 4096, 2)) -> DeviceResourceVector:
    return DeviceResourceVector(
        registry_sha256=REGISTRY,
        device_kind_id="nvidia-a100-sm80",
        values=values,
        known=(True, True, True),
    )


def entry(*, demand: DeviceResourceVector | None = None) -> DeviceServiceEntry:
    return DeviceServiceEntry(
        implementation_id="kernel-sm80",
        shape_vector=ShapeVector(shape_schema_id="kernel-v1", values=(128,)),
        epochs=(
            ServiceEpochDefinition(
                resource_vector=vector() if demand is None else demand,
                fixed_floor_ps=100,
            ),
        ),
    )


def model() -> DeviceModel:
    service_entry = DeviceServiceEntryRecord(
        service_entry_id="kernel-sm80-128",
        entry=entry(),
    )
    evidence = ServiceEntryEvidence(
        service_entry_id=service_entry.service_entry_id,
        source_selection=ServiceEntrySourceSelection.SILICON_FIT,
        source_record_sha256s=("a" * 64, "b" * 64),
        residual_record_sha256="c" * 64,
        support_envelope_sha256=SUPPORT_ENVELOPE,
        operating_envelope_sha256=ENVELOPE,
        isolated_duration_ps=1_000,
        uncertainty_bound=ExactRate(1, 20),
    )
    return DeviceModel(
        device_model_id="a100-v1",
        device_kind_id="nvidia-a100-sm80",
        acceptance_status=DeviceModelAcceptanceStatus.VALIDATED,
        target_basis=DeviceModelTargetBasis.TARGET_SILICON,
        device_identity_sha256="6" * 64,
        operating_envelope_sha256=ENVELOPE,
        support_envelope_sha256=SUPPORT_ENVELOPE,
        evidence_manifest_sha256="7" * 64,
        fit_sha256="8" * 64,
        expectations_commit=EXPECTATIONS_COMMIT,
        dispatch_signature_sha256s=(SIGNATURE,),
        shape_schemas=(shape_schema(),),
        implementation_selector_sha256="9" * 64,
        collective_stage_selector_sha256=None,
        resource_registry=registry(),
        interaction_contract=ResourceInteractionContract(
            interaction_law="independent-resource-v1", interaction_terms=()
        ),
        host_initiation_profile_sha256=None,
        service_entries=(service_entry,),
        service_entry_evidence=(evidence,),
        scalar_profile_table_sha256=None,
        gpu_spec_sha256=None,
        gpu_architecture_profile_sha256=None,
        gpu_device_config_sha256=None,
        validation_record_sha256="d" * 64,
        validation_summary_sha256="e" * 64,
        acceptance_bars_sha256="f" * 64,
        model_limits=DeviceModelLimits(
            max_shape_schemas=1,
            max_shape_axes_per_schema=1,
            max_resource_axes=3,
            max_service_entries=1,
            max_epochs_per_entry=1,
            max_resident_entries=32,
        ),
    )


def test_resource_axis_classes_require_exact_capacity_member() -> None:
    with pytest.raises(ValueError, match="throughput requires null"):
        replace(registry().axes[1], residency_capacity=1)
    with pytest.raises(ValueError, match="non-throughput axes reject a rate"):
        replace(registry().axes[2], rate=ExactRate(1, 1))
    with pytest.raises(ValueError, match="exclusive requires only"):
        replace(registry().axes[0], exclusive_capacity=None)
    with pytest.raises(ValueError, match="reject a clock domain"):
        replace(registry().axes[2], clock_domain_id="sm-clock")


def test_shape_schema_set_rejects_non_schema_members() -> None:
    with pytest.raises(TypeError, match="expected ShapeSchema"):
        validate_shape_schemas(("kernel-v1",))  # type: ignore[arg-type]


@pytest.mark.parametrize("value", (SIGNED_128_MIN - 1, SIGNED_128_MAX + 1))
def test_shape_bounds_and_values_reject_signed_128_overflow(value: int) -> None:
    axis = {
        "axis_id": "tokens",
        "unit": "tokens",
        "minimum": value,
        "maximum": value,
    }
    vector_value = {"shape_schema_id": "kernel-v1", "values": [value]}

    with pytest.raises(ValueError, match="signed 128-bit"):
        ShapeAxis(axis_id="tokens", unit="tokens", minimum=value, maximum=value)
    with pytest.raises(ValueError, match="signed 128-bit"):
        ShapeAxis.from_obj(axis)
    with pytest.raises(ValueError, match="signed 128-bit"):
        ShapeVector(shape_schema_id="kernel-v1", values=(value,))
    with pytest.raises(ValueError, match="signed 128-bit"):
        ShapeVector.from_obj(vector_value)


def test_throughput_clock_domain_selects_rate_timebase_not_demand_unit() -> None:
    wall_axis = registry().axes[1]
    cycle_axis = replace(wall_axis, clock_domain_id="hbm-clock")

    assert wall_axis.base_unit == cycle_axis.base_unit == "bytes"
    assert wall_axis.throughput_timebase is ThroughputRateTimebase.WALL_PS
    assert cycle_axis.throughput_timebase is ThroughputRateTimebase.DEVICE_CYCLE
    assert registry().axes[0].throughput_timebase is None


def test_exact_rate_is_reduced_and_bounded() -> None:
    assert ExactRate(0, 1).to_obj() == {"numerator": 0, "denominator": 1}
    with pytest.raises(ValueError, match="must be reduced"):
        ExactRate(2, 2)
    with pytest.raises(ValueError, match="signed 128-bit"):
        ExactRate(SIGNED_128_MAX + 1, 1)


def test_registry_requires_sorted_unique_active_subset() -> None:
    record = registry()
    assert tuple(axis.axis_id for axis in record.axes) == record.axis_ids

    with pytest.raises(ValueError, match="must be sorted"):
        replace(record, axes=tuple(reversed(record.axes)))
    with pytest.raises(ValueError, match="unknown axis IDs"):
        replace(record, active_axis_ids=("absent",))
    with pytest.raises(ValueError, match="duplicate"):
        replace(record, active_axis_ids=("copy-engine", "copy-engine"))


def test_resource_vector_rejects_negative_unknown_active_and_inactive_demand() -> None:
    record = registry()
    vector().validate_against(record, REGISTRY)

    with pytest.raises(ValueError, match="must be nonnegative"):
        replace(vector(), values=(1, -1, 2))
    with pytest.raises(ValueError, match="active-axis demand must be known"):
        replace(vector(), known=(True, False, True)).validate_against(record, REGISTRY)

    inactive_registry = replace(record, active_axis_ids=("hbm-bytes",))
    valid = replace(vector(), values=(0, 4096, 0), known=(False, True, False))
    valid.validate_against(inactive_registry, REGISTRY)
    with pytest.raises(ValueError, match="inactive axes require"):
        replace(valid, values=(1, 4096, 0)).validate_against(
            inactive_registry, REGISTRY
        )


def test_service_entry_validation_checks_shape_vector_and_capacity() -> None:
    validate_device_service_entries(
        registry=registry(),
        registry_sha256=REGISTRY,
        shape_schemas=(shape_schema(),),
        entries=(entry(),),
    )

    with pytest.raises(ValueError, match="outside 'tokens' domain"):
        validate_device_service_entries(
            registry=registry(),
            registry_sha256=REGISTRY,
            shape_schemas=(shape_schema(),),
            entries=(replace(entry(), shape_vector=ShapeVector("kernel-v1", (5000,))),),
        )
    with pytest.raises(ValueError, match="positive accepted demand requires"):
        validate_device_service_entries(
            registry=registry(throughput_numerator=0),
            registry_sha256=REGISTRY,
            shape_schemas=(shape_schema(),),
            entries=(entry(),),
        )
    with pytest.raises(ValueError, match="duplicate implementation and shape"):
        validate_device_service_entries(
            registry=registry(),
            registry_sha256=REGISTRY,
            shape_schemas=(shape_schema(),),
            entries=(entry(), entry()),
        )


def test_service_entry_requires_nonempty_epochs_and_nonnegative_floor() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        replace(entry(), epochs=())
    with pytest.raises(ValueError, match="must be nonnegative"):
        replace(entry().epochs[0], fixed_floor_ps=-1)


def test_interaction_contract_is_closed_and_empty() -> None:
    contract = ResourceInteractionContract(
        interaction_law="independent-resource-v1", interaction_terms=()
    )
    assert contract.to_obj()["interaction_terms"] == []
    with pytest.raises(ValueError, match="rejects interaction terms"):
        replace(contract, interaction_terms=({"kind": "pairwise"},))
    with pytest.raises(ValueError, match="only 'independent-resource-v1'"):
        ResourceInteractionContract(interaction_law="learned-v1", interaction_terms=())


def test_collective_stage_rejects_peer_port_and_data_mover_demand() -> None:
    peer_axis = DeviceResourceAxis(
        axis_id="peer-egress",
        axis_class=ResourceAxisClass.THROUGHPUT,
        service_scope=ResourceServiceScope.PEER_PORT,
        base_unit="bytes",
        clock_domain_id=None,
        capacity_source_id="nvlink",
        rate=ExactRate(1, 1),
        residency_capacity=None,
        exclusive_capacity=None,
    )
    peer_registry = DeviceResourceRegistry(
        device_kind_id="nvidia-a100-sm80",
        active_axis_ids=("peer-egress",),
        axes=(peer_axis,),
    )
    peer_entry = entry(
        demand=DeviceResourceVector(
            registry_sha256=REGISTRY,
            device_kind_id="nvidia-a100-sm80",
            values=(1,),
            known=(True,),
        )
    )
    with pytest.raises(ValueError, match="device-internal axes"):
        validate_collective_stage_service_entries(
            (ENTRY,), {ENTRY: peer_entry}, peer_registry, REGISTRY
        )


def test_complete_device_model_validates_registry_digest_explicitly() -> None:
    record = model()

    assert record.to_obj()["schema"] == "simllm-device-model-v1"
    assert record.declared_resource_registry_sha256 == REGISTRY
    record.validate_registry_sha256(REGISTRY)
    with pytest.raises(ValueError, match="does not match the selected registry"):
        record.validate_registry_sha256("0" * 64)


def test_device_model_enforces_candidate_architecture_derivation() -> None:
    record = model()
    with pytest.raises(ValueError, match="must remain candidate"):
        replace(record, target_basis=DeviceModelTargetBasis.ARCHITECTURE_DERIVED)

    candidate = replace(
        record,
        target_basis=DeviceModelTargetBasis.ARCHITECTURE_DERIVED,
        acceptance_status=DeviceModelAcceptanceStatus.CANDIDATE,
    )
    assert candidate.target_basis is DeviceModelTargetBasis.ARCHITECTURE_DERIVED


def test_device_model_entry_and_evidence_ledgers_are_sorted_one_to_one() -> None:
    first = model()
    first_entry = first.service_entries[0]
    first_evidence = first.service_entry_evidence[0]
    second_entry = DeviceServiceEntryRecord(
        service_entry_id="kernel-sm80-256",
        entry=replace(
            first_entry.entry,
            shape_vector=ShapeVector(shape_schema_id="kernel-v1", values=(256,)),
        ),
    )
    second_evidence = replace(
        first_evidence, service_entry_id=second_entry.service_entry_id
    )
    limits = replace(first.model_limits, max_service_entries=2)
    two = replace(
        first,
        service_entries=(first_entry, second_entry),
        service_entry_evidence=(first_evidence, second_evidence),
        model_limits=limits,
    )
    assert len(two.service_entries) == 2

    with pytest.raises(ValueError, match="must be sorted"):
        replace(two, service_entries=tuple(reversed(two.service_entries)))
    with pytest.raises(ValueError, match="sorted one-to-one"):
        replace(two, service_entry_evidence=(first_evidence,))
    with pytest.raises(ValueError, match="duplicate implementation and shape"):
        replace(
            two,
            service_entries=(
                first_entry,
                replace(second_entry, entry=first_entry.entry),
            ),
        )


def test_device_model_rejects_envelope_device_and_registry_splices() -> None:
    record = model()
    with pytest.raises(ValueError, match="support envelope"):
        replace(
            record,
            service_entry_evidence=(
                replace(
                    record.service_entry_evidence[0],
                    support_envelope_sha256="0" * 64,
                ),
            ),
        )
    with pytest.raises(ValueError, match="model device kind"):
        replace(record, device_kind_id="amd-mi300x-gfx942")

    epoch = record.service_entries[0].entry.epochs[0]
    changed_epoch = replace(
        epoch,
        resource_vector=replace(epoch.resource_vector, registry_sha256="0" * 64),
    )
    second_entry = DeviceServiceEntryRecord(
        service_entry_id="kernel-sm80-256",
        entry=replace(
            record.service_entries[0].entry,
            shape_vector=ShapeVector("kernel-v1", (256,)),
            epochs=(changed_epoch,),
        ),
    )
    second_evidence = replace(
        record.service_entry_evidence[0], service_entry_id=second_entry.service_entry_id
    )
    with pytest.raises(ValueError, match="one registry digest"):
        replace(
            record,
            service_entries=(record.service_entries[0], second_entry),
            service_entry_evidence=(
                record.service_entry_evidence[0],
                second_evidence,
            ),
            model_limits=replace(record.model_limits, max_service_entries=2),
        )


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "message"),
    [
        ("max_shape_schemas", 0, "must be positive"),
        ("max_shape_axes_per_schema", 0, "must be positive"),
        ("max_resource_axes", 2, "below inline count"),
        ("max_service_entries", 0, "must be positive"),
        ("max_epochs_per_entry", 0, "must be positive"),
        ("max_resident_entries", 0, "must be positive"),
    ],
)
def test_device_model_limits_are_positive_and_bound_inline_counts(
    limit_name: str, limit_value: int, message: str
) -> None:
    record = model()
    with pytest.raises(ValueError, match=message):
        replace(
            record,
            model_limits=replace(
                record.model_limits,
                **{limit_name: limit_value},
            ),
        )


def test_service_entry_evidence_has_closed_exact_source_contract() -> None:
    evidence = model().service_entry_evidence[0]
    assert evidence.to_obj()["source_selection"] == "silicon-fit"
    with pytest.raises(ValueError, match="must not be empty"):
        replace(evidence, source_record_sha256s=())
    with pytest.raises(ValueError, match="must be sorted"):
        replace(evidence, source_record_sha256s=tuple(reversed(evidence.source_record_sha256s)))
    with pytest.raises(TypeError, match="ServiceEntrySourceSelection"):
        replace(evidence, source_selection="silicon")  # type: ignore[arg-type]


def test_device_model_components_are_immutable() -> None:
    record = registry()
    with pytest.raises(FrozenInstanceError):
        record.device_kind_id = "other"  # type: ignore[misc]
