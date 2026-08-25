"""Strict dispatch, capture-binding, and resolved-binding records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from simllm.compute.device_model import (
    AnalyticalImplementationRef,
    BinaryImplementationRef,
    ImplementationRef,
    ShapeVector,
    _array,
    _fields,
    _integer,
    _object,
    _optional_sha256,
    _require_schema,
    _sha256,
    _string,
    implementation_ref_from_obj,
)

TYPED_DISPATCH_TRAIT_SCHEMA = "simllm-typed-dispatch-trait-v1"
DISPATCH_SIGNATURE_SCHEMA = "simllm-dispatch-signature-v1"
DEVICE_DISPATCH_CONTEXT_SCHEMA = "simllm-device-dispatch-context-v1"
RESOLVED_OPERATION_BINDING_SET_SCHEMA = (
    "simllm-resolved-operation-service-binding-set-v1"
)
RESOLVED_COLLECTIVE_STAGE_SET_SCHEMA = (
    "simllm-resolved-collective-device-stage-set-v1"
)
RESOLVED_DEVICE_BINDING_CLOSURE_SCHEMA = "simllm-resolved-device-binding-closure-v1"


def _check_tuple(value: object, path: str) -> tuple[Any, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{path}: in-memory contract requires a tuple")
    return value


def _unique(values: tuple[object, ...], path: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{path}: duplicate values are not allowed")


def _sorted_unique(values: tuple[object, ...], path: str) -> None:
    _unique(values, path)
    if values != tuple(sorted(values)):
        raise ValueError(f"{path}: values must be sorted")


class DispatchTraitValueType(str, Enum):
    INTEGER = "integer"
    STRING = "string"
    BOOLEAN = "boolean"


DispatchTraitValue = int | str | bool


@dataclass(frozen=True, slots=True)
class TypedDispatchTrait:
    """One typed compatibility trait in a dispatch signature."""

    trait_id: str
    value_type: DispatchTraitValueType
    value: DispatchTraitValue

    def __post_init__(self) -> None:
        _string(self.trait_id, "TypedDispatchTrait.trait_id")
        if not isinstance(self.value_type, DispatchTraitValueType):
            raise TypeError("TypedDispatchTrait.value_type: expected DispatchTraitValueType")
        valid = {
            DispatchTraitValueType.INTEGER: type(self.value) is int,
            DispatchTraitValueType.STRING: isinstance(self.value, str),
            DispatchTraitValueType.BOOLEAN: type(self.value) is bool,
        }[self.value_type]
        if not valid:
            raise TypeError(
                "TypedDispatchTrait.value: does not match "
                f"value_type {self.value_type.value!r}"
            )

    def to_obj(self) -> dict[str, Any]:
        return {
            "trait_id": self.trait_id,
            "value_type": self.value_type.value,
            "value": self.value,
        }

    @classmethod
    def from_obj(
        cls, value: object, path: str = "typed_dispatch_trait"
    ) -> TypedDispatchTrait:
        payload = _object(value, path)
        _fields(payload, path, {"trait_id", "value_type", "value"})
        raw_type = _string(payload["value_type"], f"{path}.value_type")
        try:
            value_type = DispatchTraitValueType(raw_type)
        except ValueError as exc:
            raise ValueError(
                f"{path}.value_type: expected 'integer', 'string', or 'boolean'"
            ) from exc
        return cls(
            trait_id=_string(payload["trait_id"], f"{path}.trait_id"),
            value_type=value_type,
            value=payload["value"],
        )


def _validate_traits(
    traits: tuple[TypedDispatchTrait, ...], path: str
) -> tuple[TypedDispatchTrait, ...]:
    _check_tuple(traits, path)
    for index, trait in enumerate(traits):
        if not isinstance(trait, TypedDispatchTrait):
            raise TypeError(f"{path}[{index}]: expected TypedDispatchTrait")
    _sorted_unique(tuple(trait.trait_id for trait in traits), path)
    return traits


@dataclass(frozen=True, slots=True)
class DispatchSignature:
    """The frozen software, ISA, numeric, and layout dispatch envelope."""

    framework_id: str
    framework_version: str
    backend_id: str
    backend_version: str
    kernel_library_id: str
    kernel_library_version: str
    algorithm_policy_id: str
    device_isa: str
    numeric_traits: tuple[TypedDispatchTrait, ...]
    layout_traits: tuple[TypedDispatchTrait, ...]
    schema: str = DISPATCH_SIGNATURE_SCHEMA

    def __post_init__(self) -> None:
        _require_schema(self.schema, DISPATCH_SIGNATURE_SCHEMA, "DispatchSignature.schema")
        for name in (
            "framework_id",
            "framework_version",
            "backend_id",
            "backend_version",
            "kernel_library_id",
            "kernel_library_version",
            "algorithm_policy_id",
            "device_isa",
        ):
            _string(getattr(self, name), f"DispatchSignature.{name}")
        _validate_traits(self.numeric_traits, "DispatchSignature.numeric_traits")
        _validate_traits(self.layout_traits, "DispatchSignature.layout_traits")

    def to_obj(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "framework_id": self.framework_id,
            "framework_version": self.framework_version,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "kernel_library_id": self.kernel_library_id,
            "kernel_library_version": self.kernel_library_version,
            "algorithm_policy_id": self.algorithm_policy_id,
            "device_isa": self.device_isa,
            "numeric_traits": [trait.to_obj() for trait in self.numeric_traits],
            "layout_traits": [trait.to_obj() for trait in self.layout_traits],
        }

    @classmethod
    def from_obj(cls, value: object, path: str = "dispatch_signature") -> DispatchSignature:
        payload = _object(value, path)
        expected = {
            "schema",
            "framework_id",
            "framework_version",
            "backend_id",
            "backend_version",
            "kernel_library_id",
            "kernel_library_version",
            "algorithm_policy_id",
            "device_isa",
            "numeric_traits",
            "layout_traits",
        }
        _fields(payload, path, expected)
        _require_schema(payload["schema"], DISPATCH_SIGNATURE_SCHEMA, f"{path}.schema")
        kwargs = {
            name: _string(payload[name], f"{path}.{name}")
            for name in expected
            if name not in {"schema", "numeric_traits", "layout_traits"}
        }
        return cls(
            **kwargs,
            numeric_traits=tuple(
                TypedDispatchTrait.from_obj(item, f"{path}.numeric_traits[{index}]")
                for index, item in enumerate(
                    _array(payload["numeric_traits"], f"{path}.numeric_traits")
                )
            ),
            layout_traits=tuple(
                TypedDispatchTrait.from_obj(item, f"{path}.layout_traits[{index}]")
                for index, item in enumerate(
                    _array(payload["layout_traits"], f"{path}.layout_traits")
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class RankDeviceAssignment:
    rank: int
    device_instance_id: str

    def __post_init__(self) -> None:
        _integer(self.rank, "RankDeviceAssignment.rank", nonnegative=True)
        _string(self.device_instance_id, "RankDeviceAssignment.device_instance_id")

    def to_obj(self) -> dict[str, Any]:
        return {"rank": self.rank, "device_instance_id": self.device_instance_id}

    @classmethod
    def from_obj(cls, value: object, path: str) -> RankDeviceAssignment:
        payload = _object(value, path)
        _fields(payload, path, {"rank", "device_instance_id"})
        return cls(
            rank=_integer(payload["rank"], f"{path}.rank", nonnegative=True),
            device_instance_id=_string(
                payload["device_instance_id"], f"{path}.device_instance_id"
            ),
        )


@dataclass(frozen=True, slots=True)
class SelectedDeviceModel:
    device_instance_id: str
    device_model_id: str
    device_model_sha256: str
    dispatch_signature_sha256: str

    def __post_init__(self) -> None:
        _string(self.device_instance_id, "SelectedDeviceModel.device_instance_id")
        _string(self.device_model_id, "SelectedDeviceModel.device_model_id")
        _sha256(self.device_model_sha256, "SelectedDeviceModel.device_model_sha256")
        _sha256(
            self.dispatch_signature_sha256,
            "SelectedDeviceModel.dispatch_signature_sha256",
        )

    def to_obj(self) -> dict[str, Any]:
        return {
            "device_instance_id": self.device_instance_id,
            "device_model_id": self.device_model_id,
            "device_model_sha256": self.device_model_sha256,
            "dispatch_signature_sha256": self.dispatch_signature_sha256,
        }

    @classmethod
    def from_obj(cls, value: object, path: str) -> SelectedDeviceModel:
        payload = _object(value, path)
        expected = {
            "device_instance_id",
            "device_model_id",
            "device_model_sha256",
            "dispatch_signature_sha256",
        }
        _fields(payload, path, expected)
        return cls(
            device_instance_id=_string(
                payload["device_instance_id"], f"{path}.device_instance_id"
            ),
            device_model_id=_string(
                payload["device_model_id"], f"{path}.device_model_id"
            ),
            device_model_sha256=_sha256(
                payload["device_model_sha256"], f"{path}.device_model_sha256"
            ),
            dispatch_signature_sha256=_sha256(
                payload["dispatch_signature_sha256"],
                f"{path}.dispatch_signature_sha256",
            ),
        )


@dataclass(frozen=True, slots=True)
class DeviceDispatchContext:
    instance_graph_sha256: str
    rank_device_assignments: tuple[RankDeviceAssignment, ...]
    selected_device_models: tuple[SelectedDeviceModel, ...]
    schema: str = DEVICE_DISPATCH_CONTEXT_SCHEMA

    def __post_init__(self) -> None:
        _require_schema(
            self.schema, DEVICE_DISPATCH_CONTEXT_SCHEMA, "DeviceDispatchContext.schema"
        )
        _sha256(
            self.instance_graph_sha256, "DeviceDispatchContext.instance_graph_sha256"
        )
        _check_tuple(
            self.rank_device_assignments,
            "DeviceDispatchContext.rank_device_assignments",
        )
        _check_tuple(
            self.selected_device_models,
            "DeviceDispatchContext.selected_device_models",
        )
        for index, assignment in enumerate(self.rank_device_assignments):
            if not isinstance(assignment, RankDeviceAssignment):
                raise TypeError(
                    f"DeviceDispatchContext.rank_device_assignments[{index}]: "
                    "expected RankDeviceAssignment"
                )
        for index, selection in enumerate(self.selected_device_models):
            if not isinstance(selection, SelectedDeviceModel):
                raise TypeError(
                    f"DeviceDispatchContext.selected_device_models[{index}]: "
                    "expected SelectedDeviceModel"
                )
        ranks = tuple(assignment.rank for assignment in self.rank_device_assignments)
        _sorted_unique(ranks, "DeviceDispatchContext.rank_device_assignments")
        selected_ids = tuple(
            selection.device_instance_id for selection in self.selected_device_models
        )
        _sorted_unique(selected_ids, "DeviceDispatchContext.selected_device_models")
        assigned_ids = {item.device_instance_id for item in self.rank_device_assignments}
        if set(selected_ids) != assigned_ids:
            raise ValueError(
                "DeviceDispatchContext.selected_device_models: must cover exactly "
                "the assigned device instances"
            )

    def to_obj(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "instance_graph_sha256": self.instance_graph_sha256,
            "rank_device_assignments": [
                assignment.to_obj() for assignment in self.rank_device_assignments
            ],
            "selected_device_models": [
                selection.to_obj() for selection in self.selected_device_models
            ],
        }

    @classmethod
    def from_obj(
        cls, value: object, path: str = "device_dispatch_context"
    ) -> DeviceDispatchContext:
        payload = _object(value, path)
        expected = {
            "schema",
            "instance_graph_sha256",
            "rank_device_assignments",
            "selected_device_models",
        }
        _fields(payload, path, expected)
        _require_schema(payload["schema"], DEVICE_DISPATCH_CONTEXT_SCHEMA, f"{path}.schema")
        return cls(
            instance_graph_sha256=_sha256(
                payload["instance_graph_sha256"], f"{path}.instance_graph_sha256"
            ),
            rank_device_assignments=tuple(
                RankDeviceAssignment.from_obj(
                    item, f"{path}.rank_device_assignments[{index}]"
                )
                for index, item in enumerate(
                    _array(
                        payload["rank_device_assignments"],
                        f"{path}.rank_device_assignments",
                    )
                )
            ),
            selected_device_models=tuple(
                SelectedDeviceModel.from_obj(
                    item, f"{path}.selected_device_models[{index}]"
                )
                for index, item in enumerate(
                    _array(
                        payload["selected_device_models"],
                        f"{path}.selected_device_models",
                    )
                )
            ),
        )

    def validate_graph_ranks(self, participant_ranks: Iterable[int]) -> None:
        expected = tuple(sorted(set(participant_ranks)))
        actual = tuple(item.rank for item in self.rank_device_assignments)
        if actual != expected:
            raise ValueError(
                "DeviceDispatchContext.rank_device_assignments: expected total graph "
                f"rank coverage {expected}, got {actual}"
            )

    def model_by_device(self) -> dict[str, SelectedDeviceModel]:
        return {item.device_instance_id: item for item in self.selected_device_models}

    def device_for_rank(self, rank: int) -> str:
        for assignment in self.rank_device_assignments:
            if assignment.rank == rank:
                return assignment.device_instance_id
        raise ValueError(f"rank {rank} is not assigned in the dispatch context")


@dataclass(frozen=True, slots=True)
class OperationImplementationBinding:
    instance_graph_sha256: str
    operation_id: str
    launch_ordinal: int
    implementation_ref: BinaryImplementationRef
    shape_vector: ShapeVector

    def __post_init__(self) -> None:
        _sha256(self.instance_graph_sha256, "OperationImplementationBinding.instance_graph_sha256")
        _string(self.operation_id, "OperationImplementationBinding.operation_id")
        _integer(
            self.launch_ordinal,
            "OperationImplementationBinding.launch_ordinal",
            nonnegative=True,
        )
        if not isinstance(self.implementation_ref, BinaryImplementationRef):
            raise TypeError(
                "OperationImplementationBinding.implementation_ref: observed binding "
                "requires a binary implementation"
            )
        if not isinstance(self.shape_vector, ShapeVector):
            raise TypeError("OperationImplementationBinding.shape_vector: expected ShapeVector")

    @property
    def key(self) -> tuple[str, str, int]:
        return self.instance_graph_sha256, self.operation_id, self.launch_ordinal

    def to_obj(self) -> dict[str, Any]:
        return {
            "instance_graph_sha256": self.instance_graph_sha256,
            "operation_id": self.operation_id,
            "launch_ordinal": self.launch_ordinal,
            "implementation_ref": self.implementation_ref.to_obj(),
            "shape_vector": self.shape_vector.to_obj(),
        }

    @classmethod
    def from_obj(
        cls, value: object, path: str = "operation_implementation_binding"
    ) -> OperationImplementationBinding:
        payload = _object(value, path)
        expected = {
            "instance_graph_sha256",
            "operation_id",
            "launch_ordinal",
            "implementation_ref",
            "shape_vector",
        }
        _fields(payload, path, expected)
        implementation_ref = implementation_ref_from_obj(
            payload["implementation_ref"], f"{path}.implementation_ref"
        )
        if not isinstance(implementation_ref, BinaryImplementationRef):
            raise TypeError(f"{path}.implementation_ref: observed binding requires binary kind")
        return cls(
            instance_graph_sha256=_sha256(
                payload["instance_graph_sha256"], f"{path}.instance_graph_sha256"
            ),
            operation_id=_string(payload["operation_id"], f"{path}.operation_id"),
            launch_ordinal=_integer(
                payload["launch_ordinal"], f"{path}.launch_ordinal", nonnegative=True
            ),
            implementation_ref=implementation_ref,
            shape_vector=ShapeVector.from_obj(payload["shape_vector"], f"{path}.shape_vector"),
        )


@dataclass(frozen=True, slots=True)
class CollectiveDeviceStageBinding:
    instance_graph_sha256: str
    collective_operation_id: str
    collective_plan_integrity_sha256: str
    rank: int
    launch_ordinal: int
    implementation_ref: BinaryImplementationRef
    shape_vector: ShapeVector

    def __post_init__(self) -> None:
        _sha256(self.instance_graph_sha256, "CollectiveDeviceStageBinding.instance_graph_sha256")
        _string(self.collective_operation_id, "CollectiveDeviceStageBinding.collective_operation_id")
        _sha256(
            self.collective_plan_integrity_sha256,
            "CollectiveDeviceStageBinding.collective_plan_integrity_sha256",
        )
        _integer(self.rank, "CollectiveDeviceStageBinding.rank", nonnegative=True)
        _integer(
            self.launch_ordinal,
            "CollectiveDeviceStageBinding.launch_ordinal",
            nonnegative=True,
        )
        if not isinstance(self.implementation_ref, BinaryImplementationRef):
            raise TypeError(
                "CollectiveDeviceStageBinding.implementation_ref: observed binding "
                "requires a binary implementation"
            )
        if not isinstance(self.shape_vector, ShapeVector):
            raise TypeError("CollectiveDeviceStageBinding.shape_vector: expected ShapeVector")

    @property
    def key(self) -> tuple[str, str, str, int, int]:
        return (
            self.instance_graph_sha256,
            self.collective_operation_id,
            self.collective_plan_integrity_sha256,
            self.rank,
            self.launch_ordinal,
        )

    def to_obj(self) -> dict[str, Any]:
        return {
            "instance_graph_sha256": self.instance_graph_sha256,
            "collective_operation_id": self.collective_operation_id,
            "collective_plan_integrity_sha256": self.collective_plan_integrity_sha256,
            "rank": self.rank,
            "launch_ordinal": self.launch_ordinal,
            "implementation_ref": self.implementation_ref.to_obj(),
            "shape_vector": self.shape_vector.to_obj(),
        }

    @classmethod
    def from_obj(
        cls, value: object, path: str = "collective_device_stage_binding"
    ) -> CollectiveDeviceStageBinding:
        payload = _object(value, path)
        expected = {
            "instance_graph_sha256",
            "collective_operation_id",
            "collective_plan_integrity_sha256",
            "rank",
            "launch_ordinal",
            "implementation_ref",
            "shape_vector",
        }
        _fields(payload, path, expected)
        implementation_ref = implementation_ref_from_obj(
            payload["implementation_ref"], f"{path}.implementation_ref"
        )
        if not isinstance(implementation_ref, BinaryImplementationRef):
            raise TypeError(f"{path}.implementation_ref: observed binding requires binary kind")
        return cls(
            instance_graph_sha256=_sha256(
                payload["instance_graph_sha256"], f"{path}.instance_graph_sha256"
            ),
            collective_operation_id=_string(
                payload["collective_operation_id"], f"{path}.collective_operation_id"
            ),
            collective_plan_integrity_sha256=_sha256(
                payload["collective_plan_integrity_sha256"],
                f"{path}.collective_plan_integrity_sha256",
            ),
            rank=_integer(payload["rank"], f"{path}.rank", nonnegative=True),
            launch_ordinal=_integer(
                payload["launch_ordinal"], f"{path}.launch_ordinal", nonnegative=True
            ),
            implementation_ref=implementation_ref,
            shape_vector=ShapeVector.from_obj(payload["shape_vector"], f"{path}.shape_vector"),
        )


class ResolutionSource(str, Enum):
    OBSERVED_BINDING = "observed-binding"
    SELECTOR = "selector"


def _resolution_source(value: object, path: str) -> ResolutionSource:
    raw = _string(value, path)
    try:
        return ResolutionSource(raw)
    except ValueError as exc:
        raise ValueError(f"{path}: expected 'observed-binding' or 'selector'") from exc


@dataclass(frozen=True, slots=True)
class ResolvedOperationServiceBinding:
    instance_graph_sha256: str
    operation_id: str
    launch_ordinal: int
    device_instance_id: str
    device_model_sha256: str
    semantic_key: str
    shape_vector: ShapeVector
    implementation_ref: ImplementationRef
    service_entry_id: str
    resolution_source: ResolutionSource
    observed_implementation_binding_sha256: str | None

    def __post_init__(self) -> None:
        _sha256(self.instance_graph_sha256, "ResolvedOperationServiceBinding.instance_graph_sha256")
        _string(self.operation_id, "ResolvedOperationServiceBinding.operation_id")
        _integer(
            self.launch_ordinal,
            "ResolvedOperationServiceBinding.launch_ordinal",
            nonnegative=True,
        )
        _string(self.device_instance_id, "ResolvedOperationServiceBinding.device_instance_id")
        _sha256(self.device_model_sha256, "ResolvedOperationServiceBinding.device_model_sha256")
        _string(self.semantic_key, "ResolvedOperationServiceBinding.semantic_key")
        if not isinstance(self.shape_vector, ShapeVector):
            raise TypeError("ResolvedOperationServiceBinding.shape_vector: expected ShapeVector")
        if not isinstance(
            self.implementation_ref,
            BinaryImplementationRef | AnalyticalImplementationRef,
        ):
            raise TypeError(
                "ResolvedOperationServiceBinding.implementation_ref: expected ImplementationRef"
            )
        _string(self.service_entry_id, "ResolvedOperationServiceBinding.service_entry_id")
        if not isinstance(self.resolution_source, ResolutionSource):
            raise TypeError(
                "ResolvedOperationServiceBinding.resolution_source: expected ResolutionSource"
            )
        _optional_sha256(
            self.observed_implementation_binding_sha256,
            "ResolvedOperationServiceBinding.observed_implementation_binding_sha256",
        )
        has_observed = self.observed_implementation_binding_sha256 is not None
        if has_observed != (self.resolution_source is ResolutionSource.OBSERVED_BINDING):
            raise ValueError(
                "ResolvedOperationServiceBinding: observed-binding source requires a "
                "non-null observed binding hash and selector requires null"
            )

    @property
    def key(self) -> tuple[str, str, int]:
        return self.instance_graph_sha256, self.operation_id, self.launch_ordinal

    def to_obj(self) -> dict[str, Any]:
        return {
            "instance_graph_sha256": self.instance_graph_sha256,
            "operation_id": self.operation_id,
            "launch_ordinal": self.launch_ordinal,
            "device_instance_id": self.device_instance_id,
            "device_model_sha256": self.device_model_sha256,
            "semantic_key": self.semantic_key,
            "shape_vector": self.shape_vector.to_obj(),
            "implementation_ref": self.implementation_ref.to_obj(),
            "service_entry_id": self.service_entry_id,
            "resolution_source": self.resolution_source.value,
            "observed_implementation_binding_sha256": (
                self.observed_implementation_binding_sha256
            ),
        }

    @classmethod
    def from_obj(
        cls, value: object, path: str = "resolved_operation_service_binding"
    ) -> ResolvedOperationServiceBinding:
        payload = _object(value, path)
        expected = {
            "instance_graph_sha256",
            "operation_id",
            "launch_ordinal",
            "device_instance_id",
            "device_model_sha256",
            "semantic_key",
            "shape_vector",
            "implementation_ref",
            "service_entry_id",
            "resolution_source",
            "observed_implementation_binding_sha256",
        }
        _fields(payload, path, expected)
        return cls(
            instance_graph_sha256=_sha256(
                payload["instance_graph_sha256"], f"{path}.instance_graph_sha256"
            ),
            operation_id=_string(payload["operation_id"], f"{path}.operation_id"),
            launch_ordinal=_integer(
                payload["launch_ordinal"], f"{path}.launch_ordinal", nonnegative=True
            ),
            device_instance_id=_string(
                payload["device_instance_id"], f"{path}.device_instance_id"
            ),
            device_model_sha256=_sha256(
                payload["device_model_sha256"], f"{path}.device_model_sha256"
            ),
            semantic_key=_string(payload["semantic_key"], f"{path}.semantic_key"),
            shape_vector=ShapeVector.from_obj(payload["shape_vector"], f"{path}.shape_vector"),
            implementation_ref=implementation_ref_from_obj(
                payload["implementation_ref"], f"{path}.implementation_ref"
            ),
            service_entry_id=_string(
                payload["service_entry_id"], f"{path}.service_entry_id"
            ),
            resolution_source=_resolution_source(
                payload["resolution_source"], f"{path}.resolution_source"
            ),
            observed_implementation_binding_sha256=_optional_sha256(
                payload["observed_implementation_binding_sha256"],
                f"{path}.observed_implementation_binding_sha256",
            ),
        )


@dataclass(frozen=True, slots=True)
class ResolvedOperationServiceBindingSet:
    instance_graph_sha256: str
    dispatch_context_sha256: str
    bindings: tuple[ResolvedOperationServiceBinding, ...]
    schema: str = RESOLVED_OPERATION_BINDING_SET_SCHEMA

    def __post_init__(self) -> None:
        _require_schema(
            self.schema,
            RESOLVED_OPERATION_BINDING_SET_SCHEMA,
            "ResolvedOperationServiceBindingSet.schema",
        )
        _sha256(
            self.instance_graph_sha256,
            "ResolvedOperationServiceBindingSet.instance_graph_sha256",
        )
        _sha256(
            self.dispatch_context_sha256,
            "ResolvedOperationServiceBindingSet.dispatch_context_sha256",
        )
        _check_tuple(self.bindings, "ResolvedOperationServiceBindingSet.bindings")
        keys: list[tuple[str, str, int]] = []
        for index, binding in enumerate(self.bindings):
            if not isinstance(binding, ResolvedOperationServiceBinding):
                raise TypeError(
                    f"ResolvedOperationServiceBindingSet.bindings[{index}]: "
                    "expected ResolvedOperationServiceBinding"
            )
            if binding.instance_graph_sha256 != self.instance_graph_sha256:
                raise ValueError(
                    f"ResolvedOperationServiceBindingSet.bindings[{index}]: graph splice"
                )
            keys.append(binding.key)
        _unique(tuple(keys), "ResolvedOperationServiceBindingSet.bindings")

    def to_obj(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "instance_graph_sha256": self.instance_graph_sha256,
            "dispatch_context_sha256": self.dispatch_context_sha256,
            "bindings": [binding.to_obj() for binding in self.bindings],
        }

    @classmethod
    def from_obj(
        cls, value: object, path: str = "resolved_operation_service_binding_set"
    ) -> ResolvedOperationServiceBindingSet:
        payload = _object(value, path)
        expected = {"schema", "instance_graph_sha256", "dispatch_context_sha256", "bindings"}
        _fields(payload, path, expected)
        _require_schema(
            payload["schema"], RESOLVED_OPERATION_BINDING_SET_SCHEMA, f"{path}.schema"
        )
        return cls(
            instance_graph_sha256=_sha256(
                payload["instance_graph_sha256"], f"{path}.instance_graph_sha256"
            ),
            dispatch_context_sha256=_sha256(
                payload["dispatch_context_sha256"], f"{path}.dispatch_context_sha256"
            ),
            bindings=tuple(
                ResolvedOperationServiceBinding.from_obj(
                    item, f"{path}.bindings[{index}]"
                )
                for index, item in enumerate(
                    _array(payload["bindings"], f"{path}.bindings")
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class ResolvedCollectiveDeviceStage:
    instance_graph_sha256: str
    collective_operation_id: str
    collective_plan_integrity_sha256: str
    rank: int
    launch_ordinal: int
    device_instance_id: str
    device_model_sha256: str
    implementation_ref: ImplementationRef
    shape_vector: ShapeVector
    service_entry_id: str
    resolution_source: ResolutionSource = ResolutionSource.SELECTOR

    def __post_init__(self) -> None:
        _sha256(self.instance_graph_sha256, "ResolvedCollectiveDeviceStage.instance_graph_sha256")
        _string(self.collective_operation_id, "ResolvedCollectiveDeviceStage.collective_operation_id")
        _sha256(
            self.collective_plan_integrity_sha256,
            "ResolvedCollectiveDeviceStage.collective_plan_integrity_sha256",
        )
        _integer(self.rank, "ResolvedCollectiveDeviceStage.rank", nonnegative=True)
        _integer(
            self.launch_ordinal,
            "ResolvedCollectiveDeviceStage.launch_ordinal",
            nonnegative=True,
        )
        _string(self.device_instance_id, "ResolvedCollectiveDeviceStage.device_instance_id")
        _sha256(self.device_model_sha256, "ResolvedCollectiveDeviceStage.device_model_sha256")
        if not isinstance(
            self.implementation_ref,
            BinaryImplementationRef | AnalyticalImplementationRef,
        ):
            raise TypeError(
                "ResolvedCollectiveDeviceStage.implementation_ref: expected ImplementationRef"
            )
        if not isinstance(self.shape_vector, ShapeVector):
            raise TypeError("ResolvedCollectiveDeviceStage.shape_vector: expected ShapeVector")
        _string(self.service_entry_id, "ResolvedCollectiveDeviceStage.service_entry_id")
        if self.resolution_source is not ResolutionSource.SELECTOR:
            raise ValueError(
                "ResolvedCollectiveDeviceStage.resolution_source: expected selector"
            )

    @property
    def key(self) -> tuple[str, str, str, int, int]:
        return (
            self.instance_graph_sha256,
            self.collective_operation_id,
            self.collective_plan_integrity_sha256,
            self.rank,
            self.launch_ordinal,
        )

    @property
    def frontier_key(self) -> tuple[str, str, int]:
        return (
            self.collective_operation_id,
            self.collective_plan_integrity_sha256,
            self.rank,
        )

    def to_obj(self) -> dict[str, Any]:
        return {
            "instance_graph_sha256": self.instance_graph_sha256,
            "collective_operation_id": self.collective_operation_id,
            "collective_plan_integrity_sha256": self.collective_plan_integrity_sha256,
            "rank": self.rank,
            "launch_ordinal": self.launch_ordinal,
            "device_instance_id": self.device_instance_id,
            "device_model_sha256": self.device_model_sha256,
            "implementation_ref": self.implementation_ref.to_obj(),
            "shape_vector": self.shape_vector.to_obj(),
            "service_entry_id": self.service_entry_id,
            "resolution_source": self.resolution_source.value,
        }

    @classmethod
    def from_obj(
        cls, value: object, path: str = "resolved_collective_device_stage"
    ) -> ResolvedCollectiveDeviceStage:
        payload = _object(value, path)
        expected = {
            "instance_graph_sha256",
            "collective_operation_id",
            "collective_plan_integrity_sha256",
            "rank",
            "launch_ordinal",
            "device_instance_id",
            "device_model_sha256",
            "implementation_ref",
            "shape_vector",
            "service_entry_id",
            "resolution_source",
        }
        _fields(payload, path, expected)
        return cls(
            instance_graph_sha256=_sha256(
                payload["instance_graph_sha256"], f"{path}.instance_graph_sha256"
            ),
            collective_operation_id=_string(
                payload["collective_operation_id"], f"{path}.collective_operation_id"
            ),
            collective_plan_integrity_sha256=_sha256(
                payload["collective_plan_integrity_sha256"],
                f"{path}.collective_plan_integrity_sha256",
            ),
            rank=_integer(payload["rank"], f"{path}.rank", nonnegative=True),
            launch_ordinal=_integer(
                payload["launch_ordinal"], f"{path}.launch_ordinal", nonnegative=True
            ),
            device_instance_id=_string(
                payload["device_instance_id"], f"{path}.device_instance_id"
            ),
            device_model_sha256=_sha256(
                payload["device_model_sha256"], f"{path}.device_model_sha256"
            ),
            implementation_ref=implementation_ref_from_obj(
                payload["implementation_ref"], f"{path}.implementation_ref"
            ),
            shape_vector=ShapeVector.from_obj(payload["shape_vector"], f"{path}.shape_vector"),
            service_entry_id=_string(
                payload["service_entry_id"], f"{path}.service_entry_id"
            ),
            resolution_source=_resolution_source(
                payload["resolution_source"], f"{path}.resolution_source"
            ),
        )


@dataclass(frozen=True, slots=True)
class CollectiveDeviceRankFrontier:
    collective_operation_id: str
    collective_plan_integrity_sha256: str
    rank: int
    ordered_stage_ordinals: tuple[int, ...]
    entry_action_ids: tuple[str, ...]
    terminal_action_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _string(self.collective_operation_id, "CollectiveDeviceRankFrontier.collective_operation_id")
        _sha256(
            self.collective_plan_integrity_sha256,
            "CollectiveDeviceRankFrontier.collective_plan_integrity_sha256",
        )
        _integer(self.rank, "CollectiveDeviceRankFrontier.rank", nonnegative=True)
        _check_tuple(
            self.ordered_stage_ordinals,
            "CollectiveDeviceRankFrontier.ordered_stage_ordinals",
        )
        if len(self.ordered_stage_ordinals) != 1:
            raise ValueError(
                "CollectiveDeviceRankFrontier.ordered_stage_ordinals: version 1 "
                "requires exactly one stage"
            )
        for index, ordinal in enumerate(self.ordered_stage_ordinals):
            _integer(
                ordinal,
                f"CollectiveDeviceRankFrontier.ordered_stage_ordinals[{index}]",
                nonnegative=True,
            )
        for name in ("entry_action_ids", "terminal_action_ids"):
            values = _check_tuple(getattr(self, name), f"CollectiveDeviceRankFrontier.{name}")
            for index, action_id in enumerate(values):
                _string(action_id, f"CollectiveDeviceRankFrontier.{name}[{index}]")
            _unique(values, f"CollectiveDeviceRankFrontier.{name}")

    @property
    def key(self) -> tuple[str, str, int]:
        return (
            self.collective_operation_id,
            self.collective_plan_integrity_sha256,
            self.rank,
        )

    def to_obj(self) -> dict[str, Any]:
        return {
            "collective_operation_id": self.collective_operation_id,
            "collective_plan_integrity_sha256": self.collective_plan_integrity_sha256,
            "rank": self.rank,
            "ordered_stage_ordinals": list(self.ordered_stage_ordinals),
            "entry_action_ids": list(self.entry_action_ids),
            "terminal_action_ids": list(self.terminal_action_ids),
        }

    @classmethod
    def from_obj(
        cls, value: object, path: str = "collective_device_rank_frontier"
    ) -> CollectiveDeviceRankFrontier:
        payload = _object(value, path)
        expected = {
            "collective_operation_id",
            "collective_plan_integrity_sha256",
            "rank",
            "ordered_stage_ordinals",
            "entry_action_ids",
            "terminal_action_ids",
        }
        _fields(payload, path, expected)
        return cls(
            collective_operation_id=_string(
                payload["collective_operation_id"], f"{path}.collective_operation_id"
            ),
            collective_plan_integrity_sha256=_sha256(
                payload["collective_plan_integrity_sha256"],
                f"{path}.collective_plan_integrity_sha256",
            ),
            rank=_integer(payload["rank"], f"{path}.rank", nonnegative=True),
            ordered_stage_ordinals=tuple(
                _integer(item, f"{path}.ordered_stage_ordinals[{index}]", nonnegative=True)
                for index, item in enumerate(
                    _array(
                        payload["ordered_stage_ordinals"],
                        f"{path}.ordered_stage_ordinals",
                    )
                )
            ),
            entry_action_ids=tuple(
                _string(item, f"{path}.entry_action_ids[{index}]")
                for index, item in enumerate(
                    _array(payload["entry_action_ids"], f"{path}.entry_action_ids")
                )
            ),
            terminal_action_ids=tuple(
                _string(item, f"{path}.terminal_action_ids[{index}]")
                for index, item in enumerate(
                    _array(
                        payload["terminal_action_ids"], f"{path}.terminal_action_ids"
                    )
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class ResolvedCollectiveDeviceStageSet:
    instance_graph_sha256: str
    dispatch_context_sha256: str
    stages: tuple[ResolvedCollectiveDeviceStage, ...]
    rank_frontiers: tuple[CollectiveDeviceRankFrontier, ...]
    schema: str = RESOLVED_COLLECTIVE_STAGE_SET_SCHEMA

    def __post_init__(self) -> None:
        _require_schema(
            self.schema,
            RESOLVED_COLLECTIVE_STAGE_SET_SCHEMA,
            "ResolvedCollectiveDeviceStageSet.schema",
        )
        _sha256(
            self.instance_graph_sha256,
            "ResolvedCollectiveDeviceStageSet.instance_graph_sha256",
        )
        _sha256(
            self.dispatch_context_sha256,
            "ResolvedCollectiveDeviceStageSet.dispatch_context_sha256",
        )
        _check_tuple(self.stages, "ResolvedCollectiveDeviceStageSet.stages")
        _check_tuple(
            self.rank_frontiers, "ResolvedCollectiveDeviceStageSet.rank_frontiers"
        )
        if not self.stages:
            raise ValueError("ResolvedCollectiveDeviceStageSet.stages: must not be empty")
        stage_keys: list[tuple[str, str, str, int, int]] = []
        stage_by_frontier: dict[tuple[str, str, int], list[int]] = {}
        for index, stage in enumerate(self.stages):
            if not isinstance(stage, ResolvedCollectiveDeviceStage):
                raise TypeError(
                    f"ResolvedCollectiveDeviceStageSet.stages[{index}]: expected stage"
            )
            if stage.instance_graph_sha256 != self.instance_graph_sha256:
                raise ValueError(
                    f"ResolvedCollectiveDeviceStageSet.stages[{index}]: graph splice"
                )
            stage_keys.append(stage.key)
            stage_by_frontier.setdefault(stage.frontier_key, []).append(
                stage.launch_ordinal
            )
        _unique(tuple(stage_keys), "ResolvedCollectiveDeviceStageSet.stages")
        frontier_keys: list[tuple[str, str, int]] = []
        for index, frontier in enumerate(self.rank_frontiers):
            if not isinstance(frontier, CollectiveDeviceRankFrontier):
                raise TypeError(
                    f"ResolvedCollectiveDeviceStageSet.rank_frontiers[{index}]: "
                    "expected CollectiveDeviceRankFrontier"
                )
            frontier_keys.append(frontier.key)
            if tuple(stage_by_frontier.get(frontier.key, ())) != (
                frontier.ordered_stage_ordinals
            ):
                raise ValueError(
                    f"ResolvedCollectiveDeviceStageSet.rank_frontiers[{index}]: "
                    "stage ordinals do not match exactly one resolved stage"
                )
        _unique(tuple(frontier_keys), "ResolvedCollectiveDeviceStageSet.rank_frontiers")
        if set(frontier_keys) != set(stage_by_frontier):
            raise ValueError(
                "ResolvedCollectiveDeviceStageSet.rank_frontiers: missing or extra rank"
            )

    def to_obj(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "instance_graph_sha256": self.instance_graph_sha256,
            "dispatch_context_sha256": self.dispatch_context_sha256,
            "stages": [stage.to_obj() for stage in self.stages],
            "rank_frontiers": [frontier.to_obj() for frontier in self.rank_frontiers],
        }

    @classmethod
    def from_obj(
        cls, value: object, path: str = "resolved_collective_device_stage_set"
    ) -> ResolvedCollectiveDeviceStageSet:
        payload = _object(value, path)
        expected = {
            "schema",
            "instance_graph_sha256",
            "dispatch_context_sha256",
            "stages",
            "rank_frontiers",
        }
        _fields(payload, path, expected)
        _require_schema(
            payload["schema"], RESOLVED_COLLECTIVE_STAGE_SET_SCHEMA, f"{path}.schema"
        )
        return cls(
            instance_graph_sha256=_sha256(
                payload["instance_graph_sha256"], f"{path}.instance_graph_sha256"
            ),
            dispatch_context_sha256=_sha256(
                payload["dispatch_context_sha256"], f"{path}.dispatch_context_sha256"
            ),
            stages=tuple(
                ResolvedCollectiveDeviceStage.from_obj(
                    item, f"{path}.stages[{index}]"
                )
                for index, item in enumerate(
                    _array(payload["stages"], f"{path}.stages")
                )
            ),
            rank_frontiers=tuple(
                CollectiveDeviceRankFrontier.from_obj(
                    item, f"{path}.rank_frontiers[{index}]"
                )
                for index, item in enumerate(
                    _array(payload["rank_frontiers"], f"{path}.rank_frontiers")
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class ResolvedDeviceBindingClosure:
    instance_graph_sha256: str
    operation_service_binding_set_sha256: str
    collective_device_stage_set_sha256: str | None
    schema: str = RESOLVED_DEVICE_BINDING_CLOSURE_SCHEMA

    def __post_init__(self) -> None:
        _require_schema(
            self.schema,
            RESOLVED_DEVICE_BINDING_CLOSURE_SCHEMA,
            "ResolvedDeviceBindingClosure.schema",
        )
        _sha256(
            self.instance_graph_sha256,
            "ResolvedDeviceBindingClosure.instance_graph_sha256",
        )
        _sha256(
            self.operation_service_binding_set_sha256,
            "ResolvedDeviceBindingClosure.operation_service_binding_set_sha256",
        )
        _optional_sha256(
            self.collective_device_stage_set_sha256,
            "ResolvedDeviceBindingClosure.collective_device_stage_set_sha256",
        )

    def to_obj(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "instance_graph_sha256": self.instance_graph_sha256,
            "operation_service_binding_set_sha256": (
                self.operation_service_binding_set_sha256
            ),
            "collective_device_stage_set_sha256": (
                self.collective_device_stage_set_sha256
            ),
        }

    @classmethod
    def from_obj(
        cls, value: object, path: str = "resolved_device_binding_closure"
    ) -> ResolvedDeviceBindingClosure:
        payload = _object(value, path)
        expected = {
            "schema",
            "instance_graph_sha256",
            "operation_service_binding_set_sha256",
            "collective_device_stage_set_sha256",
        }
        _fields(payload, path, expected)
        _require_schema(
            payload["schema"], RESOLVED_DEVICE_BINDING_CLOSURE_SCHEMA, f"{path}.schema"
        )
        return cls(
            instance_graph_sha256=_sha256(
                payload["instance_graph_sha256"], f"{path}.instance_graph_sha256"
            ),
            operation_service_binding_set_sha256=_sha256(
                payload["operation_service_binding_set_sha256"],
                f"{path}.operation_service_binding_set_sha256",
            ),
            collective_device_stage_set_sha256=_optional_sha256(
                payload["collective_device_stage_set_sha256"],
                f"{path}.collective_device_stage_set_sha256",
            ),
        )


def validate_observed_operation_bindings(
    bindings: tuple[OperationImplementationBinding, ...],
    expected_keys: tuple[tuple[str, str, int], ...],
) -> None:
    """Reject missing, duplicate, extra, or reordered noncollective launches."""

    actual = tuple(binding.key for binding in bindings)
    _unique(actual, "operation_implementation_bindings")
    if actual != expected_keys:
        raise ValueError(
            "operation_implementation_bindings: bindings must match expected graph "
            "operation and launch order exactly"
        )


def validate_observed_collective_stage_bindings(
    bindings: tuple[CollectiveDeviceStageBinding, ...],
    expected_keys: tuple[tuple[str, str, str, int, int], ...],
) -> None:
    """Reject missing, duplicate, extra, or reordered physical stages."""

    actual = tuple(binding.key for binding in bindings)
    _unique(actual, "collective_device_stage_bindings")
    if actual != expected_keys:
        raise ValueError(
            "collective_device_stage_bindings: bindings must match expected physical "
            "stage order exactly"
        )


def validate_resolved_operation_bindings(
    record: ResolvedOperationServiceBindingSet,
    *,
    expected_keys: tuple[tuple[str, str, int], ...],
    operation_rank_by_id: Mapping[str, int],
    dispatch_context: DeviceDispatchContext,
) -> None:
    """Validate graph order, totality, and rank/device assignment."""

    actual = tuple(binding.key for binding in record.bindings)
    if actual != expected_keys:
        raise ValueError(
            "resolved operation bindings: missing, extra, duplicate, or reordered launch"
        )
    for index, binding in enumerate(record.bindings):
        rank = operation_rank_by_id.get(binding.operation_id)
        if rank is None:
            raise ValueError(
                f"resolved operation bindings[{index}]: operation is absent from graph"
            )
        expected_device = dispatch_context.device_for_rank(rank)
        if binding.device_instance_id != expected_device:
            raise ValueError(
                f"resolved operation bindings[{index}]: rank/device assignment splice"
            )


def validate_resolved_collective_stage_plan(
    record: ResolvedCollectiveDeviceStageSet,
    *,
    expected_stage_keys: tuple[tuple[str, str, str, int, int], ...],
    expected_rank_frontiers: tuple[CollectiveDeviceRankFrontier, ...],
    dispatch_context: DeviceDispatchContext,
) -> None:
    """Validate stage order, copied plan frontiers, and rank/device assignment."""

    actual_stage_keys = tuple(stage.key for stage in record.stages)
    if actual_stage_keys != expected_stage_keys:
        raise ValueError(
            "resolved collective stages: missing, extra, duplicate, or reordered stage"
        )
    if record.rank_frontiers != expected_rank_frontiers:
        raise ValueError(
            "resolved collective stages: rank frontiers do not match the traffic plan"
        )
    for index, stage in enumerate(record.stages):
        expected_device = dispatch_context.device_for_rank(stage.rank)
        if stage.device_instance_id != expected_device:
            raise ValueError(
                f"resolved collective stages[{index}]: rank/device assignment splice"
            )


def validate_resolved_binding_closure(
    *,
    closure: ResolvedDeviceBindingClosure,
    operation_set: ResolvedOperationServiceBindingSet,
    operation_set_sha256: str,
    dispatch_context: DeviceDispatchContext,
    dispatch_context_sha256: str,
    collective_set: ResolvedCollectiveDeviceStageSet | None = None,
    collective_set_sha256: str | None = None,
) -> None:
    """Validate the graph, context, and device-model closure before publication."""

    _sha256(operation_set_sha256, "operation_set_sha256")
    _sha256(dispatch_context_sha256, "dispatch_context_sha256")
    if (collective_set is None) != (collective_set_sha256 is None):
        raise ValueError("collective set and digest must either both be present or absent")
    if closure.operation_service_binding_set_sha256 != operation_set_sha256:
        raise ValueError("resolved binding closure: operation-set digest splice")
    if closure.collective_device_stage_set_sha256 != collective_set_sha256:
        raise ValueError("resolved binding closure: collective-set digest splice")
    records: tuple[object, ...] = (closure, operation_set, dispatch_context)
    if collective_set is not None:
        _sha256(collective_set_sha256, "collective_set_sha256")
        records += (collective_set,)
    graph_hashes = {record.instance_graph_sha256 for record in records}
    if len(graph_hashes) != 1:
        raise ValueError("resolved binding closure: cross-graph splice")
    if operation_set.dispatch_context_sha256 != dispatch_context_sha256:
        raise ValueError("resolved binding closure: operation-set context splice")
    if collective_set is not None and (
        collective_set.dispatch_context_sha256 != dispatch_context_sha256
    ):
        raise ValueError("resolved binding closure: collective-set context splice")
    selected = dispatch_context.model_by_device()
    for label, members in (
        ("operation", operation_set.bindings),
        ("collective", () if collective_set is None else collective_set.stages),
    ):
        for index, member in enumerate(members):
            selection = selected.get(member.device_instance_id)
            if selection is None:
                raise ValueError(
                    f"resolved binding closure: {label}[{index}] uses an unselected device"
                )
            if selection.device_model_sha256 != member.device_model_sha256:
                raise ValueError(
                    f"resolved binding closure: {label}[{index}] model splice"
                )
            if label == "collective" and (
                dispatch_context.device_for_rank(member.rank)
                != member.device_instance_id
            ):
                raise ValueError(
                    f"resolved binding closure: {label}[{index}] device assignment splice"
                )


def binding_record_from_obj(
    value: object,
) -> (
    DispatchSignature
    | DeviceDispatchContext
    | ResolvedOperationServiceBindingSet
    | ResolvedCollectiveDeviceStageSet
    | ResolvedDeviceBindingClosure
):
    """Strict dispatcher for every schema-bearing binding record."""

    payload = _object(value, "record")
    schema = payload.get("schema")
    if schema == DISPATCH_SIGNATURE_SCHEMA:
        return DispatchSignature.from_obj(payload, "record")
    if schema == DEVICE_DISPATCH_CONTEXT_SCHEMA:
        return DeviceDispatchContext.from_obj(payload, "record")
    if schema == RESOLVED_OPERATION_BINDING_SET_SCHEMA:
        return ResolvedOperationServiceBindingSet.from_obj(payload, "record")
    if schema == RESOLVED_COLLECTIVE_STAGE_SET_SCHEMA:
        return ResolvedCollectiveDeviceStageSet.from_obj(payload, "record")
    if schema == RESOLVED_DEVICE_BINDING_CLOSURE_SCHEMA:
        return ResolvedDeviceBindingClosure.from_obj(payload, "record")
    raise ValueError(f"record.schema: unsupported binding record {schema!r}")


TYPED_RECORD_READERS = {
    DISPATCH_SIGNATURE_SCHEMA: DispatchSignature.from_obj,
    DEVICE_DISPATCH_CONTEXT_SCHEMA: DeviceDispatchContext.from_obj,
    RESOLVED_OPERATION_BINDING_SET_SCHEMA: ResolvedOperationServiceBindingSet.from_obj,
    RESOLVED_COLLECTIVE_STAGE_SET_SCHEMA: ResolvedCollectiveDeviceStageSet.from_obj,
    RESOLVED_DEVICE_BINDING_CLOSURE_SCHEMA: ResolvedDeviceBindingClosure.from_obj,
}


def validate_typed_record(value: object):
    """Validate one supported schema-bearing binding record."""

    return binding_record_from_obj(value)


__all__ = [
    "DEVICE_DISPATCH_CONTEXT_SCHEMA",
    "DISPATCH_SIGNATURE_SCHEMA",
    "RESOLVED_COLLECTIVE_STAGE_SET_SCHEMA",
    "RESOLVED_DEVICE_BINDING_CLOSURE_SCHEMA",
    "RESOLVED_OPERATION_BINDING_SET_SCHEMA",
    "TYPED_DISPATCH_TRAIT_SCHEMA",
    "TYPED_RECORD_READERS",
    "CollectiveDeviceRankFrontier",
    "CollectiveDeviceStageBinding",
    "DeviceDispatchContext",
    "DispatchSignature",
    "DispatchTraitValueType",
    "OperationImplementationBinding",
    "RankDeviceAssignment",
    "ResolutionSource",
    "ResolvedCollectiveDeviceStage",
    "ResolvedCollectiveDeviceStageSet",
    "ResolvedDeviceBindingClosure",
    "ResolvedOperationServiceBinding",
    "ResolvedOperationServiceBindingSet",
    "SelectedDeviceModel",
    "TypedDispatchTrait",
    "binding_record_from_obj",
    "validate_observed_collective_stage_bindings",
    "validate_observed_operation_bindings",
    "validate_resolved_binding_closure",
    "validate_resolved_collective_stage_plan",
    "validate_resolved_operation_bindings",
    "validate_typed_record",
]
