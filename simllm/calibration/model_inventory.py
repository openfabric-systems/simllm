"""Strict content-addressed model kernel inventory records."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from simllm.compute.device_model import ShapeSchema, ShapeVector

from .record_types import RecordObject

MODEL_KERNEL_INVENTORY_SCHEMA = "simllm-model-kernel-inventory-v1"
ABSENT_BY_DESIGN = "absent-by-design"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_ID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_PHASES = frozenset({"prefill", "decode"})


def _object(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected an object")
    return value


def _array(value: object, path: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{path}: expected an array")
    return value


def _fields(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(f"{path}: missing fields {missing}; unknown fields {unknown}")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{path}: expected a nonblank string without edge whitespace")
    return value


def _integer(value: object, path: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise ValueError(f"{path}: expected an integer")
    if value < (1 if positive else 0):
        label = "positive" if positive else "nonnegative"
        raise ValueError(f"{path}: expected a {label} integer")
    return value


def _digest(value: object, path: str) -> str:
    text = _string(value, path)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{path}: expected 64 lowercase hexadecimal digits")
    return text


def _git_id(value: object, path: str) -> str:
    text = _string(value, path)
    if _GIT_ID.fullmatch(text) is None:
        raise ValueError(f"{path}: expected a 40 or 64 digit lowercase git object ID")
    return text


@dataclass(frozen=True, slots=True)
class InventorySuiteIdentity:
    """Exact authored suite that supplied the inventory case denominators."""

    suite_id: str
    suite_sha256: str
    case_count: int

    def __post_init__(self) -> None:
        _string(self.suite_id, "InventorySuiteIdentity.suite_id")
        _digest(self.suite_sha256, "InventorySuiteIdentity.suite_sha256")
        _integer(self.case_count, "InventorySuiteIdentity.case_count", positive=True)

    def to_obj(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "suite_sha256": self.suite_sha256,
            "case_count": self.case_count,
        }

    @classmethod
    def from_obj(cls, value: object, path: str = "suite") -> InventorySuiteIdentity:
        payload = _object(value, path)
        _fields(payload, {"suite_id", "suite_sha256", "case_count"}, path)
        return cls(
            suite_id=_string(payload["suite_id"], f"{path}.suite_id"),
            suite_sha256=_digest(payload["suite_sha256"], f"{path}.suite_sha256"),
            case_count=_integer(
                payload["case_count"], f"{path}.case_count", positive=True
            ),
        )


@dataclass(frozen=True, slots=True)
class FrameworkIdentity:
    """Pinned framework release and source identity."""

    framework_id: str
    version: str
    source_commit: str
    source_tree: str | None
    entry_seam: str

    def __post_init__(self) -> None:
        _string(self.framework_id, "FrameworkIdentity.framework_id")
        _string(self.version, "FrameworkIdentity.version")
        _git_id(self.source_commit, "FrameworkIdentity.source_commit")
        if self.source_tree is not None:
            _git_id(self.source_tree, "FrameworkIdentity.source_tree")
        _string(self.entry_seam, "FrameworkIdentity.entry_seam")

    def to_obj(self) -> dict[str, Any]:
        return {
            "id": self.framework_id,
            "version": self.version,
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "entry_seam": self.entry_seam,
        }

    @classmethod
    def from_obj(cls, value: object, path: str = "framework") -> FrameworkIdentity:
        payload = _object(value, path)
        _fields(
            payload,
            {"id", "version", "source_commit", "source_tree", "entry_seam"},
            path,
        )
        source_tree = payload["source_tree"]
        if source_tree is not None:
            source_tree = _git_id(source_tree, f"{path}.source_tree")
        return cls(
            framework_id=_string(payload["id"], f"{path}.id"),
            version=_string(payload["version"], f"{path}.version"),
            source_commit=_git_id(
                payload["source_commit"], f"{path}.source_commit"
            ),
            source_tree=source_tree,
            entry_seam=_string(payload["entry_seam"], f"{path}.entry_seam"),
        )


@dataclass(frozen=True, slots=True)
class ModelGeometry:
    """Exact unsharded transformer geometry from the checkpoint config."""

    layers: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    head_size: int
    num_experts: int
    top_k: int
    vocab_size: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name in {"num_experts", "top_k"}:
                _integer(value, f"ModelGeometry.{name}")
            else:
                _integer(value, f"ModelGeometry.{name}", positive=True)
        if (self.num_experts == 0) != (self.top_k == 0):
            raise ValueError("ModelGeometry experts and top_k must both be zero or positive")
        if self.top_k > self.num_experts:
            raise ValueError("ModelGeometry.top_k cannot exceed num_experts")

    def to_obj(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_obj(cls, value: object, path: str = "geometry") -> ModelGeometry:
        payload = _object(value, path)
        expected = set(cls.__dataclass_fields__)
        _fields(payload, expected, path)
        return cls(
            **{
                name: _integer(
                    payload[name],
                    f"{path}.{name}",
                    positive=name not in {"num_experts", "top_k"},
                )
                for name in cls.__dataclass_fields__
            }
        )


@dataclass(frozen=True, slots=True)
class ModelCheckpointIdentity:
    """Exact checkpoint bytes and geometry."""

    name: str
    revision: str
    config_sha256: str
    weight_sha256: str
    weight_bytes: int
    dtype: str
    quantization: str
    geometry: ModelGeometry

    def __post_init__(self) -> None:
        _string(self.name, "ModelCheckpointIdentity.name")
        _git_id(self.revision, "ModelCheckpointIdentity.revision")
        _digest(self.config_sha256, "ModelCheckpointIdentity.config_sha256")
        _digest(self.weight_sha256, "ModelCheckpointIdentity.weight_sha256")
        _integer(self.weight_bytes, "ModelCheckpointIdentity.weight_bytes", positive=True)
        _string(self.dtype, "ModelCheckpointIdentity.dtype")
        _string(self.quantization, "ModelCheckpointIdentity.quantization")
        if not isinstance(self.geometry, ModelGeometry):
            raise TypeError("ModelCheckpointIdentity.geometry: expected ModelGeometry")

    def to_obj(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "revision": self.revision,
            "config_sha256": self.config_sha256,
            "weight_sha256": self.weight_sha256,
            "weight_bytes": self.weight_bytes,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "geometry": self.geometry.to_obj(),
        }

    @classmethod
    def from_obj(cls, value: object, path: str = "model") -> ModelCheckpointIdentity:
        payload = _object(value, path)
        expected = {
            "name",
            "revision",
            "config_sha256",
            "weight_sha256",
            "weight_bytes",
            "dtype",
            "quantization",
            "geometry",
        }
        _fields(payload, expected, path)
        return cls(
            name=_string(payload["name"], f"{path}.name"),
            revision=_git_id(payload["revision"], f"{path}.revision"),
            config_sha256=_digest(
                payload["config_sha256"], f"{path}.config_sha256"
            ),
            weight_sha256=_digest(
                payload["weight_sha256"], f"{path}.weight_sha256"
            ),
            weight_bytes=_integer(
                payload["weight_bytes"], f"{path}.weight_bytes", positive=True
            ),
            dtype=_string(payload["dtype"], f"{path}.dtype"),
            quantization=_string(
                payload["quantization"], f"{path}.quantization"
            ),
            geometry=ModelGeometry.from_obj(payload["geometry"], f"{path}.geometry"),
        )


@dataclass(frozen=True, slots=True)
class PhaseLaunchCount:
    phase: str
    logical_launch_count: int

    def __post_init__(self) -> None:
        if self.phase not in _PHASES:
            raise ValueError(f"PhaseLaunchCount.phase: unsupported phase {self.phase!r}")
        _integer(
            self.logical_launch_count,
            "PhaseLaunchCount.logical_launch_count",
            positive=True,
        )

    def to_obj(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "logical_launch_count": self.logical_launch_count,
        }

    @classmethod
    def from_obj(cls, value: object, path: str) -> PhaseLaunchCount:
        payload = _object(value, path)
        _fields(payload, {"phase", "logical_launch_count"}, path)
        return cls(
            phase=_string(payload["phase"], f"{path}.phase"),
            logical_launch_count=_integer(
                payload["logical_launch_count"],
                f"{path}.logical_launch_count",
                positive=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class KernelFamilyDefinition:
    family_id: str
    shape_schema_id: str
    phase_launch_counts: tuple[PhaseLaunchCount, ...]

    def __post_init__(self) -> None:
        _string(self.family_id, "KernelFamilyDefinition.family_id")
        _string(self.shape_schema_id, "KernelFamilyDefinition.shape_schema_id")
        phases = tuple(item.phase for item in self.phase_launch_counts)
        if phases != ("prefill", "decode"):
            raise ValueError(
                "KernelFamilyDefinition.phase_launch_counts must be prefill then decode"
            )

    def to_obj(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "shape_schema_id": self.shape_schema_id,
            "phase_launch_counts": [item.to_obj() for item in self.phase_launch_counts],
        }

    @classmethod
    def from_obj(cls, value: object, path: str) -> KernelFamilyDefinition:
        payload = _object(value, path)
        _fields(
            payload,
            {"family_id", "shape_schema_id", "phase_launch_counts"},
            path,
        )
        return cls(
            family_id=_string(payload["family_id"], f"{path}.family_id"),
            shape_schema_id=_string(
                payload["shape_schema_id"], f"{path}.shape_schema_id"
            ),
            phase_launch_counts=tuple(
                PhaseLaunchCount.from_obj(item, f"{path}.phase_launch_counts[{index}]")
                for index, item in enumerate(
                    _array(
                        payload["phase_launch_counts"],
                        f"{path}.phase_launch_counts",
                    )
                )
            ),
        )

    def count_for(self, phase: str) -> int:
        return next(
            item.logical_launch_count
            for item in self.phase_launch_counts
            if item.phase == phase
        )


@dataclass(frozen=True, slots=True)
class KernelProjection:
    family_id: str
    shape_vector: ShapeVector
    logical_launch_count: int
    aggregate_flops: int
    aggregate_hbm_bytes: int

    def __post_init__(self) -> None:
        _string(self.family_id, "KernelProjection.family_id")
        if not isinstance(self.shape_vector, ShapeVector):
            raise TypeError("KernelProjection.shape_vector: expected ShapeVector")
        _integer(
            self.logical_launch_count,
            "KernelProjection.logical_launch_count",
            positive=True,
        )
        _integer(self.aggregate_flops, "KernelProjection.aggregate_flops")
        _integer(self.aggregate_hbm_bytes, "KernelProjection.aggregate_hbm_bytes")

    def to_obj(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "shape_vector": self.shape_vector.to_obj(),
            "logical_launch_count": self.logical_launch_count,
            "aggregate_flops": self.aggregate_flops,
            "aggregate_hbm_bytes": self.aggregate_hbm_bytes,
        }

    @classmethod
    def from_obj(cls, value: object, path: str) -> KernelProjection:
        payload = _object(value, path)
        expected = {
            "family_id",
            "shape_vector",
            "logical_launch_count",
            "aggregate_flops",
            "aggregate_hbm_bytes",
        }
        _fields(payload, expected, path)
        return cls(
            family_id=_string(payload["family_id"], f"{path}.family_id"),
            shape_vector=ShapeVector.from_obj(
                payload["shape_vector"], f"{path}.shape_vector"
            ),
            logical_launch_count=_integer(
                payload["logical_launch_count"],
                f"{path}.logical_launch_count",
                positive=True,
            ),
            aggregate_flops=_integer(
                payload["aggregate_flops"], f"{path}.aggregate_flops"
            ),
            aggregate_hbm_bytes=_integer(
                payload["aggregate_hbm_bytes"], f"{path}.aggregate_hbm_bytes"
            ),
        )


@dataclass(frozen=True, slots=True)
class InventoryCase:
    case_id: str
    family: str
    phase: str
    split: str
    suite_case_sha256: str
    step_record_sha256: str
    instance_graph_sha256: str
    template_graph_sha256: str
    kernel_projections: tuple[KernelProjection, ...]

    def __post_init__(self) -> None:
        for name in ("case_id", "family", "split"):
            _string(getattr(self, name), f"InventoryCase.{name}")
        if self.phase not in _PHASES:
            raise ValueError(f"InventoryCase.phase: unsupported phase {self.phase!r}")
        for name in (
            "suite_case_sha256",
            "step_record_sha256",
            "instance_graph_sha256",
            "template_graph_sha256",
        ):
            _digest(getattr(self, name), f"InventoryCase.{name}")
        if not self.kernel_projections:
            raise ValueError("InventoryCase.kernel_projections cannot be empty")

    def to_obj(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "phase": self.phase,
            "split": self.split,
            "suite_case_sha256": self.suite_case_sha256,
            "step_record_sha256": self.step_record_sha256,
            "instance_graph_sha256": self.instance_graph_sha256,
            "template_graph_sha256": self.template_graph_sha256,
            "kernel_projections": [item.to_obj() for item in self.kernel_projections],
        }

    @classmethod
    def from_obj(cls, value: object, path: str) -> InventoryCase:
        payload = _object(value, path)
        expected = {
            "case_id",
            "family",
            "phase",
            "split",
            "suite_case_sha256",
            "step_record_sha256",
            "instance_graph_sha256",
            "template_graph_sha256",
            "kernel_projections",
        }
        _fields(payload, expected, path)
        return cls(
            case_id=_string(payload["case_id"], f"{path}.case_id"),
            family=_string(payload["family"], f"{path}.family"),
            phase=_string(payload["phase"], f"{path}.phase"),
            split=_string(payload["split"], f"{path}.split"),
            suite_case_sha256=_digest(
                payload["suite_case_sha256"], f"{path}.suite_case_sha256"
            ),
            step_record_sha256=_digest(
                payload["step_record_sha256"], f"{path}.step_record_sha256"
            ),
            instance_graph_sha256=_digest(
                payload["instance_graph_sha256"],
                f"{path}.instance_graph_sha256",
            ),
            template_graph_sha256=_digest(
                payload["template_graph_sha256"],
                f"{path}.template_graph_sha256",
            ),
            kernel_projections=tuple(
                KernelProjection.from_obj(item, f"{path}.kernel_projections[{index}]")
                for index, item in enumerate(
                    _array(payload["kernel_projections"], f"{path}.kernel_projections")
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class AbsentPhysicalIdentity:
    state: str = ABSENT_BY_DESIGN
    value: None = None

    def __post_init__(self) -> None:
        if self.state != ABSENT_BY_DESIGN or self.value is not None:
            raise ValueError("physical identity must be absent-by-design with null value")

    def to_obj(self) -> dict[str, Any]:
        return {"state": self.state, "value": self.value}

    @classmethod
    def from_obj(cls, value: object, path: str) -> AbsentPhysicalIdentity:
        payload = _object(value, path)
        _fields(payload, {"state", "value"}, path)
        if payload["state"] != ABSENT_BY_DESIGN or payload["value"] is not None:
            raise ValueError(f"{path}: expected absent-by-design with null value")
        return cls()


@dataclass(frozen=True, slots=True)
class ImplementationIdentityEnvelope:
    code_object_hashes: AbsentPhysicalIdentity
    observed_launches: AbsentPhysicalIdentity
    join_tasks: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.code_object_hashes, AbsentPhysicalIdentity):
            raise TypeError("code_object_hashes: expected AbsentPhysicalIdentity")
        if not isinstance(self.observed_launches, AbsentPhysicalIdentity):
            raise TypeError("observed_launches: expected AbsentPhysicalIdentity")
        if tuple(sorted(set(self.join_tasks))) != self.join_tasks:
            raise ValueError("ImplementationIdentityEnvelope.join_tasks must be sorted unique")
        for index, task in enumerate(self.join_tasks):
            _string(task, f"ImplementationIdentityEnvelope.join_tasks[{index}]")

    def to_obj(self) -> dict[str, Any]:
        return {
            "code_object_hashes": self.code_object_hashes.to_obj(),
            "observed_launches": self.observed_launches.to_obj(),
            "join_tasks": list(self.join_tasks),
        }

    @classmethod
    def from_obj(
        cls, value: object, path: str = "implementation_identity"
    ) -> ImplementationIdentityEnvelope:
        payload = _object(value, path)
        _fields(payload, {"code_object_hashes", "observed_launches", "join_tasks"}, path)
        return cls(
            code_object_hashes=AbsentPhysicalIdentity.from_obj(
                payload["code_object_hashes"], f"{path}.code_object_hashes"
            ),
            observed_launches=AbsentPhysicalIdentity.from_obj(
                payload["observed_launches"], f"{path}.observed_launches"
            ),
            join_tasks=tuple(
                _string(item, f"{path}.join_tasks[{index}]")
                for index, item in enumerate(
                    _array(payload["join_tasks"], f"{path}.join_tasks")
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelKernelInventory:
    """One total structure-only inventory for an exact framework and model."""

    suite: InventorySuiteIdentity
    framework: FrameworkIdentity
    model: ModelCheckpointIdentity
    shape_schemas: tuple[ShapeSchema, ...]
    kernel_families: tuple[KernelFamilyDefinition, ...]
    cases: tuple[InventoryCase, ...]
    implementation_identity: ImplementationIdentityEnvelope
    schema: str = MODEL_KERNEL_INVENTORY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != MODEL_KERNEL_INVENTORY_SCHEMA:
            raise ValueError(f"unsupported inventory schema {self.schema!r}")
        if len(self.cases) != self.suite.case_count:
            raise ValueError(
                f"inventory has {len(self.cases)} cases, expected {self.suite.case_count}"
            )
        case_ids = tuple(case.case_id for case in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("inventory case IDs must be unique")
        schema_by_id = {schema.shape_schema_id: schema for schema in self.shape_schemas}
        if len(schema_by_id) != len(self.shape_schemas):
            raise ValueError("inventory shape schema IDs must be unique")
        family_ids = tuple(family.family_id for family in self.kernel_families)
        if len(family_ids) != len(set(family_ids)):
            raise ValueError("inventory kernel family IDs must be unique")
        family_by_id = {family.family_id: family for family in self.kernel_families}
        for family in self.kernel_families:
            if family.shape_schema_id not in schema_by_id:
                raise ValueError(
                    f"family {family.family_id!r} references an unknown shape schema"
                )
        for case in self.cases:
            projection_ids = tuple(item.family_id for item in case.kernel_projections)
            if projection_ids != family_ids:
                raise ValueError(
                    f"case {case.case_id!r} kernel projection order is not total"
                )
            for projection in case.kernel_projections:
                family = family_by_id[projection.family_id]
                schema_by_id[family.shape_schema_id].validate_vector(
                    projection.shape_vector,
                    f"case[{case.case_id}].{projection.family_id}.shape_vector",
                )
                expected_count = family.count_for(case.phase)
                if projection.logical_launch_count != expected_count:
                    raise ValueError(
                        f"case {case.case_id!r} family {family.family_id!r} has "
                        f"launch count {projection.logical_launch_count}, expected "
                        f"{expected_count}"
                    )

    @property
    def record(self) -> RecordObject:
        return RecordObject.from_value(self.to_obj())

    def to_obj(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "suite": self.suite.to_obj(),
            "framework": self.framework.to_obj(),
            "model": self.model.to_obj(),
            "shape_schemas": [schema.to_obj() for schema in self.shape_schemas],
            "kernel_families": [family.to_obj() for family in self.kernel_families],
            "cases": [case.to_obj() for case in self.cases],
            "implementation_identity": self.implementation_identity.to_obj(),
        }

    @classmethod
    def from_obj(cls, value: object, path: str = "inventory") -> ModelKernelInventory:
        payload = _object(value, path)
        expected = {
            "schema",
            "suite",
            "framework",
            "model",
            "shape_schemas",
            "kernel_families",
            "cases",
            "implementation_identity",
        }
        _fields(payload, expected, path)
        if payload["schema"] != MODEL_KERNEL_INVENTORY_SCHEMA:
            raise ValueError(
                f"{path}.schema: expected {MODEL_KERNEL_INVENTORY_SCHEMA!r}"
            )
        return cls(
            suite=InventorySuiteIdentity.from_obj(payload["suite"], f"{path}.suite"),
            framework=FrameworkIdentity.from_obj(
                payload["framework"], f"{path}.framework"
            ),
            model=ModelCheckpointIdentity.from_obj(
                payload["model"], f"{path}.model"
            ),
            shape_schemas=tuple(
                ShapeSchema.from_obj(item, f"{path}.shape_schemas[{index}]")
                for index, item in enumerate(
                    _array(payload["shape_schemas"], f"{path}.shape_schemas")
                )
            ),
            kernel_families=tuple(
                KernelFamilyDefinition.from_obj(
                    item, f"{path}.kernel_families[{index}]"
                )
                for index, item in enumerate(
                    _array(payload["kernel_families"], f"{path}.kernel_families")
                )
            ),
            cases=tuple(
                InventoryCase.from_obj(item, f"{path}.cases[{index}]")
                for index, item in enumerate(
                    _array(payload["cases"], f"{path}.cases")
                )
            ),
            implementation_identity=ImplementationIdentityEnvelope.from_obj(
                payload["implementation_identity"],
                f"{path}.implementation_identity",
            ),
        )


TYPED_RECORD_READERS = {
    MODEL_KERNEL_INVENTORY_SCHEMA: ModelKernelInventory.from_obj,
}

__all__ = [
    "ABSENT_BY_DESIGN",
    "MODEL_KERNEL_INVENTORY_SCHEMA",
    "TYPED_RECORD_READERS",
    "AbsentPhysicalIdentity",
    "FrameworkIdentity",
    "ImplementationIdentityEnvelope",
    "InventoryCase",
    "InventorySuiteIdentity",
    "KernelFamilyDefinition",
    "KernelProjection",
    "ModelCheckpointIdentity",
    "ModelGeometry",
    "ModelKernelInventory",
    "PhaseLaunchCount",
]
