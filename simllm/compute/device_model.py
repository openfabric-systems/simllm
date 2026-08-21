"""Typed identities and compact resource records for device service models.

These records are deliberately independent from the execution graph. They are
safe to import on the serving path and have no dependency on the offline
calibration package, hardware collectors, or external simulators. Record
hashes are external to these objects.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import gcd
from typing import Any, NoReturn, TypeAlias

SHAPE_SCHEMA_SCHEMA = "simllm-shape-schema-v1"
DEVICE_RESOURCE_REGISTRY_SCHEMA = "simllm-device-resource-registry-v1"
DEVICE_MODEL_SCHEMA = "simllm-device-model-v1"

SIGNED_128_MIN = -(1 << 127)
SIGNED_128_MAX = (1 << 127) - 1

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT_ID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_TRUSTED_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*\Z")


def _fail(path: str, message: str) -> NoReturn:
    raise ValueError(f"{path}: {message}")


def _object(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "expected an object")
    if any(not isinstance(key, str) for key in value):
        _fail(path, "object keys must be strings")
    return value


def _fields(value: Mapping[str, Any], path: str, expected: set[str]) -> None:
    missing = sorted(expected - value.keys())
    if missing:
        _fail(path, f"missing fields {missing}")
    unknown = sorted(value.keys() - expected)
    if unknown:
        _fail(path, f"unknown fields {unknown}")


def _array(value: object, path: str) -> Sequence[Any]:
    if not isinstance(value, list | tuple):
        _fail(path, "expected an array")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        _fail(path, "expected a string")
    if not value.strip():
        _fail(path, "must not be blank")
    return value


def _optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _integer(value: object, path: str, *, nonnegative: bool = False) -> int:
    if type(value) is not int:
        _fail(path, "expected an integer")
    if nonnegative and value < 0:
        _fail(path, "must be nonnegative")
    return value


def _sha256(value: object, path: str) -> str:
    digest = _string(value, path)
    if _SHA256_RE.fullmatch(digest) is None:
        _fail(path, "expected 64 lowercase hexadecimal digits")
    return digest


def _optional_sha256(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, path)


def _git_object_id(value: object, path: str) -> str:
    object_id = _string(value, path)
    if _GIT_OBJECT_ID_RE.fullmatch(object_id) is None:
        _fail(path, "expected 40 or 64 lowercase hexadecimal digits")
    return object_id


def _trusted_identifier(value: object, path: str) -> str:
    identifier = _string(value, path)
    if _TRUSTED_IDENTIFIER_RE.fullmatch(identifier) is None:
        _fail(path, "expected a trusted data identifier")
    return identifier


def _require_schema(value: object, expected: str, path: str) -> None:
    actual = _string(value, path)
    if actual != expected:
        _fail(path, f"expected {expected!r}, got {actual!r}")


def _require_sorted_unique(values: tuple[str, ...], path: str) -> None:
    if tuple(sorted(values)) != values:
        _fail(path, "must be sorted")
    if len(values) != len(set(values)):
        _fail(path, "must not contain duplicates")


@dataclass(frozen=True, slots=True)
class ShapeAxis:
    """One ordered integer shape coordinate and its closed domain."""

    axis_id: str
    unit: str
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        _string(self.axis_id, "ShapeAxis.axis_id")
        _string(self.unit, "ShapeAxis.unit")
        _bounded_integer(self.minimum, "ShapeAxis.minimum")
        _bounded_integer(self.maximum, "ShapeAxis.maximum")
        if self.maximum < self.minimum:
            _fail("ShapeAxis.maximum", "must be greater than or equal to minimum")

    def to_obj(self) -> dict[str, Any]:
        return {
            "axis_id": self.axis_id,
            "unit": self.unit,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }

    @classmethod
    def from_obj(cls, value: object, path: str = "shape_axis") -> ShapeAxis:
        payload = _object(value, path)
        _fields(payload, path, {"axis_id", "unit", "minimum", "maximum"})
        return cls(
            axis_id=_string(payload["axis_id"], f"{path}.axis_id"),
            unit=_string(payload["unit"], f"{path}.unit"),
            minimum=_bounded_integer(payload["minimum"], f"{path}.minimum"),
            maximum=_bounded_integer(payload["maximum"], f"{path}.maximum"),
        )


@dataclass(frozen=True, slots=True)
class ShapeSchema:
    """A named, ordered integer-domain contract for service lookup."""

    shape_schema_id: str
    axes: tuple[ShapeAxis, ...]
    schema: str = SHAPE_SCHEMA_SCHEMA

    def __post_init__(self) -> None:
        _require_schema(self.schema, SHAPE_SCHEMA_SCHEMA, "ShapeSchema.schema")
        _string(self.shape_schema_id, "ShapeSchema.shape_schema_id")
        if not isinstance(self.axes, tuple):
            _fail("ShapeSchema.axes", "in-memory contract requires a tuple")
        axis_ids: list[str] = []
        for index, axis in enumerate(self.axes):
            if not isinstance(axis, ShapeAxis):
                _fail(f"ShapeSchema.axes[{index}]", "expected ShapeAxis")
            axis_ids.append(axis.axis_id)
        if len(axis_ids) != len(set(axis_ids)):
            _fail("ShapeSchema.axes", "axis IDs must be unique")

    def to_obj(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "shape_schema_id": self.shape_schema_id,
            "axes": [axis.to_obj() for axis in self.axes],
        }

    @classmethod
    def from_obj(cls, value: object, path: str = "shape_schema") -> ShapeSchema:
        payload = _object(value, path)
        _fields(payload, path, {"schema", "shape_schema_id", "axes"})
        _require_schema(payload["schema"], SHAPE_SCHEMA_SCHEMA, f"{path}.schema")
        return cls(
            shape_schema_id=_string(
                payload["shape_schema_id"], f"{path}.shape_schema_id"
            ),
            axes=tuple(
                ShapeAxis.from_obj(item, f"{path}.axes[{index}]")
                for index, item in enumerate(_array(payload["axes"], f"{path}.axes"))
            ),
        )

    def validate_vector(self, vector: ShapeVector, path: str = "shape_vector") -> None:
        if not isinstance(vector, ShapeVector):
            _fail(path, "expected ShapeVector")
        if vector.shape_schema_id != self.shape_schema_id:
            _fail(
                f"{path}.shape_schema_id",
                f"expected {self.shape_schema_id!r}, got {vector.shape_schema_id!r}",
            )
        if len(vector.values) != len(self.axes):
            _fail(f"{path}.values", f"expected {len(self.axes)} values")
        for index, (axis, value) in enumerate(zip(self.axes, vector.values, strict=True)):
            if value < axis.minimum or value > axis.maximum:
                _fail(
                    f"{path}.values[{index}]",
                    f"outside {axis.axis_id!r} domain [{axis.minimum}, {axis.maximum}]",
                )


@dataclass(frozen=True, slots=True)
class ShapeVector:
    """Integer values aligned to one named :class:`ShapeSchema`."""

    shape_schema_id: str
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        _string(self.shape_schema_id, "ShapeVector.shape_schema_id")
        if not isinstance(self.values, tuple):
            _fail("ShapeVector.values", "in-memory contract requires a tuple")
        for index, value in enumerate(self.values):
            _bounded_integer(value, f"ShapeVector.values[{index}]")

    def to_obj(self) -> dict[str, Any]:
        return {"shape_schema_id": self.shape_schema_id, "values": list(self.values)}

    @classmethod
    def from_obj(cls, value: object, path: str = "shape_vector") -> ShapeVector:
        payload = _object(value, path)
        _fields(payload, path, {"shape_schema_id", "values"})
        return cls(
            shape_schema_id=_string(
                payload["shape_schema_id"], f"{path}.shape_schema_id"
            ),
            values=tuple(
                _bounded_integer(item, f"{path}.values[{index}]")
                for index, item in enumerate(
                    _array(payload["values"], f"{path}.values")
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class BinaryImplementationRef:
    """A target-native module or code-object implementation identity."""

    implementation_id: str
    vendor_id: str
    device_isa: str
    module_sha256: str
    function_sha256: str | None
    function_symbol: str | None
    backend_id: str | None
    algorithm_id: str | None
    launch_formula_id: str
    kind: str = "binary"

    def __post_init__(self) -> None:
        if self.kind != "binary":
            _fail("BinaryImplementationRef.kind", "expected 'binary'")
        for name in ("implementation_id", "vendor_id", "device_isa"):
            _string(getattr(self, name), f"BinaryImplementationRef.{name}")
        _trusted_identifier(
            self.launch_formula_id, "BinaryImplementationRef.launch_formula_id"
        )
        _sha256(self.module_sha256, "BinaryImplementationRef.module_sha256")
        _optional_sha256(
            self.function_sha256, "BinaryImplementationRef.function_sha256"
        )
        _optional_string(self.function_symbol, "BinaryImplementationRef.function_symbol")
        _optional_string(self.backend_id, "BinaryImplementationRef.backend_id")
        _optional_string(self.algorithm_id, "BinaryImplementationRef.algorithm_id")
        if (self.function_sha256 is None) == (self.function_symbol is None):
            _fail(
                "BinaryImplementationRef",
                "exactly one of function_sha256 and function_symbol must be present",
            )
        if self.backend_id is None and self.algorithm_id is None:
            _fail(
                "BinaryImplementationRef",
                "at least one of backend_id and algorithm_id must be present",
            )

    def to_obj(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "implementation_id": self.implementation_id,
            "vendor_id": self.vendor_id,
            "device_isa": self.device_isa,
            "module_sha256": self.module_sha256,
            "function_sha256": self.function_sha256,
            "function_symbol": self.function_symbol,
            "backend_id": self.backend_id,
            "algorithm_id": self.algorithm_id,
            "launch_formula_id": self.launch_formula_id,
        }

    @classmethod
    def from_obj(
        cls, value: object, path: str = "implementation_ref"
    ) -> BinaryImplementationRef:
        payload = _object(value, path)
        expected = {
            "kind",
            "implementation_id",
            "vendor_id",
            "device_isa",
            "module_sha256",
            "function_sha256",
            "function_symbol",
            "backend_id",
            "algorithm_id",
            "launch_formula_id",
        }
        _fields(payload, path, expected)
        if payload["kind"] != "binary":
            _fail(f"{path}.kind", "expected 'binary'")
        return cls(
            implementation_id=_string(
                payload["implementation_id"], f"{path}.implementation_id"
            ),
            vendor_id=_string(payload["vendor_id"], f"{path}.vendor_id"),
            device_isa=_string(payload["device_isa"], f"{path}.device_isa"),
            module_sha256=_sha256(payload["module_sha256"], f"{path}.module_sha256"),
            function_sha256=_optional_sha256(
                payload["function_sha256"], f"{path}.function_sha256"
            ),
            function_symbol=_optional_string(
                payload["function_symbol"], f"{path}.function_symbol"
            ),
            backend_id=_optional_string(payload["backend_id"], f"{path}.backend_id"),
            algorithm_id=_optional_string(
                payload["algorithm_id"], f"{path}.algorithm_id"
            ),
            launch_formula_id=_trusted_identifier(
                payload["launch_formula_id"], f"{path}.launch_formula_id"
            ),
        )


@dataclass(frozen=True, slots=True)
class AnalyticalImplementationRef:
    """Content-addressed declarative implementation for candidate transfer."""

    implementation_id: str
    model_sha256: str
    target_vendor_id: str
    target_architecture: str
    target_isa: str
    applicability_sha256: str
    trusted_evaluator_id: str
    parameter_sha256: str
    anchor_evidence_sha256: str
    delta_evidence_sha256: str
    kind: str = "analytical"

    def __post_init__(self) -> None:
        if self.kind != "analytical":
            _fail("AnalyticalImplementationRef.kind", "expected 'analytical'")
        for name in (
            "implementation_id",
            "target_vendor_id",
            "target_architecture",
            "target_isa",
            "trusted_evaluator_id",
        ):
            _string(getattr(self, name), f"AnalyticalImplementationRef.{name}")
        for name in (
            "model_sha256",
            "applicability_sha256",
            "parameter_sha256",
            "anchor_evidence_sha256",
            "delta_evidence_sha256",
        ):
            _sha256(getattr(self, name), f"AnalyticalImplementationRef.{name}")

    def to_obj(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "implementation_id": self.implementation_id,
            "model_sha256": self.model_sha256,
            "target_vendor_id": self.target_vendor_id,
            "target_architecture": self.target_architecture,
            "target_isa": self.target_isa,
            "applicability_sha256": self.applicability_sha256,
            "trusted_evaluator_id": self.trusted_evaluator_id,
            "parameter_sha256": self.parameter_sha256,
            "anchor_evidence_sha256": self.anchor_evidence_sha256,
            "delta_evidence_sha256": self.delta_evidence_sha256,
        }

    @classmethod
    def from_obj(
        cls, value: object, path: str = "implementation_ref"
    ) -> AnalyticalImplementationRef:
        payload = _object(value, path)
        expected = {
            "kind",
            "implementation_id",
            "model_sha256",
            "target_vendor_id",
            "target_architecture",
            "target_isa",
            "applicability_sha256",
            "trusted_evaluator_id",
            "parameter_sha256",
            "anchor_evidence_sha256",
            "delta_evidence_sha256",
        }
        _fields(payload, path, expected)
        if payload["kind"] != "analytical":
            _fail(f"{path}.kind", "expected 'analytical'")
        return cls(
            implementation_id=_string(
                payload["implementation_id"], f"{path}.implementation_id"
            ),
            model_sha256=_sha256(payload["model_sha256"], f"{path}.model_sha256"),
            target_vendor_id=_string(
                payload["target_vendor_id"], f"{path}.target_vendor_id"
            ),
            target_architecture=_string(
                payload["target_architecture"], f"{path}.target_architecture"
            ),
            target_isa=_string(payload["target_isa"], f"{path}.target_isa"),
            applicability_sha256=_sha256(
                payload["applicability_sha256"], f"{path}.applicability_sha256"
            ),
            trusted_evaluator_id=_string(
                payload["trusted_evaluator_id"], f"{path}.trusted_evaluator_id"
            ),
            parameter_sha256=_sha256(
                payload["parameter_sha256"], f"{path}.parameter_sha256"
            ),
            anchor_evidence_sha256=_sha256(
                payload["anchor_evidence_sha256"],
                f"{path}.anchor_evidence_sha256",
            ),
            delta_evidence_sha256=_sha256(
                payload["delta_evidence_sha256"], f"{path}.delta_evidence_sha256"
            ),
        )


ImplementationRef: TypeAlias = BinaryImplementationRef | AnalyticalImplementationRef


def implementation_ref_from_obj(
    value: object, path: str = "implementation_ref"
) -> ImplementationRef:
    payload = _object(value, path)
    kind = payload.get("kind")
    if kind == "binary":
        return BinaryImplementationRef.from_obj(payload, path)
    if kind == "analytical":
        return AnalyticalImplementationRef.from_obj(payload, path)
    _fail(f"{path}.kind", "expected 'binary' or 'analytical'")


def validate_shape_schemas(
    schemas: tuple[ShapeSchema, ...], path: str = "shape_schemas"
) -> dict[str, ShapeSchema]:
    """Validate model-level ordering and return an immutable-input index."""

    if not isinstance(schemas, tuple):
        _fail(path, "in-memory contract requires a tuple")
    identifiers: list[str] = []
    for index, schema in enumerate(schemas):
        if not isinstance(schema, ShapeSchema):
            raise TypeError(f"{path}[{index}]: expected ShapeSchema")
        identifiers.append(schema.shape_schema_id)
    _require_sorted_unique(tuple(identifiers), path)
    return {schema.shape_schema_id: schema for schema in schemas}


def _bounded_integer(
    value: object,
    path: str,
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> int:
    result = _integer(value, path, nonnegative=nonnegative or positive)
    if positive and result == 0:
        _fail(path, "must be positive")
    if result < SIGNED_128_MIN or result > SIGNED_128_MAX:
        _fail(path, "outside the signed 128-bit domain")
    return result


class ResourceAxisClass(str, Enum):
    """Closed class of one device-internal service currency."""

    THROUGHPUT = "throughput"
    RESIDENCY = "residency"
    EXCLUSIVE = "exclusive"


class ResourceServiceScope(str, Enum):
    """Ownership boundary for a measured device demand."""

    DEVICE_INTERNAL = "device-internal"
    PEER_PORT = "peer-port"
    DATA_MOVER = "data-mover"


class ThroughputRateTimebase(str, Enum):
    """Timebase selected by the presence of an axis clock domain."""

    WALL_PS = "wall-ps"
    DEVICE_CYCLE = "device-cycle"


@dataclass(frozen=True, slots=True)
class ExactRate:
    """Reduced nonnegative rational capacity."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        _bounded_integer(self.numerator, "ExactRate.numerator", nonnegative=True)
        _bounded_integer(self.denominator, "ExactRate.denominator", positive=True)
        if gcd(self.numerator, self.denominator) != 1:
            _fail("ExactRate", "numerator and denominator must be reduced")

    def to_obj(self) -> dict[str, int]:
        return {"numerator": self.numerator, "denominator": self.denominator}


@dataclass(frozen=True, slots=True)
class DeviceResourceAxis:
    """One typed capacity axis in a compact device model."""

    axis_id: str
    axis_class: ResourceAxisClass
    service_scope: ResourceServiceScope
    base_unit: str
    clock_domain_id: str | None
    capacity_source_id: str
    rate: ExactRate | None
    residency_capacity: int | None
    exclusive_capacity: int | None

    def __post_init__(self) -> None:
        _string(self.axis_id, "DeviceResourceAxis.axis_id")
        if not isinstance(self.axis_class, ResourceAxisClass):
            raise TypeError("DeviceResourceAxis.axis_class: expected ResourceAxisClass")
        if not isinstance(self.service_scope, ResourceServiceScope):
            raise TypeError(
                "DeviceResourceAxis.service_scope: expected ResourceServiceScope"
            )
        _string(self.base_unit, "DeviceResourceAxis.base_unit")
        _optional_string(self.clock_domain_id, "DeviceResourceAxis.clock_domain_id")
        _string(self.capacity_source_id, "DeviceResourceAxis.capacity_source_id")
        if self.axis_class is ResourceAxisClass.THROUGHPUT:
            if not isinstance(self.rate, ExactRate):
                raise TypeError("DeviceResourceAxis.rate: throughput requires ExactRate")
            if self.residency_capacity is not None or self.exclusive_capacity is not None:
                _fail(
                    "DeviceResourceAxis",
                    "throughput requires null residency and exclusive capacities",
                )
            return
        if self.rate is not None:
            _fail("DeviceResourceAxis.rate", "non-throughput axes reject a rate")
        if self.clock_domain_id is not None:
            _fail(
                "DeviceResourceAxis.clock_domain_id",
                "non-throughput axes reject a clock domain",
            )
        if self.axis_class is ResourceAxisClass.RESIDENCY:
            if self.residency_capacity is None or self.exclusive_capacity is not None:
                _fail(
                    "DeviceResourceAxis",
                    "residency requires only residency_capacity",
                )
            _bounded_integer(
                self.residency_capacity,
                "DeviceResourceAxis.residency_capacity",
                nonnegative=True,
            )
            return
        if self.exclusive_capacity is None or self.residency_capacity is not None:
            _fail(
                "DeviceResourceAxis",
                "exclusive requires only exclusive_capacity",
            )
        _bounded_integer(
            self.exclusive_capacity,
            "DeviceResourceAxis.exclusive_capacity",
            positive=True,
        )

    @property
    def throughput_timebase(self) -> ThroughputRateTimebase | None:
        """Return the rate timebase encoded by ``clock_domain_id``.

        ``base_unit`` names demand and never encodes rate time. A throughput
        axis with a clock domain is per device cycle; a null clock domain is
        per wall-clock picosecond. Non-throughput axes have no rate timebase.
        """

        if self.axis_class is not ResourceAxisClass.THROUGHPUT:
            return None
        if self.clock_domain_id is None:
            return ThroughputRateTimebase.WALL_PS
        return ThroughputRateTimebase.DEVICE_CYCLE

    def to_obj(self) -> dict[str, Any]:
        return {
            "axis_id": self.axis_id,
            "axis_class": self.axis_class.value,
            "service_scope": self.service_scope.value,
            "base_unit": self.base_unit,
            "clock_domain_id": self.clock_domain_id,
            "capacity_source_id": self.capacity_source_id,
            "rate": None if self.rate is None else self.rate.to_obj(),
            "residency_capacity": self.residency_capacity,
            "exclusive_capacity": self.exclusive_capacity,
        }


@dataclass(frozen=True, slots=True)
class DeviceResourceRegistry:
    """Canonical ordered resource axes for one device kind."""

    device_kind_id: str
    active_axis_ids: tuple[str, ...]
    axes: tuple[DeviceResourceAxis, ...]
    schema: str = DEVICE_RESOURCE_REGISTRY_SCHEMA

    def __post_init__(self) -> None:
        _require_schema(
            self.schema,
            DEVICE_RESOURCE_REGISTRY_SCHEMA,
            "DeviceResourceRegistry.schema",
        )
        _string(self.device_kind_id, "DeviceResourceRegistry.device_kind_id")
        if not isinstance(self.active_axis_ids, tuple):
            raise TypeError(
                "DeviceResourceRegistry.active_axis_ids: in-memory contract requires a tuple"
            )
        if not isinstance(self.axes, tuple):
            raise TypeError(
                "DeviceResourceRegistry.axes: in-memory contract requires a tuple"
            )
        axis_ids: list[str] = []
        for index, axis in enumerate(self.axes):
            if not isinstance(axis, DeviceResourceAxis):
                raise TypeError(
                    f"DeviceResourceRegistry.axes[{index}]: expected DeviceResourceAxis"
                )
            axis_ids.append(axis.axis_id)
        _require_sorted_unique(tuple(axis_ids), "DeviceResourceRegistry.axes")
        _require_sorted_unique(
            self.active_axis_ids, "DeviceResourceRegistry.active_axis_ids"
        )
        unknown = sorted(set(self.active_axis_ids) - set(axis_ids))
        if unknown:
            _fail(
                "DeviceResourceRegistry.active_axis_ids",
                f"unknown axis IDs {unknown}",
            )

    @property
    def axis_ids(self) -> tuple[str, ...]:
        return tuple(axis.axis_id for axis in self.axes)

    def to_obj(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "device_kind_id": self.device_kind_id,
            "active_axis_ids": list(self.active_axis_ids),
            "axes": [axis.to_obj() for axis in self.axes],
        }


@dataclass(frozen=True, slots=True)
class DeviceResourceVector:
    """Dense demand and known-mask vectors aligned to a registry."""

    registry_sha256: str
    device_kind_id: str
    values: tuple[int, ...]
    known: tuple[bool, ...]

    def __post_init__(self) -> None:
        _sha256(self.registry_sha256, "DeviceResourceVector.registry_sha256")
        _string(self.device_kind_id, "DeviceResourceVector.device_kind_id")
        if not isinstance(self.values, tuple):
            raise TypeError("DeviceResourceVector.values: in-memory contract requires a tuple")
        if not isinstance(self.known, tuple):
            raise TypeError("DeviceResourceVector.known: in-memory contract requires a tuple")
        if len(self.values) != len(self.known):
            _fail("DeviceResourceVector", "values and known must have equal length")
        for index, value in enumerate(self.values):
            _bounded_integer(
                value,
                f"DeviceResourceVector.values[{index}]",
                nonnegative=True,
            )
        for index, known in enumerate(self.known):
            if type(known) is not bool:
                raise TypeError(
                    f"DeviceResourceVector.known[{index}]: expected a boolean"
                )

    def to_obj(self) -> dict[str, Any]:
        return {
            "registry_sha256": self.registry_sha256,
            "device_kind_id": self.device_kind_id,
            "values": list(self.values),
            "known": list(self.known),
        }

    def validate_against(
        self,
        registry: DeviceResourceRegistry,
        registry_sha256: str,
        path: str = "resource_vector",
    ) -> None:
        _sha256(registry_sha256, "registry_sha256")
        if self.registry_sha256 != registry_sha256:
            _fail(f"{path}.registry_sha256", "does not match the selected registry")
        self.validate_registry_structure(registry, path)

    def validate_registry_structure(
        self,
        registry: DeviceResourceRegistry,
        path: str = "resource_vector",
    ) -> None:
        """Validate inline registry alignment without calculating its digest.

        Canonical hashing stays outside this serving-safe module. A caller that
        has calculated the inline registry digest must use :meth:`validate_against`.
        """

        if not isinstance(registry, DeviceResourceRegistry):
            raise TypeError(f"{path}: expected DeviceResourceRegistry")
        if self.device_kind_id != registry.device_kind_id:
            _fail(f"{path}.device_kind_id", "does not match the selected registry")
        if len(self.values) != len(registry.axes):
            _fail(f"{path}.values", f"expected {len(registry.axes)} aligned values")
        active = set(registry.active_axis_ids)
        for index, axis in enumerate(registry.axes):
            if axis.axis_id in active:
                if not self.known[index]:
                    _fail(f"{path}.known[{index}]", "active-axis demand must be known")
            elif self.known[index] or self.values[index] != 0:
                _fail(
                    f"{path}[{index}]",
                    "inactive axes require an unknown bit and canonical zero placeholder",
                )


@dataclass(frozen=True, slots=True)
class ServiceEpochDefinition:
    """Immutable demand and optional floor for one ordered service epoch."""

    resource_vector: DeviceResourceVector
    fixed_floor_ps: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.resource_vector, DeviceResourceVector):
            raise TypeError(
                "ServiceEpochDefinition.resource_vector: expected DeviceResourceVector"
            )
        if self.fixed_floor_ps is not None:
            _bounded_integer(
                self.fixed_floor_ps,
                "ServiceEpochDefinition.fixed_floor_ps",
                nonnegative=True,
            )

    def to_obj(self) -> dict[str, Any]:
        return {
            "resource_vector": self.resource_vector.to_obj(),
            "fixed_floor_ps": self.fixed_floor_ps,
        }


@dataclass(frozen=True, slots=True)
class DeviceServiceEntry:
    """Exact implementation and shape cell with ordered immutable epochs."""

    implementation_id: str
    shape_vector: ShapeVector
    epochs: tuple[ServiceEpochDefinition, ...]

    def __post_init__(self) -> None:
        _string(self.implementation_id, "DeviceServiceEntry.implementation_id")
        if not isinstance(self.shape_vector, ShapeVector):
            raise TypeError("DeviceServiceEntry.shape_vector: expected ShapeVector")
        if not isinstance(self.epochs, tuple):
            raise TypeError("DeviceServiceEntry.epochs: in-memory contract requires a tuple")
        if not self.epochs:
            _fail("DeviceServiceEntry.epochs", "must not be empty")
        for index, epoch in enumerate(self.epochs):
            if not isinstance(epoch, ServiceEpochDefinition):
                raise TypeError(
                    f"DeviceServiceEntry.epochs[{index}]: expected ServiceEpochDefinition"
                )

    @property
    def key(self) -> tuple[str, ShapeVector]:
        return self.implementation_id, self.shape_vector

    def to_obj(self) -> dict[str, Any]:
        return {
            "implementation_id": self.implementation_id,
            "shape_vector": self.shape_vector.to_obj(),
            "epochs": [epoch.to_obj() for epoch in self.epochs],
        }


@dataclass(frozen=True, slots=True)
class ResourceInteractionContract:
    """Closed version-1 resource law declaration."""

    interaction_law: str
    interaction_terms: tuple[object, ...]

    def __post_init__(self) -> None:
        if self.interaction_law != "independent-resource-v1":
            _fail(
                "ResourceInteractionContract.interaction_law",
                "version 1 accepts only 'independent-resource-v1'",
            )
        if not isinstance(self.interaction_terms, tuple):
            raise TypeError(
                "ResourceInteractionContract.interaction_terms: in-memory contract "
                "requires a tuple"
            )
        if self.interaction_terms:
            _fail(
                "ResourceInteractionContract.interaction_terms",
                "version 1 rejects interaction terms",
            )

    def to_obj(self) -> dict[str, Any]:
        return {
            "interaction_law": self.interaction_law,
            "interaction_terms": [],
        }


class DeviceModelAcceptanceStatus(str, Enum):
    CANDIDATE = "candidate"
    VALIDATED = "validated"


class DeviceModelTargetBasis(str, Enum):
    TARGET_SILICON = "target-silicon"
    ARCHITECTURE_DERIVED = "architecture-derived"


@dataclass(frozen=True, slots=True)
class DeviceServiceEntryRecord:
    """Stable lookup ID wrapped around one exact service cell."""

    service_entry_id: str
    entry: DeviceServiceEntry

    def __post_init__(self) -> None:
        _string(self.service_entry_id, "DeviceServiceEntryRecord.service_entry_id")
        if not isinstance(self.entry, DeviceServiceEntry):
            raise TypeError("DeviceServiceEntryRecord.entry: expected DeviceServiceEntry")

    def to_obj(self) -> dict[str, Any]:
        return {"service_entry_id": self.service_entry_id, "entry": self.entry.to_obj()}


class ServiceEntrySourceSelection(str, Enum):
    """Closed source decision for one accepted service entry."""

    SILICON = "silicon"
    SILICON_FIT = "silicon-fit"
    ACCEL_SIM = "accel-sim"
    ANALYTICAL_TRANSFER = "analytical-transfer"
    SIMULATOR_DERIVED = "simulator-derived"


@dataclass(frozen=True, slots=True)
class ServiceEntryEvidence:
    """Compact provenance and uncertainty for one service entry."""

    service_entry_id: str
    source_selection: ServiceEntrySourceSelection
    source_record_sha256s: tuple[str, ...]
    residual_record_sha256: str
    support_envelope_sha256: str
    operating_envelope_sha256: str
    isolated_duration_ps: int
    uncertainty_bound: ExactRate

    def __post_init__(self) -> None:
        _string(self.service_entry_id, "ServiceEntryEvidence.service_entry_id")
        if not isinstance(self.source_selection, ServiceEntrySourceSelection):
            raise TypeError(
                "ServiceEntryEvidence.source_selection: expected "
                "ServiceEntrySourceSelection"
            )
        if not isinstance(self.source_record_sha256s, tuple):
            raise TypeError(
                "ServiceEntryEvidence.source_record_sha256s: in-memory contract "
                "requires a tuple"
            )
        if not self.source_record_sha256s:
            _fail("ServiceEntryEvidence.source_record_sha256s", "must not be empty")
        for index, digest in enumerate(self.source_record_sha256s):
            _sha256(digest, f"ServiceEntryEvidence.source_record_sha256s[{index}]")
        _require_sorted_unique(
            self.source_record_sha256s,
            "ServiceEntryEvidence.source_record_sha256s",
        )
        for name in (
            "residual_record_sha256",
            "support_envelope_sha256",
            "operating_envelope_sha256",
        ):
            _sha256(getattr(self, name), f"ServiceEntryEvidence.{name}")
        _bounded_integer(
            self.isolated_duration_ps,
            "ServiceEntryEvidence.isolated_duration_ps",
            nonnegative=True,
        )
        if not isinstance(self.uncertainty_bound, ExactRate):
            raise TypeError(
                "ServiceEntryEvidence.uncertainty_bound: expected ExactRate"
            )

    def to_obj(self) -> dict[str, Any]:
        return {
            "service_entry_id": self.service_entry_id,
            "source_selection": self.source_selection.value,
            "source_record_sha256s": list(self.source_record_sha256s),
            "residual_record_sha256": self.residual_record_sha256,
            "support_envelope_sha256": self.support_envelope_sha256,
            "operating_envelope_sha256": self.operating_envelope_sha256,
            "isolated_duration_ps": self.isolated_duration_ps,
            "uncertainty_bound": self.uncertainty_bound.to_obj(),
        }


@dataclass(frozen=True, slots=True)
class DeviceModelLimits:
    """Positive static and runtime bounds carried by a compact model."""

    max_shape_schemas: int
    max_shape_axes_per_schema: int
    max_resource_axes: int
    max_service_entries: int
    max_epochs_per_entry: int
    max_resident_entries: int

    def __post_init__(self) -> None:
        for name in (
            "max_shape_schemas",
            "max_shape_axes_per_schema",
            "max_resource_axes",
            "max_service_entries",
            "max_epochs_per_entry",
            "max_resident_entries",
        ):
            _bounded_integer(
                getattr(self, name), f"DeviceModelLimits.{name}", positive=True
            )

    def to_obj(self) -> dict[str, int]:
        return {
            "max_shape_schemas": self.max_shape_schemas,
            "max_shape_axes_per_schema": self.max_shape_axes_per_schema,
            "max_resource_axes": self.max_resource_axes,
            "max_service_entries": self.max_service_entries,
            "max_epochs_per_entry": self.max_epochs_per_entry,
            "max_resident_entries": self.max_resident_entries,
        }


@dataclass(frozen=True, slots=True)
class DeviceModel:
    """Complete strict compact device model safe for online loading.

    Content references are syntax checked here. Resolving those references and
    checking identities inside referenced records belongs to the offline
    release-closure validator.
    """

    device_model_id: str
    device_kind_id: str
    acceptance_status: DeviceModelAcceptanceStatus
    target_basis: DeviceModelTargetBasis
    device_identity_sha256: str
    operating_envelope_sha256: str
    support_envelope_sha256: str
    evidence_manifest_sha256: str
    fit_sha256: str
    expectations_commit: str
    dispatch_signature_sha256s: tuple[str, ...]
    shape_schemas: tuple[ShapeSchema, ...]
    implementation_selector_sha256: str
    collective_stage_selector_sha256: str | None
    resource_registry: DeviceResourceRegistry
    interaction_contract: ResourceInteractionContract
    host_initiation_profile_sha256: str | None
    service_entries: tuple[DeviceServiceEntryRecord, ...]
    service_entry_evidence: tuple[ServiceEntryEvidence, ...]
    scalar_profile_table_sha256: str | None
    gpu_spec_sha256: str | None
    gpu_architecture_profile_sha256: str | None
    gpu_device_config_sha256: str | None
    validation_record_sha256: str
    validation_summary_sha256: str
    acceptance_bars_sha256: str
    model_limits: DeviceModelLimits
    schema: str = DEVICE_MODEL_SCHEMA

    def __post_init__(self) -> None:
        _require_schema(self.schema, DEVICE_MODEL_SCHEMA, "DeviceModel.schema")
        _string(self.device_model_id, "DeviceModel.device_model_id")
        _string(self.device_kind_id, "DeviceModel.device_kind_id")
        if not isinstance(self.acceptance_status, DeviceModelAcceptanceStatus):
            raise TypeError(
                "DeviceModel.acceptance_status: expected DeviceModelAcceptanceStatus"
            )
        if not isinstance(self.target_basis, DeviceModelTargetBasis):
            raise TypeError("DeviceModel.target_basis: expected DeviceModelTargetBasis")
        if (
            self.target_basis is DeviceModelTargetBasis.ARCHITECTURE_DERIVED
            and self.acceptance_status is not DeviceModelAcceptanceStatus.CANDIDATE
        ):
            _fail("DeviceModel", "architecture-derived models must remain candidate")

        for name in (
            "device_identity_sha256",
            "operating_envelope_sha256",
            "support_envelope_sha256",
            "evidence_manifest_sha256",
            "fit_sha256",
            "implementation_selector_sha256",
            "validation_record_sha256",
            "validation_summary_sha256",
            "acceptance_bars_sha256",
        ):
            _sha256(getattr(self, name), f"DeviceModel.{name}")
        for name in (
            "collective_stage_selector_sha256",
            "host_initiation_profile_sha256",
            "scalar_profile_table_sha256",
            "gpu_spec_sha256",
            "gpu_architecture_profile_sha256",
            "gpu_device_config_sha256",
        ):
            _optional_sha256(getattr(self, name), f"DeviceModel.{name}")
        _git_object_id(self.expectations_commit, "DeviceModel.expectations_commit")

        if not isinstance(self.dispatch_signature_sha256s, tuple):
            raise TypeError(
                "DeviceModel.dispatch_signature_sha256s: in-memory contract "
                "requires a tuple"
            )
        if not self.dispatch_signature_sha256s:
            _fail("DeviceModel.dispatch_signature_sha256s", "must not be empty")
        for index, digest in enumerate(self.dispatch_signature_sha256s):
            _sha256(digest, f"DeviceModel.dispatch_signature_sha256s[{index}]")
        _require_sorted_unique(
            self.dispatch_signature_sha256s,
            "DeviceModel.dispatch_signature_sha256s",
        )

        if not isinstance(self.shape_schemas, tuple):
            raise TypeError(
                "DeviceModel.shape_schemas: in-memory contract requires a tuple"
            )
        if not self.shape_schemas:
            _fail("DeviceModel.shape_schemas", "must not be empty")
        validate_shape_schemas(self.shape_schemas, "DeviceModel.shape_schemas")

        if not isinstance(self.resource_registry, DeviceResourceRegistry):
            raise TypeError(
                "DeviceModel.resource_registry: expected DeviceResourceRegistry"
            )
        if self.resource_registry.device_kind_id != self.device_kind_id:
            _fail(
                "DeviceModel.resource_registry.device_kind_id",
                "does not match the model device kind",
            )
        if not isinstance(self.interaction_contract, ResourceInteractionContract):
            raise TypeError(
                "DeviceModel.interaction_contract: expected ResourceInteractionContract"
            )

        if not isinstance(self.service_entries, tuple):
            raise TypeError(
                "DeviceModel.service_entries: in-memory contract requires a tuple"
            )
        if not self.service_entries:
            _fail("DeviceModel.service_entries", "must not be empty")
        entry_ids: list[str] = []
        entries: list[DeviceServiceEntry] = []
        for index, record in enumerate(self.service_entries):
            if not isinstance(record, DeviceServiceEntryRecord):
                raise TypeError(
                    f"DeviceModel.service_entries[{index}]: expected "
                    "DeviceServiceEntryRecord"
                )
            entry_ids.append(record.service_entry_id)
            entries.append(record.entry)
        _require_sorted_unique(tuple(entry_ids), "DeviceModel.service_entries")
        _validate_device_service_entry_structure(
            registry=self.resource_registry,
            shape_schemas=self.shape_schemas,
            entries=tuple(entries),
        )
        declared_registry_digests = {
            epoch.resource_vector.registry_sha256
            for entry in entries
            for epoch in entry.epochs
        }
        if len(declared_registry_digests) != 1:
            _fail(
                "DeviceModel.service_entries",
                "all resource vectors must declare one registry digest",
            )

        if not isinstance(self.service_entry_evidence, tuple):
            raise TypeError(
                "DeviceModel.service_entry_evidence: in-memory contract requires a tuple"
            )
        evidence_ids: list[str] = []
        for index, evidence in enumerate(self.service_entry_evidence):
            if not isinstance(evidence, ServiceEntryEvidence):
                raise TypeError(
                    f"DeviceModel.service_entry_evidence[{index}]: expected "
                    "ServiceEntryEvidence"
                )
            evidence_ids.append(evidence.service_entry_id)
            if evidence.support_envelope_sha256 != self.support_envelope_sha256:
                _fail(
                    f"DeviceModel.service_entry_evidence[{index}]"
                    ".support_envelope_sha256",
                    "does not match the model support envelope",
                )
            if evidence.operating_envelope_sha256 != self.operating_envelope_sha256:
                _fail(
                    f"DeviceModel.service_entry_evidence[{index}]"
                    ".operating_envelope_sha256",
                    "does not match the model operating envelope",
                )
        if tuple(evidence_ids) != tuple(entry_ids):
            _fail(
                "DeviceModel.service_entry_evidence",
                "must be sorted one-to-one with service entries",
            )

        if not isinstance(self.model_limits, DeviceModelLimits):
            raise TypeError("DeviceModel.model_limits: expected DeviceModelLimits")
        inline_counts = (
            (
                len(self.shape_schemas),
                self.model_limits.max_shape_schemas,
                "max_shape_schemas",
            ),
            (
                max((len(schema.axes) for schema in self.shape_schemas), default=0),
                self.model_limits.max_shape_axes_per_schema,
                "max_shape_axes_per_schema",
            ),
            (
                len(self.resource_registry.axes),
                self.model_limits.max_resource_axes,
                "max_resource_axes",
            ),
            (
                len(self.service_entries),
                self.model_limits.max_service_entries,
                "max_service_entries",
            ),
            (
                max((len(entry.epochs) for entry in entries), default=0),
                self.model_limits.max_epochs_per_entry,
                "max_epochs_per_entry",
            ),
        )
        for count, limit, name in inline_counts:
            if count > limit:
                _fail(
                    f"DeviceModel.model_limits.{name}",
                    f"limit {limit} is below inline count {count}",
                )

    @property
    def declared_resource_registry_sha256(self) -> str:
        """Return the single digest declared by every inline resource vector."""

        first_entry = self.service_entries[0].entry
        return first_entry.epochs[0].resource_vector.registry_sha256

    def validate_registry_sha256(self, registry_sha256: str) -> None:
        """Check vectors against a caller-computed inline registry digest."""

        _sha256(registry_sha256, "registry_sha256")
        for entry_index, record in enumerate(self.service_entries):
            for epoch_index, epoch in enumerate(record.entry.epochs):
                epoch.resource_vector.validate_against(
                    self.resource_registry,
                    registry_sha256,
                    f"service_entries[{entry_index}].entry.epochs[{epoch_index}]"
                    ".resource_vector",
                )

    def to_obj(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "device_model_id": self.device_model_id,
            "device_kind_id": self.device_kind_id,
            "acceptance_status": self.acceptance_status.value,
            "target_basis": self.target_basis.value,
            "device_identity_sha256": self.device_identity_sha256,
            "operating_envelope_sha256": self.operating_envelope_sha256,
            "support_envelope_sha256": self.support_envelope_sha256,
            "evidence_manifest_sha256": self.evidence_manifest_sha256,
            "fit_sha256": self.fit_sha256,
            "expectations_commit": self.expectations_commit,
            "dispatch_signature_sha256s": list(self.dispatch_signature_sha256s),
            "shape_schemas": [schema.to_obj() for schema in self.shape_schemas],
            "implementation_selector_sha256": self.implementation_selector_sha256,
            "collective_stage_selector_sha256": (
                self.collective_stage_selector_sha256
            ),
            "resource_registry": self.resource_registry.to_obj(),
            "interaction_contract": self.interaction_contract.to_obj(),
            "host_initiation_profile_sha256": self.host_initiation_profile_sha256,
            "service_entries": [record.to_obj() for record in self.service_entries],
            "service_entry_evidence": [
                evidence.to_obj() for evidence in self.service_entry_evidence
            ],
            "scalar_profile_table_sha256": self.scalar_profile_table_sha256,
            "gpu_spec_sha256": self.gpu_spec_sha256,
            "gpu_architecture_profile_sha256": (
                self.gpu_architecture_profile_sha256
            ),
            "gpu_device_config_sha256": self.gpu_device_config_sha256,
            "validation_record_sha256": self.validation_record_sha256,
            "validation_summary_sha256": self.validation_summary_sha256,
            "acceptance_bars_sha256": self.acceptance_bars_sha256,
            "model_limits": self.model_limits.to_obj(),
        }


def _validate_device_service_entry_structure(
    *,
    registry: DeviceResourceRegistry,
    shape_schemas: tuple[ShapeSchema, ...],
    entries: tuple[DeviceServiceEntry, ...],
) -> None:
    """Validate exact cells, inline alignment, and demanded capacities."""

    schemas = validate_shape_schemas(shape_schemas)
    if not isinstance(entries, tuple):
        raise TypeError("service_entries: in-memory contract requires a tuple")
    keys: list[tuple[str, ShapeVector]] = []
    demanded_axes: set[int] = set()
    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, DeviceServiceEntry):
            raise TypeError(
                f"service_entries[{entry_index}]: expected DeviceServiceEntry"
            )
        keys.append(entry.key)
        schema = schemas.get(entry.shape_vector.shape_schema_id)
        if schema is None:
            _fail(
                f"service_entries[{entry_index}].shape_vector.shape_schema_id",
                "unknown shape schema",
            )
        schema.validate_vector(
            entry.shape_vector, f"service_entries[{entry_index}].shape_vector"
        )
        for epoch_index, epoch in enumerate(entry.epochs):
            path = f"service_entries[{entry_index}].epochs[{epoch_index}].resource_vector"
            epoch.resource_vector.validate_registry_structure(registry, path)
            demanded_axes.update(
                index
                for index, value in enumerate(epoch.resource_vector.values)
                if epoch.resource_vector.known[index] and value > 0
            )
    if len(keys) != len(set(keys)):
        _fail("service_entries", "duplicate implementation and shape cell")
    for index in demanded_axes:
        axis = registry.axes[index]
        if axis.axis_class is ResourceAxisClass.THROUGHPUT:
            assert axis.rate is not None
            positive = axis.rate.numerator > 0
        elif axis.axis_class is ResourceAxisClass.RESIDENCY:
            assert axis.residency_capacity is not None
            positive = axis.residency_capacity > 0
        else:
            assert axis.exclusive_capacity is not None
            positive = axis.exclusive_capacity > 0
        if not positive:
            _fail(
                f"resource_registry.axes[{index}]",
                "positive accepted demand requires positive capacity",
            )


def validate_device_service_entries(
    *,
    registry: DeviceResourceRegistry,
    registry_sha256: str,
    shape_schemas: tuple[ShapeSchema, ...],
    entries: tuple[DeviceServiceEntry, ...],
) -> None:
    """Validate exact cells, registry digest, and demanded capacities."""

    _sha256(registry_sha256, "registry_sha256")
    _validate_device_service_entry_structure(
        registry=registry,
        shape_schemas=shape_schemas,
        entries=entries,
    )
    for entry_index, entry in enumerate(entries):
        for epoch_index, epoch in enumerate(entry.epochs):
            if epoch.resource_vector.registry_sha256 != registry_sha256:
                _fail(
                    f"service_entries[{entry_index}].epochs[{epoch_index}]"
                    ".resource_vector.registry_sha256",
                    "does not match the selected registry",
                )


def validate_collective_stage_service_entries(
    service_entry_ids: Iterable[str],
    entries_by_id: Mapping[str, DeviceServiceEntry],
    registry: DeviceResourceRegistry,
    registry_sha256: str,
) -> None:
    """Reject peer-port or mover demand in version-1 resident stages."""

    for service_entry_id in service_entry_ids:
        _string(service_entry_id, "service_entry_id")
        entry = entries_by_id.get(service_entry_id)
        if entry is None:
            _fail("collective_stage", f"unknown service entry {service_entry_id!r}")
        for epoch_index, epoch in enumerate(entry.epochs):
            epoch.resource_vector.validate_against(
                registry,
                registry_sha256,
                f"collective_stage.epochs[{epoch_index}].resource_vector",
            )
            for axis_index, value in enumerate(epoch.resource_vector.values):
                if not epoch.resource_vector.known[axis_index] or value == 0:
                    continue
                axis = registry.axes[axis_index]
                if axis.service_scope is not ResourceServiceScope.DEVICE_INTERNAL:
                    _fail(
                        f"collective_stage.epochs[{epoch_index}].resource_vector"
                        f".values[{axis_index}]",
                        "version 1 collective stages permit positive demand only "
                        "on device-internal axes",
                    )


__all__ = [
    "DEVICE_MODEL_SCHEMA",
    "DEVICE_RESOURCE_REGISTRY_SCHEMA",
    "SHAPE_SCHEMA_SCHEMA",
    "SIGNED_128_MAX",
    "SIGNED_128_MIN",
    "AnalyticalImplementationRef",
    "BinaryImplementationRef",
    "DeviceModel",
    "DeviceModelAcceptanceStatus",
    "DeviceModelLimits",
    "DeviceModelTargetBasis",
    "DeviceResourceAxis",
    "DeviceResourceRegistry",
    "DeviceResourceVector",
    "DeviceServiceEntry",
    "DeviceServiceEntryRecord",
    "ExactRate",
    "ImplementationRef",
    "ResourceAxisClass",
    "ResourceInteractionContract",
    "ResourceServiceScope",
    "ServiceEntryEvidence",
    "ServiceEntrySourceSelection",
    "ServiceEpochDefinition",
    "ShapeAxis",
    "ShapeSchema",
    "ShapeVector",
    "ThroughputRateTimebase",
    "implementation_ref_from_obj",
    "validate_collective_stage_service_entries",
    "validate_device_service_entries",
    "validate_shape_schemas",
]
