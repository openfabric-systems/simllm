"""Immutable source and parameter closure for the external serving composition."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from .canonical import canonical_bytes, canonical_loads, canonical_sha256, validate_sha256
from .record_types import RecordObject
from .store import ObjectStore

if TYPE_CHECKING:
    from .external_db import ExternalOperationDatabase

EXTERNAL_COMPOSITION_SCHEMA = "simllm-external-serving-composition-v1"
_SOURCE_TABLE_SHA256 = "c6778a81cdc6078ce74f06733e4bce9d99a92b4ab3eccba4a83d14e7d063a09e"
_SOURCE_TABLE_CONTENT_SHA256 = "016a6fe8f099d60bed5aff52ff0f34ed2ea6dc3a479c258f8581c7f37e606a82"
_OPERATION_MANIFEST_SHA256 = "6be827d3dca584dce57594a2dc190b441e3de8ca932242fa71cfe56b59a89344"
COMPOSITION_UNSET = object()
EXTERNAL_ADJUSTMENT_IDS = (
    "prefill_latency_correction",
    "decode_latency_correction",
    "prefill_rate_matching_degradation",
    "decode_rate_matching_degradation",
    "autoscale_ttft_correction",
    "memory_bandwidth_empirical_scale",
    "memory_empirical_constant_latency",
    "context_attention_extra_latency_correction",
)
EXTERNAL_OPERATION_ADJUSTMENTS = EXTERNAL_ADJUSTMENT_IDS[5:]
_RATE_IDS = frozenset(EXTERNAL_ADJUSTMENT_IDS[2:4])
_FIELDS = {
    "schema",
    "scope",
    "source_table",
    "source_table_sha256",
    "operation_source",
    "operation_manifest_sha256",
    "runtime_objects_sha256",
    "source_values_hex",
    "overrides_hex",
    "parent_record_sha256",
    "intervention_reason",
}


class ExternalCompositionError(ValueError):
    """An external composition record or its selected source is inconsistent."""


def _legacy_json(value: Any) -> bytes:
    # This is the imported database's existing file recipe, not record identity.
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_legacy_json(value)).hexdigest()


def _fields(value: Any, expected: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ExternalCompositionError(f"{path}: expected exactly {sorted(expected)}")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ExternalCompositionError(f"{path}: expected nonblank text")
    return value


def _location(value: Any, path: str) -> None:
    _fields(value, {"path", "sha256", "start_line", "end_line"}, path)
    name = _text(value["path"], f"{path}.path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(part in {"..", "."} for part in name.split("/")):
        raise ExternalCompositionError(f"{path}.path: expected a relative source path")
    if "\\" in name or ":" in name or "~" in name:
        raise ExternalCompositionError(f"{path}.path: expected a portable source path")
    validate_sha256(value["sha256"], f"{path}.sha256")
    start, end = value["start_line"], value["end_line"]
    if type(start) is not int or type(end) is not int or start < 1 or end < start:
        raise ExternalCompositionError(f"{path}: expected an ordered positive line span")


def _number(value: Any, adjustment_id: str) -> float:
    if not isinstance(value, str):
        raise ExternalCompositionError(f"{adjustment_id}: expected a binary64 hex string")
    try:
        number = float.fromhex(value)
    except ValueError as error:
        raise ExternalCompositionError(f"{adjustment_id}: invalid binary64 value") from error
    if not math.isfinite(number) or number.hex() != value:
        raise ExternalCompositionError(f"{adjustment_id}: expected finite canonical binary64")
    if adjustment_id == "memory_empirical_constant_latency":
        legal = number >= 0 and value != (-0.0).hex()
    elif adjustment_id in _RATE_IDS or adjustment_id == "memory_bandwidth_empirical_scale":
        legal = 0 < number <= 1
    else:
        legal = number >= 1
    if not legal:
        raise ExternalCompositionError(f"{adjustment_id}: value is outside its physical role")
    return number


def _source_values(table: Mapping[str, Any]) -> dict[str, str]:
    _fields(
        table,
        {"schema", "scope", "external_packages", "phase_assignment_finding", "adjustments"},
        "source_table",
    )
    if table["schema"] != "simllm-matched-seam-external-adjustments-v1":
        raise ExternalCompositionError("source_table: unsupported source schema")
    _text(table["scope"], "source_table.scope")
    _text(table["phase_assignment_finding"], "source_table.phase_assignment_finding")
    _fields(
        table["external_packages"],
        {"aiconfigurator", "aiconfigurator-core"},
        "source_table.external_packages",
    )
    for name, version in table["external_packages"].items():
        _text(version, f"source_table.external_packages.{name}")
    rows = table["adjustments"]
    if not isinstance(rows, (list, tuple)) or len(rows) != 8:
        raise ExternalCompositionError("source_table: expected the eight adjustments")
    values = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ExternalCompositionError("source_table.adjustments: expected objects")
        name = row.get("id")
        if name not in EXTERNAL_ADJUSTMENT_IDS or name in values:
            raise ExternalCompositionError("source_table: duplicate or unknown adjustment")
        fields = {
            "id",
            "value",
            "removal_value",
            "kind",
            "applied_to",
            "source",
            "documentation",
            "family_r_reachable",
        }
        if name == "memory_empirical_constant_latency":
            fields.add("units")
            if row.get("units") != "seconds":
                raise ExternalCompositionError("memory constant must use seconds")
        _fields(row, fields, f"source_table.{name}")
        for field in ("value", "removal_value", "kind"):
            _text(row[field], f"source_table.{name}.{field}")
        if type(row["family_r_reachable"]) is not bool:
            raise ExternalCompositionError("family_r_reachable must be boolean")
        if not isinstance(row["applied_to"], (list, tuple)) or not row["applied_to"]:
            raise ExternalCompositionError("applied_to must contain phase roles")
        for role in row["applied_to"]:
            _text(role, "applied_to")
        _location(row["source"], f"source_table.{name}.source")
        documentation = dict(row["documentation"])
        _text(documentation.pop("meaning", None), f"source_table.{name}.meaning")
        _location(documentation, f"source_table.{name}.documentation")
        try:
            spelling = float(row["value"]).hex()
            removal = float(row["removal_value"]).hex()
        except ValueError as error:
            raise ExternalCompositionError("source values must be decimal numbers") from error
        _number(spelling, name)
        _number(removal, name)
        values[name] = spelling
    return values


def _runtime_objects(database: ExternalOperationDatabase) -> dict[str, str]:
    return {
        "system": _digest(database.system_spec),
        "model": _digest(database.model_config),
        "family_mapping": _digest(database.family_mapping),
    }


@dataclass(frozen=True, slots=True)
class ExternalServingComposition:
    """One verified record with source values and declared interventions."""

    record: RecordObject

    def __post_init__(self) -> None:
        if not isinstance(self.record, RecordObject):
            raise TypeError("record must be RecordObject")
        verified = RecordObject.from_bytes(
            self.record.canonical, expected_schema=EXTERNAL_COMPOSITION_SCHEMA
        )
        if (
            verified.record_id != self.record.record_id
            or verified.schema != self.record.schema
            or canonical_bytes(self.record.value) != verified.canonical
        ):
            raise ExternalCompositionError("record envelope disagrees with authoritative bytes")
        value = canonical_loads(verified.canonical)
        _fields(value, _FIELDS, "composition")
        if value["scope"] != "qwen3-32b-fp8-h200-trtllm-serving":
            raise ExternalCompositionError("unsupported composition scope")
        for field in ("source_table_sha256", "operation_manifest_sha256"):
            validate_sha256(value[field], field)
        if (
            value["source_table_sha256"] != _SOURCE_TABLE_SHA256
            or value["operation_manifest_sha256"] != _OPERATION_MANIFEST_SHA256
        ):
            raise ExternalCompositionError("composition is outside the pinned source closure")
        # Keep the original file hash separate from the canonical projection
        # embedded here. The original table's member order is not canonical.
        if canonical_sha256(value["source_table"]) != _SOURCE_TABLE_CONTENT_SHA256:
            raise ExternalCompositionError("source table content mismatch")
        source_values = _source_values(value["source_table"])
        if value["source_values_hex"] != source_values:
            raise ExternalCompositionError("source binary64 values disagree with attribution")
        source = _fields(
            value["operation_source"],
            {
                "tool",
                "aiconfigurator_version",
                "aiconfigurator_core_version",
                "system",
                "backend",
                "database_version",
                "data_slice_sha256",
                "database_mode",
                "shared_layer",
                "estimator_surface",
            },
            "operation_source",
        )
        for name, item in source.items():
            if name == "shared_layer":
                if type(item) is not bool:
                    raise ExternalCompositionError("shared_layer must be boolean")
            else:
                _text(item, f"operation_source.{name}")
        validate_sha256(source["data_slice_sha256"], "data_slice_sha256")
        for package, field in (
            ("aiconfigurator", "aiconfigurator_version"),
            ("aiconfigurator-core", "aiconfigurator_core_version"),
        ):
            if value["source_table"]["external_packages"][package] != source[field]:
                raise ExternalCompositionError("source package identity mismatch")
        objects = _fields(
            value["runtime_objects_sha256"],
            {"system", "model", "family_mapping"},
            "runtime_objects_sha256",
        )
        for name, digest in objects.items():
            validate_sha256(digest, f"runtime_objects_sha256.{name}")
        overrides = value["overrides_hex"]
        if not isinstance(overrides, dict) or not set(overrides) <= set(source_values):
            raise ExternalCompositionError("overrides contain unknown parameters")
        for name, spelling in overrides.items():
            _number(spelling, name)
            if spelling == source_values[name]:
                raise ExternalCompositionError("an intervention must change its source value")
        if overrides:
            validate_sha256(value["parent_record_sha256"], "parent_record_sha256")
            _text(value["intervention_reason"], "intervention_reason")
            baseline = {
                **value,
                "overrides_hex": {},
                "parent_record_sha256": None,
                "intervention_reason": None,
            }
            if RecordObject.from_value(baseline).record_id != value["parent_record_sha256"]:
                raise ExternalCompositionError(
                    "intervention parent is not its source-exact baseline"
                )
        elif value["parent_record_sha256"] is not None or value["intervention_reason"] is not None:
            raise ExternalCompositionError("baseline has no intervention metadata")
        object.__setattr__(self, "record", verified)

    @classmethod
    def promote(
        cls, database: ExternalOperationDatabase, source_table: Mapping[str, Any]
    ) -> ExternalServingComposition:
        """Build a source-exact record from an admitted database and attribution."""

        table = canonical_loads(canonical_bytes(source_table))
        value = {
            "schema": EXTERNAL_COMPOSITION_SCHEMA,
            "scope": "qwen3-32b-fp8-h200-trtllm-serving",
            "source_table": table,
            "source_table_sha256": _SOURCE_TABLE_SHA256,
            "source_values_hex": _source_values(table),
            "operation_source": database.source.as_dict(),
            "operation_manifest_sha256": _digest(database.manifest),
            "runtime_objects_sha256": _runtime_objects(database),
            "overrides_hex": {},
            "parent_record_sha256": None,
            "intervention_reason": None,
        }
        result = cls(RecordObject.from_value(value))
        result.validate_database(database)
        return result

    @classmethod
    def load(cls, root: str | Path, record_sha256: str) -> ExternalServingComposition:
        """Read one explicitly selected canonical object by content identity."""

        return cls(
            ObjectStore(root).read(record_sha256, expected_schema=EXTERNAL_COMPOSITION_SCHEMA)
        )

    @property
    def record_sha256(self) -> str:
        return self.record.record_id

    def number(self, adjustment_id: str) -> float:
        if adjustment_id not in EXTERNAL_ADJUSTMENT_IDS:
            raise ExternalCompositionError(f"unknown adjustment {adjustment_id!r}")
        value = self.record.value
        return _number(
            value["overrides_hex"].get(adjustment_id, value["source_values_hex"][adjustment_id]),
            adjustment_id,
        )

    def resolve(self, adjustment_id: str, explicit: object = COMPOSITION_UNSET) -> float:
        """Select the authoritative value and reject a conflicting explicit one."""

        selected = self.number(adjustment_id)
        if explicit is not COMPOSITION_UNSET and (
            type(explicit) is not float or explicit.hex() != selected.hex()
        ):
            raise ExternalCompositionError(
                f"{adjustment_id}: explicit value conflicts with selected record"
            )
        return selected

    def derive(self, overrides: Mapping[str, float], *, reason: str) -> ExternalServingComposition:
        """Create an immutable intervention, retaining the source-exact parent."""

        _text(reason, "reason")
        value = canonical_loads(self.record.canonical)
        baseline = {
            **value,
            "overrides_hex": {},
            "parent_record_sha256": None,
            "intervention_reason": None,
        }
        effective = dict(value["overrides_hex"])
        for name, number in overrides.items():
            if name not in EXTERNAL_ADJUSTMENT_IDS or type(number) is not float:
                raise ExternalCompositionError(
                    "interventions require known names and binary64 floats"
                )
            _number(number.hex(), name)
            effective[name] = number.hex()
        effective = {
            key: item for key, item in effective.items() if item != value["source_values_hex"][key]
        }
        if not effective:
            return type(self)(RecordObject.from_value(baseline))
        return type(self)(
            RecordObject.from_value(
                {
                    **baseline,
                    "overrides_hex": effective,
                    "parent_record_sha256": RecordObject.from_value(baseline).record_id,
                    "intervention_reason": reason,
                }
            )
        )

    def validate_database(self, database: ExternalOperationDatabase) -> None:
        """Reject source changes before any record-enabled model lookup."""

        from .external_db import ExternalOperationDatabase, ExternalSourceIdentity, _json_bytes

        if not isinstance(database, ExternalOperationDatabase):
            raise TypeError("database must be ExternalOperationDatabase")
        if database.source != ExternalSourceIdentity.from_manifest(database.manifest):
            raise ExternalCompositionError(
                "runtime source identity differs from its imported manifest"
            )
        value = self.record.value
        if (
            database.source.as_dict() != value["operation_source"]
            or _digest(database.manifest) != value["operation_manifest_sha256"]
            or _runtime_objects(database) != value["runtime_objects_sha256"]
        ):
            raise ExternalCompositionError("operation source or runtime state changed")
        # The public dictionaries must still describe the imported files. The
        # family mapping uses its own original formatting, so compare its value.
        for filename, current in (
            ("system.json", database.system_spec),
            ("model-config.json", database.model_config),
            ("family-mapping.json", database.family_mapping),
        ):
            raw = (database.artifact_dir / filename).read_bytes()
            expected = database.manifest["converted_files_sha256"][filename]
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ExternalCompositionError(f"imported {filename} hash mismatch")
            if _json_bytes(json.loads(raw)) != _json_bytes(current):
                raise ExternalCompositionError(f"runtime {filename} differs from imported source")
        gpu = database.system_spec["gpu"]
        for name, field in (
            ("memory_bandwidth_empirical_scale", "mem_bw_empirical_scaling_factor"),
            ("memory_empirical_constant_latency", "mem_empirical_constant_latency"),
        ):
            if float(gpu[field]).hex() != value["source_values_hex"][name]:
                raise ExternalCompositionError(f"source memory value mismatch for {name}")


__all__ = [
    "EXTERNAL_ADJUSTMENT_IDS",
    "EXTERNAL_COMPOSITION_SCHEMA",
    "EXTERNAL_OPERATION_ADJUSTMENTS",
    "ExternalCompositionError",
    "ExternalServingComposition",
]
