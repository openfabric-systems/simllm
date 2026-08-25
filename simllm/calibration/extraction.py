"""Framework-neutral orchestration for total offline model extraction."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from simllm.backends.step_lowerer import SerialStepLowerer, SerialStepLowererConfig
from simllm.compute import ModelDims, step_kernel, step_kernels
from simllm.compute.device_model import ShapeAxis, ShapeSchema, ShapeVector
from simllm.core import (
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    step_record_to_json,
    step_records_from_jsonl,
    write_step_records,
)

from .canonical import canonical_sha256, sha256_bytes, strict_json_loads
from .graph_identity import (
    execution_graph_template_record,
    unbound_execution_graph_record,
)
from .model_inventory import (
    AbsentPhysicalIdentity,
    FrameworkIdentity,
    ImplementationIdentityEnvelope,
    InventoryCase,
    InventorySuiteIdentity,
    KernelFamilyDefinition,
    KernelProjection,
    ModelCheckpointIdentity,
    ModelGeometry,
    ModelKernelInventory,
    PhaseLaunchCount,
)

SUPPORTED_SUITE_SCHEMA = "simllm-transformer-dag-suite-v1"
ORDERED_FAMILIES = (
    "attn_gemm",
    "attn_score",
    "mlp_gemm",
    "lm_head",
    "kv_read",
)
LAYER_REPEATED_FAMILIES = frozenset(
    {"attn_gemm", "attn_score", "mlp_gemm", "kv_read"}
)
_SHAPE_SCHEMA_IDS = {
    family: f"simllm-{family.replace('_', '-')}-invocation-shape-v1"
    for family in ORDERED_FAMILIES
}
_SHAPE_AXES = {
    "attn_gemm": ("new_tokens",),
    "attn_score": ("new_tokens", "kv_tokens"),
    "mlp_gemm": ("new_tokens",),
    "lm_head": ("sampled",),
    "kv_read": ("kv_tokens",),
}


class ModelExtractionError(ValueError):
    """A requested inventory is unsupported, inconsistent or incomplete."""


@dataclass(frozen=True, slots=True)
class FrameworkTextStack:
    """Normalized text-stack structure read from one framework config."""

    architecture: str
    wrapper_model_type: str
    text_model_type: str
    scope: str
    geometry: ModelGeometry
    layer_types: tuple[str, ...]
    linear_attention_mechanism: str
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    attn_output_gate: bool
    output_gate_type: str
    state_dtype: str
    excluded_components: tuple[str, ...]

    def __post_init__(self) -> None:
        strings = (
            self.architecture,
            self.wrapper_model_type,
            self.text_model_type,
            self.scope,
            self.linear_attention_mechanism,
            self.output_gate_type,
            self.state_dtype,
        )
        if any(not isinstance(value, str) or not value for value in strings):
            raise ValueError("framework text-stack strings must be nonblank")
        if not isinstance(self.geometry, ModelGeometry):
            raise TypeError("framework text-stack geometry must be ModelGeometry")
        if len(self.layer_types) != self.geometry.layers:
            raise ValueError("framework layer schedule length does not match geometry")
        if any(
            value not in {"linear_attention", "full_attention"}
            for value in self.layer_types
        ):
            raise ValueError("framework layer schedule has an unknown attention type")
        linear_geometry = (
            self.linear_conv_kernel_dim,
            self.linear_key_head_dim,
            self.linear_value_head_dim,
            self.linear_num_key_heads,
            self.linear_num_value_heads,
        )
        if any(type(value) is not int or value <= 0 for value in linear_geometry):
            raise ValueError("framework linear-attention geometry must be positive")
        if type(self.attn_output_gate) is not bool:
            raise TypeError("framework attention output-gate flag must be boolean")
        if not self.excluded_components or any(
            not isinstance(value, str) or not value for value in self.excluded_components
        ):
            raise ValueError("framework excluded components must be nonblank")

    def to_obj(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "wrapper_model_type": self.wrapper_model_type,
            "text_model_type": self.text_model_type,
            "scope": self.scope,
            "geometry": self.geometry.to_obj(),
            "layer_types": list(self.layer_types),
            "linear_attention_mechanism": self.linear_attention_mechanism,
            "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_value_head_dim": self.linear_value_head_dim,
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_num_value_heads": self.linear_num_value_heads,
            "attn_output_gate": self.attn_output_gate,
            "output_gate_type": self.output_gate_type,
            "state_dtype": self.state_dtype,
            "excluded_components": list(self.excluded_components),
        }


@dataclass(frozen=True, slots=True)
class FrameworkConfigurationProjection:
    """Framework identity, native binding, and normalized text structure."""

    framework: FrameworkIdentity
    configuration_seam: str
    architecture_binding: str
    text_implementation: str
    text_stack: FrameworkTextStack

    def __post_init__(self) -> None:
        if not isinstance(self.framework, FrameworkIdentity):
            raise TypeError("configuration projection framework must be typed")
        for value in (
            self.configuration_seam,
            self.architecture_binding,
            self.text_implementation,
        ):
            if not isinstance(value, str) or not value:
                raise ValueError("configuration projection strings must be nonblank")
        if not isinstance(self.text_stack, FrameworkTextStack):
            raise TypeError("configuration projection text stack must be typed")

    def to_obj(self) -> dict[str, Any]:
        framework = self.framework.to_obj()
        framework.pop("entry_seam")
        return {
            "schema": "simllm-framework-text-config-projection-v1",
            "framework": framework,
            "configuration_seam": self.configuration_seam,
            "architecture_binding": self.architecture_binding,
            "text_implementation": self.text_implementation,
            "text_stack": self.text_stack.to_obj(),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _model_identity(value: object) -> ModelCheckpointIdentity:
    if not isinstance(value, dict):
        raise ModelExtractionError("suite.reference_model: expected an object")
    core_fields = {
        "name",
        "revision",
        "config_sha256",
        "weight_sha256",
        "weight_bytes",
        "dtype",
        "quantization",
        "geometry",
    }
    metadata_fields = {
        "parameter_count",
        "weight_identity_source",
        "weight_identity_revision_url",
        "weight_identity_tree_url",
        "weight_manifest_rule",
        "local_weight_byte_verification",
        "local_weight_verification_policy",
        "weight_shards",
        "architecture",
        "model_type",
        "text_stack",
    }
    actual_fields = frozenset(value)
    if actual_fields not in {
        frozenset(core_fields),
        frozenset(core_fields | metadata_fields),
    }:
        raise ModelExtractionError(
            "suite.reference_model: expected exact local or API-metadata identity fields"
        )
    if metadata_fields.issubset(value):
        _validate_metadata_weight_identity(value)
    geometry = value["geometry"]
    if not isinstance(geometry, dict):
        raise ModelExtractionError("suite.reference_model.geometry: expected an object")
    return ModelCheckpointIdentity(
        name=value["name"],
        revision=value["revision"],
        config_sha256=value["config_sha256"],
        weight_sha256=value["weight_sha256"],
        weight_bytes=value["weight_bytes"],
        dtype=value["dtype"],
        quantization=value["quantization"],
        geometry=ModelGeometry.from_obj(geometry, "suite.reference_model.geometry"),
    )


def _validate_metadata_weight_identity(value: Mapping[str, Any]) -> None:
    if value["weight_identity_source"] != "hugging-face-api-metadata":
        raise ModelExtractionError("unsupported model weight identity source")
    for field in ("weight_identity_revision_url", "weight_identity_tree_url"):
        item = value[field]
        if not isinstance(item, str) or not item.startswith("https://huggingface.co/api/"):
            raise ModelExtractionError(f"suite.reference_model.{field}: invalid API URL")
    if value["weight_manifest_rule"] != (
        "sha256-of-canonical-json-array-of-name-sha256-bytes-records-sorted-by-name"
    ):
        raise ModelExtractionError("unsupported weight manifest rule")
    if value["local_weight_byte_verification"] is not False:
        raise ModelExtractionError("API-metadata identity must disable local weight checks")
    if value["local_weight_verification_policy"] != "intentionally-not-performed":
        raise ModelExtractionError("API-metadata identity must disclose local policy")
    parameter_count = value["parameter_count"]
    if type(parameter_count) is not int or parameter_count <= 0:
        raise ModelExtractionError("API parameter count must be a positive integer")
    shards = value["weight_shards"]
    if not isinstance(shards, list) or not shards:
        raise ModelExtractionError("API weight shard manifest must be nonempty")
    names: list[str] = []
    total_bytes = 0
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict) or set(shard) != {"name", "sha256", "bytes"}:
            raise ModelExtractionError(f"weight shard {index} has invalid fields")
        name = shard["name"]
        digest = shard["sha256"]
        size = shard["bytes"]
        if not isinstance(name, str) or not re.fullmatch(
            r"model-\d{5}-of-\d{5}[.]safetensors", name
        ):
            raise ModelExtractionError(f"weight shard {index} has invalid name")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ModelExtractionError(f"weight shard {index} has invalid digest")
        if type(size) is not int or size <= 0:
            raise ModelExtractionError(f"weight shard {index} has invalid byte count")
        names.append(name)
        total_bytes += size
    if names != sorted(names) or len(names) != len(set(names)):
        raise ModelExtractionError("API weight shards must be uniquely sorted by name")
    if total_bytes != value["weight_bytes"]:
        raise ModelExtractionError("API weight shard byte total does not match the suite")
    if canonical_sha256(shards) != value["weight_sha256"]:
        raise ModelExtractionError("API weight shard manifest hash does not match the suite")
    if 2 * parameter_count > total_bytes:
        raise ModelExtractionError("API BF16 parameter payload exceeds physical shard bytes")


def load_extraction_suite(raw: bytes) -> tuple[dict[str, Any], ModelCheckpointIdentity]:
    """Load one authored suite without accepting a partial graph grid."""

    try:
        value = strict_json_loads(raw)
    except ValueError as error:
        raise ModelExtractionError(f"invalid extraction suite: {error}") from error
    if not isinstance(value, dict):
        raise ModelExtractionError("extraction suite must be a JSON object")
    if value.get("schema") != SUPPORTED_SUITE_SCHEMA:
        raise ModelExtractionError(
            f"unsupported extraction suite schema {value.get('schema')!r}"
        )
    if value.get("state") != "authored-inputs-only":
        raise ModelExtractionError("extraction suite must remain authored-inputs-only")
    if not isinstance(value.get("suite"), str) or not value["suite"]:
        raise ModelExtractionError("suite.suite must be a nonblank string")
    cells = value.get("graph_cells")
    if not isinstance(cells, list) or not cells:
        raise ModelExtractionError("suite.graph_cells must be a nonempty array")
    case_ids = [cell.get("id") if isinstance(cell, dict) else None for cell in cells]
    if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        raise ModelExtractionError("every suite graph cell needs a nonblank ID")
    if len(case_ids) != len(set(case_ids)):
        raise ModelExtractionError("suite graph cell IDs must be unique")
    return value, _model_identity(value.get("reference_model"))


def validate_checkpoint(
    checkpoint_root: Path,
    identity: ModelCheckpointIdentity,
    *,
    metadata_only: bool = False,
) -> None:
    """Verify a local config substrate and the declared weight policy."""

    if not checkpoint_root.is_dir():
        raise ModelExtractionError(f"checkpoint root is not a directory: {checkpoint_root}")
    if checkpoint_root.name != identity.revision:
        raise ModelExtractionError(
            f"checkpoint directory names revision {checkpoint_root.name!r}, expected "
            f"{identity.revision!r}"
        )
    config = checkpoint_root / "config.json"
    if not config.is_file():
        raise ModelExtractionError("checkpoint has no config.json")
    if _sha256_file(config) != identity.config_sha256:
        raise ModelExtractionError("checkpoint config hash does not match the suite")
    weights = tuple(sorted(checkpoint_root.glob("*.safetensors")))
    if metadata_only:
        if weights:
            raise ModelExtractionError(
                "API-metadata extraction requires a weight-free local substrate"
            )
        return
    if len(weights) != 1 or not weights[0].is_file():
        raise ModelExtractionError(
            f"checkpoint requires exactly one safetensors weight object, found {len(weights)}"
        )
    if weights[0].stat().st_size != identity.weight_bytes:
        raise ModelExtractionError("checkpoint weight byte count does not match the suite")
    if _sha256_file(weights[0]) != identity.weight_sha256:
        raise ModelExtractionError("checkpoint weight hash does not match the suite")


def _expected_dims(identity: ModelCheckpointIdentity) -> ModelDims:
    geometry = identity.geometry
    if identity.dtype != "bfloat16" or identity.quantization != "none":
        raise ModelExtractionError(
            "the first extraction slice supports only unquantized bfloat16 checkpoints"
        )
    return ModelDims(
        num_layers=geometry.layers,
        hidden_size=geometry.hidden_size,
        intermediate_size=geometry.intermediate_size,
        num_heads=geometry.num_heads,
        num_kv_heads=geometry.num_kv_heads,
        head_size=geometry.head_size,
        vocab_size=geometry.vocab_size,
        dtype_bytes=2,
        weight_dtype_bytes=2,
        kv_dtype_bytes=2,
        num_experts=geometry.num_experts,
        top_k=geometry.top_k,
        moe_intermediate_size=(
            geometry.intermediate_size if geometry.num_experts else None
        ),
        local_num_experts=geometry.num_experts,
    )


def validate_framework_dims(
    dims: ModelDims,
    identity: ModelCheckpointIdentity,
) -> None:
    """Reject any framework projection that defaulted or changed geometry."""

    if dims.defaulted_fields:
        raise ModelExtractionError(
            "framework geometry used defaults: " + ", ".join(dims.defaulted_fields)
        )
    expected = _expected_dims(identity)
    fields = (
        "num_layers",
        "hidden_size",
        "intermediate_size",
        "num_heads",
        "num_kv_heads",
        "head_size",
        "vocab_size",
        "num_experts",
        "top_k",
        "moe_intermediate_size",
        "local_num_experts",
    )
    mismatches = [
        f"{name}={getattr(dims, name)!r}, expected {getattr(expected, name)!r}"
        for name in fields
        if getattr(dims, name) != getattr(expected, name)
    ]
    if dims.dtype_bytes != 2 or dims.weight_element_bytes != 2 or dims.kv_element_bytes != 2:
        mismatches.append("dtype widths are not exact bfloat16")
    if mismatches:
        raise ModelExtractionError("framework geometry mismatch: " + "; ".join(mismatches))


def _case_record(cell: dict[str, Any], ordinal: int) -> StepRecord:
    required = {"id", "family", "phase", "split"}
    if not required.issubset(cell):
        raise ModelExtractionError(
            f"suite.graph_cells[{ordinal}] is missing a required identity field"
        )
    case_id = cell["id"]
    family = cell["family"]
    phase = cell["phase"]
    if family == "compute-prefill":
        expected = required | {
            "requests",
            "prompt_tokens_per_request",
            "total_prompt_tokens",
        }
        if set(cell) != expected or phase != "prefill":
            raise ModelExtractionError(f"case {case_id!r} has an invalid prefill shape")
        count = cell["requests"]
        length = cell["prompt_tokens_per_request"]
        if type(count) is not int or count <= 0 or type(length) is not int or length <= 0:
            raise ModelExtractionError(f"case {case_id!r} has nonpositive prefill axes")
        if count * length != cell["total_prompt_tokens"]:
            raise ModelExtractionError(f"case {case_id!r} prompt partition is not exact")
        request_phase = RequestPhase.PREFILL
        contexts = (length,) * count
        new_tokens = (length,) * count
    elif family in {
        "memory-decode",
        "moe-communication-decode",
        "dense-batch-decode",
    }:
        expected = required | {"batch", "context_tokens", "new_tokens_per_request"}
        if family == "moe-communication-decode":
            expected |= {"expert_participants", "parallelism_override"}
        if set(cell) != expected or phase != "decode":
            raise ModelExtractionError(f"case {case_id!r} has an invalid decode shape")
        count = cell["batch"]
        context = cell["context_tokens"]
        one_token = cell["new_tokens_per_request"]
        if (
            type(count) is not int
            or count <= 0
            or type(context) is not int
            or context <= 0
            or one_token != 1
        ):
            raise ModelExtractionError(f"case {case_id!r} has invalid decode axes")
        if family == "moe-communication-decode" and (
            cell["expert_participants"] != 4
            or cell["parallelism_override"]
            != {
                "expert": 4
            }
        ):
            raise ModelExtractionError(
                f"case {case_id!r} has unsupported expert parallelism"
            )
        request_phase = RequestPhase.DECODE
        contexts = (context,) * count
        new_tokens = (1,) * count
    else:
        raise ModelExtractionError(f"case {case_id!r} has unknown family {family!r}")
    scheduled = [
        ScheduledRequest(
            request_id=f"{case_id}:request:{request_ordinal}",
            phase=request_phase,
            num_new_tokens=new_tokens[request_ordinal],
            context_length=contexts[request_ordinal],
        )
        for request_ordinal in range(count)
    ]
    return StepRecord(
        step_index=ordinal,
        virtual_time_ps=0,
        scheduled=scheduled,
        num_sampled=count,
        num_tokens_after_padding=sum(new_tokens),
        sampled_request_ids=[request.request_id for request in scheduled],
    )


def _records_from_suite(
    suite: dict[str, Any],
    step_records_path: Path,
) -> tuple[StepRecord, ...]:
    records = case_records_from_suite(suite)
    write_step_records(records, step_records_path)
    loaded = tuple(step_records_from_jsonl(step_records_path))
    if [step_record_to_json(record) for record in loaded] != [
        step_record_to_json(record) for record in records
    ]:
        raise ModelExtractionError("StepRecord path did not preserve the complete case set")
    return loaded


def case_records_from_suite(suite: Mapping[str, Any]) -> tuple[StepRecord, ...]:
    """Validate authored shapes and build records without writing a stream."""

    cells = suite.get("graph_cells")
    if not isinstance(cells, list):
        raise ModelExtractionError("suite.graph_cells must be an array")
    return tuple(_case_record(cell, index) for index, cell in enumerate(cells))


def _exact_work(value: float, path: str) -> int:
    if not isinstance(value, float) or not value.is_integer() or value < 0:
        raise ModelExtractionError(f"{path}: expected exact nonnegative integer work")
    return int(value)


def _case_dims(dims: ModelDims, cell: dict[str, Any]) -> ModelDims:
    if cell["family"] != "moe-communication-decode":
        return dims
    participants = cell["expert_participants"]
    if dims.num_experts % participants:
        raise ModelExtractionError(
            f"case {cell['id']!r}: experts do not divide participant count"
        )
    return replace(dims, local_num_experts=dims.num_experts // participants)


def _shape_schemas(
    specs_by_case: tuple[tuple[Any, ...], ...],
) -> tuple[ShapeSchema, ...]:
    schemas = []
    for family_index, family in enumerate(ORDERED_FAMILIES):
        axis_names = _SHAPE_AXES[family]
        vectors = [dict(specs[family_index].config) for specs in specs_by_case]
        axes = tuple(
            ShapeAxis(
                axis_id=axis,
                unit="tokens",
                minimum=min(vector[axis] for vector in vectors),
                maximum=max(vector[axis] for vector in vectors),
            )
            for axis in axis_names
        )
        schemas.append(
            ShapeSchema(shape_schema_id=_SHAPE_SCHEMA_IDS[family], axes=axes)
        )
    return tuple(schemas)


def _family_definitions(layers: int) -> tuple[KernelFamilyDefinition, ...]:
    return tuple(
        KernelFamilyDefinition(
            family_id=family,
            shape_schema_id=_SHAPE_SCHEMA_IDS[family],
            phase_launch_counts=tuple(
                PhaseLaunchCount(
                    phase=phase,
                    logical_launch_count=(
                        layers if family in LAYER_REPEATED_FAMILIES else 1
                    ),
                )
                for phase in ("prefill", "decode")
            ),
        )
        for family in ORDERED_FAMILIES
    )


def _projections(
    specs: tuple[Any, ...],
    definitions: tuple[KernelFamilyDefinition, ...],
    phase: str,
) -> tuple[KernelProjection, ...]:
    result = []
    for spec, definition in zip(specs, definitions, strict=True):
        if spec.name != definition.family_id:
            raise ModelExtractionError(
                f"kernel family order changed: found {spec.name!r}, expected "
                f"{definition.family_id!r}"
            )
        config = dict(spec.config)
        axes = _SHAPE_AXES[spec.name]
        if tuple(config) != axes:
            raise ModelExtractionError(
                f"kernel family {spec.name!r} shape axes changed from {axes!r}"
            )
        values = tuple(config[axis] for axis in axes)
        if any(type(value) is not int or value < 0 for value in values):
            raise ModelExtractionError(
                f"kernel family {spec.name!r} has a noninteger shape vector"
            )
        result.append(
            KernelProjection(
                family_id=spec.name,
                shape_vector=ShapeVector(
                    shape_schema_id=definition.shape_schema_id,
                    values=values,
                ),
                logical_launch_count=definition.count_for(phase),
                aggregate_flops=_exact_work(spec.flops, f"{spec.name}.flops"),
                aggregate_hbm_bytes=_exact_work(
                    spec.bytes_moved, f"{spec.name}.bytes_moved"
                ),
            )
        )
    return tuple(result)


def _validate_framework_declaration(
    suite: Mapping[str, Any],
    framework: FrameworkIdentity,
    projection: FrameworkConfigurationProjection | None,
) -> None:
    declared_frameworks = suite.get("frameworks")
    if not isinstance(declared_frameworks, list):
        raise ModelExtractionError("suite.frameworks must be an array")
    if any(not isinstance(item, dict) for item in declared_frameworks):
        raise ModelExtractionError("suite.frameworks entries must be objects")
    matching = [
        item for item in declared_frameworks if item.get("id") == framework.framework_id
    ]
    expected = framework.to_obj().copy()
    expected.pop("entry_seam")
    expected = {key: value for key, value in expected.items() if value is not None}
    if projection is not None:
        if projection.framework != framework:
            raise ModelExtractionError("framework projection identity changed")
        expected.update(
            {
                "architecture_binding": projection.architecture_binding,
                "text_implementation": projection.text_implementation,
            }
        )
    if matching != [expected]:
        raise ModelExtractionError("runtime framework identity does not match the suite")


def _validate_text_stack(
    reference_model: Mapping[str, Any],
    projection: FrameworkConfigurationProjection,
) -> None:
    text = reference_model.get("text_stack")
    if not isinstance(text, dict):
        raise ModelExtractionError("suite text-stack contract must be an object")
    expected_fields = {
        "scope",
        "model_type",
        "layer_pattern",
        "pattern_repetitions",
        "linear_attention_layers",
        "full_attention_layers",
        "linear_attention_mechanism",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "attn_output_gate",
        "output_gate_type",
        "state_dtype",
        "excluded_components",
    }
    if set(text) != expected_fields:
        raise ModelExtractionError("suite text-stack contract has unexpected fields")
    pattern = text["layer_pattern"]
    repetitions = text["pattern_repetitions"]
    if (
        not isinstance(pattern, list)
        or not pattern
        or type(repetitions) is not int
        or repetitions <= 0
    ):
        raise ModelExtractionError("suite text-stack layer pattern is invalid")
    schedule = tuple(pattern) * repetitions
    linear_layers = schedule.count("linear_attention")
    full_layers = schedule.count("full_attention")
    if (
        len(schedule) != projection.text_stack.geometry.layers
        or linear_layers != text["linear_attention_layers"]
        or full_layers != text["full_attention_layers"]
        or linear_layers + full_layers != len(schedule)
    ):
        raise ModelExtractionError("suite text-stack layer counts are inconsistent")
    expected = {
        "architecture": reference_model.get("architecture"),
        "wrapper_model_type": reference_model.get("model_type"),
        "text_model_type": text["model_type"],
        "scope": text["scope"],
        "geometry": reference_model["geometry"],
        "layer_types": list(schedule),
        "linear_attention_mechanism": text["linear_attention_mechanism"],
        "linear_conv_kernel_dim": text["linear_conv_kernel_dim"],
        "linear_key_head_dim": text["linear_key_head_dim"],
        "linear_value_head_dim": text["linear_value_head_dim"],
        "linear_num_key_heads": text["linear_num_key_heads"],
        "linear_num_value_heads": text["linear_num_value_heads"],
        "attn_output_gate": text["attn_output_gate"],
        "output_gate_type": text["output_gate_type"],
        "state_dtype": text["state_dtype"],
        "excluded_components": text["excluded_components"],
    }
    if projection.text_stack.to_obj() != expected:
        raise ModelExtractionError("framework text-stack projection does not match the suite")
    if linear_layers:
        raise ModelExtractionError(
            "COMP-62: Qwen3.5 Gated DeltaNet linear attention is outside the "
            "frozen inventory family set; refusing partial "
            f"{projection.framework.framework_id} inventory"
        )


def extract_model_inventory(
    *,
    suite_raw: bytes,
    framework: FrameworkIdentity,
    checkpoint_root: Path,
    framework_dims: ModelDims,
    step_records_path: Path,
    framework_projection: FrameworkConfigurationProjection | None = None,
) -> ModelKernelInventory:
    """Build one total inventory after every identity and projection check."""

    suite, model = load_extraction_suite(suite_raw)
    _validate_framework_declaration(suite, framework, framework_projection)
    reference_model = suite["reference_model"]
    metadata_only = (
        reference_model.get("weight_identity_source")
        == "hugging-face-api-metadata"
    )
    validate_checkpoint(checkpoint_root, model, metadata_only=metadata_only)
    validate_framework_dims(framework_dims, model)
    case_records_from_suite(suite)
    has_text_contract = "text_stack" in reference_model
    if has_text_contract and framework_projection is None:
        raise ModelExtractionError("suite requires a framework text-stack projection")
    if framework_projection is not None:
        if not has_text_contract:
            raise ModelExtractionError("framework supplied an undeclared text stack")
        _validate_text_stack(reference_model, framework_projection)
    records = _records_from_suite(suite, step_records_path)
    cells = suite["graph_cells"]
    case_dims = tuple(
        _case_dims(framework_dims, cell) for cell in cells
    )
    specs_by_case = tuple(
        tuple(step_kernels(dims, record, record.num_sampled or 0))
        for dims, record in zip(case_dims, records, strict=True)
    )
    if any(tuple(spec.name for spec in specs) != ORDERED_FAMILIES for specs in specs_by_case):
        raise ModelExtractionError("step_kernels() did not produce the frozen family order")
    schemas = _shape_schemas(specs_by_case)
    definitions = _family_definitions(model.geometry.layers)
    cases = []
    for ordinal, (cell, record, dims, specs) in enumerate(
        zip(cells, records, case_dims, specs_by_case, strict=True)
    ):
        fused = step_kernel(dims, record, record.num_sampled or 0)
        projected_flops = sum(_exact_work(spec.flops, "family.flops") for spec in specs)
        projected_bytes = sum(
            _exact_work(spec.bytes_moved, "family.bytes_moved") for spec in specs
        )
        if projected_flops != _exact_work(fused.flops, "fused.flops"):
            raise ModelExtractionError(f"case {cell['id']!r} family FLOPs are not exact")
        if projected_bytes != _exact_work(fused.bytes_moved, "fused.bytes_moved"):
            raise ModelExtractionError(f"case {cell['id']!r} family bytes are not exact")
        ep_ranks = (
            tuple(range(cell["expert_participants"]))
            if cell["family"] == "moe-communication-decode"
            else None
        )
        graph = SerialStepLowerer(
            SerialStepLowererConfig(
                dims=dims,
                tp_ranks=(0,),
                ep_ranks=ep_ranks,
            )
        ).lower(record)
        instance = unbound_execution_graph_record(graph)
        template = execution_graph_template_record(graph)
        projections = _projections(specs, definitions, cell["phase"])
        cases.append(
            InventoryCase(
                case_id=cell["id"],
                family=cell["family"],
                phase=cell["phase"],
                split=cell["split"],
                suite_case_sha256=canonical_sha256(cell),
                step_record_sha256=canonical_sha256(step_record_to_json(record)),
                instance_graph_sha256=instance.record_id,
                template_graph_sha256=template.record_id,
                kernel_projections=projections,
            )
        )
        if record.step_index != ordinal:
            raise ModelExtractionError("StepRecord order changed during extraction")
    join_task = "VLLM-12" if framework.framework_id == "vllm" else "SGL-10"
    inventory = ModelKernelInventory(
        suite=InventorySuiteIdentity(
            suite_id=suite["suite"],
            suite_sha256=sha256_bytes(suite_raw),
            case_count=len(cells),
        ),
        framework=framework,
        model=model,
        shape_schemas=schemas,
        kernel_families=definitions,
        cases=tuple(cases),
        implementation_identity=ImplementationIdentityEnvelope(
            code_object_hashes=AbsentPhysicalIdentity(),
            observed_launches=AbsentPhysicalIdentity(),
            join_tasks=tuple(sorted(("COMP-6", join_task))),
        ),
    )
    ModelKernelInventory.from_obj(inventory.to_obj())
    return inventory


__all__ = [
    "LAYER_REPEATED_FAMILIES",
    "ORDERED_FAMILIES",
    "FrameworkConfigurationProjection",
    "FrameworkTextStack",
    "ModelExtractionError",
    "case_records_from_suite",
    "extract_model_inventory",
    "load_extraction_suite",
    "validate_checkpoint",
    "validate_framework_dims",
]
