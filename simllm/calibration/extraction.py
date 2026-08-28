"""Framework-neutral orchestration for total offline model extraction."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from simllm.backends.step_lowerer import SerialStepLowerer, SerialStepLowererConfig
from simllm.compute import ModelDims, step_kernel, step_kernels, step_shape
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
QWEN_GATED_DELTA_NET_FAMILIES = (
    "attn_gemm",
    "attn_score",
    "kv_read",
    "gdn_input_projection",
    "gdn_short_convolution",
    "gdn_state_read",
    "gdn_state_update",
    "gdn_state_write",
    "gdn_gated_norm",
    "gdn_output_projection",
    "mlp_gemm",
    "lm_head",
)
DEEPSEEK_V3_FAMILIES = (
    "mla_q_compression",
    "mla_q_decompression",
    "mla_kv_compression",
    "mla_kv_decompression",
    "mla_rotary_split",
    "mla_attention",
    "mla_compressed_kv_read",
    "mla_output_projection",
    "dense_early_mlp",
    "moe_router",
    "moe_shared_expert",
    "moe_routed_experts",
    "lm_head",
    "multi_token_prediction_head",
)
LAYER_REPEATED_FAMILIES = frozenset(
    {"attn_gemm", "attn_score", "mlp_gemm", "kv_read"}
)
_SHAPE_SCHEMA_IDS = {
    family: f"simllm-{family.replace('_', '-')}-invocation-shape-v1"
    for family in dict.fromkeys(
        ORDERED_FAMILIES + QWEN_GATED_DELTA_NET_FAMILIES + DEEPSEEK_V3_FAMILIES
    )
}
_SHAPE_AXES = {
    "attn_gemm": ("new_tokens",),
    "attn_score": ("new_tokens", "kv_tokens"),
    "mlp_gemm": ("new_tokens",),
    "lm_head": ("sampled",),
    "kv_read": ("kv_tokens",),
    "gdn_input_projection": ("new_tokens",),
    "gdn_short_convolution": ("new_tokens", "sequences"),
    "gdn_state_read": ("sequences",),
    "gdn_state_update": ("new_tokens",),
    "gdn_state_write": ("sequences",),
    "gdn_gated_norm": ("new_tokens",),
    "gdn_output_projection": ("new_tokens",),
    "mla_q_compression": ("new_tokens",),
    "mla_q_decompression": ("new_tokens",),
    "mla_kv_compression": ("new_tokens",),
    "mla_kv_decompression": ("new_tokens",),
    "mla_rotary_split": ("new_tokens",),
    "mla_attention": ("new_tokens", "kv_tokens"),
    "mla_compressed_kv_read": ("kv_tokens",),
    "mla_output_projection": ("new_tokens",),
    "dense_early_mlp": ("new_tokens",),
    "moe_router": ("new_tokens",),
    "moe_shared_expert": ("new_tokens",),
    "moe_routed_experts": ("new_tokens",),
    "multi_token_prediction_head": (
        "new_tokens",
        "kv_tokens",
        "sampled",
        "mtp_enabled",
    ),
}


class ModelExtractionError(ValueError):
    """A requested inventory is unsupported, inconsistent or incomplete."""


@dataclass(frozen=True, slots=True)
class _ExactFamilyWork:
    """One integer-only offline family projection before record validation."""

    name: str
    flops: int
    bytes_moved: int
    config: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ModelExtractionError("inventory family name must be nonblank")
        for label, value in (("flops", self.flops), ("bytes", self.bytes_moved)):
            if type(value) is not int or value < 0:
                raise ModelExtractionError(
                    f"inventory family {self.name!r} {label} must be a nonnegative integer"
                )
        if any(
            not axis or type(value) is not int or value < 0
            for axis, value in self.config
        ):
            raise ModelExtractionError(
                f"inventory family {self.name!r} shape must be nonnegative integers"
            )


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
class FrameworkDenseStack:
    """Normalized dense text stack read from one framework config."""

    architecture: str
    model_type: str
    scope: str
    geometry: ModelGeometry
    attention_mechanism: str
    quantization: str
    weight_block_size: tuple[int, int]
    excluded_components: tuple[str, ...]

    def __post_init__(self) -> None:
        strings = (
            self.architecture,
            self.model_type,
            self.scope,
            self.attention_mechanism,
            self.quantization,
        )
        if any(not isinstance(value, str) or not value for value in strings):
            raise ValueError("framework dense-stack strings must be nonblank")
        if not isinstance(self.geometry, ModelGeometry):
            raise TypeError("framework dense-stack geometry must be ModelGeometry")
        if (
            len(self.weight_block_size) != 2
            or any(type(value) is not int or value <= 0 for value in self.weight_block_size)
        ):
            raise ValueError("framework dense-stack weight block must be two positive integers")
        if not self.excluded_components or any(
            not isinstance(value, str) or not value for value in self.excluded_components
        ):
            raise ValueError("framework dense-stack exclusions must be nonblank")

    def to_obj(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "model_type": self.model_type,
            "scope": self.scope,
            "geometry": self.geometry.to_obj(),
            "attention_mechanism": self.attention_mechanism,
            "quantization": self.quantization,
            "weight_block_size": list(self.weight_block_size),
            "excluded_components": list(self.excluded_components),
        }


@dataclass(frozen=True, slots=True)
class FrameworkDeepseekStack:
    """Normalized DeepSeek-V3 structure read from one framework config."""

    architecture: str
    wrapper_model_type: str
    scope: str
    geometry: ModelGeometry
    layer_types: tuple[str, ...]
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    first_k_dense_replace: int
    moe_intermediate_size: int
    moe_layer_freq: int
    n_shared_experts: int
    n_group: int
    topk_group: int
    scoring_func: str
    topk_method: str
    norm_topk_prob: bool
    routed_scaling_factor: str
    num_nextn_predict_layers: int
    weight_block_size: tuple[int, int]
    excluded_components: tuple[str, ...]

    def __post_init__(self) -> None:
        strings = (
            self.architecture,
            self.wrapper_model_type,
            self.scope,
            self.scoring_func,
            self.topk_method,
        )
        if any(not isinstance(value, str) or not value for value in strings):
            raise ValueError("framework DeepSeek stack strings must be nonblank")
        if not isinstance(self.geometry, ModelGeometry):
            raise TypeError("framework DeepSeek stack geometry must be ModelGeometry")
        if len(self.layer_types) != self.geometry.layers:
            raise ValueError("framework DeepSeek layer schedule length changed")
        if any(value not in {"dense", "moe"} for value in self.layer_types):
            raise ValueError("framework DeepSeek layer schedule has an unknown type")
        integer_fields = (
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.first_k_dense_replace,
            self.moe_intermediate_size,
            self.moe_layer_freq,
            self.n_shared_experts,
            self.n_group,
            self.topk_group,
            self.num_nextn_predict_layers,
            *self.weight_block_size,
        )
        if any(type(value) is not int or value <= 0 for value in integer_fields):
            raise ValueError("framework DeepSeek geometry must be positive integers")
        if type(self.norm_topk_prob) is not bool:
            raise TypeError("framework DeepSeek top-k normalization must be boolean")
        if self.routed_scaling_factor != "5/2":
            raise ValueError("framework DeepSeek routed scale must equal 5/2")
        if not self.excluded_components or any(
            not isinstance(value, str) or not value
            for value in self.excluded_components
        ):
            raise ValueError("framework DeepSeek exclusions must be nonblank")

    def to_obj(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "wrapper_model_type": self.wrapper_model_type,
            "scope": self.scope,
            "geometry": self.geometry.to_obj(),
            "layer_types": list(self.layer_types),
            "q_lora_rank": self.q_lora_rank,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            "first_k_dense_replace": self.first_k_dense_replace,
            "moe_intermediate_size": self.moe_intermediate_size,
            "moe_layer_freq": self.moe_layer_freq,
            "n_shared_experts": self.n_shared_experts,
            "n_group": self.n_group,
            "topk_group": self.topk_group,
            "scoring_func": self.scoring_func,
            "topk_method": self.topk_method,
            "norm_topk_prob": self.norm_topk_prob,
            "routed_scaling_factor": self.routed_scaling_factor,
            "num_nextn_predict_layers": self.num_nextn_predict_layers,
            "weight_block_size": list(self.weight_block_size),
            "excluded_components": list(self.excluded_components),
        }


@dataclass(frozen=True, slots=True)
class FrameworkConfigurationProjection:
    """Framework identity, native binding, and normalized text structure."""

    framework: FrameworkIdentity
    configuration_seam: str
    architecture_binding: str
    text_implementation: str
    dense_stack: FrameworkDenseStack | None = None
    text_stack: FrameworkTextStack | None = None
    deepseek_stack: FrameworkDeepseekStack | None = None

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
        stacks = (
            self.dense_stack is not None,
            self.text_stack is not None,
            self.deepseek_stack is not None,
        )
        if stacks.count(True) != 1:
            raise TypeError("configuration projection requires exactly one typed stack")
        if self.text_stack is not None and not isinstance(
            self.text_stack, FrameworkTextStack
        ):
            raise TypeError("configuration projection text stack must be typed")
        if self.dense_stack is not None and not isinstance(
            self.dense_stack, FrameworkDenseStack
        ):
            raise TypeError("configuration projection dense stack must be typed")
        if self.deepseek_stack is not None and not isinstance(
            self.deepseek_stack, FrameworkDeepseekStack
        ):
            raise TypeError("configuration projection DeepSeek stack must be typed")

    def to_obj(self) -> dict[str, Any]:
        framework = self.framework.to_obj()
        framework.pop("entry_seam")
        value = {
            "schema": "simllm-framework-text-config-projection-v1",
            "framework": framework,
            "configuration_seam": self.configuration_seam,
            "architecture_binding": self.architecture_binding,
            "text_implementation": self.text_implementation,
        }
        if self.dense_stack is not None:
            value["dense_stack"] = self.dense_stack.to_obj()
        elif self.text_stack is not None:
            value["text_stack"] = self.text_stack.to_obj()
        else:
            assert self.deepseek_stack is not None
            value["deepseek_stack"] = self.deepseek_stack.to_obj()
        return value


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
    }
    structure_fields = ({"dense_stack"}, {"text_stack"}, {"deepseek_stack"})
    actual_fields = frozenset(value)
    accepted = {frozenset(core_fields)} | {
        frozenset(core_fields | metadata_fields | structure)
        for structure in structure_fields
    }
    if actual_fields not in accepted:
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
            r"model-\d{5}-of-\d{5,6}[.]safetensors", name
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
    quantization = value["quantization"]
    if quantization == "none" and 2 * parameter_count > total_bytes:
        raise ModelExtractionError("API BF16 parameter payload exceeds physical shard bytes")
    if quantization != "none" and parameter_count > total_bytes:
        raise ModelExtractionError("API quantized parameter floor exceeds shard bytes")


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


def _expected_dims(
    identity: ModelCheckpointIdentity,
    deepseek_stack: FrameworkDeepseekStack | None = None,
) -> ModelDims:
    geometry = identity.geometry
    if identity.dtype != "bfloat16":
        raise ModelExtractionError("extraction requires bfloat16 activation geometry")
    if identity.quantization == "none" and deepseek_stack is None:
        weight_dtype_bytes = 2
        moe_intermediate_size = (
            geometry.intermediate_size if geometry.num_experts else None
        )
    elif identity.quantization == "fp8-e4m3-block-128x128":
        weight_dtype_bytes = 1
        moe_intermediate_size = (
            deepseek_stack.moe_intermediate_size
            if deepseek_stack is not None
            else None
        )
    else:
        raise ModelExtractionError(
            "checkpoint quantization does not match its declared structure"
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
        weight_dtype_bytes=weight_dtype_bytes,
        kv_dtype_bytes=2,
        num_experts=geometry.num_experts,
        top_k=geometry.top_k,
        moe_intermediate_size=moe_intermediate_size,
        local_num_experts=geometry.num_experts,
    )


def validate_framework_dims(
    dims: ModelDims,
    identity: ModelCheckpointIdentity,
    deepseek_stack: FrameworkDeepseekStack | None = None,
) -> None:
    """Reject any framework projection that defaulted or changed geometry."""

    if dims.defaulted_fields:
        raise ModelExtractionError(
            "framework geometry used defaults: " + ", ".join(dims.defaulted_fields)
        )
    expected = _expected_dims(identity, deepseek_stack)
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
    if (
        dims.dtype_bytes != expected.dtype_bytes
        or dims.weight_element_bytes != expected.weight_element_bytes
        or dims.kv_element_bytes != expected.kv_element_bytes
    ):
        mismatches.append("dtype widths do not match the checkpoint contract")
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
        if "mtp_enabled" in cell:
            expected.add("mtp_enabled")
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
        if "mtp_enabled" in cell:
            expected.add("mtp_enabled")
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
    if "mtp_enabled" in cell and type(cell["mtp_enabled"]) is not bool:
        raise ModelExtractionError(f"case {case_id!r} has a nonboolean MTP flag")
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


def _exact_work(value: object, path: str) -> int:
    if type(value) is int and value >= 0:
        return value
    if not isinstance(value, float) or not value.is_integer() or value < 0:
        raise ModelExtractionError(f"{path}: expected exact nonnegative integer work")
    return int(value)


def _validate_qwen_gdn_text_stack(text_stack: FrameworkTextStack) -> None:
    expected_identity = {
        "architecture": "Qwen3_5ForConditionalGeneration",
        "wrapper_model_type": "qwen3_5",
        "text_model_type": "qwen3_5_text",
        "scope": "text-only",
        "linear_attention_mechanism": "Qwen3.5 Gated DeltaNet",
        "attn_output_gate": True,
        "output_gate_type": "swish",
        "state_dtype": "float32",
        "excluded_components": (
            "multimodal-vision-encoder",
            "one-layer-mtp-speculative-head",
        ),
    }
    mismatches = [
        name
        for name, expected in expected_identity.items()
        if getattr(text_stack, name) != expected
    ]
    linear_layers = text_stack.layer_types.count("linear_attention")
    full_layers = text_stack.layer_types.count("full_attention")
    if mismatches:
        raise ModelExtractionError(
            "unsupported Gated DeltaNet text-stack fields: " + ", ".join(mismatches)
        )
    if not linear_layers or not full_layers:
        raise ModelExtractionError(
            "Gated DeltaNet extraction requires linear and full attention layers"
        )
    if text_stack.geometry.num_experts or text_stack.geometry.top_k:
        raise ModelExtractionError("Gated DeltaNet extraction requires a dense MLP")


def _qwen_gdn_family_work(
    dims: ModelDims,
    record: StepRecord,
    num_sampled: int,
    text_stack: FrameworkTextStack,
) -> tuple[_ExactFamilyWork, ...]:
    """Project the pinned Qwen3.5 text stack with integer-only arithmetic."""

    _validate_qwen_gdn_text_stack(text_stack)
    new_tokens, kv_tokens, attention_pairs = step_shape(record)
    sequences = len(record.scheduled)
    linear_layers = text_stack.layer_types.count("linear_attention")
    full_layers = text_stack.layer_types.count("full_attention")
    hidden = dims.hidden_size
    key_heads = text_stack.linear_num_key_heads
    value_heads = text_stack.linear_num_value_heads
    key_head_dim = text_stack.linear_key_head_dim
    value_head_dim = text_stack.linear_value_head_dim
    key_width = key_heads * key_head_dim
    value_width = value_heads * value_head_dim
    conv_width = text_stack.linear_conv_kernel_dim
    conv_channels = 2 * key_width + value_width
    conv_state_elements = conv_channels * (conv_width - 1)
    recurrent_state_elements = value_heads * value_head_dim * key_head_dim
    full_attention_params = hidden * (
        dims.num_heads * dims.head_size
        + 2 * dims.num_kv_heads * dims.head_size
    ) + dims.num_heads * dims.head_size * hidden
    input_projection_params = hidden * (
        2 * key_width + 2 * value_width + 2 * value_heads
    )
    conv_params = conv_channels * conv_width
    update_flops_per_token = key_heads * (7 * key_head_dim + 2) + value_heads * (
        7 * value_head_dim * key_head_dim + 2 * value_head_dim + 7
    )
    norm_flops_per_token = value_heads * (7 * value_head_dim + 2)
    output_projection_params = value_width * hidden
    mlp_params = 3 * hidden * dims.intermediate_size
    head_params = hidden * dims.vocab_size
    weight_bytes = int(dims.weight_element_bytes)
    activation_bytes = dims.dtype_bytes
    state_bytes = 4

    return (
        _ExactFamilyWork(
            name="attn_gemm",
            flops=2 * new_tokens * full_layers * full_attention_params,
            bytes_moved=weight_bytes * full_layers * full_attention_params,
            config=(("new_tokens", new_tokens),),
        ),
        _ExactFamilyWork(
            name="attn_score",
            flops=(
                4
                * attention_pairs
                * full_layers
                * dims.num_heads
                * dims.head_size
            ),
            bytes_moved=0,
            config=(("new_tokens", new_tokens), ("kv_tokens", kv_tokens)),
        ),
        _ExactFamilyWork(
            name="kv_read",
            flops=0,
            bytes_moved=(
                2
                * kv_tokens
                * full_layers
                * dims.num_kv_heads
                * dims.head_size
                * activation_bytes
            ),
            config=(("kv_tokens", kv_tokens),),
        ),
        _ExactFamilyWork(
            name="gdn_input_projection",
            flops=2 * new_tokens * linear_layers * input_projection_params,
            bytes_moved=weight_bytes * linear_layers * input_projection_params,
            config=(("new_tokens", new_tokens),),
        ),
        _ExactFamilyWork(
            name="gdn_short_convolution",
            flops=2 * new_tokens * linear_layers * conv_params,
            bytes_moved=(
                weight_bytes * linear_layers * conv_params
                + 2
                * activation_bytes
                * sequences
                * linear_layers
                * conv_state_elements
            ),
            config=(("new_tokens", new_tokens), ("sequences", sequences)),
        ),
        _ExactFamilyWork(
            name="gdn_state_read",
            flops=0,
            bytes_moved=(
                state_bytes
                * sequences
                * linear_layers
                * recurrent_state_elements
            ),
            config=(("sequences", sequences),),
        ),
        _ExactFamilyWork(
            name="gdn_state_update",
            flops=new_tokens * linear_layers * update_flops_per_token,
            bytes_moved=state_bytes * linear_layers * 2 * value_heads,
            config=(("new_tokens", new_tokens),),
        ),
        _ExactFamilyWork(
            name="gdn_state_write",
            flops=0,
            bytes_moved=(
                state_bytes
                * sequences
                * linear_layers
                * recurrent_state_elements
            ),
            config=(("sequences", sequences),),
        ),
        _ExactFamilyWork(
            name="gdn_gated_norm",
            flops=new_tokens * linear_layers * norm_flops_per_token,
            bytes_moved=weight_bytes * linear_layers * value_head_dim,
            config=(("new_tokens", new_tokens),),
        ),
        _ExactFamilyWork(
            name="gdn_output_projection",
            flops=2 * new_tokens * linear_layers * output_projection_params,
            bytes_moved=weight_bytes * linear_layers * output_projection_params,
            config=(("new_tokens", new_tokens),),
        ),
        _ExactFamilyWork(
            name="mlp_gemm",
            flops=2 * new_tokens * dims.num_layers * mlp_params,
            bytes_moved=weight_bytes * dims.num_layers * mlp_params,
            config=(("new_tokens", new_tokens),),
        ),
        _ExactFamilyWork(
            name="lm_head",
            flops=2 * num_sampled * head_params,
            bytes_moved=weight_bytes * head_params,
            config=(("sampled", num_sampled),),
        ),
    )


def _qwen_gdn_total_work(
    dims: ModelDims,
    record: StepRecord,
    num_sampled: int,
    text_stack: FrameworkTextStack,
) -> tuple[int, int]:
    """Independently compose the Qwen total used for conservation checks."""

    new_tokens, kv_tokens, attention_pairs = step_shape(record)
    sequences = len(record.scheduled)
    linear_layers = text_stack.layer_types.count("linear_attention")
    full_layers = text_stack.layer_types.count("full_attention")
    hidden = dims.hidden_size
    key_heads = text_stack.linear_num_key_heads
    value_heads = text_stack.linear_num_value_heads
    key_head_dim = text_stack.linear_key_head_dim
    value_head_dim = text_stack.linear_value_head_dim
    key_width = key_heads * key_head_dim
    value_width = value_heads * value_head_dim
    conv_width = text_stack.linear_conv_kernel_dim
    conv_channels = 2 * key_width + value_width
    full_attention_params = hidden * (
        dims.num_heads * dims.head_size
        + 2 * dims.num_kv_heads * dims.head_size
    ) + dims.num_heads * dims.head_size * hidden
    input_projection_params = hidden * (
        2 * key_width + 2 * value_width + 2 * value_heads
    )
    conv_params = conv_channels * conv_width
    recurrent_state_elements = value_heads * value_head_dim * key_head_dim
    update_flops = key_heads * (7 * key_head_dim + 2) + value_heads * (
        7 * value_head_dim * key_head_dim + 2 * value_head_dim + 7
    )
    norm_flops = value_heads * (7 * value_head_dim + 2)
    output_projection_params = value_width * hidden
    mlp_params = 3 * hidden * dims.intermediate_size
    head_params = hidden * dims.vocab_size
    token_flops = (
        2 * full_layers * full_attention_params
        + 2 * linear_layers * input_projection_params
        + 2 * linear_layers * conv_params
        + linear_layers * update_flops
        + linear_layers * norm_flops
        + 2 * linear_layers * output_projection_params
        + 2 * dims.num_layers * mlp_params
    )
    flops = (
        token_flops * new_tokens
        + 4
        * attention_pairs
        * full_layers
        * dims.num_heads
        * dims.head_size
        + 2 * num_sampled * head_params
    )
    weight_bytes = int(dims.weight_element_bytes)
    static_bytes = weight_bytes * (
        full_layers * full_attention_params
        + linear_layers
        * (
            input_projection_params
            + conv_params
            + value_head_dim
            + output_projection_params
        )
        + dims.num_layers * mlp_params
        + head_params
    ) + 4 * linear_layers * 2 * value_heads
    sequence_bytes = linear_layers * (
        2 * dims.dtype_bytes * conv_channels * (conv_width - 1)
        + 2 * 4 * recurrent_state_elements
    )
    kv_bytes = (
        2
        * kv_tokens
        * full_layers
        * dims.num_kv_heads
        * dims.head_size
        * dims.dtype_bytes
    )
    return flops, static_bytes + sequence_bytes * sequences + kv_bytes


def _validate_deepseek_stack(stack: FrameworkDeepseekStack) -> None:
    expected_identity = {
        "architecture": "DeepseekV3ForCausalLM",
        "wrapper_model_type": "deepseek_v3",
        "scope": "text-only",
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "routed_scaling_factor": "5/2",
        "excluded_components": (
            "input-embedding-family",
            "normalization-family",
            "r1-reasoning-checkpoint",
        ),
    }
    mismatches = [
        name
        for name, expected in expected_identity.items()
        if getattr(stack, name) != expected
    ]
    expected_schedule = (
        ("dense",) * stack.first_k_dense_replace
        + ("moe",) * (stack.geometry.layers - stack.first_k_dense_replace)
    )
    if stack.layer_types != expected_schedule:
        mismatches.append("layer_types")
    if stack.moe_layer_freq != 1:
        mismatches.append("moe_layer_freq")
    if stack.geometry.num_experts <= 0 or stack.geometry.top_k <= 0:
        mismatches.append("expert_geometry")
    if stack.num_nextn_predict_layers != 1:
        mismatches.append("num_nextn_predict_layers")
    if stack.weight_block_size != (128, 128):
        mismatches.append("weight_block_size")
    if mismatches:
        raise ModelExtractionError(
            "unsupported DeepSeek-V3 stack fields: " + ", ".join(mismatches)
        )


def _fp8_matrix_bytes(rows: int, columns: int) -> int:
    """Serialized FP8 data plus one FP32 inverse scale per 128 by 128 block."""

    return (
        rows * columns
        + 4 * ((rows + 127) // 128) * ((columns + 127) // 128)
    )


def _deepseek_family_work(
    record: StepRecord,
    num_sampled: int,
    stack: FrameworkDeepseekStack,
    *,
    phase: str,
    mtp_enabled: bool,
) -> tuple[_ExactFamilyWork, ...]:
    """Project DeepSeek-V3 MLA, MoE, dense and optional MTP families."""

    _validate_deepseek_stack(stack)
    new_tokens, kv_tokens, attention_pairs = step_shape(record)
    hidden = stack.geometry.hidden_size
    dense_intermediate = stack.geometry.intermediate_size
    heads = stack.geometry.num_heads
    experts = stack.geometry.num_experts
    top_k = stack.geometry.top_k
    vocab = stack.geometry.vocab_size
    q_rank = stack.q_lora_rank
    kv_rank = stack.kv_lora_rank
    nope = stack.qk_nope_head_dim
    rope = stack.qk_rope_head_dim
    value = stack.v_head_dim
    dense_layers = stack.first_k_dense_replace
    moe_layers = stack.geometry.layers - dense_layers
    expert_intermediate = stack.moe_intermediate_size
    q_compression_bytes = _fp8_matrix_bytes(q_rank, hidden)
    q_decompression_bytes = _fp8_matrix_bytes(heads * (nope + rope), q_rank)
    kv_compression_bytes = _fp8_matrix_bytes(kv_rank + rope, hidden)
    kv_decompression_bytes = _fp8_matrix_bytes(
        heads * (nope + value), kv_rank
    )
    output_bytes = _fp8_matrix_bytes(hidden, heads * value)
    dense_bytes = _fp8_matrix_bytes(
        2 * dense_intermediate, hidden
    ) + _fp8_matrix_bytes(hidden, dense_intermediate)
    expert_bytes = _fp8_matrix_bytes(
        2 * expert_intermediate, hidden
    ) + _fp8_matrix_bytes(hidden, expert_intermediate)
    router_bytes = 2 * hidden * experts + 4 * experts
    expert_flops = 2 * 3 * hidden * expert_intermediate
    attention_flops_per_pair = (
        2 * heads * ((nope + rope) + value)
        if phase == "prefill"
        else 2 * heads * (2 * kv_rank + rope)
    )
    mtp_scale = int(mtp_enabled)
    mtp_fixed_flops = (
        2 * (2 * hidden) * hidden
        + 2 * hidden * q_rank
        + 2 * q_rank * heads * (nope + rope)
        + 2 * hidden * (kv_rank + rope)
        + 2 * kv_rank * heads * (nope + value)
        + 3 * rope * (heads + 1)
        + 2 * heads * value * hidden
        + 2 * hidden * experts
        + 2 * experts
        + 3 * top_k
        - 1
        + expert_flops
        + top_k * expert_flops
    )
    mtp_static_bytes = (
        2 * (2 * hidden) * hidden
        + q_compression_bytes
        + q_decompression_bytes
        + kv_compression_bytes
        + kv_decompression_bytes
        + output_bytes
        + router_bytes
        + expert_bytes
        + experts * expert_bytes
        + 2 * hidden * vocab
    )
    mtp_pair_flops = (
        2 * heads * ((nope + rope) + value)
        if phase == "prefill"
        else 2 * heads * (2 * kv_rank + rope)
    )

    return (
        _ExactFamilyWork(
            "mla_q_compression",
            2 * new_tokens * stack.geometry.layers * hidden * q_rank,
            stack.geometry.layers * q_compression_bytes,
            (("new_tokens", new_tokens),),
        ),
        _ExactFamilyWork(
            "mla_q_decompression",
            2
            * new_tokens
            * stack.geometry.layers
            * q_rank
            * heads
            * (nope + rope),
            stack.geometry.layers * q_decompression_bytes,
            (("new_tokens", new_tokens),),
        ),
        _ExactFamilyWork(
            "mla_kv_compression",
            2
            * new_tokens
            * stack.geometry.layers
            * hidden
            * (kv_rank + rope),
            stack.geometry.layers * kv_compression_bytes,
            (("new_tokens", new_tokens),),
        ),
        _ExactFamilyWork(
            "mla_kv_decompression",
            2
            * new_tokens
            * stack.geometry.layers
            * kv_rank
            * heads
            * (nope + value),
            stack.geometry.layers * kv_decompression_bytes,
            (("new_tokens", new_tokens),),
        ),
        _ExactFamilyWork(
            "mla_rotary_split",
            new_tokens
            * stack.geometry.layers
            * 3
            * rope
            * (heads + 1),
            0,
            (("new_tokens", new_tokens),),
        ),
        _ExactFamilyWork(
            "mla_attention",
            attention_pairs * stack.geometry.layers * attention_flops_per_pair,
            0,
            (("new_tokens", new_tokens), ("kv_tokens", kv_tokens)),
        ),
        _ExactFamilyWork(
            "mla_compressed_kv_read",
            0,
            2 * kv_tokens * stack.geometry.layers * (kv_rank + rope),
            (("kv_tokens", kv_tokens),),
        ),
        _ExactFamilyWork(
            "mla_output_projection",
            2
            * new_tokens
            * stack.geometry.layers
            * heads
            * value
            * hidden,
            stack.geometry.layers * output_bytes,
            (("new_tokens", new_tokens),),
        ),
        _ExactFamilyWork(
            "dense_early_mlp",
            2 * new_tokens * dense_layers * 3 * hidden * dense_intermediate,
            dense_layers * dense_bytes,
            (("new_tokens", new_tokens),),
        ),
        _ExactFamilyWork(
            "moe_router",
            new_tokens
            * moe_layers
            * (2 * hidden * experts + 2 * experts + 3 * top_k - 1),
            moe_layers * router_bytes,
            (("new_tokens", new_tokens),),
        ),
        _ExactFamilyWork(
            "moe_shared_expert",
            new_tokens * moe_layers * expert_flops,
            moe_layers * expert_bytes,
            (("new_tokens", new_tokens),),
        ),
        _ExactFamilyWork(
            "moe_routed_experts",
            new_tokens * moe_layers * top_k * expert_flops,
            moe_layers * experts * expert_bytes,
            (("new_tokens", new_tokens),),
        ),
        _ExactFamilyWork(
            "lm_head",
            2 * num_sampled * hidden * vocab,
            2 * hidden * vocab,
            (("sampled", num_sampled),),
        ),
        _ExactFamilyWork(
            "multi_token_prediction_head",
            mtp_scale
            * (
                new_tokens * mtp_fixed_flops
                + attention_pairs * mtp_pair_flops
                + 2 * num_sampled * hidden * vocab
            ),
            mtp_scale
            * (
                mtp_static_bytes
                + 2 * kv_tokens * (kv_rank + rope)
            ),
            (
                ("new_tokens", new_tokens),
                ("kv_tokens", kv_tokens),
                ("sampled", num_sampled),
                ("mtp_enabled", mtp_scale),
            ),
        ),
    )


def _deepseek_total_work(
    record: StepRecord,
    num_sampled: int,
    stack: FrameworkDeepseekStack,
    *,
    phase: str,
    mtp_enabled: bool,
) -> tuple[int, int]:
    """Independently compose the DeepSeek total used for conservation."""

    new_tokens, kv_tokens, attention_pairs = step_shape(record)
    h = stack.geometry.hidden_size
    intermediate = stack.geometry.intermediate_size
    heads = stack.geometry.num_heads
    experts = stack.geometry.num_experts
    top_k = stack.geometry.top_k
    vocab = stack.geometry.vocab_size
    q_rank = stack.q_lora_rank
    kv_rank = stack.kv_lora_rank
    nope = stack.qk_nope_head_dim
    rope = stack.qk_rope_head_dim
    value = stack.v_head_dim
    layers = stack.geometry.layers
    dense_layers = stack.first_k_dense_replace
    moe_layers = layers - dense_layers
    expert_intermediate = stack.moe_intermediate_size
    expert_flops = 2 * 3 * h * expert_intermediate
    router_flops = 2 * h * experts + 2 * experts + 3 * top_k - 1
    fixed_flops = layers * (
        2 * h * q_rank
        + 2 * q_rank * heads * (nope + rope)
        + 2 * h * (kv_rank + rope)
        + 2 * kv_rank * heads * (nope + value)
        + 3 * rope * (heads + 1)
        + 2 * heads * value * h
    ) + dense_layers * 2 * 3 * h * intermediate + moe_layers * (
        router_flops + expert_flops + top_k * expert_flops
    )
    pair_flops = (
        2 * heads * ((nope + rope) + value) * layers
        if phase == "prefill"
        else 2 * heads * (2 * kv_rank + rope) * layers
    )
    q_compression_bytes = _fp8_matrix_bytes(q_rank, h)
    q_decompression_bytes = _fp8_matrix_bytes(heads * (nope + rope), q_rank)
    kv_compression_bytes = _fp8_matrix_bytes(kv_rank + rope, h)
    kv_decompression_bytes = _fp8_matrix_bytes(heads * (nope + value), kv_rank)
    output_bytes = _fp8_matrix_bytes(h, heads * value)
    dense_bytes = _fp8_matrix_bytes(
        2 * intermediate, h
    ) + _fp8_matrix_bytes(h, intermediate)
    expert_bytes = _fp8_matrix_bytes(
        2 * expert_intermediate, h
    ) + _fp8_matrix_bytes(h, expert_intermediate)
    router_bytes = 2 * h * experts + 4 * experts
    static_bytes = layers * (
        q_compression_bytes
        + q_decompression_bytes
        + kv_compression_bytes
        + kv_decompression_bytes
        + output_bytes
    ) + dense_layers * dense_bytes + moe_layers * (
        router_bytes + expert_bytes + experts * expert_bytes
    ) + 2 * h * vocab
    flops = (
        fixed_flops * new_tokens
        + pair_flops * attention_pairs
        + 2 * num_sampled * h * vocab
    )
    hbm_bytes = static_bytes + 2 * kv_tokens * layers * (kv_rank + rope)
    if mtp_enabled:
        mtp_fixed = (
            2 * (2 * h) * h
            + 2 * h * q_rank
            + 2 * q_rank * heads * (nope + rope)
            + 2 * h * (kv_rank + rope)
            + 2 * kv_rank * heads * (nope + value)
            + 3 * rope * (heads + 1)
            + 2 * heads * value * h
            + router_flops
            + expert_flops
            + top_k * expert_flops
        )
        mtp_pair = (
            2 * heads * ((nope + rope) + value)
            if phase == "prefill"
            else 2 * heads * (2 * kv_rank + rope)
        )
        mtp_static = (
            2 * (2 * h) * h
            + q_compression_bytes
            + q_decompression_bytes
            + kv_compression_bytes
            + kv_decompression_bytes
            + output_bytes
            + router_bytes
            + expert_bytes
            + experts * expert_bytes
            + 2 * h * vocab
        )
        flops += (
            mtp_fixed * new_tokens
            + mtp_pair * attention_pairs
            + 2 * num_sampled * h * vocab
        )
        hbm_bytes += mtp_static + 2 * kv_tokens * (kv_rank + rope)
    return flops, hbm_bytes


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
    ordered_families: tuple[str, ...] = ORDERED_FAMILIES,
) -> tuple[ShapeSchema, ...]:
    schemas = []
    for family_index, family in enumerate(ordered_families):
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


def _qwen_gdn_family_definitions(
    text_stack: FrameworkTextStack,
) -> tuple[KernelFamilyDefinition, ...]:
    linear_layers = text_stack.layer_types.count("linear_attention")
    full_layers = text_stack.layer_types.count("full_attention")
    layer_counts = {
        "attn_gemm": full_layers,
        "attn_score": full_layers,
        "kv_read": full_layers,
        "gdn_input_projection": linear_layers,
        "gdn_short_convolution": linear_layers,
        "gdn_state_read": linear_layers,
        "gdn_state_update": linear_layers,
        "gdn_state_write": linear_layers,
        "gdn_gated_norm": linear_layers,
        "gdn_output_projection": linear_layers,
        "mlp_gemm": text_stack.geometry.layers,
        "lm_head": 1,
    }
    return tuple(
        KernelFamilyDefinition(
            family_id=family,
            shape_schema_id=_SHAPE_SCHEMA_IDS[family],
            phase_launch_counts=tuple(
                PhaseLaunchCount(
                    phase=phase,
                    logical_launch_count=layer_counts[family],
                )
                for phase in ("prefill", "decode")
            ),
        )
        for family in QWEN_GATED_DELTA_NET_FAMILIES
    )


def _deepseek_family_definitions(
    stack: FrameworkDeepseekStack,
) -> tuple[KernelFamilyDefinition, ...]:
    moe_layers = stack.geometry.layers - stack.first_k_dense_replace
    layer_counts = {
        "mla_q_compression": stack.geometry.layers,
        "mla_q_decompression": stack.geometry.layers,
        "mla_kv_compression": stack.geometry.layers,
        "mla_kv_decompression": stack.geometry.layers,
        "mla_rotary_split": stack.geometry.layers,
        "mla_attention": stack.geometry.layers,
        "mla_compressed_kv_read": stack.geometry.layers,
        "mla_output_projection": stack.geometry.layers,
        "dense_early_mlp": stack.first_k_dense_replace,
        "moe_router": moe_layers,
        "moe_shared_expert": moe_layers,
        "moe_routed_experts": moe_layers,
        "lm_head": 1,
        "multi_token_prediction_head": 1,
    }
    return tuple(
        KernelFamilyDefinition(
            family_id=family,
            shape_schema_id=_SHAPE_SCHEMA_IDS[family],
            phase_launch_counts=tuple(
                PhaseLaunchCount(
                    phase=phase,
                    logical_launch_count=layer_counts[family],
                )
                for phase in ("prefill", "decode")
            ),
            launch_scale_axis=(
                "mtp_enabled"
                if family == "multi_token_prediction_head"
                else None
            ),
        )
        for family in DEEPSEEK_V3_FAMILIES
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
        logical_launch_count = definition.count_for(phase)
        if definition.launch_scale_axis is not None:
            logical_launch_count *= config[definition.launch_scale_axis]
        result.append(
            KernelProjection(
                family_id=spec.name,
                shape_vector=ShapeVector(
                    shape_schema_id=definition.shape_schema_id,
                    values=values,
                ),
                logical_launch_count=logical_launch_count,
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
) -> FrameworkTextStack:
    if projection.text_stack is None:
        raise ModelExtractionError("framework omitted the declared text stack")
    projected = projection.text_stack
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
        len(schedule) != projected.geometry.layers
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
    if projected.to_obj() != expected:
        raise ModelExtractionError("framework text-stack projection does not match the suite")
    if linear_layers:
        _validate_qwen_gdn_text_stack(projected)
    return projected


def _validate_dense_stack(
    reference_model: Mapping[str, Any],
    projection: FrameworkConfigurationProjection,
) -> FrameworkDenseStack:
    if projection.dense_stack is None:
        raise ModelExtractionError("framework omitted the declared dense stack")
    stack = reference_model.get("dense_stack")
    if not isinstance(stack, dict):
        raise ModelExtractionError("suite dense-stack contract must be an object")
    expected_fields = {
        "scope",
        "attention_mechanism",
        "quantization",
        "weight_block_size",
        "excluded_components",
    }
    if set(stack) != expected_fields:
        raise ModelExtractionError("suite dense-stack contract has unexpected fields")
    expected = {
        "architecture": reference_model.get("architecture"),
        "model_type": reference_model.get("model_type"),
        "geometry": reference_model["geometry"],
        **stack,
    }
    projected = projection.dense_stack
    if projected.to_obj() != expected:
        raise ModelExtractionError(
            "framework dense-stack projection does not match the suite"
        )
    if projected.geometry.num_experts or projected.geometry.top_k:
        raise ModelExtractionError("dense-stack projection cannot declare experts")
    if projected.attention_mechanism != "grouped-query causal self-attention":
        raise ModelExtractionError("dense-stack attention mechanism changed")
    if projected.quantization != "fp8-e4m3-block-128x128":
        raise ModelExtractionError("dense-stack quantization changed")
    if projected.weight_block_size != (128, 128):
        raise ModelExtractionError("dense-stack FP8 block changed")
    return projected


def _validate_deepseek_stack_contract(
    reference_model: Mapping[str, Any],
    projection: FrameworkConfigurationProjection,
) -> FrameworkDeepseekStack:
    if projection.deepseek_stack is None:
        raise ModelExtractionError("framework omitted the declared DeepSeek stack")
    stack = reference_model.get("deepseek_stack")
    if not isinstance(stack, dict):
        raise ModelExtractionError("suite DeepSeek stack contract must be an object")
    expected_fields = {
        "scope",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "first_k_dense_replace",
        "moe_intermediate_size",
        "moe_layer_freq",
        "n_shared_experts",
        "n_group",
        "topk_group",
        "scoring_func",
        "topk_method",
        "norm_topk_prob",
        "routed_scaling_factor",
        "num_nextn_predict_layers",
        "weight_block_size",
        "excluded_components",
    }
    if set(stack) != expected_fields:
        raise ModelExtractionError("suite DeepSeek stack contract has unexpected fields")
    layers = reference_model["geometry"]["layers"]
    dense_layers = stack["first_k_dense_replace"]
    schedule = ["dense"] * dense_layers + ["moe"] * (layers - dense_layers)
    expected = {
        "architecture": reference_model.get("architecture"),
        "wrapper_model_type": reference_model.get("model_type"),
        "scope": stack["scope"],
        "geometry": reference_model["geometry"],
        "layer_types": schedule,
        **{
            name: stack[name]
            for name in expected_fields
            if name != "scope"
        },
    }
    projected = projection.deepseek_stack
    if projected.to_obj() != expected:
        raise ModelExtractionError(
            "framework DeepSeek stack projection does not match the suite"
        )
    _validate_deepseek_stack(projected)
    return projected


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
    case_records_from_suite(suite)
    has_dense_contract = "dense_stack" in reference_model
    has_text_contract = "text_stack" in reference_model
    has_deepseek_contract = "deepseek_stack" in reference_model
    if sum((has_dense_contract, has_text_contract, has_deepseek_contract)) > 1:
        raise ModelExtractionError("suite cannot declare two structure contracts")
    text_stack = None
    deepseek_stack = None
    if (
        has_dense_contract or has_text_contract or has_deepseek_contract
    ) and framework_projection is None:
        raise ModelExtractionError("suite requires a framework structure projection")
    if framework_projection is not None:
        if has_dense_contract:
            _validate_dense_stack(reference_model, framework_projection)
        elif has_text_contract:
            text_stack = _validate_text_stack(reference_model, framework_projection)
        elif has_deepseek_contract:
            deepseek_stack = _validate_deepseek_stack_contract(
                reference_model, framework_projection
            )
        else:
            raise ModelExtractionError("framework supplied an undeclared structure stack")
    validate_framework_dims(framework_dims, model, deepseek_stack)
    records = _records_from_suite(suite, step_records_path)
    cells = suite["graph_cells"]
    case_dims = tuple(
        _case_dims(framework_dims, cell) for cell in cells
    )
    has_gated_delta_net = text_stack is not None and (
        "linear_attention" in text_stack.layer_types
    )
    if deepseek_stack is not None:
        ordered_families = DEEPSEEK_V3_FAMILIES
        specs_by_case = tuple(
            _deepseek_family_work(
                record,
                record.num_sampled or 0,
                deepseek_stack,
                phase=cell["phase"],
                mtp_enabled=cell["mtp_enabled"],
            )
            for cell, record in zip(cells, records, strict=True)
        )
        definitions = _deepseek_family_definitions(deepseek_stack)
    elif has_gated_delta_net:
        assert text_stack is not None
        ordered_families = QWEN_GATED_DELTA_NET_FAMILIES
        specs_by_case = tuple(
            _qwen_gdn_family_work(
                dims,
                record,
                record.num_sampled or 0,
                text_stack,
            )
            for dims, record in zip(case_dims, records, strict=True)
        )
        definitions = _qwen_gdn_family_definitions(text_stack)
    else:
        ordered_families = ORDERED_FAMILIES
        specs_by_case = tuple(
            tuple(step_kernels(dims, record, record.num_sampled or 0))
            for dims, record in zip(case_dims, records, strict=True)
        )
        definitions = _family_definitions(model.geometry.layers)
    if any(
        tuple(spec.name for spec in specs) != ordered_families
        for specs in specs_by_case
    ):
        raise ModelExtractionError("inventory builder changed the frozen family order")
    schemas = _shape_schemas(specs_by_case, ordered_families)
    cases = []
    for ordinal, (cell, record, dims, specs) in enumerate(
        zip(cells, records, case_dims, specs_by_case, strict=True)
    ):
        projected_flops = sum(_exact_work(spec.flops, "family.flops") for spec in specs)
        projected_bytes = sum(
            _exact_work(spec.bytes_moved, "family.bytes_moved") for spec in specs
        )
        if deepseek_stack is not None:
            fused_flops, fused_bytes = _deepseek_total_work(
                record,
                record.num_sampled or 0,
                deepseek_stack,
                phase=cell["phase"],
                mtp_enabled=cell["mtp_enabled"],
            )
        elif has_gated_delta_net:
            assert text_stack is not None
            fused_flops, fused_bytes = _qwen_gdn_total_work(
                dims,
                record,
                record.num_sampled or 0,
                text_stack,
            )
        else:
            fused = step_kernel(dims, record, record.num_sampled or 0)
            fused_flops = _exact_work(fused.flops, "fused.flops")
            fused_bytes = _exact_work(fused.bytes_moved, "fused.bytes_moved")
        if projected_flops != fused_flops:
            raise ModelExtractionError(f"case {cell['id']!r} family FLOPs are not exact")
        if projected_bytes != fused_bytes:
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
    if deepseek_stack is not None:
        join_task = "VLLM-38" if framework.framework_id == "vllm" else "SGL-34"
    else:
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
    "DEEPSEEK_V3_FAMILIES",
    "LAYER_REPEATED_FAMILIES",
    "ORDERED_FAMILIES",
    "QWEN_GATED_DELTA_NET_FAMILIES",
    "FrameworkConfigurationProjection",
    "FrameworkDeepseekStack",
    "FrameworkDenseStack",
    "FrameworkTextStack",
    "ModelExtractionError",
    "case_records_from_suite",
    "extract_model_inventory",
    "load_extraction_suite",
    "validate_checkpoint",
    "validate_framework_dims",
]
