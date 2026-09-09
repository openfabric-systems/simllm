"""CPU-only vLLM driver for the model kernel inventory."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
from typing import Any

from simllm.adapters.vllm._version import PINNED_VLLM_VERSION
from simllm.calibration.extraction import (
    FrameworkConfigurationProjection,
    FrameworkDeepseekStack,
    FrameworkDenseStack,
    FrameworkTextStack,
    extract_model_inventory,
)
from simllm.calibration.kimi_k3 import exact_config_ratio, validate_kimi_k3_native_contract
from simllm.calibration.model_inventory import (
    FrameworkIdentity,
    ModelGeometry,
    ModelKernelInventory,
)
from simllm.compute import ModelDims
from simllm.compute.kimi_k3 import KdaSpec, KimiK3Spec, LatentMoeSpec, MlaSpec

VLLM_KIMI_K3_BINDING = "vllm.models.kimi_k3:KimiK3ForConditionalGeneration"
VLLM_KIMI_K3_IMPLEMENTATION = "KimiK3DeltaAttention and MultiHeadLatentAttention"
VLLM_SOURCE_COMMIT = "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac"
VLLM_EXTRACTION_SEAM = "flagged-skeleton-step-record-v1"
VLLM_CONFIGURATION_SEAM = "ModelConfig-with-skip-tokenizer-init"
VLLM_QWEN35_BINDING = "model_executor/models/registry.py:573"
VLLM_QWEN35_IMPLEMENTATION = "QwenGatedDeltaNetAttention"
VLLM_QWEN3_BINDING = "model_executor/models/registry.py:199"
VLLM_QWEN3_IMPLEMENTATION = "Qwen3Attention"
VLLM_DEEPSEEK_V3_BINDING = "model_executor/models/registry.py:93"
VLLM_DEEPSEEK_V3_IMPLEMENTATION = "DeepseekV2Attention and DeepseekV2MoE"


def _framework_identity(version: str) -> FrameworkIdentity:
    return FrameworkIdentity(
        framework_id="vllm",
        version=version,
        source_commit=VLLM_SOURCE_COMMIT,
        source_tree=None,
        entry_seam=VLLM_EXTRACTION_SEAM,
    )


def _qwen35_projection(
    model_config: Any,
    dims: Any,
    framework: FrameworkIdentity,
) -> FrameworkConfigurationProjection | None:
    hf = model_config.hf_config
    architectures = tuple(hf.architectures or ())
    if architectures != ("Qwen3_5ForConditionalGeneration",):
        return None
    from vllm.model_executor.models.registry import ModelRegistry

    architecture = architectures[0]
    registered = ModelRegistry.models.get(architecture)
    model_info, resolved = ModelRegistry.inspect_model_cls(architecture, model_config)
    if (
        registered is None
        or getattr(registered, "module_name", None)
        != "vllm.model_executor.models.qwen3_5"
        or getattr(registered, "class_name", None) != architecture
        or resolved != architecture
        or not model_info.is_hybrid
        or not model_info.supports_multimodal
    ):
        raise RuntimeError("vLLM Qwen3.5 architecture binding does not match the pin")
    text = model_config.hf_text_config
    return FrameworkConfigurationProjection(
        framework=framework,
        configuration_seam=VLLM_CONFIGURATION_SEAM,
        architecture_binding=VLLM_QWEN35_BINDING,
        text_implementation=VLLM_QWEN35_IMPLEMENTATION,
        text_stack=FrameworkTextStack(
            architecture=architecture,
            wrapper_model_type=hf.model_type,
            text_model_type=text.model_type,
            scope="text-only",
            geometry=ModelGeometry(
                layers=dims.num_layers,
                hidden_size=dims.hidden_size,
                intermediate_size=dims.intermediate_size,
                num_heads=dims.num_heads,
                num_kv_heads=dims.num_kv_heads,
                head_size=dims.head_size,
                num_experts=dims.num_experts,
                top_k=dims.top_k,
                vocab_size=dims.vocab_size,
            ),
            layer_types=tuple(text.layer_types),
            linear_attention_mechanism="Qwen3.5 Gated DeltaNet",
            linear_conv_kernel_dim=text.linear_conv_kernel_dim,
            linear_key_head_dim=text.linear_key_head_dim,
            linear_value_head_dim=text.linear_value_head_dim,
            linear_num_key_heads=text.linear_num_key_heads,
            linear_num_value_heads=text.linear_num_value_heads,
            attn_output_gate=text.attn_output_gate,
            output_gate_type=text.output_gate_type,
            state_dtype=text.mamba_ssm_dtype,
            excluded_components=(
                "multimodal-vision-encoder",
                "one-layer-mtp-speculative-head",
            ),
        ),
    )


def _qwen3_projection(
    model_config: Any,
    framework: FrameworkIdentity,
) -> FrameworkConfigurationProjection | None:
    hf = model_config.hf_config
    architectures = tuple(hf.architectures or ())
    if architectures != ("Qwen3ForCausalLM",):
        return None
    from vllm.model_executor.models.registry import ModelRegistry

    architecture = architectures[0]
    registered = ModelRegistry.models.get(architecture)
    if (
        registered is None
        or getattr(registered, "module_name", None)
        != "vllm.model_executor.models.qwen3"
        or getattr(registered, "class_name", None) != architecture
    ):
        raise RuntimeError("vLLM Qwen3 architecture binding does not match the pin")
    quantization = hf.quantization_config
    if not isinstance(quantization, dict):
        raise TypeError("vLLM Qwen3 quantization config is not an object")
    geometry = ModelGeometry(
        layers=hf.num_hidden_layers,
        hidden_size=hf.hidden_size,
        intermediate_size=hf.intermediate_size,
        num_heads=hf.num_attention_heads,
        num_kv_heads=hf.num_key_value_heads,
        head_size=hf.head_dim,
        num_experts=0,
        top_k=0,
        vocab_size=hf.vocab_size,
    )
    return FrameworkConfigurationProjection(
        framework=framework,
        configuration_seam=VLLM_CONFIGURATION_SEAM,
        architecture_binding=VLLM_QWEN3_BINDING,
        text_implementation=VLLM_QWEN3_IMPLEMENTATION,
        dense_stack=FrameworkDenseStack(
            architecture=architecture,
            model_type=hf.model_type,
            scope="text-only",
            geometry=geometry,
            attention_mechanism="grouped-query causal self-attention",
            quantization="fp8-e4m3-block-128x128",
            weight_block_size=tuple(quantization["weight_block_size"]),
            excluded_components=("input-embedding-family", "normalization-family"),
        ),
    )


def _qwen3_dims(stack: FrameworkDenseStack) -> ModelDims:
    geometry = stack.geometry
    return ModelDims(
        num_layers=geometry.layers,
        hidden_size=geometry.hidden_size,
        intermediate_size=geometry.intermediate_size,
        num_heads=geometry.num_heads,
        num_kv_heads=geometry.num_kv_heads,
        head_size=geometry.head_size,
        vocab_size=geometry.vocab_size,
        dtype_bytes=2,
        weight_dtype_bytes=1,
        kv_dtype_bytes=2,
    )


def _deepseek_v3_projection(
    model_config: Any,
    framework: FrameworkIdentity,
) -> FrameworkConfigurationProjection | None:
    hf = model_config.hf_config
    architectures = tuple(hf.architectures or ())
    if architectures != ("DeepseekV3ForCausalLM",):
        return None
    from vllm.model_executor.models.registry import ModelRegistry

    architecture = architectures[0]
    registered = ModelRegistry.models.get(architecture)
    if (
        registered is None
        or getattr(registered, "module_name", None)
        != "vllm.model_executor.models.deepseek_v2"
        or getattr(registered, "class_name", None) != architecture
    ):
        raise RuntimeError("vLLM DeepSeek-V3 architecture binding does not match the pin")
    quantization = hf.quantization_config
    if not isinstance(quantization, dict):
        raise TypeError("vLLM DeepSeek-V3 quantization config is not an object")
    if float(hf.routed_scaling_factor) != 2.5:
        raise RuntimeError("vLLM DeepSeek-V3 routed scaling factor changed")
    geometry = ModelGeometry(
        layers=hf.num_hidden_layers,
        hidden_size=hf.hidden_size,
        intermediate_size=hf.intermediate_size,
        num_heads=hf.num_attention_heads,
        num_kv_heads=hf.num_key_value_heads,
        head_size=hf.qk_nope_head_dim + hf.qk_rope_head_dim,
        num_experts=hf.n_routed_experts,
        top_k=hf.num_experts_per_tok,
        vocab_size=hf.vocab_size,
    )
    return FrameworkConfigurationProjection(
        framework=framework,
        configuration_seam=VLLM_CONFIGURATION_SEAM,
        architecture_binding=VLLM_DEEPSEEK_V3_BINDING,
        text_implementation=VLLM_DEEPSEEK_V3_IMPLEMENTATION,
        deepseek_stack=FrameworkDeepseekStack(
            architecture=architecture,
            wrapper_model_type=hf.model_type,
            scope="text-only",
            geometry=geometry,
            layer_types=("dense",) * hf.first_k_dense_replace
            + ("moe",) * (hf.num_hidden_layers - hf.first_k_dense_replace),
            q_lora_rank=hf.q_lora_rank,
            kv_lora_rank=hf.kv_lora_rank,
            qk_nope_head_dim=hf.qk_nope_head_dim,
            qk_rope_head_dim=hf.qk_rope_head_dim,
            v_head_dim=hf.v_head_dim,
            first_k_dense_replace=hf.first_k_dense_replace,
            moe_intermediate_size=hf.moe_intermediate_size,
            moe_layer_freq=hf.moe_layer_freq,
            n_shared_experts=hf.n_shared_experts,
            n_group=hf.n_group,
            topk_group=hf.topk_group,
            scoring_func=hf.scoring_func,
            topk_method=hf.topk_method,
            norm_topk_prob=hf.norm_topk_prob,
            routed_scaling_factor="5/2",
            num_nextn_predict_layers=hf.num_nextn_predict_layers,
            weight_block_size=tuple(quantization["weight_block_size"]),
            excluded_components=(
                "input-embedding-family",
                "normalization-family",
                "r1-reasoning-checkpoint",
            ),
        ),
    )


def _deepseek_v3_dims(stack: FrameworkDeepseekStack) -> ModelDims:
    geometry = stack.geometry
    return ModelDims(
        num_layers=geometry.layers,
        hidden_size=geometry.hidden_size,
        intermediate_size=geometry.intermediate_size,
        num_heads=geometry.num_heads,
        num_kv_heads=geometry.num_kv_heads,
        head_size=geometry.head_size,
        vocab_size=geometry.vocab_size,
        dtype_bytes=2,
        weight_dtype_bytes=1,
        kv_dtype_bytes=2,
        num_experts=geometry.num_experts,
        top_k=geometry.top_k,
        moe_intermediate_size=stack.moe_intermediate_size,
        local_num_experts=geometry.num_experts,
    )


def _kimi_k3_projection(
    model_config: Any,
    framework: FrameworkIdentity,
) -> FrameworkConfigurationProjection | None:
    hf = model_config.hf_config
    if tuple(hf.architectures or ()) != ("KimiK3ForConditionalGeneration",):
        return None
    from vllm.model_executor.models.registry import ModelRegistry

    registered = ModelRegistry.models.get("KimiK3ForConditionalGeneration")
    if (registered is None or getattr(registered, "module_name", None) != "vllm.models.kimi_k3"
            or getattr(registered, "class_name", None) != "KimiK3ForConditionalGeneration"):
        raise RuntimeError("vLLM K3 architecture binding does not match the pin")
    text = model_config.hf_text_config
    layer_types = tuple("kda" if text.is_kda_layer(i) else "mla"
                        for i in range(text.num_hidden_layers))
    quantization = validate_kimi_k3_native_contract(text, layer_types, quant_method="mxfp4")
    if str(model_config.dtype) not in {"bfloat16", "torch.bfloat16"}:
        raise ValueError("native K3 extraction requires BF16 activation configuration")
    linear = text.linear_attn_config
    spec = KimiK3Spec(
        hidden_size=text.hidden_size,
        vocab_size=text.vocab_size,
        dense_intermediate_size=text.intermediate_size,
        dense_prefix_layers=text.first_k_dense_replace,
        layer_types=layer_types,
        residual_block_size=text.attn_res_block_size,
        kda=KdaSpec(
            heads=linear["num_heads"], head_dim=linear["head_dim"],
            convolution_width=linear["short_conv_kernel_size"],
            full_rank_output_gate=linear["use_full_rank_gate"],
            gate_lower_bound=exact_config_ratio(linear["gate_lower_bound"]),
            recurrent_dtype="float32", convolution_state_dtype="bfloat16",
        ),
        mla=MlaSpec(
            heads=text.num_attention_heads, query_rank=text.q_lora_rank,
            kv_rank=text.kv_lora_rank, score_nope_dim=text.qk_nope_head_dim,
            score_extra_dim=text.qk_rope_head_dim, value_dim=text.v_head_dim,
            skip_rotary=text.mla_use_nope, output_gate=text.mla_use_output_gate,
            cache_dtype="bfloat16",
        ),
        moe=LatentMoeSpec(
            experts=text.num_experts, selected_experts=text.num_experts_per_token,
            latent_width=text.routed_expert_hidden_size,
            intermediate_size=text.moe_intermediate_size,
            shared_experts=text.num_shared_experts,
            normalize_latent_output=text.latent_moe_use_norm,
            router_activation=text.moe_router_activation_func,
            topk_method=text.topk_method, renormalize=text.moe_renormalize,
            expert_groups=text.num_expert_group, topk_groups=text.topk_group,
            routed_scale=exact_config_ratio(text.routed_scaling_factor),
            activation=text.hidden_act, situ_beta=exact_config_ratio(text.activation_situ_beta),
            situ_linear_beta=exact_config_ratio(text.activation_situ_linear_beta),
        ),
        rms_norm_epsilon=exact_config_ratio(text.rms_norm_eps),
        activation_dtype="bfloat16", tied_embeddings=text.tie_word_embeddings,
        max_context=text.max_position_embeddings,
        checkpoint_quantization=quantization,
    )
    return FrameworkConfigurationProjection(
        framework=framework, configuration_seam=VLLM_CONFIGURATION_SEAM,
        architecture_binding=VLLM_KIMI_K3_BINDING,
        text_implementation=VLLM_KIMI_K3_IMPLEMENTATION, kimi_k3_stack=spec,
    )


def _configuration(
    checkpoint_root: Path,
) -> tuple[Any, Any, FrameworkConfigurationProjection | None]:
    if os.environ.get("SIMLLM_VLLM_WORKER_MODE") != "skeleton":
        raise RuntimeError(
            "vLLM extraction requires SIMLLM_VLLM_WORKER_MODE=skeleton"
        )
    version = importlib.metadata.version("vllm")
    if version != PINNED_VLLM_VERSION:
        raise RuntimeError(
            f"vLLM extraction requires version {PINNED_VLLM_VERSION}, found {version}"
        )
    from vllm.config import ModelConfig, ParallelConfig, VllmConfig

    from simllm.adapters.vllm.executor import model_dims_from_vllm_config

    model_config = ModelConfig(
        model=str(checkpoint_root),
        revision=checkpoint_root.name,
        trust_remote_code=False,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        enforce_eager=True,
    )
    framework = _framework_identity(version)
    kimi_projection = _kimi_k3_projection(model_config, framework)
    if kimi_projection is not None:
        return None, framework, kimi_projection
    qwen3_projection = _qwen3_projection(model_config, framework)
    if qwen3_projection is not None:
        assert qwen3_projection.dense_stack is not None
        return (
            _qwen3_dims(qwen3_projection.dense_stack),
            framework,
            qwen3_projection,
        )
    config = VllmConfig(
        model_config=model_config,
        parallel_config=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
    )
    deepseek_projection = _deepseek_v3_projection(model_config, framework)
    if deepseek_projection is not None:
        assert deepseek_projection.deepseek_stack is not None
        dims = _deepseek_v3_dims(deepseek_projection.deepseek_stack)
        projection = deepseek_projection
    else:
        dims = model_dims_from_vllm_config(config)
        projection = _qwen35_projection(model_config, dims, framework)
    return dims, framework, projection


def inspect_configuration(checkpoint_root: Path) -> dict[str, Any]:
    """Return the pinned framework structure without loading weights."""

    _, _, projection = _configuration(checkpoint_root)
    if projection is None:
        raise RuntimeError("vLLM configuration has no pinned extraction wrapper")
    return projection.to_obj()


def extract(
    *,
    suite_raw: bytes,
    checkpoint_root: Path,
    step_records_path: Path,
    attention_shape_version: int = 1,
) -> ModelKernelInventory:
    """Extract through vLLM's flagged skeleton configuration boundary."""

    dims, framework, projection = _configuration(checkpoint_root)
    return extract_model_inventory(
        suite_raw=suite_raw,
        framework=framework,
        checkpoint_root=checkpoint_root,
        framework_dims=dims,
        step_records_path=step_records_path,
        framework_projection=projection,
        attention_shape_version=attention_shape_version,
    )


__all__ = [
    "VLLM_CONFIGURATION_SEAM",
    "VLLM_DEEPSEEK_V3_BINDING",
    "VLLM_DEEPSEEK_V3_IMPLEMENTATION",
    "VLLM_EXTRACTION_SEAM",
    "VLLM_QWEN3_BINDING",
    "VLLM_QWEN3_IMPLEMENTATION",
    "VLLM_QWEN35_BINDING",
    "VLLM_QWEN35_IMPLEMENTATION",
    "VLLM_SOURCE_COMMIT",
    "extract",
    "inspect_configuration",
]
