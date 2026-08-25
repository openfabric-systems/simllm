"""CPU-engine SGLang driver for the model kernel inventory."""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from simllm.calibration.extraction import (
    FrameworkConfigurationProjection,
    FrameworkTextStack,
    extract_model_inventory,
)
from simllm.calibration.model_inventory import (
    FrameworkIdentity,
    ModelGeometry,
    ModelKernelInventory,
)

SGLANG_VERSION = "0.5.19.dev345+gbfeae4e79"
SGLANG_SOURCE_COMMIT = "bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3"
SGLANG_SOURCE_TREE = "9ffe149f40e1cd5bff7dadc6806ad1927d312e69"
SGLANG_EXTRACTION_SEAM = "cpu-engine-step-record-v1"
SGLANG_CONFIGURATION_SEAM = (
    "DeviceConfig-cpu-plus-ModelConfig-with-multimodal-disabled"
)
SGLANG_QWEN35_BINDING = "python/sglang/srt/models/qwen3_5.py:2319"
SGLANG_QWEN35_IMPLEMENTATION = "Qwen3_5GatedDeltaNet with RadixLinearAttention"


def _source_root(distribution: importlib.metadata.Distribution) -> Path:
    raw = distribution.read_text("direct_url.json")
    if raw is None:
        raise RuntimeError("SGLang extraction requires editable source metadata")
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"url", "dir_info"}:
        raise RuntimeError("SGLang direct_url.json has unexpected fields")
    if value["dir_info"] != {"editable": True}:
        raise RuntimeError("SGLang extraction requires an editable pinned source tree")
    parsed = urlparse(value["url"])
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise RuntimeError("SGLang source metadata must name a local file URL")
    package_root = Path(unquote(parsed.path)).resolve()
    repository = package_root.parent
    if not (repository / ".git").exists():
        raise RuntimeError("SGLang editable source is not inside a git repository")
    return repository


def _git_object(repository: Path, expression: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", expression],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _framework_identity() -> FrameworkIdentity:
    return FrameworkIdentity(
        framework_id="sglang",
        version=SGLANG_VERSION,
        source_commit=SGLANG_SOURCE_COMMIT,
        source_tree=SGLANG_SOURCE_TREE,
        entry_seam=SGLANG_EXTRACTION_SEAM,
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
    from sglang.srt.models import qwen3_5

    architecture = architectures[0]
    entry_names = tuple(item.__name__ for item in qwen3_5.EntryClass)
    if (
        architecture not in entry_names
        or qwen3_5.Qwen3_5GatedDeltaNet.__name__ != "Qwen3_5GatedDeltaNet"
        or qwen3_5.RadixLinearAttention.__name__ != "RadixLinearAttention"
    ):
        raise RuntimeError("SGLang Qwen3.5 architecture binding does not match the pin")
    text = model_config.hf_text_config
    layer_map = {
        "attention": "full_attention",
        "linear_attention": "linear_attention",
    }
    try:
        layer_types = tuple(layer_map[value] for value in text.layers_block_type)
    except KeyError as error:
        raise RuntimeError(
            f"SGLang Qwen3.5 config has unknown layer type {error.args[0]!r}"
        ) from error
    return FrameworkConfigurationProjection(
        framework=framework,
        configuration_seam=SGLANG_CONFIGURATION_SEAM,
        architecture_binding=SGLANG_QWEN35_BINDING,
        text_implementation=SGLANG_QWEN35_IMPLEMENTATION,
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
            layer_types=layer_types,
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


def _configuration(
    checkpoint_root: Path,
) -> tuple[Any, FrameworkIdentity, FrameworkConfigurationProjection | None]:
    if os.environ.get("SIMLLM_SGLANG_ENABLE") != "1":
        raise RuntimeError("SGLang extraction requires SIMLLM_SGLANG_ENABLE=1")
    distribution = importlib.metadata.distribution("sglang")
    if distribution.version != SGLANG_VERSION:
        raise RuntimeError(
            f"SGLang extraction requires version {SGLANG_VERSION}, "
            f"found {distribution.version}"
        )
    repository = _source_root(distribution)
    if _git_object(repository, "HEAD") != SGLANG_SOURCE_COMMIT:
        raise RuntimeError("SGLang source commit does not match the extraction pin")
    if _git_object(repository, "HEAD^{tree}") != SGLANG_SOURCE_TREE:
        raise RuntimeError("SGLang source tree does not match the extraction pin")

    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.model_config import ModelConfig

    from simllm.adapters.sglang.worker import model_dims_from_sglang

    device = DeviceConfig(device="cpu")
    if device.device_type != "cpu":
        raise RuntimeError("SGLang did not preserve the requested CPU config device")
    model_config = ModelConfig(
        model_path=str(checkpoint_root),
        revision=checkpoint_root.name,
        trust_remote_code=False,
        dtype="bfloat16",
        enable_multimodal=False,
    )
    dims = model_dims_from_sglang(model_config)
    framework = _framework_identity()
    return dims, framework, _qwen35_projection(model_config, dims, framework)


def inspect_configuration(checkpoint_root: Path) -> dict[str, Any]:
    """Return the pinned Qwen text-stack projection without loading weights."""

    _, _, projection = _configuration(checkpoint_root)
    if projection is None:
        raise RuntimeError("SGLang configuration is not the pinned Qwen3.5 wrapper")
    return projection.to_obj()


def extract(
    *,
    suite_raw: bytes,
    checkpoint_root: Path,
    step_records_path: Path,
) -> ModelKernelInventory:
    """Extract through SGLang's explicit CPU device and model config path."""

    dims, framework, projection = _configuration(checkpoint_root)
    return extract_model_inventory(
        suite_raw=suite_raw,
        framework=framework,
        checkpoint_root=checkpoint_root,
        framework_dims=dims,
        step_records_path=step_records_path,
        framework_projection=projection,
    )


__all__ = [
    "SGLANG_CONFIGURATION_SEAM",
    "SGLANG_EXTRACTION_SEAM",
    "SGLANG_QWEN35_BINDING",
    "SGLANG_QWEN35_IMPLEMENTATION",
    "SGLANG_SOURCE_COMMIT",
    "SGLANG_SOURCE_TREE",
    "SGLANG_VERSION",
    "extract",
    "inspect_configuration",
]
