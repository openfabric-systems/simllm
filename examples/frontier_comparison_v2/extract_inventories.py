"""Offline config-only replay through the existing inventory extractor.

Framework bindings are retained suite declarations, not fresh native-framework
observations. Geometry and FP8 layout are read from the pinned config.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from simllm.calibration.extraction import (
    FrameworkConfigurationProjection,
    FrameworkDenseStack,
    extract_model_inventory,
)
from simllm.calibration.model_inventory import (
    FrameworkIdentity,
    ModelGeometry,
    ModelKernelInventory,
)
from simllm.compute import ModelDims

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "offline/calibration/suites/qwen3-32b-fp8-text-v1-frameworks-2026-08-28/suite.json"
OLD_IDS = {
    "vllm": "c8832ba8ba21e49517b6b74e89554c2abdb0d9e76530f647a7849f3f8448ec56",
    "sglang": "51740b52625002a964e75fddb679e9f8394a08a7d7c62556d2535c3bc60515e3",
}
REGISTRY = ROOT / "offline/calibration/model-inventories"


def config_inputs(config: dict, declaration: dict, framework: FrameworkIdentity):
    """Read geometry independently of the suite's numerical expectations."""
    geometry = ModelGeometry(
        layers=config["num_hidden_layers"],
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        num_heads=config["num_attention_heads"],
        num_kv_heads=config["num_key_value_heads"],
        head_size=config["head_dim"],
        vocab_size=config["vocab_size"],
        num_experts=0,
        top_k=0,
    )
    quant = config["quantization_config"]
    if quant["quant_method"] != "fp8" or config["architectures"] != ["Qwen3ForCausalLM"]:
        raise ValueError("config must describe the frozen dense FP8 Qwen3 architecture")
    stack = FrameworkDenseStack(
        architecture=config["architectures"][0],
        model_type=config["model_type"],
        scope="text-only",
        geometry=geometry,
        attention_mechanism="grouped-query causal self-attention",
        quantization="fp8-e4m3-block-128x128",
        weight_block_size=tuple(quant["weight_block_size"]),
        excluded_components=("input-embedding-family", "normalization-family"),
    )
    dims = ModelDims(
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
    projection = FrameworkConfigurationProjection(
        framework=framework,
        configuration_seam="retained-suite-binding-with-local-config-replay",
        architecture_binding=declaration["architecture_binding"],
        text_implementation=declaration["text_implementation"],
        dense_stack=stack,
    )
    return dims, projection


def extract(checkpoint_root: Path, output_root: Path) -> dict[str, ModelKernelInventory]:
    """Write new objects and step records only to the requested bulk root."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    suite_raw = SUITE.read_bytes().replace(b"\r\n", b"\n")
    suite = json.loads(suite_raw)
    config = json.loads((checkpoint_root / "config.json").read_bytes())
    output_root.mkdir(parents=True, exist_ok=True)
    inventories = {}
    for declaration in suite["frameworks"]:
        name = declaration["id"]
        old = ModelKernelInventory.from_obj(
            json.loads((REGISTRY / f"{OLD_IDS[name]}.json").read_bytes())
        )
        dims, projection = config_inputs(config, declaration, old.framework)
        inventory = extract_model_inventory(
            suite_raw=suite_raw,
            framework=old.framework,
            checkpoint_root=checkpoint_root,
            framework_dims=dims,
            framework_projection=projection,
            step_records_path=output_root / f"{name}-steps.jsonl",
            attention_shape_version=2,
        )
        (output_root / f"{inventory.record.record_id}.json").write_bytes(inventory.record.canonical)
        inventories[name] = inventory
    return inventories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.resolve().is_relative_to(ROOT):
        parser.error("output-root must be outside the repository under SIMLLM_DATA_ROOT")
    repeats = [extract(args.checkpoint_root, args.output_root / f"repeat-{i}") for i in (1, 2)]
    receipt = {
        "mode": "offline-local-config-with-retained-framework-bindings",
        "config_sha256": next(iter(repeats[0].values())).model.config_sha256,
        "frameworks": {},
    }
    for name, inventory in repeats[0].items():
        if inventory.record.canonical != repeats[1][name].record.canonical:
            raise ValueError("repeat extraction did not reproduce canonical inventory bytes")
        receipt["frameworks"][name] = {
            "inventory_sha256": [repeat[name].record.record_id for repeat in repeats],
            "step_records_sha256": [
                hashlib.sha256(
                    (args.output_root / f"repeat-{i}" / f"{name}-steps.jsonl").read_bytes()
                ).hexdigest()
                for i in (1, 2)
            ],
        }
    raw = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    (args.output_root / "extraction.json").write_bytes(raw)
    print(raw.decode(), end="")


if __name__ == "__main__":
    main()
