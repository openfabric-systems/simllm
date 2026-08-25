"""CPU-only vLLM driver for the model kernel inventory."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path

from simllm.adapters.vllm._version import PINNED_VLLM_VERSION
from simllm.calibration.extraction import extract_model_inventory
from simllm.calibration.model_inventory import FrameworkIdentity, ModelKernelInventory

VLLM_SOURCE_COMMIT = "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac"
VLLM_EXTRACTION_SEAM = "flagged-skeleton-step-record-v1"


def extract(
    *,
    suite_raw: bytes,
    checkpoint_root: Path,
    step_records_path: Path,
) -> ModelKernelInventory:
    """Extract through vLLM's flagged skeleton configuration boundary."""

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
    config = VllmConfig(
        model_config=model_config,
        parallel_config=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
    )
    return extract_model_inventory(
        suite_raw=suite_raw,
        framework=FrameworkIdentity(
            framework_id="vllm",
            version=version,
            source_commit=VLLM_SOURCE_COMMIT,
            source_tree=None,
            entry_seam=VLLM_EXTRACTION_SEAM,
        ),
        checkpoint_root=checkpoint_root,
        framework_dims=model_dims_from_vllm_config(config),
        step_records_path=step_records_path,
    )


__all__ = ["VLLM_EXTRACTION_SEAM", "VLLM_SOURCE_COMMIT", "extract"]
