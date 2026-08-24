"""CPU-engine SGLang driver for the model kernel inventory."""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

from simllm.calibration.extraction import extract_model_inventory
from simllm.calibration.model_inventory import FrameworkIdentity, ModelKernelInventory

SGLANG_VERSION = "0.0.0.dev1+g8f2a3ad6d"
SGLANG_SOURCE_COMMIT = "8f2a3ad6d7d68c58ae65b61a75bb2115449addca"
SGLANG_SOURCE_TREE = "5be26db1f559064c0f9e724e78c1a8f619754867"
SGLANG_EXTRACTION_SEAM = "cpu-engine-step-record-v1"


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


def extract(
    *,
    suite_raw: bytes,
    checkpoint_root: Path,
    step_records_path: Path,
) -> ModelKernelInventory:
    """Extract through SGLang's explicit CPU device and model config path."""

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
        raise RuntimeError("SGLang did not preserve the requested CPU engine device")
    model_config = ModelConfig(
        model_path=str(checkpoint_root),
        revision=checkpoint_root.name,
        trust_remote_code=False,
        dtype="bfloat16",
        enable_multimodal=False,
    )
    return extract_model_inventory(
        suite_raw=suite_raw,
        framework=FrameworkIdentity(
            framework_id="sglang",
            version=distribution.version,
            source_commit=SGLANG_SOURCE_COMMIT,
            source_tree=SGLANG_SOURCE_TREE,
            entry_seam=SGLANG_EXTRACTION_SEAM,
        ),
        checkpoint_root=checkpoint_root,
        framework_dims=model_dims_from_sglang(model_config),
        step_records_path=step_records_path,
    )


__all__ = [
    "SGLANG_EXTRACTION_SEAM",
    "SGLANG_SOURCE_COMMIT",
    "SGLANG_SOURCE_TREE",
    "SGLANG_VERSION",
    "extract",
]
