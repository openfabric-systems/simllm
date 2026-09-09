"""Emit source and import evidence from the process that extracts a K3 inventory."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from examples.kimi_k3_structure_v1.run_study import ROOT, SUITE_ID, digest
from simllm.calibration.canonical import canonical_bytes
from simllm.calibration.store import ObjectStore

HERE = Path(__file__).resolve().parent
SOURCE_PATHS = {
    "vllm-nvidia-model.py": "models/kimi_k3/nvidia/model.py",
    "vllm-nvidia-kda.py": "models/kimi_k3/nvidia/kda.py",
    "vllm-nvidia-mla.py": "models/kimi_k3/nvidia/mla.py",
    "vllm-linear-config.py": "transformers_utils/configs/kimi_linear.py",
    "vllm-kda-fused_recurrent.py": "models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py",
    "vllm-ops-attn_res.py": "models/kimi_k3/nvidia/ops/attn_res.py",
    "vllm-runner-latent_moe_runner.py": "model_executor/layers/fused_moe/runner/latent_moe_runner.py",
    "vllm-gpu-model-runner.py": "v1/worker/gpu_model_runner.py",
    "sglang-kimi-k3.py": "srt/models/kimi_k3.py",
    "sglang-mamba-config.py": "srt/configs/mamba_utils.py",
    "sglang-residual.py": "srt/layers/attn_residual.py",
}


def source_snapshot(package: Path, framework: str) -> list[dict]:
    rows = []
    for name, relative in SOURCE_PATHS.items():
        if not name.startswith(framework + "-"):
            continue
        path = (package / relative).resolve(strict=True)
        if not path.is_relative_to(package):
            raise ValueError("semantic source escapes installed package")
        rows.append({"name": name, "file": str(path), "sha256": digest(path.read_bytes())})
    return rows


def import_origins(framework: str, required: dict) -> list[dict]:
    rows = []
    for name, module in sorted(sys.modules.items()):
        if not name.startswith(framework + ".") or not (
            name in required or "kimi" in name or name.endswith("model_config")
        ):
            continue
        raw = getattr(module, "__file__", None)
        if raw is not None:
            path = Path(raw).resolve(strict=True)
            rows.append({"module": name, "file": str(path), "sha256": digest(path.read_bytes())})
    return rows


def capture(framework: str, checkpoint: Path, output: Path) -> dict:
    frozen = json.loads((HERE / "expectations.json").read_bytes())
    suite = ROOT / "offline/calibration/suites" / SUITE_ID / "suite.json"
    config = checkpoint / "config.json"
    module = importlib.import_module(f"simllm.adapters.{framework}.extraction")
    distribution = importlib.metadata.distribution(framework)
    package = (
        Path(distribution.locate_file("vllm"))
        if framework == "vllm"
        else module._source_root(distribution) / "python/sglang"
    ).resolve(strict=True)
    before = source_snapshot(package, framework)
    inputs_before = {"suite": digest(suite.read_bytes()), "config": digest(config.read_bytes())}
    projection = module.inspect_configuration(checkpoint)
    inventory = module.extract(
        suite_raw=suite.read_bytes(), checkpoint_root=checkpoint,
        step_records_path=output / "steps.jsonl",
    )
    record = ObjectStore(output / "objects").write(inventory.record)
    after = source_snapshot(package, framework)
    inputs_after = {"suite": digest(suite.read_bytes()), "config": digest(config.read_bytes())}
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None)
    proof = {
        "schema": "simllm-kimi-k3-native-source-proof-v1",
        "pid": os.getpid(), "interpreter": sys.executable,
        "framework": inventory.framework.to_obj(),
        "installed_version": distribution.version,
        "package": str(package), "projection": projection,
        "inputs_before": inputs_before, "inputs_after": inputs_after,
        "sources_before": before, "sources_after": after,
        "origins": import_origins(framework, frozen["required_import_origins"][framework]),
        "inventory_sha256": record.record_id,
        "steps_sha256": digest((output / "steps.jsonl").read_bytes()),
        "gpu_initialized": False if cuda is None else cuda.is_initialized(),
    }
    raw = canonical_bytes(proof)
    (output / "native-proof.json").write_bytes(raw)
    return {
        "record_sha256": record.record_id, "proof_sha256": digest(raw),
        "pid": os.getpid(), "steps_sha256": proof["steps_sha256"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(capture(args.framework, args.checkpoint_root, args.output_root)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
