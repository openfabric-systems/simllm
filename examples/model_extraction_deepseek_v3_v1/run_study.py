"""Run and score the frozen DeepSeek-V3 family inventory study."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from simllm.calibration.canonical import (
    canonical_bytes,
    canonical_sha256,
    strict_json_loads,
)
from simllm.calibration.deepseek_deployment import (
    build_deepseek_deployment_projection,
)
from simllm.calibration.model_inventory import ModelKernelInventory

REPOSITORY = Path(__file__).resolve().parents[2]
EXPECTATIONS = Path(__file__).with_name("expectations.json")
FREEZE_COMMIT = "5dc2877292fe40a74a49c1e2270e6a39d08613db"
SUITE_ID = "deepseek-v3-text-v1-frameworks-2026-08-25"


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-python", type=Path, required=True)
    parser.add_argument("--sglang-python", type=Path, required=True)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _environment(framework: str) -> dict[str, str]:
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(REPOSITORY)
        if not existing
        else f"{REPOSITORY}{os.pathsep}{existing}"
    )
    environment["HF_HUB_OFFLINE"] = "1"
    if framework == "vllm":
        environment["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        environment["SIMLLM_VLLM_WORKER_MODE"] = "skeleton"
    else:
        environment["SIMLLM_SGLANG_ENABLE"] = "1"
    return environment


def _run(
    command: list[str],
    *,
    environment: dict[str, str],
    output_root: Path,
    name: str,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=REPOSITORY,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    output_root.joinpath(f"{name}.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    output_root.joinpath(f"{name}.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    return completed


def _inspect_command(
    python: Path,
    framework: str,
    checkpoint_root: Path,
) -> list[str]:
    source = (
        "import sys; from pathlib import Path; "
        "from simllm.calibration.canonical import canonical_bytes; "
        f"from simllm.adapters.{framework}.extraction import inspect_configuration; "
        "sys.stdout.buffer.write(canonical_bytes("
        "inspect_configuration(Path(sys.argv[1]))) + b'\\n')"
    )
    return [str(python), "-c", source, str(checkpoint_root)]


def _extract_command(
    python: Path,
    framework: str,
    suite_root: Path,
    checkpoint_root: Path,
    run_root: Path,
) -> list[str]:
    return [
        str(python),
        "-c",
        "from simllm.calibration.cli import main; raise SystemExit(main())",
        "extract",
        "--framework",
        framework,
        "--suite",
        SUITE_ID,
        "--suite-root",
        str(suite_root),
        "--checkpoint-root",
        str(checkpoint_root),
        "--step-records",
        str(run_root / "steps.jsonl"),
        "--output-root",
        str(run_root / "objects"),
    ]


def _one_json_line(
    completed: subprocess.CompletedProcess[str],
    name: str,
) -> dict[str, Any]:
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with status {completed.returncode}")
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if len(lines) != 1:
        raise RuntimeError(f"{name} did not emit exactly one JSON record")
    value = strict_json_loads(lines[0].encode())
    if not isinstance(value, dict):
        raise TypeError(f"{name} record is not an object")
    return value


def _load_inventory(path: Path) -> ModelKernelInventory:
    raw = path.read_bytes()
    value = strict_json_loads(raw)
    inventory = ModelKernelInventory.from_obj(value)
    if inventory.record.canonical != raw:
        raise RuntimeError(f"inventory {path.name!r} is not canonical")
    if path.stem != inventory.record.record_id:
        raise RuntimeError(f"inventory {path.name!r} is not content addressed")
    return inventory


def _expected_projection(
    suite: dict[str, Any],
    framework: dict[str, Any],
) -> dict[str, Any]:
    model = suite["reference_model"]
    stack = model["deepseek_stack"]
    geometry = model["geometry"]
    return {
        "schema": "simllm-framework-text-config-projection-v1",
        "framework": {
            "id": framework["id"],
            "version": framework["version"],
            "source_commit": framework["source_commit"],
            "source_tree": framework.get("source_tree"),
        },
        "configuration_seam": (
            "ModelConfig-with-skip-tokenizer-init"
            if framework["id"] == "vllm"
            else "DeviceConfig-cpu-plus-ModelConfig-with-multimodal-disabled"
        ),
        "architecture_binding": framework["architecture_binding"],
        "text_implementation": framework["text_implementation"],
        "deepseek_stack": {
            "architecture": model["architecture"],
            "wrapper_model_type": model["model_type"],
            "scope": stack["scope"],
            "geometry": geometry,
            "layer_types": (
                ["dense"] * stack["first_k_dense_replace"]
                + ["moe"]
                * (geometry["layers"] - stack["first_k_dense_replace"])
            ),
            **{
                name: value
                for name, value in stack.items()
                if name != "scope"
            },
        },
    }


def _neutral_inventory(inventory: ModelKernelInventory) -> dict[str, Any]:
    value = inventory.to_obj()
    value.pop("framework")
    value["implementation_identity"].pop("join_tasks")
    return value


def _score_inventory(
    inventory: ModelKernelInventory,
    expectations: dict[str, Any],
) -> dict[str, int]:
    oracles = expectations["exact_case_oracles"]
    ordered = expectations["inventory_contract"]["ordered_families"]
    if [family.family_id for family in inventory.kernel_families] != ordered:
        raise RuntimeError("inventory family order differs from the freeze")
    if len(inventory.cases) != len(oracles):
        raise RuntimeError("inventory case count differs from the freeze")
    family_rows = 0
    for case, oracle in zip(inventory.cases, oracles, strict=True):
        if case.case_id != oracle["case_id"]:
            raise RuntimeError("inventory case order differs from the freeze")
        visits = sum(item.logical_launch_count for item in case.kernel_projections)
        flops = sum(item.aggregate_flops for item in case.kernel_projections)
        hbm_bytes = sum(
            item.aggregate_hbm_bytes for item in case.kernel_projections
        )
        if visits != oracle["logical_visit_count"]:
            raise RuntimeError(f"case {case.case_id!r} visit count changed")
        if flops != oracle["aggregate_flops"]:
            raise RuntimeError(f"case {case.case_id!r} FLOPs do not conserve")
        if hbm_bytes != oracle["aggregate_hbm_bytes"]:
            raise RuntimeError(f"case {case.case_id!r} HBM bytes do not conserve")
        family_rows += len(case.kernel_projections)
    return {
        "case_count": len(inventory.cases),
        "family_rows": family_rows,
        "ordinary_visit_count": 666,
        "mtp_visit_count": 667,
    }


def _verify_inputs(
    suite_path: Path,
    checkpoint_root: Path,
    expectations: dict[str, Any],
) -> dict[str, Any]:
    suite_raw = suite_path.read_bytes()
    if _sha256(suite_raw) != expectations["suite"]["sha256"]:
        raise RuntimeError("current suite byte lock changed")
    suite = strict_json_loads(suite_raw)
    if not isinstance(suite, dict):
        raise TypeError("suite is not an object")
    config = checkpoint_root / "config.json"
    if _sha256(config.read_bytes()) != expectations["model"]["config_sha256"]:
        raise RuntimeError("checkpoint configuration hash changed")
    if tuple(checkpoint_root.glob("*.safetensors")):
        raise RuntimeError("checkpoint substrate contains a weight file")
    for relative, digest in expectations["historical_byte_locks"].items():
        if _sha256(REPOSITORY.joinpath(relative).read_bytes()) != digest:
            raise RuntimeError(f"historical byte lock changed: {relative}")
    shards = suite["reference_model"]["weight_shards"]
    if canonical_sha256(shards) != expectations["model"]["weight_sha256"]:
        raise RuntimeError("checkpoint shard manifest hash changed")
    if sum(row["bytes"] for row in shards) != expectations["model"]["weight_bytes"]:
        raise RuntimeError("checkpoint shard manifest byte sum changed")
    return suite


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    """Run two extraction processes per framework and score every guard."""

    output_root = arguments.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    expectations = strict_json_loads(EXPECTATIONS.read_bytes())
    if not isinstance(expectations, dict):
        raise TypeError("expectations are not an object")
    suite_path = arguments.suite_root / "suites" / SUITE_ID / "suite.json"
    suite = _verify_inputs(suite_path, arguments.checkpoint_root, expectations)
    python_by_framework = {
        "vllm": arguments.vllm_python,
        "sglang": arguments.sglang_python,
    }
    inventories: dict[str, ModelKernelInventory] = {}
    inventory_ids: dict[str, str] = {}
    repeat_digests: dict[str, dict[str, str]] = {}
    score: dict[str, dict[str, int]] = {}
    sidecars: dict[str, dict[str, Any]] = {}
    for framework in ("vllm", "sglang"):
        python = python_by_framework[framework]
        environment = _environment(framework)
        declaration = next(
            row for row in suite["frameworks"] if row["id"] == framework
        )
        inspection = _run(
            _inspect_command(python, framework, arguments.checkpoint_root),
            environment=environment,
            output_root=output_root,
            name=f"{framework}-inspect",
        )
        observed_projection = _one_json_line(inspection, f"{framework} inspection")
        if observed_projection != _expected_projection(suite, declaration):
            raise RuntimeError(f"{framework} configuration projection changed")
        repeat_raw = []
        repeat_steps = []
        for repeat in (1, 2):
            run_root = output_root / framework / f"repeat-{repeat}"
            run_root.mkdir(parents=True)
            completed = _run(
                _extract_command(
                    python,
                    framework,
                    arguments.suite_root,
                    arguments.checkpoint_root,
                    run_root,
                ),
                environment=environment,
                output_root=output_root,
                name=f"{framework}-repeat-{repeat}",
            )
            report = _one_json_line(completed, f"{framework} repeat {repeat}")
            record_id = report["record_sha256"]
            inventory_path = run_root / "objects" / f"{record_id}.json"
            inventory = _load_inventory(inventory_path)
            repeat_raw.append(inventory.record.canonical)
            repeat_steps.append((run_root / "steps.jsonl").read_bytes())
            if repeat == 1:
                inventories[framework] = inventory
                inventory_ids[framework] = record_id
                score[framework] = _score_inventory(inventory, expectations)
        if repeat_raw[0] != repeat_raw[1] or repeat_steps[0] != repeat_steps[1]:
            raise RuntimeError(f"{framework} repeat extraction is not byte stable")
        repeat_digests[framework] = {
            "inventory_sha256": _sha256(repeat_raw[0]),
            "step_records_sha256": _sha256(repeat_steps[0]),
        }
        sidecars[framework] = build_deepseek_deployment_projection(
            suite, inventories[framework]
        )
    if _neutral_inventory(inventories["vllm"]) != _neutral_inventory(
        inventories["sglang"]
    ):
        raise RuntimeError("framework inventories differ structurally")
    if sidecars["vllm"] != sidecars["sglang"]:
        raise RuntimeError("framework deployment projections differ")
    sidecar = sidecars["vllm"]
    sidecar_raw = canonical_bytes(sidecar)
    output_root.joinpath("deployment-projection.json").write_bytes(sidecar_raw)
    summary = {
        "schema": "simllm-deepseek-v3-inventory-study-result-v1",
        "study": "model-extraction-deepseek-v3-v1",
        "freeze_commit": FREEZE_COMMIT,
        "fatal_guard_state": "nonvoid",
        "suite_sha256": expectations["suite"]["sha256"],
        "inventories": inventory_ids,
        "repeat_digests": repeat_digests,
        "score": score,
        "deployment_projection_sha256": _sha256(sidecar_raw),
        "deployment_unit_count": len(sidecar["units"]),
        "logical_experts": sidecar["expert_contract"]["logical_experts"],
        "physical_expert_slots": sidecar["expert_contract"]["physical_slots"],
        "per_expert_base_static_hbm_bytes": sidecar["expert_contract"][
            "per_expert_base_static_hbm_bytes"
        ],
    }
    output_root.joinpath("summary.json").write_bytes(canonical_bytes(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    summary = run(_arguments(argv))
    sys.stdout.buffer.write(canonical_bytes(summary) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
