"""Run and score the frozen CPU-only model extraction study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from simllm.calibration.canonical import (
    canonical_bytes,
    canonical_sha256,
    strict_json_loads,
)
from simllm.calibration.model_inventory import (
    ABSENT_BY_DESIGN,
    ModelKernelInventory,
)
from simllm.compute import ModelDims, step_kernel, step_kernels
from simllm.core import RequestPhase, ScheduledRequest, StepRecord, step_record_to_json

REPOSITORY = Path(__file__).resolve().parents[2]
EXPECTATIONS = Path(__file__).with_name("expectations.json")
FREEZE_COMMIT = "d5ec23ed13380df6e2fafbb2267494c55fc64380"
FAMILIES = ("attn_gemm", "attn_score", "mlp_gemm", "lm_head", "kv_read")
SHAPE_AXES = {
    "attn_gemm": ("new_tokens",),
    "attn_score": ("new_tokens", "kv_tokens"),
    "mlp_gemm": ("new_tokens",),
    "lm_head": ("sampled",),
    "kv_read": ("kv_tokens",),
}


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
    (output_root / f"{name}.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (output_root / f"{name}.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    return completed


def _environment(framework: str, *, skeleton: bool = True) -> dict[str, str]:
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
        if skeleton:
            environment["SIMLLM_VLLM_WORKER_MODE"] = "skeleton"
        else:
            environment.pop("SIMLLM_VLLM_WORKER_MODE", None)
    else:
        environment["SIMLLM_SGLANG_ENABLE"] = "1"
    return environment


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
        "--suite-root",
        str(suite_root),
        "--checkpoint-root",
        str(checkpoint_root),
        "--step-records",
        str(run_root / "steps.jsonl"),
        "--output-root",
        str(run_root / "objects"),
    ]


def _parse_report(completed: subprocess.CompletedProcess[str], name: str) -> dict[str, Any]:
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with status {completed.returncode}")
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if len(lines) != 1:
        raise RuntimeError(f"{name} did not emit exactly one JSON report")
    value = strict_json_loads(lines[0].encode())
    if not isinstance(value, dict):
        raise TypeError(f"{name} report is not an object")
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


def _step_record(cell: dict[str, Any], ordinal: int) -> StepRecord:
    family = cell["family"]
    case_id = cell["id"]
    if family == "compute-prefill":
        count = cell["requests"]
        phase = RequestPhase.PREFILL
        contexts = (cell["prompt_tokens_per_request"],) * count
        new_tokens = contexts
    else:
        count = cell["batch"]
        phase = RequestPhase.DECODE
        contexts = (cell["context_tokens"],) * count
        new_tokens = (cell["new_tokens_per_request"],) * count
    scheduled = [
        ScheduledRequest(
            request_id=f"{case_id}:request:{index}",
            phase=phase,
            num_new_tokens=new_tokens[index],
            context_length=contexts[index],
        )
        for index in range(count)
    ]
    return StepRecord(
        step_index=ordinal,
        virtual_time_ps=0,
        scheduled=scheduled,
        num_sampled=count,
        num_tokens_after_padding=sum(new_tokens),
        sampled_request_ids=[request.request_id for request in scheduled],
    )


def _dims(inventory: ModelKernelInventory) -> ModelDims:
    geometry = inventory.model.geometry
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
        moe_intermediate_size=geometry.intermediate_size,
        local_num_experts=geometry.num_experts,
    )


def _integer_work(value: float) -> int:
    if not isinstance(value, float) or not value.is_integer() or value < 0:
        raise ValueError("projected work is not an exact nonnegative integer")
    return int(value)


def _relation_one(
    inventory: ModelKernelInventory,
    cells: list[dict[str, Any]],
) -> tuple[bool, dict[str, int]]:
    base_dims = _dims(inventory)
    checked_families = 0
    for ordinal, (cell, case) in enumerate(zip(cells, inventory.cases, strict=True)):
        record = _step_record(cell, ordinal)
        case_dims = base_dims
        if cell["family"] == "moe-communication-decode":
            case_dims = replace(
                base_dims,
                local_num_experts=(
                    base_dims.num_experts // cell["expert_participants"]
                ),
            )
        specs = tuple(step_kernels(case_dims, record, record.num_sampled or 0))
        fused = step_kernel(case_dims, record, record.num_sampled or 0)
        if tuple(spec.name for spec in specs) != FAMILIES:
            return False, {"cases": ordinal, "families": checked_families}
        if case.case_id != cell["id"]:
            return False, {"cases": ordinal, "families": checked_families}
        if case.suite_case_sha256 != canonical_sha256(cell):
            return False, {"cases": ordinal, "families": checked_families}
        if case.step_record_sha256 != canonical_sha256(step_record_to_json(record)):
            return False, {"cases": ordinal, "families": checked_families}
        flops = 0
        hbm_bytes = 0
        for spec, projection in zip(specs, case.kernel_projections, strict=True):
            config = dict(spec.config)
            expected_shape = tuple(config[axis] for axis in SHAPE_AXES[spec.name])
            expected_flops = _integer_work(spec.flops)
            expected_bytes = _integer_work(spec.bytes_moved)
            if (
                projection.family_id != spec.name
                or projection.shape_vector.values != expected_shape
                or projection.aggregate_flops != expected_flops
                or projection.aggregate_hbm_bytes != expected_bytes
            ):
                return False, {"cases": ordinal, "families": checked_families}
            flops += expected_flops
            hbm_bytes += expected_bytes
            checked_families += 1
        if flops != _integer_work(fused.flops):
            return False, {"cases": ordinal, "families": checked_families}
        if hbm_bytes != _integer_work(fused.bytes_moved):
            return False, {"cases": ordinal, "families": checked_families}
    return True, {"cases": len(cells), "families": checked_families}


def _relation_two(inventory: ModelKernelInventory) -> tuple[bool, dict[str, int]]:
    layers = inventory.model.geometry.layers
    repeated = {"attn_gemm", "attn_score", "mlp_gemm", "kv_read"}
    for family in inventory.kernel_families:
        expected = layers if family.family_id in repeated else 1
        if any(item.logical_launch_count != expected for item in family.phase_launch_counts):
            return False, {"layers": layers, "launches_per_case": -1}
    totals = {
        sum(projection.logical_launch_count for projection in case.kernel_projections)
        for case in inventory.cases
    }
    expected_total = 4 * layers + 1
    return totals == {expected_total}, {
        "layers": layers,
        "launches_per_case": expected_total,
    }


def _relation_three(
    inventory: ModelKernelInventory,
    cells: list[dict[str, Any]],
) -> tuple[bool, dict[str, list[int]]]:
    observed = {
        "prefill_prompt_tokens": [],
        "decode_context_tokens": [],
        "moe_decode_batch": [],
    }
    for cell, case in zip(cells, inventory.cases, strict=True):
        projection = {item.family_id: item for item in case.kernel_projections}
        if cell["family"] == "compute-prefill":
            total = cell["total_prompt_tokens"]
            exact = (
                projection["attn_gemm"].shape_vector.values == (total,)
                and projection["attn_score"].shape_vector.values == (total, total)
                and projection["kv_read"].shape_vector.values == (total,)
            )
            observed["prefill_prompt_tokens"].append(
                cell["prompt_tokens_per_request"]
            )
        else:
            batch = cell["batch"]
            kv_tokens = batch * cell["context_tokens"]
            exact = (
                projection["attn_gemm"].shape_vector.values == (batch,)
                and projection["attn_score"].shape_vector.values
                == (batch, kv_tokens)
                and projection["lm_head"].shape_vector.values == (batch,)
                and projection["kv_read"].shape_vector.values == (kv_tokens,)
            )
            key = (
                "moe_decode_batch"
                if cell["family"] == "moe-communication-decode"
                else "decode_context_tokens"
            )
            observed[key].append(
                batch if key == "moe_decode_batch" else cell["context_tokens"]
            )
        if not exact:
            return False, observed
    exact_sweeps = {
        "prefill_prompt_tokens": [32, 192, 512, 128, 256],
        "decode_context_tokens": [128, 1024, 8192, 512, 2048],
        "moe_decode_batch": [1, 16, 64, 4, 8],
    }
    return observed == exact_sweeps, observed


def _relation_four(inventory: ModelKernelInventory) -> tuple[bool, dict[str, int]]:
    single = {case.template_graph_sha256 for case in inventory.cases[:10]}
    moe = {case.template_graph_sha256 for case in inventory.cases[10:]}
    return len(single) == 1 and len(moe) == 1 and single.isdisjoint(moe), {
        "single_rank_templates": len(single),
        "moe_templates": len(moe),
        "total_templates": len(single | moe),
    }


def _without_framework(value: dict[str, Any]) -> dict[str, Any]:
    copy = json.loads(json.dumps(value))
    copy.pop("framework")
    copy["implementation_identity"].pop("join_tasks")
    return copy


def _guard_values(
    inventories: dict[str, ModelKernelInventory],
    expectations: dict[str, Any],
    suite_raw: bytes,
    ordinary_imports: dict[str, bool],
) -> dict[str, bool]:
    expected_model = expectations["model"]
    expected_frameworks = {item["id"]: item for item in expectations["frameworks"]}
    guards: dict[str, bool] = {
        "suite-bytes-changed": _sha256(suite_raw) == expectations["suite"]["sha256"],
        "checkpoint-revision-config-weight-or-size-mismatch": True,
        "framework-version-or-source-identity-mismatch": True,
        "missing-or-duplicate-suite-case": True,
        "unknown-or-partial-kernel-family-projection": True,
        "nonintegral-or-negative-projected-work": True,
        "physical-identity-field-not-absent-by-design": True,
        "ordinary-simllm-import-loads-framework-runtime": all(
            ordinary_imports.values()
        ),
    }
    for framework, inventory in inventories.items():
        model = inventory.model
        guards["checkpoint-revision-config-weight-or-size-mismatch"] &= (
            model.name == expected_model["name"]
            and model.revision == expected_model["revision"]
            and model.config_sha256 == expected_model["config_sha256"]
            and model.weight_sha256 == expected_model["weight_sha256"]
            and model.weight_bytes == expected_model["weight_bytes"]
        )
        expected_framework = expected_frameworks[framework]
        guards["framework-version-or-source-identity-mismatch"] &= (
            inventory.framework.version == expected_framework["version"]
            and inventory.framework.source_commit
            == expected_framework["source_commit"]
            and inventory.framework.source_tree == expected_framework["source_tree"]
            and inventory.framework.entry_seam == expected_framework["entry_seam"]
        )
        case_ids = [case.case_id for case in inventory.cases]
        guards["missing-or-duplicate-suite-case"] &= (
            len(case_ids) == expectations["suite"]["case_count"]
            and len(case_ids) == len(set(case_ids))
        )
        guards["unknown-or-partial-kernel-family-projection"] &= all(
            tuple(item.family_id for item in case.kernel_projections) == FAMILIES
            for case in inventory.cases
        )
        guards["nonintegral-or-negative-projected-work"] &= all(
            type(value) is int and value >= 0
            for case in inventory.cases
            for projection in case.kernel_projections
            for value in (projection.aggregate_flops, projection.aggregate_hbm_bytes)
        )
        envelope = inventory.implementation_identity
        guards["physical-identity-field-not-absent-by-design"] &= all(
            marker.state == ABSENT_BY_DESIGN and marker.value is None
            for marker in (envelope.code_object_hashes, envelope.observed_launches)
        )
    return guards


def main(argv: list[str] | None = None) -> int:
    arguments = _arguments(argv)
    if arguments.output_root.exists():
        raise RuntimeError("output root already exists; choose a fresh study directory")
    arguments.output_root.mkdir(parents=True)
    expectations_raw = EXPECTATIONS.read_bytes()
    expectations = strict_json_loads(expectations_raw)
    suite_file = arguments.suite_root / "suites" / "transformer-dag-v1" / "suite.json"
    suite_raw = suite_file.read_bytes()
    suite = strict_json_loads(suite_raw)
    cells = suite["graph_cells"]

    framework_pythons = {
        "vllm": arguments.vllm_python,
        "sglang": arguments.sglang_python,
    }
    reports: dict[str, list[dict[str, Any]]] = {}
    inventories: dict[str, ModelKernelInventory] = {}
    repeat_bytes: dict[str, tuple[bytes, bytes]] = {}
    ordinary_imports: dict[str, bool] = {}
    for framework, python in framework_pythons.items():
        import_check = _run(
            [
                str(python),
                "-c",
                (
                    "import sys; import simllm; "
                    "assert 'vllm' not in sys.modules; "
                    "assert 'sglang' not in sys.modules"
                ),
            ],
            environment=_environment(framework),
            output_root=arguments.output_root,
            name=f"{framework}-ordinary-import",
        )
        ordinary_imports[framework] = import_check.returncode == 0
        reports[framework] = []
        raw_pair = []
        for repetition in (1, 2):
            run_root = arguments.output_root / f"{framework}-{repetition}"
            completed = _run(
                _extract_command(
                    python,
                    framework,
                    arguments.suite_root,
                    arguments.checkpoint_root,
                    run_root,
                ),
                environment=_environment(framework),
                output_root=arguments.output_root,
                name=f"{framework}-{repetition}",
            )
            report = _parse_report(completed, f"{framework}-{repetition}")
            reports[framework].append(report)
            object_path = run_root / "objects" / f"{report['record_sha256']}.json"
            raw_pair.append(object_path.read_bytes())
            if repetition == 1:
                inventories[framework] = _load_inventory(object_path)
        repeat_bytes[framework] = (raw_pair[0], raw_pair[1])

    negative_root = arguments.output_root / "vllm-unflagged-negative"
    negative = _run(
        _extract_command(
            arguments.vllm_python,
            "vllm",
            arguments.suite_root,
            arguments.checkpoint_root,
            negative_root,
        ),
        environment=_environment("vllm", skeleton=False),
        output_root=arguments.output_root,
        name="vllm-unflagged-negative",
    )
    unflagged_rejected = (
        negative.returncode == 2
        and not (negative_root / "objects").exists()
        and "SIMLLM_VLLM_WORKER_MODE=skeleton" in negative.stderr
    )

    guards = _guard_values(inventories, expectations, suite_raw, ordinary_imports)
    r1 = {framework: _relation_one(inventory, cells) for framework, inventory in inventories.items()}
    r2 = {framework: _relation_two(inventory) for framework, inventory in inventories.items()}
    r3 = {framework: _relation_three(inventory, cells) for framework, inventory in inventories.items()}
    r4 = {framework: _relation_four(inventory) for framework, inventory in inventories.items()}
    r5 = all(first == second for first, second in repeat_bytes.values()) and (
        repeat_bytes["vllm"][0] != repeat_bytes["sglang"][0]
    )
    structural_match = _without_framework(inventories["vllm"].to_obj()) == (
        _without_framework(inventories["sglang"].to_obj())
    )
    relations = [
        {
            "id": "R1-exact-family-projection",
            "passed": all(item[0] for item in r1.values()),
            "framework_evidence": {key: value[1] for key, value in r1.items()},
        },
        {
            "id": "R2-layer-launch-scaling",
            "passed": all(item[0] for item in r2.values()),
            "framework_evidence": {key: value[1] for key, value in r2.items()},
        },
        {
            "id": "R3-shape-axis-sensitivity",
            "passed": all(item[0] for item in r3.values()),
            "framework_evidence": {key: value[1] for key, value in r3.items()},
        },
        {
            "id": "R4-template-equivalence-classes",
            "passed": all(item[0] for item in r4.values()),
            "framework_evidence": {key: value[1] for key, value in r4.items()},
        },
    ]
    fatal_violations = sorted(name for name, passed in guards.items() if not passed)
    all_relations_pass = all(item["passed"] for item in relations)
    state = (
        "void"
        if fatal_violations
        else "passed"
        if all_relations_pass and r5 and structural_match and unflagged_rejected
        else "failed"
    )
    result = {
        "schema": "simllm-model-extraction-study-result-v1",
        "study": "model-extraction-v1",
        "state": state,
        "freeze_commit": FREEZE_COMMIT,
        "expectations_sha256": _sha256(expectations_raw),
        "suite_sha256": _sha256(suite_raw),
        "complete_framework_inventories": len(inventories),
        "inventories": [
            {
                "framework": framework,
                "record_sha256": inventory.record.record_id,
                "size_bytes": len(inventory.record.canonical),
                "case_count": len(inventory.cases),
            }
            for framework, inventory in inventories.items()
        ],
        "run_configuration": {
            "checkpoint": inventories["vllm"].model.to_obj(),
            "frameworks": [
                inventories[framework].framework.to_obj()
                for framework in framework_pythons
            ],
            "repetitions_per_framework": 2,
            "gpu_execution": "none",
        },
        "fatal_guards": {
            "violations": fatal_violations,
            "checks": guards,
        },
        "behavioral_relations": relations,
        "exact_oracles": [
            {
                "id": "R5-byte-determinism",
                "passed": r5,
                "same_framework_repeat_pairs": 2,
                "cross_framework_records_differ": (
                    repeat_bytes["vllm"][0] != repeat_bytes["sglang"][0]
                ),
            }
        ],
        "structural_invariants": [
            {
                "id": "cross-framework-denominators-identical",
                "passed": structural_match,
            },
            {
                "id": "unflagged-vllm-skeleton-rejects-without-object",
                "passed": unflagged_rejected,
            },
        ],
        "physical_sanity": {
            "timing_or_rate_measurements": "none",
            "logical_launches_per_case": 97,
            "logical_count_bound": "exactly-four-per-layer-plus-one-per-step",
            "independent_angles": [
                "framework-config-geometry-against-checkpoint-suite",
                "step-family-work-conservation-against-fused-step",
                "normalized-graph-topology-equivalence-classes",
            ],
            "physical_identity": "absent-by-design",
        },
    }
    (arguments.output_root / "results.json").write_bytes(canonical_bytes(result))
    sys.stdout.buffer.write(canonical_bytes(result))
    sys.stdout.buffer.write(b"\n")
    return 0 if state == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
