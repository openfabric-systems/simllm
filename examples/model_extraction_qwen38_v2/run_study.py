"""Run and score the frozen Qwen3.8 Gated DeltaNet inventory study."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from simllm.calibration.extraction import case_records_from_suite
from simllm.calibration.model_inventory import (
    ABSENT_BY_DESIGN,
    ModelKernelInventory,
)
from simllm.core import step_record_to_json

REPOSITORY = Path(__file__).resolve().parents[2]
EXPECTATIONS = Path(__file__).with_name("expectations.json")
FREEZE_COMMIT = "bf14d7563bdb52a0c8052309f477a022f1951cc4"
SUITE_ID = "qwen3.8-27b-text-v1-frameworks-2026-08-25"


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


def _json_report(
    completed: subprocess.CompletedProcess[str], name: str
) -> dict[str, Any]:
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


def _expected_projection(
    suite: dict[str, Any],
    framework: dict[str, Any],
) -> dict[str, Any]:
    model = suite["reference_model"]
    text = model["text_stack"]
    return {
        "schema": "simllm-framework-text-config-projection-v1",
        "framework": {
            "id": framework["id"],
            "version": framework["version"],
            "source_commit": framework["source_commit"],
            "source_tree": framework["source_tree"],
        },
        "configuration_seam": framework["configuration_seam"],
        "architecture_binding": framework["architecture_binding"],
        "text_implementation": framework["text_implementation"],
        "text_stack": {
            "architecture": model["architecture"],
            "wrapper_model_type": model["model_type"],
            "text_model_type": text["model_type"],
            "scope": text["scope"],
            "geometry": model["geometry"],
            "layer_types": text["layer_pattern"] * text["pattern_repetitions"],
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
        },
    }


def _expected_family_work(
    family: dict[str, Any],
    oracle: dict[str, Any],
) -> tuple[int, int, tuple[int, ...]]:
    layers = family["layers"]
    tokens = oracle["new_tokens"]
    sequences = oracle["sequences"]
    kv_tokens = oracle["kv_tokens"]
    pairs = oracle["attention_pairs"]
    sampled = oracle["sampled"]
    shapes = {
        "new_tokens": tokens,
        "sequences": sequences,
        "kv_tokens": kv_tokens,
        "sampled": sampled,
    }
    flops = 0
    hbm_bytes = 0
    flops += family.get("flops_per_new_token_per_layer", 0) * tokens * layers
    flops += family.get("flops_per_attention_pair_per_layer", 0) * pairs * layers
    flops += family.get("flops_per_sampled_token", 0) * sampled
    hbm_bytes += family.get("static_hbm_bytes_per_layer", 0) * layers
    hbm_bytes += family.get("static_hbm_bytes", 0)
    hbm_bytes += (
        family.get("hbm_bytes_per_kv_token_per_layer", 0) * kv_tokens * layers
    )
    hbm_bytes += (
        family.get("hbm_bytes_per_sequence_per_layer", 0) * sequences * layers
    )
    hbm_bytes += (
        family.get("state_read_write_hbm_bytes_per_sequence_per_layer", 0)
        * sequences
        * layers
    )
    return flops, hbm_bytes, tuple(shapes[axis] for axis in family["shape_axes"])


def _family_relations(
    inventory: ModelKernelInventory,
    suite: dict[str, Any],
    expectations: dict[str, Any],
) -> tuple[bool, dict[str, int]]:
    contract = expectations["inventory_contract"]
    family_rows = contract["families"]
    family_by_id = {row["id"]: row for row in family_rows}
    ordered = tuple(contract["ordered_families"])
    records = case_records_from_suite(suite)
    exact_families = 0
    conserved_cases = 0
    if tuple(item.family_id for item in inventory.kernel_families) != ordered:
        return False, {"cases": 0, "families": 0, "conserved_cases": 0}
    schemas = {schema.shape_schema_id: schema for schema in inventory.shape_schemas}
    for family in inventory.kernel_families:
        row = family_by_id[family.family_id]
        schema = schemas[family.shape_schema_id]
        if tuple(axis.axis_id for axis in schema.axes) != tuple(row["shape_axes"]):
            return False, {
                "cases": conserved_cases,
                "families": exact_families,
                "conserved_cases": conserved_cases,
            }
        if any(
            item.logical_launch_count != row["layers"]
            for item in family.phase_launch_counts
        ):
            return False, {
                "cases": conserved_cases,
                "families": exact_families,
                "conserved_cases": conserved_cases,
            }
    for cell, record, case, oracle in zip(
        suite["graph_cells"],
        records,
        inventory.cases,
        expectations["exact_case_oracles"],
        strict=True,
    ):
        if (
            case.case_id != cell["id"]
            or case.case_id != oracle["case_id"]
            or case.family != cell["family"]
            or case.phase != cell["phase"]
            or case.split != cell["split"]
            or case.suite_case_sha256 != canonical_sha256(cell)
            or case.step_record_sha256
            != canonical_sha256(step_record_to_json(record))
            or tuple(item.family_id for item in case.kernel_projections) != ordered
        ):
            return False, {
                "cases": conserved_cases,
                "families": exact_families,
                "conserved_cases": conserved_cases,
            }
        projected_flops = 0
        projected_bytes = 0
        launches = 0
        for projection in case.kernel_projections:
            expected_flops, expected_bytes, expected_shape = _expected_family_work(
                family_by_id[projection.family_id], oracle
            )
            if (
                projection.aggregate_flops != expected_flops
                or projection.aggregate_hbm_bytes != expected_bytes
                or projection.shape_vector.values != expected_shape
                or any(
                    type(value) is not int or value < 0
                    for value in (
                        projection.aggregate_flops,
                        projection.aggregate_hbm_bytes,
                        *projection.shape_vector.values,
                    )
                )
            ):
                return False, {
                    "cases": conserved_cases,
                    "families": exact_families,
                    "conserved_cases": conserved_cases,
                }
            projected_flops += projection.aggregate_flops
            projected_bytes += projection.aggregate_hbm_bytes
            launches += projection.logical_launch_count
            exact_families += 1
        formula_flops = (
            contract["fixed_flops_per_new_token"] * oracle["new_tokens"]
            + contract["flops_per_attention_pair"] * oracle["attention_pairs"]
            + contract["flops_per_sampled_token"] * oracle["sampled"]
        )
        formula_bytes = (
            contract["static_hbm_bytes"]
            + contract["hbm_bytes_per_sequence"] * oracle["sequences"]
            + contract["hbm_bytes_per_kv_token"] * oracle["kv_tokens"]
        )
        if (
            projected_flops != oracle["aggregate_flops"]
            or projected_flops != formula_flops
            or projected_bytes != oracle["aggregate_hbm_bytes"]
            or projected_bytes != formula_bytes
            or launches != contract["logical_launch_count_per_case"]
        ):
            return False, {
                "cases": conserved_cases,
                "families": exact_families,
                "conserved_cases": conserved_cases,
            }
        conserved_cases += 1
    return True, {
        "cases": len(inventory.cases),
        "families": exact_families,
        "conserved_cases": conserved_cases,
    }


def _schedule_relation(projection: dict[str, Any]) -> tuple[bool, dict[str, int]]:
    schedule = projection.get("text_stack", {}).get("layer_types", [])
    blocks = [schedule[index : index + 4] for index in range(0, len(schedule), 4)]
    expected = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    passed = (
        len(schedule) == 64
        and len(blocks) == 16
        and all(block == expected for block in blocks)
        and schedule.count("linear_attention") == 48
        and schedule.count("full_attention") == 16
    )
    return passed, {
        "layers": len(schedule),
        "linear_layers": schedule.count("linear_attention"),
        "full_layers": schedule.count("full_attention"),
        "exact_four_layer_blocks": sum(block == expected for block in blocks),
    }


def _shape_relation(
    inventory: ModelKernelInventory,
    suite: dict[str, Any],
) -> tuple[bool, dict[str, list[int]]]:
    observed = {
        "prefill_prompt_tokens": [],
        "memory_decode_context_tokens": [],
        "dense_decode_batch": [],
    }
    memory_state_bytes = []
    dense_state_bytes = []
    for cell, case in zip(suite["graph_cells"], inventory.cases, strict=True):
        families = {item.family_id: item for item in case.kernel_projections}
        if cell["family"] == "compute-prefill":
            tokens = cell["total_prompt_tokens"]
            observed["prefill_prompt_tokens"].append(
                cell["prompt_tokens_per_request"]
            )
            exact = (
                families["gdn_input_projection"].shape_vector.values == (tokens,)
                and families["gdn_short_convolution"].shape_vector.values
                == (tokens, cell["requests"])
            )
        else:
            batch = cell["batch"]
            kv_tokens = batch * cell["context_tokens"]
            exact = (
                families["attn_score"].shape_vector.values == (batch, kv_tokens)
                and families["kv_read"].shape_vector.values == (kv_tokens,)
                and families["gdn_state_read"].shape_vector.values == (batch,)
                and families["lm_head"].shape_vector.values == (batch,)
            )
            if cell["family"] == "memory-decode":
                observed["memory_decode_context_tokens"].append(
                    cell["context_tokens"]
                )
                memory_state_bytes.append(
                    families["gdn_state_read"].aggregate_hbm_bytes
                )
            else:
                observed["dense_decode_batch"].append(batch)
                dense_state_bytes.append(
                    families["gdn_state_read"].aggregate_hbm_bytes
                )
        if not exact:
            return False, observed
    expected = {
        "prefill_prompt_tokens": [32, 192, 512, 128, 256],
        "memory_decode_context_tokens": [128, 1024, 8192, 512, 2048],
        "dense_decode_batch": [1, 16, 64, 4, 8],
    }
    passed = (
        observed == expected
        and len(set(memory_state_bytes)) == 1
        and all(
            value == 150994944 * batch
            for value, batch in zip(
                dense_state_bytes,
                observed["dense_decode_batch"],
                strict=True,
            )
        )
    )
    return passed, observed


def _neutral_inventory(inventory: ModelKernelInventory) -> dict[str, Any]:
    value = json.loads(json.dumps(inventory.to_obj()))
    value.pop("framework")
    value["implementation_identity"].pop("join_tasks")
    return value


def _historical_locks(expectations: dict[str, Any]) -> bool:
    return all(
        (path := REPOSITORY / relative).is_file()
        and _sha256(path.read_bytes()) == digest
        for relative, digest in expectations["historical_byte_locks"].items()
    )


def main(argv: list[str] | None = None) -> int:
    arguments = _arguments(argv)
    if arguments.output_root.exists():
        raise RuntimeError("output root already exists; choose a fresh study directory")
    arguments.output_root.mkdir(parents=True)
    expectations_raw = EXPECTATIONS.read_bytes()
    expectations = strict_json_loads(expectations_raw)
    suite_file = arguments.suite_root / "suites" / SUITE_ID / "suite.json"
    suite_raw = suite_file.read_bytes()
    suite = strict_json_loads(suite_raw)
    framework_rows = {
        item["id"]: item for item in expectations["frameworks"]
    }
    framework_pythons = {
        "vllm": arguments.vllm_python,
        "sglang": arguments.sglang_python,
    }
    inspections: dict[str, dict[str, Any]] = {}
    inventories: dict[str, ModelKernelInventory] = {}
    inventory_bytes: dict[str, tuple[bytes, bytes]] = {}
    step_bytes: dict[str, tuple[bytes, bytes]] = {}
    ordinary_imports: dict[str, bool] = {}
    reports: dict[str, list[dict[str, Any]]] = {}
    for framework, python in framework_pythons.items():
        imported = _run(
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
        ordinary_imports[framework] = imported.returncode == 0
        inspected = _run(
            _inspect_command(python, framework, arguments.checkpoint_root),
            environment=_environment(framework),
            output_root=arguments.output_root,
            name=f"{framework}-configuration",
        )
        inspections[framework] = _json_report(
            inspected, f"{framework}-configuration"
        )
        raw_objects = []
        raw_steps = []
        reports[framework] = []
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
            report = _json_report(completed, f"{framework}-{repetition}")
            reports[framework].append(report)
            object_path = run_root / "objects" / f"{report['record_sha256']}.json"
            raw_objects.append(object_path.read_bytes())
            raw_steps.append((run_root / "steps.jsonl").read_bytes())
            if repetition == 1:
                inventories[framework] = _load_inventory(object_path)
        inventory_bytes[framework] = (raw_objects[0], raw_objects[1])
        step_bytes[framework] = (raw_steps[0], raw_steps[1])

    family_checks = {
        framework: _family_relations(inventory, suite, expectations)
        for framework, inventory in inventories.items()
    }
    schedule_checks = {
        framework: _schedule_relation(projection)
        for framework, projection in inspections.items()
    }
    shape_checks = {
        framework: _shape_relation(inventory, suite)
        for framework, inventory in inventories.items()
    }
    expected_projections = {
        framework: _expected_projection(suite, framework_rows[framework])
        for framework in framework_pythons
    }
    exact_projections = {
        framework: inspections[framework] == expected_projections[framework]
        for framework in framework_pythons
    }
    repeat_inventory = {
        framework: first == second
        for framework, (first, second) in inventory_bytes.items()
    }
    repeat_steps = {
        framework: first == second
        for framework, (first, second) in step_bytes.items()
    }
    structural_match = _neutral_inventory(inventories["vllm"]) == (
        _neutral_inventory(inventories["sglang"])
    )
    model = suite["reference_model"]
    checkpoint_config = arguments.checkpoint_root / "config.json"
    family_ids = tuple(expectations["inventory_contract"]["ordered_families"])
    checkpoint_identity = (
        arguments.checkpoint_root.name == expectations["model"]["revision"]
        and checkpoint_config.is_file()
        and _sha256(checkpoint_config.read_bytes())
        == expectations["model"]["config_sha256"]
        and model["parameter_count"] == expectations["model"]["parameter_count"]
        and model["weight_sha256"] == expectations["model"]["weight_sha256"]
        and model["weight_bytes"] == expectations["model"]["weight_bytes"]
        and len(model["weight_shards"]) == 18
        and sum(item["bytes"] for item in model["weight_shards"])
        == expectations["model"]["weight_bytes"]
        and canonical_sha256(model["weight_shards"])
        == expectations["model"]["weight_sha256"]
    )
    inventory_model_identity = all(
        inventory.model.to_obj()
        == {
            key: model[key]
            for key in (
                "name",
                "revision",
                "config_sha256",
                "weight_sha256",
                "weight_bytes",
                "dtype",
                "quantization",
                "geometry",
            )
        }
        for inventory in inventories.values()
    )
    case_ids = [cell["id"] for cell in suite["graph_cells"]]
    guards = {
        "current-suite-bytes-changed": (
            _sha256(suite_raw) == expectations["suite"]["sha256"]
        ),
        "historical-byte-lock-changed": _historical_locks(expectations),
        "checkpoint-api-manifest-revision-config-or-weight-identity-mismatch": (
            checkpoint_identity and inventory_model_identity
        ),
        "local-configuration-substrate-contains-a-weight-file": not any(
            arguments.checkpoint_root.rglob("*.safetensors")
        ),
        "framework-version-source-binding-or-text-implementation-mismatch": all(
            exact_projections.values()
        ),
        "framework-text-geometry-layer-order-mechanism-or-exclusion-mismatch": all(
            exact_projections.values()
        ),
        "missing-duplicate-or-reordered-suite-case": (
            len(case_ids) == expectations["suite"]["case_count"]
            and len(case_ids) == len(set(case_ids))
            and all(
                [case.case_id for case in inventory.cases] == case_ids
                for inventory in inventories.values()
            )
        ),
        "unknown-missing-duplicate-or-reordered-kernel-family": all(
            tuple(family.family_id for family in inventory.kernel_families)
            == family_ids
            for inventory in inventories.values()
        ),
        "noninteger-negative-or-nonconserving-projected-work": all(
            check[0] for check in family_checks.values()
        ),
        "step-record-loss-or-byte-nondeterminism": all(repeat_steps.values()),
        "inventory-noncanonical-not-content-addressed-or-byte-nondeterministic": all(
            repeat_inventory.values()
        ),
        "physical-identity-field-not-absent-by-design": all(
            marker.state == ABSENT_BY_DESIGN and marker.value is None
            for inventory in inventories.values()
            for marker in (
                inventory.implementation_identity.code_object_hashes,
                inventory.implementation_identity.observed_launches,
            )
        ),
        "ordinary-simllm-import-loads-framework-runtime": all(
            ordinary_imports.values()
        ),
    }
    relations = [
        {
            "id": "R1-exact-framework-text-projection",
            "passed": all(exact_projections.values()),
            "frameworks": exact_projections,
        },
        {
            "id": "R4-ordered-hybrid-schedule",
            "passed": all(item[0] for item in schedule_checks.values()),
            "framework_evidence": {
                key: value[1] for key, value in schedule_checks.items()
            },
        },
        {
            "id": "R5-shape-axis-sensitivity",
            "passed": all(item[0] for item in shape_checks.values()),
            "framework_evidence": {
                key: value[1] for key, value in shape_checks.items()
            },
        },
        {
            "id": "R7-cross-framework-structural-agreement",
            "passed": structural_match,
        },
    ]
    exact_oracles = [
        {
            "id": "R2-exact-family-projection",
            "passed": all(item[0] for item in family_checks.values()),
            "framework_evidence": {
                key: value[1] for key, value in family_checks.items()
            },
        },
        {
            "id": "R6-byte-determinism",
            "passed": all(repeat_inventory.values()) and all(repeat_steps.values()),
            "inventory_repeat_pairs": repeat_inventory,
            "step_record_repeat_pairs": repeat_steps,
        },
    ]
    structural_invariants = [
        {
            "id": "R3-family-conservation",
            "passed": all(item[0] for item in family_checks.values()),
            "conserved_cases_per_framework": {
                key: value[1]["conserved_cases"]
                for key, value in family_checks.items()
            },
            "logical_launches_per_case": 449,
        },
        {
            "id": "R8-scope-exclusion",
            "passed": (
                guards["physical-identity-field-not-absent-by-design"]
                and guards["ordinary-simllm-import-loads-framework-runtime"]
                and all(
                    not any(
                        term in family.family_id
                        for term in ("vision", "multimodal", "mtp", "speculative")
                    )
                    for inventory in inventories.values()
                    for family in inventory.kernel_families
                )
            ),
        },
    ]
    fatal_violations = sorted(name for name, passed in guards.items() if not passed)
    all_evidence_passed = all(
        item["passed"]
        for evidence in (relations, exact_oracles, structural_invariants)
        for item in evidence
    )
    state = "void" if fatal_violations else "passed" if all_evidence_passed else "failed"
    physical = expectations["physical_sanity"]
    result = {
        "schema": "simllm-model-extraction-qwen38-study-result-v2",
        "study": "model-extraction-qwen38-v2",
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
                "conserved_cases": family_checks[framework][1]["conserved_cases"],
                "logical_launches_per_case": 449,
            }
            for framework, inventory in inventories.items()
        ],
        "run_configuration": {
            "checkpoint": inventories["vllm"].model.to_obj(),
            "framework_projections": [
                inspections[framework] for framework in framework_pythons
            ],
            "repetitions_per_framework": 2,
            "gpu_execution": "none",
            "weight_loading": "none",
            "local_weight_byte_verification": False,
        },
        "fatal_guards": {"violations": fatal_violations, "checks": guards},
        "behavioral_relations": relations,
        "exact_oracles": exact_oracles,
        "structural_invariants": structural_invariants,
        "physical_sanity": {
            "batch_1_recurrent_hbm_floor_picoseconds": physical[
                "batch_1_hbm_floor_picoseconds"
            ],
            "batch_16_recurrent_hbm_floor_picoseconds": physical[
                "batch_16_hbm_floor_picoseconds"
            ],
            "retained_decode_median_anchor_microseconds": physical[
                "retained_decode_median_anchor_microseconds"
            ],
            "static_inventory_hbm_bytes": physical["static_inventory_hbm_bytes"],
            "bf16_checkpoint_payload_ceiling_bytes": physical[
                "bf16_checkpoint_payload_ceiling_bytes"
            ],
            "bounds": {
                "floor": "recurrent-state-bytes-divided-by-A100-HBM-bandwidth",
                "ceiling": "text-inventory-static-bytes-below-full-checkpoint-payload",
                "measured_position": "analytic-decode-floor-below-or-within-retained-7.7-to-57-us-medians",
            },
            "independent_angles": [
                "recurrent-state-memory-traffic",
                "text-stack-weights-versus-full-checkpoint-payload",
                "end-to-end-kernel-median-sanity-without-fitting",
            ],
        },
    }
    (arguments.output_root / "results.json").write_bytes(canonical_bytes(result))
    sys.stdout.buffer.write(canonical_bytes(result))
    sys.stdout.buffer.write(b"\n")
    return 0 if state == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
