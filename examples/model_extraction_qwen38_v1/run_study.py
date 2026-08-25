"""Run and score the frozen Qwen3.8 configuration-only extraction study."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from simllm.calibration.canonical import canonical_bytes, canonical_sha256, strict_json_loads
from simllm.calibration.extraction import case_records_from_suite, load_extraction_suite

REPOSITORY = Path(__file__).resolve().parents[2]
EXPECTATIONS = Path(__file__).with_name("expectations.json")
FREEZE_COMMIT = "f95d05a9bc0defa7171e371bcd2b2ad03db46954"
SUITE_ID = "qwen3.8-27b-text-v1"


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


def _json_report(completed: subprocess.CompletedProcess[str], name: str) -> dict[str, Any]:
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if completed.returncode != 0 or len(lines) != 1:
        raise RuntimeError(f"{name} did not emit one successful JSON projection")
    value = strict_json_loads(lines[0].encode())
    if not isinstance(value, dict):
        raise TypeError(f"{name} projection is not an object")
    return value


def _rejection_line(completed: subprocess.CompletedProcess[str]) -> str | None:
    lines = [
        line
        for line in completed.stderr.splitlines()
        if line.startswith("simllm-calibrate: extract:")
    ]
    return lines[0] if len(lines) == 1 else None


def _expected_text_stack(suite: dict[str, Any]) -> dict[str, Any]:
    model = suite["reference_model"]
    text = model["text_stack"]
    schedule = text["layer_pattern"] * text["pattern_repetitions"]
    return {
        "architecture": model["architecture"],
        "wrapper_model_type": model["model_type"],
        "text_model_type": text["model_type"],
        "scope": text["scope"],
        "geometry": model["geometry"],
        "layer_types": schedule,
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


def _framework_identity_matches(
    projection: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    framework = projection.get("framework")
    return (
        projection.get("schema") == "simllm-framework-text-config-projection-v1"
        and isinstance(framework, dict)
        and framework.get("id") == expected["id"]
        and framework.get("version") == expected["version"]
        and framework.get("source_commit") == expected["source_commit"]
        and framework.get("source_tree") == expected["source_tree"]
        and projection.get("configuration_seam") == expected["configuration_seam"]
        and projection.get("architecture_binding") == expected["architecture_binding"]
    )


def _shape_relation(
    suite: dict[str, Any], expectations: dict[str, Any]
) -> tuple[bool, dict[str, list[int]]]:
    records = case_records_from_suite(suite)
    cells = suite["graph_cells"]
    observed = {
        "prefill-prompt-tokens-per-request": [],
        "decode-context-tokens": [],
        "dense-decode-batch": [],
    }
    exact = True
    for cell, record in zip(cells, records, strict=True):
        if cell["family"] == "compute-prefill":
            observed["prefill-prompt-tokens-per-request"].append(
                cell["prompt_tokens_per_request"]
            )
            exact &= (
                len(record.scheduled) == cell["requests"]
                and record.num_tokens_after_padding == cell["total_prompt_tokens"]
                and all(
                    request.num_new_tokens == cell["prompt_tokens_per_request"]
                    and request.context_length == cell["prompt_tokens_per_request"]
                    for request in record.scheduled
                )
            )
        elif cell["family"] == "memory-decode":
            observed["decode-context-tokens"].append(cell["context_tokens"])
            exact &= (
                len(record.scheduled) == cell["batch"]
                and all(
                    request.num_new_tokens == 1
                    and request.context_length == cell["context_tokens"]
                    for request in record.scheduled
                )
            )
        else:
            observed["dense-decode-batch"].append(len(record.scheduled))
            exact &= (
                len(record.scheduled) == cell["batch"]
                and all(
                    request.num_new_tokens == 1
                    and request.context_length == cell["context_tokens"]
                    for request in record.scheduled
                )
            )
    expected = {
        item["id"]: item["values"] for item in expectations["sweep"]["parameters"]
    }
    exact &= all(sorted(observed[key]) == value for key, value in expected.items())
    return exact, observed


def main(argv: list[str] | None = None) -> int:
    arguments = _arguments(argv)
    if arguments.output_root.exists():
        raise RuntimeError("output root already exists; choose a fresh study directory")
    arguments.output_root.mkdir(parents=True)
    expectations_raw = EXPECTATIONS.read_bytes()
    expectations = strict_json_loads(expectations_raw)
    suite_file = arguments.suite_root / "suites" / SUITE_ID / "suite.json"
    suite_raw = suite_file.read_bytes()
    suite, _ = load_extraction_suite(suite_raw)
    model = suite["reference_model"]
    shards = model["weight_shards"]
    expected_frameworks = {item["id"]: item for item in expectations["frameworks"]}
    expected_text = _expected_text_stack(suite)
    framework_pythons = {
        "vllm": arguments.vllm_python,
        "sglang": arguments.sglang_python,
    }

    ordinary_imports: dict[str, bool] = {}
    projections: dict[str, dict[str, Any]] = {}
    inspection_ok: dict[str, bool] = {}
    extraction_runs: dict[str, list[subprocess.CompletedProcess[str]]] = {}
    rejection_bytes: dict[str, list[bytes]] = {}
    no_outputs: dict[str, bool] = {}
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
        inspected = _run(
            _inspect_command(python, framework, arguments.checkpoint_root),
            environment=_environment(framework),
            output_root=arguments.output_root,
            name=f"{framework}-configuration",
        )
        try:
            projections[framework] = _json_report(
                inspected, f"{framework}-configuration"
            )
        except (RuntimeError, TypeError, ValueError):
            projections[framework] = {}
        inspection_ok[framework] = bool(projections[framework])
        extraction_runs[framework] = []
        rejection_bytes[framework] = []
        no_outputs[framework] = True
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
            extraction_runs[framework].append(completed)
            line = _rejection_line(completed)
            rejection = {
                "schema": "simllm-model-extraction-rejection-v1",
                "framework": framework,
                "state": "rejected",
                "reason": line,
            }
            raw = canonical_bytes(rejection)
            rejection_bytes[framework].append(raw)
            (arguments.output_root / f"{framework}-{repetition}.rejection.json").write_bytes(
                raw
            )
            no_outputs[framework] &= (
                not (run_root / "steps.jsonl").exists()
                and not (run_root / "objects").exists()
            )

    checkpoint_config = arguments.checkpoint_root / "config.json"
    suite_case_ids = [cell["id"] for cell in suite["graph_cells"]]
    framework_identity_ok = all(
        _framework_identity_matches(projections[framework], expected_frameworks[framework])
        for framework in framework_pythons
    )
    text_projection_ok = all(
        projections[framework].get("text_stack") == expected_text
        for framework in framework_pythons
    )
    rejection_ok = all(
        completed.returncode == 2
        and (line := _rejection_line(completed)) is not None
        and "COMP-62" in line
        and "Qwen3.5 Gated DeltaNet" in line
        for completed_runs in extraction_runs.values()
        for completed in completed_runs
    )
    all_no_outputs = all(no_outputs.values())
    guards = {
        "suite-bytes-changed": _sha256(suite_raw) == expectations["suite"]["sha256"],
        "api-metadata-revision-parameter-count-shard-count-manifest-or-byte-total-mismatch": (
            model["revision"] == expectations["model"]["revision"]
            and model["parameter_count"] == expectations["model"]["parameter_count"]
            and len(shards) == expectations["model"]["weight_shard_count"]
            and sum(item["bytes"] for item in shards)
            == expectations["model"]["weight_bytes"]
            and canonical_sha256(shards) == expectations["model"]["weight_sha256"]
        ),
        "local-config-revision-or-hash-mismatch": (
            arguments.checkpoint_root.name == expectations["model"]["revision"]
            and checkpoint_config.is_file()
            and _sha256(checkpoint_config.read_bytes())
            == expectations["model"]["config_sha256"]
        ),
        "local-configuration-substrate-contains-a-weight-file": not any(
            arguments.checkpoint_root.glob("*.safetensors")
        ),
        "framework-version-source-or-architecture-binding-mismatch": (
            all(inspection_ok.values()) and framework_identity_ok
        ),
        "framework-text-geometry-layer-order-or-mechanism-mismatch": (
            all(inspection_ok.values()) and text_projection_ok
        ),
        "missing-duplicate-or-reordered-suite-case": (
            len(suite_case_ids) == expectations["suite"]["case_count"]
            and len(suite_case_ids) == len(set(suite_case_ids))
            and suite_case_ids == [cell["id"] for cell in suite["graph_cells"]]
        ),
        "step-record-or-inventory-written-after-required-rejection": all_no_outputs,
        "ordinary-simllm-import-loads-framework-runtime": all(
            ordinary_imports.values()
        ),
    }
    schedule = expected_text["layer_types"]
    r1 = {
        framework: projections[framework].get("text_stack") == expected_text
        for framework in framework_pythons
    }
    r2 = (
        len(schedule) == 64
        and schedule == [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
        * 16
        and schedule.count("linear_attention") == 48
        and schedule.count("full_attention") == 16
    )
    r3, r3_evidence = _shape_relation(suite, expectations)
    r5 = all(first == second for first, second in rejection_bytes.values())
    r6 = projections["vllm"].get("text_stack") == projections["sglang"].get(
        "text_stack"
    )
    fatal_violations = sorted(name for name, passed in guards.items() if not passed)
    relations = [
        {
            "id": "R1-exact-framework-text-projection",
            "passed": all(r1.values()),
            "frameworks": r1,
        },
        {
            "id": "R2-hybrid-layer-scaling",
            "passed": r2,
            "evidence": {
                "layers": len(schedule),
                "linear_attention_layers": schedule.count("linear_attention"),
                "full_attention_layers": schedule.count("full_attention"),
            },
        },
        {
            "id": "R3-shape-axis-sensitivity",
            "passed": r3,
            "evidence": r3_evidence,
        },
        {
            "id": "R6-cross-framework-structural-agreement",
            "passed": r6,
        },
    ]
    all_relations_pass = all(item["passed"] for item in relations)
    state = (
        "void"
        if fatal_violations
        else "passed"
        if all_relations_pass and rejection_ok and r5
        else "failed"
    )
    result = {
        "schema": "simllm-model-extraction-qwen38-study-result-v1",
        "study": "model-extraction-qwen38-v1",
        "state": state,
        "freeze_commit": FREEZE_COMMIT,
        "expectations_sha256": _sha256(expectations_raw),
        "suite_sha256": _sha256(suite_raw),
        "complete_framework_inventories": 0,
        "requested_framework_inventories": 2,
        "published_inventories": [],
        "run_configuration": {
            "model": {
                "name": model["name"],
                "revision": model["revision"],
                "weight_identity_source": model["weight_identity_source"],
                "local_weight_byte_verification": model[
                    "local_weight_byte_verification"
                ],
            },
            "framework_projections": [
                projections[framework] for framework in framework_pythons
            ],
            "repetitions_per_framework": 2,
            "gpu_execution": "none",
            "weight_loading": "none",
        },
        "fatal_guards": {"violations": fatal_violations, "checks": guards},
        "behavioral_relations": relations,
        "exact_oracles": [
            {
                "id": "R5-rejection-byte-determinism",
                "passed": r5,
                "same_framework_repeat_pairs": 2,
            }
        ],
        "structural_invariants": [
            {
                "id": "R4-total-rejection",
                "passed": rejection_ok and all_no_outputs,
                "frameworks": {
                    framework: {
                        "statuses": [
                            completed.returncode
                            for completed in extraction_runs[framework]
                        ],
                        "no_step_or_inventory_output": no_outputs[framework],
                    }
                    for framework in framework_pythons
                },
            }
        ],
        "physical_sanity": {
            "timing_or_rate_measurements": "none",
            "bf16_parameter_payload_bytes": 2 * model["parameter_count"],
            "physical_shard_bytes": model["weight_bytes"],
            "safetensors_overhead_bytes": (
                model["weight_bytes"] - 2 * model["parameter_count"]
            ),
            "hybrid_layer_conservation": "48-linear-plus-16-full-equals-64",
            "framework_mechanism": "stateful-Qwen3.5-Gated-DeltaNet",
        },
    }
    (arguments.output_root / "results.json").write_bytes(canonical_bytes(result))
    sys.stdout.buffer.write(canonical_bytes(result))
    sys.stdout.buffer.write(b"\n")
    return 0 if state == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
