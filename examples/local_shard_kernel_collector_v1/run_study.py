"""Run the frozen nonphysical local-shard collector conformance grid."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simllm.calibration.canonical import canonical_bytes
from simllm.calibration.local_shard import (
    LOCAL_SHARD_MEASUREMENT_SCOPE,
    LOCAL_SHARD_REQUEST_SCHEMA,
    run_local_shard_capture,
    synthetic_token_rows,
    validate_local_shard_request,
)

EXPECTATION_COMMIT = "808125133ebfcd930b28a0b4f962ecad111e6d3f"
FIXTURE_TARGET = Path(__file__).with_name("fixture_target.py")


def _request(tensor_parallel: int, batch_size: int) -> dict[str, object]:
    return {
        "schema": LOCAL_SHARD_REQUEST_SCHEMA,
        "measurement_scope": LOCAL_SHARD_MEASUREMENT_SCOPE,
        "dispatch_signature": {
            "schema": "simllm-dispatch-signature-v1",
            "framework_id": "fixture-framework",
            "framework_version": "1.0",
            "backend_id": "fixture-backend",
            "backend_version": "1.0",
            "kernel_library_id": "fixture-kernels",
            "kernel_library_version": "1.0",
            "algorithm_policy_id": "deterministic",
            "device_isa": "sm80",
            "numeric_traits": [
                {
                    "trait_id": "dtype",
                    "value_type": "string",
                    "value": "bf16",
                }
            ],
            "layout_traits": [
                {
                    "trait_id": "weight-layout",
                    "value_type": "string",
                    "value": "row-major",
                }
            ],
        },
        "model": {
            "name": "fixture/model",
            "revision": "fixture-revision",
            "family": "dense",
        },
        "parallelism": {
            "tensor_parallel": tensor_parallel,
            "pipeline_parallel": 1,
            "data_parallel": 1,
            "expert_parallel": 1,
        },
        "physical_shard": {
            "device_ordinal": 0,
            "tensor_rank": 0,
            "pipeline_rank": 0,
            "data_rank": 0,
            "expert_rank": 0,
        },
        "phase": "decode",
        "shape": {
            "batch_size": batch_size,
            "per_request_kv_lengths": [8] * batch_size,
        },
        "launch_mode": "eager",
        "synthetic_input": {
            "kind": "token-ids-v1",
            "seed": 17,
            "vocabulary_size": 32_000,
        },
        "replays": 8,
    }


def run_study() -> dict[str, object]:
    cells: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="simllm-local-shard-study-") as temporary:
        root = Path(temporary)
        for tensor_parallel in (1, 4):
            for batch_size in (1, 8):
                request = validate_local_shard_request(
                    _request(tensor_parallel, batch_size)
                )
                runs = [
                    run_local_shard_capture(
                        request,
                        target=sys.executable,
                        target_args=(str(FIXTURE_TARGET),),
                        output_root=root
                        / f"tp{tensor_parallel}-b{batch_size}-repeat{repeat}",
                    )
                    for repeat in (1, 2)
                ]
                first, second = runs
                operation_ids = tuple(
                    kernel["operation_id"] for kernel in first.result.value["kernels"]
                )
                repeat_operation_ids = tuple(
                    kernel["operation_id"] for kernel in second.result.value["kernels"]
                )
                local_width = int(str(operation_ids[0]).rsplit("-", 1)[1])
                cells.append(
                    {
                        "cell_id": f"tp{tensor_parallel}-b{batch_size}",
                        "tensor_parallel": tensor_parallel,
                        "batch_size": batch_size,
                        "request_sha256": request.request_sha256,
                        "result_sha256": first.result.result_sha256,
                        "repeat_result_sha256": second.result.result_sha256,
                        "synthetic_row_count": len(synthetic_token_rows(request)),
                        "local_gemm_output_width": local_width,
                        "kernel_operation_ids": list(operation_ids),
                        "repeat_kernel_operation_ids": list(repeat_operation_ids),
                    }
                )
    row_count_matches = all(cell["synthetic_row_count"] == cell["batch_size"] for cell in cells)
    width_matches = all(
        cell["local_gemm_output_width"] == 4096 // cell["tensor_parallel"]
        for cell in cells
    )
    stable_replays = all(
        cell["result_sha256"] == cell["repeat_result_sha256"]
        and cell["kernel_operation_ids"] == cell["repeat_kernel_operation_ids"]
        for cell in cells
    )
    unique_results = len({cell["result_sha256"] for cell in cells}) == len(cells)
    exact_identity = len({cell["request_sha256"] for cell in cells}) == len(cells)
    relations = {
        "all_four_cells_preserve_exact_request_identity": exact_identity,
        "synthetic_input_row_count_equals_batch_size": row_count_matches,
        "local_gemm_output_width_scales_inversely_with_tensor_parallel": width_matches,
        "kernel_result_order_is_stable_across_identical_replays": stable_replays,
        "batch_or_tensor_parallel_change_changes_content_addressed_result": unique_results,
    }
    passed = all(relations.values())
    return {
        "schema": "simllm-local-shard-kernel-collector-study-result-v1",
        "study": "local_shard_kernel_collector_v1",
        "expectation_commit": EXPECTATION_COMMIT,
        "status": "PASS" if passed else "FAIL",
        "evidence_class": "nonphysical-contract-conformance",
        "scored_cell_count": len(cells),
        "passed_cell_count": len(cells) if passed else 0,
        "fatal_guards": {
            "fixture_target_exit": "pass",
            "request_and_result_validation": "pass" if passed else "fail",
            "sample_blob_closure": "pass" if passed else "fail",
        },
        "relations": relations,
        "physical_sanity": {
            "status": "not-applicable",
            "reason": "Fixture durations are nonphysical contract values and are not interpreted as GPU timing.",
        },
        "cells": cells,
        "project_effect": (
            "The local run slice of COMP-50 is executable behind one external "
            "framework target contract."
        ),
        "project_non_effect": (
            "COMP-50 and COMP-1 stay open; no GPU or network constant, calibration "
            "status, TTFT, TPOT or framework coverage claim changes."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    result = run_study()
    arguments.output.write_bytes(canonical_bytes(result))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
