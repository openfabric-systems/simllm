from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from simllm.calibration.canonical import canonical_bytes, sha256_bytes
from simllm.calibration.local_shard import (
    LOCAL_SHARD_EXCLUDED_WORK,
    LOCAL_SHARD_MEASUREMENT_SCOPE,
    LOCAL_SHARD_REQUEST_SCHEMA,
    LOCAL_SHARD_RESULT_SCHEMA,
    LocalShardCaptureError,
    run_local_shard_capture,
    synthetic_input_sha256,
    synthetic_token_rows,
    validate_local_shard_request,
    validate_local_shard_result,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_TARGET = ROOT / "examples" / "local_shard_kernel_collector_v1" / "fixture_target.py"


def request_value(
    *,
    tensor_parallel: int = 1,
    tensor_rank: int = 0,
    batch_size: int = 2,
    device_isa: str = "sm80",
) -> dict[str, object]:
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
            "device_isa": device_isa,
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
            "tensor_rank": tensor_rank,
            "pipeline_rank": 0,
            "data_rank": 0,
            "expert_rank": 0,
        },
        "phase": "decode",
        "shape": {
            "batch_size": batch_size,
            "per_request_kv_lengths": [4 + index for index in range(batch_size)],
        },
        "launch_mode": "eager",
        "synthetic_input": {
            "kind": "rank-local-input-v1",
            "token_seed": 17,
            "state_seed": 23,
            "vocabulary_size": 32_000,
        },
        "replays": 8,
    }


def result_value(request_raw: dict[str, object]) -> dict[str, object]:
    request = validate_local_shard_request(request_raw)
    samples = canonical_bytes([10_000] * 8)
    return {
        "schema": LOCAL_SHARD_RESULT_SCHEMA,
        "evidence_status": "candidate",
        "measurement_scope": LOCAL_SHARD_MEASUREMENT_SCOPE,
        "request_sha256": request.request_sha256,
        "dispatch_signature": request.value["dispatch_signature"],
        "model": request.value["model"],
        "parallelism": request.value["parallelism"],
        "physical_shard": request.value["physical_shard"],
        "phase": request.value["phase"],
        "shape": request.value["shape"],
        "launch_mode": request.value["launch_mode"],
        "synthetic_input_sha256": synthetic_input_sha256(request),
        "device": {
            "device_ordinal": 0,
            "device_isa": request.dispatch_signature.device_isa,
            "name": "fixture-device",
            "uuid": "fixture-uuid",
        },
        "excluded_work": list(LOCAL_SHARD_EXCLUDED_WORK),
        "kernels": [
            {
                "ordinal": 0,
                "operation_id": "local-gemm",
                "implementation_id": "fixture.gemm",
                "name": "fixture_gemm",
                "launch_count_per_replay": 1,
                "sample_count": 8,
                "median_elapsed_ps": 10_000,
                "median_elapsed_cycles": None,
                "samples_file": "samples-0.json",
                "samples_sha256": sha256_bytes(samples),
                "samples_bytes": len(samples),
            }
        ],
    }


def test_request_is_canonical_and_separates_logical_parallelism_from_shard() -> None:
    request = validate_local_shard_request(
        request_value(tensor_parallel=4, tensor_rank=3)
    )
    replay = validate_local_shard_request(request.record.canonical)

    assert replay.request_sha256 == request.request_sha256
    assert request.value["parallelism"]["tensor_parallel"] == 4
    assert request.value["physical_shard"]["tensor_rank"] == 3


def test_request_rejects_a_physical_rank_outside_logical_parallelism() -> None:
    with pytest.raises(LocalShardCaptureError, match="outside tensor_parallel"):
        validate_local_shard_request(
            request_value(tensor_parallel=4, tensor_rank=4)
        )


def test_decode_synthetic_rows_are_new_tokens_not_recomputed_kv_history() -> None:
    request = validate_local_shard_request(request_value(batch_size=3))

    first = synthetic_token_rows(request)
    second = synthetic_token_rows(request.record.canonical)

    assert tuple(map(len, first)) == (1, 1, 1)
    assert len(first) == 3
    assert first == second
    assert first[0] != first[1]


def test_prefill_rows_cover_only_computed_tokens_not_existing_kv() -> None:
    raw = request_value()
    raw["phase"] = "prefill"
    raw["shape"] = {
        "computed_new_tokens": 128,
        "existing_context_tokens": 512,
    }

    rows = synthetic_token_rows(raw)

    assert len(rows) == 1
    assert len(rows[0]) == 128


def test_result_rejects_architecture_and_model_identity_mismatch() -> None:
    request_raw = request_value()
    request = validate_local_shard_request(request_raw)
    result = result_value(request_raw)
    result["device"]["device_isa"] = "sm90"
    with pytest.raises(LocalShardCaptureError, match="architecture differs"):
        validate_local_shard_result(result, request)

    result = result_value(request_raw)
    result["model"] = {**result["model"], "revision": "other-revision"}
    with pytest.raises(LocalShardCaptureError, match="model.*differs"):
        validate_local_shard_result(result, request)


def test_result_requires_explicit_collective_and_network_exclusion() -> None:
    request_raw = request_value()
    result = result_value(request_raw)
    result["excluded_work"] = ["network-service"]

    with pytest.raises(LocalShardCaptureError, match="must exclude"):
        validate_local_shard_result(result, request_raw)


def test_missing_target_writes_no_output(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist"

    with pytest.raises(LocalShardCaptureError, match="target command is unavailable"):
        run_local_shard_capture(
            request_value(),
            target="simllm-target-that-does-not-exist",
            output_root=output,
        )

    assert not output.exists()


def test_nonempty_output_rejects_before_target_execution(tmp_path: Path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    marker = output / "owned.txt"
    marker.write_text("user data", encoding="utf-8")

    with pytest.raises(LocalShardCaptureError, match="absent or an empty directory"):
        run_local_shard_capture(
            request_value(),
            target=sys.executable,
            target_args=(str(FIXTURE_TARGET),),
            output_root=output,
        )

    assert marker.read_text(encoding="utf-8") == "user data"


def test_external_fixture_target_produces_valid_content_addressed_result(
    tmp_path: Path,
) -> None:
    run = run_local_shard_capture(
        request_value(tensor_parallel=4, tensor_rank=1, batch_size=8),
        target=sys.executable,
        target_args=(str(FIXTURE_TARGET),),
        output_root=tmp_path / "capture",
    )

    summary = json.loads((run.output_root / "run.json").read_text(encoding="utf-8"))
    assert summary == run.to_obj()
    assert summary["kernel_count"] == 2
    assert run.result.value["kernels"][0]["operation_id"] == (
        "local-gemm-output-width-1024"
    )
    assert run.result.value["excluded_work"] == LOCAL_SHARD_EXCLUDED_WORK


def test_runner_rejects_a_sample_blob_hash_mismatch(tmp_path: Path) -> None:
    with pytest.raises(LocalShardCaptureError, match="sample blob (size|hash) mismatch"):
        run_local_shard_capture(
            request_value(),
            target=sys.executable,
            target_args=(str(FIXTURE_TARGET), "--corrupt-sample"),
            output_root=tmp_path / "capture",
        )
