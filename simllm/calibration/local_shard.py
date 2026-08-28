"""Framework-neutral orchestration for one physical kernel shard capture.

The framework target remains the authority for what it compiled and ran. This
module owns a strict request and result boundary, deterministic synthetic token
inputs, fail-closed identity checks and a subprocess launcher that never uses a
shell. It does not turn an isolated shard into distributed-system evidence.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simllm.compute.device_model import (
    _array,
    _fields,
    _integer,
    _object,
    _sha256,
    _string,
)

from .bindings import DispatchSignature
from .canonical import canonical_bytes, sha256_bytes
from .record_types import RecordObject, record_object

LOCAL_SHARD_REQUEST_SCHEMA = "simllm-local-shard-kernel-capture-request-v1"
LOCAL_SHARD_RESULT_SCHEMA = "simllm-local-shard-kernel-capture-result-v1"
LOCAL_SHARD_RUN_SCHEMA = "simllm-local-shard-kernel-capture-run-v1"
LOCAL_SHARD_MEASUREMENT_SCOPE = "rank-local-compute-only"
LOCAL_SHARD_EXCLUDED_WORK = ("distributed-collectives", "network-service")


class LocalShardCaptureError(ValueError):
    """The local shard request, target or result violated its contract."""


@dataclass(frozen=True, slots=True)
class LocalShardCaptureRequest:
    """One immutable request to compile and measure a physical model shard."""

    record: RecordObject
    dispatch_signature: DispatchSignature

    @property
    def request_sha256(self) -> str:
        return self.record.record_id

    @property
    def value(self) -> Mapping[str, Any]:
        return self.record.value


@dataclass(frozen=True, slots=True)
class LocalShardCaptureResult:
    """One candidate result emitted by the exact requested framework target."""

    record: RecordObject

    @property
    def result_sha256(self) -> str:
        return self.record.record_id

    @property
    def value(self) -> Mapping[str, Any]:
        return self.record.value


@dataclass(frozen=True, slots=True)
class LocalShardCaptureRun:
    """Validated request and result identities for one completed target run."""

    request: LocalShardCaptureRequest
    result: LocalShardCaptureResult
    output_root: Path

    def to_obj(self) -> dict[str, Any]:
        return {
            "schema": LOCAL_SHARD_RUN_SCHEMA,
            "request_sha256": self.request.request_sha256,
            "result_sha256": self.result.result_sha256,
            "kernel_count": len(self.result.value["kernels"]),
        }


RecordInput = Mapping[str, Any] | str | bytes | bytearray | memoryview


def _positive_integer(value: object, path: str) -> int:
    result = _integer(value, path, nonnegative=True)
    if result == 0:
        raise LocalShardCaptureError(f"{path}: expected a positive integer")
    return result


def _exact_string(value: object, expected: str, path: str) -> str:
    actual = _string(value, path)
    if actual != expected:
        raise LocalShardCaptureError(
            f"{path}: expected {expected!r}; found {actual!r}"
        )
    return actual


def _validate_model(value: object, path: str) -> Mapping[str, Any]:
    model = _object(value, path)
    _fields(model, path, {"name", "revision", "family"})
    for field in ("name", "revision", "family"):
        _string(model[field], f"{path}.{field}")
    return model


def _validate_parallelism(value: object, path: str) -> dict[str, int]:
    parallelism = _object(value, path)
    fields = {
        "tensor_parallel",
        "pipeline_parallel",
        "data_parallel",
        "expert_parallel",
    }
    _fields(parallelism, path, fields)
    return {
        field: _positive_integer(parallelism[field], f"{path}.{field}")
        for field in fields
    }


def _validate_physical_shard(
    value: object,
    parallelism: Mapping[str, int],
    path: str,
) -> Mapping[str, Any]:
    shard = _object(value, path)
    rank_fields = {
        "tensor_rank": "tensor_parallel",
        "pipeline_rank": "pipeline_parallel",
        "data_rank": "data_parallel",
        "expert_rank": "expert_parallel",
    }
    _fields(shard, path, {"device_ordinal", *rank_fields})
    _integer(shard["device_ordinal"], f"{path}.device_ordinal", nonnegative=True)
    for rank_field, size_field in rank_fields.items():
        rank = _integer(shard[rank_field], f"{path}.{rank_field}", nonnegative=True)
        if rank >= parallelism[size_field]:
            raise LocalShardCaptureError(
                f"{path}.{rank_field}: rank {rank} is outside "
                f"{size_field} size {parallelism[size_field]}"
            )
    return shard


def _validate_shape(value: object, phase: str, path: str) -> Mapping[str, Any]:
    shape = _object(value, path)
    if phase == "decode":
        _fields(shape, path, {"batch_size", "per_request_kv_lengths"})
        batch_size = _positive_integer(shape["batch_size"], f"{path}.batch_size")
        lengths = _array(shape["per_request_kv_lengths"], f"{path}.per_request_kv_lengths")
        if len(lengths) != batch_size:
            raise LocalShardCaptureError(
                f"{path}.per_request_kv_lengths: expected {batch_size} values"
            )
        for index, length in enumerate(lengths):
            _positive_integer(length, f"{path}.per_request_kv_lengths[{index}]")
    elif phase == "prefill":
        _fields(shape, path, {"computed_new_tokens", "existing_context_tokens"})
        _positive_integer(shape["computed_new_tokens"], f"{path}.computed_new_tokens")
        _integer(
            shape["existing_context_tokens"],
            f"{path}.existing_context_tokens",
            nonnegative=True,
        )
    else:
        raise LocalShardCaptureError(f"{path}: unsupported phase {phase!r}")
    return shape


def _validate_synthetic_input(value: object, path: str) -> Mapping[str, Any]:
    synthetic = _object(value, path)
    _fields(synthetic, path, {"kind", "seed", "vocabulary_size"})
    _exact_string(synthetic["kind"], "token-ids-v1", f"{path}.kind")
    _integer(synthetic["seed"], f"{path}.seed", nonnegative=True)
    vocabulary_size = _positive_integer(
        synthetic["vocabulary_size"], f"{path}.vocabulary_size"
    )
    if vocabulary_size < 256:
        raise LocalShardCaptureError(
            f"{path}.vocabulary_size: expected at least 256 token IDs"
        )
    return synthetic


def validate_local_shard_request(value: RecordInput) -> LocalShardCaptureRequest:
    """Validate and content-address one strict local-shard request."""

    try:
        record = record_object(value).require_schema(LOCAL_SHARD_REQUEST_SCHEMA)
        payload = _object(record.value, "request")
        _fields(
            payload,
            "request",
            {
                "schema",
                "measurement_scope",
                "dispatch_signature",
                "model",
                "parallelism",
                "physical_shard",
                "phase",
                "shape",
                "launch_mode",
                "synthetic_input",
                "replays",
            },
        )
        _exact_string(
            payload["measurement_scope"],
            LOCAL_SHARD_MEASUREMENT_SCOPE,
            "request.measurement_scope",
        )
        dispatch = DispatchSignature.from_obj(
            payload["dispatch_signature"], "request.dispatch_signature"
        )
        _validate_model(payload["model"], "request.model")
        parallelism = _validate_parallelism(
            payload["parallelism"], "request.parallelism"
        )
        _validate_physical_shard(
            payload["physical_shard"], parallelism, "request.physical_shard"
        )
        phase = _string(payload["phase"], "request.phase")
        _validate_shape(payload["shape"], phase, "request.shape")
        launch_mode = _string(payload["launch_mode"], "request.launch_mode")
        if launch_mode not in {"cuda-graph", "eager"}:
            raise LocalShardCaptureError(
                "request.launch_mode: expected 'cuda-graph' or 'eager'"
            )
        _validate_synthetic_input(payload["synthetic_input"], "request.synthetic_input")
        _positive_integer(payload["replays"], "request.replays")
    except (TypeError, ValueError) as error:
        if isinstance(error, LocalShardCaptureError):
            raise
        raise LocalShardCaptureError(str(error)) from error
    return LocalShardCaptureRequest(record=record, dispatch_signature=dispatch)


def synthetic_token_rows(
    request: LocalShardCaptureRequest | RecordInput,
) -> tuple[tuple[int, ...], ...]:
    """Materialize deterministic token rows from the request recipe."""

    validated = (
        request
        if isinstance(request, LocalShardCaptureRequest)
        else validate_local_shard_request(request)
    )
    payload = validated.value
    shape = payload["shape"]
    if payload["phase"] == "decode":
        lengths = tuple(int(length) for length in shape["per_request_kv_lengths"])
    else:
        lengths = (
            int(shape["computed_new_tokens"]) + int(shape["existing_context_tokens"]),
        )
    seed = int(payload["synthetic_input"]["seed"])
    vocabulary_size = int(payload["synthetic_input"]["vocabulary_size"])
    return tuple(
        tuple(
            (seed + row_index * 1_000_003 + token_index) % vocabulary_size
            for token_index in range(length)
        )
        for row_index, length in enumerate(lengths)
    )


def synthetic_input_sha256(
    request: LocalShardCaptureRequest | RecordInput,
) -> str:
    """Hash the exact deterministic token rows supplied to a target."""

    rows = synthetic_token_rows(request)
    return sha256_bytes(canonical_bytes([list(row) for row in rows]))


def _same_member(
    result: Mapping[str, Any],
    request: Mapping[str, Any],
    field: str,
) -> None:
    if canonical_bytes(result[field]) != canonical_bytes(request[field]):
        raise LocalShardCaptureError(
            f"result.{field}: target identity differs from the request"
        )


def _validate_kernel_rows(value: object, request: LocalShardCaptureRequest) -> None:
    kernels = _array(value, "result.kernels")
    if not kernels:
        raise LocalShardCaptureError("result.kernels: at least one observation is required")
    operation_ids: list[str] = []
    for index, raw in enumerate(kernels):
        path = f"result.kernels[{index}]"
        kernel = _object(raw, path)
        _fields(
            kernel,
            path,
            {
                "ordinal",
                "operation_id",
                "implementation_id",
                "name",
                "launch_count_per_replay",
                "sample_count",
                "median_elapsed_ps",
                "median_elapsed_cycles",
                "samples_file",
                "samples_sha256",
                "samples_bytes",
            },
        )
        ordinal = _integer(kernel["ordinal"], f"{path}.ordinal", nonnegative=True)
        if ordinal != index:
            raise LocalShardCaptureError(
                f"{path}.ordinal: expected contiguous ordinal {index}"
            )
        operation_ids.append(_string(kernel["operation_id"], f"{path}.operation_id"))
        _string(kernel["implementation_id"], f"{path}.implementation_id")
        _string(kernel["name"], f"{path}.name")
        _positive_integer(
            kernel["launch_count_per_replay"], f"{path}.launch_count_per_replay"
        )
        sample_count = _positive_integer(kernel["sample_count"], f"{path}.sample_count")
        if sample_count < int(request.value["replays"]):
            raise LocalShardCaptureError(
                f"{path}.sample_count: expected at least request.replays samples"
            )
        _positive_integer(kernel["median_elapsed_ps"], f"{path}.median_elapsed_ps")
        cycles = kernel["median_elapsed_cycles"]
        if cycles is not None:
            _positive_integer(cycles, f"{path}.median_elapsed_cycles")
        samples_file = Path(_string(kernel["samples_file"], f"{path}.samples_file"))
        if samples_file.is_absolute() or ".." in samples_file.parts:
            raise LocalShardCaptureError(
                f"{path}.samples_file: expected a path below output_root"
            )
        _sha256(kernel["samples_sha256"], f"{path}.samples_sha256")
        _positive_integer(kernel["samples_bytes"], f"{path}.samples_bytes")
    if len(operation_ids) != len(set(operation_ids)):
        raise LocalShardCaptureError("result.kernels: operation IDs must be unique")


def validate_local_shard_result(
    value: RecordInput,
    request: LocalShardCaptureRequest | RecordInput,
) -> LocalShardCaptureResult:
    """Validate one target result against the exact request identity."""

    validated_request = (
        request
        if isinstance(request, LocalShardCaptureRequest)
        else validate_local_shard_request(request)
    )
    try:
        record = record_object(value).require_schema(LOCAL_SHARD_RESULT_SCHEMA)
        payload = _object(record.value, "result")
        _fields(
            payload,
            "result",
            {
                "schema",
                "evidence_status",
                "measurement_scope",
                "request_sha256",
                "dispatch_signature",
                "model",
                "parallelism",
                "physical_shard",
                "phase",
                "shape",
                "launch_mode",
                "synthetic_input_sha256",
                "device",
                "excluded_work",
                "kernels",
            },
        )
        _exact_string(payload["evidence_status"], "candidate", "result.evidence_status")
        _exact_string(
            payload["measurement_scope"],
            LOCAL_SHARD_MEASUREMENT_SCOPE,
            "result.measurement_scope",
        )
        request_sha256 = _sha256(payload["request_sha256"], "result.request_sha256")
        if request_sha256 != validated_request.request_sha256:
            raise LocalShardCaptureError(
                "result.request_sha256: target result belongs to a different request"
            )
        for field in (
            "dispatch_signature",
            "model",
            "parallelism",
            "physical_shard",
            "phase",
            "shape",
            "launch_mode",
        ):
            _same_member(payload, validated_request.value, field)
        expected_input = synthetic_input_sha256(validated_request)
        actual_input = _sha256(
            payload["synthetic_input_sha256"], "result.synthetic_input_sha256"
        )
        if actual_input != expected_input:
            raise LocalShardCaptureError(
                "result.synthetic_input_sha256: target used different synthetic input"
            )
        device = _object(payload["device"], "result.device")
        _fields(device, "result.device", {"device_ordinal", "device_isa", "name", "uuid"})
        ordinal = _integer(
            device["device_ordinal"], "result.device.device_ordinal", nonnegative=True
        )
        expected_ordinal = int(validated_request.value["physical_shard"]["device_ordinal"])
        if ordinal != expected_ordinal:
            raise LocalShardCaptureError(
                "result.device.device_ordinal: target used a different physical device"
            )
        device_isa = _string(device["device_isa"], "result.device.device_isa")
        if device_isa != validated_request.dispatch_signature.device_isa:
            raise LocalShardCaptureError(
                "result.device.device_isa: target architecture differs from the request"
            )
        _string(device["name"], "result.device.name")
        _string(device["uuid"], "result.device.uuid")
        excluded = tuple(
            _string(item, f"result.excluded_work[{index}]")
            for index, item in enumerate(_array(payload["excluded_work"], "result.excluded_work"))
        )
        if excluded != LOCAL_SHARD_EXCLUDED_WORK:
            raise LocalShardCaptureError(
                "result.excluded_work: isolated capture must exclude distributed "
                "collectives and network service"
            )
        _validate_kernel_rows(payload["kernels"], validated_request)
    except (TypeError, ValueError) as error:
        if isinstance(error, LocalShardCaptureError):
            raise
        raise LocalShardCaptureError(str(error)) from error
    return LocalShardCaptureResult(record=record)


def _resolve_target(target: str | os.PathLike[str]) -> str:
    spelling = os.fspath(target)
    if not spelling:
        raise LocalShardCaptureError("target command must not be empty")
    if os.sep in spelling or (os.altsep is not None and os.altsep in spelling):
        path = Path(spelling)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise LocalShardCaptureError(
                f"target command is unavailable or not executable: {spelling}"
            )
        return str(path)
    resolved = shutil.which(spelling)
    if resolved is None:
        raise LocalShardCaptureError(f"target command is unavailable: {spelling}")
    return resolved


def _verify_sample_blobs(
    result: LocalShardCaptureResult,
    output_root: Path,
) -> None:
    resolved_root = output_root.resolve()
    for index, kernel in enumerate(result.value["kernels"]):
        path = output_root / str(kernel["samples_file"])
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(resolved_root)
            if path.is_symlink() or not resolved.is_file():
                raise OSError("sample blob is not a regular file")
            payload = resolved.read_bytes()
        except OSError as error:
            raise LocalShardCaptureError(
                f"result.kernels[{index}].samples_file: cannot read {path.name}"
            ) from error
        except ValueError as error:
            raise LocalShardCaptureError(
                f"result.kernels[{index}].samples_file: resolves outside output_root"
            ) from error
        if len(payload) != int(kernel["samples_bytes"]):
            raise LocalShardCaptureError(
                f"result.kernels[{index}].samples_bytes: sample blob size mismatch"
            )
        if sha256_bytes(payload) != kernel["samples_sha256"]:
            raise LocalShardCaptureError(
                f"result.kernels[{index}].samples_sha256: sample blob hash mismatch"
            )


def run_local_shard_capture(
    request: LocalShardCaptureRequest | RecordInput,
    *,
    target: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    target_args: Sequence[str] = (),
    environment: Mapping[str, str] | None = None,
) -> LocalShardCaptureRun:
    """Run one declared target and validate its candidate kernel result."""

    validated_request = (
        request
        if isinstance(request, LocalShardCaptureRequest)
        else validate_local_shard_request(request)
    )
    executable = _resolve_target(target)
    if isinstance(target_args, (str, bytes)):
        raise TypeError("target_args must be a sequence of argument strings")
    arguments = tuple(target_args)
    if any(not isinstance(argument, str) for argument in arguments):
        raise TypeError("target_args must contain only strings")
    output = Path(output_root)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise LocalShardCaptureError("output_root must be absent or an empty directory")
    output.mkdir(parents=True, exist_ok=True)
    request_path = output / "request.json"
    result_path = output / "result.json"
    request_path.write_bytes(validated_request.record.canonical)
    command = [
        executable,
        *arguments,
        "--request",
        str(request_path),
        "--result",
        str(result_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        env=None if environment is None else dict(environment),
    )
    if completed.stdout:
        (output / "target.stdout").write_bytes(completed.stdout)
    if completed.stderr:
        (output / "target.stderr").write_bytes(completed.stderr)
    if completed.returncode != 0:
        raise LocalShardCaptureError(
            f"target command failed with status {completed.returncode}"
        )
    if not result_path.is_file():
        raise LocalShardCaptureError("target command did not write result.json")
    result = validate_local_shard_result(result_path.read_bytes(), validated_request)
    _verify_sample_blobs(result, output)
    run = LocalShardCaptureRun(
        request=validated_request,
        result=result,
        output_root=output,
    )
    (output / "run.json").write_bytes(canonical_bytes(run.to_obj()))
    return run


__all__ = [
    "LOCAL_SHARD_EXCLUDED_WORK",
    "LOCAL_SHARD_MEASUREMENT_SCOPE",
    "LOCAL_SHARD_REQUEST_SCHEMA",
    "LOCAL_SHARD_RESULT_SCHEMA",
    "LOCAL_SHARD_RUN_SCHEMA",
    "LocalShardCaptureError",
    "LocalShardCaptureRequest",
    "LocalShardCaptureResult",
    "LocalShardCaptureRun",
    "run_local_shard_capture",
    "synthetic_input_sha256",
    "synthetic_token_rows",
    "validate_local_shard_request",
    "validate_local_shard_result",
]
