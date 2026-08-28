"""Deterministic nonphysical target for the local-shard contract study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simllm.calibration.canonical import canonical_bytes, sha256_bytes
from simllm.calibration.local_shard import (
    LOCAL_SHARD_EXCLUDED_WORK,
    LOCAL_SHARD_MEASUREMENT_SCOPE,
    LOCAL_SHARD_RESULT_SCHEMA,
    synthetic_input_sha256,
    synthetic_token_rows,
    validate_local_shard_request,
)


def _sample_row(
    *,
    ordinal: int,
    operation_id: str,
    implementation_id: str,
    name: str,
    base_ps: int,
    replays: int,
    result_root: Path,
) -> dict[str, object]:
    samples = [base_ps + (index % 3) for index in range(replays)]
    sample_bytes = canonical_bytes(samples)
    samples_file = f"samples-{ordinal}.json"
    result_root.joinpath(samples_file).write_bytes(sample_bytes)
    return {
        "ordinal": ordinal,
        "operation_id": operation_id,
        "implementation_id": implementation_id,
        "name": name,
        "launch_count_per_replay": 1,
        "sample_count": replays,
        "median_elapsed_ps": samples[replays // 2],
        "median_elapsed_cycles": None,
        "samples_file": samples_file,
        "samples_sha256": sha256_bytes(sample_bytes),
        "samples_bytes": len(sample_bytes),
    }


def run(request_path: Path, result_path: Path, *, corrupt_sample: bool = False) -> None:
    request = validate_local_shard_request(request_path.read_bytes())
    if request.value["phase"] != "decode":
        raise ValueError("the conformance target supports decode only")
    tensor_parallel = int(request.value["parallelism"]["tensor_parallel"])
    if 4096 % tensor_parallel:
        raise ValueError("fixture hidden width must divide by tensor parallel size")
    local_output_width = 4096 // tensor_parallel
    rows = synthetic_token_rows(request)
    replays = int(request.value["replays"])
    batch_size = len(rows)
    result = {
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
            "device_ordinal": request.value["physical_shard"]["device_ordinal"],
            "device_isa": request.dispatch_signature.device_isa,
            "name": "fixture-nonphysical-device",
            "uuid": "fixture-device-0",
        },
        "excluded_work": list(LOCAL_SHARD_EXCLUDED_WORK),
        "kernels": [
            _sample_row(
                ordinal=0,
                operation_id=f"local-gemm-output-width-{local_output_width}",
                implementation_id=(
                    f"fixture.gemm.batch-{batch_size}.width-{local_output_width}"
                ),
                name="fixture_local_gemm",
                base_ps=10_000 + batch_size * 100 + local_output_width,
                replays=replays,
                result_root=result_path.parent,
            ),
            _sample_row(
                ordinal=1,
                operation_id="local-normalization",
                implementation_id=f"fixture.norm.batch-{batch_size}",
                name="fixture_local_norm",
                base_ps=2_000 + batch_size * 10,
                replays=replays,
                result_root=result_path.parent,
            ),
        ],
    }
    if corrupt_sample:
        result_path.parent.joinpath("samples-0.json").write_bytes(b"corrupt")
    result_path.write_bytes(canonical_bytes(result))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--corrupt-sample", action="store_true")
    arguments = parser.parse_args()
    run(
        arguments.request,
        arguments.result,
        corrupt_sample=arguments.corrupt_sample,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
