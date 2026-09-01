"""Run and score the frozen VLLM-48 live collective timing study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from simllm.core import (
    CollectiveServiceEnvironment,
    step_record_from_json,
    step_record_to_json,
)
from simllm.traffic import (
    CollectiveFloorCalibration,
    CollectiveFloorEnvironmentMismatchError,
    CollectiveFloorRegime,
    CollectiveFloorSourceIdentity,
    compare_collective_service_to_floor,
)

CONFIG_PATH = Path(__file__).with_name("study_config.json")
RESULT_SCHEMA = "simllm-vllm-collective-timing-result-v1"
OLD_CANONICAL_BYTES = (
    b'{"finished_request_ids":[],"preempted_request_ids":[],"scheduled":[],'
    b'"schema":"atlahs-closed-loop-step-v1","step_index":4,'
    b'"virtual_time_ps":9}'
)


class FatalGuardError(RuntimeError):
    """A frozen precondition failed, so the behavioral score is void."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _require_external_directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_absolute() or not resolved.is_dir():
        raise SystemExit(f"{label} must be an existing absolute directory: {path}")
    if resolved == REPOSITORY_ROOT or REPOSITORY_ROOT in resolved.parents:
        raise SystemExit(f"{label} must remain outside the repository")
    return resolved


def _validate_pins(args: argparse.Namespace, config: dict[str, Any]) -> None:
    pins = config["pins"]
    source = args.vllm_source.resolve()
    model = args.model_path.resolve()
    if not source.is_dir():
        raise FatalGuardError("pinned vLLM source directory is missing")
    if not model.is_dir():
        raise FatalGuardError("pinned model snapshot is missing")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise FatalGuardError("could not read the vLLM source commit") from error
    if commit != pins["vllm_commit"]:
        raise FatalGuardError(
            f"vLLM source commit changed: expected {pins['vllm_commit']}, got {commit}"
        )
    pinned_files = {
        "vllm/distributed/parallel_state.py": pins["parallel_state_sha256"],
        "vllm/v1/worker/gpu_model_runner.py": pins["model_runner_sha256"],
        "vllm/model_executor/models/granitemoe.py": pins["granite_model_sha256"],
    }
    for relative, expected in pinned_files.items():
        observed = _sha256(source / relative)
        if observed != expected:
            raise FatalGuardError(f"pinned source hash changed for {relative}: {observed}")
    if _sha256(model / "config.json") != pins["model_config_sha256"]:
        raise FatalGuardError("pinned Granite model config hash changed")


def _child_preflight(args: argparse.Namespace) -> None:
    version = importlib.metadata.version("vllm")
    if version != "0.27.1+cpu":
        raise FatalGuardError(f"child vLLM version is {version}, not 0.27.1+cpu")
    import torch
    from vllm.platforms import current_platform

    if not current_platform.is_cpu():
        raise FatalGuardError(f"child selected non-CPU platform {type(current_platform)}")
    if not torch.cpu._is_avx2_supported() or torch.cpu._is_avx512_supported():
        raise FatalGuardError("child did not select the frozen AVX2-only host path")


def _run_child(args: argparse.Namespace) -> None:
    _child_preflight(args)
    from vllm import LLM, SamplingParams

    config = _load_config()
    engine = config["engine"]
    requests = config["requests"]
    llm: Any | None = None
    assigned_ids: list[str] = []
    output_ids: dict[str, list[int]] = {entry["request_id"]: [] for entry in requests}
    finished: set[str] = set()
    step_count = 0
    try:
        llm = LLM(
            model=str(args.model_path),
            tensor_parallel_size=engine["tensor_parallel_size"],
            pipeline_parallel_size=engine["pipeline_parallel_size"],
            distributed_executor_backend="mp",
            enforce_eager=engine["enforce_eager"],
            dtype="bfloat16",
            seed=engine["seed"],
            max_model_len=engine["max_model_len"],
            max_num_batched_tokens=engine["max_num_batched_tokens"],
            max_num_seqs=engine["max_num_seqs"],
            disable_log_stats=True,
            enable_chunked_prefill=engine["chunked_prefill"],
            enable_prefix_caching=False,
            async_scheduling=False,
        )
        for request in requests:
            request_id = llm.llm_engine.add_request(
                request["request_id"],
                {"prompt_token_ids": list(request["prompt_token_ids"])},
                SamplingParams(
                    temperature=0.0,
                    max_tokens=request["max_tokens"],
                    ignore_eos=request["ignore_eos"],
                    detokenize=False,
                ),
            )
            assigned_ids.append(request_id)
        while llm.llm_engine.has_unfinished_requests():
            step_count += 1
            if step_count > 8:
                raise FatalGuardError("live engine exceeded the frozen manual-step bound")
            for output in llm.llm_engine.step():
                request_id = output.request_id
                token_ids = list(output.outputs[0].token_ids) if output.outputs else []
                output_ids[request_id] = token_ids
                if output.finished:
                    finished.add(request_id)
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()

    payload = {
        "schema": "simllm-vllm-collective-timing-child-v1",
        "vllm_version": importlib.metadata.version("vllm"),
        "assigned_request_ids": assigned_ids,
        "finished_request_ids": sorted(finished),
        "output_token_ids": output_ids,
        "manual_step_count": step_count,
    }
    args.child_result.write_bytes(_canonical(payload) + b"\n")


def _child_environment(
    args: argparse.Namespace, capture_path: Path, system_label: str
) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_USE_V2_MODEL_RUNNER": "0",
            "VLLM_CPU_KVCACHE_SPACE": "1",
            "VLLM_CPU_OMP_THREADS_BIND": args.omp_threads_bind,
            "SIMLLM_VLLM_COLLECTIVE_CAPTURE": "1",
            "SIMLLM_VLLM_COLLECTIVE_CAPTURE_PATH": str(capture_path),
            "SIMLLM_VLLM_COLLECTIVE_SYSTEM": system_label,
        }
    )
    if args.library_path:
        environment["LD_LIBRARY_PATH"] = args.library_path
    return environment


def _launch_fresh_run(
    args: argparse.Namespace,
    attempt_dir: Path,
    run_index: int,
    system_label: str,
) -> dict[str, Path]:
    run_dir = attempt_dir / f"live-run-{run_index}"
    run_dir.mkdir()
    capture_path = run_dir / "steps.jsonl"
    child_result = run_dir / "child-result.json"
    log_path = run_dir / "live.log"
    command = [
        str(args.vllm_python),
        str(Path(__file__).resolve()),
        "--child",
        "--model-path",
        str(args.model_path),
        "--child-result",
        str(child_result),
    ]
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        completed = subprocess.run(
            command,
            check=False,
            env=_child_environment(args, capture_path, system_label),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        raise FatalGuardError(
            f"fresh live run {run_index} failed with status {completed.returncode}; see {log_path}"
        )
    if not capture_path.is_file() or not child_result.is_file():
        raise FatalGuardError(f"fresh live run {run_index} omitted required output")
    return {
        "run_dir": run_dir,
        "capture": capture_path,
        "child_result": child_result,
        "log": log_path,
    }


def _read_capture(path: Path) -> tuple[list[dict[str, Any]], list[Any]]:
    raw_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records = [step_record_from_json(row) for row in raw_rows]
    return raw_rows, records


def _expected_projection(record: Any, model_config: dict[str, Any]) -> list[dict[str, Any]]:
    hidden_size = int(model_config["hidden_size"])
    layers = int(model_config["num_hidden_layers"])
    tensor_bytes = record.total_new_tokens * hidden_size * 2
    tensor_shape = [record.total_new_tokens, hidden_size]
    expected = [
        {
            "sequence": 0,
            "kind": "all_reduce",
            "payload_bytes": tensor_bytes,
            "world_size": 2,
            "dtype": "bfloat16",
            "element_width_bytes": 2,
            "tensor_shape": tensor_shape,
            "group_tag": "tp:0",
            "layer_index": None,
            "layer_name": None,
        }
    ]
    for layer in range(layers):
        for _site in ("attention-output", "moe-output"):
            expected.append(
                {
                    "sequence": len(expected),
                    "kind": "all_reduce",
                    "payload_bytes": tensor_bytes,
                    "world_size": 2,
                    "dtype": "bfloat16",
                    "element_width_bytes": 2,
                    "tensor_shape": tensor_shape,
                    "group_tag": "tp:0",
                    "layer_index": layer,
                    "layer_name": f"model.layers.{layer}",
                }
            )
    padded_vocab = ((int(model_config["vocab_size"]) + 63) // 64) * 64
    local_vocab = padded_vocab // 2
    expected.append(
        {
            "sequence": len(expected),
            "kind": "gather",
            "payload_bytes": int(record.num_sampled) * local_vocab * 2,
            "world_size": 2,
            "dtype": "bfloat16",
            "element_width_bytes": 2,
            "tensor_shape": [int(record.num_sampled), local_vocab],
            "group_tag": "tp:0",
            "layer_index": None,
            "layer_name": None,
        }
    )
    return expected


def _observed_projection(record: Any) -> list[dict[str, Any]]:
    capture = record.collective_service
    if capture is None:
        return []
    return [
        {
            "sequence": invocation.sequence,
            "kind": invocation.kind,
            "payload_bytes": invocation.payload_bytes,
            "world_size": invocation.world_size,
            "dtype": invocation.dtype,
            "element_width_bytes": invocation.element_width_bytes,
            "tensor_shape": list(invocation.tensor_shape),
            "group_tag": invocation.group_tag,
            "layer_index": invocation.layer_index,
            "layer_name": invocation.layer_name,
        }
        for invocation in capture.invocations
    ]


def _run_shape_projection(records: list[Any]) -> list[dict[str, Any]]:
    projection = []
    for record in records:
        projection.append(
            {
                "step_index": record.step_index,
                "num_sampled": record.num_sampled,
                "invocations": _observed_projection(record),
                "environment": {
                    "system": record.collective_service.environment.system,
                    "backend": record.collective_service.environment.backend,
                    "device_type": record.collective_service.environment.device_type,
                    "framework": record.collective_service.environment.framework,
                    "framework_version": (record.collective_service.environment.framework_version),
                    "timer": record.collective_service.environment.timer,
                },
            }
        )
    return projection


def _fixture_calibration() -> CollectiveFloorCalibration:
    source = CollectiveFloorSourceIdentity(
        artifact_sha256="0" * 64,
        tool="vllm48-contract-fixture",
        aiconfigurator_version="1",
        aiconfigurator_core_version="1",
        system="floor-fixture",
        backend="nccl",
        database_version="1",
        row_version="1",
        duplicate_resolution="none",
    )
    regime = CollectiveFloorRegime(
        dtype="half",
        operation="all_reduce",
        ranks=2,
        regime_index=0,
        lower_bytes=1,
        upper_bytes=1_000_000,
        floor_ps=Fraction(100),
        slope_ps_per_byte=Fraction(1),
        training_cell_ids=("fixture-a", "fixture-b"),
    )
    return CollectiveFloorCalibration(
        calibration_id="vllm48-contract-fixture",
        source=source,
        fitted_byte_range=(1, 1_000_000),
        regimes=(regime,),
    )


def _score_schema(raw_rows: list[dict[str, Any]]) -> tuple[bool, bool]:
    loaded_old = step_record_from_json(json.loads(OLD_CANONICAL_BYTES))
    old_pass = _canonical(step_record_to_json(loaded_old)) == OLD_CANONICAL_BYTES
    new_pass = _new_schema_predicate(raw_rows)
    return old_pass, new_pass


def _new_schema_predicate(raw_rows: list[dict[str, Any]]) -> bool:
    loaded = [step_record_from_json(row) for row in raw_rows]
    return all(
        record.collective_service is not None
        and _canonical(step_record_to_json(record)) == _canonical(row)
        for record, row in zip(loaded, raw_rows, strict=True)
    )


def _refusal_predicate(action: Any) -> bool:
    try:
        action()
    except CollectiveFloorEnvironmentMismatchError:
        return True
    return False


def _score_comparator(records: list[Any]) -> tuple[bool, bool, dict[str, Any]]:
    first = records[0].collective_service
    source_invocation = next(
        invocation for invocation in first.invocations if invocation.kind == "all_reduce"
    )
    invocation = replace(source_invocation, dtype="float16")
    calibration = _fixture_calibration()
    refusal = _refusal_predicate(
        lambda: compare_collective_service_to_floor(
            invocation=invocation,
            environment=first.environment,
            calibration=calibration,
            floor_dtype="half",
        )
    )
    comparison = compare_collective_service_to_floor(
        invocation=invocation,
        environment=first.environment,
        calibration=calibration,
        floor_dtype="half",
        acknowledge_cross_environment=True,
    )
    acknowledged = comparison.cross_environment_acknowledged
    return refusal, acknowledged, comparison.as_dict()


def _acknowledgement_predicate(comparison: dict[str, Any]) -> bool:
    return comparison.get("cross_environment_acknowledged") is True


def _bypassed_comparator() -> None:
    """Mutation control that skips the comparator's default refusal."""


def _mutation_controls(
    expected_runs: list[list[list[dict[str, Any]]]],
    observed_runs: list[list[list[dict[str, Any]]]],
    shape_runs: list[list[dict[str, Any]]],
    raw_rows: list[dict[str, Any]],
    acknowledgement: dict[str, Any],
) -> dict[str, bool]:
    dropped = deepcopy(observed_runs[0])
    dropped[0] = dropped[0][1:]
    changed_payload = deepcopy(observed_runs[0])
    changed_payload[0][0]["payload_bytes"] += 1
    changed_kind_shape = deepcopy(shape_runs[1])
    changed_kind_shape[0]["invocations"][0]["kind"] = "broadcast"
    no_envelope = deepcopy(raw_rows)
    no_envelope[0].pop("collective_service")
    acknowledged_without_stamp = deepcopy(acknowledgement)
    acknowledged_without_stamp["cross_environment_acknowledged"] = False
    return {
        "drop-first-call": dropped != expected_runs[0],
        "increment-first-payload-byte": changed_payload != expected_runs[0],
        "change-second-run-first-kind": changed_kind_shape != shape_runs[0],
        "delete-optional-envelope": not _new_schema_predicate(no_envelope),
        "bypass-cross-environment-refusal": not _refusal_predicate(_bypassed_comparator),
        "drop-cross-environment-acknowledgement-stamp": not (
            _acknowledgement_predicate(acknowledged_without_stamp)
        ),
    }


def _fatal_guards(
    config: dict[str, Any],
    system_label: str,
    run_artifacts: list[dict[str, Path]],
    records_by_run: list[list[Any]],
) -> dict[str, bool]:
    requests = [entry["request_id"] for entry in config["requests"]]
    guards: dict[str, bool] = {}
    guards["fresh-distinct-run-directories"] = (
        run_artifacts[0]["run_dir"] != run_artifacts[1]["run_dir"]
    )
    guards["two-manual-steps"] = True
    for index, (artifact, records) in enumerate(
        zip(run_artifacts, records_by_run, strict=True), start=1
    ):
        child = json.loads(artifact["child_result"].read_text())
        guards[f"run-{index}-version"] = child["vllm_version"] == "0.27.1+cpu"
        assigned = child["assigned_request_ids"]
        scheduled_ids = [request.request_id for request in records[0].scheduled] if records else []
        guards[f"run-{index}-request-identity"] = (
            len(assigned) == len(requests)
            and all(
                internal.startswith(f"{logical}-")
                for internal, logical in zip(assigned, requests, strict=True)
            )
            and scheduled_ids == assigned
            and child["finished_request_ids"] == sorted(requests)
            and sorted(child["output_token_ids"]) == sorted(requests)
        )
        guards[f"run-{index}-two-manual-steps"] = child["manual_step_count"] == 2
        guards[f"run-{index}-two-records"] = len(records) == 2
        invocations = [
            invocation for record in records for invocation in record.collective_service.invocations
        ]
        guards[f"run-{index}-population-100"] = len(invocations) == 100
        guards[f"run-{index}-positive-service"] = all(
            invocation.service_ps > 0 for invocation in invocations
        )
        guards[f"run-{index}-world-size-two"] = all(
            invocation.world_size == 2 for invocation in invocations
        )
        guards[f"run-{index}-environment"] = all(
            record.collective_service.environment
            == CollectiveServiceEnvironment(
                system=system_label,
                backend="gloo",
                device_type="cpu",
                framework="vllm",
                framework_version="0.27.1+cpu",
                timer="host-monotonic-ns",
            )
            for record in records
        )
    return guards


def _score(
    config: dict[str, Any],
    model_config: dict[str, Any],
    system_label: str,
    run_artifacts: list[dict[str, Path]],
) -> dict[str, Any]:
    raw_by_run: list[list[dict[str, Any]]] = []
    records_by_run: list[list[Any]] = []
    for artifact in run_artifacts:
        raw, records = _read_capture(artifact["capture"])
        raw_by_run.append(raw)
        records_by_run.append(records)

    guards = _fatal_guards(config, system_label, run_artifacts, records_by_run)
    expected_runs = [
        [_expected_projection(record, model_config) for record in records]
        for records in records_by_run
    ]
    observed_runs = [
        [_observed_projection(record) for record in records] for records in records_by_run
    ]
    shape_runs = [_run_shape_projection(records) for records in records_by_run]
    schema_old, schema_new = _score_schema(raw_by_run[0])
    refusal, acknowledged, acknowledgement = _score_comparator(records_by_run[0])
    mutations = _mutation_controls(
        expected_runs,
        observed_runs,
        shape_runs,
        raw_by_run[0],
        acknowledgement,
    )
    guards["all-mutation-controls-fire"] = all(mutations.values())
    if not all(guards.values()):
        failed = sorted(name for name, passed in guards.items() if not passed)
        raise FatalGuardError(f"fatal guards failed: {failed}")

    families = {
        "M1": [observed_runs[index] == expected_runs[index] for index in range(2)],
        "M2": [shape_runs[0] == shape_runs[1]],
        "M3": [schema_old, schema_new],
        "M4": [refusal],
        "M5": [acknowledged],
    }
    passed = sum(result for results in families.values() for result in results)
    total = sum(len(results) for results in families.values())
    services = [
        invocation.service_ps
        for records in records_by_run
        for record in records
        for invocation in record.collective_service.invocations
    ]
    metadata_mismatches = []
    for run_index, (expected_run, observed_run) in enumerate(
        zip(expected_runs, observed_runs, strict=True), start=1
    ):
        for step_index, (expected_step, observed_step) in enumerate(
            zip(expected_run, observed_run, strict=True)
        ):
            for invocation_index, (expected, observed) in enumerate(
                zip(expected_step, observed_step, strict=True)
            ):
                if expected != observed:
                    metadata_mismatches.append(
                        {
                            "run": run_index,
                            "step": step_index,
                            "invocation": invocation_index,
                            "expected": expected,
                            "observed": observed,
                        }
                    )
                    break
    return {
        "schema": RESULT_SCHEMA,
        "task": "VLLM-48",
        "status": "PASS" if passed == total else "FAIL",
        "expectation_commit": "2808a04",
        "live_environment": {
            "vllm_version": "0.27.1+cpu",
            "device": "cpu",
            "backend": "gloo",
            "tensor_parallel_size": 2,
            "system": system_label,
            "local_timing_score": "unscored",
        },
        "population": {
            "fresh_runs": 2,
            "steps_per_run": [len(records) for records in records_by_run],
            "collectives_per_run": [
                sum(len(record.collective_service.invocations) for record in records)
                for records in records_by_run
            ],
            "metadata_conservation": families["M1"],
            "metadata_mismatches": metadata_mismatches,
        },
        "shape_determinism": families["M2"][0],
        "timing_diagnostics_ps": {
            "count": len(services),
            "minimum": min(services),
            "maximum": max(services),
            "values_recorded": True,
            "scored": False,
        },
        "schema_compatibility": {
            "old_exact_bytes": schema_old,
            "new_exact_round_trip": schema_new,
        },
        "comparator": {
            "default_cross_environment_refusal": refusal,
            "acknowledged_cross_environment_use": acknowledged,
            "acknowledged_result": acknowledgement,
        },
        "fatal_guards": guards,
        "mutation_controls": mutations,
        "families": families,
        "behavioral_score": {"passed": passed, "total": total},
        "bulk_artifacts": [
            {
                name: str(path)
                for name, path in artifact.items()
                if name in {"run_dir", "capture", "child_result", "log"}
            }
            for artifact in run_artifacts
        ],
    }


def _run_parent(args: argparse.Namespace) -> None:
    config = _load_config()
    _validate_pins(args, config)
    run_root = _require_external_directory(args.run_root, "run root")
    if not args.vllm_python.is_absolute() or not args.vllm_python.is_file():
        raise SystemExit("vLLM Python must be an existing absolute executable path")
    attempt_dir = run_root / args.attempt_name
    attempt_dir.mkdir()
    run_artifacts = [
        _launch_fresh_run(args, attempt_dir, index, args.system_label) for index in (1, 2)
    ]
    model_config = json.loads((args.model_path / "config.json").read_text())
    try:
        result = _score(config, model_config, args.system_label, run_artifacts)
    except FatalGuardError as error:
        result = {
            "schema": RESULT_SCHEMA,
            "task": "VLLM-48",
            "status": "VOID",
            "finding": str(error),
            "behavioral_score": None,
        }
    (attempt_dir / "results.json").write_bytes(_canonical(result) + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


def _check_only() -> None:
    config = _load_config()
    if config["task"] != "VLLM-48":
        raise SystemExit("study config task changed")
    if sum(config["families"].values()) != 7:
        raise SystemExit("frozen behavioral denominator changed")
    if config["source_population"] != {
        "steps_per_run": 2,
        "calls_per_step": 50,
        "calls_per_run": 100,
        "decoder_layers": 24,
        "per_step_sequence": [
            "embedding:all_reduce",
            "repeat-24:attention:all_reduce,moe:all_reduce",
            "logits:gather",
        ],
    }:
        raise SystemExit("frozen source population changed")
    loaded = step_record_from_json(json.loads(OLD_CANONICAL_BYTES))
    if _canonical(step_record_to_json(loaded)) != OLD_CANONICAL_BYTES:
        raise SystemExit("old step-record canonical bytes changed")
    print("VLLM-48 standalone checks passed")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--vllm-python", type=Path)
    parser.add_argument("--vllm-source", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--attempt-name")
    parser.add_argument("--system-label", default="local-cpu-gloo")
    parser.add_argument("--library-path", default="")
    parser.add_argument("--omp-threads-bind", default="nobind")
    parser.add_argument("--child-result", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.check:
        _check_only()
        return
    if args.child:
        if args.model_path is None or args.child_result is None:
            raise SystemExit("--child requires --model-path and --child-result")
        _run_child(args)
        return
    required = {
        "--vllm-python": args.vllm_python,
        "--vllm-source": args.vllm_source,
        "--model-path": args.model_path,
        "--run-root": args.run_root,
        "--attempt-name": args.attempt_name,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"missing required arguments: {', '.join(missing)}")
    _run_parent(args)


if __name__ == "__main__":
    main()
