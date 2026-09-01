"""Run one isolated pinned-vLLM smoke cell for each P3-T1 groundwork piece."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

EXPECTED_VLLM_VERSION = "0.27.1"
EXPECTED_SCHEDULER_SHA256 = (
    "c67bda2886b52865ddafabaae7d797c359e930752f374421a33e537d94a5f45a"
)
REQUEST_IDS = ("smoke-r0", "smoke-r1")
PRESSURE_REQUEST_IDS = ("pressure-r0", "pressure-r1", "pressure-r2")
PROMPT_TOKEN_IDS = tuple(range(32, 64))
NUM_GPU_BLOCKS = 8
BLOCK_TOKENS = 16


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piece", choices=("sampled", "native", "kv"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _prepare_environment(piece: str, output_dir: Path) -> None:
    values = {
        "HF_HUB_OFFLINE": "1",
        "SIMLLM_VLLM_MODE": "virtual",
        "SIMLLM_VLLM_WORKER_MODE": "skeleton",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_CACHE_ROOT": str(output_dir / "vllm-cache"),
        "VLLM_LOGGING_LEVEL": "ERROR",
        "VLLM_TARGET_DEVICE": "cpu",
        "VLLM_USE_V1": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
    }
    if piece == "kv":
        values.update(
            {
                "SIMLLM_VLLM_ORACLE_CAPTURE": "1",
                "SIMLLM_VLLM_ORACLE_LOG": str(output_dir / "oracle.jsonl"),
                "SIMLLM_VLLM_ORACLE_SCOPE": "kv",
                "VLLM_PLUGINS": "simllm_oracle",
            }
        )
    else:
        values.update(
            {
                "SIMLLM_VLLM_ORACLE_CAPTURE": "",
                "SIMLLM_VLLM_ORACLE_LOG": "",
                "SIMLLM_VLLM_ORACLE_SCOPE": "",
                "VLLM_PLUGINS": "",
            }
        )
    os.environ.update(values)


def _qualify_vllm() -> dict[str, object]:
    from vllm.v1.core.sched import scheduler

    scheduler_path = Path(inspect.getsourcefile(scheduler) or "")
    if not scheduler_path.is_file():
        raise RuntimeError("could not resolve the pinned vLLM scheduler source")
    scheduler_hash = hashlib.sha256(scheduler_path.read_bytes()).hexdigest()
    distribution_version = importlib.metadata.version("vllm")
    if distribution_version != EXPECTED_VLLM_VERSION:
        raise RuntimeError(
            f"vLLM version {distribution_version!r} is not {EXPECTED_VLLM_VERSION!r}"
        )
    if scheduler_hash != EXPECTED_SCHEDULER_SHA256:
        raise RuntimeError(
            f"vLLM scheduler hash {scheduler_hash} is not the frozen source"
        )
    return {
        "distribution_version": distribution_version,
        "scheduler_sha256": scheduler_hash,
    }


def _run_requests(
    llm: Any,
    sampling_params: Any,
    *,
    sequential: bool,
    max_tokens: int,
) -> None:
    groups = ((REQUEST_IDS[0],), (REQUEST_IDS[1],)) if sequential else (REQUEST_IDS,)
    for group in groups:
        for request_id in group:
            assigned = llm.llm_engine.add_request(
                request_id,
                {"prompt_token_ids": list(PROMPT_TOKEN_IDS)},
                sampling_params(
                    temperature=0.0,
                    max_tokens=max_tokens,
                    min_tokens=1,
                    detokenize=False,
                ),
            )
            if assigned != request_id:
                raise RuntimeError("vLLM changed an explicit smoke request identity")
        while llm.llm_engine.has_unfinished_requests():
            llm.llm_engine.step()


def _run_allocation_pressure(llm: Any, sampling_params: Any) -> None:
    for index, request_id in enumerate(PRESSURE_REQUEST_IDS):
        prompt_start = 256 + index * len(PROMPT_TOKEN_IDS)
        assigned = llm.llm_engine.add_request(
            request_id,
            {
                "prompt_token_ids": list(
                    range(prompt_start, prompt_start + len(PROMPT_TOKEN_IDS))
                )
            },
            sampling_params(
                temperature=0.0,
                max_tokens=31,
                min_tokens=1,
                detokenize=False,
            ),
        )
        if assigned != request_id:
            raise RuntimeError("vLLM changed an explicit pressure request identity")
    while llm.llm_engine.has_unfinished_requests():
        llm.llm_engine.step()


def _sampled_summary(records: list[Any]) -> dict[str, object]:
    from simllm.core import step_record_to_json

    if not records:
        raise RuntimeError("sampled smoke emitted no step records")
    for record in records:
        if record.sampled_request_ids is None:
            raise RuntimeError("sampled identity selection emitted a legacy record")
        if len(record.sampled_request_ids) != record.num_sampled:
            raise RuntimeError("sampled identity cardinality disagrees with its count")
    sampled = [
        request_id
        for record in records
        for request_id in record.sampled_request_ids or ()
    ]
    if set(sampled) != set(REQUEST_IDS):
        raise RuntimeError(f"sampled identities do not cover the requests: {sampled}")
    example = next(record for record in records if record.sampled_request_ids)
    return {
        "step_count": len(records),
        "sampled_request_ids": sampled,
        "worked_example": step_record_to_json(example),
    }


def _native_summary(path: Path) -> dict[str, object]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise RuntimeError("native smoke emitted no capture rows")
    for row in rows:
        native = row["native_scheduler_output"]
        ordered = native["ordered_scheduled_request_ids"]
        token_rows = native["num_scheduled_tokens"]
        if ordered != [value["request_id"] for value in token_rows]:
            raise RuntimeError("native scheduled IDs lost SchedulerOutput order")
        if native["total_num_scheduled_tokens"] != sum(
            value["num_scheduled_tokens"] for value in token_rows
        ):
            raise RuntimeError("native scheduled-token total is not conserved")
        if row["step_index"] != row["step_record"]["step_index"]:
            raise RuntimeError("native and projected step indices disagree")
    return {
        "step_count": len(rows),
        "worked_example": rows[0],
    }


def _kv_summary(output_dir: Path, worker: Any) -> dict[str, object]:
    from simllm.adapters.vllm import (
        VllmKvGeometry,
        read_vllm_kv_sidecar,
        vllm_kv_projection_to_json,
    )

    dtype_bytes = worker.dims.kv_dtype_bytes
    if int(dtype_bytes) != dtype_bytes:
        raise RuntimeError("KV smoke requires an integral cache dtype width")
    geometry = VllmKvGeometry(
        pool_id="vllm:kv:rank0",
        block_tokens=BLOCK_TOKENS,
        capacity_blocks=NUM_GPU_BLOCKS,
        num_layers=worker.dims.num_layers,
        num_kv_heads=worker.dims.num_kv_heads,
        head_size=worker.dims.head_size,
        dtype=str(worker._answers.kv_cache_dtype()),
        dtype_bytes=int(dtype_bytes),
    )
    projection = read_vllm_kv_sidecar(output_dir / "oracle.jsonl", geometry)
    payload = vllm_kv_projection_to_json(projection)
    _write_json(output_dir / "kv_projection.json", payload)
    actions = [row["action"] for row in payload["operations"]]
    required = {
        "allocate",
        "bind-prefix",
        "free",
        "recompute",
        "release",
        "touch",
    }
    missing = sorted(required - set(actions))
    if missing:
        raise RuntimeError(f"KV smoke did not reach required operations: {missing}")
    if "reserve" in actions:
        raise RuntimeError("KV bridge emitted an unwitnessed reserve operation")
    return {
        "action_counts": dict(sorted(Counter(actions).items())),
        "allocation_failure_preemptions": actions.count("recompute"),
        "pool": payload["pool"],
        "source_event_count": payload["source_event_count"],
        "worked_example": payload["operations"][:4],
    }


def _run_smoke(args: argparse.Namespace) -> dict[str, object]:
    import vllm.platforms
    from vllm.platforms.cpu import CpuPlatform

    vllm.platforms._current_platform = CpuPlatform()
    from vllm.v1.executor import uniproc_executor

    uniproc_executor.get_ip = lambda: "127.0.0.1"
    uniproc_executor.get_open_port = lambda: 29_500
    from vllm import LLM, SamplingParams

    from simllm.adapters.vllm import (
        SimExecutorConfig,
        configure,
        latest_worker,
        mark_oracle_capture_start,
        reset_configuration,
    )

    provenance = _qualify_vllm()
    provenance["platform"] = type(vllm.platforms.current_platform).__name__
    steps_path = args.output_dir / "steps.jsonl"
    native_path = args.output_dir / "native_steps.jsonl"
    config = SimExecutorConfig(
        mode="virtual",
        token_id=512,
        step_records_path=str(steps_path),
        emit_sampled_request_ids=args.piece in {"sampled", "native"},
        native_step_capture_path=str(native_path) if args.piece == "native" else None,
    )
    reset_configuration()
    configure(config=config)
    llm = None
    try:
        llm = LLM(
            model=str(args.model),
            worker_cls="simllm.adapters.vllm.SimWorker",
            enforce_eager=True,
            max_model_len=64,
            max_num_seqs=2,
            max_num_batched_tokens=64,
            num_gpu_blocks_override=NUM_GPU_BLOCKS,
            block_size=BLOCK_TOKENS,
            disable_log_stats=True,
            enable_chunked_prefill=False,
            enable_prefix_caching=args.piece == "kv",
            async_scheduling=False,
        )
        worker = latest_worker()
        if worker is None:
            raise RuntimeError("pinned engine did not construct SimWorker")
        if args.piece == "kv":
            mark_oracle_capture_start((*REQUEST_IDS, *PRESSURE_REQUEST_IDS))
        _run_requests(
            llm,
            SamplingParams,
            sequential=args.piece == "kv",
            max_tokens=2 if args.piece == "kv" else 1,
        )
        if args.piece == "kv":
            _run_allocation_pressure(llm, SamplingParams)

        if args.piece == "sampled":
            summary = _sampled_summary(worker.step_records)
        elif args.piece == "native":
            summary = _native_summary(native_path)
        else:
            summary = _kv_summary(args.output_dir, worker)
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        reset_configuration()

    return {
        "piece": args.piece,
        "status": "PASS",
        "fixed_configuration": {
            "async_scheduling": False,
            "block_tokens": BLOCK_TOKENS,
            "distributed_init_method": "tcp://127.0.0.1:29500",
            "in_process": True,
            "num_gpu_blocks_override": NUM_GPU_BLOCKS,
            "platform_override": "CpuPlatform",
            "v1_multiprocessing": False,
        },
        "provenance": provenance,
        "summary": summary,
    }


def main() -> int:
    args = _parse_args()
    if not args.model.is_dir() or not (args.model / "config.json").is_file():
        raise SystemExit("--model must name a local model snapshot")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _prepare_environment(args.piece, args.output_dir)
    try:
        result = _run_smoke(args)
    except BaseException as exc:
        _write_json(
            args.output_dir / "result.json",
            {
                "error": repr(exc),
                "piece": args.piece,
                "status": "FAIL",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    _write_json(args.output_dir / "result.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
