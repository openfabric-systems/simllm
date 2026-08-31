#!/usr/bin/env python3
"""Execute and publish the frozen surrogate conformance study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from simllm.adapters.vllm import (
    SimExecutorConfig,
    VllmKvGeometry,
    configure,
    latest_worker,
    mark_oracle_capture_start,
    read_vllm_kv_sidecar,
    reset_configuration,
    vllm_kv_projection_to_json,
)
from simllm.adapters.vllm.oracle import register_oracle_hooks
from simllm.backends import (
    DeviceRuntimeStepSink,
    SerialStepLowererConfig,
)
from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
from simllm.core import (
    BookkeepingScope,
    CreatedObjectKind,
    CreatedObjectRecord,
    CreatedObjectRef,
    KvCacheAction,
    KvCacheWork,
    KvPoolSpec,
    ObjectOwner,
    OperationCorrelation,
    RequestBookkeeper,
    StepRecord,
    StepResult,
    VirtualClock,
    step_record_to_json,
    step_result_to_json,
)
from simllm.deploy import (
    EstimateStamp,
    EvidenceClass,
    NamedTermEstimate,
    SurrogateLoopConfig,
    SurrogateQueuePolicy,
    SurrogateRequest,
    SurrogateReserveMode,
    SurrogateServingLoop,
    SurrogateStopPolicy,
    TermEstimate,
    surrogate_loop_stamp,
)
from simllm.workload import AdmissionMode, RequestAdmissionGate

CONFIG_PATH = Path(__file__).with_name("study_config.json")
CONFIG_FREEZE_COMMIT = "d947cea75ace88f220d2c06f58c6939d5929c932"
RESULT_SCHEMA = "simllm-surrogate-conformance-record-v1"
ATTEMPT_SCHEMA = "simllm-surrogate-conformance-attempt-v1"
EXPECTED_CONFIG_SCHEMA = "simllm-surrogate-conformance-config-v1"
PS_PER_SECOND = 1_000_000_000_000


@dataclass(frozen=True)
class FrozenRequest:
    request_id: str
    arrived_at_ps: int
    prompt_token_ids: tuple[int, ...]
    max_output_tokens: int
    priority: int


@dataclass(frozen=True)
class FrozenCell:
    family: str
    cell_id: str
    family_clause: str
    engine: Mapping[str, Any]
    requests: tuple[FrozenRequest, ...]
    execution_mode: str = "drain"


@dataclass(frozen=True)
class LiveCellResult:
    cell: FrozenCell
    causal_tuple: Mapping[str, Any]
    records: tuple[StepRecord, ...]
    results: tuple[StepResult, ...]
    raw_native: tuple[Mapping[str, Any], ...]
    paired_native: tuple[Mapping[str, Any], ...]
    kv_operations: tuple[tuple[str, KvCacheWork], ...]
    admission_order: tuple[str, ...]
    output_token_ids: Mapping[str, tuple[int, ...]]
    first_release_ps: Mapping[str, int]
    final_virtual_time_ps: int
    stopped_at_head: bool
    evidence_dir: Path


@dataclass(frozen=True)
class SurrogateCellResult:
    cell: FrozenCell
    causal_tuple: Mapping[str, Any]
    records: tuple[StepRecord, ...]
    results: tuple[StepResult, ...]
    kv_operations: tuple[tuple[str, KvCacheWork], ...]
    admission_order: tuple[str, ...]
    output_token_ids: Mapping[str, tuple[int, ...]]
    first_release_ps: Mapping[str, int]
    final_virtual_time_ps: int
    stopped_at_head: bool
    waiting_request_ids: tuple[str, ...]
    evidence_dir: Path | None


class FixedStepProvider(ComputeProvider):
    """Return the frozen synthetic service for every priced step."""

    def __init__(self, duration_ps: int) -> None:
        self.duration_ps = duration_ps

    def estimate(self, kernel: object, gpu: object) -> DurationEstimate:
        return DurationEstimate(
            duration_ps=self.duration_ps,
            bound="surrogate-conformance-fixed-step",
        )


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"append-only evidence already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value if isinstance(value, bytes) else _canonical_bytes(value)
    with path.open("xb") as stream:
        stream.write(payload)


def _write_jsonl_once(path: Path, values: Iterable[object]) -> None:
    if path.exists():
        raise FileExistsError(f"append-only evidence already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(_canonical_bytes(value).decode("utf-8"))


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != EXPECTED_CONFIG_SCHEMA:
        raise ValueError(f"unsupported study configuration {value.get('schema')!r}")
    return value


def _expand_prompt(value: Mapping[str, Any]) -> tuple[int, ...]:
    explicit = value.get("prompt_token_ids")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise ValueError("prompt_token_ids must be a nonempty list")
        return tuple(int(token) for token in explicit)
    pattern = value.get("prompt")
    if not isinstance(pattern, Mapping):
        raise TypeError("request must define prompt or prompt_token_ids")
    start = int(pattern["start"])
    count = int(pattern["count"])
    if count <= 0:
        raise ValueError("prompt count must be positive")
    return tuple(range(start, start + count))


def _request(value: Mapping[str, Any]) -> FrozenRequest:
    return FrozenRequest(
        request_id=str(value["request_id"]),
        arrived_at_ps=int(value["arrived_at_ps"]),
        prompt_token_ids=_expand_prompt(value),
        max_output_tokens=int(value["max_output_tokens"]),
        priority=int(value.get("priority", 0)),
    )


def frozen_cells(config: Mapping[str, Any]) -> tuple[FrozenCell, ...]:
    cells: list[FrozenCell] = []
    families = config["families"]
    for family in ("F1", "F2", "F3", "F4"):
        for value in families[family]["cells"]:
            cells.append(
                FrozenCell(
                    family=family,
                    cell_id=value["cell_id"],
                    family_clause=value["family_clause"],
                    engine=value["engine"],
                    requests=tuple(_request(row) for row in value["requests"]),
                    execution_mode=value.get("execution_mode", "drain"),
                )
            )
    f5 = families["F5"]
    for value in f5["cells"]:
        requests = tuple(
            FrozenRequest(
                request_id=request_id,
                arrived_at_ps=(
                    0 if request_id == "r0" else int(value["arrival_offset_ps"])
                ),
                prompt_token_ids=tuple(f5["prompt_token_ids"]),
                max_output_tokens=int(f5["output_tokens"][request_id]),
                priority=0,
            )
            for request_id in value["request_ids"]
        )
        cells.append(
            FrozenCell(
                family="F5",
                cell_id=value["cell_id"],
                family_clause=value["family_clause"],
                engine=f5["engine"],
                requests=requests,
            )
        )
    identifiers = tuple(cell.cell_id for cell in cells)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("study configuration repeats a cell ID")
    return tuple(cells)


def wall_cell(config: Mapping[str, Any]) -> FrozenCell:
    value = config["wall_time"]
    pattern = value["workload"]
    requests = tuple(
        FrozenRequest(
            request_id=pattern["request_id_format"].format(index=index),
            arrived_at_ps=int(pattern["arrived_at_ps"]),
            prompt_token_ids=tuple(
                range(
                    int(pattern["prompt_start"]) + index * int(pattern["prompt_stride"]),
                    int(pattern["prompt_start"])
                    + index * int(pattern["prompt_stride"])
                    + int(pattern["prompt_tokens"]),
                )
            ),
            max_output_tokens=int(pattern["max_output_tokens"]),
            priority=int(pattern["priority"]),
        )
        for index in range(int(pattern["request_count"]))
    )
    return FrozenCell(
        family="W",
        cell_id="w-largest-frozen-workload",
        family_clause=value["clause"],
        engine=value["engine"],
        requests=requests,
    )


def _model_dims(config: Mapping[str, Any]) -> ModelDims:
    geometry = config["oracle"]["model_geometry"]
    return ModelDims(
        num_layers=int(geometry["num_layers"]),
        hidden_size=int(geometry["hidden_size"]),
        intermediate_size=int(geometry["intermediate_size"]),
        num_heads=int(geometry["num_attention_heads"]),
        num_kv_heads=int(geometry["num_kv_heads"]),
        head_size=int(geometry["head_size"]),
        vocab_size=int(geometry["vocab_size"]),
        dtype_bytes=int(geometry["dtype_bytes"]),
    )


def _pricing_stamp() -> EstimateStamp:
    return surrogate_loop_stamp(
        EstimateStamp(
            candidate_key="3" * 64,
            terms=(
                NamedTermEstimate(
                    "conformance-fixed-step",
                    TermEstimate(
                        1,
                        EvidenceClass.DECLARED,
                        "frozen synthetic conformance service",
                    ),
                ),
            ),
        )
    )


def _fixed_pricer(duration_ps: int):
    def price(record: StepRecord) -> StepResult:
        return StepResult(
            step_index=record.step_index,
            step_latency_ps=duration_ps,
            completed_at_ps=record.virtual_time_ps + duration_ps,
        )

    return price


def _surrogate_config(cell: FrozenCell, config: Mapping[str, Any]) -> SurrogateLoopConfig:
    fixed = config["fixed_engine"]
    engine = cell.engine
    return SurrogateLoopConfig(
        resolved_max_num_scheduled_tokens=int(engine["budget"]),
        max_num_seqs=int(engine["max_num_seqs"]),
        enable_chunked_prefill=bool(engine["chunked_prefill"]),
        long_prefill_token_threshold=int(engine["long_prefill_threshold"]),
        max_model_len=int(engine["max_model_len"]),
        queue_policy=SurrogateQueuePolicy(str(fixed["queue_policy"])),
        scheduler_block_size=int(fixed["scheduler_block_size"]),
        num_kv_blocks=int(engine["num_kv_blocks"]),
        reserve_mode=SurrogateReserveMode(str(fixed["reserve_mode"])),
        watermark=float(fixed["watermark"]),
        enable_prefix_caching=bool(engine["prefix_caching"]),
    )


def _surrogate_requests(
    cell: FrozenCell, config: Mapping[str, Any]
) -> tuple[SurrogateRequest, ...]:
    token_id = int(config["oracle"]["simulated_token_id"])
    return tuple(
        SurrogateRequest(
            request_id=row.request_id,
            arrived_at_ps=row.arrived_at_ps,
            prompt_token_ids=row.prompt_token_ids,
            max_output_tokens=row.max_output_tokens,
            priority=row.priority,
            stop_policy=SurrogateStopPolicy(default_token_id=token_id),
        )
        for row in cell.requests
    )


def _pool_spec(cell: FrozenCell, config: Mapping[str, Any], pool_id: str) -> KvPoolSpec:
    geometry = config["oracle"]["model_geometry"]
    return KvPoolSpec(
        pool_id=pool_id,
        block_bytes=int(geometry["kv_block_bytes"]),
        block_tokens=int(config["fixed_engine"]["scheduler_block_size"]),
        capacity_blocks=int(cell.engine["num_kv_blocks"]),
    )


def _kv_work_json(operation: tuple[str, KvCacheWork]) -> dict[str, Any]:
    operation_id, work = operation
    return {
        "operation_id": operation_id,
        "action": work.action.value,
        "pool_id": work.pool_id,
        "request_id": work.request_id,
        "block_ids": list(work.block_ids),
        "token_start": work.token_start,
        "token_end": work.token_end,
        "layer": work.layer,
        "dtype": work.dtype,
        "byte_count": work.byte_count,
        "placement_epoch": work.placement_epoch,
        "reference_count": work.reference_count,
        "cause": work.cause,
        "correlation_id": work.correlation_id,
    }


def run_surrogate_cell(
    cell: FrozenCell,
    config: Mapping[str, Any],
    evidence_dir: Path | None = None,
) -> SurrogateCellResult:
    loop_config = _surrogate_config(cell, config)
    loop = SurrogateServingLoop(
        loop_config,
        _surrogate_requests(cell, config),
        _pool_spec(cell, config, "conformance:kv"),
        _pricing_stamp(),
        kv_dtype=str(config["fixed_engine"]["kv_dtype"]),
    )
    stopped = False
    result = None
    if cell.execution_mode == "one-decision":
        try:
            loop.step(
                step_sink=_fixed_pricer(
                    int(config["fixed_engine"]["pricing_step_ps"])
                )
            )
        except RuntimeError as exc:
            if "waiting head cannot be admitted" not in str(exc):
                raise
            stopped = True
    else:
        result = loop.run(
            step_sink=_fixed_pricer(int(config["fixed_engine"]["pricing_step_ps"]))
        )
    records = tuple(emission.record for emission in loop.emissions)
    results = tuple(emission.result for emission in loop.emissions)
    operations = tuple(
        operation for emission in loop.emissions for operation in emission.kv_operations
    )
    request_results = () if result is None else result.request_results
    output_tokens = {
        row.request_id: row.output_token_ids for row in request_results
    }
    first_release = {
        row.request_id: row.first_released_at_ps for row in request_results
    }
    value = SurrogateCellResult(
        cell=cell,
        causal_tuple=dict(loop_config.causal_tuple),
        records=records,
        results=results,
        kv_operations=operations,
        admission_order=loop.admission_gate.admitted_request_ids,
        output_token_ids=output_tokens,
        first_release_ps=first_release,
        final_virtual_time_ps=loop.clock.now_ps,
        stopped_at_head=stopped,
        waiting_request_ids=loop.waiting_request_ids,
        evidence_dir=evidence_dir,
    )
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=False)
        _write_jsonl_once(
            evidence_dir / "steps.jsonl",
            (step_record_to_json(record) for record in records),
        )
        _write_jsonl_once(
            evidence_dir / "kv_operations.jsonl",
            (_kv_work_json(operation) for operation in operations),
        )
        _write_once(evidence_dir / "summary.json", surrogate_summary(value))
    return value


def surrogate_summary(value: SurrogateCellResult) -> dict[str, Any]:
    return {
        "cell_id": value.cell.cell_id,
        "family": value.cell.family,
        "family_clause": value.cell.family_clause,
        "causal_tuple": value.causal_tuple,
        "records": [step_record_to_json(record) for record in value.records],
        "kv_operations": [
            _kv_work_json(operation) for operation in value.kv_operations
        ],
        "admission_order": list(value.admission_order),
        "output_token_ids": {
            key: list(tokens) for key, tokens in value.output_token_ids.items()
        },
        "first_release_ps": dict(value.first_release_ps),
        "final_virtual_time_ps": value.final_virtual_time_ps,
        "stopped_at_head": value.stopped_at_head,
        "waiting_request_ids": list(value.waiting_request_ids),
    }


def _bookkeeper(cell: FrozenCell) -> RequestBookkeeper:
    bookkeeper = RequestBookkeeper()
    bookkeeper.extend(
        CreatedObjectRecord(
            ref=CreatedObjectRef(
                CreatedObjectKind.FRAMEWORK_REQUEST,
                f"conformance-request:{row.request_id}",
            ),
            owner=ObjectOwner.FRAMEWORK,
            created_at_ps=row.arrived_at_ps,
            scope=BookkeepingScope(
                correlation=OperationCorrelation(request_ids=(row.request_id,))
            ),
            native_id=row.request_id,
        )
        for row in cell.requests
    )
    return bookkeeper


def _append_native_probe(path: Path, scheduler_output: object) -> None:
    tokens = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not isinstance(tokens, dict):
        raise TypeError("SchedulerOutput.num_scheduled_tokens must be a dict")
    row = {
        "schema": "simllm-surrogate-conformance-native-probe-v1",
        "ordered_scheduled_request_ids": list(tokens),
        "num_scheduled_tokens": [
            {"request_id": request_id, "num_scheduled_tokens": int(count)}
            for request_id, count in tokens.items()
        ],
        "total_num_scheduled_tokens": int(
            scheduler_output.total_num_scheduled_tokens
        ),
        "preempted_request_ids": sorted(
            getattr(scheduler_output, "preempted_req_ids", None) or ()
        ),
        "finished_request_ids": sorted(
            getattr(scheduler_output, "finished_req_ids", None) or ()
        ),
    }
    encoded = _canonical_bytes(row)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("short native scheduler probe write")
    finally:
        os.close(descriptor)


def _install_native_scheduler_probe() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    original = Scheduler.schedule
    if getattr(original, "_simllm_conformance_probe", False):
        return

    def observed_schedule(scheduler: object, *args: object, **kwargs: object):
        output = original(scheduler, *args, **kwargs)
        path = os.environ.get("SIMLLM_CONFORMANCE_NATIVE_PROBE", "")
        if path:
            _append_native_probe(Path(path), output)
        return output

    observed_schedule._simllm_conformance_probe = True  # type: ignore[attr-defined]
    Scheduler.schedule = observed_schedule


def _initialize_vllm(capture_oracle: bool) -> tuple[Any, Any]:
    import vllm.platforms
    from vllm.platforms.cpu import CpuPlatform

    vllm.platforms._current_platform = CpuPlatform()
    from vllm.v1.executor import uniproc_executor

    uniproc_executor.get_ip = lambda: "127.0.0.1"
    uniproc_executor.get_open_port = lambda: 29_500
    from vllm import LLM, SamplingParams

    _install_native_scheduler_probe()
    if capture_oracle:
        register_oracle_hooks()
    return LLM, SamplingParams


@contextmanager
def _engine_environment(
    run_dir: Path,
    *,
    capture_oracle: bool,
    raw_native_path: Path | None,
):
    names = {
        "HF_HUB_OFFLINE": "1",
        "SIMLLM_VLLM_MODE": "virtual",
        "SIMLLM_VLLM_WORKER_MODE": "skeleton",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_CACHE_ROOT": str(run_dir / "vllm-cache"),
        "VLLM_LOGGING_LEVEL": "ERROR",
        "VLLM_TARGET_DEVICE": "cpu",
        "VLLM_USE_V1": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "SIMLLM_CONFORMANCE_NATIVE_PROBE": (
            "" if raw_native_path is None else str(raw_native_path)
        ),
        "SIMLLM_VLLM_ORACLE_CAPTURE": "1" if capture_oracle else "",
        "SIMLLM_VLLM_ORACLE_LOG": (
            str(run_dir / "oracle.jsonl") if capture_oracle else ""
        ),
        "SIMLLM_VLLM_ORACLE_SCOPE": "kv" if capture_oracle else "",
        "VLLM_PLUGINS": "",
    }
    before = {name: os.environ.get(name) for name in names}
    os.environ.update(names)
    try:
        yield
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.exists():
        return ()
    result = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise TypeError(f"{path}:{line_number}: expected an object")
        result.append(value)
    return tuple(result)


def _observe_outputs(
    outputs: Sequence[object],
    served: dict[str, list[int]],
) -> None:
    for output in outputs:
        request_id = str(output.request_id)
        choices = output.outputs
        if len(choices) != 1:
            raise RuntimeError(f"request {request_id} returned multiple choices")
        tokens = tuple(int(token) for token in choices[0].token_ids)
        prior = tuple(served[request_id])
        if tokens[: len(prior)] != prior:
            raise RuntimeError(f"request {request_id} changed an emitted prefix")
        served[request_id].extend(tokens[len(prior) :])


def _resolved_tuple(llm: object, cell: FrozenCell) -> dict[str, Any]:
    vllm_config = llm.llm_engine.vllm_config
    scheduler = vllm_config.scheduler_config
    cache = vllm_config.cache_config
    resolved = (
        scheduler.max_num_scheduled_tokens
        if scheduler.max_num_scheduled_tokens is not None
        else scheduler.max_num_batched_tokens
    )
    return {
        "resolved_max_num_scheduled_tokens": int(resolved),
        "max_num_batched_tokens": int(scheduler.max_num_batched_tokens),
        "max_num_seqs": int(scheduler.max_num_seqs),
        "enable_chunked_prefill": bool(scheduler.enable_chunked_prefill),
        "enable_prefix_caching": bool(cache.enable_prefix_caching),
        "long_prefill_token_threshold": int(
            scheduler.long_prefill_token_threshold
        ),
        "max_model_len": int(vllm_config.model_config.max_model_len),
        "queue_policy": str(scheduler.policy),
        "scheduler_block_size": int(cache.block_size),
        "num_kv_blocks": int(cache.num_gpu_blocks),
        "reserve_mode": (
            "full-isl" if scheduler.scheduler_reserve_full_isl else "none"
        ),
        "watermark": float(scheduler.watermark),
        "prefill_schedule_interval": int(scheduler.prefill_schedule_interval),
        "async_scheduling": bool(scheduler.async_scheduling),
        "pipeline_parallel_size": int(vllm_config.parallel_config.pipeline_parallel_size),
        "tensor_parallel_size": int(vllm_config.parallel_config.tensor_parallel_size),
        "speculative_decoding": vllm_config.speculative_config is not None,
        "lora": vllm_config.lora_config is not None,
        "multimodal": bool(vllm_config.model_config.is_multimodal_model),
        "construction_validation_bypass": bool(
            getattr(llm.llm_engine, "_simllm_chunking_validation_bypass", False)
        ),
        "construction_capacity_validation_bypass": bool(
            getattr(llm.llm_engine, "_simllm_capacity_validation_bypass", False)
        ),
        "cell_id": cell.cell_id,
    }


def _construct_live_engine(
    cell: FrozenCell,
    config: Mapping[str, Any],
    model: Path,
    run_dir: Path,
    *,
    capture: bool,
):
    raw_path = run_dir / "raw_native_steps.jsonl" if capture else None
    LLM, SamplingParams = _initialize_vllm(capture)
    paired_path = run_dir / "native_steps.jsonl"
    steps_path = run_dir / "steps.jsonl"
    reset_configuration()
    configure(
        step_sink=_fixed_pricer(int(config["fixed_engine"]["pricing_step_ps"])),
        config=SimExecutorConfig(
            mode="virtual",
            token_id=int(config["oracle"]["simulated_token_id"]),
            emit_sampled_request_ids=True,
            step_records_path=str(steps_path) if capture else None,
            native_step_capture_path=str(paired_path) if capture else None,
        ),
    )
    engine = cell.engine
    desired_chunking = bool(engine["chunked_prefill"])
    validation_bypass = (
        not desired_chunking
        and int(engine["budget"]) < int(engine["max_model_len"])
    )
    desired_max_model_len = int(engine["max_model_len"])
    construction_max_model_len = min(
        desired_max_model_len,
        int(engine["num_kv_blocks"])
        * int(config["fixed_engine"]["scheduler_block_size"]),
    )
    capacity_validation_bypass = construction_max_model_len < desired_max_model_len
    llm = LLM(
        model=str(model),
        worker_cls="simllm.adapters.vllm.SimWorker",
        enforce_eager=True,
        max_model_len=construction_max_model_len,
        max_num_seqs=int(engine["max_num_seqs"]),
        max_num_batched_tokens=int(engine["budget"]),
        max_num_scheduled_tokens=int(engine["budget"]),
        num_gpu_blocks_override=int(engine["num_kv_blocks"]),
        block_size=int(config["fixed_engine"]["scheduler_block_size"]),
        disable_log_stats=True,
        enable_chunked_prefill=(True if validation_bypass else desired_chunking),
        enable_prefix_caching=bool(engine["prefix_caching"]),
        long_prefill_token_threshold=int(engine["long_prefill_threshold"]),
        scheduler_reserve_full_isl=True,
        watermark=float(config["fixed_engine"]["watermark"]),
        prefill_schedule_interval=1,
        scheduling_policy=str(config["fixed_engine"]["queue_policy"]),
        async_scheduling=False,
        enable_lora=False,
    )
    if validation_bypass:
        # SchedulerConfig rejects this tuple at EngineArgs validation even
        # though the stock scheduler has a frozen stop-at-head rule for it.
        # Construct the empty in-process engine with chunking enabled, then
        # switch the one shared config authority before the first request.
        core = llm.llm_engine.engine_core.engine_core
        if core.scheduler.has_requests():
            raise RuntimeError("chunking validation bypass requires an empty scheduler")
        scheduler_config = llm.llm_engine.vllm_config.scheduler_config
        if core.scheduler.scheduler_config is not scheduler_config:
            raise RuntimeError("engine and scheduler do not share one config authority")
        scheduler_config.enable_chunked_prefill = False
        llm.llm_engine._simllm_chunking_validation_bypass = True
    if capacity_validation_bypass:
        core = llm.llm_engine.engine_core.engine_core
        if core.scheduler.has_requests():
            raise RuntimeError("capacity validation bypass requires an empty scheduler")
        model_config = llm.llm_engine.vllm_config.model_config
        model_config.max_model_len = desired_max_model_len
        core.scheduler.max_model_len = desired_max_model_len
        core.scheduler.kv_cache_manager.max_model_len = desired_max_model_len
        llm.llm_engine._simllm_capacity_validation_bypass = True
    worker = latest_worker()
    if worker is None:
        llm.llm_engine.engine_core.shutdown()
        reset_configuration()
        raise RuntimeError("pinned engine did not construct SimWorker")
    if capture:
        mark_oracle_capture_start(row.request_id for row in cell.requests)
    return llm, worker, SamplingParams, raw_path, paired_path


def _drive_live(
    llm: object,
    worker: object,
    sampling_params: object,
    cell: FrozenCell,
) -> tuple[tuple[str, ...], dict[str, tuple[int, ...]], bool]:
    bookkeeper = _bookkeeper(cell)
    gate = RequestAdmissionGate(
        worker.clock,
        bookkeeper,
        mode=AdmissionMode.ARRIVAL_GATED,
    )
    served = {row.request_id: [] for row in cell.requests}
    requests = {row.request_id: row for row in cell.requests}

    def submit(arrival: object) -> None:
        row = requests[arrival.request_id]
        assigned = llm.llm_engine.add_request(
            row.request_id,
            {"prompt_token_ids": list(row.prompt_token_ids)},
            sampling_params(
                temperature=0.0,
                max_tokens=row.max_output_tokens,
                min_tokens=row.max_output_tokens,
                ignore_eos=True,
                stop=[],
                detokenize=False,
            ),
            arrival_time=row.arrived_at_ps / PS_PER_SECOND,
            priority=row.priority,
        )
        if assigned != row.request_id:
            raise RuntimeError(
                f"vLLM changed request ID {row.request_id!r} to {assigned!r}"
            )

    stopped = False
    if cell.execution_mode == "one-decision":
        gate.admit_ready(submit)
        before = len(worker.step_records)
        outputs = llm.llm_engine.step()
        _observe_outputs(outputs, served)
        stopped = len(worker.step_records) == before
    else:
        while gate.has_pending or llm.llm_engine.has_unfinished_requests():
            gate.admit_ready(submit)
            if llm.llm_engine.has_unfinished_requests():
                outputs = llm.llm_engine.step()
                _observe_outputs(outputs, served)
            elif gate.has_pending:
                gate.advance_to_next_arrival()
    return (
        gate.admitted_request_ids,
        {request_id: tuple(tokens) for request_id, tokens in served.items()},
        stopped,
    )


def run_live_cell(
    cell: FrozenCell,
    config: Mapping[str, Any],
    model: Path,
    evidence_dir: Path,
) -> LiveCellResult:
    evidence_dir.mkdir(parents=True, exist_ok=False)
    llm = None
    worker = None
    raw_path = evidence_dir / "raw_native_steps.jsonl"
    paired_path = evidence_dir / "native_steps.jsonl"
    try:
        with _engine_environment(
            evidence_dir,
            capture_oracle=True,
            raw_native_path=raw_path,
        ):
            llm, worker, sampling_params, _, _ = _construct_live_engine(
                cell, config, model, evidence_dir, capture=True
            )
            causal_tuple = _resolved_tuple(llm, cell)
            admission, outputs, stopped = _drive_live(
                llm, worker, sampling_params, cell
            )
            records = tuple(worker.step_records)
            results = tuple(worker.step_results)
            final_time = int(worker.clock.now_ps)
            llm.llm_engine.engine_core.shutdown()
            llm = None
            reset_configuration()
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        reset_configuration()
    if worker is None:
        raise RuntimeError("live cell ended without a worker")
    geometry = VllmKvGeometry(
        pool_id="conformance:kv",
        block_tokens=int(config["fixed_engine"]["scheduler_block_size"]),
        capacity_blocks=int(cell.engine["num_kv_blocks"]),
        num_layers=int(config["oracle"]["model_geometry"]["num_layers"]),
        num_kv_heads=int(config["oracle"]["model_geometry"]["num_kv_heads"]),
        head_size=int(config["oracle"]["model_geometry"]["head_size"]),
        dtype=str(config["fixed_engine"]["kv_dtype"]),
        dtype_bytes=int(config["oracle"]["model_geometry"]["dtype_bytes"]),
    )
    projection = read_vllm_kv_sidecar(evidence_dir / "oracle.jsonl", geometry)
    _write_once(
        evidence_dir / "kv_projection.json",
        vllm_kv_projection_to_json(projection),
    )
    first_release: dict[str, int] = {}
    for record in records:
        for row in record.scheduled:
            first_release.setdefault(row.request_id, record.virtual_time_ps)
    value = LiveCellResult(
        cell=cell,
        causal_tuple=causal_tuple,
        records=records,
        results=results,
        raw_native=_read_jsonl(raw_path),
        paired_native=_read_jsonl(paired_path),
        kv_operations=projection.operations,
        admission_order=admission,
        output_token_ids=outputs,
        first_release_ps=first_release,
        final_virtual_time_ps=final_time,
        stopped_at_head=stopped,
        evidence_dir=evidence_dir,
    )
    _write_once(evidence_dir / "summary.json", live_summary(value))
    return value


def live_summary(value: LiveCellResult) -> dict[str, Any]:
    return {
        "cell_id": value.cell.cell_id,
        "family": value.cell.family,
        "family_clause": value.cell.family_clause,
        "causal_tuple": dict(value.causal_tuple),
        "records": [step_record_to_json(record) for record in value.records],
        "raw_native": list(value.raw_native),
        "paired_native": list(value.paired_native),
        "kv_operations": [
            _kv_work_json(operation) for operation in value.kv_operations
        ],
        "admission_order": list(value.admission_order),
        "output_token_ids": {
            key: list(tokens) for key, tokens in value.output_token_ids.items()
        },
        "first_release_ps": dict(value.first_release_ps),
        "final_virtual_time_ps": value.final_virtual_time_ps,
        "stopped_at_head": value.stopped_at_head,
    }


def _decision_records(records: Sequence[StepRecord]) -> tuple[StepRecord, ...]:
    return tuple(
        record
        for record in records
        if record.scheduled or record.preempted_request_ids
    )


def _record_signature(record: StepRecord) -> dict[str, Any]:
    return {
        "virtual_time_ps": record.virtual_time_ps,
        "scheduled": [
            {
                "request_id": row.request_id,
                "phase": row.phase.value,
                "num_new_tokens": row.num_new_tokens,
                "num_cached_tokens": row.num_cached_tokens,
                "context_length": row.context_length,
            }
            for row in record.scheduled
        ],
        "total_new_tokens": record.total_new_tokens,
        "preempted_request_ids": list(record.preempted_request_ids),
        "finished_request_ids": list(record.finished_request_ids),
        "num_sampled": record.num_sampled,
        "sampled_request_ids": list(record.sampled_request_ids or ()),
    }


def _native_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    native = row.get("native_scheduler_output", row)
    return {
        "ordered_scheduled_request_ids": list(
            native["ordered_scheduled_request_ids"]
        ),
        "num_scheduled_tokens": list(native["num_scheduled_tokens"]),
        "total_num_scheduled_tokens": int(native["total_num_scheduled_tokens"]),
        "preempted_request_ids": list(native["preempted_request_ids"]),
        "finished_request_ids": list(native["finished_request_ids"]),
    }


def _diff(expected: object, actual: object, path: str = "$") -> list[dict[str, Any]]:
    if type(expected) is not type(actual):
        return [
            {
                "path": path,
                "expected": expected,
                "actual": actual,
                "reason": "type",
            }
        ]
    if isinstance(expected, Mapping):
        result: list[dict[str, Any]] = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                result.append(
                    {
                        "path": f"{path}.{key}",
                        "expected": expected.get(key),
                        "actual": actual.get(key),
                        "reason": "missing-key",
                    }
                )
            else:
                result.extend(_diff(expected[key], actual[key], f"{path}.{key}"))
        return result
    if isinstance(expected, (list, tuple)):
        result = []
        if len(expected) != len(actual):
            result.append(
                {
                    "path": path,
                    "expected": len(expected),
                    "actual": len(actual),
                    "reason": "length",
                }
            )
        for index, (left, right) in enumerate(zip(expected, actual)):
            result.extend(_diff(left, right, f"{path}[{index}]"))
        return result
    if expected != actual:
        return [
            {
                "path": path,
                "expected": expected,
                "actual": actual,
                "reason": "value",
            }
        ]
    return []


def _kv_signature(operation: tuple[str, KvCacheWork]) -> dict[str, Any]:
    _, work = operation
    return {
        "action": work.action.value,
        "request_id": work.request_id,
        "block_ids": list(work.block_ids),
        "token_start": work.token_start,
        "token_end": work.token_end,
    }


def compare_kv_streams(
    live: Sequence[tuple[str, KvCacheWork]],
    surrogate: Sequence[tuple[str, KvCacheWork]],
    witnessed_actions: set[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    live_rows = [
        _kv_signature(operation)
        for operation in live
        if operation[1].action.value in witnessed_actions
    ]
    surrogate_rows = [
        _kv_signature(operation)
        for operation in surrogate
        if operation[1].action.value in witnessed_actions
    ]
    mismatches: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    inverse: dict[str, str] = {}
    if len(live_rows) != len(surrogate_rows):
        mismatches.append(
            {
                "path": "$.kv_operations",
                "expected": len(live_rows),
                "actual": len(surrogate_rows),
                "reason": "length",
            }
        )
    for index, (live_row, surrogate_row) in enumerate(
        zip(live_rows, surrogate_rows)
    ):
        for name in ("action", "request_id", "token_start", "token_end"):
            if live_row[name] != surrogate_row[name]:
                mismatches.append(
                    {
                        "path": f"$.kv_operations[{index}].{name}",
                        "expected": live_row[name],
                        "actual": surrogate_row[name],
                        "reason": "value",
                    }
                )
        live_blocks = live_row["block_ids"]
        surrogate_blocks = surrogate_row["block_ids"]
        if len(live_blocks) != len(surrogate_blocks):
            mismatches.append(
                {
                    "path": f"$.kv_operations[{index}].block_ids",
                    "expected": len(live_blocks),
                    "actual": len(surrogate_blocks),
                    "reason": "length",
                }
            )
        for live_block, surrogate_block in zip(live_blocks, surrogate_blocks):
            prior = mapping.get(live_block)
            reverse = inverse.get(surrogate_block)
            if prior is not None and prior != surrogate_block:
                mismatches.append(
                    {
                        "path": f"$.kv_operations[{index}].block_bijection",
                        "expected": prior,
                        "actual": surrogate_block,
                        "reason": f"live block {live_block} remapped",
                    }
                )
            elif reverse is not None and reverse != live_block:
                mismatches.append(
                    {
                        "path": f"$.kv_operations[{index}].block_bijection",
                        "expected": reverse,
                        "actual": live_block,
                        "reason": f"surrogate block {surrogate_block} reused",
                    }
                )
            else:
                mapping[live_block] = surrogate_block
                inverse[surrogate_block] = live_block
    return mismatches, mapping


def _native_projection_guard(live: LiveCellResult) -> list[dict[str, Any]]:
    raw = [
        _native_signature(row)
        for row in live.raw_native
        if row["total_num_scheduled_tokens"]
        or row["preempted_request_ids"]
        or row["finished_request_ids"]
    ]
    paired = [_native_signature(row) for row in live.paired_native]
    return _diff(raw, paired, "$.native-versus-paired")


def _scheduler_mismatches(
    live: LiveCellResult, surrogate: SurrogateCellResult
) -> list[dict[str, Any]]:
    live_records = [_record_signature(record) for record in _decision_records(live.records)]
    surrogate_records = [
        _record_signature(record) for record in _decision_records(surrogate.records)
    ]
    mismatches = _diff(live_records, surrogate_records, "$.decision_records")
    raw = [
        _native_signature(row)
        for row in live.raw_native
        if row["total_num_scheduled_tokens"] or row["preempted_request_ids"]
    ]
    if len(raw) != len(surrogate_records):
        mismatches.append(
            {
                "path": "$.native-decision-count",
                "expected": len(raw),
                "actual": len(surrogate_records),
                "reason": "length",
            }
        )
    for index, (native, record) in enumerate(zip(raw, surrogate_records)):
        direct = {
            "ordered_scheduled_request_ids": [
                row["request_id"] for row in record["scheduled"]
            ],
            "num_scheduled_tokens": [
                {
                    "request_id": row["request_id"],
                    "num_scheduled_tokens": row["num_new_tokens"],
                }
                for row in record["scheduled"]
            ],
            "total_num_scheduled_tokens": record["total_new_tokens"],
            "preempted_request_ids": record["preempted_request_ids"],
            "finished_request_ids": record["finished_request_ids"],
        }
        mismatches.extend(_diff(native, direct, f"$.native[{index}]"))
    return mismatches


def _token_conservation(
    cell: FrozenCell,
    records: Sequence[StepRecord],
    outputs: Mapping[str, tuple[int, ...]],
    *,
    stopped: bool,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    if stopped:
        if any(record.total_new_tokens for record in records):
            mismatches.append(
                {
                    "path": "$.stopped.total_new_tokens",
                    "expected": 0,
                    "actual": sum(record.total_new_tokens for record in records),
                    "reason": "value",
                }
            )
        return mismatches
    expected_lengths = {row.request_id: row.max_output_tokens for row in cell.requests}
    actual_lengths = {key: len(value) for key, value in outputs.items()}
    mismatches.extend(_diff(expected_lengths, actual_lengths, "$.output-token-counts"))
    for index, record in enumerate(records):
        if record.total_new_tokens != sum(row.num_new_tokens for row in record.scheduled):
            mismatches.append(
                {
                    "path": f"$.records[{index}].total_new_tokens",
                    "expected": sum(row.num_new_tokens for row in record.scheduled),
                    "actual": record.total_new_tokens,
                    "reason": "value",
                }
            )
        if record.sampled_request_ids is None:
            mismatches.append(
                {
                    "path": f"$.records[{index}].sampled_request_ids",
                    "expected": "explicit identities",
                    "actual": None,
                    "reason": "missing",
                }
            )
        elif record.num_sampled != len(record.sampled_request_ids):
            mismatches.append(
                {
                    "path": f"$.records[{index}].num_sampled",
                    "expected": len(record.sampled_request_ids),
                    "actual": record.num_sampled,
                    "reason": "value",
                }
            )
    return mismatches


def _configuration_mismatches(
    cell: FrozenCell,
    config: Mapping[str, Any],
    live: LiveCellResult,
    surrogate: SurrogateCellResult,
) -> list[dict[str, Any]]:
    fixed = config["fixed_engine"]
    engine = cell.engine
    expected = {
        "resolved_max_num_scheduled_tokens": int(engine["budget"]),
        "max_num_seqs": int(engine["max_num_seqs"]),
        "enable_chunked_prefill": bool(engine["chunked_prefill"]),
        "enable_prefix_caching": bool(engine["prefix_caching"]),
        "long_prefill_token_threshold": int(engine["long_prefill_threshold"]),
        "max_model_len": int(engine["max_model_len"]),
        "queue_policy": str(fixed["queue_policy"]),
        "scheduler_block_size": int(fixed["scheduler_block_size"]),
        "num_kv_blocks": int(engine["num_kv_blocks"]),
        "reserve_mode": str(fixed["reserve_mode"]),
        "watermark": float(fixed["watermark"]),
    }
    live_causal = {key: live.causal_tuple[key] for key in expected}
    surrogate_causal = {key: surrogate.causal_tuple[key] for key in expected}
    mismatches = _diff(expected, live_causal, "$.live-causal-tuple")
    mismatches.extend(
        _diff(expected, surrogate_causal, "$.surrogate-causal-tuple")
    )
    inactive = {
        "max_num_batched_tokens": int(engine["budget"]),
        "prefill_schedule_interval": 1,
        "async_scheduling": False,
        "pipeline_parallel_size": 1,
        "tensor_parallel_size": 1,
        "speculative_decoding": False,
        "lora": False,
        "multimodal": False,
    }
    mismatches.extend(
        _diff(
            inactive,
            {key: live.causal_tuple[key] for key in inactive},
            "$.inactive-features",
        )
    )
    expected_bypass = (
        not bool(engine["chunked_prefill"])
        and int(engine["budget"]) < int(engine["max_model_len"])
    )
    mismatches.extend(
        _diff(
            expected_bypass,
            live.causal_tuple["construction_validation_bypass"],
            "$.construction-validation-bypass",
        )
    )
    expected_capacity_bypass = int(engine["max_model_len"]) > (
        int(engine["num_kv_blocks"])
        * int(fixed["scheduler_block_size"])
    )
    mismatches.extend(
        _diff(
            expected_capacity_bypass,
            live.causal_tuple["construction_capacity_validation_bypass"],
            "$.construction-capacity-validation-bypass",
        )
    )
    return mismatches


def _check_row(
    family: str,
    cell_id: str,
    clause: str,
    mismatches: Sequence[Mapping[str, Any]],
    *,
    expected: object,
    actual: object,
    evidence_class: str = "behavioral",
) -> dict[str, Any]:
    return {
        "evidence_class": evidence_class,
        "family": family,
        "cell_id": cell_id,
        "clause": clause,
        "status": "PASS" if not mismatches else "FAIL",
        "expected": expected,
        "actual": actual,
        "mismatch_count": len(mismatches),
        "mismatches": list(mismatches),
    }


def _pricing_config(config: Mapping[str, Any]) -> SerialStepLowererConfig:
    return SerialStepLowererConfig(
        _model_dims(config),
        (0,),
        provider=FixedStepProvider(
            int(config["families"]["F6"]["pricing"]["duration_ps"])
        ),
    )


def _price_records(
    records: Sequence[StepRecord], config: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    clock = VirtualClock()
    sink = DeviceRuntimeStepSink(_pricing_config(config))
    sink.bind_clock(clock)
    results = []
    for record in records:
        if clock.now_ps < record.virtual_time_ps:
            clock.advance_to(record.virtual_time_ps)
        if clock.now_ps != record.virtual_time_ps:
            raise ValueError("record pricing clock passed a release timestamp")
        result = sink(record, None)
        results.append(step_result_to_json(result))
        clock.advance_to(max(clock.now_ps, result.completed_at_ps))
    return tuple(results)


def _kv_accounting_summary(
    operations: Sequence[tuple[str, KvCacheWork]],
) -> dict[str, Any]:
    actions = Counter(work.action.value for _, work in operations)
    tokens = defaultdict(int)
    blocks = defaultdict(int)
    for _, work in operations:
        if work.token_start is not None and work.token_end is not None:
            tokens[work.action.value] += work.token_end - work.token_start
        blocks[work.action.value] += len(work.block_ids)
    return {
        "action_counts": dict(sorted(actions.items())),
        "token_spans": dict(sorted(tokens.items())),
        "block_visits": dict(sorted(blocks.items())),
    }


def _evaluate_f6(
    config: Mapping[str, Any],
    live_by_id: Mapping[str, LiveCellResult],
    surrogate_by_id: Mapping[str, SurrogateCellResult],
) -> list[dict[str, Any]]:
    rows = []
    for cell_id in config["families"]["F6"]["source_cells"]:
        live = live_by_id[cell_id]
        surrogate = surrogate_by_id[cell_id]
        live_records = _decision_records(live.records)
        surrogate_records = _decision_records(surrogate.records)
        first = _price_records(live_records, config)
        replay = _price_records(live_records, config)
        surrogate_priced = _price_records(surrogate_records, config)
        expected = {
            "live_replay": first,
            "live_kv_accounting": _kv_accounting_summary(live.kv_operations),
        }
        actual = {
            "live_replay": replay,
            "surrogate_priced": surrogate_priced,
            "surrogate_kv_accounting": _kv_accounting_summary(
                surrogate.kv_operations
            ),
        }
        mismatches = _diff(first, replay, "$.identical-live-replay")
        mismatches.extend(_diff(first, surrogate_priced, "$.surrogate-pricing"))
        rows.append(
            _check_row(
                "F6",
                cell_id,
                "F6 identical pricing chain and metric reachability",
                mismatches,
                expected=expected,
                actual=actual,
            )
        )
    return rows


def _evaluate_cells(
    config: Mapping[str, Any],
    live_by_id: Mapping[str, LiveCellResult],
    surrogate_by_id: Mapping[str, SurrogateCellResult],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    guards: list[dict[str, Any]] = []
    witnessed = set(config["families"]["F7"]["witnessed_actions"])
    for cell in frozen_cells(config):
        live = live_by_id[cell.cell_id]
        surrogate = surrogate_by_id[cell.cell_id]
        native_guard = _native_projection_guard(live)
        guards.append(
            _check_row(
                "GUARD",
                f"native-output:{cell.cell_id}",
                "native SchedulerOutput retained beside its projection",
                native_guard,
                expected="direct native fields equal paired capture fields",
                actual={"mismatch_count": len(native_guard)},
                evidence_class="fatal-guard",
            )
        )
        config_guard = _configuration_mismatches(cell, config, live, surrogate)
        guards.append(
            _check_row(
                "GUARD",
                f"configuration:{cell.cell_id}",
                "causal tuple, capacity, and inactive features remain pinned",
                config_guard,
                expected="frozen tuple",
                actual={"mismatch_count": len(config_guard)},
                evidence_class="fatal-guard",
            )
        )
        live_tokens = _token_conservation(
            cell, live.records, live.output_token_ids, stopped=live.stopped_at_head
        )
        surrogate_tokens = _token_conservation(
            cell,
            surrogate.records,
            surrogate.output_token_ids,
            stopped=surrogate.stopped_at_head,
        )
        guards.append(
            _check_row(
                "GUARD",
                f"token-conservation:{cell.cell_id}",
                "both authorities conserve scheduled and returned tokens",
                (*live_tokens, *surrogate_tokens),
                expected="all requests reach their frozen output caps or the head stops",
                actual={
                    "live_mismatches": len(live_tokens),
                    "surrogate_mismatches": len(surrogate_tokens),
                },
                evidence_class="fatal-guard",
            )
        )
        mismatches = _scheduler_mismatches(live, surrogate)
        if cell.execution_mode == "one-decision":
            expected_stop = {
                "live_stopped": True,
                "surrogate_stopped": True,
                "surrogate_waiting": ["head", "follower"],
            }
            actual_stop = {
                "live_stopped": live.stopped_at_head,
                "surrogate_stopped": surrogate.stopped_at_head,
                "surrogate_waiting": list(surrogate.waiting_request_ids),
            }
            mismatches.extend(_diff(expected_stop, actual_stop, "$.stop-at-head"))
        if cell.family in {"F3", "F4"}:
            kv_mismatches, mapping = compare_kv_streams(
                live.kv_operations, surrogate.kv_operations, witnessed
            )
            mismatches.extend(kv_mismatches)
        else:
            mapping = {}
        if cell.family == "F5":
            admission_expected = {
                "admission_order": list(live.admission_order),
                "first_release_ps": dict(live.first_release_ps),
                "output_token_counts": {
                    key: len(value) for key, value in live.output_token_ids.items()
                },
            }
            admission_actual = {
                "admission_order": list(surrogate.admission_order),
                "first_release_ps": dict(surrogate.first_release_ps),
                "output_token_counts": {
                    key: len(value)
                    for key, value in surrogate.output_token_ids.items()
                },
            }
            mismatches.extend(
                _diff(admission_expected, admission_actual, "$.arrival-admission")
            )
        if cell.family == "F1":
            f1_guard = []
            if any(record.preempted_request_ids for record in live.records):
                f1_guard.append(
                    {
                        "path": "$.live.preemptions",
                        "expected": 0,
                        "actual": sum(
                            len(record.preempted_request_ids)
                            for record in live.records
                        ),
                        "reason": "F1 allocation guard",
                    }
                )
            if any(record.preempted_request_ids for record in surrogate.records):
                f1_guard.append(
                    {
                        "path": "$.surrogate.preemptions",
                        "expected": 0,
                        "actual": sum(
                            len(record.preempted_request_ids)
                            for record in surrogate.records
                        ),
                        "reason": "F1 allocation guard",
                    }
                )
            guards.append(
                _check_row(
                    "GUARD",
                    f"F1-allocation:{cell.cell_id}",
                    "F1 has no allocation failure or preemption",
                    f1_guard,
                    expected=0,
                    actual=len(f1_guard),
                    evidence_class="fatal-guard",
                )
            )
        rows.append(
            _check_row(
                cell.family,
                cell.cell_id,
                cell.family_clause,
                mismatches,
                expected={
                    "oracle": "live vLLM SchedulerOutput and frozen literals",
                    "decision_steps": len(_decision_records(live.records)),
                },
                actual={
                    "surrogate_decision_steps": len(
                        _decision_records(surrogate.records)
                    ),
                    "live_completion_drains": len(live.records)
                    - len(_decision_records(live.records)),
                    "surrogate_completion_drains": len(surrogate.records)
                    - len(_decision_records(surrogate.records)),
                    "block_bijection": mapping,
                },
            )
        )
    rows.extend(_evaluate_f6(config, live_by_id, surrogate_by_id))
    for cell_id in config["families"]["F7"]["source_cells"]:
        live = live_by_id[cell_id]
        surrogate = surrogate_by_id[cell_id]
        mismatches, mapping = compare_kv_streams(
            live.kv_operations, surrogate.kv_operations, witnessed
        )
        reserve_count = sum(
            work.action is KvCacheAction.RESERVE
            for _, work in surrogate.kv_operations
        )
        rows.append(
            _check_row(
                "F7",
                cell_id,
                "F7 witnessed KV alphabet under one stable block bijection",
                mismatches,
                expected={
                    "witnessed_actions": sorted(witnessed),
                    "live_operation_count": sum(
                        work.action.value in witnessed
                        for _, work in live.kv_operations
                    ),
                },
                actual={
                    "surrogate_operation_count": sum(
                        work.action.value in witnessed
                        for _, work in surrogate.kv_operations
                    ),
                    "surrogate_reserve_rows_unscored": reserve_count,
                    "block_bijection": mapping,
                },
            )
        )
    return rows, guards


def _time_live_once(
    cell: FrozenCell,
    config: Mapping[str, Any],
    model: Path,
    run_dir: Path,
) -> tuple[int, dict[str, Any]]:
    run_dir.mkdir(parents=True, exist_ok=False)
    llm = None
    try:
        with _engine_environment(
            run_dir,
            capture_oracle=False,
            raw_native_path=None,
        ):
            llm, worker, sampling_params, _, _ = _construct_live_engine(
                cell, config, model, run_dir, capture=False
            )
            resolved = _resolved_tuple(llm, cell)
            started = time.perf_counter_ns()
            admission, outputs, stopped = _drive_live(
                llm, worker, sampling_params, cell
            )
            elapsed = time.perf_counter_ns() - started
            summary = {
                "resolved": resolved,
                "admission_count": len(admission),
                "output_token_count": sum(
                    len(tokens) for tokens in outputs.values()
                ),
                "step_count": len(worker.step_records),
                "stopped": stopped,
            }
            llm.llm_engine.engine_core.shutdown()
            llm = None
            reset_configuration()
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        reset_configuration()
    return elapsed, summary


def _time_surrogate_once(
    cell: FrozenCell,
    config: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    loop = SurrogateServingLoop(
        _surrogate_config(cell, config),
        _surrogate_requests(cell, config),
        _pool_spec(cell, config, "conformance:kv"),
        _pricing_stamp(),
    )
    started = time.perf_counter_ns()
    result = loop.run(
        step_sink=_fixed_pricer(int(config["fixed_engine"]["pricing_step_ps"]))
    )
    elapsed = time.perf_counter_ns() - started
    return elapsed, {
        "admission_count": len(result.admission_order),
        "output_token_count": sum(
            len(row.output_token_ids) for row in result.request_results
        ),
        "step_count": len(result.records),
    }


def _machine() -> dict[str, Any]:
    uname = platform.uname()
    return {
        "system": uname.system,
        "release": uname.release,
        "machine": uname.machine,
        "processor": platform.processor() or "undisclosed-by-platform",
        "logical_cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
    }


def _evaluate_wall_time(
    config: Mapping[str, Any],
    model: Path,
    evidence_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cell = wall_cell(config)
    repetitions = int(config["wall_time"]["repetitions"])
    live_times = []
    surrogate_times = []
    live_summaries = []
    surrogate_summaries = []
    for index in range(repetitions):
        elapsed, summary = _time_live_once(
            cell, config, model, evidence_dir / f"live-{index + 1:02d}"
        )
        live_times.append(elapsed)
        live_summaries.append(summary)
    for _ in range(repetitions):
        elapsed, summary = _time_surrogate_once(cell, config)
        surrogate_times.append(elapsed)
        surrogate_summaries.append(summary)
    live_median = int(statistics.median(live_times))
    surrogate_median = int(statistics.median(surrogate_times))
    ratio = surrogate_median / live_median
    maximum = float(config["wall_time"]["maximum_surrogate_to_live_ratio"])
    mismatches = []
    if ratio > maximum:
        mismatches.append(
            {
                "path": "$.surrogate_to_live_ratio",
                "expected": f"<= {maximum}",
                "actual": ratio,
                "reason": "band",
            }
        )
    detail = {
        "live_ns": live_times,
        "surrogate_ns": surrogate_times,
        "live_median_ns": live_median,
        "surrogate_median_ns": surrogate_median,
        "surrogate_to_live_ratio": ratio,
        "live_to_surrogate_speedup": live_median / surrogate_median,
        "maximum_surrogate_to_live_ratio": maximum,
        "repetitions": repetitions,
        "machine": _machine(),
        "live_summaries": live_summaries,
        "surrogate_summaries": surrogate_summaries,
        "construction_inside_timed_region": False,
        "capture_inside_timed_region": False,
    }
    _write_once(evidence_dir / "timing.json", detail)
    row = _check_row(
        "W",
        cell.cell_id,
        cell.family_clause,
        mismatches,
        expected={"surrogate_to_live_ratio": f"<= {maximum}"},
        actual={
            "live_median_ns": live_median,
            "surrogate_median_ns": surrogate_median,
            "surrogate_to_live_ratio": ratio,
        },
        evidence_class="wall-time-band",
    )
    return row, detail


def _mutate_record_signature(value: Mapping[str, Any]) -> dict[str, Any]:
    mutated = json.loads(json.dumps(value))
    for record in mutated:
        if record["scheduled"]:
            record["scheduled"][0]["num_new_tokens"] += 1
            record["total_new_tokens"] += 1
            return mutated
    raise ValueError("mutation control found no scheduled row")


def _mutation_guard(
    config: Mapping[str, Any],
    live_by_id: Mapping[str, LiveCellResult],
    surrogate_by_id: Mapping[str, SurrogateCellResult],
) -> dict[str, Any]:
    source = "f1-budget16-seqs2"
    live_signatures = [
        _record_signature(record)
        for record in _decision_records(live_by_id[source].records)
    ]
    surrogate_signatures = [
        _record_signature(record)
        for record in _decision_records(surrogate_by_id[source].records)
    ]
    mutated_records = _mutate_record_signature(live_signatures)
    record_detected = bool(
        _diff(mutated_records, surrogate_signatures, "$.record-mutation")
    )
    f7_source = "f3-blocks3-seqs2"
    live_ops = list(live_by_id[f7_source].kv_operations)
    if not live_ops:
        raise RuntimeError("KV mutation control found no live operations")
    operation_id, work = live_ops[0]
    mutated_work = KvCacheWork(
        action=(
            KvCacheAction.RECOMPUTE
            if work.action is not KvCacheAction.RECOMPUTE
            else KvCacheAction.ALLOCATE
        ),
        pool_id=work.pool_id,
        request_id=work.request_id,
        block_ids=work.block_ids,
        token_start=work.token_start,
        token_end=work.token_end,
        dtype=work.dtype,
        placement_epoch=work.placement_epoch,
        cause=work.cause,
    )
    live_ops[0] = (operation_id, mutated_work)
    kv_detected = bool(
        compare_kv_streams(
            live_ops,
            surrogate_by_id[f7_source].kv_operations,
            set(config["families"]["F7"]["witnessed_actions"]),
        )[0]
    )
    priced = list(
        _price_records(_decision_records(live_by_id[source].records), config)
    )
    mutated_priced = json.loads(json.dumps(priced))
    mutated_priced[0]["step_latency_ps"] += 1
    pricing_detected = bool(_diff(priced, mutated_priced, "$.pricing-mutation"))
    values = {
        "record_mutation_detected": record_detected,
        "kv_mutation_detected": kv_detected,
        "pricing_mutation_detected": pricing_detected,
    }
    mismatches = [
        {
            "path": f"$.{name}",
            "expected": True,
            "actual": value,
            "reason": "mutation escaped",
        }
        for name, value in values.items()
        if not value
    ]
    return _check_row(
        "GUARD",
        "end-to-end-mutation-controls",
        "record, KV, and pricing corruptions must be detected",
        mismatches,
        expected={name: True for name in values},
        actual=values,
        evidence_class="fatal-guard",
    )


def _git(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args), cwd=REPOSITORY_ROOT, text=True
    ).strip()


def _preflight(config: Mapping[str, Any], model: Path) -> list[dict[str, Any]]:
    rows = []
    chronology_mismatches = []
    try:
        subprocess.run(
            ("git", "merge-base", "--is-ancestor", CONFIG_FREEZE_COMMIT, "HEAD"),
            cwd=REPOSITORY_ROOT,
            check=True,
        )
        last_config_commit = _git("log", "-1", "--format=%H", "--", str(CONFIG_PATH))
        if last_config_commit != CONFIG_FREEZE_COMMIT:
            chronology_mismatches.append(
                {
                    "path": "$.configuration_commit",
                    "expected": CONFIG_FREEZE_COMMIT,
                    "actual": last_config_commit,
                    "reason": "chronology",
                }
            )
        dirty = _git("status", "--porcelain", "--untracked-files=no")
        if dirty:
            chronology_mismatches.append(
                {
                    "path": "$.tracked_worktree",
                    "expected": "clean before first comparison",
                    "actual": dirty.splitlines(),
                    "reason": "chronology",
                }
            )
    except (subprocess.CalledProcessError, OSError) as exc:
        chronology_mismatches.append(
            {
                "path": "$.git",
                "expected": "configuration freeze is an ancestor",
                "actual": repr(exc),
                "reason": "chronology",
            }
        )
    rows.append(
        _check_row(
            "GUARD",
            "chronology",
            "the frozen cells precede the runner and first comparison",
            chronology_mismatches,
            expected=CONFIG_FREEZE_COMMIT,
            actual=_git("rev-parse", "HEAD"),
            evidence_class="fatal-guard",
        )
    )
    source_mismatches = []
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        version = "absent"
    if version != config["oracle"]["version"]:
        source_mismatches.append(
            {
                "path": "$.vllm.version",
                "expected": config["oracle"]["version"],
                "actual": version,
                "reason": "pin",
            }
        )
    if not model.is_dir() or not (model / "config.json").is_file():
        source_mismatches.append(
            {
                "path": "$.model",
                "expected": "local snapshot with config.json",
                "actual": "absent",
                "reason": "pin",
            }
        )
    elif _sha256(model / "config.json") != config["oracle"]["model_config_sha256"]:
        source_mismatches.append(
            {
                "path": "$.model.config_sha256",
                "expected": config["oracle"]["model_config_sha256"],
                "actual": _sha256(model / "config.json"),
                "reason": "pin",
            }
        )
    if version == config["oracle"]["version"]:
        from vllm.v1.core.sched import scheduler

        scheduler_path = Path(scheduler.__file__ or "")
        actual_hash = _sha256(scheduler_path)
        if actual_hash != config["oracle"]["scheduler_sha256"]:
            source_mismatches.append(
                {
                    "path": "$.vllm.scheduler_sha256",
                    "expected": config["oracle"]["scheduler_sha256"],
                    "actual": actual_hash,
                    "reason": "pin",
                }
            )
    rows.append(
        _check_row(
            "GUARD",
            "source-hash",
            "vLLM distribution, scheduler source, and model config match the pin",
            source_mismatches,
            expected={
                "vllm": config["oracle"]["version"],
                "scheduler_sha256": config["oracle"]["scheduler_sha256"],
                "model_config_sha256": config["oracle"]["model_config_sha256"],
            },
            actual={"vllm": version},
            evidence_class="fatal-guard",
        )
    )
    identifiers = [
        row.request_id for cell in frozen_cells(config) for row in cell.requests
    ]
    per_cell_unique = all(
        len(cell.requests) == len({row.request_id for row in cell.requests})
        for cell in frozen_cells(config)
    )
    rows.append(
        _check_row(
            "GUARD",
            "identifier-uniqueness",
            "every cell has explicit unique stable request identities",
            [] if per_cell_unique else [{"path": "$.request_ids", "reason": "duplicate"}],
            expected="unique within every cell",
            actual={"request_rows": len(identifiers), "unique_per_cell": per_cell_unique},
            evidence_class="fatal-guard",
        )
    )
    fixture = REPOSITORY_ROOT / config["families"]["F5"]["fixture"]
    fixture_mismatches = []
    actual_fixture_hash = _sha256(fixture) if fixture.is_file() else "absent"
    if actual_fixture_hash != config["families"]["F5"]["fixture_sha256"]:
        fixture_mismatches.append(
            {
                "path": "$.F5.fixture_sha256",
                "expected": config["families"]["F5"]["fixture_sha256"],
                "actual": actual_fixture_hash,
                "reason": "pin",
            }
        )
    rows.append(
        _check_row(
            "GUARD",
            "F5-frozen-fixture",
            "F5 reuses the byte-locked arrival_admission_v1 source fixture",
            fixture_mismatches,
            expected=config["families"]["F5"]["fixture_sha256"],
            actual=actual_fixture_hash,
            evidence_class="fatal-guard",
        )
    )
    return rows


def _family_tallies(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for family in (*[f"F{index}" for index in range(1, 8)], "W"):
        selected = [row for row in rows if row["family"] == family]
        result[family] = {
            "passed": sum(row["status"] == "PASS" for row in selected),
            "failed": sum(row["status"] == "FAIL" for row in selected),
            "total": len(selected),
        }
    return result


def _evidence_manifest(attempt_dir: Path) -> list[dict[str, Any]]:
    ignored = {
        "portable_record.json",
        "record.sha256",
        "results.csv",
        "RESULTS.draft.md",
        "attempt_completed.json",
    }
    rows = []
    for path in sorted(attempt_dir.rglob("*")):
        if not path.is_file() or path.name in ignored:
            continue
        rows.append(
            {
                "path": path.relative_to(attempt_dir).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"append-only evidence already exists: {path}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "evidence_class",
                "family",
                "cell_id",
                "status",
                "mismatch_count",
                "clause",
                "expected",
                "actual",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "evidence_class": row["evidence_class"],
                    "family": row["family"],
                    "cell_id": row["cell_id"],
                    "status": row["status"],
                    "mismatch_count": row["mismatch_count"],
                    "clause": row["clause"],
                    "expected": json.dumps(row["expected"], sort_keys=True),
                    "actual": json.dumps(row["actual"], sort_keys=True),
                }
            )


def _render_results(
    record: Mapping[str, Any],
    *,
    registry_effect: str,
    limits: str,
) -> str:
    verdict = record["verdict"]
    lines = [
        "# Surrogate conformance result",
        "",
        f"**{verdict['status']}: {verdict['statement']}**",
        "",
        "## What ran",
        "",
        (
            "The frozen F1 through F7 exact families and the W wall-time band ran "
            "against the in-process vLLM 0.27.1 scheduler and the framework-free "
            "surrogate from identical causal tuples and workloads."
        ),
        "",
        "## What came out",
        "",
    ]
    for family, tally in record["family_tallies"].items():
        lines.append(
            f"- {family}: {tally['passed']} passed, {tally['failed']} failed, "
            f"{tally['total']} total."
        )
    wall = record["wall_time"]
    lines.extend(
        [
            "",
            (
                f"W measured a live median of {wall['live_median_ns']} ns and a "
                f"surrogate median of {wall['surrogate_median_ns']} ns over "
                f"{wall['repetitions']} runs. The surrogate-to-live ratio was "
                f"{wall['surrogate_to_live_ratio']:.9f}, or a "
                f"{wall['live_to_surrogate_speedup']:.3f} times speedup, against "
                f"the frozen maximum ratio of "
                f"{wall['maximum_surrogate_to_live_ratio']:.2f}."
            ),
            "",
            "## Row-level findings",
            "",
            "| Family | Cell | Status | Misses | Clause |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in record["checks"]:
        lines.append(
            f"| {row['family']} | `{row['cell_id']}` | {row['status']} | "
            f"{row['mismatch_count']} | {row['clause']} |"
        )
        if row["status"] == "FAIL":
            for mismatch in row["mismatches"]:
                lines.append(
                    f"| {row['family']} finding | `{mismatch['path']}` | FAIL | 1 | "
                    f"expected `{json.dumps(mismatch.get('expected'), sort_keys=True)}`, "
                    f"observed `{json.dumps(mismatch.get('actual'), sort_keys=True)}` |"
                )
    lines.extend(
        [
            "",
            "## Fatal guards",
            "",
            (
                f"The run is {'VOID' if verdict['void'] else 'nonvoid'}. "
                f"{record['guard_tally']['passed']} fatal guards passed and "
                f"{record['guard_tally']['failed']} failed. Fatal guards are not "
                "part of any behavioral denominator."
            ),
            "",
            "| Guard | Status | Misses |",
            "|---|---:|---:|",
        ]
    )
    for row in record["guards"]:
        lines.append(
            f"| `{row['cell_id']}` | {row['status']} | {row['mismatch_count']} |"
        )
    lines.extend(
        [
            "",
            "## What it changes for the project",
            "",
            registry_effect,
            "",
            "## What it does not change",
            "",
            limits,
            "",
            "## Scope and chronology",
            "",
            (
                "Certification, when earned, applies only to the frozen cells, the "
                "declared witnessed KV alphabet, the deterministic synthetic pricing "
                "chain, and vLLM 0.27.1 at scheduler source SHA-256 "
                f"`{record['pin']['scheduler_sha256']}`. It is re-earned at every "
                "framework pin bump."
            ),
            "",
            f"The final pre-run configuration commit is `{CONFIG_FREEZE_COMMIT}`.",
            "",
            (
                "The native SchedulerOutput captures, their paired projections, KV "
                "sidecars, per-cell summaries, and every timing repetition remain in "
                "the append-only bulk attempt named in the tracked record."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_study(run_root: Path, model: Path, attempt_id: str) -> Path:
    config = load_config()
    attempt_dir = run_root / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    _write_once(
        attempt_dir / "attempt_started.json",
        {
            "schema": ATTEMPT_SCHEMA,
            "attempt_id": attempt_id,
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "head_commit": _git("rev-parse", "HEAD"),
            "configuration_commit": CONFIG_FREEZE_COMMIT,
        },
    )
    _write_once(attempt_dir / "study_config.json", config)
    try:
        guards = _preflight(config, model)
        if any(row["status"] == "FAIL" for row in guards):
            raise RuntimeError("preflight fatal guard failed")

        wall_row, wall_detail = _evaluate_wall_time(
            config, model, attempt_dir / "wall_time"
        )
        live_by_id: dict[str, LiveCellResult] = {}
        surrogate_by_id: dict[str, SurrogateCellResult] = {}
        for cell in frozen_cells(config):
            cell_dir = attempt_dir / "cells" / cell.cell_id
            live_by_id[cell.cell_id] = run_live_cell(
                cell, config, model, cell_dir / "live"
            )
            surrogate_by_id[cell.cell_id] = run_surrogate_cell(
                cell, config, cell_dir / "surrogate"
            )
        checks, cell_guards = _evaluate_cells(config, live_by_id, surrogate_by_id)
        guards.extend(cell_guards)
        guards.append(_mutation_guard(config, live_by_id, surrogate_by_id))
        checks.append(wall_row)
        fatal_failed = [row for row in guards if row["status"] == "FAIL"]
        behavioral_failed = [row for row in checks if row["status"] == "FAIL"]
        if fatal_failed:
            verdict = {
                "status": "VOID",
                "void": True,
                "certified": False,
                "statement": (
                    "a fatal guard failed, so no family tally supports certification"
                ),
            }
        elif behavioral_failed:
            verdict = {
                "status": "NOT CERTIFIED",
                "void": False,
                "certified": False,
                "statement": (
                    f"{len(behavioral_failed)} frozen family rows missed and bound "
                    "the surrogate envelope"
                ),
            }
        else:
            verdict = {
                "status": "CERTIFIED",
                "void": False,
                "certified": True,
                "statement": (
                    "the surrogate is a faithful stand-in on the frozen vLLM 0.27.1 "
                    "surface and meets the frozen wall-time class"
                ),
            }
        guard_tally = {
            "passed": sum(row["status"] == "PASS" for row in guards),
            "failed": len(fatal_failed),
            "total": len(guards),
        }
        record = {
            "schema": RESULT_SCHEMA,
            "attempt_id": attempt_id,
            "verdict": verdict,
            "pin": {
                "vllm_version": config["oracle"]["version"],
                "release_commit": config["oracle"]["release_commit"],
                "scheduler_sha256": config["oracle"]["scheduler_sha256"],
                "model_revision": config["oracle"]["model_revision"],
                "model_config_sha256": config["oracle"]["model_config_sha256"],
            },
            "chronology": {
                "expectation_commit": config["expectation_commit"],
                "amendment_commit": config["governing_amendment_commit"],
                "configuration_commit": CONFIG_FREEZE_COMMIT,
                "run_commit": _git("rev-parse", "HEAD"),
            },
            "family_tallies": _family_tallies(checks),
            "guard_tally": guard_tally,
            "checks": checks,
            "guards": guards,
            "wall_time": wall_detail,
            "configuration_sha256": _sha256(CONFIG_PATH),
            "machine": _machine(),
            "evidence_manifest": _evidence_manifest(attempt_dir),
        }
        record_path = attempt_dir / "portable_record.json"
        _write_once(record_path, record)
        _write_once(
            attempt_dir / "record.sha256",
            f"{_sha256(record_path)}  record.json\n".encode(),
        )
        _write_csv(attempt_dir / "results.csv", (*checks, *guards))
        draft = _render_results(
            record,
            registry_effect=(
                "Registry effects are assigned during publication from the observed "
                "findings."
            ),
            limits=(
                "The run does not claim silicon timing, asynchronous scheduling, "
                "speculative decoding, LoRA, multimodal input, pipeline parallelism, "
                "multi-pool serving, or framework pins other than 0.27.1."
            ),
        )
        _write_once(attempt_dir / "RESULTS.draft.md", draft.encode("utf-8"))
        _write_once(
            attempt_dir / "attempt_completed.json",
            {
                "schema": ATTEMPT_SCHEMA,
                "attempt_id": attempt_id,
                "completed_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "verdict": verdict,
                "record_sha256": _sha256(record_path),
            },
        )
    except BaseException as exc:
        _write_once(
            attempt_dir / "attempt_failed.json",
            {
                "schema": ATTEMPT_SCHEMA,
                "attempt_id": attempt_id,
                "failed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    return attempt_dir


def publish_attempt(
    attempt_dir: Path,
    publish_dir: Path,
    *,
    registry_effect: str,
    limits: str,
) -> None:
    record_path = attempt_dir / "portable_record.json"
    checksum = (attempt_dir / "record.sha256").read_text(encoding="utf-8").split()[0]
    if _sha256(record_path) != checksum:
        raise ValueError("attempt record checksum mismatch")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    publish_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        publish_dir / "record.json": record_path.read_bytes(),
        publish_dir / "record.sha256": (
            f"{checksum}  record.json\n".encode()
        ),
        publish_dir / "results.csv": (attempt_dir / "results.csv").read_bytes(),
        publish_dir / "RESULTS.md": _render_results(
            record,
            registry_effect=registry_effect,
            limits=limits,
        ).encode("utf-8"),
    }
    for path, payload in targets.items():
        _write_once(path, payload)


def _default_attempt_id() -> str:
    return time.strftime("attempt-%Y%m%dT%H%M%SZ", time.gmtime())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="execute one append-only attempt")
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--attempt-id", default=_default_attempt_id())
    check = subparsers.add_parser("check", help="validate the frozen configuration")
    check.add_argument("--model", type=Path)
    publish = subparsers.add_parser("publish", help="publish one retained attempt")
    publish.add_argument("--attempt-dir", type=Path, required=True)
    publish.add_argument("--publish-dir", type=Path, default=Path(__file__).parent)
    publish.add_argument("--registry-effect", required=True)
    publish.add_argument("--limits", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "check":
        config = load_config()
        cells = frozen_cells(config)
        summary = {
            "schema": config["schema"],
            "configuration_commit": CONFIG_FREEZE_COMMIT,
            "cell_counts": dict(Counter(cell.family for cell in cells)),
            "wall_repetitions": config["wall_time"]["repetitions"],
        }
        if args.model is not None:
            summary["preflight"] = _preflight(config, args.model)
        print(json.dumps(summary, sort_keys=True))
        return 0
    if args.command == "publish":
        publish_attempt(
            args.attempt_dir,
            args.publish_dir,
            registry_effect=args.registry_effect,
            limits=args.limits,
        )
        return 0
    attempt = run_study(args.run_root, args.model, args.attempt_id)
    record = json.loads((attempt / "portable_record.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "attempt_id": record["attempt_id"],
                "family_tallies": record["family_tallies"],
                "guard_tally": record["guard_tally"],
                "verdict": record["verdict"],
                "wall_time": {
                    key: record["wall_time"][key]
                    for key in (
                        "live_median_ns",
                        "surrogate_median_ns",
                        "surrogate_to_live_ratio",
                    )
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
