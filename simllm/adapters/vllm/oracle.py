"""Observation-only vLLM hooks for the CPU framework oracle.

The hooks project decisions made by vLLM's stock CPU worker, routed-expert
capturer, scheduler, KV manager, and block pool. They never replace sampling,
dispatch, scheduling, cache allocation, or eviction policy.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from contextvars import ContextVar
from pathlib import Path
from typing import Any

__all__ = [
    "ORACLE_ENABLE_ENV",
    "ORACLE_LOG_ENV",
    "ORACLE_OBSERVATION_SCHEMA",
    "ORACLE_SCOPE_ENV",
    "mark_oracle_capture_start",
    "mark_oracle_request_mapping",
    "mark_oracle_submission_group",
    "oracle_enabled",
    "register_oracle_hooks",
]

ORACLE_ENABLE_ENV = "SIMLLM_VLLM_ORACLE_CAPTURE"
ORACLE_LOG_ENV = "SIMLLM_VLLM_ORACLE_LOG"
ORACLE_SCOPE_ENV = "SIMLLM_VLLM_ORACLE_SCOPE"
ORACLE_OBSERVATION_SCHEMA = "simllm-vllm-framework-observation-v1"

_hooks_registered = False
_worker_qualified = False
_dispatch_qualified = False
_active_dispatch_capturer: Any | None = None
_cpu_layer_ids: dict[int, int] = {}
_active_cpu_layer_id: ContextVar[int | None] = ContextVar(
    "simllm_vllm_cpu_layer_id", default=None
)


def oracle_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the vLLM observation plugin was explicitly selected."""

    values = os.environ if env is None else env
    return values.get(ORACLE_ENABLE_ENV, "") == "1"


def _oracle_scope(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    scope = values.get(ORACLE_SCOPE_ENV, "full") or "full"
    if scope not in {"full", "kv"}:
        raise ValueError(f"{ORACLE_SCOPE_ENV} must be full or kv, got {scope!r}")
    return scope


def _log_path() -> Path:
    value = os.environ.get(ORACLE_LOG_ENV, "")
    if not value:
        raise RuntimeError(f"{ORACLE_LOG_ENV} must name the oracle sidecar")
    path = Path(value)
    if not path.is_absolute():
        raise RuntimeError(f"{ORACLE_LOG_ENV} must be an absolute path")
    return path


def _emit(kind: str, **fields: Any) -> None:
    payload = {
        "schema": ORACLE_OBSERVATION_SCHEMA,
        "kind": kind,
        **fields,
    }
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    path = _log_path()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError(f"short oracle sidecar write: {written} of {len(encoded)}")
    finally:
        os.close(descriptor)


def _request_ids(values: Iterable[str]) -> list[str]:
    normalized = list(values)
    if not normalized or any(
        not isinstance(value, str) or not value for value in normalized
    ):
        raise ValueError("request_ids must contain nonempty strings")
    if len(normalized) != len(set(normalized)):
        raise ValueError("request_ids contain duplicates")
    return normalized


def mark_oracle_capture_start(request_ids: Iterable[str]) -> None:
    """Delimit engine initialization observations from request capture."""

    _emit("capture-start", request_ids=_request_ids(request_ids))


def mark_oracle_submission_group(
    group_index: int, request_ids: Iterable[str]
) -> None:
    """Mark a logical request group before it enters the vLLM engine."""

    if type(group_index) is not int or group_index < 0:
        raise ValueError("group_index must be a nonnegative integer")
    _emit(
        "submission-group-start",
        group_index=group_index,
        request_ids=_request_ids(request_ids),
    )


def mark_oracle_request_mapping(
    group_index: int,
    internal_to_logical: Mapping[str, str],
) -> None:
    """Record the IDs assigned by vLLM to one already-marked group."""

    if type(group_index) is not int or group_index < 0:
        raise ValueError("group_index must be a nonnegative integer")
    mappings = [
        {"internal_request_id": internal, "request_id": logical}
        for internal, logical in internal_to_logical.items()
    ]
    if not mappings:
        raise ValueError("request mapping must not be empty")
    internal_ids = _request_ids(row["internal_request_id"] for row in mappings)
    logical_ids = _request_ids(row["request_id"] for row in mappings)
    _emit(
        "request-mapping",
        group_index=group_index,
        mappings=[
            {"internal_request_id": internal, "request_id": logical}
            for internal, logical in zip(internal_ids, logical_ids, strict=True)
        ],
    )


def _flatten_block_ids(blocks: Any) -> list[int]:
    if blocks is None:
        return []
    get_ids = getattr(blocks, "get_block_ids", None)
    values = get_ids() if callable(get_ids) else blocks
    if values is None:
        return []
    result: list[int] = []
    for group in values:
        result.extend(int(value) for value in group)
    return result


def _block_size(manager: Any) -> int:
    groups = manager.kv_cache_config.kv_cache_groups
    sizes = {int(group.kv_cache_spec.block_size) for group in groups}
    if len(sizes) != 1:
        raise RuntimeError(f"framework oracle requires uniform KV block size: {sizes}")
    return sizes.pop()


def _observe_manager_init(
    original: Callable[..., Any], manager: Any, *args: Any, **kwargs: Any
) -> None:
    original(manager, *args, **kwargs)
    block_size = _block_size(manager)
    num_blocks = int(manager.kv_cache_config.num_blocks)
    _emit(
        "kv-manager-qualified",
        block_size=block_size,
        manager_class=type(manager).__name__,
        num_blocks=num_blocks,
        token_capacity=block_size * num_blocks,
    )


def _observe_prefix_hit(
    original: Callable[..., Any], manager: Any, request: Any
) -> Any:
    result = original(manager, request)
    blocks, token_count, _shared_prefix_boundary = result
    _emit(
        "prefix-hit",
        block_ids=_flatten_block_ids(blocks),
        request_id=str(request.request_id),
        token_count=int(token_count),
    )
    return result


def _observe_allocate_slots(
    original: Callable[..., Any],
    manager: Any,
    request: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    result = original(manager, request, *args, **kwargs)
    block_ids = _flatten_block_ids(result)
    if block_ids:
        block_size = _block_size(manager)
        _emit(
            "allocation",
            block_ids=block_ids,
            request_id=str(request.request_id),
            token_count=len(block_ids) * block_size,
        )
    return result


def _observe_free(
    original: Callable[..., Any], manager: Any, request: Any
) -> Any:
    request_id = str(request.request_id)
    try:
        block_ids = _flatten_block_ids(manager.get_blocks(request_id))
    except KeyError:
        block_ids = []
    result = original(manager, request)
    if block_ids:
        _emit(
            "release",
            block_ids=block_ids,
            request_id=request_id,
            token_count=len(block_ids) * _block_size(manager),
        )
    return result


def _observe_eviction(
    original: Callable[..., Any], pool: Any, block: Any
) -> bool:
    block_id = int(block.block_id)
    result = bool(original(pool, block))
    if result:
        _emit(
            "eviction",
            block_ids=[block_id],
            request_id=None,
            token_count=int(pool.hash_block_size),
            reason="prefix-cache-capacity",
        )
    return result


def _observe_preemption(
    original: Callable[..., Any],
    scheduler: Any,
    request: Any,
    timestamp: float,
    *,
    drop_stale_output: bool = False,
) -> Any:
    request_id = str(request.request_id)
    try:
        block_ids = _flatten_block_ids(
            scheduler.kv_cache_manager.get_blocks(request_id)
        )
    except KeyError:
        block_ids = []
    result = original(
        scheduler,
        request,
        timestamp,
        drop_stale_output=drop_stale_output,
    )
    _emit(
        "preemption",
        block_ids=block_ids,
        request_id=request_id,
        token_count=len(block_ids) * _block_size(scheduler.kv_cache_manager),
        reason="scheduler-recompute",
    )
    return result


def _observe_request_finish(
    original: Callable[..., Any], scheduler: Any, request: Any, *args: Any, **kwargs: Any
) -> Any:
    result = original(scheduler, request, *args, **kwargs)
    _emit(
        "request-final-counters",
        num_preemptions=int(request.num_preemptions),
        request_id=str(request.request_id),
    )
    return result


def _observe_worker_load(
    original: Callable[..., Any], worker: Any, *args: Any, **kwargs: Any
) -> Any:
    global _worker_qualified
    import torch

    cuda_available_before = bool(torch.cuda.is_available())
    cuda_memory_allocated_before = (
        int(torch.cuda.memory_allocated()) if cuda_available_before else 0
    )
    result = original(worker, *args, **kwargs)
    if not _worker_qualified:
        cuda_available_after = bool(torch.cuda.is_available())
        cuda_memory_allocated_after = (
            int(torch.cuda.memory_allocated()) if cuda_available_after else 0
        )
        runner = worker.model_runner
        model = runner.get_model()
        parameter_devices = sorted(
            {str(parameter.device) for parameter in model.parameters()}
        )
        _emit(
            "worker-qualified",
            cuda_available_after=cuda_available_after,
            cuda_available_before=cuda_available_before,
            cuda_memory_allocated_after=cuda_memory_allocated_after,
            cuda_memory_allocated_before=cuda_memory_allocated_before,
            model_class=type(model).__name__,
            model_runner_class=type(runner).__name__,
            parameter_devices=parameter_devices,
            worker_class=type(worker).__name__,
        )
        _worker_qualified = True
    return result


def _observe_dispatch_capture(
    original: Callable[..., Any], capturer: Any, layer_id: int, topk_ids: Any
) -> Any:
    global _dispatch_qualified
    if not _dispatch_qualified:
        _emit(
            "dispatch-qualified",
            capture_class=type(capturer).__name__,
            capture_source="post-selection-router-output",
            selected_experts_unchanged=True,
        )
        _dispatch_qualified = True
    return original(capturer, layer_id, topk_ids)


def _observe_capturer_init(
    original: Callable[..., Any], runner: Any, *args: Any, **kwargs: Any
) -> Any:
    global _active_dispatch_capturer, _cpu_layer_ids
    result = original(runner, *args, **kwargs)
    from vllm.model_executor.layers.fused_moe.layer import MoERunner

    layers = [module for module in runner.model.modules() if isinstance(module, MoERunner)]
    layer_ids = [int(module.layer_id) for module in layers]
    if not layers or len(layer_ids) != len(set(layer_ids)):
        raise RuntimeError("vLLM CPU oracle could not identify unique MoE layers")
    _cpu_layer_ids = {id(module.routed_experts): layer_id for module, layer_id in zip(layers, layer_ids, strict=True)}
    _active_dispatch_capturer = runner.routed_experts_capturer
    _emit(
        "dispatch-path-qualified",
        capture_source="cpu-monolithic-select-experts-return",
        layer_ids=layer_ids,
        selected_experts_unchanged=True,
    )
    return result


def _observe_cpu_moe_call(
    original: Callable[..., Any], cpu_moe: Any, layer: Any, *args: Any, **kwargs: Any
) -> Any:
    if _active_dispatch_capturer is None:
        return original(cpu_moe, layer, *args, **kwargs)
    layer_id = _cpu_layer_ids.get(id(layer))
    if layer_id is None:
        raise RuntimeError("vLLM CPU oracle saw an unmapped routed-expert layer")
    token = _active_cpu_layer_id.set(layer_id)
    try:
        return original(cpu_moe, layer, *args, **kwargs)
    finally:
        _active_cpu_layer_id.reset(token)


def _observe_cpu_select(
    original: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    result = original(*args, **kwargs)
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("vLLM CPU select_experts returned an unexpected value")
    if _active_dispatch_capturer is None:
        return result
    layer_id = _active_cpu_layer_id.get()
    if layer_id is None:
        raise RuntimeError("vLLM CPU dispatch capture has no active layer context")
    _active_dispatch_capturer.capture(layer_id, result[1])
    return result


def _install_around(owner: Any, name: str, observer: Callable[..., Any]) -> None:
    original = getattr(owner, name)

    def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
        return observer(original, instance, *args, **kwargs)

    wrapped.__name__ = original.__name__
    wrapped.__doc__ = original.__doc__
    setattr(owner, name, wrapped)


def register_oracle_hooks() -> None:
    """vLLM general-plugin entry point, inert unless explicitly enabled."""

    global _hooks_registered
    if not oracle_enabled():
        return
    scope = _oracle_scope()
    if scope == "full" and (
        os.environ.get("SIMLLM_VLLM_WORKER_MODE") == "skeleton"
        or os.environ.get("SIMLLM_VLLM_MODE") in {"paced", "virtual"}
    ):
        raise RuntimeError("vLLM framework observation and simulation are exclusive")
    if _hooks_registered:
        return
    _log_path().parent.mkdir(parents=True, exist_ok=True)

    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler

    for owner, name, observer in (
        (KVCacheManager, "__init__", _observe_manager_init),
        (KVCacheManager, "get_computed_blocks", _observe_prefix_hit),
        (KVCacheManager, "allocate_slots", _observe_allocate_slots),
        (KVCacheManager, "free", _observe_free),
        (BlockPool, "_maybe_evict_cached_block", _observe_eviction),
        (Scheduler, "_preempt_request", _observe_preemption),
        (Scheduler, "_free_request", _observe_request_finish),
    ):
        _install_around(owner, name, observer)
    if scope == "full":
        from vllm.model_executor.layers.fused_moe import cpu_fused_moe
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            RoutedExpertsCapturer,
        )
        from vllm.v1.worker.cpu_worker import CPUWorker
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        for owner, name, observer in (
            (CPUWorker, "load_model", _observe_worker_load),
            (GPUModelRunner, "init_routed_experts_capturer", _observe_capturer_init),
            (cpu_fused_moe.CPUFusedMOE, "__call__", _observe_cpu_moe_call),
            (RoutedExpertsCapturer, "capture", _observe_dispatch_capture),
        ):
            _install_around(owner, name, observer)
        original_select = cpu_fused_moe.select_experts

        def observed_select(*args: Any, **kwargs: Any) -> Any:
            return _observe_cpu_select(original_select, *args, **kwargs)

        cpu_fused_moe.select_experts = observed_select
    _hooks_registered = True
    if scope == "full":
        _emit("plugin-active", process_id=os.getpid())
    else:
        _emit("plugin-active", process_id=os.getpid(), scope=scope)
