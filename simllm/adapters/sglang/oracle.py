"""Observation-only SGLang hooks for the CPU framework oracle.

The hooks never replace scheduling or model execution. They project decisions
from the stock paged allocator, prefix matcher, eviction boundary, and decode
retraction path into an append-only sidecar. SGLang's own
``return_routed_experts`` response remains the dispatch authority.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

__all__ = [
    "ORACLE_ENABLE_ENV",
    "ORACLE_LOG_ENV",
    "ORACLE_OBSERVATION_SCHEMA",
    "mark_oracle_capture_start",
    "mark_oracle_submission_group",
    "oracle_enabled",
    "register_oracle_hooks",
]

ORACLE_ENABLE_ENV = "SIMLLM_SGLANG_ORACLE_CAPTURE"
ORACLE_LOG_ENV = "SIMLLM_SGLANG_ORACLE_LOG"
ORACLE_OBSERVATION_SCHEMA = "simllm-sglang-framework-observation-v1"

_ALLOC_EXTEND_TARGET = "sglang.srt.mem_cache.allocation.alloc_for_extend"
_ALLOC_DECODE_TARGET = "sglang.srt.mem_cache.allocation.alloc_for_decode"
_PREFIX_TARGET = "sglang.srt.mem_cache.radix_cache.RadixCache.match_prefix"
_REMOVE_TARGET = "sglang.srt.mem_cache.events.KVCacheEventMixin._record_remove_event"
_RETRACT_TARGET = "sglang.srt.managers.schedule_batch.ScheduleBatch.retract_decode"
_HOST_CACHE_TARGET = "sglang.srt.state_capturer.base.BaseHostCache.__init__"
_ROUTED_CAPTURE_TARGET = (
    "sglang.srt.state_capturer.routed_experts.RoutedExpertsCapturer.capture"
)
_WORKER_TARGET = (
    "sglang.srt.managers.tp_worker.TpModelWorker.forward_batch_generation"
)

_worker_qualified_emitted = False
_capture_layer_qualified_emitted = False
_capture_next_layer: dict[int, int] = {}


def oracle_enabled(env: Any = None) -> bool:
    """Whether observation-only capture was explicitly requested."""

    env = os.environ if env is None else env
    return env.get(ORACLE_ENABLE_ENV, "") == "1"


def _log_path() -> Path:
    value = os.environ.get(ORACLE_LOG_ENV, "")
    if not value:
        raise RuntimeError(f"{ORACLE_LOG_ENV} must name the oracle sidecar")
    path = Path(value)
    if not path.is_absolute():
        raise RuntimeError(f"{ORACLE_LOG_ENV} must be an absolute path")
    return path


def _jsonable_ints(value: Any) -> list[int]:
    if value is None:
        return []
    current = value
    for method_name in ("detach", "cpu", "flatten"):
        method = getattr(current, method_name, None)
        if callable(method):
            current = method()
    to_list = getattr(current, "tolist", None)
    if callable(to_list):
        current = to_list()
    if isinstance(current, int):
        return [current]
    if not isinstance(current, Iterable) or isinstance(current, (str, bytes)):
        raise TypeError("KV location projection must be an integer iterable")
    result: list[int] = []
    for entry in current:
        if isinstance(entry, Iterable) and not isinstance(entry, (str, bytes)):
            result.extend(_jsonable_ints(entry))
        else:
            result.append(int(entry))
    return result


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


def mark_oracle_capture_start(request_ids: Iterable[str]) -> None:
    """Delimit initialization observations from one request capture."""

    normalized = list(request_ids)
    if not normalized or any(not isinstance(value, str) or not value for value in normalized):
        raise ValueError("request_ids must contain nonempty strings")
    if len(normalized) != len(set(normalized)):
        raise ValueError("request_ids contain duplicates")
    _emit("capture-start", request_ids=normalized)


def mark_oracle_submission_group(
    group_index: int, request_ids: Iterable[str]
) -> None:
    """Mark the exact request group about to enter the framework scheduler."""

    if type(group_index) is not int or group_index < 0:
        raise ValueError("group_index must be a nonnegative integer")
    normalized = list(request_ids)
    if not normalized or any(
        not isinstance(value, str) or not value for value in normalized
    ):
        raise ValueError("request_ids must contain nonempty strings")
    if len(normalized) != len(set(normalized)):
        raise ValueError("request_ids contain duplicates")
    _emit(
        "submission-group-start",
        group_index=group_index,
        request_ids=normalized,
    )


def _request_id(req: Any) -> str:
    value = getattr(req, "rid", None)
    if not isinstance(value, str) or not value:
        raise RuntimeError("SGLang request has no stable rid")
    return value


def _framework_step(batch: Any) -> int | None:
    value = getattr(batch, "forward_iter", None)
    return value if type(value) is int and value >= 0 else None


def _split_locations(
    locations: list[int], lengths: Iterable[int], request_count: int
) -> list[list[int]]:
    normalized = [int(length) for length in lengths]
    if len(normalized) != request_count:
        raise RuntimeError("SGLang allocation length count does not match requests")
    if any(length < 0 for length in normalized):
        raise RuntimeError("SGLang allocation length is negative")
    if sum(normalized) != len(locations):
        raise RuntimeError("SGLang allocation locations do not match requested tokens")
    result = []
    offset = 0
    for length in normalized:
        result.append(locations[offset : offset + length])
        offset += length
    return result


def _observe_alloc_for_extend(
    original: Callable[..., Any], batch: Any
) -> Any:
    result = original(batch)
    out_cache_loc = result[0]
    reqs = list(batch.reqs)
    by_request = _split_locations(
        _jsonable_ints(out_cache_loc), batch.extend_lens, len(reqs)
    )
    for req, slots in zip(reqs, by_request, strict=True):
        if slots:
            _emit(
                "allocation",
                allocation_phase="extend",
                framework_step=_framework_step(batch),
                request_id=_request_id(req),
                token_count=len(slots),
                token_slot_ids=slots,
            )
    return result


def _observe_alloc_for_decode(
    original: Callable[..., Any], batch: Any, token_per_req: int
) -> Any:
    result = original(batch, token_per_req)
    reqs = list(batch.reqs)
    lengths = [int(token_per_req)] * len(reqs)
    by_request = _split_locations(_jsonable_ints(result), lengths, len(reqs))
    for req, slots in zip(reqs, by_request, strict=True):
        if slots:
            _emit(
                "allocation",
                allocation_phase="decode",
                framework_step=_framework_step(batch),
                request_id=_request_id(req),
                token_count=len(slots),
                token_slot_ids=slots,
            )
    return result


def _observe_match_prefix(
    original: Callable[..., Any],
    tree_cache: Any,
    params: Any,
) -> Any:
    result = original(tree_cache, params)
    if getattr(tree_cache, "req_to_token_pool", None) is None:
        return result
    req = getattr(params, "req", None)
    if req is None:
        return result
    device_indices = _jsonable_ints(getattr(result, "device_indices", None))
    host_hit_length = int(getattr(result, "host_hit_length", 0))
    matched = len(device_indices) + host_hit_length
    _emit(
        "prefix-hit",
        framework_step=None,
        request_id=_request_id(req),
        token_count=matched,
        device_token_count=len(device_indices),
        host_token_count=host_hit_length,
        token_slot_ids=device_indices,
    )
    return result


def _observe_remove_event(
    original: Callable[..., Any], cache: Any, node: Any, *args: Any, **kwargs: Any
) -> Any:
    slots = _jsonable_ints(getattr(node, "value", None))
    result = original(cache, node, *args, **kwargs)
    if slots:
        _emit(
            "eviction",
            framework_step=None,
            request_id=None,
            token_count=len(slots),
            token_slot_ids=slots,
            reason="radix-capacity",
        )
    return result


def _request_slots(batch: Any, req: Any) -> list[int]:
    req_pool_idx = getattr(req, "req_pool_idx", None)
    kv = getattr(req, "kv", None)
    if req_pool_idx is None or kv is None:
        return []
    length = int(getattr(kv, "kv_allocated_len", 0))
    if length <= 0:
        return []
    pool = batch.req_to_token_pool.req_to_token
    return _jsonable_ints(pool[req_pool_idx, :length])


def _observe_retract_decode(
    original: Callable[..., Any], batch: Any, server_args: Any
) -> Any:
    snapshots = {
        _request_id(req): _request_slots(batch, req) for req in list(batch.reqs)
    }
    result = original(batch, server_args)
    retracted_reqs = result[0]
    for req in retracted_reqs:
        request_id = _request_id(req)
        slots = snapshots.get(request_id, [])
        _emit(
            "preemption",
            framework_step=_framework_step(batch),
            request_id=request_id,
            token_count=len(slots),
            token_slot_ids=slots,
            reason="decode-pressure",
        )
        if slots:
            _emit(
                "release",
                framework_step=_framework_step(batch),
                request_id=request_id,
                token_count=len(slots),
                token_slot_ids=slots,
            )
    return result


def _observe_worker_forward(
    original: Callable[..., Any], worker: Any, *args: Any, **kwargs: Any
) -> Any:
    global _worker_qualified_emitted
    if not _worker_qualified_emitted:
        model_runner = getattr(worker, "model_runner", None)
        model = getattr(model_runner, "model", None)
        parameters = getattr(model, "parameters", None)
        devices = []
        if callable(parameters):
            devices = sorted({str(parameter.device) for parameter in parameters()})
        _emit(
            "worker-qualified",
            model_class=(None if model is None else type(model).__name__),
            model_runner_class=(
                None if model_runner is None else type(model_runner).__name__
            ),
            parameter_devices=devices,
            worker_class=type(worker).__name__,
        )
        _worker_qualified_emitted = True
    return original(worker, *args, **kwargs)


def _initialize_cpu_host_cache(
    original: Callable[..., Any],
    cache: Any,
    num_tokens: int,
    num_layers: int,
    topk_size: int,
    name: str,
    *,
    _torch_module: Any = None,
) -> None:
    """Use ordinary CPU storage for SGLang's CPU-only dispatch capturer.

    The upstream host cache always requests pinned memory. PyTorch initializes
    the CUDA pinned allocator for that request even when both the model and
    capture device are CPU. An ordinary CPU tensor preserves every captured
    expert ID and changes only asynchronous device-to-host transfer support,
    which the CPU engine does not use.
    """

    del original
    if os.environ.get("SGLANG_USE_CPU_ENGINE") != "1":
        raise RuntimeError("the unpinned host-cache hook is CPU-engine only")
    torch = (
        importlib.import_module("torch") if _torch_module is None else _torch_module
    )
    cache.buffer = torch.zeros(
        (num_tokens, num_layers, topk_size),
        dtype=torch.int32,
        device="cpu",
        pin_memory=False,
    )
    cache.num_layers = num_layers
    cache.topk_size = topk_size
    cache.name = name
    cache._log_allocation()
    _emit(
        "capture-storage-qualified",
        device=str(cache.buffer.device),
        name=name,
        pinned=bool(cache.buffer.is_pinned()),
        shape=list(cache.buffer.shape),
    )


def _observe_routed_capture(
    original: Callable[..., Any],
    capturer: Any,
    layer_id: int | None,
    topk_indices: Any,
) -> Any:
    """Label Granite's actual post-selection rows without recomputing them.

    The pinned SGLang Granite model does not pass its constructor layer ID to
    ``TopK``. Its 24 MoE modules execute in model-list order, so the observer
    assigns missing capture labels in that order and leaves ``topk_indices``
    byte-for-byte untouched.
    """

    global _capture_layer_qualified_emitted
    if layer_id is None:
        identity = id(capturer)
        inferred = _capture_next_layer.get(identity, 0)
        layer_count = int(capturer.num_layers)
        if not 0 <= inferred < layer_count:
            raise RuntimeError("SGLang inferred dispatch layer is out of range")
        _capture_next_layer[identity] = (inferred + 1) % layer_count
        layer_id = inferred
        if not _capture_layer_qualified_emitted:
            _emit(
                "dispatch-layer-qualified",
                layer_count=layer_count,
                mapping="granite-model-order",
                selected_experts_unchanged=True,
            )
            _capture_layer_qualified_emitted = True
    return original(capturer, layer_id, topk_indices)


def register_oracle_hooks() -> None:
    """Register observation-only hooks with SGLang's plugin registry."""

    _log_path().parent.mkdir(parents=True, exist_ok=True)
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for target, hook in (
        (_HOST_CACHE_TARGET, _initialize_cpu_host_cache),
        (_ROUTED_CAPTURE_TARGET, _observe_routed_capture),
        (_ALLOC_EXTEND_TARGET, _observe_alloc_for_extend),
        (_ALLOC_DECODE_TARGET, _observe_alloc_for_decode),
        (_PREFIX_TARGET, _observe_match_prefix),
        (_REMOVE_TARGET, _observe_remove_event),
        (_RETRACT_TARGET, _observe_retract_decode),
        (_WORKER_TARGET, _observe_worker_forward),
    ):
        HookRegistry.register(target, hook, HookType.AROUND)
    _emit(
        "plugin-active",
        hook_targets=[
            _HOST_CACHE_TARGET,
            _ROUTED_CAPTURE_TARGET,
            _ALLOC_EXTEND_TARGET,
            _ALLOC_DECODE_TARGET,
            _PREFIX_TARGET,
            _REMOVE_TARGET,
            _RETRACT_TARGET,
            _WORKER_TARGET,
        ],
        process_id=os.getpid(),
    )
