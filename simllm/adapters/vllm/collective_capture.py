"""Optional in-situ timing for stock vLLM communicator invocations.

The general plugin is inert unless its explicit gate is enabled. An enabled
CPU run brackets each multi-rank call with ``time.monotonic_ns``. A CUDA run
records events on the calling stream and resolves them after the enclosing
model-runner step. Only global rank zero writes the distributed operation,
so one JSONL row represents one scheduler step rather than one row per rank.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from simllm.adapters.vllm._version import PINNED_VLLM_VERSION
from simllm.adapters.vllm.executor import StepTranslator, translate_scheduler_output
from simllm.core.step import (
    CollectiveServiceCapture,
    CollectiveServiceEnvironment,
    CollectiveServiceInvocation,
    StepRecord,
    step_record_to_json,
)

COLLECTIVE_CAPTURE_ENABLE_ENV = "SIMLLM_VLLM_COLLECTIVE_CAPTURE"
COLLECTIVE_CAPTURE_PATH_ENV = "SIMLLM_VLLM_COLLECTIVE_CAPTURE_PATH"
COLLECTIVE_CAPTURE_SYSTEM_ENV = "SIMLLM_VLLM_COLLECTIVE_SYSTEM"

_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)$")
_active_session: ContextVar[_CaptureSession | None] = ContextVar(
    "simllm_vllm_collective_capture_session", default=None
)
_active_layers: ContextVar[tuple[tuple[int, str], ...]] = ContextVar(
    "simllm_vllm_collective_capture_layers", default=()
)
_hooks_registered = False
_writer_started = False
_translator = StepTranslator(emit_sampled_request_ids=True)
_step_index = 0
_capture_origin_ns = time.monotonic_ns()


def collective_capture_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the stock-live collective timing seam is selected."""

    values = os.environ if env is None else env
    return values.get(COLLECTIVE_CAPTURE_ENABLE_ENV, "") == "1"


def _required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value.strip():
        raise RuntimeError(f"{name} must be a nonblank value")
    return value


def _capture_path() -> Path:
    path = Path(_required_env(COLLECTIVE_CAPTURE_PATH_ENV))
    if not path.is_absolute():
        raise RuntimeError(f"{COLLECTIVE_CAPTURE_PATH_ENV} must be an absolute path")
    return path


def _normalize_backend(value: object) -> str:
    backend = str(value).lower()
    if ":" in backend:
        backend = backend.rsplit(":", 1)[-1]
    if not backend:
        raise RuntimeError("vLLM communicator has no backend identity")
    return backend


@dataclass(frozen=True)
class _TensorMetadata:
    payload_bytes: int
    dtype: str
    element_width_bytes: int
    tensor_shape: tuple[int, ...]
    device_type: str


def _one_tensor_metadata(tensor: Any) -> _TensorMetadata:
    numel = int(tensor.numel())
    width = int(tensor.element_size())
    shape = tuple(int(extent) for extent in tensor.shape)
    if not shape:
        shape = (1,)
    dtype = str(tensor.dtype).removeprefix("torch.")
    device_type = str(tensor.device.type)
    return _TensorMetadata(
        payload_bytes=numel * width,
        dtype=dtype,
        element_width_bytes=width,
        tensor_shape=shape,
        device_type=device_type,
    )


def _tensor_metadata(value: Any) -> _TensorMetadata:
    tensors = list(value) if isinstance(value, (list, tuple)) else [value]
    if not tensors:
        raise ValueError("collective tensor list must not be empty")
    metadata = [_one_tensor_metadata(tensor) for tensor in tensors]
    widths = {entry.element_width_bytes for entry in metadata}
    dtypes = {entry.dtype for entry in metadata}
    devices = {entry.device_type for entry in metadata}
    if len(widths) != 1 or len(dtypes) != 1 or len(devices) != 1:
        raise ValueError("collective tensor list must share dtype, width and device")
    payload_bytes = sum(entry.payload_bytes for entry in metadata)
    width = widths.pop()
    return (
        _TensorMetadata(
            payload_bytes=payload_bytes,
            dtype=dtypes.pop(),
            element_width_bytes=width,
            tensor_shape=(payload_bytes // width,),
            device_type=devices.pop(),
        )
        if len(metadata) > 1
        else metadata[0]
    )


@dataclass
class _PendingInvocation:
    kind: str
    metadata: _TensorMetadata
    world_size: int
    group_tag: str
    backend: str
    layer: tuple[int, str] | None
    host_service_ps: int | None = None
    cuda_start: Any | None = None
    cuda_end: Any | None = None


@dataclass
class _CaptureSession:
    pending: list[_PendingInvocation]

    def resolve(self) -> CollectiveServiceCapture:
        if not self.pending:
            raise RuntimeError("model step executed no captured collectives")
        invocations: list[CollectiveServiceInvocation] = []
        backends = {entry.backend for entry in self.pending}
        device_types = {entry.metadata.device_type for entry in self.pending}
        timers = {
            "cuda-event" if entry.cuda_end is not None else "host-monotonic-ns"
            for entry in self.pending
        }
        if len(backends) != 1 or len(device_types) != 1 or len(timers) != 1:
            raise RuntimeError("one step capture must use one backend, device type and timer")
        for sequence, entry in enumerate(self.pending):
            service_ps = entry.host_service_ps
            if entry.cuda_end is not None:
                entry.cuda_end.synchronize()
                elapsed_ms = float(entry.cuda_start.elapsed_time(entry.cuda_end))
                service_ps = max(1, round(elapsed_ms * 1_000_000_000))
            if service_ps is None:
                raise RuntimeError("collective invocation has no resolved service")
            layer_index = entry.layer[0] if entry.layer is not None else None
            layer_name = entry.layer[1] if entry.layer is not None else None
            invocations.append(
                CollectiveServiceInvocation(
                    sequence=sequence,
                    kind=entry.kind,
                    payload_bytes=entry.metadata.payload_bytes,
                    world_size=entry.world_size,
                    dtype=entry.metadata.dtype,
                    element_width_bytes=entry.metadata.element_width_bytes,
                    tensor_shape=entry.metadata.tensor_shape,
                    group_tag=entry.group_tag,
                    service_ps=service_ps,
                    layer_index=layer_index,
                    layer_name=layer_name,
                )
            )
        return CollectiveServiceCapture(
            environment=CollectiveServiceEnvironment(
                system=_required_env(COLLECTIVE_CAPTURE_SYSTEM_ENV),
                backend=backends.pop(),
                device_type=device_types.pop(),
                framework="vllm",
                framework_version=importlib.metadata.version("vllm"),
                timer=timers.pop(),
            ),
            invocations=tuple(invocations),
        )


def _cuda_events() -> tuple[Any, Any]:
    import torch

    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _timed_collective_call(
    coordinator: Any,
    kind: str,
    original: Callable[..., Any],
    input_: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    session = _active_session.get()
    world_size = int(getattr(coordinator, "world_size", 0))
    global_rank = int(getattr(coordinator, "rank", -1))
    if session is None or world_size <= 1 or global_rank != 0:
        return original(coordinator, input_, *args, **kwargs)

    metadata = _tensor_metadata(input_)
    backend = _normalize_backend(coordinator.torch_distributed_backend)
    group_tag = str(coordinator.unique_name)
    layers = _active_layers.get()
    layer = layers[-1] if layers else None
    if metadata.device_type == "cuda":
        start, end = _cuda_events()
        start.record()
        result = original(coordinator, input_, *args, **kwargs)
        end.record()
        session.pending.append(
            _PendingInvocation(
                kind=kind,
                metadata=metadata,
                world_size=world_size,
                group_tag=group_tag,
                backend=backend,
                layer=layer,
                cuda_start=start,
                cuda_end=end,
            )
        )
        return result

    started_ns = time.monotonic_ns()
    result = original(coordinator, input_, *args, **kwargs)
    service_ps = max(1, (time.monotonic_ns() - started_ns) * 1_000)
    session.pending.append(
        _PendingInvocation(
            kind=kind,
            metadata=metadata,
            world_size=world_size,
            group_tag=group_tag,
            backend=backend,
            layer=layer,
            host_service_ps=service_ps,
        )
    )
    return result


def _append_record(record: StepRecord) -> None:
    global _writer_started
    encoded = (
        json.dumps(
            step_record_to_json(record),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    path = _capture_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_APPEND | os.O_WRONLY
    if not _writer_started:
        flags |= os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError(f"short collective capture write: {written} of {len(encoded)}")
    finally:
        os.close(descriptor)
    _writer_started = True


def _capture_execute(
    original: Callable[..., Any],
    runner: Any,
    scheduler_output: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    global _step_index
    session = _CaptureSession(pending=[])
    token = _active_session.set(session)
    try:
        result = original(runner, scheduler_output, *args, **kwargs)
    finally:
        _active_session.reset(token)
    if not session.pending:
        return result
    translated = translate_scheduler_output(
        _translator,
        scheduler_output,
        step_index=_step_index,
        virtual_time_ps=(time.monotonic_ns() - _capture_origin_ns) * 1_000,
    )
    _append_record(replace(translated.record, collective_service=session.resolve()))
    _step_index += 1
    return result


def _install_layer_hooks(runner: Any) -> None:
    if getattr(runner, "_simllm_collective_layer_hooks", None) is not None:
        return
    model = runner.get_model()
    handles = []
    for name, module in model.named_modules():
        match = _LAYER_PATTERN.search(name)
        if match is None:
            continue
        layer = (int(match.group(1)), name)

        def enter(_module: Any, _args: Sequence[Any], *, layer: tuple[int, str] = layer):
            _active_layers.set((*_active_layers.get(), layer))

        def leave(
            _module: Any,
            _args: Sequence[Any],
            output: Any,
            *,
            layer: tuple[int, str] = layer,
        ) -> Any:
            stack = _active_layers.get()
            if not stack or stack[-1] != layer:
                raise RuntimeError("vLLM collective layer hook stack is inconsistent")
            _active_layers.set(stack[:-1])
            return output

        handles.append(module.register_forward_pre_hook(enter))
        try:
            handles.append(module.register_forward_hook(leave, always_call=True))
        except TypeError:
            handles.append(module.register_forward_hook(leave))
    if not handles:
        raise RuntimeError("vLLM collective capture found no decoder layers")
    runner._simllm_collective_layer_hooks = tuple(handles)


def _install_around(owner: Any, name: str, observer: Callable[..., Any]) -> None:
    original = getattr(owner, name)

    def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
        return observer(original, instance, *args, **kwargs)

    wrapped.__name__ = original.__name__
    wrapped.__doc__ = original.__doc__
    setattr(owner, name, wrapped)


def _observe_load(original: Callable[..., Any], runner: Any, *args: Any, **kwargs: Any) -> Any:
    result = original(runner, *args, **kwargs)
    _install_layer_hooks(runner)
    return result


def _observe_execute(
    original: Callable[..., Any], runner: Any, scheduler_output: Any, *args: Any, **kwargs: Any
) -> Any:
    return _capture_execute(original, runner, scheduler_output, *args, **kwargs)


def register_collective_timing_hooks() -> None:
    """vLLM general-plugin entry point, inert unless explicitly enabled."""

    global _hooks_registered
    if not collective_capture_enabled():
        return
    if _hooks_registered:
        return
    if os.environ.get("SIMLLM_VLLM_WORKER_MODE") == "skeleton" or os.environ.get(
        "SIMLLM_VLLM_MODE"
    ) in {"paced", "virtual"}:
        raise RuntimeError("live collective timing and vLLM simulation are exclusive")
    _capture_path()
    _required_env(COLLECTIVE_CAPTURE_SYSTEM_ENV)
    installed_version = importlib.metadata.version("vllm")
    if installed_version != f"{PINNED_VLLM_VERSION}+cpu" and installed_version != (
        PINNED_VLLM_VERSION
    ):
        raise RuntimeError(
            "live collective timing requires the pinned vLLM distribution "
            f"{PINNED_VLLM_VERSION}, got {installed_version}"
        )

    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.worker.cpu_model_runner import CPUModelRunner
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    for name in (
        "all_reduce",
        "all_gather",
        "all_gatherv",
        "reduce_scatter",
        "reduce_scatterv",
        "gather",
        "broadcast",
    ):
        original = getattr(GroupCoordinator, name)

        def observed(
            instance: Any,
            input_: Any,
            *args: Any,
            _kind: str = name,
            _original: Callable[..., Any] = original,
            **kwargs: Any,
        ) -> Any:
            return _timed_collective_call(instance, _kind, _original, input_, *args, **kwargs)

        observed.__name__ = original.__name__
        observed.__doc__ = original.__doc__
        setattr(GroupCoordinator, name, observed)

    _install_around(CPUModelRunner, "load_model", _observe_load)
    _install_around(GPUModelRunner, "load_model", _observe_load)
    _install_around(GPUModelRunner, "execute_model", _observe_execute)

    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    except ImportError:
        GPUModelRunnerV2 = None
    if GPUModelRunnerV2 is not None:
        _install_around(GPUModelRunnerV2, "load_model", _observe_load)
        _install_around(GPUModelRunnerV2, "execute_model", _observe_execute)
    _hooks_registered = True


__all__ = [
    "COLLECTIVE_CAPTURE_ENABLE_ENV",
    "COLLECTIVE_CAPTURE_PATH_ENV",
    "COLLECTIVE_CAPTURE_SYSTEM_ENV",
    "collective_capture_enabled",
    "register_collective_timing_hooks",
]
