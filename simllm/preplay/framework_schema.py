"""In-memory contract for observed serving-framework pre-play traces.

Version 2 records the decisions made by a serving framework. Expert IDs must
come from the post-selection values consumed by its MoE implementation, and KV
events must come from its scheduler-owned cache objects. Version 1 remains the
separate Transformers reconstruction contract in :mod:`simllm.preplay.schema`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from simllm.core._wire import (
    _fail,
    _integer,
    _optional_integer,
    _optional_string,
    _require_tuple,
    _string,
    _validate_unique,
)
from simllm.preplay.schema import (
    ForwardPhase,
    PromptFormat,
    SamplingConfig,
    StopReason,
    validate_sampling_config,
)

FRAMEWORK_PREPLAY_TRACE_SCHEMA = "simllm-preplay-trace-v2"
OBSERVED_DISPATCH_SOURCE = "observed-dispatch"
# No runner emits "granite-model-order" any more: SGL-16 replaced the SGLang
# fallback's model-order surrogate with SGLang's own layer identity. The value
# stays readable so traces captured before that replacement still load.
_DISPATCH_LAYER_MAPPINGS = {"framework-layer-id", "granite-model-order"}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class KvEventKind(str, Enum):
    """Scheduler-owned KV lifecycle events represented by trace v2."""

    ALLOCATION = "allocation"
    PREFIX_HIT = "prefix-hit"
    EVICTION = "eviction"
    PREEMPTION = "preemption"
    RELEASE = "release"


@dataclass(frozen=True, kw_only=True)
class FrameworkTraceProvenance:
    """Serving framework, model, sampler, and dispatch provenance."""

    model_id: str
    model_revision: str
    model_class: str
    dtype: str
    tokenizer_sha256: str
    sampling: SamplingConfig
    capture_host: str
    runner: str
    framework: str
    framework_version: str
    observed_source: str
    authored_against_source: str
    torch_version: str
    device: str
    torch_num_threads: int
    engine_seed: int
    eos_token_id: int
    top_k: int
    expert_count: int
    moe_layer_indices: tuple[int, ...]
    kv_page_size: int
    kv_token_capacity: int
    dispatch_layer_mapping: str
    routing_source: str = OBSERVED_DISPATCH_SOURCE
    schema: str = FRAMEWORK_PREPLAY_TRACE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "moe_layer_indices", tuple(self.moe_layer_indices))


@dataclass(frozen=True, kw_only=True)
class ObservedLayerDispatch:
    """Expert identities selected and consumed at one MoE layer."""

    layer_index: int
    expert_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "expert_ids", tuple(self.expert_ids))


@dataclass(frozen=True, kw_only=True)
class ObservedTokenDispatch:
    """Post-selection dispatch for one executed input token."""

    phase: ForwardPhase
    token_index: int
    token_id: int
    routing: tuple[ObservedLayerDispatch, ...]
    routing_source: str = OBSERVED_DISPATCH_SOURCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "routing", tuple(self.routing))


@dataclass(frozen=True, kw_only=True)
class FrameworkRequestTrace:
    """One request outcome plus the dispatch that produced it."""

    request_id: str
    prompt_sha256: str
    prompt_format: PromptFormat
    input_token_ids: tuple[int, ...]
    max_new_tokens: int
    stop_strings: tuple[str, ...]
    output_text: str
    output_token_ids: tuple[int, ...]
    output_length: int
    stop_reason: StopReason
    matched_stop_string: str | None
    framework_cached_tokens: int
    framework_preemption_count: int
    prefill_dispatch: tuple[ObservedTokenDispatch, ...]
    decode_dispatch: tuple[ObservedTokenDispatch, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_token_ids", tuple(self.input_token_ids))
        object.__setattr__(self, "stop_strings", tuple(self.stop_strings))
        object.__setattr__(self, "output_token_ids", tuple(self.output_token_ids))
        object.__setattr__(self, "prefill_dispatch", tuple(self.prefill_dispatch))
        object.__setattr__(self, "decode_dispatch", tuple(self.decode_dispatch))


@dataclass(frozen=True, kw_only=True)
class KvCacheEvent:
    """One ordered projection of a scheduler-owned KV lifecycle decision."""

    sequence: int
    kind: KvEventKind
    request_id: str | None
    framework_step: int | None
    token_count: int
    block_ids: tuple[int, ...] = ()
    token_slot_ids: tuple[int, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_ids", tuple(self.block_ids))
        object.__setattr__(self, "token_slot_ids", tuple(self.token_slot_ids))


@dataclass(frozen=True, kw_only=True)
class FrameworkPreplayTrace:
    """A complete strict v2 framework trace."""

    provenance: FrameworkTraceProvenance
    requests: tuple[FrameworkRequestTrace, ...]
    kv_events: tuple[KvCacheEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "requests", tuple(self.requests))
        object.__setattr__(self, "kv_events", tuple(self.kv_events))

    def by_request_id(self, request_id: str) -> FrameworkRequestTrace:
        for request in self.requests:
            if request.request_id == request_id:
                return request
        raise KeyError(f"request ID {request_id!r} not in framework trace")


def _sha256(value: object, path: str) -> str:
    digest = _string(value, path)
    if _SHA256_RE.fullmatch(digest) is None:
        _fail(path, "expected 64 lowercase hexadecimal digits")
    return digest


def validate_framework_trace_provenance(
    provenance: FrameworkTraceProvenance,
    path: str = "provenance",
) -> None:
    """Validate framework, model, sampler, dispatch, and KV dimensions."""

    if not isinstance(provenance, FrameworkTraceProvenance):
        _fail(path, "expected FrameworkTraceProvenance")
    if provenance.schema != FRAMEWORK_PREPLAY_TRACE_SCHEMA:
        _fail(f"{path}.schema", f"expected {FRAMEWORK_PREPLAY_TRACE_SCHEMA!r}")
    for field_name in (
        "model_id",
        "model_revision",
        "model_class",
        "dtype",
        "capture_host",
        "runner",
        "framework",
        "framework_version",
        "observed_source",
        "authored_against_source",
        "torch_version",
        "dispatch_layer_mapping",
    ):
        _string(getattr(provenance, field_name), f"{path}.{field_name}")
    _sha256(provenance.tokenizer_sha256, f"{path}.tokenizer_sha256")
    validate_sampling_config(provenance.sampling, f"{path}.sampling")
    if provenance.device != "cpu":
        _fail(f"{path}.device", "framework trace v2 requires cpu")
    _integer(provenance.torch_num_threads, f"{path}.torch_num_threads", minimum=1)
    _integer(provenance.engine_seed, f"{path}.engine_seed", nonnegative=True)
    _integer(provenance.eos_token_id, f"{path}.eos_token_id", nonnegative=True)
    _integer(provenance.top_k, f"{path}.top_k", minimum=1)
    _integer(provenance.expert_count, f"{path}.expert_count", minimum=1)
    if provenance.top_k > provenance.expert_count:
        _fail(f"{path}.top_k", "must not exceed expert_count")
    layers = _require_tuple(provenance.moe_layer_indices, f"{path}.moe_layer_indices")
    if not layers:
        _fail(f"{path}.moe_layer_indices", "must not be empty")
    for index, layer in enumerate(layers):
        _integer(layer, f"{path}.moe_layer_indices[{index}]", nonnegative=True)
    _validate_unique(layers, f"{path}.moe_layer_indices")
    if tuple(sorted(layers)) != layers:
        _fail(f"{path}.moe_layer_indices", "must be strictly increasing")
    _integer(provenance.kv_page_size, f"{path}.kv_page_size", minimum=1)
    _integer(
        provenance.kv_token_capacity,
        f"{path}.kv_token_capacity",
        minimum=1,
    )
    if provenance.kv_token_capacity % provenance.kv_page_size:
        _fail(
            f"{path}.kv_token_capacity",
            "must be divisible by kv_page_size",
        )
    if provenance.dispatch_layer_mapping not in _DISPATCH_LAYER_MAPPINGS:
        _fail(
            f"{path}.dispatch_layer_mapping",
            f"expected one of {sorted(_DISPATCH_LAYER_MAPPINGS)!r}",
        )
    if provenance.routing_source != OBSERVED_DISPATCH_SOURCE:
        _fail(
            f"{path}.routing_source",
            f"expected {OBSERVED_DISPATCH_SOURCE!r}",
        )


def _validate_token_dispatch(
    dispatch: ObservedTokenDispatch,
    provenance: FrameworkTraceProvenance,
    *,
    phase: ForwardPhase,
    token_index: int,
    token_id: int,
    path: str,
) -> None:
    if not isinstance(dispatch, ObservedTokenDispatch):
        _fail(path, "expected ObservedTokenDispatch")
    if dispatch.routing_source != OBSERVED_DISPATCH_SOURCE:
        _fail(f"{path}.routing_source", "must identify observed dispatch")
    if dispatch.phase is not phase:
        _fail(f"{path}.phase", f"expected {phase.value!r}")
    if dispatch.token_index != token_index:
        _fail(f"{path}.token_index", f"expected {token_index}")
    if dispatch.token_id != token_id:
        _fail(f"{path}.token_id", f"expected {token_id}")
    routing = _require_tuple(dispatch.routing, f"{path}.routing")
    if len(routing) != len(provenance.moe_layer_indices):
        _fail(
            f"{path}.routing",
            f"expected {len(provenance.moe_layer_indices)} layer rows",
        )
    for layer_position, (route, layer_index) in enumerate(
        zip(routing, provenance.moe_layer_indices, strict=True)
    ):
        route_path = f"{path}.routing[{layer_position}]"
        if not isinstance(route, ObservedLayerDispatch):
            _fail(route_path, "expected ObservedLayerDispatch")
        if route.layer_index != layer_index:
            _fail(f"{route_path}.layer_index", f"expected {layer_index}")
        expert_ids = _require_tuple(route.expert_ids, f"{route_path}.expert_ids")
        if len(expert_ids) != provenance.top_k:
            _fail(
                f"{route_path}.expert_ids",
                f"expected top_k={provenance.top_k} values",
            )
        for expert_position, expert_id in enumerate(expert_ids):
            _integer(
                expert_id,
                f"{route_path}.expert_ids[{expert_position}]",
                nonnegative=True,
            )
            if expert_id >= provenance.expert_count:
                _fail(
                    f"{route_path}.expert_ids[{expert_position}]",
                    f"must be less than expert_count={provenance.expert_count}",
                )
        _validate_unique(expert_ids, f"{route_path}.expert_ids")


def validate_framework_request_trace(
    request: FrameworkRequestTrace,
    provenance: FrameworkTraceProvenance,
    path: str = "request",
) -> None:
    """Validate a request outcome and every required observed dispatch row."""

    if not isinstance(request, FrameworkRequestTrace):
        _fail(path, "expected FrameworkRequestTrace")
    _string(request.request_id, f"{path}.request_id")
    _sha256(request.prompt_sha256, f"{path}.prompt_sha256")
    if not isinstance(request.prompt_format, PromptFormat):
        _fail(f"{path}.prompt_format", "expected PromptFormat")
    input_ids = _require_tuple(request.input_token_ids, f"{path}.input_token_ids")
    if not input_ids:
        _fail(f"{path}.input_token_ids", "must not be empty")
    for index, token_id in enumerate(input_ids):
        _integer(token_id, f"{path}.input_token_ids[{index}]", nonnegative=True)
    _integer(request.max_new_tokens, f"{path}.max_new_tokens", minimum=1)
    stop_strings = _require_tuple(request.stop_strings, f"{path}.stop_strings")
    for index, stop_string in enumerate(stop_strings):
        _string(stop_string, f"{path}.stop_strings[{index}]")
    _validate_unique(stop_strings, f"{path}.stop_strings")
    _string(request.output_text, f"{path}.output_text", nonblank=False)
    output_ids = _require_tuple(request.output_token_ids, f"{path}.output_token_ids")
    if not output_ids:
        _fail(f"{path}.output_token_ids", "must not be empty")
    for index, token_id in enumerate(output_ids):
        _integer(token_id, f"{path}.output_token_ids[{index}]", nonnegative=True)
    _integer(request.output_length, f"{path}.output_length", minimum=1)
    if request.output_length != len(output_ids):
        _fail(f"{path}.output_length", "must equal output_token_ids length")
    if request.output_length > request.max_new_tokens:
        _fail(f"{path}.output_length", "must not exceed max_new_tokens")
    if not isinstance(request.stop_reason, StopReason):
        _fail(f"{path}.stop_reason", "expected StopReason")
    _optional_string(request.matched_stop_string, f"{path}.matched_stop_string")
    _integer(
        request.framework_cached_tokens,
        f"{path}.framework_cached_tokens",
        nonnegative=True,
    )
    _integer(
        request.framework_preemption_count,
        f"{path}.framework_preemption_count",
        nonnegative=True,
    )
    if request.stop_reason is StopReason.LENGTH_CAP:
        if request.output_length != request.max_new_tokens:
            _fail(f"{path}.stop_reason", "length-cap must reach max_new_tokens")
        if request.matched_stop_string is not None:
            _fail(f"{path}.matched_stop_string", "must be null for length-cap")
    elif request.stop_reason is StopReason.EOS:
        if output_ids[-1] != provenance.eos_token_id:
            _fail(f"{path}.stop_reason", "eos output must end in eos_token_id")
        if request.matched_stop_string is not None:
            _fail(f"{path}.matched_stop_string", "must be null for eos")
    elif request.stop_reason is StopReason.STOP_STRING:
        if request.matched_stop_string not in stop_strings:
            _fail(
                f"{path}.matched_stop_string",
                "must name one configured stop string",
            )
        if request.matched_stop_string not in request.output_text:
            _fail(
                f"{path}.matched_stop_string",
                "must occur in output_text",
            )

    prefill = _require_tuple(request.prefill_dispatch, f"{path}.prefill_dispatch")
    if len(prefill) != len(input_ids):
        _fail(
            f"{path}.prefill_dispatch",
            "must contain one row per input token",
        )
    for index, (dispatch, token_id) in enumerate(zip(prefill, input_ids, strict=True)):
        _validate_token_dispatch(
            dispatch,
            provenance,
            phase=ForwardPhase.PREFILL,
            token_index=index,
            token_id=token_id,
            path=f"{path}.prefill_dispatch[{index}]",
        )

    decode = _require_tuple(request.decode_dispatch, f"{path}.decode_dispatch")
    expected_decode = output_ids[:-1]
    if len(decode) != len(expected_decode):
        _fail(
            f"{path}.decode_dispatch",
            "must contain output_token_ids without the terminal token",
        )
    for index, (dispatch, token_id) in enumerate(zip(decode, expected_decode, strict=True)):
        _validate_token_dispatch(
            dispatch,
            provenance,
            phase=ForwardPhase.DECODE,
            token_index=index,
            token_id=token_id,
            path=f"{path}.decode_dispatch[{index}]",
        )


def validate_kv_cache_event(
    event: KvCacheEvent,
    *,
    request_ids: set[str],
    expected_sequence: int,
    path: str,
) -> None:
    """Validate event ordering, identity, and kind-specific fields."""

    if not isinstance(event, KvCacheEvent):
        _fail(path, "expected KvCacheEvent")
    if event.sequence != expected_sequence:
        _fail(f"{path}.sequence", f"expected {expected_sequence}")
    if not isinstance(event.kind, KvEventKind):
        _fail(f"{path}.kind", "expected KvEventKind")
    if event.request_id is not None:
        _string(event.request_id, f"{path}.request_id")
        if event.request_id not in request_ids:
            _fail(f"{path}.request_id", "does not name a trace request")
    _optional_integer(event.framework_step, f"{path}.framework_step", nonnegative=True)
    _integer(event.token_count, f"{path}.token_count", nonnegative=True)
    block_ids = _require_tuple(event.block_ids, f"{path}.block_ids")
    slot_ids = _require_tuple(event.token_slot_ids, f"{path}.token_slot_ids")
    for field_name, values in (("block_ids", block_ids), ("token_slot_ids", slot_ids)):
        for index, value in enumerate(values):
            _integer(value, f"{path}.{field_name}[{index}]", nonnegative=True)
        _validate_unique(values, f"{path}.{field_name}")
    _optional_string(event.reason, f"{path}.reason")

    if event.kind is KvEventKind.PREFIX_HIT:
        if event.request_id is None:
            _fail(f"{path}.request_id", "prefix-hit requires a request")
        if event.reason is not None:
            _fail(f"{path}.reason", "must be null for prefix-hit")
        return
    if event.kind is KvEventKind.PREEMPTION:
        if event.request_id is None:
            _fail(f"{path}.request_id", "preemption requires a request")
        if event.reason is None:
            _fail(f"{path}.reason", "preemption requires a reason")
        return
    if event.token_count < 1:
        _fail(f"{path}.token_count", f"{event.kind.value} must affect tokens")
    if not block_ids and not slot_ids:
        _fail(path, f"{event.kind.value} requires block_ids or token_slot_ids")
    if (
        event.kind in (KvEventKind.ALLOCATION, KvEventKind.RELEASE)
        and event.request_id is None
    ):
        _fail(f"{path}.request_id", f"{event.kind.value} requires a request")
    if event.kind is not KvEventKind.EVICTION and event.reason is not None:
        _fail(f"{path}.reason", f"must be null for {event.kind.value}")


def validate_framework_preplay_trace(
    trace: FrameworkPreplayTrace,
    path: str = "trace",
) -> None:
    """Validate one complete serving-framework trace."""

    if not isinstance(trace, FrameworkPreplayTrace):
        _fail(path, "expected FrameworkPreplayTrace")
    validate_framework_trace_provenance(trace.provenance, f"{path}.provenance")
    requests = _require_tuple(trace.requests, f"{path}.requests")
    if not requests:
        _fail(f"{path}.requests", "must not be empty")
    request_ids: list[str] = []
    for index, request in enumerate(requests):
        validate_framework_request_trace(
            request,
            trace.provenance,
            f"{path}.requests[{index}]",
        )
        request_ids.append(request.request_id)
    _validate_unique(tuple(request_ids), f"{path}.requests.request_id")
    events = _require_tuple(trace.kv_events, f"{path}.kv_events")
    for sequence, event in enumerate(events):
        validate_kv_cache_event(
            event,
            request_ids=set(request_ids),
            expected_sequence=sequence,
            path=f"{path}.kv_events[{sequence}]",
        )
