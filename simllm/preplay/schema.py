"""In-memory contract for ``simllm-preplay-trace-v1``.

The contract contains only standard-library types. Torch and Transformers are
runtime choices of the CPU runner, not dependencies of the trace boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from simllm.core._wire import (
    _enum_value,
    _fail,
    _integer,
    _number,
    _optional_string,
    _require_tuple,
    _string,
    _validate_unique,
)

PREPLAY_TRACE_SCHEMA = "simllm-preplay-trace-v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_WEIGHT_SUM_TOLERANCE = 1e-5


class SamplingMode(str, Enum):
    """Supported deterministic CPU decoding policies."""

    GREEDY = "greedy"
    SEEDED_SAMPLING = "seeded-sampling"


class StopReason(str, Enum):
    """The terminal condition that ended one request."""

    EOS = "eos"
    LENGTH_CAP = "length-cap"
    STOP_STRING = "stop-string"


class PromptFormat(str, Enum):
    """How the runner converted prompt text into model input tokens."""

    TEXT = "text"
    CHAT = "chat"


@dataclass(frozen=True, kw_only=True)
class SamplingConfig:
    """Sampling fields recorded in every trace header."""

    mode: SamplingMode
    seed: int | None = None
    temperature: float | None = None
    top_p: float | None = None

    @classmethod
    def greedy(cls) -> SamplingConfig:
        return cls(mode=SamplingMode.GREEDY)

    @classmethod
    def seeded(
        cls,
        seed: int,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> SamplingConfig:
        return cls(
            mode=SamplingMode.SEEDED_SAMPLING,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
        )


@dataclass(frozen=True, kw_only=True)
class TraceProvenance:
    """Model, sampler, host, and router identity for one trace."""

    model_id: str
    model_revision: str
    model_class: str
    dtype: str
    tokenizer_sha256: str
    sampling: SamplingConfig
    capture_host: str
    runner: str
    transformers_version: str
    torch_version: str
    device: str
    torch_num_threads: int
    eos_token_id: int
    top_k: int
    expert_count: int
    moe_layer_indices: tuple[int, ...]
    schema: str = PREPLAY_TRACE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "moe_layer_indices", tuple(self.moe_layer_indices))


@dataclass(frozen=True, kw_only=True)
class LayerRouting:
    """One generated token's selected experts at one MoE layer."""

    layer_index: int
    expert_ids: tuple[int, ...]
    gate_weights: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "expert_ids", tuple(self.expert_ids))
        object.__setattr__(self, "gate_weights", tuple(self.gate_weights))


@dataclass(frozen=True, kw_only=True)
class TokenTrace:
    """One output token and all routing decisions caused by that token."""

    token_index: int
    token_id: int
    routing: tuple[LayerRouting, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "routing", tuple(self.routing))


@dataclass(frozen=True, kw_only=True)
class RequestTrace:
    """Complete output realization for one stable request identity."""

    request_id: str
    prompt_sha256: str
    prompt_format: PromptFormat
    input_token_ids: tuple[int, ...]
    max_new_tokens: int
    stop_strings: tuple[str, ...]
    output_text: str
    stop_reason: StopReason
    matched_stop_string: str | None
    tokens: tuple[TokenTrace, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_token_ids", tuple(self.input_token_ids))
        object.__setattr__(self, "stop_strings", tuple(self.stop_strings))
        object.__setattr__(self, "tokens", tuple(self.tokens))

    @property
    def output_token_ids(self) -> tuple[int, ...]:
        """Generated token IDs in decode order."""

        return tuple(token.token_id for token in self.tokens)


@dataclass(frozen=True, kw_only=True)
class PreplayTrace:
    """A fully validated trace loaded into memory."""

    provenance: TraceProvenance
    requests: tuple[RequestTrace, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "requests", tuple(self.requests))

    def by_request_id(self, request_id: str) -> RequestTrace:
        for request in self.requests:
            if request.request_id == request_id:
                return request
        raise KeyError(f"request ID {request_id!r} not in pre-play trace")


def _sha256(value: object, path: str) -> str:
    digest = _string(value, path)
    if _SHA256_RE.fullmatch(digest) is None:
        _fail(path, "expected 64 lowercase hexadecimal digits")
    return digest


def validate_sampling_config(config: SamplingConfig, path: str = "provenance.sampling") -> None:
    """Validate mode-specific seed and sampling parameter rules."""

    if not isinstance(config, SamplingConfig):
        _fail(path, "expected SamplingConfig")
    if not isinstance(config.mode, SamplingMode):
        _fail(f"{path}.mode", "expected SamplingMode")

    if config.mode is SamplingMode.GREEDY:
        if config.seed is not None:
            _fail(f"{path}.seed", "must be null in greedy mode")
        if config.temperature is not None:
            _fail(f"{path}.temperature", "must be null in greedy mode")
        if config.top_p is not None:
            _fail(f"{path}.top_p", "must be null in greedy mode")
        return

    if config.seed is None:
        _fail(f"{path}.seed", "is required in seeded-sampling mode")
    _integer(config.seed, f"{path}.seed", nonnegative=True)
    if config.temperature is None:
        _fail(f"{path}.temperature", "is required in seeded-sampling mode")
    temperature = _number(config.temperature, f"{path}.temperature")
    if temperature <= 0:
        _fail(f"{path}.temperature", "must be greater than zero")
    if config.top_p is None:
        _fail(f"{path}.top_p", "is required in seeded-sampling mode")
    top_p = _number(config.top_p, f"{path}.top_p")
    if not 0 < top_p <= 1:
        _fail(f"{path}.top_p", "must be greater than zero and at most one")


def validate_trace_provenance(
    provenance: TraceProvenance,
    path: str = "provenance",
) -> None:
    """Validate trace-wide identity and routing dimensions."""

    if not isinstance(provenance, TraceProvenance):
        _fail(path, "expected TraceProvenance")
    if provenance.schema != PREPLAY_TRACE_SCHEMA:
        _fail(f"{path}.schema", f"unsupported schema {provenance.schema!r}")
    for field_name in (
        "model_id",
        "model_revision",
        "model_class",
        "dtype",
        "capture_host",
        "runner",
        "transformers_version",
        "torch_version",
    ):
        _string(getattr(provenance, field_name), f"{path}.{field_name}")
    _sha256(provenance.tokenizer_sha256, f"{path}.tokenizer_sha256")
    validate_sampling_config(provenance.sampling, f"{path}.sampling")
    if provenance.device != "cpu":
        _fail(f"{path}.device", "pre-play traces require device 'cpu'")
    _integer(provenance.torch_num_threads, f"{path}.torch_num_threads", minimum=1)
    _integer(provenance.eos_token_id, f"{path}.eos_token_id", nonnegative=True)
    top_k = _integer(provenance.top_k, f"{path}.top_k", minimum=1)
    expert_count = _integer(
        provenance.expert_count,
        f"{path}.expert_count",
        minimum=1,
    )
    if top_k > expert_count:
        _fail(f"{path}.top_k", "must not exceed expert_count")
    layers = _require_tuple(provenance.moe_layer_indices, f"{path}.moe_layer_indices")
    if not layers:
        _fail(f"{path}.moe_layer_indices", "must not be empty")
    for index, layer in enumerate(layers):
        _integer(layer, f"{path}.moe_layer_indices[{index}]", nonnegative=True)
    _validate_unique(layers, f"{path}.moe_layer_indices")
    if layers != tuple(sorted(layers)):
        _fail(f"{path}.moe_layer_indices", "must be in increasing order")


def validate_layer_routing(
    routing: LayerRouting,
    provenance: TraceProvenance,
    path: str,
) -> None:
    """Validate one layer decision against trace-wide router dimensions."""

    if not isinstance(routing, LayerRouting):
        _fail(path, "expected LayerRouting")
    _integer(routing.layer_index, f"{path}.layer_index", nonnegative=True)
    expert_ids = _require_tuple(routing.expert_ids, f"{path}.expert_ids")
    if len(expert_ids) != provenance.top_k:
        _fail(
            f"{path}.expert_ids",
            f"expected exactly top_k={provenance.top_k} entries",
        )
    for index, expert_id in enumerate(expert_ids):
        value = _integer(expert_id, f"{path}.expert_ids[{index}]", nonnegative=True)
        if value >= provenance.expert_count:
            _fail(
                f"{path}.expert_ids[{index}]",
                f"must be below expert_count={provenance.expert_count}",
            )
    _validate_unique(expert_ids, f"{path}.expert_ids")

    gate_weights = _require_tuple(routing.gate_weights, f"{path}.gate_weights")
    if len(gate_weights) != provenance.top_k:
        _fail(
            f"{path}.gate_weights",
            f"expected exactly top_k={provenance.top_k} entries",
        )
    total = 0.0
    for index, weight in enumerate(gate_weights):
        total += _number(weight, f"{path}.gate_weights[{index}]", nonnegative=True)
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        _fail(
            f"{path}.gate_weights",
            f"must sum to one within {_WEIGHT_SUM_TOLERANCE:g}; got {total:.12g}",
        )


def validate_request_trace(
    request: RequestTrace,
    provenance: TraceProvenance,
    path: str = "request",
) -> None:
    """Validate one complete request, including stop and routing semantics."""

    if not isinstance(request, RequestTrace):
        _fail(path, "expected RequestTrace")
    _string(request.request_id, f"{path}.request_id")
    _sha256(request.prompt_sha256, f"{path}.prompt_sha256")
    if not isinstance(request.prompt_format, PromptFormat):
        _fail(f"{path}.prompt_format", "expected PromptFormat")
    input_token_ids = _require_tuple(request.input_token_ids, f"{path}.input_token_ids")
    if not input_token_ids:
        _fail(f"{path}.input_token_ids", "must not be empty")
    for index, token_id in enumerate(input_token_ids):
        _integer(token_id, f"{path}.input_token_ids[{index}]", nonnegative=True)
    max_new_tokens = _integer(
        request.max_new_tokens,
        f"{path}.max_new_tokens",
        minimum=1,
    )
    stop_strings = _require_tuple(request.stop_strings, f"{path}.stop_strings")
    for index, stop_string in enumerate(stop_strings):
        _string(stop_string, f"{path}.stop_strings[{index}]")
    _validate_unique(stop_strings, f"{path}.stop_strings")
    _string(request.output_text, f"{path}.output_text", nonblank=False)
    if not isinstance(request.stop_reason, StopReason):
        _fail(f"{path}.stop_reason", "expected StopReason")
    _optional_string(request.matched_stop_string, f"{path}.matched_stop_string")

    tokens = _require_tuple(request.tokens, f"{path}.tokens")
    if not tokens:
        _fail(f"{path}.tokens", "must not be empty")
    if len(tokens) > max_new_tokens:
        _fail(f"{path}.tokens", "output exceeds max_new_tokens")
    for token_index, token in enumerate(tokens):
        token_path = f"{path}.tokens[{token_index}]"
        if not isinstance(token, TokenTrace):
            _fail(token_path, "expected TokenTrace")
        if token.token_index != token_index:
            _fail(
                f"{token_path}.token_index",
                f"expected contiguous index {token_index}",
            )
        _integer(token.token_id, f"{token_path}.token_id", nonnegative=True)
        routing = _require_tuple(token.routing, f"{token_path}.routing")
        layer_indices = tuple(layer.layer_index for layer in routing)
        if layer_indices != provenance.moe_layer_indices:
            _fail(
                f"{token_path}.routing",
                "layer indices must exactly match provenance.moe_layer_indices",
            )
        for route_index, route in enumerate(routing):
            validate_layer_routing(
                route,
                provenance,
                f"{token_path}.routing[{route_index}]",
            )

    matched = request.matched_stop_string
    configured_matches = tuple(value for value in stop_strings if value in request.output_text)
    final_token_id = tokens[-1].token_id
    if request.stop_reason is StopReason.EOS:
        if final_token_id != provenance.eos_token_id:
            _fail(f"{path}.stop_reason", "eos output must end in eos_token_id")
        if matched is not None:
            _fail(f"{path}.matched_stop_string", "must be null for eos")
        if configured_matches:
            _fail(f"{path}.stop_reason", "a configured stop string appeared before EOS")
    elif request.stop_reason is StopReason.LENGTH_CAP:
        if len(tokens) != max_new_tokens:
            _fail(f"{path}.stop_reason", "length-cap output must reach max_new_tokens")
        if final_token_id == provenance.eos_token_id:
            _fail(f"{path}.stop_reason", "EOS takes precedence over length cap")
        if matched is not None:
            _fail(f"{path}.matched_stop_string", "must be null for length-cap")
        if configured_matches:
            _fail(f"{path}.stop_reason", "a configured stop string appeared before length cap")
    else:
        if matched is None:
            _fail(f"{path}.matched_stop_string", "is required for stop-string")
        if matched not in stop_strings:
            _fail(f"{path}.matched_stop_string", "must name a configured stop string")
        if matched not in request.output_text:
            _fail(f"{path}.matched_stop_string", "must appear in output_text")
        if final_token_id == provenance.eos_token_id:
            _fail(f"{path}.stop_reason", "EOS takes precedence over stop-string")


def validate_preplay_trace(trace: PreplayTrace) -> None:
    """Validate a fully materialized trace and unique request identities."""

    if not isinstance(trace, PreplayTrace):
        _fail("trace", "expected PreplayTrace")
    validate_trace_provenance(trace.provenance)
    requests = _require_tuple(trace.requests, "trace.requests")
    request_ids: list[str] = []
    for index, request in enumerate(requests):
        validate_request_trace(request, trace.provenance, f"trace.requests[{index}]")
        request_ids.append(request.request_id)
    _validate_unique(tuple(request_ids), "trace.requests request IDs")


def sampling_mode_from_json(value: object, path: str) -> SamplingMode:
    """Parse a sampling mode through the shared strict enum helper."""

    return _enum_value(SamplingMode, value, path)


def stop_reason_from_json(value: object, path: str) -> StopReason:
    """Parse a stop reason through the shared strict enum helper."""

    return _enum_value(StopReason, value, path)


def prompt_format_from_json(value: object, path: str) -> PromptFormat:
    """Parse a prompt format through the shared strict enum helper."""

    return _enum_value(PromptFormat, value, path)
