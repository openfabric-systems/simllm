"""Strict joined projection of captured per-token expert assignments."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simllm.core._wire import (
    _array,
    _fields,
    _integer,
    _object,
    _string,
)
from simllm.preplay.join import PreplayReplayRun, validate_preplay_replay_run
from simllm.preplay.schema import PREPLAY_TRACE_SCHEMA, ForwardPhase
from simllm.preplay.trace import read_preplay_trace

ROUTED_EXPERTS_SCHEMA = "simllm-routed-experts-v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, kw_only=True)
class RoutedLayer:
    """One MoE layer's selected expert identities for an input token."""

    layer_index: int
    expert_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "expert_ids", tuple(self.expert_ids))


@dataclass(frozen=True, kw_only=True)
class RoutedToken:
    """One input token that executed a prefill or decode forward pass."""

    phase: ForwardPhase
    token_index: int
    token_id: int
    layers: tuple[RoutedLayer, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))


@dataclass(frozen=True, kw_only=True)
class RoutedRequest:
    """Joined request identity plus only the routing inputs consumers need."""

    request_id: str
    prompt_token_count: int
    output_token_count: int
    tokens: tuple[RoutedToken, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", tuple(self.tokens))

    @property
    def prefill_tokens(self) -> tuple[RoutedToken, ...]:
        return tuple(token for token in self.tokens if token.phase is ForwardPhase.PREFILL)

    @property
    def decode_tokens(self) -> tuple[RoutedToken, ...]:
        return tuple(token for token in self.tokens if token.phase is ForwardPhase.DECODE)


@dataclass(frozen=True, kw_only=True)
class RoutedExperts:
    """Versioned per-token assignment projection of one joined pre-play run."""

    trace_schema: str
    trace_sha256: str
    expert_count: int
    top_k: int
    moe_layer_indices: tuple[int, ...]
    requests: tuple[RoutedRequest, ...]
    schema: str = ROUTED_EXPERTS_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "moe_layer_indices", tuple(self.moe_layer_indices))
        object.__setattr__(self, "requests", tuple(self.requests))

    def by_request_id(self, request_id: str) -> RoutedRequest:
        """Return one request by its stable joined identity."""

        for request in self.requests:
            if request.request_id == request_id:
                return request
        raise KeyError(f"request ID {request_id!r} not in routed-experts projection")


def _require_sha256(value: object, path: str) -> str:
    digest = _string(value, path)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{path}: expected 64 lowercase hexadecimal digits")
    return digest


def _validate_layer(
    layer: RoutedLayer,
    *,
    layer_index: int,
    top_k: int,
    expert_count: int,
    path: str,
) -> None:
    if not isinstance(layer, RoutedLayer):
        raise TypeError(f"{path}: expected RoutedLayer")
    if layer.layer_index != layer_index:
        raise ValueError(f"{path}.layer_index: expected {layer_index}")
    if not isinstance(layer.expert_ids, tuple):
        raise TypeError(f"{path}.expert_ids: in-memory contract requires a tuple")
    if len(layer.expert_ids) != top_k:
        raise ValueError(f"{path}.expert_ids: expected {top_k} expert IDs")
    for index, expert_id in enumerate(layer.expert_ids):
        _integer(expert_id, f"{path}.expert_ids[{index}]", nonnegative=True)
        if expert_id >= expert_count:
            raise ValueError(
                f"{path}.expert_ids[{index}]: expert ID {expert_id} is outside "
                f"[0, {expert_count})"
            )
    if len(layer.expert_ids) != len(set(layer.expert_ids)):
        raise ValueError(f"{path}.expert_ids: contains duplicate values")


def _validate_token(
    token: RoutedToken,
    *,
    phase: ForwardPhase,
    token_index: int,
    layer_indices: tuple[int, ...],
    top_k: int,
    expert_count: int,
    path: str,
) -> None:
    if not isinstance(token, RoutedToken):
        raise TypeError(f"{path}: expected RoutedToken")
    if token.phase is not phase:
        raise ValueError(f"{path}.phase: expected {phase.value!r}")
    if token.token_index != token_index:
        raise ValueError(f"{path}.token_index: expected contiguous index {token_index}")
    _integer(token.token_id, f"{path}.token_id", nonnegative=True)
    if not isinstance(token.layers, tuple):
        raise TypeError(f"{path}.layers: in-memory contract requires a tuple")
    if len(token.layers) != len(layer_indices):
        raise ValueError(f"{path}.layers: expected {len(layer_indices)} layers")
    for index, (layer, layer_index) in enumerate(
        zip(token.layers, layer_indices, strict=True)
    ):
        _validate_layer(
            layer,
            layer_index=layer_index,
            top_k=top_k,
            expert_count=expert_count,
            path=f"{path}.layers[{index}]",
        )


def _validate_request(
    request: RoutedRequest,
    *,
    layer_indices: tuple[int, ...],
    top_k: int,
    expert_count: int,
    path: str,
) -> None:
    if not isinstance(request, RoutedRequest):
        raise TypeError(f"{path}: expected RoutedRequest")
    _string(request.request_id, f"{path}.request_id")
    prompt_count = _integer(
        request.prompt_token_count,
        f"{path}.prompt_token_count",
        minimum=1,
    )
    output_count = _integer(
        request.output_token_count,
        f"{path}.output_token_count",
        minimum=1,
    )
    if not isinstance(request.tokens, tuple):
        raise TypeError(f"{path}.tokens: in-memory contract requires a tuple")
    expected_count = prompt_count + output_count - 1
    if len(request.tokens) != expected_count:
        raise ValueError(f"{path}.tokens: expected {expected_count} forwarded tokens")
    for index, token in enumerate(request.tokens[:prompt_count]):
        _validate_token(
            token,
            phase=ForwardPhase.PREFILL,
            token_index=index,
            layer_indices=layer_indices,
            top_k=top_k,
            expert_count=expert_count,
            path=f"{path}.tokens[{index}]",
        )
    for decode_index, token in enumerate(request.tokens[prompt_count:]):
        index = prompt_count + decode_index
        _validate_token(
            token,
            phase=ForwardPhase.DECODE,
            token_index=decode_index,
            layer_indices=layer_indices,
            top_k=top_k,
            expert_count=expert_count,
            path=f"{path}.tokens[{index}]",
        )


def validate_routed_experts(value: RoutedExperts) -> None:
    """Validate schema identity, request association and routing shape."""

    if not isinstance(value, RoutedExperts):
        raise TypeError("projection: expected RoutedExperts")
    if value.schema != ROUTED_EXPERTS_SCHEMA:
        raise ValueError(f"projection.schema: unsupported schema {value.schema!r}")
    if value.trace_schema != PREPLAY_TRACE_SCHEMA:
        raise ValueError(
            "projection.trace_schema: unsupported schema "
            f"{value.trace_schema!r}"
        )
    _require_sha256(value.trace_sha256, "projection.trace_sha256")
    expert_count = _integer(
        value.expert_count,
        "projection.expert_count",
        minimum=1,
    )
    if expert_count > 256:
        raise ValueError(
            "projection.expert_count: uint8 routing supports at most 256 experts"
        )
    top_k = _integer(value.top_k, "projection.top_k", minimum=1)
    if top_k > expert_count:
        raise ValueError("projection.top_k: cannot exceed expert_count")
    if not isinstance(value.moe_layer_indices, tuple):
        raise TypeError(
            "projection.moe_layer_indices: in-memory contract requires a tuple"
        )
    if not value.moe_layer_indices:
        raise ValueError("projection.moe_layer_indices: must not be empty")
    for index, layer_index in enumerate(value.moe_layer_indices):
        _integer(
            layer_index,
            f"projection.moe_layer_indices[{index}]",
            nonnegative=True,
        )
    if tuple(sorted(set(value.moe_layer_indices))) != value.moe_layer_indices:
        raise ValueError(
            "projection.moe_layer_indices: must be unique and increasing"
        )
    if not isinstance(value.requests, tuple):
        raise TypeError("projection.requests: in-memory contract requires a tuple")
    if not value.requests:
        raise ValueError("projection.requests: must not be empty")
    request_ids: list[str] = []
    for index, request in enumerate(value.requests):
        _validate_request(
            request,
            layer_indices=value.moe_layer_indices,
            top_k=top_k,
            expert_count=expert_count,
            path=f"projection.requests[{index}]",
        )
        request_ids.append(request.request_id)
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("projection.requests: duplicate request identity")


def _layer_to_json(layer: RoutedLayer) -> dict[str, Any]:
    return {
        "layer_index": layer.layer_index,
        "expert_ids": list(layer.expert_ids),
    }


def _token_to_json(token: RoutedToken) -> dict[str, Any]:
    return {
        "phase": token.phase.value,
        "token_index": token.token_index,
        "token_id": token.token_id,
        "layers": [_layer_to_json(layer) for layer in token.layers],
    }


def routed_experts_to_json(value: RoutedExperts) -> dict[str, Any]:
    """Return the strict JSON object for one routed-experts projection."""

    validate_routed_experts(value)
    return {
        "schema": value.schema,
        "trace_schema": value.trace_schema,
        "trace_sha256": value.trace_sha256,
        "expert_count": value.expert_count,
        "top_k": value.top_k,
        "moe_layer_indices": list(value.moe_layer_indices),
        "requests": [
            {
                "request_id": request.request_id,
                "prompt_token_count": request.prompt_token_count,
                "output_token_count": request.output_token_count,
                "tokens": [_token_to_json(token) for token in request.tokens],
            }
            for request in value.requests
        ],
    }


def _layer_from_json(value: object, path: str) -> RoutedLayer:
    payload = _object(value, path)
    _fields(payload, path, required={"layer_index", "expert_ids"})
    return RoutedLayer(
        layer_index=_integer(
            payload["layer_index"],
            f"{path}.layer_index",
            nonnegative=True,
        ),
        expert_ids=tuple(
            _integer(expert_id, f"{path}.expert_ids[{index}]", nonnegative=True)
            for index, expert_id in enumerate(
                _array(payload["expert_ids"], f"{path}.expert_ids")
            )
        ),
    )


def _token_from_json(value: object, path: str) -> RoutedToken:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={"phase", "token_index", "token_id", "layers"},
    )
    try:
        phase = ForwardPhase(_string(payload["phase"], f"{path}.phase"))
    except ValueError as exc:
        raise ValueError(f"{path}.phase: unsupported phase") from exc
    return RoutedToken(
        phase=phase,
        token_index=_integer(
            payload["token_index"],
            f"{path}.token_index",
            nonnegative=True,
        ),
        token_id=_integer(
            payload["token_id"],
            f"{path}.token_id",
            nonnegative=True,
        ),
        layers=tuple(
            _layer_from_json(layer, f"{path}.layers[{index}]")
            for index, layer in enumerate(_array(payload["layers"], f"{path}.layers"))
        ),
    )


def routed_experts_from_json(value: object) -> RoutedExperts:
    """Parse and validate one strict routed-experts JSON object."""

    payload = _object(value, "projection")
    _fields(
        payload,
        "projection",
        required={
            "schema",
            "trace_schema",
            "trace_sha256",
            "expert_count",
            "top_k",
            "moe_layer_indices",
            "requests",
        },
    )
    requests: list[RoutedRequest] = []
    for request_index, value_request in enumerate(
        _array(payload["requests"], "projection.requests")
    ):
        path = f"projection.requests[{request_index}]"
        request = _object(value_request, path)
        _fields(
            request,
            path,
            required={
                "request_id",
                "prompt_token_count",
                "output_token_count",
                "tokens",
            },
        )
        requests.append(
            RoutedRequest(
                request_id=_string(request["request_id"], f"{path}.request_id"),
                prompt_token_count=_integer(
                    request["prompt_token_count"],
                    f"{path}.prompt_token_count",
                    minimum=1,
                ),
                output_token_count=_integer(
                    request["output_token_count"],
                    f"{path}.output_token_count",
                    minimum=1,
                ),
                tokens=tuple(
                    _token_from_json(token, f"{path}.tokens[{token_index}]")
                    for token_index, token in enumerate(
                        _array(request["tokens"], f"{path}.tokens")
                    )
                ),
            )
        )
    projection = RoutedExperts(
        schema=_string(payload["schema"], "projection.schema"),
        trace_schema=_string(payload["trace_schema"], "projection.trace_schema"),
        trace_sha256=_string(payload["trace_sha256"], "projection.trace_sha256"),
        expert_count=_integer(
            payload["expert_count"],
            "projection.expert_count",
            minimum=1,
        ),
        top_k=_integer(payload["top_k"], "projection.top_k", minimum=1),
        moe_layer_indices=tuple(
            _integer(
                layer_index,
                f"projection.moe_layer_indices[{index}]",
                nonnegative=True,
            )
            for index, layer_index in enumerate(
                _array(payload["moe_layer_indices"], "projection.moe_layer_indices")
            )
        ),
        requests=tuple(requests),
    )
    validate_routed_experts(projection)
    return projection


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object field {key!r}")
        result[key] = value
    return result


def write_routed_experts(
    value: RoutedExperts,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one canonical projection, protecting an existing path by default."""

    payload = routed_experts_to_json(value)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with target.open(mode, encoding="utf-8", newline="\n") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        stream.write("\n")
    return target


def read_routed_experts(path: str | Path) -> RoutedExperts:
    """Read one strict routed-experts JSON object."""

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid routed-experts JSON: {exc}") from exc
    return routed_experts_from_json(value)


def project_preplay_routing(run: PreplayReplayRun) -> RoutedExperts:
    """Project the exact trace bytes and joined request order into assignments."""

    validate_preplay_replay_run(run)
    source = Path(run.trace.path)
    trace_bytes = source.read_bytes()
    digest = hashlib.sha256(trace_bytes).hexdigest()
    if digest != run.trace.sha256:
        raise ValueError(
            "run.trace.sha256: trace bytes changed after the pre-play join"
        )
    trace = read_preplay_trace(source)
    if trace.provenance.schema != run.trace.schema:
        raise ValueError("run.trace.schema: parsed trace schema disagrees with join")
    trace_requests = {request.request_id: request for request in trace.requests}
    requests: list[RoutedRequest] = []
    for index, joined in enumerate(run.requests):
        path = f"run.requests[{index}]"
        try:
            request = trace_requests[joined.routing_reference.request_id]
        except KeyError as exc:
            raise ValueError(
                f"{path}.routing_reference.request_id: request is absent from trace"
            ) from exc
        if request.request_id != joined.request_id:
            raise ValueError(
                f"{path}.routing_reference.request_id: must match joined request"
            )
        if request.output_token_ids != joined.output_token_ids:
            raise ValueError(f"{path}.output_token_ids: disagree with trace authority")
        tokens = (*request.prefill_tokens, *request.decode_tokens)
        requests.append(
            RoutedRequest(
                request_id=request.request_id,
                prompt_token_count=len(request.input_token_ids),
                output_token_count=len(request.output_token_ids),
                tokens=tuple(
                    RoutedToken(
                        phase=token.phase,
                        token_index=token.token_index,
                        token_id=token.token_id,
                        layers=tuple(
                            RoutedLayer(
                                layer_index=layer.layer_index,
                                expert_ids=layer.expert_ids,
                            )
                            for layer in token.routing
                        ),
                    )
                    for token in tokens
                ),
            )
        )
    projection = RoutedExperts(
        trace_schema=trace.provenance.schema,
        trace_sha256=digest,
        expert_count=trace.provenance.expert_count,
        top_k=trace.provenance.top_k,
        moe_layer_indices=trace.provenance.moe_layer_indices,
        requests=tuple(requests),
    )
    validate_routed_experts(projection)
    return projection
