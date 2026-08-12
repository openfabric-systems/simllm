"""Packed, read-only expert-routing authority for a joined replay run.

The binary payload stores only expert identities. Its layout is
``[joined request][forwarded token][MoE layer slot][top-k slot]`` with one
unsigned byte per identity. The adjacent canonical JSON index carries every
dimension and request extent needed to interpret those bytes. Gate weights
are intentionally absent because routed traffic does not consume them.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from simllm.core._wire import _array, _fields, _integer, _object, _string
from simllm.preplay.join import PreplayReplayRun, validate_preplay_replay_run
from simllm.preplay.schema import PREPLAY_TRACE_SCHEMA, PreplayTrace
from simllm.preplay.trace import read_preplay_trace

if TYPE_CHECKING:
    from simllm.core.request_lifetime import RequestLifetimeRegistry

ROUTING_ARENA_SCHEMA = "simllm-routing-arena-index-v1"
ROUTING_ARENA_LAYOUT = "joined-request-token-moe-layer-topk-uint8"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True, kw_only=True)
class RoutingArenaRequest:
    """One joined request's contiguous token extent in an arena."""

    request_id: str
    token_offset: int
    token_count: int
    prompt_token_count: int
    output_token_count: int

    @property
    def decode_token_count(self) -> int:
        """Return the number of output tokens that execute a forward pass."""

        return self.output_token_count - 1


@dataclass(frozen=True, kw_only=True)
class RoutingArenaIndex:
    """Strict metadata required to validate and interpret one payload."""

    trace_schema: str
    trace_sha256: str
    expert_count: int
    top_k: int
    moe_layer_indices: tuple[int, ...]
    payload_file: str
    payload_bytes: int
    payload_sha256: str
    requests: tuple[RoutingArenaRequest, ...]
    layout: str = ROUTING_ARENA_LAYOUT
    schema: str = ROUTING_ARENA_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "moe_layer_indices", tuple(self.moe_layer_indices))
        object.__setattr__(self, "requests", tuple(self.requests))

    def by_request_id(self, request_id: str) -> RoutingArenaRequest:
        """Return one request extent by its stable joined identity."""

        for request in self.requests:
            if request.request_id == request_id:
                return request
        raise KeyError(f"request ID {request_id!r} not in routing arena")


def _require_sha256(value: object, path: str) -> str:
    digest = _string(value, path)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{path}: expected 64 lowercase hexadecimal digits")
    return digest


def _require_sidecar_name(value: object, path: str) -> str:
    name = _string(value, path)
    forbidden = ("/", "\\", ":", "\0")
    if name in {".", ".."} or any(character in name for character in forbidden):
        raise ValueError(f"{path}: expected a plain sibling file name")
    return name


def _validate_request(value: RoutingArenaRequest, path: str) -> None:
    if not isinstance(value, RoutingArenaRequest):
        raise TypeError(f"{path}: expected RoutingArenaRequest")
    _string(value.request_id, f"{path}.request_id")
    _integer(value.token_offset, f"{path}.token_offset", nonnegative=True)
    token_count = _integer(value.token_count, f"{path}.token_count", minimum=1)
    prompt_count = _integer(
        value.prompt_token_count,
        f"{path}.prompt_token_count",
        minimum=1,
    )
    output_count = _integer(
        value.output_token_count,
        f"{path}.output_token_count",
        minimum=1,
    )
    expected_count = prompt_count + output_count - 1
    if token_count != expected_count:
        raise ValueError(
            f"{path}.token_count: expected {expected_count} forwarded tokens"
        )


def validate_routing_arena_index(value: RoutingArenaIndex) -> None:
    """Validate dimensions, request order and exact contiguous payload extent."""

    if not isinstance(value, RoutingArenaIndex):
        raise TypeError("index: expected RoutingArenaIndex")
    if value.schema != ROUTING_ARENA_SCHEMA:
        raise ValueError(f"index.schema: unsupported schema {value.schema!r}")
    if value.layout != ROUTING_ARENA_LAYOUT:
        raise ValueError(f"index.layout: unsupported layout {value.layout!r}")
    if value.trace_schema != PREPLAY_TRACE_SCHEMA:
        raise ValueError(
            f"index.trace_schema: unsupported schema {value.trace_schema!r}"
        )
    _require_sha256(value.trace_sha256, "index.trace_sha256")
    expert_count = _integer(value.expert_count, "index.expert_count", minimum=1)
    if expert_count > 256:
        raise ValueError("index.expert_count: uint8 routing supports at most 256 experts")
    top_k = _integer(value.top_k, "index.top_k", minimum=1)
    if top_k > expert_count:
        raise ValueError("index.top_k: cannot exceed expert_count")
    if not isinstance(value.moe_layer_indices, tuple):
        raise TypeError(
            "index.moe_layer_indices: in-memory contract requires a tuple"
        )
    if not value.moe_layer_indices:
        raise ValueError("index.moe_layer_indices: must not be empty")
    if len(value.moe_layer_indices) > 64:
        raise ValueError("index.moe_layer_indices: supports at most 64 MoE layers")
    for index, layer in enumerate(value.moe_layer_indices):
        _integer(layer, f"index.moe_layer_indices[{index}]", nonnegative=True)
    if tuple(sorted(set(value.moe_layer_indices))) != value.moe_layer_indices:
        raise ValueError(
            "index.moe_layer_indices: must be unique and increasing"
        )
    _require_sidecar_name(value.payload_file, "index.payload_file")
    payload_bytes = _integer(
        value.payload_bytes,
        "index.payload_bytes",
        nonnegative=True,
    )
    _require_sha256(value.payload_sha256, "index.payload_sha256")
    if not isinstance(value.requests, tuple):
        raise TypeError("index.requests: in-memory contract requires a tuple")
    if not value.requests:
        raise ValueError("index.requests: must not be empty")

    request_ids: list[str] = []
    for index, request in enumerate(value.requests):
        _validate_request(request, f"index.requests[{index}]")
        request_ids.append(request.request_id)
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("index.requests: duplicate request identity")
    next_offset = 0
    for index, request in enumerate(value.requests):
        path = f"index.requests[{index}]"
        if request.token_offset != next_offset:
            relation = "overlaps" if request.token_offset < next_offset else "leaves a gap"
            raise ValueError(
                f"{path}.token_offset: expected contiguous offset {next_offset}; "
                f"value {request.token_offset} {relation}"
            )
        next_offset += request.token_count

    stride = len(value.moe_layer_indices) * top_k
    expected_bytes = next_offset * stride
    if payload_bytes != expected_bytes:
        raise ValueError(
            f"index.payload_bytes: expected {expected_bytes} from request extents"
        )


def routing_arena_index_to_json(value: RoutingArenaIndex) -> dict[str, Any]:
    """Return the strict JSON object for an arena index."""

    validate_routing_arena_index(value)
    return {
        "schema": value.schema,
        "layout": value.layout,
        "trace_schema": value.trace_schema,
        "trace_sha256": value.trace_sha256,
        "expert_count": value.expert_count,
        "top_k": value.top_k,
        "moe_layer_indices": list(value.moe_layer_indices),
        "payload_file": value.payload_file,
        "payload_bytes": value.payload_bytes,
        "payload_sha256": value.payload_sha256,
        "requests": [
            {
                "request_id": request.request_id,
                "token_offset": request.token_offset,
                "token_count": request.token_count,
                "prompt_token_count": request.prompt_token_count,
                "output_token_count": request.output_token_count,
            }
            for request in value.requests
        ],
    }


def routing_arena_index_from_json(value: object) -> RoutingArenaIndex:
    """Parse and validate a strict arena-index JSON object."""

    payload = _object(value, "index")
    _fields(
        payload,
        "index",
        required={
            "schema",
            "layout",
            "trace_schema",
            "trace_sha256",
            "expert_count",
            "top_k",
            "moe_layer_indices",
            "payload_file",
            "payload_bytes",
            "payload_sha256",
            "requests",
        },
    )
    layer_values = _array(payload["moe_layer_indices"], "index.moe_layer_indices")
    request_values = _array(payload["requests"], "index.requests")
    requests: list[RoutingArenaRequest] = []
    for index, item in enumerate(request_values):
        path = f"index.requests[{index}]"
        request = _object(item, path)
        _fields(
            request,
            path,
            required={
                "request_id",
                "token_offset",
                "token_count",
                "prompt_token_count",
                "output_token_count",
            },
        )
        requests.append(
            RoutingArenaRequest(
                request_id=_string(request["request_id"], f"{path}.request_id"),
                token_offset=_integer(
                    request["token_offset"],
                    f"{path}.token_offset",
                ),
                token_count=_integer(
                    request["token_count"],
                    f"{path}.token_count",
                ),
                prompt_token_count=_integer(
                    request["prompt_token_count"],
                    f"{path}.prompt_token_count",
                ),
                output_token_count=_integer(
                    request["output_token_count"],
                    f"{path}.output_token_count",
                ),
            )
        )
    index = RoutingArenaIndex(
        schema=_string(payload["schema"], "index.schema"),
        layout=_string(payload["layout"], "index.layout"),
        trace_schema=_string(payload["trace_schema"], "index.trace_schema"),
        trace_sha256=_string(payload["trace_sha256"], "index.trace_sha256"),
        expert_count=_integer(payload["expert_count"], "index.expert_count"),
        top_k=_integer(payload["top_k"], "index.top_k"),
        moe_layer_indices=tuple(
            _integer(layer, f"index.moe_layer_indices[{layer_index}]")
            for layer_index, layer in enumerate(layer_values)
        ),
        payload_file=_string(payload["payload_file"], "index.payload_file"),
        payload_bytes=_integer(payload["payload_bytes"], "index.payload_bytes"),
        payload_sha256=_string(
            payload["payload_sha256"],
            "index.payload_sha256",
        ),
        requests=tuple(requests),
    )
    validate_routing_arena_index(index)
    return index


def _canonical_index_bytes(value: RoutingArenaIndex) -> bytes:
    text = json.dumps(
        routing_arena_index_to_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return text.encode("utf-8") + b"\n"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object field {key!r}")
        result[key] = value
    return result


def read_routing_arena_index(path: str | Path) -> RoutingArenaIndex:
    """Read a canonical arena index and reject alternate JSON encodings."""

    source = Path(path)
    raw = source.read_bytes()
    try:
        text = raw.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid routing-arena index JSON: {exc}") from exc
    index = routing_arena_index_from_json(value)
    if raw != _canonical_index_bytes(index):
        raise ValueError(
            f"{source}: routing-arena index is not canonical JSON with one LF"
        )
    return index


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_run_trace(run: PreplayReplayRun) -> PreplayTrace:
    validate_preplay_replay_run(run)
    source = Path(run.trace.path)
    digest = _file_sha256(source)
    if digest != run.trace.sha256:
        raise ValueError("run.trace.sha256: trace bytes changed after the pre-play join")
    trace = read_preplay_trace(source)
    if trace.provenance.schema != run.trace.schema:
        raise ValueError("run.trace.schema: parsed trace schema disagrees with join")
    return trace


def _validate_index_against_run(
    index: RoutingArenaIndex,
    run: PreplayReplayRun,
) -> None:
    trace = _load_run_trace(run)
    if index.trace_schema != run.trace.schema:
        raise ValueError("index.trace_schema: disagrees with joined run")
    if index.trace_sha256 != run.trace.sha256:
        raise ValueError("index.trace_sha256: disagrees with joined run")
    joined_ids = tuple(request.request_id for request in run.requests)
    index_ids = tuple(request.request_id for request in index.requests)
    if index_ids != joined_ids:
        raise ValueError("index.requests: order disagrees with joined run")
    trace_requests = {request.request_id: request for request in trace.requests}
    for position, (entry, joined) in enumerate(zip(index.requests, run.requests)):
        path = f"index.requests[{position}]"
        try:
            captured = trace_requests[joined.request_id]
        except KeyError as exc:
            raise ValueError(f"{path}.request_id: absent from source trace") from exc
        expected_prompt_count = len(captured.input_token_ids)
        expected_output_count = len(captured.output_token_ids)
        expected_token_count = expected_prompt_count + expected_output_count - 1
        if entry.prompt_token_count != expected_prompt_count:
            raise ValueError(f"{path}.prompt_token_count: disagrees with source trace")
        if entry.output_token_count != expected_output_count:
            raise ValueError(f"{path}.output_token_count: disagrees with source trace")
        if entry.token_count != expected_token_count:
            raise ValueError(f"{path}.token_count: disagrees with source trace")
        if captured.output_token_ids != joined.output_token_ids:
            raise ValueError(f"run.requests[{position}].output_token_ids: disagree with trace")


def _write_payload(
    stream: BinaryIO,
    run: PreplayReplayRun,
    trace: PreplayTrace,
) -> tuple[tuple[RoutingArenaRequest, ...], int, str]:
    trace_requests = {request.request_id: request for request in trace.requests}
    requests: list[RoutingArenaRequest] = []
    token_offset = 0
    payload_bytes = 0
    digest = hashlib.sha256()
    for position, joined in enumerate(run.requests):
        path = f"run.requests[{position}]"
        try:
            captured = trace_requests[joined.routing_reference.request_id]
        except KeyError as exc:
            raise ValueError(
                f"{path}.routing_reference.request_id: request is absent from trace"
            ) from exc
        if captured.request_id != joined.request_id:
            raise ValueError(f"{path}.routing_reference.request_id: must match joined request")
        if captured.output_token_ids != joined.output_token_ids:
            raise ValueError(f"{path}.output_token_ids: disagree with trace authority")
        tokens = (*captured.prefill_tokens, *captured.decode_tokens)
        requests.append(
            RoutingArenaRequest(
                request_id=joined.request_id,
                token_offset=token_offset,
                token_count=len(tokens),
                prompt_token_count=len(captured.input_token_ids),
                output_token_count=len(captured.output_token_ids),
            )
        )
        for token in tokens:
            for route in token.routing:
                packed = bytes(route.expert_ids)
                stream.write(packed)
                digest.update(packed)
                payload_bytes += len(packed)
        token_offset += len(tokens)
    return tuple(requests), payload_bytes, digest.hexdigest()


def build_routing_arena(
    run: PreplayReplayRun,
    index_path: str | Path,
    *,
    overwrite: bool = False,
) -> RoutingArena:
    """Build a packed sidecar directly from a joined run's source trace.

    The returned arena is already open read-only. The builder never creates a
    ``RoutedExperts`` object and never copies gate weights into either sidecar.
    """

    trace = _load_run_trace(run)
    provenance = trace.provenance
    if provenance.expert_count > 256:
        raise ValueError("trace.provenance.expert_count: uint8 supports at most 256 experts")
    if len(provenance.moe_layer_indices) > 64:
        raise ValueError(
            "trace.provenance.moe_layer_indices: supports at most 64 MoE layers"
        )

    target = Path(index_path)
    payload_target = target.with_suffix(".bin")
    if payload_target == target:
        raise ValueError("index_path: must not resolve to its .bin payload path")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        if target.exists():
            raise FileExistsError(target)
        if payload_target.exists():
            raise FileExistsError(payload_target)

    mode = "wb" if overwrite else "xb"
    payload_created = False
    index_created = False
    try:
        with payload_target.open(mode) as stream:
            payload_created = True
            requests, payload_bytes, payload_sha256 = _write_payload(
                stream,
                run,
                trace,
            )
        index = RoutingArenaIndex(
            trace_schema=provenance.schema,
            trace_sha256=run.trace.sha256,
            expert_count=provenance.expert_count,
            top_k=provenance.top_k,
            moe_layer_indices=provenance.moe_layer_indices,
            payload_file=payload_target.name,
            payload_bytes=payload_bytes,
            payload_sha256=payload_sha256,
            requests=requests,
        )
        validate_routing_arena_index(index)
        with target.open(mode) as stream:
            index_created = True
            stream.write(_canonical_index_bytes(index))
    except BaseException:
        if not overwrite:
            if index_created:
                target.unlink(missing_ok=True)
            if payload_created:
                payload_target.unlink(missing_ok=True)
        raise
    return open_routing_arena(target)


class RoutingArenaRequestView:
    """Explicitly acquired request ownership over one arena extent.

    The handle does not export a Python ``memoryview``. This lets the arena
    close its mmap portably after every request handle has been released.
    """

    def __init__(
        self,
        arena: RoutingArena,
        request: RoutingArenaRequest,
        view_id: int,
    ) -> None:
        self._arena = arena
        self._request = request
        self._view_id = view_id
        self._released = False

    @property
    def request_id(self) -> str:
        return self._request.request_id

    @property
    def arena_id(self) -> str:
        return self._arena.arena_id

    @property
    def token_offset(self) -> int:
        return self._request.token_offset

    @property
    def token_count(self) -> int:
        return self._request.token_count

    @property
    def prompt_token_count(self) -> int:
        return self._request.prompt_token_count

    @property
    def output_token_count(self) -> int:
        return self._request.output_token_count

    @property
    def released(self) -> bool:
        return self._released

    def expert_ids(self, token_index: int, model_layer: int) -> tuple[int, ...]:
        """Read one token and model-layer assignment without advancing state."""

        if self._released:
            raise RuntimeError(f"request view {self.request_id!r} has been released")
        return self._arena.expert_ids_at(
            self.token_offset,
            self.token_count,
            token_index,
            model_layer,
        )

    def expert_id(
        self,
        token_index: int,
        model_layer: int,
        topk_slot: int,
    ) -> int:
        """Read one expert identity from this request extent."""

        if self._released:
            raise RuntimeError(f"request view {self.request_id!r} has been released")
        return self._arena.expert_id_at(
            self.token_offset,
            self.token_count,
            token_index,
            model_layer,
            topk_slot,
        )

    def release(self) -> None:
        """Release this handle exactly once from the arena's live-view count."""

        if self._released:
            return
        self._arena._release_view(self._view_id)
        self._released = True

    def __enter__(self):
        if self._released:
            raise RuntimeError(f"request view {self.request_id!r} has been released")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


class RoutingArena:
    """Validated read-only mmap plus stateless expert lookup methods."""

    def __init__(
        self,
        index_path: Path,
        index: RoutingArenaIndex,
        payload_path: Path,
        payload_stream: BinaryIO,
        mapping: mmap.mmap,
    ) -> None:
        self.index_path = index_path
        self.index = index
        self.payload_path = payload_path
        self._payload_stream = payload_stream
        self._mapping = mapping
        self._layer_slots = {
            model_layer: slot
            for slot, model_layer in enumerate(index.moe_layer_indices)
        }
        self._requests_by_id = {
            request.request_id: request for request in index.requests
        }
        self._requests_by_extent = {
            (request.token_offset, request.token_count): request
            for request in index.requests
        }
        self._live_view_ids: set[int] = set()
        self._next_view_id = 0
        self._closed = False

    @classmethod
    def open(
        cls,
        index_path: str | Path,
        *,
        expected_run: PreplayReplayRun | None = None,
    ) -> RoutingArena:
        """Open and fully validate one existing arena sidecar pair."""

        return open_routing_arena(index_path, expected_run=expected_run)

    @property
    def arena_id(self) -> str:
        """Return the content identity used by core request descriptors."""

        return self.index.payload_sha256

    @property
    def expert_count(self) -> int:
        return self.index.expert_count

    @property
    def top_k(self) -> int:
        return self.index.top_k

    @property
    def moe_layer_indices(self) -> tuple[int, ...]:
        return self.index.moe_layer_indices

    @property
    def trace_schema(self) -> str:
        return self.index.trace_schema

    @property
    def trace_sha256(self) -> str:
        return self.index.trace_sha256

    @property
    def requests(self) -> tuple[RoutingArenaRequest, ...]:
        return self.index.requests

    @property
    def live_view_count(self) -> int:
        return len(self._live_view_ids)

    @property
    def closed(self) -> bool:
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("routing arena is closed")

    def by_request_id(self, request_id: str) -> RoutingArenaRequest:
        """Return immutable metadata for one request without acquiring a view."""

        self._require_open()
        try:
            return self._requests_by_id[request_id]
        except KeyError as exc:
            raise KeyError(f"request ID {request_id!r} not in routing arena") from exc

    def acquire_request(self, request_id: str) -> RoutingArenaRequestView:
        """Acquire one explicit live view that must be released before close."""

        request = self.by_request_id(request_id)
        view_id = self._next_view_id
        self._next_view_id += 1
        self._live_view_ids.add(view_id)
        return RoutingArenaRequestView(self, request, view_id)

    def _release_view(self, view_id: int) -> None:
        self._require_open()
        try:
            self._live_view_ids.remove(view_id)
        except KeyError as exc:
            raise RuntimeError("routing arena request view is not live") from exc

    def _request_for_extent(
        self,
        token_offset: int,
        token_count: int,
    ) -> RoutingArenaRequest:
        _integer(token_offset, "token_offset", nonnegative=True)
        _integer(token_count, "token_count", minimum=1)
        try:
            return self._requests_by_extent[(token_offset, token_count)]
        except KeyError as exc:
            raise ValueError(
                "token_offset/token_count: not a registered request extent"
            ) from exc

    def expert_ids_at(
        self,
        token_offset: int,
        token_count: int,
        token_index: int,
        model_layer: int,
    ) -> tuple[int, ...]:
        """Read one assignment through a core-neutral request descriptor."""

        self._require_open()
        request = self._request_for_extent(token_offset, token_count)
        local_index = _integer(token_index, "token_index", nonnegative=True)
        if local_index >= request.token_count:
            raise ValueError(
                f"token_index: {local_index} is outside request token count "
                f"{request.token_count}"
            )
        layer = _integer(model_layer, "model_layer", nonnegative=True)
        try:
            layer_slot = self._layer_slots[layer]
        except KeyError as exc:
            raise ValueError(f"model_layer: {layer} is absent from the routing arena") from exc
        token_stride = len(self.moe_layer_indices) * self.top_k
        start = (
            (request.token_offset + local_index) * token_stride
            + layer_slot * self.top_k
        )
        return tuple(self._mapping[start : start + self.top_k])

    def expert_id_at(
        self,
        token_offset: int,
        token_count: int,
        token_index: int,
        model_layer: int,
        topk_slot: int,
    ) -> int:
        """Read one slot through a core-neutral request descriptor."""

        slot = _integer(topk_slot, "topk_slot", nonnegative=True)
        if slot >= self.top_k:
            raise ValueError(f"topk_slot: {slot} is outside top_k {self.top_k}")
        return self.expert_ids_at(
            token_offset,
            token_count,
            token_index,
            model_layer,
        )[slot]

    def expert_ids(
        self,
        request_id: str,
        token_index: int,
        model_layer: int,
    ) -> tuple[int, ...]:
        """Read one assignment by joined request identity."""

        request = self.by_request_id(request_id)
        return self.expert_ids_at(
            request.token_offset,
            request.token_count,
            token_index,
            model_layer,
        )

    def expert_id(
        self,
        request_id: str,
        token_index: int,
        model_layer: int,
        topk_slot: int,
    ) -> int:
        """Read one expert identity by request, token, layer and slot."""

        request = self.by_request_id(request_id)
        return self.expert_id_at(
            request.token_offset,
            request.token_count,
            token_index,
            model_layer,
            topk_slot,
        )

    def close(self) -> None:
        """Close the mmap only after every explicitly acquired view is released."""

        if self._closed:
            return
        if self._live_view_ids:
            raise BufferError(
                "routing arena cannot close with "
                f"{len(self._live_view_ids)} live request view(s)"
            )
        self._mapping.close()
        self._payload_stream.close()
        self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _open_payload(
    index_path: Path,
    index: RoutingArenaIndex,
) -> tuple[Path, BinaryIO, mmap.mmap]:
    payload_path = index_path.parent / index.payload_file
    stream = payload_path.open("rb")
    try:
        actual_bytes = payload_path.stat().st_size
        if actual_bytes != index.payload_bytes:
            relation = "truncated" if actual_bytes < index.payload_bytes else "has extra bytes"
            raise ValueError(
                f"{payload_path}: payload is {relation}; expected "
                f"{index.payload_bytes} bytes, got {actual_bytes}"
            )
        mapping = mmap.mmap(stream.fileno(), length=0, access=mmap.ACCESS_READ)
        try:
            digest = hashlib.sha256()
            for offset in range(0, index.payload_bytes, _HASH_CHUNK_BYTES):
                chunk = mapping[offset : offset + _HASH_CHUNK_BYTES]
                digest.update(chunk)
                if index.expert_count < 256 and chunk and max(chunk) >= index.expert_count:
                    bad_offset = offset + next(
                        position
                        for position, expert_id in enumerate(chunk)
                        if expert_id >= index.expert_count
                    )
                    raise ValueError(
                        f"{payload_path}: expert identity {mapping[bad_offset]} at "
                        f"byte {bad_offset} is outside [0, {index.expert_count})"
                    )
            actual_digest = digest.hexdigest()
            if actual_digest != index.payload_sha256:
                raise ValueError(f"{payload_path}: payload SHA-256 disagrees with index")
        except BaseException:
            mapping.close()
            raise
    except BaseException:
        stream.close()
        raise
    return payload_path, stream, mapping


def open_routing_arena(
    index_path: str | Path,
    *,
    expected_run: PreplayReplayRun | None = None,
) -> RoutingArena:
    """Open an arena read-only after strict index and payload validation."""

    source = Path(index_path)
    index = read_routing_arena_index(source)
    if expected_run is not None:
        _validate_index_against_run(index, expected_run)
    payload_path, stream, mapping = _open_payload(source, index)
    return RoutingArena(source, index, payload_path, stream, mapping)


def create_request_lifetimes(
    run: PreplayReplayRun,
    arena: RoutingArena,
) -> RequestLifetimeRegistry:
    """Acquire every arena view into one framework-neutral core registry."""

    from simllm.core.request_lifetime import (
        JoinProvenance,
        RequestLifetimeRegistry,
        RoutingViewDescriptor,
    )

    if not isinstance(arena, RoutingArena):
        raise TypeError("arena: expected RoutingArena")
    arena._require_open()
    _validate_index_against_run(arena.index, run)
    provenance = JoinProvenance(
        run_schema=run.schema,
        trace_schema=run.trace.schema,
        trace_sha256=run.trace.sha256,
    )
    registry = RequestLifetimeRegistry(arena.moe_layer_indices)
    acquired: list[RoutingArenaRequestView] = []
    try:
        for joined in run.requests:
            view = arena.acquire_request(joined.request_id)
            acquired.append(view)
            registry.register(
                request_id=joined.request_id,
                provenance=provenance,
                arrived_at_ps=joined.arrived_at_ps,
                view=RoutingViewDescriptor(
                    arena_id=view.arena_id,
                    token_offset=view.token_offset,
                    token_count=view.token_count,
                    prompt_token_count=view.prompt_token_count,
                    release_callback=view.release,
                ),
            )
    except BaseException:
        for view in reversed(acquired):
            view.release()
        raise
    return registry


__all__ = [
    "ROUTING_ARENA_LAYOUT",
    "ROUTING_ARENA_SCHEMA",
    "RoutingArena",
    "RoutingArenaIndex",
    "RoutingArenaRequest",
    "RoutingArenaRequestView",
    "build_routing_arena",
    "create_request_lifetimes",
    "open_routing_arena",
    "read_routing_arena_index",
    "routing_arena_index_from_json",
    "routing_arena_index_to_json",
    "validate_routing_arena_index",
]
