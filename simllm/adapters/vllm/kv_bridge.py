"""Normalize observed vLLM KV sidecar rows into core lifecycle operations.

The stock vLLM manager remains the only decision authority. This bridge reads
its append-only observation rows and emits an ordered projection with stable
logical request and pool identities. It neither predicts an allocation nor
changes a cache decision.

This first slice sizes one block across all layers served by a pool. It does
not invent per-layer traffic, native reference counts, or framework
correlation identifiers that the sidecar does not carry.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from simllm.adapters.vllm.oracle import ORACLE_OBSERVATION_SCHEMA
from simllm.core import KvCacheAction, KvCacheWork, KvPoolSpec

VLLM_KV_PROJECTION_SCHEMA = "simllm-vllm-kv-projection-v1"

VllmKvOperation: TypeAlias = tuple[str, KvCacheWork]

_EVENT_KINDS = frozenset(
    {"allocation", "eviction", "prefix-hit", "preemption", "release"}
)
_KNOWN_NON_EVENT_KINDS = frozenset(
    {
        "capture-start",
        "dispatch-path-qualified",
        "dispatch-qualified",
        "kv-manager-qualified",
        "plugin-active",
        "request-final-counters",
        "request-mapping",
        "submission-group-start",
        "worker-qualified",
    }
)
_KNOWN_KINDS = _EVENT_KINDS | _KNOWN_NON_EVENT_KINDS


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class VllmKvGeometry:
    """Pinned, aggregate geometry for one uniform vLLM KV block pool."""

    pool_id: str
    block_tokens: int
    capacity_blocks: int
    num_layers: int
    num_kv_heads: int
    head_size: int
    dtype: str
    dtype_bytes: int
    placement_epoch: int = 0

    def __post_init__(self) -> None:
        _text("pool_id", self.pool_id)
        _text("dtype", self.dtype)
        for name in (
            "block_tokens",
            "capacity_blocks",
            "num_layers",
            "num_kv_heads",
            "head_size",
            "dtype_bytes",
        ):
            _positive_int(name, getattr(self, name))
        _nonnegative_int("placement_epoch", self.placement_epoch)

    @property
    def bytes_per_token(self) -> int:
        """Aggregate key and value bytes for one token across all layers."""

        return (
            2
            * self.num_layers
            * self.num_kv_heads
            * self.head_size
            * self.dtype_bytes
        )

    @property
    def block_bytes(self) -> int:
        return self.block_tokens * self.bytes_per_token

    @property
    def pool_spec(self) -> KvPoolSpec:
        return KvPoolSpec(
            pool_id=self.pool_id,
            block_bytes=self.block_bytes,
            block_tokens=self.block_tokens,
            capacity_blocks=self.capacity_blocks,
        )


@dataclass(frozen=True)
class VllmKvProjection:
    """One qualified manager geometry and its ordered read-only projection."""

    geometry: VllmKvGeometry
    operations: tuple[VllmKvOperation, ...]
    source_event_count: int

    def __post_init__(self) -> None:
        _nonnegative_int("source_event_count", self.source_event_count)
        operation_ids = tuple(operation_id for operation_id, _ in self.operations)
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("KV projection operation IDs must be unique")
        if any(
            work.pool_id != self.geometry.pool_id for _, work in self.operations
        ):
            raise ValueError("KV projection contains an operation for another pool")

    @property
    def pool_spec(self) -> KvPoolSpec:
        return self.geometry.pool_spec


def _request_mapping(rows: tuple[Mapping[str, Any], ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    logical_ids: set[str] = set()
    for row in rows:
        if row.get("kind") != "request-mapping":
            continue
        mappings = row.get("mappings")
        if not isinstance(mappings, list) or not mappings:
            raise ValueError("request-mapping must contain a nonempty mappings list")
        for value in mappings:
            if not isinstance(value, Mapping):
                raise TypeError("request-mapping entries must be objects")
            internal = _text("internal_request_id", value.get("internal_request_id"))
            logical = _text("request_id", value.get("request_id"))
            if internal in result:
                raise ValueError(f"duplicate internal request ID {internal!r}")
            if logical in logical_ids:
                raise ValueError(f"duplicate logical request ID {logical!r}")
            result[internal] = logical
            logical_ids.add(logical)
    return result


def _blocks(row: Mapping[str, Any], *, row_index: int) -> tuple[str, ...]:
    raw = row.get("block_ids")
    if not isinstance(raw, list):
        raise TypeError(f"sidecar row {row_index} block_ids must be a list")
    blocks = tuple(str(_nonnegative_int("block_id", value)) for value in raw)
    if len(blocks) != len(set(blocks)):
        raise ValueError(f"sidecar row {row_index} repeats a block ID")
    return blocks


def _event_tokens(
    row: Mapping[str, Any],
    blocks: tuple[str, ...],
    geometry: VllmKvGeometry,
    *,
    kind: object,
    row_index: int,
) -> int:
    tokens = _nonnegative_int("token_count", row.get("token_count"))
    expected = len(blocks) * geometry.block_tokens
    if kind == "prefix-hit":
        if tokens > expected or (not blocks and tokens):
            raise ValueError(
                f"sidecar row {row_index} prefix hit carries {tokens} tokens "
                f"through {len(blocks)} blocks with capacity {expected}"
            )
        return tokens
    if tokens != expected:
        raise ValueError(
            f"sidecar row {row_index} token_count {tokens} disagrees with "
            f"{len(blocks)} blocks of {geometry.block_tokens} tokens"
        )
    return tokens


def _logical_request_id(
    row: Mapping[str, Any],
    mapping: Mapping[str, str],
    *,
    row_index: int,
) -> str:
    native = _text("request_id", row.get("request_id"))
    if not mapping:
        return native
    logical = mapping.get(native)
    if logical is None:
        raise ValueError(
            f"sidecar row {row_index} names unmapped request {native!r}"
        )
    return logical


def normalize_vllm_kv_events(
    rows: Iterable[Mapping[str, Any]],
    geometry: VllmKvGeometry,
) -> VllmKvProjection:
    """Project one sidecar into ordered ``KvCacheWork`` operations.

    Every output operation is tied to its source-row ordinal. Generated
    operation IDs supply stable projection identity but are not presented as
    native framework correlation identifiers. Reference-sensitive ledger
    replay remains VLLM-47 work.
    """

    if not isinstance(geometry, VllmKvGeometry):
        raise TypeError("geometry must be a VllmKvGeometry")
    normalized = tuple(rows)
    for row_index, row in enumerate(normalized):
        if not isinstance(row, Mapping):
            raise TypeError(f"sidecar row {row_index} must be an object")
        if row.get("schema") != ORACLE_OBSERVATION_SCHEMA:
            raise ValueError(
                f"sidecar row {row_index} has unsupported schema "
                f"{row.get('schema')!r}"
            )
        kind = row.get("kind")
        if kind not in _KNOWN_KINDS:
            raise ValueError(
                f"sidecar row {row_index} has unknown event kind {kind!r}; "
                "update the KV normalization bridge before replaying this sidecar"
            )

    managers = [
        (index, row)
        for index, row in enumerate(normalized)
        if row.get("kind") == "kv-manager-qualified"
    ]
    if len(managers) != 1:
        raise ValueError("sidecar must contain exactly one KV manager qualification")
    manager_index, manager = managers[0]
    if manager.get("manager_class") != "KVCacheManager":
        raise ValueError("sidecar did not qualify the stock KVCacheManager")
    observed_block_tokens = _positive_int("block_size", manager.get("block_size"))
    observed_capacity = _positive_int("num_blocks", manager.get("num_blocks"))
    observed_tokens = _positive_int("token_capacity", manager.get("token_capacity"))
    if (
        observed_block_tokens != geometry.block_tokens
        or observed_capacity != geometry.capacity_blocks
        or observed_tokens != geometry.block_tokens * geometry.capacity_blocks
    ):
        raise ValueError(
            f"sidecar manager geometry at row {manager_index} disagrees with the pin"
        )

    mapping = _request_mapping(normalized)
    next_token: dict[str, int] = {}
    operations: list[VllmKvOperation] = []
    source_event_count = 0

    def append(
        row_index: int,
        suffix: str,
        action: KvCacheAction,
        *,
        request_id: str | None = None,
        block_ids: tuple[str, ...] = (),
        token_start: int | None = None,
        token_end: int | None = None,
        cause: str | None = None,
    ) -> None:
        operation_id = f"vllm-kv-{row_index:06d}-{suffix}"
        operations.append(
            (
                operation_id,
                KvCacheWork(
                    action=action,
                    pool_id=geometry.pool_id,
                    request_id=request_id,
                    block_ids=block_ids,
                    token_start=token_start,
                    token_end=token_end,
                    dtype=geometry.dtype,
                    placement_epoch=geometry.placement_epoch,
                    cause=cause,
                ),
            )
        )

    for row_index, row in enumerate(normalized):
        kind = row.get("kind")
        if kind not in _EVENT_KINDS:
            continue
        source_event_count += 1
        blocks = _blocks(row, row_index=row_index)
        tokens = _event_tokens(
            row, blocks, geometry, kind=kind, row_index=row_index
        )

        if kind == "eviction":
            if not blocks:
                raise ValueError(f"sidecar row {row_index} eviction has no block")
            cause = _text("reason", row.get("reason"))
            append(
                row_index,
                "evict",
                KvCacheAction.EVICT,
                block_ids=blocks,
                token_start=0,
                token_end=tokens,
                cause=cause,
            )
            continue

        request_id = _logical_request_id(
            row, mapping, row_index=row_index
        )
        if kind == "prefix-hit":
            if not blocks:
                if tokens != 0:
                    raise AssertionError("zero-block prefix hit carried tokens")
                continue
            append(
                row_index,
                "bind-prefix",
                KvCacheAction.BIND_PREFIX,
                request_id=request_id,
                block_ids=blocks,
                token_start=0,
                token_end=tokens,
            )
            append(
                row_index,
                "touch",
                KvCacheAction.TOUCH,
                request_id=request_id,
                block_ids=blocks,
                token_start=0,
                token_end=tokens,
            )
            next_token[request_id] = max(next_token.get(request_id, 0), tokens)
        elif kind == "allocation":
            if not blocks:
                raise ValueError(f"sidecar row {row_index} allocation has no block")
            token_start = next_token.get(request_id, 0)
            token_end = token_start + tokens
            append(
                row_index,
                "allocate",
                KvCacheAction.ALLOCATE,
                request_id=request_id,
                block_ids=blocks,
                token_start=token_start,
                token_end=token_end,
            )
            next_token[request_id] = token_end
        elif kind == "release":
            if not blocks:
                raise ValueError(f"sidecar row {row_index} release has no block")
            append(
                row_index,
                "release",
                KvCacheAction.RELEASE,
                request_id=request_id,
                block_ids=blocks,
                token_start=0,
                token_end=tokens,
            )
            append(
                row_index,
                "free",
                KvCacheAction.FREE,
                request_id=request_id,
                block_ids=blocks,
                token_start=0,
                token_end=tokens,
                cause="request-release",
            )
            next_token.pop(request_id, None)
        else:
            cause = _text("reason", row.get("reason"))
            append(
                row_index,
                "recompute",
                KvCacheAction.RECOMPUTE,
                request_id=request_id,
                token_start=0 if tokens else None,
                token_end=tokens or None,
                cause=cause,
            )
            next_token[request_id] = 0

    return VllmKvProjection(
        geometry=geometry,
        operations=tuple(operations),
        source_event_count=source_event_count,
    )


def read_vllm_kv_sidecar(
    path: str | Path,
    geometry: VllmKvGeometry,
) -> VllmKvProjection:
    """Read an append-only CPU-oracle JSONL sidecar and normalize it."""

    source = Path(path)
    rows: list[Mapping[str, Any]] = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
            if not isinstance(value, Mapping):
                raise TypeError(f"{source}:{line_number}: sidecar row must be an object")
            rows.append(value)
    return normalize_vllm_kv_events(rows, geometry)


def vllm_kv_projection_to_json(projection: VllmKvProjection) -> dict[str, Any]:
    """Render the first-slice projection with every absent native field visible."""

    if not isinstance(projection, VllmKvProjection):
        raise TypeError("projection must be a VllmKvProjection")
    spec = projection.pool_spec
    return {
        "schema": VLLM_KV_PROJECTION_SCHEMA,
        "pool": {
            "pool_id": spec.pool_id,
            "block_bytes": spec.block_bytes,
            "block_tokens": spec.block_tokens,
            "capacity_blocks": spec.capacity_blocks,
            "tier": spec.tier,
            "dtype": projection.geometry.dtype,
            "bytes_per_token": spec.bytes_per_token,
        },
        "source_event_count": projection.source_event_count,
        "operations": [
            {
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
            for operation_id, work in projection.operations
        ],
    }
