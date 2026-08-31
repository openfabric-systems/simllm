"""Unit coverage for the observation-only vLLM KV normalization bridge."""

from __future__ import annotations

import json

import pytest

from simllm.adapters.vllm import (
    ORACLE_OBSERVATION_SCHEMA,
    VLLM_KV_PROJECTION_SCHEMA,
    VllmKvGeometry,
    normalize_vllm_kv_events,
    read_vllm_kv_sidecar,
    vllm_kv_projection_to_json,
)
from simllm.core import KvCacheAction


def _row(kind: str, **fields: object) -> dict[str, object]:
    return {"schema": ORACLE_OBSERVATION_SCHEMA, "kind": kind, **fields}


def _geometry() -> VllmKvGeometry:
    return VllmKvGeometry(
        pool_id="vllm:kv:rank0",
        block_tokens=16,
        capacity_blocks=8,
        num_layers=4,
        num_kv_heads=2,
        head_size=8,
        dtype="bfloat16",
        dtype_bytes=2,
        placement_epoch=3,
    )


def _full_sidecar() -> list[dict[str, object]]:
    return [
        _row(
            "kv-manager-qualified",
            block_size=16,
            manager_class="KVCacheManager",
            num_blocks=8,
            token_capacity=128,
        ),
        _row(
            "request-mapping",
            group_index=0,
            mappings=[
                {"internal_request_id": "engine-r0", "request_id": "request-a"}
            ],
        ),
        _row(
            "prefix-hit",
            block_ids=[0],
            request_id="engine-r0",
            token_count=16,
        ),
        _row(
            "eviction",
            block_ids=[2],
            request_id=None,
            token_count=16,
            reason="prefix-cache-capacity",
        ),
        _row(
            "allocation",
            block_ids=[2, 3],
            request_id="engine-r0",
            token_count=32,
        ),
        _row(
            "release",
            block_ids=[3, 2, 0],
            request_id="engine-r0",
            token_count=48,
        ),
        _row(
            "preemption",
            block_ids=[4, 5],
            request_id="engine-r0",
            token_count=32,
            reason="scheduler-recompute",
        ),
    ]


def test_bridge_emits_ordered_identity_and_capacity_projection():
    rows = _full_sidecar()
    before = json.dumps(rows, sort_keys=True, separators=(",", ":"))

    projection = normalize_vllm_kv_events(rows, _geometry())

    assert json.dumps(rows, sort_keys=True, separators=(",", ":")) == before
    assert projection.source_event_count == 5
    assert projection.pool_spec.pool_id == "vllm:kv:rank0"
    assert projection.pool_spec.bytes_per_token == 256
    assert projection.pool_spec.block_bytes == 4_096
    assert projection.pool_spec.capacity_bytes == 32_768

    actions = [work.action for _, work in projection.operations]
    assert actions == [
        KvCacheAction.BIND_PREFIX,
        KvCacheAction.TOUCH,
        KvCacheAction.EVICT,
        KvCacheAction.RESERVE,
        KvCacheAction.ALLOCATE,
        KvCacheAction.RELEASE,
        KvCacheAction.FREE,
        KvCacheAction.RECOMPUTE,
    ]
    operation_ids = [operation_id for operation_id, _ in projection.operations]
    assert operation_ids == [
        "vllm-kv-000002-bind-prefix",
        "vllm-kv-000002-touch",
        "vllm-kv-000003-evict",
        "vllm-kv-000004-reserve",
        "vllm-kv-000004-allocate",
        "vllm-kv-000005-release",
        "vllm-kv-000005-free",
        "vllm-kv-000006-recompute",
    ]

    works = [work for _, work in projection.operations]
    assert [(work.token_start, work.token_end) for work in works] == [
        (0, 16),
        (0, 16),
        (0, 16),
        (16, 48),
        (16, 48),
        (0, 48),
        (0, 48),
        (0, 32),
    ]
    assert works[3].block_ids == ()
    assert works[4].block_ids == ("2", "3")
    assert works[5].block_ids == ("3", "2", "0")
    assert all(work.pool_id == "vllm:kv:rank0" for work in works)
    assert all(work.request_id == "request-a" for work in works if work.request_id)
    assert all(work.dtype == "bfloat16" for work in works)
    assert all(work.byte_count == 0 for work in works)
    assert all(work.placement_epoch == 3 for work in works)
    assert all(work.reference_count is None for work in works)
    assert all(work.correlation_id is None for work in works)


def test_projection_json_discloses_aggregate_bytes_and_absent_native_fields():
    payload = vllm_kv_projection_to_json(
        normalize_vllm_kv_events(_full_sidecar(), _geometry())
    )

    assert payload["schema"] == VLLM_KV_PROJECTION_SCHEMA
    assert payload["pool"] == {
        "pool_id": "vllm:kv:rank0",
        "block_bytes": 4_096,
        "block_tokens": 16,
        "capacity_blocks": 8,
        "tier": "device",
        "dtype": "bfloat16",
        "bytes_per_token": 256,
    }
    allocate = next(
        row for row in payload["operations"] if row["action"] == "allocate"
    )
    assert allocate["block_ids"] == ["2", "3"]
    assert allocate["layer"] is None
    assert allocate["reference_count"] is None
    assert allocate["correlation_id"] is None


def test_sidecar_reader_preserves_zero_hit_and_zero_block_preemption(tmp_path):
    rows = [
        _full_sidecar()[0],
        _row("prefix-hit", block_ids=[], request_id="r0", token_count=0),
        _row(
            "preemption",
            block_ids=[],
            request_id="r0",
            token_count=0,
            reason="scheduler-recompute",
        ),
    ]
    path = tmp_path / "oracle.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    projection = read_vllm_kv_sidecar(path, _geometry())

    assert len(projection.operations) == 1
    _, recompute = projection.operations[0]
    assert recompute.action is KvCacheAction.RECOMPUTE
    assert (recompute.token_start, recompute.token_end) == (None, None)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda rows: rows.__setitem__(
                0, {**rows[0], "block_size": 32, "token_capacity": 256}
            ),
            "disagrees with the pin",
        ),
        (
            lambda rows: rows[4].__setitem__("token_count", 16),
            "token_count 16 disagrees",
        ),
        (
            lambda rows: rows[4].__setitem__("request_id", "unknown"),
            "unmapped request",
        ),
        (
            lambda rows: rows.__setitem__(0, {**rows[0], "schema": "future"}),
            "unsupported schema",
        ),
    ],
)
def test_bridge_refuses_unqualified_or_lossy_inputs(mutate, match):
    rows = _full_sidecar()
    mutate(rows)
    with pytest.raises((TypeError, ValueError), match=match):
        normalize_vllm_kv_events(rows, _geometry())


def test_bridge_requires_one_stock_manager_qualification():
    rows = _full_sidecar()
    with pytest.raises(ValueError, match="exactly one"):
        normalize_vllm_kv_events(rows[1:], _geometry())
    rows.insert(1, dict(rows[0]))
    with pytest.raises(ValueError, match="exactly one"):
        normalize_vllm_kv_events(rows, _geometry())
