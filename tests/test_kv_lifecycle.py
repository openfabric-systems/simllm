"""Unit coverage for the KV lifecycle authority.

The ledger replays framework decisions; it never makes them. These tests pin
the state machine, the invariants that make an illegal observation stream fail
closed, and the transactional clone that keeps a refused batch from leaving
half-applied state behind.
"""

from __future__ import annotations

import pytest

from simllm.core import (
    KvBlockState,
    KvCacheAction,
    KvCacheWork,
    KvLifecycleLedger,
    KvPoolSpec,
)

BLOCK_TOKENS = 16
BLOCK_BYTES = 2_097_152
BYTES_PER_TOKEN = BLOCK_BYTES // BLOCK_TOKENS


def _spec(capacity: int = 8) -> KvPoolSpec:
    return KvPoolSpec(
        pool_id="kv:0",
        block_bytes=BLOCK_BYTES,
        block_tokens=BLOCK_TOKENS,
        capacity_blocks=capacity,
    )


def _ledger(capacity: int = 8) -> KvLifecycleLedger:
    return KvLifecycleLedger([_spec(capacity)])


def _work(action: KvCacheAction, **kwargs) -> KvCacheWork:
    return KvCacheWork(action, "kv:0", **kwargs)


def _prefill(request: str, blocks: tuple[str, ...], tokens: int):
    return [
        (f"{request}-reserve", _work(
            KvCacheAction.RESERVE,
            request_id=request,
            token_start=0,
            token_end=tokens,
        )),
        (f"{request}-allocate", _work(
            KvCacheAction.ALLOCATE,
            request_id=request,
            block_ids=blocks,
            token_start=0,
            token_end=tokens,
        )),
        (f"{request}-write", _work(
            KvCacheAction.WRITE,
            request_id=request,
            block_ids=blocks,
            token_start=0,
            token_end=tokens,
            byte_count=tokens * BYTES_PER_TOKEN,
        )),
    ]


def test_pool_geometry_rejects_a_block_that_is_not_a_whole_number_of_tokens():
    with pytest.raises(ValueError, match="divide evenly"):
        KvPoolSpec(
            pool_id="kv:0",
            block_bytes=100,
            block_tokens=16,
            capacity_blocks=4,
        )
    spec = _spec()
    assert spec.bytes_per_token == 131_072
    assert spec.capacity_bytes == 8 * BLOCK_BYTES


def test_allocate_write_release_leaves_a_full_block_reclaimable():
    ledger = _ledger()
    ledger.consume(_prefill("r", ("b0", "b1"), 32))
    assert ledger.block_state("kv:0", "b0") is KvBlockState.LIVE
    ledger.consume([("release", _work(
        KvCacheAction.RELEASE,
        request_id="r",
        block_ids=("b1", "b0"),
    ))])
    assert ledger.block_state("kv:0", "b0") is KvBlockState.RECLAIMABLE
    row = ledger.report().pool("kv:0")
    assert (row.live_blocks, row.reclaimable_blocks, row.free_blocks) == (0, 2, 6)
    assert row.write_bytes == 32 * BYTES_PER_TOKEN
    assert row.peak_live_blocks == 2


def test_a_partially_written_block_is_freed_rather_than_kept_reclaimable():
    ledger = _ledger()
    ledger.consume(_prefill("r", ("b0", "b1"), 20))
    assert ledger.block_state("kv:0", "b0") is KvBlockState.LIVE
    assert ledger.block_state("kv:0", "b1") is KvBlockState.LIVE
    ledger.consume([("release", _work(
        KvCacheAction.RELEASE,
        request_id="r",
        block_ids=("b1", "b0"),
    ))])
    assert ledger.block_state("kv:0", "b1") is KvBlockState.FREE
    assert ledger.block_state("kv:0", "b0") is KvBlockState.RECLAIMABLE
    row = ledger.report().pool("kv:0")
    assert (row.reclaimable_blocks, row.freed_blocks) == (1, 1)


def test_prefix_reuse_binds_touches_and_shares_without_double_counting():
    ledger = _ledger()
    ledger.consume(_prefill("first", ("b0", "b1"), 32))
    ledger.consume([("release", _work(
        KvCacheAction.RELEASE,
        request_id="first",
        block_ids=("b1", "b0"),
    ))])
    ledger.consume([
        ("bind", _work(
            KvCacheAction.BIND_PREFIX,
            request_id="second",
            block_ids=("b0", "b1"),
            token_start=0,
            token_end=32,
        )),
        ("touch", _work(
            KvCacheAction.TOUCH,
            request_id="second",
            block_ids=("b0", "b1"),
        )),
        ("retain", _work(
            KvCacheAction.RETAIN,
            request_id="third",
            block_ids=("b0",),
            reference_count=2,
        )),
    ])
    row = ledger.report().pool("kv:0")
    assert (row.prefix_hit_blocks, row.prefix_hit_tokens) == (2, 32)
    assert (row.touched_blocks, row.retained_blocks) == (2, 1)
    assert row.live_blocks == 2


def test_a_cached_block_cannot_be_reallocated_without_an_explicit_eviction():
    ledger = _ledger()
    ledger.consume(_prefill("r", ("b0",), 16))
    ledger.consume([("release", _work(
        KvCacheAction.RELEASE,
        request_id="r",
        block_ids=("b0",),
    ))])
    with pytest.raises(ValueError, match="needs an explicit evict"):
        ledger.consume([("bad", _work(
            KvCacheAction.ALLOCATE,
            request_id="next",
            block_ids=("b0",),
        ))])
    ledger.consume([
        ("evict", _work(
            KvCacheAction.EVICT,
            block_ids=("b0",),
            cause="capacity",
        )),
        ("allocate", _work(
            KvCacheAction.ALLOCATE,
            request_id="next",
            block_ids=("b0",),
        )),
    ])
    row = ledger.report().pool("kv:0")
    assert row.evicted_blocks == 1
    assert row.eviction_causes == (("capacity", 1),)


def test_eviction_requires_a_cause_and_a_reclaimable_block():
    ledger = _ledger()
    ledger.consume(_prefill("r", ("b0",), 16))
    with pytest.raises(ValueError, match="requires a cause"):
        ledger.consume([("evict", _work(KvCacheAction.EVICT, block_ids=("b0",)))])
    with pytest.raises(ValueError, match="only a "):
        ledger.consume([("evict", _work(
            KvCacheAction.EVICT,
            block_ids=("b0",),
            cause="capacity",
        ))])


def test_capacity_and_reservation_refusals_are_explicit():
    ledger = _ledger(capacity=4)
    with pytest.raises(ValueError, match="above the 4 the pool can supply"):
        ledger.consume([("reserve", _work(
            KvCacheAction.RESERVE,
            request_id="r",
            token_start=0,
            token_end=80,
        ))])
    ledger.consume([("reserve", _work(
        KvCacheAction.RESERVE,
        request_id="r",
        token_start=0,
        token_end=32,
    ))])
    assert ledger.report().pool("kv:0").reserved_blocks == 2
    with pytest.raises(ValueError, match="above its outstanding reservation"):
        ledger.consume([("allocate", _work(
            KvCacheAction.ALLOCATE,
            request_id="r",
            block_ids=("b0", "b1", "b2"),
        ))])


def test_ownership_and_reference_count_disagreements_fail_closed():
    ledger = _ledger()
    ledger.consume(_prefill("r", ("b0",), 16))
    with pytest.raises(ValueError, match="does not hold"):
        ledger.consume([("read", _work(
            KvCacheAction.READ,
            request_id="other",
            block_ids=("b0",),
            byte_count=BYTES_PER_TOKEN,
        ))])
    with pytest.raises(ValueError, match="disagrees with the accounted count"):
        ledger.consume([("read", _work(
            KvCacheAction.READ,
            request_id="r",
            block_ids=("b0",),
            byte_count=BYTES_PER_TOKEN,
            reference_count=4,
        ))])


def test_byte_conservation_binds_tokens_blocks_and_metadata_actions():
    ledger = _ledger()
    ledger.consume(_prefill("r", ("b0",), 16))
    with pytest.raises(ValueError, match="must move"):
        ledger.consume([("write", _work(
            KvCacheAction.WRITE,
            request_id="r",
            block_ids=("b0",),
            token_start=0,
            token_end=4,
            byte_count=7,
        ))])
    with pytest.raises(ValueError, match="must carry no bytes"):
        ledger.consume([("bind", _work(
            KvCacheAction.BIND_PREFIX,
            request_id="r",
            block_ids=("b0",),
            byte_count=BYTES_PER_TOKEN,
        ))])
    with pytest.raises(ValueError, match="no slot in the blocks"):
        ledger.consume([("write", _work(
            KvCacheAction.WRITE,
            request_id="r",
            block_ids=("b0",),
            byte_count=BYTES_PER_TOKEN,
        ))])
    ledger.consume([("allocate", _work(
        KvCacheAction.ALLOCATE,
        request_id="r",
        block_ids=("b1",),
    ))])
    with pytest.raises(ValueError, match="exceeds the 16 tokens resident"):
        ledger.consume([("read", _work(
            KvCacheAction.READ,
            request_id="r",
            block_ids=("b0", "b1"),
            token_start=0,
            token_end=20,
            byte_count=20 * BYTES_PER_TOKEN,
        ))])


def test_swap_moves_a_tier_and_recompute_records_replayed_tokens():
    ledger = _ledger()
    ledger.consume(_prefill("r", ("b0",), 16))
    ledger.consume([("swap", _work(
        KvCacheAction.SWAP,
        request_id="r",
        block_ids=("b0",),
        byte_count=BLOCK_BYTES,
        cause="host",
    ))])
    with pytest.raises(ValueError, match="tier it already occupies"):
        ledger.consume([("swap", _work(
            KvCacheAction.SWAP,
            request_id="r",
            block_ids=("b0",),
            byte_count=BLOCK_BYTES,
            cause="host",
        ))])
    ledger.consume([("recompute", _work(
        KvCacheAction.RECOMPUTE,
        request_id="r",
        token_start=0,
        token_end=258,
    ))])
    row = ledger.report().pool("kv:0")
    assert row.swap_bytes == BLOCK_BYTES
    assert row.recomputed_tokens == 258


def test_a_clone_keeps_a_refused_batch_from_touching_committed_state():
    ledger = _ledger()
    ledger.consume(_prefill("r", ("b0",), 16))
    before = ledger.report().pool("kv:0")
    pending = ledger.clone()
    with pytest.raises(ValueError, match="does not hold"):
        pending.consume([
            ("ok", _work(
                KvCacheAction.READ,
                request_id="r",
                block_ids=("b0",),
                token_start=0,
                token_end=16,
                byte_count=16 * BYTES_PER_TOKEN,
            )),
            ("bad", _work(
                KvCacheAction.RELEASE,
                request_id="ghost",
                block_ids=("b0",),
            )),
        ])
    assert ledger.report().pool("kv:0") == before
    assert pending.report().pool("kv:0").read_bytes == 16 * BYTES_PER_TOKEN


def test_unknown_pools_and_malformed_observations_are_refused():
    ledger = _ledger()
    with pytest.raises(ValueError, match="is not configured"):
        ledger.observe(KvCacheWork(KvCacheAction.FREE, "other", block_ids=("b0",)), operation_id="x")
    with pytest.raises(ValueError, match="requires an owning request"):
        ledger.observe(_work(KvCacheAction.ALLOCATE, block_ids=("b0",)), operation_id="x")
    with pytest.raises(ValueError, match="names a block twice"):
        ledger.observe(
            _work(KvCacheAction.ALLOCATE, request_id="r", block_ids=("b0", "b0")),
            operation_id="x",
        )
    with pytest.raises(ValueError, match="half a token interval"):
        ledger.observe(
            _work(KvCacheAction.RESERVE, request_id="r", token_start=0),
            operation_id="x",
        )
    with pytest.raises(ValueError, match="at least one pool"):
        KvLifecycleLedger([])
    with pytest.raises(ValueError, match="duplicate KV pool"):
        KvLifecycleLedger([_spec(), _spec()])
