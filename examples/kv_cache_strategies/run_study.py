"""Replay one request against a constrained KV cache and measure TTFT/TPOT.

CORE-3: SimLLM had no KV lifecycle state at all. Every ``KvCacheWork``
operation was a zero-cost marker and a byte-carrying READ or WRITE was refused
in preflight, so a replayed request's timing could not respond to memory
pressure. This study lands the pool authority, its invariants and the one
lowering that lets its bytes reach a reported metric.

Every literal below was frozen in ``expectations.md`` before any
implementation existed, and none of them may be edited to agree with an
observation. ``--check-only`` proves the arithmetic is self consistent and
that the mechanism is still absent; it is the pre-implementation dry run.

The fixture geometry is the published Llama-3.1-8B attention shape so the byte
arithmetic is checkable against real hardware. The pool sizes are a mechanism
knob, not a deployment sizing claim.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Frozen fixture geometry. Quoted from expectations.md.
# ---------------------------------------------------------------------------

POOL_ID = "kv:0"

LAYERS = 32
KV_HEADS = 8
HEAD_SIZE = 128
DTYPE_BYTES = 2
BLOCK_TOKENS = 16

#: 2 * block_tokens * kv_heads * head_size * dtype_bytes, per layer
PAGE_BYTES_PER_LAYER = 2 * BLOCK_TOKENS * KV_HEADS * HEAD_SIZE * DTYPE_BYTES
BLOCK_BYTES = LAYERS * PAGE_BYTES_PER_LAYER
BYTES_PER_TOKEN = BLOCK_BYTES // BLOCK_TOKENS

#: 8 TB/s and the halved comparator, expressed in bits per second
HBM_RATE_FAST_BPS = 64_000_000_000_000
HBM_RATE_SLOW_BPS = 32_000_000_000_000

#: KV picoseconds per token at each rate
K_FAST = 16_384
K_SLOW = 32_768

PREFILL_PS_PER_TOKEN = 40_000_000
DECODE_STEP_PS = 2_000_000_000

# ---------------------------------------------------------------------------
# Frozen family A ladder and tables.
# ---------------------------------------------------------------------------

A_PREFIX_TOKENS = 256
A_PRIVATE_TOKENS = 128
F_TOKENS = 512
R_TOKENS = 512

CAPACITIES = (32, 40, 44, 48, 56, 64)

#: capacity -> (evictions during step 1, R hit blocks)
FROZEN_LADDER: dict[int, tuple[int, int]] = {
    32: (24, 0),
    40: (16, 8),
    44: (12, 12),
    48: (8, 16),
    56: (0, 16),
    64: (0, 16),
}

#: (capacity, k) -> (ttft_ps, kv_ps)
FROZEN_FAMILY_A: dict[tuple[int, int], tuple[int, int]] = {
    (32, K_FAST): (20_488_388_608, 8_388_608),
    (40, K_FAST): (15_366_291_456, 6_291_456),
    (44, K_FAST): (12_805_242_880, 5_242_880),
    (48, K_FAST): (10_244_194_304, 4_194_304),
    (56, K_FAST): (10_244_194_304, 4_194_304),
    (64, K_FAST): (10_244_194_304, 4_194_304),
    (32, K_SLOW): (20_496_777_216, 16_777_216),
    (40, K_SLOW): (15_372_582_912, 12_582_912),
    (44, K_SLOW): (12_810_485_760, 10_485_760),
    (48, K_SLOW): (10_248_388_608, 8_388_608),
    (56, K_SLOW): (10_248_388_608, 8_388_608),
    (64, K_SLOW): (10_248_388_608, 8_388_608),
}

# ---------------------------------------------------------------------------
# Frozen family B tables.
# ---------------------------------------------------------------------------

D_PROMPT_TOKENS = 256
D_DECODE_TOKENS = 4
G_TOKENS = 384

#: capacity that forces the preemption, and the capacity that does not
B_CAPACITY_TIGHT = 24
B_CAPACITY_LOOSE = 64

#: (arm, k) -> (ttft_ps, tpot_ps, token_4_interval_ps)
FROZEN_FAMILY_B: dict[tuple[str, int], tuple[int, Fraction, int]] = {
    ("unconstrained", K_FAST): (
        10_244_194_304,
        Fraction(5_845_808_128),
        17_370_534_912,
    ),
    ("preempted", K_FAST): (
        10_244_194_304,
        Fraction(7_935_808_128),
        25_730_534_912,
    ),
    ("unconstrained", K_SLOW): (
        10_248_388_608,
        Fraction(5_851_616_256),
        17_381_069_824,
    ),
    ("preempted", K_SLOW): (
        10_248_388_608,
        Fraction(7_941_616_256),
        25_741_069_824,
    ),
}

#: the frozen TPOT rise caused by the preemption, identical at every HBM rate
#: because the resumed step and the decode step it replaces move exactly the
#: same 259 token equivalents of KV
FROZEN_TPOT_DELTA_PS = 2_090_000_000

#: blocks D releases and blocks of D that are evicted, preempted arm
FROZEN_D_RELEASED_BLOCKS = 17
FROZEN_D_EVICTED_BLOCKS = 16


def _fail(name: str, expected: object, observed: object) -> None:
    raise AssertionError(f"{name}: expected {expected!r}, observed {observed!r}")


def _equal(name: str, expected: object, observed: object) -> None:
    if expected != observed:
        _fail(name, expected, observed)


def kv_ps_per_token(rate_bps: int) -> int:
    """Return the frozen HBM service of one token of KV at ``rate_bps``."""

    return -(-BYTES_PER_TOKEN * 8 * 10**12 // rate_bps)


def family_a_ladder(capacity: int) -> tuple[int, int]:
    """Return (evictions in step 1, R hit blocks) for one pool capacity.

    Derived from vLLM 0.26.0: a request's blocks are freed in reverse order
    (``single_type_kv_cache_manager.py:503``) and allocation pops the LRU front
    (``block_pool.py:661``), so A's private tail is reclaimed before the shared
    prefix, and the surviving hit is a chained-hash prefix run
    (``single_type_kv_cache_manager.py:708-711``).
    """

    a_blocks = (A_PREFIX_TOKENS + A_PRIVATE_TOKENS) // BLOCK_TOKENS
    f_blocks = F_TOKENS // BLOCK_TOKENS
    prefix_blocks = A_PREFIX_TOKENS // BLOCK_TOKENS
    private_blocks = a_blocks - prefix_blocks
    evictions = max(0, f_blocks - (capacity - a_blocks))
    hit_blocks = prefix_blocks - max(0, evictions - private_blocks)
    return evictions, max(0, hit_blocks)


def family_a_expected(capacity: int, k: int) -> tuple[int, int]:
    """Return the frozen (TTFT, kv_ps) closed form for one family A cell."""

    _, hit_blocks = family_a_ladder(capacity)
    recomputed = R_TOKENS - hit_blocks * BLOCK_TOKENS
    return recomputed * (PREFILL_PS_PER_TOKEN + k), recomputed * k


def family_b_steps(arm: str, k: int) -> tuple[int, ...]:
    """Return the frozen step latencies of family B, in graph order."""

    prompt = D_PROMPT_TOKENS
    #: preemption zeroes the token cursor while the sequence keeps its three
    #: generated tokens, so the resumed step recomputes every one of them
    resumed_tokens = prompt + 3
    step_4 = (
        resumed_tokens * PREFILL_PS_PER_TOKEN + resumed_tokens * k
        if arm == "preempted"
        else DECODE_STEP_PS + (prompt + 3) * k
    )
    return (
        prompt * PREFILL_PS_PER_TOKEN + prompt * k,
        DECODE_STEP_PS + (prompt + 1) * k,
        DECODE_STEP_PS + (prompt + 2) * k,
        G_TOKENS * PREFILL_PS_PER_TOKEN + G_TOKENS * k,
        step_4,
        DECODE_STEP_PS + (prompt + 4) * k,
    )


def family_b_expected(arm: str, k: int) -> tuple[int, Fraction, int]:
    """Return the frozen (TTFT, TPOT, token-4 interval) for one family B arm."""

    steps = family_b_steps(arm, k)
    intervals = (steps[1], steps[2], steps[3] + steps[4], steps[5])
    return steps[0], Fraction(sum(intervals), len(intervals)), intervals[2]


def check_only() -> None:
    """Validate the frozen registry against its own arithmetic, run no study.

    This is the pre-implementation dry run. It also asserts that the mechanism
    the study measures does not exist yet, so a later claim of
    pre-registration is checkable rather than asserted.
    """

    _equal("page bytes per layer", 65_536, PAGE_BYTES_PER_LAYER)
    _equal("block bytes", 2_097_152, BLOCK_BYTES)
    _equal("bytes per token", 131_072, BYTES_PER_TOKEN)
    _equal("k fast", K_FAST, kv_ps_per_token(HBM_RATE_FAST_BPS))
    _equal("k slow", K_SLOW, kv_ps_per_token(HBM_RATE_SLOW_BPS))
    _equal("k doubling", 2 * K_FAST, K_SLOW)

    for capacity in CAPACITIES:
        _equal(f"ladder C={capacity}", FROZEN_LADDER[capacity], family_a_ladder(capacity))
    for (capacity, k), frozen in FROZEN_FAMILY_A.items():
        _equal(f"family A C={capacity} k={k}", frozen, family_a_expected(capacity, k))

    for k in (K_FAST, K_SLOW):
        plateau = {FROZEN_FAMILY_A[(capacity, k)] for capacity in (48, 56, 64)}
        if len(plateau) != 1:
            _fail(f"plateau k={k}", "one distinct cell", sorted(plateau))
        ttfts = [FROZEN_FAMILY_A[(capacity, k)][0] for capacity in CAPACITIES]
        if any(left < right for left, right in itertools.pairwise(ttfts)):
            _fail(f"monotonicity k={k}", "non-increasing in capacity", ttfts)
        _equal(
            f"constrained ratio k={k}",
            2 * FROZEN_FAMILY_A[(64, k)][0],
            FROZEN_FAMILY_A[(32, k)][0],
        )
    for capacity in CAPACITIES:
        _equal(
            f"kv doubling C={capacity}",
            2 * FROZEN_FAMILY_A[(capacity, K_FAST)][1],
            FROZEN_FAMILY_A[(capacity, K_SLOW)][1],
        )
        kernel_fast = (
            FROZEN_FAMILY_A[(capacity, K_FAST)][0] - FROZEN_FAMILY_A[(capacity, K_FAST)][1]
        )
        kernel_slow = (
            FROZEN_FAMILY_A[(capacity, K_SLOW)][0] - FROZEN_FAMILY_A[(capacity, K_SLOW)][1]
        )
        _equal(f"kernel invariance C={capacity}", kernel_fast, kernel_slow)

    for (arm, k), frozen in FROZEN_FAMILY_B.items():
        _equal(f"family B {arm} k={k}", frozen, family_b_expected(arm, k))
    for k in (K_FAST, K_SLOW):
        delta = FROZEN_FAMILY_B[("preempted", k)][1] - FROZEN_FAMILY_B[("unconstrained", k)][1]
        _equal(f"tpot delta k={k}", Fraction(FROZEN_TPOT_DELTA_PS), delta)
        _equal(
            f"tpot delta closed form k={k}",
            Fraction(10_360_000_000 - 2_000_000_000, 4),
            delta,
        )
        _equal(
            f"ttft unchanged k={k}",
            FROZEN_FAMILY_B[("unconstrained", k)][0],
            FROZEN_FAMILY_B[("preempted", k)][0],
        )
        resumed_kv = (D_PROMPT_TOKENS + 3) * k
        replaced_kv = (D_PROMPT_TOKENS + 2) * k + k
        _equal(f"resumed kv cancels k={k}", replaced_kv, resumed_kv)

    state = _check_mechanism_absent()
    print(
        "check-only validated the frozen KV ladder, both metric tables and the "
        f"physical byte arithmetic, and confirmed the {state}"
    )


def _check_mechanism_absent() -> str:
    """Assert a KV byte cannot reach a metric without a declared pool.

    Before the implementation landed this was the absence proof that made the
    pre-registration claim checkable: no pool type existed and every
    byte-carrying KV operation was refused. Afterwards the same assertion is
    the off-path guard, because a runtime with no declared pool must still
    refuse that operation exactly as it did then.
    """

    from simllm import core
    from simllm.core import (
        CoarseDeviceRuntime,
        ExecutionGraph,
        ExecutionOperation,
        KvCacheAction,
        KvCacheWork,
    )

    landed = all(hasattr(core, name) for name in ("KvPoolSpec", "KvLifecycleLedger"))
    graph = ExecutionGraph(
        "kv-absent",
        0,
        0,
        (
            ExecutionOperation(
                "kv",
                0,
                "kv",
                KvCacheWork(KvCacheAction.WRITE, "pool", byte_count=BLOCK_BYTES),
            ),
        ),
    )
    try:
        CoarseDeviceRuntime().execute(graph)
    except ValueError:
        return "off path intact" if landed else "mechanism absent"
    _fail("byte-carrying KV without a pool", "refused in preflight", "accepted")
    raise AssertionError("unreachable")


class _CacheModel:
    """The framework half of the replay: vLLM 0.26.0's block policy, replayed.

    This is the capture stand-in that VLLM-11 and SGL-9 will eventually
    replace with a real adapter observation stream. It lives in the study, not
    in ``simllm.core``, because it makes policy decisions: which blocks are
    reclaimed, how long a prefix hit runs and which block a write lands in.
    The ledger under test consumes what this model emits and never consults
    it.

    The free queue is ordered front-first for eviction
    (``vllm/v1/core/kv_cache_utils.py:195``), allocation pops the front
    (``vllm/v1/core/block_pool.py:661``), a request frees in reverse so its
    tail is reclaimed before its head
    (``vllm/v1/core/single_type_kv_cache_manager.py:503``), unhashed blocks go
    to the front (``vllm/v1/core/block_pool.py:738-740``), and a hit is the
    leading run of chained hashes
    (``vllm/v1/core/single_type_kv_cache_manager.py:708-711``).
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.free_queue: list[str] = [f"b{index}" for index in range(capacity)]
        self.content: dict[str, tuple[str, int]] = {}
        self.index: dict[tuple[str, int], str] = {}

    def longest_hit(self, keys: Sequence[tuple[str, int]]) -> list[str]:
        hit: list[str] = []
        for key in keys:
            block = self.index.get(key)
            if block is None:
                break
            hit.append(block)
        return hit

    def touch(self, blocks: Sequence[str]) -> None:
        for block in blocks:
            self.free_queue.remove(block)

    def allocate(self, count: int) -> tuple[list[str], list[str]]:
        if count > len(self.free_queue):
            raise AssertionError("fixture asked for more blocks than the pool holds")
        taken = self.free_queue[:count]
        del self.free_queue[:count]
        evicted = [block for block in taken if block in self.content]
        for block in evicted:
            self.index.pop(self.content.pop(block), None)
        return evicted, taken

    def cache(self, blocks: Sequence[str], keys: Sequence[tuple[str, int]]) -> None:
        for block, key in zip(blocks, keys):
            self.content[block] = key
            self.index[key] = block

    def release(self, blocks: Sequence[str], full: Sequence[bool]) -> None:
        without: list[str] = []
        with_hash: list[str] = []
        for block, is_full in zip(reversed(list(blocks)), reversed(list(full))):
            (with_hash if is_full else without).append(block)
        for block in without:
            self.content.pop(block, None)
        self.free_queue = without + self.free_queue + with_hash


def _chain(prefix: str, entries: Sequence[tuple[str, Any]], request: str):
    """Return one fully serialized operation chain and its final operation ID.

    Each operation is correlated to the request whose decision it observes, so
    a step that frees one request's blocks while another request prefills
    keeps both identities in the graph.
    """

    from simllm.core import ComputeWork, ExecutionOperation, OperationCorrelation

    operations = []
    previous: str | None = None
    for suffix, work in entries:
        operation_id = f"{prefix}-{suffix}"
        queue = "compute" if isinstance(work, ComputeWork) else "kv"
        owner = getattr(work, "request_id", None) or request
        operations.append(
            ExecutionOperation(
                operation_id,
                0,
                queue,
                work,
                depends_on=() if previous is None else (previous,),
                correlation=OperationCorrelation(request_ids=(owner,)),
            )
        )
        previous = operation_id
    assert previous is not None
    return tuple(operations), previous


def _prefill_entries(
    request: str,
    *,
    total_tokens: int,
    hit_blocks: Sequence[str],
    hit_tokens: int,
    evicted: Sequence[str],
    new_blocks: Sequence[str],
    recompute_tokens: int | None = None,
):
    from simllm.core import ComputeWork, KvCacheAction, KvCacheWork

    def work(action, **kwargs):
        return KvCacheWork(action, POOL_ID, request_id=request, **kwargs)

    entries: list[tuple[str, Any]] = []
    if recompute_tokens is not None:
        entries.append(
            (
                "recompute",
                work(
                    KvCacheAction.RECOMPUTE,
                    token_start=0,
                    token_end=recompute_tokens,
                    cause="preemption",
                ),
            )
        )
    entries.append(
        (
            "reserve",
            work(
                KvCacheAction.RESERVE,
                token_start=hit_tokens,
                token_end=total_tokens,
            ),
        )
    )
    if hit_blocks:
        entries.append(
            (
                "bind",
                work(
                    KvCacheAction.BIND_PREFIX,
                    block_ids=tuple(hit_blocks),
                    token_start=0,
                    token_end=hit_tokens,
                ),
            )
        )
        entries.append(("touch", work(KvCacheAction.TOUCH, block_ids=tuple(hit_blocks))))
    if evicted:
        entries.append(
            (
                "evict",
                KvCacheWork(
                    KvCacheAction.EVICT,
                    POOL_ID,
                    block_ids=tuple(evicted),
                    cause="capacity",
                ),
            )
        )
    computed = total_tokens - hit_tokens
    entries.append(
        (
            "allocate",
            work(
                KvCacheAction.ALLOCATE,
                block_ids=tuple(new_blocks),
                token_start=hit_tokens,
                token_end=total_tokens,
            ),
        )
    )
    entries.append(
        (
            "kernel",
            ComputeWork(
                f"{request}-prefill",
                nominal_duration_ps=computed * PREFILL_PS_PER_TOKEN,
            ),
        )
    )
    entries.append(
        (
            "write",
            work(
                KvCacheAction.WRITE,
                block_ids=tuple(new_blocks),
                token_start=hit_tokens,
                token_end=total_tokens,
                byte_count=computed * BYTES_PER_TOKEN,
            ),
        )
    )
    return entries


def _decode_entries(
    request: str,
    *,
    resident_tokens: int,
    held_blocks: Sequence[str],
    new_block: str | None,
):
    from simllm.core import ComputeWork, KvCacheAction, KvCacheWork

    def work(action, **kwargs):
        return KvCacheWork(action, POOL_ID, request_id=request, **kwargs)

    entries: list[tuple[str, Any]] = []
    if new_block is not None:
        entries.append(
            (
                "reserve",
                work(
                    KvCacheAction.RESERVE,
                    token_start=resident_tokens,
                    token_end=resident_tokens + 1,
                ),
            )
        )
        entries.append(
            (
                "allocate",
                work(
                    KvCacheAction.ALLOCATE,
                    block_ids=(new_block,),
                    token_start=resident_tokens,
                    token_end=resident_tokens + 1,
                ),
            )
        )
    entries.append(
        (
            "read",
            work(
                KvCacheAction.READ,
                block_ids=tuple(held_blocks),
                token_start=0,
                token_end=resident_tokens,
                byte_count=resident_tokens * BYTES_PER_TOKEN,
            ),
        )
    )
    entries.append(
        ("kernel", ComputeWork(f"{request}-decode", nominal_duration_ps=DECODE_STEP_PS))
    )
    tail = new_block if new_block is not None else held_blocks[-1]
    entries.append(
        (
            "write",
            work(
                KvCacheAction.WRITE,
                block_ids=(tail,),
                token_start=resident_tokens,
                token_end=resident_tokens + 1,
                byte_count=BYTES_PER_TOKEN,
            ),
        )
    )
    return entries


def _release_entries(request: str, held_blocks: Sequence[str], suffix="release"):
    from simllm.core import KvCacheAction, KvCacheWork

    return [
        (
            suffix,
            KvCacheWork(
                KvCacheAction.RELEASE,
                POOL_ID,
                request_id=request,
                block_ids=tuple(reversed(list(held_blocks))),
            ),
        )
    ]


def _runtime(capacity: int, rate_bps: int):
    from simllm.core import CoarseDeviceProfile, CoarseDeviceRuntime, KvPoolSpec

    return CoarseDeviceRuntime(
        CoarseDeviceProfile(hbm_rate_bps=rate_bps),
        kv_pools=[
            KvPoolSpec(
                pool_id=POOL_ID,
                block_bytes=BLOCK_BYTES,
                block_tokens=BLOCK_TOKENS,
                capacity_blocks=capacity,
            )
        ],
    )


def _step(
    runtime,
    reducer,
    clock,
    *,
    step_index: int,
    request: str,
    phase_name: str,
    operations,
    completion_id: str,
    num_new_tokens: int,
    num_cached_tokens: int,
    context_length: int,
):
    from simllm.core import ExecutionGraph, RequestPhase, ScheduledRequest, StepRecord

    phase = RequestPhase.PREFILL if phase_name == "prefill" else RequestPhase.DECODE
    record = StepRecord(
        step_index=step_index,
        virtual_time_ps=clock.now_ps,
        scheduled=[
            ScheduledRequest(
                request,
                phase,
                num_new_tokens,
                num_cached_tokens=num_cached_tokens,
                context_length=context_length,
            )
        ],
        num_sampled=1,
        sampled_request_ids=[request],
    )
    graph = ExecutionGraph(
        f"{request}-step-{step_index}",
        step_index,
        clock.now_ps,
        operations,
        completion_operation_ids=(completion_id,),
    )
    execution = runtime.execute(graph)
    return reducer.reduce(record, graph, execution, runtime.last_report)


def _family_a_cell(capacity: int, rate_bps: int) -> dict[str, Any]:
    """Replay A, F and R against one pool capacity and one HBM rate."""

    from simllm.core import CompletionReducer, VirtualClock

    k = kv_ps_per_token(rate_bps)
    cache = _CacheModel(capacity)
    runtime = _runtime(capacity, rate_bps)
    clock = VirtualClock(0)
    reducer = CompletionReducer(clock)

    prefix_blocks = A_PREFIX_TOKENS // BLOCK_TOKENS
    prefix_keys = [("S", index) for index in range(prefix_blocks)]
    a_keys = prefix_keys + [
        ("A", index) for index in range(A_PRIVATE_TOKENS // BLOCK_TOKENS)
    ]
    f_keys = [("F", index) for index in range(F_TOKENS // BLOCK_TOKENS)]
    r_keys = prefix_keys + [
        ("R", index)
        for index in range((R_TOKENS - A_PREFIX_TOKENS) // BLOCK_TOKENS)
    ]

    evictions: dict[str, int] = {}
    for step_index, (request, keys, tokens) in enumerate(
        (
            ("A", a_keys, A_PREFIX_TOKENS + A_PRIVATE_TOKENS),
            ("F", f_keys, F_TOKENS),
        )
    ):
        evicted, taken = cache.allocate(len(keys))
        evictions[request] = len(evicted)
        entries = _prefill_entries(
            request,
            total_tokens=tokens,
            hit_blocks=(),
            hit_tokens=0,
            evicted=evicted,
            new_blocks=taken,
        ) + _release_entries(request, taken)
        operations, completion_id = _chain(request, entries, request)
        _step(
            runtime,
            reducer,
            clock,
            step_index=step_index,
            request=request,
            phase_name="prefill",
            operations=operations,
            completion_id=completion_id,
            num_new_tokens=tokens,
            num_cached_tokens=0,
            context_length=tokens,
        )
        cache.cache(taken, keys)
        cache.release(taken, [True] * len(taken))

    hit_blocks = cache.longest_hit(r_keys)
    hit_tokens = len(hit_blocks) * BLOCK_TOKENS
    cache.touch(hit_blocks)
    evicted, taken = cache.allocate(len(r_keys) - len(hit_blocks))
    entries = _prefill_entries(
        "R",
        total_tokens=R_TOKENS,
        hit_blocks=hit_blocks,
        hit_tokens=hit_tokens,
        evicted=evicted,
        new_blocks=taken,
    )
    operations, completion_id = _chain("R", entries, "R")
    result = _step(
        runtime,
        reducer,
        clock,
        step_index=2,
        request="R",
        phase_name="prefill",
        operations=operations,
        completion_id=completion_id,
        num_new_tokens=R_TOKENS - hit_tokens,
        num_cached_tokens=hit_tokens,
        context_length=R_TOKENS,
    )

    metric = result.request_metrics[0]
    accounting = runtime.last_kv_report.pool(POOL_ID)
    expected_ttft, expected_kv = family_a_expected(capacity, k)
    return {
        "capacity_blocks": capacity,
        "hbm_rate_bps": rate_bps,
        "kv_ps_per_token": k,
        "hit_blocks": len(hit_blocks),
        "hit_tokens": hit_tokens,
        "new_blocks": len(taken),
        "evictions_in_filler_step": evictions["F"],
        "r_write_bytes": (R_TOKENS - hit_tokens) * BYTES_PER_TOKEN,
        "replay_write_bytes": (
            A_PREFIX_TOKENS + A_PRIVATE_TOKENS + F_TOKENS + R_TOKENS - hit_tokens
        )
        * BYTES_PER_TOKEN,
        "ttft_ps": metric.ttft_ps,
        "kv_ps": metric.attribution.kv_ps,
        "kernel_ps": metric.attribution.kernel_ps,
        "queue_ps": metric.attribution.queue_ps,
        "expected_ttft_ps": expected_ttft,
        "expected_kv_ps": expected_kv,
        "expected_kernel_ps": expected_ttft - expected_kv,
        "ledger_prefix_hit_blocks": accounting.prefix_hit_blocks,
        "ledger_prefix_hit_tokens": accounting.prefix_hit_tokens,
        "ledger_write_bytes": accounting.write_bytes,
        "ledger_live_blocks": accounting.live_blocks,
        "ledger_reclaimable_blocks": accounting.reclaimable_blocks,
        "ledger_free_blocks": accounting.free_blocks,
        "ledger_capacity_blocks": accounting.capacity_blocks,
    }


def _family_b_arm(arm: str, rate_bps: int) -> dict[str, Any]:
    """Replay D through a preemption, or through the same steps without one."""

    from simllm.core import CompletionReducer, KvCacheAction, VirtualClock

    k = kv_ps_per_token(rate_bps)
    capacity = B_CAPACITY_TIGHT if arm == "preempted" else B_CAPACITY_LOOSE
    cache = _CacheModel(capacity)
    runtime = _runtime(capacity, rate_bps)
    clock = VirtualClock(0)
    reducer = CompletionReducer(clock)
    step_latencies: list[int] = []

    prompt_blocks = D_PROMPT_TOKENS // BLOCK_TOKENS
    d_keys = [("D", index) for index in range(prompt_blocks + 1)]
    g_keys = [("G", index) for index in range(G_TOKENS // BLOCK_TOKENS)]

    evicted, taken = cache.allocate(prompt_blocks)
    entries = _prefill_entries(
        "D",
        total_tokens=D_PROMPT_TOKENS,
        hit_blocks=(),
        hit_tokens=0,
        evicted=evicted,
        new_blocks=taken,
    )
    operations, completion_id = _chain("D0", entries, "D")
    result = _step(
        runtime,
        reducer,
        clock,
        step_index=0,
        request="D",
        phase_name="prefill",
        operations=operations,
        completion_id=completion_id,
        num_new_tokens=D_PROMPT_TOKENS,
        num_cached_tokens=0,
        context_length=D_PROMPT_TOKENS,
    )
    cache.cache(taken, d_keys[:prompt_blocks])
    held = list(taken)
    ttft_ps = result.request_metrics[0].ttft_ps
    step_latencies.append(result.step_latency_ps)

    resident = D_PROMPT_TOKENS
    for step_index in (1, 2):
        new_block = None
        if step_index == 1:
            _, allocated = cache.allocate(1)
            new_block = allocated[0]
        entries = _decode_entries(
            "D",
            resident_tokens=resident,
            held_blocks=held,
            new_block=new_block,
        )
        if new_block is not None:
            held.append(new_block)
        operations, completion_id = _chain(f"D{step_index}", entries, "D")
        result = _step(
            runtime,
            reducer,
            clock,
            step_index=step_index,
            request="D",
            phase_name="decode",
            operations=operations,
            completion_id=completion_id,
            num_new_tokens=1,
            num_cached_tokens=0,
            context_length=resident + 1,
        )
        step_latencies.append(result.step_latency_ps)
        resident += 1

    entries = []
    preempted = arm == "preempted"
    if preempted:
        full_flags = [block != held[-1] for block in held]
        entries.extend(_release_entries("D", held, "preempt-release"))
        cache.release(held, full_flags)
        held = []
    evicted, taken = cache.allocate(len(g_keys))
    entries.extend(
        _prefill_entries(
            "G",
            total_tokens=G_TOKENS,
            hit_blocks=(),
            hit_tokens=0,
            evicted=evicted,
            new_blocks=taken,
        )
    )
    entries.extend(_release_entries("G", taken))
    operations, completion_id = _chain("G", entries, "G")
    result = _step(
        runtime,
        reducer,
        clock,
        step_index=3,
        request="G",
        phase_name="prefill",
        operations=operations,
        completion_id=completion_id,
        num_new_tokens=G_TOKENS,
        num_cached_tokens=0,
        context_length=G_TOKENS,
    )
    step_latencies.append(result.step_latency_ps)
    cache.cache(taken, g_keys)
    cache.release(taken, [True] * len(taken))
    pressure_evictions = len(evicted)
    pressure_report = runtime.last_kv_report
    d_released_blocks = sum(
        demand.block_count
        for demand in pressure_report.demands
        if demand.action is KvCacheAction.RELEASE and demand.request_id == "D"
    )
    d_freed_blocks = pressure_report.pool(POOL_ID).freed_blocks

    if preempted:
        sequence = resident + 1
        blocks_needed = -(-sequence // BLOCK_TOKENS)
        hit_blocks = cache.longest_hit(d_keys)
        cache.touch(hit_blocks)
        evicted, taken = cache.allocate(blocks_needed - len(hit_blocks))
        entries = _prefill_entries(
            "D",
            total_tokens=sequence,
            hit_blocks=hit_blocks,
            hit_tokens=len(hit_blocks) * BLOCK_TOKENS,
            evicted=evicted,
            new_blocks=taken,
            recompute_tokens=sequence,
        )
        held = list(hit_blocks) + list(taken)
        full = sequence // BLOCK_TOKENS
        cache.cache(held[:full], d_keys[:full])
        phase_name = "prefill"
        new_tokens = sequence - len(hit_blocks) * BLOCK_TOKENS
        resident = sequence
    else:
        entries = _decode_entries(
            "D",
            resident_tokens=resident,
            held_blocks=held,
            new_block=None,
        )
        phase_name = "decode"
        new_tokens = 1
        resident += 1
    operations, completion_id = _chain("D4", entries, "D")
    result = _step(
        runtime,
        reducer,
        clock,
        step_index=4,
        request="D",
        phase_name=phase_name,
        operations=operations,
        completion_id=completion_id,
        num_new_tokens=new_tokens,
        num_cached_tokens=0,
        context_length=resident,
    )
    step_latencies.append(result.step_latency_ps)
    token_4_interval_ps = result.request_metrics[0].latency_ps

    entries = _decode_entries(
        "D",
        resident_tokens=resident,
        held_blocks=held,
        new_block=None,
    )
    operations, completion_id = _chain("D5", entries, "D")
    result = _step(
        runtime,
        reducer,
        clock,
        step_index=5,
        request="D",
        phase_name="decode",
        operations=operations,
        completion_id=completion_id,
        num_new_tokens=1,
        num_cached_tokens=0,
        context_length=resident + 1,
    )
    step_latencies.append(result.step_latency_ps)

    metric = result.request_metrics[0]
    accounting = runtime.last_kv_report.pool(POOL_ID)
    expected_ttft, expected_tpot, expected_interval = family_b_expected(arm, k)
    return {
        "arm": arm,
        "capacity_blocks": capacity,
        "hbm_rate_bps": rate_bps,
        "kv_ps_per_token": k,
        "ttft_ps": ttft_ps,
        "tpot": [metric.tpot_ps.numerator, metric.tpot_ps.denominator],
        "token_4_interval_ps": token_4_interval_ps,
        "step_latencies_ps": step_latencies,
        "expected_ttft_ps": expected_ttft,
        "expected_tpot": [expected_tpot.numerator, expected_tpot.denominator],
        "expected_token_4_interval_ps": expected_interval,
        "expected_step_latencies_ps": list(family_b_steps(arm, k)),
        "pressure_evictions": pressure_evictions,
        "d_released_blocks": d_released_blocks,
        "d_freed_blocks": d_freed_blocks,
        "ledger_released_references": accounting.released_references,
        "ledger_evicted_blocks": accounting.evicted_blocks,
        "ledger_freed_blocks": accounting.freed_blocks,
        "ledger_recomputed_tokens": accounting.recomputed_tokens,
        "ledger_eviction_causes": [list(entry) for entry in accounting.eviction_causes],
    }


def _instance(name: str, expected: object, observed: object) -> dict[str, Any]:
    return {
        "instance": name,
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def _score(family_a, family_b) -> dict[str, Any]:
    """Return the scored instances: exact rows plus the timing-invisible ledger.

    The freeze scored the relation families directly. That reading was too
    generous, because an exact per-cell oracle already pins every relation
    derived from the same measurement: once TTFT equals the frozen table, its
    direction, plateau and bandwidth scaling follow arithmetically. The scored
    set is therefore the exact rows themselves plus the two accounting facts
    that carry no time at all, and the derived relations are reported below as
    entailed consistency rows.
    """

    by_cell = {(row["capacity_blocks"], row["kv_ps_per_token"]): row for row in family_a}
    by_arm = {(row["arm"], row["kv_ps_per_token"]): row for row in family_b}
    instances: list[dict[str, Any]] = []

    for (capacity, k), row in sorted(by_cell.items()):
        instances.append(
            _instance(
                f"A exact C={capacity} k={k}",
                [row["expected_ttft_ps"], row["expected_kv_ps"], row["expected_kernel_ps"]],
                [row["ttft_ps"], row["kv_ps"], row["kernel_ps"]],
            )
        )
    for k in (K_FAST, K_SLOW):
        instances.append(
            _instance(
                f"A eviction ladder k={k}",
                [FROZEN_LADDER[capacity][0] for capacity in CAPACITIES],
                [
                    by_cell[(capacity, k)]["evictions_in_filler_step"]
                    for capacity in CAPACITIES
                ],
            )
        )
    for (arm, k), row in sorted(by_arm.items()):
        instances.append(
            _instance(
                f"B exact {arm} k={k}",
                [
                    row["expected_step_latencies_ps"],
                    row["expected_ttft_ps"],
                    row["expected_tpot"],
                    row["expected_token_4_interval_ps"],
                ],
                [
                    row["step_latencies_ps"],
                    row["ttft_ps"],
                    row["tpot"],
                    row["token_4_interval_ps"],
                ],
            )
        )
    for k in (K_FAST, K_SLOW):
        tight = by_arm[("preempted", k)]
        instances.append(
            _instance(
                f"B preemption accounting k={k}",
                [FROZEN_D_RELEASED_BLOCKS, FROZEN_D_EVICTED_BLOCKS, 1],
                [
                    tight["d_released_blocks"],
                    tight["pressure_evictions"],
                    tight["d_freed_blocks"],
                ],
            )
        )
    passed = sum(1 for entry in instances if entry["passed"])
    return {"instances": instances, "passed": passed, "total": len(instances)}


def _entailed(family_a, family_b) -> dict[str, Any]:
    """Return the registered relations that the exact rows already pin."""

    by_cell = {(row["capacity_blocks"], row["kv_ps_per_token"]): row for row in family_a}
    by_arm = {(row["arm"], row["kv_ps_per_token"]): row for row in family_b}
    rows: list[dict[str, Any]] = []
    for k in (K_FAST, K_SLOW):
        ttfts = [by_cell[(capacity, k)]["ttft_ps"] for capacity in CAPACITIES]
        rows.append(
            _instance(
                f"A1 direction k={k}",
                True,
                all(left >= right for left, right in itertools.pairwise(ttfts))
                and ttfts[0] > ttfts[-1],
            )
        )
        rows.append(
            _instance(
                f"A2 plateau k={k}",
                1,
                len({by_cell[(capacity, k)]["ttft_ps"] for capacity in (48, 56, 64)}),
            )
        )
        rows.append(
            _instance(
                f"A6 hit ladder k={k}",
                [FROZEN_LADDER[capacity][1] for capacity in CAPACITIES],
                [by_cell[(capacity, k)]["hit_blocks"] for capacity in CAPACITIES],
            )
        )
        loose = by_arm[("unconstrained", k)]
        tight = by_arm[("preempted", k)]
        rows.append(
            _instance(
                f"B1 direction k={k}",
                True,
                Fraction(*tight["tpot"]) > Fraction(*loose["tpot"]),
            )
        )
        rows.append(
            _instance(
                f"B2 delta k={k}",
                FROZEN_TPOT_DELTA_PS,
                int(Fraction(*tight["tpot"]) - Fraction(*loose["tpot"])),
            )
        )
        rows.append(
            _instance(f"B4 ttft unchanged k={k}", loose["ttft_ps"], tight["ttft_ps"])
        )
        rows.append(
            _instance(
                f"B5 unconstrained control k={k}",
                [0, 0, 0],
                [
                    loose["pressure_evictions"],
                    loose["ledger_recomputed_tokens"],
                    loose["d_freed_blocks"],
                ],
            )
        )
    rows.append(
        _instance(
            "A4 bandwidth doubles the KV term and leaves the kernel term alone",
            [
                (2 * by_cell[(capacity, K_FAST)]["kv_ps"],
                 by_cell[(capacity, K_FAST)]["kernel_ps"])
                for capacity in CAPACITIES
            ],
            [
                (by_cell[(capacity, K_SLOW)]["kv_ps"],
                 by_cell[(capacity, K_SLOW)]["kernel_ps"])
                for capacity in CAPACITIES
            ],
        )
    )
    rows.append(
        _instance(
            "A7 constrained TTFT is exactly twice the unconstrained TTFT",
            [[2, 1], [2, 1]],
            [
                [
                    Fraction(
                        by_cell[(32, k)]["ttft_ps"], by_cell[(64, k)]["ttft_ps"]
                    ).numerator,
                    Fraction(
                        by_cell[(32, k)]["ttft_ps"], by_cell[(64, k)]["ttft_ps"]
                    ).denominator,
                ]
                for k in (K_FAST, K_SLOW)
            ],
        )
    )
    rows.append(
        _instance(
            "B3 the TPOT rise does not move with HBM bandwidth",
            int(
                Fraction(*by_arm[("preempted", K_FAST)]["tpot"])
                - Fraction(*by_arm[("unconstrained", K_FAST)]["tpot"])
            ),
            int(
                Fraction(*by_arm[("preempted", K_SLOW)]["tpot"])
                - Fraction(*by_arm[("unconstrained", K_SLOW)]["tpot"])
            ),
        )
    )
    held = sum(1 for row in rows if row["passed"])
    return {"rows": rows, "held": held, "total": len(rows)}


def _fatal_guards(family_a, family_b) -> dict[str, Any]:
    """Return the fatal, unscored guards; a violation voids the run."""

    guards: list[dict[str, Any]] = []
    for row in family_a:
        name = f"A C={row['capacity_blocks']} k={row['kv_ps_per_token']}"
        guards.append(
            _instance(
                f"{name} conservation",
                row["ledger_capacity_blocks"],
                row["ledger_live_blocks"]
                + row["ledger_reclaimable_blocks"]
                + row["ledger_free_blocks"],
            )
        )
        guards.append(_instance(f"{name} zero queue", 0, row["queue_ps"]))
        guards.append(
            _instance(
                f"{name} ledger write bytes",
                row["replay_write_bytes"],
                row["ledger_write_bytes"],
            )
        )
        guards.append(
            _instance(
                f"{name} ledger hit tokens",
                row["hit_tokens"],
                row["ledger_prefix_hit_tokens"],
            )
        )
    by_arm = {(row["arm"], row["kv_ps_per_token"]): row for row in family_b}
    for row in family_b:
        name = f"B {row['arm']} k={row['kv_ps_per_token']}"
        expected_recompute = 259 if row["arm"] == "preempted" else 0
        guards.append(
            _instance(
                f"{name} recomputed tokens",
                expected_recompute,
                row["ledger_recomputed_tokens"],
            )
        )
    for k in (K_FAST, K_SLOW):
        guards.append(
            _instance(
                f"B pressure step identical across arms k={k}",
                by_arm[("unconstrained", k)]["step_latencies_ps"][3],
                by_arm[("preempted", k)]["step_latencies_ps"][3],
            )
        )
        guards.append(
            _instance(
                f"B first three steps identical across arms k={k}",
                by_arm[("unconstrained", k)]["step_latencies_ps"][:3],
                by_arm[("preempted", k)]["step_latencies_ps"][:3],
            )
        )
    violated = [guard["instance"] for guard in guards if not guard["passed"]]
    return {"guards": guards, "violated": violated, "total": len(guards)}


def _run(out: Path) -> dict[str, Any]:
    rates = (HBM_RATE_FAST_BPS, HBM_RATE_SLOW_BPS)
    family_a = [
        _family_a_cell(capacity, rate) for rate in rates for capacity in CAPACITIES
    ]
    family_b = [
        _family_b_arm(arm, rate)
        for rate in rates
        for arm in ("unconstrained", "preempted")
    ]
    scored = _score(family_a, family_b)
    entailed = _entailed(family_a, family_b)
    fatal = _fatal_guards(family_a, family_b)
    print(
        f"scored {scored['passed']} of {scored['total']} genuine-risk instances; "
        f"{entailed['held']} of {entailed['total']} entailed relations held; "
        f"{len(fatal['violated'])} of {fatal['total']} fatal guards violated"
    )
    for entry in scored["instances"]:
        if not entry["passed"]:
            print(f"  FAILED {entry['instance']}: {entry['expected']} != {entry['observed']}")
    for entry in entailed["rows"]:
        if not entry["passed"]:
            print(f"  BROKEN {entry['instance']}: {entry['expected']} != {entry['observed']}")
    for guard in fatal["guards"]:
        if not guard["passed"]:
            print(f"  VOID {guard['instance']}: {guard['expected']} != {guard['observed']}")
    return {
        "family_a": family_a,
        "family_b": family_b,
        "scored": scored,
        "entailed": entailed,
        "fatal": fatal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate the frozen registry and exit without running the study",
    )
    args = parser.parse_args()
    if args.check_only:
        check_only()
        return
    if args.out is None:
        parser.error("--out is required unless --check-only is given")
    payload = _run(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
