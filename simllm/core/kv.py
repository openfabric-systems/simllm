"""Explicit KV-cache lifecycle accounting, consumed before resource contention.

A serving framework's KV cache decides whether a request is admitted, when it
is admitted, and whether it has to recompute.  ``KvCacheWork`` carries the
framework's observed decisions; this module is the authority that replays them
against a pool of a fixed size and reports what they cost.

The module holds no policy.  It never decides which block to evict, how long a
prefix hit is, or whether a request is preempted: those are framework
decisions, observed by an adapter and replayed here.  What it does own is the
pool state that those decisions imply, the invariants a legal observation
stream must satisfy, and the byte demand each byte-carrying observation places
on memory.  The device runtime turns that demand into service time; nothing in
this module knows about queues, contention or picoseconds.

The vocabulary's semantics are derived from vLLM 0.26.0.  Paths below are
relative to the installed ``vllm`` package root.

``RESERVE``
    The capacity question ``KVCacheManager.allocate_slots`` asks before it
    allocates: ``available_blocks = self.block_pool.get_num_free_blocks() -
    reserved_blocks`` and ``if required_blocks > available_blocks: return
    None`` (``vllm/v1/core/kv_cache_manager.py:462-466``).  A refusal is what
    drives preemption, so the demand is accounted before any block moves.
``ALLOCATE``
    ``BlockPool.get_new_blocks`` pops the least recently used blocks from the
    free queue and raises their reference count
    (``vllm/v1/core/block_pool.py:661-668``).  A popped block that still held
    cached content is evicted first, in
    ``_maybe_evict_cached_block`` (``block_pool.py:666``, ``:679``), which is
    why this ledger requires an explicit ``EVICT`` before a cached block may be
    reallocated: vLLM's eviction is lazy and would otherwise be invisible.
``BIND_PREFIX``
    The reuse decision, ``find_longest_cache_hit``
    (``vllm/v1/core/single_type_kv_cache_manager.py:658``), whose result the
    scheduler realizes by fast-forwarding the token cursor
    (``vllm/v1/core/sched/scheduler.py:1031``).  Chained hashes make a hit a
    prefix run (``single_type_kv_cache_manager.py:708-711``), and a full hit
    still recomputes one token (``kv_cache_manager.py:237``).
``TOUCH`` and ``RETAIN``
    The two branches of ``BlockPool.touch`` (``block_pool.py:702-715``).  With
    ``ref_cnt == 0`` the block leaves the free queue and stops being an
    eviction candidate, which is ``TOUCH``; with ``ref_cnt > 0`` another owner
    simply joins, which is ``RETAIN``.  Splitting them keeps a prefix hit from
    being counted twice, once as a decision and once as a mechanism.
``READ`` and ``WRITE``
    Physical KV traffic against blocks the request holds.  One block costs
    ``2 * block_size * num_kv_heads * head_dim * dtype_size`` per layer
    (``vllm/v1/kv_cache_interface.py:203-218``).
``RELEASE``
    ``BlockPool.free_blocks`` decrements the reference count and, at zero,
    returns the block to the free queue: hashed blocks keep their content and
    are appended, unhashed blocks are prepended so they are consumed first
    (``block_pool.py:731-740``).  A request frees in reverse order so its tail
    is reclaimed before its head (``single_type_kv_cache_manager.py:503``),
    which is what lets a shared prefix outlive the request that created it.
``EVICT`` and ``FREE``
    Both discard reusable content at reference count zero.  ``EVICT`` is
    involuntary reclamation and carries a cause; ``FREE`` is the voluntary
    return of a block that holds nothing reusable, the
    ``prepend_n(blocks_without_hash)`` half of ``free_blocks`` and
    ``reset_prefix_cache`` (``block_pool.py:761-785``).
``SWAP`` and ``TRANSFER``
    Tier movement and cross-pool movement.  vLLM v1 never swaps to recover
    memory: preemption is recompute-only
    (``vllm/v1/core/sched/scheduler.py:1212-1233``, and ``PreemptionMode`` no
    longer exists), so these are exercised only by the opt-in offload
    connector under ``vllm/v1/kv_offload/``.
``RECOMPUTE``
    The work a miss or a preemption forces back onto the model.  Preemption
    frees the blocks and sets ``request.num_computed_tokens = 0``
    (``scheduler.py:1221-1225``).
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from simllm.core.execution import KvCacheAction, KvCacheWork

#: Versioned name of the accounting report this module produces.
KV_ACCOUNTING_SCHEMA = "simllm-kv-accounting-v1"

#: Actions that may carry a positive byte count.
BYTE_CARRYING_ACTIONS = frozenset(
    {
        KvCacheAction.READ,
        KvCacheAction.WRITE,
        KvCacheAction.SWAP,
        KvCacheAction.TRANSFER,
    }
)


class KvBlockState(str, enum.Enum):
    """The three states one pool block can occupy."""

    #: reference count zero, holds nothing reusable
    FREE = "free"
    #: reference count at least one, held by one or more requests
    LIVE = "live"
    #: reference count zero, still holds reusable content, evictable
    RECLAIMABLE = "reclaimable"


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _require_int(name: str, value: object, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class KvPoolSpec:
    """Fixed geometry and capacity of one KV block pool on one rank.

    ``capacity_blocks`` is the pool's whole block array
    (``vllm/v1/core/block_pool.py:175``); it never grows at run time.
    ``block_bytes`` is the pool-wide cost of one block across every layer the
    pool serves, i.e. ``num_layers`` times the per-layer page size.
    """

    pool_id: str
    block_bytes: int
    block_tokens: int
    capacity_blocks: int
    tier: str = "device"

    def __post_init__(self) -> None:
        _require_text("pool_id", self.pool_id)
        _require_text("tier", self.tier)
        _require_int("block_bytes", self.block_bytes, positive=True)
        _require_int("block_tokens", self.block_tokens, positive=True)
        _require_int("capacity_blocks", self.capacity_blocks, positive=True)
        if self.block_bytes % self.block_tokens:
            raise ValueError(
                "block_bytes must divide evenly into block_tokens so one token "
                "has an exact byte cost"
            )

    @property
    def bytes_per_token(self) -> int:
        """Return the exact byte cost of one token of KV in this pool."""

        return self.block_bytes // self.block_tokens

    @property
    def capacity_bytes(self) -> int:
        """Return the pool's whole capacity in bytes."""

        return self.capacity_blocks * self.block_bytes


@dataclass(frozen=True)
class KvServiceDemand:
    """The memory demand one observed KV operation places, before contention.

    ``byte_count`` is zero for a pure lifecycle observation.  The runtime is
    free to lower a zero-byte demand as a timing-neutral marker, which is what
    keeps metadata-only bookkeeping free of invented cost.
    """

    operation_id: str
    pool_id: str
    action: KvCacheAction
    byte_count: int
    tier: str
    request_id: str | None = None
    block_count: int = 0


@dataclass(frozen=True)
class KvPoolAccounting:
    """Cumulative accounting and current occupancy of one pool."""

    pool_id: str
    capacity_blocks: int
    block_bytes: int
    block_tokens: int
    live_blocks: int
    reclaimable_blocks: int
    free_blocks: int
    reserved_blocks: int
    peak_live_blocks: int
    allocated_blocks: int
    released_references: int
    evicted_blocks: int
    freed_blocks: int
    prefix_hit_blocks: int
    prefix_hit_tokens: int
    touched_blocks: int
    retained_blocks: int
    read_bytes: int
    write_bytes: int
    swap_bytes: int
    transfer_bytes: int
    recomputed_tokens: int
    eviction_causes: tuple[tuple[str, int], ...] = ()

    @property
    def live_bytes(self) -> int:
        return self.live_blocks * self.block_bytes

    @property
    def reclaimable_bytes(self) -> int:
        return self.reclaimable_blocks * self.block_bytes

    @property
    def free_bytes(self) -> int:
        return self.free_blocks * self.block_bytes


@dataclass(frozen=True)
class KvAccountingReport:
    """One read-only projection of the ledger after a consumed batch."""

    schema: str
    pools: tuple[KvPoolAccounting, ...]
    demands: tuple[KvServiceDemand, ...]

    def pool(self, pool_id: str) -> KvPoolAccounting:
        """Return the accounting row for ``pool_id``."""

        for row in self.pools:
            if row.pool_id == pool_id:
                return row
        raise KeyError(f"pool {pool_id!r} is not accounted")

    def demand(self, operation_id: str) -> KvServiceDemand | None:
        """Return the demand of ``operation_id``, or None when it has none."""

        for demand in self.demands:
            if demand.operation_id == operation_id:
                return demand
        return None


@dataclass
class _Block:
    """Mutable state of one block that is not free."""

    block_id: str
    #: tokens whose KV has been written into this block
    filled_tokens: int = 0
    #: the block is full, so its content is reusable by another request
    content: bool = False
    tier: str = "device"
    owners: dict[str, int] = field(default_factory=dict)

    @property
    def ref_count(self) -> int:
        return sum(self.owners.values())

    def state(self) -> KvBlockState:
        if self.ref_count > 0:
            return KvBlockState.LIVE
        return KvBlockState.RECLAIMABLE if self.content else KvBlockState.FREE

    def clone(self) -> _Block:
        return _Block(
            block_id=self.block_id,
            filled_tokens=self.filled_tokens,
            content=self.content,
            tier=self.tier,
            owners=dict(self.owners),
        )


@dataclass
class _PoolState:
    """Mutable state and cumulative counters of one pool."""

    spec: KvPoolSpec
    blocks: dict[str, _Block] = field(default_factory=dict)
    reservations: dict[str, int] = field(default_factory=dict)
    peak_live_blocks: int = 0
    allocated_blocks: int = 0
    released_references: int = 0
    evicted_blocks: int = 0
    freed_blocks: int = 0
    prefix_hit_blocks: int = 0
    prefix_hit_tokens: int = 0
    touched_blocks: int = 0
    retained_blocks: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    swap_bytes: int = 0
    transfer_bytes: int = 0
    recomputed_tokens: int = 0
    eviction_causes: dict[str, int] = field(default_factory=dict)

    def clone(self) -> _PoolState:
        copy = _PoolState(spec=self.spec)
        copy.blocks = {key: block.clone() for key, block in self.blocks.items()}
        copy.reservations = dict(self.reservations)
        copy.eviction_causes = dict(self.eviction_causes)
        for name in (
            "peak_live_blocks",
            "allocated_blocks",
            "released_references",
            "evicted_blocks",
            "freed_blocks",
            "prefix_hit_blocks",
            "prefix_hit_tokens",
            "touched_blocks",
            "retained_blocks",
            "read_bytes",
            "write_bytes",
            "swap_bytes",
            "transfer_bytes",
            "recomputed_tokens",
        ):
            setattr(copy, name, getattr(self, name))
        return copy

    def live_blocks(self) -> int:
        return sum(1 for block in self.blocks.values() if block.ref_count > 0)

    def reclaimable_blocks(self) -> int:
        return sum(
            1
            for block in self.blocks.values()
            if block.ref_count == 0 and block.content
        )

    def accounting(self) -> KvPoolAccounting:
        live = self.live_blocks()
        reclaimable = self.reclaimable_blocks()
        return KvPoolAccounting(
            pool_id=self.spec.pool_id,
            capacity_blocks=self.spec.capacity_blocks,
            block_bytes=self.spec.block_bytes,
            block_tokens=self.spec.block_tokens,
            live_blocks=live,
            reclaimable_blocks=reclaimable,
            free_blocks=self.spec.capacity_blocks - live - reclaimable,
            reserved_blocks=sum(self.reservations.values()),
            peak_live_blocks=self.peak_live_blocks,
            allocated_blocks=self.allocated_blocks,
            released_references=self.released_references,
            evicted_blocks=self.evicted_blocks,
            freed_blocks=self.freed_blocks,
            prefix_hit_blocks=self.prefix_hit_blocks,
            prefix_hit_tokens=self.prefix_hit_tokens,
            touched_blocks=self.touched_blocks,
            retained_blocks=self.retained_blocks,
            read_bytes=self.read_bytes,
            write_bytes=self.write_bytes,
            swap_bytes=self.swap_bytes,
            transfer_bytes=self.transfer_bytes,
            recomputed_tokens=self.recomputed_tokens,
            eviction_causes=tuple(sorted(self.eviction_causes.items())),
        )


class KvLifecycleLedger:
    """Replay observed KV lifecycle decisions against fixed-size pools.

    The ledger is the sole mutable authority for pool state.  Its report is a
    read-only projection, and the byte demands it returns are the only thing a
    runtime may turn into service time.  Consuming a batch is transactional
    through :meth:`clone`: a caller validates on a copy and adopts it only when
    the whole batch is legal, so a refused graph never leaves half-applied
    state behind.
    """

    def __init__(self, specs: Iterable[KvPoolSpec]) -> None:
        pools: dict[str, _PoolState] = {}
        for spec in specs:
            if not isinstance(spec, KvPoolSpec):
                raise TypeError("specs must contain KvPoolSpec records")
            if spec.pool_id in pools:
                raise ValueError(f"duplicate KV pool {spec.pool_id!r}")
            pools[spec.pool_id] = _PoolState(spec=spec)
        if not pools:
            raise ValueError("a KV ledger requires at least one pool")
        self._pools = pools

    @property
    def pool_ids(self) -> tuple[str, ...]:
        """Return every accounted pool identifier in declaration order."""

        return tuple(self._pools)

    def spec(self, pool_id: str) -> KvPoolSpec:
        """Return the fixed geometry of ``pool_id``."""

        return self._pool(pool_id).spec

    def clone(self) -> KvLifecycleLedger:
        """Return an independent copy for transactional consumption."""

        copy = KvLifecycleLedger(pool.spec for pool in self._pools.values())
        copy._pools = {key: pool.clone() for key, pool in self._pools.items()}
        return copy

    def block_state(self, pool_id: str, block_id: str) -> KvBlockState:
        """Return the current state of one block."""

        pool = self._pool(pool_id)
        block = pool.blocks.get(block_id)
        return KvBlockState.FREE if block is None else block.state()

    def report(self, demands: Sequence[KvServiceDemand] = ()) -> KvAccountingReport:
        """Return the accounting projection, optionally with a batch's demands."""

        return KvAccountingReport(
            schema=KV_ACCOUNTING_SCHEMA,
            pools=tuple(pool.accounting() for pool in self._pools.values()),
            demands=tuple(demands),
        )

    def consume(
        self,
        operations: Iterable[tuple[str, KvCacheWork]],
    ) -> tuple[KvServiceDemand, ...]:
        """Observe an ordered batch and return each operation's byte demand."""

        return tuple(
            self.observe(work, operation_id=operation_id)
            for operation_id, work in operations
        )

    def observe(self, work: KvCacheWork, *, operation_id: str) -> KvServiceDemand:
        """Apply one observed lifecycle decision and return its byte demand."""

        _require_text("operation_id", operation_id)
        if not isinstance(work, KvCacheWork):
            raise TypeError("work must be a KvCacheWork record")
        pool = self._pool(work.pool_id)
        tokens = self._token_span(work)
        self._check_bytes(pool, work, tokens)

        handler = getattr(self, _HANDLER_NAMES[work.action])
        handler(pool, work, tokens)

        self._check_reference_count(pool, work)
        self._check_conservation(pool)
        live = pool.live_blocks()
        pool.peak_live_blocks = max(pool.peak_live_blocks, live)
        return KvServiceDemand(
            operation_id=operation_id,
            pool_id=work.pool_id,
            action=work.action,
            byte_count=work.byte_count,
            tier=self._demand_tier(pool, work),
            request_id=work.request_id,
            block_count=len(work.block_ids),
        )

    # -- validation helpers ------------------------------------------------

    def _pool(self, pool_id: str) -> _PoolState:
        pool = self._pools.get(pool_id)
        if pool is None:
            raise ValueError(f"KV pool {pool_id!r} is not configured")
        return pool

    @staticmethod
    def _token_span(work: KvCacheWork) -> int | None:
        if work.token_start is None and work.token_end is None:
            return None
        if work.token_start is None or work.token_end is None:
            raise ValueError(
                f"{work.action.value} declares half a token interval; supply both "
                "token_start and token_end or neither"
            )
        _require_int("token_start", work.token_start)
        _require_int("token_end", work.token_end)
        if work.token_end <= work.token_start:
            raise ValueError(f"{work.action.value} token interval must be positive")
        return work.token_end - work.token_start

    @staticmethod
    def _require_request(work: KvCacheWork) -> str:
        if work.request_id is None:
            raise ValueError(f"{work.action.value} requires an owning request")
        return work.request_id

    @staticmethod
    def _require_blocks(work: KvCacheWork) -> tuple[str, ...]:
        if not work.block_ids:
            raise ValueError(f"{work.action.value} requires at least one block")
        if len(set(work.block_ids)) != len(work.block_ids):
            raise ValueError(f"{work.action.value} names a block twice")
        for block_id in work.block_ids:
            _require_text("block_id", block_id)
        return work.block_ids

    def _check_bytes(
        self,
        pool: _PoolState,
        work: KvCacheWork,
        tokens: int | None,
    ) -> None:
        _require_int("byte_count", work.byte_count)
        if work.action not in BYTE_CARRYING_ACTIONS:
            if work.byte_count:
                raise ValueError(
                    f"{work.action.value} is a metadata observation and must carry "
                    "no bytes"
                )
            return
        if work.byte_count == 0:
            return
        if tokens is not None:
            expected = tokens * pool.spec.bytes_per_token
            if work.byte_count != expected:
                raise ValueError(
                    f"{work.action.value} of {tokens} tokens must move {expected} "
                    f"bytes, not {work.byte_count}"
                )
        if work.block_ids:
            ceiling = len(work.block_ids) * pool.spec.block_bytes
            if work.byte_count > ceiling:
                raise ValueError(
                    f"{work.action.value} moves {work.byte_count} bytes through "
                    f"{len(work.block_ids)} blocks holding {ceiling} bytes"
                )

    def _check_reference_count(self, pool: _PoolState, work: KvCacheWork) -> None:
        if work.reference_count is None:
            return
        _require_int("reference_count", work.reference_count)
        if not work.block_ids:
            raise ValueError(
                "reference_count describes named blocks and needs at least one"
            )
        observed = {
            self._block_ref_count(pool, block_id) for block_id in work.block_ids
        }
        if len(observed) != 1:
            raise ValueError(
                "reference_count is ambiguous across blocks with different counts"
            )
        if observed.pop() != work.reference_count:
            raise ValueError(
                f"observed reference_count {work.reference_count} disagrees with the "
                "accounted count"
            )

    @staticmethod
    def _block_ref_count(pool: _PoolState, block_id: str) -> int:
        block = pool.blocks.get(block_id)
        return 0 if block is None else block.ref_count

    @staticmethod
    def _check_conservation(pool: _PoolState) -> None:
        live = pool.live_blocks()
        reclaimable = pool.reclaimable_blocks()
        free = pool.spec.capacity_blocks - live - reclaimable
        if free < 0:
            raise ValueError(
                f"pool {pool.spec.pool_id!r} holds {live + reclaimable} blocks above "
                f"its capacity of {pool.spec.capacity_blocks}"
            )

    @staticmethod
    def _demand_tier(pool: _PoolState, work: KvCacheWork) -> str:
        for block_id in work.block_ids:
            block = pool.blocks.get(block_id)
            if block is not None:
                return block.tier
        return pool.spec.tier

    # -- action handlers ---------------------------------------------------

    def _observe_reserve(
        self,
        pool: _PoolState,
        work: KvCacheWork,
        tokens: int | None,
    ) -> None:
        request_id = self._require_request(work)
        if work.block_ids:
            raise ValueError("reserve declares a demand and names no blocks")
        if tokens is None:
            raise ValueError("reserve requires the token interval it must cover")
        demand = -(-tokens // pool.spec.block_tokens)
        available = pool.spec.capacity_blocks - pool.live_blocks()
        outstanding = sum(pool.reservations.values())
        if demand + outstanding > available:
            raise ValueError(
                f"request {request_id!r} reserves {demand} blocks with {outstanding} "
                f"already outstanding, above the {available} the pool can supply"
            )
        pool.reservations[request_id] = pool.reservations.get(request_id, 0) + demand

    def _observe_allocate(
        self,
        pool: _PoolState,
        work: KvCacheWork,
        tokens: int | None,
    ) -> None:
        request_id = self._require_request(work)
        block_ids = self._require_blocks(work)
        for block_id in block_ids:
            block = pool.blocks.get(block_id)
            if block is not None and block.state() is not KvBlockState.FREE:
                raise ValueError(
                    f"allocate names block {block_id!r} in state "
                    f"{block.state().value}; a cached block needs an explicit evict"
                )
        outstanding = pool.reservations.get(request_id)
        if outstanding is not None:
            if len(block_ids) > outstanding:
                raise ValueError(
                    f"request {request_id!r} allocates {len(block_ids)} blocks above "
                    f"its outstanding reservation of {outstanding}"
                )
            remaining = outstanding - len(block_ids)
            if remaining:
                pool.reservations[request_id] = remaining
            else:
                del pool.reservations[request_id]
        for block_id in block_ids:
            block = _Block(block_id=block_id, tier=pool.spec.tier)
            block.owners[request_id] = 1
            pool.blocks[block_id] = block
        pool.allocated_blocks += len(block_ids)

    def _observe_bind_prefix(
        self,
        pool: _PoolState,
        work: KvCacheWork,
        tokens: int | None,
    ) -> None:
        self._require_request(work)
        block_ids = self._require_blocks(work)
        for block_id in block_ids:
            block = pool.blocks.get(block_id)
            if block is None or not block.content:
                raise ValueError(
                    f"bind-prefix names block {block_id!r}, which holds no reusable "
                    "content"
                )
        pool.prefix_hit_blocks += len(block_ids)
        if tokens is not None:
            pool.prefix_hit_tokens += tokens

    def _observe_touch(
        self,
        pool: _PoolState,
        work: KvCacheWork,
        tokens: int | None,
    ) -> None:
        request_id = self._require_request(work)
        block_ids = self._require_blocks(work)
        for block_id in block_ids:
            block = pool.blocks.get(block_id)
            if block is None or block.state() is not KvBlockState.RECLAIMABLE:
                state = KvBlockState.FREE.value if block is None else block.state().value
                raise ValueError(
                    f"touch names block {block_id!r} in state {state}; touch takes a "
                    "reclaimable block out of the eviction candidates"
                )
            block.owners[request_id] = block.owners.get(request_id, 0) + 1
        pool.touched_blocks += len(block_ids)

    def _observe_retain(
        self,
        pool: _PoolState,
        work: KvCacheWork,
        tokens: int | None,
    ) -> None:
        request_id = self._require_request(work)
        block_ids = self._require_blocks(work)
        for block_id in block_ids:
            block = pool.blocks.get(block_id)
            if block is None or block.state() is not KvBlockState.LIVE:
                state = KvBlockState.FREE.value if block is None else block.state().value
                raise ValueError(
                    f"retain names block {block_id!r} in state {state}; retain adds an "
                    "owner to a block that is already live"
                )
            block.owners[request_id] = block.owners.get(request_id, 0) + 1
        pool.retained_blocks += len(block_ids)

    def _observe_traffic(
        self,
        pool: _PoolState,
        work: KvCacheWork,
        tokens: int | None,
    ) -> None:
        request_id = self._require_request(work)
        block_ids = self._require_blocks(work)
        blocks = [
            self._require_owned(pool, block_id, request_id, work.action.value)
            for block_id in block_ids
        ]
        moved_tokens = self._moved_tokens(pool, work, tokens)
        if work.action is KvCacheAction.WRITE:
            self._fill(pool, blocks, moved_tokens)
            pool.write_bytes += work.byte_count
        elif work.action is KvCacheAction.READ:
            resident = sum(block.filled_tokens for block in blocks)
            if moved_tokens > resident:
                raise ValueError(
                    f"read of {moved_tokens} tokens exceeds the {resident} tokens "
                    "resident in the blocks it names"
                )
            pool.read_bytes += work.byte_count
        else:
            pool.transfer_bytes += work.byte_count

    @staticmethod
    def _moved_tokens(
        pool: _PoolState,
        work: KvCacheWork,
        tokens: int | None,
    ) -> int:
        if tokens is not None:
            return tokens
        if work.byte_count % pool.spec.bytes_per_token:
            raise ValueError(
                f"{work.action.value} moves {work.byte_count} bytes, which is not a "
                "whole number of tokens"
            )
        return work.byte_count // pool.spec.bytes_per_token

    @staticmethod
    def _fill(pool: _PoolState, blocks: list[_Block], tokens: int) -> None:
        """Append ``tokens`` of written KV across ``blocks`` in token order.

        A block becomes reusable only once it is full, which is when vLLM
        hashes and caches it (``vllm/v1/core/block_pool.py:225``).  A partially
        filled tail block therefore never survives its owner's release, and
        that asymmetry is what separates ``FREE`` from ``EVICT``.
        """

        remaining = tokens
        for block in blocks:
            if remaining == 0:
                break
            room = pool.spec.block_tokens - block.filled_tokens
            taken = min(room, remaining)
            block.filled_tokens += taken
            remaining -= taken
            if block.filled_tokens == pool.spec.block_tokens:
                block.content = True
        if remaining:
            raise ValueError(
                f"write leaves {remaining} tokens with no slot in the blocks it names"
            )

    def _observe_release(
        self,
        pool: _PoolState,
        work: KvCacheWork,
        tokens: int | None,
    ) -> None:
        request_id = self._require_request(work)
        block_ids = self._require_blocks(work)
        for block_id in block_ids:
            block = self._require_owned(pool, block_id, request_id, "release")
            held = block.owners[request_id]
            if held == 1:
                del block.owners[request_id]
            else:
                block.owners[request_id] = held - 1
            if block.ref_count == 0 and not block.content:
                del pool.blocks[block_id]
                pool.freed_blocks += 1
        pool.released_references += len(block_ids)

    def _observe_evict(
        self,
        pool: _PoolState,
        work: KvCacheWork,
        tokens: int | None,
    ) -> None:
        block_ids = self._require_blocks(work)
        if work.cause is None:
            raise ValueError("evict requires a cause; reclamation is never anonymous")
        for block_id in block_ids:
            block = pool.blocks.get(block_id)
            if block is None or block.state() is not KvBlockState.RECLAIMABLE:
                state = KvBlockState.FREE.value if block is None else block.state().value
                raise ValueError(
                    f"evict names block {block_id!r} in state {state}; only a "
                    "reclaimable block may be reclaimed"
                )
            del pool.blocks[block_id]
        pool.evicted_blocks += len(block_ids)
        pool.eviction_causes[work.cause] = pool.eviction_causes.get(work.cause, 0) + len(
            block_ids
        )

    def _observe_free(
        self,
        pool: _PoolState,
        work: KvCacheWork,
        tokens: int | None,
    ) -> None:
        block_ids = self._require_blocks(work)
        removed = 0
        for block_id in block_ids:
            block = pool.blocks.get(block_id)
            if block is None:
                continue
            if block.ref_count:
                raise ValueError(
                    f"free names block {block_id!r} while {block.ref_count} owners "
                    "still hold it"
                )
            del pool.blocks[block_id]
            removed += 1
        pool.freed_blocks += removed

    def _observe_swap(
        self,
        pool: _PoolState,
        work: KvCacheWork,
        tokens: int | None,
    ) -> None:
        block_ids = self._require_blocks(work)
        if work.cause is None:
            raise ValueError("swap requires a cause naming the destination tier")
        for block_id in block_ids:
            block = pool.blocks.get(block_id)
            if block is None:
                raise ValueError(f"swap names block {block_id!r}, which holds nothing")
            if block.tier == work.cause:
                raise ValueError(
                    f"swap moves block {block_id!r} to the tier it already occupies"
                )
            block.tier = work.cause
        pool.swap_bytes += work.byte_count

    def _observe_recompute(
        self,
        pool: _PoolState,
        work: KvCacheWork,
        tokens: int | None,
    ) -> None:
        self._require_request(work)
        if tokens is None:
            raise ValueError("recompute requires the token interval it replays")
        if work.block_ids:
            raise ValueError(
                "recompute restores tokens rather than blocks; the allocation that "
                "follows names the blocks"
            )
        pool.recomputed_tokens += tokens

    @staticmethod
    def _require_owned(
        pool: _PoolState,
        block_id: str,
        request_id: str,
        action: str,
    ) -> _Block:
        block = pool.blocks.get(block_id)
        if block is None or request_id not in block.owners:
            raise ValueError(
                f"{action} names block {block_id!r}, which request {request_id!r} "
                "does not hold"
            )
        return block

#: One handler per observed action; the ledger dispatches on this table so an
#: unhandled vocabulary member is a construction error rather than a silent
#: no-op that would leave the pool state stale.
_HANDLER_NAMES: dict[KvCacheAction, str] = {
    KvCacheAction.RESERVE: "_observe_reserve",
    KvCacheAction.ALLOCATE: "_observe_allocate",
    KvCacheAction.BIND_PREFIX: "_observe_bind_prefix",
    KvCacheAction.TOUCH: "_observe_touch",
    KvCacheAction.RETAIN: "_observe_retain",
    KvCacheAction.READ: "_observe_traffic",
    KvCacheAction.WRITE: "_observe_traffic",
    KvCacheAction.TRANSFER: "_observe_traffic",
    KvCacheAction.RELEASE: "_observe_release",
    KvCacheAction.EVICT: "_observe_evict",
    KvCacheAction.FREE: "_observe_free",
    KvCacheAction.SWAP: "_observe_swap",
    KvCacheAction.RECOMPUTE: "_observe_recompute",
}

if set(_HANDLER_NAMES) != set(KvCacheAction):
    missing = sorted(action.value for action in set(KvCacheAction) - set(_HANDLER_NAMES))
    raise RuntimeError(f"KV vocabulary members without an accounting rule: {missing}")


__all__ = [
    "BYTE_CARRYING_ACTIONS",
    "KV_ACCOUNTING_SCHEMA",
    "KvAccountingReport",
    "KvBlockState",
    "KvLifecycleLedger",
    "KvPoolAccounting",
    "KvPoolSpec",
    "KvServiceDemand",
]
