"""Framework-free continuous batching for deployment estimation.

The surrogate is a scheduling authority, not a duration model. It reproduces
the pinned single-engine vLLM v1 decision surface and emits the repository's
existing :class:`~simllm.core.StepRecord` and
:class:`~simllm.core.KvCacheWork` contracts. A caller-supplied step sink prices
each decision, and the loop advances its one virtual clock to that
``StepResult`` exactly as the live adapter does.

Source references in this module are relative to the pinned vLLM 0.27.1
package. They are the implementation evidence frozen by the P3 source audit.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import pickle
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TypeAlias

from simllm.core import (
    BookkeepingLedger,
    BookkeepingScope,
    CreatedObjectKind,
    CreatedObjectRecord,
    CreatedObjectRef,
    ExecutionObservations,
    ExecutionOperation,
    KvCacheAction,
    KvCacheWork,
    KvPoolSpec,
    ObjectOwner,
    OperationCorrelation,
    RequestBookkeeper,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    StepResult,
    VirtualClock,
)
from simllm.deploy.estimator import EstimateStamp, EstimatorClass
from simllm.deploy.frontier import PointClass
from simllm.workload import AdmissionMode, RequestAdmissionGate

SURROGATE_LOOP_SCHEMA = "simllm-surrogate-loop-v1"
SURROGATE_BLOCK_TOKENS = 16

SurrogateKvOperation: TypeAlias = tuple[str, KvCacheWork]
LegacyStepSink: TypeAlias = Callable[[StepRecord], StepResult | None]
ObservationStepSink: TypeAlias = Callable[
    [StepRecord, ExecutionObservations | None], StepResult
]


def _nonblank(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _signed_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _token_ids(name: str, values: object) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    for index, token_id in enumerate(values):
        _integer(f"{name}[{index}]", token_id)
    return values


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


class SurrogateQueuePolicy(str, Enum):
    """Pinned request-queue policies supported by the surrogate."""

    FCFS = "fcfs"
    PRIORITY = "priority"


class SurrogateReserveMode(str, Enum):
    """Whether waiting admission reserves the complete known input extent."""

    NONE = "none"
    FULL_ISL = "full-isl"


@dataclass(frozen=True, slots=True)
class SurrogateStopPolicy:
    """Deterministic sampled-token stream and token-level stop rules.

    ``sampled_token_ids`` supplies engine output in request order. When it is
    shorter than the output cap, ``default_token_id`` supplies the remainder.
    This keeps sampling outside the scheduling model while making token-content
    prefix hashes reproducible when a generated block becomes cacheable.
    """

    sampled_token_ids: tuple[int, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    eos_token_id: int | None = None
    ignore_eos: bool = False
    default_token_id: int = 0

    def __post_init__(self) -> None:
        _token_ids("stop_policy.sampled_token_ids", self.sampled_token_ids)
        stop_ids = _token_ids("stop_policy.stop_token_ids", self.stop_token_ids)
        if len(stop_ids) != len(set(stop_ids)):
            raise ValueError("stop_policy.stop_token_ids must be unique")
        if self.eos_token_id is not None:
            _integer("stop_policy.eos_token_id", self.eos_token_id)
        if not isinstance(self.ignore_eos, bool):
            raise TypeError("stop_policy.ignore_eos must be a boolean")
        _integer("stop_policy.default_token_id", self.default_token_id)

    def token_at(self, output_index: int) -> int:
        _integer("output_index", output_index)
        if output_index < len(self.sampled_token_ids):
            return self.sampled_token_ids[output_index]
        return self.default_token_id

    def stops_on(self, token_id: int) -> bool:
        if token_id in self.stop_token_ids:
            return True
        return (
            not self.ignore_eos
            and self.eos_token_id is not None
            and token_id == self.eos_token_id
        )


@dataclass(frozen=True, slots=True)
class SurrogateRequest:
    """One stable ordered workload row consumed by the admission gate."""

    request_id: str
    arrived_at_ps: int
    prompt_token_ids: tuple[int, ...]
    max_output_tokens: int
    priority: int = 0
    stop_policy: SurrogateStopPolicy = field(default_factory=SurrogateStopPolicy)

    def __post_init__(self) -> None:
        _nonblank("request.request_id", self.request_id)
        _integer("request.arrived_at_ps", self.arrived_at_ps)
        prompt = _token_ids("request.prompt_token_ids", self.prompt_token_ids)
        if not prompt:
            raise ValueError("request.prompt_token_ids must not be empty")
        _integer("request.max_output_tokens", self.max_output_tokens, minimum=1)
        _signed_integer("request.priority", self.priority)
        if not isinstance(self.stop_policy, SurrogateStopPolicy):
            raise TypeError("request.stop_policy must be a SurrogateStopPolicy")


@dataclass(frozen=True, slots=True)
class SurrogateLoopConfig:
    """The frozen causal tuple for one single-engine surrogate."""

    resolved_max_num_scheduled_tokens: int
    max_num_seqs: int
    enable_chunked_prefill: bool
    long_prefill_token_threshold: int
    max_model_len: int
    queue_policy: SurrogateQueuePolicy
    scheduler_block_size: int
    num_kv_blocks: int
    reserve_mode: SurrogateReserveMode
    watermark: float
    enable_prefix_caching: bool = True

    def __post_init__(self) -> None:
        _integer(
            "config.resolved_max_num_scheduled_tokens",
            self.resolved_max_num_scheduled_tokens,
            minimum=1,
        )
        _integer("config.max_num_seqs", self.max_num_seqs, minimum=1)
        if not isinstance(self.enable_chunked_prefill, bool):
            raise TypeError("config.enable_chunked_prefill must be a boolean")
        _integer(
            "config.long_prefill_token_threshold",
            self.long_prefill_token_threshold,
        )
        _integer("config.max_model_len", self.max_model_len, minimum=1)
        if not isinstance(self.queue_policy, SurrogateQueuePolicy):
            raise TypeError("config.queue_policy must be a SurrogateQueuePolicy")
        block_size = _integer(
            "config.scheduler_block_size",
            self.scheduler_block_size,
            minimum=1,
        )
        if block_size != SURROGATE_BLOCK_TOKENS:
            raise ValueError(
                f"config.scheduler_block_size must be {SURROGATE_BLOCK_TOKENS} "
                "for the registered surrogate model"
            )
        _integer("config.num_kv_blocks", self.num_kv_blocks, minimum=2)
        if not isinstance(self.reserve_mode, SurrogateReserveMode):
            raise TypeError("config.reserve_mode must be a SurrogateReserveMode")
        if isinstance(self.watermark, bool) or type(self.watermark) not in (
            int,
            float,
        ):
            raise TypeError("config.watermark must be a finite number")
        if not math.isfinite(float(self.watermark)) or self.watermark < 0:
            raise ValueError("config.watermark must be finite and nonnegative")
        if not isinstance(self.enable_prefix_caching, bool):
            raise TypeError("config.enable_prefix_caching must be a boolean")

    @property
    def watermark_blocks(self) -> int:
        """Return vLLM's integer watermark reservation.

        Evidence: ``v1/core/kv_cache_manager.py:165-168``.
        """

        return int(self.watermark * self.num_kv_blocks)

    @property
    def causal_tuple(self) -> tuple[tuple[str, str | int | float | bool], ...]:
        """Return the ordered, record-ready causal tuple."""

        return (
            (
                "resolved_max_num_scheduled_tokens",
                self.resolved_max_num_scheduled_tokens,
            ),
            ("max_num_seqs", self.max_num_seqs),
            ("enable_chunked_prefill", self.enable_chunked_prefill),
            ("enable_prefix_caching", self.enable_prefix_caching),
            ("long_prefill_token_threshold", self.long_prefill_token_threshold),
            ("max_model_len", self.max_model_len),
            ("queue_policy", self.queue_policy.value),
            ("scheduler_block_size", self.scheduler_block_size),
            ("num_kv_blocks", self.num_kv_blocks),
            ("reserve_mode", self.reserve_mode.value),
            ("watermark", self.watermark),
        )


def surrogate_loop_stamp(pricing_stamp: EstimateStamp) -> EstimateStamp:
    """Register an existing pricing stamp under the loop estimator model class."""

    if not isinstance(pricing_stamp, EstimateStamp):
        raise TypeError("pricing_stamp must be an EstimateStamp")
    pricing_stamp.__post_init__()
    return replace(pricing_stamp, estimator_class=EstimatorClass.ESTIMATE_LOOP)


@dataclass(frozen=True, slots=True)
class SurrogateRequestResult:
    """One request's scheduler-visible outcome."""

    request_id: str
    arrived_at_ps: int
    first_released_at_ps: int
    completed_at_ps: int
    output_token_ids: tuple[int, ...]
    num_preemptions: int


@dataclass(frozen=True, slots=True)
class SurrogateStepEmission:
    """One complete decision record, KV projection, and priced result."""

    record: StepRecord
    kv_operations: tuple[SurrogateKvOperation, ...]
    result: StepResult
    stamp: EstimateStamp
    point_class: PointClass = PointClass.ESTIMATE_LOOP

    def __post_init__(self) -> None:
        if not isinstance(self.record, StepRecord):
            raise TypeError("emission.record must be a StepRecord")
        if not isinstance(self.result, StepResult):
            raise TypeError("emission.result must be a StepResult")
        if self.result.step_index != self.record.step_index:
            raise ValueError("emission result belongs to another step")
        if not isinstance(self.stamp, EstimateStamp):
            raise TypeError("emission.stamp must be an EstimateStamp")
        self.stamp.__post_init__()
        if self.stamp.estimator_class is not EstimatorClass.ESTIMATE_LOOP:
            raise ValueError("emission stamp must select ESTIMATE-LOOP")
        if self.point_class is not PointClass.ESTIMATE_LOOP:
            raise ValueError("emission point class must be ESTIMATE-LOOP")
        operation_ids = tuple(operation_id for operation_id, _ in self.kv_operations)
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("emission KV operation IDs must be unique")

    def kv_observations(
        self,
        *,
        rank: int = 0,
        logical_queue: str = "surrogate:kv",
    ) -> ExecutionObservations:
        """Return the native KV stream in the lowerer's observation contract."""

        _integer("rank", rank)
        _nonblank("logical_queue", logical_queue)
        operations: list[ExecutionOperation] = []
        predecessor: str | None = None
        request_tails: dict[str, str] = {}
        for operation_id, work in self.kv_operations:
            request_ids = () if work.request_id is None else (work.request_id,)
            operations.append(
                ExecutionOperation(
                    operation_id=operation_id,
                    rank=rank,
                    logical_queue=logical_queue,
                    work=work,
                    depends_on=() if predecessor is None else (predecessor,),
                    correlation=OperationCorrelation(request_ids=request_ids),
                    placement_epoch=work.placement_epoch,
                )
            )
            predecessor = operation_id
            if work.request_id is not None:
                request_tails[work.request_id] = operation_id
        scheduled_ids = tuple(row.request_id for row in self.record.scheduled)
        missing = tuple(
            request_id for request_id in scheduled_ids if request_id not in request_tails
        )
        if missing:
            raise ValueError(
                f"scheduled requests have no KV completion endpoint: {missing}"
            )
        completion_ids = tuple(request_tails[request_id] for request_id in scheduled_ids)
        return ExecutionObservations(
            operations=tuple(operations),
            completion_operation_ids=(
                completion_ids
                if completion_ids
                else (() if predecessor is None else (predecessor,))
            ),
        )


@dataclass(frozen=True, slots=True)
class SurrogateLoopResult:
    """One drained surrogate execution and its immutable projections."""

    schema: str
    causal_tuple: tuple[tuple[str, str | int | float | bool], ...]
    stamp: EstimateStamp
    point_class: PointClass
    emissions: tuple[SurrogateStepEmission, ...]
    admission_order: tuple[str, ...]
    request_results: tuple[SurrogateRequestResult, ...]
    final_virtual_time_ps: int
    bookkeeping: BookkeepingLedger

    @property
    def records(self) -> tuple[StepRecord, ...]:
        return tuple(emission.record for emission in self.emissions)

    @property
    def results(self) -> tuple[StepResult, ...]:
        return tuple(emission.result for emission in self.emissions)

    @property
    def kv_operations(self) -> tuple[SurrogateKvOperation, ...]:
        return tuple(
            operation
            for emission in self.emissions
            for operation in emission.kv_operations
        )


class _RequestStatus(Enum):
    PENDING = "pending"
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"


@dataclass(eq=False, slots=True)
class _RequestState:
    spec: SurrogateRequest
    sequence: int
    status: _RequestStatus = _RequestStatus.PENDING
    computed_tokens: int = 0
    output_token_ids: list[int] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    cached_tokens: int = 0
    cached_reported: bool = False
    num_preemptions: int = 0
    first_released_at_ps: int | None = None
    completed_at_ps: int | None = None

    @property
    def request_id(self) -> str:
        return self.spec.request_id

    @property
    def arrived_at_ps(self) -> int:
        return self.spec.arrived_at_ps

    @property
    def priority(self) -> int:
        return self.spec.priority

    @property
    def prompt_length(self) -> int:
        return len(self.spec.prompt_token_ids)

    @property
    def all_token_ids(self) -> tuple[int, ...]:
        return self.spec.prompt_token_ids + tuple(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        return self.prompt_length + len(self.output_token_ids)

    @property
    def priority_key(self) -> tuple[int, int, str]:
        # Evidence: ``v1/request.py:329-340``. Stable IDs replace the forbidden
        # object-identity fallback after priority and arrival.
        return (self.priority, self.arrived_at_ps, self.request_id)


class _WaitingQueue:
    """FCFS deque or stable priority heap with one common narrow surface."""

    def __init__(self, policy: SurrogateQueuePolicy) -> None:
        self.policy = policy
        self._fcfs: deque[_RequestState] = deque()
        self._priority: list[tuple[tuple[int, int, str], _RequestState]] = []

    def __bool__(self) -> bool:
        return bool(self._fcfs if self.policy is SurrogateQueuePolicy.FCFS else self._priority)

    def __len__(self) -> int:
        return len(self._fcfs if self.policy is SurrogateQueuePolicy.FCFS else self._priority)

    def add(self, request: _RequestState) -> None:
        # Evidence: ``v1/core/sched/request_queue.py:75-95,131-165``.
        if self.policy is SurrogateQueuePolicy.FCFS:
            self._fcfs.append(request)
        else:
            heapq.heappush(self._priority, (request.priority_key, request))

    def prepend(self, request: _RequestState) -> None:
        # Recompute preemption prepends only for FCFS. A priority queue has no
        # front and reuses its native ordering.
        if self.policy is SurrogateQueuePolicy.FCFS:
            self._fcfs.appendleft(request)
        else:
            heapq.heappush(self._priority, (request.priority_key, request))

    def peek(self) -> _RequestState:
        if self.policy is SurrogateQueuePolicy.FCFS:
            return self._fcfs[0]
        return self._priority[0][1]

    def pop(self) -> _RequestState:
        if self.policy is SurrogateQueuePolicy.FCFS:
            return self._fcfs.popleft()
        return heapq.heappop(self._priority)[1]

    @property
    def request_ids(self) -> tuple[str, ...]:
        if self.policy is SurrogateQueuePolicy.FCFS:
            return tuple(request.request_id for request in self._fcfs)
        return tuple(request.request_id for _, request in sorted(self._priority))


@dataclass(slots=True)
class _Block:
    block_id: int
    owners: set[str] = field(default_factory=set)
    block_hash: bytes | None = None

    @property
    def ref_count(self) -> int:
        return len(self.owners)


class _KvBlockPool:
    """One uniform vLLM-shaped block pool and its native KV projection."""

    _NONE_HASH = hashlib.sha256(b"simllm-surrogate-prefix-root-v1").digest()

    def __init__(
        self,
        config: SurrogateLoopConfig,
        pool_spec: KvPoolSpec,
        *,
        dtype: str,
        placement_epoch: int,
    ) -> None:
        if not isinstance(pool_spec, KvPoolSpec):
            raise TypeError("kv_pool must be a KvPoolSpec")
        if pool_spec.block_tokens != config.scheduler_block_size:
            raise ValueError("KV pool block tokens disagree with scheduler block size")
        if pool_spec.capacity_blocks != config.num_kv_blocks:
            raise ValueError("KV pool capacity disagrees with causal block count")
        self.config = config
        self.spec = pool_spec
        self.dtype = _nonblank("kv_dtype", dtype)
        self.placement_epoch = _integer("placement_epoch", placement_epoch)
        self.blocks = {index: _Block(index) for index in range(config.num_kv_blocks)}
        # vLLM removes block zero as the null block at construction
        # (``v1/core/block_pool.py:175-191``).
        self.free_queue: deque[int] = deque(range(1, config.num_kv_blocks))
        self.prefix_blocks: dict[bytes, list[int]] = {}

    @property
    def free_block_ids(self) -> tuple[int, ...]:
        return tuple(self.free_queue)

    def _hashes(self, request: _RequestState) -> tuple[bytes, ...]:
        """Hash chained full extents using vLLM's pickle plus SHA-256 shape.

        Evidence: ``utils/hashing.py:26-40`` and
        ``v1/core/kv_cache_utils.py:691-746``. The registered model pins one
        process-local root. Prefix salt and cross-interpreter byte identity are
        deliberately separate residual modes; equality decisions within this
        model are content-exact.
        """

        tokens = request.all_token_ids
        block_size = self.config.scheduler_block_size
        parent = self._NONE_HASH
        result: list[bytes] = []
        for start in range(0, len(tokens) - block_size + 1, block_size):
            extent = tuple(tokens[start : start + block_size])
            payload = (parent, extent, None)
            parent = hashlib.sha256(
                pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            ).digest()
            result.append(parent)
        return tuple(result)

    def prefix_hit(self, request: _RequestState) -> tuple[list[int], int]:
        """Return the longest chained full-block hit below the logits token.

        Evidence: ``v1/core/kv_cache_manager.py:207-241`` and
        ``v1/core/single_type_kv_cache_manager.py:703-751``.
        """

        if not self.config.enable_prefix_caching:
            return [], 0
        max_hit_length = request.num_tokens - 1
        max_blocks = max_hit_length // self.config.scheduler_block_size
        hits: list[int] = []
        for block_hash in self._hashes(request)[:max_blocks]:
            candidates = self.prefix_blocks.get(block_hash, ())
            block_id = next(
                (
                    candidate
                    for candidate in candidates
                    if self.blocks[candidate].block_hash == block_hash
                ),
                None,
            )
            if block_id is None:
                break
            hits.append(block_id)
        return hits, len(hits) * self.config.scheduler_block_size

    def _new_block_count(self, target_tokens: int, held_blocks: int) -> int:
        required = _ceil_div(target_tokens, self.config.scheduler_block_size)
        return max(required - held_blocks, 0)

    def can_allocate(
        self,
        request: _RequestState,
        *,
        target_tokens: int,
        prefix_blocks: Sequence[int],
        full_sequence_must_fit: bool,
        has_scheduled_reqs: bool,
    ) -> bool:
        """Apply the full-ISL and ordinary watermark capacity checks.

        Evidence: ``v1/core/kv_cache_manager.py:402-427,449-466``.
        Reclaimable prefix blocks leave the free queue when touched, so they
        count beside newly allocated blocks in both checks.
        """

        held = len(request.block_ids) + len(prefix_blocks)
        reclaimable_hits = sum(
            self.blocks[block_id].ref_count == 0 for block_id in prefix_blocks
        )
        watermark = (
            self.config.watermark_blocks
            if has_scheduled_reqs
            and request.status in (_RequestStatus.WAITING, _RequestStatus.PREEMPTED)
            else 0
        )
        if full_sequence_must_fit:
            full_tokens = min(request.num_tokens, self.config.max_model_len)
            full_required = (
                self._new_block_count(full_tokens, held) + reclaimable_hits + watermark
            )
            if full_required > len(self.free_queue):
                return False
        required = (
            self._new_block_count(target_tokens, held)
            + reclaimable_hits
            + watermark
        )
        return required <= len(self.free_queue)

    def allocate(
        self,
        request: _RequestState,
        *,
        target_tokens: int,
        prefix_blocks: Sequence[int],
        prefix_tokens: int,
        append: Callable[..., None],
    ) -> None:
        """Commit a capacity-approved prefix binding and block allocation."""

        if request.block_ids and prefix_blocks:
            raise RuntimeError("a running request cannot acquire a second prefix")
        if prefix_blocks:
            block_ids = tuple(str(block_id) for block_id in prefix_blocks)
            append(
                "bind-prefix",
                KvCacheAction.BIND_PREFIX,
                request_id=request.request_id,
                block_ids=block_ids,
                token_start=0,
                token_end=prefix_tokens,
            )

        new_count = self._new_block_count(
            target_tokens,
            len(request.block_ids) + len(prefix_blocks),
        )
        capacity_start = (
            len(request.block_ids) + len(prefix_blocks)
        ) * self.config.scheduler_block_size
        capacity_end = capacity_start + new_count * self.config.scheduler_block_size
        if new_count:
            append(
                "reserve",
                KvCacheAction.RESERVE,
                request_id=request.request_id,
                token_start=capacity_start,
                token_end=capacity_end,
            )

        touched: list[str] = []
        retained: list[str] = []
        for block_id in prefix_blocks:
            block = self.blocks[block_id]
            if block.ref_count == 0:
                self.free_queue.remove(block_id)
                touched.append(str(block_id))
            else:
                retained.append(str(block_id))
            block.owners.add(request.request_id)
        if touched:
            append(
                "touch",
                KvCacheAction.TOUCH,
                request_id=request.request_id,
                block_ids=tuple(touched),
                token_start=0,
                token_end=len(touched) * self.config.scheduler_block_size,
            )
        if retained:
            append(
                "retain",
                KvCacheAction.RETAIN,
                request_id=request.request_id,
                block_ids=tuple(retained),
                token_start=0,
                token_end=len(retained) * self.config.scheduler_block_size,
            )
        request.block_ids.extend(prefix_blocks)

        new_blocks: list[int] = []
        for _ in range(new_count):
            block_id = self.free_queue.popleft()
            block = self.blocks[block_id]
            if block.block_hash is not None:
                append(
                    "evict",
                    KvCacheAction.EVICT,
                    block_ids=(str(block_id),),
                    token_start=0,
                    token_end=self.config.scheduler_block_size,
                    cause="prefix-cache-capacity",
                )
                candidates = self.prefix_blocks.get(block.block_hash)
                if candidates is not None:
                    candidates.remove(block_id)
                    if not candidates:
                        del self.prefix_blocks[block.block_hash]
                block.block_hash = None
            if block.ref_count:
                raise RuntimeError("free queue contains a referenced KV block")
            block.owners.add(request.request_id)
            new_blocks.append(block_id)
        request.block_ids.extend(new_blocks)
        if new_blocks:
            append(
                "allocate",
                KvCacheAction.ALLOCATE,
                request_id=request.request_id,
                block_ids=tuple(str(block_id) for block_id in new_blocks),
                token_start=capacity_start,
                token_end=capacity_end,
            )

        if self.config.enable_prefix_caching:
            # vLLM installs hashes for all complete extents covered by this
            # schedule in ``allocate_slots``
            # (``kv_cache_manager.py:493-502`` and
            # ``single_type_kv_cache_manager.py:421-453``).
            hashes = self._hashes(request)
            full_blocks = target_tokens // self.config.scheduler_block_size
            for index in range(full_blocks):
                block = self.blocks[request.block_ids[index]]
                block_hash = hashes[index]
                if block.block_hash is None:
                    block.block_hash = block_hash
                    self.prefix_blocks.setdefault(block_hash, []).append(
                        block.block_id
                    )
                elif block.block_hash != block_hash:
                    raise RuntimeError(
                        "request block hash changed while still referenced"
                    )
        write_start = prefix_tokens if prefix_blocks else request.computed_tokens
        if target_tokens > write_start:
            # This zero-byte marker makes full extents reusable by the shared KV
            # ledger without stealing duration authority from the pricing sink.
            append(
                "write",
                KvCacheAction.WRITE,
                request_id=request.request_id,
                block_ids=tuple(str(block_id) for block_id in request.block_ids),
                token_start=write_start,
                token_end=target_tokens,
            )

    def release(
        self,
        request: _RequestState,
        *,
        append: Callable[..., None],
        cause: str,
    ) -> None:
        """Release in reverse order and preserve the LRU reclaim queue.

        Evidence: ``v1/core/single_type_kv_cache_manager.py:476-503`` and
        ``v1/core/block_pool.py:719-740``.
        """

        if not request.block_ids:
            return
        ordered = list(reversed(request.block_ids))
        block_ids = tuple(str(block_id) for block_id in ordered)
        capacity_tokens = len(ordered) * self.config.scheduler_block_size
        append(
            "release",
            KvCacheAction.RELEASE,
            request_id=request.request_id,
            block_ids=block_ids,
            token_start=0,
            token_end=capacity_tokens,
            cause=cause,
        )
        prepend_blocks: list[int] = []
        append_blocks: list[int] = []
        freed_blocks: list[int] = []
        for block_id in ordered:
            block = self.blocks[block_id]
            if request.request_id not in block.owners:
                raise RuntimeError("request released a KV block it does not own")
            block.owners.remove(request.request_id)
            if block.ref_count == 0:
                if block.block_hash is None:
                    freed_blocks.append(block_id)
                if block.block_hash is None and self.config.enable_prefix_caching:
                    prepend_blocks.append(block_id)
                else:
                    append_blocks.append(block_id)
        # vLLM 0.27.1 appends every block when caching is disabled so the next
        # allocation keeps locality. With caching enabled, never-cacheable
        # hashless blocks retain the prior prepend behavior and cached blocks
        # remain at the least-recently-used tail.
        self.free_queue = deque((*prepend_blocks, *self.free_queue))
        self.free_queue.extend(append_blocks)
        if freed_blocks:
            append(
                "free",
                KvCacheAction.FREE,
                request_id=request.request_id,
                block_ids=tuple(str(block_id) for block_id in freed_blocks),
                token_start=0,
                token_end=len(freed_blocks) * self.config.scheduler_block_size,
                cause=cause,
            )
        request.block_ids.clear()


@dataclass(slots=True)
class _ScheduledDecision:
    request: _RequestState
    computed_before: int
    num_new_tokens: int


class SurrogateServingLoop:
    """One framework-free, continuously batched estimator engine."""

    def __init__(
        self,
        config: SurrogateLoopConfig,
        workload: Sequence[SurrogateRequest],
        kv_pool: KvPoolSpec,
        stamp: EstimateStamp,
        *,
        kv_dtype: str = "bfloat16",
        placement_epoch: int = 0,
        clock: VirtualClock | None = None,
    ) -> None:
        if not isinstance(config, SurrogateLoopConfig):
            raise TypeError("config must be a SurrogateLoopConfig")
        if not isinstance(stamp, EstimateStamp):
            raise TypeError("stamp must be an EstimateStamp")
        stamp.__post_init__()
        if stamp.estimator_class is not EstimatorClass.ESTIMATE_LOOP:
            raise ValueError("surrogate loop requires an ESTIMATE-LOOP stamp")
        if not isinstance(workload, Sequence) or isinstance(workload, (str, bytes)):
            raise TypeError("workload must be a sequence of SurrogateRequest rows")
        requests = tuple(workload)
        if not requests:
            raise ValueError("workload must not be empty")
        if any(not isinstance(request, SurrogateRequest) for request in requests):
            raise TypeError("workload must contain SurrogateRequest rows")
        request_ids = tuple(request.request_id for request in requests)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("workload request IDs must be unique")
        if any(len(request.prompt_token_ids) > config.max_model_len for request in requests):
            raise ValueError("workload prompt exceeds config.max_model_len")
        if clock is not None and not isinstance(clock, VirtualClock):
            raise TypeError("clock must be a VirtualClock or None")

        self.config = config
        self.workload = requests
        self.stamp = stamp
        self.clock = clock or VirtualClock()
        self.bookkeeper = RequestBookkeeper()
        self._states = {
            request.request_id: _RequestState(request, sequence)
            for sequence, request in enumerate(requests)
        }
        self.bookkeeper.extend(
            CreatedObjectRecord(
                ref=CreatedObjectRef(
                    CreatedObjectKind.FRAMEWORK_REQUEST,
                    f"surrogate-request:{request.request_id}",
                ),
                owner=ObjectOwner.FRAMEWORK,
                created_at_ps=request.arrived_at_ps,
                scope=BookkeepingScope(
                    correlation=OperationCorrelation(
                        request_ids=(request.request_id,)
                    )
                ),
                native_id=request.request_id,
            )
            for request in requests
        )
        self.admission_gate = RequestAdmissionGate(
            self.clock,
            self.bookkeeper,
            mode=AdmissionMode.ARRIVAL_GATED,
        )
        self.waiting = _WaitingQueue(config.queue_policy)
        self.running: list[_RequestState] = []
        self._kv = _KvBlockPool(
            config,
            kv_pool,
            dtype=kv_dtype,
            placement_epoch=placement_epoch,
        )
        self._emissions: list[SurrogateStepEmission] = []
        self._pending_finished: list[str] = []
        self._finished_order: list[str] = []
        self._step_index = 0
        self._bound_sink_id: int | None = None

    @property
    def point_class(self) -> PointClass:
        return PointClass.ESTIMATE_LOOP

    @property
    def emissions(self) -> tuple[SurrogateStepEmission, ...]:
        return tuple(self._emissions)

    @property
    def waiting_request_ids(self) -> tuple[str, ...]:
        return self.waiting.request_ids

    @property
    def running_request_ids(self) -> tuple[str, ...]:
        return tuple(request.request_id for request in self.running)

    @property
    def free_block_ids(self) -> tuple[int, ...]:
        return self._kv.free_block_ids

    @property
    def has_work(self) -> bool:
        return bool(
            self.admission_gate.has_pending
            or self.waiting
            or self.running
            or self._pending_finished
        )

    def _admit(self, arrival: object) -> None:
        request_id = getattr(arrival, "request_id", None)
        request = self._states.get(request_id)
        if request is None:
            raise ValueError(f"admission gate released unknown request {request_id!r}")
        if request.status is not _RequestStatus.PENDING:
            raise RuntimeError("admission gate released a request twice")
        request.status = _RequestStatus.WAITING
        self.waiting.add(request)

    def _admit_or_advance(self) -> None:
        # Evidence: ``simllm/workload/admission.py:21-93`` and the one-step
        # harness loop in ``examples/arrival_admission_v1/run_study.py:429-439``.
        while not (self.waiting or self.running or self._pending_finished):
            self.admission_gate.admit_ready(self._admit)
            if self.waiting:
                return
            if not self.admission_gate.has_pending:
                return
            self.admission_gate.advance_to_next_arrival()
        self.admission_gate.admit_ready(self._admit)

    def _append_kv_factory(
        self,
        operations: list[SurrogateKvOperation],
    ) -> Callable[..., None]:
        def append(
            suffix: str,
            action: KvCacheAction,
            *,
            request_id: str | None = None,
            block_ids: tuple[str, ...] = (),
            token_start: int | None = None,
            token_end: int | None = None,
            cause: str | None = None,
        ) -> None:
            operation_id = (
                f"surrogate-kv-{self._step_index:06d}-{len(operations):04d}-{suffix}"
            )
            operations.append(
                (
                    operation_id,
                    KvCacheWork(
                        action=action,
                        pool_id=self._kv.spec.pool_id,
                        request_id=request_id,
                        block_ids=block_ids,
                        token_start=token_start,
                        token_end=token_end,
                        dtype=self._kv.dtype,
                        placement_epoch=self._kv.placement_epoch,
                        cause=cause,
                    ),
                )
            )

        return append

    def _running_new_tokens(self, request: _RequestState, token_budget: int) -> int:
        # Evidence: ``v1/core/sched/scheduler.py:502-518``. The long-prefill
        # threshold is applied before the budget and model-length caps.
        num_new_tokens = request.num_tokens - request.computed_tokens
        threshold = self.config.long_prefill_token_threshold
        if 0 < threshold < num_new_tokens:
            num_new_tokens = threshold
        num_new_tokens = min(num_new_tokens, token_budget)
        num_new_tokens = min(
            num_new_tokens,
            self.config.max_model_len - request.computed_tokens - 1,
        )
        return max(num_new_tokens, 0)

    def _waiting_new_tokens(
        self,
        request: _RequestState,
        computed_tokens: int,
        token_budget: int,
    ) -> int | None:
        # Evidence: ``v1/core/sched/scheduler.py:837-876``. Cached tokens are
        # subtracted first, the long cap is next, and the chunking-off stop is
        # tested before taking the minimum with the remaining budget.
        num_new_tokens = request.num_tokens - computed_tokens
        threshold = self.config.long_prefill_token_threshold
        if 0 < threshold < num_new_tokens:
            num_new_tokens = threshold
        if not self.config.enable_chunked_prefill and num_new_tokens > token_budget:
            return None
        return min(num_new_tokens, token_budget)

    def _preemption_victim(self) -> _RequestState:
        # Evidence: ``v1/core/sched/scheduler.py:574-603``.
        if self.config.queue_policy is SurrogateQueuePolicy.FCFS:
            return self.running[-1]
        # The native max key omits request ID. Python's stable ``max`` therefore
        # keeps the first running request when priority and arrival tie.
        return max(
            self.running,
            key=lambda request: (request.priority, request.arrived_at_ps),
        )

    def _preempt(
        self,
        request: _RequestState,
        append: Callable[..., None],
    ) -> None:
        computed = request.computed_tokens
        self._kv.release(request, append=append, cause="scheduler-recompute")
        if computed:
            append(
                "recompute",
                KvCacheAction.RECOMPUTE,
                request_id=request.request_id,
                token_start=0,
                token_end=computed,
                cause="scheduler-recompute",
            )
        # Evidence: ``v1/core/sched/scheduler.py:1212-1233``.
        request.computed_tokens = 0
        request.cached_tokens = 0
        request.num_preemptions += 1
        request.status = _RequestStatus.PREEMPTED
        self.waiting.prepend(request)

    def _schedule_running(
        self,
        token_budget: int,
        append: Callable[..., None],
    ) -> tuple[list[_ScheduledDecision], list[str], int]:
        decisions: list[_ScheduledDecision] = []
        preempted: list[str] = []
        request_index = 0
        # Evidence: running requests consume the budget first in
        # ``v1/core/sched/scheduler.py:469-620``.
        while request_index < len(self.running) and token_budget > 0:
            request = self.running[request_index]
            num_new_tokens = self._running_new_tokens(request, token_budget)
            if num_new_tokens == 0:
                request_index += 1
                continue
            target_tokens = request.computed_tokens + num_new_tokens

            while not self._kv.can_allocate(
                request,
                target_tokens=target_tokens,
                prefix_blocks=(),
                full_sequence_must_fit=False,
                has_scheduled_reqs=True,
            ):
                victim = self._preemption_victim()
                self.running.remove(victim)
                prior = next(
                    (decision for decision in decisions if decision.request is victim),
                    None,
                )
                if prior is not None:
                    decisions.remove(prior)
                    token_budget += prior.num_new_tokens
                    request_index -= 1
                self._preempt(victim, append)
                preempted.append(victim.request_id)
                if victim is request:
                    break
            else:
                self._kv.allocate(
                    request,
                    target_tokens=target_tokens,
                    prefix_blocks=(),
                    prefix_tokens=0,
                    append=append,
                )
                decisions.append(
                    _ScheduledDecision(request, request.computed_tokens, num_new_tokens)
                )
                token_budget -= num_new_tokens
                request_index += 1
                continue
            break
        return decisions, preempted, token_budget

    def _schedule_waiting(
        self,
        decisions: list[_ScheduledDecision],
        token_budget: int,
        append: Callable[..., None],
    ) -> int:
        # No waiting request is considered after a same-step preemption. This is
        # the guard at ``v1/core/sched/scheduler.py:665-669``.
        while self.waiting and token_budget > 0:
            # Evidence: ``v1/core/sched/scheduler.py:669-674``.
            if len(self.running) >= self.config.max_num_seqs:
                break
            request = self.waiting.peek()
            prefix_blocks, prefix_tokens = self._kv.prefix_hit(request)
            num_new_tokens = self._waiting_new_tokens(
                request,
                prefix_tokens,
                token_budget,
            )
            if num_new_tokens is None:
                break
            if num_new_tokens <= 0:
                raise RuntimeError("waiting request has no schedulable tokens")
            target_tokens = prefix_tokens + num_new_tokens
            if not self._kv.can_allocate(
                request,
                target_tokens=target_tokens,
                prefix_blocks=prefix_blocks,
                full_sequence_must_fit=(
                    self.config.reserve_mode is SurrogateReserveMode.FULL_ISL
                ),
                has_scheduled_reqs=bool(self.running),
            ):
                break
            popped = self.waiting.pop()
            if popped is not request:
                raise RuntimeError("waiting queue order changed during admission")
            self._kv.allocate(
                request,
                target_tokens=target_tokens,
                prefix_blocks=prefix_blocks,
                prefix_tokens=prefix_tokens,
                append=append,
            )
            request.computed_tokens = prefix_tokens
            request.cached_tokens = prefix_tokens
            request.status = _RequestStatus.RUNNING
            self.running.append(request)
            decisions.append(
                _ScheduledDecision(request, prefix_tokens, num_new_tokens)
            )
            token_budget -= num_new_tokens
        return token_budget

    def _schedule(
        self,
        operations: list[SurrogateKvOperation],
    ) -> tuple[list[_ScheduledDecision], list[str]]:
        # Evidence: every call resets the resolved budget at
        # ``v1/core/sched/scheduler.py:425-445`` and subtracts running then
        # waiting work at ``:613-620`` and ``:1025-1030``.
        append = self._append_kv_factory(operations)
        decisions, preempted, token_budget = self._schedule_running(
            self.config.resolved_max_num_scheduled_tokens,
            append,
        )
        if not preempted:
            self._schedule_waiting(decisions, token_budget, append)
        return decisions, preempted

    def _record(
        self,
        decisions: Sequence[_ScheduledDecision],
        preempted: Sequence[str],
    ) -> StepRecord:
        scheduled: list[ScheduledRequest] = []
        sampled_request_ids: list[str] = []
        for decision in decisions:
            request = decision.request
            context_length = decision.computed_before + decision.num_new_tokens
            phase = (
                RequestPhase.PREFILL
                if decision.computed_before < request.prompt_length
                else RequestPhase.DECODE
            )
            cached = 0 if request.cached_reported else request.cached_tokens
            request.cached_reported = True
            scheduled.append(
                ScheduledRequest(
                    request_id=request.request_id,
                    phase=phase,
                    num_new_tokens=decision.num_new_tokens,
                    num_cached_tokens=cached,
                    context_length=context_length,
                )
            )
            if context_length >= request.prompt_length:
                sampled_request_ids.append(request.request_id)
            if request.first_released_at_ps is None:
                request.first_released_at_ps = self.clock.now_ps
        # Evidence: phase and context use pre-step computed counts and cached
        # tokens report once (SimLLM's pinned translator, ``executor.py:589-650``).
        return StepRecord(
            step_index=self._step_index,
            virtual_time_ps=self.clock.now_ps,
            scheduled=scheduled,
            preempted_request_ids=sorted(preempted),
            finished_request_ids=sorted(self._pending_finished),
            num_sampled=len(sampled_request_ids),
            num_tokens_after_padding=None,
            sampled_request_ids=sampled_request_ids,
        )

    def _bind_sink(self, sink: object) -> None:
        sink_id = id(sink)
        if self._bound_sink_id is not None:
            if self._bound_sink_id != sink_id:
                raise RuntimeError("surrogate loop is already bound to another step sink")
            return
        bind_clock = getattr(sink, "bind_clock", None)
        if callable(bind_clock):
            bind_clock(self.clock)
        self._bound_sink_id = sink_id

    def _price(
        self,
        record: StepRecord,
        *,
        step_sink: LegacyStepSink | None,
        observation_step_sink: ObservationStepSink | None,
    ) -> StepResult:
        if (step_sink is None) == (observation_step_sink is None):
            raise ValueError(
                "supply exactly one of step_sink or observation_step_sink"
            )
        sink: object = step_sink if step_sink is not None else observation_step_sink
        self._bind_sink(sink)
        result = (
            step_sink(record)
            if step_sink is not None
            else observation_step_sink(record, None)  # type: ignore[misc]
        )
        if result is None:
            raise RuntimeError("surrogate step sink returned no priced StepResult")
        if not isinstance(result, StepResult):
            raise TypeError("surrogate step sink must return a StepResult")
        if result.step_index != record.step_index:
            raise ValueError("priced StepResult belongs to another step")
        if result.completed_at_ps != record.virtual_time_ps + result.step_latency_ps:
            raise ValueError("priced StepResult does not conserve release plus latency")
        return result

    def _settle(
        self,
        decisions: Sequence[_ScheduledDecision],
        sampled_request_ids: Sequence[str],
        result: StepResult,
        append: Callable[..., None],
    ) -> None:
        sampled = set(sampled_request_ids)
        finished: list[_RequestState] = []
        for decision in decisions:
            request = decision.request
            request.computed_tokens += decision.num_new_tokens
            if request.request_id not in sampled:
                continue
            output_index = len(request.output_token_ids)
            token_id = request.spec.stop_policy.token_at(output_index)
            request.output_token_ids.append(token_id)
            if (
                len(request.output_token_ids) >= request.spec.max_output_tokens
                or request.spec.stop_policy.stops_on(token_id)
                or request.num_tokens >= self.config.max_model_len
            ):
                finished.append(request)
        for request in finished:
            self.running.remove(request)
            self._kv.release(request, append=append, cause="request-release")
            request.status = _RequestStatus.FINISHED
            request.completed_at_ps = result.completed_at_ps
            self._finished_order.append(request.request_id)
        self._pending_finished = [request.request_id for request in finished]

    def step(
        self,
        *,
        step_sink: LegacyStepSink | None = None,
        observation_step_sink: ObservationStepSink | None = None,
    ) -> SurrogateStepEmission:
        """Admit ready work, decide one engine step, price it, and settle it."""

        if not self.has_work:
            raise StopIteration("surrogate loop is drained")
        self._admit_or_advance()
        prior_finished = tuple(self._pending_finished)
        operations: list[SurrogateKvOperation] = []
        decisions, preempted = self._schedule(operations)
        record = self._record(decisions, preempted)
        self._pending_finished = []
        if not record.scheduled and not (
            record.preempted_request_ids or record.finished_request_ids
        ):
            self._pending_finished = list(prior_finished)
            raise RuntimeError(
                "surrogate scheduler made no progress; the waiting head cannot be admitted"
            )
        result = self._price(
            record,
            step_sink=step_sink,
            observation_step_sink=observation_step_sink,
        )
        # The same arrival harness advances its sole virtual clock to the priced
        # completion boundary before it admits the next ready workload prefix.
        self.clock.advance_to(max(result.completed_at_ps, self.clock.now_ps))
        append = self._append_kv_factory(operations)
        self._settle(
            decisions,
            record.sampled_request_ids or (),
            result,
            append,
        )
        emission = SurrogateStepEmission(
            record=record,
            kv_operations=tuple(operations),
            result=result,
            stamp=self.stamp,
        )
        self._emissions.append(emission)
        self._step_index += 1
        return emission

    def run(
        self,
        *,
        step_sink: LegacyStepSink | None = None,
        observation_step_sink: ObservationStepSink | None = None,
    ) -> SurrogateLoopResult:
        """Drain the workload and return immutable scheduler projections."""

        while self.has_work:
            self.step(
                step_sink=step_sink,
                observation_step_sink=observation_step_sink,
            )
        request_results: list[SurrogateRequestResult] = []
        for request_id in self._finished_order:
            request = self._states[request_id]
            if request.first_released_at_ps is None or request.completed_at_ps is None:
                raise RuntimeError("finished request is missing scheduler timestamps")
            request_results.append(
                SurrogateRequestResult(
                    request_id=request.request_id,
                    arrived_at_ps=request.arrived_at_ps,
                    first_released_at_ps=request.first_released_at_ps,
                    completed_at_ps=request.completed_at_ps,
                    output_token_ids=tuple(request.output_token_ids),
                    num_preemptions=request.num_preemptions,
                )
            )
        return SurrogateLoopResult(
            schema=SURROGATE_LOOP_SCHEMA,
            causal_tuple=self.config.causal_tuple,
            stamp=self.stamp,
            point_class=self.point_class,
            emissions=tuple(self._emissions),
            admission_order=self.admission_gate.admitted_request_ids,
            request_results=tuple(request_results),
            final_virtual_time_ps=self.clock.now_ps,
            bookkeeping=self.bookkeeper.snapshot(),
        )


__all__ = [
    "SURROGATE_BLOCK_TOKENS",
    "SURROGATE_LOOP_SCHEMA",
    "SurrogateKvOperation",
    "SurrogateLoopConfig",
    "SurrogateLoopResult",
    "SurrogateQueuePolicy",
    "SurrogateRequest",
    "SurrogateRequestResult",
    "SurrogateReserveMode",
    "SurrogateServingLoop",
    "SurrogateStepEmission",
    "SurrogateStopPolicy",
    "surrogate_loop_stamp",
]
