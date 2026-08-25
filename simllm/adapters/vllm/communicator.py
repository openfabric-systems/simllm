"""Shape-only vLLM GroupCoordinator mirror with zero-time observations.

The public operation names and signatures mirror vLLM v0.27.1's
``GroupCoordinator``. Construction is deliberately simulation-specific: the
caller supplies resolved ranks and the only :class:`VirtualClock`, so this
module never creates a torch process group or imports vLLM.

Every successful boundary lowers to :class:`CollectiveWork`. Multi-rank calls
also enter the existing COMP-15 ``ncclAllReduce``-shaped skeleton. That lower
stack is structural only. Runtime projection, completion delivery, and any
communication timing model remain outside this slice.

The current nonzero multi-rank payload domain is exactly COMP-15's ring
domain. Payload bytes must divide evenly over world size, channel count, and
warps per channel; the resulting per-lane share must contain an integral,
nonzero number of configured chunks. An unservable payload raises before an
upper event or operation identifier is consumed. A zero-byte call remains a
successful upper observation with ``stack_disposition="zero_payload_bypass"``
and does not enter the positive-payload lower stack.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from simllm.compute.nccl_stack import (
    NcclRoute,
    NcclStack,
    NcclStackConfig,
    NcclStackEvent,
    ncclAllReduce,
    ncclCommInitRank,
)
from simllm.core import CollectiveWork, VirtualClock

if TYPE_CHECKING:
    import torch

GROUP_COORDINATOR_EVENT_SCHEMA = "simllm-vllm-group-coordinator-event-v1"


@dataclass(frozen=True)
class ShapeDType:
    """Import-free dtype descriptor for a shape-only tensor."""

    name: str
    itemsize: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("dtype name must be a nonempty string")
        if isinstance(self.itemsize, bool) or not isinstance(self.itemsize, int):
            raise TypeError("dtype itemsize must be an integer")
        if self.itemsize <= 0:
            raise ValueError("dtype itemsize must be positive")


FLOAT32 = ShapeDType("float32", 4)
INT32 = ShapeDType("int32", 4)


@dataclass(frozen=True)
class ShapeTensor:
    """Small tensor-shaped value used when torch is intentionally absent."""

    shape: tuple[int, ...]
    dtype: Any = FLOAT32
    element_size_bytes: int = 4
    device: Any = "simulated"

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if any(isinstance(extent, bool) or not isinstance(extent, int) for extent in shape):
            raise TypeError("tensor shape extents must be integers")
        if any(extent < 0 for extent in shape):
            raise ValueError("tensor shape extents must be nonnegative")
        if (
            isinstance(self.element_size_bytes, bool)
            or not isinstance(self.element_size_bytes, int)
        ):
            raise TypeError("tensor element size must be an integer")
        if self.element_size_bytes <= 0:
            raise ValueError("tensor element size must be positive")
        object.__setattr__(self, "shape", shape)

    def numel(self) -> int:
        return math.prod(self.shape)

    def element_size(self) -> int:
        return self.element_size_bytes

    def dim(self) -> int:
        return len(self.shape)

    def size(self, dim: int | None = None) -> tuple[int, ...] | int:
        return self.shape if dim is None else self.shape[dim]

    def new_empty(self, size: Sequence[int]) -> ShapeTensor:
        return ShapeTensor(
            tuple(size),
            dtype=self.dtype,
            element_size_bytes=self.element_size_bytes,
            device=self.device,
        )


@dataclass(frozen=True)
class GroupCoordinatorEvent:
    """One immutable observation at a simulated coordinator boundary."""

    schema: ClassVar[str] = GROUP_COORDINATOR_EVENT_SCHEMA
    sequence: int
    timestamp_ps: int
    operation_id: str
    operation: str
    group: str
    rank: int
    ranks: tuple[int, ...]
    payload_bytes: int
    work: CollectiveWork
    stack_disposition: str
    stack_events: tuple[NcclStackEvent, ...]


class GroupCoordinatorObserver:
    """Shared event order for all simulated groups owned by one runner."""

    def __init__(self, clock: VirtualClock) -> None:
        if not isinstance(clock, VirtualClock):
            raise TypeError("clock must be a simllm.core.VirtualClock")
        self.clock = clock
        self._events: list[GroupCoordinatorEvent] = []

    @property
    def events(self) -> tuple[GroupCoordinatorEvent, ...]:
        return tuple(self._events)

    def record(
        self,
        *,
        timestamp_ps: int,
        operation_id: str,
        operation: str,
        group: str,
        rank: int,
        ranks: tuple[int, ...],
        payload_bytes: int,
        work: CollectiveWork,
        stack_disposition: str,
        stack_events: tuple[NcclStackEvent, ...],
    ) -> GroupCoordinatorEvent:
        event = GroupCoordinatorEvent(
            sequence=len(self._events),
            timestamp_ps=timestamp_ps,
            operation_id=operation_id,
            operation=operation,
            group=group,
            rank=rank,
            ranks=ranks,
            payload_bytes=payload_bytes,
            work=work,
            stack_disposition=stack_disposition,
            stack_events=stack_events,
        )
        self._events.append(event)
        return event


TensorFactory = Callable[[Sequence[int], Any], Any]


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _payload_bytes(tensor: Any) -> int:
    try:
        numel = tensor.numel()
        element_size = tensor.element_size()
    except (AttributeError, TypeError) as exc:
        raise TypeError("tensor must provide numel() and element_size()") from exc
    numel = _positive_int("tensor.numel()", numel)
    element_size = _positive_int("tensor.element_size()", element_size)
    if element_size == 0:
        raise ValueError("tensor.element_size() must be positive")
    return numel * element_size


def _shape(tensor: Any) -> tuple[int, ...]:
    try:
        shape = tuple(tensor.shape)
    except (AttributeError, TypeError) as exc:
        raise TypeError("tensor must expose an iterable shape") from exc
    for extent in shape:
        _positive_int("tensor shape extent", extent)
    return shape


def _empty_like_shape(tensor: Any, shape: Sequence[int]) -> Any:
    factory = getattr(tensor, "new_empty", None)
    if not callable(factory):
        raise TypeError("tensor must provide new_empty(size) for shape-only output")
    return factory(tuple(shape))


def _dtype_itemsize(dtype: Any) -> int:
    itemsize = getattr(dtype, "itemsize", None)
    if isinstance(itemsize, int) and not isinstance(itemsize, bool) and itemsize > 0:
        return itemsize
    names = {
        "bool": 1,
        "float16": 2,
        "bfloat16": 2,
        "int16": 2,
        "float32": 4,
        "int32": 4,
        "float64": 8,
        "int64": 8,
    }
    name = str(dtype).removeprefix("torch.")
    if name in names:
        return names[name]
    raise TypeError(f"cannot determine the element size of dtype {dtype!r}")


def _default_tensor_factory(size: Sequence[int], dtype: Any) -> Any:
    if isinstance(dtype, ShapeDType):
        return ShapeTensor(
            tuple(size),
            dtype=dtype,
            element_size_bytes=dtype.itemsize,
        )
    try:
        import torch
    except ImportError:
        return ShapeTensor(
            tuple(size),
            dtype=dtype,
            element_size_bytes=_dtype_itemsize(dtype),
        )
    return torch.empty(tuple(size), dtype=dtype)


class SimGroupCoordinator:
    """Trimmed, shape-only mirror of vLLM's ``GroupCoordinator``."""

    def __init__(
        self,
        *,
        group_name: str,
        ranks: Sequence[int],
        rank: int,
        local_rank: int,
        clock: VirtualClock,
        observer: GroupCoordinatorObserver | None = None,
        route: NcclRoute = NcclRoute.INTRA_NODE,
        stack_config: NcclStackConfig | None = None,
        tensor_factory: TensorFactory | None = None,
    ) -> None:
        if not isinstance(group_name, str) or not group_name:
            raise ValueError("group_name must be a nonempty string")
        if not isinstance(clock, VirtualClock):
            raise TypeError("clock must be a simllm.core.VirtualClock")
        if not isinstance(route, NcclRoute):
            raise TypeError("route must be a simllm.compute.NcclRoute")
        resolved_ranks = list(ranks)
        if not resolved_ranks:
            raise ValueError("ranks must not be empty")
        if len(set(resolved_ranks)) != len(resolved_ranks):
            raise ValueError("ranks must be unique")
        for member in resolved_ranks:
            _positive_int("rank member", member)
        _positive_int("rank", rank)
        _positive_int("local_rank", local_rank)
        if rank not in resolved_ranks:
            raise ValueError(f"rank {rank} is not a member of {resolved_ranks}")
        if observer is not None and observer.clock is not clock:
            raise ValueError("observer and coordinator must share the same clock")

        self.unique_name = group_name
        self.group_name = group_name
        self.rank = rank
        self.ranks = resolved_ranks
        self.world_size = len(resolved_ranks)
        self.local_rank = local_rank
        self.rank_in_group = resolved_ranks.index(rank)
        self.group_ranks = [list(resolved_ranks)]
        self.clock = clock
        self.observer = observer or GroupCoordinatorObserver(clock)
        self.route = route
        self.device = "cpu"
        self.cpu_group = self
        self.device_group = self
        self._tensor_factory = tensor_factory or _default_tensor_factory
        self._events: list[GroupCoordinatorEvent] = []
        self._operation_count = 0
        self.stack = NcclStack(
            clock=clock,
            config=stack_config
            or NcclStackConfig(
                channel_count=1,
                chunk_bytes=1,
                fifo_slots_per_channel=2,
            ),
        )
        self._communicator = None
        if self.world_size > 1:
            self._communicator = ncclCommInitRank(
                self.stack,
                nranks=self.world_size,
                communicator_id=group_name,
                rank=self.rank_in_group,
            )

    @property
    def events(self) -> tuple[GroupCoordinatorEvent, ...]:
        return tuple(self._events)

    @property
    def stack_events(self) -> tuple[NcclStackEvent, ...]:
        return self.stack.events

    @property
    def first_rank(self) -> int:
        return self.ranks[0]

    @property
    def last_rank(self) -> int:
        return self.ranks[-1]

    @property
    def is_first_rank(self) -> bool:
        return self.rank == self.first_rank

    @property
    def is_last_rank(self) -> bool:
        return self.rank == self.last_rank

    @property
    def next_rank(self) -> int:
        return self.ranks[(self.rank_in_group + 1) % self.world_size]

    @property
    def prev_rank(self) -> int:
        return self.ranks[(self.rank_in_group - 1) % self.world_size]

    def _validate_local_peer(self, name: str, peer: int | None) -> int:
        if peer is None:
            return (
                (self.rank_in_group + 1) % self.world_size
                if name == "dst"
                else (self.rank_in_group - 1) % self.world_size
            )
        _positive_int(name, peer)
        if peer >= self.world_size:
            raise ValueError(f"invalid {name} rank {peer} for group size {self.world_size}")
        return peer

    def _observe(self, operation: str, payload_bytes: int) -> GroupCoordinatorEvent:
        timestamp_ps = self.clock.now_ps
        collective = operation.replace("_", "-")
        work = CollectiveWork(
            collective,
            tuple(self.ranks),
            payload_bytes,
            "ring",
        )
        operation_id = f"{self.group_name}:{operation}:{self._operation_count}"
        stack_events: tuple[NcclStackEvent, ...] = ()
        if payload_bytes == 0:
            stack_disposition = "zero_payload_bypass"
        elif self._communicator is None:
            stack_disposition = "singleton_bypass"
        else:
            result = ncclAllReduce(
                self._communicator,
                payload_bytes=payload_bytes,
                operation_id=operation_id,
                route=self.route,
            )
            stack_events = result.events
            stack_disposition = "entered"
        event = self.observer.record(
            timestamp_ps=timestamp_ps,
            operation_id=operation_id,
            operation=operation,
            group=self.group_name,
            rank=self.rank,
            ranks=tuple(self.ranks),
            payload_bytes=payload_bytes,
            work=work,
            stack_disposition=stack_disposition,
            stack_events=stack_events,
        )
        self._events.append(event)
        self._operation_count += 1
        return event

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """Return an input-shaped value and record one all-reduce boundary."""

        payload_bytes = _payload_bytes(input_)
        output = input_ if self.world_size == 1 else _empty_like_shape(input_, _shape(input_))
        self._observe("all_reduce", payload_bytes)
        return output

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Return an empty value with the selected axis gathered."""

        shape = _shape(input_)
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError("dim must be an integer")
        if not -len(shape) <= dim < len(shape):
            raise ValueError(f"invalid dim {dim} for tensor shape {shape}")
        payload_bytes = _payload_bytes(input_)
        if self.world_size == 1:
            output = input_
        else:
            gathered_shape = list(shape)
            gathered_shape[dim] *= self.world_size
            output = _empty_like_shape(input_, gathered_shape)
        self._observe("all_gather", payload_bytes)
        return output

    def broadcast(self, input_: torch.Tensor, src: int = 0):
        """Return the input unchanged after recording a broadcast boundary."""

        self._validate_local_peer("src", src)
        self._observe("broadcast", _payload_bytes(input_))
        return input_

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Record a blocking send boundary without moving tensor contents."""

        self._validate_local_peer("dst", dst)
        self._observe("send", _payload_bytes(tensor))

    def recv(
        self,
        size: torch.Size,
        dtype: torch.dtype,
        src: int | None = None,
    ) -> torch.Tensor:
        """Return an empty tensor with the requested receive shape and dtype."""

        self._validate_local_peer("src", src)
        output = self._tensor_factory(tuple(size), dtype)
        self._observe("recv", _payload_bytes(output))
        return output


__all__ = [
    "FLOAT32",
    "GROUP_COORDINATOR_EVENT_SCHEMA",
    "INT32",
    "GroupCoordinatorEvent",
    "GroupCoordinatorObserver",
    "ShapeDType",
    "ShapeTensor",
    "SimGroupCoordinator",
]
