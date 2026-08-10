"""SGLang-shaped simulated communicator on the shared VLLM-14 base.

SGLang vendors vLLM's ``GroupCoordinator`` but extends ``all_gather`` with
an optional caller-owned output list. The landed VLLM-14 implementation is
torch-optional and already owns shape values, zero-time boundary events,
``CollectiveWork`` lowering, and the COMP-15 compatibility call. This module
subclasses that implementation only to expose SGLang's pinned signature and
output-list behavior. The shared base remains unchanged.

The historical event schema retains its VLLM name because both adapters emit
the exact same immutable event type. This slice adds no runtime projection or
communication timing.
"""

# Exact spellings mirror the pinned SGLang public annotations.
# ruff: noqa: UP006, UP035, UP045

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from simllm.adapters.vllm.communicator import (
    FLOAT32,
    GROUP_COORDINATOR_EVENT_SCHEMA,
    INT32,
    GroupCoordinatorEvent,
    GroupCoordinatorObserver,
    ShapeDType,
    ShapeTensor,
    _payload_bytes,
    _shape,
)
from simllm.adapters.vllm.communicator import (
    SimGroupCoordinator as _SharedSimGroupCoordinator,
)

if TYPE_CHECKING:
    import torch

SGLANG_TP_PAYLOAD_BYTES = 4_096


class SimGroupCoordinator(_SharedSimGroupCoordinator):
    """SGLang signature mirror backed by the shared zero-time coordinator."""

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """Return an input-shaped value and observe SGLang's boundary."""

        return super().all_reduce(input_)

    def all_gather(
        self,
        input_: torch.Tensor,
        dim: int = -1,
        output_tensor_list: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Gather a shape result or observe SGLang's caller-owned output form."""

        if output_tensor_list is None:
            return super().all_gather(input_, dim=dim)

        if not isinstance(output_tensor_list, list):
            raise TypeError("output_tensor_list must be a list")
        if len(output_tensor_list) != self.world_size:
            raise ValueError(
                "output_tensor_list length must equal world size; "
                f"got {len(output_tensor_list)} for {self.world_size} ranks"
            )
        input_shape = _shape(input_)
        for index, output in enumerate(output_tensor_list):
            output_shape = _shape(output)
            if output_shape != input_shape:
                raise ValueError(
                    f"output_tensor_list[{index}] shape {output_shape} "
                    f"does not match input shape {input_shape}"
                )
        self._observe("all_gather", _payload_bytes(input_))
        return None

    def broadcast(self, input_: torch.Tensor, src: int = 0):
        """Return the input after observing SGLang's broadcast boundary."""

        return super().broadcast(input_, src=src)

    def send(
        self,
        tensor: torch.Tensor,
        dst: Optional[int] = None,
    ) -> None:
        """Observe SGLang's blocking send boundary."""

        return super().send(tensor, dst=dst)

    def recv(
        self,
        size: torch.Size,
        dtype: torch.dtype,
        src: Optional[int] = None,
    ) -> torch.Tensor:
        """Return the requested shape after observing the receive boundary."""

        return super().recv(size, dtype, src=src)


def coordinator_event_to_json(event: GroupCoordinatorEvent) -> dict[str, Any]:
    """Return the portable JSON projection used by subprocess smoke tests."""

    if not isinstance(event, GroupCoordinatorEvent):
        raise TypeError("event must be a GroupCoordinatorEvent")
    return {
        "schema": event.schema,
        "sequence": event.sequence,
        "timestamp_ps": event.timestamp_ps,
        "operation_id": event.operation_id,
        "operation": event.operation,
        "group": event.group,
        "rank": event.rank,
        "ranks": list(event.ranks),
        "payload_bytes": event.payload_bytes,
        "work": {
            "collective": event.work.collective,
            "ranks": list(event.work.ranks),
            "payload_bytes": event.work.payload_bytes,
            "algorithm_hint": event.work.algorithm_hint,
            "channel_hint": event.work.channel_hint,
        },
        "stack_disposition": event.stack_disposition,
        "stack_events": [stack_event.to_json() for stack_event in event.stack_events],
    }


class GroupCoordinatorEventStream:
    """Append coordinator events durably from SGLang's scheduler subprocess."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._started = False

    @property
    def path(self) -> Path:
        return self._path

    def append(self, event: GroupCoordinatorEvent) -> None:
        if not self._started:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text("")
            self._started = True
        with open(self._path, "a", newline="\n") as handle:
            handle.write(
                json.dumps(
                    coordinator_event_to_json(event),
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )


__all__ = [
    "FLOAT32",
    "GROUP_COORDINATOR_EVENT_SCHEMA",
    "INT32",
    "SGLANG_TP_PAYLOAD_BYTES",
    "GroupCoordinatorEvent",
    "GroupCoordinatorEventStream",
    "GroupCoordinatorObserver",
    "ShapeDType",
    "ShapeTensor",
    "SimGroupCoordinator",
    "coordinator_event_to_json",
]
