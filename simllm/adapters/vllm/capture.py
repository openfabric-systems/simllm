"""Native vLLM scheduler-output capture paired with its step projection.

The native half is read directly from ``SchedulerOutput``. It is deliberately
not reconstructed from ``StepRecord``: the surrogate conformance study needs
an independent record that can expose a translator defect. Capture is inert
until a caller supplies a path through ``SimExecutorConfig``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simllm.core import StepRecord, step_record_to_json

NATIVE_STEP_CAPTURE_SCHEMA = "simllm-vllm-native-step-capture-v1"


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _request_id(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class NativeScheduledTokens:
    """One native request/token row in scheduler iteration order."""

    request_id: str
    num_scheduled_tokens: int

    def __post_init__(self) -> None:
        _request_id("request_id", self.request_id)
        _nonnegative_int("num_scheduled_tokens", self.num_scheduled_tokens)


@dataclass(frozen=True)
class VllmNativeStepCapture:
    """One native scheduler decision and the adapter projection beside it."""

    step_index: int
    scheduled: tuple[NativeScheduledTokens, ...]
    total_num_scheduled_tokens: int
    preempted_request_ids: tuple[str, ...]
    finished_request_ids: tuple[str, ...]
    step_record: StepRecord

    def __post_init__(self) -> None:
        _nonnegative_int("step_index", self.step_index)
        _nonnegative_int(
            "total_num_scheduled_tokens", self.total_num_scheduled_tokens
        )
        if self.step_record.step_index != self.step_index:
            raise ValueError("native and projected step indices disagree")
        request_ids = tuple(row.request_id for row in self.scheduled)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("native scheduled request IDs must be unique")
        native_sum = sum(row.num_scheduled_tokens for row in self.scheduled)
        if native_sum != self.total_num_scheduled_tokens:
            raise ValueError(
                "native total_num_scheduled_tokens disagrees with the ordered rows"
            )


def capture_vllm_native_step(
    scheduler_output: Any,
    step_record: StepRecord,
) -> VllmNativeStepCapture:
    """Snapshot the pinned native fields without consulting the projection."""

    native_tokens = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not isinstance(native_tokens, dict):
        raise TypeError("SchedulerOutput.num_scheduled_tokens must be a dict")
    scheduled = tuple(
        NativeScheduledTokens(
            _request_id("scheduled request ID", request_id),
            _nonnegative_int(
                f"num_scheduled_tokens[{request_id!r}]", token_count
            ),
        )
        for request_id, token_count in native_tokens.items()
    )
    total = _nonnegative_int(
        "SchedulerOutput.total_num_scheduled_tokens",
        getattr(scheduler_output, "total_num_scheduled_tokens", None),
    )
    preempted = tuple(
        sorted(
            _request_id("preempted request ID", value)
            for value in (
                getattr(scheduler_output, "preempted_req_ids", None) or ()
            )
        )
    )
    finished = tuple(
        sorted(
            _request_id("finished request ID", value)
            for value in (
                getattr(scheduler_output, "finished_req_ids", None) or ()
            )
        )
    )
    return VllmNativeStepCapture(
        step_index=step_record.step_index,
        scheduled=scheduled,
        total_num_scheduled_tokens=total,
        preempted_request_ids=preempted,
        finished_request_ids=finished,
        step_record=step_record,
    )


def vllm_native_step_capture_to_json(
    capture: VllmNativeStepCapture,
) -> dict[str, Any]:
    """Return the versioned JSON object retained by live harnesses."""

    if not isinstance(capture, VllmNativeStepCapture):
        raise TypeError("capture must be a VllmNativeStepCapture")
    ordered_ids = [row.request_id for row in capture.scheduled]
    return {
        "schema": NATIVE_STEP_CAPTURE_SCHEMA,
        "step_index": capture.step_index,
        "native_scheduler_output": {
            "ordered_scheduled_request_ids": ordered_ids,
            "num_scheduled_tokens": [
                {
                    "request_id": row.request_id,
                    "num_scheduled_tokens": row.num_scheduled_tokens,
                }
                for row in capture.scheduled
            ],
            "total_num_scheduled_tokens": capture.total_num_scheduled_tokens,
            "preempted_request_ids": list(capture.preempted_request_ids),
            "finished_request_ids": list(capture.finished_request_ids),
        },
        "step_record": step_record_to_json(capture.step_record),
    }


class VllmNativeStepCaptureStream:
    """Append paired native/projected steps as soon as they are translated."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._started = False

    @property
    def path(self) -> Path:
        return self._path

    def append(self, capture: VllmNativeStepCapture) -> None:
        if not self._started:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text("", encoding="utf-8")
            self._started = True
        encoded = json.dumps(
            vllm_native_step_capture_to_json(capture),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded + "\n")
