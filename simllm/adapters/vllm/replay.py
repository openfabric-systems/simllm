"""Strict pre-play token serving for both supported vLLM adapter paths."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simllm.preplay import (
    JoinedRequest,
    PreplayReplayRun,
    read_preplay_replay_run,
    read_preplay_trace,
    validate_preplay_replay_run,
)


@dataclass(frozen=True, kw_only=True)
class ReplayServingSnapshot:
    """Read-only evidence from one adapter-local replay projection."""

    served_token_ids: tuple[tuple[str, tuple[int, ...]], ...]
    completed_request_ids: tuple[str, ...]
    drained_request_ids: tuple[str, ...]


class ReplayTokenSource:
    """Serve joined tokens at scheduler-reported per-request output indices.

    vLLM remains the lifecycle and completion authority. This object neither
    advances a request nor predicts a scheduler step. It validates the joined
    trace once, then projects the token at the output index vLLM reports for
    each sampling visit.
    """

    def __init__(self, run: PreplayReplayRun, *, max_model_len: int) -> None:
        validate_preplay_replay_run(run)
        if type(max_model_len) is not int or max_model_len <= 0:
            raise ValueError("max_model_len must be a positive integer")
        self.run = run
        self.max_model_len = max_model_len
        self._requests = {request.request_id: request for request in run.requests}
        self._served: dict[str, list[int]] = {
            request.request_id: [] for request in run.requests
        }
        self._completed: set[str] = set()
        self._drained: set[str] = set()
        self._validate_trace_authority()

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        max_model_len: int,
    ) -> ReplayTokenSource:
        """Load a strict replay-run record and verify its named trace."""

        return cls(read_preplay_replay_run(path), max_model_len=max_model_len)

    def _validate_trace_authority(self) -> None:
        source = Path(self.run.trace.path)
        try:
            trace_bytes = source.read_bytes()
        except OSError as exc:
            raise ValueError(
                f"joined replay trace {self.run.trace.path!r} is not readable"
            ) from exc
        digest = hashlib.sha256(trace_bytes).hexdigest()
        if digest != self.run.trace.sha256:
            raise ValueError(
                "joined replay trace SHA-256 mismatch: "
                f"recorded {self.run.trace.sha256}, observed {digest}"
            )
        trace = read_preplay_trace(source)
        for request in self.run.requests:
            try:
                oracle = trace.by_request_id(request.routing_reference.request_id)
            except KeyError as exc:
                raise ValueError(
                    f"joined request {request.request_id!r} has no routing trace row"
                ) from exc
            if oracle.output_token_ids != request.output_token_ids:
                raise ValueError(
                    f"joined request {request.request_id!r} token IDs disagree "
                    "with its trace row"
                )
            if oracle.stop_reason is not request.stop_reason:
                raise ValueError(
                    f"joined request {request.request_id!r} stop reason disagrees "
                    "with its trace row"
                )
            total_length = len(oracle.input_token_ids) + request.output_length
            if total_length > self.max_model_len:
                raise ValueError(
                    f"joined request {request.request_id!r} needs {total_length} "
                    f"tokens, beyond max_model_len={self.max_model_len}"
                )

    @property
    def trace_sha256(self) -> str:
        return self.run.trace.sha256

    def update_max_model_len(self, max_model_len: int) -> None:
        """Apply a later engine cap only if every joined oracle still fits."""

        if type(max_model_len) is not int or max_model_len <= 0:
            raise ValueError("max_model_len must be a positive integer")
        previous = self.max_model_len
        self.max_model_len = max_model_len
        try:
            self._validate_trace_authority()
        except Exception:
            self.max_model_len = previous
            raise

    def _resolve_joined_id(self, runtime_request_id: str) -> str:
        if runtime_request_id in self._requests:
            return runtime_request_id
        raise RuntimeError(
            f"vLLM request {runtime_request_id!r} is missing from the joined replay run"
        )

    def request(self, request_id: str) -> JoinedRequest:
        """Return the joined outcome for one exact scheduler request ID."""

        return self._requests[self._resolve_joined_id(request_id)]

    @staticmethod
    def _prompt_length(new_request: Any) -> int:
        token_ids = getattr(new_request, "prompt_token_ids", None)
        if token_ids is not None:
            return len(token_ids)
        shape = getattr(getattr(new_request, "prompt_embeds", None), "shape", None)
        if shape:
            return int(shape[0])
        raise RuntimeError(
            f"replay request {getattr(new_request, 'req_id', None)!r} has no "
            "exact prompt length"
        )

    @staticmethod
    def _early_stop_positions(
        token_ids: tuple[int, ...],
        stop_ids: set[int],
        min_tokens: int,
    ) -> list[int]:
        return [
            position
            for position, token_id in enumerate(token_ids[:-1], start=1)
            if position >= min_tokens and token_id in stop_ids
        ]

    def _validate_sampling_contract(
        self,
        runtime_request_id: str,
        joined_id: str,
        sampling_params: Any,
        prompt_length: int,
    ) -> None:
        request = self._requests[joined_id]
        if sampling_params is None:
            raise RuntimeError(
                f"replay request {runtime_request_id!r} is not a plain generation "
                "request"
            )
        max_tokens = getattr(sampling_params, "max_tokens", None)
        if type(max_tokens) is not int or max_tokens != request.output_length:
            raise RuntimeError(
                f"replay request {runtime_request_id!r} must enter vLLM with max_tokens="
                f"{request.output_length}, got {max_tokens!r}"
            )
        raw_min_tokens = getattr(sampling_params, "min_tokens", 0)
        if type(raw_min_tokens) is not int or raw_min_tokens < 0:
            raise RuntimeError(
                f"replay request {runtime_request_id!r} has invalid "
                f"min_tokens={raw_min_tokens!r}"
            )
        min_tokens = raw_min_tokens
        if min_tokens > request.output_length:
            raise RuntimeError(
                f"replay request {runtime_request_id!r} has min_tokens={min_tokens} "
                f"beyond its oracle length {request.output_length}"
            )
        eos_token_id = getattr(sampling_params, "eos_token_id", None)
        if eos_token_id is not None and type(eos_token_id) is not int:
            raise RuntimeError(
                f"replay request {runtime_request_id!r} has invalid "
                f"eos_token_id={eos_token_id!r}"
            )
        eos_positions = self._early_stop_positions(
            request.output_token_ids,
            set() if eos_token_id is None else {eos_token_id},
            min_tokens,
        )
        if eos_positions:
            raise RuntimeError(
                f"replay request {runtime_request_id!r} hits eos_token_id "
                f"before its oracle length at output positions {eos_positions}"
            )
        raw_stop_token_ids = getattr(sampling_params, "stop_token_ids", ()) or ()
        if any(type(token_id) is not int for token_id in raw_stop_token_ids):
            raise RuntimeError(
                f"replay request {runtime_request_id!r} has invalid "
                f"stop_token_ids={raw_stop_token_ids!r}"
            )
        stop_token_ids = set(raw_stop_token_ids)
        stop_positions = self._early_stop_positions(
            request.output_token_ids,
            stop_token_ids,
            min_tokens,
        )
        if stop_positions:
            raise RuntimeError(
                f"replay request {runtime_request_id!r} hits a stop token before "
                f"its oracle length at output positions {stop_positions}"
            )
        total_length = prompt_length + request.output_length
        if total_length > self.max_model_len:
            raise RuntimeError(
                f"replay request {runtime_request_id!r} needs {total_length} "
                f"tokens, beyond max_model_len={self.max_model_len}"
            )

    def validate_new_requests(self, scheduler_output: Any) -> None:
        """Validate every admission without changing replay state."""

        for new_request in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            runtime_id = getattr(new_request, "req_id", None)
            if not isinstance(runtime_id, str) or not runtime_id:
                raise RuntimeError("replay scheduler output has an invalid request ID")
            joined_id = self._resolve_joined_id(runtime_id)
            self._validate_sampling_contract(
                runtime_id,
                joined_id,
                getattr(new_request, "sampling_params", None),
                self._prompt_length(new_request),
            )

    @staticmethod
    def _reported_output_indices(scheduler_output: Any) -> dict[str, int]:
        indices: dict[str, int] = {}

        def add(request_id: Any, output_index: int) -> None:
            if not isinstance(request_id, str) or not request_id:
                raise RuntimeError("replay scheduler output has an invalid request ID")
            if request_id in indices:
                raise RuntimeError(
                    f"replay scheduler output repeats request {request_id!r}"
                )
            indices[request_id] = output_index

        for new_request in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            request_id = getattr(new_request, "req_id", None)
            add(request_id, 0)
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        request_ids = list(getattr(cached, "req_ids", ()) or ())
        output_counts = list(getattr(cached, "num_output_tokens", ()) or ())
        if len(request_ids) != len(output_counts):
            raise RuntimeError(
                "replay cached request IDs and output counts have different lengths"
            )
        for request_id, output_count in zip(request_ids, output_counts, strict=True):
            if type(output_count) is not int or output_count < 0:
                raise RuntimeError(
                    f"replay request {request_id!r} has invalid output index "
                    f"{output_count!r}"
                )
            add(request_id, output_count)
        return indices

    def _validated_completion_ids(self, scheduler_output: Any) -> tuple[str, ...]:
        joined_ids: list[str] = []
        seen: set[str] = set()
        for runtime_id in getattr(scheduler_output, "finished_req_ids", ()) or ():
            joined_id = self._resolve_joined_id(runtime_id)
            if joined_id in seen:
                raise RuntimeError(
                    f"replay scheduler output repeats completion {runtime_id!r}"
                )
            seen.add(joined_id)
            request = self._requests[joined_id]
            served = self._served[joined_id]
            if tuple(served) != request.output_token_ids:
                raise RuntimeError(
                    f"replay request {runtime_id!r} drained after {len(served)} "
                    f"tokens, expected {request.output_length}"
                )
            joined_ids.append(joined_id)
        return tuple(joined_ids)

    def _plan_sample(
        self,
        req_ids: Sequence[str],
        produces_token: Sequence[bool],
        scheduler_output: Any,
    ) -> tuple[
        list[str],
        dict[str, int],
        tuple[tuple[str, int | None], ...],
        tuple[str, ...],
    ]:
        """Validate a whole batch and return its mutation-free replay plan."""

        if len(req_ids) != len(produces_token):
            raise ValueError("req_ids and produces_token must have the same length")
        ordered = list(req_ids)
        req_id_to_index = {request_id: index for index, request_id in enumerate(ordered)}
        if len(req_id_to_index) != len(ordered):
            raise ValueError("duplicate request id in a single step")

        drained = self._validated_completion_ids(scheduler_output)
        self.validate_new_requests(scheduler_output)
        output_indices = self._reported_output_indices(scheduler_output)
        decisions: list[tuple[str, int | None]] = []
        for runtime_id, produced in zip(ordered, produces_token, strict=True):
            joined_id = self._resolve_joined_id(runtime_id)
            request = self._requests[joined_id]
            if not produced:
                decisions.append((joined_id, None))
                continue
            if runtime_id not in output_indices:
                raise RuntimeError(
                    f"replay request {runtime_id!r} has no scheduler-reported "
                    "output index"
                )
            output_index = output_indices[runtime_id]
            served = self._served[joined_id]
            if output_index != len(served):
                raise RuntimeError(
                    f"replay request {runtime_id!r} reported output index "
                    f"{output_index}, expected {len(served)}"
                )
            if output_index >= request.output_length:
                raise RuntimeError(
                    f"replay request {runtime_id!r} exhausted its oracle at "
                    f"output index {output_index}"
                )
            decisions.append((joined_id, request.output_token_ids[output_index]))
        return ordered, req_id_to_index, tuple(decisions), drained

    def validate_step(
        self,
        req_ids: Sequence[str],
        produces_token: Sequence[bool],
        scheduler_output: Any,
    ) -> None:
        """Validate a replay batch before its simulated work is settled."""

        self._plan_sample(req_ids, produces_token, scheduler_output)

    def validate_completions(self, scheduler_output: Any) -> None:
        """Validate an empty drain step without changing replay state."""

        self._validated_completion_ids(scheduler_output)

    def sample(
        self,
        req_ids: Sequence[str],
        produces_token: Sequence[bool],
        scheduler_output: Any,
    ) -> tuple[list[str], dict[str, int], list[list[int]]]:
        """Return exact trace tokens in vLLM's required request order."""

        ordered, req_id_to_index, decisions, drained = self._plan_sample(
            req_ids, produces_token, scheduler_output
        )
        self._drained.update(drained)
        sampled: list[list[int]] = []
        for joined_id, token_id in decisions:
            if token_id is None:
                sampled.append([])
                continue
            request = self._requests[joined_id]
            served = self._served[joined_id]
            served.append(token_id)
            sampled.append([token_id])
            if len(served) == request.output_length:
                self._completed.add(joined_id)
        return ordered, req_id_to_index, sampled

    def observe_completions(self, scheduler_output: Any) -> None:
        """Validate delayed scheduler completions without owning their timing."""

        self._drained.update(self._validated_completion_ids(scheduler_output))

    def snapshot(self) -> ReplayServingSnapshot:
        """Return deterministic replay evidence without exposing mutable lists."""

        return ReplayServingSnapshot(
            served_token_ids=tuple(
                (request_id, tuple(self._served[request_id]))
                for request_id in self._requests
            ),
            completed_request_ids=tuple(
                request_id for request_id in self._requests if request_id in self._completed
            ),
            drained_request_ids=tuple(
                request_id for request_id in self._requests if request_id in self._drained
            ),
        )


def sample_adapter_tokens(
    replay: ReplayTokenSource | None,
    req_ids: Sequence[str],
    produces_token: Sequence[bool],
    token_id: int,
    scheduler_output: Any,
    *,
    fabricate: Callable[
        [Sequence[str], Sequence[bool], int],
        tuple[list[str], dict[str, int], list[list[int]]],
    ],
) -> tuple[list[str], dict[str, int], list[list[int]]]:
    """Select strict replay or the unchanged fixed-token identity off path."""

    if replay is None:
        return fabricate(req_ids, produces_token, token_id)
    return replay.sample(req_ids, produces_token, scheduler_output)
