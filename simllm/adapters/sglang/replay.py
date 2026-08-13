"""Strict pre-play token serving for the SGLang worker seam.

The SGLang scheduler stays the lifecycle, completion and KV authority. This
object neither advances a request nor predicts a step: it validates the joined
trace once, refuses every configuration whose finish rule would cut a request
before its oracle length, and then serves the token at the output index the
scheduler itself reports (``len(Req.output_ids)`` at forward time).

Every SGLang field is read with ``getattr`` at the pinned commit's names, so
the whole module is exercised without SGLang installed, in the same style as
:func:`simllm.adapters.sglang.worker.observe_schedule_batch`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
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

__all__ = [
    "SglReplayServingSnapshot",
    "SglReplayTokenSource",
    "sample_adapter_tokens",
]


@dataclass(frozen=True, kw_only=True)
class SglReplayServingSnapshot:
    """Read-only evidence from one adapter-local replay projection."""

    served_token_ids: tuple[tuple[str, tuple[int, ...]], ...]
    completed_request_ids: tuple[str, ...]


def _text_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _int_set(value: Any, label: str, request_id: str) -> set[int]:
    if value is None:
        return set()
    if type(value) is int:
        return {value}
    result: set[int] = set()
    for item in value:
        if type(item) is not int:
            raise RuntimeError(
                f"replay request {request_id!r} has a non-integer {label} entry {item!r}"
            )
        result.add(item)
    return result


class SglReplayTokenSource:
    """Serve joined tokens at the scheduler-reported per-request output index."""

    def __init__(self, run: PreplayReplayRun, *, max_context_len: int) -> None:
        validate_preplay_replay_run(run)
        if type(max_context_len) is not int or max_context_len <= 0:
            raise ValueError("max_context_len must be a positive integer")
        self.run = run
        self.max_context_len = max_context_len
        self._requests = {request.request_id: request for request in run.requests}
        self._served: dict[str, list[int]] = {
            request.request_id: [] for request in run.requests
        }
        self._completed: set[str] = set()
        self._validate_trace_authority()

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        max_context_len: int,
    ) -> SglReplayTokenSource:
        """Load a strict replay-run record and verify its named trace."""

        return cls(read_preplay_replay_run(path), max_context_len=max_context_len)

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
            if total_length > self.max_context_len:
                raise ValueError(
                    f"joined request {request.request_id!r} needs {total_length} "
                    f"tokens, beyond max_context_len={self.max_context_len}"
                )

    @property
    def trace_sha256(self) -> str:
        return self.run.trace.sha256

    def request(self, request_id: str) -> JoinedRequest:
        """Return the joined outcome for one exact scheduler request ID."""

        return self._requests[self._resolve_joined_id(request_id)]

    def _resolve_joined_id(self, runtime_request_id: str) -> str:
        if runtime_request_id in self._requests:
            return runtime_request_id
        raise RuntimeError(
            f"SGLang request {runtime_request_id!r} is missing from the joined replay run"
        )

    @staticmethod
    def _early_stop_positions(
        token_ids: tuple[int, ...],
        stop_ids: set[int],
        min_new_tokens: int,
    ) -> list[int]:
        return [
            position
            for position, token_id in enumerate(token_ids[:-1], start=1)
            if position >= min_new_tokens and token_id in stop_ids
        ]

    def _validate_batch(self, batch: Any) -> None:
        """Refuse the batch-level configurations a replayed token cannot serve."""

        if bool(getattr(batch, "return_logprob", False)):
            raise NotImplementedError(
                "SGLang replay serves no logprobs; remove logprob sampling params (SGL-5)"
            )
        spec = getattr(batch, "spec_algorithm", None)
        is_none = getattr(spec, "is_none", None)
        if spec is not None and callable(is_none) and not is_none():
            raise NotImplementedError(
                "SGLang replay cannot serve speculative decoding: the oracle pins one "
                "token per accepted position and models no draft acceptance (SGL-5)"
            )

    def _validate_request(self, req: Any, row_rid: str, joined_id: str) -> None:
        request = self._requests[joined_id]
        if getattr(req, "grammar", None) is not None:
            raise NotImplementedError(
                f"replay request {row_rid!r} uses structured output, whose grammar "
                "would reject the replayed token"
            )
        params = getattr(req, "sampling_params", None)
        if params is None:
            raise RuntimeError(
                f"replay request {row_rid!r} is not a plain generation request"
            )
        max_new_tokens = getattr(params, "max_new_tokens", None)
        if type(max_new_tokens) is not int or max_new_tokens != request.output_length:
            raise RuntimeError(
                f"replay request {row_rid!r} must enter SGLang with max_new_tokens="
                f"{request.output_length}, got {max_new_tokens!r}"
            )
        raw_min_new_tokens = getattr(params, "min_new_tokens", 0) or 0
        if type(raw_min_new_tokens) is not int or raw_min_new_tokens < 0:
            raise RuntimeError(
                f"replay request {row_rid!r} has invalid "
                f"min_new_tokens={raw_min_new_tokens!r}"
            )
        if raw_min_new_tokens > request.output_length:
            raise RuntimeError(
                f"replay request {row_rid!r} has min_new_tokens={raw_min_new_tokens} "
                f"beyond its oracle length {request.output_length}"
            )
        stop_strings = _text_sequence(getattr(params, "stop_strs", None))
        stop_regexes = _text_sequence(getattr(params, "stop_regex_strs", None))
        if stop_strings or stop_regexes:
            raise NotImplementedError(
                f"replay request {row_rid!r} declares stop strings, which SGLang "
                "matches on detokenized text this worker never produces"
            )
        vocab_size = getattr(req, "vocab_size", None)
        if type(vocab_size) is int:
            outside = [
                token_id
                for token_id in request.output_token_ids
                if not 0 <= token_id < vocab_size
            ]
            if outside:
                raise RuntimeError(
                    f"replay request {row_rid!r} has oracle tokens outside the "
                    f"vocabulary of {vocab_size}: {outside}"
                )
        if not bool(getattr(params, "ignore_eos", False)):
            stop_ids = _int_set(
                getattr(params, "stop_token_ids", None), "stop_token_ids", row_rid
            )
            stop_ids |= _int_set(
                getattr(req, "eos_token_ids", None), "eos_token_ids", row_rid
            )
            tokenizer = getattr(req, "tokenizer", None)
            if tokenizer is not None:
                stop_ids |= _int_set(
                    getattr(tokenizer, "eos_token_id", None), "eos_token_id", row_rid
                )
                stop_ids |= _int_set(
                    getattr(tokenizer, "additional_stop_token_ids", None),
                    "additional_stop_token_ids",
                    row_rid,
                )
            positions = self._early_stop_positions(
                request.output_token_ids, stop_ids, raw_min_new_tokens
            )
            if positions:
                raise RuntimeError(
                    f"replay request {row_rid!r} hits a stop token before its oracle "
                    f"length at output positions {positions}"
                )
        prompt_length = len(getattr(req, "origin_input_ids", ()) or ())
        if prompt_length:
            total_length = prompt_length + request.output_length
            if total_length > self.max_context_len:
                raise RuntimeError(
                    f"replay request {row_rid!r} needs {total_length} tokens, beyond "
                    f"max_context_len={self.max_context_len}"
                )

    def _plan(self, batch: Any, rows: Sequence[Any]) -> tuple[tuple[str, int | None], ...]:
        """Validate a whole batch and return its mutation-free replay plan."""

        self._validate_batch(batch)
        reqs = list(getattr(batch, "reqs", ()) or ())
        if len(reqs) != len(rows):
            raise RuntimeError(
                "replay batch rows and SGLang requests disagree: "
                f"{len(rows)} rows against {len(reqs)} requests"
            )
        seen: set[str] = set()
        decisions: list[tuple[str, int | None]] = []
        for index, (req, row) in enumerate(zip(reqs, rows, strict=True)):
            runtime_id = str(getattr(req, "rid", index))
            if runtime_id != row.rid:
                raise RuntimeError(
                    f"replay batch row {index} names {row.rid!r} but the scheduler "
                    f"request is {runtime_id!r}"
                )
            if runtime_id in seen:
                raise RuntimeError(f"replay batch repeats request {runtime_id!r}")
            seen.add(runtime_id)
            joined_id = self._resolve_joined_id(runtime_id)
            self._validate_request(req, runtime_id, joined_id)
            if not row.produces_token:
                decisions.append((joined_id, None))
                continue
            request = self._requests[joined_id]
            served = self._served[joined_id]
            output_index = row.num_output_tokens
            if output_index != len(served):
                raise RuntimeError(
                    f"replay request {runtime_id!r} reported output index "
                    f"{output_index}, expected {len(served)}"
                )
            if output_index >= request.output_length:
                raise RuntimeError(
                    f"replay request {runtime_id!r} exhausted its oracle at output "
                    f"index {output_index}"
                )
            decisions.append((joined_id, request.output_token_ids[output_index]))
        return tuple(decisions)

    def validate_step(self, batch: Any, rows: Sequence[Any]) -> None:
        """Validate a replay batch without changing replay state."""

        self._plan(batch, rows)

    def sample(
        self,
        batch: Any,
        rows: Sequence[Any],
        *,
        fallback_token_id: int,
    ) -> list[int]:
        """Return one token id per row, in the scheduler's own request order.

        A row that produces no token (a mid-prompt chunked prefill) still needs
        an entry in the worker's ``next_token_ids`` tensor. SGLang discards
        that position in ``process_batch_result_prefill``, so it carries the
        adapter's fabricated token rather than an oracle token, and the oracle
        index stays where the scheduler left it.
        """

        decisions = self._plan(batch, rows)
        tokens: list[int] = []
        for joined_id, token_id in decisions:
            if token_id is None:
                tokens.append(fallback_token_id)
                continue
            request = self._requests[joined_id]
            served = self._served[joined_id]
            served.append(token_id)
            tokens.append(token_id)
            if len(served) == request.output_length:
                self._completed.add(joined_id)
        return tokens

    def snapshot(self) -> SglReplayServingSnapshot:
        """Return deterministic replay evidence without exposing mutable lists."""

        return SglReplayServingSnapshot(
            served_token_ids=tuple(
                (request_id, tuple(self._served[request_id]))
                for request_id in self._requests
            ),
            completed_request_ids=tuple(
                request_id
                for request_id in self._requests
                if request_id in self._completed
            ),
        )


def sample_adapter_tokens(
    replay: SglReplayTokenSource | None,
    batch: Any,
    rows: Sequence[Any],
    token_id: int,
) -> list[int]:
    """Select strict replay or the unchanged fixed-token identity off path."""

    if replay is None:
        return [token_id] * len(rows)
    return replay.sample(batch, rows, fallback_token_id=token_id)
