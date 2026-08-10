"""Streaming JSONL writer and strict reader for pre-play traces."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from simllm.core._wire import (
    _array,
    _fail,
    _fields,
    _integer,
    _number,
    _object,
    _optional_string,
    _string,
    _string_tuple,
)
from simllm.preplay.schema import (
    PREPLAY_TRACE_SCHEMA,
    ForwardPhase,
    ForwardTokenTrace,
    LayerRouting,
    PreplayTrace,
    RequestTrace,
    SamplingConfig,
    TraceProvenance,
    forward_phase_from_json,
    prompt_format_from_json,
    sampling_mode_from_json,
    stop_reason_from_json,
    validate_request_trace,
    validate_trace_provenance,
)

_HEADER = "header"
_REQUEST = "request"
_FORWARD_TOKEN = "forward-token"
_FOOTER = "footer"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _integer_tuple(value: object, path: str) -> tuple[int, ...]:
    return tuple(
        _integer(item, f"{path}[{index}]")
        for index, item in enumerate(_array(value, path))
    )


def _number_tuple(value: object, path: str) -> tuple[float, ...]:
    return tuple(
        _number(item, f"{path}[{index}]")
        for index, item in enumerate(_array(value, path))
    )


def _sampling_to_json(config: SamplingConfig) -> dict[str, Any]:
    return {
        "mode": config.mode.value,
        "seed": config.seed,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }


def _sampling_from_json(value: object, path: str) -> SamplingConfig:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={"mode", "seed", "temperature", "top_p"},
    )
    seed_value = payload["seed"]
    temperature_value = payload["temperature"]
    top_p_value = payload["top_p"]
    return SamplingConfig(
        mode=sampling_mode_from_json(payload["mode"], f"{path}.mode"),
        seed=None if seed_value is None else _integer(seed_value, f"{path}.seed"),
        temperature=(
            None
            if temperature_value is None
            else _number(temperature_value, f"{path}.temperature")
        ),
        top_p=None if top_p_value is None else _number(top_p_value, f"{path}.top_p"),
    )


def _trace_provenance_json(provenance: TraceProvenance) -> dict[str, Any]:
    return {
        "schema": provenance.schema,
        "model_id": provenance.model_id,
        "model_revision": provenance.model_revision,
        "model_class": provenance.model_class,
        "dtype": provenance.dtype,
        "tokenizer_sha256": provenance.tokenizer_sha256,
        "sampling": _sampling_to_json(provenance.sampling),
        "capture_host": provenance.capture_host,
        "runner": provenance.runner,
        "transformers_version": provenance.transformers_version,
        "torch_version": provenance.torch_version,
        "device": provenance.device,
        "torch_num_threads": provenance.torch_num_threads,
        "eos_token_id": provenance.eos_token_id,
        "top_k": provenance.top_k,
        "expert_count": provenance.expert_count,
        "moe_layer_indices": list(provenance.moe_layer_indices),
    }


def trace_provenance_to_json(provenance: TraceProvenance) -> dict[str, Any]:
    """Return the strict JSON object for trace provenance."""

    validate_trace_provenance(provenance)
    return _trace_provenance_json(provenance)


def trace_provenance_from_json(value: object, path: str = "provenance") -> TraceProvenance:
    """Parse and validate the strict provenance object."""

    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={
            "schema",
            "model_id",
            "model_revision",
            "model_class",
            "dtype",
            "tokenizer_sha256",
            "sampling",
            "capture_host",
            "runner",
            "transformers_version",
            "torch_version",
            "device",
            "torch_num_threads",
            "eos_token_id",
            "top_k",
            "expert_count",
            "moe_layer_indices",
        },
    )
    provenance = TraceProvenance(
        schema=_string(payload["schema"], f"{path}.schema"),
        model_id=_string(payload["model_id"], f"{path}.model_id"),
        model_revision=_string(payload["model_revision"], f"{path}.model_revision"),
        model_class=_string(payload["model_class"], f"{path}.model_class"),
        dtype=_string(payload["dtype"], f"{path}.dtype"),
        tokenizer_sha256=_string(
            payload["tokenizer_sha256"],
            f"{path}.tokenizer_sha256",
        ),
        sampling=_sampling_from_json(payload["sampling"], f"{path}.sampling"),
        capture_host=_string(payload["capture_host"], f"{path}.capture_host"),
        runner=_string(payload["runner"], f"{path}.runner"),
        transformers_version=_string(
            payload["transformers_version"],
            f"{path}.transformers_version",
        ),
        torch_version=_string(payload["torch_version"], f"{path}.torch_version"),
        device=_string(payload["device"], f"{path}.device"),
        torch_num_threads=_integer(
            payload["torch_num_threads"],
            f"{path}.torch_num_threads",
        ),
        eos_token_id=_integer(payload["eos_token_id"], f"{path}.eos_token_id"),
        top_k=_integer(payload["top_k"], f"{path}.top_k"),
        expert_count=_integer(payload["expert_count"], f"{path}.expert_count"),
        moe_layer_indices=_integer_tuple(
            payload["moe_layer_indices"],
            f"{path}.moe_layer_indices",
        ),
    )
    validate_trace_provenance(provenance, path)
    return provenance


def _routing_to_json(routing: LayerRouting) -> dict[str, Any]:
    return {
        "layer_index": routing.layer_index,
        "expert_ids": list(routing.expert_ids),
        "gate_weights": list(routing.gate_weights),
    }


def _routing_from_json(value: object, path: str) -> LayerRouting:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={"layer_index", "expert_ids", "gate_weights"},
    )
    return LayerRouting(
        layer_index=_integer(payload["layer_index"], f"{path}.layer_index"),
        expert_ids=_integer_tuple(payload["expert_ids"], f"{path}.expert_ids"),
        gate_weights=_number_tuple(payload["gate_weights"], f"{path}.gate_weights"),
    )


def _request_row(request: RequestTrace) -> dict[str, Any]:
    return {
        "schema": PREPLAY_TRACE_SCHEMA,
        "row_type": _REQUEST,
        "request_id": request.request_id,
        "prompt_sha256": request.prompt_sha256,
        "prompt_format": request.prompt_format.value,
        "input_token_ids": list(request.input_token_ids),
        "max_new_tokens": request.max_new_tokens,
        "stop_strings": list(request.stop_strings),
        "output_token_ids": list(request.output_token_ids),
        "output_text": request.output_text,
        "stop_reason": request.stop_reason.value,
        "matched_stop_string": request.matched_stop_string,
    }


def _forward_token_row(request_id: str, token: ForwardTokenTrace) -> dict[str, Any]:
    return {
        "schema": PREPLAY_TRACE_SCHEMA,
        "row_type": _FORWARD_TOKEN,
        "request_id": request_id,
        "phase": token.phase.value,
        "token_index": token.token_index,
        "token_id": token.token_id,
        "routing": [_routing_to_json(route) for route in token.routing],
    }


def _validate_row_envelope(payload: Mapping[str, Any], path: str, row_type: str) -> None:
    schema = _string(payload.get("schema"), f"{path}.schema")
    if schema != PREPLAY_TRACE_SCHEMA:
        _fail(f"{path}.schema", f"unsupported schema {schema!r}")
    actual_type = _string(payload.get("row_type"), f"{path}.row_type")
    if actual_type != row_type:
        _fail(f"{path}.row_type", f"expected {row_type!r}, got {actual_type!r}")


@dataclass(frozen=True)
class _RequestHeader:
    request_id: str
    prompt_sha256: str
    prompt_format: Any
    input_token_ids: tuple[int, ...]
    max_new_tokens: int
    stop_strings: tuple[str, ...]
    output_token_ids: tuple[int, ...]
    output_text: str
    stop_reason: Any
    matched_stop_string: str | None


def _request_header_from_row(payload: Mapping[str, Any], path: str) -> _RequestHeader:
    _fields(
        payload,
        path,
        required={
            "schema",
            "row_type",
            "request_id",
            "prompt_sha256",
            "prompt_format",
            "input_token_ids",
            "max_new_tokens",
            "stop_strings",
            "output_token_ids",
            "output_text",
            "stop_reason",
            "matched_stop_string",
        },
    )
    _validate_row_envelope(payload, path, _REQUEST)
    return _RequestHeader(
        request_id=_string(payload["request_id"], f"{path}.request_id"),
        prompt_sha256=_string(payload["prompt_sha256"], f"{path}.prompt_sha256"),
        prompt_format=prompt_format_from_json(
            payload["prompt_format"],
            f"{path}.prompt_format",
        ),
        input_token_ids=_integer_tuple(
            payload["input_token_ids"],
            f"{path}.input_token_ids",
        ),
        max_new_tokens=_integer(payload["max_new_tokens"], f"{path}.max_new_tokens"),
        stop_strings=_string_tuple(payload["stop_strings"], f"{path}.stop_strings"),
        output_token_ids=_integer_tuple(
            payload["output_token_ids"],
            f"{path}.output_token_ids",
        ),
        output_text=_string(payload["output_text"], f"{path}.output_text", nonblank=False),
        stop_reason=stop_reason_from_json(payload["stop_reason"], f"{path}.stop_reason"),
        matched_stop_string=_optional_string(
            payload["matched_stop_string"],
            f"{path}.matched_stop_string",
        ),
    )


def _forward_token_from_row(
    payload: Mapping[str, Any],
    path: str,
    *,
    request_id: str,
) -> ForwardTokenTrace:
    _fields(
        payload,
        path,
        required={
            "schema",
            "row_type",
            "request_id",
            "phase",
            "token_index",
            "token_id",
            "routing",
        },
    )
    _validate_row_envelope(payload, path, _FORWARD_TOKEN)
    actual_request_id = _string(payload["request_id"], f"{path}.request_id")
    if actual_request_id != request_id:
        _fail(
            f"{path}.request_id",
            f"expected {request_id!r}, got {actual_request_id!r}",
        )
    return ForwardTokenTrace(
        phase=forward_phase_from_json(payload["phase"], f"{path}.phase"),
        token_index=_integer(payload["token_index"], f"{path}.token_index"),
        token_id=_integer(payload["token_id"], f"{path}.token_id"),
        routing=tuple(
            _routing_from_json(route, f"{path}.routing[{index}]")
            for index, route in enumerate(_array(payload["routing"], f"{path}.routing"))
        ),
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object field {key!r}")
        result[key] = value
    return result


class PreplayTraceWriter:
    """Append requests and flush each before accepting the next.

    Existing paths are protected by exclusive creation. Select
    ``overwrite=True`` only when replacing that exact artifact is intended.
    """

    def __init__(
        self,
        path: str | Path,
        provenance: TraceProvenance,
        *,
        overwrite: bool = False,
    ):
        validate_trace_provenance(provenance)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.provenance = provenance
        self._request_ids: set[str] = set()
        self._request_count = 0
        self._prefill_forward_token_count = 0
        self._decode_forward_token_count = 0
        self._closed = False
        mode = "w" if overwrite else "x"
        self._handle = self.path.open(mode, encoding="utf-8", newline="\n")
        self._write_row(
            {
                "schema": PREPLAY_TRACE_SCHEMA,
                "row_type": _HEADER,
                "provenance": _trace_provenance_json(provenance),
            }
        )
        self._handle.flush()

    def _write_row(self, payload: Mapping[str, Any]) -> None:
        self._handle.write(_canonical_json(payload))
        self._handle.write("\n")

    def append(self, request: RequestTrace) -> None:
        """Validate, append, and flush one complete request."""

        if self._closed:
            raise RuntimeError("pre-play trace writer is closed")
        validate_request_trace(request, self.provenance)
        if request.request_id in self._request_ids:
            raise ValueError(f"request.request_id: duplicate value {request.request_id!r}")
        self._write_row(_request_row(request))
        for token in request.prefill_tokens:
            self._write_row(_forward_token_row(request.request_id, token))
        for token in request.decode_tokens:
            self._write_row(_forward_token_row(request.request_id, token))
        self._handle.flush()
        self._request_ids.add(request.request_id)
        self._request_count += 1
        self._prefill_forward_token_count += len(request.prefill_tokens)
        self._decode_forward_token_count += len(request.decode_tokens)

    def close(self) -> None:
        """Write the completeness footer and close the stream."""

        if self._closed:
            return
        self._write_row(
            {
                "schema": PREPLAY_TRACE_SCHEMA,
                "row_type": _FOOTER,
                "request_count": self._request_count,
                "prefill_forward_token_count": self._prefill_forward_token_count,
                "decode_forward_token_count": self._decode_forward_token_count,
            }
        )
        self._handle.flush()
        self._handle.close()
        self._closed = True

    def abort(self) -> None:
        """Close without a footer so readers reject the partial trace."""

        if self._closed:
            return
        self._handle.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


class PreplayTraceReader:
    """Stream complete requests while enforcing row order and the footer."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle: TextIO = self.path.open("r", encoding="utf-8")
        self._line_number = 0
        self._request_ids: set[str] = set()
        self._request_count = 0
        self._prefill_forward_token_count = 0
        self._decode_forward_token_count = 0
        self._complete = False
        self._closed = False
        try:
            payload, row_path = self._read_required_row("trace header")
            _fields(
                payload,
                row_path,
                required={"schema", "row_type", "provenance"},
            )
            _validate_row_envelope(payload, row_path, _HEADER)
            self.provenance = trace_provenance_from_json(
                payload["provenance"],
                f"{row_path}.provenance",
            )
        except BaseException:
            self._handle.close()
            self._closed = True
            raise

    def _read_required_row(self, description: str) -> tuple[Mapping[str, Any], str]:
        line = self._handle.readline()
        if line == "":
            _fail(f"line {self._line_number + 1}", f"missing {description}")
        self._line_number += 1
        path = f"line {self._line_number}"
        if not line.strip():
            _fail(path, "blank JSONL rows are not allowed")
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, ValueError) as exc:
            _fail(path, f"invalid JSON: {exc}")
        return _object(value, path), path

    def _finish(self, payload: Mapping[str, Any], path: str) -> None:
        _fields(
            payload,
            path,
            required={
                "schema",
                "row_type",
                "request_count",
                "prefill_forward_token_count",
                "decode_forward_token_count",
            },
        )
        _validate_row_envelope(payload, path, _FOOTER)
        declared = {
            "request_count": _integer(
                payload["request_count"],
                f"{path}.request_count",
                nonnegative=True,
            ),
            "prefill_forward_token_count": _integer(
                payload["prefill_forward_token_count"],
                f"{path}.prefill_forward_token_count",
                nonnegative=True,
            ),
            "decode_forward_token_count": _integer(
                payload["decode_forward_token_count"],
                f"{path}.decode_forward_token_count",
                nonnegative=True,
            ),
        }
        observed = {
            "request_count": self._request_count,
            "prefill_forward_token_count": self._prefill_forward_token_count,
            "decode_forward_token_count": self._decode_forward_token_count,
        }
        for field_name, observed_value in observed.items():
            if declared[field_name] != observed_value:
                _fail(
                    f"{path}.{field_name}",
                    f"declares {declared[field_name]}, observed {observed_value}",
                )
        for trailing_line in self._handle:
            self._line_number += 1
            if trailing_line.strip():
                _fail(f"line {self._line_number}", "content after completeness footer")
        self._complete = True

    def _read_forward_tokens(
        self,
        *,
        request_id: str,
        phase: ForwardPhase,
        count: int,
    ) -> tuple[ForwardTokenTrace, ...]:
        tokens: list[ForwardTokenTrace] = []
        for token_index in range(count):
            payload, path = self._read_required_row(
                f"{phase.value} forward-token row {token_index} "
                f"for request {request_id!r}"
            )
            tokens.append(
                _forward_token_from_row(
                    payload,
                    path,
                    request_id=request_id,
                )
            )
        return tuple(tokens)

    def __iter__(self) -> PreplayTraceReader:
        return self

    def __next__(self) -> RequestTrace:
        if self._complete:
            raise StopIteration
        payload, path = self._read_required_row("request row or completeness footer")
        row_type = _string(payload.get("row_type"), f"{path}.row_type")
        if row_type == _FOOTER:
            self._finish(payload, path)
            raise StopIteration
        if row_type != _REQUEST:
            _fail(f"{path}.row_type", f"expected {_REQUEST!r} or {_FOOTER!r}")
        header = _request_header_from_row(payload, path)
        if header.request_id in self._request_ids:
            _fail(f"{path}.request_id", f"duplicate request ID {header.request_id!r}")

        prefill_tokens = self._read_forward_tokens(
            request_id=header.request_id,
            phase=ForwardPhase.PREFILL,
            count=len(header.input_token_ids),
        )
        decode_tokens = self._read_forward_tokens(
            request_id=header.request_id,
            phase=ForwardPhase.DECODE,
            count=max(len(header.output_token_ids) - 1, 0),
        )
        request = RequestTrace(
            request_id=header.request_id,
            prompt_sha256=header.prompt_sha256,
            prompt_format=header.prompt_format,
            input_token_ids=header.input_token_ids,
            max_new_tokens=header.max_new_tokens,
            stop_strings=header.stop_strings,
            output_text=header.output_text,
            output_token_ids=header.output_token_ids,
            stop_reason=header.stop_reason,
            matched_stop_string=header.matched_stop_string,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
        )
        validate_request_trace(request, self.provenance, path)
        self._request_ids.add(request.request_id)
        self._request_count += 1
        self._prefill_forward_token_count += len(request.prefill_tokens)
        self._decode_forward_token_count += len(request.decode_tokens)
        return request

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            if exc_type is None and not self._complete:
                for _ in self:
                    pass
        finally:
            self.close()


def write_preplay_trace(
    path: str | Path,
    provenance: TraceProvenance,
    requests: Iterable[RequestTrace],
    *,
    overwrite: bool = False,
) -> Path:
    """Stream requests without replacing an existing path by default."""

    with PreplayTraceWriter(path, provenance, overwrite=overwrite) as writer:
        for request in requests:
            writer.append(request)
    return Path(path)


def read_preplay_trace(path: str | Path) -> PreplayTrace:
    """Load a trace already validated by the strict streaming reader."""

    with PreplayTraceReader(path) as reader:
        return PreplayTrace(provenance=reader.provenance, requests=tuple(reader))
