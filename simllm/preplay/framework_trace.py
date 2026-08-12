"""Canonical JSONL I/O for observed serving-framework trace v2."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

from simllm.core._wire import (
    _array,
    _enum_value,
    _fail,
    _fields,
    _integer,
    _object,
    _optional_integer,
    _optional_string,
    _string,
    _string_tuple,
)
from simllm.preplay.framework_schema import (
    FRAMEWORK_PREPLAY_TRACE_SCHEMA,
    FrameworkPreplayTrace,
    FrameworkRequestTrace,
    FrameworkTraceProvenance,
    KvCacheEvent,
    KvEventKind,
    ObservedLayerDispatch,
    ObservedTokenDispatch,
    validate_framework_preplay_trace,
    validate_framework_request_trace,
    validate_framework_trace_provenance,
)
from simllm.preplay.schema import (
    ForwardPhase,
    PromptFormat,
    SamplingConfig,
    SamplingMode,
    StopReason,
)

_HEADER = "header"
_REQUEST = "request"
_DISPATCH = "observed-dispatch"
_KV_EVENT = "kv-event"
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
        _integer(item, f"{path}[{index}]", nonnegative=True)
        for index, item in enumerate(_array(value, path))
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object field {key!r}")
        result[key] = value
    return result


def _sampling_to_json(config: SamplingConfig) -> dict[str, Any]:
    return {
        "mode": config.mode.value,
        "seed": config.seed,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }


def _sampling_from_json(value: object, path: str) -> SamplingConfig:
    payload = _object(value, path)
    _fields(payload, path, required={"mode", "seed", "temperature", "top_p"})
    seed = payload["seed"]
    temperature = payload["temperature"]
    top_p = payload["top_p"]
    from simllm.core._wire import _number

    return SamplingConfig(
        mode=_enum_value(SamplingMode, payload["mode"], f"{path}.mode"),
        seed=None if seed is None else _integer(seed, f"{path}.seed"),
        temperature=(
            None if temperature is None else _number(temperature, f"{path}.temperature")
        ),
        top_p=None if top_p is None else _number(top_p, f"{path}.top_p"),
    )


def framework_trace_provenance_to_json(
    provenance: FrameworkTraceProvenance,
) -> dict[str, Any]:
    """Return the strict v2 provenance object."""

    validate_framework_trace_provenance(provenance)
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
        "framework": provenance.framework,
        "framework_version": provenance.framework_version,
        "observed_source": provenance.observed_source,
        "authored_against_source": provenance.authored_against_source,
        "torch_version": provenance.torch_version,
        "device": provenance.device,
        "torch_num_threads": provenance.torch_num_threads,
        "engine_seed": provenance.engine_seed,
        "eos_token_id": provenance.eos_token_id,
        "top_k": provenance.top_k,
        "expert_count": provenance.expert_count,
        "moe_layer_indices": list(provenance.moe_layer_indices),
        "kv_page_size": provenance.kv_page_size,
        "kv_token_capacity": provenance.kv_token_capacity,
        "dispatch_layer_mapping": provenance.dispatch_layer_mapping,
        "routing_source": provenance.routing_source,
    }


def framework_trace_provenance_from_json(
    value: object,
    path: str = "provenance",
) -> FrameworkTraceProvenance:
    """Parse and validate the strict v2 provenance object."""

    payload = _object(value, path)
    required = {
        "schema",
        "model_id",
        "model_revision",
        "model_class",
        "dtype",
        "tokenizer_sha256",
        "sampling",
        "capture_host",
        "runner",
        "framework",
        "framework_version",
        "observed_source",
        "authored_against_source",
        "torch_version",
        "device",
        "torch_num_threads",
        "engine_seed",
        "eos_token_id",
        "top_k",
        "expert_count",
        "moe_layer_indices",
        "kv_page_size",
        "kv_token_capacity",
        "dispatch_layer_mapping",
        "routing_source",
    }
    _fields(payload, path, required=required)
    provenance = FrameworkTraceProvenance(
        schema=_string(payload["schema"], f"{path}.schema"),
        model_id=_string(payload["model_id"], f"{path}.model_id"),
        model_revision=_string(payload["model_revision"], f"{path}.model_revision"),
        model_class=_string(payload["model_class"], f"{path}.model_class"),
        dtype=_string(payload["dtype"], f"{path}.dtype"),
        tokenizer_sha256=_string(
            payload["tokenizer_sha256"], f"{path}.tokenizer_sha256"
        ),
        sampling=_sampling_from_json(payload["sampling"], f"{path}.sampling"),
        capture_host=_string(payload["capture_host"], f"{path}.capture_host"),
        runner=_string(payload["runner"], f"{path}.runner"),
        framework=_string(payload["framework"], f"{path}.framework"),
        framework_version=_string(
            payload["framework_version"], f"{path}.framework_version"
        ),
        observed_source=_string(
            payload["observed_source"], f"{path}.observed_source"
        ),
        authored_against_source=_string(
            payload["authored_against_source"], f"{path}.authored_against_source"
        ),
        torch_version=_string(payload["torch_version"], f"{path}.torch_version"),
        device=_string(payload["device"], f"{path}.device"),
        torch_num_threads=_integer(
            payload["torch_num_threads"], f"{path}.torch_num_threads"
        ),
        engine_seed=_integer(payload["engine_seed"], f"{path}.engine_seed"),
        eos_token_id=_integer(payload["eos_token_id"], f"{path}.eos_token_id"),
        top_k=_integer(payload["top_k"], f"{path}.top_k"),
        expert_count=_integer(payload["expert_count"], f"{path}.expert_count"),
        moe_layer_indices=_integer_tuple(
            payload["moe_layer_indices"], f"{path}.moe_layer_indices"
        ),
        kv_page_size=_integer(payload["kv_page_size"], f"{path}.kv_page_size"),
        kv_token_capacity=_integer(
            payload["kv_token_capacity"], f"{path}.kv_token_capacity"
        ),
        dispatch_layer_mapping=_string(
            payload["dispatch_layer_mapping"],
            f"{path}.dispatch_layer_mapping",
        ),
        routing_source=_string(
            payload["routing_source"], f"{path}.routing_source"
        ),
    )
    validate_framework_trace_provenance(provenance, path)
    return provenance


def _request_to_json(request: FrameworkRequestTrace) -> dict[str, Any]:
    return {
        "schema": FRAMEWORK_PREPLAY_TRACE_SCHEMA,
        "row_type": _REQUEST,
        "request_id": request.request_id,
        "prompt_sha256": request.prompt_sha256,
        "prompt_format": request.prompt_format.value,
        "input_token_ids": list(request.input_token_ids),
        "max_new_tokens": request.max_new_tokens,
        "stop_strings": list(request.stop_strings),
        "output_text": request.output_text,
        "output_token_ids": list(request.output_token_ids),
        "output_length": request.output_length,
        "stop_reason": request.stop_reason.value,
        "matched_stop_string": request.matched_stop_string,
        "framework_cached_tokens": request.framework_cached_tokens,
        "framework_preemption_count": request.framework_preemption_count,
    }


def _request_from_json(value: Mapping[str, Any], path: str) -> dict[str, Any]:
    required = {
        "schema",
        "row_type",
        "request_id",
        "prompt_sha256",
        "prompt_format",
        "input_token_ids",
        "max_new_tokens",
        "stop_strings",
        "output_text",
        "output_token_ids",
        "output_length",
        "stop_reason",
        "matched_stop_string",
        "framework_cached_tokens",
        "framework_preemption_count",
    }
    _fields(value, path, required=required)
    _validate_envelope(value, path, _REQUEST)
    return {
        "request_id": _string(value["request_id"], f"{path}.request_id"),
        "prompt_sha256": _string(value["prompt_sha256"], f"{path}.prompt_sha256"),
        "prompt_format": _enum_value(
            PromptFormat, value["prompt_format"], f"{path}.prompt_format"
        ),
        "input_token_ids": _integer_tuple(
            value["input_token_ids"], f"{path}.input_token_ids"
        ),
        "max_new_tokens": _integer(
            value["max_new_tokens"], f"{path}.max_new_tokens"
        ),
        "stop_strings": _string_tuple(
            value["stop_strings"], f"{path}.stop_strings"
        ),
        "output_text": _string(
            value["output_text"], f"{path}.output_text", nonblank=False
        ),
        "output_token_ids": _integer_tuple(
            value["output_token_ids"], f"{path}.output_token_ids"
        ),
        "output_length": _integer(
            value["output_length"], f"{path}.output_length"
        ),
        "stop_reason": _enum_value(
            StopReason, value["stop_reason"], f"{path}.stop_reason"
        ),
        "matched_stop_string": _optional_string(
            value["matched_stop_string"], f"{path}.matched_stop_string"
        ),
        "framework_cached_tokens": _integer(
            value["framework_cached_tokens"],
            f"{path}.framework_cached_tokens",
            nonnegative=True,
        ),
        "framework_preemption_count": _integer(
            value["framework_preemption_count"],
            f"{path}.framework_preemption_count",
            nonnegative=True,
        ),
    }


def _dispatch_to_json(
    request_id: str,
    dispatch: ObservedTokenDispatch,
) -> dict[str, Any]:
    return {
        "schema": FRAMEWORK_PREPLAY_TRACE_SCHEMA,
        "row_type": _DISPATCH,
        "request_id": request_id,
        "routing_source": dispatch.routing_source,
        "phase": dispatch.phase.value,
        "token_index": dispatch.token_index,
        "token_id": dispatch.token_id,
        "routing": [
            {
                "layer_index": route.layer_index,
                "expert_ids": list(route.expert_ids),
            }
            for route in dispatch.routing
        ],
    }


def _dispatch_from_json(
    value: Mapping[str, Any],
    path: str,
    *,
    request_id: str,
) -> ObservedTokenDispatch:
    required = {
        "schema",
        "row_type",
        "request_id",
        "routing_source",
        "phase",
        "token_index",
        "token_id",
        "routing",
    }
    _fields(value, path, required=required)
    _validate_envelope(value, path, _DISPATCH)
    actual_request_id = _string(value["request_id"], f"{path}.request_id")
    if actual_request_id != request_id:
        _fail(f"{path}.request_id", f"expected {request_id!r}")
    routing = []
    for index, raw_route in enumerate(_array(value["routing"], f"{path}.routing")):
        route_path = f"{path}.routing[{index}]"
        route = _object(raw_route, route_path)
        _fields(route, route_path, required={"layer_index", "expert_ids"})
        routing.append(
            ObservedLayerDispatch(
                layer_index=_integer(
                    route["layer_index"], f"{route_path}.layer_index"
                ),
                expert_ids=_integer_tuple(
                    route["expert_ids"], f"{route_path}.expert_ids"
                ),
            )
        )
    return ObservedTokenDispatch(
        phase=_enum_value(ForwardPhase, value["phase"], f"{path}.phase"),
        token_index=_integer(value["token_index"], f"{path}.token_index"),
        token_id=_integer(value["token_id"], f"{path}.token_id"),
        routing=tuple(routing),
        routing_source=_string(
            value["routing_source"], f"{path}.routing_source"
        ),
    )


def _event_to_json(event: KvCacheEvent) -> dict[str, Any]:
    return {
        "schema": FRAMEWORK_PREPLAY_TRACE_SCHEMA,
        "row_type": _KV_EVENT,
        "sequence": event.sequence,
        "event_kind": event.kind.value,
        "request_id": event.request_id,
        "framework_step": event.framework_step,
        "token_count": event.token_count,
        "block_ids": list(event.block_ids),
        "token_slot_ids": list(event.token_slot_ids),
        "reason": event.reason,
    }


def _event_from_json(value: Mapping[str, Any], path: str) -> KvCacheEvent:
    required = {
        "schema",
        "row_type",
        "sequence",
        "event_kind",
        "request_id",
        "framework_step",
        "token_count",
        "block_ids",
        "token_slot_ids",
        "reason",
    }
    _fields(value, path, required=required)
    _validate_envelope(value, path, _KV_EVENT)
    return KvCacheEvent(
        sequence=_integer(value["sequence"], f"{path}.sequence"),
        kind=_enum_value(KvEventKind, value["event_kind"], f"{path}.event_kind"),
        request_id=_optional_string(value["request_id"], f"{path}.request_id"),
        framework_step=_optional_integer(
            value["framework_step"], f"{path}.framework_step", nonnegative=True
        ),
        token_count=_integer(value["token_count"], f"{path}.token_count"),
        block_ids=_integer_tuple(value["block_ids"], f"{path}.block_ids"),
        token_slot_ids=_integer_tuple(
            value["token_slot_ids"], f"{path}.token_slot_ids"
        ),
        reason=_optional_string(value["reason"], f"{path}.reason"),
    )


def _validate_envelope(
    payload: Mapping[str, Any], path: str, expected_row_type: str
) -> None:
    schema = _string(payload.get("schema"), f"{path}.schema")
    if schema != FRAMEWORK_PREPLAY_TRACE_SCHEMA:
        _fail(f"{path}.schema", f"unsupported schema {schema!r}")
    row_type = _string(payload.get("row_type"), f"{path}.row_type")
    if row_type != expected_row_type:
        _fail(f"{path}.row_type", f"expected {expected_row_type!r}")


def write_framework_preplay_trace(
    path: str | Path,
    trace: FrameworkPreplayTrace,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate completely, then write canonical v2 JSONL."""

    validate_framework_preplay_trace(trace)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    prefill_count = sum(len(request.prefill_dispatch) for request in trace.requests)
    decode_count = sum(len(request.decode_dispatch) for request in trace.requests)
    with output.open(mode, encoding="utf-8", newline="\n") as handle:
        rows: list[Mapping[str, Any]] = [
            {
                "schema": FRAMEWORK_PREPLAY_TRACE_SCHEMA,
                "row_type": _HEADER,
                "provenance": framework_trace_provenance_to_json(trace.provenance),
            }
        ]
        for request in trace.requests:
            rows.append(_request_to_json(request))
            rows.extend(
                _dispatch_to_json(request.request_id, dispatch)
                for dispatch in request.prefill_dispatch
            )
            rows.extend(
                _dispatch_to_json(request.request_id, dispatch)
                for dispatch in request.decode_dispatch
            )
        rows.extend(_event_to_json(event) for event in trace.kv_events)
        rows.append(
            {
                "schema": FRAMEWORK_PREPLAY_TRACE_SCHEMA,
                "row_type": _FOOTER,
                "request_count": len(trace.requests),
                "prefill_dispatch_count": prefill_count,
                "decode_dispatch_count": decode_count,
                "kv_event_count": len(trace.kv_events),
            }
        )
        for row in rows:
            handle.write(_canonical_json(row))
            handle.write("\n")
    return output


class _RowReader:
    def __init__(self, handle: TextIO):
        self.handle = handle
        self.line_number = 0

    def read(self, description: str) -> tuple[Mapping[str, Any], str]:
        line = self.handle.readline()
        if line == "":
            _fail(f"line {self.line_number + 1}", f"missing {description}")
        self.line_number += 1
        path = f"line {self.line_number}"
        if not line.strip():
            _fail(path, "blank JSONL rows are not allowed")
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, ValueError) as exc:
            _fail(path, f"invalid JSON: {exc}")
        return _object(value, path), path


def read_framework_preplay_trace(path: str | Path) -> FrameworkPreplayTrace:
    """Load and strictly validate one canonical-shape v2 trace."""

    requests: list[FrameworkRequestTrace] = []
    events: list[KvCacheEvent] = []
    prefill_count = 0
    decode_count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = _RowReader(handle)
        header, header_path = reader.read("trace header")
        _fields(
            header,
            header_path,
            required={"schema", "row_type", "provenance"},
        )
        _validate_envelope(header, header_path, _HEADER)
        provenance = framework_trace_provenance_from_json(
            header["provenance"], f"{header_path}.provenance"
        )

        payload, row_path = reader.read("request, KV event, or footer")
        while payload.get("row_type") == _REQUEST:
            fields = _request_from_json(payload, row_path)
            prefill_dispatch = []
            for index in range(len(fields["input_token_ids"])):
                raw, path_value = reader.read(
                    f"prefill dispatch {index} for request {fields['request_id']!r}"
                )
                prefill_dispatch.append(
                    _dispatch_from_json(
                        raw,
                        path_value,
                        request_id=fields["request_id"],
                    )
                )
            decode_dispatch = []
            for index in range(max(len(fields["output_token_ids"]) - 1, 0)):
                raw, path_value = reader.read(
                    f"decode dispatch {index} for request {fields['request_id']!r}"
                )
                decode_dispatch.append(
                    _dispatch_from_json(
                        raw,
                        path_value,
                        request_id=fields["request_id"],
                    )
                )
            request = FrameworkRequestTrace(
                **fields,
                prefill_dispatch=tuple(prefill_dispatch),
                decode_dispatch=tuple(decode_dispatch),
            )
            validate_framework_request_trace(
                request, provenance, f"request[{len(requests)}]"
            )
            requests.append(request)
            prefill_count += len(prefill_dispatch)
            decode_count += len(decode_dispatch)
            payload, row_path = reader.read("request, KV event, or footer")

        while payload.get("row_type") == _KV_EVENT:
            events.append(_event_from_json(payload, row_path))
            payload, row_path = reader.read("KV event or footer")

        required_footer = {
            "schema",
            "row_type",
            "request_count",
            "prefill_dispatch_count",
            "decode_dispatch_count",
            "kv_event_count",
        }
        _fields(payload, row_path, required=required_footer)
        _validate_envelope(payload, row_path, _FOOTER)
        declared = {
            "request_count": _integer(
                payload["request_count"], f"{row_path}.request_count", nonnegative=True
            ),
            "prefill_dispatch_count": _integer(
                payload["prefill_dispatch_count"],
                f"{row_path}.prefill_dispatch_count",
                nonnegative=True,
            ),
            "decode_dispatch_count": _integer(
                payload["decode_dispatch_count"],
                f"{row_path}.decode_dispatch_count",
                nonnegative=True,
            ),
            "kv_event_count": _integer(
                payload["kv_event_count"],
                f"{row_path}.kv_event_count",
                nonnegative=True,
            ),
        }
        observed = {
            "request_count": len(requests),
            "prefill_dispatch_count": prefill_count,
            "decode_dispatch_count": decode_count,
            "kv_event_count": len(events),
        }
        for field_name, value in observed.items():
            if declared[field_name] != value:
                _fail(
                    f"{row_path}.{field_name}",
                    f"declares {declared[field_name]}, observed {value}",
                )
        for trailing in handle:
            reader.line_number += 1
            if trailing.strip():
                _fail(f"line {reader.line_number}", "content after completeness footer")

    trace = FrameworkPreplayTrace(
        provenance=provenance,
        requests=tuple(requests),
        kv_events=tuple(events),
    )
    validate_framework_preplay_trace(trace)
    return trace
