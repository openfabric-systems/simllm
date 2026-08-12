"""Lazy SGLang CPU runner for observed framework trace v2."""

from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from array import array
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simllm.core._wire import _integer, _string, _validate_unique
from simllm.preplay.framework_schema import (
    FrameworkPreplayTrace,
    FrameworkRequestTrace,
    FrameworkTraceProvenance,
    KvCacheEvent,
    KvEventKind,
    ObservedLayerDispatch,
    ObservedTokenDispatch,
)
from simllm.preplay.framework_trace import write_framework_preplay_trace
from simllm.preplay.schema import (
    ForwardPhase,
    PromptFormat,
    SamplingConfig,
    StopReason,
)

_REPOSITORY_ROOT = Path(__file__).parents[2]
_SGLANG_ORACLE_SCHEMA = "simllm-sglang-framework-observation-v1"
_VLLM_ORACLE_SCHEMA = "simllm-vllm-framework-observation-v1"


@dataclass(frozen=True, kw_only=True)
class FrameworkOracleRequest:
    """One already-tokenized request submitted to a framework oracle."""

    request_id: str
    prompt_sha256: str
    prompt_format: PromptFormat
    input_token_ids: tuple[int, ...]
    max_new_tokens: int
    stop_strings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_token_ids", tuple(self.input_token_ids))
        object.__setattr__(self, "stop_strings", tuple(self.stop_strings))
        _string(self.request_id, "request.request_id")
        digest = _string(self.prompt_sha256, "request.prompt_sha256")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                "request.prompt_sha256: expected 64 lowercase hexadecimal digits"
            )
        if not isinstance(self.prompt_format, PromptFormat):
            raise TypeError("request.prompt_format: expected PromptFormat")
        if not self.input_token_ids:
            raise ValueError("request.input_token_ids must not be empty")
        for index, token_id in enumerate(self.input_token_ids):
            _integer(token_id, f"request.input_token_ids[{index}]", nonnegative=True)
        _integer(self.max_new_tokens, "request.max_new_tokens", minimum=1)
        for index, stop_string in enumerate(self.stop_strings):
            _string(stop_string, f"request.stop_strings[{index}]")
        _validate_unique(self.stop_strings, "request.stop_strings")

    @classmethod
    def from_v1_request(cls, request: Any) -> FrameworkOracleRequest:
        """Copy only request inputs from a validated v1 request row."""

        return cls(
            request_id=request.request_id,
            prompt_sha256=request.prompt_sha256,
            prompt_format=request.prompt_format,
            input_token_ids=tuple(request.input_token_ids),
            max_new_tokens=request.max_new_tokens,
            stop_strings=tuple(request.stop_strings),
        )


def prompt_sha256(prompt: str) -> str:
    """Hash one prompt using the v1 UTF-8 convention."""

    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


@contextmanager
def _oracle_environment(sidecar: Path, torch_num_threads: int) -> Iterator[None]:
    existing_pythonpath = os.environ.get("PYTHONPATH")
    repository = str(_REPOSITORY_ROOT)
    pythonpath = (
        repository
        if not existing_pythonpath
        else repository + os.pathsep + existing_pythonpath
    )
    values = {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": str(torch_num_threads),
        "SIMLLM_SGLANG_ENABLE": "0",
        "SIMLLM_SGLANG_ORACLE_CAPTURE": "1",
        "SIMLLM_SGLANG_ORACLE_LOG": str(sidecar),
        "SGLANG_PLUGINS": "simllm",
        "SGLANG_USE_CPU_ENGINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONPATH": pythonpath,
    }
    previous = {name: os.environ.get(name) for name in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _read_sidecar(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError("SGLang oracle plugin produced no observation sidecar")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise RuntimeError(f"SGLang oracle sidecar line {line_number} is blank")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(
                f"SGLang oracle sidecar line {line_number} is not an object"
            )
        if value.get("schema") != _SGLANG_ORACLE_SCHEMA:
            raise RuntimeError(
                f"SGLang oracle sidecar line {line_number} has the wrong schema"
            )
        rows.append(value)
    return rows


def _read_observation_sidecar(
    path: Path, *, framework: str, schema: str
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"{framework} oracle plugin produced no observation sidecar")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise RuntimeError(
                f"{framework} oracle sidecar line {line_number} is blank"
            )
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(
                f"{framework} oracle sidecar line {line_number} is not an object"
            )
        if value.get("schema") != schema:
            raise RuntimeError(
                f"{framework} oracle sidecar line {line_number} has the wrong schema"
            )
        rows.append(value)
    return rows


def _routed_expert_ids(
    response: dict[str, Any],
    *,
    token_count: int,
    layer_count: int,
    top_k: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    meta_info = response.get("meta_info")
    if not isinstance(meta_info, dict):
        raise TypeError("SGLang response has no meta_info object")
    encoded = meta_info.get("routed_experts")
    if not isinstance(encoded, str) or not encoded:
        raise RuntimeError("SGLang returned no post-selection routed experts")
    raw = base64.b64decode(encoded, validate=True)
    values = array("i")
    if values.itemsize != 4:
        raise RuntimeError("native int width is not compatible with SGLang int32")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    expected = token_count * layer_count * top_k
    if len(values) != expected:
        raise RuntimeError(
            "SGLang routed-expert cardinality differs from "
            f"tokens*layers*top_k: {len(values)} != {expected}"
        )
    result = []
    offset = 0
    for _token_index in range(token_count):
        layers = []
        for _layer_index in range(layer_count):
            layers.append(tuple(int(value) for value in values[offset : offset + top_k]))
            offset += top_k
        result.append(tuple(layers))
    return tuple(result)


def _normalize_finish_reason(
    response: dict[str, Any],
    request: FrameworkOracleRequest,
    *,
    eos_token_id: int,
) -> tuple[StopReason, str | None]:
    meta_info = response.get("meta_info")
    reason = None if not isinstance(meta_info, dict) else meta_info.get("finish_reason")
    if not isinstance(reason, dict):
        raise TypeError("SGLang response has no structured finish_reason")
    kind = reason.get("type")
    if kind == "length":
        return StopReason.LENGTH_CAP, None
    if kind != "stop":
        raise RuntimeError(f"unsupported SGLang finish reason {reason!r}")
    matched = reason.get("matched")
    if isinstance(matched, str):
        if matched not in request.stop_strings:
            raise RuntimeError(
                f"SGLang matched unconfigured stop string {matched!r}"
            )
        return StopReason.STOP_STRING, matched
    matched_ids = matched if isinstance(matched, list) else [matched]
    if eos_token_id not in matched_ids:
        raise RuntimeError(f"SGLang stopped on unsupported token {matched!r}")
    return StopReason.EOS, None


def _request_from_response(
    response: dict[str, Any],
    request: FrameworkOracleRequest,
    provenance: FrameworkTraceProvenance,
) -> FrameworkRequestTrace:
    output_ids_value = response.get("output_ids")
    if not isinstance(output_ids_value, list) or not output_ids_value:
        raise RuntimeError(f"SGLang request {request.request_id!r} returned no token IDs")
    output_ids = tuple(int(value) for value in output_ids_value)
    stop_reason, matched_stop_string = _normalize_finish_reason(
        response,
        request,
        eos_token_id=provenance.eos_token_id,
    )
    forwarded_ids = request.input_token_ids + output_ids[:-1]
    routed = _routed_expert_ids(
        response,
        token_count=len(forwarded_ids),
        layer_count=len(provenance.moe_layer_indices),
        top_k=provenance.top_k,
    )
    token_rows = []
    for flat_index, (token_id, layer_experts) in enumerate(
        zip(forwarded_ids, routed, strict=True)
    ):
        if flat_index < len(request.input_token_ids):
            phase = ForwardPhase.PREFILL
            token_index = flat_index
        else:
            phase = ForwardPhase.DECODE
            token_index = flat_index - len(request.input_token_ids)
        token_rows.append(
            ObservedTokenDispatch(
                phase=phase,
                token_index=token_index,
                token_id=token_id,
                routing=tuple(
                    ObservedLayerDispatch(
                        layer_index=layer_index,
                        expert_ids=expert_ids,
                    )
                    for layer_index, expert_ids in zip(
                        provenance.moe_layer_indices,
                        layer_experts,
                        strict=True,
                    )
                ),
            )
        )
    split = len(request.input_token_ids)
    output_text = response.get("text")
    if not isinstance(output_text, str):
        raise TypeError("SGLang response has no output text")
    meta_info = response["meta_info"]
    cached_tokens = meta_info.get("cached_tokens")
    preemption_count = meta_info.get("num_retractions")
    if type(cached_tokens) is not int or cached_tokens < 0:
        raise TypeError("SGLang response cached_tokens is not a nonnegative integer")
    if type(preemption_count) is not int or preemption_count < 0:
        raise TypeError("SGLang response num_retractions is not a nonnegative integer")
    return FrameworkRequestTrace(
        request_id=request.request_id,
        prompt_sha256=request.prompt_sha256,
        prompt_format=request.prompt_format,
        input_token_ids=request.input_token_ids,
        max_new_tokens=request.max_new_tokens,
        stop_strings=request.stop_strings,
        output_text=output_text,
        output_token_ids=output_ids,
        output_length=len(output_ids),
        stop_reason=stop_reason,
        matched_stop_string=matched_stop_string,
        framework_cached_tokens=cached_tokens,
        framework_preemption_count=preemption_count,
        prefill_dispatch=tuple(token_rows[:split]),
        decode_dispatch=tuple(token_rows[split:]),
    )


def _sidecar_events(
    rows: Sequence[dict[str, Any]], request_ids: set[str]
) -> tuple[KvCacheEvent, ...]:
    storage_rows = [
        row for row in rows if row.get("kind") == "capture-storage-qualified"
    ]
    if not storage_rows:
        raise RuntimeError("SGLang CPU dispatch capture storage was not qualified")
    for row in storage_rows:
        if row.get("device") != "cpu" or row.get("pinned") is not False:
            raise RuntimeError(
                "SGLang CPU dispatch capture storage has unexpected placement"
            )
    marker_positions = [
        index for index, row in enumerate(rows) if row.get("kind") == "capture-start"
    ]
    if len(marker_positions) != 1:
        raise RuntimeError("SGLang oracle sidecar requires exactly one capture-start")
    captured = rows[marker_positions[0] + 1 :]
    layer_rows = [
        row for row in captured if row.get("kind") == "dispatch-layer-qualified"
    ]
    if not layer_rows:
        raise RuntimeError("SGLang Granite dispatch layer mapping was not qualified")
    for row in layer_rows:
        if (
            row.get("mapping") != "granite-model-order"
            or row.get("selected_experts_unchanged") is not True
        ):
            raise RuntimeError("SGLang dispatch layer mapping is not observation-only")
    worker_rows = [row for row in captured if row.get("kind") == "worker-qualified"]
    if not worker_rows:
        raise RuntimeError("SGLang stock worker qualification was not observed")
    for row in worker_rows:
        if row.get("worker_class") != "TpModelWorker":
            raise RuntimeError(f"unexpected SGLang worker {row.get('worker_class')!r}")
        if row.get("model_runner_class") != "ModelRunner":
            raise RuntimeError(
                f"unexpected SGLang model runner {row.get('model_runner_class')!r}"
            )
        if row.get("model_class") != "GraniteMoeForCausalLM":
            raise RuntimeError(
                f"unexpected SGLang model class {row.get('model_class')!r}"
            )
        if row.get("parameter_devices") != ["cpu"]:
            raise RuntimeError(
                f"SGLang model parameters are not all on CPU: {row.get('parameter_devices')!r}"
            )

    events = []
    mapping = {
        "allocation": KvEventKind.ALLOCATION,
        "prefix-hit": KvEventKind.PREFIX_HIT,
        "eviction": KvEventKind.EVICTION,
        "preemption": KvEventKind.PREEMPTION,
        "release": KvEventKind.RELEASE,
    }
    for row in captured:
        kind_value = row.get("kind")
        if kind_value not in mapping:
            continue
        request_id = row.get("request_id")
        if request_id is not None and request_id not in request_ids:
            raise RuntimeError(
                f"SGLang KV event names unknown request {request_id!r}"
            )
        slots = row.get("token_slot_ids", [])
        if not isinstance(slots, list):
            raise TypeError("SGLang KV event token_slot_ids is not a list")
        events.append(
            KvCacheEvent(
                sequence=len(events),
                kind=mapping[kind_value],
                request_id=request_id,
                framework_step=row.get("framework_step"),
                token_count=int(row.get("token_count", 0)),
                token_slot_ids=tuple(int(value) for value in slots),
                reason=row.get("reason"),
            )
        )
    if not any(event.kind is KvEventKind.ALLOCATION for event in events):
        raise RuntimeError("SGLang paged allocation hook produced no events")
    if not any(event.kind is KvEventKind.PREFIX_HIT for event in events):
        raise RuntimeError("SGLang prefix hook produced no events")
    return tuple(events)


class SglangCpuRunner:
    """Capture actual SGLang CPU outcomes, dispatch, and KV decisions."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        model_path: str | Path,
        tokenizer_sha256: str,
        observed_source: str,
        authored_against_source: str,
        dtype: str = "float32",
        torch_num_threads: int = 8,
        engine_seed: int = 173,
        capture_host: str | None = None,
    ):
        self.model_id = _string(model_id, "model_id")
        self.model_revision = _string(model_revision, "model_revision")
        self.model_path = Path(model_path)
        if not self.model_path.is_dir() or not (self.model_path / "config.json").is_file():
            raise FileNotFoundError(f"model path is not a local snapshot: {self.model_path}")
        self.tokenizer_sha256 = _string(tokenizer_sha256, "tokenizer_sha256")
        self.observed_source = _string(observed_source, "observed_source")
        self.authored_against_source = _string(
            authored_against_source, "authored_against_source"
        )
        self.dtype = _string(dtype, "dtype")
        self.torch_num_threads = _integer(
            torch_num_threads, "torch_num_threads", minimum=1
        )
        self.engine_seed = _integer(engine_seed, "engine_seed", nonnegative=True)
        self.capture_host = platform.node() if capture_host is None else _string(
            capture_host, "capture_host"
        )

    def capture_groups(
        self,
        groups: Iterable[Iterable[FrameworkOracleRequest]],
        path: str | Path,
        *,
        max_total_tokens: int,
        context_length: int,
        max_running_requests: int,
        page_size: int = 1,
        overwrite: bool = False,
        observation_path: str | Path | None = None,
        raw_response_path: str | Path | None = None,
    ) -> Path:
        """Run submission groups in order and write one strict v2 trace."""

        normalized_groups = tuple(tuple(group) for group in groups)
        requests = tuple(request for group in normalized_groups for request in group)
        if not requests or any(not group for group in normalized_groups):
            raise ValueError("capture groups must be nonempty")
        if any(not isinstance(request, FrameworkOracleRequest) for request in requests):
            raise TypeError("capture groups must contain FrameworkOracleRequest values")
        request_ids = tuple(request.request_id for request in requests)
        _validate_unique(request_ids, "requests.request_id")
        _integer(max_total_tokens, "max_total_tokens", minimum=1)
        _integer(context_length, "context_length", minimum=1)
        _integer(max_running_requests, "max_running_requests", minimum=1)
        _integer(page_size, "page_size", minimum=1)
        output = Path(path)
        if output.exists() and not overwrite:
            raise FileExistsError(output)
        sidecar = (
            Path(observation_path)
            if observation_path is not None
            else output.with_name(output.name + ".sglang-observations.jsonl")
        ).resolve()
        if sidecar.exists() and not overwrite:
            raise FileExistsError(sidecar)
        raw_responses = (
            Path(raw_response_path)
            if raw_response_path is not None
            else output.with_name(output.name + ".sglang-responses.json")
        )
        if raw_responses.exists() and not overwrite:
            raise FileExistsError(raw_responses)
        if len({output.resolve(), sidecar, raw_responses.resolve()}) != 3:
            raise ValueError("trace, observation, and raw-response paths must differ")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text("", encoding="utf-8")

        with _oracle_environment(sidecar, self.torch_num_threads):
            if "sglang" in sys.modules:
                raise RuntimeError(
                    "SGLang was imported before CPU oracle environment selection"
                )
            sglang = importlib.import_module("sglang")
            torch = importlib.import_module("torch")
            framework_version = importlib.metadata.version("sglang")
            config = json.loads((self.model_path / "config.json").read_text())
            text_config = config.get("text_config", config)
            eos_value = text_config.get("eos_token_id", config.get("eos_token_id"))
            if isinstance(eos_value, list):
                eos_token_id = int(eos_value[0])
            else:
                eos_token_id = int(eos_value)
            top_k = int(text_config["num_experts_per_tok"])
            expert_count = int(text_config["num_local_experts"])
            layer_count = int(text_config["num_hidden_layers"])

            engine = None
            responses: dict[str, dict[str, Any]] = {}
            try:
                engine = sglang.Engine(
                    model_path=str(self.model_path),
                    device="cpu",
                    dtype=self.dtype,
                    disable_overlap_schedule=True,
                    context_length=context_length,
                    max_running_requests=max_running_requests,
                    max_total_tokens=max_total_tokens,
                    page_size=page_size,
                    random_seed=self.engine_seed,
                    tp_size=1,
                    enable_return_routed_experts=True,
                )
                actual_page_size = int(engine.server_args.page_size)
                actual_token_capacity = int(engine.server_args.max_total_tokens)
                if actual_page_size != page_size:
                    raise RuntimeError(
                        "SGLang page size differs from the requested cell: "
                        f"{actual_page_size} != {page_size}"
                    )
                if actual_token_capacity != max_total_tokens:
                    raise RuntimeError(
                        "SGLang token capacity differs from the requested cell: "
                        f"{actual_token_capacity} != {max_total_tokens}"
                    )
                from simllm.adapters.sglang.oracle import (
                    mark_oracle_capture_start,
                    mark_oracle_submission_group,
                )

                mark_oracle_capture_start(request_ids)
                for group_index, group in enumerate(normalized_groups):
                    mark_oracle_submission_group(
                        group_index,
                        (request.request_id for request in group),
                    )
                    result = engine.generate(
                        input_ids=[list(request.input_token_ids) for request in group],
                        sampling_params=[
                            {
                                "temperature": 0,
                                "max_new_tokens": request.max_new_tokens,
                                **(
                                    {
                                        "stop": list(request.stop_strings),
                                        "no_stop_trim": True,
                                    }
                                    if request.stop_strings
                                    else {}
                                ),
                            }
                            for request in group
                        ],
                        rid=[request.request_id for request in group],
                        return_routed_experts=True,
                    )
                    result_rows = [result] if isinstance(result, dict) else list(result)
                    if len(result_rows) != len(group):
                        raise RuntimeError("SGLang response count differs from submission")
                    for request, response in zip(group, result_rows, strict=True):
                        if not isinstance(response, dict):
                            raise TypeError("SGLang response is not an object")
                        meta_info = response.get("meta_info")
                        response_id = (
                            None if not isinstance(meta_info, dict) else meta_info.get("id")
                        )
                        if response_id != request.request_id:
                            raise RuntimeError(
                                "SGLang response identity differs from submission: "
                                f"{response_id!r} != {request.request_id!r}"
                            )
                        if request.request_id in responses:
                            raise RuntimeError(
                                f"duplicate SGLang response {request.request_id!r}"
                            )
                        responses[request.request_id] = response
            finally:
                if engine is not None:
                    engine.shutdown()

        raw_responses.parent.mkdir(parents=True, exist_ok=True)
        raw_responses.write_text(
            json.dumps(
                {
                    "framework": "sglang",
                    "framework_version": framework_version,
                    "requests": [
                        {
                            "request_id": request.request_id,
                            "response": responses[request.request_id],
                        }
                        for request in requests
                    ],
                },
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )

        provenance = FrameworkTraceProvenance(
            model_id=self.model_id,
            model_revision=self.model_revision,
            model_class="GraniteMoeForCausalLM",
            dtype=self.dtype,
            tokenizer_sha256=self.tokenizer_sha256,
            sampling=SamplingConfig.greedy(),
            capture_host=self.capture_host,
            runner="sglang-cpu",
            framework="sglang",
            framework_version=framework_version,
            observed_source=self.observed_source,
            authored_against_source=self.authored_against_source,
            torch_version=str(torch.__version__),
            device="cpu",
            torch_num_threads=self.torch_num_threads,
            engine_seed=self.engine_seed,
            eos_token_id=eos_token_id,
            top_k=top_k,
            expert_count=expert_count,
            moe_layer_indices=tuple(range(layer_count)),
            kv_page_size=actual_page_size,
            kv_token_capacity=actual_token_capacity,
            dispatch_layer_mapping="granite-model-order",
        )
        request_rows = tuple(
            _request_from_response(responses[request.request_id], request, provenance)
            for request in requests
        )
        sidecar_rows = _read_sidecar(sidecar)
        if not any(row.get("kind") == "plugin-active" for row in sidecar_rows):
            raise RuntimeError("SGLang oracle plugin activation was not observed")
        trace = FrameworkPreplayTrace(
            provenance=provenance,
            requests=request_rows,
            kv_events=_sidecar_events(sidecar_rows, set(request_ids)),
        )
        return write_framework_preplay_trace(
            output,
            trace,
            overwrite=overwrite,
        )

    def capture(
        self,
        requests: Iterable[FrameworkOracleRequest],
        path: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Capture one concurrent submission group."""

        return self.capture_groups((tuple(requests),), path, **kwargs)


@contextmanager
def _vllm_oracle_environment(
    sidecar: Path, torch_num_threads: int
) -> Iterator[None]:
    existing_pythonpath = os.environ.get("PYTHONPATH")
    repository = str(_REPOSITORY_ROOT)
    pythonpath = (
        repository
        if not existing_pythonpath
        else repository + os.pathsep + existing_pythonpath
    )
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is None:
        raise RuntimeError("the vLLM CPU oracle requires Linux CPU affinity support")
    cpu_ids = sorted(get_affinity(0))[:torch_num_threads]
    if len(cpu_ids) != torch_num_threads:
        raise RuntimeError(
            f"vLLM oracle requested {torch_num_threads} CPU threads but only "
            f"{len(cpu_ids)} are available"
        )
    values = {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": str(torch_num_threads),
        "PYTHONPATH": pythonpath,
        "SIMLLM_VLLM_MODE": "",
        "SIMLLM_VLLM_ORACLE_CAPTURE": "1",
        "SIMLLM_VLLM_ORACLE_LOG": str(sidecar),
        "SIMLLM_VLLM_WORKER_MODE": "",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_CPU_OMP_THREADS_BIND": ",".join(str(value) for value in cpu_ids),
        "VLLM_PLUGINS": "simllm_oracle",
        "VLLM_TARGET_DEVICE": "cpu",
    }
    previous = {name: os.environ.get(name) for name in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _vllm_kv_bytes_per_token(config: dict[str, Any], dtype: str) -> int:
    text_config = config.get("text_config", config)
    hidden_size = int(text_config["hidden_size"])
    attention_heads = int(text_config["num_attention_heads"])
    kv_heads = int(text_config["num_key_value_heads"])
    layer_count = int(text_config["num_hidden_layers"])
    if hidden_size % attention_heads:
        raise RuntimeError("model hidden size is not divisible by attention heads")
    dtype_bytes = {"float32": 4, "bfloat16": 2}.get(dtype)
    if dtype_bytes is None:
        raise ValueError(f"unsupported vLLM oracle dtype {dtype!r}")
    head_size = hidden_size // attention_heads
    return 2 * layer_count * kv_heads * head_size * dtype_bytes


def _vllm_stop_reason(
    response: dict[str, Any],
    request: FrameworkOracleRequest,
    *,
    eos_token_id: int,
) -> tuple[StopReason, str | None]:
    finish_reason = response.get("finish_reason")
    stop_reason = response.get("stop_reason")
    if finish_reason == "length":
        return StopReason.LENGTH_CAP, None
    if finish_reason != "stop":
        raise RuntimeError(f"unsupported vLLM finish reason {finish_reason!r}")
    if isinstance(stop_reason, str):
        if stop_reason not in request.stop_strings:
            raise RuntimeError(f"vLLM matched unconfigured stop string {stop_reason!r}")
        return StopReason.STOP_STRING, stop_reason
    if stop_reason not in (None, eos_token_id):
        raise RuntimeError(f"vLLM stopped on unsupported token {stop_reason!r}")
    return StopReason.EOS, None


def _vllm_request_from_response(
    response: dict[str, Any],
    request: FrameworkOracleRequest,
    provenance: FrameworkTraceProvenance,
    *,
    preemption_count: int,
) -> FrameworkRequestTrace:
    output_ids = tuple(int(value) for value in response["output_token_ids"])
    if not output_ids:
        raise RuntimeError(f"vLLM request {request.request_id!r} returned no tokens")
    stop_reason, matched_stop_string = _vllm_stop_reason(
        response,
        request,
        eos_token_id=provenance.eos_token_id,
    )
    shape = tuple(int(value) for value in response["routed_experts_shape"])
    expected_shape = (
        len(request.input_token_ids) + len(output_ids) - 1,
        len(provenance.moe_layer_indices),
        provenance.top_k,
    )
    if shape != expected_shape:
        raise RuntimeError(
            f"vLLM routed-expert shape {shape} differs from {expected_shape}"
        )
    flattened = tuple(int(value) for value in response["routed_experts"])
    if len(flattened) != shape[0] * shape[1] * shape[2]:
        raise RuntimeError("vLLM routed-expert payload disagrees with its shape")
    forwarded = request.input_token_ids + output_ids[:-1]
    rows = []
    offset = 0
    for flat_index, token_id in enumerate(forwarded):
        phase = (
            ForwardPhase.PREFILL
            if flat_index < len(request.input_token_ids)
            else ForwardPhase.DECODE
        )
        token_index = (
            flat_index
            if phase is ForwardPhase.PREFILL
            else flat_index - len(request.input_token_ids)
        )
        routing = []
        for layer_index in provenance.moe_layer_indices:
            experts = flattened[offset : offset + provenance.top_k]
            offset += provenance.top_k
            routing.append(
                ObservedLayerDispatch(
                    layer_index=layer_index,
                    expert_ids=experts,
                )
            )
        rows.append(
            ObservedTokenDispatch(
                phase=phase,
                token_index=token_index,
                token_id=token_id,
                routing=tuple(routing),
            )
        )
    split = len(request.input_token_ids)
    return FrameworkRequestTrace(
        request_id=request.request_id,
        prompt_sha256=request.prompt_sha256,
        prompt_format=request.prompt_format,
        input_token_ids=request.input_token_ids,
        max_new_tokens=request.max_new_tokens,
        stop_strings=request.stop_strings,
        output_text=str(response["output_text"]),
        output_token_ids=output_ids,
        output_length=len(output_ids),
        stop_reason=stop_reason,
        matched_stop_string=matched_stop_string,
        framework_cached_tokens=int(response["num_cached_tokens"]),
        framework_preemption_count=preemption_count,
        prefill_dispatch=tuple(rows[:split]),
        decode_dispatch=tuple(rows[split:]),
    )


def _vllm_sidecar_projection(
    rows: Sequence[dict[str, Any]],
    request_ids: set[str],
) -> tuple[tuple[KvCacheEvent, ...], int, int, str, dict[str, int]]:
    markers = [
        index for index, row in enumerate(rows) if row.get("kind") == "capture-start"
    ]
    if len(markers) != 1:
        raise RuntimeError("vLLM oracle sidecar requires exactly one capture-start")
    mapping_rows = [row for row in rows if row.get("kind") == "request-mapping"]
    internal_to_logical: dict[str, str] = {}
    for row in mapping_rows:
        for mapping in row.get("mappings", []):
            internal = str(mapping["internal_request_id"])
            logical = str(mapping["request_id"])
            if internal in internal_to_logical:
                raise RuntimeError(f"duplicate vLLM internal request ID {internal!r}")
            internal_to_logical[internal] = logical
    if set(internal_to_logical.values()) != request_ids:
        raise RuntimeError("vLLM request mappings do not cover the trace requests")

    manager_rows = [row for row in rows if row.get("kind") == "kv-manager-qualified"]
    if len(manager_rows) != 1:
        raise RuntimeError("vLLM KV manager qualification is not unique")
    manager = manager_rows[0]
    if manager.get("manager_class") != "KVCacheManager":
        raise RuntimeError("vLLM did not construct the stock KVCacheManager")
    worker_rows = [row for row in rows if row.get("kind") == "worker-qualified"]
    if len(worker_rows) != 1:
        raise RuntimeError("vLLM stock worker qualification is not unique")
    worker = worker_rows[0]
    cuda_before = worker.get("cuda_memory_allocated_before")
    cuda_after = worker.get("cuda_memory_allocated_after")
    if (
        worker.get("worker_class") != "CPUWorker"
        or worker.get("model_runner_class") != "CPUModelRunner"
        or worker.get("model_class") != "GraniteMoeForCausalLM"
        or worker.get("parameter_devices") != ["cpu"]
        or type(worker.get("cuda_available_before")) is not bool
        or type(worker.get("cuda_available_after")) is not bool
        or worker.get("cuda_available_after")
        != worker.get("cuda_available_before")
        or type(cuda_before) is not int
        or type(cuda_after) is not int
        or cuda_before < 0
        or cuda_after != cuda_before
    ):
        raise RuntimeError(f"vLLM CPU worker qualification failed: {worker!r}")
    captured = rows[markers[0] + 1 :]
    path_rows = [
        row for row in rows if row.get("kind") == "dispatch-path-qualified"
    ]
    if len(path_rows) != 1:
        raise RuntimeError("vLLM CPU dispatch path qualification is not unique")
    path_row = path_rows[0]
    layer_ids = path_row.get("layer_ids")
    if (
        path_row.get("capture_source")
        != "cpu-monolithic-select-experts-return"
        or path_row.get("selected_experts_unchanged") is not True
        or not isinstance(layer_ids, list)
        or layer_ids != list(range(len(layer_ids)))
    ):
        raise RuntimeError("vLLM CPU dispatch path is not observation-only")
    dispatch_rows = [row for row in captured if row.get("kind") == "dispatch-qualified"]
    if len(dispatch_rows) != 1 or (
        dispatch_rows[0].get("capture_class") != "RoutedExpertsCapturer"
        or dispatch_rows[0].get("capture_source")
        != "post-selection-router-output"
        or dispatch_rows[0].get("selected_experts_unchanged") is not True
    ):
        raise RuntimeError("vLLM post-selection dispatch was not qualified")

    kind_map = {
        "allocation": KvEventKind.ALLOCATION,
        "prefix-hit": KvEventKind.PREFIX_HIT,
        "eviction": KvEventKind.EVICTION,
        "preemption": KvEventKind.PREEMPTION,
        "release": KvEventKind.RELEASE,
    }
    events = []
    for row in captured:
        kind = row.get("kind")
        if kind not in kind_map:
            continue
        internal_id = row.get("request_id")
        request_id = None
        if internal_id is not None:
            request_id = internal_to_logical.get(str(internal_id))
            if request_id is None:
                raise RuntimeError(
                    f"vLLM KV event has unmapped request {internal_id!r}"
                )
        events.append(
            KvCacheEvent(
                sequence=len(events),
                kind=kind_map[str(kind)],
                request_id=request_id,
                framework_step=None,
                token_count=int(row.get("token_count", 0)),
                block_ids=tuple(int(value) for value in row.get("block_ids", [])),
                reason=row.get("reason"),
            )
        )
    if not any(event.kind is KvEventKind.ALLOCATION for event in events):
        raise RuntimeError("vLLM paged allocation hook produced no events")
    if not any(event.kind is KvEventKind.PREFIX_HIT for event in events):
        raise RuntimeError("vLLM prefix hook produced no events")
    final_preemptions: dict[str, int] = {}
    for row in captured:
        if row.get("kind") != "request-final-counters":
            continue
        internal = str(row["request_id"])
        logical = internal_to_logical.get(internal)
        if logical is None:
            raise RuntimeError(
                f"vLLM final counter has unmapped request {internal!r}"
            )
        if logical in final_preemptions:
            raise RuntimeError(f"duplicate vLLM final counter for {logical!r}")
        count = row.get("num_preemptions")
        if type(count) is not int or count < 0:
            raise RuntimeError(f"invalid vLLM preemption counter for {logical!r}")
        final_preemptions[logical] = count
    if set(final_preemptions) != request_ids:
        raise RuntimeError("vLLM final counters do not cover the trace requests")
    return (
        tuple(events),
        int(manager["block_size"]),
        int(manager["token_capacity"]),
        str(worker["model_class"]),
        final_preemptions,
    )


class VllmCpuRunner:
    """Capture stock vLLM CPU outcomes, routed experts, and paged-KV events."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        model_path: str | Path,
        tokenizer_sha256: str,
        observed_source: str,
        authored_against_source: str,
        dtype: str = "float32",
        torch_num_threads: int = 8,
        engine_seed: int = 173,
        capture_host: str | None = None,
    ):
        self.model_id = _string(model_id, "model_id")
        self.model_revision = _string(model_revision, "model_revision")
        self.model_path = Path(model_path)
        if not self.model_path.is_dir() or not (self.model_path / "config.json").is_file():
            raise FileNotFoundError(f"model path is not a local snapshot: {self.model_path}")
        self.tokenizer_sha256 = _string(tokenizer_sha256, "tokenizer_sha256")
        self.observed_source = _string(observed_source, "observed_source")
        self.authored_against_source = _string(
            authored_against_source, "authored_against_source"
        )
        self.dtype = _string(dtype, "dtype")
        self.torch_num_threads = _integer(
            torch_num_threads, "torch_num_threads", minimum=1
        )
        self.engine_seed = _integer(engine_seed, "engine_seed", nonnegative=True)
        self.capture_host = platform.node() if capture_host is None else _string(
            capture_host, "capture_host"
        )

    def capture_groups(
        self,
        groups: Iterable[Iterable[FrameworkOracleRequest]],
        path: str | Path,
        *,
        kv_token_capacity: int,
        context_length: int,
        max_running_requests: int,
        block_size: int = 32,
        overwrite: bool = False,
        observation_path: str | Path | None = None,
        raw_response_path: str | Path | None = None,
    ) -> Path:
        """Run request groups through one stock vLLM CPU engine."""

        normalized_groups = tuple(tuple(group) for group in groups)
        requests = tuple(request for group in normalized_groups for request in group)
        if not requests or any(not group for group in normalized_groups):
            raise ValueError("capture groups must be nonempty")
        if any(not isinstance(request, FrameworkOracleRequest) for request in requests):
            raise TypeError("capture groups must contain FrameworkOracleRequest values")
        request_ids = tuple(request.request_id for request in requests)
        _validate_unique(request_ids, "requests.request_id")
        _integer(kv_token_capacity, "kv_token_capacity", minimum=1)
        _integer(context_length, "context_length", minimum=1)
        _integer(max_running_requests, "max_running_requests", minimum=1)
        _integer(block_size, "block_size", minimum=1)
        if kv_token_capacity % block_size:
            raise ValueError("kv_token_capacity must be block aligned")
        output = Path(path)
        if output.exists() and not overwrite:
            raise FileExistsError(output)
        sidecar = (
            Path(observation_path)
            if observation_path is not None
            else output.with_name(output.name + ".vllm-observations.jsonl")
        ).resolve()
        raw_responses = (
            Path(raw_response_path)
            if raw_response_path is not None
            else output.with_name(output.name + ".vllm-responses.json")
        )
        for artifact in (sidecar, raw_responses):
            if artifact.exists() and not overwrite:
                raise FileExistsError(artifact)
        if len({output.resolve(), sidecar, raw_responses.resolve()}) != 3:
            raise ValueError("trace, observation, and raw-response paths must differ")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text("", encoding="utf-8")

        config = json.loads((self.model_path / "config.json").read_text())
        text_config = config.get("text_config", config)
        eos_value = text_config.get("eos_token_id", config.get("eos_token_id"))
        eos_token_id = int(eos_value[0] if isinstance(eos_value, list) else eos_value)
        top_k = int(text_config["num_experts_per_tok"])
        expert_count = int(text_config["num_local_experts"])
        layer_count = int(text_config["num_hidden_layers"])
        kv_memory_bytes = kv_token_capacity * _vllm_kv_bytes_per_token(
            config, self.dtype
        )
        response_by_logical: dict[str, dict[str, Any]] = {}

        with _vllm_oracle_environment(sidecar, self.torch_num_threads):
            if "vllm" in sys.modules:
                raise RuntimeError(
                    "vLLM was imported before CPU oracle environment selection"
                )
            vllm = importlib.import_module("vllm")
            torch = importlib.import_module("torch")
            framework_version = importlib.metadata.version("vllm")
            from simllm.adapters.vllm.oracle import (
                mark_oracle_capture_start,
                mark_oracle_request_mapping,
                mark_oracle_submission_group,
            )

            llm = None
            try:
                llm = vllm.LLM(
                    model=str(self.model_path),
                    tokenizer=str(self.model_path),
                    dtype=self.dtype,
                    seed=self.engine_seed,
                    enforce_eager=True,
                    enable_prefix_caching=True,
                    enable_return_routed_experts=True,
                    kv_cache_memory_bytes=kv_memory_bytes,
                    block_size=block_size,
                    max_model_len=context_length,
                    max_num_batched_tokens=context_length,
                    max_num_seqs=max_running_requests,
                )
                mark_oracle_capture_start(request_ids)
                for group_index, group in enumerate(normalized_groups):
                    logical_ids = tuple(request.request_id for request in group)
                    mark_oracle_submission_group(group_index, logical_ids)
                    internal_ids = llm.enqueue(
                        [
                            {"prompt_token_ids": list(request.input_token_ids)}
                            for request in group
                        ],
                        [
                            vllm.SamplingParams(
                                temperature=0,
                                max_tokens=request.max_new_tokens,
                                stop=list(request.stop_strings) or None,
                                include_stop_str_in_output=True,
                            )
                            for request in group
                        ],
                        use_tqdm=False,
                    )
                    internal_to_logical = dict(
                        zip(internal_ids, logical_ids, strict=True)
                    )
                    mark_oracle_request_mapping(group_index, internal_to_logical)
                    external_to_logical = {
                        internal_id.rsplit("-", 1)[0]: logical_id
                        for internal_id, logical_id in internal_to_logical.items()
                    }
                    if len(external_to_logical) != len(internal_to_logical):
                        raise RuntimeError("vLLM external request IDs are not unique")
                    for response in llm.wait_for_completion(use_tqdm=False):
                        logical_id = external_to_logical.get(response.request_id)
                        if logical_id is None:
                            raise RuntimeError(
                                f"vLLM returned unknown request {response.request_id!r}"
                            )
                        if logical_id in response_by_logical:
                            raise RuntimeError(
                                f"duplicate vLLM response {logical_id!r}"
                            )
                        if len(response.outputs) != 1:
                            raise RuntimeError(
                                f"vLLM request {logical_id!r} returned "
                                f"{len(response.outputs)} completions"
                            )
                        completion = response.outputs[0]
                        routed = completion.routed_experts
                        if routed is None:
                            raise RuntimeError("vLLM returned no routed experts")
                        response_by_logical[logical_id] = {
                            "finish_reason": completion.finish_reason,
                            "external_request_id": response.request_id,
                            "internal_request_id": next(
                                internal
                                for internal, logical in internal_to_logical.items()
                                if logical == logical_id
                            ),
                            "num_cached_tokens": int(response.num_cached_tokens or 0),
                            "output_text": completion.text,
                            "output_token_ids": [
                                int(value) for value in completion.token_ids
                            ],
                            "request_id": logical_id,
                            "routed_experts": [
                                int(value) for value in routed.reshape(-1).tolist()
                            ],
                            "routed_experts_dtype": str(routed.dtype),
                            "routed_experts_shape": list(routed.shape),
                            "stop_reason": completion.stop_reason,
                        }
            finally:
                if llm is not None:
                    llm.llm_engine.engine_core.shutdown()

        if set(response_by_logical) != set(request_ids):
            raise RuntimeError("vLLM responses do not cover every submitted request")
        raw_responses.parent.mkdir(parents=True, exist_ok=True)
        raw_responses.write_text(
            json.dumps(
                {
                    "framework": "vllm",
                    "framework_version": framework_version,
                    "requests": [response_by_logical[request_id] for request_id in request_ids],
                },
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )

        sidecar_rows = _read_observation_sidecar(
            sidecar,
            framework="vLLM",
            schema=_VLLM_ORACLE_SCHEMA,
        )
        kv_events, actual_block_size, actual_capacity, model_class, preemptions = (
            _vllm_sidecar_projection(sidecar_rows, set(request_ids))
        )
        if actual_block_size != block_size:
            raise RuntimeError(
                f"vLLM block size differs from the requested cell: "
                f"{actual_block_size} != {block_size}"
            )
        if actual_capacity != kv_token_capacity:
            raise RuntimeError(
                f"vLLM token capacity differs from the requested cell: "
                f"{actual_capacity} != {kv_token_capacity}"
            )
        provenance = FrameworkTraceProvenance(
            model_id=self.model_id,
            model_revision=self.model_revision,
            model_class=model_class,
            dtype=self.dtype,
            tokenizer_sha256=self.tokenizer_sha256,
            sampling=SamplingConfig.greedy(),
            capture_host=self.capture_host,
            runner="vllm-cpu",
            framework="vllm",
            framework_version=framework_version,
            observed_source=self.observed_source,
            authored_against_source=self.authored_against_source,
            torch_version=str(torch.__version__),
            device="cpu",
            torch_num_threads=self.torch_num_threads,
            engine_seed=self.engine_seed,
            eos_token_id=eos_token_id,
            top_k=top_k,
            expert_count=expert_count,
            moe_layer_indices=tuple(range(layer_count)),
            kv_page_size=actual_block_size,
            kv_token_capacity=actual_capacity,
            dispatch_layer_mapping="framework-layer-id",
        )
        request_rows = tuple(
            _vllm_request_from_response(
                response_by_logical[request.request_id],
                request,
                provenance,
                preemption_count=preemptions[request.request_id],
            )
            for request in requests
        )
        trace = FrameworkPreplayTrace(
            provenance=provenance,
            requests=request_rows,
            kv_events=kv_events,
        )
        return write_framework_preplay_trace(output, trace, overwrite=overwrite)

    def capture(
        self,
        requests: Iterable[FrameworkOracleRequest],
        path: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Capture one concurrent vLLM submission group."""

        return self.capture_groups((tuple(requests),), path, **kwargs)
