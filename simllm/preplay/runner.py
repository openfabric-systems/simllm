"""Lazy Transformers CPU inference runner for pre-play traces.

Importing this module does not import Torch or Transformers. Constructing a
``TransformersCpuRunner`` is the execution boundary that loads the optional
runtime and the model.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simllm.core._wire import _integer, _string, _validate_unique
from simllm.preplay.schema import (
    LayerRouting,
    PromptFormat,
    RequestTrace,
    SamplingConfig,
    SamplingMode,
    StopReason,
    TokenTrace,
    TraceProvenance,
    validate_sampling_config,
)
from simllm.preplay.trace import PreplayTraceWriter

_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


@dataclass(frozen=True, kw_only=True)
class PreplayRequest:
    """One prompt and its request-local stopping policy."""

    request_id: str
    prompt: str
    max_new_tokens: int
    stop_strings: tuple[str, ...] = ()
    prompt_format: PromptFormat = PromptFormat.CHAT

    def __post_init__(self) -> None:
        object.__setattr__(self, "stop_strings", tuple(self.stop_strings))
        _string(self.request_id, "request.request_id")
        _string(self.prompt, "request.prompt")
        _integer(self.max_new_tokens, "request.max_new_tokens", minimum=1)
        for index, stop_string in enumerate(self.stop_strings):
            _string(stop_string, f"request.stop_strings[{index}]")
        _validate_unique(self.stop_strings, "request.stop_strings")
        if not isinstance(self.prompt_format, PromptFormat):
            raise TypeError("request.prompt_format: expected PromptFormat")


@dataclass(frozen=True)
class _Router:
    layer_index: int
    name: str
    module: Any
    top_k: int
    expert_count: int


def _load_optional_runtime() -> tuple[Any, Any]:
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Transformers CPU pre-play requires optional dependencies; "
            "install the project with the 'preplay' extra"
        ) from exc
    return torch, transformers


def _tokenizer_sha256(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "to_str"):
        tokenizer_state: Any = backend.to_str()
    else:
        tokenizer_state = {
            "vocabulary": sorted(tokenizer.get_vocab().items()),
            "added_vocabulary": sorted(tokenizer.get_added_vocab().items()),
            "special_tokens": {
                key: str(value)
                for key, value in sorted(tokenizer.special_tokens_map.items())
            },
        }
    canonical = json.dumps(
        {
            "tokenizer_class": type(tokenizer).__name__,
            "state": tokenizer_state,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _discover_routers(model: Any) -> tuple[_Router, ...]:
    routers: list[_Router] = []
    for name, module in model.named_modules():
        match = _LAYER_INDEX_RE.search(name)
        top_k = getattr(module, "top_k", None)
        expert_count = getattr(module, "num_experts", None)
        if not name.endswith(".router") or match is None:
            continue
        if type(top_k) is not int or type(expert_count) is not int:
            continue
        routers.append(
            _Router(
                layer_index=int(match.group(1)),
                name=name,
                module=module,
                top_k=top_k,
                expert_count=expert_count,
            )
        )
    routers.sort(key=lambda router: router.layer_index)
    if not routers:
        raise RuntimeError(
            "model exposes no supported MoE routers; expected layer router modules "
            "with top_k, num_experts, and router-logit outputs"
        )
    layer_indices = tuple(router.layer_index for router in routers)
    if len(layer_indices) != len(set(layer_indices)):
        raise RuntimeError(f"model exposes duplicate MoE layer routers: {layer_indices}")
    dimensions = {(router.top_k, router.expert_count) for router in routers}
    if len(dimensions) != 1:
        raise RuntimeError(f"trace v1 requires uniform router dimensions, found {dimensions}")
    return tuple(routers)


def _classify_stop_reason(
    output_token_ids: tuple[int, ...],
    output_text: str,
    *,
    eos_token_id: int,
    max_new_tokens: int,
    stop_strings: tuple[str, ...],
) -> tuple[StopReason, str | None]:
    if not output_token_ids:
        raise RuntimeError("model produced no output tokens")
    if output_token_ids[-1] == eos_token_id:
        return StopReason.EOS, None
    for stop_string in stop_strings:
        if stop_string in output_text:
            return StopReason.STOP_STRING, stop_string
    if len(output_token_ids) == max_new_tokens:
        return StopReason.LENGTH_CAP, None
    raise RuntimeError(
        "generation stopped without EOS, a configured stop string, or the length cap"
    )


class TransformersCpuRunner:
    """Load one pinned model on CPU and stream one or more trace captures."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        cache_dir: str | Path,
        dtype: str = "float32",
        torch_num_threads: int | None = None,
        capture_host: str | None = None,
    ):
        _string(model_id, "model_id")
        _string(revision, "revision")
        self.model_id = model_id
        self.revision = revision
        self.cache_dir = Path(cache_dir)
        self._torch, self._transformers = _load_optional_runtime()

        dtype_by_name = {
            "float32": self._torch.float32,
            "bfloat16": self._torch.bfloat16,
        }
        if dtype not in dtype_by_name:
            choices = ", ".join(sorted(dtype_by_name))
            raise ValueError(f"dtype: expected one of {choices}")
        if torch_num_threads is not None:
            _integer(torch_num_threads, "torch_num_threads", minimum=1)
            self._torch.set_num_threads(torch_num_threads)
        self.torch_num_threads = int(self._torch.get_num_threads())
        self.capture_host = capture_host or platform.node() or "unknown-host"

        auto_tokenizer = self._transformers.AutoTokenizer
        auto_model = self._transformers.AutoModelForCausalLM
        load_kwargs = {
            "revision": revision,
            "cache_dir": str(self.cache_dir),
            "local_files_only": True,
            "trust_remote_code": False,
        }
        self.tokenizer = auto_tokenizer.from_pretrained(model_id, **load_kwargs)
        self.model = auto_model.from_pretrained(
            model_id,
            torch_dtype=dtype_by_name[dtype],
            **load_kwargs,
        )
        self.model.to("cpu")
        self.model.eval()
        parameter = next(self.model.parameters())
        if parameter.device.type != "cpu":
            raise RuntimeError(f"pre-play model loaded on unexpected device {parameter.device}")
        self.dtype = str(parameter.dtype).removeprefix("torch.")
        if self.dtype != dtype:
            raise RuntimeError(f"requested dtype {dtype!r}, model loaded as {self.dtype!r}")

        eos_token_id = self.tokenizer.eos_token_id
        if type(eos_token_id) is not int:
            eos_token_id = getattr(self.model.config, "eos_token_id", None)
        if type(eos_token_id) is not int or eos_token_id < 0:
            raise RuntimeError("model must expose one nonnegative EOS token ID")
        self.eos_token_id = eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        if type(self.pad_token_id) is not int:
            self.pad_token_id = self.eos_token_id
        self.tokenizer_sha256 = _tokenizer_sha256(self.tokenizer)
        self.routers = _discover_routers(self.model)
        self.top_k = self.routers[0].top_k
        self.expert_count = self.routers[0].expert_count
        self.moe_layer_indices = tuple(router.layer_index for router in self.routers)

    def _encode_request(self, request: PreplayRequest) -> Any:
        if request.prompt_format is PromptFormat.CHAT:
            encoded = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": request.prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            input_ids = getattr(encoded, "input_ids", encoded)
        else:
            encoded = self.tokenizer(request.prompt, return_tensors="pt")
            input_ids = encoded["input_ids"]
        if getattr(input_ids, "ndim", None) != 2 or input_ids.shape[0] != 1:
            raise RuntimeError(
                f"request {request.request_id!r} did not tokenize to one rank-two input"
            )
        if input_ids.shape[1] == 0:
            raise RuntimeError(f"request {request.request_id!r} tokenized to an empty prompt")
        return input_ids.to(device="cpu", dtype=self._torch.long)

    def _forward_with_routing(
        self,
        input_ids: Any,
        *,
        past_key_values: Any,
    ) -> tuple[Any, tuple[LayerRouting, ...]]:
        captures: dict[int, LayerRouting] = {}
        handles: list[Any] = []

        def make_hook(router: _Router):
            def capture(module: Any, inputs: Any, output: Any) -> None:
                del module, inputs
                if router.layer_index in captures:
                    raise RuntimeError(
                        f"router {router.name!r} executed more than once in one model forward"
                    )
                if not isinstance(output, (tuple, list)) or not output:
                    raise RuntimeError(
                        f"router {router.name!r} did not return router logits"
                    )
                router_logits = output[-1]
                shape = getattr(router_logits, "shape", ())
                if len(shape) != 2 or shape[-1] != router.expert_count:
                    raise RuntimeError(
                        f"router {router.name!r} returned logits with shape {tuple(shape)}"
                    )
                if shape[0] != input_ids.numel():
                    raise RuntimeError(
                        f"router {router.name!r} returned {shape[0]} token rows, "
                        f"expected {input_ids.numel()}"
                    )
                top_logits, expert_ids = self._torch.topk(
                    router_logits.float(),
                    router.top_k,
                    dim=-1,
                )
                gate_weights = self._torch.softmax(top_logits, dim=-1)
                token_row = shape[0] - 1
                captures[router.layer_index] = LayerRouting(
                    layer_index=router.layer_index,
                    expert_ids=tuple(int(value) for value in expert_ids[token_row].tolist()),
                    gate_weights=tuple(
                        float(value) for value in gate_weights[token_row].tolist()
                    ),
                )

            return capture

        for router in self.routers:
            handles.append(router.module.register_forward_hook(make_hook(router)))
        try:
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        if tuple(sorted(captures)) != self.moe_layer_indices:
            raise RuntimeError(
                "router hooks did not capture every configured MoE layer; "
                f"captured {tuple(sorted(captures))}, expected {self.moe_layer_indices}"
            )
        return outputs, tuple(captures[layer] for layer in self.moe_layer_indices)

    def _select_token(self, logits: Any, sampling: SamplingConfig, generator: Any) -> int:
        if sampling.mode is SamplingMode.GREEDY:
            return int(self._torch.argmax(logits, dim=-1).item())

        scaled = logits.float() / float(sampling.temperature)
        probabilities = self._torch.softmax(scaled, dim=-1)
        if float(sampling.top_p) < 1.0:
            sorted_probabilities, sorted_indices = self._torch.sort(
                probabilities,
                descending=True,
            )
            cumulative = self._torch.cumsum(sorted_probabilities, dim=-1)
            remove = cumulative - sorted_probabilities >= float(sampling.top_p)
            sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
            probabilities = self._torch.zeros_like(probabilities).scatter(
                -1,
                sorted_indices,
                sorted_probabilities,
            )
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        return int(self._torch.multinomial(probabilities, 1, generator=generator).item())

    def _run_request(
        self,
        request: PreplayRequest,
        sampling: SamplingConfig,
        generator: Any,
    ) -> RequestTrace:
        input_ids = self._encode_request(request)
        prompt_token_ids = tuple(int(value) for value in input_ids[0].tolist())
        prompt_sha256 = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()
        output_token_ids: list[int] = []
        token_traces: list[TokenTrace] = []

        with self._torch.inference_mode():
            outputs, _ = self._forward_with_routing(input_ids, past_key_values=None)
            past_key_values = outputs.past_key_values
            while True:
                next_token = self._select_token(outputs.logits[:, -1, :], sampling, generator)
                output_token_ids.append(next_token)
                output_text = self.tokenizer.decode(
                    output_token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                stop_reason, matched_stop_string = _classify_stop_reason(
                    tuple(output_token_ids),
                    output_text,
                    eos_token_id=self.eos_token_id,
                    max_new_tokens=request.max_new_tokens,
                    stop_strings=request.stop_strings,
                ) if (
                    next_token == self.eos_token_id
                    or len(output_token_ids) == request.max_new_tokens
                    or any(value in output_text for value in request.stop_strings)
                ) else (None, None)

                token_input = self._torch.tensor([[next_token]], dtype=self._torch.long)
                outputs, routing = self._forward_with_routing(
                    token_input,
                    past_key_values=past_key_values,
                )
                token_traces.append(
                    TokenTrace(
                        token_index=len(token_traces),
                        token_id=next_token,
                        routing=routing,
                    )
                )
                past_key_values = outputs.past_key_values
                if stop_reason is not None:
                    break

        return RequestTrace(
            request_id=request.request_id,
            prompt_sha256=prompt_sha256,
            prompt_format=request.prompt_format,
            input_token_ids=prompt_token_ids,
            max_new_tokens=request.max_new_tokens,
            stop_strings=request.stop_strings,
            output_text=output_text,
            stop_reason=stop_reason,
            matched_stop_string=matched_stop_string,
            tokens=tuple(token_traces),
        )

    def capture(
        self,
        requests: Iterable[PreplayRequest],
        output_path: str | Path,
        *,
        sampling: SamplingConfig,
    ) -> Path:
        """Run requests in order and flush each completed trace to disk."""

        validate_sampling_config(sampling)
        provenance = TraceProvenance(
            model_id=self.model_id,
            model_revision=self.revision,
            model_class=type(self.model).__name__,
            dtype=self.dtype,
            tokenizer_sha256=self.tokenizer_sha256,
            sampling=sampling,
            capture_host=self.capture_host,
            runner="transformers-cpu",
            transformers_version=str(self._transformers.__version__),
            torch_version=str(self._torch.__version__),
            device="cpu",
            torch_num_threads=self.torch_num_threads,
            eos_token_id=self.eos_token_id,
            top_k=self.top_k,
            expert_count=self.expert_count,
            moe_layer_indices=self.moe_layer_indices,
        )
        generator = None
        if sampling.mode is SamplingMode.SEEDED_SAMPLING:
            generator = self._torch.Generator(device="cpu")
            generator.manual_seed(sampling.seed)

        deterministic_before = self._torch.are_deterministic_algorithms_enabled()
        self._torch.use_deterministic_algorithms(True)
        try:
            with PreplayTraceWriter(output_path, provenance) as writer:
                for request in requests:
                    if not isinstance(request, PreplayRequest):
                        raise TypeError("requests must yield PreplayRequest values")
                    writer.append(self._run_request(request, sampling, generator))
        finally:
            self._torch.use_deterministic_algorithms(deterministic_before)
        return Path(output_path)


def run_transformers_preplay(
    requests: Iterable[PreplayRequest],
    output_path: str | Path,
    *,
    model_id: str,
    revision: str,
    cache_dir: str | Path,
    sampling: SamplingConfig,
    dtype: str = "float32",
    torch_num_threads: int | None = None,
    capture_host: str | None = None,
) -> Path:
    """Load a pinned CPU model and write one complete pre-play trace."""

    runner = TransformersCpuRunner(
        model_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        dtype=dtype,
        torch_num_threads=torch_num_threads,
        capture_host=capture_host,
    )
    return runner.capture(requests, output_path, sampling=sampling)
