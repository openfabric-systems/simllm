#!/usr/bin/env python3
"""Capture the frozen four-layer vLLM decode graph on one GPU."""

from __future__ import annotations

import argparse
import functools
import importlib.metadata
import json
import math
import os
import platform
import statistics
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "1")

import nvtx
import torch
from vllm import LLM, SamplingParams
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
    override_forward_context,
)
from vllm.inputs import TokensPrompt

CAPTURE: dict[str, Any] = {}
VOCAB_FLOOR = 100
VOCAB_SPAN = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--expectations", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--active-file", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def tensor_shapes(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (tuple, list)):
        return [tensor_shapes(item) for item in value]
    if isinstance(value, dict):
        return {str(key): tensor_shapes(item) for key, item in value.items()}
    return type(value).__name__


def elapsed_ps(start: torch.cuda.Event, end: torch.cuda.Event) -> int:
    return round(start.elapsed_time(end) * 1_000_000_000)


def prompts(input_lengths: list[int]) -> list[TokensPrompt]:
    rows = []
    for batch_index, input_len in enumerate(input_lengths):
        token_ids = [
            VOCAB_FLOOR + ((batch_index * 37 + index) % VOCAB_SPAN)
            for index in range(input_len)
        ]
        rows.append(TokensPrompt(prompt_token_ids=token_ids))
    return rows


def install_root_capture(model: torch.nn.Module, *, marker: str) -> dict[str, Any]:
    marker_path = Path(marker)
    original = model.forward

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any):
        output = original(*args, **kwargs)
        if marker_path.exists() and not CAPTURE:
            forward_context = (
                get_forward_context() if is_forward_context_available() else None
            )
            CAPTURE.update(
                {
                    "original": original,
                    "args": args,
                    "kwargs": kwargs,
                    "forward_context": forward_context,
                    "moe_layer_index": (
                        forward_context.moe_layer_index
                        if forward_context is not None
                        else None
                    ),
                    "output_shape": tensor_shapes(output),
                }
            )
        return output

    model.forward = wrapped
    return {
        "target": "DeepseekV3ForCausalLM.forward",
        "root_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "layer_mlp_types": [type(layer.mlp).__name__ for layer in model.model.layers],
    }


def event_overhead_ps(trials: int, stream: torch.cuda.Stream) -> int:
    samples = []
    with torch.cuda.stream(stream):
        for _ in range(trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            end.record(stream)
            end.synchronize()
            samples.append(elapsed_ps(start, end))
    return round(statistics.median(samples))


def run_graph(
    model: torch.nn.Module,
    *,
    repeats: tuple[int, ...],
    warmup: int,
    trials: int,
    flush_mib: int,
) -> dict[str, Any]:
    del model
    if not CAPTURE:
        raise RuntimeError("root model call was not captured at the exact boundary")
    original = CAPTURE["original"]
    args = CAPTURE["args"]
    kwargs = CAPTURE["kwargs"]
    forward_context = CAPTURE["forward_context"]
    captured_moe_layer_index = CAPTURE["moe_layer_index"]

    def replay() -> None:
        if forward_context is None:
            original(*args, **kwargs)
            return
        forward_context.moe_layer_index = captured_moe_layer_index
        with override_forward_context(forward_context):
            original(*args, **kwargs)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.inference_mode():
        for _ in range(warmup):
            replay()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), torch.inference_mode():
        replay()
    stream.synchronize()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            graph.replay()
    stream.synchronize()

    flush_buffer = torch.empty(
        flush_mib * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda",
    )
    setup_finish_ps = event_overhead_ps(trials, stream)
    repeat_rows = []
    for repeat in repeats:
        raw_samples = []
        for trial in range(trials):
            flush_buffer.zero_()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            label = f"core66-kladder|repeat={repeat}|trial={trial}"
            with torch.cuda.stream(stream):
                start.record(stream)
                with nvtx.annotate(label, domain="simllm-core66-kladder"):
                    for call_index in range(repeat):
                        with nvtx.annotate(
                            f"{label}|call={call_index}",
                            domain="simllm-core66-kladder",
                        ):
                            graph.replay()
                end.record(stream)
            end.synchronize()
            raw_samples.append(elapsed_ps(start, end))
        raw_median_ps = round(statistics.median(raw_samples))
        subtracted_total_ps = raw_median_ps - setup_finish_ps
        if subtracted_total_ps <= 0:
            raise RuntimeError("event subtraction removed all graph service")
        repeat_rows.append(
            {
                "repeat": repeat,
                "raw_samples_ps": raw_samples,
                "raw_median_ps": raw_median_ps,
                "subtracted_total_ps": subtracted_total_ps,
                "subtracted_service_ps": round(subtracted_total_ps / repeat),
                "nvtx_label_prefix": f"core66-kladder|repeat={repeat}",
            }
        )
    return {
        "stream_id": stream.cuda_stream,
        "setup_finish_ps": setup_finish_ps,
        "repeat_rows": repeat_rows,
        "subtraction_method": (
            "median device-event interval for N graph replays minus the median "
            "empty device-event interval, divided by N"
        ),
        "capture_input_shapes": tensor_shapes(args),
        "capture_keyword_shapes": tensor_shapes(kwargs),
        "capture_output_shape": CAPTURE["output_shape"],
    }


def main() -> int:
    args = parse_args()
    expectations = json.loads(args.expectations.read_text(encoding="utf-8"))
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    frozen_shape = expectations["shape"]
    if alignment["batch_size"] != frozen_shape["batch_size"]:
        raise RuntimeError("alignment batch does not match the freeze")
    if alignment["remote_kv_tokens"] != frozen_shape["kv_tokens_per_request"]:
        raise RuntimeError("alignment KV length does not match the freeze")
    input_lengths = [int(value) for value in alignment["prompt_lengths"]]
    if len(input_lengths) != frozen_shape["batch_size"]:
        raise RuntimeError("alignment does not have the frozen request count")

    capture_freeze = expectations["confirmatory_capture"]
    measurement_freeze = capture_freeze["measurement"]
    repeats = tuple(int(value) for value in measurement_freeze["repeats"])
    trials = int(measurement_freeze["trials"])
    warmup = int(measurement_freeze["warmup_replays"])
    flush_mib = int(measurement_freeze["cache_flush_mib_before_each_trial"])

    llm = LLM(
        model=args.model,
        revision=args.revision,
        load_format="dummy",
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=True,
        skip_tokenizer_init=True,
        enable_prefix_caching=False,
        max_model_len=8192,
        max_num_seqs=32,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.88,
        seed=17,
        disable_log_stats=True,
        hf_overrides={"num_hidden_layers": 4, "first_k_dense_replace": 3},
    )
    installed = llm.llm_engine.apply_model(
        functools.partial(install_root_capture, marker=str(args.marker))
    )
    if len(installed) != 1:
        raise RuntimeError("root capture hook was not installed on one model")

    write_json(
        args.active_file,
        {
            "enabled": True,
            "batch_size": frozen_shape["batch_size"],
            "remote_kv_tokens": frozen_shape["kv_tokens_per_request"],
        },
    )
    sampling = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=int(alignment["max_tokens"]),
        detokenize=False,
    )
    request_ids = []
    for index, prompt in enumerate(prompts(input_lengths)):
        request_id = f"core66-kladder-{index:02d}"
        llm.llm_engine.add_request(request_id, prompt, sampling)
        request_ids.append(request_id)

    max_steps = math.ceil(sum(input_lengths) / 4096) + int(alignment["max_tokens"]) + 128
    for _ in range(max_steps):
        llm.llm_engine.step()
        if args.marker.exists():
            llm.llm_engine.abort_request(request_ids)
            break
        if not llm.llm_engine.has_unfinished_requests():
            break
    else:
        raise RuntimeError("exact decode boundary was not reached")
    if not args.marker.exists():
        raise RuntimeError("requests ended before exact decode boundary")

    measured = llm.llm_engine.apply_model(
        functools.partial(
            run_graph,
            repeats=repeats,
            warmup=warmup,
            trials=trials,
            flush_mib=flush_mib,
        )
    )
    if len(measured) != 1:
        raise RuntimeError("graph was not measured on one model")

    layer_types = installed[0]["layer_mlp_types"]
    dense_layers = sum("MoE" not in value for value in layer_types)
    moe_layers = sum("MoE" in value for value in layer_types)
    write_json(
        args.output,
        {
            "schema": "simllm-deployment-curve-core66-decode-capture-v1",
            "evidence": "confirmatory, dummy weights",
            "model": args.model,
            "revision": args.revision,
            "framework": {
                "name": "vllm",
                "version": importlib.metadata.version("vllm"),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "python": platform.python_version(),
                "machine": platform.machine(),
            },
            "shape": {
                "batch_size": frozen_shape["batch_size"],
                "kv_tokens_per_request": frozen_shape["kv_tokens_per_request"],
                "layers": 4,
                "dense_layers": dense_layers,
                "moe_layers": moe_layers,
                "first_k_dense_replace": 3,
            },
            "installed": installed[0],
            "marker": json.loads(args.marker.read_text(encoding="utf-8")),
            "prediction": {
                "graph_service_ps": capture_freeze["composition"][
                    "predicted_graph_service_ps"
                ],
                "tolerance_ps": capture_freeze["scored_relations"][
                    "composition_residual_absolute_ps_at_most"
                ],
                "physical_floor_ps": capture_freeze["scored_relations"][
                    "physical_service_floor_ps"
                ],
                "physical_ceiling_ps": capture_freeze["scored_relations"][
                    "physical_service_ceiling_ps"
                ],
            },
            "measurement": measured[0],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
