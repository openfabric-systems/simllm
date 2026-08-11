"""Run the frozen PLAY-5 independent-oracle and routed-replay study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

MODEL_ID = "ibm-granite/granite-3.0-1b-a400m-instruct"
MODEL_REVISION = "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
MODEL_RELATIVE_PATH = (
    Path("hub")
    / "models--ibm-granite--granite-3.0-1b-a400m-instruct"
    / "snapshots"
    / MODEL_REVISION
)
SOURCE_HASHES = {
    "platforms/__init__.py": "a2bd800acc39b3215ccb78808d43317b351f137072b03e7f0f0ab3d069d91521",
    "platforms/cpu.py": "067f92d391b1c131e12a7ba9631921e4b9dd57d3c55b1d8724e9963e2fdc9c7d",
    "v1/worker/cpu_worker.py": "ccf18240a7605ebda0dbf27bdbda83a39e83e7baece95bff68e0f3e5beb6103e",
    "v1/worker/cpu_model_runner.py": "dd3a7686b567c52363454ed6b353bf8a968fa60c6ad3dbcf30c8044f5602d7ed",
    "model_executor/models/granitemoe.py": "b60e452c3f28b25aa104c88869daa25c06a7fb6ed45bd34e908fa6a8395efda1",
    "v1/sample/sampler.py": "315af950ef4c35fced53dc3a5df49a80af20b47e417e8f12bf315f535769bab2",
    "v1/sample/ops/topk_topp_sampler.py": "ad9406a08a9bfcc84f182dab4522920f73605d4999191ee8f0dbb1479d946506",
    "v1/engine/output_processor.py": "ee10351275d90796c8b901a5f4b23d5a046ef6ee72fd2921aff2ae78ca58bd9b",
    "v1/core/sched/scheduler.py": "2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941",
}
NEAR_TIE_ABS_LOGIT = 1e-5
BANDWIDTHS_BPS = (200_000_000_000, 400_000_000_000)
EXPECTED_SCORED = 19
EXPECTED_REPLAY_SCORED = 13
FREEZE_COMMIT = "bc5eb9e"
VECTOR_BYTES = 2_048
REQUEST_SPECS = (
    {
        "request_id": "eos-brief",
        "prompt": "Reply with exactly one word: OK",
        "max_new_tokens": 16,
        "stop_strings": (),
    },
    {
        "request_id": "length-cap",
        "prompt": "Continue this sequence with ten more integers: 1 2 3",
        "max_new_tokens": 1,
        "stop_strings": (),
    },
    {
        "request_id": "stop-string",
        "prompt": "Reply with exactly SIMLLM_STOP and no other text",
        "max_new_tokens": 16,
        "stop_strings": ("SIMLLM_STOP",),
    },
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_only(args: argparse.Namespace) -> None:
    model = args.cache_dir / MODEL_RELATIVE_PATH
    if not model.is_dir() or not (model / "config.json").is_file():
        raise SystemExit(f"pinned model snapshot is missing: {model}")
    if importlib.metadata.version("vllm") != "0.26.0":
        raise SystemExit("PLAY-5 requires vLLM 0.26.0")
    for relative, expected in SOURCE_HASHES.items():
        source = args.vllm_package_root / relative
        if not source.is_file() or file_sha256(source) != expected:
            raise SystemExit(f"pinned external source changed: {relative}")
    if not args.htsim_rnic.is_file() or not args.htsim_rnic.stat().st_mode & 0o111:
        raise SystemExit(f"htsim_rnic is missing or not executable: {args.htsim_rnic}")
    if NEAR_TIE_ABS_LOGIT <= 0:
        raise AssertionError("near-tie margin must be positive")
    if BANDWIDTHS_BPS[1] != 2 * BANDWIDTHS_BPS[0]:
        raise AssertionError("bandwidth pair must retain the frozen 2x relation")
    if 8_000_000_000_000 // BANDWIDTHS_BPS[0] != 40:
        raise AssertionError("200 Gbit/s byte serialization literal changed")
    if 8_000_000_000_000 // BANDWIDTHS_BPS[1] != 20:
        raise AssertionError("400 Gbit/s byte serialization literal changed")
    if EXPECTED_SCORED != 6 + 6 + 2 + 3 + 2:
        raise AssertionError("full evidence denominator changed")
    if EXPECTED_REPLAY_SCORED != 6 + 2 + 3 + 2:
        raise AssertionError("replay evidence denominator changed")
    print(
        f"check-only run-dir={args.run_dir}; validated frozen PLAY-5 inputs "
        "and produced no artifacts"
    )


@contextmanager
def offline_environment(*, replay: bool = False) -> Iterator[None]:
    values = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_V1": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
    }
    if replay:
        values.update(
            {
                "SIMLLM_VLLM_WORKER_MODE": "skeleton",
                "SIMLLM_VLLM_MODE": "virtual",
            }
        )
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


def _boundary_json(
    selected_ids: list[int], boundary_ids: list[int], margin: float
) -> dict[str, Any]:
    return {
        "selected_ids": selected_ids,
        "boundary_ids": boundary_ids,
        "margin": margin,
    }


def _transformers_capture(args: argparse.Namespace) -> None:
    import torch

    from simllm.preplay import (
        PreplayRequest,
        SamplingConfig,
        TransformersCpuRunner,
        read_preplay_trace,
    )

    output_dir = args.run_dir / "transformers"
    output_dir.mkdir(parents=True, exist_ok=False)
    requests = tuple(
        PreplayRequest(
            request_id=spec["request_id"],
            prompt=spec["prompt"],
            max_new_tokens=spec["max_new_tokens"],
            stop_strings=spec["stop_strings"],
        )
        for spec in REQUEST_SPECS
    )
    with offline_environment():
        runner = TransformersCpuRunner(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=args.cache_dir,
            dtype="float32",
            torch_num_threads=8,
            capture_host="play5-validation-host",
        )
        observations: dict[str, list[dict[str, Any]]] = {}
        for label, sampling in (
            ("greedy", SamplingConfig.greedy()),
            (
                "seeded-sampling",
                SamplingConfig.seeded(173, temperature=0.8, top_p=0.9),
            ),
        ):
            forward_groups: list[list[list[dict[str, Any]]]] = []
            token_rows: list[dict[str, Any]] = []
            original_forward = runner._forward_with_routing
            original_select = runner._select_token

            def capture_forward(
                input_ids: Any,
                *,
                past_key_values: Any,
                _original_forward: Any = original_forward,
                _forward_groups: list[list[list[dict[str, Any]]]] = forward_groups,
            ) -> tuple[Any, Any]:
                captures: dict[int, list[dict[str, Any]]] = {}
                handles = []
                for router in runner.routers:

                    def hook(
                        _module: Any,
                        _inputs: Any,
                        output: Any,
                        *,
                        layer: int = router.layer_index,
                    ) -> None:
                        logits = output[-1]
                        values, indices = torch.topk(logits.detach().float(), 9, dim=-1)
                        captures[layer] = [
                            _boundary_json(
                                [int(value) for value in row_indices[:8]],
                                [int(value) for value in row_indices[7:9]],
                                float(row_values[7] - row_values[8]),
                            )
                            for row_values, row_indices in zip(
                                values.cpu().tolist(),
                                indices.cpu().tolist(),
                                strict=True,
                            )
                        ]

                    handles.append(router.module.register_forward_hook(hook))
                try:
                    result = _original_forward(
                        input_ids,
                        past_key_values=past_key_values,
                    )
                finally:
                    for handle in handles:
                        handle.remove()
                if tuple(sorted(captures)) != tuple(range(24)):
                    raise AssertionError("Transformers diagnostics missed a router")
                _forward_groups.append(
                    [
                        [captures[layer][token] for layer in range(24)]
                        for token in range(input_ids.numel())
                    ]
                )
                return result

            def capture_select(
                logits: Any,
                config: Any,
                generator: Any,
                _original_select: Any = original_select,
                _token_rows: list[dict[str, Any]] = token_rows,
            ) -> int:
                values, indices = torch.topk(logits.detach().float(), 2, dim=-1)
                selected = _original_select(logits, config, generator)
                _token_rows.append(
                    _boundary_json(
                        [selected],
                        [int(value) for value in indices[0].cpu().tolist()],
                        float(values[0, 0] - values[0, 1]),
                    )
                )
                return selected

            runner._forward_with_routing = capture_forward
            runner._select_token = capture_select
            trace_path = output_dir / f"{label}.jsonl"
            try:
                runner.capture(requests, trace_path, sampling=sampling)
            finally:
                runner._forward_with_routing = original_forward
                runner._select_token = original_select

            trace = read_preplay_trace(trace_path)
            group_cursor = 0
            token_cursor = 0
            mode_rows = []
            for request in trace.requests:
                groups_needed = 1 + len(request.decode_tokens)
                groups = forward_groups[group_cursor : group_cursor + groups_needed]
                boundaries = token_rows[token_cursor : token_cursor + len(request.output_token_ids)]
                group_cursor += groups_needed
                token_cursor += len(request.output_token_ids)
                if len(groups) != groups_needed or len(boundaries) != len(request.output_token_ids):
                    raise AssertionError("Transformers diagnostic cardinality changed")
                routed = []
                for token, rows in zip(request.prefill_tokens, groups[0], strict=True):
                    for layer, boundary in enumerate(rows):
                        routed.append(
                            {
                                "phase": "prefill",
                                "token_index": token.token_index,
                                "token_id": token.token_id,
                                "layer_index": layer,
                                "boundary": boundary,
                            }
                        )
                for token, group in zip(request.decode_tokens, groups[1:], strict=True):
                    if len(group) != 1:
                        raise AssertionError("decode diagnostic was not one token")
                    for layer, boundary in enumerate(group[0]):
                        routed.append(
                            {
                                "phase": "decode",
                                "token_index": token.token_index,
                                "token_id": token.token_id,
                                "layer_index": layer,
                                "boundary": boundary,
                            }
                        )
                mode_rows.append(
                    {
                        "request_id": request.request_id,
                        "input_token_ids": list(request.input_token_ids),
                        "output_token_ids": list(request.output_token_ids),
                        "stop_reason": request.stop_reason.value,
                        "token_boundaries": boundaries,
                        "routed_decisions": routed,
                    }
                )
            if group_cursor != len(forward_groups) or token_cursor != len(token_rows):
                raise AssertionError("Transformers diagnostics left unassociated rows")
            observations[label] = mode_rows

    (output_dir / "observations.json").write_text(
        json.dumps(observations, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalized_vllm_stop(choice: Any, eos_token_id: int) -> str:
    if choice.finish_reason == "length":
        return "length-cap"
    if choice.finish_reason != "stop":
        return f"unknown:{choice.finish_reason}"
    if choice.stop_reason == eos_token_id:
        return "eos"
    if isinstance(choice.stop_reason, str):
        return "stop-string"
    return f"unknown-stop:{choice.stop_reason!r}"


def _cpu_observations(
    steps: list[dict[str, Any]],
    outputs: list[tuple[str, tuple[int, ...], str]],
    prompts: dict[str, tuple[int, ...]],
) -> list[dict[str, Any]]:
    routed: dict[str, list[dict[str, Any]]] = {name: [] for name in prompts}
    token_boundaries: dict[str, list[dict[str, Any]]] = {name: [] for name in prompts}
    forwarded = {name: 0 for name in prompts}
    observed_outputs = {name: ids for name, ids, _ in outputs}
    for step in steps:
        offset = 0
        for request_id, count in step["num_scheduled_tokens"].items():
            start = forwarded[request_id]
            prompt_count = len(prompts[request_id])
            for local_index in range(count):
                absolute = start + local_index
                if absolute < prompt_count:
                    phase = "prefill"
                    token_index = absolute
                    token_id = prompts[request_id][absolute]
                else:
                    phase = "decode"
                    token_index = absolute - prompt_count
                    token_id = observed_outputs[request_id][token_index]
                for layer in range(24):
                    boundary = step["gate_rows"][str(layer)][offset + local_index]
                    routed[request_id].append(
                        {
                            "phase": phase,
                            "token_index": token_index,
                            "token_id": token_id,
                            "layer_index": layer,
                            "boundary": boundary,
                        }
                    )
            forwarded[request_id] += count
            offset += count
        for token_row in step["token_rows"]:
            boundary = token_row["boundary"]
            if boundary is None:
                raise AssertionError("vLLM sampler diagnostic is missing")
            sampled = token_row["sampled_token_ids"]
            if len(sampled) != 1:
                raise AssertionError("vLLM emitted a non-unit sampled-token row")
            token_boundaries[token_row["request_id"]].append(
                _boundary_json(sampled, boundary["boundary_ids"], boundary["margin"])
            )
    result = []
    for request_id, token_ids, stop_reason in outputs:
        result.append(
            {
                "request_id": request_id,
                "input_token_ids": list(prompts[request_id]),
                "output_token_ids": list(token_ids),
                "stop_reason": stop_reason,
                "token_boundaries": token_boundaries[request_id],
                "routed_decisions": routed[request_id],
            }
        )
    return result


def _attempt_vllm_cpu(args: argparse.Namespace) -> None:
    output_dir = args.run_dir / "vllm_cpu"
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "result.json"
    reached = ["child-started"]
    partial: list[str] = []
    llm: Any | None = None
    try:
        import torch
        import vllm.platforms
        from vllm import LLM, SamplingParams
        from vllm.platforms.cpu import CpuPlatform

        observed = json.loads(
            (args.run_dir / "transformers" / "observations.json").read_text(encoding="utf-8")
        )
        prompts = {row["request_id"]: tuple(row["input_token_ids"]) for row in observed["greedy"]}
        before_cuda = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        vllm.platforms._current_platform = CpuPlatform()
        reached.append("cpu-platform-selected")
        from oracle_worker import ValidationCPUWorker, latest_validation_worker
        from vllm.v1.worker.cpu_model_runner import CPUModelRunner
        from vllm.v1.worker.cpu_worker import CPUWorker

        with offline_environment():
            reached.append("llm-construction-entered")
            llm = LLM(
                model=str(args.cache_dir / MODEL_RELATIVE_PATH),
                device="cpu",
                dtype="float32",
                worker_cls=ValidationCPUWorker,
                enforce_eager=True,
                max_model_len=64,
                kv_cache_memory_bytes=256 * 1024 * 1024,
                disable_log_stats=True,
                enable_chunked_prefill=False,
                async_scheduling=False,
            )
            reached.append("llm-constructed")
            worker = latest_validation_worker()
            if worker is None:
                raise AssertionError("validation CPU worker was not retained")
            model = worker.model_runner.get_model()
            qualification = {
                "platform": type(vllm.platforms.current_platform).__name__,
                "worker": type(worker).__name__,
                "worker_delegates_to_cpu_worker": isinstance(worker, CPUWorker),
                "model_runner": type(worker.model_runner).__name__,
                "cpu_model_runner": isinstance(worker.model_runner, CPUModelRunner),
                "model_class": type(model).__name__,
                "all_parameters_cpu": all(
                    parameter.device.type == "cpu" for parameter in model.parameters()
                ),
            }
            if qualification != {
                "platform": "CpuPlatform",
                "worker": "ValidationCPUWorker",
                "worker_delegates_to_cpu_worker": True,
                "model_runner": "CPUModelRunner",
                "cpu_model_runner": True,
                "model_class": "GraniteMoeForCausalLM",
                "all_parameters_cpu": True,
            }:
                raise AssertionError(f"CPU qualification failed: {qualification}")

            all_observations = {}
            for mode in ("greedy", "seeded-sampling"):
                start_step = len(worker.validation_steps)
                output_rows = []
                for spec in REQUEST_SPECS:
                    request_id = spec["request_id"]
                    if mode == "greedy":
                        sampling = SamplingParams(
                            temperature=0.0,
                            max_tokens=spec["max_new_tokens"],
                            stop=list(spec["stop_strings"]),
                        )
                    else:
                        sampling = SamplingParams(
                            temperature=0.8,
                            top_p=0.9,
                            seed=173,
                            max_tokens=spec["max_new_tokens"],
                            stop=list(spec["stop_strings"]),
                        )
                    generated = llm.generate(
                        [{"prompt_token_ids": list(prompts[request_id])}],
                        sampling,
                        use_tqdm=False,
                    )
                    if len(generated) != 1 or len(generated[0].outputs) != 1:
                        raise AssertionError("vLLM returned unexpected output cardinality")
                    choice = generated[0].outputs[0]
                    output_rows.append(
                        (
                            request_id,
                            tuple(choice.token_ids),
                            _normalized_vllm_stop(choice, llm.get_tokenizer().eos_token_id),
                        )
                    )
                steps = worker.validation_steps[start_step:]
                all_observations[mode] = _cpu_observations(steps, output_rows, prompts)
            after_cuda = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            if after_cuda != before_cuda:
                raise AssertionError(
                    f"CUDA allocation changed from {before_cuda} to {after_cuda} bytes"
                )
            payload = {
                "status": "qualified",
                "qualification": qualification,
                "cuda_allocated_before": before_cuda,
                "cuda_allocated_after": after_cuda,
                "observations": all_observations,
                "reached": reached,
                "partial_artifacts": partial,
            }
    except Exception as exc:  # noqa: BLE001 - every CPU boundary failure is evidence
        payload = {
            "status": "blocked",
            "attempted_platform": "CpuPlatform",
            "attempted_worker": "ValidationCPUWorker(CPUWorker)",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "reached": reached,
            "partial_artifacts": partial,
            "traceback": traceback.format_exc(),
        }
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prepare_replay(args: argparse.Namespace) -> None:
    from simllm.core import RequestBookkeeper
    from simllm.preplay import (
        RequestArrival,
        join_preplay_arrivals,
        project_preplay_routing,
        read_preplay_trace,
        write_preplay_replay_run,
        write_routed_experts,
    )

    replay_dir = args.run_dir / "replay"
    replay_dir.mkdir(parents=True, exist_ok=False)
    trace_path = args.run_dir / "transformers" / "greedy.jsonl"
    trace = read_preplay_trace(trace_path)
    accepted = {
        "eos-brief": (15, 3, "eos"),
        "length-cap": (22, 1, "length-cap"),
        "stop-string": (20, 5, "stop-string"),
    }
    observed = {
        request.request_id: (
            len(request.input_token_ids),
            len(request.output_token_ids),
            request.stop_reason.value,
        )
        for request in trace.requests
    }
    if observed != accepted:
        raise AssertionError(f"greedy PLAY-1 oracle changed: {observed}")
    run = join_preplay_arrivals(
        (
            RequestArrival(request_id=request.request_id, arrived_at_ps=0)
            for request in trace.requests
        ),
        trace_path,
        RequestBookkeeper(),
    )
    write_preplay_replay_run(run, replay_dir / "run.json")
    write_routed_experts(
        project_preplay_routing(run),
        replay_dir / "routed-experts.json",
    )


def _traffic_dims() -> Any:
    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=24,
        hidden_size=1024,
        intermediate_size=512,
        num_heads=16,
        num_kv_heads=8,
        head_size=64,
        vocab_size=49_152,
        dtype_bytes=2,
        num_experts=32,
        top_k=8,
        moe_intermediate_size=512,
        local_num_experts=16,
    )


def _goal_sends_by_tag(text: str) -> dict[int, dict[tuple[int, int], int]]:
    result: dict[int, dict[tuple[int, int], int]] = {}
    rank = -1
    for line in text.splitlines():
        if line.startswith("rank "):
            rank = int(line.split()[1])
        if ": send " not in line:
            continue
        words = line.split()
        pair = (rank, int(words[4]))
        tag = int(words[6])
        if pair in result.setdefault(tag, {}):
            raise AssertionError(f"duplicate GOAL send for tag {tag}, pair {pair}")
        result[tag][pair] = int(words[2].removesuffix("b"))
    return result


def _scheduled_tokens(record: Any, routed: Any) -> tuple[Any, ...]:
    tokens = []
    for scheduled in record.scheduled:
        request = routed.by_request_id(scheduled.request_id)
        if scheduled.num_new_tokens == 0:
            continue
        if scheduled.phase.value == "prefill":
            end = scheduled.context_length
            start = end - scheduled.num_new_tokens
            candidates = request.prefill_tokens
        else:
            start = scheduled.context_length - scheduled.num_new_tokens - request.prompt_token_count
            end = start + scheduled.num_new_tokens
            candidates = request.decode_tokens
        if start < 0 or end > len(candidates) or start >= end:
            raise AssertionError("independent routed-token slice is invalid")
        tokens.extend(candidates[start:end])
    if len(tokens) != record.total_new_tokens:
        raise AssertionError("independent captured-token conservation failed")
    return tuple(tokens)


def _expected_pairs(record: Any, routed: Any) -> dict[int, dict[tuple[int, int], int]]:
    tokens = _scheduled_tokens(record, routed)
    result = {}
    for layer in range(24):
        dispatch: dict[tuple[int, int], int] = {}
        for source in (0, 1):
            for token in tokens:
                destinations = {
                    0 if expert < 16 else 1 for expert in token.layers[layer].expert_ids
                }
                for destination in destinations:
                    if source != destination:
                        pair = (source, destination)
                        dispatch[pair] = dispatch.get(pair, 0) + VECTOR_BYTES
        combine = {(destination, source): size for (source, destination), size in dispatch.items()}
        result[1000 + layer * 2] = dispatch
        result[1000 + layer * 2 + 1] = combine
    return result


def _expected_step_delta(expected: dict[int, dict[tuple[int, int], int]]) -> int:
    total = 0
    for layer in range(24):
        dispatch = expected[1000 + layer * 2]
        combine = expected[1000 + layer * 2 + 1]
        total += max(dispatch.values(), default=0)
        total += max(combine.values(), default=0)
    return -20 * total


def _observe_outputs(
    outputs: list[Any],
    step_index: int,
    completed_at_ps: int,
    served: dict[str, list[int]],
    token_times: dict[str, list[int]],
    token_steps: dict[str, list[int]],
    finishes: dict[str, dict[str, Any]],
    completion_order: list[str],
    eos_token_id: int,
) -> None:
    for output in outputs:
        request_id = output.request_id
        if request_id not in served or len(output.outputs) != 1:
            raise AssertionError("scheduler returned an unknown or multi-choice request")
        choice = output.outputs[0]
        cumulative = tuple(choice.token_ids)
        previous = tuple(served[request_id])
        if cumulative[: len(previous)] != previous:
            raise AssertionError(f"request {request_id} changed an emitted prefix")
        new_tokens = cumulative[len(previous) :]
        served[request_id].extend(new_tokens)
        token_times[request_id].extend([completed_at_ps] * len(new_tokens))
        token_steps[request_id].extend([step_index] * len(new_tokens))
        if output.finished:
            if request_id in finishes:
                raise AssertionError(f"request {request_id} finished twice")
            finishes[request_id] = {
                "step_index": step_index,
                "finish_reason": choice.finish_reason,
                "stop_reason": choice.stop_reason,
                "normalized": _normalized_vllm_stop(choice, eos_token_id),
            }
            completion_order.append(request_id)


def _replay_cell(args: argparse.Namespace, bandwidth: int) -> None:
    from vllm import LLM, SamplingParams

    from simllm.adapters.vllm import (
        SimExecutorConfig,
        configure,
        latest_worker,
        reset_configuration,
    )
    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
    from simllm.compute import ComputeProvider, DurationEstimate
    from simllm.preplay import read_preplay_trace, read_routed_experts
    from simllm.traffic import ExpertPlacementSnapshot, RoutedMoeSupply

    class FixedProvider(ComputeProvider):
        def estimate(self, kernel: Any, gpu: Any) -> DurationEstimate:
            return DurationEstimate(duration_ps=24_000, bound="measured")

    label = f"{bandwidth // 1_000_000_000}g"
    cell_dir = args.run_dir / "replay" / label
    cell_dir.mkdir(parents=True, exist_ok=False)
    trace = read_preplay_trace(args.run_dir / "transformers" / "greedy.jsonl")
    routed = read_routed_experts(args.run_dir / "replay" / "routed-experts.json")
    placement = ExpertPlacementSnapshot(
        placement_epoch=0,
        expert_owners=tuple(
            (layer, expert, 0 if expert < 16 else 1) for layer in range(24) for expert in range(32)
        ),
    )
    supply = RoutedMoeSupply(
        routed_experts=routed,
        placements=(placement,),
        step_placement_epochs=tuple((step, 0) for step in range(128)),
    )
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0,),
            dims=_traffic_dims(),
            workdir=cell_dir / "htsim",
            ep_ranks=(0, 1),
            linkspeed_bps=bandwidth,
            provider=FixedProvider(),
            routed_moe_supply=supply,
        )
    )
    served = {request.request_id: [] for request in trace.requests}
    token_times = {request.request_id: [] for request in trace.requests}
    token_steps = {request.request_id: [] for request in trace.requests}
    finishes: dict[str, dict[str, Any]] = {}
    completion_order: list[str] = []
    llm: Any | None = None
    reset_configuration()
    configure(
        step_sink=sink,
        config=SimExecutorConfig(
            mode="virtual",
            token_id=512,
            step_records_path=str(cell_dir / "steps.jsonl"),
            replay_run_path=str(args.run_dir / "replay" / "run.json"),
        ),
    )
    try:
        with offline_environment(replay=True):
            llm = LLM(
                model=str(args.cache_dir / MODEL_RELATIVE_PATH),
                worker_cls="simllm.adapters.vllm.SimWorker",
                enforce_eager=True,
                max_model_len=64,
                num_gpu_blocks_override=64,
                disable_log_stats=True,
                enable_chunked_prefill=False,
                async_scheduling=False,
            )
            worker = latest_worker()
            if worker is None or worker.replay is None:
                raise AssertionError("replay SimWorker was not constructed")
            eos_token_id = llm.get_tokenizer().eos_token_id
            for request in trace.requests:
                request_id = llm.llm_engine.add_request(
                    request.request_id,
                    {"prompt_token_ids": list(request.input_token_ids)},
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=len(request.output_token_ids),
                        min_tokens=0,
                        stop=list(request.stop_strings),
                        detokenize=True,
                    ),
                )
                if request_id != request.request_id:
                    raise AssertionError("scheduler changed a request identity")
            while llm.llm_engine.has_unfinished_requests():
                before = len(worker.step_records)
                outputs = llm.llm_engine.step()
                if len(worker.step_records) != before + 1:
                    raise AssertionError("one scheduler step did not emit one record")
                record = worker.step_records[-1]
                _observe_outputs(
                    outputs,
                    record.step_index,
                    worker.clock.now_ps,
                    served,
                    token_times,
                    token_steps,
                    finishes,
                    completion_order,
                    eos_token_id,
                )
            records = tuple(worker.step_records)
            results = tuple(worker.step_results)
            replay_snapshot = worker.replay.snapshot()
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        reset_configuration()

    if len(records) != len(results):
        raise AssertionError("step record and result cardinality differ")
    expected_deltas = []
    goal_checks = []
    network_index = 0
    for record in records:
        if record.total_new_tokens == 0:
            expected_deltas.append(0)
            continue
        expected = _expected_pairs(record, routed)
        goal_path = cell_dir / "htsim" / f"step-{record.step_index:06d}.goal"
        actual = _goal_sends_by_tag(goal_path.read_text(encoding="utf-8"))
        exact = actual == expected
        goal_checks.append(
            {
                "step_index": record.step_index,
                "exact": exact,
                "send_count": sum(len(pairs) for pairs in actual.values()),
            }
        )
        expected_deltas.append(_expected_step_delta(expected))
        outcome = sink.outcomes[network_index]
        if (
            outcome.step_index != record.step_index
            or outcome.routing_mode != "captured"
            or outcome.placement_epoch != 0
            or not outcome.quiescent
        ):
            raise AssertionError("backend outcome failed its captured-routing gates")
        network_index += 1
    if network_index != len(sink.outcomes):
        raise AssertionError("network outcome cardinality changed")

    payload = {
        "bandwidth_bps": bandwidth,
        "tokens": served,
        "token_times_ps": token_times,
        "token_steps": token_steps,
        "finishes": finishes,
        "completion_order": completion_order,
        "step_composition": [
            [request.request_id for request in record.scheduled] for record in records
        ],
        "step_latencies_ps": [result.step_latency_ps for result in results],
        "expected_400_minus_200_step_ps": expected_deltas,
        "goal_checks": goal_checks,
        "goal_stream_exact": bool(goal_checks) and all(check["exact"] for check in goal_checks),
        "outcomes": [asdict(outcome) for outcome in sink.outcomes],
        "replay_completed": list(replay_snapshot.completed_request_ids),
        "replay_drained": list(replay_snapshot.drained_request_ids),
    }
    (cell_dir / "cell.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _observation_from_json(row: dict[str, Any]) -> Any:
    from simllm.preplay.validation import (
        DecisionBoundary,
        OracleRequestObservation,
        RoutedDecision,
    )

    def boundary(value: dict[str, Any]) -> DecisionBoundary:
        return DecisionBoundary(
            selected_ids=tuple(value["selected_ids"]),
            boundary_ids=tuple(value["boundary_ids"]),
            margin=value["margin"],
        )

    return OracleRequestObservation(
        request_id=row["request_id"],
        input_token_ids=tuple(row["input_token_ids"]),
        output_token_ids=tuple(row["output_token_ids"]),
        stop_reason=row["stop_reason"],
        token_boundaries=tuple(boundary(value) for value in row["token_boundaries"]),
        routed_decisions=tuple(
            RoutedDecision(
                phase=value["phase"],
                token_index=value["token_index"],
                token_id=value["token_id"],
                layer_index=value["layer_index"],
                boundary=boundary(value["boundary"]),
            )
            for value in row["routed_decisions"]
        ),
    )


def _oracle_summary(args: argparse.Namespace) -> dict[str, Any]:
    from simllm.preplay.validation import compare_oracle_requests

    cpu = json.loads((args.run_dir / "vllm_cpu" / "result.json").read_text(encoding="utf-8"))
    if cpu["status"] != "qualified":
        return {
            "status": "blocked",
            "executed_scored": 0,
            "blocked_rows": 6,
            "blocker": cpu,
            "rows": [],
        }
    transformers = json.loads(
        (args.run_dir / "transformers" / "observations.json").read_text(encoding="utf-8")
    )
    rows = []
    for mode in ("greedy", "seeded-sampling"):
        left_by_id = {row["request_id"]: row for row in transformers[mode]}
        right_by_id = {row["request_id"]: row for row in cpu["observations"][mode]}
        if set(left_by_id) != set(right_by_id):
            raise AssertionError("oracle request sets differ")
        for request_id, left in left_by_id.items():
            comparison = compare_oracle_requests(
                _observation_from_json(left),
                _observation_from_json(right_by_id[request_id]),
                sampling_mode=mode,
                near_tie_abs_logit=NEAR_TIE_ABS_LOGIT,
            )
            rows.append(
                {
                    **asdict(comparison),
                    "divergences": [
                        {
                            **asdict(divergence),
                            "classification": divergence.classification.value,
                        }
                        for divergence in comparison.divergences
                    ],
                    "passed": comparison.passed,
                    "unclassified_count": comparison.unclassified_count,
                }
            )
    return {
        "status": "executed",
        "executed_scored": len(rows),
        "blocked_rows": 0,
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
    }


def _fraction_json(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _replay_summary(args: argparse.Namespace) -> dict[str, Any]:
    cells = {
        bandwidth: json.loads(
            (args.run_dir / "replay" / f"{bandwidth // 1_000_000_000}g" / "cell.json").read_text(
                encoding="utf-8"
            )
        )
        for bandwidth in BANDWIDTHS_BPS
    }
    slow = cells[BANDWIDTHS_BPS[0]]
    fast = cells[BANDWIDTHS_BPS[1]]
    trace_rows = json.loads(
        (args.run_dir / "transformers" / "observations.json").read_text(encoding="utf-8")
    )["greedy"]
    trace_by_id = {row["request_id"]: row for row in trace_rows}
    completion_rows = []
    for bandwidth, cell in cells.items():
        for request_id, expected in trace_by_id.items():
            finish = cell["finishes"].get(request_id)
            completion_rows.append(
                {
                    "bandwidth_bps": bandwidth,
                    "request_id": request_id,
                    "passed": (
                        cell["tokens"].get(request_id) == expected["output_token_ids"]
                        and finish is not None
                        and finish["normalized"] == expected["stop_reason"]
                    ),
                    "output_length": len(cell["tokens"].get(request_id, [])),
                    "stop_reason": finish["normalized"] if finish else None,
                }
            )
    bandwidth_identity = all(
        slow[field] == fast[field]
        for field in ("tokens", "finishes", "completion_order", "step_composition")
    )
    if slow["expected_400_minus_200_step_ps"] != fast["expected_400_minus_200_step_ps"]:
        raise AssertionError("independent pair sizes changed with bandwidth")
    step_deltas = slow["expected_400_minus_200_step_ps"]
    if len(slow["step_latencies_ps"]) != len(fast["step_latencies_ps"]):
        raise AssertionError("bandwidth changed step cardinality")
    measured_step_deltas = [
        right - left
        for left, right in zip(slow["step_latencies_ps"], fast["step_latencies_ps"], strict=True)
    ]
    step_relation_exact = measured_step_deltas == step_deltas
    cumulative = []
    total = 0
    for delta in step_deltas:
        total += delta
        cumulative.append(total)

    ttft_rows = []
    tpot_rows = []
    for request_id, expected in trace_by_id.items():
        slow_times = slow["token_times_ps"][request_id]
        fast_times = fast["token_times_ps"][request_id]
        steps = slow["token_steps"][request_id]
        observed_ttft_delta = fast_times[0] - slow_times[0]
        expected_ttft_delta = cumulative[steps[0]]
        ttft_rows.append(
            {
                "request_id": request_id,
                "observed_400_minus_200_ps": observed_ttft_delta,
                "expected_400_minus_200_ps": expected_ttft_delta,
                "passed": (observed_ttft_delta == expected_ttft_delta and observed_ttft_delta < 0),
            }
        )
        if len(expected["output_token_ids"]) > 1:
            slow_tpot = Fraction(
                sum(later - earlier for earlier, later in pairwise(slow_times)),
                len(slow_times) - 1,
            )
            fast_tpot = Fraction(
                sum(later - earlier for earlier, later in pairwise(fast_times)),
                len(fast_times) - 1,
            )
            expected_delta = Fraction(
                cumulative[steps[-1]] - cumulative[steps[0]],
                len(steps) - 1,
            )
            observed_delta = fast_tpot - slow_tpot
            tpot_rows.append(
                {
                    "request_id": request_id,
                    "observed_400_minus_200_ps": _fraction_json(observed_delta),
                    "expected_400_minus_200_ps": _fraction_json(expected_delta),
                    "passed": observed_delta == expected_delta and observed_delta < 0,
                }
            )
    replay = {
        "completion_rows": completion_rows,
        "routed_stream_rows": [
            {
                "bandwidth_bps": bandwidth,
                "passed": cell["goal_stream_exact"],
                "step_count": len(cell["goal_checks"]),
            }
            for bandwidth, cell in cells.items()
        ],
        "ttft_rows": ttft_rows,
        "tpot_rows": tpot_rows,
        "fatal_unscored": {
            "bandwidth_scheduler_identity": bandwidth_identity,
            "step_bandwidth_relation": step_relation_exact,
            "replay_completed": all(
                set(cell["replay_completed"]) == set(trace_by_id) for cell in cells.values()
            ),
            "backend_quiescence": all(
                all(outcome["quiescent"] for outcome in cell["outcomes"]) for cell in cells.values()
            ),
        },
    }
    scored = completion_rows + replay["routed_stream_rows"] + ttft_rows + tpot_rows
    replay["executed_scored"] = len(scored)
    replay["passed_scored"] = sum(row["passed"] for row in scored)
    replay["passed"] = replay["passed_scored"] == len(scored) and all(
        replay["fatal_unscored"].values()
    )
    return replay


def _run_child(args: argparse.Namespace, mode: str) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cache-dir",
        str(args.cache_dir),
        "--vllm-package-root",
        str(args.vllm_package_root),
        "--htsim-rnic",
        str(args.htsim_rnic),
        "--run-dir",
        str(args.run_dir),
        "--internal",
        mode,
    ]
    log_path = args.run_dir / f"{mode}.log"
    with log_path.open("x", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    args.run_dir.mkdir(parents=True, exist_ok=False)
    os.environ["SIMLLM_HTSIM_RNIC"] = str(args.htsim_rnic)
    _run_child(args, "transformers")
    _run_child(args, "vllm-cpu")
    _prepare_replay(args)
    for bandwidth in BANDWIDTHS_BPS:
        _run_child(args, f"replay-{bandwidth}")
    oracle = _oracle_summary(args)
    replay = _replay_summary(args)
    executed = oracle["executed_scored"] + replay["executed_scored"]
    passed = (oracle["status"] == "blocked" or oracle.get("passed", False)) and replay["passed"]
    summary = {
        "freeze_commit": FREEZE_COMMIT,
        "oracle_consistency": oracle,
        "replay": replay,
        "evidence": {
            "executed_scored": executed,
            "passed_scored": (
                (0 if oracle["status"] == "blocked" else oracle["executed_scored"])
                + replay["passed_scored"]
            ),
            "blocked_scored": oracle["blocked_rows"],
            "genuine_risk_numerator": executed,
            "genuine_risk_denominator": executed,
        },
        "play5_complete": oracle["status"] == "executed" and passed,
        "replay_complete": replay["passed"],
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not replay["passed"]:
        raise AssertionError("PLAY-5 replay half failed its frozen acceptance bar")
    if oracle["status"] == "executed" and not oracle["passed"]:
        raise AssertionError("PLAY-5 oracle comparison found an unclassified divergence")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--vllm-package-root", type=Path, required=True)
    parser.add_argument("--htsim-rnic", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--internal",
        choices=("transformers", "vllm-cpu")
        + tuple(f"replay-{bandwidth}" for bandwidth in BANDWIDTHS_BPS),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_only(args)
    if args.check_only:
        return
    if args.internal == "transformers":
        _transformers_capture(args)
        return
    if args.internal == "vllm-cpu":
        _attempt_vllm_cpu(args)
        return
    if args.internal and args.internal.startswith("replay-"):
        _replay_cell(args, int(args.internal.removeprefix("replay-")))
        return
    summary = run_study(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
