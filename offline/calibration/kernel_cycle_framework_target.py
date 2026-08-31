#!/usr/bin/env python3
"""Run one pinned framework cell without loading model weights.

This executable is the framework-facing half of ``kernel_cycle_capture.py``.
It intentionally fails before engine construction when the public offline API
cannot prove a requested placement or routing contract. That keeps an
unsupported cell from being mistaken for digest-complete campaign evidence.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

SUPPORTED_FRAMEWORKS = {"sglang", "vllm"}
VOCABULARY_FLOOR = 100
VOCABULARY_SPAN = 1000


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def load_cell(path: Path) -> dict[str, Any]:
    """Load the exact cell projection consumed by a framework target."""

    cell = _object(json.loads(path.read_text(encoding="utf-8")), "cell")
    required = {
        "cell_id",
        "framework",
        "kv_placement",
        "launch_mode",
        "model",
        "parallelism",
        "pool",
        "replays",
        "routing_evidence_required",
        "shape",
    }
    missing = sorted(required - cell.keys())
    if missing:
        raise ValueError(f"cell is missing fields {missing}")
    if cell["framework"] not in SUPPORTED_FRAMEWORKS:
        raise ValueError(f"unsupported framework {cell['framework']!r}")
    if cell["launch_mode"] not in {"cuda-graph", "eager"}:
        raise ValueError("launch_mode must be cuda-graph or eager")
    if not isinstance(cell["replays"], int) or cell["replays"] < 1:
        raise ValueError("replays must be a positive integer")
    model = _object(cell["model"], "cell.model")
    if not all(isinstance(model.get(name), str) and model[name] for name in ("name", "revision")):
        raise ValueError("cell.model must pin a nonblank name and revision")
    parallelism = _object(cell["parallelism"], "cell.parallelism")
    for name in ("tensor_parallel", "pipeline_parallel", "data_parallel", "expert_parallel"):
        if not isinstance(parallelism.get(name), int) or parallelism[name] < 1:
            raise ValueError(f"cell.parallelism.{name} must be a positive integer")
    if parallelism["pipeline_parallel"] != 1:
        raise RuntimeError("the staged offline targets do not implement pipeline parallelism")
    if parallelism["data_parallel"] != 1 or parallelism["expert_parallel"] != 1:
        raise RuntimeError("the staged offline targets require physical DP1 and EP1")
    return cell


def capability_gaps(cell: dict[str, Any]) -> list[str]:
    """Return contracts that the public framework target cannot prove."""

    gaps: list[str] = []
    if cell["kv_placement"] == "deliberately-fragmented":
        gaps.append("deliberately-fragmented KV placement has no public offline-engine control")
    shape = _object(cell["shape"], "cell.shape")
    if cell["pool"] == "prefill" and int(shape.get("existing_context_tokens", 0)):
        gaps.append("prefill existing-context placement has no public offline-engine control")
    if cell["routing_evidence_required"]:
        gaps.append("the run-cell contract has no routed-expert sidecar output path")
    gaps.append("the run-cell contract has no two-clean-run code-object index output path")
    return gaps


def prompt_rows(cell: dict[str, Any]) -> tuple[list[list[int]], int]:
    """Build deterministic token rows and the number of emitted tokens."""

    shape = _object(cell["shape"], "cell.shape")
    if cell["pool"] == "decode":
        batch_size = int(shape["batch_size"])
        lengths = list(shape["per_request_kv_lengths"])
        if len(lengths) != batch_size or any(
            not isinstance(value, int) or value < 1 for value in lengths
        ):
            raise ValueError("decode KV vector must match the positive batch size")
        output_tokens = 2
    elif cell["pool"] == "prefill":
        computed = int(shape["computed_new_tokens"])
        existing = int(shape["existing_context_tokens"])
        if computed < 1 or existing < 0:
            raise ValueError("prefill token counts are invalid")
        lengths = [computed + existing]
        output_tokens = 1
    else:
        raise ValueError(f"unsupported pool {cell['pool']!r}")
    rows = [
        [
            VOCABULARY_FLOOR + ((row_index * length + token_index) % VOCABULARY_SPAN)
            for token_index in range(length)
        ]
        for row_index, length in enumerate(lengths)
    ]
    return rows, output_tokens


def _vllm_runner(cell: dict[str, Any]) -> tuple[Callable[[], object], object]:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    rows, output_tokens = prompt_rows(cell)
    parallelism = cell["parallelism"]
    llm = LLM(
        model=cell["model"]["name"],
        revision=cell["model"]["revision"],
        load_format="dummy",
        dtype="bfloat16",
        tensor_parallel_size=parallelism["tensor_parallel"],
        enforce_eager=cell["launch_mode"] == "eager",
        trust_remote_code=True,
        skip_tokenizer_init=True,
        max_model_len=max(map(len, rows)) + output_tokens,
        max_num_seqs=len(rows),
        max_num_batched_tokens=sum(map(len, rows)) + len(rows) * output_tokens,
        gpu_memory_utilization=0.80,
        seed=17,
        disable_log_stats=True,
        profiler_config={"profiler": "cuda"},
    )
    prompts = [TokensPrompt(prompt_token_ids=row) for row in rows]
    sampling = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=output_tokens,
        detokenize=False,
    )

    def run() -> object:
        return llm.generate(prompts, sampling_params=sampling, use_tqdm=False)

    return run, llm


def _sglang_runner(cell: dict[str, Any]) -> tuple[Callable[[], object], object]:
    import sglang as sgl

    rows, output_tokens = prompt_rows(cell)
    parallelism = cell["parallelism"]
    llm = sgl.Engine(
        model_path=cell["model"]["name"],
        revision=cell["model"]["revision"],
        load_format="dummy",
        dtype="bfloat16",
        tp_size=parallelism["tensor_parallel"],
        disable_cuda_graph=cell["launch_mode"] == "eager",
        trust_remote_code=True,
        skip_tokenizer_init=True,
        context_length=max(map(len, rows)) + output_tokens,
        mem_fraction_static=0.80,
        random_seed=17,
    )
    sampling = {
        "temperature": 0.0,
        "ignore_eos": True,
        "max_new_tokens": output_tokens,
    }

    def run() -> object:
        return llm.generate(input_ids=rows, sampling_params=sampling)

    return run, llm


def run_target(cell: dict[str, Any], result_path: Path) -> dict[str, Any]:
    gaps = capability_gaps(cell)
    if gaps:
        raise RuntimeError("cell cannot be scored: " + "; ".join(gaps))
    factory = _vllm_runner if cell["framework"] == "vllm" else _sglang_runner
    run, engine = factory(cell)
    run()
    engine.start_profile()
    try:
        for _ in range(cell["replays"]):
            run()
    finally:
        engine.stop_profile()
    result = {
        "schema": "simllm-kernel-cycle-framework-target-result-v1",
        "cell_id": cell["cell_id"],
        "framework": cell["framework"],
        "framework_version": importlib.metadata.version(cell["framework"]),
        "load_format": "dummy",
        "model_weights_loaded": False,
        "replays": cell["replays"],
    }
    with result_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-spec", required=True, type=Path)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    cell = load_cell(args.cell_spec)
    result_path = args.cell_spec.parent / f"target-run-{os.getpid()}.json"
    run_target(cell, result_path)
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
