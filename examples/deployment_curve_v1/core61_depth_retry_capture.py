#!/usr/bin/env python3
"""Reach one exact depth-8 batch-32 KV-2000 decode step under a small startup cap."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

MODEL = "deepseek-ai/DeepSeek-V3"
REVISION = "e815299b0bcbac849fa540c768ef21845365c9eb"
CONFIG_SHA256 = "cbf0b95dc614de208a109bb5fd4e7eed11385e9c68411d2c17db5319443035d9"
CELL = "decode_b32_c2000"
VOCAB_FLOOR = 100
VOCAB_SPAN = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--reduced-layers", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--startup-max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--remote-kv-tokens", type=int, default=2000)
    parser.add_argument("--phase", choices=("align", "profile"), required=True)
    parser.add_argument("--active-file", type=Path, required=True)
    parser.add_argument("--marker-dir", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def prompts(input_lengths: list[int]) -> list[TokensPrompt]:
    rows = []
    for batch_index, input_len in enumerate(input_lengths):
        token_ids = [
            VOCAB_FLOOR + ((batch_index * 37 + index) % VOCAB_SPAN)
            for index in range(input_len)
        ]
        rows.append(TokensPrompt(prompt_token_ids=token_ids))
    return rows


def activate(args: argparse.Namespace, **fields: Any) -> Path:
    payload = {
        "model": args.model,
        "revision": args.revision,
        "reduced_layers": args.reduced_layers,
        **fields,
    }
    write_json(args.active_file, payload)
    cell = str(fields["cell"]).replace("/", "_")
    marker = args.marker_dir / f"{cell}-rank0.json"
    if marker.exists():
        raise RuntimeError(f"refusing to overwrite retained marker {marker}")
    return marker


def add_requests(
    llm: LLM,
    prefix: str,
    input_lengths: list[int],
    max_tokens: int,
) -> list[str]:
    sampling = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=max_tokens,
        detokenize=False,
    )
    request_ids = []
    for index, prompt in enumerate(prompts(input_lengths)):
        request_id = f"{prefix}-{index:02d}"
        llm.llm_engine.add_request(request_id, prompt, sampling)
        request_ids.append(request_id)
    return request_ids


def drive_until_marker(
    llm: LLM,
    request_ids: list[str],
    marker: Path,
    max_steps: int,
) -> dict[str, Any]:
    engine = llm.llm_engine
    for _ in range(max_steps):
        engine.step()
        if marker.exists():
            payload = json.loads(marker.read_text(encoding="utf-8"))
            engine.abort_request(request_ids)
            return payload
        if not engine.has_unfinished_requests():
            break
    engine.abort_request(request_ids)
    raise RuntimeError(
        f"all requests ended or {max_steps} steps elapsed before {marker.name}"
    )


def output_headroom(
    input_lengths: list[int], max_model_len: int, scheduler_tokens: int, batch: int
) -> int:
    prefill_steps = math.ceil(sum(input_lengths) / scheduler_tokens)
    longest_steps = math.ceil(max(input_lengths) / scheduler_tokens)
    overlap_steps = max(0, prefill_steps - longest_steps)
    value = min(
        max_model_len - max(input_lengths),
        max(2, overlap_steps + max(64, batch * 128)),
    )
    if value <= overlap_steps:
        raise RuntimeError("alignment consumed the supported output-token headroom")
    return value


def calibrate(args: argparse.Namespace, llm: LLM) -> dict[str, Any]:
    lengths = [args.remote_kv_tokens] * args.batch_size
    rounds = []
    for round_index in range(5):
        max_tokens = output_headroom(
            lengths,
            args.max_model_len,
            args.startup_max_num_batched_tokens,
            args.batch_size,
        )
        cell = f"{CELL}_alignment_{round_index}"
        marker = activate(
            args,
            cell=cell,
            kind="kv_calibration",
            batch_size=args.batch_size,
            target_kv_tokens=args.remote_kv_tokens,
        )
        request_ids = add_requests(llm, cell, lengths, max_tokens)
        payload = drive_until_marker(
            llm,
            request_ids,
            marker,
            math.ceil(sum(lengths) / args.startup_max_num_batched_tokens)
            + max_tokens
            + 128,
        )
        computed_by_request = payload["scheduler"][
            "cached_num_computed_tokens_by_request"
        ]
        observed = []
        for request_id in request_ids:
            matches = [
                value
                for observed_id, value in computed_by_request.items()
                if observed_id == request_id or observed_id.startswith(request_id + "-")
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one scheduler identity for {request_id}, found {len(matches)}"
                )
            observed.append(int(matches[0]))
        rounds.append(
            {
                "round": round_index,
                "prompt_lengths": lengths,
                "observed_kv_lengths": observed,
            }
        )
        if all(value == args.remote_kv_tokens for value in observed):
            return {
                "batch_size": args.batch_size,
                "remote_kv_tokens": args.remote_kv_tokens,
                "prompt_lengths": lengths,
                "max_tokens": max_tokens,
                "rounds": rounds,
            }
        lengths = [
            max(1, length - (actual - args.remote_kv_tokens))
            for length, actual in zip(lengths, observed)
        ]
    raise RuntimeError("five alignment rounds did not produce the exact KV batch")


def model_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(
        hf_hub_download(
            repo_id=args.model,
            filename="config.json",
            revision=args.revision,
            local_files_only=True,
        )
    )
    if sha256(config_path) != CONFIG_SHA256:
        raise RuntimeError("pinned DeepSeek config digest changed")
    config = AutoConfig.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        local_files_only=True,
    )
    if args.model != MODEL or args.revision != REVISION:
        raise RuntimeError("CORE-61 retry refuses a different model or revision")
    if args.reduced_layers != 8 or int(config.num_hidden_layers) != 61:
        raise RuntimeError("CORE-61 retry requires the frozen 61-to-8 depth identity")
    if args.batch_size != 32 or args.remote_kv_tokens != 2000:
        raise RuntimeError("CORE-61 retry refuses a changed decode shape")
    if args.max_num_seqs != 32 or args.startup_max_num_batched_tokens != 4096:
        raise RuntimeError("CORE-61 retry refuses a changed startup amendment")
    return {
        "requested_revision": args.revision,
        "resolved_revision": args.revision,
        "config_sha256": CONFIG_SHA256,
        "full_num_hidden_layers": 61,
        "effective_num_hidden_layers": 8,
        "retained_dense_layers": 3,
        "retained_moe_layers": 5,
        "hidden_size": int(config.hidden_size),
    }


def engine(args: argparse.Namespace) -> LLM:
    return LLM(
        model=args.model,
        revision=args.revision,
        load_format="dummy",
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=False,
        trust_remote_code=True,
        skip_tokenizer_init=True,
        enable_prefix_caching=False,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.startup_max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=17,
        disable_log_stats=True,
        profiler_config={"profiler": "cuda"},
        hf_overrides={"num_hidden_layers": args.reduced_layers},
    )


def validate_exact_marker(args: argparse.Namespace, marker: dict[str, Any]) -> None:
    scheduler = marker["scheduler"]
    if scheduler["is_decode"] is not True or scheduler["num_requests"] != 32:
        raise RuntimeError("scheduler marker is not one batch-32 decode step")
    computed = scheduler["cached_num_computed_tokens_by_request"].values()
    if len(scheduler["cached_num_computed_tokens_by_request"]) != 32 or any(
        int(value) != 2000 for value in computed
    ):
        raise RuntimeError("scheduler marker does not contain 32 exact KV-2000 states")


def run_profile(
    args: argparse.Namespace,
    llm: LLM,
    config: dict[str, Any],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    if alignment["batch_size"] != 32 or alignment["remote_kv_tokens"] != 2000:
        raise RuntimeError("alignment file changed the frozen shape")
    prompt_lengths = [int(value) for value in alignment["prompt_lengths"]]
    if len(prompt_lengths) != 32:
        raise RuntimeError("alignment file must contain 32 prompt lengths")
    marker_path = activate(
        args,
        cell=CELL,
        kind="core61_exact_decode",
        batch_size=32,
        target_kv_tokens=2000,
        require_exact_kv=True,
    )
    request_ids = add_requests(llm, CELL, prompt_lengths, int(alignment["max_tokens"]))
    started_epoch_ns = time.time_ns()
    llm.start_profile()
    marker = drive_until_marker(
        llm,
        request_ids,
        marker_path,
        math.ceil(sum(prompt_lengths) / args.startup_max_num_batched_tokens)
        + int(alignment["max_tokens"])
        + 128,
    )
    llm.stop_profile()
    finished_epoch_ns = time.time_ns()
    validate_exact_marker(args, marker)
    return {
        "schema": "simllm-core61-depth8-retry-capture-v1",
        "model": args.model,
        "revision": args.revision,
        "model_config": config,
        "framework": {
            "name": "vllm",
            "version": importlib.metadata.version("vllm"),
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
        "machine": {
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "reduced_layers": 8,
        "startup_max_num_batched_tokens": 4096,
        "max_num_seqs": 32,
        "alignment": alignment,
        "cases": [
            {
                "cell": CELL,
                "pool": "decode",
                "batch_size": 32,
                "remote_kv_tokens_per_request": 2000,
                "prompt_lengths": prompt_lengths,
                "decode_steps": 1,
                "started_epoch_ns": started_epoch_ns,
                "finished_epoch_ns": finished_epoch_ns,
                "scheduler_marker": marker,
            }
        ],
    }


def main() -> int:
    args = parse_args()
    config = model_config(args)
    llm = engine(args)
    if args.phase == "align":
        write_json(args.alignment, calibrate(args, llm))
        return 0
    if args.output is None:
        raise ValueError("--output is required for profile phase")
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    write_json(args.output, run_profile(args, llm, config, alignment))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
