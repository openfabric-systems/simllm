"""Run the frozen PLAY-1 Granite CPU capture study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

from simllm._local_config import path_from_env
from simllm.preplay import (
    PreplayRequest,
    SamplingConfig,
    SamplingMode,
    StopReason,
    TransformersCpuRunner,
    read_preplay_trace,
    write_preplay_trace,
)

MODEL_ID = "ibm-granite/granite-3.0-1b-a400m-instruct"
MODEL_REVISION = "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
EXPECTATIONS_COMMITS = {
    "original": "1fee0891dc127da91c2e75a10da1151164ae3d7f",
    "integration_review_amendment": "24116f1aedafb11ad9dc6698d8d70eeefde85cfb",
}


def frozen_requests() -> tuple[PreplayRequest, ...]:
    return (
        PreplayRequest(
            request_id="eos-brief",
            prompt="Reply with exactly one word: OK",
            max_new_tokens=16,
        ),
        PreplayRequest(
            request_id="length-cap",
            prompt="Continue this sequence with ten more integers: 1 2 3",
            max_new_tokens=1,
        ),
        PreplayRequest(
            request_id="stop-string",
            prompt="Reply with exactly SIMLLM_STOP and no other text",
            max_new_tokens=16,
            stop_strings=("SIMLLM_STOP",),
        ),
    )


def _capture(
    runner: TransformersCpuRunner,
    requests: tuple[PreplayRequest, ...],
    path: Path,
    sampling: SamplingConfig,
) -> float:
    started = time.perf_counter()
    runner.capture(requests, path, sampling=sampling, overwrite=True)
    return time.perf_counter() - started


def _round_trip(path: Path) -> tuple[bool, Path]:
    trace = read_preplay_trace(path)
    target = path.with_name(f"{path.stem}-roundtrip.jsonl")
    write_preplay_trace(target, trace.provenance, trace.requests, overwrite=True)
    return path.read_bytes() == target.read_bytes(), target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _structural_check(trace) -> dict[str, object]:
    provenance = trace.provenance
    expected_layers = tuple(range(24))
    request_ids = tuple(request.request_id for request in trace.requests)
    output_token_count = sum(len(request.output_token_ids) for request in trace.requests)
    prefill_forward_token_count = sum(
        len(request.prefill_tokens) for request in trace.requests
    )
    decode_forward_token_count = sum(
        len(request.decode_tokens) for request in trace.requests
    )
    layer_route_count = 0
    max_weight_sum_error = 0.0
    expert_ids: list[int] = []
    valid = (
        provenance.top_k == 8
        and provenance.expert_count == 32
        and provenance.moe_layer_indices == expected_layers
        and len(request_ids) == len(set(request_ids))
    )
    for request in trace.requests:
        valid = valid and tuple(
            (token.phase.value, token.token_index, token.token_id)
            for token in request.prefill_tokens
        ) == tuple(
            ("prefill", token_index, token_id)
            for token_index, token_id in enumerate(request.input_token_ids)
        )
        valid = valid and tuple(
            (token.phase.value, token.token_index, token.token_id)
            for token in request.decode_tokens
        ) == tuple(
            ("decode", token_index, token_id)
            for token_index, token_id in enumerate(request.output_token_ids[:-1])
        )
        for token in (*request.prefill_tokens, *request.decode_tokens):
            valid = valid and len(token.routing) == 24
            layer_route_count += len(token.routing)
            for route in token.routing:
                valid = valid and len(route.expert_ids) == 8
                valid = valid and len(set(route.expert_ids)) == 8
                valid = valid and len(route.gate_weights) == 8
                valid = valid and all(0 <= expert_id < 32 for expert_id in route.expert_ids)
                valid = valid and all(
                    math.isfinite(weight) and weight >= 0.0
                    for weight in route.gate_weights
                )
                weight_sum_error = abs(sum(route.gate_weights) - 1.0)
                max_weight_sum_error = max(max_weight_sum_error, weight_sum_error)
                valid = valid and weight_sum_error <= 1e-5
                expert_ids.extend(route.expert_ids)
    return {
        "passed": valid,
        "request_count": len(trace.requests),
        "output_token_count": output_token_count,
        "prefill_forward_token_count": prefill_forward_token_count,
        "decode_forward_token_count": decode_forward_token_count,
        "forward_token_count": (
            prefill_forward_token_count + decode_forward_token_count
        ),
        "layer_route_count": layer_route_count,
        "max_weight_sum_error": max_weight_sum_error,
        "expert_id_min": min(expert_ids),
        "expert_id_max": max(expert_ids),
        "top_k": provenance.top_k,
        "expert_count": provenance.expert_count,
        "moe_layer_count": len(provenance.moe_layer_indices),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requests = frozen_requests()
    runner = TransformersCpuRunner(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=args.cache_dir,
        dtype="float32",
        torch_num_threads=args.torch_threads,
    )

    paths = {
        "greedy": args.output_dir / "granite-greedy.jsonl",
        "sampled_a": args.output_dir / "granite-sampled-a.jsonl",
        "sampled_b": args.output_dir / "granite-sampled-b.jsonl",
    }
    durations = {
        "greedy": _capture(runner, requests, paths["greedy"], SamplingConfig.greedy()),
        "sampled_a": _capture(
            runner,
            requests,
            paths["sampled_a"],
            SamplingConfig.seeded(173, temperature=0.8, top_p=0.9),
        ),
        "sampled_b": _capture(
            runner,
            requests,
            paths["sampled_b"],
            SamplingConfig.seeded(173, temperature=0.8, top_p=0.9),
        ),
    }
    traces = {name: read_preplay_trace(path) for name, path in paths.items()}

    b1 = paths["sampled_a"].read_bytes() == paths["sampled_b"].read_bytes()
    greedy_provenance = traces["greedy"].provenance
    sampled_provenance = traces["sampled_a"].provenance
    b2 = (
        greedy_provenance.sampling.mode is SamplingMode.GREEDY
        and greedy_provenance.sampling.seed is None
        and greedy_provenance.sampling.temperature is None
        and greedy_provenance.sampling.top_p is None
        and sampled_provenance.sampling
        == SamplingConfig.seeded(173, temperature=0.8, top_p=0.9)
        and replace(sampled_provenance, sampling=greedy_provenance.sampling)
        == greedy_provenance
    )
    greedy_by_id = {
        request.request_id: request for request in traces["greedy"].requests
    }
    b3 = (
        greedy_by_id["eos-brief"].stop_reason is StopReason.EOS
        and greedy_by_id["eos-brief"].output_token_ids[-1] == 0
        and greedy_by_id["length-cap"].stop_reason is StopReason.LENGTH_CAP
        and len(greedy_by_id["length-cap"].output_token_ids) == 1
        and greedy_by_id["stop-string"].stop_reason is StopReason.STOP_STRING
        and greedy_by_id["stop-string"].matched_stop_string == "SIMLLM_STOP"
    )
    round_trips = {name: _round_trip(path) for name, path in paths.items()}
    e1 = all(passed for passed, _ in round_trips.values())
    structural = {name: _structural_check(trace) for name, trace in traces.items()}

    if args.fixture_output is not None:
        write_preplay_trace(
            args.fixture_output,
            greedy_provenance,
            (greedy_by_id["length-cap"],),
            overwrite=True,
        )

    summary = {
        "expectations_commits": EXPECTATIONS_COMMITS,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime": {
            "transformers_version": greedy_provenance.transformers_version,
            "torch_version": greedy_provenance.torch_version,
            "dtype": greedy_provenance.dtype,
            "device": greedy_provenance.device,
            "torch_num_threads": greedy_provenance.torch_num_threads,
        },
        "captures": {
            name: {
                "path": str(path),
                "duration_seconds": durations[name],
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "behavioral_checks": {
            "B1_seeded_determinism": {"passed": b1},
            "B2_sampling_provenance": {
                "passed": b2,
                "greedy_output_token_ids": {
                    request.request_id: list(request.output_token_ids)
                    for request in traces["greedy"].requests
                },
                "sampled_output_token_ids": {
                    request.request_id: list(request.output_token_ids)
                    for request in traces["sampled_a"].requests
                },
            },
            "B3_stop_semantics": {
                "passed": b3,
                "observed": {
                    request.request_id: {
                        "stop_reason": request.stop_reason.value,
                        "output_token_count": len(request.output_token_ids),
                        "matched_stop_string": request.matched_stop_string,
                    }
                    for request in traces["greedy"].requests
                },
            },
        },
        "exact_oracle_checks": {
            "E1_schema_round_trip": {
                "passed": e1,
                "instances": {
                    name: {
                        "passed": passed,
                        "roundtrip_path": str(roundtrip_path),
                        "sha256": _sha256(roundtrip_path),
                    }
                    for name, (passed, roundtrip_path) in round_trips.items()
                },
            }
        },
        "structural_checks_unscored": structural,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    all_passed = (
        b1
        and b2
        and b3
        and e1
        and all(result["passed"] for result in structural.values())
    )
    if not all_passed:
        raise RuntimeError(f"PLAY-1 study failed; inspect {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    hf_home = os.environ.get("HF_HOME")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(hf_home).expanduser() if hf_home else None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
    )
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--fixture-output", type=Path)
    args = parser.parse_args()
    if args.cache_dir is None:
        parser.error("--cache-dir is required when HF_HOME is not set")
    if args.output_dir is None:
        data_root = path_from_env("SIMLLM_DATA_ROOT")
        if data_root is None:
            parser.error("--output-dir is required when SIMLLM_DATA_ROOT is not set")
        args.output_dir = data_root / "preplay_trace_v1"
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
