"""Run the frozen SGLang worker-seam sampled-count study.

The expectations-only version of this file implements only the artifact-free
``--check-only`` gate, which re-derives every frozen literal in
``expectations.json`` from the physical constants and row shapes it also
freezes. It imports no simllm code and produces no artifacts. The
result-producing harness lands after the freeze.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
EXPECTATIONS = STUDY_DIR / "expectations.json"
PS_PER_SECOND = 10**12
SCHEMA = "simllm-sglang-worker-seam-expectations-v1"
#: the smallest distance to a quantization boundary that keeps a predicted
#: step latency immune to the provider's picosecond truncation
MIN_QUANTIZATION_MARGIN_PS = 2


def load_expectations() -> dict[str, Any]:
    document = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        raise SystemExit(f"expectations schema must be {SCHEMA}")
    return document


def derived_geometry(dims: dict[str, int]) -> dict[str, int]:
    """Recompute the frozen geometry from the frozen model dimensions."""

    layers = dims["num_layers"]
    hidden = dims["hidden_size"]
    heads = dims["num_heads"]
    kv_heads = dims["num_kv_heads"]
    head_size = dims["head_size"]
    dtype_bytes = dims["dtype_bytes"]
    query_dim = heads * head_size
    kv_dim = kv_heads * head_size
    attention_params = layers * (hidden * (query_dim + 2 * kv_dim) + query_dim * hidden)
    mlp_params = 3 * hidden * dims["intermediate_size"] * layers
    weight_bytes = (attention_params + mlp_params) * dtype_bytes
    lm_head_bytes = hidden * dims["vocab_size"] * dtype_bytes
    return {
        "attention_params": attention_params,
        "mlp_params": mlp_params,
        "weight_bytes": weight_bytes,
        "lm_head_bytes": lm_head_bytes,
        "base_bytes": weight_bytes + lm_head_bytes,
        "kv_bytes_per_token": 2 * layers * kv_heads * head_size * dtype_bytes,
        "dense_flops_per_new_token": 2 * (attention_params + mlp_params),
        "attention_flops_per_pair": 4 * layers * heads * head_size,
        "head_flops_per_sampled_token": 2 * hidden * dims["vocab_size"],
    }


def expected_rows(document: dict[str, Any], composition: str, chunks: int) -> list[list[list]]:
    """Rebuild one cell's batch rows from the frozen sweep parameters."""

    fixture = document["fixture"]
    sizes = fixture["chunk_sizes"][str(chunks)]
    decode_context = fixture["decode_companion_context"]
    steps: list[list[list]] = []
    context = 0
    for index, size in enumerate(sizes):
        context += size
        step: list[list] = [["R", size, context]]
        if composition == "mixed":
            step.append(["D", 1, decode_context + 1 + index])
        if composition == "paired" and index == 0:
            step.append(["S", fixture["paired_prompt_tokens"], fixture["paired_prompt_tokens"]])
        steps.append(step)
    last: list[list] = [["R", 1, fixture["prompt_tokens"] + 1]]
    if composition == "mixed":
        last.append(["D", 1, decode_context + 1 + len(sizes)])
    steps.append(last)
    return steps


def check_only(document: dict[str, Any]) -> str:
    """Validate the frozen arithmetic without producing artifacts."""

    fixture = document["fixture"]
    if fixture["efficiency"] != "1/2":
        raise SystemExit("frozen roofline efficiency changed")
    efficiency = Fraction(fixture["efficiency"])
    geometry = derived_geometry(document["model_dims"])
    if geometry != document["derived_geometry"]:
        raise SystemExit("frozen derived geometry does not match the frozen dimensions")
    quantum = fixture["quantum_ps"]
    if quantum != document["model_dims"]["num_layers"] * 1000:
        raise SystemExit("frozen quantum must be one whole nanosecond per layer")
    effective_flops = int(fixture["peak_flops"] * efficiency)
    base_bytes = geometry["base_bytes"]
    kv_bytes = geometry["kv_bytes_per_token"]

    scored = {"E1_step_latency_ps": 0, "E2_request_ttft_ps": 0, "E3_control_ttft_ps": 0}
    max_context = 0
    min_bound_ratio = None
    min_margin_ps = None
    for name, cell in sorted(document["cells"].items()):
        composition = cell["composition"]
        chunks = cell["chunks"]
        bandwidth = cell["bandwidth_bytes_per_s"]
        if name != f"{composition}-{chunks}chunk-" + (
            "fast" if bandwidth == max(fixture["bandwidths_bytes_per_s"].values()) else "slow"
        ):
            raise SystemExit(f"cell {name} does not match its own parameters")
        if cell["rows"] != expected_rows(document, composition, chunks):
            raise SystemExit(f"cell {name} rows do not match the frozen sweep parameters")
        effective_bandwidth = int(bandwidth * efficiency)
        completions: list[int] = []
        running = 0
        for index, step in enumerate(cell["rows"]):
            context_total = sum(row[2] for row in step)
            max_context = max(max_context, context_total)
            step_bytes = base_bytes + kv_bytes * context_total
            if step_bytes != cell["step_bytes"][index]:
                raise SystemExit(f"cell {name} step {index} byte count is not the frozen value")
            flops = geometry["head_flops_per_sampled_token"] * len(step)
            for _, new_tokens, context in step:
                prior = max(context - new_tokens, 0)
                flops += geometry["dense_flops_per_new_token"] * new_tokens
                flops += geometry["attention_flops_per_pair"] * (
                    new_tokens * prior + new_tokens * new_tokens // 2
                )
            memory_ps = step_bytes * PS_PER_SECOND // effective_bandwidth
            compute_ps = flops * PS_PER_SECOND // effective_flops
            if memory_ps != cell["roofline_memory_ps"][index]:
                raise SystemExit(f"cell {name} step {index} memory picoseconds changed")
            if compute_ps != cell["roofline_compute_ps"][index]:
                raise SystemExit(f"cell {name} step {index} compute picoseconds changed")
            if compute_ps >= memory_ps:
                raise SystemExit(f"cell {name} step {index} is not memory bound (guard G2)")
            ratio = Fraction(memory_ps, compute_ps)
            min_bound_ratio = ratio if min_bound_ratio is None else min(min_bound_ratio, ratio)
            margin = memory_ps % quantum
            if margin != cell["quantization_margin_ps"][index]:
                raise SystemExit(f"cell {name} step {index} quantization margin changed")
            if margin < MIN_QUANTIZATION_MARGIN_PS:
                raise SystemExit(f"cell {name} step {index} sits on a quantum boundary (guard G3)")
            min_margin_ps = margin if min_margin_ps is None else min(min_margin_ps, margin)
            latency = quantum * (memory_ps // quantum)
            if latency != cell["step_latency_ps"][index]:
                raise SystemExit(f"cell {name} step {index} latency is not the frozen value")
            running += latency
            completions.append(running)
        if completions != cell["step_completed_at_ps"]:
            raise SystemExit(f"cell {name} step completions are not the frozen values")
        scored["E1_step_latency_ps"] += len(cell["step_latency_ps"])

        request = cell["request_R"]
        last_step = len(cell["rows"]) - 1
        if request["compat_sampling_steps"] != list(range(last_step + 1)):
            raise SystemExit(f"cell {name} compatibility sampling steps changed")
        if request["fixed_sampling_steps"] != [chunks - 1, last_step]:
            raise SystemExit(f"cell {name} fixed sampling steps changed")
        for arm in ("compat", "fixed"):
            signature = request[f"{arm}_sampling_steps"]
            if request[f"{arm}_ttft_ps"] != completions[signature[0]]:
                raise SystemExit(f"cell {name} {arm} TTFT is not its first sampling completion")
            if request[f"{arm}_token_count"] != len(signature):
                raise SystemExit(f"cell {name} {arm} token count disagrees with its signature")
            span = completions[signature[-1]] - completions[signature[0]]
            expected_tpot = Fraction(span, len(signature) - 1)
            if Fraction(request[f"{arm}_tpot_ps"]) != expected_tpot:
                raise SystemExit(f"cell {name} {arm} TPOT disagrees with its signature")
        error = request["fixed_ttft_ps"] - request["compat_ttft_ps"]
        if error != request["ttft_error_ps"] or error <= 0:
            raise SystemExit(f"cell {name} TTFT error is not a positive frozen value")
        if error != sum(cell["step_latency_ps"][1:chunks]):
            raise SystemExit(f"cell {name} TTFT error is not the skipped mid-prompt steps")
        scored["E2_request_ttft_ps"] += 2

        control = cell["control_request"]
        if (control is None) != (composition == "solo"):
            raise SystemExit(f"cell {name} control request disagrees with its composition")
        if control is not None:
            expected_steps = (
                list(range(last_step + 1)) if composition == "mixed" else [0]
            )
            if control["sampling_steps"] != expected_steps:
                raise SystemExit(f"cell {name} control sampling steps changed")
            if control["ttft_ps"] != completions[control["sampling_steps"][0]]:
                raise SystemExit(f"cell {name} control TTFT is not its first sampling completion")
            scored["E3_control_ttft_ps"] += 2

    for name, fast_cell in sorted(document["cells"].items()):
        if not name.endswith("-fast"):
            continue
        slow_cell = document["cells"][name[: -len("fast")] + "slow"]
        for fast, slow in zip(fast_cell["step_latency_ps"], slow_cell["step_latency_ps"]):
            if not 2 * fast <= slow <= 2 * fast + quantum:
                raise SystemExit(f"cell {name} does not double under the halved bandwidth")

    frozen = document["scored_rows"]
    if any(frozen[key] != value for key, value in scored.items()):
        raise SystemExit(f"scored row counts changed: recomputed {scored}, frozen {frozen}")
    if frozen["total"] != sum(scored.values()) or frozen["total"] != 82:
        raise SystemExit("the frozen genuine-risk denominator changed")

    floor_ps = base_bytes * PS_PER_SECOND // int(
        max(fixture["bandwidths_bytes_per_s"].values()) * efficiency
    )
    ceiling_ps = (base_bytes + kv_bytes * max_context) * PS_PER_SECOND // int(
        max(fixture["bandwidths_bytes_per_s"].values()) * efficiency
    )
    return (
        f"check-only cells={len(document['cells'])} scored={frozen['total']} "
        f"fast-bandwidth floor={floor_ps} ps ceiling={ceiling_ps} ps "
        f"min memory-to-compute ratio={float(min_bound_ratio):.3f} "
        f"min quantization margin={min_margin_ps} ps; no artifacts produced"
    )


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    """Run the frozen cells after the expectations commit."""

    raise NotImplementedError(
        "the result-producing worker-seam harness lands after the expectations freeze"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="directory for run artifacts")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = load_expectations()
    print(check_only(document))
    if args.check_only:
        return
    result = run_study(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
