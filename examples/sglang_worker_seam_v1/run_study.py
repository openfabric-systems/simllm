"""Run the frozen SGLang worker-seam sampled-count study.

The ``--check-only`` gate re-derives every frozen literal in
``expectations.json`` from the physical constants and row shapes it also
freezes; it imports no simllm code and produces no artifacts. It ran, and
passed, before the expectations commit.

The measuring half replays one stub SGLang batch stream per cell through the
adapter's own observation and translation seam and then through the shared
metric chain, once with the exact sampled identity and once on the explicit
compatibility path::

    stub ScheduleBatch -> observe_schedule_batch -> SglStepTranslator
      -> StepRecord -> ObservedStepLowerer -> ExecutionGraph
      -> CoarseDeviceRuntime -> CompletionReducer -> StepResult -> TTFT, TPOT

Run it with an explicit ``--run-dir``; nothing is written anywhere else.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
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


# Stub SGLang batches: the pinned attribute names, nothing else.


class StubForwardMode:
    def __init__(self, mode: str) -> None:
        self._mode = mode

    def is_extend(self) -> bool:
        return self._mode in ("extend", "mixed")

    def is_decode(self) -> bool:
        return self._mode == "decode"

    def is_idle(self) -> bool:
        return self._mode == "idle"


@dataclass
class StubReq:
    """Shaped like ``sglang.srt.managers.schedule_batch.Req``."""

    rid: str
    output_ids: list = field(default_factory=list)
    origin_input_ids: list = field(default_factory=list)
    inflight_middle_chunks: int = 0
    is_retracted: bool = False
    finished_reason: object | None = None
    cached_tokens: int = 0

    def finished(self) -> bool:
        return self.finished_reason is not None


@dataclass
class StubScheduleBatch:
    """Shaped like ``sglang.srt.managers.schedule_batch.ScheduleBatch``."""

    reqs: list
    forward_mode: StubForwardMode
    seq_lens_cpu: list
    extend_lens: list
    decoding_reqs: list | None = None
    return_logprob: bool = False
    device: str = "cpu"


def build_batches(cell: dict[str, Any]) -> list[StubScheduleBatch]:
    """Rebuild one cell's SGLang batch stream from its frozen row shapes.

    The chunked request `R` carries a positive ``inflight_middle_chunks`` on
    every chunk before the last, exactly as SGLang's scheduler leaves it, so
    the sampled rows are derived by the adapter and never declared here.
    """

    chunks = cell["chunks"]
    steps = cell["rows"]
    batches: list[StubScheduleBatch] = []
    for index, step in enumerate(steps):
        is_decode_step = index == len(steps) - 1
        reqs: list[StubReq] = []
        decoding: list[StubReq] = []
        for request_id, new_tokens, context in step:
            if request_id == "R":
                produced = 1 if is_decode_step else 0
                req = StubReq(
                    rid="R",
                    origin_input_ids=[0] * cell["fixture"]["prompt_tokens"],
                    output_ids=[0] * produced,
                    inflight_middle_chunks=0 if index >= chunks - 1 else 1,
                )
            elif request_id == "D":
                # the decode companion has already produced index tokens
                req = StubReq(
                    rid="D",
                    origin_input_ids=[0] * (context - index - 1),
                    output_ids=[0] * index,
                )
                decoding.append(req)
            else:
                req = StubReq(rid=request_id, origin_input_ids=[0] * context)
            reqs.append(req)
        if is_decode_step:
            mode = StubForwardMode("decode")
        else:
            mode = StubForwardMode("mixed" if decoding else "extend")
        batches.append(
            StubScheduleBatch(
                reqs=reqs,
                forward_mode=mode,
                seq_lens_cpu=[context for _, _, context in step],
                extend_lens=[new_tokens for _, new_tokens, _ in step],
                decoding_reqs=decoding or None,
            )
        )
    return batches


def run_arm(cell: dict[str, Any], *, sample_identity: bool) -> dict[str, Any]:
    """Replay one cell's batch stream through the full metric chain once."""

    from simllm.adapters.sglang import SglStepTranslator, observe_schedule_batch
    from simllm.backends.device_step_sink import DeviceRuntimeStepSink
    from simllm.backends.step_lowerer import SerialStepLowererConfig
    from simllm.compute import GpuSpec, HostInitiationModel, ModelDims, RooflineProvider
    from simllm.core import VirtualClock, step_record_to_json

    fixture = cell["fixture"]
    dims = ModelDims(**fixture["model_dims"])
    gpu = GpuSpec(
        name=fixture["gpu_name"],
        peak_flops=float(fixture["peak_flops"]),
        mem_bandwidth=float(cell["bandwidth_bytes_per_s"]),
    )
    sink = DeviceRuntimeStepSink(
        SerialStepLowererConfig(
            dims=dims,
            tp_ranks=tuple(fixture["tp_ranks"]),
            provider=RooflineProvider(efficiency=float(Fraction(fixture["efficiency"]))),
            gpu=gpu,
            host_model=HostInitiationModel.ideal(),
        )
    )
    clock = VirtualClock()
    sink.bind_clock(clock)
    translator = SglStepTranslator(sample_identity=sample_identity)

    step_latency_ps: list[int] = []
    step_completed_at_ps: list[int] = []
    sampling_steps: dict[str, list[int]] = {}
    metrics: dict[str, list[dict[str, Any]]] = {}
    record_keys: list[list[str]] = []
    for index, batch in enumerate(build_batches(cell)):
        rows = observe_schedule_batch(batch)
        record = translator.translate(
            step_index=index,
            virtual_time_ps=clock.now_ps,
            rows=rows,
        )
        record_keys.append(sorted(step_record_to_json(record)))
        result = sink(record, None)
        step_latency_ps.append(result.step_latency_ps)
        step_completed_at_ps.append(result.completed_at_ps)
        for metric in result.request_metrics:
            sampling_steps.setdefault(metric.request_id, []).append(index)
            metrics.setdefault(metric.request_id, []).append(
                {
                    "token_index": metric.token_index,
                    "completed_at_ps": metric.completed_at_ps,
                    "ttft_ps": metric.ttft_ps,
                    "tpot_ps": None if metric.tpot_ps is None else str(metric.tpot_ps),
                }
            )
    return {
        "step_latency_ps": step_latency_ps,
        "step_completed_at_ps": step_completed_at_ps,
        "sampling_steps": sampling_steps,
        "metrics": metrics,
        "record_keys": record_keys,
    }


def _row(rows: list[dict[str, Any]], name: str, expected: Any, observed: Any) -> bool:
    passed = expected == observed
    rows.append(
        {"row": name, "expected": expected, "observed": observed, "passed": passed}
    )
    return passed


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    """Run the frozen cells after the expectations commit."""

    document = load_expectations()
    fixture = dict(document["fixture"])
    fixture["model_dims"] = document["model_dims"]
    fixture["gpu_name"] = "study-roofline"
    scored: dict[str, list[dict[str, Any]]] = {
        "E1_step_latency_ps": [],
        "E2_request_ttft_ps": [],
        "E3_control_ttft_ps": [],
    }
    entailed: list[dict[str, Any]] = []
    guards: list[dict[str, Any]] = []
    observations: dict[str, Any] = {}

    for name, cell in sorted(document["cells"].items()):
        arms = {}
        for arm, sample_identity in (("compat", False), ("fixed", True)):
            arms[arm] = run_arm({**cell, "fixture": fixture}, sample_identity=sample_identity)
        observations[name] = arms

        # E1: the step latency of every step, scored once per cell.
        for index, expected in enumerate(cell["step_latency_ps"]):
            _row(
                scored["E1_step_latency_ps"],
                f"{name}:step{index}",
                expected,
                arms["fixed"]["step_latency_ps"][index],
            )
        # E2 and E3: which completion each first token is attributed to.
        request = cell["request_R"]
        for arm in ("compat", "fixed"):
            first = arms[arm]["metrics"]["R"][0]
            _row(
                scored["E2_request_ttft_ps"],
                f"{name}:R:{arm}",
                request[f"{arm}_ttft_ps"],
                first["ttft_ps"],
            )
        control = cell["control_request"]
        if control is not None:
            for arm in ("compat", "fixed"):
                first = arms[arm]["metrics"][control["request_id"]][0]
                _row(
                    scored["E3_control_ttft_ps"],
                    f"{name}:{control['request_id']}:{arm}",
                    control["ttft_ps"],
                    first["ttft_ps"],
                )

        # Entailed conformance: signatures, token counts, TPOT, error, ratios.
        for arm in ("compat", "fixed"):
            _row(
                entailed,
                f"{name}:R:{arm}:sampling-steps",
                request[f"{arm}_sampling_steps"],
                arms[arm]["sampling_steps"]["R"],
            )
            _row(
                entailed,
                f"{name}:R:{arm}:token-count",
                request[f"{arm}_token_count"],
                arms[arm]["metrics"]["R"][-1]["token_index"],
            )
            _row(
                entailed,
                f"{name}:R:{arm}:tpot",
                request[f"{arm}_tpot_ps"],
                arms[arm]["metrics"]["R"][-1]["tpot_ps"],
            )
            if control is not None:
                _row(
                    entailed,
                    f"{name}:{control['request_id']}:{arm}:sampling-steps",
                    control["sampling_steps"],
                    arms[arm]["sampling_steps"][control["request_id"]],
                )
        _row(
            entailed,
            f"{name}:R:ttft-error",
            request["ttft_error_ps"],
            arms["fixed"]["metrics"]["R"][0]["ttft_ps"]
            - arms["compat"]["metrics"]["R"][0]["ttft_ps"],
        )

        # G1: the two arms must price every step identically.
        guards.append(
            {
                "guard": "G1",
                "cell": name,
                "held": arms["compat"]["step_latency_ps"] == arms["fixed"]["step_latency_ps"],
            }
        )
        # G4: the compatibility record carries neither field, the fixed one both.
        compat_clean = all(
            "num_sampled" not in keys and "sampled_request_ids" not in keys
            for keys in arms["compat"]["record_keys"]
        )
        fixed_complete = all(
            "num_sampled" in keys and "sampled_request_ids" in keys
            for keys in arms["fixed"]["record_keys"]
        )
        guards.append(
            {"guard": "G4", "cell": name, "held": compat_clean and fixed_complete}
        )
        # G7: both arms consumed the frozen number of steps from time zero.
        guards.append(
            {
                "guard": "G7",
                "cell": name,
                "held": all(
                    len(arms[arm]["step_latency_ps"]) == len(cell["rows"])
                    and arms[arm]["step_completed_at_ps"][-1]
                    == sum(arms[arm]["step_latency_ps"])
                    for arm in ("compat", "fixed")
                ),
            }
        )

    for name, fast_cell in sorted(document["cells"].items()):
        if not name.endswith("-fast"):
            continue
        slow_name = name[: -len("fast")] + "slow"
        fast_ttft = observations[name]["fixed"]["metrics"]["R"][0]["ttft_ps"]
        slow_ttft = observations[slow_name]["fixed"]["metrics"]["R"][0]["ttft_ps"]
        _row(
            entailed,
            f"{name}:R:bandwidth-ratio-in-band",
            True,
            2 * fast_ttft <= slow_ttft <= 2 * fast_ttft + len(fast_cell["rows"]) * 32_000,
        )

    summary = {
        "scored": {
            key: {
                "passed": sum(1 for row in rows if row["passed"]),
                "total": len(rows),
            }
            for key, rows in scored.items()
        },
        "entailed_conformance": {
            "passed": sum(1 for row in entailed if row["passed"]),
            "total": len(entailed),
        },
        "guards_violated": [guard for guard in guards if not guard["held"]],
    }
    summary["scored_total"] = {
        "passed": sum(value["passed"] for value in summary["scored"].values()),
        "total": sum(value["total"] for value in summary["scored"].values()),
    }
    result = {
        "schema": "simllm-sglang-worker-seam-results-v1",
        "summary": summary,
        "scored_rows": scored,
        "entailed_rows": entailed,
        "guards": guards,
        "observations": observations,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="directory for run artifacts")
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.check_only and arguments.run_dir is None:
        parser.error("--run-dir is required for a measuring run")
    return arguments


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
