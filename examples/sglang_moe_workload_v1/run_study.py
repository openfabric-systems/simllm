"""Audit the frozen SGLang MoE geometry and workload-driver expectations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from simllm.adapters.sglang import model_dims_from_sglang, sglang_generate_payload
from simllm.compute import GpuSpec, RooflineProvider, step_kernel
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.workload import (
    FixedLengths,
    GenerationRequest,
    HashedTokenPrompts,
    PoissonArrivals,
    TraceArrivals,
    TraceLengths,
    TransportRequestObservation,
    realize_generation_requests,
    reduce_transport_observations,
)

GEOMETRY_CELLS: dict[str, dict[str, Any]] = {
    "granite-pilot": {
        "model_type": "granitemoe",
        "architectures": ["GraniteMoeForCausalLM"],
        "intermediate_size": 512,
        "num_local_experts": 32,
        "num_experts_per_tok": 8,
        "expected": (32, 8, 512, 32),
    },
    "mixtral": {
        "model_type": "mixtral",
        "architectures": ["MixtralForCausalLM"],
        "intermediate_size": 14336,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "expected": (8, 2, 14336, 8),
    },
    "qwen3-moe": {
        "model_type": "qwen3_moe",
        "architectures": ["Qwen3MoeForCausalLM"],
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 1408,
        "expected": (64, 8, 1408, 64),
    },
    "llama-dense-tp4": {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "intermediate_size": 14336,
        "tp_size": 4,
        "expected": (0, 0, None, 0),
    },
}

#: Cells whose expected geometry is forced by the configuration rather than
#: measured. A dense model cannot report routed-MoE fields, so its all-zero
#: tuple cannot fail for any correct reader. AGENTS.md keeps these fatal when
#: violated and out of every behavioral denominator.
CONFIGURATION_FORCED_GEOMETRY_CELLS = frozenset({"llama-dense-tp4"})


def _model_config(**overrides: Any) -> Any:
    fields = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "vocab_size": 49152,
    }
    fields.update(overrides)
    hf = SimpleNamespace(**fields)
    config = SimpleNamespace(
        hf_text_config=hf,
        num_hidden_layers=24,
        num_attention_heads=hf.num_attention_heads,
        head_dim=hf.hidden_size // hf.num_attention_heads,
        vocab_size=hf.vocab_size,
        dtype=SimpleNamespace(itemsize=2),
    )
    config.get_num_kv_heads = lambda tp: max(hf.num_key_value_heads // tp, 1)
    return config


def _granite_config(**overrides: Any) -> Any:
    fields = {
        "model_type": "granitemoe",
        "architectures": ["GraniteMoeForCausalLM"],
        "intermediate_size": 512,
        "num_local_experts": 32,
        "num_experts_per_tok": 8,
    }
    fields.update(overrides)
    return _model_config(**fields)


def _guard(
    name: str,
    operation: Callable[[], Any],
    expected_exception: type[BaseException],
) -> dict[str, Any]:
    try:
        operation()
    except expected_exception as exc:
        return {"guard": name, "passed": True, "detail": str(exc)}
    except BaseException as exc:  # noqa: BLE001 - the audit records wrong failures
        return {
            "guard": name,
            "passed": False,
            "detail": f"raised {type(exc).__name__}: {exc}",
        }
    return {
        "guard": name,
        "passed": False,
        "detail": f"did not raise {expected_exception.__name__}",
    }


def _geometry_results() -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any]:
    scored: list[dict[str, Any]] = []
    forced_guards: list[dict[str, Any]] = []
    for name, raw_cell in GEOMETRY_CELLS.items():
        cell = dict(raw_cell)
        expected = cell.pop("expected")
        tp_size = cell.pop("tp_size", 1)
        dims = model_dims_from_sglang(_model_config(**cell), tp_size=tp_size)
        actual = (
            dims.num_experts,
            dims.top_k,
            dims.moe_intermediate_size,
            dims.local_num_experts,
        )
        row = {
            "family": "geometry",
            "cell": name,
            "passed": actual == expected,
            "expected": list(expected),
            "actual": list(actual),
        }
        if name in CONFIGURATION_FORCED_GEOMETRY_CELLS:
            # A dense config cannot report routed-MoE geometry, so this cell's
            # all-zero tuple is configuration-forced. AGENTS.md keeps such an
            # assertion fatal when violated and unscored otherwise, and the same
            # claim is already asserted by the dense-identity guard below.
            forced_guards.append(
                {
                    "guard": f"geometry-configuration-forced-{name}",
                    "passed": row["passed"],
                    "detail": f"expected {list(expected)}, observed {list(actual)}",
                }
            )
        else:
            scored.append(row)

    dense = model_dims_from_sglang(_model_config(intermediate_size=512))
    granite = model_dims_from_sglang(_granite_config())
    guards = [
        _guard(
            "geometry-unknown-moe-rejected",
            lambda: model_dims_from_sglang(
                _model_config(
                    model_type="custom_moe",
                    architectures=["CustomMoeForCausalLM"],
                    num_experts=8,
                    num_experts_per_tok=2,
                    moe_intermediate_size=512,
                )
            ),
            NotImplementedError,
        ),
        _guard(
            "geometry-missing-field-rejected",
            lambda: model_dims_from_sglang(
                _model_config(
                    model_type="granitemoe",
                    architectures=["GraniteMoeForCausalLM"],
                    intermediate_size=512,
                    num_local_experts=32,
                )
            ),
            ValueError,
        ),
        _guard(
            "geometry-noninteger-field-rejected",
            lambda: model_dims_from_sglang(_granite_config(num_local_experts=32.0)),
            ValueError,
        ),
        _guard(
            "geometry-nonpositive-field-rejected",
            lambda: model_dims_from_sglang(_granite_config(num_local_experts=0)),
            ValueError,
        ),
        _guard(
            "geometry-disagreeing-alias-rejected",
            lambda: model_dims_from_sglang(_granite_config(num_experts=16)),
            ValueError,
        ),
        _guard(
            "geometry-top-k-lower-bound",
            lambda: model_dims_from_sglang(_granite_config(num_experts_per_tok=0)),
            ValueError,
        ),
        _guard(
            "geometry-top-k-upper-bound",
            lambda: model_dims_from_sglang(_granite_config(num_experts_per_tok=33)),
            ValueError,
        ),
        _guard(
            "geometry-shared-expert-rejected",
            lambda: model_dims_from_sglang(
                _granite_config(shared_expert_intermediate_size=1024)
            ),
            NotImplementedError,
        ),
        _guard(
            "geometry-mixed-layer-rejected",
            lambda: model_dims_from_sglang(_granite_config(decoder_sparse_step=2)),
            NotImplementedError,
        ),
        _guard(
            "geometry-mla-rejected",
            lambda: model_dims_from_sglang(_granite_config(use_mla=True)),
            NotImplementedError,
        ),
        _guard(
            "geometry-next-token-layer-rejected",
            lambda: model_dims_from_sglang(
                _granite_config(num_nextn_predict_layers=1)
            ),
            NotImplementedError,
        ),
        _guard(
            "geometry-distributed-rejected",
            lambda: model_dims_from_sglang(
                _granite_config(), tp_size=2, moe_tp_size=2
            ),
            NotImplementedError,
        ),
        _guard(
            "geometry-expert-parallel-rejected",
            lambda: model_dims_from_sglang(_granite_config(), moe_ep_size=2),
            NotImplementedError,
        ),
        _guard(
            "geometry-moe-data-parallel-rejected",
            lambda: model_dims_from_sglang(_granite_config(), moe_dp_size=2),
            NotImplementedError,
        ),
        {
            "guard": "geometry-dense-identity-and-granite-ratios",
            "passed": (
                dense.num_experts == 0
                and dense.defaulted_fields == ()
                and granite.mlp_active_params == 8 * dense.mlp_active_params
                and granite.mlp_resident_params == 32 * dense.mlp_resident_params
            ),
            "detail": {
                "dense_num_experts": dense.num_experts,
                "dense_defaulted_fields": list(dense.defaulted_fields),
                "active_ratio": granite.mlp_active_params // dense.mlp_active_params,
                "resident_ratio": (
                    granite.mlp_resident_params // dense.mlp_resident_params
                ),
            },
        },
    ]
    return scored, forced_guards + guards, granite


def _write_inputs(run_dir: Path) -> dict[str, Path]:
    paths = {
        "arrivals": run_dir / "arrivals.txt",
        "short_arrivals": run_dir / "short_arrivals.txt",
        "prompt_lengths": run_dir / "prompt_lengths.txt",
        "short_lengths": run_dir / "short_lengths.txt",
        "output_lengths": run_dir / "output_lengths.txt",
    }
    paths["arrivals"].write_text("0\n1.5e-12\n1e-6\n", encoding="utf-8")
    paths["short_arrivals"].write_text("0\n1e-6\n", encoding="utf-8")
    paths["prompt_lengths"].write_text("3\n7\n11\n", encoding="utf-8")
    paths["short_lengths"].write_text("3\n7\n", encoding="utf-8")
    paths["output_lengths"].write_text("3\n1\n2\n", encoding="utf-8")
    return paths


def _realize_cell(
    paths: dict[str, Path], arrival_name: str, length_name: str, *, prompt_seed: int = 17
) -> tuple[GenerationRequest, ...]:
    arrivals = (
        TraceArrivals(paths["arrivals"])
        if arrival_name == "trace"
        else PoissonArrivals(rate_rps=100, seed=31)
    )
    if length_name == "fixed":
        prompt_lengths = FixedLengths(8)
        output_lengths = FixedLengths(3)
    else:
        prompt_lengths = TraceLengths(paths["prompt_lengths"])
        output_lengths = TraceLengths(paths["output_lengths"])
    return realize_generation_requests(
        ("r0", "r1", "r2"),
        arrivals=arrivals,
        prompt_lengths=prompt_lengths,
        output_lengths=output_lengths,
        prompt_builder=HashedTokenPrompts(
            vocab_size=49152, seed=prompt_seed, first_token_id=10
        ),
    )


def _workload_results(
    paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    payload_guards: list[dict[str, Any]] = []
    for arrival_name in ("trace", "poisson"):
        for length_name in ("fixed", "trace"):
            name = f"{arrival_name}-{length_name}"
            first = _realize_cell(paths, arrival_name, length_name)
            repeated = _realize_cell(paths, arrival_name, length_name)
            changed = _realize_cell(paths, arrival_name, length_name, prompt_seed=18)
            trace_quantization_ok = arrival_name != "trace" or tuple(
                item.arrived_at_ps for item in first
            ) == (0, 2, 1_000_000)
            stable_shape = all(
                (
                    before.request_id,
                    before.arrived_at_ps,
                    len(before.prompt_token_ids),
                    before.output_token_count,
                )
                == (
                    after.request_id,
                    after.arrived_at_ps,
                    len(after.prompt_token_ids),
                    after.output_token_count,
                )
                for before, after in zip(first, changed, strict=True)
            )
            tokens_changed = any(
                before.prompt_token_ids != after.prompt_token_ids
                for before, after in zip(first, changed, strict=True)
            )
            scored.append(
                {
                    "family": "workload",
                    "cell": name,
                    "passed": (
                        first == repeated
                        and trace_quantization_ok
                        and stable_shape
                        and tokens_changed
                    ),
                    "arrivals_ps": [item.arrived_at_ps for item in first],
                    "prompt_lengths": [len(item.prompt_token_ids) for item in first],
                    "output_lengths": [item.output_token_count for item in first],
                }
            )
            for item in first:
                payload = sglang_generate_payload(item)
                payload_guards.append(
                    {
                        "guard": f"payload-shape[{name}/{item.request_id}]",
                        "passed": payload
                        == {
                            "rid": item.request_id,
                            "input_ids": list(item.prompt_token_ids),
                            "sampling_params": {
                                "temperature": 0.0,
                                "max_new_tokens": item.output_token_count,
                                "ignore_eos": True,
                            },
                            "stream": True,
                            "return_logprob": False,
                        },
                        "detail": payload,
                    }
                )

    short_arrival_guard = _guard(
        "workload-short-arrival-trace-rejected",
        lambda: realize_generation_requests(
            ("r0", "r1", "r2"),
            arrivals=TraceArrivals(paths["short_arrivals"]),
            prompt_lengths=FixedLengths(1),
            output_lengths=FixedLengths(1),
            prompt_builder=HashedTokenPrompts(16),
        ),
        ValueError,
    )
    short_length_guard = _guard(
        "workload-short-length-trace-rejected",
        lambda: realize_generation_requests(
            ("r0", "r1", "r2"),
            arrivals=TraceArrivals(paths["arrivals"]),
            prompt_lengths=TraceLengths(paths["short_lengths"]),
            output_lengths=FixedLengths(1),
            prompt_builder=HashedTokenPrompts(16),
        ),
        ValueError,
    )
    workload_guards = [
        short_arrival_guard,
        short_length_guard,
        _guard(
            "workload-duplicate-id-rejected",
            lambda: realize_generation_requests(
                ("r0", "r0"),
                arrivals=TraceArrivals(paths["short_arrivals"]),
                prompt_lengths=FixedLengths(1),
                output_lengths=FixedLengths(1),
                prompt_builder=HashedTokenPrompts(16),
            ),
            ValueError,
        ),
        _guard(
            "workload-decreasing-arrival-rejected",
            lambda: realize_generation_requests(
                ("r0", "r1"),
                arrivals=SimpleNamespace(times=lambda count: (1, 0)),
                prompt_lengths=FixedLengths(1),
                output_lengths=FixedLengths(1),
                prompt_builder=HashedTokenPrompts(16),
            ),
            ValueError,
        ),
        _guard(
            "workload-nonpositive-length-rejected",
            lambda: realize_generation_requests(
                ("r0",),
                arrivals=SimpleNamespace(times=lambda count: (0,)),
                prompt_lengths=SimpleNamespace(sample=lambda count: (0,)),
                output_lengths=FixedLengths(1),
                prompt_builder=HashedTokenPrompts(16),
            ),
            ValueError,
        ),
        _guard(
            "workload-prompt-length-rejected",
            lambda: realize_generation_requests(
                ("r0",),
                arrivals=SimpleNamespace(times=lambda count: (0,)),
                prompt_lengths=FixedLengths(2),
                output_lengths=FixedLengths(1),
                prompt_builder=lambda request_id, length, ordinal: (1,),
            ),
            ValueError,
        ),
        _guard(
            "workload-prompt-token-bound-rejected",
            lambda: realize_generation_requests(
                ("r0",),
                arrivals=SimpleNamespace(times=lambda count: (0,)),
                prompt_lengths=FixedLengths(1),
                output_lengths=FixedLengths(1),
                prompt_builder=lambda request_id, length, ordinal: (-1,),
            ),
            ValueError,
        ),
    ]
    return scored, workload_guards + payload_guards


def _metric_results() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requests = (
        GenerationRequest(0, "r0", 0, (1, 2, 3), 3),
        GenerationRequest(1, "r1", 1_000, (4, 5, 6), 1),
    )
    observations = (
        TransportRequestObservation("r1", 1_200, (2_000,)),
        TransportRequestObservation("r0", 100, (1_100, 1_300, 1_500)),
    )
    timings = reduce_transport_observations(requests, observations)
    expected = {
        "r0": (1_100, Fraction(200), 100),
        "r1": (1_000, None, 200),
    }
    scored = [
        {
            "family": "metrics",
            "cell": timing.request_id,
            "passed": (
                timing.ttft_ps,
                timing.tpot_ps,
                timing.submission_lateness_ps,
            )
            == expected[timing.request_id],
            "ttft_ps": timing.ttft_ps,
            "tpot_ps": None
            if timing.tpot_ps is None
            else {
                "numerator": timing.tpot_ps.numerator,
                "denominator": timing.tpot_ps.denominator,
            },
            "submission_lateness_ps": timing.submission_lateness_ps,
        }
        for timing in timings
    ]
    one = (GenerationRequest(0, "r0", 10, (1,), 2),)
    guards = [
        _guard(
            "metrics-missing-observation-rejected",
            lambda: reduce_transport_observations(one, ()),
            ValueError,
        ),
        _guard(
            "metrics-duplicate-observation-rejected",
            lambda: reduce_transport_observations(
                one,
                (
                    TransportRequestObservation("r0", 10, (20, 30)),
                    TransportRequestObservation("r0", 10, (20, 30)),
                ),
            ),
            ValueError,
        ),
        _guard(
            "metrics-foreign-observation-rejected",
            lambda: reduce_transport_observations(
                one, (TransportRequestObservation("foreign", 10, (20, 30)),)
            ),
            ValueError,
        ),
        _guard(
            "metrics-wrong-token-count-rejected",
            lambda: reduce_transport_observations(
                one, (TransportRequestObservation("r0", 10, (20,)),)
            ),
            ValueError,
        ),
        _guard(
            "metrics-submission-before-arrival-rejected",
            lambda: reduce_transport_observations(
                one, (TransportRequestObservation("r0", 9, (20, 30)),)
            ),
            ValueError,
        ),
        _guard(
            "metrics-token-before-submission-rejected",
            lambda: reduce_transport_observations(
                one, (TransportRequestObservation("r0", 20, (19, 30)),)
            ),
            ValueError,
        ),
        _guard(
            "metrics-decreasing-tokens-rejected",
            lambda: reduce_transport_observations(
                one, (TransportRequestObservation("r0", 10, (30, 20)),)
            ),
            ValueError,
        ),
        {
            "guard": "metrics-ordinal-order-restored",
            "passed": tuple(timing.request_id for timing in timings) == ("r0", "r1"),
            "detail": [timing.request_id for timing in timings],
        },
    ]
    return scored, guards


def _component_sanity(granite: Any) -> list[dict[str, Any]]:
    """Post-specified roofline rows, separate from request-metric evidence."""

    rows: list[dict[str, Any]] = []
    provider = RooflineProvider(efficiency=1.0)
    for batch_size in (1, 8):
        record = StepRecord(
            step_index=0,
            virtual_time_ps=0,
            scheduled=[
                ScheduledRequest(
                    request_id=f"r{index}",
                    phase=RequestPhase.DECODE,
                    num_new_tokens=1,
                    context_length=2048,
                )
                for index in range(batch_size)
            ],
            num_sampled=batch_size,
        )
        kernel = step_kernel(granite, record, batch_size)
        for bandwidth in (8.0e12, 4.0e12):
            gpu = GpuSpec(
                name=f"analytical-{bandwidth:.0f}",
                peak_flops=1.8e15,
                mem_bandwidth=bandwidth,
            )
            estimate = provider.estimate(kernel, gpu)
            rows.append(
                {
                    "batch_size": batch_size,
                    "context_length": 2048,
                    "peak_flops": gpu.peak_flops,
                    "mem_bandwidth": bandwidth,
                    "flops": kernel.flops,
                    "bytes_moved": kernel.bytes_moved,
                    "compute_floor_ps": int(kernel.flops / gpu.peak_flops * 1e12),
                    "memory_floor_ps": int(kernel.bytes_moved / bandwidth * 1e12),
                    "roofline_ps": estimate.duration_ps,
                    "bound": estimate.bound,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    paths = _write_inputs(args.run_dir)

    geometry_scored, geometry_guards, granite = _geometry_results()
    workload_scored, workload_guards = _workload_results(paths)
    metric_scored, metric_guards = _metric_results()
    scored = geometry_scored + workload_scored + metric_scored
    guards = geometry_guards + workload_guards + metric_guards
    failures = [row for row in guards if not row["passed"]]
    passed = sum(1 for row in scored if row["passed"])
    status = "VOID" if failures else ("PASS" if passed == len(scored) else "FAIL")
    summary = {
        "study": "sglang_moe_workload_v1",
        "expectations_commit": "c48e785",
        "status": status,
        "scored_passed_retained": passed,
        "scored_total_retained": len(scored),
        "fatal_guards": len(guards),
        "fatal_failures": len(failures),
        "void_reason": (
            "fatal guard failures: "
            + ", ".join(str(row["guard"]) for row in failures)
            if status == "VOID"
            else None
        ),
        "scored": scored,
        "guards": guards,
        "post_specified_component_sanity": _component_sanity(granite),
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "scored_passed_retained",
                    "scored_total_retained",
                    "fatal_guards",
                    "fatal_failures",
                    "status",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
