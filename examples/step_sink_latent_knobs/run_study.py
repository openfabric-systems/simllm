"""Run the checks frozen for COMP-16 and VLLM-15."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from simllm.adapters.vllm import (
    StepTranslator,
    fabricate_sampled_tokens,
    translate_scheduler_output,
)
from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
from simllm.compute import (
    GPU_ENVELOPES,
    ComputeProvider,
    DurationEstimate,
    KernelSpec,
    ModelDims,
    RooflineProvider,
    step_kernel,
)
from simllm.core import (
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    step_record_from_json,
    step_record_to_json,
)

EXPECTATIONS_COMMIT = "25d098c997f078eb92dcf155cd36c44d9d6b2313"
DEFAULT_OUT = Path("/data3/yifeng/simllm-dev/wave2-runs/comp16_latent_knobs")
MODEL = Path(
    "/home/yifeng/packages/vllm-rnic-capture/hf-cache/hub/"
    "models--ibm-granite--granite-3.0-1b-a400m-instruct/snapshots/"
    "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
)

G = 1_000_000_000
LINKSPEED_BPS = 400 * G
PS_PER_BYTE_400G = 20
PROPAGATION_PS = 2_000_000
DEFAULT_GOAL_SHA256 = "f8aade109ba8e3a581b7d965b3a0c76c1247016a1e37491fa84efbbf377677a5"

FROZEN_A = {
    (2, 2): ((14_811, 20_663), (14, 21), 17, 35_474, 16_044_240, 16_045_240, 16),
    (2, 4): ((14_811, 20_663), (14, 21), 17, 35_474, 48_049_360, 48_050_360, 96),
    (4, 2): (
        (14_811, 14_811, 14_812, 20_663),
        (14, 15, 15, 21),
        16,
        65_097,
        32_084_480,
        32_085_480,
        32,
    ),
    (4, 4): (
        (14_811, 14_811, 14_812, 20_663),
        (14, 15, 15, 21),
        16,
        65_097,
        96_094_720,
        96_095_720,
        192,
    ),
}


class FlopProvider(ComputeProvider):
    """One modeled FLOP costs one picosecond for the sample-count oracle."""

    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=int(kernel.flops), bound="compute")


@dataclass
class SchedulerNewRequest:
    req_id: str
    prompt_token_ids: list[int]
    num_computed_tokens: int = 0


@dataclass
class SchedulerCachedRequests:
    req_ids: list[str] = field(default_factory=list)
    num_computed_tokens: list[int] = field(default_factory=list)
    num_output_tokens: list[int] = field(default_factory=list)


@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[SchedulerNewRequest] = field(default_factory=list)
    scheduled_cached_reqs: SchedulerCachedRequests = field(
        default_factory=SchedulerCachedRequests
    )
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    finished_req_ids: set[str] = field(default_factory=set)
    preempted_req_ids: set[str] | None = None


def dims(num_layers: int) -> ModelDims:
    return ModelDims(
        num_layers=num_layers,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        num_kv_heads=4,
        head_size=16,
        vocab_size=256,
        dtype_bytes=2,
    )


def default_dims() -> ModelDims:
    return ModelDims(
        num_layers=2,
        hidden_size=1024,
        intermediate_size=4096,
        num_heads=8,
        num_kv_heads=8,
        head_size=128,
        vocab_size=32000,
        dtype_bytes=2,
    )


def one_decode() -> StepRecord:
    return StepRecord(
        0,
        0,
        [ScheduledRequest("d", RequestPhase.DECODE, 1, context_length=4)],
        num_sampled=1,
    )


def layer_calcs(path: Path, rank: int) -> tuple[int, ...]:
    text = path.read_text()
    match = re.search(rf"rank {rank} \{{\n(.*?)\n\}}", text, flags=re.DOTALL)
    if match is None:
        raise AssertionError(f"rank {rank} missing from {path}")
    return tuple(int(value) for value in re.findall(r": calc (\d+)(?:\s|$)", match.group(1)))


def frozen_jct(num_layers: int, world: int, calc_ns: tuple[int, ...]) -> int:
    chunk = 128 // world
    round_ps = chunk * PS_PER_BYTE_400G + PROPAGATION_PS
    network_ps = 2 * num_layers * 2 * (world - 1) * round_ps
    return sum(calc_ns) * 1000 + network_ps


def make_sink(
    out: Path,
    label: str,
    *,
    num_layers: int,
    world: int,
    provider: ComputeProvider,
) -> HtsimStepSink:
    return HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=tuple(range(world)),
            dims=dims(num_layers),
            workdir=out / label,
            linkspeed_bps=LINKSPEED_BPS,
            provider=provider,
        )
    )


def run_check_a(out: Path, emit) -> None:
    measured_enabled: dict[tuple[int, int], int] = {}
    for (num_layers, world), frozen in FROZEN_A.items():
        (
            expected_layer_ps,
            expected_calc_ns,
            expected_even_ns,
            expected_estimate_ps,
            expected_disabled_jct,
            expected_enabled_jct,
            expected_flows,
        ) = frozen
        record = one_decode()
        disabled = make_sink(
            out,
            f"a-l{num_layers}-w{world}-disabled",
            num_layers=num_layers,
            world=world,
            provider=RooflineProvider(efficiency=0.7),
        )
        enabled = make_sink(
            out,
            f"a-l{num_layers}-w{world}-enabled",
            num_layers=num_layers,
            world=world,
            provider=RooflineProvider(
                efficiency=0.7,
                enable_layer_breakdown=True,
            ),
        )

        disabled_result = disabled(record)
        enabled_result = enabled(record)
        assert disabled_result is not None
        assert enabled_result is not None
        disabled_outcome = disabled.outcomes[0]
        enabled_outcome = enabled.outcomes[0]
        disabled_calc = (expected_even_ns,) * num_layers

        assert frozen_jct(num_layers, world, disabled_calc) == expected_disabled_jct
        assert frozen_jct(num_layers, world, expected_calc_ns) == expected_enabled_jct
        assert disabled_outcome.compute_estimate_ps == expected_estimate_ps
        assert enabled_outcome.compute_estimate_ps == expected_estimate_ps
        assert disabled_outcome.per_layer_calc_ns == expected_even_ns
        assert disabled_outcome.layer_calc_ns == disabled_calc
        assert enabled_outcome.per_layer_calc_ns is None
        assert enabled_outcome.layer_calc_ns == expected_calc_ns
        assert tuple(
            estimate.duration_ps
            for estimate in enabled.config.provider.estimate_layers(
                step_kernel(dims(num_layers), record, num_sampled=1),
                GPU_ENVELOPES["b100"],
                num_layers,
            )
            or ()
        ) == expected_layer_ps
        assert disabled_result.step_latency_ps == expected_disabled_jct
        assert enabled_result.step_latency_ps == expected_enabled_jct
        assert enabled_result.step_latency_ps - disabled_result.step_latency_ps == 1_000
        assert disabled_outcome.num_flows == expected_flows
        assert enabled_outcome.num_flows == expected_flows
        for rank in range(world):
            assert layer_calcs(
                out / f"a-l{num_layers}-w{world}-disabled" / "step-000000.goal",
                rank,
            ) == disabled_calc
            assert layer_calcs(
                out / f"a-l{num_layers}-w{world}-enabled" / "step-000000.goal",
                rank,
            ) == expected_calc_ns

        measured_enabled[(num_layers, world)] = enabled_result.step_latency_ps
        emit(
            section="A",
            layers=num_layers,
            world=world,
            estimate_ps=enabled_outcome.compute_estimate_ps,
            layer_ps=";".join(map(str, expected_layer_ps)),
            disabled_calc_ns=";".join(map(str, disabled_calc)),
            enabled_calc_ns=";".join(map(str, enabled_outcome.layer_calc_ns)),
            disabled_ttft_ps=disabled_result.step_latency_ps,
            enabled_ttft_ps=enabled_result.step_latency_ps,
            expected_enabled_ttft_ps=expected_enabled_jct,
            signed_delta_ps=enabled_result.step_latency_ps - disabled_result.step_latency_ps,
            residual_ps=enabled_result.step_latency_ps - expected_enabled_jct,
            flows=enabled_outcome.num_flows,
            physical_quiescence="verified",
        )

    assert measured_enabled[(2, 2)] < measured_enabled[(4, 2)]
    assert measured_enabled[(2, 4)] < measured_enabled[(4, 4)]
    assert measured_enabled[(2, 2)] < measured_enabled[(2, 4)]
    assert measured_enabled[(4, 2)] < measured_enabled[(4, 4)]


def translated_mixed_record() -> tuple[StepRecord, list[bool]]:
    translator = StepTranslator()
    output = SchedulerOutput(
        scheduled_new_reqs=[
            SchedulerNewRequest("p", list(range(12)), num_computed_tokens=4)
        ],
        scheduled_cached_reqs=SchedulerCachedRequests(
            req_ids=["d"],
            num_computed_tokens=[31],
            num_output_tokens=[1],
        ),
        num_scheduled_tokens={"p": 4, "d": 1},
    )
    translated = translate_scheduler_output(
        translator,
        output,
        step_index=0,
        virtual_time_ps=0,
    )
    return translated.record, translated.produces_token


def run_check_b1(out: Path, emit) -> None:
    exact_record, produces_token = translated_mixed_record()
    bypass_record = replace(exact_record, num_sampled=None)
    req_ids = [request.request_id for request in exact_record.scheduled]
    _, _, sampled = fabricate_sampled_tokens(req_ids, produces_token, token_id=7)
    assert produces_token == [False, True]
    assert exact_record.num_sampled == 1
    assert sum(bool(row) for row in sampled) == exact_record.num_sampled

    bypass = make_sink(
        out,
        "b1-bypass",
        num_layers=2,
        world=2,
        provider=FlopProvider(),
    )
    exact = make_sink(
        out,
        "b1-adapter-exact",
        num_layers=2,
        world=2,
        provider=FlopProvider(),
    )
    bypass_result = bypass(bypass_record)
    exact_result = exact(exact_record)
    assert bypass_result is not None
    assert exact_result is not None
    bypass_outcome = bypass.outcomes[0]
    exact_outcome = exact.outcomes[0]

    assert bypass_outcome.num_sampled == 2
    assert exact_outcome.num_sampled == 1
    assert bypass_outcome.compute_estimate_ps == 912_896
    assert exact_outcome.compute_estimate_ps == 880_128
    assert bypass_outcome.layer_calc_ns == (456, 456)
    assert exact_outcome.layer_calc_ns == (440, 440)
    assert bypass_result.step_latency_ps == 16_963_200
    assert exact_result.step_latency_ps == 16_931_200
    assert exact_result.step_latency_ps - bypass_result.step_latency_ps == -32_000
    emit(
        section="B1",
        bypass_samples=bypass_outcome.num_sampled,
        adapter_samples=exact_outcome.num_sampled,
        sampled_output_rows=sum(bool(row) for row in sampled),
        bypass_estimate_ps=bypass_outcome.compute_estimate_ps,
        adapter_estimate_ps=exact_outcome.compute_estimate_ps,
        estimate_delta_ps=(
            exact_outcome.compute_estimate_ps - bypass_outcome.compute_estimate_ps
        ),
        bypass_ttft_ps=bypass_result.step_latency_ps,
        adapter_ttft_ps=exact_result.step_latency_ps,
        signed_delta_ps=exact_result.step_latency_ps - bypass_result.step_latency_ps,
        expected_signed_delta_ps=-32_000,
        residual_ps=(exact_result.step_latency_ps - bypass_result.step_latency_ps) + 32_000,
        physical_quiescence="verified",
    )


def check_attribution_case(
    case: str,
    translated,
    expected: int,
    emit,
) -> None:
    _, _, sampled = fabricate_sampled_tokens(
        translated.req_ids,
        translated.produces_token,
        token_id=7,
    )
    output_rows = sum(bool(row) for row in sampled)
    assert translated.record.num_sampled == expected
    assert sum(translated.produces_token) == expected
    assert output_rows == expected
    emit(
        section="B2",
        case=case,
        scheduled_rows=len(translated.record.scheduled),
        record_samples=translated.record.num_sampled,
        output_rows=output_rows,
        expected_samples=expected,
        residual=translated.record.num_sampled - expected,
    )


def run_check_b2(emit) -> None:
    chunked = StepTranslator()
    mid_prompt = translate_scheduler_output(
        chunked,
        SchedulerOutput(
            scheduled_new_reqs=[
                SchedulerNewRequest("p", list(range(12)), num_computed_tokens=4)
            ],
            num_scheduled_tokens={"p": 4},
        ),
        step_index=0,
        virtual_time_ps=0,
    )
    check_attribution_case("mid-prompt", mid_prompt, 0, emit)
    prompt_complete = translate_scheduler_output(
        chunked,
        SchedulerOutput(
            scheduled_cached_reqs=SchedulerCachedRequests(["p"], [8], [0]),
            num_scheduled_tokens={"p": 4},
        ),
        step_index=1,
        virtual_time_ps=0,
    )
    check_attribution_case("prompt-completing-after-prefix", prompt_complete, 1, emit)
    decode = translate_scheduler_output(
        chunked,
        SchedulerOutput(
            scheduled_cached_reqs=SchedulerCachedRequests(["p"], [12], [1]),
            num_scheduled_tokens={"p": 1},
        ),
        step_index=2,
        virtual_time_ps=0,
    )
    check_attribution_case("decode", decode, 1, emit)

    prefix = StepTranslator()
    prefix_complete = translate_scheduler_output(
        prefix,
        SchedulerOutput(
            scheduled_new_reqs=[
                SchedulerNewRequest("cached", list(range(8)), num_computed_tokens=4)
            ],
            num_scheduled_tokens={"cached": 4},
        ),
        step_index=0,
        virtual_time_ps=0,
    )
    check_attribution_case("prefix-cache-completion", prefix_complete, 1, emit)

    attached = StepTranslator()
    attach_midflight = translate_scheduler_output(
        attached,
        SchedulerOutput(
            scheduled_cached_reqs=SchedulerCachedRequests(["orphan"], [31], [1]),
            num_scheduled_tokens={"orphan": 1},
        ),
        step_index=0,
        virtual_time_ps=0,
    )
    check_attribution_case("attach-mid-flight", attach_midflight, 1, emit)


def run_check_d(out: Path, emit) -> None:
    record = StepRecord(
        0,
        0,
        [ScheduledRequest("prefill", RequestPhase.PREFILL, 256, context_length=256)],
    )
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=default_dims(),
            workdir=out / "d-default-baseline",
            linkspeed_bps=LINKSPEED_BPS,
        )
    )
    result = sink(record)
    assert result is not None
    goal_path = out / "d-default-baseline" / "step-000000.goal"
    digest = hashlib.sha256(goal_path.read_bytes()).hexdigest()
    assert digest == DEFAULT_GOAL_SHA256
    assert sink.outcomes[0].per_layer_calc_ns == 12_030
    assert sink.outcomes[0].layer_calc_ns == (12_030, 12_030)

    payload = step_record_to_json(record)
    assert "num_sampled" not in payload
    assert step_record_from_json(payload) == record
    present_zero = replace(record, num_sampled=0)
    present_payload = step_record_to_json(present_zero)
    assert present_payload["num_sampled"] == 0
    assert step_record_from_json(present_payload) == present_zero

    default_provider = RooflineProvider(efficiency=0.7)
    default_kernel = step_kernel(default_dims(), record, num_sampled=1)
    assert default_provider.estimate_layers(
        default_kernel,
        GPU_ENVELOPES["b100"],
        2,
    ) is None
    enabled = RooflineProvider(efficiency=0.7, enable_layer_breakdown=True)
    assert enabled.estimate_layers(
        KernelSpec("plain", flops=1.0, bytes_moved=1.0),
        GPU_ENVELOPES["b100"],
        2,
    ) is None
    bad_family = KernelSpec("body", flops=1.0, bytes_moved=1.0)
    try:
        enabled.estimate_layers(
            KernelSpec(
                "bad",
                flops=2.0,
                bytes_moved=1.0,
                family_kernels=(bad_family,),
            ),
            GPU_ENVELOPES["b100"],
            2,
        )
    except ValueError:
        invalid_rejected = True
    else:
        raise AssertionError("enabled roofline accepted nonconserving families")

    emit(
        section="D",
        default_goal_sha256=digest,
        expected_goal_sha256=DEFAULT_GOAL_SHA256,
        per_layer_calc_ns=sink.outcomes[0].per_layer_calc_ns,
        default_breakdown_absent=True,
        absent_sample_field_omitted=True,
        present_zero_round_trips=True,
        invalid_family_metadata_rejected=invalid_rejected,
        physical_quiescence="verified",
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_deterministic(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    def emit(**row) -> None:
        rows.append(row)
        print(" ".join(f"{key}={value}" for key, value in row.items()))

    run_check_a(out, emit)
    run_check_b1(out, emit)
    run_check_b2(emit)
    run_check_d(out, emit)
    summary = out / "summary.csv"
    write_rows(summary, rows)
    print(
        f"completed {len(rows)} deterministic rows against expectations commit "
        f"{EXPECTATIONS_COMMIT}; summary={summary}"
    )


def run_live_vllm(out: Path) -> None:
    from vllm import LLM, SamplingParams

    from simllm.adapters.vllm import SimModelRunner, latest_worker

    live_dir = out / "live-vllm"
    record_path = live_dir / "steps.jsonl"
    summary_path = live_dir / "summary.json"
    if record_path.exists() or summary_path.exists():
        raise RuntimeError(f"refusing stale live evidence under {live_dir}")
    live_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SIMLLM_VLLM_STEP_RECORDS"] = str(record_path)

    llm = LLM(
        model=str(MODEL),
        worker_cls="simllm.adapters.vllm.SimWorker",
        enforce_eager=True,
        max_model_len=64,
        max_num_batched_tokens=2,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        num_gpu_blocks_override=64,
        disable_log_stats=True,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": [100, 101, 102]}],
        SamplingParams(max_tokens=2, ignore_eos=True),
    )
    worker = latest_worker()
    assert worker is not None
    assert isinstance(worker.model_runner, SimModelRunner)
    assert len(outputs) == 1
    assert len(outputs[0].outputs) == 1
    sampled_token_ids = tuple(outputs[0].outputs[0].token_ids)
    assert sampled_token_ids == (worker.token_id, worker.token_id)

    records = [
        json.loads(line)
        for line in record_path.read_text().splitlines()
        if line.strip()
    ]
    scheduled_tokens = tuple(
        sum(row["num_new_tokens"] for row in record["scheduled"])
        for record in records
    )
    exact_samples = tuple(record.get("num_sampled") for record in records)
    assert scheduled_tokens == (2, 1, 1)
    assert exact_samples == (0, 1, 1)
    assert sum(exact_samples) == len(sampled_token_ids)

    summary = {
        "expectations_commit": EXPECTATIONS_COMMIT,
        "vllm_version": importlib.metadata.version("vllm"),
        "worker_class": type(worker).__name__,
        "runner_class": type(worker.model_runner).__name__,
        "fabricated_token_id": worker.token_id,
        "sampled_token_ids": sampled_token_ids,
        "scheduled_tokens": scheduled_tokens,
        "exact_samples": exact_samples,
        "record_count": len(records),
        "scored_relation": "pass",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(" ".join(f"{key}={value}" for key, value in summary.items()))
    print(f"live summary={summary_path}")


def require_file_env(name: str) -> None:
    value = os.environ.get(name)
    if value is None or not Path(value).is_file():
        raise RuntimeError(f"{name} must name an existing file")


def check_only(mode: str, out: Path) -> None:
    if mode == "deterministic":
        require_file_env("SIMLLM_HTSIM_RNIC")
        require_file_env("SIMLLM_TXT2BIN")
    else:
        if importlib.metadata.version("vllm") != "0.26.0":
            raise RuntimeError("live-vllm mode requires vLLM 0.26.0")
        if not MODEL.is_dir():
            raise RuntimeError(f"cached model is missing: {MODEL}")
    print(f"check-only mode={mode} out={out}; no results produced")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("deterministic", "live-vllm"), required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        check_only(args.mode, args.out)
    elif args.mode == "deterministic":
        run_deterministic(args.out)
    else:
        run_live_vllm(args.out)


if __name__ == "__main__":
    main()
