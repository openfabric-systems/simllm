"""Run the frozen PLAY-3 adapter replay metric study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

from simllm.adapters.vllm import (
    ReplayTokenSource,
    SimExecutor,
    SimExecutorConfig,
    SimWorker,
    StepTranslator,
    configure,
    reset_configuration,
    translate_scheduler_output,
)
from simllm.core import RequestBookkeeper, StepRecord, StepResult, VirtualClock
from simllm.preplay import (
    ForwardPhase,
    PreplayTrace,
    RequestArrival,
    StopReason,
    join_preplay_arrivals,
    read_preplay_trace,
    write_preplay_replay_run,
    write_preplay_trace,
)

DEFAULT_RUN_DIR = Path(
    "/data3/yifeng/simllm-dev/wave2-runs/"
    "codex_play23_arrival_replay/preplay_adapter_replay_v1"
)
EXPECTED_GRANITE_SHA256 = (
    "36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341"
)
FIXED_OVERHEAD_PS = 1_000
CONTEXT_TOKEN_PS = 10
ARRIVALS_PS = {"r0": 0, "r1": 500}
ORACLE_TOKENS = {"r0": (38,), "r1": (61, 62, 63, 64)}
BASELINE_TOKENS = {"r0": (512, 512, 512, 512), "r1": (512, 512, 512, 512)}


@dataclass
class FakeNewRequest:
    req_id: str
    prompt_token_ids: list[int]
    sampling_params: object
    num_computed_tokens: int = 0


@dataclass
class FakeCachedRequests:
    req_ids: list[str] = field(default_factory=list)
    num_computed_tokens: list[int] = field(default_factory=list)
    num_output_tokens: list[int] = field(default_factory=list)


@dataclass
class FakeSchedulerOutput:
    scheduled_new_reqs: list[FakeNewRequest] = field(default_factory=list)
    scheduled_cached_reqs: FakeCachedRequests = field(default_factory=FakeCachedRequests)
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    finished_req_ids: set[str] = field(default_factory=set)
    preempted_req_ids: set[str] | None = None
    has_structured_output_requests: bool = False

    @property
    def total_num_scheduled_tokens(self) -> int:
        return sum(self.num_scheduled_tokens.values())


class FakeDType:
    itemsize = 2


class FakeIrOpPriority:
    @staticmethod
    def set_default():
        return


class FakeModelConfig:
    runner_type = "generate"
    dtype = FakeDType()
    hf_text_config = SimpleNamespace(intermediate_size=256)
    max_model_len = 4096

    @staticmethod
    def get_hidden_size():
        return 128

    @staticmethod
    def get_num_layers(parallel_config):
        return 4

    @staticmethod
    def get_num_attention_heads(parallel_config):
        return 8

    @staticmethod
    def get_num_kv_heads(parallel_config):
        return 2

    @staticmethod
    def get_head_size():
        return 16

    @staticmethod
    def get_vocab_size():
        return 1024

    @staticmethod
    def get_total_num_hidden_layers():
        return 4


def fake_vllm_config():
    return SimpleNamespace(
        model_config=FakeModelConfig(),
        cache_config=SimpleNamespace(
            block_size=16,
            cache_dtype="auto",
            num_gpu_blocks=None,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            world_size=1,
            rank=0,
        ),
        scheduler_config=SimpleNamespace(),
        device_config=SimpleNamespace(device="cuda"),
        speculative_config=None,
        lora_config=None,
        load_config=None,
        observability_config=None,
        kv_transfer_config=None,
        compilation_config=SimpleNamespace(ir_enable_torch_wrap=True),
        kernel_config=SimpleNamespace(ir_op_priority=FakeIrOpPriority()),
        profiler_config=SimpleNamespace(profiler=None),
        quant_config=None,
        use_v2_model_runner=False,
    )


def sampling_params(output_length: int):
    return SimpleNamespace(
        max_tokens=output_length,
        min_tokens=0,
        stop_token_ids=[],
    )


@contextmanager
def skeleton_environment():
    names = {
        "SIMLLM_VLLM_WORKER_MODE": "skeleton",
        "SIMLLM_VLLM_MODE": "virtual",
    }
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update(names)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class LinearStepSink:
    def __init__(self, token_cost_ps: int) -> None:
        self.token_cost_ps = token_cost_ps

    def __call__(self, record: StepRecord) -> StepResult:
        latency_ps = (
            FIXED_OVERHEAD_PS
            + self.token_cost_ps
            * sum(request.num_new_tokens for request in record.scheduled)
            + CONTEXT_TOKEN_PS
            * sum(request.context_length for request in record.scheduled)
        )
        return StepResult(
            step_index=record.step_index,
            step_latency_ps=latency_ps,
            completed_at_ps=record.virtual_time_ps + latency_ps,
        )


def _metric_trace(granite_path: Path) -> PreplayTrace:
    granite = read_preplay_trace(granite_path)
    source = granite.requests[0]
    r0_prefill = tuple(source.prefill_tokens[:2])
    r0 = replace(
        source,
        request_id="r0",
        input_token_ids=tuple(token.token_id for token in r0_prefill),
        max_new_tokens=1,
        output_token_ids=ORACLE_TOKENS["r0"],
        output_text="4",
        stop_reason=StopReason.LENGTH_CAP,
        prefill_tokens=r0_prefill,
        decode_tokens=(),
    )
    r1_prefill = tuple(source.prefill_tokens[:3])
    route_template = source.prefill_tokens[0]
    r1_decode = tuple(
        replace(
            route_template,
            phase=ForwardPhase.DECODE,
            token_index=index,
            token_id=token_id,
        )
        for index, token_id in enumerate(ORACLE_TOKENS["r1"][:-1])
    )
    r1 = replace(
        source,
        request_id="r1",
        prompt_sha256="d" * 64,
        input_token_ids=tuple(token.token_id for token in r1_prefill),
        max_new_tokens=4,
        output_token_ids=ORACLE_TOKENS["r1"],
        output_text="synthetic follower",
        stop_reason=StopReason.LENGTH_CAP,
        prefill_tokens=r1_prefill,
        decode_tokens=r1_decode,
    )
    return PreplayTrace(provenance=granite.provenance, requests=(r0, r1))


def build_replay_run(run_dir: Path) -> Path:
    granite = Path(__file__).parents[1] / "preplay_trace_v1/granite_length_cap.jsonl"
    observed = hashlib.sha256(granite.read_bytes()).hexdigest()
    if observed != EXPECTED_GRANITE_SHA256:
        raise AssertionError(f"tracked Granite hash {observed} != {EXPECTED_GRANITE_SHA256}")
    trace = _metric_trace(granite)
    trace_path = write_preplay_trace(
        run_dir / "metric-trace.jsonl",
        trace.provenance,
        trace.requests,
    )
    run = join_preplay_arrivals(
        (
            RequestArrival(request_id="r0", arrived_at_ps=ARRIVALS_PS["r0"]),
            RequestArrival(request_id="r1", arrived_at_ps=ARRIVALS_PS["r1"]),
        ),
        trace_path,
        RequestBookkeeper(),
    )
    return write_preplay_replay_run(run, run_dir / "joined-replay.json")


def schedule(replay: bool) -> list[FakeSchedulerOutput]:
    r0_length = 1 if replay else 4
    first = FakeSchedulerOutput(
        scheduled_new_reqs=[
            FakeNewRequest("r0", [10, 11], sampling_params(r0_length))
        ],
        num_scheduled_tokens={"r0": 2},
    )
    if replay:
        return [
            first,
            FakeSchedulerOutput(
                scheduled_new_reqs=[
                    FakeNewRequest("r1", [20, 21, 22], sampling_params(4))
                ],
                num_scheduled_tokens={"r1": 3},
                finished_req_ids={"r0"},
            ),
            FakeSchedulerOutput(
                scheduled_cached_reqs=FakeCachedRequests(["r1"], [3], [1]),
                num_scheduled_tokens={"r1": 1},
            ),
            FakeSchedulerOutput(
                scheduled_cached_reqs=FakeCachedRequests(["r1"], [4], [2]),
                num_scheduled_tokens={"r1": 1},
            ),
            FakeSchedulerOutput(
                scheduled_cached_reqs=FakeCachedRequests(["r1"], [5], [3]),
                num_scheduled_tokens={"r1": 1},
            ),
            FakeSchedulerOutput(finished_req_ids={"r1"}),
        ]
    return [
        first,
        FakeSchedulerOutput(
            scheduled_new_reqs=[
                FakeNewRequest("r1", [20, 21, 22], sampling_params(4))
            ],
            scheduled_cached_reqs=FakeCachedRequests(["r0"], [2], [1]),
            num_scheduled_tokens={"r0": 1, "r1": 3},
        ),
        FakeSchedulerOutput(
            scheduled_cached_reqs=FakeCachedRequests(
                ["r0", "r1"], [3, 3], [2, 1]
            ),
            num_scheduled_tokens={"r0": 1, "r1": 1},
        ),
        FakeSchedulerOutput(
            scheduled_cached_reqs=FakeCachedRequests(
                ["r0", "r1"], [4, 4], [3, 2]
            ),
            num_scheduled_tokens={"r0": 1, "r1": 1},
        ),
        FakeSchedulerOutput(
            scheduled_cached_reqs=FakeCachedRequests(["r1"], [5], [3]),
            num_scheduled_tokens={"r1": 1},
            finished_req_ids={"r0"},
        ),
        FakeSchedulerOutput(finished_req_ids={"r1"}),
    ]


def drive_executor_probe(
    replay_path: Path,
    scheduler_outputs: list[FakeSchedulerOutput],
    replay: bool,
) -> dict[str, tuple[int, ...]]:
    executor = object.__new__(SimExecutor)
    executor.replay = (
        ReplayTokenSource.from_path(replay_path, max_model_len=4096)
        if replay
        else None
    )
    executor.token_id = 512
    translator = StepTranslator()
    served: dict[str, list[int]] = {"r0": [], "r1": []}
    for step_index, scheduler_output in enumerate(scheduler_outputs):
        if not scheduler_output.num_scheduled_tokens:
            if executor.replay is not None:
                executor.replay.observe_completions(scheduler_output)
            continue
        translated = translate_scheduler_output(
            translator,
            scheduler_output,
            step_index=step_index,
            virtual_time_ps=0,
        )
        req_ids, _, sampled = executor._sample_output_fields(
            translated, scheduler_output
        )
        for request_id, token_ids in zip(req_ids, sampled, strict=True):
            served[request_id].extend(token_ids)
    return {request_id: tuple(tokens) for request_id, tokens in served.items()}


def drive_worker(
    run_dir: Path,
    replay_path: Path,
    token_cost_ps: int,
    replay: bool,
) -> dict:
    label = "replay" if replay else "baseline"
    stream_path = run_dir / f"{label}_c{token_cost_ps}_steps.jsonl"
    config = SimExecutorConfig(
        mode="virtual",
        token_id=512,
        step_records_path=str(stream_path),
        replay_run_path=str(replay_path) if replay else None,
    )
    reset_configuration()
    configure(step_sink=LinearStepSink(token_cost_ps), config=config)
    scheduler_outputs = schedule(replay)
    completion_times: dict[str, list[int]] = {"r0": [], "r1": []}
    served: dict[str, list[int]] = {"r0": [], "r1": []}
    try:
        with skeleton_environment():
            worker = SimWorker(
                fake_vllm_config(),
                local_rank=0,
                rank=0,
                distributed_init_method="tcp://127.0.0.1:1",
                is_driver_worker=True,
                _simllm_clock=VirtualClock(),
            )
            worker.init_device()
            for scheduler_output in scheduler_outputs:
                output = worker.execute_model(scheduler_output)
                if scheduler_output.num_scheduled_tokens:
                    if output is not None:
                        raise AssertionError(
                            "nonempty skeleton execute must split sampling"
                        )
                    output = worker.sample_tokens(None)
                for request_id, token_ids in zip(
                    output.req_ids,
                    output.sampled_token_ids or (),
                    strict=True,
                ):
                    served[request_id].extend(token_ids)
                    if token_ids:
                        completion_times[request_id].append(worker.clock.now_ps)
    finally:
        reset_configuration()

    r1_times = completion_times["r1"]
    ttft = r1_times[0] - ARRIVALS_PS["r1"]
    tpot = Fraction(
        sum(later - earlier for earlier, later in pairwise(r1_times)),
        len(r1_times) - 1,
    )
    executor_served = drive_executor_probe(
        replay_path,
        scheduler_outputs,
        replay,
    )
    replay_snapshot = worker.replay.snapshot() if worker.replay is not None else None
    return {
        "mode": label,
        "token_cost_ps": token_cost_ps,
        "ttft_ps": ttft,
        "tpot_ps": tpot,
        "served": {name: tuple(tokens) for name, tokens in served.items()},
        "executor_served": executor_served,
        "completion_visits": {
            name: len(times) for name, times in completion_times.items()
        },
        "step_count": len(worker.step_records),
        "step_composition": [
            tuple(request.request_id for request in record.scheduled)
            for record in worker.step_records
        ],
        "step_latencies": tuple(result.step_latency_ps for result in worker.step_results),
        "replay_snapshot": replay_snapshot,
    }


def _fraction_json(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def run_study(run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=False)
    replay_path = build_replay_run(run_dir)
    rows = [
        drive_worker(run_dir, replay_path, token_cost_ps, replay)
        for token_cost_ps in (100, 200)
        for replay in (False, True)
    ]
    by_key = {(row["mode"], row["token_cost_ps"]): row for row in rows}

    b1 = all(
        all(
            (
                by_key[("replay", token_cost)]["served"] == ORACLE_TOKENS,
                by_key[("replay", token_cost)]["executor_served"] == ORACLE_TOKENS,
                by_key[("replay", token_cost)]["completion_visits"]
                == {"r0": 1, "r1": 4},
            )
        )
        for token_cost in (100, 200)
    )
    expected = {
        100: {
            "baseline_ttft": 2180,
            "replay_ttft": 2050,
            "baseline_tpot": Fraction(3740, 3),
            "replay_tpot": Fraction(1150, 1),
            "delta_ttft": -130,
            "delta_tpot": Fraction(-290, 3),
        },
        200: {
            "baseline_ttft": 2780,
            "replay_ttft": 2550,
            "baseline_tpot": Fraction(4240, 3),
            "replay_tpot": Fraction(1250, 1),
            "delta_ttft": -230,
            "delta_tpot": Fraction(-490, 3),
        },
    }
    deltas: dict[int, tuple[int, Fraction]] = {}
    b2 = True
    for token_cost, oracle in expected.items():
        baseline = by_key[("baseline", token_cost)]
        replay = by_key[("replay", token_cost)]
        delta_ttft = replay["ttft_ps"] - baseline["ttft_ps"]
        delta_tpot = replay["tpot_ps"] - baseline["tpot_ps"]
        deltas[token_cost] = (delta_ttft, delta_tpot)
        b2 = b2 and all(
            (
                baseline["ttft_ps"] == oracle["baseline_ttft"],
                replay["ttft_ps"] == oracle["replay_ttft"],
                baseline["tpot_ps"] == oracle["baseline_tpot"],
                replay["tpot_ps"] == oracle["replay_tpot"],
                delta_ttft == oracle["delta_ttft"],
                delta_tpot == oracle["delta_tpot"],
            )
        )
    b3 = (
        deltas[200][0] - deltas[100][0] == -100
        and deltas[200][1] - deltas[100][1] == Fraction(-200, 3)
    )
    structural = {
        "baseline_tokens": all(
            by_key[("baseline", token_cost)]["served"] == BASELINE_TOKENS
            for token_cost in (100, 200)
        ),
        "baseline_schedule": all(
            by_key[("baseline", token_cost)]["step_composition"]
            == [("r0",), ("r0", "r1"), ("r0", "r1"), ("r0", "r1"), ("r1",), ()]
            for token_cost in (100, 200)
        ),
        "replay_schedule": all(
            by_key[("replay", token_cost)]["step_composition"]
            == [("r0",), ("r1",), ("r1",), ("r1",), ("r1",), ()]
            for token_cost in (100, 200)
        ),
        "replay_drained": all(
            by_key[("replay", token_cost)]["replay_snapshot"].drained_request_ids
            == ("r0", "r1")
            for token_cost in (100, 200)
        ),
    }

    public_rows = []
    for row in rows:
        public_rows.append(
            {
                "mode": row["mode"],
                "token_cost_ps": row["token_cost_ps"],
                "ttft_ps": row["ttft_ps"],
                "tpot_ps": _fraction_json(row["tpot_ps"]),
                "served": {name: list(tokens) for name, tokens in row["served"].items()},
                "executor_served": {
                    name: list(tokens) for name, tokens in row["executor_served"].items()
                },
                "completion_visits": row["completion_visits"],
                "step_count": row["step_count"],
                "step_composition": [list(ids) for ids in row["step_composition"]],
                "step_latencies": list(row["step_latencies"]),
            }
        )
    summary = {
        "scored": {"B1": b1, "B2": b2},
        "fatal_unscored": {**structural, "B3_coefficient_scaling": b3},
        "rows": public_rows,
        "deltas": {
            str(token_cost): {
                "ttft_ps": delta[0],
                "tpot_ps": _fraction_json(delta[1]),
            }
            for token_cost, delta in deltas.items()
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("mode", "token_cost_ps", "ttft_ps", "tpot_ps"))
        for row in rows:
            writer.writerow(
                (
                    row["mode"],
                    row["token_cost_ps"],
                    row["ttft_ps"],
                    str(row["tpot_ps"]),
                )
            )
    if not all(summary["scored"].values()):
        raise AssertionError(f"scored relation failed: {summary['scored']}")
    if not all(summary["fatal_unscored"].values()):
        raise AssertionError(f"fatal guard failed: {summary['fatal_unscored']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    fixture = Path(__file__).parents[1] / "preplay_trace_v1/granite_length_cap.jsonl"
    if not fixture.is_file():
        raise SystemExit(f"missing tracked fixture: {fixture}")
    if args.check_only:
        return
    summary = run_study(args.run_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
