"""Run the review-frozen PLAY-3 relation through vLLM's real scheduler."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from simllm._local_config import path_from_env
from simllm.adapters.vllm import (
    SimExecutorConfig,
    configure,
    latest_worker,
    reset_configuration,
)
from simllm.core import RequestBookkeeper, StepRecord, StepResult, step_records_to_json
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

MODEL_CACHE_PATH = (
    Path("hub")
    / "models--ibm-granite--granite-3.0-1b-a400m-instruct"
    / "snapshots"
    / "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
)
GRANITE_FIXTURE = (
    Path(__file__).parents[1] / "preplay_trace_v1/granite_length_cap.jsonl"
)
EXPECTED_GRANITE_SHA256 = (
    "36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341"
)
PINNED_SOURCE_HASHES = {
    "v1/engine/input_processor.py": (
        "c5673988c0f7cfec268220e3f044e718702c015a4f236c020937cfd40a793f15"
    ),
    "sampling_params.py": (
        "d2f9789ba2b93819c4918159cfd29818eab3ba4f9098241e2febadcc690aa767"
    ),
    "v1/core/sched/utils.py": (
        "85e82eae555a03497ad2ac1540ed562a6c36fc26185aa6233725c914816aa1b3"
    ),
    "v1/core/sched/scheduler.py": (
        "2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941"
    ),
    "v1/engine/llm_engine.py": (
        "17e5edfc625c77e9663368c7d69136e5e5935ee81608a65be3996411d502225e"
    ),
    "v1/engine/output_processor.py": (
        "ee10351275d90796c8b901a5f4b23d5a046ef6ee72fd2921aff2ae78ca58bd9b"
    ),
}
FIXED_OVERHEAD_PS = 1_000
CONTEXT_TOKEN_PS = 10
TOKEN_COSTS_PS = (100, 200)
ORACLE_TOKENS = {"r0": (38,), "r1": (61, 62, 63, 64)}
BASELINE_TOKENS = {"r0": (512, 512, 512, 512), "r1": (512, 512, 512, 512)}


def _required_env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set to an existing directory")
    path = Path(value).expanduser()
    if not path.is_dir():
        raise RuntimeError(f"{name} must name an existing directory: {path}")
    return path


def _model_path() -> Path:
    return _required_env_path("HF_HOME") / MODEL_CACHE_PATH


def _vllm_package_root() -> Path:
    configured = _required_env_path("SIMLLM_VLLM_PACKAGE_ROOT")
    if (configured / "sampling_params.py").is_file():
        return configured
    raise RuntimeError(
        "SIMLLM_VLLM_PACKAGE_ROOT must name the installed vllm package directory"
    )


class LinearStepSink:
    """Price every engine-produced record with the frozen deterministic rule."""

    def __init__(self, token_cost_ps: int) -> None:
        self.token_cost_ps = token_cost_ps

    def latency(self, record: StepRecord) -> int:
        return (
            FIXED_OVERHEAD_PS
            + self.token_cost_ps
            * sum(request.num_new_tokens for request in record.scheduled)
            + CONTEXT_TOKEN_PS
            * sum(request.context_length for request in record.scheduled)
        )

    def __call__(self, record: StepRecord) -> StepResult:
        latency_ps = self.latency(record)
        return StepResult(
            step_index=record.step_index,
            step_latency_ps=latency_ps,
            completed_at_ps=record.virtual_time_ps + latency_ps,
        )


@dataclass(frozen=True, kw_only=True)
class CellResult:
    mode: str
    token_cost_ps: int
    arrival_r1_ps: int
    tokens: dict[str, tuple[int, ...]]
    token_times_ps: dict[str, tuple[int, ...]]
    finish_steps: dict[str, int]
    ttft_ps: int
    tpot_ps: Fraction
    step_composition: tuple[tuple[str, ...], ...]
    step_latencies_ps: tuple[int, ...]
    replay_completed: tuple[str, ...]
    replay_drained: tuple[str, ...]


def _metric_trace() -> PreplayTrace:
    granite = read_preplay_trace(GRANITE_FIXTURE)
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


def build_metric_trace(
    run_dir: Path,
) -> tuple[Path, dict[str, tuple[int, ...]]]:
    trace = _metric_trace()
    trace_path = write_preplay_trace(
        run_dir / "metric-trace.jsonl",
        trace.provenance,
        trace.requests,
    )
    prompts = {
        request.request_id: request.input_token_ids for request in trace.requests
    }
    return trace_path, prompts


def build_replay_run(
    run_dir: Path,
    trace_path: Path,
    token_cost_ps: int,
) -> Path:
    arrival_r1_ps = FIXED_OVERHEAD_PS + 2 * token_cost_ps + 2 * CONTEXT_TOKEN_PS
    run = join_preplay_arrivals(
        (
            RequestArrival(request_id="r0", arrived_at_ps=0),
            RequestArrival(request_id="r1", arrived_at_ps=arrival_r1_ps),
        ),
        trace_path,
        RequestBookkeeper(),
    )
    return write_preplay_replay_run(
        run,
        run_dir / f"joined-replay-c{token_cost_ps}.json",
    )


@contextmanager
def engine_environment() -> Iterator[None]:
    values = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_V1": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "SIMLLM_VLLM_WORKER_MODE": "skeleton",
        "SIMLLM_VLLM_MODE": "virtual",
    }
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


def _observe_outputs(
    outputs: list[Any],
    step_index: int,
    completed_at_ps: int,
    served: dict[str, list[int]],
    token_times: dict[str, list[int]],
    finish_steps: dict[str, int],
) -> None:
    for output in outputs:
        request_id = output.request_id
        if request_id not in served:
            raise AssertionError(f"engine returned unknown request {request_id!r}")
        if len(output.outputs) != 1:
            raise AssertionError(f"request {request_id!r} returned multiple choices")
        cumulative = tuple(output.outputs[0].token_ids)
        previous = tuple(served[request_id])
        if cumulative[: len(previous)] != previous:
            raise AssertionError(f"request {request_id!r} changed an emitted prefix")
        new_tokens = cumulative[len(previous) :]
        served[request_id].extend(new_tokens)
        token_times[request_id].extend([completed_at_ps] * len(new_tokens))
        if output.finished:
            if request_id in finish_steps:
                raise AssertionError(f"request {request_id!r} finished twice")
            finish_steps[request_id] = step_index


def drive_cell(
    run_dir: Path,
    replay_path: Path,
    prompts: dict[str, tuple[int, ...]],
    *,
    token_cost_ps: int,
    replay: bool,
) -> CellResult:
    from vllm import LLM, SamplingParams

    mode = "replay" if replay else "baseline"
    stream_path = run_dir / f"engine_{mode}_c{token_cost_ps}_steps.jsonl"
    sink = LinearStepSink(token_cost_ps)
    config = SimExecutorConfig(
        mode="virtual",
        token_id=512,
        step_records_path=str(stream_path),
        replay_run_path=str(replay_path) if replay else None,
    )
    served: dict[str, list[int]] = {"r0": [], "r1": []}
    token_times: dict[str, list[int]] = {"r0": [], "r1": []}
    finish_steps: dict[str, int] = {}
    llm: Any | None = None
    worker: Any | None = None
    reset_configuration()
    configure(step_sink=sink, config=config)
    try:
        with engine_environment():
            llm = LLM(
                model=str(_model_path()),
                worker_cls="simllm.adapters.vllm.SimWorker",
                enforce_eager=True,
                max_model_len=64,
                num_gpu_blocks_override=64,
                disable_log_stats=True,
                enable_chunked_prefill=False,
                async_scheduling=False,
            )
            worker = latest_worker()
            if worker is None:
                raise AssertionError("SimWorker was not constructed")
            if (worker.replay is not None) is not replay:
                raise AssertionError(f"{mode} cell selected the wrong replay path")

            r0_limit = 1 if replay else 4
            request_id = llm.llm_engine.add_request(
                "r0",
                {"prompt_token_ids": list(prompts["r0"])},
                SamplingParams(
                    temperature=0.0,
                    max_tokens=r0_limit,
                    min_tokens=0,
                    detokenize=False,
                ),
            )
            if request_id != "r0":
                raise AssertionError(f"engine changed r0 identity to {request_id!r}")

            outputs = llm.llm_engine.step()
            if len(worker.step_records) != 1:
                raise AssertionError("engine step 0 did not emit exactly one record")
            _observe_outputs(
                outputs,
                worker.step_records[-1].step_index,
                worker.clock.now_ps,
                served,
                token_times,
                finish_steps,
            )
            arrival_r1_ps = worker.clock.now_ps
            expected_arrival_r1_ps = (
                FIXED_OVERHEAD_PS
                + 2 * token_cost_ps
                + 2 * CONTEXT_TOKEN_PS
            )
            if arrival_r1_ps != expected_arrival_r1_ps:
                raise AssertionError(
                    f"engine admitted r1 at {arrival_r1_ps}, "
                    f"expected {expected_arrival_r1_ps}"
                )

            request_id = llm.llm_engine.add_request(
                "r1",
                {"prompt_token_ids": list(prompts["r1"])},
                SamplingParams(
                    temperature=0.0,
                    max_tokens=4,
                    min_tokens=0,
                    detokenize=False,
                ),
            )
            if request_id != "r1":
                raise AssertionError(f"engine changed r1 identity to {request_id!r}")

            while llm.llm_engine.has_unfinished_requests():
                before = len(worker.step_records)
                outputs = llm.llm_engine.step()
                if len(worker.step_records) != before + 1:
                    raise AssertionError("one engine step did not emit one step record")
                _observe_outputs(
                    outputs,
                    worker.step_records[-1].step_index,
                    worker.clock.now_ps,
                    served,
                    token_times,
                    finish_steps,
                )

            records = tuple(worker.step_records)
            results = tuple(worker.step_results)
            if [result.step_latency_ps for result in results] != [
                sink.latency(record) for record in records
            ]:
                raise AssertionError("engine records bypassed the common step sink")
            streamed = [
                json.loads(line)
                for line in stream_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if streamed != step_records_to_json(records):
                raise AssertionError("durable stream differs from engine-produced records")
            replay_snapshot = worker.replay.snapshot() if worker.replay else None
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        reset_configuration()

    if worker is None:
        raise AssertionError("cell ended without a worker")
    r1_times = token_times["r1"]
    if len(r1_times) != 4:
        raise AssertionError(f"r1 emitted {len(r1_times)} tokens instead of four")
    ttft_ps = r1_times[0] - arrival_r1_ps
    tpot_ps = Fraction(
        sum(later - earlier for earlier, later in pairwise(r1_times)),
        len(r1_times) - 1,
    )
    return CellResult(
        mode=mode,
        token_cost_ps=token_cost_ps,
        arrival_r1_ps=arrival_r1_ps,
        tokens={name: tuple(token_ids) for name, token_ids in served.items()},
        token_times_ps={name: tuple(times) for name, times in token_times.items()},
        finish_steps=finish_steps,
        ttft_ps=ttft_ps,
        tpot_ps=tpot_ps,
        step_composition=tuple(
            tuple(request.request_id for request in record.scheduled)
            for record in records
        ),
        step_latencies_ps=tuple(result.step_latency_ps for result in results),
        replay_completed=(
            replay_snapshot.completed_request_ids if replay_snapshot else ()
        ),
        replay_drained=replay_snapshot.drained_request_ids if replay_snapshot else (),
    )


def _fraction_json(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _cell_json(cell: CellResult) -> dict[str, Any]:
    return {
        "mode": cell.mode,
        "token_cost_ps": cell.token_cost_ps,
        "arrival_r1_ps": cell.arrival_r1_ps,
        "tokens": {name: list(tokens) for name, tokens in cell.tokens.items()},
        "token_times_ps": {
            name: list(times) for name, times in cell.token_times_ps.items()
        },
        "finish_steps": cell.finish_steps,
        "ttft_ps": cell.ttft_ps,
        "tpot_ps": _fraction_json(cell.tpot_ps),
        "step_composition": [list(ids) for ids in cell.step_composition],
        "step_latencies_ps": list(cell.step_latencies_ps),
        "replay_completed": list(cell.replay_completed),
        "replay_drained": list(cell.replay_drained),
    }


def run_study(run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    trace_path, prompts = build_metric_trace(run_dir)
    replay_paths = {
        token_cost_ps: build_replay_run(run_dir, trace_path, token_cost_ps)
        for token_cost_ps in TOKEN_COSTS_PS
    }
    cells = [
        drive_cell(
            run_dir,
            replay_paths[token_cost_ps],
            prompts,
            token_cost_ps=token_cost_ps,
            replay=replay,
        )
        for token_cost_ps in TOKEN_COSTS_PS
        for replay in (False, True)
    ]
    by_key = {(cell.mode, cell.token_cost_ps): cell for cell in cells}
    expected_metrics = {
        100: (1460, 1330, Fraction(3740, 3), Fraction(1150, 1)),
        200: (1860, 1630, Fraction(4240, 3), Fraction(1250, 1)),
    }
    deltas: dict[int, tuple[int, Fraction]] = {}
    metrics_exact = True
    for token_cost_ps, expected in expected_metrics.items():
        baseline = by_key[("baseline", token_cost_ps)]
        replay = by_key[("replay", token_cost_ps)]
        delta = (
            replay.ttft_ps - baseline.ttft_ps,
            replay.tpot_ps - baseline.tpot_ps,
        )
        deltas[token_cost_ps] = delta
        expected_delta = (
            -(token_cost_ps + 30),
            Fraction(-(2 * token_cost_ps + 90), 3),
        )
        metrics_exact = metrics_exact and (
            baseline.ttft_ps,
            replay.ttft_ps,
            baseline.tpot_ps,
            replay.tpot_ps,
        ) == expected and delta == expected_delta

    engine_completion_exact = all(
        by_key[("baseline", token_cost_ps)].tokens == BASELINE_TOKENS
        and by_key[("baseline", token_cost_ps)].finish_steps == {"r0": 3, "r1": 4}
        and by_key[("replay", token_cost_ps)].tokens == ORACLE_TOKENS
        and by_key[("replay", token_cost_ps)].finish_steps == {"r0": 0, "r1": 4}
        and by_key[("replay", token_cost_ps)].replay_completed == ("r0", "r1")
        for token_cost_ps in TOKEN_COSTS_PS
    )
    expected_baseline_shape = (
        ("r0",),
        ("r0", "r1"),
        ("r0", "r1"),
        ("r0", "r1"),
        ("r1",),
    )
    expected_replay_shape = (("r0",), ("r1",), ("r1",), ("r1",), ("r1",))
    fatal_unscored = {
        "baseline_scheduler_shape": all(
            by_key[("baseline", cost)].step_composition == expected_baseline_shape
            for cost in TOKEN_COSTS_PS
        ),
        "replay_scheduler_shape": all(
            by_key[("replay", cost)].step_composition == expected_replay_shape
            for cost in TOKEN_COSTS_PS
        ),
        "engine_delays_only_final_drain": all(
            by_key[("replay", cost)].replay_drained == ("r0",)
            for cost in TOKEN_COSTS_PS
        ),
        "coefficient_scaling": (
            deltas[200][0] - deltas[100][0] == -100
            and deltas[200][1] - deltas[100][1] == Fraction(-200, 3)
        ),
    }
    summary = {
        "scored": {
            "B2_real_scheduler_completion": engine_completion_exact,
            "B2_real_scheduler_metrics": metrics_exact,
        },
        "fatal_unscored": fatal_unscored,
        "cells": [_cell_json(cell) for cell in cells],
        "deltas": {
            str(cost): {
                "ttft_ps": delta[0],
                "tpot_ps": _fraction_json(delta[1]),
            }
            for cost, delta in deltas.items()
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("mode", "token_cost_ps", "ttft_ps", "tpot_ps"))
        for cell in cells:
            writer.writerow(
                (cell.mode, cell.token_cost_ps, cell.ttft_ps, str(cell.tpot_ps))
            )
    if not all(summary["scored"].values()):
        raise AssertionError(f"scored relation failed: {summary['scored']}")
    if not all(fatal_unscored.values()):
        raise AssertionError(f"fatal guard failed: {fatal_unscored}")
    return summary


def check_inputs() -> None:
    model = _model_path()
    vllm_root = _vllm_package_root()
    if not model.is_dir() or not GRANITE_FIXTURE.is_file():
        raise SystemExit("cached model or tracked Granite fixture is missing")
    observed = hashlib.sha256(GRANITE_FIXTURE.read_bytes()).hexdigest()
    if observed != EXPECTED_GRANITE_SHA256:
        raise SystemExit(f"tracked Granite hash {observed} is not pinned")
    if importlib.metadata.version("vllm") != "0.26.0":
        raise SystemExit("the review study requires vLLM 0.26.0")
    for relative_path, expected in PINNED_SOURCE_HASHES.items():
        observed = hashlib.sha256((vllm_root / relative_path).read_bytes()).hexdigest()
        if observed != expected:
            raise SystemExit(f"pinned vLLM source changed: {relative_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    check_inputs()
    if args.check_only:
        return
    if args.run_dir is None:
        data_root = path_from_env("SIMLLM_DATA_ROOT")
        if data_root is None:
            parser.error("--run-dir is required when SIMLLM_DATA_ROOT is not set")
        args.run_dir = data_root / "preplay_adapter_replay_v1" / "engine"
    summary = run_study(args.run_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
