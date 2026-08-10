"""One scored in-process vLLM skeleton smoke with joined replay."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPOSITORY_ROOT = Path(__file__).parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from simllm.adapters.vllm import reset_configuration
from simllm.core import RequestBookkeeper
from simllm.preplay import (
    RequestArrival,
    join_preplay_arrivals,
    read_preplay_trace,
    write_preplay_replay_run,
)

MODEL = Path(
    "/home/yifeng/packages/vllm-rnic-capture/hf-cache/hub/"
    "models--ibm-granite--granite-3.0-1b-a400m-instruct/snapshots/"
    "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
)
DEFAULT_RUN_DIR = Path(
    "/data3/yifeng/simllm-dev/wave2-runs/"
    "codex_play23_arrival_replay/preplay_adapter_replay_live"
)
EXPECTED_TOKEN_IDS = (38,)
EXPECTED_STEP_SCHEMA = "atlahs-closed-loop-step-v1"


def build_joined_run(run_dir: Path) -> tuple[Path, tuple[int, ...]]:
    fixture = Path(__file__).parents[1] / "preplay_trace_v1/granite_length_cap.jsonl"
    trace = read_preplay_trace(fixture)
    request = trace.by_request_id("length-cap")
    run = join_preplay_arrivals(
        (RequestArrival(request_id="length-cap", arrived_at_ps=0),),
        fixture,
        RequestBookkeeper(),
    )
    path = write_preplay_replay_run(run, run_dir / "joined-replay.json")
    return path, request.input_token_ids


def run_smoke(run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=False)
    replay_path, prompt_token_ids = build_joined_run(run_dir)
    stream_path = run_dir / "steps.jsonl"
    environment = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1",
        "VLLM_USE_V1": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "SIMLLM_VLLM_WORKER_MODE": "skeleton",
        "SIMLLM_VLLM_MODE": "virtual",
        "SIMLLM_VLLM_REPLAY_RUN": str(replay_path),
        "SIMLLM_VLLM_STEP_RECORDS": str(stream_path),
    }
    previous = {name: os.environ.get(name) for name in environment}
    reset_configuration()
    try:
        os.environ.update(environment)
        from vllm import LLM, SamplingParams

        from simllm.adapters.vllm import SimModelRunner, latest_worker

        llm = LLM(
            model=str(MODEL),
            worker_cls="simllm.adapters.vllm.SimWorker",
            enforce_eager=True,
            max_model_len=64,
            num_gpu_blocks_override=64,
            disable_log_stats=True,
        )
        llm.request_counter = iter(("length-cap",))
        internal_request_ids = llm._add_completion_requests(
            prompts=[{"prompt_token_ids": list(prompt_token_ids)}],
            params=SamplingParams(
                temperature=0.0,
                max_tokens=1,
                min_tokens=0,
                detokenize=False,
            ),
            use_tqdm=False,
        )
        if internal_request_ids != ["length-cap"]:
            raise AssertionError(
                f"expected exact external request identity, got {internal_request_ids}"
            )
        outputs = []
        while llm.llm_engine.has_unfinished_requests():
            outputs.extend(
                output
                for output in llm.llm_engine.step()
                if output.finished
            )
        worker = latest_worker()
        if worker is None:
            raise AssertionError("SimWorker was not constructed")
        if not isinstance(worker.model_runner, SimModelRunner):
            raise TypeError(f"unexpected runner {type(worker.model_runner)!r}")
        if worker.replay is None:
            raise AssertionError("SimWorker did not load the joined replay run")
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise AssertionError("live replay did not return one completion")
        if outputs[0].request_id != "length-cap":
            raise AssertionError(
                f"live output changed request identity to {outputs[0].request_id!r}"
            )
        sampled = tuple(outputs[0].outputs[0].token_ids)
        if sampled != EXPECTED_TOKEN_IDS:
            raise AssertionError(f"sampled {sampled}, expected {EXPECTED_TOKEN_IDS}")

        before_drain = worker.replay.snapshot()
        if before_drain.served_token_ids != (("length-cap", EXPECTED_TOKEN_IDS),):
            raise AssertionError(f"unexpected replay snapshot {before_drain}")
        if before_drain.completed_request_ids != ("length-cap",):
            raise AssertionError("replay did not observe oracle completion")

        drain_output = worker.execute_model(
            SimpleNamespace(
                scheduled_new_reqs=[],
                scheduled_cached_reqs=SimpleNamespace(
                    req_ids=[],
                    num_computed_tokens=[],
                    num_output_tokens=[],
                ),
                num_scheduled_tokens={},
                finished_req_ids={"length-cap"},
                preempted_req_ids=None,
                has_structured_output_requests=False,
            )
        )
        if drain_output.req_ids != [] or drain_output.sampled_token_ids not in (None, []):
            raise AssertionError("drain produced a token")
        after_drain = worker.replay.snapshot()
        if after_drain.drained_request_ids != ("length-cap",):
            raise AssertionError("drain did not retain the finished request identity")
        if worker.step_records[-1].finished_request_ids != ["length-cap"]:
            raise AssertionError("drain step omitted the finished request identity")
        if worker.step_results[-1].step_latency_ps != 0:
            raise AssertionError("drain step advanced simulated time")

        records = [
            json.loads(line)
            for line in stream_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records) != 2:
            raise AssertionError(f"expected two streamed records, got {len(records)}")
        if {record.get("schema") for record in records} != {EXPECTED_STEP_SCHEMA}:
            raise AssertionError("unexpected streamed step schema")
        if records[-1]["finished_request_ids"] != ["length-cap"]:
            raise AssertionError("streamed drain omitted the finished request")

        summary = {
            "request_id": outputs[0].request_id,
            "internal_request_id": internal_request_ids[0],
            "sampled_token_ids": list(sampled),
            "runner": type(worker.model_runner).__name__,
            "trace_sha256": worker.replay.trace_sha256,
            "step_record_count": len(records),
            "drain_finished_request_ids": records[-1]["finished_request_ids"],
            "drain_latency_ps": worker.step_results[-1].step_latency_ps,
            "step_schema": EXPECTED_STEP_SCHEMA,
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary
    finally:
        reset_configuration()
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    fixture = Path(__file__).parents[1] / "preplay_trace_v1/granite_length_cap.jsonl"
    if not MODEL.is_dir() or not fixture.is_file():
        raise SystemExit("cached model or tracked fixture is missing")
    if args.check_only:
        return
    summary = run_smoke(args.run_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
