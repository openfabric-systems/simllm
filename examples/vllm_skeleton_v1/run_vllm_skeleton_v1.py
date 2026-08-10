"""Run the import-free vLLM worker skeleton study."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from simllm.adapters.vllm import (
    SKELETON_INIT_CALL_SEQUENCE,
    SKELETON_STEP_CALL_SEQUENCE,
    SimModelRunner,
    SimWorker,
    step_records_to_json,
)
from simllm.core import VirtualClock

RESULTS = Path(__file__).with_name("results.csv")
DEFAULT_RUN_DIR = Path(
    "/data3/yifeng/simllm-dev/wave1-runs/"
    "codex_vllm13_skeleton_mode/vllm_skeleton_v1"
)
CLOCK_START_PS = 123_000


@dataclass
class FakeNewRequest:
    req_id: str
    prompt_token_ids: list[int]
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

    @property
    def total_num_scheduled_tokens(self) -> int:
        return sum(self.num_scheduled_tokens.values())


class FakeDType:
    itemsize = 2


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
        quant_config=None,
        use_v2_model_runner=False,
    )


@contextmanager
def patched_environment(**updates: str | None):
    previous = {name: os.environ.get(name) for name in updates}
    try:
        for name, value in updates.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def make_worker(clock: VirtualClock) -> SimWorker:
    return SimWorker(
        fake_vllm_config(),
        local_rank=0,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:1",
        is_driver_worker=True,
        _simllm_clock=clock,
    )


def drive_initialization(worker: SimWorker) -> list[str]:
    worker.init_device()
    worker.load_model()
    worker.get_kv_cache_spec()
    worker.determine_available_memory()
    worker.initialize_from_config(SimpleNamespace(num_blocks=64))
    worker.compile_or_warm_up_model()
    worker.reset_mm_cache()
    worker.get_supported_tasks()
    return [name for name in worker.mirrored_call_names if name.startswith("worker.")]


def drive_step(worker: SimWorker, scheduler_output: FakeSchedulerOutput) -> tuple[Any, list[str]]:
    call_start = len(worker.mirrored_calls)
    execute_output = worker.execute_model(scheduler_output)
    if execute_output is not None:
        raise AssertionError("nonempty V1 execute_model must return None before sampling")
    output = worker.sample_tokens(None)
    return output, worker.mirrored_call_names[call_start:]


def run_cell(request_count: int, prompt_tokens: int, run_dir: Path) -> dict[str, int | str]:
    stream_path = run_dir / f"r{request_count}_p{prompt_tokens}_steps.jsonl"
    with patched_environment(
        SIMLLM_VLLM_WORKER_MODE="skeleton",
        SIMLLM_VLLM_MODE="virtual",
        SIMLLM_VLLM_TOKEN_ID="512",
        SIMLLM_VLLM_STEP_RECORDS=str(stream_path),
    ):
        worker = make_worker(VirtualClock(start_ps=CLOCK_START_PS))
        init_sequence = drive_initialization(worker)
        runner = worker.model_runner

        req_ids = [f"r{index}" for index in range(request_count)]
        prefill = FakeSchedulerOutput(
            scheduled_new_reqs=[
                FakeNewRequest(req_id, list(range(prompt_tokens))) for req_id in req_ids
            ],
            num_scheduled_tokens={req_id: prompt_tokens for req_id in req_ids},
        )
        prefill_output, prefill_calls = drive_step(worker, prefill)
        decode = FakeSchedulerOutput(
            scheduled_cached_reqs=FakeCachedRequests(
                req_ids=req_ids,
                num_computed_tokens=[prompt_tokens] * request_count,
                num_output_tokens=[1] * request_count,
            ),
            num_scheduled_tokens={req_id: 1 for req_id in req_ids},
        )
        decode_output, decode_calls = drive_step(worker, decode)

    streamed = [json.loads(line) for line in stream_path.read_text().splitlines()]
    expected_total_new_tokens = request_count * (prompt_tokens + 1)
    scheduled_entries = sum(len(record.scheduled) for record in worker.step_records)
    sampled_tokens = sum(
        len(tokens)
        for output in (prefill_output, decode_output)
        for tokens in output.sampled_token_ids
    )
    exact_failures: list[str] = []
    for actual, expected, name in (
        (len(worker.step_records), 2, "step_count"),
        (scheduled_entries, 2 * request_count, "scheduled_entries"),
        (sampled_tokens, 2 * request_count, "sampled_tokens"),
        (
            sum(record.total_new_tokens for record in worker.step_records),
            expected_total_new_tokens,
            "total_new_tokens",
        ),
        (worker.step_records[0].total_new_tokens, request_count * prompt_tokens, "prefill"),
        (worker.step_records[1].total_new_tokens, request_count, "decode"),
    ):
        if actual != expected:
            exact_failures.append(f"{name}={actual} expected {expected}")

    structural_failures: list[str] = []
    if tuple(init_sequence) != SKELETON_INIT_CALL_SEQUENCE:
        structural_failures.append("init sequence mismatch")
    if tuple(prefill_calls) != SKELETON_STEP_CALL_SEQUENCE:
        structural_failures.append("prefill sequence mismatch")
    if tuple(decode_calls) != SKELETON_STEP_CALL_SEQUENCE:
        structural_failures.append("decode sequence mismatch")
    if worker.device is not None or not isinstance(runner, SimModelRunner):
        structural_failures.append("physical device or non-sim runner present")
    if runner is None or runner.runtime.clock is not worker.clock:
        structural_failures.append("runner does not share worker clock")
    if any(record.virtual_time_ps != CLOCK_START_PS for record in worker.step_records):
        structural_failures.append("record timestamp mismatch")
    if any(result.step_latency_ps != 0 for result in worker.step_results):
        structural_failures.append("nonzero empty-compute latency")
    if any(result.completed_at_ps != CLOCK_START_PS for result in worker.step_results):
        structural_failures.append("completion timestamp mismatch")
    if worker.clock.now_ps != CLOCK_START_PS:
        structural_failures.append("central clock advanced")
    if any(
        call.started_at_ps != CLOCK_START_PS
        or call.completed_at_ps != CLOCK_START_PS
        or call.completed_at_ps < call.started_at_ps
        for call in worker.mirrored_calls
    ):
        structural_failures.append("call timestamp mismatch")
    if streamed != step_records_to_json(worker.step_records):
        structural_failures.append("stream differs from in-memory records")
    if {entry.get("schema") for entry in streamed} != {"atlahs-closed-loop-step-v1"}:
        structural_failures.append("step schema mismatch")
    if len(worker.step_records) != len(worker.step_results):
        structural_failures.append("record/result cardinality mismatch")

    return {
        "request_count": request_count,
        "prompt_tokens": prompt_tokens,
        "step_count": len(worker.step_records),
        "scheduled_entries": scheduled_entries,
        "sampled_tokens": sampled_tokens,
        "prefill_new_tokens": worker.step_records[0].total_new_tokens,
        "decode_new_tokens": worker.step_records[1].total_new_tokens,
        "total_new_tokens": sum(record.total_new_tokens for record in worker.step_records),
        "expected_total_new_tokens": expected_total_new_tokens,
        "clock_start_ps": CLOCK_START_PS,
        "final_clock_ps": worker.clock.now_ps,
        "zero_latency_steps": sum(result.step_latency_ps == 0 for result in worker.step_results),
        "schema_records": len(streamed),
        "exact_oracle": "PASS" if not exact_failures else "; ".join(exact_failures),
        "structural_check": (
            "PASS" if not structural_failures else "; ".join(structural_failures)
        ),
    }


def check_flag_gates() -> list[str]:
    failures: list[str] = []
    for value in (None, "", "virtual"):
        with patched_environment(SIMLLM_VLLM_WORKER_MODE=value):
            try:
                make_worker(VirtualClock(start_ps=CLOCK_START_PS))
            except RuntimeError as exc:
                if "SIMLLM_VLLM_WORKER_MODE=skeleton" not in str(exc):
                    failures.append(f"gate {value!r} omitted the flag name")
            else:
                failures.append(f"gate {value!r} accepted construction")
    return failures


def check_relations(rows: list[dict[str, int | str]]) -> list[str]:
    by_key = {(row["request_count"], row["prompt_tokens"]): row for row in rows}
    failures: list[str] = []
    for prompt_tokens in (4, 16):
        one = by_key[(1, prompt_tokens)]
        three = by_key[(3, prompt_tokens)]
        failed_fields = []
        for name in ("scheduled_entries", "sampled_tokens", "total_new_tokens"):
            if int(three[name]) != 3 * int(one[name]):
                failed_fields.append(name)
        if failed_fields:
            failures.append(
                f"request fanout P={prompt_tokens} failed for {', '.join(failed_fields)}"
            )
    for request_count in (1, 3):
        short = by_key[(request_count, 4)]
        long = by_key[(request_count, 16)]
        failed_fields = []
        if int(long["prefill_new_tokens"]) != 4 * int(short["prefill_new_tokens"]):
            failed_fields.append("prefill")
        if int(long["decode_new_tokens"]) != int(short["decode_new_tokens"]):
            failed_fields.append("decode")
        if int(long["step_count"]) != int(short["step_count"]):
            failed_fields.append("step_count")
        if failed_fields:
            failures.append(
                f"prompt scaling R={request_count} failed for {', '.join(failed_fields)}"
            )
    return failures


def render_csv(rows: list[dict[str, int | str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if measured CSV differs from the tracked results",
    )
    arguments = parser.parse_args()
    arguments.run_dir.mkdir(parents=True, exist_ok=True)

    if "vllm" in sys.modules:
        raise SystemExit("study must not import vllm")
    gate_failures = check_flag_gates()
    rows = [
        run_cell(request_count, prompt_tokens, arguments.run_dir)
        for request_count in (1, 3)
        for prompt_tokens in (4, 16)
    ]
    if "vllm" in sys.modules:
        raise SystemExit("study imported vllm")

    relation_failures = check_relations(rows)
    rendered = render_csv(rows)
    if arguments.check:
        if not RESULTS.is_file() or RESULTS.read_bytes() != rendered:
            raise SystemExit(f"measured rows differ from tracked {RESULTS}")
        print(f"tracked results match {len(rows)} measured rows")
    else:
        RESULTS.write_bytes(rendered)
        print(f"wrote {len(rows)} rows to {RESULTS}")

    exact_passes = sum(row["exact_oracle"] == "PASS" for row in rows)
    structural_failures = [
        str(row["structural_check"])
        for row in rows
        if row["structural_check"] != "PASS"
    ]
    print(f"exact-oracle rows: {exact_passes}/{len(rows)} PASS")
    print(f"behavioral relation instances: {4 - len(relation_failures)}/4 PASS")
    print(
        "fatal structural cells: "
        f"{len(rows) - len(structural_failures)}/{len(rows)} PASS"
    )
    print(f"flag-gate negative controls: {3 - len(gate_failures)}/3 PASS")
    failures = gate_failures + relation_failures + structural_failures
    if exact_passes != len(rows):
        failures.extend(str(row["exact_oracle"]) for row in rows if row["exact_oracle"] != "PASS")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
