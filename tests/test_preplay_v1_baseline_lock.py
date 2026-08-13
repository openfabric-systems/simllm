"""Byte and timestamp locks for the accepted v1 join and absent-replay paths.

PLAY-8 teaches the pre-play join a second trace schema. These locks are the
guard that the already accepted paths keep producing the exact bytes and the
exact timestamps they produced before that change, proved here as pytest
against tracked baseline artifacts rather than only inside a study harness.

Every baseline under ``tests/fixtures/preplay`` was captured from the accepted
code before any version 2 join existed, so a later diff cannot quietly move the
reference. Two artifacts embed the resolved trace path, which is machine
specific: the comparison replaces exactly that one JSON string value with
``"<trace-path>"`` and asserts the substitution applied exactly once, so every
other byte, including every timestamp, is compared literally.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from simllm.adapters.vllm import (
    SimExecutorConfig,
    SimWorker,
    configure,
    reset_configuration,
)
from simllm.core import (
    RequestBookkeeper,
    StepRecord,
    StepResult,
    VirtualClock,
    bookkeeping_ledger_to_json,
)
from simllm.preplay import (
    ForwardPhase,
    ForwardTokenTrace,
    LayerRouting,
    PromptFormat,
    RequestArrival,
    RequestTrace,
    SamplingConfig,
    StopReason,
    TraceProvenance,
    join_preplay_arrivals,
    write_preplay_replay_run,
    write_preplay_trace,
)

FIXTURES = Path(__file__).parent / "fixtures/preplay"
TRACE_PLACEHOLDER = "<trace-path>"

#: the frozen linear step cost, in integer picoseconds
FIXED_STEP_PS = 1_000
NEW_TOKEN_PS = 100
CONTEXT_TOKEN_PS = 10

ARRIVALS_PS = {"alpha": 0, "beta": 500}
ORACLE_TOKENS = {"alpha": (101, 102, 0), "beta": (201,)}
PROMPTS = {"alpha": (10, 11), "beta": (20, 21, 22)}


@dataclass
class FakeNewRequest:
    """Shaped like ``vllm.v1.core.sched.output.NewRequestData``."""

    req_id: str
    prompt_token_ids: list[int]
    sampling_params: object | None = None
    num_computed_tokens: int = 0


@dataclass
class FakeCachedRequests:
    """Shaped like ``vllm.v1.core.sched.output.CachedRequestData``."""

    req_ids: list[str] = field(default_factory=list)
    num_computed_tokens: list[int] = field(default_factory=list)
    num_output_tokens: list[int] = field(default_factory=list)


@dataclass
class FakeSchedulerOutput:
    """Shaped like ``vllm.v1.core.sched.output.SchedulerOutput``."""

    scheduled_new_reqs: list[FakeNewRequest] = field(default_factory=list)
    scheduled_cached_reqs: FakeCachedRequests = field(default_factory=FakeCachedRequests)
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    finished_req_ids: set = field(default_factory=set)
    preempted_req_ids: set | None = None
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
        cache_config=SimpleNamespace(block_size=16, cache_dtype="auto", num_gpu_blocks=None),
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


def sampling_params(max_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(max_tokens=max_tokens, min_tokens=0, stop_token_ids=[])


class LinearStepSink:
    """Integer-only step cost, so the locked timestamps are bit exact."""

    def __call__(self, record: StepRecord) -> StepResult:
        latency_ps = (
            FIXED_STEP_PS
            + NEW_TOKEN_PS * sum(request.num_new_tokens for request in record.scheduled)
            + CONTEXT_TOKEN_PS * sum(request.context_length for request in record.scheduled)
        )
        return StepResult(
            step_index=record.step_index,
            step_latency_ps=latency_ps,
            completed_at_ps=record.virtual_time_ps + latency_ps,
        )


def _routing(phase: ForwardPhase, token_index: int, token_id: int) -> ForwardTokenTrace:
    return ForwardTokenTrace(
        phase=phase,
        token_index=token_index,
        token_id=token_id,
        routing=(LayerRouting(layer_index=0, expert_ids=(0,), gate_weights=(1.0,)),),
    )


def _request(request_id: str) -> RequestTrace:
    prompt = PROMPTS[request_id]
    output = ORACLE_TOKENS[request_id]
    return RequestTrace(
        request_id=request_id,
        prompt_sha256=hashlib.sha256(request_id.encode()).hexdigest(),
        prompt_format=PromptFormat.TEXT,
        input_token_ids=prompt,
        max_new_tokens=len(output),
        stop_strings=(),
        output_text=request_id,
        output_token_ids=output,
        stop_reason=StopReason.EOS if output[-1] == 0 else StopReason.LENGTH_CAP,
        matched_stop_string=None,
        prefill_tokens=tuple(
            _routing(ForwardPhase.PREFILL, index, token_id)
            for index, token_id in enumerate(prompt)
        ),
        decode_tokens=tuple(
            _routing(ForwardPhase.DECODE, index, token_id)
            for index, token_id in enumerate(output[:-1])
        ),
    )


def _provenance() -> TraceProvenance:
    return TraceProvenance(
        model_id="test/model",
        model_revision="test-revision",
        model_class="TestMoeForCausalLM",
        dtype="float32",
        tokenizer_sha256="a" * 64,
        sampling=SamplingConfig.greedy(),
        capture_host="test-host",
        runner="test-runner",
        transformers_version="5.14.1",
        torch_version="2.11.0",
        device="cpu",
        torch_num_threads=1,
        eos_token_id=0,
        top_k=1,
        expert_count=1,
        moe_layer_indices=(0,),
    )


def write_baseline_trace(directory: Path) -> Path:
    """Write the deterministic v1 trace both locked paths are joined from."""

    return write_preplay_trace(
        directory / "baseline-trace.jsonl",
        _provenance(),
        (_request("alpha"), _request("beta")),
    )


def build_v1_join(directory: Path) -> tuple[Path, Path, RequestBookkeeper]:
    """Join the baseline trace exactly the way the accepted v1 path does."""

    trace_path = write_baseline_trace(directory)
    bookkeeper = RequestBookkeeper()
    run = join_preplay_arrivals(
        (
            RequestArrival(request_id="alpha", arrived_at_ps=ARRIVALS_PS["alpha"]),
            RequestArrival(request_id="beta", arrived_at_ps=ARRIVALS_PS["beta"]),
        ),
        trace_path,
        bookkeeper,
    )
    run_path = write_preplay_replay_run(run, directory / "replay-run.json")
    return trace_path, run_path, bookkeeper


def normalized(payload: bytes, trace_path: Path) -> bytes:
    """Replace exactly the one machine-specific trace path with a placeholder."""

    encoded = json.dumps(str(trace_path.resolve()))[1:-1].encode("utf-8")
    if payload.count(encoded) < 1:
        raise AssertionError("artifact does not name the trace path at all")
    return payload.replace(encoded, TRACE_PLACEHOLDER.encode("utf-8"))


def canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def baseline_schedule() -> list[FakeSchedulerOutput]:
    """One frozen schedule, identical for the replay and absent-replay paths."""

    return [
        FakeSchedulerOutput(
            scheduled_new_reqs=[
                FakeNewRequest("alpha", list(PROMPTS["alpha"]), sampling_params(3))
            ],
            num_scheduled_tokens={"alpha": 2},
        ),
        FakeSchedulerOutput(
            scheduled_new_reqs=[
                FakeNewRequest("beta", list(PROMPTS["beta"]), sampling_params(1))
            ],
            scheduled_cached_reqs=FakeCachedRequests(["alpha"], [2], [1]),
            num_scheduled_tokens={"beta": 3, "alpha": 1},
        ),
        FakeSchedulerOutput(
            scheduled_cached_reqs=FakeCachedRequests(["alpha"], [3], [2]),
            num_scheduled_tokens={"alpha": 1},
            finished_req_ids={"beta"},
        ),
        FakeSchedulerOutput(finished_req_ids={"alpha"}),
    ]


def drive_worker(directory: Path, replay_run_path: Path | None, label: str) -> tuple[Path, bytes]:
    """Drive the frozen schedule once and return its stream path and results."""

    stream_path = directory / f"{label}_steps.jsonl"
    served: dict[str, list[int]] = {"alpha": [], "beta": []}
    reset_configuration()
    try:
        configure(
            step_sink=LinearStepSink(),
            config=SimExecutorConfig(
                mode="virtual",
                token_id=512,
                step_records_path=str(stream_path),
                replay_run_path=None if replay_run_path is None else str(replay_run_path),
            ),
        )
        worker = SimWorker(
            fake_vllm_config(),
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:1",
            is_driver_worker=True,
            _simllm_clock=VirtualClock(),
        )
        worker.init_device()
        for scheduler_output in baseline_schedule():
            output = worker.execute_model(scheduler_output)
            if scheduler_output.num_scheduled_tokens:
                if output is not None:
                    raise AssertionError("a nonempty skeleton step must split sampling")
                output = worker.sample_tokens(None)
            if output is None:
                continue
            for request_id, token_ids in zip(
                output.req_ids, output.sampled_token_ids or (), strict=True
            ):
                served[request_id].extend(token_ids)
        results = canonical_bytes(
            {
                "served_token_ids": {
                    request_id: tokens for request_id, tokens in sorted(served.items())
                },
                "step_results": [
                    {
                        "step_index": result.step_index,
                        "step_latency_ps": result.step_latency_ps,
                        "completed_at_ps": result.completed_at_ps,
                    }
                    for result in worker.step_results
                ],
                "final_clock_ps": worker.clock.now_ps,
            }
        )
    finally:
        reset_configuration()
    return stream_path, results


def test_v1_join_replay_run_is_byte_locked(tmp_path):
    trace_path, run_path, _ = build_v1_join(tmp_path)
    observed = normalized(run_path.read_bytes(), trace_path)
    assert observed == (FIXTURES / "v1_join_replay_run.json").read_bytes()


def test_v1_join_bookkeeping_is_timestamp_locked(tmp_path):
    trace_path, _, bookkeeper = build_v1_join(tmp_path)
    ledger = canonical_bytes(bookkeeping_ledger_to_json(bookkeeper.snapshot()))
    observed = normalized(ledger, trace_path)
    assert observed == (FIXTURES / "v1_join_bookkeeping.json").read_bytes()


def test_v1_replay_worker_stream_is_byte_locked(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMLLM_VLLM_WORKER_MODE", "skeleton")
    _, run_path, _ = build_v1_join(tmp_path)
    stream_path, results = drive_worker(tmp_path, run_path, "v1_replay")
    assert stream_path.read_bytes() == (FIXTURES / "v1_replay_steps.jsonl").read_bytes()
    assert results == (FIXTURES / "v1_replay_results.json").read_bytes()


def test_absent_replay_worker_stream_is_byte_locked(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMLLM_VLLM_WORKER_MODE", "skeleton")
    stream_path, results = drive_worker(tmp_path, None, "absent_replay")
    assert stream_path.read_bytes() == (FIXTURES / "absent_replay_steps.jsonl").read_bytes()
    assert results == (FIXTURES / "absent_replay_results.json").read_bytes()
