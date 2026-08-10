"""vLLM adapter tests using only transcribed inputs.

The tests never import vLLM directly. Adapter imports select the real worker
when vLLM is installed and the transcribed worker base otherwise, so the same
mirror tests run in both environments.
"""

import importlib.util
import json
import sys
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import ClassVar

import pytest

from simllm.adapters.vllm import (
    SKELETON_EMPTY_STEP_CALL_SEQUENCE,
    SKELETON_INIT_CALL_SEQUENCE,
    SKELETON_STEP_CALL_SEQUENCE,
    ModelDims,
    PlacementExporter,
    SimExecutor,
    SimExecutorConfig,
    SimModelRunner,
    SimWorker,
    StepTranslator,
    fabricate_sampled_tokens,
    manifest_from_worker_entries,
    step_kernel,
    step_records_to_json,
    translate_scheduler_output,
    write_step_records,
)
from simllm.compute import GpuSpec, HostInitiationModel, RooflineProvider
from simllm.core import RequestPhase, VirtualClock
from simllm.placement import PlacementManifest

VLLM_INSTALLED = importlib.util.find_spec("vllm") is not None

# SchedulerOutput stubs: same attribute names as vLLM v0.26.0, nothing else


@dataclass
class FakeNewRequest:
    """Shaped like ``vllm.v1.core.sched.output.NewRequestData``."""

    req_id: str
    prompt_token_ids: list[int]
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


def prompt(length: int) -> list[int]:
    return list(range(length))


def llama8b_dims() -> ModelDims:
    """Llama-3.1-8B geometry on one GPU (TP=1), i.e. 7.0 G non-embedding params."""
    return ModelDims(
        num_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
        vocab_size=128256,
    )


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


def make_sim_worker(
    clock=None,
    *,
    vllm_config=None,
    rank=0,
    is_driver_worker=True,
):
    if vllm_config is None:
        vllm_config = fake_vllm_config()
    return SimWorker(
        vllm_config,
        local_rank=0,
        rank=rank,
        distributed_init_method="tcp://127.0.0.1:1",
        is_driver_worker=is_driver_worker,
        _simllm_clock=clock,
    )


# Import surface

@pytest.mark.skipif(VLLM_INSTALLED, reason="the executor's guarded import pulls vLLM in when present")
def test_package_imports_without_vllm():
    # The package must not drag vLLM in, and neither must the executor module
    # that vLLM itself imports early (config post-init reads
    # supports_async_scheduling before any engine exists).
    assert "vllm" not in sys.modules
    import simllm.adapters.vllm.executor  # noqa: F401 - the import is the test

    assert "vllm" not in sys.modules
    assert SimExecutor.supports_async_scheduling() is False


def test_pinned_version_has_one_source():
    import simllm.adapters.vllm as package
    import simllm.adapters.vllm.executor as executor_module

    assert package.PINNED_VLLM_VERSION is executor_module.PINNED_VLLM_VERSION
    assert package.PINNED_VLLM_VERSION == "0.26.0"


def test_lazy_exports_do_not_import_the_executor_eagerly():
    # Re-importing leaves fresh module objects behind, so the originals are
    # restored afterwards: the executor module carries process-wide singletons
    # (_HOOKS, _LATEST) that later tests must keep sharing.
    names = ("simllm.adapters.vllm", "simllm.adapters.vllm.executor")
    saved = {name: sys.modules.get(name) for name in names}
    try:
        for name in names:
            sys.modules.pop(name, None)
        package = importlib.import_module("simllm.adapters.vllm")
        assert "simllm.adapters.vllm.executor" not in sys.modules
        assert package.SimExecutor.__name__ == "SimExecutor"
        assert "simllm.adapters.vllm.executor" in sys.modules
        with pytest.raises(AttributeError):
            _ = package.NotAnExport
    finally:
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)


@pytest.mark.skipif(VLLM_INSTALLED, reason="vLLM present, construction should succeed")
def test_construction_without_vllm_raises_a_clear_error():
    with pytest.raises(ImportError) as excinfo:
        SimExecutor(object())
    message = str(excinfo.value)
    assert "vLLM v0.26.0" in message
    assert "distributed-executor-backend" in message


def test_sim_worker_requires_the_explicit_skeleton_flag(monkeypatch):
    for value in (None, "", "virtual"):
        if value is None:
            monkeypatch.delenv("SIMLLM_VLLM_WORKER_MODE", raising=False)
        else:
            monkeypatch.setenv("SIMLLM_VLLM_WORKER_MODE", value)
        with pytest.raises(RuntimeError) as excinfo:
            make_sim_worker()
        assert "SIMLLM_VLLM_WORKER_MODE=skeleton" in str(excinfo.value)


def test_sim_executor_refuses_structured_scheduler_output():
    executor = object.__new__(SimExecutor)
    scheduler_output = FakeSchedulerOutput(
        num_scheduled_tokens={"r0": 1},
        has_structured_output_requests=True,
    )
    with pytest.raises(RuntimeError, match="VLLM-8"):
        executor.execute_model(scheduler_output)


def test_sim_worker_rejects_v2_and_device_requiring_executor_paths(monkeypatch):
    monkeypatch.setenv("SIMLLM_VLLM_WORKER_MODE", "skeleton")

    v2_config = fake_vllm_config()
    v2_config.use_v2_model_runner = True
    with pytest.raises(RuntimeError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
        make_sim_worker(vllm_config=v2_config)

    for backend in ("ray", "external_launcher"):
        backend_config = fake_vllm_config()
        backend_config.parallel_config.distributed_executor_backend = backend
        with pytest.raises(RuntimeError, match=backend):
            make_sim_worker(vllm_config=backend_config)

    async_config = fake_vllm_config()
    async_config.scheduler_config.async_scheduling = True
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    with pytest.raises(RuntimeError, match="no-async-scheduling"):
        make_sim_worker(vllm_config=async_config)

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    assert make_sim_worker(vllm_config=async_config).is_time_authority


def test_sim_worker_mirrors_the_source_frozen_init_sequence(monkeypatch):
    monkeypatch.setenv("SIMLLM_VLLM_WORKER_MODE", "skeleton")
    worker = make_sim_worker(VirtualClock(start_ps=123_000))
    assert worker.device is None
    assert worker.model_runner is None

    worker.init_device()
    worker.load_model()
    specs = worker.get_kv_cache_spec()
    available = worker.determine_available_memory()
    worker.initialize_from_config(SimpleNamespace(num_blocks=64))
    compilation = worker.compile_or_warm_up_model()
    worker.reset_mm_cache()
    tasks = worker.get_supported_tasks()

    assert isinstance(worker.model_runner, SimModelRunner)
    assert worker.device is None
    assert len(specs) == 4
    assert available == 64 * 1024**3
    assert worker.cache_config.num_gpu_blocks == 64
    assert (compilation.language_model, compilation.encoder) == (0.0, 0.0)
    assert tasks == ("generate",)
    worker_calls = tuple(
        name for name in worker.mirrored_call_names if name.startswith("worker.")
    )
    assert worker_calls == SKELETON_INIT_CALL_SEQUENCE
    assert all(
        call.started_at_ps == call.completed_at_ps == 123_000
        for call in worker.mirrored_calls
    )


def test_sim_worker_split_step_uses_one_clock_and_stream(monkeypatch, tmp_path):
    stream_path = tmp_path / "skeleton_steps.jsonl"
    monkeypatch.setenv("SIMLLM_VLLM_WORKER_MODE", "skeleton")
    monkeypatch.setenv("SIMLLM_VLLM_STEP_RECORDS", str(stream_path))
    worker = make_sim_worker(VirtualClock(start_ps=123_000))
    worker.init_device()
    runner = worker.model_runner
    assert isinstance(runner, SimModelRunner)
    assert runner.runtime.clock is worker.clock

    prefill = FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewRequest("r0", prompt(4))],
        num_scheduled_tokens={"r0": 4},
    )
    call_start = len(worker.mirrored_calls)
    assert worker.execute_model(prefill) is None
    prefill_output = worker.sample_tokens(None)
    assert tuple(worker.mirrored_call_names[call_start:]) == SKELETON_STEP_CALL_SEQUENCE
    assert prefill_output.req_ids == ["r0"]
    assert prefill_output.sampled_token_ids == [[512]]

    decode = FakeSchedulerOutput(
        scheduled_cached_reqs=FakeCachedRequests(["r0"], [4], [1]),
        num_scheduled_tokens={"r0": 1},
    )
    call_start = len(worker.mirrored_calls)
    assert worker.execute_model(decode) is None
    decode_output = worker.sample_tokens(None)
    assert tuple(worker.mirrored_call_names[call_start:]) == SKELETON_STEP_CALL_SEQUENCE
    assert decode_output.sampled_token_ids == [[512]]

    assert [record.step_index for record in worker.step_records] == [0, 1]
    assert [record.virtual_time_ps for record in worker.step_records] == [123_000, 123_000]
    assert [record.total_new_tokens for record in worker.step_records] == [4, 1]
    assert [record.num_sampled for record in worker.step_records] == [1, 1]
    assert [result.step_latency_ps for result in worker.step_results] == [0, 0]
    assert [result.completed_at_ps for result in worker.step_results] == [123_000, 123_000]
    assert worker.clock.now_ps == 123_000
    assert all(
        call.started_at_ps == call.completed_at_ps == 123_000
        for call in worker.mirrored_calls
    )
    streamed = [json.loads(line) for line in stream_path.read_text().splitlines()]
    assert streamed == step_records_to_json(worker.step_records)
    assert {entry["schema"] for entry in streamed} == {"atlahs-closed-loop-step-v1"}


def test_sim_runner_serves_dp_coordination_then_tp_collective(monkeypatch):
    monkeypatch.setenv("SIMLLM_VLLM_WORKER_MODE", "skeleton")
    config = fake_vllm_config()
    config.parallel_config.tensor_parallel_size = 4
    config.parallel_config.data_parallel_size = 4
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.world_size = 4
    worker = make_sim_worker(VirtualClock(start_ps=123_000), vllm_config=config)
    worker.init_device()

    step = FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewRequest("r0", prompt(4))],
        num_scheduled_tokens={"r0": 4},
    )
    assert worker.execute_model(step) is None
    events = worker.coordinator_events

    assert tuple((event.operation, event.group, event.payload_bytes) for event in events) == (
        ("all_reduce", "dp", 64),
        ("all_reduce", "tp", 4_096),
    )
    assert tuple(len(event.stack_events) for event in events) == (32, 14)
    assert all(event.timestamp_ps == 123_000 for event in events)
    assert worker.step_records[0].num_tokens_after_padding == 4
    assert step_records_to_json(worker.step_records)[0]["num_tokens_after_padding"] == 4
    assert worker.sample_tokens(None).sampled_token_ids == [[512]]
    assert worker.clock.now_ps == 123_000


def test_dp_coordinator_return_controls_the_padding_record_field(monkeypatch):
    monkeypatch.setenv("SIMLLM_VLLM_WORKER_MODE", "skeleton")
    config = fake_vllm_config()
    config.parallel_config.data_parallel_size = 4
    worker = make_sim_worker(VirtualClock(start_ps=123_000), vllm_config=config)
    worker.init_device()
    original_all_reduce = worker.dp_group.all_reduce

    def return_different_padding(input_):
        output = original_all_reduce(input_)
        return replace(output, num_tokens_across_dp=(9, 9, 9, 9))

    monkeypatch.setattr(worker.dp_group, "all_reduce", return_different_padding)
    step = FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewRequest("r0", prompt(4))],
        num_scheduled_tokens={"r0": 4},
    )

    assert worker.execute_model(step) is None
    assert worker.step_records[0].num_tokens_after_padding == 9
    assert step_records_to_json(worker.step_records)[0]["num_tokens_after_padding"] == 9
    assert worker.sample_tokens(None).sampled_token_ids == [[512]]


def test_sim_worker_empty_completion_preserves_v1_update_order(monkeypatch):
    monkeypatch.setenv("SIMLLM_VLLM_WORKER_MODE", "skeleton")
    worker = make_sim_worker(VirtualClock(start_ps=123_000))
    worker.init_device()
    prefill = FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewRequest("r0", prompt(4))],
        num_scheduled_tokens={"r0": 4},
    )
    assert worker.execute_model(prefill) is None
    worker.sample_tokens(None)

    drain = FakeSchedulerOutput(finished_req_ids={"r0"})
    call_start = len(worker.mirrored_calls)
    output = worker.execute_model(drain)
    assert output.req_ids == []
    assert tuple(worker.mirrored_call_names[call_start:]) == (
        SKELETON_EMPTY_STEP_CALL_SEQUENCE
    )
    assert [record.step_index for record in worker.step_records] == [0, 1]
    assert worker.step_records[1].finished_request_ids == ["r0"]
    assert worker.step_results[1].step_latency_ps == 0
    assert worker.clock.now_ps == 123_000


def test_only_global_rank_zero_owns_records_and_stream(monkeypatch, tmp_path):
    stream_path = tmp_path / "non_authority.jsonl"
    monkeypatch.setenv("SIMLLM_VLLM_WORKER_MODE", "skeleton")
    monkeypatch.setenv("SIMLLM_VLLM_STEP_RECORDS", str(stream_path))
    config = fake_vllm_config()
    config.parallel_config.tensor_parallel_size = 2
    config.parallel_config.world_size = 2
    worker = make_sim_worker(
        VirtualClock(start_ps=123_000),
        vllm_config=config,
        rank=1,
        is_driver_worker=True,
    )
    worker.init_device()
    prefill = FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewRequest("r0", prompt(4))],
        num_scheduled_tokens={"r0": 4},
    )
    assert worker.execute_model(prefill) is None
    worker.sample_tokens(None)

    assert not worker.is_time_authority
    assert worker.step_records == []
    assert worker.step_results == []
    assert not stream_path.exists()


def test_nonzero_data_parallel_rank_is_not_a_time_authority(monkeypatch):
    monkeypatch.setenv("SIMLLM_VLLM_WORKER_MODE", "skeleton")
    config = fake_vllm_config()
    config.parallel_config.data_parallel_size = 2
    config.parallel_config.data_parallel_rank = 1
    worker = make_sim_worker(vllm_config=config, rank=0)

    assert not worker.is_time_authority
    assert worker.dp_group.rank == 1
    assert worker.dp_group.ranks == [0, 1]


def test_worker_extension_exposes_exactly_one_public_name():
    # vLLM asserts that no non-dunder attribute of the extension class exists
    # on the worker class, so every added public name is a collision risk.
    public = [name for name in dir(PlacementExporter) if not name.startswith("__")]
    assert public == ["simllm_placement_entry"]


# Step translation

def test_chunked_prefill_then_decode_translation():
    translator = StepTranslator()
    # 1000-token prompt, 128 tokens served from the prefix cache, chunked at
    # 512 tokens per step.
    first = FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewRequest("r0", prompt(1000), num_computed_tokens=128)],
        num_scheduled_tokens={"r0": 512},
    )
    step0 = translate_scheduler_output(translator, first, step_index=0, virtual_time_ps=0)
    request = step0.record.scheduled[0]
    assert request.phase is RequestPhase.PREFILL
    assert (request.num_new_tokens, request.num_cached_tokens) == (512, 128)
    assert request.context_length == 640
    assert step0.produces_token == [False]
    assert step0.num_sampled == 0
    assert step0.record.num_sampled == 0

    second = FakeSchedulerOutput(
        scheduled_cached_reqs=FakeCachedRequests(["r0"], [640], [0]),
        num_scheduled_tokens={"r0": 360},
    )
    step1 = translate_scheduler_output(translator, second, step_index=1, virtual_time_ps=1_000)
    request = step1.record.scheduled[0]
    assert request.phase is RequestPhase.PREFILL
    assert request.context_length == 1000
    # The prefix hit is reported once, on admission, not every step.
    assert request.num_cached_tokens == 0
    assert step1.produces_token == [True]
    assert step1.record.num_sampled == 1

    third = FakeSchedulerOutput(
        scheduled_cached_reqs=FakeCachedRequests(["r0"], [1000], [1]),
        num_scheduled_tokens={"r0": 1},
    )
    step2 = translate_scheduler_output(translator, third, step_index=2, virtual_time_ps=2_000)
    request = step2.record.scheduled[0]
    assert request.phase is RequestPhase.DECODE
    assert (request.num_new_tokens, request.context_length) == (1, 1001)
    assert step2.produces_token == [True]
    assert step2.record.num_sampled == 1
    assert step2.record.total_new_tokens == 1


def test_mixed_batch_translation_and_bookkeeping():
    translator = StepTranslator()
    output = FakeSchedulerOutput(
        scheduled_new_reqs=[
            FakeNewRequest("new", prompt(300)),
            FakeNewRequest("short", prompt(4)),
        ],
        scheduled_cached_reqs=FakeCachedRequests(
            ["decoding"], num_computed_tokens=[50], num_output_tokens=[10]
        ),
        num_scheduled_tokens={"new": 128, "short": 4, "decoding": 1},
        finished_req_ids={"gone", "also-gone"},
        preempted_req_ids={"evicted"},
    )
    step = translate_scheduler_output(translator, output, step_index=7, virtual_time_ps=42)
    record = step.record
    assert record.step_index == 7
    assert record.virtual_time_ps == 42
    assert record.total_new_tokens == 133
    phases = {r.request_id: r.phase for r in record.scheduled}
    assert phases == {
        "new": RequestPhase.PREFILL,
        "short": RequestPhase.PREFILL,
        "decoding": RequestPhase.DECODE,
    }
    # A prompt that fits in one chunk samples immediately; a chunked one does not.
    assert dict(zip(step.req_ids, step.produces_token)) == {
        "new": False,
        "short": True,
        "decoding": True,
    }
    assert record.num_sampled == 2
    # Finished and preempted ids are sorted for reproducible traces.
    assert record.finished_request_ids == ["also-gone", "gone"]
    assert record.preempted_request_ids == ["evicted"]
    # The two prefills plus the decoding request are still tracked; finished
    # ids never entered the table.
    assert len(translator) == 3


def test_finished_requests_are_forgotten():
    translator = StepTranslator()
    admit = FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewRequest("r0", prompt(8))],
        num_scheduled_tokens={"r0": 8},
    )
    translate_scheduler_output(translator, admit, step_index=0, virtual_time_ps=0)
    assert len(translator) == 1
    finish = FakeSchedulerOutput(
        scheduled_cached_reqs=FakeCachedRequests(["r0"], [8], [1]),
        num_scheduled_tokens={"r0": 1},
        finished_req_ids={"r0"},
    )
    translate_scheduler_output(translator, finish, step_index=1, virtual_time_ps=1)
    assert len(translator) == 0


def test_drain_step_translation_carries_the_last_completions():
    # The scheduler stays live while its finished set is non-empty, so the
    # engine's final step schedules nothing and carries the last finished
    # ids. The translation must not drop them (they are the completions of
    # the previous step; preempted ids by contrast are same-step).
    translator = StepTranslator()
    admit = FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewRequest("r0", prompt(8))],
        num_scheduled_tokens={"r0": 8},
    )
    translate_scheduler_output(translator, admit, step_index=0, virtual_time_ps=0)
    drain = FakeSchedulerOutput(
        num_scheduled_tokens={},
        finished_req_ids={"r0"},
    )
    step = translate_scheduler_output(translator, drain, step_index=1, virtual_time_ps=9_000)
    assert step.record.scheduled == []
    assert step.record.finished_request_ids == ["r0"]
    assert step.record.total_new_tokens == 0
    assert step.num_sampled == 0
    assert step.record.num_sampled == 0
    assert len(translator) == 0


def test_request_seen_only_as_cached_reconstructs_its_prompt_length():
    # Attaching mid-flight: the request was admitted before this executor
    # existed, so prompt_len comes from computed minus output tokens.
    translator = StepTranslator()
    output = FakeSchedulerOutput(
        scheduled_cached_reqs=FakeCachedRequests(["orphan"], [500], [20]),
        num_scheduled_tokens={"orphan": 1},
    )
    step = translate_scheduler_output(translator, output, step_index=0, virtual_time_ps=0)
    assert step.record.scheduled[0].phase is RequestPhase.DECODE
    assert step.produces_token == [True]
    assert step.record.num_sampled == 1


def test_preemption_resets_the_computed_count():
    translator = StepTranslator()
    admit = FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewRequest("r0", prompt(64))],
        num_scheduled_tokens={"r0": 64},
    )
    translate_scheduler_output(translator, admit, step_index=0, virtual_time_ps=0)
    # Preempted and resumed: the scheduler reports computed=0 again, so the
    # request is back in prefill even though it had finished one before.
    resumed = FakeSchedulerOutput(
        scheduled_cached_reqs=FakeCachedRequests(["r0"], [0], [1]),
        num_scheduled_tokens={"r0": 64},
    )
    step = translate_scheduler_output(translator, resumed, step_index=1, virtual_time_ps=1)
    assert step.record.scheduled[0].phase is RequestPhase.PREFILL
    assert step.record.scheduled[0].context_length == 64
    assert step.produces_token == [True]
    assert step.record.num_sampled == 1


# ModelRunnerOutput fabrication

def test_fabricate_sampled_tokens_shape():
    req_ids, req_id_to_index, sampled = fabricate_sampled_tokens(
        ["a", "b", "c"], [True, False, True], token_id=1234
    )
    assert req_ids == ["a", "b", "c"]
    assert req_id_to_index == {"a": 0, "b": 1, "c": 2}
    # One inner list per request, in req_ids order; empty means "no token".
    assert sampled == [[1234], [], [1234]]


def test_fabricated_output_covers_every_scheduled_request():
    # Scheduler.update_from_output indexes req_id_to_index[req_id] for every
    # entry of num_scheduled_tokens, so a missing id is a KeyError crash.
    translator = StepTranslator()
    output = FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewRequest("a", prompt(10)), FakeNewRequest("b", prompt(2000))],
        scheduled_cached_reqs=FakeCachedRequests(["c"], [700], [3]),
        num_scheduled_tokens={"a": 10, "b": 512, "c": 1},
    )
    step = translate_scheduler_output(translator, output, step_index=0, virtual_time_ps=0)
    req_ids, req_id_to_index, sampled = fabricate_sampled_tokens(
        step.req_ids, step.produces_token, token_id=7
    )
    assert set(req_id_to_index) == set(output.num_scheduled_tokens)
    assert len(sampled) == len(req_ids)
    assert sampled[req_id_to_index["a"]] == [7]
    assert sampled[req_id_to_index["b"]] == []
    assert sampled[req_id_to_index["c"]] == [7]
    assert sum(bool(tokens) for tokens in sampled) == step.record.num_sampled


def test_fabricate_rejects_inconsistent_input():
    with pytest.raises(ValueError, match="same length"):
        fabricate_sampled_tokens(["a", "b"], [True], token_id=1)
    with pytest.raises(ValueError, match="duplicate request id"):
        fabricate_sampled_tokens(["a", "a"], [True, True], token_id=1)


# Step cost model

def test_step_kernel_counts_prefill_and_decode_work():
    dims = llama8b_dims()
    translator = StepTranslator()
    prefill = FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewRequest("r0", prompt(512))],
        num_scheduled_tokens={"r0": 512},
    )
    step = translate_scheduler_output(translator, prefill, step_index=0, virtual_time_ps=0)
    kernel = step_kernel(dims, step.record, step.num_sampled)
    dense = 2 * 512 * (dims.attention_params + dims.mlp_params)
    attention = 4 * (512 * 512 // 2) * 32 * 32 * 128
    head = 2 * 1 * 4096 * 128256
    assert kernel.flops == pytest.approx(dense + attention + head)
    assert kernel.config == (("new_tokens", 512), ("sampled", 1), ("kv_tokens", 512))

    decode = FakeSchedulerOutput(
        scheduled_cached_reqs=FakeCachedRequests(["r0"], [512], [1]),
        num_scheduled_tokens={"r0": 1},
    )
    step = translate_scheduler_output(translator, decode, step_index=1, virtual_time_ps=0)
    kernel = step_kernel(dims, step.record, step.num_sampled)
    kv_bytes = 2 * 513 * 32 * 8 * 128 * 2
    assert kernel.bytes_moved == pytest.approx(
        dims.weight_bytes + dims.lm_head_bytes + kv_bytes
    )


def test_roofline_bound_flips_between_prefill_and_decode():
    dims = llama8b_dims()
    gpu = GpuSpec(name="b100", peak_flops=1.8e15, mem_bandwidth=8.0e12)
    provider = RooflineProvider(efficiency=0.7)
    translator = StepTranslator()

    prefill = FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewRequest("r0", prompt(4096))],
        num_scheduled_tokens={"r0": 4096},
    )
    step = translate_scheduler_output(translator, prefill, step_index=0, virtual_time_ps=0)
    assert provider.estimate(step_kernel(dims, step.record, 1), gpu).bound == "compute"

    decode = FakeSchedulerOutput(
        scheduled_cached_reqs=FakeCachedRequests(["r0"], [4096], [1]),
        num_scheduled_tokens={"r0": 1},
    )
    step = translate_scheduler_output(translator, decode, step_index=1, virtual_time_ps=0)
    # A single decode token reads all the weights and computes almost nothing.
    assert provider.estimate(step_kernel(dims, step.record, 1), gpu).bound == "memory"


def test_kv_cache_dtype_sizes_the_kv_read_independently():
    # --kv-cache-dtype fp8 halves the KV term but not the weight read; the
    # bf16-only construction keeps the historical numbers bit for bit.
    dims16 = llama8b_dims()
    dims8 = ModelDims(
        num_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
        vocab_size=128256,
        kv_dtype_bytes=1.0,
    )
    translator = StepTranslator()
    decode = FakeSchedulerOutput(
        scheduled_cached_reqs=FakeCachedRequests(["r0"], [2048], [1]),
        num_scheduled_tokens={"r0": 1},
    )
    step = translate_scheduler_output(translator, decode, step_index=0, virtual_time_ps=0)
    kernel16 = step_kernel(dims16, step.record, 1)
    kernel8 = step_kernel(dims8, step.record, 1)
    kv16 = 2 * 2049 * 32 * 8 * 128 * 2
    assert kernel16.bytes_moved - kernel8.bytes_moved == pytest.approx(kv16 / 2)
    assert kernel16.flops == pytest.approx(kernel8.flops)


def test_weight_quantization_sizes_the_weight_read_independently():
    dense = llama8b_dims()
    quantized = ModelDims(
        num_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
        vocab_size=128256,
        weight_dtype_bytes=0.5,
    )
    assert quantized.weight_bytes == dense.weight_bytes // 4
    assert quantized.lm_head_bytes == dense.lm_head_bytes // 4
    assert quantized.kv_element_bytes == 2.0


def test_quant_and_kv_dtype_heuristics():
    from simllm.adapters.vllm.executor import (
        _kv_element_bytes_from_cache_config,
        _weight_element_bytes_from_quant,
    )

    class FakeCacheConfig:
        cache_dtype = "fp8_e4m3"

    class FakeQuant:
        @staticmethod
        def get_name():
            return "gptq_marlin"

    class UnknownQuant:
        @staticmethod
        def get_name():
            return "mystery"

    assert _kv_element_bytes_from_cache_config(FakeCacheConfig(), 2.0) == 1.0
    assert _kv_element_bytes_from_cache_config(None, 2.0) == 2.0
    assert _weight_element_bytes_from_quant(FakeQuant(), 2.0) == 0.5
    assert _weight_element_bytes_from_quant(None, 2.0) == 2.0
    # Unrecognized method: keep the activation width (and warn) rather than guess.
    assert _weight_element_bytes_from_quant(UnknownQuant(), 2.0) == 2.0


def test_estimate_adds_the_host_initiation_delay():
    from simllm.adapters.vllm import estimate_step_latency_ps

    dims = llama8b_dims()
    gpu = GpuSpec(name="b100", peak_flops=1.8e15, mem_bandwidth=8.0e12)
    provider = RooflineProvider(efficiency=0.7)
    translator = StepTranslator()
    output = FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewRequest("r0", prompt(128))],
        num_scheduled_tokens={"r0": 128},
    )
    step = translate_scheduler_output(translator, output, step_index=0, virtual_time_ps=0)
    ideal = estimate_step_latency_ps(
        dims, step.record, 1, provider, gpu, HostInitiationModel()
    )
    proxied = estimate_step_latency_ps(
        dims,
        step.record,
        1,
        provider,
        gpu,
        HostInitiationModel(initiation_delay_ps=5_000_000, profile="cpu-proxy"),
    )
    assert proxied - ideal == 5_000_000


# Configuration

def test_config_from_env_reads_every_knob():
    config = SimExecutorConfig.from_env(
        {
            "SIMLLM_VLLM_MODE": "paced",
            "SIMLLM_VLLM_KV_MEMORY_BYTES": "1000",
            "SIMLLM_VLLM_GPU": "H100",
            "SIMLLM_VLLM_MEM_BANDWIDTH": "1e12",
            "SIMLLM_VLLM_EFFICIENCY": "0.5",
            "SIMLLM_VLLM_HOST_INIT_PS": "1234",
            "SIMLLM_VLLM_TOKEN_ID": "99",
            "SIMLLM_VLLM_STEP_RECORDS": "/tmp/steps.jsonl",
        }
    )
    assert config.mode == "paced"
    assert config.kv_memory_bytes == 1000
    assert config.efficiency == 0.5
    assert config.host_initiation_ps == 1234
    assert config.token_id == 99
    gpu = config.gpu_spec()
    assert gpu.name == "h100"
    assert gpu.peak_flops == 989.5e12  # envelope default kept
    assert gpu.mem_bandwidth == 1e12  # override applied


def test_config_rejects_bad_values():
    with pytest.raises(ValueError, match="virtual or paced"):
        SimExecutorConfig.from_env({"SIMLLM_VLLM_MODE": "fast"})
    with pytest.raises(ValueError, match="must be an integer"):
        SimExecutorConfig.from_env({"SIMLLM_VLLM_KV_MEMORY_BYTES": "lots"})
    with pytest.raises(ValueError, match="unknown SIMLLM_VLLM_GPU"):
        SimExecutorConfig.from_env({"SIMLLM_VLLM_GPU": "gtx280"}).gpu_spec()


# Record export

def test_step_records_dump_is_json_round_trippable(tmp_path):
    translator = StepTranslator()
    records = []
    admit = FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewRequest("r0", prompt(16), num_computed_tokens=8)],
        num_scheduled_tokens={"r0": 8},
    )
    records.append(
        translate_scheduler_output(translator, admit, step_index=0, virtual_time_ps=0).record
    )
    decode = FakeSchedulerOutput(
        scheduled_cached_reqs=FakeCachedRequests(["r0"], [16], [1]),
        num_scheduled_tokens={"r0": 1},
        finished_req_ids={"r0"},
    )
    records.append(
        translate_scheduler_output(translator, decode, step_index=1, virtual_time_ps=5_000).record
    )

    path = write_step_records(records, tmp_path / "steps.jsonl")
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == step_records_to_json(records)
    # One schema for the offline dump and the closed-loop step manifest.
    assert {line["schema"] for line in lines} == {"atlahs-closed-loop-step-v1"}
    assert lines[0]["scheduled"][0]["phase"] == "prefill"
    assert lines[0]["scheduled"][0]["num_cached_tokens"] == 8
    assert lines[0]["num_sampled"] == 1
    assert lines[1]["scheduled"][0]["phase"] == "decode"
    assert lines[1]["finished_request_ids"] == ["r0"]
    assert lines[1]["num_sampled"] == 1


# Placement manifest assembly

def worker_entries() -> list[dict]:
    """Four ranks, TP=2 x PP=2, as the worker extension would report them."""
    entries = []
    for rank in range(4):
        tp_ranks = [0, 1] if rank < 2 else [2, 3]
        pp_ranks = [rank, rank + 2] if rank < 2 else [rank - 2, rank]
        entries.append(
            {
                "global_rank": rank,
                "hostname": "node-a",
                "local_rank": rank,
                "gpu_uuid": f"GPU-{rank:08x}",
                "pci_bus_id": f"0000:{rank:02d}:00.0",
                "groups": {
                    "tp": {"rank_in_group": rank % 2, "global_ranks": tp_ranks},
                    "pp": {"rank_in_group": 0 if rank < 2 else 1, "global_ranks": pp_ranks},
                },
                "pipeline_layer_range": [0, 16] if rank < 2 else [16, 32],
                "local_expert_ids": {3: [0, 1, 2, 3]},
                "placement_epoch": 2,
            }
        )
    return entries


def test_manifest_assembly_round_trips_through_save_load(tmp_path):
    manifest = manifest_from_worker_entries(worker_entries(), framework_version="0.26.0")
    assert manifest.source == "extracted"
    assert manifest.framework == "vllm"
    assert manifest.framework_version == "0.26.0"

    path = manifest.save(tmp_path / "placement.json")
    loaded = PlacementManifest.load(path)
    assert [rank.global_rank for rank in loaded.ranks] == [0, 1, 2, 3]
    rank3 = loaded.by_rank(3)
    assert rank3.hostname == "node-a"
    assert rank3.gpu_uuid == "GPU-00000003"
    assert rank3.pci_bus_id == "0000:03:00.0"
    assert loaded.group_ranks(3, "tp") == [2, 3]
    assert loaded.group_ranks(3, "pp") == [1, 3]
    assert rank3.groups["pp"].rank_in_group == 1
    # Layer ranges survive as a tuple and expert-map keys as ints, both of
    # which JSON would otherwise flatten.
    assert rank3.pipeline_layer_range == (16, 32)
    assert rank3.local_expert_ids == {3: [0, 1, 2, 3]}
    assert rank3.placement_epoch == 2


def test_manifest_assembly_sorts_and_tolerates_optional_fields():
    entries = [
        {"global_rank": 2, "hostname": "node-b", "local_rank": 0},
        {"global_rank": 0, "hostname": "node-a", "local_rank": 0},
    ]
    manifest = manifest_from_worker_entries(entries, framework_version="0.26.0")
    assert [rank.global_rank for rank in manifest.ranks] == [0, 2]
    lonely = manifest.by_rank(2)
    assert lonely.groups == {}
    assert lonely.gpu_uuid is None
    assert lonely.pipeline_layer_range is None
    assert lonely.local_expert_ids == {}
    assert lonely.placement_epoch == 0


def test_manifest_assembly_rejects_broken_input():
    with pytest.raises(ValueError, match="no placement entries"):
        manifest_from_worker_entries([])
    with pytest.raises(ValueError, match="missing 'hostname'"):
        manifest_from_worker_entries([{"global_rank": 0}])
    duplicated = [
        {"global_rank": 1, "hostname": "node-a", "local_rank": 1},
        {"global_rank": 1, "hostname": "node-a", "local_rank": 1},
    ]
    with pytest.raises(ValueError, match="duplicate global_rank"):
        manifest_from_worker_entries(duplicated)


# Worker-side discovery helpers (fed fakes, never a real worker)

def test_owned_expert_ids_from_expert_map():
    from simllm.adapters.vllm.worker_ext import owned_expert_ids

    # EP=4 over 8 experts, linear placement, this rank owns 6 and 7.
    expert_map = [-1, -1, -1, -1, -1, -1, 0, 1]
    assert owned_expert_ids(expert_map, 2, 8) == [6, 7]
    # EP disabled: the map is None and the rank owns every expert.
    assert owned_expert_ids(None, 8, 8) == list(range(8))
    # Nothing discoverable: no guess.
    assert owned_expert_ids(None, None, None) == []


def test_local_expert_discovery_keys_on_the_layer_index():
    from simllm.adapters.vllm.worker_ext import discover_local_experts

    class FakeManager:
        expert_map: ClassVar[list[int]] = [-1, 0, -1, 1]
        local_num_experts = 2
        global_num_experts = 4

    class FakeMoE:
        expert_map_manager = FakeManager()

    class FakeModel:
        def named_modules(self):
            return [
                ("model.layers.0.self_attn", object()),
                ("model.layers.5.mlp.experts", FakeMoE()),
                ("lm_head", object()),
            ]

    assert discover_local_experts(FakeModel()) == {5: [1, 3]}


def test_layer_range_discovery_prefers_the_models_own_attributes():
    from simllm.adapters.vllm.worker_ext import discover_layer_range

    class Inner:
        start_layer = 16
        end_layer = 32

    class Outer:
        model = Inner()

    assert discover_layer_range(Outer()) == (16, 32)
    assert discover_layer_range(object()) is None


def test_placement_entry_never_raises_on_a_bare_worker():
    from simllm.adapters.vllm import placement_entry

    class BareWorker:
        rank = 5
        local_rank = 1

    entry = placement_entry(BareWorker())
    assert entry["global_rank"] == 5
    assert entry["local_rank"] == 1
    assert entry["groups"] == {}
    assert entry["local_expert_ids"] == {}
    assert entry["placement_epoch"] == 0
    assert isinstance(entry["hostname"], str) and entry["hostname"]
    # The entry is exactly what the manifest assembler consumes.
    manifest = manifest_from_worker_entries([entry], framework_version="0.26.0")
    assert manifest.by_rank(5).local_rank == 1
