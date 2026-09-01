"""vLLM adapter tests using only transcribed inputs.

The tests never import vLLM directly. Adapter imports select the real worker
when vLLM is installed and the transcribed worker base otherwise, so the same
mirror tests run in both environments.
"""

import importlib.util
import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from simllm.adapters.vllm import (
    GRANITE_ALL2ALL_BACKEND,
    NATIVE_STEP_CAPTURE_SCHEMA,
    OBSERVED_SCHEDULE_GRANITE_DBO,
    SKELETON_EMPTY_STEP_CALL_SEQUENCE,
    SKELETON_INIT_CALL_SEQUENCE,
    SKELETON_STEP_CALL_SEQUENCE,
    ModelDims,
    ObservationStepSink,
    PlacementExporter,
    ReplayTokenSource,
    SimExecutor,
    SimExecutorConfig,
    SimModelRunner,
    SimWorker,
    StepTranslator,
    VllmBatchSlice,
    build_granite_execution_observations,
    capture_vllm_native_step,
    configure,
    fabricate_sampled_tokens,
    manifest_from_worker_entries,
    observations_from_vllm_step,
    reset_configuration,
    sample_adapter_tokens,
    step_kernel,
    step_records_to_json,
    translate_scheduler_output,
    vllm_batch_slices,
    write_step_records,
)
from simllm.adapters.vllm.worker import _skeleton_fallback_latency
from simllm.compute import GpuSpec, HostInitiationModel, RooflineProvider
from simllm.core import (
    CollectiveWork,
    ComputeWork,
    ExecutionObservations,
    RequestBookkeeper,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    StepResult,
    VirtualClock,
)
from simllm.placement import PlacementManifest
from simllm.preplay import (
    ForwardPhase,
    RequestArrival,
    join_preplay_arrivals,
    read_preplay_trace,
    write_preplay_replay_run,
    write_preplay_trace,
)

VLLM_INSTALLED = importlib.util.find_spec("vllm") is not None

# SchedulerOutput stubs: same attribute names as vLLM v0.26.0, nothing else


@dataclass
class FakeNewRequest:
    """Shaped like ``vllm.v1.core.sched.output.NewRequestData``."""

    req_id: str
    prompt_token_ids: list[int]
    num_computed_tokens: int = 0
    sampling_params: object | None = None


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


class CapturingObservationSink:
    def __init__(self, latency_ps=7_000):
        self.latency_ps = latency_ps
        self.clock = None
        self.calls = []

    def bind_clock(self, clock):
        if self.clock is not None and self.clock is not clock:
            raise RuntimeError("clock changed")
        self.clock = clock

    def __call__(self, record, observations):
        self.calls.append((record, observations))
        return StepResult(
            step_index=record.step_index,
            step_latency_ps=self.latency_ps,
            completed_at_ps=record.virtual_time_ps + self.latency_ps,
        )


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
            enable_fault_tolerance=False,
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


class FakeGraniteModelConfig(FakeModelConfig):
    hf_text_config = SimpleNamespace(
        model_type="granitemoe",
        architectures=["GraniteMoeForCausalLM"],
        intermediate_size=512,
        num_local_experts=32,
        num_experts_per_tok=8,
    )

    @staticmethod
    def get_hidden_size():
        return 1024

    @staticmethod
    def get_num_layers(parallel_config):
        return 24

    @staticmethod
    def get_num_attention_heads(parallel_config):
        return 16

    @staticmethod
    def get_num_kv_heads(parallel_config):
        return 8

    @staticmethod
    def get_head_size():
        return 64

    @staticmethod
    def get_vocab_size():
        return 49_155

    @staticmethod
    def get_total_num_hidden_layers():
        return 24


def fake_granite_vllm_config():
    config = fake_vllm_config()
    config.model_config = FakeGraniteModelConfig()
    config.parallel_config.data_parallel_size = 8
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.enable_expert_parallel = True
    config.parallel_config.all2all_backend = GRANITE_ALL2ALL_BACKEND
    config.parallel_config.enable_dbo = True
    config.parallel_config.ubatch_size = 0
    config.parallel_config.dbo_decode_token_threshold = 2
    config.parallel_config.dbo_prefill_token_threshold = 512
    return config


def make_sim_worker(
    clock=None,
    *,
    vllm_config=None,
    rank=0,
    is_driver_worker=True,
    simllm_config=None,
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
        _simllm_config=simllm_config,
    )


def test_skeleton_worker_nonideal_host_cost_requires_timing_sink(monkeypatch):
    monkeypatch.setenv("SIMLLM_VLLM_WORKER_MODE", "skeleton")
    model = HostInitiationModel.turing_cuda_graph(440)
    gpu = GpuSpec("gtx1660-ti-sm75", peak_flops=1.0, mem_bandwidth=1.0)
    reset_configuration()
    try:
        configure(host_model=model, gpu=gpu)
        with pytest.raises(RuntimeError, match="requires a host-model-aware"):
            make_sim_worker()
    finally:
        reset_configuration()

    assert _skeleton_fallback_latency(HostInitiationModel.ideal()) == 0
    with pytest.raises(RuntimeError, match="requires a timing sink result"):
        _skeleton_fallback_latency(model)


def joined_replay_path(tmp_path: Path, *, trace_path: Path | None = None) -> Path:
    if trace_path is None:
        trace_path = (
            Path(__file__).parents[1]
            / "examples/preplay_trace_v1/writer_golden.jsonl"
        )
    run = join_preplay_arrivals(
        (RequestArrival(request_id="request-golden", arrived_at_ps=0),),
        trace_path,
        RequestBookkeeper(),
    )
    return write_preplay_replay_run(run, tmp_path / "joined-replay.json")


def joined_two_token_replay_path(tmp_path: Path) -> Path:
    source_path = (
        Path(__file__).parents[1]
        / "examples/preplay_trace_v1/writer_golden.jsonl"
    )
    source = read_preplay_trace(source_path)
    request = source.requests[0]
    decode = replace(
        request.prefill_tokens[0],
        phase=ForwardPhase.DECODE,
        token_index=0,
        token_id=20,
    )
    request = replace(
        request,
        max_new_tokens=2,
        output_token_ids=(20, 21),
        decode_tokens=(decode,),
    )
    trace_path = write_preplay_trace(
        tmp_path / "two-token-trace.jsonl",
        source.provenance,
        (request,),
    )
    return joined_replay_path(tmp_path, trace_path=trace_path)


def replay_sampling_params(output_length: int = 1):
    return SimpleNamespace(
        max_tokens=output_length,
        min_tokens=0,
        eos_token_id=None,
        stop_token_ids=[],
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
    assert package.PINNED_VLLM_VERSION == "0.27.1"


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
    assert "vLLM v0.27.1" in message
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

    fault_tolerant_config = fake_vllm_config()
    fault_tolerant_config.parallel_config.enable_fault_tolerance = True
    with pytest.raises(RuntimeError, match="fault tolerance"):
        make_sim_worker(vllm_config=fault_tolerant_config)

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


def test_sampled_identities_and_native_output_capture_are_explicit(monkeypatch, tmp_path):
    native_path = tmp_path / "native_steps.jsonl"
    record_path = tmp_path / "steps.jsonl"
    reset_configuration()
    monkeypatch.setenv("SIMLLM_VLLM_WORKER_MODE", "skeleton")
    try:
        worker = make_sim_worker(
            VirtualClock(start_ps=123_000),
            simllm_config=SimExecutorConfig(
                token_id=512,
                step_records_path=str(record_path),
                emit_sampled_request_ids=True,
                native_step_capture_path=str(native_path),
            ),
        )
        worker.init_device()
        output = FakeSchedulerOutput(
            scheduled_new_reqs=[
                FakeNewRequest("chunked", prompt(8)),
                FakeNewRequest("sampled", prompt(2)),
            ],
            num_scheduled_tokens={"chunked": 4, "sampled": 2},
            preempted_req_ids={"old"},
            finished_req_ids={"done"},
        )

        assert worker.execute_model(output) is None
        worker.sample_tokens(None)

        record = worker.step_records[0]
        assert record.num_sampled == 1
        assert record.sampled_request_ids == ["sampled"]
        capture = json.loads(native_path.read_text(encoding="utf-8"))
        assert capture["schema"] == NATIVE_STEP_CAPTURE_SCHEMA
        assert capture["native_scheduler_output"] == {
            "ordered_scheduled_request_ids": ["chunked", "sampled"],
            "num_scheduled_tokens": [
                {"request_id": "chunked", "num_scheduled_tokens": 4},
                {"request_id": "sampled", "num_scheduled_tokens": 2},
            ],
            "total_num_scheduled_tokens": 6,
            "preempted_request_ids": ["old"],
            "finished_request_ids": ["done"],
        }
        assert capture["step_record"] == step_records_to_json([record])[0]
        assert capture["step_record"]["sampled_request_ids"] == ["sampled"]
    finally:
        reset_configuration()


def test_native_capture_does_not_reconstruct_scheduler_fields_from_projection():
    native = FakeSchedulerOutput(num_scheduled_tokens={"request": 4})
    projection = StepRecord(
        step_index=7,
        virtual_time_ps=123,
        scheduled=[
            ScheduledRequest(
                request_id="request",
                phase=RequestPhase.PREFILL,
                num_new_tokens=99,
            )
        ],
        num_sampled=0,
        sampled_request_ids=[],
    )

    capture = capture_vllm_native_step(native, projection)

    assert capture.scheduled[0].num_scheduled_tokens == 4
    assert capture.step_record.scheduled[0].num_new_tokens == 99
    native.num_scheduled_tokens = {7: 4}
    with pytest.raises(ValueError, match="scheduled request ID"):
        capture_vllm_native_step(native, projection)


def test_no_replay_worker_stream_is_byte_locked(monkeypatch, tmp_path):
    stream_path = tmp_path / "steps.jsonl"
    expected_path = (
        Path(__file__).parent
        / "fixtures/vllm/no_replay_r1_p4_steps.jsonl"
    )
    reset_configuration()
    monkeypatch.setenv("SIMLLM_VLLM_WORKER_MODE", "skeleton")
    try:
        worker = make_sim_worker(
            VirtualClock(start_ps=123_000),
            simllm_config=SimExecutorConfig(
                mode="virtual",
                token_id=512,
                step_records_path=str(stream_path),
                replay_run_path=None,
            ),
        )
        worker.init_device()
        prefill = FakeSchedulerOutput(
            scheduled_new_reqs=[FakeNewRequest("r0", prompt(4))],
            num_scheduled_tokens={"r0": 4},
        )
        assert worker.execute_model(prefill) is None
        assert worker.sample_tokens(None).sampled_token_ids == [[512]]
        decode = FakeSchedulerOutput(
            scheduled_cached_reqs=FakeCachedRequests(["r0"], [4], [1]),
            num_scheduled_tokens={"r0": 1},
        )
        assert worker.execute_model(decode) is None
        assert worker.sample_tokens(None).sampled_token_ids == [[512]]

        assert stream_path.read_bytes() == expected_path.read_bytes()
        assert all(record.sampled_request_ids is None for record in worker.step_records)
    finally:
        reset_configuration()
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


def test_granite_worker_emits_source_ordered_single_and_dbo_schedules(monkeypatch):
    monkeypatch.setenv("SIMLLM_VLLM_WORKER_MODE", "skeleton")
    reset_configuration()
    sink = CapturingObservationSink()
    try:
        configure(step_sink=sink)
        worker = make_sim_worker(
            VirtualClock(),
            vllm_config=fake_granite_vllm_config(),
            simllm_config=SimExecutorConfig(
                observed_schedule=OBSERVED_SCHEDULE_GRANITE_DBO
            ),
        )
        worker.init_device()
        prefill = FakeSchedulerOutput(
            scheduled_new_reqs=[
                FakeNewRequest("r0", prompt(1)),
                FakeNewRequest("r1", prompt(1)),
                FakeNewRequest("r2", prompt(1)),
            ],
            num_scheduled_tokens={"r0": 1, "r1": 1, "r2": 1},
        )
        assert worker.execute_model(prefill) is None
        prefill_observations = sink.calls[-1][1]
        assert isinstance(prefill_observations, ExecutionObservations)
        prefill_collectives = [
            operation
            for operation in prefill_observations.operations
            if isinstance(operation.work, CollectiveWork)
        ]
        assert len(prefill_collectives) == 48
        assert {operation.correlation.microbatch for operation in prefill_collectives} == {
            None
        }
        assert worker.sample_tokens(None).req_ids == ["r0", "r1", "r2"]

        decode = FakeSchedulerOutput(
            scheduled_cached_reqs=FakeCachedRequests(
                ["r0", "r1", "r2"],
                [1, 1, 1],
                [1, 1, 1],
            ),
            num_scheduled_tokens={"r0": 1, "r1": 1, "r2": 1},
        )
        assert worker.execute_model(decode) is None
        observations = sink.calls[-1][1]
        assert isinstance(observations, ExecutionObservations)
        collectives = [
            operation
            for operation in observations.operations
            if isinstance(operation.work, CollectiveWork)
        ]
        assert len(collectives) == 96
        assert {
            (operation.correlation.layer, operation.work.channel_hint)
            for operation in collectives
        } == {
            (layer, site)
            for layer in range(24)
            for site in ("dispatch", "combine")
        }
        assert all(operation.work.payload_bytes == 0 for operation in collectives)
        assert {
            operation.correlation.microbatch for operation in collectives
        } == {0, 1}
        requests_by_microbatch = {
            microbatch: {
                operation.correlation.request_ids
                for operation in collectives
                if operation.correlation.microbatch == microbatch
            }
            for microbatch in (0, 1)
        }
        assert requests_by_microbatch == {0: {("r0",)}, 1: {("r1", "r2")}}

        comm_layer_zero = [
            (operation.correlation.microbatch, operation.work.channel_hint)
            for operation in collectives
            if operation.correlation.layer == 0
        ]
        assert comm_layer_zero == [
            (0, "dispatch"),
            (1, "dispatch"),
            (0, "combine"),
            (1, "combine"),
        ]
        expert_one = next(
            operation
            for operation in observations.operations
            if operation.operation_id
            == "step-1:ubatch-1:layer-0:rank-0:experts"
        )
        combine_zero = "step-1:ubatch-0:layer-0:ep-combine"
        assert combine_zero not in expert_one.depends_on
        assert combine_zero not in expert_one.participant_local_depends_on
        next_pre_zero = next(
            operation
            for operation in observations.operations
            if operation.operation_id
            == "step-1:ubatch-0:layer-1:rank-0:pre-dispatch"
        )
        assert next_pre_zero.participant_local_depends_on == (combine_zero,)

        rank_zero_logits = next(
            operation
            for operation in observations.operations
            if operation.operation_id == "step-1:rank-0:logits"
        )
        assert rank_zero_logits.depends_on == ()
        assert rank_zero_logits.participant_local_depends_on == (
            "step-1:ubatch-0:layer-23:ep-combine",
            "step-1:ubatch-1:layer-23:ep-combine",
        )

        assert observations.completion_operation_ids == (
            "step-1:ubatch-0:requests-visible",
            "step-1:ubatch-1:requests-visible",
        )
        completion_by_id = {
            operation.operation_id: operation
            for operation in observations.operations
            if operation.operation_id in observations.completion_operation_ids
        }
        assert completion_by_id[
            "step-1:ubatch-0:requests-visible"
        ].correlation.request_ids == ("r0",)
        assert completion_by_id[
            "step-1:ubatch-1:requests-visible"
        ].correlation.request_ids == ("r1", "r2")

        runner = worker.model_runner
        assert isinstance(runner, SimModelRunner)
        assert runner.latest_observations is observations
        assert runner.latest_schedule_timing is not None
        rank_zero_compute_ps = sum(
            operation.work.nominal_duration_ps or 0
            for operation in observations.operations
            if operation.rank == 0 and isinstance(operation.work, ComputeWork)
        )
        assert rank_zero_compute_ps == (
            runner.latest_schedule_timing.represented_compute_ps
        )
        decode_record = sink.calls[-1][0]
        decode_kernel = step_kernel(
            worker.dims,
            decode_record,
            decode_record.num_sampled or 0,
        )
        rank_zero_compute = [
            operation.work
            for operation in observations.operations
            if operation.rank == 0 and isinstance(operation.work, ComputeWork)
        ]
        assert sum(work.flops for work in rank_zero_compute) == int(
            decode_kernel.flops
        )
        assert sum(work.hbm_bytes for work in rank_zero_compute) == int(
            decode_kernel.bytes_moved
        )
    finally:
        reset_configuration()


def test_granite_schedule_rejects_unaudited_configuration_and_splits():
    config = fake_granite_vllm_config()
    dims = ModelDims(
        24,
        1024,
        512,
        16,
        8,
        64,
        49_155,
        2,
        num_experts=32,
        top_k=8,
        moe_intermediate_size=512,
        local_num_experts=4,
    )
    decode = StepRecord(
        0,
        0,
        [
            ScheduledRequest("r0", RequestPhase.DECODE, 1, context_length=8),
            ScheduledRequest("r1", RequestPhase.DECODE, 1, context_length=8),
        ],
        num_sampled=2,
    )
    provider = RooflineProvider(efficiency=0.7)
    gpu = GpuSpec("b100", 1.8e15, 8.0e12)
    host = HostInitiationModel()

    bad_backend = fake_granite_vllm_config()
    bad_backend.parallel_config.all2all_backend = "allgather_reducescatter"
    with pytest.raises(RuntimeError, match="all2all_backend"):
        observations_from_vllm_step(
            decode, bad_backend, dims, tuple(range(8)), provider, gpu, host
        )

    bad_model = fake_granite_vllm_config()
    bad_model.model_config = FakeModelConfig()
    with pytest.raises(RuntimeError, match="GraniteMoeForCausalLM"):
        observations_from_vllm_step(
            decode, bad_model, dims, tuple(range(8)), provider, gpu, host
        )

    bad_tp = fake_granite_vllm_config()
    bad_tp.parallel_config.tensor_parallel_size = 2
    with pytest.raises(RuntimeError, match="tensor_parallel_size=1"):
        observations_from_vllm_step(
            decode, bad_tp, dims, tuple(range(8)), provider, gpu, host
        )

    multi_token = StepRecord(
        1,
        0,
        [ScheduledRequest("r0", RequestPhase.PREFILL, 512, context_length=512)],
    )
    with pytest.raises(RuntimeError, match="multi-token requests"):
        vllm_batch_slices(multi_token, config.parallel_config)

    padded = replace(decode, num_tokens_after_padding=4)
    with pytest.raises(RuntimeError, match="padding changes"):
        vllm_batch_slices(padded, config.parallel_config)

    with pytest.raises(ValueError, match="partition scheduled requests"):
        build_granite_execution_observations(
            decode,
            dims,
            tuple(range(8)),
            (
                VllmBatchSlice(0, ("r1",), 1),
                VllmBatchSlice(1, ("r0",), 1),
            ),
            provider,
            gpu,
            host,
        )


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
    translator = StepTranslator(emit_sampled_request_ids=True)
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
    assert record.sampled_request_ids == ["short", "decoding"]
    # Finished and preempted ids are sorted for reproducible traces.
    assert record.finished_request_ids == ["also-gone", "gone"]
    assert record.preempted_request_ids == ["evicted"]
    # The two prefills plus the decoding request are still tracked; finished
    # ids never entered the table.
    assert len(translator) == 3


def test_exported_step_translator_defaults_sampled_identities_off():
    output = FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewRequest("sampled", prompt(2))],
        num_scheduled_tokens={"sampled": 2},
    )

    default_step = translate_scheduler_output(
        StepTranslator(), output, step_index=0, virtual_time_ps=0
    )
    opted_in_step = translate_scheduler_output(
        StepTranslator(emit_sampled_request_ids=True),
        output,
        step_index=0,
        virtual_time_ps=0,
    )

    assert default_step.record.num_sampled == 1
    assert default_step.record.sampled_request_ids is None
    assert opted_in_step.record.sampled_request_ids == ["sampled"]


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


# Pre-play replay serving

def test_replay_source_serves_exact_tokens_through_executor_path(tmp_path):
    source = ReplayTokenSource.from_path(
        joined_replay_path(tmp_path), max_model_len=4096
    )
    executor = object.__new__(SimExecutor)
    executor.replay = source
    executor.token_id = 512
    scheduler_output = FakeSchedulerOutput(
        scheduled_new_reqs=[
            FakeNewRequest(
                "request-golden",
                [10],
                sampling_params=replay_sampling_params(),
            )
        ],
        num_scheduled_tokens={"request-golden": 1},
    )
    translated = SimpleNamespace(
        req_ids=["request-golden"],
        produces_token=[True],
    )

    assert executor._sample_output_fields(translated, scheduler_output) == (
        ["request-golden"],
        {"request-golden": 0},
        [[20]],
    )
    assert source.snapshot().served_token_ids == (("request-golden", (20,)),)
    assert source.snapshot().completed_request_ids == ("request-golden",)


def test_replay_source_drains_through_skeleton_worker(monkeypatch, tmp_path):
    replay_path = joined_replay_path(tmp_path)
    monkeypatch.setenv("SIMLLM_VLLM_WORKER_MODE", "skeleton")
    monkeypatch.setenv("SIMLLM_VLLM_REPLAY_RUN", str(replay_path))
    worker = make_sim_worker(VirtualClock(start_ps=123_000))
    worker.init_device()
    prefill = FakeSchedulerOutput(
        scheduled_new_reqs=[
            FakeNewRequest(
                "request-golden",
                [10],
                sampling_params=replay_sampling_params(),
            )
        ],
        num_scheduled_tokens={"request-golden": 1},
    )
    assert worker.execute_model(prefill) is None
    output = worker.sample_tokens(None)
    assert output.req_ids == ["request-golden"]
    assert output.sampled_token_ids == [[20]]

    drain = FakeSchedulerOutput(finished_req_ids={"request-golden"})
    drained = worker.execute_model(drain)
    assert drained.req_ids == []
    assert drained.sampled_token_ids in (None, [])
    assert worker.step_records[-1].finished_request_ids == ["request-golden"]
    assert worker.step_results[-1].step_latency_ps == 0
    assert worker.replay is not None
    snapshot = worker.replay.snapshot()
    assert snapshot.served_token_ids == (("request-golden", (20,)),)
    assert snapshot.completed_request_ids == ("request-golden",)
    assert snapshot.drained_request_ids == ("request-golden",)


def test_replay_rejects_suffix_shaped_unjoined_identity_without_mutation(tmp_path):
    source = ReplayTokenSource.from_path(
        joined_replay_path(tmp_path), max_model_len=4096
    )
    runtime_id = "request-golden-deadbeef"
    scheduler_output = FakeSchedulerOutput(
        scheduled_new_reqs=[
            FakeNewRequest(
                runtime_id,
                [10],
                sampling_params=replay_sampling_params(),
            )
        ],
        num_scheduled_tokens={runtime_id: 1},
    )

    with pytest.raises(RuntimeError, match="missing from the joined replay run"):
        source.sample([runtime_id], [True], scheduler_output)
    snapshot = source.snapshot()
    assert snapshot.served_token_ids == (("request-golden", ()),)
    assert snapshot.completed_request_ids == ()
    assert snapshot.drained_request_ids == ()
    with pytest.raises(RuntimeError, match="missing from the joined replay run"):
        source.request("request-golden-not-hex")
    with pytest.raises(RuntimeError, match="missing from the joined replay run"):
        source.request("unknown-deadbeef")


def test_replay_rejects_unpinned_unknown_and_exhausted_requests(tmp_path):
    replay_path = joined_replay_path(tmp_path)

    source = ReplayTokenSource.from_path(replay_path, max_model_len=4096)
    bad_limit = FakeSchedulerOutput(
        scheduled_new_reqs=[
            FakeNewRequest(
                "request-golden",
                [10],
                sampling_params=replay_sampling_params(output_length=2),
            )
        ],
        num_scheduled_tokens={"request-golden": 1},
    )
    with pytest.raises(RuntimeError, match="max_tokens=1"):
        source.sample(["request-golden"], [True], bad_limit)
    assert source.snapshot().served_token_ids == (("request-golden", ()),)

    unknown = FakeSchedulerOutput(
        scheduled_new_reqs=[
            FakeNewRequest(
                "unknown",
                [10],
                sampling_params=replay_sampling_params(),
            )
        ],
        num_scheduled_tokens={"unknown": 1},
    )
    with pytest.raises(RuntimeError, match="missing from the joined replay run"):
        source.sample(["unknown"], [True], unknown)

    valid = FakeSchedulerOutput(
        scheduled_new_reqs=[
            FakeNewRequest(
                "request-golden",
                [10],
                sampling_params=replay_sampling_params(),
            )
        ],
        num_scheduled_tokens={"request-golden": 1},
    )
    assert source.sample(["request-golden"], [True], valid)[2] == [[20]]
    exhausted = FakeSchedulerOutput(
        scheduled_cached_reqs=FakeCachedRequests(
            ["request-golden"], [1], [1]
        ),
        num_scheduled_tokens={"request-golden": 1},
    )
    with pytest.raises(RuntimeError, match="exhausted its oracle"):
        source.sample(["request-golden"], [True], exhausted)


def test_replay_batch_validation_is_atomic_on_a_late_index_error(tmp_path):
    source = ReplayTokenSource.from_path(
        joined_replay_path(tmp_path), max_model_len=4096
    )
    invalid_batch = FakeSchedulerOutput(
        scheduled_cached_reqs=FakeCachedRequests(
            ["request-golden", "unknown"],
            [1, 1],
            [0, 0],
        ),
        num_scheduled_tokens={"request-golden": 1, "unknown": 1},
    )

    with pytest.raises(RuntimeError, match="missing from the joined replay run"):
        source.sample(
            ["request-golden", "unknown"],
            [True, True],
            invalid_batch,
        )
    assert source.snapshot().served_token_ids == (("request-golden", ()),)

    index_gap = FakeSchedulerOutput(
        scheduled_cached_reqs=FakeCachedRequests(
            ["request-golden"],
            [2],
            [1],
        ),
        num_scheduled_tokens={"request-golden": 1},
    )
    with pytest.raises(RuntimeError, match="reported output index 1, expected 0"):
        source.sample(["request-golden"], [True], index_gap)
    assert source.snapshot().served_token_ids == (("request-golden", ()),)


def test_replay_rejects_early_eos_before_settlement(monkeypatch, tmp_path):
    replay_path = joined_two_token_replay_path(tmp_path)
    stream_path = tmp_path / "rejected-steps.jsonl"
    params = replay_sampling_params(output_length=2)
    params.eos_token_id = 20
    monkeypatch.setenv("SIMLLM_VLLM_WORKER_MODE", "skeleton")
    worker = make_sim_worker(
        VirtualClock(start_ps=123_000),
        simllm_config=SimExecutorConfig(
            step_records_path=str(stream_path),
            replay_run_path=str(replay_path),
        ),
    )
    worker.init_device()
    scheduler_output = FakeSchedulerOutput(
        scheduled_new_reqs=[
            FakeNewRequest("request-golden", [10], sampling_params=params)
        ],
        num_scheduled_tokens={"request-golden": 1},
    )

    with pytest.raises(RuntimeError, match="hits eos_token_id before"):
        worker.execute_model(scheduler_output)
    assert worker.step_records == []
    assert worker.step_results == []
    assert worker.clock.now_ps == 123_000
    assert not stream_path.exists()
    assert worker.replay is not None
    assert worker.replay.snapshot().served_token_ids == (("request-golden", ()),)


def test_replay_allows_eos_at_the_oracle_final_position(tmp_path):
    source = ReplayTokenSource.from_path(
        joined_two_token_replay_path(tmp_path), max_model_len=3
    )
    params = replay_sampling_params(output_length=2)
    params.eos_token_id = 21
    admission = FakeSchedulerOutput(
        scheduled_new_reqs=[
            FakeNewRequest("request-golden", [10], sampling_params=params)
        ],
        num_scheduled_tokens={"request-golden": 1},
    )
    assert source.sample(["request-golden"], [True], admission)[2] == [[20]]
    final = FakeSchedulerOutput(
        scheduled_cached_reqs=FakeCachedRequests(
            ["request-golden"], [2], [1]
        ),
        num_scheduled_tokens={"request-golden": 1},
    )
    assert source.sample(["request-golden"], [True], final)[2] == [[21]]
    assert source.snapshot().completed_request_ids == ("request-golden",)


def test_replay_rejects_early_stop_token_and_model_length_overflow(tmp_path):
    replay_path = joined_two_token_replay_path(tmp_path)
    source = ReplayTokenSource.from_path(replay_path, max_model_len=3)
    params = replay_sampling_params(output_length=2)
    params.stop_token_ids = [20]
    early_stop = FakeSchedulerOutput(
        scheduled_new_reqs=[
            FakeNewRequest("request-golden", [10], sampling_params=params)
        ],
        num_scheduled_tokens={"request-golden": 1},
    )
    with pytest.raises(RuntimeError, match="hits a stop token before"):
        source.validate_step(["request-golden"], [True], early_stop)

    source = ReplayTokenSource.from_path(replay_path, max_model_len=3)
    too_long = FakeSchedulerOutput(
        scheduled_new_reqs=[
            FakeNewRequest(
                "request-golden",
                [10, 11],
                sampling_params=replay_sampling_params(output_length=2),
            )
        ],
        num_scheduled_tokens={"request-golden": 2},
    )
    with pytest.raises(RuntimeError, match="beyond max_model_len=3"):
        source.validate_step(["request-golden"], [False], too_long)

    with pytest.raises(ValueError, match="beyond max_model_len=2"):
        ReplayTokenSource.from_path(replay_path, max_model_len=2)


def test_executor_late_model_length_shrink_revalidates_replay(tmp_path):
    source = ReplayTokenSource.from_path(
        joined_replay_path(tmp_path), max_model_len=4096
    )
    executor = object.__new__(SimExecutor)
    executor.replay = source

    executor._rpc_update_max_model_len(0, 2)
    assert source.max_model_len == 2
    with pytest.raises(ValueError, match="beyond max_model_len=1"):
        executor._rpc_update_max_model_len(0, 1)
    assert source.max_model_len == 2


def test_replay_rejects_changed_trace_bytes(tmp_path):
    source_trace = (
        Path(__file__).parents[1]
        / "examples/preplay_trace_v1/writer_golden.jsonl"
    )
    copied_trace = tmp_path / "trace.jsonl"
    copied_trace.write_bytes(source_trace.read_bytes())
    replay_path = joined_replay_path(tmp_path, trace_path=copied_trace)
    copied_trace.write_bytes(copied_trace.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ReplayTokenSource.from_path(replay_path, max_model_len=4096)


def test_absent_replay_calls_the_exact_fabricated_token_path():
    expected = fabricate_sampled_tokens(
        ["a", "b", "c"], [True, False, True], token_id=1234
    )
    actual = sample_adapter_tokens(
        None,
        ["a", "b", "c"],
        [True, False, True],
        1234,
        FakeSchedulerOutput(),
        fabricate=fabricate_sampled_tokens,
    )
    assert actual == expected


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
            "SIMLLM_VLLM_SAMPLED_REQUEST_IDS": "true",
            "SIMLLM_VLLM_NATIVE_STEPS": "/tmp/native.jsonl",
            "SIMLLM_VLLM_REPLAY_RUN": "/tmp/replay.json",
        }
    )
    assert config.mode == "paced"
    assert config.kv_memory_bytes == 1000
    assert config.efficiency == 0.5
    assert config.host_initiation_ps == 1234
    assert config.token_id == 99
    assert config.emit_sampled_request_ids is True
    assert config.native_step_capture_path == "/tmp/native.jsonl"
    assert config.replay_run_path == "/tmp/replay.json"
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
    with pytest.raises(ValueError, match="SIMLLM_VLLM_OBSERVED_SCHEDULE"):
        SimExecutorConfig.from_env(
            {"SIMLLM_VLLM_OBSERVED_SCHEDULE": "plausible-overlap"}
        )
    with pytest.raises(ValueError, match="must be a boolean"):
        SimExecutorConfig.from_env(
            {"SIMLLM_VLLM_SAMPLED_REQUEST_IDS": "sometimes"}
        )


def _granite_config_with_moe_field(name, value):
    config = fake_granite_vllm_config()
    config.model_config = FakeGraniteModelConfig()
    config.model_config.hf_text_config = SimpleNamespace(
        **{**vars(FakeGraniteModelConfig.hf_text_config), name: value}
    )
    return config


def test_shared_expert_and_mixed_schedule_moe_configs_are_refused():
    """Geometries whose reduction inventory ModelDims cannot carry fail closed.

    A shared expert is all-reduced over the tensor-parallel group even when the
    combine kernel already reduced the routed output (pinned vLLM 0.26.0,
    model_executor/layers/fused_moe/runner/moe_runner.py:416-433), and a dense
    prefix leaves some layers with two allreduce sites and no all-to-all, so
    both are refused rather than priced as fully routed (VLLM-25).
    """
    from simllm.adapters.vllm.executor import (
        UNSUPPORTED_LAYER_LIST_MOE_FIELDS,
        UNSUPPORTED_POSITIVE_MOE_FIELDS,
        UNSUPPORTED_STRIDE_MOE_FIELDS,
        model_dims_from_vllm_config,
    )

    # every field the SGLang reader refuses is refused here too
    assert set(UNSUPPORTED_POSITIVE_MOE_FIELDS) >= {
        "n_shared_experts",
        "num_shared_experts",
        "shared_expert_intermediate_size",
        "moe_shared_expert_intermediate_size",
        "shared_intermediate_size",
        "first_k_dense_replace",
        "num_dense_layers",
    }
    assert set(UNSUPPORTED_STRIDE_MOE_FIELDS) == {
        "moe_layer_freq",
        "decoder_sparse_step",
    }
    assert set(UNSUPPORTED_LAYER_LIST_MOE_FIELDS) == {"mlp_only_layers"}

    # a positive count or size declares an unsupported mechanism, zero does not
    for name in UNSUPPORTED_POSITIVE_MOE_FIELDS:
        with pytest.raises(NotImplementedError, match="VLLM-25"):
            model_dims_from_vllm_config(_granite_config_with_moe_field(name, 1))
        zeroed = _granite_config_with_moe_field(name, 0)
        assert model_dims_from_vllm_config(zeroed).num_experts == 32

    # a routed stride is fully routed at exactly 1 and mixed at anything else
    for name in UNSUPPORTED_STRIDE_MOE_FIELDS:
        fully_routed = _granite_config_with_moe_field(name, 1)
        assert model_dims_from_vllm_config(fully_routed).num_experts == 32
        for value in (0, 2, 4):
            with pytest.raises(NotImplementedError, match="VLLM-25"):
                model_dims_from_vllm_config(
                    _granite_config_with_moe_field(name, value)
                )

    # a per-layer exception list is mixed exactly when it is non-empty
    for name in UNSUPPORTED_LAYER_LIST_MOE_FIELDS:
        empty = _granite_config_with_moe_field(name, [])
        assert model_dims_from_vllm_config(empty).num_experts == 32
        with pytest.raises(NotImplementedError, match="VLLM-25"):
            model_dims_from_vllm_config(_granite_config_with_moe_field(name, [3]))

    # a non-integer count is a reader defect rather than an unsupported model
    with pytest.raises(TypeError, match="must be an integer"):
        model_dims_from_vllm_config(
            _granite_config_with_moe_field("n_shared_experts", "1")
        )


def test_expert_group_binds_only_for_a_reducing_combine():
    """Naive expert parallelism must not declare an all-to-all group.

    Pinned vLLM 0.26.0 enables all-to-all kernels only with dp, pcp or
    sequence parallelism (fused_moe/config.py:1052-1055), and only some
    backends reduce in the combine (config/parallel.py:186 defaults to
    allgather_reducescatter, whose prepare-finalize returns False at
    fused_moe/prepare_finalize/naive_dp_ep.py:109 and :242).
    """
    from simllm.adapters.vllm.executor import expert_parallel_geometry

    # the live Granite shape: dp 8, deepep high throughput, reducing combine
    granite = expert_parallel_geometry(fake_granite_vllm_config())
    assert granite.use_ep and granite.use_all2all_kernels
    assert granite.combine_is_reducing and granite.renders_expert_combine

    # tp=8, ep=8, dp=1 is naive expert parallelism: no all-to-all at all
    naive = fake_granite_vllm_config()
    naive.parallel_config.data_parallel_size = 1
    naive.parallel_config.tensor_parallel_size = 8
    geometry = expert_parallel_geometry(naive)
    assert geometry.use_ep
    assert not geometry.use_all2all_kernels
    assert not geometry.renders_expert_combine

    # an all-to-all that does not reduce leaves the mlp-site allreduce in place
    unreduced = fake_granite_vllm_config()
    unreduced.parallel_config.all2all_backend = "allgather_reducescatter"
    geometry = expert_parallel_geometry(unreduced)
    assert geometry.use_all2all_kernels
    assert not geometry.combine_is_reducing
    assert not geometry.renders_expert_combine

    # a backend we cannot place is refused rather than assumed either way
    unknown = fake_granite_vllm_config()
    unknown.parallel_config.all2all_backend = "pplx"
    with pytest.raises(NotImplementedError, match="TRAF-40"):
        expert_parallel_geometry(unknown)


def test_non_reducing_all_to_all_binding_fails_closed():
    """An unrendered allgather and reduce-scatter path is refused, not guessed.

    With that backend the framework both moves expert activations and
    all-reduces the fused output, so neither declaring nor omitting the group
    describes it, and binding must raise rather than silently drop traffic.
    """
    from simllm.adapters.vllm.executor import (
        SimExecutor,
        expert_parallel_geometry,
    )

    class _Sink:
        def __init__(self):
            self.bound = None

        def bind_expert_group(self, ep_ranks):
            self.bound = tuple(ep_ranks)

    executor = SimExecutor.__new__(SimExecutor)
    executor.step_sink = _Sink()
    executor.ep_ranks = tuple(range(8))

    config = fake_granite_vllm_config()
    config.parallel_config.all2all_backend = "allgather_reducescatter"
    executor.expert_parallel = expert_parallel_geometry(config)
    with pytest.raises(NotImplementedError, match="allgather and a reduce-scatter"):
        executor._bind_expert_group()
    with pytest.raises(NotImplementedError, match="TRAF-40"):
        executor._bind_expert_group()
    assert executor.step_sink.bound is None

    # naive expert parallelism binds nothing and raises nothing
    naive = fake_granite_vllm_config()
    naive.parallel_config.data_parallel_size = 1
    naive.parallel_config.tensor_parallel_size = 8
    executor.expert_parallel = expert_parallel_geometry(naive)
    executor._bind_expert_group()
    assert executor.step_sink.bound is None

    # the live reducing shape, dp 8 with deepep, binds the group
    executor.expert_parallel = expert_parallel_geometry(fake_granite_vllm_config())
    executor._bind_expert_group()
    assert executor.step_sink.bound == tuple(range(8))


def test_geometry_study_binding_cells_match_the_binding_semantics():
    """The published geometry study's binding cells must not silently diverge.

    That study calls _bind_expert_group directly, so a change to the binding
    precondition can break a doc-linked published result while the suite stays
    green. This executes exactly its binding cells, nothing else in it.
    """
    import importlib.util
    import sys
    from pathlib import Path

    from simllm.adapters.vllm.executor import REDUCING_ALL2ALL_BACKENDS

    study_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "vllm_moe_geometry_v1"
        / "run_study.py"
    )
    spec = importlib.util.spec_from_file_location(
        "vllm_moe_geometry_study", study_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # every cell declares a backend the binding decision can classify
    assert module.CELL_ALL2ALL_BACKEND in REDUCING_ALL2ALL_BACKENDS
    for cell in module.GEOMETRY_INPUTS:
        config = module._vllm_config(module.GEOMETRY_INPUTS[cell])
        assert getattr(config.parallel_config, "all2all_backend", None) is not None

    rows = module._score_binding()
    assert [row["cell"] for row in rows] == ["g-ep-dp8", "g-ep-flag-world1"]
    assert all(row["passed"] for row in rows)
    assert rows[0]["observed"] == [list(range(8))]
    assert rows[1]["observed"] == []


def test_granite_model_dims_include_enabled_ep_geometry():
    from simllm.adapters.vllm.executor import model_dims_from_vllm_config

    dims = model_dims_from_vllm_config(fake_granite_vllm_config())

    assert dims.num_layers == 24
    assert dims.hidden_size == 1024
    assert dims.num_experts == 32
    assert dims.top_k == 8
    assert dims.moe_intermediate_size == 512
    assert dims.local_num_experts == 4
    assert dims.defaulted_fields == ()


def test_reset_and_injected_config_prevent_replay_contamination(monkeypatch, tmp_path):
    replay_path = joined_replay_path(tmp_path)
    monkeypatch.setenv("SIMLLM_VLLM_WORKER_MODE", "skeleton")
    monkeypatch.delenv("SIMLLM_VLLM_REPLAY_RUN", raising=False)
    reset_configuration()
    try:
        stale = SimExecutorConfig(replay_run_path=str(replay_path))
        configure(config=stale)
        injected = SimExecutorConfig(replay_run_path=None)
        explicit_worker = make_sim_worker(simllm_config=injected)
        assert explicit_worker.sim_config is injected
        assert explicit_worker.replay is None

        reset_configuration()
        clean_worker = make_sim_worker()
        assert clean_worker.sim_config.replay_run_path is None
        assert clean_worker.replay is None
    finally:
        reset_configuration()


def test_observation_sink_binds_the_worker_clock_and_receives_absent_schedule(
    monkeypatch,
):
    monkeypatch.setenv("SIMLLM_VLLM_WORKER_MODE", "skeleton")
    reset_configuration()
    sink = CapturingObservationSink()
    try:
        assert isinstance(sink, ObservationStepSink)
        configure(step_sink=sink)
        clock = VirtualClock(start_ps=123_000)
        worker = make_sim_worker(clock)
        worker.init_device()
        step = FakeSchedulerOutput(
            scheduled_new_reqs=[FakeNewRequest("r0", prompt(4))],
            num_scheduled_tokens={"r0": 4},
        )

        assert worker.execute_model(step) is None

        assert sink.clock is clock
        assert len(sink.calls) == 1
        assert sink.calls[0][0] is worker.step_records[0]
        assert sink.calls[0][1] is None
        assert worker.step_results[0].step_latency_ps == 7_000
        assert clock.now_ps == 130_000
    finally:
        reset_configuration()


def test_observation_producer_off_preserves_the_one_argument_legacy_sink(
    monkeypatch,
):
    monkeypatch.setenv("SIMLLM_VLLM_WORKER_MODE", "skeleton")
    reset_configuration()
    calls = []

    def legacy_sink(record):
        calls.append(record)
        return StepResult(
            step_index=record.step_index,
            step_latency_ps=7_000,
            completed_at_ps=record.virtual_time_ps + 7_000,
        )

    try:
        configure(step_sink=legacy_sink)
        clock = VirtualClock(start_ps=123_000)
        worker = make_sim_worker(
            clock,
            simllm_config=SimExecutorConfig(observed_schedule="off"),
        )
        worker.init_device()
        step = FakeSchedulerOutput(
            scheduled_new_reqs=[FakeNewRequest("r0", prompt(4))],
            num_scheduled_tokens={"r0": 4},
        )

        assert worker.execute_model(step) is None

        assert calls == [worker.step_records[0]]
        assert worker.model_runner.latest_observations is None
        assert worker.step_results[0].step_latency_ps == 7_000
        assert clock.now_ps == 130_000
    finally:
        reset_configuration()


def test_framework_observations_cannot_be_silently_sent_to_a_legacy_sink():
    from simllm.adapters.vllm.executor import _SimStepRuntime

    calls = []
    runtime = _SimStepRuntime(
        config=SimExecutorConfig(),
        step_sink=lambda record: calls.append(record),
        fallback_latency=lambda translated: 0,
    )
    translated = runtime.translate(
        FakeSchedulerOutput(
            scheduled_new_reqs=[FakeNewRequest("r0", prompt(4))],
            num_scheduled_tokens={"r0": 4},
        )
    )

    with pytest.raises(TypeError, match="legacy sink"):
        runtime.settle(translated, ExecutionObservations())

    assert calls == []


# Record export

def test_step_records_dump_is_json_round_trippable(tmp_path):
    translator = StepTranslator(emit_sampled_request_ids=True)
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
    assert lines[0]["sampled_request_ids"] == ["r0"]
    assert lines[1]["scheduled"][0]["phase"] == "decode"
    assert lines[1]["finished_request_ids"] == ["r0"]
    assert lines[1]["num_sampled"] == 1
    assert lines[1]["sampled_request_ids"] == ["r0"]


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
