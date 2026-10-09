"""PP queue conformance without importing or executing a GPU framework."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from simllm.adapters.vllm import pipeline
from simllm.adapters.vllm.executor import SimExecutor, SimExecutorConfig, _SimStepRuntime
from simllm.backends.pipeline_step_sink import PipelineRuntimeStepSink
from simllm.backends.step_lowerer import SerialStepLowererConfig
from simllm.compute import (
    ComputeProvider,
    DurationEstimate,
    GpuSpec,
    HostInitiationModel,
    ModelDims,
)
from simllm.core import (
    CoarseDeviceProfile,
    CoarseDeviceRuntime,
    CollectiveWork,
    ComputeWork,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
)


class FixedProvider(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(10_000, "measured")


def dims(layers=4):
    return ModelDims(num_layers=layers, hidden_size=128, intermediate_size=256,
                     num_heads=8, num_kv_heads=2, head_size=16, vocab_size=1024)


def record(index=0, tokens=1, release=0):
    return StepRecord(index, release, (
        ScheduledRequest(f"r{index}", RequestPhase.DECODE, tokens, 32, 32 + tokens),
    ), num_sampled=1)


def scheduler_output(request_id="a", tokens=8, *, finished=(), structured=False):
    return SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id=request_id,
                                           prompt_token_ids=list(range(tokens)),
                                           num_computed_tokens=0)] if tokens else [],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], num_computed_tokens=[]),
        num_scheduled_tokens={request_id: tokens} if tokens else {},
        finished_req_ids=set(finished), preempted_req_ids=set(),
        has_structured_output_requests=structured,
    )


def make_executor(monkeypatch, *, pp=2, tp=1, sink=None, stream=None):
    monkeypatch.setitem(sys.modules, "vllm.v1.outputs", SimpleNamespace(
        ModelRunnerOutput=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setattr(pipeline, "pipeline_layer_ranges",
                        lambda config, width: tuple((2 * i, 2 * i + 2)
                                                    for i in range(width)))
    executor = object.__new__(SimExecutor)
    executor.tp_size, executor.pp_size, executor.world_size = tp, pp, tp * pp
    executor.dims = dims()
    executor.config = SimExecutorConfig(mode="virtual", step_records_path=stream)
    executor.gpu = GpuSpec("declared-test", 1e12, 1e12)
    executor.host_model = HostInitiationModel.ideal()
    executor.compute_provider = FixedProvider()
    executor.replay = None
    executor.token_id = 512
    executor.step_sink = sink
    executor.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
            enforce_eager=True, runner_type="generate"),
        parallel_config=SimpleNamespace(), scheduler_config=SimpleNamespace(),
    )
    executor._runtime = _SimStepRuntime(
        config=executor.config, step_sink=sink, fallback_latency=lambda _: 17_000,
        host_model=executor.host_model, gpu=executor.gpu,
    )
    executor.clock = executor._runtime.clock
    executor.step_records = executor._runtime.step_records
    executor.step_results = executor._runtime.step_results
    executor._record_stream = executor._runtime.record_stream
    executor._pipeline = pipeline.PipelineController(executor) if pp > 1 else None
    executor._pending_output = None
    return executor


def test_enqueue_sample_fifo_and_idempotent_completion(monkeypatch, tmp_path):
    executor = make_executor(monkeypatch, stream=str(tmp_path / "steps.jsonl"))
    execute_a = executor.execute_model(scheduler_output("a"), non_block=True)
    execute_b = executor.execute_model(scheduler_output("b"), non_block=True)
    sample_a = executor.sample_tokens(non_block=True)
    sample_b = executor.sample_tokens(non_block=True)
    assert executor.clock.now_ps == 0 and not executor.step_records
    assert not sample_a.done() and not sample_b.done()
    assert not execute_a.cancel()
    # Out-of-order consumption retires the older work first, exactly once.
    assert sample_b.result().req_ids == ["b"]
    completed = executor.clock.now_ps
    assert sample_a.result().req_ids == ["a"]
    assert execute_a.result() is None and execute_b.result() is None
    assert sample_b.result().sampled_token_ids == [[512]]
    assert executor.clock.now_ps == completed
    assert [row.step_index for row in executor.step_results] == [0, 1]
    assert len((tmp_path / "steps.jsonl").read_text().splitlines()) == 2
    with pytest.raises(RuntimeError, match="no pending"):
        executor.sample_tokens()


def test_drain_does_not_steal_sampling_slot(monkeypatch):
    executor = make_executor(monkeypatch)
    executor.execute_model(scheduler_output("a"), non_block=True)
    sample = executor.sample_tokens(non_block=True)
    drain = executor.execute_model(scheduler_output(tokens=0, finished=("old",)), non_block=True)
    assert drain.result().req_ids == []
    assert sample.result().req_ids == ["a"]
    assert len(executor.step_records) == 2
    assert executor.step_results[-1].step_latency_ps == 0
    assert executor.step_results[-1].completed_at_ps == executor.clock.now_ps
    with pytest.raises(RuntimeError, match="no pending"):
        executor.sample_tokens()


def test_capacity_and_shutdown_preserve_clock(monkeypatch):
    executor = make_executor(monkeypatch)
    future = executor.execute_model(scheduler_output("a"), non_block=True)
    sample = executor.sample_tokens(non_block=True)
    executor.execute_model(scheduler_output("b"), non_block=True)
    with pytest.raises(RuntimeError, match="capacity"):
        executor.execute_model(scheduler_output("c"), non_block=True)
    executor.shutdown()
    assert executor.clock.now_ps == 0 and not executor.step_results
    with pytest.raises(RuntimeError, match="shut down"):
        future.result()
    assert isinstance(sample.exception(), RuntimeError)
    with pytest.raises(RuntimeError, match="closed"):
        executor.execute_model(scheduler_output("d"))


def test_pp1_exact_immediate_bypass(monkeypatch):
    executor = make_executor(monkeypatch, pp=1)
    output = executor.execute_model(scheduler_output("a"), non_block=True)
    assert output.done()
    assert output.result().req_ids == ["a"]
    assert executor.clock.now_ps == 17_000
    assert executor.step_results[0].step_latency_ps == 17_000
    assert executor._pipeline is None


@pytest.mark.parametrize("tp,pp", [(1, 2), (2, 2), (2, 4), (4, 2)])
def test_stage_ownership_and_transfer_conservation(monkeypatch, tp, pp):
    executor = make_executor(monkeypatch, tp=tp, pp=pp)
    executor.execute_model(scheduler_output("a", 8), non_block=True)
    executor.sample_tokens()
    outcome = executor._pipeline.sink.outcomes[0]
    assert len(outcome.stages) == pp
    assert sum(end - start for start, end in executor._pipeline.layer_ranges) == 2 * pp
    configs = [lowerer.config for lowerer in executor._pipeline.sink._lowerers]
    assert [config.dims.vocab_size for config in configs] == [0] * (pp - 1) + [1024]
    sends = [op.work for op in outcome.graph.operations
             if ":pp-boundary-" in op.operation_id and ":send-" in op.operation_id
             and isinstance(op.work, CollectiveWork)]
    assert len(sends) == 2 * (pp - 1)
    assert sum(size for work in sends for _, _, size in work.pair_payload_bytes) == (
        2 * (pp - 1) * 8 * 128 * 2)
    for index, stage in enumerate(outcome.stages):
        assert stage.step_index == 0 and stage.stage_index == index
        assert stage.request_ids == ("a",)
        if index:
            assert stage.eligible_at_ps >= outcome.stages[index - 1].completed_at_ps
    assert outcome.execution_result.events
    assert outcome.step_result.completed_at_ps == executor.clock.now_ps


def test_pipeline_overlap_and_previous_send_hold(monkeypatch):
    executor = make_executor(monkeypatch)
    for name in ("a", "b"):
        executor.execute_model(scheduler_output(name), non_block=True)
        executor.sample_tokens(non_block=True)
    first, second = executor._pipeline.sink.outcomes
    source_release = next(op.completed_at_ps for op in first.runtime_report.operations
                          if op.operation_id.endswith(":send-release-0"))
    second_compute = [visit for visit in second.runtime_report.visits
                      if "pp-stage-0" in visit.operation_id and visit.service_ps > 0]
    assert min(visit.started_at_ps for visit in second_compute) >= source_release
    assert min(visit.started_at_ps for visit in second_compute) < (
        first.stages[-1].completed_at_ps)
    assert second.stages[0].completed_at_ps == first.stages[-1].completed_at_ps


@pytest.mark.parametrize("change,match", [
    (lambda ex: setattr(ex.vllm_config.scheduler_config, "async_scheduling", True), "synchronous"),
    (lambda ex: setattr(ex.vllm_config.parallel_config, "data_parallel_size", 2), "data_parallel"),
    (lambda ex: setattr(ex.vllm_config.model_config, "enforce_eager", False), "eager"),
    (lambda ex: setattr(ex.vllm_config, "kv_transfer_config", object()), "kv_transfer"),
    (lambda ex: setattr(ex, "dims", replace(ex.dims, defaulted_fields=("hidden_size",))),
     "explicit dense"),
])
def test_unsupported_modes_rejected(monkeypatch, change, match):
    executor = make_executor(monkeypatch)
    change(executor)
    with pytest.raises(ValueError, match=match):
        pipeline.validate_pipeline_config(executor)


def test_legacy_sink_refused_and_structured_failure_emits_nothing(monkeypatch):
    with pytest.raises(TypeError, match="single-stage"):
        make_executor(monkeypatch, sink=lambda record: None)
    executor = make_executor(monkeypatch)
    with pytest.raises(RuntimeError, match="structured"):
        executor.execute_model(scheduler_output(structured=True))
    assert executor.clock.now_ps == 0 and not executor.step_records
    assert not executor._pipeline.pending


def test_link_rate_changes_future_latency(monkeypatch):
    times = []
    for rate in (100_000_000_000, 200_000_000_000):
        sink = PipelineRuntimeStepSink(
            runtime=CoarseDeviceRuntime(CoarseDeviceProfile(rnic_rate_bps=rate)),
            rank_map=(0, 8),
        )
        executor = make_executor(monkeypatch, sink=sink)
        executor.execute_model(scheduler_output(), non_block=True)
        assert executor.clock.now_ps == 0
        executor.sample_tokens()
        times.append(executor.clock.now_ps)
    # Two stages of 10 ns plus two 2048-byte serialized tensor transfers.
    assert times == [20_000 + 2 * 163_840, 20_000 + 2 * 81_920]


def test_compute_only_pipeline_envelope():
    from simllm.core import ExecutionGraph, ExecutionOperation
    from simllm.traffic.pipeline import compose_pipeline_graph

    runtime = CoarseDeviceRuntime()
    completions = []
    for batch in range(3):
        step = record(batch)
        stages = [ExecutionGraph(f"step-{batch}", batch, 0, (
            ExecutionOperation("compute", rank, "compute",
                               ComputeWork("fixed", nominal_duration_ps=10_000)),
        )) for rank in range(4)]
        # Timing-neutral boundary still holds each source until its send.
        boundaries = [ExecutionGraph(f"boundary-{batch}-{rank}", batch, 0, (
            ExecutionOperation("fence", rank, "send",
                               ComputeWork("fence", nominal_duration_ps=0)),
        )) for rank in range(3)]
        graph = compose_pipeline_graph(step, dims(), 4, (0, 1, 2, 3), stages,
                                       boundary_graphs=boundaries)
        completions.append(runtime.execute(graph).completed_at_ps)
    assert completions == [40_000, 50_000, 60_000]


def test_boundaries_refuse_indivisible_payload():
    with pytest.raises(ValueError, match="divide evenly"):
        pipeline.dense_pipeline_boundaries(record(), ((0, 1, 2), (3, 4, 5)), 128, 2)
    with pytest.raises(ValueError, match="divide evenly"):
        pipeline.dense_pipeline_boundaries(record(), ((0, 1), (2, 3)), 129, 2)


def test_execution_failure_poisoning_reaches_pending_futures(monkeypatch):
    executor = make_executor(monkeypatch)
    future = executor.execute_model(scheduler_output("a"), non_block=True)
    sample = executor.sample_tokens(non_block=True)

    def fail(record):
        raise ValueError("invalid runtime submission")

    monkeypatch.setattr(executor._pipeline.sink, "prepare", fail)
    with pytest.raises(ValueError, match="invalid runtime"):
        executor.execute_model(scheduler_output("b"), non_block=True)
    with pytest.raises(ValueError, match="invalid runtime"):
        sample.result()
    assert isinstance(future.exception(), ValueError)
    assert executor.clock.now_ps == 0 and not executor.step_records
    with pytest.raises(RuntimeError, match="closed"):
        executor.execute_model(scheduler_output("c"))


def test_completion_callback_can_consume_sibling(monkeypatch):
    executor = make_executor(monkeypatch)
    execute = executor.execute_model(scheduler_output("a"), non_block=True)
    sample = executor.sample_tokens(non_block=True)
    received = []
    execute.add_done_callback(lambda future: received.append(sample.result().req_ids))
    assert execute.result() is None
    assert received == [["a"]]


def test_shutdown_callbacks_cannot_retire_later_work(monkeypatch):
    executor = make_executor(monkeypatch)
    first = executor.execute_model(scheduler_output("a"), non_block=True)
    executor.sample_tokens(non_block=True)
    executor.execute_model(scheduler_output("b"), non_block=True)
    later = executor.sample_tokens(non_block=True)
    errors = []
    first.add_done_callback(lambda future: errors.append(later.exception()))
    executor.shutdown()
    assert isinstance(errors[0], RuntimeError)
    assert executor.clock.now_ps == 0 and not executor.step_records


def test_record_write_failure_fails_futures_instead_of_hanging(monkeypatch):
    executor = make_executor(monkeypatch)

    def fail(record):
        raise OSError("record write failed")

    executor._runtime.record_stream = SimpleNamespace(append=fail)
    execute = executor.execute_model(scheduler_output("a"), non_block=True)
    sample = executor.sample_tokens(non_block=True)
    with pytest.raises(OSError, match="record write failed"):
        sample.result()
    assert isinstance(execute.exception(), OSError)
    assert executor._pipeline.closed


def test_paced_mode_sleeps_only_new_clock_increment(monkeypatch):
    executor = make_executor(monkeypatch)
    executor._runtime.config = replace(executor.config, mode="paced")
    sleeps = []
    monkeypatch.setattr(pipeline.time, "sleep", sleeps.append)
    for name in ("a", "b"):
        executor.execute_model(scheduler_output(name), non_block=True)
    samples = [executor.sample_tokens(non_block=True) for _ in range(2)]
    samples[-1].result()
    samples[0].result()
    assert len(sleeps) == 2
    assert sum(round(seconds * 1e12) for seconds in sleeps) == executor.clock.now_ps
    assert sum(row.step_latency_ps for row in executor.step_results) > executor.clock.now_ps


def test_rank_mapping_requires_complete_unique_world(monkeypatch):
    with pytest.raises(ValueError, match="distinct"):
        PipelineRuntimeStepSink(rank_map=(0, 0))
    with pytest.raises(ValueError, match="exactly"):
        make_executor(monkeypatch, sink=PipelineRuntimeStepSink(rank_map=(0,)))
    with pytest.raises(ValueError, match="exactly"):
        make_executor(monkeypatch, sink=PipelineRuntimeStepSink(rank_map=()))


def test_prepare_requires_bound_nonempty_pipeline():
    sink = PipelineRuntimeStepSink()
    with pytest.raises(RuntimeError, match="bind_pipeline"):
        sink.prepare(record())
    config = SerialStepLowererConfig(dims=dims(), tp_ranks=(0,))
    sink.bind_pipeline((config, replace(config, tp_ranks=(1,))), ((0, 2), (2, 4)),
                       lambda step: ())
    with pytest.raises(ValueError, match="drains"):
        sink.prepare(StepRecord(0, 0, ()))
