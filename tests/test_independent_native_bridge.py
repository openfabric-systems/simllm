"""Component fixtures exercise the native call boundary without a vLLM install."""

import sys
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from simllm.adapters.vllm.executor import (
    SimExecutor,
    SimExecutorConfig,
    TranslatedStep,
    _SimStepRuntime,
)
from simllm.adapters.vllm.independent import (
    NativeEngineCompletion,
    native_engine_state,
    native_value_snapshot,
)
from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
from simllm.compute import GPU_ENVELOPES, HostInitiationModel, ModelDims, RooflineProvider
from simllm.core import RequestPhase, ScheduledRequest, StepRecord, VirtualClock
from simllm.core.engine_steps import EngineStepRuntime
from simllm.placement import declared_manifest


@dataclass
class NativeOutput:
    req_ids: list
    req_id_to_index: dict
    sampled_token_ids: list | None = None


@dataclass
class SchedulerOutput:
    num_scheduled_tokens: dict
    total_num_scheduled_tokens: int
    has_structured_output_requests: bool = False
    finished_req_ids: set[str] = field(default_factory=set)


class Status(Enum):
    WAITING = 0
    RUNNING = 1


@pytest.fixture
def bridge_fixture(tmp_path, monkeypatch):
    calls = []
    clock = VirtualClock()
    authority = EngineStepRuntime(clock)
    config = SimExecutorConfig(mode="virtual")
    dims = ModelDims(num_layers=1, hidden_size=64, intermediate_size=128,
                     num_heads=4, num_kv_heads=4, head_size=16, vocab_size=256, dtype_bytes=2)
    provider = RooflineProvider()
    host = HostInitiationModel.ideal()
    sink = HtsimStepSink(HtsimStepSinkConfig(
        profile="rnic-nn-fluid", tp_ranks=(0, 1), dims=dims, workdir=tmp_path,
        provider=provider, placement_manifest=declared_manifest(tp=2, nodes=1, gpus_per_node=2),
        collective_fixed_cost_envelope="intra-node-fixed-cost-v1", collective_fixed_cost_arm="lower"))
    executor = SimExecutor.__new__(SimExecutor)
    local = _SimStepRuntime(config=config, step_sink=sink, fallback_latency=lambda step: 1,
                            clock=clock, host_model=host, gpu=GPU_ENVELOPES["b100"])
    executor.config, executor.dims, executor.compute_provider = config, dims, provider
    executor.gpu, executor.host_model, executor.step_sink = GPU_ENVELOPES["b100"], host, sink
    executor._runtime, executor.clock, executor.replay = local, clock, None
    executor.step_records, executor.step_results = local.step_records, local.step_results
    executor._pending_output, executor.token_id, executor.world_size = None, 128, 2

    def translate(output):
        if output.total_num_scheduled_tokens:
            record = StepRecord(local.step_index, clock.now_ps,
                                [ScheduledRequest("native-a", RequestPhase.PREFILL, 8, context_length=8)], num_sampled=1)
        else:
            record = StepRecord(local.step_index, clock.now_ps, finished_request_ids=list(output.finished_req_ids))
        local.step_index += 1
        return TranslatedStep(record, ["native-a"], [True])

    monkeypatch.setattr(local, "translate", translate)
    block = NS(block_id=1, ref_cnt=0)
    free = [block]
    group = NS(req_to_blocks={})
    pool = NS(blocks=[NS(block_id=0, ref_cnt=0), block], free_block_queue=NS(get_all_free_blocks=lambda: list(free)))
    coordinator = NS(block_pool=pool, single_type_managers=[group])
    request = NS(request_id="native-a", max_tokens=1, status=Status.WAITING, num_computed_tokens=0, num_in_flight_tokens=0,
                 output_token_ids=[], all_token_ids=list(range(8)), stop_reason=None, kv_transfer_params=None,
                 structured_output_request=None, mm_features=[], pooling_params=None, lora_request=None,
                 prompt_embeds=None, resumable=False)
    visible = NS(external_req_id="a", detokenizer=NS(output_token_ids=[]), is_prefilling=True, sent_tokens_offset=0)
    cfg = NS(parallel_config=NS(tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1),
             scheduler_config=NS(max_num_seqs=1, async_scheduling=False, policy="fcfs", enable_chunked_prefill=False),
             cache_config=NS(enable_prefix_caching=False), speculative_config=None)

    class Scheduler:
        def __init__(self):
            self.requests = {"native-a": request}
            self.max_model_len = 64
            self.waiting, self.running, self.finished = [request], [], set()
            self.kv_cache_manager = NS(coordinator=coordinator)

        def has_requests(self):
            return bool(self.requests or self.finished)

        def schedule(self, throttle):
            calls.append(("schedule", throttle, clock.now_ps))
            if self.finished:
                finished, self.finished = self.finished, set()
                return SchedulerOutput({}, 0, finished_req_ids=finished)
            self.waiting, self.running = [], [request]
            request.status, request.num_computed_tokens, request.num_in_flight_tokens = Status.RUNNING, 8, 8
            block.ref_cnt = 1
            free.clear()
            group.req_to_blocks["native-a"] = [block]
            return SchedulerOutput({"native-a": 8}, 8)

        def get_grammar_bitmask(self, output):
            calls.append(("grammar", output))

        def update_from_output(self, output, model):
            calls.append(("update", output, model, clock.now_ps))
            if not output.total_num_scheduled_tokens:
                return {0: model}
            self.running, self.finished = [], {"native-a"}
            request.output_token_ids.extend(model.sampled_token_ids[0])
            request.num_in_flight_tokens = 0
            self.requests.clear()
            group.req_to_blocks.clear()
            block.ref_cnt = 0
            free.append(block)
            return {0: model}

    def no_op(*args, **kwargs):
        calls.append(("native-entry", args, kwargs))

    class Core:
        def __init__(self):
            self.model_executor, self.scheduler = executor, Scheduler()
            self.async_scheduling = self.log_stats = self.check_for_draft_tokens = False
            self.step_fn = self.step

        def step(self):
            raise AssertionError("old whole-step executor must not run")

        def _should_throttle_prefills(self):
            return False

        @contextmanager
        def capture_iteration_details(self, output):
            calls.append(("iteration-enter", output))
            yield None
            calls.append(("iteration-exit",))

        @contextmanager
        def log_error_detail(self, output):
            calls.append(("error-enter", output))
            yield
            calls.append(("error-exit",))

        def _process_aborts_queue(self):
            calls.append(("aborts",))

        def _attach_iteration_details(self, outputs, details):
            calls.append(("attach", outputs, details))

        def post_step(self, model_executed):
            calls.append(("post-step", model_executed))

        abort_requests = reset_prefix_cache = reset_mm_cache = reset_encoder_cache = shutdown = no_op

    class InprocClient:
        def __init__(self):
            self.engine_core = Core()

        def get_output(self):
            outputs, executed = self.engine_core.step_fn()
            self.engine_core.post_step(model_executed=executed)
            return outputs[0]

        def abort_requests(self, ids):
            return self.engine_core.abort_requests(ids)

    class Processor:
        def __init__(self):
            self.request_states = {"native-a": visible}

        def process_outputs(self, model):
            calls.append(("output", clock.now_ps))
            self.request_states.clear()
            return model.sampled_token_ids or []

        update_scheduler_stats = no_op

    class Frontend:
        def __init__(self):
            self.engine_core, self.vllm_config, self.log_stats = InprocClient(), cfg, False
            self.output_processor = Processor()

        def step(self):
            model = self.engine_core.get_output()
            output = self.output_processor.process_outputs(model)
            self.engine_core.abort_requests([])
            return [NS(request_id="a", finished=True, outputs=[NS(token_ids=tokens)], kv_transfer_params=None)
                    for tokens in output]

        abort_request = reset_prefix_cache = reset_mm_cache = reset_encoder_cache = sleep = no_op

    class Struct:
        pass

    for name, members in (
        ("vllm", {"__version__": "0.27.1"}),
        ("vllm.v1.engine.core_client", {"InprocClient": InprocClient}),
        ("vllm.v1.outputs", {"ModelRunnerOutput": NativeOutput}),
        ("msgspec", {"Struct": Struct}),
    ):
        module = ModuleType(name)
        vars(module).update(members)
        monkeypatch.setitem(sys.modules, name, module)
    frontend = Frontend()
    engine = NS(engine_id="engine-a", llm=NS(llm_engine=frontend), executor=executor, step_sink=sink)
    checkpoints = []
    bridge = NativeEngineCompletion(engine, authority,
                                     observer=lambda phase, engine, state: checkpoints.append((phase, state)))
    return NS(bridge=bridge, runtime=authority, calls=calls, executor=executor, sink=sink,
              frontend=frontend, request=request, visible=visible, group=group, block=block,
              free=free, checkpoints=checkpoints, struct=Struct)


def test_native_submission_reserves_cache_and_only_due_processing_publishes(bridge_fixture):
    f = bridge_fixture
    outputs = []
    receipt = f.bridge.submit(lambda result, record: outputs.append((result, record, f.runtime.clock.now_ps)))
    assert [row[0] for row in f.calls] == ["schedule", "grammar"]
    assert f.block.ref_cnt == 1 and f.free == [] and f.request.output_token_ids == []
    assert f.visible.detokenizer.output_token_ids == []
    assert len(f.executor.step_records) == 1 and f.executor.step_results == [] and f.sink.outcomes == []
    f.runtime.advance_to(receipt.completed_at_ps - 1)
    assert f.runtime.complete_due() == () and outputs == []
    f.runtime.advance_to(receipt.completed_at_ps)
    f.runtime.complete_due()
    assert [row[0] for row in f.calls] == ["schedule", "grammar", "iteration-enter", "error-enter",
        "error-exit", "iteration-exit", "aborts", "update", "attach", "post-step", "output", "native-entry"]
    assert f.calls[1][1] is f.calls[7][1]
    assert outputs[0][0][0].outputs[0].token_ids == [128] and outputs[0][2] == receipt.completed_at_ps
    assert f.block.ref_cnt == 0 and f.free == [f.block] and len(f.executor.step_results) == 1
    assert [row[0] for row in f.checkpoints] == ["before-submit", "submitted", "before-retire", "retired"]
    f.bridge.close()
    assert f.bridge.core.step_fn == f.bridge.core.step


@pytest.mark.parametrize("mutation", ["native-token", "frontend-token", "cache-ref", "cache-owner", "cache-free", "native-config", "sink-payload", "request-limit", "stale-output", "live-limit"])
def test_pending_native_state_cannot_change_before_completion(bridge_fixture, mutation):
    f = bridge_fixture
    receipt = f.bridge.submit(lambda result, record: None)
    if mutation == "native-token":
        f.request.output_token_ids.append(128)
    elif mutation == "frontend-token":
        f.visible.detokenizer.output_token_ids.append(128)
    elif mutation == "cache-ref":
        f.block.ref_cnt = 0
    elif mutation == "cache-owner":
        f.group.req_to_blocks.clear()
    elif mutation == "cache-free":
        f.free.append(f.block)
    elif mutation == "native-config":
        f.frontend.vllm_config.scheduler_config.max_num_seqs = 2
    elif mutation == "request-limit":
        f.request.max_tokens = 2
    elif mutation == "stale-output":
        f.request.drop_stale_output = True
    elif mutation == "live-limit":
        f.bridge.core.scheduler.max_model_len = 32
    else:
        f.sink.outcomes.append(object())
    f.runtime.advance_to(receipt.completed_at_ps)
    with pytest.raises((RuntimeError, TypeError, ValueError), match="changed|unsupported"):
        f.runtime.complete_due()
    assert f.runtime.failure and not f.executor.step_results and not f.runtime.results


@pytest.mark.parametrize("entry", ["abort_request", "reset_prefix_cache", "reset_mm_cache", "reset_encoder_cache", "sleep"])
def test_pending_frontend_entry_rejects_before_any_native_mutation(bridge_fixture, entry):
    f = bridge_fixture
    f.bridge.submit(lambda result, record: None)
    before = native_engine_state(f.frontend)
    with pytest.raises(RuntimeError, match="CORE-69"):
        getattr(f.frontend, entry)([])
    assert f.runtime.failure and native_engine_state(f.frontend) == before
    assert [row[0] for row in f.calls] == ["schedule", "grammar"]


def test_native_step_without_due_callback_and_unsupported_request_reject(bridge_fixture):
    f = bridge_fixture
    f.request.structured_output_request = object()
    with pytest.raises(NotImplementedError, match="unsupported independent native"):
        f.bridge.submit(lambda result, record: None)
    assert f.runtime.failure and f.calls == [] and f.executor.step_records == []


def test_native_struct_snapshot_preserves_each_typed_field(bridge_fixture):
    class Params(bridge_fixture.struct):
        __struct_fields__ = ("selection", "nested")

        def __init__(self):
            self.selection, self.nested = "chosen", {"value": [1]}

    params = Params()
    state = native_value_snapshot(params)
    params.nested["value"].append(2)
    assert native_value_snapshot(params) != state


def test_mode_off_and_unsupported_configuration_before_engine_construction(tmp_path, monkeypatch):
    from examples.pd_session_v1.run_study import _session_config
    from simllm.adapters.vllm.pd_session import VllmDisaggregatedSession

    config = _session_config(tmp_path)
    assert config.engine_timing == "serialized"
    with pytest.raises(NotImplementedError, match="CORE-70"):
        replace(config, engine_timing="independent")
    calls = []
    monkeypatch.setattr(VllmDisaggregatedSession, "_build_pools", lambda self: calls.append(self))
    with pytest.raises(ValueError, match="completion_observer requires"):
        VllmDisaggregatedSession(config, completion_observer=lambda *args: None)
    assert calls == []


def test_finished_only_native_drain_is_consumed_once_at_current_time(bridge_fixture):
    f = bridge_fixture
    receipt = f.bridge.submit(lambda output, record: None)
    f.runtime.advance_to(receipt.completed_at_ps)
    f.runtime.complete_due()
    assert f.bridge.has_work and not f.frontend.output_processor.request_states
    drain = f.bridge.submit(lambda output, record: None)
    assert drain.completed_at_ps == f.runtime.clock.now_ps
    assert f.executor.step_records[-1].finished_request_ids == ["native-a"]
    assert f.runtime.complete_due() == (drain,)
    assert not f.bridge.has_work and f.executor.step_results[-1].step_latency_ps == 0


def test_missing_wrapper_still_restores_every_other_binding(bridge_fixture):
    f = bridge_fixture
    del f.frontend.reset_mm_cache
    with pytest.raises(RuntimeError, match="restoration failed"):
        f.bridge.close()
    assert f.bridge.core.step_fn == f.bridge.core.step
    assert "abort_request" not in vars(f.frontend) and "abort_requests" not in vars(f.bridge.core)
    assert f.bridge._restorations == [] and f.runtime.failure
    f.runtime.close()


def test_occupied_clock_rejects_before_constructing_native_pools(tmp_path, monkeypatch):
    from examples.pd_session_v1.run_study import _session_config
    from simllm.adapters.vllm.pd_session import VllmDisaggregatedSession

    config = replace(_session_config(tmp_path), max_num_seqs=1, engine_timing="independent")
    clock = VirtualClock()
    clock.schedule(1, object())
    calls = []
    monkeypatch.setattr(VllmDisaggregatedSession, "_build_pools", lambda self: calls.append(self))
    with pytest.raises(ValueError, match="empty event clock"):
        VllmDisaggregatedSession(config, clock=clock)
    assert not calls


@pytest.mark.parametrize("corruption", ["none", "running", "reorder"])
def test_new_waiting_admission_preserves_existing_pending_request(bridge_fixture, corruption):
    from copy import deepcopy

    f = bridge_fixture
    receipt = f.bridge.submit(lambda output, record: None)
    added = deepcopy(f.request)
    added.request_id, added.status = "native-b", Status.WAITING
    added.num_computed_tokens = added.num_in_flight_tokens = 0
    f.bridge.core.scheduler.requests["native-b"] = added
    f.bridge.core.scheduler.waiting.append(added)
    f.frontend.output_processor.request_states["native-b"] = deepcopy(f.visible)
    if corruption == "running":
        added.status = Status.RUNNING
    elif corruption == "reorder":
        f.bridge.core.scheduler.running.insert(0, added)
    f.runtime.advance_to(receipt.completed_at_ps)
    if corruption == "none":
        f.runtime.complete_due()
        assert f.runtime.failure is None
    else:
        with pytest.raises(RuntimeError, match="changed|executed"):
            f.runtime.complete_due()
        assert f.runtime.failure and not f.executor.step_results


def test_partial_bridge_installation_restores_the_original_native_entries(bridge_fixture, monkeypatch):
    f = bridge_fixture
    engine = f.bridge.engine
    f.bridge.close()
    original = NativeEngineCompletion._replace
    calls = []

    def failing_replace(self, owner, name, value):
        calls.append(name)
        if len(calls) == 3:
            raise RuntimeError("installation failed")
        original(self, owner, name, value)

    monkeypatch.setattr(NativeEngineCompletion, "_replace", failing_replace)
    with pytest.raises(RuntimeError, match="installation failed"):
        NativeEngineCompletion(engine, f.runtime)
    assert f.bridge.core.step_fn == f.bridge.core.step
    assert "abort_request" not in vars(f.frontend)
