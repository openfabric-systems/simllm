"""Native scheduling and output processing around one core-owned due event.

Scheduling reserves native work now. Its deterministic output stays private
until the core completion authority drives the unchanged frontend step.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from simllm.backends.step_sink import DeferredStepPrice, HtsimStepSink
from simllm.core.engine_steps import EngineStepReceipt, EngineStepRuntime, PublicationBinding
from simllm.core.step import StepResult
from simllm.core.value_snapshot import value_snapshot


def native_value_snapshot(value: object) -> tuple:
    """Project every field of native message structs, with no opaque fallback."""
    import msgspec

    def project(item: object) -> object:
        if isinstance(item, msgspec.Struct):
            return {name: getattr(item, name) for name in type(item).__struct_fields__}
        return NotImplemented

    return value_snapshot(value, project=project)


def _bindings(owner: object, names: tuple[str, ...]) -> tuple:
    rows = []
    for name in names:
        method = getattr(owner, name)
        function = getattr(method, "__func__", method)
        rows.append((name, id(getattr(method, "__self__", None)), id(function),
                     id(getattr(function, "__code__", None))))
    return tuple(rows)


@dataclass(frozen=True)
class PreparedExecutorStep:
    """An unpublished executor projection bound privately by the core runtime."""

    executor: Any
    scheduler_output: Any
    translated: Any
    price: DeferredStepPrice
    model_output: Any


def _executor_values(prepared: PreparedExecutorStep) -> tuple:
    executor = prepared.executor
    local = executor._runtime
    return (id(executor), id(local), id(executor.step_sink), id(local.step_sink),
            id(executor.clock), id(local.clock), id(executor.config), id(local.config),
            id(executor.compute_provider), type(executor.compute_provider).__module__,
            type(executor.compute_provider).__qualname__, vars(executor.compute_provider),
            executor.config, local.config, executor.dims, executor.gpu, executor.host_model,
            local.gpu, local.host_model, executor.token_id, executor.world_size,
            id(executor.step_records), id(local.step_records), id(executor.step_results), id(local.step_results),
            executor.step_records, executor.step_results, local.step_index, vars(local.translator),
            prepared.translated, id(prepared.price), native_value_snapshot(prepared.scheduler_output),
            native_value_snapshot(prepared.model_output), native_value_snapshot(executor._pending_output),
            _bindings(executor, ("prepare_independent", "_sample_output_fields", "_empty_output", "_run_step")),
            _bindings(local, ("translate", "settle", "drain")))


def validate_executor(executor: Any) -> None:
    """Reject unsupported service paths before the native scheduler is called."""
    from simllm.adapters.vllm.executor import SimExecutor

    if type(executor) is not SimExecutor:
        raise TypeError("independent completion requires the simulated native executor")
    if executor.config.mode != "virtual" or executor.replay is not None:
        raise NotImplementedError("independent completion excludes paced or replay service (CORE-70)")
    local = executor._runtime
    if (not local.is_authority or executor.clock is not local.clock
            or executor.step_records is not local.step_records or executor.step_results is not local.step_results
            or executor.step_sink is not local.step_sink or executor.config is not local.config):
        raise RuntimeError("independent executor aliases do not name one runtime")
    if type(executor.step_sink) is not HtsimStepSink:
        raise NotImplementedError("independent completion requires declared local service (CORE-70)")
    if executor.step_sink.config.provider is not executor.compute_provider:
        raise RuntimeError("independent executor and sink must share the compute provider")
    local._validate_host_model_selection()
    executor.step_sink.validate_deferred_mode()


def prepare_executor_step(executor: Any, scheduler_output: Any) -> PreparedExecutorStep:
    """Keep model output private and append only the dispatched input record."""
    from vllm.v1.outputs import ModelRunnerOutput

    validate_executor(executor)
    if getattr(scheduler_output, "has_structured_output_requests", False):
        raise NotImplementedError("independent structured output is unavailable (VLLM-8)")
    translated = executor._runtime.translate(scheduler_output)
    price = executor.step_sink.prepare_deferred(translated.record)
    if translated.record.scheduled:
        req_ids, by_id, sampled = executor._sample_output_fields(translated, scheduler_output)
        output = ModelRunnerOutput(req_ids=req_ids, req_id_to_index=by_id, sampled_token_ids=sampled)
    else:
        output = executor._empty_output()
    # Strictly snapshot native inputs before publishing even the input record.
    native_value_snapshot(scheduler_output)
    native_value_snapshot(output)
    executor.step_records.append(translated.record)
    if executor._runtime.record_stream is not None:
        executor._runtime.record_stream.append(translated.record)
    return PreparedExecutorStep(executor, scheduler_output, translated, price, output)


def _publish_executor_step(prepared: PreparedExecutorStep, runtime: EngineStepRuntime,
                           receipt: EngineStepReceipt) -> Any:
    executor = prepared.executor
    runtime.claim_publication(receipt, publisher=executor, payload=prepared,
                              record=prepared.translated.record, result=prepared.price.result)
    result = executor.step_sink.publish_deferred(prepared.price, runtime, receipt)
    executor.step_results.append(result)
    executor._pending_output = prepared.model_output
    return prepared.model_output


def native_engine_state(frontend: Any) -> dict:
    """Read request/cache identities without advancing either native lifecycle."""
    core = frontend.engine_core.engine_core
    scheduler = core.scheduler
    manager = scheduler.kv_cache_manager
    coordinator = manager.coordinator
    pool = coordinator.block_pool
    requests = {}
    for request_id, request in scheduler.requests.items():
        requests[request_id] = {
            "status": request.status.name,
            "num_computed_tokens": request.num_computed_tokens,
            "num_in_flight_tokens": request.num_in_flight_tokens,
            "output_token_ids": list(request.output_token_ids),
            "all_token_ids": list(request.all_token_ids),
            "stop_reason": request.stop_reason,
            "kv_transfer_params": deepcopy(request.kv_transfer_params),
        }
    visible = {}
    for request_id, request in frontend.output_processor.request_states.items():
        visible[request_id] = {
            "external_request_id": request.external_req_id,
            "output_token_ids": list(request.detokenizer.output_token_ids),
            "is_prefilling": request.is_prefilling,
            "sent_tokens_offset": request.sent_tokens_offset,
        }
    cache = {
        "groups": [
            {request_id: [block.block_id for block in blocks]
             for request_id, blocks in manager.req_to_blocks.items()}
            for manager in coordinator.single_type_managers
        ],
        "blocks": [[block.block_id, block.ref_cnt] for block in pool.blocks],
        "free_queue": [block.block_id for block in pool.free_block_queue.get_all_free_blocks()],
    }
    queues = {name: [request.request_id for request in getattr(scheduler, name)]
              for name in ("waiting", "running")}
    return {"requests": requests, "visible": visible, "cache": cache, "queues": queues}


def _pending_native_values(frontend: Any, request_ids: tuple[str, ...], visible_ids: tuple[str, ...]) -> tuple:
    """Freeze complete retained request values behind the native update call.

    Native ConstantList fields are read-only aliases of the underlying token
    lists. The optional block hasher is a bound source identity, not value data.
    All other request members, including sampling and stopping state, retain
    their exact typed values. A missing request is an observable state change.
    """
    scheduler = frontend.engine_core.engine_core.scheduler
    requests = {}
    for request_id in request_ids:
        request = scheduler.requests.get(request_id)
        if request is None:
            requests[request_id] = None
            continue
        values = {name: value for name, value in vars(request).items()
                  if name not in ("output_token_ids", "all_token_ids", "_block_hasher")}
        values["output_token_ids"] = list(request.output_token_ids)
        values["all_token_ids"] = list(request.all_token_ids)
        hasher = getattr(request, "_block_hasher", None)
        values["_block_hasher_binding"] = (id(hasher), id(getattr(hasher, "__code__", None)))
        requests[request_id] = values
    visible = {}
    for request_id in visible_ids:
        request = frontend.output_processor.request_states.get(request_id)
        if request is None:
            visible[request_id] = None
            continue
        values = dict(vars(request))
        for name in ("detokenizer", "logprobs_processor"):
            member = values.get(name)
            if member is not None:
                values[name] = (type(member).__module__, type(member).__qualname__, vars(member))
        visible[request_id] = values
    return native_value_snapshot((requests, visible, scheduler.max_model_len))


class NativeEngineCompletion:
    """Bind one native step callback to the existing shared completion runtime."""

    def __init__(self, engine: Any, runtime: EngineStepRuntime, *,
                 observer: Callable[[str, Any, dict], None] | None = None):
        from vllm import __version__
        from vllm.v1.engine.core_client import InprocClient

        self.engine = engine
        self.runtime = runtime
        self.observer = observer
        self.frontend = engine.llm.llm_engine
        self.client = self.frontend.engine_core
        if __version__ != "0.27.1" or type(self.client) is not InprocClient:
            raise NotImplementedError("independent completion requires in-process vLLM 0.27.1 (CORE-70)")
        self.core = self.client.engine_core
        self.executor = engine.executor
        self._delivery: tuple[PreparedExecutorStep, EngineStepReceipt] | None = None
        self._original_step = self.core.step_fn
        if self._original_step != self.core.step:
            raise NotImplementedError("independent completion requires the native synchronous step (VLLM-10)")
        self._validate_mode()
        self._restorations: list[tuple[object, str, bool, Any]] = []
        try:
            self._replace(self.core, "step_fn", self._retire_native)
            # Frontend abort mutates its output processor before forwarding to
            # core. Guard each entry before either layer can release state.
            for owner, names in (
                (self.frontend, ("abort_request", "reset_prefix_cache", "reset_mm_cache", "reset_encoder_cache", "sleep")),
                (self.core, ("abort_requests", "reset_prefix_cache", "reset_mm_cache", "reset_encoder_cache", "shutdown")),
            ):
                for name in names:
                    original = getattr(owner, name)

                    def guarded(*args, _original=original, _name=name, **kwargs):
                        if self.pending is not None and self._delivery is None:
                            self._fail("pending native work cannot " + _name + " (CORE-69)")
                        return _original(*args, **kwargs)

                    self._replace(owner, name, guarded)
            self._source_bindings = self._read_bindings()
        except BaseException:
            self._restore()
            raise

    @property
    def pending(self) -> EngineStepReceipt | None:
        return self.runtime.pending_for(self.engine.engine_id)

    @property
    def has_work(self) -> bool:
        """Include the native scheduler's completion-bearing zero-work drain."""
        return self.core.scheduler.has_requests()

    def _fail(self, reason: str) -> None:
        self.runtime.invalidate(reason)
        raise RuntimeError(reason)

    def _replace(self, owner: object, name: str, value: Any) -> None:
        self._restorations.append((owner, name, name in vars(owner), getattr(owner, name)))
        setattr(owner, name, value)

    def _validate_mode(self) -> None:
        validate_executor(self.executor)
        if (self.executor.clock is not self.runtime.clock or self.core.model_executor is not self.executor
                or self.engine.step_sink is not self.executor.step_sink):
            self._fail("native engine, executor, sink or clock binding disagrees")
        config = self.frontend.vllm_config
        parallel, scheduler = config.parallel_config, config.scheduler_config
        if (parallel.pipeline_parallel_size != 1 or parallel.data_parallel_size != 1
                or parallel.tensor_parallel_size != self.executor.world_size
                or scheduler.async_scheduling or scheduler.policy != "fcfs" or scheduler.max_num_seqs != 1
                or self.core.async_scheduling or self.core.log_stats or self.frontend.log_stats
                or config.speculative_config is not None or self.core.check_for_draft_tokens):
            raise NotImplementedError("independent completion requires synchronous single-sequence TP service (CORE-70)")
        for request in self.core.scheduler.requests.values():
            if (request.structured_output_request is not None or request.mm_features
                    or request.pooling_params is not None or request.lora_request is not None
                    or request.prompt_embeds is not None or request.resumable):
                raise NotImplementedError("unsupported independent native request (CORE-70, VLLM-8)")

    def _selected_values(self) -> tuple:
        config = self.frontend.vllm_config
        parallel, scheduler = config.parallel_config, config.scheduler_config
        return (id(config), id(config.parallel_config), id(config.scheduler_config),
                parallel.tensor_parallel_size, parallel.pipeline_parallel_size, parallel.data_parallel_size,
                scheduler.max_num_seqs, scheduler.async_scheduling, scheduler.policy,
                scheduler.enable_chunked_prefill, config.cache_config.enable_prefix_caching,
                config.speculative_config, self.core.async_scheduling, self.core.log_stats, self.frontend.log_stats,
                self.core.check_for_draft_tokens, native_value_snapshot(vars(scheduler)),
                self.executor.config, self.executor.dims, self.executor.gpu, self.executor.host_model)

    def _read_bindings(self) -> tuple:
        return (id(self.engine.llm.llm_engine), id(self.frontend.engine_core), id(self.client.engine_core),
                id(self.core.model_executor), id(self.core.scheduler), id(self.engine.executor),
                id(self.engine.step_sink), id(self.frontend.output_processor),
                _bindings(self.frontend, ("step", "abort_request", "reset_prefix_cache", "reset_mm_cache", "reset_encoder_cache", "sleep")),
                _bindings(self.client, ("get_output", "abort_requests")),
                _bindings(self.core, ("step_fn", "post_step", "_should_throttle_prefills", "capture_iteration_details",
                                      "log_error_detail", "_process_aborts_queue", "_attach_iteration_details",
                                      "abort_requests", "reset_prefix_cache", "reset_mm_cache", "reset_encoder_cache", "shutdown")),
                _bindings(self.core.scheduler, ("schedule", "get_grammar_bitmask", "update_from_output", "has_requests")),
                _bindings(self.frontend.output_processor, ("process_outputs", "update_scheduler_stats")))

    def _observe(self, phase: str, outputs: list[Any] | None = None) -> None:
        if self.observer is not None:
            state = native_engine_state(self.frontend)
            state["emitted"] = [
                {"request_id": output.request_id, "finished": output.finished,
                 "token_ids": list(output.outputs[0].token_ids),
                 "kv_transfer_params": deepcopy(output.kv_transfer_params)}
                for output in (outputs or [])
            ]
            self.observer(phase, self.engine, state)

    def submit(self, consume_outputs: Callable[[list[Any], Any], None]) -> EngineStepReceipt:
        """Schedule once; only the core due callback may reveal its output."""
        try:
            next_due = self.runtime.next_completion_ps
            if self.pending is not None:
                self._fail("native engine already has pending work")
            if next_due == self.runtime.clock.now_ps:
                self._fail("existing native completions must retire before submission")
            self._validate_mode()
            if self._read_bindings() != self._source_bindings:
                self._fail("native source or owner binding changed")
            if not self.core.scheduler.has_requests():
                self._fail("native engine has no request or completion to schedule")
            self._observe("before-submit")
            scheduler_output = self.core.scheduler.schedule(self.core._should_throttle_prefills())
            prepared = self.executor.prepare_independent(scheduler_output)
            grammar_output = self.core.scheduler.get_grammar_bitmask(scheduler_output)
            if grammar_output is not None:
                self._fail("independent native grammar output must be absent (VLLM-8)")
            native_after_submit = native_engine_state(self.frontend)
            native_state = value_snapshot(native_after_submit)
            request_ids = tuple(native_after_submit["requests"])
            visible_ids = tuple(native_after_submit["visible"])
            pending_reader = _pending_native_values
            pending_values = pending_reader(self.frontend, request_ids, visible_ids)
            selected_values = value_snapshot(self._selected_values())
            source_bindings = self._source_bindings
            binding_reader = NativeEngineCompletion._read_bindings
            selection_reader = NativeEngineCompletion._selected_values

            def guard() -> None:
                if binding_reader(self) != source_bindings:
                    self._fail("pending native source or owner binding changed")
                if value_snapshot(selection_reader(self)) != selected_values:
                    self._fail("pending native selected configuration changed")
                current = native_engine_state(self.frontend)
                # New request admissions may append waiting entries while this
                # step runs. They cannot alter previously admitted state/cache.
                new_ids = set(current["requests"]) - set(request_ids)
                prior_waiting = native_after_submit["queues"]["waiting"]
                added_waiting = current["queues"]["waiting"][len(prior_waiting):]
                if (current["queues"]["waiting"][:len(prior_waiting)] != prior_waiting
                        or len(added_waiting) != len(new_ids) or set(added_waiting) != new_ids
                        or set(current["visible"]) - set(visible_ids) != new_ids
                        or any(current["requests"][key]["status"] != "WAITING"
                               or current["requests"][key]["num_computed_tokens"] != 0
                               or current["requests"][key]["num_in_flight_tokens"] != 0
                               or current["requests"][key]["output_token_ids"]
                               or current["visible"][key]["output_token_ids"] for key in new_ids)):
                    self._fail("new native admission changed waiting order or executed before scheduling")
                current["queues"]["waiting"] = prior_waiting
                current["requests"] = {key: current["requests"].get(key)
                                       for key in native_after_submit["requests"]}
                current["visible"] = {key: current["visible"].get(key)
                                      for key in native_after_submit["visible"]}
                if value_snapshot(current) != native_state:
                    self._fail("native request or cache state changed before due completion")
                if pending_reader(self.frontend, request_ids, visible_ids) != pending_values:
                    self._fail("native pending request values changed before due completion")
                prepared.price.validate()

            receipt = None

            def publish(result: StepResult) -> None:
                if receipt is None or result is not prepared.price.result:
                    self._fail("native callback lost its owned engine receipt")
                self._observe("before-retire")
                guard()
                self._delivery = (prepared, receipt)
                try:
                    outputs = self.frontend.step()
                    consume_outputs(outputs, prepared.translated.record)
                finally:
                    self._delivery = None
                self._observe("retired", outputs)

            receipt = self.runtime.submit(
                self.engine.engine_id, prepared.translated.record, prepared.price.result,
                guard=guard, publish=publish,
                publications=(
                    PublicationBinding(self.executor, prepared, _executor_values),
                    self.executor.step_sink.deferred_publication(prepared.price),
                ),
            )
            self._observe("submitted")
            return receipt
        except Exception as error:
            self.runtime.invalidate(str(error))
            raise

    def _retire_native(self) -> tuple[dict, bool]:
        if self._delivery is None:
            self._fail("native step called outside its due engine retirement")
        prepared, receipt = self._delivery
        scheduler_output = prepared.scheduler_output
        with (self.core.capture_iteration_details(scheduler_output) as iteration_details,
              self.core.log_error_detail(scheduler_output)):
            model_output = _publish_executor_step(prepared, self.runtime, receipt)
        self.core._process_aborts_queue()
        outputs = self.core.scheduler.update_from_output(scheduler_output, model_output)
        self.core._attach_iteration_details(outputs, iteration_details)
        return outputs, scheduler_output.total_num_scheduled_tokens > 0

    def close(self) -> None:
        if self.pending is not None:
            self._fail("pending native engine cannot close (CORE-69)")
        self._restore()

    def _restore(self) -> None:
        failures = []
        for owner, name, own, original in reversed(self._restorations):
            try:
                if own:
                    setattr(owner, name, original)
                elif name in vars(owner):
                    delattr(owner, name)
                else:
                    failures.append(name + " wrapper is missing")
            except Exception as error:  # noqa: BLE001, restore every binding before raising the collected failure.
                failures.append(name + ": " + str(error))
        self._restorations.clear()
        if failures:
            self._fail("native restoration failed: " + "; ".join(failures))
