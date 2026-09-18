"""Pinned vLLM dense PP declaration and deferred batch-queue completion.

The declaration follows v0.27.1 GPUWorker and GroupCoordinator: two eager
intermediate tensors, TP-sharded sends, then receiver TP all-gathers. The
all-gathers use a volume-equivalent pairwise expansion; they are not a
capture of RCCL's algorithm. The shared device runtime owns all service.
"""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import Any

from simllm.backends.pipeline_step_sink import PipelineRuntimeStepSink, PipelineStepOutcome
from simllm.backends.step_lowerer import SerialStepLowererConfig
from simllm.core import (
    CollectiveWork,
    ComputeWork,
    ExecutionGraph,
    ExecutionOperation,
    OperationCorrelation,
    StepRecord,
    StepResult,
)


def pipeline_layer_ranges(config: Any, width: int) -> tuple[tuple[int, int], ...]:
    """Use the pinned engine's partition, including its explicit split env."""
    from vllm.distributed.utils import get_pp_indices

    total = config.model_config.get_total_num_hidden_layers()
    ranges = tuple(get_pp_indices(total, index, width) for index in range(width))
    if (
        ranges[0][0] != 0 or ranges[-1][1] != total
        or any(start >= end for start, end in ranges)
        or any(left[1] != right[0] for left, right in pairwise(ranges))
    ):
        raise ValueError("PP requires a contiguous partition with at least one layer per stage")
    return ranges


def validate_pipeline_config(executor: Any) -> None:
    config = executor.vllm_config
    model = config.model_config
    parallel = config.parallel_config
    hf = getattr(model, "hf_text_config", None)
    architectures = getattr(hf, "architectures", None)
    if architectures is None:
        architectures = getattr(getattr(model, "hf_config", None), "architectures", ())
    if not architectures or not set(architectures) <= {
        "LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM",
    }:
        raise ValueError("PP declares only dense Llama/Qwen2/Qwen3 intermediate tensors")
    if executor.dims.num_experts or executor.dims.defaulted_fields:
        raise ValueError("PP requires explicit dense model geometry")
    if executor.world_size != executor.tp_size * executor.pp_size:
        raise ValueError("PP world must equal TP times PP")
    for name in ("data_parallel_size", "prefill_context_parallel_size",
                 "decode_context_parallel_size"):
        if getattr(parallel, name, 1) != 1:
            raise ValueError(f"PP does not support {name} > 1 (VLLM-8)")
    for name in ("enable_expert_parallel", "enable_dbo", "use_sequence_parallel_moe"):
        if getattr(parallel, name, False):
            raise ValueError(f"PP does not support {name} (VLLM-8)")
    if getattr(config.scheduler_config, "async_scheduling", False):
        raise ValueError("PP requires synchronous scheduling (VLLM-8)")
    if getattr(model, "runner_type", "generate") != "generate":
        raise ValueError("PP supports generation only (VLLM-8)")
    if not getattr(model, "enforce_eager", False):
        raise ValueError("PP requires enforce_eager=True for its declared tensor layout")
    passes = getattr(getattr(config, "compilation_config", None), "pass_config", None)
    if getattr(passes, "enable_sp", False):
        raise ValueError("PP does not declare sequence-parallel residual tensors (VLLM-8)")
    for name in ("speculative_config", "kv_transfer_config", "lora_config"):
        if getattr(config, name, None) is not None:
            raise ValueError(f"PP does not support {name} (VLLM-8)")
    if getattr(model, "is_multimodal_model", False):
        raise ValueError("PP requires a text-only model (VLLM-8)")


def dense_pipeline_boundaries(
    record: StepRecord,
    stages: tuple[tuple[int, ...], ...],
    hidden_size: int,
    dtype_bytes: int,
) -> tuple[ExecutionGraph, ...]:
    """Declare both intermediate tensors and the source's next-batch fence."""
    tensor_elements = record.total_new_tokens * hidden_size
    tensor_bytes = tensor_elements * dtype_bytes
    tp = len(stages[0])
    if tensor_bytes <= 0 or tensor_elements % tp:
        raise ValueError("PP eager intermediate tensors must divide evenly across TP")
    if any(len(stage) != tp for stage in stages):
        raise ValueError("PP stages must have equal tensor widths")
    shard = tensor_bytes // tp
    correlation = OperationCorrelation(
        request_ids=tuple(request.request_id for request in record.scheduled),
        batch_id=f"step-{record.step_index}",
    )
    graphs = []
    for boundary, (source, destination) in enumerate(pairwise(stages)):
        operations = []
        prior: tuple[str, ...] = ()
        for tensor in ("hidden_states", "residual"):
            name = f"send-{tensor}"
            operations.append(ExecutionOperation(
                name, source[0], f"pp-send-{boundary}",
                CollectiveWork(
                    "all-to-allv", source + destination, 0,
                    algorithm_hint="pairwise", channel_hint=f"pp-send-{boundary}",
                    pair_payload_bytes=tuple((src, dst, shard)
                                             for src, dst in zip(source, destination)),
                ),
                depends_on=prior, correlation=correlation,
            ))
            prior = (name,)
        # GPUWorker waits for the previous batch's send work before its next
        # execute_model. A zero-service operation reserves that causal fence
        # on the existing GPU calendar; it is not another timing authority.
        fences = []
        for rank in source:
            name = f"send-release-{rank}"
            fences.append(name)
            operations.append(ExecutionOperation(
                name, rank, f"pp-source-{rank}",
                ComputeWork("pp-send-release", nominal_duration_ps=0),
                participant_local_depends_on=prior, correlation=correlation,
            ))
        if tp > 1:
            for tensor in ("hidden_states", "residual"):
                name = f"receive-allgather-{tensor}"
                operations.append(ExecutionOperation(
                    name, destination[0], f"pp-receive-{boundary}",
                    CollectiveWork(
                        "all-to-allv", destination, 0,
                        algorithm_hint="pairwise", channel_hint=f"pp-receive-{boundary}",
                        pair_payload_bytes=tuple((src, dst, shard)
                                                 for src in destination
                                                 for dst in destination if src != dst),
                    ),
                    depends_on=prior, correlation=correlation,
                ))
                prior = (name,)
        else:
            prior = ()  # the source fence already consumes the final send
        graphs.append(ExecutionGraph(
            f"step-{record.step_index}:boundary-{boundary}", record.step_index,
            record.virtual_time_ps, tuple(operations), tuple(fences) + prior,
        ))
    return tuple(graphs)


class _PipelineFuture(Future):
    """Completion becomes visible only when the engine consumes this future."""

    def __init__(self, controller: PipelineController, ticket: _Ticket) -> None:
        super().__init__()
        self.controller = controller
        self.ticket = ticket
        self.set_running_or_notify_cancel()

    def result(self, timeout: float | None = None) -> Any:
        self._complete()
        if not self.done() and self.ticket.retired:
            # A sibling future's callback may consume us while retirement is
            # publishing both results. All ticket state is already committed.
            if self.ticket.error is not None:
                raise self.ticket.error
            return (None if self is self.ticket.execute_future and self.ticket.outcome is not None
                    else self.ticket.output)
        return super().result(timeout=timeout)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        self._complete()
        if not self.done() and self.ticket.retired:
            return self.ticket.error
        return super().exception(timeout=timeout)

    def _complete(self) -> None:
        if not self.done():
            try:
                self.controller.retire_through(self.ticket)
            except Exception as error:  # noqa: BLE001 - re-emitted by Future.result/exception
                self.controller.close(error)


@dataclass(eq=False)
class _Ticket:
    output: Any
    translated: Any = None
    outcome: PipelineStepOutcome | None = None
    execute_future: _PipelineFuture | None = None
    sample_future: _PipelineFuture | None = None
    retired: bool = False
    error: Exception | None = None
    futures: list[_PipelineFuture] = field(default_factory=list)


class PipelineController:
    """Bounded execute/sample FIFO over one runtime and one adapter clock."""

    def __init__(self, executor: Any) -> None:
        validate_pipeline_config(executor)
        self.executor = executor
        sink = executor.step_sink
        if sink is None:
            sink = PipelineRuntimeStepSink(host_model=executor.host_model)
        if not isinstance(sink, PipelineRuntimeStepSink):
            raise TypeError("PP requires PipelineRuntimeStepSink, not a single-stage sink")
        self.sink = sink
        ranks = tuple(range(executor.world_size)) if sink.rank_map is None else sink.rank_map
        if len(ranks) != executor.world_size:
            raise ValueError("rank_map must contain exactly TP times PP endpoints")
        tp = executor.tp_size
        self.stages = tuple(tuple(ranks[index * tp:(index + 1) * tp])
                            for index in range(executor.pp_size))
        self.layer_ranges = pipeline_layer_ranges(executor.vllm_config, executor.pp_size)
        configs = tuple(SerialStepLowererConfig(
            dims=replace(executor.dims, num_layers=end - start,
                         vocab_size=executor.dims.vocab_size
                         if index == executor.pp_size - 1 else 0),
            tp_ranks=self.stages[index], provider=executor.compute_provider,
            gpu=executor.gpu, host_model=executor.host_model,
        ) for index, (start, end) in enumerate(self.layer_ranges))
        sink.bind_pipeline(configs, self.layer_ranges, self._boundaries)
        self.pending: deque[_Ticket] = deque()
        self.unsampled: deque[_Ticket] = deque()
        self.closed = False

    def _boundaries(self, record: StepRecord) -> tuple[ExecutionGraph, ...]:
        dims = self.executor.dims
        return dense_pipeline_boundaries(record, self.stages, dims.hidden_size, dims.dtype_bytes)

    def execute(self, scheduler_output: Any, non_block: bool) -> Any:
        if self.closed:
            raise RuntimeError("pipeline executor is closed")
        if max(len(self.pending), len(self.unsampled)) >= self.executor.pp_size:
            raise RuntimeError("pipeline in-flight batch capacity exceeded")
        if getattr(scheduler_output, "has_structured_output_requests", False):
            raise RuntimeError("PP does not support structured output (VLLM-8)")
        try:
            return self._prepare(scheduler_output, non_block)
        except Exception as error:
            # Translation and replay have mutable request state. A failed
            # submission poisons this controller instead of reusing that state
            # to fabricate a later successful result.
            self.close(error)
            raise

    def _prepare(self, scheduler_output: Any, non_block: bool) -> Any:
        has_work = bool(getattr(scheduler_output, "num_scheduled_tokens", None) or {})
        runtime = self.executor._runtime
        ticket = _Ticket(output=self.executor._empty_output())
        if has_work:
            translated = runtime.translate(scheduler_output)
            ticket.translated = translated
            if self.executor.replay is not None:
                self.executor.replay.validate_step(
                    translated.req_ids, translated.produces_token, scheduler_output,
                )
            req_ids, indexes, sampled = self.executor._sample_output_fields(
                translated, scheduler_output,
            )
            from vllm.v1.outputs import ModelRunnerOutput

            ticket.output = ModelRunnerOutput(
                req_ids=req_ids, req_id_to_index=indexes, sampled_token_ids=sampled,
            )
            ticket.outcome = self.sink.prepare(translated.record)
            ticket.sample_future = _PipelineFuture(self, ticket)
            ticket.futures.append(ticket.sample_future)
            self.unsampled.append(ticket)
        elif (getattr(scheduler_output, "finished_req_ids", None)
              or getattr(scheduler_output, "preempted_req_ids", None)):
            ticket.translated = runtime.translate(scheduler_output)
        ticket.execute_future = _PipelineFuture(self, ticket)
        ticket.futures.append(ticket.execute_future)
        self.pending.append(ticket)
        return ticket.execute_future if non_block else ticket.execute_future.result()

    def sample(self, non_block: bool) -> Any:
        if not self.unsampled:
            raise RuntimeError("sample_tokens called with no pending pipeline output")
        ticket = self.unsampled.popleft()
        assert ticket.sample_future is not None
        return ticket.sample_future if non_block else ticket.sample_future.result()

    def retire_through(self, target: _Ticket) -> None:
        if target.retired:
            return
        if target not in self.pending:
            raise RuntimeError("unknown pipeline completion")
        runtime = self.executor._runtime
        while self.pending:
            ticket = self.pending[0]
            previous = runtime.clock.now_ps
            if ticket.outcome is not None:
                result = ticket.outcome.step_result
            else:
                result = None if ticket.translated is None else StepResult(
                    ticket.translated.record.step_index, 0, previous,
                )
            if result is not None:
                record = ticket.translated.record
                runtime.clock.advance_to(max(previous, result.completed_at_ps))
                runtime.step_records.append(record)
                runtime.step_results.append(result)
                if runtime.record_stream is not None:
                    runtime.record_stream.append(record)
                if runtime.config.mode == "paced" and runtime.clock.now_ps > previous:
                    time.sleep((runtime.clock.now_ps - previous) / 1_000_000_000_000)
            self.pending.popleft()
            ticket.retired = True
            assert ticket.execute_future is not None
            ticket.execute_future.set_result(None if ticket.outcome is not None else ticket.output)
            if ticket.sample_future is not None:
                ticket.sample_future.set_result(ticket.output)
            if ticket is target:
                return

    def close(self, error: Exception | None = None) -> None:
        self.closed = True
        error = error or RuntimeError("pipeline executor shut down before completion")
        tickets = tuple(self.pending)
        self.pending.clear()
        self.unsampled.clear()
        for ticket in tickets:
            ticket.retired = True
            ticket.error = error
        for ticket in tickets:
            for future in ticket.futures:
                if not future.done():
                    future.set_exception(error)
