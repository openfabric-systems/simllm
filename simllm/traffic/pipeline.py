"""Declared forward pipeline activations composed with existing stage graphs.

The caller supplies stage membership and each stage's existing lowerer. This
keeps compute and intra-stage collectives under their original authority.
TRAF-8 owns captured stage attribution and microbatch scheduling beyond this
single-forward-chain declaration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from itertools import pairwise

from simllm.compute import ModelDims
from simllm.core import (
    CollectiveWork,
    ExecutionGraph,
    ExecutionObservations,
    ExecutionOperation,
    OperationCorrelation,
    StepRecord,
)
from simllm.core.execution import ExecutionLowerer
from simllm.core.execution_io import (
    effective_dependency_edges,
    operation_participant_ranks,
    validate_execution_graph,
)
from simllm.traffic.collective_plan import plan_execution_graph_collectives


def pipeline_stage_membership(
    pipeline_width: int,
    pp_stages: Sequence[int | Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Validate ordered stages; a flat PP group means one rank per stage.

    A nested sequence explicitly declares the ranks executing each stage.
    Its last rank sends to the next stage's first rank. No placement or
    tensor-parallel membership is inferred from arithmetic on rank numbers.
    """
    if type(pipeline_width) is not int:
        raise TypeError("pipeline_width must be an integer")
    if pipeline_width < 1:
        raise ValueError("pipeline_width must be positive")
    stages = tuple(
        (stage,) if type(stage) is int else tuple(stage)
        for stage in pp_stages
    )
    if len(stages) != pipeline_width:
        raise ValueError("PP membership must contain exactly pipeline_width stages")
    if any(not stage for stage in stages):
        raise ValueError("every PP stage must contain at least one rank")
    ranks = tuple(rank for stage in stages for rank in stage)
    if any(type(rank) is not int or rank < 0 for rank in ranks):
        raise ValueError("PP ranks must be nonnegative integers")
    if len(set(ranks)) != len(ranks):
        raise ValueError("PP stage ranks must be distinct, including across stages")
    return stages


def step_pp_activations(
    record: StepRecord,
    dims: ModelDims,
    pipeline_width: int,
    pp_stages: Sequence[int | Sequence[int]],
) -> tuple[CollectiveWork, ...]:
    """Return one sparse forward transfer per boundary in a declared step.

    Each transfer uses the existing pairwise vocabulary with exactly one
    directed pair. Payload is new tokens times hidden width times dtype bytes,
    without tensor-width division or an implicit backward transfer. Request
    partitions preserve the identity of every byte in the activation.
    """
    if not isinstance(record, StepRecord):
        raise TypeError("record must be a StepRecord")
    if not isinstance(dims, ModelDims):
        raise TypeError("dims must be ModelDims")
    stages = pipeline_stage_membership(pipeline_width, pp_stages)
    for name in ("hidden_size", "dtype_bytes"):
        value = getattr(dims, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"dims.{name} must be a positive integer")
    request_ids = [request.request_id for request in record.scheduled]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("scheduled request IDs must be unique")
    if any(
        type(request.num_new_tokens) is not int or request.num_new_tokens < 0
        for request in record.scheduled
    ):
        raise ValueError("new token counts must be nonnegative integers")
    if pipeline_width == 1 or record.total_new_tokens == 0:
        return ()
    vector_bytes = dims.hidden_size * dims.dtype_bytes
    payload_bytes = record.total_new_tokens * vector_bytes
    return tuple(
        CollectiveWork(
            collective="all-to-allv",
            ranks=(source[-1], destination[0]),
            payload_bytes=0,
            algorithm_hint="pairwise",
            channel_hint=f"pp-forward-{boundary}",
            pair_payload_bytes=((source[-1], destination[0], payload_bytes),),
            request_pair_payload_bytes=tuple(sorted(
                (request.request_id, source[-1], destination[0],
                 request.num_new_tokens * vector_bytes)
                for request in record.scheduled if request.num_new_tokens
            )),
        )
        for boundary, (source, destination) in enumerate(pairwise(stages))
    )


def _stage_terminals(graph: ExecutionGraph) -> tuple[str, ...]:
    """Require a full stage frontier before making its activation visible."""
    consumed = {edge.predecessor_id for edge in effective_dependency_edges(graph)}
    terminals = tuple(
        operation.operation_id for operation in graph.operations
        if operation.operation_id not in consumed
    )
    if graph.completion_operation_ids and set(graph.completion_operation_ids) != set(terminals):
        raise ValueError("PP stages require a complete terminal frontier, without background work")
    return terminals


def compose_pipeline_graph(
    record: StepRecord,
    dims: ModelDims,
    pipeline_width: int,
    pp_stages: Sequence[int | Sequence[int]],
    stage_graphs: Sequence[ExecutionGraph],
) -> ExecutionGraph:
    """Join complete stage graphs with causal forward activation transfers.

    Width one returns the original graph object, including its existing wire
    and collective-plan bytes. For a singleton stage, rank-local dependencies
    encode the activation chain directly in one GOAL artifact. Multi-rank
    stage barriers keep whole-operation semantics; the existing projection
    may split these at process boundaries rather than weaken their ordering.
    """
    stages = pipeline_stage_membership(pipeline_width, pp_stages)
    activations = step_pp_activations(record, dims, pipeline_width, stages)
    graphs = tuple(stage_graphs)
    if len(graphs) != pipeline_width:
        raise ValueError("one execution graph is required per PP stage")
    for stage, graph in zip(stages, graphs):
        validate_execution_graph(graph)
        if graph.step_index != record.step_index or graph.released_at_ps != record.virtual_time_ps:
            raise ValueError("stage graph step and release must agree with the StepRecord")
        participants = {
            rank for operation in graph.operations
            for rank in operation_participant_ranks(operation)
        }
        if not participants <= set(stage):
            raise ValueError("stage graph contains ranks outside its declared PP stage")
        if record.total_new_tokens and participants != set(stage):
            raise ValueError("every declared stage rank must participate in its stage graph")
        if pipeline_width > 1:
            _stage_terminals(graph)
    if pipeline_width == 1:
        return graphs[0]
    for graph in graphs:
        if graph.collective_plans and graph.collective_plans != (
            plan_execution_graph_collectives(replace(graph, collective_plans=())).collective_plans
        ):
            raise ValueError("PP composition requires canonical traffic-owned stage plans")
    if not activations:
        if any(graph.operations for graph in graphs):
            raise ValueError("a zero-token pipeline must have empty stage graphs")
        return graphs[0]

    operations: list[ExecutionOperation] = []
    completion_ids: tuple[str, ...] = ()
    transfer_ids: list[str] = []
    incoming: str | None = None
    execution_id = graphs[0].execution_id
    for stage_index, (stage, graph) in enumerate(zip(stages, graphs)):
        ids = {
            operation.operation_id: f"{execution_id}:pp-stage-{stage_index}:{operation.operation_id}"
            for operation in graph.operations
        }
        targets = {edge.operation_id for edge in effective_dependency_edges(graph)}
        for operation in graph.operations:
            whole = tuple(ids[dependency] for dependency in operation.depends_on)
            local = tuple(ids[dependency] for dependency in operation.participant_local_depends_on)
            if incoming is not None and operation.operation_id not in targets:
                if len(stage) == 1:
                    local += (incoming,)
                else:
                    whole += (incoming,)
            operations.append(replace(
                operation,
                operation_id=ids[operation.operation_id],
                logical_queue=f"pp-stage-{stage_index}:{operation.logical_queue}",
                depends_on=whole,
                participant_local_depends_on=local,
            ))
        completion_ids = tuple(ids[name] for name in _stage_terminals(graph))
        if stage_index < len(activations):
            incoming = f"{execution_id}:pp-forward-{stage_index}"
            transfer_ids.append(incoming)
            operations.append(ExecutionOperation(
                operation_id=incoming,
                rank=stage[-1],
                logical_queue=f"pp:{stage[-1]}:forward",
                work=activations[stage_index],
                depends_on=completion_ids if len(stage) > 1 else (),
                participant_local_depends_on=completion_ids if len(stage) == 1 else (),
                correlation=OperationCorrelation(
                    request_ids=tuple(request.request_id for request in record.scheduled),
                    batch_id=execution_id,
                ),
            ))
    graph = ExecutionGraph(
        execution_id=execution_id,
        step_index=record.step_index,
        released_at_ps=record.virtual_time_ps,
        operations=tuple(operations),
        completion_operation_ids=(*transfer_ids, *completion_ids),
    )
    validate_execution_graph(graph)
    return plan_execution_graph_collectives(graph)


class PipelineStepLowerer:
    """Opt-in ExecutionLowerer composition over caller-owned stage lowerers.

    Existing lowerers choose their own compute and TP/EP traffic. This wrapper
    adds only stage namespaces, causal gates and forward activations. Width
    one delegates directly, including observations, as an exact identity.
    Multi-stage observations must be lowered by stage-specific producers;
    guessing stage assignment from an unpartitioned observation is refused.
    """

    def __init__(
        self,
        dims: ModelDims,
        pipeline_width: int,
        pp_stages: Sequence[int | Sequence[int]],
        stage_lowerers: Sequence[ExecutionLowerer],
    ) -> None:
        self.stages = pipeline_stage_membership(pipeline_width, pp_stages)
        if not isinstance(dims, ModelDims):
            raise TypeError("dims must be ModelDims")
        self.dims = dims
        self.pipeline_width = pipeline_width
        self.stage_lowerers = tuple(stage_lowerers)
        if len(self.stage_lowerers) != pipeline_width:
            raise ValueError("one lowerer is required per PP stage")
        if any(not isinstance(lowerer, ExecutionLowerer) for lowerer in self.stage_lowerers):
            raise TypeError("stage lowerers must implement ExecutionLowerer")

    def lower(
        self,
        record: StepRecord,
        observations: ExecutionObservations | None = None,
    ) -> ExecutionGraph:
        if self.pipeline_width == 1:
            return self.stage_lowerers[0].lower(record, observations)
        if observations is not None:
            raise ValueError("multi-stage PP needs explicitly partitioned stage observations")
        graphs = tuple(lowerer.lower(record) for lowerer in self.stage_lowerers)
        return compose_pipeline_graph(
            record, self.dims, self.pipeline_width, self.stages, graphs,
        )
