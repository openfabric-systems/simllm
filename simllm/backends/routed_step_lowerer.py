"""Causal single-engine expert routing through the standard execution graph."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from simllm.compute import KernelSpec, grouped_expert_kernels, step_kernels
from simllm.core import (
    CollectiveWork,
    ComputeWork,
    ExecutionGraph,
    ExecutionOperation,
    OperationCorrelation,
    StepRecord,
)
from simllm.core.execution_io import validate_execution_graph
from simllm.traffic import step_routed_moe_work

if TYPE_CHECKING:
    from simllm.backends.step_lowerer import SerialStepLowererConfig


def _integer(value: float, name: str) -> int:
    integer = int(value)
    if value != integer or integer < 0:
        raise ValueError(f"{name} must be a nonnegative whole quantity")
    return integer


def _split(total: int, count: int, index: int) -> int:
    quotient, remainder = divmod(total, count)
    return quotient + (index < remainder)


def lower_routed_step(
    record: StepRecord, config: SerialStepLowererConfig
) -> ExecutionGraph:
    """Build attention, dispatch, owner compute, combine and output stages."""

    mode = config.routed_compute
    supply = config.routed_moe_supply
    assert mode is not None and supply is not None and config.ep_ranks is not None
    mode.validate_gpu(config.gpu)
    if (
        record.num_tokens_after_padding is not None
        and record.num_tokens_after_padding != record.total_new_tokens
    ):
        raise ValueError("routed compute requires exact unpadded token rows")
    execution_id = f"step-{record.step_index}"
    if record.total_new_tokens <= 0:
        graph = ExecutionGraph(
            execution_id=execution_id, step_index=record.step_index,
            released_at_ps=record.virtual_time_ps,
        )
        validate_execution_graph(graph)
        return graph
    layers = step_routed_moe_work(record, config.dims, config.ep_ranks, supply)
    if len(layers) != config.dims.num_layers:
        raise ValueError("routed compute must cover every declared MoE layer")
    requests = tuple(request.request_id for request in record.scheduled)
    sampled = record.num_sampled
    if sampled is None:
        sampled = len(record.scheduled)
    families = {
        kernel.name: kernel for kernel in step_kernels(config.dims, record, sampled)
    }
    before_families = [families[name] for name in ("attn_gemm", "attn_score", "kv_read")]
    before_flops = _integer(sum(k.flops for k in before_families), "attention FLOPs")
    before_bytes = _integer(sum(k.bytes_moved for k in before_families), "attention bytes")
    operations = []

    def correlation(layer, request_ids=requests):
        return OperationCorrelation(
            request_ids=request_ids, batch_id=execution_id, layer=layer,
        )

    def compute(operation_id, rank, kernel, dependencies, layer, epoch, request_ids, expert=False):
        estimate = (
            mode.estimate(config.provider, kernel, config.gpu)
            if expert else config.provider.estimate(kernel, config.gpu)
        )
        if type(estimate.duration_ps) is not int or estimate.duration_ps < 0:
            raise ValueError("provider must return nonnegative integer service")
        extra_config = (
            ("routed_compute", 1),
            ("declared_gemm_minimum_ps", mode.gemm_minimum_ps if expert else 0),
        )
        operations.append(ExecutionOperation(
            operation_id=operation_id,
            rank=rank,
            logical_queue=f"cuda:{rank}:compute",
            work=ComputeWork(
                kernel=kernel.name, config=kernel.config + extra_config,
                flops=_integer(kernel.flops, "kernel FLOPs"),
                hbm_bytes=_integer(kernel.bytes_moved, "kernel weight/attention bytes"),
                nominal_duration_ps=estimate.duration_ps,
                uncertainty_fraction=estimate.uncertainty,
            ),
            depends_on=dependencies,
            correlation=correlation(layer, request_ids),
            placement_epoch=epoch,
        ))
        return operation_id

    def collective(work, dependencies, layer, epoch):
        if not work.pair_payload_bytes:
            return dependencies
        operation_id = f"{execution_id}:layer-{layer}:ep-{work.phase}"
        operations.append(ExecutionOperation(
            operation_id=operation_id,
            rank=supply.engine_rank,
            logical_queue=f"cuda:{supply.engine_rank}:nccl:ep",
            work=CollectiveWork(
                collective="all-to-allv", ranks=work.ranks, payload_bytes=0,
                algorithm_hint="pairwise", channel_hint=work.phase,
                pair_payload_bytes=work.pair_payload_bytes,
                request_pair_payload_bytes=work.request_pair_payload_bytes,
            ),
            depends_on=dependencies,
            correlation=correlation(layer),
            placement_epoch=epoch,
        ))
        return (operation_id,)

    tails = ()
    for layer in layers:
        assignments = layer.expert_assignments
        expected_rows = record.total_new_tokens * config.dims.top_k
        if len(assignments) != expected_rows:
            raise ValueError("routed expert rows do not conserve selected top-k work")
        positions = [
            (row.request_id, row.token_index, row.top_k_index) for row in assignments
        ]
        if len(set(positions)) != len(positions):
            raise ValueError("routed expert work duplicates an assignment")
        counts = Counter(row.request_id for row in assignments)
        expected = {
            row.request_id: row.num_new_tokens * config.dims.top_k
            for row in record.scheduled if row.num_new_tokens
        }
        if dict(counts) != expected:
            raise ValueError("routed expert work changes request membership")
        before = KernelSpec(
            "moe_pre_dispatch",
            _split(before_flops, config.dims.num_layers, layer.layer),
            _split(before_bytes, config.dims.num_layers, layer.layer),
            (("layer", layer.layer), ("new_tokens", record.total_new_tokens)),
        )
        before_id = compute(
            f"{execution_id}:layer-{layer.layer}:pre-dispatch",
            supply.engine_rank, before, tails, layer.layer,
            layer.placement_epoch, requests,
        )
        dispatch = collective(
            layer.dispatch, (before_id,), layer.layer, layer.placement_epoch,
        )
        expert_tails = []
        for rank in layer.ranks:
            histogram = layer.expert_rows(rank)
            if not histogram:
                continue
            owner_requests = tuple(dict.fromkeys(
                row.request_id for row in assignments if row.owner_rank == rank
            ))
            kernels = grouped_expert_kernels(
                config.dims, layer=layer.layer, rank=rank, expert_rows=histogram,
            )
            owner_tail = dispatch
            for kernel in kernels:
                operation_id = f"{execution_id}:layer-{layer.layer}:rank-{rank}:{kernel.name}"
                owner_tail = (compute(
                    operation_id, rank, kernel, owner_tail, layer.layer,
                    layer.placement_epoch, owner_requests, expert=True,
                ),)
            expert_tails.extend(owner_tail)
        tails = collective(
            layer.combine, tuple(expert_tails), layer.layer, layer.placement_epoch,
        )
    if sampled:
        output_id = compute(
            f"{execution_id}:output-head", supply.engine_rank, families["lm_head"],
            tails, None, layers[-1].placement_epoch, requests,
        )
        tails = (output_id,)
    graph = ExecutionGraph(
        execution_id=execution_id,
        step_index=record.step_index,
        released_at_ps=record.virtual_time_ps,
        operations=tuple(operations),
        completion_operation_ids=tails,
    )
    validate_execution_graph(graph)
    return graph
