"""Serial ``StepRecord`` to ``ExecutionGraph`` compatibility lowering.

The fallback schedule in this module deliberately matches
``HtsimStepSink`` and ``traffic.render_step_goal``.  It is not a claim about
the concurrency of a real framework.  Adapters that captured a real device
schedule pass framework-neutral :class:`ExecutionObservations` instead, and
those observations become the graph without reconstructing framework policy
from token counts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from simllm.compute import (
    GPU_ENVELOPES,
    ComputeProvider,
    GpuSpec,
    HostInitiationModel,
    ModelDims,
    RooflineProvider,
    step_kernel,
)
from simllm.core.execution import (
    CollectiveWork,
    ComputeWork,
    ExecutionGraph,
    ExecutionLowerer,
    ExecutionObservations,
    ExecutionOperation,
    OperationCorrelation,
)
from simllm.core.execution_io import (
    execution_graph_from_observations,
    validate_execution_graph,
)
from simllm.core.step import StepRecord
from simllm.traffic import (
    MOE_A2A_PHASES,
    TP_ALLREDUCE_SITES,
    RoutedMoeSupply,
    lower_step_observations,
    step_moe_alltoalls,
    step_tp_allreduces,
)


@dataclass(frozen=True)
class SerialStepLowererConfig:
    """Framework-neutral inputs for the serial compatibility schedule.

    ``tp_ranks`` and ``ep_ranks`` are semantic global ranks. A later
    backend mapping step may translate them to simulator endpoint ranks.
    ``routed_moe_supply`` optionally replaces the uniform MoE compatibility
    payload with captured ordered-pair sizes and the selected placement epoch.
    """

    dims: ModelDims
    tp_ranks: Sequence[int]
    ep_ranks: Sequence[int] | None = None
    provider: ComputeProvider = field(
        default_factory=lambda: RooflineProvider(efficiency=0.7)
    )
    gpu: GpuSpec = GPU_ENVELOPES["b100"]
    host_model: HostInitiationModel = field(default_factory=HostInitiationModel)
    routed_moe_supply: RoutedMoeSupply | None = None

    def __post_init__(self) -> None:
        tp_ranks = tuple(self.tp_ranks)
        ep_ranks = None if self.ep_ranks is None else tuple(self.ep_ranks)
        if not tp_ranks:
            raise ValueError("tp_ranks must contain at least one rank")
        if len(set(tp_ranks)) != len(tp_ranks):
            raise ValueError("tp_ranks must not contain duplicates")
        if any(rank < 0 for rank in tp_ranks):
            raise ValueError("tp_ranks must be non-negative")
        if ep_ranks is not None:
            if len(set(ep_ranks)) != len(ep_ranks):
                raise ValueError("ep_ranks must not contain duplicates")
            if any(rank < 0 for rank in ep_ranks):
                raise ValueError("ep_ranks must be non-negative")
        if self.dims.num_layers <= 0:
            raise ValueError("dims.num_layers must be positive")
        if self.routed_moe_supply is not None and not isinstance(
            self.routed_moe_supply, RoutedMoeSupply
        ):
            raise TypeError("routed_moe_supply must be RoutedMoeSupply or None")
        object.__setattr__(self, "tp_ranks", tp_ranks)
        object.__setattr__(self, "ep_ranks", ep_ranks)


def _execution_id(record: StepRecord) -> str:
    return f"step-{record.step_index}"


def _unique(values: Sequence[int] | Sequence[str]) -> tuple:
    return tuple(dict.fromkeys(values))


def _split_integer(total: int, count: int, index: int) -> int:
    quotient, remainder = divmod(total, count)
    return quotient + (1 if index < remainder else 0)


class SerialStepLowerer(ExecutionLowerer):
    """Lower a step to observed work or to the exact serial fallback.

    Supplying ``observations`` is an explicit choice, including when the
    observation tuple is empty.  With no observations, a non-drain record is
    rendered as one compute operation per participant and layer, followed by
    the same TP and EP collective sequence as the diagnostic GOAL path.
    """

    def __init__(self, config: SerialStepLowererConfig) -> None:
        self.config = config

    def lower(
        self,
        record: StepRecord,
        observations: ExecutionObservations | None = None,
    ) -> ExecutionGraph:
        execution_id = _execution_id(record)
        if observations is not None:
            return execution_graph_from_observations(
                record,
                observations,
                execution_id=execution_id,
            )

        if record.total_new_tokens <= 0:
            graph = ExecutionGraph(
                execution_id=execution_id,
                step_index=record.step_index,
                released_at_ps=record.virtual_time_ps,
            )
            validate_execution_graph(graph)
            return graph

        return self._lower_serial(record, execution_id)

    def _lower_serial(self, record: StepRecord, execution_id: str) -> ExecutionGraph:
        cfg = self.config
        tp_ops = step_tp_allreduces(record, cfg.dims, cfg.tp_ranks)
        moe_ops = step_moe_alltoalls(
            record,
            cfg.dims,
            cfg.ep_ranks if cfg.ep_ranks is not None else (),
            routed_supply=cfg.routed_moe_supply,
        )

        participants = list(cfg.tp_ranks)
        if moe_ops:
            for rank in moe_ops[0].ranks:
                if rank not in participants:
                    participants.append(rank)

        num_sampled = record.num_sampled
        if num_sampled is None:
            num_sampled = len(record.scheduled)
        kernel = step_kernel(cfg.dims, record, num_sampled=num_sampled)
        estimate = cfg.provider.estimate(kernel, cfg.gpu)
        whole_step_ps = estimate.duration_ps + cfg.host_model.delay_ps()
        per_layer_calc_ns = whole_step_ps // (cfg.dims.num_layers * 1000)
        per_layer_duration_ps = per_layer_calc_ns * 1000
        request_ids = tuple(request.request_id for request in record.scheduled)

        tp_by_key = {(op.layer, op.site): op for op in tp_ops}
        moe_by_key = {(op.layer, op.phase): op for op in moe_ops}
        operations: list[ExecutionOperation] = []
        tails: dict[int, str] = {}

        total_flops = int(kernel.flops)
        total_hbm_bytes = int(kernel.bytes_moved)
        for layer in range(cfg.dims.num_layers):
            correlation = OperationCorrelation(
                request_ids=request_ids,
                batch_id=execution_id,
                layer=layer,
            )
            for rank in participants:
                operation_id = f"{execution_id}:layer-{layer}:rank-{rank}:compute"
                local_depends_on = (tails[rank],) if rank in tails else ()
                operations.append(
                    ExecutionOperation(
                        operation_id=operation_id,
                        rank=rank,
                        logical_queue=f"cuda:{rank}:compute",
                        work=ComputeWork(
                            kernel=kernel.name,
                            config=kernel.config + (("layer", layer),),
                            flops=_split_integer(
                                total_flops, cfg.dims.num_layers, layer
                            ),
                            hbm_bytes=_split_integer(
                                total_hbm_bytes, cfg.dims.num_layers, layer
                            ),
                            nominal_duration_ps=per_layer_duration_ps,
                            uncertainty_fraction=estimate.uncertainty,
                        ),
                        correlation=correlation,
                        participant_local_depends_on=local_depends_on,
                    )
                )
                tails[rank] = operation_id

            for site in TP_ALLREDUCE_SITES:
                tp_op = tp_by_key.get((layer, site))
                if tp_op is None:
                    continue
                operation_id = f"{execution_id}:layer-{layer}:tp-{site}"
                dependencies = _unique(
                    tuple(tails[rank] for rank in tp_op.ranks)
                )
                operations.append(
                    ExecutionOperation(
                        operation_id=operation_id,
                        rank=tp_op.ranks[0],
                        logical_queue=f"cuda:{tp_op.ranks[0]}:nccl:tp",
                        work=CollectiveWork(
                            collective="all-reduce",
                            ranks=tp_op.ranks,
                            payload_bytes=tp_op.payload_bytes,
                            algorithm_hint="ring",
                            channel_hint=site,
                        ),
                        correlation=correlation,
                        participant_local_depends_on=dependencies,
                    )
                )
                for rank in tp_op.ranks:
                    tails[rank] = operation_id

            for phase in MOE_A2A_PHASES:
                moe_op = moe_by_key.get((layer, phase))
                if moe_op is None:
                    continue
                operation_id = f"{execution_id}:layer-{layer}:ep-{phase}"
                dependencies = _unique(
                    tuple(tails[rank] for rank in moe_op.ranks)
                )
                operations.append(
                    ExecutionOperation(
                        operation_id=operation_id,
                        rank=moe_op.ranks[0],
                        logical_queue=f"cuda:{moe_op.ranks[0]}:nccl:ep",
                        work=CollectiveWork(
                            collective="all-to-allv",
                            ranks=moe_op.ranks,
                            payload_bytes=moe_op.per_pair_bytes,
                            algorithm_hint="pairwise",
                            channel_hint=phase,
                            pair_payload_bytes=moe_op.pair_payload_bytes,
                        ),
                        correlation=correlation,
                        placement_epoch=moe_op.placement_epoch,
                        participant_local_depends_on=dependencies,
                    )
                )
                for rank in moe_op.ranks:
                    tails[rank] = operation_id

        completion_ids = _unique(tuple(tails[rank] for rank in participants))
        graph = ExecutionGraph(
            execution_id=execution_id,
            step_index=record.step_index,
            released_at_ps=record.virtual_time_ps,
            operations=tuple(operations),
            completion_operation_ids=completion_ids,
        )
        validate_execution_graph(graph)
        return graph


class ObservedStepLowerer(ExecutionLowerer):
    """Lower observed step streams, with the serial schedule as the exact off path.

    Supplying observations opts into traffic-side binding of semantic
    collectives while preserving the adapter's queues and dependency edges.
    Omitting observations delegates to :class:`SerialStepLowerer` so accepted
    serial graph and GOAL artifacts are unchanged.
    """

    def __init__(self, config: SerialStepLowererConfig) -> None:
        self.config = config
        self._serial = SerialStepLowerer(config)

    def lower(
        self,
        record: StepRecord,
        observations: ExecutionObservations | None = None,
    ) -> ExecutionGraph:
        if observations is None:
            return self._serial.lower(record)
        cfg = self.config
        return lower_step_observations(
            record,
            cfg.dims,
            cfg.tp_ranks,
            observations,
            ep_ranks=cfg.ep_ranks,
            routed_supply=cfg.routed_moe_supply,
        )
