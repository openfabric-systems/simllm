"""Per-step collective work (TP allreduces, MoE all-to-alls) from a step record.

Maps one :class:`~simllm.core.StepRecord` plus a per-rank
:class:`~simllm.compute.ModelDims` plus GOAL rank groups to the collective
traffic of that engine step.

Tensor parallel (:func:`step_tp_allreduces`): per transformer layer, the ring
allreduces :func:`layer_tp_allreduce_sites` reports for this model. A dense
layer has two (the attention output projection and the MLP output
projection); a routed layer whose experts are expert-parallel has only the
attention one, because its combine all-to-all already returns finished expert
vectors and no partial sum spans the TP group. Each allreduce moves

    payload_bytes = record.total_new_tokens * dims.hidden_size * dims.dtype_bytes

which is the activation tensor of every token computed this step. A TP
world of size 1 (or a drain record with zero new tokens) produces no
operations.

Expert parallel (:func:`step_moe_alltoalls`): per MoE layer, a dispatch
pairwise all-to-allv (tokens to their experts' owner ranks) followed by a
combine pairwise all-to-allv (expert outputs back). An optional captured
routing supply selects per-token destinations at an explicit placement epoch
and declares the one engine rank that owns this record's tokens. Its absence
uses the first EP rank as the documented single-engine source for a uniform
destination approximation.

This is a deliberately first-order model of step traffic; each
simplification is a numbered task in docs/modules/traffic.md:

- no sequence parallelism (which would replace each allreduce with a
  reduce-scatter plus allgather of 1/W the bytes framed around the
  norm/dropout regions): TRAF-6;
- `render_step_goal` retains the strict serial compatibility schedule;
  `lower_step_observations` instead preserves adapter-observed queues and
  dependency edges; VLLM-22 supplies the accepted Granite MoE producer;
- no pipeline-parallel activation traffic (records carry no PP stage
  information yet): TRAF-8;
- the MoE layer is rendered as one calc, then the TP allreduces, then
  dispatch and combine back to back, instead of splitting the layer's
  compute around the all-to-alls: TRAF-9.

Rendering reuses :func:`simllm.traffic.patterns.ring_allreduce` and
:func:`simllm.traffic.patterns.pairwise_all_to_allv` unchanged, so the GOAL
structures are exactly the patterns validated by the M1/M4/M5 studies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum

from simllm.compute import ModelDims
from simllm.core import (
    CollectiveWork,
    ComputeWork,
    ExecutionGraph,
    ExecutionObservations,
    ExecutionOperation,
    RequestPhase,
    StepRecord,
    collective_goal_tags,
    execution_graph_from_observations,
    validate_execution_graph,
)
from simllm.core.execution_io import effective_dependency_edges
from simllm.goal import GoalMessage, GoalTrace
from simllm.placement import RankMapper
from simllm.preplay.routing import RoutedToken
from simllm.traffic.collective_plan import (
    collective_plan_by_operation,
    plan_execution_graph_collectives,
)
from simllm.traffic.locality import (
    DEFAULT_NVLINK_BANDWIDTH_BYTES_PER_SECOND,
    ClassifiedCommunicationPhase,
    CollectiveCommunicationPhase,
    DirectedCollectiveSegment,
    StepLocalityPlan,
    classify_step_locality,
)
from simllm.traffic.patterns import (
    ordered_pairwise_messages,
    pairwise_all_to_allv,
    ring_allreduce,
)
from simllm.traffic.request_fidelity import (
    AggregatePairRow,
    RequestFidelityReport,
    RequestPairRow,
    compare_goal_request_attribution,
)
from simllm.traffic.routed_conservation import (
    ROUTED_EVIDENCE_CAPTURED,
    ROUTED_EVIDENCE_UNIFORM,
    RoutedConservationReport,
    RoutedPhaseTable,
    RoutedTokenOwnership,
    routed_moe_conservation_report,
)
from simllm.traffic.routed_moe import RoutedMoeSupply

#: the two allreduce sites of one transformer layer, in execution order
TP_ALLREDUCE_SITES = ("attention", "mlp")

#: the sites a layer executes when its routed experts are expert-parallel
EXPERT_PARALLEL_TP_ALLREDUCE_SITES = ("attention",)

#: the two all-to-allv phases of one MoE layer, in execution order
MOE_A2A_PHASES = ("dispatch", "combine")


def layer_tp_allreduce_sites(dims: ModelDims) -> tuple[str, ...]:
    """The tensor-parallel allreduce sites one layer of ``dims`` executes.

    A dense layer reduces twice, after the attention output projection and
    after the MLP down projection, because both are row-parallel and every
    rank holds a partial sum. A routed layer whose experts are expert-parallel
    reduces once. Under expert parallelism the expert weights are not
    tensor-sharded (``moe_tp`` is 1), so each expert's down projection is
    computed whole on its owner rank, the combine all-to-all returns finished
    expert vectors, and the token's home rank forms the layer output by a
    local weighted sum. No partial sum spans the tensor-parallel group, so
    there is no mlp-site allreduce to render.

    A routed layer whose experts are all resident keeps both sites: with no
    expert parallelism the experts are tensor-sharded over the whole
    tensor-parallel group instead, and the reduction after the expert down
    projection is real. That configuration also renders no all-to-all, so
    dropping the site would lose the layer's only MLP collective.

    ``ModelDims`` carries one whole-model mixture geometry. ``num_experts > 0``
    means every layer's MLP is routed, and ``resident_experts < num_experts``
    means this rank holds a strict subset, which is what expert parallelism
    is. Those two fields are the whole input: this rule never reads a
    tensor-parallel or expert-parallel group width. Mixed dense and routed
    layer schedules are not expressible (TRAF-34), and an expert-parallel
    group that leaves ``moe_tp`` above 1 is not expressible either (TRAF-35).
    """

    if not isinstance(dims, ModelDims):
        raise TypeError("dims must be ModelDims")
    if dims.num_experts > 0 and dims.resident_experts < dims.num_experts:
        return EXPERT_PARALLEL_TP_ALLREDUCE_SITES
    return TP_ALLREDUCE_SITES


def _planned_collective_work(
    record: StepRecord,
    dims: ModelDims,
    tp_ranks: Sequence[int],
    ep_ranks: Sequence[int] | None,
    routed_supply: RoutedMoeSupply | None,
) -> dict[tuple[str, int, str], tuple[CollectiveWork, int]]:
    """Return traffic-owned collective work indexed by semantic call site."""

    planned: dict[tuple[str, int, str], tuple[CollectiveWork, int]] = {}
    for operation in step_tp_allreduces(record, dims, tp_ranks):
        key = ("tp", operation.layer, operation.site)
        planned[key] = (
            CollectiveWork(
                collective="all-reduce",
                ranks=operation.ranks,
                payload_bytes=operation.payload_bytes,
                algorithm_hint="ring",
                channel_hint=operation.site,
            ),
            0,
        )
    for operation in step_moe_alltoalls(
        record,
        dims,
        ep_ranks if ep_ranks is not None else (),
        routed_supply=routed_supply,
    ):
        key = ("moe", operation.layer, operation.phase)
        planned[key] = (
            CollectiveWork(
                collective="all-to-allv",
                ranks=operation.ranks,
                payload_bytes=operation.per_pair_bytes,
                algorithm_hint="pairwise",
                channel_hint=operation.phase,
                pair_payload_bytes=operation.pair_payload_bytes,
                request_pair_payload_bytes=operation.request_pair_payload_bytes,
            ),
            operation.placement_epoch,
        )
    return planned


def _observed_collective_key(
    operation: ExecutionOperation,
    index: int,
) -> tuple[str, int, str]:
    work = operation.work
    assert isinstance(work, CollectiveWork)
    layer = operation.correlation.layer
    if layer is None:
        raise ValueError(
            f"observations.operations[{index}]: collective needs correlation.layer"
        )
    site = work.channel_hint
    if site is None:
        raise ValueError(
            f"observations.operations[{index}]: collective needs a semantic "
            "channel_hint site"
        )
    if work.collective == "all-reduce":
        return ("tp", layer, site)
    if work.collective == "all-to-allv":
        return ("moe", layer, site)
    raise ValueError(
        f"observations.operations[{index}]: unsupported step collective "
        f"{work.collective!r}"
    )


def _validate_observed_collective(
    observed: ExecutionOperation,
    planned: CollectiveWork,
    placement_epoch: int,
    index: int,
) -> None:
    work = observed.work
    assert isinstance(work, CollectiveWork)
    path = f"observations.operations[{index}]"
    if work.ranks != planned.ranks:
        raise ValueError(f"{path}: collective ranks disagree with the step plan")
    if observed.rank not in planned.ranks:
        raise ValueError(f"{path}: collective anchor rank is outside the step plan")
    semantic_pairwise_marker = (
        work.collective == "all-to-allv"
        and work.payload_bytes == 0
        and not work.pair_payload_bytes
    )
    if work.payload_bytes != planned.payload_bytes and not semantic_pairwise_marker:
        raise ValueError(f"{path}: collective payload disagrees with the step plan")
    if work.pair_payload_bytes not in ((), planned.pair_payload_bytes):
        raise ValueError(f"{path}: collective pair payloads disagree with the step plan")
    if work.request_pair_payload_bytes not in ((), planned.request_pair_payload_bytes):
        raise ValueError(
            f"{path}: collective request attribution disagrees with the step plan"
        )
    if work.algorithm_hint not in (None, planned.algorithm_hint):
        raise ValueError(f"{path}: collective algorithm disagrees with the traffic plan")
    if observed.placement_epoch not in (0, placement_epoch):
        raise ValueError(f"{path}: placement epoch disagrees with the traffic plan")


def _observed_microbatch_records(
    record: StepRecord,
    observations: ExecutionObservations,
) -> dict[int | None, StepRecord]:
    """Return the request partition declared by collective correlations."""

    request_ids_by_microbatch: dict[int, tuple[str, ...]] = {}
    saw_unbatched_collective = False
    for index, operation in enumerate(observations.operations):
        if not isinstance(operation.work, CollectiveWork):
            continue
        microbatch = operation.correlation.microbatch
        if microbatch is None:
            saw_unbatched_collective = True
            continue
        if isinstance(microbatch, bool) or not isinstance(microbatch, int):
            raise TypeError(
                f"observations.operations[{index}]: microbatch must be an integer"
            )
        if microbatch < 0:
            raise ValueError(
                f"observations.operations[{index}]: microbatch must be nonnegative"
            )
        request_ids = operation.correlation.request_ids
        if not request_ids:
            raise ValueError(
                f"observations.operations[{index}]: microbatch collective needs "
                "request correlation"
            )
        previous = request_ids_by_microbatch.setdefault(microbatch, request_ids)
        if previous != request_ids:
            raise ValueError(
                f"observations.operations[{index}]: microbatch {microbatch} "
                "request correlation is inconsistent"
            )

    if not request_ids_by_microbatch:
        return {None: record}
    if saw_unbatched_collective:
        raise ValueError("observations mix batched and unbatched collective sites")
    indices = tuple(sorted(request_ids_by_microbatch))
    if indices != tuple(range(len(indices))):
        raise ValueError("observations microbatch indices must be contiguous from zero")

    scheduled_by_id = {
        request.request_id: request for request in record.scheduled
    }
    if len(scheduled_by_id) != len(record.scheduled):
        raise ValueError("record.scheduled contains duplicate request identities")
    flattened = tuple(
        request_id
        for microbatch in indices
        for request_id in request_ids_by_microbatch[microbatch]
    )
    expected = tuple(request.request_id for request in record.scheduled)
    if flattened != expected:
        raise ValueError(
            "observed microbatches must partition scheduled requests in source order"
        )

    records: dict[int | None, StepRecord] = {}
    for microbatch in indices:
        request_ids = request_ids_by_microbatch[microbatch]
        records[microbatch] = StepRecord(
            step_index=record.step_index,
            virtual_time_ps=record.virtual_time_ps,
            scheduled=[scheduled_by_id[request_id] for request_id in request_ids],
        )
    return records


def _planned_collective_instances(
    record: StepRecord,
    dims: ModelDims,
    tp_ranks: Sequence[int],
    ep_ranks: Sequence[int] | None,
    routed_supply: RoutedMoeSupply | None,
    observations: ExecutionObservations,
) -> tuple[
    dict[tuple[int | None, str, int, str], tuple[CollectiveWork, int]],
    dict[int | None, StepRecord],
]:
    batch_records = _observed_microbatch_records(record, observations)
    planned: dict[
        tuple[int | None, str, int, str], tuple[CollectiveWork, int]
    ] = {}
    for microbatch, batch_record in batch_records.items():
        for key, value in _planned_collective_work(
            batch_record,
            dims,
            tp_ranks,
            ep_ranks,
            routed_supply,
        ).items():
            planned[(microbatch, *key)] = value
    return planned, batch_records


def _aggregate_request_rows(
    rows: Sequence[tuple[str, int, int, int]],
) -> tuple[tuple[int, int, int], ...]:
    totals: dict[tuple[int, int], int] = {}
    for _, source, destination, size in rows:
        pair = (source, destination)
        totals[pair] = totals.get(pair, 0) + size
    return tuple(
        (source, destination, size)
        for (source, destination), size in sorted(totals.items())
    )


def _validate_microbatch_partition(
    record: StepRecord,
    dims: ModelDims,
    tp_ranks: Sequence[int],
    ep_ranks: Sequence[int] | None,
    routed_supply: RoutedMoeSupply | None,
    batch_records: dict[int | None, StepRecord],
    planned: dict[tuple[int | None, str, int, str], tuple[CollectiveWork, int]],
) -> None:
    if set(batch_records) == {None}:
        return
    full = _planned_collective_work(
        record,
        dims,
        tp_ranks,
        ep_ranks,
        routed_supply,
    )
    microbatches = tuple(sorted(key for key in batch_records if key is not None))
    for semantic_key, (full_work, _) in full.items():
        works = [planned[(microbatch, *semantic_key)][0] for microbatch in microbatches]
        if full_work.pair_payload_bytes:
            request_rows = tuple(
                sorted(
                    row
                    for work in works
                    for row in work.request_pair_payload_bytes
                )
            )
            if request_rows != full_work.request_pair_payload_bytes:
                raise ValueError(
                    f"observed microbatch request partition loses bytes at {semantic_key!r}"
                )
            if _aggregate_request_rows(request_rows) != full_work.pair_payload_bytes:
                raise ValueError(
                    f"observed microbatch aggregate partition loses bytes at {semantic_key!r}"
                )
        elif sum(work.payload_bytes for work in works) != full_work.payload_bytes:
            raise ValueError(
                f"observed microbatch scalar partition loses bytes at {semantic_key!r}"
            )


#: an observation whose all-to-allv sites carry no directed byte table
OBSERVED_NO_BYTE_EVIDENCE = "no-byte-evidence"
#: an observation whose all-to-allv sites all carry a directed byte table
OBSERVED_PAIR_TABLE_EVIDENCE = "observed-pair-table"
#: an observation carrying byte tables on some routed sites and not others
OBSERVED_MIXED_BYTE_EVIDENCE = "mixed"
#: an observation with no routed site at all
OBSERVED_ABSENT_BYTE_EVIDENCE = "absent"


def observed_routed_byte_evidence(observations: ExecutionObservations) -> str:
    """Name what routed byte evidence an observation carries, if any.

    The Granite producer emits zero-byte semantic all-to-allv markers, which
    are a legitimate way to identify a call site without claiming its bytes.
    They are not byte evidence, and naming that explicitly keeps a study from
    citing the marker path as a byte-correctness guard.
    """

    if not isinstance(observations, ExecutionObservations):
        raise TypeError("observations must be ExecutionObservations")
    with_table = 0
    without_table = 0
    for operation in observations.operations:
        work = operation.work
        if not isinstance(work, CollectiveWork) or work.collective != "all-to-allv":
            continue
        if work.pair_payload_bytes or work.request_pair_payload_bytes:
            with_table += 1
        else:
            without_table += 1
    if not with_table and not without_table:
        return OBSERVED_ABSENT_BYTE_EVIDENCE
    if with_table and without_table:
        return OBSERVED_MIXED_BYTE_EVIDENCE
    return (
        OBSERVED_PAIR_TABLE_EVIDENCE if with_table else OBSERVED_NO_BYTE_EVIDENCE
    )


def _routed_phase_tables(
    operations: Sequence[MoeAllToAll],
) -> tuple[RoutedPhaseTable, ...]:
    return tuple(
        RoutedPhaseTable(
            layer=operation.layer,
            phase=operation.phase,
            pair_payload_bytes=operation.pair_payload_bytes,
            request_pair_payload_bytes=operation.request_pair_payload_bytes,
        )
        for operation in operations
    )


def routed_moe_conservation(
    record: StepRecord,
    dims: ModelDims,
    ep_ranks: Sequence[int] | None,
    routed_supply: RoutedMoeSupply | None,
) -> RoutedConservationReport | None:
    """Check the step's routed byte plan against independent token ownership.

    Returns ``None`` when the step plans no MoE traffic. The ownership side is
    built from the record's per-request scheduled token counts, the declared
    engine rank and the model geometry, none of which comes from the per-token
    routing walk that produced the byte table.
    """

    ranks = tuple(ep_ranks) if ep_ranks is not None else ()
    if not ranks:
        return None
    operations = step_moe_alltoalls(
        record,
        dims,
        ranks,
        routed_supply=routed_supply,
    )
    if not operations:
        return None
    ownership = RoutedTokenOwnership(
        engine_rank=(
            routed_supply.engine_rank if routed_supply is not None else ranks[0]
        ),
        request_token_counts=tuple(
            (request.request_id, request.num_new_tokens)
            for request in record.scheduled
            if request.num_new_tokens > 0
        ),
        num_layers=dims.num_layers,
        top_k=dims.top_k,
        vector_bytes=dims.hidden_size * dims.dtype_bytes,
    )
    return routed_moe_conservation_report(
        _routed_phase_tables(operations),
        ownership,
        ranks,
        evidence_mode=(
            ROUTED_EVIDENCE_CAPTURED
            if routed_supply is not None
            else ROUTED_EVIDENCE_UNIFORM
        ),
    )


def lower_step_observations(
    record: StepRecord,
    dims: ModelDims,
    tp_ranks: Sequence[int],
    observations: ExecutionObservations,
    *,
    ep_ranks: Sequence[int] | None = None,
    routed_supply: RoutedMoeSupply | None = None,
    attach_collective_plan: bool = True,
) -> ExecutionGraph:
    """Bind adapter-observed ordering to traffic-planned collective work.

    The adapter owns tuple order, logical queues, explicit dependency edges,
    timing gates, priorities, correlations, and the completion frontier. Its
    collective observations identify a semantic call site with
    ``correlation.layer`` and ``CollectiveWork.channel_hint``. The traffic
    planner verifies the observed group and bytes, selects the algorithm, and
    supplies routed pair tables and placement epochs. Compute work passes
    through byte for byte.

    Every collective planned from each observed microbatch must be observed
    exactly once. Repeated semantic sites are legal only across distinct,
    request-partitioned microbatches, whose traffic must recombine to the
    full-step plan exactly.

    An observation may identify a routed call site with a zero-byte semantic
    marker instead of a directed byte table; use
    :func:`observed_routed_byte_evidence` to name which it did. That marker is
    not byte evidence and does not stand in for one, so the full-step routed
    plan always passes through :func:`routed_moe_conservation`, whose ownership
    side is independent of the routing walk that produced the plan.

    The returned standard :class:`ExecutionGraph` leaves realized concurrency to
    :class:`~simllm.core.DeviceRuntime`; this function has no timing or overlap
    parameter.

    ``attach_collective_plan`` defaults to True, so the returned graph carries
    the immutable traffic-owned :class:`~simllm.core.CollectivePlan` for every
    collective and the runtime never reconstructs one. Setting it False is the
    explicit compatibility bypass: the graph then has no plan, serializes to
    the accepted v1 wire form without the ``collective_plans`` field, and the
    runtime falls back to its own semantic expansion.
    """

    if not isinstance(record, StepRecord):
        raise TypeError("record must be a StepRecord")
    if not isinstance(dims, ModelDims):
        raise TypeError("dims must be ModelDims")
    if not isinstance(observations, ExecutionObservations):
        raise TypeError("observations must be ExecutionObservations")
    if not isinstance(observations.operations, tuple):
        raise TypeError("observations.operations must be a tuple")
    if not isinstance(observations.completion_operation_ids, tuple):
        raise TypeError("observations.completion_operation_ids must be a tuple")

    planned, batch_records = _planned_collective_instances(
        record,
        dims,
        tp_ranks,
        ep_ranks,
        routed_supply,
        observations,
    )
    _validate_microbatch_partition(
        record,
        dims,
        tp_ranks,
        ep_ranks,
        routed_supply,
        batch_records,
        planned,
    )
    conservation = routed_moe_conservation(record, dims, ep_ranks, routed_supply)
    if conservation is not None:
        conservation.require_conserved()
    lowered: list[ExecutionOperation] = []
    observed_keys: set[tuple[int | None, str, int, str]] = set()
    for index, operation in enumerate(observations.operations):
        if not isinstance(operation, ExecutionOperation):
            raise TypeError(
                f"observations.operations[{index}] must be an ExecutionOperation"
            )
        if isinstance(operation.work, ComputeWork):
            lowered.append(operation)
            continue
        if not isinstance(operation.work, CollectiveWork):
            raise TypeError(
                f"observations.operations[{index}]: step lowering supports only "
                "ComputeWork and CollectiveWork"
            )
        key = (
            operation.correlation.microbatch,
            *_observed_collective_key(operation, index),
        )
        if key in observed_keys:
            raise ValueError(
                f"observations.operations[{index}]: duplicate collective site {key!r}"
            )
        try:
            planned_work, placement_epoch = planned[key]
        except KeyError as exc:
            raise ValueError(
                f"observations.operations[{index}]: collective site {key!r} "
                "is absent from the step plan"
            ) from exc
        _validate_observed_collective(
            operation,
            planned_work,
            placement_epoch,
            index,
        )
        lowered.append(
            replace(
                operation,
                work=planned_work,
                placement_epoch=placement_epoch,
            )
        )
        observed_keys.add(key)

    missing = sorted(set(planned) - observed_keys)
    if missing:
        raise ValueError(f"observations: missing planned collective sites {missing!r}")
    graph = execution_graph_from_observations(
        record,
        ExecutionObservations(
            operations=tuple(lowered),
            completion_operation_ids=observations.completion_operation_ids,
        ),
    )
    if not attach_collective_plan:
        return graph
    return plan_execution_graph_collectives(graph)


@dataclass(frozen=True)
class TpAllReduce:
    """One tensor-parallel allreduce of one layer of one step."""

    layer: int
    #: "attention" or "mlp"
    site: str
    ranks: tuple[int, ...]
    payload_bytes: int


def step_tp_allreduces(
    record: StepRecord, dims: ModelDims, tp_ranks: Sequence[int]
) -> list[TpAllReduce]:
    """The step's TP collectives, empty when the step produces no traffic.

    Each layer contributes the sites :func:`layer_tp_allreduce_sites` reports
    for this model, in execution order, so a routed layer whose experts are
    expert-parallel contributes the attention site only.

    Empty means either a TP world of size 1 (nothing to reduce across) or a
    record with zero new tokens (a drain record carrying only completions).
    """
    ranks = tuple(tp_ranks)
    if len(ranks) < 2:
        return []
    payload = record.total_new_tokens * dims.hidden_size * dims.dtype_bytes
    if payload <= 0:
        return []
    sites = layer_tp_allreduce_sites(dims)
    return [
        TpAllReduce(layer=layer, site=site, ranks=ranks, payload_bytes=payload)
        for layer in range(dims.num_layers)
        for site in sites
    ]


@dataclass(frozen=True)
class MoeAllToAll:
    """One MoE all-to-allv phase of one layer of one step."""

    layer: int
    #: "dispatch" or "combine"
    phase: str
    ranks: tuple[int, ...]
    #: bytes each rank sends to EACH other rank (uniform routing)
    per_pair_bytes: int
    #: sparse ordered-pair bytes when captured routing is authoritative
    pair_payload_bytes: tuple[tuple[int, int, int], ...] = ()
    #: read-only request partition of the sparse ordered-pair table
    request_pair_payload_bytes: tuple[tuple[str, int, int, int], ...] = ()
    #: expert ownership epoch used to derive the sparse table
    placement_epoch: int = 0


class MoeMessageGrouping(str, Enum):
    """Declared coalescing rule for captured MoE message replay."""

    PER_TOKEN = "per-token"
    PER_EXPERT_GROUP = "per-expert-group"


@dataclass(frozen=True)
class RoutedMoeMessage:
    """One request-owned message and its captured routing positions."""

    request_id: str
    source_rank: int
    destination_rank: int
    payload_bytes: int
    routing_ordinals: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("message.request_id: expected a nonblank string")
        for field_name, value in (
            ("source_rank", self.source_rank),
            ("destination_rank", self.destination_rank),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"message.{field_name}: expected a nonnegative integer")
        if self.source_rank == self.destination_rank:
            raise ValueError("message ranks must differ")
        if (
            isinstance(self.payload_bytes, bool)
            or not isinstance(self.payload_bytes, int)
            or self.payload_bytes <= 0
        ):
            raise ValueError("message.payload_bytes: expected a positive integer")
        if not isinstance(self.routing_ordinals, tuple) or not self.routing_ordinals:
            raise ValueError("message.routing_ordinals: expected a nonempty tuple")
        for index, ordinal in enumerate(self.routing_ordinals):
            if not isinstance(ordinal, tuple) or len(ordinal) != 2:
                raise TypeError(
                    f"message.routing_ordinals[{index}]: expected a two-item tuple"
                )
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in ordinal
            ):
                raise ValueError(
                    f"message.routing_ordinals[{index}]: expected nonnegative integers"
                )
        if len(self.routing_ordinals) != len(set(self.routing_ordinals)):
            raise ValueError("message.routing_ordinals: duplicate routing position")


@dataclass(frozen=True)
class MoeMessageSequence:
    """One ordered captured-message phase plus aggregate projections."""

    layer: int
    phase: str
    ranks: tuple[int, ...]
    grouping: MoeMessageGrouping
    messages: tuple[RoutedMoeMessage, ...]
    pair_payload_bytes: tuple[tuple[int, int, int], ...]
    request_pair_payload_bytes: tuple[tuple[str, int, int, int], ...]
    placement_epoch: int

    def __post_init__(self) -> None:
        if self.phase not in MOE_A2A_PHASES:
            raise ValueError("sequence.phase: expected dispatch or combine")
        if not isinstance(self.grouping, MoeMessageGrouping):
            raise TypeError("sequence.grouping: expected MoeMessageGrouping")
        if not self.messages:
            raise ValueError("sequence.messages: expected at least one message")


@dataclass(frozen=True)
class _ScheduledRoutedRequest:
    request_id: str
    tokens: tuple[RoutedToken, ...] = ()
    arena_token_indices: tuple[int, ...] = ()
    phase_token_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class _RoutedMoeDispatchLayer:
    """One layer of ordered dispatch contributions from the engine owner."""

    layer: int
    ranks: tuple[int, ...]
    messages: tuple[RoutedMoeMessage, ...]
    placement_epoch: int


def _scheduled_routed_tokens(
    record: StepRecord,
    supply: RoutedMoeSupply,
) -> tuple[_ScheduledRoutedRequest, ...]:

    request_ids = [request.request_id for request in record.scheduled]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("record.scheduled: duplicate request identity")
    requests: list[_ScheduledRoutedRequest] = []
    token_count = 0
    for index, scheduled in enumerate(record.scheduled):
        path = f"record.scheduled[{index}]"
        if isinstance(scheduled.num_new_tokens, bool) or not isinstance(
            scheduled.num_new_tokens, int
        ):
            raise TypeError(f"{path}.num_new_tokens: expected an integer")
        if scheduled.num_new_tokens < 0:
            raise ValueError(f"{path}.num_new_tokens: expected a nonnegative integer")
        if isinstance(scheduled.context_length, bool) or not isinstance(
            scheduled.context_length, int
        ):
            raise TypeError(f"{path}.context_length: expected an integer")
        if supply.routed_experts is not None:
            try:
                routed_request = supply.routed_experts.by_request_id(
                    scheduled.request_id
                )
            except KeyError as exc:
                raise ValueError(
                    f"{path}.request_id: absent from routed-experts projection"
                ) from exc
            prompt_token_count = routed_request.prompt_token_count
        else:
            arena = supply.routing_arena
            lifetimes = supply.lifetimes
            assert arena is not None and lifetimes is not None
            try:
                arena_request = arena.by_request_id(scheduled.request_id)
                lifetime = lifetimes.by_request_id(scheduled.request_id)
            except KeyError as exc:
                raise ValueError(
                    f"{path}.request_id: absent from routing arena authority"
                ) from exc
            if lifetime.view.arena_id != arena.arena_id:
                raise ValueError(f"{path}.request_id: lifetime view belongs to another arena")
            prompt_token_count = arena_request.prompt_token_count
        if scheduled.num_new_tokens == 0:
            continue
        if scheduled.phase is RequestPhase.PREFILL:
            end = scheduled.context_length
            start = end - scheduled.num_new_tokens
            captured_count = prompt_token_count
            phase = "prefill"
        elif scheduled.phase is RequestPhase.DECODE:
            start = (
                scheduled.context_length
                - scheduled.num_new_tokens
                - prompt_token_count
            )
            end = start + scheduled.num_new_tokens
            captured_count = (
                len(routed_request.decode_tokens)
                if supply.routed_experts is not None
                else arena_request.decode_token_count
            )
            phase = "decode"
        else:
            raise TypeError(f"{path}.phase: expected RequestPhase")
        if start < 0 or end > captured_count or start >= end:
            raise ValueError(
                f"{path}: {phase} token slice [{start}, {end}) is outside "
                f"captured count {captured_count}"
            )
        if supply.routed_experts is not None:
            phase_tokens = (
                routed_request.prefill_tokens
                if scheduled.phase is RequestPhase.PREFILL
                else routed_request.decode_tokens
            )
            selected = tuple(phase_tokens[start:end])
            arena_indices: tuple[int, ...] = ()
        else:
            selected = ()
            absolute_start = (
                start
                if scheduled.phase is RequestPhase.PREFILL
                else prompt_token_count + start
            )
            absolute_end = absolute_start + scheduled.num_new_tokens
            arena_indices = tuple(range(absolute_start, absolute_end))
        requests.append(
            _ScheduledRoutedRequest(
                request_id=scheduled.request_id,
                tokens=selected,
                arena_token_indices=arena_indices,
                phase_token_indices=tuple(range(start, end)),
            )
        )
        token_count += len(selected) + len(arena_indices)
    if token_count != record.total_new_tokens:
        raise ValueError("record.scheduled: captured token count disagrees with total_new_tokens")
    return tuple(requests)


def _routed_moe_dispatch_layers(
    record: StepRecord,
    dims: ModelDims,
    ranks: tuple[int, ...],
    supply: RoutedMoeSupply,
) -> tuple[_RoutedMoeDispatchLayer, ...]:
    """Build the shared ordered authority for captured routed traffic."""

    if dims.num_experts <= 0 or len(ranks) < 2 or record.total_new_tokens <= 0:
        return ()
    if len(ranks) != len(set(ranks)):
        raise ValueError("ep_ranks: contains duplicate ranks")
    if any(
        isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
        for rank in ranks
    ):
        raise ValueError("ep_ranks: expected distinct nonnegative integer ranks")
    routing = supply.routed_experts or supply.routing_arena
    assert routing is not None
    authority_name = (
        "routed_experts" if supply.routed_experts is not None else "routing_arena"
    )
    if routing.expert_count != dims.num_experts:
        raise ValueError(
            f"{authority_name}.expert_count: disagrees with model num_experts"
        )
    if routing.top_k != dims.top_k:
        raise ValueError(f"{authority_name}.top_k: disagrees with model top_k")
    expected_layers = tuple(range(dims.num_layers))
    if routing.moe_layer_indices != expected_layers:
        raise ValueError(
            f"{authority_name}.moe_layer_indices: disagree with model layers"
        )
    placement = supply.placement_for_step(record.step_index)
    owners = placement.owner_map()
    expected_keys = {
        (layer, expert) for layer in expected_layers for expert in range(dims.num_experts)
    }
    owner_keys = set(owners)
    missing = expected_keys - owner_keys
    extra = owner_keys - expected_keys
    if missing:
        layer, expert = min(missing)
        raise ValueError(
            f"placement epoch {placement.placement_epoch}: missing owner for "
            f"layer {layer} expert {expert}"
        )
    if extra:
        layer, expert = min(extra)
        raise ValueError(
            f"placement epoch {placement.placement_epoch}: unexpected owner for "
            f"layer {layer} expert {expert}"
        )
    invalid_ranks = set(owners.values()) - set(ranks)
    if invalid_ranks:
        raise ValueError(
            "placement snapshot: owner ranks outside ep_ranks: "
            + ", ".join(str(rank) for rank in sorted(invalid_ranks))
        )
    source = supply.engine_rank
    if source not in ranks:
        raise ValueError(f"supply.engine_rank: rank {source} is outside ep_ranks")

    scheduled_requests = _scheduled_routed_tokens(record, supply)
    vector_bytes = dims.hidden_size * dims.dtype_bytes
    layers: list[_RoutedMoeDispatchLayer] = []
    for layer in expected_layers:
        messages: list[RoutedMoeMessage] = []
        for scheduled_request in scheduled_requests:
            if supply.routed_experts is not None:
                token_rows = (
                    (token.token_index, token.layers[layer].expert_ids)
                    for token in scheduled_request.tokens
                )
            else:
                arena = supply.routing_arena
                lifetimes = supply.lifetimes
                assert arena is not None and lifetimes is not None
                view = lifetimes.by_request_id(scheduled_request.request_id).view
                token_rows = (
                    (
                        phase_token_index,
                        arena.expert_ids_at(
                            view.token_offset,
                            view.token_count,
                            arena_token_index,
                            layer,
                        ),
                    )
                    for phase_token_index, arena_token_index in zip(
                        scheduled_request.phase_token_indices,
                        scheduled_request.arena_token_indices,
                        strict=True,
                    )
                )
            for token_index, expert_ids in token_rows:
                destinations: dict[int, int] = {}
                for top_k_index, expert in enumerate(expert_ids):
                    destination = owners[(layer, expert)]
                    destinations.setdefault(destination, top_k_index)
                for destination, top_k_index in destinations.items():
                    if destination == source:
                        continue
                    messages.append(
                        RoutedMoeMessage(
                            request_id=scheduled_request.request_id,
                            source_rank=source,
                            destination_rank=destination,
                            payload_bytes=vector_bytes,
                            routing_ordinals=((token_index, top_k_index),),
                        )
                    )
        layers.append(
            _RoutedMoeDispatchLayer(
                layer=layer,
                ranks=ranks,
                messages=tuple(messages),
                placement_epoch=placement.placement_epoch,
            )
        )
    return tuple(layers)


def _message_request_pairs(
    messages: tuple[RoutedMoeMessage, ...],
) -> tuple[tuple[str, int, int, int], ...]:
    totals: dict[tuple[str, int, int], int] = {}
    for message in messages:
        key = (
            message.request_id,
            message.source_rank,
            message.destination_rank,
        )
        totals[key] = totals.get(key, 0) + message.payload_bytes
    return tuple(
        (request_id, source, destination, size)
        for (request_id, source, destination), size in sorted(totals.items())
    )


def _aggregate_request_pairs(
    entries: tuple[tuple[str, int, int, int], ...],
) -> tuple[tuple[int, int, int], ...]:
    totals: dict[tuple[int, int], int] = {}
    for _, source, destination, size in entries:
        pair = (source, destination)
        totals[pair] = totals.get(pair, 0) + size
    return tuple(
        (source, destination, size)
        for (source, destination), size in sorted(totals.items())
    )


def _transpose_routed_messages(
    messages: tuple[RoutedMoeMessage, ...],
) -> tuple[RoutedMoeMessage, ...]:
    return tuple(
        RoutedMoeMessage(
            request_id=message.request_id,
            source_rank=message.destination_rank,
            destination_rank=message.source_rank,
            payload_bytes=message.payload_bytes,
            routing_ordinals=message.routing_ordinals,
        )
        for message in messages
    )


def _routed_moe_alltoalls_from_layers(
    layers: tuple[_RoutedMoeDispatchLayer, ...],
) -> list[MoeAllToAll]:
    operations: list[MoeAllToAll] = []
    for layer in layers:
        request_dispatch = _message_request_pairs(layer.messages)
        dispatch = _aggregate_request_pairs(request_dispatch)
        request_combine = _message_request_pairs(
            _transpose_routed_messages(layer.messages)
        )
        combine = _aggregate_request_pairs(request_combine)
        operations.extend(
            (
                MoeAllToAll(
                    layer=layer.layer,
                    phase="dispatch",
                    ranks=layer.ranks,
                    per_pair_bytes=0,
                    pair_payload_bytes=dispatch,
                    request_pair_payload_bytes=request_dispatch,
                    placement_epoch=layer.placement_epoch,
                ),
                MoeAllToAll(
                    layer=layer.layer,
                    phase="combine",
                    ranks=layer.ranks,
                    per_pair_bytes=0,
                    pair_payload_bytes=combine,
                    request_pair_payload_bytes=request_combine,
                    placement_epoch=layer.placement_epoch,
                ),
            )
        )
    return operations


def _routed_moe_alltoalls(
    record: StepRecord,
    dims: ModelDims,
    ranks: tuple[int, ...],
    supply: RoutedMoeSupply,
) -> list[MoeAllToAll]:
    return _routed_moe_alltoalls_from_layers(
        _routed_moe_dispatch_layers(record, dims, ranks, supply)
    )


def step_moe_alltoalls(
    record: StepRecord,
    dims: ModelDims,
    ep_ranks: Sequence[int],
    *,
    routed_supply: RoutedMoeSupply | None = None,
) -> list[MoeAllToAll]:
    """The step's MoE all-to-alls, empty when the step produces none.

    Per MoE layer (every layer of an MoE ``dims``): a dispatch pairwise
    all-to-allv routing each token's activation to its ``top_k`` experts'
    owner ranks, then a combine pairwise all-to-allv returning the expert
    outputs, both over the expert-parallel group ``ep_ranks`` of W ranks.

    With ``routed_supply``, the scheduled request slices select exact captured
    input tokens owned by ``routed_supply.engine_rank``. The other EP ranks
    own experts but contribute zero scheduled tokens to this isolated engine
    step. Expert ownership at the step's explicit placement epoch maps each
    token to destination ranks. Dispatch deduplicates several selected experts
    on the same destination into one hidden vector, and combine is the exact
    reverse table after owner-side pre-reduction.

    Without a routed supply, the uniform approximation uses ``ep_ranks[0]``
    as the one engine rank. It spreads that engine's
    ``total_new_tokens * top_k`` (token, expert) assignments evenly over the W
    expert-owner ranks, so the engine sends

        per_pair_bytes = total_new_tokens * top_k * hidden_size
                         * dtype_bytes // W

    to every other rank in dispatch, and combine transposes those pairs. The
    1/W share routed to the engine rank's resident experts stays local and
    never touches the fabric. The floor division remains part of this explicit
    approximation.

    Empty means: dense dims (``num_experts == 0``), an EP world smaller
    than 2, or a zero-new-token drain record. The uniform path also returns
    empty when its per-pair share rounds to zero bytes.
    """
    ranks = tuple(ep_ranks)
    if dims.num_experts <= 0 or len(ranks) < 2:
        return []
    if record.total_new_tokens <= 0:
        return []
    if routed_supply is not None:
        return _routed_moe_alltoalls(record, dims, ranks, routed_supply)
    per_pair = (record.total_new_tokens * dims.top_k * dims.hidden_size * dims.dtype_bytes) // len(
        ranks
    )
    if per_pair <= 0:
        return []
    source = ranks[0]
    dispatch = tuple(
        (source, destination, per_pair)
        for destination in ranks
        if destination != source
    )
    combine = tuple(
        (destination, source, per_pair)
        for destination in ranks
        if destination != source
    )
    return [
        MoeAllToAll(
            layer=layer,
            phase=phase,
            ranks=ranks,
            per_pair_bytes=0,
            pair_payload_bytes=(dispatch if phase == "dispatch" else combine),
        )
        for layer in range(dims.num_layers)
        for phase in MOE_A2A_PHASES
    ]


def _group_routed_messages(
    messages: tuple[RoutedMoeMessage, ...],
    grouping: MoeMessageGrouping,
) -> tuple[RoutedMoeMessage, ...]:
    if grouping is MoeMessageGrouping.PER_TOKEN:
        return messages
    if grouping is not MoeMessageGrouping.PER_EXPERT_GROUP:
        raise TypeError("grouping must be a MoeMessageGrouping")

    order: list[tuple[str, int, int]] = []
    totals: dict[tuple[str, int, int], int] = {}
    ordinals: dict[tuple[str, int, int], list[tuple[int, int]]] = {}
    for message in messages:
        key = (
            message.request_id,
            message.source_rank,
            message.destination_rank,
        )
        if key not in totals:
            order.append(key)
            totals[key] = 0
            ordinals[key] = []
        totals[key] += message.payload_bytes
        ordinals[key].extend(message.routing_ordinals)
    return tuple(
        RoutedMoeMessage(
            request_id=request_id,
            source_rank=source,
            destination_rank=destination,
            payload_bytes=totals[(request_id, source, destination)],
            routing_ordinals=tuple(ordinals[(request_id, source, destination)]),
        )
        for request_id, source, destination in order
    )


def step_moe_message_sequences(
    record: StepRecord,
    dims: ModelDims,
    ep_ranks: Sequence[int],
    *,
    routed_supply: RoutedMoeSupply,
    grouping: MoeMessageGrouping,
) -> list[MoeMessageSequence]:
    """Derive ordered MoE messages from captured per-token routing.

    The base order is scheduled request, selected token and returned top-k
    position from the declared engine owner. Multiple experts on one
    destination produce one vector transfer for that token at the first
    matching top-k position. ``PER_EXPERT_GROUP`` coalesces a request's
    whole-layer traffic for one ordered pair and keeps the first contributing
    position as its issue position. Combine transposes the dispatch stream
    without claiming a kernel or wire issue order that the capture did not
    observe.
    """

    if not isinstance(grouping, MoeMessageGrouping):
        raise TypeError("grouping must be a MoeMessageGrouping")
    ranks = tuple(ep_ranks)
    layers = _routed_moe_dispatch_layers(
        record, dims, ranks, routed_supply
    )
    if not layers:
        return []

    aggregate = _routed_moe_alltoalls_from_layers(layers)
    aggregate_by_key = {
        (operation.layer, operation.phase): operation for operation in aggregate
    }
    sequences: list[MoeMessageSequence] = []

    for layer in layers:
        dispatch_messages = _group_routed_messages(layer.messages, grouping)
        combine_messages = _group_routed_messages(
            _transpose_routed_messages(layer.messages),
            grouping,
        )
        for phase, messages in (
            ("dispatch", dispatch_messages),
            ("combine", combine_messages),
        ):
            authority = aggregate_by_key[(layer.layer, phase)]
            request_pairs = _message_request_pairs(messages)
            pair_payloads = _aggregate_request_pairs(request_pairs)
            if request_pairs != authority.request_pair_payload_bytes:
                raise AssertionError(
                    f"sequenced {phase} request projection disagrees with aggregate authority"
                )
            if pair_payloads != authority.pair_payload_bytes:
                raise AssertionError(
                    f"sequenced {phase} pair projection disagrees with aggregate authority"
                )
            sequences.append(
                MoeMessageSequence(
                    layer=layer.layer,
                    phase=phase,
                    ranks=ranks,
                    grouping=grouping,
                    messages=messages,
                    pair_payload_bytes=pair_payloads,
                    request_pair_payload_bytes=request_pairs,
                    placement_epoch=layer.placement_epoch,
                )
            )
    return sequences


def _moe_operation_id(record: StepRecord, operation: MoeAllToAll) -> str:
    return f"step-{record.step_index}:layer-{operation.layer}:ep-{operation.phase}"


def _tp_operation_id(record: StepRecord, operation: TpAllReduce) -> str:
    return f"step-{record.step_index}:layer-{operation.layer}:tp-{operation.site}"


def _compute_operation_id(record: StepRecord, layer: int, rank: int) -> str:
    return f"step-{record.step_index}:layer-{layer}:rank-{rank}:compute"


def _request_partitions(
    operation: MoeAllToAll,
) -> dict[tuple[int, int], tuple[tuple[str, int], ...]]:
    by_pair: dict[tuple[int, int], list[tuple[str, int]]] = {}
    for request_id, source, destination, size in operation.request_pair_payload_bytes:
        by_pair.setdefault((source, destination), []).append((request_id, size))
    return {pair: tuple(sorted(entries)) for pair, entries in by_pair.items()}


def _expected_fidelity_rows(
    record: StepRecord,
    operations: Sequence[MoeAllToAll],
) -> tuple[tuple[RequestPairRow, ...], tuple[AggregatePairRow, ...]]:
    request_rows = []
    aggregate_rows = []
    for operation in operations:
        if not operation.request_pair_payload_bytes:
            continue
        operation_id = _moe_operation_id(record, operation)
        request_rows.extend(
            (operation_id, request_id, source, destination, size)
            for request_id, source, destination, size in (operation.request_pair_payload_bytes)
        )
        aggregate_rows.extend(
            (operation_id, source, destination, size)
            for source, destination, size in operation.pair_payload_bytes
        )
    return tuple(request_rows), tuple(aggregate_rows)


def _compare_operations_to_messages(
    record: StepRecord,
    operations: Sequence[MoeAllToAll],
    messages: Sequence[GoalMessage],
) -> RequestFidelityReport:
    request_rows, aggregate_rows = _expected_fidelity_rows(record, operations)
    return compare_goal_request_attribution(
        request_rows,
        aggregate_rows,
        messages,
    )


def compare_request_moe_fidelity(
    record: StepRecord,
    dims: ModelDims,
    ep_ranks: Sequence[int],
    routed_supply: RoutedMoeSupply,
    messages: Sequence[GoalMessage],
) -> RequestFidelityReport:
    """Compare rendered messages with the scheduled routed-supply authority."""

    operations = step_moe_alltoalls(
        record,
        dims,
        ep_ranks,
        routed_supply=routed_supply,
    )
    return _compare_operations_to_messages(record, operations, messages)


def validate_request_moe_fidelity(
    record: StepRecord,
    dims: ModelDims,
    ep_ranks: Sequence[int],
    routed_supply: RoutedMoeSupply,
    messages: Sequence[GoalMessage],
) -> RequestFidelityReport:
    """Fail closed unless every rendered request byte matches captured routing."""

    return compare_request_moe_fidelity(
        record,
        dims,
        ep_ranks,
        routed_supply,
        messages,
    ).require_match()


def render_step_goal(
    record: StepRecord,
    dims: ModelDims,
    tp_ranks: Sequence[int],
    per_layer_calc_ns: int | Sequence[int],
    *,
    ep_ranks: Sequence[int] | None = None,
    routed_supply: RoutedMoeSupply | None = None,
    num_goal_ranks: int | None = None,
    base_tag: int = 1000,
) -> GoalTrace:
    """Render one step as a GOAL program over the TP (and optionally EP) groups.

    Every participating rank executes the serial chain over layers: ``calc``
    of the corresponding ``per_layer_calc_ns`` GOAL units (ns), then the
    layer's allreduce sites in execution order (only when the TP world
    produces collectives), then, for MoE dims with ``ep_ranks`` given,
    the dispatch and combine all-to-allvs over the EP group; the next layer's
    calc waits for the previous layer's last collective. This is the serial
    compatibility off path; observation-aware execution uses
    :func:`lower_step_observations`. The fixed
    calc/allreduce/dispatch/combine order is TRAF-9. A scalar calc
    repeats on every layer for compatibility; a sequence supplies unequal
    layer costs. Participants are the TP ranks plus, when MoE all-to-alls
    exist, the EP ranks; other ranks below ``num_goal_ranks`` get one
    zero-cost calc so every rank block is populated.

    Tags: allreduce k, counted over the emitted sites in layer-major order,
    takes the disjoint block ``base_tag + k * 2(W-1)`` onward, one tag per
    round; MoE all-to-allvs take one tag each, ``base_tag + tp_tag_slots + j``
    for all-to-all j (layer * 2 + phase index), starting right after the
    allreduce blocks. A layer that emits one site therefore consumes one
    block, so the all-to-all base moves down with the shortened list instead
    of leaving a hole. A model with two sites per layer keeps the historical
    ``layer * 2 + site index`` allocation exactly, and a step without MoE work
    renders byte-identically to the pre-MoE emitter (golden-tested).

    Raises ``ValueError`` when the step has neither TP collectives nor MoE
    all-to-alls (callers decide what "no network work" means; the
    closed-loop sink returns None so the frontend's own compute estimate
    stands).
    """
    tp_ops = step_tp_allreduces(record, dims, tp_ranks)
    moe_ops = step_moe_alltoalls(
        record,
        dims,
        ep_ranks if ep_ranks is not None else [],
        routed_supply=routed_supply,
    )
    if not tp_ops and not moe_ops:
        raise ValueError(
            "step has no tensor-parallel collectives and no MoE all-to-alls "
            "to render (TP world < 2 or zero new tokens, and no expert traffic)"
        )
    if isinstance(per_layer_calc_ns, int):
        layer_calc_ns = (per_layer_calc_ns,) * dims.num_layers
    else:
        layer_calc_ns = tuple(per_layer_calc_ns)
    if len(layer_calc_ns) != dims.num_layers:
        raise ValueError(
            f"received {len(layer_calc_ns)} layer calc values for num_layers={dims.num_layers}"
        )
    if any(value < 0 for value in layer_calc_ns):
        raise ValueError("layer calc values must be nonnegative")
    ranks = list(tp_ranks)
    participants = list(ranks)
    if moe_ops:
        for rank in moe_ops[0].ranks:
            if rank not in participants:
                participants.append(rank)
    tag_stride = 2 * (len(ranks) - 1) if tp_ops else 0
    moe_base_tag = base_tag + len(tp_ops) * tag_stride
    tp_by_key = {
        (op.layer, op.site): (index, op) for index, op in enumerate(tp_ops)
    }
    moe_by_key = {(op.layer, op.phase): op for op in moe_ops}
    moe_tag_by_key = {
        (op.layer, op.phase): moe_base_tag + index
        for index, op in enumerate(moe_ops)
    }
    if num_goal_ranks is None:
        num_goal_ranks = max(participants) + 1
    minimum_ranks = max(participants) + 1
    if num_goal_ranks < minimum_ranks:
        raise ValueError(f"num_goal_ranks={num_goal_ranks} cannot contain rank {minimum_ranks - 1}")
    trace = GoalTrace(num_goal_ranks)

    previous: dict[int, str] = {}
    for layer in range(dims.num_layers):
        # start of a layer: the per-layer compute, chained to the previous
        # layer's last collective
        calc_done: dict[int, str] = {}
        for rank in participants:
            calc = trace.rank(rank).calc(
                layer_calc_ns[layer],
                operation_id=_compute_operation_id(record, layer, rank),
            )
            if rank in previous:
                trace.rank(rank).requires(calc, previous[rank])
            calc_done[rank] = calc
        previous = {**previous, **calc_done}
        if tp_ops:
            for site in TP_ALLREDUCE_SITES:
                entry = tp_by_key.get((layer, site))
                if entry is None:
                    continue
                op_index, op = entry
                done = ring_allreduce(
                    trace,
                    ranks=list(op.ranks),
                    size_bytes=op.payload_bytes,
                    base_tag=base_tag + op_index * tag_stride,
                    after=previous,
                    operation_id=_tp_operation_id(record, op),
                )
                previous = {**previous, **done}
        if moe_ops:
            for phase_index in range(len(MOE_A2A_PHASES)):
                op = moe_by_key.get((layer, MOE_A2A_PHASES[phase_index]))
                if op is None:
                    continue
                if op.pair_payload_bytes:
                    send_bytes = {
                        (source, destination): size
                        for source, destination, size in op.pair_payload_bytes
                    }
                else:
                    send_bytes = {
                        (s, d): op.per_pair_bytes for s in op.ranks for d in op.ranks if s != d
                    }
                done = pairwise_all_to_allv(
                    trace,
                    ranks=list(op.ranks),
                    send_bytes=send_bytes,
                    tag=moe_tag_by_key[(op.layer, op.phase)],
                    after=previous,
                    operation_id=_moe_operation_id(record, op),
                    request_send_bytes=(
                        _request_partitions(op) if op.request_pair_payload_bytes else None
                    ),
                )
                previous = {**previous, **done}

    used = set(participants)
    for rank in range(num_goal_ranks):
        if rank not in used:
            trace.rank(rank).calc(0)
    if moe_ops:
        report = routed_moe_conservation(record, dims, ep_ranks, routed_supply)
        if report is not None:
            report.require_conserved()
    if any(operation.request_pair_payload_bytes for operation in moe_ops):
        _compare_operations_to_messages(record, moe_ops, trace.messages).require_match()
    return trace


def render_sequenced_step_goal(
    record: StepRecord,
    dims: ModelDims,
    tp_ranks: Sequence[int],
    per_layer_calc_ns: int | Sequence[int],
    *,
    ep_ranks: Sequence[int],
    routed_supply: RoutedMoeSupply,
    message_grouping: MoeMessageGrouping,
    num_goal_ranks: int | None = None,
    base_tag: int = 1000,
) -> GoalTrace:
    """Render the explicit ``captured-message-sequence`` precision level.

    This opt-in entry point preserves the declared engine rank's
    framework-returned routing sequence and the selected grouping rule. The
    established :func:`render_step_goal` aggregate compatibility renderer
    remains the default and has no message-granularity selector.
    """

    tp_ops = step_tp_allreduces(record, dims, tp_ranks)
    moe_sequences = step_moe_message_sequences(
        record,
        dims,
        ep_ranks,
        routed_supply=routed_supply,
        grouping=message_grouping,
    )
    if not tp_ops and not moe_sequences:
        raise ValueError(
            "step has no tensor-parallel collectives and no sequenced MoE "
            "messages to render"
        )
    if isinstance(per_layer_calc_ns, int):
        layer_calc_ns = (per_layer_calc_ns,) * dims.num_layers
    else:
        layer_calc_ns = tuple(per_layer_calc_ns)
    if len(layer_calc_ns) != dims.num_layers:
        raise ValueError(
            f"received {len(layer_calc_ns)} layer calc values for "
            f"num_layers={dims.num_layers}"
        )
    if any(value < 0 for value in layer_calc_ns):
        raise ValueError("layer calc values must be nonnegative")

    ranks = list(tp_ranks)
    participants = list(ranks)
    if moe_sequences:
        for rank in moe_sequences[0].ranks:
            if rank not in participants:
                participants.append(rank)
    tag_stride = 2 * (len(ranks) - 1) if tp_ops else 0
    moe_base_tag = base_tag + len(tp_ops) * tag_stride
    tp_by_key = {
        (operation.layer, operation.site): (index, operation)
        for index, operation in enumerate(tp_ops)
    }
    moe_by_key = {
        (sequence.layer, sequence.phase): sequence
        for sequence in moe_sequences
    }
    if num_goal_ranks is None:
        num_goal_ranks = max(participants) + 1
    minimum_ranks = max(participants) + 1
    if num_goal_ranks < minimum_ranks:
        raise ValueError(
            f"num_goal_ranks={num_goal_ranks} cannot contain rank "
            f"{minimum_ranks - 1}"
        )
    trace = GoalTrace(num_goal_ranks)

    previous: dict[int, str] = {}
    for layer in range(dims.num_layers):
        calc_done: dict[int, str] = {}
        for rank in participants:
            calc = trace.rank(rank).calc(
                layer_calc_ns[layer],
                operation_id=_compute_operation_id(record, layer, rank),
            )
            if rank in previous:
                trace.rank(rank).requires(calc, previous[rank])
            calc_done[rank] = calc
        previous = {**previous, **calc_done}
        if tp_ops:
            for site in TP_ALLREDUCE_SITES:
                entry = tp_by_key.get((layer, site))
                if entry is None:
                    continue
                op_index, operation = entry
                done = ring_allreduce(
                    trace,
                    ranks=list(operation.ranks),
                    size_bytes=operation.payload_bytes,
                    base_tag=base_tag + op_index * tag_stride,
                    after=previous,
                    operation_id=_tp_operation_id(record, operation),
                )
                previous = {**previous, **done}
        if moe_sequences:
            for phase_index, phase in enumerate(MOE_A2A_PHASES):
                moe_index = layer * len(MOE_A2A_PHASES) + phase_index
                sequence = moe_by_key.get((layer, phase))
                if sequence is None:
                    continue
                operation_id = (
                    f"step-{record.step_index}:layer-{sequence.layer}:"
                    f"ep-{sequence.phase}"
                )
                done = ordered_pairwise_messages(
                    trace,
                    ranks=list(sequence.ranks),
                    messages=tuple(
                        (
                            message.request_id,
                            message.source_rank,
                            message.destination_rank,
                            message.payload_bytes,
                        )
                        for message in sequence.messages
                    ),
                    tag=moe_base_tag + moe_index,
                    after=previous,
                    operation_id=operation_id,
                )
                previous = {**previous, **done}

    used = set(participants)
    for rank in range(num_goal_ranks):
        if rank not in used:
            trace.rank(rank).calc(0)

    aggregate = step_moe_alltoalls(
        record,
        dims,
        ep_ranks,
        routed_supply=routed_supply,
    )
    _compare_operations_to_messages(record, aggregate, trace.messages).require_match()
    for sequence in moe_sequences:
        operation_id = (
            f"step-{record.step_index}:layer-{sequence.layer}:ep-{sequence.phase}"
        )
        rendered = tuple(
            (
                message.request_payload_bytes[0][0],
                message.source_rank,
                message.destination_rank,
                message.payload_bytes,
            )
            for message in trace.messages
            if message.operation_id == operation_id
        )
        expected = tuple(
            (
                message.request_id,
                message.source_rank,
                message.destination_rank,
                message.payload_bytes,
            )
            for message in sequence.messages
        )
        if rendered != expected:
            raise AssertionError(
                f"rendered {sequence.phase} message order disagrees with plan"
            )
    return trace


def _ring_communication_phases(
    operation: TpAllReduce,
    *,
    operation_base_tag: int,
    operation_id: str | None = None,
) -> tuple[CollectiveCommunicationPhase, ...]:
    world = len(operation.ranks)
    chunk = max(1, operation.payload_bytes // world)
    phases = []
    for round_index in range(2 * (world - 1)):
        segments = tuple(
            DirectedCollectiveSegment(
                source_rank=rank,
                destination_rank=operation.ranks[(index + 1) % world],
                payload_bytes=chunk,
                tag=operation_base_tag + round_index,
            )
            for index, rank in enumerate(operation.ranks)
        )
        phases.append(
            CollectiveCommunicationPhase(
                phase_id=(
                    f"layer-{operation.layer}:tp-{operation.site}:"
                    f"round-{round_index}"
                ),
                layer=operation.layer,
                participants=operation.ranks,
                segments=segments,
                operation_id=operation_id,
            )
        )
    return tuple(phases)


def _moe_communication_phase(
    operation: MoeAllToAll,
    *,
    tag: int,
    operation_id: str | None = None,
) -> CollectiveCommunicationPhase:
    request_partitions = _request_partitions(operation)
    if operation.pair_payload_bytes:
        pair_payloads = operation.pair_payload_bytes
    else:
        pair_payloads = tuple(
            (source, destination, operation.per_pair_bytes)
            for source in operation.ranks
            for destination in operation.ranks
            if source != destination
        )
    return CollectiveCommunicationPhase(
        phase_id=f"layer-{operation.layer}:ep-{operation.phase}",
        layer=operation.layer,
        participants=operation.ranks,
        segments=tuple(
            DirectedCollectiveSegment(
                source_rank=source,
                destination_rank=destination,
                payload_bytes=payload_bytes,
                tag=tag,
                request_payload_bytes=request_partitions.get(
                    (source, destination),
                    (),
                ),
            )
            for source, destination, payload_bytes in pair_payloads
            if payload_bytes > 0
        ),
        operation_id=operation_id,
    )


def step_communication_phases(
    record: StepRecord,
    dims: ModelDims,
    tp_ranks: Sequence[int],
    *,
    ep_ranks: Sequence[int] | None = None,
    routed_supply: RoutedMoeSupply | None = None,
    base_tag: int = 1000,
) -> tuple[CollectiveCommunicationPhase, ...]:
    """Expand a step into serial phases of positive directed transfers."""

    tp_ops = step_tp_allreduces(record, dims, tp_ranks)
    moe_ops = step_moe_alltoalls(
        record,
        dims,
        ep_ranks if ep_ranks is not None else (),
        routed_supply=routed_supply,
    )
    tp_by_key = {
        (operation.layer, operation.site): (index, operation)
        for index, operation in enumerate(tp_ops)
    }
    moe_by_key = {(operation.layer, operation.phase): operation for operation in moe_ops}
    tp_ranks_tuple = tuple(tp_ranks)
    tag_stride = 2 * (len(tp_ranks_tuple) - 1) if tp_ops else 0
    moe_base_tag = base_tag + len(tp_ops) * tag_stride
    moe_tag_by_key = {
        (operation.layer, operation.phase): moe_base_tag + index
        for index, operation in enumerate(moe_ops)
    }

    phases = []
    for layer in range(dims.num_layers):
        for site in TP_ALLREDUCE_SITES:
            entry = tp_by_key.get((layer, site))
            if entry is None:
                continue
            operation_index, operation = entry
            phases.extend(
                _ring_communication_phases(
                    operation,
                    operation_base_tag=base_tag + operation_index * tag_stride,
                )
            )
        for phase_index, phase in enumerate(MOE_A2A_PHASES):
            operation = moe_by_key.get((layer, phase))
            if operation is None:
                continue
            phases.append(
                _moe_communication_phase(
                    operation,
                    tag=moe_tag_by_key[(operation.layer, operation.phase)],
                    operation_id=_moe_operation_id(record, operation),
                )
            )
    return tuple(phases)


def _execution_graph_communication_phases(
    graph: ExecutionGraph,
    *,
    base_tag: int,
) -> tuple[CollectiveCommunicationPhase, ...]:
    """Expand graph-owned collectives without reconstructing step policy."""

    validate_execution_graph(graph)
    if type(base_tag) is not int or base_tag < 0:
        raise ValueError("base_tag must be a nonnegative integer")

    tags = collective_goal_tags(graph, base_tag=base_tag)
    plan_by_operation = collective_plan_by_operation(graph)
    phases: list[CollectiveCommunicationPhase] = []
    for index, operation in enumerate(graph.operations):
        work = operation.work
        if isinstance(work, ComputeWork):
            continue
        if not isinstance(work, CollectiveWork):
            raise TypeError(
                f"graph.operations[{index}] carries unsupported "
                f"{type(work).__name__} in the locality projection"
            )
        layer = operation.correlation.layer
        if layer is None:
            raise ValueError(
                f"graph.operations[{index}] collective needs correlation.layer"
            )
        channel = work.channel_hint
        if channel is None:
            raise ValueError(
                f"graph.operations[{index}] collective needs channel_hint"
            )

        if plan_by_operation:
            plan = plan_by_operation[operation.operation_id]
            site = "tp" if plan.collective == "all-reduce" else "ep"
            extents_by_round: dict[int, list] = {}
            for extent in plan.extents:
                extents_by_round.setdefault(extent.round_index, []).append(extent)
            for round_ in plan.rounds:
                extents = extents_by_round.get(round_.round_index, [])
                if not extents:
                    continue
                suffix = (
                    f":round-{round_.round_index}" if len(plan.rounds) > 1 else ""
                )
                phases.append(
                    CollectiveCommunicationPhase(
                        phase_id=f"layer-{layer}:{site}-{channel}{suffix}",
                        layer=layer,
                        participants=plan.rank_order,
                        segments=tuple(
                            DirectedCollectiveSegment(
                                source_rank=extent.source_rank,
                                destination_rank=extent.destination_rank,
                                payload_bytes=extent.payload_bytes,
                                tag=round_.tag,
                                request_payload_bytes=(
                                    extent.request_payload_bytes
                                ),
                            )
                            for extent in extents
                        ),
                        operation_id=operation.operation_id,
                    )
                )
            continue

        operation_tags = tags[operation.operation_id]
        if work.collective == "all-reduce" and work.algorithm_hint == "ring":
            expected_rounds = 2 * (len(work.ranks) - 1)
            if len(operation_tags) != expected_rounds:
                raise AssertionError("ring tag allocation disagrees with its round count")
            phases.extend(
                _ring_communication_phases(
                    TpAllReduce(
                        layer=layer,
                        site=channel,
                        ranks=work.ranks,
                        payload_bytes=work.payload_bytes,
                    ),
                    operation_base_tag=operation_tags[0],
                    operation_id=operation.operation_id,
                )
            )
            continue

        if work.collective == "all-to-allv" and work.algorithm_hint == "pairwise":
            if len(operation_tags) != 1:
                raise AssertionError("pairwise tag allocation must contain one tag")
            if work.pair_payload_bytes:
                pair_payload_bytes = work.pair_payload_bytes
                per_pair_bytes = 0
            else:
                pair_payload_bytes = ()
                per_pair_bytes = work.payload_bytes
            if not pair_payload_bytes and per_pair_bytes <= 0:
                continue
            phases.append(
                _moe_communication_phase(
                    MoeAllToAll(
                        layer=layer,
                        phase=channel,
                        ranks=work.ranks,
                        per_pair_bytes=per_pair_bytes,
                        pair_payload_bytes=pair_payload_bytes,
                        request_pair_payload_bytes=work.request_pair_payload_bytes,
                        placement_epoch=operation.placement_epoch,
                    ),
                    tag=operation_tags[0],
                    operation_id=operation.operation_id,
                )
            )
            continue

        raise ValueError(
            f"graph.operations[{index}] uses unsupported collective "
            f"{work.collective!r} with algorithm {work.algorithm_hint!r}"
        )
    return tuple(phases)


def plan_execution_graph_locality(
    graph: ExecutionGraph,
    *,
    rank_mapper: RankMapper | None = None,
    nvlink_bandwidth_bytes_per_second: int = (
        DEFAULT_NVLINK_BANDWIDTH_BYTES_PER_SECOND
    ),
    base_tag: int = 1000,
) -> StepLocalityPlan:
    """Project graph-owned collective work into placement service phases.

    Operation order, collective identities, algorithms, payloads and tags all
    come from the validated ``ExecutionGraph``. The placement manifest only
    classifies the resulting directed segments as local or fabric traffic.
    Unsupported work fails during preflight, before any renderer is mutated.
    """

    if rank_mapper is not None and not isinstance(rank_mapper, RankMapper):
        raise TypeError("rank_mapper must be RankMapper or None")
    phases = _execution_graph_communication_phases(graph, base_tag=base_tag)
    if rank_mapper is not None:
        for phase in phases:
            for rank in phase.participants:
                rank_mapper.goal_rank(rank)
    classified = classify_step_locality(
        phases,
        rank_mapper=rank_mapper,
        bandwidth_bytes_per_second=nvlink_bandwidth_bytes_per_second,
    )
    return replace(
        classified,
        graph_execution_id=graph.execution_id,
        dependency_edges=effective_dependency_edges(graph),
    )


def validate_execution_graph_locality_projection(
    graph: ExecutionGraph,
    plan: StepLocalityPlan,
    *,
    rank_mapper: RankMapper | None = None,
    nvlink_bandwidth_bytes_per_second: int = (
        DEFAULT_NVLINK_BANDWIDTH_BYTES_PER_SECOND
    ),
    base_tag: int = 1000,
) -> None:
    """Reject any loss, duplication or mutation in a graph locality plan."""

    if not isinstance(plan, StepLocalityPlan):
        raise TypeError("plan must be a StepLocalityPlan")
    expected = plan_execution_graph_locality(
        graph,
        rank_mapper=rank_mapper,
        nvlink_bandwidth_bytes_per_second=nvlink_bandwidth_bytes_per_second,
        base_tag=base_tag,
    )
    if plan != expected:
        raise ValueError("locality plan does not exactly project the execution graph")


def plan_step_locality(
    record: StepRecord,
    dims: ModelDims,
    tp_ranks: Sequence[int],
    *,
    ep_ranks: Sequence[int] | None = None,
    routed_supply: RoutedMoeSupply | None = None,
    rank_mapper: RankMapper | None = None,
    nvlink_bandwidth_bytes_per_second: int = (
        DEFAULT_NVLINK_BANDWIDTH_BYTES_PER_SECOND
    ),
    base_tag: int = 1000,
) -> StepLocalityPlan:
    """Return the placement split before any renderer mutates a GOAL trace."""

    phases = step_communication_phases(
        record,
        dims,
        tp_ranks,
        ep_ranks=ep_ranks,
        routed_supply=routed_supply,
        base_tag=base_tag,
    )
    return classify_step_locality(
        phases,
        rank_mapper=rank_mapper,
        bandwidth_bytes_per_second=nvlink_bandwidth_bytes_per_second,
    )


def render_fabric_phase_goal(
    phase: ClassifiedCommunicationPhase,
    *,
    rank_mapper: RankMapper | None,
    num_goal_ranks: int | None = None,
) -> GoalTrace:
    """Render one isolated phase's cross-node segments to the fabric.

    Local analytic service is intentionally outside this GOAL rank space. This
    artifact contains service for one already ordered graph-owned phase. The
    caller must use the plan's effective dependency boundaries when composing
    several phase results. With no placement mapper, semantic ranks are GOAL
    ranks and every segment in the classified phase stays on the fabric.
    """

    if not isinstance(phase, ClassifiedCommunicationPhase):
        raise TypeError("phase must be a ClassifiedCommunicationPhase")
    if not phase.fabric_segments:
        raise ValueError("a fabric phase needs at least one cross-node segment")
    if rank_mapper is not None and not isinstance(rank_mapper, RankMapper):
        raise TypeError("rank_mapper must be RankMapper or None")
    endpoint_pairs = tuple(
        (
            (
                segment.source_rank
                if rank_mapper is None
                else rank_mapper.goal_rank(segment.source_rank)
            ),
            (
                segment.destination_rank
                if rank_mapper is None
                else rank_mapper.goal_rank(segment.destination_rank)
            ),
            segment,
        )
        for segment in phase.fabric_segments
    )
    for source, destination, _ in endpoint_pairs:
        if source == destination:
            raise ValueError("GOAL mapping collapsed a cross-node segment")
    used_goal_ranks = {
        rank
        for source, destination, _ in endpoint_pairs
        for rank in (source, destination)
    }
    minimum_ranks = max(used_goal_ranks) + 1
    if num_goal_ranks is None:
        mapped_ranks = 0 if rank_mapper is None else rank_mapper.num_goal_ranks()
        num_goal_ranks = max(minimum_ranks, mapped_ranks)
    if num_goal_ranks < minimum_ranks:
        raise ValueError(
            f"num_goal_ranks={num_goal_ranks} cannot contain rank {minimum_ranks - 1}"
        )
    trace = GoalTrace(num_goal_ranks)
    for source, destination, segment in endpoint_pairs:
        send = trace.rank(source).send(
            segment.payload_bytes,
            to=destination,
            tag=segment.tag,
            operation_id=phase.phase.operation_id,
        )
        receive = trace.rank(destination).recv(
            segment.payload_bytes,
            source=source,
            tag=segment.tag,
            operation_id=phase.phase.operation_id,
        )
        trace.record_message(
            GoalMessage(
                operation_id=phase.phase.operation_id,
                source_rank=source,
                destination_rank=destination,
                payload_bytes=segment.payload_bytes,
                tag=segment.tag,
                send_label=send,
                receive_label=receive,
                request_payload_bytes=segment.request_payload_bytes,
            )
        )
    for goal_rank in range(num_goal_ranks):
        if goal_rank not in used_goal_ranks:
            trace.rank(goal_rank).calc(0)
    return trace
