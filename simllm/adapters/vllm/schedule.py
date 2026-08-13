"""Source-backed Granite MoE execution observations for vLLM v0.26.0.

The producer in this module translates one already-translated scheduler step
using the active vLLM parallel configuration. It mirrors Granite layer order,
DBO token slicing, the shared compute and communication streams, and the
DeepEP event waits. It contains no overlap amount or compatibility-graph
rewrite. The device runtime decides how much of the source-legal concurrency
is realized.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from simllm.compute import (
    ComputeProvider,
    GpuSpec,
    HostInitiationModel,
    ModelDims,
    step_kernel,
)
from simllm.core import (
    CollectiveWork,
    ComputeWork,
    ExecutionObservations,
    ExecutionOperation,
    OperationCorrelation,
    RequestPhase,
    StepRecord,
)

OBSERVED_SCHEDULE_OFF = "off"
OBSERVED_SCHEDULE_GRANITE_DBO = "granite-dbo"
OBSERVED_SCHEDULE_MODES = (
    OBSERVED_SCHEDULE_OFF,
    OBSERVED_SCHEDULE_GRANITE_DBO,
)
GRANITE_ALL2ALL_BACKEND = "deepep_high_throughput"


@dataclass(frozen=True)
class VllmBatchSlice:
    """One source-ordered vLLM model-forward slice."""

    microbatch: int | None
    request_ids: tuple[str, ...]
    token_count: int

    def __post_init__(self) -> None:
        if self.microbatch is not None and (
            isinstance(self.microbatch, bool)
            or not isinstance(self.microbatch, int)
            or self.microbatch < 0
        ):
            raise ValueError("microbatch must be a nonnegative integer or None")
        if not isinstance(self.request_ids, tuple) or not self.request_ids:
            raise ValueError("request_ids must be a nonempty tuple")
        if any(
            not isinstance(request_id, str) or not request_id.strip()
            for request_id in self.request_ids
        ):
            raise ValueError("request_ids must contain nonblank strings")
        if len(self.request_ids) != len(set(self.request_ids)):
            raise ValueError("request_ids must be unique within a batch slice")
        if (
            isinstance(self.token_count, bool)
            or not isinstance(self.token_count, int)
            or self.token_count <= 0
        ):
            raise ValueError("token_count must be a positive integer")


@dataclass(frozen=True)
class GraniteScheduleTiming:
    """Compute service projected onto the source schedule."""

    represented_compute_ps: int
    provider_compute_ps: int
    provider_bound: str
    uncertainty_fraction: float


def _text_config(vllm_config: Any) -> Any:
    model_config = vllm_config.model_config
    return getattr(model_config, "hf_text_config", None) or getattr(
        model_config, "hf_config", None
    )


def is_granite_moe_config(vllm_config: Any) -> bool:
    """Whether a vLLM config names the audited Granite MoE model family."""

    text_config = _text_config(vllm_config)
    model_type = str(getattr(text_config, "model_type", "") or "").lower()
    architectures = tuple(getattr(text_config, "architectures", ()) or ())
    return model_type == "granitemoe" or "GraniteMoeForCausalLM" in architectures


def _validate_source_configuration(vllm_config: Any, dims: ModelDims) -> None:
    parallel = vllm_config.parallel_config
    if not is_granite_moe_config(vllm_config):
        raise RuntimeError(
            "granite-dbo observations require GraniteMoeForCausalLM from the "
            "audited vLLM v0.26.0 source"
        )
    if int(getattr(parallel, "pipeline_parallel_size", 1) or 1) != 1:
        raise RuntimeError("granite-dbo observations require pipeline_parallel_size=1")
    if int(getattr(parallel, "tensor_parallel_size", 1) or 1) != 1:
        raise RuntimeError(
            "granite-dbo observations currently require tensor_parallel_size=1; "
            "VLLM-23 owns additional parallel schedule shapes"
        )
    if not bool(getattr(parallel, "enable_expert_parallel", False)):
        raise RuntimeError("granite-dbo observations require expert parallelism")
    backend = str(getattr(parallel, "all2all_backend", "") or "")
    if backend != GRANITE_ALL2ALL_BACKEND:
        raise RuntimeError(
            "granite-dbo observations require "
            f"all2all_backend={GRANITE_ALL2ALL_BACKEND}; got {backend!r}"
        )
    if not bool(getattr(parallel, "enable_dbo", False)):
        raise RuntimeError("granite-dbo observations require enable_dbo=True")
    if int(getattr(parallel, "ubatch_size", 0) or 0) > 1:
        raise RuntimeError(
            "explicit ubatch_size schedules are outside the audited DBO path; "
            "VLLM-23 owns additional microbatch shapes"
        )
    if dims.num_experts <= 0 or dims.top_k <= 0 or dims.moe_intermediate_size is None:
        raise RuntimeError("granite-dbo observations require complete MoE ModelDims")
    if dims.local_num_experts <= 0:
        raise RuntimeError("granite-dbo observations require local expert geometry")
    if dims.defaulted_fields:
        raise RuntimeError(
            "granite-dbo observations reject defaulted model geometry: "
            + ", ".join(dims.defaulted_fields)
        )


def vllm_batch_slices(record: StepRecord, parallel_config: Any) -> tuple[VllmBatchSlice, ...]:
    """Mirror vLLM's threshold choice and two-way ordered token split.

    The current producer deliberately accepts the frozen uniform-decode DBO
    shape. A multi-token request can cross vLLM's token split and needs a
    token-interval correlation surface before routed bytes can be partitioned
    without guessing, so that shape is rejected under VLLM-23.
    """

    request_ids = tuple(request.request_id for request in record.scheduled)
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("record.scheduled contains duplicate request IDs")
    total_tokens = record.total_new_tokens
    if total_tokens <= 0:
        return ()

    dp_size = int(getattr(parallel_config, "data_parallel_size", 1) or 1)
    enable_dbo = bool(getattr(parallel_config, "enable_dbo", False))
    uniform_decode = all(
        request.phase is RequestPhase.DECODE and request.num_new_tokens == 1
        for request in record.scheduled
    )
    threshold_name = (
        "dbo_decode_token_threshold"
        if uniform_decode
        else "dbo_prefill_token_threshold"
    )
    threshold_default = 32 if uniform_decode else 512
    threshold = int(getattr(parallel_config, threshold_name, threshold_default) or 0)
    should_ubatch = enable_dbo and dp_size > 1 and total_tokens >= threshold
    if not should_ubatch:
        return (VllmBatchSlice(None, request_ids, total_tokens),)
    if not uniform_decode:
        raise RuntimeError(
            "DBO split across multi-token requests is rejected because request "
            "token intervals are not represented; VLLM-23 owns that completeness"
        )
    padded = record.num_tokens_after_padding
    if padded is not None and padded != total_tokens:
        raise RuntimeError(
            "DBO padding changes the token split but has no request correlation; "
            "VLLM-23 owns padded DBO schedules"
        )

    split_point = total_tokens // 2
    if split_point <= 0 or split_point >= total_tokens:
        raise RuntimeError("DBO requires two nonempty microbatches")
    first = request_ids[:split_point]
    second = request_ids[split_point:]
    if not first or not second:
        raise RuntimeError("DBO request slices must both be nonempty")
    return (
        VllmBatchSlice(0, first, len(first)),
        VllmBatchSlice(1, second, len(second)),
    )


def _represented_compute(
    record: StepRecord,
    dims: ModelDims,
    provider: ComputeProvider,
    gpu: GpuSpec,
    host_model: HostInitiationModel,
) -> tuple[int, int, str, float, Any]:
    num_sampled = record.num_sampled
    if num_sampled is None:
        num_sampled = len(record.scheduled)
    kernel = step_kernel(dims, record, num_sampled=num_sampled)
    fused = provider.estimate(kernel, gpu)
    represented = host_model.represented_estimate(fused, gpu)
    provider_ps = represented.provider_duration_ps
    estimates = provider.estimate_layers(kernel, gpu, dims.num_layers)
    if estimates is None:
        per_layer_ns = represented.duration_ps // (dims.num_layers * 1000)
        represented_ps = per_layer_ns * dims.num_layers * 1000
    else:
        if len(estimates) != dims.num_layers:
            raise ValueError("provider layer breakdown length disagrees with ModelDims")
        durations = [estimate.duration_ps for estimate in estimates]
        if any(duration < 0 for duration in durations):
            raise ValueError("provider layer breakdown durations must be nonnegative")
        if sum(durations) != fused.duration_ps:
            raise ValueError("provider layer breakdown does not conserve fused duration")
        durations[0] += represented.exposed_ps
        represented_ps = (sum(durations) // 1000) * 1000
    if host_model.is_calibrated:
        represented_ps = ((represented.duration_ps + 999) // 1000) * 1000
    return (
        represented_ps,
        provider_ps,
        represented.bound,
        represented.uncertainty_fraction,
        kernel,
    )


def _integer_family_value(family: Any, name: str) -> int:
    value = getattr(family, name)
    integer = int(value)
    if value != integer:
        raise ValueError(f"kernel family {name} demand must be integral")
    return integer


def _phase_weights(kernel: Any, bound: str, num_layers: int) -> tuple[Fraction, Fraction, Fraction]:
    selected = "bytes_moved" if bound == "memory" else "flops"
    by_name = {
        family.name: Fraction(_integer_family_value(family, selected))
        for family in kernel.family_kernels
    }
    pre = sum(
        (
            by_name.get(name, Fraction(0))
            for name in ("attn_gemm", "attn_score", "kv_read")
        ),
        start=Fraction(0),
    ) / num_layers
    expert = by_name.get("mlp_gemm", Fraction(0)) / num_layers
    logits = by_name.get("lm_head", Fraction(0))
    if pre == expert == logits == 0:
        return Fraction(1), Fraction(1), Fraction(1)
    return pre, expert, logits


def _apportion(total: int, weights: Sequence[Fraction]) -> tuple[int, ...]:
    if total < 0:
        raise ValueError("apportion total must be nonnegative")
    denominator = sum(weights, start=Fraction(0))
    if denominator <= 0:
        raise ValueError("apportion weights must have a positive sum")
    cumulative = Fraction(0)
    previous = 0
    values = []
    for index, weight in enumerate(weights):
        if weight < 0:
            raise ValueError("apportion weights must be nonnegative")
        cumulative += weight
        boundary = total if index == len(weights) - 1 else int(total * cumulative / denominator)
        values.append(boundary - previous)
        previous = boundary
    return tuple(values)


def _compute_allocations(
    record: StepRecord,
    dims: ModelDims,
    batch_slices: tuple[VllmBatchSlice, ...],
    provider: ComputeProvider,
    gpu: GpuSpec,
    host_model: HostInitiationModel,
) -> tuple[
    dict[tuple[int, int, str], tuple[int, int, int]],
    tuple[int, int, int],
    GraniteScheduleTiming,
]:
    represented_ps, provider_ps, bound, uncertainty, kernel = _represented_compute(
        record,
        dims,
        provider,
        gpu,
        host_model,
    )
    microbatch_count = len(batch_slices)
    slots = [
        (layer, index, phase)
        for layer in range(dims.num_layers)
        for phase in ("pre-dispatch", "experts")
        for index in range(microbatch_count)
    ]

    def slot_weights(selected_bound: str) -> list[Fraction]:
        pre_weight, expert_weight, logits_weight = _phase_weights(
            kernel, selected_bound, dims.num_layers
        )
        result = [
            (pre_weight if phase == "pre-dispatch" else expert_weight)
            / microbatch_count
            for _, _, phase in slots
        ]
        result.append(logits_weight)
        return result

    durations = _apportion(represented_ps, slot_weights(bound))
    total_flops = int(kernel.flops)
    total_bytes = int(kernel.bytes_moved)
    if kernel.flops != total_flops or kernel.bytes_moved != total_bytes:
        raise ValueError("Granite schedule requires integral compute demand")
    flops = _apportion(total_flops, slot_weights("compute"))
    hbm_bytes = _apportion(total_bytes, slot_weights("memory"))
    allocations = {
        slot: (durations[index], flops[index], hbm_bytes[index])
        for index, slot in enumerate(slots)
    }
    logits = (durations[-1], flops[-1], hbm_bytes[-1])
    timing = GraniteScheduleTiming(
        represented_compute_ps=represented_ps,
        provider_compute_ps=provider_ps,
        provider_bound=bound,
        uncertainty_fraction=uncertainty,
    )
    return allocations, logits, timing


def _operation_prefix(record: StepRecord, microbatch: int | None) -> str:
    index = 0 if microbatch is None else microbatch
    return f"step-{record.step_index}:ubatch-{index}"


def build_granite_execution_observations(
    record: StepRecord,
    dims: ModelDims,
    ep_ranks: Sequence[int],
    batch_slices: Sequence[VllmBatchSlice],
    provider: ComputeProvider,
    gpu: GpuSpec,
    host_model: HostInitiationModel,
) -> tuple[ExecutionObservations, GraniteScheduleTiming]:
    """Build the source-ordered Granite schedule for one translated step."""

    ranks = tuple(ep_ranks)
    slices = tuple(batch_slices)
    if not isinstance(record, StepRecord):
        raise TypeError("record must be a StepRecord")
    if not isinstance(dims, ModelDims):
        raise TypeError("dims must be ModelDims")
    if not ranks or len(set(ranks)) != len(ranks):
        raise ValueError("ep_ranks must be a nonempty unique sequence")
    if any(type(rank) is not int or rank < 0 for rank in ranks):
        raise ValueError("ep_ranks must contain nonnegative integers")
    if not slices:
        raise ValueError("batch_slices must not be empty")
    microbatches = tuple(batch.microbatch for batch in slices)
    expected_microbatches = (
        (None,) if len(slices) == 1 else tuple(range(len(slices)))
    )
    if microbatches != expected_microbatches:
        raise ValueError(
            "batch_slices must be one unbatched slice or contiguous microbatches"
        )
    correlated = tuple(request_id for batch in slices for request_id in batch.request_ids)
    expected = tuple(request.request_id for request in record.scheduled)
    if correlated != expected:
        raise ValueError("batch_slices must partition scheduled requests in source order")
    if len(correlated) != len(set(correlated)):
        raise ValueError("batch_slices must not duplicate request identities")
    tokens_by_request = {
        request.request_id: request.num_new_tokens for request in record.scheduled
    }
    if any(
        batch.token_count
        != sum(tokens_by_request[request_id] for request_id in batch.request_ids)
        for batch in slices
    ):
        raise ValueError("batch_slices token counts disagree with scheduled requests")
    if dims.local_num_experts * len(ranks) != dims.num_experts:
        raise ValueError("ep_ranks disagree with the local expert geometry")

    allocations, logits_allocation, timing = _compute_allocations(
        record,
        dims,
        slices,
        provider,
        gpu,
        host_model,
    )
    operations: list[ExecutionOperation] = []
    previous_combine: dict[int, str] = {}
    uncertainty = timing.uncertainty_fraction

    for layer in range(dims.num_layers):
        for slice_index, batch in enumerate(slices):
            prefix = _operation_prefix(record, batch.microbatch)
            correlation = OperationCorrelation(
                request_ids=batch.request_ids,
                batch_id=f"step-{record.step_index}",
                layer=layer,
                microbatch=batch.microbatch,
            )
            duration, flops, hbm_bytes = allocations[
                (layer, slice_index, "pre-dispatch")
            ]
            for rank in ranks:
                operation_id = f"{prefix}:layer-{layer}:rank-{rank}:pre-dispatch"
                dependencies = (
                    (previous_combine[slice_index],)
                    if slice_index in previous_combine
                    else ()
                )
                operations.append(
                    ExecutionOperation(
                        operation_id=operation_id,
                        rank=rank,
                        logical_queue=f"cuda:{rank}:compute",
                        work=ComputeWork(
                            kernel="vllm:granitemoe:pre-dispatch",
                            config=(("layer", layer), ("microbatch", slice_index)),
                            flops=flops,
                            hbm_bytes=hbm_bytes,
                            nominal_duration_ps=duration,
                            uncertainty_fraction=uncertainty,
                        ),
                        correlation=correlation,
                        participant_local_depends_on=dependencies,
                    )
                )

        for slice_index, batch in enumerate(slices):
            prefix = _operation_prefix(record, batch.microbatch)
            correlation = OperationCorrelation(
                request_ids=batch.request_ids,
                batch_id=f"step-{record.step_index}",
                layer=layer,
                microbatch=batch.microbatch,
            )
            dispatch_id = f"{prefix}:layer-{layer}:ep-dispatch"
            operations.append(
                ExecutionOperation(
                    operation_id=dispatch_id,
                    rank=ranks[0],
                    logical_queue=f"cuda:{ranks[0]}:comm:ep",
                    work=CollectiveWork(
                        collective="all-to-allv",
                        ranks=ranks,
                        payload_bytes=0,
                        channel_hint="dispatch",
                    ),
                    correlation=correlation,
                    participant_local_depends_on=tuple(
                        f"{prefix}:layer-{layer}:rank-{rank}:pre-dispatch"
                        for rank in ranks
                    ),
                )
            )
            duration, flops, hbm_bytes = allocations[(layer, slice_index, "experts")]
            for rank in ranks:
                operations.append(
                    ExecutionOperation(
                        operation_id=f"{prefix}:layer-{layer}:rank-{rank}:experts",
                        rank=rank,
                        logical_queue=f"cuda:{rank}:compute",
                        work=ComputeWork(
                            kernel="vllm:granitemoe:experts",
                            config=(("layer", layer), ("microbatch", slice_index)),
                            flops=flops,
                            hbm_bytes=hbm_bytes,
                            nominal_duration_ps=duration,
                            uncertainty_fraction=uncertainty,
                        ),
                        correlation=correlation,
                        participant_local_depends_on=(dispatch_id,),
                    )
                )

        for slice_index, batch in enumerate(slices):
            prefix = _operation_prefix(record, batch.microbatch)
            correlation = OperationCorrelation(
                request_ids=batch.request_ids,
                batch_id=f"step-{record.step_index}",
                layer=layer,
                microbatch=batch.microbatch,
            )
            combine_id = f"{prefix}:layer-{layer}:ep-combine"
            operations.append(
                ExecutionOperation(
                    operation_id=combine_id,
                    rank=ranks[0],
                    logical_queue=f"cuda:{ranks[0]}:comm:ep",
                    work=CollectiveWork(
                        collective="all-to-allv",
                        ranks=ranks,
                        payload_bytes=0,
                        channel_hint="combine",
                    ),
                    correlation=correlation,
                    participant_local_depends_on=tuple(
                        f"{prefix}:layer-{layer}:rank-{rank}:experts"
                        for rank in ranks
                    ),
                )
            )
            previous_combine[slice_index] = combine_id

    all_request_ids = tuple(request.request_id for request in record.scheduled)
    final_combines = tuple(previous_combine[index] for index in range(len(slices)))
    logits_duration, logits_flops, logits_bytes = logits_allocation
    logits_ids = []
    for rank in ranks:
        operation_id = f"step-{record.step_index}:rank-{rank}:logits"
        operations.append(
            ExecutionOperation(
                operation_id=operation_id,
                rank=rank,
                logical_queue=f"cuda:{rank}:compute",
                work=ComputeWork(
                    kernel="vllm:compute-logits",
                    config=(("sampled", record.num_sampled or 0),),
                    flops=logits_flops,
                    hbm_bytes=logits_bytes,
                    nominal_duration_ps=logits_duration,
                    uncertainty_fraction=uncertainty,
                ),
                participant_local_depends_on=final_combines,
                correlation=OperationCorrelation(
                    request_ids=all_request_ids,
                    batch_id=f"step-{record.step_index}",
                ),
            )
        )
        logits_ids.append(operation_id)

    completion_ids = []
    for slice_index, batch in enumerate(slices):
        prefix = _operation_prefix(record, batch.microbatch)
        operation_id = f"{prefix}:requests-visible"
        operations.append(
            ExecutionOperation(
                operation_id=operation_id,
                rank=ranks[0],
                logical_queue=f"cuda:{ranks[0]}:visibility:{slice_index}",
                work=ComputeWork(
                    kernel="vllm:requests-visible",
                    config=(("microbatch", slice_index),),
                    nominal_duration_ps=0,
                ),
                depends_on=tuple(logits_ids),
                correlation=OperationCorrelation(
                    request_ids=batch.request_ids,
                    batch_id=f"step-{record.step_index}",
                    microbatch=batch.microbatch,
                ),
            )
        )
        completion_ids.append(operation_id)

    return (
        ExecutionObservations(
            operations=tuple(operations),
            completion_operation_ids=tuple(completion_ids),
        ),
        timing,
    )


def observations_from_vllm_step(
    record: StepRecord,
    vllm_config: Any,
    dims: ModelDims,
    ep_ranks: Sequence[int],
    provider: ComputeProvider,
    gpu: GpuSpec,
    host_model: HostInitiationModel,
) -> tuple[ExecutionObservations, GraniteScheduleTiming]:
    """Validate the active source path and emit one step's observations."""

    _validate_source_configuration(vllm_config, dims)
    ranks = tuple(ep_ranks)
    dp_size = int(
        getattr(vllm_config.parallel_config, "data_parallel_size", 1) or 1
    )
    if len(ranks) != dp_size:
        raise RuntimeError(
            "granite-dbo observation ranks must equal data_parallel_size"
        )
    slices = vllm_batch_slices(record, vllm_config.parallel_config)
    return build_granite_execution_observations(
        record,
        dims,
        ranks,
        slices,
        provider,
        gpu,
        host_model,
    )


__all__ = [
    "GRANITE_ALL2ALL_BACKEND",
    "OBSERVED_SCHEDULE_GRANITE_DBO",
    "OBSERVED_SCHEDULE_MODES",
    "OBSERVED_SCHEDULE_OFF",
    "GraniteScheduleTiming",
    "VllmBatchSlice",
    "build_granite_execution_observations",
    "is_granite_moe_config",
    "observations_from_vllm_step",
    "vllm_batch_slices",
]
