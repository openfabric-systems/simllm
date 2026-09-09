"""Declared study inputs, shared by component fixtures and the campaign."""

from __future__ import annotations

from simllm.backends import SerialStepLowererConfig
from simllm.compute import GpuSpec, ModelDims, RooflineProvider, RoutedComputeConfig
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.preplay import (
    PREPLAY_TRACE_SCHEMA,
    ForwardPhase,
    RoutedExperts,
    RoutedLayer,
    RoutedRequest,
    RoutedToken,
)
from simllm.traffic import ExpertPlacementSnapshot, RoutedMoeSupply

GPU = GpuSpec("declared-routed-compute-study", 1_000_000_000, 1_000_000_000_000_000)


def specimen(
    selections: tuple[tuple[int, ...], ...],
    *,
    minimum_ps: int = 0,
    prompt_tokens: int = 1,
) -> SerialStepLowererConfig:
    top_k = len(selections[0])
    dims = ModelDims(
        num_layers=2, hidden_size=16, intermediate_size=32,
        num_heads=1, num_kv_heads=1, head_size=16, vocab_size=64,
        dtype_bytes=2, num_experts=4, top_k=top_k,
        moe_intermediate_size=32, local_num_experts=2,
    )
    requests = []
    for request_index, experts in enumerate(selections):
        tokens = tuple(
            RoutedToken(
                phase=phase, token_index=index, token_id=1,
                layers=tuple(
                    RoutedLayer(layer_index=layer, expert_ids=experts)
                    for layer in range(dims.num_layers)
                ),
            )
            for phase, count in ((ForwardPhase.PREFILL, prompt_tokens), (ForwardPhase.DECODE, 2))
            for index in range(count)
        )
        requests.append(RoutedRequest(
            request_id=f"request-{request_index}",
            prompt_token_count=prompt_tokens,
            output_token_count=3,
            tokens=tokens,
        ))
    routing = RoutedExperts(
        trace_schema=PREPLAY_TRACE_SCHEMA, trace_sha256="a" * 64,
        expert_count=4, top_k=top_k, moe_layer_indices=(0, 1),
        requests=tuple(requests),
    )
    placement = ExpertPlacementSnapshot(
        placement_epoch=0,
        expert_owners=tuple(
            (layer, expert, 0 if expert < 2 else 3)
            for layer in range(2) for expert in range(4)
        ),
    )
    supply = RoutedMoeSupply(
        engine_rank=0, placements=(placement,),
        step_placement_epochs=tuple((index, 0) for index in range(4)),
        routed_experts=routing,
    )
    return SerialStepLowererConfig(
        dims, (0,), ep_ranks=(0, 3), provider=RooflineProvider(efficiency=1),
        gpu=GPU, routed_moe_supply=supply,
        routed_compute=RoutedComputeConfig(minimum_ps, GPU),
    )


def campaign_config(count: int, routing: str, minimum_ps: int) -> SerialStepLowererConfig:
    if count not in (8, 16) or routing not in ("balanced", "hot"):
        raise ValueError("unsupported frozen count or route")
    if minimum_ps not in (0, 5_000_000, 20_000_000):
        raise ValueError("unsupported frozen declared minimum")
    selections = tuple(
        (2,) if routing == "hot" or index >= count // 2 else (0,)
        for index in range(count)
    )
    return specimen(selections, minimum_ps=minimum_ps)


def step(config: SerialStepLowererConfig, index: int, now_ps: int = 0) -> StepRecord:
    assert config.routed_moe_supply is not None
    routing = config.routed_moe_supply.routed_experts
    assert routing is not None
    return StepRecord(
        step_index=index, virtual_time_ps=now_ps,
        scheduled=[
            ScheduledRequest(
                request.request_id,
                RequestPhase.PREFILL if index == 0 else RequestPhase.DECODE,
                request.prompt_token_count if index == 0 else 1,
                context_length=request.prompt_token_count + index,
            )
            for request in routing.requests
        ],
        num_sampled=len(routing.requests),
        sampled_request_ids=[request.request_id for request in routing.requests],
    )
