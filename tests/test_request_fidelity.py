"""Per-request MoE traffic projection and fidelity-gate tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from simllm.backends import SerialStepLowerer, SerialStepLowererConfig
from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
from simllm.core import (
    CollectiveWork,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    execution_graph_from_json,
    execution_graph_to_json,
)
from simllm.goal import GoalMessage
from simllm.preplay import (
    PREPLAY_TRACE_SCHEMA,
    ForwardPhase,
    RoutedExperts,
    RoutedLayer,
    RoutedRequest,
    RoutedToken,
)
from simllm.traffic import (
    ExpertPlacementSnapshot,
    RequestFidelityError,
    RoutedMoeSupply,
    compare_request_moe_fidelity,
    project_execution_graph_goal,
    render_step_goal,
    step_moe_alltoalls,
    validate_request_moe_fidelity,
)

DIMS = ModelDims(
    num_layers=2,
    hidden_size=4,
    intermediate_size=8,
    num_heads=2,
    num_kv_heads=2,
    head_size=2,
    vocab_size=16,
    dtype_bytes=2,
    num_experts=4,
    top_k=2,
    moe_intermediate_size=4,
    local_num_experts=2,
)

ROUTES = {
    "alpha": ((0, 2), (0, 1)),
    "beta": ((0, 1), (2, 3)),
    "gamma": ((2, 3), (1, 3)),
}

GOAL_HASHES = {
    (0, 1): "1eb2bbff8a981523b5f6733420aa9d5d3509aa473ed991409b8d455e619e5864",
    (0, 2): "78a8e80589b156374b965634dd82251931219398c1e2cf2454b06cbe3629916c",
    (0, 3): "8e38bf44631b9f3d7020452886552502fa567ec44559d05b5401a5dbbc825ab6",
    (1, 1): "8c1738dbd01f320b0f5f005b9ea6acd19145c77db67af89eaac4a78219d494de",
    (1, 2): "3023c39e472980ed6c689410a21fa626db3a73cf8a3d83bde425d8d41cfd4361",
    (1, 3): "60cb32ca80a57d03b627de51d01fd292c0e87da3ec1482760faa8d304b075440",
}

PERMUTATION_ERRORS = {
    (0, 2): (12, 96, -16),
    (0, 3): (12, 96, -16),
    (1, 2): (4, 32, 16),
    (1, 3): (4, 32, 16),
}


class FixedProvider(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=2_000, bound="measured")


def _token(token_id: int, routes: tuple[tuple[int, int], ...]) -> RoutedToken:
    return RoutedToken(
        phase=ForwardPhase.PREFILL,
        token_index=0,
        token_id=token_id,
        layers=tuple(
            RoutedLayer(layer_index=layer, expert_ids=experts)
            for layer, experts in enumerate(routes)
        ),
    )


def _supply() -> RoutedMoeSupply:
    requests = tuple(
        RoutedRequest(
            request_id=request_id,
            prompt_token_count=1,
            output_token_count=1,
            tokens=(_token(10 + index, routes),),
        )
        for index, (request_id, routes) in enumerate(ROUTES.items())
    )
    routed = RoutedExperts(
        trace_schema=PREPLAY_TRACE_SCHEMA,
        trace_sha256="a" * 64,
        expert_count=4,
        top_k=2,
        moe_layer_indices=(0, 1),
        requests=requests,
    )
    epoch0 = ExpertPlacementSnapshot(
        placement_epoch=0,
        expert_owners=tuple(
            (layer, expert, 0 if expert < 2 else 1) for layer in range(2) for expert in range(4)
        ),
    )
    epoch1_owners = (
        {0: 0, 1: 1, 2: 0, 3: 1},
        {0: 0, 1: 1, 2: 1, 3: 0},
    )
    epoch1 = ExpertPlacementSnapshot(
        placement_epoch=1,
        expert_owners=tuple(
            (layer, expert, epoch1_owners[layer][expert])
            for layer in range(2)
            for expert in range(4)
        ),
    )
    return RoutedMoeSupply(
        routed_experts=routed,
        placements=(epoch0, epoch1),
        step_placement_epochs=tuple(
            (epoch * 3 + count - 1, epoch) for epoch in (0, 1) for count in (1, 2, 3)
        ),
    )


def _record(epoch: int, count: int) -> StepRecord:
    return StepRecord(
        step_index=epoch * 3 + count - 1,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                request_id,
                RequestPhase.PREFILL,
                1,
                context_length=1,
            )
            for request_id in tuple(ROUTES)[:count]
        ],
        num_sampled=count,
    )


def _swap_requests(
    messages: tuple[GoalMessage, ...],
    left: str,
    right: str,
) -> tuple[GoalMessage, ...]:
    result = []
    for message in messages:
        partition = tuple(
            sorted(
                (
                    right if request_id == left else left if request_id == right else request_id,
                    size,
                )
                for request_id, size in message.request_payload_bytes
            )
        )
        result.append(replace(message, request_payload_bytes=partition))
    return tuple(result)


@pytest.mark.parametrize(
    ("epoch", "count"),
    [(epoch, count) for epoch in (0, 1) for count in (1, 2, 3)],
)
def test_direct_and_graph_projection_preserve_request_identity_and_physical_messages(
    epoch, count
):
    supply = _supply()
    record = _record(epoch, count)
    direct = render_step_goal(
        record,
        DIMS,
        (0,),
        1,
        ep_ranks=(0, 1),
        routed_supply=supply,
    )
    direct_report = compare_request_moe_fidelity(
        record,
        DIMS,
        (0, 1),
        supply,
        direct.messages,
    )
    lowerer = SerialStepLowerer(
        SerialStepLowererConfig(
            DIMS,
            (0,),
            ep_ranks=(0, 1),
            provider=FixedProvider(),
            routed_moe_supply=supply,
        )
    )
    graph = lowerer.lower(record)
    replay_graph = execution_graph_from_json(execution_graph_to_json(graph))
    projection = project_execution_graph_goal(replay_graph)
    graph_messages = tuple(
        message
        for artifact in projection.artifacts
        for message in artifact.trace.messages
    )
    graph_report = compare_request_moe_fidelity(
        record,
        DIMS,
        (0, 1),
        supply,
        graph_messages,
    )

    assert direct_report.per_request_matches
    assert graph_report.per_request_matches
    assert graph_report.observed_request_rows == direct_report.observed_request_rows
    def physical_rows(messages):
        return tuple(
            (
                message.operation_id,
                message.source_rank,
                message.destination_rank,
                message.payload_bytes,
                message.tag,
                message.request_payload_bytes,
            )
            for message in messages
        )

    assert physical_rows(graph_messages) == physical_rows(direct.messages)
    assert hashlib.sha256(direct.render().encode()).hexdigest() == GOAL_HASHES[(epoch, count)]
    collectives = [
        operation.work
        for operation in replay_graph.operations
        if isinstance(operation.work, CollectiveWork)
    ]
    assert all(work.request_pair_payload_bytes for work in collectives)


@pytest.mark.parametrize(
    ("epoch", "count"),
    [(epoch, count) for epoch in (0, 1) for count in (2, 3)],
)
def test_request_permutation_preserves_aggregate_but_fails_fidelity(epoch, count):
    supply = _supply()
    record = _record(epoch, count)
    trace = render_step_goal(
        record,
        DIMS,
        (0,),
        1,
        ep_ranks=(0, 1),
        routed_supply=supply,
    )
    permuted = _swap_requests(trace.messages, "alpha", "beta")

    report = compare_request_moe_fidelity(
        record,
        DIMS,
        (0, 1),
        supply,
        permuted,
    )

    expected_mismatches, expected_l1, alpha_delta = PERMUTATION_ERRORS[(epoch, count)]
    assert report.aggregate_matches
    assert report.aggregate_mismatch_count == 0
    assert not report.per_request_matches
    assert report.mismatch_count == expected_mismatches
    assert report.l1_error_bytes == expected_l1
    assert dict(report.request_delta_bytes)["alpha"] == alpha_delta
    with pytest.raises(RequestFidelityError, match="request fidelity failed"):
        validate_request_moe_fidelity(
            record,
            DIMS,
            (0, 1),
            supply,
            permuted,
        )


def test_routed_operation_partition_sums_to_existing_aggregate_table():
    operations = step_moe_alltoalls(
        _record(0, 3),
        DIMS,
        (0, 1),
        routed_supply=_supply(),
    )

    for operation in operations:
        totals: dict[tuple[int, int], int] = {}
        for _, source, destination, size in operation.request_pair_payload_bytes:
            pair = (source, destination)
            totals[pair] = totals.get(pair, 0) + size
        assert (
            tuple(
                (source, destination, size)
                for (source, destination), size in sorted(totals.items())
            )
            == operation.pair_payload_bytes
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_fidelity_gate_rejects_missing_or_extra_attributed_message(mutation):
    supply = _supply()
    record = _record(0, 2)
    trace = render_step_goal(
        record,
        DIMS,
        (0,),
        1,
        ep_ranks=(0, 1),
        routed_supply=supply,
    )
    if mutation == "missing":
        messages = trace.messages[:-1]
    else:
        messages = trace.messages + (
            replace(trace.messages[0], operation_id="unexpected-operation"),
        )

    report = compare_request_moe_fidelity(
        record,
        DIMS,
        (0, 1),
        supply,
        messages,
    )

    assert not report.aggregate_matches
    assert not report.per_request_matches
    with pytest.raises(RequestFidelityError, match="request fidelity failed"):
        validate_request_moe_fidelity(
            record,
            DIMS,
            (0, 1),
            supply,
            messages,
        )
