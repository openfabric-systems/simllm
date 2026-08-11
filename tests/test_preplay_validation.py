"""Tests for the PLAY-5 causal divergence taxonomy."""

from __future__ import annotations

from dataclasses import replace

import pytest

from simllm.preplay.validation import (
    DecisionBoundary,
    DivergenceKind,
    OracleRequestObservation,
    RoutedDecision,
    compare_oracle_requests,
)


def boundary(
    selected: tuple[int, ...],
    *,
    edge: tuple[int, int] | None = None,
    margin: float = 1.0,
) -> DecisionBoundary:
    return DecisionBoundary(
        selected_ids=selected,
        boundary_ids=edge or (selected[-1], selected[-1] + 1),
        margin=margin,
    )


def observation() -> OracleRequestObservation:
    return OracleRequestObservation(
        request_id="r0",
        input_token_ids=(10, 11),
        output_token_ids=(20, 21),
        stop_reason="length-cap",
        token_boundaries=(
            boundary((20,), edge=(20, 30)),
            boundary((21,), edge=(21, 31)),
        ),
        routed_decisions=(
            RoutedDecision(
                phase="prefill",
                token_index=0,
                token_id=10,
                layer_index=0,
                boundary=boundary((0, 1), edge=(1, 2)),
            ),
            RoutedDecision(
                phase="prefill",
                token_index=1,
                token_id=11,
                layer_index=0,
                boundary=boundary((1, 2), edge=(2, 3)),
            ),
            RoutedDecision(
                phase="decode",
                token_index=0,
                token_id=20,
                layer_index=0,
                boundary=boundary((2, 3), edge=(3, 4)),
            ),
        ),
    )


def test_exact_observations_have_no_divergences():
    value = observation()
    result = compare_oracle_requests(
        value,
        value,
        sampling_mode="greedy",
        near_tie_abs_logit=1e-5,
    )
    assert result.passed
    assert result.divergences == ()
    assert result.input_tokens_exact
    assert result.output_tokens_exact
    assert result.output_length_exact
    assert result.stop_reason_exact
    assert result.routing_exact


def test_seeded_sampler_difference_classifies_downstream_cascade():
    left = observation()
    right_routes = tuple(
        replace(
            route,
            token_id=40,
            boundary=boundary((5, 6), edge=(6, 7)),
        )
        if route.phase == "decode"
        else route
        for route in left.routed_decisions
    )
    right = replace(
        left,
        output_token_ids=(40,),
        stop_reason="eos",
        token_boundaries=(boundary((40,), edge=(40, 50)),),
        routed_decisions=right_routes,
    )
    result = compare_oracle_requests(
        left,
        right,
        sampling_mode="seeded-sampling",
        near_tie_abs_logit=1e-5,
    )
    assert result.passed
    assert {item.classification for item in result.divergences} == {
        DivergenceKind.SAMPLER_DIFFERENCE
    }
    assert any(item.field == "output_length" and item.cascade for item in result.divergences)
    assert any(item.field == "stop_reason" and item.cascade for item in result.divergences)
    assert any(item.field == "routing" and item.cascade for item in result.divergences)


def test_seeded_sampler_difference_requires_identical_prompt_tokens():
    left = observation()
    right = replace(
        left,
        input_token_ids=(10, 12),
        output_token_ids=(40, 21),
        token_boundaries=(boundary((40,), edge=(40, 20)), left.token_boundaries[1]),
    )
    result = compare_oracle_requests(
        left,
        right,
        sampling_mode="seeded-sampling",
        near_tie_abs_logit=1e-5,
    )
    assert not result.passed
    assert result.unclassified_count == 2
    assert all(
        item.classification is not DivergenceKind.SAMPLER_DIFFERENCE
        for item in result.divergences
    )


def test_greedy_near_tie_token_flip_is_admissible():
    left = observation()
    right = replace(
        left,
        output_token_ids=(30, 21),
        token_boundaries=(
            boundary((30,), edge=(30, 20), margin=5e-6),
            left.token_boundaries[1],
        ),
    )
    left = replace(
        left,
        token_boundaries=(
            boundary((20,), edge=(20, 30), margin=4e-6),
            left.token_boundaries[1],
        ),
    )
    result = compare_oracle_requests(
        left,
        right,
        sampling_mode="greedy",
        near_tie_abs_logit=1e-5,
    )
    assert result.passed
    assert result.divergences[0].classification is DivergenceKind.NUMERICS_NEAR_TIE_FLIP
    assert not result.divergences[0].cascade


def test_prefill_route_flip_requires_the_frozen_boundary_margin():
    left = observation()
    right_routes = list(left.routed_decisions)
    right_routes[0] = replace(
        right_routes[0],
        boundary=boundary((0, 2), edge=(2, 1), margin=2e-6),
    )
    left_routes = list(left.routed_decisions)
    left_routes[0] = replace(
        left_routes[0],
        boundary=boundary((0, 1), edge=(1, 2), margin=3e-6),
    )
    result = compare_oracle_requests(
        replace(left, routed_decisions=tuple(left_routes)),
        replace(left, routed_decisions=tuple(right_routes)),
        sampling_mode="greedy",
        near_tie_abs_logit=1e-5,
    )
    assert result.passed
    assert len(result.divergences) == 1
    assert result.divergences[0].classification is DivergenceKind.NUMERICS_NEAR_TIE_FLIP
    assert not result.divergences[0].cascade


def test_prefill_route_flip_requires_each_runner_to_straddle_changed_experts():
    left = observation()
    left_routes = list(left.routed_decisions)
    left_routes[0] = replace(
        left_routes[0],
        boundary=boundary((0, 1), edge=(1, 9), margin=3e-6),
    )
    right_routes = list(left.routed_decisions)
    right_routes[0] = replace(
        right_routes[0],
        boundary=boundary((0, 2), edge=(2, 1), margin=2e-6),
    )
    result = compare_oracle_requests(
        replace(left, routed_decisions=tuple(left_routes)),
        replace(left, routed_decisions=tuple(right_routes)),
        sampling_mode="greedy",
        near_tie_abs_logit=1e-5,
    )
    assert not result.passed
    assert result.unclassified_count == 1


def test_large_margin_and_prompt_changes_remain_unclassified():
    left = observation()
    routes = list(left.routed_decisions)
    routes[0] = replace(
        routes[0],
        boundary=boundary((0, 2), edge=(2, 1), margin=1e-2),
    )
    right = replace(left, input_token_ids=(10, 12), routed_decisions=tuple(routes))
    result = compare_oracle_requests(
        left,
        right,
        sampling_mode="greedy",
        near_tie_abs_logit=1e-5,
    )
    assert not result.passed
    assert result.unclassified_count == 2


def test_observation_rejects_missing_token_boundary():
    with pytest.raises(ValueError, match="boundary count"):
        replace(observation(), token_boundaries=())
