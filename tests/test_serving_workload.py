from fractions import Fraction

import numpy as np
import pytest

from simllm.workload import (
    FixedLengths,
    GenerationRequest,
    HashedTokenPrompts,
    ObservedRequestTiming,
    PoissonArrivals,
    TransportRequestObservation,
    realize_generation_requests,
    reduce_transport_observations,
)


class Values:
    def __init__(self, values):
        self.values = values

    def times(self, n):
        del n
        return self.values

    def sample(self, n):
        del n
        return self.values


def request(
    request_id="r0", *, ordinal=0, arrived_at_ps=0, output_token_count=3
):
    return GenerationRequest(
        ordinal=ordinal,
        request_id=request_id,
        arrived_at_ps=arrived_at_ps,
        prompt_token_ids=(1, 2, 3),
        output_token_count=output_token_count,
    )


def test_realize_generation_requests_quantizes_ties_and_preserves_order():
    requests = realize_generation_requests(
        ("r0", "r1", "r2"),
        arrivals=Values([0, 1.5e-12, 1e-6]),
        prompt_lengths=Values(np.array([3, 7, 11], dtype=np.int64)),
        output_lengths=Values(np.array([3, 1, 2], dtype=np.int64)),
        prompt_builder=HashedTokenPrompts(vocab_size=49152, seed=17, first_token_id=10),
    )
    assert tuple(item.ordinal for item in requests) == (0, 1, 2)
    assert tuple(item.arrived_at_ps for item in requests) == (0, 2, 1_000_000)
    assert tuple(len(item.prompt_token_ids) for item in requests) == (3, 7, 11)
    assert tuple(item.output_token_count for item in requests) == (3, 1, 2)
    assert all(10 <= token < 49152 for item in requests for token in item.prompt_token_ids)


def test_realize_generation_requests_is_seed_reproducible_without_moving_shape():
    def build(seed):
        return realize_generation_requests(
            ("r0", "r1", "r2", "r3"),
            arrivals=PoissonArrivals(rate_rps=100, seed=31),
            prompt_lengths=FixedLengths(8),
            output_lengths=FixedLengths(3),
            prompt_builder=HashedTokenPrompts(vocab_size=4096, seed=seed),
        )

    first = build(9)
    repeated = build(9)
    changed = build(10)
    assert first == repeated
    assert [item.arrived_at_ps for item in first] == [item.arrived_at_ps for item in changed]
    assert [len(item.prompt_token_ids) for item in first] == [
        len(item.prompt_token_ids) for item in changed
    ]
    assert any(
        before.prompt_token_ids != after.prompt_token_ids
        for before, after in zip(first, changed, strict=True)
    )


@pytest.mark.parametrize(
    ("ids", "arrivals", "prompts", "outputs", "builder", "match"),
    [
        (("r", "r"), [0, 1], [1, 1], [1, 1], HashedTokenPrompts(10), "unique"),
        (("r",), [], [1], [1], HashedTokenPrompts(10), "returned 0"),
        (("r0", "r1"), [1, 0], [1, 1], [1, 1], HashedTokenPrompts(10), "nondecreasing"),
        (("r",), [-1], [1], [1], HashedTokenPrompts(10), "nonnegative"),
        (("r",), [0], [0], [1], HashedTokenPrompts(10), "positive"),
        (("r",), [0], [2], [1], lambda rid, length, ordinal: [1], "expected 2"),
    ],
)
def test_realize_generation_requests_rejects_ambiguous_plans(
    ids, arrivals, prompts, outputs, builder, match
):
    with pytest.raises((TypeError, ValueError), match=match):
        realize_generation_requests(
            ids,
            arrivals=Values(arrivals),
            prompt_lengths=Values(prompts),
            output_lengths=Values(outputs),
            prompt_builder=builder,
        )


def test_reduce_transport_observations_restores_plan_order_and_exact_metrics():
    requests = (
        request("r0", ordinal=0, arrived_at_ps=0, output_token_count=3),
        request("r1", ordinal=1, arrived_at_ps=1_000, output_token_count=1),
    )
    observations = (
        TransportRequestObservation("r1", 1_200, (2_000,)),
        TransportRequestObservation("r0", 100, (1_100, 1_300, 1_500)),
    )
    assert reduce_transport_observations(requests, observations) == (
        ObservedRequestTiming(
            ordinal=0,
            request_id="r0",
            arrived_at_ps=0,
            submitted_at_ps=100,
            first_token_at_ps=1_100,
            completed_at_ps=1_500,
            output_token_count=3,
            submission_lateness_ps=100,
            ttft_ps=1_100,
            tpot_ps=Fraction(200),
        ),
        ObservedRequestTiming(
            ordinal=1,
            request_id="r1",
            arrived_at_ps=1_000,
            submitted_at_ps=1_200,
            first_token_at_ps=2_000,
            completed_at_ps=2_000,
            output_token_count=1,
            submission_lateness_ps=200,
            ttft_ps=1_000,
            tpot_ps=None,
        ),
    )


@pytest.mark.parametrize(
    ("observations", "match"),
    [
        ((), "missing"),
        (
            (
                TransportRequestObservation("r0", 0, (1, 2, 3)),
                TransportRequestObservation("r0", 0, (1, 2, 3)),
            ),
            "duplicate",
        ),
        ((TransportRequestObservation("foreign", 0, (1, 2, 3)),), "foreign"),
        ((TransportRequestObservation("r0", 0, (1, 2)),), "observed 2"),
        ((TransportRequestObservation("r0", -1, (1, 2, 3)),), "nonnegative"),
        ((TransportRequestObservation("r0", 2, (1, 2, 3)),), "before submission"),
        ((TransportRequestObservation("r0", 0, (1, 3, 2)),), "decreasing"),
    ],
)
def test_reduce_transport_observations_rejects_invalid_evidence(observations, match):
    with pytest.raises((TypeError, ValueError), match=match):
        reduce_transport_observations((request(),), observations)
