"""Frozen pure oracles for the analytic-composition collective evaluator."""

from __future__ import annotations

import pytest

from simllm.core import CollectiveWork
from simllm.placement import PlacementManifest, RankMapper, RankPlacement
from simllm.traffic import (
    NCCL_CHANNEL_REGISTRATION_MODEL,
    CollectiveBandwidthCurve,
    CollectiveCommunicationPhase,
    CollectiveLatencyProfile,
    CollectiveRegistrationLedger,
    DirectedCollectiveSegment,
    analytic_collective_price_ps,
    analytic_step_service_ps,
    classify_step_locality,
)


def _profile(*, with_curve: bool = True) -> CollectiveLatencyProfile:
    curves = ()
    if with_curve:
        curves = (
            (
                2,
                CollectiveBandwidthCurve(
                    curve_id="analytic-composition-freeze-w2",
                    points=((4, 1_000_000_000), (8, 2_000_000_000)),
                ),
            ),
        )
    return CollectiveLatencyProfile(
        profile_id="analytic-composition-freeze",
        bandwidth_bytes_per_second=1_000_000_000,
        participant_latency_ps=((2, 10_000), (4, 20_000)),
        source_payload_bytes_min=4,
        source_payload_bytes_max=16,
        propagation_reference_ps=2_000_000,
        bandwidth_curves=curves,
    )


def _classified_phase(
    work: CollectiveWork,
    pairs: tuple[tuple[int, int, int], ...],
    *,
    hosts: tuple[str, ...] | None = None,
    phase_id: str = "phase",
):
    phase = CollectiveCommunicationPhase(
        phase_id=phase_id,
        layer=0,
        participants=work.ranks,
        segments=tuple(
            DirectedCollectiveSegment(source, destination, size, 7)
            for source, destination, size in pairs
        ),
    )
    mapper = None
    if hosts is not None:
        mapper = RankMapper(
            PlacementManifest(
                ranks=[
                    RankPlacement(
                        global_rank=rank,
                        hostname=host,
                        local_rank=sum(1 for prior in hosts[:rank] if prior == host),
                    )
                    for rank, host in enumerate(hosts)
                ]
            )
        )
    return classify_step_locality((phase,), rank_mapper=mapper).phases[0]


def _e1_work_and_phases():
    work = CollectiveWork("all-reduce", (0, 1), 8, "ring")
    phase = _classified_phase(work, ((0, 1, 4), (1, 0, 4)))
    return work, (phase, phase)


def test_e1_width_two_all_remote_ring_prices_4_014_000_ps():
    work, phases = _e1_work_and_phases()

    assert analytic_collective_price_ps(work, phases, _profile()) == 4_014_000


def test_e2_width_four_mixed_ring_prices_12_044_000_ps():
    work = CollectiveWork("all-reduce", (0, 1, 2, 3), 16, "ring")
    phase = _classified_phase(
        work,
        ((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
        hosts=("a", "a", "b", "b"),
    )

    assert analytic_collective_price_ps(work, (phase,) * 6, _profile()) == 12_044_000


def test_all_local_phases_use_the_selected_rate_and_whole_nanoseconds():
    work = CollectiveWork("all-reduce", (0, 1), 8, "ring")
    phase = _classified_phase(
        work,
        ((0, 1, 4), (1, 0, 4)),
        hosts=("a", "a"),
    )

    assert analytic_collective_price_ps(work, (phase, phase), _profile()) == 14_000


def test_e3_width_four_uniform_mixed_all_to_allv_prices_2_028_000_ps():
    work = CollectiveWork("all-to-allv", (0, 1, 2, 3), 4, "pairwise")
    pairs = tuple(
        (source, destination, 4)
        for source in work.ranks
        for destination in work.ranks
        if source != destination
    )
    phase = _classified_phase(work, pairs, hosts=("a", "a", "b", "b"))

    assert analytic_collective_price_ps(work, (phase,), _profile()) == 2_028_000


def test_e4_sparse_all_remote_all_to_allv_prices_2_035_000_ps():
    pairs = ((0, 2, 3), (1, 2, 5), (3, 2, 7))
    work = CollectiveWork(
        "all-to-allv",
        (0, 1, 2, 3),
        0,
        "pairwise",
        pair_payload_bytes=pairs,
    )
    phase = _classified_phase(work, pairs)

    assert analytic_collective_price_ps(work, (phase,), _profile()) == 2_035_000


def test_e5_full_endpoint_bytes_select_the_curve_and_save_4_000_ps():
    work = CollectiveWork("all-to-allv", (0, 1), 8, "pairwise")
    phase = _classified_phase(work, ((0, 1, 8), (1, 0, 8)))

    curve_price = analytic_collective_price_ps(work, (phase,), _profile())
    flat_price = analytic_collective_price_ps(
        work,
        (phase,),
        _profile(with_curve=False),
    )
    assert curve_price == 2_014_000
    assert flat_price == 2_018_000
    assert flat_price - curve_price == 4_000


def test_e6_registration_is_charged_once_then_reused_at_zero_cost():
    work, phases = _e1_work_and_phases()
    ledger = CollectiveRegistrationLedger(model=NCCL_CHANNEL_REGISTRATION_MODEL)
    assert (
        analytic_collective_price_ps(
            work,
            phases,
            _profile(),
            registration_ledger=ledger,
            operation_id="step-0:layer-0:tp",
            step_index=0,
        )
        == 24_014_000
    )
    assert (
        analytic_collective_price_ps(
            work,
            phases,
            _profile(),
            registration_ledger=ledger,
            operation_id="step-1:layer-0:tp",
            step_index=1,
        )
        == 4_014_000
    )
    assert ledger.charged_ps == 20_000_000


def test_empty_pairwise_work_prices_only_registration():
    work = CollectiveWork("all-to-allv", (0, 1), 0, "pairwise")
    ledger = CollectiveRegistrationLedger(model=NCCL_CHANNEL_REGISTRATION_MODEL)

    assert (
        analytic_collective_price_ps(
            work,
            (),
            _profile(),
            registration_ledger=ledger,
            operation_id="step-0:layer-0:dispatch",
        )
        == 20_000_000
    )
    assert (
        analytic_collective_price_ps(
            work,
            (),
            _profile(),
            registration_ledger=ledger,
            operation_id="step-1:layer-0:dispatch",
        )
        == 0
    )


def test_e7_step_service_is_the_comp_75_maximum():
    work, phases = _e1_work_and_phases()
    collective_price = analytic_collective_price_ps(work, phases, _profile())

    assert analytic_step_service_ps(9_000_000, (collective_price,) * 2) == 9_000_000
    assert analytic_step_service_ps(8_000_000, (collective_price,) * 2) == 8_028_000


@pytest.mark.parametrize("profile", (None, "legacy"))
def test_a_resolvable_profile_is_required(profile):
    work, phases = _e1_work_and_phases()

    with pytest.raises(ValueError, match="requires a resolved collective"):
        analytic_collective_price_ps(work, phases, profile)


def test_participant_width_outside_the_profile_envelope_is_refused():
    work = CollectiveWork("all-reduce", (0, 1, 2), 12, "ring")
    phase = _classified_phase(work, ((0, 1, 4), (1, 2, 4), (2, 0, 4)))

    with pytest.raises(ValueError, match="does not support participant count 3"):
        analytic_collective_price_ps(work, (phase,) * 4, _profile())


def test_endpoint_bytes_outside_the_profile_envelope_are_refused():
    work = CollectiveWork("all-reduce", (0, 1), 64, "ring")
    phase = _classified_phase(work, ((0, 1, 32), (1, 0, 32)))
    ledger = CollectiveRegistrationLedger(model=NCCL_CHANNEL_REGISTRATION_MODEL)

    with pytest.raises(ValueError, match="outside.*envelope"):
        analytic_collective_price_ps(
            work,
            (phase, phase),
            _profile(),
            registration_ledger=ledger,
            operation_id="step-0:layer-0:tp",
        )
    assert ledger.charged_ps == 0


@pytest.mark.parametrize(
    "work",
    (
        CollectiveWork("broadcast", (0, 1), 8, "tree"),
        CollectiveWork("all-reduce", (0, 1), 8, "tree"),
        CollectiveWork("all-to-allv", (0, 1), 8, "ring"),
    ),
)
def test_unsupported_collective_shapes_are_refused(work):
    with pytest.raises(ValueError, match="supported collectives"):
        analytic_collective_price_ps(work, (), _profile())
