"""Deterministic in-process composition of collective communication service."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from simllm.core.execution import CollectiveWork
from simllm.traffic.collective_latency import (
    PICOSECONDS_PER_SECOND,
    CollectiveLatencyProfile,
    critical_collective_endpoint_bytes,
    resolve_collective_latency_profile,
)
from simllm.traffic.collective_registration import CollectiveRegistrationLedger
from simllm.traffic.locality import ClassifiedCommunicationPhase

NANOSECONDS_PER_SECOND = 1_000_000_000
PICOSECONDS_PER_NANOSECOND = 1_000


def _require_nonnegative_int(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _ceil_service_units(endpoint_bytes: int, units_per_second: int, rate: Fraction) -> int:
    numerator = endpoint_bytes * units_per_second * rate.denominator
    return (numerator + rate.numerator - 1) // rate.numerator


def _validate_phases(
    work: CollectiveWork,
    phases: Sequence[ClassifiedCommunicationPhase],
    endpoint_bytes: int,
) -> tuple[ClassifiedCommunicationPhase, ...]:
    if not isinstance(phases, Sequence):
        raise TypeError("phases must be a sequence of classified communication phases")
    normalized = tuple(phases)
    if any(not isinstance(phase, ClassifiedCommunicationPhase) for phase in normalized):
        raise TypeError("phases must contain classified communication phases")

    key = (work.collective, work.algorithm_hint)
    if key == ("all-reduce", "ring"):
        if endpoint_bytes == 0:
            raise ValueError("analytic ring all-reduce requires a positive payload")
        expected_count = 2 * (len(work.ranks) - 1)
    elif key == ("all-to-allv", "pairwise"):
        expected_count = 0 if endpoint_bytes == 0 else 1
    else:
        raise ValueError(
            "supported collectives are ring all-reduce and pairwise all-to-allv"
        )
    if len(normalized) != expected_count:
        raise ValueError(
            f"analytic {work.collective} with algorithm {work.algorithm_hint!r} "
            f"requires {expected_count} communication phases, got {len(normalized)}"
        )
    if any(phase.phase.participants != work.ranks for phase in normalized):
        raise ValueError("analytic communication phase participants disagree with work.ranks")
    return normalized


def analytic_collective_price_ps(
    work: CollectiveWork,
    phases: Sequence[ClassifiedCommunicationPhase],
    profile: str | CollectiveLatencyProfile | None,
    *,
    registration_ledger: CollectiveRegistrationLedger | None = None,
    operation_id: str | None = None,
    step_index: int | None = None,
) -> int:
    """Price one supported collective from its semantic work and locality phases.

    The optional registration ledger remains the sole mutable identity
    authority. All profile, shape and phase validation finishes before the
    evaluator asks it to charge one semantic operation.
    """

    if not isinstance(work, CollectiveWork):
        raise TypeError("work must be a CollectiveWork")
    if registration_ledger is not None and not isinstance(
        registration_ledger,
        CollectiveRegistrationLedger,
    ):
        raise TypeError(
            "registration_ledger must be a CollectiveRegistrationLedger or None"
        )
    if registration_ledger is not None and (
        not isinstance(operation_id, str) or not operation_id.strip()
    ):
        raise ValueError(
            "operation_id must be a nonblank string when registration is supplied"
        )
    resolved_profile = resolve_collective_latency_profile(profile)
    if resolved_profile is None:
        raise ValueError(
            "analytic composition requires a resolved collective latency profile"
        )

    endpoint_bytes = critical_collective_endpoint_bytes(work)
    participant_count = len(work.ranks)
    base_latency_ps = resolved_profile.base_latency_ps(participant_count)
    classified_phases = _validate_phases(work, phases, endpoint_bytes)
    price_ps = 0
    if endpoint_bytes > 0:
        rate = resolved_profile.effective_bandwidth_bytes_per_second(
            participant_count,
            endpoint_bytes,
        )
        price_ps = base_latency_ps
        for phase in classified_phases:
            local_service_ps = PICOSECONDS_PER_NANOSECOND * _ceil_service_units(
                phase.nvlink_peak_endpoint_bytes,
                NANOSECONDS_PER_SECOND,
                rate,
            )
            fabric_service_ps = 0
            if phase.fabric_segments:
                fabric_service_ps = (
                    resolved_profile.propagation_reference_ps
                    + _ceil_service_units(
                        phase.fabric_peak_endpoint_bytes,
                        PICOSECONDS_PER_SECOND,
                        rate,
                    )
                )
            price_ps += max(local_service_ps, fabric_service_ps)
    if registration_ledger is not None:
        price_ps += registration_ledger.charge_collective(
            work,
            operation_id,
            step_index=step_index,
        )
    return price_ps


def analytic_step_service_ps(
    compute_service_ps: int,
    collective_prices_ps: Sequence[int],
) -> int:
    """Apply the COMP-75 maximum to compute and serial collective service."""

    compute_service_ps = _require_nonnegative_int(
        "compute_service_ps",
        compute_service_ps,
    )
    if not isinstance(collective_prices_ps, Sequence):
        raise TypeError("collective_prices_ps must be a sequence")
    prices = tuple(
        _require_nonnegative_int(f"collective_prices_ps[{index}]", price)
        for index, price in enumerate(collective_prices_ps)
    )
    return max(compute_service_ps, sum(prices))


__all__ = [
    "analytic_collective_price_ps",
    "analytic_step_service_ps",
]
