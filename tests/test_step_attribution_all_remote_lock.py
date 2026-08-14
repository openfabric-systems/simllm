"""Byte-identity lock for the all-remote per-request attribution path.

The literals below were captured from the reducer as it stood at
`e18b9b0`, before per-artifact resource ownership existed, on a step
sequence shaped like the accepted all-remote studies: per-layer compute
artifacts with zero fabric service, collective artifacts that each cost one
backend run, two co-scheduled requests, a step that samples only one of them,
and a scheduler gap ahead of the first step.

The published metric chain of that path must not move. This module replays the
sequence twice: once in the locality shape recorded before the medium
projection existed, and once with the projection the sink now publishes. The
first replay must reproduce the captured literals, and the second must equal
the first field for field.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from simllm.backends import (
    GPU_COMPUTE_MEDIUM,
    NVLINK_MEDIUM,
    HtsimRequestMetricReducer,
    MaskedMediumService,
    MediumAttribution,
    StepLocalityOutcome,
    attribute_step_detail,
)
from simllm.core import (
    LatencyAttribution,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    StepResult,
)

ARRIVALS = {"r0": 0, "r1": 1_000}

#: (step index, release, latency, composed, fabric, scheduled, num_sampled)
STEPS = (
    (
        0,
        3_000,
        27_000,
        (1_000, 8_000, 1_000, 9_000, 1_000, 7_000),
        (0, 8_000, 0, 9_000, 0, 7_000),
        (("r0", RequestPhase.PREFILL, 4, 4), ("r1", RequestPhase.PREFILL, 6, 6)),
        2,
    ),
    (
        1,
        30_000,
        13_000,
        (1_000, 5_000, 1_000, 6_000),
        (0, 5_000, 0, 6_000),
        (("r0", RequestPhase.DECODE, 1, 5), ("r1", RequestPhase.PREFILL, 2, 8)),
        1,
    ),
    (
        2,
        43_000,
        11_000,
        (1_000, 4_000, 1_000, 5_000),
        (0, 4_000, 0, 5_000),
        (("r0", RequestPhase.DECODE, 1, 6), ("r1", RequestPhase.DECODE, 1, 9)),
        2,
    ),
)

#: step index -> captured (kernel_ps, collective_ps)
FROZEN_STEP_ATTRIBUTION = {0: (3_000, 24_000), 1: (2_000, 11_000), 2: (2_000, 9_000)}

#: (step index, request) -> captured (token index, latency, ttft, tpot,
#: queue_ps, kernel_ps, collective_ps)
FROZEN_METRICS = {
    (0, "r0"): (1, 30_000, 30_000, None, 3_000, 3_000, 24_000),
    (0, "r1"): (1, 29_000, 29_000, None, 2_000, 3_000, 24_000),
    (1, "r0"): (2, 13_000, 30_000, Fraction(13_000), 0, 2_000, 11_000),
    (2, "r0"): (3, 11_000, 30_000, Fraction(12_000), 0, 2_000, 9_000),
    (2, "r1"): (2, 24_000, 29_000, Fraction(24_000), 0, 4_000, 20_000),
}

#: request -> captured (ttft, tpot, tokens, first token, last token,
#: ttft partition, decode partition)
FROZEN_TOTALS = {
    "r0": (
        30_000,
        Fraction(12_000),
        3,
        30_000,
        54_000,
        LatencyAttribution(queue_ps=3_000, kernel_ps=3_000, collective_ps=24_000),
        LatencyAttribution(kernel_ps=4_000, collective_ps=20_000),
    ),
    "r1": (
        29_000,
        Fraction(24_000),
        2,
        30_000,
        54_000,
        LatencyAttribution(queue_ps=2_000, kernel_ps=3_000, collective_ps=24_000),
        LatencyAttribution(kernel_ps=4_000, collective_ps=20_000),
    ),
}


def _legacy_locality(step_index, composed, fabric) -> StepLocalityOutcome:
    return StepLocalityOutcome(
        step_index=step_index,
        authority="execution-graph",
        compatibility_fast_path=True,
        total_directed_bytes=4_096,
        fabric_directed_bytes=4_096,
        nvlink_directed_bytes=0,
        fabric_segments=sum(1 for value in fabric if value),
        nvlink_segments=0,
        phase_count=len(composed),
        backend_runs=sum(1 for value in fabric if value),
        compute_service_ps=sum(
            value for value, service in zip(composed, fabric, strict=True) if not service
        ),
        nvlink_service_ps=0,
        nvlink_bandwidth_bytes_per_second=450_000_000_000,
        fabric_phase_service_ps=fabric,
        composed_phase_service_ps=composed,
        artifact_count=len(composed),
    )


def _enriched_locality(outcome: StepLocalityOutcome) -> StepLocalityOutcome:
    """The same all-remote step in the shape the sink now publishes."""

    return replace(
        outcome,
        local_phase_service_ps=tuple(
            0 if service else composed
            for composed, service in zip(
                outcome.composed_phase_service_ps,
                outcome.fabric_phase_service_ps,
                strict=True,
            )
        ),
        base_phase_latency_ps=tuple(0 for _ in outcome.composed_phase_service_ps),
        local_phase_medium=tuple(
            NVLINK_MEDIUM if service else GPU_COMPUTE_MEDIUM
            for service in outcome.fabric_phase_service_ps
        ),
    )


def _replay(enriched: bool):
    reducer = HtsimRequestMetricReducer(ARRIVALS)
    steps = []
    published = []
    for step_index, release, latency, composed, fabric, scheduled, num_sampled in STEPS:
        record = StepRecord(
            step_index,
            release,
            [
                ScheduledRequest(request_id, phase, tokens, context_length=context)
                for request_id, phase, tokens, context in scheduled
            ],
            num_sampled=num_sampled,
        )
        result = StepResult(
            step_index=step_index,
            step_latency_ps=latency,
            completed_at_ps=release + latency,
        )
        locality = _legacy_locality(step_index, composed, fabric)
        if enriched:
            locality = _enriched_locality(locality)
        steps.append(attribute_step_detail(result, locality))
        published.extend(
            (step_index, metric)
            for metric in reducer.consume(record, result, locality)
        )
    return tuple(steps), tuple(published), reducer.totals()


def test_the_all_remote_path_reproduces_its_captured_literals():
    steps, metrics, totals = _replay(enriched=False)

    for index, step in enumerate(steps):
        kernel_ps, collective_ps = FROZEN_STEP_ATTRIBUTION[index]
        assert step.attribution == LatencyAttribution(
            kernel_ps=kernel_ps, collective_ps=collective_ps
        )

    by_key = {
        (step_index, metric.request_id): metric for step_index, metric in metrics
    }
    assert set(by_key) == set(FROZEN_METRICS)
    for key, frozen in FROZEN_METRICS.items():
        metric = by_key[key]
        assert (
            metric.token_index,
            metric.latency_ps,
            metric.ttft_ps,
            metric.tpot_ps,
            metric.attribution.queue_ps,
            metric.attribution.kernel_ps,
            metric.attribution.collective_ps,
        ) == frozen

    rows = {row.request_id: row for row in totals}
    for request_id, frozen in FROZEN_TOTALS.items():
        row = rows[request_id]
        assert (
            row.ttft_ps,
            row.tpot_ps,
            row.token_count,
            row.first_token_at_ps,
            row.last_token_at_ps,
            row.ttft_attribution,
            row.decode_attribution,
        ) == frozen


def test_the_published_medium_projection_changes_no_all_remote_value():
    legacy_steps, legacy_metrics, legacy_totals = _replay(enriched=False)
    enriched_steps, enriched_metrics, enriched_totals = _replay(enriched=True)

    assert enriched_steps == legacy_steps
    assert enriched_metrics == legacy_metrics
    assert enriched_totals == legacy_totals


def test_the_all_remote_path_charges_no_nvlink_and_masks_nothing():
    steps, _, totals = _replay(enriched=True)

    for step in steps:
        assert step.media.nvlink_ps == 0
        assert step.media.co_critical_ps == 0
        assert step.media.collective_base_ps == 0
        assert step.masked == MaskedMediumService()
        assert step.media.fabric_ps == step.attribution.collective_ps
    for row in totals:
        assert row.ttft_media == MediumAttribution(
            queue_ps=row.ttft_attribution.queue_ps,
            kernel_ps=row.ttft_attribution.kernel_ps,
            fabric_ps=row.ttft_attribution.collective_ps,
        )
