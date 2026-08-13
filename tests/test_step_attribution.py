from fractions import Fraction

import pytest

from simllm.backends import (
    HtsimRequestMetricReducer,
    RequestLatencyTotals,
    StepLocalityOutcome,
    attribute_step,
)
from simllm.core import (
    LatencyAttribution,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    StepResult,
    sampled_request_ids,
)


def _locality(
    step_index: int,
    *,
    composed: tuple[int, ...],
    fabric: tuple[int, ...],
    compute_service_ps: int = 0,
    nvlink_directed_bytes: int = 0,
    nvlink_service_ps: int = 0,
) -> StepLocalityOutcome:
    return StepLocalityOutcome(
        step_index=step_index,
        authority="execution-graph",
        compatibility_fast_path=True,
        total_directed_bytes=1024,
        fabric_directed_bytes=1024,
        nvlink_directed_bytes=nvlink_directed_bytes,
        fabric_segments=1,
        nvlink_segments=0,
        phase_count=len(composed),
        backend_runs=sum(1 for value in fabric if value),
        compute_service_ps=compute_service_ps,
        nvlink_service_ps=nvlink_service_ps,
        nvlink_bandwidth_bytes_per_second=1,
        fabric_phase_service_ps=fabric,
        composed_phase_service_ps=composed,
    )


def _record(
    step_index: int,
    released_at_ps: int,
    scheduled: list[ScheduledRequest],
    *,
    num_sampled: int | None = None,
) -> StepRecord:
    return StepRecord(
        step_index,
        released_at_ps,
        scheduled,
        num_sampled=len(scheduled) if num_sampled is None else num_sampled,
    )


def _result(step_index: int, released_at_ps: int, latency_ps: int) -> StepResult:
    return StepResult(
        step_index=step_index,
        step_latency_ps=latency_ps,
        completed_at_ps=released_at_ps + latency_ps,
    )


def test_attribute_step_partitions_artifacts_without_remainder():
    result = _result(0, 1_000, 300)
    attribution = attribute_step(
        result,
        _locality(0, composed=(100, 150, 50), fabric=(0, 150, 0)),
    )
    assert attribution == LatencyAttribution(kernel_ps=150, collective_ps=150)
    assert attribution.total_ps == result.step_latency_ps
    assert attribution.kv_ps == 0
    assert attribution.dma_ps == 0
    assert attribution.nic_ps == 0
    assert attribution.control_ps == 0


def test_attribute_step_charges_an_unsimulated_step_to_control():
    result = _result(3, 500, 42)
    assert attribute_step(result, None) == LatencyAttribution(control_ps=42)


def test_attribute_step_refuses_a_makespan_its_artifacts_do_not_conserve():
    with pytest.raises(ValueError, match="conserve the step makespan"):
        attribute_step(
            _result(0, 0, 999),
            _locality(0, composed=(100, 150), fabric=(0, 150)),
        )


def test_attribute_step_refuses_mixed_nvlink_locality():
    with pytest.raises(ValueError, match="all-remote level"):
        attribute_step(
            _result(0, 0, 250),
            _locality(
                0,
                composed=(100, 150),
                fabric=(0, 150),
                nvlink_directed_bytes=64,
                nvlink_service_ps=10,
            ),
        )


def test_attribute_step_refuses_a_foreign_step_index():
    with pytest.raises(ValueError, match="another step"):
        attribute_step(
            _result(0, 0, 250),
            _locality(1, composed=(100, 150), fabric=(0, 150)),
        )


def test_reducer_conserves_ttft_and_tpot_for_one_request():
    reducer = HtsimRequestMetricReducer({"r0": 1_000})
    prefill = _record(0, 4_000, [ScheduledRequest("r0", RequestPhase.PREFILL, 8, context_length=8)])
    metrics = reducer.consume(
        prefill,
        _result(0, 4_000, 600),
        _locality(0, composed=(200, 400), fabric=(0, 400)),
    )
    assert [metric.token_index for metric in metrics] == [1]
    first = metrics[0]
    assert first.ttft_ps == 3_600
    assert first.attribution == LatencyAttribution(
        queue_ps=3_000, kernel_ps=200, collective_ps=400
    )
    assert first.attribution.total_ps == first.latency_ps == 3_600
    assert first.tpot_ps is None

    for step_index in (1, 2):
        released_at_ps = 4_600 + (step_index - 1) * 500
        decode = _record(
            step_index,
            released_at_ps,
            [ScheduledRequest("r0", RequestPhase.DECODE, 1, context_length=8 + step_index)],
        )
        reducer.consume(
            decode,
            _result(step_index, released_at_ps, 500),
            _locality(step_index, composed=(100, 400), fabric=(0, 400)),
        )

    (totals,) = reducer.totals()
    assert isinstance(totals, RequestLatencyTotals)
    assert totals.ttft_ps == 3_600
    assert totals.token_count == 3
    assert totals.tpot_ps == Fraction(500)
    assert totals.ttft_attribution.total_ps == totals.ttft_ps
    assert totals.decode_attribution == LatencyAttribution(kernel_ps=200, collective_ps=800)
    assert totals.decode_attribution.total_ps == 1_000
    assert totals.tpot_ps * (totals.token_count - 1) == 1_000


def test_reducer_charges_every_co_scheduled_request_the_whole_step():
    reducer = HtsimRequestMetricReducer({"r0": 0, "r1": 1_000})
    record = _record(
        0,
        2_000,
        [
            ScheduledRequest("r0", RequestPhase.PREFILL, 4, context_length=4),
            ScheduledRequest("r1", RequestPhase.PREFILL, 4, context_length=4),
        ],
    )
    metrics = reducer.consume(
        record,
        _result(0, 2_000, 700),
        _locality(0, composed=(300, 400), fabric=(0, 400)),
    )
    by_id = {metric.request_id: metric for metric in metrics}
    assert by_id["r0"].attribution.kernel_ps == by_id["r1"].attribution.kernel_ps == 300
    assert by_id["r0"].attribution.collective_ps == by_id["r1"].attribution.collective_ps == 400
    assert by_id["r0"].attribution.queue_ps == 2_000
    assert by_id["r1"].attribution.queue_ps == 1_000
    assert by_id["r0"].ttft_ps == 2_700
    assert by_id["r1"].ttft_ps == 1_700


def test_reducer_carries_a_non_sampling_interval_into_the_next_token():
    reducer = HtsimRequestMetricReducer({"r0": 0, "r1": 0})
    chunk = _record(
        0,
        0,
        [
            ScheduledRequest("r0", RequestPhase.PREFILL, 4, context_length=4),
            ScheduledRequest("r1", RequestPhase.DECODE, 1, context_length=9),
        ],
        num_sampled=1,
    )
    assert sampled_request_ids(chunk) == {"r1"}
    metrics = reducer.consume(
        chunk,
        _result(0, 0, 400),
        _locality(0, composed=(100, 300), fabric=(0, 300)),
    )
    assert [metric.request_id for metric in metrics] == ["r1"]

    follow_up = _record(1, 400, [ScheduledRequest("r0", RequestPhase.DECODE, 1, context_length=5)])
    (metric,) = reducer.consume(
        follow_up,
        _result(1, 400, 400),
        _locality(1, composed=(100, 300), fabric=(0, 300)),
    )
    assert metric.request_id == "r0"
    assert metric.ttft_ps == 800
    assert metric.attribution == LatencyAttribution(kernel_ps=200, collective_ps=600)
    assert metric.attribution.total_ps == 800


def test_reducer_refuses_a_request_scheduled_before_its_arrival():
    reducer = HtsimRequestMetricReducer({"r0": 5_000})
    record = _record(0, 1_000, [ScheduledRequest("r0", RequestPhase.PREFILL, 4, context_length=4)])
    with pytest.raises(ValueError, match="predates its arrival"):
        reducer.consume(record, _result(0, 1_000, 100), _locality(0, composed=(100,), fabric=(0,)))


def test_reducer_refuses_an_undeclared_request():
    reducer = HtsimRequestMetricReducer({})
    record = _record(0, 0, [ScheduledRequest("r0", RequestPhase.PREFILL, 4, context_length=4)])
    with pytest.raises(ValueError, match="no declared arrival"):
        reducer.consume(record, _result(0, 0, 100), _locality(0, composed=(100,), fabric=(0,)))


def test_reducer_refuses_a_repeated_step_index():
    reducer = HtsimRequestMetricReducer({"r0": 0})
    record = _record(0, 0, [ScheduledRequest("r0", RequestPhase.PREFILL, 4, context_length=4)])
    reducer.consume(record, _result(0, 0, 100), _locality(0, composed=(100,), fabric=(0,)))
    with pytest.raises(ValueError, match="already been reduced"):
        reducer.consume(record, _result(0, 0, 100), _locality(0, composed=(100,), fabric=(0,)))


def test_reducer_leaves_history_unchanged_when_a_step_is_refused():
    reducer = HtsimRequestMetricReducer({"r0": 0, "r1": 9_000})
    good = _record(0, 0, [ScheduledRequest("r0", RequestPhase.PREFILL, 4, context_length=4)])
    reducer.consume(good, _result(0, 0, 100), _locality(0, composed=(100,), fabric=(0,)))
    before = reducer.totals()
    bad = _record(
        1,
        100,
        [
            ScheduledRequest("r0", RequestPhase.DECODE, 1, context_length=5),
            ScheduledRequest("r1", RequestPhase.PREFILL, 4, context_length=4),
        ],
    )
    with pytest.raises(ValueError, match="predates its arrival"):
        reducer.consume(bad, _result(1, 100, 100), _locality(1, composed=(100,), fabric=(0,)))
    assert reducer.totals() == before


def test_request_latency_totals_refuse_a_partition_that_does_not_conserve():
    with pytest.raises(ValueError, match="TTFT attribution does not conserve"):
        RequestLatencyTotals(
            request_id="r0",
            arrived_at_ps=0,
            first_token_at_ps=100,
            last_token_at_ps=200,
            token_count=2,
            ttft_ps=100,
            tpot_ps=Fraction(100),
            ttft_attribution=LatencyAttribution(kernel_ps=99),
            decode_attribution=LatencyAttribution(kernel_ps=100),
        )
