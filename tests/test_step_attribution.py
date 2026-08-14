from dataclasses import replace
from fractions import Fraction

import pytest

from simllm.backends import (
    GPU_COMPUTE_MEDIUM,
    NVLINK_MEDIUM,
    HtsimRequestMetricReducer,
    MaskedMediumService,
    MediumAttribution,
    RequestLatencyTotals,
    StepLocalityOutcome,
    attribute_step,
    attribute_step_detail,
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
    local: tuple[int, ...] = (),
    base: tuple[int, ...] = (),
    medium: tuple[str, ...] = (),
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
        local_phase_service_ps=local,
        base_phase_latency_ps=base,
        local_phase_medium=medium,
    )


#: One step that mixes every ownership case the composition can produce:
#: a compute artifact, a fully intra-node collective, a mixed artifact the
#: fabric decides, a mixed artifact NVLink decides, and a fabric artifact
#: carrying a semantic base latency.
_MIXED_COMPOSED = (1_000, 5_000, 9_000, 12_000, 4_000)
_MIXED_FABRIC = (0, 0, 9_000, 4_000, 3_000)
_MIXED_LOCAL = (1_000, 5_000, 2_000, 12_000, 0)
_MIXED_BASE = (0, 0, 0, 0, 1_000)
_MIXED_MEDIUM = (
    GPU_COMPUTE_MEDIUM,
    NVLINK_MEDIUM,
    NVLINK_MEDIUM,
    NVLINK_MEDIUM,
    NVLINK_MEDIUM,
)
_MIXED_LATENCY_PS = sum(_MIXED_COMPOSED)


def _mixed_locality(step_index: int, **overrides) -> StepLocalityOutcome:
    fields = {
        "composed": _MIXED_COMPOSED,
        "fabric": _MIXED_FABRIC,
        "local": _MIXED_LOCAL,
        "base": _MIXED_BASE,
        "medium": _MIXED_MEDIUM,
        "nvlink_directed_bytes": 4_096,
        "nvlink_service_ps": sum(
            local
            for local, medium in zip(_MIXED_LOCAL, _MIXED_MEDIUM, strict=True)
            if medium == NVLINK_MEDIUM
        ),
    }
    fields.update(overrides)
    return _locality(step_index, **fields)


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


def test_attribute_step_refuses_nvlink_service_without_the_medium_projection():
    with pytest.raises(ValueError, match="requires the per-artifact medium projection"):
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


def test_attribute_step_charges_each_artifact_to_the_medium_that_realized_it():
    result = _result(0, 0, _MIXED_LATENCY_PS)
    step = attribute_step_detail(result, _mixed_locality(0))

    assert step.media == MediumAttribution(
        kernel_ps=1_000,
        nvlink_ps=17_000,
        fabric_ps=12_000,
        collective_base_ps=1_000,
    )
    assert step.media.total_ps == result.step_latency_ps
    assert step.attribution == LatencyAttribution(kernel_ps=1_000, collective_ps=30_000)
    assert step.attribution.total_ps == result.step_latency_ps
    assert step.media.collective_ps == step.attribution.collective_ps


def test_masked_service_is_the_medium_the_artifact_did_not_wait_for():
    step = attribute_step_detail(
        _result(0, 0, _MIXED_LATENCY_PS),
        _mixed_locality(0),
    )

    #: artifact 2 hid 2,000 ps of NVLink service under the fabric, artifact 3
    #: hid 4,000 ps of fabric service under NVLink, and neither is latency
    assert step.masked == MaskedMediumService(nvlink_ps=2_000, fabric_ps=4_000)
    assert not hasattr(step.masked, "total_ps")
    assert step.media.total_ps == _MIXED_LATENCY_PS


def test_attribute_step_calls_an_exact_tie_co_critical():
    step = attribute_step_detail(
        _result(0, 0, 6_000),
        _locality(
            0,
            composed=(6_000,),
            fabric=(6_000,),
            local=(6_000,),
            base=(0,),
            medium=(NVLINK_MEDIUM,),
            nvlink_directed_bytes=64,
            nvlink_service_ps=6_000,
        ),
    )

    assert step.media == MediumAttribution(co_critical_ps=6_000)
    assert step.media.nvlink_ps == step.media.fabric_ps == 0
    assert step.masked == MaskedMediumService()
    assert step.attribution == LatencyAttribution(collective_ps=6_000)


def test_attribute_step_ignores_an_artifact_that_realized_no_service():
    step = attribute_step_detail(
        _result(0, 0, 1_000),
        _locality(
            0,
            composed=(1_000, 0),
            fabric=(0, 0),
            local=(1_000, 0),
            base=(0, 0),
            medium=(GPU_COMPUTE_MEDIUM, NVLINK_MEDIUM),
        ),
    )

    assert step.media == MediumAttribution(kernel_ps=1_000)


@pytest.mark.parametrize("index", [0, 1])
def test_attribute_step_catches_a_swapped_artifact_medium(index):
    swapped = list(_MIXED_MEDIUM)
    swapped[index] = (
        NVLINK_MEDIUM if swapped[index] == GPU_COMPUTE_MEDIUM else GPU_COMPUTE_MEDIUM
    )
    with pytest.raises(ValueError, match="conserve the step's published"):
        attribute_step_detail(
            _result(0, 0, _MIXED_LATENCY_PS),
            _mixed_locality(0, medium=tuple(swapped)),
        )


def test_attribute_step_refuses_a_compute_artifact_that_carries_fabric_service():
    swapped = list(_MIXED_MEDIUM)
    swapped[2] = GPU_COMPUTE_MEDIUM
    with pytest.raises(ValueError, match="cannot carry fabric service"):
        attribute_step_detail(
            _result(0, 0, _MIXED_LATENCY_PS),
            _mixed_locality(0, medium=tuple(swapped)),
        )


def test_attribute_step_refuses_a_compute_artifact_that_carries_a_base_latency():
    with pytest.raises(ValueError, match="cannot carry a base latency"):
        attribute_step_detail(
            _result(0, 0, 3_000),
            _locality(
                0,
                composed=(3_000,),
                fabric=(0,),
                local=(2_000,),
                base=(1_000,),
                medium=(GPU_COMPUTE_MEDIUM,),
            ),
        )


def test_attribute_step_refuses_a_composed_service_its_terms_do_not_produce():
    with pytest.raises(ValueError, match="disagrees with its base and medium terms"):
        attribute_step_detail(
            _result(0, 0, 9_000),
            _locality(
                0,
                composed=(9_000,),
                fabric=(4_000,),
                local=(2_000,),
                base=(0,),
                medium=(NVLINK_MEDIUM,),
                nvlink_directed_bytes=64,
                nvlink_service_ps=2_000,
            ),
        )


def test_attribute_step_refuses_an_unsupported_local_service_medium():
    with pytest.raises(ValueError, match="unsupported local service medium"):
        attribute_step_detail(
            _result(0, 0, 1_000),
            _locality(
                0,
                composed=(1_000,),
                fabric=(0,),
                local=(1_000,),
                base=(0,),
                medium=("pcie",),
            ),
        )


def test_attribute_step_refuses_a_medium_projection_of_another_length():
    with pytest.raises(ValueError, match="disagrees in length"):
        attribute_step_detail(
            _result(0, 0, 1_000),
            _locality(
                0,
                composed=(1_000,),
                fabric=(0,),
                local=(1_000, 1_000),
                base=(0, 0),
                medium=(GPU_COMPUTE_MEDIUM, GPU_COMPUTE_MEDIUM),
            ),
        )


def test_the_published_projection_reproduces_the_all_remote_shape_exactly():
    #: the same all-remote step, once in the shape recorded before the
    #: projection existed and once with the projection the sink now publishes
    legacy = _locality(0, composed=(1_000, 8_000, 1_000, 9_000), fabric=(0, 8_000, 0, 9_000))
    enriched = replace(
        legacy,
        local_phase_service_ps=(1_000, 0, 1_000, 0),
        base_phase_latency_ps=(0, 0, 0, 0),
        local_phase_medium=(
            GPU_COMPUTE_MEDIUM,
            NVLINK_MEDIUM,
            GPU_COMPUTE_MEDIUM,
            NVLINK_MEDIUM,
        ),
    )
    result = _result(0, 0, 19_000)

    assert attribute_step_detail(result, enriched) == attribute_step_detail(
        result, legacy
    )
    assert attribute_step_detail(result, legacy).media == MediumAttribution(
        kernel_ps=2_000, fabric_ps=17_000
    )
    assert attribute_step(result, legacy) == LatencyAttribution(
        kernel_ps=2_000, collective_ps=17_000
    )


def test_a_step_the_sink_did_not_simulate_partitions_both_ways():
    step = attribute_step_detail(_result(3, 500, 42), None)

    assert step.attribution == LatencyAttribution(control_ps=42)
    assert step.media == MediumAttribution(control_ps=42)
    assert step.masked == MaskedMediumService()


def test_reducer_reaches_ttft_and_tpot_for_an_nvlink_bearing_step():
    reducer = HtsimRequestMetricReducer({"r0": 0})
    prefill = _record(
        0,
        0,
        [ScheduledRequest("r0", RequestPhase.PREFILL, 4, context_length=4)],
    )
    (first,) = reducer.consume(prefill, _result(0, 0, _MIXED_LATENCY_PS), _mixed_locality(0))
    assert first.ttft_ps == _MIXED_LATENCY_PS
    assert first.attribution == LatencyAttribution(kernel_ps=1_000, collective_ps=30_000)

    decode = _record(
        1,
        _MIXED_LATENCY_PS,
        [ScheduledRequest("r0", RequestPhase.DECODE, 1, context_length=5)],
    )
    reducer.consume(
        decode,
        _result(1, _MIXED_LATENCY_PS, _MIXED_LATENCY_PS),
        _mixed_locality(1),
    )

    (totals,) = reducer.totals()
    assert totals.ttft_media == MediumAttribution(
        kernel_ps=1_000,
        nvlink_ps=17_000,
        fabric_ps=12_000,
        collective_base_ps=1_000,
    )
    assert totals.decode_media == totals.ttft_media
    assert totals.ttft_media.total_ps == totals.ttft_ps == _MIXED_LATENCY_PS
    assert totals.tpot_ps == Fraction(_MIXED_LATENCY_PS)
    #: both media are on the same request's critical path, under their own
    #: names, and they roll up to the coarse collective component
    assert totals.ttft_media.nvlink_ps > 0
    assert totals.ttft_media.fabric_ps > 0
    assert totals.ttft_media.collective_ps == totals.ttft_attribution.collective_ps


def test_reducer_carries_the_medium_partition_across_a_step_without_a_token():
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
    reducer.consume(chunk, _result(0, 0, _MIXED_LATENCY_PS), _mixed_locality(0))
    follow_up = _record(
        1,
        _MIXED_LATENCY_PS,
        [ScheduledRequest("r0", RequestPhase.DECODE, 1, context_length=5)],
    )
    (metric,) = reducer.consume(
        follow_up,
        _result(1, _MIXED_LATENCY_PS, _MIXED_LATENCY_PS),
        _mixed_locality(1),
    )

    assert metric.request_id == "r0"
    totals = {row.request_id: row for row in reducer.totals()}
    assert totals["r0"].ttft_media == MediumAttribution(
        kernel_ps=2_000,
        nvlink_ps=34_000,
        fabric_ps=24_000,
        collective_base_ps=2_000,
    )
    assert totals["r0"].ttft_media.total_ps == totals["r0"].ttft_ps


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
            ttft_media=MediumAttribution(kernel_ps=99),
            decode_media=MediumAttribution(kernel_ps=100),
        )


def test_request_latency_totals_refuse_a_medium_partition_that_disagrees():
    with pytest.raises(ValueError, match="TTFT medium components do not roll up"):
        RequestLatencyTotals(
            request_id="r0",
            arrived_at_ps=0,
            first_token_at_ps=100,
            last_token_at_ps=200,
            token_count=2,
            ttft_ps=100,
            tpot_ps=Fraction(100),
            ttft_attribution=LatencyAttribution(collective_ps=100),
            decode_attribution=LatencyAttribution(collective_ps=100),
            #: same total, wrong owner: kernel time is not collective time
            ttft_media=MediumAttribution(kernel_ps=100),
            decode_media=MediumAttribution(fabric_ps=100),
        )
