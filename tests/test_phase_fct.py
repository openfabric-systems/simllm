"""Conservation checks distinguish scheduler order from impossible service."""

from dataclasses import replace

import pytest

from simllm.backends.fct import (
    earliest_completion_byte_floors,
    normalized_fct,
    normalized_phase_makespan,
)
from simllm.backends.htsim_rnic import FlowCompletion


def flow(source=1, size=100, finish=1100, start=0, tag=1000):
    return FlowCompletion("rnic-cn", source, source, 0, tag, size,
                          start, finish, finish - start)


def floors(rows, rate=8_000_000_000_000, propagation=100):
    return earliest_completion_byte_floors(rows, link_rate_bps=rate, propagation_ps=propagation)


def test_unfair_scheduler_can_beat_fair_small_completion_without_creating_bytes():
    # One byte/ps, 100 ps propagation: fair pair finishes at 300 and 600.
    ideal = [flow(size=100, finish=300), flow(source=2, size=400, finish=600)]
    physical = [flow(size=100, finish=200), flow(source=2, size=400, finish=650)]
    assert [n.slowdown for n in normalized_fct(physical, ideal)] == [2 / 3, 650 / 600]
    assert normalized_phase_makespan(physical, ideal).slowdown == 650 / 600
    assert [p.slack_ps for p in floors(physical)] == [0, 50]


def test_early_prefix_detects_double_service_even_when_final_makespan_passes():
    rows = [flow(finish=200), flow(source=2, finish=200), flow(source=3, finish=1000)]
    prefix = floors(rows)
    assert [p.ok for p in prefix] == [True, False, True]
    assert prefix[1].cumulative_bytes == 200
    assert prefix[1].floor_ps == 300
    assert prefix[1].slack_ps == -100


def test_phase_baseline_can_fail_even_when_payload_prefixes_pass():
    physical = [flow(finish=200), flow(source=2, finish=300)]
    ideal = [flow(finish=290), flow(source=2, finish=310)]
    assert all(p.ok for p in floors(physical))
    assert normalized_phase_makespan(physical, ideal).slowdown < 1


def test_phase_span_includes_idle_gap_and_uses_each_runs_first_start():
    physical = [flow(finish=200, start=100), flow(source=2, finish=600, start=500)]
    ideal = [flow(finish=100), flow(source=2, finish=200)]
    result = normalized_phase_makespan(physical, ideal)
    assert (result.makespan_ps, result.baseline_makespan_ps, result.slowdown) == (500, 200, 2.5)
    assert sum(f.fct_ps for f in physical) == 200


@pytest.mark.parametrize("mutation", ("missing", "extra", "payload", "multiplicity"))
def test_phase_normalization_rejects_different_transfer_populations(mutation):
    ideal = [flow(), flow(source=2)]
    changed = {"missing": [flow()], "extra": [*ideal, flow(source=3)],
               "payload": [flow(size=101), flow(source=2)],
               "multiplicity": [*ideal, flow()]}[mutation]
    with pytest.raises(ValueError):
        normalized_phase_makespan(changed, ideal)


def test_phase_repeated_keys_keep_existing_start_order_matching():
    ideal = [flow(finish=100, size=10), flow(finish=400, start=200, size=20)]
    physical = [flow(finish=600, start=400, size=20), flow(finish=200, size=10)]
    assert normalized_phase_makespan(physical, ideal).slowdown == 1.5


@pytest.mark.parametrize("bad", [[], [flow(start=-1)], [flow(size=0)],
                                 [flow(finish=0)], [replace(flow(), fct_ps=7)]])
def test_new_metrics_reject_invalid_phase_evidence(bad):
    with pytest.raises(ValueError):
        normalized_phase_makespan(bad, [flow()])
    with pytest.raises(ValueError):
        floors(bad)


@pytest.mark.parametrize("rate,propagation", [(0, 0), (-1, 0), (1.5, 0),
                                              (True, 0), (1, -1), (1, 0.5)])
def test_prefix_units_require_integer_positive_capacity(rate, propagation):
    with pytest.raises(ValueError):
        floors([flow()], rate, propagation)


def test_prefix_service_rounds_up_exactly_without_float_precision_loss():
    row, = floors([flow(size=1, finish=3)], rate=3_000_000_000_000, propagation=0)
    assert row.floor_ps == 3 and row.ok
    huge = 2 ** 54 + 1
    row, = floors([flow(size=huge, finish=huge + 100)])
    assert row.floor_ps == huge + 100 and row.ok


def test_receiver_partition_and_tie_order_are_explicit():
    with pytest.raises(ValueError, match="one receiver"):
        floors([flow(), replace(flow(source=2), destination=3)])
    a = floors([flow(source=2), flow(source=1)])
    b = floors([flow(source=1), flow(source=2)])
    assert a == b and [p.source for p in a] == [1, 2]


def test_staggered_receiver_floor_is_conservative_from_first_phase_start():
    rows = [flow(start=10, finish=210), flow(source=2, start=1000, finish=1200)]
    assert [p.elapsed_ps for p in floors(rows)] == [200, 1190]
    assert [p.floor_ps for p in floors(rows)] == [200, 300]
