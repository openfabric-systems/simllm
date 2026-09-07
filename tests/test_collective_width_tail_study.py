"""Offline checks for collective-study evidence and fail-closed boundaries."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import simllm.backends.step_sink as sink_module
from simllm.backends.htsim_rnic import FlowCompletion
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.traffic import step_moe_alltoalls

RUNNER = Path(__file__).resolve().parents[1] / "examples/collective_width_tail_v1/run_study.py"
SPEC = importlib.util.spec_from_file_location("collective_width_tail_study", RUNNER)
STUDY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STUDY
SPEC.loader.exec_module(STUDY)
COMPUTE_PS = STUDY.COMPUTE_PS
FROZEN_RING_PS = STUDY.FROZEN_RING_PS
PAIR_BYTES = STUDY.PAIR_BYTES
PAYLOAD = STUDY.PAYLOAD
Cell = STUDY.Cell
aligned_comparison = STUDY.aligned_comparison
bounds = STUDY.bounds
build_trace = STUDY.build_trace
check = STUDY.check
expected_messages = STUDY.expected_messages
flow_guards = STUDY.flow_guards
flow_summary = STUDY.flow_summary
make_sink = STUDY.make_sink
nearest_rank = STUDY.nearest_rank
ranks_for = STUDY.ranks_for
topology_for = STUDY.topology_for
verdict = STUDY.verdict

def flow(source=0, destination=8, tag=1000, start=0, fct=5_000_000, payload=PAIR_BYTES):
    return FlowCompletion("rnic-nn", tag, source, destination, tag, payload,
                          start, start + fct, fct)


@pytest.mark.parametrize("width", (8, 16, 32, 64))
def test_striping_uses_all_eight_nodes_and_no_local_ring_edge(width):
    ranks = ranks_for(width)
    assert len(ranks) == len(set(ranks)) == width
    assert {r // 8 for r in ranks} == set(range(8))
    for i, rank in enumerate(ranks):
        assert rank // 8 != ranks[(i + 1) % width] // 8


@pytest.mark.parametrize("width", (8, 16, 32, 64))
def test_actual_patterns_conserve_frozen_bytes(width):
    ring = build_trace(Cell("ring", width, 400, "rnic-nn"))
    text = ring.render()
    assert text.count(": send ") == 2 * width * (width - 1)
    assert text.count(f"send {PAYLOAD // width}b") == 2 * width * (width - 1)
    messages = expected_messages(Cell("all-to-all", width, 400, "rnic-nn"))
    assert sum(messages.values()) == width * (7 * width // 8)
    assert sum(key[3] * n for key, n in messages.items()) == width * (7 * width // 8) * PAIR_BYTES
    collective = build_trace(Cell("all-to-all", width, 400, "rnic-nn")).render()
    assert collective.count(": send ") == width * (7 * width // 8)


def test_nearest_rank_retains_small_sample_tails():
    assert nearest_rank([9, 1, 5, 3], .5) == 3
    assert nearest_rank([9, 1, 5, 3], .99) == 9
    with pytest.raises(ValueError):
        nearest_rank([], .5)
    with pytest.raises(ValueError):
        nearest_rank([1], 0)


def test_flow_summary_measures_phase_span_including_staggered_starts():
    rows = [flow(start=100, fct=1000), flow(tag=1001, start=4000, fct=2000)]
    summary = flow_summary(rows)
    assert summary["phase_makespan_ps"] == 5900
    assert summary["fct_p50_ps"] == 1000
    assert summary["fct_p99_ps"] == 2000
    assert summary["phase_makespan_ps"] != sum(r.fct_ps for r in rows)


def test_normalization_excludes_shifted_starts_without_changing_raw_fct():
    baseline = [flow(), flow(tag=1001)]
    physical = [replace(flow(fct=7_000_000), profile="rnic-cn"),
                replace(flow(tag=1001, start=1, fct=3_000_000), profile="rnic-cn")]
    ratios, excluded = aligned_comparison(physical, baseline)
    assert excluded == 1
    assert len(ratios) == 1
    assert ratios[0]["slowdown"] == 1.4
    assert physical[1].fct_ps == 3_000_000


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "bytes"))
def test_normalization_refuses_mismatched_populations(mutation):
    baseline = [flow()]
    physical = {"missing": [], "duplicate": [flow(), flow()],
                "bytes": [flow(payload=PAIR_BYTES + 1)]}[mutation]
    with pytest.raises(ValueError):
        aligned_comparison(physical, baseline)


def test_structural_guard_catches_duplicate_or_impossible_flow():
    cell = Cell("all-to-all", 8, 400, "rnic-nn")
    rows = [flow(s, d, tag=tag, payload=b) for s, d, tag, b in expected_messages(cell)]
    rows = [replace(f, flow_id=i) for i, f in enumerate(rows)]
    assert all(g["ok"] for g in flow_guards(cell, rows))
    broken = [replace(rows[0], fct_ps=1, completion_time_ps=1), *rows[1:]]
    checks = {g["family"]: g["ok"] for g in flow_guards(cell, broken)}
    assert not checks["payload_floor"]
    assert not checks["propagation_floor"]
    duplicated = [*rows, rows[0]]
    checks = {g["family"]: g["ok"] for g in flow_guards(cell, duplicated)}
    assert not checks["message_identity"]
    assert not checks["unique_flow_id"]


def test_one_fatal_guard_voids_the_whole_run_instead_of_losing_a_point():
    rows = [{"name": "bad", "status": "complete",
             "guards": [check("payload_floor", False)]}]
    outcome = verdict(rows, [check("oracle", True)], [check("direction", True)])
    assert outcome["status"] == "void"
    assert outcome["behavioral_relations"]["failed"] is None
    assert not outcome["behavioral_relations"]["interpretable_for_closure"]
    assert not outcome["comp9_closed"]


def test_rejection_and_component_miss_never_close_comp9():
    rows = [{"name": "cn-step", "status": "unsupported"}]
    outcome = verdict(rows, [check("oracle", False)], [check("twofold", False)])
    assert outcome["status"] == "component-evidence-only"
    assert outcome["exact_oracles"]["failed"] == 1
    assert outcome["behavioral_relations"]["failed"] == 1
    assert outcome["rejected_steps"] == ["cn-step"]
    assert not outcome["comp9_closed"]


def test_halving_rate_changes_both_tier_rates_and_no_latency(tmp_path):
    slow = topology_for(tmp_path, 200).read_text()
    assert slow.count("Downlink_speed_Gbps 200") == 2
    assert slow.count("Downlink_Latency_ns 1000") == 2
    assert slow.count("Switch_Latency_ns 0") == 2


@pytest.mark.parametrize("width", (8, 16, 32, 64))
def test_frozen_ring_oracle_has_linear_latency_and_inverse_rate_serialization(width):
    fast = FROZEN_RING_PS[(width, 400)]
    slow = FROZEN_RING_PS[(width, 200)]
    propagation = 2 * (width - 1) * 2_000_000
    assert slow - propagation == 2 * (fast - propagation)
    floor = bounds(Cell("ring", width, 400, "rnic-nn"))["phase_floor_ps"]
    assert fast >= floor
    assert bounds(Cell("ring", width, 400, "rnic-cn"))["phase_ceiling_ps"] is None


@pytest.mark.parametrize("width", (8, 16, 32, 64))
def test_step_expert_payload_is_fixed_and_single_engine(tmp_path, width):
    cell = Cell("all-to-all", width, 400, "rnic-nn", "step")
    sink = make_sink(cell, tmp_path, None)
    assert list(sink.config.ep_ranks) == sorted(ranks_for(width))
    record = StepRecord(0, 0, [ScheduledRequest("r", RequestPhase.PREFILL, width * 32)])
    dispatch, combine = step_moe_alltoalls(record, sink.config.dims, sink.config.ep_ranks)
    assert len(dispatch.pair_payload_bytes) == width - 1
    assert {s for s, _, _ in dispatch.pair_payload_bytes} == {0}
    assert {d for _, d, _ in combine.pair_payload_bytes} == {0}
    assert {b for _, _, b in dispatch.pair_payload_bytes} == {PAIR_BYTES}
    assert sink.config.provider.estimate(None, None).duration_ps == COMPUTE_PS


@pytest.mark.parametrize("pattern", ("ring", "all-to-all"))
def test_physical_step_guard_runs_before_any_binary(tmp_path, monkeypatch, pattern):
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported stateful step must not start a backend")

    monkeypatch.setattr(sink_module, "to_binary", forbidden)
    monkeypatch.setattr(sink_module, "run_htsim_rnic", forbidden)
    cell = Cell(pattern, 8, 400, "rnic-cn", "step")
    sink = make_sink(cell, tmp_path, tmp_path / "topology.topo")
    record = StepRecord(0, 0, [ScheduledRequest("r", RequestPhase.PREFILL, 512, context_length=512)])
    with pytest.raises(RuntimeError, match="BACK-38"):
        sink(record)


def test_empty_step_cannot_masquerade_as_a_measured_network_share():
    with pytest.raises(ValueError, match="no completed flows"):
        flow_summary([])
    empty = SimpleNamespace(status="error")
    outcome = verdict([{"name": "empty", "status": empty.status}], [], [])
    assert outcome["status"] == "incomplete"
    assert not outcome["behavioral_relations"]["interpretable_for_closure"]


def test_partial_matrix_is_incomplete_even_without_an_explicit_backend_error():
    result = verdict([], [], [], expected_count=64)
    assert result["status"] == "incomplete"
    assert result["missing_configurations"] == 64
    assert not result["behavioral_relations"]["interpretable_for_closure"]
