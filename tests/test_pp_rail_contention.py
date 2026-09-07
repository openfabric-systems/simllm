from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest

from simllm.backends.htsim_rnic import FlowCompletion
from simllm.backends.rail_topology import project_declared_clos
from simllm.placement import declared_pipeline_placement, declared_rail_fabric
from simllm.traffic import render_serial_execution_graph_goal

_spec = importlib.util.spec_from_file_location(
    "pp_rail_contention", Path(__file__).resolve().parents[1]
    / "examples" / "pp_rail_contention_v1" / "run_study.py")
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)


@pytest.mark.parametrize("variant,spines,rate", list(product(study.VARIANTS, (2, 4, 8), (200, 400, 800))))
def test_capacity_variant_geometry_and_projection(variant, spines, rate):
    placement = declared_pipeline_placement(8)
    fabric = declared_rail_fabric(placement, variant=variant, spine_count=spines,
                                  uplink_rate_bps=rate * 10**9)
    fabric.validate()
    assert len(fabric.links) == 64 + 8 * spines
    assert len(fabric.switches) == 8 + spines
    projection = project_declared_clos(fabric)
    expected = tuple(rank % 8 * 8 + rank // 8 if variant == "rail" else rank for rank in range(64))
    assert projection.endpoint_by_rank == expected
    tier0, tier1 = projection.topology_text.split("Tier 1\n")
    assert "Downlink_speed_Gbps 400\n" in tier0
    assert f"Radix_Up {spines}\n" in tier0
    assert f"Downlink_speed_Gbps {rate}\n" in tier1
    assert (f"Oversubscribed {8 // spines}\n" in tier0) == (spines < 8)
    endpoint_ids = {nic.link_id for node in fabric.nodes for nic in node.nics}
    assert {link.link_rate_bps for link in fabric.links if link.link_id in endpoint_ids} == {study.RATE}
    assert {link.link_rate_bps for link in fabric.links if link.link_id not in endpoint_ids} == {rate * 10**9}
    for source, destination in ((0, 8), (0, 1), (7, 63), (0, 63)):
        expected_links = 2 if study.leaf_for(source, variant) == study.leaf_for(destination, variant) else 4
        assert len(fabric.path_between_ranks(source, destination)) == expected_links
    # Distinct capacities retain semantic message identity through both projections.
    graph = study.reference.build_graph(8, 8)
    trace = render_serial_execution_graph_goal(projection.project_graph(graph), num_goal_ranks=64)
    semantic = render_serial_execution_graph_goal(graph, num_goal_ranks=64)
    recovered = []
    for index, message in enumerate(trace.messages):
        flow = FlowCompletion("rnic-cn", index, message.source_rank, message.destination_rank,
                              message.tag, message.payload_bytes, 1000, 100000000, 99999000)
        recovered.append(study.reference.flow_key(projection.semantic_flow(flow)))
    assert sorted(recovered) == sorted((m.source_rank, m.destination_rank, m.tag, m.payload_bytes)
                                       for m in semantic.messages)


def test_default_manifest_and_topology_golden_bytes():
    assert len(study.identity_guards()) == 2
    for variant in study.VARIANTS:
        p = declared_pipeline_placement(8)
        assert declared_rail_fabric(p, variant=variant) == declared_rail_fabric(
            p, variant=variant, spine_count=8, uplink_rate_bps=400_000_000_000)


@pytest.mark.parametrize("spines", [0, 1, 3, 16, True, 2.0])
def test_invalid_spine_count(spines):
    with pytest.raises(ValueError, match="spine_count"):
        declared_rail_fabric(declared_pipeline_placement(), variant="rail", spine_count=spines)


@pytest.mark.parametrize("rate", [0, -1, True, 1.0, 400_000_000_001])
def test_invalid_uplink_rate(rate):
    with pytest.raises(ValueError, match="uplink_rate"):
        declared_rail_fabric(declared_pipeline_placement(), variant="rail", uplink_rate_bps=rate)


def test_nonuniform_uplink_and_incomplete_wiring_rejected():
    fabric = declared_rail_fabric(declared_pipeline_placement(), variant="rail", spine_count=2)
    links = (*fabric.links[:-1], replace(fabric.links[-1], link_rate_bps=200_000_000_000))
    with pytest.raises(ValueError, match="uniform"):
        project_declared_clos(replace(fabric, links=links))
    with pytest.raises(ValueError):
        project_declared_clos(replace(fabric, links=fabric.links[:-1]))
    with pytest.raises(ValueError, match="whole"):
        project_declared_clos(replace(fabric, switch_latency_ps=1))


@pytest.mark.parametrize("width", study.WIDTHS)
def test_frozen_shared_byte_bounds(width):
    for variant, expected_bytes in (("rail", 176160768), ("node-local", 117440512)):
        b8 = study.bounds(variant, 8, width, 32)
        b2 = study.bounds(variant, 2, width, 32)
        assert b2["ep_busiest_leaf_uplink_bytes"] == expected_bytes
        assert b2["ep_cut_drain_floor_ps"] == expected_bytes * 10
        assert b8["ep_cut_drain_floor_ps"] == expected_bytes * 20 // 8
        assert b2["shared_throughput_ceiling_bytes_per_second"] == 100_000_000_000
        assert b2["ep_endpoint_bytes"] == 28 * study.M
        for hop in b2["pp_hop_cut_bounds"]:
            assert hop["source_leaf_ep_uplink_bytes"] == (0 if variant == "rail" else 112 * study.M)
            assert hop["guaranteed_ep_bytes_ahead"] == hop["additional_queue_floor_ps"] == 0
            assert hop["hop_data_arrival_floor_ps"] == (3310720 if variant == "rail" else 5310720)
    rail8 = study.bounds("rail", 2, width, 8)
    local8 = study.bounds("node-local", 2, width, 8)
    assert rail8["ep_busiest_leaf_uplink_bytes"] == 0
    assert local8["ep_busiest_leaf_uplink_bytes"] == 7 * study.M
    assert rail8["ep_effective_serialization_floor_ps"] == local8["ep_effective_serialization_floor_ps"]


def test_text_digest_and_writes_are_cross_platform(tmp_path):
    raw = b'{"a": 1}\n'
    digest = hashlib.sha256(raw).hexdigest()
    assert study.text_digest_matches(raw, digest)
    assert study.text_digest_matches(raw.replace(b"\n", b"\r\n"), digest)
    assert not study.text_digest_matches(b"other", digest)
    path = tmp_path / "result.json"
    study.write_json(path, {"a": 1})
    assert b"\r" not in path.read_bytes()
    assert study.reference_inputs()["configuration_count"] == 36


def fake_run(graph, duplicate=False):
    trace = render_serial_execution_graph_goal(graph, num_goal_ranks=64)
    message = trace.messages[0]
    flow = FlowCompletion("rnic-nn", 0, message.source_rank, message.destination_rank, message.tag,
                          study.B, 1001000, 4415400, 3414400)
    return SimpleNamespace(flows=[flow, flow] if duplicate else [flow], quiescent=True,
                           manifest=["[RNIC manifest] physical_quiescence=verified"],
                           job_completion_time_ps=lambda: 5416000)


def test_sink_native_boundary_to_live_metrics_and_fatal_suppression(tmp_path, monkeypatch):
    calls = []

    def native(self, plan, goal, completion):
        calls.append((plan.profile, goal.read_bytes(), completion.name))
        return fake_run(self.projection.project_graph(self.graph))

    monkeypatch.setattr(study.ContentionStepSink, "_run_goal", native)
    sink = study.ContentionStepSink(tmp_path / "clear", "rail", 2, 2, 0, "rnic-nn")
    result = sink(study.reference.step_record())
    assert calls[0][0] == "rnic-nn"
    assert calls[0][2] == "completion.csv"
    assert result.step_latency_ps == result.request_metrics[0].ttft_ps == 5416000
    assert result.request_metrics[0].attribution.total_ps == 5416000
    assert result.request_metrics[0].tpot_ps is None
    assert sink.row["cell_status"] == "clear"
    assert sink.row["pp_hop_critical_path_share"] == 3414400 / 5416000
    monkeypatch.setattr(study.ContentionStepSink, "_run_goal", lambda self, *_: fake_run(
        self.projection.project_graph(self.graph), duplicate=True))
    void = study.ContentionStepSink(tmp_path / "void", "rail", 2, 2, 0, "rnic-nn")
    assert void(study.reference.step_record()) is None
    assert void.row["cell_status"] == "void"
    assert "packet_metric_projection" not in void.row
    assert not (tmp_path / "void" / "step_result.json").exists()


def test_void_cells_cannot_score_relations():
    row = {"variant": "rail", "spine_count": 8, "pp_width": 2, "ep_width": 0,
           "profile": "rnic-cn", "cell_status": "void", "pp_fct_p99_ps": 10087000}
    relations = study.relation_rows([row], study.reference_inputs())
    assert all(relation["matched"] is None for relation in relations)
    assert all(relation["status"] == "unscored-void-or-missing" for relation in relations)


@pytest.mark.parametrize("profile", study.PROFILES)
def test_native_seam_enforces_frozen_seed(tmp_path, monkeypatch, profile):
    sink = study.ContentionStepSink(tmp_path, "node-local", 2, 2, 0, profile)
    goal = tmp_path / "step.goal"
    binary = tmp_path / "step.bin"
    monkeypatch.setattr(study, "to_binary", lambda path: binary if path == goal else None)
    calls = []
    monkeypatch.setattr(study, "run_htsim_rnic", lambda config, timeout_s: calls.append((config, timeout_s)))
    sink._run_goal(sink.config, goal, tmp_path / "completion.csv")
    config, timeout = calls[0]
    assert config.goal_bin == binary
    assert config.extra_flags == ({"-rnic_cn_prbs_seed": "1"} if profile == "rnic-cn" else {})
    assert config.topology == (tmp_path / "clos.topo" if profile == "rnic-cn" else None)
    assert timeout == 120


def test_tracked_grid_conserves_metrics_and_matches_declared_relations():
    path = study.HERE / "results.json"
    if not path.exists():
        pytest.skip("native artifact has not been generated")
    summary = json.loads(path.read_bytes())
    assert summary["expectations_commit"] == study.EXPECTATIONS_COMMIT
    rows = summary["configurations"]
    keys = {study.cell_key(row) for row in rows}
    keys.update(tuple(error["cell"]) for error in summary["execution_errors"])
    assert keys == set(product(study.VARIANTS, study.SPINES, study.WIDTHS, study.EP_WIDTHS, study.PROFILES))
    assert len(keys) == 72
    for row in rows:
        assert row["bounds"] == study.bounds(*study.cell_key(row)[:4])
        if row["cell_status"] == "void":
            assert row["fatal_findings"]
            continue
        assert row["fatal_findings"] == []
        samples = row["pp_fct_samples_ps"]
        assert len(samples) == row["pp_width"] - 1
        assert row["pp_fct_p99_ps"] == max(samples)
        assert row["pp_fct_p50_ps"] == study.reference.quantile(samples, 0.5)
        metric = row["packet_metric_projection"]
        assert metric["request_ttft_ps"] == metric["step_latency_ps"] == row["pp_final_stage_projection_ps"]
        assert sum(metric["attribution"].values()) == metric["step_latency_ps"]
        assert 0 <= row["pp_hop_critical_path_share"] <= row["pp_communication_projection_share"] <= 1
    assert summary["behavioral_and_exact_relations"] == study.relation_rows(rows, study.reference_inputs())
