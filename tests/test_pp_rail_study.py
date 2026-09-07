from __future__ import annotations

import importlib.util
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from simllm.backends.htsim_rnic import FlowCompletion, RnicRunResult
from simllm.traffic import render_serial_execution_graph_goal

STUDY = Path(__file__).resolve().parents[1] / "examples" / "pp_rail_topology_v1"
spec = importlib.util.spec_from_file_location("pp_rail_study", STUDY / "run_study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


@pytest.mark.parametrize(("width", "messages", "peers"), [(0, 0, 0), (8, 56, 7), (32, 896, 28)])
def test_ep_inventory_and_reserved_pp_rail(width, messages, peers):
    pairs = study.ep_pairs(width)
    assert len(pairs) == messages
    assert all(source % 8 != 0 and destination % 8 != 0
               and source // 8 != destination // 8 for source, destination, _ in pairs)
    assert set(Counter(destination for _, destination, _ in pairs).values()) == ({peers} if width else set())
    assert sum(size for _, _, size in pairs) == messages * 1_048_576


@pytest.mark.parametrize("width", [2, 4, 8])
@pytest.mark.parametrize("ep", [0, 8, 32])
def test_background_does_not_add_dependency_on_pp(width, ep):
    graph = study.build_graph(width, ep)
    trace = render_serial_execution_graph_goal(graph, num_goal_ranks=64)
    assert len(trace.messages) == width - 1 + len(study.ep_pairs(ep))
    if ep:
        assert graph.operations[0].operation_id == "background-ep"
        assert not graph.operations[0].depends_on
        assert all("background-ep" not in operation.depends_on
                   and "background-ep" not in operation.participant_local_depends_on
                   for operation in graph.operations[1:])
        assert "background-ep" not in graph.completion_operation_ids


def test_physical_bounds_keep_data_and_sender_completion_distinct():
    rail = study.bounds(8, 32, "rail")
    local = study.bounds(8, 32, "node-local")
    assert rail["serialization_floor_ps"] == 1_310_720
    assert rail["data_arrival_floor_ps"] == 3_310_720
    assert local["data_arrival_floor_ps"] == 5_310_720
    assert rail["payload_only_queue_envelope_ps"] == 0
    assert local["payload_only_queue_envelope_ps"] == 4 * (896 * 1_048_576 + 7 * 65_536) * 20
    assert not local["control_and_ack_included_in_ceiling"]


def test_quantile_is_finite_sample_nearest_rank():
    assert study.quantile([5, 2, 9], 0.5) == 5
    assert study.quantile([5, 2, 9], 0.99) == 9
    with pytest.raises(ValueError):
        study.quantile([], 0.99)


def test_bad_completion_is_fatal_without_behavioral_denominator():
    trace = render_serial_execution_graph_goal(study.build_graph(2, 0), num_goal_ranks=64)
    message = trace.messages[0]
    flow = FlowCompletion("rnic-cn", 1, 0, 8, message.tag, 65536, 1_001_000, 11_001_000, 10_000_000)
    result = RnicRunResult([flow], [], True, 12_002_000)
    row, _ = study.evaluate_cell("rail", 2, 0, "rnic-cn", trace, [flow], result,
                                 study.bounds(2, 0, "rail"))
    assert row["fatal_findings"] == []
    assert not row["exact_rail_oracle"]
    broken = replace(flow, payload_bytes=65535)
    bad, _ = study.evaluate_cell("rail", 2, 0, "rnic-cn", trace, [broken], result,
                                 study.bounds(2, 0, "rail"))
    assert "PP count or bytes differ from frozen inventory" in bad["fatal_findings"]
    assert not any("score" in key or "denominator" in key for key in bad)


def test_published_results_are_complete_and_separate_evidence_classes():
    result = json.loads((STUDY / "results.json").read_text())
    assert result["expectations_commit"] == study.EXPECTATIONS_COMMIT
    assert result["configuration_count"] == result["requested_configuration_count"] == 36
    assert result["fatal_status"] == "clear"
    assert result["execution_errors"] == []
    assert all(not row["fatal_findings"] for row in result["configurations"])
    assert all(row["matched"] for row in result["coarse_metric_reachability"])
    relations = result["behavioral_and_exact_relations"]
    physical = [row for row in relations if row.get("profile") == "rnic-cn"]
    assert all(row["matched"] for row in physical if row["family"] == "R2-rail-independence")
    assert not any(row["matched"] for row in physical if row["family"] == "R3-node-local-load-growth")
    assert not any(row["exact_rail_oracle"] for row in result["configurations"] if row["variant"] == "rail")
    text = json.dumps(result)
    assert "/mnt/" not in text and "/home/" not in text
