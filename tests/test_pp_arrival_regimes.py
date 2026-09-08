"""Prospective guards for the finite pipeline arrival qualification."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
from itertools import product
from types import SimpleNamespace

import pytest

from examples.pp_arrival_regimes_v1 import publish_results
from examples.pp_arrival_regimes_v1 import run_study as study
from simllm.backends.htsim_rnic import FlowCompletion
from simllm.core import ComputeWork
from simllm.traffic import render_serial_execution_graph_goal


def pick(**fields):
    return next(cell for cell in study.cells() if all(getattr(cell, key) == value for key, value in fields.items()))


def test_exact_population_has_72_executions_and_two_independent_sweep_axes():
    assert len(study.cells()) == len({cell.name for cell in study.cells()}) == 56
    assert len(study.jobs()) == 72
    assert Counter(cell.population for cell, _ in study.jobs()) == {
        "physical": 32, "ideal": 16, "regression": 12, "legacy": 12}
    physical = [cell for cell in study.cells() if cell.population == "physical"]
    assert {(cell.variant, cell.width, cell.ep_width, cell.arrival_ps) for cell in physical} == set(
        product(("rail", "node-local"), (4, 8), (0, 32), (16_000, 80_000)))
    assert all(cell.spines == 2 for cell in physical)
    assert all(cell.arrival_ps == 0 for cell in study.cells() if cell.population in {"regression", "legacy"})


@pytest.mark.parametrize("cell", [cell for cell in study.cells() if cell.population == "physical"], ids=lambda c: c.name)
def test_only_first_declared_compute_changes_and_messages_plans_stay_exact(cell):
    baseline = study.reference.build_graph(cell.width, cell.ep_width)
    shifted = study.build_graph(cell)
    assert replace(shifted, operations=baseline.operations) == baseline
    assert shifted.collective_plans == baseline.collective_plans
    changed = [(before, after) for before, after in zip(baseline.operations, shifted.operations) if before != after]
    assert len(changed) == 1
    before, after = changed[0]
    assert isinstance(after.work, ComputeWork) and after.rank == 0
    assert replace(after, work=before.work) == before
    assert replace(after.work, nominal_duration_ps=before.work.nominal_duration_ps) == before.work
    assert after.work.nominal_duration_ps == 1_000_000 + cell.arrival_ps
    old_trace = render_serial_execution_graph_goal(baseline, num_goal_ranks=64)
    new_trace = render_serial_execution_graph_goal(shifted, num_goal_ranks=64)
    assert new_trace.messages == old_trace.messages
    assert study.build_graph(replace(cell, arrival_ps=0)) == baseline


@pytest.mark.parametrize("cell", study.cells(), ids=lambda c: c.name)
def test_policy_and_observation_selection_preserve_the_frozen_regime(cell, tmp_path):
    config = study.configuration(cell, tmp_path, tmp_path, False)
    assert config.initial_window_bytes is config.initial_window_fan_in is None
    if cell.population in {"physical", "regression"}:
        assert config.control_recovery == "headroom" and config.data_recovery == "exponential"
        assert config.extra_flags["-rnic_cn_ring_window_ps"] == "10566400"
        assert config.retry_probe_windows == 4
    else:
        assert config.control_recovery == config.data_recovery == "none"
    if cell.population == "physical":
        traced = study.configuration(cell, tmp_path, tmp_path, True)
        flags = dict(traced.extra_flags)
        assert flags.pop("-rnic_cn_trace_dir") == str(tmp_path / "trace")
        assert flags == config.extra_flags
    else:
        with pytest.raises(ValueError, match="trace arm"):
            study.configuration(cell, tmp_path, tmp_path, True)


@pytest.mark.parametrize("variant,budget", [("rail", 16_742_451_200), ("node-local", 11_457_628_160)])
@pytest.mark.parametrize("width,arrival", list(product((4, 8), study.ARRIVALS_PS)))
def test_bounds_count_added_compute_without_enlarging_loaded_budget(variant, budget, width, arrival):
    cell = pick(population="physical", variant=variant, width=width, ep_width=32, arrival_ps=arrival)
    bounds = study.physical_bounds(cell)
    hop = 65_536 * 20 + (2 if variant == "rail" else 4) * 1_000_000
    assert bounds["pp_step_data_floor_ps"] == width * 1_000_000 + arrival + (width - 1) * hop
    assert bounds["declared_compute_ps"] == width * 1_000_000 + arrival
    assert bounds["engineering_budget_ps"] == budget
    assert bounds["budget_queue_allowance_ps"] == 70_997_760
    assert bounds["unconditional_fct_ceiling_ps"] is None


def serial_pp(cell):
    flows, release = [], 1_001_000 + cell.arrival_ps
    for index in range(cell.width - 1):
        duration = 10_000_200 + 16_000 * index
        flows.append(SimpleNamespace(start_time_ps=release, completion_time_ps=release + duration, fct_ps=duration))
        release += duration + 1_002_000
    return flows


@pytest.mark.parametrize("width,arrival", list(product((4, 8), study.ARRIVALS_PS)))
def test_live_event_and_request_attribution_assign_arrival_to_compute(width, arrival):
    cell = pick(population="physical", width=width, arrival_ps=arrival)
    pp, findings = serial_pp(cell), []
    timestamp = study.exact_goal_timing(pp, cell, findings)
    event = study.previous.final_stage_event(study.build_graph(cell), pp)
    result = study.packet_step_result(study.reference.step_record(), event, cell, pp)
    assert findings == [] and event.timestamp_ps == timestamp
    metric = result.request_metrics[0]
    assert metric.ttft_ps == timestamp and metric.tpot_ps is None
    assert metric.attribution.kernel_ps == width * 1_000_000 + arrival
    assert metric.attribution.nic_ps == sum(flow.fct_ps for flow in pp)
    assert metric.attribution.control_ps == 2000 * (width - 1) - pp[-1].completion_time_ps % 1000
    assert metric.attribution.total_ps == result.step_latency_ps
    with pytest.raises(ValueError, match="decomposition"):
        study.packet_step_result(study.reference.step_record(), event, replace(cell, arrival_ps=0), pp)


@pytest.mark.parametrize("hop", (0, 1, 2))
def test_exact_release_check_rejects_equal_fct_shift(hop):
    cell = study.cells()[0]
    pp = serial_pp(cell)
    pp[hop].start_time_ps += 1000
    pp[hop].completion_time_ps += 1000
    findings = []
    study.exact_goal_timing(pp, cell, findings)
    assert any("release differs" in finding for finding in findings)


def ideal_rows(cell):
    projection = study.projection_for(cell)
    flows = [FlowCompletion("rnic-nn", 1, projection.endpoint_by_rank[0], projection.endpoint_by_rank[8],
                            10, 65536, 1_001_000, 2_500_000, 1_499_000),
             FlowCompletion("rnic-nn", 2, projection.endpoint_by_rank[1], projection.endpoint_by_rank[9],
                            11, 1048576, 0, 20_971_520, 20_971_520)]
    transformed = [replace(flows[0], start_time_ps=flows[0].start_time_ps + cell.arrival_ps,
                           completion_time_ps=flows[0].completion_time_ps + cell.arrival_ps), flows[1]]
    return flows, transformed


@pytest.mark.parametrize("variant", ("rail", "node-local"))
def test_ideal_transformation_joins_semantic_pipeline_ranks_across_endpoint_permutation(variant, monkeypatch, tmp_path):
    cell = pick(population="ideal", variant=variant)
    old, expected = ideal_rows(cell)
    monkeypatch.setattr(study, "parse_completion_csv", lambda _: old)
    assert study.shifted_ideal(cell, tmp_path) == expected
    assert study.ideal_transform_guard(list(reversed(expected)), expected)
    assert not study.ideal_transform_guard(old, expected)
    assert not study.ideal_transform_guard([expected[0], replace(expected[1], start_time_ps=1)], expected)
    assert study.ideal_transform_guard([replace(expected[0], flow_id=7), expected[1]], expected)
    with pytest.raises(ValueError, match="duplicated"):
        study.ideal_transform_guard([expected[0], replace(expected[1], flow_id=1)], expected)
    with pytest.raises(ValueError, match="duplicated"):
        study.ideal_transform_guard([expected[0], expected[0]], expected)


def input_fixture(root, cell):
    inputs = root / cell.name
    inputs.mkdir()
    names = ("step.goal", "step.bin", "clos.topo", "fabric.json", "placement.json",
             "semantic_graph.json", "endpoint_by_rank.json")
    for name in names:
        (inputs / name).write_text(name + "\n")
    study.write_json(inputs / "inputs.json", {"cell": asdict(cell), "expectations_commit": study.FREEZE,
                                             "input_sha256": {name: study.digest(inputs / name) for name in names}})
    return inputs


def test_raw_execution_record_precedes_auditor_failure(monkeypatch, tmp_path):
    cell = study.cells()[0]
    inputs = input_fixture(tmp_path, cell)
    monkeypatch.setattr(study, "run_owned_process", lambda *args, **kwargs:
                        SimpleNamespace(stdout="retained native output", stderr="", returncode=0))

    def reject(cell, out, *args):
        execution = json.loads((out / "execution.json").read_bytes())
        assert execution["run_log_sha256"] == study.digest(out / "run.log")
        assert "retained native output" in (out / "run.log").read_text()
        raise ValueError("synthetic audit failure")

    monkeypatch.setattr(study, "analyze_execution", reject)
    row = study.execute((cell, True), root=tmp_path, references=tmp_path, regression_locks={},
                        input_locks={cell.name: study.digest(inputs / "inputs.json")},
                        binary=tmp_path / "binary", timeout_s=1)
    assert row["cell_status"] == "void" and row["request_ttft_ps"] is None
    assert row["fatal_findings"] == ["ValueError: synthetic audit failure"]


def test_changed_input_never_reaches_native_execution(monkeypatch, tmp_path):
    cell = study.cells()[0]
    inputs = input_fixture(tmp_path, cell)
    (inputs / "step.goal").write_text("changed")
    monkeypatch.setattr(study, "run_owned_process", lambda *args, **kwargs: pytest.fail("invalid input ran"))
    row = study.execute((cell, False), root=tmp_path, references=tmp_path, regression_locks={},
                        input_locks={cell.name: study.digest(inputs / "inputs.json")},
                        binary=tmp_path / "binary", timeout_s=1)
    assert row["cell_status"] == "void" and "input bytes changed" in row["fatal_findings"][0]
    assert (inputs / "trace-off/execution.json").exists()


def void_rows():
    return [study.empty_row(cell, traced, ["synthetic interrupted execution"]) for cell, traced in study.jobs()]


def packet(index, attempt=0):
    return {"lifecycle_id": index * 10 + attempt + 1, "packet_index": index, "attempt": attempt,
            "source_start_ps": 100, "source_end_ps": 200, "eta_ps": 1000, "arrival_ps": 1000,
            "logical_release_ps": 2000, "rx_service_start_ps": 2000, "rx_service_end_ps": 3000,
            "delivery_ps": 3000, "admission": "admitted", "terminal": "endpoint_consumed",
            "retry_authorization": None, "drop_occupancy": None, "visits": []}


def audit_for(cell):
    flows = []
    for stage in range(cell.width - 1):
        packets = [packet(index) for index in range(16)]
        flows.append({"flow_id": stage + 1, "source_rank": 8 * stage, "destination_rank": 8 * (stage + 1),
                      "fct_ps": 10_000_000, "packets": packets, "completion_packet_lifecycle_id": 151,
                      "completion_trigger_lifecycle_id": 151, "trigger_packet_timeline": {"total_ps": 10_000_000},
                      "completion_dependency_witness": {"complete_resource_dependency_graph": False}})
    loaded = cell.variant == "node-local" and cell.ep_width == 32
    if loaded:
        flow = flows[1]
        flow["fct_ps"] += 2_520_000_000
        original = flow["packets"][14]
        attempts = [original] + [packet(14, number) for number in range(1, 8)]
        for number, candidate in enumerate(attempts):
            if number < 7:
                candidate.update(terminal="fabric_drop", admission=None, drop_occupancy={
                    "shared_partition_bytes": {"ep_data": 1_040_000}, "admits_without_ep_data": True})
            if number:
                interval = 10_000 if number == 1 else study.PROBE_WINDOWS_PS[number - 2]
                when = attempts[number - 1]["source_end_ps"] + interval
                candidate["retry_authorization"] = {
                    "cause": "gap_nack" if number == 1 else "probe_timeout", "time_ps": when,
                    "deadline_ps": None if number == 1 else when,
                    "origin_attempt": None if number == 1 else number - 1}
                candidate.update(source_start_ps=when + 10, source_end_ps=when + 110)
        flow["packets"].extend(attempts[1:])
        flow["completion_trigger_lifecycle_id"] = attempts[-1]["lifecycle_id"]
        flow["packets"][0]["visits"] = [{"service_ahead_ps": {"ep_data": 83_200}, "ep_data_bound_ps": 83_200}]
    return {"schema": "pp-queue-audit-v1", "status": "valid", "egresses": [], "pp_flows": flows,
            "queue_work": {"ep_data_service_ahead_ps": 83_200 if loaded else 0}}


def valid_rows():
    rows = []
    for cell, traced in study.jobs():
        row = study.empty_row(cell, traced, [])
        loaded = cell.variant == "node-local" and cell.ep_width == 32
        penalty = 2_520_000_000 if loaded else 0
        row.update(cell_status="valid", request_ttft_ps=100_000_000 + penalty,
                   pp_hop_maximum_ps=10_000_000 + penalty, completion_sha256="a" * 64,
                   physical_quiescence=True, physical_quiescence_time_ps=5_000_000_000,
                   complete_flow_phase_ps=3_000_000_000, ep_phase_ps=3_000_000_000 if cell.ep_width else 0,
                   job_completion_ps=3_000_000_000, request_attribution={"kernel_ps": 1_000_000 * cell.width + cell.arrival_ps},
                   pp_fct_samples_ps=[10_000_000 + penalty], pp_hops=[
                       {"source": 8 * stage, "destination": 8 * (stage + 1), "payload_bytes": 65536,
                        "tag": 1000 + stage + bool(cell.ep_width), "start_time_ps": stage * 10_000_000,
                        "completion_time_ps": (stage + 1) * 10_000_000 + penalty,
                        "fct_ps": 10_000_000 + penalty}
                       for stage in range(cell.width - 1)],
                   native_manifest=["identity"], trace_audit=audit_for(cell) if traced else None)
        rows.append(row)
    return rows


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "metadata"))
def test_frozen_population_cannot_be_silently_reduced_or_relabelled(mutation):
    rows = void_rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    else:
        rows[-1]["arrival_ps"] = 1
    with pytest.raises(ValueError, match="execution"):
        study.summarize(rows, {})


def test_valid_scoped_result_keeps_evidence_classes_separate():
    result = study.summarize(valid_rows(), {})
    assert result["verdict"] == "valid-qualified-penalty" and result["traf88_acceptance"] == "met"
    assert len(result["behavioral_relations"]) == 12
    assert Counter(row["family"] for row in result["behavioral_relations"]) == {
        "B1-visible-penalty": 4, "B2-independent-contention": 4, "B3-retry-mechanism": 4}
    assert len(result["trace_identity_oracles"]) == 16 and len(result["rail_identity_oracles"]) == 4
    assert result["control_counts"] == {"ideal": 16, "regression": 12, "legacy": 12}
    assert "behavioral_score" not in result


def rail_pair():
    return [row for row in valid_rows() if row["population"] == "physical" and row["traced"]
            and row["variant"] == "rail" and row["width"] == 4 and row["arrival_ps"] == 16000]


def test_rail_timing_matches_across_different_renderer_tag_allocations():
    unloaded, loaded = rail_pair()
    assert unloaded["pp_hops"][0]["tag"] != loaded["pp_hops"][0]["tag"]
    loaded["pp_hops"].reverse()
    oracle = study.rail_identity(unloaded, loaded)
    assert oracle["matches"] and oracle["pp_flow_times_exact"]


@pytest.mark.parametrize("field", ("start_time_ps", "completion_time_ps", "fct_ps",
                                  "source", "destination", "payload_bytes"))
def test_rail_timing_rejects_changed_time_or_hop_correspondence(field):
    unloaded, loaded = rail_pair()
    loaded["pp_hops"][0][field] += 1
    oracle = study.rail_identity(unloaded, loaded)
    assert not oracle["matches"] and not oracle["pp_flow_times_exact"]


@pytest.mark.parametrize("mutation", ("missing", "duplicate"))
def test_rail_timing_rejects_incomplete_hop_inventory(mutation):
    unloaded, loaded = rail_pair()
    if mutation == "missing":
        loaded["pp_hops"].pop()
    else:
        loaded["pp_hops"][1] = deepcopy(loaded["pp_hops"][0])
    with pytest.raises(ValueError, match="complete unique PP hop"):
        study.rail_identity(unloaded, loaded)


def loaded_trace(rows):
    return next(row for row in rows if row["population"] == "physical" and row["traced"]
                and row["variant"] == "node-local" and row["ep_width"] == 32)


@pytest.mark.parametrize("mutation", ("band", "occupancy", "recovery"))
def test_a_different_valid_behavior_is_a_refutation(mutation):
    rows = valid_rows()
    row = loaded_trace(rows)
    if mutation == "band":
        for match in rows:
            if match["cell"] == row["cell"]:
                match["request_ttft_ps"] = 100_000_000
    elif mutation == "occupancy":
        row["trace_audit"]["pp_flows"][1]["packets"][14]["drop_occupancy"]["admits_without_ep_data"] = False
    else:
        row["trace_audit"]["pp_flows"][1]["completion_trigger_lifecycle_id"] = 151
    result = study.summarize(rows, {})
    assert result["verdict"] == "valid-refutation" and result["traf88_acceptance"] == "not-met"
    assert result["fatal_findings"] == []


@pytest.mark.parametrize("mutation", ("trace", "rail-time", "rail-ep", "missing-witness", "valid-with-fatal"))
def test_fatal_guard_voids_all_task_acceptance(mutation):
    rows = valid_rows()
    if mutation == "trace":
        rows[1]["physical_quiescence_time_ps"] += 1
    elif mutation == "rail-time":
        row = next(row for row in rows if row["traced"] and row["variant"] == "rail" and row["ep_width"] == 32)
        row["trace_audit"]["pp_flows"][0]["packets"][0]["source_start_ps"] += 1
    elif mutation == "rail-ep":
        rows[1]["trace_audit"]["queue_work"]["ep_data_service_ahead_ps"] = 1
    elif mutation == "missing-witness":
        loaded_trace(rows)["trace_audit"]["pp_flows"][1]["packets"][14]["drop_occupancy"] = None
    else:
        rows[-1]["fatal_findings"] = ["fatal while status was incorrectly valid"]
    result = study.summarize(rows, {})
    assert result["verdict"] == "void" and result["traf88_acceptance"] == "not-met"
    assert result["fatal_findings"] and "behavioral_score" not in result


def test_void_run_has_only_uninterpretable_relations():
    result = study.summarize(void_rows(), {})
    assert result["verdict"] == "void"
    assert all(row["matched"] is None for row in result["behavioral_relations"])


def test_compact_projection_retains_verdict_and_retry_counterfactual_without_rescoring(tmp_path):
    complete = study.summarize(valid_rows(), {})
    source = tmp_path / "results.json"
    study.write_json(source, complete)
    for row in complete["physical_configurations"]:
        study.write_json(tmp_path / row["cell"] / "trace-on/trace_audit.json", row["trace_audit"])
    projected = publish_results.project(source)
    assert projected["verdict"] == complete["verdict"] and projected["traf88_acceptance"] == complete["traf88_acceptance"]
    assert projected["behavioral_relations"] == json.loads(source.read_bytes())["behavioral_relations"]
    assert projected["bulk_results_sha256"] == study.digest(source)
    assert all("trace_audit" not in row for row in projected["physical_configurations"])
    row = next(row for row in projected["physical_configurations"] if row["variant"] == "node-local" and row["ep_width"] == 32)
    attempts = row["audit_projection"]["pp_flows"][1]["delivery_trigger_attempts"]
    assert len(attempts) == 8 and attempts[0]["drop_occupancy"]["admits_without_ep_data"]
    study.write_json(source, projected)
    with pytest.raises(ValueError, match="complete"):
        publish_results.project(source)


def test_publisher_rejects_changed_retained_audit(tmp_path):
    complete = study.summarize(valid_rows(), {})
    source = tmp_path / "results.json"
    study.write_json(source, complete)
    row = complete["physical_configurations"][0]
    altered = deepcopy(row["trace_audit"])
    altered["queue_work"]["ep_data_service_ahead_ps"] += 1
    study.write_json(tmp_path / row["cell"] / "trace-on/trace_audit.json", altered)
    with pytest.raises(ValueError, match="disagree"):
        publish_results.project(source)


def test_rewritten_input_manifest_cannot_self_authorize_changed_bytes(tmp_path):
    cell = study.cells()[0]
    inputs = input_fixture(tmp_path, cell)
    frozen_digest = study.digest(inputs / "inputs.json")
    study.verify_inputs(inputs, cell, frozen_digest)
    (inputs / "step.goal").write_text("changed")
    manifest = json.loads((inputs / "inputs.json").read_bytes())
    manifest["input_sha256"]["step.goal"] = study.digest(inputs / "step.goal")
    study.write_json(inputs / "inputs.json", manifest)
    with pytest.raises(ValueError, match="manifest changed"):
        study.verify_inputs(inputs, cell, frozen_digest)


def test_extra_audited_attempt_does_not_change_the_frozen_seventh_attempt_hypothesis():
    rows = valid_rows()
    row = loaded_trace(rows)
    flow = row["trace_audit"]["pp_flows"][1]
    extra = packet(14, 8)
    extra["retry_authorization"] = {"cause": "probe_timeout", "time_ps": 10_000_000_000,
                                    "deadline_ps": 10_000_000_000, "origin_attempt": 7}
    flow["packets"].append(extra)
    result = study.summarize(rows, {})
    assert result["verdict"] == "valid-qualified-penalty"
    b2 = next(item for item in result["behavioral_relations"] if item["family"] == "B2-independent-contention")
    witness = b2["details"]["first_positive_visit"]
    assert witness["source_rank"] == 8 and witness["destination_rank"] == 16
    assert witness["packet_index"] == 0 and witness["attempt"] == 0
    assert witness["visit"]["ep_data_bound_ps"] == 83_200


def test_missing_service_witness_voids_the_aggregate_instead_of_raising_or_closing():
    rows = valid_rows()
    loaded_trace(rows)["trace_audit"]["pp_flows"][1]["packets"][0].pop("visits")
    result = study.summarize(rows, {})
    assert result["verdict"] == "void" and result["traf88_acceptance"] == "not-met"
    assert all(item["matched"] is None for item in result["behavioral_relations"])


@pytest.mark.parametrize("variant", ("rail", "node-local"))
def test_preparation_locks_unchanged_inputs_and_records_bounds_before_conversion(variant, monkeypatch, tmp_path):
    cell = pick(population="physical", variant=variant, ep_width=32)
    retained = tmp_path / "reference" / cell.reference_name
    retained.mkdir(parents=True)
    projection = study.projection_for(cell)
    placement = study.declared_pipeline_placement(cell.width)
    fabric = study.declared_rail_fabric(placement, variant=variant, spine_count=cell.spines)
    baseline = study.build_graph(replace(cell, arrival_ps=0))
    baseline_trace = render_serial_execution_graph_goal(projection.project_graph(baseline), num_goal_ranks=64)
    (retained / "step.goal").write_bytes(baseline_trace.render().encode())
    (retained / "step.bin").write_bytes(b"frozen baseline binary")
    (retained / "clos.topo").write_bytes(projection.topology_text.encode())
    for name, value in (("placement.json", asdict(placement)), ("fabric.json", asdict(fabric)),
                        ("semantic_graph.json", study.execution_graph_to_json(baseline)),
                        ("endpoint_by_rank.json", projection.endpoint_by_rank)):
        study.write_json(retained / name, value)
    ideal = retained.parent / cell.reference_name.replace("rnic-cn", "rnic-nn")
    ideal.mkdir()
    (ideal / "completion.csv").write_bytes(b"frozen ideal completion\n")
    lock = {field: study.digest(retained / name) for name, field in (
        ("step.goal", "goal_text_sha256"), ("step.bin", "goal_sha256"), ("clos.topo", "topology_sha256"))}
    lock["ideal_completion_sha256"] = study.digest(ideal / "completion.csv")
    out = tmp_path / "out"
    out.mkdir()

    def convert(path, *, tool):
        bounds = json.loads((path.parent / "pre_run_bounds.json").read_bytes())
        assert bounds["declared_compute_ps"] == cell.width * 1_000_000 + cell.arrival_ps
        assert path.read_bytes() != (retained / "step.goal").read_bytes()
        path.with_suffix(".bin").write_bytes(b"synthetic converter output")

    monkeypatch.setattr(study, "to_binary", convert)
    study.prepare_cell(cell, out, retained.parent, {cell.reference_name: lock}, tmp_path / "converter")
    inputs = out / cell.name
    study.verify_inputs(inputs, cell, study.digest(inputs / "inputs.json"))
    assert (inputs / "clos.topo").read_bytes() == (retained / "clos.topo").read_bytes()


@pytest.mark.parametrize("variant", ("rail", "node-local"))
def test_zero_offset_goal_matches_published_digest_list(variant):
    cell = pick(population="physical", variant=variant, width=4, ep_width=0)
    graph = study.build_graph(replace(cell, arrival_ps=0))
    raw = render_serial_execution_graph_goal(
        study.projection_for(cell).project_graph(graph), num_goal_ranks=64).render().encode()
    accepted = study.predecessor.reference_locks()[cell.reference_name]["goal_text_sha256"]
    assert isinstance(accepted, list) and accepted
    study.require_rendered_goal(raw, accepted)
    study.require_rendered_goal(raw, accepted[0])
    with pytest.raises(ValueError, match="published reference"):
        study.require_rendered_goal(raw + b"changed", accepted)
