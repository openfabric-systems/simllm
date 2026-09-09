"""Prospective reference and admission controls, with no native installation."""

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_independent_native_bridge import bridge_fixture  # noqa: F401

from examples.independent_engine_completion_v1 import run_study as campaign
from examples.independent_engine_completion_v1.checks import (
    check_checkpoints,
    check_component_checkpoints,
    check_components,
    check_engines,
    check_native_relations,
    check_projections,
    check_selected_envelopes,
    expected_times,
)
from examples.independent_engine_completion_v1.common import (
    HERE,
    PUBLICATIONS,
    Evidence,
    GuardFailure,
    exact_json_bytes,
    json_value,
    read,
    sha,
    sink_publications,
    step_stream,
    write,
)
from examples.independent_engine_completion_v1.reference import component_run, native_reference
from examples.independent_engine_completion_v1.run_study import (
    RUNTIME_MUTATIONS,
    file_receipts,
    progress_rows,
    reread_tree,
    runtime_corruption,
)
from simllm.core import StepRecord
from simllm.core.step import step_record_to_json

FROZEN = json.loads((HERE / "expectations.json").read_bytes())


@pytest.mark.parametrize("mode,expected_exit,reason", [
    ("startup", None, "process construction failed"),
    ("timeout", -9, "frozen native timeout reached"),
    ("exit", 7, "native process failed"),
])
def test_failed_native_attempt_retains_monitor_and_first_receipts(tmp_path, monkeypatch, mode, expected_exit, reason):
    args = SimpleNamespace(output_root=tmp_path, native_python=tmp_path / "python",
                           vllm_source=tmp_path / "source", hf_hub_cache=tmp_path / "cache")
    spec = FROZEN["native_processes"][0]
    monitors, initial, root_receipts = {}, {}, {}
    times = iter((0, FROZEN["limits"]["native_timeout_seconds"] + 1,
                  FROZEN["limits"]["native_timeout_seconds"] + 2))
    monkeypatch.setattr(campaign.time, "monotonic", lambda: next(times))

    class Process:
        pid = 1_000_000_000

        def __init__(self):
            self.returncode = 7 if mode == "exit" else None

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode

    def start(*args, **kwargs):
        assert spec["id"] in monitors and monitors[spec["id"]]["pid"] is None
        kwargs["stdout"].write(b"retained native output\n")
        if mode == "startup":
            raise OSError("process construction failed")
        return Process()

    monkeypatch.setattr(campaign.subprocess, "Popen", start)
    with pytest.raises((OSError, GuardFailure), match=reason):
        campaign.capture_native(args, spec, FROZEN, monitors, initial, root_receipts)
    path = tmp_path / spec["id"]
    monitor = read(path / "process.json")
    assert monitors == {spec["id"]: monitor}
    assert monitor["exit_code"] == expected_exit and reason in monitor["stopping_reason"]
    assert monitor["pid"] == (None if mode == "startup" else Process.pid)
    assert monitor["sampled_max_current_rss_kib"] == 0 and monitor["wall_seconds"] > 0
    assert (path / "native.log").read_bytes() == b"retained native output\n"
    first = deepcopy(initial[spec["id"]])
    assert first == file_receipts(path)
    receipt_path = tmp_path / (spec["id"] + "-raw-receipts.json")
    assert read(receipt_path) == first
    assert root_receipts[receipt_path.name] == {"sha256": sha(receipt_path.read_bytes()), "bytes": receipt_path.stat().st_size}
    (path / "native.log").write_bytes(b"changed later\n")
    with pytest.raises(GuardFailure, match="raw-bytes-and-domain"):
        reread_tree(path, first, Evidence([]), "failed-capture")
    assert initial[spec["id"]] == read(receipt_path) == first


def source_selection_fixture():
    """Real source envelopes over a selection-only, post-specified identity fixture."""
    from simllm.compute import GPU_ENVELOPES, HostInitiationModel, RooflineProvider
    from simllm.traffic import resolve_collective_fixed_cost_envelope

    data = json.loads((Path(__file__).parent / "fixtures/independent_engine_selection.json").read_bytes())["data"]
    profile = resolve_collective_fixed_cost_envelope("intra-node-fixed-cost-v1").arm_profile("lower")
    for selected in data["selected_before"]:
        selected.update(gpu=asdict(GPU_ENVELOPES["b100"]), host_model=asdict(HostInitiationModel.ideal()),
                        provider_state=vars(RooflineProvider(efficiency=0.7)), collective_profile=asdict(profile))
    data["selected_after"] = deepcopy(data["selected_before"])
    return json_value(data)


def test_source_selection_survives_exact_writer_and_physical_bound_join(tmp_path):
    path = tmp_path / "selection.json"
    write(path, source_selection_fixture())
    data = read(path)
    evidence = Evidence([])
    check_engines(data, FROZEN["native_processes"][0], FROZEN, evidence)
    check_selected_envelopes(data, "source-selection", evidence)
    assert type(data["selected_before"][0]["gpu"]["mem_bandwidth"]) is float
    assert evidence.oracles == evidence.relations == []


@pytest.mark.parametrize("changed", [4.0e12, 8.0e12 + 0.5, True, "8000000000000", 8000000000000])
def test_source_selection_rejects_changed_bandwidth_and_coerced_type(tmp_path, changed):
    data = source_selection_fixture()
    for selected in data["selected_before"]:
        selected["gpu"]["mem_bandwidth"] = changed
    data["selected_after"] = deepcopy(data["selected_before"])
    path = tmp_path / "selection.json"
    write(path, data)
    data = read(path)
    with pytest.raises(GuardFailure, match="memory-bandwidth"):
        check_engines(data, FROZEN["native_processes"][0], FROZEN, Evidence([]))
    with pytest.raises(GuardFailure, match="gpu"):
        check_selected_envelopes(data, "source-selection", Evidence([]))


@pytest.mark.parametrize("changed", [float("nan"), float("inf"), -float("inf")])
def test_source_selection_refuses_nonfinite_encoding(tmp_path, changed):
    data = source_selection_fixture()
    data["selected_before"][0]["gpu"]["mem_bandwidth"] = changed
    with pytest.raises(ValueError):
        write(tmp_path / "invalid.json", data)


def reference_data():
    """Project mathematical timelines into the fields used by joint relations."""
    result = {}
    for process in FROZEN["native_processes"]:
        cells = []
        start = 0
        for spec in process["cells"]:
            reference = native_reference(process, spec, FROZEN["known_services_ps"])
            rows = [{"result": {"request_id": spec["id"] + ":request-" + str(index), **expected_times(row, start)}}
                    for index, row in enumerate(reference["requests"])]
            cells.append({"id": spec["id"], "start_ps": start, "end_ps": start + reference["makespan_ps"], "requests": rows})
            start += reference["makespan_ps"]
        result[process["id"]] = {"cells": cells}
    return result


def test_all_frozen_relations_have_their_exact_unpadded_domain():
    evidence = Evidence(FROZEN["required_stages"])
    components = [component_run(spec) for spec in FROZEN["component_cases"]]
    check_components(components, FROZEN, evidence)
    check_native_relations(reference_data(), FROZEN, evidence)
    assert len(components) == 14
    assert len(evidence.relations) == 25 and len(evidence.oracles) == 2
    actual = {(row["family"], row["name"]) for row in evidence.relations}
    expected = {(family, name) for family, spec in FROZEN["behavioral_families"].items() for name in spec["instances"]}
    assert actual == expected


@pytest.mark.parametrize("mutation", ["service", "routing", "release", "context", "finished"])
def test_component_inputs_are_bound_to_frozen_jobs(mutation):
    spec = FROZEN["component_cases"][0]
    row = component_run(spec)
    entry = row["inputs"][0]
    if mutation == "service":
        entry["result"]["step_latency_ps"] -= 1
        row["completed"][0]["result"]["step_latency_ps"] -= 1
    elif mutation == "routing":
        entry["engine_id"] = entry["receipt"]["engine_id"] = "foreign"
    elif mutation == "release":
        entry["record"]["virtual_time_ps"] = entry["receipt"]["submitted_at_ps"] = 1
    elif mutation == "context":
        entry["record"]["scheduled"][0]["context_length"] = 1
    else:
        entry["record"]["finished_request_ids"] = ["foreign"]
    with pytest.raises(GuardFailure, match="frozen-input"):
        check_component_checkpoints(row, spec, Evidence([]))


@pytest.mark.parametrize("process", FROZEN["native_processes"], ids=lambda row: row["id"])
def test_finite_native_oracles_respect_causal_clock_and_engine_work(process):
    for cell in process["cells"]:
        reference = native_reference(process, cell, FROZEN["known_services_ps"])
        previous = 0
        for row in reference["clock_advances"]:
            assert row["before_ps"] == previous and row["after_ps"] > previous
            previous = row["after_ps"]
        assert previous == reference["makespan_ps"]
        steps = reference["service_steps"]
        assert len(steps) == cell["requests"] * 5
        for index, row in enumerate(reference["requests"]):
            owned = [step for step in steps if step["request_index"] == index]
            assert len(owned) == 5
            assert row["prefill_end"] + cell["handoff_ps"] <= row["decode_start"]
            assert owned[0]["start_ps"] >= row["arrival"]
            assert owned[-1]["end_ps"] == row["tokens"][-1]
        for engine in {step["engine_id"] for step in steps}:
            local = sorted((step for step in steps if step["engine_id"] == engine), key=lambda row: row["start_ps"])
            assert all(a["end_ps"] <= b["start_ps"] for a, b in zip(local, local[1:]))  # noqa: RUF007
        if cell["kind"] == "burst":
            prompt, width = cell["prompt_tokens"][0], process["prefill_engines"]
            service = FROZEN["known_services_ps"][str(prompt)]
            expected = (service["prefill"] + cell["handoff_ps"] + 16 // width * service["decode"][0]
                        if process["mode"] == "independent" else 4 * service["prefill"] + 16 * service["decode"][0])
            assert reference["makespan_ps"] == expected


@pytest.mark.parametrize("name", sorted(RUNTIME_MUTATIONS))
def test_real_pending_runtime_rejects_semantic_corruption(tmp_path, name):
    assert runtime_corruption(name, tmp_path)


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1}\n', b'{ "x": 1 }'])
def test_canonical_reader_rejects_duplicate_nonfinite_and_alternate_writers(tmp_path, raw):
    path = tmp_path / "value.json"
    path.write_bytes(raw)
    with pytest.raises(GuardFailure):
        read(path)


def test_source_writer_stream_has_a_distinct_exact_encoding(tmp_path):
    path = tmp_path / "step-records.jsonl"
    row = step_record_to_json(StepRecord(0, 0, finished_request_ids=["done"]))
    valid = (json.dumps(row) + "\n").encode("ascii")
    path.write_bytes(valid)
    assert step_stream(path) == [row]
    for raw in (valid.rstrip(b"\n"), valid.replace(b"\n", b"\r\n"), exact_json_bytes(row) + b"\n"):
        path.write_bytes(raw)
        with pytest.raises(GuardFailure):
            step_stream(path)


def test_progress_and_retained_disk_bytes_are_independent_guards(tmp_path):
    path = tmp_path / "progress.jsonl"
    raw = exact_json_bytes({"at_ps": 1}) + b"\n"
    path.write_bytes(raw)
    assert progress_rows(path) == [{"at_ps": 1}]
    receipts = file_receipts(tmp_path)
    path.write_bytes(raw.replace(b"1", b"2"))
    with pytest.raises(GuardFailure, match="raw-bytes"):
        reread_tree(tmp_path, receipts, Evidence([]), "receipt")
    path.write_bytes(raw.replace(b"\n", b"\r\n"))
    with pytest.raises(GuardFailure, match="progress writer"):
        progress_rows(path)


def capture_bridge(f):
    checkpoints = []

    def observe(phase, engine, native):
        publications = sink_publications(engine)
        checkpoints.append({
            "index": len(checkpoints), "phase": phase, "engine_id": engine.engine_id,
            "at_ps": f.runtime.clock.now_ps,
            "step_index": engine.executor._runtime.step_index if phase == "before-submit" else engine.executor.step_records[-1].step_index,
            "record_count": len(engine.executor.step_records), "result_count": len(engine.executor.step_results),
            "sink_counts": {name: len(publications[name]) for name in PUBLICATIONS},
            "sink_sha256": {name: sha(exact_json_bytes(publications[name])) for name in PUBLICATIONS},
            "native": native,
        })

    f.executor.token_id = 512
    f.bridge.observer = observe
    first = f.bridge.submit(lambda outputs, record: None)
    f.runtime.advance_to(first.completed_at_ps)
    f.runtime.complete_due()
    f.bridge.submit(lambda outputs, record: None)
    f.runtime.complete_due()
    data = json_value({
        "checkpoints": checkpoints, "sinks": {"engine-a": sink_publications(f.bridge.engine)},
        "projections": {
            "events": [asdict(row) for row in f.runtime.events], "visits": [asdict(row) for row in f.runtime.visits],
            "completed": [{"receipt": asdict(receipt), "result": asdict(result)} for receipt, result in f.runtime.results],
        },
    })
    steps = {("engine-a", record.step_index): {"engine_id": "engine-a", "record": step_record_to_json(record), "result": json_value(asdict(result))}
             for record, result in zip(f.executor.step_records, f.executor.step_results, strict=True)}
    return data, steps


def test_actual_bridge_publications_and_finished_drain_join_admission(bridge_fixture):  # noqa: F811
    data, steps = capture_bridge(bridge_fixture)
    spec = {"id": "component-bridge", "mode": "independent"}
    evidence = Evidence([])
    check_projections(data, spec, steps, evidence)
    check_checkpoints(data, spec, steps, {("engine-a", "native-a"): "a"}, evidence, expected_cache_blocks=2)
    assert len(data["projections"]["completed"]) == 2
    assert data["projections"]["completed"][-1]["result"]["step_latency_ps"] == 0
    assert len(evidence.guards) > 100
    assert evidence.oracles == evidence.relations == []


@pytest.mark.parametrize("mutation", ["token", "cache", "owned-free", "early-sink", "duplicate", "missing", "foreign", "record", "time", "native-missing", "service"])
def test_projection_and_checkpoint_corruptions_fail_admission(bridge_fixture, mutation):  # noqa: F811
    data, steps = capture_bridge(bridge_fixture)
    spec = {"id": "corrupt-bridge", "mode": "independent"}
    if mutation == "token":
        data["checkpoints"][1]["native"]["visible"]["native-a"]["output_token_ids"] = [512]
    elif mutation == "cache":
        data["checkpoints"][2]["native"]["cache"]["free_queue"] = [1]
    elif mutation == "owned-free":
        data["checkpoints"][1]["native"]["cache"]["blocks"][1][1] = 0
        data["checkpoints"][1]["native"]["cache"]["free_queue"] = [1]
    elif mutation == "early-sink":
        data["checkpoints"][1]["sink_counts"]["outcomes"] = 1
    elif mutation == "duplicate":
        data["projections"]["completed"].append(deepcopy(data["projections"]["completed"][0]))
    elif mutation == "missing":
        data["projections"]["events"].pop()
    elif mutation == "foreign":
        data["projections"]["completed"][0]["receipt"]["engine_id"] = "foreign"
    elif mutation == "record":
        steps[("engine-a", 0)]["record"]["virtual_time_ps"] = 1
    elif mutation == "time":
        data["projections"]["events"][0]["timestamp_ps"] = 1
    elif mutation == "native-missing":
        for capture in data["checkpoints"]:
            capture["native"] = {"requests": {}, "visible": {}, "emitted": [],
                                 "queues": {"waiting": [], "running": []},
                                 "cache": {"groups": [{}], "blocks": [[0, 0], [1, 0]], "free_queue": [1]}}
    elif mutation == "service":
        steps[("engine-a", 0)]["result"]["step_latency_ps"] -= 1
        data["projections"]["completed"][0]["result"]["step_latency_ps"] -= 1
    with pytest.raises(GuardFailure):
        evidence = Evidence([])
        check_projections(data, spec, steps, evidence)
        check_checkpoints(data, spec, steps, {("engine-a", "native-a"): "a"}, evidence, expected_cache_blocks=2)


def test_json_projection_preserves_wire_enums_and_rejects_opaque_values(tmp_path):
    from simllm.core.execution import EventPhase
    from simllm.core.runtime import QueueVisit

    assert json_value({"tuple": (EventPhase.COMPLETED,)}) == {"tuple": ["completed"]}
    with pytest.raises(TypeError):
        json_value(object())
    with pytest.raises(TypeError):
        json_value(QueueVisit)
    write(tmp_path / "valid.json", {"path": str(Path("relative"))})
    assert read(tmp_path / "valid.json") == {"path": "relative"}


@pytest.mark.parametrize("prompt,visit", [(8, None), (8, 0), (8, 3), (16, None), (16, 0), (16, 3)])
def test_declared_granite_sink_decomposition_admits_full_artifacts(tmp_path, prompt, visit):
    from examples.independent_engine_completion_v1.checks import check_sink_decomposition
    from examples.pd_session_v1 import run_study as baseline
    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
    from simllm.compute import RooflineProvider
    from simllm.core import RequestPhase, ScheduledRequest
    from simllm.placement import declared_manifest
    from simllm.traffic import resolve_collective_fixed_cost_envelope

    prefill = visit is None
    record = StepRecord(0, 0, [ScheduledRequest(
        "native", RequestPhase.PREFILL if prefill or visit == 0 else RequestPhase.DECODE,
        prompt if prefill else 1, num_cached_tokens=prompt if visit == 0 else 0,
        context_length=prompt if prefill else prompt + visit + 1)], num_sampled=1)
    sink = HtsimStepSink(HtsimStepSinkConfig(
        profile="rnic-nn-fluid", tp_ranks=tuple(range(8)), dims=baseline._granite_dims(), workdir=tmp_path,
        provider=RooflineProvider(efficiency=0.7), placement_manifest=declared_manifest(tp=8, nodes=1, gpus_per_node=8),
        collective_fixed_cost_envelope="intra-node-fixed-cost-v1", collective_fixed_cost_arm="lower"))
    result = sink(record)
    assert result.step_latency_ps >= FROZEN["bounds"]["resident_streaming_floor_ps"]
    published = json_value({name: asdict(getattr(sink, name)[0]) for name in ("outcomes", "locality_outcomes", "collective_timing_outcomes")})
    envelope = resolve_collective_fixed_cost_envelope("intra-node-fixed-cost-v1")
    check_sink_decomposition(
        {"record": step_record_to_json(record), "result": json_value(asdict(result))},
        published["outcomes"], published["locality_outcomes"], published["collective_timing_outcomes"],
        envelope.arm_profile("lower"), envelope, "sink-fixture", Evidence([]))
