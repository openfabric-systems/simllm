"""Probe transparency and hostile evidence admission without native engines."""

import json
import sys
from copy import deepcopy
from dataclasses import asdict
from itertools import count, pairwise
from types import SimpleNamespace

import pytest
from test_pd_session_target_scale import retained  # noqa: F401

from examples.pd_host_execution_diagnostic_v1 import checks, run_study, timing
from examples.pd_host_execution_diagnostic_v1 import native as diagnostic
from examples.pd_session_identity_v1.run_study import exact_json_bytes, sha
from examples.pd_session_target_scale_v1 import native as target
from simllm.core import DeclaredKvHandoffPolicy
from simllm.core.pd_session import DisaggregatedRequestTimeline

FROZEN = json.loads((run_study.HERE / "expectations.json").read_bytes())


def test_probe_preserves_inherited_method_arguments_result_and_structure():
    sentinel, calls = object(), []

    class Parent:
        def method(self, *args, **kwargs):
            calls.append((self, args, kwargs))
            return sentinel

    instance = Parent()
    counters = timing.PhaseCounters()
    counters.bind(instance, "method", "native")
    argument = object()
    assert instance.method(argument, flag=argument) is sentinel
    assert calls == [(instance, (argument,), {"flag": argument})]
    assert "method" in vars(instance)
    counters.close()
    assert "method" not in vars(instance)
    assert instance.method.__func__ is Parent.method
    assert counters.stack == [] and counters.restored[0]["restored"] is True
    assert counters.snapshot()["native"]["calls"] == 1


def test_probe_preserves_exception_and_restores_existing_own_binding():
    error = RuntimeError("sentinel exception")

    def original(*args, **kwargs):
        raise error

    owner = SimpleNamespace(call=original)
    counters = timing.PhaseCounters()
    counters.bind(owner, "call", "exception")
    with pytest.raises(RuntimeError) as caught:
        owner.call(1, flag=2)
    assert caught.value is error and not counters.stack
    counters.close()
    assert owner.call is original and counters.restored[0]["original_own_attribute"] is True


def test_nested_counters_keep_inclusive_and_exclusive_time_distinct():
    marker = object()

    def inner():
        return marker

    owner = SimpleNamespace(inner=inner)

    def outer():
        return owner.inner()

    owner.outer = outer
    counters = timing.PhaseCounters(wall_timer=count(10, 10).__next__, cpu_timer=count(10, 10).__next__)
    counters.bind(owner, "inner", "inner")
    counters.bind(owner, "outer", "outer")
    before = counters.snapshot()
    assert owner.outer() is marker
    rows = timing.counter_delta(before, counters.snapshot())
    assert rows["outer"]["wall_ns"] == 30
    assert rows["outer"]["child_wall_ns"] == 10
    assert rows["outer"]["exclusive_wall_ns"] == 20
    assert rows["inner"]["exclusive_wall_ns"] == 10
    assert sum(row["exclusive_wall_ns"] for row in rows.values()) == 30
    assert sum(row["wall_ns"] for row in rows.values()) == 40
    checks.check_counter_rows(rows, ("inner", "outer"), checks.fresh_evidence(), "nested")
    counters.close()


def test_replaced_binding_is_restored_but_rejected():
    def original():
        return 1

    owner = SimpleNamespace(call=original)
    counters = timing.PhaseCounters()
    counters.bind(owner, "call", "phase")
    owner.call = lambda: 2
    with pytest.raises(RuntimeError, match="replaced"):
        counters.close()
    assert owner.call is original and counters.closed


@pytest.mark.parametrize("inherited", [False, True])
def test_missing_binding_does_not_prevent_exhaustive_restoration(inherited):
    class Owner:
        def first(self):
            return 1

        def second(self):
            return 2

    owner = Owner() if inherited else SimpleNamespace(first=Owner.first, second=Owner.second)
    counters = timing.PhaseCounters()
    counters.bind(owner, "first", "first")
    counters.bind(owner, "second", "second")
    del owner.second
    with pytest.raises(RuntimeError, match="replaced"):
        counters.check_bindings()
    with pytest.raises(RuntimeError, match="replaced"):
        counters.close()
    assert counters.closed and len(counters.restored) == 2
    assert all(row["restored"] for row in counters.restored)
    assert ("first" in vars(owner)) is not inherited
    assert ("second" in vars(owner)) is not inherited


@pytest.mark.parametrize("failure", [False, True])
def test_profile_preserves_return_exception_and_complete_caller_keys(tmp_path, failure):
    result, error = object(), RuntimeError("same exception")

    def child():
        if failure:
            raise error
        return result

    def parent():
        return child()

    path = tmp_path / "cell.pstats"
    if failure:
        with pytest.raises(RuntimeError) as caught:
            timing.profiled_call(parent, profile_path=path)
        assert caught.value is error
    else:
        returned, rows = timing.profiled_call(parent, profile_path=path)
        assert returned is result
        checks.check_profile({"schema": "simllm-host-call-profile-v1", "cell_id": "cell", "functions": rows},
                             checks.fresh_evidence(), "profile")
        assert any(row["function"][2] == "child" and row["callers"] for row in rows)
    assert path.is_file() and sys.getprofile() is None


def test_existing_profile_hook_is_not_overwritten():
    def hook(*args):
        return None

    sys.setprofile(hook)
    try:
        with pytest.raises(RuntimeError, match="absent prior"):
            timing.profiled_call(lambda: None)
        assert sys.getprofile() is hook
    finally:
        sys.setprofile(None)


@pytest.fixture
def serial_trace(retained):  # noqa: F811
    """Build declared timeline fixtures, independently of the admission code."""
    data = {key: None for key in checks.ROOT_FIELDS.split()}
    data.update(schema="simllm-pd-host-execution-native-v1", arm="uninstrumented", identity={},
                baseline_controls=[], cells=[], unique_steps=[], clock_advances=[],
                phase_counters={}, baseline_counters={}, profiles=[], probe_restoration=[],
                baseline_host_call={"wall_ns": 10, "process_cpu_ns": 7},
                probe_stack_empty=True, profile_hook_restored=True, weak_objects_alive_after=True,
                cuda_initialized_before=False, cuda_initialized_after=False, backend_runs=0)
    data["engines_before"] = retained["retention_before"]
    for index, engine in enumerate(data["engines_before"]):
        engine.update(scheduler_object_id=4000 + index, output_processor_object_id=5000 + index)
    data["engines_after"] = deepcopy(data["engines_before"])
    data["construction"] = [{"index": i, "identity": deepcopy(engine), "clock_ps": 0,
                             "elapsed_ns": i + 1, "current_rss_kib": 100,
                             "all_observed_objects_alive": True}
                            for i, engine in enumerate(data["engines_before"])]
    now, indices, previous_finished = 0, {"prefill": 0, "decode": 0}, {"prefill": [], "decode": []}

    def request(name, spec):
        nonlocal now
        data["clock_advances"].append({"before_ps": now, "after_ps": now})
        prompt = spec["prompt_tokens"]
        service = FROZEN["known_services_ps"][str(prompt)]
        start, prefill_end = now, now + service["prefill"]
        handoff = DeclaredKvHandoffPolicy(spec["handoff_ps"]).schedule(
            submitted_at_ps=prefill_end, request_id=name, kv_bytes=prompt * 49152)
        decode_start = handoff.completed_at_ps
        ends = tuple(decode_start + (i + 1) * service["decode"][0] for i in range(4))
        raw = DisaggregatedRequestTimeline(name, start, start, prefill_end, handoff, decode_start, ends).to_json()
        row = {"result": raw}
        for role, boundaries in (("prefill", [start, prefill_end]), ("decode", [decode_start, *ends])):
            identifier, engine = name + ":" + role, "simllm-" + role + "-0"
            raw[role + "_engine_id"], raw[role + "_internal_request_id"] = engine, identifier
            raw[role + "_step_count"] = len(boundaries) - 1
            row[role + "_record_indices"] = []
            for visit, (lower, upper) in enumerate(pairwise(boundaries)):
                index = indices[role]
                row[role + "_record_indices"].append(index)
                data["unique_steps"].append({"engine_id": engine,
                    "record": {"schema": "atlahs-closed-loop-step-v1", "step_index": index,
                               "virtual_time_ps": lower, "num_sampled": 1,
                               "preempted_request_ids": [], "finished_request_ids": previous_finished[role],
                               "scheduled": [{"request_id": identifier,
                                              "phase": "prefill" if role == "prefill" or visit == 0 else "decode",
                                              "num_new_tokens": prompt if role == "prefill" else 1,
                                              "num_cached_tokens": prompt if role == "decode" and visit == 0 else 0,
                                              "context_length": prompt if role == "prefill" else prompt + visit + 1}]},
                    "result": {"step_index": index, "step_latency_ps": upper - lower,
                               "completed_at_ps": upper, "request_metrics": [], "additive_visit_totals": None}})
                previous_finished[role] = [identifier] if visit == len(boundaries) - 2 else []
                data["clock_advances"].append({"before_ps": lower, "after_ps": upper})
                if role == "prefill":
                    data["clock_advances"].append({"before_ps": prefill_end, "after_ps": decode_start})
                indices[role] += 1
        raw.update(bootstrap_token_id=512, decode_token_ids=[512] * 4,
                   kv_transfer_params={"schema": "simllm-pd-kv-params-v1", "remote_request_id": name,
                                       "session_request_id": name, "remote_num_tokens": prompt, "bootstrap_token_id": 512,
                                       "do_remote_prefill": True, "do_remote_decode": False,
                                       "remote_engine_id": "simllm-prefill-0", "worker_tensor_transfer": False,
                                       "timing_authority": "simllm-declared-kv-handoff-v1"})
        now = ends[-1]
        return row

    for spec in FROZEN["historical_control_specs"]:
        row = request(spec["id"], spec)
        comparison = checks.compare_request(row["result"], FROZEN)
        row.update(id=spec["id"], comparison=comparison, comparison_sha256=sha(exact_json_bytes(comparison)))
        data["baseline_controls"].append(row)
    for spec in FROZEN["cells"]:
        start, advance = now, len(data["clock_advances"])
        rows = [request(f"{spec['id']}:request-{i}", spec) for i in range(spec["requests"])]
        steps = sorted([row for row in data["unique_steps"] if row["record"]["virtual_time_ps"] >= start],
                       key=lambda row: (0 if row["engine_id"] == "simllm-prefill-0" else 1, row["record"]["step_index"]))
        cell = {"id": spec["id"], "requests": rows, "start_ps": start, "end_ps": now,
                "wall_time_ns": 10, "wall_seconds": 1e-8, "arrival_mode": "serial-completion-admission",
                "offered_rate_requests_per_second": None,
                "engine_request_counts": {role: {"simllm-" + role + "-0": spec["requests"]} for role in ("prefill", "decode")},
                "step_identities": [[s["engine_id"], s["record"]["step_index"]] for s in steps],
                "clock_advance_indices": list(range(advance, len(data["clock_advances"])))}
        data["cells"].append({"cell": cell, "host_call": {"wall_ns": 12, "process_cpu_ns": 8},
                              "phase_counters": {}, "profile": None})
    data["unique_steps"].sort(key=lambda row: (0 if row["engine_id"] == "simllm-prefill-0" else 1, row["record"]["step_index"]))
    data["final_clock_ps"], data["sink_outcomes"] = now, []
    envelope, profile = checks.collective_selection(FROZEN)
    for role in ("prefill", "decode"):
        engine = "simllm-" + role + "-0"
        collections = {name: [] for name in FROZEN["sink_collections"]}
        for step in (s for s in data["unique_steps"] if s["engine_id"] == engine):
            i, service = step["record"]["step_index"], step["result"]["step_latency_ps"]
            collections["outcomes"].append(asdict(checks.StepNetworkOutcome(i, service, None, service, 0)))
            collections["locality_outcomes"].append(asdict(checks.StepLocalityOutcome(
                i, "execution-graph", False, 0, 0, 0, 0, 0, 0, 0, service, 0, 1)))
            collections["collective_timing_outcomes"].append(asdict(checks.StepCollectiveTimingOutcome(
                i, profile.profile_id, profile.bandwidth_bytes_per_second, profile.participant_latency_ps,
                profile.propagation_reference_ps, (), envelope_id="intra-node-fixed-cost-v1", arm="lower",
                evidence_class=envelope.arm_evidence_class("lower"))))
        data["sink_outcomes"].append({"engine_id": engine, "collections": collections})
    return json.loads(json.dumps(data))


def test_declared_fixture_admission_and_exact_baseline_hashes(serial_trace):
    checks.admit(serial_trace, FROZEN, checks.fresh_evidence())
    checks.compare_pair(serial_trace, {**serial_trace, "arm": "instrumented"}, FROZEN, checks.fresh_evidence())


@pytest.fixture
def instrumented_trace(serial_trace, tmp_path):
    """Exercise all instrumented joins with real small call-profile files."""
    data = deepcopy(serial_trace)
    data["arm"] = "instrumented"
    source = {"repository": FROZEN["source_sha256"], "native": FROZEN["native_source_sha256"],
              "origins": {name: str((run_study.ROOT / name).resolve()) for name in FROZEN["source_sha256"]},
              "native_origins": {name: str((tmp_path / name).resolve()) for name in FROZEN["native_source_sha256"]}}
    data["identity"] = {"sources_before": deepcopy(source), "sources_after": deepcopy(source)}
    phases = [entry["phase"] for entry in FROZEN["instrumentation"]["entries"]]

    def counters(requests):
        rows = {}
        for phase in phases:
            calls = requests * 5 if phase.startswith("native-") else requests
            rows[phase] = {"calls": calls, "wall_ns": calls * 3, "cpu_ns": calls * 2,
                           "child_wall_ns": calls, "child_cpu_ns": calls,
                           "exclusive_wall_ns": calls * 2, "exclusive_cpu_ns": calls}
        return rows

    data["baseline_counters"], data["phase_counters"] = counters(4), counters(14)
    path = tmp_path / "instrumented"
    (path / "profiles").mkdir(parents=True)
    for entry, spec in zip(data["cells"], FROZEN["cells"], strict=True):
        entry["phase_counters"] = counters(spec["requests"])
        if spec["profile_in_instrumented_arm"]:
            raw = path / "profiles" / (spec["id"] + ".pstats")
            _, rows = timing.profiled_call(lambda: sum(range(100)), profile_path=raw)
            export = {"schema": "simllm-host-call-profile-v1", "cell_id": spec["id"], "functions": rows}
            raw.with_suffix(".json").write_bytes(exact_json_bytes(export))
            profile = {"cell_id": spec["id"], "raw_path": raw.relative_to(path).as_posix(),
                       "export_path": raw.with_suffix(".json").relative_to(path).as_posix(),
                       "raw_sha256": sha(raw.read_bytes()), "export_sha256": sha(exact_json_bytes(export)),
                       "functions": len(rows)}
            entry["profile"] = profile
            data["profiles"].append(profile)
    for index, entry in enumerate(FROZEN["instrumentation"]["entries"]):
        field = "output_processor_object_id" if entry["phase"] == "native-output" else "scheduler_object_id"
        owners = ([engine[field] for engine in data["engines_before"]]
                  if entry["binding"] == "native-instance" else [6000 + index])
        for owner in owners:
            data["probe_restoration"].append({"phase": entry["phase"], "restored": True,
                "attribute": entry["target"].split(".")[-1].split(":")[-1],
                "original_own_attribute": entry["binding"] != "native-instance",
                "owner_object_id": owner, "source": {}})
    return data, path


def retain_fixture(data, path):
    def lines(rows):
        return b"".join(exact_json_bytes(row) + b"\n" for row in rows)

    raw = {"request-progress.jsonl": lines(checks.request_rows(data)),
           "construction-progress.jsonl": lines(data["construction"]),
           "probe-restoration.json": exact_json_bytes(data["probe_restoration"])}
    for entry in data["cells"]:
        raw[entry["cell"]["id"] + ".json"] = exact_json_bytes(entry["cell"])
    for role in ("prefill", "decode"):
        engine = "simllm-" + role + "-0"
        raw["engine-work/" + engine + "/step-records.jsonl"] = lines(
            [step["record"] for step in data["unique_steps"] if step["engine_id"] == engine])
    for name, content in raw.items():
        file = path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(content)
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def test_complete_instrumented_admission_and_frozen_corruptions(serial_trace, instrumented_trace, tmp_path):
    data, path = instrumented_trace
    checks.admit(data, FROZEN, checks.fresh_evidence())
    run_study.source_receipts(data, FROZEN, tmp_path / "vllm", checks.fresh_evidence())
    checks.compare_pair(serial_trace, data, FROZEN, checks.fresh_evidence())
    exports = run_study.raw_admission(path, data, FROZEN, checks.fresh_evidence(), retain_fixture(data, path))
    controls = run_study.mutations(serial_trace, data, exports, FROZEN, tmp_path, tmp_path / "vllm")
    assert [row["name"] for row in controls] == FROZEN["semantic_mutations"]
    assert len(controls) == 10 and all(row["rejected"] for row in controls)


def test_coherent_profile_export_truncation_cannot_override_raw_profile(instrumented_trace):
    data, path = instrumented_trace
    receipt = data["profiles"][0]
    export = path / receipt["export_path"]
    document = json.loads(export.read_bytes())
    document["functions"].pop()
    export.write_bytes(exact_json_bytes(document))
    receipt.update(export_sha256=sha(export.read_bytes()), functions=len(document["functions"]))
    with pytest.raises(checks.GuardFailure, match="profile:complete-export"):
        run_study.raw_admission(path, data, FROZEN, checks.fresh_evidence(), retain_fixture(data, path))


def test_unexpected_profile_file_rejects(instrumented_trace):
    data, path = instrumented_trace
    (path / "profiles/extra.json").write_bytes(b"{}")
    with pytest.raises(checks.GuardFailure, match="profile-file-domain"):
        run_study.raw_admission(path, data, FROZEN, checks.fresh_evidence(), retain_fixture(data, path))


@pytest.mark.parametrize("change", ["content", "addition", "removal"])
def test_final_reread_checks_original_bytes_and_complete_file_set(tmp_path, change):
    path = tmp_path / "raw.json"
    path.write_bytes(b"{}")
    receipts = {"raw.json": sha(path.read_bytes())}
    run_study.reread_tree(tmp_path, receipts, checks.fresh_evidence(), "reread")
    if change == "content":
        path.write_bytes(b"[]")
    elif change == "addition":
        (tmp_path / "new.json").write_bytes(b"{}")
    else:
        path.unlink()
    with pytest.raises(checks.GuardFailure, match="reread"):
        run_study.reread_tree(tmp_path, receipts, checks.fresh_evidence(), "reread")


def test_physics_is_bounded_before_native_output():
    run_study.physical_protocol(FROZEN, checks.fresh_evidence())
    changed = deepcopy(FROZEN)
    changed["known_services_ps"]["8"]["decode"][0] = 1
    with pytest.raises(checks.GuardFailure, match="surrogate-service"):
        run_study.physical_protocol(changed, checks.fresh_evidence())


@pytest.mark.parametrize("collection,field,value", [
    ("outcomes", "host_profile", "unfrozen-host"),
    ("collective_timing_outcomes", "arm", "upper"),
    ("collective_timing_outcomes", "envelope_id", "unfrozen-envelope"),
    ("collective_timing_outcomes", "bandwidth_bytes_per_second", 1),
    ("collective_timing_outcomes", "step_index", True),
])
def test_matching_wrong_sink_selections_cannot_pass(serial_trace, collection, field, value):
    serial_trace["sink_outcomes"][0]["collections"][collection][0][field] = value
    with pytest.raises(checks.GuardFailure, match="host-selection|frozen-profile|typed-profile"):
        checks.admit(serial_trace, FROZEN, checks.fresh_evidence())


def test_imported_native_origin_cannot_be_replaced_by_matching_disk_hash(tmp_path, monkeypatch):
    repo, package = tmp_path / "repo", tmp_path / "site/vllm"
    repo.mkdir()
    package.mkdir(parents=True)
    (repo / "local.py").write_text("value = 1\n")
    (package / "native.py").write_text("value = 2\n")
    wrong = tmp_path / "shadow.py"
    wrong.write_bytes((package / "native.py").read_bytes())
    modules = {"local": SimpleNamespace(__file__=repo / "local.py"),
               "vllm.native": SimpleNamespace(__file__=wrong)}
    monkeypatch.setattr(diagnostic, "ROOT", repo)
    monkeypatch.setattr(diagnostic.importlib, "import_module", modules.__getitem__)
    frozen = {"source_sha256": {"local.py": sha((repo / "local.py").read_bytes())},
              "native_source_sha256": {"vllm/native.py": sha(wrong.read_bytes())}}
    snapshot = diagnostic.source_snapshot(frozen, package)
    assert snapshot["native"] == frozen["native_source_sha256"]
    with pytest.raises(ValueError, match="another vLLM source"):
        diagnostic.source_admission(snapshot, frozen, package)


def test_prompt_fixture_is_checked_before_and_after(tmp_path, monkeypatch):
    trace = tmp_path / "trace.jsonl"
    trace.write_bytes(b"{\"prompt\":[1,2]}\n")
    monkeypatch.setattr(run_study.baseline, "TRACE_PATH", trace)
    frozen = {"source_sha256": {}, "historical_files_sha256": {}, "native_source_sha256": {},
              "frontend": {"model_id": "fixture/model", "model_revision": "fixed",
                           "model_config_sha256": sha(b"{}"), "fixture_sha256": sha(trace.read_bytes())}}
    config = tmp_path / "models--fixture--model/snapshots/fixed/config.json"
    config.parent.mkdir(parents=True)
    config.write_bytes(b"{}")
    args = SimpleNamespace(vllm_source=tmp_path / "vllm", hf_hub_cache=tmp_path)
    run_study.file_locks(args, frozen, checks.fresh_evidence(), "before")
    trace.write_bytes(b"{\"prompt\":[2,1]}\n")
    with pytest.raises(checks.GuardFailure, match="after:prompt-fixture"):
        run_study.file_locks(args, frozen, checks.fresh_evidence(), "after")


@pytest.mark.parametrize("mutation,guard", [
    (lambda d: d["unique_steps"].pop(), "steps:complete"),
    (lambda d: d["unique_steps"][0]["record"].update(sampled_request_ids=[]), "steps:complete"),
    (lambda d: d["unique_steps"][0]["result"].update(completed_at_ps=0), "steps:complete"),
    (lambda d: d["engines_before"][0]["workers"][0].update(device_absent=False), "engines:retained"),
    (lambda d: d["construction"][0].update(clock_ps=1), "construction:0:clock"),
    (lambda d: d["cells"][0]["cell"].update(offered_rate_requests_per_second=1), "rate"),
    (lambda d: d["cells"][0]["cell"].update(clock_advance_indices=[]), "advances"),
    (lambda d: d["sink_outcomes"][0]["collections"]["outcomes"][0].pop("host_launch_class"), "fields"),
    (lambda d: d["sink_outcomes"][0]["collections"]["outcomes"][0].update(makespan_ps=0), "service"),
    (lambda d: d.update(probe_stack_empty=False), "probe_stack_empty"),
    (lambda d: d.update(cuda_initialized_after=True), "cuda_initialized_after"),
])
def test_native_admission_rejects_semantic_corruption(serial_trace, mutation, guard):
    mutation(serial_trace)
    with pytest.raises(checks.GuardFailure, match=guard):
        checks.admit(serial_trace, FROZEN, checks.fresh_evidence())


def test_engine_qualified_mapping_rejects_opposite_role_same_public_request(serial_trace):
    steps = checks.normalized_steps(serial_trace)
    assert steps[0]["record"]["scheduled"][0]["request_id"] == serial_trace["baseline_controls"][0]["id"]
    serial_trace["unique_steps"][0]["record"]["scheduled"][0]["request_id"] = (
        serial_trace["baseline_controls"][0]["result"]["decode_internal_request_id"])
    with pytest.raises(checks.GuardFailure, match="engine-qualified"):
        checks.normalized_steps(serial_trace)


def test_complete_pair_comparison_preserves_nonopaque_metadata(serial_trace):
    changed = deepcopy(serial_trace)
    changed["arm"] = "instrumented"
    changed["baseline_controls"][0]["result"]["kv_transfer_params"]["remote_engine_id"] = "wrong"
    with pytest.raises(checks.GuardFailure, match="pair:requests"):
        checks.compare_pair(serial_trace, changed, FROZEN, checks.fresh_evidence())


@pytest.mark.parametrize("raw", [b'{"x":1,"x":1}', b'{"x": 1}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_noncanonical_or_nonfinite_data_is_never_rewritten(raw):
    with pytest.raises(ValueError):
        run_study.parse(raw, "fixture")


def test_probe_source_is_joined_to_actual_function_definition(tmp_path):
    source = timing.source_identity(target.append_progress)
    data = {"probe_restoration": [{"source": source, "attribute": "append_progress"}]}
    args = SimpleNamespace(vllm_source=tmp_path / "vllm")
    run_study.probe_source_admission(data, args, FROZEN, checks.fresh_evidence())
    source["first_line"] += 1
    with pytest.raises(checks.GuardFailure, match="definition"):
        run_study.probe_source_admission(data, args, FROZEN, checks.fresh_evidence())


def test_ordinary_execution_failure_retains_void_and_traceback(tmp_path, monkeypatch):
    def failure(*args, **kwargs):
        raise RuntimeError("diagnostic admission failed")

    monkeypatch.setattr(run_study, "file_locks", failure)
    monkeypatch.setattr(run_study.subprocess, "run", lambda *a, **k: None)
    def output(command, **kwargs):
        if command[1] == "show":
            return (run_study.HERE / command[2].rsplit("/", 1)[-1]).read_bytes()
        assert command[1:] == ["rev-parse", "HEAD"]
        return "1" * 40

    monkeypatch.setattr(run_study.subprocess, "check_output", output)
    result = run_study.execute(SimpleNamespace(output_root=tmp_path / "fresh"))
    assert result["verdict"] == "VOID" and result["behavioral_score"] is None
    assert result["failure"] == {"type": "RuntimeError", "reason": "diagnostic admission failed"}
    assert result["evidence"]["behavioral_relations"] == []
    assert "diagnostic admission failed" in (tmp_path / "fresh/exception.txt").read_text()


def test_post_capture_void_retains_first_receipts_and_failed_guard(tmp_path, monkeypatch, serial_trace):
    output_root = tmp_path / "fresh"

    def launch(args, arm, frozen):
        path = args.output_root / arm
        path.mkdir()
        raw = exact_json_bytes(serial_trace)
        (path / "native.json").write_bytes(raw)
        (path / "receipt.json").write_bytes(exact_json_bytes({"pid": 123, "native_sha256": sha(raw)}))
        return {"pid": 123}

    def fail(data, args, frozen, monitor, config, evidence):
        evidence.check("injected-native-failure", False)

    def output(command, **kwargs):
        if command[1] == "show":
            return (run_study.HERE / command[2].rsplit("/", 1)[-1]).read_bytes()
        assert command[1:] == ["rev-parse", "HEAD"]
        return "1" * 40

    monkeypatch.setattr(run_study, "launch", launch)
    monkeypatch.setattr(run_study, "admit_identity", fail)
    monkeypatch.setattr(run_study, "file_locks", lambda *a: tmp_path)
    monkeypatch.setattr(run_study.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(run_study.subprocess, "check_output", output)
    result = run_study.execute(SimpleNamespace(output_root=output_root))
    assert result["verdict"] == "VOID" and result["behavioral_score"] is None
    expected = sha(exact_json_bytes(serial_trace))
    assert result["raw_receipts"]["uninstrumented"]["native.json"] == expected
    assert json.loads((output_root / "first-raw-receipts.json").read_bytes()) == result["raw_receipts"]
    assert {"name": "uninstrumented:injected-native-failure", "passed": False} in result["evidence"]["unscored_guards"]
