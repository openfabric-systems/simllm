"""Retain independent evidence surfaces despite failures in another surface."""

import json
import subprocess
import xml.etree.ElementTree as ET
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from examples.pd_session_target_scale_v1.checks import Evidence, GuardFailure
from examples.shared_kv_handoff_v1 import capture, components, native, run_study
from examples.shared_kv_handoff_v1.common import HERE, PUBLICATIONS, read
from simllm.core.step import StepRecord, StepResult, step_record_to_json


def test_partial_capture_keeps_later_engine_and_runtime_after_first_failure(tmp_path, monkeypatch):
    journal = capture.EvidenceJournal(tmp_path)
    error = ValueError("first engine unreadable")
    first = NS(engine_id="first")
    record, result = StepRecord(step_index=0, virtual_time_ps=3), StepResult(0, 0, 3)
    second = NS(engine_id="second", executor=NS(step_records=[record], step_results=[result]))
    monkeypatch.setattr(capture, "sink_publications", lambda engine: {name: [] for name in PUBLICATIONS})
    real = journal.engine

    def observe(engine, *args, **kwargs):
        if engine is first:
            raise error
        return real(engine, *args, **kwargs)

    monkeypatch.setattr(journal, "engine", observe)
    runtime = NS(events=[{"completed_at_ps": 3}], visits=[{"completed_at_ps": 3}],
                 results=[({"operation_id": "second-step"}, result)], failure="original run failure")
    session = NS(engine_runtime=runtime, clock=NS(now_ps=3, advances=[{"before_ps": 0, "after_ps": 3}]))
    with pytest.raises(ValueError) as raised:
        journal.partial(session, [first, second])
    assert raised.value is error
    summary = read(tmp_path / "partial-evidence.json")
    assert summary["capture_failures"] == [{"surface": "engine:first", "type": "ValueError", "message": str(error)}]
    assert summary["runtime_failure"] == "original run failure"
    assert summary["engine_positions"]["second"]["records"] == 1
    assert summary["runtime_positions"] == dict.fromkeys(("events", "visits", "completed", "clock_advances"), 1)
    assert journal.engine_rows[0]["values"]["records"][0]["virtual_time_ps"] == 3
    assert journal.runtime_rows[0]["values"]["completed"][0]["receipt"] == {"operation_id": "second-step"}
    assert (tmp_path / "engine-evidence.jsonl").is_file() and (tmp_path / "runtime-evidence.jsonl").is_file()


def observed_owner(tmp_path):
    frozen, deadline = (json.loads((HERE / name).read_bytes()) for name in ("expectations.json", "deadline-expectations.json"))
    spec = frozen["native_processes"][0]
    args = NS(output_root=tmp_path, htsim_rnic="unused-no-process")
    owner = native.recorded_owner(args, spec, frozen, deadline, [])
    owner._session = NS(transcript=(("request", b"latest-request"), ("response", b"latest-response")), stderr=b"native diagnostic")
    return owner


def test_failed_observation_write_still_retains_latest_transcript(tmp_path, monkeypatch):
    owner = observed_owner(tmp_path)
    error = OSError("observation log full")
    monkeypatch.setattr(native, "network_snapshot", lambda _: {"full": "snapshot"})
    monkeypatch.setattr(native, "append_progress", lambda *args: (_ for _ in ()).throw(error))
    with pytest.raises(OSError) as raised:
        owner.capture("progress")
    assert raised.value is error
    path = tmp_path / "network-transcript"
    assert (path / "00000-request.bin").read_bytes() == b"latest-request"
    assert (path / "00001-response.bin").read_bytes() == b"latest-response"
    assert (path / "stderr.bin").read_bytes() == b"native diagnostic"
    assert owner._capture_failures[0]["surface"] == "observation"


def test_native_failure_remains_primary_when_both_capture_surfaces_fail(tmp_path, monkeypatch):
    owner = observed_owner(tmp_path)
    primary = RuntimeError("native failure")
    monkeypatch.setattr(native, "network_snapshot", lambda _: (_ for _ in ()).throw(ValueError("snapshot unreadable")))
    monkeypatch.setattr(native, "save_transcript", lambda *args: (_ for _ in ()).throw(OSError("wire disk full")))
    monkeypatch.setattr(native, "append_progress", lambda *args: None)
    with pytest.raises(RuntimeError) as raised:
        owner.invoke("progress", lambda *args, **kwargs: (_ for _ in ()).throw(primary), (), through_ps=None)
    assert raised.value is primary
    assert [row["surface"] for row in owner._capture_failures] == ["observation", "transcript"]


def test_engine_journal_retains_detached_complete_prefixes(tmp_path, monkeypatch):
    journal = capture.EvidenceJournal(tmp_path)
    publications = {name: [] for name in PUBLICATIONS}
    publications["outcomes"] = [{"nested": [1]}]
    monkeypatch.setattr(capture, "sink_publications", lambda _: deepcopy(publications))
    engine = NS(engine_id="engine", executor=NS(step_records=[], step_results=[]))
    journal.engine(engine, "retired", 5)
    publications["outcomes"][0]["nested"].append(2)
    assert journal.engine_rows[0]["values"]["outcomes"] == [{"nested": [1]}]
    assert journal.engine_rows[0]["starts"]["outcomes"] == 0
    assert journal.engine_rows[0]["stops"]["outcomes"] == 1


def test_early_worker_failure_survives_failed_first_receipts(tmp_path, monkeypatch):
    primary = RuntimeError("worker setup failed before monitor creation")
    spec, monitors = {"id": "worker"}, {}

    def launch(*args):
        (tmp_path / spec["id"]).mkdir()
        raise primary

    monkeypatch.setattr(run_study, "launch", launch)
    monkeypatch.setattr(run_study, "receipts", lambda *args: (_ for _ in ()).throw(OSError("receipt unreadable")))
    with pytest.raises(RuntimeError) as raised:
        run_study.capture(NS(output_root=tmp_path), spec, {}, monitors, {}, {})
    assert raised.value is primary
    assert monitors["worker"]["receipt_failure"]["message"] == "receipt unreadable"


def test_component_timeout_survives_failed_first_receipt_write(tmp_path, monkeypatch):
    primary = subprocess.TimeoutExpired("component fixtures", 180, output=b"partial test output")
    real_write = components.write

    def write(path, value):
        if path.name == "components-first-receipts.json":
            raise OSError("receipt disk full")
        return real_write(path, value)

    monkeypatch.setattr(components, "write", write)
    monkeypatch.setattr(components.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(primary))
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        components.capture(tmp_path / "components")
    assert raised.value is primary
    assert primary.output == b"partial test output"
    assert primary.receipt_failure["message"] == "receipt disk full"


@pytest.mark.parametrize("surface", ["raw", "checks", "summary"])
@pytest.mark.parametrize("has_primary", [False, True])
def test_reporting_failure_voids_score_and_preserves_other_receipts(tmp_path, monkeypatch, capsys, surface, has_primary):
    for name in ("broken", "good"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "raw.bin").write_bytes(name.encode())
    first = {name: run_study.receipts(tmp_path / name) for name in ("broken", "good")}
    original = {"type": "RuntimeError", "message": "first native failure", "traceback": "retained"} if has_primary else None
    real_receipts, real_write = run_study.receipts, run_study.write

    def receipts(path):
        if surface == "raw" and path.name == "broken":
            raise OSError("cannot read one raw tree")
        return real_receipts(path)

    def write(path, value):
        if path.name == surface + ".json":
            raise OSError("cannot write " + surface)
        return real_write(path, value)

    monkeypatch.setattr(run_study, "receipts", receipts)
    monkeypatch.setattr(run_study, "write", write)
    summary = run_study.finalize(tmp_path, {"behavioral_instance_total": 12}, ["native"], Evidence(["native"]),
                                {}, first, {}, {}, None, original)
    assert summary["verdict"] == "VOID" and summary["behavioral_score"] is None
    assert summary["final_receipts"]["good"] == first["good"]
    assert summary["first_receipts"] == first and summary["reporting_failures"]
    if has_primary:
        assert summary["failure"] == original
    if surface != "summary":
        assert read(tmp_path / "summary.json") == summary
    else:
        assert not (tmp_path / "summary.json").exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["verdict"] == "VOID" and printed["failure"] == summary["failure"]


@pytest.mark.parametrize("kind", ["truncated", "changed", "missing", "extra", "encoding"])
def test_native_step_streams_reject_changed_or_missing_original_records(tmp_path, kind):
    record = step_record_to_json(StepRecord(0, 0))
    data = {"retained_before": [{"engine_id": "first"}, {"engine_id": "second"}],
            "steps": [{"engine_id": "first", "record": record}]}
    paths = []
    for engine in ("first", "second"):
        path = tmp_path / "engine-work" / engine / "step-records.jsonl"
        path.parent.mkdir(parents=True)
        path.write_bytes((json.dumps(record) + "\n").encode("ascii") if engine == "first" else b"")
        paths.append(path)
    run_study.admit_step_streams(tmp_path, data, "valid", Evidence([]))
    if kind == "truncated":
        paths[0].write_bytes(b"")
    elif kind == "changed":
        paths[0].write_bytes((json.dumps(step_record_to_json(StepRecord(1, 2))) + "\n").encode("ascii"))
    elif kind == "missing":
        paths[1].unlink()
    elif kind == "extra":
        (tmp_path / "step-records.jsonl").write_bytes(b"")
    else:
        paths[0].write_bytes((json.dumps(record, separators=(",", ":")) + "\n").encode("ascii"))
    with pytest.raises(GuardFailure):
        run_study.admit_step_streams(tmp_path, data, "corrupt", Evidence([]))


@pytest.mark.parametrize("surface", ["raw-change", "raw-delete", "root-change"])
def test_final_reread_is_fatal_when_a_first_receipt_changes(tmp_path, surface):
    path = tmp_path / "worker"
    path.mkdir()
    raw, root = path / "raw.bin", tmp_path / "source-manifest.json"
    raw.write_bytes(b"original native evidence")
    root.write_bytes(b"original source evidence")
    first = {"worker": run_study.receipts(path)}
    root_receipts = {root.name: {"sha256": run_study.sha(root), "bytes": root.stat().st_size}}
    if surface == "raw-change":
        raw.write_bytes(b"changed native evidence")
    elif surface == "raw-delete":
        raw.unlink()
    else:
        root.write_bytes(b"changed source evidence")
    summary = run_study.finalize(tmp_path, {"behavioral_instance_total": 12}, [], Evidence([]),
                                {}, first, root_receipts, {}, None, None)
    assert summary["verdict"] == "VOID" and summary["behavioral_score"] is None
    assert summary["first_receipts"] == first and summary["root_receipts"] == root_receipts
    assert summary["failure"]["type"] == "GuardFailure"
    if surface.startswith("raw"):
        assert summary["final_receipts"]["worker"] != first["worker"]
    else:
        assert summary["final_root_receipts"][root.name] != root_receipts[root.name]


def test_protocol_failure_keeps_initial_root_receipts(tmp_path, monkeypatch):
    primary = ValueError("preflight source mismatch")
    originals = {}

    def freeze(*args):
        (tmp_path / "before-packages.json").write_bytes(b"captured package bytes")
        (tmp_path / "runtime-probe.stdout").write_bytes(b"captured runtime bytes")
        originals.update(run_study.receipts(tmp_path))
        raise primary

    monkeypatch.setattr(run_study.protocol, "freeze", freeze)
    retained = {}
    with pytest.raises(ValueError) as error:
        run_study.capture_protocol(NS(output_root=tmp_path), {}, {}, {}, {}, Evidence([]), retained)
    assert error.value is primary
    assert read(tmp_path / "protocol-first-receipts.json") == originals
    assert {name: retained[name] for name in originals} == originals
    assert "protocol-first-receipts.json" in retained


@pytest.mark.parametrize("kind", ["missing-sequence", "skipped-endpoint", "missing-cleanup", "duplicate-identity", "missing-service-vector"])
def test_component_catalog_requires_every_exact_unskipped_fixture(tmp_path, kind):
    path = tmp_path / "components"
    path.mkdir()
    frozen = json.loads((HERE / "expectations.json").read_bytes())
    aliases = json.loads((HERE / "alias-expectations.json").read_bytes())
    suite = ET.Element("testsuite")
    names = sorted({"test_" + name for patterns in components.CASES.values() for name in patterns})
    for name in names:
        ET.SubElement(suite, "testcase", classname="tests.test_shared_kv_handoff", name=name)
    for name in aliases["fatal_controls"]:
        ET.SubElement(suite, "testcase", classname="tests.test_shared_handoff_study",
                      name="test_alias_fatal_controls_reject_before_comparison[" + name + "]")
    for name in components.service_cases():
        ET.SubElement(suite, "testcase", classname="tests.test_shared_handoff_study", name=name)
    components.write(path / "process.json", {"exit_code": 0})

    def retain():
        ET.ElementTree(suite).write(path / "software.xml")
        components.write(tmp_path / "components-first-receipts.json", components.receipts(path))

    retain()
    catalog = components.admit(path, frozen, aliases, Evidence([]))
    assert catalog["behavioral_instances"] == 0 and catalog["behavioral_score"] is None
    if kind == "duplicate-identity":
        suite.append(deepcopy(suite[0]))
    else:
        suffix = {"missing-sequence": "poison_and_reap[sequence]",
                  "skipped-endpoint": "before_native_launch[endpoint-count]",
                  "missing-cleanup": "preserves_first_failure[body-failure]",
                  "missing-service-vector": "test_service_vector_rejects_reference_engine_misbinding"}[kind]
        case = next(row for row in suite if row.attrib["name"].endswith(suffix))
        if kind == "skipped-endpoint":
            ET.SubElement(case, "skipped")
        else:
            suite.remove(case)
    retain()
    with pytest.raises(GuardFailure):
        components.admit(path, frozen, aliases, Evidence([]))


@pytest.mark.parametrize("kind", ["success", "nonzero", "timeout"])
@pytest.mark.parametrize("capture_failure", [False, True])
def test_runtime_probe_retains_output_and_preserves_primary_failure(tmp_path, monkeypatch, kind, capture_failure):
    protocol = run_study.protocol
    primary = subprocess.TimeoutExpired("runtime identity", 60, output=b"partial output", stderr=b"partial diagnostic")
    stdout = b'{"identity":"retained"}\n' if kind != "timeout" else primary.output
    stderr = b"probe diagnostic" if kind != "timeout" else primary.stderr
    real_write_bytes = type(tmp_path).write_bytes

    def write_bytes(path, value):
        if capture_failure and path.name == "runtime-probe.stdout":
            raise OSError("stdout storage failed")
        return real_write_bytes(path, value)

    def launch(command, **kwargs):
        assert read(tmp_path / "runtime-probe-command.json") == command
        assert kwargs["check"] is False and kwargs["timeout"] == 60
        if kind == "timeout":
            raise primary
        return subprocess.CompletedProcess(command, 7 if kind == "nonzero" else 0, stdout, stderr)

    monkeypatch.setattr(type(tmp_path), "write_bytes", write_bytes)
    monkeypatch.setattr(protocol.subprocess, "run", launch)
    args = NS(native_python="scripted-runtime-no-executable", output_root=tmp_path)
    if kind == "success" and not capture_failure:
        assert protocol.runtime_probe(args) == {"identity": "retained"}
    else:
        expected = subprocess.TimeoutExpired if kind == "timeout" else subprocess.CalledProcessError if kind == "nonzero" else OSError
        with pytest.raises(expected) as error:
            protocol.runtime_probe(args)
        if kind == "timeout":
            assert error.value is primary
        elif kind == "nonzero":
            assert error.value.returncode == 7 and error.value.output == stdout
        if capture_failure:
            assert error.value.receipt_failure[0]["surface"] == "stdout"
    assert (tmp_path / "runtime-probe.stderr").read_bytes() == stderr
    outcome = read(tmp_path / "runtime-probe-outcome.json")
    assert outcome["exit_code"] == (None if kind == "timeout" else 7 if kind == "nonzero" else 0)
    if not capture_failure:
        assert (tmp_path / "runtime-probe.stdout").read_bytes() == stdout
