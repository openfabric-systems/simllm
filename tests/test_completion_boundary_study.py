"""Pure and fake-command checks for the frozen study runner, never native runs."""

import copy
import importlib.util
import json
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "completion_boundary_study", Path(__file__).resolve().parents[1]
    / "examples/completion_boundary_v1/run_study.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def counters(count):
    return {"native_session_constructed": 1, "native_posts": count,
            "legacy_ledger_constructed": 0, "legacy_posts": 0,
            "legacy_mutations": 0, "legacy_aborts": 0}


def fake_native(binary, hardware, out, name, point, *, fresh_release=None):
    rows = []

    def add(label, payload, release, finish):
        rows.append({"sequence": len(rows) + 1, "operation_id": label, "payload_bytes": payload,
                     "start_time_ps": release, "completion_time_ps": finish,
                     "fct_ps": finish - release, "completion_status": "success"})

    if "length" in point:
        release = 0
        for index in range(point["length"]):
            finish = release + study.flow_oracle(point["payload"], point["rate"])
            add(f"chain-{index}", point["payload"], release, finish)
            release = finish
    elif fresh_release is not None:
        add("successor", point["payload"], fresh_release,
            fresh_release + study.flow_oracle(point["payload"], point["rate"]))
    else:
        trigger = study.flow_oracle(4096, point["rate"])
        background = study.flow_oracle(1_048_576, point["rate"])
        add("trigger", 4096, 0, trigger)
        add("background", 1_048_576, 0, background)
        add("successor", point["payload"], trigger, background + study.flow_oracle(point["payload"], point["rate"]))
    return {"rows": rows, "drain": {"completion_rows": rows, "authority_counters": counters(len(rows)),
            "quiesced_at_ps": max(row["completion_time_ps"] for row in rows),
            "sq_high_watermarks": [1, 2 if len(rows) == 3 and "length" not in point else 1]}}


def test_population_and_independent_arithmetic():
    points = study.population()
    assert {key: len(points[key]) for key in ("chain", "retention", "old_transcripts", "live", "off", "rejections")} == {
        "chain": 8, "retention": 4, "old_transcripts": 4, "live": 8, "off": 8, "rejections": 4}
    assert [study.live_oracle(batch, rate) for rate in study.RATES for batch in study.BATCHES] == [
        50_662_400, 51_993_600, 49_331_200, 49_996_800]
    assert min(study.live_floor(batch, rate) - study.live_oracle(batch, rate)
               for rate in study.RATES for batch in study.BATCHES) == 14_627_840
    assert points["declared_inputs"]["new_graph_counts"]["active_calc_joins"] == 0


def flow_comparison_fixture():
    evidence = study.Evidence()
    for profile in ("rnic-nn", "rnic-cn"):
        for batch in study.BATCHES:
            for rate in study.RATES:
                for step in range(3):
                    for message in range(16):
                        evidence.flows.append({"profile": profile, "batch": batch, "rate": rate,
                            "step": step, "operation_id": f"op-{message // 4}",
                            "flow_id": f"declared-message-{message}", "source": message % 2,
                            "destination": 1 - message % 2, "tag": message, "payload_bytes": 4096,
                            "sequence": message + 1 if profile == "rnic-nn" else 16 - message,
                            "fct_ps": 1000 + message if profile == "rnic-nn" else 2000 + 2 * message})
    return evidence


def test_cross_profile_join_uses_messages_and_preserves_local_sequences():
    evidence = flow_comparison_fixture()
    before = copy.deepcopy(evidence.flows)
    study.normalize_flow_metrics(evidence)
    assert all(guard["passed"] for guard in evidence.guards)
    for row, original in zip(evidence.flows, before, strict=True):
        assert all(row[key] == value for key, value in original.items())
        if row["profile"] == "rnic-cn":
            assert row["matched_nn_sequence"] == 17 - row["sequence"]
            assert row["normalized_fct_to_matched_nn"] == {"numerator": 2, "denominator": 1}


@pytest.mark.parametrize("fault", ("missing", "duplicate", "missing-field", "operation_id", "flow_id",
                                   "source", "destination", "tag", "payload_bytes", "batch", "rate", "step"))
def test_bad_cross_profile_message_inventory_is_fatal(fault):
    evidence = flow_comparison_fixture()
    row = evidence.flows[-1]
    if fault == "missing":
        evidence.flows.pop()
    elif fault == "duplicate":
        evidence.flows.append(dict(row))
    elif fault == "missing-field":
        del row["flow_id"]
    else:
        row[fault] = "wrong" if isinstance(row[fault], str) else row[fault] + 100
    study.normalize_flow_metrics(evidence)
    assert evidence.summary()["status"] == "void"
    assert evidence.summary()["behavioral_score"] is None
    assert not any("normalized_fct_to_matched_nn" in row for row in evidence.flows)


def test_fatal_void_removes_score_and_collection_continues():
    evidence = study.Evidence()

    def bad():
        raise RuntimeError("terminal transport")

    assert evidence.capture("bad", bad) is None
    assert evidence.capture("later", lambda: 19) == 19
    evidence.relation("retention", "positive", True)
    assert evidence.summary()["status"] == "void"
    assert evidence.summary()["behavioral_score"] is None
    assert [row["name"] for row in evidence.configurations] == ["bad", "later"]
    valid = study.Evidence()
    valid.relation("retention", "negative", False)
    assert valid.summary()["status"] == "refuted"


@pytest.mark.parametrize("payload,rate", ((8192, 200_000_000_000), (16384, 400_000_000_000)))
def test_retention_guards_and_refutation(payload, rate):
    point = {"payload": payload, "rate": rate}
    retained = fake_native(None, None, None, "retained", point)
    fresh = fake_native(None, None, None, "fresh", point, fresh_release=retained["rows"][0]["completion_time_ps"])
    evidence = study.Evidence()
    study.check_retention(evidence, "pair", point, retained, fresh)
    assert evidence.summary()["status"] == "passed"
    retained["drain"]["sq_high_watermarks"][1] = 1
    evidence = study.Evidence()
    study.check_retention(evidence, "pair", point, retained, fresh)
    assert evidence.summary()["status"] == "refuted"


def test_native_chain_rejects_stale_eligibility_even_with_correct_fct():
    point = {"payload": 4096, "length": 2, "rate": 400_000_000_000}
    result = fake_native(None, None, None, "chain", point)
    result["rows"][1]["start_time_ps"] += 1
    evidence = study.Evidence()
    study.check_chain(evidence, "chain", point, result)
    assert evidence.summary()["status"] == "void"


def test_action_evidence_joins_native_lifecycle_without_rewriting_time():
    from test_session_projection import projected

    from simllm.backends.session_projection import build_session_evidence

    native = study.plain(build_session_evidence(*projected()))
    evidence = study.Evidence()
    study.check_action_evidence(evidence, "fake", native)
    assert all(row["passed"] for row in evidence.guards)
    changed = copy.deepcopy(native)
    changed["native_events"][0]["timestamp_ps"] += 1
    evidence = study.Evidence()
    study.check_action_evidence(evidence, "changed", changed)
    assert evidence.summary()["status"] == "void"


def test_legacy_fake_transport_preserves_exact_frame_bytes_and_aborts(monkeypatch, tmp_path):
    import simllm.backends._child_process as child

    class FakeProcess:
        last = None

        def __init__(self, command, timeout_s):
            self.pending = b""
            self.stderr = b"retained stderr"
            self.aborted = False
            FakeProcess.last = self
            assert timeout_s == 60

        def write(self, raw):
            request = json.loads(raw[4:])
            self.pending += study.frame({"schema": study.SCHEMA, "status": "ok", "verb": request["verb"]})

        def read_exact(self, size):
            raw, self.pending = self.pending[:size], self.pending[size:]
            return raw

        def finish(self):
            return 0

        def abort(self):
            self.aborted = True

    monkeypatch.setattr(child, "OwnedBinaryProcess", FakeProcess)
    requests = study.legacy_requests("0" * 64, {"payload": 4096, "rate": study.RATES[0]})
    wire = study.run_legacy(Path("fake"), requests, tmp_path)
    assert FakeProcess.last.aborted
    assert len(json.loads((tmp_path / "frames/index.json").read_text())) == 10
    assert (tmp_path / "frames/stderr.bin").read_bytes() == b"retained stderr"
    assert wire == b"".join(study.frame({"schema": study.SCHEMA, "status": "ok", "verb": request["verb"]}) for request in requests)


def test_timeout_retains_both_raw_streams(monkeypatch, tmp_path):
    import simllm.backends._child_process as child

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(["fake"], 60, output=b"stdout\x00\xff", stderr=b"stderr\x01\xfe")

    monkeypatch.setattr(child, "run_owned_process", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        study.run_command(["fake"], tmp_path / "failure")
    assert (tmp_path / "failure.stdout.bin").read_bytes() == b"stdout\x00\xff"
    assert (tmp_path / "failure.stderr.bin").read_bytes() == b"stderr\x01\xfe"


def test_live_flow_floor_catches_early_peer_hidden_by_valid_step_total(monkeypatch):
    point = {"profile": "rnic-cn", "batch": 16, "rate": 400_000_000_000}
    latency = study.live_floor(point["batch"], point["rate"])
    floor = 4096 * 8 * 10**12 // point["rate"] + 4_000_000
    steps = []
    for index in range(3):
        release, complete = index * latency, (index + 1) * latency
        rows = [{"sequence": sequence, "payload_bytes": 4096, "completion_status": "success",
                 "start_time_ps": release, "completion_time_ps": release + floor,
                 "fct_ps": floor} for sequence in range(1, 17)]
        drain = {"completion_rows": rows, "authority_counters": counters(16),
                 "quiesced_at_ps": complete, "sq_high_watermarks": [1] * 8}
        metrics = [{"request_id": f"r{i}", "completed_at_ps": complete, "latency_ps": latency,
                    "ttft_ps": latency, "tpot_ps": None if index == 0 else
                    {"numerator": latency, "denominator": 1}} for i in range(16)]
        steps.append({"result": {"step_latency_ps": latency, "completed_at_ps": complete,
                                 "request_metrics": metrics}, "released_at_ps": release,
                      "graph_operations": 8, "artifact_count": 6, "message_count": 16,
                      "session_evidence": {"native_drain": drain, "execution_result": {
                          "completed_at_ps": complete, "quiesced_at_ps": complete}}})
    early = steps[0]["session_evidence"]["native_drain"]["completion_rows"][0]
    early["fct_ps"] -= 1
    early["completion_time_ps"] -= 1
    monkeypatch.setattr(study, "check_action_evidence", lambda *args: None)
    evidence = study.Evidence()
    study.check_live(evidence, "masked", point, {"status": "ok", "session_count": 3, "steps": steps})
    assert [row["name"] for row in evidence.guards if not row["passed"]] == ["live-flow-floor:masked:0:1"]
    assert evidence.summary()["status"] == "void"


def test_absent_cn_scenario_retains_pure_projection_before_refusing(monkeypatch, tmp_path):
    import simllm.backends.htsim_rnic as native

    def forbidden(*args, **kwargs):
        raise AssertionError("the absent CN option must refuse before starting a child")

    monkeypatch.setattr(native, "run_owned_process", forbidden)
    for key in ("SIMLLM_HTSIM_RNIC", "SIMLLM_TXT2BIN", "SIMLLM_CHILD_LIFETIME_MARKER_DIR",
                "SIMLLM_CHILD_LIFETIME_RUN_NONCE"):
        monkeypatch.setenv(key, "unused")
    manifest = tmp_path / "identity.json"
    manifest.write_text("{}")
    out = tmp_path / "scenario"
    result = study.scenario_worker({"code_root": str(Path(__file__).resolve().parents[1]),
        "identity_manifest": str(manifest), "out": str(out), "binary": "fake-native",
        "txt2bin": "fake-txt2bin", "name": "pure-refusal", "profile": "rnic-cn",
        "rate": 400_000_000_000, "batch": 16, "session": False})
    assert result["status"] == "rejected" and result["exception_type"] == "RuntimeError"
    assert "starts a fresh backend process" in result["error"]
    assert result["native_commands"] == result["child_markers"] == []
    assert result["session_count"] == 0
    assert (out / "step-0.graph.json").is_file()
    assert len(json.loads((out / "step-0.projection.json").read_text())["artifacts"]) == 6


def test_off_controls_fail_closed_on_byte_changes_or_started_child():
    before = {"status": "ok", "steps": [{"result": {"time": 1}}] * 3,
              "artifact_inventory": [{"path": "flow.bin", "sha256": "same"}]}
    after = copy.deepcopy(before)
    after["artifact_inventory"][0]["sha256"] = "changed"
    evidence = study.Evidence()
    study.compare_off(evidence, "bytes", before, after, rejection=False)
    assert evidence.summary()["status"] == "void"
    refused = {"status": "rejected", "exception_type": "RuntimeError",
               "error": "each artifact starts a fresh backend process", "native_commands": [],
               "child_markers": [], "session_count": 0}
    started = {**refused, "native_commands": [["child"]]}
    evidence = study.Evidence()
    study.compare_off(evidence, "refusal", refused, started, rejection=True)
    assert evidence.summary()["status"] == "void"


def test_full_fake_population_continues_after_native_failure(monkeypatch, tmp_path):
    calls = []

    def native(*args, **kwargs):
        calls.append(args[3])
        if len(calls) == 1:
            raise RuntimeError("first configuration failed")
        return fake_native(*args, **kwargs)

    def scenario(args, identity, point, name, *, session, baseline):
        calls.append(name)
        if not session and point["profile"] == "rnic-cn":
            return {"status": "rejected", "exception_type": "RuntimeError",
                    "error": "each artifact starts a fresh backend process", "native_commands": [],
                    "child_markers": [], "session_count": 0}
        return {"status": "ok", "steps": [{"result": {"time": 1}}] * 3, "artifact_inventory": []}

    def live(evidence, name, point, result):
        latency = (study.live_oracle if point["profile"] == "rnic-nn" else study.live_floor)(point["batch"], point["rate"])
        if point["profile"] == "rnic-nn":
            for index in range(3):
                evidence.exact("live-nn-step", f"{name}:{index}", latency, latency)
        return [latency] * 3

    monkeypatch.setattr(study, "run_native", native)
    monkeypatch.setattr(study, "launch_scenario", scenario)
    monkeypatch.setattr(study, "check_live", live)
    monkeypatch.setattr(study, "run_legacy", lambda *args: b"same framed bytes")
    args = SimpleNamespace(out=tmp_path, candidate_binary=Path("candidate"), baseline_binary=Path("base"))
    report = study.execute(args, {"hardware": {"effective_hardware_sha256": "0" * 64}})
    assert report["status"] == "void" and report["behavioral_score"] is None
    assert len(report["configurations"]) == 56
    assert len(report["unscored_compatibility_controls"]) == 16
    assert calls[-1].startswith("rejections-")
    assert Counter(row["family"] for row in report["behavioral_relations"])["retention"] == 4
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "void"
