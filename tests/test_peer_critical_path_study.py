"""Software fixtures for original capture admission, separate from campaign evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from examples.peer_critical_path_v1 import capture, run_study
from examples.routed_compute_v1.run_study import Evidence


def canonical(path, value):
    path.write_bytes((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode())


@pytest.fixture(scope="module")
def captured_fixtures(tmp_path_factory):
    root = run_study.ROOT
    names = subprocess.check_output(["git", "ls-files"], cwd=root).decode().splitlines()
    identity = {
        "commit": "software-fixture-only",
        "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                  for name in names if (root / name).is_file()},
    }
    output = tmp_path_factory.mktemp("peer-critical-capture-fixtures")
    rows = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(capture, "source", lambda _: identity)
        patch.setattr(sys, "path", list(sys.path))
        for topology in ("direct", "switched"):
            for arm in ("candidate-off", "candidate-on"):
                directory = output / (topology + "-" + arm)
                arguments = SimpleNamespace(
                    code_root=root, expected_commit=identity["commit"], output=directory,
                    arm=arm, topology=topology, rate=25000000000, donors=3,
                    payload=1024, native_library=None,
                )
                assert capture.run(arguments) == 0, (directory / "capture.json").read_text()
                rows[(topology, arm)] = directory
    return identity, rows


def admit(directory, identity, evidence, *, topology="switched", arm="candidate-on"):
    return run_study.admit_capture(
        evidence, directory, arm=arm, topology=topology, rate=25000000000,
        donors=3, payload=1024, expected_source=identity, code_root=run_study.ROOT,
    )


@pytest.mark.parametrize("topology", ["direct", "switched"])
def test_actual_capture_admits_and_selected_request_history_matches_absent(captured_fixtures, tmp_path, topology):
    identity, fixtures = captured_fixtures
    results = []
    for arm in ("candidate-off", "candidate-on"):
        evidence = Evidence(tmp_path / arm)
        results.append(admit(fixtures[(topology, arm)], identity, evidence, topology=topology, arm=arm))
        assert all(row["passed"] for row in evidence.guards)
        assert len(evidence.oracles) == (2 if arm.endswith("-on") else 0)
        assert not evidence.relations
    assert run_study.strip_reporting(results[0]) == run_study.strip_reporting(results[1])


def reseal_fixture(directory, name, value):
    """Alter a new test fixture to reach semantic checks after receipt admission."""
    canonical(directory / name, value)
    captured = run_study.read(directory / "capture.json")
    captured["first_receipts"][name] = run_study.receipt(directory / name)
    canonical(directory / "capture.json", captured)
    (directory / "first-receipts.jsonl").write_bytes(b"".join(
        (json.dumps({"path": path, **captured["first_receipts"][path]}, sort_keys=True) + "\n").encode()
        for path in captured["receipt_order"]
    ))


@pytest.mark.parametrize("mutation", [
    "raw-byte", "missing-file", "extra-file", "journal-loss", "journal-order",
    "capture-void", "source", "origin", "unlocked-origin", "input-rate",
    "input-width", "packet-time", "causal-parent", "phase", "step-input",
    "step-completion", "request-total", "request-identity", "request-history", "final-drain",
])
def test_corrupt_capture_cannot_be_admitted(captured_fixtures, tmp_path, mutation):
    identity, fixtures = captured_fixtures
    directory = tmp_path / "altered-software-fixture"
    shutil.copytree(fixtures[("switched", "candidate-on")], directory)
    if mutation == "raw-byte":
        path = directory / "metrics.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "missing-file":
        (directory / "metrics.json").unlink()
    elif mutation == "extra-file":
        (directory / "unrecorded.json").write_bytes(b"{}\n")
    elif mutation in ("journal-loss", "journal-order"):
        path = directory / "first-receipts.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b"".join(lines[1:] if mutation == "journal-loss" else reversed(lines)))
    elif mutation == "capture-void":
        value = run_study.read(directory / "capture.json")
        value["verdict"] = "VOID"
        canonical(directory / "capture.json", value)
    else:
        names = {
            "source": "source-after.json", "origin": "loaded-origins.json",
            "unlocked-origin": "loaded-origins.json", "input-rate": "inputs.json",
            "input-width": "inputs.json", "packet-time": "component-visible.json",
            "causal-parent": "component-visible.json", "phase": "component-visible.json",
            "step-input": "step-1-input.json", "step-completion": "step-1-output.json",
            "request-total": "metrics.json", "request-identity": "metrics.json",
            "request-history": "request-breakdowns.json", "final-drain": "serving-drained.json",
        }
        name = names[mutation]
        value = run_study.read(directory / name)
        if mutation == "source":
            value["commit"] = "different-source"
        elif mutation == "origin":
            value["modules"][next(iter(value["modules"]))] = str(tmp_path / "foreign.py")
        elif mutation == "unlocked-origin":
            value["sha256"][next(iter(value["sha256"]))] = "0" * 64
        elif mutation == "input-rate":
            value["profile"]["rx"]["ingress_rate_bytes_per_second"] *= 2
        elif mutation == "input-width":
            value["dims"]["hidden_size"] *= 2
        elif mutation == "packet-time":
            value["packets"][-1]["visible_at_ps"] += 1
        elif mutation == "causal-parent":
            value["critical_path"]["nodes"][4]["parents"] = ["missing"]
        elif mutation == "phase":
            value["critical_path"]["phases"][0]["operation_id"] = "foreign"
        elif mutation == "step-input":
            value["virtual_time_ps"] += 1
        elif mutation == "step-completion":
            value["result"]["completed_at_ps"] += 1
        elif mutation == "request-total":
            value["request_totals"]["tpot_ps"]["numerator"] += 1
        elif mutation == "request-identity":
            value["request_totals"]["request_id"] = "foreign"
        elif mutation == "request-history":
            value["peer-star"][1]["operation_latency_ps"] += 1
        else:
            value[0]["has_pending_physical_work"] = True
        reseal_fixture(directory, name, value)
    with pytest.raises((AssertionError, ValueError, TypeError, KeyError, FileNotFoundError)):
        admit(directory, identity, Evidence(tmp_path / "rejected"))


@pytest.mark.parametrize("raw", [b'{"a":1,"a":1}\n', b'{"a":NaN}\n', b'{"a":1}\r\n', b'{ "a": 1 }\n'])
def test_recorded_json_rejects_ambiguous_or_changed_bytes(tmp_path, raw):
    path = tmp_path / "record.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        run_study.read(path)


def test_first_process_receipts_precede_failed_exit_admission(tmp_path):
    evidence = Evidence(tmp_path / "raw")
    with pytest.raises(RuntimeError, match="exit 7"):
        run_study.capture_process(evidence, "failed", [sys.executable, "-c", "print('original'); raise SystemExit(7)"], tmp_path)
    assert (evidence.root / "failed.stdout").read_bytes().strip() == b"original"
    assert {"failed-command.json", "failed.stdout", "failed.stderr"} <= set(evidence.receipts)
    original = copy.deepcopy(evidence.receipts)
    (evidence.root / "failed.stdout").write_bytes(b"changed")
    with pytest.raises(ValueError, match="first receipt changed"):
        run_study.seal(evidence, evidence.root / "failed.stdout")
    assert evidence.receipts == original


def test_failed_process_keeps_original_failure_when_receipt_storage_also_fails(tmp_path, monkeypatch):
    evidence = Evidence(tmp_path / "raw")
    def failed_seal(*_):
        raise OSError("receipt storage failure")
    monkeypatch.setattr(run_study, "seal", failed_seal)
    with pytest.raises(RuntimeError, match="exit 7") as caught:
        run_study.capture_process(evidence, "failed", [sys.executable, "-c", "raise SystemExit(7)"], tmp_path)
    assert caught.value.reporting_failures[0]["message"] == "receipt storage failure"


def test_configure_inventory_is_sealed_after_build_before_execution(tmp_path):
    evidence = Evidence(tmp_path / "raw")
    build = evidence.root / "native-build"
    build.mkdir()
    cache = build / "CMakeCache.txt"
    cache.write_bytes(b"configure")
    run_study.capture_process(evidence, "configure", [sys.executable, "-c", "pass"], tmp_path, mutable_build=build)
    assert "native-build/CMakeCache.txt" not in evidence.receipts
    cache.write_bytes(b"built")
    run_study.capture_process(evidence, "build", [sys.executable, "-c", "pass"], tmp_path)
    assert evidence.receipts["native-build/CMakeCache.txt"] == run_study.receipt(cache)
    cache.write_bytes(b"changed-after-first-build")
    with pytest.raises(ValueError, match="first receipt changed"):
        run_study.capture_process(evidence, "execute", [sys.executable, "-c", "pass"], tmp_path)


def test_source_failure_stays_void_even_when_check_report_cannot_be_written(tmp_path, monkeypatch):
    def failed_source(_):
        raise ValueError("original source mismatch")
    monkeypatch.setattr(run_study, "source", failed_source)
    original_write = Evidence.write
    def failed_checks(self, name, value, **kwargs):
        if name == "checks.json":
            raise OSError("later report failure")
        return original_write(self, name, value, **kwargs)
    monkeypatch.setattr(Evidence, "write", failed_checks)
    output = tmp_path / "void"
    result = run_study.run(output, tmp_path)
    assert result["verdict"] == "VOID" and result["behavioral_score"] is None
    assert result["failure"]["message"] == "original source mismatch"
    assert result["reporting_failures"][0]["message"] == "later report failure"
    assert run_study.read(output / "summary.json") == result
