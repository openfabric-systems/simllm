"""Exercise the real capture boundary and independently reject damaged evidence."""

import json
import os
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from examples.publication_snapshot_v1 import checks, common, run_study, worker
from simllm.core.step import RequestPhase, ScheduledRequest, StepRecord
from simllm.core.value_snapshot import value_snapshot


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    root = tmp_path_factory.mktemp("publication-snapshot")
    output = root / "after"
    output.mkdir()
    frozen = json.loads((common.HERE / "expectations.json").read_bytes())
    args = Namespace(repository=common.ROOT, output=output, expectations=common.HERE / "expectations.json", arm="after")
    # The actual standalone process defines this class in __main__.
    with patch.object(worker.SyntheticRow, "__module__", "__main__"):
        worker.execute(args)
    return output, frozen, common.read(output / "worker.json")


def admit(captured, data=None):
    output, frozen, original = captured
    evidence = checks.Evidence([])
    values = original if data is None else data
    return checks.admit(values, "after", common.ROOT, output, frozen, common.packages(common.ROOT),
                        {"pid": os.getpid()}, evidence), evidence


def test_actual_reader_counts_jobs_and_integrity_controls_are_admitted(captured):
    counts, evidence = admit(captured)
    _, frozen, data = captured
    assert len(counts) == len(frozen["cases"]) == 10
    assert len(evidence.oracles) == 10
    assert not evidence.relations
    assert len(data["mutations"]) == 14 and len(data["helpers"]) == 6


@pytest.mark.parametrize("mutation,reason", [
    ("frozen-record", "frozen-record"), ("state", ":state"),
    ("outer-count", "outer-count"), ("profile", "full-profile"),
    ("mutation-delta", "declared-delta"), ("mutation-result", "rejection"),
    ("missing-mutation", "inventory"), ("job-clock", ":clock"),
    ("history-prefix", "history-prefix"), ("source", "source-before"),
    ("wrapper", "wrappers"), ("source-binding", "bindings"),
    ("null-mutation-history", "initial-publications"), ("changed-fixture", "complete-fixture"),
    ("coherent-post-trigger", "complete-post-trigger"), ("helper-return", "actual-helper-return"),
    ("untracked-origin", "tracked-origin"), ("shadow-study", "study-origin"),
])
def test_changed_evidence_cannot_keep_its_verdict(captured, mutation, reason):
    _, frozen, original = captured
    data = deepcopy(original)
    first = data["cases"][0]
    if mutation == "frozen-record":
        changed = StepRecord(0, 0, [ScheduledRequest("request-0", RequestPhase.DECODE, 2, context_length=16)], num_sampled=1)
        first["before"][0] = common.pack(value_snapshot(changed))
        first["after"] = deepcopy(first["before"])
    elif mutation == "state":
        first["state_snapshot"] = common.pack(value_snapshot(None))
    elif mutation == "outer-count":
        first["expected"]["outer_input_visits"] += 1
    elif mutation == "profile":
        next(row for row in first["profiles"] if row["function"][2] == "visit")["calls"] += 1
    elif mutation == "mutation-delta":
        data["mutations"][0]["delta"]["prior_host_changed"] = False
    elif mutation == "mutation-result":
        data["mutations"][0]["exception"]["message"] = "unrelated failure"
    elif mutation == "missing-mutation":
        data["mutations"].pop()
    elif mutation == "job-clock":
        data["jobs"][0]["completed_at_ps"] += 1
    elif mutation == "history-prefix":
        job = next(row for row in data["jobs"] if row["id"] == "l1-h1")
        job["before"] = common.pack(value_snapshot({name: [] for name in frozen["publication_names"]}))
        job["early"]["publications"] = deepcopy(job["before"])
    elif mutation == "source":
        data["sources_before"]["simllm/backends/step_sink.py"] = "0" * 64
    elif mutation == "wrapper":
        first["state_wrapper_exact"] = False
    elif mutation == "null-mutation-history":
        data["mutations"][0]["initial_publications"] = data["mutations"][0]["restored_publications"] = None
    elif mutation == "changed-fixture":
        data["mutations"][0]["fixture"]["clock_at_ps"] += 1
    elif mutation == "coherent-post-trigger":
        data["mutations"][0]["post_trigger"]["publications"] = data["mutations"][0]["fixture"]["publications"]
    elif mutation == "helper-return":
        data["helpers"][0]["helper_return"] = None
    elif mutation == "untracked-origin":
        data["origins"]["simllm.untracked"] = str(common.ROOT / "simllm/untracked.py")
    elif mutation == "shadow-study":
        data["study_origins"]["examples.publication_snapshot_v1.common"] = str(common.ROOT.parent / "shadow/examples/publication_snapshot_v1/common.py")
    else:
        first["bindings"].remove("_deferred_state_values")
    with pytest.raises(checks.GuardFailure, match=reason):
        admit(captured, data)


def test_coherent_null_publication_rows_cannot_pass_as_a_completed_job(captured):
    _, frozen, original = captured
    data = deepcopy(original)
    job = data["jobs"][0]
    for key in ("ordinary_publications", "deferred_publications"):
        for name, rows in job[key][2]:
            if name[2] in frozen["synthetic_populated_names"]:
                rows[2][:] = [common.pack(value_snapshot(None)) for _ in rows[2]]
    with pytest.raises(ValueError, match="typed row class disagrees"):
        admit(captured, data)


@pytest.mark.parametrize("bad,reason", [
    (["builtins", "float", "nan"], "canonical finite hex"),
    (["builtins", "float", "1.0"], "canonical finite hex"),
    (["builtins", "int", True], "body disagrees"),
    (["builtins", "int", 1, "ignored"], "arity"),
])
def test_coherent_invalid_scalar_in_full_job_is_rejected(captured, bad, reason):
    _, _, original = captured
    data = deepcopy(original)
    job = data["jobs"][0]
    for key in ("ordinary_publications", "deferred_publications"):
        outcome = checks.entry(job[key], "outcomes")[2][0]
        checks.change_member(outcome, "compute_estimate_ps", deepcopy(bad))
    outcome = checks.member(job["prepared_simulation"], "outcome")
    checks.change_member(outcome, "compute_estimate_ps", deepcopy(bad))
    with pytest.raises(ValueError, match=reason):
        admit(captured, data)


def test_complete_mutation_fixture_is_retained_before_an_unexpected_trigger_failure(tmp_path, monkeypatch):
    frozen = json.loads((common.HERE / "expectations.json").read_bytes())

    def fail(self):
        raise OSError("unexpected retirement failure")

    monkeypatch.setattr(worker.EngineStepRuntime, "complete_due", fail)
    with pytest.raises(OSError, match="unexpected retirement"):
        worker.capture_mutation("prior-row", frozen, tmp_path / "work", tmp_path)
    capture = common.read(tmp_path / "prior-row-before.json")
    assert set(capture) == {"name", "delta", "fixture", "pre_trigger", "helper_return"}
    assert capture["fixture"]["publications"] != capture["pre_trigger"]["publications"]
    assert not (tmp_path / "prior-row.json").exists()


@pytest.mark.parametrize("failure_stage", ["job", "mutation"])
def test_finished_capture_survives_a_later_capture_failure(tmp_path, monkeypatch, failure_stage):
    args = Namespace(repository=common.ROOT, output=tmp_path, expectations=common.HERE / "expectations.json", arm="after")
    name = "capture_" + failure_stage
    original = getattr(worker, name)
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("later capture failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(worker, name, fail_second)
    with pytest.raises(OSError, match="later capture failed"):
        worker.execute(args)
    file = "l1-h0-job.json" if failure_stage == "job" else "prior-row.json"
    assert common.read(tmp_path / file)


@pytest.mark.parametrize("body", ['{"x":1,"x":1}\n', '{"x":NaN}\n', '{ "x":1}\n'])
def test_noncanonical_raw_json_is_rejected(tmp_path, body):
    path = tmp_path / "row.json"
    path.write_text(body)
    with pytest.raises(ValueError):
        common.read(path)


def test_snapshot_wire_preserves_type_and_unordered_members():
    values = [{"x": [1, True, 1.0]}, (frozenset((1, 2)), b"\x00\xff"), Path("model")]
    for value in values:
        snapshot = value_snapshot(value)
        assert common.unpack(common.pack(snapshot)) == snapshot
    assert value_snapshot([1]) != value_snapshot((1,))
    assert common.original_nodes(value_snapshot(values)) == common.nodes(values)


def test_failed_process_still_gets_its_first_raw_receipt(tmp_path, monkeypatch):
    args = Namespace(output=tmp_path)
    monitors, raw, roots = {}, {}, {}

    def fail(arm, root, args, frozen, monitors):
        path = args.output / arm
        path.mkdir()
        common.write(path / "process.json", {"exit_code": 1})
        (path / "process.log").write_text("retained failure\n")
        raise checks.GuardFailure("original process failure")

    monkeypatch.setattr(run_study, "launch", fail)
    with pytest.raises(checks.GuardFailure, match="original process failure"):
        run_study.capture("after", common.ROOT, args, {}, monitors, raw, roots)
    assert raw["after"] == common.receipts(tmp_path / "after")
    assert common.read(tmp_path / "after-raw-receipts.json") == raw["after"]
    assert roots["after-raw-receipts.json"]["sha256"] == common.sha(tmp_path / "after-raw-receipts.json")
