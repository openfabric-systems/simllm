"""Exercise complete captures, hostile semantic values and damaged admission."""

import json
import runpy
import subprocess
from argparse import Namespace
from copy import deepcopy
from unittest.mock import patch

import pytest

from examples.publication_snapshot_v1.common import packages, read, receipts
from examples.snapshot_dispatch_v1 import checks, run_study, worker
from examples.snapshot_dispatch_v1.semantics import observe
from simllm.core.value_snapshot import value_snapshot


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    output = tmp_path_factory.mktemp("snapshot-dispatch") / "after"
    output.mkdir()
    frozen = json.loads((checks.HERE / "expectations.json").read_bytes())
    old = json.loads((checks.HERE.parent / "publication_snapshot_v1/expectations.json").read_bytes())
    original = subprocess.check_output(["git", "show", frozen["before_commit"] + ":simllm/core/value_snapshot.py"], cwd=checks.ROOT)
    path = output.parent / "reference.py"
    path.write_bytes(original)
    reference = runpy.run_path(str(path))["value_snapshot"]
    worker.execute(Namespace(repository=checks.ROOT, output=output, arm="after", mode="main"))
    return output, frozen, old, reference, read(output / "worker.json")


def admit(captured, data=None):
    output, frozen, old, reference, original = captured
    evidence = checks.Evidence([])
    result = checks.admit(original if data is None else data, "after", checks.ROOT, output,
                          frozen, old, packages(checks.ROOT), reference, evidence)
    return result, evidence


def test_complete_values_and_all_mutation_fixtures_are_admitted(captured):
    vectors, evidence = admit(captured)
    assert len(vectors) == 10
    assert vectors["h16-q16"]["visit_isinstance"] == 51
    assert vectors["h16-q16"]["visit"] == 1553
    assert not evidence.oracles and not evidence.relations
    assert len(captured[-1]["mutations"]) == 14
    assert len(captured[-1]["helpers"]) == 6


@pytest.mark.parametrize("mutation,reason", [
    ("profile", "full-profile"), ("missing-semantic", "semantic-domain"),
    ("hidden-field", "semantic:dataclass-fields"), ("opaque-identity", "semantic:metaclass-spoof"),
    ("nonfinite-error", "semantic:nonfinite-nan"), ("source", "sources-before"),
    ("primitive-type", "complete-input"), ("publication", "all-visits"),
    ("record", "all-visits"), ("binding", "identity-slots"),
    ("job-clock", ":clock"), ("missing-mutation", "inventory"),
    ("mutated-fixture", "complete-mutation"), ("helper-return", "actual-helper-return"),
])
def test_damaged_complete_evidence_is_rejected(captured, mutation, reason):
    data = deepcopy(captured[-1])
    if mutation == "profile":
        next(row for row in data["primitives"][0]["profiles"] if row["function"][2] == "visit")["calls"] += 1
    elif mutation == "missing-semantic":
        data["semantics"].pop()
    elif mutation == "hidden-field":
        row = next(row for row in data["semantics"] if row["name"] == "dataclass-fields")
        row["calls"][1] = row["calls"][0]
    elif mutation == "opaque-identity":
        next(row for row in data["semantics"] if row["name"] == "metaclass-spoof")["calls"][0]["opaque"]["same_object"] = False
    elif mutation == "nonfinite-error":
        next(row for row in data["semantics"] if row["name"] == "nonfinite-nan")["calls"][0]["error"]["type"] = "TypeError"
    elif mutation == "source":
        data["sources_before"]["simllm/core/value_snapshot.py"] = "0" * 64
    elif mutation == "primitive-type":
        row = data["primitives"][1]
        for name in ("snapshot", "before", "after"):
            row[name][2][0][2][2] = ["builtins", "int", 1]
    elif mutation in ("publication", "record", "binding"):
        row = data["readers"][1]
        index = {"publication": 1, "record": 5, "binding": 2}[mutation]
        for name in ("snapshot", "before", "after"):
            row[name][2][index] = ["builtins", "int", 123]
    elif mutation == "job-clock":
        data["jobs"][0]["job"]["completed_at_ps"] += 1
    elif mutation == "missing-mutation":
        data["mutations"].pop()
    elif mutation == "mutated-fixture":
        data["mutations"][0]["pre_trigger"]["clock_at_ps"] += 1
    elif mutation == "helper-return":
        data["helpers"][0]["helper_return"] = None
    with pytest.raises(checks.GuardFailure, match=reason):
        admit(captured, data)


def test_declared_semantics_match_frozen_original(captured):
    _, frozen, _, reference, _ = captured
    for name in frozen["semantics"]:
        assert observe(name, value_snapshot) == observe(name, reference)


@pytest.mark.parametrize("mode", ["fraction-before", "fraction-during"])
def test_fraction_virtual_registration_stays_in_original_fallback(tmp_path, mode):
    output = tmp_path / mode
    output.mkdir()
    script = 'from examples.snapshot_dispatch_v1.semantics import observe; from simllm.core.value_snapshot import value_snapshot; import json; print(json.dumps(observe(' + repr(mode) + ', value_snapshot)))'
    result = subprocess.check_output([run_study.sys.executable, "-c", script], cwd=checks.ROOT)
    row = json.loads(result)
    assert row["calls"] == [{"error": {"type": "AttributeError", "message": "'list' object has no attribute 'numerator'"}}]
    assert row["trace"] == (["register-list"] if mode == "fraction-during" else [])


def test_initial_receipt_survives_process_startup_failure(tmp_path):
    frozen = json.loads((checks.HERE / "expectations.json").read_bytes())
    monitors, raw, root_receipts = {}, {}, {}
    path = tmp_path / "failed"
    with (patch.object(run_study.subprocess, "Popen", side_effect=OSError("declared spawn failure")),
          pytest.raises(OSError, match="declared spawn failure")):
        run_study.capture(checks.ROOT, path, "after", "main", frozen, monitors, raw, root_receipts)
    assert read(path / "process.json")["failure"] == "declared spawn failure"
    assert receipts(path) == raw["failed"] == read(tmp_path / "failed-raw-receipts.json")
    assert "failed-raw-receipts.json" in root_receipts


def test_guard_call_counts_cannot_hide_a_removed_check(captured):
    rows = deepcopy(captured[-1]["jobs"][0]["profiles"])
    original = checks.package_calls(rows, checks.ROOT)
    selected = next(row for row in rows if row["function"][2] == "_check_values")
    selected["calls"] -= 1
    assert checks.package_calls(rows, checks.ROOT) != original


def test_unchanged_input_visits_do_not_count_as_behavior(captured):
    vectors, evidence = admit(captured)
    assert vectors["h0-q1"]["visit_isinstance"] == 3
    assert not evidence.relations


@pytest.mark.parametrize("location", ["runtime_before", "runtime_after"])
def test_changed_interpreter_bytes_are_rejected(captured, location):
    data = deepcopy(captured[-1])
    data[location]["executable_sha256"] = "0" * 64
    with pytest.raises(checks.GuardFailure, match="runtime-"):
        admit(captured, data)


def test_coherent_binding_aliases_cannot_disappear_in_normalization(captured):
    data = deepcopy(captured[-1])
    for row in data["readers"]:
        alias = row["identities"][0]
        for entry, witness in zip(row["binding_identities"], row["binding_witnesses"], strict=True):
            entry[1:] = [alias, alias]
            witness.update(function_id=alias, code_id=alias)
        for field in ("before", "snapshot", "after"):
            for entry in row[field][2][7][2]:
                entry[2][1][2] = alias
                entry[2][2][2] = alias
    assert checks.normalized_reader(data["readers"][0]) == checks.normalized_reader(captured[-1]["readers"][0])
    with pytest.raises(checks.GuardFailure, match="binding-alias-graph"):
        admit(captured, data)


@pytest.mark.parametrize("field", ["code_sha256", "source_sha256", "qualname"])
def test_forged_binding_source_witness_is_rejected(captured, field):
    data = deepcopy(captured[-1])
    data["readers"][0]["binding_witnesses"][0][field] = "forged"
    with pytest.raises(checks.GuardFailure, match="source-witness"):
        admit(captured, data)


def test_same_basename_cannot_impersonate_standard_library_profile_origin(captured):
    rows = deepcopy(captured[-1]["primitives"][1]["profiles"])
    row = next(row for row in rows if row["function"][2] == "is_dataclass")
    row["function"][0] = str(checks.ROOT / "dataclasses.py")
    assert checks.counts(rows, checks.ROOT)["is_dataclass"] == 0


def test_binding_ids_must_stay_bound_to_the_loaded_module(captured):
    data = deepcopy(captured[-1])
    row = data["readers"][1]
    row["binding_identities"][0][1] += 1
    with pytest.raises(checks.GuardFailure, match="persistent-method-identities"):
        admit(captured, data)



def test_code_identity_ignores_reference_bookkeeping_and_keeps_values():
    from examples.snapshot_dispatch_v1.common import code_hash

    source = "def f(x):\n    return (x, -0.0, 2j, b'payload', 'text', (1, 2))\n"
    from types import CodeType

    code_a, code_b = (next(value for value in compile(source, filename, "exec").co_consts
                          if isinstance(value, CodeType)) for filename in ("left.py", "right.py"))
    retained = list(code_a.co_consts)
    assert retained
    assert code_hash(code_a, "source.py") == code_hash(code_b, "source.py")
    altered = code_b.replace(co_consts=tuple(0.0 if type(value) is float else value for value in code_b.co_consts))
    assert code_hash(code_a, "source.py") != code_hash(altered, "source.py")
