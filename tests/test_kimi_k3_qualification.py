"""Adversarial checks for K3 qualification identity and dependency boundaries."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.kimi_k3_qualification_v1 import run_study as study
from examples.kimi_k3_qualification_v1.checks import (
    Evidence,
    causal_cell,
    causal_guards,
    corrupt_graph,
    discriminate,
    successor_ids,
    validate_evidence,
)
from examples.kimi_k3_qualification_v1.run_study import finalize, proof_guards
from examples.kimi_k3_structure_v1.run_study import ROOT, SUITE_ID, digest
from examples.kimi_k3_structure_v1.run_study import Evidence as RecordingEvidence
from simllm.backends.kimi_k3_lowerer import KimiK3Lowerer, KimiK3LowererConfig
from simllm.calibration.canonical import canonical_bytes
from simllm.calibration.extraction import case_records_from_suite
from simllm.compute.kimi_k3 import KimiK3Spec

FROZEN = json.loads((ROOT / "examples/kimi_k3_qualification_v1/expectations.json").read_bytes())


@pytest.fixture(scope="module")
def graphs():
    suite = json.loads((ROOT / "offline/calibration/suites" / SUITE_ID / "suite.json").read_bytes())
    spec = KimiK3Spec.from_obj(suite["reference_model"]["geometry"])
    records = case_records_from_suite(suite)
    return {
        (framework, phase): KimiK3Lowerer(KimiK3LowererConfig(spec, framework)).lower(
            next(r for r in records if r.scheduled[0].phase.value == phase)
        ) for framework in ("vllm", "sglang") for phase in ("prefill", "decode")
    }


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_existing_logical_graph_keeps_every_join_and_cache_boundary(graphs, framework, phase):
    evidence = RecordingEvidence()
    causal_guards(graphs[framework, phase], phase, evidence)
    assert evidence.valid
    assert len([r for r in evidence.guards if r["name"].startswith("join-parents:")]) == 92
    assert len([r for r in evidence.guards if r["name"].startswith("cache-ancestry:")]) == 24


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("shared", [1000, 3000, 10000])
def test_shared_tie_and_final_addition_have_separate_causal_costs(graphs, phase, shared):
    evidence = RecordingEvidence()
    a = causal_cell(graphs["vllm", phase], phase, shared, 1000, FROZEN, evidence)
    b = causal_cell(graphs["vllm", phase], phase, shared, 2000, FROZEN, evidence)
    assert evidence.valid
    assert b["completion_ps"] - a["completion_ps"] == 92000
    if shared == 10000:
        assert a["completion_ps"] == (3744000 if phase == "prefill" else 3792000)


@pytest.mark.parametrize("mode", FROZEN["corruption_controls"])
def test_same_visit_count_cannot_hide_wrong_dependencies(graphs, mode):
    phase = "decode" if mode == "decode-remove-cache-wait" else "prefill"
    graph = graphs["vllm", phase]
    corrupt, owning = corrupt_graph(graph, mode)
    check = RecordingEvidence()
    causal_guards(corrupt, phase, check)
    assert len(corrupt.operations) == len(graph.operations)
    assert any(r["name"] == owning and not r["passed"] for r in check.guards)
    discrimination = RecordingEvidence()
    discriminate(graph, phase, mode, discrimination)
    assert discrimination.valid


def test_frozen_manifest_enumerates_every_successor_stage_and_causal_oracle():
    expected = successor_ids(FROZEN)
    stages = FROZEN["successor_stage_ids"]
    assert len(stages) == len(set(stages)) == 78
    assert sum(kind == "oracle" for kind in expected.values()) == 24
    assert all(any(name.startswith("successor:" + stage + ":") for name in expected)
               for stage in stages)
    evidence = Evidence(expected, stages)
    assert evidence.completeness()


def valid_rows():
    return {"guards": [{"name": "g", "actual": 1, "expected": 1, "passed": True}],
            "oracles": [], "relations": [{"name": "r", "actual": 2, "expected": 2,
                                            "passed": True, "family": "family"}]}


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "wrong-kind", "false",
                                      "forged-pass", "unknown-family", "numeric-bool",
                                      "missing-actual", "guard-scored", "empty-stage", "coerced-actual",
                                      "extra-field"])
def test_evidence_admission_rejects_forged_success(mutation):
    rows = valid_rows()
    stages = ["stage"]
    if mutation == "missing":
        rows["guards"].clear()
    elif mutation == "duplicate":
        rows["guards"].append(copy.deepcopy(rows["guards"][0]))
    elif mutation == "extra":
        rows["guards"].append(dict(rows["guards"][0], name="extra"))
    elif mutation == "wrong-kind":
        rows["oracles"] = rows["guards"]
        rows["guards"] = []
    elif mutation == "false":
        rows["guards"][0]["passed"] = False
    elif mutation == "forged-pass":
        rows["guards"][0]["actual"] = 0
    elif mutation == "unknown-family":
        rows["relations"][0]["family"] = "invented"
    elif mutation == "numeric-bool":
        rows["guards"][0]["passed"] = 1
    elif mutation == "coerced-actual":
        rows["guards"][0]["actual"] = True
    elif mutation == "extra-field":
        rows["guards"][0]["unused"] = 0
    elif mutation == "missing-actual":
        del rows["guards"][0]["actual"]
    elif mutation == "guard-scored":
        rows["guards"][0]["family"] = "family"
    elif mutation == "empty-stage":
        stages = []
    assert validate_evidence(rows, {"g": "guard", "r": "relation"}, ["stage"], stages, {"r": "family"})


def test_decoder_prompt_identity_has_no_behavioral_score():
    name = "vllm:decode-prompt-independence"
    evidence = Evidence({name: "guard"}, [])
    evidence.check(name, 7, 7, kind="relation", family="synthetic-request-service-scaling")
    assert evidence.relations == []
    assert "family" not in evidence.guards[0]
    assert evidence.completeness() == []


def test_duplicate_completed_stage_cannot_pass_by_set_equality():
    evidence = Evidence({}, ["stage"])
    evidence.finish_stage("stage")
    evidence.finish_stage("stage")
    assert evidence.completeness()


def native_fixture(tmp_path):
    frozen = copy.deepcopy(FROZEN)
    frozen["successor_stage_ids"] = ["native:vllm:0"]
    package = tmp_path / "package"
    package.mkdir()
    source = package / "source.py"
    source.write_bytes(b"selected implementation")
    sha = digest(source.read_bytes())
    frozen["source_files"] = {"vllm-model.py": sha}
    frozen["required_import_origins"] = {"vllm": {"vllm.config.model": sha}}
    config = tmp_path / "config.json"
    config.write_bytes(b"configuration")
    frozen["model"]["config_sha256"] = digest(config.read_bytes())
    binding = {"id": "vllm", "architecture_binding": "fixture-binding",
               "text_implementation": "fixture-implementation"}
    suite = canonical_bytes({"frameworks": [binding]})
    steps = tmp_path / "steps.jsonl"
    steps.write_bytes(b"step records")
    declaration = frozen["frameworks"][0]
    framework = SimpleNamespace(framework_id="vllm", version=declaration["version"],
                                source_commit=declaration["source_commit"], source_tree=None,
                                to_obj=lambda: {"framework": "vllm"})
    record = SimpleNamespace(record_id=digest(b"canonical"), canonical=b"canonical")
    inventory = SimpleNamespace(framework=framework, record=record,
                                suite=SimpleNamespace(suite_sha256=digest(suite)),
                                model=SimpleNamespace(geometry=SimpleNamespace(to_obj=dict)))
    inputs = {"suite": digest(suite), "config": digest(config.read_bytes())}
    sources = [{"name": "vllm-model.py", "file": str(source), "sha256": sha}]
    projection = {"schema": "simllm-framework-text-config-projection-v1",
                  "framework": {key: declaration[key] for key in
                                ("id", "version", "source_commit", "source_tree")},
                  "configuration_seam": declaration["native_configuration"],
                  "architecture_binding": binding["architecture_binding"],
                  "text_implementation": binding["text_implementation"], "kimi_k3_stack": {}}
    proof = {"schema": "simllm-kimi-k3-native-source-proof-v1", "pid": 17, "interpreter": str(tmp_path / "python"),
             "framework": framework.to_obj(), "installed_version": declaration["version"],
             "projection": projection, "inputs_before": dict(inputs),
             "inputs_after": dict(inputs), "inventory_sha256": record.record_id,
             "steps_sha256": digest(steps.read_bytes()), "sources_before": copy.deepcopy(sources),
             "sources_after": copy.deepcopy(sources), "package": str(package),
             "origins": [{"module": "vllm.config.model", "file": str(source), "sha256": sha}],
             "gpu_initialized": False}
    capture = {"framework": "vllm", "repetition": 0, "proof": proof, "directory": tmp_path,
               "inventory": inventory, "raw": b"canonical", "pid": 17,
               "python": tmp_path / "python", "receipt": {"pid": 17,
               "record_sha256": record.record_id, "steps_sha256": proof["steps_sha256"]}}
    (tmp_path / "objects").mkdir()
    (tmp_path / "objects" / f"{record.record_id}.json").write_bytes(record.canonical)
    raw_proof = canonical_bytes(proof)
    (tmp_path / "native-proof.json").write_bytes(raw_proof)
    capture["receipt"]["proof_sha256"] = digest(raw_proof)
    return frozen, suite, config, capture


@pytest.mark.parametrize("mutation", [None, "pid", "interpreter", "source", "missing-origin",
                                      "origin-hash", "duplicate-source", "config", "record",
                                      "gpu-initialized", "disk-proof", "disk-inventory"])
def test_native_proof_joins_actual_files_and_extraction_identity(tmp_path, mutation):
    frozen, suite, config, capture = native_fixture(tmp_path)
    proof = capture["proof"]
    if mutation == "pid":
        proof["pid"] += 1
    elif mutation == "interpreter":
        proof["interpreter"] += "-other"
    elif mutation == "source":
        Path(proof["sources_after"][0]["file"]).write_bytes(b"different bytes")
    elif mutation == "missing-origin":
        proof["origins"] = []
    elif mutation == "origin-hash":
        proof["origins"][0]["sha256"] = "0" * 64
    elif mutation == "duplicate-source":
        proof["sources_after"].append(copy.deepcopy(proof["sources_after"][0]))
    elif mutation == "config":
        config.write_bytes(b"changed configuration")
    elif mutation == "record":
        proof["inventory_sha256"] = "0" * 64
    elif mutation == "gpu-initialized":
        proof["gpu_initialized"] = True
    elif mutation == "disk-proof":
        (tmp_path / "native-proof.json").write_bytes(b"changed retained proof")
    elif mutation == "disk-inventory":
        path = tmp_path / "objects" / f"{capture['inventory'].record.record_id}.json"
        path.write_bytes(b"changed retained inventory")
    expected = {n: k for n, k in successor_ids(frozen).items()
                if n.startswith("successor:native:vllm:0:")}
    evidence = Evidence(expected, ["native:vllm:0"])
    proof_guards(capture, frozen, suite, config, evidence)
    assert evidence.completeness() == []
    assert evidence.valid is (mutation is None)


def test_known_family_reassignment_and_family_collapse_are_fatal():
    rows = valid_rows()
    rows["relations"].append(dict(rows["relations"][0], name="s", family="second"))
    expected = {"g": "guard", "r": "relation", "s": "relation"}
    families = {"r": "family", "s": "second"}
    assert validate_evidence(rows, expected, [], [], families) == []
    rows["relations"][1]["family"] = "family"
    assert validate_evidence(rows, expected, [], [], families)
    rows["relations"][0]["family"] = "second"
    assert validate_evidence(rows, expected, [], [], families)


@pytest.mark.parametrize("mutation", ["missing-class", "non-string-family", "bad-stages", "nan"])
def test_malformed_evidence_returns_fatal_findings(mutation):
    rows, finished = valid_rows(), []
    if mutation == "missing-class":
        rows["relations"] = None
    elif mutation == "non-string-family":
        rows["relations"][0]["family"] = []
    elif mutation == "bad-stages":
        finished = [{}]
    elif mutation == "nan":
        rows["guards"][0]["actual"] = float("nan")
    assert validate_evidence(rows, {"g": "guard", "r": "relation"}, [], finished, {"r": "family"})


@pytest.mark.parametrize("field", ["schema", "framework", "configuration_seam",
                                   "architecture_binding", "text_implementation"])
def test_coherent_source_receipt_cannot_hide_wrong_projection(tmp_path, field):
    frozen, suite, config, capture = native_fixture(tmp_path)
    proof = capture["proof"]
    proof["projection"][field] = "wrong native claim"
    raw = canonical_bytes(proof)
    (tmp_path / "native-proof.json").write_bytes(raw)
    capture["receipt"]["proof_sha256"] = digest(raw)
    expected = successor_ids(frozen)
    evidence = Evidence(expected, ["native:vllm:0"])
    proof_guards(capture, frozen, suite, config, evidence)
    assert evidence.completeness() == []
    assert not evidence.valid
    assert any(r["name"].endswith(":projection-geometry") and not r["passed"]
               for r in evidence.guards)
    assert next(r["passed"] for r in evidence.guards if r["name"].endswith(":source-stable"))


def test_noncanonical_evidence_is_retained_with_a_void_summary(tmp_path):
    evidence = Evidence({"g": "guard"}, [])
    evidence.check("g", float("nan"), 1)
    frozen = {"coverage": {"behavioral_instances": 0}, "successor_stage_ids": []}
    summary = finalize(tmp_path, frozen, evidence, {}, [], {})
    retained = json.loads((tmp_path / "summary.json").read_bytes())
    assert summary == retained
    assert retained["verdict"] == "VOID"
    assert retained["behavioral_score"] is None
    assert "nan" in (tmp_path / "guards.noncanonical.txt").read_text()
    assert retained["noncanonical_evidence"]["guards.noncanonical.txt"] == digest(
        (tmp_path / "guards.noncanonical.txt").read_bytes()
    )


@pytest.mark.parametrize("error", [StopIteration, IndexError])
def test_execution_contract_exceptions_leave_void_evidence(tmp_path, monkeypatch, error):
    monkeypatch.setattr(study, "inherited_ids", lambda *_: {})
    monkeypatch.setattr(study, "inherited_families", lambda *_: {
        str(i): name for i, name in enumerate(FROZEN["coverage"]["behavioral_families"])
    })
    def fail(*_):
        raise error("broken result contract")
    monkeypatch.setattr(study, "execute", fail)
    output = tmp_path / "evidence"
    arguments = [item for key in ("vllm-python", "sglang-python", "checkpoint-root", "source-root",
                                 "legacy-checkpoints", "historical-root")
                 for item in ("--" + key, str(tmp_path))]
    assert study.main([*arguments, "--output-root", str(output)]) == 2
    result = json.loads((output / "summary.json").read_bytes())
    assert result["verdict"] == "VOID" and result["behavioral_score"] is None
    assert error.__name__ in (output / "exception.txt").read_text()


@pytest.mark.parametrize("framework,digest", [
    ("vllm", "825cb7628cc6635ae77c9d200992189f166c55621ee7ec00d2658d5e21f59bb8"),
    ("sglang", "7668beca110dc9418636996057be9349fb79e0b481f293347375dd069f809f3a"),
])
def test_qualified_inventory_bytes_preserve_unknown_physical_demand(framework, digest):
    from simllm.calibration.model_inventory import ModelKernelInventory

    raw = (ROOT / "offline/calibration/model-inventories" / f"{digest}.json").read_bytes()
    payload = json.loads(raw)
    inventory = ModelKernelInventory.from_obj(payload)
    assert canonical_bytes(inventory.to_obj()) == raw
    assert study.digest(raw) == digest
    assert inventory.framework.framework_id == framework
    assert len(inventory.cases) == 12
    suite_raw = (ROOT / "offline/calibration/suites" / SUITE_ID / "suite.json").read_bytes()
    assert inventory.suite.suite_sha256 == study.digest(suite_raw)
    assert json.loads(suite_raw)["state"] == "authored-inputs-only"
    for case in payload["cases"]:
        projections = case["kernel_projections"]
        assert sum(row["logical_launch_count"] for row in projections) == (
            3244 if case["phase"] == "prefill" else 3268
        )
        for row in projections:
            assert row["scope"] == "logical-operator"
            if row["logical_launch_count"]:
                assert row["aggregate_hbm_bytes"] is None
            else:
                assert row["aggregate_hbm_bytes"] == 0
    for field in ("code_object_hashes", "observed_launches"):
        assert payload["implementation_identity"][field] == {
            "state": "absent-by-design", "value": None,
        }
