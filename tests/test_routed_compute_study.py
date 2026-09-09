"""Complete recorded model-output admission and corruption controls."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from examples.routed_compute_v1 import run_study
from examples.routed_compute_v1.inputs import campaign_config, step
from examples.routed_compute_v1.run_study import Evidence, check_step, primitive
from simllm.backends import DeviceRuntimeStepSink
from simllm.core import CoarseDeviceProfile, CoarseDeviceRuntime, VirtualClock
from simllm.core.execution_io import execution_graph_to_json, execution_result_to_json
from simllm.core.step_io import step_result_to_json


@pytest.fixture
def observation():
    cfg = campaign_config(8, "balanced", 5_000_000)
    sink = DeviceRuntimeStepSink(cfg, runtime=CoarseDeviceRuntime(CoarseDeviceProfile(
        rnic_rate_bps=8_000_000_000_000, nvlink_rate_bps=8_000_000_000_000,
    )))
    sink.bind_clock(VirtualClock())
    result = sink(step(cfg, 0), None)
    outcome = sink.outcomes[-1]
    return {
        "graph": execution_graph_to_json(outcome.graph),
        "execution": execution_result_to_json(outcome.execution_result),
        "runtime": primitive(outcome.runtime_report),
        "step_result": step_result_to_json(result),
    }


def test_actual_writer_output_is_admitted_completely(observation, tmp_path):
    evidence = Evidence(tmp_path / "evidence")
    evidence.write("actual.json", observation)
    rows = check_step(evidence, "fixture", 0, 8, "balanced", 5_000_000, observation)
    assert len(rows) == 8
    assert len(evidence.oracles) == 1
    assert all(row["passed"] for row in evidence.guards)


@pytest.mark.parametrize("mutation", [
    "lost-operation", "duplicate-operation", "duplicate-visit", "missing-visit",
    "changed-price", "changed-epoch", "changed-row-count", "changed-completion",
    "missing-request", "foreign-request", "unknown-result-field",
])
def test_corruptions_void_admission(observation, tmp_path, mutation):
    payload = copy.deepcopy(observation)
    experts = [
        row for row in payload["graph"]["operations"]
        if row["work"]["kind"] == "compute" and row["work"]["kernel"] == "moe_gate_up"
    ]
    if mutation == "lost-operation":
        payload["runtime"]["operations"].pop()
    elif mutation == "duplicate-operation":
        payload["runtime"]["operations"].append(copy.deepcopy(payload["runtime"]["operations"][0]))
    elif mutation == "duplicate-visit":
        visit = next(row for row in payload["runtime"]["visits"] if row["resource"]["kind"] == "gpu-work-queue")
        payload["runtime"]["visits"].append(copy.deepcopy(visit))
    elif mutation == "missing-visit":
        payload["runtime"]["visits"] = [row for row in payload["runtime"]["visits"] if row["operation_id"] != experts[0]["operation_id"]]
    elif mutation == "changed-price":
        experts[0]["work"]["nominal_duration_ps"] += 1
    elif mutation == "changed-epoch":
        experts[0]["placement_epoch"] = 1
    elif mutation == "changed-row-count":
        next(row for row in experts[0]["work"]["config"] if row[0] == "routed_rows")[1] += 1
    elif mutation == "changed-completion":
        payload["step_result"]["completed_at_ps"] += 1
    elif mutation == "missing-request":
        payload["step_result"]["request_metrics"].pop()
    elif mutation == "foreign-request":
        payload["step_result"]["request_metrics"][0]["request_id"] = "foreign"
    else:
        payload["step_result"]["unknown"] = 1
    with pytest.raises((AssertionError, ValueError, TypeError)):
        check_step(Evidence(tmp_path / "evidence"), "fixture", 0, 8, "balanced", 5_000_000, payload)


def test_work_generator_config_cannot_select_a_forged_minimum_envelope():
    cfg = campaign_config(8, "balanced", 5_000_000)
    with pytest.raises(ValueError, match="different GPU"):
        replace(cfg, gpu=replace(cfg.gpu, name="different-architecture"))


def test_first_receipt_is_persisted_before_any_admission(tmp_path):
    evidence = Evidence(tmp_path / "evidence")
    evidence.write("input.json", {"actual": 1})
    rows = [json.loads(row) for row in evidence.journal.read_text().splitlines()]
    assert rows == [{"path": "input.json", **evidence.receipts["input.json"]}]
    with pytest.raises(FileExistsError):
        evidence.write("input.json", {"changed": 2})
    assert len(evidence.journal.read_text().splitlines()) == 1


def test_reporting_failure_keeps_original_void_and_writes_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(run_study, "git", lambda *args: "dirty")
    write = Evidence.write

    def fail_checks(self, name, value, **kwargs):
        if name == "checks.json":
            raise OSError("checks storage failure")
        return write(self, name, value, **kwargs)

    monkeypatch.setattr(Evidence, "write", fail_checks)
    root = tmp_path / "void"
    assert run_study.run(root) == 1
    summary = json.loads((root / "summary.json").read_text())
    assert summary["verdict"] == "VOID" and summary["behavioral_score"] is None
    assert summary["failure"]["message"] == "clean-source"
    assert summary["reporting_failures"][0]["message"] == "checks storage failure"


def test_evidence_classes_have_separate_denominators(tmp_path):
    evidence = Evidence(tmp_path / "evidence")
    evidence.guard("ownership", True)
    evidence.oracle("service", True)
    evidence.relation("scaling", "larger-work", True)
    assert len(evidence.guards) == len(evidence.oracles) == len(evidence.relations) == 1
